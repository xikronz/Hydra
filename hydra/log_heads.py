#!/usr/bin/env python3
"""Analyze Hydra head acceptance during actual speculative decoding inference.

Runs the model in real inference mode (same as evals) and tracks:
  - Accept length at each generation step
  - Which Hydra heads were accepted (contributed to the accepted prefix)
  - Distribution of accepted heads across the entire response
  - Effective speedup (tokens generated per decoding step)

Usage:
  python -m hydra.log_heads \
    --model ankner/hydra-vicuna-7b-v1.3 \
    --data data/sharegpt/raw/val.json \
    --sample_idx 0 \
    --output logs/hydra_head_analysis.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hydra.model.hydra_model import HydraModel
from hydra.model.utils import (
    generate_hydra_buffers,
    initialize_hydra,
    reset_hydra_mode,
    tree_decoding,
    evaluate_posterior,
    update_inference_inputs,
)
from hydra.model.kv_cache import initialize_past_key_values
from hydra.model.hydra_choices import mc_sim_7b_63


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _next_token_entropy(logits: torch.Tensor) -> float:
    """Entropy (in nats) of the LM head's next-token distribution at the
    last position of `logits` (which has shape [1, T, vocab]).

    Uses log_softmax in fp32 for numerical stability.
    """
    last = logits[0, -1].float()
    log_probs = torch.log_softmax(last, dim=-1)
    probs = log_probs.exp()
    return float(-(probs * log_probs).sum().item())


class BiasCorrectedEMA:
    """Exponential moving average with Adam-style bias correction.

    For window size `W`, we set ``beta = 1 - 1/W`` so the average has an
    effective half-life of roughly ``W`` steps. The raw accumulator

        m_t = beta * m_{t-1} + (1 - beta) * x_t,   m_0 = 0

    is biased toward zero in early steps (since we initialise at 0). Adam
    fixes this with the correction factor ``1 - beta**t``:

        m_hat_t = m_t / (1 - beta**t)

    which is the convex combination of the seen samples weighted by
    ``(1-beta) * beta**(t-i)`` and renormalised so the weights sum to 1.

    This is identical in spirit to a simple sliding-window mean over the
    most recent W samples, but smooth and stateless to update.
    """

    __slots__ = ("window", "beta", "m", "t")

    def __init__(self, window: int):
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.window = int(window)
        self.beta = 1.0 - 1.0 / float(window)
        self.m = 0.0
        self.t = 0

    def reset(self) -> None:
        self.m = 0.0
        self.t = 0

    def update(self, x: float) -> float:
        self.t += 1
        self.m = self.beta * self.m + (1.0 - self.beta) * x
        denom = 1.0 - (self.beta ** self.t)
        if denom <= 0.0:
            return float(x)
        return self.m / denom


def _normalize_turn(turn: dict) -> tuple[str, str]:
    """Map a chat turn from common schemas to (role, content).

    Supports:
      - ShareGPT:  {"from": "human"|"gpt"|..., "value": "..."}
      - OpenAI/HumanEval: {"role": "user"|"assistant"|..., "content": "..."}

    Returns role normalized to one of {"user", "assistant", "system"}.
    """
    if "from" in turn and "value" in turn:
        raw = str(turn["from"]).lower()
        content = turn["value"]
    elif "role" in turn and "content" in turn:
        raw = str(turn["role"]).lower()
        content = turn["content"]
    else:
        raise ValueError(f"Unrecognized turn schema (keys={list(turn.keys())})")
    if raw in ("human", "user"):
        role = "user"
    elif raw in ("gpt", "assistant", "bot", "model"):
        role = "assistant"
    elif raw in ("system",):
        role = "system"
    else:
        role = raw
    return role, content


def _load_conversations(path: str) -> list:
    """Load conversations from either a JSON array file or a JSONL file."""
    with open(path) as f:
        first_nonspace = ""
        for ch in f.read(64):
            if not ch.isspace():
                first_nonspace = ch
                break
        f.seek(0)
        if first_nonspace == "[":
            data = json.load(f)
            return data if isinstance(data, list) else [data]
        out = []
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out


@torch.inference_mode()
def run_hydra_generation_with_logging(
    model,
    tokenizer,
    conversation: list[dict],
    prompt_text: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    posterior_threshold: float = 0.09,
    posterior_alpha: float = 0.3,
    hydra_choices=mc_sim_7b_63,
    smoothing_windows: tuple[int, ...] = (4, 16, 32),
    sample_idx: int | None = None,
) -> tuple[list[dict], dict]:
    """Run Hydra speculative decoding and log head acceptance at each step.
    
    This uses the exact same inference flow as hydra_generate() but logs
    detailed per-step statistics about which heads were accepted.
    
    Returns:
        rows: List of per-step records with accept_length and head acceptance
        summary: Overall statistics for the generation
    """
    if prompt_text is not None:
        prompt = prompt_text
    else:
        prompt_parts = []
        for turn in conversation:
            role, content = _normalize_turn(turn)
            if role == "system":
                prompt_parts.append(f"SYSTEM: {content}")
            elif role == "user":
                prompt_parts.append(f"USER: {content}")
            elif role == "assistant":
                break

        prompt = "\n".join(prompt_parts) + "\nASSISTANT:"
    
    enc = tokenizer(prompt, return_tensors="pt")
    dev = _model_device(model)
    input_ids = enc["input_ids"].to(dev)
    
    print(f"[info] prompt length: {input_ids.shape[1]}, generating up to {max_new_tokens} tokens")
    
    hydra_buffers = generate_hydra_buffers(hydra_choices, device=dev)
    model.hydra_buffers = hydra_buffers
    model.hydra_choices = hydra_choices

    # retrieve_indices is built from this sort order (see generate_hydra_buffers),
    # so positions inside retrieve_indices map back via sorted_hydra_choices, not
    # the user-supplied hydra_choices order.
    sorted_hydra_choices = sorted(hydra_choices, key=lambda x: (len(x), x))
    
    (
        past_key_values,
        past_key_values_data,
        current_length_data,
    ) = initialize_past_key_values(model.base_model, model.hydra_head_arch)
    model.past_key_values = past_key_values
    model.past_key_values_data = past_key_values_data
    model.current_length_data = current_length_data
    
    input_len = input_ids.shape[1]
    num_heads = model.hydra
    
    max_position_embeddings = model.config.max_position_embeddings
    tree_size = len(hydra_choices) + 1
    
    reset_hydra_mode(model)
    hidden_states, logits = initialize_hydra(
        input_ids, model, hydra_buffers["hydra_attn_mask"], past_key_values, hydra_buffers["proposal_cross_attn_masks"]
    )
    
    rows = []
    step_accept_lengths = []
    head_acceptance_counts = defaultdict(int)
    accepted_topk_index_counts = defaultdict(lambda: defaultdict(int))
    step_entropies: list[float] = []
    # One bias-corrected EMA per requested window size. State is per-call,
    # so each conversation starts fresh (no leakage across samples).
    smoothers = {int(w): BiasCorrectedEMA(int(w)) for w in smoothing_windows}
    total_generated = 0

    for idx in range(max_new_tokens):
        current_len = input_ids.shape[1]
        if current_len + tree_size > max_position_embeddings:
            print(f"[info] Stopping: would exceed max_position_embeddings ({current_len} + {tree_size} > {max_position_embeddings})")
            break

        # Free entropy: the LM head's distribution at logits[0, -1] is the
        # distribution from which the FIRST token committed this step is
        # drawn. Across steps, this naturally advances by (accept_length + 1)
        # tokens because update_inference_inputs trims `logits` so that
        # logits[0, -1] becomes the position right after the last commit.
        block_start_entropy = _next_token_entropy(logits)
        smoothed_entropies = {w: ema.update(block_start_entropy) for w, ema in smoothers.items()}
        step_entropies.append(block_start_entropy)

        to_pass_input_ids = input_ids if idx == 0 else None
        
        candidates, tree_candidates = model.hydra_head.proposal(
            logits, hidden_states, hydra_buffers, past_key_values, to_pass_input_ids
        )
        
        hidden_states, logits = tree_decoding(
            model,
            tree_candidates,
            past_key_values,
            hydra_buffers["hydra_position_ids"],
            input_ids,
            hydra_buffers["retrieve_indices"],
        )
        
        best_candidate, accept_length = evaluate_posterior(
            logits, candidates, temperature, posterior_threshold, posterior_alpha, hydra_buffers["max_accepts"]
        )
        
        accept_len_int = int(accept_length.item())
        step_accept_lengths.append(accept_len_int)

        accepted_tokens = candidates[best_candidate, :accept_len_int + 1].tolist()
        accepted_text = tokenizer.decode(accepted_tokens, skip_special_tokens=False)

        # ----- Recover the topk-index coordinate path the verifier walked. -----
        # retrieve_indices[best_candidate] is the root-to-leaf row in the tree.
        # Layout: column 0 is always 0 (root); columns 1..L hold 1-indexed
        # positions into sorted_hydra_choices (or -1 for padding).
        winning_path_idx = int(best_candidate.item())
        path_positions = (
            hydra_buffers["retrieve_indices"][winning_path_idx].tolist()
        )
        # Drop the root (col 0) and any padding (-1).
        real_positions = [p for p in path_positions[1:] if p > 0]
        if real_positions:
            full_winning_path = tuple(
                sorted_hydra_choices[real_positions[-1] - 1]
            )
        else:
            full_winning_path = ()
        accept_depth = min(accept_len_int, len(full_winning_path))
        accepted_topk_indices = tuple(full_winning_path[:accept_depth])

        # Aggregate stats: at each accepted depth, how often was the top-1 vs
        # not the top-1 chosen? (k=0 means top-1 child of parent.)
        for d, k in enumerate(accepted_topk_indices):
            accepted_topk_index_counts[d][int(k)] += 1

        row = {}
        if sample_idx is not None:
            row["sample_idx"] = int(sample_idx)
        row.update({
            "step": idx,
            "position": int(input_ids.shape[1]),
            "accept_length": accept_len_int,
            "tokens_this_step": accept_len_int + 1,
            "accept_depth": accept_depth,
            "accepted_topk_indices": repr(accepted_topk_indices),
            "full_winning_path": repr(full_winning_path),
            "block_start_entropy": block_start_entropy,
        })
        for w in sorted(smoothed_entropies.keys()):
            row[f"entropy_smoothed_w{w}"] = smoothed_entropies[w]
        row["accepted_text"] = accepted_text.replace("\n", "\\n")

        for head_idx in range(num_heads):
            if head_idx < accept_len_int:
                head_acceptance_counts[head_idx] += 1

        rows.append(row)
        
        input_ids, logits, hidden_states, total_generated = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            hydra_buffers["retrieve_indices"],
            logits,
            hidden_states,
            total_generated,
            past_key_values_data,
            current_length_data,
            model.hydra_head_arch,
        )
        
        if tokenizer.eos_token_id in input_ids[0, input_len:]:
            break
    
    generated_text = tokenizer.decode(
        input_ids[0, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    
    accept_dist = {}
    if step_accept_lengths:
        unique, counts = np.unique(step_accept_lengths, return_counts=True)
        accept_dist = {int(k): int(v) for k, v in zip(unique, counts)}
    
    accepted_topk_index_distribution = {
        d: dict(counts) for d, counts in accepted_topk_index_counts.items()
    }

    if step_entropies:
        ent_arr = np.asarray(step_entropies, dtype=np.float64)
        entropy_stats = {
            "mean": float(ent_arr.mean()),
            "std": float(ent_arr.std()),
            "min": float(ent_arr.min()),
            "p25": float(np.quantile(ent_arr, 0.25)),
            "p50": float(np.quantile(ent_arr, 0.50)),
            "p75": float(np.quantile(ent_arr, 0.75)),
            "max": float(ent_arr.max()),
        }
    else:
        entropy_stats = {}

    summary = {
        "num_steps": len(step_accept_lengths),
        "total_tokens": total_generated,
        "mean_accept_length": float(np.mean(step_accept_lengths)) if step_accept_lengths else 0.0,
        "mean_tokens_per_step": total_generated / max(len(step_accept_lengths), 1),
        "accept_length_distribution": accept_dist,
        "head_acceptance_counts": dict(head_acceptance_counts),
        "accepted_topk_index_distribution": accepted_topk_index_distribution,
        "block_start_entropy_stats": entropy_stats,
        "generated_text": generated_text,
    }

    return rows, summary


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    all_keys = list(rows[0].keys())
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        w.writerows(rows)


class StreamingCSVWriter:
    """Append-only CSV writer with a fixed header.

    The header is materialised on the first call to ``append_rows``. We open
    in line-buffered mode so that a SLURM kill/preempt mid-run still leaves
    a valid CSV up to the last completed sample.
    """

    def __init__(self, path: str, fieldnames: list[str] | None = None):
        self.path = path
        self.fieldnames = list(fieldnames) if fieldnames else None
        self._file = None
        self._writer = None
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def _open_if_needed(self, fieldnames: list[str]) -> None:
        if self._file is not None:
            return
        if self.fieldnames is None:
            self.fieldnames = list(fieldnames)
        # buffering=1 = line buffered (text mode); fine for CSV.
        self._file = open(self.path, "w", newline="", buffering=1)
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        self._writer.writeheader()

    def append_rows(self, rows: list[dict]) -> None:
        if not rows:
            return
        self._open_if_needed(list(rows[0].keys()))
        for row in rows:
            # Drop any unknown keys quietly; missing keys become blank strings.
            self._writer.writerow({k: row.get(k, "") for k in self.fieldnames})
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


def print_summary(rows: list[dict], summary: dict, num_heads: int) -> None:
    print("\n" + "=" * 70)
    print("HYDRA HEAD ACCEPTANCE ANALYSIS (Actual Inference Mode)")
    print("=" * 70)
    
    print(f"\nGeneration Statistics:")
    print(f"  Total decoding steps: {summary['num_steps']}")
    print(f"  Total tokens generated: {summary['total_tokens']}")
    print(f"  Mean accept length: {summary['mean_accept_length']:.3f}")
    print(f"  Mean tokens/step (speedup): {summary['mean_tokens_per_step']:.2f}x")
    
    print(f"\nDistribution of Accept Lengths:")
    print(f"  (accept_length=0 means only base model token accepted)")
    print(f"  (accept_length=N means base + N hydra head tokens accepted)")
    dist = summary['accept_length_distribution']
    total_steps = summary['num_steps']
    for length in sorted(dist.keys()):
        count = dist[length]
        pct = 100.0 * count / total_steps if total_steps > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"  accept_length={length}: {count:4d} ({pct:5.1f}%) {bar}")
    
    print(f"\nPer-Head Acceptance Rate:")
    print(f"  (How often each head's prediction was accepted)")
    head_counts = summary['head_acceptance_counts']
    for head_idx in range(num_heads):
        count = head_counts.get(head_idx, 0)
        rate = count / total_steps if total_steps > 0 else 0
        bar = "#" * int(rate * 40)
        print(f"  Head {head_idx}: {count:4d}/{total_steps} ({rate:.3f}) {bar}")

    ent_stats = summary.get("block_start_entropy_stats", {})
    if ent_stats:
        print(f"\nBlock-start entropy (LM head, in nats):")
        print(f"  (Entropy of base model's distribution at the FIRST token committed each step.")
        print(f"   Lower => the base model is confident about the start of the next block.)")
        print(f"  mean: {ent_stats['mean']:.3f}  std: {ent_stats['std']:.3f}")
        print(f"  min/p25/p50/p75/max: "
              f"{ent_stats['min']:.3f} / {ent_stats['p25']:.3f} / "
              f"{ent_stats['p50']:.3f} / {ent_stats['p75']:.3f} / {ent_stats['max']:.3f}")

    topk_dist = summary.get("accepted_topk_index_distribution", {})
    if topk_dist:
        print(f"\nAccepted-token top-k index by depth:")
        print(f"  (Among accepted tokens at each depth, which top-k slot did they come from?)")
        for d in sorted(topk_dist.keys()):
            counts = topk_dist[d]
            total_at_d = sum(counts.values())
            parts = []
            for k in sorted(counts.keys()):
                pct = 100.0 * counts[k] / total_at_d if total_at_d else 0
                parts.append(f"top{k+1}={counts[k]} ({pct:.1f}%)")
            print(f"  depth={d}: " + ", ".join(parts))
    
    print(f"\nGenerated Response (first 600 chars):")
    print("-" * 50)
    text = summary['generated_text']
    print(text[:600])
    if len(text) > 600:
        print(f"... [{len(text) - 600} more chars]")


def _to_serializable(obj):
    """Convert torch / numpy types into JSON-friendly Python types."""
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    return obj


def _extract_conversation(sample) -> list:
    """Pull a list-of-turns out of common humaneval / sharegpt sample shapes."""
    if isinstance(sample, dict):
        return sample.get("conversations", sample)
    return sample


def _build_user_prompt_and_count_tokens(
    conversation: list, tokenizer, prompt_text: str | None = None
) -> tuple[str, int]:
    if prompt_text is not None:
        n_tokens = int(tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1])
        return prompt_text, n_tokens

    parts = []
    for turn in conversation:
        try:
            role, content = _normalize_turn(turn)
        except Exception:
            continue
        if role == "system":
            parts.append(f"SYSTEM: {content}")
        elif role == "user":
            parts.append(f"USER: {content}")
        elif role == "assistant":
            break
    prompt = "\n".join(parts) + "\nASSISTANT:"
    n_tokens = int(tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1])
    return prompt, n_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Hydra head acceptance during actual inference"
    )
    parser.add_argument(
        "--model",
        default="ankner/hydra-vicuna-7b-v1.3",
        help="Hydra model name or path (default: ankner/hydra-vicuna-7b-v1.3)",
    )
    parser.add_argument(
        "--base_model",
        default=None,
        help="Override base model name or path",
    )
    parser.add_argument(
        "--data",
        default="data/sharegpt/raw/val.json",
        help="JSONL file with conversations (one JSON per line)",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Which sample to analyze (0-indexed). Ignored if --all_samples.",
    )
    parser.add_argument(
        "--all_samples",
        action="store_true",
        default=False,
        help=(
            "Iterate over every sample in the data file, streaming rows into "
            "a single CSV with a 'sample_idx' column. Per-sample summaries "
            "are written to <output>_summary.json."
        ),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="If set with --all_samples, only run the first N samples.",
    )
    parser.add_argument(
        "--smoothing_windows",
        type=str,
        default="4,16,32",
        help=(
            "Comma-separated EMA window sizes for smoothed block-start "
            "entropy. Each window W uses Adam-style bias-corrected EMA with "
            "beta = 1 - 1/W."
        ),
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (will stop early if approaching context limit)",
    )
    parser.add_argument(
        "--output",
        default="logs/hydra_head_analysis.csv",
        help="Output CSV path for per-step records",
    )
    parser.add_argument(
        "--mode",
        choices=["generation", "both", "teacher_forced"],
        default="generation",
        help=(
            "Compatibility flag. This script always runs real inference "
            "(speculative decoding), so all modes map to generation."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy)",
    )
    parser.add_argument(
        "--posterior_threshold",
        type=float,
        default=0.09,
        help="Posterior threshold for acceptance",
    )
    parser.add_argument(
        "--posterior_alpha",
        type=float,
        default=0.3,
        help="Posterior alpha for acceptance",
    )
    parser.add_argument("--bf16", action="store_true", default=True)
    args = parser.parse_args()

    if args.mode != "generation":
        print(
            f"[warn] --mode {args.mode!r} requested, but this script only runs "
            "actual inference mode. Proceeding with generation analysis."
        )

    os.chdir(ROOT)

    smoothing_windows = tuple(
        int(w.strip()) for w in args.smoothing_windows.split(",") if w.strip()
    )
    if not smoothing_windows:
        smoothing_windows = (4, 16, 32)

    print(f"[info] Loading data from {args.data}")
    conversations = _load_conversations(args.data)
    print(f"[info] Loaded {len(conversations)} conversations")

    if args.all_samples:
        sample_indices = list(range(len(conversations)))
        if args.max_samples is not None:
            sample_indices = sample_indices[: args.max_samples]
    else:
        if args.sample_idx >= len(conversations):
            raise ValueError(
                f"sample_idx {args.sample_idx} >= number of samples {len(conversations)}"
            )
        sample_indices = [args.sample_idx]

    print(f"[info] Will process {len(sample_indices)} sample(s) (windows={smoothing_windows})")

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    print(f"\n[info] Loading Hydra model: {args.model}")
    model = HydraModel.from_pretrained(
        args.model,
        base_model=args.base_model,
        torch_dtype=dtype,
        device_map="auto",
    )
    tokenizer = model.get_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_heads = model.hydra
    max_position_embeddings = int(model.config.max_position_embeddings)
    print(f"[info] Model loaded: {num_heads} Hydra heads, arch={model.hydra_head_arch}, "
          f"max_position_embeddings={max_position_embeddings}")

    summary_path = args.output.replace(".csv", "_summary.json")
    csv_writer = StreamingCSVWriter(args.output)

    all_summaries: dict[str, dict] = {}
    last_raw_summary: dict | None = None
    last_rows: list[dict] = []
    skipped: list[dict] = []
    t_start = time.time()
    successful_samples = 0

    for k, sample_idx in enumerate(sample_indices):
        sample = conversations[sample_idx]
        prompt_text = sample.get("prompt_text") if isinstance(sample, dict) else None
        conversation = _extract_conversation(sample)
        if prompt_text is None and not conversation:
            print(f"[warn] sample {sample_idx}: empty conversation, skipping")
            skipped.append({"sample_idx": sample_idx, "reason": "empty"})
            continue

        try:
            _, n_prompt_tokens = _build_user_prompt_and_count_tokens(
                conversation, tokenizer, prompt_text=prompt_text
            )
        except Exception as e:
            print(f"[warn] sample {sample_idx}: tokenizer failed ({e!r}), skipping")
            skipped.append({"sample_idx": sample_idx, "reason": f"tokenize_failed: {e!r}"})
            continue

        # Refuse to start a sample whose prompt + minimum tree (1 step) won't
        # fit in the KV cache. We also leave a small headroom for safety.
        tree_size = len(mc_sim_7b_63) + 1
        if n_prompt_tokens + tree_size + 8 > max_position_embeddings:
            print(f"[warn] sample {sample_idx}: prompt has {n_prompt_tokens} tokens, "
                  f"would exceed max_position_embeddings={max_position_embeddings} on first step; skipping")
            skipped.append({
                "sample_idx": sample_idx,
                "reason": "prompt_too_long",
                "n_prompt_tokens": n_prompt_tokens,
            })
            continue

        if prompt_text is not None:
            preview = prompt_text[:120].replace("\n", " ")
        else:
            try:
                preview_role, preview_content = _normalize_turn(conversation[0])
                preview = str(preview_content)[:120].replace("\n", " ")
            except Exception:
                preview = str(conversation[0])[:120]

        elapsed_min = (time.time() - t_start) / 60.0
        print(f"\n[info] === sample {sample_idx} ({k+1}/{len(sample_indices)}, "
              f"elapsed {elapsed_min:.1f}min, prompt_tokens={n_prompt_tokens}) ===")
        print(f"[info] preview: {preview}...")

        try:
            rows, summary = run_hydra_generation_with_logging(
                model,
                tokenizer,
                conversation,
                prompt_text=prompt_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                posterior_threshold=args.posterior_threshold,
                posterior_alpha=args.posterior_alpha,
                smoothing_windows=smoothing_windows,
                sample_idx=sample_idx if args.all_samples else None,
            )
        except Exception as e:
            print(f"[error] sample {sample_idx} failed: {e!r}")
            traceback.print_exc()
            skipped.append({
                "sample_idx": sample_idx,
                "reason": f"runtime_error: {e!r}",
            })
            # Try to recover GPU state for the next sample.
            torch.cuda.empty_cache()
            continue

        if rows:
            csv_writer.append_rows(rows)
        successful_samples += 1
        last_raw_summary = summary
        last_rows = rows

        per_sample_summary = {
            k2: _to_serializable(v) for k2, v in summary.items() if k2 != "generated_text"
        }
        per_sample_summary["generated_text_length"] = len(summary.get("generated_text", ""))
        per_sample_summary["n_prompt_tokens"] = n_prompt_tokens
        all_summaries[str(sample_idx)] = per_sample_summary

        print(f"[info] sample {sample_idx}: steps={summary['num_steps']}, "
              f"tokens={summary['total_tokens']}, "
              f"mean_accept={summary['mean_accept_length']:.3f}, "
              f"mean_tokens/step={summary['mean_tokens_per_step']:.2f}x")

        # Best-effort GPU cleanup between samples to avoid fragmentation.
        torch.cuda.empty_cache()

    csv_writer.close()
    if successful_samples > 0:
        print(f"\n[info] Per-step records streamed to: {args.output} "
              f"({successful_samples}/{len(sample_indices)} samples succeeded)")

    aggregate = _aggregate_summaries(all_summaries)
    out_doc = {
        "args": vars(args),
        "smoothing_windows": list(smoothing_windows),
        "num_samples_attempted": len(sample_indices),
        "num_samples_completed": successful_samples,
        "skipped": skipped,
        "aggregate": aggregate,
        "per_sample": all_summaries,
    }
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(_to_serializable(out_doc), f, indent=2)
    print(f"[info] Summary written to: {summary_path}")

    # Friendly aggregate printout when running multi-sample.
    if args.all_samples and aggregate:
        agg = aggregate
        print("\n" + "=" * 70)
        print(f"AGGREGATE OVER {successful_samples} SAMPLES")
        print("=" * 70)
        print(f"  total decoding steps: {agg['total_steps']}")
        print(f"  total tokens generated: {agg['total_tokens']}")
        print(f"  mean accept_length (over all steps): {agg['mean_accept_length']:.3f}")
        print(f"  mean tokens/step (speedup): {agg['mean_tokens_per_step']:.2f}x")
        if agg.get("accept_length_distribution"):
            print("  accept_length distribution (step counts):")
            dist = agg["accept_length_distribution"]
            total = sum(dist.values()) or 1
            for L in sorted(dist.keys()):
                pct = 100.0 * dist[L] / total
                bar = "#" * int(pct / 2)
                print(f"    accept_length={L}: {dist[L]:6d} ({pct:5.1f}%) {bar}")
        ent = agg.get("block_start_entropy")
        if ent:
            print(f"  block_start_entropy: mean={ent['mean']:.3f} std={ent['std']:.3f} "
                  f"min/p25/p50/p75/max={ent['min']:.3f}/{ent['p25']:.3f}/{ent['p50']:.3f}/"
                  f"{ent['p75']:.3f}/{ent['max']:.3f}")

    # Single-sample mode keeps the old detailed printout for parity.
    if not args.all_samples and last_raw_summary is not None:
        print_summary(last_rows, last_raw_summary, num_heads)


def _aggregate_summaries(per_sample: dict[str, dict]) -> dict:
    """Pool per-sample stats into a single aggregate summary."""
    if not per_sample:
        return {}
    total_steps = 0
    total_tokens = 0
    accept_dist: dict[int, int] = defaultdict(int)
    head_counts: dict[int, int] = defaultdict(int)
    topk_dist: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    weighted_accept_sum = 0.0
    ent_means: list[tuple[float, int]] = []  # (mean, weight)
    ent_min, ent_max = math.inf, -math.inf
    for s in per_sample.values():
        n = int(s.get("num_steps", 0) or 0)
        if n == 0:
            continue
        total_steps += n
        total_tokens += int(s.get("total_tokens", 0) or 0)
        weighted_accept_sum += float(s.get("mean_accept_length", 0.0)) * n
        for L, c in (s.get("accept_length_distribution", {}) or {}).items():
            accept_dist[int(L)] += int(c)
        for h, c in (s.get("head_acceptance_counts", {}) or {}).items():
            head_counts[int(h)] += int(c)
        for d, by_k in (s.get("accepted_topk_index_distribution", {}) or {}).items():
            for k, c in (by_k or {}).items():
                topk_dist[int(d)][int(k)] += int(c)
        ent_stats = s.get("block_start_entropy_stats", {}) or {}
        if "mean" in ent_stats:
            ent_means.append((float(ent_stats["mean"]), n))
            if "min" in ent_stats:
                ent_min = min(ent_min, float(ent_stats["min"]))
            if "max" in ent_stats:
                ent_max = max(ent_max, float(ent_stats["max"]))
    out = {
        "total_steps": total_steps,
        "total_tokens": total_tokens,
        "mean_accept_length": (weighted_accept_sum / total_steps) if total_steps else 0.0,
        "mean_tokens_per_step": (total_tokens / total_steps) if total_steps else 0.0,
        "accept_length_distribution": dict(accept_dist),
        "head_acceptance_counts": dict(head_counts),
        "accepted_topk_index_distribution": {d: dict(v) for d, v in topk_dist.items()},
    }
    if ent_means:
        wsum = sum(w for _, w in ent_means) or 1
        out["block_start_entropy"] = {
            "mean": sum(m * w for m, w in ent_means) / wsum,
            "min": ent_min if ent_min != math.inf else 0.0,
            "max": ent_max if ent_max != -math.inf else 0.0,
            "std": float("nan"),  # full std needs raw values; per-sample stds collected in per_sample
            "p25": float("nan"),
            "p50": float("nan"),
            "p75": float("nan"),
        }
    return out


if __name__ == "__main__":
    main()
