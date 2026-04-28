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
import os
import sys
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


@torch.inference_mode()
def run_hydra_generation_with_logging(
    model,
    tokenizer,
    conversation: list[dict],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    posterior_threshold: float = 0.09,
    posterior_alpha: float = 0.3,
    hydra_choices=mc_sim_7b_63,
) -> tuple[list[dict], dict]:
    """Run Hydra speculative decoding and log head acceptance at each step.
    
    This uses the exact same inference flow as hydra_generate() but logs
    detailed per-step statistics about which heads were accepted.
    
    Returns:
        rows: List of per-step records with accept_length and head acceptance
        summary: Overall statistics for the generation
    """
    prompt_parts = []
    for turn in conversation:
        role = turn["from"]
        content = turn["value"]
        if role == "human":
            prompt_parts.append(f"USER: {content}")
        elif role == "gpt":
            break
    
    prompt = "\n".join(prompt_parts) + "\nASSISTANT:"
    
    enc = tokenizer(prompt, return_tensors="pt")
    dev = _model_device(model)
    input_ids = enc["input_ids"].to(dev)
    
    print(f"[info] prompt length: {input_ids.shape[1]}, generating up to {max_new_tokens} tokens")
    
    hydra_buffers = generate_hydra_buffers(hydra_choices, device=dev)
    model.hydra_buffers = hydra_buffers
    model.hydra_choices = hydra_choices
    
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
    
    reset_hydra_mode(model)
    hidden_states, logits = initialize_hydra(
        input_ids, model, hydra_buffers["hydra_attn_mask"], past_key_values, hydra_buffers["proposal_cross_attn_masks"]
    )
    
    rows = []
    step_accept_lengths = []
    head_acceptance_counts = defaultdict(int)
    total_generated = 0
    
    for idx in range(max_new_tokens):
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
        
        row = {
            "step": idx,
            "position": int(input_ids.shape[1]),
            "accept_length": accept_len_int,
            "tokens_this_step": accept_len_int + 1,
            "accepted_text": accepted_text.replace("\n", "\\n"),
        }
        
        for head_idx in range(num_heads):
            accepted = head_idx < accept_len_int
            row[f"head{head_idx}_accepted"] = int(accepted)
            if accepted:
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
    
    summary = {
        "num_steps": len(step_accept_lengths),
        "total_tokens": total_generated,
        "mean_accept_length": float(np.mean(step_accept_lengths)) if step_accept_lengths else 0.0,
        "mean_tokens_per_step": total_generated / max(len(step_accept_lengths), 1),
        "accept_length_distribution": accept_dist,
        "head_acceptance_counts": dict(head_acceptance_counts),
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
    
    print(f"\nGenerated Response (first 600 chars):")
    print("-" * 50)
    text = summary['generated_text']
    print(text[:600])
    if len(text) > 600:
        print(f"... [{len(text) - 600} more chars]")


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
        help="Which sample to analyze (0-indexed)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--output",
        default="logs/hydra_head_analysis.csv",
        help="Output CSV path for per-step records",
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

    os.chdir(ROOT)

    print(f"[info] Loading data from {args.data}")
    conversations = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if line:
                conversations.append(json.loads(line))
    
    print(f"[info] Loaded {len(conversations)} conversations")

    if args.sample_idx >= len(conversations):
        raise ValueError(f"sample_idx {args.sample_idx} >= number of samples {len(conversations)}")

    sample = conversations[args.sample_idx]
    conversation = sample.get("conversations", sample)
    print(f"[info] Analyzing sample {args.sample_idx} with {len(conversation)} turns")
    
    if conversation:
        first_turn = conversation[0]
        preview = first_turn.get("value", str(first_turn))[:150]
        print(f"[info] First turn preview: {preview}...")

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
    print(f"[info] Model loaded: {num_heads} Hydra heads, arch={model.hydra_head_arch}")
    
    print(f"\n[info] Running inference with speculative decoding...")
    rows, summary = run_hydra_generation_with_logging(
        model,
        tokenizer,
        conversation,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        posterior_threshold=args.posterior_threshold,
        posterior_alpha=args.posterior_alpha,
    )
    
    print_summary(rows, summary, num_heads)
    
    if rows:
        _write_csv(args.output, rows)
        print(f"\n[info] Per-step records written to: {args.output}")
    
    summary_path = args.output.replace(".csv", "_summary.json")
    summary_to_save = {k: v for k, v in summary.items() if k != "generated_text"}
    summary_to_save["generated_text_length"] = len(summary["generated_text"])
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary_to_save, f, indent=2)
    print(f"[info] Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
