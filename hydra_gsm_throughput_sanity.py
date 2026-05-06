"""Throughput + continuation perplexity (base LM) for Hydra vs mc_sim_7b_63.

Uses prompts from any JSON/JSONL dataset understood by `load_sharegpt_prompts`
(GSM8k, HumanEval, Alpaca, ShareGPT, …) and a greedy self-rollout JSON for the
custom tree topology.

Examples (GPU, repo root):

  # GSM8k (default tree from greedy_gsm_self.json)
  python hydra_gsm_throughput_sanity.py --n-prompts 50

  # HumanEval val + n_nodes=25 greedy frontier tree from greedy_humaneval_self.json
  python hydra_gsm_throughput_sanity.py \\
    --data-path data/humaneval/humaneval_val.json \\
    --self-json logs/outputs/pareto_frontier/greedy_humaneval_self.json \\
    --tree-n-nodes 25 --tree-depth 4 \\
    --target-mean-accept 3.005773149196287 \\
    --n-prompts 50 --pool-size 500
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path("/share/cuvl/cc2864/interpretable/Hydra")
sys.path.insert(0, str(ROOT))

from hydra.model.hydra_choices import mc_sim_7b_63  # noqa: E402
from hydra.model.hydra_model import HydraModel  # noqa: E402
from hydra.model.kv_cache import initialize_past_key_values  # noqa: E402
from hydra.model.utils import (  # noqa: E402
    evaluate_posterior,
    generate_hydra_buffers,
    initialize_hydra,
    reset_hydra_mode,
    tree_decoding,
    update_inference_inputs,
)
from scripts.pareto_experiment import load_sharegpt_prompts  # noqa: E402

# Same forward as llm_judge/gen_model_answer_hydra.hydra_forward (no fastchat import)
def hydra_forward(
    input_ids,
    model: HydraModel,
    tokenizer,
    hydra_choices,
    temperature,
    posterior_threshold,
    posterior_alpha,
    max_steps: int = 512,
):
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    input_ids = input_ids.clone()

    if hasattr(model, "hydra_choices") and model.hydra_choices == hydra_choices:
        hydra_buffers = model.hydra_buffers
    else:
        hydra_buffers = generate_hydra_buffers(
            hydra_choices, device=model.base_model.device
        )
    model.hydra_buffers = hydra_buffers
    model.hydra_choices = hydra_choices

    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        current_length_data.zero_()
    else:
        past_key_values, past_key_values_data, current_length_data = (
            initialize_past_key_values(model.base_model, model.hydra_head_arch)
        )
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

    input_len = input_ids.shape[1]
    reset_hydra_mode(model)
    hidden_states, logits = initialize_hydra(
        input_ids,
        model,
        hydra_buffers["hydra_attn_mask"],
        past_key_values,
        hydra_buffers["proposal_cross_attn_masks"],
    )
    new_token = 0
    idx = 0

    for idx in range(max_steps):
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
            logits,
            candidates,
            temperature,
            posterior_threshold,
            posterior_alpha,
            hydra_buffers["max_accepts"],
        )
        input_ids, logits, hidden_states, new_token = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            hydra_buffers["retrieve_indices"],
            logits,
            hidden_states,
            new_token,
            past_key_values_data,
            current_length_data,
            model.hydra_head_arch,
        )
        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token > 1024:
            break
    return input_ids, new_token, idx


DEFAULT_SELF_JSON = ROOT / "logs/outputs/pareto_frontier/greedy_gsm_self.json"
DEFAULT_DATA_PATH = ROOT / "data/gsm8k/val.jsonl"
MAX_STEPS = 256
TEMP = 0.7
EPS = 0.09
ALPHA = 0.3
SAMPLE_SEED = 42


def continuation_ppl_base(model: HydraModel, full_ids: torch.Tensor, prompt_len: int) -> float:
    """exp(mean cross-entropy) of continuation tokens under base LM (teacher-forced on generated text)."""
    reset_hydra_mode(model)
    with torch.no_grad():
        out = model.base_model(full_ids)
        logits = out.logits[0].float()
    ces = []
    for i in range(prompt_len, full_ids.shape[1]):
        ce = F.cross_entropy(
            logits[i - 1 : i], full_ids[0, i : i + 1], reduction="mean"
        )
        ces.append(ce.item())
    if not ces:
        return float("nan")
    return math.exp(sum(ces) / len(ces))


def run_one(
    name: str,
    hydra_choices,
    model: HydraModel,
    tok,
    prompt_text: str,
    device: torch.device,
    max_steps: int,
    temperature: float,
    posterior_threshold: float,
    posterior_alpha: float,
) -> dict:
    input_ids = tok(prompt_text, return_tensors="pt").input_ids.to(device)
    prompt_len = int(input_ids.shape[1])
    if prompt_len > 1400:
        return {"name": name, "error": "prompt too long", "prompt_len": prompt_len}

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out_ids, new_token, last_idx = hydra_forward(
        input_ids,
        model,
        tok,
        hydra_choices,
        temperature,
        posterior_threshold,
        posterior_alpha,
        max_steps=max_steps,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    gen_len = int(out_ids.shape[1]) - prompt_len
    elapsed = t1 - t0
    tps = gen_len / elapsed if elapsed > 0 else 0.0
    ppl = continuation_ppl_base(model, out_ids, prompt_len)
    return {
        "name": name,
        "prompt_tokens": prompt_len,
        "gen_tokens": gen_len,
        "seconds": elapsed,
        "tokens_per_sec": tps,
        "continuation_ppl": ppl,
        "decode_steps": int(last_idx) + 1,
    }


def select_custom_tree(
    data: dict,
    n_nodes: int,
    depth: int,
    target_mean_accept: float | None,
) -> dict:
    cands = [
        r
        for r in data["all_results"]
        if r["n_nodes"] == n_nodes and r["depth"] == depth
    ]
    if not cands:
        raise SystemExit(
            f"No all_results row with n_nodes={n_nodes} and depth={depth} in self-json."
        )
    if target_mean_accept is not None:
        return min(
            cands, key=lambda r: abs(r["mean_accept"] - target_mean_accept)
        )
    if len(cands) > 1:
        raise SystemExit(
            f"{len(cands)} rows match n_nodes={n_nodes} depth={depth}; "
            "disambiguate with --target-mean-accept (biased or true mean_accept from JSON)."
        )
    return cands[0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Calibration/benchmark JSON or JSONL (GSM8k, HumanEval, …).",
    )
    p.add_argument(
        "--self-json",
        type=str,
        default=str(DEFAULT_SELF_JSON),
        help="greedy_*_self.json containing all_results + published_default.",
    )
    p.add_argument(
        "--tree-n-nodes",
        type=int,
        default=28,
        help="Select custom topology row with this n_nodes.",
    )
    p.add_argument(
        "--tree-depth",
        type=int,
        default=4,
        help="Select custom topology row with this depth.",
    )
    p.add_argument(
        "--target-mean-accept",
        type=float,
        default=None,
        help=(
            "If multiple rows match --tree-n-nodes/--tree-depth, pick the one whose "
            "mean_accept is closest to this value (e.g. true self-rollout mean from JSON)."
        ),
    )
    p.add_argument(
        "--n-prompts",
        type=int,
        default=10,
        help="How many prompts to benchmark (after shuffle with --sample-seed).",
    )
    p.add_argument(
        "--pool-size",
        type=int,
        default=2000,
        help="Load this many prompts first (must be >= --n-prompts).",
    )
    p.add_argument("--sample-seed", type=int, default=SAMPLE_SEED)
    p.add_argument("--max-new-tokens", type=int, default=MAX_STEPS)
    p.add_argument("--temperature", type=float, default=TEMP)
    p.add_argument("--posterior-threshold", type=float, default=EPS)
    p.add_argument("--posterior-alpha", type=float, default=ALPHA)
    return p.parse_args()


def main():
    args = parse_args()
    n_prompts = args.n_prompts
    pool_size = max(args.pool_size, n_prompts)
    data_path = Path(args.data_path)
    self_json = Path(args.self_json)

    if not torch.cuda.is_available():
        print(
            "CUDA unavailable — run on a GPU node, e.g.:\n"
            "  cd /share/cuvl/cc2864/interpretable/Hydra && python hydra_gsm_throughput_sanity.py --n-prompts 50"
        )
        sys.exit(1)

    data = json.loads(self_json.read_text())
    custom = select_custom_tree(
        data,
        n_nodes=args.tree_n_nodes,
        depth=args.tree_depth,
        target_mean_accept=args.target_mean_accept,
    )
    custom_paths = custom["tree"]
    custom_tag = f"n{custom['n_nodes']}_d{custom['depth']}"

    pool = load_sharegpt_prompts(str(data_path), n=pool_size, seed=123)
    if len(pool) < n_prompts:
        raise SystemExit(
            f"Only got {len(pool)} prompts from {data_path} with pool_size={pool_size}; "
            f"need at least {n_prompts}. Increase --pool-size or lower --n-prompts."
        )
    rng = random.Random(args.sample_seed)
    rng.shuffle(pool)
    prompts = [p["prompt_text"] for p in pool[:n_prompts]]

    print(f"data_path={data_path}")
    print(f"self_json={self_json}")
    print(
        f"Custom tree: n_nodes={custom['n_nodes']} depth={custom['depth']} "
        f"true mean_accept={custom['mean_accept']:.4f}"
    )
    pd = data.get("published_default")
    if pd is not None:
        print(f"Published default mc_sim_7b_63: mean_accept={pd['mean_accept']:.4f}")
    else:
        print("Published default: (missing in JSON)")
    print(
        f"{n_prompts} prompts (pool_size={pool_size}), sample_seed={args.sample_seed}, "
        f"max_steps={args.max_new_tokens}, tau={args.temperature} "
        f"eps={args.posterior_threshold} alpha={args.posterior_alpha}\n"
    )

    print("Loading HydraModel…")
    model = HydraModel.from_pretrained(
        "ankner/hydra-vicuna-7b-v1.3",
        base_model="lmsys/vicuna-7b-v1.3",
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    tok = model.get_tokenizer()
    dev = model.base_model.device

    rows_custom = []
    rows_pub = []
    for i, text in enumerate(prompts):
        print(f"--- prompt {i + 1}/{n_prompts} ---")
        rc = run_one(
            f"custom_{custom_tag}",
            custom_paths,
            model,
            tok,
            text,
            dev,
            args.max_new_tokens,
            args.temperature,
            args.posterior_threshold,
            args.posterior_alpha,
        )
        rp = run_one(
            "mc_sim_7b_63",
            mc_sim_7b_63,
            model,
            tok,
            text,
            dev,
            args.max_new_tokens,
            args.temperature,
            args.posterior_threshold,
            args.posterior_alpha,
        )
        rows_custom.append(rc)
        rows_pub.append(rp)
        if "error" in rc:
            print(f"  custom:   ERROR {rc}")
        else:
            print(
                f"  custom:   {rc['gen_tokens']:3d} gen tok  {rc['tokens_per_sec']:7.2f} tok/s  "
                f"ppl={rc['continuation_ppl']:.2f}"
            )
        if "error" in rp:
            print(f"  default:  ERROR {rp}")
        else:
            print(
                f"  default:  {rp['gen_tokens']:3d} gen tok  {rp['tokens_per_sec']:7.2f} tok/s  "
                f"ppl={rp['continuation_ppl']:.2f}"
            )

    def avg(rows, key):
        xs = [r[key] for r in rows if key in r and "error" not in r]
        return sum(xs) / len(xs) if xs else float("nan")

    print(f"\n=== Mean over {n_prompts} prompts ===")
    print(
        f"custom {custom_tag}: mean tok/s = {avg(rows_custom, 'tokens_per_sec'):.3f}   "
        f"mean PPL = {avg(rows_custom, 'continuation_ppl'):.3f}"
    )
    print(
        f"mc_sim_7b_63:    mean tok/s = {avg(rows_pub, 'tokens_per_sec'):.3f}   "
        f"mean PPL = {avg(rows_pub, 'continuation_ppl'):.3f}"
    )


if __name__ == "__main__":
    main()
