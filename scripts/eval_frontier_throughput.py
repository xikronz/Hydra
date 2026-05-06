"""Stage-two throughput selection for Hydra frontier trees.

Stage one (`eval_frontier_self_rollout.py`) measures true self-rollout
acceptance length for each greedy-search frontier topology. That is necessary
but not sufficient: Hydra's paper selects the final tree by measuring
end-to-end throughput, because topology shape changes per-step cost.

This script performs that second stage. It takes a `greedy_*_self.json`, runs
every frontier tree in `all_results` plus the published `mc_sim_7b_63` topology
on the same prompt basket, and records:

  * output_tokens_per_second: generated continuation tokens / timed seconds
  * seconds_per_decode_step: timed seconds / Hydra decoding loop iterations
  * output_tokens_per_decode_step: generated tokens / Hydra decoding loop iterations

The timing path uses the same proposal -> tree_decoding -> evaluate_posterior ->
update_inference_inputs loop as Hydra's generation code. By default, prompt
prefill is included in the timed region, matching the practical end-to-end
generation cost for each prompt. Use `--exclude-prefill` to time only the
decode loop after `initialize_hydra`.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

DEFAULT_HYDRA_REPO = str(Path(__file__).resolve().parents[1])
HYDRA_REPO = os.environ.get("HYDRA_REPO", DEFAULT_HYDRA_REPO)
if HYDRA_REPO not in sys.path:
    sys.path.insert(0, HYDRA_REPO)

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass

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
from scripts.pareto_experiment import enumerate_paths, load_sharegpt_prompts  # noqa: E402


def release_hydra_decode_state(model: HydraModel) -> None:
    """Free decode caches/buffers between candidates to limit fragmentation on long jobs."""
    for attr in (
        "past_key_values",
        "past_key_values_data",
        "current_length_data",
        "hydra_buffers",
        "hydra_choices",
    ):
        if hasattr(model, attr):
            delattr(model, attr)
    reset_hydra_mode(model)


def canonical_tree_key(paths: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(p) for p in sorted(paths, key=lambda x: (len(x), x)))


def save_partial(out_path: Path, payload: Dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(out_path)


def pareto_rows(rows: List[Dict], score_key: str) -> List[Dict]:
    pts = sorted(rows, key=lambda r: (r["n_nodes"], -r[score_key]))
    out: List[Dict] = []
    best = -float("inf")
    for row in pts:
        score = row[score_key]
        if score > best + 1e-9:
            out.append(row)
            best = score
    return out


def select_source_rows(source_data: Dict, candidate_mode: str) -> List[Dict]:
    rows = [
        r
        for r in source_data.get("all_results", [])
        if not r.get("is_published_default")
    ]
    if not rows and "per_depth_history" in source_data:
        rows = [
            r
            for history in source_data["per_depth_history"].values()
            for r in history
            if not r.get("is_published_default")
        ]

    if candidate_mode == "all":
        return rows
    if candidate_mode == "global-frontier":
        return pareto_rows(rows, "mean_accept")
    if candidate_mode == "per-depth-frontier":
        selected: List[Dict] = []
        seen: set[Tuple[int, ...] | Tuple[Tuple[int, ...], ...]] = set()
        by_depth: Dict[int, List[Dict]] = {}
        for row in rows:
            by_depth.setdefault(row["depth"], []).append(row)
        for depth in sorted(by_depth):
            for row in pareto_rows(by_depth[depth], "mean_accept"):
                key = row_identity(row)
                if key not in seen:
                    seen.add(key)
                    selected.append(row)
        return selected
    raise ValueError(f"Unknown candidate_mode={candidate_mode}")


def row_identity(row: Dict):
    if "tree" in row:
        return canonical_tree_key(row["tree"])
    if isinstance(row.get("widths"), list):
        return tuple(row["widths"])
    raise KeyError("Expected row to have either `tree` or list-valued `widths`")


def row_tree(row: Dict) -> List[List[int]]:
    if "tree" in row:
        return row["tree"]
    widths = row.get("widths")
    if isinstance(widths, list):
        return enumerate_paths(widths)
    raise KeyError("Expected row to have either `tree` or list-valued `widths`")


def make_frontier_candidates(source_data: Dict, candidate_mode: str) -> List[Dict]:
    candidates: List[Dict] = []
    seen: set[Tuple[Tuple[int, ...], ...]] = set()
    for idx, row in enumerate(select_source_rows(source_data, candidate_mode)):
        tree = row_tree(row)
        key = canonical_tree_key(tree)
        if key in seen:
            continue
        seen.add(key)
        widths = row.get("widths")
        widths_tag = (
            "_w" + "-".join(str(x) for x in widths)
            if isinstance(widths, list)
            else ""
        )
        candidates.append({
            "candidate_id": (
                f"frontier_{idx:04d}_n{row['n_nodes']}_d{row['depth']}{widths_tag}"
            ),
            "depth": row["depth"],
            "n_nodes": row["n_nodes"],
            "tree": tree,
            "widths": widths,
            "self_rollout_mean_accept": row.get("mean_accept"),
            "biased_mean_accept": row.get("biased_mean_accept"),
            "budget_target": row.get("budget_target"),
        })
    candidates.sort(key=lambda r: (r["n_nodes"], r["depth"], r["candidate_id"]))
    return candidates


def make_default_candidate() -> Dict:
    paths = [list(p) for p in mc_sim_7b_63]
    return {
        "candidate_id": "mc_sim_7b_63",
        "depth": max(len(p) for p in paths),
        "n_nodes": len(paths),
        "tree": paths,
        "is_published_default": True,
    }


def generate_with_metrics(
    model: HydraModel,
    tok,
    input_ids: torch.Tensor,
    hydra_choices: List[List[int]],
    *,
    max_steps: int,
    max_position_embeddings: int,
    temperature: float,
    posterior_threshold: float,
    posterior_alpha: float,
    include_prefill: bool,
) -> Dict:
    """Run one Hydra generation and return timing + token metrics."""
    assert input_ids.shape[0] == 1
    input_ids = input_ids.clone()
    input_len = input_ids.shape[1]
    # Same headroom rule as `score_candidate_self_rollout` in pareto_experiment.py:
    # tree_decoding extends the KV cache by up to ~len(paths)+1 positions per step.
    tree_size = len(hydra_choices) + 1

    with torch.inference_mode():
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

        reset_hydra_mode(model)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        hidden_states, logits = initialize_hydra(
            input_ids,
            model,
            hydra_buffers["hydra_attn_mask"],
            past_key_values,
            hydra_buffers["proposal_cross_attn_masks"],
        )
        if not include_prefill:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        new_token = 0
        total_accept_length = 0
        decode_steps = 0
        for _ in range(max_steps):
            if input_ids.shape[1] + tree_size > max_position_embeddings:
                break
            to_pass_input_ids = input_ids if decode_steps == 0 else None
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
            total_accept_length += int(accept_length.item())
            decode_steps += 1
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
            if tok.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > 1024:
                break

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        output_tokens = int(input_ids.shape[1] - input_len)
        return {
            "output_tokens": output_tokens,
            "decode_steps": decode_steps,
            "accept_length_sum": total_accept_length,
            "seconds": elapsed,
            "output_tokens_per_second": output_tokens / elapsed if elapsed > 0 else 0.0,
            "seconds_per_decode_step": elapsed / max(decode_steps, 1),
            "output_tokens_per_decode_step": output_tokens / max(decode_steps, 1),
            "accept_length_per_decode_step": total_accept_length / max(decode_steps, 1),
        }


def evaluate_candidate(
    model: HydraModel,
    tok,
    prompts: List[Dict],
    candidate: Dict,
    args,
    *,
    warmup_prompt: Dict | None,
) -> Dict:
    tree = candidate["tree"]
    device = model.base_model.device
    max_pos = int(model.base_model.config.max_position_embeddings)
    tree_size = len(tree) + 1

    try:
        if warmup_prompt is not None:
            ids = tok(
                warmup_prompt["prompt_text"], return_tensors="pt"
            ).input_ids.to(device)
            if ids.shape[1] <= args.max_input_tokens and ids.shape[1] + tree_size <= max_pos:
                generate_with_metrics(
                    model,
                    tok,
                    ids,
                    tree,
                    max_steps=args.max_new_tokens,
                    max_position_embeddings=max_pos,
                    temperature=args.temperature,
                    posterior_threshold=args.posterior_threshold,
                    posterior_alpha=args.posterior_alpha,
                    include_prefill=not args.exclude_prefill,
                )

        totals = {
            "output_tokens": 0,
            "decode_steps": 0,
            "accept_length_sum": 0,
            "seconds": 0.0,
        }
        per_prompt = []
        n_skipped = 0
        for p_idx, prompt in enumerate(prompts, start=1):
            ids = tok(prompt["prompt_text"], return_tensors="pt").input_ids.to(device)
            if ids.shape[1] > args.max_input_tokens:
                n_skipped += 1
                continue
            if ids.shape[1] + tree_size > max_pos:
                n_skipped += 1
                continue
            metrics = generate_with_metrics(
                model,
                tok,
                ids,
                tree,
                max_steps=args.max_new_tokens,
                max_position_embeddings=max_pos,
                temperature=args.temperature,
                posterior_threshold=args.posterior_threshold,
                posterior_alpha=args.posterior_alpha,
                include_prefill=not args.exclude_prefill,
            )
            per_prompt.append({
                "prompt_index": p_idx - 1,
                "prompt_tokens": int(ids.shape[1]),
                **metrics,
            })
            totals["output_tokens"] += metrics["output_tokens"]
            totals["decode_steps"] += metrics["decode_steps"]
            totals["accept_length_sum"] += metrics["accept_length_sum"]
            totals["seconds"] += metrics["seconds"]

            if p_idx % 10 == 0:
                tps = totals["output_tokens"] / max(totals["seconds"], 1e-9)
                print(
                    f"    [{p_idx}/{len(prompts)}] tokens={totals['output_tokens']} "
                    f"steps={totals['decode_steps']} tps={tps:.2f} skipped={n_skipped}"
                )

        result = {
            **{k: v for k, v in candidate.items() if k != "tree"},
            "n_prompts": len(prompts),
            "n_completed_prompts": len(per_prompt),
            "n_skipped": n_skipped,
            "output_tokens": totals["output_tokens"],
            "decode_steps": totals["decode_steps"],
            "accept_length_sum": totals["accept_length_sum"],
            "seconds": totals["seconds"],
            "output_tokens_per_second": (
                totals["output_tokens"] / max(totals["seconds"], 1e-9)
            ),
            "seconds_per_decode_step": totals["seconds"] / max(totals["decode_steps"], 1),
            "output_tokens_per_decode_step": (
                totals["output_tokens"] / max(totals["decode_steps"], 1)
            ),
            "accept_length_per_decode_step": (
                totals["accept_length_sum"] / max(totals["decode_steps"], 1)
            ),
            "tree": tree,
        }
        if args.save_per_prompt:
            result["per_prompt"] = per_prompt
        return result
    finally:
        release_hydra_decode_state(model)
        gc.collect()
        torch.cuda.empty_cache()


def throughput_frontier(rows: List[Dict]) -> List[Dict]:
    pts = sorted(rows, key=lambda r: (r["n_nodes"], -r["output_tokens_per_second"]))
    out: List[Dict] = []
    best = -float("inf")
    for row in pts:
        score = row["output_tokens_per_second"]
        if score > best + 1e-9:
            out.append(row)
            best = score
    return out


def plot_results(payload: Dict, out_plot: Path) -> None:
    import matplotlib.pyplot as plt

    rows = [r for r in payload["all_results"] if not r.get("is_published_default")]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    by_depth: Dict[int, List[Dict]] = {}
    for row in rows:
        by_depth.setdefault(row["depth"], []).append(row)
    for depth, rs in sorted(by_depth.items()):
        ax.scatter(
            [r["n_nodes"] for r in rs],
            [r["output_tokens_per_second"] for r in rs],
            s=48,
            alpha=0.8,
            label=f"depth={depth} (n={len(rs)})",
        )
    front = throughput_frontier(rows)
    ax.plot(
        [r["n_nodes"] for r in front],
        [r["output_tokens_per_second"] for r in front],
        "k--",
        linewidth=2,
        label=f"throughput frontier (n={len(front)})",
    )
    ax.scatter(
        [r["n_nodes"] for r in front],
        [r["output_tokens_per_second"] for r in front],
        color="red",
        marker="*",
        s=110,
        edgecolor="black",
        linewidth=0.7,
        zorder=5,
    )
    pd = payload.get("published_default")
    if pd is not None:
        ax.scatter(
            [pd["n_nodes"]],
            [pd["output_tokens_per_second"]],
            color="gold",
            marker="*",
            s=360,
            edgecolor="black",
            linewidth=1.4,
            zorder=6,
            label="mc_sim_7b_63",
        )
        ax.annotate(
            f"mc_sim_7b_63\n{pd['output_tokens_per_second']:.1f} tok/s",
            (pd["n_nodes"], pd["output_tokens_per_second"]),
            xytext=(8, -12),
            textcoords="offset points",
            fontsize=9,
            color="darkgoldenrod",
            fontweight="bold",
        )
    ax.set_xlabel("Number of tree nodes")
    ax.set_ylabel("Output tokens per second")
    ax.set_title(
        f"Hydra stage-two throughput selection | {Path(payload['data_path']).name}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=150, bbox_inches="tight")
    print(f"[ok] saved {out_plot}")


def already_done(payload: Dict) -> set[str]:
    done = {
        row["candidate_id"]
        for row in payload.get("all_results", [])
        if row.get("candidate_id")
    }
    if payload.get("published_default") is not None:
        done.add("mc_sim_7b_63")
    return done


def main(args) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for throughput timing.")

    self_data = json.loads(Path(args.self_json).read_text())
    candidates = make_frontier_candidates(self_data, args.candidate_mode)
    if not args.skip_default:
        candidates = [make_default_candidate()] + candidates

    prompts = load_sharegpt_prompts(args.data_path, n=args.n_prompts, seed=args.seed)
    if len(prompts) < args.n_prompts:
        raise SystemExit(
            f"Loaded only {len(prompts)} prompts from {args.data_path}; "
            f"expected {args.n_prompts}."
        )
    warmup_prompt = prompts[0] if args.warmup else None

    out_path = Path(args.out_json)
    if out_path.exists() and args.resume:
        payload = json.loads(out_path.read_text())
        done = already_done(payload)
        print(f"[resume] loaded {out_path}; {len(done)} candidates already done")
    else:
        payload = {
            "method": "frontier_throughput",
            "source_self_json": args.self_json,
            "data_path": args.data_path,
            "seed": args.seed,
            "n_prompts": len(prompts),
            "max_new_tokens": args.max_new_tokens,
            "max_input_tokens": args.max_input_tokens,
            "temperature": args.temperature,
            "posterior_threshold": args.posterior_threshold,
            "posterior_alpha": args.posterior_alpha,
            "include_prefill": not args.exclude_prefill,
            "warmup": args.warmup,
            "candidate_mode": args.candidate_mode,
            "all_results": [],
            "published_default": None,
        }
        done = set()

    print(f"Loading Hydra: {args.hydra_checkpoint}")
    model = HydraModel.from_pretrained(
        args.hydra_checkpoint,
        base_model=args.base_model,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    tok = model.get_tokenizer()

    pending = [c for c in candidates if c["candidate_id"] not in done]
    print(
        f"Throughput eval: {len(candidates)} candidates total, "
        f"{len(pending)} pending, {len(prompts)} prompts"
    )

    for i, candidate in enumerate(pending, start=1):
        print(
            f"\n[{i}/{len(pending)}] {candidate['candidate_id']} "
            f"nodes={candidate['n_nodes']} depth={candidate['depth']} "
            f"mean_accept={candidate.get('self_rollout_mean_accept')}"
        )
        result = evaluate_candidate(
            model,
            tok,
            prompts,
            candidate,
            args,
            warmup_prompt=warmup_prompt,
        )
        print(
            f"  {result['output_tokens_per_second']:.2f} tok/s | "
            f"{result['output_tokens_per_decode_step']:.3f} tok/step | "
            f"{1000 * result['seconds_per_decode_step']:.2f} ms/step"
        )
        if result.get("is_published_default"):
            payload["published_default"] = result
        else:
            payload["all_results"].append(result)
        save_partial(out_path, payload)

    rows = [r for r in payload["all_results"] if not r.get("is_published_default")]
    if rows:
        best = max(rows, key=lambda r: r["output_tokens_per_second"])
        print(
            "\nBest frontier tree by throughput: "
            f"{best['candidate_id']} nodes={best['n_nodes']} depth={best['depth']} "
            f"{best['output_tokens_per_second']:.2f} tok/s "
            f"(mean_accept={best.get('self_rollout_mean_accept')})"
        )
    if payload.get("published_default") is not None:
        pd = payload["published_default"]
        print(
            "Published default throughput: "
            f"{pd['output_tokens_per_second']:.2f} tok/s "
            f"({pd['output_tokens_per_decode_step']:.3f} tok/step)"
        )

    if args.out_plot:
        plot_results(payload, Path(args.out_plot))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hydra-checkpoint", default="ankner/hydra-vicuna-7b-v1.3")
    p.add_argument("--base-model", default="lmsys/vicuna-7b-v1.3")
    p.add_argument("--self-json", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-plot", default="")
    p.add_argument("--n-prompts", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-input-tokens", type=int, default=1400)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--posterior-threshold", type=float, default=0.09)
    p.add_argument("--posterior-alpha", type=float, default=0.3)
    p.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--exclude-prefill", action="store_true")
    p.add_argument("--skip-default", action="store_true")
    p.add_argument("--save-per-prompt", action="store_true")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--candidate-mode",
        choices=["all", "global-frontier", "per-depth-frontier"],
        default="all",
        help=(
            "Which rows to throughput-evaluate from --self-json. Use "
            "`per-depth-frontier` to ensure each depth has its own frontier."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
