"""De-biased self-rollout evaluation of greedy-search frontier trees.

The greedy tree search in `scripts/greedy_tree_search.py` reports per-tree
`mean_accept` numbers that are biased upward: each candidate tree
`current_tree + frontier_node` is scored *post-hoc* during a rollout that uses
a SUPERSET tree (`current_tree | frontier_nodes`) as the actual draft tree.
The proposals come from the larger tree, so the typical-acceptance scorer
finds tokens at positions that the smaller candidate could not have proposed
on its own. The Hydra paper (Section 4) instead specifies that each tree must
be measured by *simulating speculative decoding using that tree*, i.e. the
tree's own buffers, proposal, verification, and KV-cache update.

This script takes an existing `greedy_*.json`, computes the (biased) Pareto
frontier of all rows across all depths, and re-evaluates each frontier tree
using the canonical Hydra rollout (`evaluate_posterior` + `update_inference_inputs`
with the candidate tree's *own* buffers). It also re-evaluates `mc_sim_7b_63`
on the same prompt basket so it can be plotted as a star at its TRUE mean
accept length.

The rollout function imported from `pareto_experiment.py`
(`score_candidate_self_rollout`) is identical to the published
`hydra_forward` in `llm_judge/gen_model_answer_hydra.py` and to
`HydraModel.hydra_generate` in `hydra/model/hydra_model.py`: same proposal /
tree_decoding / evaluate_posterior / update_inference_inputs sequence.

`mc_sim_7b_63` is evaluated **first** (unless `--skip-default`), then frontier
trees. That way a killed job still leaves `published_default` filled if that
step completed; previously default ran last and partial JSON often had
`published_default: null`.

Usage:
  python scripts/eval_frontier_self_rollout.py \
    --greedy-json   logs/outputs/pareto_frontier/greedy_alpaca.json \
    --data-path     data/alpaca/train.jsonl \
    --out-json      logs/outputs/pareto_frontier/greedy_alpaca_self.json \
    --out-plot      logs/outputs/pareto_frontier/greedy_alpaca_self.png \
    --seed          123 \
    --calibration-prompts 100 \
    --max-new-tokens 256 \
    --max-input-tokens 1400
"""
from __future__ import annotations

import argparse
import json
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
from scripts.pareto_experiment import (  # noqa: E402
    load_sharegpt_prompts,
    score_candidate_self_rollout,
)


def pareto_frontier(rows: List[Dict]) -> List[Dict]:
    """Return rows that are Pareto-optimal in (n_nodes ↑, mean_accept ↑).

    Sorted by `n_nodes` ascending; we keep a row only if it strictly improves
    the running best mean_accept over rows with fewer nodes.
    """
    pts = sorted(rows, key=lambda r: (r["n_nodes"], -r["mean_accept"]))
    out: List[Dict] = []
    best = -float("inf")
    for r in pts:
        if r["mean_accept"] > best + 1e-9:
            out.append(r)
            best = r["mean_accept"]
    return out


def collect_history_rows(greedy_data: Dict) -> List[Dict]:
    """Flatten `per_depth_history` into a single list, attaching depth keys."""
    if "per_depth_history" in greedy_data:
        rows: List[Dict] = []
        for _, history in greedy_data["per_depth_history"].items():
            rows.extend(history)
        return rows
    if "all_results" in greedy_data:
        return [r for r in greedy_data["all_results"] if not r.get("is_published_default")]
    raise KeyError(
        "Expected per_depth_history or all_results in greedy JSON"
    )


def tree_paths_from_row(row: Dict) -> List[List[int]]:
    """Recover the sorted list of paths for a greedy-history row."""
    if "tree" in row:
        return [list(p) for p in row["tree"]]
    raise KeyError(
        "Greedy history row missing `tree` field; cannot reconstruct topology."
    )


def make_frontier_meta(row: Dict) -> Dict:
    paths = tree_paths_from_row(row)
    return {
        "depth": row["depth"],
        "budget_target": row.get("budget_target"),
        "widths": row.get("widths"),
        "n_nodes": row["n_nodes"],
        "paths": paths,
        "first_stage_mean_accept": row["mean_accept"],
    }


def make_default_meta() -> Dict:
    paths = [list(p) for p in mc_sim_7b_63]
    return {
        "depth": max(len(p) for p in paths),
        "budget_target": len(paths),
        "widths": "mc_sim_7b_63",
        "n_nodes": len(paths),
        "paths": paths,
        "first_stage_mean_accept": None,
    }


def save_partial(out_path: Path, payload: Dict) -> None:
    """Atomic-ish write so a kill -9 mid-write doesn't corrupt the file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(out_path)


def already_evaluated_keys(payload: Dict) -> set[Tuple[int, ...]]:
    """Use sorted tuple-of-paths as a stable identity for resume support."""
    keys: set[Tuple[int, ...]] = set()
    for r in payload.get("all_results", []):
        if r.get("is_published_default"):
            continue
        if "tree" in r:
            keys.add(canonical_tree_key(r["tree"]))
    if payload.get("published_default") is not None:
        keys.add(("__published_default__",))
    return keys


def canonical_tree_key(paths: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(p) for p in sorted(paths, key=lambda x: (len(x), x)))


def plot_unbiased_frontier(payload: Dict, out_plot: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [r for r in payload["all_results"] if not r.get("is_published_default")]
    if not rows:
        print(f"[warn] no results to plot in {out_plot}")
        return
    rows.sort(key=lambda r: (r["n_nodes"], -r["mean_accept"]))

    by_depth: Dict[int, List[Dict]] = {}
    for r in rows:
        by_depth.setdefault(r["depth"], []).append(r)
    depths = sorted(by_depth)
    cmap = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, max(len(depths), 1)))
    depth_color = {d: c for d, c in zip(depths, cmap)}

    fig, ax = plt.subplots(figsize=(9, 6))
    for d in depths:
        rs = by_depth[d]
        ax.scatter(
            [r["n_nodes"] for r in rs],
            [r["mean_accept"] for r in rs],
            color=depth_color[d],
            s=46,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
            label=f"depth={d} (n={len(rs)})",
        )
        if "biased_mean_accept" in rs[0]:
            ax.scatter(
                [r["n_nodes"] for r in rs],
                [r["biased_mean_accept"] for r in rs],
                color=depth_color[d],
                s=22,
                alpha=0.35,
                marker="x",
            )

    front = pareto_frontier(rows)
    if front:
        ax.plot(
            [r["n_nodes"] for r in front],
            [r["mean_accept"] for r in front],
            "k--",
            linewidth=2,
            label=f"unbiased Pareto frontier (n={len(front)})",
        )
        ax.scatter(
            [r["n_nodes"] for r in front],
            [r["mean_accept"] for r in front],
            color="red",
            s=110,
            marker="*",
            edgecolor="black",
            linewidth=0.7,
            zorder=5,
        )

    pd = payload.get("published_default")
    if pd is not None:
        ax.scatter(
            [pd["n_nodes"]], [pd["mean_accept"]],
            color="gold", s=380, marker="*",
            edgecolor="black", linewidth=1.6, zorder=6,
            label=f"mc_sim_7b_63 (published default)",
        )
        ax.annotate(
            f"mc_sim_7b_63\n{pd['n_nodes']} nodes, {pd['mean_accept']:.3f}",
            (pd["n_nodes"], pd["mean_accept"]),
            xytext=(8, -12), textcoords="offset points",
            fontsize=9, fontweight="bold", color="darkgoldenrod",
        )

    ax.set_xlabel("Number of tree nodes")
    ax.set_ylabel("True mean accept length per step (self-rollout)")
    title = (
        f"Unbiased greedy-frontier self-rollout  |  "
        f"{Path(payload['data_path']).name}  |  "
        f"{payload['calibration_prompts']} prompts (seed={payload['seed']})"
    )
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_plot, dpi=150, bbox_inches="tight")
    print(f"[ok] saved {out_plot}")
    plt.close(fig)


def build_args_for_rollout(args) -> argparse.Namespace:
    """Bundle the eval hyperparameters that `score_candidate_self_rollout` reads."""
    return argparse.Namespace(
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
        temperature=args.temperature,
        posterior_threshold=args.posterior_threshold,
        posterior_alpha=args.posterior_alpha,
    )


def main(args):
    greedy_path = Path(args.greedy_json)
    out_path = Path(args.out_json)
    out_plot = Path(args.out_plot) if args.out_plot else None

    greedy_data = json.loads(greedy_path.read_text())
    rows = collect_history_rows(greedy_data)
    if not rows:
        raise ValueError(f"No greedy history rows found in {greedy_path}")

    # Compute the global Pareto frontier across all depths from the (biased)
    # mean_accept values that the greedy search itself recorded. We are not
    # using these biased values for any decision other than which trees to
    # re-evaluate; the user accepts that this restricts our truth-test to
    # topologies that the biased greedy thought were best.
    if args.frontier_mode == "global":
        target_rows = pareto_frontier(rows)
    elif args.frontier_mode == "per-depth":
        target_rows = []
        seen: set[Tuple[Tuple[int, ...], ...]] = set()
        for d, rs in sorted(_group_by_depth(rows).items()):
            for r in pareto_frontier(rs):
                key = canonical_tree_key(r["tree"])
                if key not in seen:
                    seen.add(key)
                    target_rows.append(r)
    elif args.frontier_mode == "all":
        target_rows = rows
    else:
        raise ValueError(f"Unknown frontier_mode={args.frontier_mode}")

    target_rows.sort(key=lambda r: (r["n_nodes"], r["depth"]))
    print(
        f"Selected {len(target_rows)} candidate trees from {greedy_path.name} "
        f"(frontier_mode={args.frontier_mode}, total history rows={len(rows)})"
    )

    if out_path.exists() and args.resume:
        payload = json.loads(out_path.read_text())
        done_keys = already_evaluated_keys(payload)
        print(f"[resume] {len(done_keys)} already-evaluated trees in {out_path}")
    else:
        payload = {
            "method": "frontier_self_rollout",
            "source_greedy_json": str(greedy_path),
            "data_path": args.data_path,
            "seed": args.seed,
            "calibration_prompts": args.calibration_prompts,
            "max_new_tokens": args.max_new_tokens,
            "max_input_tokens": args.max_input_tokens,
            "verification_rule": "typical",
            "tau": args.temperature,
            "eps": args.posterior_threshold,
            "alpha": args.posterior_alpha,
            "frontier_mode": args.frontier_mode,
            "n_frontier_trees_target": len(target_rows),
            "all_results": [],
            "published_default": None,
        }
        done_keys = set()

    print(f"Loading Hydra: {args.hydra_checkpoint}")
    model = HydraModel.from_pretrained(
        args.hydra_checkpoint,
        base_model=args.base_model,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    tok = model.get_tokenizer()
    max_position_embeddings = model.config.max_position_embeddings

    prompts = load_sharegpt_prompts(
        path=args.data_path,
        n=args.calibration_prompts,
        seed=args.seed,
    )
    print(f"Calibration prompts ({Path(args.data_path).name}): {len(prompts)}")
    if len(prompts) < args.calibration_prompts:
        print(
            f"[warn] only loaded {len(prompts)} prompts; "
            f"expected {args.calibration_prompts}"
        )

    rollout_args = build_args_for_rollout(args)

    do_default = not args.skip_default and ("__published_default__",) not in done_keys

    # Run mc_sim_7b_63 first so a partial run (preemption, OOM, wall clock)
    # still records TRUE published-default metrics before frontier trees finish.
    pending: List[Tuple[str, Dict]] = []
    if do_default:
        pending.append(("default", make_default_meta()))

    for row in target_rows:
        meta = make_frontier_meta(row)
        key = canonical_tree_key(meta["paths"])
        if key in done_keys:
            continue
        pending.append(("tree", meta))

    print(f"Trees to evaluate: {len(pending)} (skipping {len(target_rows) - sum(1 for _, m in pending if _ == 'tree')} already done)")

    t_start = time.time()
    for idx, (kind, meta) in enumerate(pending, start=1):
        label = "mc_sim_7b_63" if kind == "default" else f"d={meta['depth']} n={meta['n_nodes']}"
        print(
            f"\n[{idx}/{len(pending)}] self-rollout {label} "
            f"(biased_mean={meta.get('first_stage_mean_accept')})"
        )
        result = score_candidate_self_rollout(
            model, tok, prompts, meta, rollout_args, max_position_embeddings
        )
        print(
            f"  TRUE mean_accept={result['mean_accept']:.4f} "
            f"steps={result['n_steps']} skipped={result['n_skipped']}"
        )

        if kind == "default":
            payload["published_default"] = {
                "name": "mc_sim_7b_63",
                "widths": "mc_sim_7b_63",
                "n_nodes": result["n_nodes"],
                "depth": result["depth"],
                "mean_accept": result["mean_accept"],
                "n_steps": result["n_steps"],
                "n_skipped": result["n_skipped"],
                "is_published_default": True,
            }
        else:
            payload["all_results"].append({
                "depth": result["depth"],
                "budget_target": result["budget_target"],
                "widths": result["widths"],
                "n_nodes": result["n_nodes"],
                "mean_accept": result["mean_accept"],
                "biased_mean_accept": result.get("first_stage_mean_accept"),
                "n_steps": result["n_steps"],
                "n_skipped": result["n_skipped"],
                "tree": meta["paths"],
            })
        save_partial(out_path, payload)

    elapsed = time.time() - t_start
    print(f"\nFinished {len(pending)} self-rollouts in {elapsed/60:.1f} min")
    print(f"Saved {out_path}")

    front = pareto_frontier(
        [r for r in payload["all_results"] if not r.get("is_published_default")]
    )
    print("\nUnbiased Pareto frontier (true self-rollout):")
    print(f"  {'nodes':>6s}  {'true_mean':>9s}  {'biased':>9s}  {'depth':>5s}")
    for r in front:
        print(
            f"  {r['n_nodes']:>6d}  {r['mean_accept']:>9.4f}  "
            f"{(r.get('biased_mean_accept') or float('nan')):>9.4f}  {r['depth']:>5d}"
        )
    if payload.get("published_default") is not None:
        pd = payload["published_default"]
        print(
            f"\nmc_sim_7b_63 true mean_accept = {pd['mean_accept']:.4f} "
            f"(nodes={pd['n_nodes']}, depth={pd['depth']}, "
            f"steps={pd['n_steps']}, skipped={pd['n_skipped']})"
        )

    if out_plot is not None:
        plot_unbiased_frontier(payload, out_plot)


def _group_by_depth(rows: List[Dict]) -> Dict[int, List[Dict]]:
    out: Dict[int, List[Dict]] = {}
    for r in rows:
        out.setdefault(r["depth"], []).append(r)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hydra-checkpoint", default="ankner/hydra-vicuna-7b-v1.3")
    p.add_argument("--base-model", default="lmsys/vicuna-7b-v1.3")
    p.add_argument("--greedy-json", required=True,
                   help="Path to a greedy_*.json from scripts/greedy_tree_search.py")
    p.add_argument("--data-path", required=True,
                   help=("Calibration prompt file. Must be the same dataset and "
                         "format that produced --greedy-json."))
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-plot", default="")
    p.add_argument("--calibration-prompts", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-input-tokens", type=int, default=1400)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--posterior-threshold", type=float, default=0.09)
    p.add_argument("--posterior-alpha", type=float, default=0.3)
    p.add_argument("--frontier-mode", choices=["global", "per-depth", "all"],
                   default="global",
                   help=("Which set of trees from the greedy history to "
                         "re-evaluate. 'global' is the cross-depth Pareto "
                         "frontier; 'per-depth' takes each depth's frontier; "
                         "'all' re-evaluates every history row (slow)."))
    p.add_argument("--skip-default", action="store_true",
                   help="Don't run a self-rollout for mc_sim_7b_63.")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="If --out-json exists, skip already-evaluated trees.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
