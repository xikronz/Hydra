"""Merge multiple pareto_results_*.json files into one for joint analysis.

When you run multiple pareto_experiment.py jobs (e.g. pareto.sub + pareto_deep.sub
in parallel), each produces an independent set of (shape -> mean_accept) measurements
on its own calibration corpus. This script merges them by re-weighting each shape's
mean_accept by the calibration steps it contributed, so a shape measured in both
runs gets a combined estimate.

Within each input file, two shapes that share the same widths but differ in
budget_target are kept as separate rows (the budget_target is a labeling
convenience, not part of the shape).

Usage:
    python scripts/merge_pareto.py \
        logs/outputs/pareto_results_*.json logs/outputs/pareto_deep_*.json \
        logs/outputs/pareto_merged.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def shape_key(r: Dict) -> Tuple[int, int, Tuple[int, ...]]:
    return (r["depth"], r["budget_target"], tuple(r["widths"]))


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)

    *in_paths, out_path = sys.argv[1:]
    expanded: List[Path] = []
    for p in in_paths:
        expanded.extend(sorted(Path().glob(p)) if "*" in p else [Path(p)])
    expanded = [p for p in expanded if p.is_file()]
    if not expanded:
        print("No input files matched.")
        sys.exit(1)

    print(f"Merging {len(expanded)} input files:")
    for p in expanded:
        print(f"  - {p}")

    sum_weighted: Dict = defaultdict(float)
    sum_steps: Dict = defaultdict(int)
    n_nodes_of: Dict = {}
    total_steps = 0
    total_prompts = 0
    sources = []

    for p in expanded:
        d = json.loads(p.read_text())
        n_steps = d.get("calibration_steps", 0)
        for r in d["all_results"]:
            k = shape_key(r)
            sum_weighted[k] += r["mean_accept"] * n_steps
            sum_steps[k] += n_steps
            n_nodes_of[k] = r["n_nodes"]
        total_steps += n_steps
        total_prompts += d.get("calibration_prompts", 0)
        sources.append({
            "path": str(p),
            "superset_caps": d.get("superset_caps"),
            "budgets": d.get("budgets"),
            "tolerance": d.get("tolerance"),
            "calibration_steps": n_steps,
            "calibration_prompts": d.get("calibration_prompts"),
            "tau": d.get("tau"),
            "n_candidates": d.get("n_candidates", len(d.get("all_results", []))),
        })

    merged_results = []
    for k, w_sum in sum_weighted.items():
        depth, budget, widths = k
        steps = sum_steps[k]
        merged_results.append({
            "depth": depth,
            "budget_target": budget,
            "widths": list(widths),
            "n_nodes": n_nodes_of[k],
            "mean_accept": w_sum / max(steps, 1),
            "support_steps": steps,
        })

    cells: Dict[str, List[Dict]] = defaultdict(list)
    for r in merged_results:
        cells[f"d{r['depth']}_B{r['budget_target']}"].append(r)
    cell_winners = {}
    for key, rs in cells.items():
        cell_winners[key] = sorted(rs, key=lambda r: (-r["mean_accept"], r["n_nodes"]))[0]

    out = {
        "merged_from": sources,
        "total_calibration_steps": total_steps,
        "total_calibration_prompts": total_prompts,
        "n_unique_shapes": len(merged_results),
        "all_results": sorted(merged_results, key=lambda r: (r["depth"], r["n_nodes"])),
        "per_cell_winners": cell_winners,
    }

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out, indent=2))
    print(f"\n[ok] merged: {len(merged_results)} unique shapes, {total_steps:,} total steps")
    print(f"     -> {out_p}")

    print("\nPer-cell winners after merge:")
    print(f"  {'cell':>10s}  {'best widths':>16s}  {'nodes':>5s}  {'mean_accept':>11s}  {'steps':>7s}")
    for key in sorted(cell_winners.keys()):
        w = cell_winners[key]
        print(f"  {key:>10s}  {str(w['widths']):>16s}  {w['n_nodes']:>5d}  "
              f"{w['mean_accept']:>11.4f}  {w['support_steps']:>7d}")


if __name__ == "__main__":
    main()
