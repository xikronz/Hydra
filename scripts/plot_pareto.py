"""Visualize the Pareto frontier of draft-tree shapes from pareto_experiment.py.

Reads `pareto_results.json` and produces:
  1. Scatter of all candidate trees in (n_nodes, mean_accept) space, colored by depth.
  2. Overall Pareto frontier (best mean_accept at each node count, any depth).
  3. Per-depth Pareto frontiers (separate curve for each depth).
  4. Labels for the per-cell budget-target winners.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

def format_intish(value) -> str:
    """Comma-separated thousands for integers; plain str otherwise."""
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):,}"
    if isinstance(value, str) and value.isdigit():
        return f"{int(value):,}"
    return str(value)


def pareto_frontier(points: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    """Given (n_nodes, mean_accept) points, return the upper-left Pareto frontier
    (more accept, fewer nodes is better).
    """
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    out: List[Tuple[int, float]] = []
    best = -1.0
    for n, m in pts:
        if m > best + 1e-9:
            out.append((n, m))
            best = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", default="logs/outputs/greedy_tree_382610.json")
    ap.add_argument("--out", default="logs/outputs/greedy_alpaca.png")
    ap.add_argument("--annotate_winners", action="store_true", default=True,
                    help="Label each per-cell budget winner with its width tuple.")
    ap.add_argument("--max_label_n", type=int, default=20,
                    help="Skip annotation labels beyond this many points to avoid clutter.")
    args = ap.parse_args()

    data = json.loads(Path(args.in_json).read_text())
    if "all_results" in data:
        rows = data["all_results"]
    elif "per_depth_history" in data:
        rows = [
            row
            for history in data["per_depth_history"].values()
            for row in history
        ]
    else:
        raise KeyError("Expected all_results or per_depth_history in input JSON")
    winners = data["per_cell_winners"]
    published_default = data.get("published_default")
    if published_default is None:
        published_default = next((r for r in rows if r.get("is_published_default")), None)

    swept_rows = [r for r in rows if not r.get("is_published_default")]
    by_depth: Dict[int, List[Dict]] = {}
    for r in swept_rows:
        by_depth.setdefault(r["depth"], []).append(r)
    depths_sorted = sorted(by_depth.keys())

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(depths_sorted)))
    depth_color = {d: c for d, c in zip(depths_sorted, colors)}

    # ----- Panel 1: scatter + overall Pareto frontier -----
    ax = axes[0]
    for d in depths_sorted:
        rs = by_depth[d]
        ns = [r["n_nodes"] for r in rs]
        ms = [r["mean_accept"] for r in rs]
        ax.scatter(ns, ms, color=depth_color[d], s=42, alpha=0.65,
                   edgecolor="white", linewidth=0.5,
                   label=f"depth={d} (n={len(rs)})")

    all_pts = [(r["n_nodes"], r["mean_accept"]) for r in swept_rows]
    front = pareto_frontier(all_pts)
    fxs, fys = zip(*front)
    ax.plot(fxs, fys, "k--", linewidth=2, label=f"Pareto frontier (n={len(front)})")
    ax.scatter(fxs, fys, color="red", s=90, marker="*",
               edgecolor="black", linewidth=0.6, zorder=5,
               label="frontier points")

    if published_default is not None:
        pd_n = published_default["n_nodes"]
        pd_m = published_default["mean_accept"]
        ax.scatter([pd_n], [pd_m], color="gold", s=320, marker="*",
                   edgecolor="black", linewidth=1.4, zorder=6,
                   label=f"mc_sim_7b_63 (published default)")
        ax.annotate(f"mc_sim_7b_63\n{pd_n} nodes, {pd_m:.3f}",
                    (pd_n, pd_m),
                    xytext=(8, -12), textcoords="offset points",
                    fontsize=9, fontweight="bold", color="darkgoldenrod")

    ax.set_xlabel("Number of tree nodes (compute budget)")
    ax.set_ylabel("Mean accept length per step")
    ax.set_title("All candidate trees + global Pareto frontier")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    if args.annotate_winners:
        for cell_key, w in winners.items():
            label_value = w.get("widths", f"added={w.get('added_node')}")
            label = f"{cell_key}\n{label_value}"
            ax.annotate(label, (w["n_nodes"], w["mean_accept"]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=7, alpha=0.85)

    # ----- Panel 2: per-depth Pareto frontiers -----
    ax = axes[1]
    for d in depths_sorted:
        rs = by_depth[d]
        pts = [(r["n_nodes"], r["mean_accept"]) for r in rs]
        ns = [p[0] for p in pts]
        ms = [p[1] for p in pts]
        front_d = pareto_frontier(pts)
        fxs, fys = zip(*front_d) if front_d else ([], [])

        ax.scatter(ns, ms, color=depth_color[d], s=24, alpha=0.35)
        ax.plot(fxs, fys, "-o", color=depth_color[d], linewidth=2, markersize=6,
                label=f"depth={d} frontier ({len(front_d)} pts)")

    if published_default is not None:
        pd_n = published_default["n_nodes"]
        pd_m = published_default["mean_accept"]
        ax.scatter([pd_n], [pd_m], color="gold", s=320, marker="*",
                   edgecolor="black", linewidth=1.4, zorder=6,
                   label="mc_sim_7b_63")

    ax.set_xlabel("Number of tree nodes (compute budget)")
    ax.set_ylabel("Mean accept length per step")
    ax.set_title("Per-depth Pareto frontiers")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    def fmt(v):
        """Format integers with commas; pass strings through."""
        if isinstance(v, int):
            return f"{v:,}"
        return str(v)

    if "merged_from" in data:
        n_steps = data.get("total_calibration_steps", "?")
        n_prompts = data.get("total_calibration_prompts", "?")
        sources = ", ".join(str(s.get("superset_caps")) for s in data["merged_from"])
        title = (f"Hydra draft-tree Pareto analysis (MERGED)  |  supersets={sources}  |  "
                 f"{fmt(n_steps)} steps / {fmt(n_prompts)} prompts")
    else:
        superset = data.get("candidate_caps")
        if superset is None:
            superset = data.get("superset_caps", "?")
        n_steps = data.get("calibration_steps")
        if n_steps is None:
            lo = data.get("calibration_steps_min")
            hi = data.get("calibration_steps_max")
            if isinstance(lo, int) and isinstance(hi, int):
                n_steps = lo if lo == hi else f"{fmt(lo)}-{fmt(hi)}"
            else:
                n_steps = "?"
        n_prompts = data.get("calibration_prompts", "?")
        rule = data.get("verification_rule", "?")
        tau = data.get("tau", "?")
        title = (f"Hydra draft-tree Pareto analysis  |  superset={superset}  |  "
                 f"{fmt(n_steps)} steps from {fmt(n_prompts)} prompts  |  rule={rule}, tau={tau}")
    fig.suptitle(title, fontsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[ok] saved {out_path}")

    print("\nGlobal Pareto frontier (sweep candidates only):")
    print(f"  {'nodes':>6s}  {'mean_accept':>11s}  {'depth':>5s}  widths")
    rows_by_n = {(r["n_nodes"], r["mean_accept"]): r for r in swept_rows}
    for n, m in front:
        r = rows_by_n[(n, m)]
        if 'widths' in r:
            print(f"  {n:>6d}  {m:>11.4f}  {r['depth']:>5d}  {r['widths']}")
        else:
            print(f"  {n:>6d}  {m:>11.4f}  {r['depth']:>5d}")
    if published_default is not None:
        pd_n = published_default["n_nodes"]
        pd_m = published_default["mean_accept"]
        pd_d = published_default["depth"]
        print(f"\nPublished default tree (separately measured):")
        print(f"  {pd_n:>6d}  {pd_m:>11.4f}  {pd_d:>5d}  mc_sim_7b_63")


if __name__ == "__main__":
    main()
