"""
Greedy Hydra tree topology search.

This mirrors the offline tree construction described in the Hydra paper:
start from a one-node tree, repeatedly evaluate the next valid child that could
be added to each expandable parent, and add the child with the best marginal
acceptance-length improvement.  For each requested max depth, the script grows
one greedy sequence of trees and reports the best tree whose node count falls
inside each requested budget tolerance window.
"""
from __future__ import annotations

import argparse
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
from scripts.pareto_experiment import (  # noqa: E402
    build_path_to_pos,
    compute_step_accepts_for_all_candidates,
    load_sharegpt_prompts,
)


PathTuple = Tuple[int, ...]


def sorted_paths(paths: set[PathTuple]) -> List[List[int]]:
    return [list(path) for path in sorted(paths, key=lambda x: (len(x), x))]


def node_frontier(
    tree: set[PathTuple],
    max_depth: int,
    max_children_per_parent: int,
) -> List[PathTuple]:
    """Return the next contiguous top-k child for every expandable parent."""
    parents = {()}
    parents.update(path for path in tree if len(path) < max_depth)

    frontier: List[PathTuple] = []
    for parent in sorted(parents, key=lambda x: (len(x), x)):
        if len(parent) >= max_depth:
            continue
        child_count = sum(
            1
            for path in tree
            if len(path) == len(parent) + 1 and path[:-1] == parent
        )
        if child_count < max_children_per_parent:
            frontier.append(parent + (child_count,))
    return frontier


def longest_current_prefix_lengths(
    retrieve_indices: torch.Tensor,
    path_to_pos: Dict[PathTuple, int],
    current_tree: set[PathTuple],
    device: torch.device,
) -> torch.Tensor:
    pos_to_path = {pos: path for path, pos in path_to_pos.items() if path}
    out: List[int] = []
    for row in retrieve_indices.cpu().tolist():
        accepted = 0
        for pos in row[1:]:
            if pos <= 0:
                break
            path = pos_to_path[pos]
            if path not in current_tree:
                break
            accepted += 1
        out.append(accepted)
    return torch.tensor(out, dtype=torch.long, device=device)


def score_frontier_once(
    model: HydraModel,
    tok,
    prompts: List[Dict],
    current_tree: set[PathTuple],
    frontier: List[PathTuple],
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
    posterior_threshold: float,
    posterior_alpha: float,
) -> Dict:
    """Score current_tree and current_tree + each frontier node on one rollout."""
    scoring_tree = set(current_tree)
    scoring_tree.update(frontier)
    scoring_paths = sorted_paths(scoring_tree)
    path_to_pos = build_path_to_pos(scoring_paths)
    buffers = generate_hydra_buffers(scoring_paths, device=model.base_model.device)
    tree_size = len(scoring_paths) + 1
    max_position_embeddings = model.config.max_position_embeddings

    if tree_size + max_input_tokens > max_position_embeddings:
        raise ValueError(
            f"scoring tree_size={tree_size} leaves too little context for "
            f"max_input_tokens={max_input_tokens}"
        )

    candidate_metadata = [{"paths": sorted_paths(current_tree)}]
    candidate_metadata.extend(
        {"paths": sorted_paths(current_tree | {node})}
        for node in frontier
    )
    accept_sum = [0.0] * len(candidate_metadata)
    n_steps = 0
    n_skipped = 0

    past_kv, pkv_data, current_length_data = initialize_past_key_values(
        model.base_model, model.hydra_head_arch
    )
    current_max_accepts = longest_current_prefix_lengths(
        buffers["retrieve_indices"], path_to_pos, current_tree, model.base_model.device
    )

    t0 = time.time()
    for p_idx, prompt in enumerate(prompts):
        text = prompt["prompt_text"]
        input_ids = tok(text, return_tensors="pt").input_ids.to(model.base_model.device)
        if input_ids.shape[1] > max_input_tokens:
            n_skipped += 1
            continue
        if input_ids.shape[1] + tree_size > max_position_embeddings:
            n_skipped += 1
            continue

        current_length_data.zero_()
        reset_hydra_mode(model)
        hidden_states, base_logits = initialize_hydra(
            input_ids,
            model,
            buffers["hydra_attn_mask"],
            past_kv,
            buffers["proposal_cross_attn_masks"],
        )

        new_token = 0
        for step in range(max_new_tokens):
            if input_ids.shape[1] + tree_size > max_position_embeddings:
                break

            to_pass = input_ids if step == 0 else None
            cands_tensor, tree_cands = model.hydra_head.proposal(
                base_logits, hidden_states, buffers, past_kv, to_pass
            )
            hidden_states, logits = tree_decoding(
                model,
                tree_cands,
                past_kv,
                buffers["hydra_position_ids"],
                input_ids,
                buffers["retrieve_indices"],
            )

            retrieve = buffers["retrieve_indices"].cpu().tolist()
            pos_to_pd: Dict[int, Tuple[int, int]] = {}
            for p_i, row in enumerate(retrieve):
                for d_i, pos in enumerate(row):
                    if pos < 0:
                        continue
                    pos_to_pd.setdefault(pos, (p_i, d_i))

            verify_logits: Dict[PathTuple, torch.Tensor] = {
                (): logits[0, 0].detach().cpu()
            }
            cand_tokens: Dict[PathTuple, int] = {
                (): int(tree_cands[0, 0].item())
            }
            for path in scoring_paths:
                path_t = tuple(path)
                pos = path_to_pos[path_t]
                if pos not in pos_to_pd:
                    continue
                p_i, d_i = pos_to_pd[pos]
                verify_logits[path_t] = logits[p_i, d_i].detach().cpu()
                cand_tokens[path_t] = int(cands_tensor[p_i, d_i].item())

            step_accepts = compute_step_accepts_for_all_candidates(
                candidate_metadata,
                cand_tokens,
                verify_logits,
                temperature,
                posterior_threshold,
                posterior_alpha,
            )
            for j, accept in enumerate(step_accepts):
                accept_sum[j] += accept
            n_steps += 1

            # Advance according to the current tree, even though the scoring
            # pass also includes one-step frontier probes.
            best_cand, accept_length = evaluate_posterior(
                logits,
                cands_tensor,
                temperature,
                posterior_threshold,
                posterior_alpha,
                current_max_accepts,
            )
            input_ids, base_logits, hidden_states, new_token = update_inference_inputs(
                input_ids,
                cands_tensor,
                best_cand,
                accept_length,
                buffers["retrieve_indices"],
                logits,
                hidden_states,
                new_token,
                pkv_data,
                current_length_data,
                model.hydra_head_arch,
            )
            if tok.eos_token_id in input_ids[0]:
                break

        if (p_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(
                f"    [{p_idx + 1}/{len(prompts)}] {n_steps} steps "
                f"({n_steps / max(elapsed, 1e-9):.1f} steps/s, "
                f"{n_skipped} skipped)"
            )

    means = [x / max(n_steps, 1) for x in accept_sum]
    return {
        "tree_size": len(current_tree),
        "scoring_tree_size": len(scoring_tree),
        "frontier": [list(node) for node in frontier],
        "current_mean_accept": means[0],
        "candidate_mean_accepts": means[1:],
        "n_steps": n_steps,
        "n_skipped": n_skipped,
    }


def choose_budget_winners(
    history: List[Dict],
    budgets: List[int],
    tolerance: float,
) -> Dict[str, Dict]:
    winners = {}
    for budget in budgets:
        lo = int(budget * (1 - tolerance))
        hi = int(budget * (1 + tolerance))
        eligible = [
            row for row in history
            if lo <= row["n_nodes"] <= hi
        ]
        if not eligible:
            winners[f"B{budget}"] = {
                "budget_target": budget,
                "error": "no greedy tree in tolerance window",
            }
            continue
        winners[f"B{budget}"] = max(
            eligible,
            key=lambda row: (row["mean_accept"], -row["n_nodes"]),
        )
    return winners


def main(args):
    print(f"Loading Hydra: {args.hydra_checkpoint}")
    model = HydraModel.from_pretrained(
        args.hydra_checkpoint,
        base_model=args.base_model,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    tok = model.get_tokenizer()

    budgets = [int(x) for x in args.budgets.split(",")]
    depths = [int(x) for x in args.depths.split(",")]
    tolerance = float(args.tolerance)
    max_budget_nodes = int(max(budgets) * (1 + tolerance))

    prompts = load_sharegpt_prompts(
        path=args.sharegpt_path,
        n=args.calibration_prompts,
        seed=args.seed,
    )
    print(
        f"Search: depths={depths}, budgets={budgets}, tolerance={tolerance}, "
        f"max_nodes={max_budget_nodes}, max_children_per_parent={args.max_children_per_parent}"
    )
    print(f"Calibration prompts: {len(prompts)}")

    all_depth_histories = {}
    per_cell_winners = {}

    for max_depth in depths:
        print(f"\n=== Greedy search for max_depth={max_depth} ===")
        tree: set[PathTuple] = {(0,)}
        history: List[Dict] = []

        while len(tree) <= max_budget_nodes:
            frontier = node_frontier(
                tree,
                max_depth=max_depth,
                max_children_per_parent=args.max_children_per_parent,
            )
            if not frontier:
                print("  frontier exhausted")
                break

            print(
                f"  size={len(tree)} frontier={len(frontier)} "
                f"scoring_tree_size={len(tree | set(frontier))}"
            )
            scores = score_frontier_once(
                model=model,
                tok=tok,
                prompts=prompts,
                current_tree=tree,
                frontier=frontier,
                max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                posterior_threshold=args.posterior_threshold,
                posterior_alpha=args.posterior_alpha,
            )

            best_idx = max(
                range(len(frontier)),
                key=lambda i: (
                    scores["candidate_mean_accepts"][i],
                    -sum(frontier[i]),
                    -len(frontier[i]),
                ),
            )
            best_node = frontier[best_idx]
            best_mean = scores["candidate_mean_accepts"][best_idx]
            improvement = best_mean - scores["current_mean_accept"]
            tree.add(best_node)

            row = {
                "depth": max_depth,
                "n_nodes": len(tree),
                "mean_accept": best_mean,
                "added_node": list(best_node),
                "improvement": improvement,
                "tree": sorted_paths(tree),
                "n_steps": scores["n_steps"],
                "n_skipped": scores["n_skipped"],
            }
            history.append(row)
            print(
                f"    add {list(best_node)} -> nodes={len(tree)} "
                f"mean_accept={best_mean:.4f} improvement={improvement:.4f}"
            )

            if len(tree) >= max_budget_nodes:
                break

        depth_key = f"d{max_depth}"
        all_depth_histories[depth_key] = history
        winners = choose_budget_winners(history, budgets, tolerance)
        for budget_key, winner in winners.items():
            per_cell_winners[f"{depth_key}_{budget_key}"] = winner

    out = {
        "method": "greedy_node_addition",
        "budgets": budgets,
        "depths": depths,
        "tolerance": tolerance,
        "max_children_per_parent": args.max_children_per_parent,
        "calibration_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "posterior_threshold": args.posterior_threshold,
        "posterior_alpha": args.posterior_alpha,
        "per_depth_history": all_depth_histories,
        "per_cell_winners": per_cell_winners,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"\nSaved {args.out_json}")

    print("\nPer-cell greedy winners:")
    print(f"{'cell':>8s}  {'nodes':>5s}  {'mean_accept':>11s}  {'last_added':>16s}")
    for key in sorted(per_cell_winners):
        winner = per_cell_winners[key]
        if "error" in winner:
            print(f"{key:>8s}  {winner['error']}")
            continue
        print(
            f"{key:>8s}  {winner['n_nodes']:>5d}  "
            f"{winner['mean_accept']:>11.3f}  {str(winner['added_node']):>16s}"
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hydra-checkpoint", default="ankner/hydra-vicuna-7b-v1.3")
    p.add_argument("--base-model", default="lmsys/vicuna-7b-v1.3")
    p.add_argument("--sharegpt-path", required=True)
    p.add_argument("--out-json", default="logs/outputs/greedy_tree_search.json")
    p.add_argument("--calibration-prompts", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-input-tokens", type=int, default=1400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--posterior-threshold", type=float, default=0.09)
    p.add_argument("--posterior-alpha", type=float, default=0.3)
    p.add_argument("--budgets", default="16,32,64")
    p.add_argument("--depths", default="1,2,3,4")
    p.add_argument("--tolerance", type=float, default=0.40)
    p.add_argument("--max-children-per-parent", type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
