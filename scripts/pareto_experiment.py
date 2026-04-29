"""
Pareto experiment: For each (depth, budget) cell, score all monotone
non-increasing width allocations by mean accept length, and identify the
Pareto-optimal shape per cell.

Setup:
    * Candidate shapes: all monotone non-increasing (m_1 >= ... >= m_d) with
      m_k <= cap_k, depth d in {1, 2, 3, 4}, node count within +/-30% of
      target budgets {4, 8, 16, 32, 63}.
    * Scoring: log under the superset on a calibration corpus, then for each
      candidate compute its post-hoc mean accept length under typical
      acceptance (no extra base-model forwards).

Output: a JSON file with per-shape mean accept length, grouped by
(depth, budget bin), plus the Pareto-best shape per cell.
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
    generate_hydra_buffers,
    initialize_hydra,
    reset_hydra_mode,
    tree_decoding,
    update_inference_inputs,
    evaluate_posterior,
)
from hydra.model.hydra_choices import mc_sim_7b_63  # noqa: E402


def load_sharegpt_prompts(path: str, n: int, seed: int = 42) -> List[Dict]:
    """Load ShareGPT prompts from a JSONL file (one JSON object per line).

    Each conversation has a `conversations` list of {from, value} dicts. We
    build a prompt text out of the human turns up to (but not including) the
    first gpt response, ending with `ASSISTANT:` so the model continues from
    there. This matches the format used in `hydra/log_heads.py`.
    """
    import random

    convs: List[Dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            convs.append(json.loads(line))

    rng = random.Random(seed)
    rng.shuffle(convs)

    out: List[Dict] = []
    for sample in convs:
        turns = sample.get("conversations", [])
        if not turns:
            continue
        parts: List[str] = []
        for t in turns:
            role = t.get("from")
            content = t.get("value", "")
            if role == "human":
                parts.append(f"USER: {content}")
            elif role == "gpt":
                break
        if not parts:
            continue
        prompt_text = "\n".join(parts) + "\nASSISTANT:"
        out.append({
            "id": sample.get("id", ""),
            "prompt_text": prompt_text,
        })
        if len(out) >= n:
            break
    return out


# Tree enumeration / shape generation

SUPERSET_CAPS = (12, 4, 2, 1) # default. overridden by --superset-caps
BUDGETS = [4, 8, 16, 32, 63, 128] # default. overridden by --budgets
TOLERANCE = 0.30 # default. overridden by --tolerance


def enumerate_paths(widths: List[int]) -> List[List[int]]:
    paths: List[List[int]] = []
    def recurse(prefix, depth):
        if depth == len(widths):
            return
        for i in range(widths[depth]):
            new_path = prefix + [i]
            paths.append(new_path)
            recurse(new_path, depth + 1)
    recurse([], 0)
    return paths


def node_count(widths: Tuple[int, ...]) -> int:
    n, prod = 0, 1
    for w in widths:
        prod *= w
        n += prod
    return n


def enumerate_non_increasing(depth: int, max_w: int = 15) -> List[Tuple[int, ...]]:
    out: List[Tuple[int, ...]] = []
    def rec(prefix, max_allowed):
        if len(prefix) == depth:
            out.append(tuple(prefix))
            return
        for m in range(1, max_allowed + 1):
            rec(prefix + [m], m)
    rec([], max_w)
    return out


def fits_in_superset(widths: Tuple[int, ...], caps: Tuple[int, ...]) -> bool:
    return all(widths[k] <= caps[k] for k in range(len(widths)))


def gather_candidates(
    superset_caps: Tuple[int, ...] = SUPERSET_CAPS,
    budgets: List[int] = BUDGETS,
    tolerance: float = TOLERANCE,
    max_depth: int = None,
    max_w: int = 16,
) -> List[Dict]:
    """All non-increasing shapes within budget tolerances and inside the superset."""
    if max_depth is None:
        max_depth = len(superset_caps)
    out = []
    for d in range(1, max_depth + 1):
        shapes = enumerate_non_increasing(d, max_w=max_w)
        for B in budgets:
            lo = int(B * (1 - tolerance))
            hi = int(B * (1 + tolerance))
            for s in shapes:
                if not (lo <= node_count(s) <= hi):
                    continue
                if not fits_in_superset(s, superset_caps):
                    continue
                out.append({
                    "depth": d,
                    "budget_target": B,
                    "widths": list(s),
                    "n_nodes": node_count(s),
                    "paths": enumerate_paths(list(s)),
                })
    return out


# ---------------------------------------------------------------------------
# Path-position bookkeeping for the superset
# ---------------------------------------------------------------------------

def build_path_to_pos(hydra_choices: List[List[int]]) -> Dict[Tuple[int, ...], int]:
    out = {(): 0}
    for i, p in enumerate(hydra_choices):
        out[tuple(p)] = i + 1
    return out


# ---------------------------------------------------------------------------
# Per-step accept-length under typical acceptance, for an arbitrary sub-tree
# (cached parent-distribution computations across candidates)
# ---------------------------------------------------------------------------

def compute_step_accepts_for_all_candidates(
    candidates_metadata: List[Dict],
    cand_tokens: Dict[Tuple[int, ...], int],
    verify_logits: Dict[Tuple[int, ...], torch.Tensor],
    tau: float, eps: float, alpha: float,
) -> List[int]:
    """For one decoding step, compute accept_length for each candidate shape.

    Returns a list of length len(candidates_metadata) with the accept length
    for each.

    We cache the typical-acceptance threshold and softmaxed probs per parent
    node so that scoring all candidates costs O(unique_nodes) softmaxes plus
    O(total_paths) lookups, not O(candidates * paths).
    """
    # Cache: parent path -> (probs_tensor_on_cpu, threshold_float)
    parent_cache: Dict[Tuple[int, ...], Tuple[torch.Tensor, float]] = {}

    def get_parent_dist(parent: Tuple[int, ...]):
        if parent in parent_cache:
            return parent_cache[parent]
        if parent not in verify_logits:
            return None
        logits = verify_logits[parent]
        probs = torch.softmax(logits.float() / tau, dim=-1)
        H = float(-(probs * torch.log(probs + 1e-9)).sum().item())
        thresh = min(eps, alpha * math.exp(-H))
        parent_cache[parent] = (probs, thresh)
        return parent_cache[parent]

    out: List[int] = []
    for meta in candidates_metadata:
        best = 0
        for path in meta["paths"]:
            accepted = 0
            for k in range(len(path)):
                parent = tuple(path[:k])
                here = tuple(path[:k + 1])
                pd = get_parent_dist(parent)
                if pd is None or here not in cand_tokens:
                    break
                probs, thresh = pd
                cand = cand_tokens[here]
                p_cand = float(probs[cand].item())
                if p_cand > thresh:
                    accepted += 1
                else:
                    break
            if accepted > best:
                best = accepted
        out.append(best)
    return out

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
    K = model.hydra
    print(f"Hydra heads: K={K}")

    superset_caps = tuple(int(x) for x in args.superset_caps.split(","))
    budgets = [int(x) for x in args.budgets.split(",")]
    tolerance = float(args.tolerance)
    print(f"Search space: superset_caps={superset_caps}, budgets={budgets}, tolerance={tolerance}")

    super_paths = enumerate_paths(list(superset_caps))
    super_p2p = build_path_to_pos(super_paths)
    super_buffers = generate_hydra_buffers(super_paths, device=model.base_model.device)
    print(f"Superset {superset_caps}: {len(super_paths)} non-root nodes")

    max_position_embeddings = model.config.max_position_embeddings
    tree_size = len(super_paths) + 1
    print(f"Max position embeddings: {max_position_embeddings}, tree size per step: {tree_size}")
    if tree_size + args.max_input_tokens > max_position_embeddings:
        print(f"[warn] tree_size ({tree_size}) + max_input_tokens ({args.max_input_tokens}) "
              f"> max_position_embeddings ({max_position_embeddings}) -- many prompts will be skipped")

    candidates_metadata = gather_candidates(
        superset_caps=superset_caps, budgets=budgets, tolerance=tolerance,
    )
    print(f"Candidate shapes to score: {len(candidates_metadata)}")
    by_cell: Dict[Tuple[int, int], List[int]] = {}
    for i, m in enumerate(candidates_metadata):
        by_cell.setdefault((m["depth"], m["budget_target"]), []).append(i)
    print(f"  spread across {len(by_cell)} (depth, budget) cells")

    # Per-candidate accumulators
    accept_sum = [0.0] * len(candidates_metadata)
    n_steps = 0

    # KV cache
    past_kv, pkv_data, current_length_data = initialize_past_key_values(
        model.base_model, model.hydra_head_arch
    )

    prompts = load_sharegpt_prompts(
        path=args.sharegpt_path,
        n=args.calibration_prompts,
        seed=args.seed,
    )
    print(f"Calibration prompts (ShareGPT): {len(prompts)}")

    t0 = time.time()
    n_skipped = 0
    for p_idx, prompt in enumerate(prompts):
        text = prompt["prompt_text"]
        input_ids = tok(text, return_tensors="pt").input_ids.to(model.base_model.device)
        if input_ids.shape[1] > args.max_input_tokens:
            n_skipped += 1
            continue
        if input_ids.shape[1] + tree_size > max_position_embeddings:
            n_skipped += 1
            continue

        current_length_data.zero_()
        reset_hydra_mode(model)
        hidden_states, base_logits = initialize_hydra(
            input_ids, model,
            super_buffers["hydra_attn_mask"],
            past_kv,
            super_buffers["proposal_cross_attn_masks"],
        )

        new_token = 0
        for step in range(args.max_new_tokens):
            if input_ids.shape[1] + tree_size > max_position_embeddings:
                break

            to_pass = input_ids if step == 0 else None
            cands_tensor, tree_cands = model.hydra_head.proposal(
                base_logits, hidden_states, super_buffers, past_kv, to_pass
            )
            hidden_states, logits = tree_decoding(
                model, tree_cands, past_kv,
                super_buffers["hydra_position_ids"], input_ids,
                super_buffers["retrieve_indices"],
            )

            # Build path -> (logits, candidate_token) dicts
            retrieve = super_buffers["retrieve_indices"].cpu().tolist()
            pos_to_pd: Dict[int, Tuple[int, int]] = {}
            for p_i, row in enumerate(retrieve):
                for d_i, pos in enumerate(row):
                    if pos < 0:
                        continue
                    pos_to_pd.setdefault(pos, (p_i, d_i))

            verify_logits: Dict[Tuple[int, ...], torch.Tensor] = {}
            cand_tokens: Dict[Tuple[int, ...], int] = {}
            verify_logits[()] = logits[0, 0].detach().cpu()
            cand_tokens[()] = int(tree_cands[0, 0].item())
            for path in super_paths:
                pos = super_p2p[tuple(path)]
                if pos not in pos_to_pd:
                    continue
                p_i, d_i = pos_to_pd[pos]
                verify_logits[tuple(path)] = logits[p_i, d_i].detach().cpu()
                cand_tokens[tuple(path)] = int(cands_tensor[p_i, d_i].item())

            # Score every candidate shape this step
            step_accepts = compute_step_accepts_for_all_candidates(
                candidates_metadata, cand_tokens, verify_logits,
                args.temperature, args.posterior_threshold, args.posterior_alpha,
            )
            for j, a in enumerate(step_accepts):
                accept_sum[j] += a
            n_steps += 1

            # Advance state under the superset's own decision
            best_cand, accept_length = evaluate_posterior(
                logits, cands_tensor, args.temperature,
                args.posterior_threshold, args.posterior_alpha,
                super_buffers["max_accepts"],
            )
            input_ids, base_logits, hidden_states, new_token = update_inference_inputs(
                input_ids, cands_tensor, best_cand, accept_length,
                super_buffers["retrieve_indices"], logits, hidden_states,
                new_token, pkv_data, current_length_data, model.hydra_head_arch,
            )
            if tok.eos_token_id in input_ids[0]:
                break

        if (p_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"[{p_idx+1}/{len(prompts)}] {n_steps} steps  "
                  f"({n_steps / max(elapsed, 1e-9):.1f} steps/s, "
                  f"{n_skipped} skipped for context limit)")

    print(f"Done: {n_steps} total steps across {len(prompts)-n_skipped}/{len(prompts)} prompts "
          f"({n_skipped} skipped for exceeding context limit)")

    #non monotonic default paper-given tree 
    if args.score_default_tree:
        print(f"\nDedicated rollout for mc_sim_7b_63 (n_nodes={len(mc_sim_7b_63)})...")
        mc_buffers = generate_hydra_buffers(mc_sim_7b_63, device=model.base_model.device)
        mc_tree_size = len(mc_sim_7b_63) + 1
        mc_max_depth = max(len(p) for p in mc_sim_7b_63)
        print(f"  tree_size={mc_tree_size}, max_depth={mc_max_depth}")

        mc_accept_sum = 0.0
        mc_n_steps = 0
        mc_n_skipped = 0
        t1 = time.time()
        for p_idx, prompt in enumerate(prompts):
            text = prompt["prompt_text"]
            input_ids = tok(text, return_tensors="pt").input_ids.to(model.base_model.device)
            if input_ids.shape[1] > args.max_input_tokens:
                mc_n_skipped += 1
                continue
            if input_ids.shape[1] + mc_tree_size > max_position_embeddings:
                mc_n_skipped += 1
                continue

            current_length_data.zero_()
            reset_hydra_mode(model)
            hidden_states, base_logits = initialize_hydra(
                input_ids, model,
                mc_buffers["hydra_attn_mask"],
                past_kv,
                mc_buffers["proposal_cross_attn_masks"],
            )

            new_token = 0
            for step in range(args.max_new_tokens):
                if input_ids.shape[1] + mc_tree_size > max_position_embeddings:
                    break
                to_pass = input_ids if step == 0 else None
                cands_tensor, tree_cands = model.hydra_head.proposal(
                    base_logits, hidden_states, mc_buffers, past_kv, to_pass
                )
                hidden_states, logits = tree_decoding(
                    model, tree_cands, past_kv,
                    mc_buffers["hydra_position_ids"], input_ids,
                    mc_buffers["retrieve_indices"],
                )
                best_cand, accept_length = evaluate_posterior(
                    logits, cands_tensor, args.temperature,
                    args.posterior_threshold, args.posterior_alpha,
                    mc_buffers["max_accepts"],
                )
                mc_accept_sum += float(accept_length.item())
                mc_n_steps += 1

                input_ids, base_logits, hidden_states, new_token = update_inference_inputs(
                    input_ids, cands_tensor, best_cand, accept_length,
                    mc_buffers["retrieve_indices"], logits, hidden_states,
                    new_token, pkv_data, current_length_data, model.hydra_head_arch,
                )
                if tok.eos_token_id in input_ids[0]:
                    break

            if (p_idx + 1) % 50 == 0:
                elapsed = time.time() - t1
                print(f"  [mc_sim_7b_63 {p_idx+1}/{len(prompts)}] {mc_n_steps} steps "
                      f"({mc_n_steps / max(elapsed, 1e-9):.1f} steps/s, "
                      f"{mc_n_skipped} skipped)")

        mc_mean_accept = mc_accept_sum / max(mc_n_steps, 1)
        print(f"  mc_sim_7b_63: {mc_n_steps} steps, mean_accept={mc_mean_accept:.4f} "
              f"({mc_n_skipped} prompts skipped)")
        published_default = {
            "name": "mc_sim_7b_63",
            "widths": "mc_sim_7b_63",
            "n_nodes": len(mc_sim_7b_63),
            "depth": mc_max_depth,
            "mean_accept": mc_mean_accept,
            "n_steps": mc_n_steps,
            "n_skipped": mc_n_skipped,
            "is_published_default": True,
        }

    results = []
    for j, meta in enumerate(candidates_metadata):
        results.append({
            "depth": meta["depth"],
            "budget_target": meta["budget_target"],
            "widths": meta["widths"],
            "n_nodes": meta["n_nodes"],
            "mean_accept": accept_sum[j] / max(n_steps, 1),
        })

    if published_default is not None:
        results.append({
            "depth": published_default["depth"],
            "budget_target": published_default["n_nodes"],
            "widths": published_default["widths"],
            "n_nodes": published_default["n_nodes"],
            "mean_accept": published_default["mean_accept"],
            "is_published_default": True,
        })

    cells: Dict[str, List[Dict]] = {}
    for r in results:
        if r.get("is_published_default"):
            continue
        key = f"d{r['depth']}_B{r['budget_target']}"
        cells.setdefault(key, []).append(r)

    #per-cell Pareto-best (max mean_accept; tiebreak smaller n_nodes)
    cell_winners = {}
    for key, rs in cells.items():
        rs_sorted = sorted(rs, key=lambda r: (-r["mean_accept"], r["n_nodes"]))
        cell_winners[key] = rs_sorted[0]

    out = {
        "superset_caps": list(superset_caps),
        "superset_n_nodes": len(super_paths),
        "budgets": budgets,
        "tolerance": tolerance,
        "n_candidates": len(candidates_metadata),
        "calibration_steps": n_steps,
        "calibration_prompts": len(prompts),
        "verification_rule": "typical",
        "tau": args.temperature,
        "eps": args.posterior_threshold,
        "alpha": args.posterior_alpha,
        "all_results": results,
        "per_cell_winners": cell_winners,
        "published_default": published_default,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"\nSaved {args.out_json}")

    #pretty-print summary
    print("\nPer-cell Pareto winners:")
    print(f"{'cell':>10s}  {'best widths':>14s}  {'nodes':>5s}  {'mean_accept':>11s}")
    for key in sorted(cell_winners.keys()):
        w = cell_winners[key]
        print(f"{key:>10s}  {str(w['widths']):>14s}  {w['n_nodes']:>5d}  "
              f"{w['mean_accept']:>11.3f}")

    if published_default is not None:
        print(f"\nPublished default mc_sim_7b_63: "
              f"nodes={published_default['n_nodes']}, "
              f"depth={published_default['depth']}, "
              f"mean_accept={published_default['mean_accept']:.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hydra-checkpoint", default="ankner/hydra-vicuna-7b-v1.3")
    p.add_argument("--base-model", default="lmsys/vicuna-7b-v1.3")
    p.add_argument("--sharegpt-path", required=True)
    p.add_argument("--out-json", default="pareto_results.json")
    p.add_argument("--calibration-prompts", type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-input-tokens", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--posterior-threshold", type=float, default=0.09)
    p.add_argument("--posterior-alpha", type=float, default=0.3)
    p.add_argument("--superset-caps", default="12,4,2,1",
                   help="Comma-separated max width per depth, e.g. '10,5,3,2'.")
    p.add_argument("--budgets", default="4,8,16,32,63,128",
                   help="Comma-separated node budget targets to test.")
    p.add_argument("--tolerance", type=float, default=0.30,
                   help="Per-budget tolerance window: candidates within +/- this "
                        "fraction of each budget are scored.")
    p.add_argument("--score-default-tree", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Run a dedicated rollout under the published mc_sim_7b_63 "
                        "tree and include it as a benchmark in the results.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
