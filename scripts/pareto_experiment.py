"""
Pareto experiment: For each (depth, budget) cell, score all width allocations
by mean accept length, and identify the Pareto-optimal shape per cell.

Setup:
    * Candidate shapes: all positive integer width allocations, depth d in
      {1, 2, 3, 4}, with node count within +/-30% of target budgets
      {16, 32, 64}.
    * Scoring: pack candidates into union-tree batches that fit the model
      position limit, then compute each candidate's post-hoc mean accept
      length under typical acceptance.

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
from typing import Any, Dict, List, Tuple

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
    """Load chat prompts from JSON/JSONL calibration data.

    Supports ShareGPT/OpenAI-style conversations, preformatted `prompt_text`
    rows, and the local ARC/LitBench/HumanEval/GSM8K benchmark schemas. The
    returned text always asks the user prompt and ends at `ASSISTANT:` so the
    model continues from the answer boundary.
    """
    import random

    with open(path) as f:
        first_nonspace = ""
        for ch in f.read(64):
            if not ch.isspace():
                first_nonspace = ch
                break
        f.seek(0)
        if first_nonspace == "[":
            data = json.load(f)
            convs: List[Dict] = data if isinstance(data, list) else [data]
        else:
            convs = []
            for line in f:
                line = line.strip()
                if line:
                    convs.append(json.loads(line))

    def ensure_assistant_prompt(text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        if "ASSISTANT:" in text:
            return text
        if "USER:" in text:
            return text + "\nASSISTANT:"
        return f"USER: {text}\nASSISTANT:"

    def choice_items(choices: Any) -> List[Tuple[str, str]]:
        if isinstance(choices, dict) and "label" in choices and "text" in choices:
            return [
                (str(label), str(text))
                for label, text in zip(choices["label"], choices["text"])
            ]
        if isinstance(choices, dict):
            return [(str(label), str(text)) for label, text in choices.items()]
        if isinstance(choices, list):
            return [
                (chr(ord("A") + i), str(text))
                for i, text in enumerate(choices)
            ]
        return []

    def build_arc_prompt(sample: Dict) -> str:
        choices = "\n".join(
            f"{label}. {text}"
            for label, text in choice_items(sample.get("choices", {}))
        )
        instruction = (
            "Answer the multiple-choice science question.\n\n"
            f"Question: {sample['question']}\n\n"
            f"Choices:\n{choices}\n\n"
            "Give the correct choice and a brief explanation."
        )
        return ensure_assistant_prompt(instruction)

    def build_gsm8k_prompt(sample: Dict) -> str:
        instruction = (
            "Solve this grade school math problem. Show your reasoning and end "
            "with the final answer.\n\n"
            f"{sample['question']}"
        )
        return (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the "
            "user's questions.\n\n"
            f"USER: {instruction}\nASSISTANT:"
        )

    def build_humaneval_prompt(sample: Dict) -> str:
        return ensure_assistant_prompt(
            "Complete this Python function:\n\n" + str(sample["prompt"])
        )

    def build_litbench_prompt(sample: Dict) -> str:
        if "prompt" in sample:
            instruction = (
                "Continue the creative-writing prompt with an engaging story.\n\n"
                f"Prompt: {sample['prompt']}"
            )
        else:
            instruction = (
                "This LitBench test row provides only paired story comment IDs.\n\n"
                f"Chosen comment ID: {sample['chosen_comment_id']}\n"
                f"Rejected comment ID: {sample['rejected_comment_id']}\n\n"
                "Explain what information would be needed to compare the two stories."
            )
        return ensure_assistant_prompt(instruction)

    def build_alpaca_prompt(sample: Dict) -> str:
        instruction = str(sample.get("instruction", "")).strip()
        input_text = str(sample.get("input", "") or "").strip()
        if not instruction:
            return ""
        user_text = f"{instruction}\n\n{input_text}" if input_text else instruction
        return (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the "
            "user's questions.\n\n"
            f"USER: {user_text}\nASSISTANT:"
        )

    def build_prompt_text(sample) -> str:
        # Explicit prompt text from materialized benchmark files.
        if isinstance(sample, dict) and sample.get("prompt_text"):
            return ensure_assistant_prompt(str(sample["prompt_text"]))

        if (
            isinstance(sample, dict)
            and sample.get("source") == "tatsu-lab/alpaca"
            and "instruction" in sample
        ):
            return build_alpaca_prompt(sample)

        if (
            isinstance(sample, dict)
            and "instruction" in sample
            and "output" in sample
            and ("input" in sample or "text" in sample)
        ):
            return build_alpaca_prompt(sample)

        if isinstance(sample, dict) and "question" in sample and "choices" in sample:
            return build_arc_prompt(sample)

        if (
            isinstance(sample, dict)
            and sample.get("source") == "gsm8k"
            and "question" in sample
        ):
            return build_gsm8k_prompt(sample)

        if isinstance(sample, dict) and "question" in sample and "answer" in sample:
            return build_gsm8k_prompt(sample)

        if (
            isinstance(sample, dict)
            and "prompt" in sample
            and (
                "task_id" in sample
                or "canonical_solution" in sample
                or "entry_point" in sample
            )
        ):
            return build_humaneval_prompt(sample)

        if isinstance(sample, dict) and (
            "prompt" in sample
            or {"chosen_comment_id", "rejected_comment_id"}.issubset(sample)
        ):
            return build_litbench_prompt(sample)

        # ShareGPT-style dict with `conversations`.
        if isinstance(sample, dict):
            turns = sample.get("conversations", [])
        # Chat-style top-level list of {role, content} dicts.
        elif isinstance(sample, list):
            turns = sample
        else:
            turns = []

        if not isinstance(turns, list) or not turns:
            return ""

        parts: List[str] = []
        for t in turns:
            if not isinstance(t, dict):
                continue
            role = t.get("from", t.get("role"))
            content = t.get("value", t.get("content", ""))
            if not content:
                continue
            if role == "system":
                parts.append(f"SYSTEM: {content}")
            elif role in {"human", "user"}:
                parts.append(f"USER: {content}")
            elif role in {"gpt", "assistant"}:
                break
        if not parts:
            return ""
        return ensure_assistant_prompt("\n".join(parts))

    rng = random.Random(seed)
    rng.shuffle(convs)

    out: List[Dict] = []
    for i, sample in enumerate(convs):
        prompt_text = build_prompt_text(sample)
        if not prompt_text:
            continue
        sample_id = sample.get("id", "") if isinstance(sample, dict) else f"sample_{i}"
        out.append({
            "id": sample_id,
            "prompt_text": prompt_text,
        })
        if len(out) >= n:
            break
    return out


# Tree enumeration / shape generation

BUDGETS = [16, 32, 64] # default. overridden by --budgets
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


def enumerate_widths_with_node_limit(depth: int, max_nodes: int) -> List[Tuple[int, ...]]:
    """Enumerate all width tuples whose full tree has at most max_nodes nodes."""
    out: List[Tuple[int, ...]] = []

    def rec(prefix: List[int], prod: int, nodes: int):
        if len(prefix) == depth:
            out.append(tuple(prefix))
            return
        remaining_levels = depth - len(prefix) - 1
        # Every later level has at least width 1, so reserve its minimum cost.
        max_w = max_nodes - nodes - prod * remaining_levels
        for w in range(1, max_w + 1):
            next_prod = prod * w
            next_nodes = nodes + next_prod
            if next_nodes + next_prod * remaining_levels > max_nodes:
                break
            rec(prefix + [w], next_prod, next_nodes)

    rec([], 1, 0)
    return out


def fits_in_caps(widths: Tuple[int, ...], caps: Tuple[int, ...] | None) -> bool:
    if caps is None:
        return True
    return len(widths) <= len(caps) and all(widths[k] <= caps[k] for k in range(len(widths)))


def gather_candidates(
    budgets: List[int] = BUDGETS,
    tolerance: float = TOLERANCE,
    max_depth: int = 4,
    caps: Tuple[int, ...] | None = None,
    monotone_only: bool = False,
) -> List[Dict]:
    """All candidate shapes within budget tolerances, optionally capped."""
    out = []
    for d in range(1, max_depth + 1):
        for B in budgets:
            lo = int(B * (1 - tolerance))
            hi = int(B * (1 + tolerance))
            if monotone_only:
                shapes = enumerate_non_increasing(d, max_w=hi)
            else:
                shapes = enumerate_widths_with_node_limit(d, max_nodes=hi)
            for s in shapes:
                if not (lo <= node_count(s) <= hi):
                    continue
                if not fits_in_caps(s, caps):
                    continue
                out.append({
                    "depth": d,
                    "budget_target": B,
                    "widths": list(s),
                    "n_nodes": node_count(s),
                    "paths": enumerate_paths(list(s)),
                })
    return out


def union_paths(candidates_metadata: List[Dict]) -> List[List[int]]:
    paths = {
        tuple(path)
        for meta in candidates_metadata
        for path in meta["paths"]
    }
    return [list(path) for path in sorted(paths, key=lambda x: (len(x), x))]


def pack_candidate_batches(candidates_metadata: List[Dict], max_union_paths: int) -> List[List[int]]:
    """Greedily pack candidates into union trees that fit the position budget."""
    if max_union_paths <= 0:
        raise ValueError("max_union_paths must be positive")

    ordered = sorted(
        range(len(candidates_metadata)),
        key=lambda i: len(candidates_metadata[i]["paths"]),
        reverse=True,
    )
    batches: List[List[int]] = []
    batch_path_sets: List[set[Tuple[int, ...]]] = []

    for idx in ordered:
        cand_paths = {tuple(path) for path in candidates_metadata[idx]["paths"]}
        if len(cand_paths) > max_union_paths:
            raise ValueError(
                f"candidate {idx} has {len(cand_paths)} paths, exceeding max "
                f"union size {max_union_paths}"
            )
        placed = False
        for batch_idx, path_set in enumerate(batch_path_sets):
            if len(path_set | cand_paths) <= max_union_paths:
                batches[batch_idx].append(idx)
                path_set.update(cand_paths)
                placed = True
                break
        if not placed:
            batches.append([idx])
            batch_path_sets.append(set(cand_paths))

    return batches


# ---------------------------------------------------------------------------
# Path-position bookkeeping for the superset
# ---------------------------------------------------------------------------

def build_path_to_pos(hydra_choices: List[List[int]]) -> Dict[Tuple[int, ...], int]:
    out = {(): 0}
    for i, p in enumerate(sorted(hydra_choices, key=lambda x: (len(x), x))):
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


def parse_depth_topk(spec: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        depth_s, topk_s = part.split(":", 1)
        out[int(depth_s)] = int(topk_s)
    return out


def select_second_stage_candidates(source_json: str, topk_by_depth: Dict[int, int]) -> List[Dict]:
    """Select top candidates per (depth, budget) from a packed first-stage run."""
    data = json.loads(Path(source_json).read_text())
    selected: List[Dict] = []
    seen: set[Tuple[int, ...]] = set()

    budgets = data.get("budgets", [])
    rows = [r for r in data["all_results"] if not r.get("is_published_default")]
    for depth in sorted(topk_by_depth):
        topk = topk_by_depth[depth]
        for budget in budgets:
            cell_rows = [
                r for r in rows
                if r["depth"] == depth and r.get("budget_target") == budget
            ]
            cell_rows.sort(key=lambda r: (-r["mean_accept"], r["n_nodes"], r["widths"]))
            for rank, row in enumerate(cell_rows[:topk], start=1):
                widths = tuple(row["widths"])
                if widths in seen:
                    continue
                seen.add(widths)
                selected.append({
                    "depth": row["depth"],
                    "budget_target": row["budget_target"],
                    "widths": list(widths),
                    "n_nodes": row["n_nodes"],
                    "paths": enumerate_paths(list(widths)),
                    "first_stage_mean_accept": row["mean_accept"],
                    "selection_rank": rank,
                })

    return selected


def score_candidate_self_rollout(
    model: HydraModel,
    tok,
    prompts: List[Dict],
    meta: Dict,
    args,
    max_position_embeddings: int,
) -> Dict:
    """Score one topology by rolling out and advancing with that topology only."""
    paths = meta["paths"]
    buffers = generate_hydra_buffers(paths, device=model.base_model.device)
    tree_size = len(paths) + 1
    accept_sum = 0.0
    n_steps = 0
    n_skipped = 0
    completed_prompts = 0

    past_kv, pkv_data, current_length_data = initialize_past_key_values(
        model.base_model, model.hydra_head_arch
    )

    t0 = time.time()
    for p_idx, prompt in enumerate(prompts):
        text = prompt["prompt_text"]
        input_ids = tok(text, return_tensors="pt").input_ids.to(model.base_model.device)
        if input_ids.shape[1] > args.max_input_tokens:
            n_skipped += 1
            continue
        if input_ids.shape[1] + tree_size > max_position_embeddings:
            n_skipped += 1
            continue
        completed_prompts += 1

        current_length_data.zero_()
        reset_hydra_mode(model)
        hidden_states, base_logits = initialize_hydra(
            input_ids, model,
            buffers["hydra_attn_mask"],
            past_kv,
            buffers["proposal_cross_attn_masks"],
        )

        new_token = 0
        for step in range(args.max_new_tokens):
            if input_ids.shape[1] + tree_size > max_position_embeddings:
                break
            to_pass = input_ids if step == 0 else None
            cands_tensor, tree_cands = model.hydra_head.proposal(
                base_logits, hidden_states, buffers, past_kv, to_pass
            )
            hidden_states, logits = tree_decoding(
                model, tree_cands, past_kv,
                buffers["hydra_position_ids"], input_ids,
                buffers["retrieve_indices"],
            )
            best_cand, accept_length = evaluate_posterior(
                logits, cands_tensor, args.temperature,
                args.posterior_threshold, args.posterior_alpha,
                buffers["max_accepts"],
            )
            accept_sum += float(accept_length.item())
            n_steps += 1

            input_ids, base_logits, hidden_states, new_token = update_inference_inputs(
                input_ids, cands_tensor, best_cand, accept_length,
                buffers["retrieve_indices"], logits, hidden_states,
                new_token, pkv_data, current_length_data, model.hydra_head_arch,
            )
            if tok.eos_token_id in input_ids[0]:
                break

        if completed_prompts % 10 == 0:
            elapsed = time.time() - t0
            print(
                f"    [completed {completed_prompts}/{len(prompts)}] {n_steps} steps "
                f"({n_steps / max(elapsed, 1e-9):.1f} steps/s, {n_skipped} skipped)"
            )

    return {
        "depth": meta["depth"],
        "budget_target": meta["budget_target"],
        "widths": meta["widths"],
        "n_nodes": meta["n_nodes"],
        "mean_accept": accept_sum / max(n_steps, 1),
        "n_steps": n_steps,
        "n_skipped": n_skipped,
        "first_stage_mean_accept": meta.get("first_stage_mean_accept"),
        "selection_rank": meta.get("selection_rank"),
    }


def build_cell_winners(results: List[Dict]) -> Dict[str, Dict]:
    cells: Dict[str, List[Dict]] = {}
    for r in results:
        if r.get("is_published_default"):
            continue
        key = f"d{r['depth']}_B{r['budget_target']}"
        cells.setdefault(key, []).append(r)

    cell_winners = {}
    for key, rs in cells.items():
        rs_sorted = sorted(rs, key=lambda r: (-r["mean_accept"], r["n_nodes"]))
        cell_winners[key] = rs_sorted[0]
    return cell_winners


def pareto_frontier_rows(rows: List[Dict]) -> List[Dict]:
    pts = sorted(rows, key=lambda r: (r["n_nodes"], -r["mean_accept"]))
    out: List[Dict] = []
    best = -1.0
    for row in pts:
        if row["mean_accept"] > best + 1e-9:
            out.append(row)
            best = row["mean_accept"]
    return out


def save_second_stage_plot(results: List[Dict], out_path: str, title: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    by_depth: Dict[int, List[Dict]] = {}
    for r in results:
        by_depth.setdefault(r["depth"], []).append(r)
    depths = sorted(by_depth)
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(depths)))
    depth_color = {d: c for d, c in zip(depths, colors)}

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    ax = axes[0]
    for depth in depths:
        rows = by_depth[depth]
        ax.scatter(
            [r["n_nodes"] for r in rows],
            [r["mean_accept"] for r in rows],
            color=depth_color[depth],
            s=52,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.5,
            label=f"depth={depth} (n={len(rows)})",
        )
    front = pareto_frontier_rows(results)
    ax.plot(
        [r["n_nodes"] for r in front],
        [r["mean_accept"] for r in front],
        "k--",
        linewidth=2,
        label=f"true Pareto frontier (n={len(front)})",
    )
    ax.scatter(
        [r["n_nodes"] for r in front],
        [r["mean_accept"] for r in front],
        color="red",
        s=90,
        marker="*",
        edgecolor="black",
        linewidth=0.6,
        zorder=5,
    )
    ax.set_xlabel("Number of tree nodes")
    ax.set_ylabel("Mean accept length per step")
    ax.set_title("Second-stage self-rollout candidates")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for depth in depths:
        rows = by_depth[depth]
        front_d = pareto_frontier_rows(rows)
        ax.scatter(
            [r["n_nodes"] for r in rows],
            [r["mean_accept"] for r in rows],
            color=depth_color[depth],
            s=28,
            alpha=0.35,
        )
        ax.plot(
            [r["n_nodes"] for r in front_d],
            [r["mean_accept"] for r in front_d],
            "-o",
            color=depth_color[depth],
            linewidth=2,
            markersize=6,
            label=f"depth={depth} frontier ({len(front_d)} pts)",
        )
    ax.set_xlabel("Number of tree nodes")
    ax.set_ylabel("Mean accept length per step")
    ax.set_title("Per-depth true Pareto frontiers")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[ok] saved {path}")


def run_second_stage_self_rollout(model: HydraModel, tok, args) -> None:
    topk_by_depth = parse_depth_topk(args.second_stage_topk_by_depth)
    candidates = select_second_stage_candidates(args.second_stage_from_json, topk_by_depth)
    print(
        f"Second-stage self-rollout: selected {len(candidates)} candidates "
        f"from {args.second_stage_from_json}"
    )

    prompts = load_sharegpt_prompts(
        path=args.sharegpt_path,
        n=args.calibration_prompts,
        seed=args.seed,
    )
    print(f"Calibration prompts (ShareGPT): {len(prompts)}")
    max_position_embeddings = model.config.max_position_embeddings

    results = []
    for idx, meta in enumerate(candidates, start=1):
        print(
            f"\n[{idx}/{len(candidates)}] self-rollout widths={meta['widths']} "
            f"depth={meta['depth']} nodes={meta['n_nodes']} "
            f"first_stage={meta.get('first_stage_mean_accept'):.4f}"
        )
        result = score_candidate_self_rollout(
            model, tok, prompts, meta, args, max_position_embeddings
        )
        results.append(result)
        print(
            f"  mean_accept={result['mean_accept']:.4f} "
            f"steps={result['n_steps']} skipped={result['n_skipped']}"
        )

    cell_winners = build_cell_winners(results)
    out = {
        "method": "second_stage_self_rollout",
        "source_json": args.second_stage_from_json,
        "selection_topk_by_depth": topk_by_depth,
        "calibration_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "seed": args.seed,
        "verification_rule": "typical",
        "tau": args.temperature,
        "eps": args.posterior_threshold,
        "alpha": args.posterior_alpha,
        "n_candidates": len(results),
        "calibration_steps_min": min((r["n_steps"] for r in results), default=0),
        "calibration_steps_max": max((r["n_steps"] for r in results), default=0),
        "all_results": results,
        "per_cell_winners": cell_winners,
        "published_default": None,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"\nSaved {args.out_json}")

    if args.second_stage_out_plot:
        save_second_stage_plot(
            results,
            args.second_stage_out_plot,
            (
                "Hydra second-stage true Pareto frontier | "
                f"{len(results)} self-rollout candidates | "
                f"{len(prompts)} prompts"
            ),
        )


@torch.inference_mode()
def run_base_chain_sanity_check(model: HydraModel, tok, args) -> None:
    """Verify that base-generated greedy tokens accept as a synthetic chain."""
    prompts = load_sharegpt_prompts(
        path=args.sharegpt_path,
        n=1,
        seed=args.seed,
    )
    if not prompts:
        raise ValueError("No prompts available for base-chain sanity check")

    device = model.base_model.device
    input_ids = tok(prompts[0]["prompt_text"], return_tensors="pt").input_ids.to(device)
    if input_ids.shape[1] > args.max_input_tokens:
        raise ValueError(
            f"sanity prompt has {input_ids.shape[1]} tokens, exceeding "
            f"max_input_tokens={args.max_input_tokens}"
        )

    reset_hydra_mode(model)
    generated: List[int] = []
    current_ids = input_ids.clone()
    for _ in range(args.base_chain_length):
        outputs = model.base_model(current_ids)
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        generated.append(int(next_token.item()))
        current_ids = torch.cat([current_ids, next_token[:, None]], dim=-1)

    # Verify the whole synthetic chain in one base-model forward pass.  The
    # parent paths use -1 to make it explicit that no Hydra head/top-k node was
    # involved in proposing these tokens.
    reset_hydra_mode(model)
    verify_outputs = model.base_model(current_ids)
    prompt_len = input_ids.shape[1]
    verify_logits: Dict[Tuple[int, ...], torch.Tensor] = {}
    cand_tokens: Dict[Tuple[int, ...], int] = {}
    token_stats = []

    for depth, token_id in enumerate(generated):
        parent = tuple([-1] * depth)
        here = tuple([-1] * (depth + 1))
        logits = verify_outputs.logits[0, prompt_len - 1 + depth].detach().cpu()
        verify_logits[parent] = logits
        cand_tokens[here] = token_id

        probs = torch.softmax(logits.float() / args.temperature, dim=-1)
        entropy = float(-(probs * torch.log(probs + 1e-9)).sum().item())
        threshold = min(args.posterior_threshold, args.posterior_alpha * math.exp(-entropy))
        token_stats.append({
            "depth": depth + 1,
            "token_id": token_id,
            "text": tok.decode([token_id]),
            "prob": float(probs[token_id].item()),
            "threshold": threshold,
        })

    synthetic_path = [[-1] * args.base_chain_length]
    accept_length = compute_step_accepts_for_all_candidates(
        [{"paths": synthetic_path}],
        cand_tokens,
        verify_logits,
        args.temperature,
        args.posterior_threshold,
        args.posterior_alpha,
    )[0]

    print("\nBase-chain sanity check")
    print(f"  prompt_tokens={prompt_len}")
    print(f"  synthetic_path={synthetic_path[0]}")
    print(f"  generated_token_ids={generated}")
    print(f"  generated_text={tok.decode(generated)!r}")
    for stat in token_stats:
        print(
            "  "
            f"depth={stat['depth']} token_id={stat['token_id']} "
            f"prob={stat['prob']:.6g} threshold={stat['threshold']:.6g} "
            f"passes={stat['prob'] > stat['threshold']} text={stat['text']!r}"
        )
    print(f"  accept_length={accept_length}")
    print(f"  mean_accept={float(accept_length):.1f}")


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

    if args.base_chain_sanity_check:
        run_base_chain_sanity_check(model, tok, args)
        return
    if args.second_stage_from_json:
        run_second_stage_self_rollout(model, tok, args)
        return

    budgets = [int(x) for x in args.budgets.split(",")]
    tolerance = float(args.tolerance)
    candidate_caps = (
        tuple(int(x) for x in args.superset_caps.split(","))
        if args.superset_caps
        else None
    )
    print(
        f"Search space: budgets={budgets}, tolerance={tolerance}, "
        f"max_depth={args.max_depth}, caps={candidate_caps}, "
        f"monotone_only={args.monotone_only}"
    )

    candidates_metadata = gather_candidates(
        budgets=budgets,
        tolerance=tolerance,
        max_depth=args.max_depth,
        caps=candidate_caps,
        monotone_only=args.monotone_only,
    )
    if args.single_widths:
        widths = tuple(int(x) for x in args.single_widths.split(","))
        if not fits_in_caps(widths, candidate_caps):
            raise ValueError(
                f"single widths {widths} do not fit candidate caps {candidate_caps}"
            )
        candidates_metadata = [{
            "depth": len(widths),
            "budget_target": node_count(widths),
            "widths": list(widths),
            "n_nodes": node_count(widths),
            "paths": enumerate_paths(list(widths)),
        }]
    print(f"Candidate shapes to score: {len(candidates_metadata)}")
    by_cell: Dict[Tuple[int, int], List[int]] = {}
    for i, m in enumerate(candidates_metadata):
        by_cell.setdefault((m["depth"], m["budget_target"]), []).append(i)
    print(f"  spread across {len(by_cell)} (depth, budget) cells")

    missing_cells = [
        (d, B)
        for d in range(1, args.max_depth + 1)
        for B in budgets
        if (d, B) not in by_cell
    ]
    if missing_cells:
        print(f"[warn] no candidates for cells: {missing_cells}")

    max_position_embeddings = model.config.max_position_embeddings
    max_union_paths = max_position_embeddings - args.max_input_tokens - 1
    if max_union_paths <= 0:
        raise ValueError(
            f"max_input_tokens={args.max_input_tokens} leaves no room for a tree "
            f"under max_position_embeddings={max_position_embeddings}"
        )
    candidate_batches = pack_candidate_batches(candidates_metadata, max_union_paths)
    full_union_n_nodes = len(union_paths(candidates_metadata))
    print(
        f"Max position embeddings: {max_position_embeddings}; "
        f"max union paths per batch: {max_union_paths}"
    )
    print(
        f"Full union would have {full_union_n_nodes} non-root nodes; "
        f"packed into {len(candidate_batches)} scoring batches"
    )

    # Per-candidate accumulators
    accept_sum = [0.0] * len(candidates_metadata)
    step_count = [0] * len(candidates_metadata)
    batch_summaries = []
    published_default = None

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

    for batch_idx, candidate_indices in enumerate(candidate_batches, start=1):
        batch_candidates = [candidates_metadata[i] for i in candidate_indices]
        super_paths = union_paths(batch_candidates)
        super_p2p = build_path_to_pos(super_paths)
        super_buffers = generate_hydra_buffers(super_paths, device=model.base_model.device)
        tree_size = len(super_paths) + 1
        print(
            f"\nScoring batch {batch_idx}/{len(candidate_batches)}: "
            f"{len(candidate_indices)} candidates, tree_size={tree_size}"
        )

        t0 = time.time()
        batch_steps = 0
        n_skipped = 0
        completed_prompts = 0
        for p_idx, prompt in enumerate(prompts):
            text = prompt["prompt_text"]
            input_ids = tok(text, return_tensors="pt").input_ids.to(model.base_model.device)
            if input_ids.shape[1] > args.max_input_tokens:
                n_skipped += 1
                continue
            if input_ids.shape[1] + tree_size > max_position_embeddings:
                n_skipped += 1
                continue
            completed_prompts += 1

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

                # Score every candidate shape in this batch.
                step_accepts = compute_step_accepts_for_all_candidates(
                    batch_candidates, cand_tokens, verify_logits,
                    args.temperature, args.posterior_threshold, args.posterior_alpha,
                )
                for local_j, a in enumerate(step_accepts):
                    global_j = candidate_indices[local_j]
                    accept_sum[global_j] += a
                    step_count[global_j] += 1
                batch_steps += 1

                # Advance state under this batch's superset decision.
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

            if completed_prompts % 10 == 0:
                elapsed = time.time() - t0
                print(f"  [completed {completed_prompts}/{len(prompts)}] {batch_steps} steps  "
                      f"({batch_steps / max(elapsed, 1e-9):.1f} steps/s, "
                      f"{n_skipped} skipped for context limit)")

        batch_summaries.append({
            "batch": batch_idx,
            "n_candidates": len(candidate_indices),
            "tree_size": tree_size,
            "n_steps": batch_steps,
            "n_skipped": n_skipped,
        })
        print(
            f"Batch {batch_idx} done: {batch_steps} total steps across "
            f"{len(prompts)-n_skipped}/{len(prompts)} prompts "
            f"({n_skipped} skipped for exceeding context limit)"
        )

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
        mc_completed_prompts = 0
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
            mc_completed_prompts += 1

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

            if mc_completed_prompts % 10 == 0:
                elapsed = time.time() - t1
                print(f"  [mc_sim_7b_63 completed {mc_completed_prompts}/{len(prompts)}] {mc_n_steps} steps "
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
            "mean_accept": accept_sum[j] / max(step_count[j], 1),
            "n_steps": step_count[j],
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
        "candidate_caps": list(candidate_caps) if candidate_caps is not None else None,
        "full_union_n_nodes": full_union_n_nodes,
        "budgets": budgets,
        "tolerance": tolerance,
        "max_depth": args.max_depth,
        "monotone_only": args.monotone_only,
        "n_scoring_batches": len(candidate_batches),
        "batch_summaries": batch_summaries,
        "n_candidates": len(candidates_metadata),
        "calibration_steps_min": min(step_count) if step_count else 0,
        "calibration_steps_max": max(step_count) if step_count else 0,
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
    p.add_argument("--superset-caps", default="",
                   help="Optional comma-separated max width per depth. Empty means "
                        "exhaustive widths constrained only by node budget.")
    p.add_argument("--budgets", default="16,32,64",
                   help="Comma-separated node budget targets to test.")
    p.add_argument("--tolerance", type=float, default=0.30,
                   help="Per-budget tolerance window: candidates within +/- this "
                        "fraction of each budget are scored.")
    p.add_argument("--max-depth", type=int, default=4,
                   help="Maximum tree depth to enumerate.")
    p.add_argument("--monotone-only", action="store_true",
                   help="Restrict candidates to monotone non-increasing widths "
                        "for the old search space.")
    p.add_argument("--single-widths", default="",
                   help="Optional comma-separated topology to score by itself, "
                        "for example '2,2,2'.")
    p.add_argument("--second-stage-from-json", default="",
                   help="First-stage packed Pareto JSON to select candidates from "
                        "and rerun with individual self-rollouts.")
    p.add_argument("--second-stage-topk-by-depth", default="1:2,2:2,3:3,4:4",
                   help="Comma-separated depth:topk-per-budget selection rule for "
                        "second-stage self-rollout.")
    p.add_argument("--second-stage-out-plot", default="",
                   help="Optional path for the second-stage true Pareto plot.")
    p.add_argument("--score-default-tree", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Run a dedicated rollout under the published mc_sim_7b_63 "
                        "tree and include it as a benchmark in the results.")
    p.add_argument("--base-chain-sanity-check", action="store_true",
                   help="Generate a greedy base-model chain and verify that the "
                        "Pareto acceptance scorer accepts all generated tokens.")
    p.add_argument("--base-chain-length", type=int, default=4,
                   help="Number of base-model greedy tokens to sanity-check.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
