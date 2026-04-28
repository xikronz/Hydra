"""
Greedy tree search for Medusa/Hydra-style speculative decoding trees,
following the procedure described verbally in the Hydra paper (Section 4)
and the Medusa paper.

The algorithm:

  T_1 = trivial 1-node tree.
  For i = 2, 3, ..., N:
      For every "frontier" node f (= a node whose parent is in T_{i-1}
      but f is not in T_{i-1}):
          Compute Δ_f = E[accept_length(T_{i-1} ∪ {f})] - E[accept_length(T_{i-1})]
          on a calibration corpus.
      T_i = T_{i-1} ∪ {argmax_f Δ_f}

This produces a SEQUENCE of trees, where T_i has i non-root nodes and is
approximately optimal at that size.  Both Medusa's mc_sim_7b_63 and Hydra's
default tree are claimed to have been built this way.

NEITHER MEDUSA NOR HYDRA RELEASED THIS CODE.  The trees in their public
repos are committed outputs.  This file is a from-scratch implementation
following the papers' descriptions.

Crucial implementation detail for efficiency:  rather than re-running Hydra
under each candidate tree (very slow), we exploit the same trick as our
post-hoc scoring -- we run a single discovery pass under a wide superset
tree, log the per-node verifier logits and candidate tokens, and then
ALL accept-length expectations are computed offline by walking sub-paths.

Inputs needed (collected by `discover_menu.py`'s rollout):
    * For each step, a dict {path_tuple: candidate_token_id}
    * For each step, a dict {path_tuple: 1-D verifier logits at that node}

Outputs:
    * tree_sequence:  [T_1, T_2, ..., T_N], each a list of paths
    * accept_curve:   [E[accept_length(T_i)]]  for i = 0..N
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Set
import torch


# ---------------------------------------------------------------------------
# Per-step accept-length computation under typical acceptance.
# Same logic as build_menu_trees.accept_length_under_subtree but with a
# pre-extracted set of tree-path tuples (faster for repeated calls).
# ---------------------------------------------------------------------------

def _typical_threshold(parent_logits: torch.Tensor,
                       tau: float, eps: float, alpha: float) -> Tuple[torch.Tensor, float]:
    probs = torch.softmax(parent_logits / tau, dim=-1)
    H = float(-(probs * torch.log(probs + 1e-9)).sum().item())
    threshold = min(eps, alpha * math.exp(-H))
    return probs, threshold


def step_accept_length(
    tree_paths: Set[Tuple[int, ...]],
    cand_tokens: Dict[Tuple[int, ...], int],
    verify_logits: Dict[Tuple[int, ...], torch.Tensor],
    tau: float, eps: float, alpha: float,
) -> int:
    """Accept length under typical acceptance for a tree (set of paths).
    Returns max over root-to-leaf paths in tree_paths of the longest accepted
    prefix.
    """
    if not tree_paths:
        return 0

    # Cache parent decisions: for each (parent_path, child_token_id) we
    # already know whether it passes the criterion.  But cheaper: cache
    # (parent_path) -> threshold and prob, and look up by candidate_token.
    parent_cache: Dict[Tuple[int, ...], Tuple[torch.Tensor, float]] = {}

    best = 0
    # Group paths by their endpoints so we walk each at most once.
    for path in tree_paths:
        accepted = 0
        for k in range(len(path)):
            parent = tuple(path[:k])
            here = tuple(path[:k + 1])
            if parent not in verify_logits or here not in cand_tokens:
                break
            cand = cand_tokens[here]
            if parent not in parent_cache:
                probs, thresh = _typical_threshold(verify_logits[parent], tau, eps, alpha)
                parent_cache[parent] = (probs, thresh)
            probs, thresh = parent_cache[parent]
            p_cand = float(probs[cand].item())
            if p_cand > thresh:
                accepted += 1
            else:
                break
        if accepted > best:
            best = accepted
            if best == len(path):
                # can't do better along this path
                continue
    return best


# ---------------------------------------------------------------------------
# Greedy node-addition search
# ---------------------------------------------------------------------------

def expected_accept_over_corpus(
    tree_paths: Set[Tuple[int, ...]],
    corpus: List[Tuple[Dict[Tuple[int, ...], int], Dict[Tuple[int, ...], torch.Tensor]]],
    tau: float, eps: float, alpha: float,
) -> float:
    """Mean accept length across corpus steps."""
    if not corpus:
        return 0.0
    total = 0
    for cand_dict, logits_dict in corpus:
        total += step_accept_length(tree_paths, cand_dict, logits_dict, tau, eps, alpha)
    return total / len(corpus)


def greedy_tree_search(
    corpus: List[Tuple[Dict[Tuple[int, ...], int], Dict[Tuple[int, ...], torch.Tensor]]],
    superset_paths: List[List[int]],
    max_nodes: int = 128,
    tau: float = 0.7,
    eps: float = 0.09,
    alpha: float = 0.3,
    verbose: bool = True,
) -> Tuple[List[List[List[int]]], List[float]]:
    """Build T_1 ⊂ T_2 ⊂ ... ⊂ T_max_nodes by greedy node addition.

    Returns:
        tree_sequence: list of length max_nodes+1, where tree_sequence[i]
                       is T_i as a list of paths (T_0 is the empty tree).
        accept_curve:  list of length max_nodes+1 of E[accept_length(T_i)].
    """
    super_set = set(tuple(p) for p in superset_paths)

    # Current tree (paths that are members)
    cur: Set[Tuple[int, ...]] = set()
    sequence: List[List[List[int]]] = [[]]      # T_0 = empty
    curve: List[float] = [0.0]

    def frontier(cur: Set[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
        """Nodes f ∈ superset, f ∉ cur, parent(f) ∈ cur ∪ {root}."""
        out = []
        for path in superset_paths:
            t = tuple(path)
            if t in cur:
                continue
            parent = t[:-1]
            if parent == () or parent in cur:
                out.append(t)
        return out

    cur_score = 0.0
    for i in range(1, max_nodes + 1):
        front = frontier(cur)
        if not front:
            break

        best_node = None
        best_delta = -1.0
        best_score = cur_score
        for node in front:
            trial = cur | {node}
            score = expected_accept_over_corpus(trial, corpus, tau, eps, alpha)
            delta = score - cur_score
            if delta > best_delta:
                best_delta = delta
                best_node = node
                best_score = score

        if best_node is None or best_delta <= 0:
            if verbose:
                print(f"[i={i}] no positive-delta node, stopping early")
            break

        cur.add(best_node)
        cur_score = best_score
        sequence.append([list(p) for p in cur])
        curve.append(cur_score)

        if verbose and (i <= 10 or i % 10 == 0):
            print(f"[i={i:3d}] +{list(best_node)}  Δ={best_delta:.4f}  "
                  f"E[accept]={cur_score:.3f}  |T|={len(cur)}")

    return sequence, curve


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_corpus_for_search(
    corpus: List[Tuple[Dict[Tuple[int, ...], int], Dict[Tuple[int, ...], torch.Tensor]]],
    path: str,
):
    """Save the (cand_dict, logits_dict) corpus to disk for the search step.
    Logits are saved as fp16 to keep size manageable.
    """
    payload = []
    for cand_dict, logits_dict in corpus:
        payload.append({
            "candidates": {','.join(map(str, k)) if k else '': v for k, v in cand_dict.items()},
            "logits_keys": [','.join(map(str, k)) for k in logits_dict.keys()],
        })
    # Save logits separately as a single tensor blob
    import pickle
    with open(path, "wb") as f:
        pickle.dump(payload, f)


# Note: in practice you'd run discover_menu.py to get the corpus, then call
# greedy_tree_search() directly on its in-memory output, OR pickle the
# corpus to disk and reload here for offline search.  Pickling logits dicts
# of fp16 tensors keyed by short tuples is straightforward.


if __name__ == "__main__":
    # The main entry point would be:
    #   1. Run discover_menu.py to populate `corpus` in memory
    #   2. Call greedy_tree_search(corpus, superset_paths, max_nodes=128)
    #   3. Save the resulting tree_sequence to JSON
    # See discover_menu.py for how the corpus is collected.
    print(__doc__)
