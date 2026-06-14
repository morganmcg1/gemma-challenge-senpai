"""OPT-Tree per-step adaptive draft-tree construction (arXiv:2406.17276 §3.2).

Pure-CPU planning routine (no torch, no GPU). Given a per-step *acceptance
surface* -- the conditional probability that the verifier's greedy argmax
matches the drafter's rank-r candidate at a node of depth d -- build the M-node
draft tree that MAXIMISES expected committed tokens

    E[T] = Sum_v pathproduct(v),   pathproduct(v) = Prod_{u on root->v} p_edge(u)

(the exact objective ``scripts/profiler/tree_spec.expected_committed_tokens``
evaluates for a FIXED tree, here maximised over tree SHAPE).

Relation to the deployed verify path
-------------------------------------
The verify kernel and the acceptance CRITERION are byte-for-byte the deployed
ones (``rejected = draft_token_id != target_argmax_id``). OPT-Tree only changes
which candidate nodes are PRESENT in the tree; every token the deployed static
verify would accept is still accepted -> greedy identity is preserved by
construction (proven empirically in ``profile.py`` self-test step 4). The DP is
a planning-time routine that runs at the START of each decode step, before
draft sampling, and emits a ``tree_spec``-compatible ``parent`` array.

Why best-first is the exact DP optimum (no search)
--------------------------------------------------
``pathproduct(child) = pathproduct(parent) * p_edge`` with ``p_edge in [0, 1]``,
so ``pp`` is non-increasing along any root->leaf path. Best-first selection of
the M nodes with the largest ``pp`` is therefore

  (a) prefix-closed: a node's parent has ``pp >= pp(node)`` so the parent is
      always selected before the node -> the selected set is a valid subtree;
  (b) optimal for the additive objective ``Sum_v pp(v)`` (standard exchange
      argument: any unselected frontier node has ``pp`` no larger than every
      selected node, so no swap can increase the sum).

This is exactly the OPT-Tree result: the greedy max-heap expansion is the DP
optimum. Latency is ``O(M * W * log(M * W))`` heap ops -- microseconds for the
deployed budgets (see ``profile.py`` latency self-test).
"""

from __future__ import annotations

import heapq
import sys
from importlib import util as _u
from pathlib import Path
from typing import Callable

_ROOT = Path(__file__).resolve().parents[3]  # research/topology/opttree -> repo root
_spec = _u.spec_from_file_location("tree_spec", _ROOT / "scripts/profiler/tree_spec.py")
tree_spec = _u.module_from_spec(_spec)
sys.modules.setdefault("tree_spec", tree_spec)
_spec.loader.exec_module(tree_spec)

TreeSpec = tree_spec.TreeSpec

# An acceptance surface maps (depth, rank) -> conditional accept probability of
# the rank-r (1-based) candidate landing at `depth` (its parent is at depth-1).
EdgeProb = Callable[[int, int], float]


def build_opt_tree(
    edge_prob: EdgeProb,
    budget: int,
    max_width: int,
    max_depth: int,
) -> tuple[list[int], list[float], float]:
    """Best-first OPT-Tree construction.

    Args:
      edge_prob: ``edge_prob(depth, rank) -> p`` in [0, 1]. Conditional accept
        probability of a parent's rank-``rank`` candidate (1-based) that lands at
        tree depth ``depth``.
      budget: M = total nodes INCLUDING the root (match the deployed verify
        width, e.g. 8).
      max_width: max branching factor per node = the drafter's top-w candidate
        count (only ranks 1..max_width are reachable).
      max_depth: deepest tree depth (the drafter depth budget); root is depth 0.

    Returns ``(parent, pp, e_t)``:
      parent: tree_spec-compatible parent array (``parent[0] == -1``,
        ``parent[i] < i``), children rank-ordered (first child = rank-1 spine).
      pp: per-node path-product (``pp[0] == 1.0``; the always-emitted bonus).
      e_t: ``sum(pp)`` = E[committed tokens/step] of the constructed tree.
    """
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")
    parent: list[int] = [-1]
    depth: list[int] = [0]
    pp: list[float] = [1.0]  # root: the always-emitted bonus token

    # Max-heap via negated pp. Tie-break by (depth, rank) for determinism so the
    # constructed array is reproducible across runs/platforms.
    heap: list[tuple[float, int, int, int, int]] = []  # (-pp, depth, rank, parent_id, seq)
    seq = 0

    def _push_children(node_id: int, node_depth: int, node_pp: float) -> None:
        nonlocal seq
        if node_depth >= max_depth:
            return
        cd = node_depth + 1
        for rank in range(1, max_width + 1):
            pe = edge_prob(cd, rank)
            if pe <= 0.0:
                continue
            cpp = node_pp * pe
            if cpp <= 0.0:
                continue
            heapq.heappush(heap, (-cpp, cd, rank, node_id, seq))
            seq += 1

    _push_children(0, 0, 1.0)
    while len(parent) < budget and heap:
        neg_pp, d, _rank, par, _s = heapq.heappop(heap)
        node_pp = -neg_pp
        node_id = len(parent)
        parent.append(par)
        depth.append(d)
        pp.append(node_pp)
        _push_children(node_id, d, node_pp)

    e_t = sum(pp)
    return parent, pp, e_t


def static_tree_et(tree: "TreeSpec", edge_prob: EdgeProb) -> tuple[float, list[float]]:
    """E[T] of a FIXED ``tree`` under the per-step ``edge_prob`` surface.

    Each node's path-product uses the surface probability for the (depth, rank)
    of every edge on its root->node path -- the same accounting
    ``build_opt_tree`` accumulates, so the two are directly comparable at equal
    budget. Returns ``(e_t, pp)``.
    """
    pp = [0.0] * tree.num_nodes
    pp[0] = 1.0
    total = 1.0
    for i in range(1, tree.num_nodes):
        par = tree.parent[i]
        pe = edge_prob(tree.depth[i], tree.rank_in_parent[i])
        pp[i] = pp[par] * pe
        total += pp[i]
    return total, pp


def ladder_edge_prob(q_by_depth: list[float], rho_by_rank: dict[int, float]) -> EdgeProb:
    """Build an ``edge_prob`` surface from a per-depth rank-1 ladder ``q[d]`` and a
    rank-catch ladder ``rho_k`` (P(target argmax == drafter rank-k | rank-1 miss)).

    edge_prob(d, 1)   = q[d]                          (rank-1 / spine accept)
    edge_prob(d, k>=2) = (1 - q[d]) * rho_k           (rank-k salvage)

    This is the deployed acceptance model (``tree_spec`` measured-p convention:
    p[0]=top1, p[k-1]=(1-top1)*rho_k), generalised to a depth-dependent q[d].
    ``q_by_depth`` is indexed by depth-1 (q_by_depth[0] == q at depth 1).
    """
    max_d = len(q_by_depth)

    def _edge(depth: int, rank: int) -> float:
        if depth < 1 or depth > max_d:
            return 0.0
        q = q_by_depth[depth - 1]
        if rank == 1:
            return q
        rho = rho_by_rank.get(rank, 0.0)
        return (1.0 - q) * rho

    return _edge
