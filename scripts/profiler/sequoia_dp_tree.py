#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Sequoia DP-optimal draft tree vs balanced vs linear (PR #49, CPU-only).

WHAT THIS ANSWERS
-----------------
Sequoia (arXiv 2402.12374, Chen et al. 2024) builds a draft tree that MAXIMISES
the expected number of accepted tokens per speculative step, E[T], by a dynamic
program over a fixed node budget — instead of a fixed *balanced* topology. The
hypothesis (PR #49): a DP-optimal tree gives +3-15% E[T] over a balanced tree on
the measured acceptance distribution, which (verify being ~flat in M, EXPERIMENTS
_LOG #43) maps ~directly into TPS.

The team's prior tree cost-models (#26/#28/#33/#37) all assumed a BALANCED width-4
tree (`scripts/profiler/tree_acceptance_model.py`, parent=(i-1)//W) and locked
K=11/M=45 as the serving optimum (#37). This script asks the question those never
did: **is balanced-W4 actually optimal, or does the Sequoia DP tree beat it?**

THIS IS A LOCAL, CPU-ONLY ANALYTIC STUDY. No GPU, no vLLM, no HF Job. It cannot be
deployed: vLLM 0.22's MTP/EAGLE spec-decode emits a LINEAR chain (the served
`fa2sw_precache_kenyan` is linear MTP K=7, M=8 verify), there is no tree-attention
verify path, and tree-causal masking is a merged dead-end (#33 saves 0 ms). So the
deliverable is the analytic verdict the PR Notes authorise, reusing the measured
acceptance scalars and the measured tile-corrected verify-latency curve.

ACCEPTANCE MODEL (Sequoia path-product)
---------------------------------------
F(T) = sum over nodes v of path_product(v), where path_product(v) = product of
p[rank(u)] over the root->v path (root contributes 1). p[k] = probability the
rank-k sibling token is accepted; p[1] >= p[2] >= ... A single width-W node's
first-level expected-accepted = sum_{k<=W} p[k], so we pin p[k] = C[k]-C[k-1] to
the measured cumulative acceptance (C[1]=top1, C[W]=topW); the rank-2..W split is
a decay model we sweep for robustness (Sequoia claims the optimum is decay-stable).
A linear chain (W=1) then reproduces the team's geometric E exactly with p[1]=top1.

CORRECTNESS: the DP is checked against EXHAUSTIVE brute force (all labelled rooted
trees) for n<=7, and every recovered topology is re-scored by score_tree and
asserted == the DP value, for every budget in the sweep.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os

import numpy as np

NEG = -1e18
DEFAULT_CURVE = "research/spec_cost_model/results_msweep.json"


def _json_default(o):
    """Make numpy scalars/arrays JSON-serializable (fail-safe for dump)."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


# --------------------------------------------------------------------------- #
# Per-rank acceptance vector from measured cumulative top-1 / top-W
# --------------------------------------------------------------------------- #
def derive_per_rank(top1: float, topW: float, W: int, decay: str = "geom") -> np.ndarray:
    """p[1..W] from cumulative C[1]=top1, C[W]=topW. p[0]=0 (1-indexed).

    p[k] = C[k]-C[k-1] are the marginal acceptance gains per added rank; they must
    be non-negative and decreasing. We have only C[1] and C[W], so the rank-2..W
    split uses a decay model:
      geom    : marginals decay geometrically (standard empirical shape)
      uniform : the rescue mass (topW-top1) split evenly over ranks 2..W
      sqrt    : marginals ~ 1/sqrt(rank) (gentler decay; favours wider trees)
    """
    p = np.zeros(W + 1, dtype=np.float64)
    p[1] = top1
    rescue = max(0.0, topW - top1)
    if W == 1 or rescue <= 0:
        return p
    ranks = np.arange(2, W + 1)
    if decay == "uniform":
        w = np.ones(W - 1)
    elif decay == "sqrt":
        w = 1.0 / np.sqrt(ranks - 1.0)
    else:  # geom: fit ratio r so marginals m2..mW decay as m2*r^(k-2)
        # choose r so the shape is concave; r solved to spread `rescue` with a
        # monotone-decreasing profile anchored at a gentle default, then renorm.
        r = 0.45
        w = r ** (ranks - 2.0)
    w = w / w.sum() * rescue
    p[2:] = w
    # enforce strictly non-increasing (numerical safety)
    for k in range(2, W + 1):
        p[k] = min(p[k], p[k - 1])
    return p


# --------------------------------------------------------------------------- #
# Score an arbitrary tree (parent array, node 0 = root) under p
# --------------------------------------------------------------------------- #
def score_tree(parent: list[int], p: np.ndarray) -> tuple[float, int]:
    """F(T)=sum path_product, and max depth. Sibling rank = ascending node id
    (birth order) among nodes sharing a parent. parent[0] = -1 (root)."""
    n = len(parent)
    children: list[list[int]] = [[] for _ in range(n)]
    for i in range(1, n):
        children[parent[i]].append(i)  # ascending i => birth-order rank
    pp = np.zeros(n)
    pp[0] = 1.0
    depth = [0] * n
    F = 1.0  # root
    # BFS in id order keeps parents before children (ids assigned BFS at build)
    for par in range(n):
        for rank, c in enumerate(children[par], start=1):
            r = rank if rank < len(p) else len(p) - 1
            pp[c] = pp[par] * p[r]
            depth[c] = depth[par] + 1
            F += pp[c]
    return float(F), int(max(depth))


def simulate_greedy_E(parent: list[int], p: np.ndarray, trials: int, seed: int) -> float:
    """Monte-Carlo E[committed tokens] under GREEDY tree-spec-decode acceptance.

    This is the ground-truth check that F(=score_tree) really is E[T] for a single-
    pass drafted tree, and is NOT the team's salvage-spine geometric(top-W). Model:
    at an accepted node u the target has ONE argmax; it equals u's rank-k child token
    w.p. p[k] (mutually exclusive across ranks, sum_k p[k] = top-W <= 1), else no
    child matches and the path stops. Committed tokens = accepted path length + 1
    (root/bonus). By linearity of expectation this mean == sum of path-products = F.
    """
    n = len(parent)
    children: list[list[int]] = [[] for _ in range(n)]
    for i in range(1, n):
        children[parent[i]].append(i)
    rng = np.random.default_rng(seed)
    total = 0
    for _ in range(trials):
        u, length = 0, 0
        while children[u]:
            kids = children[u]
            # marginal accept prob per present child rank (rank = birth order)
            probs = np.array([p[r if r < len(p) else len(p) - 1]
                              for r in range(1, len(kids) + 1)], dtype=np.float64)
            u_draw = rng.random()
            cum, chosen = 0.0, -1
            for idx, pr in enumerate(probs):
                cum += pr
                if u_draw < cum:
                    chosen = idx
                    break
            if chosen < 0:  # target argmax not among drafted children -> stop
                break
            u = kids[chosen]
            length += 1
        total += length + 1  # +1 root/bonus token
    return total / trials


def build_linear(n: int) -> list[int]:
    return [-1] + list(range(n - 1))


def build_balanced(n: int, W: int) -> list[int]:
    return [-1] + [(i - 1) // W for i in range(1, n)]


# --------------------------------------------------------------------------- #
# Sequoia DP: maximise F over trees with n nodes, depth<=D, branch<=Bmax
# --------------------------------------------------------------------------- #
def build_sequoia_tree(p: np.ndarray, n: int, max_depth: int, max_branch: int):
    """Returns (parent_list, F, depth). DP per arXiv 2402.12374 Alg. 1.

    T[m][l][b] = max F over a subtree of exactly m nodes (incl. its own root),
    max remaining depth l, and exactly b children at that root. T_max[m][l]=max_b.
    """
    Bmax = min(max_branch, len(p) - 1)
    D = max_depth
    T = np.full((n + 1, D + 1, Bmax + 1), NEG)
    Tmax = np.full((n + 1, D + 1), NEG)
    # backpointer: choice[(m,l,b)] -> ("leaf") | ("b1", child_b*) | ("split", y, child_b*)
    choice: dict = {}

    # A single node (leaf) is feasible at ANY remaining depth, including l=0
    # (the deepest allowed level). Omitting l=0 truncates every chain by one node.
    for l in range(0, D + 1):
        T[1][l][0] = 1.0
        Tmax[1][l] = 1.0
        choice[(1, l, 0)] = ("leaf",)

    for m in range(2, n + 1):
        # m>=2 nodes need >=1 child, so the root must have l>=1 (child at l-1>=0).
        for l in range(1, D + 1):
            best_b, best_v = -1, NEG
            # b = 1: single child (rank 1) gets all m-1 nodes at remaining depth l-1
            if Tmax[m - 1][l - 1] > NEG / 2:
                v = 1.0 + p[1] * Tmax[m - 1][l - 1]
                bc = int(np.argmax(T[m - 1][l - 1]))
                T[m][l][1] = v
                choice[(m, l, 1)] = ("b1", bc)
                if v > best_v:
                    best_b, best_v = 1, v
            # b >= 2: split y to first b-1 children, m-y to the new b-th child
            for b in range(2, Bmax + 1):
                vbest, ybest, cbest = NEG, -1, -1
                for y in range(1, m):
                    left = T[y][l][b - 1]
                    if left <= NEG / 2:
                        continue
                    if Tmax[m - y][l - 1] <= NEG / 2:
                        continue
                    v = left + p[b] * Tmax[m - y][l - 1]
                    if v > vbest:
                        vbest, ybest = v, y
                        cbest = int(np.argmax(T[m - y][l - 1]))
                if ybest > 0:
                    T[m][l][b] = vbest
                    choice[(m, l, b)] = ("split", ybest, cbest)
                    if vbest > best_v:
                        best_b, best_v = b, vbest
            Tmax[m][l] = best_v if best_b >= 0 else NEG

    # pick root branch
    root_b = int(np.argmax(T[n][D]))
    F = float(T[n][D][root_b])
    if F <= NEG / 2:
        # depth-limited infeasible; fall back to linear
        par = build_linear(n)
        f, d = score_tree(par, p)
        return par, f, d

    # ---- recover topology: build children lists, then BFS-assign ids ----
    def collect_children(m, l, b) -> list[tuple[int, int, int]]:
        """Return list of child subtree states (m_c, l_c, b_c) in rank order."""
        ch = choice[(m, l, b)]
        if ch[0] == "leaf":
            return []
        if ch[0] == "b1":
            return [(m - 1, l - 1, ch[1])]
        _, y, cb = ch
        first = collect_children(y, l, b - 1)
        return first + [(m - y, l - 1, cb)]

    parent = [-1]
    # node 0 is root with state (n, D, root_b)
    queue = [(0, n, D, root_b)]
    while queue:
        node_id, m, l, b = queue.pop(0)
        for (mc, lc, bc) in collect_children(m, l, b):
            cid = len(parent)
            parent.append(node_id)
            queue.append((cid, mc, lc, bc))
    f_check, depth = score_tree(parent, p)
    assert abs(f_check - F) < 1e-6, f"recovery mismatch {f_check} vs {F} (n={n})"
    assert len(parent) == n, f"recovered {len(parent)} nodes, expected {n}"
    return parent, F, depth


# --------------------------------------------------------------------------- #
# Brute-force ground truth (small n): exhaustive over labelled rooted trees
# --------------------------------------------------------------------------- #
def brute_force_opt(p: np.ndarray, n: int, max_depth: int, max_branch: int):
    """Max F over ALL trees with n nodes (parent[i] in 0..i-1). Enumerating every
    parent sequence covers every (shape, rank-assignment) because sibling rank =
    birth order, so different orderings realise every labelling. n<=7 only."""
    best_F, best_par = NEG, None
    ranges = [range(i) for i in range(1, n)]  # node i parent in 0..i-1
    for combo in itertools.product(*ranges):
        par = [-1] + list(combo)
        # branch/depth caps
        nchild = [0] * n
        ok = True
        for i in range(1, n):
            nchild[par[i]] += 1
            if nchild[par[i]] > max_branch:
                ok = False
                break
        if not ok:
            continue
        F, d = score_tree(par, p)
        if d > max_depth:
            continue
        if F > best_F:
            best_F, best_par = F, par
    return best_par, best_F


def selfcheck() -> None:
    rng = np.random.default_rng(0)
    pvecs = [
        np.array([0.0, 0.9, 0.7, 0.5, 0.3, 0.15]),
        derive_per_rank(0.6792, 0.8605, 5, "geom"),
        derive_per_rank(0.85, 0.95, 5, "uniform"),
        np.concatenate([[0.0], np.sort(rng.uniform(0, 1, 5))[::-1]]),
    ]
    for p in pvecs:
        for n in range(2, 8):
            for D in (n, 3):
                for B in (2, 4):
                    _, fdp, _ = build_sequoia_tree(p, n, D, B)
                    _, fbf = brute_force_opt(p, n, D, B)
                    assert abs(fdp - fbf) < 1e-6, (
                        f"DP {fdp} != BF {fbf} n={n} D={D} B={B} p={p}")
    # linear closed form
    p = derive_per_rank(0.6792, 0.8605, 4)
    for n in (4, 8, 12):
        F, _ = score_tree(build_linear(n), p)
        geom = sum(p[1] ** i for i in range(n))
        assert abs(F - geom) < 1e-9, f"linear {F} vs geom {geom}"
    # Monte-Carlo: F really IS E[committed tokens] under greedy tree verify (this is
    # what separates the achievable path-product from the salvage-spine upper bound).
    p = derive_per_rank(0.6792, 0.8605, 4, "geom")
    for build, n in ((build_linear, 16), (lambda k: build_balanced(k, 4), 21),
                     (lambda k: build_sequoia_tree(p, k, 24, 4)[0], 16)):
        par = build(n)
        F, _ = score_tree(par, p)
        mc = simulate_greedy_E(par, p, trials=200_000, seed=1)
        assert abs(mc - F) < 0.02, f"MC E[T] {mc:.4f} != path-product F {F:.4f} (n={n})"
    print("[sequoia] selfcheck PASS (DP==bruteforce n<=7; linear==geometric; "
          "MC E[T]==path-product F)", flush=True)


# --------------------------------------------------------------------------- #
# Measured tile-corrected verify-latency curve V(M)
# --------------------------------------------------------------------------- #
class LatencyCurve:
    def __init__(self, path: str, key: str = "graph|ctx256"):
        d = json.load(open(path))
        node = d["cost_model"][key]
        lat = {int(k): float(v) for k, v in node["latency_ms_by_M"].items()}
        self.lat = lat
        self.M = sorted(lat)
        self.mmin, self.mmax = self.M[0], self.M[-1]
        a, b = self.M[-2], self.M[-1]
        self.tail = (lat[b] - lat[a]) / (b - a)

    def at(self, M: float) -> float:
        M = max(1.0, M)
        if M <= self.mmin:
            return self.lat[self.mmin]
        if M >= self.mmax:
            return self.lat[self.mmax] + self.tail * (M - self.mmax)
        lo = max(m for m in self.M if m <= M)
        hi = min(m for m in self.M if m >= M)
        if lo == hi:
            return self.lat[lo]
        t = (M - lo) / (hi - lo)
        return self.lat[lo] * (1 - t) + self.lat[hi] * t

    def in_range(self, M: float) -> bool:
        return self.mmin <= M <= self.mmax


def tps(F: float, M: int, depth: int, curve: LatencyCurve,
        drafter_ms: float, draft_per_depth: float, depth_scaled: bool) -> float:
    V = curve.at(M)
    draft = draft_per_depth * depth if depth_scaled else drafter_ms
    denom = (draft + V) / 1000.0
    return F / denom if denom > 0 else 0.0


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top1", type=float, default=0.6792, help="measured top-1 acc")
    ap.add_argument("--topW", type=float, default=0.8605, help="measured top-W (W=4) acc")
    ap.add_argument("--W", type=int, default=4, help="max measured rank / max branch")
    ap.add_argument("--p-scenarios", type=float, nargs="+", default=[0.6792, 0.78, 0.85],
                    help="top-1 scenarios (topW rescaled by measured rescue ratio)")
    ap.add_argument("--decays", nargs="+", default=["geom", "uniform", "sqrt"])
    ap.add_argument("--budget-max", type=int, default=49, help="max nodes M (=verify positions)")
    ap.add_argument("--max-depth", type=int, default=24)
    ap.add_argument("--max-branch", type=int, default=4)
    ap.add_argument("--drafter-ms", type=float, default=1.446, help="flat per-step drafter ms (#43)")
    ap.add_argument("--draft-per-depth", type=float, default=1.446 / 7,
                    help="depth-scaled per-expansion drafter ms")
    ap.add_argument("--curve", default=DEFAULT_CURVE)
    ap.add_argument("--cost-key", default="graph|ctx256")
    ap.add_argument("--output", default="research/spec_cost_model/sequoia_dp_results.json")
    ap.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT"))
    ap.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--wandb-group", default="sequoia-dp-tree")
    ap.add_argument("--wandb-name", default="sequoia-dp-b7-vs-balanced")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    selfcheck()
    curve = LatencyCurve(args.curve, args.cost_key)
    rescue_ratio = (args.topW - args.top1) / (1.0 - args.top1)  # measured rescue
    print(f"[sequoia] measured top1={args.top1} top{args.W}={args.topW} "
          f"rescue_ratio={rescue_ratio:.4f}; curve {args.cost_key} "
          f"M={curve.mmin}..{curve.mmax}", flush=True)

    budgets = [b for b in range(2, args.budget_max + 1)]
    results = {"config": vars(args), "rescue_ratio": rescue_ratio,
               "scenarios": {}, "budget_sweep": {}}

    headline = {}
    for top1 in args.p_scenarios:
        topW = min(0.999, top1 + rescue_ratio * (1.0 - top1))
        for decay in args.decays:
            p = derive_per_rank(top1, topW, args.W, decay)
            tag = f"p{top1:.4f}|{decay}"
            rows = []
            for n in budgets:
                lin_par = build_linear(n)
                bal_par = build_balanced(n, args.W)
                F_lin, d_lin = score_tree(lin_par, p)
                F_bal, d_bal = score_tree(bal_par, p)
                dp_par, F_dp, d_dp = build_sequoia_tree(p, n, args.max_depth, args.max_branch)
                row = {"n": n, "M": n,
                       "F_linear": F_lin, "F_balanced": F_bal, "F_dp": F_dp,
                       "depth_linear": d_lin, "depth_balanced": d_bal, "depth_dp": d_dp,
                       "dp_over_balanced": F_dp / F_bal if F_bal > 0 else None,
                       "dp_over_linear": F_dp / F_lin if F_lin > 0 else None,
                       "dp_parent": dp_par if n <= 16 else None,
                       "M_in_range": curve.in_range(n)}
                for ds, label in ((False, "flat"), (True, "depthscaled")):
                    row[f"tps_linear_{label}"] = tps(F_lin, n, d_lin, curve, args.drafter_ms, args.draft_per_depth, ds)
                    row[f"tps_balanced_{label}"] = tps(F_bal, n, d_bal, curve, args.drafter_ms, args.draft_per_depth, ds)
                    row[f"tps_dp_{label}"] = tps(F_dp, n, d_dp, curve, args.drafter_ms, args.draft_per_depth, ds)
                rows.append(row)
            results["budget_sweep"][tag] = rows

            def opt(field):
                best = max(rows, key=lambda r: r[field])
                return {"n": best["n"], "M": best["M"], field: best[field],
                        "F_dp": best["F_dp"], "F_balanced": best["F_balanced"],
                        "F_linear": best["F_linear"], "M_in_range": best["M_in_range"]}

            scen = {
                "top1": top1, "topW": topW, "decay": decay,
                "p_vector": [float(x) for x in p],
                "opt_dp_flat": opt("tps_dp_flat"),
                "opt_balanced_flat": opt("tps_balanced_flat"),
                "opt_linear_flat": opt("tps_linear_flat"),
                "opt_dp_depthscaled": opt("tps_dp_depthscaled"),
                "opt_balanced_depthscaled": opt("tps_balanced_depthscaled"),
                "opt_linear_depthscaled": opt("tps_linear_depthscaled"),
            }
            # matched-budget gains at the team's locked M=45 and at deployed M=8.
            # DP-vs-LINEAR is the decision-relevant one: the served stack is linear
            # MTP, so a DP tree only helps if it beats LINEAR, not just balanced-W4.
            for M0 in (8, 45):
                r = next((x for x in rows if x["n"] == M0), None)
                if r:
                    scen[f"ET_gain_dp_vs_balanced_M{M0}"] = r["dp_over_balanced"]
                    scen[f"ET_gain_dp_vs_linear_M{M0}"] = r["dp_over_linear"]
                    scen[f"F_dp_M{M0}"] = r["F_dp"]
                    scen[f"F_balanced_M{M0}"] = r["F_balanced"]
                    scen[f"F_linear_M{M0}"] = r["F_linear"]
            # best E[T] gains across budgets
            gains_b = [r["dp_over_balanced"] for r in rows if r["dp_over_balanced"]]
            gains_l = [r["dp_over_linear"] for r in rows if r["dp_over_linear"]]
            scen["max_ET_gain_dp_vs_balanced"] = max(gains_b) if gains_b else None
            scen["max_ET_gain_dp_vs_linear"] = max(gains_l) if gains_l else None
            scen["tps_gain_dp_vs_balanced_flat"] = (
                scen["opt_dp_flat"]["tps_dp_flat"] / scen["opt_balanced_flat"]["tps_balanced_flat"])
            # Decision-relevant: each topology at its OWN TPS-optimal budget. The
            # served stack is linear, so this is the gain a (hypothetical) DP-tree
            # verifier would buy over the deployed linear MTP.
            scen["tps_gain_dp_vs_linear_flat"] = (
                scen["opt_dp_flat"]["tps_dp_flat"] / scen["opt_linear_flat"]["tps_linear_flat"])
            scen["tps_gain_dp_vs_linear_depthscaled"] = (
                scen["opt_dp_depthscaled"]["tps_dp_depthscaled"]
                / scen["opt_linear_depthscaled"]["tps_linear_depthscaled"])
            # DP-optimal tree at its best budget, F vs the linear chain of the same size
            dp_opt_n = scen["opt_dp_flat"]["n"]
            r_opt = next((x for x in rows if x["n"] == dp_opt_n), None)
            scen["dp_vs_linear_at_dp_opt"] = r_opt["dp_over_linear"] if r_opt else None
            results["scenarios"][tag] = scen
            if decay == "geom":
                headline[f"p{top1:.4f}"] = scen

    # -------- console summary --------
    print("\n[sequoia] ===== E[T] (geom decay): DP-optimal vs balanced-W4 vs linear =====", flush=True)
    print(f"{'top1':>7} {'ET/bal':>8} {'ET/lin':>8} {'DP n*':>6} {'bal n*':>7} {'lin n*':>7} "
          f"{'TPS/bal':>8} {'TPS/lin':>8} {'TPS/lin(ds)':>12}", flush=True)
    for top1 in args.p_scenarios:
        s = headline[f"p{top1:.4f}"]
        print(f"{top1:7.4f} {s['max_ET_gain_dp_vs_balanced']:8.4f} "
              f"{s['max_ET_gain_dp_vs_linear']:8.4f} "
              f"{s['opt_dp_flat']['n']:6d} {s['opt_balanced_flat']['n']:7d} {s['opt_linear_flat']['n']:7d} "
              f"{s['tps_gain_dp_vs_balanced_flat']:8.4f} {s['tps_gain_dp_vs_linear_flat']:8.4f} "
              f"{s['tps_gain_dp_vs_linear_depthscaled']:12.4f}", flush=True)

    # verdict at the canonical operating point (p=0.6792 measured, geom)
    base = headline["p0.6792"]
    paper_band = (1.03, 1.15)
    verdict = {
        "max_ET_gain_dp_vs_balanced": base["max_ET_gain_dp_vs_balanced"],
        "max_ET_gain_dp_vs_linear": base["max_ET_gain_dp_vs_linear"],
        "dp_vs_linear_at_dp_opt": base["dp_vs_linear_at_dp_opt"],
        "ET_gain_at_M45": base.get("ET_gain_dp_vs_balanced_M45"),
        "tps_gain_dp_vs_balanced_flat": base["tps_gain_dp_vs_balanced_flat"],
        "tps_gain_dp_vs_linear_flat": base["tps_gain_dp_vs_linear_flat"],
        "tps_gain_dp_vs_linear_depthscaled": base["tps_gain_dp_vs_linear_depthscaled"],
        "dp_beats_balanced": bool(base["max_ET_gain_dp_vs_balanced"] > 1.001),
        "dp_beats_linear": bool(base["max_ET_gain_dp_vs_linear"] > 1.001),
        "in_paper_3to15pct_band": bool(paper_band[0] <= base["max_ET_gain_dp_vs_balanced"] <= paper_band[1]),
        "dp_opt_n": base["opt_dp_flat"]["n"],
        "balanced_opt_n": base["opt_balanced_flat"]["n"],
        "linear_opt_n": base["opt_linear_flat"]["n"],
    }
    results["verdict"] = verdict
    print(f"\n[sequoia] VERDICT (p=0.6792 measured, geom):", flush=True)
    print(f"  DP vs balanced-W4 : max E[T] gain = {verdict['max_ET_gain_dp_vs_balanced']:.4f} "
          f"({'beats' if verdict['dp_beats_balanced'] else 'ties/loses'}); "
          f"in paper +3-15% band = {verdict['in_paper_3to15pct_band']}", flush=True)
    print(f"  DP vs linear-MTP  : max E[T] gain = {verdict['max_ET_gain_dp_vs_linear']:.4f} "
          f"(matched budget); each@own-opt TPS gain = {verdict['tps_gain_dp_vs_linear_flat']:.4f} "
          f"(flat V) / {verdict['tps_gain_dp_vs_linear_depthscaled']:.4f} (depth-scaled drafter)", flush=True)
    print(f"  budgets: DP n*={verdict['dp_opt_n']}  balanced n*={verdict['balanced_opt_n']}  "
          f"linear n*={verdict['linear_opt_n']}  "
          f"({'beats' if verdict['dp_beats_linear'] else 'ties/loses'} the DEPLOYED topology)", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"[sequoia] wrote {args.output}", flush=True)

    if not args.no_wandb:
        try:
            log_wandb(args, results, headline, verdict, budgets)
        except Exception as e:  # noqa: BLE001
            print(f"[sequoia] W&B logging failed (non-fatal): {e!r}", flush=True)
    print("[sequoia] DONE", flush=True)


def log_wandb(args, results, headline, verdict, budgets):
    import wandb
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type="profiling",
                     config={"top1": args.top1, "topW": args.topW, "W": args.W,
                             "max_branch": args.max_branch, "max_depth": args.max_depth,
                             "drafter_ms": args.drafter_ms, "cost_key": args.cost_key,
                             "p_scenarios": args.p_scenarios, "decays": args.decays})
    summary = {}
    for top1 in args.p_scenarios:
        s = headline[f"p{top1:.4f}"]
        t = f"p{round(top1*1e4):05d}"
        summary[f"{t}_max_ET_gain_dp_vs_balanced"] = s["max_ET_gain_dp_vs_balanced"]
        summary[f"{t}_ET_gain_M8"] = s.get("ET_gain_dp_vs_balanced_M8")
        summary[f"{t}_ET_gain_M45"] = s.get("ET_gain_dp_vs_balanced_M45")
        summary[f"{t}_tps_gain_dp_vs_balanced_flat"] = s["tps_gain_dp_vs_balanced_flat"]
        summary[f"{t}_dp_opt_n"] = s["opt_dp_flat"]["n"]
        summary[f"{t}_balanced_opt_n"] = s["opt_balanced_flat"]["n"]
        summary[f"{t}_tps_dp_opt_flat"] = s["opt_dp_flat"]["tps_dp_flat"]
        summary[f"{t}_tps_balanced_opt_flat"] = s["opt_balanced_flat"]["tps_balanced_flat"]
    summary.update({f"verdict_{k}": v for k, v in verdict.items()})
    run.summary.update({k: v for k, v in summary.items() if v is not None})
    # E[T] and TPS vs budget table for the measured scenario (geom)
    rows = results["budget_sweep"]["p0.6792|geom"]
    cols = ["n", "F_linear", "F_balanced", "F_dp", "dp_over_balanced",
            "depth_dp", "depth_balanced", "tps_dp_flat", "tps_balanced_flat",
            "tps_linear_flat", "M_in_range"]
    tbl = wandb.Table(columns=cols)
    for r in rows:
        tbl.add_data(*[r.get(c) for c in cols])
    run.log({"sequoia_budget_table": tbl})
    run.finish()
    print(f"[sequoia] W&B run: {run.url}", flush=True)


if __name__ == "__main__":
    main()
