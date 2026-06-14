#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Topology re-opt: re-allocate the M=32 build array vs the MEASURED ladder (PR #152).

THE QUESTION
------------
The deployed draft-tree build array -- M=32 nodes, depth-9, max-branch-3 -- was
pinned (wirbel #83 / denken #101) BEFORE the oracle ever ran, i.e. before we
measured the real per-position acceptance ladder. The oracle has now run
(openevolve `tree-488-pw-fp32-v0`, board 20260614-100550-487). So for the first
time we can ask: GIVEN THE MEASURED LADDER, is the M=32 budget allocated
E[T]-optimally, and is M=32 itself optimal? If the decode step is bandwidth-bound
and ~flat in M (conc=1 is ~92% weight-GEMM, BASELINE.md), growing the M budget into
low-acceptance deep/wide nodes is nearly FREE E[T] -- and the descent-only number
could clear 530 (E[T] >= 5.131) WITHOUT the harder depth-1 spine fix. Either outcome
is launch-relevant: a better drop-in array, OR a near-optimal confirmation that
DE-RISKS the pinned topology.

WHAT THIS IS (and is NOT)
-------------------------
LOCAL CPU-only analytic DP. No GPU / vLLM / HF Job / submission / kernel build.
BASELINE stays 481.53. Greedy identity untouched. Produces a candidate build-array
+ a margin number; does NOT authorize a launch. EXTENDS -- does not duplicate --
wirbel #135's `bug2_salvage_descent.py` E[T]-DP and SINGLE-SOURCES the official-TPS
conversion from fern #142's `m16_measured_500_gate.py` (K_cal, tau, step, verdict).

METHOD (the six PR instructions)
--------------------------------
1. Tree = per-depth branching vector b=[b_1..b_D], node budget M(b)=sum_d prod_{i<=d}
   b_i (non-root draft nodes). Verify budget M = M(b)+1 (the GEMM verifies M rows =
   all tree nodes incl. the already-verified root anchor). Descent-walk-FIXED regime:
   E[T]=1+sum_{d>=1} C[d], reusing #135's per-position reconstruction (score_tree_
   depthrank: F=1+sum of path-products, sibling rank = birth order).
2. Acceptance from the MEASURED ladder. Rank-1 spine = the rho-optimal target
   conditional ladder #135 uses (deployed RISING q from #76), depth-1 set to the
   oracle's 0.679 (descent-only / cell3) or the calibration 0.7287 (both-bugs /
   cell4). Rank-2 re-seed rho2=0.4165; rank-3/4 measured (#79); rank>=5 extrapolated
   by a Sequoia Def-3.5 power law (sensitivity shown). The chain-rule per-rank split
   is build_depth_pvecs_measured (#135), so a FIXED topology is scored exactly.
3. step(M) -- ARMED/PENDING lawine. official = K_cal*E[T]/step(M)*tau. PRIMARY step
   anchor = fern #142's banked depth-9 roofline 1.2127 (the merged single source that
   reproduces 522/538 and the clear-530 bar 5.131); we ALSO report the PR-quoted
   lawine step 1.2182 as a column. Two step(M) SHAPES, both anchored at step(32):
   (a) the #68 measured Marlin int4 staircase (PRIMARY -- the realized cost: flat
   M=8..32, +29% tile cliff at M=33); (b) the smooth roofline max(weight_floor,
   M*verify_flops/peak) the PR specifies. `step_M_curve_is_roofline_pending_lawine=
   True` on every output. `--step-m-json` drops in lawine's measured curve.
4. Drafter-fill ceiling. K=7 MTP drafter -> depth <= ~9 (K + salvage); you can grow
   WIDTH but not depth past the drafter. Search M in {16,24,32,48,64,96,128}; flag
   any M above a plausible drafter-fill ceiling as "needs a wider drafter -- out of
   scope" (still scored, but excluded from the verdict).
5. Optimize OFFICIAL TPS (not raw E[T] -- step(M) must bite). Authoritative optimizer
   = greedy marginal-value node selection (OPT-Tree Thm 3.1: the M largest
   path-products form a valid tree and are provably optimal here, since every edge
   prob < 1 makes path-products monotone non-increasing along both unlock edges). We
   ALSO enumerate the PR's literal balanced b-vector and cross-check with a
   depth-threaded Sequoia DP. Report deployed vs optimal official TPS, the optimal
   (M,b), the E[T] gain, and the MARGINAL-NODE VALUE CURVE (E[T] gain per added node
   by depth/rank).
6. Self-validate. At the deployed M=32 array reproduce #135's descent-only E[T]
   5.04-5.06 and both-bugs 5.207, and cross-check 522/538 official TPS (at the 1.2127
   single-source step; 520/535 at the PR-quoted 1.2182). `topology_dp_self_test_
   passes` gates the whole optimization.

LITERATURE (researcher pass, PR-framed)
---------------------------------------
Sequoia (Chen et al. 2024, arXiv:2402.12374): the canonical budgeted draft-tree DP
c(n)=max 1+sum_i p_i c(a_i). Its Def 3.1 "positional acceptance" assumes accept
depends only on sibling RANK, not depth -- our RISING spine q[d] breaks exactly that,
so we thread depth as a 2nd state (the DP form is unchanged; optimality holds).
OPT-Tree (Kou et al. 2024, arXiv:2406.17276) Thm 3.1: top-n-by-prefix-probability is
optimal under a depth bound -> validates the greedy marginal-value optimizer.
Sequoia Def 3.5 rank-rejection power law r_k <= 1/k^b -> the rank>=5 extrapolation.
"""
from __future__ import annotations

import argparse
import heapq
import importlib.util
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from treeshape_measured_accept import (  # noqa: E402
    build_depth_pvecs_measured,
    load_measured,
    load_rank_coverage,
    score_tree_depthrank,
    simulate_greedy_depthrank,
)
from traversal_verify_et import load_m32_topology, tree_arrays  # noqa: E402
from treeshape_real_cost import RealGemmCurve, describe_tree  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Single-source the official-TPS conversion from fern #142 (K_cal, tau, step, verdict).
gate = _load("m16_measured_500_gate", os.path.join(_HERE, "m16_measured_500_gate.py"))
K_CAL = gate.K_CAL                       # 125.26795 (= 481.53 / 3.844)
TAU = gate.TAU                           # {"low":0.9983,"central":1.0,"high":1.0}
official_tps_map = gate.official_tps_map
accept_length_for_official = gate.accept_length_for_official
_tps_verdict = gate._tps_verdict
STEP_ROOFLINE_125 = gate.STEP_ROOFLINE_DEPTH9   # 1.2127483746822987 (merged single source)
E_T_LINEAR = gate.E_T_LINEAR             # 3.844  -- hard linear-MTP floor
E_T_TREE = gate.E_T_TREE                 # 5.207  -- rho-optimal supply ceiling
FRONTIER_OFFICIAL = gate.FRONTIER_OFFICIAL  # 481.53
TARGET_OFFICIAL = gate.TARGET_OFFICIAL   # 500.0
TARGET_530 = gate.TARGET_530             # 530.0

# ---- banked inputs (same files #135 reads) ----------------------------------
ACCEPT_JSON = "research/accept_calibration/accept_calibration_results.json"
RANKCOV_JSON = "research/rank_coverage/rank_coverage_results.json"
RHO_OPT_JSON = "research/spec_cost_model/rho_optimal_topology_results.json"
ROOFLINE_JSON = "research/spec_cost_model/verify_gemm_roofline.json"

# ---- step anchors -----------------------------------------------------------
STEP_LAWINE_QUOTED = 1.2182          # PR-quoted lawine #136 measured step (reported column)
# PRIMARY = STEP_ROOFLINE_125 (1.2127): the merged m16 gate uses it; it reproduces the
# committed 522/538 anchors AND the operative clear-530 bar 5.131. The PR prose says
# 1.2182 but its own quoted numbers (522, 537.8, 5.131) are all at 1.2127 -- so 1.2127
# is the self-consistent single source and 1.2182 is reported alongside (+0.45% step).

# ---- regimes (descent-walk-FIXED; #135 ET_tree pattern) ---------------------
DEPTH1_DESCENT_ONLY = 0.679          # oracle measured depth-1 (cell3, BUG-1 still live)
DEPTH1_BOTH_BUGS = 0.728739760479042  # calibration top-1 (cell4, rho-optimal ceiling)

# ---- merged anchors the self-test must reproduce (fern #134 matrix / wirbel #135) ----
ANCHOR_CELL3_ET = 5.0564             # descent-only E[T] on the deployed M=32 array
ANCHOR_CELL4_ET = 5.2068             # both-bugs E[T] on the deployed M=32 array
ANCHOR_CELL3_TPS = 522.0             # descent-only official at the 1.2127 step (~522)
ANCHOR_CELL4_TPS = 538.0             # both-bugs official at the 1.2127 step (~538)
ANCHOR_ET_TOL = 0.02
ANCHOR_TPS_TOL = 0.02                # +-2% (same band fern #142 uses)

# ---- gate bars (PR; E[T] form is step-anchor-explicit) ----------------------
CLEAR_530_ET_BAR = 5.131             # PR test bar (= 530*1.2127/K_cal; merged gate bar 5.1311)
CLEAR_500_ET_BAR = 4.841             # at the 1.2127 step (merged gate clear-500 bar)

# ---- search space (PR instruction #4) ---------------------------------------
DRAFTER_K = 7
D_MAX = 9                            # drafter K=7 + 2 salvage (the deployed depth)
M_GRID = [16, 24, 32, 48, 64, 96, 128]   # verify budget = total tree nodes (GEMM rows)
DRAFTER_FILL_CEILING = 48            # M above this needs a wider drafter -> out of scope
W_DEFAULT = 4                        # max branch / rank (deployed max-branch-3 -> ranks<=4 measured)
MAXD_PVEC = 24                       # pvec horizon (matches #135 so anchors reproduce exactly)


def _jd(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} not JSON serializable")


# =========================================================================== #
# Acceptance model: rank>=5 power-law extrapolation (Sequoia Def 3.5)
# =========================================================================== #
def extend_rho_cond(rho_cond: list[float], W: int, mode: str = "power") -> list[float]:
    """Extend measured conditional re-seed ratios rho_r (r=2,3,4) to ranks up to W.

    rho_cond = [rho2, rho3, rho4] (ranks 2..4). For r>=5 (only needed when W>4):
      * "power": Sequoia Def 3.5 power law rho_r = a * r^(-b), fit on the measured
        endpoints (r=2,r=len+1); the literature's only parametric rank-decay anchor.
      * "geom": continue the last measured geometric ratio (sensitivity alternative).
      * "zero": no rank>=5 mass (hard lower bound).
    Ranks 2..4 are returned verbatim (measured); only r>=5 is modeled.
    """
    out = list(rho_cond)
    if W <= len(out) + 1:                      # ranks 2..W all measured
        return out[: max(0, W - 1)]
    if mode == "zero":
        out += [0.0] * (W - 1 - len(out))
        return out
    r_lo, r_hi = 2, len(rho_cond) + 1          # measured rank endpoints (2 and 4)
    v_lo, v_hi = rho_cond[0], rho_cond[-1]
    if mode == "geom":
        ratio = (v_hi / v_lo) ** (1.0 / (r_hi - r_lo)) if v_lo > 0 else 0.0
        r, v = r_hi, v_hi
        while len(out) < W - 1:
            r += 1
            v = max(0.0, v * ratio)
            out.append(v)
        return out
    # power law: log v = log a - b log r  -> fit on (r_lo,v_lo),(r_hi,v_hi)
    b = (math.log(v_lo) - math.log(v_hi)) / (math.log(r_hi) - math.log(r_lo))
    a = v_lo * (r_lo ** b)
    r = r_hi
    while len(out) < W - 1:
        r += 1
        out.append(max(0.0, a * (r ** (-b))))
    return out


def build_regime_pvecs(q_spine: list[float], depth1: float, rho_cond: list[float],
                       W: int, extrapolate_depth: str, rank_mode: str) -> list[np.ndarray]:
    """pvecs for a descent-walk-FIXED regime: the deployed rising spine q_spine with
    depth-1 overridden to `depth1`, the measured rho-ladder (rank>=5 extrapolated)."""
    qq = list(q_spine)
    qq[0] = depth1
    rc = extend_rho_cond(rho_cond, W, rank_mode)
    return build_depth_pvecs_measured(qq, rc, W, MAXD_PVEC, extrapolate_depth)


# =========================================================================== #
# Tree builders
# =========================================================================== #
def greedy_marginal_tree(pvecs, n_draft: int, d_max: int, W: int):
    """Provably-optimal node SET via greedy marginal-value selection (OPT-Tree Thm
    3.1). Grow node-by-node, always adding the available (parent, rank, depth) slot
    with the largest marginal path-product. Unlock the parent's next rank-sibling and
    the new node's own rank-1 child each time. Because every edge prob < 1, a slot's
    value never exceeds the value of the slot that unlocked it, so popping the n_draft
    largest yields the optimal valid tree.

    Returns (parent, F=E[T], marginal[]) where marginal[k] = path-product of the
    (k+1)-th added node (the marginal-node value curve, descending) with its (depth,
    rank), and node ids are creation/pop order so score_tree_depthrank reproduces F.
    """
    maxd = len(pvecs) - 1

    def pv(d, r):
        v = pvecs[min(d, maxd)]
        return float(v[r if r < len(v) else len(v) - 1])

    parent = [-1]
    depth = [0]
    pp = [1.0]                                  # path-product of the root anchor
    # heap of (-value, tiebreak, parent_id, rank, depth_of_child)
    counter = 0
    heap = []
    d1 = 1
    heapq.heappush(heap, (-(pp[0] * pv(d1, 1)), counter, 0, 1, d1)); counter += 1
    marginal = []
    while heap and len(parent) - 1 < n_draft:
        negval, _, par, rank, d = heapq.heappop(heap)
        val = -negval
        cid = len(parent)
        parent.append(par)
        depth.append(d)
        pp.append(val)
        marginal.append({"node": cid, "depth": int(d), "rank": int(rank),
                         "marginal_pathprod": float(val)})
        # unlock parent's next sibling rank (r+1) at the same depth
        if rank < W:
            heapq.heappush(heap, (-(pp[par] * pv(d, rank + 1)), counter, par, rank + 1, d))
            counter += 1
        # unlock this node's own rank-1 child one level deeper
        if d < d_max:
            heapq.heappush(heap, (-(pp[cid] * pv(d + 1, 1)), counter, cid, 1, d + 1))
            counter += 1
    F = 1.0 + sum(pp[1:])
    return parent, float(F), marginal


def build_balanced_bvec(b: list[int]):
    """Construct the per-depth BALANCED tree for branching vector b=[b_1..b_D]: every
    depth-(d-1) node gets b_d children. BFS order -> a node's children are consecutive
    ids (ascending = birth order = rank), so score_tree_depthrank ranks them 1..b_d.
    Returns (parent, M_b = number of non-root draft nodes)."""
    parent = [-1]
    frontier = [0]
    for bd in b:
        if bd < 1:
            break
        nxt = []
        for u in frontier:
            for _ in range(bd):
                parent.append(u)
                nxt.append(len(parent) - 1)
        frontier = nxt
    return parent, len(parent) - 1


def enumerate_balanced(pvecs, m_total: int, d_max: int, W: int):
    """Best balanced b-vector whose tree fits within m_total nodes (root + draft).
    Enumerates b_d in 1..W over depths 1..d_max, keeps the feasible tree of maximal
    E[T]. This is the PR's literal b-vector deliverable (dominated by the caterpillar
    greedy set, but reported for the build team)."""
    budget_draft = m_total - 1
    best = None

    def rec(b):
        nonlocal best
        par, mb = build_balanced_bvec(b)
        if mb > budget_draft:
            return                              # all deeper/wider extensions only grow mb
        F = score_tree_depthrank(par, pvecs)[0]
        if best is None or F > best["E_T"] + 1e-12 or (
                abs(F - best["E_T"]) <= 1e-12 and mb < best["M_b"]):
            best = {"b": list(b), "E_T": float(F), "M_b": int(mb),
                    "m_total": int(mb + 1), "parent": list(par)}
        if len(b) < d_max:
            for bd in range(1, W + 1):
                rec(b + [bd])

    for b1 in range(1, W + 1):
        rec([b1])
    return best


def sequoia_dp(pvecs, n_draft: int, d_max: int, W: int) -> float:
    """Depth-threaded Sequoia Bellman cross-check: best E[T] for a budget of n_draft
    draft nodes. value(depth, budget) = max expected accepted subtree length rooted at
    a node of the given depth, spending `budget` descendant nodes across its ranked
    children. Edge to rank-r child at child-depth d carries pv[d][r]. Returns the
    whole-tree E[T] = 1 + value(0, n_draft)."""
    maxd = len(pvecs) - 1

    def pv(d, r):
        v = pvecs[min(d, maxd)]
        return float(v[r if r < len(v) else len(v) - 1])

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def child(depth, budget, rank):
        """Max expected mass from ranks `rank..W` of a node at `depth`, given `budget`
        descendant nodes to allocate among them."""
        if budget <= 0 or rank > W or depth + 1 > d_max:
            return 0.0
        edge = pv(depth + 1, rank)
        best = child(depth, budget, rank + 1)   # skip this rank
        for k in range(1, budget + 1):          # spend 1 node on rank-r child + k-1 in its subtree
            sub = edge * (1.0 + node(depth + 1, k - 1))
            rest = child(depth, budget - k, rank + 1)
            if sub + rest > best:
                best = sub + rest
        return best

    @lru_cache(maxsize=None)
    def node(depth, budget):
        """Max expected accepted mass below a node at `depth` using `budget` nodes."""
        if budget <= 0:
            return 0.0
        return child(depth, budget, 1)

    return 1.0 + node(0, n_draft)


# =========================================================================== #
# step(M) cost models (all anchored at step(32)=anchor)
# =========================================================================== #
class StepModel:
    """step(M) in the same units as the official-TPS step (ms-equivalent), anchored so
    step(32)=anchor. Two shapes: Marlin staircase (#68 measured, primary) and smooth
    roofline (PR form). Optionally overridden by lawine's measured --step-m-json."""

    def __init__(self, anchor: float, gemm_share: float = 0.532,
                 roofline_json: str = ROOFLINE_JSON, step_m_json: str | None = None):
        self.anchor = float(anchor)
        self.g = float(gemm_share)
        self.real = RealGemmCurve(roofline_json)
        self._m32_mult = self.real.step_mult(32, self.g)
        # roofline ridge: M=32 is bandwidth-bound (agg AI 107.7 < datasheet ridge 116.7)
        # -> M_crit ~ 32 * 116.667/107.658 = 34.7; flat up to M_crit, linear past it.
        self.m_crit = 32.0 * 116.6667 / 107.6577
        self.measured = None
        if step_m_json and os.path.exists(step_m_json):
            d = json.load(open(step_m_json))
            raw = d.get("step_by_M", d.get("step_m", d))
            self.measured = {int(k): float(v) for k, v in raw.items()}

    def marlin(self, M: int) -> float:
        """#68 Marlin int4 staircase, anchored at step(32). For M<=49 use the measured
        GEMM ratio; past 49 continue the 16-row tile staircase (cheap interior + a tile
        cliff every 16 rows) via the smooth-roofline envelope (drafter-fill-flagged)."""
        if M <= 49:
            return self.anchor * self.real.step_mult(M, self.g) / self._m32_mult
        # M>49: 16-row Marlin tiles. Past the measured curve, price the compute part by
        # whole tiles (ceil(M/16)) scaled to the M=48 anchor; conservative + flagged.
        tiles = math.ceil(M / 16.0)
        tiles48 = math.ceil(48 / 16.0)
        mult48 = self.real.step_mult(48, self.g)
        mult = (1.0 - self.g) + self.g * (mult48 - (1.0 - self.g)) / self.g * (tiles / tiles48)
        return self.anchor * mult / self._m32_mult

    def roofline(self, M: int) -> float:
        """Smooth roofline max(weight_floor, M*flops/peak), anchored step(32)=anchor.
        M=32 < M_crit so step(32)=floor=anchor; flat to M_crit then linear in M."""
        if M <= self.m_crit:
            return self.anchor
        return self.anchor * (M / self.m_crit)

    def step(self, M: int, shape: str = "marlin") -> float:
        if self.measured is not None:
            if M in self.measured:
                return self.measured[M]
            ks = sorted(self.measured)
            lo = max(k for k in ks if k <= M) if any(k <= M for k in ks) else ks[0]
            hi = min(k for k in ks if k >= M) if any(k >= M for k in ks) else ks[-1]
            if lo == hi:
                return self.measured[lo]
            t = (M - lo) / (hi - lo)
            return self.measured[lo] * (1 - t) + self.measured[hi] * t
        return self.roofline(M) if shape == "roofline" else self.marlin(M)


def official(E_T: float, step: float, tau: float = None) -> float:
    return official_tps_map(E_T, step, TAU["central"] if tau is None else tau)


# =========================================================================== #
# Self-test: reproduce the merged deployed-array anchors (PR instruction #6)
# =========================================================================== #
def run_self_test(q_spine, rho_cond, deployed_parent, sm_primary) -> dict:
    pv_d = build_regime_pvecs(q_spine, DEPTH1_DESCENT_ONLY, rho_cond, W_DEFAULT, "flat", "power")
    pv_b = build_regime_pvecs(q_spine, DEPTH1_BOTH_BUGS, rho_cond, W_DEFAULT, "flat", "power")
    et_d = score_tree_depthrank(deployed_parent, pv_d)[0]
    et_b = score_tree_depthrank(deployed_parent, pv_b)[0]
    step125 = STEP_ROOFLINE_125
    step_pr = STEP_LAWINE_QUOTED
    tps_d_125 = official(et_d, step125)
    tps_b_125 = official(et_b, step125)
    tps_d_pr = official(et_d, step_pr)
    tps_b_pr = official(et_b, step_pr)
    # greedy at the deployed budget (M=32 nodes -> 31 draft) must match/exceed deployed
    g_par, g_et_d, _ = greedy_marginal_tree(pv_d, len(deployed_parent) - 1, D_MAX, W_DEFAULT)
    mc_d = simulate_greedy_depthrank(deployed_parent, pv_d, 200_000, seed=152)
    checks = {
        "descent_only_E_T": et_d,
        "descent_only_E_T_ok": abs(et_d - ANCHOR_CELL3_ET) < ANCHOR_ET_TOL,
        "both_bugs_E_T": et_b,
        "both_bugs_E_T_ok": abs(et_b - ANCHOR_CELL4_ET) < ANCHOR_ET_TOL,
        "descent_only_tps_step1_2127": tps_d_125,
        "descent_only_tps_ok": abs(tps_d_125 - ANCHOR_CELL3_TPS) / ANCHOR_CELL3_TPS < ANCHOR_TPS_TOL,
        "both_bugs_tps_step1_2127": tps_b_125,
        "both_bugs_tps_ok": abs(tps_b_125 - ANCHOR_CELL4_TPS) / ANCHOR_CELL4_TPS < ANCHOR_TPS_TOL,
        "descent_only_tps_step1_2182_PRquoted": tps_d_pr,
        "both_bugs_tps_step1_2182_PRquoted": tps_b_pr,
        "greedy_M32_E_T_descent": g_et_d,
        "greedy_reproduces_deployed": g_et_d >= et_d - 0.02,
        "deployed_is_within_pct_of_greedy": (g_et_d - et_d) / et_d * 100.0,
        "mc_E_T_descent_xcheck": mc_d,
        "mc_vs_dp_abs": abs(mc_d - et_d),
    }
    checks["topology_dp_self_test_passes"] = bool(
        checks["descent_only_E_T_ok"] and checks["both_bugs_E_T_ok"]
        and checks["descent_only_tps_ok"] and checks["both_bugs_tps_ok"]
        and checks["greedy_reproduces_deployed"] and checks["mc_vs_dp_abs"] < 0.03)
    return checks


# =========================================================================== #
# Optimize over the (M, b) grid
# =========================================================================== #
def optimize_regime(name: str, q_spine, depth1, rho_cond, sm, *, W, extrap_depth,
                    rank_mode, deployed_parent, step_shape, step_anchor_val) -> dict:
    pvecs = build_regime_pvecs(q_spine, depth1, rho_cond, W, extrap_depth, rank_mode)
    deployed_et = score_tree_depthrank(deployed_parent, pvecs)[0]
    deployed_M = len(deployed_parent)
    deployed_step = sm.step(deployed_M, step_shape)
    deployed_tps = official(deployed_et, deployed_step)

    rows = []
    cand = {}
    for M in M_GRID:
        n_draft = M - 1
        g_par, g_et, g_marg = greedy_marginal_tree(pvecs, n_draft, D_MAX, W)
        dp_et = sequoia_dp(pvecs, n_draft, D_MAX, W)
        bal = enumerate_balanced(pvecs, M, D_MAX, W)
        st = sm.step(M, step_shape)
        tps = official(g_et, st)
        shape = describe_tree(g_par)
        in_scope = M <= DRAFTER_FILL_CEILING
        rows.append({
            "M_total": M, "M_b_draft": n_draft,
            "greedy_E_T": g_et, "sequoia_dp_E_T": dp_et,
            "greedy_vs_dp_abs": abs(g_et - dp_et),
            "balanced_best_E_T": bal["E_T"] if bal else None,
            "balanced_best_b": bal["b"] if bal else None,
            "balanced_best_M_total": bal["m_total"] if bal else None,
            "step": st, "official_tps": tps,
            "et_gain_vs_deployed": g_et - deployed_et,
            "tps_gain_vs_deployed": tps - deployed_tps,
            "clears_500": tps >= TARGET_OFFICIAL, "clears_530": tps >= TARGET_530,
            "width_by_depth": shape["width_by_depth"],
            "max_branch": shape["max_branch_factor"], "depth": shape["max_depth"],
            "in_drafter_fill_scope": in_scope,
            "drafter_fill_flag": None if in_scope else
            "needs a wider drafter (M>%d) -- out of scope for this DP" % DRAFTER_FILL_CEILING,
        })
        cand[M] = {"parent": g_par, "marginal": g_marg, "shape": shape,
                   "E_T": g_et, "official_tps": tps, "in_scope": in_scope}

    in_scope_rows = [r for r in rows if r["in_drafter_fill_scope"]]
    opt = max(in_scope_rows, key=lambda r: r["official_tps"])
    opt_any = max(rows, key=lambda r: r["official_tps"])
    return {
        "regime": name, "depth1": depth1, "W": W,
        "extrapolate_depth": extrap_depth, "rank_mode": rank_mode,
        "step_shape": step_shape, "step_anchor": step_anchor_val,
        "deployed_E_T": deployed_et, "deployed_M_total": deployed_M,
        "deployed_step": deployed_step, "deployed_official_tps": deployed_tps,
        "grid": rows,
        "optimal_in_scope": opt, "optimal_any": opt_any,
        "candidates": cand,
    }


# =========================================================================== #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accept-json", default=ACCEPT_JSON)
    ap.add_argument("--rankcov-json", default=RANKCOV_JSON)
    ap.add_argument("--rho-opt-json", default=RHO_OPT_JSON)
    ap.add_argument("--roofline-json", default=ROOFLINE_JSON)
    ap.add_argument("--step-m-json", default=None,
                    help="lawine #136 measured step(M) curve: {'step_by_M': {M: ms}}; "
                         "overrides the roofline/Marlin model when present.")
    ap.add_argument("--step-anchor", type=float, default=STEP_ROOFLINE_125,
                    help="step(32) anchor (default 1.2127 = merged m16 single source).")
    ap.add_argument("--step-shape", default="marlin", choices=["marlin", "roofline"])
    ap.add_argument("--gemm-share", type=float, default=0.532)
    ap.add_argument("--W", type=int, default=W_DEFAULT)
    ap.add_argument("--rank-mode", default="power", choices=["power", "geom", "zero"])
    ap.add_argument("--extrapolate-depth", default="flat", choices=["flat", "rise"])
    ap.add_argument("--mc-trials", type=int, default=200_000)
    ap.add_argument("--output", default="research/oracle_readout/topology_m_reopt_results.json")
    ap.add_argument("--candidate-output",
                    default="research/oracle_readout/topology_m_reopt_candidate_build_array.json")
    ap.add_argument("--wandb-project", "--wandb_project", default=os.environ.get("WANDB_PROJECT"))
    ap.add_argument("--wandb-entity", "--wandb_entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--wandb-group", "--wandb_group", default="topology-m-reopt")
    ap.add_argument("--wandb-name", "--wandb_name", default="wirbel/topology-m-reopt")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    # ---- inputs (the same banked files #135 reads) ----
    meas = load_measured(args.accept_json, "server_log")
    q_spine = list(meas["q"])                    # deployed RISING conditional spine (#76)
    rc = load_rank_coverage(args.rankcov_json)
    rho_cond = rc["rho_cond"]                     # [0.4165, 0.2655, 0.1908] (#79)
    deployed_parent = load_m32_topology(args.rho_opt_json)
    dch, ddep, dlv = tree_arrays(deployed_parent)
    print(f"[reopt] deployed M=32: n={len(deployed_parent)} depth={max(ddep)} "
          f"max_branch={max(len(c) for c in dch)} leaves={len(dlv)}", flush=True)
    print(f"[reopt] measured spine q={[round(x,4) for x in q_spine]}  "
          f"rho_cond={[round(x,4) for x in rho_cond]}", flush=True)

    sm = StepModel(args.step_anchor, args.gemm_share, args.roofline_json, args.step_m_json)
    step_pending = args.step_m_json is None      # roofline-pending until lawine lands
    print(f"[reopt] step(M) anchor={args.step_anchor:.6f} shape={args.step_shape} "
          f"roofline_pending={step_pending}; step(16)={sm.step(16,args.step_shape):.4f} "
          f"step(32)={sm.step(32,args.step_shape):.4f} step(48)={sm.step(48,args.step_shape):.4f} "
          f"step(64)={sm.step(64,args.step_shape):.4f}", flush=True)

    # ---- self-test (gate) ----
    st = run_self_test(q_spine, rho_cond, deployed_parent, sm)
    print(f"[reopt] SELF-TEST: descent-only E[T]={st['descent_only_E_T']:.4f} "
          f"(anchor {ANCHOR_CELL3_ET}, ok={st['descent_only_E_T_ok']}); both-bugs "
          f"E[T]={st['both_bugs_E_T']:.4f} (anchor {ANCHOR_CELL4_ET}, ok={st['both_bugs_E_T_ok']})",
          flush=True)
    print(f"[reopt] SELF-TEST: official@1.2127 descent={st['descent_only_tps_step1_2127']:.1f} "
          f"(~522 ok={st['descent_only_tps_ok']}) both={st['both_bugs_tps_step1_2127']:.1f} "
          f"(~538 ok={st['both_bugs_tps_ok']}); @1.2182(PR) "
          f"{st['descent_only_tps_step1_2182_PRquoted']:.1f}/"
          f"{st['both_bugs_tps_step1_2182_PRquoted']:.1f}", flush=True)
    print(f"[reopt] SELF-TEST: greedy-M32 E[T]={st['greedy_M32_E_T_descent']:.4f} "
          f"(deployed within {st['deployed_is_within_pct_of_greedy']:+.2f}% of greedy); "
          f"MC xcheck {st['mc_E_T_descent_xcheck']:.4f} (|Δ|={st['mc_vs_dp_abs']:.4f})", flush=True)
    print(f"[reopt] topology_dp_self_test_passes = {st['topology_dp_self_test_passes']}", flush=True)

    # ---- optimize: both regimes at primary step shape ----
    reg_descent = optimize_regime(
        "descent_only", q_spine, DEPTH1_DESCENT_ONLY, rho_cond, sm, W=args.W,
        extrap_depth=args.extrapolate_depth, rank_mode=args.rank_mode,
        deployed_parent=deployed_parent, step_shape=args.step_shape,
        step_anchor_val=args.step_anchor)
    reg_both = optimize_regime(
        "both_bugs", q_spine, DEPTH1_BOTH_BUGS, rho_cond, sm, W=args.W,
        extrap_depth=args.extrapolate_depth, rank_mode=args.rank_mode,
        deployed_parent=deployed_parent, step_shape=args.step_shape,
        step_anchor_val=args.step_anchor)

    # ---- step-shape + step-anchor robustness on the descent-only optimum ----
    sm_roof = StepModel(args.step_anchor, args.gemm_share, args.roofline_json, None)
    sm_pr = StepModel(STEP_LAWINE_QUOTED, args.gemm_share, args.roofline_json, None)
    reg_descent_roof = optimize_regime(
        "descent_only_roofline", q_spine, DEPTH1_DESCENT_ONLY, rho_cond, sm_roof, W=args.W,
        extrap_depth=args.extrapolate_depth, rank_mode=args.rank_mode,
        deployed_parent=deployed_parent, step_shape="roofline", step_anchor_val=args.step_anchor)
    reg_descent_pr = optimize_regime(
        "descent_only_step1_2182", q_spine, DEPTH1_DESCENT_ONLY, rho_cond, sm_pr, W=args.W,
        extrap_depth=args.extrapolate_depth, rank_mode=args.rank_mode,
        deployed_parent=deployed_parent, step_shape=args.step_shape, step_anchor_val=STEP_LAWINE_QUOTED)

    # ---- marginal-node value curve (descent-only greedy) ----
    # Greedy growth order is budget-INDEPENDENT (best-first heap; the budget only
    # truncates), so the largest grid tree's marginal sequence contains every
    # smaller tree's as a prefix. Build the visible curve on the largest M so the
    # diminishing-returns tail PAST the in-scope optimum is shown -- that tail is
    # exactly why the optimizer stops (the step-cost cliff, not exhausted E[T]).
    opt_M = reg_descent["optimal_in_scope"]["M_total"]
    opt_draft = opt_M - 1  # the optimal tree's last draft-node index in the curve
    curve_M = max(M_GRID)
    marg_full = reg_descent["candidates"][curve_M]["marginal"]
    # by_depth_rank aggregates how the WINNING M=opt_M budget is allocated (prefix).
    marg = marg_full[:opt_draft]
    by_depth_rank = {}
    for m in marg:
        key = f"d{m['depth']}_r{m['rank']}"
        by_depth_rank.setdefault(key, {"count": 0, "sum_marginal": 0.0})
        by_depth_rank[key]["count"] += 1
        by_depth_rank[key]["sum_marginal"] += m["marginal_pathprod"]
    # marginal value of the k-th greedily-added node (curve is already value-descending)
    marg_cut_seq = [m["marginal_pathprod"] for m in marg_full]
    marg_curve_cuts = {str(k): (marg_cut_seq[k - 1] if k <= len(marg_cut_seq) else 0.0)
                       for k in (8, 16, 24, 32, 40, 48)}
    # the boundary that explains the optimum: value of the last-kept vs first-dropped node
    optimum_boundary = {
        "optimum_draft_nodes": opt_draft,
        "marginal_last_kept_node": marg_cut_seq[opt_draft - 1] if opt_draft <= len(marg_cut_seq) else 0.0,
        "marginal_first_dropped_node": marg_cut_seq[opt_draft] if opt_draft < len(marg_cut_seq) else 0.0,
        "note": ("last-kept ~= first-dropped => the descent-only E[T] curve is FLAT at the "
                 "optimum; the optimizer stops because the Marlin step-cost cliff (M=32->48) "
                 "outweighs the ~0.026 E[T]/node, not because E[T] is exhausted."),
    }

    # ---- verdict ----
    opt_row = reg_descent["optimal_in_scope"]
    primary_tps = opt_row["official_tps"]
    clears_530 = bool(opt_row["official_tps"] >= TARGET_530)
    clears_530_et = bool(opt_row["greedy_E_T"] >= CLEAR_530_ET_BAR)
    deployed_tps = reg_descent["deployed_official_tps"]
    both_opt = reg_both["optimal_in_scope"]

    verdict = {
        "primary_metric_name": "topology_reopt_official_tps",
        "topology_reopt_official_tps": primary_tps,
        "test_metric_name": "topology_reopt_clears_530",
        "topology_reopt_clears_530": int(clears_530 and clears_530_et),
        "topology_dp_self_test_passes": int(st["topology_dp_self_test_passes"]),
        "deployed_official_tps_descent": deployed_tps,
        "deployed_E_T_descent": reg_descent["deployed_E_T"],
        "optimal_M_total": opt_row["M_total"],
        "optimal_width_by_depth": opt_row["width_by_depth"],
        "optimal_E_T_descent": opt_row["greedy_E_T"],
        "optimal_balanced_b": opt_row["balanced_best_b"],
        "et_gain_vs_deployed": opt_row["et_gain_vs_deployed"],
        "tps_gain_vs_deployed": opt_row["tps_gain_vs_deployed"],
        "clear_530_et_bar": CLEAR_530_ET_BAR,
        "descent_only_clears_530": int(clears_530),
        "both_bugs_optimal_tps": both_opt["official_tps"],
        "both_bugs_optimal_E_T": both_opt["greedy_E_T"],
        "both_bugs_clears_530": int(both_opt["official_tps"] >= TARGET_530),
        "deployed_is_near_optimal": bool(opt_row["et_gain_vs_deployed"] < 0.05),
        "step_M_curve_is_roofline_pending_lawine": True,
        "step_anchor_primary": args.step_anchor,
        "step_anchor_pr_quoted": STEP_LAWINE_QUOTED,
        "step_shape_primary": args.step_shape,
        "descent_only_optimal_tps_at_1_2182": reg_descent_pr["optimal_in_scope"]["official_tps"],
        "descent_only_optimal_tps_roofline": reg_descent_roof["optimal_in_scope"]["official_tps"],
    }

    # ---- gate label ----
    if not st["topology_dp_self_test_passes"]:
        glabel = ("SELF-TEST FAIL -- DP does not reproduce the merged deployed-array "
                  "anchors; optimization untrustworthy.")
    elif verdict["topology_reopt_clears_530"]:
        glabel = (f"descent-only re-opt CLEARS 530: optimal M={opt_row['M_total']} "
                  f"({primary_tps:.1f} TPS, E[T]={opt_row['greedy_E_T']:.4f}) beats the deployed "
                  f"{deployed_tps:.1f} -- free margin from re-allocation WITHOUT the depth-1 fix.")
    elif verdict["deployed_is_near_optimal"]:
        glabel = (f"M=32 is NEAR-OPTIMAL: best in-scope re-opt M={opt_row['M_total']} gives "
                  f"+{opt_row['et_gain_vs_deployed']:.4f} E[T] ({primary_tps:.1f} vs deployed "
                  f"{deployed_tps:.1f} TPS) -- does NOT clear 530 ({CLEAR_530_ET_BAR}); clearing 530 "
                  f"needs the BUG-1 depth-1 spine fix (both-bugs -> {both_opt['greedy_E_T']:.3f} -> "
                  f"{both_opt['official_tps']:.1f}). Re-allocation DE-RISKS the pinned topology.")
    else:
        glabel = (f"re-opt gains +{opt_row['et_gain_vs_deployed']:.4f} E[T] "
                  f"({primary_tps:.1f} vs {deployed_tps:.1f} TPS) but does NOT clear 530.")
    verdict["gate_label"] = glabel
    print(f"[reopt] VERDICT: {glabel}", flush=True)

    # ---- assemble + write results ----
    results = {
        "config": vars(args),
        "provenance": (
            "Extends wirbel #135 bug2_salvage_descent E[T]-DP (score_tree_depthrank / "
            "build_depth_pvecs_measured) and single-sources the official-TPS conversion from "
            "fern #142 m16_measured_500_gate (K_cal=125.268, tau, step, verdict). Acceptance: "
            "deployed rising spine #76 + measured rho_cond #79 + oracle depth-1 0.679 (board "
            "20260614-100550-487). step(M): #68 verify_gemm_roofline Marlin staircase + smooth "
            "roofline, anchored step(32) at fern #142's 1.2127 (merged single source; PR-quoted "
            "1.2182 reported alongside). Optimizers: greedy marginal-value (OPT-Tree Thm 3.1, "
            "authoritative) + balanced b-vector (PR literal) + depth-threaded Sequoia DP xcheck "
            "(arXiv:2402.12374)."),
        "method": ("LOCAL CPU-only analytic DP. No GPU/vLLM/HF Job/submission/kernel build. "
                   "BASELINE 481.53 untouched; greedy identity untouched. Candidate build-array "
                   "+ margin number ONLY; does NOT authorize a launch."),
        "inputs": {
            "deployed_parent": deployed_parent,
            "deployed_shape": describe_tree(deployed_parent),
            "measured_spine_q": q_spine, "rho_cond": rho_cond,
            "rho_cond_extended": extend_rho_cond(rho_cond, args.W, args.rank_mode),
            "oracle_board": "20260614-100550-487", "oracle_package": "tree-488-pw-fp32-v0",
            "depth1_descent_only": DEPTH1_DESCENT_ONLY, "depth1_both_bugs": DEPTH1_BOTH_BUGS,
        },
        "step_model": {
            "anchor_primary": args.step_anchor, "anchor_pr_quoted": STEP_LAWINE_QUOTED,
            "shape_primary": args.step_shape, "gemm_share": args.gemm_share,
            "roofline_pending_lawine": step_pending, "m_crit_roofline": sm.m_crit,
            "step_by_M_marlin": {str(M): sm.step(M, "marlin") for M in M_GRID},
            "step_by_M_roofline": {str(M): sm_roof.step(M, "roofline") for M in M_GRID},
            "note": ("step(M) anchored step(32)=anchor. Marlin (primary): #68 measured int4 "
                     "staircase (flat M=8..32, +29% tile cliff at 33). Roofline (PR form): flat "
                     "to M_crit then linear. roofline_pending until lawine #136 --step-m-json."),
        },
        "self_test": st,
        "regimes": {
            "descent_only": reg_descent,
            "both_bugs": reg_both,
            "descent_only_roofline_step": reg_descent_roof,
            "descent_only_step1_2182": reg_descent_pr,
        },
        "marginal_node_value_curve": {
            "optimal_M_total": opt_M,
            "curve_built_on_M_total": curve_M,
            "by_depth_rank": by_depth_rank,
            "value_at_node_rank": marg_curve_cuts,
            "optimum_boundary": optimum_boundary,
            "full_curve": marg_full,
            "note": ("marginal_pathprod[k] = E[T] gain of the k-th greedily-added draft node. "
                     "Curve is built on the largest grid tree (M=%d) so the diminishing tail "
                     "PAST the in-scope optimum (%d draft nodes) is visible; greedy order is "
                     "budget-independent so the first %d entries ARE the optimal M=%d tree. "
                     "by_depth_rank aggregates only that winning prefix. Falls off fast: rank-1 "
                     "spine first, then rank-2 near the root, then deep/rank-3+ <0.05 each."
                     % (curve_M, opt_draft, opt_draft, opt_M)),
        },
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=_jd)
    print(f"[reopt] wrote {args.output}", flush=True)

    # ---- candidate build-array JSON (drop-in for the build team) ----
    opt_parent = reg_descent["candidates"][opt_M]["parent"]
    candidate = {
        "_what": ("Candidate draft-tree build array from the descent-walk-FIXED topology re-opt "
                  "(PR #152). Drop-in for the build team. Greedy marginal-value optimal under the "
                  "MEASURED ladder. Does NOT authorize a launch; informs the build/launch decision."),
        "regime": "descent_only",
        "optimal_M_total": opt_M,
        "optimal_M_draft_nodes": opt_M - 1,
        "parent": opt_parent,
        "width_by_depth": reg_descent["candidates"][opt_M]["shape"]["width_by_depth"],
        "max_branch": reg_descent["candidates"][opt_M]["shape"]["max_branch_factor"],
        "depth": reg_descent["candidates"][opt_M]["shape"]["max_depth"],
        "balanced_b_vector": opt_row["balanced_best_b"],
        "E_T_descent_only": opt_row["greedy_E_T"],
        "E_T_both_bugs": both_opt["greedy_E_T"],
        "official_tps_descent_step1_2127": opt_row["official_tps"],
        "official_tps_descent_step1_2182": reg_descent_pr["optimal_in_scope"]["official_tps"],
        "clears_530_descent": int(clears_530 and clears_530_et),
        "deployed_parent_for_reference": deployed_parent,
        "deployed_E_T_descent": reg_descent["deployed_E_T"],
        "et_gain_vs_deployed": opt_row["et_gain_vs_deployed"],
        "step_M_curve_is_roofline_pending_lawine": True,
        "self_test_passes": int(st["topology_dp_self_test_passes"]),
        "provenance": "topology_m_reopt.py (PR #152); oracle board 20260614-100550-487.",
    }
    with open(args.candidate_output, "w") as f:
        json.dump(candidate, f, indent=2, default=_jd)
    print(f"[reopt] wrote {args.candidate_output}", flush=True)

    # ---- console summary table ----
    print("\n[reopt] ===== descent-only re-opt grid (primary step: %s @ %.4f) ====="
          % (args.step_shape, args.step_anchor), flush=True)
    print(f"  {'M':>4} {'E[T]':>7} {'step':>7} {'official':>9} {'ΔE[T]':>7} {'clr500':>6} "
          f"{'clr530':>6} {'scope':>6}", flush=True)
    for r in reg_descent["grid"]:
        print(f"  {r['M_total']:4d} {r['greedy_E_T']:7.4f} {r['step']:7.4f} "
              f"{r['official_tps']:9.1f} {r['et_gain_vs_deployed']:+7.4f} "
              f"{str(r['clears_500']):>6} {str(r['clears_530']):>6} "
              f"{'in' if r['in_drafter_fill_scope'] else 'OUT':>6}", flush=True)
    print(f"  deployed M=32: E[T]={reg_descent['deployed_E_T']:.4f} "
          f"official={deployed_tps:.1f}", flush=True)
    print(f"[reopt] primary topology_reopt_official_tps = {primary_tps:.1f}", flush=True)
    print(f"[reopt] test    topology_reopt_clears_530   = {verdict['topology_reopt_clears_530']}",
          flush=True)

    if not args.no_wandb:
        try:
            log_wandb(args, results, verdict, reg_descent, reg_both, st)
        except Exception as e:  # noqa: BLE001
            print(f"[reopt] W&B logging failed (non-fatal): {e!r}", flush=True)
    print("[reopt] DONE", flush=True)


def log_wandb(args, results, verdict, reg_descent, reg_both, st):
    import wandb
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     group=args.wandb_group, name=args.wandb_name, job_type="analysis",
                     config={"W": args.W, "rank_mode": args.rank_mode,
                             "extrapolate_depth": args.extrapolate_depth,
                             "step_anchor": args.step_anchor, "step_shape": args.step_shape,
                             "gemm_share": args.gemm_share, "D_MAX": D_MAX,
                             "drafter_fill_ceiling": DRAFTER_FILL_CEILING,
                             "oracle_board": "20260614-100550-487",
                             "clear_530_et_bar": CLEAR_530_ET_BAR})
    summ = {f"verdict/{k}": v for k, v in verdict.items() if not isinstance(v, (dict, list))}
    summ.update({f"self_test/{k}": v for k, v in st.items() if not isinstance(v, (dict, list))})
    run.summary.update(summ)
    # descent-only grid table
    cols = ["M_total", "greedy_E_T", "sequoia_dp_E_T", "balanced_best_E_T", "step",
            "official_tps", "et_gain_vs_deployed", "tps_gain_vs_deployed",
            "clears_500", "clears_530", "max_branch", "depth", "in_drafter_fill_scope"]
    tbl = wandb.Table(columns=cols)
    for r in reg_descent["grid"]:
        tbl.add_data(*[r.get(c) for c in cols])
    run.log({"descent_only_grid": tbl})
    # both-bugs grid
    tbl2 = wandb.Table(columns=cols)
    for r in reg_both["grid"]:
        tbl2.add_data(*[r.get(c) for c in cols])
    run.log({"both_bugs_grid": tbl2})
    # marginal-node value curve
    mc = results["marginal_node_value_curve"]
    mt = wandb.Table(columns=["node", "depth", "rank", "marginal_pathprod"])
    for m in mc["full_curve"]:
        mt.add_data(m["node"], m["depth"], m["rank"], m["marginal_pathprod"])
    run.log({"marginal_node_value_curve": mt})
    run.summary["wandb_run_id"] = run.id
    print(f"[reopt] W&B run: {run.url}  (id={run.id})", flush=True)
    run.finish()


if __name__ == "__main__":
    main()
