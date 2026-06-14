#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Re-OPTIMISE the M=32/M=16 draft-tree TOPOLOGY under the MEASURED rho ladder (PR #83).

WHAT THIS ANSWERS
-----------------
PR #74 built the M=32/M=16 parent arrays (the topology land #71 serves) with a
Sequoia DP run over a BORROWED, FLAT cumulative rescue rho=0.565 (EAGLE-3). PR #79
then MEASURED the real per-rank rescue ladder on our own deployed stack and it is
steeply DECLINING:

    rho2 = 0.4165  >>  rho3 = 0.2655  >  rho4 = 0.1908     (34.7% hard-miss beyond top-4)

A declining ladder changes the DP-optimal branch ALLOCATION: rank-2 breadth is worth
more, and deep rank-4 branches worth less, than the flat 0.565 assumed. This script
re-runs the tree optimisation with the measured ladder and asks:

  (1) What is the measured-rho-OPTIMAL M=32/M=16 topology (parent arrays)?
  (2) How much extra E[T]/TPS does re-optimising buy over just deploying #74's shape?
      (decision: hand land #71 new arrays, or confirm #74 is robust within the
      wall_tps MDE and bank #74 as-is.)
  (3) The per-spine-position EXPECTED salvage / branch-hit ORACLE -- land #71's
      debug-gate target (a correct width-2 branch should salvage ~rho2~0.41 at a
      divergence; byteshark's broken tree-v2 read 3.3%).
  (4) Re-price on three bases (relative / local wall_tps x454.1 / official x481.53)
      and confirm M=32 max-branch-4 stays optimal under the measured ladder.

THE DEPTH-DEPENDENT DP (the one new piece of machinery)
-------------------------------------------------------
#74's `build_sequoia_tree` optimises a POSITION-INDEPENDENT per-rank vector p[r].
The measured model is DEPTH-DEPENDENT: the rank-1 spine acceptance q[d] RISES with
depth (0.729 -> 0.847, #76) and the rank>=2 rescue follows the measured chain-rule
ladder applied to the residual (1-q[d]):

    pv[d][1] = q[d]
    pv[d][r] = prod_{j<r}(1-rho_j) * rho_r * (1-q[d])      (r = 2..W)   [#79 measured]

so the edge weight of a rank-r child depends on the child's ABSOLUTE depth d. We
generalise the Sequoia DP to index the cost-to-go table by absolute depth:

    Tmax[m][d] = max E[T]-contribution (root counted as 1) of an m-node subtree whose
                 ROOT sits at absolute depth d; an edge to a rank-r child at depth d+1
                 carries weight pv[d+1][r].
    G[m][d][b] = same, root constrained to exactly b children.

The recursion is identical to #74's Alg-1 DP with the single substitution
p[b] -> pv[d+1][b]; because all children of a node share the SAME depth d+1, the
within-sibling per-rank vector pv[d+1][.] is monotone non-increasing in rank, so the
rearrangement/exchange argument that proves the Sequoia DP optimal still holds. We
VALIDATE this empirically the same way #49/#74 did: DP == EXHAUSTIVE brute force over
all labelled rooted trees for n<=7 (incl. depth-varying pv), and DP == the original
position-independent `build_sequoia_tree` when pv is held depth-constant.

LOCAL, CPU-ONLY, ANALYTIC. No GPU, no vLLM, no HF Job, no submission, no served-file
change. Reuses #49/#76/#79 machinery verbatim; the only borrowed input (rho) is now
the MEASURED ladder, so nothing here is borrowed.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sequoia_dp_tree import (  # noqa: E402
    build_linear,
    build_sequoia_tree,
    derive_per_rank,
    score_tree,
)
from treeshape_measured_accept import (  # noqa: E402
    TOPO_74,
    build_depth_pvecs_measured,
    load_measured,
    load_rank_coverage,
    score_tree_depthrank,
    simulate_greedy_depthrank,
)
from treeshape_real_cost import RealGemmCurve  # noqa: E402

# Robust anchors (lawine #72): the OLD "428.37 local steady" was a fragile estimator.
WALL_TPS_LINEAR_M8 = 454.1        # robust local wall_tps anchor, fa2sw_precache_kenyan (#52)
OFFICIAL_TPS_LINEAR_M8 = 481.53   # official a10g leaderboard anchor (#52, PRIVATE-VERIFIED)
LEGACY_LOCAL_STEADY = 428.37      # #74/#76 anchor, kept only for back-compat reporting
HARD_M_CAP = 32                   # #68 M=33 Marlin tile cliff (+53%)
GEMM_SHARE_DEFAULT = 0.532
ROOFLINE = "research/spec_cost_model/verify_gemm_roofline.json"
ACCEPT_JSON = "research/accept_calibration/accept_calibration_results.json"
RANKCOV_JSON = "research/rank_coverage/rank_coverage_results.json"

NEG = -1e18


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} not JSON serializable")


# --------------------------------------------------------------------------- #
# Depth-dependent Sequoia DP (the one new piece; #74's DP indexed by abs depth)
# --------------------------------------------------------------------------- #
def build_depth_dp(pvecs: list[np.ndarray], n: int, max_depth: int, max_branch: int):
    """Returns (parent, F, depth): the E[T]-optimal n-node tree under the
    DEPTH-DEPENDENT per-rank model pvecs[d][r] (abs depth d, sibling rank r).

    Generalises sequoia_dp_tree.build_sequoia_tree by indexing the cost-to-go on
    absolute depth so the rising-spine q[d] and the per-rank rescue ladder both enter.
    """
    maxd_pv = len(pvecs) - 1

    def pv(d: int, r: int) -> float:
        v = pvecs[min(d, maxd_pv)]
        rr = r if r < len(v) else len(v) - 1
        return float(v[rr])

    Bmax = min(max_branch, max(len(v) - 1 for v in pvecs))
    D = max_depth
    G = np.full((n + 1, D + 1, Bmax + 1), NEG)
    Tmax = np.full((n + 1, D + 1), NEG)
    choice: dict = {}

    # A single node (leaf) is feasible at ANY absolute depth d in 0..D.
    for d in range(0, D + 1):
        G[1][d][0] = 1.0
        Tmax[1][d] = 1.0
        choice[(1, d, 0)] = ("leaf",)

    for m in range(2, n + 1):
        # root at depth d needs >=1 child at depth d+1 <= D  => d <= D-1
        for d in range(0, D):
            best_b, best_v = -1, NEG
            # b = 1: single rank-1 child gets all m-1 nodes at depth d+1
            if Tmax[m - 1][d + 1] > NEG / 2:
                v = 1.0 + pv(d + 1, 1) * Tmax[m - 1][d + 1]
                bc = int(np.argmax(G[m - 1][d + 1]))
                G[m][d][1] = v
                choice[(m, d, 1)] = ("b1", bc)
                if v > best_v:
                    best_b, best_v = 1, v
            # b >= 2: y nodes -> first b-1 children (same root), m-y -> new rank-b child
            for b in range(2, Bmax + 1):
                vbest, ybest, cbest = NEG, -1, -1
                for y in range(1, m):
                    left = G[y][d][b - 1]
                    if left <= NEG / 2:
                        continue
                    if Tmax[m - y][d + 1] <= NEG / 2:
                        continue
                    v = left + pv(d + 1, b) * Tmax[m - y][d + 1]
                    if v > vbest:
                        vbest, ybest = v, y
                        cbest = int(np.argmax(G[m - y][d + 1]))
                if ybest > 0:
                    G[m][d][b] = vbest
                    choice[(m, d, b)] = ("split", ybest, cbest)
                    if vbest > best_v:
                        best_b, best_v = b, vbest
            Tmax[m][d] = best_v if best_b >= 0 else NEG

    root_b = int(np.argmax(G[n][0]))
    F = float(G[n][0][root_b])
    if F <= NEG / 2:  # depth-limited infeasible -> linear fallback
        par = build_linear(n)
        f, d = score_tree_depthrank(par, pvecs)
        return par, f, d

    def collect_children(m, d, b):
        ch = choice[(m, d, b)]
        if ch[0] == "leaf":
            return []
        if ch[0] == "b1":
            return [(m - 1, d + 1, ch[1])]
        _, y, cb = ch
        first = collect_children(y, d, b - 1)
        return first + [(m - y, d + 1, cb)]

    parent = [-1]
    queue = [(0, n, 0, root_b)]
    while queue:
        node_id, m, d, b = queue.pop(0)
        for (mc, dc, bc) in collect_children(m, d, b):
            cid = len(parent)
            parent.append(node_id)
            queue.append((cid, mc, dc, bc))
    f_check, depth = score_tree_depthrank(parent, pvecs)
    assert abs(f_check - F) < 1e-6, f"recovery mismatch {f_check} vs {F} (n={n})"
    assert len(parent) == n, f"recovered {len(parent)} nodes, expected {n}"
    return parent, F, depth


def brute_force_depthrank(pvecs, n, max_depth, max_branch):
    """Max F over ALL labelled rooted n-node trees under the depth-dependent model."""
    best_F, best_par = NEG, None
    ranges = [range(i) for i in range(1, n)]
    for combo in itertools.product(*ranges):
        par = [-1] + list(combo)
        nchild = [0] * n
        ok = True
        for i in range(1, n):
            nchild[par[i]] += 1
            if nchild[par[i]] > max_branch:
                ok = False
                break
        if not ok:
            continue
        F, d = score_tree_depthrank(par, pvecs)
        if d > max_depth:
            continue
        if F > best_F:
            best_F, best_par = F, par
    return best_par, best_F


def selfcheck_depth_dp() -> None:
    """(1) depth-DP == brute force for n<=7 (incl. depth-varying pv);
    (2) depth-DP == original position-independent DP when pv is depth-constant."""
    rng = np.random.default_rng(0)

    # (1) brute-force equivalence on depth-VARYING per-rank vectors
    def const_pv(p, maxd):
        return [np.zeros_like(p)] + [p.copy() for _ in range(maxd)]

    def varying_pv(maxd, W):
        # rank-1 rises with depth, rank>=2 follows a declining residual ladder
        pvs = [np.zeros(W + 1)]
        for d in range(1, maxd + 1):
            qd = min(0.95, 0.70 + 0.02 * d)
            rho = [0.42, 0.27, 0.19, 0.12][: W - 1]
            pv = np.zeros(W + 1)
            pv[1] = qd
            surv, miss = 1.0, 1.0 - qd
            for r in range(2, W + 1):
                rr = rho[r - 2] if (r - 2) < len(rho) else 0.0
                pv[r] = surv * rr * miss
                surv *= (1.0 - rr)
            pvs.append(pv)
        return pvs

    pv_sets = [
        const_pv(np.array([0.0, 0.9, 0.7, 0.5, 0.3]), 8),
        const_pv(derive_per_rank(0.6792, 0.8605, 4, "geom"), 8),
        varying_pv(8, 4),
        varying_pv(8, 3),
        [np.zeros(5)] + [np.concatenate([[0.0], np.sort(rng.uniform(0, 1, 4))[::-1]])
                         for _ in range(8)],
    ]
    for pvs in pv_sets:
        for n in range(2, 8):
            for D in (n, 3):
                for B in (2, 4):
                    _, fdp, _ = build_depth_dp(pvs, n, D, B)
                    _, fbf = brute_force_depthrank(pvs, n, D, B)
                    assert abs(fdp - fbf) < 1e-6, f"depth-DP {fdp} != BF {fbf} n={n} D={D} B={B}"

    # (2) reduce to the original DP under a depth-CONSTANT vector
    for top1, topW, W, decay in [(0.6792, 0.8605, 4, "geom"), (0.729, 0.90, 4, "uniform")]:
        p = derive_per_rank(top1, topW, W, decay)
        pvs = const_pv(p, 30)
        for n in (8, 16, 24, 32):
            par_o, f_o, _ = build_sequoia_tree(p, n, 24, W)
            par_d, f_d, _ = build_depth_dp(pvs, n, 24, W)
            assert abs(f_o - f_d) < 1e-9, f"depth-DP {f_d} != orig DP {f_o} (n={n})"
    print("[rho-opt] selfcheck PASS (depth-DP == brute force n<=7; == orig DP when "
          "depth-constant)", flush=True)


# --------------------------------------------------------------------------- #
# Topology description helpers
# --------------------------------------------------------------------------- #
def describe_topology(parent: list[int]) -> dict:
    n = len(parent)
    children = [[] for _ in range(n)]
    depth = [0] * n
    for i in range(1, n):
        children[parent[i]].append(i)
        depth[i] = depth[parent[i]] + 1
    width_by_depth: dict[int, int] = {}
    for i in range(n):
        d = depth[i]
        width_by_depth[d] = width_by_depth.get(d, 0) + 1
    branch_pts = sum(1 for c in children if len(c) >= 2)
    max_branch = max((len(c) for c in children), default=0)
    # spine = rank-1 chain from root (first child at each step)
    spine = [0]
    u = 0
    while children[u]:
        u = children[u][0]
        spine.append(u)
    # branch width AT each spine position k>=1 = number of children of spine node k-1
    spine_branch_width = []
    for k in range(1, len(spine)):
        par_node = spine[k - 1]
        spine_branch_width.append(len(children[par_node]))
    return {
        "parent": list(parent),
        "n": n,
        "depth": max(depth),
        "branch_points_rank2plus": branch_pts,
        "max_branch": max_branch,
        "width_by_depth": {int(k): int(v) for k, v in sorted(width_by_depth.items())},
        "spine_len": len(spine),
        "spine_branch_width": spine_branch_width,  # width at spine depth 1,2,3,...
    }


def cov_present(width: int, rho_cond: list[float], rho2_depth: float | None = None) -> float:
    """Cumulative rank>=2 coverage given miss-1 for a branch of the given width.

    width=1 -> 0 (no salvage branch); width=2 -> rho2; width>=3 adds chain-rule mass.
    rho2_depth (if given) overrides the pooled rho2 with the per-depth measured value.
    """
    if width <= 1:
        return 0.0
    r2 = rho2_depth if rho2_depth is not None else rho_cond[0]
    cov = r2
    surv = 1.0 - r2
    for r in range(3, width + 1):
        idx = r - 2
        if idx >= len(rho_cond):
            break
        cov += surv * rho_cond[idx]
        surv *= (1.0 - rho_cond[idx])
    return cov


def depth_swept_optimum(pvecs, M, max_depth, max_branch, price_fn):
    """Drafter-aware TPS-best tree for a node budget M and branch cap.

    The #68 M-only cost is degenerate under the rising q[d]: it makes a deep rank-1 spine
    ~free, so the unconstrained DP balloons the spine. The MTP drafter actually costs
    `depth` sequential weight-re-reading passes (#69/#77), so we sweep the depth cap, take
    the E[T]-optimal tree at each depth, and return the one with the best drafter-aware TPS.
    """
    sweep, best = [], None
    for Dcap in range(3, min(M - 1, max_depth) + 1):
        par, F, dep = build_depth_dp(pvecs, M, Dcap, max_branch)
        pr = price_fn(F, M, dep)
        row = {"depth_cap": Dcap, "depth": dep, "F_tree": F,
               "gain_drafter_aware": pr["gain"],
               "gain_m_only": price_fn(F, M, dep, gd=0.0)["gain"], "parent": par}
        sweep.append(row)
        if best is None or pr["gain"] > best["gain_drafter_aware"]:
            best = row
    return best, sweep


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accept-json", default=ACCEPT_JSON)
    ap.add_argument("--accept-source", default="server_log",
                    choices=["server_log", "prometheus"])
    ap.add_argument("--rank-coverage-json", default=RANKCOV_JSON)
    ap.add_argument("--roofline", default=ROOFLINE)
    ap.add_argument("--W", type=int, default=4)
    ap.add_argument("--max-branch", type=int, default=4)
    ap.add_argument("--max-depth", type=int, default=24)
    ap.add_argument("--budgets", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--gemm-share", type=float, default=GEMM_SHARE_DEFAULT)
    ap.add_argument("--gemm-share-sweep", type=float, nargs="+", default=[0.42, 0.532, 0.60, 0.70])
    # drafter-depth cost (#69/#77): the MTP drafter runs `depth` SEQUENTIAL weight-
    # re-reading passes, so its cost scales with tree DEPTH, not node budget M. At the
    # deployed depth-7 chain the drafter is 15.5-18.1% of the 11.6 ms step (central 0.168).
    # g-drafter=0 recovers the PR-literal #68 M-only cost (the optimistic upper bound).
    ap.add_argument("--g-drafter", type=float, default=0.168,
                    help="drafter share of decode step at base depth (0 => #68 M-only)")
    ap.add_argument("--g-drafter-sweep", type=float, nargs="+", default=[0.0, 0.10, 0.155, 0.168, 0.181, 0.25])
    ap.add_argument("--base-drafter-depth", type=int, default=7, help="deployed K=7 chain depth")
    ap.add_argument("--extrapolate", default="flat", choices=["flat", "rise"])
    ap.add_argument("--mc-trials", type=int, default=400_000)
    ap.add_argument("--rho5-beyond", type=float, default=0.19,
                    help="optimistic rho5 upper bound for the beyond-width-4 check "
                         "(declining ladder => true rho5 <= rho4=0.19)")
    ap.add_argument("--output",
                    default="research/spec_cost_model/rho_optimal_topology_results.json")
    ap.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT"))
    ap.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--wandb-group", default="rho-optimal-topology")
    ap.add_argument("--wandb-name", default="wirbel/rho-optimal-topology")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    selfcheck_depth_dp()

    meas = load_measured(args.accept_json, args.accept_source)
    rank_cov = load_rank_coverage(args.rank_coverage_json)
    real = RealGemmCurve(args.roofline)
    g = args.gemm_share
    g_d = args.g_drafter
    base_d = args.base_drafter_depth
    rho_cond = rank_cov["rho_cond"]          # [rho2, rho3, rho4] measured (#79)
    q = meas["q"]                            # depth-dependent conditional acceptance (#76)

    # per-depth rho2 (the only per-depth ladder element measured #79); pooled rho3/rho4
    rc_raw = json.load(open(args.rank_coverage_json))["analysis"]
    rho2_by_depth = {int(k): float(v) for k, v in rc_raw.get("rho2_by_depth", {}).items()}
    per_depth_div = rc_raw.get("per_depth_div_count", [])
    rank_fd_hist = {int(k): int(v) for k, v in rc_raw.get("rank_fd_hist_pooled", {}).items()}
    n_div = int(rc_raw.get("n_divergences", sum(rank_fd_hist.values())))

    print(f"[rho-opt] measured ladder (#79): rho_cond={rho_cond}  "
          f"cov4={rank_cov['cov_W']:.4f}  top1={rank_cov['top1_measured']:.4f}  "
          f"hard_miss_beyond_top4={rank_cov['frac_true_beyond_topW']:.4f}", flush=True)
    print(f"[rho-opt] depth q (#76): {[round(x,4) for x in q]}  E[T]={meas['E_T']:.4f}", flush=True)
    print(f"[rho-opt] cost: g_verify(M)={g} g_drafter(depth)={g_d} base_depth={base_d}", flush=True)

    # ---- measured depth-dependent per-rank vectors (the DP/scoring substrate) ----
    pvecs = build_depth_pvecs_measured(q, rho_cond, args.W, args.max_depth, args.extrapolate)

    # consistency anchor: F_linear(8) under measured pv == measured E[T]
    F_lin8, _ = score_tree_depthrank(build_linear(8), pvecs)
    anchor_gap = abs(F_lin8 - meas["E_T"])
    print(f"[rho-opt] ANCHOR F_linear(8)={F_lin8:.5f} vs measured E[T]={meas['E_T']:.4f} "
          f"(gap {anchor_gap:.4f} = log rounding)", flush=True)
    assert anchor_gap < 5e-3, "F_linear(8) departs from measured E[T] beyond log precision"

    def step_mult(M: int, depth: int, gv: float = None, gd: float = None) -> float:
        """Decode-step multiplier vs the deployed (M=8, depth=7) chain.

        verify weight-GEMM scales with node budget M (#68); the MTP drafter scales with
        tree DEPTH (#69/#77: `depth` sequential weight-re-reading passes). gd=0 => the
        PR-literal #68 M-only cost.
        """
        gv = g if gv is None else gv
        gd = g_d if gd is None else gd
        return (1.0 - gv - gd) + gv * real.ratio(M) + gd * (depth / base_d)

    def price(F_tree: float, M: int, depth: int, gv: float = None, gd: float = None) -> dict:
        cB = step_mult(M, depth, gv, gd)
        et_ratio = F_tree / F_lin8
        gain = et_ratio / cB - 1.0
        return {
            "F_tree": F_tree, "M": M, "depth": depth, "cost_mult": cB,
            "ET_ratio_vs_lin8": et_ratio, "gain": gain,
            "proj_wall_tps": WALL_TPS_LINEAR_M8 * (1.0 + gain),
            "proj_official_tps": OFFICIAL_TPS_LINEAR_M8 * (1.0 + gain),
            "proj_legacy_local": LEGACY_LOCAL_STEADY * (1.0 + gain),
        }

    # ---- (1)(2) measured-rho-OPTIMAL topology vs #74, with the drafter-DEPTH cost ----
    per_budget = {}
    for M in args.budgets:
        # M-only (PR-literal) optimum: unconstrained DP (deep-spine cost-model artifact)
        par_monly, F_monly, d_monly = build_depth_dp(pvecs, M, args.max_depth, args.max_branch)
        # drafter-aware TPS-optimal topology (depth-swept)
        best, depth_sweep = depth_swept_optimum(pvecs, M, args.max_depth, args.max_branch, price)
        opt_par = best["parent"]
        F_opt = best["F_tree"]
        d_opt = best["depth"]
        F_74, d_74 = score_tree_depthrank(TOPO_74[M], pvecs)
        opt_price = price(F_opt, M, d_opt)
        p74_price = price(F_74, M, d_74)
        monly_price = price(F_monly, M, d_monly)            # drafter-aware price of the M-only optimum
        monly_price_monlycost = price(F_monly, M, d_monly, gd=0.0)
        p74_price_monlycost = price(F_74, M, d_74, gd=0.0)
        mc = simulate_greedy_depthrank(opt_par, pvecs, args.mc_trials, seed=7)
        per_budget[M] = {
            "optimal": {**describe_topology(opt_par), "F_tree": F_opt,
                        "mc_E_T": mc, "mc_abs_err": abs(mc - F_opt), "pricing": opt_price},
            "m_only_optimum": {**describe_topology(par_monly), "F_tree": F_monly,
                               "pricing_drafter_aware": monly_price,
                               "pricing_m_only": monly_price_monlycost},
            "topo74": {**describe_topology(TOPO_74[M]), "F_tree": F_74,
                       "pricing": p74_price, "pricing_m_only": p74_price_monlycost},
            "depth_sweep": depth_sweep,
            "reopt_ET_gain_over_74": F_opt / F_74 - 1.0,
            "reopt_tps_gain_over_74": opt_price["gain"] - p74_price["gain"],
            "reopt_wall_tps_delta": opt_price["proj_wall_tps"] - p74_price["proj_wall_tps"],
            "reopt_tps_gain_over_74_m_only": monly_price_monlycost["gain"] - p74_price_monlycost["gain"],
            "same_topology_as_74": opt_par == TOPO_74[M],
        }
        print(f"[rho-opt] M={M}: drafter-aware opt depth={d_opt} F={F_opt:.4f} (MC {mc:.4f}) "
              f"gain {opt_price['gain']*100:+.2f}% | #74 (depth {d_74}) gain {p74_price['gain']*100:+.2f}% "
              f"-> re-opt {per_budget[M]['reopt_tps_gain_over_74']*100:+.2f}pp", flush=True)
        print(f"[rho-opt]   M-only optimum: depth={d_monly} F={F_monly:.4f} "
              f"-> M-only gain {monly_price_monlycost['gain']*100:+.2f}% but drafter-aware "
              f"only {monly_price['gain']*100:+.2f}% (deep-spine artifact)", flush=True)
        print(f"[rho-opt]   drafter-aware optimal parent (M={M}): {opt_par}", flush=True)

    M32, M16 = per_budget.get(32), per_budget.get(16)
    m32_dominates_m16 = (M32["optimal"]["pricing"]["proj_wall_tps"]
                         > M16["optimal"]["pricing"]["proj_wall_tps"]) if (M32 and M16) else None

    # ---- (3) per-position salvage / branch-hit ORACLE on the chosen M=32 topology ----
    chosen = M32 if M32 else per_budget[max(per_budget)]
    chosen_M = chosen["optimal"]["n"]
    spine_widths = chosen["optimal"]["spine_branch_width"]
    # empirical first-divergence weights per spine depth (#79 per_depth_div_count)
    oracle_rows = []
    pooled_num_full, pooled_num_rank2, pooled_den = 0.0, 0.0, 0.0
    for k in range(1, len(spine_widths) + 1):
        width = spine_widths[k - 1]
        r2d = rho2_by_depth.get(k - 1)                 # per-depth rho2 (0-indexed)
        w_div = per_depth_div[k - 1] if (k - 1) < len(per_depth_div) else 0
        salv_full = cov_present(width, rho_cond, r2d)  # actual branch width salvage
        salv_rank2 = (r2d if r2d is not None else rho_cond[0]) if width >= 2 else 0.0
        oracle_rows.append({
            "spine_position": k,
            "branch_width": width,
            "q_spine": q[k - 1] if (k - 1) < len(q) else q[-1],
            "rho2_at_pos": r2d,
            "expected_salvage_full_width": salv_full,
            "expected_salvage_rank2_only": salv_rank2,
            "first_div_weight": w_div,
        })
        pooled_num_full += w_div * salv_full
        pooled_num_rank2 += w_div * salv_rank2
        pooled_den += w_div
    # positions beyond the measured horizon (k>7) contribute 0 div weight (rarely reached)
    pooled_full = pooled_num_full / pooled_den if pooled_den else 0.0
    pooled_rank2 = pooled_num_rank2 / pooled_den if pooled_den else 0.0
    # universal rank-2 gate (independent of topology): rho2 pooled over all divergences
    rank2_hit_global = rank_fd_hist.get(2, 0) / n_div if n_div else rho_cond[0]
    cov4_global = sum(rank_fd_hist.get(r, 0) for r in (2, 3, 4)) / n_div if n_div else rank_cov["cov_W"]
    salvage_oracle = {
        "chosen_topology_M": chosen_M,
        "per_position": oracle_rows,
        "pooled_full_width_salvage": pooled_full,
        "pooled_rank2_branch_hit_salvage": pooled_rank2,
        "rank2_branch_hit_global": rank2_hit_global,     # == rho2 = land #71's gate ~0.41
        "cov4_full_salvage_global": cov4_global,         # all-4-ranks ceiling ~0.65
        "byteshark_broken_tree_v2_read": 0.033,
        "gate_note": ("land #71 salvage-sanity gate: a width-2 branch at a first "
                      "divergence must hit ~rho2~0.417 (NOT byteshark's broken 3.3%); "
                      "a full width-4 branch should rescue ~0.653 of divergences."),
    }
    print(f"[rho-opt] SALVAGE ORACLE (M={chosen_M}): pooled full-width "
          f"{pooled_full:.4f} | pooled rank-2 {pooled_rank2:.4f} | global rank-2 hit "
          f"{rank2_hit_global:.4f} (gate) | cov4 {cov4_global:.4f}", flush=True)

    # ---- (4) width / branch-factor optimality under the measured ladder ----
    # 4a: is full width-4 justified? Compare the BEST drafter-aware tree achievable under
    #     each branch cap (each depth-swept the same way as the headline optimum).
    width_used = {}
    for M in args.budgets:
        for B in (2, 3, 4):
            best_B, _ = depth_swept_optimum(pvecs, M, args.max_depth, B, price)
            width_used[f"M{M}_branch{B}"] = {
                "F": best_B["F_tree"], "depth": best_B["depth"], "max_branch_cap": B,
                "gain": best_B["gain_drafter_aware"]}
    # 4b: beyond width-4 -- extend the ladder with an OPTIMISTIC rho5 (<= rho4) and a
    #     declining rho6; if the DP still never uses rank-5, width-4 is optimal.
    rho_ext = list(rho_cond) + [args.rho5_beyond, args.rho5_beyond * (rho_cond[2] / rho_cond[1])]
    pvecs_w6 = build_depth_pvecs_measured(q, rho_ext, 6, args.max_depth, args.extrapolate)
    beyond4 = {}
    for M in args.budgets:
        par6, F6, _ = build_depth_dp(pvecs_w6, M, args.max_depth, 6)
        desc6 = describe_topology(par6)
        uses_rank5plus = desc6["max_branch"] >= 5
        # marginal value of the single best rank-5 leaf vs the least-valuable node
        # actually placed in the width-4 optimal M tree
        opt_par = per_budget[M]["optimal"]["parent"]
        pp = _path_products(opt_par, pvecs)
        least_node_pp = min(pp[1:]) if len(pp) > 1 else 0.0
        # best rank-5 leaf marginal = max over depth d of pv6[d][5] (parent pp ~ along spine)
        best_rank5_leaf = 0.0
        # rank-5 child hangs at depth d off a node whose own path-product is pp_parent;
        # the most valuable placement is the shallowest high-pp parent. Evaluate at the
        # spine nodes of the optimal tree.
        children = [[] for _ in range(len(opt_par))]
        depth = [0] * len(opt_par)
        for i in range(1, len(opt_par)):
            children[opt_par[i]].append(i)
            depth[i] = depth[opt_par[i]] + 1
        for node in range(len(opt_par)):
            d_child = depth[node] + 1
            v = pvecs_w6[min(d_child, len(pvecs_w6) - 1)]
            r5 = float(v[5]) if len(v) > 5 else 0.0
            best_rank5_leaf = max(best_rank5_leaf, pp[node] * r5)
        beyond4[f"M{M}"] = {
            "dp_max_branch_with_W6": desc6["max_branch"],
            "uses_rank5plus": uses_rank5plus,
            "F_with_W6": F6, "F_width4": per_budget[M]["optimal"]["F_tree"],
            "gain_from_W6_over_W4": F6 / per_budget[M]["optimal"]["F_tree"] - 1.0,
            "least_node_path_product_in_W4_tree": least_node_pp,
            "best_rank5_leaf_marginal": best_rank5_leaf,
            "rank5_beats_least_node": best_rank5_leaf > least_node_pp,
        }
    print(f"[rho-opt] WIDTH-4 check: M=32 DP max-branch (W=6 allowed, optimistic rho5="
          f"{args.rho5_beyond}) = {beyond4['M32']['dp_max_branch_with_W6']}; "
          f"uses rank5+ = {beyond4['M32']['uses_rank5plus']}", flush=True)

    # ---- cost-model robustness on the chosen M=32 re-optimised topology ----
    d32_opt = M32["optimal"]["depth"]
    d32_74 = M32["topo74"]["depth"]
    # verify-GEMM share sweep (g_v): re-price both topologies at their own depths
    gemm_share_sens = {}
    for gs in args.gemm_share_sweep:
        gemm_share_sens[f"g{gs}"] = {
            "M32_opt_gain": price(M32["optimal"]["F_tree"], 32, d32_opt, gv=gs)["gain"],
            "M32_74_gain": price(M32["topo74"]["F_tree"], 32, d32_74, gv=gs)["gain"],
            "reopt_gain": (price(M32["optimal"]["F_tree"], 32, d32_opt, gv=gs)["gain"]
                           - price(M32["topo74"]["F_tree"], 32, d32_74, gv=gs)["gain"]),
        }
    # drafter-depth share sweep (g_d): g_d=0 is the PR-literal #68 M-only cost, the
    # central 0.168 the measured drafter share; this is the dominant modeling uncertainty.
    g_drafter_sens = {}
    for gd in args.g_drafter_sweep:
        g_drafter_sens[f"gd{gd}"] = {
            "M32_opt_gain": price(M32["optimal"]["F_tree"], 32, d32_opt, gd=gd)["gain"],
            "M32_74_gain": price(M32["topo74"]["F_tree"], 32, d32_74, gd=gd)["gain"],
            "M32_opt_depth": d32_opt, "M32_74_depth": d32_74,
        }

    # ---- decision: hand land #71 new arrays, or bank #74? (wall_tps MDE 0.1-0.2%) ----
    wall_mde = 0.002  # lawine #72 upper MDE
    reopt32 = M32["reopt_tps_gain_over_74"]
    materially_better = reopt32 > wall_mde
    primary_gain = M32["optimal"]["pricing"]["gain"]   # measured_rho_optimal_M32_gain_pct

    verdict = {
        "primary_metric_name": "measured_rho_optimal_M32_gain_pct",
        "measured_rho_optimal_M32_gain_pct": primary_gain,
        "test_metric_name": "expected_pooled_branch_hit_salvage",
        "expected_pooled_branch_hit_salvage": rank2_hit_global,
        "M32_optimal_proj_wall_tps": M32["optimal"]["pricing"]["proj_wall_tps"],
        "M32_optimal_proj_official_tps": M32["optimal"]["pricing"]["proj_official_tps"],
        "M32_optimal_F_tree": M32["optimal"]["F_tree"],
        "M32_74_F_tree": M32["topo74"]["F_tree"],
        "reopt_ET_gain_over_74_M32": M32["reopt_ET_gain_over_74"],
        "reopt_tps_gain_over_74_M32": reopt32,
        "reopt_wall_tps_delta_M32": M32["reopt_wall_tps_delta"],
        "reopt_within_wall_mde": reopt32 <= wall_mde,
        "materially_better_than_74": materially_better,
        "decision": ("hand land #71 the re-optimised M=32 parent array"
                     if materially_better else
                     "bank #74 as-is (re-opt within wall_tps MDE; #74 is rho-ladder robust)"),
        "M16_optimal_proj_wall_tps": M16["optimal"]["pricing"]["proj_wall_tps"],
        "reopt_ET_gain_over_74_M16": M16["reopt_ET_gain_over_74"],
        "M32_dominates_M16": m32_dominates_m16,
        # branch-factor verdict: the measured DECLINING ladder pulls the M=32 optimum
        # from #74's max-branch-4 down to max-branch-3; width-4 adds ZERO E[T] (the DP
        # declines to use a 4th child), and width-5+ never pays. THIS is the re-opt win.
        "M32_optimal_max_branch": M32["optimal"]["max_branch"],
        "M32_74_max_branch": M32["topo74"]["max_branch"],
        "width4_buys_over_width3_pp": (width_used["M32_branch4"]["gain"]
                                       - width_used["M32_branch3"]["gain"]),
        "width3_buys_over_width2_pp": (width_used["M32_branch3"]["gain"]
                                       - width_used["M32_branch2"]["gain"]),
        "width4_used_by_optimum": M32["optimal"]["max_branch"] >= 4,
        "beyond_width4_pays": beyond4["M32"]["rank5_beats_least_node"],
        "hard_M_cap_33_cliff": real.step_mult(33, g),
    }

    handoff = {
        "consumer": "land #71 (tree-verify serving path)",
        "build_target_M32_parent": M32["optimal"]["parent"],
        "build_target_M16_parent": M16["optimal"]["parent"],
        "topo74_M32_parent": TOPO_74[32],
        "topo74_M16_parent": TOPO_74[16],
        "decision": verdict["decision"],
        "salvage_gate_rank2_target": rank2_hit_global,
        "salvage_gate_cov4_target": cov4_global,
        "salvage_per_position_target": [
            {"pos": r["spine_position"], "width": r["branch_width"],
             "target_salvage": r["expected_salvage_full_width"]} for r in oracle_rows],
    }

    results = {
        "config": vars(args),
        "inputs": {
            "rho_cond_measured": rho_cond,
            "cov4_measured": rank_cov["cov_W"],
            "top1_measured": rank_cov["top1_measured"],
            "hard_miss_beyond_top4": rank_cov["frac_true_beyond_topW"],
            "depth_q_76": q,
            "rho2_by_depth": rho2_by_depth,
            "per_depth_div_count": per_depth_div,
            "rank_fd_hist_pooled": rank_fd_hist,
            "anchors": {"wall_tps": WALL_TPS_LINEAR_M8, "official": OFFICIAL_TPS_LINEAR_M8,
                        "legacy_local": LEGACY_LOCAL_STEADY},
            "gemm_cost_mult": {str(M): real.step_mult(M, g) for M in (8, 16, 24, 32, 33)},
        },
        "anchor_F_linear8": F_lin8,
        "per_budget": per_budget,
        "m32_dominates_m16": m32_dominates_m16,
        "salvage_oracle": salvage_oracle,
        "width_branch_factor": width_used,
        "beyond_width4": beyond4,
        "gemm_share_sensitivity": gemm_share_sens,
        "g_drafter_sensitivity": g_drafter_sens,
        "verdict": verdict,
        "handoff_land71": handoff,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"[rho-opt] wrote {args.output}", flush=True)

    # ---- console summary ----
    print("\n[rho-opt] ===== measured-rho-OPTIMAL topology vs #74 borrowed-rho =====", flush=True)
    for M in args.budgets:
        b = per_budget[M]
        o, s = b["optimal"], b["topo74"]
        print(f"  M={M}: opt F={o['F_tree']:.4f} (wid/depth {o['width_by_depth']}) "
              f"vs #74 F={s['F_tree']:.4f}  re-opt E[T] {b['reopt_ET_gain_over_74']*100:+.2f}% "
              f"| opt wall_tps {o['pricing']['proj_wall_tps']:.1f} "
              f"({o['pricing']['gain']*100:+.1f}%)  official {o['pricing']['proj_official_tps']:.1f}",
              flush=True)
    print(f"  M=32 dominates M=16: {m32_dominates_m16}", flush=True)
    print(f"  primary measured_rho_optimal_M32_gain_pct = {primary_gain*100:+.2f}%", flush=True)
    print(f"  test expected_pooled_branch_hit_salvage = {rank2_hit_global:.4f}", flush=True)
    print(f"  DECISION: {verdict['decision']}", flush=True)

    if not args.no_wandb:
        try:
            log_wandb(args, results, verdict, per_budget, oracle_rows)
        except Exception as e:  # noqa: BLE001
            print(f"[rho-opt] W&B logging failed (non-fatal): {e!r}", flush=True)
    print("[rho-opt] DONE", flush=True)


def _path_products(parent, pvecs):
    n = len(parent)
    children = [[] for _ in range(n)]
    depth = [0] * n
    for i in range(1, n):
        children[parent[i]].append(i)
    pp = np.zeros(n)
    pp[0] = 1.0
    maxd = len(pvecs) - 1
    for par in range(n):
        for rank, c in enumerate(children[par], start=1):
            d = depth[par] + 1
            v = pvecs[min(d, maxd)]
            r = rank if rank < len(v) else len(v) - 1
            pp[c] = pp[par] * v[r]
            depth[c] = d
    return pp


def log_wandb(args, results, verdict, per_budget, oracle_rows):
    import wandb
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     group=args.wandb_group, name=args.wandb_name, job_type="profiling",
                     config={"W": args.W, "max_branch": args.max_branch, "gemm_share": args.gemm_share,
                             "extrapolate": args.extrapolate, "accept_source": args.accept_source,
                             "wall_tps_anchor": WALL_TPS_LINEAR_M8,
                             "official_anchor": OFFICIAL_TPS_LINEAR_M8, "hard_M_cap": HARD_M_CAP})
    run.summary.update({k: v for k, v in verdict.items() if not isinstance(v, (dict, list))})
    # per-budget table
    tb = wandb.Table(columns=["M", "F_opt", "F_74", "reopt_ET_gain", "opt_gain",
                              "opt_wall_tps", "opt_official_tps", "depth", "max_branch"])
    for M in args.budgets:
        b = per_budget[M]
        o = b["optimal"]
        tb.add_data(M, o["F_tree"], b["topo74"]["F_tree"], b["reopt_ET_gain_over_74"],
                    o["pricing"]["gain"], o["pricing"]["proj_wall_tps"],
                    o["pricing"]["proj_official_tps"], o["depth"], o["max_branch"])
    run.log({"per_budget": tb})
    # salvage oracle table
    so = wandb.Table(columns=["spine_position", "branch_width", "q_spine", "rho2_at_pos",
                              "salvage_full_width", "salvage_rank2_only", "first_div_weight"])
    for r in oracle_rows:
        so.add_data(r["spine_position"], r["branch_width"], r["q_spine"], r["rho2_at_pos"],
                    r["expected_salvage_full_width"], r["expected_salvage_rank2_only"],
                    r["first_div_weight"])
    run.log({"salvage_oracle": so})
    run.finish()
    print(f"[rho-opt] W&B run: {run.url}", flush=True)


if __name__ == "__main__":
    main()
