#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Re-price #74's M=16/M=32 DP trees with the MEASURED deployed-chain acceptance (PR #76).

WHAT THIS ANSWERS
-----------------
PR #74 projected a +18-20% tree-verify TPS gain, but priced E[T] with a MODELED
geometric acceptance pinned to top-1 = 0.6792 -- a scalar measured on an EAGLE-3
drafter (#16/#26), NOT the deployed MTP drafter. PR #76 MEASURED the real per-
position acceptance on the served `fa2sw_precache_kenyan` stack (linear MTP K=7,
M=8 verify, #43 split-KV) by reading vLLM's own spec-decode counters
(`scripts/profiler/accept_calibration.py`). The measurement (cross-validated by the
server-log aggregate AND the Prometheus per-position counters) is:

    E[T] = 3.844 tok/step          (deployed chain, conc=1, 128 prompts x 512 tok)
    cumulative  C[1..7] = [0.7287, 0.5531, 0.4383, 0.3602, 0.3005, 0.2510, 0.2126]
    conditional q[1..7] = [0.7287, 0.7590, 0.7925, 0.8217, 0.8343, 0.8353, 0.8473]

Two facts the modeled curve missed:
  (1) the real top-1 is 0.729, NOT 0.6792 (better drafter) and NOT 0.775 (#68's
      geometric back-solve from 3.8 tok/step);
  (2) acceptance is NOT geometric -- the conditional acceptance RISES with depth
      (0.729 -> 0.847): once a few tokens are accepted, the chain is in an "easy"
      run and keeps accepting. A geometric model with a single p CANNOT represent
      this; it is exactly why #68's constant-p back-solve overstated top-1.

This script re-prices the EXACT #74 M=16 and M=32 DP topologies (the parent arrays
land #71 will build) under a DEPTH-DEPENDENT acceptance model that reproduces the
measured baseline E[T]=3.844 by construction, and reports the tightened TPS gain
band that replaces #74's modeled +18-20%.

DEPTH-DEPENDENT ACCEPTANCE MODEL
--------------------------------
Edge acceptance a(rank r, child-depth d):
  * rank-1 (spine):  a(1,d) = q[d]   -- the MEASURED conditional acceptance at depth
    d (held flat at q[7]=0.847 for d>7; the M=32 tree reaches depth 9).
  * rank>=2 (branch rescue): the residual mass (1-q[d]) is split between a "rescue"
    fraction rho that ranks 2..W recover and a "hard miss" (target token absent from
    the drafter top-W -> path stops). We reuse the validated #49 per-rank split:
        p_d = derive_per_rank(top1=q[d], topW=q[d]+rho*(1-q[d]), W, decay)
    so a(r,d) = p_d[r]. The rank-2..W SHAPE (geom/uniform/sqrt) is swept exactly as
    in #49/#74.

rho (rescue ratio) is the ONE quantity the linear deployed chain cannot reveal -- a
linear chain only ever proposes rank-1, so rank-2+ coverage is unmeasured here. We
default to the EAGLE-3 measured rho = (0.8605-0.6792)/(1-0.6792) = 0.565 (modeled,
flagged) and SWEEP it: rho=0 (branching useless -> tree ~ deeper linear, hard lower
bound) ... rho=0.75 (generous). The verdict's robustness to rho is the headline
uncertainty, since rho is the only borrowed parameter.

CONSISTENCY ANCHOR: F_linear(8) under this model = 1 + sum_{d<=7} C_meas[d] = 3.844
= the MEASURED E[T]. The model reproduces the deployed baseline exactly (the analog
of #74's geometric model reproducing its 2.976). The tree is scored under the SAME
model, so the gain ratio is apples-to-apples.

COST (acceptance-INDEPENDENT, unchanged from #74/#68)
-----------------------------------------------------
verify cost depends only on the node budget M, not the topology or acceptance
(#68 caveat 4). We reuse #68's measured real GEMM curve verbatim:
    proj_tps(M) = 428.37 * [F_tree(M)/F_linear(8)_measured] / c_B(M)
    c_B(M) = (1-g) + g*GEMM68(M)/GEMM68(8),  g=0.532  (M=16: 1.034, M=32: 1.098)

FAIL-FAST: if the measured acceptance makes gain(M=32) < +5%, report the negative
decisively (the modeled 0.6792 inflated #74's projection) and flag land #71.

LOCAL, CPU-ONLY, ANALYTIC. No GPU, no vLLM, no HF Job, no submission, no served-file
change. Reuses the #49 DP / acceptance / Monte-Carlo machinery verbatim.
"""
from __future__ import annotations

import argparse
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
    selfcheck,
)
from treeshape_real_cost import RealGemmCurve  # noqa: E402

FRONTIER_TPS_LINEAR_M8 = 428.37   # local steady, fa2sw_precache_kenyan (#52), as in #74
HARD_M_CAP = 32                   # #68 M=33 Marlin tile cliff
GEMM_SHARE_DEFAULT = 0.532
ROOFLINE = "research/spec_cost_model/verify_gemm_roofline.json"
ACCEPT_JSON = "research/accept_calibration/accept_calibration_results.json"

# EXACT #74 DP topologies (the parent arrays land #71 builds; report_treeshape_real_cost.md).
TOPO_74 = {
    16: [-1, 0, 0, 0, 1, 1, 2, 4, 4, 5, 6, 7, 11, 12, 13, 14],
    32: [-1, 0, 0, 0, 0, 1, 1, 1, 2, 3, 5, 5, 5, 6, 7, 8, 9, 10, 10, 11,
         13, 15, 17, 17, 18, 19, 20, 21, 22, 28, 29, 30],
}

# #74 modeled geometric anchors (for the before/after table).
F_LINEAR_M8_MODELED_74 = 2.976
GAIN_74_M16 = 0.131
GAIN_74_M32 = 0.201


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} not JSON serializable")


# --------------------------------------------------------------------------- #
# Load the measured per-position acceptance (PR #76 calibration run)
# --------------------------------------------------------------------------- #
def load_measured(path: str, source: str = "server_log") -> dict:
    """Return measured cumulative C, conditional q, E[T] from the #76 calib JSON.

    source: 'server_log' (draft-weighted aggregate, the primary metric) or
            'prometheus' (exact whole-run counters, the independent cross-check).
    """
    d = json.load(open(path))
    key = "server_log_metrics" if source == "server_log" else "prometheus_metrics"
    node = d[key]
    C = [float(x) for x in node["cumulative_acceptance_C"]]
    q = [float(x) for x in node["conditional_acceptance_p"]]
    ET = float(node["mean_tokens_per_step_E_T"])
    return {"C": C, "q": q, "E_T": ET, "source": source,
            "num_drafts": node.get("num_drafts")}


# --------------------------------------------------------------------------- #
# Depth-dependent per-rank vectors and tree scoring
# --------------------------------------------------------------------------- #
def build_depth_pvecs(q: list[float], rho: float, W: int, decay: str,
                      max_depth: int, extrapolate: str = "flat") -> list[np.ndarray]:
    """pvecs[d] = per-rank acceptance vector for a CHILD at depth d (1-indexed).

    pvecs[d][1] = q_eff[d]   (rank-1 = measured conditional acceptance at depth d)
    pvecs[d][r] = rescue mass for rank r (#49 derive_per_rank split, ratio rho).
    q_eff extrapolates past the measured horizon (len(q)) per `extrapolate`.
    pvecs[0] is a dummy (no depth-0 child). Index with min(d, max_depth).
    """
    qm = list(q)
    K = len(qm)
    pvecs: list[np.ndarray] = [np.zeros(W + 1)]  # depth 0 dummy
    for d in range(1, max_depth + 1):
        if d <= K:
            qd = qm[d - 1]
        elif extrapolate == "rise":
            # continue the last measured increment, capped below 1
            inc = qm[-1] - qm[-2] if K >= 2 else 0.0
            qd = min(0.995, qm[-1] + inc * (d - K))
        else:  # flat (conservative default)
            qd = qm[-1]
        topW = min(0.999, qd + rho * (1.0 - qd))
        pvecs.append(derive_per_rank(qd, topW, W, decay))
    return pvecs


def build_depth_pvecs_measured(q: list[float], rho_cond: list[float], W: int,
                               max_depth: int, extrapolate: str = "flat") -> list[np.ndarray]:
    """pvecs[d] using the MEASURED per-rank conditional rescue ratios (PR #79).

    Unlike ``build_depth_pvecs`` (which splits a single cumulative rescue scalar
    ``rho`` across ranks 2..W with a decay SHAPE), this consumes the directly
    measured conditional ratios ``rho_cond = [rho2, rho3, rho4, ...]`` where
    ``rho_r = P(target argmax == drafter rank r | drafter ranks 1..r-1 all
    missed)`` on the true greedy prefix. The per-rank ABSOLUTE marginals follow the
    chain rule (each rank only fires when all shallower ranks missed):

        p[1] = q[d]
        p[2] = rho2 * (1-q[d])
        p[3] = (1-rho2)*rho3 * (1-q[d])
        p[r] = prod_{j<r}(1-rho_j) * rho_r * (1-q[d])

    so the cumulative top-W coverage given miss-1, sum_{r>=2} p[r]/(1-q[d]), equals
    the measured cov_W = rho2 + (1-rho2)rho3 + ... exactly. rho_cond is depth-pooled
    (the probe checks depth-stability separately); only the rank-1 spine q[d] stays
    depth-dependent. NOTE: measured marginals are NOT forced monotone -- score_tree_
    depthrank assigns edges by birth-order rank regardless, so scoring a FIXED
    topology is exact; topology RE-optimisation (Sequoia DP) still assumes p
    non-increasing and is reported only as a cross-check.
    """
    qm = list(q)
    K = len(qm)
    pvecs: list[np.ndarray] = [np.zeros(W + 1)]  # depth-0 dummy
    for d in range(1, max_depth + 1):
        if d <= K:
            qd = qm[d - 1]
        elif extrapolate == "rise":
            inc = qm[-1] - qm[-2] if K >= 2 else 0.0
            qd = min(0.995, qm[-1] + inc * (d - K))
        else:
            qd = qm[-1]
        pv = np.zeros(W + 1, dtype=np.float64)
        pv[1] = qd
        miss = 1.0 - qd
        surv = 1.0          # P(all shallower ranks missed), starts at 1 for rank-2
        for r in range(2, W + 1):
            rr = rho_cond[r - 2] if (r - 2) < len(rho_cond) else 0.0
            pv[r] = surv * rr * miss
            surv *= (1.0 - rr)
        pvecs.append(pv)
    return pvecs


def score_tree_depthrank(parent: list[int], pvecs: list[np.ndarray]) -> tuple[float, int]:
    """F = sum of path-products, where a child's edge prob = pvecs[child_depth][rank].

    Generalises sequoia_dp_tree.score_tree to a depth-DEPENDENT per-rank vector.
    Sibling rank = birth order (ascending node id), identical to score_tree.
    """
    n = len(parent)
    children: list[list[int]] = [[] for _ in range(n)]
    for i in range(1, n):
        children[parent[i]].append(i)
    pp = np.zeros(n)
    pp[0] = 1.0
    depth = [0] * n
    F = 1.0
    maxd = len(pvecs) - 1
    for par in range(n):
        for rank, c in enumerate(children[par], start=1):
            d = depth[par] + 1
            pv = pvecs[min(d, maxd)]
            r = rank if rank < len(pv) else len(pv) - 1
            pp[c] = pp[par] * pv[r]
            depth[c] = d
            F += pp[c]
    return float(F), int(max(depth))


def simulate_greedy_depthrank(parent: list[int], pvecs: list[np.ndarray],
                              trials: int, seed: int) -> float:
    """Monte-Carlo E[committed tokens] under greedy tree-verify with depth-dependent
    marginals. Ground-truth check that score_tree_depthrank really is E[T]."""
    n = len(parent)
    children: list[list[int]] = [[] for _ in range(n)]
    depth = [0] * n
    for i in range(1, n):
        children[parent[i]].append(i)
        depth[i] = depth[parent[i]] + 1
    maxd = len(pvecs) - 1
    rng = np.random.default_rng(seed)
    total = 0
    for _ in range(trials):
        u, length = 0, 0
        while children[u]:
            kids = children[u]
            d = depth[u] + 1
            pv = pvecs[min(d, maxd)]
            probs = np.array([pv[r if r < len(pv) else len(pv) - 1]
                              for r in range(1, len(kids) + 1)], dtype=np.float64)
            draw = rng.random()
            cum, chosen = 0.0, -1
            for idx, pr in enumerate(probs):
                cum += pr
                if draw < cum:
                    chosen = idx
                    break
            if chosen < 0:
                break
            u = kids[chosen]
            length += 1
        total += length + 1
    return total / trials


def reprice(meas: dict, real: RealGemmCurve, *, rho: float, W: int, decay: str,
            g: float, extrapolate: str, max_depth: int = 40) -> dict:
    """Score the #74 M=16/M=32 topologies under the depth-dependent measured model."""
    q = meas["q"]
    pvecs = build_depth_pvecs(q, rho, W, decay, max_depth, extrapolate)
    F_lin8, _ = score_tree_depthrank(build_linear(8), pvecs)  # == measured E[T]
    out = {"rho": rho, "decay": decay, "g": g, "extrapolate": extrapolate,
           "F_linear8": F_lin8, "F_linear8_vs_measured_ET": F_lin8 - meas["E_T"]}
    for M in (16, 32):
        F_tree, d_tree = score_tree_depthrank(TOPO_74[M], pvecs)
        cB = real.step_mult(M, g)
        et_ratio = F_tree / F_lin8
        out[f"M{M}"] = {
            "F_tree": F_tree, "depth": d_tree, "cost_mult": cB,
            "ET_ratio_vs_lin8": et_ratio,
            "tps_gain": et_ratio / cB - 1.0,
            "proj_tps": FRONTIER_TPS_LINEAR_M8 * et_ratio / cB,
        }
    out["M32_dominates_M16"] = out["M32"]["proj_tps"] > out["M16"]["proj_tps"]
    out["pvecs_depth1"] = [float(x) for x in pvecs[1]]
    out["coverage_by_depth"] = [
        float(pvecs[min(d, max_depth)][1:W + 1].sum()) for d in range(1, 10)
    ]
    return out


def load_rank_coverage(path: str) -> dict:
    """Load measured rho2/rho3/rho4 + cumulative cov_W from PR #79 rank_coverage JSON."""
    d = json.load(open(path))
    a = d["analysis"] if "analysis" in d else d
    rho = a["rho_marginal"]
    cov = a["cumulative_coverage"]
    W = int(a.get("W", 4))
    rho_cond = [float(rho[str(r)]) for r in range(2, W + 1) if rho.get(str(r)) is not None]
    cov_W = float(cov[str(W)]) if cov.get(str(W)) is not None else None
    return {
        "rho_cond": rho_cond,            # [rho2, rho3, rho4]
        "cov_W": cov_W,                  # cumulative rank-2..W coverage | miss-1
        "rho2": float(rho["2"]) if rho.get("2") is not None else None,
        "W": W,
        "top1_measured": a.get("top1_acceptance"),
        "n_divergences": a.get("n_divergences"),
        "frac_true_beyond_topW": a.get("frac_true_beyond_topW"),
        "wandb_run_id": d.get("wandb_run_id"),
    }


def reprice_measured_split(meas: dict, real: RealGemmCurve, *, rho_cond: list[float],
                           W: int, g: float, extrapolate: str, max_depth: int = 40) -> dict:
    """Score #74 M=16/M=32 under the MEASURED per-rank split (chain-rule pvecs)."""
    q = meas["q"]
    pvecs = build_depth_pvecs_measured(q, rho_cond, W, max_depth, extrapolate)
    F_lin8, _ = score_tree_depthrank(build_linear(8), pvecs)
    cov_W = float(sum(pvecs[1][2:W + 1]) / (1.0 - pvecs[1][1])) if pvecs[1][1] < 1 else 0.0
    out = {"rho_cond": list(rho_cond), "cov_W_reconstructed": cov_W, "g": g,
           "extrapolate": extrapolate, "F_linear8": F_lin8,
           "F_linear8_vs_measured_ET": F_lin8 - meas["E_T"]}
    for M in (16, 32):
        F_tree, d_tree = score_tree_depthrank(TOPO_74[M], pvecs)
        cB = real.step_mult(M, g)
        et_ratio = F_tree / F_lin8
        out[f"M{M}"] = {
            "F_tree": F_tree, "depth": d_tree, "cost_mult": cB,
            "ET_ratio_vs_lin8": et_ratio,
            "tps_gain": et_ratio / cB - 1.0,
            "proj_tps": FRONTIER_TPS_LINEAR_M8 * et_ratio / cB,
        }
    out["M32_dominates_M16"] = out["M32"]["proj_tps"] > out["M16"]["proj_tps"]
    out["pvecs_depth1"] = [float(x) for x in pvecs[1]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accept-json", default=ACCEPT_JSON)
    ap.add_argument("--accept-source", default="server_log",
                    choices=["server_log", "prometheus"])
    ap.add_argument("--roofline", default=ROOFLINE)
    ap.add_argument("--rho", type=float, default=0.565, help="rescue ratio (EAGLE-3 modeled)")
    ap.add_argument("--rho-sweep", type=float, nargs="+", default=[0.0, 0.35, 0.565, 0.75])
    ap.add_argument("--rank-coverage-json", default=None,
                    help="PR #79 rank_coverage_results.json: use MEASURED per-rank rho2/3/4 "
                         "split as the central case (replaces the borrowed 0.565) and sets "
                         "--rho to the measured cumulative cov_W for the robustness sweep.")
    ap.add_argument("--W", type=int, default=4)
    ap.add_argument("--decays", nargs="+", default=["geom", "uniform", "sqrt"])
    ap.add_argument("--gemm-share", type=float, default=GEMM_SHARE_DEFAULT)
    ap.add_argument("--gemm-share-sweep", type=float, nargs="+", default=[0.45, 0.532, 0.60, 0.70])
    ap.add_argument("--extrapolate", default="flat", choices=["flat", "rise"])
    ap.add_argument("--mc-trials", type=int, default=400_000)
    ap.add_argument("--fail-fast-gain", type=float, default=0.05)
    ap.add_argument("--output", default="research/accept_calibration/treeshape_measured_results.json")
    ap.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT"))
    ap.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--wandb-group", default="acceptance-calibration")
    ap.add_argument("--wandb-name", default="wirbel/tree-reprice-measured")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    # ---- self-check the reused #49 machinery (DP==BF, linear==geom, MC==F) ----
    selfcheck()

    meas = load_measured(args.accept_json, args.accept_source)
    meas_x = load_measured(args.accept_json, "prometheus"
                           if args.accept_source == "server_log" else "server_log")
    real = RealGemmCurve(args.roofline)
    g = args.gemm_share

    # PR #79: replace the borrowed EAGLE-3 rho=0.565 with the MEASURED rank-coverage.
    rank_cov = None
    if args.rank_coverage_json:
        rank_cov = load_rank_coverage(args.rank_coverage_json)
        if rank_cov["cov_W"] is not None:
            args.rho = rank_cov["cov_W"]   # center scalar sweep on measured cumulative cov_W
            args.rho_sweep = sorted(set([round(rank_cov["cov_W"], 4)] + list(args.rho_sweep)))
        print(f"[reprice] MEASURED rank-coverage (PR #79): rho_cond(2..W)={rank_cov['rho_cond']} "
              f"cov_W={rank_cov['cov_W']}  top1_meas={rank_cov['top1_measured']}  "
              f"n_div={rank_cov['n_divergences']}; central rho -> {args.rho}", flush=True)

    print(f"[reprice] measured ({meas['source']}): E[T]={meas['E_T']:.4f}  "
          f"top1={meas['C'][0]:.4f}  q={[round(x,4) for x in meas['q']]}", flush=True)
    print(f"[reprice] cross-check ({meas_x['source']}): E[T]={meas_x['E_T']:.4f}  "
          f"top1={meas_x['C'][0]:.4f}", flush=True)

    # ---- consistency anchor: the depth model reproduces the measured cumulative
    #      acceptance EXACTLY (linear-chain path product at depth d telescopes to
    #      C[d], since q[d]=C[d]/C[d-1]), so F_linear(8) == 1 + sum(C). It matches the
    #      reported E[T] to logging precision (the server log rounds per-interval
    #      cumulative rates to 3 decimals; the Prometheus integer counters are exact). ----
    central = reprice(meas, real, rho=args.rho, W=args.W, decay="geom", g=g,
                      extrapolate=args.extrapolate)
    anchor_exact = 1.0 + sum(meas["C"])
    anchor_err = abs(central["F_linear8"] - anchor_exact)
    et_precision_gap = abs(central["F_linear8"] - meas["E_T"])
    print(f"[reprice] ANCHOR: F_linear(8) under depth model = {central['F_linear8']:.6f} "
          f"vs 1+sum(C) {anchor_exact:.6f}  (|err|={anchor_err:.2e}); "
          f"vs reported E[T] {meas['E_T']:.4f} (gap {et_precision_gap:.4f} = log rounding)", flush=True)
    assert anchor_err < 1e-9, "depth model does not reproduce the measured cumulative C"
    assert et_precision_gap < 5e-3, "F_linear(8) departs from measured E[T] beyond log precision"

    # ---- MC-validate the two recommended topologies under the depth model ----
    pvecs = build_depth_pvecs(meas["q"], args.rho, args.W, "geom", 40, args.extrapolate)
    validation = {}
    for M in (16, 32):
        F_path = score_tree_depthrank(TOPO_74[M], pvecs)[0]
        mc = simulate_greedy_depthrank(TOPO_74[M], pvecs, args.mc_trials, seed=7)
        validation[f"M{M}"] = {"F_path": F_path, "mc_E_T": mc,
                               "abs_err": abs(mc - F_path), "rel_err": abs(mc - F_path) / F_path}
        print(f"[reprice] validate M={M}: F_path={F_path:.4f}  MC={mc:.4f}  "
              f"|err|={abs(mc-F_path):.4f}", flush=True)

    # ---- rho sweep (the headline uncertainty: rank-2+ coverage is unmeasured) ----
    rho_sweep = {f"rho{r}": reprice(meas, real, rho=r, W=args.W, decay="geom", g=g,
                                    extrapolate=args.extrapolate) for r in args.rho_sweep}
    # ---- decay-split robustness ----
    decay_robust = {dc: reprice(meas, real, rho=args.rho, W=args.W, decay=dc, g=g,
                                extrapolate=args.extrapolate) for dc in args.decays}
    # ---- GEMM-share robustness ----
    share_sens = {f"g{gs}": reprice(meas, real, rho=args.rho, W=args.W, decay="geom",
                                    g=gs, extrapolate=args.extrapolate)
                  for gs in args.gemm_share_sweep}
    # ---- extrapolation (flat vs continued-rise past the measured depth-7 horizon) ----
    extrap_sens = {ex: reprice(meas, real, rho=args.rho, W=args.W, decay="geom", g=g,
                               extrapolate=ex) for ex in ("flat", "rise")}

    # ---- cross-check on the Prometheus per-position vector (independent counters) ----
    central_prom = reprice(meas_x, real, rho=args.rho, W=args.W, decay="geom", g=g,
                           extrapolate=args.extrapolate)

    # ---- position-INDEPENDENT cross-check: what #74's own framework says at the
    #      corrected top-1=0.729 (geometric, NOT depth-dependent). Different model
    #      class (baseline is the geometric F_linear(8), not 3.844); shown only to
    #      separate "corrected top-1" from "depth-dependent profile". ----
    posindep = {}
    rescue_ratio = args.rho
    for top1 in (0.6792, meas["C"][0]):
        topW = min(0.999, top1 + rescue_ratio * (1.0 - top1))
        p = derive_per_rank(top1, topW, args.W, "geom")
        FL8 = score_tree(build_linear(8), p)[0]
        d = {"top1": top1, "topW": topW, "F_linear8_geom": FL8}
        for M in (16, 32):
            Fdp = build_sequoia_tree(p, M, 24, 4)[1]
            cB = real.step_mult(M, g)
            d[f"M{M}"] = {"F_dp": Fdp, "ET_ratio": Fdp / FL8,
                          "tps_gain": (Fdp / FL8) / cB - 1.0,
                          "proj_tps": FRONTIER_TPS_LINEAR_M8 * (Fdp / FL8) / cB}
        posindep[f"top1_{top1:.4f}"] = d

    # ---- topology stability: is the #74 shape still ~optimal under the measured
    #      (rising) profile? The Sequoia DP is position-INDEPENDENT, so it cannot
    #      directly optimise a depth-dependent objective; we approximate by building
    #      DP trees at several effective flat-p anchors (incl. the mean conditional)
    #      and scoring THEM under the depth-dependent model, then compare to the fixed
    #      #74 topology. If #74 is within ~1% of the best, land #71 builds it as-is. ----
    topo_check = {}
    for M in (16, 32):
        f74 = score_tree_depthrank(TOPO_74[M], pvecs)[0]
        best_f, best_anchor, best_par = f74, "topo74", TOPO_74[M]
        for anchor in (0.6792, meas["C"][0], float(np.mean(meas["q"]))):
            topW = min(0.999, anchor + args.rho * (1.0 - anchor))
            p_anchor = derive_per_rank(anchor, topW, args.W, "geom")
            par = build_sequoia_tree(p_anchor, M, 24, 4)[0]
            f = score_tree_depthrank(par, pvecs)[0]
            if f > best_f + 1e-9:
                best_f, best_anchor, best_par = f, f"dp_top1_{anchor:.4f}", par
        topo_check[f"M{M}"] = {
            "F_topo74": f74, "F_best": best_f, "best_anchor": best_anchor,
            "reopt_gain_over_74": best_f / f74 - 1.0,
            "topo74_is_near_optimal": (best_f / f74 - 1.0) < 0.01,
            "better_parent": None if best_anchor == "topo74" else best_par,
        }
        print(f"[reprice] topo-stability M={M}: #74 F={f74:.4f}  best F={best_f:.4f} "
              f"({best_anchor})  reopt-gain={topo_check[f'M{M}']['reopt_gain_over_74']*100:+.2f}%",
              flush=True)

    # ---- verdict + fail-fast ----
    gain32 = central["M32"]["tps_gain"]
    gain16 = central["M16"]["tps_gain"]
    gains32_rho = [v["M32"]["tps_gain"] for v in rho_sweep.values()]
    gains32_all = (gains32_rho
                   + [v["M32"]["tps_gain"] for v in decay_robust.values()]
                   + [v["M32"]["tps_gain"] for v in share_sens.values()]
                   + [v["M32"]["tps_gain"] for v in extrap_sens.values()])
    verdict = {
        "primary_metric_name": "deployed_chain_mean_tokens_per_step",
        "deployed_chain_mean_tokens_per_step": meas["E_T"],
        "deployed_chain_top1": meas["C"][0],
        "deployed_chain_ET_prometheus_xcheck": meas_x["E_T"],
        "F_linear8_measured": central["F_linear8"],
        "M32_proj_tps": central["M32"]["proj_tps"],
        "M32_gain_central": gain32,
        "M16_proj_tps": central["M16"]["proj_tps"],
        "M16_gain_central": gain16,
        "M32_dominates_M16": central["M32_dominates_M16"],
        "tree_beats_deployed_linear": gain32 > 0.0,
        "M32_gain_band_min": min(gains32_all),
        "M32_gain_band_max": max(gains32_all),
        "M32_gain_rho0_lowerbound": rho_sweep["rho0.0"]["M32"]["tps_gain"],
        "fail_fast_triggered": gain32 < args.fail_fast_gain,
        "mc_max_rel_err": max(v["rel_err"] for v in validation.values()),
        # #74 before/after
        "gain74_M32_modeled": GAIN_74_M32,
        "gain74_M16_modeled": GAIN_74_M16,
        "gain_shift_M32_vs_74": gain32 - GAIN_74_M32,
        # topology stability under the measured profile
        "M32_topo74_near_optimal": topo_check["M32"]["topo74_is_near_optimal"],
        "M32_reopt_gain_over_74": topo_check["M32"]["reopt_gain_over_74"],
        "M16_topo74_near_optimal": topo_check["M16"]["topo74_is_near_optimal"],
    }

    reconciliation = {
        "question": "Reconcile #49 top-1=0.6792 vs #68 ~3.8 tok/step (implied top-1~0.775).",
        "measured_top1": meas["C"][0],
        "measured_E_T": meas["E_T"],
        "verdict": (
            "Both off, opposite directions, different reasons. #49's 0.6792 was an "
            "EAGLE-3 drafter scalar (#16/#26), NOT the deployed MTP drafter, which "
            "accepts MORE at rank-1 (0.729). #68's 3.8 tok/step is correct (measured "
            "E[T]=3.844) but its geometric back-solve to top-1~0.775 OVERSTATES "
            "first-position acceptance because the real per-position profile is NOT "
            "constant -- conditional acceptance RISES with depth (0.729->0.847), so a "
            "single constant p forced to hit E[T]=3.84 must sit above the true top-1. "
            "Authoritative: top-1=0.729, E[T]=3.844 tok/step (= 1 bonus + 2.844 "
            "accepted drafts; draft-acceptance-rate 0.406 = 2.844/7)."),
        "is_real_disagreement": False,
        "bonus_token_included_in_ET": True,
    }

    handoff = {
        "consumer": "land #71 (tree-verify serving path)",
        "tightened_gain_band_replaces_74_modeled_18_20pct": [
            round(min(gains32_all), 4), round(max(gains32_all), 4)],
        "primary_build_target_M32": {
            "proj_tps": central["M32"]["proj_tps"], "gain": gain32,
            "E_T": central["M32"]["F_tree"], "parent": TOPO_74[32]},
        "secondary_build_target_M16": {
            "proj_tps": central["M16"]["proj_tps"], "gain": gain16,
            "E_T": central["M16"]["F_tree"], "parent": TOPO_74[16]},
        "still_build": gain32 >= args.fail_fast_gain,
        "note": ("M=32 dominates M=16 under the measured profile; the tree still "
                 "breaks the deployed M=8 linear chain's acceptance ceiling because "
                 "the rising profile lifts the tree's deep spine AND its baseline "
                 "together, preserving the ratio. rho (rank-2+ coverage) is the only "
                 "borrowed parameter; even rho=0 keeps the gain positive via the "
                 "deeper spine."),
    }

    # PR #79: exact measured per-rank split (rho2/rho3/rho4) as the precise central
    # estimate; the scalar `central` above (geom decay at rho=cov_W) is its decay-model
    # counterpart and should agree closely (decay-robustness sanity check).
    measured_split = None
    if rank_cov is not None:
        measured_split = reprice_measured_split(
            meas, real, rho_cond=rank_cov["rho_cond"], W=args.W, g=g,
            extrapolate=args.extrapolate)
        verdict["measured_split_M32_proj_tps"] = measured_split["M32"]["proj_tps"]
        verdict["measured_split_M32_gain"] = measured_split["M32"]["tps_gain"]
        verdict["measured_split_M16_proj_tps"] = measured_split["M16"]["proj_tps"]
        verdict["measured_split_M16_gain"] = measured_split["M16"]["tps_gain"]
        verdict["measured_split_M32_dominates_M16"] = measured_split["M32_dominates_M16"]
        verdict["measured_split_fail_fast_triggered"] = (
            measured_split["M32"]["tps_gain"] < args.fail_fast_gain)
        verdict["measured_cov_W"] = rank_cov["cov_W"]
        verdict["measured_rho2"] = rank_cov["rho2"]
        verdict["measured_top1"] = rank_cov["top1_measured"]
        print(f"[reprice] MEASURED-SPLIT verdict: M32 {measured_split['M32']['tps_gain']*100:+.1f}% "
              f"({measured_split['M32']['proj_tps']:.1f} TPS), M16 "
              f"{measured_split['M16']['tps_gain']*100:+.1f}% "
              f"({measured_split['M16']['proj_tps']:.1f} TPS); M32>M16="
              f"{measured_split['M32_dominates_M16']}", flush=True)

    results = {
        "config": vars(args),
        "measured_acceptance": {"server_log": meas, "prometheus": meas_x},
        "rank_coverage_input": rank_cov,
        "measured_per_rank_split": measured_split,
        "anchor_F_linear8": central["F_linear8"],
        "central_geom_rho0565": central,
        "central_prometheus_xcheck": central_prom,
        "rho_sweep": rho_sweep,
        "decay_robustness": decay_robust,
        "gemm_share_sensitivity": share_sens,
        "extrapolation_sensitivity": extrap_sens,
        "position_independent_crosscheck": posindep,
        "validation_mc": validation,
        "topology_stability": topo_check,
        "reconciliation": reconciliation,
        "verdict": verdict,
        "handoff_land71": handoff,
        "topologies_74": {str(k): v for k, v in TOPO_74.items()},
    }

    # ---- console summary ----
    print("\n[reprice] ===== #74 trees re-priced under MEASURED depth-dependent acceptance =====", flush=True)
    print(f"  deployed chain (measured): E[T]={meas['E_T']:.4f}  top1={meas['C'][0]:.4f}  "
          f"(prom xcheck E[T]={meas_x['E_T']:.4f})", flush=True)
    print(f"  baseline F_linear(8) reproduced = {central['F_linear8']:.4f} (== measured E[T])", flush=True)
    print(f"  {'M':>4} {'F_tree':>7} {'ET/lin8':>8} {'cB':>6} {'projTPS':>8} {'gain':>7}", flush=True)
    for M in (16, 32):
        r = central[f"M{M}"]
        print(f"  {M:4d} {r['F_tree']:7.4f} {r['ET_ratio_vs_lin8']:8.4f} {r['cost_mult']:6.4f} "
              f"{r['proj_tps']:8.1f} {r['tps_gain']*100:6.1f}%", flush=True)
    print(f"  M=32 dominates M=16: {central['M32_dominates_M16']}", flush=True)
    print(f"  rho sweep M=32 gain: " + "  ".join(
        f"rho={r}:{rho_sweep[f'rho{r}']['M32']['tps_gain']*100:+.1f}%" for r in args.rho_sweep), flush=True)
    print(f"  M=32 gain band (rho x decay x g x extrap): "
          f"[{min(gains32_all)*100:+.1f}%, {max(gains32_all)*100:+.1f}%]", flush=True)
    print(f"  vs #74 modeled M=32 +{GAIN_74_M32*100:.1f}%  ->  shift {verdict['gain_shift_M32_vs_74']*100:+.1f}pp", flush=True)
    print(f"  MC max rel-err: {verdict['mc_max_rel_err']:.4%}", flush=True)
    if verdict["fail_fast_triggered"]:
        print(f"  *** FAIL-FAST: M=32 gain {gain32*100:.1f}% < {args.fail_fast_gain*100:.0f}% "
              f"-- modeled 0.6792 inflated the projection; flag land #71 ***", flush=True)
    else:
        print(f"  VERDICT: tree still wins (M=32 +{gain32*100:.1f}%); land #71 should build.", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"[reprice] wrote {args.output}", flush=True)

    if not args.no_wandb:
        try:
            log_wandb(args, results, verdict, central, rho_sweep)
        except Exception as e:  # noqa: BLE001
            print(f"[reprice] W&B logging failed (non-fatal): {e!r}", flush=True)
    print("[reprice] DONE", flush=True)


def log_wandb(args, results, verdict, central, rho_sweep):
    import wandb
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     group=args.wandb_group, name=args.wandb_name, job_type="profiling",
                     config={"rho": args.rho, "W": args.W, "gemm_share": args.gemm_share,
                             "extrapolate": args.extrapolate, "accept_source": args.accept_source,
                             "anchor_tps": FRONTIER_TPS_LINEAR_M8, "hard_M_cap": HARD_M_CAP})
    run.summary.update({k: v for k, v in verdict.items() if not isinstance(v, (dict, list))})
    tbl = wandb.Table(columns=["rho", "M16_gain", "M16_proj_tps", "M32_gain", "M32_proj_tps"])
    for r in args.rho_sweep:
        s = rho_sweep[f"rho{r}"]
        tbl.add_data(r, s["M16"]["tps_gain"], s["M16"]["proj_tps"],
                     s["M32"]["tps_gain"], s["M32"]["proj_tps"])
    run.log({"rho_sweep_gain": tbl})
    run.finish()
    print(f"[reprice] W&B run: {run.url}", flush=True)


if __name__ == "__main__":
    main()
