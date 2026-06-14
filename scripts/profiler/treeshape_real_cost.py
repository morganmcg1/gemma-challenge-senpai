#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""TPS-optimal draft-tree SHAPE under denken #68's REAL verify-GEMM cost curve.

WHAT THIS ANSWERS (PR #74)
--------------------------
My PR #49 proved a Sequoia DP-optimal draft tree gives +34% E[T] / +16% TPS over
the deployed linear MTP K=7 chain at matched node budget -- but it priced the
verify step with the MODELED tile-corrected curve V(M) (#28/#33). denken #68 (just
merged) MEASURED the real int4-Marlin verify-GEMM cost per verify-width M and found
it is (a) weight-bandwidth-bound at M=8 (so widening is affordable) and (b)
*non-uniform*: a Marlin 16-row tile staircase with cheap interiors and hard steps
at the tile boundaries M=17 and M=33. The decisive M=33 cliff (+53%) is a hard
ceiling: M <= 32.

This script EXTENDS the #49 DP cost model (it imports the #49 DP, acceptance model,
brute-force and Monte-Carlo self-checks verbatim) and swaps the modeled V(M) for
#68's measured per-M cost curve, then re-solves the TPS-optimal (tree-shape, M)
operating point(s) and hands land #71 the 1-2 concrete topologies to build first.

COST MODEL
----------
The full speculative decode step S(M) = drafter + verify(M). Only the int4 weight
GEMM block scales materially with the verify width M (#68 caveat 4: a width-W tree
changes only the attention MASK, not the weight GEMM, which processes all M rows).
We model

    S(M) = S(8) * [ (1 - g) + g * GEMM68(M) / GEMM68(8) ]                    (model B)

where GEMM68(M) = #68 aggregate `total_gemm_us[M]` (the measured real curve) and g
is the GEMM share of the decode step (#30/#68: 0.532; the #28/#33 msweep model says
~0.62; we sweep g in [0.45, 0.70]). S(8) cancels in every TPS *ratio*, so the
projected GAIN is independent of the absolute step time / E[T] anchor; we report
absolute projected TPS by pinning the deployed linear M=8 chain to its measured
428.37 local-steady TPS.

As an independent cross-check we also price with the #28/#33 full-step measured
curve V_old(M) = latency_ms_by_M (model A, what #49 used): S_A(M) = drafter+V_old(M).
The two cost models agree to ~1pp, so the verdict is robust to the pricing method.

PROJECTED TPS
-------------
    TPS(topology, M) = 428.37 * [F(topology,M) / F_linear(8)] / c_model(M)
    F_linear(8) = 2.976  (deployed linear MTP K=7, M=8 verify, measured p)

LOCAL, CPU-ONLY, ANALYTIC. No GPU, no vLLM, no HF Job, no submission, no served-file
change. Contract-safe: the recommended tree stays verifier-authoritative (greedy
identity preserved); this script only PRICES shapes, it does not build the path
(that is land #71). The cost curve it consumes is #68 (merged), not re-derived here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# Reuse the #49 machinery verbatim (DP, acceptance model, brute force, Monte-Carlo).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sequoia_dp_tree import (  # noqa: E402
    build_balanced,
    build_linear,
    build_sequoia_tree,
    derive_per_rank,
    score_tree,
    selfcheck,
    simulate_greedy_E,
)

# Deployed linear MTP K=7 (M=8 verify) anchor -- PR #74 baseline.
FRONTIER_TPS_LINEAR_M8 = 428.37          # local steady, fa2sw_precache_kenyan (#52)
F_LINEAR_M8 = None                       # filled from the #49 model at run time
HARD_M_CAP = 32                          # #68 M=33 Marlin tile cliff (+53%): M <= 32
GEMM_SHARE_DEFAULT = 0.532               # #30/#68 GEMM share of decode step
GEMM_SHARE_SWEEP = [0.45, 0.532, 0.60, 0.70]
DRAFTER_MS = 1.446                       # #43 flat per-step MTP drafter
ROOFLINE = "research/spec_cost_model/verify_gemm_roofline.json"
OLD_CURVE = "research/spec_cost_model/results_msweep.json"


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} not JSON serializable")


# --------------------------------------------------------------------------- #
# #68 REAL verify-GEMM cost curve (the curve this PR consumes)
# --------------------------------------------------------------------------- #
class RealGemmCurve:
    """#68 aggregate verify-GEMM time per verify-width M (measured, launch-free)."""

    def __init__(self, path: str = ROOFLINE):
        d = json.load(open(path))
        agg = d["aggregate_by_M"]
        self.gemm_us = {int(k): float(v["total_gemm_us"]) for k, v in agg.items()}
        self.M = sorted(self.gemm_us)
        self.base = self.gemm_us[8]                       # GEMM68(8)
        # measured per-row marginals (the non-uniform staircase #68 reported)
        self.marginal = d.get("marginal_per_row", {})

    def ratio(self, M: int) -> float:
        """GEMM68(M)/GEMM68(8); linear-interpolate between measured M points."""
        if M in self.gemm_us:
            return self.gemm_us[M] / self.base
        lo = max(m for m in self.M if m <= M)
        hi = min(m for m in self.M if m >= M)
        t = (M - lo) / (hi - lo)
        g = self.gemm_us[lo] * (1 - t) + self.gemm_us[hi] * t
        return g / self.base

    def step_mult(self, M: int, g: float = GEMM_SHARE_DEFAULT) -> float:
        """c_B(M) = full decode-step multiplier vs M=8 under #68 real GEMM curve."""
        return (1.0 - g) + g * self.ratio(M)


class OldFullStepCurve:
    """#28/#33 modeled full-step verify curve V_old(M)=latency_ms_by_M (model A)."""

    def __init__(self, path: str = OLD_CURVE, key: str = "graph|ctx256"):
        d = json.load(open(path))
        lat = d["cost_model"][key]["latency_ms_by_M"]
        self.lat = {int(k): float(v) for k, v in lat.items()}
        self.M = sorted(self.lat)

    def V(self, M: int) -> float:
        if M in self.lat:
            return self.lat[M]
        lo = max(m for m in self.M if m <= M)
        hi = min(m for m in self.M if m >= M)
        t = (M - lo) / (hi - lo)
        return self.lat[lo] * (1 - t) + self.lat[hi] * t

    def step_mult(self, M: int, drafter: float = DRAFTER_MS) -> float:
        """c_A(M) = (drafter+V_old(M)) / (drafter+V_old(8))."""
        return (drafter + self.V(M)) / (drafter + self.V(8))


# --------------------------------------------------------------------------- #
# Topology pretty-printer for the land #71 hand-off
# --------------------------------------------------------------------------- #
def describe_tree(parent: list[int]) -> dict:
    """Human-readable shape: per-depth width, branch factors, child-rank profile."""
    n = len(parent)
    children: list[list[int]] = [[] for _ in range(n)]
    depth = [0] * n
    for i in range(1, n):
        children[parent[i]].append(i)
        depth[i] = depth[parent[i]] + 1
    width_by_depth: dict[int, int] = {}
    for dd in depth:
        width_by_depth[dd] = width_by_depth.get(dd, 0) + 1
    nbranch = [len(children[i]) for i in range(n)]
    # rank-2+ branch points: nodes that add a sibling beyond the rank-1 spine child
    branch_points = sum(max(0, len(children[i]) - 1) for i in range(n))
    spine_len = 0
    u = 0
    while children[u]:
        spine_len += 1
        u = children[u][0]  # rank-1 child (birth order)
    return {
        "n_nodes": n,
        "max_depth": max(depth),
        "spine_length": spine_len,
        "width_by_depth": {int(k): int(v) for k, v in sorted(width_by_depth.items())},
        "n_rank2plus_branches": int(branch_points),
        "max_branch_factor": int(max(nbranch)),
        "parent": list(parent),
    }


# --------------------------------------------------------------------------- #
def projected_tps(F: float, mult: float) -> float:
    return FRONTIER_TPS_LINEAR_M8 * (F / F_LINEAR_M8) / mult


def main() -> None:
    global F_LINEAR_M8
    ap = argparse.ArgumentParser()
    ap.add_argument("--top1", type=float, default=0.6792)
    ap.add_argument("--topW", type=float, default=0.8605)
    ap.add_argument("--W", type=int, default=4)
    ap.add_argument("--decays", nargs="+", default=["geom", "uniform", "sqrt"])
    ap.add_argument("--max-depth", type=int, default=24)
    ap.add_argument("--max-branch", type=int, default=4)
    ap.add_argument("--gemm-share", type=float, default=GEMM_SHARE_DEFAULT)
    ap.add_argument("--roofline", default=ROOFLINE)
    ap.add_argument("--old-curve", default=OLD_CURVE)
    ap.add_argument("--mc-trials", type=int, default=400_000)
    ap.add_argument("--output", default="research/spec_cost_model/treeshape_real_cost_results.json")
    ap.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT"))
    ap.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--wandb-group", default="tree-shape-cost-model")
    ap.add_argument("--wandb-name", default="wirbel/tree-shape-real-cost")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    # ---- 1. self-check the #49 machinery we are extending (DP==BF, MC==F) ----
    selfcheck()

    real = RealGemmCurve(args.roofline)
    old = OldFullStepCurve(args.old_curve)
    g = args.gemm_share

    # canonical measured acceptance (geom), and the deployed linear M=8 anchor
    p0 = derive_per_rank(args.top1, args.topW, args.W, "geom")
    F_LINEAR_M8 = score_tree(build_linear(8), p0)[0]
    print(f"[treeshape] p={[round(float(x),4) for x in p0]}  "
          f"F_linear(M=8)={F_LINEAR_M8:.4f} -> anchored to {FRONTIER_TPS_LINEAR_M8} TPS",
          flush=True)
    print(f"[treeshape] #68 real GEMM ratio: "
          + "  ".join(f"M{M}={real.ratio(M):.4f}" for M in (8, 12, 16, 24, 32, 33)),
          flush=True)

    # ---- 2. budget sweep up to the M=33 cliff (+a couple past it to show crater) ----
    budgets = list(range(2, 35))
    rows = []
    topo: dict[int, list[int]] = {}
    for M in budgets:
        F_lin, d_lin = score_tree(build_linear(M), p0)
        F_bal, d_bal = score_tree(build_balanced(M, args.W), p0)
        dp_par, F_dp, d_dp = build_sequoia_tree(p0, M, args.max_depth, args.max_branch)
        topo[M] = dp_par
        cB = real.step_mult(M, g)
        cA = old.step_mult(M)
        feasible = M <= HARD_M_CAP
        rows.append({
            "M": M, "F_linear": F_lin, "F_balanced": F_bal, "F_dp": F_dp,
            "depth_dp": d_dp,
            "ET_ratio_dp_vs_lin8": F_dp / F_LINEAR_M8,
            "cost_mult_real_B": cB, "cost_mult_old_A": cA,
            "gemm_ratio_68": real.ratio(M),
            "tps_dp_realcost": projected_tps(F_dp, cB) if feasible else None,
            "tps_dp_oldcost": projected_tps(F_dp, cA) if feasible else None,
            "tps_linear_realcost": projected_tps(F_lin, cB) if feasible else None,
            "tps_dp_realcost_incl_cliff": projected_tps(F_dp, cB),
            "feasible_M_le_32": feasible,
        })

    feas = [r for r in rows if r["feasible_M_le_32"]]
    opt_real = max(feas, key=lambda r: r["tps_dp_realcost"])
    opt_old = max(feas, key=lambda r: r["tps_dp_oldcost"])
    lin_opt = max(feas, key=lambda r: r["tps_linear_realcost"])

    # the two build targets land #71 needs: the tile-top sweet spots
    pick = {M: next(r for r in rows if r["M"] == M) for M in (16, 32)}

    # ---- 3. validate the two recommended trees (brute-force already in selfcheck;
    #         here Monte-Carlo the exact recommended topologies) + decay robustness ----
    validation = {}
    for M in (16, 32):
        par = topo[M]
        F_path = score_tree(par, p0)[0]
        mc = simulate_greedy_E(par, p0, trials=args.mc_trials, seed=7)
        validation[f"M{M}"] = {
            "F_path_product": F_path, "mc_E_T": mc,
            "abs_err": abs(mc - F_path), "rel_err": abs(mc - F_path) / F_path,
            "shape": describe_tree(par),
        }
        print(f"[treeshape] validate M={M}: F_path={F_path:.4f}  MC={mc:.4f}  "
              f"|err|={abs(mc-F_path):.4f}", flush=True)

    # decay robustness: re-price the M=16/32 GAIN under uniform/sqrt rank splits
    decay_robust = {}
    for decay in args.decays:
        p = derive_per_rank(args.top1, args.topW, args.W, decay)
        FL8 = score_tree(build_linear(8), p)[0]
        d = {}
        for M in (16, 32):
            Fdp = build_sequoia_tree(p, M, args.max_depth, args.max_branch)[1]
            cB = real.step_mult(M, g)
            d[f"M{M}"] = {
                "F_dp": Fdp, "ET_ratio": Fdp / FL8,
                "tps_gain_real": (Fdp / FL8) / cB,
                "proj_tps_real": FRONTIER_TPS_LINEAR_M8 * (Fdp / FL8) / cB,
            }
        decay_robust[decay] = d

    # GEMM-share sensitivity on the M=16/32 gains
    share_sens = {}
    for gs in GEMM_SHARE_SWEEP:
        share_sens[f"g{gs}"] = {
            f"M{M}": {
                "cost_mult": real.step_mult(M, gs),
                "tps_gain": pick[M]["ET_ratio_dp_vs_lin8"] / real.step_mult(M, gs),
                "proj_tps": FRONTIER_TPS_LINEAR_M8 * pick[M]["ET_ratio_dp_vs_lin8"]
                / real.step_mult(M, gs),
            } for M in (16, 32)
        }

    # Base-acceptance sensitivity -- the dominant axis (#49: gain shrinks as top-1
    # rises). #68 notes the deployed chain emits ~3.8 tok/step, which under a pure
    # geometric linear chain would imply top-1 ~ 0.775 (Sum_{i<8} x^i = 3.8) > the
    # measured 0.6792 -- so this brackets how much higher real acceptance erodes the
    # tree gain. topW tracks top-1 by the measured rescue ratio (as in #49).
    rescue_ratio = (args.topW - args.top1) / (1.0 - args.top1)
    accept_sens = {}
    for top1 in (0.6792, 0.74, 0.78):
        topW = min(0.999, top1 + rescue_ratio * (1.0 - top1))
        p = derive_per_rank(top1, topW, args.W, "geom")
        FL8 = score_tree(build_linear(8), p)[0]
        d = {"top1": top1, "topW": topW, "F_linear8_implied_tok": FL8}
        for M in (16, 32):
            Fdp = build_sequoia_tree(p, M, args.max_depth, args.max_branch)[1]
            cB = real.step_mult(M, g)
            d[f"M{M}"] = {"F_dp": Fdp, "ET_ratio": Fdp / FL8,
                          "tps_gain_real": (Fdp / FL8) / cB,
                          "proj_tps_real": FRONTIER_TPS_LINEAR_M8 * (Fdp / FL8) / cB}
        accept_sens[f"top1_{top1}"] = d

    # ---- 4. assemble verdict / hand-off ----
    def gain(M, model="real"):
        r = next(x for x in rows if x["M"] == M)
        key = "tps_dp_realcost" if model == "real" else "tps_dp_oldcost"
        return r[key] / FRONTIER_TPS_LINEAR_M8

    primary = gain(opt_real["M"], "real") - 1.0  # treeshape_opt_proj_tps_gain_real_costcurve
    verdict = {
        "primary_metric_name": "treeshape_opt_proj_tps_gain_real_costcurve",
        "treeshape_opt_proj_tps_gain_real_costcurve": primary,
        "opt_M_realcost": opt_real["M"],
        "opt_proj_tps_realcost": opt_real["tps_dp_realcost"],
        "opt_M_oldcost_49": opt_old["M"],
        "opt_proj_tps_oldcost_49": opt_old["tps_dp_oldcost"],
        "optimum_shifted_vs_49": opt_real["M"] != opt_old["M"],
        "M16_proj_tps_real": pick[16]["tps_dp_realcost"],
        "M16_gain_real": gain(16, "real") - 1.0,
        "M32_proj_tps_real": pick[32]["tps_dp_realcost"],
        "M32_gain_real": gain(32, "real") - 1.0,
        "M32_gain_old_49": gain(32, "old") - 1.0,
        "linear_own_opt_M": lin_opt["M"],
        "linear_own_opt_proj_tps": lin_opt["tps_linear_realcost"],
        "hard_M_cap": HARD_M_CAP,
        "M33_craters_to_tps": projected_tps(
            next(x for x in rows if x["M"] == 33)["F_dp"], real.step_mult(33, g)),
        "gemm_share_g": g,
        "mc_max_rel_err": max(v["rel_err"] for v in validation.values()),
        "M32_gain_real_top1_074": accept_sens["top1_0.74"]["M32"]["tps_gain_real"] - 1.0,
        "M32_gain_real_top1_078": accept_sens["top1_0.78"]["M32"]["tps_gain_real"] - 1.0,
    }

    handoff = {
        "consumer": "land #71 (tree-verify serving path)",
        "build_first": {
            "primary_target_M32": {
                "M": 32, "proj_tps": pick[32]["tps_dp_realcost"],
                "gain_vs_deployed_linear": gain(32, "real") - 1.0,
                "E_T": validation["M32"]["F_path_product"],
                "shape": validation["M32"]["shape"],
                "why": "tile-top of Marlin tile-2 (M<=32 cap); max E[T] within the cliff",
            },
            "smaller_first_target_M16": {
                "M": 16, "proj_tps": pick[16]["tps_dp_realcost"],
                "gain_vs_deployed_linear": gain(16, "real") - 1.0,
                "E_T": validation["M16"]["F_path_product"],
                "shape": validation["M16"]["shape"],
                "why": "tile-top of Marlin tile-1; cheapest verify, simplest first build",
            },
        },
        "avoid_widths": {
            "M12": "stops 4 rows short of the M=16 tile-top (rows 13-16 ~9us/row, near free)",
            "M24": "paid the M=17 tile step but stops 8 rows short of M=32 (rows 25-32 ~9us/row)",
            "M33_plus": "+53% Marlin tile cliff -- hard ceiling, craters TPS",
        },
    }

    results = {
        "config": vars(args),
        "anchor": {"deployed_linear_M8_tps": FRONTIER_TPS_LINEAR_M8,
                   "F_linear_M8": F_LINEAR_M8, "hard_M_cap": HARD_M_CAP},
        "real_gemm_curve_68": {str(M): real.gemm_us[M] for M in real.M},
        "budget_sweep": rows,
        "recommended_topologies": {str(M): topo[M] for M in (16, 32)},
        "validation": validation,
        "decay_robustness": decay_robust,
        "gemm_share_sensitivity": share_sens,
        "base_acceptance_sensitivity": accept_sens,
        "verdict": verdict,
        "handoff_land71": handoff,
    }

    # ---- 5. console summary ----
    print("\n[treeshape] ===== TPS-optimal tree-shape under #68 REAL cost curve =====",
          flush=True)
    print(f"{'M':>4} {'F_dp':>7} {'ET/lin8':>8} {'cB(real)':>9} {'cA(old)':>8} "
          f"{'TPS_real':>9} {'TPS_old':>8} {'gain%':>7}", flush=True)
    for r in rows:
        if r["M"] > HARD_M_CAP + 2:
            break
        star = " *" if r["M"] == opt_real["M"] else ("  16" if r["M"] == 16 else "")
        gn = (r["tps_dp_realcost"] / FRONTIER_TPS_LINEAR_M8 - 1.0) * 100 \
            if r["tps_dp_realcost"] else float("nan")
        tr = f"{r['tps_dp_realcost']:9.1f}" if r["tps_dp_realcost"] else f"{'CLIFF':>9}"
        to = f"{r['tps_dp_oldcost']:8.1f}" if r["tps_dp_oldcost"] else f"{'--':>8}"
        print(f"{r['M']:4d} {r['F_dp']:7.4f} {r['ET_ratio_dp_vs_lin8']:8.4f} "
              f"{r['cost_mult_real_B']:9.4f} {r['cost_mult_old_A']:8.4f} {tr} {to} "
              f"{gn:6.1f}%{star}", flush=True)
    print(f"\n[treeshape] VERDICT (real #68 cost curve, g={g}):", flush=True)
    print(f"  TPS-optimal: M={opt_real['M']}  -> {opt_real['tps_dp_realcost']:.1f} TPS "
          f"(+{primary*100:.1f}% vs deployed linear {FRONTIER_TPS_LINEAR_M8})", flush=True)
    print(f"  vs #49 old-cost optimum: M={opt_old['M']}  "
          f"(shifted={verdict['optimum_shifted_vs_49']})", flush=True)
    print(f"  secondary build target: M=16 -> {pick[16]['tps_dp_realcost']:.1f} TPS "
          f"(+{(gain(16,'real')-1)*100:.1f}%)", flush=True)
    print(f"  linear own-optimum: M={lin_opt['M']} -> {lin_opt['tps_linear_realcost']:.1f} "
          f"TPS (linear saturates; tree breaks the geometric ceiling)", flush=True)
    print(f"  M=33 craters to {verdict['M33_craters_to_tps']:.1f} TPS (tile cliff)", flush=True)
    print(f"  accept-sensitivity M=32 gain: top1=0.6792 +{primary*100:.1f}%  "
          f"top1=0.74 +{verdict['M32_gain_real_top1_074']*100:.1f}%  "
          f"top1=0.78 +{verdict['M32_gain_real_top1_078']*100:.1f}%", flush=True)
    print(f"  MC validation max rel-err: {verdict['mc_max_rel_err']:.4%}", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"[treeshape] wrote {args.output}", flush=True)

    if not args.no_wandb:
        try:
            log_wandb(args, results, rows, verdict, validation)
        except Exception as e:  # noqa: BLE001
            print(f"[treeshape] W&B logging failed (non-fatal): {e!r}", flush=True)
    print("[treeshape] DONE", flush=True)


def log_wandb(args, results, rows, verdict, validation):
    import wandb
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     group=args.wandb_group, name=args.wandb_name, job_type="profiling",
                     config={"top1": args.top1, "topW": args.topW, "W": args.W,
                             "gemm_share": args.gemm_share, "hard_M_cap": HARD_M_CAP,
                             "anchor_tps": FRONTIER_TPS_LINEAR_M8})
    run.summary.update({k: v for k, v in verdict.items() if not isinstance(v, (dict, list))})
    cols = ["M", "F_dp", "F_linear", "ET_ratio_dp_vs_lin8", "cost_mult_real_B",
            "cost_mult_old_A", "gemm_ratio_68", "tps_dp_realcost", "tps_dp_oldcost",
            "tps_linear_realcost", "feasible_M_le_32"]
    tbl = wandb.Table(columns=cols)
    for r in rows:
        tbl.add_data(*[r.get(c) for c in cols])
    run.log({"treeshape_budget_table": tbl})
    run.finish()
    print(f"[treeshape] W&B run: {run.url}", flush=True)


if __name__ == "__main__":
    main()
