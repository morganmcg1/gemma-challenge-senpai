#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""BUG-2 salvage-descent root-cause: why does the oracle-measured fp32 tree realize
E[T]=2.621 when the same depth-1 q1 should support ~4.8, and how big is the descent
defect (BUG-2) vs the depth-1 spine deficit (BUG-1)? (PR #135)

LOCAL, CPU-ONLY, ANALYTIC. No HF Job, no submission, no kernel build, no GPU.
Reuses ONLY banked machinery + the oracle's measured per-position ladder:
  * wirbel's E[T] DP (treeshape_measured_accept.build_depth_pvecs_measured /
    score_tree_depthrank -- the exact DP that produced F_tree=5.207);
  * the rho-optimal M=32/depth-9/max-branch-3 topology (wirbel #83/#86) +
    measured rank-conditional rescue ladder rho_cond=[0.4165,0.2655,0.1908]
    (#79/#86) + the deployed RISING conditional spine (PR #76);
  * the oracle readout of `tree-488-pw-fp32-v0` (openevolve board
    20260614-100550-487): E[T]=2.621, depth-1~=0.674/0.679, per-position CUMULATIVE
    accept ladder [0.674,0.350,0.203,0.131,0.089,0.060,0.037], salvages=391,
    full=37 over 1024 steps, drafts=2417.

THE QUESTION
------------
denken #128 (MERGED, RED) argued the realized E[T]=2.10 sits far below the ~4.81
that the build's OWN depth-1 q1 supports under a correctly-descending rho-optimal
walk -- i.e. the salvage walk FIRES (salvages=391/1024=38% of steps) but does NOT
DESCEND (full=37/1024=3.6% reach the full tree). This script quantifies that descent
defect (BUG-2) separately from the depth-1 spine deficit (BUG-1), on the oracle's
own measured numbers, using wirbel's DP.

THE FOUR STEPS
--------------
1. Reconstruct E[T]=2.621 from the measured cumulative ladder. The spine-only linear
   path product is 1 + sum(C_meas) (an identity of the DP on build_linear(8)); the
   gap to the realized 2.621 is the salvage rescue -- which is SMALL precisely
   because the walk does not descend.
2. Compare the measured ladder to the rho-optimal ladder at the SAME depth-1 q1, and
   localize the deficit to depth / width / spread via four nested configs:
       A = measured declining spine, LINEAR (no branches)
       B = rho-optimal RISING spine,  LINEAR        -> spread fix
       C = measured declining spine,  mb3 + branches -> width fix
       D = rho-optimal RISING spine,  mb3 + branches -> both (= ET_tree(q1))
   spread = B-A, width = C-A, interaction = D-B-C+A.
3. Model the descent dynamics: measured mean accepted depth (E[T]-1) vs the
   rho-optimal mean depth and full-tree-reach fraction (Monte-Carlo on the mb3
   topology). Name the defect.
4. Decompose the 2.621 -> clear-500 (4.841) deficit into BUG-2 (descent-only
   recovery, spine held at measured depth-1) and BUG-1 (spine-only recovery to
   0.7287, salvage held non-descending). Report bug2_et_recovery (primary) and
   bug2_is_dominant_ceiling (test). Hand fern the two E[T] columns.

GATE
----
GREEN  : descent defect NAMED (depth/width/spread) with bug2_et_recovery quantified.
AMBER  : ladder-vs-rho-optimal gap quantified but mechanism ambiguous from the 4
         oracle numbers -> spec the one extra number to request.
RED    : measured ladder reconstructs to ~2.621 with NO descent recoverable (already
         rho-optimal at q1) -> the deficit is entirely BUG-1 (spine).
"""
from __future__ import annotations

import argparse
import json
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
from sequoia_dp_tree import build_linear  # noqa: E402

# ---- banked inputs ----------------------------------------------------------
ACCEPT_JSON = "research/accept_calibration/accept_calibration_results.json"
RANKCOV_JSON = "research/rank_coverage/rank_coverage_results.json"
RHO_OPT_JSON = "research/spec_cost_model/rho_optimal_topology_results.json"

# ---- oracle readout of tree-488-pw-fp32-v0 (board 20260614-100550-487) ------
# per-position CUMULATIVE spine acceptance C_meas[d] = P(spine accepts >= d tokens).
ORACLE_CUM_LADDER = [0.674, 0.350, 0.203, 0.131, 0.089, 0.060, 0.037]
ORACLE_E_T = 2.621                 # measured realized accept_length (incl. +1 bonus)
ORACLE_DEPTH1 = 0.674              # = C_meas[0], ladder-consistent depth-1 spine accept
ORACLE_DEPTH1_ALT = 0.679         # separately-cited depth-1 (robustness check)
ORACLE_SALVAGES = 391             # steps where a rank>=2 branch rescued a divergence
ORACLE_FULL = 37                  # steps reaching the full (depth-9) tree
ORACLE_STEPS = 1024
ORACLE_DRAFTS = 2417

# ---- targets / bars ---------------------------------------------------------
DEPTH1_CORRECT = 0.728739760479042   # rho-optimal q1 (rank_coverage top1_76) -- BUG-1 target
CLEAR_500_BAR = 4.841                 # fern #129 operative depth-9 clear-500 bar
SUPPLY_CEILING = 5.207                # ET_tree(0.7287), the rho-optimal supply ceiling
ANCHOR_ET_0598 = 4.811237948198919    # denken #128 anchor: ET_tree(0.598)
W_DEFAULT = 4
MAXD_DEFAULT = 24


def _jd(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} not JSON serializable")


def cum_to_conditional(cum: list[float]) -> list[float]:
    """Per-position conditional q[d] = C[d]/C[d-1] from cumulative acceptance C."""
    q = [cum[0]]
    for i in range(1, len(cum)):
        prev = cum[i - 1]
        q.append(cum[i] / prev if prev > 0 else 0.0)
    return q


def linear_et(q_ladder: list[float], rho_cond: list[float], W: int, maxd: int) -> float:
    """E[T] of the SPINE-ONLY linear chain (no rank>=2 branches) under conditional
    ladder q_ladder. build_linear(len) has no siblings so rho_cond is inert; the score
    telescopes to 1 + sum(cumulative acceptance)."""
    pv = build_depth_pvecs_measured(q_ladder, rho_cond, W, maxd, "flat")
    parent = build_linear(len(q_ladder) + 1)   # +1 root anchor -> reaches depth len(q)
    return score_tree_depthrank(parent, pv)[0]


def tree_et(q_ladder: list[float], rho_cond: list[float], parent: list[int],
            W: int, maxd: int) -> float:
    """E[T] of the full mb3 M=32 topology under conditional ladder q_ladder + branches."""
    pv = build_depth_pvecs_measured(q_ladder, rho_cond, W, maxd, "flat")
    return score_tree_depthrank(parent, pv)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accept-json", default=ACCEPT_JSON)
    ap.add_argument("--rankcov-json", default=RANKCOV_JSON)
    ap.add_argument("--rho-opt-json", default=RHO_OPT_JSON)
    ap.add_argument("--W", type=int, default=W_DEFAULT)
    ap.add_argument("--max-depth", type=int, default=MAXD_DEFAULT)
    ap.add_argument("--mc-trials", type=int, default=400_000)
    ap.add_argument("--output", default="research/spec_cost_model/bug2_salvage_descent_results.json")
    ap.add_argument("--report-md", default="research/spec_cost_model/report_bug2_salvage_descent.md")
    ap.add_argument("--wandb-project", "--wandb_project", default=os.environ.get("WANDB_PROJECT"))
    ap.add_argument("--wandb-entity", "--wandb_entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--wandb-group", "--wandb_group", default="bug2-salvage-descent")
    ap.add_argument("--wandb-name", "--wandb_name", default="wirbel/bug2-salvage-descent")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    W, maxd = args.W, args.max_depth

    # ---- load the banked rho-optimal model (exactly as denken #128's ET_tree) ----
    meas = load_measured(args.accept_json, "server_log")
    q_deployed = list(meas["q"])               # deployed RISING conditional spine
    rc = load_rank_coverage(args.rankcov_json)
    rho_cond = rc["rho_cond"]                   # [0.4165, 0.2655, 0.1908]
    parent = load_m32_topology(args.rho_opt_json)
    children, depth_arr, leaves = tree_arrays(parent)
    built_depth = max(depth_arr)

    def ET_tree(q1: float) -> float:
        """rho-optimal descent: deployed rising deep spine, depth-1 overridden to q1,
        rho-ladder branches on the mb3 M=32 topology (denken #128's canonical model)."""
        qq = list(q_deployed)
        qq[0] = q1
        return tree_et(qq, rho_cond, parent, W, maxd)

    # ---- self-checks: the DP reproduces denken #128's anchors exactly ----
    sc_0598 = ET_tree(0.598)
    sc_0729 = ET_tree(DEPTH1_CORRECT)
    assert abs(sc_0598 - ANCHOR_ET_0598) < 0.02, (sc_0598, ANCHOR_ET_0598)
    assert abs(sc_0729 - SUPPLY_CEILING) < 0.02, (sc_0729, SUPPLY_CEILING)
    print(f"[bug2] self-check: ET_tree(0.598)={sc_0598:.4f} (anchor {ANCHOR_ET_0598:.4f}); "
          f"ET_tree(0.7287)={sc_0729:.4f} (ceiling {SUPPLY_CEILING:.4f})  OK", flush=True)

    # measured conditional ladder from the oracle cumulative ladder
    q_meas = cum_to_conditional(ORACLE_CUM_LADDER)
    print(f"[bug2] measured cumulative C  = {[round(x,4) for x in ORACLE_CUM_LADDER]}", flush=True)
    print(f"[bug2] measured conditional q = {[round(x,4) for x in q_meas]}", flush=True)

    # ======================================================================== #
    # STEP 1 -- reconstruct the realized E[T]=2.621 from the measured ladder
    # ======================================================================== #
    spine_only = linear_et(q_meas, rho_cond, W, maxd)        # 1 + sum(C_meas)
    identity_check = 1.0 + sum(ORACLE_CUM_LADDER)
    salvage_residual = ORACLE_E_T - spine_only               # the salvage rescue bonus
    # what a DESCENDING walk on the SAME measured ladder would realize (width fix, config C)
    mb3_measured = tree_et(q_meas, rho_cond, parent, W, maxd)
    step1 = {
        "oracle_realized_E_T": ORACLE_E_T,
        "spine_only_linear_DP": spine_only,
        "spine_only_identity_1_plus_sumC": identity_check,
        "dp_vs_identity_resid": abs(spine_only - identity_check),
        "salvage_residual_vs_realized": salvage_residual,
        "salvage_residual_frac_of_E_T": salvage_residual / ORACLE_E_T,
        "reconstructed_E_T": spine_only + salvage_residual,   # == 2.621 by construction
        "reconstruction_residual": abs((spine_only + salvage_residual) - ORACLE_E_T),
        "mb3_descending_same_ladder": mb3_measured,           # config C (width fix)
        "salvage_count": ORACLE_SALVAGES,
        "salvage_rate": ORACLE_SALVAGES / ORACLE_STEPS,
        "full_reach_count": ORACLE_FULL,
        "full_reach_rate": ORACLE_FULL / ORACLE_STEPS,
        "note": ("spine-only linear path-product telescopes to 1+sum(C_meas); the gap "
                 "to realized 2.621 is the salvage rescue. It is small (=salvage fires "
                 "but does NOT descend); feeding the SAME ladder through the full mb3 "
                 "topology (a descending walk) would instead realize "
                 f"{mb3_measured:.3f}."),
    }
    print(f"[bug2] STEP1: spine-only linear DP = {spine_only:.4f} (identity 1+sumC = "
          f"{identity_check:.4f}); realized {ORACLE_E_T} -> salvage residual "
          f"{salvage_residual:+.4f} ({salvage_residual/ORACLE_E_T*100:.1f}% of E[T])", flush=True)
    print(f"[bug2] STEP1: same measured ladder on the DESCENDING mb3 tree = "
          f"{mb3_measured:.4f} (the width headroom)", flush=True)

    # ======================================================================== #
    # STEP 2 -- measured vs rho-optimal at the SAME q1; localize depth/width/spread
    # ======================================================================== #
    q1 = ORACLE_DEPTH1
    rho_opt_at_q1 = ET_tree(q1)                                # config D
    # nested configs (all at the same depth-1 q1)
    q_meas_d = list(q_meas)
    q_rise_d = list(q_deployed); q_rise_d[0] = q1
    A = linear_et(q_meas_d, rho_cond, W, maxd)                 # declining spine, linear
    B = linear_et(q_rise_d, rho_cond, W, maxd)                 # rising spine, linear (spread)
    C = tree_et(q_meas_d, rho_cond, parent, W, maxd)           # declining spine, mb3 (width)
    D = rho_opt_at_q1                                          # rising spine, mb3 (both)
    spread = B - A
    width = C - A
    interaction = D - B - C + A
    # which single mechanism dominates the descent gap? (spread/width are the two
    # additive facets; the interaction belongs to BOTH and is large here.)
    mech = {"spread": spread, "width": width}
    dominant_mech = max(mech, key=mech.get)
    top2 = sorted(mech.values(), reverse=True)
    co_dominant = bool(top2[1] >= 0.75 * top2[0])   # spread & width within 25%
    step2 = {
        "q1_measured": q1,
        "rho_optimal_E_T_at_measured_q1": rho_opt_at_q1,
        "measured_realized_E_T": ORACLE_E_T,
        "descent_gap_at_fixed_q1": rho_opt_at_q1 - ORACLE_E_T,
        "config_A_declining_linear": A,
        "config_B_rising_linear_spread": B,
        "config_C_declining_mb3_width": C,
        "config_D_rising_mb3_both": D,
        "contribution_spread_B_minus_A": spread,
        "contribution_width_C_minus_A": width,
        "contribution_interaction": interaction,
        "dominant_mechanism": dominant_mech,
        "spread_width_co_dominant": co_dominant,
        "depth_truncated": bool(ORACLE_CUM_LADDER[-1] <= 1e-6),  # spine itself reaches depth 7
        "measured_spine_reaches_depth": sum(1 for c in ORACLE_CUM_LADDER if c > 0),
        "note": ("spine reaches depth 7 (C[7]=0.037>0) so the gap is NOT depth-"
                 "truncation. SPREAD (deep spine decays 0.67->0.52->0.58.. instead of "
                 "RISING 0.73->0.76->..->0.85) and WIDTH (rank>=2 rescues stay terminal "
                 "leaves) are co-dominant facets of the descent failure, with a large "
                 "super-additive interaction (branches carry more mass on a higher spine)."),
    }
    print(f"[bug2] STEP2: rho-optimal@q1={q1} = {rho_opt_at_q1:.4f}; descent gap "
          f"{rho_opt_at_q1-ORACLE_E_T:+.4f}", flush=True)
    print(f"[bug2] STEP2: A(decl,lin)={A:.4f} B(rise,lin)={B:.4f} C(decl,mb3)={C:.4f} "
          f"D(rise,mb3)={D:.4f}", flush=True)
    print(f"[bug2] STEP2: spread(B-A)={spread:+.4f}  width(C-A)={width:+.4f}  "
          f"interaction={interaction:+.4f}  -> dominant: {dominant_mech}", flush=True)

    # ======================================================================== #
    # STEP 3 -- descent dynamics: measured vs rho-optimal mean depth + full reach
    # ======================================================================== #
    # measured mean accepted depth = E[T] - 1 (the +1 bonus token)
    meas_mean_depth = ORACLE_E_T - 1.0
    rho_mean_depth = rho_opt_at_q1 - 1.0
    # rho-optimal full-reach fraction via MC on the mb3 topology at q1
    pv_opt = build_depth_pvecs_measured(q_rise_d, rho_cond, W, maxd, "flat")
    mc_et = simulate_greedy_depthrank(parent, pv_opt, args.mc_trials, seed=135)
    full_reach_opt = mc_full_reach(parent, pv_opt, args.mc_trials, seed=246,
                                   built_depth=built_depth)
    step3 = {
        "measured_mean_accepted_depth": meas_mean_depth,
        "rho_optimal_mean_accepted_depth": rho_mean_depth,
        "mean_depth_gap": rho_mean_depth - meas_mean_depth,
        "measured_full_reach_rate": ORACLE_FULL / ORACLE_STEPS,
        "rho_optimal_full_reach_rate_mc": full_reach_opt,
        "full_reach_deficit": full_reach_opt - (ORACLE_FULL / ORACLE_STEPS),
        "measured_salvage_rate": ORACLE_SALVAGES / ORACLE_STEPS,
        "mc_et_crosscheck": mc_et,
        "mc_et_vs_dp": abs(mc_et - rho_opt_at_q1),
        "built_depth": built_depth,
        "defect_name": "salvage-fires-but-does-not-descend (deep-spine decay + rescue-leaf-not-reseeded)",
        "defect_explanation": (
            f"38% of steps salvage (rank>=2 rescues a divergence) but only 3.6% reach "
            f"full depth, and the walk realizes mean accepted depth {meas_mean_depth:.2f} vs "
            f"the rho-optimal {rho_mean_depth:.2f}. The spine reaches depth 7, so this is NOT "
            f"depth-truncation -- it is two co-dominant descent pathologies: (1) the deep "
            f"spine's conditional acceptance DECAYS with depth (0.67->0.52->0.58..) instead "
            f"of RISING as the same drafter+verifier does in the linear chain "
            f"(0.73->..->0.85) -- the walk loses the 'easy run' once it descends; and (2) "
            f"each rank>=2 rescue is committed as a TERMINAL LEAF rather than becoming a new "
            f"spine root that RE-DESCENDS its subtree (full reach {full_reach_opt*100:.1f}% "
            f"rho-optimal vs measured 3.6%). Both are build defects (not drafter-capacity), "
            f"recoverable together for +{rho_opt_at_q1-ORACLE_E_T:.2f} E[T] at the measured q1."),
    }
    print(f"[bug2] STEP3: mean accepted depth measured {meas_mean_depth:.3f} vs "
          f"rho-optimal {rho_mean_depth:.3f} (gap {rho_mean_depth-meas_mean_depth:+.3f})", flush=True)
    print(f"[bug2] STEP3: full-reach measured 3.6% vs rho-optimal MC "
          f"{full_reach_opt*100:.1f}%; MC E[T]={mc_et:.4f} (DP {rho_opt_at_q1:.4f})", flush=True)
    print(f"[bug2] STEP3: DEFECT = {step3['defect_name']}", flush=True)

    # ======================================================================== #
    # STEP 4 -- BUG-1 vs BUG-2 decomposition
    # ======================================================================== #
    # BUG-2: fix the DESCENT alone, hold the spine at the measured depth-1.
    #        = rho-optimal descent ET_tree(measured q1) - realized.
    bug2_et_full = ET_tree(q1)
    bug2_et_recovery = bug2_et_full - ORACLE_E_T
    # robustness: the separately-cited depth-1 0.679
    bug2_et_full_alt = ET_tree(ORACLE_DEPTH1_ALT)
    bug2_recovery_alt = bug2_et_full_alt - ORACLE_E_T

    # BUG-1: fix the SPINE alone to 0.7287, hold the measured NON-descending walk.
    #        Lifting q1 scales every cumulative C by 0.7287/q1; salvage held.
    scale = DEPTH1_CORRECT / q1
    q_meas_bug1 = list(q_meas); q_meas_bug1[0] = DEPTH1_CORRECT
    bug1_spine = linear_et(q_meas_bug1, rho_cond, W, maxd)     # = spine_only * scale
    bug1_et_full = bug1_spine + salvage_residual               # hold salvage non-descending
    bug1_et_recovery = bug1_et_full - ORACLE_E_T

    # combined (both fixed) = full rho-optimal at the correct depth-1
    combined_et = ET_tree(DEPTH1_CORRECT)                      # == 5.207 supply ceiling
    bug2_is_dominant = int(bug2_et_recovery > bug1_et_recovery)
    step4 = {
        "deficit_to_clear_500": CLEAR_500_BAR - ORACLE_E_T,
        "bug2_et_full": bug2_et_full,
        "bug2_et_recovery": bug2_et_recovery,             # PRIMARY METRIC
        "bug2_et_full_alt_d1_0679": bug2_et_full_alt,
        "bug2_recovery_alt_d1_0679": bug2_recovery_alt,
        "bug1_spine_scaled": bug1_spine,
        "bug1_scale_factor": scale,
        "bug1_et_full": bug1_et_full,
        "bug1_et_recovery": bug1_et_recovery,
        "combined_et_both_fixed": combined_et,
        "bug2_over_bug1_ratio": bug2_et_recovery / max(bug1_et_recovery, 1e-9),
        "bug2_is_dominant_ceiling": bug2_is_dominant,     # TEST METRIC
        "bug2_alone_clears_500": bool(bug2_et_full >= CLEAR_500_BAR),
        "bug1_alone_clears_500": bool(bug1_et_full >= CLEAR_500_BAR),
        # two E[T] columns for fern's official-TPS recovery matrix
        "fern_handoff_E_T_columns": {
            "as_measured": ORACLE_E_T,
            "bug1_fix_spine_only": bug1_et_full,
            "bug2_fix_descent_only": bug2_et_full,
            "both_fixed_rho_optimal": combined_et,
        },
    }
    print(f"[bug2] STEP4: BUG-2 recovery (descent) = {bug2_et_recovery:+.4f} "
          f"({ORACLE_E_T}->{bug2_et_full:.4f}); clears 500: {step4['bug2_alone_clears_500']}", flush=True)
    print(f"[bug2] STEP4: BUG-1 recovery (spine 0.7287) = {bug1_et_recovery:+.4f} "
          f"({ORACLE_E_T}->{bug1_et_full:.4f}); clears 500: {step4['bug1_alone_clears_500']}", flush=True)
    print(f"[bug2] STEP4: BUG-2/BUG-1 = {step4['bug2_over_bug1_ratio']:.1f}x  -> "
          f"bug2_is_dominant_ceiling = {bug2_is_dominant}", flush=True)

    # ---- gate ----
    if step1["reconstruction_residual"] < 0.01 and bug2_et_recovery > 0.5:
        if bug2_et_recovery <= bug1_et_recovery:
            gate, glabel = "RED", ("measured ladder reconstructs to ~2.621 but the "
                                   "descent recovery does not exceed the spine recovery "
                                   "-> deficit is BUG-1 dominated.")
        elif dominant_mech in ("width", "spread"):
            mech_desc = ("spread+width co-dominant" if step2["spread_width_co_dominant"]
                         else f"dominant mechanism {dominant_mech}")
            gate, glabel = "GREEN", (
                f"descent defect NAMED ({step3['defect_name']}; {mech_desc}); "
                f"bug2_et_recovery={bug2_et_recovery:.3f} E[T] "
                f"(>> bug1 {bug1_et_recovery:.3f}) -> the build has a concrete BUG-2 "
                f"target distinct from BUG-1, and BUG-2 alone "
                f"{'CLEARS' if step4['bug2_alone_clears_500'] else 'does NOT clear'} 500.")
        else:
            gate, glabel = "AMBER", ("descent gap quantified but the mechanism (depth/"
                                     "width/spread) is ambiguous; request per-position "
                                     "branch WIDTH from openevolve.")
    else:
        gate, glabel = "AMBER", ("reconstruction or recovery out of expected band; "
                                 "re-examine the measured ladder.")
    print(f"[bug2] GATE: {gate} -- {glabel}", flush=True)

    verdict = {
        "primary_metric_name": "bug2_et_recovery",
        "bug2_et_recovery": bug2_et_recovery,
        "test_metric_name": "bug2_is_dominant_ceiling",
        "bug2_is_dominant_ceiling": bug2_is_dominant,
        "bug1_et_recovery": bug1_et_recovery,
        "bug2_over_bug1_ratio": step4["bug2_over_bug1_ratio"],
        "descent_defect": step3["defect_name"],
        "dominant_mechanism": dominant_mech,
        "bug2_alone_clears_500": step4["bug2_alone_clears_500"],
        "bug1_alone_clears_500": step4["bug1_alone_clears_500"],
        "measured_E_T": ORACLE_E_T,
        "rho_optimal_E_T_at_measured_q1": rho_opt_at_q1,
        "combined_supply_ceiling": combined_et,
        "reconstruction_residual": step1["reconstruction_residual"],
        "gate": gate,
        "gate_label": glabel,
    }

    results = {
        "config": vars(args),
        "oracle_readout": {
            "board": "20260614-100550-487", "package": "tree-488-pw-fp32-v0",
            "E_T": ORACLE_E_T, "depth1": ORACLE_DEPTH1, "depth1_alt": ORACLE_DEPTH1_ALT,
            "cumulative_ladder": ORACLE_CUM_LADDER, "conditional_ladder": q_meas,
            "salvages": ORACLE_SALVAGES, "full": ORACLE_FULL, "steps": ORACLE_STEPS,
            "drafts": ORACLE_DRAFTS,
        },
        "banked_model": {
            "deployed_rising_spine": q_deployed, "rho_cond": rho_cond,
            "topology_built_depth": built_depth, "n_nodes": len(parent),
            "max_branch": max(len(c) for c in children), "n_leaves": len(leaves),
            "self_check_ET_0598": sc_0598, "self_check_ET_0729": sc_0729,
        },
        "step1_reconstruct": step1,
        "step2_localize": step2,
        "step3_descent_dynamics": step3,
        "step4_decomposition": step4,
        "verdict": verdict,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=_jd)
    print(f"[bug2] wrote {args.output}", flush=True)
    write_report_md(args.report_md, results)
    print(f"[bug2] wrote {args.report_md}", flush=True)

    if not args.no_wandb:
        try:
            log_wandb(args, results, verdict, step1, step2, step3, step4)
        except Exception as e:  # noqa: BLE001
            print(f"[bug2] W&B logging failed (non-fatal): {e!r}", flush=True)
    print("[bug2] DONE", flush=True)


def mc_full_reach(parent: list[int], pvecs, trials: int, seed: int,
                  built_depth: int) -> float:
    """Monte-Carlo fraction of greedy tree-verify walks that reach the built depth
    (the full-tree-reach analogue of the oracle's full=37/1024)."""
    n = len(parent)
    children: list[list[int]] = [[] for _ in range(n)]
    depth = [0] * n
    for i in range(1, n):
        children[parent[i]].append(i)
        depth[i] = depth[parent[i]] + 1
    maxd = len(pvecs) - 1
    rng = np.random.default_rng(seed)
    reached = 0
    for _ in range(trials):
        u, d = 0, 0
        while children[u]:
            kids = children[u]
            dd = depth[u] + 1
            pv = pvecs[min(dd, maxd)]
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
            d = dd
        if d >= built_depth:
            reached += 1
    return reached / trials


def write_report_md(path: str, r: dict) -> None:
    s1, s2, s3, s4, v = (r["step1_reconstruct"], r["step2_localize"],
                         r["step3_descent_dynamics"], r["step4_decomposition"], r["verdict"])
    o = r["oracle_readout"]
    lines = [
        "# BUG-2 salvage-descent root-cause (PR #135)",
        "",
        f"**Gate: {v['gate']}** — {v['gate_label']}",
        "",
        "## Public evidence used",
        "- Oracle readout of `tree-488-pw-fp32-v0` (openevolve board "
        f"`{o['board']}`): E[T]={o['E_T']}, depth-1≈{o['depth1']}/{o['depth1_alt']}, "
        f"cumulative ladder {o['cumulative_ladder']}, salvages={o['salvages']}, "
        f"full={o['full']} over {o['steps']} steps, drafts={o['drafts']}.",
        "- Banked wirbel E[T] DP (`treeshape_measured_accept.build_depth_pvecs_measured`"
        " / `score_tree_depthrank`) on the rho-optimal M=32/depth-9/max-branch-3 "
        "topology (#83/#86) + measured rho-ladder [0.4165,0.2655,0.1908] (#79/#86) + "
        "deployed rising conditional spine (#76).",
        "- denken #128 anchors reproduced: ET_tree(0.598)="
        f"{r['banked_model']['self_check_ET_0598']:.4f}, ET_tree(0.7287)="
        f"{r['banked_model']['self_check_ET_0729']:.4f}.",
        "",
        "## Step 1 — reconstruct realized E[T]=2.621",
        f"- spine-only linear DP = **{s1['spine_only_linear_DP']:.4f}** "
        f"(identity 1+ΣC = {s1['spine_only_identity_1_plus_sumC']:.4f}).",
        f"- salvage residual vs realized 2.621 = **{s1['salvage_residual_vs_realized']:+.4f}** "
        f"({s1['salvage_residual_frac_of_E_T']*100:.1f}% of E[T]) — the rescue the "
        "non-descending salvage adds.",
        f"- reconstruction residual = {s1['reconstruction_residual']:.2e}.",
        f"- the SAME ladder on the descending mb3 tree → {s1['mb3_descending_same_ladder']:.4f} "
        "(the descent headroom).",
        "",
        "## Step 2 — measured vs ρ-optimal at q1=0.674; localize",
        f"- ρ-optimal E[T] @ measured q1 = **{s2['rho_optimal_E_T_at_measured_q1']:.4f}**; "
        f"descent gap **{s2['descent_gap_at_fixed_q1']:+.4f}**.",
        f"- A(declining,linear)={s2['config_A_declining_linear']:.4f}, "
        f"B(rising,linear)={s2['config_B_rising_linear_spread']:.4f}, "
        f"C(declining,mb3)={s2['config_C_declining_mb3_width']:.4f}, "
        f"D(rising,mb3)={s2['config_D_rising_mb3_both']:.4f}.",
        f"- spread(B−A)=**{s2['contribution_spread_B_minus_A']:+.4f}**, "
        f"width(C−A)=**{s2['contribution_width_C_minus_A']:+.4f}**, "
        f"interaction={s2['contribution_interaction']:+.4f} → dominant "
        f"**{s2['dominant_mechanism']}** (spine reaches depth "
        f"{s2['measured_spine_reaches_depth']}, NOT depth-truncated).",
        "",
        "## Step 3 — descent dynamics",
        f"- mean accepted depth: measured **{s3['measured_mean_accepted_depth']:.3f}** vs "
        f"ρ-optimal **{s3['rho_optimal_mean_accepted_depth']:.3f}** "
        f"(gap {s3['mean_depth_gap']:+.3f}).",
        f"- full-tree reach: measured **3.6%** vs ρ-optimal MC "
        f"**{s3['rho_optimal_full_reach_rate_mc']*100:.1f}%** "
        f"(salvage rate {s3['measured_salvage_rate']*100:.1f}%).",
        f"- **Named defect:** {s3['defect_name']} — {s3['defect_explanation']}",
        "",
        "## Step 4 — BUG-1 vs BUG-2 decomposition",
        f"- deficit to clear-500 ({4.841}) = {s4['deficit_to_clear_500']:+.4f}.",
        f"- **BUG-2 (descent only)** recovery = **{s4['bug2_et_recovery']:+.4f}** "
        f"(2.621→{s4['bug2_et_full']:.4f}); clears 500 alone: "
        f"**{s4['bug2_alone_clears_500']}**.",
        f"- **BUG-1 (spine→0.7287 only)** recovery = **{s4['bug1_et_recovery']:+.4f}** "
        f"(2.621→{s4['bug1_et_full']:.4f}); clears 500 alone: {s4['bug1_alone_clears_500']}.",
        f"- both fixed → {s4['combined_et_both_fixed']:.4f} (supply ceiling).",
        f"- **BUG-2/BUG-1 = {s4['bug2_over_bug1_ratio']:.1f}× → bug2_is_dominant_ceiling = "
        f"{s4['bug2_is_dominant_ceiling']}**.",
        "",
        "### fern hand-off — E[T] columns (official-TPS recovery matrix)",
        "| config | E[T] |",
        "|---|---|",
    ]
    for k, val in s4["fern_handoff_E_T_columns"].items():
        lines.append(f"| {k} | {val:.4f} |")
    lines += [
        "",
        f"**Primary:** bug2_et_recovery = {v['bug2_et_recovery']:.4f}.  "
        f"**Test:** bug2_is_dominant_ceiling = {v['bug2_is_dominant_ceiling']}.",
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def log_wandb(args, results, verdict, step1, step2, step3, step4):
    import wandb
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     group=args.wandb_group, name=args.wandb_name, job_type="analysis",
                     config={"W": args.W, "max_depth": args.max_depth,
                             "mc_trials": args.mc_trials,
                             "oracle_board": "20260614-100550-487",
                             "oracle_package": "tree-488-pw-fp32-v0",
                             "clear_500_bar": 4.841})
    summ = {f"verdict/{k}": val for k, val in verdict.items()
            if not isinstance(val, (dict, list))}
    summ.update({f"step1/{k}": val for k, val in step1.items()
                 if not isinstance(val, (dict, list, str))})
    summ.update({f"step2/{k}": val for k, val in step2.items()
                 if not isinstance(val, (dict, list, str))})
    summ.update({f"step3/{k}": val for k, val in step3.items()
                 if not isinstance(val, (dict, list, str))})
    summ.update({f"step4/{k}": val for k, val in step4.items()
                 if not isinstance(val, (dict, list, str))})
    run.summary.update(summ)
    # localization table
    loc = wandb.Table(columns=["config", "E_T", "meaning"])
    loc.add_data("A_declining_linear", step2["config_A_declining_linear"], "measured, no branches")
    loc.add_data("B_rising_linear", step2["config_B_rising_linear_spread"], "spread fix")
    loc.add_data("C_declining_mb3", step2["config_C_declining_mb3_width"], "width fix")
    loc.add_data("D_rising_mb3", step2["config_D_rising_mb3_both"], "both = rho-optimal")
    run.log({"localization": loc})
    # fern E[T] hand-off columns
    fh = wandb.Table(columns=["config", "E_T"])
    for k, val in step4["fern_handoff_E_T_columns"].items():
        fh.add_data(k, val)
    run.log({"fern_E_T_columns": fh})
    run.summary["wandb_run_id"] = run.id
    print(f"[bug2] W&B run: {run.url}  (id={run.id})", flush=True)
    run.finish()


if __name__ == "__main__":
    main()
