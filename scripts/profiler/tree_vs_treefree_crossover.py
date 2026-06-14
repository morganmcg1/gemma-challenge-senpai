#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Tree-vs-tree-free CROSSOVER + build-milestone ladder gate (PR #106) -- the
CAPSTONE synthesis of fern #100 (forward composition), fern #102 (inverse
break-even), denken #101 (recoverable band) and denken #105 (tree-free ceiling).

WHY THIS GATE EXISTS
--------------------
fern #100 (MERGED, GREEN) priced the tree's composed official-TPS landscape;
fern #102 (MERGED, AMBER) inverted it for the MINIMUM realized tree accept_length
E[T]* that clears 500 (break-even 4.624 alone). Together they fully characterize
the tree path IN ISOLATION: it clears 500 iff realized E[T] >= 4.624, and it is a
REGRESSION below 4.45 (ties linear) -- and the as-built build delivers only 2.097
(denken #101: a FIXABLE build defect, recoverable band [3.844, 5.207]).

But the tree no longer stands alone. denken #105 is pricing the build-COMPLETE
tree-FREE stack (SplitK #84 + LK #95 + double-quant #104, NO tree). The decision
the fleet now needs is the CROSSOVER:

    At what realized tree-E[T] does the tree path OVERTAKE the best tree-free
    lever stack -- and is continuing the (build-blocked, AMBER) tree build worth
    it given how much TPS the tree-free path already delivers?

Below the crossover the tree-free path wins and the build is not worth
continuing; above it the tree is the win. This gate computes the crossover as a
function of denken #105's ceiling, lays the build-milestone ladder official(E[T])
with the load-bearing ship-gates marked, and converts "the tree is AMBER and
build-blocked" into "here is exactly when it is worth continuing vs when to
pivot to the tree-free path."

THE MODEL (faithful to #100 and #102 by construction)
-----------------------------------------------------
The #100 forward model is official_TPS = K_cal * (E[T] / step_time) * tau, LINEAR
in the accept_length numerator E[T]. For the tree the step_time is an M=32
TOPOLOGY fact -- independent of how many drafted tokens are accepted. So:

  tree_official(E[T]) = K_cal * (E[T] * lk_factor) / step_time_tree * tau
                      = compose(levers, p).official * (E[T] / E_T_TREE)     [Step 1]

i.e. an exact linear rescaling of #100's value at E[T]=5.207. The CROSSOVER with a
tree-free ceiling C is the E[T] that makes tree_official(E[T]) = C:

  E[T]_x(C) = E_T_TREE * C / compose(levers, p).official                    [Step 2]
            = C * step_time / (K_cal * lk_factor * tau)

which is EXACTLY fern #102's break-even inversion with target=C instead of 500.
We REUSE `tree_et_breakeven.breakeven_raw_et(levers, p, target=C)` verbatim, so
the crossover is faithful to #102 (and through it to #100). At C=500 the crossover
collapses to #102's break-even 4.624 -- the synthesis is consistent by
construction.

WHAT WE PUT ON THE AXES
-----------------------
  E[T] axis gates (PR-named): 2.097 (as-built regression ~227) | 3.844 (denken
    #101 structural floor, still a TPS loss ~416) | 4.45 (ties linear 481.53,
    the abort line) | 4.624 (clears 500 alone, fern #102) | 5.207 (analytical
    ceiling, ~563).
  tree-free ceiling C: PARAMETERIZED (denken #105 lands it; sweep 470..565 until
    then). A rough INDEPENDENT bracket from fern's OWN #100 linear-chain levers
    (splitk+lk +/- double-quant; NOT denken's branch) marks the likely region.
  denken #101 recoverable band [3.844, 5.207]: placed against the crossover --
    does a PLAUSIBLY-RECOVERED tree clear it?

PRIMARY metric  tree_vs_treefree_crossover_ET  (E[T]_x as a function of C)
TEST    metric  build_milestone_ladder         (official(E[T]) with gates marked)

PR #111 EXTENSION -- settle the headline at the LANDED ceiling + post-500 climb ROI
-----------------------------------------------------------------------------------
denken #105 LANDED the tree-free ceiling at C = 518.1 central [496.8, 540.8]
(SplitK 4.44% threshold). Two follow-ups close out the synthesis:

  STEP 1  Settle the crossover headline at the landed C=518.1 (no longer a sweep):
          500 <= 518.1 < 540.7 -> AMBER, the tree is UPSIDE not the critical path;
          the build must recover realized E[T] >= the crossover to overtake tree-
          free. The landed BAND spans the whole verdict spectrum -- the
          conservative ceiling corner 496.8 (< 500) FLIPS GREEN (tree-free can't
          hit 500 -> tree critical), the optimistic corner 540.8 sits on the
          xover~5.0 RED edge -> the AMBER call rests on confidence in the tree-free
          CENTRAL ~518 (denken's ship-readiness PR pins it).

  STEP 2  The post-500 climb lever-ROI map (PRIMARY): now that tree-free locks ~500
          and the ceiling is 556, rank the 500->556 climb levers by official-TPS-
          per-unit-build-effort. Each lever's central ΔofficialTPS is its marginal
          gain on the realized tree-free 500-lock (SplitK 4.44%, LK 1.010, double-
          quant banked, realized tau=0.96), priced via the SAME #100 composition
          model; effort is a T-shirt S/M/L. Tests denken #105's claim that tau->1.00
          is ~3x cheaper than any other margin lever.

  PR #111 PRIMARY metric  post500_top_lever_roi_rank  (the #1 lever by ΔTPS/effort)
  PR #111 TEST    metric  climb_to_ceiling_tps_at_realistic_stack (top-2 ROI levers)

GATE (Step 3, central headline; parameterized on denken #105's ceiling C)
-------------------------------------------------------------------------
  GREEN / tree clearly worth it   C < 500 (tree-free CANNOT hit 500 alone) AND
        crossover comfortably < 4.624 -> the tree is on the critical path for 500
        and beats tree-free at a reachable E[T] -> keep the build the #1 priority.
  AMBER / conditional             C >= 500 (denken #105 GREEN) but the tree
        overtakes only if recovery pushes past the crossover (4.624..5.0) -> 500
        is reachable tree-free; the tree is UPSIDE worth continuing only if the
        build recovers most of the way -> name the recovery threshold.
  RED / deprioritize the tree     crossover >= 5.0 (tree barely beats tree-free
        even near its own ceiling) -> the build-blocked tree is not worth the
        build risk; pivot to the tree-free levers + escalate for a fresh
        accept-length lever class. Deep-RED if crossover > 5.207 (the tree can
        NEVER beat tree-free, even at its analytical ceiling).

LOCAL, CPU-ONLY, ANALYTIC. No GPU, no vLLM, no HF Job, no submission, no
served-file change. A projection model computes nothing served -> greedy
identity untouched by construction.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# Reuse the PR #100 forward model + PR #102 inverse verbatim so the crossover is
# guaranteed consistent with the two gates it synthesizes.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lever_composition import (  # noqa: E402
    E_T_LINEAR,
    E_T_TREE,
    FRONTIER_OFFICIAL,
    K_CAL,
    TARGET_OFFICIAL,
    compose,
    point,
)
from tree_et_breakeven import (  # noqa: E402
    ANALYTICAL_ET,
    BYTESHARK_ASBUILT_ET,
    LINEAR_CHAIN_ET,
    breakeven_raw_et,
)

CORNERS = ("conservative", "central", "optimistic")

# denken #101 recoverable accept_length band [floor, ceiling] (re-measure-pending).
REC_BAND = (E_T_LINEAR, E_T_TREE)   # [3.844, 5.207]

# The two tree paths we cross against the tree-free ceiling:
#   PRIMARY  = tree ALONE (matches the PR's official(E[T]) ladder + the gate
#              anchors 4.45/4.624; the honest "is the bare tree worth the entire
#              tree-free stack" bar).
#   COMPANION= tree + splitk + lk (the build-realistic marginal-tree decision: if
#              the fleet keeps the tree it also builds the compounding levers, so
#              the tree gets their help -> a lower, more tree-favorable crossover).
TREE_PATHS = [("tree_alone", {"tree"}),
              ("tree_full_stack", {"tree", "splitk", "lk"})]

# denken #105 tree-free ceiling: PARAMETERIZED sweep until it lands. Covers the
# PR's 480..540 plus the deep-RED boundary (the tree's own central ceiling ~563).
TREEFREE_CEILING_SWEEP = [470, 480, 490, 500, 507, 510, 511, 520,
                          530, 540, 550, 560, 563, 565]

# ---------------------------------------------------------------------------
# PR #111 -- denken #105 LANDED tree-free ceiling (consumed, no longer swept).
# ---------------------------------------------------------------------------
LANDED_C_CENTRAL = 518.1                  # denken #105 central tree-free ceiling
LANDED_C_BAND = (496.8, 540.8)            # [conservative, optimistic] corners
DENKEN_CEILING = 556.0                    # denken #105 climb ceiling [533, 581]
DENKEN_CEILING_BAND = (533.0, 581.0)

# The realized tree-free 500-lock operating point we climb FROM (denken #105):
SPLITK_LOCK_500 = 0.0444                  # SplitK speedup that locks 500 (#105)
LK_LOCK_500 = 1.010                       # LK prediction-channel central (#95)
DOUBLE_QUANT_CENTRAL = 0.0075             # wirbel #104 banked byte saving (post-mult)
TAU_REALIZED_LOW = 0.96                   # realized local->official gap low (lawine #99)

# climb-lever upgrade targets (the 500->556 candidates the PR names).
SPLITK_UPGRADES = (0.085, 0.12)           # ubel #84: 8.5% then 12% (M<=32 cliff)
LK_RERANK_HIGH = 1.024                    # fern #95 prediction-channel upside
SCALE_PALETTE_CENTRAL = 0.006             # wirbel scale-palette byte-lever (~0.3-1%)
TREE_RECOVERY_ET = 4.7                    # land #71 tree-recovery accept_length target

# T-shirt build-effort -> ROI denominator (doubling scale; one size ~= 2x the prior).
EFFORT_POINTS = {"S": 1, "M": 2, "L": 4}


# ----------------------------------------------------------------------------
# tree official-TPS at an arbitrary realized accept_length (exact #100 rescale).
# ----------------------------------------------------------------------------
def tree_official_at_et(et: float, levers: set[str], p: dict) -> float:
    """official-TPS of a tree lever-stack at a GIVEN raw tree accept_length `et`,
    holding step_time at its M=32 topology value. Exact linear rescaling of #100's
    compose value at E[T]=5.207 -> faithful by construction (lk folded in)."""
    return compose(levers, p)["official_tps"] * (et / E_T_TREE)


def crossover_et(ceiling: float, levers: set[str], p: dict) -> float:
    """The realized tree accept_length E[T]_x at which this tree stack's official
    TPS equals the tree-free `ceiling`. Identical to fern #102's break-even with
    target=ceiling (REUSED verbatim) -> the crossover IS the generalized #102."""
    return breakeven_raw_et(levers, p, target=ceiling)["breakeven_raw_et"]


# ----------------------------------------------------------------------------
# Rough INDEPENDENT tree-free bracket from fern's OWN merged #100 linear-chain
# levers (NOT denken's unmerged branch). Marks the likely region pending #105.
# ----------------------------------------------------------------------------
# wirbel #104 double-quant verify-GEMM scales band (+0.4..1.1%); central ~0.75%.
DOUBLE_QUANT_PCT = {"conservative": 0.004, "central": 0.0075, "optimistic": 0.011}


def treefree_bracket_own_model() -> dict:
    """tree-FREE official-TPS (no tree) from #100's splitk+lk on the linear chain,
    +/- double-quant #104. A SANITY bracket for denken #105's authoritative number
    (which also folds in double-quant properly). persist ~0 at the conservative
    corner (denken #97: only 2.17% reclaimable idle) -> excluded from the floor."""
    base = {c: compose({"splitk", "lk"}, point(c))["official_tps"] for c in CORNERS}
    with_dq = {c: base[c] * (1.0 + DOUBLE_QUANT_PCT[c]) for c in CORNERS}
    return {"splitk_lk": base, "splitk_lk_double_quant": with_dq,
            "note": ("fern's OWN #100 linear-chain levers; rough bracket only -- "
                     "denken #105 is the authoritative tree-free ceiling lane")}


# ----------------------------------------------------------------------------
# STEP 1 -- the build-milestone ladder official(E[T]).
# ----------------------------------------------------------------------------
def build_milestone_ladder() -> dict:
    """official_TPS as a function of realized tree E[T] over 2.097..5.207 at the
    three corners (tree ALONE -- the PR's primary curve), with the load-bearing
    ship-gates marked. Returns a machine-readable ladder the build team can use to
    set intermediate ship-gates."""
    pts = {c: point(c) for c in CORNERS}

    # dense curve for plotting / interpolation
    grid = sorted(set(np.round(np.arange(BYTESHARK_ASBUILT_ET, E_T_TREE + 1e-9, 0.1), 4))
                  | {BYTESHARK_ASBUILT_ET, E_T_LINEAR, E_T_TREE})
    curve = []
    for et in grid:
        curve.append({"E_T": float(et),
                      **{f"{c}_official": tree_official_at_et(et, {"tree"}, pts[c])
                         for c in CORNERS}})

    # FIXED-E[T] gates (official varies by corner).
    fixed_gates = [
        ("asbuilt", BYTESHARK_ASBUILT_ET,
         "byteshark as-built tok/step -- accept-collapse REGRESSION"),
        ("linear_floor", E_T_LINEAR,
         "denken #101 structural floor (linear-chain accept_length) -- still a TPS LOSS"),
        ("analytical_ceiling", E_T_TREE,
         "analytical ceiling (de-risked, fern #92)"),
    ]
    # THRESHOLD-E[T] gates (E[T] varies by corner; official is fixed by definition).
    #   beat_linear  = E[T] at which tree official ties the linear frontier 481.53.
    #   breakeven500 = E[T] at which tree official hits the 500 target (fern #102).
    ladder_gates = {}
    for key, et, desc in fixed_gates:
        ladder_gates[key] = {
            "kind": "fixed_E_T", "E_T": et, "desc": desc,
            "official": {c: tree_official_at_et(et, {"tree"}, pts[c]) for c in CORNERS},
        }
    ladder_gates["beat_linear"] = {
        "kind": "threshold_official", "target_official": FRONTIER_OFFICIAL,
        "desc": "ties linear 481.53 official -- the ABORT line (below it the tree is a regression)",
        "E_T": {c: crossover_et(FRONTIER_OFFICIAL, {"tree"}, pts[c]) for c in CORNERS},
    }
    ladder_gates["breakeven_500"] = {
        "kind": "threshold_official", "target_official": TARGET_OFFICIAL,
        "desc": "clears the 500 target ALONE (fern #102 break-even)",
        "E_T": {c: crossover_et(TARGET_OFFICIAL, {"tree"}, pts[c]) for c in CORNERS},
    }

    # ordered ship-gate ladder (central E[T], ascending) for the build team.
    ship_gates = [
        {"order": 1, "gate": "beat-linear", "E_T_central": ladder_gates["beat_linear"]["E_T"]["central"],
         "rule": "first ship-gate: realized E[T] >= ~4.45 or the tree is a NET REGRESSION vs linear 481.53"},
        {"order": 2, "gate": "clear-500", "E_T_central": ladder_gates["breakeven_500"]["E_T"]["central"],
         "rule": "target ship-gate: realized E[T] >= ~4.62 clears 500 with the bare tree (fern #102)"},
        {"order": 3, "gate": "analytical-ceiling", "E_T_central": E_T_TREE,
         "rule": "stretch: realized E[T] -> 5.207 gives ~563 central (the de-risked ceiling)"},
    ]
    return {"curve_tree_alone": curve, "gates": ladder_gates, "ship_gates": ship_gates}


# ----------------------------------------------------------------------------
# STEP 2 -- the tree-vs-tree-free crossover as a function of denken #105's ceiling.
# ----------------------------------------------------------------------------
def crossover_curve() -> dict:
    """E[T]_x(C) for the PRIMARY (tree-alone) and COMPANION (tree+splitk+lk) paths,
    swept over the parameterized tree-free ceiling C, at the three corners. Places
    denken #101's recoverable band [3.844, 5.207] against each crossover."""
    pts = {c: point(c) for c in CORNERS}
    be500 = {name: crossover_et(TARGET_OFFICIAL, lv, pts["central"])
             for name, lv in TREE_PATHS}

    rows = []
    for C in TREEFREE_CEILING_SWEEP:
        row = {"treefree_ceiling": float(C),
               "treefree_clears_500": C >= TARGET_OFFICIAL}
        for name, lv in TREE_PATHS:
            xc = {c: crossover_et(C, lv, pts[c]) for c in CORNERS}
            row[name] = {
                "crossover_E_T": xc,
                # is the crossover reachable inside denken #101's band [3.844,5.207]?
                "reachable_within_rec_band_central": xc["central"] <= REC_BAND[1],
                "above_500_breakeven_central": xc["central"] > be500[name],
                # the tree's official at its band ceiling minus the tree-free C
                # (>0 => a fully-recovered tree beats this tree-free ceiling).
                "margin_at_band_ceiling_central":
                    tree_official_at_et(REC_BAND[1], lv, pts["central"]) - C,
            }
        rows.append(row)

    # the slope d(crossover)/dC (central, tree-alone) + the three threshold Cs.
    slope_central = (crossover_et(600.0, {"tree"}, pts["central"])
                     - crossover_et(500.0, {"tree"}, pts["central"])) / 100.0
    off_tree_central = compose({"tree"}, pts["central"])["official_tps"]  # @5.207

    def ceiling_for_crossover(target_xc: float) -> float:
        # invert E[T]_x(C) = E_T_TREE*C/off  ->  C = target_xc*off/E_T_TREE
        return target_xc * off_tree_central / E_T_TREE

    thresholds = {
        "C_treefree_just_clears_500": TARGET_OFFICIAL,
        "C_crossover_equals_5p0_central": ceiling_for_crossover(5.0),
        "C_crossover_equals_ceiling_5p207_central": ceiling_for_crossover(E_T_TREE),
        "interpretation": {
            "below_500": "tree-free CANNOT hit 500 -> tree on the critical path (GREEN region)",
            "500_to_C5p0": "tree-free clears 500; tree is upside, overtakes only if E[T]>=crossover in [4.624,5.0] (AMBER region)",
            "C5p0_to_C5p207": "crossover in [5.0,5.207] -> tree barely beats tree-free even near its ceiling (RED region)",
            "above_C5p207": "crossover > 5.207 -> tree can NEVER beat tree-free, even at its analytical ceiling (deep-RED)",
        },
    }

    # corner-MATCHED robustness: if denken #105's C is itself banded and we compare
    # corner-for-corner (the same optimism lifts BOTH the tree and tree-free), the
    # crossover is far more stable than the exogenous-scalar sweep suggests.
    own = treefree_bracket_own_model()["splitk_lk_double_quant"]
    corner_matched = {c: crossover_et(own[c], {"tree"}, pts[c]) for c in CORNERS}

    return {"rows": rows, "be500_by_path": be500,
            "slope_dcrossover_dC_central": slope_central,
            "thresholds_central": thresholds,
            "corner_matched_crossover_tree_alone": corner_matched}


# ----------------------------------------------------------------------------
# STEP 3 -- the gate (parameterized on denken #105's ceiling C).
# ----------------------------------------------------------------------------
def gate_for_ceiling(C: float, pts: dict) -> dict:
    """The continue-vs-pivot verdict at a given tree-free ceiling C (central)."""
    xc = crossover_et(C, {"tree"}, pts["central"])
    xc_stack = crossover_et(C, {"tree", "splitk", "lk"}, pts["central"])
    be500 = crossover_et(TARGET_OFFICIAL, {"tree"}, pts["central"])
    if C < TARGET_OFFICIAL:
        verdict, label = "GREEN", (
            f"tree on the CRITICAL PATH for 500 (tree-free central {C:.0f} < 500); "
            f"crossover {xc:.3f} < {be500:.3f} -> keep the tree build the #1 priority")
    elif xc < 5.0:
        verdict, label = "AMBER", (
            f"tree-free clears 500; the tree is UPSIDE -- it overtakes only if the "
            f"build recovers E[T] >= {xc:.3f} (alone) / {xc_stack:.3f} (with splitk+lk), "
            f"i.e. {(xc - REC_BAND[0]) / (REC_BAND[1] - REC_BAND[0]) * 100:.0f}% of the "
            f"way up denken #101's band -> continue only if recovery clears that")
    elif xc <= REC_BAND[1]:
        verdict, label = "RED", (
            f"crossover {xc:.3f} >= 5.0: the tree barely beats tree-free even near "
            f"its own ceiling -> deprioritize the build-blocked tree, pivot to the "
            f"tree-free levers")
    else:
        verdict, label = "RED", (
            f"crossover {xc:.3f} > 5.207: the tree can NEVER beat tree-free, even at "
            f"its analytical ceiling -> pivot to tree-free + escalate for a fresh "
            f"accept-length lever class")
    return {"treefree_ceiling": C, "crossover_tree_alone": xc,
            "crossover_tree_full_stack": xc_stack, "verdict": verdict,
            "verdict_label": label}


# ============================================================================
# PR #111 STEP 1 -- settle the crossover headline at denken #105's LANDED ceiling.
# ============================================================================
def settle_headline_at_landed_C(pts: dict) -> dict:
    """Fix the crossover verdict at the LANDED tree-free ceiling C=518.1 and lay
    the band corner table [496.8, 518.1, 540.8]. The verdict at each corner reuses
    `gate_for_ceiling` (central crossover), so the table shows the full spectrum:
    the conservative ceiling corner (496.8 < 500) FLIPS GREEN, the optimistic
    corner (540.8) sits on the xover~5.0 RED edge, and the AMBER central rests on
    confidence in the tree-free CENTRAL ~518."""
    corner_table = []
    for corner, C in (("conservative_ceiling", LANDED_C_BAND[0]),
                      ("central_landed", LANDED_C_CENTRAL),
                      ("optimistic_ceiling", LANDED_C_BAND[1])):
        g = gate_for_ceiling(C, pts)
        corner_table.append({
            "corner": corner, "treefree_ceiling": C,
            "treefree_clears_500": C >= TARGET_OFFICIAL,
            "crossover_tree_alone": g["crossover_tree_alone"],
            "crossover_tree_full_stack": g["crossover_tree_full_stack"],
            "verdict": g["verdict"]})
    headline = gate_for_ceiling(LANDED_C_CENTRAL, pts)
    rec_alone = headline["crossover_tree_alone"]
    rec_stack = headline["crossover_tree_full_stack"]
    frac_up_band = (rec_alone - REC_BAND[0]) / (REC_BAND[1] - REC_BAND[0])
    return {
        "landed_C_central": LANDED_C_CENTRAL,
        "landed_C_band": list(LANDED_C_BAND),
        "headline_verdict": headline["verdict"],
        "headline_verdict_label": headline["verdict_label"],
        "recovery_gate_E_T_tree_alone": rec_alone,
        "recovery_gate_E_T_tree_full_stack": rec_stack,
        "recovery_frac_up_rec_band": frac_up_band,
        "corner_table": corner_table,
        "interpretation": (
            f"LANDED C=518.1 central -> AMBER: tree-free clears 500, the tree is "
            f"UPSIDE; it overtakes only if realized E[T] >= {rec_alone:.3f} (tree "
            f"alone) / {rec_stack:.3f} (tree+splitk+lk), ~{frac_up_band*100:.0f}% up "
            f"denken #101's recoverable band. The landed BAND spans the spectrum: "
            f"conservative corner 496.8 (<500) is GREEN (tree-free can't hit 500 -> "
            f"tree critical); optimistic corner 540.8 is the xover~5.0 RED edge "
            f"(tree barely beats tree-free even near its ceiling). The AMBER call "
            f"therefore rests on confidence in the tree-free CENTRAL ~518."),
    }


# ============================================================================
# PR #111 STEP 2 -- the post-500 climb lever-ROI map (PRIMARY).
# ============================================================================
def treefree_official_at(splitk_s: float, lk_mult: float, tau: float,
                         double_quant: bool = True, byte_mult: float = 1.0) -> float:
    """tree-FREE composed official-TPS at an explicit (splitk_s, lk_mult, tau)
    operating point, off the central composition point so ONLY the named knobs
    move. double-quant (#104) and the scale-palette byte-lever apply as the same
    post-multipliers fern's #106 own-bracket already uses -> faithful to #100."""
    p = point("central")
    p["splitk_s"], p["lk_mult"], p["tau"] = splitk_s, lk_mult, tau
    o = compose({"splitk", "lk"}, p)["official_tps"]
    if double_quant:
        o *= (1.0 + DOUBLE_QUANT_CENTRAL)
    return o * byte_mult


def post500_lever_roi() -> dict:
    """Rank the 500->556 climb levers by central ΔofficialTPS / build-effort.

    Each lever's ΔTPS is its MARGINAL central official gain on the REALIZED tree-
    free 500-lock baseline (SplitK 4.44%, LK 1.010, double-quant banked, realized
    tau=0.96). The tau lever closes the [0.96,1.00] realization gap; every other
    lever is priced on the SAME realized serve (tau held at 0.96) so the ROI
    comparison is apples-to-apples. Tests denken #105's 'tau ~3x cheaper' claim by
    reporting tau's ROI under BOTH effort readings (local-calibration S vs
    official-anchor M)."""
    # realized 500-lock baseline (the serve we climb from) and the tau-secured point.
    base = treefree_official_at(SPLITK_LOCK_500, LK_LOCK_500, TAU_REALIZED_LOW)
    base_tau_secured = treefree_official_at(SPLITK_LOCK_500, LK_LOCK_500, 1.0)

    def lever(name, d_tps, size, basis, blocked=False, note=""):
        eff = EFFORT_POINTS[size]
        return {"lever": name, "delta_official_tps": d_tps,
                "delta_pct_of_baseline": 100.0 * d_tps / base,
                "effort_tshirt": size, "effort_points": eff,
                "roi_tps_per_effort": d_tps / eff,
                "build_blocked": blocked, "effort_basis": basis, "note": note}

    d_tau = base_tau_secured - base
    levers = [
        lever("tau->1.00", d_tau, "S",
              "local-calibration of the local->official ratio (lawine #99)",
              note=("official-anchor fallback = effort M; the denken-3x test below "
                    "reports ROI under BOTH so the claim is conditional, not assumed")),
        lever("splitk->12%",
              treefree_official_at(SPLITK_UPGRADES[1], LK_LOCK_500, TAU_REALIZED_LOW) - base,
              "M", "verify-GEMM SplitK kernel to the M<=32 tile cliff (ubel #84, in flight)"),
        lever("splitk->8.5%",
              treefree_official_at(SPLITK_UPGRADES[0], LK_LOCK_500, TAU_REALIZED_LOW) - base,
              "M", "verify-GEMM SplitK kernel, intermediate milestone (ubel #84)"),
        lever("lk-rerank->1.024",
              treefree_official_at(SPLITK_LOCK_500, LK_RERANK_HIGH, TAU_REALIZED_LOW) - base,
              "M", "LK prediction-channel re-rank upside if it pans out (fern #95, AMBER)"),
        lever("scale-palette-byte",
              treefree_official_at(SPLITK_LOCK_500, LK_LOCK_500, TAU_REALIZED_LOW,
                                   byte_mult=1.0 + SCALE_PALETTE_CENTRAL) - base,
              "S", "lossless scale-palette packing (wirbel); composes with SplitK"),
        lever("tree-recovery->4.7", DENKEN_CEILING - LANDED_C_CENTRAL, "L",
              "tree drafter build + accept-recovery (land #71); the 518->556 climb",
              blocked=True,
              note=("ΔTPS credits the FULL 518->556 climb (PR framing); my model puts "
                    "tree+splitk+lk @ the literal E[T]=4.7 at ~531 (marginal ~+13 over "
                    "tree-free 518) and needs E[T]~=4.92 to reach 556 -> the tree's ROI "
                    "is the most assumption-sensitive AND build-blocked by the #101 defect")),
    ]
    ranked = sorted(levers, key=lambda x: -x["roi_tps_per_effort"])

    # --- denken #105 'tau ~3x cheaper than any other lever' test ---------------
    tau = next(l for l in levers if l["lever"] == "tau->1.00")
    tau_roi_localcal = tau["roi_tps_per_effort"]                 # S effort
    tau_roi_anchor = tau["delta_official_tps"] / EFFORT_POINTS["M"]  # M effort
    others = [l for l in levers if l["lever"] != "tau->1.00"]
    ratios_localcal = {l["lever"]: tau_roi_localcal / l["roi_tps_per_effort"] for l in others}
    ratios_anchor = {l["lever"]: tau_roi_anchor / l["roi_tps_per_effort"] for l in others}
    # tau is the CLEAR top lever iff its ROI beats the best OTHER buildable lever.
    best_other_buildable = max(
        (l for l in others if not l["build_blocked"]), key=lambda x: x["roi_tps_per_effort"])
    denken_3x = {
        "claim": "denken #105: tau->1.00 is ~3x cheaper than any other margin lever",
        "tau_roi_localcal_S": tau_roi_localcal,
        "tau_roi_official_anchor_M": tau_roi_anchor,
        "ratio_tau_over_each_lever_localcal": ratios_localcal,
        "ratio_tau_over_each_lever_anchor": ratios_anchor,
        "best_other_buildable_lever": best_other_buildable["lever"],
        "best_other_buildable_roi": best_other_buildable["roi_tps_per_effort"],
        "tau_clear_top_if_localcal": tau_roi_localcal > best_other_buildable["roi_tps_per_effort"],
        "tau_ties_best_other_if_anchor":
            tau_roi_anchor <= best_other_buildable["roi_tps_per_effort"] * 1.25,
        "verdict": (
            "CONFIRMED iff tau-realization is local-calibration (S): tau ROI is "
            f"{min(ratios_localcal.values()):.1f}-{max(ratios_localcal.values()):.1f}x "
            "every other lever and beats them all. If it needs an official anchor "
            f"(M), tau ROI only {tau_roi_anchor/best_other_buildable['roi_tps_per_effort']:.2f}x "
            f"the best other lever ({best_other_buildable['lever']}) -> the 3x claim "
            "BREAKS; tau merely co-leads. The claim and the gate hinge on the SAME "
            "unknown -> denken's ship-readiness PR must resolve the tau path."),
    }

    return {"baseline_realized_tau096": base, "baseline_tau_secured": base_tau_secured,
            "levers": levers, "ranked": ranked,
            "top_lever_roi_rank": ranked[0]["lever"],
            "top_buildable_lever":
                next(l["lever"] for l in ranked if not l["build_blocked"]),
            "denken_3x_test": denken_3x}


def climb_realistic_stack(roi: dict) -> dict:
    """climb_to_ceiling_tps_at_realistic_stack (TEST metric): projected official-TPS
    if the fleet executes the top-2 ROI levers, plus the FULL non-tree realistic
    stack (all immediately-buildable levers) -- the honest ceiling without the
    build-blocked tree."""
    # top-2 immediately-buildable ROI levers (exclude the build-blocked tree).
    buildable = [l for l in roi["ranked"] if not l["build_blocked"]]
    top2 = buildable[:2]
    top2_names = [l["lever"] for l in top2]

    # Compose the named stacks on the tau-secured serve (tau->1.00 banked first
    # whenever it is in the chosen set).
    def stack(splitk_s, lk_mult, tau, byte_mult=1.0):
        return treefree_official_at(splitk_s, lk_mult, tau, byte_mult=byte_mult)

    tau_secured = "tau->1.00" in top2_names
    tau_val = 1.0 if tau_secured else TAU_REALIZED_LOW
    splitk_val = (SPLITK_UPGRADES[1] if "splitk->12%" in top2_names else
                  SPLITK_UPGRADES[0] if "splitk->8.5%" in top2_names else SPLITK_LOCK_500)
    lk_val = LK_RERANK_HIGH if "lk-rerank->1.024" in top2_names else LK_LOCK_500
    byte_val = 1.0 + SCALE_PALETTE_CENTRAL if "scale-palette-byte" in top2_names else 1.0
    top2_tps = stack(splitk_val, lk_val, tau_val, byte_val)

    # full non-tree realistic stack: every buildable lever at its upgrade target.
    full_nontree = stack(SPLITK_UPGRADES[1], LK_RERANK_HIGH, 1.0,
                         byte_mult=1.0 + SCALE_PALETTE_CENTRAL)

    # the tree corner (build-blocked): tree+splitk+lk at the recovered E[T]=4.7 and
    # at the analytical ceiling 5.207 -- what the 556 ceiling actually needs.
    pc = point("central")
    tree_stack_full = compose({"tree", "splitk", "lk"}, pc)["official_tps"]
    tree_at_4p7 = tree_stack_full * (TREE_RECOVERY_ET / E_T_TREE)
    # marginal of building the tree on TOP of the tree-free 518 lock, at the literal
    # E[T]=4.7 (not the full 518->556 climb the PR credits the lever) -> ROI floor.
    tree_marginal_4p7_over_518 = tree_at_4p7 - LANDED_C_CENTRAL
    # the realized tree E[T] that the 556 ceiling actually requires on this stack.
    et_for_556 = E_T_TREE * DENKEN_CEILING / tree_stack_full

    return {
        "top2_levers": top2_names,
        "climb_to_ceiling_tps_at_realistic_stack": top2_tps,
        "full_nontree_realistic_stack_tps": full_nontree,
        "nontree_stack_clears_540": full_nontree >= 540.0,
        "ceiling_556_needs_tree": full_nontree < DENKEN_CEILING_BAND[0],
        "tree_recovered_4p7_tps": tree_at_4p7,
        "tree_marginal_4p7_over_treefree_518": tree_marginal_4p7_over_518,
        "tree_E_T_required_for_556_ceiling": et_for_556,
        "tree_full_stack_ceiling_5p207_tps": tree_stack_full,
        "note": (f"top-2 buildable ROI levers ({'+'.join(top2_names)}) reach "
                 f"{top2_tps:.1f}; the full non-tree stack tops out at "
                 f"{full_nontree:.1f} (< denken's 556 lower band {DENKEN_CEILING_BAND[0]:.0f}) "
                 f"-> the central 556 ceiling is tree-gated: tree+splitk+lk reaches "
                 f"{tree_at_4p7:.1f} at the literal E[T]=4.7 (marginal +{tree_marginal_4p7_over_518:.1f} "
                 f"over tree-free 518) and needs E[T]~={et_for_556:.2f} to hit 556; "
                 f"{tree_stack_full:.1f} at the 5.207 ceiling."),
    }


def post500_gate(roi: dict, climb: dict) -> dict:
    """The 500->556 build-allocation gate (PR #111 Step 3).
      GREEN: coherent ROI ranking with a CLEAR top lever (tau local-cal, or
             SplitK) -> hand the fleet the allocation map.
      AMBER: ROI flat (no clear winner under the official-anchor tau reading).
      RED-flag: the full non-tree realistic stack tops out < 540 -> 556 is
             tree-gated / optimistic-corner-only."""
    d3 = roi["denken_3x_test"]
    clear_top_localcal = d3["tau_clear_top_if_localcal"]
    flat_under_anchor = d3["tau_ties_best_other_if_anchor"]
    ceiling_red_flag = not climb["nontree_stack_clears_540"]

    if clear_top_localcal:
        verdict = "GREEN"
        label = (
            f"coherent ROI ranking -- tau->1.00 is the CLEAR #1 lever (ROI "
            f"{d3['tau_roi_localcal_S']:.1f} vs best-other {d3['best_other_buildable_roi']:.1f}); "
            f"build order: tau, then SplitK->12%. denken's 'tau ~3x cheaper' CONFIRMED "
            f"under local-calibration."
            + (f" RED-FLAG: the full non-tree stack tops out at "
               f"{climb['full_nontree_realistic_stack_tps']:.1f} < 540 -> the 556 ceiling "
               f"is tree-gated (build-blocked), not reachable by the cheap levers alone."
               if ceiling_red_flag else ""))
    elif flat_under_anchor:
        verdict = "AMBER"
        label = (
            f"ROI flattens if tau needs an official anchor (M): tau ties "
            f"{d3['best_other_buildable_lever']} -> no single clear winner; the climb "
            f"past 500 is a multi-lever grind.")
    else:
        verdict = "AMBER"
        label = "ROI ranking present but sensitive to the tau-effort reading."

    return {"verdict": verdict, "verdict_label": label,
            "tau_clear_top_localcal": clear_top_localcal,
            "ceiling_red_flag_nontree_below_540": ceiling_red_flag,
            "post500_top_lever_roi_rank": roi["top_lever_roi_rank"],
            "climb_to_ceiling_tps_at_realistic_stack":
                climb["climb_to_ceiling_tps_at_realistic_stack"]}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="research/spec_cost_model/tree_vs_treefree_crossover_results.json")
    ap.add_argument("--treefree-ceiling", type=float, default=LANDED_C_CENTRAL,
                    help="consume denken #105's LANDED ceiling for the HEADLINE verdict "
                         "(default = the landed central 518.1)")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="gemma-challenge-senpai")
    ap.add_argument("--wandb-entity", default="wandb-applied-ai-team")
    ap.add_argument("--wandb-name", default="fern/post500-climb-roi")
    ap.add_argument("--wandb-group", default="post500-climb-roi")
    args = ap.parse_args()

    pts = {c: point(c) for c in CORNERS}

    ladder = build_milestone_ladder()
    crossover = crossover_curve()
    own_bracket = treefree_bracket_own_model()

    # ---- headline verdict ----
    # If denken #105 has landed (--treefree-ceiling), consume it; else use fern's
    # own central bracket as the LIKELY ceiling and flag the verdict pending #105.
    if args.treefree_ceiling is not None:
        likely_C = args.treefree_ceiling
        source = "denken #105 (consumed)"
    else:
        likely_C = own_bracket["splitk_lk_double_quant"]["central"]
        source = "fern's own #100 bracket (PENDING denken #105)"
    headline = gate_for_ceiling(likely_C, pts)
    headline["ceiling_source"] = source

    # gate across the whole sweep (the decision surface).
    gate_surface = [gate_for_ceiling(float(C), pts) for C in TREEFREE_CEILING_SWEEP]

    # ---- PR #111: settle headline at the LANDED C + post-500 climb ROI map ----
    settle = settle_headline_at_landed_C(pts)
    roi = post500_lever_roi()
    climb = climb_realistic_stack(roi)
    post500 = post500_gate(roi, climb)

    # ---- PRIMARY metric: tree_vs_treefree_crossover_ET (as a function of C) ----
    primary = {
        "name": "tree_vs_treefree_crossover_ET",
        "definition": "realized tree E[T] at which tree_official(E[T]) = tree_free_ceiling",
        "at_C_500_tree_alone_central": crossover["be500_by_path"]["tree_alone"],
        "at_likely_ceiling_tree_alone_central": headline["crossover_tree_alone"],
        "likely_ceiling": likely_C,
        "slope_dE_T_per_TPS_central": crossover["slope_dcrossover_dC_central"],
        "thresholds": crossover["thresholds_central"],
        "corner_matched_robust_band": crossover["corner_matched_crossover_tree_alone"],
    }

    out = {
        "gate": {
            "primary_metric_name": "tree_vs_treefree_crossover_ET",
            "test_metric_name": "build_milestone_ladder",
            "headline_verdict": headline["verdict"],
            "headline_verdict_label": headline["verdict_label"],
            "headline_ceiling": likely_C,
            "headline_ceiling_source": source,
            "primary_metric": primary,
            "rule": ("GREEN=C<500 (tree critical path) / AMBER=C>=500 & crossover<5.0 "
                     "(tree upside, name recovery threshold) / RED=crossover>=5.0 "
                     "(tree barely beats tree-free; pivot)"),
            # ---- PR #111 headline metrics (the post-500 build-allocation map) ----
            "pr111_primary_metric_name": "post500_top_lever_roi_rank",
            "pr111_test_metric_name": "climb_to_ceiling_tps_at_realistic_stack",
            "pr111_step1_settled_verdict": settle["headline_verdict"],
            "pr111_step1_recovery_gate_tree_alone": settle["recovery_gate_E_T_tree_alone"],
            "pr111_step1_recovery_gate_tree_full_stack": settle["recovery_gate_E_T_tree_full_stack"],
            "post500_top_lever_roi_rank": post500["post500_top_lever_roi_rank"],
            "climb_to_ceiling_tps_at_realistic_stack": post500["climb_to_ceiling_tps_at_realistic_stack"],
            "pr111_step3_verdict": post500["verdict"],
            "pr111_step3_verdict_label": post500["verdict_label"],
            "pr111_rule": ("GREEN=coherent ROI ranking w/ clear top lever (tau local-cal "
                           "or SplitK) / AMBER=ROI flat under official-anchor tau / "
                           "RED-flag=full non-tree stack tops out <540 (556 tree-gated)"),
        },
        "step1_build_milestone_ladder": ladder,
        "step2_crossover": crossover,
        "step3_gate_surface": gate_surface,
        "pr111_step1_settle_headline_landed_C": settle,
        "pr111_step2_post500_lever_roi": roi,
        "pr111_step2_climb_realistic_stack": climb,
        "pr111_step3_post500_gate": post500,
        "treefree_bracket_own_model": own_bracket,
        "anchors": {
            "frontier_official": FRONTIER_OFFICIAL, "target_official": TARGET_OFFICIAL,
            "E_T_linear": E_T_LINEAR, "E_T_tree": E_T_TREE, "K_cal": K_CAL,
            "byteshark_asbuilt_et": BYTESHARK_ASBUILT_ET, "rec_band": list(REC_BAND),
            "fern_102_breakeven_500_central": crossover["be500_by_path"]["tree_alone"],
        },
        "method": ("CPU-only analytic synthesis of fern #100 (forward) + fern #102 "
                   "(inverse, REUSED verbatim) + denken #101 (recoverable band) + "
                   "denken #105 (tree-free ceiling, PARAMETERIZED). The crossover is "
                   "#102's break-even generalized from target=500 to target=C; at "
                   "C=500 it collapses to 4.624 by construction. No GPU, no served "
                   "change -> greedy identity untouched."),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: o.item() if isinstance(
            o, np.generic) else o)

    # ----------------------------- console -----------------------------
    print("=" * 88)
    print("TREE-vs-TREE-FREE CROSSOVER + BUILD-MILESTONE LADDER (PR #106) -- capstone synthesis")
    print("=" * 88)
    print(f"\nmodel: tree_official(E[T]) = K_cal*(E[T]/step)*tau (LINEAR in E[T]); "
          f"crossover E[T]_x(C) = #102 break-even @ target=C  (K_cal={K_CAL:.3f})")

    print("\n[STEP 1] build-milestone ladder official(E[T]) -- tree ALONE (cons / central / opt):")
    for key in ("asbuilt", "linear_floor", "beat_linear", "breakeven_500", "analytical_ceiling"):
        g = ladder["gates"][key]
        if g["kind"] == "fixed_E_T":
            o = g["official"]
            print(f"  E[T]={g['E_T']:.3f}  {key:20s} -> {o['conservative']:6.1f} / "
                  f"{o['central']:6.1f} / {o['optimistic']:6.1f}   ({g['desc']})")
        else:
            e = g["E_T"]
            print(f"  official={g['target_official']:.2f}  {key:20s} -> E[T]= "
                  f"{e['conservative']:.3f} / {e['central']:.3f} / {e['optimistic']:.3f}   ({g['desc']})")
    print("  ship-gates (build team): "
          + " | ".join(f"{s['gate']} E[T]>={s['E_T_central']:.2f}" for s in ladder["ship_gates"]))

    print("\n[STEP 2] crossover E[T]_x(C) vs tree-free ceiling C  (tree-alone central):")
    print(f"  {'C':>6s}  {'xover(alone)':>12s}  {'xover(+sk+lk)':>13s}  treefree>=500?  reach<=5.207?")
    for r in crossover["rows"]:
        a = r["tree_alone"]; s = r["tree_full_stack"]
        print(f"  {r['treefree_ceiling']:6.1f}  {a['crossover_E_T']['central']:12.3f}  "
              f"{s['crossover_E_T']['central']:13.3f}  "
              f"{str(r['treefree_clears_500']):>13s}  {a['reachable_within_rec_band_central']}")
    th = crossover["thresholds_central"]
    print(f"\n  thresholds (central): tree-free needs the tree below C={th['C_treefree_just_clears_500']:.0f} | "
          f"crossover=5.0 at C={th['C_crossover_equals_5p0_central']:.1f} | "
          f"tree-NEVER-wins above C={th['C_crossover_equals_ceiling_5p207_central']:.1f}")
    cm = crossover["corner_matched_crossover_tree_alone"]
    print(f"  corner-MATCHED crossover (robust; same optimism lifts both): "
          f"{cm['conservative']:.3f} / {cm['central']:.3f} / {cm['optimistic']:.3f}")

    print("\n[fern's own rough tree-free bracket -- PENDING denken #105]")
    b = own_bracket["splitk_lk"]; bd = own_bracket["splitk_lk_double_quant"]
    print(f"  splitk+lk:            cons {b['conservative']:.1f} / central {b['central']:.1f} / opt {b['optimistic']:.1f}")
    print(f"  +double-quant #104:   cons {bd['conservative']:.1f} / central {bd['central']:.1f} / opt {bd['optimistic']:.1f}")

    print(f"\n[STEP 3 / VERDICT] headline ceiling C={likely_C:.1f}  ({source})")
    print(f"  {headline['verdict']} -- {headline['verdict_label']}")
    print("\n  gate surface across the sweep:")
    for g in gate_surface:
        print(f"    C={g['treefree_ceiling']:6.1f} -> {g['verdict']:5s}  "
              f"(crossover alone {g['crossover_tree_alone']:.3f})")

    # ===================== PR #111 -- settle + post-500 ROI =====================
    print("\n" + "=" * 88)
    print("PR #111 -- SETTLE THE HEADLINE AT LANDED C=518.1 + POST-500 CLIMB-ROI MAP")
    print("=" * 88)
    print(f"\n[STEP 1] settled crossover at denken #105's LANDED C={LANDED_C_CENTRAL} "
          f"central [{LANDED_C_BAND[0]}, {LANDED_C_BAND[1]}]:")
    print(f"  HEADLINE: {settle['headline_verdict']} -- recovery gate realized E[T] >= "
          f"{settle['recovery_gate_E_T_tree_alone']:.3f} (tree alone) / "
          f"{settle['recovery_gate_E_T_tree_full_stack']:.3f} (tree+splitk+lk), "
          f"~{settle['recovery_frac_up_rec_band']*100:.0f}% up denken #101's band")
    print(f"  {'corner':22s} {'C':>7s}  {'>=500?':>6s}  {'xover_alone':>11s}  "
          f"{'xover+sk+lk':>11s}  verdict")
    for r in settle["corner_table"]:
        print(f"  {r['corner']:22s} {r['treefree_ceiling']:7.1f}  "
              f"{str(r['treefree_clears_500']):>6s}  {r['crossover_tree_alone']:11.3f}  "
              f"{r['crossover_tree_full_stack']:11.3f}  {r['verdict']}")
    print(f"  => the landed band spans GREEN(496.8)->AMBER(518.1)->RED-edge(540.8); "
          f"the AMBER rests on the tree-free CENTRAL ~518.")

    print(f"\n[STEP 2] post-500 climb lever-ROI (central ΔofficialTPS / build-effort):")
    print(f"  realized 500-lock baseline (tau=0.96) = {roi['baseline_realized_tau096']:.1f} "
          f"| tau-secured = {roi['baseline_tau_secured']:.1f}")
    print(f"  {'rank lever':26s} {'ΔTPS':>7s} {'Δ%':>6s} {'effort':>7s} {'ROI':>7s}  basis")
    for i, l in enumerate(roi["ranked"], 1):
        blk = " [BLOCKED]" if l["build_blocked"] else ""
        print(f"  {i}. {l['lever']:23s} {l['delta_official_tps']:7.2f} "
              f"{l['delta_pct_of_baseline']:5.1f}% {l['effort_tshirt']:>5s}({l['effort_points']}) "
              f"{l['roi_tps_per_effort']:7.2f}  {l['effort_basis'][:38]}{blk}")
    print(f"  PRIMARY post500_top_lever_roi_rank = {roi['top_lever_roi_rank']} "
          f"(top BUILDABLE = {roi['top_buildable_lever']})")

    d3 = roi["denken_3x_test"]
    print(f"\n  [denken '#105 tau ~3x cheaper' TEST]")
    print(f"    tau ROI: local-cal(S)={d3['tau_roi_localcal_S']:.2f}  "
          f"official-anchor(M)={d3['tau_roi_official_anchor_M']:.2f}  "
          f"(best other buildable = {d3['best_other_buildable_lever']} @ "
          f"{d3['best_other_buildable_roi']:.2f})")
    print(f"    tau/other ROI ratio (local-cal): "
          + "  ".join(f"{k}={v:.1f}x" for k, v in
                      d3["ratio_tau_over_each_lever_localcal"].items()))
    print(f"    -> {d3['verdict']}")

    print(f"\n[STEP 2 TEST] climb_to_ceiling_tps_at_realistic_stack:")
    print(f"  top-2 buildable ({'+'.join(climb['top2_levers'])}) -> "
          f"{climb['climb_to_ceiling_tps_at_realistic_stack']:.1f} official")
    print(f"  FULL non-tree realistic stack -> {climb['full_nontree_realistic_stack_tps']:.1f} "
          f"(clears 540? {climb['nontree_stack_clears_540']}; 556 needs tree? "
          f"{climb['ceiling_556_needs_tree']})")
    print(f"  tree corner (build-blocked): tree+splitk+lk @E[T]=4.7 -> "
          f"{climb['tree_recovered_4p7_tps']:.1f} (marginal +{climb['tree_marginal_4p7_over_treefree_518']:.1f} "
          f"over tree-free 518) | needs E[T]~={climb['tree_E_T_required_for_556_ceiling']:.2f} for 556 | "
          f"@5.207 ceiling -> {climb['tree_full_stack_ceiling_5p207_tps']:.1f}")

    print(f"\n[STEP 3 / PR #111 VERDICT] {post500['verdict']}")
    print(f"  {post500['verdict_label']}")

    print(f"\nwrote {args.out}")

    # ------------------------------- W&B -------------------------------
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=args.wandb_name, group=args.wandb_group,
                         job_type="analysis", config={
                             "gate": "tree-vs-treefree-crossover",
                             "method": "cpu-analytic-synthesis-100-102-101-105",
                             "frontier_official": FRONTIER_OFFICIAL,
                             "target_official": TARGET_OFFICIAL,
                             "E_T_linear": E_T_LINEAR, "E_T_tree": E_T_TREE,
                             "K_cal": K_CAL, "rec_band": list(REC_BAND),
                             "headline_ceiling": likely_C,
                             "headline_ceiling_source": source})
        s = wandb.summary
        s["tree_vs_treefree_crossover_ET"] = primary["at_likely_ceiling_tree_alone_central"]
        s["tree_vs_treefree_crossover_ET_at_C500"] = primary["at_C_500_tree_alone_central"]
        s["crossover_slope_dE_T_per_TPS_central"] = primary["slope_dE_T_per_TPS_central"]
        s["C_treefree_just_clears_500"] = th["C_treefree_just_clears_500"]
        s["C_crossover_equals_5p0_central"] = th["C_crossover_equals_5p0_central"]
        s["C_tree_never_wins_above_central"] = th["C_crossover_equals_ceiling_5p207_central"]
        s["corner_matched_crossover_central"] = cm["central"]
        s["treefree_ceiling_own_bracket_central"] = bd["central"]
        s["headline_verdict"] = headline["verdict"]
        s["headline_verdict_label"] = headline["verdict_label"]
        # break-even-500 cross-check (must equal fern #102's 4.624).
        s["crossover_at_C500_equals_fern102_breakeven"] = primary["at_C_500_tree_alone_central"]

        # Step 1 ladder table
        lt = wandb.Table(columns=["E_T", "official_conservative", "official_central",
                                  "official_optimistic"])
        for c in ladder["curve_tree_alone"]:
            lt.add_data(c["E_T"], c["conservative_official"], c["central_official"],
                        c["optimistic_official"])
        wandb.log({"build_milestone_ladder": lt})

        # Step 2 crossover table
        ct = wandb.Table(columns=["treefree_ceiling", "crossover_tree_alone_central",
                                  "crossover_tree_full_stack_central",
                                  "treefree_clears_500", "reachable_within_band",
                                  "verdict"])
        for r, g in zip(crossover["rows"], gate_surface):
            ct.add_data(r["treefree_ceiling"],
                        r["tree_alone"]["crossover_E_T"]["central"],
                        r["tree_full_stack"]["crossover_E_T"]["central"],
                        r["treefree_clears_500"],
                        r["tree_alone"]["reachable_within_rec_band_central"],
                        g["verdict"])
        wandb.log({"crossover_vs_ceiling": ct})

        # gate anchors table (the named milestone gates)
        gt = wandb.Table(columns=["gate", "kind", "E_T_or_target",
                                  "official_or_E_T_central", "desc"])
        for key in ("asbuilt", "linear_floor", "beat_linear", "breakeven_500",
                    "analytical_ceiling"):
            g = ladder["gates"][key]
            if g["kind"] == "fixed_E_T":
                gt.add_data(key, g["kind"], g["E_T"], g["official"]["central"], g["desc"])
            else:
                gt.add_data(key, g["kind"], g["target_official"], g["E_T"]["central"], g["desc"])
        wandb.log({"milestone_gates": gt})

        # ---------------- PR #111 -- settle + post-500 climb-ROI ----------------
        d3 = roi["denken_3x_test"]
        s["pr111_step1_settled_verdict"] = settle["headline_verdict"]
        s["pr111_step1_recovery_gate_tree_alone"] = settle["recovery_gate_E_T_tree_alone"]
        s["pr111_step1_recovery_gate_tree_full_stack"] = settle["recovery_gate_E_T_tree_full_stack"]
        s["pr111_step1_recovery_frac_up_band"] = settle["recovery_frac_up_rec_band"]
        s["post500_top_lever_roi_rank"] = roi["top_lever_roi_rank"]
        s["post500_top_buildable_lever"] = roi["top_buildable_lever"]
        s["post500_tau_roi_localcal_S"] = d3["tau_roi_localcal_S"]
        s["post500_tau_roi_official_anchor_M"] = d3["tau_roi_official_anchor_M"]
        s["post500_tau_clear_top_if_localcal"] = d3["tau_clear_top_if_localcal"]
        s["post500_tau_ties_best_other_if_anchor"] = d3["tau_ties_best_other_if_anchor"]
        s["climb_to_ceiling_tps_at_realistic_stack"] = (
            climb["climb_to_ceiling_tps_at_realistic_stack"])
        s["climb_full_nontree_realistic_stack_tps"] = climb["full_nontree_realistic_stack_tps"]
        s["climb_nontree_stack_clears_540"] = climb["nontree_stack_clears_540"]
        s["climb_ceiling_556_needs_tree"] = climb["ceiling_556_needs_tree"]
        s["climb_tree_recovered_4p7_tps"] = climb["tree_recovered_4p7_tps"]
        s["pr111_verdict"] = post500["verdict"]
        s["pr111_verdict_label"] = post500["verdict_label"]
        s["pr111_ceiling_red_flag_nontree_below_540"] = post500["ceiling_red_flag_nontree_below_540"]

        # Step 1 corner table (the landed band GREEN->AMBER->RED spectrum).
        st = wandb.Table(columns=["corner", "treefree_ceiling", "treefree_clears_500",
                                  "crossover_tree_alone", "crossover_tree_full_stack",
                                  "verdict"])
        for r in settle["corner_table"]:
            st.add_data(r["corner"], r["treefree_ceiling"], r["treefree_clears_500"],
                        r["crossover_tree_alone"], r["crossover_tree_full_stack"],
                        r["verdict"])
        wandb.log({"pr111_landed_band_corner_table": st})

        # Step 2 lever-ROI ranked table (the build-allocation map).
        rt = wandb.Table(columns=["rank", "lever", "delta_official_tps",
                                  "delta_pct_of_baseline", "effort_tshirt",
                                  "effort_points", "roi_tps_per_effort",
                                  "build_blocked", "effort_basis"])
        for i, l in enumerate(roi["ranked"], 1):
            rt.add_data(i, l["lever"], l["delta_official_tps"],
                        l["delta_pct_of_baseline"], l["effort_tshirt"],
                        l["effort_points"], l["roi_tps_per_effort"],
                        l["build_blocked"], l["effort_basis"])
        wandb.log({"pr111_post500_lever_roi": rt})

        # denken-3x ratio table (tau ROI over each other lever, both effort reads).
        dt = wandb.Table(columns=["other_lever", "tau_over_lever_localcal",
                                  "tau_over_lever_anchor"])
        for k in d3["ratio_tau_over_each_lever_localcal"]:
            dt.add_data(k, d3["ratio_tau_over_each_lever_localcal"][k],
                        d3["ratio_tau_over_each_lever_anchor"][k])
        wandb.log({"pr111_denken_3x_ratios": dt})

        print(f"\nW&B run: {run.id}  ({run.url})")
        wandb.finish()


if __name__ == "__main__":
    main()
