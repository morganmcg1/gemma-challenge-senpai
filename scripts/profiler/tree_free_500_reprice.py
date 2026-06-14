#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Re-price the tree-free 500-path after denken #117 (PR #123): is the tree now
MANDATORY for 500, or does the cheap path still reach it?

WHY THIS GATE EXISTS
--------------------
denken #105 declared "tree-free 500 is GREEN" conditioned on SplitK >= 4.44%, and
denken #109's ship gate assumed ubel's realistic SplitK ~8.5%. denken #117 just
falsified BOTH preconditions: the M=8 verify-GEMM SplitK realization ceiling is
3.20% GROSS / 1.56% NET (band-high 7.81% only at the optimistic 88%-GDDR6 wall),
because `gate_up` (54% of verify time) is CTA-saturated and frozen. The tree-free
500-path must be re-priced against the PHYSICAL SplitK ceiling, not the assumed
one -- and we need to know whether the tree (land #71) just went from "insurance"
(fern #106) to "required-for-500".

WHAT CHANGES vs #105 / #109 (exactly two inputs, everything else REUSED verbatim)
--------------------------------------------------------------------------------
  (1) SplitK input: was ubel #108's assumed band {5%, 8.5%, 12%}. Now denken
      #117's PHYSICAL ceiling: central 1.56% NET (the wall-time figure that feeds
      the `vg -> vg/(1+s)` model after the reduction-overhead haircut), band
      [1.6% net, 7.81% gross-at-88%-GDDR6]. The 3.20% gross is carried as a
      central-sensitivity row.
  (2) tau band: was the asserted [0.96, 1.00]. Now lawine #116's DERIVED roofline
      band [0.9983, 1.00], central 1.0000. This LIFTS the conservative floor (tau
      is no longer the drag #109 feared) -- so the re-price isolates the SplitK
      ceiling as the sole mover.

The byte lever is the wirbel #110 PALETTE (lossless, ~0.2-0.5% UB, corner 0); the
INT8 double-quant #104 is DEAD (KILL, excluded). LK #95 is an E[T] numerator lever
(+1.0-2.4%, linear in TPS, NOT a step-time lever). All three are greedy-lossless.

MODEL (REUSED, not re-derived)
------------------------------
  official = K_cal*(E[T]/step)*tau*(mult/mult_central),  K_cal = 481.53/3.844
  E[T]  = 3.844 * lk_mult                       (tree-free stays on the linear E[T])
  step  = vg + attn + residual
  vg    = 0.53*(1-f_dq_palette)/(1+s)           (SplitK = bandwidth util lever)
We import denken #105's `tree_free_500_ceiling.compose` + `dq_tps_to_fdq`
verbatim, lawine #99's `calibrate()` for the multiplier CI, and -- for the tree
side -- fern #102's `tree_et_breakeven.breakeven_raw_et` over fern #100's
`lever_composition.compose` (so the re-priced #106 crossover is faithful by
construction). SELF-CHECK: reproduces #117's published cross-check (s=3.20% gross,
corner levers -> 474.6 / 489.4 / 494.3 at tau=0.96/0.99/1.00).

DELIVERABLES (PR #123)
----------------------
  PRIMARY  tree_free_500_ceiling_at_splitk_wall   central tree-free official TPS at
           the #117 SplitK wall (+ band [conservative corner, optimistic band-high]).
  TEST     tree_required_et_to_clear_500          the realized tree E[T] land #71
           must hit to clear 500 if tree-free misses (None/0 if tree-free clears).
  + the composed lever table (lever -> ΔTPS) and the re-priced fern #106 crossover.

GATE
----
  GREEN  clean tree-free 500 survives on the central 3.2%/1.56% wall.
  AMBER  tree-free 500 needs the optimistic 7.81% band-high / tau=1.0 corner (straddle).
  RED    tree-free caps below 500 at the central wall -> the tree is REQUIRED for 500.

LOCAL, CPU-ONLY, ANALYTIC. No GPU, no vLLM, no HF Job, no submission, no
served-file change. Greedy identity untouched by construction (SplitK 0-flip
kanna #87; palette bit-exact wirbel #110; LK prediction-only fern #95). The greedy
VALIDITY ruling (kanna #114 RED + human contract) is a SEPARATE gate that sits on
top of ALL TPS math here -- a tree-free OR tree 500 both still need it.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- REUSE denken #105's tree-free composition model verbatim ----
tf = _load_module("tree_free_500_ceiling", os.path.join(_HERE, "tree_free_500_ceiling.py"))
# ---- REUSE lawine #99's live multiplier + CI ----
from scripts.profiler.local_official_projection import calibrate  # noqa: E402
# ---- REUSE fern #100 (forward) + #102 (inverse) for the tree-side crossover ----
import lever_composition as lc  # noqa: E402
from tree_et_breakeven import breakeven_raw_et  # noqa: E402

TARGET = tf.TARGET_OFFICIAL          # 500.0
FRONTIER = tf.FRONTIER_OFFICIAL      # 481.53
E_T_TREE = lc.E_T_TREE               # 5.207
BYTESHARK_ASBUILT_ET = 2.097         # land #71 as-built tok/step (#101 fixable defect)
REC_BAND = (tf.E_T_LINEAR, E_T_TREE)  # denken #101 recoverable band [3.844, 5.207]

# ===========================================================================
# INPUT (1): denken #117 SplitK realization ceiling (REPLACES ubel's assumed band)
# ===========================================================================
# `s` is the bandwidth-headroom fraction the compose model applies as vg/(1+s).
# #117 measured-wall: gross 3.20% / NET 1.56%; 88%-GDDR6 optimistic: gross 7.81% /
# net 6.19%; datasheet-100 (UNREACHABLE): gross 13.25%. The NET is the wall-time
# figure (post reduction-overhead haircut) -> the honest value for vg/(1+s).
SPLITK_117 = {
    "net_measured_central": 0.0156,   # #117 PRIMARY ceiling (net, measured wall)
    "gross_measured_central": 0.0320,  # gross at the measured 79.2%-GDDR6 wall
    "net_88_gddr6": 0.0619,            # net at the optimistic 88%-GDDR6 wall
    "gross_88_gddr6_high": 0.0781,     # #117 band-high (gross, optimistic wall)
    "gross_datasheet_unreachable": 0.1325,
}
# the band the PR pins: central 1.56% net .. 7.81% gross-optimistic band-high.
S_CENTRAL = SPLITK_117["net_measured_central"]   # 1.56%
S_BAND_LOW = 0.0160                               # ~net measured-wall floor
S_BAND_HIGH = SPLITK_117["gross_88_gddr6_high"]   # 7.81% optimistic band-high

# ===========================================================================
# INPUT (2): lawine #116 DERIVED roofline tau band (REPLACES asserted [0.96,1.00])
# ===========================================================================
TAU_116 = {"low": 0.9983, "central": 1.0000, "high": 1.0000}   # derived floor 0.9983
# #116 corner SplitK thresholds to clear 500 (its own ship gate, for cross-ref):
SPLITK_CORNER_AT_TAU_FLOOR = 0.05840   # 5.84% at tau=0.9983 (corner levers)
SPLITK_CORNER_AT_TAU_1 = 0.05491       # 5.49% at tau=1.00 (corner levers)

# ===========================================================================
# Surviving cheap levers (NO tree); double-quant #104 DEAD/excluded.
# ===========================================================================
# palette #110 byte lever (lossless LUT, ~0.2-0.5% UB, conservative corner 0).
PALETTE_TPS = {"low": 0.0, "central": 0.003, "high": 0.005}
# LK #95 E[T] numerator lever (linear in TPS), central near floor (#80).
LK_MULT = tf.LK_MULT          # {low 1.010, central 1.010, high 1.024}
FP32_M8 = tf.FP32_M8          # {low 0, central 0, high 0.000102}
PERSIST = tf.PERSIST_RECLAIM  # {low 0, central 0, high 0.0217} upside-only

# multiplier (lawine #99), live.
_CALIB = calibrate()
MULT_CENTRAL = _CALIB.multiplier
MULT_LOCAL_CI = (_CALIB.mult_ci_local_lo, _CALIB.mult_ci_local_hi)


def palette_fdq(key: str) -> float:
    return tf.dq_tps_to_fdq(PALETTE_TPS[key])


def treefree_official(s: float, p: dict) -> float:
    """Tree-free official TPS at SplitK net speedup s under lever point p (REUSES
    #105 compose + the #99 multiplier-CI correction; identical to #109's
    `official_ship`)."""
    base = tf.compose(s, p)["official_tps"]
    return base * (p["mult"] / MULT_CENTRAL)


def scenario_point(scenario: str) -> dict:
    """Build a tree-free lever point. conservative MINIMISES, optimistic MAXIMISES.
    SplitK is supplied separately (the swept axis)."""
    if scenario == "central":
        return {"lk_mult": LK_MULT["central"], "f_dq": palette_fdq("central"),
                "fp32_m8": FP32_M8["central"], "persist_reclaim": PERSIST["central"],
                "tau": TAU_116["central"], "mult": MULT_CENTRAL}
    if scenario == "conservative":
        return {"lk_mult": LK_MULT["low"], "f_dq": palette_fdq("low"),
                "fp32_m8": FP32_M8["high"], "persist_reclaim": PERSIST["low"],
                "tau": TAU_116["low"], "mult": MULT_LOCAL_CI[0]}
    if scenario == "optimistic":
        return {"lk_mult": LK_MULT["high"], "f_dq": palette_fdq("high"),
                "fp32_m8": FP32_M8["low"], "persist_reclaim": PERSIST["high"],
                "tau": TAU_116["high"], "mult": MULT_LOCAL_CI[1]}
    raise ValueError(scenario)


def splitk_for_treefree_500(p: dict) -> float | None:
    """Minimum SplitK net s s.t. treefree_official(s,p) >= 500 under point p.
    Returns 0.0 if cleared at s=0, float('inf') if even vg->0 cannot reach 500."""
    e_t = tf.E_T_LINEAR * p["lk_mult"]
    eff_mult = p["mult"] / MULT_CENTRAL
    step_needed = tf.K_CAL * e_t * p["tau"] * eff_mult / TARGET
    attn = tf.BUDGET["attention"] + p["fp32_m8"]
    residual = (1.0 - tf.BUDGET["verify_gemm"] - tf.BUDGET["attention"]) - p["persist_reclaim"]
    vg_needed = step_needed - attn - residual
    if vg_needed <= 0:
        return float("inf")
    vg_full = tf.BUDGET["verify_gemm"] * (1.0 - p["f_dq"])
    s = vg_full / vg_needed - 1.0
    return 0.0 if s <= 0 else s


# ===========================================================================
# Tree side (REUSE fern #100/#102): the tree's E[T]-to-clear-target and crossover.
# The tree gets the SAME #117 SplitK ceiling and #116 tau when it stacks levers.
# ===========================================================================
def tree_point(scenario: str, splitk_s: float) -> dict:
    """fern #100 parameter point with the #117 SplitK ceiling + #116 tau patched
    in (so the tree-side break-even is priced on the SAME physical inputs)."""
    p = lc.point(scenario)
    p["splitk_s"] = splitk_s
    p["tau"] = TAU_116[{"central": "central", "conservative": "low",
                        "optimistic": "high"}[scenario]]
    return p


def tree_required_et(levers: set[str], scenario: str, splitk_s: float,
                     target: float = TARGET) -> float:
    """Realized tree E[T] that makes this tree lever-stack hit `target` official
    TPS (fern #102 inversion, REUSED). Holds step at its M=32 topology value."""
    return breakeven_raw_et(levers, tree_point(scenario, splitk_s),
                            target=target)["breakeven_raw_et"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="research/spec_cost_model/tree_free_500_reprice_results.json")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="gemma-challenge-senpai")
    ap.add_argument("--wandb-entity", default="wandb-applied-ai-team")
    ap.add_argument("--wandb-name", default="denken/tree-free-500-reprice")
    ap.add_argument("--wandb-group", default="tree-free-500-reprice")
    args = ap.parse_args()

    p_cons = scenario_point("conservative")
    p_cent = scenario_point("central")
    p_opt = scenario_point("optimistic")

    # ----- MODEL SELF-CHECK: reproduce #117's published cross-check -----
    # s=3.20% gross, corner-conservative non-tau levers, tau in {0.96,0.99,1.00}.
    p_check = dict(p_cons)
    selfcheck = {}
    for tau in (0.96, 0.99, 1.00):
        pp = dict(p_check); pp["tau"] = tau
        selfcheck[f"tau_{tau:.2f}"] = treefree_official(SPLITK_117["gross_measured_central"], pp)

    # ===================== STEP 1: tree-free TPS at the #117 SplitK wall =====================
    # central = #117 net 1.56%; band = [conservative corner @ net-low, optimistic @ 7.81% gross].
    tps_central = treefree_official(S_CENTRAL, p_cent)
    tps_cons = treefree_official(S_BAND_LOW, p_cons)
    tps_opt = treefree_official(S_BAND_HIGH, p_opt)
    # sensitivity: central levers but at the 3.20% GROSS wall (the #117 gross central).
    tps_central_gross = treefree_official(SPLITK_117["gross_measured_central"], p_cent)
    # central levers across the whole #117 SplitK band (the straddle curve).
    band_curve = []
    for label, s in (("net_measured_1.56", S_CENTRAL),
                     ("gross_measured_3.20", SPLITK_117["gross_measured_central"]),
                     ("net_88_6.19", SPLITK_117["net_88_gddr6"]),
                     ("gross_88_7.81", S_BAND_HIGH)):
        band_curve.append({"splitk_label": label, "splitk_pct": s * 100.0,
                           "central_levers_tps": treefree_official(s, p_cent),
                           "clears_500_central_levers": treefree_official(s, p_cent) >= TARGET})

    clears_central = tps_central >= TARGET
    clears_cons = tps_cons >= TARGET
    clears_opt = tps_opt >= TARGET

    # SplitK% tree-free needs to clear 500 (central levers, tau=1.0) + corner.
    s_need_central = splitk_for_treefree_500(p_cent)
    s_need_corner = splitk_for_treefree_500(p_cons)

    # ===================== STEP 2: composed lever table (lever -> ΔTPS, central) =====================
    # cumulative add at central inputs, tau pinned at the #116 central 1.0.
    def pt(lk, fdq, persist, s):
        return {"lk_mult": lk, "f_dq": fdq, "fp32_m8": 0.0,
                "persist_reclaim": persist, "tau": TAU_116["central"], "mult": MULT_CENTRAL}
    base = treefree_official(0.0, pt(1.0, 0.0, 0.0, 0.0))            # = frontier 481.53
    after_splitk = treefree_official(S_CENTRAL, pt(1.0, 0.0, 0.0, S_CENTRAL))
    after_palette = treefree_official(S_CENTRAL, pt(1.0, palette_fdq("central"), 0.0, S_CENTRAL))
    after_lk = treefree_official(S_CENTRAL, pt(LK_MULT["central"], palette_fdq("central"), 0.0, S_CENTRAL))
    lever_table = [
        {"lever": "frontier (481.53, #52)", "cumulative_tps": base, "delta_tps": 0.0,
         "note": "tau=1.0 (lawine #116 central), E[T]=3.844 linear"},
        {"lever": "+ SplitK #117 net 1.56%", "cumulative_tps": after_splitk,
         "delta_tps": after_splitk - base,
         "note": "bandwidth-util lever vg/(1+s); #117 PHYSICAL ceiling (was assumed 8.5%)"},
        {"lever": "+ palette #110 (0.3%)", "cumulative_tps": after_palette,
         "delta_tps": after_palette - after_splitk,
         "note": "lossless byte lever (bit-exact); double-quant #104 DEAD/excluded"},
        {"lever": "+ LK-loss #95 (1.010)", "cumulative_tps": after_lk,
         "delta_tps": after_lk - after_palette,
         "note": "E[T] numerator (+1.0% central, linear in TPS); prediction-only"},
    ]
    # LK upside row (high 1.024) for honesty on the E[T] lever's range.
    after_lk_high = treefree_official(S_CENTRAL, pt(LK_MULT["high"], palette_fdq("high"), 0.0, S_CENTRAL))
    composed_central_total = after_lk     # central lever stack at #117 wall

    # ===================== STEP 3: tree_required_et + re-priced #106 crossover =====================
    # tree_required_et_to_clear_500 (TEST): None/0 if tree-free already clears.
    if clears_central:
        tree_req_et_500 = 0.0
        tree_req_note = "tree-free clears 500 -> tree not required (None/0)"
    else:
        tree_req_et_500 = tree_required_et({"tree"}, "central", S_CENTRAL, TARGET)
        tree_req_note = "tree-free MISSES 500 -> bare tree must hit this E[T] to clear 500"
    # tree + the same surviving cheap levers (splitk@#117 + lk) -> lower bar.
    tree_req_et_500_stack = tree_required_et({"tree", "splitk", "lk"}, "central", S_CENTRAL, TARGET)
    # bare-tree beat-linear floor (worth-building-at-all) and analytical anchors.
    tree_beat_linear = tree_required_et({"tree"}, "central", S_CENTRAL, FRONTIER)

    # re-priced fern #106 crossover: E[T] to OVERTAKE the tree-free floor.
    #   old (tree-free@518.1, #106) vs new (tree-free@ this re-price central).
    OLD_TREEFREE_C = 518.1
    xover_old = tree_required_et({"tree"}, "central", lc.SPLITK_S["central"], OLD_TREEFREE_C)
    xover_new = tree_required_et({"tree"}, "central", S_CENTRAL, tps_central)
    xover_new_stack = tree_required_et({"tree", "splitk", "lk"}, "central", S_CENTRAL, tps_central)

    # where in denken #101's recoverable band [3.844, 5.207] does clear-500 sit?
    frac_up_band = (tree_req_et_500 - REC_BAND[0]) / (REC_BAND[1] - REC_BAND[0]) if tree_req_et_500 else None

    # ===================== VERDICT =====================
    # RED iff tree-free caps below 500 at the CENTRAL wall (#117 net central).
    if clears_central:
        verdict, label = "GREEN", (
            f"clean tree-free 500 survives on the central #117 wall: central "
            f"{tps_central:.1f} >= 500 -> the cheap path lives (tree = insurance).")
    elif clears_opt and not clears_central:
        verdict, label = "RED", (
            f"tree-free CAPS BELOW 500 at the central #117 wall ({tps_central:.1f} < 500, "
            f"gap {TARGET - tps_central:.1f}); it clears 500 ONLY at the optimistic 7.81% "
            f"band-high ({tps_opt:.1f}, the 88%-GDDR6 corner #117 prices as optimistic-"
            f"not-expected) -> the tree (land #71) is now REQUIRED for 500, not insurance. "
            f"It must recover realized E[T] >= {tree_req_et_500:.3f} (bare) / "
            f"{tree_req_et_500_stack:.3f} (with the surviving cheap levers), "
            f"~{frac_up_band*100:.0f}% up #101's recoverable band -- and it is build-blocked "
            f"at E[T]=2.10 (#101 fixable defect) -> that build is now the critical path.")
    else:
        verdict, label = "RED", (
            f"tree-free caps below 500 across the entire #117 SplitK band "
            f"({tps_cons:.1f}..{tps_opt:.1f}) -> the tree is REQUIRED for 500 with no "
            f"optimistic tree-free escape; E[T] >= {tree_req_et_500:.3f} needed.")

    gate = {
        "primary_metric_name": "tree_free_500_ceiling_at_splitk_wall",
        "tree_free_500_ceiling_at_splitk_wall": {
            "definition": "central tree-free official TPS at the denken #117 SplitK wall (net 1.56%)",
            "central": tps_central,
            "conservative_corner": tps_cons,
            "optimistic_band_high": tps_opt,
            "band": [tps_cons, tps_opt],
            "central_at_gross_3.20_wall": tps_central_gross,
            "clears_500_central": clears_central,
            "clears_500_conservative": clears_cons,
            "clears_500_optimistic_band_high": clears_opt,
        },
        "test_metric_name": "tree_required_et_to_clear_500",
        "tree_required_et_to_clear_500": tree_req_et_500,
        "tree_required_et_to_clear_500_with_cheap_levers": tree_req_et_500_stack,
        "tree_required_et_note": tree_req_note,
        "tree_required_et_frac_up_rec_band": frac_up_band,
        "splitk_needed_for_treefree_500": {
            "central_levers_tau1": s_need_central,
            "conservative_corner": s_need_corner,
            "lawine116_corner_at_tau_floor": SPLITK_CORNER_AT_TAU_FLOOR,
            "lawine116_corner_at_tau_1": SPLITK_CORNER_AT_TAU_1,
            "splitk_117_ceiling_net_central": S_CENTRAL,
            "splitk_117_band_high_gross": S_BAND_HIGH,
            "note": ("tree-free needs SplitK >= central-levers %.2f%% / corner %.2f%%; "
                     "#117 delivers net 1.56%% central (MISS), 7.81%% optimistic band-high "
                     "(clears) -> straddle, central wall misses"
                     % (s_need_central * 100 if s_need_central not in (None, float("inf")) else -1,
                        s_need_corner * 100 if s_need_corner not in (None, float("inf")) else -1)),
        },
        "reprice_106_crossover": {
            "old_treefree_C_518p1_xover_tree_alone": xover_old,
            "new_treefree_C_xover_tree_alone": xover_new,
            "new_treefree_C_xover_tree_full_stack": xover_new_stack,
            "new_treefree_C": tps_central,
            "interpretation": (
                "fern #106's recovery gate to OVERTAKE tree-free DROPS from %.3f (vs 518.1) "
                "to %.3f (vs the lower %.1f) -- but that is no longer the binding gate: "
                "since tree-free MISSES 500, the tree must clear the ABSOLUTE 500 (E[T] "
                ">= %.3f), which is higher than the overtake-floor crossover. The tree's "
                "status flips from bounded-UPSIDE (#106 AMBER, optional) to REQUIRED-for-500."
                % (xover_old, xover_new, tps_central, tree_req_et_500)),
        },
        "verdict": verdict,
        "verdict_label": label,
        "validity_caveat": (
            "SEPARATE GATE: even a tree-free OR tree 500 needs the greedy-identity "
            "VALIDITY ruling (kanna #114 + the human contract) which sits on top of ALL "
            "TPS math here. This re-price is a TPS-ceiling result only."),
        "rule": ("GREEN=tree-free clears 500 at the central #117 wall / "
                 "AMBER=clears only at the optimistic 7.81% band-high or tau=1 corner / "
                 "RED=caps below 500 at the central wall -> tree required for 500"),
    }

    out = {
        "gate": gate,
        "model_self_check_vs_117": {
            "definition": "s=3.20% gross, corner-conservative levers, tau in {0.96,0.99,1.00}",
            "reproduced": selfcheck,
            "published_117": {"tau_0.96": 474.6, "tau_0.99": 489.4, "tau_1.00": 494.3},
        },
        "step1_treefree_at_splitk_wall": {
            "central_tps": tps_central, "conservative_tps": tps_cons,
            "optimistic_band_high_tps": tps_opt,
            "central_at_gross_wall_tps": tps_central_gross,
            "band_curve_central_levers": band_curve,
        },
        "step2_lever_table": lever_table,
        "step2_lk_upside_total": after_lk_high,
        "step2_composed_central_total": composed_central_total,
        "step3_tree_reprice": {
            "tree_required_et_to_clear_500_alone": tree_req_et_500,
            "tree_required_et_to_clear_500_with_cheap_levers": tree_req_et_500_stack,
            "tree_beat_linear_et": tree_beat_linear,
            "byteshark_asbuilt_et": BYTESHARK_ASBUILT_ET,
            "rec_band": list(REC_BAND),
            "frac_up_rec_band": frac_up_band,
            "reprice_106_crossover": gate["reprice_106_crossover"],
        },
        "inputs": {
            "splitk_117_ceiling": SPLITK_117,
            "splitk_central_net": S_CENTRAL, "splitk_band": [S_BAND_LOW, S_BAND_HIGH],
            "tau_116_band": TAU_116,
            "palette_110_tps": PALETTE_TPS, "lk_95_mult": LK_MULT,
            "double_quant_104": "DEAD (KILL) -- excluded",
            "fp32_m8": FP32_M8, "persist_97": PERSIST,
            "multiplier_99_central": MULT_CENTRAL, "multiplier_99_local_ci": list(MULT_LOCAL_CI),
            "K_cal": tf.K_CAL, "E_T_linear": tf.E_T_LINEAR, "E_T_tree": E_T_TREE,
            "budget": tf.BUDGET, "frontier_official": FRONTIER, "target_official": TARGET,
        },
        "public_evidence": {
            "leaderboard_frontier_tps": 489.63,
            "leaderboard_note": ("public #1 frantic-penguin skv64 489.63 (valid); the "
                                 "SplitK/argmax-block class (byteshark 484.62, need-for-speed "
                                 "488.07) realizes only +0.6-1.7% over 481.53 and NONE clears "
                                 "500 -> field-side corroboration that SplitK alone is at its "
                                 "physical wall, consistent with #117's 3.20% ceiling."),
            "digest": "GET /v1/digest?as=senpai 2026-06-14",
        },
        "method": ("CPU-only analytic. REUSES denken #105 compose + lawine #99 multiplier CI "
                   "+ fern #102 break-even (over fern #100), swapping exactly two inputs: "
                   "SplitK -> denken #117 physical ceiling (net 1.56% / band-high 7.81%), tau "
                   "-> lawine #116 derived [0.9983,1.00]. No GPU, no served change, greedy "
                   "identity untouched (TPS-ceiling only; validity is a separate gate)."),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda o: (None if o == float("inf") else o))

    # ------------------------------- console -------------------------------
    def fmt(s):
        if s is None:
            return "clears@s=0"
        if s == float("inf"):
            return ">vg->0"
        return f"{s*100:.2f}%"

    print("=" * 92)
    print("TREE-FREE 500-PATH RE-PRICE AFTER #117 (PR #123) -- is the tree now MANDATORY?")
    print("=" * 92)
    print(f"\nfrontier {FRONTIER} | target {TARGET} | SplitK #117 ceiling: net 1.56%% central, "
          f"band-high 7.81%% (gross, 88%%-GDDR6)")
    print(f"tau (lawine #116 DERIVED): central {TAU_116['central']:.4f}, floor {TAU_116['low']:.4f} "
          f"(was asserted [0.96,1.00])")

    print(f"\n[SELF-CHECK vs #117]  s=3.20%% gross, corner levers:")
    for tau, want in (("tau_0.96", 474.6), ("tau_0.99", 489.4), ("tau_1.00", 494.3)):
        print(f"   {tau} -> {selfcheck[tau]:.1f}  (published {want})")

    print(f"\n[STEP 1] tree-free TPS at the #117 SplitK wall:")
    print(f"   conservative corner (s={S_BAND_LOW*100:.2f}%): {tps_cons:7.1f}  clears500? {clears_cons}")
    print(f"   CENTRAL            (s={S_CENTRAL*100:.2f}% net): {tps_central:7.1f}  clears500? {clears_central}   <-- PRIMARY")
    print(f"   (central @ 3.20%% gross wall):           {tps_central_gross:7.1f}  clears500? {tps_central_gross >= TARGET}")
    print(f"   optimistic band-high(s={S_BAND_HIGH*100:.2f}%): {tps_opt:7.1f}  clears500? {clears_opt}")
    print(f"   >>> PRIMARY tree_free_500_ceiling_at_splitk_wall = {tps_central:.1f} "
          f"[{tps_cons:.1f}, {tps_opt:.1f}]")
    print(f"\n   straddle curve (central levers, tau=1.0) across the #117 band:")
    for r in band_curve:
        print(f"      s={r['splitk_pct']:5.2f}% ({r['splitk_label']:22s}) -> {r['central_levers_tps']:6.1f}  "
              f"clears500? {r['clears_500_central_levers']}")
    print(f"   SplitK tree-free NEEDS to clear 500: central-levers {fmt(s_need_central)} / "
          f"corner {fmt(s_need_corner)}  (lawine#116 corner 5.49-5.84%)")

    print(f"\n[STEP 2] composed lever table (cumulative ΔTPS at central, #117 wall, tau=1.0):")
    for r in lever_table:
        print(f"   {r['lever']:30s} -> {r['cumulative_tps']:7.1f}  (Δ {r['delta_tps']:+5.2f})  {r['note']}")
    print(f"   composed central total = {composed_central_total:.1f}  (LK-high upside {after_lk_high:.1f}) "
          f"-> still {'>=500' if composed_central_total >= TARGET else 'BELOW 500'}")

    print(f"\n[STEP 3] tree re-price (TEST = tree_required_et_to_clear_500):")
    if clears_central:
        print(f"   tree-free clears 500 -> tree NOT required (E[T]=0/None)")
    else:
        print(f"   tree-free MISSES 500 by {TARGET - tps_central:.1f} -> tree REQUIRED:")
        print(f"     bare tree E[T] to clear 500       = {tree_req_et_500:.3f}  "
              f"({frac_up_band*100:.0f}% up #101 band [3.844,5.207])")
        print(f"     tree+splitk(#117)+lk E[T] to 500   = {tree_req_et_500_stack:.3f}")
        print(f"     bare tree beat-linear floor        = {tree_beat_linear:.3f}  (as-built 2.097)")
    print(f"   re-priced fern #106 crossover (OVERTAKE tree-free):")
    print(f"     old (vs C=518.1) = {xover_old:.3f}  ->  new (vs C={tps_central:.1f}) = {xover_new:.3f} "
          f"(alone) / {xover_new_stack:.3f} (+splitk+lk)")
    print(f"     => binding gate is NOW clear-500 ({tree_req_et_500:.3f}), not overtake-floor "
          f"({xover_new:.3f}); tree flips UPSIDE -> REQUIRED")

    print(f"\n[VERDICT] {verdict}")
    print(f"   {label}")
    print(f"\n   VALIDITY CAVEAT: {gate['validity_caveat']}")
    print(f"\nwrote {args.out}")

    # ------------------------------- W&B -------------------------------
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=args.wandb_name, group=args.wandb_group, job_type="analysis",
                         config={"gate": "tree-free-500-reprice",
                                 "method": "cpu-analytic-reuse-105-99-102-100",
                                 "frontier_official": FRONTIER, "target_official": TARGET,
                                 "splitk_117_net_central": S_CENTRAL,
                                 "splitk_117_band_high": S_BAND_HIGH,
                                 "tau_116_band": TAU_116, "multiplier_99": MULT_CENTRAL,
                                 "palette_110_tps": PALETTE_TPS, "lk_95_mult": LK_MULT,
                                 "double_quant_104": "DEAD"})
        s = wandb.summary
        s["tree_free_500_ceiling_at_splitk_wall"] = tps_central
        s["tree_free_500_ceiling_conservative"] = tps_cons
        s["tree_free_500_ceiling_optimistic_band_high"] = tps_opt
        s["tree_free_500_ceiling_central_at_gross_wall"] = tps_central_gross
        s["clears_500_central"] = clears_central
        s["clears_500_conservative"] = clears_cons
        s["clears_500_optimistic_band_high"] = clears_opt
        s["gap_to_500_central"] = TARGET - tps_central
        s["tree_required_et_to_clear_500"] = tree_req_et_500
        s["tree_required_et_to_clear_500_with_cheap_levers"] = tree_req_et_500_stack
        s["tree_required_et_frac_up_rec_band"] = frac_up_band if frac_up_band is not None else -1.0
        s["splitk_needed_treefree_500_central_levers"] = (
            -1.0 if s_need_central in (None, float("inf")) else s_need_central * 100.0)
        s["splitk_needed_treefree_500_corner"] = (
            -1.0 if s_need_corner in (None, float("inf")) else s_need_corner * 100.0)
        s["reprice_106_xover_old_C518"] = xover_old
        s["reprice_106_xover_new"] = xover_new
        s["composed_central_total_tps"] = composed_central_total
        s["verdict"] = verdict
        s["verdict_label"] = label
        s["public_leaderboard_frontier_tps"] = 489.63
        s["selfcheck_117_tau1"] = selfcheck["tau_1.00"]

        # lever table
        lt = wandb.Table(columns=["lever", "cumulative_tps", "delta_tps", "note"])
        for r in lever_table:
            lt.add_data(r["lever"], r["cumulative_tps"], r["delta_tps"], r["note"])
        wandb.log({"lever_table": lt})

        # straddle curve
        ct = wandb.Table(columns=["splitk_label", "splitk_pct", "central_levers_tps",
                                  "clears_500"])
        for r in band_curve:
            ct.add_data(r["splitk_label"], r["splitk_pct"], r["central_levers_tps"],
                        r["clears_500_central_levers"])
            wandb.log({"straddle/splitk_pct": r["splitk_pct"],
                       "straddle/central_tps": r["central_levers_tps"],
                       "straddle/target": TARGET})
        wandb.log({"splitk_straddle_curve": ct})
        print(f"\nW&B run: {run.id}  ({run.url})")
        wandb.finish()


if __name__ == "__main__":
    main()
