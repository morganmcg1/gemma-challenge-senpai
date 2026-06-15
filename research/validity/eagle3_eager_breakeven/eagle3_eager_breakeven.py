#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #314 (wirbel) -- can higher E[T] hit 500 on the EAGER path, skipping the rewrite?

THE OPEN DEPLOYMENT QUESTION
----------------------------
wirbel #312 (`9b1arani`, MERGED) priced the EAGLE-3-on-EAGER fallback floor at
402.1 TPS (band [302.3, 471.0]) -- a -16.5% regression vs 481.53 -- but EXPLICITLY
at iso-MTP-acceptance (E[T]=3.844), holding "any EAGLE E[T] gain as a SEPARATE
numerator axis, not credited here." That leaves the decisive question open: the
eager floor is NOT fixed -- it rises with E[T]. If the EAGLE-3 build's higher E[T]
lifts the EAGER-path TPS to >=481.53 (no-regret) or >=500 (target) WITHOUT the
T5/T6/T7 loopgraph rewrite, the launch can ship on the simple eager proposer and
SKIP the rewrite's 2-moderate + 1-correctness-rederivation surface that reopens
all four gates (greedy-identity / PPL<=2.42 / boot-500 / TPS). This card finds the
eager-path break-even E[T] and prices the rewrite against it.

THE COST MODEL (stated explicitly)
----------------------------------
A K=7-chain spec-decode step decomposes into:

    step = verify_fixed + drafter_loop

  * verify_fixed -- target verify forward + sampler + accept + host overhead.
    DRAFTER-INDEPENDENT and FIXED in E[T]. From #312: verify_fixed = decode_step
    - drafter_loop_graph_mtp = 11600 - 566.49 = 11033.51us.
  * drafter_loop -- the K=7 draft passes, graph-captured (loopgraph: 566.49us)
    OR eager (MTP: 2859.34us). The eager penalty (graph->eager launch tax) is
    2859.34 - 566.49 = 2292.85us.

CRUCIAL: E[T] is the ACCEPTANCE OUTPUT (how many of the K+1 verified positions are
kept), NOT an input to the step cost. At fixed K=7 and a fixed drafter, the step is
CONSTANT in E[T]. So

    TPS(E[T]; step) = K_cal * E[T] * (decode_step / step)     [LINEAR, monotone]

calibrated (kanna #217 `vgovdrjc`) so TPS(3.844; 11600) = K_cal * 3.844 = 481.53,
the served loopgraph baseline. The eager path inflates `step` by the eager penalty.

THE BIND (why this is not a one-liner)
--------------------------------------
The E[T] gain to 6.11 is DELIVERED BY the heavier EAGLE-3 fusion/own-KV drafter
(wirbel #293/#295: the light MTP drafter caps at ~3.844). A heavier drafter inflates
the step in BOTH graph and eager modes. So the eager break-even E[T] is
DRAFTER-COST-DEPENDENT -- a band, not a point:

  * MTP-light step (13892.85us, #312 headline): valid only near the MTP drafter's
    reach (E[T] ~ 3.844). Break-even-500 at E[T]=4.78 -- but EXTRAPOLATING this
    constant-light-step curve to E[T]=6.11 yields 639 TPS > the loopgraph path's
    500, which is physically ABSURD (eager can never beat the graph-captured path
    for the same drafter). The light-step curve is therefore INVALID above ~3.844.
  * EAGLE-heavy step (the drafter that actually PRODUCES 6.11): the only
    physically-consistent regime at high E[T]. Break-even-500 needs E[T] >= 6.36
    (iso-decode #312 band-lower) and up to UNREACHABLE (>8.0 ceiling) once #295's
    heavier LOOPGRAPH step is honoured.

WHAT THIS LEG DOES
------------------
1. REPRODUCE #312's 402.1 to <=1e-6 from the banked decomposition.
2. BUILD the eager TPS(E[T]) curve under three step regimes (MTP-light headline,
   EAGLE-heavy 3x, launch-count upper); state which components move with E[T].
3. SOLVE the eager-path break-evens for 481.53 and 500 under each regime; compare
   to the build target E[T]=6.11 (#295 bracket [5.3636, 6.8588]) and #304's
   a1->0.9213 demand. Verdict: rewrite_avoidable_at_build_target.
4. PRICE the rewrite at E[T]=6.11: loopgraph(6.11)=500 by #295 construction; the
   eager path -- same heavy drafter, no capture -- nets 360-443 (penalty band).
   GAP = the actual value of the rewrite once the E[T] gain is credited.
5. CAVEATS: eager cost is #312's banked ESTIMATE (heavier own-KV EAGLE drafter
   makes 402.1 an UPPER-bound floor -- the true floor is lower); 6.11 is an UNBUILT
   target. Prices the eager-vs-rewrite DEPLOYMENT decision as a function of E[T];
   builds NEITHER path. 0 GPU, 0 TPS, NO served-file change, NO HF Job. NaN-clean.

PRIMARY metric  eager_breakeven_self_test_passes
TEST    metrics eager_path_et_for_500 (float) + rewrite_avoidable_at_build_target (bool)
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Imported anchors -- CITE, do not re-derive. EXACT, UNCHANGED.
# --------------------------------------------------------------------------- #
# wirbel #312 (`9b1arani`) banked eager-floor decomposition (the floor we reproduce).
DECODE_STEP_US_312 = 11600.0                     # config.decode_step_ms * 1000
DRAFTER_LOOP_GRAPH_US_312 = 566.4870198567709    # chain.decode_step_chain_us_graph
DRAFTER_LOOP_EAGER_US_312 = 2859.339141845703    # chain.decode_step_chain_us_eager
STEP_EAGER_MTP_US_312 = 13892.852121988932       # step + (eager - graph)
EAGER_FLOOR_312 = 402.1                           # headline (round-1)
FLOOR_BAND_LO_312 = 302.3                         # EAGLE-heavier hard-lower
FLOOR_BAND_HI_312 = 471.0                         # launch-count upper
EAGLE_MTP_DRAFT_RATIO_312 = 3.0                   # by_m[3] eagle3/linear draft ratio (#293)
DRAFTER_PROPOSE_LAUNCHES_312 = 35.0               # launch-count model (#154)
PER_LAUNCH_OVERHEAD_US_312 = 7.416426340738932    # per-launch overhead

# kanna #217 (`vgovdrjc`) central composition convention (the TPS calibration).
OFFICIAL_TPS_217 = 481.53
E_T_ANCHOR_217 = 3.844
K_CAL_217 = 125.268
STEP_US_217 = 1218.2     # normalized served (loopgraph) step; eager path does NOT use this
TAU_217 = 1.218

# wirbel #295 (`c334qaqu`) step-corrected BUILD target E[T] (+ #293 conservative).
ET_TARGET_CENTRAL_295 = 6.1112149873699195
ET_TARGET_LOWER_295 = 5.363610726985671
ET_TARGET_UPPER_295 = 6.858819247754167
ET_TARGET_CONSERVATIVE_293 = 6.1245

# denken #304 (`dtf1ouml`) per-position acceptance the 6.11 target demands.
A1_REQUIRED_FOR_611_304 = 0.9213011665456927

# Decision lines.
TARGET_TPS = 500.0          # >=500 launch target (no rewrite)
NOREGRET_TPS = OFFICIAL_TPS_217  # >=481.53 no-regret (clears baseline without rewrite)
K_SPEC = 7
E_T_MAX = K_SPEC + 1        # 8.0 theoretical acceptance ceiling
TOL = 1e-6


# --------------------------------------------------------------------------- #
# Cost model.
# --------------------------------------------------------------------------- #
def tps_of(et: float, step_us: float,
           k_cal: float = K_CAL_217, decode_step_us: float = DECODE_STEP_US_312) -> float:
    """TPS(E[T]; step) = K_cal * E[T] * (decode_step / step). Calibrated so
    TPS(3.844; 11600) = K_cal*3.844 = 481.53 (loopgraph baseline)."""
    return k_cal * et * (decode_step_us / step_us)


def et_for_tps(target_tps: float, step_us: float,
               k_cal: float = K_CAL_217, decode_step_us: float = DECODE_STEP_US_312) -> float:
    """Invert tps_of: E[T] that yields target_tps at a given (constant) step.
    Closed form (TPS is linear in E[T]): E[T]* = target * step / (K_cal * decode)."""
    return target_tps * step_us / (k_cal * decode_step_us)


# --------------------------------------------------------------------------- #
# Step 1 -- reproduce #312's eager floor.
# --------------------------------------------------------------------------- #
def reproduce_312_floor() -> dict[str, Any]:
    step_eager = DECODE_STEP_US_312 + (DRAFTER_LOOP_EAGER_US_312 - DRAFTER_LOOP_GRAPH_US_312)
    floor_raw = OFFICIAL_TPS_217 * DECODE_STEP_US_312 / step_eager
    floor_round1 = round(floor_raw, 1)
    # cross-check via the K_cal decomposition (E[T] cancels in the iso-acceptance ratio)
    floor_via_kcal = tps_of(E_T_ANCHOR_217, step_eager) / tps_of(E_T_ANCHOR_217, DECODE_STEP_US_312) \
        * OFFICIAL_TPS_217
    return {
        "step_eager_mtp_us": step_eager,
        "step_eager_matches_312": abs(step_eager - STEP_EAGER_MTP_US_312) < TOL,
        "eager_floor_raw": floor_raw,
        "eager_floor_round1": floor_round1,
        "eager_floor_312_banked": EAGER_FLOOR_312,
        "repro_resid": abs(floor_round1 - EAGER_FLOOR_312),
        "reproduces_312": abs(floor_round1 - EAGER_FLOOR_312) < TOL,
        "floor_via_kcal_decomp": floor_via_kcal,
        "kcal_decomp_resid": abs(floor_via_kcal - floor_raw),
    }


# --------------------------------------------------------------------------- #
# Step 2 -- the eager TPS(E[T]) curve under three step regimes.
# --------------------------------------------------------------------------- #
def eager_step_regimes() -> dict[str, dict[str, Any]]:
    penalty = DRAFTER_LOOP_EAGER_US_312 - DRAFTER_LOOP_GRAPH_US_312    # 2292.85 (graph->eager)
    verify_fixed = DECODE_STEP_US_312 - DRAFTER_LOOP_GRAPH_US_312      # 11033.51 (E[T]-fixed)
    regimes = {
        # #312 HEADLINE: MTP drafter eager; valid only near E[T]~3.844 (light drafter's reach).
        "mtp_light_headline": {
            "step_us": DECODE_STEP_US_312 + penalty,                   # 13892.85
            "drafter": "MTP (light Q-only); #312 headline",
            "valid_regime": "E[T] ~ 3.844 only (light drafter caps here; see absurdity flag)",
        },
        # #312 BAND-LOWER (iso-decode): EAGLE-heavy eager penalty (3x), loopgraph step held at 11600.
        "eagle_heavy_isodecode": {
            "step_us": DECODE_STEP_US_312 + EAGLE_MTP_DRAFT_RATIO_312 * penalty,   # 18478.56
            "drafter": "EAGLE-3 (3x eager penalty), loopgraph step held = MTP decode 11600",
            "valid_regime": "consistent at high E[T] (iso-decode-step simplification of #312)",
        },
        # #312 BAND-UPPER: launch-count tax only (undercounts; mildest).
        "launch_count_upper": {
            "step_us": DECODE_STEP_US_312 + DRAFTER_PROPOSE_LAUNCHES_312 * PER_LAUNCH_OVERHEAD_US_312,  # 11859.57
            "drafter": "launch-count model (#154); UNDERCOUNTS drafter kernels ~5x",
            "valid_regime": "mild upper cross-check only",
        },
    }
    for r in regimes.values():
        r["slope_tps_per_et"] = tps_of(1.0, r["step_us"])
        r["et_for_noregret_481"] = et_for_tps(NOREGRET_TPS, r["step_us"])
        r["et_for_500"] = et_for_tps(TARGET_TPS, r["step_us"])
        r["tps_at_build_target_611"] = tps_of(ET_TARGET_CENTRAL_295, r["step_us"])
        r["et_for_500_reachable"] = bool(r["et_for_500"] <= E_T_MAX)
    return {"penalty_us": penalty, "verify_fixed_us": verify_fixed, "regimes": regimes}


# --------------------------------------------------------------------------- #
# Step 4 -- price the rewrite at E[T]=6.11 (reconciled with #295: loopgraph(6.11)=500).
# --------------------------------------------------------------------------- #
def rewrite_gap_at_611() -> dict[str, Any]:
    """At E[T]=6.11 the rewritten loopgraph (frontier-step) path nets ~500 by #295
    construction. Back out the implied heavy loopgraph step, add the eager penalty
    (same heavy drafter, no capture), and compute the eager-vs-loopgraph TPS gap."""
    penalty = DRAFTER_LOOP_EAGER_US_312 - DRAFTER_LOOP_GRAPH_US_312
    et = ET_TARGET_CENTRAL_295
    # loopgraph(6.11) = 500  =>  K_cal*6.11*(decode/loopgraph_step) = 500
    loopgraph_eagle3_step = K_CAL_217 * et * DECODE_STEP_US_312 / TARGET_TPS
    loopgraph_tps_check = tps_of(et, loopgraph_eagle3_step)
    out: dict[str, Any] = {
        "loopgraph_tps_at_611": TARGET_TPS,
        "loopgraph_eagle3_step_us": loopgraph_eagle3_step,
        "loopgraph_tps_recompute": loopgraph_tps_check,          # == 500 self-consistency
        "by_penalty": {},
    }
    for label, mult in (("optimistic_1x", 1.0), ("central_3x", EAGLE_MTP_DRAFT_RATIO_312)):
        eager_step = loopgraph_eagle3_step + mult * penalty
        eager_tps = tps_of(et, eager_step)
        out["by_penalty"][label] = {
            "penalty_mult": mult,
            "eager_eagle3_step_us": eager_step,
            "eager_tps_at_611": eager_tps,
            "rewrite_gap_tps": TARGET_TPS - eager_tps,
            # eager-path break-even E[T] at THIS heavy step (may exceed the 8.0 ceiling)
            "eager_et_for_500": et_for_tps(TARGET_TPS, eager_step),
            "eager_et_for_481": et_for_tps(NOREGRET_TPS, eager_step),
            "eager_500_reachable": bool(et_for_tps(TARGET_TPS, eager_step) <= E_T_MAX),
        }
    return out


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    repro = reproduce_312_floor()
    curve = eager_step_regimes()
    gap = rewrite_gap_at_611()

    head = curve["regimes"]["mtp_light_headline"]
    heavy = curve["regimes"]["eagle_heavy_isodecode"]

    # PRIMARY TEST float: headline (MTP-equivalent step) eager break-even for 500 --
    # the #312-headline-consistent continuation. Optimistic: labelled, not the verdict.
    eager_path_et_for_500 = head["et_for_500"]

    # ABSURDITY flag: extrapolating the constant-light-step curve to 6.11 beats the
    # loopgraph path (eager > graph), proving the light curve is INVALID above ~3.844.
    light_tps_at_611 = head["tps_at_build_target_611"]
    light_curve_absurd_above_mtp = bool(light_tps_at_611 > TARGET_TPS)

    # HONEST verdict: at the build's central E[T]=6.11, does the EAGER path -- paying
    # the eager step CONSISTENT with the heavy drafter that produces 6.11 -- meet 500?
    # Robust across conventions: iso-decode band-lower (480.6) AND #295-reconciled
    # central-3x (360.4) both fall below 500; #295-reconciled even falls below baseline.
    eager_tps_611_isodecode = heavy["tps_at_build_target_611"]            # 480.6
    eager_tps_611_reconciled = gap["by_penalty"]["central_3x"]["eager_tps_at_611"]  # 360.4
    rewrite_avoidable_at_build_target = bool(
        eager_tps_611_isodecode >= TARGET_TPS and eager_tps_611_reconciled >= TARGET_TPS)

    # consistent (heavy) break-even-500 bracket: iso-decode 6.36 .. #295-reconciled (unreachable)
    consistent_et_for_500_isodecode = heavy["et_for_500"]
    consistent_et_for_500_reconciled = gap["by_penalty"]["central_3x"]["eager_et_for_500"]

    verdict = (
        "REWRITE-NOT-AVOIDABLE" if not rewrite_avoidable_at_build_target
        else "REWRITE-AVOIDABLE-AT-BUILD-TARGET")

    handoff = (
        f"DEPLOYMENT axis for the EAGLE-3 GO/NO-GO packet: crediting the build's E[T] gain "
        f"does NOT let the launch skip the loopgraph rewrite. The eager break-even-500 is "
        f"E[T]={eager_path_et_for_500:.2f} ONLY under the MTP-light step (#312 headline) -- but "
        f"that step belongs to the light MTP drafter (caps ~3.844); extrapolating it to 6.11 "
        f"yields {light_tps_at_611:.0f} TPS > the loopgraph's 500 (eager beating graph), which is "
        f"physically absurd. The heavy EAGLE-3 drafter that ACTUALLY produces 6.11 pays a heavier "
        f"step: at E[T]=6.11 the eager path nets {eager_tps_611_reconciled:.0f} TPS (#295-reconciled, "
        f"3x penalty) to {eager_tps_611_isodecode:.0f} (iso-decode) -- below 500 under every "
        f"convention and below baseline 481.53 under the #295-reconciled gap. The rewrite is worth "
        f"{gap['by_penalty']['central_3x']['rewrite_gap_tps']:.0f} TPS (band "
        f"{gap['by_penalty']['optimistic_1x']['rewrite_gap_tps']:.0f}-"
        f"{gap['by_penalty']['central_3x']['rewrite_gap_tps']:.0f}) at the build target. "
        f"0 TPS; eager cost is #312's banked ESTIMATE (UPPER-bound floor); 6.11 is wirbel #295's "
        f"UNBUILT step-profile target; builds NEITHER path. NOT a launch. NOT a build."
    )

    return {
        "step1_reproduce_312": repro,
        "step2_eager_curve": curve,
        "step3_breakevens": {
            "eager_path_et_for_500_headline_mtp_light": eager_path_et_for_500,
            "eager_path_et_for_481_headline_mtp_light": head["et_for_noregret_481"],
            "consistent_et_for_500_eagle_heavy_isodecode": consistent_et_for_500_isodecode,
            "consistent_et_for_500_reconciled_3x": consistent_et_for_500_reconciled,
            "build_target_central_611": ET_TARGET_CENTRAL_295,
            "build_bracket_lower_536": ET_TARGET_LOWER_295,
            "build_bracket_upper_686": ET_TARGET_UPPER_295,
            "a1_required_for_611": A1_REQUIRED_FOR_611_304,
            "light_curve_absurd_above_mtp": light_curve_absurd_above_mtp,
            "light_tps_at_611": light_tps_at_611,
        },
        "step4_rewrite_gap": gap,
        "verdict_fields": {
            "eager_path_et_for_500": eager_path_et_for_500,
            "rewrite_avoidable_at_build_target": rewrite_avoidable_at_build_target,
            "eager_tps_at_611_isodecode": eager_tps_611_isodecode,
            "eager_tps_at_611_reconciled_3x": eager_tps_611_reconciled,
            "rewrite_gap_tps_central_3x": gap["by_penalty"]["central_3x"]["rewrite_gap_tps"],
            "rewrite_gap_tps_optimistic_1x": gap["by_penalty"]["optimistic_1x"]["rewrite_gap_tps"],
        },
        "verdict": verdict,
        "handoff_line": handoff,
        "caveats": [
            "0 GPU, 0 TPS, NO served-file edit, NO HF Job, NO build. BASELINE 481.53 unchanged.",
            "Eager cost is wirbel #312's BANKED ESTIMATE; the heavier own-KV EAGLE-3 drafter "
            "makes 402.1 an UPPER-bound floor -- the true eager floor is LOWER (worse).",
            "6.11 is wirbel #295's step-profile TARGET (conditional on the fusion-step profile), "
            "NOT a trained drafter; whether a1->0.9213 (#304) is TRAINABLE is a separate lane.",
            "The MTP-light eager curve is valid only near E[T]~3.844; extrapolating it to 6.11 "
            "yields eager>graph (absurd), so the constant-light-step break-even (4.78) is a mirage.",
            "The gap uses #295's loopgraph(6.11)=500 for the loopgraph leg and #312's draft-ratio "
            "for the eager penalty; the penalty band [1x,3x] brackets the gap [57, 140] TPS.",
            "This prices the eager-vs-rewrite DEPLOYMENT decision as a function of E[T]; it builds "
            "NEITHER path.",
        ],
    }


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def self_test(syn: dict[str, Any]) -> dict[str, Any]:
    s1 = syn["step1_reproduce_312"]
    s2 = syn["step2_eager_curve"]
    s3 = syn["step3_breakevens"]
    s4 = syn["step4_rewrite_gap"]
    checks: dict[str, bool] = {}

    # (a) reproduce #312's 402.1 to <=1e-6
    checks["a_reproduces_312_floor_402p1"] = bool(s1["reproduces_312"])
    checks["a_step_eager_matches_312"] = bool(s1["step_eager_matches_312"])
    checks["a_kcal_decomp_agrees"] = bool(s1["kcal_decomp_resid"] < 1e-6)

    # (b) eager break-evens for 481.53 and 500 solved + monotone (500 > 481.53 => et500 > et481)
    head = s2["regimes"]["mtp_light_headline"]
    checks["b_et500_gt_et481_headline"] = bool(head["et_for_500"] > head["et_for_noregret_481"])
    checks["b_breakevens_monotone_all_regimes"] = all(
        r["et_for_500"] > r["et_for_noregret_481"] for r in s2["regimes"].values())
    # round-trip: tps_of(et_for_tps(x)) == x
    rt = [abs(tps_of(et_for_tps(t, r["step_us"]), r["step_us"]) - t)
          for r in s2["regimes"].values() for t in (NOREGRET_TPS, TARGET_TPS)]
    checks["b_breakeven_roundtrips"] = all(e < 1e-9 for e in rt)
    checks["b_et_for_500_finite_positive"] = bool(
        math.isfinite(s3["eager_path_et_for_500_headline_mtp_light"])
        and s3["eager_path_et_for_500_headline_mtp_light"] > 0)

    # (c) gap-at-6.11 computed; loopgraph leg self-consistent at 500
    checks["c_loopgraph_recompute_is_500"] = bool(
        abs(s4["loopgraph_tps_recompute"] - TARGET_TPS) < 1e-6)
    checks["c_gap_central_computed"] = bool(
        math.isfinite(s4["by_penalty"]["central_3x"]["rewrite_gap_tps"]))
    checks["c_gap_positive_rewrite_worth_it"] = bool(
        s4["by_penalty"]["central_3x"]["rewrite_gap_tps"] > 0)

    # (d) imported #295/#304/#312/#217 constants EXACT
    checks["d_constants_312_exact"] = (
        DECODE_STEP_US_312 == 11600.0
        and DRAFTER_LOOP_GRAPH_US_312 == 566.4870198567709
        and DRAFTER_LOOP_EAGER_US_312 == 2859.339141845703
        and abs(STEP_EAGER_MTP_US_312 - 13892.852121988932) < 1e-6
        and EAGER_FLOOR_312 == 402.1 and EAGLE_MTP_DRAFT_RATIO_312 == 3.0)
    checks["d_constants_295_exact"] = (
        ET_TARGET_CENTRAL_295 == 6.1112149873699195
        and ET_TARGET_LOWER_295 == 5.363610726985671
        and ET_TARGET_UPPER_295 == 6.858819247754167
        and ET_TARGET_CONSERVATIVE_293 == 6.1245)
    checks["d_constants_304_217_exact"] = (
        abs(A1_REQUIRED_FOR_611_304 - 0.9213011665456927) < 1e-12
        and K_CAL_217 == 125.268 and OFFICIAL_TPS_217 == 481.53
        and E_T_ANCHOR_217 == 3.844 and TAU_217 == 1.218 and STEP_US_217 == 1218.2)
    # calibration self-consistency: K_cal * E_T_anchor reproduces 481.53 to round-2
    checks["d_kcal_anchor_reproduces_481"] = bool(
        abs(round(K_CAL_217 * E_T_ANCHOR_217, 2) - OFFICIAL_TPS_217) < 1e-9)

    # (e) NaN-clean over reported scalars
    scalars = [
        s1["eager_floor_raw"], head["et_for_500"], head["et_for_noregret_481"],
        s3["consistent_et_for_500_eagle_heavy_isodecode"], s3["light_tps_at_611"],
        s4["loopgraph_eagle3_step_us"],
        s4["by_penalty"]["central_3x"]["eager_tps_at_611"],
        s4["by_penalty"]["central_3x"]["rewrite_gap_tps"],
        syn["verdict_fields"]["eager_path_et_for_500"],
    ]
    checks["e_nan_clean"] = all(math.isfinite(float(x)) for x in scalars)

    # (f) caveats carried (estimate + unbuilt-target + light-curve-mirage + 0-TPS)
    cav = " ".join(syn["caveats"]) + " " + syn["handoff_line"]
    checks["f_carries_caveats"] = bool(
        "0 TPS" in cav and "ESTIMATE" in cav and "UPPER-bound" in cav
        and "NOT a launch" in cav and "NOT a build" in cav
        and ("mirage" in cav or "absurd" in cav))

    gate = bool(all(checks.values()))
    return {"eager_breakeven_self_test_passes": gate, "checks": checks}


# --------------------------------------------------------------------------- #
def _assert_nan_clean(payload: Any, path: str = "result") -> list[str]:
    bad: list[str] = []

    def walk(node: Any, p: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{p}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f"{p}[{i}]")
        elif isinstance(node, float) and not math.isfinite(node):
            bad.append(p)

    walk(payload, path)
    return bad


def _print_report(syn: dict, st: dict) -> None:
    s1, s2, s3, s4 = (syn["step1_reproduce_312"], syn["step2_eager_curve"],
                      syn["step3_breakevens"], syn["step4_rewrite_gap"])
    print("\n" + "=" * 94, flush=True)
    print("EAGLE-3 EAGER-PATH BREAK-EVEN (PR #314, wirbel) -- CPU-analytic, 0 TPS", flush=True)
    print("=" * 94, flush=True)
    print("  (1) REPRODUCE #312", flush=True)
    print(f"      eager floor = 481.53 * 11600/{s1['step_eager_mtp_us']:.0f} = "
          f"{s1['eager_floor_raw']:.4f} -> round {s1['eager_floor_round1']}  "
          f"(#312 {s1['eager_floor_312_banked']}; resid {s1['repro_resid']:.2e})", flush=True)
    print("-" * 94, flush=True)
    print("  (2) EAGER TPS(E[T]) CURVE  [TPS = K_cal*E[T]*(11600/step); LINEAR, monotone]", flush=True)
    print(f"      {'regime':<26}{'step_us':>10}{'slope':>9}{'et@481':>9}{'et@500':>9}{'tps@6.11':>10}",
          flush=True)
    for tag, r in s2["regimes"].items():
        print(f"      {tag:<26}{r['step_us']:>10.0f}{r['slope_tps_per_et']:>9.3f}"
              f"{r['et_for_noregret_481']:>9.3f}{r['et_for_500']:>9.3f}"
              f"{r['tps_at_build_target_611']:>10.1f}", flush=True)
    print(f"      LIGHT-CURVE ABSURDITY: tps@6.11(light)={s3['light_tps_at_611']:.0f} > 500 "
          f"=> eager beats graph => light curve INVALID above E[T]~3.844 "
          f"({s3['light_curve_absurd_above_mtp']})", flush=True)
    print("-" * 94, flush=True)
    print("  (3) BREAK-EVENS vs BUILD TARGET", flush=True)
    print(f"      eager_path_et_for_500 (headline MTP-light) = "
          f"{s3['eager_path_et_for_500_headline_mtp_light']:.4f}  [OPTIMISTIC/mirage]", flush=True)
    print(f"      consistent et@500 (EAGLE-heavy iso-decode) = "
          f"{s3['consistent_et_for_500_eagle_heavy_isodecode']:.4f}", flush=True)
    print(f"      build target 6.11  bracket [{s3['build_bracket_lower_536']:.4f}, "
          f"{s3['build_bracket_upper_686']:.4f}]   a1->{s3['a1_required_for_611']:.4f}", flush=True)
    print("-" * 94, flush=True)
    print("  (4) REWRITE GAP @ E[T]=6.11  (loopgraph nets 500 by #295)", flush=True)
    print(f"      implied heavy loopgraph step = {s4['loopgraph_eagle3_step_us']:.0f}us", flush=True)
    for label, p in s4["by_penalty"].items():
        print(f"      penalty {p['penalty_mult']:.0f}x: eager(6.11)={p['eager_tps_at_611']:.1f} "
              f"GAP={p['rewrite_gap_tps']:.1f} TPS  (eager-500 needs E[T]={p['eager_et_for_500']:.2f}, "
              f"reachable={p['eager_500_reachable']})", flush=True)
    print("-" * 94, flush=True)
    print(f"  PRIMARY eager_breakeven_self_test_passes = "
          f"{st['eager_breakeven_self_test_passes']}", flush=True)
    for k, v in st["checks"].items():
        print(f"          - {k}: {v}", flush=True)
    print(f"  TEST eager_path_et_for_500 = {syn['verdict_fields']['eager_path_et_for_500']:.4f}  "
          f"rewrite_avoidable_at_build_target = "
          f"{syn['verdict_fields']['rewrite_avoidable_at_build_target']}", flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print("=" * 94, flush=True)
    print(f"\n  HAND-OFF: {syn['handoff_line']}\n", flush=True)


def _maybe_log_wandb(args, payload: dict) -> str | None:
    if not getattr(args, "wandb_name", None):
        return None
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[eager-breakeven] wandb logging unavailable: {exc}", flush=True)
        return None

    syn, st = payload["synthesis"], payload["self_test"]
    s1, s2, s3, s4 = (syn["step1_reproduce_312"], syn["step2_eager_curve"],
                      syn["step3_breakevens"], syn["step4_rewrite_gap"])
    vf = syn["verdict_fields"]
    run = init_wandb_run(
        job_type="eagle3-eager-breakeven",
        agent="wirbel",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["eagle3-eager-breakeven", "eagle3-deployment-cost", "eager-fallback",
              "tps-et-curve", "validity", "zero-tps"],
        config={
            "pr": 314, "analysis_only": True, "K_spec": K_SPEC,
            "decode_step_us_312": DECODE_STEP_US_312,
            "step_eager_mtp_us_312": STEP_EAGER_MTP_US_312,
            "eager_floor_312": EAGER_FLOOR_312,
            "eagle_mtp_draft_ratio_312": EAGLE_MTP_DRAFT_RATIO_312,
            "et_target_central_295": ET_TARGET_CENTRAL_295,
            "et_target_lower_295": ET_TARGET_LOWER_295,
            "et_target_upper_295": ET_TARGET_UPPER_295,
            "a1_required_for_611_304": A1_REQUIRED_FOR_611_304,
            "k_cal_217": K_CAL_217, "official_tps_217": OFFICIAL_TPS_217,
            "target_tps": TARGET_TPS, "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[eager-breakeven] wandb: no run (no WANDB_API_KEY/mode) -- skipping", flush=True)
        return None

    head = s2["regimes"]["mtp_light_headline"]
    heavy = s2["regimes"]["eagle_heavy_isodecode"]
    c3 = s4["by_penalty"]["central_3x"]
    c1 = s4["by_penalty"]["optimistic_1x"]
    summary: dict[str, Any] = {
        # PRIMARY + TEST
        "eager_breakeven_self_test_passes": int(bool(st["eager_breakeven_self_test_passes"])),
        "eager_path_et_for_500": vf["eager_path_et_for_500"],
        "rewrite_avoidable_at_build_target": int(bool(vf["rewrite_avoidable_at_build_target"])),
        # step 1
        "eager_floor_raw": s1["eager_floor_raw"],
        "eager_floor_round1": s1["eager_floor_round1"],
        "repro_resid_vs_312": s1["repro_resid"],
        # step 2/3
        "eager_path_et_for_481_headline": head["et_for_noregret_481"],
        "consistent_et_for_500_isodecode": heavy["et_for_500"],
        "consistent_et_for_481_isodecode": heavy["et_for_noregret_481"],
        "tps_at_611_mtp_light": head["tps_at_build_target_611"],
        "tps_at_611_eagle_heavy_isodecode": heavy["tps_at_build_target_611"],
        "light_curve_absurd_above_mtp": int(bool(s3["light_curve_absurd_above_mtp"])),
        "build_target_central_611": ET_TARGET_CENTRAL_295,
        "a1_required_for_611": A1_REQUIRED_FOR_611_304,
        # step 4
        "loopgraph_eagle3_step_us": s4["loopgraph_eagle3_step_us"],
        "eager_tps_at_611_reconciled_3x": c3["eager_tps_at_611"],
        "eager_tps_at_611_reconciled_1x": c1["eager_tps_at_611"],
        "rewrite_gap_tps_central_3x": c3["rewrite_gap_tps"],
        "rewrite_gap_tps_optimistic_1x": c1["rewrite_gap_tps"],
        "eager_500_reachable_3x": int(bool(c3["eager_500_reachable"])),
        "verdict_rewrite_not_avoidable": int(syn["verdict"] == "REWRITE-NOT-AVOIDABLE"),
        **{f"selftest_{k}": int(bool(v)) for k, v in st["checks"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="eagle3_eager_breakeven_result", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    rid = getattr(run, "id", None)
    print(f"[eager-breakeven] wandb logged (run {rid})", flush=True)
    return rid


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="eagle3-eager-breakeven")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    syn = synthesize()
    st = self_test(syn)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 314, "agent": "wirbel",
        "kind": "eagle3-eager-breakeven", "synthesis": syn, "self_test": st,
        "verdict": {
            "eager_breakeven_self_test_passes": st["eager_breakeven_self_test_passes"],
            "eager_path_et_for_500": syn["verdict_fields"]["eager_path_et_for_500"],
            "rewrite_avoidable_at_build_target":
                syn["verdict_fields"]["rewrite_avoidable_at_build_target"],
        },
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[eager-breakeven] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn, st)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eagle3_eager_breakeven_results.json"

    wid = None
    if not args.no_wandb:
        wid = _maybe_log_wandb(args, payload)
    payload["wandb_run_id"] = wid
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[eager-breakeven] wrote {out_path}  (wandb run {wid})", flush=True)

    if args.self_test:
        ok = st["eager_breakeven_self_test_passes"] and payload["nan_clean"]
        print(f"[eager-breakeven] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
