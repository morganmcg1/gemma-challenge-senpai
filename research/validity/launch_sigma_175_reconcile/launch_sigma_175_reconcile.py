#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Launch-sigma #175-READING RECONCILE: does the robust-YES survive the larger sampling half-width? (PR #207).

WHAT THIS IS
------------
ubel #204 (m7vwuus2, MERGED) re-based #201's mixed-basis launch sigma onto a clean
1-sigma footing and resolved the GO trigger to a ROBUST YES: lambda=1 clears 500 at
P95 both centrally (512.41, +8.54 under the 520.95 ceiling) AND worst-case (514.63,
+6.32). The acceptance leg of #204 traces to denken #187's `h_out`=5.178 TPS. But
#195/#190 carry a LARGER reading of the SAME #175 finite-sample TPS CI: 10.906 TPS,
~2.1x bigger -- coincidentally close to #190's design-effect sqrt(D)=2.100. This is
the LAST input-magnitude lever that could move the #204 trigger: if 10.906 is the
launch-correct iid half-width, the de-duped acceptance magnitude (and the GO trigger)
moves UP, possibly back toward the ceiling. This PR RECONCILES the two readings,
re-solves the #204 trigger under each, and answers: does the robust-YES survive the
LARGER (conservative) reading?

THE CRUX -- WHY ARE THERE TWO #175 READINGS? (answer: DIFFERENT BENCH SIZES, not sqrt(D))
----------------------------------------------------------------------------------------
Both readings are the SAME finite-sample-CI formula  HW = z2 * slope * sigma_L / sqrt(N_steps)
(N_steps = bench_tokens / E[T]) at DIFFERENT benchmark token budgets and operating points:

  reading           source                  bench B   N_steps    E[T]   lambda  sigma_L   HW
  ---------------    --------------------    -------   --------   -----  ------  -------   ------
  h_out (#204 basis) #187 OUTPUT route       65536     13109.26   4.999  0.905   2.9417    5.178
  #175-sampling      #175 primary_16384      16384      3146.56   5.207  1.000   3.0354   10.906

  ratio = 10.906 / 5.178 = 2.106  =  (sigma_L 3.0354/2.9417 = 1.0319)  x  sqrt(N_steps 13109/3147 = 2.0411)
        = OPERATING-POINT factor 1.032  x  BENCH-SIZE sqrt(N) factor 2.041   -- NEITHER is sqrt(D).

The cleanest proof that the gap is BENCH SIZE: #175's OWN two readings at the SAME
lambda=1 operating point -- 10.906 (B=16384) and 5.4531 (B=65536) -- are EXACTLY a
factor of 2 apart ( = sqrt(65536/16384) = sqrt(4) = 2 ). The ~2.1 ~= sqrt(D)=2.100
coincidence is numerology: sqrt(D) is the ICC design-effect that #190 applies ON TOP
of whichever iid half-width you pick (10.906 -> 22.905), NOT the factor separating the
two readings. Reading 10.906 as "5.178 x sqrt(D)" would DOUBLE-COUNT the design effect.

WHICH READING IS LAUNCH-CORRECT? (answer: the FULL-generation B=65536 reading -> 5.178/5.4531)
--------------------------------------------------------------------------------------------
The official benchmark scores TPS over the FULL 512-token generation per prompt
(128 prompts x 512 = 65536 tokens; official/main_bucket/README.md: "total generated
tokens / wall-clock generation time"). #175's nominal B=16384 is the 128-token-per-
prompt TPS WINDOW (16384/128 = 128), a SUB-budget -- NOT the full-benchmark CI. So the
launch-correct finite-sample half-width is the B=65536 reading (h_out 5.178 at the build
op point; 5.4531 at lambda=1), the SMALLER one. The 10.906 reading is the 128-tok-window
artifact. Therefore #204's robust-YES holds on the launch-correct reading; the larger
reading only breaches the ceiling because it is the wrong (sub-bench) quantity.

RE-SOLVED TRIGGERS (same #204 machinery: combined = hypot(acc_1sigma, sigma_hw, sigma_priv); mu = 500 + z1*combined)
-------------------------------------------------------------------------------------------------------------------
  A  h_out 5.178 (LAUNCH-CORRECT, #204 basis, lambda=0.905) : 512.41 central / 514.63 worst-case  [ANCHOR; both BELOW 520.95]
  B  #175-sampling 10.906 (CONSERVATIVE, B=16384 sub-bench) : 520.98 central / 523.60 worst-case  [both ABOVE 520.95 -> breach]
  C  5.4531 (launch-correct refinement, lambda=1, B=65536)  : 512.77 central / 515.03 worst-case  [both BELOW; verdict robust to op-point]

VERDICT
-------
lambda1_clears_under_conservative_reading (TEST) = FALSE: under the LARGER 10.906
reading the worst-case trigger 523.60 (and even central 520.98) sits ABOVE the 520.95
ceiling. BUT 10.906 is NOT launch-correct -- it is the 16384-token (128-tok-window)
sub-bench; the official benchmark scores the full 65536-token generation. The launch-
correct reading (h_out 5.178 / lambda=1 refinement 5.4531) clears comfortably (512.41-
512.77 central, +8.18..+8.54 headroom). So #204's robust-YES SURVIVES; the reading
choice is verdict-changing ONLY if one mis-selects the sub-bench reading.

SCOPE
-----
LOCAL CPU-ONLY analytic reconciliation over EXISTING MERGED #204/#195/#190/#187/#175
curves. No GPU / vLLM / HF Job / submission / served-file change. Takes NO official
draws, authorizes none. BASELINE stays 481.53; greedy/PPL untouched; adds 0 TPS
(PRIMARY = self-test). The launch-correctness judgment rests on the banked provenance
strings (#175/#187/#195/#190), not a new measurement. NOT a launch.

SELF-TEST (PR step 5 -- PRIMARY)
-------------------------------
(a) the h_out reading reproduces #204's 512.41 central / 514.63 worst-case EXACTLY,
    reconstructed END-TO-END from the raw h_out through de-dup x sqrt(D) / z2 (import anchor);
(b) interpretation self-consistency: whichever branch the ratio test selects is provable
    -- here ratio != sqrt(D), and the bench-size x op-point decomposition reproduces the
    ratio to machine precision (the gap is bench size, NOT the ICC design effect);
(c) collapse-at-D->1 is CONSISTENT with the ratio verdict: the two readings collapse to
    one trigger when D->1 IFF they are sqrt(D)-separated -- both are FALSE here, consistently;
(d) monotone: a larger acceptance magnitude => a higher trigger (A < C < B, no sign flips);
(e) NaN-clean across all reported scalars.
PRIMARY = reconcile_175_self_test_passes (bool);
TEST    = lambda1_clears_under_conservative_reading (bool).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # research/validity/launch_sigma_175_reconcile -> repo root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# source-of-truth artifacts (imported verbatim; one source per constant).
# ---------------------------------------------------------------------------
REBASE_204 = os.path.join(_ROOT, "research/validity/launch_sigma_unit_rebase/launch_sigma_unit_rebase_results.json")
LAMBDA_CI_187 = os.path.join(_ROOT, "research/validity/lambda_built_ci/lambda_built_ci_results.json")
ICC_NEFF_190 = os.path.join(_ROOT, "research/validity/icc_neff/icc_neff_results.json")
ET_2MOM_175 = os.path.join(_ROOT, "research/oracle_readout/et_second_moment/et_second_moment_results.json")

# z conventions (identical to #204). A 95% TWO-SIDED CI half-width is z2*sigma; a one-
# sided P95 LCB/trigger uses z1. clean 1-sigma = 95% half-width / z2.
Z95_ONE_SIDED = 1.6448536269514722  # scipy.stats.norm.ppf(0.95)   -> trigger multiplier
Z95_TWO_SIDED = 1.959963984540054   # scipy.stats.norm.ppf(0.975)  -> 95% CI half-width / sigma
TARGET = 500.0
RHO_PLUS = 0.3                      # #195 bounded rho(*,hw) worst-case corner (PSD-admissible)

# tolerances
ANCHOR_TOL_TPS = 1e-6              # self-test (a): h_out reading must reproduce #204 to < 1e-6 TPS
DECOMP_TOL = 1e-6                  # self-test (b): bench-decomposition must reproduce the ratio
TRIGGER_TOL_TPS = 1e-6            # numerical equality of triggers
RATIO_SQRTD_RELTOL = 1e-3         # self-test: ratio "equals" sqrt(D) only within 0.1% (it is 0.29% off)


def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _combined_sigma_rho(sig: np.ndarray, rho: float) -> float:
    """Combined sigma over axes with a common off-diagonal rho (PSD-admissible). Identical to #204."""
    R = np.full((sig.size, sig.size), rho, dtype=float)
    np.fill_diagonal(R, 1.0)
    C = np.outer(sig, sig) * R
    return math.sqrt(max(float(np.ones(sig.size) @ C @ np.ones(sig.size)), 0.0))


# ---------------------------------------------------------------------------
# Step 0 -- import the banked legs (NOT re-derived).
# ---------------------------------------------------------------------------
def import_banked() -> dict[str, Any]:
    r204 = _load(REBASE_204)
    legs204 = r204["imported_legs_201"]
    trig204 = r204["clean_trigger"]

    c187 = _load(LAMBDA_CI_187)["synthesis"]
    ioc = c187["input_output_compose"]
    op187 = c187["operating_point"]

    icc190 = _load(ICC_NEFF_190)
    rci190 = icc190["realistic_ci"]

    c175 = _load(ET_2MOM_175)
    fs175 = c175["finite_sample_tps_ci"]["both_bugs"]
    bb16384 = fs175["primary_16384_tau1"]
    bb65536 = fs175["sensitivity_65536_tau1"]

    out = {
        # --- #204 launch legs + clean trigger (the curve being reconciled; ANCHOR) ---
        "acc_iid_halfwidth_204": float(legs204["acc_iid_halfwidth"]),                 # 5.318697 (HW, iid, de-duped)
        "acc_realistic_halfwidth_204": float(legs204["acc_realistic_halfwidth"]),     # 11.170041 (HW, realistic)
        "sqrt_design_effect": float(legs204["sqrt_design_effect"]),                   # 2.100146 (dimensionless)
        "sigma_hw": float(legs204["sigma_hw"]),                                       # 4.864469 (1-sigma)
        "sigma_private": float(legs204["sigma_private"]),                             # 0.883918 (1-sigma)
        "lambda1_ceiling": float(legs204["lambda1_ceiling"]),                         # 520.952732
        "acc_1sigma_clean_204": float(trig204["acceptance_1sigma_clean"]),            # 5.699105 (realistic 1-sigma)
        "combined_central_204": float(trig204["combined_sigma_launch_clean_central"]),    # 7.544811
        "combined_worstcase_204": float(trig204["combined_sigma_launch_clean_worstcase"]),  # 8.897216
        "mu_central_204": float(trig204["mu_clears_500_clean_central"]),              # 512.410110 (ANCHOR)
        "mu_worstcase_204": float(trig204["mu_clears_500_clean_worstcase"]),          # 514.634617 (ANCHOR)
        # --- #187 de-dup inputs + h_out operating point ---
        "h_in_187": float(ioc["h_in_tps_lambda_route"]),                             # 3.710490 (INPUT route HW)
        "h_out_187": float(ioc["h_out_tps_lbar_route_175"]),                         # 5.178248 (OUTPUT route HW @ 65536)
        "overlap_corrected_187": float(ioc["overlap_corrected_same_bench_tps"]),     # 5.318697 (de-duped)
        "overlap_fraction": float(ioc["overlap_fraction"]),                          # 0.892917
        "sigma_L_187": float(ioc["sigma_L"]),                                        # 2.941728 (@ lambda=0.905)
        "E_T_187": float(op187["E_T"]),                                              # 4.999213
        "N_steps_187": float(op187["N_steps"]),                                      # 13109.26 (B=65536)
        "lambda_187": float(op187["lambda"]),                                        # 0.905229 (build bar)
        "n_prompts_187": int(op187["n_prompts"]),                                    # 128
        "output_len_187": int(op187["output_len"]),                                  # 512  -> bench = 128*512 = 65536
        # --- #175 finite-sample CI: the LARGER reading + the same-formula 65536 cross-read ---
        "sampling_175": float(c175["tps_finite_sample_ci_halfwidth"]),               # 10.906182 (HW @ B=16384, lambda=1)
        "N_steps_175_16384": float(bb16384["N_steps"]),                              # 3146.56
        "sigma_L_175": float(bb16384["sigma_L"]),                                    # 3.035437 (@ lambda=1)
        "slope_175": float(bb16384["tps_slope_per_Lbar"]),                           # 102.830
        "E_T_175": float(bb16384["E_T"]),                                            # 5.206954 (lambda=1)
        "bench_tokens_175_16384": int(bb16384["n_tokens"]),                          # 16384
        "hw_175_65536": float(bb65536["ci_halfwidth_tps"]),                          # 5.453091 (same formula, B=65536)
        "N_steps_175_65536": float(bb65536["N_steps"]),                              # 12586.24
        "bench_tokens_175_65536": int(bb65536["n_tokens"]),                          # 65536
        # --- #190 design effect (the sqrt(D) the ratio is coincidentally near) ---
        "design_effect_190": float(rci190["design_effect_hat"]),                     # 4.410614
        "halfwidth_iid_190": float(rci190["halfwidth_iid_tps"]),                     # 10.906182 (== sampling_175)
        "halfwidth_realistic_190": float(rci190["halfwidth_realistic_tps"]),         # 22.904577 (= 10.906 * D... no: *sqrt(D))
        "icc_hat_190": float(icc190["icc_estimate"]["icc_hat"]),                     # 0.144625
    }
    return out


# ---------------------------------------------------------------------------
# Step 1 -- trace both readings to their #175 source; decompose the gap.
# ---------------------------------------------------------------------------
def trace_readings(b: dict[str, Any]) -> dict[str, Any]:
    ratio = b["sampling_175"] / b["h_out_187"]
    sqrtD = b["sqrt_design_effect"]
    # decompose the ratio: HW = z2*slope*sigma_L/sqrt(N_steps) ; z2, slope identical ->
    # ratio = (sigma_L_175 / sigma_L_187) * sqrt(N_steps_187 / N_steps_175_16384).
    sigma_L_factor = b["sigma_L_175"] / b["sigma_L_187"]
    bench_sqrtN_factor = math.sqrt(b["N_steps_187"] / b["N_steps_175_16384"])
    decomp_ratio = sigma_L_factor * bench_sqrtN_factor
    decomp_err = abs(ratio - decomp_ratio)

    # #175's OWN bench-size effect at a FIXED operating point (lambda=1): 10.906 (B=16384) vs 5.4531 (B=65536).
    same_op_bench_ratio = b["sampling_175"] / b["hw_175_65536"]
    pure_bench_sqrtN = math.sqrt(b["bench_tokens_175_65536"] / b["bench_tokens_175_16384"])  # sqrt(4) = 2
    same_op_bench_err = abs(same_op_bench_ratio - pure_bench_sqrtN)

    rel_err_vs_sqrtD = abs(ratio - sqrtD) / sqrtD
    ratio_equals_sqrtD = bool(rel_err_vs_sqrtD <= RATIO_SQRTD_RELTOL)

    return {
        "ratio_175_readings": ratio,
        "sqrt_design_effect": sqrtD,
        "abs_gap_ratio_vs_sqrtD": abs(ratio - sqrtD),
        "rel_gap_ratio_vs_sqrtD": rel_err_vs_sqrtD,
        "ratio_equals_sqrtD": ratio_equals_sqrtD,
        "decomposition": {
            "sigma_L_operating_point_factor": sigma_L_factor,        # 1.0319 (lambda=0.905 vs 1.0)
            "bench_size_sqrtN_factor": bench_sqrtN_factor,           # 2.0411 (65536 vs 16384, mixed op-points)
            "product_reproduces_ratio": decomp_ratio,
            "decomp_err_vs_ratio": decomp_err,
            "note": "ratio = sigma_L-operating-point factor x bench-size sqrt(N) factor; NEITHER is sqrt(D). "
            "The product reproduces the ratio to < 1e-6, proving the gap is bench-size x op-point, not the ICC design effect.",
        },
        "pure_bench_size_demo": {
            "reading_16384_lambda1_tps": b["sampling_175"],          # 10.906 @ B=16384
            "reading_65536_lambda1_tps": b["hw_175_65536"],          # 5.4531 @ B=65536 (SAME lambda=1)
            "same_op_bench_ratio": same_op_bench_ratio,              # 2.0000
            "pure_sqrt_bench_token_ratio": pure_bench_sqrtN,         # sqrt(65536/16384) = 2.0000
            "same_op_bench_err": same_op_bench_err,
            "note": "#175's OWN two readings at FIXED lambda=1 (10.906 @ 16384, 5.4531 @ 65536) are EXACTLY a "
            "factor 2 = sqrt(4) apart -> the cleanest proof the readings differ by BENCH SIZE, not sqrt(D).",
        },
        "axis_interpretation": (
            "SAME finite-sample TPS CI formula (HW = z2*slope*sigma_L/sqrt(N_steps)) at DIFFERENT bench budgets; "
            "h_out is the B=65536 FULL-generation read (de-duped with the input route), #175-sampling is the "
            "B=16384 128-tok-window sub-budget. NOT genuinely-different axes (h_out is already input/output "
            "de-duped via #187), and NOT sqrt(D)-apart (the ~2.1 ~= 2.100 is coincidence)."
        ),
        "provenance": {
            "h_out": "denken #187 OUTPUT route: #175 sigma_L/sqrt(N_steps) x slope, recomputed on the 128x512 "
            "(B=65536) bench at the lambda=0.905 build op point (E[T]=4.999, sigma_L=2.9417) -> 5.178; "
            "then #187 de-dups it with the INPUT route h_in=3.710 (overlap_fraction=0.893) -> 5.31870.",
            "sampling_175": "wirbel #175 tps_finite_sample_ci_HALFWIDTH (both-bugs, B=16384 nominal, z=1.96) at "
            "lambda=1 (E[T]=5.207, sigma_L=3.0354) -> 10.906; B=16384 with 128 prompts = a 128-tok-per-prompt "
            "TPS WINDOW, a sub-budget of the official 512-tok-per-prompt (B=65536) full generation.",
            "sqrt_D": "wirbel #190 design effect sqrt(D)=sqrt(4.4106)=2.100 = halfwidth_realistic 22.905 / "
            "halfwidth_iid 10.906 -- the ICC inflation applied ON TOP of the iid 10.906, NOT the factor between "
            "the two readings (reading 10.906 as 5.178*sqrt(D) would DOUBLE-COUNT the design effect).",
        },
    }


# ---------------------------------------------------------------------------
# Step 2 -- acceptance magnitude under a reading: de-dup (h_in (+) out-route),
#           x sqrt(D), / z2.  iid-1sigma and realistic-1sigma both reported.
# ---------------------------------------------------------------------------
def acceptance_under_reading(b: dict[str, Any], out_route_hw: float) -> dict[str, Any]:
    # #187 overlap-corrected de-dup: keep the OUTPUT route at full weight, add the (1-overlap) deflated input route.
    dedup_hw_iid = math.sqrt(out_route_hw ** 2 + (1.0 - b["overlap_fraction"]) * b["h_in_187"] ** 2)
    dedup_hw_realistic = dedup_hw_iid * b["sqrt_design_effect"]   # #190 ICC inflation
    acc_1sigma_iid = dedup_hw_iid / Z95_TWO_SIDED                 # clean iid 1-sigma (pre-ICC)
    acc_1sigma_realistic = dedup_hw_realistic / Z95_TWO_SIDED     # clean realistic 1-sigma (drives the trigger)
    return {
        "out_route_halfwidth": out_route_hw,
        "dedup_halfwidth_iid": dedup_hw_iid,
        "dedup_halfwidth_realistic": dedup_hw_realistic,
        "acceptance_1sigma_iid": acc_1sigma_iid,
        "acceptance_1sigma_realistic": acc_1sigma_realistic,
    }


def trigger_from_acc(b: dict[str, Any], acc_1sigma_realistic: float) -> dict[str, Any]:
    sig = np.array([acc_1sigma_realistic, b["sigma_hw"], b["sigma_private"]], dtype=float)
    comb_central = _combined_sigma_rho(sig, 0.0)
    comb_worstcase = _combined_sigma_rho(sig, RHO_PLUS)
    mu_central = TARGET + Z95_ONE_SIDED * comb_central
    mu_worstcase = TARGET + Z95_ONE_SIDED * comb_worstcase
    ceiling = b["lambda1_ceiling"]
    return {
        "combined_central": comb_central,
        "combined_worstcase": comb_worstcase,
        "trigger_central": mu_central,
        "trigger_worstcase": mu_worstcase,
        "central_below_ceiling": bool(mu_central <= ceiling),
        "worstcase_below_ceiling": bool(mu_worstcase <= ceiling),
        "central_headroom_below_ceiling_tps": ceiling - mu_central,
        "worstcase_headroom_below_ceiling_tps": ceiling - mu_worstcase,
    }


# ---------------------------------------------------------------------------
# Step 3 -- re-solve the clean trigger under each reading.
# ---------------------------------------------------------------------------
def resolve_readings(b: dict[str, Any]) -> dict[str, Any]:
    # Reading A -- h_out (LAUNCH-CORRECT, #204 basis). Reconstructed END-TO-END from raw h_out.
    accA = acceptance_under_reading(b, b["h_out_187"])
    trigA = trigger_from_acc(b, accA["acceptance_1sigma_realistic"])
    # Reading B -- #175-sampling 10.906 (CONSERVATIVE, B=16384 sub-bench).
    accB = acceptance_under_reading(b, b["sampling_175"])
    trigB = trigger_from_acc(b, accB["acceptance_1sigma_realistic"])
    # Reading C -- 5.4531 (launch-correct refinement: lambda=1 sigma_L at B=65536).
    accC = acceptance_under_reading(b, b["hw_175_65536"])
    trigC = trigger_from_acc(b, accC["acceptance_1sigma_realistic"])
    return {
        "A_hout_launch_correct": {"acceptance": accA, "trigger": trigA},
        "B_175sampling_conservative": {"acceptance": accB, "trigger": trigB},
        "C_lambda1_full_bench_refinement": {"acceptance": accC, "trigger": trigC},
    }


# ---------------------------------------------------------------------------
# Step 4 -- the headline verdict.
# ---------------------------------------------------------------------------
def headline_verdict(b: dict[str, Any], trace: dict[str, Any], readings: dict[str, Any]) -> dict[str, Any]:
    A = readings["A_hout_launch_correct"]
    B = readings["B_175sampling_conservative"]
    C = readings["C_lambda1_full_bench_refinement"]
    ceiling = b["lambda1_ceiling"]

    # TEST: under the LARGER reading, does the worst-case trigger still sit below the ceiling?
    lambda1_clears_conservative = bool(B["trigger"]["worstcase_below_ceiling"])
    # the launch-correct reading is the FULL-generation (B=65536) one -> A (and its lambda=1 refinement C).
    robust_yes_survives = bool(A["trigger"]["central_below_ceiling"] and A["trigger"]["worstcase_below_ceiling"]
                               and C["trigger"]["central_below_ceiling"] and C["trigger"]["worstcase_below_ceiling"])

    delta_central = B["trigger"]["trigger_central"] - A["trigger"]["trigger_central"]
    delta_worstcase = B["trigger"]["trigger_worstcase"] - A["trigger"]["trigger_worstcase"]

    return {
        "lambda1_clears_under_conservative_reading": lambda1_clears_conservative,   # <-- TEST
        "launch_correct_reading": "h_out 5.178 (B=65536 full-generation; lambda=1 refinement 5.4531)",
        "conservative_reading_is_launch_correct": False,
        "robust_yes_survives": robust_yes_survives,
        "verdict_changing": bool(not lambda1_clears_conservative),  # the 16384 reading WOULD flip it -- but it is the wrong bench
        "delta_trigger_reading_central": delta_central,            # 520.98 - 512.41 = 8.57
        "delta_trigger_reading_worstcase": delta_worstcase,        # 523.60 - 514.63 = 8.96
        "explanation": (
            "lambda1_clears_under_conservative_reading = %s: under the LARGER 10.906 reading the worst-case "
            "trigger %.2f (central %.2f) sits ABOVE the %.2f ceiling. BUT 10.906 is the B=16384 128-tok-window "
            "sub-bench, NOT the official full-generation (B=65536) CI -- so it is NOT launch-correct. The launch-"
            "correct reading (h_out 5.178 -> %.2f/%.2f; lambda=1 refinement 5.4531 -> %.2f/%.2f) clears 500 at P95 "
            "central AND worst-case, both well below the ceiling. #204's robust-YES SURVIVES; the reading choice "
            "is verdict-changing ONLY if one mis-selects the sub-bench reading."
            % (lambda1_clears_conservative, B["trigger"]["trigger_worstcase"], B["trigger"]["trigger_central"],
               ceiling, A["trigger"]["trigger_central"], A["trigger"]["trigger_worstcase"],
               C["trigger"]["trigger_central"], C["trigger"]["trigger_worstcase"])
        ),
    }


# ---------------------------------------------------------------------------
# Step 5 -- self-test (PRIMARY).
# ---------------------------------------------------------------------------
def self_test(b: dict[str, Any], trace: dict[str, Any], readings: dict[str, Any]) -> dict[str, Any]:
    A = readings["A_hout_launch_correct"]
    B = readings["B_175sampling_conservative"]
    C = readings["C_lambda1_full_bench_refinement"]

    # (a) the h_out reading reproduces #204's 512.41 / 514.63 EXACTLY, reconstructed end-to-end from raw h_out.
    a_central_err = abs(A["trigger"]["trigger_central"] - b["mu_central_204"])
    a_worstcase_err = abs(A["trigger"]["trigger_worstcase"] - b["mu_worstcase_204"])
    a_acc_err = abs(A["acceptance"]["acceptance_1sigma_realistic"] - b["acc_1sigma_clean_204"])
    a_dedup_err = abs(A["acceptance"]["dedup_halfwidth_iid"] - b["acc_iid_halfwidth_204"])  # reproduce #204's 5.31870
    a_ok = bool(a_central_err <= ANCHOR_TOL_TPS and a_worstcase_err <= ANCHOR_TOL_TPS
                and a_acc_err <= ANCHOR_TOL_TPS and a_dedup_err <= ANCHOR_TOL_TPS)

    # (b) interpretation self-consistency: whichever branch the ratio test selects is provable.
    ratio = trace["ratio_175_readings"]
    sqrtD = trace["sqrt_design_effect"]
    if trace["ratio_equals_sqrtD"]:
        # branch (a): the readings' acceptance legs must be exactly sqrt(D) apart.
        acc_ratio = B["acceptance"]["acceptance_1sigma_iid"] / A["acceptance"]["acceptance_1sigma_iid"]
        b_ok = bool(abs(acc_ratio - sqrtD) / sqrtD <= RATIO_SQRTD_RELTOL)
        b_detail = "branch sqrt(D): acc_ratio=%.6f vs sqrt(D)=%.6f" % (acc_ratio, sqrtD)
    else:
        # branch bench-size: the sigma_L x sqrt(N) decomposition reproduces the ratio to machine precision.
        b_ok = bool(trace["decomposition"]["decomp_err_vs_ratio"] <= DECOMP_TOL)
        b_detail = "branch bench: decomp %.10f reproduces ratio %.10f (err %.2e)" % (
            trace["decomposition"]["product_reproduces_ratio"], ratio, trace["decomposition"]["decomp_err_vs_ratio"])

    # (c) collapse-at-D->1 is CONSISTENT with the ratio verdict.
    # trigger at D->1 (no sqrt(D) inflation) = 500 + z1*hypot(acc_iid, sigma_hw, sigma_priv).
    def trig_D1(acc_iid: float) -> float:
        return TARGET + Z95_ONE_SIDED * _combined_sigma_rho(
            np.array([acc_iid, b["sigma_hw"], b["sigma_private"]]), 0.0)
    trigA_D1 = trig_D1(A["acceptance"]["acceptance_1sigma_iid"])
    trigB_D1 = trig_D1(B["acceptance"]["acceptance_1sigma_iid"])
    readings_collapse_at_D1 = bool(abs(trigA_D1 - trigB_D1) <= TRIGGER_TOL_TPS)
    c_ok = bool(readings_collapse_at_D1 == trace["ratio_equals_sqrtD"])

    # (d) monotone: larger acceptance magnitude => higher trigger (A < C < B), no sign flips.
    accs = [A["acceptance"]["acceptance_1sigma_realistic"], C["acceptance"]["acceptance_1sigma_realistic"],
            B["acceptance"]["acceptance_1sigma_realistic"]]
    trigs = [A["trigger"]["trigger_central"], C["trigger"]["trigger_central"], B["trigger"]["trigger_central"]]
    d_ok = bool(accs[0] < accs[1] < accs[2] and trigs[0] < trigs[1] < trigs[2])

    # (e) NaN-clean across all reported scalars.
    scalars = [
        ratio, sqrtD, trace["decomposition"]["product_reproduces_ratio"],
        A["acceptance"]["acceptance_1sigma_iid"], A["acceptance"]["acceptance_1sigma_realistic"],
        B["acceptance"]["acceptance_1sigma_iid"], B["acceptance"]["acceptance_1sigma_realistic"],
        C["acceptance"]["acceptance_1sigma_realistic"],
        A["trigger"]["trigger_central"], A["trigger"]["trigger_worstcase"],
        B["trigger"]["trigger_central"], B["trigger"]["trigger_worstcase"],
        C["trigger"]["trigger_central"], C["trigger"]["trigger_worstcase"],
        trigA_D1, trigB_D1, a_central_err, a_worstcase_err,
    ]
    e_ok = all(_finite(x) for x in scalars)

    checks = {
        "a_hout_reproduces_204_anchor": a_ok,
        "b_interpretation_self_consistent": b_ok,
        "c_collapse_at_D1_consistent_with_ratio": c_ok,
        "d_monotone_trigger_in_acceptance": d_ok,
        "e_nan_clean": e_ok,
    }
    passes = all(checks.values())
    return {
        "reconcile_175_self_test_passes": bool(passes),   # <-- PRIMARY
        "checks": checks,
        "evidence": {
            "a_central_err_tps": a_central_err,
            "a_worstcase_err_tps": a_worstcase_err,
            "a_acc_realistic_err_tps": a_acc_err,
            "a_dedup_iid_err_tps": a_dedup_err,
            "b_detail": b_detail,
            "c_trigA_at_D1": trigA_D1,
            "c_trigB_at_D1": trigB_D1,
            "c_readings_collapse_at_D1": readings_collapse_at_D1,
            "c_ratio_equals_sqrtD": trace["ratio_equals_sqrtD"],
            "d_acceptance_1sigma_ACB": accs,
            "d_trigger_central_ACB": trigs,
            "n_scalars_checked": len(scalars),
        },
    }


def _build_result(b, trace, readings, verdict, st) -> dict[str, Any]:
    A = readings["A_hout_launch_correct"]
    B = readings["B_175sampling_conservative"]
    C = readings["C_lambda1_full_bench_refinement"]
    handoff = (
        "fern #185 + land #71: the two #175 readings (h_out 5.178 vs #175-sampling 10.906) are the SAME "
        "finite-sample TPS CI at DIFFERENT bench sizes (B=65536 full-generation vs B=16384 128-tok-window), "
        "NOT sqrt(D)-apart (the 2.106 ratio = bench-sqrt(N) 2.04 x op-point 1.03, only COINCIDENTALLY ~= #190's "
        "sqrt(D) 2.100); the launch-correct acceptance magnitude is the FULL-bench h_out -> de-duped 1sigma "
        "%.4f (lambda=1 refinement %.4f), giving a clean trigger of %.2f central / %.2f worst-case (lambda=1 "
        "refinement %.2f / %.2f); the robust-YES at lambda=1 DOES survive the conservative reading because "
        "10.906 is the 128-tok-window sub-bench, not the official full-generation CI; fern wires the launch-"
        "correct ~%.1f central trigger and land #71's co-log (n=385) remains the lever that retires rho(*,hw)."
        % (A["acceptance"]["acceptance_1sigma_iid"], C["acceptance"]["acceptance_1sigma_iid"],
           A["trigger"]["trigger_central"], A["trigger"]["trigger_worstcase"],
           C["trigger"]["trigger_central"], C["trigger"]["trigger_worstcase"],
           A["trigger"]["trigger_central"])
    )
    return {
        "pr": 207,
        "metric_primary": "reconcile_175_self_test_passes",
        "metric_test": "lambda1_clears_under_conservative_reading",
        "reconcile_175_self_test_passes": st["reconcile_175_self_test_passes"],
        "lambda1_clears_under_conservative_reading": verdict["lambda1_clears_under_conservative_reading"],
        # --- step 1: trace + ratio ---
        "ratio_175_readings": trace["ratio_175_readings"],
        "ratio_equals_sqrtD": trace["ratio_equals_sqrtD"],
        # --- step 2: acceptance magnitudes (iid 1-sigma; the launch-correct anchor is 2.7137) ---
        "acceptance_1sigma_hout": A["acceptance"]["acceptance_1sigma_iid"],            # = 2.7137 (#204 anchor)
        "acceptance_1sigma_175sampling": B["acceptance"]["acceptance_1sigma_iid"],
        "acceptance_1sigma_hout_realistic": A["acceptance"]["acceptance_1sigma_realistic"],         # 5.6991 (drives trigger)
        "acceptance_1sigma_175sampling_realistic": B["acceptance"]["acceptance_1sigma_realistic"],  # 11.758
        # --- step 3: re-solved triggers ---
        "trigger_central_hout": A["trigger"]["trigger_central"],                       # = 512.41 (anchor)
        "trigger_worstcase_hout": A["trigger"]["trigger_worstcase"],                   # = 514.63 (anchor)
        "trigger_central_175sampling": B["trigger"]["trigger_central"],                # 520.98
        "trigger_worstcase_175sampling": B["trigger"]["trigger_worstcase"],            # 523.60
        "delta_trigger_reading": verdict["delta_trigger_reading_central"],            # 8.57 central
        "delta_trigger_reading_worstcase": verdict["delta_trigger_reading_worstcase"],
        # --- step 4: verdict ---
        "robust_yes_survives": verdict["robust_yes_survives"],
        "launch_correct_reading": verdict["launch_correct_reading"],
        "lambda1_ceiling": b["lambda1_ceiling"],
        "law": "both readings are HW = z2*slope*sigma_L/sqrt(N_steps) at different bench budgets; acceptance_1sigma "
        "= dedup(h_in (+) out_route) * sqrt(D) / z2; combined = hypot(acc, sigma_hw, sigma_priv)[+rho]; "
        "trigger = 500 + z1*combined. Launch-correct bench = official full generation B=65536 (NOT 128-tok window 16384).",
        "trace_readings": trace,
        "readings": readings,
        "verdict": verdict,
        "self_test": st,
        "handoff": handoff,
        "scope": "Pure CPU-only analytic reconciliation of two banked #175 finite-sample-CI readings (h_out 5.178 "
        "vs #175-sampling 10.906) and re-solution of ubel's OWN #204 GO trigger under each. Takes NO official "
        "draws, authorizes none. BASELINE stays 481.53; adds 0 TPS; greedy/PPL untouched. The launch-correctness "
        "judgment rests on the banked provenance (#175/#187/#195/#190), not a new measurement. The rho(*,hw) "
        "[-0.3,+0.3] band still needs land #71's co-log (separate lever). NOT a launch.",
        "public_evidence_used": [
            "ubel #204 (m7vwuus2, MERGED) launch_sigma_unit_rebase: clean trigger 512.41 central / 514.63 "
            "worst-case, combined sigma 7.5448/8.8972, acceptance 1-sigma 5.6991 (realistic) / 2.7137 (iid), "
            "lambda1 ceiling 520.95 -- the trigger re-solved here under each #175 reading.",
            "denken #187 (lambda_built_ci) input_output_compose: h_in 3.710 (+) h_out 5.178 overlap-corrected "
            "(overlap_fraction 0.893) -> 5.31870; h_out is the OUTPUT route on the B=65536 (128x512) bench at the "
            "lambda=0.905 build op point (E[T]=4.999, sigma_L=2.9417).",
            "wirbel #175 (et_second_moment) finite_sample_tps_ci: 10.906 HW @ B=16384 lambda=1 (E[T]=5.207, "
            "sigma_L=3.0354, slope 102.830) and 5.4531 HW @ B=65536 SAME lambda=1 -- EXACTLY factor 2 apart "
            "(= sqrt(65536/16384)) -> the bench-size proof.",
            "wirbel #190 (icc_neff) realistic_ci: design_effect 4.4106, sqrt(D)=2.100 = halfwidth_realistic "
            "22.905 / halfwidth_iid 10.906 -- the ICC inflation applied ON TOP of 10.906, NOT the factor between "
            "the two readings.",
        ],
        "method": "LOCAL CPU-only analytic reconciliation over EXISTING MERGED results; no GPU/vLLM/HF Job/"
        "submission/served-file change. BASELINE stays 481.53; adds 0 TPS. Greedy/PPL untouched. NOT a launch.",
        "convention_note": "Identical to #204: 95%-two-sided HW = z2*sigma so 1-sigma = HW/z2; trigger = 500 + "
        "z1*hypot(acc_1sigma, sigma_hw, sigma_priv)[+rho]; z1=1.64485 (one-sided P95), z2=1.95996 (two-sided 95%).",
        "metrics_nan_clean": 1 if st["checks"]["e_nan_clean"] else 0,
    }


def _log_wandb(args, result: dict[str, Any]) -> None:
    if args.no_wandb:
        return
    try:
        from scripts import wandb_logging
    except Exception as exc:  # noqa: BLE001
        print(f"[175-reconcile] wandb_logging import failed ({exc}); skipping", flush=True)
        return
    try:
        run = wandb_logging.init_wandb_run(
            job_type="launch-sigma-175-reconcile", agent="ubel",
            name=args.wandb_name or "ubel/launch-sigma-175-reconcile",
            group=args.wandb_group,
            tags=["launch-sigma", "175-reconcile", "finite-sample-ci", "bench-size", "footing", "pr207"],
            config={"baseline_tps": 481.53, "method": "cpu-only-analytic", "z_one_sided_p95": Z95_ONE_SIDED,
                    "z_two_sided_95": Z95_TWO_SIDED, "rho_plus_worstcase": RHO_PLUS},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[175-reconcile] wandb init failed ({exc}); skipping", flush=True)
        return
    if run is None:
        print("[175-reconcile] wandb disabled; skipping", flush=True)
        return
    try:
        trace = result["trace_readings"]
        A = result["readings"]["A_hout_launch_correct"]["trigger"]
        B = result["readings"]["B_175sampling_conservative"]["trigger"]
        C = result["readings"]["C_lambda1_full_bench_refinement"]["trigger"]
        st = result["self_test"]
        flat = {
            "ratio_175_readings": result["ratio_175_readings"],
            "sqrt_design_effect": trace["sqrt_design_effect"],
            "rel_gap_ratio_vs_sqrtD": trace["rel_gap_ratio_vs_sqrtD"],
            "ratio_equals_sqrtD": 1.0 if result["ratio_equals_sqrtD"] else 0.0,
            "decomp_sigma_L_factor": trace["decomposition"]["sigma_L_operating_point_factor"],
            "decomp_bench_sqrtN_factor": trace["decomposition"]["bench_size_sqrtN_factor"],
            "decomp_err_vs_ratio": trace["decomposition"]["decomp_err_vs_ratio"],
            "pure_bench_same_op_ratio": trace["pure_bench_size_demo"]["same_op_bench_ratio"],
            "acceptance_1sigma_hout": result["acceptance_1sigma_hout"],
            "acceptance_1sigma_175sampling": result["acceptance_1sigma_175sampling"],
            "acceptance_1sigma_hout_realistic": result["acceptance_1sigma_hout_realistic"],
            "acceptance_1sigma_175sampling_realistic": result["acceptance_1sigma_175sampling_realistic"],
            "trigger_central_hout": A["trigger_central"],
            "trigger_worstcase_hout": A["trigger_worstcase"],
            "trigger_central_175sampling": B["trigger_central"],
            "trigger_worstcase_175sampling": B["trigger_worstcase"],
            "trigger_central_lambda1_refinement": C["trigger_central"],
            "trigger_worstcase_lambda1_refinement": C["trigger_worstcase"],
            "delta_trigger_reading_central": result["delta_trigger_reading"],
            "delta_trigger_reading_worstcase": result["delta_trigger_reading_worstcase"],
            "lambda1_ceiling": result["lambda1_ceiling"],
            "hout_central_below_ceiling": 1.0 if A["central_below_ceiling"] else 0.0,
            "hout_worstcase_below_ceiling": 1.0 if A["worstcase_below_ceiling"] else 0.0,
            "175sampling_central_below_ceiling": 1.0 if B["central_below_ceiling"] else 0.0,
            "175sampling_worstcase_below_ceiling": 1.0 if B["worstcase_below_ceiling"] else 0.0,
            "robust_yes_survives": 1.0 if result["robust_yes_survives"] else 0.0,
            "lambda1_clears_under_conservative_reading": 1.0 if result["lambda1_clears_under_conservative_reading"] else 0.0,
            # per-check booleans
            "self_test_a_hout_anchor": 1.0 if st["checks"]["a_hout_reproduces_204_anchor"] else 0.0,
            "self_test_b_interpretation": 1.0 if st["checks"]["b_interpretation_self_consistent"] else 0.0,
            "self_test_c_collapse_consistent": 1.0 if st["checks"]["c_collapse_at_D1_consistent_with_ratio"] else 0.0,
            "self_test_d_monotone": 1.0 if st["checks"]["d_monotone_trigger_in_acceptance"] else 0.0,
            "self_test_e_nan_clean": 1.0 if st["checks"]["e_nan_clean"] else 0.0,
            "reconcile_175_self_test_passes": 1.0 if st["reconcile_175_self_test_passes"] else 0.0,
        }
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run, name="launch_sigma_175_reconcile", artifact_type="launch-sigma-175-reconcile", data=result)
    except Exception as exc:  # noqa: BLE001
        print(f"[175-reconcile] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:  # noqa: BLE001
            pass


def _print(result: dict[str, Any]) -> None:
    trace = result["trace_readings"]
    A = result["readings"]["A_hout_launch_correct"]
    B = result["readings"]["B_175sampling_conservative"]
    C = result["readings"]["C_lambda1_full_bench_refinement"]
    st = result["self_test"]
    print("\n[175-reconcile] ===== LAUNCH-SIGMA #175-READING RECONCILE (PR #207) =====", flush=True)
    print("  step 1 -- trace both readings:", flush=True)
    print(f"    ratio_175_readings = 10.906 / 5.178 = {result['ratio_175_readings']:.6f}", flush=True)
    print(f"    sqrt(D) = {trace['sqrt_design_effect']:.6f}  | rel gap = {trace['rel_gap_ratio_vs_sqrtD']*100:.3f}%  "
          f"-> ratio_equals_sqrtD = {result['ratio_equals_sqrtD']}", flush=True)
    dec = trace["decomposition"]
    print(f"    decomposition: sigma_L op-point {dec['sigma_L_operating_point_factor']:.4f} x bench sqrt(N) "
          f"{dec['bench_size_sqrtN_factor']:.4f} = {dec['product_reproduces_ratio']:.6f}  (err {dec['decomp_err_vs_ratio']:.2e})", flush=True)
    pb = trace["pure_bench_size_demo"]
    print(f"    pure-bench demo (FIXED lambda=1): 10.906@16384 / 5.4531@65536 = {pb['same_op_bench_ratio']:.4f} "
          f"= sqrt(4) = {pb['pure_sqrt_bench_token_ratio']:.4f}  (the bench-size proof)", flush=True)
    print("  step 2/3 -- acceptance 1-sigma (iid) and re-solved triggers:", flush=True)
    print(f"    A h_out (LAUNCH-CORRECT) : acc_iid={A['acceptance']['acceptance_1sigma_iid']:.4f} "
          f"-> trigger {A['trigger']['trigger_central']:.2f} central / {A['trigger']['trigger_worstcase']:.2f} worst   "
          f"[ceiling {result['lambda1_ceiling']:.2f}]", flush=True)
    print(f"    B 175-samp (CONSERVATIVE): acc_iid={B['acceptance']['acceptance_1sigma_iid']:.4f} "
          f"-> trigger {B['trigger']['trigger_central']:.2f} central / {B['trigger']['trigger_worstcase']:.2f} worst", flush=True)
    print(f"    C lambda=1 refinement    : acc_iid={C['acceptance']['acceptance_1sigma_iid']:.4f} "
          f"-> trigger {C['trigger']['trigger_central']:.2f} central / {C['trigger']['trigger_worstcase']:.2f} worst", flush=True)
    print(f"    delta_trigger_reading (B-A) = {result['delta_trigger_reading']:+.2f} central / "
          f"{result['delta_trigger_reading_worstcase']:+.2f} worst", flush=True)
    print("  step 4 -- HEADLINE:", flush=True)
    print(f"    lambda1_clears_under_conservative_reading (TEST) = {result['lambda1_clears_under_conservative_reading']}", flush=True)
    print(f"    robust_yes_survives = {result['robust_yes_survives']}  (launch-correct reading = {result['launch_correct_reading']})", flush=True)
    print(f"\n  SELF-TEST (PRIMARY) passes = {st['reconcile_175_self_test_passes']}", flush=True)
    for k, v in st["checks"].items():
        print(f"    [{'ok' if v else 'XX'}] {k}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Launch-sigma #175-reading reconcile (PR #207)")
    ap.add_argument("--out", default=os.path.join(_HERE, "launch_sigma_175_reconcile_results.json"))
    ap.add_argument("--wandb-name", "--wandb_name", default="ubel/launch-sigma-175-reconcile")
    ap.add_argument("--wandb-group", "--wandb_group", default="launch-sigma-175-reconcile")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    b = import_banked()
    trace = trace_readings(b)
    readings = resolve_readings(b)
    verdict = headline_verdict(b, trace, readings)
    st = self_test(b, trace, readings)

    result = _build_result(b, trace, readings, verdict, st)
    result["elapsed_s"] = round(time.time() - t0, 4)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    _print(result)
    print(f"\n[175-reconcile] HANDOFF: {result['handoff']}", flush=True)
    print(f"[175-reconcile] artifacts -> {args.out}", flush=True)
    _log_wandb(args, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
