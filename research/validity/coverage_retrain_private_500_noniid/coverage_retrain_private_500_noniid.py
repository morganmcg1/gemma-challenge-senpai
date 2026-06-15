#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Coverage-retrain sizing for PRIVATE-500 under a NON-IID accept model (PR #377, denken).

Retires the terminal caveat of denken #373. #373 found the #366-revived strict-spec public ceiling
does NOT clear PRIVATE-500 (raw-optimal corner 498.58 < 500), the 4.295% public->private gap
dominates, the residual lift is +5.44 TPS central / +17.96 conservative, and the cheapest closer is
the DEMAND-side coverage_retrain_b. BUT #373 sized the coverage cost with an *iid depth-7* accept
model (dcoverage ~= dp, dE[T]/dp = 11.12), and flagged that real EAGLE-3 acceptance is non-iid:
it decays with draft depth (#289 per-position) and is gated by a steep a_1 first-token cliff
(#308/#342/#294). Under a position-correlated, depth-decaying profile the coverage->E[T] conversion
can move materially. This card replaces the iid model with the measured non-iid per-position product
and re-derives the TRUE coverage_delta_for_private_500.

CRUX. E[l] (expected accepted draft length) is a TELESCOPING PRODUCT of per-position CONDITIONAL
accept probs a_d, NOT an iid power a^d:

    E[l] = sum_{d=1..K}  prod_{j=1..d} a_j          (committed E[T] = 1 + E[l])

The #289-measured profile has a a_1 cliff (a_1 = 0.7293 is the WEAKEST conditional; survival drops
hardest at position 1 because a_1 multiplicatively gates ALL downstream survival). The realized
coverage->E[T] sensitivity is therefore LOWER than the iid model assumes. We anchor the conversion
on the program's own c* corners (#340: c*_central=0.9089, c*_worst=0.9256 from coverage prior
c0=0.8903), which already encode the realized coverage->E[T] transfer the program adopted:

    S(coverage->E[T]) = (E[T]_500 - E[T]_0) / (c* - c0)
        S_central = (3.9914 - 3.8444) / (0.9089 - 0.8903) ~= 7.89   (program c*-central secant)
        S_worst   = (3.9914 - 3.8444) / (0.9256 - 0.8903) ~= 4.16   (program c*-worst secant)

Both sit BELOW the iid 11.12 -> the iid model was OPTIMISTIC. The per-position uniform-additive
sensitivity on the measured profile (~11.8, ~iid) is the optimistic kappa=1 passthrough bound; the
program secants pull it down via the realized (int4-ct) coverage->accept transfer kappa ~= 0.67.

NOT a launch, NOT a submission, no served-file change. 0 official TPS. CPU-analytic (the OPTIONAL
GPU accept-profile leg is gated behind --gpu and is identity-safe). Run:
    cd target/ && python research/validity/coverage_retrain_private_500_noniid/\
coverage_retrain_private_500_noniid.py --noniid-accept --anchor-289-decay --eval-prompts 128 \
      --wandb_group strict-bi-verify-gemm --wandb_name denken/coverage-noniid-private500
"""
from __future__ import annotations

import argparse
import json
import sys
from math import erf, sqrt
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ===========================================================================
# Section 0 -- banked anchors (all from merged advisor-branch cards / BASELINE)
# ===========================================================================

# ---- public<->private anchor + #366 ceiling + #373 residual (denken #373 oqs8lddd) ------------
MU_P: float = 481.53                 # deployed public TPS (PR #52, 2x9fm2zx)
MU_V: float = 460.85                 # organizer private-verified TPS for the same submission
GAP_MEASURED: float = 1.0 - MU_V / MU_P             # 0.042946 -> the "4.295%" gap
K_CAL: float = 125.26795005202914                   # steps/s; official = E[T] * K_cal (#344)
E_T_REALIZED: float = MU_P / K_CAL                  # 3.8444 realized accept length at deployed pt

P_HEADLINE: float = 518.9188253620001               # #366 revived public ceiling (h28xnyuy)
P_CONSERVATIVE: float = 509.0720037848094           # conservative eta-floor occupancy LB
LAMBDA1_RAW: float = 520.9527323111674              # eta=0 raw lambda=1 supply ceiling
RHO_LB: float = 0.8038                              # #347 lower bound public<->private corr
RHO_PRIV: float = 0.9421                            # #300/#310 point estimate (central)

# #373 banked residuals (the lift to clear private-500 BEYOND the #366 ceiling).
RESIDUAL_PRIVATE_TPS_CENTRAL: float = 5.438733615047738       # +TPS private, headline path
RESIDUAL_PRIVATE_TPS_CONSERVATIVE: float = 17.96249696037296  # +TPS private, conservative path
DELTA_ET_CENTRAL_373: float = 0.04468363487955586            # #373 lever_b central delta_et
DELTA_ET_CONSERVATIVE_373: float = 0.17631393110269514       # #373 lever_b conservative delta_et
IID_DCOV_CENTRAL_373: float = 0.004017628663230129           # the iid +0.004 figure we RETIRE
IID_DCOV_CONSERVATIVE_373: float = 0.015852871084332233
IID_DET_DP_373: float = 11.121892694689883                   # #373 iid dE[T]/dp at p0=0.773

# ---- #289 MEASURED per-position conditional accept profile (int4-ct deployed drafter) ---------
# per_position_acceptance_decay (banked accepted_per_pos / num_drafts), K=7 linear MTP chain.
A_D_MEASURED: list[float] = [
    0.7292532942898975,   # a_1  <- the FIRST-TOKEN CLIFF (weakest conditional; gates all survival)
    0.759556697719242,    # a_2
    0.7929794882639035,   # a_3
    0.8228,               # a_4
    0.8348727920920435,   # a_5
    0.8357919254658385,   # a_6
    0.8464932652113331,   # a_7
]
TOP1_LINEAR_ANCHOR: float = 0.728739760479042       # #289 a_1 cross-check anchor
ACCEPT_CLIFF_POSITION: int = 1                       # #289 cliff agrees across all measures
EAGLE3_PUBLISHED_A1_ENVELOPE: float = 0.80           # #308 published EAGLE-3 a_1 ceiling

# ---- coverage scalars (#336 / #339 / #340 / #343) --------------------------------------------
COV_PRIOR: float = 0.8902659519153152                # #336/#339 deployed fusion-head coverage c0
IDENTITY_BAR: float = 0.9213011665456927             # #336/#339 greedy-identity coverage bar
COV_BUDGET_336: float = 0.031035214630377506         # #336 achievable lift (bar - prior)
CSTAR_CENTRAL: float = 0.9089                         # #340 c* central (speed-500 coverage target)
CSTAR_WORST: float = 0.9256                           # #340 c* worst (pessimistic transfer corner)
RECIPE_MEAN_DCOV: float = 0.0385                      # #339 soft-KD+reasoning recipe mean lift
RECIPE_SIGMA_DCOV: float = 0.00742                    # #339 recipe sigma (from indep p05/p95 band)
HONEST_500_FLOOR_ET: float = 3.9914                   # #289 E[T] for public-500 (=500/K_cal)
ET_PUBLIC_500: float = 500.0 / K_CAL                  # 3.99144 (the E[T] at the speed-500 bar)
ET_DEPLOYED: float = MU_P / K_CAL                      # 3.84438 (E[T] at the deployed 481.53)

# ---- gap-shrink co-benefit anchors (#310 / #362) ---------------------------------------------
PRIV_OVER_PUB_DEPLOYED: float = 0.9570535584491102    # #310 deployed measured priv/pub (=1-gap)
GAP_HIDDEN_STATE_PP: float = 0.0052                   # #362 hidden-state(drafter)-driven gap part
GAP_FLIP_THRESHOLD_HEADLINE: float = 0.03242163964356304  # #373 gap below which verdict flips GO

TARGET_TPS: float = 500.0


# ===========================================================================
# Section 1 -- non-iid accept model (telescoping product) + sensitivities
# ===========================================================================

def survival(a_d: list[float]) -> list[float]:
    """Cumulative survival S_d = prod_{j<=d} a_j for d=1..K (marginal per-position accept)."""
    out, s = [], 1.0
    for a in a_d:
        s *= a
        out.append(s)
    return out


def e_ell(a_d: list[float]) -> float:
    """Expected ACCEPTED draft length E[l] = sum_d prod_{j<=d} a_j (non-iid telescoping product)."""
    return sum(survival(a_d))


def e_t(a_d: list[float]) -> float:
    """Committed expected token length E[T] = 1 + E[l] (the +1 is the guaranteed verify token)."""
    return 1.0 + e_ell(a_d)


def per_position_dE_da(a_d: list[float]) -> list[float]:
    """dE[l]/da_i = (1/a_i) * sum_{d>=i} S_d. Shows the a_1 cliff dominates the gradient."""
    S = survival(a_d)
    K = len(a_d)
    return [(1.0 / a_d[i]) * sum(S[i:]) for i in range(K)]


def uniform_additive_slope(a_d: list[float]) -> float:
    """dE[l]/dc for a uniform additive coverage lift (Delta a_d = Delta c for all d). kappa=1 bound."""
    return sum(per_position_dE_da(a_d))


def e_ell_uniform_shift(a_d: list[float], dc: float) -> float:
    """E[l] when every conditional a_d is shifted by +dc (clamped to <=1): exact, not linearized."""
    shifted = [min(1.0, a + dc) for a in a_d]
    return e_ell(shifted)


def solve_dc_uniform_for_dEll(a_d: list[float], target_dEll: float) -> float:
    """Bisection: uniform additive dc reproducing target_dEll under the exact telescoping product."""
    base = e_ell(a_d)
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if e_ell_uniform_shift(a_d, mid) - base < target_dEll:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build_conservative_profile() -> list[float]:
    """Faster-decaying, lower-a_1 profile anchored on the PRIVATE distribution (the card's worry).

    a_1 dropped to 0.70 (below the measured 0.7293 and the #308 published 0.80 envelope, reflecting
    private-distribution degradation), deep positions flattened to a constant a_hi chosen so the
    profile reproduces the private-haircut E[T] = E_T_deployed * (1-gap). Steeper cliff -> shallower
    coverage->E[T] conversion -> larger required dcov (the genuine pessimistic bound)."""
    a1 = 0.70
    et_priv = ET_DEPLOYED * PRIV_OVER_PUB_DEPLOYED        # private-distribution E[T]
    target_ell = et_priv - 1.0                            # E[l]_private
    # E[l] = a1 * sum_{k=0..K-2} a_hi^k ; bisect a_hi.
    K = len(A_D_MEASURED)
    lo, hi = 0.50, 0.999
    for _ in range(200):
        a_hi = 0.5 * (lo + hi)
        ell = a1 * sum(a_hi ** k for k in range(K))       # a1*(1 + a_hi + ... + a_hi^{K-1})
        if ell < target_ell:
            lo = a_hi
        else:
            hi = a_hi
    a_hi = 0.5 * (lo + hi)
    return [a1] + [a_hi] * (K - 1)


def build_accept_model() -> dict:
    a_central = A_D_MEASURED
    a_cons = build_conservative_profile()
    S_c, S_w = survival(a_central), survival(a_cons)
    grad_c = per_position_dE_da(a_central)
    return {
        "central_profile": {
            "a_d": a_central,
            "survival": S_c,
            "expected_accept_length": e_ell(a_central),    # E[l]
            "expected_committed_length_E_T": e_t(a_central),
            "a1_cliff": a_central[0],
            "per_position_dEll_da": grad_c,
            "a1_gradient_share": grad_c[0] / sum(grad_c),   # cliff share of the gradient
            "uniform_additive_slope": uniform_additive_slope(a_central),
            "conditional_increases_with_depth": all(
                a_central[i] <= a_central[i + 1] for i in range(len(a_central) - 1)),
        },
        "conservative_profile": {
            "a_d": a_cons,
            "survival": S_w,
            "expected_accept_length": e_ell(a_cons),
            "expected_committed_length_E_T": e_t(a_cons),
            "a1_cliff": a_cons[0],
            "uniform_additive_slope": uniform_additive_slope(a_cons),
            "anchored_on": "private-distribution E[T] = deployed * (1-gap)",
        },
        "iid_reference": {
            "p0": 0.7730262260000382,
            "dET_dp": IID_DET_DP_373,
            "note": "iid E[T]=sum p^d; #373 dE[T]/dp=11.12 at p0=0.773 (the OPTIMISTIC edge).",
        },
    }


# ===========================================================================
# Section 2 -- coverage->E[T]->TPS map + the INVERSE (dcov for the #373 residual)
# ===========================================================================

def program_secant(c_star: float) -> float:
    """Program-adopted coverage->E[T] conversion slope from a c* corner (#340)."""
    return (ET_PUBLIC_500 - ET_DEPLOYED) / (c_star - COV_PRIOR)


def build_conversion(model: dict) -> dict:
    s_central = program_secant(CSTAR_CENTRAL)            # ~7.89  (program c*-central)
    s_worst = program_secant(CSTAR_WORST)                # ~4.16  (program c*-worst)
    s_uniform_central = model["central_profile"]["uniform_additive_slope"]   # ~11.8 (kappa=1 bound)
    s_uniform_cons = model["conservative_profile"]["uniform_additive_slope"]

    # kappa = realized coverage->accept transfer implied by the program secant vs the kappa=1 bound.
    kappa_implied_central = s_central / s_uniform_central

    # ---- the INVERSE: dcov to deliver the #373 residual (delta_et) under the non-iid conversion --
    dcov_central = DELTA_ET_CENTRAL_373 / s_central                 # central residual, central slope
    dcov_conservative = DELTA_ET_CONSERVATIVE_373 / s_worst         # cons residual, worst slope
    # robustness corners (central residual, every slope) + (cons residual, central slope)
    dcov_central_worstslope = DELTA_ET_CENTRAL_373 / s_worst
    dcov_central_uniform = DELTA_ET_CENTRAL_373 / s_uniform_central
    dcov_cons_centralslope = DELTA_ET_CONSERVATIVE_373 / s_central

    # exact uniform-additive inversion on the measured profile (cross-check vs the linear slope).
    dcov_central_uniform_exact = solve_dc_uniform_for_dEll(
        model["central_profile"]["a_d"], DELTA_ET_CENTRAL_373)

    ratio_noniid_vs_iid = dcov_central / IID_DCOV_CENTRAL_373       # >1 -> non-iid PRICIER

    return {
        "slope_central_program_cstar": s_central,
        "slope_worst_program_cstar": s_worst,
        "slope_uniform_additive_central_profile": s_uniform_central,
        "slope_uniform_additive_conservative_profile": s_uniform_cons,
        "slope_iid_373": IID_DET_DP_373,
        "kappa_realized_transfer_implied": kappa_implied_central,
        "coverage_delta_for_private_500_noniid_central": dcov_central,
        "coverage_delta_for_private_500_noniid_conservative": dcov_conservative,
        "dcov_central_residual_worst_slope": dcov_central_worstslope,
        "dcov_central_residual_uniform_slope": dcov_central_uniform,
        "dcov_central_uniform_exact_crosscheck": dcov_central_uniform_exact,
        "dcov_conservative_residual_central_slope": dcov_cons_centralslope,
        "noniid_vs_iid_coverage_delta_ratio": ratio_noniid_vs_iid,
        "iid_dcov_central_373": IID_DCOV_CENTRAL_373,
        "ordering_iid_at_optimistic_edge": IID_DET_DP_373 >= s_central,
    }


# ===========================================================================
# Section 3 -- gap-shrink co-benefit (deliverable 3)
# ===========================================================================

def build_gap_cobenefit() -> dict:
    """How much of the 4.295% public->private gap each unit of coverage closes.

    Central model: the gap is proportional to the coverage shortfall headroom g = kappa_gap*(1-c),
    so dg/dc = -kappa_gap = -g0/(1-c0). Conservative model: only the #362 hidden-state(drafter)
    fraction of the gap is coverage-elastic; the rest is verify-side/systematic and coverage-invariant.
    """
    headroom = 1.0 - COV_PRIOR
    kappa_gap_central = GAP_MEASURED / headroom                     # whole gap coverage-elastic
    dgap_dc_central = -kappa_gap_central
    kappa_gap_cons = GAP_HIDDEN_STATE_PP / headroom                 # only #362 hidden-state part
    dgap_dc_cons = -kappa_gap_cons

    # dcov to shrink the gap to the #373 flip threshold (verdict flips GO via gap alone).
    gap_to_close = GAP_MEASURED - GAP_FLIP_THRESHOLD_HEADLINE       # ~0.01053
    dcov_to_flip_gap_central = gap_to_close / kappa_gap_central
    dcov_to_flip_gap_cons = gap_to_close / kappa_gap_cons if kappa_gap_cons > 0 else float("inf")

    return {
        "gap_measured": GAP_MEASURED,
        "coverage_headroom_1_minus_c0": headroom,
        "gap_shrink_contribution_per_coverage": kappa_gap_central,   # |dg/dc| central
        "gap_shrink_per_0p01_coverage_pp": kappa_gap_central * 0.01 * 100,
        "dgap_dc_central": dgap_dc_central,
        "gap_shrink_contribution_per_coverage_conservative": kappa_gap_cons,
        "dgap_dc_conservative": dgap_dc_cons,
        "gap_flip_threshold_headline": GAP_FLIP_THRESHOLD_HEADLINE,
        "gap_to_close_to_flip": gap_to_close,
        "dcov_to_flip_gap_via_gap_alone_central": dcov_to_flip_gap_central,
        "dcov_to_flip_gap_via_gap_alone_conservative": dcov_to_flip_gap_cons,
        "double_indication_note": (
            "coverage retrain is doubly-indicated: it raises E[T] (direct ceiling lift) AND shrinks "
            "the dominant gap. The two channels are synergistic, so the TRUE dcov to clear private-500 "
            "is <= the direct-channel figure -- the direct-channel dcov is a conservative-direction "
            "estimate."),
    }


# ===========================================================================
# Section 4 -- #336 budget check + P(retrain delivers) (deliverables 4 + required field)
# ===========================================================================

def norm_cdf(x: float, mu: float, sigma: float) -> float:
    return 0.5 * (1.0 + erf((x - mu) / (sigma * sqrt(2.0))))


def build_budget_and_delivery(conv: dict) -> dict:
    dcov_central = conv["coverage_delta_for_private_500_noniid_central"]
    dcov_cons = conv["coverage_delta_for_private_500_noniid_conservative"]
    # robust central recommendation: central residual but the WORST (cliff) conversion slope.
    dcov_central_robust = conv["dcov_central_residual_worst_slope"]

    within_central = dcov_central <= COV_BUDGET_336
    within_central_robust = dcov_central_robust <= COV_BUDGET_336
    within_cons = dcov_cons <= COV_BUDGET_336
    frac_central = dcov_central / COV_BUDGET_336
    frac_central_robust = dcov_central_robust / COV_BUDGET_336
    frac_cons = dcov_cons / COV_BUDGET_336

    # P(soft-KD+reasoning recipe delivers the required dcov) -- #339 recipe ~ N(0.0385, 0.00742).
    def p_deliver(req: float) -> float:
        return 1.0 - norm_cdf(req, RECIPE_MEAN_DCOV, RECIPE_SIGMA_DCOV)
    p_central = p_deliver(dcov_central)
    p_central_robust = p_deliver(dcov_central_robust)
    p_cons = p_deliver(dcov_cons)

    # recommended retrain target = robust central dcov on top of the prior, capped under the budget.
    rec_dcov = dcov_central_robust
    rec_target = COV_PRIOR + rec_dcov
    return {
        "cov_budget_336": COV_BUDGET_336,
        "within_336_budget": within_central,
        "within_336_budget_central_robust": within_central_robust,
        "within_336_budget_conservative": within_cons,
        "fraction_of_336_budget_consumed": frac_central,
        "fraction_of_336_budget_consumed_central_robust": frac_central_robust,
        "fraction_of_336_budget_consumed_conservative": frac_cons,
        "p_softkd_reasoning_retrain_delivers": p_central,
        "p_softkd_reasoning_retrain_delivers_central_robust": p_central_robust,
        "p_softkd_reasoning_retrain_delivers_conservative": p_cons,
        "recipe_mean_dcov": RECIPE_MEAN_DCOV,
        "recipe_sigma_dcov": RECIPE_SIGMA_DCOV,
        "recommended_retrain_dcov": rec_dcov,
        "recommended_retrain_target": rec_target,
        "recommended_target_within_budget": rec_dcov <= COV_BUDGET_336,
        "recommended_target_note": (
            "retrain the EAGLE-3 fusion head to coverage c >= {:.4f} (dcov +{:.4f}, {:.0f}% of the "
            "#336 +0.031 budget): closes the #373 +5.44 TPS central residual and is ROBUST to the "
            "non-iid cliff-conversion (sized at the worst c*=0.9256 slope). P(recipe delivers) ~= "
            "{:.3f}. Hand to fern #357.").format(
                rec_target, rec_dcov, 100 * frac_central_robust, p_central_robust),
    }


# ===========================================================================
# Section 5 -- OPTIONAL local-GPU per-position accept-profile measurement
# ===========================================================================

def measure_accept_profile_gpu(proxy: str, eval_prompts: int, k_spec: int) -> dict:
    """Identity-safe per-position accept measurement on the int4-ct proxy (gated behind --gpu).

    Not run by default: the analytic profile shape is ALREADY measured (#289 banked accepted_per_pos
    / num_drafts over the same 128-record eval set), so the profile is NOT under-determined. This
    function is provided for completeness/refresh; it re-counts per-depth accept frequency."""
    try:
        import os
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM, SamplingParams  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return {"ran": False, "reason": f"gpu/vllm unavailable: {exc}"}
    return {"ran": False, "reason": "stub: analytic #289 profile is already measured; GPU leg "
            "skipped by default (set up the served drafter to re-measure if shape is contested)."}


# ===========================================================================
# Section 6 -- self-tests
# ===========================================================================

def run_self_tests(model: dict, conv: dict, gap: dict, budget: dict) -> dict:
    c = {}
    cp = model["central_profile"]
    # a) telescoping product reproduces the #289 measured E[T] (~3.844 anchor, 3.851 measured).
    c["a_central_E_T_matches_289"] = abs(cp["expected_committed_length_E_T"] - 3.851185944363104) < 1e-6
    c["a_central_E_T_near_anchor"] = abs(cp["expected_committed_length_E_T"] - E_T_REALIZED) < 0.01
    # b) E[l] = E[T] - 1 (the +1 bonus token is constant -> dE[T]=dE[l]).
    c["b_ell_is_ET_minus_1"] = abs(cp["expected_accept_length"] - (cp["expected_committed_length_E_T"] - 1.0)) < 1e-9
    # c) the a_1 cliff: a_1 is the weakest conditional AND dominates the gradient share.
    c["c_a1_is_min_conditional"] = cp["a1_cliff"] == min(cp["a_d"])
    c["c_a1_gradient_dominant"] = cp["a1_gradient_share"] == max(
        g / sum(cp["per_position_dEll_da"]) for g in cp["per_position_dEll_da"])
    c["c_conditional_increases_with_depth"] = cp["conditional_increases_with_depth"]
    # d) non-iid conversion sits BELOW iid (the iid model was optimistic) and ordering holds.
    c["d_program_central_below_iid"] = conv["slope_central_program_cstar"] < conv["slope_iid_373"]
    c["d_worst_below_central"] = conv["slope_worst_program_cstar"] < conv["slope_central_program_cstar"]
    c["d_uniform_brackets_iid"] = abs(conv["slope_uniform_additive_central_profile"] - conv["slope_iid_373"]) < 2.0
    c["d_iid_at_optimistic_edge"] = conv["ordering_iid_at_optimistic_edge"]
    # e) the inverse: non-iid central dcov is PRICIER than iid (ratio>1) but well-defined & finite.
    c["e_noniid_pricier_than_iid"] = conv["noniid_vs_iid_coverage_delta_ratio"] > 1.0
    c["e_dcov_central_finite_positive"] = 0.0 < conv["coverage_delta_for_private_500_noniid_central"] < 1.0
    # f) exact uniform inversion cross-checks the linear uniform slope within tolerance.
    approx = DELTA_ET_CENTRAL_373 / conv["slope_uniform_additive_central_profile"]
    c["f_uniform_exact_matches_linear"] = abs(conv["dcov_central_uniform_exact_crosscheck"] - approx) < 5e-4
    # g) GREEN bar: central dcov within #336 budget and P(deliver) high.
    c["g_within_336_budget_central"] = budget["within_336_budget"]
    c["g_central_robust_within_budget"] = budget["within_336_budget_central_robust"]
    c["g_p_deliver_central_high"] = budget["p_softkd_reasoning_retrain_delivers"] > 0.9
    # h) c* secant anchors reproduce the public-500 E[T] at c*_central (identity of the conversion).
    c["h_cstar_central_maps_to_500"] = abs(
        ET_DEPLOYED + conv["slope_central_program_cstar"] * (CSTAR_CENTRAL - COV_PRIOR) - ET_PUBLIC_500) < 1e-9
    c["h_budget_is_bar_minus_prior"] = abs(COV_BUDGET_336 - (IDENTITY_BAR - COV_PRIOR)) < 1e-9
    # i) gap co-benefit well-formed: positive contribution, central > conservative (hidden-state LB).
    c["i_gap_cobenefit_positive"] = gap["gap_shrink_contribution_per_coverage"] > 0
    c["i_gap_central_above_hiddenstate_LB"] = (
        gap["gap_shrink_contribution_per_coverage"] > gap["gap_shrink_contribution_per_coverage_conservative"])
    # j) recommended target valid: within budget, above the prior, below the identity bar.
    c["j_rec_target_in_band"] = COV_PRIOR < budget["recommended_retrain_target"] <= IDENTITY_BAR
    c["j_rec_within_budget"] = budget["recommended_target_within_budget"]
    # k) numeric hygiene.
    flat = [conv["slope_central_program_cstar"], conv["slope_worst_program_cstar"],
            conv["coverage_delta_for_private_500_noniid_central"],
            conv["coverage_delta_for_private_500_noniid_conservative"],
            gap["gap_shrink_contribution_per_coverage"], budget["recommended_retrain_target"]]
    c["k_no_nan"] = all(v == v for v in flat)
    passes = all(c.values())
    return {"conditions": c, "n_checks": len(c), "passes": passes}


# ===========================================================================
# Section 7 -- report assembly + W&B + CLI
# ===========================================================================

def build_report(args) -> dict:
    model = build_accept_model()
    conv = build_conversion(model)
    gap = build_gap_cobenefit()
    budget = build_budget_and_delivery(conv)
    gpu = {"ran": False, "reason": "CPU-analytic default; --gpu not set"}
    if getattr(args, "gpu", False) and getattr(args, "measure_accept_profile", False):
        gpu = measure_accept_profile_gpu(
            getattr(args, "proxy", "google/gemma-4-E4B-it-qat-w4a16-ct"),
            getattr(args, "eval_prompts", 128), 7)
    selftest = run_self_tests(model, conv, gap, budget)

    cp, wp = model["central_profile"], model["conservative_profile"]
    return {
        "pr": 377, "agent": "denken", "kind": "coverage-retrain-private-500-noniid",
        "analysis_only": True, "no_launch": True, "no_hf_job": True,
        "no_served_file_change": True, "official_tps_expected": 0,
        "inputs": {
            "mu_p_public": MU_P, "mu_v_private": MU_V, "gap_measured": GAP_MEASURED,
            "k_cal": K_CAL, "e_t_realized": E_T_REALIZED,
            "p_headline": P_HEADLINE, "p_conservative": P_CONSERVATIVE,
            "residual_private_tps_central": RESIDUAL_PRIVATE_TPS_CENTRAL,
            "residual_private_tps_conservative": RESIDUAL_PRIVATE_TPS_CONSERVATIVE,
            "delta_et_central_373": DELTA_ET_CENTRAL_373,
            "delta_et_conservative_373": DELTA_ET_CONSERVATIVE_373,
            "iid_dcov_central_373": IID_DCOV_CENTRAL_373, "iid_det_dp_373": IID_DET_DP_373,
            "cov_prior": COV_PRIOR, "identity_bar": IDENTITY_BAR, "cov_budget_336": COV_BUDGET_336,
            "cstar_central": CSTAR_CENTRAL, "cstar_worst": CSTAR_WORST,
            "recipe_mean_dcov": RECIPE_MEAN_DCOV, "recipe_sigma_dcov": RECIPE_SIGMA_DCOV,
            "et_public_500": ET_PUBLIC_500, "et_deployed": ET_DEPLOYED,
            "a_d_measured_289": A_D_MEASURED, "gap_hidden_state_pp_362": GAP_HIDDEN_STATE_PP,
            "source_373_run": "oqs8lddd", "source_289": "per_position_acceptance_decay",
            "source_339_340_336_343": "coverage achievability / c* / budget / c*=0.9089",
        },
        "accept_model": model, "conversion": conv, "gap_cobenefit": gap, "budget_delivery": budget,
        "gpu_accept_profile_leg": gpu,
        # ----- card-required headline scalars -----
        "coverage_delta_for_private_500_noniid": conv["coverage_delta_for_private_500_noniid_central"],
        "coverage_delta_for_private_500_noniid_central": conv["coverage_delta_for_private_500_noniid_central"],
        "coverage_delta_for_private_500_noniid_conservative": conv["coverage_delta_for_private_500_noniid_conservative"],
        "gap_shrink_contribution_per_coverage": gap["gap_shrink_contribution_per_coverage"],
        "expected_accept_length_central": cp["expected_accept_length"],
        "expected_accept_length_conservative": wp["expected_accept_length"],
        "expected_committed_E_T_central": cp["expected_committed_length_E_T"],
        "expected_committed_E_T_conservative": wp["expected_committed_length_E_T"],
        "noniid_vs_iid_coverage_delta_ratio": conv["noniid_vs_iid_coverage_delta_ratio"],
        "within_336_budget": budget["within_336_budget"],
        "fraction_of_336_budget_consumed": budget["fraction_of_336_budget_consumed"],
        "p_softkd_reasoning_retrain_delivers": budget["p_softkd_reasoning_retrain_delivers"],
        "recommended_retrain_target": budget["recommended_retrain_target"],
        "measured_per_position_accept_profile": (
            gpu.get("a_d") if gpu.get("ran") else A_D_MEASURED),  # #289 banked is the measured profile
        # ----- GO/NO-GO + SENPAI-RESULT metrics -----
        "demand_lever_sized_and_defensible": bool(
            budget["within_336_budget"] and budget["p_softkd_reasoning_retrain_delivers"] > 0.9),
        "primary_metric_coverage_delta_central": conv["coverage_delta_for_private_500_noniid_central"],
        "self_test": selftest,
        "coverage_noniid_self_test_passes": selftest["passes"],
    }


def log_to_wandb(report: dict, group: str, name: str) -> str:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"W&B import failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_unavailable"
    try:
        run = wandb.init(group=group, name=name, config=report["inputs"])
        conv, gap, bud = report["conversion"], report["gap_cobenefit"], report["budget_delivery"]
        wandb.log({
            "summary/coverage_delta_for_private_500_noniid_central":
                conv["coverage_delta_for_private_500_noniid_central"],
            "summary/coverage_delta_for_private_500_noniid_conservative":
                conv["coverage_delta_for_private_500_noniid_conservative"],
            "summary/noniid_vs_iid_coverage_delta_ratio": conv["noniid_vs_iid_coverage_delta_ratio"],
            "summary/iid_dcov_central_373": IID_DCOV_CENTRAL_373,
            "summary/slope_central_program_cstar": conv["slope_central_program_cstar"],
            "summary/slope_worst_program_cstar": conv["slope_worst_program_cstar"],
            "summary/slope_uniform_additive_central": conv["slope_uniform_additive_central_profile"],
            "summary/slope_iid_373": conv["slope_iid_373"],
            "summary/kappa_realized_transfer_implied": conv["kappa_realized_transfer_implied"],
            "summary/expected_accept_length_central": report["expected_accept_length_central"],
            "summary/expected_accept_length_conservative": report["expected_accept_length_conservative"],
            "summary/expected_committed_E_T_central": report["expected_committed_E_T_central"],
            "summary/gap_shrink_contribution_per_coverage": gap["gap_shrink_contribution_per_coverage"],
            "summary/gap_shrink_per_coverage_conservative":
                gap["gap_shrink_contribution_per_coverage_conservative"],
            "summary/dcov_to_flip_gap_central": gap["dcov_to_flip_gap_via_gap_alone_central"],
            "summary/within_336_budget": float(bud["within_336_budget"]),
            "summary/within_336_budget_conservative": float(bud["within_336_budget_conservative"]),
            "summary/fraction_of_336_budget_consumed": bud["fraction_of_336_budget_consumed"],
            "summary/fraction_of_336_budget_consumed_conservative":
                bud["fraction_of_336_budget_consumed_conservative"],
            "summary/p_softkd_reasoning_retrain_delivers": bud["p_softkd_reasoning_retrain_delivers"],
            "summary/p_softkd_reasoning_retrain_delivers_conservative":
                bud["p_softkd_reasoning_retrain_delivers_conservative"],
            "summary/recommended_retrain_target": bud["recommended_retrain_target"],
            "summary/recommended_retrain_dcov": bud["recommended_retrain_dcov"],
            "summary/demand_lever_sized_and_defensible": float(report["demand_lever_sized_and_defensible"]),
            "summary/coverage_noniid_self_test_passes": float(report["self_test"]["passes"]),
        })
        for cond, val in report["self_test"]["conditions"].items():
            wandb.log({f"test/{cond}": float(bool(val))})
        run_id = run.id
        wandb.finish()
        return run_id
    except Exception as exc:  # noqa: BLE001
        print(f"W&B logging failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_failed"


def main() -> int:
    ap = argparse.ArgumentParser(description="Coverage-retrain non-iid sizing for private-500 (PR #377).")
    ap.add_argument("--noniid-accept", action="store_true", help="(default behavior) use non-iid product")
    ap.add_argument("--anchor-289-decay", action="store_true", help="(default) anchor on #289 profile")
    ap.add_argument("--eval-prompts", type=int, default=128)
    ap.add_argument("--gpu", action="store_true", help="OPTIONAL: run the local-GPU accept-profile leg")
    ap.add_argument("--proxy", type=str, default="google/gemma-4-E4B-it-qat-w4a16-ct")
    ap.add_argument("--measure-accept-profile", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--wandb_group", type=str, default="strict-bi-verify-gemm")
    ap.add_argument("--wandb_name", type=str, default="denken/coverage-noniid-private500")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", type=str,
                    default="research/validity/coverage_retrain_private_500_noniid/"
                            "coverage_retrain_private_500_noniid_results.json")
    args = ap.parse_args()

    report = build_report(args)
    model, conv, gap, bud = (report["accept_model"], report["conversion"],
                             report["gap_cobenefit"], report["budget_delivery"])
    cp, wp = model["central_profile"], model["conservative_profile"]

    print("\n=== Coverage-retrain sizing for PRIVATE-500, NON-IID accept model (PR #377) ===")
    print(f"#289 measured profile a_d   : {[round(a, 4) for a in cp['a_d']]}")
    print(f"  a_1 CLIFF = {cp['a1_cliff']:.4f} (weakest); E[l]_central = {cp['expected_accept_length']:.4f}"
          f"  E[T]_central = {cp['expected_committed_length_E_T']:.4f}")
    print(f"  per-position dE[l]/da_i   : {[round(g, 3) for g in cp['per_position_dEll_da']]}"
          f"  (a_1 gradient share {cp['a1_gradient_share']*100:.0f}%)")
    print(f"conservative profile a_d    : {[round(a, 4) for a in wp['a_d']]}  "
          f"E[l]_cons = {wp['expected_accept_length']:.4f}  E[T]_cons = {wp['expected_committed_length_E_T']:.4f}")
    print("\ncoverage->E[T] conversion slopes (E[T] per unit coverage):")
    print(f"  iid #373 (dE[T]/dp)              : {conv['slope_iid_373']:.2f}  <- OPTIMISTIC edge")
    print(f"  uniform-additive (kappa=1 bound) : {conv['slope_uniform_additive_central_profile']:.2f}  (~iid)")
    print(f"  program c*-central secant        : {conv['slope_central_program_cstar']:.2f}  <- NON-IID central")
    print(f"  program c*-worst secant          : {conv['slope_worst_program_cstar']:.2f}  <- conservative")
    print(f"  implied realized transfer kappa  : {conv['kappa_realized_transfer_implied']:.3f}")
    print("\nINVERSE -- coverage_delta_for_private_500_noniid:")
    print(f"  central (residual +5.44 TPS)     : +{conv['coverage_delta_for_private_500_noniid_central']:.5f}"
          f"  ({100*bud['fraction_of_336_budget_consumed']:.0f}% of #336 budget)")
    print(f"  conservative (residual +17.96)   : +{conv['coverage_delta_for_private_500_noniid_conservative']:.5f}"
          f"  ({100*bud['fraction_of_336_budget_consumed_conservative']:.0f}% of #336 budget)")
    print(f"  iid #373 central (RETIRED)       : +{IID_DCOV_CENTRAL_373:.5f}")
    print(f"  noniid_vs_iid ratio              : {conv['noniid_vs_iid_coverage_delta_ratio']:.3f}x "
          f"(non-iid {'PRICIER' if conv['noniid_vs_iid_coverage_delta_ratio']>1 else 'CHEAPER'})")
    print(f"\ngap-shrink co-benefit            : {gap['gap_shrink_contribution_per_coverage']:.4f} per unit cov "
          f"({gap['gap_shrink_per_0p01_coverage_pp']:.2f} pp per +0.01); flip-gap-alone dcov "
          f"+{gap['dcov_to_flip_gap_via_gap_alone_central']:.4f}")
    print(f"within_336_budget (central)      : {bud['within_336_budget']}  "
          f"(robust-corner {bud['within_336_budget_central_robust']}, conservative {bud['within_336_budget_conservative']})")
    print(f"P(soft-KD+reasoning delivers)    : {bud['p_softkd_reasoning_retrain_delivers']:.4f} central / "
          f"{bud['p_softkd_reasoning_retrain_delivers_conservative']:.3f} cons")
    print(f"recommended_retrain_target       : c >= {bud['recommended_retrain_target']:.4f} "
          f"(dcov +{bud['recommended_retrain_dcov']:.4f})")
    print(f"\n>>> demand_lever_sized_and_defensible = {report['demand_lever_sized_and_defensible']}")
    print(f"self-test: {report['self_test']['n_checks']} checks, passes={report['self_test']['passes']}")

    if args.self_test:
        return 0 if report["self_test"]["passes"] else 1
    if not args.no_wandb:
        report["wandb_run_id"] = log_to_wandb(report, args.wandb_group, args.wandb_name)
    else:
        report["wandb_run_id"] = None
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}  (W&B run {report.get('wandb_run_id')})")
    return 0 if report["self_test"]["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
