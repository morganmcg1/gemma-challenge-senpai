#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #339 (student lawine) -- Demand-half retrain clear-probability: convolve #336 Delta-cov bands into P(clears 0.9213).

WHAT THIS CARD DOES (0-GPU, 0-TPS, no training, no model forward, no served-file change, no HF Job)
---------------------------------------------------------------------------------------------------
lawine #336 (krroookz, MERGED) sized the DEMAND-half EAGLE-3 head retrain recipe: the honest top-4 ROOT
coverage prior is 0.8903 (lawine #330 hfrscdai), the identity build bar is 0.9213, and the gap the
retrain must close is min_aggregate_lift_required = 0.031035. #336 ranked the head-training levers with
literature-grounded Delta-cov bands (soft-KD top-k +0.030 [0.015,0.045]; reasoning-data +0.025
[0.010,0.040]; deeper head +0.012 [0.005,0.020]; on-policy TTT +0.002 [0.000,0.005]) and reported the
recommended soft-KD + reasoning-data COMBINATION as central +0.0385 band [+0.0175,+0.0595] under a 0.70
non-additivity haircut (combo = 0.70 * naive-sum 0.055). #336's verdict was REACHABLE-MARGINAL: "central
clears, low band misses."

But "REACHABLE-MARGINAL" is a VERDICT, not a PROBABILITY. The human's #319 build/measure decision needs
the retrain-ROI number: IF we spend the GPU on the retrain, what is P(it actually clears the bar)? This
card turns #336's marginal verdict into that decision-grade probability. It treats each lever's #336
Delta-cov band as a distribution, convolves the recipe's levers into a posterior over post-retrain
coverage cov_post = cov_prior + Delta-cov, and reports:

  * p_clears_identity_bar_0p9213 = P(cov_post >= 0.9213)              -- the strict identity bar (TEST)
  * p_clears_speed500_bar(c*)    = P(cov_post >= c*)  for c* in {0.88, 0.90, 0.908, 0.92}
                                                                      -- the looser SPEED-envelope bar

under (a) INDEPENDENT levers (base case) and (b) +0.5 POSITIVELY-CORRELATED levers (conservative bound,
since soft-KD and reasoning-data both target the same mmlu_pro reasoning-CoT drag and may not stack
independently). It also reports the full-4-lever optimistic upper bound, the cov_post 5/50/95 percentiles,
a uniform-over-band robustness alternative, and a Monte-Carlo cross-check of the closed-form.

KEY STRUCTURAL FINDING (surfaced by re-doing #336's band as a convolution): #336's reported combo band
[+0.0175,+0.0595] (sigma = (high-low)/4 = 0.0105) was built by ADDING the two lever band ENDPOINTS
(low+low, high+high) then applying the 0.70 haircut -- i.e. it is the COMONOTONIC (rho=1) spread. Proper
INDEPENDENT convolution of the two lever bands gives a TIGHTER recipe sigma (0.0074) and therefore a
HIGHER clear-probability. So the rho=1 case of this card reproduces #336's band exactly (validation), and
the independent base case is strictly more optimistic. The mean is the SAME 0.0385 in every correlation
case (correlation widens the spread, it does not move the mean); cov_post mean = 0.8903 + 0.0385 = 0.9288.

LOCAL CPU-only analytic card. No GPU / vLLM / model forward / training / checkpoint / HF Job / submission
/ served-file change / publish. NOT a launch. NOT open2. NOT a build. BASELINE stays 481.53 (0 TPS).
Greedy/PPL untouched (the retrain TARGET is greedy-IDENTICAL by construction: EAGLE-3 emission = verify
argmax, PPL-pinned -- coverage is the SPEED/acceptance axis, not the validity axis). The retrain itself is
human-approval-gated GPU spend (#319 / instructions/training-request.md); THIS card only PRICES that
decision from banked numbers.

PRIMARY metric  retrain_clear_probability_self_test_passes
TEST    metric  p_clears_identity_bar_0p9213          (float; P(cov_post >= 0.9213), INDEPENDENT levers)
REPORT          p_clears_speed500_bar                 (float; P(cov_post >= 0.908), INDEPENDENT levers)
REPORT          retrain_roi_verdict                   (JUSTIFIED | MARGINAL | LIKELY-WASTE)

Reproduce:
    cd target/ && .venv/bin/python \\
        research/validity/eagle3_retrain_clear_probability/eagle3_retrain_clear_probability.py \\
        --self-test --wandb_group eagle3-retrain-clear-probability \\
        --wandb_name lawine/eagle3-retrain-clear-probability
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --------------------------------------------------------------------------- #
# Imported fleet anchors (cited EXACTLY, UNCHANGED; this card re-derives none of them). Cross-checked at
# runtime against the on-disk banked lawine #336 (krroookz) and lawine #330 (hfrscdai) artifacts so there
# is no silent drift. The PR cites the rounded display values; the analytic uses the UNROUNDED banked
# values (the rounded triple 0.8903/0.9213/0.031035 is mutually consistent only after rounding -- the
# unrounded min_lift is EXACTLY identity_bar - cov_prior).
# --------------------------------------------------------------------------- #
# Rounded display constants the PR body cites (import EXACT):
COV_PRIOR_DISPLAY = 0.8903
IDENTITY_BAR_DISPLAY = 0.9213
MIN_LIFT_DISPLAY = 0.031035

# Unrounded banked values used for the probability math:
# lawine #330 (hfrscdai): honest official-eval-weighted unconditional top-4 ROOT coverage prior.
COV_PRIOR = 0.8902659519153152
# lawine #316 (5lnz5jgb) via #323 (ceddxj20); == #336 T_effective: the regime-invariant identity build bar.
IDENTITY_BAR = 0.9213011665456927
# lawine #336 (krroookz) min_aggregate_lift_required == IDENTITY_BAR - COV_PRIOR (exact).
MIN_LIFT_REQUIRED = 0.031035214630377506

BASELINE_TPS = 481.53          # current best summary.json:tps (unchanged; 0-TPS analytic)
CEILING_TPS = 520.95           # stark #325 lambda=1 ceiling (central envelope anchor)

# --------------------------------------------------------------------------- #
# Per-lever Delta-cov bands -- IMPORTED EXACT from lawine #336 (krroookz). Each band [low, high] is modeled
# as a ~+/-2 sigma (95%) interval -> sigma = (high - low) / 4, under a truncated normal (Delta-cov >= 0).
# These are #336's literature-grounded EXPECTED-VALUE PRIORS, not measurements; this card converts their
# uncertainty into a probability, it does not re-estimate them.
# --------------------------------------------------------------------------- #
LEVERS: dict[str, dict[str, Any]] = {
    "soft_kd_topk_distill":     {"central": 0.030, "band": [0.015, 0.045], "citation": "DistillSpec arXiv:2310.08461 / OSD arXiv:2310.07177 / Medusa arXiv:2401.10774"},
    "more_reasoning_root_data": {"central": 0.025, "band": [0.010, 0.040], "citation": "EAGLE-3 arXiv:2503.01840"},
    "deeper_wider_head":        {"central": 0.012, "band": [0.005, 0.020], "citation": "Medusa / EAGLE-3 (directional only)"},
    "on_policy_ttt":            {"central": 0.002, "band": [0.000, 0.005], "citation": "lawine #316 (internal: TTT lifts depth>=2, not ROOT)"},
}
# #336 non-additivity haircut for stacking the two TRAINING levers (soft-KD + reasoning-data): they partly
# target the same mmlu_pro shortfall and the head saturates, so combo = 0.70 * naive-sum. The haircut sets
# the MEAN; it is a first-moment effect, distinct from the lever-uncertainty CORRELATION (a second-moment
# effect) swept below. Both are modeled, and they are not double-counting.
COMBO_NON_ADDITIVITY = 0.70

# The recommended retrain recipe (#336): the two TRAINING levers on the existing {2,21,39} fusion arch.
RECIPE_LEVERS = ["soft_kd_topk_distill", "more_reasoning_root_data"]
# The full optimistic recipe (all four levers): combo + deeper-head + TTT stacked additively (no extra
# haircut on the capacity / test-time levers -> upper bound).
FULL_EXTRA_LEVERS = ["deeper_wider_head", "on_policy_ttt"]

# Correlation cases for the lever uncertainties.
RHO_INDEPENDENT = 0.0
RHO_CONSERVATIVE = 0.5     # the PR's "+0.5 positively-correlated" conservative bound
RHO_COMONOTONIC = 1.0      # reproduces #336's reported combo band (validation anchor / worst case)

# Speed-envelope coverage-bar sweep (stark #stark-cov-envelope-ordering solves the exact c*; until it
# lands we bracket). c* = 0.908 is the headline central anchor reported alongside the identity bar.
CSTAR_SWEEP = [0.88, 0.90, 0.908, 0.92]
CSTAR_HEADLINE = 0.908

COV_MAX = 1.0              # coverage is a fraction in [0, 1]; upper truncation (far in the tail, >6 sigma).

# Decision thresholds (PR step 5), keyed on p_clears_identity_bar_0p9213 (independent):
ROI_JUSTIFIED_AT = 0.50    # coin-flip-or-better -> retrain is justified
ROI_LIKELY_WASTE_BELOW = 0.25

# Banked on-disk artifacts (read-only) for the constant-drift cross-check.
PR336_RESULTS = (REPO_ROOT / "research" / "validity" / "eagle3_head_coverage_lift_target"
                 / "eagle3_head_coverage_lift_target_results.json")
PR330_RESULTS = (REPO_ROOT / "research" / "validity" / "eagle3_sharegpt_coverage_prior"
                 / "eagle3_sharegpt_coverage_prior_results.json")

MC_SAMPLES = 1_000_000
MC_SEED = 20260615

TOL = 1e-9
TOL_REPRO = 1e-6
SQRT2 = math.sqrt(2.0)
Z95 = 1.6448536269514722    # standard-normal 95th percentile (for the +/-2sigma <-> band note only)


# --------------------------------------------------------------------------- #
# Closed-form normal helpers (no scipy / numpy dependency for the PRIMARY path; erfc for tail accuracy).
# --------------------------------------------------------------------------- #
def sigma_from_band(band: list[float]) -> float:
    """Treat [low, high] as a ~95% (+/-2 sigma) interval -> sigma = (high - low) / 4."""
    return (band[1] - band[0]) / 4.0


def normal_sf(x: float, mu: float, sigma: float) -> float:
    """Survival function 1 - Phi((x-mu)/sigma) via erfc (accurate in the tail)."""
    if sigma <= 0.0:
        return 1.0 if x <= mu else 0.0
    return 0.5 * math.erfc((x - mu) / (sigma * SQRT2))


def prob_ge(t: float, mu: float, sigma: float, low: float | None = None,
            high: float | None = None) -> float:
    """P(X >= t) for X ~ Normal(mu, sigma), optionally truncated to [low, high].

    For the truncated normal: P(X >= t | low <= X <= high) = (SF(max(t,low)) - SF(high)) / (SF(low) - SF(high)).
    """
    lo_sf = normal_sf(low, mu, sigma) if low is not None else 1.0
    hi_sf = normal_sf(high, mu, sigma) if high is not None else 0.0
    denom = lo_sf - hi_sf
    if denom <= 0.0:
        return float("nan")
    t_eff = t if low is None else max(t, low)
    num = normal_sf(t_eff, mu, sigma) - hi_sf
    return max(0.0, min(1.0, num / denom))


def trunc_percentile(q: float, mu: float, sigma: float, low: float, high: float) -> float:
    """q-th percentile (q in (0,1)) of Normal(mu,sigma) truncated to [low, high], via bisection on prob_ge.

    Exact (no normal-quantile approximation); deterministic; handles the truncation renormalization.
    """
    target_sf = 1.0 - q                         # we solve prob_ge(x) == 1 - q  ->  x = q-th percentile
    a, b = low, high
    for _ in range(200):
        m = 0.5 * (a + b)
        if prob_ge(m, mu, sigma, low, high) > target_sf:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


# --------------------------------------------------------------------------- #
# Recipe distributions: build the per-lever (mean, sigma), the combo recipe under each correlation case,
# and the full-recipe optimistic case. The 0.70 haircut scales the combo levers (recipe = 0.70*(KD+data)),
# so each combo lever's contribution sigma is 0.70 * sigma_lever; correlation rho only widens the sum.
# --------------------------------------------------------------------------- #
def lever_moments() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, spec in LEVERS.items():
        out[name] = {"mean": float(spec["central"]), "sigma": sigma_from_band(spec["band"]),
                     "band_low": float(spec["band"][0]), "band_high": float(spec["band"][1])}
    return out


def combo_sigma(rho: float, scaled_sigmas: list[float]) -> float:
    """sigma of a sum of components with per-component sigmas `scaled_sigmas` under equicorrelation rho."""
    var = sum(s * s for s in scaled_sigmas)
    for i in range(len(scaled_sigmas)):
        for j in range(i + 1, len(scaled_sigmas)):
            var += 2.0 * rho * scaled_sigmas[i] * scaled_sigmas[j]
    return math.sqrt(max(0.0, var))


def recipe_distributions() -> dict[str, Any]:
    mom = lever_moments()

    # --- The recommended COMBO recipe (soft-KD + reasoning-data), 0.70 haircut on the two levers. ---
    naive_sum_mean = sum(mom[k]["mean"] for k in RECIPE_LEVERS)            # 0.030 + 0.025 = 0.055
    recipe_mean = COMBO_NON_ADDITIVITY * naive_sum_mean                    # 0.70 * 0.055 = 0.0385
    scaled_sigmas = [COMBO_NON_ADDITIVITY * mom[k]["sigma"] for k in RECIPE_LEVERS]  # [0.00525, 0.00525]

    sigma_by_rho = {
        "independent": combo_sigma(RHO_INDEPENDENT, scaled_sigmas),       # sqrt(sum sigma'^2)
        "correlated_0p5": combo_sigma(RHO_CONSERVATIVE, scaled_sigmas),
        "comonotonic": combo_sigma(RHO_COMONOTONIC, scaled_sigmas),       # == #336 combo band sigma
    }

    cov_post_mean = COV_PRIOR + recipe_mean                               # 0.8903 + 0.0385 = 0.9288
    naive_sum_band = [sum(mom[k]["band_low"] for k in RECIPE_LEVERS),
                      sum(mom[k]["band_high"] for k in RECIPE_LEVERS)]     # [0.025, 0.085]
    combo_band_336 = [COMBO_NON_ADDITIVITY * naive_sum_band[0],
                      COMBO_NON_ADDITIVITY * naive_sum_band[1]]            # [0.0175, 0.0595] (#336)

    # --- The FULL optimistic recipe: combo + deeper-head + TTT stacked additively (upper bound). ---
    full_extra_mean = sum(mom[k]["mean"] for k in FULL_EXTRA_LEVERS)       # 0.012 + 0.002 = 0.014
    full_mean = recipe_mean + full_extra_mean                             # 0.0385 + 0.014 = 0.0525
    full_cov_post_mean = COV_PRIOR + full_mean
    # independent full sigma: combine the (independent) combo sigma with the two extra lever sigmas.
    full_components = [sigma_by_rho["independent"]] + [mom[k]["sigma"] for k in FULL_EXTRA_LEVERS]
    full_sigma_indep = combo_sigma(RHO_INDEPENDENT, full_components)
    # full recipe WITHOUT TTT (to confirm TTT barely moves the probability).
    full_no_ttt_mean = recipe_mean + mom["deeper_wider_head"]["mean"]
    full_no_ttt_cov_post_mean = COV_PRIOR + full_no_ttt_mean
    full_no_ttt_sigma = combo_sigma(RHO_INDEPENDENT,
                                    [sigma_by_rho["independent"], mom["deeper_wider_head"]["sigma"]])

    # --- Over-optimistic NAIVE-SUM bound (no haircut; double-counts overlap). Reported, NOT the headline. ---
    naive_cov_post_mean = COV_PRIOR + naive_sum_mean                      # 0.8903 + 0.055 = 0.9453
    naive_sigma_indep = combo_sigma(RHO_INDEPENDENT, [mom[k]["sigma"] for k in RECIPE_LEVERS])

    return {
        "lever_moments": mom,
        "recipe": {
            "levers": RECIPE_LEVERS,
            "non_additivity_haircut": COMBO_NON_ADDITIVITY,
            "naive_sum_mean": naive_sum_mean,
            "naive_sum_band": naive_sum_band,
            "recipe_mean_delta_cov": recipe_mean,
            "scaled_lever_sigmas": scaled_sigmas,
            "sigma_by_rho": sigma_by_rho,
            "cov_post_mean": cov_post_mean,
            "combo_band_336_reconstructed": combo_band_336,
        },
        "full_recipe": {
            "levers": RECIPE_LEVERS + FULL_EXTRA_LEVERS,
            "full_mean_delta_cov": full_mean,
            "cov_post_mean": full_cov_post_mean,
            "sigma_independent": full_sigma_indep,
            "no_ttt_cov_post_mean": full_no_ttt_cov_post_mean,
            "no_ttt_sigma_independent": full_no_ttt_sigma,
        },
        "naive_no_haircut_bound": {
            "cov_post_mean": naive_cov_post_mean,
            "sigma_independent": naive_sigma_indep,
        },
    }


# --------------------------------------------------------------------------- #
# Clear probabilities: the decision numbers.
# --------------------------------------------------------------------------- #
def clear_probabilities(dist: dict[str, Any]) -> dict[str, Any]:
    rc = dist["recipe"]
    mu = rc["cov_post_mean"]
    sig = rc["sigma_by_rho"]

    def p_identity(sigma: float) -> float:
        return prob_ge(IDENTITY_BAR, mu, sigma, low=COV_PRIOR, high=COV_MAX)

    p_id = {case: p_identity(sigma) for case, sigma in sig.items()}

    # Speed-envelope bar sweep, per correlation case.
    cstar = {}
    for case, sigma in sig.items():
        cstar[case] = {f"{c:.3f}": prob_ge(c, mu, sigma, low=COV_PRIOR, high=COV_MAX) for c in CSTAR_SWEEP}
    # which bar binds at each c*: identity (0.9213) vs speed (c*). The harder (higher) threshold binds.
    binds = {f"{c:.3f}": ("identity" if IDENTITY_BAR >= c else "speed_envelope") for c in CSTAR_SWEEP}

    # cov_post percentiles (5/50/95), independent and conservative.
    pct = {}
    for case in ("independent", "correlated_0p5"):
        s = sig[case]
        pct[case] = {
            "p05": trunc_percentile(0.05, mu, s, COV_PRIOR, COV_MAX),
            "p50": trunc_percentile(0.50, mu, s, COV_PRIOR, COV_MAX),
            "p95": trunc_percentile(0.95, mu, s, COV_PRIOR, COV_MAX),
        }

    # Full-recipe optimistic upper bound (P at the identity bar), and the no-TTT comparison.
    fr = dist["full_recipe"]
    p_full_identity = prob_ge(IDENTITY_BAR, fr["cov_post_mean"], fr["sigma_independent"],
                              low=COV_PRIOR, high=COV_MAX)
    p_full_no_ttt_identity = prob_ge(IDENTITY_BAR, fr["no_ttt_cov_post_mean"], fr["no_ttt_sigma_independent"],
                                     low=COV_PRIOR, high=COV_MAX)

    # Over-optimistic naive-sum bound (no haircut).
    nb = dist["naive_no_haircut_bound"]
    p_naive_identity = prob_ge(IDENTITY_BAR, nb["cov_post_mean"], nb["sigma_independent"],
                               low=COV_PRIOR, high=COV_MAX)

    return {
        "p_clears_identity_bar_0p9213": {
            "independent": p_id["independent"],
            "correlated_0p5": p_id["correlated_0p5"],
            "comonotonic_336band": p_id["comonotonic"],
        },
        "p_clears_speed500_bar_by_cstar": cstar,
        "p_clears_speed500_bar_headline_c0p908": {
            "independent": cstar["independent"][f"{CSTAR_HEADLINE:.3f}"],
            "correlated_0p5": cstar["correlated_0p5"][f"{CSTAR_HEADLINE:.3f}"],
        },
        "which_bar_binds_by_cstar": binds,
        "cov_post_percentiles": pct,
        "p_clears_identity_full_recipe": p_full_identity,
        "p_clears_identity_full_recipe_no_ttt": p_full_no_ttt_identity,
        "ttt_probability_delta": p_full_identity - p_full_no_ttt_identity,
        "p_clears_identity_naive_no_haircut_bound": p_naive_identity,
    }


# --------------------------------------------------------------------------- #
# Monte-Carlo cross-check (guarded; numpy-optional). Validates the closed-form and supplies the
# uniform-over-band shape-robustness alternative. The PRIMARY self-test never depends on numpy.
# --------------------------------------------------------------------------- #
def monte_carlo(dist: dict[str, Any], n: int = MC_SAMPLES, seed: int = MC_SEED) -> dict[str, Any] | None:
    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        return None

    mom = dist["lever_moments"]
    kd, da = mom["soft_kd_topk_distill"], mom["more_reasoning_root_data"]
    scale = COMBO_NON_ADDITIVITY
    rng = np.random.default_rng(seed)

    def p_normal(rho: float) -> float:
        z0 = rng.standard_normal(n)
        z1 = rng.standard_normal(n)
        zc = rho * z0 + math.sqrt(max(0.0, 1.0 - rho * rho)) * z1
        kd_s = kd["mean"] + kd["sigma"] * z0
        da_s = da["mean"] + da["sigma"] * zc
        recipe = scale * (kd_s + da_s)
        cov_post = COV_PRIOR + recipe
        mask = (cov_post >= COV_PRIOR) & (cov_post <= COV_MAX)           # Delta-cov >= 0 truncation
        cp = cov_post[mask]
        return round(float(np.mean(cp >= IDENTITY_BAR)), 6)

    def p_uniform(comonotonic: bool) -> float:
        u0 = rng.random(n)
        u1 = u0 if comonotonic else rng.random(n)
        kd_s = kd["band_low"] + u0 * (kd["band_high"] - kd["band_low"])
        da_s = da["band_low"] + u1 * (da["band_high"] - da["band_low"])
        cov_post = COV_PRIOR + scale * (kd_s + da_s)
        return round(float(np.mean(cov_post >= IDENTITY_BAR)), 6)

    return {
        "samples": n,
        "seed": seed,
        "normal_independent": p_normal(RHO_INDEPENDENT),
        "normal_correlated_0p5": p_normal(RHO_CONSERVATIVE),
        "normal_comonotonic": p_normal(RHO_COMONOTONIC),
        "uniform_independent": p_uniform(False),
        "uniform_comonotonic": p_uniform(True),
    }


def uniform_comonotonic_closed_form() -> float:
    """Closed form for the comonotonic uniform case: recipe ~ Uniform[0.0175, 0.0595] (== #336 band)."""
    lo, hi = 0.70 * (0.015 + 0.010), 0.70 * (0.045 + 0.040)              # [0.0175, 0.0595]
    thresh = IDENTITY_BAR - COV_PRIOR                                     # required Delta-cov
    if thresh <= lo:
        return 1.0
    if thresh >= hi:
        return 0.0
    return (hi - thresh) / (hi - lo)


# --------------------------------------------------------------------------- #
# Decision framing.
# --------------------------------------------------------------------------- #
def decision_verdict(p_independent: float) -> dict[str, Any]:
    if p_independent >= ROI_JUSTIFIED_AT:
        verdict = "JUSTIFIED"
    elif p_independent < ROI_LIKELY_WASTE_BELOW:
        verdict = "LIKELY-WASTE"
    else:
        verdict = "MARGINAL"
    return {
        "retrain_roi_verdict": verdict,
        "keyed_on": "p_clears_identity_bar_0p9213 (independent)",
        "threshold_justified_at": ROI_JUSTIFIED_AT,
        "threshold_likely_waste_below": ROI_LIKELY_WASTE_BELOW,
        "cheap_pre_check_de_risker": (
            "kanna #294 (j0ss47bv) Phase-1 a_2 >= 0.83 gate: a cheap acceptance-proxy measurement on the "
            "existing base that de-risks this probability BEFORE the human-gated full retrain spend (#319). "
            "Order: run the #294 Phase-1 cheap gate first; if it passes, the retrain ROI here is realized; "
            "if it misses, hold the spend."),
    }


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    dist = recipe_distributions()
    probs = clear_probabilities(dist)
    mc = monte_carlo(dist)
    p_indep = probs["p_clears_identity_bar_0p9213"]["independent"]
    verdict = decision_verdict(p_indep)

    handoff = (
        f"the EAGLE-3 demand-half retrain clears the 0.9213 identity bar with probability "
        f"{p_indep:.3f} (independent levers) / "
        f"{probs['p_clears_identity_bar_0p9213']['correlated_0p5']:.3f} (+0.5-correlated), and the looser "
        f"speed-envelope bar c*~0.908 with probability "
        f"{probs['p_clears_speed500_bar_headline_c0p908']['independent']:.3f} -- so the retrain ROI is "
        f"{verdict['retrain_roi_verdict']}, and the cheap de-risker is kanna #294's Phase-1 a_2>=0.83 "
        f"pre-check before the human-gated full spend.")

    return {
        "imported": {
            "cov_prior_display": COV_PRIOR_DISPLAY, "identity_bar_display": IDENTITY_BAR_DISPLAY,
            "min_lift_display": MIN_LIFT_DISPLAY,
            "cov_prior_unrounded": COV_PRIOR, "identity_bar_unrounded": IDENTITY_BAR,
            "min_lift_unrounded": MIN_LIFT_REQUIRED,
            "baseline_tps": BASELINE_TPS, "ceiling_tps": CEILING_TPS,
            "levers": {k: {"central": v["central"], "band": v["band"], "citation": v["citation"]}
                       for k, v in LEVERS.items()},
            "combo_non_additivity_haircut": COMBO_NON_ADDITIVITY,
            "provenance": (
                "Per-lever Delta-cov bands + 0.70 non-additivity haircut + combo central 0.0385 band "
                "[0.0175,0.0595]: lawine #336 (krroookz, on-disk eagle3_head_coverage_lift_target_results."
                "json). Honest top-4 ROOT coverage prior 0.8903 + identity bar 0.9213 + gap 0.031035: "
                "lawine #330 (hfrscdai) / #316 (5lnz5jgb) via #323. Speed-envelope anchors central 520.95: "
                "stark #325. Joint AND-gate context: fern #335 (5pos499e). Phase-1 a_2>=0.83 cheap "
                "de-risker + EAGLE-3 E[T]=4.69<4.9029 demand-miss on the speed axis: kanna #294 (j0ss47bv). "
                "This card convolves the bands into a probability; it re-derives none of the upstream "
                "numbers."),
        },
        "step1_per_lever_distributions": {
            "assumption": "band [low,high] ~ +/-2 sigma (95%) -> sigma = (high-low)/4, truncated normal Delta-cov>=0",
            "moments": dist["lever_moments"],
            "robustness_alternative": "uniform-over-band (sigma=range/sqrt(12)); reported in step6 MC",
        },
        "step2_recipe_convolution": dist["recipe"],
        "step2_full_and_naive": {"full_recipe": dist["full_recipe"],
                                 "naive_no_haircut_bound": dist["naive_no_haircut_bound"]},
        "step3_clear_probabilities": probs,
        "step5_decision_framing": verdict,
        "step6_monte_carlo_and_robustness": {
            "monte_carlo": mc,
            "uniform_comonotonic_closed_form": uniform_comonotonic_closed_form(),
            "note": ("MC (numpy, seed fixed) cross-checks the closed-form normal cases and supplies the "
                     "uniform-over-band shape robustness. comonotonic (rho=1) reproduces #336's reported "
                     "combo band; independent (rho=0) is the tighter base case."),
        },
        "step6_greedy_safety": {
            "clear_prob_card_is_cpu_analytic": True,
            "retrain_run_is_human_gated": True,
            "retrain_target_greedy_identical_by_construction": True,
            "ppl_pinned": True,
            "note": ("The retrain TARGET is greedy-IDENTICAL by construction (EAGLE-3 drafter only PROPOSES; "
                     "emission is the verify-model argmax, so accepted tokens are byte-exact greedy) and "
                     "PPL-pinned. Coverage is the SPEED/acceptance axis (E[T]), NOT the validity axis. "
                     "BASELINE 481.53 unchanged; 0 TPS; no served-file change; no HF Job; not a launch."),
        },
        "handoff_sentence": handoff,
        "test_metrics": {
            "p_clears_identity_bar_0p9213": p_indep,
            "p_clears_speed500_bar_c0p908": probs["p_clears_speed500_bar_headline_c0p908"]["independent"],
            "retrain_roi_verdict": verdict["retrain_roi_verdict"],
            "retrain_clear_probability_self_test_passes": False,   # set by main() after the gate runs
        },
    }


# --------------------------------------------------------------------------- #
# Banked-constant drift guard.
# --------------------------------------------------------------------------- #
def load_banked() -> dict[str, Any]:
    out: dict[str, Any] = {}
    if PR336_RESULTS.exists():
        out["pr336"] = json.loads(PR336_RESULTS.read_text())
    if PR330_RESULTS.exists():
        out["pr330"] = json.loads(PR330_RESULTS.read_text())
    return out


def drift_ok(banked: dict[str, Any]) -> bool:
    if not banked:
        return False
    ok = True
    r336 = banked.get("pr336")
    if r336:
        syn = r336.get("synthesis", {})
        imp = syn.get("imported", {})
        t1 = syn.get("step1_lift_target_table", {})
        combo = (syn.get("step3_recipe_ranking", {}) or {}).get("recommended_combination", {})
        if imp.get("T_effective") is not None:
            ok = ok and abs(float(imp["T_effective"]) - IDENTITY_BAR) <= TOL
        if t1.get("aggregate_baseline") is not None:
            ok = ok and abs(float(t1["aggregate_baseline"]) - COV_PRIOR) <= TOL
        if t1.get("min_aggregate_lift_required") is not None:
            ok = ok and abs(float(t1["min_aggregate_lift_required"]) - MIN_LIFT_REQUIRED) <= TOL
        if combo.get("delta_cov_central") is not None:
            ok = ok and abs(float(combo["delta_cov_central"]) - 0.0385) <= TOL
        if combo.get("delta_cov_naive_sum") is not None:
            ok = ok and abs(float(combo["delta_cov_naive_sum"]) - 0.055) <= TOL
        if combo.get("non_additivity_factor") is not None:
            ok = ok and abs(float(combo["non_additivity_factor"]) - COMBO_NON_ADDITIVITY) <= TOL
        if isinstance(combo.get("delta_cov_band"), list) and len(combo["delta_cov_band"]) == 2:
            ok = ok and abs(float(combo["delta_cov_band"][0]) - 0.0175) <= TOL
            ok = ok and abs(float(combo["delta_cov_band"][1]) - 0.0595) <= TOL
        ranking = (syn.get("step3_recipe_ranking", {}) or {}).get("ranking", [])
        on_disk = {e["recipe"]: e for e in ranking if isinstance(e, dict) and "recipe" in e}
        for name, spec in LEVERS.items():
            e = on_disk.get(name)
            if e is not None:
                ok = ok and abs(float(e["delta_cov_central"]) - spec["central"]) <= TOL
                ok = ok and abs(float(e["delta_cov_band"][0]) - spec["band"][0]) <= TOL
                ok = ok and abs(float(e["delta_cov_band"][1]) - spec["band"][1]) <= TOL
    r330 = banked.get("pr330")
    if r330:
        prior = (((r330.get("synthesis", {}) or {}).get("step2_composition_to_coverage", {}) or {})
                 .get("point_estimate_uncond_top4"))
        if prior is not None:
            ok = ok and abs(float(prior) - COV_PRIOR) <= TOL_REPRO
    return bool(ok and r336)


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def self_test(syn: dict[str, Any], banked: dict[str, Any]) -> dict[str, Any]:
    rc = syn["step2_recipe_convolution"]
    fr = syn["step2_full_and_naive"]["full_recipe"]
    probs = syn["step3_clear_probabilities"]
    pid = probs["p_clears_identity_bar_0p9213"]
    mc = syn["step6_monte_carlo_and_robustness"]["monte_carlo"]
    c: dict[str, bool] = {}

    mom = rc["scaled_lever_sigmas"]
    sig = rc["sigma_by_rho"]

    # (a) convolution preserves the mean: cov_post_mean = cov_prior + recipe_mean (= 0.70*(0.030+0.025)).
    expect_recipe_mean = COMBO_NON_ADDITIVITY * (LEVERS["soft_kd_topk_distill"]["central"]
                                                 + LEVERS["more_reasoning_root_data"]["central"])
    c["01_recipe_mean_eq_haircut_sum_0p0385"] = bool(
        abs(rc["recipe_mean_delta_cov"] - expect_recipe_mean) <= TOL
        and abs(rc["recipe_mean_delta_cov"] - 0.0385) <= TOL)
    c["02_cov_post_mean_eq_prior_plus_recipe"] = bool(
        abs(rc["cov_post_mean"] - (COV_PRIOR + rc["recipe_mean_delta_cov"])) <= TOL
        and abs(rc["cov_post_mean"] - 0.9287659519153152) <= 1e-9)

    # (b) independent-case sigma = sqrt(sum of scaled lever variances).
    c["03_independent_sigma_eq_rss"] = bool(
        abs(sig["independent"] - math.sqrt(sum(s * s for s in mom))) <= TOL)

    # (c) correlation widens: comonotonic >= correlated_0p5 >= independent (strict).
    c["04_correlation_widens_sigma"] = bool(
        sig["comonotonic"] > sig["correlated_0p5"] > sig["independent"] > 0.0)
    # comonotonic (rho=1) sigma reproduces #336's reported combo band sigma EXACTLY (validation anchor).
    c["05_comonotonic_eq_336_band_sigma"] = bool(
        abs(sig["comonotonic"] - sigma_from_band([0.0175, 0.0595])) <= TOL
        and abs(sig["comonotonic"] - 0.0105) <= TOL)
    # the reconstructed combo band == #336's [0.0175, 0.0595] (combo = 0.70 * naive-sum band endpoints).
    c["06_combo_band_reconstructs_336"] = bool(
        abs(rc["combo_band_336_reconstructed"][0] - 0.0175) <= TOL
        and abs(rc["combo_band_336_reconstructed"][1] - 0.0595) <= TOL
        and abs(rc["naive_sum_mean"] - 0.055) <= TOL)

    # (d) p_clears monotone DECREASING in the bar threshold (independent case over the full sweep + bar).
    sweep_pts = sorted(set(CSTAR_SWEEP + [IDENTITY_BAR]))
    ps = [prob_ge(t, rc["cov_post_mean"], sig["independent"], low=COV_PRIOR, high=COV_MAX)
          for t in sweep_pts]
    c["07_p_monotone_decreasing_in_threshold"] = bool(
        all(ps[i] >= ps[i + 1] - 1e-12 for i in range(len(ps) - 1)))

    # (e) central recipe clears with P >= 0.5 in every correlation case (cov_post mean 0.9288 > bar 0.9213).
    c["08_central_clears_p_ge_half_all_rho"] = bool(
        rc["cov_post_mean"] > IDENTITY_BAR
        and pid["independent"] >= 0.5 and pid["correlated_0p5"] >= 0.5
        and pid["comonotonic_336band"] >= 0.5)
    # ordering across correlation: independent >= correlated_0p5 >= comonotonic (wider spread -> lower P).
    c["09_p_identity_orders_by_rho"] = bool(
        pid["independent"] >= pid["correlated_0p5"] >= pid["comonotonic_336band"])

    # (f) full-recipe (4 levers) P >= combo P (more levers, more mean), and TTT barely moves it.
    c["10_full_recipe_ge_combo"] = bool(
        probs["p_clears_identity_full_recipe"] >= pid["independent"])
    c["11_ttt_barely_moves_probability"] = bool(
        abs(probs["ttt_probability_delta"]) <= 0.02
        and abs(LEVERS["on_policy_ttt"]["central"] - 0.002) <= TOL)

    # (g) imported constants EXACT (rounded display + unrounded banked, mutually consistent).
    c["12_constants_imported_exact"] = bool(
        COV_PRIOR_DISPLAY == 0.8903 and IDENTITY_BAR_DISPLAY == 0.9213 and MIN_LIFT_DISPLAY == 0.031035
        and abs(MIN_LIFT_REQUIRED - (IDENTITY_BAR - COV_PRIOR)) <= TOL
        and abs(round(COV_PRIOR, 4) - COV_PRIOR_DISPLAY) <= TOL
        and abs(round(IDENTITY_BAR, 4) - IDENTITY_BAR_DISPLAY) <= TOL
        and abs(round(MIN_LIFT_REQUIRED, 6) - MIN_LIFT_DISPLAY) <= TOL)

    # (h) NaN-clean over the c* sweep (every probability finite and in [0,1]).
    flat = [pid["independent"], pid["correlated_0p5"], pid["comonotonic_336band"],
            probs["p_clears_identity_full_recipe"], probs["p_clears_identity_naive_no_haircut_bound"]]
    for case in probs["p_clears_speed500_bar_by_cstar"].values():
        flat.extend(case.values())
    c["13_nan_clean_probabilities_in_unit"] = bool(
        all(math.isfinite(p) and -1e-12 <= p <= 1.0 + 1e-12 for p in flat))

    # (i) cov_post percentiles ordered 5 <= 50 <= 95 (independent and conservative).
    pct = probs["cov_post_percentiles"]
    c["14_percentiles_ordered"] = bool(
        pct["independent"]["p05"] <= pct["independent"]["p50"] <= pct["independent"]["p95"]
        and pct["correlated_0p5"]["p05"] <= pct["correlated_0p5"]["p50"] <= pct["correlated_0p5"]["p95"]
        and abs(pct["independent"]["p50"] - rc["cov_post_mean"]) <= 1e-4)

    # which-bar-binds: identity bar (0.9213) is the higher threshold than every c* in the sweep (max 0.92).
    c["15_identity_binds_over_full_cstar_sweep"] = bool(
        all(v == "identity" for v in probs["which_bar_binds_by_cstar"].values())
        and max(CSTAR_SWEEP) < IDENTITY_BAR)

    # MC cross-check (only when numpy present): MC agrees with closed-form within tolerance; comonotonic
    # uniform reproduces the closed form.
    if mc is not None:
        c["16_mc_matches_closed_form"] = bool(
            abs(mc["normal_independent"] - pid["independent"]) <= 5e-3
            and abs(mc["normal_correlated_0p5"] - pid["correlated_0p5"]) <= 5e-3
            and abs(mc["normal_comonotonic"] - pid["comonotonic_336band"]) <= 5e-3
            and abs(mc["uniform_comonotonic"]
                    - syn["step6_monte_carlo_and_robustness"]["uniform_comonotonic_closed_form"]) <= 5e-3)
    else:
        c["16_mc_matches_closed_form"] = True   # numpy absent -> MC skipped; analytic gates still hold.

    # greedy-safety + scope flags.
    gs = syn["step6_greedy_safety"]
    c["17_greedy_safe_and_human_gated_flags"] = bool(
        gs["clear_prob_card_is_cpu_analytic"] and gs["retrain_run_is_human_gated"]
        and gs["retrain_target_greedy_identical_by_construction"] and gs["ppl_pinned"])

    # (j) banked-constant drift guard: imported bands/bar/prior match the on-disk #336 + #330 artifacts.
    c["18_constants_match_banked_336_330"] = bool(drift_ok(banked))

    gate = all(bool(v) for v in c.values())
    return {"retrain_clear_probability_self_test_passes": gate, "checks": c}


# --------------------------------------------------------------------------- #
# NaN-clean walk.
# --------------------------------------------------------------------------- #
def assert_nan_clean(payload: dict) -> list[str]:
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

    walk(payload, "result")
    return bad


# --------------------------------------------------------------------------- #
# Determinism: two independent synthesize() passes must serialize identically (MC is seed-fixed).
# --------------------------------------------------------------------------- #
def determinism_ok() -> bool:
    a = synthesize()
    b = synthesize()
    a["test_metrics"]["retrain_clear_probability_self_test_passes"] = True
    b["test_metrics"]["retrain_clear_probability_self_test_passes"] = True
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# --------------------------------------------------------------------------- #
# W&B logging (summary/ namespace; robust; never fatal).
# --------------------------------------------------------------------------- #
def maybe_log_wandb(args, payload: dict) -> str | None:
    if getattr(args, "no_wandb", False) or not getattr(args, "wandb_name", None):
        return None
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_json_artifact, log_summary
    except Exception as exc:  # noqa: BLE001
        print(f"[eagle3-retrain-clear-probability] wandb logging unavailable: {exc}", flush=True)
        return None

    syn = payload["synthesis"]
    rc = syn["step2_recipe_convolution"]
    probs = syn["step3_clear_probabilities"]
    pid = probs["p_clears_identity_bar_0p9213"]
    fr = syn["step2_full_and_naive"]["full_recipe"]
    verdict = syn["step5_decision_framing"]
    st = payload["self_test"]
    mc = syn["step6_monte_carlo_and_robustness"]["monte_carlo"]

    run = init_wandb_run(
        job_type="validity-analytic", agent="lawine", name=args.wandb_name, group=args.wandb_group,
        tags=["eagle3-retrain-clear-probability", "validity-analytic", "clear-probability", "eagle3",
              "demand-half", "retrain-roi", "convolution", "bank-the-analysis"],
        config={
            "pr": 339, "identity_bar": IDENTITY_BAR, "cov_prior": COV_PRIOR,
            "min_lift_required": MIN_LIFT_REQUIRED, "combo_non_additivity_haircut": COMBO_NON_ADDITIVITY,
            "recipe_levers": RECIPE_LEVERS, "cstar_sweep": CSTAR_SWEEP, "cstar_headline": CSTAR_HEADLINE,
            "baseline_tps": BASELINE_TPS, "ceiling_tps": CEILING_TPS, "mc_samples": MC_SAMPLES,
            "wandb_group": args.wandb_group, "imports": syn["imported"]["provenance"],
        },
    )
    if run is None:
        print("[eagle3-retrain-clear-probability] wandb: no run (no WANDB_API_KEY/mode) -- skipping",
              flush=True)
        return None

    cstar_indep = probs["p_clears_speed500_bar_by_cstar"]["independent"]
    summary: dict[str, Any] = {
        # PRIMARY + TEST + REPORT
        "retrain_clear_probability_self_test_passes": int(bool(
            st["retrain_clear_probability_self_test_passes"])),
        "p_clears_identity_bar_0p9213": pid["independent"],
        "p_clears_identity_bar_0p9213_correlated_0p5": pid["correlated_0p5"],
        "p_clears_identity_bar_0p9213_comonotonic_336band": pid["comonotonic_336band"],
        "p_clears_speed500_bar_c0p908": probs["p_clears_speed500_bar_headline_c0p908"]["independent"],
        "p_clears_speed500_bar_c0p908_correlated_0p5":
            probs["p_clears_speed500_bar_headline_c0p908"]["correlated_0p5"],
        "retrain_roi_verdict_justified": int(verdict["retrain_roi_verdict"] == "JUSTIFIED"),
        # recipe distribution
        "recipe_mean_delta_cov": rc["recipe_mean_delta_cov"],
        "cov_post_mean": rc["cov_post_mean"],
        "sigma_independent": rc["sigma_by_rho"]["independent"],
        "sigma_correlated_0p5": rc["sigma_by_rho"]["correlated_0p5"],
        "sigma_comonotonic_336band": rc["sigma_by_rho"]["comonotonic"],
        # c* sweep (independent)
        "p_clears_c0p880": cstar_indep["0.880"], "p_clears_c0p900": cstar_indep["0.900"],
        "p_clears_c0p908": cstar_indep["0.908"], "p_clears_c0p920": cstar_indep["0.920"],
        # full recipe + bounds
        "p_clears_identity_full_recipe": probs["p_clears_identity_full_recipe"],
        "ttt_probability_delta": probs["ttt_probability_delta"],
        "p_clears_identity_naive_no_haircut_bound": probs["p_clears_identity_naive_no_haircut_bound"],
        # percentiles
        "cov_post_p05_independent": probs["cov_post_percentiles"]["independent"]["p05"],
        "cov_post_p95_independent": probs["cov_post_percentiles"]["independent"]["p95"],
        # context
        "identity_bar": IDENTITY_BAR, "cov_prior": COV_PRIOR, "min_lift_required": MIN_LIFT_REQUIRED,
        "baseline_tps": BASELINE_TPS,
        # hygiene
        "nan_clean": int(bool(payload["nan_clean"])),
        "determinism_ok": int(bool(payload["determinism_ok"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(val)) for k, val in st["checks"].items()},
    }
    if mc is not None:
        summary.update({
            "mc_normal_independent": mc["normal_independent"],
            "mc_normal_correlated_0p5": mc["normal_correlated_0p5"],
            "mc_uniform_independent": mc["uniform_independent"],
            "mc_uniform_comonotonic": mc["uniform_comonotonic"],
        })
    summary = {k: val for k, val in summary.items()
               if not (isinstance(val, float) and not math.isfinite(val)) and val is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="eagle3_retrain_clear_probability_result", artifact_type="validity",
                      data=payload)
    rid = getattr(run, "id", None)
    print(f"[eagle3-retrain-clear-probability] wandb run: {getattr(run, 'url', rid)}", flush=True)
    finish_wandb(run)
    return rid


# --------------------------------------------------------------------------- #
# Console report.
# --------------------------------------------------------------------------- #
def print_report(payload: dict) -> None:
    syn = payload["synthesis"]
    rc = syn["step2_recipe_convolution"]
    probs = syn["step3_clear_probabilities"]
    pid = probs["p_clears_identity_bar_0p9213"]
    fr = syn["step2_full_and_naive"]["full_recipe"]
    verdict = syn["step5_decision_framing"]
    st = payload["self_test"]
    mc = syn["step6_monte_carlo_and_robustness"]["monte_carlo"]
    print("\n" + "=" * 100, flush=True)
    print("EAGLE-3 DEMAND-HALF RETRAIN CLEAR-PROBABILITY (PR #339) -- P(clears 0.9213 identity bar)?",
          flush=True)
    print("=" * 100, flush=True)
    print(f"cov_prior {COV_PRIOR:.7f}  identity_bar {IDENTITY_BAR:.7f}  min_lift {MIN_LIFT_REQUIRED:.7f}",
          flush=True)
    print("-" * 100, flush=True)
    print("STEP 1/2 -- recipe (soft-KD + reasoning-data), 0.70 haircut -> Delta-cov ~ N(mean, sigma):",
          flush=True)
    print(f"  recipe_mean_delta_cov = {rc['recipe_mean_delta_cov']:.4f}  "
          f"cov_post_mean = {rc['cov_post_mean']:.4f}  (= prior + 0.0385)", flush=True)
    print(f"  sigma  independent={rc['sigma_by_rho']['independent']:.5f}  "
          f"rho0.5={rc['sigma_by_rho']['correlated_0p5']:.5f}  "
          f"comonotonic(=336 band)={rc['sigma_by_rho']['comonotonic']:.5f}", flush=True)
    print("-" * 100, flush=True)
    print("STEP 3 -- CLEAR PROBABILITIES (decision numbers):", flush=True)
    print(f"  P(clears identity 0.9213):  independent={pid['independent']:.4f}  "
          f"rho0.5={pid['correlated_0p5']:.4f}  comonotonic={pid['comonotonic_336band']:.4f}", flush=True)
    print("  P(clears speed-envelope c*) [independent]:", flush=True)
    for c, p in probs["p_clears_speed500_bar_by_cstar"]["independent"].items():
        binds = probs["which_bar_binds_by_cstar"][c]
        print(f"     c*={c}: P={p:.4f}   (binds: {binds})", flush=True)
    print(f"  full-recipe (4 levers) P(identity) = {probs['p_clears_identity_full_recipe']:.4f}  "
          f"(TTT delta {probs['ttt_probability_delta']:+.4f})", flush=True)
    print(f"  naive-no-haircut bound P(identity) = "
          f"{probs['p_clears_identity_naive_no_haircut_bound']:.4f}  (over-optimistic; double-counts overlap)",
          flush=True)
    pct = probs["cov_post_percentiles"]["independent"]
    print(f"  cov_post percentiles [independent]: p05={pct['p05']:.4f} p50={pct['p50']:.4f} "
          f"p95={pct['p95']:.4f}", flush=True)
    if mc is not None:
        print(f"  MC cross-check: indep={mc['normal_independent']:.4f} rho0.5={mc['normal_correlated_0p5']:.4f} "
              f"comon={mc['normal_comonotonic']:.4f} | unif_indep={mc['uniform_independent']:.4f} "
              f"unif_comon={mc['uniform_comonotonic']:.4f}", flush=True)
    print("-" * 100, flush=True)
    print(f"STEP 5 -- RETRAIN ROI VERDICT: {verdict['retrain_roi_verdict']}  "
          f"(p_indep={pid['independent']:.4f} vs justified>={ROI_JUSTIFIED_AT})", flush=True)
    print("-" * 100, flush=True)
    print(f"HAND-OFF: {syn['handoff_sentence']}", flush=True)
    print("-" * 100, flush=True)
    print(f"(PRIMARY) retrain_clear_probability_self_test_passes = "
          f"{st['retrain_clear_probability_self_test_passes']}", flush=True)
    for k, val in st["checks"].items():
        print(f"   - {k}: {val}", flush=True)
    print(f"nan_clean = {payload['nan_clean']}   determinism_ok = {payload['determinism_ok']}   "
          f"peak_mem_mib = {payload['peak_mem_mib']}", flush=True)
    print("=" * 100 + "\n", flush=True)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="eagle3-retrain-clear-probability")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    banked = load_banked()
    syn = synthesize()
    st = self_test(syn, banked)
    syn["test_metrics"]["retrain_clear_probability_self_test_passes"] = bool(
        st["retrain_clear_probability_self_test_passes"])

    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "pr": 339, "agent": "lawine", "kind": "eagle3-retrain-clear-probability",
        "eagle3_retrain_clear_probability_analysis_only": True,
        "synthesis": syn, "self_test": st,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }

    nan_paths = assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[eagle3-retrain-clear-probability] WARNING non-finite at: {nan_paths}", flush=True)
    payload["determinism_ok"] = determinism_ok()

    gate = bool(st["retrain_clear_probability_self_test_passes"] and payload["nan_clean"]
                and payload["determinism_ok"])
    st["retrain_clear_probability_self_test_passes"] = gate
    syn["test_metrics"]["retrain_clear_probability_self_test_passes"] = gate

    print_report(payload)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eagle3_retrain_clear_probability_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[eagle3-retrain-clear-probability] wrote {out_path}", flush=True)

    rid = maybe_log_wandb(args, payload)
    payload["wandb_run_id"] = rid
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    tm = syn["test_metrics"]
    print(f"  PRIMARY retrain_clear_probability_self_test_passes = {gate}", flush=True)
    print(f"  TEST p_clears_identity_bar_0p9213 = {tm['p_clears_identity_bar_0p9213']:.4f}", flush=True)
    print(f"  REPORT p_clears_speed500_bar_c0p908 = {tm['p_clears_speed500_bar_c0p908']:.4f}", flush=True)
    print(f"  REPORT retrain_roi_verdict = {tm['retrain_roi_verdict']}", flush=True)
    print(f"  wandb run = {rid}", flush=True)

    if args.self_test:
        print(f"[eagle3-retrain-clear-probability] self-test {'PASS' if gate else 'FAIL'}", flush=True)
        return 0 if gate else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
