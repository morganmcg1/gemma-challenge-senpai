#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #346 (student lawine) -- PPL-only retrain achievability: P(a feasible retrain hits the LOWERED PPL-only coverage target).

WHAT THIS CARD DOES (0-GPU, 0-TPS, no training, no model forward, no served-file change, no HF Job)
---------------------------------------------------------------------------------------------------
wirbel #343 (kklof4wr, MERGED, research/validity/ppl_only_gate_500_envelope/) SIZED the PPL-only >500
coverage TARGET: under the human's #124 accepted-risk PPL-only world (drop the strict greedy-identity bar,
keep PPL <= 2.42), the >500 lane needs coverage c*_central = 0.9089 (a +0.0186 lift from the measured
honest prior 0.8903) for central-500, and c*_worst = 0.9256 (+0.0353) for private-stable worst-500. The
central target sits INSIDE lawine #336's +0.031 reachable lift budget; the worst target sits ~14% PAST it.

But "the target is within budget" is NOT "a retrain will hit it." This card converts wirbel #343's
*feasible target* into an *expected outcome*: treating each #336 recipe lever's Delta-cov band as a
distribution and convolving the recommended soft-KD + reasoning-data combination, it reports the
decision-grade achievability numbers the human's #319 retrain decision needs:

  * p_retrain_clears_ppl_only_central_500 = P(combo lift >= +0.01863633)   (central-500; TEST)
  * p_retrain_clears_ppl_only_worst_500   = P(combo lift >= +0.03530365)   (worst-500;   TEST)

under (a) INDEPENDENT levers (base case) and (b) +0.5 POSITIVELY-CORRELATED levers (conservative bound,
since soft-KD and reasoning-data both target the same mmlu_pro reasoning-CoT drag) -- the same correlation
treatment lawine #339 (0aq16szh) used against the strict 0.9213 identity bar. This card RE-POINTS that
machinery from the strict identity bar (lift +0.031) to the LOWER PPL-only targets (lift +0.0186 / +0.0353).

DISTRIBUTION SHAPE. The PR's natural default is a TRIANGULAR over [band_lo, central, band_hi] per lever
(headline); UNIFORM over [band_lo, band_hi] is reported as a shape sensitivity; the NORMAL +/-2 sigma model
(lawine #339's) is carried as an explicit BRIDGE so the re-pointed numbers are directly comparable to #339.
For the two TRAINING levers both bands are symmetric (central == midpoint), so:
  * the INDEPENDENT triangular combo is EXACT in closed form -- the sum of two symmetric triangulars of
    equal half-width is an Irwin-Hall(4): L = 0.0175 + 0.0105 * IH4  (mean 0.0385, support [0.0175,0.0595]);
  * the INDEPENDENT uniform combo is EXACT -- two equal-width uniforms sum to a symmetric triangular, so
    L ~ Triangular[0.0175, 0.0385, 0.0595];
  * the NORMAL combo is exact under any rho via sigma = haircut * sqrt(sx^2 + sy^2 + 2 rho sx sy).
The +0.5-correlated triangular/uniform cells (no elementary closed form) come from a deterministic
Gaussian-copula Monte-Carlo (seed-fixed), cross-checked against the closed forms where they exist.

KEY READING. The central target +0.0186 sits only +0.0011 above the combo's HARD lower support bound
0.0175 and far below the combo central +0.0385, so it is a NEAR-CERTAIN clear under every shape
(P ~ 0.996-1.000): central-500 retrain is a LIKELY-WIN. The worst target +0.0353 sits just below the combo
central +0.0385 (and just past #336's +0.031 budget), so its achievability is ~0.64-0.69 independent /
~0.62-0.64 at rho=+0.5: coin-flip-OR-BETTER but NOT a slam-dunk -- a LEAN-WIN.

LOCAL CPU-only analytic card. No GPU / vLLM / model forward / training / checkpoint / HF Job / submission /
served-file change / publish. NOT a launch. BASELINE stays 481.53 (0 TPS). Greedy/PPL untouched (the
retrain TARGET is greedy-IDENTICAL by construction: EAGLE-3 emission = verify argmax, PPL-pinned --
coverage is the SPEED/acceptance axis, not the validity axis; and in the PPL-only world the strict
identity bar is intentionally dropped per #124, with PPL still pinned <= 2.42). The retrain itself is
human-approval-gated GPU spend (#319 / instructions/training-request.md); THIS card only PRICES it.

PRIMARY metric  retrain_achievability_self_test_passes
TEST    metric  p_retrain_clears_ppl_only_central_500   (float; triangular, INDEPENDENT levers; headline)
TEST    metric  p_retrain_clears_ppl_only_worst_500     (float; triangular, INDEPENDENT levers; headline)
REPORT          central_500_verdict / worst_500_verdict (LIKELY-WIN | LEAN-WIN | UNLIKELY)

Reproduce:
    cd target/ && .venv/bin/python \\
        research/validity/ppl_only_retrain_achievability/ppl_only_retrain_achievability.py \\
        --self-test --wandb_group ppl-only-retrain-achievability \\
        --wandb_name lawine/ppl-only-retrain-achievability
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
# runtime against the on-disk banked lawine #336 (krroookz), lawine #330 (hfrscdai) and wirbel #343
# (kklof4wr) artifacts so there is no silent drift.
# --------------------------------------------------------------------------- #
# Honest top-4 ROOT coverage prior -- lawine #330 (hfrscdai): the official 128 eval is 100% reasoning/STEM.
COV_PRIOR = 0.8902659519153152
COV_PRIOR_DISPLAY = 0.8903                 # the rounded "measured" prior wirbel #343 used to price the lift

# wirbel #343 (kklof4wr) PPL-only >500 coverage targets (UNROUNDED, on-disk), and their lift from 0.8903:
CSTAR_CENTRAL = 0.9089363308345582         # PPL-only central-500 coverage target
CSTAR_WORST = 0.925603648491971            # PPL-only worst-500 (private-stable) coverage target
TARGET_CENTRAL = 0.01863633083455818       # == CSTAR_CENTRAL - 0.8903  (rounds to +0.0186)
TARGET_WORST = 0.035303648491970985        # == CSTAR_WORST   - 0.8903  (rounds to +0.0353)
TARGET_CENTRAL_DISPLAY = 0.0186
TARGET_WORST_DISPLAY = 0.0353

# lawine #336 (krroookz) reachable retrain lift budget (the headline "+0.031 reachable" figure).
RETRAIN_LIFT_BUDGET_336 = 0.031

# lawine #316/#323 strict greedy-identity build bar (the bar the PPL-only world INTENTIONALLY drops; kept
# only as the #339 continuity anchor -- this card no longer targets it).
IDENTITY_BAR = 0.9213011665456927
IDENTITY_LIFT = IDENTITY_BAR - COV_PRIOR   # 0.031035... (the #339 threshold; used in the continuity check)

BASELINE_TPS = 481.53          # current best summary.json:tps (unchanged; 0-TPS analytic)
CEILING_TPS = 520.95           # stark #325/#340 lambda=1 ceiling (central envelope anchor)

# --------------------------------------------------------------------------- #
# Per-lever Delta-cov bands -- IMPORTED EXACT from lawine #336 (krroookz). #336's literature-grounded
# EXPECTED-VALUE PRIORS, not measurements. The recommended COMBO recipe = soft-KD + reasoning-data on the
# existing {2,21,39} fusion arch (no capacity change -> same VRAM/deploy path), combo = 0.70 * naive-sum.
# --------------------------------------------------------------------------- #
LEVERS: dict[str, dict[str, Any]] = {
    "soft_kd_topk_distill":     {"central": 0.030, "band": [0.015, 0.045], "citation": "DistillSpec arXiv:2310.08461 / OSD arXiv:2310.07177 / Medusa arXiv:2401.10774"},
    "more_reasoning_root_data": {"central": 0.025, "band": [0.010, 0.040], "citation": "EAGLE-3 arXiv:2503.01840"},
    "deeper_wider_head":        {"central": 0.012, "band": [0.005, 0.020], "citation": "Medusa / EAGLE-3 (directional only)"},
    "on_policy_ttt":            {"central": 0.002, "band": [0.000, 0.005], "citation": "lawine #316 (internal: TTT lifts depth>=2, not ROOT)"},
}
COMBO_NON_ADDITIVITY = 0.70
RECIPE_LEVERS = ["soft_kd_topk_distill", "more_reasoning_root_data"]   # recommended 2-lever combo
FULL_EXTRA_LEVERS = ["deeper_wider_head", "on_policy_ttt"]             # the optimistic +2-lever upper bound

# Correlation cases for the lever uncertainties.
RHO_INDEPENDENT = 0.0
RHO_CONSERVATIVE = 0.5         # the PR's "+0.5 positively-correlated" conservative bound (per #339)
RHO_COMONOTONIC = 1.0         # reproduces #336's reported combo band (validation / worst-case anchor)

# Distribution shapes for the per-lever bands.
SHAPES = ["triangular", "uniform", "normal"]   # triangular = PR default/headline; uniform = sensitivity;
HEADLINE_SHAPE = "triangular"                  # normal = #339 bridge (NOT the headline)

# Decision thresholds on the achievability probability (independent base case):
LIKELY_WIN_AT = 0.90          # P >= 0.90 -> likely-win
LEAN_WIN_AT = 0.50            # 0.50 <= P < 0.90 -> lean-win (coin-flip-or-better)

# Banked on-disk artifacts (read-only) for the constant-drift cross-check.
PR336_RESULTS = (REPO_ROOT / "research" / "validity" / "eagle3_head_coverage_lift_target"
                 / "eagle3_head_coverage_lift_target_results.json")
PR330_RESULTS = (REPO_ROOT / "research" / "validity" / "eagle3_sharegpt_coverage_prior"
                 / "eagle3_sharegpt_coverage_prior_results.json")
PR343_RESULTS = (REPO_ROOT / "research" / "validity" / "ppl_only_gate_500_envelope"
                 / "ppl_only_gate_500_envelope_results.json")

MC_SAMPLES = 4_000_000
MC_SEED = 20260615

TOL = 1e-9
TOL_REPRO = 1e-6
SQRT2 = math.sqrt(2.0)


# --------------------------------------------------------------------------- #
# Closed-form helpers (pure Python; no numpy needed for the PRIMARY path).
# --------------------------------------------------------------------------- #
def sigma_from_band_normal(band: list[float]) -> float:
    """NORMAL model: treat [low, high] as a ~95% (+/-2 sigma) interval -> sigma = (high - low) / 4."""
    return (band[1] - band[0]) / 4.0


def sigma_lever(shape: str, band: list[float], central: float) -> float:
    """Per-lever standard deviation for the given band shape."""
    lo, hi = band
    if shape == "normal":
        return (hi - lo) / 4.0
    if shape == "uniform":
        return (hi - lo) / math.sqrt(12.0)
    if shape == "triangular":
        # general triangular variance = (a^2+b^2+c^2 - ab - ac - bc)/18 ; symmetric -> (hi-lo)^2/24.
        a, b, c = lo, hi, central
        var = (a * a + b * b + c * c - a * b - a * c - b * c) / 18.0
        return math.sqrt(max(0.0, var))
    raise ValueError(shape)


def combo_sigma(rho: float, scaled_sigmas: list[float]) -> float:
    """sigma of a sum of components with per-component sigmas under equicorrelation rho."""
    var = sum(s * s for s in scaled_sigmas)
    for i in range(len(scaled_sigmas)):
        for j in range(i + 1, len(scaled_sigmas)):
            var += 2.0 * rho * scaled_sigmas[i] * scaled_sigmas[j]
    return math.sqrt(max(0.0, var))


def normal_sf(x: float, mu: float, sigma: float) -> float:
    """Survival function 1 - Phi((x-mu)/sigma) via erfc (accurate in the tail)."""
    if sigma <= 0.0:
        return 1.0 if x <= mu else 0.0
    return 0.5 * math.erfc((x - mu) / (sigma * SQRT2))


def normal_prob_ge(t: float, mu: float, sigma: float, low: float | None = None,
                   high: float | None = None) -> float:
    """P(X >= t), X ~ Normal(mu, sigma), optionally truncated to [low, high] (#339's prob_ge)."""
    lo_sf = normal_sf(low, mu, sigma) if low is not None else 1.0
    hi_sf = normal_sf(high, mu, sigma) if high is not None else 0.0
    denom = lo_sf - hi_sf
    if denom <= 0.0:
        return float("nan")
    t_eff = t if low is None else max(t, low)
    num = normal_sf(t_eff, mu, sigma) - hi_sf
    return max(0.0, min(1.0, num / denom))


def ih4_cdf(z: float) -> float:
    """Irwin-Hall(4) CDF: CDF of the sum of four iid Uniform[0,1], z in [0,4]."""
    if z <= 0.0:
        return 0.0
    if z >= 4.0:
        return 1.0
    k = int(math.floor(z))
    s = 0.0
    for j in range(k + 1):
        s += ((-1) ** j) * math.comb(4, j) * (z - j) ** 4
    return s / 24.0


def triangular_sf(t: float, a: float, c: float, b: float) -> float:
    """P(T >= t) for T ~ Triangular(low=a, mode=c, high=b)."""
    if t <= a:
        return 1.0
    if t >= b:
        return 0.0
    if t <= c:
        return 1.0 - (t - a) ** 2 / ((b - a) * (c - a))
    return (b - t) ** 2 / ((b - a) * (b - c))


# --------------------------------------------------------------------------- #
# Combo distribution per shape: mean (shape-independent 0.0385), support, per-rho sigma, and the EXACT
# closed-form survival for the cells that have one (triangular/uniform INDEPENDENT; normal any rho).
# --------------------------------------------------------------------------- #
def combo_geometry() -> dict[str, Any]:
    h = COMBO_NON_ADDITIVITY
    kd, da = LEVERS["soft_kd_topk_distill"], LEVERS["more_reasoning_root_data"]
    naive_sum_mean = kd["central"] + da["central"]                       # 0.055
    combo_mean = h * naive_sum_mean                                      # 0.0385
    support_lo = h * (kd["band"][0] + da["band"][0])                     # 0.0175
    support_hi = h * (kd["band"][1] + da["band"][1])                     # 0.0595
    combo_band_336 = [support_lo, support_hi]                           # == #336 reported combo band
    per_shape: dict[str, Any] = {}
    for shape in SHAPES:
        s_kd = sigma_lever(shape, kd["band"], kd["central"])
        s_da = sigma_lever(shape, da["band"], da["central"])
        scaled = [h * s_kd, h * s_da]
        per_shape[shape] = {
            "lever_sigma_kd": s_kd, "lever_sigma_data": s_da,
            "scaled_lever_sigmas": scaled,
            "sigma_independent": combo_sigma(RHO_INDEPENDENT, scaled),
            "sigma_correlated_0p5": combo_sigma(RHO_CONSERVATIVE, scaled),
            "sigma_comonotonic": combo_sigma(RHO_COMONOTONIC, scaled),
        }
    return {
        "haircut": h, "naive_sum_mean": naive_sum_mean, "combo_mean": combo_mean,
        "support": [support_lo, support_hi], "combo_band_336_reconstructed": combo_band_336,
        "per_shape": per_shape,
    }


def closed_form_prob_ge(shape: str, t: float, rho: float, geo: dict[str, Any]) -> float | None:
    """EXACT closed-form P(combo lift >= t) where one exists; else None (-> Monte-Carlo)."""
    mean = geo["combo_mean"]
    a, b = geo["support"]
    if shape == "normal":
        sig = {RHO_INDEPENDENT: "sigma_independent", RHO_CONSERVATIVE: "sigma_correlated_0p5",
               RHO_COMONOTONIC: "sigma_comonotonic"}[rho]
        sigma = geo["per_shape"]["normal"][sig]
        # truncate lift to [0, 1 - cov_prior] (mirrors #339's cov_post in [cov_prior, 1]); negligible.
        return normal_prob_ge(t, mean, sigma, low=0.0, high=1.0 - COV_PRIOR)
    if rho == RHO_INDEPENDENT and shape == "triangular":
        # two symmetric triangulars of equal half-width -> Irwin-Hall(4): L = a + (b-a)/4 * IH4.
        width = (b - a) / 4.0                                            # 0.0105
        return 1.0 - ih4_cdf((t - a) / width)
    if rho == RHO_INDEPENDENT and shape == "uniform":
        # two equal-width uniforms -> combo is Triangular[a, mean, b].
        return triangular_sf(t, a, mean, b)
    return None


# --------------------------------------------------------------------------- #
# Monte-Carlo (numpy-optional; deterministic, seed-fixed). Supplies the +0.5-correlated triangular/uniform
# cells (no closed form) and cross-checks every closed-form cell. Gaussian copula matches #339's z-mixing.
# --------------------------------------------------------------------------- #
def _np_ncdf(z):
    """Vectorised standard-normal CDF via A&S 7.1.26 erf approximation (max err ~1.5e-7)."""
    import numpy as np
    x = z / math.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * np.abs(x))
    a1, a2, a3, a4, a5 = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)
    erf = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    erf = np.sign(x) * erf
    return 0.5 * (1.0 + erf)


def _np_invcdf(shape: str, u, band: list[float], central: float):
    import numpy as np
    lo, hi = band
    if shape == "uniform":
        return lo + u * (hi - lo)
    if shape == "triangular":
        fc = (central - lo) / (hi - lo)
        left = lo + np.sqrt(u * (hi - lo) * (central - lo))
        right = hi - np.sqrt((1.0 - u) * (hi - lo) * (hi - central))
        return np.where(u < fc, left, right)
    raise ValueError(shape)


def monte_carlo(geo: dict[str, Any], n: int = MC_SAMPLES, seed: int = MC_SEED) -> dict[str, Any] | None:
    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        return None
    h = COMBO_NON_ADDITIVITY
    kd, da = LEVERS["soft_kd_topk_distill"], LEVERS["more_reasoning_root_data"]
    rng = np.random.default_rng(seed)
    z0 = rng.standard_normal(n)
    z1 = rng.standard_normal(n)

    def draw(shape: str, rho: float):
        zc = z0 if rho >= 1.0 else (rho * z0 + math.sqrt(max(0.0, 1.0 - rho * rho)) * z1)
        if shape == "normal":
            s_kd = h * sigma_lever("normal", kd["band"], kd["central"])
            s_da = h * sigma_lever("normal", da["band"], da["central"])
            x = kd["central"] * h + s_kd * z0          # scaled so combo = N(0.0385, sigma)
            y = da["central"] * h + s_da * zc
            return x + y
        u0, uc = _np_ncdf(z0), _np_ncdf(zc)
        x = _np_invcdf(shape, u0, kd["band"], kd["central"])
        y = _np_invcdf(shape, uc, da["band"], da["central"])
        return h * (x + y)

    out: dict[str, Any] = {"samples": n, "seed": seed, "cells": {}}
    for shape in SHAPES:
        for rho, tag in ((RHO_INDEPENDENT, "independent"), (RHO_CONSERVATIVE, "correlated_0p5"),
                         (RHO_COMONOTONIC, "comonotonic")):
            L = draw(shape, rho)
            out["cells"][f"{shape}__{tag}"] = {
                "central": round(float(np.mean(L >= TARGET_CENTRAL)), 6),
                "worst": round(float(np.mean(L >= TARGET_WORST)), 6),
                "mean": round(float(np.mean(L)), 6),
            }
    # full optimistic 4-lever recipe (combo + deeper-head + TTT, additive), independent, triangular+normal.
    dh, tt = LEVERS["deeper_wider_head"], LEVERS["on_policy_ttt"]
    z2, z3 = rng.standard_normal(n), rng.standard_normal(n)
    for shape in ("triangular", "normal"):
        if shape == "normal":
            base = (kd["central"] * h + h * sigma_lever("normal", kd["band"], kd["central"]) * z0
                    + da["central"] * h + h * sigma_lever("normal", da["band"], da["central"]) * z1)
            extra = (dh["central"] + sigma_lever("normal", dh["band"], dh["central"]) * z2
                     + tt["central"] + sigma_lever("normal", tt["band"], tt["central"]) * z3)
            Lf = base + extra
        else:
            base = h * (_np_invcdf(shape, _np_ncdf(z0), kd["band"], kd["central"])
                        + _np_invcdf(shape, _np_ncdf(z1), da["band"], da["central"]))
            extra = (_np_invcdf(shape, _np_ncdf(z2), dh["band"], dh["central"])
                     + _np_invcdf(shape, _np_ncdf(z3), tt["band"], tt["central"]))
            Lf = base + extra
        out["cells"][f"full4_{shape}__independent"] = {
            "central": round(float(np.mean(Lf >= TARGET_CENTRAL)), 6),
            "worst": round(float(np.mean(Lf >= TARGET_WORST)), 6),
            "mean": round(float(np.mean(Lf)), 6),
        }
    return out


# --------------------------------------------------------------------------- #
# Achievability probabilities (the decision numbers): closed-form where available, else MC.
# --------------------------------------------------------------------------- #
def achievability(geo: dict[str, Any], mc: dict[str, Any] | None) -> dict[str, Any]:
    grid: dict[str, Any] = {}
    for shape in SHAPES:
        for rho, tag in ((RHO_INDEPENDENT, "independent"), (RHO_CONSERVATIVE, "correlated_0p5"),
                         (RHO_COMONOTONIC, "comonotonic")):
            cf_c = closed_form_prob_ge(shape, TARGET_CENTRAL, rho, geo)
            cf_w = closed_form_prob_ge(shape, TARGET_WORST, rho, geo)
            mc_cell = (mc or {}).get("cells", {}).get(f"{shape}__{tag}")
            if cf_c is not None:
                p_c, p_w, src = cf_c, cf_w, "closed_form"
            elif mc_cell is not None:
                p_c, p_w, src = mc_cell["central"], mc_cell["worst"], "monte_carlo"
            else:
                p_c, p_w, src = float("nan"), float("nan"), "unavailable"
            grid[f"{shape}__{tag}"] = {
                "p_central": p_c, "p_worst": p_w, "source": src,
                "sigma": geo["per_shape"][shape][
                    {"independent": "sigma_independent", "correlated_0p5": "sigma_correlated_0p5",
                     "comonotonic": "sigma_comonotonic"}[tag]],
            }
    # headline = triangular, independent (closed form / Irwin-Hall) for the two TEST floats.
    head = grid[f"{HEADLINE_SHAPE}__independent"]
    head_corr = grid[f"{HEADLINE_SHAPE}__correlated_0p5"]
    # cross-shape spread (independent) so the headline is read as a range, not false precision.
    spread_central = sorted(grid[f"{s}__independent"]["p_central"] for s in SHAPES)
    spread_worst = sorted(grid[f"{s}__independent"]["p_worst"] for s in SHAPES)
    return {
        "grid": grid,
        "headline_triangular_independent": {"p_central": head["p_central"], "p_worst": head["p_worst"]},
        "headline_triangular_correlated_0p5": {"p_central": head_corr["p_central"],
                                               "p_worst": head_corr["p_worst"]},
        "cross_shape_independent_range": {
            "p_central_min": spread_central[0], "p_central_max": spread_central[-1],
            "p_worst_min": spread_worst[0], "p_worst_max": spread_worst[-1],
        },
        "full4_optimistic_upper_bound": {
            "triangular": (mc or {}).get("cells", {}).get("full4_triangular__independent"),
            "normal": (mc or {}).get("cells", {}).get("full4_normal__independent"),
        },
    }


def continuity_check_339() -> dict[str, Any]:
    """Re-point this card's NORMAL engine to #339's strict identity lift (+0.031035) -> must reproduce 0.843."""
    geo = combo_geometry()
    sigma = geo["per_shape"]["normal"]["sigma_independent"]
    p = normal_prob_ge(IDENTITY_LIFT, geo["combo_mean"], sigma, low=0.0, high=1.0 - COV_PRIOR)
    return {"identity_lift": IDENTITY_LIFT, "p_clears_identity_independent": p,
            "pr339_reported": 0.843, "reproduces_339": abs(p - 0.843) <= 2e-3}


# --------------------------------------------------------------------------- #
# Decision verdicts.
# --------------------------------------------------------------------------- #
def verdict_for(p_independent: float) -> str:
    if not math.isfinite(p_independent):
        return "UNAVAILABLE"
    if p_independent >= LIKELY_WIN_AT:
        return "LIKELY-WIN"
    if p_independent >= LEAN_WIN_AT:
        return "LEAN-WIN"
    return "UNLIKELY"


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    geo = combo_geometry()
    mc = monte_carlo(geo)
    ach = achievability(geo, mc)
    cont = continuity_check_339()

    p_c = ach["headline_triangular_independent"]["p_central"]
    p_w = ach["headline_triangular_independent"]["p_worst"]
    central_verdict = verdict_for(p_c)
    worst_verdict = verdict_for(p_w)
    rng_c = ach["cross_shape_independent_range"]
    rng_w = ach["cross_shape_independent_range"]

    handoff = (
        f"a feasible soft-KD + reasoning-data EAGLE-3 retrain clears the PPL-only CENTRAL-500 coverage "
        f"target (+{TARGET_CENTRAL_DISPLAY}, c*={CSTAR_CENTRAL:.4f}) with achievability "
        f"{p_c:.3f} (triangular, independent; {rng_c['p_central_min']:.3f}-{rng_c['p_central_max']:.3f} "
        f"across shapes) -> {central_verdict}; and the WORST-500 target (+{TARGET_WORST_DISPLAY}, "
        f"c*={CSTAR_WORST:.4f}) with achievability {p_w:.3f} "
        f"({rng_w['p_worst_min']:.3f}-{rng_w['p_worst_max']:.3f} across shapes; "
        f"{ach['headline_triangular_correlated_0p5']['p_worst']:.3f} at rho=+0.5) -> {worst_verdict}. "
        f"Central-500 is a LIKELY-WIN; worst-500 is coin-flip-OR-BETTER but not a slam-dunk. A definitive "
        f"number needs the measured read (gated, #319/#322).")

    return {
        "imported": {
            "cov_prior": COV_PRIOR, "cov_prior_display": COV_PRIOR_DISPLAY,
            "cstar_central": CSTAR_CENTRAL, "cstar_worst": CSTAR_WORST,
            "target_central": TARGET_CENTRAL, "target_worst": TARGET_WORST,
            "target_central_display": TARGET_CENTRAL_DISPLAY, "target_worst_display": TARGET_WORST_DISPLAY,
            "retrain_lift_budget_336": RETRAIN_LIFT_BUDGET_336,
            "identity_bar": IDENTITY_BAR, "identity_lift": IDENTITY_LIFT,
            "baseline_tps": BASELINE_TPS, "ceiling_tps": CEILING_TPS,
            "levers": {k: {"central": v["central"], "band": v["band"], "citation": v["citation"]}
                       for k, v in LEVERS.items()},
            "combo_non_additivity_haircut": COMBO_NON_ADDITIVITY,
            "provenance": (
                "PPL-only >500 coverage targets c*_central=0.9089 (+0.0186) / c*_worst=0.9256 (+0.0353): "
                "wirbel #343 (kklof4wr, on-disk ppl_only_gate_500_envelope_results.json), itself built on "
                "stark #340 (jwv1vbug) c* map and the human's #124 PPL-only accepted-risk call. Per-lever "
                "Delta-cov bands + 0.70 non-additivity haircut + combo central 0.0385 band [0.0175,0.0595] "
                "+ the +0.031 reachable budget: lawine #336 (krroookz). Honest top-4 ROOT coverage prior "
                "0.8903 (eval 100% reasoning/STEM): lawine #330 (hfrscdai). The independent-vs-+0.5 "
                "correlation convolution method: lawine #339 (0aq16szh). This card convolves the bands into "
                "an achievability probability against the LOWERED PPL-only targets; it re-derives none of "
                "the upstream numbers."),
        },
        "step1_combo_geometry": geo,
        "step2_achievability": ach,
        "step3_continuity_339": cont,
        "step4_monte_carlo": mc,
        "step5_decision": {
            "central_500_target_lift": TARGET_CENTRAL, "worst_500_target_lift": TARGET_WORST,
            "combo_central_lift": geo["combo_mean"], "retrain_lift_budget_336": RETRAIN_LIFT_BUDGET_336,
            "central_within_budget": TARGET_CENTRAL <= RETRAIN_LIFT_BUDGET_336,
            "worst_within_budget": TARGET_WORST <= RETRAIN_LIFT_BUDGET_336,
            "worst_overshoot_fraction_of_budget": (TARGET_WORST - RETRAIN_LIFT_BUDGET_336)
            / RETRAIN_LIFT_BUDGET_336,
            "p_retrain_clears_ppl_only_central_500": p_c,
            "p_retrain_clears_ppl_only_worst_500": p_w,
            "central_500_verdict": central_verdict,
            "worst_500_verdict": worst_verdict,
            "thresholds": {"likely_win_at": LIKELY_WIN_AT, "lean_win_at": LEAN_WIN_AT},
            "note": (
                "Central target +0.0186 sits only +0.0011 above the combo HARD support floor 0.0175 and far "
                "below the combo central +0.0385 -> near-certain clear under every shape (LIKELY-WIN). Worst "
                "target +0.0353 sits just below the combo central +0.0385 and ~14% past #336's +0.031 budget "
                "-> coin-flip-OR-BETTER (LEAN-WIN), not a slam-dunk."),
        },
        "step6_greedy_safety": {
            "card_is_cpu_analytic": True,
            "retrain_run_is_human_gated": True,
            "retrain_target_greedy_identical_by_construction": True,
            "ppl_pinned_le_2p42": True,
            "ppl_only_world_drops_strict_identity_bar_per_124": True,
            "note": (
                "The retrain TARGET is greedy-IDENTICAL by construction (EAGLE-3 drafter only PROPOSES; "
                "emission = verify-model argmax) and PPL-pinned <= 2.42. In the #124 PPL-only world the "
                "STRICT greedy-identity bar is intentionally dropped (accepted risk) while PPL stays the "
                "guardrail. Coverage is the SPEED/acceptance axis (E[T]), NOT the validity axis. BASELINE "
                "481.53 unchanged; 0 TPS; no served-file change; no HF Job; not a launch."),
        },
        "handoff_sentence": handoff,
        "caveats": [
            "The per-lever Delta-cov bands are workload-dependent literature-grounded POINT ESTIMATES "
            "(carry ranges, not false precision); the achievability is conditional on those priors being "
            "calibrated, not a guarantee.",
            "Even a lit-central fully-trained head (~0.913 coverage) clears the central target, but the "
            "official eval is 100% reasoning/STEM (lawine #330) -- the HARD distribution; a head that hits "
            "0.913 on a generic mix may fall short here.",
            "A definitive number needs the MEASURED post-retrain coverage read (human-gated, #319/#322); "
            "this card is an achievability SCREEN, not a build recommendation.",
            "The central probability is shape-sensitive precisely because the target sits near the combo "
            "support floor: 0.996-1.000 across triangular/uniform/normal -- read it as 'near-certain', the "
            "exact decimal is not load-bearing.",
            "Worst-500 achievability lifts well above 0.9 ONLY if the capacity levers (deeper head + TTT, "
            "the full-4-lever optimistic bound) are added on top of the 2-lever combo -- reported as an "
            "upper bound, not the headline.",
        ],
        "test_metrics": {
            "p_retrain_clears_ppl_only_central_500": p_c,
            "p_retrain_clears_ppl_only_worst_500": p_w,
            "central_500_verdict": central_verdict,
            "worst_500_verdict": worst_verdict,
            "retrain_achievability_self_test_passes": False,   # set by main() after the gate runs
        },
    }


# --------------------------------------------------------------------------- #
# Banked-constant drift guard.
# --------------------------------------------------------------------------- #
def load_banked() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, path in (("pr336", PR336_RESULTS), ("pr330", PR330_RESULTS), ("pr343", PR343_RESULTS)):
        if path.exists():
            out[key] = json.loads(path.read_text())
    return out


def drift_ok(banked: dict[str, Any]) -> dict[str, bool]:
    res = {"pr336": False, "pr330": False, "pr343": False}
    r336 = banked.get("pr336")
    if r336:
        ok = True
        syn = r336.get("synthesis", {})
        combo = (syn.get("step3_recipe_ranking", {}) or {}).get("recommended_combination", {})
        if combo.get("delta_cov_central") is not None:
            ok = ok and abs(float(combo["delta_cov_central"]) - 0.0385) <= TOL_REPRO
        if combo.get("non_additivity_factor") is not None:
            ok = ok and abs(float(combo["non_additivity_factor"]) - COMBO_NON_ADDITIVITY) <= TOL_REPRO
        if isinstance(combo.get("delta_cov_band"), list) and len(combo["delta_cov_band"]) == 2:
            ok = ok and abs(float(combo["delta_cov_band"][0]) - 0.0175) <= TOL_REPRO
            ok = ok and abs(float(combo["delta_cov_band"][1]) - 0.0595) <= TOL_REPRO
        ranking = (syn.get("step3_recipe_ranking", {}) or {}).get("ranking", [])
        on_disk = {e["recipe"]: e for e in ranking if isinstance(e, dict) and "recipe" in e}
        for name, spec in LEVERS.items():
            e = on_disk.get(name)
            if e is not None:
                ok = ok and abs(float(e["delta_cov_central"]) - spec["central"]) <= TOL_REPRO
                ok = ok and abs(float(e["delta_cov_band"][0]) - spec["band"][0]) <= TOL_REPRO
                ok = ok and abs(float(e["delta_cov_band"][1]) - spec["band"][1]) <= TOL_REPRO
        t1 = syn.get("step1_lift_target_table", {})
        if t1.get("aggregate_baseline") is not None:
            ok = ok and abs(float(t1["aggregate_baseline"]) - COV_PRIOR) <= TOL_REPRO
        res["pr336"] = bool(ok)
    r330 = banked.get("pr330")
    if r330:
        prior = (((r330.get("synthesis", {}) or {}).get("step2_composition_to_coverage", {}) or {})
                 .get("point_estimate_uncond_top4"))
        res["pr330"] = bool(prior is not None and abs(float(prior) - COV_PRIOR) <= TOL_REPRO)
    r343 = banked.get("pr343")
    if r343:
        d2 = (((r343.get("synthesis", {}) or {}).get("deliverable2_price_and_lift", {})) or {})
        ok = (d2.get("coverage_lift_for_ppl_only_central_500") is not None
              and d2.get("coverage_lift_for_ppl_only_worst_500") is not None
              and d2.get("c_star_central_for_500") is not None
              and d2.get("c_star_worst_for_500") is not None)
        if ok:
            ok = (abs(float(d2["coverage_lift_for_ppl_only_central_500"]) - TARGET_CENTRAL) <= TOL
                  and abs(float(d2["coverage_lift_for_ppl_only_worst_500"]) - TARGET_WORST) <= TOL
                  and abs(float(d2["c_star_central_for_500"]) - CSTAR_CENTRAL) <= TOL
                  and abs(float(d2["c_star_worst_for_500"]) - CSTAR_WORST) <= TOL)
        res["pr343"] = bool(ok)
    return res


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def self_test(syn: dict[str, Any], banked: dict[str, Any]) -> dict[str, Any]:
    geo = syn["step1_combo_geometry"]
    ach = syn["step2_achievability"]
    cont = syn["step3_continuity_339"]
    mc = syn["step4_monte_carlo"]
    grid = ach["grid"]
    c: dict[str, bool] = {}

    drift = drift_ok(banked)

    # (5a) #336 recipe bands round-trip <= 1e-6 (on-disk drift guard + literal band/combo constants).
    c["01_recipe_bands_roundtrip_336"] = bool(
        drift["pr336"] and drift["pr330"]
        and abs(geo["combo_mean"] - 0.0385) <= TOL
        and abs(geo["combo_band_336_reconstructed"][0] - 0.0175) <= TOL
        and abs(geo["combo_band_336_reconstructed"][1] - 0.0595) <= TOL
        and abs(geo["naive_sum_mean"] - 0.055) <= TOL
        and abs(LEVERS["soft_kd_topk_distill"]["central"] - 0.030) <= TOL
        and abs(LEVERS["more_reasoning_root_data"]["central"] - 0.025) <= TOL)

    # (5b) wirbel #343 targets imported EXACT (+0.0186 / +0.0353), and == c* - 0.8903, and on-disk drift.
    c["02_targets_imported_exact_343"] = bool(
        drift["pr343"]
        and round(TARGET_CENTRAL, 4) == TARGET_CENTRAL_DISPLAY
        and round(TARGET_WORST, 4) == TARGET_WORST_DISPLAY
        and abs(TARGET_CENTRAL - (CSTAR_CENTRAL - COV_PRIOR_DISPLAY)) <= TOL
        and abs(TARGET_WORST - (CSTAR_WORST - COV_PRIOR_DISPLAY)) <= TOL
        and TARGET_CENTRAL < TARGET_WORST)

    # combo support + targets bracket: both targets inside [support_lo, combo_mean); worst nearer the mean
    # than central; the +0.031 budget splits them (central within budget < budget < worst past budget).
    c["03_combo_support_and_targets_bracket"] = bool(
        abs(geo["support"][0] - 0.0175) <= TOL and abs(geo["support"][1] - 0.0595) <= TOL
        and geo["support"][0] < TARGET_CENTRAL < geo["combo_mean"]
        and geo["support"][0] < TARGET_WORST < geo["combo_mean"]
        and (geo["combo_mean"] - TARGET_WORST) < (geo["combo_mean"] - TARGET_CENTRAL)
        and TARGET_CENTRAL < RETRAIN_LIFT_BUDGET_336 < TARGET_WORST)

    # sigma ordering across shapes (independent): triangular < normal < uniform; comonotonic-normal=0.0105.
    s_ind = {s: geo["per_shape"][s]["sigma_independent"] for s in SHAPES}
    c["04_sigma_ordering_shapes"] = bool(
        s_ind["triangular"] < s_ind["normal"] < s_ind["uniform"]
        and abs(geo["per_shape"]["normal"]["sigma_comonotonic"] - 0.0105) <= TOL
        and abs(geo["per_shape"]["normal"]["sigma_independent"] - 0.70 * 0.0075 * SQRT2) <= TOL)

    # correlation widens sigma (independent < 0.5 < comonotonic) for every shape.
    c["05_correlation_widens_sigma"] = bool(all(
        geo["per_shape"][s]["sigma_independent"] < geo["per_shape"][s]["sigma_correlated_0p5"]
        < geo["per_shape"][s]["sigma_comonotonic"] for s in SHAPES))

    # (5c) both convolutions NaN-clean: every P over shapes x {independent, correlated_0p5} finite in [0,1].
    flat = []
    for s in SHAPES:
        for tag in ("independent", "correlated_0p5"):
            flat.extend([grid[f"{s}__{tag}"]["p_central"], grid[f"{s}__{tag}"]["p_worst"]])
    c["06_nan_clean_probabilities_in_unit"] = bool(
        all(math.isfinite(p) and -1e-9 <= p <= 1.0 + 1e-9 for p in flat))

    # (5d) monotone: P(>= +0.0186) >= P(>= +0.0353) for every shape x rho (central target < worst target).
    c["07_p_monotone_central_ge_worst"] = bool(all(
        grid[f"{s}__{tag}"]["p_central"] >= grid[f"{s}__{tag}"]["p_worst"] - 1e-9
        for s in SHAPES for tag in ("independent", "correlated_0p5", "comonotonic")))

    # (5e) independent vs correlated BOTH reported, and wider spread lowers P (targets below combo mean).
    c["08_independent_and_correlated_reported"] = bool(all(
        grid[f"{s}__independent"]["p_central"] >= grid[f"{s}__correlated_0p5"]["p_central"] - 1e-6
        and grid[f"{s}__independent"]["p_worst"] >= grid[f"{s}__correlated_0p5"]["p_worst"] - 1e-6
        for s in SHAPES))

    # headline = triangular independent closed-form; both in [0,1]; central high, worst moderate.
    head = ach["headline_triangular_independent"]
    c["09_headline_triangular_closed_form"] = bool(
        grid["triangular__independent"]["source"] == "closed_form"
        and 0.0 <= head["p_central"] <= 1.0 and 0.0 <= head["p_worst"] <= 1.0
        and head["p_central"] >= 0.99 and 0.5 <= head["p_worst"] < 0.9)

    # closed-form vs MC agreement (<=5e-3) where both exist (skipped cleanly if numpy absent).
    if mc is not None:
        agree = True
        for s in SHAPES:
            for tag in ("independent", "correlated_0p5", "comonotonic"):
                cell = mc["cells"].get(f"{s}__{tag}")
                cf_c = closed_form_prob_ge(s, TARGET_CENTRAL,
                                           {"independent": 0.0, "correlated_0p5": 0.5,
                                            "comonotonic": 1.0}[tag], geo)
                cf_w = closed_form_prob_ge(s, TARGET_WORST,
                                           {"independent": 0.0, "correlated_0p5": 0.5,
                                            "comonotonic": 1.0}[tag], geo)
                if cell is not None and cf_c is not None:
                    agree = agree and abs(cell["central"] - cf_c) <= 5e-3
                    agree = agree and abs(cell["worst"] - cf_w) <= 5e-3
        c["10_mc_matches_closed_form"] = bool(agree)
    else:
        c["10_mc_matches_closed_form"] = True

    # (continuity) the NORMAL engine re-points to #339's identity lift -> reproduces 0.843 (<=2e-3).
    c["11_normal_reproduces_339_at_identity"] = bool(cont["reproduces_339"])

    # verdicts: central LIKELY-WIN under every shape (P>=0.9), worst LEAN-WIN under every shape (0.5..0.9).
    c["12_central_likelywin_worst_leanwin"] = bool(
        all(grid[f"{s}__independent"]["p_central"] >= LIKELY_WIN_AT for s in SHAPES)
        and all(LEAN_WIN_AT <= grid[f"{s}__independent"]["p_worst"] < LIKELY_WIN_AT for s in SHAPES)
        and syn["step5_decision"]["central_500_verdict"] == "LIKELY-WIN"
        and syn["step5_decision"]["worst_500_verdict"] == "LEAN-WIN")

    # full-4-lever optimistic bound (if MC present): worst achievability rises ABOVE the 2-lever combo.
    if mc is not None:
        f4 = mc["cells"].get("full4_triangular__independent")
        combo_w = grid["triangular__independent"]["p_worst"]
        c["13_full4_optimistic_ge_combo_worst"] = bool(f4 is not None and f4["worst"] >= combo_w)
    else:
        c["13_full4_optimistic_ge_combo_worst"] = True

    gate = all(bool(v) for v in c.values())
    return {"retrain_achievability_self_test_passes": gate, "checks": c, "drift": drift}


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
# Determinism: two synthesize() passes must serialize identically (MC seed-fixed).
# --------------------------------------------------------------------------- #
def determinism_ok() -> bool:
    a = synthesize()
    b = synthesize()
    a["test_metrics"]["retrain_achievability_self_test_passes"] = True
    b["test_metrics"]["retrain_achievability_self_test_passes"] = True
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
        print(f"[ppl-only-retrain-achievability] wandb logging unavailable: {exc}", flush=True)
        return None

    syn = payload["synthesis"]
    geo = syn["step1_combo_geometry"]
    ach = syn["step2_achievability"]
    dec = syn["step5_decision"]
    grid = ach["grid"]
    st = payload["self_test"]

    run = init_wandb_run(
        job_type="validity-analytic", agent="lawine", name=args.wandb_name, group=args.wandb_group,
        tags=["ppl-only-retrain-achievability", "validity-analytic", "achievability", "eagle3",
              "demand-half", "retrain-roi", "ppl-only", "convolution", "bank-the-analysis"],
        config={
            "pr": 346, "cov_prior": COV_PRIOR, "cstar_central": CSTAR_CENTRAL, "cstar_worst": CSTAR_WORST,
            "target_central": TARGET_CENTRAL, "target_worst": TARGET_WORST,
            "combo_non_additivity_haircut": COMBO_NON_ADDITIVITY, "recipe_levers": RECIPE_LEVERS,
            "shapes": SHAPES, "headline_shape": HEADLINE_SHAPE, "retrain_lift_budget_336":
            RETRAIN_LIFT_BUDGET_336, "baseline_tps": BASELINE_TPS, "ceiling_tps": CEILING_TPS,
            "mc_samples": MC_SAMPLES, "wandb_group": args.wandb_group,
            "imports": syn["imported"]["provenance"],
        },
    )
    if run is None:
        print("[ppl-only-retrain-achievability] wandb: no run (no WANDB_API_KEY/mode) -- skipping",
              flush=True)
        return None

    summary: dict[str, Any] = {
        # PRIMARY + TEST + REPORT
        "retrain_achievability_self_test_passes": int(bool(
            st["retrain_achievability_self_test_passes"])),
        "p_retrain_clears_ppl_only_central_500": dec["p_retrain_clears_ppl_only_central_500"],
        "p_retrain_clears_ppl_only_worst_500": dec["p_retrain_clears_ppl_only_worst_500"],
        "central_500_likely_win": int(dec["central_500_verdict"] == "LIKELY-WIN"),
        "worst_500_lean_win": int(dec["worst_500_verdict"] == "LEAN-WIN"),
        # headline shape grid (central / worst), independent + correlated, all three shapes
        "tri_independent_central": grid["triangular__independent"]["p_central"],
        "tri_independent_worst": grid["triangular__independent"]["p_worst"],
        "tri_correlated_0p5_central": grid["triangular__correlated_0p5"]["p_central"],
        "tri_correlated_0p5_worst": grid["triangular__correlated_0p5"]["p_worst"],
        "uniform_independent_central": grid["uniform__independent"]["p_central"],
        "uniform_independent_worst": grid["uniform__independent"]["p_worst"],
        "uniform_correlated_0p5_central": grid["uniform__correlated_0p5"]["p_central"],
        "uniform_correlated_0p5_worst": grid["uniform__correlated_0p5"]["p_worst"],
        "normal_independent_central": grid["normal__independent"]["p_central"],
        "normal_independent_worst": grid["normal__independent"]["p_worst"],
        "normal_correlated_0p5_central": grid["normal__correlated_0p5"]["p_central"],
        "normal_correlated_0p5_worst": grid["normal__correlated_0p5"]["p_worst"],
        # cross-shape ranges
        "central_range_min": ach["cross_shape_independent_range"]["p_central_min"],
        "central_range_max": ach["cross_shape_independent_range"]["p_central_max"],
        "worst_range_min": ach["cross_shape_independent_range"]["p_worst_min"],
        "worst_range_max": ach["cross_shape_independent_range"]["p_worst_max"],
        # geometry
        "combo_central_lift": geo["combo_mean"], "combo_support_lo": geo["support"][0],
        "combo_support_hi": geo["support"][1],
        "sigma_tri_independent": geo["per_shape"]["triangular"]["sigma_independent"],
        "sigma_normal_independent": geo["per_shape"]["normal"]["sigma_independent"],
        "sigma_uniform_independent": geo["per_shape"]["uniform"]["sigma_independent"],
        # budget
        "central_within_budget": int(dec["central_within_budget"]),
        "worst_within_budget": int(dec["worst_within_budget"]),
        "worst_overshoot_fraction_of_budget": dec["worst_overshoot_fraction_of_budget"],
        # continuity with #339
        "continuity_339_p_identity": syn["step3_continuity_339"]["p_clears_identity_independent"],
        "continuity_339_reproduces": int(bool(syn["step3_continuity_339"]["reproduces_339"])),
        # context
        "cstar_central": CSTAR_CENTRAL, "cstar_worst": CSTAR_WORST, "baseline_tps": BASELINE_TPS,
        # hygiene
        "nan_clean": int(bool(payload["nan_clean"])),
        "determinism_ok": int(bool(payload["determinism_ok"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(val)) for k, val in st["checks"].items()},
        **{f"drift_{k}": int(bool(val)) for k, val in st["drift"].items()},
    }
    mc = syn["step4_monte_carlo"]
    if mc is not None:
        f4t = mc["cells"].get("full4_triangular__independent")
        if f4t is not None:
            summary["full4_triangular_worst"] = f4t["worst"]
            summary["full4_triangular_central"] = f4t["central"]
    summary = {k: val for k, val in summary.items()
               if not (isinstance(val, float) and not math.isfinite(val)) and val is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="ppl_only_retrain_achievability_result", artifact_type="validity",
                      data=payload)
    rid = getattr(run, "id", None)
    print(f"[ppl-only-retrain-achievability] wandb run: {getattr(run, 'url', rid)}", flush=True)
    finish_wandb(run)
    return rid


# --------------------------------------------------------------------------- #
# Console report.
# --------------------------------------------------------------------------- #
def print_report(payload: dict) -> None:
    syn = payload["synthesis"]
    geo = syn["step1_combo_geometry"]
    ach = syn["step2_achievability"]
    dec = syn["step5_decision"]
    grid = ach["grid"]
    st = payload["self_test"]
    print("\n" + "=" * 100, flush=True)
    print("PPL-ONLY RETRAIN ACHIEVABILITY (PR #346) -- P(retrain hits the LOWERED PPL-only coverage target)?",
          flush=True)
    print("=" * 100, flush=True)
    print(f"cov_prior 0.8903   combo lift ~ (mean {geo['combo_mean']:.4f}, support "
          f"[{geo['support'][0]:.4f},{geo['support'][1]:.4f}])   +0.031 budget (lawine #336)", flush=True)
    print(f"targets: central +{TARGET_CENTRAL:.6f} (c*={CSTAR_CENTRAL:.4f}, within budget)   "
          f"worst +{TARGET_WORST:.6f} (c*={CSTAR_WORST:.4f}, "
          f"{dec['worst_overshoot_fraction_of_budget']*100:.1f}% past budget)", flush=True)
    print("-" * 100, flush=True)
    print("ACHIEVABILITY GRID  P(combo lift >= target):  [shape x correlation]  central | worst", flush=True)
    for s in SHAPES:
        for tag in ("independent", "correlated_0p5", "comonotonic"):
            g = grid[f"{s}__{tag}"]
            mark = "  <== HEADLINE" if (s == HEADLINE_SHAPE and tag == "independent") else ""
            print(f"   {s:11s} {tag:15s} (sig={g['sigma']:.5f}, {g['source']:11s}):  "
                  f"central={g['p_central']:.4f}  worst={g['p_worst']:.4f}{mark}", flush=True)
    rng = ach["cross_shape_independent_range"]
    print(f"   cross-shape independent range: central {rng['p_central_min']:.4f}-{rng['p_central_max']:.4f}"
          f"   worst {rng['p_worst_min']:.4f}-{rng['p_worst_max']:.4f}", flush=True)
    f4 = ach["full4_optimistic_upper_bound"]["triangular"]
    if f4 is not None:
        print(f"   full-4-lever optimistic upper bound (triangular, indep): central={f4['central']:.4f}  "
              f"worst={f4['worst']:.4f}", flush=True)
    print("-" * 100, flush=True)
    print(f"DECISION:  central-500 -> {dec['central_500_verdict']}  "
          f"(P={dec['p_retrain_clears_ppl_only_central_500']:.4f})    "
          f"worst-500 -> {dec['worst_500_verdict']}  (P={dec['p_retrain_clears_ppl_only_worst_500']:.4f})",
          flush=True)
    cont = syn["step3_continuity_339"]
    print(f"CONTINUITY #339: normal engine @ identity lift {cont['identity_lift']:.5f} -> "
          f"P={cont['p_clears_identity_independent']:.4f} (vs #339 0.843; reproduces={cont['reproduces_339']})",
          flush=True)
    print("-" * 100, flush=True)
    print(f"HAND-OFF: {syn['handoff_sentence']}", flush=True)
    print("-" * 100, flush=True)
    print(f"(PRIMARY) retrain_achievability_self_test_passes = "
          f"{st['retrain_achievability_self_test_passes']}", flush=True)
    for k, val in st["checks"].items():
        print(f"   - {k}: {val}", flush=True)
    print(f"   drift: {st['drift']}", flush=True)
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
                    default="ppl-only-retrain-achievability")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    banked = load_banked()
    syn = synthesize()
    st = self_test(syn, banked)
    syn["test_metrics"]["retrain_achievability_self_test_passes"] = bool(
        st["retrain_achievability_self_test_passes"])

    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "pr": 346, "agent": "lawine", "kind": "ppl-only-retrain-achievability",
        "ppl_only_retrain_achievability_analysis_only": True,
        "synthesis": syn, "self_test": st,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }

    nan_paths = assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[ppl-only-retrain-achievability] WARNING non-finite at: {nan_paths}", flush=True)
    payload["determinism_ok"] = determinism_ok()

    gate = bool(st["retrain_achievability_self_test_passes"] and payload["nan_clean"]
                and payload["determinism_ok"])
    st["retrain_achievability_self_test_passes"] = gate
    syn["test_metrics"]["retrain_achievability_self_test_passes"] = gate

    print_report(payload)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ppl_only_retrain_achievability_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[ppl-only-retrain-achievability] wrote {out_path}", flush=True)

    rid = maybe_log_wandb(args, payload)
    payload["wandb_run_id"] = rid
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    dec = syn["step5_decision"]
    print(f"  PRIMARY retrain_achievability_self_test_passes = {gate}", flush=True)
    print(f"  TEST p_retrain_clears_ppl_only_central_500 = "
          f"{dec['p_retrain_clears_ppl_only_central_500']:.4f}  ({dec['central_500_verdict']})", flush=True)
    print(f"  TEST p_retrain_clears_ppl_only_worst_500   = "
          f"{dec['p_retrain_clears_ppl_only_worst_500']:.4f}  ({dec['worst_500_verdict']})", flush=True)
    print(f"  wandb run = {rid}", flush=True)

    if args.self_test:
        print(f"[ppl-only-retrain-achievability] self-test {'PASS' if gate else 'FAIL'}", flush=True)
        return 0 if gate else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
