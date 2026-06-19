# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #741 (land) -- Calibrate the G1 gate-model: turn delta_stock into a real P(DQ).

ANALYSIS ONLY. Reads PUBLIC in-repo competition artifacts. No GPU, no HF Job,
no submission, no served-file change. Guard flags live in wandb.summary.

The binding #730-fire gate is G1: the organizer's private re-run TPS must be
>= 0.95 x the submission's own public TPS (realized drift Delta <= 5%).

kanna #737 reported P(DQ)=0.80. That number is P(U[4%,9%] > 5%) -- the tail mass
of a PESSIMISTIC PRIOR whose center (6.5%) already exceeds the gate. It is NOT a
gate-calibrated probability; it never uses the gate's measurement model nor the
empirical realized-Delta distribution of the spec-dec stacks already on the board.

This card pins the gate model from public verifier artifacts and reconstructs the
empirical realized-Delta distribution, then calibrates P(DQ | delta_stock) =
P(realized Delta > 5% | delta_stock) as a parametric curve over delta in [0,12%].
The advisor reads kanna/lawine's MEASURED delta_stock straight off the curve.

Sources (all in-repo, public):
  - official/main_bucket/shared_resources/tps_repro_gap_itaca/README.md  (17 cmpatino
    verifier private re-runs with realized Delta%, PPL, valid/invalid)
  - research/validity/precache_gate_provenance/cmpatino_verifier_20260613-230441-229.md
    (#52 flagship: 481.53 -> 460.85 private, Delta 4.3%, VALID, completed=128)
  - BASELINE.md L36-49 (gate rule, 4-9% spec-dec drift band, #52 anchor)
  - research/validity/served_gate_reconciliation.md (harness = PPL+completion+modalities)
  - research/validity/public_private_gap_decomposition (#318: the 4.295% gap is a
    SYSTEMATIC OOD-acceptance property, not draw noise -> small sigma_gate)
  - research/validity/private_adverse_skew/PLAN.md (#164: same aggregate-4.3%, two
    private mixes -> drops 0.34pp apart -> shape/construction realization noise)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------
# Gate constants (PINNED from public artifacts)
# ----------------------------------------------------------------------------
GATE_TOL = 0.05          # private TPS must be >= 0.95 x public  => Delta <= 5%
N_PRIV = 128             # private prompt-set size (cmpatino #52 re-run completed=128)
GATE_IS_SINGLE_DRAW = True   # itaca open-Q #2 ("does double-sampling help?") => single
GATE_METRIC_IS_RAW_TPS = True  # "re-run TPS (private set)" vs "reported TPS", raw ratio

# Named delta_stock anchors used across the fleet (#737)
DRIFT_FLAGSHIP_52 = 0.043   # #52 flagship private re-run (wide-trained drafter), VALID
DRIFT_PRIOR_CENTER = 0.065  # U[4,9] documented-prior CENTER (the 0.80 driver)
DRIFT_44_PROXY = 0.124      # kanna #44 chat-proxy upper bound, would-FAIL
DRIFT_PRIOR_LO, DRIFT_PRIOR_HI = 0.04, 0.09  # #725/#737 documented band
NAIVE_P_DQ_737 = 0.80       # = P(U[4,9] > 5%) = (9-5)/(9-4)

# Reference delta_stock for the scalar primary_metric. We use the #52 same-stack-family
# (int4 MTP) VALID anchor 4.3% -- the empirically-grounded central drift for a near-
# frontier int4-MTP stack. NOTE: this is the FAVORABLE (wide-trained) anchor; the fire's
# stock-Hub drafter is NOT wide-trained (#737 L62) and may drift higher -- the curve, not
# this point, is the deliverable.
DELTA_REF = DRIFT_FLAGSHIP_52
SIGMA_CENTRAL = 0.010       # balanced central gate-noise (between 0.5% construction and
                            # 1.5% same-submission replicate); see sigma evidence below.

# ----------------------------------------------------------------------------
# Empirical verifier records (PUBLIC, in-repo)
# ----------------------------------------------------------------------------
# itaca's 17 cmpatino-verifier private re-runs (tps_repro_gap_itaca/README.md verdict
# table). Every record is a drafter/MTP spec-dec frontier stack (osoi/osoi5 family).
# fields: reported_tps, private_tps, reported_delta_pct, pub_ppl, priv_ppl, verdict, method
ITACA = [
    (419.34, 395.00, 5.80, 2.3813, 2.3811, "invalid", "kenyan-duma osoi5-feopt2-w20-e1-kduma-v1"),
    (418.80, 403.12, 3.70, 2.3813, 2.3806, "valid",   "kenyan-duma osoi5-feopt2-w20-e1-kduma-v1"),
    (416.65, 395.96, 5.00, 2.3806, 2.3806, "valid",   "vejja fsab32-vejja-v0"),
    (416.57, 405.30, 2.70, 2.3806, 2.3808, "valid",   "pupa-agent w24-probe-v0"),
    (415.25, 403.43, 2.80, 2.3811, 2.3806, "valid",   "kenyan-duma osoi5-feopt2-w20-e1-kduma-v1"),
    (412.10, 379.74, 7.90, 2.2558, 2.2555, "invalid", "kenyan-duma osoi-drafterft-feopt2-kduma-v1"),
    (411.58, 396.21, 3.70, 2.3806, 2.3806, "valid",   "jake-bot-2 epoch1-v0"),
    (404.58, 368.53, 8.90, 2.2557, 2.2555, "invalid", "braiam-fable osoi-v0-drafterft-feopt2-v0"),
    (399.41, 389.86, 2.40, 2.3811, 2.3811, "valid",   "jake-bot-2 osoi5-feopt2-w20-v0"),
    (389.00, 365.36, 6.08, 2.3806, 2.3806, "invalid", "braiam-fable osoi5-drafterft-w40-v0"),
    (388.63, 363.66, 6.43, 2.3806, 2.3806, "invalid", "neuralaxsagent-2 adaptive-centroid-onegraph-v12"),
    (388.46, 367.02, 5.52, 2.3813, 2.3806, "invalid", "hayai-agent osoi5-drafterft-w40-ct48-v0"),
    (387.50, 364.08, 6.04, 2.3811, 2.3813, "invalid", "kenyan-duma osoi5-drafterft-kduma-v1"),
    (387.15, 362.80, 6.29, 2.3811, 2.3808, "invalid", "paxenos-gemma-boom osoi5-drafterft-syspack-v0"),
    (378.71, 358.14, 5.40, 2.2555, 2.2556, "invalid", "braiam-fable osoi-v0-drafterft-w40-v0"),
    (378.42, 357.01, 5.66, 2.2555, 2.2556, "invalid", "hayai-agent osoi-drafterft-w40-v0"),
    (377.32, 361.02, 4.32, 2.2555, 2.2555, "valid",   "kenyan-duma osoi-drafterft-kduma-v1"),
]
# #52 flagship anchor (cmpatino_verifier_20260613-230441-229.md). Later window than
# itaca's; the highest-TPS int4-MTP stack verified VALID. SAME stack family as the fire.
FLAGSHIP_52 = (481.53, 460.85, 4.30, 2.3772, 2.3777, "valid", "lawine #52 fa2sw_precache_kenyan (int4 MTP K=7)")

FRONTIER_TPS_FLOOR = 410.0   # itaca pattern #3: >=410 TPS cluster is the near-frontier analog

# ----------------------------------------------------------------------------
# math helpers (scipy unavailable; use erf)
# ----------------------------------------------------------------------------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam rational approximation; abs err < 1.2e-9)."""
    if not (0.0 < p < 1.0):
        return float("-inf") if p <= 0 else float("inf")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def p_dq(delta_stock: float, sigma: float, n_rerun: int = 1) -> float:
    """P(realized single/averaged Delta > 5% | true drift delta_stock).

    Gate draws realized Delta = delta_stock + eps, eps ~ N(0, sigma_eff),
    sigma_eff = sigma / sqrt(n_rerun) (averaging shrinks the noise). DQ iff > 5%.
    """
    if sigma <= 0:
        return 1.0 if delta_stock > GATE_TOL else 0.0
    sig_eff = sigma / math.sqrt(max(1, n_rerun))
    return 1.0 - norm_cdf((GATE_TOL - delta_stock) / sig_eff)


def delta_at_pdq(target: float, sigma: float) -> float:
    """delta_stock at which calibrated P(DQ)=target (breakeven solve)."""
    if sigma <= 0:
        return GATE_TOL
    return GATE_TOL - sigma * norm_ppf(1.0 - target)


def prior_integral_pdq(sigma: float, lo: float, hi: float, n: int = 20001) -> float:
    """E_{delta ~ U[lo,hi]}[P(DQ|delta,sigma)] -- apples-to-apples vs naive 0.80."""
    xs = np.linspace(lo, hi, n)
    ys = np.array([p_dq(x, sigma) for x in xs])
    return float(np.trapezoid(ys, xs) / (hi - lo))


# ----------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------
def analyze() -> dict:
    rows = list(ITACA) + [FLAGSHIP_52]
    rep = np.array([r[0] for r in rows])
    priv = np.array([r[1] for r in rows])
    rep_delta = np.array([r[2] for r in rows]) / 100.0     # as reported in the artifact
    verdict = [r[5] for r in rows]
    is_invalid = np.array([v == "invalid" for v in verdict])

    # recomputed Delta = (public - private)/public  -- self-test vs the reported Delta%
    recomp_delta = (rep - priv) / rep
    delta_recompute_maxerr = float(np.max(np.abs(recomp_delta - rep_delta)))

    # --- gate-model pin: is the 5% rule a clean deterministic step on realized Delta? ---
    valid_mask = ~is_invalid
    max_valid_delta = float(np.max(rep_delta[valid_mask]))      # highest Delta still VALID
    min_invalid_delta = float(np.min(rep_delta[is_invalid]))    # lowest Delta that DQ'd
    clean_5pct_separation = bool(max_valid_delta <= GATE_TOL < min_invalid_delta)

    # --- empirical realized-Delta distribution (the spec-dec population) ---
    d = rep_delta
    dist = {
        "n_records": int(len(d)),
        "n_itaca": len(ITACA),
        "n_flagship_anchor": 1,
        "mean_pct": float(np.mean(d) * 100),
        "median_pct": float(np.median(d) * 100),
        "std_pct": float(np.std(d, ddof=1) * 100),
        "min_pct": float(np.min(d) * 100),
        "max_pct": float(np.max(d) * 100),
        "q25_pct": float(np.percentile(d, 25) * 100),
        "q75_pct": float(np.percentile(d, 75) * 100),
        "invalid_fraction": float(np.mean(is_invalid)),
        "n_invalid": int(np.sum(is_invalid)),
    }
    # near-frontier (>=410 TPS) cluster -- the relevant analog for a near-frontier fire
    fr = d[rep >= FRONTIER_TPS_FLOOR]
    fr_inv = is_invalid[rep >= FRONTIER_TPS_FLOOR]
    frontier = {
        "tps_floor": FRONTIER_TPS_FLOOR,
        "n": int(len(fr)),
        "mean_pct": float(np.mean(fr) * 100),
        "median_pct": float(np.median(fr) * 100),
        "max_pct": float(np.max(fr) * 100),
        "invalid_fraction": float(np.mean(fr_inv)),
    }

    # --- sigma_gate evidence: the ONLY same-config replicate in the public data ---
    # kenyan-duma osoi5-feopt2-w20-e1-kduma-v1 was re-verified 3x (rows 0,1,4): same code,
    # same weights, three verdicts. Direct measure of per-config gate-verdict noise.
    kd_idx = [i for i, r in enumerate(ITACA) if r[6] == "kenyan-duma osoi5-feopt2-w20-e1-kduma-v1"]
    kd_delta = np.array([ITACA[i][2] for i in kd_idx]) / 100.0
    kd_invalid = np.array([ITACA[i][5] == "invalid" for i in kd_idx])
    kd_mean = float(np.mean(kd_delta))
    kd_std = float(np.std(kd_delta, ddof=1))
    kd_dq_rate = float(np.mean(kd_invalid))
    # sigma implied by fitting the DQ-rate at the replicate's mean drift to a Gaussian gate
    if 0 < kd_dq_rate < 1:
        sigma_from_dqrate = (GATE_TOL - kd_mean) / norm_ppf(1.0 - kd_dq_rate)
    else:
        sigma_from_dqrate = float("nan")
    replicate = {
        "submission": "kenyan-duma osoi5-feopt2-w20-e1-kduma-v1",
        "n_reverifications": len(kd_idx),
        "realized_delta_pct": [float(x * 100) for x in kd_delta],
        "mean_delta_pct": kd_mean * 100,
        "std_delta_pct": kd_std * 100,            # empirical per-config gate noise (upper)
        "empirical_dq_rate": kd_dq_rate,          # 1/3 at mean-drift ~4.1% -- NOT 0.80
        "sigma_implied_by_dqrate_pct": (sigma_from_dqrate * 100
                                        if not math.isnan(sigma_from_dqrate) else None),
    }

    # sigma_gate bracket (per-config realized-Delta noise around its true drift):
    #   0.2%  itaca within-bucket engine-noise floor (deterministic-gate extreme)
    #   0.5%  #164 construction/shape sensitivity (same aggregate-4.3%, 0.34pp spread)
    #   1.0%  balanced central (this card's headline sigma)
    #   1.5%  kenyan-duma 3-repost sample std (noisy-single-draw upper bound)
    SIGMAS = {"engine_floor_0p2": 0.002, "construction_0p5": 0.005,
              "central_1p0": 0.010, "replicate_upper_1p5": 0.015}

    # --- calibrated P(DQ | delta_stock) curve over [0,12%] ---
    grid = np.round(np.arange(0.0, 0.1201, 0.0005), 6)
    curve = {name: [p_dq(float(x), s) for x in grid] for name, s in SIGMAS.items()}

    anchors = {"flagship_52_4p3": DRIFT_FLAGSHIP_52, "gate_5p0": GATE_TOL,
               "prior_center_6p5": DRIFT_PRIOR_CENTER, "chatproxy_44_12p4": DRIFT_44_PROXY}
    anchor_pdq = {a: {name: p_dq(dv, s) for name, s in SIGMAS.items()}
                  for a, dv in anchors.items()}

    # breakeven: below which delta_stock the calibrated P(DQ) < the naive 0.80
    breakeven_vs_0p80 = {name: delta_at_pdq(NAIVE_P_DQ_737, s) for name, s in SIGMAS.items()}
    coinflip_delta = {name: delta_at_pdq(0.5, s) for name, s in SIGMAS.items()}  # = 5.0%

    # apples-to-apples: calibrated P(DQ) integrated over kanna's SAME U[4,9] prior
    same_prior = {name: prior_integral_pdq(s, DRIFT_PRIOR_LO, DRIFT_PRIOR_HI)
                  for name, s in SIGMAS.items()}

    # "single draw vs averaged" lever (PR Q1a): averaging n re-runs shrinks sigma_eff.
    averaging = {f"n_rerun_{n}": p_dq(DELTA_REF, SIGMA_CENTRAL, n_rerun=n) for n in (1, 2, 3, 5)}

    # --- the scalar primary_metric: calibrated P(DQ) at the stated reference delta ---
    p_dq_calibrated_ref = p_dq(DELTA_REF, SIGMA_CENTRAL)
    p_dq_ref_range = {name: p_dq(DELTA_REF, s) for name, s in SIGMAS.items()}

    # central of the empirical spec-dec realized-Delta distribution (the TEST metric)
    empirical_central_pct = dist["median_pct"]
    # decision-relevant readouts: P(DQ) at the unconditional central vs the frontier central
    p_dq_at_empirical_median = p_dq(dist["median_pct"] / 100.0, SIGMA_CENTRAL)
    p_dq_at_frontier_median = p_dq(frontier["median_pct"] / 100.0, SIGMA_CENTRAL)

    # --- verdict ---
    safer_if_delta_below_central = breakeven_vs_0p80["central_1p0"] * 100
    verdict = {
        "primary_metric_name": "p_dq_g1_calibrated",
        "primary_metric_value": round(p_dq_calibrated_ref, 4),
        "primary_reference_delta_stock_pct": DELTA_REF * 100,
        "primary_reference_sigma_gate_pct": SIGMA_CENTRAL * 100,
        "test_metric_name": "empirical_specdec_delta_central_pct",
        "test_metric_value": round(empirical_central_pct, 4),
        "naive_737_p_dq": NAIVE_P_DQ_737,
        "naive_737_is_prior_tail_not_gate": True,
        # the headline comparison
        "calibrated_at_ref_4p3_lt_naive_0p80": bool(p_dq_calibrated_ref < NAIVE_P_DQ_737),
        "p_dq_ref_range_over_sigma": {k: round(v, 4) for k, v in p_dq_ref_range.items()},
        "empirical_replicate_p_dq_at_4p1": kd_dq_rate,  # direct empirical anchor (1/3)
        "p_dq_at_empirical_median_5p5_sigma1p0": round(p_dq_at_empirical_median, 4),
        "p_dq_at_frontier_median_4p0_sigma1p0": round(p_dq_at_frontier_median, 4),
        "fire_safer_than_0p80_if_delta_stock_below_pct": round(safer_if_delta_below_central, 3),
        "dominant_correction": "prior_recenter_6p5_to_empirical_4p3_NOT_gate_noise",
        "gate_noise_alone_over_same_U49_prior": {k: round(v, 4) for k, v in same_prior.items()},
        "verdict_band": _verdict_band(p_dq_calibrated_ref, NAIVE_P_DQ_737, safer_if_delta_below_central),
        "fire_now": "PARAMETRIC -- read kanna/lawine measured delta_stock off curve; "
                    "SAFER than 0.80 iff measured central < ~%.1f%% (sigma 1.0%%); the "
                    "#52 same-family anchor 4.3%% sits there, but the fire's stock drafter "
                    "is NOT wide-trained (#737 L62) and may drift higher." % safer_if_delta_below_central,
    }

    return {
        "pr": 741, "student": "land", "analysis_only": True, "official_tps": 0,
        "gate_model": {
            "binding_gate": "G1_private_TPS_reproduction",
            "rule": "private_TPS >= 0.95 x own_public_TPS  (realized Delta <= 5%)",
            "tolerance_pct": GATE_TOL * 100,
            "direction": "one_sided_private_below_public",
            "single_draw_not_averaged": GATE_IS_SINGLE_DRAW,
            "private_set_size": N_PRIV,
            "metric_is_raw_tps_ratio": GATE_METRIC_IS_RAW_TPS,
            "no_token_identity_stage": True,  # served_gate_reconciliation + #318 harness audit
            "clean_5pct_step_separation": clean_5pct_separation,
            "max_valid_delta_pct": max_valid_delta * 100,
            "min_invalid_delta_pct": min_invalid_delta * 100,
            "evidence": "tps_repro_gap_itaca (17 verdicts) + cmpatino #52 verifier + BASELINE L36-49",
        },
        "empirical_distribution": dist,
        "frontier_cluster": frontier,
        "replicate_sigma_evidence": replicate,
        "sigma_gate_bracket_pct": {k: v * 100 for k, v in SIGMAS.items()},
        "calibration": {
            "model": "realized Delta = delta_stock + N(0,sigma_gate); DQ iff > 5%",
            "delta_grid_pct": [float(x * 100) for x in grid],
            "p_dq_curve": curve,
            "anchor_p_dq": anchor_pdq,
            "breakeven_delta_vs_naive_0p80_pct": {k: v * 100 for k, v in breakeven_vs_0p80.items()},
            "coinflip_delta_pct": {k: v * 100 for k, v in coinflip_delta.items()},
            "gate_noise_over_U49_prior": same_prior,
            "averaging_lever_p_dq_at_ref": averaging,
        },
        "self_test": _self_tests(
            delta_recompute_maxerr, clean_5pct_separation, curve, grid, same_prior,
            coinflip_delta, p_dq_calibrated_ref, kd_dq_rate),
        "verdict": verdict,
    }


def _verdict_band(p_cal: float, naive: float, breakeven_pct: float) -> str:
    if p_cal < naive:
        return ("GREEN_gate_calibration_makes_fire_SAFER_than_0p80_at_empirical_central"
                "_conditional_on_measured_delta_below_%.1fpct" % breakeven_pct)
    return "AMBER_calibration_neutral_at_ref_read_curve_at_measured_delta"


def _self_tests(recompute_err, clean_sep, curve, grid, same_prior, coinflip,
                p_cal_ref, kd_dq_rate) -> dict:
    checks = {}
    # 1. reported Delta% reproduced from raw (public, private) TPS
    checks["delta_recompute_matches_reported"] = bool(recompute_err < 5e-4)
    # 2. clean deterministic 5% separation (the gate IS a step on realized Delta)
    checks["clean_5pct_step_separation"] = bool(clean_sep)
    # 3. every curve monotone increasing in delta_stock
    checks["curves_monotone"] = all(
        all(np.diff(np.array(c)) >= -1e-12) for c in curve.values())
    # 4. P(DQ)=0.5 exactly at delta=5% for every sigma (coin-flip is the gate)
    checks["coinflip_at_5pct"] = all(abs(v - GATE_TOL) < 1e-9 for v in coinflip.values())
    # 5. as sigma->0 the gate-noise-over-prior integral -> the naive 0.80
    checks["sharp_gate_recovers_naive_0p80"] = bool(abs(same_prior["engine_floor_0p2"] - 0.80) < 0.02)
    # 6. larger sigma => gate-noise softens P(DQ) below 0.80 over the SAME prior (monotone)
    ordered = [same_prior["engine_floor_0p2"], same_prior["construction_0p5"],
               same_prior["central_1p0"], same_prior["replicate_upper_1p5"]]
    checks["gate_noise_softens_below_naive"] = bool(
        all(ordered[i] >= ordered[i + 1] - 1e-9 for i in range(len(ordered) - 1))
        and ordered[-1] < 0.80)
    # 7. calibrated P(DQ) at the 4.3% same-family anchor is well below 0.80
    checks["calibrated_ref_below_naive"] = bool(p_cal_ref < 0.5)
    # 8. empirical replicate DQ-rate (1/3) is far below naive 0.80
    checks["empirical_replicate_below_naive"] = bool(kd_dq_rate < 0.5)
    # 9. NaN-clean
    flat = []
    for c in curve.values():
        flat.extend(c)
    checks["nan_clean"] = bool(np.all(np.isfinite(flat)) and np.all(np.isfinite(grid)))
    passed = all(checks.values())
    return {"passed": bool(passed), "n_checks": len(checks),
            "n_passed": int(sum(checks.values())), "checks": checks}


# ----------------------------------------------------------------------------
# plot + print
# ----------------------------------------------------------------------------
def _plot(out: dict, path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.array(out["calibration"]["delta_grid_pct"])
    curve = out["calibration"]["p_dq_curve"]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    styles = {"engine_floor_0p2": ("#1b7837", "sigma=0.2% (engine floor / sharp gate)"),
              "construction_0p5": ("#5aae61", "sigma=0.5% (#164 construction/shape)"),
              "central_1p0": ("#2166ac", "sigma=1.0% (central, headline)"),
              "replicate_upper_1p5": ("#b2182b", "sigma=1.5% (kenyan-duma 3-repost)")}
    for name, (col, lab) in styles.items():
        ax.plot(grid, curve[name], color=col, lw=2, label=lab)
    # naive 0.80 horizontal + the U[4,9] prior band
    ax.axhline(NAIVE_P_DQ_737, color="k", ls="--", lw=1, alpha=0.7,
               label="kanna #737 naive 0.80 (= P(U[4,9]>5%), prior tail)")
    ax.axvspan(4.0, 9.0, color="orange", alpha=0.08, label="U[4,9] documented prior")
    ax.axvline(GATE_TOL * 100, color="gray", ls=":", lw=1.2, label="gate = 5%")
    # named anchors
    for dv, txt, col in [(4.3, "#52 flagship 4.3% (VALID)", "#1b7837"),
                         (6.5, "prior center 6.5%", "#b35806"),
                         (12.4, "#44 chat-proxy 12.4%", "#b2182b")]:
        ax.axvline(dv, color=col, ls="-.", lw=0.9, alpha=0.6)
        ax.text(dv + 0.1, 0.04, txt, rotation=90, fontsize=6.5, color=col, va="bottom")
    # empirical replicate point (kenyan-duma: P(DQ)=1/3 at ~4.1%)
    rep = out["replicate_sigma_evidence"]
    ax.scatter([rep["mean_delta_pct"]], [rep["empirical_dq_rate"]], s=70, marker="D",
               color="purple", zorder=5,
               label="empirical replicate: DQ 1/3 @ %.1f%%" % rep["mean_delta_pct"])
    ax.set_xlabel("config true private drift  delta_stock  (%)")
    ax.set_ylabel("P(DQ) = P(realized private Delta > 5%)")
    ax.set_title("PR #741: G1 gate-calibrated P(DQ | delta_stock) vs kanna #737 naive 0.80")
    ax.set_xlim(0, 12); ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _print(out: dict) -> None:
    g = out["gate_model"]; d = out["empirical_distribution"]; v = out["verdict"]
    c = out["calibration"]
    print("=" * 78)
    print("PR #741 -- G1 gate-model calibration (land)")
    print("=" * 78)
    print("\n[1] GATE MODEL (pinned from public artifacts)")
    print(f"    rule: {g['rule']}")
    print(f"    single-draw={g['single_draw_not_averaged']}  private_set_N={g['private_set_size']}  "
          f"raw_tps_ratio={g['metric_is_raw_tps_ratio']}  no_token_identity={g['no_token_identity_stage']}")
    print(f"    clean 5% step: max VALID Delta={g['max_valid_delta_pct']:.2f}%  <= 5% <  "
          f"min INVALID Delta={g['min_invalid_delta_pct']:.2f}%   -> {g['clean_5pct_step_separation']}")
    print("\n[2] EMPIRICAL spec-dec realized-Delta distribution (17 itaca + #52)")
    print(f"    n={d['n_records']}  mean={d['mean_pct']:.2f}%  median={d['median_pct']:.2f}%  "
          f"std={d['std_pct']:.2f}%  range=[{d['min_pct']:.2f},{d['max_pct']:.2f}]%")
    print(f"    invalid fraction={d['invalid_fraction']:.2f} ({d['n_invalid']}/{d['n_records']})")
    fr = out["frontier_cluster"]
    print(f"    near-frontier (>={fr['tps_floor']:.0f} TPS): n={fr['n']}  median={fr['median_pct']:.2f}%  "
          f"invalid_frac={fr['invalid_fraction']:.2f}")
    r = out["replicate_sigma_evidence"]
    print(f"    same-config replicate (kenyan-duma 3x): Delta={r['realized_delta_pct']}  "
          f"std={r['std_delta_pct']:.2f}%  DQ-rate={r['empirical_dq_rate']:.2f}")
    print("\n[3] CALIBRATED P(DQ|delta_stock) at named anchors")
    for a, dv in [("flagship_52_4p3", 4.3), ("gate_5p0", 5.0),
                  ("prior_center_6p5", 6.5), ("chatproxy_44_12p4", 12.4)]:
        ap = out["calibration"]["anchor_p_dq"][a]
        print(f"    delta={dv:>5.1f}%  P(DQ): sigma0.2={ap['engine_floor_0p2']:.3f}  "
              f"sigma0.5={ap['construction_0p5']:.3f}  sigma1.0={ap['central_1p0']:.3f}  "
              f"sigma1.5={ap['replicate_upper_1p5']:.3f}")
    print(f"\n    breakeven (calibrated P(DQ)=0.80) at delta = "
          f"{ {k: round(x,2) for k,x in c['breakeven_delta_vs_naive_0p80_pct'].items()} }")
    print(f"    gate-noise over SAME U[4,9] prior: "
          f"{ {k: round(x,3) for k,x in c['gate_noise_over_U49_prior'].items()} }  (vs naive 0.80)")
    print(f"    averaging lever P(DQ@{DELTA_REF*100:.1f}%): {c['averaging_lever_p_dq_at_ref']}")
    print("\n[VERDICT]")
    print(f"    primary  p_dq_g1_calibrated = {v['primary_metric_value']}  "
          f"(at delta_stock={v['primary_reference_delta_stock_pct']:.1f}%, sigma={v['primary_reference_sigma_gate_pct']:.1f}%)")
    print(f"    test     empirical_specdec_delta_central_pct = {v['test_metric_value']}%")
    print(f"    range over sigma at ref: {v['p_dq_ref_range_over_sigma']}")
    print(f"    conditional read (sigma1.0): P(DQ)@frontier-median-4.0%={v['p_dq_at_frontier_median_4p0_sigma1p0']}  "
          f"@empirical-median-5.5%={v['p_dq_at_empirical_median_5p5_sigma1p0']}")
    print(f"    SAFER than 0.80 iff measured delta_stock < {v['fire_safer_than_0p80_if_delta_stock_below_pct']}%")
    print(f"    dominant correction: {v['dominant_correction']}")
    print(f"    band: {v['verdict_band']}")
    st = out["self_test"]
    print(f"\n    self-test: {st['n_passed']}/{st['n_checks']} passed -> {st['passed']}")
    if not st["passed"]:
        for k, val in st["checks"].items():
            if not val:
                print(f"      FAIL: {k}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_group", default="land-g1-gate-model")
    ap.add_argument("--name", default="land/g1-gate-model")
    ap.add_argument("--out", default=str(HERE / "results/g1_gate_model_calibration.json"))
    ap.add_argument("--plot", default=str(HERE / "results/g1_gate_model_calibration_curve.png"))
    args = ap.parse_args()

    out = analyze()
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    plot_path = _plot(out, args.plot)
    _print(out)
    print("\nWROTE", outp)
    print("WROTE", plot_path)

    if args.wandb:
        import wandb
        v = out["verdict"]; d = out["empirical_distribution"]; g = out["gate_model"]
        c = out["calibration"]; r = out["replicate_sigma_evidence"]
        run = wandb.init(
            project="gemma-challenge-senpai", entity="wandb-applied-ai-team",
            group=args.wandb_group, name=args.name, job_type="analysis",
            config={
                "pr": 741, "student": "land",
                "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
                "binding_gate": "G1_5pct_private_TPS_reproduction",
                "gate_tolerance_pct": GATE_TOL * 100,
                "gate_single_draw": int(GATE_IS_SINGLE_DRAW),
                "private_set_size": N_PRIV,
                "reference_delta_stock_pct": DELTA_REF * 100,
                "central_sigma_gate_pct": SIGMA_CENTRAL * 100,
                "candidate": "int4_mtp_batchinv un-rescued stock-Hub drafter K=6 (#730 fire)",
                "naive_737_p_dq": NAIVE_P_DQ_737,
            },
            tags=["pr741", "land", "analysis_only", "g1-gate-model", "730-fire",
                  "p-dq-calibration", "private-repro-gate", "gate-side-complement-737"],
        )
        summary = {
            "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
            "primary_metric_name": v["primary_metric_name"],
            "primary_metric_value": v["primary_metric_value"],
            "test_metric_name": v["test_metric_name"],
            "test_metric_value": v["test_metric_value"],
            "p_dq_g1_calibrated": v["primary_metric_value"],
            "empirical_specdec_delta_central_pct": v["test_metric_value"],
            # gate model
            "gate_clean_5pct_step": int(g["clean_5pct_step_separation"]),
            "gate_max_valid_delta_pct": g["max_valid_delta_pct"],
            "gate_min_invalid_delta_pct": g["min_invalid_delta_pct"],
            "gate_no_token_identity": int(g["no_token_identity_stage"]),
            # empirical distribution
            "empirical_mean_pct": d["mean_pct"],
            "empirical_median_pct": d["median_pct"],
            "empirical_std_pct": d["std_pct"],
            "empirical_invalid_fraction": d["invalid_fraction"],
            "frontier_median_pct": out["frontier_cluster"]["median_pct"],
            # sigma evidence / replicate
            "replicate_std_delta_pct": r["std_delta_pct"],
            "replicate_empirical_dq_rate": r["empirical_dq_rate"],
            # the headline comparison
            "naive_737_p_dq": NAIVE_P_DQ_737,
            "calibrated_at_4p3_vs_naive": v["calibrated_at_ref_4p3_lt_naive_0p80"],
            "p_dq_ref_sigma0p5": v["p_dq_ref_range_over_sigma"]["construction_0p5"],
            "p_dq_ref_sigma1p0": v["p_dq_ref_range_over_sigma"]["central_1p0"],
            "p_dq_ref_sigma1p5": v["p_dq_ref_range_over_sigma"]["replicate_upper_1p5"],
            "fire_safer_than_0p80_if_delta_below_pct": v["fire_safer_than_0p80_if_delta_stock_below_pct"],
            "gate_noise_over_U49_central_sigma1p0": c["gate_noise_over_U49_prior"]["central_1p0"],
            # anchors
            "anchor_pdq_prior_center_6p5_sigma1p0": out["calibration"]["anchor_p_dq"]["prior_center_6p5"]["central_1p0"],
            "anchor_pdq_chatproxy_12p4_sigma1p0": out["calibration"]["anchor_p_dq"]["chatproxy_44_12p4"]["central_1p0"],
            "p_dq_at_empirical_median_sigma1p0": v["p_dq_at_empirical_median_5p5_sigma1p0"],
            "p_dq_at_frontier_median_sigma1p0": v["p_dq_at_frontier_median_4p0_sigma1p0"],
            "verdict_band": v["verdict_band"],
            "fire_now": v["fire_now"],
            "self_test_passed": int(out["self_test"]["passed"]),
            "self_test_n_passed": out["self_test"]["n_passed"],
        }
        run.summary.update(summary)
        wandb.log(summary)
        try:
            wandb.log({"p_dq_vs_delta_stock_curve": wandb.Image(plot_path)})
        except Exception as e:
            print("plot-log skipped:", e)
        print("WANDB_RUN_ID", run.id)
        print("WANDB_RUN_URL", run.url)
        wandb.finish()


if __name__ == "__main__":
    main()
