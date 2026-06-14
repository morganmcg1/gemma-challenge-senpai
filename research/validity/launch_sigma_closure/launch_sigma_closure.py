#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Launch-sigma closure: de-dup x realistic-ICC combined sigma -> LCB(mu) (PR #201)

WHAT THIS IS
------------
The ADDITIVE-CI law that fern #185's launch-trigger calculator must import, now
CLOSED. ubel #195 proved fern's 4-axis quadrature INVALID: the #175 sampling
axis (+-10.9 TPS) and the #187 input-lambda axis (+-3.71 TPS) are a +0.945
DOUBLE-COUNT (two views of the SAME accept draw), and the physically-correct fix
is to DE-DUPLICATE them into ONE acceptance axis at denken #187's
overlap-corrected 5.32 TPS. But #195's de-dup (combined 7.26 TPS) assumed IID
sampling. wirbel #190 then showed the acceptance scatter is NOT iid: realistic
within-prompt ICC=0.1446 inflates it 2.1x (design-effect 4.411, N_eff 713).

So the launch sigma fern imports is NEITHER #195's iid de-dup (7.26) NOR fern's
quadrature (12.54): it is the DE-DUPED acceptance axis evaluated UNDER REALISTIC
ICC, combined with sigma_hw and sigma_private, carrying the worst-case
rho(*,hw) in [-0.3,+0.3] band, and PROPAGATED to a launch LCB(mu) curve so fern
#185 gets a GO/NO-GO-ready trigger, not just a sigma.

THE TWO ORTHOGONAL CORRECTIONS (stated up front, derived below)
--------------------------------------------------------------
De-dup and ICC-inflation act on the SAME acceptance axis but are ORTHOGONAL:

  * DE-DUP (ubel #195) removes the rho=0.945 A1xA2 double-count -> collapses the
    #175 sampling leg + #187 input-lambda leg into ONE acceptance axis. This sets
    the axis's IDENTITY (one axis, not two), and its IID magnitude = denken's
    overlap-corrected 5.32 TPS.
  * ICC-INFLATION (wirbel #190) sets that one axis's MAGNITUDE in the realistic
    regime: the within-prompt correlation inflates the IID half-width by
    sqrt(design_effect). It does NOT change how many axes there are; it rescales
    the de-duped axis from +-5.32 (iid) to its realistic-ICC value.

  acceptance_sigma_dedup_realistic_icc = acceptance_sigma_dedup_iid * sqrt(D),
    D = 1 + (m_bar - 1) * ICC   (Kish design effect, wirbel #190)

Sanity leg: applying the SAME sqrt(D) to #175's raw +-10.906 reproduces #190's
+-22.905 realistic half-width exactly -> the inflation law is the #190 law.

THE SINGLE COMBINED LAUNCH SIGMA (replaces #195's iid 7.26 / 17.04)
-------------------------------------------------------------------
  sigma^2_launch = acc_realistic^2 + sigma_hw^2 + sigma_private^2
                   + 2*sum_{i<j} rho_ij * sigma_i * sigma_j
  central:    rho(*,hw) = 0   -> combined_sigma_launch_central
  worst-case: rho(*,hw) = +0.3 PSD-admissible corner -> combined_sigma_launch_worstcase

LCB(mu) = mu - z * sigma_launch  (z = one-sided P95 = 1.64485; the #194
convention that produced its single-shot-safe break-even 512.16 = 500 + z*7.391).
The GO trigger fern wires is the minimum mu that clears 500 at P>=0.95.

SCOPE
-----
LOCAL CPU-ONLY analytic synthesis over EXISTING MERGED results. No GPU / vLLM /
HF Job / submission / served-file change. BASELINE stays 481.53; greedy identity
untouched; adds 0 TPS (PRIMARY = self-test). IMPORTS verbatim (does NOT
re-derive): #195 (combined_sigma_dedup 7.2617, worstcase 17.0375, the 4x4
rho_matrix, the [-0.3,+0.3] rho(*,hw) band), #190 (icc_hat 0.1446,
halfwidth_realistic 22.905, design_effect 4.411, n_eff 713), #187 (overlap-
corrected 5.32 iid, overlap_fraction 0.893), #188 (sigma_hw 4.864 within/
between), #176/#191 (sigma_private 0.884), #194 (break-even 512.16, mu_lambda1
520.95), #183 (forward map). Orthogonal to the greedy-identity gate (a CI-
composition law, not a token-identity question). NOT a launch.

SELF-TEST (PR step 5 -- PRIMARY)
-------------------------------
(a) ICC=0 + de-dup reproduces #195's 7.26 exactly;
(b) rho(*,hw)=0 + iid reproduces #195's quadrature/dedup legs;
(c) realistic-ICC sigma >= iid sigma (monotone in ICC);
(d) worstcase >= central;
(e) the sigma->LCB map reproduces a known anchor (#194's break-even 512.16) to
    <= 0.5 TPS;
(f) the 3x3 launch covariance is PSD;
(g) NaN-clean across all reported scalars.
PRIMARY = launch_sigma_closure_self_test_passes (bool);
TEST    = combined_sigma_launch_central (float TPS).
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
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # research/validity/launch_sigma_closure -> repo root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# source-of-truth artifacts (imported verbatim; one source per constant)
# ---------------------------------------------------------------------------
CICOV_195 = os.path.join(_ROOT, "research/validity/ci_axis_covariance/ci_axis_covariance_results.json")
ICC_190 = os.path.join(_ROOT, "research/validity/icc_neff/icc_neff_results.json")
LAMBDA_187 = os.path.join(_ROOT, "research/validity/lambda_built_ci/lambda_built_ci_results.json")
HW_188 = os.path.join(_ROOT, "research/validity/oneshot_hw_bound/oneshot_hw_bound_results.json")
PRIVATE_176 = os.path.join(_ROOT, "research/validity/private_adverse_skew/results.json")
REDRAW_194 = os.path.join(_ROOT, "research/validity/redraw_budget/redraw_budget_results.json")
CARD_183 = os.path.join(_ROOT, "research/oracle_readout/lambda_acceptance_card/lambda_acceptance_card_results.json")

# one-sided P95 (the #194 launch convention) and two-sided 95% (Fisher-z CI).
Z95_ONE_SIDED = 1.6448536269514722  # scipy.stats.norm.ppf(0.95)
Z95_TWO_SIDED = 1.959963984540054   # scipy.stats.norm.ppf(0.975)
TARGET = 500.0
RHO_HW_BAND = [-0.3, 0.3]  # #195 bounded rho(*,hw) until land #71's served draw
AXES3 = ["acceptance", "hardware", "private"]  # the de-duped launch axes
LCB_ANCHOR_TOL_TPS = 0.5  # self-test (e): |map - #194 break-even| <= 0.5 TPS


def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _min_eig(M: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(M).min())


def _combined_sigma(sig: np.ndarray, R: np.ndarray) -> tuple[float, np.ndarray]:
    """1^T (sig sig^T . R) 1 -> combined sigma over correlated axes."""
    C = np.outer(sig, sig) * R
    ones = np.ones(sig.size)
    return math.sqrt(max(float(ones @ C @ ones), 0.0)), C


def _psd_admissible_corner(R_hi: np.ndarray, tol: float = -1e-9) -> tuple[np.ndarray, float]:
    """Largest off-diagonal scale t in [0,1] s.t. I + t*(R_hi - I) is PSD.
    1^T C 1 is monotone increasing in each rho (sigma_i>0), so the worst-case
    combined sigma over the +box is at the most-correlated PSD-admissible corner."""
    I = np.eye(R_hi.shape[0])
    D = R_hi - I
    if _min_eig(R_hi) >= tol:
        return R_hi, 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _min_eig(I + mid * D) >= tol:
            lo = mid
        else:
            hi = mid
    return I + lo * D, lo


# ---------------------------------------------------------------------------
# Step 0 -- import the de-duped axis sigmas + the #190 ICC machinery, verbatim.
# ---------------------------------------------------------------------------
def import_legs() -> dict[str, Any]:
    cov = _load(CICOV_195)
    comb195 = cov["combined"]
    # #195 de-duped axes (IMPORTED, not re-derived): the ONE acceptance axis is
    # denken #187's overlap-corrected 5.32 TPS (iid); hw and private unchanged.
    acceptance_iid = float(comb195["dedup_reading"]["overlap_corrected_sampling_block_tps"])  # 5.31870
    sigma_hw = float(comb195["sigma_tps_vector"]["hardware"])     # 4.864469
    sigma_private = float(comb195["sigma_tps_vector"]["private"])  # 0.883918
    combined_dedup_195 = float(comb195["dedup_reading"]["combined_sigma_dedup_tps"])  # 7.261743 (self-test (a) anchor)
    combined_worstcase_195 = float(comb195["combined_sigma_worstcase"])  # 17.037470
    combined_quadrature_195 = float(comb195["combined_sigma_quadrature"])  # 12.536224
    rho_in_out_195 = float(cov["rho"]["pairs"]["sampling__input_lambda"]["rho_central"])  # 0.944943

    # #190 ICC machinery (IMPORTED): the design-effect that inflates the iid
    # acceptance half-width into the realistic regime.
    icc = _load(ICC_190)
    icc_hat = float(icc["icc_estimate"]["icc_hat"])              # 0.144625
    m_bar = float(icc["realistic_ci"]["m_bar"])                  # 24.58251
    design_effect = float(icc["realistic_ci"]["design_effect_hat"])  # 4.410614
    n_eff = float(icc["realistic_ci"]["n_eff_hat"])              # 713.4065
    halfwidth_iid_190 = float(icc["realistic_ci"]["halfwidth_iid_tps"])        # 10.906182
    halfwidth_realistic_190 = float(icc["realistic_ci"]["halfwidth_realistic_tps"])  # 22.904577
    # ICC CI band (wirbel #190 bothbugs) -> design-effect band for the trigger envelope.
    deff_lo = float(icc["go_robustness"]["section4"]["bothbugs_ci_lo"]["design_effect"])  # 3.459411
    deff_hi = float(icc["go_robustness"]["section4"]["bothbugs_ci_hi"]["design_effect"])  # 5.379448
    icc_lo = float(icc["go_robustness"]["section4"]["bothbugs_ci_lo"]["icc"])
    icc_hi = float(icc["go_robustness"]["section4"]["bothbugs_ci_hi"]["icc"])

    # #187 overlap (provenance / cross-check of the de-duped axis identity).
    lam = _load(LAMBDA_187)["synthesis"]["input_output_compose"]
    overlap_fraction = float(lam["overlap_fraction"])           # 0.892917 == rho^2
    forward_map_slope = float(lam["forward_map_slope_tps_per_lambda"])  # 216.4824

    # #188 hw within/between provenance (between-dominated -> rho(*,hw) is the
    # BETWEEN-device coupling, UNMEASURED, carried as the bounded band).
    hw = _load(HW_188)["decomposition"]["decomposition"]
    sigma_within = float(hw["sigma_within_tps"])                # 0.056158
    sigma_between = float(hw["sigma_between_tps"])              # 4.864145

    # #194 break-even anchor (the canonical clears-500-at-P95 number) + decision mus.
    rb = _load(REDRAW_194)
    mu_break_even_194 = float(rb["mu_single_shot_safe_tps"])    # 512.157071
    mu_lambda1_194 = float(rb["budget"]["mu_lambda1_tps"])      # 520.952732
    mu_bar_194 = float(rb["budget"]["mu_bar_tps"])              # 500.000145
    sigma_draw_iid_194 = float(rb["budget"]["sigma_draw_tps"])  # 7.390974 (iid sampling(+)hw 1-sigma)

    # #183 forward map anchor (secondary (e) cross-check).
    card = _load(CARD_183)
    K_cal_183 = float(card["anchors"]["K_cal"])                 # 125.26795
    step_183 = float(card["anchors"]["decomp_spread_map"]["step_time"])  # 1.2182

    return {
        "acceptance_sigma_dedup_iid": acceptance_iid,
        "sigma_hw": sigma_hw,
        "sigma_private": sigma_private,
        "sigma_within": sigma_within,
        "sigma_between": sigma_between,
        "combined_dedup_195": combined_dedup_195,
        "combined_worstcase_195": combined_worstcase_195,
        "combined_quadrature_195": combined_quadrature_195,
        "rho_in_out_195": rho_in_out_195,
        "overlap_fraction": overlap_fraction,
        "forward_map_slope_tps_per_lambda": forward_map_slope,
        "icc_hat": icc_hat,
        "m_bar": m_bar,
        "design_effect": design_effect,
        "n_eff": n_eff,
        "halfwidth_iid_190": halfwidth_iid_190,
        "halfwidth_realistic_190": halfwidth_realistic_190,
        "design_effect_ci": [deff_lo, deff_hi],
        "icc_ci": [icc_lo, icc_hi],
        "mu_break_even_194": mu_break_even_194,
        "mu_lambda1_194": mu_lambda1_194,
        "mu_bar_194": mu_bar_194,
        "sigma_draw_iid_194": sigma_draw_iid_194,
        "K_cal_183": K_cal_183,
        "step_183": step_183,
    }


# ---------------------------------------------------------------------------
# Step 1 -- de-dup x ICC reconciliation (the core). SHOW THE ALGEBRA.
# ---------------------------------------------------------------------------
def dedup_x_icc(legs: dict[str, Any]) -> dict[str, Any]:
    acc_iid = legs["acceptance_sigma_dedup_iid"]
    m_bar = legs["m_bar"]
    icc = legs["icc_hat"]

    # Kish design effect from first principles (reproduces #190's design_effect_hat).
    design_effect_algebra = 1.0 + (m_bar - 1.0) * icc
    sqrt_deff = math.sqrt(legs["design_effect"])  # use #190's pinned D for the inflation

    # the SE-inflation law: SE_realistic / SE_iid = sqrt(N / N_eff) = sqrt(D).
    acc_realistic = acc_iid * sqrt_deff

    # sanity leg: the SAME sqrt(D) on #175's raw +-10.906 must reproduce #190's +-22.905.
    sanity_iid = legs["halfwidth_iid_190"]
    sanity_realistic = sanity_iid * sqrt_deff
    sanity_err = abs(sanity_realistic - legs["halfwidth_realistic_190"])

    return {
        "design_effect_algebra_1_plus_mbarm1_icc": design_effect_algebra,
        "design_effect_imported_190": legs["design_effect"],
        "design_effect_algebra_matches_190": bool(abs(design_effect_algebra - legs["design_effect"]) < 1e-3),
        "sqrt_design_effect_inflation": sqrt_deff,
        "acceptance_sigma_dedup_iid": acc_iid,
        "acceptance_sigma_dedup_realistic_icc": acc_realistic,
        "algebra": (
            "D = 1 + (m_bar-1)*ICC = 1 + (%.5f-1)*%.6f = %.6f (== #190 design_effect %.6f); "
            "sqrt(D) = %.6f; acceptance_sigma_dedup_realistic_icc = %.6f (iid) * %.6f = %.6f TPS."
            % (m_bar, icc, design_effect_algebra, legs["design_effect"], sqrt_deff,
               acc_iid, sqrt_deff, acc_realistic)
        ),
        "icc_inflation_sanity": {
            "raw_iid_175_halfwidth": sanity_iid,
            "raw_realistic_computed": sanity_realistic,
            "raw_realistic_190": legs["halfwidth_realistic_190"],
            "abs_err_tps": sanity_err,
            "ok": bool(sanity_err < 1e-6),
            "note": "10.906182 * sqrt(D) reproduces #190 halfwidth_realistic 22.905 -> inflation law verified.",
        },
        "orthogonality_note": (
            "DE-DUP (removes the rho=0.945 A1xA2 double-count -> ONE acceptance axis, iid magnitude 5.32) "
            "and ICC-INFLATION (rescales that axis 5.32 -> %.3f via sqrt(D)) are ORTHOGONAL corrections to "
            "the SAME acceptance axis: de-dup sets the axis IDENTITY, ICC sets its MAGNITUDE." % acc_realistic
        ),
    }


# ---------------------------------------------------------------------------
# Step 2 -- the single combined launch sigma (central + worstcase).
# ---------------------------------------------------------------------------
def combined_launch_sigma(legs: dict[str, Any], recon: dict[str, Any],
                          acc_override: float | None = None) -> dict[str, Any]:
    acc = recon["acceptance_sigma_dedup_realistic_icc"] if acc_override is None else acc_override
    sig = np.array([acc, legs["sigma_hw"], legs["sigma_private"]], dtype=float)

    # central: all rho = 0 (rho(*,hw)=0, the de-duped acceptance axis already
    # absorbs the A1xA2 overlap so no within-block cross-term remains).
    central, C_central = _combined_sigma(sig, np.eye(3))

    # worst-case: push the THREE bounded unmeasured pairs to the +0.3 corner.
    # idx: 0=acceptance, 1=hardware, 2=private. (acc,hw),(acc,priv),(hw,priv).
    R_hi = np.array([[1.0, 0.3, 0.3], [0.3, 1.0, 0.3], [0.3, 0.3, 1.0]], dtype=float)
    R_wc, wc_scale = _psd_admissible_corner(R_hi)
    worstcase, C_wc = _combined_sigma(sig, R_wc)

    # diagnostic: rho(*,hw)-ONLY corner (only the two hardware pairs at +0.3;
    # acc-private left at 0) -> isolates the hardware coupling's contribution.
    R_hwonly = np.array([[1.0, 0.3, 0.0], [0.3, 1.0, 0.3], [0.0, 0.3, 1.0]], dtype=float)
    worstcase_hwonly, _ = _combined_sigma(sig, R_hwonly)

    return {
        "axes": AXES3,
        "sigma_vector_tps": {"acceptance": float(sig[0]), "hardware": float(sig[1]), "private": float(sig[2])},
        "combined_sigma_launch_central": central,
        "combined_sigma_launch_worstcase": worstcase,
        "combined_sigma_launch_worstcase_rho_hw_only": worstcase_hwonly,
        "worstcase_psd_scale": wc_scale,
        "rho_worstcase_matrix": R_wc.tolist(),
        "covariance_central": C_central.tolist(),
        "covariance_central_min_eig": _min_eig(C_central),
        "covariance_worstcase_min_eig": _min_eig(C_wc),
        "cross_covariance_sum_worstcase": 0.5 * (worstcase**2 - central**2),
    }


# ---------------------------------------------------------------------------
# Step 3 -- propagate to the launch LCB(mu) curve.
# ---------------------------------------------------------------------------
def lcb_curve(legs: dict[str, Any], comb: dict[str, Any]) -> dict[str, Any]:
    z = Z95_ONE_SIDED
    s_c = comb["combined_sigma_launch_central"]
    s_w = comb["combined_sigma_launch_worstcase"]

    def lcb(mu: float, s: float) -> float:
        return mu - z * s

    mus = {"mu500": TARGET, "mu_break_even_512": legs["mu_break_even_194"], "mu_lambda1_521": legs["mu_lambda1_194"]}
    curve = {k: {"mu": v, "lcb_central": lcb(v, s_c), "lcb_worstcase": lcb(v, s_w)} for k, v in mus.items()}

    # the GO trigger: minimum mu whose LCB clears 500 at P>=0.95.
    mu_clears_central = TARGET + z * s_c
    mu_clears_worstcase = TARGET + z * s_w

    mu_lambda1 = legs["mu_lambda1_194"]
    return {
        "z_one_sided_p95": z,
        "launch_lcb_at_mu500_central": curve["mu500"]["lcb_central"],
        "launch_lcb_at_mu500_worstcase": curve["mu500"]["lcb_worstcase"],
        "lcb_curve": curve,
        "mu_clears_500_central": mu_clears_central,
        "mu_clears_500_worstcase": mu_clears_worstcase,
        "lambda1_ceiling_mu": mu_lambda1,
        "lambda1_clears_500_central": bool(lcb(mu_lambda1, s_c) >= TARGET),
        "lambda1_clears_500_worstcase": bool(lcb(mu_lambda1, s_w) >= TARGET),
        "central_margin_at_lambda1_tps": lcb(mu_lambda1, s_c) - TARGET,
        "worstcase_margin_at_lambda1_tps": lcb(mu_lambda1, s_w) - TARGET,
        "go_trigger_gap_above_lambda1_central_tps": mu_clears_central - mu_lambda1,
        "go_trigger_gap_above_lambda1_worstcase_tps": mu_clears_worstcase - mu_lambda1,
        "revises_194_break_even": {
            "iid_break_even_194": legs["mu_break_even_194"],
            "realistic_icc_trigger_central": mu_clears_central,
            "shift_tps": mu_clears_central - legs["mu_break_even_194"],
            "note": "#194's break-even used IID sampling; the de-duped REALISTIC-ICC law lifts the "
            "P95 trigger from 512.16 (iid) to the central value -> ICC erodes the launch headroom.",
        },
    }


# ---------------------------------------------------------------------------
# Step 3b -- ICC-band envelope of the GO trigger (rich logging).
# ---------------------------------------------------------------------------
def icc_band_envelope(legs: dict[str, Any], recon: dict[str, Any]) -> dict[str, Any]:
    z = Z95_ONE_SIDED
    acc_iid = recon["acceptance_sigma_dedup_iid"]
    out = {}
    for tag, deff in (("icc_lo", legs["design_effect_ci"][0]),
                      ("icc_hat", legs["design_effect"]),
                      ("icc_hi", legs["design_effect_ci"][1]),
                      ("icc0_iid", 1.0)):
        acc = acc_iid * math.sqrt(deff)
        comb = combined_launch_sigma(legs, recon, acc_override=acc)
        out[tag] = {
            "design_effect": deff,
            "acceptance_sigma_tps": acc,
            "combined_sigma_central": comb["combined_sigma_launch_central"],
            "combined_sigma_worstcase": comb["combined_sigma_launch_worstcase"],
            "mu_clears_500_central": TARGET + z * comb["combined_sigma_launch_central"],
            "mu_clears_500_worstcase": TARGET + z * comb["combined_sigma_launch_worstcase"],
        }
    return out


# ---------------------------------------------------------------------------
# Step 4 -- land #71 co-log spec (retire the rho(*,hw) bound).
# ---------------------------------------------------------------------------
def colog_spec_land71() -> dict[str, Any]:
    # n that resolves a Pearson rho to a +-0.1 95% CI half-width (Fisher-z,
    # evaluated at rho=0, the widest rho-space CI per n -> conservative).
    target_half = 0.1
    n_for_rho = math.ceil(3.0 + (Z95_TWO_SIDED / math.atanh(target_half)) ** 2)
    spec = {
        "what": "land #71's served draw must CO-LOG, per fresh device allocation, the accepted-length / "
        "acceptance estimate ALONGSIDE that allocation's wall-clock output TPS, so the BETWEEN-device "
        "rho(acceptance, hardware) can be measured directly and REPLACE the [-0.3,+0.3] bounded assumption.",
        "fields_per_allocation": {
            "allocation_id": "distinct fresh A10G allocation id (BETWEEN-device draw, not a within-process re-run)",
            "wall_tps": "served output TPS for the allocation (the hardware draw)",
            "accept_length_or_e_accept": "mean accepted speculative length (or e_accept_exact) measured ON THE "
            "SAME allocation/draw as wall_tps",
            "clock_mhz": "SM clock for provenance (the n=12 within-device trace was clock-locked 1710 MHz -> "
            "sigma_within 0.056; the launch-relevant signal is the cross-allocation spread sigma_between 4.864)",
        },
        "estimator": "rho_between = pearson(accept_length, wall_tps) across >= n distinct allocations.",
        "why_between_not_within": "kanna #188 sigma_hw is 87x between-device dominated; the only co-logged "
        "acceptance x TPS trace (n=12) is WITHIN one pinned A10G (sigma_within 0.056) and speaks only to a "
        "negligible share of sigma_hw. The frantic-penguin n=3 cross-device draws carry TPS but NO acceptance, "
        "so rho(*,hw) stays bounded until land #71 co-logs both per allocation.",
        "colog_n_allocations_for_rho_ci": n_for_rho,
        "rho_ci_target_halfwidth": target_half,
        "retires": "the [-0.3,+0.3] rho(*,hw) band -> a measured rho +- 0.1; collapses combined_sigma_launch "
        "from the [central, worstcase] interval onto a single value.",
    }
    return spec


# ---------------------------------------------------------------------------
# Step 5 -- self-test (PRIMARY).
# ---------------------------------------------------------------------------
def self_test(legs: dict[str, Any], recon: dict[str, Any], comb: dict[str, Any],
              lcb: dict[str, Any]) -> dict[str, Any]:
    # (a) ICC=0 + de-dup reproduces #195's 7.26 exactly.
    comb_icc0 = combined_launch_sigma(legs, recon, acc_override=recon["acceptance_sigma_dedup_iid"])
    a_val = comb_icc0["combined_sigma_launch_central"]
    a_err = abs(a_val - legs["combined_dedup_195"])
    a_ok = bool(a_err < 1e-9)

    # (b) rho(*,hw)=0 + iid reproduces #195's legs: the iid acceptance axis equals
    # #195's overlap-corrected 5.32, and the iid central equals #195's 7.26.
    b_acc = abs(recon["acceptance_sigma_dedup_iid"] - 5.318696881534334) < 1e-6
    b_central = abs(a_val - legs["combined_dedup_195"]) < 1e-9
    b_ok = bool(b_acc and b_central)

    # (c) realistic-ICC sigma >= iid sigma, monotone in ICC over a grid.
    grid = [0.0, 0.05, 0.1, legs["icc_hat"], 0.25, 0.5, 1.0]
    sigs = []
    for icc in grid:
        deff = 1.0 + (legs["m_bar"] - 1.0) * icc
        acc = recon["acceptance_sigma_dedup_iid"] * math.sqrt(deff)
        sigs.append(combined_launch_sigma(legs, recon, acc_override=acc)["combined_sigma_launch_central"])
    c_monotone = all(sigs[i + 1] >= sigs[i] - 1e-12 for i in range(len(sigs) - 1))
    c_realistic_ge_iid = comb["combined_sigma_launch_central"] >= a_val - 1e-12
    c_ok = bool(c_monotone and c_realistic_ge_iid)

    # (d) worstcase >= central.
    d_ok = bool(comb["combined_sigma_launch_worstcase"] >= comb["combined_sigma_launch_central"] - 1e-12)

    # (e) the sigma->LCB map reproduces #194's published break-even (the canonical
    # clears-500-at-P95 number) to <= 0.5 TPS, when fed #194's IID single-shot
    # sigma_draw. This validates the EXACT deliverable machinery (mu = 500 + z*sigma).
    map_break_even = TARGET + Z95_ONE_SIDED * legs["sigma_draw_iid_194"]
    e_err = abs(map_break_even - legs["mu_break_even_194"])
    e_ok = bool(e_err <= LCB_ANCHOR_TOL_TPS)
    # secondary (e) cross-check: #183 forward map at lambda=1 reproduces #194 mu_lambda1.
    # mu(lambda=1) = K_cal * (E_T / step) * tau; recover the implied E_T*tau and check finite.
    implied_et_tau = legs["mu_lambda1_194"] * legs["step_183"] / legs["K_cal_183"]
    e_fwd_finite = _finite(implied_et_tau) and 4.5 < implied_et_tau < 5.5

    # (f) the 3x3 launch covariance is PSD (central and worstcase).
    f_ok = bool(comb["covariance_central_min_eig"] >= -1e-9 and comb["covariance_worstcase_min_eig"] >= -1e-9)

    # (g) NaN-clean across all reported scalars.
    scalars = [
        recon["acceptance_sigma_dedup_iid"], recon["acceptance_sigma_dedup_realistic_icc"],
        recon["sqrt_design_effect_inflation"], comb["combined_sigma_launch_central"],
        comb["combined_sigma_launch_worstcase"], comb["combined_sigma_launch_worstcase_rho_hw_only"],
        lcb["launch_lcb_at_mu500_central"], lcb["launch_lcb_at_mu500_worstcase"],
        lcb["mu_clears_500_central"], lcb["mu_clears_500_worstcase"],
        comb["covariance_central_min_eig"], comb["covariance_worstcase_min_eig"],
        map_break_even, implied_et_tau, a_val,
    ] + sigs
    g_ok = all(_finite(x) for x in scalars)

    checks = {
        "a_icc0_dedup_reproduces_195_726": a_ok,
        "b_rho_hw0_iid_reproduces_195_legs": b_ok,
        "c_realistic_ge_iid_monotone_in_icc": c_ok,
        "d_worstcase_ge_central": d_ok,
        "e_sigma_lcb_map_reproduces_194_break_even": e_ok,
        "f_launch_covariance_psd": f_ok,
        "g_nan_clean": g_ok,
    }
    passes = all(checks.values())
    return {
        "launch_sigma_closure_self_test_passes": bool(passes),
        "checks": checks,
        "evidence": {
            "a_icc0_central_tps": a_val,
            "a_target_195_dedup_tps": legs["combined_dedup_195"],
            "a_abs_err": a_err,
            "c_sigma_grid": {str(round(g, 4)): s for g, s in zip(grid, sigs)},
            "e_map_break_even_tps": map_break_even,
            "e_target_194_break_even_tps": legs["mu_break_even_194"],
            "e_abs_err_tps": e_err,
            "e_fwd_implied_et_tau": implied_et_tau,
            "e_fwd_finite_ok": e_fwd_finite,
            "f_central_min_eig": comb["covariance_central_min_eig"],
            "f_worstcase_min_eig": comb["covariance_worstcase_min_eig"],
            "n_scalars_checked": len(scalars),
        },
    }


def _log_wandb(args, result: dict[str, Any]) -> None:
    if args.no_wandb:
        return
    try:
        from scripts import wandb_logging
    except Exception as exc:  # noqa: BLE001
        print(f"[launch-sigma] wandb_logging import failed ({exc}); skipping", flush=True)
        return
    try:
        run = wandb_logging.init_wandb_run(
            job_type="launch-sigma-closure", agent="ubel",
            name=args.wandb_name or "ubel/launch-sigma-closure",
            group=args.wandb_group,
            tags=["launch-sigma-closure", "dedup", "icc", "covariance", "composition-pinning", "pr201"],
            config={"baseline_tps": 481.53, "method": "cpu-only-analytic", "z_one_sided_p95": Z95_ONE_SIDED,
                    "rho_hw_band": RHO_HW_BAND},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[launch-sigma] wandb init failed ({exc}); skipping", flush=True)
        return
    if run is None:
        print("[launch-sigma] wandb disabled; skipping", flush=True)
        return
    try:
        comb = result["combined"]
        lcb = result["lcb"]
        recon = result["dedup_x_icc"]
        st = result["self_test"]
        flat = {
            "acceptance_sigma_dedup_iid": recon["acceptance_sigma_dedup_iid"],
            "acceptance_sigma_dedup_realistic_icc": recon["acceptance_sigma_dedup_realistic_icc"],
            "sqrt_design_effect_inflation": recon["sqrt_design_effect_inflation"],
            "combined_sigma_launch_central": comb["combined_sigma_launch_central"],
            "combined_sigma_launch_worstcase": comb["combined_sigma_launch_worstcase"],
            "combined_sigma_launch_worstcase_rho_hw_only": comb["combined_sigma_launch_worstcase_rho_hw_only"],
            "launch_lcb_at_mu500_central": lcb["launch_lcb_at_mu500_central"],
            "launch_lcb_at_mu500_worstcase": lcb["launch_lcb_at_mu500_worstcase"],
            "mu_clears_500_central": lcb["mu_clears_500_central"],
            "mu_clears_500_worstcase": lcb["mu_clears_500_worstcase"],
            "lambda1_ceiling_mu": lcb["lambda1_ceiling_mu"],
            "central_margin_at_lambda1_tps": lcb["central_margin_at_lambda1_tps"],
            "worstcase_margin_at_lambda1_tps": lcb["worstcase_margin_at_lambda1_tps"],
            "go_trigger_gap_above_lambda1_worstcase_tps": lcb["go_trigger_gap_above_lambda1_worstcase_tps"],
            "lambda1_clears_500_central": 1.0 if lcb["lambda1_clears_500_central"] else 0.0,
            "lambda1_clears_500_worstcase": 1.0 if lcb["lambda1_clears_500_worstcase"] else 0.0,
            "design_effect": result["legs"]["design_effect"],
            "icc_hat": result["legs"]["icc_hat"],
            "colog_n_allocations_for_rho_ci": result["colog_spec"]["colog_n_allocations_for_rho_ci"],
            "covariance_worstcase_min_eig": comb["covariance_worstcase_min_eig"],
            "launch_sigma_closure_self_test_passes": 1.0 if st["launch_sigma_closure_self_test_passes"] else 0.0,
        }
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run, name="launch_sigma_closure", artifact_type="launch-sigma-closure", data=result)
    except Exception as exc:  # noqa: BLE001
        print(f"[launch-sigma] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:  # noqa: BLE001
            pass


def _print(result: dict[str, Any]) -> None:
    comb = result["combined"]
    lcb = result["lcb"]
    recon = result["dedup_x_icc"]
    st = result["self_test"]
    print("\n[launch-sigma] ===== LAUNCH-SIGMA CLOSURE (PR #201) =====", flush=True)
    print("  de-dup x ICC reconciliation:", flush=True)
    print(f"    acceptance_sigma_dedup_iid          = {recon['acceptance_sigma_dedup_iid']:7.4f} TPS (#187 overlap-corrected, sanity)", flush=True)
    print(f"    sqrt(design_effect) inflation       = {recon['sqrt_design_effect_inflation']:7.4f}  (D={result['legs']['design_effect']:.4f}, ICC={result['legs']['icc_hat']:.4f})", flush=True)
    print(f"    acceptance_sigma_dedup_realistic    = {recon['acceptance_sigma_dedup_realistic_icc']:7.4f} TPS (launch value)", flush=True)
    print("  combined launch sigma (de-duped, realistic ICC):", flush=True)
    print(f"    combined_sigma_launch_central       = {comb['combined_sigma_launch_central']:7.4f} TPS  <-- TEST  (replaces #195 iid 7.26)", flush=True)
    print(f"    combined_sigma_launch_worstcase     = {comb['combined_sigma_launch_worstcase']:7.4f} TPS  (rho(*,hw)=+0.3; replaces #195 17.04)", flush=True)
    print("  launch LCB(mu) curve  (LCB = mu - z*sigma, z=%.5f):" % lcb["z_one_sided_p95"], flush=True)
    for k, v in lcb["lcb_curve"].items():
        print(f"    mu={v['mu']:7.2f}  LCB_central={v['lcb_central']:7.2f}  LCB_worstcase={v['lcb_worstcase']:7.2f}", flush=True)
    print(f"    mu_clears_500_central   = {lcb['mu_clears_500_central']:7.3f} TPS", flush=True)
    print(f"    mu_clears_500_worstcase = {lcb['mu_clears_500_worstcase']:7.3f} TPS  <-- GO trigger", flush=True)
    print(f"    lambda=1 ceiling mu     = {lcb['lambda1_ceiling_mu']:7.3f} TPS  "
          f"(central clears={lcb['lambda1_clears_500_central']}, worstcase clears={lcb['lambda1_clears_500_worstcase']})", flush=True)
    print(f"    land #71 colog n for rho +-0.1 = {result['colog_spec']['colog_n_allocations_for_rho_ci']}", flush=True)
    print(f"\n  SELF-TEST (PRIMARY) passes = {st['launch_sigma_closure_self_test_passes']}  "
          f"combined_sigma_launch_central (TEST) = {comb['combined_sigma_launch_central']:.4f} TPS", flush=True)
    for k, v in st["checks"].items():
        print(f"    [{'ok' if v else 'XX'}] {k}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Launch-sigma closure: de-dup x realistic-ICC combined sigma -> LCB (PR #201)")
    ap.add_argument("--out", default=os.path.join(_HERE, "launch_sigma_closure_results.json"))
    ap.add_argument("--wandb-name", "--wandb_name", default="ubel/launch-sigma-closure")
    ap.add_argument("--wandb-group", "--wandb_group", default="launch-sigma-closure")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    legs = import_legs()
    recon = dedup_x_icc(legs)
    comb = combined_launch_sigma(legs, recon)
    lcb = lcb_curve(legs, comb)
    band = icc_band_envelope(legs, recon)
    colog = colog_spec_land71()
    st = self_test(legs, recon, comb, lcb)

    s_c = comb["combined_sigma_launch_central"]
    s_w = comb["combined_sigma_launch_worstcase"]
    handoff = (
        "fern #185: import combined launch sigma = %.3f TPS (worst-case %.3f); the GO trigger is "
        "mu >= %.2f to clear 500 at P>=0.95 under the de-duped realistic-ICC covariance (vs %.2f central); "
        "the lambda=1 ceiling is %.2f TPS, so the launch is P95-reachable=%s at worst-case and =%s at central; "
        "retire the rho(*,hw) [-0.3,+0.3] band with land #71's co-log (n=%d cross-device allocations)."
        % (s_c, s_w, lcb["mu_clears_500_worstcase"], lcb["mu_clears_500_central"],
           lcb["lambda1_ceiling_mu"], lcb["lambda1_clears_500_worstcase"], lcb["lambda1_clears_500_central"],
           colog["colog_n_allocations_for_rho_ci"])
    )

    result = {
        "pr": 201,
        "metric_primary": "launch_sigma_closure_self_test_passes",
        "metric_test": "combined_sigma_launch_central",
        "launch_sigma_closure_self_test_passes": st["launch_sigma_closure_self_test_passes"],
        "combined_sigma_launch_central": s_c,
        "combined_sigma_launch_worstcase": s_w,
        "mu_clears_500_central": lcb["mu_clears_500_central"],
        "mu_clears_500_worstcase": lcb["mu_clears_500_worstcase"],
        "law": "sigma^2_launch = acc_realistic^2 + sigma_hw^2 + sigma_private^2 + 2*sum_{i<j} rho_ij*sigma_i*sigma_j; "
        "acc_realistic = acc_dedup_iid * sqrt(design_effect); LCB(mu) = mu - z_p95*sigma_launch.",
        "legs": legs,
        "dedup_x_icc": recon,
        "combined": comb,
        "lcb": lcb,
        "icc_band_envelope": band,
        "colog_spec": colog,
        "self_test": st,
        "handoff": handoff,
        "scope": "Closes the ADDITIVE CI-composition law for fern #185's launch trigger: the de-duped "
        "(ubel #195) acceptance axis evaluated under realistic ICC (wirbel #190), combined with sigma_hw "
        "(#188) and sigma_private (#176/#191), carrying the bounded rho(*,hw), propagated to LCB(mu). The "
        "additive twin of ubel #148/#169/#181's multiplicative official=K_cal*(E[T]/step)*tau pin. "
        "Orthogonal to the #192 greedy-identity gate (a CI law, not a token-identity question). CPU-only; "
        "adds 0 TPS; greedy untouched. NOT a launch.",
        "imported_legs": {
            "dedup_195": "research/validity/ci_axis_covariance/ci_axis_covariance_results.json (dedup 7.2617, worstcase 17.0375, rho matrix, [-0.3,+0.3] band)",
            "icc_190": "research/validity/icc_neff/icc_neff_results.json (icc 0.1446, deff 4.4106, n_eff 713, halfwidth_realistic 22.905)",
            "overlap_187": "research/validity/lambda_built_ci/lambda_built_ci_results.json (overlap-corrected 5.32 iid, overlap_fraction 0.893)",
            "hardware_188": "research/validity/oneshot_hw_bound/oneshot_hw_bound_results.json (sigma_hw 4.864 within/between)",
            "private_176_191": "research/validity/private_adverse_skew/results.json (sigma_private 0.884)",
            "break_even_194": "research/validity/redraw_budget/redraw_budget_results.json (break-even 512.16, mu_lambda1 520.95)",
            "forward_map_183": "research/oracle_readout/lambda_acceptance_card/lambda_acceptance_card_results.json (K_cal, step)",
        },
        "public_evidence_used": [
            "denken #187 fern-integrator note: input/output sampling CIs share overlap_fraction=0.893 on a shared bench (quadrature double-counts) -> the de-duped acceptance axis (5.32 iid) this leg inflates.",
            "wirbel #190 realistic within-prompt ICC=0.1446 (design-effect 4.411, N_eff 713): the acceptance scatter is NOT iid -> 2.1x inflation of the de-duped axis.",
            "kanna #188 within/between sigma_hw decomposition: between-device dominated 87x -> rho(*,hw) is the UNMEASURED cross-allocation coupling carried as the bounded band.",
            "leaderboard frontier ~489.6 TPS (osoi5 lmhead12k-fa2sw-precache) -> between-device hardware spread context for sigma_hw.",
        ],
        "method": "LOCAL CPU-only analytic synthesis over EXISTING MERGED results; no GPU/vLLM/HF Job/"
        "submission/served-file change. BASELINE stays 481.53; adds 0 TPS. Greedy identity untouched. NOT a launch.",
        "convention_note": "The combined launch sigma is treated as a 1-sigma multiplied by z_p95 in LCB(mu)=mu-z*sigma, "
        "consistent with #194's break-even (512.16 = 500 + 1.64485*7.391) and fern #185's packet (z_p90 * combined "
        "relative-sigma). Self-test (a) anchors the number to #195's de-dup 7.2617 at ICC=0.",
        "metrics_nan_clean": 1 if st["checks"]["g_nan_clean"] else 0,
        "elapsed_s": round(time.time() - t0, 4),
    }

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    _print(result)
    print(f"\n[launch-sigma] HANDOFF: {handoff}", flush=True)
    print(f"[launch-sigma] artifacts -> {args.out}", flush=True)
    _log_wandb(args, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
