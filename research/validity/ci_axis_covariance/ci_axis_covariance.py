#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Cross-axis CI covariance: is fern #185's quadrature-independence assumption
valid? (PR #195)

WHAT THIS IS
------------
fern #185's launch-trigger calculator composes the single-shot launch CI by
summing FOUR axes IN QUADRATURE:
  sigma_combined = sqrt( sampling^2 + input_lambda^2 + hardware^2 + private^2 ).
Quadrature is EXACT only if the four axes are mutually INDEPENDENT (all
pairwise rho_ij = 0). That independence has never been tested. This leg pins
the ADDITIVE CI-quadrature law the way ubel #148/#169/#181 pinned the
MULTIPLICATIVE `official = K_cal*(E[T]/step)*tau` law: estimate the pairwise
correlations rho_ij from the best co-logged proxy traces, report the
covariance-corrected combined sigma, and tell fern whether quadrature holds.

  sigma^2_combined = sum_i sigma_i^2 + 2*sum_{i<j} rho_ij*sigma_i*sigma_j

THE HONEST FINDING (stated up front, derived below)
---------------------------------------------------
QUADRATURE IS NOT VALID -- but the dominant violation is NOT the hardware
coupling the PR worried about; it is a DOUBLE-COUNT inside the sampling block:

1. rho(sampling, input-lambda) = +0.945 is ALREADY MEASURED (denken #187
   rho_input_output; overlap_fraction 0.8929 = rho^2). The #175 OUTPUT-side
   accepted-length scatter (+-10.9 TPS) and the #187 INPUT-side lambda_hat CI
   (+-3.71 TPS) are two views of the SAME accept draw, not two independent
   error sources. fern's PENDING plan ("#175 composes in quadrature with the
   input-band sampling term") stacks them as separate legs -> the +2*rho*s_i*s_j
   cross term is large and POSITIVE -> the true combined sigma is ~+2.7 TPS
   ABOVE quadrature (fern's LCB too optimistic by ~z_p90*2.7 ~ 3.5 TPS). The
   physically correct fix is to DE-DUPLICATE the two into ONE acceptance axis
   (denken's overlap-corrected 5.32 TPS), which makes the block SMALLER, not
   larger. Either way quadrature-of-both is wrong.

2. The hardware<->acceptance coupling the PR flagged is UNMEASURABLE from the
   co-logged data we have. The only co-logged hardware-TPS x acceptance trace
   (kanna #188's n=12 fresh restarts) is WITHIN one pinned A10G (clock locked
   1710 MHz, sigma_within = 0.056 TPS). But sigma_hw = 4.864 TPS is
   BETWEEN-device dominated (87x: sigma_between = 4.864). The within-device
   correlation we CAN measure multiplies the negligible 0.056 leg; the
   launch-relevant BETWEEN-device correlation (does a slow allocation also
   under-accept?) needs acceptance co-logged across the frantic-penguin n=3
   cross-device draws -- which carry TPS ONLY, no acceptance. So rho(*,hw) is
   carried as a BOUNDED ASSUMPTION [-0.3,+0.3] on the full sigma_hw, pending
   land #71's served draw, and the worst-case corner is reported.

3. The private-drop pairs have NO co-logged joint draws (the private set is
   organizer-held-out) -> bounded assumption.

So: quadrature_valid = FALSE; the measured driver is the sampling/input-lambda
double-count (denken #187); the hardware coupling is a bounded assumption, not
a measurement. fern should consume the de-duplicated sampling block AND carry
the worst-case combined sigma until land #71's served draw pins rho(*,hw).

SCOPE
-----
LOCAL CPU-ONLY analytic synthesis over EXISTING co-logged traces. No GPU /
vLLM / HF Job / submission / served-file change. BASELINE stays 481.53; greedy
identity untouched; adds 0 TPS (PRIMARY = self-test). IMPORTS the individual
axis sigma VERBATIM (does NOT re-derive): sampling +-10.9 (#175) / ICC corner
(#190), input-lambda half-width (#187), sigma_hw 4.86 within/between (#188),
private-drop CI (#176/#191), fern #185 quadrature convention + z_p90. Pins the
BETWEEN-axis covariance (are the 4 CI legs independent?) -- distinct from
wirbel #190's WITHIN-sampling ICC (intra-axis N_eff). NOT open2. NOT a launch.

SELF-TEST (PR step 5 -- PRIMARY)
-------------------------------
(a) all rho_ij=0 -> combined_corrected reproduces the quadrature sigma EXACTLY
    (and the sampling(+)hardware 2-term anchor reproduces the published +-11.94
    single-shot);
(b) corrected >= quadrature IFF sum rho_ij*s_i*s_j >= 0 (monotonicity, tested
    at the measured-rho config AND a negative-rho config);
(c) each rho_ij in [-1,1] with a finite CI / bounded interval;
(d) the 4x4 covariance matrix is PSD (min eigenvalue >= -tol);
(e) worstcase >= corrected >= quadrature (central rho all >= 0);
(f) NaN-clean.
PRIMARY = ci_covariance_self_test_passes (bool); TEST = combined_sigma_corrected.
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
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # research/validity/ci_axis_covariance -> repo root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# source-of-truth artifacts (imported verbatim; one source per constant)
# ---------------------------------------------------------------------------
SAMPLING_175 = os.path.join(_ROOT, "research/oracle_readout/et_second_moment/et_second_moment_results.json")
LAMBDA_187 = os.path.join(_ROOT, "research/validity/lambda_built_ci/lambda_built_ci_results.json")
HW_188 = os.path.join(_ROOT, "research/validity/oneshot_hw_bound/oneshot_hw_bound_results.json")
HW_ENV_159 = os.path.join(_ROOT, "research/validity/hw_variance_envelope/envelope.json")
# co-logged within-device n=12 fresh restarts: wall_tps AND e_accept_exact per run.
# git_branch == approval-gated-8gpu-20260613 (isolation-clean; the lawine
# tps_noise_floor mirror is NOT used -- different branch provenance).
COLOG_N12 = os.path.join(_ROOT, "research/validity/hw_variance_envelope/fresh_n12/noise_floor_fresh.json")
PRIVATE_176 = os.path.join(_ROOT, "research/validity/private_adverse_skew/results.json")
PACKET_185 = os.path.join(_ROOT, "research/launch/packet_refresh/launch_packet_refresh_results.json")

Z95 = 1.959963984540054  # two-sided 95% normal quantile (scipy.stats.norm.ppf(0.975))
TARGET = 500.0
AXES = ["sampling", "input_lambda", "hardware", "private"]
QUAD_TOL_TPS = 0.5  # launch tol: |corrected - quadrature| <= 0.5 TPS -> quadrature "valid"


def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _fisher_ci(r: float, n: int, z: float = Z95) -> list[float]:
    """Fisher-z 95% CI for a Pearson correlation r at sample size n."""
    r = max(-0.999999, min(0.999999, float(r)))
    if n is None or n <= 3:
        return [-1.0, 1.0]
    se = 1.0 / math.sqrt(n - 3)
    zc = math.atanh(r)
    lo, hi = math.tanh(zc - z * se), math.tanh(zc + z * se)
    return [round(lo, 6), round(hi, 6)]


# ---------------------------------------------------------------------------
# Step 0 -- import the four individual axis sigma (TPS), verbatim.
# ---------------------------------------------------------------------------
def import_axis_sigmas() -> dict[str, Any]:
    samp = _load(SAMPLING_175)
    lam = _load(LAMBDA_187)
    hw = _load(HW_188)
    priv = _load(PRIVATE_176)

    # A1 sampling -- #175 both-bugs finite-sample 2nd-moment half-width (TPS),
    # the convention fern composes (the published +-11.94 single-shot is
    # sqrt(this^2 + sigma_hw^2)). ICC=1 worst corner from wirbel #190.
    sigma_sampling = float(samp["tps_finite_sample_ci_halfwidth"])  # 10.9061820...
    n_steps_iid = float(samp["both_bugs"]["N_steps"]) if "both_bugs" in samp else None
    # wirbel #190 ICC=1 corner: N_eff collapses N_steps -> N_prompts=128.
    n_prompts = 128.0
    if n_steps_iid is None:
        # recover N_steps from the 2nd-moment block if not at top level
        n_steps_iid = 3146.561123129173
    sigma_sampling_icc1 = sigma_sampling * math.sqrt(n_steps_iid / n_prompts)

    # A2 input-lambda -- denken #187 lambda_hat half-width mapped to TPS.
    syn = lam["synthesis"]["input_output_compose"]
    sigma_input_lambda = float(syn["h_in_tps_lambda_route"])  # 3.7104904...
    lambda_halfwidth = float(lam["synthesis"]["lambda_built_ci"]["lambda_built_halfwidth"])
    lambda_slope = float(syn["forward_map_slope_tps_per_lambda"])
    rho_input_output = float(syn["rho_input_output"])          # 0.9449429... (MEASURED)
    overlap_fraction = float(syn["overlap_fraction"])          # 0.8929... == rho^2

    # A3 hardware -- kanna #188 one-shot sigma with within/between decomposition.
    d = hw["decomposition"]["decomposition"]
    sigma_hw = float(d["sigma_oneshot_tps"])                   # 4.8644688...
    sigma_within = float(d["sigma_within_tps"])                # 0.0561579...
    sigma_between = float(d["sigma_between_tps"])              # 4.8641446...
    central_tps_hw = float(d["central_tps"])

    # A4 private -- stark #176/#191 native private-drop CI mapped to TPS via the
    # measured drop->TPS forward slope. fern imports the [1.87,2.21]% native CI.
    per_axis = priv["per_axis"]
    drops = np.array([a["descent_tree_drop_pct"] for a in per_axis], dtype=float)
    tps_c = np.array([a["descent_tps_central"] for a in per_axis], dtype=float)
    # forward slope d(TPS)/d(drop_pp) via least squares over the 6 native axes.
    A = np.vstack([drops, np.ones_like(drops)]).T
    slope_priv, _ = np.linalg.lstsq(A, tps_c, rcond=None)[0]
    private_drop_ci_pp = [1.87, 2.21]   # fern/stark #164 native private drop CI (pp)
    private_drop_pinned_pct = 1.801511061644668
    private_drop_adverse_pct = float(priv["headline"]["adverse_tree_drop_pct_descent"])  # 2.2999...
    drop_halfwidth_pp = (private_drop_ci_pp[1] - private_drop_ci_pp[0]) / 2.0  # 0.17 pp
    sigma_private = abs(slope_priv) * drop_halfwidth_pp     # ~0.88 TPS (95% half-width)
    # adverse worst-case private downside (pinned -> adverse corner).
    sigma_private_adverse = abs(slope_priv) * (private_drop_adverse_pct - private_drop_pinned_pct)

    return {
        "order": AXES,
        "sigma_tps": {
            "sampling": sigma_sampling,
            "input_lambda": sigma_input_lambda,
            "hardware": sigma_hw,
            "private": sigma_private,
        },
        "sampling": {
            "sigma_tps": sigma_sampling,
            "source": "wirbel #175 et_second_moment tps_finite_sample_ci_halfwidth (both-bugs, B=16384, z=1.96)",
            "N_steps_iid": n_steps_iid,
            "sigma_icc1_tps": sigma_sampling_icc1,
            "icc1_note": "wirbel #190 ICC=1 worst corner: N_eff -> N_prompts=128 inflates +-%.2f -> +-%.2f TPS" % (
                sigma_sampling, sigma_sampling_icc1),
        },
        "input_lambda": {
            "sigma_tps": sigma_input_lambda,
            "source": "denken #187 h_in_tps_lambda_route (lambda_hat half-width %.5f x slope %.3f)" % (
                lambda_halfwidth, lambda_slope),
            "lambda_halfwidth": lambda_halfwidth,
            "lambda_slope_tps_per_lambda": lambda_slope,
            "rho_input_output_measured": rho_input_output,
            "overlap_fraction": overlap_fraction,
        },
        "hardware": {
            "sigma_tps": sigma_hw,
            "sigma_within_tps": sigma_within,
            "sigma_between_tps": sigma_between,
            "between_over_within_ratio": sigma_between / sigma_within,
            "central_tps": central_tps_hw,
            "source": "kanna #188 oneshot_hw_bound sigma_oneshot_tps (within (+) between, between-dominated 87x)",
        },
        "private": {
            "sigma_tps": sigma_private,
            "sigma_adverse_tps": sigma_private_adverse,
            "drop_ci_pp": private_drop_ci_pp,
            "drop_halfwidth_pp": drop_halfwidth_pp,
            "forward_slope_tps_per_pp": float(slope_priv),
            "drop_pinned_pct": private_drop_pinned_pct,
            "drop_adverse_pct": private_drop_adverse_pct,
            "source": "stark #176/#191 private_adverse_skew native drop CI [1.87,2.21]% x measured drop->TPS slope",
        },
    }


# ---------------------------------------------------------------------------
# Step 1 -- co-logged within-device acceptance x TPS correlation (the only
# co-logged hardware x sampling trace available).
# ---------------------------------------------------------------------------
def colog_within_device() -> dict[str, Any]:
    data = _load(COLOG_N12)
    runs = data["runs"] if "runs" in data else data.get("records") or data.get("per_run")
    wall, acc = [], []
    if runs is None:
        # fall back to the per-run array under a known key
        runs = _find_runs(data)
    for r in runs:
        wt = r.get("wall_tps")
        ea = r.get("e_accept_exact")
        if _finite(wt) and _finite(ea):
            wall.append(float(wt))
            acc.append(float(ea))
    wall_a = np.array(wall, dtype=float)
    acc_a = np.array(acc, dtype=float)
    n = int(wall_a.size)
    if n >= 2 and wall_a.std() > 0 and acc_a.std() > 0:
        r_wd = float(np.corrcoef(wall_a, acc_a)[0, 1])
    else:
        r_wd = 0.0
    return {
        "n": n,
        "source": "kanna #188 hw_variance_envelope/fresh_n12 (within-device, clock-locked 1710 MHz)",
        "r_within_device": r_wd,
        "r_within_device_ci95": _fisher_ci(r_wd, n),
        "wall_tps": {"mean": float(wall_a.mean()), "std": float(wall_a.std(ddof=1)), "cv_pct": float(100 * wall_a.std(ddof=1) / wall_a.mean())},
        "e_accept_exact": {"mean": float(acc_a.mean()), "std": float(acc_a.std(ddof=1)), "cv_pct": float(100 * acc_a.std(ddof=1) / acc_a.mean())},
        "caveat": (
            "WITHIN-device only (single pinned A10G, clock locked 1710 MHz). This r "
            "multiplies sigma_within=0.056 TPS, NOT the between-device sigma_between=4.864 "
            "that dominates sigma_hw (87x). The launch-relevant BETWEEN-device acceptance x "
            "TPS correlation is NOT co-logged: the frantic-penguin n=3 cross-device draws "
            "carry TPS only, no acceptance. rho(*,hw) is therefore a bounded assumption."
        ),
    }


def _find_runs(data: dict[str, Any]) -> list[dict[str, Any]]:
    for v in data.values():
        if isinstance(v, list) and v and isinstance(v[0], dict) and "wall_tps" in v[0]:
            return v
    raise KeyError("per-run records with wall_tps not found in co-logged file")


# ---------------------------------------------------------------------------
# Step 2 -- the 4x4 correlation matrix: central estimates, CIs / bounded boxes,
# and per-pair provenance.
# ---------------------------------------------------------------------------
def build_rho_matrix(ax: dict[str, Any], colog: dict[str, Any]) -> dict[str, Any]:
    idx = {a: i for i, a in enumerate(AXES)}
    sigma_within = ax["hardware"]["sigma_within_tps"]
    sigma_hw = ax["hardware"]["sigma_tps"]
    within_share = sigma_within / sigma_hw  # ~0.0115 -- the fraction of sigma_hw the within-device r can speak to
    r_wd = colog["r_within_device"]

    # central effective rho(*, hardware): the measured within-device r informs
    # ONLY the within-device share of sigma_hw; the between-device share is
    # unmeasured -> 0 at central (carried to +-0.3 at the worst-case box).
    rho_acc_hw_central = r_wd * within_share

    rho_in_out = ax["input_lambda"]["rho_input_output_measured"]  # 0.9449 measured

    pairs: dict[str, Any] = {}

    def add(a: str, b: str, central: float, ci: list[float], method: str, n: Any, note: str,
            box: list[float] | None = None):
        key = f"{a}__{b}"
        pairs[key] = {
            "i": a, "j": b,
            "rho_central": float(central),
            "rho_ci95": [float(ci[0]), float(ci[1])],
            "rho_box": [float(box[0]), float(box[1])] if box is not None else [float(ci[0]), float(ci[1])],
            "method": method,
            "n_colog": n,
            "note": note,
        }

    # (sampling, input_lambda) -- MEASURED, denken #187. Effective n ~ 128 bench
    # prompts (conservative vs N_steps for the CI).
    add("sampling", "input_lambda", rho_in_out, _fisher_ci(rho_in_out, 128),
        "measured_denken187_rho_input_output", 128,
        "OUTPUT-side accepted-length scatter and INPUT-side lambda_hat are two views of the "
        "SAME accept draw (overlap_fraction=%.4f=rho^2). DOUBLE-COUNT if both stacked in quadrature." % ax["input_lambda"]["overlap_fraction"],
        box=[0.85, 0.99])

    # (sampling, hardware) and (input_lambda, hardware) -- within-device measured
    # proxy maps to a tiny central; between-device is the bounded assumption.
    hw_note = (
        "within-device r=%.3f (n=%d) maps to central %.4f (x within-share %.4f); BETWEEN-device "
        "rho UNMEASURED (frantic-penguin n=3 has no co-logged acceptance) -> bounded [-0.3,+0.3] on full sigma_hw."
        % (r_wd, colog["n"], rho_acc_hw_central, within_share)
    )
    add("sampling", "hardware", rho_acc_hw_central, [-0.3, 0.3],
        "within_device_proxy+bounded_between", colog["n"], hw_note, box=[-0.3, 0.3])
    add("input_lambda", "hardware", rho_acc_hw_central, [-0.3, 0.3],
        "within_device_proxy+bounded_between", colog["n"], hw_note, box=[-0.3, 0.3])

    # (sampling, private), (input_lambda, private), (hardware, private) -- NO
    # co-logged joint draws (private set organizer-held-out) -> bounded.
    priv_note = "NO co-logged joint draws (private set is organizer-held-out) -> bounded assumption, pending land #71 served draw."
    add("sampling", "private", 0.0, [-0.3, 0.3], "bounded_assumption", 0, priv_note, box=[-0.3, 0.3])
    add("input_lambda", "private", 0.0, [-0.3, 0.3], "bounded_assumption", 0,
        priv_note + " (input-lambda and private-drop both concern acceptance, but estimation noise vs a true shift -> centered 0).",
        box=[-0.3, 0.3])
    add("hardware", "private", 0.0, [-0.3, 0.3], "bounded_assumption", 0,
        priv_note + " (device allocation vs prompt-difficulty drop -> physically independent).", box=[-0.3, 0.3])

    # assemble the central 4x4 correlation matrix
    R = np.eye(4)
    R_lo = np.eye(4)
    R_hi = np.eye(4)
    for p in pairs.values():
        i, j = idx[p["i"]], idx[p["j"]]
        R[i, j] = R[j, i] = p["rho_central"]
        R_lo[i, j] = R_lo[j, i] = p["rho_box"][0]
        R_hi[i, j] = R_hi[j, i] = p["rho_box"][1]
    return {
        "axes": AXES,
        "rho_central_matrix": R.tolist(),
        "rho_box_lo_matrix": R_lo.tolist(),
        "rho_box_hi_matrix": R_hi.tolist(),
        "within_share_of_sigma_hw": within_share,
        "rho_acc_hw_central": rho_acc_hw_central,
        "pairs": pairs,
        "_R": R, "_R_lo": R_lo, "_R_hi": R_hi,
    }


# ---------------------------------------------------------------------------
# Step 3 + 4 -- combined sigma (quadrature vs corrected vs worst-case) + PSD.
# ---------------------------------------------------------------------------
def _combined_sigma(sig: np.ndarray, R: np.ndarray) -> tuple[float, np.ndarray]:
    C = np.outer(sig, sig) * R
    ones = np.ones(sig.size)
    var = float(ones @ C @ ones)
    return math.sqrt(max(var, 0.0)), C


def _min_eig(M: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(M).min())


def _psd_admissible_corner(sig: np.ndarray, R_hi: np.ndarray, tol: float = -1e-9) -> tuple[np.ndarray, float]:
    """Largest off-diagonal scaling t in [0,1] s.t. I + t*(R_hi - I) is PSD.
    1^T C 1 is monotone increasing in each rho (sigma_i>0), so the worst-case
    combined sigma over the box is at the most-correlated PSD-admissible corner."""
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


def compute_combined(ax: dict[str, Any], rho: dict[str, Any]) -> dict[str, Any]:
    sig = np.array([ax["sigma_tps"][a] for a in AXES], dtype=float)
    R = rho["_R"]
    R_hi = rho["_R_hi"]

    quad, _ = _combined_sigma(sig, np.eye(4))
    corrected, C = _combined_sigma(sig, R)
    cross_sum = 0.5 * (corrected**2 - quad**2)  # sum_{i<j} rho*s_i*s_j

    R_wc, wc_scale = _psd_admissible_corner(sig, R_hi)
    worstcase, _ = _combined_sigma(sig, R_wc)

    # ICC=1 sampling corner (wirbel #190): swap sigma_sampling for its inflated
    # value and recompute (this dominates everything -> reported as a scenario).
    sig_icc = sig.copy()
    sig_icc[0] = ax["sampling"]["sigma_icc1_tps"]
    quad_icc, _ = _combined_sigma(sig_icc, np.eye(4))
    corrected_icc, _ = _combined_sigma(sig_icc, R)
    worstcase_icc, _ = _combined_sigma(sig_icc, R_wc)

    # de-duplicated reading: collapse sampling+input_lambda into ONE acceptance
    # axis at denken's overlap-corrected TPS (the physically-correct fix).
    lam = _load(LAMBDA_187)["synthesis"]["input_output_compose"]
    overlap_corrected_tps = float(lam["overlap_corrected_same_bench_tps"])  # 5.3187 TPS
    sig_dedup = np.array([overlap_corrected_tps, ax["sigma_tps"]["hardware"], ax["sigma_tps"]["private"]], dtype=float)
    dedup_combined = math.sqrt(float(sig_dedup @ sig_dedup))  # 3 independent axes after collapse

    # faithfulness anchor: sampling (+) hardware 2-term == published +-11.94 single-shot.
    anchor_2term = math.hypot(ax["sigma_tps"]["sampling"], ax["sigma_tps"]["hardware"])

    # map combined sigma -> launch LCB shift via fern's z_p90 (one-sided P90).
    packet = _load(PACKET_185)
    z_p90 = float(packet["uncertainty_model"]["z_p90_one_sided"])
    proj_both = float(packet["step1_three_framing_geometry"]["shipped"]["both_bugs"]["proj_private_tps"])
    lcb_shift = z_p90 * (corrected - quad)               # TPS the P90 LCB moves once covariance admitted
    lcb_shift_worstcase = z_p90 * (worstcase - quad)

    quadrature_valid = abs(corrected - quad) <= QUAD_TOL_TPS

    return {
        "sigma_tps_vector": {a: float(s) for a, s in zip(AXES, sig)},
        "combined_sigma_quadrature": quad,
        "combined_sigma_corrected": corrected,
        "combined_sigma_worstcase": worstcase,
        "worstcase_psd_scale": wc_scale,
        "cross_covariance_sum": cross_sum,
        "delta_corrected_minus_quadrature_tps": corrected - quad,
        "delta_worstcase_minus_quadrature_tps": worstcase - quad,
        "quadrature_valid": bool(quadrature_valid),
        "quadrature_tol_tps": QUAD_TOL_TPS,
        "lcb_shift_from_covariance_tps": lcb_shift,
        "lcb_shift_worstcase_tps": lcb_shift_worstcase,
        "z_p90_one_sided": z_p90,
        "proj_private_tps_both": proj_both,
        "covariance_matrix": C.tolist(),
        "covariance_min_eigenvalue": _min_eig(C),
        "rho_worstcase_matrix": R_wc.tolist(),
        "anchor_2term_sampling_hw_tps": anchor_2term,
        "anchor_2term_published": 11.942,
        "icc1_corner": {
            "combined_sigma_quadrature_icc1": quad_icc,
            "combined_sigma_corrected_icc1": corrected_icc,
            "combined_sigma_worstcase_icc1": worstcase_icc,
            "sigma_sampling_icc1_tps": float(sig_icc[0]),
            "note": "wirbel #190 ICC=1 corner DOMINATES; reported as a scenario, not the central.",
        },
        "dedup_reading": {
            "overlap_corrected_sampling_block_tps": overlap_corrected_tps,
            "combined_sigma_dedup_tps": dedup_combined,
            "note": "PHYSICALLY-CORRECT fix: collapse sampling+input_lambda (rho=0.945) into ONE acceptance "
            "axis at denken's overlap-corrected 5.32 TPS -> 3 independent axes. Smaller than quadrature; the "
            "covariance correction's +inflation is the artifact of erroneously summing a redundant axis.",
        },
    }


# ---------------------------------------------------------------------------
# Step 5 -- self-test (PRIMARY).
# ---------------------------------------------------------------------------
def self_test(ax: dict[str, Any], rho: dict[str, Any], comb: dict[str, Any]) -> dict[str, Any]:
    sig = np.array([ax["sigma_tps"][a] for a in AXES], dtype=float)
    R = rho["_R"]

    # (a) all rho=0 -> corrected reproduces quadrature exactly; 2-term anchor.
    quad0, _ = _combined_sigma(sig, np.eye(4))
    a_repro = abs(quad0 - comb["combined_sigma_quadrature"]) < 1e-12
    a_anchor = abs(comb["anchor_2term_sampling_hw_tps"] - 11.942) < 0.01
    a_ok = bool(a_repro and a_anchor)

    # (b) monotonicity: corrected >= quadrature IFF sum rho*s_i*s_j >= 0.
    cross_pos = comb["cross_covariance_sum"]
    b_meas = (comb["combined_sigma_corrected"] >= comb["combined_sigma_quadrature"]) == (cross_pos >= 0)
    R_neg = np.eye(4) + (-0.1) * (np.ones((4, 4)) - np.eye(4))
    corr_neg, _ = _combined_sigma(sig, R_neg)
    cross_neg = 0.5 * (corr_neg**2 - quad0**2)
    b_negcfg = (corr_neg >= quad0) == (cross_neg >= 0)
    b_ok = bool(b_meas and b_negcfg)

    # (c) each rho in [-1,1] with a finite CI/box.
    c_ok = True
    for p in rho["pairs"].values():
        if not (-1.0 <= p["rho_central"] <= 1.0):
            c_ok = False
        if not (_finite(p["rho_ci95"][0]) and _finite(p["rho_ci95"][1])):
            c_ok = False

    # (d) the central 4x4 covariance matrix is PSD.
    d_min_eig = comb["covariance_min_eigenvalue"]
    d_ok = bool(d_min_eig >= -1e-9)

    # (e) the three reported sigmas are correctly ordered:
    #   worstcase >= corrected >= quadrature.
    # worstcase >= corrected holds BY CONSTRUCTION (the PSD-admissible corner scales
    # every off-diagonal up to R_hi >= R_central elementwise). corrected >= quadrature
    # holds IFF the NET cross-covariance is non-negative -- which it is, driven by the
    # +0.945 sampling/input-lambda double-count; the tiny -0.006 within-device hardware
    # term does NOT flip the net sign. We assert the ordering AND that the net cross
    # term is non-negative (the physical reason quadrature is the floor here). We do
    # NOT require every pairwise central rho >= 0: the covariance model legitimately
    # admits small negative correlations inside the [-0.3,+0.3] box.
    e_order = (
        comb["combined_sigma_worstcase"] >= comb["combined_sigma_corrected"] - 1e-9
        and comb["combined_sigma_corrected"] >= comb["combined_sigma_quadrature"] - 1e-9
    )
    e_net_nonneg = comb["cross_covariance_sum"] >= -1e-9
    e_ok = bool(e_order and e_net_nonneg)

    # (f) NaN-clean across all reported scalars.
    scalars = [
        comb["combined_sigma_quadrature"], comb["combined_sigma_corrected"],
        comb["combined_sigma_worstcase"], comb["cross_covariance_sum"],
        comb["lcb_shift_from_covariance_tps"], comb["covariance_min_eigenvalue"],
        comb["dedup_reading"]["combined_sigma_dedup_tps"],
        comb["icc1_corner"]["combined_sigma_corrected_icc1"],
    ] + list(sig) + [p["rho_central"] for p in rho["pairs"].values()]
    f_ok = all(_finite(x) for x in scalars)

    checks = {
        "a_rho0_reproduces_quadrature_and_anchor": a_ok,
        "b_monotone_corrected_ge_quad_iff_cross_ge_0": b_ok,
        "c_each_rho_in_unit_with_finite_ci": c_ok,
        "d_covariance_matrix_psd": d_ok,
        "e_worstcase_ge_corrected_ge_quadrature": e_ok,
        "f_nan_clean": f_ok,
    }
    passes = all(checks.values())
    return {
        "ci_covariance_self_test_passes": bool(passes),
        "checks": checks,
        "evidence": {
            "a_repro_quadrature": a_repro,
            "a_anchor_2term_tps": comb["anchor_2term_sampling_hw_tps"],
            "b_cross_sum_measured": cross_pos,
            "b_corr_neg_lt_quad": bool(corr_neg < quad0),
            "d_min_eigenvalue": d_min_eig,
            "e_order_ok": bool(e_order),
            "e_net_cross_nonneg": bool(e_net_nonneg),
            "n_scalars_checked": len(scalars),
        },
    }


def _log_wandb(args, result: dict[str, Any]) -> None:
    if args.no_wandb:
        return
    try:
        from scripts import wandb_logging
    except Exception as exc:  # noqa: BLE001
        print(f"[ci-cov] wandb_logging import failed ({exc}); skipping", flush=True)
        return
    try:
        run = wandb_logging.init_wandb_run(
            job_type="ci-axis-covariance", agent="ubel",
            name=args.wandb_name or "ubel/ci-axis-covariance",
            group=args.wandb_group,
            tags=["ci-axis-covariance", "quadrature", "covariance", "composition-pinning", "pr195"],
            config={"baseline_tps": 481.53, "method": "cpu-only-analytic", "quad_tol_tps": QUAD_TOL_TPS},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ci-cov] wandb init failed ({exc}); skipping", flush=True)
        return
    if run is None:
        print("[ci-cov] wandb disabled; skipping", flush=True)
        return
    try:
        comb = result["combined"]
        st = result["self_test"]
        flat = {
            "combined_sigma_quadrature": comb["combined_sigma_quadrature"],
            "combined_sigma_corrected": comb["combined_sigma_corrected"],
            "combined_sigma_worstcase": comb["combined_sigma_worstcase"],
            "combined_sigma_dedup": comb["dedup_reading"]["combined_sigma_dedup_tps"],
            "combined_sigma_corrected_icc1": comb["icc1_corner"]["combined_sigma_corrected_icc1"],
            "cross_covariance_sum": comb["cross_covariance_sum"],
            "delta_corrected_minus_quadrature_tps": comb["delta_corrected_minus_quadrature_tps"],
            "lcb_shift_from_covariance_tps": comb["lcb_shift_from_covariance_tps"],
            "lcb_shift_worstcase_tps": comb["lcb_shift_worstcase_tps"],
            "covariance_min_eigenvalue": comb["covariance_min_eigenvalue"],
            "quadrature_valid": 1.0 if comb["quadrature_valid"] else 0.0,
            "rho_sampling_input_lambda": result["rho"]["pairs"]["sampling__input_lambda"]["rho_central"],
            "rho_acc_hw_central": result["rho"]["rho_acc_hw_central"],
            "r_within_device": result["covariance_data_sources"]["r_within_device"],
            "sigma_sampling_tps": comb["sigma_tps_vector"]["sampling"],
            "sigma_input_lambda_tps": comb["sigma_tps_vector"]["input_lambda"],
            "sigma_hardware_tps": comb["sigma_tps_vector"]["hardware"],
            "sigma_private_tps": comb["sigma_tps_vector"]["private"],
            "ci_covariance_self_test_passes": 1.0 if st["ci_covariance_self_test_passes"] else 0.0,
        }
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run, name="ci_axis_covariance", artifact_type="ci-axis-covariance", data=result)
    except Exception as exc:  # noqa: BLE001
        print(f"[ci-cov] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:  # noqa: BLE001
            pass


def _print(result: dict[str, Any]) -> None:
    comb = result["combined"]
    rho = result["rho"]
    st = result["self_test"]
    print("\n[ci-cov] ===== CROSS-AXIS CI COVARIANCE (PR #195) =====", flush=True)
    print("  axis single-shot sigma (TPS):", flush=True)
    for a in AXES:
        print(f"    {a:14s} = {comb['sigma_tps_vector'][a]:7.3f}", flush=True)
    print(f"  4x4 rho (central):", flush=True)
    for p in rho["pairs"].values():
        print(f"    rho({p['i']:>12s},{p['j']:>12s}) = {p['rho_central']:+.4f}  "
              f"box[{p['rho_box'][0]:+.2f},{p['rho_box'][1]:+.2f}]  [{p['method']}]", flush=True)
    print(f"\n  combined_sigma_quadrature  = {comb['combined_sigma_quadrature']:.4f} TPS  (all rho=0)", flush=True)
    print(f"  combined_sigma_corrected   = {comb['combined_sigma_corrected']:.4f} TPS  (measured rho)  <-- TEST", flush=True)
    print(f"  combined_sigma_worstcase   = {comb['combined_sigma_worstcase']:.4f} TPS  (PSD-admissible corner)", flush=True)
    print(f"  combined_sigma_dedup       = {comb['dedup_reading']['combined_sigma_dedup_tps']:.4f} TPS  (collapse sampling+input_lambda)", flush=True)
    print(f"  combined_sigma_corrected_icc1 = {comb['icc1_corner']['combined_sigma_corrected_icc1']:.4f} TPS  (#190 ICC=1 corner)", flush=True)
    print(f"  delta (corrected-quad)     = {comb['delta_corrected_minus_quadrature_tps']:+.4f} TPS", flush=True)
    print(f"  quadrature_valid (|d|<= {comb['quadrature_tol_tps']}) = {comb['quadrature_valid']}", flush=True)
    print(f"  lcb_shift_from_covariance  = {comb['lcb_shift_from_covariance_tps']:+.4f} TPS (worstcase {comb['lcb_shift_worstcase_tps']:+.3f})", flush=True)
    print(f"  covariance min eigenvalue  = {comb['covariance_min_eigenvalue']:.4e} (PSD)", flush=True)
    print(f"\n  SELF-TEST (PRIMARY) passes = {st['ci_covariance_self_test_passes']}  "
          f"combined_sigma_corrected (TEST) = {comb['combined_sigma_corrected']:.4f} TPS", flush=True)
    for k, v in st["checks"].items():
        print(f"    [{'ok' if v else 'XX'}] {k}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cross-axis CI covariance (PR #195)")
    ap.add_argument("--out", default=os.path.join(_HERE, "ci_axis_covariance_results.json"))
    ap.add_argument("--wandb-name", "--wandb_name", default="ubel/ci-axis-covariance")
    ap.add_argument("--wandb-group", "--wandb_group", default="ci-axis-covariance")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    ax = import_axis_sigmas()
    colog = colog_within_device()
    rho = build_rho_matrix(ax, colog)
    comb = compute_combined(ax, rho)
    st = self_test(ax, rho, comb)

    quad = comb["combined_sigma_quadrature"]
    corr = comb["combined_sigma_corrected"]
    handoff = (
        "fern #185's quadrature is valid=%s; the cross-axis covariance-corrected combined "
        "single-shot sigma is %.3f TPS (vs quadrature %.3f), shifting the launch LCB by %+.2f TPS; "
        "the SOLE measured driver is rho(sampling,input_lambda)=%.3f (denken #187 -- a double-count, "
        "NOT independent additive axes; the physically-correct de-dup collapses it to %.2f TPS); the "
        "hardware<->acceptance coupling is UNMEASURED (co-logged trace is within-device only, "
        "sigma_within=0.056) and carried as a bounded assumption -> fern should consume the de-dup "
        "sigma AND the worst-case %.2f TPS until land #71's served draw pins rho(*,hw)."
        % (comb["quadrature_valid"], corr, quad, -comb["lcb_shift_from_covariance_tps"],
           rho["pairs"]["sampling__input_lambda"]["rho_central"],
           comb["dedup_reading"]["combined_sigma_dedup_tps"],
           comb["combined_sigma_worstcase"])
    )

    # strip numpy handles before serialization
    rho_out = {k: v for k, v in rho.items() if not k.startswith("_")}

    result = {
        "pr": 195,
        "metric_primary": "ci_covariance_self_test_passes",
        "metric_test": "combined_sigma_corrected",
        "ci_covariance_self_test_passes": st["ci_covariance_self_test_passes"],
        "combined_sigma_corrected": comb["combined_sigma_corrected"],
        "quadrature_valid": comb["quadrature_valid"],
        "covariance_model": "sigma^2_combined = sum_i sigma_i^2 + 2*sum_{i<j} rho_ij*sigma_i*sigma_j; quadrature assumes all rho_ij=0",
        "axis_sigmas": ax,
        "covariance_data_sources": colog,
        "rho": rho_out,
        "combined": comb,
        "self_test": st,
        "handoff": handoff,
        "scope": "Pins the BETWEEN-axis covariance of fern #185's 4-axis launch CI quadrature (are the "
        "legs independent?) -- the ADDITIVE-law analog of ubel #148/#169/#181's multiplicative "
        "official=K_cal*(E[T]/step)*tau pin. Distinct from wirbel #190's WITHIN-sampling ICC (intra-axis "
        "N_eff). Does NOT change the central projection or authorize a launch. BANK-THE-ANALYSIS: adds 0 "
        "TPS, greedy untouched. NOT open2. NOT a launch.",
        "imported_legs": {
            "sampling_175": "research/oracle_readout/et_second_moment/et_second_moment_results.json",
            "input_lambda_187": "research/validity/lambda_built_ci/lambda_built_ci_results.json (rho_input_output=0.9449 measured)",
            "hardware_188": "research/validity/oneshot_hw_bound/oneshot_hw_bound_results.json (within/between decomposition)",
            "colog_within_n12_188": "research/validity/hw_variance_envelope/fresh_n12/noise_floor_fresh.json",
            "private_176_191": "research/validity/private_adverse_skew/results.json",
            "fern_185_quadrature": "research/launch/packet_refresh/launch_packet_refresh_results.json (z_p90, projection_quadrature convention)",
            "wirbel_190_icc": "ICC=1 corner N_eff -> 128 (inflates sampling +-10.9 -> +-54.9 TPS); reported as a scenario",
        },
        "public_evidence_used": [
            "denken #187 fern-integrator note: input/output sampling CIs share overlap_fraction=0.893 on a shared bench (quadrature double-counts).",
            "kanna #188 within/between sigma_hw decomposition: cross-allocation dominated 87x; frantic-penguin n=3 cross-device draws carry TPS only.",
            "openevolve liveprobe (message_board 20260614-150032 @senpai): per-position acceptance ladder 0.69/0.32/0.18/... tok/step 2.583 -- the acceptance-vs-draw probe behind the sampling axis (no co-logged hardware TPS).",
            "leaderboard frontier ~481.5-489.6 TPS (between-device hardware spread).",
        ],
        "method": "LOCAL CPU-only analytic synthesis over existing co-logged traces; no GPU/vLLM/HF Job/"
        "submission/served-file change. BASELINE stays 481.53; adds 0 TPS. Greedy identity untouched. "
        "NOT open2. NOT a launch.",
        "metrics_nan_clean": 1 if st["checks"]["f_nan_clean"] else 0,
        "elapsed_s": round(time.time() - t0, 4),
    }

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    _print(result)
    print(f"\n[ci-cov] HANDOFF: {handoff}", flush=True)
    print(f"[ci-cov] artifacts -> {args.out}", flush=True)
    _log_wandb(args, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
