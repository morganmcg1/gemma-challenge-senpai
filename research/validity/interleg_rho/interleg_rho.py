#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Inter-leg RHO grounding: the grounded worst-case combined-sigma launch trigger (PR #218).

WHAT THIS IS
------------
Capstone of ubel's launch-sigma lane (#204 -> #207 -> this). #204 re-based the three
launch-noise legs onto a clean 1-sigma footing and re-solved the GO trigger as
mu = 500 + z1 * combined_sigma. It reported TWO trigger corners:

  central    : combined_sigma = hypot(sigma_accept, sigma_hw, sigma_private) = 7.5448  (rho = 0, INDEPENDENT)
               -> GO trigger 512.41,  lambda=1 margin 520.95 - 512.41 = +8.54
  worst-case : flat rho = +0.3 on ALL three leg-pairs            = 8.8972  (a HEDGE)
               -> GO trigger 514.63,  lambda=1 margin 520.95 - 514.63 = +6.32

The 2.2 TPS GAP between those two triggers (and a QUARTER of the lambda=1 margin) is
driven ENTIRELY by that flat rho = +0.3 -- an ungrounded hedge applied uniformly to all
three pairs. This PR GROUNDS the inter-leg correlation from the PHYSICAL SOURCE of each
leg, builds the real 3x3 correlation matrix, and re-solves the trigger across its
defensible band.

THE THREE LEGS AND THEIR PHYSICAL SOURCES (imported from #204, NOT re-derived)
-----------------------------------------------------------------------------
  sigma_accept  = 5.6991  self-KV-recovery ACCEPTANCE variance  (a model/content property;
                          #207 h_out 5.178 launch-correct source, #190 ICC/design-effect)
  sigma_hw      = 4.8645  between-instance HARDWARE-timing draw  (an allocation/thermal draw;
                          kanna #188 hypot(within,between), #209 within/between split)
  sigma_private = 0.8839  adverse-domain PRIVATE re-grade         (a domain property;
                          stark #176/#191 adverse-domain drop -> TPS)

THE GROUNDED RHO-MATRIX (the deliverable, step 1-2)
---------------------------------------------------
A single official re-benchmark draw fixes an acceptance pattern, a hardware allocation,
and a private-domain skew at once. Whether those three co-move:

  (accept, hw)      ~ 0      : acceptance is greedy-identical across pods (same tokens on any
                              allocation); timing jitter is orthogonal to accept/reject. Only a
                              weak 2nd-order coupling (more accepts -> fewer steps -> less
                              CLT averaging of per-step timing) -> bound |rho| <= 0.10.
  (accept, private) MILD +   : the ONE real coupling. An adverse private re-grade LOWERS
                              acceptance (drafter less aligned) -> LOWERS TPS through the SAME
                              acceptance channel sigma_accept measures. Not +1: sigma_accept also
                              carries domain-independent KV-recovery variance and sigma_private
                              carries non-acceptance (PPL-margin) components. -> rho in [0.10, 0.50].
  (hw, private)     ~ 0      : hardware allocation is orthogonal to the grading domain. Tiny
                              incidental upper bound 0.05.

So the grounded matrix has rho ~ 0 on TWO of three pairs and a mild positive on ONE.
#204's flat +0.3 puts 0.3 on ALL three -> it is STRICTLY MORE correlated -> CONSERVATIVE.

  combined_sigma = sqrt(w^T Sigma w),  w = unit leg weights [1,1,1],  Sigma = D R D,  D = diag(sigma)
  GO trigger     = 500 + z1 * combined_sigma,    z1 = 1.64485 (one-sided P95)
  lambda1 margin = 520.953 (ceiling) - GO trigger

  tight  (most independent, rho_pair at LOW)  : combined 7.611, trigger 512.52, margin +8.43
  central(rho_pair at CENTRAL)                : combined 7.743, trigger 512.74, margin +8.22
  loose  (most correlated, rho_pair at HIGH)  : combined 8.242, trigger 513.56, margin +7.40  <- worst-case

The grounded worst-case trigger 513.56 sits BELOW #204's flat-+0.3 514.63, RECOVERING ~1.08
TPS of lambda=1 margin. The accept<->private pair is the only thing keeping combined_sigma
above the pure-independent 7.5448 floor.

SCOPE
-----
LOCAL CPU-ONLY analytic synthesis over the banked #204 combined-sigma + the three legs'
physical sources. No GPU / vLLM / HF Job / submission / served-file change. Takes NO official
draws, authorizes none. BASELINE stays 481.53; greedy/PPL untouched; adds 0 TPS (PRIMARY =
self-test). The rho's are FIRST-PRINCIPLES BOUNDS (no joint re-measurement of the legs -- that
needs paired official draws), so the band is an ARGUED ENVELOPE, not a measured rho. NOT a
launch. Orthogonal to Issue #192.

SELF-TEST (PR step 4 -- PRIMARY)
--------------------------------
(a) rho-matrix all-ZERO reproduces #204's central combined 7.5448 + trigger 512.41 EXACTLY;
(b) rho-matrix all-+0.3 (flat) reproduces #204's worst-case 8.8972 + trigger 514.63 EXACTLY;
(c) the correlation matrix is positive-semidefinite at EVERY reported corner (valid covariance);
(d) combined_sigma is MONOTONE INCREASING in each rho_pair (d sigma^2 / d rho_ij = 2 sig_i sig_j > 0);
(e) grounded combined_sigma in [7.5448, 8.8972] (between #204's two corners);
(f) NaN-clean across all reported scalars.
PRIMARY = interleg_rho_self_test_passes (bool);
TEST    = go_trigger_grounded_worstcase (float TPS).
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
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # research/validity/interleg_rho -> repo root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# source-of-truth artifacts (imported verbatim; one source per constant).
# We re-read ubel's OWN banked #204 unit-rebase result (read-only) for the three clean
# 1-sigma legs, the two reference combined sigmas/triggers, and the lambda=1 ceiling.
# ---------------------------------------------------------------------------
REBASE_204 = os.path.join(_ROOT, "research/validity/launch_sigma_unit_rebase/launch_sigma_unit_rebase_results.json")

Z95_ONE_SIDED = 1.6448536269514722  # scipy.stats.norm.ppf(0.95)  -> LCB / GO-trigger multiplier (z1)
TARGET = 500.0
RHO_FLAT_WORSTCASE_204 = 0.3        # #204's flat worst-case hedge (the number being grounded)
PSD_TOL = 1e-12                     # min-eigenvalue tolerance for "valid covariance"
REPRO_TOL = 1e-9                    # anchor reproduction tolerance (machine-exact expected)

# --- grounded pairwise rho bands [low, central, high] (PR step 1; first-principles bounds) ---
# index order is (accept, hw, private); pairs keyed by the two legs they couple.
RHO_BANDS = {
    "accept_hw": {
        "low": 0.0, "central": 0.0, "high": 0.10,
        "sign": "approx_zero",
        "reasoning": (
            "sigma_accept is a MODEL/CONTENT property: which draft tokens the target accepts under "
            "self-KV-recovery is fixed by the checkpoint + prompt set and is greedy-IDENTICAL across "
            "pods (#207 h_out 5.178 is the same on any allocation). sigma_hw is an ALLOCATION/THERMAL "
            "draw: which physical A10G, its clock/thermal state, neighbour noise. A single re-benchmark "
            "draw fixes both, but they are mechanistically ORTHOGONAL -- the same tokens are produced "
            "regardless of pod, and timing jitter does not flip accept/reject decisions. central rho=0. "
            "The only coupling is 2nd-order: a higher-acceptance draw runs FEWER target forward steps, "
            "so per-step timing noise averages over fewer steps (slightly less CLT smoothing) -- a weak, "
            "sign-ambiguous VARIANCE effect, bounded |rho| <= 0.10. We take the conservative "
            "(margin-shrinking) upper end 0.10 as the high corner."
        ),
    },
    "accept_private": {
        "low": 0.10, "central": 0.30, "high": 0.50,
        "sign": "mild_positive",
        "reasoning": (
            "THE ONE REAL COUPLING. sigma_private is the adverse-domain private re-grade: when the "
            "private grading set skews adverse, the drafter is less aligned with that content and "
            "acceptance DROPS, which lowers TPS through the SAME acceptance channel sigma_accept "
            "measures. So a single re-grade draw that lands adverse SIMULTANEOUSLY lowers acceptance and "
            "lowers TPS -> POSITIVE correlation in TPS-noise terms. NOT +1: sigma_accept also carries "
            "domain-independent KV-recovery variance (#190 ICC across identical-domain repeats), and "
            "sigma_private carries re-grade components beyond acceptance (the PPL/quality-margin part of "
            "the drop, #176/#191). The acceptance channel is the dominant-but-PARTIAL shared path -> "
            "mild-to-moderate positive. central rho=0.30, band [0.10, 0.50]."
        ),
    },
    "hw_private": {
        "low": 0.0, "central": 0.0, "high": 0.05,
        "sign": "approx_zero",
        "reasoning": (
            "Hardware timing (allocation/thermal) is orthogonal to which DOMAIN the private grading set "
            "draws: the pod does not know the grading domain, and the domain does not pick the pod. "
            "central rho=0. Tiny upper bound 0.05 for any incidental coupling (e.g. adverse domains tend "
            "to longer sequences -> marginally different memory-bandwidth regime); given sigma_private="
            "0.884 is the smallest leg it contributes < 0.05 TPS to combined sigma even at its high end."
        ),
    },
}

# corner -> which end of each pair's band to use. tight=most-independent, loose=most-correlated.
CORNER_ENDS = {
    "tight": "low",
    "central": "central",
    "loose": "high",
}


def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _corr_matrix(rho_accept_hw: float, rho_accept_private: float, rho_hw_private: float) -> np.ndarray:
    """3x3 correlation matrix over legs (accept, hw, private)."""
    return np.array(
        [
            [1.0, rho_accept_hw, rho_accept_private],
            [rho_accept_hw, 1.0, rho_hw_private],
            [rho_accept_private, rho_hw_private, 1.0],
        ],
        dtype=float,
    )


def _combined_sigma_matrix(sig: np.ndarray, R: np.ndarray) -> float:
    """combined_sigma = sqrt(w^T Sigma w), w = unit leg weights, Sigma = D R D (D = diag(sig)).

    Equivalent to sqrt(sum_ij sig_i sig_j R_ij): the standard deviation of the SUM of the three
    correlated noise legs. Generalises #204's common-rho helper to a full correlation matrix.
    """
    D = np.diag(sig)
    Sigma = D @ R @ D                      # covariance from the sigmas + correlation matrix
    w = np.ones(sig.size)                  # unit leg weights
    var = float(w @ Sigma @ w)
    return math.sqrt(max(var, 0.0))


def _combined_sigma_flat_rho(sig: np.ndarray, rho: float) -> float:
    """#204's worst-case machinery: a single common off-diagonal rho on all pairs."""
    R = np.full((sig.size, sig.size), rho, dtype=float)
    np.fill_diagonal(R, 1.0)
    return _combined_sigma_matrix(sig, R)


def _min_eig(R: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(R).min())


# ---------------------------------------------------------------------------
# Step 0 -- import the banked legs + the two #204 reference corners (NOT re-derived).
# ---------------------------------------------------------------------------
def import_banked() -> dict[str, Any]:
    r204 = _load(REBASE_204)
    trig = r204["clean_trigger"]
    audit = r204["leg_footing_audit"]
    legs = r204["imported_legs_201"]
    out = {
        # three clean 1-sigma legs (exact, from #204's footing audit)
        "sigma_accept": float(audit["acceptance"]["clean_1sigma"]),     # 5.6991054426 (=11.170/z2)
        "sigma_hw": float(audit["hardware"]["clean_1sigma"]),           # 4.8644688149
        "sigma_private": float(audit["private"]["clean_1sigma"]),       # 0.8839182724
        # #204's two reference corners (the anchors this PR must reproduce)
        "combined_204_central": float(trig["combined_sigma_launch_clean_central"]),       # 7.5448108797 (rho=0)
        "combined_204_worstcase": float(trig["combined_sigma_launch_clean_worstcase"]),   # 8.8972155989 (flat 0.3)
        "trigger_204_central": float(trig["mu_clears_500_clean_central"]),                # 512.4101095
        "trigger_204_worstcase": float(trig["mu_clears_500_clean_worstcase"]),            # 514.6346173
        "lambda1_ceiling": float(trig["lambda1_ceiling_mu"]),                             # 520.9527323
        # provenance carry-throughs
        "acc_realistic_halfwidth_204": float(audit["acceptance"]["value"]),               # 11.170 (HW pre-rebase)
        "lambda1_margin_204_central": float(trig["central_headroom_below_ceiling_tps"]),  # +8.5426
        "lambda1_margin_204_worstcase": float(trig["worstcase_headroom_below_ceiling_tps"]),  # +6.3181
        "h_out_207": 5.178,  # #207 17vi7fda launch-correct acceptance source (provenance only)
    }
    return out


# ---------------------------------------------------------------------------
# Step 1 -- pairwise rho from physical sources (the mechanism).
# ---------------------------------------------------------------------------
def pairwise_rho_bands() -> dict[str, Any]:
    pairs = {}
    for key, band in RHO_BANDS.items():
        pairs[key] = {
            "low": band["low"],
            "central": band["central"],
            "high": band["high"],
            "sign": band["sign"],
            "reasoning": band["reasoning"],
        }
    pairs["_summary"] = (
        "rho_accept_hw=[%.2f,%.2f,%.2f] (approx 0, orthogonal sources, weak 2nd-order step-count "
        "coupling); rho_accept_private=[%.2f,%.2f,%.2f] (MILD POSITIVE -- the only real coupling, "
        "shared acceptance channel); rho_hw_private=[%.2f,%.2f,%.2f] (approx 0, allocation orthogonal "
        "to domain)."
        % (
            RHO_BANDS["accept_hw"]["low"], RHO_BANDS["accept_hw"]["central"], RHO_BANDS["accept_hw"]["high"],
            RHO_BANDS["accept_private"]["low"], RHO_BANDS["accept_private"]["central"], RHO_BANDS["accept_private"]["high"],
            RHO_BANDS["hw_private"]["low"], RHO_BANDS["hw_private"]["central"], RHO_BANDS["hw_private"]["high"],
        )
    )
    return pairs


def _rhos_for_corner(corner: str) -> tuple[float, float, float]:
    end = CORNER_ENDS[corner]
    return (
        RHO_BANDS["accept_hw"][end],
        RHO_BANDS["accept_private"][end],
        RHO_BANDS["hw_private"][end],
    )


# ---------------------------------------------------------------------------
# Step 2 -- combined sigma under the grounded rho-matrix + re-solved trigger (the deliverable).
# ---------------------------------------------------------------------------
def resolve_grounded_trigger(b: dict[str, Any]) -> dict[str, Any]:
    z = Z95_ONE_SIDED
    sig = np.array([b["sigma_accept"], b["sigma_hw"], b["sigma_private"]], dtype=float)
    ceiling = b["lambda1_ceiling"]

    trigger_vs_rho: dict[str, Any] = {}
    for corner in ("tight", "central", "loose"):
        ahw, ap, hwp = _rhos_for_corner(corner)
        R = _corr_matrix(ahw, ap, hwp)
        comb = _combined_sigma_matrix(sig, R)
        trig = TARGET + z * comb
        trigger_vs_rho[corner] = {
            "rho_accept_hw": ahw,
            "rho_accept_private": ap,
            "rho_hw_private": hwp,
            "combined_sigma": comb,
            "go_trigger": trig,
            "lambda1_margin": ceiling - trig,
            "min_eigenvalue": _min_eig(R),
            "psd": bool(_min_eig(R) >= -PSD_TOL),
        }

    comb_central = trigger_vs_rho["central"]["combined_sigma"]
    comb_tight = trigger_vs_rho["tight"]["combined_sigma"]
    comb_loose = trigger_vs_rho["loose"]["combined_sigma"]
    return {
        "combined_sigma_grounded_central": comb_central,
        "combined_sigma_grounded_band": [comb_tight, comb_loose],          # [tight, loose]
        "go_trigger_grounded_central": trigger_vs_rho["central"]["go_trigger"],
        "go_trigger_grounded_band": [trigger_vs_rho["tight"]["go_trigger"],
                                     trigger_vs_rho["loose"]["go_trigger"]],
        "go_trigger_grounded_worstcase": trigger_vs_rho["loose"]["go_trigger"],   # <-- TEST (= loose corner)
        "lambda1_margin_grounded_central": trigger_vs_rho["central"]["lambda1_margin"],
        "lambda1_margin_grounded_band": [trigger_vs_rho["tight"]["lambda1_margin"],
                                         trigger_vs_rho["loose"]["lambda1_margin"]],
        "lambda1_margin_grounded_worstcase": trigger_vs_rho["loose"]["lambda1_margin"],
        "lambda1_ceiling_mu": ceiling,
        "trigger_vs_rho": trigger_vs_rho,
        "independent_floor_combined_sigma": b["combined_204_central"],     # 7.5448 (pure rho=0)
        "accept_private_keeps_above_floor_tps": comb_tight - b["combined_204_central"],
    }


# ---------------------------------------------------------------------------
# Step 3 -- is #204's flat rho=+0.3 conservative? (the honest anchor).
# ---------------------------------------------------------------------------
def conservatism_check(b: dict[str, Any], grounded: dict[str, Any]) -> dict[str, Any]:
    flat_combined = b["combined_204_worstcase"]      # 8.8972 (flat 0.3)
    flat_trigger = b["trigger_204_worstcase"]        # 514.6346
    flat_margin = b["lambda1_margin_204_worstcase"]  # +6.3181

    grounded_loose_combined = grounded["combined_sigma_grounded_band"][1]
    grounded_loose_trigger = grounded["go_trigger_grounded_worstcase"]
    grounded_loose_margin = grounded["lambda1_margin_grounded_worstcase"]

    # flat 0.3 is conservative iff even the LOOSE (most-correlated) grounded corner stays under it.
    flat_03_is_conservative = bool(
        grounded_loose_combined < flat_combined - REPRO_TOL
        and grounded_loose_trigger < flat_trigger - REPRO_TOL
        and grounded_loose_margin > flat_margin + REPRO_TOL
    )
    return {
        "flat_03_is_conservative": flat_03_is_conservative,
        "flat_03_combined_sigma": flat_combined,
        "flat_03_go_trigger": flat_trigger,
        "flat_03_lambda1_margin": flat_margin,
        "grounded_loose_combined_sigma": grounded_loose_combined,
        "grounded_loose_go_trigger": grounded_loose_trigger,
        "grounded_loose_lambda1_margin": grounded_loose_margin,
        "trigger_recovered_vs_flat03_tps": flat_trigger - grounded_loose_trigger,      # how much trigger drops
        "lambda1_margin_recovered_vs_flat03_tps": grounded_loose_margin - flat_margin,  # how much margin recovered
        "residual_coupling_note": (
            "The grounded matrix puts rho approx 0 on TWO of three pairs (accept-hw, hw-private) and a "
            "mild positive on ONE (accept-private), whereas #204's flat +0.3 puts 0.3 on ALL three -> the "
            "grounded matrix is STRICTLY less correlated -> combined sigma %.4f < flat %.4f, trigger %.2f "
            "< %.2f, RECOVERING +%.2f TPS of lambda=1 margin. The accept<->private coupling is the ONLY "
            "leg-pair that keeps grounded combined sigma above the pure-independent 7.5448 floor (+%.4f "
            "TPS at the tight corner) -- carried honestly, not assumed away."
            % (
                grounded_loose_combined, flat_combined, grounded_loose_trigger, flat_trigger,
                flat_trigger - grounded_loose_trigger,
                grounded["accept_private_keeps_above_floor_tps"],
            )
        ),
    }


# ---------------------------------------------------------------------------
# Step 4 -- self-test (PRIMARY).
# ---------------------------------------------------------------------------
def self_test(b: dict[str, Any], grounded: dict[str, Any]) -> dict[str, Any]:
    z = Z95_ONE_SIDED
    sig = np.array([b["sigma_accept"], b["sigma_hw"], b["sigma_private"]], dtype=float)

    # (a) rho-matrix all-ZERO reproduces #204's central combined 7.5448 + trigger 512.41 EXACTLY.
    comb_zero = _combined_sigma_matrix(sig, _corr_matrix(0.0, 0.0, 0.0))
    trig_zero = TARGET + z * comb_zero
    a_sig_err = abs(comb_zero - b["combined_204_central"])
    a_trig_err = abs(trig_zero - b["trigger_204_central"])
    # cross-check: full-matrix and #204's flat-rho helper agree at rho=0.
    a_helper_err = abs(comb_zero - _combined_sigma_flat_rho(sig, 0.0))
    a_ok = bool(a_sig_err < REPRO_TOL and a_trig_err < REPRO_TOL and a_helper_err < REPRO_TOL)

    # (b) rho-matrix all-+0.3 (flat) reproduces #204's worst-case 8.8972 + trigger 514.63 EXACTLY.
    comb_flat = _combined_sigma_matrix(sig, _corr_matrix(0.3, 0.3, 0.3))
    trig_flat = TARGET + z * comb_flat
    b_sig_err = abs(comb_flat - b["combined_204_worstcase"])
    b_trig_err = abs(trig_flat - b["trigger_204_worstcase"])
    b_helper_err = abs(comb_flat - _combined_sigma_flat_rho(sig, RHO_FLAT_WORSTCASE_204))
    b_ok = bool(b_sig_err < REPRO_TOL and b_trig_err < REPRO_TOL and b_helper_err < REPRO_TOL)

    # (c) the correlation matrix is PSD at every reported corner (valid covariance).
    corner_min_eigs = {c: grounded["trigger_vs_rho"][c]["min_eigenvalue"]
                       for c in ("tight", "central", "loose")}
    corner_min_eigs["anchor_zero"] = _min_eig(_corr_matrix(0.0, 0.0, 0.0))
    corner_min_eigs["anchor_flat03"] = _min_eig(_corr_matrix(0.3, 0.3, 0.3))
    c_ok = bool(all(e >= -PSD_TOL for e in corner_min_eigs.values()))

    # (d) combined_sigma is MONOTONE INCREASING in each rho_pair (others held at central).
    #     analytic: d var / d rho_ij = 2 sig_i sig_j > 0. Verify numerically on a fine sweep.
    cap, app, hpp = (RHO_BANDS["accept_hw"], RHO_BANDS["accept_private"], RHO_BANDS["hw_private"])
    mono = {}
    pair_idx = {"accept_hw": (0, 1), "accept_private": (0, 2), "hw_private": (1, 2)}
    central_rhos = {"accept_hw": cap["central"], "accept_private": app["central"], "hw_private": hpp["central"]}
    for pair, band in (("accept_hw", cap), ("accept_private", app), ("hw_private", hpp)):
        grid = np.linspace(band["low"], band["high"], 25)
        combs = []
        for val in grid:
            rr = dict(central_rhos)
            rr[pair] = float(val)
            R = _corr_matrix(rr["accept_hw"], rr["accept_private"], rr["hw_private"])
            combs.append(_combined_sigma_matrix(sig, R))
        diffs = np.diff(np.array(combs))
        i, j = pair_idx[pair]
        analytic_grad = 2.0 * sig[i] * sig[j]
        mono[pair] = {
            "monotone_increasing": bool(np.all(diffs >= -1e-15)),
            "strictly_increasing": bool(np.all(diffs > 0.0)) if band["high"] > band["low"] else True,
            "analytic_dvar_drho": analytic_grad,
            "analytic_positive": bool(analytic_grad > 0.0),
        }
    d_ok = bool(all(m["monotone_increasing"] and m["analytic_positive"] for m in mono.values()))

    # (e) grounded combined_sigma in [7.5448, 8.8972] (between #204's two corners), at every corner.
    lo, hi = b["combined_204_central"], b["combined_204_worstcase"]
    grounded_combs = [grounded["trigger_vs_rho"][c]["combined_sigma"] for c in ("tight", "central", "loose")]
    e_ok = bool(all(lo - REPRO_TOL <= c <= hi + REPRO_TOL for c in grounded_combs))

    # (f) NaN-clean across all reported scalars.
    scalars = [
        comb_zero, trig_zero, comb_flat, trig_flat,
        grounded["combined_sigma_grounded_central"], grounded["go_trigger_grounded_central"],
        grounded["go_trigger_grounded_worstcase"], grounded["lambda1_margin_grounded_central"],
        grounded["lambda1_margin_grounded_worstcase"],
        *grounded["combined_sigma_grounded_band"], *grounded["go_trigger_grounded_band"],
        *grounded["lambda1_margin_grounded_band"], *corner_min_eigs.values(), *grounded_combs,
    ]
    f_ok = all(_finite(x) for x in scalars)

    checks = {
        "a_allzero_reproduces_204_central": a_ok,
        "b_flat03_reproduces_204_worstcase": b_ok,
        "c_psd_at_every_corner": c_ok,
        "d_monotone_increasing_in_each_rho": d_ok,
        "e_grounded_sigma_between_204_corners": e_ok,
        "f_nan_clean": f_ok,
    }
    passes = all(checks.values())
    return {
        "interleg_rho_self_test_passes": bool(passes),   # <-- PRIMARY
        "go_trigger_grounded_worstcase": grounded["go_trigger_grounded_worstcase"],  # <-- TEST
        "checks": checks,
        "evidence": {
            "a_allzero_sigma_err": a_sig_err,
            "a_allzero_trigger_err": a_trig_err,
            "a_helper_agreement_err": a_helper_err,
            "a_allzero_combined_sigma": comb_zero,
            "a_allzero_trigger": trig_zero,
            "b_flat03_sigma_err": b_sig_err,
            "b_flat03_trigger_err": b_trig_err,
            "b_helper_agreement_err": b_helper_err,
            "b_flat03_combined_sigma": comb_flat,
            "b_flat03_trigger": trig_flat,
            "c_corner_min_eigenvalues": corner_min_eigs,
            "d_monotonicity": mono,
            "e_grounded_combined_sigmas": grounded_combs,
            "e_204_corner_bounds": [lo, hi],
            "n_scalars_checked": len(scalars),
        },
    }


def _build_result(b, pairs, grounded, conserv, st) -> dict[str, Any]:
    ap = RHO_BANDS["accept_private"]
    handoff = (
        "fern #185: the three launch-noise legs are near-independent except a mild accept<->private "
        "coupling (rho band [%.2f,%.2f]), so the grounded worst-case GO trigger is %.2f (vs #204's "
        "flat-rho=+0.3 514.63), recovering +%.2f TPS of lambda=1 margin (now +%.2f vs +%.2f) -- #204's "
        "flat +0.3 is conservative, and fern carries the grounded [tight,loose] trigger band "
        "[%.2f, %.2f] with the accept<->private pair as the only real correlation."
        % (
            ap["low"], ap["high"],
            grounded["go_trigger_grounded_worstcase"],
            conserv["lambda1_margin_recovered_vs_flat03_tps"],
            grounded["lambda1_margin_grounded_worstcase"], b["lambda1_margin_204_worstcase"],
            grounded["go_trigger_grounded_band"][0], grounded["go_trigger_grounded_band"][1],
        )
    )
    return {
        "pr": 218,
        "metric_primary": "interleg_rho_self_test_passes",
        "metric_test": "go_trigger_grounded_worstcase",
        "interleg_rho_self_test_passes": st["interleg_rho_self_test_passes"],
        "go_trigger_grounded_worstcase": st["go_trigger_grounded_worstcase"],
        "combined_sigma_grounded_central": grounded["combined_sigma_grounded_central"],
        "combined_sigma_grounded_band": grounded["combined_sigma_grounded_band"],
        "go_trigger_grounded_central": grounded["go_trigger_grounded_central"],
        "go_trigger_grounded_band": grounded["go_trigger_grounded_band"],
        "lambda1_margin_grounded_central": grounded["lambda1_margin_grounded_central"],
        "lambda1_margin_grounded_band": grounded["lambda1_margin_grounded_band"],
        "lambda1_margin_grounded_worstcase": grounded["lambda1_margin_grounded_worstcase"],
        "flat_03_is_conservative": conserv["flat_03_is_conservative"],
        "lambda1_margin_recovered_vs_flat03_tps": conserv["lambda1_margin_recovered_vs_flat03_tps"],
        "trigger_recovered_vs_flat03_tps": conserv["trigger_recovered_vs_flat03_tps"],
        "law": (
            "combined_sigma = sqrt(w^T Sigma w), w = unit leg weights [1,1,1], Sigma = D R D, D = "
            "diag(sigma_accept, sigma_hw, sigma_private), R = grounded 3x3 correlation matrix; GO trigger "
            "= 500 + z1 * combined_sigma, z1 = 1.64485 (one-sided P95); lambda=1 margin = 520.953 - GO "
            "trigger. R has rho approx 0 on (accept,hw) and (hw,private), mild positive on (accept,private)."
        ),
        "imported_legs_204": {
            "sigma_accept": b["sigma_accept"],
            "sigma_hw": b["sigma_hw"],
            "sigma_private": b["sigma_private"],
            "combined_204_central": b["combined_204_central"],
            "combined_204_worstcase": b["combined_204_worstcase"],
            "trigger_204_central": b["trigger_204_central"],
            "trigger_204_worstcase": b["trigger_204_worstcase"],
            "lambda1_ceiling": b["lambda1_ceiling"],
            "lambda1_margin_204_central": b["lambda1_margin_204_central"],
            "lambda1_margin_204_worstcase": b["lambda1_margin_204_worstcase"],
            "h_out_207": b["h_out_207"],
            "z1_one_sided_p95": Z95_ONE_SIDED,
        },
        "pairwise_rho_bands": pairs,
        "grounded_trigger": grounded,
        "conservatism_check": conserv,
        "self_test": st,
        "handoff": handoff,
        "scope": (
            "Pure CPU-only analytic synthesis over ubel's banked #204 combined-sigma + the three legs' "
            "physical sources: grounds the inter-leg correlation rho from first principles, builds the 3x3 "
            "correlation matrix, re-solves the GO trigger and lambda=1 margin across the defensible rho "
            "band, and confirms #204's flat rho=+0.3 worst-case is conservative. Takes NO official draws, "
            "authorizes none. The rho's are FIRST-PRINCIPLES BOUNDS (no joint re-measurement of the legs "
            "-- that needs paired official draws), so the band is an ARGUED ENVELOPE, not a measured rho. "
            "The sigma magnitudes, bar 0.9780, and ceiling 520.95 are imported unchanged. BASELINE stays "
            "481.53; adds 0 TPS (PRIMARY = self-test); greedy/PPL untouched. Orthogonal to Issue #192. NOT a launch."
        ),
        "public_evidence_used": [
            "ubel #204 (launch_sigma_unit_rebase, MERGED) -- the clean-1-sigma legs sigma_accept 5.6991 / "
            "sigma_hw 4.8645 / sigma_private 0.8839, combined 7.5448 central (rho=0) / 8.8972 worst-case "
            "(flat rho=+0.3), GO trigger 512.41 / 514.63, lambda=1 ceiling 520.95 -- the curve grounded here.",
            "ubel #207 (17vi7fda, launch_sigma_175_reconcile, MERGED) -- h_out 5.178 launch-correct "
            "acceptance source confirming the robust-YES survives; the central<->worst-case spread this PR grounds.",
            "kanna #188 (pp1r5orx) -- sigma_hw = 4.864 = hypot(within, between) hardware-timing leg "
            "(physical source: between-instance allocation/thermal draw -> orthogonal to accept/domain).",
            "wirbel #190 (fva6o4ug) -- sigma_accept ICC/design-effect structure (acceptance variance is a "
            "model/content property with domain-independent KV-recovery components -> partial accept<->private channel).",
            "stark #176/#191 (jeclr39w) -- sigma_private adverse-domain re-grade (lowers acceptance via "
            "the shared channel -> the one mild-positive accept<->private coupling).",
        ],
        "method": (
            "LOCAL CPU-only analytic synthesis over EXISTING MERGED #204 results + the legs' physical "
            "sources; no GPU/vLLM/HF Job/submission/served-file change. BASELINE stays 481.53; adds 0 TPS. "
            "Greedy/PPL identity untouched. NOT a launch."
        ),
        "metrics_nan_clean": 1 if st["checks"]["f_nan_clean"] else 0,
    }


def _log_wandb(args, result: dict[str, Any]) -> None:
    if args.no_wandb:
        return
    try:
        from scripts import wandb_logging
    except Exception as exc:  # noqa: BLE001
        print(f"[interleg-rho] wandb_logging import failed ({exc}); skipping", flush=True)
        return
    try:
        run = wandb_logging.init_wandb_run(
            job_type="launch-sigma-interleg-rho", agent="ubel",
            name=args.wandb_name or "ubel/interleg-rho",
            group=args.wandb_group,
            tags=["launch-sigma", "interleg-rho", "covariance", "correlation-grounding",
                  "worst-case-trigger", "pr218"],
            config={"baseline_tps": 481.53, "method": "cpu-only-analytic", "z_one_sided_p95": Z95_ONE_SIDED,
                    "rho_flat_worstcase_204": RHO_FLAT_WORSTCASE_204,
                    "rho_accept_private_central": RHO_BANDS["accept_private"]["central"]},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[interleg-rho] wandb init failed ({exc}); skipping", flush=True)
        return
    if run is None:
        print("[interleg-rho] wandb disabled; skipping", flush=True)
        return
    try:
        grounded = result["grounded_trigger"]
        conserv = result["conservatism_check"]
        st = result["self_test"]
        tvr = grounded["trigger_vs_rho"]
        flat = {
            # legs (imported)
            "sigma_accept": result["imported_legs_204"]["sigma_accept"],
            "sigma_hw": result["imported_legs_204"]["sigma_hw"],
            "sigma_private": result["imported_legs_204"]["sigma_private"],
            # grounded rho centrals
            "rho_accept_hw_central": RHO_BANDS["accept_hw"]["central"],
            "rho_accept_private_central": RHO_BANDS["accept_private"]["central"],
            "rho_hw_private_central": RHO_BANDS["hw_private"]["central"],
            "rho_accept_private_low": RHO_BANDS["accept_private"]["low"],
            "rho_accept_private_high": RHO_BANDS["accept_private"]["high"],
            # grounded combined sigma + trigger + margin at the three corners
            "combined_sigma_tight": tvr["tight"]["combined_sigma"],
            "combined_sigma_grounded_central": grounded["combined_sigma_grounded_central"],
            "combined_sigma_loose": tvr["loose"]["combined_sigma"],
            "go_trigger_tight": tvr["tight"]["go_trigger"],
            "go_trigger_grounded_central": grounded["go_trigger_grounded_central"],
            "go_trigger_grounded_worstcase": grounded["go_trigger_grounded_worstcase"],
            "lambda1_margin_tight": tvr["tight"]["lambda1_margin"],
            "lambda1_margin_grounded_central": grounded["lambda1_margin_grounded_central"],
            "lambda1_margin_grounded_worstcase": grounded["lambda1_margin_grounded_worstcase"],
            "lambda1_ceiling_mu": grounded["lambda1_ceiling_mu"],
            # #204 anchors
            "combined_204_central": result["imported_legs_204"]["combined_204_central"],
            "combined_204_worstcase": result["imported_legs_204"]["combined_204_worstcase"],
            "trigger_204_central": result["imported_legs_204"]["trigger_204_central"],
            "trigger_204_worstcase": result["imported_legs_204"]["trigger_204_worstcase"],
            # conservatism deltas
            "trigger_recovered_vs_flat03_tps": conserv["trigger_recovered_vs_flat03_tps"],
            "lambda1_margin_recovered_vs_flat03_tps": conserv["lambda1_margin_recovered_vs_flat03_tps"],
            "accept_private_keeps_above_floor_tps": grounded["accept_private_keeps_above_floor_tps"],
            "flat_03_is_conservative": 1.0 if conserv["flat_03_is_conservative"] else 0.0,
            # PSD min eigenvalues
            "min_eig_tight": tvr["tight"]["min_eigenvalue"],
            "min_eig_central": tvr["central"]["min_eigenvalue"],
            "min_eig_loose": tvr["loose"]["min_eigenvalue"],
            # per-check booleans
            "self_test_a_allzero_repro": 1.0 if st["checks"]["a_allzero_reproduces_204_central"] else 0.0,
            "self_test_b_flat03_repro": 1.0 if st["checks"]["b_flat03_reproduces_204_worstcase"] else 0.0,
            "self_test_c_psd": 1.0 if st["checks"]["c_psd_at_every_corner"] else 0.0,
            "self_test_d_monotone": 1.0 if st["checks"]["d_monotone_increasing_in_each_rho"] else 0.0,
            "self_test_e_between_corners": 1.0 if st["checks"]["e_grounded_sigma_between_204_corners"] else 0.0,
            "self_test_f_nan_clean": 1.0 if st["checks"]["f_nan_clean"] else 0.0,
            "interleg_rho_self_test_passes": 1.0 if st["interleg_rho_self_test_passes"] else 0.0,
        }
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run, name="interleg_rho", artifact_type="launch-sigma-interleg-rho", data=result)
    except Exception as exc:  # noqa: BLE001
        print(f"[interleg-rho] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:  # noqa: BLE001
            pass


def _print(result: dict[str, Any]) -> None:
    grounded = result["grounded_trigger"]
    conserv = result["conservatism_check"]
    st = result["self_test"]
    pairs = result["pairwise_rho_bands"]
    print("\n[interleg-rho] ===== INTER-LEG RHO GROUNDING (PR #218) =====", flush=True)
    print("  pairwise rho bands [low, central, high] (first-principles):", flush=True)
    for pair in ("accept_hw", "accept_private", "hw_private"):
        p = pairs[pair]
        print(f"    rho_{pair:15s} = [{p['low']:.2f}, {p['central']:.2f}, {p['high']:.2f}]  ({p['sign']})", flush=True)
    print("  grounded combined sigma + GO trigger + lambda=1 margin (trigger_vs_rho):", flush=True)
    tvr = grounded["trigger_vs_rho"]
    for corner in ("tight", "central", "loose"):
        c = tvr[corner]
        tag = "  <- worst-case (TEST)" if corner == "loose" else ""
        print(f"    {corner:8s} sigma={c['combined_sigma']:7.4f}  trigger={c['go_trigger']:8.3f}  "
              f"margin=+{c['lambda1_margin']:.3f}  PSD={c['psd']}{tag}", flush=True)
    print(f"    grounded band: sigma [{grounded['combined_sigma_grounded_band'][0]:.4f}, "
          f"{grounded['combined_sigma_grounded_band'][1]:.4f}]  trigger "
          f"[{grounded['go_trigger_grounded_band'][0]:.3f}, {grounded['go_trigger_grounded_band'][1]:.3f}]", flush=True)
    print("  #204 anchors:  central 7.5448/512.41 (rho=0) | worst-case 8.8972/514.63 (flat 0.3)", flush=True)
    print("  conservatism check:", flush=True)
    print(f"    flat rho=+0.3 IS conservative?  {conserv['flat_03_is_conservative']}", flush=True)
    print(f"    grounded worst-case trigger {conserv['grounded_loose_go_trigger']:.3f} < flat-0.3 "
          f"{conserv['flat_03_go_trigger']:.3f}  -> recovers +{conserv['lambda1_margin_recovered_vs_flat03_tps']:.3f} "
          f"TPS lambda=1 margin", flush=True)
    print(f"    accept<->private keeps grounded sigma +{grounded['accept_private_keeps_above_floor_tps']:.4f} "
          f"TPS above the 7.5448 independent floor", flush=True)
    print(f"\n  SELF-TEST (PRIMARY) passes = {st['interleg_rho_self_test_passes']}  "
          f"go_trigger_grounded_worstcase (TEST) = {st['go_trigger_grounded_worstcase']:.4f} TPS", flush=True)
    for k, v in st["checks"].items():
        print(f"    [{'ok' if v else 'XX'}] {k}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Inter-leg rho grounding: the grounded worst-case combined-sigma launch trigger (PR #218)")
    ap.add_argument("--out", default=os.path.join(_HERE, "interleg_rho_results.json"))
    ap.add_argument("--wandb-name", "--wandb_name", default="ubel/interleg-rho")
    ap.add_argument("--wandb-group", "--wandb_group", default="launch-sigma-unit-rebase")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    b = import_banked()
    pairs = pairwise_rho_bands()
    grounded = resolve_grounded_trigger(b)
    conserv = conservatism_check(b, grounded)
    st = self_test(b, grounded)

    result = _build_result(b, pairs, grounded, conserv, st)
    result["elapsed_s"] = round(time.time() - t0, 4)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    _print(result)
    print(f"\n[interleg-rho] HANDOFF: {result['handoff']}", flush=True)
    print(f"[interleg-rho] artifacts -> {args.out}", flush=True)
    _log_wandb(args, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
