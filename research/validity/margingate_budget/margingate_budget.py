#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""MarginGate compliant-500 budget (PR #223, wirbel) — CPU-only analytic synthesis.

THE QUESTION (capstone of #199 → #213 → #216 → this; Issue #192 lane-a)
----------------------------------------------------------------------
My #216 (`kernel-feasibility`, MERGED, `pc8g6s04`) priced the only compliant-500 route the
fleet had: a CUSTOM batch-invariant int4 verify kernel (overhead band [0.95%, 31.4%] vs the
#213 budget; buildable for λ ≳ 0.8572). A fresh literature sweep surfaced a HIGHER-ranked
(researcher P=0.45) STRUCTURAL valid-verify path the fleet has NOT priced — **MarginGate**
(arxiv 2605.30218):

    compute the logit margin  m = top1 − top2 ;
    if  m > 2·ε_max  (ε_max = the int4-Marlin split-K per-result perturbation bound),
    the argmax is PROVABLY stable across ALL reduction orders → that position needs NO
    verify GEMM at all (the cheap batch-VARIANT argmax already equals M=1 AR); only the
    (1−skip) LOW-margin fraction needs a deterministic fallback (bf16 M=1 — which IS the
    M=1 AR decode → greedy-identical by construction).

The whole viability of MarginGate turns on the (UNMEASURED) provable-stable skip rate
``skip = P(m > 2·ε_max)``. The researcher estimates skip ≈ 81–85%. This leg PRICES the
THRESHOLD that skip must beat — ``skip_rate_min(λ)`` = the skip at which MarginGate's overhead
equals my banked #213 budget ``max_kernel_overhead_pct(λ)`` — so when the cheap GPU diagnostic
(`verify_flip_probe`: measured margin distribution → skip_rate, ε_max) lands we know INSTANTLY
whether MarginGate clears. It also prices the MarginGate+DVR hybrid (DVR = LLM-42, arxiv
2601.17768) on the residual low-margin fraction, and the lowest-overhead compliant ROUTE as a
function of skip.

THE MECHANISM (imported anchors, NOT re-derived)
------------------------------------------------
Everything is a fraction of the deployed M=32 verify STEP wall-time (the unit #213/#216 use).
The #213 budget is the fractional step-time inflation a compliant kernel may add and still hit
500 TPS at acceptance λ: ``max_kernel_overhead_pct(λ) = (E[T](λ)/bar(τ) − 1)·100`` (the SAME
`LambdaCurve`, imported through #216 → #213 → #199). MarginGate's overhead, in the SAME unit:

    margingate_overhead_pct(skip) = margin_compute_pct + (1−skip)·fallback_full_pct
    fallback_full_pct = verify_gemm_cost_share_of_step · (bf16/int4 ratio) · 100
    margin_compute_pct = margin_compute_frac_of_step · 100      (≈0 — top-2 over already-computed top-1)

  * ``verify_gemm_cost_share_of_step = 0.6066`` is IMPORTED from #216 (`_cost_decomposition`).
  * the bf16 M=1 fallback recomputes a full M=1 AR decode of the low-margin position; its cost
    is the int4 verify-GEMM share scaled by the bf16/int4 byte (BW-bound) ratio ≈ 4.
  * ``spec_speedup_retained`` (the PR's conceptual third term) is ALREADY in the #213 budget
    baseline (the budget is computed AT the spec operating point E[T](λ)), so it is NOT
    re-subtracted here — comparing overhead vs the #213 budget is the apples-to-apples form.

THE THRESHOLD (the core)
------------------------
    skip_rate_min(λ) = 1 − (max_kernel_overhead_pct(λ) − margin_compute_pct) / fallback_full_pct

Monotone ↓ in λ (more acceptance ⇒ bigger budget ⇒ less skip needed). Headline
``skip_rate_min_at_lambda1`` (both-bugs, τ=1, ratio=4). We overlay the researcher's 81–85%
estimate and the empirical anchor from kanna #114's measured 56.08% per-token flip rate.

THE SOUNDNESS BOUND (why the empirical flip rate is binding)
------------------------------------------------------------
A SOUND gate must use the worst-case ε_max (≥ any realized reduction-order perturbation, incl.
the deployed M=32-vs-M=1 δ). Hence ``{m > 2·ε_max} ⊆ {does not flip M=1 vs M=32}``: every
PROVABLY-stable position is a NON-flipping position. So the measured flip rate caps the sound
skip: ``skip ≤ 1 − flip_rate = 1 − 0.5608 = 0.4392``. And ALL flips live in the (1−skip)
low-margin residual ⇒ the MarginGate+DVR hybrid's effective rollback = (1−skip)·[flip_rate/(1−skip)]
= flip_rate (CONSTANT in skip): MarginGate removes only never-flipping positions from DVR's
domain and gives ZERO rollback reduction. The hybrid therefore inherits standalone DVR's E[T]
collapse (researcher Rank-4: ~380–430 TPS).

LOCAL, CPU-ONLY, ANALYTIC. No GPU / vLLM / HF Job / submission / kernel build / probe /
served-file change. BASELINE stays 481.53; adds **0 TPS**; greedy/PPL untouched. ε_max / skip
are UNMEASURED — this leg prices the threshold the GPU diagnostic must beat (it ARMS, does NOT
run, `verify_flip_probe`). #199's three optimisms + the bf16/int4-ratio + top-2-readout
estimates carry as a NOTED band. **NOT a launch. NOT open2.**

PRIMARY metric  margingate_budget_self_test_passes
TEST    metric  skip_rate_min_at_lambda1   (both-bugs, τ=1, default bf16/int4 ratio)
"""
from __future__ import annotations

import argparse
import importlib.util
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
# Import #216's feasibility machinery (path-based) — which transitively imports #213's budget
# curve (`KB.LambdaCurve` / `budget_pct` / `lambda_crit_clears_500` / `lambda_for_budget`) and
# #199's banked compliant-spec object (`C`). Re-running their OWN code on their OWN committed
# inputs is the canonical "import the banked result" → bit-identical spines / share / budget.
# We do NOT re-derive the curve, λ_crit, 7.33%, 0.8345, 0.6066, or 51.78%.
# --------------------------------------------------------------------------- #
def _load(name: str, relpath: str):
    path = REPO_ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


KF = _load("kernel_feasibility",
           "research/validity/kernel_feasibility/kernel_feasibility.py")
KB = KF.KB   # kernel_budget_lambda (#213) — the SAME LambdaCurve / budget solvers.
C = KF.C     # compliant_spec_et (#199) — shared banked spines / clear_bar / official_tps.

# Pinned launch composition (imported via #216 → #213 → #199 → #172/#148/#168/#181).
K_CAL = KF.K_CAL                  # 125.26795005202914
STEP = KF.STEP                    # 1.2182 (#168 deployed M=32 step, normalized)
TAU_CENTRAL = KF.TAU_CENTRAL      # 1.0
TAU_CONS = KF.TAU_CONS            # 0.9924
TAU_CORNERS = KF.TAU_CORNERS
TARGET = KF.TARGET                # 500.0
BENCH_TOKENS = KF.BENCH_TOKENS    # 16384
KANNA122_OFFSHELF_OVERHEAD = KF.KANNA122_OFFSHELF_OVERHEAD  # 0.5178 (whole-model, NON-working)
REGIMES = KF.REGIMES              # ("both_bugs", "descent_only")
SPINE_0997 = KF.SPINE_0997        # 0.997 (land #71 interim optimistic spine)

# --------------------------------------------------------------------------- #
# MarginGate / DVR mechanism parameters.
#   * IMPORTED measured anchor: kanna #114's per-token flip rate (`9q5yy9l1`).
#   * UNMEASURED estimates (swept in `sensitivity`): the bf16/int4 fallback ratio and the
#     top-2-readout cost. The researcher skip estimate + standalone-DVR TPS band are the
#     researcher's PRIORS, carried verbatim (not re-derived).
# --------------------------------------------------------------------------- #
BF16_INT4_RATIO = 4.0                 # bf16/int4 byte (BW-bound) cost ratio (16b/4b); DEFAULT.
MARGIN_COMPUTE_FRAC_OF_STEP = 0.002   # conservative UB on top-2-vs-top-1 readout (truly ≈0).
KANNA114_PERTOKEN_FLIP_RATE = 0.5608  # #114 measured M=1-vs-M=32 per-token argmax divergence.
RESEARCHER_SKIP_ESTIMATE = (0.81, 0.85)   # researcher MarginGate prior (UNMEASURED).
DVR_STANDALONE_TPS_BAND = (380.0, 430.0)  # researcher Rank-4 standalone-DVR anchor (misses 500).

# sensitivity bands
RATIO_SWEEP = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]            # 1.0 = unphysical "bf16 as cheap as int4".
MARGIN_COMPUTE_SWEEP = [0.0, 0.001, 0.002, 0.005, 0.01]
SKIP_ANCHORS = [0.4392, 0.81, 0.85, 0.95, 0.97, 0.9969]  # empirical-cap … researcher … crossover.
LAMBDA_THRESHOLD_GRID = [0.9, SPINE_0997, 1.0]           # λ_crit handled as a special (budget=0) row.
LAMBDA_MONOTONE_GRID = [0.9, 0.95, SPINE_0997, 1.0]      # budget > margin_compute over this grid.

TOL_ROUNDTRIP = 1e-9


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# The overhead model (the mechanism) — all values are % of the deployed step.
# --------------------------------------------------------------------------- #
def fallback_full_pct(ratio: float, share: float) -> float:
    """Cost (%-of-step) of recomputing the WHOLE verify in bf16 M=1 (skip=0 fallback)."""
    return share * ratio * 100.0


def margin_compute_pct(mc_frac: float) -> float:
    return mc_frac * 100.0


def margingate_overhead_pct(skip: float, ratio: float, share: float, mc_frac: float) -> float:
    """margin_compute + (1−skip)·full-bf16-fallback ; monotone ↓ in skip."""
    return margin_compute_pct(mc_frac) + (1.0 - skip) * fallback_full_pct(ratio, share)


def skip_rate_min(budget_pct_val: float, ratio: float, share: float, mc_frac: float) -> float:
    """Skip at which margingate_overhead_pct == budget. May exceed 1 (margin-compute alone
    over budget ⇒ infeasible even at skip=1) or fall below 0 (skip=0 already clears)."""
    fe = fallback_full_pct(ratio, share)
    return 1.0 - (budget_pct_val - margin_compute_pct(mc_frac)) / fe


def margingate_clears(skip: float, budget_pct_val: float, ratio: float, share: float,
                      mc_frac: float) -> bool:
    return margingate_overhead_pct(skip, ratio, share, mc_frac) <= budget_pct_val + 1e-12


# --------------------------------------------------------------------------- #
# (2) The skip-rate threshold per λ (the core).
# --------------------------------------------------------------------------- #
def _skip_threshold_row(curve: Any, lam: float, tau: float, ratio: float, share: float,
                        mc_frac: float) -> dict[str, Any]:
    budget = KB.budget_pct(curve.et_of_lambda(lam), tau)
    smin = skip_rate_min(budget, ratio, share, mc_frac)
    ov_skip1 = margingate_overhead_pct(1.0, ratio, share, mc_frac)
    return {
        "lambda": lam,
        "budget_pct": budget,
        "skip_rate_min": smin,
        "skip_rate_min_in_unit_interval": bool(0.0 <= smin <= 1.0),
        "feasible_at_skip_1": bool(ov_skip1 <= budget + 1e-12),
        "researcher_estimate_lo_clears": bool(margingate_clears(
            RESEARCHER_SKIP_ESTIMATE[0], budget, ratio, share, mc_frac)),
        "researcher_estimate_hi_clears": bool(margingate_clears(
            RESEARCHER_SKIP_ESTIMATE[1], budget, ratio, share, mc_frac)),
        "empirical_floor_clears": bool(margingate_clears(
            1.0 - KANNA114_PERTOKEN_FLIP_RATE, budget, ratio, share, mc_frac)),
    }


# --------------------------------------------------------------------------- #
# (3) Route comparison + MarginGate+DVR hybrid.
# --------------------------------------------------------------------------- #
def _route_comparison(curve: Any, tau: float, decomp: dict[str, Any], ratio: float,
                      share: float, mc_frac: float) -> dict[str, Any]:
    floor_pct = decomp["custom_kernel_overhead_floor_pct"]
    attributable_pct = decomp["offtheshelf_overhead_attributable_to_verify_gemm_pct"]
    offshelf_whole = decomp["offtheshelf_whole_model_overhead_pct"]
    budget_one = KB.budget_pct(curve.et_of_lambda(1.0), tau)
    lam_crit = KB.lambda_crit_clears_500(curve, tau)
    fe = fallback_full_pct(ratio, share)

    # MarginGate overhead at the anchor skips (% of step), and its λ_min at the researcher skip.
    mg_overhead_at_skip = {
        f"{s:.4f}": margingate_overhead_pct(s, ratio, share, mc_frac) for s in SKIP_ANCHORS}
    mg_lam_min_researcher = KB.lambda_for_budget(
        curve, margingate_overhead_pct(RESEARCHER_SKIP_ESTIMATE[1], ratio, share, mc_frac), tau)
    mg_lam_min_at_threshold = KB.lambda_for_budget(  # at skip = skip_min(λ=1) ⇒ λ_min ≈ 1.0
        curve, margingate_overhead_pct(
            skip_rate_min(budget_one, ratio, share, mc_frac), ratio, share, mc_frac), tau)

    # Lowest-overhead route as f(skip): MarginGate beats the custom-kernel FLOOR iff
    # margin_compute + (1−skip)·fe < floor  ⇔  skip > 1 − (floor − margin_compute)/fe.
    skip_cross_floor = 1.0 - (floor_pct - margin_compute_pct(mc_frac)) / fe

    routes = [
        {
            "route": "off_the_shelf_BI_122",
            "overhead_at_lambda1_pct": offshelf_whole,                     # 51.78% whole-model
            "lambda_min_feasible": None,
            "lambda_min_physical": False,
            "clears_at_any_physical_lambda": False,
            "skip_dependent": False,
            "needs_measured": "nothing — VLLM_BATCH_INVARIANT=1 already measured (+51.78% whole-"
                              "model); clears at NO physical λ≤1 (max budget at prob-sat <51.78%).",
        },
        {
            "route": "custom_BI_kernel_216",
            "overhead_at_lambda1_pct": floor_pct,                          # 0.95% split-K floor
            "overhead_band_pct": [floor_pct, attributable_pct],
            "lambda_min_feasible": KB.lambda_for_budget(curve, floor_pct, tau)["lambda_for_target"],
            "lambda_min_physical": bool(
                KB.lambda_for_budget(curve, floor_pct, tau)["is_physical_lambda_le_1"]),
            "clears_at_any_physical_lambda": True,
            "skip_dependent": False,
            "needs_measured": "the actual custom kernel's split-K realization penalty (does it "
                              "hit the s_net=1.56% floor ⇒ 0.95% of step?).",
        },
        {
            "route": "MarginGate",
            "overhead_at_lambda1_pct_formula":
                f"{margin_compute_pct(mc_frac):.3f} + (1−skip)·{fe:.3f}  (% of step)",
            "overhead_at_skip_anchors_pct": mg_overhead_at_skip,
            "lambda_min_feasible_at_researcher_skip": mg_lam_min_researcher["lambda_for_target"],
            "lambda_min_physical_at_researcher_skip":
                bool(mg_lam_min_researcher["is_physical_lambda_le_1"]),
            "lambda_min_at_skip_threshold": mg_lam_min_at_threshold["lambda_for_target"],
            "skip_dependent": True,
            "needs_measured": "skip_rate = P(margin > 2·ε_max) — the verify_flip_probe margin "
                              "distribution + ε_max (UNMEASURED). Clears @λ=1 iff skip ≥ "
                              "skip_rate_min_at_lambda1.",
        },
        {
            "route": "MarginGate_plus_DVR_hybrid",
            "effective_rollback_sound": KANNA114_PERTOKEN_FLIP_RATE,       # = flip rate, CONSTANT
            "rollback_reduction_vs_standalone_dvr": 0.0,
            "skip_dependent": True,
            "clears_at_any_physical_lambda": False,
            "needs_measured": "skip_rate AND flip|low-margin; but a SOUND gate's skip ⊆ non-flips "
                              "⇒ effective rollback = full flip rate (56.08%) regardless of skip "
                              "(see hybrid block) ⇒ inherits standalone-DVR E[T] collapse.",
        },
        {
            "route": "fixed_tile_triton_BI",
            "overhead_at_lambda1_pct": floor_pct,   # same split-K floor family as custom kernel
            "overhead_band_pct": [floor_pct, attributable_pct],
            "lambda_min_feasible": KB.lambda_for_budget(curve, floor_pct, tau)["lambda_for_target"],
            "lambda_min_physical": bool(
                KB.lambda_for_budget(curve, floor_pct, tau)["is_physical_lambda_le_1"]),
            "clears_at_any_physical_lambda": True,
            "skip_dependent": False,
            "needs_measured": "the Triton fixed-tile int4 GEMM efficiency vs hand-tuned Marlin "
                              "(Triton typically ≥ the 0.95% Marlin floor — UNMEASURED).",
        },
    ]
    return {
        "tau": tau,
        "budget_at_lambda1_pct": budget_one,
        "lambda_crit": lam_crit,
        "routes": routes,
        "skip_crossover_margingate_beats_custom_floor": skip_cross_floor,
        "lowest_overhead_route_for_skip_below_crossover": "custom_BI_kernel_216",
        "lowest_overhead_route_for_skip_above_crossover": "MarginGate",
        "lowest_overhead_route_note": (
            f"For any skip < {skip_cross_floor:.5f} the custom batch-invariant int4 kernel (#216, "
            f"flat {floor_pct:.3f}% floor) is the lowest-overhead compliant route; MarginGate only "
            f"undercuts it at skip > {skip_cross_floor:.5f} (its bf16 M=1 fallback is "
            f"{fe:.1f}% of step at full fraction). The empirical skip cap "
            f"({1.0 - KANNA114_PERTOKEN_FLIP_RATE:.4f}, from #114's flip rate) and the researcher's "
            f"{RESEARCHER_SKIP_ESTIMATE[1]:.2f} both sit far below the crossover ⇒ the custom kernel "
            f"stays the route."),
    }


def _hybrid_block(curve: Any, tau: float) -> dict[str, Any]:
    """MarginGate+DVR hybrid: the soundness bound that collapses effective rollback to flip_rate."""
    flip = KANNA114_PERTOKEN_FLIP_RATE
    skip_cap = 1.0 - flip
    bar = C.clear_bar(tau)
    # researcher standalone-DVR anchor → implied E[T] band (E_T = TPS·step/K_cal).
    dvr_et_lo = DVR_STANDALONE_TPS_BAND[0] * STEP / K_CAL
    dvr_et_hi = DVR_STANDALONE_TPS_BAND[1] * STEP / K_CAL
    return {
        "tau": tau,
        "mechanism": (
            "MarginGate skips the high-margin majority (provably stable, no DVR); DVR handles "
            "ONLY the (1−skip) low-margin residual at fixed M=1 → effective rollback ≈ "
            "(1−skip)·rollback|low-margin."),
        "soundness_argument": (
            "A SOUND gate uses worst-case ε_max ≥ the realized M=32-vs-M=1 perturbation, so "
            "{m > 2·ε_max} ⊆ {does not flip M=1-vs-M=32}: every provably-stable position is a "
            "NON-flipping one, and ALL measured flips live in the (1−skip) low-margin residual. "
            "Hence rollback|low-margin = flip_rate/(1−skip) and effective rollback = "
            "(1−skip)·flip_rate/(1−skip) = flip_rate — CONSTANT in skip."),
        "skip_upper_bound_from_flip_rate": skip_cap,        # skip ≤ 1 − flip_rate (sound gate)
        "effective_rollback_sound": flip,                   # = flip_rate, independent of skip
        "rollback_reduction_from_margingate": 0.0,
        "dvr_standalone_tps_band": list(DVR_STANDALONE_TPS_BAND),
        "dvr_standalone_et_band": [dvr_et_lo, dvr_et_hi],
        "clear500_bar_et": bar,
        "hybrid_et_le_bar": bool(dvr_et_hi < bar),          # inherits DVR collapse ⇒ misses bar
        "hybrid_clears_500": False,
        "note": (
            "MarginGate removes only never-flipping positions from DVR's domain; the rollback "
            "burden (= flip_rate) is unchanged, so the hybrid inherits standalone DVR's E[T] "
            "collapse (researcher Rank-4, ~380–430 TPS < 500). Only an UNSOUND tighter gate "
            "(ε < worst-case ⇒ skips positions that DO flip ⇒ compliance-violating) could reduce "
            "it — not admissible under Issue #192 strict-A."),
    }


# --------------------------------------------------------------------------- #
# (4) Sensitivity: ε_max→skip threshold robustness + bf16/int4 ratio + margin-compute.
# --------------------------------------------------------------------------- #
def _sensitivity(curve: Any, share: float) -> dict[str, Any]:
    budget_one = KB.budget_pct(curve.et_of_lambda(1.0), TAU_CENTRAL)
    researcher_hi = RESEARCHER_SKIP_ESTIMATE[1]
    empirical_skip = 1.0 - KANNA114_PERTOKEN_FLIP_RATE

    ratio_rows = []
    for r in RATIO_SWEEP:
        smin = skip_rate_min(budget_one, r, share, MARGIN_COMPUTE_FRAC_OF_STEP)
        ratio_rows.append({
            "bf16_int4_ratio": r,
            "skip_rate_min_at_lambda1": smin,
            "researcher_hi_clears": bool(margingate_clears(
                researcher_hi, budget_one, r, share, MARGIN_COMPUTE_FRAC_OF_STEP)),
            "empirical_floor_clears": bool(margingate_clears(
                empirical_skip, budget_one, r, share, MARGIN_COMPUTE_FRAC_OF_STEP)),
        })
    smins = [row["skip_rate_min_at_lambda1"] for row in ratio_rows]

    mc_rows = []
    for mc in MARGIN_COMPUTE_SWEEP:
        smin = skip_rate_min(budget_one, BF16_INT4_RATIO, share, mc)
        mc_rows.append({"margin_compute_frac_of_step": mc, "skip_rate_min_at_lambda1": smin})

    # The crossover bf16/int4 ratio below which the researcher's 0.85 would clear (decision flip).
    # skip_rate_min(λ=1)=0.85 ⇔ 0.15 = (budget_one − mc_pct)/(share·r·100) ⇔ r = (budget−mc)/(0.15·share·100).
    mc_pct = margin_compute_pct(MARGIN_COMPUTE_FRAC_OF_STEP)
    ratio_for_researcher_to_clear = (budget_one - mc_pct) / (
        (1.0 - researcher_hi) * share * 100.0)

    # skip anchors vs the λ=1 threshold (default ratio). SKIP_ANCHORS[0]=0.4392 already
    # IS the empirical cap (1 − #114 flip rate), so no separate append is needed.
    anchor_rows = []
    for s in SKIP_ANCHORS:
        anchor_rows.append({
            "skip": s,
            "clears_at_lambda1": bool(margingate_clears(
                s, budget_one, BF16_INT4_RATIO, share, MARGIN_COMPUTE_FRAC_OF_STEP)),
            "overhead_pct": margingate_overhead_pct(
                s, BF16_INT4_RATIO, share, MARGIN_COMPUTE_FRAC_OF_STEP),
        })

    return {
        "ratio_sweep": ratio_rows,
        "skip_rate_min_at_lambda1_band_over_ratio": [min(smins), max(smins)],
        "verdict_decision_stable_over_ratio_band": bool(min(smins) > researcher_hi),
        "margin_compute_sweep": mc_rows,
        "bf16_int4_ratio_for_researcher_85_to_clear": ratio_for_researcher_to_clear,
        "bf16_int4_ratio_for_researcher_85_to_clear_is_physical":
            bool(ratio_for_researcher_to_clear >= 1.0),
        "skip_anchor_clears_at_lambda1": anchor_rows,
    }


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize(shard: Path, max_records: int | None = None) -> dict[str, Any]:
    # (0) import #199's banked object + #216's cost decomposition (their OWN code/inputs).
    c199 = C.synthesize(shard, max_records)
    lam_hat = c199["lambda_hat"]
    decomp = KF._cost_decomposition()
    share = decomp["verify_gemm_cost_share_of_step"]          # 0.6066 (IMPORTED #216)
    floor_pct = decomp["custom_kernel_overhead_floor_pct"]    # 0.9455 (IMPORTED #216)

    fe_default = fallback_full_pct(BF16_INT4_RATIO, share)
    mc_pct = margin_compute_pct(MARGIN_COMPUTE_FRAC_OF_STEP)

    regimes: dict[str, Any] = {}
    for regime in REGIMES:
        curve = KF._build_curve(c199, regime, lam_hat)
        per_tau: dict[str, Any] = {}
        for tag, tau in TAU_CORNERS:
            lam_crit = KB.lambda_crit_clears_500(curve, tau)
            # (2) the skip-rate threshold table over the λ grid (+ λ_crit special row).
            rows = [_skip_threshold_row(curve, lam, tau, BF16_INT4_RATIO, share,
                                        MARGIN_COMPUTE_FRAC_OF_STEP)
                    for lam in LAMBDA_THRESHOLD_GRID]
            crit_row = _skip_threshold_row(curve, lam_crit, tau, BF16_INT4_RATIO, share,
                                           MARGIN_COMPUTE_FRAC_OF_STEP) if lam_crit else None
            # monotone ↓ in λ (over a grid where budget > margin_compute).
            mono_smins = [skip_rate_min(KB.budget_pct(curve.et_of_lambda(l), tau),
                                        BF16_INT4_RATIO, share, MARGIN_COMPUTE_FRAC_OF_STEP)
                          for l in LAMBDA_MONOTONE_GRID]
            monotone_skip_min_in_lambda = all(
                mono_smins[i] >= mono_smins[i + 1] - 1e-12 for i in range(len(mono_smins) - 1))

            per_tau[tag] = {
                "lambda_crit": lam_crit,
                "skip_threshold_table": rows,
                "skip_threshold_at_lambda_crit": crit_row,
                "skip_rate_min_at_lambda1": skip_rate_min(
                    KB.budget_pct(curve.et_of_lambda(1.0), tau), BF16_INT4_RATIO, share,
                    MARGIN_COMPUTE_FRAC_OF_STEP),
                "skip_rate_min_at_spine_0997": skip_rate_min(
                    KB.budget_pct(curve.et_of_lambda(SPINE_0997), tau), BF16_INT4_RATIO, share,
                    MARGIN_COMPUTE_FRAC_OF_STEP),
                "monotone_skip_min_in_lambda": bool(monotone_skip_min_in_lambda),
                "route_comparison": _route_comparison(
                    curve, tau, decomp, BF16_INT4_RATIO, share, MARGIN_COMPUTE_FRAC_OF_STEP),
                "hybrid": _hybrid_block(curve, tau),
            }
        regimes[regime] = per_tau

    # sensitivity on the headline curve (both-bugs).
    head_curve = KF._build_curve(c199, "both_bugs", lam_hat)
    sensitivity = _sensitivity(head_curve, share)

    head = regimes["both_bugs"]["tau_central_1p0"]
    skip_min_l1 = head["skip_rate_min_at_lambda1"]            # TEST metric
    skip_min_0997 = head["skip_rate_min_at_spine_0997"]
    budget_one_bb1 = head["route_comparison"]["budget_at_lambda1_pct"]
    empirical_skip = 1.0 - KANNA114_PERTOKEN_FLIP_RATE
    researcher_hi_clears = head["skip_threshold_table"][-1]["researcher_estimate_hi_clears"]
    empirical_clears = head["skip_threshold_table"][-1]["empirical_floor_clears"]

    # ---------- overhead model block (Part 1) ---------- #
    overhead_model = {
        "formula": ("margingate_overhead_pct(skip) = margin_compute_pct + "
                    "(1−skip)·verify_gemm_cost_share_of_step·(bf16/int4 ratio)·100"),
        "verify_gemm_cost_share_of_step_imported_216": share,
        "bf16_int4_ratio_default": BF16_INT4_RATIO,
        "margin_compute_frac_of_step": MARGIN_COMPUTE_FRAC_OF_STEP,
        "margin_compute_pct": mc_pct,
        "fallback_full_pct_default": fe_default,             # 0.6066·4·100 = 242.65% of step
        "overhead_at_skip_0_no_gate_ceiling_pct": margingate_overhead_pct(
            0.0, BF16_INT4_RATIO, share, MARGIN_COMPUTE_FRAC_OF_STEP),
        "overhead_at_skip_1_margin_compute_only_pct": margingate_overhead_pct(
            1.0, BF16_INT4_RATIO, share, MARGIN_COMPUTE_FRAC_OF_STEP),
        "spec_speedup_retained_note": (
            "The PR's conceptual −spec_speedup_retained term is ALREADY in the #213 budget "
            "baseline (budget computed at the spec operating point E[T](λ)); comparing overhead "
            "vs that budget is the apples-to-apples form, so it is not re-subtracted (=0 here)."),
        "skip_estimate_researcher": list(RESEARCHER_SKIP_ESTIMATE),
        "empirical_skip_anchor_from_114_flip_rate": empirical_skip,
        "empirical_skip_is_a_ceiling_note": (
            "kanna #114's 56.08% per-token flip ⇒ provably-stable ⊆ non-flip ⇒ the empirical "
            "44% is a CEILING on the SOUND skip, not a floor: skip ≤ 1 − 0.5608 = 0.4392."),
        "fallback_linear_in_one_minus_skip_caveat": (
            "(1−skip)·fallback is the OPTIMISTIC (amortized one-bf16-pass) model the PR specifies; "
            "the strict M=1-sequential AR-identity requirement makes the true fallback cost higher "
            "(per-low-margin-position M=1 forwards), pushing skip_rate_min UP — so the MISS verdict "
            "is conservative."),
    }

    # ---------- self-test (PRIMARY) ---------- #
    # (a) skip=1 ⇒ overhead == margin_compute only, in [0, budget@λ=1].
    ov1 = margingate_overhead_pct(1.0, BF16_INT4_RATIO, share, MARGIN_COMPUTE_FRAC_OF_STEP)
    cond_a = (abs(ov1 - mc_pct) <= TOL_ROUNDTRIP and ov1 >= 0.0 and ov1 <= budget_one_bb1 + 1e-12)
    # (b) skip=0 ⇒ overhead == margin_compute + full fallback (the no-gate ceiling), > budget@λ=1.
    ov0 = margingate_overhead_pct(0.0, BF16_INT4_RATIO, share, MARGIN_COMPUTE_FRAC_OF_STEP)
    cond_b = (abs(ov0 - (mc_pct + fe_default)) <= TOL_ROUNDTRIP and ov0 > budget_one_bb1)
    # (c) margingate_overhead_pct monotone ↓ in skip.
    skip_grid = [i / 20.0 for i in range(21)]
    ov_grid = [margingate_overhead_pct(s, BF16_INT4_RATIO, share, MARGIN_COMPUTE_FRAC_OF_STEP)
               for s in skip_grid]
    cond_c = all(ov_grid[i] >= ov_grid[i + 1] - 1e-12 for i in range(len(ov_grid) - 1))
    # (d) skip_rate_min(λ) monotone ↓ in λ (both regimes, both τ).
    cond_d = all(regimes[r][t]["monotone_skip_min_in_lambda"]
                 for r in REGIMES for t, _ in TAU_CORNERS)
    # (e) round-trip #213 / #216 endpoints EXACTLY.
    rt_budget = abs(budget_one_bb1 - 7.331808522875782)
    rt_crit = abs((head["lambda_crit"] or -1) - 0.8344533978886615)
    rt_share = abs(share - 0.606620584396473)
    cond_e = (rt_budget <= TOL_ROUNDTRIP and rt_crit <= TOL_ROUNDTRIP and rt_share <= TOL_ROUNDTRIP)
    conditions = {
        "a_skip1_overhead_is_margin_compute_only_and_in_budget": bool(cond_a),
        "b_skip0_overhead_is_full_bf16_fallback_no_gate_ceiling": bool(cond_b),
        "c_margingate_overhead_monotone_decreasing_in_skip": bool(cond_c),
        "d_skip_rate_min_monotone_decreasing_in_lambda": bool(cond_d),
        "e_roundtrips_213_216_endpoints_budget733_crit0p8345_share0p6066": bool(cond_e),
        "f_nan_clean": True,   # set by the caller after the full payload walk.
    }

    verdict = _verdict(skip_min_l1, empirical_skip, researcher_hi_clears, empirical_clears)
    handoff = _handoff(skip_min_l1, skip_min_0997, researcher_hi_clears, empirical_clears,
                       regimes["both_bugs"]["tau_central_1p0"]["hybrid"])
    return {
        "self_test": {
            "margingate_budget_self_test_passes": bool(all(conditions.values())),
            "conditions": conditions,
        },
        "test_metric": {"skip_rate_min_at_lambda1": skip_min_l1},
        "headline": {
            "skip_rate_min_at_lambda1": skip_min_l1,
            "skip_rate_min_at_spine_0997": skip_min_0997,
            "budget_at_lambda1_both_bugs_tau1": budget_one_bb1,
            "researcher_estimate_hi_clears_at_lambda1": bool(researcher_hi_clears),
            "empirical_floor_clears_at_lambda1": bool(empirical_clears),
            "skip_upper_bound_from_flip_rate": empirical_skip,
            "hybrid_effective_rollback_sound": KANNA114_PERTOKEN_FLIP_RATE,
            "hybrid_clears_500": False,
            "lowest_overhead_route_for_achievable_skip": "custom_BI_kernel_216",
            "skip_crossover_margingate_beats_custom_floor":
                head["route_comparison"]["skip_crossover_margingate_beats_custom_floor"],
            "custom_kernel_overhead_floor_pct": floor_pct,
            "verify_gemm_cost_share_of_step": share,
            "verdict_decision_stable_over_ratio_band":
                sensitivity["verdict_decision_stable_over_ratio_band"],
        },
        "margingate_overhead_model": overhead_model,
        "regimes": regimes,
        "sensitivity": sensitivity,
        "imported_anchors": {
            "verify_gemm_cost_share_of_step_216": share,
            "custom_kernel_overhead_floor_pct_216": floor_pct,
            "offtheshelf_overhead_attributable_to_verify_gemm_pct_216":
                decomp["offtheshelf_overhead_attributable_to_verify_gemm_pct"],
            "offtheshelf_whole_model_overhead_pct_122": decomp["offtheshelf_whole_model_overhead_pct"],
            "budget_at_lambda1_both_bugs_tau1_213": budget_one_bb1,
            "lambda_crit_both_bugs_tau1_213": head["lambda_crit"],
            "kanna114_pertoken_flip_rate": KANNA114_PERTOKEN_FLIP_RATE,
            "researcher_skip_estimate": list(RESEARCHER_SKIP_ESTIMATE),
            "dvr_standalone_tps_band": list(DVR_STANDALONE_TPS_BAND),
        },
        "lambda_hat": lam_hat,
        "spine_0997": SPINE_0997,
        "composition": {
            "K_cal": K_CAL, "step": STEP, "tau_central": TAU_CENTRAL,
            "tau_conservative": TAU_CONS, "target_official": TARGET,
            "clear500_bar_et_tau1": C.clear_bar(TAU_CENTRAL),
            "clear500_bar_et_tau_cons": C.clear_bar(TAU_CONS),
            "bench_tokens": BENCH_TOKENS,
            "kanna122_offshelf_overhead_nonworking_ref": KANNA122_OFFSHELF_OVERHEAD,
            "bf16_int4_ratio_default": BF16_INT4_RATIO,
            "margin_compute_frac_of_step": MARGIN_COMPUTE_FRAC_OF_STEP,
        },
        "optimism_band_note": (
            "Carries #199's THREE optimisms (rank-1 coverage 0.7304 over-counts the true compliant "
            "accept; λ-realism vs λ̂=0.342; zero OTHER overhead) and #216's scope/split-K estimates "
            "(inherited through the imported budget + share), PLUS this leg's TWO new estimates: "
            "(i) the bf16/int4 fallback ratio (default 4 = BW bytes; swept 1–8), (ii) the top-2 "
            "margin-compute cost (default 0.2% of step, conservative UB; truly ≈0). ε_max / skip "
            "are UNMEASURED — this prices the THRESHOLD the verify_flip_probe diagnostic must beat. "
            "The MISS verdict is decision-stable across the whole ratio band (see sensitivity)."),
        "verdict": verdict,
        "handoff_line": handoff,
    }


def _verdict(skip_min_l1: float, empirical_skip: float, researcher_hi_clears: bool,
             empirical_clears: bool) -> str:
    if researcher_hi_clears or empirical_clears:
        return f"MARGINGATE-CLEARS-AT-skip_min_{skip_min_l1:.4f}"
    return (f"MARGINGATE-MISSES-COMPLIANT-500-skip_min_{skip_min_l1:.4f}-GT-"
            f"empirical_cap_{empirical_skip:.4f}-AND-researcher_0p85")


def _handoff(skip_min_l1: float, skip_min_0997: float, researcher_hi_clears: bool,
             empirical_clears: bool, hybrid: dict[str, Any]) -> str:
    r_s = "CLEARS" if researcher_hi_clears else "MISSES"
    e_s = "CLEARS" if empirical_clears else "MISSES"
    flip = KANNA114_PERTOKEN_FLIP_RATE
    return (
        f"MarginGate clears the compliant-500 budget iff the provable-stable skip rate ≥ "
        f"skip_rate_min_at_lambda1 = {skip_min_l1:.4f} (λ=1; {skip_min_0997:.4f} at #71 spine "
        f"0.997); the researcher's 81–85% estimate {r_s} and the 44% empirical floor {e_s} "
        f"(itself a CEILING — provably-stable ⊆ non-flip ⇒ skip ≤ {1.0 - flip:.4f} from #114's "
        f"{flip * 100:.2f}% flip rate), so the cheap verify_flip_probe diagnostic (measured "
        f"margin distribution → skip_rate, ε_max) is the single number that decides MarginGate — "
        f"the highest-ranked (P=0.45) valid-verify path; standalone DVR fails at our "
        f"{flip * 100:.2f}% flip (Rank-4) and the MarginGate+DVR hybrid ALSO MISSES because a "
        f"SOUND margin gate's skip set ⊆ non-flips, so its effective rollback = the full "
        f"{flip * 100:.2f}% flip rate (zero reduction). Lowest-overhead compliant route for any "
        f"achievable skip stays the custom batch-invariant int4 kernel (#216). Adds 0 TPS; "
        f"authorizes nothing; ARMS (does not run) verify_flip_probe. NOT a launch. NOT open2."
    )


# --------------------------------------------------------------------------- #
# NaN guard + report + W&B (mirrors #213/#216; never fatal).
# --------------------------------------------------------------------------- #
def _assert_nan_clean(payload: dict, path: str = "result") -> list[str]:
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


def _print_report(syn: dict) -> None:
    st, hd, om = syn["self_test"], syn["headline"], syn["margingate_overhead_model"]
    head = syn["regimes"]["both_bugs"]["tau_central_1p0"]
    print("\n" + "=" * 96, flush=True)
    print("MARGINGATE COMPLIANT-500 BUDGET (PR #223, wirbel) — Issue #192 lane-a, CPU-only",
          flush=True)
    print("=" * 96, flush=True)
    print(f"  (1) OVERHEAD MODEL (% of deployed M=32 step):", flush=True)
    print(f"      margingate_overhead(skip) = {om['margin_compute_pct']:.3f} + (1−skip)·"
          f"{om['fallback_full_pct_default']:.3f}   "
          f"[share={om['verify_gemm_cost_share_of_step_imported_216']:.4f} (#216) × "
          f"ratio={om['bf16_int4_ratio_default']:.1f}]", flush=True)
    print(f"      skip=1 ⇒ {om['overhead_at_skip_1_margin_compute_only_pct']:.3f}% (margin-compute "
          f"only)   skip=0 ⇒ {om['overhead_at_skip_0_no_gate_ceiling_pct']:.2f}% (no-gate ceiling)",
          flush=True)
    print("-" * 96, flush=True)
    print(f"  (2) SKIP-RATE THRESHOLD (both-bugs, τ=1):", flush=True)
    print(f"      {'λ':>8}  {'budget%':>9}  {'skip_rate_min':>13}  feasible@skip=1", flush=True)
    cr = head["skip_threshold_at_lambda_crit"]
    if cr:
        print(f"      {cr['lambda']:8.4f}  {cr['budget_pct']:+9.3f}  {cr['skip_rate_min']:13.5f}  "
              f"{cr['feasible_at_skip_1']}  (λ_crit — budget≈0)", flush=True)
    for row in head["skip_threshold_table"]:
        tag = "  (#71 spine)" if abs(row["lambda"] - syn["spine_0997"]) < 1e-9 else ""
        print(f"      {row['lambda']:8.4f}  {row['budget_pct']:+9.3f}  {row['skip_rate_min']:13.5f}  "
              f"{row['feasible_at_skip_1']}{tag}", flush=True)
    print(f"      HEADLINE skip_rate_min_at_lambda1 = {hd['skip_rate_min_at_lambda1']:.5f}  "
          f"(spine 0.997 = {hd['skip_rate_min_at_spine_0997']:.5f})", flush=True)
    print(f"      researcher 0.81–0.85 clears@λ=1 = {hd['researcher_estimate_hi_clears_at_lambda1']}"
          f"   empirical 44% (cap {hd['skip_upper_bound_from_flip_rate']:.4f}) clears@λ=1 = "
          f"{hd['empirical_floor_clears_at_lambda1']}", flush=True)
    print("-" * 96, flush=True)
    print(f"  (3) ROUTE COMPARISON (both-bugs, τ=1, overhead %-of-step @λ=1):", flush=True)
    for rt in head["route_comparison"]["routes"]:
        ov = rt.get("overhead_at_lambda1_pct")
        ov_s = f"{ov:7.3f}%" if isinstance(ov, (int, float)) else \
            rt.get("overhead_at_lambda1_pct_formula", "(rollback-bound)")
        lm = rt.get("lambda_min_feasible", rt.get("lambda_min_feasible_at_researcher_skip"))
        lm_s = f"{lm:.4f}" if isinstance(lm, (int, float)) else "—"
        print(f"      {rt['route']:28s} ovh@1={ov_s:>26}  λ_min={lm_s:>7}", flush=True)
    rc = head["route_comparison"]
    print(f"      crossover (MarginGate beats #216 floor) at skip > "
          f"{rc['skip_crossover_margingate_beats_custom_floor']:.5f}  ⇒ lowest-overhead route for "
          f"achievable skip = {hd['lowest_overhead_route_for_achievable_skip']}", flush=True)
    hy = head["hybrid"]
    print(f"      HYBRID MarginGate+DVR: effective rollback = {hy['effective_rollback_sound']:.4f} "
          f"(= flip rate, CONSTANT in skip; reduction {hy['rollback_reduction_from_margingate']:.2f}) "
          f"⇒ clears 500 = {hy['hybrid_clears_500']}", flush=True)
    print("-" * 96, flush=True)
    se = syn["sensitivity"]
    band = se["skip_rate_min_at_lambda1_band_over_ratio"]
    print(f"  (4) SENSITIVITY: skip_rate_min@λ=1 over ratio∈{RATIO_SWEEP} = "
          f"[{band[0]:.4f}, {band[1]:.4f}]   decision-stable(>0.85)="
          f"{se['verdict_decision_stable_over_ratio_band']}", flush=True)
    print(f"      bf16/int4 ratio that would let researcher-0.85 clear = "
          f"{se['bf16_int4_ratio_for_researcher_85_to_clear']:.4f} "
          f"(physical≥1 = {se['bf16_int4_ratio_for_researcher_85_to_clear_is_physical']})", flush=True)
    print("-" * 96, flush=True)
    print(f"  PRIMARY margingate_budget_self_test_passes = "
          f"{st['margingate_budget_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"        - {k}: {v}", flush=True)
    print(f"  TEST skip_rate_min_at_lambda1 (both-bugs, τ1, ratio={BF16_INT4_RATIO:.0f}) = "
          f"{syn['test_metric']['skip_rate_min_at_lambda1']:.5f}", flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print("=" * 96, flush=True)
    print(f"\n  HAND-OFF: {syn['handoff_line']}\n", flush=True)


def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[margingate-budget] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    run = init_wandb_run(
        job_type="compliant-spec-et-ceiling",
        agent="wirbel",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["compliant-spec-et-ceiling", "issue-192", "batch-invariant", "validity-gate",
              "margingate", "dvr-hybrid", "lane-a"],
        config={
            "K_cal": K_CAL, "step": STEP, "tau_central": TAU_CENTRAL,
            "tau_conservative": TAU_CONS, "target_official": TARGET,
            "lambda_hat": syn["lambda_hat"], "spine_0997": SPINE_0997,
            "bench_tokens": BENCH_TOKENS,
            "bf16_int4_ratio_default": BF16_INT4_RATIO,
            "margin_compute_frac_of_step": MARGIN_COMPUTE_FRAC_OF_STEP,
            "kanna114_pertoken_flip_rate": KANNA114_PERTOKEN_FLIP_RATE,
            "researcher_skip_estimate": list(RESEARCHER_SKIP_ESTIMATE),
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[margingate-budget] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    st, hd = syn["self_test"], syn["headline"]
    head = syn["regimes"]["both_bugs"]["tau_central_1p0"]
    des = syn["regimes"]["descent_only"]["tau_central_1p0"]
    om, se = syn["margingate_overhead_model"], syn["sensitivity"]
    summary: dict[str, Any] = {
        # PRIMARY + TEST
        "margingate_budget_self_test_passes":
            int(bool(st["margingate_budget_self_test_passes"])),
        "skip_rate_min_at_lambda1": hd["skip_rate_min_at_lambda1"],
        # threshold curve
        "skip_rate_min_at_spine_0997": hd["skip_rate_min_at_spine_0997"],
        "skip_rate_min_at_lambda1_descent": des["skip_rate_min_at_lambda1"],
        "budget_at_lambda1_both_bugs_tau1": hd["budget_at_lambda1_both_bugs_tau1"],
        # clears? (the decision)
        "researcher_estimate_hi_clears_at_lambda1":
            int(bool(hd["researcher_estimate_hi_clears_at_lambda1"])),
        "empirical_floor_clears_at_lambda1": int(bool(hd["empirical_floor_clears_at_lambda1"])),
        "skip_upper_bound_from_flip_rate": hd["skip_upper_bound_from_flip_rate"],
        # overhead model
        "fallback_full_pct_default": om["fallback_full_pct_default"],
        "margin_compute_pct": om["margin_compute_pct"],
        "overhead_at_skip0_no_gate_ceiling_pct": om["overhead_at_skip_0_no_gate_ceiling_pct"],
        # hybrid
        "hybrid_effective_rollback_sound": hd["hybrid_effective_rollback_sound"],
        "hybrid_clears_500": int(bool(hd["hybrid_clears_500"])),
        # route comparison
        "skip_crossover_margingate_beats_custom_floor":
            hd["skip_crossover_margingate_beats_custom_floor"],
        "custom_kernel_overhead_floor_pct": hd["custom_kernel_overhead_floor_pct"],
        "verify_gemm_cost_share_of_step": hd["verify_gemm_cost_share_of_step"],
        # sensitivity
        "skip_rate_min_at_lambda1_band_lo_over_ratio":
            se["skip_rate_min_at_lambda1_band_over_ratio"][0],
        "skip_rate_min_at_lambda1_band_hi_over_ratio":
            se["skip_rate_min_at_lambda1_band_over_ratio"][1],
        "verdict_decision_stable_over_ratio_band":
            int(bool(se["verdict_decision_stable_over_ratio_band"])),
        "bf16_int4_ratio_for_researcher_85_to_clear":
            se["bf16_int4_ratio_for_researcher_85_to_clear"],
        # bars / composition
        "clear500_bar_et_tau1": syn["composition"]["clear500_bar_et_tau1"],
        "lambda_hat": syn["lambda_hat"],
        "verdict_margingate_misses": int(syn["verdict"].startswith("MARGINGATE-MISSES")),
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    # per-ratio skip_rate_min curve as logged scalars.
    for row in se["ratio_sweep"]:
        key = f"skip_min_l1_ratio_{row['bf16_int4_ratio']:.1f}".replace(".", "p")
        summary[key] = row["skip_rate_min_at_lambda1"]
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="margingate_budget_result", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    print(f"[margingate-budget] wandb logged {len(summary)} keys", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--shard", type=Path, default=C.RANKPROBE_SHARD,
                    help="in-scope PR#86 rankprobe shard (read-only; #199 source)")
    ap.add_argument("--max-records", type=int, default=None, help="debug: cap records parsed")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="compliant-spec-et-ceiling")
    args = ap.parse_args(argv)

    syn = synthesize(args.shard, args.max_records)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 223, "agent": "wirbel",
        "kind": "margingate-budget", "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    syn["self_test"]["conditions"]["f_nan_clean"] = not nan_paths
    syn["self_test"]["margingate_budget_self_test_passes"] = bool(
        all(syn["self_test"]["conditions"].values()))
    if nan_paths:
        print(f"[margingate-budget] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "margingate_budget_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[margingate-budget] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = syn["self_test"]["margingate_budget_self_test_passes"] and payload["nan_clean"]
        print(f"[margingate-budget] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
