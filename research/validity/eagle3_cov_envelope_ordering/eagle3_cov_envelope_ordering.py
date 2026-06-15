#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""EAGLE-3 coverage -> envelope ORDERING: invert the speed-500 map (PR #340, stark).

THE GOVERNING QUESTION (the inverse of stark #337)
--------------------------------------------------
stark #337 (lbuirkpt, MERGED) answered the FORWARD question: at the MEASURED honest fusion coverage
c=0.8903 the compliant-500 envelope COLLAPSES -- honest_envelope_central=470.35 / worst=444.99, both
< 500 -- because E[T](0.8903)=5.5176 drops the E[T] lever to x0.9029 of the E[T](0.9213)=6.1112 anchor.

This card asks the INVERSE question the #319 retrain decision actually needs:
  *what coverage c must we REACH for the SPEED-500 envelope to hold -- and how does that required
   coverage compare to the strict 0.9213 IDENTITY bar?*
Clearing the identity bar (0.9213) is NECESSARY for greedy-validity. Is it SUFFICIENT for speed-500?
  - If the central envelope needs c*_central < 0.9213, clearing identity already buys central-500.
  - If the WORST envelope needs c*_worst > 0.9213, clearing identity does NOT buy worst-case 500.

THE METHOD (CPU-analytic over banked W&B numbers; re-derives nothing measured)
------------------------------------------------------------------------------
Reconstruct stark #337's envelope as an explicit function of coverage c and INVERT it:
  E[T](c)            = 1 + sum_{d=1..7} c^d                         (depth-7 top-4 survival product)
  envelope_central(c)= CENTRAL_ANCHOR * E[T](c) / E[T](0.9213)      (anchor 520.95 @ E[T]=6.1112)
  envelope_worst(c)  = WORST_ANCHOR   * E[T](c) / E[T](0.9213)      (anchor 492.87, same E[T])
E[T](c) is strictly increasing in c, so both envelopes are monotone -> a UNIQUE root for env(c)=500.
Solve envelope_central(c*_central)=500 and envelope_worst(c*_worst)=500 by monotone bisection.

The envelope scales at the SUB-CLIFF W=4 operating point (step x1); stark #337 proved widening to
restore E[T] is dominated (cliff mu=1.16981 > mu_tie=1.1076), so the ONLY lever for higher c is a
BETTER HEAD -- exactly what lawine #336's +0.031 retrain delivers. Hence c* is a HEAD-coverage target.

THE RESULT (anticipated by the anchors: central 520.95 > 500 already at 0.9213; worst 492.87 < 500)
---------------------------------------------------------------------------------------------------
  c*_central ~ 0.9089  <  identity_bar 0.9213  <  c*_worst ~ 0.9256
=> clearing the strict identity bar BUYS central-500 (headroom 0.0124 of coverage to spare) but is
   NOT sufficient for WORST-case 500 (shortfall 0.0043 above the bar). Tying to lawine #336: the
   +0.031 retrain budget lands exactly at 0.9213, so worst-500 needs 0.0043 MORE coverage than even
   the retrain delivers (total lift 0.0353 from the honest 0.8903 prior, just past the +0.031 budget).
STANDING CAVEAT (fern #335): coverage is the DEMAND axis only; EVEN at c*_worst the SUPPLY floor must
still revive (phi>=0.255) -- coverage alone never reaches 500.

LOCAL, CPU-ONLY, ANALYTIC. 0 GPU, no model forward, no training, no publish, no HF Job, no submission,
no served-file change, no official draw. BASELINE stays 481.53; adds 0 TPS -- it INVERTS stark #337's
banked envelope map. Imports verbatim: stark #337 lbuirkpt (E[T](c) chain law, anchors 520.95/492.87,
honest corners 470.35/444.99, cliff mu=1.16981, mu_tie=1.1076), lawine #330 hfrscdai (cov prior 0.8903,
identity bar 0.9213), lawine #336 krroookz (+0.031 retrain lift), fern #335 5pos499e (supply floor).
Re-derives nothing measured. NOT a launch / build / submission / open2.

PRIMARY metric  cov_envelope_ordering_self_test_passes
TEST    metric  c_star_central_for_500   (coverage at which the central envelope == 500)
REPORT          c_star_worst_for_500     (coverage at which the worst envelope == 500; the strict bar)
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Imported EXACT from stark #337 (lbuirkpt) -- all banked, W&B-verified. Re-derive NOTHING.
# Full-precision constants; the PR's displayed forms (0.9213, 520.95, 492.87, 0.8903) are their
# round-to-display values, asserted in the self-test (g).
# --------------------------------------------------------------------------- #
K_SPEC = 7                              # deployed speculative depth (chain-law K; ubel #311)
IDENTITY_BAR = 0.9213011665456927      # strict greedy-identity per-depth c_eff bar (== displayed 0.9213)
COV_PRIOR = 0.8903                     # lawine #330 honest fusion top-4 c_eff (the measured shortfall)
E_T_AT_IDENTITY = 6.111214987369918    # stark #337 E[T](0.9213) == 1 + sum_{d=1..7} 0.92130117^d
E_T_AT_COV_PRIOR = 5.517578068867642   # stark #337 E[T](0.8903) (the honest sub-cliff lever)

# fern #325 compliant-500 banked corners, both at E[T]=6.11 (imported through stark #337).
# central is CAP-BOUND (= lambda ceiling 520.95); worst is the uncapped private-tax corner.
CENTRAL_ANCHOR = 520.9527323111674     # stark #337 fern325_central_at_611 (cap-bound)
WORST_ANCHOR = 492.865273281899        # stark #337 fern325_worst_at_611 (uncapped)
LAMBDA_CEIL = 520.9527323111674        # int4-spec batch-invariant verify ceiling (== central cap)

# stark #337 banked honest corners at COV_PRIOR (for an EXACT #337 round-trip in the self-test).
HONEST_CENTRAL_337 = 470.347938447151
HONEST_WORST_337 = 444.9888652889661

# stark #337 sub-cliff / supra-cliff tile structure (the inverse map lives on the SUB-CLIFF branch).
MU_CLIFF = 1.1698066045772872          # measured M=32->33 verify-GEMM step ratio (widen penalty)
MU_TIE = 1.1075901257930196            # E[T](0.9213)/E[T](0.8903): widen-vs-stay tie multiplier
KNEE_MSTAR = 32                        # last M before the Marlin 2->3 block cliff (cliff @ M=33)
RAW_A1_DEMAND = 0.7730729805683441     # rank-1 acceptance a1 (salvage relation; for c->rank-cov color)
COV4_LINEAR = 0.6531976066516435       # deployed LINEAR spine top-4 rank coverage (context anchor)

# lawine #336 (krroookz) retrain head-coverage lift budget: 0.9213 - 0.8903 = +0.031.
RETRAIN_LIFT_BUDGET = 0.031

# fern #335 (5pos499e) joint AND-gate: binding axis = SUPPLY; supply floor phi for the GO corner.
SUPPLY_FLOOR_PHI = 0.255

TARGET = 500.0
BASELINE_TPS = 481.53

TOL_EXACT = 1e-9          # anchor round-trip / import-exact checks
TOL_337 = 1e-6            # reproduce stark #337 banked honest corners
TOL_ROOT = 1e-7          # inverse root residual (|env(c*) - 500|)
TOL_DISPLAY_C = 5e-5     # full-precision constant rounds to its displayed 4-dp form
TOL_DISPLAY_TPS = 5e-3   # full-precision anchor rounds to its displayed 2-dp form


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# Core laws (stark #337 conventions) + the envelope map and its inverse.
# --------------------------------------------------------------------------- #
def e_t(c: float, k: int = K_SPEC) -> float:
    """Chain-law expected accepted tokens: E[T] = 1 + sum_{d=1..K} c^d (stark #337)."""
    return 1.0 + sum(c ** d for d in range(1, k + 1))


def envelope_central(c: float) -> float:
    """Central compliant-500 envelope as a function of coverage c (stark #337 lever convention)."""
    return CENTRAL_ANCHOR * e_t(c) / E_T_AT_IDENTITY


def envelope_worst(c: float) -> float:
    """Worst (uncapped private-tax) envelope as a function of coverage c."""
    return WORST_ANCHOR * e_t(c) / E_T_AT_IDENTITY


def solve_c_for_envelope(env_fn: Callable[[float], float], target_env: float,
                         lo: float = 0.0, hi: float = 1.0, iters: int = 200) -> float:
    """Monotone bisection: smallest-residual c in [lo,hi] with env_fn(c) == target_env.

    env_fn is strictly increasing in c on [0,1] (E[T](c) is), so the root is unique.
    """
    f_lo, f_hi = env_fn(lo) - target_env, env_fn(hi) - target_env
    if f_lo > 0.0 or f_hi < 0.0:
        # target outside the achievable envelope range on [lo,hi]
        return float("nan")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if env_fn(mid) < target_env:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cov_from_c_eff(c_eff: float, a1: float = RAW_A1_DEMAND) -> float:
    """Salvage relation (denken #320): rank coverage cov = (c_eff - a1) / (1 - a1). For color only."""
    return (c_eff - a1) / (1.0 - a1)


# --------------------------------------------------------------------------- #
# (D1) Envelope map: explicit envelope_central(c) / envelope_worst(c), validated to round-trip.
# --------------------------------------------------------------------------- #
def deliverable1_envelope_map() -> dict[str, Any]:
    et_identity = e_t(IDENTITY_BAR)
    et_prior = e_t(COV_PRIOR)
    # anchor round-trip at the identity bar (exact by construction).
    central_at_bar = envelope_central(IDENTITY_BAR)
    worst_at_bar = envelope_worst(IDENTITY_BAR)
    # #337 honest-corner round-trip at the cov prior.
    central_at_prior = envelope_central(COV_PRIOR)
    worst_at_prior = envelope_worst(COV_PRIOR)
    # monotonicity probe over a fine c grid (uniqueness of the inverse root).
    grid = [i / 1000.0 for i in range(0, 1001)]
    ets = [e_t(c) for c in grid]
    env_c = [envelope_central(c) for c in grid]
    env_w = [envelope_worst(c) for c in grid]
    mono_et = all(ets[i + 1] >= ets[i] - TOL_EXACT for i in range(len(ets) - 1))
    mono_c = all(env_c[i + 1] >= env_c[i] - TOL_EXACT for i in range(len(env_c) - 1))
    mono_w = all(env_w[i + 1] >= env_w[i] - TOL_EXACT for i in range(len(env_w) - 1))
    tile_factor = WORST_ANCHOR / CENTRAL_ANCHOR     # worst/central tile factor (~0.9461)
    return {
        "chain_law": "E[T](c) = 1 + sum_{d=1..7} c^d  (depth-7 top-4 survival product; stark #337)",
        "envelope_law": ("envelope_X(c) = X_ANCHOR * E[T](c) / E[T](0.9213); "
                         "X in {central=520.95, worst=492.87}, both anchored at E[T]=6.1112"),
        "K": K_SPEC,
        "e_t_at_identity_0p9213": et_identity,
        "e_t_at_cov_prior_0p8903": et_prior,
        "et_reproduces_banked_identity": bool(abs(et_identity - E_T_AT_IDENTITY) <= TOL_EXACT),
        "et_reproduces_banked_prior": bool(abs(et_prior - E_T_AT_COV_PRIOR) <= TOL_EXACT),
        "worst_over_central_tile_factor": tile_factor,
        "anchor_roundtrip": {
            "envelope_central_at_identity": central_at_bar,
            "envelope_worst_at_identity": worst_at_bar,
            "matches_central_anchor": bool(abs(central_at_bar - CENTRAL_ANCHOR) <= TOL_EXACT),
            "matches_worst_anchor": bool(abs(worst_at_bar - WORST_ANCHOR) <= TOL_EXACT),
        },
        "honest_corner_roundtrip_337": {
            "envelope_central_at_prior": central_at_prior,
            "envelope_worst_at_prior": worst_at_prior,
            "matches_337_central_470p35": bool(abs(central_at_prior - HONEST_CENTRAL_337) <= TOL_337),
            "matches_337_worst_444p99": bool(abs(worst_at_prior - HONEST_WORST_337) <= TOL_337),
            "both_below_500": bool(central_at_prior < TARGET and worst_at_prior < TARGET),
        },
        "monotone_increasing": {
            "e_t": bool(mono_et), "envelope_central": bool(mono_c), "envelope_worst": bool(mono_w),
            "all": bool(mono_et and mono_c and mono_w),
        },
    }


# --------------------------------------------------------------------------- #
# (D2) Inverse solve (PRIMARY): c*_central and c*_worst at which each envelope == 500.
# --------------------------------------------------------------------------- #
def deliverable2_inverse_solve() -> dict[str, Any]:
    c_star_central = solve_c_for_envelope(envelope_central, TARGET)
    c_star_worst = solve_c_for_envelope(envelope_worst, TARGET)
    res_central = envelope_central(c_star_central) - TARGET
    res_worst = envelope_worst(c_star_worst) - TARGET
    # target E[T] at each root (sanity): env=500 -> E[T] = 500 * E[T](0.9213) / anchor.
    et_target_central = TARGET * E_T_AT_IDENTITY / CENTRAL_ANCHOR
    et_target_worst = TARGET * E_T_AT_IDENTITY / WORST_ANCHOR
    return {
        "c_star_central_for_500": c_star_central,           # TEST metric
        "c_star_worst_for_500": c_star_worst,               # REPORT metric
        "root_residual_central": res_central,
        "root_residual_worst": res_worst,
        "e_t_target_central": et_target_central,
        "e_t_target_worst": et_target_worst,
        "e_t_at_c_star_central": e_t(c_star_central),
        "e_t_at_c_star_worst": e_t(c_star_worst),
        "roots_in_unit_interval": bool(0.0 < c_star_central < 1.0 and 0.0 < c_star_worst < 1.0),
        "central_root_valid": bool(_finite(c_star_central) and abs(res_central) <= TOL_ROOT),
        "worst_root_valid": bool(_finite(c_star_worst) and abs(res_worst) <= TOL_ROOT),
        "note": ("monotone bisection on env_X(c)=500; E[T](c) strictly increasing -> unique root. "
                 "c*_central solves the (cap-bound) central envelope; since 500 < cap 520.95 the "
                 "root lies BELOW the bar where central scales (above the bar central is flat at the "
                 "ceiling, irrelevant to a 500 crossing)."),
    }


# --------------------------------------------------------------------------- #
# (D3) The ORDERING verdict: {c*_central, identity_bar, c*_worst} + sufficiency booleans.
# --------------------------------------------------------------------------- #
def deliverable3_ordering(d2: dict) -> dict[str, Any]:
    cc = d2["c_star_central_for_500"]
    cw = d2["c_star_worst_for_500"]
    suff_central = bool(cc <= IDENTITY_BAR)
    suff_worst = bool(cw <= IDENTITY_BAR)
    ordering_confirmed = bool(cc < IDENTITY_BAR < cw)
    ordering_str = "c*_central < identity_bar < c*_worst" if ordering_confirmed else "OTHER"
    return {
        "ordering_values": {
            "c_star_central_for_500": cc,
            "identity_bar": IDENTITY_BAR,
            "c_star_worst_for_500": cw,
        },
        "ordering_string": ordering_str,
        "anticipated_ordering_confirmed": ordering_confirmed,
        "identity_bar_suffices_for_central_500": suff_central,
        "identity_bar_suffices_for_worst_500": suff_worst,
        "c_star_worst_gt_c_star_central": bool(cw > cc),
        "verdict": (
            "identity bar is SUFFICIENT for central-500 (c*_central={:.4f} <= 0.9213) but NOT "
            "sufficient for worst-case 500 (c*_worst={:.4f} > 0.9213): clearing the strict "
            "greedy-identity bar buys central-500, while private-stable worst-500 demands a STRICTER "
            "coverage.".format(cc, cw)),
    }


# --------------------------------------------------------------------------- #
# (D4) Coverage headroom / shortfall + tie to lawine #336's +0.031 retrain budget.
# --------------------------------------------------------------------------- #
def deliverable4_headroom_shortfall(d2: dict) -> dict[str, Any]:
    cc = d2["c_star_central_for_500"]
    cw = d2["c_star_worst_for_500"]
    central_headroom = IDENTITY_BAR - cc          # coverage slack the bar gives above central-500 need
    worst_shortfall = cw - IDENTITY_BAR           # extra coverage beyond the bar for worst-500
    total_lift_central = cc - COV_PRIOR           # lift from honest prior to reach central-500
    total_lift_worst = cw - COV_PRIOR             # lift from honest prior to reach worst-500
    # the +0.031 retrain lands exactly at the identity bar (0.9213). worst-500 sits above it.
    worst_within_retrain_budget = bool(total_lift_worst <= RETRAIN_LIFT_BUDGET + TOL_EXACT)
    extra_lift_beyond_retrain = total_lift_worst - RETRAIN_LIFT_BUDGET
    shortfall_frac_of_budget = worst_shortfall / RETRAIN_LIFT_BUDGET
    return {
        "central_500_headroom": central_headroom,
        "worst_500_shortfall": worst_shortfall,
        "total_lift_for_central_500_from_prior": total_lift_central,
        "total_lift_for_worst_500_from_prior": total_lift_worst,
        "retrain_lift_budget_336": RETRAIN_LIFT_BUDGET,
        "worst_500_within_retrain_budget": worst_within_retrain_budget,
        "extra_lift_needed_beyond_retrain": extra_lift_beyond_retrain,
        "worst_shortfall_as_frac_of_retrain_budget": shortfall_frac_of_budget,
        "note": (
            "central-500 sits {:.4f} of coverage BELOW the identity bar (the bar gives slack); "
            "worst-500 sits {:.4f} ABOVE it. lawine #336's +{:.3f} retrain lands the head exactly at "
            "0.9213, so worst-500 needs {:.4f} MORE coverage than the retrain delivers (total lift "
            "{:.4f} from the honest 0.8903 prior, vs the +{:.3f} budget). The shortfall is small "
            "({:.1f}% of the retrain budget) but NONZERO: a worst-500 retrain must TARGET c*_worst, "
            "not merely the identity bar.".format(
                central_headroom, worst_shortfall, RETRAIN_LIFT_BUDGET, worst_shortfall,
                total_lift_worst, RETRAIN_LIFT_BUDGET, 100.0 * shortfall_frac_of_budget)),
    }


# --------------------------------------------------------------------------- #
# (D5) Sub-cliff structure: c* are HEAD-coverage targets at W=4 step x1 (widening dominated).
# --------------------------------------------------------------------------- #
def deliverable5_subcliff_structure(d2: dict) -> dict[str, Any]:
    cc = d2["c_star_central_for_500"]
    cw = d2["c_star_worst_for_500"]
    widen_dominated = bool(MU_CLIFF > MU_TIE)     # stark #337: widening to restore E[T] loses
    # map c* to the underlying top-4 rank coverage via the salvage relation (color only).
    rankcov_central = cov_from_c_eff(cc)
    rankcov_worst = cov_from_c_eff(cw)
    rankcov_identity = cov_from_c_eff(IDENTITY_BAR)   # == COV4_LINEAR by construction
    return {
        "operating_point": "sub-cliff W=4 (M=29 <= knee 32), verify step x1",
        "widening_dominated": widen_dominated,
        "mu_cliff": MU_CLIFF,
        "mu_tie": MU_TIE,
        "lever_argument": (
            "the inverse map scales the envelope at the SUB-CLIFF W=4 point (step x1). stark #337 "
            "proved widening to restore E[T] crosses the M=32->33 cliff (mu={:.5f} > mu_tie={:.5f}), "
            "a net loss -- so the ONLY lever for higher coverage is a BETTER HEAD. Hence c*_central / "
            "c*_worst are HEAD-COVERAGE targets a retrain must hit at fixed W=4, not tree-width "
            "changes.".format(MU_CLIFF, MU_TIE)),
        "rank_coverage_map": {
            "note": ("c is per-depth effective acceptance c_eff; underlying top-4 rank coverage via "
                     "salvage cov=(c_eff-a1)/(1-a1), a1=0.7731 (color only, NOT the ordering axis)"),
            "rankcov_for_c_star_central": rankcov_central,
            "rankcov_for_c_star_worst": rankcov_worst,
            "rankcov_at_identity_bar": rankcov_identity,
            "rankcov_identity_equals_linear_cov4": bool(abs(rankcov_identity - COV4_LINEAR) <= 1e-6),
            "worst_needs_rankcov_above_linear_spine": bool(rankcov_worst > COV4_LINEAR),
            "color_note": ("c*_worst needs top-4 rank coverage {:.4f} > the deployed LINEAR spine's "
                           "{:.4f}: worst-500 asks for a head STRONGER than the linear spine "
                           "itself.".format(rankcov_worst, COV4_LINEAR)),
        },
    }


# --------------------------------------------------------------------------- #
# (D6) Decision framing + greedy-safety + fern #335 supply caveat.
# --------------------------------------------------------------------------- #
def deliverable6_decision(d2: dict, d3: dict, d4: dict) -> dict[str, Any]:
    cc = d2["c_star_central_for_500"]
    cw = d2["c_star_worst_for_500"]
    return {
        "central_500_target_coverage": IDENTITY_BAR,     # identity bar already suffices
        "worst_500_target_coverage": cw,                 # strictly above the identity bar
        "binding_constraint_for_central_500": (
            "identity_bar 0.9213 (c*_central={:.4f} <= bar) -- clearing greedy-identity IS the "
            "central-500 target".format(cc)),
        "binding_constraint_for_worst_500": (
            "c*_worst={:.4f} > identity_bar -- worst-500 needs a STRICTER coverage than greedy "
            "identity".format(cw)),
        "retrain_sizing": (
            "if the human's #319 retrain targets CENTRAL-500, set coverage target = identity bar "
            "0.9213 (the +0.031 lawine #336 lift suffices); if it targets WORST-case (private-stable) "
            "500, set coverage target = c*_worst={:.4f}, i.e. +{:.4f} beyond the identity bar and "
            "+{:.4f} total from the honest 0.8903 prior (just past the +0.031 budget).".format(
                cw, d4["worst_500_shortfall"], d4["total_lift_for_worst_500_from_prior"])),
        "supply_caveat_fern335": (
            "coverage is the DEMAND axis ONLY. fern #335 (5pos499e) found the binding axis is SUPPLY: "
            "EVEN at c*_worst the supply floor must still revive (phi>=%.3f) -- coverage alone never "
            "reaches 500. c* is a NECESSARY demand-side target, not sufficient on its own." % SUPPLY_FLOOR_PHI),
        "greedy_safety": {
            "cov_ordering_card_is_cpu_analytic": True,
            "no_gpu": True,
            "no_served_change": True,
            "no_model_forward": True,
            "greedy_identity_preserved_by_construction": True,
            "argmax_note": ("coverage/E[T] is the SPEED axis: the draft acceptance rate changes HOW "
                            "FAST tokens are verified, not WHICH token is emitted -- the target "
                            "model's argmax remains the emitted token, so greedy identity is "
                            "invariant to c. No served file, kernel, or decode path is touched."),
        },
    }


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def _selftests(d1: dict, d2: dict, d3: dict, d4: dict, d5: dict) -> dict[str, Any]:
    cc = d2["c_star_central_for_500"]
    cw = d2["c_star_worst_for_500"]
    rt = d1["anchor_roundtrip"]
    h337 = d1["honest_corner_roundtrip_337"]
    conditions = {
        # (a) anchor round-trip: envelope_X(0.9213) == anchor (520.95 / 492.87) within tol.
        "a_anchor_roundtrip_identity": bool(
            rt["matches_central_anchor"] and rt["matches_worst_anchor"]
            and abs(rt["envelope_central_at_identity"] - CENTRAL_ANCHOR) <= TOL_EXACT
            and abs(rt["envelope_worst_at_identity"] - WORST_ANCHOR) <= TOL_EXACT),
        # (b) reproduce stark #337: envelope_X(0.8903) == 470.35 / 444.99 within tol.
        "b_reproduce_337_honest_corners": bool(
            h337["matches_337_central_470p35"] and h337["matches_337_worst_444p99"]
            and h337["both_below_500"]),
        # (c) E[T](c) and both envelopes monotone increasing in c (unique root).
        "c_monotone_increasing": bool(d1["monotone_increasing"]["all"]),
        # (d) c*_central solves envelope_central == 500 within tol.
        "d_central_root_solves_500": bool(
            d2["central_root_valid"] and abs(envelope_central(cc) - TARGET) <= TOL_ROOT),
        # (e) c*_worst solves envelope_worst == 500 within tol.
        "e_worst_root_solves_500": bool(
            d2["worst_root_valid"] and abs(envelope_worst(cw) - TARGET) <= TOL_ROOT),
        # (f) ordering consistent: worst anchor < central anchor => c*_worst > c*_central.
        "f_ordering_worst_gt_central": bool(WORST_ANCHOR < CENTRAL_ANCHOR and cw > cc),
        # (g) imported EXACT: constants match #337 banked AND round to displayed forms.
        "g_imports_exact": bool(
            abs(IDENTITY_BAR - 0.9213) <= TOL_DISPLAY_C
            and abs(COV_PRIOR - 0.8903) <= TOL_DISPLAY_C
            and abs(CENTRAL_ANCHOR - 520.95) <= TOL_DISPLAY_TPS
            and abs(WORST_ANCHOR - 492.87) <= TOL_DISPLAY_TPS
            and abs(E_T_AT_IDENTITY - 6.1112) <= 1e-3
            and abs(MU_CLIFF - 1.16981) <= 1e-4),
        # (h) NaN-clean (set by caller).
        "h_nan_clean": True,
        # (i) c*_central and c*_worst in (0,1) (valid coverage).
        "i_roots_valid_coverage": bool(d2["roots_in_unit_interval"]),
        # (j) [extra] anticipated ordering c*_central < 0.9213 < c*_worst CONFIRMED.
        "j_anticipated_ordering_confirmed": bool(d3["anticipated_ordering_confirmed"]),
        # (k) [extra] sufficiency booleans: central YES, worst NO.
        "k_sufficiency_central_yes_worst_no": bool(
            d3["identity_bar_suffices_for_central_500"]
            and not d3["identity_bar_suffices_for_worst_500"]),
        # (l) [extra] headroom > 0 and shortfall > 0 (strict ordering, both sides nonzero).
        "l_headroom_and_shortfall_positive": bool(
            d4["central_500_headroom"] > 0.0 and d4["worst_500_shortfall"] > 0.0),
        # (m) [extra] worst-500 exceeds the +0.031 retrain budget (total lift > 0.031).
        "m_worst_exceeds_retrain_budget": bool(not d4["worst_500_within_retrain_budget"]),
        # (n) [extra] inverse self-consistency: e_t(c*) reproduces the target E[T] = 500*E_ref/anchor.
        "n_inverse_self_consistent": bool(
            abs(e_t(cc) - d2["e_t_target_central"]) <= 1e-6
            and abs(e_t(cw) - d2["e_t_target_worst"]) <= 1e-6),
    }
    return {
        "conditions": conditions,
        "cov_envelope_ordering_self_test_passes": bool(all(conditions.values())),
        "n_checks": len(conditions),
        "detail": {
            "c_star_central": cc, "c_star_worst": cw, "identity_bar": IDENTITY_BAR,
            "central_headroom": d4["central_500_headroom"], "worst_shortfall": d4["worst_500_shortfall"],
        },
    }


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    d1 = deliverable1_envelope_map()
    d2 = deliverable2_inverse_solve()
    d3 = deliverable3_ordering(d2)
    d4 = deliverable4_headroom_shortfall(d2)
    d5 = deliverable5_subcliff_structure(d2)
    d6 = deliverable6_decision(d2, d3, d4)
    st = _selftests(d1, d2, d3, d4, d5)

    cc = d2["c_star_central_for_500"]
    cw = d2["c_star_worst_for_500"]
    handoff = (
        "the SPEED-500 envelope needs coverage c*_central={:.4f} (central) / c*_worst={:.4f} (worst), "
        "so the ordering is c*_central < 0.9213 < c*_worst -> clearing the strict identity bar buys "
        "central-500 but is NOT sufficient for worst-case 500 (shortfall {:.4f} vs lawine #336's "
        "+0.031 retrain budget, total lift {:.4f} just past it), and EVEN at c*_worst supply must "
        "still revive (fern #335, phi>=0.255) -- coverage alone never reaches 500.".format(
            cc, cw, d4["worst_500_shortfall"], d4["total_lift_for_worst_500_from_prior"]))

    headline = {
        "cov_envelope_ordering_self_test_passes": bool(st["cov_envelope_ordering_self_test_passes"]),  # PRIMARY
        "c_star_central_for_500": cc,                                                                   # TEST
        "c_star_worst_for_500": cw,                                                                     # REPORT
        "identity_bar": IDENTITY_BAR,
        "ordering_string": d3["ordering_string"],
        "anticipated_ordering_confirmed": d3["anticipated_ordering_confirmed"],
        "identity_bar_suffices_for_central_500": d3["identity_bar_suffices_for_central_500"],
        "identity_bar_suffices_for_worst_500": d3["identity_bar_suffices_for_worst_500"],
        "central_500_headroom": d4["central_500_headroom"],
        "worst_500_shortfall": d4["worst_500_shortfall"],
        "worst_500_within_retrain_budget": d4["worst_500_within_retrain_budget"],
        "total_lift_for_worst_500_from_prior": d4["total_lift_for_worst_500_from_prior"],
    }
    return {
        "headline": headline,
        "deliverable1_envelope_map": d1,
        "deliverable2_inverse_solve": d2,
        "deliverable3_ordering": d3,
        "deliverable4_headroom_shortfall": d4,
        "deliverable5_subcliff_structure": d5,
        "deliverable6_decision": d6,
        "self_test": st,
        "handoff": handoff,
        "imports": {
            "provenance": (
                "stark #337 lbuirkpt (E[T](c)=1+sum_{d=1..7} c^d chain law, anchors central 520.95 / "
                "worst 492.87 @E[T]=6.1112, honest corners 470.35/444.99 @0.8903, cliff mu=1.16981, "
                "mu_tie=1.1076) x lawine #330 hfrscdai (cov prior 0.8903, identity bar 0.9213) x "
                "lawine #336 krroookz (+0.031 retrain lift) x fern #335 5pos499e (supply floor "
                "phi>=0.255). All run-ids in wandb-applied-ai-team/gemma-challenge-senpai."),
            "caveats": [
                "INVERSE of stark #337: this re-prices NOTHING measured -- it inverts the SAME banked "
                "envelope map (anchors x E[T] lever ratio) to find the coverage c* at env=500. No "
                "EAGLE-3 fusion checkpoint runs here; NOT a running EagleProposer.",
                "the central anchor is CAP-BOUND at the lambda ceiling 520.95. Above the identity bar "
                "the central envelope is FLAT at the ceiling (a higher E[T] cannot exceed the cap), so "
                "envelope_central(c) scaling is the load-bearing model ONLY below the bar -- exactly "
                "where the 500 crossing (c*_central) lies. The worst corner is uncapped and scales "
                "throughout.",
                "c is per-depth effective acceptance c_eff (the E[T](c) axis), in the SAME units as "
                "the identity bar 0.9213 and cov prior 0.8903 and lawine #336's +0.031. The rank-cov "
                "map (D5) is color only.",
                "DEMAND axis only: c* is a NECESSARY coverage target; fern #335's SUPPLY floor "
                "(phi>=0.255) is the complementary binding constraint -- coverage alone never reaches "
                "500. NOT a launch / build / served-file change / HF Job / submission / open2.",
            ],
        },
    }


# --------------------------------------------------------------------------- #
# NaN guard + report + W&B.
# --------------------------------------------------------------------------- #
def _nan_paths(node: Any, p: str = "result") -> list[str]:
    bad: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            bad += _nan_paths(v, f"{p}.{k}")
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            bad += _nan_paths(v, f"{p}[{i}]")
    elif isinstance(node, float) and not math.isfinite(node):
        bad.append(p)
    return bad


def _print_report(syn: dict) -> None:
    h = syn["headline"]
    d1 = syn["deliverable1_envelope_map"]
    d2 = syn["deliverable2_inverse_solve"]
    d3 = syn["deliverable3_ordering"]
    d4 = syn["deliverable4_headroom_shortfall"]
    d5 = syn["deliverable5_subcliff_structure"]
    d6, st = syn["deliverable6_decision"], syn["self_test"]
    print("\n" + "=" * 98, flush=True)
    print("EAGLE-3 COVERAGE -> ENVELOPE ORDERING (PR #340, stark) — invert the speed-500 map", flush=True)
    print("=" * 98, flush=True)
    print("  (D1) ENVELOPE MAP  envelope_X(c) = X_ANCHOR * E[T](c) / E[T](0.9213)", flush=True)
    rt = d1["anchor_roundtrip"]
    h337 = d1["honest_corner_roundtrip_337"]
    print(f"      anchor round-trip @0.9213: central={rt['envelope_central_at_identity']:.2f} "
          f"worst={rt['envelope_worst_at_identity']:.2f}  (match {rt['matches_central_anchor']}/"
          f"{rt['matches_worst_anchor']})", flush=True)
    print(f"      #337 round-trip   @0.8903: central={h337['envelope_central_at_prior']:.2f} "
          f"worst={h337['envelope_worst_at_prior']:.2f}  (match {h337['matches_337_central_470p35']}/"
          f"{h337['matches_337_worst_444p99']})", flush=True)
    print(f"      monotone increasing in c: {d1['monotone_increasing']['all']}", flush=True)
    print("-" * 98, flush=True)
    print("  (D2) INVERSE SOLVE (PRIMARY)  env_X(c*) = 500", flush=True)
    print(f"      c*_central = {d2['c_star_central_for_500']:.6f}  (residual {d2['root_residual_central']:+.2e})", flush=True)
    print(f"      c*_worst   = {d2['c_star_worst_for_500']:.6f}  (residual {d2['root_residual_worst']:+.2e})", flush=True)
    print("-" * 98, flush=True)
    print("  (D3) ORDERING", flush=True)
    print(f"      {d2['c_star_central_for_500']:.4f} (c*_central)  <  {IDENTITY_BAR:.4f} (identity)  "
          f"<  {d2['c_star_worst_for_500']:.4f} (c*_worst)   confirmed={d3['anticipated_ordering_confirmed']}", flush=True)
    print(f"      identity suffices: central-500={d3['identity_bar_suffices_for_central_500']}  "
          f"worst-500={d3['identity_bar_suffices_for_worst_500']}", flush=True)
    print("-" * 98, flush=True)
    print("  (D4) HEADROOM / SHORTFALL  (lawine #336 budget +0.031)", flush=True)
    print(f"      central_500_headroom = {d4['central_500_headroom']:+.4f}   "
          f"worst_500_shortfall = {d4['worst_500_shortfall']:+.4f}", flush=True)
    print(f"      worst total lift from prior = {d4['total_lift_for_worst_500_from_prior']:.4f}  "
          f"(within +0.031 budget: {d4['worst_500_within_retrain_budget']}; "
          f"{100.0*d4['worst_shortfall_as_frac_of_retrain_budget']:.1f}% of budget)", flush=True)
    print("-" * 98, flush=True)
    print("  (D5) SUB-CLIFF STRUCTURE  (c* are head-coverage targets at W=4 step x1)", flush=True)
    print(f"      widening dominated (mu={d5['mu_cliff']:.5f} > mu_tie={d5['mu_tie']:.5f}): "
          f"{d5['widening_dominated']}", flush=True)
    rc = d5["rank_coverage_map"]
    print(f"      c*_worst rank-cov {rc['rankcov_for_c_star_worst']:.4f} > linear spine "
          f"{COV4_LINEAR:.4f}: {rc['worst_needs_rankcov_above_linear_spine']}", flush=True)
    print("-" * 98, flush=True)
    print(f"  HAND-OFF: {syn['handoff']}", flush=True)
    print("-" * 98, flush=True)
    print(f"  PRIMARY cov_envelope_ordering_self_test_passes = "
          f"{st['cov_envelope_ordering_self_test_passes']} ({st['n_checks']} checks)", flush=True)
    for k, v in st["conditions"].items():
        print(f"        - {k}: {v}", flush=True)
    print("=" * 98 + "\n", flush=True)


def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[eagle3-cov-envelope-ordering] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    d1 = syn["deliverable1_envelope_map"]
    d2 = syn["deliverable2_inverse_solve"]
    d3 = syn["deliverable3_ordering"]
    d4 = syn["deliverable4_headroom_shortfall"]
    d5 = syn["deliverable5_subcliff_structure"]
    st = syn["self_test"]
    run = init_wandb_run(
        job_type="eagle3-cov-envelope-ordering",
        agent="stark",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["eagle3-cov-envelope-ordering", "issue-192", "eagle3", "inverse-envelope-map",
              "coverage-ordering", "identity-bar", "compliant-500", "validity-gate",
              "bank-the-analysis"],
        config={
            "K_spec": K_SPEC, "identity_bar": IDENTITY_BAR, "cov_prior": COV_PRIOR,
            "central_anchor": CENTRAL_ANCHOR, "worst_anchor": WORST_ANCHOR,
            "e_t_at_identity": E_T_AT_IDENTITY, "mu_cliff": MU_CLIFF, "mu_tie": MU_TIE,
            "knee_Mstar": KNEE_MSTAR, "retrain_lift_budget_336": RETRAIN_LIFT_BUDGET,
            "supply_floor_phi_335": SUPPLY_FLOOR_PHI, "target": TARGET, "baseline_tps": BASELINE_TPS,
            "wandb_group": args.wandb_group,
            "source_runs": "stark#337(lbuirkpt), lawine#330(hfrscdai), lawine#336(krroookz), fern#335(5pos499e)",
        },
    )
    if run is None:
        print("[eagle3-cov-envelope-ordering] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "cov_envelope_ordering_self_test_passes": int(bool(st["cov_envelope_ordering_self_test_passes"])),  # PRIMARY
        "c_star_central_for_500": d2["c_star_central_for_500"],                                              # TEST
        "c_star_worst_for_500": d2["c_star_worst_for_500"],                                                  # REPORT
        "identity_bar": IDENTITY_BAR,
        "anticipated_ordering_confirmed": int(bool(d3["anticipated_ordering_confirmed"])),
        "identity_bar_suffices_for_central_500": int(bool(d3["identity_bar_suffices_for_central_500"])),
        "identity_bar_suffices_for_worst_500": int(bool(d3["identity_bar_suffices_for_worst_500"])),
        "central_500_headroom": d4["central_500_headroom"],
        "worst_500_shortfall": d4["worst_500_shortfall"],
        "total_lift_for_central_500_from_prior": d4["total_lift_for_central_500_from_prior"],
        "total_lift_for_worst_500_from_prior": d4["total_lift_for_worst_500_from_prior"],
        "worst_500_within_retrain_budget": int(bool(d4["worst_500_within_retrain_budget"])),
        "extra_lift_needed_beyond_retrain": d4["extra_lift_needed_beyond_retrain"],
        "worst_shortfall_as_frac_of_retrain_budget": d4["worst_shortfall_as_frac_of_retrain_budget"],
        "root_residual_central": d2["root_residual_central"],
        "root_residual_worst": d2["root_residual_worst"],
        "e_t_target_central": d2["e_t_target_central"],
        "e_t_target_worst": d2["e_t_target_worst"],
        "worst_over_central_tile_factor": d1["worst_over_central_tile_factor"],
        "rankcov_for_c_star_worst": d5["rank_coverage_map"]["rankcov_for_c_star_worst"],
        "rankcov_for_c_star_central": d5["rank_coverage_map"]["rankcov_for_c_star_central"],
        "worst_needs_rankcov_above_linear_spine": int(bool(
            d5["rank_coverage_map"]["worst_needs_rankcov_above_linear_spine"])),
        "widening_dominated": int(bool(d5["widening_dominated"])),
        "monotone_all": int(bool(d1["monotone_increasing"]["all"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="eagle3_cov_envelope_ordering_result", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    print(f"[eagle3-cov-envelope-ordering] wandb logged {len(summary)} keys", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="eagle3-cov-envelope-ordering")
    args = ap.parse_args(argv)

    syn = synthesize()

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 340, "agent": "stark",
        "kind": "eagle3-cov-envelope-ordering", "analysis_only": True, "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _nan_paths(payload)
    payload["nan_clean"] = not nan_paths
    syn["self_test"]["conditions"]["h_nan_clean"] = not nan_paths
    syn["self_test"]["cov_envelope_ordering_self_test_passes"] = bool(
        all(syn["self_test"]["conditions"].values()))
    syn["headline"]["cov_envelope_ordering_self_test_passes"] = syn["self_test"][
        "cov_envelope_ordering_self_test_passes"]
    if nan_paths:
        print(f"[eagle3-cov-envelope-ordering] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eagle3_cov_envelope_ordering_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[eagle3-cov-envelope-ordering] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = (syn["self_test"]["cov_envelope_ordering_self_test_passes"] and payload["nan_clean"])
        print(f"[eagle3-cov-envelope-ordering] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
