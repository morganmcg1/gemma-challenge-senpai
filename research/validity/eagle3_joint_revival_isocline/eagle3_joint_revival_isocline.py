#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Joint compliant-500 revival ISOCLINE (PR #341, fern) -- CPU-only analytic synthesis.

THE GOVERNING QUESTION (the CONTINUOUS map between #335's discrete corners)
---------------------------------------------------------------------------
fern #335 (`5pos499e`, MERGED) collapsed the joint compliant-500 feasibility to a 2x2 CORNER
table: only (supply revives to B, demand clears) = 520.95 is GO; the measured corner (C, miss) =
469.68 / 481.53 is NO-GO; p_both_revivals_land = 0.0 at the MEASURED corner. That answered "is the
measured corner alive" (no). It did NOT answer the CONTINUOUS question the human's #319 decision
needs: between the dead measured corner and the live GO corner, what is the LOCUS of
(phi_supply, Delta_cov_demand) pairs that hit exactly compliant-500 -- how far from dead are we,
and which axis is cheaper to move?

This card maps the joint revival ISOCLINE: the set of (phi, Delta_cov) where the compliant envelope
= 500, with two CONTINUOUS axes:
  * SUPPLY phi -- denken #327's deterministic-SDPA recovery fraction. Sets the TPS CEILING via the
    LINEAR BW-gap law: phi=0 -> 469.68 (full BI floor paid), phi=1 -> 520.95 (full recovery).
  * DEMAND Delta_cov -- lawine #336's coverage lift above the 0.8903 prior. Sets the E[T] FRACTION
    of that ceiling via stark #337's chain law E[T](c)=1+sum_{d=1..7} c^d, full ceiling realized at
    the identity-bar coverage 0.9213.
      envelope(phi, Delta_cov) = ceiling(phi) * E[T](0.8903 + Delta_cov) / E[T](0.9213)

  Deliverables: the isocline curve, the closest-to-origin feasible point (the minimum joint effort),
  the substitution rate dDelta_cov/dphi (how much demand-lift one unit of supply-recovery saves),
  and the distance from the (phi=0, Delta_cov=0) measured-dead corner to the isocline.

A LOAD-BEARING CORRECTION TO THE PR'S BREAK-EVEN ANCHOR (read before the self-test)
----------------------------------------------------------------------------------
The PR asks to "Validate envelope(phi=0.255, Delta_cov=0.031) ~ 500 (the GO break-even)" and to
have the isocline "pass through (0.255, 0.031)". Under the PR's OWN specified model this is FALSE:

    envelope(0.255, 0.031) = ceiling(0.255) * E[T](0.9213)/E[T](0.9213)
                           = ceiling(0.255) * 1.0
                           = 482.76   (== fern #335's banked `ceiling_at_breakeven_B_edge`),  NOT 500.

The two anchors that DO pin the model -- envelope(0,0)=469.68*0.9028~=424 and
envelope(1,0.031)=520.95 -- over-determine it (linear ceiling 469.68->520.95, demand fraction 1.0 at
the bar). Given those, envelope(0.255,0.031)=482.76 is FORCED; there is no model with the stated
functional forms that also puts 500 there.

ROOT CAUSE -- a THREE-axis conflation. fern #335's "rho=0.8038 internal 500 check" is a point on a
THIRD axis (the private-tax rho), NOT a (phi, Delta_cov) point:
    #335 500  =  min( HONEST_PUBLIC_611 * rho , LAMBDA_CEIL )  =  min(622.08 * 0.8038, 520.95) = 500
It lives at the FULL-B ceiling (phi=1 -> 520.95) with demand CLEAR (Delta_cov>=0.031) and the
private-tax rho at its break-even 0.8038. The supply break-even phi*=0.255 is only the THRESHOLD at
which #335's DISCRETE model flips supply C->B (and, being discrete, jumps straight to the full
520.95 ceiling). In THIS card's CONTINUOUS linear law the ceiling at phi=0.255 is only 482.76 -- a
mere 25.5% of the way up the floor-recovery -- so the supply break-even does NOT put you at 500.

This card therefore builds the model from the two CONSISTENT anchors, maps the TRUE (phi,Delta_cov)
500-isocline (which at Delta_cov=0.031 passes through phi=0.5925, NOT 0.255), reconciles #335's 500
as the rho-axis break-even at the full-B corner, and reports the discrepancy LOUDLY rather than
asserting a value the model cannot produce. The self-test validates the CORRECT round-trips.

LOCAL, CPU-ONLY, ANALYTIC. 0 GPU, no model forward, no publish, no served-file change, no HF Job, no
submission, no build, no launch. BASELINE stays 481.53; this card adds 0 TPS -- it maps the joint
(phi, Delta_cov) frontier from banked numbers. Imports fern #335 (5pos499e), denken #327 (kcjlr5ny),
lawine #336 (krroookz), stark #337 (lbuirkpt); re-derives nothing.

PRIMARY metric  joint_revival_isocline_self_test_passes
TEST    metric  min_joint_effort_to_500            (normalized distance origin->isocline)
REPORT          dDeltacov_dphi_at_closest          (supply<->demand substitution rate)
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
# IMPORTED ANCHORS -- cited EXACTLY, UNCHANGED, with source. This card re-derives none.
# --------------------------------------------------------------------------- #
# SUPPLY axis -- denken #327 (kcjlr5ny) via fern #335 (5pos499e). Linear BW-gap law.
CEILING_PHI0 = 469.6844761311386       # ceiling at phi=0 (full batch-invariant bf16 floor paid)
CEILING_PHI1 = 520.9527323111674       # = LAMBDA_CEIL int4-spec verify ceiling at phi=1 (full recovery)
PHI_STAR = 0.2549920813842095          # recovery break-even = 1 - budget/floor (supply C->B threshold)
PSI_STAR = 0.7450079186157905          # kept-slack break-even = budget/floor
CEILING_AT_PHI_STAR_335 = 482.757475483423   # fern #335 `ceiling_at_breakeven_B_edge` (round-trip target)

# DEMAND axis -- lawine #330 (hfrscdai) / lawine #336 (krroookz).
COV_PRIOR = 0.8902659519153152         # official-eval unconditional top-4 coverage prior ("0.8903")
COV_BAR = 0.9213011665456927           # lawine #316 regime-invariant build bar (the identity bar)
DELTA_COV_STAR = 0.031035214630377506  # demand break-even = COV_BAR - COV_PRIOR
DELTA_COV_RECIPE_CEIL = 0.06           # lawine #336 recipe band ceiling (high band +0.0595)
DELTA_COV_336_CENTRAL_COMBO = 0.0385   # lawine #336 soft-KD + reasoning-data central combination
DELTA_COV_336_BAND = (0.0175, 0.0595)  # lawine #336 recommended-combination band

# DEMAND -> ceiling fraction -- stark #337 (lbuirkpt) chain law (K=7, stark #331 convention).
K_DEPTH = 7
ET_BAR_337 = 6.111214987369919         # stark #337 E[T](0.9213) = build target 6.11 (cross-check)
ET_FUSION_8903_337 = 5.517578068867642 # stark #337 E[T](0.8903) banked (rounded-prior cross-check)
FRAC0_PR = 0.9029                      # PR's quoted "envelope x0.9029 at 0.8903" (rounded-prior frac)

# fern #335 (5pos499e) -- the rho-axis 500 (for the RECONCILIATION, NOT a (phi,Delta_cov) point).
HONEST_PUBLIC_611 = 622.080888         # = K_cal * realized_public_et (fern #325 honest public TPS)
RHO_INTERNAL_500 = 0.8037539966988988  # #335 bclear_at_breakeven_rho: 622.08*rho=500 at full-B ceiling
P_DEMAND_CLEARS_335 = 0.06031894029725235   # #335 P(official-eval uncond top-4 >= 0.9213)
SERVED = 481.53                        # deployed frontier (== #335 (B,miss) corner)
TARGET = 500.0

TOL = 1e-9
TOL_REPRO = 1e-6
TOL_TPS = 1e-3            # isocline / round-trip TPS tolerance


# --------------------------------------------------------------------------- #
# Model primitives (all imported; nothing re-measured).
# --------------------------------------------------------------------------- #
def et_chain(c: float) -> float:
    """stark #337 chain law: E[T](c) = 1 + sum_{d=1..K} c^d (committed-survivorship, uniform c_eff)."""
    return 1.0 + sum(c ** d for d in range(1, K_DEPTH + 1))


ET_BAR = et_chain(COV_BAR)             # full-ceiling E[T] reference (reproduces stark #337's 6.1112)


def ceiling(phi: float) -> float:
    """denken #327 linear BW-gap law: phi=0 -> 469.68 (full floor), phi=1 -> 520.95 (full recovery)."""
    return CEILING_PHI0 + phi * (CEILING_PHI1 - CEILING_PHI0)


def demand_fraction(delta_cov: float) -> float:
    """stark #337 E[T] fraction of the ceiling: full (1.0) at the identity-bar coverage 0.9213."""
    return et_chain(COV_PRIOR + delta_cov) / ET_BAR


def envelope(phi: float, delta_cov: float) -> float:
    """The two-axis joint compliant envelope: supply sets the ceiling, demand sets its E[T] fraction."""
    return ceiling(phi) * demand_fraction(delta_cov)


def invert_et(target_et: float, lo: float = 0.50, hi: float = 1.06) -> float:
    """Invert the (monotone-increasing) chain law: smallest c with E[T](c) >= target_et."""
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if et_chain(mid) < target_et:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# Isocline (envelope == TARGET) as a function of either axis.
def phi_on_isocline(delta_cov: float) -> float:
    """Supply phi needed to hit TARGET at a given demand-lift (may fall outside [0,1])."""
    need_ceiling = TARGET / demand_fraction(delta_cov)
    return (need_ceiling - CEILING_PHI0) / (CEILING_PHI1 - CEILING_PHI0)


def dcov_on_isocline(phi: float) -> float:
    """Demand-lift needed to hit TARGET at a given supply phi (may fall outside [0, recipe ceiling])."""
    need_frac = TARGET / ceiling(phi)
    return invert_et(need_frac * ET_BAR) - COV_PRIOR


# --------------------------------------------------------------------------- #
# (1) Two-axis envelope model -- validate the anchors + surface the corrected break-even.
# --------------------------------------------------------------------------- #
def envelope_model() -> dict[str, Any]:
    frac0 = demand_fraction(0.0)
    env_00 = envelope(0.0, 0.0)
    env_1_star = envelope(1.0, DELTA_COV_STAR)
    env_star_star = envelope(PHI_STAR, DELTA_COV_STAR)
    # the rho-axis 500 of #335 (a DIFFERENT object than the (phi,Delta_cov) plane).
    rho500 = min(HONEST_PUBLIC_611 * RHO_INTERNAL_500, CEILING_PHI1)
    return {
        "law": "envelope(phi, Delta_cov) = ceiling(phi) * E[T](0.8903 + Delta_cov) / E[T](0.9213)",
        "ceiling_phi0": CEILING_PHI0,
        "ceiling_phi1": CEILING_PHI1,
        "et_bar_reproduced": ET_BAR,
        "et_bar_matches_337": bool(abs(ET_BAR - ET_BAR_337) <= TOL_REPRO),
        "et_fusion_8903_reproduced": et_chain(0.8903),
        "et_fusion_8903_matches_337": bool(abs(et_chain(0.8903) - ET_FUSION_8903_337) <= TOL_REPRO),
        "frac0_exact_prior": frac0,                    # 0.90276 (exact 0.89026595 prior)
        "frac0_rounded_prior": et_chain(0.8903) / ET_BAR,  # 0.90286 (PR's "x0.9029")
        "frac0_matches_pr_0p9029": bool(abs(et_chain(0.8903) / ET_BAR - FRAC0_PR) <= 1e-3),
        # --- the three PR validation anchors ---
        "anchor_measured_dead_env_00": env_00,         # ~424 (PR: 469.68*0.9029~=424)  [CONSISTENT]
        "anchor_go_corner_env_1_star": env_1_star,     # 520.95 (PR: GO corner)          [CONSISTENT]
        "anchor_breakeven_env_star_star": env_star_star,   # 482.76 (PR claims ~500)     [INCONSISTENT]
        # --- the LOUD correction ---
        "pr_breakeven_assertion": "envelope(0.255, 0.031) ~ 500",
        "pr_breakeven_assertion_holds": bool(abs(env_star_star - 500.0) <= 1.0),   # FALSE
        "env_star_star_equals_335_b_edge": bool(
            abs(env_star_star - CEILING_AT_PHI_STAR_335) <= TOL_REPRO),
        "rho_axis_500_reconciled": rho500,             # 500.0 (the #335 rho-break-even, full-B ceiling)
        "rho_axis_500_matches": bool(abs(rho500 - 500.0) <= TOL_TPS),
        "correction_note": (
            "PR self-test (a)/(e) assert envelope(0.255, 0.031) ~ 500. The model gives "
            "{:.3f} == fern #335's banked ceiling_at_breakeven_B_edge, NOT 500. #335's '500' is the "
            "PRIVATE-TAX rho break-even at the FULL-B ceiling (min(622.08*0.8038, 520.95) = 500), a "
            "THIRD-axis point; phi*=0.255 is only the DISCRETE C->B threshold, and in this CONTINUOUS "
            "linear law ceiling(0.255)=482.76. The TRUE 500-isocline at Delta_cov=0.031 passes through "
            "phi={:.4f}.".format(env_star_star, phi_on_isocline(DELTA_COV_STAR))),
        "monotone_in_phi": _is_increasing(lambda p: envelope(p, 0.02),
                                          [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        "monotone_in_dcov": _is_increasing(lambda d: envelope(0.5, d),
                                           [0.0, 0.015, 0.03, 0.045, 0.06]),
    }


def _is_increasing(fn: Callable[[float], float], xs: list[float]) -> bool:
    ys = [fn(x) for x in xs]
    return all(ys[i + 1] > ys[i] for i in range(len(ys) - 1))


# --------------------------------------------------------------------------- #
# (2) Solve the isocline (PRIMARY deliverable) -- the (phi, Delta_cov) locus where envelope = 500.
# --------------------------------------------------------------------------- #
def solve_isocline() -> dict[str, Any]:
    dcov_grid = [0.0, 0.015, DELTA_COV_STAR, 0.045, DELTA_COV_RECIPE_CEIL]
    phi_grid = [0.0, 0.25, 0.5, 0.75, 1.0]

    by_dcov = []
    for d in dcov_grid:
        phi = phi_on_isocline(d)
        by_dcov.append({
            "delta_cov": d, "coverage": COV_PRIOR + d, "demand_fraction": demand_fraction(d),
            "phi_needed": phi, "feasible": bool(0.0 <= phi <= 1.0),
            "regime": ("phi>1 (supply alone cannot reach 500 even at full recovery)" if phi > 1.0
                       else "phi<0 (demand alone overshoots 500)" if phi < 0.0 else "feasible"),
        })
    by_phi = []
    for p in phi_grid:
        d = dcov_on_isocline(p)
        by_phi.append({
            "phi": p, "ceiling": ceiling(p), "delta_cov_needed": d,
            "coverage_needed": COV_PRIOR + d,
            "feasible": bool(0.0 <= d <= DELTA_COV_RECIPE_CEIL),
            "within_336_band": bool(DELTA_COV_336_BAND[0] <= d <= DELTA_COV_336_BAND[1]),
        })

    # Entry / exit of the feasible box [phi in 0..1] x [Delta_cov in 0..recipe ceil].
    dcov_at_phi1 = dcov_on_isocline(1.0)   # min Delta_cov on the feasible isocline (supply maxed)
    dcov_at_phi0 = dcov_on_isocline(0.0)   # max Delta_cov before supply goes negative (demand alone)
    phi_at_dcov_star = phi_on_isocline(DELTA_COV_STAR)   # the TRUE point at the demand break-even
    return {
        "objective": "envelope(phi, Delta_cov) = 500",
        "isocline_by_delta_cov": by_dcov,
        "isocline_by_phi": by_phi,
        "feasible_box": {"phi": [0.0, 1.0], "delta_cov": [0.0, DELTA_COV_RECIPE_CEIL]},
        "entry_supply_maxed": {"phi": 1.0, "delta_cov": dcov_at_phi1,
                               "coverage": COV_PRIOR + dcov_at_phi1},
        "exit_demand_only": {"phi": 0.0, "delta_cov": dcov_at_phi0,
                             "coverage": COV_PRIOR + dcov_at_phi0},
        "feasible_dcov_span": [dcov_at_phi1, dcov_at_phi0],
        "phi_at_demand_breakeven": phi_at_dcov_star,     # 0.5925 (NOT the PR's 0.255)
        "phi_at_demand_breakeven_ne_phi_star": bool(abs(phi_at_dcov_star - PHI_STAR) > 0.05),
        "note": (
            "The feasible 500-isocline runs from (phi=1, Delta_cov={:.4f}) [supply maxed, least demand] "
            "down to (phi=0, Delta_cov={:.4f}) [demand alone]. Below Delta_cov={:.4f} even full supply "
            "recovery cannot reach 500 (phi>1); above Delta_cov={:.4f} demand alone overshoots (phi<0). "
            "At the demand break-even Delta_cov=0.031 the isocline sits at phi={:.4f} -- the PR's "
            "claimed 0.255 is the C->B threshold, not the 500 crossing."
            .format(dcov_at_phi1, dcov_at_phi0, dcov_at_phi1, dcov_at_phi0, phi_at_dcov_star)),
    }


# --------------------------------------------------------------------------- #
# (3) Closest feasible point + margin (TEST) -- normalize each axis by its break-even.
# --------------------------------------------------------------------------- #
def normalized_dist2(phi: float, dcov: float) -> float:
    """Squared distance from origin with each axis in break-even units (phi/0.255, Delta_cov/0.031)."""
    return (phi / PHI_STAR) ** 2 + (dcov / DELTA_COV_STAR) ** 2


def closest_feasible_point() -> dict[str, Any]:
    # Coarse grid over the feasible phi span, then ternary refine (objective is unimodal along it).
    lo_phi, hi_phi = 0.0, 1.0
    best = None
    N = 4000
    for i in range(N + 1):
        phi = lo_phi + (hi_phi - lo_phi) * i / N
        dcov = dcov_on_isocline(phi)
        if dcov < 0.0 or dcov > DELTA_COV_RECIPE_CEIL:
            continue
        d2 = normalized_dist2(phi, dcov)
        if best is None or d2 < best[0]:
            best = (d2, phi, dcov)
    # ternary-search refine in a bracket around the grid minimum
    span = (hi_phi - lo_phi) / N
    a, b = max(0.0, best[1] - 2 * span), min(1.0, best[1] + 2 * span)

    def obj(phi: float) -> float:
        d = dcov_on_isocline(phi)
        if d < 0.0 or d > DELTA_COV_RECIPE_CEIL:
            return math.inf
        return normalized_dist2(phi, d)

    for _ in range(200):
        m1 = a + (b - a) / 3.0
        m2 = b - (b - a) / 3.0
        if obj(m1) < obj(m2):
            b = m2
        else:
            a = m1
    phi_c = 0.5 * (a + b)
    dcov_c = dcov_on_isocline(phi_c)
    dist = math.sqrt(normalized_dist2(phi_c, dcov_c))
    env_c = envelope(phi_c, dcov_c)

    # how much of EACH axis (in break-even units) the cheapest point spends.
    u = phi_c / PHI_STAR
    v = dcov_c / DELTA_COV_STAR
    # the two single-axis references for context.
    pure_demand = {"phi": 0.0, "delta_cov": dcov_on_isocline(0.0),
                   "norm_dist": math.sqrt(normalized_dist2(0.0, dcov_on_isocline(0.0)))}
    supply_alone_phi = phi_on_isocline(DELTA_COV_STAR)   # at demand break-even (still 0.59)
    return {
        "metric": "normalized distance origin->isocline, axes in break-even units (phi/0.255, dcov/0.031)",
        "min_joint_effort_to_500": dist,                 # TEST metric
        "closest_phi": phi_c,
        "closest_delta_cov": dcov_c,
        "closest_coverage": COV_PRIOR + dcov_c,
        "closest_envelope": env_c,
        "closest_on_isocline": bool(abs(env_c - TARGET) <= TOL_TPS),
        "supply_breakeven_units_spent": u,               # 0.395 (below 1 -> supply UNDER its break-even)
        "demand_breakeven_units_spent": v,               # 1.498 (above 1 -> demand OVER its break-even)
        "both_axes_move": bool(phi_c > 1e-4 and dcov_c > 1e-4),
        "demand_dominates": bool(v > u),
        "distance_from_measured_dead_corner": dist,      # origin == measured-dead (phi=0, dcov=0)
        "pure_demand_reference": pure_demand,            # phi=0 endpoint, dist 1.597
        "supply_at_demand_breakeven_phi": supply_alone_phi,
        "note": (
            "Cheapest 500 point: phi={:.4f} ({:.3f} supply break-even units) + Delta_cov={:.4f} "
            "({:.3f} demand break-even units), coverage {:.4f}, normalized distance {:.4f}. BOTH axes "
            "move, but DEMAND carries it ({:.3f} vs {:.3f} break-even units): supply spends LESS than "
            "its own break-even while demand spends ~1.5x its break-even. The pure-demand endpoint "
            "(phi=0, Delta_cov={:.4f}) sits at distance {:.4f}, only {:.1f}% farther -- the supply "
            "assist barely helps because phi*=0.255 buys only {:.1f} TPS of ceiling."
            .format(phi_c, u, dcov_c, v, COV_PRIOR + dcov_c, dist, v, u,
                    pure_demand["delta_cov"], pure_demand["norm_dist"],
                    100.0 * (pure_demand["norm_dist"] / dist - 1.0),
                    CEILING_AT_PHI_STAR_335 - CEILING_PHI0)),
    }


# --------------------------------------------------------------------------- #
# (4) Substitution rate dDelta_cov/dphi along the isocline at the closest point.
# --------------------------------------------------------------------------- #
def substitution_rate(phi_c: float) -> dict[str, Any]:
    eps = 1e-5
    slope = (dcov_on_isocline(phi_c + eps) - dcov_on_isocline(phi_c - eps)) / (2 * eps)
    # normalized substitution: demand break-even units saved per supply break-even unit spent.
    slope_norm = slope * (PHI_STAR / DELTA_COV_STAR)
    return {
        "dDeltacov_dphi_at_closest": slope,              # REPORT metric (raw)
        "dDeltacov_dphi_normalized": slope_norm,         # break-even units saved per be unit spent
        "trade_off_not_complement": bool(slope < 0.0),
        "cheaper_axis": "demand",
        "note": (
            "dDelta_cov/dphi = {:.5f} at the closest point: one FULL unit of supply recovery (phi: "
            "0->1) saves only {:.4f} of coverage-lift. In break-even units, spending 1 supply "
            "break-even buys back only {:.3f} demand break-even -- a losing trade (<1). So DEMAND is "
            "the cheaper axis to push: denken #327 found SDPA-recovery hard (the 9.451% BW gap) and "
            "phi*=0.255 buys only ~13 TPS of ceiling, while lawine #336 rated +0.031 demand-lift "
            "REACHABLE-MARGINAL. The negative slope confirms a substitution (trade-off), not a "
            "complement.".format(slope, abs(slope), abs(slope_norm))),
    }


# --------------------------------------------------------------------------- #
# (5) Feasibility verdict -- combine the cheapest joint point with the marginal probabilities.
# --------------------------------------------------------------------------- #
def feasibility_verdict(cp: dict[str, Any]) -> dict[str, Any]:
    dcov_c = cp["closest_delta_cov"]
    phi_c = cp["closest_phi"]
    demand_above_336_central = bool(dcov_c > DELTA_COV_336_CENTRAL_COMBO)
    demand_within_336_band = bool(DELTA_COV_336_BAND[0] <= dcov_c <= DELTA_COV_336_BAND[1])
    supply_below_cb_threshold = bool(phi_c < PHI_STAR)
    return {
        "cheapest_point": {"phi": phi_c, "delta_cov": dcov_c, "coverage": COV_PRIOR + dcov_c},
        "demand_above_336_central_combo": demand_above_336_central,  # 0.046 > 0.0385 -> optimistic edge
        "demand_within_336_band": demand_within_336_band,           # 0.046 in [0.0175, 0.0595] -> yes
        "supply_below_C_to_B_threshold": supply_below_cb_threshold,  # 0.10 < 0.255 -> no full revival needed
        "p_demand_clears_bar_335": P_DEMAND_CLEARS_335,             # 0.0603 (clears 0.9213 prior, #335)
        "supply_marginal_source": "denken #332 (in-flight): P(phi >= 0.255) UNMEASURED",
        "demand_marginal_source": "lawine #336 (krroookz): +0.031 REACHABLE-MARGINAL; #339 in-flight",
        "verdict": "REACHABLE-BUT-OPTIMISTIC",
        "reconcile_335_p_both": (
            "fern #335 reported p_both_revivals_land = 0.0 at the MEASURED corner (phi=0, Delta_cov=0). "
            "This isocline quantifies the climb: the measured-dead corner sits {:.4f} normalized "
            "break-even units from the live 500-isocline. The cheapest crossing is NOT the discrete "
            "(B, clear) corner -- it needs only phi={:.3f} (well below the C->B threshold 0.255) but "
            "Delta_cov={:.4f} (ABOVE lawine #336's central combination 0.0385, inside its optimistic "
            "band [0.0175, 0.0595]). So the burden falls almost entirely on DEMAND, near the top of "
            "its plausible range; supply contributes a small assist. Verdict REACHABLE-BUT-OPTIMISTIC: "
            "the 500 crossing is reachable iff demand lands at the optimistic edge (#336 rated this "
            "NOT a slam-dunk), with at most a marginal supply recovery -- it does NOT require the full "
            "C->B kernel revival #335's discrete corner implied."
            .format(cp["min_joint_effort_to_500"], phi_c, dcov_c)),
        "note": (
            "Measured-dead corner envelope(0,0)={:.2f} is the EAGLE-3 lane's own envelope at zero lift "
            "-- BELOW the 481.53 deployed fallback, i.e. at zero coverage-lift the EAGLE-3 build is "
            "worse than keeping the deployed system. The (phi, Delta_cov) plane drops the private-tax "
            "rho axis #335 carried; the 500 here is realized public TPS, the rho>=0.8038 robustness "
            "gate is a SEPARATE check that still applies on top.".format(envelope(0.0, 0.0))),
    }


# --------------------------------------------------------------------------- #
# (6) Greedy-safety note.
# --------------------------------------------------------------------------- #
def greedy_safety() -> dict[str, Any]:
    return {
        "isocline_card_is_cpu_analytic": True,
        "gpu_used": False,
        "served_file_changed": False,
        "supply_axis_greedy_safe": "BI int4 verify is bit-exact (verify argmax preserved)",
        "demand_axis_greedy_safe": "EAGLE-3 emission is verify-gated (drafter proposes, target verifies argmax)",
        "note": "Both axes preserve greedy-identity by construction; this card maps a frontier, it "
                "changes no served bytes, runs no model forward, and spends no GPU or HF quota.",
    }


# --------------------------------------------------------------------------- #
# Synthesis (pure: no time, no randomness -> deterministic).
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    model = envelope_model()
    iso = solve_isocline()
    cp = closest_feasible_point()
    sub = substitution_rate(cp["closest_phi"])
    feas = feasibility_verdict(cp)
    greedy = greedy_safety()

    handoff = (
        "the joint compliant-500 revival isocline does NOT pass through (phi=0.255, Delta_cov=0.031) "
        "[that point = 482.76 = #335's B-edge ceiling, not 500 -- the PR conflated #335's rho-axis 500 "
        "with this plane]; it passes through (phi={:.4f}, Delta_cov=0.031). The cheapest feasible point "
        "demands normalized joint effort {:.4f} (phi={:.4f}, Delta_cov={:.4f}, coverage {:.4f}), the "
        "substitution rate is dDelta_cov/dphi={:.5f} (so DEMAND is the cheaper axis to push -- supply's "
        "phi*=0.255 buys only ~13 TPS of ceiling), and given denken #332's pending P(phi>=0.255) and "
        "lawine #336's REACHABLE-MARGINAL +0.031, the closest feasible point is REACHABLE-BUT-OPTIMISTIC "
        "(demand at Delta_cov={:.4f} sits above #336's central 0.0385, inside its optimistic band) -- "
        "the measured-dead corner sits {:.4f} break-even units from the live GO region."
        .format(iso["phi_at_demand_breakeven"], cp["min_joint_effort_to_500"], cp["closest_phi"],
                cp["closest_delta_cov"], cp["closest_coverage"], sub["dDeltacov_dphi_at_closest"],
                cp["closest_delta_cov"], cp["min_joint_effort_to_500"]))

    headline = {
        "min_joint_effort_to_500": cp["min_joint_effort_to_500"],            # TEST
        "dDeltacov_dphi_at_closest": sub["dDeltacov_dphi_at_closest"],       # REPORT
        "closest_phi": cp["closest_phi"],
        "closest_delta_cov": cp["closest_delta_cov"],
        "closest_coverage": cp["closest_coverage"],
        "phi_at_demand_breakeven": iso["phi_at_demand_breakeven"],
        "cheaper_axis": sub["cheaper_axis"],
        "feasibility_verdict": feas["verdict"],
        "env_at_pr_breakeven_0p255_0p031": model["anchor_breakeven_env_star_star"],   # 482.76 (NOT 500)
        "pr_breakeven_assertion_holds": model["pr_breakeven_assertion_holds"],         # False
        "env_measured_dead_00": model["anchor_measured_dead_env_00"],
        "env_go_corner_1_star": model["anchor_go_corner_env_1_star"],
    }
    return {
        "headline": headline,
        "envelope_model": model,
        "isocline": iso,
        "closest_point": cp,
        "substitution": sub,
        "feasibility": feas,
        "greedy_safety": greedy,
        "handoff": handoff,
        "imports": {
            "provenance": (
                "fern #335 5pos499e (2x2 corner table; ceiling_at_breakeven_B_edge 482.757; "
                "rho-break-even 0.8038 internal-500 at full-B; P(demand clears) 0.0603) x denken #327 "
                "kcjlr5ny (linear BW-gap law ceiling phi0 469.68 / phi1 520.95; recovery break-even "
                "phi*=0.255) x lawine #336 krroookz (cov prior 0.8903, bar 0.9213, Delta_cov* 0.031, "
                "REACHABLE-MARGINAL, central combo 0.0385, band [0.0175,0.0595]) x stark #337 lbuirkpt "
                "(E[T](c)=1+sum_{d=1..7} c^d; E[T](0.9213)=6.1112; E[T](0.8903)=5.5176). All run-ids in "
                "wandb-applied-ai-team/gemma-challenge-senpai."),
        },
    }


# --------------------------------------------------------------------------- #
# (7) Self-test (PRIMARY) -- validates the CORRECT model; documents the PR break-even discrepancy.
# --------------------------------------------------------------------------- #
def self_test(syn: dict[str, Any]) -> dict[str, Any]:
    model, iso, cp = syn["envelope_model"], syn["isocline"], syn["closest_point"]
    sub = syn["substitution"]
    c: dict[str, bool] = {}

    # (a-corrected) envelope(0.255, 0.031) reproduces #335's B-edge ceiling 482.76 (NOT 500).
    c["01_env_breakeven_reproduces_335_b_edge_482p76"] = bool(
        abs(model["anchor_breakeven_env_star_star"] - CEILING_AT_PHI_STAR_335) <= TOL_REPRO
        and model["env_star_star_equals_335_b_edge"]
        and not model["pr_breakeven_assertion_holds"])           # the PR's ~500 does NOT hold
    # (b) envelope(1, 0.031) = 520.95 (GO corner) within tol.
    c["02_env_go_corner_520p95"] = bool(
        abs(model["anchor_go_corner_env_1_star"] - CEILING_PHI1) <= TOL_TPS)
    # (c) envelope(0,0) = 469.68 * E[T](0.8903)/E[T](0.9213) ~ 424 (measured-dead).
    c["03_env_measured_dead_424"] = bool(
        abs(model["anchor_measured_dead_env_00"] - CEILING_PHI0 * model["frac0_exact_prior"]) <= TOL_REPRO
        and 423.0 <= model["anchor_measured_dead_env_00"] <= 425.0
        and model["frac0_matches_pr_0p9029"])
    # (d) envelope monotone increasing in BOTH phi and Delta_cov.
    c["04_envelope_monotone_both_axes"] = bool(model["monotone_in_phi"] and model["monotone_in_dcov"])
    # (e) E[T] chain law reproduces stark #337's anchors (6.1112 at bar; 5.5176 at 0.8903).
    c["05_et_chain_reproduces_337"] = bool(
        model["et_bar_matches_337"] and model["et_fusion_8903_matches_337"])
    # (e-corrected) the isocline passes through (phi=0.5925, 0.031) -- the TRUE point, NOT 0.255.
    phi_be = iso["phi_at_demand_breakeven"]
    c["06_isocline_through_true_point_not_0p255"] = bool(
        abs(envelope(phi_be, DELTA_COV_STAR) - TARGET) <= TOL_TPS
        and iso["phi_at_demand_breakeven_ne_phi_star"])
    # (f) closest-point normalized distance >= 0 and lies ON the isocline.
    c["07_closest_on_isocline_nonneg"] = bool(
        cp["min_joint_effort_to_500"] >= 0.0 and cp["closest_on_isocline"])
    # (g) substitution rate dDelta_cov/dphi < 0 (trade-off, not complement).
    c["08_substitution_negative"] = bool(sub["trade_off_not_complement"]
                                         and sub["dDeltacov_dphi_at_closest"] < 0.0)
    # (h) constants imported EXACT.
    c["09_constants_imported_exact"] = bool(
        abs(PHI_STAR - 0.2549920813842095) <= TOL and abs(CEILING_PHI0 - 469.6844761311386) <= TOL
        and abs(CEILING_PHI1 - 520.9527323111674) <= TOL and abs(COV_PRIOR - 0.8902659519153152) <= TOL
        and abs(COV_BAR - 0.9213011665456927) <= TOL and abs(SERVED - 481.53) <= TOL)
    # (i) NaN-clean (filled by caller).
    c["10_nan_clean"] = True
    # (j) reconcile #335: the rho=0.8038 internal-500 is the full-B rho break-even (a DIFFERENT 500).
    c["11_reconcile_335_rho_500_distinct"] = bool(
        model["rho_axis_500_matches"]                           # 622.08*0.8038 capped = 500
        and abs(model["rho_axis_500_reconciled"] - model["anchor_breakeven_env_star_star"]) > 1.0)  # != 482.76
    # determinism: two pure syntheses are identical.
    a = json.dumps(synthesize(), sort_keys=True)
    b = json.dumps(synthesize(), sort_keys=True)
    c["12_determinism_two_runs_identical"] = bool(a == b)
    # explicit: the PR break-even discrepancy is REAL and documented (defensive: |482.76 - 500| > 1).
    c["13_pr_breakeven_discrepancy_documented"] = bool(
        abs(model["anchor_breakeven_env_star_star"] - 500.0) > 1.0
        and not model["pr_breakeven_assertion_holds"])

    gate = all(bool(v) for v in c.values())
    return {"joint_revival_isocline_self_test_passes": gate, "checks": c}


# --------------------------------------------------------------------------- #
# NaN-clean walk.
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


# --------------------------------------------------------------------------- #
# Console report.
# --------------------------------------------------------------------------- #
def _print_report(syn: dict, st: dict) -> None:
    model, iso, cp = syn["envelope_model"], syn["isocline"], syn["closest_point"]
    sub, feas = syn["substitution"], syn["feasibility"]
    print("\n" + "=" * 100, flush=True)
    print("JOINT COMPLIANT-500 REVIVAL ISOCLINE (PR #341, fern) -- continuous (phi, Delta_cov) frontier",
          flush=True)
    print("=" * 100, flush=True)
    print("  (1) TWO-AXIS ENVELOPE MODEL  envelope = ceiling(phi) * E[T](0.8903+Dcov)/E[T](0.9213)",
          flush=True)
    print(f"      anchors: env(0,0)={model['anchor_measured_dead_env_00']:.2f} (~424, measured-dead) | "
          f"env(1,0.031)={model['anchor_go_corner_env_1_star']:.2f} (GO corner)", flush=True)
    print(f"      *** CORRECTION: env(0.255,0.031)={model['anchor_breakeven_env_star_star']:.2f} "
          f"== #335 B-edge ceiling, NOT 500 (PR self-test a/e conflate #335's rho-axis 500) ***",
          flush=True)
    print(f"      #335 rho-500 reconciled: min(622.08*0.8038, 520.95)="
          f"{model['rho_axis_500_reconciled']:.2f} at the FULL-B ceiling (phi=1), a third-axis point",
          flush=True)
    print("-" * 100, flush=True)
    print("  (2) ISOCLINE envelope=500 (phi needed at each Delta_cov):", flush=True)
    for row in iso["isocline_by_delta_cov"]:
        flag = "OK " if row["feasible"] else "XX "
        print(f"      {flag}Delta_cov={row['delta_cov']:.4f} cov={row['coverage']:.4f} "
              f"phi_needed={row['phi_needed']:+.4f}  ({row['regime']})", flush=True)
    print(f"      feasible span: phi=1 -> Delta_cov={iso['entry_supply_maxed']['delta_cov']:.4f} ... "
          f"phi=0 -> Delta_cov={iso['exit_demand_only']['delta_cov']:.4f}; demand break-even 0.031 -> "
          f"phi={iso['phi_at_demand_breakeven']:.4f}", flush=True)
    print("-" * 100, flush=True)
    print(f"  (3) CLOSEST FEASIBLE POINT (normalized phi/0.255, Delta_cov/0.031):", flush=True)
    print(f"      phi={cp['closest_phi']:.4f} ({cp['supply_breakeven_units_spent']:.3f} supply-be) + "
          f"Delta_cov={cp['closest_delta_cov']:.4f} ({cp['demand_breakeven_units_spent']:.3f} demand-be), "
          f"cov={cp['closest_coverage']:.4f}", flush=True)
    print(f"      min_joint_effort_to_500 = {cp['min_joint_effort_to_500']:.4f}  (TEST)  | "
          f"envelope at point = {cp['closest_envelope']:.3f}", flush=True)
    print("-" * 100, flush=True)
    print(f"  (4) SUBSTITUTION  dDelta_cov/dphi = {sub['dDeltacov_dphi_at_closest']:.5f}  "
          f"(normalized {sub['dDeltacov_dphi_normalized']:.4f}) -> cheaper axis = {sub['cheaper_axis']}",
          flush=True)
    print("-" * 100, flush=True)
    print(f"  (5) FEASIBILITY: {feas['verdict']}  (demand {cp['closest_delta_cov']:.4f} vs #336 central "
          f"0.0385; supply {cp['closest_phi']:.3f} vs C->B threshold 0.255)", flush=True)
    print("-" * 100, flush=True)
    print(f"  PRIMARY joint_revival_isocline_self_test_passes = "
          f"{st['joint_revival_isocline_self_test_passes']}", flush=True)
    for k, v in st["checks"].items():
        print(f"        - {k}: {v}", flush=True)
    print("=" * 100, flush=True)
    print(f"\n  HAND-OFF: {syn['handoff']}\n", flush=True)


# --------------------------------------------------------------------------- #
# W&B logging.
# --------------------------------------------------------------------------- #
def _maybe_log_wandb(args, payload: dict) -> str | None:
    if getattr(args, "no_wandb", False) or not getattr(args, "wandb_name", None):
        return None
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[isocline] wandb logging unavailable: {exc}", flush=True)
        return None

    syn, st = payload["synthesis"], payload["self_test"]
    model, iso, cp = syn["envelope_model"], syn["isocline"], syn["closest_point"]
    sub, feas = syn["substitution"], syn["feasibility"]
    run = init_wandb_run(
        job_type="eagle3-joint-revival-isocline",
        agent="fern",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["eagle3-joint-revival-isocline", "issue-192", "eagle3", "isocline",
              "supply-demand-substitution", "supply-floor-327", "demand-coverage-336",
              "et-curve-337", "validity-gate", "zero-tps", "bank-the-analysis"],
        config={
            "pr": 341, "analysis_only": True,
            "ceiling_phi0": CEILING_PHI0, "ceiling_phi1": CEILING_PHI1, "phi_star": PHI_STAR,
            "cov_prior": COV_PRIOR, "cov_bar": COV_BAR, "delta_cov_star": DELTA_COV_STAR,
            "k_depth": K_DEPTH, "target": TARGET, "served": SERVED, "rho_internal_500": RHO_INTERNAL_500,
            "wandb_group": args.wandb_group,
            "source_runs": "fern#335 5pos499e, denken#327 kcjlr5ny, lawine#336 krroookz, stark#337 lbuirkpt",
        },
    )
    if run is None:
        print("[isocline] wandb: no run (no WANDB_API_KEY/mode) -- skipping", flush=True)
        return None

    summary: dict[str, Any] = {
        "joint_revival_isocline_self_test_passes":
            int(bool(st["joint_revival_isocline_self_test_passes"])),          # PRIMARY
        "min_joint_effort_to_500": cp["min_joint_effort_to_500"],              # TEST
        "dDeltacov_dphi_at_closest": sub["dDeltacov_dphi_at_closest"],         # REPORT
        "dDeltacov_dphi_normalized": sub["dDeltacov_dphi_normalized"],
        # closest point
        "closest_phi": cp["closest_phi"],
        "closest_delta_cov": cp["closest_delta_cov"],
        "closest_coverage": cp["closest_coverage"],
        "closest_envelope": cp["closest_envelope"],
        "supply_breakeven_units_spent": cp["supply_breakeven_units_spent"],
        "demand_breakeven_units_spent": cp["demand_breakeven_units_spent"],
        "demand_dominates": int(bool(cp["demand_dominates"])),
        # isocline geometry
        "phi_at_demand_breakeven": iso["phi_at_demand_breakeven"],
        "feasible_dcov_min_at_phi1": iso["entry_supply_maxed"]["delta_cov"],
        "feasible_dcov_max_at_phi0": iso["exit_demand_only"]["delta_cov"],
        # envelope anchors + the correction
        "env_measured_dead_00": model["anchor_measured_dead_env_00"],
        "env_go_corner_1_star": model["anchor_go_corner_env_1_star"],
        "env_at_pr_breakeven_0p255_0p031": model["anchor_breakeven_env_star_star"],   # 482.76
        "pr_breakeven_assertion_holds": int(bool(model["pr_breakeven_assertion_holds"])),  # 0
        "env_star_star_equals_335_b_edge": int(bool(model["env_star_star_equals_335_b_edge"])),
        "rho_axis_500_reconciled": model["rho_axis_500_reconciled"],
        "frac0_exact_prior": model["frac0_exact_prior"],
        "et_bar_reproduced": model["et_bar_reproduced"],
        # feasibility
        "p_demand_clears_bar_335": feas["p_demand_clears_bar_335"],
        "demand_above_336_central_combo": int(bool(feas["demand_above_336_central_combo"])),
        "supply_below_C_to_B_threshold": int(bool(feas["supply_below_C_to_B_threshold"])),
        "feasibility_reachable_but_optimistic": int(feas["verdict"] == "REACHABLE-BUT-OPTIMISTIC"),
        # hygiene
        "nan_clean": int(bool(payload["nan_clean"])), "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v)) for k, v in st["checks"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="eagle3_joint_revival_isocline_result",
                      artifact_type="validity", data=payload)
    rid = getattr(run, "id", None)
    print(f"[isocline] wandb run: {getattr(run, 'url', rid)}", flush=True)
    finish_wandb(run)
    return rid


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="eagle3-joint-revival-isocline")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    syn = synthesize()
    st = self_test(syn)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 341, "agent": "fern",
        "kind": "eagle3-joint-revival-isocline", "analysis_only": True,
        "synthesis": syn, "self_test": st,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _nan_paths(payload)
    payload["nan_clean"] = not nan_paths
    st["checks"]["10_nan_clean"] = not nan_paths
    st["joint_revival_isocline_self_test_passes"] = all(bool(v) for v in st["checks"].values())
    if nan_paths:
        print(f"[isocline] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn, st)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eagle3_joint_revival_isocline_results.json"

    rid = _maybe_log_wandb(args, payload)
    payload["wandb_run_id"] = rid
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[isocline] wrote {out_path}  (wandb run {rid})", flush=True)

    print(f"  PRIMARY joint_revival_isocline_self_test_passes = "
          f"{st['joint_revival_isocline_self_test_passes']}", flush=True)
    print(f"  TEST min_joint_effort_to_500 = {syn['headline']['min_joint_effort_to_500']:.4f}", flush=True)
    print(f"  REPORT dDeltacov_dphi_at_closest = "
          f"{syn['headline']['dDeltacov_dphi_at_closest']:.5f}", flush=True)

    if args.self_test:
        ok = st["joint_revival_isocline_self_test_passes"] and payload["nan_clean"]
        print(f"[isocline] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
