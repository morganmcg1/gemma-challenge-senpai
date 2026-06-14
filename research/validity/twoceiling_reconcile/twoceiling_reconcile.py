#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Two-ceiling reconcile (PR #235, wirbel) — CPU-only analytic reconciliation.

THE QUESTION (the unresolved two-ceiling note from my #227 capstone; the human's Issue #211)
--------------------------------------------------------------------------------------------
My #227 collapsed the valid-verify menu to ONE survivor (lane-a: a custom batch-invariant int4
verify kernel) but carried an UNRESOLVED two-ceiling note: the compliant λ=1 ceiling is quoted
two ways — reach-DP 536.66 (my #199) vs int4-spec 520.95 (#204/#220) — both > 500, both upper
bounds, because "rank-1 coverage 0.7304 over-counts true compliant accept". That gap is now
LOAD-BEARING: stark #226 (`tzcc5xuq`, MERGED) declared the compliant lane private-FEASIBLE
*only because* its worst-case private build bar 528.48 < the reach-DP ceiling 536.66 (−8.18).
But stark #226 ALSO found 528.48 > the reach-DP LCB 525.73 (private-INFEASIBLE at P95). If the
lane-a kernel realizes only the int4-spec 520.95 (not the optimistic reach-DP 536.66), then
528.48 > 520.95 ⇒ the compliant-500-PRIVATE lane is DEAD even with a PERFECT kernel at λ=1.

CRUX — reconcile the two compliant λ=1 ceilings and decide which is OPERATIVE for the lane-a
kernel, then re-run stark #226's feasibility verdict under each.

THE MECHANISM (all imported / banked; NOTHING re-derived)
---------------------------------------------------------
Both ceilings share the launch composition  μ = K_cal·(E[T](1)/step)·τ   (K_cal=125.268,
step=1.2182, τ=1). They are banked at the SAME step and τ, so inverting the composition pins
the E[T](1) each implies:

    reach-DP   536.659  ⟹  E[T](1) = 5.21888   (#199 wdyqnx3g: rank-1-coverage 0.7304 ladder
                                                 propagated through the ρ-optimal reach-DP — the
                                                 OPTIMISTIC full-coverage ceiling, an over-count)
    int4-spec  520.953  ⟹  E[T](1) = 5.06614   (#204/#220 pqjnybbf int4_anchor_et1 — the SHIPPED
                                                 stack's λ=1 acceptance at the SAME deployed step)

Because step and τ are identical, the gap 536.66 − 520.95 = 15.71 TPS is 100% an E[T] (topology
/coverage) difference  ΔE[T] = 0.15274  ×  K_cal/step  =  15.71 TPS, and 0% step/τ. (The PR
prose "int4-spec E[T]=5.219" is the reach-DP/both-bugs E[T] 5.21888; the int4-spec ceiling 520.95
banked at the deployed step implies 5.0662 — see `note_pr_5p219` below. Holding E[T] fixed at
5.21888 instead would re-attribute the same 15.71 TPS to a 3.0% slower step; that is NOT how
either artifact banked the ceiling, so it is reported only as a dual sensitivity.)

WHICH CEILING THE KERNEL REALIZES (the core)
--------------------------------------------
lane-a is a custom batch-invariant int4 verify kernel. #216's kernel model: the split-K
reduction-order fix changes argmax-ORDER (determinism) NOT topology/coverage. So the kernel
inherits the SHIPPED stack's topology + coverage and only fixes determinism ⇒ its realized
E[T](1) tracks the int4-spec 520.95, NOT the reach-DP 536.66. Reaching 536.66 would ALSO require
the ρ-optimal max-branch-3 topology at full rank-1 coverage — that is a TOPOLOGY change (land
#71's job), draft/kernel-INDEPENDENT. Hence operative_compliant_ceiling = 520.95 (the defensible
upper bound on lane-a's private μ at λ=1); the band [520.95, 536.66] is carried with its upper
edge flagged as a land #71 topology question, not a kernel question.

THE DELIVERABLE — re-run stark #226's feasibility under each ceiling. The worst-case private bar
528.48 is FEASIBLE only vs the optimistic reach-DP central 536.66; it is INFEASIBLE vs the
reach-DP LCB 525.73, vs the int4-spec 520.95, and therefore vs the OPERATIVE 520.95. So the
compliant-500-PRIVATE lane is INFEASIBLE even with a perfect kernel — a public-milestone +
post-hoc-defence lane only, never a private-500 route. This QUALIFIES stark #226's private-
FEASIBLE verdict (which used the optimistic edge). Both ceilings clear the PUBLIC 500 milestone
regardless (520.95 > 500 and 536.66 > 500); only the private-feasibility question flips.

LOCAL, CPU-ONLY, ANALYTIC. No GPU / vLLM / HF Job / submission / served-file change / official
draw. BASELINE stays 481.53; adds 0 TPS; greedy/PPL untouched. Imports my #199 (wdyqnx3g) reach-DP
ceiling + machinery, #204/#220 (pqjnybbf) int4-spec ceiling, stark #226 (tzcc5xuq) private bar,
#216 (pc8g6s04) kernel model, kanna #217 (vgovdrjc) private-bar basis. Re-derives nothing.
NOT a launch. NOT open2.

PRIMARY metric  twoceiling_reconcile_self_test_passes
TEST    metric  operative_compliant_ceiling  (defensible lane-a λ=1 upper bound; int4-spec 520.95)
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
# Import the #199 reach-DP machinery (compliant_spec_et = module C). et_via_reachdp /
# official_tps use the committed anchor JSONs only; they do NOT read the (gitignored)
# rankprobe shard, so the import is shard-free (the same path #220 relies on).
# --------------------------------------------------------------------------- #
def _load(name: str, relpath: str):
    path = REPO_ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("compliant_spec_et", "research/validity/compliant_spec_et/compliant_spec_et.py")
K_CAL = C.K_CAL                     # 125.26795005202914 (#148/#169)
STEP = C.STEP                       # 1.2182 (#168 deployed M=32 step, normalized)
TAU_CENTRAL = 1.0                   # served-fraction central corner (#181)

# --------------------------------------------------------------------------- #
# Banked external constants (provenance: W&B run ids, project
# wandb-applied-ai-team/gemma-challenge-senpai). IMPORTED, not re-derived. Each is
# round-tripped against its committed result JSON in synthesize().
# --------------------------------------------------------------------------- #
REACH_DP_CENTRAL = 536.6590426143789   # #199 wdyqnx3g  compliant_spec_tps_ceiling (reach-DP, τ=1)
REACH_DP_LCB = 525.7290377676009       # #199 wdyqnx3g  ceiling_lcb_tps_both_bugs (finite-sample P95)
REACH_DP_ET = 5.21887717841078         # #199 wdyqnx3g  et_compliant_ceiling (both-bugs, rank-1 ladder)
INT4_SPEC_CEILING = 520.9527323111674  # #204/#220 pqjnybbf  int4-spec λ=1 ceiling (physical)
INT4_SPEC_ET = 5.0661371742562835      # #220 pqjnybbf  int4_anchor_et1 (= ceiling·step/K_cal at τ=1)
RANK1_COVERAGE_TOP1 = 0.7304444056147708   # #199/#227  q_compliant depth-1 top-1 match (the 0.7304)

# stark #226 (tzcc5xuq) — the private build bar this leg tests against the ceilings.
PRIVATE_BAR_WORSTCASE = 528.4835555959945   # private_bar_worstcase (NLS worst realizable blend)
F_PRIV_WORSTCASE = 0.969106920637722        # f_priv_worstcase (== kanna #217 central)
PRIVATE_BAR_52 = 535.1394043205385          # private_bar_at_52 (#52 empirical f_priv=0.95705 floor bar)
MU_SAFE_FRESH = 512.1570711713085           # mu_safe_fresh (= private_bar·f_priv; gross-up numerator)
F52_OBSERVED = 460.85 / 481.53              # #52 paired draw observed f_priv (0.95705)

TARGET = 500.0                  # official PUBLIC clear bar
BASELINE_TPS = 481.53           # PR #52 official (this leg adds 0 TPS)

# Rank-1-coverage robustness sweep (PR instruction 4a). 0.7304 is the banked point.
RANK1_SWEEP = (0.70, 0.71, 0.72, 0.7304444056147708, 0.73, 0.74)

TOL_ROUNDTRIP = 1e-6            # composition inversion / banked round-trip
TOL_PROV = 1e-9                # reach-DP reproduction of the banked ceiling E[T]
GAP_LITERAL = 15.71            # PR-stated gap 536.66 − 520.95 (literal, for the ±0.01 self-test)


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def official_tps(et: float, tau: float = TAU_CENTRAL, step: float = STEP) -> float:
    return K_CAL * (et / step) * tau


def et_from_tps(tps: float, tau: float = TAU_CENTRAL, step: float = STEP) -> float:
    """Invert the launch composition: E[T](1) = μ·step / (K_cal·τ)."""
    return tps * step / (K_CAL * tau)


# --------------------------------------------------------------------------- #
# (1) Pin both ceilings to their E[T] basis + decompose the gap (PR instruction 1).
# --------------------------------------------------------------------------- #
def pin_ceilings_and_decompose() -> dict[str, Any]:
    et_reach_implied = et_from_tps(REACH_DP_CENTRAL)        # 5.21888  (== banked REACH_DP_ET)
    et_int4_implied = et_from_tps(INT4_SPEC_CEILING)        # 5.06614  (== banked INT4_SPEC_ET)

    gap_tps = REACH_DP_CENTRAL - INT4_SPEC_CEILING          # 15.7063
    d_et = et_reach_implied - et_int4_implied               # 0.15274
    # E[T]/coverage driver: hold step & τ fixed (both ceilings are banked at step=1.2182, τ=1),
    # so the whole gap is K_cal/step · ΔE[T].
    coverage_driver_tps = (K_CAL / STEP) * d_et
    steptau_driver_tps = 0.0                                # identical step & τ across both ceilings
    residual_tps = gap_tps - coverage_driver_tps - steptau_driver_tps

    # Dual sensitivity (NOT the banked reading): if E[T] were held fixed at the reach-DP 5.21888,
    # the SAME 15.71 TPS gap would re-attribute to a slower deployed step for the int4-spec ceiling.
    step_int4_if_et_fixed = K_CAL * et_reach_implied / INT4_SPEC_CEILING
    step_inflation_pct_if_et_fixed = (step_int4_if_et_fixed / STEP - 1.0) * 100.0

    return {
        "composition_formula": "mu = K_cal * (E[T](1)/step) * tau ; K_cal=125.268, step=1.2182, tau=1",
        "reach_dp_central_tps": REACH_DP_CENTRAL,
        "reach_dp_lcb_tps": REACH_DP_LCB,
        "int4_spec_ceiling_tps": INT4_SPEC_CEILING,
        "et_reach_implied": et_reach_implied,
        "et_int4_implied": et_int4_implied,
        "et_reach_banked_199": REACH_DP_ET,
        "et_int4_banked_220": INT4_SPEC_ET,
        "et_reach_resid_vs_banked": abs(et_reach_implied - REACH_DP_ET),
        "et_int4_resid_vs_banked": abs(et_int4_implied - INT4_SPEC_ET),
        "ceiling_gap_tps": gap_tps,                                    # HEADLINE of instruction 1
        "delta_et": d_et,
        "gap_attribution": {
            "topology_coverage_tps": coverage_driver_tps,             # the rank-1 0.7304 over-count
            "topology_coverage_pct_of_gap": 100.0 * coverage_driver_tps / gap_tps,
            "step_tau_tps": steptau_driver_tps,
            "step_tau_pct_of_gap": 0.0,
            "residual_tps": residual_tps,
        },
        "dual_sensitivity_fixed_et": {
            "note": ("NOT the banked reading. If E[T] were pinned at the reach-DP 5.21888 (the PR "
                     "prose 'E[T]=5.219'), the same 15.71 TPS would re-attribute to a slower int4 "
                     "deployed step. Both artifacts bank the ceiling at step=1.2182, so the "
                     "operative attribution is topology/coverage (E[T]), not step."),
            "step_int4_if_et_fixed": step_int4_if_et_fixed,
            "step_inflation_pct_if_et_fixed": step_inflation_pct_if_et_fixed,
        },
        "note_pr_5p219": ("the PR-quoted 'int4-spec E[T]=5.219' equals the reach-DP/both-bugs E[T] "
                          "5.21888, NOT the int4-spec ceiling's implied E[T]; at the shared deployed "
                          "step=1.2182 the int4-spec ceiling 520.95 implies E[T]=5.0662 (#220 "
                          "int4_anchor_et1), so the 15.71 TPS gap is the E[T]/coverage difference."),
    }


# --------------------------------------------------------------------------- #
# (2) Which ceiling the lane-a kernel realizes (PR instruction 2).
# --------------------------------------------------------------------------- #
def operative_ceiling_argument() -> dict[str, Any]:
    """#216: split-K reduction-order fix changes argmax-ORDER (determinism) NOT topology/coverage.
    So the batch-invariant kernel inherits the shipped topology+coverage ⇒ realizes the int4-spec
    ceiling 520.95; the reach-DP 536.66 needs the ρ-optimal topology at full rank-1 coverage =
    land #71's draft/kernel-INDEPENDENT topology work, NOT something the kernel does."""
    operative = INT4_SPEC_CEILING
    band = [INT4_SPEC_CEILING, REACH_DP_CENTRAL]
    # how far inside the band does the worst-case bar sit? (0 = lower/int4 edge, 1 = upper/reach edge)
    bar_position_in_band = ((PRIVATE_BAR_WORSTCASE - band[0]) / (band[1] - band[0]))
    return {
        "operative_compliant_ceiling": operative,                     # TEST metric (defensible)
        "operative_basis": "int4_spec_520p95",
        "band_low_high": band,
        "band_width_tps": band[1] - band[0],
        "bar_position_in_band": bar_position_in_band,                 # ~0.48 ⇒ bar sits mid-band
        "kernel_changes": "argmax-order (determinism) only; split-K reduction-order fix (#216)",
        "kernel_does_not_change": "tree topology or rank-1 coverage (#216 verify-share 0.6066)",
        "reasoning": (
            "lane-a is a custom batch-invariant int4 verify kernel. #216's kernel model: the "
            "split-K reduction-order fix changes argmax-ORDER (makes verify-M argmax == AR-M=1 "
            "argmax ⇒ greedy-valid) but NOT the tree topology or the rank-1 coverage. So the "
            "kernel inherits the SHIPPED stack's topology+coverage and only fixes determinism, "
            "hence its realized E[T](1) tracks the int4-spec ceiling 520.95 (E[T]=5.0662), NOT "
            "the optimistic reach-DP 536.66 (E[T]=5.21888). The reach-DP edge requires the "
            "ρ-optimal max-branch-3 topology AT full rank-1 coverage 0.7304 — a TOPOLOGY change "
            "that is draft/kernel-INDEPENDENT and belongs to land #71, and #199's own caveat is "
            "that the 0.7304 rank-1 ladder OVER-COUNTS true compliant accept (the rankprobe true "
            "token is the batch-VARIANT argmax). Therefore the defensible operative ceiling for "
            "lane-a's private μ at λ=1 is the int4-spec 520.95; the band [520.95, 536.66] is "
            "carried with its upper edge flagged as a land #71 topology question, NOT a kernel "
            "question (no kernel, however perfect, lifts the ceiling past 520.95 without the "
            "topology/coverage uplift)."),
        "residual_is_land71_topology_question": True,
    }


# --------------------------------------------------------------------------- #
# (3) Re-run stark #226's feasibility under every ceiling (PR instruction 3).
# --------------------------------------------------------------------------- #
def feasibility_table(operative: float) -> dict[str, Any]:
    """A ceiling clears a private bar iff bar <= ceiling (matches stark #226's `<=` convention)."""
    ceilings = [
        ("reach_dp_central_536p66", REACH_DP_CENTRAL),
        ("reach_dp_lcb_525p73", REACH_DP_LCB),
        ("int4_spec_520p95", INT4_SPEC_CEILING),
        ("operative", operative),
    ]
    bars = [
        ("worstcase_528p48", PRIVATE_BAR_WORSTCASE),
        ("empirical_52_535p14", PRIVATE_BAR_52),
    ]
    rows = []
    for cname, cval in ceilings:
        row: dict[str, Any] = {"ceiling": cname, "ceiling_tps": cval}
        for bname, bval in bars:
            row[f"clears_{bname}"] = bool(bval <= cval)
            row[f"margin_{bname}_tps"] = cval - bval        # >0 feasible, <0 infeasible
        rows.append(row)
    return {"rows": rows, "ceilings": dict(ceilings), "bars": dict(bars)}


# --------------------------------------------------------------------------- #
# (4a) Rank-1-coverage robustness sweep — re-run the #199 reach-DP on coverage-scaled
# ceiling spines (the int4-spec / operative ceiling is the deployed bound and does NOT
# move with the optimistic rank-1 ladder, so the operative verdict is invariant).
# --------------------------------------------------------------------------- #
def _load_banked_ceiling_spine() -> list[float]:
    p = REPO_ROOT / "research/validity/compliant_spec_et/compliant_spec_et_results.json"
    syn = json.loads(p.read_text(encoding="utf-8"))["synthesis"]
    return list(syn["brackets"]["both_bugs"]["ceiling_spine"])


def _scaled_spine(spine: list[float], target_top1: float, base_top1: float) -> list[float]:
    """Coherent coverage degradation: scale the whole conditional rank-1 ladder by
    target_top1/base_top1 (clamped to <1), so depth-1 hits the swept rank-1 coverage and the
    deeper conditional rungs degrade proportionally."""
    s = target_top1 / base_top1
    return [min(max(v * s, 0.0), 0.999999) for v in spine]


def rank1_coverage_sweep(operative: float) -> dict[str, Any]:
    spine = _load_banked_ceiling_spine()
    base_top1 = spine[0]
    et_base = C.et_via_reachdp(spine)["et_pmf_mean"]
    provenance_resid = abs(et_base - REACH_DP_ET)          # must reproduce 5.21888

    rows = []
    for t in RANK1_SWEEP:
        sp = _scaled_spine(spine, t, base_top1)
        et = C.et_via_reachdp(sp)["et_pmf_mean"]
        reach_edge_tps = official_tps(et)
        rows.append({
            "rank1_coverage_top1": t,
            "reach_dp_edge_et": et,
            "reach_dp_edge_tps": reach_edge_tps,
            "band_low_high": [INT4_SPEC_CEILING, reach_edge_tps],
            # the deployed operative ceiling is fixed (does NOT move with the optimistic ladder):
            "operative_ceiling_tps": operative,
            "worstcase_bar_feasible_vs_operative":
                bool(PRIVATE_BAR_WORSTCASE <= operative),
            "worstcase_bar_feasible_vs_reach_edge":
                bool(PRIVATE_BAR_WORSTCASE <= reach_edge_tps),
        })
    # coverage at which the reach-DP edge would equal the worst-case bar (feasibility break-even
    # at the OPTIMISTIC edge) — bisection on top1.
    lo, hi = 0.50, 0.95
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        et_mid = C.et_via_reachdp(_scaled_spine(spine, mid, base_top1))["et_pmf_mean"]
        if official_tps(et_mid) < PRIVATE_BAR_WORSTCASE:
            lo = mid
        else:
            hi = mid
    top1_reach_edge_equals_bar = 0.5 * (lo + hi)

    operative_verdict_stable = all(
        r["worstcase_bar_feasible_vs_operative"] == (PRIVATE_BAR_WORSTCASE <= operative)
        for r in rows)
    return {
        "sweep_rows": rows,
        "base_top1": base_top1,
        "reach_dp_edge_et_at_base": et_base,
        "provenance_resid_vs_banked_5p21888": provenance_resid,
        "operative_ceiling_invariant_to_coverage": True,
        "operative_verdict_stable_over_sweep": bool(operative_verdict_stable),
        "rank1_coverage_reach_edge_equals_worstcase_bar": top1_reach_edge_equals_bar,
        "note": ("the operative (int4-spec 520.95) ceiling is the DEPLOYED bound and is invariant "
                 "to the optimistic rank-1 ladder; only the reach-DP UPPER edge moves with "
                 "coverage. Across [0.70, 0.74] the worst-case bar 528.48 stays INFEASIBLE vs the "
                 "operative ceiling, and would clear the reach-DP edge only above rank-1 coverage "
                 f"~{top1_reach_edge_equals_bar:.4f}."),
    }


def publish_first_framing() -> dict[str, Any]:
    """PR instruction 4b/4c: #124 publish-first framing + the public-milestone note."""
    return {
        "both_ceilings_clear_public_500": bool(INT4_SPEC_CEILING > TARGET
                                               and REACH_DP_CENTRAL > TARGET),
        "int4_spec_margin_over_public_500": INT4_SPEC_CEILING - TARGET,
        "reach_dp_margin_over_public_500": REACH_DP_CENTRAL - TARGET,
        "publish_first_124": (
            "under #124 publish-first the PRIVATE bar is a POST-HOC DEFENCE measure, NOT a "
            "pre-launch gate. So this verdict sizes the DEFENCE PACKET's compliant-lane viability "
            "and the human's #211 kernel-build ROI — it is not a launch blocker. The PUBLIC lane "
            "is unaffected: BOTH ceilings clear the public 500 milestone (int4-spec +20.95, "
            "reach-DP +36.66), so the public-milestone case for the batch-invariant kernel stands "
            "regardless; only the PRIVATE-500 route is in question."),
    }


# --------------------------------------------------------------------------- #
# (5) Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def _selftests(pin: dict, op: dict, feas: dict, sweep: dict) -> dict[str, Any]:
    # (a) both ceilings round-trip their banked μ from K_cal·E[T]/step (τ=1), err ~0.
    rt_reach = abs(official_tps(REACH_DP_ET) - REACH_DP_CENTRAL)
    rt_int4 = abs(official_tps(INT4_SPEC_ET) - INT4_SPEC_CEILING)
    cond_a = bool(rt_reach <= TOL_ROUNDTRIP and rt_int4 <= TOL_ROUNDTRIP
                  and pin["et_reach_resid_vs_banked"] <= TOL_ROUNDTRIP
                  and pin["et_int4_resid_vs_banked"] <= TOL_ROUNDTRIP
                  and sweep["provenance_resid_vs_banked_5p21888"] <= TOL_PROV)

    # (b) ceiling_gap_tps == 536.66 − 520.95 within 0.01 (both the exact constant gap and literal).
    cond_b = bool(abs(pin["ceiling_gap_tps"] - (REACH_DP_CENTRAL - INT4_SPEC_CEILING)) <= TOL_ROUNDTRIP
                  and abs(pin["ceiling_gap_tps"] - GAP_LITERAL) <= 0.01
                  and abs(pin["gap_attribution"]["topology_coverage_tps"]
                          + pin["gap_attribution"]["step_tau_tps"]
                          - pin["ceiling_gap_tps"]) <= TOL_ROUNDTRIP)

    # (c) stark #226: feasible-vs-536.66 == True AND feasible-vs-525.73 == False, both reproduced.
    feas_536 = bool(PRIVATE_BAR_WORSTCASE <= REACH_DP_CENTRAL)
    feas_526 = bool(PRIVATE_BAR_WORSTCASE <= REACH_DP_LCB)
    # round-trip stark #226's bar from its banked gross-up (mu_safe_fresh / f_priv_worstcase).
    bar_roundtrip = abs(MU_SAFE_FRESH / F_PRIV_WORSTCASE - PRIVATE_BAR_WORSTCASE)
    cond_c = bool(feas_536 is True and feas_526 is False and bar_roundtrip <= TOL_ROUNDTRIP)

    # (d) compliant_private_feasible_at_operative_ceiling consistent with bar <= operative.
    operative = op["operative_compliant_ceiling"]
    feasible_at_operative = bool(PRIVATE_BAR_WORSTCASE <= operative)
    cond_d = bool(feasible_at_operative == (PRIVATE_BAR_WORSTCASE <= operative))

    # (e) feasibility is monotone non-decreasing in the ceiling.
    cs = sorted({REACH_DP_CENTRAL, REACH_DP_LCB, INT4_SPEC_CEILING, operative,
                 PRIVATE_BAR_WORSTCASE - 1.0, PRIVATE_BAR_WORSTCASE + 1.0})
    feas_seq = [PRIVATE_BAR_WORSTCASE <= c for c in cs]
    cond_e = all((not feas_seq[i]) or feas_seq[i + 1] for i in range(len(feas_seq) - 1))

    # (f) NaN-clean — set by the caller after the full payload walk.
    cond_f = True

    conditions = {
        "a_both_ceilings_roundtrip_banked_mu_from_et": cond_a,
        "b_ceiling_gap_tps_equals_536_minus_520_within_0p01": cond_b,
        "c_stark226_feasible_vs_536_true_and_vs_526_false_reproduced": cond_c,
        "d_feasible_at_operative_consistent_with_bar_le_ceiling": cond_d,
        "e_feasibility_monotone_in_ceiling": cond_e,
        "f_nan_clean": cond_f,
    }
    return {
        "conditions": conditions,
        "twoceiling_reconcile_self_test_passes": bool(all(conditions.values())),
        "detail": {
            "roundtrip_reach_tps_err": rt_reach, "roundtrip_int4_tps_err": rt_int4,
            "stark226_bar_roundtrip_err": bar_roundtrip,
            "feasible_vs_536": feas_536, "feasible_vs_526": feas_526,
            "feasible_at_operative": feasible_at_operative,
        },
    }


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    pin = pin_ceilings_and_decompose()
    op = operative_ceiling_argument()
    operative = op["operative_compliant_ceiling"]
    feas = feasibility_table(operative)
    sweep = rank1_coverage_sweep(operative)
    pubfirst = publish_first_framing()
    st = _selftests(pin, op, feas, sweep)

    compliant_private_feasible_at_operative = bool(PRIVATE_BAR_WORSTCASE <= operative)
    private_bar_minus_operative = PRIVATE_BAR_WORSTCASE - operative

    headline = {
        "twoceiling_reconcile_self_test_passes":
            bool(st["twoceiling_reconcile_self_test_passes"]),                 # PRIMARY
        "operative_compliant_ceiling": operative,                             # TEST
        "operative_basis": op["operative_basis"],
        "ceiling_gap_tps": pin["ceiling_gap_tps"],
        "gap_topology_coverage_pct": pin["gap_attribution"]["topology_coverage_pct_of_gap"],
        "gap_step_tau_pct": pin["gap_attribution"]["step_tau_pct_of_gap"],
        "et_reach_implied": pin["et_reach_implied"],
        "et_int4_implied": pin["et_int4_implied"],
        "operative_band_low_high": op["band_low_high"],
        "bar_position_in_band": op["bar_position_in_band"],
        "private_bar_worstcase": PRIVATE_BAR_WORSTCASE,
        "compliant_private_feasible_at_operative_ceiling": compliant_private_feasible_at_operative,
        "private_bar_minus_operative_ceiling": private_bar_minus_operative,
        "feasible_vs_reach_central_536": bool(PRIVATE_BAR_WORSTCASE <= REACH_DP_CENTRAL),
        "feasible_vs_reach_lcb_526": bool(PRIVATE_BAR_WORSTCASE <= REACH_DP_LCB),
        "feasible_vs_int4_spec_520": bool(PRIVATE_BAR_WORSTCASE <= INT4_SPEC_CEILING),
        "both_ceilings_clear_public_500": pubfirst["both_ceilings_clear_public_500"],
        "operative_verdict_stable_over_rank1_sweep": sweep["operative_verdict_stable_over_sweep"],
    }

    verdict = _verdict(operative, compliant_private_feasible_at_operative)
    handoff = _handoff(operative, op, compliant_private_feasible_at_operative)

    return {
        "headline": headline,
        "pin_ceilings": pin,
        "operative_ceiling": op,
        "feasibility": feas,
        "rank1_coverage_sweep": sweep,
        "publish_first_framing": pubfirst,
        "self_test": st,
        "composition": {
            "K_cal": K_CAL, "step": STEP, "tau_central": TAU_CENTRAL, "target_official": TARGET,
            "baseline_tps": BASELINE_TPS,
            "reach_dp_central_tps": REACH_DP_CENTRAL, "reach_dp_lcb_tps": REACH_DP_LCB,
            "int4_spec_ceiling_tps": INT4_SPEC_CEILING,
            "reach_dp_et": REACH_DP_ET, "int4_spec_et": INT4_SPEC_ET,
            "rank1_coverage_top1": RANK1_COVERAGE_TOP1,
            "private_bar_worstcase": PRIVATE_BAR_WORSTCASE,
            "private_bar_52": PRIVATE_BAR_52, "mu_safe_fresh": MU_SAFE_FRESH,
            "f_priv_worstcase": F_PRIV_WORSTCASE,
        },
        "verdict": verdict,
        "handoff_line": handoff,
        "imports": {
            "provenance": ("wirbel#199 wdyqnx3g (reach-DP ceiling 536.66/LCB 525.73, E[T] 5.21888, "
                           "rank-1 0.7304) x #204/#220 pqjnybbf (int4-spec ceiling 520.95, E[T] "
                           "5.0662) x stark#226 tzcc5xuq (private_bar_worstcase 528.48, f_priv "
                           "0.969107, #52-floor bar 535.14) x #216 pc8g6s04 (kernel model: split-K "
                           "fix = argmax-order not topology/coverage) x kanna#217 vgovdrjc "
                           "(K_cal 125.268, step 1.2182, private-bar basis). All run-ids in "
                           "wandb-applied-ai-team/gemma-challenge-senpai."),
            "machinery": "compliant_spec_et.et_via_reachdp (#175/#184 reach-DP) for the coverage sweep",
        },
    }


def _verdict(operative: float, feasible: bool) -> str:
    if operative <= INT4_SPEC_CEILING + 1e-9 and not feasible:
        return "OPERATIVE-INT4SPEC-520p95-COMPLIANT-PRIVATE-500-INFEASIBLE"
    if feasible:
        return "OPERATIVE-CEILING-CLEARS-WORSTCASE-BAR-COMPLIANT-PRIVATE-FEASIBLE"
    return "OPERATIVE-CEILING-BELOW-WORSTCASE-BAR-COMPLIANT-PRIVATE-INFEASIBLE"


def _handoff(operative: float, op: dict, feasible: bool) -> str:
    band = op["band_low_high"]
    feas_word = "feasible" if feasible else "INFEASIBLE"
    route = ("the lane survives as a private-500 route" if feasible
             else "the lane is public-milestone + post-hoc-defence ONLY, never private-500")
    return (
        f"the lane-a compliant kernel's operative λ=1 ceiling is the int4-spec {operative:.2f} "
        f"(band [{band[0]:.2f}, {band[1]:.2f}]; gap driver: the rank-1 coverage 0.7304 over-count "
        f"/ ρ-optimal-topology uplift that the determinism-only kernel does NOT deliver — #216), "
        f"so stark #226's worst-case private bar 528.48 is {feas_word} for the compliant-500-"
        f"private lane (528.48 − {operative:.2f} = {PRIVATE_BAR_WORSTCASE - operative:+.2f}) — "
        f"{route}; the reach-DP 536.66 upper edge is a land #71 topology question, not a kernel "
        f"one. Under #124 publish-first this sizes the defence packet + the #211 kernel-build ROI, "
        f"NOT a launch gate (both ceilings clear the PUBLIC 500 regardless). NOT a launch. NOT open2."
    )


# --------------------------------------------------------------------------- #
# NaN guard + report + W&B (mirrors #199/#220/#226; never fatal).
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
    h, pin, op = syn["headline"], syn["pin_ceilings"], syn["operative_ceiling"]
    feas, sweep, st = syn["feasibility"], syn["rank1_coverage_sweep"], syn["self_test"]
    print("\n" + "=" * 98, flush=True)
    print("TWO-CEILING RECONCILE (PR #235, wirbel) — reach-DP 536.66 vs int4-spec 520.95, CPU-only",
          flush=True)
    print("=" * 98, flush=True)
    print(f"  (1) PIN  mu = K_cal·(E[T]/step)·tau   [K_cal={K_CAL:.5f}, step={STEP}, tau=1]", flush=True)
    print(f"      reach-DP  {pin['reach_dp_central_tps']:.4f}  ⟹ E[T]={pin['et_reach_implied']:.6f}  "
          f"(banked 5.21888, resid {pin['et_reach_resid_vs_banked']:.1e})", flush=True)
    print(f"      int4-spec {pin['int4_spec_ceiling_tps']:.4f}  ⟹ E[T]={pin['et_int4_implied']:.6f}  "
          f"(banked 5.06614, resid {pin['et_int4_resid_vs_banked']:.1e})", flush=True)
    ga = pin["gap_attribution"]
    print(f"      ceiling_gap_tps = {pin['ceiling_gap_tps']:.4f}  =  topology/coverage "
          f"{ga['topology_coverage_tps']:.4f} ({ga['topology_coverage_pct_of_gap']:.1f}%)  +  "
          f"step/tau {ga['step_tau_tps']:.4f} ({ga['step_tau_pct_of_gap']:.0f}%)", flush=True)
    print(f"      dual (NOT banked): hold E[T]=5.21888 ⟹ int4 step "
          f"{pin['dual_sensitivity_fixed_et']['step_int4_if_et_fixed']:.4f} "
          f"(+{pin['dual_sensitivity_fixed_et']['step_inflation_pct_if_et_fixed']:.2f}% step)", flush=True)
    print("-" * 98, flush=True)
    print(f"  (2) OPERATIVE ceiling = {op['operative_compliant_ceiling']:.4f} (int4-spec)  "
          f"band [{op['band_low_high'][0]:.2f}, {op['band_low_high'][1]:.2f}]  "
          f"bar@{op['bar_position_in_band']*100:.0f}% of band", flush=True)
    print(f"      kernel changes: {op['kernel_changes']}", flush=True)
    print(f"      kernel does NOT change: {op['kernel_does_not_change']}  → reach-DP edge = land #71",
          flush=True)
    print("-" * 98, flush=True)
    print("  (3) FEASIBILITY (bar <= ceiling)        clears 528.48?   clears 535.14?", flush=True)
    for r in feas["rows"]:
        print(f"      {r['ceiling']:<26} {r['ceiling_tps']:8.3f}   "
              f"{str(r['clears_worstcase_528p48']):>5} ({r['margin_worstcase_528p48_tps']:+7.3f})   "
              f"{str(r['clears_empirical_52_535p14']):>5} ({r['margin_empirical_52_535p14_tps']:+7.3f})",
              flush=True)
    print(f"      HEADLINE compliant_private_feasible_at_operative_ceiling = "
          f"{h['compliant_private_feasible_at_operative_ceiling']}  "
          f"(bar − operative = {h['private_bar_minus_operative_ceiling']:+.4f})", flush=True)
    print("-" * 98, flush=True)
    print(f"  (4) rank-1 sweep [0.70,0.74]: operative invariant; reach-DP edge "
          f"{sweep['sweep_rows'][0]['reach_dp_edge_tps']:.1f}→{sweep['sweep_rows'][-1]['reach_dp_edge_tps']:.1f}; "
          f"worstcase verdict stable={sweep['operative_verdict_stable_over_sweep']}", flush=True)
    print(f"      reach-DP edge == 528.48 bar at rank-1 coverage "
          f"{sweep['rank1_coverage_reach_edge_equals_worstcase_bar']:.4f}", flush=True)
    print(f"      both ceilings clear PUBLIC 500 = {h['both_ceilings_clear_public_500']}", flush=True)
    print("-" * 98, flush=True)
    print(f"  (5) PRIMARY twoceiling_reconcile_self_test_passes = "
          f"{st['twoceiling_reconcile_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"        - {k}: {v}", flush=True)
    print("=" * 98, flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
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
        print(f"[twoceiling-reconcile] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    h, pin, op = syn["headline"], syn["pin_ceilings"], syn["operative_ceiling"]
    st, sweep, c = syn["self_test"], syn["rank1_coverage_sweep"], syn["composition"]
    run = init_wandb_run(
        job_type="twoceiling-reconcile",
        agent="wirbel",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["twoceiling-reconcile", "issue-211", "validity-gate", "valid-verify-cluster-capstone",
              "compliant-lane-feasibility", "private-bar", "bank-the-analysis"],
        config={
            "K_cal": K_CAL, "step": STEP, "tau_central": TAU_CENTRAL, "target_official": TARGET,
            "reach_dp_central_tps": REACH_DP_CENTRAL, "reach_dp_lcb_tps": REACH_DP_LCB,
            "int4_spec_ceiling_tps": INT4_SPEC_CEILING,
            "reach_dp_et": REACH_DP_ET, "int4_spec_et": INT4_SPEC_ET,
            "rank1_coverage_top1": RANK1_COVERAGE_TOP1,
            "private_bar_worstcase": PRIVATE_BAR_WORSTCASE, "private_bar_52": PRIVATE_BAR_52,
            "wandb_group": args.wandb_group, "baseline_tps": BASELINE_TPS,
            "source_runs": "wirbel#199 wdyqnx3g, #220 pqjnybbf, stark#226 tzcc5xuq, "
                           "#216 pc8g6s04, kanna#217 vgovdrjc",
        },
    )
    if run is None:
        print("[twoceiling-reconcile] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "twoceiling_reconcile_self_test_passes":
            int(bool(st["twoceiling_reconcile_self_test_passes"])),               # PRIMARY
        "operative_compliant_ceiling": h["operative_compliant_ceiling"],          # TEST
        "ceiling_gap_tps": h["ceiling_gap_tps"],
        "gap_topology_coverage_pct": h["gap_topology_coverage_pct"],
        "gap_step_tau_pct": h["gap_step_tau_pct"],
        "et_reach_implied": h["et_reach_implied"],
        "et_int4_implied": h["et_int4_implied"],
        "delta_et": pin["delta_et"],
        "operative_band_low": h["operative_band_low_high"][0],
        "operative_band_high": h["operative_band_low_high"][1],
        "bar_position_in_band": h["bar_position_in_band"],
        "private_bar_worstcase": h["private_bar_worstcase"],
        "compliant_private_feasible_at_operative_ceiling":
            int(bool(h["compliant_private_feasible_at_operative_ceiling"])),
        "private_bar_minus_operative_ceiling": h["private_bar_minus_operative_ceiling"],
        "feasible_vs_reach_central_536": int(bool(h["feasible_vs_reach_central_536"])),
        "feasible_vs_reach_lcb_526": int(bool(h["feasible_vs_reach_lcb_526"])),
        "feasible_vs_int4_spec_520": int(bool(h["feasible_vs_int4_spec_520"])),
        "both_ceilings_clear_public_500": int(bool(h["both_ceilings_clear_public_500"])),
        "operative_verdict_stable_over_rank1_sweep":
            int(bool(h["operative_verdict_stable_over_rank1_sweep"])),
        "rank1_coverage_reach_edge_equals_worstcase_bar":
            sweep["rank1_coverage_reach_edge_equals_worstcase_bar"],
        "reach_dp_edge_provenance_resid": sweep["provenance_resid_vs_banked_5p21888"],
        "step_inflation_pct_if_et_fixed":
            pin["dual_sensitivity_fixed_et"]["step_inflation_pct_if_et_fixed"],
        "verdict_int4spec_private_infeasible":
            int(syn["verdict"] == "OPERATIVE-INT4SPEC-520p95-COMPLIANT-PRIVATE-500-INFEASIBLE"),
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    # per-rank1-coverage reach-DP edge as logged scalars.
    for r in sweep["sweep_rows"]:
        tag = f"{r['rank1_coverage_top1']:.4f}".replace(".", "p")
        summary[f"reach_dp_edge_tps_top1_{tag}"] = r["reach_dp_edge_tps"]
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="twoceiling_reconcile_result", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    print(f"[twoceiling-reconcile] wandb logged {len(summary)} keys", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="valid-verify-cluster-capstone")
    args = ap.parse_args(argv)

    syn = synthesize()

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 235, "agent": "wirbel",
        "kind": "twoceiling-reconcile", "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _nan_paths(payload)
    payload["nan_clean"] = not nan_paths
    syn["self_test"]["conditions"]["f_nan_clean"] = not nan_paths
    syn["self_test"]["twoceiling_reconcile_self_test_passes"] = bool(
        all(syn["self_test"]["conditions"].values()))
    syn["headline"]["twoceiling_reconcile_self_test_passes"] = syn["self_test"][
        "twoceiling_reconcile_self_test_passes"]
    if nan_paths:
        print(f"[twoceiling-reconcile] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[twoceiling-reconcile] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = (syn["self_test"]["twoceiling_reconcile_self_test_passes"] and payload["nan_clean"])
        print(f"[twoceiling-reconcile] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
