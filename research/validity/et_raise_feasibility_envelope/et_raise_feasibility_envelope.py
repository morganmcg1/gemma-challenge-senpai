#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #281 -- E[T]-raise feasibility: the three-axis Path-A closure verdict.

CPU-only analytic portfolio integration. NO GPU, NO HF Job, NO submission, NO
served-file change. Consumes MEASURED anchors only (does NOT re-measure).

The question
-----------
Path-A (reach honest-500 TPS by a speculative-decoding lever) has three axes:
    draft-cut  (closed: fern #274 brnmnl60 -- cutting draft passes LOWERS E[T])
    width      (closed: denken #271 9mlmaen3 -- M*=32 -> 479.57 < 500, step race lost)
    E[T]-raise (this PR -- open with a GO-point, or closed?)
This leg integrates the portfolio into ONE feasibility envelope and delivers the
verdict: does the best ACHIEVABLE E[T]_real (measured ceiling, private-degraded)
ever reach the honest-500 E[T] floor (lowered by greedy-safe step-shaving) at any
tree width M?

Composition law (clean K_cal frame, identical to fern #274)
-----------------------------------------------------------
    official = K_cal * E[T]_real / step_rel          (step_rel = step/step_deployed)
    honest-500 floor:  E[T]_floor(step_rel) = 500 * step_rel / K_cal
At the deployed step (step_rel = 1) this reproduces #274's E[T]_floor = 3.9914 and
official = 481.53 (within the 0.012% K_cal calibration residual, as #274). The
literal PR form  official = K_cal * (E[T]/step) * tau  with the deployed step
normalised to tau is algebraically identical (tau cancels) -- both are checked.

Anchors are pulled from merged/finished artifacts (provenance in comments). Two
W&B values differ from the PR body and are flagged honestly:
  * ubel #263's MEASURED decode private/public E[T] ratio is 0.804 (decode_priv_ET
    3.0898 / decode_pub_ET 3.8444, run 2khp8gzs, anchor self-test passes), NOT 0.73.
    The PR's "0.73" is NOT reproducible as the decode E[T] ratio in 2khp8gzs (it is a
    more pessimistic, agreement-style figure). The verdict is reported at the measured
    0.804 AND swept over [0.65, 0.73, 0.85], which brackets both -- and is CLOSED at
    every realizable point, so the discrepancy does not flip the conclusion.
  * land #245's "faithful M=16 tree" run was not found in the project; the nearest
    measured proxy (fern/m16 mjynhw39 et_m16 = 3.8444) equals the deployed linear
    E[T]. This matches the PR's own note that the faithful tree is "as greedy-
    faithful as linear (delta -0.0021)": it buys FAITHFULNESS, not an E[T] raise.
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

# ============================================================================
# ANCHORS -- merged/finished artifacts, imported EXACT (provenance in comments)
# ============================================================================
BASE_TPS = 481.53                          # official served TPS (PR #52)
TARGET_TPS = 500.0
K_CAL = 125.26795005202914                 # denken #257 steps/sec calibration
E_T_REAL = 3.8444537125748504              # stark #266 deployed public accepted tok/step @ M=8
LAMBDA1_CEIL = 520.9527323111674           # K_cal * E[T] * tau ceiling
TAU_LO = 1.03524                           # lawine #267 tau_lo local->official
HONEST_ET_FLOOR_274 = 3.9914439391107512   # fern #274 brnmnl60 = 500 / K_cal

# phi edges (fern #274 brnmnl60): realization_ratio = honest_gain / comp_gain = phi.
PHI_LO = 0.12526795005202912               # assumed g_d edge (served_wall_clock_est wall)
PHI_HI = 0.735133735318471                 # measured g_d edge (kcal_clean wall) -> PR's "0.735"

# denken #271 step model (run 9mlmaen3) -- step(M;g_d) in microseconds.
STEP_SERVED_US = 1218.2
V8 = 5163.71                               # verify_us(M=8) reference
VERIFY_US = {8: 5163.71, 16: 5405.0, 32: 5979.95}   # denken #257 h1gj2ved / #271
N_TREE = 5
K_SPEC = 7
G_D_DEPLOYED = 0.0191                       # denken #271 deployed g_d
DENKEN271_OFFICIAL_M32 = 479.57075450999776  # 9mlmaen3 summary/tps_at_Mstar32_deployed (width NO-GO)

# ubel #263 private/OOD E[T] degradation (run 2khp8gzs).
UBEL263_PUB_ET = 3.8444                     # decode_pub_ET
UBEL263_PRIV_ET = 3.0898                    # decode_priv_ET
UBEL263_PRIV_PUB_RATIO = UBEL263_PRIV_ET / UBEL263_PUB_ET   # 0.8037 MEASURED
PR_STATED_PRIV_FACTOR = 0.73               # PR body; NOT reproducible as the decode E[T] ratio --
#                                            run 2khp8gzs MEASURES 0.804 (decode_priv_ET/decode_pub_ET,
#                                            anchor self-test passes). 0.73 is a more pessimistic figure;
#                                            swept (it is in PRIV_FACTOR_REALIZABLE), not used as central.
PRIV_FACTOR_CENTRAL = UBEL263_PRIV_PUB_RATIO                  # 0.8037 MEASURED
# realizable private-factor range = PR's [0.65, 0.85] (brackets PR-0.73 AND measured-0.804):
PRIV_FACTOR_REALIZABLE = [0.65, 0.73, PRIV_FACTOR_CENTRAL, 0.85]
# sensitivity sweep adds the priv=1.0 COUNTERFACTUAL (no degradation at all -- not realizable,
# since ubel #263 MEASURES 0.804; kept only to show the raise is missing even before degradation):
PRIV_FACTOR_SENS = [*PRIV_FACTOR_REALIZABLE, 1.0]


def _pf_key(pf: float) -> str:
    return f"{pf:.6f}"

# land #245 faithful M=16 tree E[T] -- proxy (run not found; fern/m16 mjynhw39 et_m16).
LAND245_FAITHFUL_M16_ET = 3.8444           # ~= deployed linear; faithful tree = faithfulness, not raise
LAND245_GREEDY_FAITHFUL_DELTA = -0.0021    # PR: as greedy-faithful as linear

# kanna #269 draft-MLP fold (run epl52mkq).
KANNA269_FOLD_GAIN_PCT = 4.392280827418404   # projected_tps_gain_pct (composition, draft-side)
# kanna verify-step-roofline (run sdrerk5h, FINISHED): measured verify-side step-shave ceiling.
KANNA_VERIFY_ROOFLINE_LANDED = True
KANNA_VERIFY_ROOFLINE_RUN = "sdrerk5h"
KANNA_VERIFY_ROOFLINE_PCT = 1.1853573814381013   # max_verify_side_step_shaving_tps_gain_pct (composition, verify-side)
# max greedy-safe step-shave = draft-side fold (#269) + verify-side roofline (kanna), first-order sum:
STEPSHAVE_MAX_MEASURED_PCT = KANNA269_FOLD_GAIN_PCT + KANNA_VERIFY_ROOFLINE_PCT   # ~5.5776 %
# composition step-shave grid (% TPS, before phi-realization): 0, kanna verify-side alone, the #269
# fold alone, the combined MEASURED max, and two beyond-ceiling STRESS points (10, 20) showing the
# verdict is robust even far past the measured step-shave ceiling.
STEPSHAVE_COMP_PCT_GRID = [0.0, KANNA_VERIFY_ROOFLINE_PCT, KANNA269_FOLD_GAIN_PCT,
                           STEPSHAVE_MAX_MEASURED_PCT, 10.0, 20.0]
STEPSHAVE_MEASURED_SET = {0.0, KANNA_VERIFY_ROOFLINE_PCT, KANNA269_FOLD_GAIN_PCT, STEPSHAVE_MAX_MEASURED_PCT}

M_GRID = [8, 16, 32]
TOL = 1e-6
RELTOL = 0.01                              # #274 convention for the 481.53 reproduction


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# ============================================================================
# Composition law + step model
# ============================================================================
def official(e_t_real: float, step_rel: float) -> float:
    """Clean K_cal frame: official = K_cal * E[T]_real / step_rel."""
    return K_CAL * e_t_real / step_rel


def official_literal(e_t_real: float, step_norm: float) -> float:
    """PR literal form official = K_cal * (E[T]/step) * tau (deployed step_norm = tau)."""
    return K_CAL * (e_t_real / step_norm) * TAU_LO


def et_floor(step_rel: float) -> float:
    """Honest-500 E[T] floor at a given relative step: E[T]_floor = 500 * step_rel / K_cal."""
    return TARGET_TPS * step_rel / K_CAL


def step_us(m: int, g_d: float = G_D_DEPLOYED) -> float:
    """denken #271 step model in microseconds."""
    return STEP_SERVED_US * (VERIFY_US[m] / V8 + N_TREE * g_d) / (1.0 + K_SPEC * g_d)


def step_rel(m: int, g_d: float = G_D_DEPLOYED) -> float:
    """Deployed-anchored relative step (= 1 at M=8): step(M)/step(8)."""
    return step_us(m, g_d) / step_us(8, g_d)


def honest_stepshave_pct(comp_pct: float, phi: float) -> float:
    """phi-realization: an E[T]-independent step-shave worth comp_pct composition TPS realizes
    honest_pct = phi * comp_pct at the wall (fern #274 realization ratio)."""
    return phi * comp_pct


def s_rel_after_shave(comp_pct: float, phi: float) -> float:
    """Relative step after a greedy-safe step-shave: official rises by (1 + honest/100)."""
    return 1.0 / (1.0 + honest_stepshave_pct(comp_pct, phi) / 100.0)


def et_floor_after_shave(comp_pct: float, phi: float) -> float:
    """Honest-500 floor at the deployed M=8 step lowered by a phi-corrected step-shave."""
    return (TARGET_TPS / K_CAL) * s_rel_after_shave(comp_pct, phi)


# ============================================================================
# (1) The honest-500 E[T] floor as a function of step-shave
# ============================================================================
def floor_vs_stepshave() -> dict[str, Any]:
    rows = []
    for comp in STEPSHAVE_COMP_PCT_GRID:
        row = {
            "stepshave_comp_pct": comp,
            "known": bool(comp in STEPSHAVE_MEASURED_SET),  # 0 + kanna verify-side + #269 fold + combined = measured
            "phi_lo": {"phi": PHI_LO,
                       "honest_stepshave_pct": honest_stepshave_pct(comp, PHI_LO),
                       "et_floor": et_floor_after_shave(comp, PHI_LO)},
            "phi_hi": {"phi": PHI_HI,
                       "honest_stepshave_pct": honest_stepshave_pct(comp, PHI_HI),
                       "et_floor": et_floor_after_shave(comp, PHI_HI)},
        }
        rows.append(row)
    # lowest floor over the grid (most favourable to GO): max step-shave, phi_hi
    lowest = min(r["phi_hi"]["et_floor"] for r in rows)
    return {
        "deployed_floor": et_floor(1.0),                 # 3.9914 (self-test a)
        "rows": rows,
        "lowest_floor_over_grid": lowest,
        "note": ("step-shaving LOWERS the floor; phi_hi (measured g_d) realizes more of each "
                 "composition step-shave so it lowers the floor most. MEASURED step-shaves: kanna "
                 "verify-side roofline (sdrerk5h, %.2f%%) + #269 draft-MLP fold (%.2f%%) = %.2f%% "
                 "combined max; the 10/20%% rows are beyond-ceiling STRESS points."
                 % (KANNA_VERIFY_ROOFLINE_PCT, KANNA269_FOLD_GAIN_PCT, STEPSHAVE_MAX_MEASURED_PCT)),
    }


# ============================================================================
# (2) The best-achievable E[T]_real ceiling from the measured screens + M-coupling
# ============================================================================
def implied_width_et32() -> float:
    """Back-out denken #271's deliverable public E[T](M=32) from its measured 479.5708 via the
    step model: official = K_cal * E[T] / step_rel  =>  E[T] = official * step_rel / K_cal."""
    return DENKEN271_OFFICIAL_M32 * step_rel(32) / K_CAL


def et_real_ceiling() -> dict[str, Any]:
    et32 = implied_width_et32()
    # measured PUBLIC E[T] screens (full-forward), with their width M:
    public_screens = {
        "deployed_linear_m8": {"M": 8, "e_t_public": E_T_REAL,
                               "src": "stark #266 deployed accepted tok/step"},
        "faithful_tree_m16": {"M": 16, "e_t_public": LAND245_FAITHFUL_M16_ET,
                              "src": "land #245 proxy (mjynhw39 et_m16); faithful=linear, delta -0.0021"},
        "width_tree_m32": {"M": 32, "e_t_public": et32,
                           "src": "denken #271 9mlmaen3 implied (from 479.5708 + step model)"},
    }
    # M-coupling: NET official per M at the measured public E[T] (full-forward, priv=1):
    m_coupling = {}
    for name, s in public_screens.items():
        m = s["M"]
        sr = step_rel(m)
        net_pub = official(s["e_t_public"], sr)
        m_coupling[name] = {
            "M": m, "step_rel": sr, "e_t_public": s["e_t_public"],
            "net_official_public": net_pub,
            "clears_500_public": net_pub >= TARGET_TPS,
            # floor that THIS M's step demands to clear 500:
            "et_floor_at_M": et_floor(sr),
            "e_t_public_clears_M_floor": s["e_t_public"] >= et_floor(sr),
            "src": s["src"],
        }
    # the (E[T], M) that maximises NET public official == max E[T]/step_rel:
    best_name = max(m_coupling, key=lambda n: m_coupling[n]["net_official_public"])
    best = m_coupling[best_name]
    # private ceiling: the achievable E[T]_real at the net-maximising M, degraded by the private factor.
    # express in the deployed-M8-step frame so it is directly comparable to the floor:
    #   e_t_real_equiv(M) = priv * e_t_public(M) / step_rel(M)   (deployed-step-equivalent E[T])
    best_achievable = {}
    for pf in PRIV_FACTOR_SENS:
        equiv = {n: pf * m_coupling[n]["e_t_public"] / m_coupling[n]["step_rel"] for n in m_coupling}
        argmax = max(equiv, key=lambda n: equiv[n])
        best_achievable[_pf_key(pf)] = {
            "priv_factor": pf,
            "best_e_t_real_deployed_equiv": equiv[argmax],
            "argmax_screen": argmax, "argmax_M": m_coupling[argmax]["M"],
            "per_screen_deployed_equiv": equiv,
        }
    return {
        "implied_width_et32": et32,
        "public_screens": public_screens,
        "m_coupling": m_coupling,
        "net_maximising": {"screen": best_name, "M": best["M"],
                           "net_official_public": best["net_official_public"]},
        "best_achievable_et_real_by_priv_factor": best_achievable,
        "width_nogo_special_case": {
            "M": 32, "official": DENKEN271_OFFICIAL_M32,
            "et_floor_at_M32": et_floor(step_rel(32)),
            "deliverable_et32": et32,
            "closed": DENKEN271_OFFICIAL_M32 < TARGET_TPS,
            "note": ("at M=32 the step penalty raises the 500-floor to %.4f while width only "
                     "delivers E[T]=%.4f -> %.2f < 500. denken #271 width NO-GO drops out."
                     % (et_floor(step_rel(32)), et32, DENKEN271_OFFICIAL_M32)),
        },
        "note": ("NET-maximising M is M=8 (acceptance-per-candidate), NOT a wider tree: widening "
                 "raises E[T] but raises the step more (denken #271). The best measured public "
                 "acceptance-per-candidate E[T] is %.4f -- already BELOW the 3.9914 honest-500 "
                 "floor before any private degradation." % E_T_REAL),
    }


# ============================================================================
# (3) The decisive 2-D feasibility map (E[T]_real x step-shave)
# ============================================================================
def feasibility_map(ceiling: dict[str, Any]) -> dict[str, Any]:
    # best achievable E[T]_real (deployed-equiv) at the central measured private factor:
    central = ceiling["best_achievable_et_real_by_priv_factor"][_pf_key(PRIV_FACTOR_CENTRAL)]
    best_achievable_central = central["best_e_t_real_deployed_equiv"]

    # min E[T] needed = lowest floor over the MEASURED max step-shave (kanna verify-side + #269 fold)
    # at phi_hi (best case for GO -- most realized step-shave, most favourable phi edge):
    floor_maxshave_phi_hi = et_floor_after_shave(STEPSHAVE_MAX_MEASURED_PCT, PHI_HI)
    floor_deployed = et_floor(1.0)

    # scan the grid -> any GO cell inside the REALIZABLE envelope?
    go_cells = []
    counterfactual_go_cells = []
    grid = []
    for comp in STEPSHAVE_COMP_PCT_GRID:
        for phi in (PHI_LO, PHI_HI):
            fl = et_floor_after_shave(comp, phi)
            for pf in PRIV_FACTOR_SENS:
                # priv=1.0 is the no-degradation COUNTERFACTUAL; ubel #263 MEASURES 0.804, so a GO
                # that needs priv=1.0 is not realizable and must not create a false GO-region.
                realizable = pf in PRIV_FACTOR_REALIZABLE
                ba = ceiling["best_achievable_et_real_by_priv_factor"][_pf_key(pf)][
                    "best_e_t_real_deployed_equiv"]
                go = ba >= fl
                cell = {"stepshave_comp_pct": comp, "phi": phi, "priv_factor": pf,
                        "realizable": realizable,
                        "et_floor": fl, "best_achievable_et_real": ba, "go": go}
                grid.append(cell)
                if go and realizable:
                    go_cells.append(cell)
                elif go and not realizable:
                    counterfactual_go_cells.append(cell)
    # decisive: a GO-region exists only if a REALIZABLE cell clears the floor.
    go_region_exists = len(go_cells) > 0

    # crossing analyses (what it would take to flip GO):
    #  (i) public E[T] raise needed at the central private factor + deployed step:
    et_public_needed_deployed = floor_deployed / PRIV_FACTOR_CENTRAL
    et_public_needed_maxshave = floor_maxshave_phi_hi / PRIV_FACTOR_CENTRAL
    #  (ii) step-shave needed (phi_hi) to drop the floor to the central best-achievable:
    #       3.99144 / (1 + phi_hi*c/100) = best_achievable  ->  c = ...
    if best_achievable_central > 0:
        comp_needed = (floor_deployed / best_achievable_central - 1.0) * 100.0 / PHI_HI
    else:
        comp_needed = float("inf")
    #  (iii) private factor needed at deployed step with NO raise (public E[T]=E_T_REAL):
    priv_needed_no_raise_deployed = floor_deployed / E_T_REAL    # > 1 => impossible even public
    priv_needed_no_raise_maxshave = floor_maxshave_phi_hi / E_T_REAL

    return {
        "best_achievable_et_real_central": best_achievable_central,
        "central_priv_factor": PRIV_FACTOR_CENTRAL,
        "min_et_real_needed": floor_maxshave_phi_hi,  # lowest MEASURED floor (kanna+#269 max shave, phi_hi)
        "min_et_real_needed_deployed": floor_deployed,
        "stepshave_max_measured_pct": STEPSHAVE_MAX_MEASURED_PCT,
        "go_region_exists": go_region_exists,
        "n_go_cells": len(go_cells),
        "go_cells": go_cells,
        "n_counterfactual_go_cells": len(counterfactual_go_cells),
        "counterfactual_go_cells": counterfactual_go_cells,
        "grid": grid,
        "crossing": {
            "public_et_needed_at_deployed_step": et_public_needed_deployed,
            "public_et_needed_at_maxshave_floor": et_public_needed_maxshave,
            "measured_public_et_ceiling_m8": E_T_REAL,
            "public_et_raise_gap_deployed": et_public_needed_deployed - E_T_REAL,
            "stepshave_comp_pct_needed_phi_hi": comp_needed,
            "priv_factor_needed_no_raise_deployed": priv_needed_no_raise_deployed,
            "priv_factor_needed_no_raise_maxshave_floor": priv_needed_no_raise_maxshave,
        },
        "note": ("side-by-side: min E[T]_real needed (kanna+#269 max measured step-shave %.2f%%, "
                 "phi_hi) = %.4f vs best achievable E[T]_real (private %.4f) = %.4f. Gap = %.4f. "
                 "Even at private=1.0 (no degradation) the best public acceptance-per-candidate "
                 "E[T]=%.4f < the deployed floor %.4f, so the raise is missing before degradation "
                 "even enters."
                 % (STEPSHAVE_MAX_MEASURED_PCT, floor_maxshave_phi_hi, PRIV_FACTOR_CENTRAL,
                    best_achievable_central, best_achievable_central - floor_maxshave_phi_hi,
                    E_T_REAL, floor_deployed)),
    }


# ============================================================================
# (4) The three-axis closure verdict
# ============================================================================
def closure_verdict(fmap: dict[str, Any]) -> dict[str, Any]:
    et_raise_open = fmap["go_region_exists"]
    axes = {"draft_cut": True, "width": True, "et_raise": not et_raise_open}
    fully_closed = all(axes.values())
    return {
        "path_a_axes_closed": axes,
        "path_a_fully_closed": fully_closed,
        "headline": (
            ("Path-A FULLY CLOSED on all three axes (draft-cut x fern #274, width x denken #271, "
             "E[T]-raise x this leg): the deployed 481.53 frontier cannot reach 500 by any "
             "speculative-decoding lever under the measured constraints."
             ) if fully_closed else
            ("E[T]-raise axis OPEN -- a GO-point exists; hand to land #245's build.")),
        "mechanism_if_closed": (
            "the best MEASURED public acceptance-per-candidate E[T] (%.4f, M=8) sits below the "
            "honest-500 floor (%.4f) BEFORE private degradation; width raises E[T] but pays a "
            "larger step (M=32 -> %.2f < 500); private degradation (x%.4f) drops the achievable "
            "E[T]_real to %.4f, ~%.2f short of even the step-shave-lowered floor. Closing the gap "
            "needs a public E[T] >= %.3f at the deployed step -- a BUILT drafter/tree raise that "
            "no measured screen delivers (land #245's faithful tree is faithfulness, not a raise)."
            % (E_T_REAL, fmap["min_et_real_needed_deployed"], DENKEN271_OFFICIAL_M32,
               PRIV_FACTOR_CENTRAL, fmap["best_achievable_et_real_central"],
               fmap["min_et_real_needed"] - fmap["best_achievable_et_real_central"],
               fmap["crossing"]["public_et_needed_at_deployed_step"])),
    }


# ============================================================================
# (5) Sensitivity / honest caveats (which input flips GO <-> NO-GO)
# ============================================================================
def verdict_sensitivity(ceiling: dict[str, Any]) -> dict[str, Any]:
    floor_deployed = et_floor(1.0)
    floor_maxshave_hi = et_floor_after_shave(STEPSHAVE_MAX_MEASURED_PCT, PHI_HI)  # lowest MEASURED floor
    ba_central = PRIV_FACTOR_CENTRAL * E_T_REAL
    rows = []

    # private factor (dominant): best_achievable = pf * E_T_REAL (M=8 deployed-equiv) vs the lowest
    # MEASURED floor (kanna+#269 max shave, phi_hi -- the most GO-favourable floor). pf=1.0 is the
    # no-degradation COUNTERFACTUAL (ubel #263 MEASURES 0.804), flagged not-realizable.
    for pf in PRIV_FACTOR_SENS:
        ba = pf * E_T_REAL
        rows.append({"input": "private_factor (ubel #263)", "value": pf,
                     "realizable": pf in PRIV_FACTOR_REALIZABLE,
                     "best_achievable_et_real": ba, "floor_compared": floor_maxshave_hi,
                     "go": ba >= floor_maxshave_hi})
    # phi edge: floor under the MAX measured step-shave at each edge; best at central private factor.
    for phi, tag in ((PHI_LO, "phi_lo"), (PHI_HI, "phi_hi")):
        fl = et_floor_after_shave(STEPSHAVE_MAX_MEASURED_PCT, phi)
        rows.append({"input": "phi_edge (max-shave floor)", "value": tag, "realizable": True,
                     "best_achievable_et_real": ba_central, "floor_compared": fl,
                     "go": ba_central >= fl})
    # step-shave comp pct (kanna verify-side LANDED + #269 fold): floor at phi_hi; central private.
    for comp in STEPSHAVE_COMP_PCT_GRID:
        fl = et_floor_after_shave(comp, PHI_HI)
        rows.append({"input": "stepshave_comp_pct (kanna+#269)", "value": comp,
                     "realizable": comp in STEPSHAVE_MEASURED_SET,
                     "best_achievable_et_real": ba_central, "floor_compared": fl,
                     "go": ba_central >= fl})
    # public E[T] ceiling (the build): which MEASURED public E[T] screen flips GO at central private.
    for name, sc in ceiling["m_coupling"].items():
        ba = PRIV_FACTOR_CENTRAL * sc["e_t_public"] / sc["step_rel"]
        rows.append({"input": "public_et_ceiling (build)", "value": name, "realizable": True,
                     "best_achievable_et_real": ba, "floor_compared": floor_deployed,
                     "go": ba >= floor_deployed})

    any_realizable_go = any(r["go"] and r["realizable"] for r in rows)
    any_counterfactual_go = any(r["go"] and not r["realizable"] for r in rows)
    comp_to_close = ((et_floor(1.0) / ba_central) - 1.0) * 100.0 / PHI_HI
    return {
        "rows": rows,
        "any_input_flips_to_go": any_realizable_go,          # DECISIVE: realizable inputs only
        "any_realizable_input_flips_to_go": any_realizable_go,
        "any_counterfactual_flips_to_go": any_counterfactual_go,
        "dominant_input": "private_factor (ubel #263) -- the closure hinges on it being < 1.0",
        "note": ("no REALIZABLE input flips GO. The ONLY GO cells are counterfactual: at the full "
                 "MEASURED step-shave (%.2f%%, phi_hi) the deployed public E[T]=%.4f clears the "
                 "lowered floor %.4f by a hair -- but ONLY at private_factor=1.0 (zero degradation). "
                 "ubel #263 MEASURES 0.804, dropping the achievable E[T]_real to %.4f and closing "
                 "it decisively. Most sensitive to (a) ubel #263's private factor being < 1.0 and "
                 "(b) whether a BUILT drafter/tree raises public E[T] above %.3f -- land #245's "
                 "build to measure, not this leg."
                 % (STEPSHAVE_MAX_MEASURED_PCT, E_T_REAL, floor_maxshave_hi, ba_central, E_T_REAL)),
        "caveats": [
            "composition-priced: this leg measures 0 TPS; the launch gate stays land #245's "
            "MEASURED >=500 at lambda_hat >= 0.9780 (human-approval-gated).",
            "ubel #263 private factor: the decode E[T] ratio MEASURES 0.804 (2khp8gzs decode_priv_ET "
            "3.0898 / decode_pub_ET 3.8444, anchor self-test passes); the PR's 0.73 is NOT that "
            "ratio (not reproducible as the decode E[T] ratio) -- a more pessimistic figure. Swept "
            "[0.65, 0.73, 0.804, 0.85]; no CI logged. Closed at every realizable swept value (only "
            "the counterfactual private=1.0 + full step-shave clears, by ~0.01 -- not realizable).",
            "land #245 faithful-tree run not found in W&B; M=16 E[T] proxy = deployed linear "
            "(3.8444), consistent with the PR's faithful=linear (delta -0.0021) note: it is "
            "faithfulness, not an E[T] raise.",
            "kanna verify-step-roofline LANDED (sdrerk5h, %.2f%% verify-side); combined with the "
            "#269 fold (%.2f%%) the MEASURED max step-shave is %.2f%% composition. Dropping the "
            "deployed floor to the achievable ceiling would need ~%.0f%% composition (phi_hi) -- "
            "~7x the measured ceiling, implausible (verify is ~half the step)."
            % (KANNA_VERIFY_ROOFLINE_PCT, KANNA269_FOLD_GAIN_PCT, STEPSHAVE_MAX_MEASURED_PCT,
               comp_to_close),
        ],
    }


# ============================================================================
# (6) Self-tests (PRIMARY)
# ============================================================================
def self_tests(floor_tab, ceiling, fmap, verdict) -> dict[str, Any]:
    # (a) E[T]_floor(deployed step) reproduces fern #274's 3.9914 within tol.
    a = abs(et_floor(1.0) - HONEST_ET_FLOOR_274) < TOL

    # (b) denken #271 width NO-GO (M*=32 -> 479.6) drops out of the M-coupling as a special case:
    #     reproduce 479.5708 from the step model + the deliverable E[T](32), and confirm < 500
    #     AND that the deliverable E[T](32) is below the M=32 floor.
    et32 = ceiling["implied_width_et32"]
    repro_m32 = official(et32, step_rel(32))
    b = (abs(repro_m32 - DENKEN271_OFFICIAL_M32) < 1e-4
         and repro_m32 < TARGET_TPS
         and et32 < et_floor(step_rel(32)))

    # (c) composition official = K_cal*(E[T]/step)*tau reproduces 481.53 at the deployed point
    #     (deployed step_norm = tau; tau cancels -> K_cal*E_T_REAL), within #274's 1% tol.
    rep_lit = official_literal(E_T_REAL, TAU_LO)
    rep_clean = official(E_T_REAL, 1.0)
    c = (abs(rep_lit - BASE_TPS) / BASE_TPS < RELTOL
         and abs(rep_clean - BASE_TPS) / BASE_TPS < RELTOL
         and abs(rep_lit - rep_clean) < TOL)   # literal == clean (tau cancels)

    # (d) ubel 73%/0.804 applied consistently: public->private LOWERS E[T]_real (raises the needed
    #     public E[T]); the central private ceiling equals priv*E_T_REAL at M=8.
    central_equiv = ceiling["best_achievable_et_real_by_priv_factor"][
        _pf_key(PRIV_FACTOR_CENTRAL)]["best_e_t_real_deployed_equiv"]
    d = (PRIV_FACTOR_CENTRAL < 1.0
         and central_equiv < E_T_REAL
         and abs(central_equiv - PRIV_FACTOR_CENTRAL * E_T_REAL) < TOL
         and fmap["crossing"]["public_et_needed_at_deployed_step"] > E_T_REAL)

    # (e) NaN-clean over the whole payload (filled later); here check the scalar headline outputs.
    headline_scalars = [et_floor(1.0), repro_m32, rep_lit, rep_clean, central_equiv,
                        fmap["min_et_real_needed"], fmap["best_achievable_et_real_central"]]
    e = all(_finite(x) for x in headline_scalars)

    # (f) exact imports.
    f = (BASE_TPS == 481.53 and abs(LAMBDA1_CEIL - 520.9527323111674) < 1e-9
         and abs(K_CAL - 125.26795005202914) < 1e-12 and STEP_SERVED_US == 1218.2
         and G_D_DEPLOYED == 0.0191 and abs(HONEST_ET_FLOOR_274 - 3.9914439391107512) < 1e-12
         and abs(PHI_LO - 0.125) < 0.001 and abs(PHI_HI - 0.735) < 0.001
         and abs(TAU_LO - 1.03524) < 1e-9)

    # (g) the verdict carries the composition-priced + ubel-sensitivity caveats.
    cav = " ".join(verdict_sensitivity(ceiling)["caveats"]).lower()
    g = ("composition-priced" in cav and "0 tps" in cav and "ubel #263 private factor" in cav)

    conds = {"a_floor_reproduces_274_3991": bool(a),
             "b_width_nogo_m32_479_drops_out": bool(b),
             "c_composition_reproduces_481_53": bool(c),
             "d_ubel_private_applied_consistently": bool(d),
             "e_headline_scalars_finite": bool(e),
             "f_anchors_imported_exact": bool(f),
             "g_verdict_carries_caveats": bool(g)}
    return {"conditions": conds, "partial_passes_a_to_g": all(conds.values()),
            "detail": {"floor_deployed": et_floor(1.0), "repro_m32": repro_m32,
                       "rep_literal": rep_lit, "rep_clean": rep_clean,
                       "central_priv_equiv": central_equiv, "implied_et32": et32}}


# ============================================================================
# NaN walk
# ============================================================================
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


# ============================================================================
# Assemble
# ============================================================================
def synthesize() -> dict[str, Any]:
    floor_tab = floor_vs_stepshave()
    ceiling = et_real_ceiling()
    fmap = feasibility_map(ceiling)
    verdict = closure_verdict(fmap)
    sens = verdict_sensitivity(ceiling)
    st = self_tests(floor_tab, ceiling, fmap, verdict)

    handoff = (
        "honest 500 needs E[T]_real >= %.4f at the (step-shave-lowered) deployed step; the best "
        "MEASURED-achievable E[T]_real (private-%.3f-degraded) is %.4f, so the E[T]-raise axis is "
        "CLOSED -- Path-A fully closed on all three axes, 481.53 cannot reach 500 by any "
        "speculative lever under the measured constraints, most sensitive to ubel #263's private "
        "factor (and the missing public E[T] raise that land #245's build would have to deliver)."
        % (fmap["min_et_real_needed"], PRIV_FACTOR_CENTRAL,
           fmap["best_achievable_et_real_central"]))

    return {
        "constants": {
            "BASE_TPS": BASE_TPS, "TARGET_TPS": TARGET_TPS, "K_CAL": K_CAL,
            "E_T_REAL": E_T_REAL, "LAMBDA1_CEIL": LAMBDA1_CEIL, "TAU_LO": TAU_LO,
            "HONEST_ET_FLOOR_274": HONEST_ET_FLOOR_274, "PHI_LO": PHI_LO, "PHI_HI": PHI_HI,
            "STEP_SERVED_US": STEP_SERVED_US, "G_D_DEPLOYED": G_D_DEPLOYED,
            "VERIFY_US": VERIFY_US, "N_TREE": N_TREE, "K_SPEC": K_SPEC,
            "DENKEN271_OFFICIAL_M32": DENKEN271_OFFICIAL_M32,
            "UBEL263_PRIV_PUB_RATIO": UBEL263_PRIV_PUB_RATIO,
            "PR_STATED_PRIV_FACTOR": PR_STATED_PRIV_FACTOR,
            "KANNA269_FOLD_GAIN_PCT": KANNA269_FOLD_GAIN_PCT,
            "KANNA_VERIFY_ROOFLINE_PCT": KANNA_VERIFY_ROOFLINE_PCT,
            "STEPSHAVE_MAX_MEASURED_PCT": STEPSHAVE_MAX_MEASURED_PCT,
            "LAND245_FAITHFUL_M16_ET": LAND245_FAITHFUL_M16_ET,
            "kanna_verify_roofline_landed": KANNA_VERIFY_ROOFLINE_LANDED,
            "kanna_verify_roofline_run": KANNA_VERIFY_ROOFLINE_RUN,
        },
        "step1_floor_vs_stepshave": floor_tab,
        "step2_et_real_ceiling": ceiling,
        "step3_feasibility_map": fmap,
        "step4_closure_verdict": verdict,
        "step5_verdict_sensitivity": sens,
        "self_test": st,
        "handoff": handoff,
    }


# ============================================================================
# W&B logging (repo helpers; never fatal)
# ============================================================================
def _maybe_log_wandb(args, payload: dict) -> str:
    if not getattr(args, "wandb_name", None):
        return ""
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                           log_json_artifact, log_summary)
    except Exception as exc:  # noqa: BLE001
        print(f"[et-raise] wandb logging unavailable: {exc}", flush=True)
        return ""

    syn = payload["synthesis"]
    fmap = syn["step3_feasibility_map"]
    vd = syn["step4_closure_verdict"]
    run = init_wandb_run(
        job_type="et-raise-feasibility", agent="fern", name=args.wandb_name,
        group=args.wandb_group,
        tags=["et-raise-feasibility", "path-a-closure", "composition", "validity-gate",
              "feasibility-envelope", "analytic"],
        config={"target_official": TARGET_TPS, "base_tps": BASE_TPS, "k_cal": K_CAL,
                "honest_et_floor": HONEST_ET_FLOOR_274, "phi_lo": PHI_LO, "phi_hi": PHI_HI,
                "tau_lo": TAU_LO, "priv_factor_central": PRIV_FACTOR_CENTRAL,
                "denken271_official_m32": DENKEN271_OFFICIAL_M32,
                "kanna269_fold_gain_pct": KANNA269_FOLD_GAIN_PCT,
                "kanna_verify_roofline_landed": KANNA_VERIFY_ROOFLINE_LANDED,
                "kanna_verify_roofline_pct": KANNA_VERIFY_ROOFLINE_PCT,
                "stepshave_max_measured_pct": STEPSHAVE_MAX_MEASURED_PCT,
                "wandb_group": args.wandb_group})
    if run is None:
        print("[et-raise] wandb: no run (no WANDB_API_KEY/mode) -- skipping", flush=True)
        return ""

    summary = {
        "et_raise_feasibility_self_test_passes": int(bool(payload["self_test_passes"])),
        "path_a_fully_closed": int(bool(vd["path_a_fully_closed"])),
        "path_a_axis_draft_cut_closed": int(bool(vd["path_a_axes_closed"]["draft_cut"])),
        "path_a_axis_width_closed": int(bool(vd["path_a_axes_closed"]["width"])),
        "path_a_axis_et_raise_closed": int(bool(vd["path_a_axes_closed"]["et_raise"])),
        "go_region_exists": int(bool(fmap["go_region_exists"])),
        "n_go_cells_realizable": fmap["n_go_cells"],
        "n_counterfactual_go_cells": fmap["n_counterfactual_go_cells"],
        "kanna_verify_roofline_pct": KANNA_VERIFY_ROOFLINE_PCT,
        "stepshave_max_measured_pct": STEPSHAVE_MAX_MEASURED_PCT,
        "min_et_real_needed": fmap["min_et_real_needed"],
        "min_et_real_needed_deployed": fmap["min_et_real_needed_deployed"],
        "best_achievable_et_real": fmap["best_achievable_et_real_central"],
        "et_real_gap": fmap["best_achievable_et_real_central"] - fmap["min_et_real_needed"],
        "deployed_floor": syn["step1_floor_vs_stepshave"]["deployed_floor"],
        "lowest_floor_over_grid": syn["step1_floor_vs_stepshave"]["lowest_floor_over_grid"],
        "implied_width_et32": syn["step2_et_real_ceiling"]["implied_width_et32"],
        "width_m32_official": DENKEN271_OFFICIAL_M32,
        "net_maximising_M": syn["step2_et_real_ceiling"]["net_maximising"]["M"],
        "net_maximising_official_public": syn["step2_et_real_ceiling"]["net_maximising"]["net_official_public"],
        "public_et_needed_at_deployed_step": fmap["crossing"]["public_et_needed_at_deployed_step"],
        "public_et_raise_gap_deployed": fmap["crossing"]["public_et_raise_gap_deployed"],
        "priv_factor_central": PRIV_FACTOR_CENTRAL,
        "priv_factor_needed_no_raise_deployed": fmap["crossing"]["priv_factor_needed_no_raise_deployed"],
        "any_input_flips_to_go": int(bool(syn["step5_verdict_sensitivity"]["any_input_flips_to_go"])),
        "nan_clean": int(bool(payload["nan_clean"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v)) for k, v in syn["self_test"]["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="et_raise_feasibility_envelope_result",
                      artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[et-raise] wandb logged: {summary}", flush=True)
    return run.id if run is not None else ""


# ============================================================================
# Console report
# ============================================================================
def _print_report(syn: dict) -> None:
    c = syn["constants"]
    f1 = syn["step1_floor_vs_stepshave"]
    f2 = syn["step2_et_real_ceiling"]
    f3 = syn["step3_feasibility_map"]
    vd = syn["step4_closure_verdict"]
    st = syn["self_test"]
    print("\n" + "=" * 96, flush=True)
    print("E[T]-RAISE FEASIBILITY (PR #281) -- the three-axis Path-A closure verdict", flush=True)
    print("=" * 96, flush=True)
    print(f"  K_cal={c['K_CAL']:.5f}  E_T_REAL={c['E_T_REAL']:.5f}  honest-500 floor={f1['deployed_floor']:.4f}  "
          f"phi=[{c['PHI_LO']:.4f},{c['PHI_HI']:.4f}]  tau={c['TAU_LO']}", flush=True)
    print("-" * 96, flush=True)
    print("  (1) honest-500 E[T] floor vs greedy-safe step-shave (phi-corrected):", flush=True)
    for r in f1["rows"]:
        kn = "measured" if r["known"] else "stress>ceiling"
        print(f"      comp={r['stepshave_comp_pct']:6.2f}%  floor[phi_lo]={r['phi_lo']['et_floor']:.4f}  "
              f"floor[phi_hi]={r['phi_hi']['et_floor']:.4f}   [{kn}]", flush=True)
    print("-" * 96, flush=True)
    print("  (2) M-coupling: NET public official per width M (full-forward, priv=1):", flush=True)
    for name, m in f2["m_coupling"].items():
        print(f"      {name:20s} M={m['M']:2d}  step_rel={m['step_rel']:.4f}  E[T]_pub={m['e_t_public']:.4f}  "
              f"net_official={m['net_official_public']:7.2f}  clears500={m['clears_500_public']}", flush=True)
    print(f"      net-maximising M = {f2['net_maximising']['M']} ({f2['net_maximising']['net_official_public']:.2f}); "
          f"width NO-GO: M=32 -> {c['DENKEN271_OFFICIAL_M32']:.2f} (floor@M32={f2['width_nogo_special_case']['et_floor_at_M32']:.4f} "
          f"> deliverable {f2['width_nogo_special_case']['deliverable_et32']:.4f})", flush=True)
    print("-" * 96, flush=True)
    print("  (3) decisive map (E[T]_real x step-shave):", flush=True)
    print(f"      min E[T]_real needed (kanna+#269 max measured shave {STEPSHAVE_MAX_MEASURED_PCT:.2f}%, "
          f"phi_hi) = {f3['min_et_real_needed']:.4f}", flush=True)
    print(f"      best achievable E[T]_real (private {f3['central_priv_factor']:.4f})   = "
          f"{f3['best_achievable_et_real_central']:.4f}", flush=True)
    print(f"      GAP = {f3['best_achievable_et_real_central'] - f3['min_et_real_needed']:+.4f}   "
          f"go_region_exists = {f3['go_region_exists']} (realizable cells: {f3['n_go_cells']}, "
          f"counterfactual priv=1.0 cells: {f3['n_counterfactual_go_cells']})", flush=True)
    print(f"      public E[T] needed @deployed step (priv {f3['central_priv_factor']:.4f}) = "
          f"{f3['crossing']['public_et_needed_at_deployed_step']:.4f}  "
          f"(measured ceiling {c['E_T_REAL']:.4f}, gap {f3['crossing']['public_et_raise_gap_deployed']:+.4f})",
          flush=True)
    print("-" * 96, flush=True)
    print("  (4) THREE-AXIS CLOSURE VERDICT:", flush=True)
    print(f"      axes closed: {vd['path_a_axes_closed']}", flush=True)
    print(f"      path_a_fully_closed = {vd['path_a_fully_closed']}", flush=True)
    print(f"      {vd['headline']}", flush=True)
    print("-" * 96, flush=True)
    print("  (6) SELF-TEST (PRIMARY):", flush=True)
    for k, v in st["conditions"].items():
        print(f"        - {k}: {v}", flush=True)
    print("=" * 96, flush=True)
    print(f"  HAND-OFF: {syn['handoff']}", flush=True)


# ============================================================================
# CLI
# ============================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="et-raise-feasibility")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    syn = synthesize()
    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {"created_at": created_at, "pr": 281, "agent": "fern",
               "kind": "et-raise-feasibility-envelope (CPU analytic; no GPU/HF/submission)",
               "synthesis": syn}

    nan_bad = _nan_paths(payload)
    payload["nan_clean"] = not nan_bad
    if nan_bad:
        print(f"[et-raise] WARNING non-finite values at: {nan_bad}", flush=True)
    syn["self_test"]["conditions"]["e_headline_scalars_finite"] = bool(
        syn["self_test"]["conditions"]["e_headline_scalars_finite"] and payload["nan_clean"])

    cond = syn["self_test"]["conditions"]
    self_test_passes = bool(all(cond.values()) and payload["nan_clean"])
    payload["self_test_passes"] = self_test_passes
    syn["self_test"]["et_raise_feasibility_self_test_passes"] = self_test_passes

    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload["peak_mem_mib"] = round(peak_kib / 1024.0, 3)

    _print_report(syn)
    fmap = syn["step3_feasibility_map"]
    vd = syn["step4_closure_verdict"]
    print(f"\n  PRIMARY et_raise_feasibility_self_test_passes = {self_test_passes}", flush=True)
    print(f"  TEST    path_a_fully_closed = {vd['path_a_fully_closed']}", flush=True)
    print(f"  min_et_real_needed = {fmap['min_et_real_needed']:.4f}   "
          f"best_achievable_et_real = {fmap['best_achievable_et_real_central']:.4f}", flush=True)
    print(f"  peak_mem_mib = {payload['peak_mem_mib']}", flush=True)

    wandb_id = ""
    if not args.no_wandb:
        wandb_id = _maybe_log_wandb(args, payload)
    payload["wandb_run_ids"] = [wandb_id] if wandb_id else []

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[et-raise] wrote {out_path}", flush=True)

    if args.self_test:
        ok = self_test_passes and payload["nan_clean"]
        print(f"[et-raise] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
