#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Demand route ALONE to 500 on the corrected realized strict base 471.42 (PR #396, denken).

THE QUESTION (the deployable, no-kernel sibling of #392):
  wirbel #390 (`5y64zbjz`) re-attributed the served-strict ceiling: the OLD [357.32, 469.68]
  bracket double-counted a phantom bf16 body+lm_head determinization. The CORRECTED realized
  shippable strict decode base is 471.41635 TPS (deployed servable basis), gap_to_500 = 28.5837,
  supply-alone does NOT clear 500. #392 (`2evhfxi7`) then showed the COMBINED route -- cb3 body
  shrink (a flagged source-build) lifts 469.68 -> 512.60, leaving a residual demand of only
  +0.0117 d-cov. But the cb3 supply leg is KERNEL-CONTINGENT.

  THE DEPLOYABLE ALTERNATIVE is the DEMAND route ALONE: raise drafter coverage -> acceptance ->
  E[T] -> TPS, entirely within the #336 robust budget, with NO served-kernel change. The
  decision-critical question: on the corrected realized strict base 471.41635, how much drafter
  d-cov closes the FULL 28.5837 TPS to 500 with NO cb3 supply, and is that inside the #336 robust
  budget (+0.031) under the #389-MEASURED attention slope (`fqt33bj3`, 0.353x the #386 pessimistic
  interpolation, which just REFUTED the #386 breach -- all corners clear 3.2%)?

THE ANSWER (decision-critical, pure-CPU analytic):
  KNIFE-EDGE. In the bare deployed-basis frame (#390's frame: 471.42 vs literal-500) demand-alone
  closes the 28.58 gap with required_dcov = 0.02946 -- INSIDE the +0.031 budget, but consuming 95%
  of it (only 0.0016 d-cov headroom). At the FULL +0.031 budget demand-alone buys 501.53 TPS
  (+1.53 over 500). BUT it is NOT robust to the private attention-identity charge: even the
  MINIMUM #389-MEASURED irreducible attn floor (central 0.5764%, which itself CLEARS the 3.2%
  identity margin) charges the base enough to push required_dcov to 0.03244 -- just OUTSIDE +0.031.
  The #386-pessimistic floor (3.5235%, REFUTED by #389) would have been a HARD bust (0.0483); the
  #389 measurement softens it to a MILD bust. So demand-alone is feasible ONLY if the public->
  private gap is ignored entirely; any realistic private realization charge busts the budget.
  CONCLUSION: demand-alone reaches deployed-basis-500 within budget but with no robustness margin
  -- it needs the small cb3 supply assist (#392 combined route, residual +0.0117) for a safe
  private-500 path.

WHAT THIS IS / IS NOT:
  Pure-CPU analytic card (stdlib math). 0 GPU, 0 official TPS, 0 HF Job, NO served-file change,
  NO submission. BASELINE 481.53 TPS / PPL 2.3772 UNCHANGED. Reuses denken #392's MTP-loop
  decomposition + #289 per-depth acceptance ladder + #387 anchor (deployed drafter = MTP K=7,
  E[T]_realized 3.844) + the #383/#387 program coverage->E[T] central secant. Only the DEMAND side
  moves: T_step (incl. T_verify) is HELD FIXED -- no supply shrink. The #389 attention slope enters
  ONLY as the robustness stress on the realized base, NOT as a step-time inflation (raising E[T]
  does not change the per-step average context length, so T_step is physically fixed).

REPRODUCE (0-GPU):
    cd target/ && .venv/bin/python research/validity/demand_alone_500_budget/\
demand_alone_500_budget.py --self-test
    cd target/ && .venv/bin/python research/validity/demand_alone_500_budget/\
demand_alone_500_budget.py \
      --wandb_group demand-alone-500-budget --wandb_name denken/demand-alone-500-budget
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# ===========================================================================
# Section 0 -- banked anchors (imported EXACTLY from merged advisor-branch cards / PR #396 body)
# ===========================================================================

# ---- #390 (5y64zbjz) corrected realized strict base + literal-500 gap -----------------------------
# strict_ceiling_corrected_rollup_results.json: realized_shippable_strict_tps_decode / gap_to_500_tps.
REALIZED_STRICT_BASE_390: float = 471.41634950257713    # deployed servable-basis strict decode TPS
GAP_TO_500_390: float = 28.583650497422866              # = 500 - REALIZED_STRICT_BASE_390 (banked)
SHIPPABLE_STRICT_CEILING_390: float = 509.77660237793333  # corrected strict ceiling (Arm A, lambda<1)
FLOOR_BASE_378: float = 469.6847174760462               # OLD #378 better-case floor (#392's BAND_FLOOR)

# ---- #389 (fqt33bj3) per-L attention slope under VBI=1 (the robustness stress on the base) --------
# per_l_attention_vbi1_results.json: measured local-penalty slope vs the #386/#375 interpolation, and
# the irreducible attn-identity gap floor per private-eval corner (all clear the 3.2% identity margin).
CEILING_500_389: float = 520.953                         # ladder lambda=1 ceiling (ladder_constants)
MEASURED_SLOPE_389: float = 0.0010710812125055698        # measured local attn-penalty slope (per L)
INTERP_386_SLOPE: float = 0.003037485347520844           # #386/#375 pessimistic interpolation slope
SLOPE_RATIO_389: float = 0.35262103021493496             # measured / interp == "0.353x" (PR body)
# irreducible attn-identity gap floor (abs %), MEASURED corners (floor_full_528); all clear 3.2%.
MEAS_FLOOR_BANKED_389_PCT: float = 0.0                   # banked corner L=528
MEAS_FLOOR_CENTRAL_389_PCT: float = 0.5763925836099941   # central corner L=578 (modeled private len)
MEAS_FLOOR_PESS_389_PCT: float = 1.272268511379609       # pessimistic corner L=658 (worst clearing)
# #386/#375 INTERPOLATED corners (roundtrip_386); pessimistic BREACHES 3.2% -- the refuted breach.
INTERP_FLOOR_CENTRAL_386_PCT: float = 1.3097036287951451
INTERP_FLOOR_PESS_386_PCT: float = 3.523494549873982     # > 3.2 -> the #386 pessimistic breach
IDENTITY_MARGIN_PCT: float = 3.2                         # the #389 identity margin all corners clear
ALL_CORNERS_CLEAR_3P2_389: bool = True                   # #389 floor_full_528.all_corners_clear_3p2
PESS_BREACHES_3P2_389: bool = False                      # #389 floor_full_528.pessimistic_breaches_3p2

# ---- #289 (fi34s269) DEPLOYED MTP per-position conditional acceptance ladder a_1..a_7 (K=7) -------
# The deployed spec-tree per-DEPTH conditional-accept profile (== #392's LADDER_289). E[T] anchors
# the realized accept length; instruction-4 sanity: this ladder's E[T] must match E_T_REALIZED 3.844.
LADDER_289: list[float] = [
    0.7292532942898975, 0.759556697719242, 0.7929794882639035, 0.8228,
    0.8348727920920435, 0.8357919254658385, 0.8464932652113331,
]

# ---- #387/#383 public<->private inversion plumbing + #336 demand budget ---------------------------
MU_P: float = 481.53                 # deployed public TPS (PR #52, 2x9fm2zx)
MU_V: float = 460.85                 # organizer private-verified TPS for the same submission
GAP_MEASURED: float = 1.0 - MU_V / MU_P              # 0.042946 public->private gap (full)
K_CAL: float = 125.26795005202914                    # steps/s; official TPS = E[T] * K_cal (#344)
E_T_REALIZED: float = MU_P / K_CAL                   # 3.844 realized accept length at deployed point
ET_PUBLIC_500: float = 500.0 / K_CAL                 # 3.99144 (E[T] at the speed-500 bar)
ET_DEPLOYED: float = MU_P / K_CAL                     # 3.844 (E[T] at deployed 481.53)
COV_PRIOR: float = 0.8902659519153152                # #336/#330 modeled top-4 coverage anchor c0
IDENTITY_BAR: float = 0.9213011665456927             # #336 greedy-identity coverage bar
COV_BUDGET_336: float = 0.031035214630377506         # #336 trainable coverage headroom (bar - prior)
CSTAR_CENTRAL: float = 0.9089                         # #340 c* central (program coverage->E[T] secant)
TARGET: float = 500.0
# #383/#387 program coverage->E[T] central secant (anchor-coupled at the central prior). Same slope
# as #392; a property of the coverage->E[T] map, base-independent (~7.91 E[T] per unit coverage).
S_CENTRAL: float = (ET_PUBLIC_500 - ET_DEPLOYED) / (CSTAR_CENTRAL - COV_PRIOR)
# Cross-checks: #392 residual-after-supply, #383 demand-alone-to-private-500.
PUBLISHED_392_RESIDUAL_DCOV: float = 0.0117          # #392 combined-route residual demand (m1 tier)
PUBLISHED_383_DEMAND_ALONE_PRIV_DCOV: float = 0.05716864498666053  # #383 demand-alone @469.68 -> priv-500


# ===========================================================================
# Section 1 -- E[T] from the #289 MTP ladder (the acceptance-weighted step)
# ===========================================================================

def expected_accepted(ladder: list[float]) -> float:
    """E[number of accepted DRAFT tokens] per spec-decode step = sum_k prod_{j<=k} a_j.

    Accepting depth k requires accepting all shallower depths (conditional ladder)."""
    cum = 1.0
    acc = 0.0
    for a in ladder:
        cum *= a
        acc += cum
    return acc


def expected_tokens_per_step(ladder: list[float]) -> float:
    """E[T] = tokens emitted per step = 1 (always-emitted verify token) + E[accepted]."""
    return 1.0 + expected_accepted(ladder)


# ===========================================================================
# Section 2 -- the demand-alone composition: TPS = (1 + E[accepted]) / T_step, T_step FIXED
# ===========================================================================

def tps_from_et(base_tps: float, base_et: float, et: float) -> float:
    """Demand-alone served TPS at acceptance length `et`, holding T_step (incl. T_verify) FIXED.

    served TPS = (1 + E[accepted]) / T_step = E[T] / T_step. With ONLY the demand side moving,
    T_step is fixed (raising acceptance does not change the per-step average context length, so the
    verify-forward attention cost is unchanged -- the #389 slope enters as a base stress, not here).
    Hence TPS scales linearly in E[T]: TPS(et) = base_tps * et / base_et."""
    return base_tps * et / base_et


def required_dcov_demand_alone(base_tps: float, target: float = TARGET,
                               base_et: float = E_T_REALIZED, slope: float = S_CENTRAL) -> float:
    """Delta-coverage the DEMAND side alone must add to lift `base_tps` -> `target`, T_step fixed.

    TPS prop E[T] => needed E[T] = base_et * target/base_tps; delta_et = base_et*(target/base_tps-1).
    Coverage->E[T] via the program central secant: dcov = delta_et / slope."""
    delta_et = base_et * (target / base_tps - 1.0)
    return delta_et / slope


def tps_at_dcov(base_tps: float, dcov: float,
                base_et: float = E_T_REALIZED, slope: float = S_CENTRAL) -> float:
    """Demand-alone served TPS after spending `dcov` coverage (T_step fixed)."""
    return tps_from_et(base_tps, base_et, base_et + slope * dcov)


# ===========================================================================
# Section 3 -- deliverables 1-3: bare demand-alone-500 on the corrected base
# ===========================================================================

def demand_alone_card() -> dict:
    """The bare deployed-basis demand-alone inversion on the #390 corrected strict base 471.42."""
    base = REALIZED_STRICT_BASE_390
    gap = TARGET - base
    req = required_dcov_demand_alone(base)
    tps_full = tps_at_dcov(base, COV_BUDGET_336)
    return {
        "corrected_base_tps": base,
        "gap_to_500_corrected": gap,
        "required_dcov_demand_alone_500": req,
        "dcov_budget_336": COV_BUDGET_336,
        "demand_alone_500_inside_budget": bool(req <= COV_BUDGET_336),
        "required_frac_of_336_budget": req / COV_BUDGET_336,
        "budget_headroom_dcov": COV_BUDGET_336 - req,
        "tps_at_full_dcov_budget_demand_alone": tps_full,
        "demand_alone_reaches_500": bool(tps_full >= TARGET),
        "demand_alone_margin_tps": tps_full - TARGET,
    }


# ===========================================================================
# Section 4 -- deliverable 2 (robustness): the #389 attention-slope stress on the base
# ===========================================================================

def robustness_card() -> dict:
    """Stress the realized base by the irreducible attn-identity floor (the public->private charge),
    under the #389-MEASURED slope and the #386-pessimistic (refuted) slope, and re-invert.

    The bare 471.42 is the deployed (servable, public) basis. The #389 irreducible attn floor is the
    MINIMUM unavoidable private haircut from attention identity (the full public->private gap is the
    larger 4.29%). robust_under_389_slope := demand-alone-500 stays inside +0.031 even after the
    #389-MEASURED *central* (modeled-private-corner) attn-identity charge -- the minimal honest
    private realization. If even that minimum busts, demand-alone is not a robust private-500 route."""
    base = REALIZED_STRICT_BASE_390
    charges = {
        "bare_deployed_basis": 0.0,
        "meas_central_389": MEAS_FLOOR_CENTRAL_389_PCT,      # min honest private charge (clears 3.2%)
        "meas_pessimistic_389": MEAS_FLOOR_PESS_389_PCT,     # #389 worst clearing corner
        "interp_pessimistic_386": INTERP_FLOOR_PESS_386_PCT,  # refuted breach (>3.2%)
        "full_public_private_gap": GAP_MEASURED * 100.0,      # entire 4.29% gap (-> ~ #383 priv-500)
    }
    rows: dict[str, dict] = {}
    for name, fl_pct in charges.items():
        bc = base * (1.0 - fl_pct / 100.0)
        req = required_dcov_demand_alone(bc)
        rows[name] = {
            "attn_floor_charge_pct": fl_pct,
            "charged_base_tps": bc,
            "required_dcov": req,
            "inside_budget": bool(req <= COV_BUDGET_336),
            "frac_of_336_budget": req / COV_BUDGET_336,
        }
    robust = rows["meas_central_389"]["inside_budget"]
    # Reading (A) -- distinct from the headline robustness: is the BASE 471.42 itself valid under
    # the #389 slope? YES -- the measured slope is milder than the #386 interpolation (ratio<1), the
    # measured floor is below the interpolated floor, and all corners clear the 3.2% identity margin,
    # so the base does NOT degrade to the refuted #386-pessimistic value. (This is why 471.42 is a
    # legitimate base.) The HEADLINE robust_under_389_slope is reading (B): does the demand-alone-500
    # CONCLUSION survive the private attn-identity charge the slope implies -- and it does NOT.
    base_holds = bool(SLOPE_RATIO_389 < 1.0 and MEAS_FLOOR_PESS_389_PCT < INTERP_FLOOR_PESS_386_PCT
                      and ALL_CORNERS_CLEAR_3P2_389 and not PESS_BREACHES_3P2_389)
    return {
        "measured_slope_389": MEASURED_SLOPE_389,
        "interp_386_slope": INTERP_386_SLOPE,
        "slope_ratio_389": SLOPE_RATIO_389,
        "all_corners_clear_3p2_389": ALL_CORNERS_CLEAR_3P2_389,
        "pessimistic_breaches_3p2_389": PESS_BREACHES_3P2_389,
        "rows": rows,
        # (A) the base 471.42 is confirmed valid by #389 (slope mild, #386 breach refuted).
        "base_holds_under_389_slope": base_holds,
        # (B) HEADLINE: survives even the MINIMUM #389-measured private attn-identity charge?
        "robust_under_389_slope": bool(robust),
        # decision context: how the #389 measurement changes the verdict vs the #386 interpolation.
        "measured_central_busts": not rows["meas_central_389"]["inside_budget"],
        "interp386_pess_busts_harder_than_meas": (
            rows["interp_pessimistic_386"]["required_dcov"] > rows["meas_pessimistic_389"]["required_dcov"]),
        "knife_edge_no_margin": (rows["bare_deployed_basis"]["inside_budget"]
                                 and not rows["meas_central_389"]["inside_budget"]),
    }


# ===========================================================================
# Section 5 -- self-tests (>=20 checks)
# ===========================================================================

def _finite(x: float) -> bool:
    return isinstance(x, (int, float)) and not (math.isnan(x) or math.isinf(x))


def run_self_tests(et: float, card: dict, rob: dict) -> dict:
    c: dict[str, bool] = {}
    base = REALIZED_STRICT_BASE_390

    # a) provenance: base / gap / ceiling / budget match the banked #390/#389/#336 anchors.
    c["a_base_matches_390"] = abs(base - 471.41634950257713) < 1e-9
    c["a_gap_matches_390_banked"] = abs(card["gap_to_500_corrected"] - GAP_TO_500_390) < 1e-6
    c["a_gap_rounds_28p58"] = round(card["gap_to_500_corrected"], 2) == 28.58
    c["a_budget_is_336_0p031"] = round(COV_BUDGET_336, 3) == 0.031
    c["a_secant_positive"] = S_CENTRAL > 0.0
    c["a_ceiling_500_389_above_500"] = CEILING_500_389 > TARGET

    # b) #289 ladder: K=7, in (0,1), MONOTONE non-decreasing; E[T] matches E_T_REALIZED (instr-4).
    c["b_ladder_len_7"] = len(LADDER_289) == 7
    c["b_ladder_in_unit"] = all(0.0 < a < 1.0 for a in LADDER_289)
    c["b_ladder_monotone_nondecreasing"] = all(LADDER_289[i] <= LADDER_289[i + 1]
                                               for i in range(len(LADDER_289) - 1))
    c["b_et_ladder_matches_realized"] = abs(et - E_T_REALIZED) / E_T_REALIZED < 0.01   # instruction 4
    c["b_et_in_1_to_8"] = 1.0 < et < 8.0

    # c) demand-alone composition: TPS prop E[T] with T_step FIXED (identity + monotone + inversion).
    c["c_tps_at_zero_dcov_is_base"] = abs(tps_at_dcov(base, 0.0) - base) < 1e-9
    c["c_tps_monotone_in_dcov"] = tps_at_dcov(base, 0.02) > tps_at_dcov(base, 0.0)
    c["c_required_dcov_hits_500"] = abs(
        tps_at_dcov(base, card["required_dcov_demand_alone_500"]) - TARGET) < 1e-6
    c["c_full_budget_tps_consistent"] = abs(
        card["tps_at_full_dcov_budget_demand_alone"] - tps_at_dcov(base, COV_BUDGET_336)) < 1e-9

    # d) deliverable 1: required dcov is positive, inside budget, < the full-gap private demand.
    c["d_required_dcov_positive"] = card["required_dcov_demand_alone_500"] > 0.0
    c["d_required_inside_budget"] = card["demand_alone_500_inside_budget"]
    c["d_required_lt_full_gap_demand"] = (card["required_dcov_demand_alone_500"]
                                          < rob["rows"]["full_public_private_gap"]["required_dcov"])
    c["d_required_lt_383_priv_demand"] = (card["required_dcov_demand_alone_500"]
                                          < PUBLISHED_383_DEMAND_ALONE_PRIV_DCOV)
    c["d_required_gt_392_residual"] = (card["required_dcov_demand_alone_500"]
                                       > PUBLISHED_392_RESIDUAL_DCOV)   # demand-alone > residual-after-supply

    # e) deliverable 3: full-budget demand-alone reaches 500 with positive margin (bare frame).
    c["e_full_budget_reaches_500"] = card["demand_alone_reaches_500"]
    c["e_full_budget_margin_positive"] = card["demand_alone_margin_tps"] > 0.0
    c["e_margin_rounds_1p5"] = round(card["demand_alone_margin_tps"], 1) == 1.5

    # f) deliverable 2 (robustness): #389 slope is mild (0.353x), all corners clear; the MEASURED
    #    central attn charge BUSTS the budget (knife-edge) -> robust_under_389_slope is FALSE.
    c["f_slope_ratio_matches_0p353"] = abs(SLOPE_RATIO_389 - 0.35262103021493496) < 1e-9
    c["f_measured_slope_milder_than_interp"] = SLOPE_RATIO_389 < 1.0
    c["f_measured_floor_below_interp"] = MEAS_FLOOR_PESS_389_PCT < INTERP_FLOOR_PESS_386_PCT
    c["f_all_corners_clear_3p2"] = ALL_CORNERS_CLEAR_3P2_389 and not PESS_BREACHES_3P2_389
    c["f_interp386_pess_breaches_3p2"] = INTERP_FLOOR_PESS_386_PCT > IDENTITY_MARGIN_PCT
    c["f_measured_central_busts_budget"] = not rob["rows"]["meas_central_389"]["inside_budget"]
    c["f_robust_under_389_slope_false"] = rob["robust_under_389_slope"] is False
    c["f_base_holds_under_389_slope_true"] = rob["base_holds_under_389_slope"] is True   # reading (A)
    c["f_knife_edge_no_margin"] = rob["knife_edge_no_margin"]

    # g) charge monotonicity: more attn charge -> lower base -> more required dcov (ordered spectrum).
    order = ["bare_deployed_basis", "meas_central_389", "meas_pessimistic_389",
             "interp_pessimistic_386", "full_public_private_gap"]
    reqs = [rob["rows"][k]["required_dcov"] for k in order]
    c["g_required_dcov_monotone_in_charge"] = all(reqs[i] < reqs[i + 1] for i in range(len(reqs) - 1))
    c["g_bare_inside_central_outside"] = (rob["rows"]["bare_deployed_basis"]["inside_budget"]
                                          and not rob["rows"]["meas_central_389"]["inside_budget"])

    # h) numeric hygiene across all headline scalars.
    flat = [et, S_CENTRAL, card["gap_to_500_corrected"], card["required_dcov_demand_alone_500"],
            card["tps_at_full_dcov_budget_demand_alone"], card["demand_alone_margin_tps"],
            rob["rows"]["meas_central_389"]["required_dcov"]]
    c["h_no_nan_inf"] = all(_finite(v) for v in flat)

    passes = all(c.values())
    return {"conditions": c, "n_checks": len(c), "n_passed": sum(1 for v in c.values() if v),
            "passes": passes}


# ===========================================================================
# Section 6 -- report assembly + W&B + CLI
# ===========================================================================

def build_report() -> dict:
    et = expected_tokens_per_step(LADDER_289)
    card = demand_alone_card()
    rob = robustness_card()
    selftest = run_self_tests(et, card, rob)
    return {
        "pr": 396, "agent": "denken", "kind": "demand-alone-500-budget",
        "analysis_only": True, "no_launch": True, "no_hf_job": True, "no_submission": True,
        "no_served_file_change": True, "gpu_used": False, "official_tps": 0,
        "baseline_unchanged_tps": 481.53, "baseline_unchanged_ppl": 2.3772,
        "inputs": {
            "realized_strict_base_390": REALIZED_STRICT_BASE_390, "gap_to_500_390": GAP_TO_500_390,
            "shippable_strict_ceiling_390": SHIPPABLE_STRICT_CEILING_390, "ceiling_500_389": CEILING_500_389,
            "floor_base_378": FLOOR_BASE_378,
            "measured_slope_389": MEASURED_SLOPE_389, "interp_386_slope": INTERP_386_SLOPE,
            "slope_ratio_389": SLOPE_RATIO_389,
            "meas_floor_central_389_pct": MEAS_FLOOR_CENTRAL_389_PCT,
            "meas_floor_pess_389_pct": MEAS_FLOOR_PESS_389_PCT,
            "interp_floor_pess_386_pct": INTERP_FLOOR_PESS_386_PCT,
            "ladder_289": LADDER_289, "mu_p": MU_P, "mu_v": MU_V, "k_cal": K_CAL,
            "gap_measured": GAP_MEASURED, "cov_prior": COV_PRIOR, "identity_bar": IDENTITY_BAR,
            "cov_budget_336": COV_BUDGET_336, "cstar_central": CSTAR_CENTRAL, "s_central": S_CENTRAL,
            "e_t_realized": E_T_REALIZED, "target": TARGET,
            "published_392_residual_dcov": PUBLISHED_392_RESIDUAL_DCOV,
            "published_383_demand_alone_priv_dcov": PUBLISHED_383_DEMAND_ALONE_PRIV_DCOV,
            "source_390_run": "5y64zbjz", "source_389_run": "fqt33bj3", "source_392_run": "2evhfxi7",
            "source_387_run": "z8osvif8", "source_289_run": "fi34s269", "source_383_run": "t68af2yw",
            "source_340_anchor": "cstar_central=0.9089", "source_336_budget": "0.031035",
        },
        "expected_tokens_per_step": et,
        "expected_accepted_draft": et - 1.0,
        "s_central_et_per_cov": S_CENTRAL,
        "demand_alone": card,
        "robustness_389": rob,
        # ---- card-required headline scalars (deliverables; SENPAI-RESULT load-bearing) ----
        "gap_to_500_corrected": card["gap_to_500_corrected"],
        "required_dcov_demand_alone_500": card["required_dcov_demand_alone_500"],
        "dcov_budget_336": COV_BUDGET_336,
        "demand_alone_500_inside_budget": card["demand_alone_500_inside_budget"],
        "tps_at_full_dcov_budget_demand_alone": card["tps_at_full_dcov_budget_demand_alone"],
        "demand_alone_reaches_500": card["demand_alone_reaches_500"],
        "demand_alone_margin_tps": card["demand_alone_margin_tps"],
        "robust_under_389_slope": rob["robust_under_389_slope"],
        "base_holds_under_389_slope": rob["base_holds_under_389_slope"],
        "self_test": selftest,
        "demand_alone_500_self_test_passes": selftest["passes"],
    }


def log_to_wandb(report: dict, group: str, name: str) -> str:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"W&B import failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_unavailable"
    try:
        run = wandb.init(group=group, name=name, config=report["inputs"])
        card, rob = report["demand_alone"], report["robustness_389"]
        wandb.log({
            "summary/gap_to_500_corrected": report["gap_to_500_corrected"],
            "summary/required_dcov_demand_alone_500": report["required_dcov_demand_alone_500"],
            "summary/dcov_budget_336": report["dcov_budget_336"],
            "summary/demand_alone_500_inside_budget": float(report["demand_alone_500_inside_budget"]),
            "summary/required_frac_of_336_budget": card["required_frac_of_336_budget"],
            "summary/budget_headroom_dcov": card["budget_headroom_dcov"],
            "summary/tps_at_full_dcov_budget_demand_alone": report["tps_at_full_dcov_budget_demand_alone"],
            "summary/demand_alone_reaches_500": float(report["demand_alone_reaches_500"]),
            "summary/demand_alone_margin_tps": report["demand_alone_margin_tps"],
            "summary/robust_under_389_slope": float(report["robust_under_389_slope"]),
            "summary/base_holds_under_389_slope": float(rob["base_holds_under_389_slope"]),
            "summary/knife_edge_no_margin": float(rob["knife_edge_no_margin"]),
            "summary/measured_central_busts": float(rob["measured_central_busts"]),
            "summary/slope_ratio_389": rob["slope_ratio_389"],
            "summary/measured_slope_389": rob["measured_slope_389"],
            "summary/interp_386_slope": rob["interp_386_slope"],
            "summary/all_corners_clear_3p2_389": float(rob["all_corners_clear_3p2_389"]),
            "summary/expected_tokens_per_step": report["expected_tokens_per_step"],
            "summary/expected_accepted_draft": report["expected_accepted_draft"],
            "summary/s_central_et_per_cov": report["s_central_et_per_cov"],
            "summary/corrected_base_tps": card["corrected_base_tps"],
            "summary/self_test_passes": float(report["self_test"]["passes"]),
            "summary/self_test_n_checks": float(report["self_test"]["n_checks"]),
            "summary/demand_alone_500_self_test_passes": float(report["demand_alone_500_self_test_passes"]),
        })
        for chname, chrow in rob["rows"].items():
            wandb.log({f"charge/{chname}/attn_floor_charge_pct": chrow["attn_floor_charge_pct"],
                       f"charge/{chname}/charged_base_tps": chrow["charged_base_tps"],
                       f"charge/{chname}/required_dcov": chrow["required_dcov"],
                       f"charge/{chname}/inside_budget": float(chrow["inside_budget"])})
        for cond, val in report["self_test"]["conditions"].items():
            wandb.log({f"test/{cond}": float(bool(val))})
        run_id = run.id
        wandb.finish()
        return run_id
    except Exception as exc:  # noqa: BLE001
        print(f"W&B logging failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_failed"


def print_report(r: dict) -> None:
    card, rob = r["demand_alone"], r["robustness_389"]
    print("\n=== Demand route ALONE to 500 on the corrected strict base (PR #396, denken) ===")
    print(f"corrected strict base (#390 5y64zbjz) = {card['corrected_base_tps']:.4f} TPS   "
          f"gap_to_500 = {card['gap_to_500_corrected']:.4f}   ceiling(lambda=1) = {CEILING_500_389}")
    print(f"#289 ladder E[accepted]={r['expected_accepted_draft']:.4f}  E[T]={r['expected_tokens_per_step']:.4f}  "
          f"(== E_T_REALIZED {E_T_REALIZED:.4f} to {abs(r['expected_tokens_per_step']-E_T_REALIZED)/E_T_REALIZED*100:.2f}%)")
    print(f"program coverage->E[T] central secant S = {r['s_central_et_per_cov']:.4f} E[T]/cov  (T_step FIXED, demand-only)")
    print("\n-- deliverable 1+3: BARE demand-alone-500 (deployed-basis frame, no attn charge) --")
    print(f"  required_dcov_demand_alone_500 = +{card['required_dcov_demand_alone_500']:.5f}  "
          f"(budget +{COV_BUDGET_336:.5f}; uses {card['required_frac_of_336_budget']*100:.1f}%, "
          f"headroom +{card['budget_headroom_dcov']:.5f})")
    print(f"  demand_alone_500_inside_budget = {card['demand_alone_500_inside_budget']}")
    print(f"  TPS at FULL +{COV_BUDGET_336:.4f} budget = {card['tps_at_full_dcov_budget_demand_alone']:.2f}  "
          f"(reaches_500={card['demand_alone_reaches_500']}, margin {card['demand_alone_margin_tps']:+.2f} TPS)")
    print("\n-- deliverable 2: ROBUSTNESS under the #389 attention slope (0.353x #386 interp) --")
    print(f"  measured slope {rob['measured_slope_389']:.6f} / interp {rob['interp_386_slope']:.6f} "
          f"= ratio {rob['slope_ratio_389']:.4f}   all_corners_clear_3p2={rob['all_corners_clear_3p2_389']}  "
          f"pess_breaches_3p2={rob['pessimistic_breaches_3p2_389']}")
    print("  attn-identity charge spectrum (charge%  ->  charged base  required_dcov  inside_budget):")
    for chname, chrow in rob["rows"].items():
        print(f"    {chname:<26}: {chrow['attn_floor_charge_pct']:>6.4f}%  ->  {chrow['charged_base_tps']:.2f}  "
              f"+{chrow['required_dcov']:.5f}  ({chrow['frac_of_336_budget']*100:>5.1f}% budget)  "
              f"inside={chrow['inside_budget']}")
    print(f"\n  base_holds_under_389_slope = {rob['base_holds_under_389_slope']}  "
          f"(A: base 471.42 valid -- #389 slope mild, #386 breach refuted, all corners clear 3.2%)")
    print(f"  robust_under_389_slope     = {rob['robust_under_389_slope']}  "
          f"(B: survives even the MINIMUM #389-measured central {MEAS_FLOOR_CENTRAL_389_PCT:.4f}% attn charge?)")
    print(f"  knife_edge_no_margin = {rob['knife_edge_no_margin']}  "
          f"(bare fits, but the mild measured private attn floor pushes it just outside +0.031)")
    print(f"\nself-test: {r['self_test']['n_passed']}/{r['self_test']['n_checks']} checks  "
          f"demand_alone_500_self_test_passes = {r['demand_alone_500_self_test_passes']}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Demand route ALONE to 500 on the corrected strict base (PR #396).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="0-GPU analytic gate (PR #396 deliverables)")
    ap.add_argument("--reanalyze", action="store_true", help="0-GPU full re-analysis (alias of default)")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="demand-alone-500-budget")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default="denken/demand-alone-500-budget")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", type=str,
                    default="research/validity/demand_alone_500_budget/demand_alone_500_budget_results.json")
    args = ap.parse_args()

    report = build_report()
    print_report(report)

    if args.self_test:
        out = Path("research/validity/demand_alone_500_budget/demand_alone_500_budget_selftest.json")
        out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {out}")
        print(f"\ndemand_alone_500_self_test_passes = {report['self_test']['passes']}")
        return 0 if report["self_test"]["passes"] else 1

    report["wandb_run_id"] = None if args.no_wandb else log_to_wandb(report, args.wandb_group, args.wandb_name)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}  (W&B run {report.get('wandb_run_id')})")

    print("\nSENPAI-RESULT " + json.dumps({
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [report["wandb_run_id"]] if report.get("wandb_run_id") else [],
        "no_hf_job": True, "official_tps": 0.0, "analysis_only": True,
        "gap_to_500_corrected": float(report["gap_to_500_corrected"]),
        "required_dcov_demand_alone_500": float(report["required_dcov_demand_alone_500"]),
        "dcov_budget_336": float(report["dcov_budget_336"]),
        "demand_alone_500_inside_budget": bool(report["demand_alone_500_inside_budget"]),
        "tps_at_full_dcov_budget_demand_alone": float(report["tps_at_full_dcov_budget_demand_alone"]),
        "demand_alone_reaches_500": bool(report["demand_alone_reaches_500"]),
        "demand_alone_margin_tps": float(report["demand_alone_margin_tps"]),
        "robust_under_389_slope": bool(report["robust_under_389_slope"]),
        "demand_alone_500_self_test_passes": bool(report["demand_alone_500_self_test_passes"]),
        "primary_metric": {"name": "demand_alone_500_self_test_passes",
                           "value": float(report["demand_alone_500_self_test_passes"])},
        "test_metric": {"name": "required_dcov_demand_alone_500",
                        "value": float(report["required_dcov_demand_alone_500"])},
    }))
    return 0 if report["self_test"]["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
