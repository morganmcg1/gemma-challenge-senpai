#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Re-price the demand-side residual on the HONEST deployable-strict base (PR #383, denken).

denken #377 sized the demand-side closer to PRIVATE-strict-500 at c >= 0.9010 (+0.00565 central /
+0.0107 robust, "35% of #336's +0.031 budget"). That sizing rests on a PUBLIC base of 518.92 -- the
#366/#370 eta-axis "revival" ceiling -- via the #373 regression-to-the-mean projection
project(518.92; rho=0.9421) = 494.56, residual-to-private-500 = +5.44 TPS.

wirbel #378 (gghmgtk9, just merged) BREAKS that premise. The honest deployable-strict served band
TODAY is full_vbi_today_bracket = [357.32, 469.68] < 500. The only strict byte-exact served knob is
VLLM_BATCH_INVARIANT=1, which imposes bf16 lm_head-BI determinism on the WHOLE step (matches denken
#327's 469.68 bf16 lm_head+attn ceiling). The 518.92 pin needs a KERNEL REBUILD that buys only
~11 TPS (eta_attn = 0.0215, NOT #326's whole-step eta = 0.3141): the dominant strict overhead is
lm_head-BI, not the attention un-pack. So the public-strict base the demand closer transfers from is
<= 469.68 (floor) / ~480.8 (floor + the ~11-TPS attention rebuild), NOT 518.92.

CRUX. Demand-side coverage retrain (a) raises E[T] (acceptance), which lifts the public-strict base
multiplicatively (TPS = E[T] * steps/s), AND (b) shrinks the public->private gap (rho -> 1, but
gap-closure can at best drive private -> public, and only to the #379 irreducible ctxlen floor
0.633%). The question this card answers HONESTLY: starting from a sub-500 public-strict base, can the
demand-side route reach PRIVATE-strict-500 ALONE -- at any coverage within the #336 +0.031 envelope --
or is it NECESSARY-BUT-NOT-SUFFICIENT without a supply-side lm_head-BI lift?

It re-prices the residual on the honest bases, inverts to the required Delta-coverage, and shows the
verdict FLIP (#377 GO -> demand-alone NO-GO) is driven ENTIRELY by the base move 518.92 -> <=480.8,
NOT a harness change (round-trip reproduces #377's +5.44 / c>=0.9010 under the OLD 518.92 base).

NOT a launch, NOT a submission, no served-file change, 0 GPU, 0 official TPS. CPU-analytic (numpy/
math). Run:
    cd target/ && .venv/bin/python research/validity/demand_residual_honest_base/\
demand_residual_honest_base.py --honest-base --reconcile-377 \
      --wandb_group strict-bi-verify-gemm --wandb_name denken/demand-residual-honest-base
  self-test only (0-GPU, no W&B): ... --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ===========================================================================
# Section 0 -- banked anchors (all from merged advisor-branch cards / PR #383 body)
# ===========================================================================

# ---- public<->private anchor pair (PR #52 deployed point; organizer re-run 2026-06-13) --------
MU_P: float = 481.53                 # deployed public TPS (PR #52, 2x9fm2zx); NON-strict
MU_V: float = 460.85                 # organizer private-verified TPS for the same submission
GAP_MEASURED: float = 1.0 - MU_V / MU_P              # 0.042946 -> the "4.295%" public->private gap
K_CAL: float = 125.26795005202914                    # steps/s; official TPS = E[T] * K_cal (#344)
E_T_REALIZED: float = MU_P / K_CAL                   # 3.8444 realized accept length at deployed pt
RHO_DEPLOYED_318: float = 0.9570535584491102         # #318 deployed priv/pub = 1-gap (simple haircut)

# ---- public<->private regression correlation (#347 LB / #300/#310 point) ----------------------
RHO_PRIV: float = 0.9421             # #300/#310 central regression correlation
RHO_LB: float = 0.8038               # #347 lower-bound regression correlation (conservative)

# ---- HONEST deployable-strict bases (wirbel #378 gghmgtk9) -- the WHOLE point of this card -----
BASE_DEPLOYABLE_FLOOR: float = 469.68          # full VBI=1 floor (= denken #327 bf16 lm_head+attn ceiling)
BASE_DEPLOYABLE_OFFTHESHELF: float = 357.32    # full VBI=1 off-the-shelf (4.8x M=1 attn penalty corner)
ETA_ATTN_378: float = 0.0215                   # #378 attention-pin rebuild eta (NOT whole-step 0.3141)
P_HEADLINE: float = 518.9188253620001          # #366/#370 eta-axis "revival" ceiling (#377 reference base)
# the ~11-TPS attention-pin KERNEL REBUILD: floor + eta_attn * revival-ceiling.
ATTN_REBUILD_ROI_TPS: float = ETA_ATTN_378 * P_HEADLINE                       # ~11.16 TPS
BASE_PLUS_ATTN_REBUILD: float = BASE_DEPLOYABLE_FLOOR + ATTN_REBUILD_ROI_TPS  # ~480.84 (PR's ~480.7)

# ---- coverage scalars (#336 / #339 / #340 / #343) ---------------------------------------------
COV_PRIOR: float = 0.8902659519153152                # #336/#339 deployed fusion-head coverage c0
IDENTITY_BAR: float = 0.9213011665456927             # #336/#339 greedy-identity coverage bar
COV_BUDGET_336: float = 0.031035214630377506         # #336 achievable lift (bar - prior) == trainable headroom
CSTAR_CENTRAL: float = 0.9089                         # #340 c* central (program coverage->E[T] secant)
CSTAR_WORST: float = 0.9256                           # #340 c* worst (pessimistic transfer corner)
ET_PUBLIC_500: float = 500.0 / K_CAL                  # 3.99144 (E[T] at the speed-500 bar)
ET_DEPLOYED: float = MU_P / K_CAL                      # 3.84438 (E[T] at the deployed 481.53)

# ---- #289 measured per-position conditional accept profile (int4-ct deployed drafter) ---------
A_D_MEASURED: list[float] = [
    0.7292532942898975,   # a_1 <- first-token CLIFF (weakest conditional, gates all survival)
    0.759556697719242, 0.7929794882639035, 0.8228, 0.8348727920920435,
    0.8357919254658385, 0.8464932652113331,
]
A1_MARGINAL_289: float = 3.9097333761644713          # #377 per_position_dEll_da[0]; #379 slope basis dE[T]/dcov (a1-only)
EAGLE3_PUBLISHED_A1: float = 0.80                     # #308 published EAGLE-3 a_1 envelope

# ---- #379 public->private gap structural decomposition (ubel, 5kpb73tb) -----------------------
GAP_BUCKET_ACCEPTANCE_PCT: float = 85.25171082960598  # acceptance-addressable share of the gap
GAP_BUCKET_CTXLEN_PCT: float = 14.748289170394163     # ctxlen/step share of the gap
GAP_IRREDUCIBLE_FLOOR_PCT: float = 0.6333865388319535 # irreducible ctxlen floor (% absolute)
GAP_IRREDUCIBLE_FLOOR_FRAC: float = GAP_IRREDUCIBLE_FLOOR_PCT / 100.0          # 0.006334
SLOPE_379_TPS_PER_COV: float = 489.76448056537095     # #379 dTPS/dcov = a1_marginal * K_cal (gap/E[T] channel)
RHO_MAX_DEMAND: float = 1.0 - GAP_IRREDUCIBLE_FLOOR_FRAC                       # 0.99367 max priv/pub via gap-closure

# ---- #377 non-iid sizing (the card we re-price) -- W&B 030uc5mk --------------------------------
RESIDUAL_544_377: float = 5.438733615047738          # #373/#377 central residual at the 518.92 base
DELTA_ET_CENTRAL_377: float = 0.04468363487955586    # #373 lever-b central delta_et at 518.92
DCOV_CENTRAL_377: float = 0.005647142947793298       # #377 central coverage delta (point)
DCOV_ROBUST_377: float = 0.010708162797026063        # #377 robust coverage delta (recommended)
REC_TARGET_377: float = 0.9009741147123412           # #377 recommended retrain target (c>=0.9010)
KAPPA_REALIZED_TRANSFER_377: float = 0.6716346752190123  # #377 implied realized coverage->accept transfer
KAPPA_GAP_377: float = GAP_MEASURED / (1.0 - COV_PRIOR)  # #377 |dgap/dcov| co-benefit (0.3914 per unit cov)

# ---- #380 coverage-retrain deliverability (W&B 00oijpwg) + #352 pilot --------------------------
P_DELIVER_CENTRAL_380: float = 0.958                 # central deliverable prob
P_DELIVER_ROBUST_380: float = 0.811                  # robust deliverable prob (pending pilot)
PILOT_GPU_HR_352: float = 25.0                       # #352 robust coverage pilot cost (A10G-GPU-hr)

TARGET: float = 500.0

# Program coverage->E[T] secants (#377): realized transfer the program adopted, NOT iid.
S_CENTRAL: float = (ET_PUBLIC_500 - ET_DEPLOYED) / (CSTAR_CENTRAL - COV_PRIOR)   # ~7.91 E[T]/cov
S_WORST: float = (ET_PUBLIC_500 - ET_DEPLOYED) / (CSTAR_WORST - COV_PRIOR)       # ~4.17 E[T]/cov


# ===========================================================================
# Section 1 -- transfer operators (public-strict base -> private-strict)
# ===========================================================================

def project(P: float, rho: float, g: float = GAP_MEASURED) -> float:
    """#373 regression-to-the-mean projection: private = (1-g)*(mu_P + rho*(P - mu_P)).

    rho=1 recovers the naive proportional (1-g)*P. Extrapolating above the anchor mu_P, rho<1 pulls
    private BELOW proportional (regression haircut). This is the #377 reference transfer; at
    P=518.92, rho=0.9421 it reproduces the +5.44 residual exactly.
    """
    return (1.0 - g) * (MU_P + rho * (P - MU_P))


def haircut(P: float, rho_dep: float = RHO_DEPLOYED_318) -> float:
    """#318 simple deployed-rho haircut: private = rho_dep * P (the gap=fixed cross-check)."""
    return rho_dep * P


def public_for_private_500_regression(rho: float, g: float = GAP_MEASURED) -> float:
    """Invert project(P; rho, g) = 500 -> the PUBLIC ceiling needed to clear private-500."""
    return MU_P + (TARGET / (1.0 - g) - MU_P) / rho


def public_for_private_500_haircut(rho_dep: float = RHO_DEPLOYED_318) -> float:
    """Invert haircut(P) = 500 -> public ceiling needed for private-500 under the #318 haircut."""
    return TARGET / rho_dep


# ===========================================================================
# Section 2 -- coverage -> E[T] -> public-strict -> private-strict joint map
# ===========================================================================

def et_factor(dc: float, s: float) -> float:
    """Multiplicative public-strict lift from a coverage retrain of +dc (E[T] channel, #377 secant)."""
    return 1.0 + s * dc / E_T_REALIZED


def gap_at_coverage(dc: float) -> float:
    """Gap after a +dc coverage retrain (#377 co-benefit), FLOORED at the #379 irreducible ctxlen."""
    return max(GAP_IRREDUCIBLE_FLOOR_FRAC, GAP_MEASURED - KAPPA_GAP_377 * dc)


def private_at_coverage(base: float, dc: float, s: float, rho: float,
                        model: str = "regression") -> float:
    """PRIVATE-strict TPS after a +dc coverage retrain on a public-strict `base`.

    JOINT of both demand channels: (1) E[T] lift raises the public-strict base multiplicatively;
    (2) gap-closure (rho->1, floored at #379 irreducible) lifts priv/pub. At dc=0 reduces to the
    base transfer.
    """
    pub_c = base * et_factor(dc, s)
    g_c = gap_at_coverage(dc)
    if model == "regression":
        return project(pub_c, rho, g_c)
    return (1.0 - g_c) * pub_c                     # simple-haircut joint


def required_dcov_for_private_500(base: float, s: float, rho: float,
                                  model: str = "regression") -> dict:
    """Delta-coverage (E[T] channel, #377 secant) to reach private-500 from `base`. Gap held fixed
    (conservative: the gap co-benefit only makes the true dcov smaller). Reports the directly-grounded
    delta_et and the #379-slope cross-check."""
    if model == "regression":
        pstar = public_for_private_500_regression(rho)
    else:
        pstar = public_for_private_500_haircut()
    delta_et = E_T_REALIZED * (pstar / base - 1.0)        # E[T] lift needed (K_cal-linear, base-relative)
    dcov = delta_et / s                                    # #377 program-secant inversion
    dcov_379 = delta_et / A1_MARGINAL_289                  # #379 a1-only-slope cross-check (more conservative)
    return {
        "pstar_public_for_private_500": pstar,
        "residual_public_lift_tps": pstar - base,
        "delta_et_needed": delta_et,
        "required_coverage_delta": dcov,
        "required_coverage_delta_379_a1slope": dcov_379,
        "within_336_budget": dcov <= COV_BUDGET_336,
        "frac_of_336_budget": dcov / COV_BUDGET_336,
    }


def max_private_at_full_budget(base: float, s: float, rho: float) -> dict:
    """Max PRIVATE-strict reachable spending the FULL #336 +0.031 coverage budget (joint channels)."""
    dc = COV_BUDGET_336
    reg = private_at_coverage(base, dc, s, rho, "regression")
    hair = private_at_coverage(base, dc, s, rho, "haircut")
    pub_c = base * et_factor(dc, s)
    return {
        "public_strict_at_full_budget": pub_c,
        "private_strict_at_max_coverage_regression": reg,
        "private_strict_at_max_coverage_haircut": hair,
        "clears_500_regression": reg >= TARGET,
        "clears_500_haircut": hair >= TARGET,
        "gap_at_full_budget": gap_at_coverage(dc),
    }


# ===========================================================================
# Section 3 -- per-base re-pricing (deliverables 1-4)
# ===========================================================================

def price_base(name: str, base: float) -> dict:
    """Full honest re-pricing for one public-strict base, central + conservative corners."""
    # (2) transfer to private at CURRENT coverage 0.8903 (dc=0).
    priv_reg_central = project(base, RHO_PRIV)
    priv_reg_conservative = project(base, RHO_LB)
    priv_haircut = haircut(base)
    # (3) honest residual-to-private-500.
    resid_reg_central = TARGET - priv_reg_central
    resid_reg_conservative = TARGET - priv_reg_conservative
    resid_haircut = TARGET - priv_haircut
    # (4) required Delta-coverage (E[T] channel) + max-private at full budget.
    req_reg_central = required_dcov_for_private_500(base, S_CENTRAL, RHO_PRIV, "regression")
    req_reg_worst = required_dcov_for_private_500(base, S_WORST, RHO_PRIV, "regression")
    req_haircut = required_dcov_for_private_500(base, S_CENTRAL, RHO_DEPLOYED_318, "haircut")
    maxp_central = max_private_at_full_budget(base, S_CENTRAL, RHO_PRIV)
    maxp_worst = max_private_at_full_budget(base, S_WORST, RHO_PRIV)
    return {
        "base_public_strict": base,
        # (2) private at current coverage
        "private_strict_at_base_regression_central": priv_reg_central,
        "private_strict_at_base_regression_conservative": priv_reg_conservative,
        "private_strict_at_base_haircut_318": priv_haircut,
        # (3) residual
        "residual_to_private_500_regression_central": resid_reg_central,
        "residual_to_private_500_regression_conservative": resid_reg_conservative,
        "residual_to_private_500_haircut_318": resid_haircut,
        # (4) required dcov
        "required_coverage_delta_central": req_reg_central["required_coverage_delta"],
        "required_coverage_delta_worst": req_reg_worst["required_coverage_delta"],
        "required_coverage_delta_haircut": req_haircut["required_coverage_delta"],
        "required_coverage_delta_379_a1slope": req_reg_central["required_coverage_delta_379_a1slope"],
        "required_dcov_within_336_central": req_reg_central["within_336_budget"],
        "required_dcov_frac_of_336_central": req_reg_central["frac_of_336_budget"],
        "required_dcov_exceeds_336": req_reg_central["required_coverage_delta"] > COV_BUDGET_336,
        "delta_et_needed_central": req_reg_central["delta_et_needed"],
        "pstar_public_for_private_500": req_reg_central["pstar_public_for_private_500"],
        # max private at full #336 budget (the demand-alone ceiling)
        "max_private_at_full_budget_central": maxp_central["private_strict_at_max_coverage_regression"],
        "max_private_at_full_budget_central_haircut": maxp_central["private_strict_at_max_coverage_haircut"],
        "max_private_at_full_budget_worst": maxp_worst["private_strict_at_max_coverage_regression"],
        "public_strict_at_full_budget_central": maxp_central["public_strict_at_full_budget"],
        "demand_reaches_500_at_full_budget_central": maxp_central["clears_500_regression"],
        "demand_reaches_500_at_full_budget_worst": maxp_worst["clears_500_regression"],
        # pure gap-closure cap (rho -> 1): private <= public base (no E[T] credit)
        "rho_to_1_cap_no_et_credit": base,
        "rho_to_1_cap_clears_500": base >= TARGET,
    }


# ===========================================================================
# Section 4 -- supply lift required first (deliverable 5)
# ===========================================================================

def supply_lift_required(base: float) -> dict:
    """How much the public-strict base must RISE (supply-side lm_head-BI lever) before the demand
    route can finish to private-500 within the #336 budget.

    Two corners: (A) E[T]-channel only (no gap co-benefit) -- the conservative/robust requirement;
    (B) joint (E[T] + gap co-benefit) -- the central requirement. Solve for the base B* such that the
    demand route reaches private-500 spending exactly the full #336 budget."""
    # (A) E[T]-channel only: need public lift to Pstar at the deployed gap; demand supplies dcov<=budget.
    #     B* such that required_dcov(B*) == budget  ->  delta_et(B*) == budget*S_CENTRAL.
    pstar = public_for_private_500_regression(RHO_PRIV)
    delta_et_budget = COV_BUDGET_336 * S_CENTRAL
    bstar_et_only = pstar / (1.0 + delta_et_budget / E_T_REALIZED)
    lift_et_only = max(0.0, bstar_et_only - base)
    # (B) joint: B* such that private_at_coverage(B*, budget, S_CENTRAL, rho, regression) == 500.
    dc = COV_BUDGET_336
    g_c = gap_at_coverage(dc)
    f = et_factor(dc, S_CENTRAL)
    # project(B**f; RHO_PRIV, g_c) == 500  ->  solve linear in B*.
    #   (1-g_c)*(MU_P + RHO_PRIV*(B**f - MU_P)) = 500
    inner = TARGET / (1.0 - g_c)                       # MU_P + RHO_PRIV*(B**f - MU_P)
    bstar_joint = (MU_P + (inner - MU_P) / RHO_PRIV) / f
    lift_joint = max(0.0, bstar_joint - base)
    return {
        "supply_lift_required_et_only_tps": lift_et_only,
        "supply_lift_required_joint_tps": lift_joint,
        "bstar_public_strict_et_only": bstar_et_only,
        "bstar_public_strict_joint": bstar_joint,
        "attn_rebuild_roi_tps": ATTN_REBUILD_ROI_TPS,
        "attn_rebuild_alone_sufficient": ATTN_REBUILD_ROI_TPS >= lift_joint,
    }


# ===========================================================================
# Section 5 -- a2 / accept-headroom pre-check (deliverable 7, secondary)
# ===========================================================================

def accept_headroom_check(required_dcov_floor: float, required_dcov_attn: float) -> dict:
    """Is there trainable headroom in the accept vector to deliver the required Delta-coverage?

    The #336 budget IS the trainable coverage headroom (identity_bar - prior). a_1=0.7293 sits below
    the #308 published 0.80 envelope (some a_1 headroom), but the deeper conditionals 0.82-0.85 are
    near their per-position ceiling and the realized coverage->accept transfer is only kappa~0.67.
    A required dcov ABOVE the #336 budget cannot be delivered by the trainable head."""
    a1_headroom = EAGLE3_PUBLISHED_A1 - A_D_MEASURED[0]            # +0.0707 a_1 room vs published
    deep_near_ceiling = all(a >= 0.80 for a in A_D_MEASURED[3:])   # depths 4..7 already >= published a_1
    floor_ok = required_dcov_floor <= COV_BUDGET_336
    attn_ok = required_dcov_attn <= COV_BUDGET_336

    def verdict(ok: bool, req: float) -> str:
        if ok:
            return "sufficient"
        # marginal if within 1.25x of budget (a stretch pilot might reach), else False.
        return "marginal" if req <= 1.25 * COV_BUDGET_336 else "False"

    return {
        "trainable_coverage_headroom": COV_BUDGET_336,             # == identity_bar - prior
        "a1_headroom_vs_published": a1_headroom,
        "deep_positions_near_ceiling": deep_near_ceiling,
        "kappa_realized_transfer": KAPPA_REALIZED_TRANSFER_377,
        "accept_headroom_sufficient_for_required_delta_floor": verdict(floor_ok, required_dcov_floor),
        "accept_headroom_sufficient_for_required_delta_attn_rebuild": verdict(attn_ok, required_dcov_attn),
        "note": ("required dcov on the honest bases ({:.4f} floor / {:.4f} attn-rebuild) EXCEEDS the "
                 "#336 trainable headroom 0.0310; the deeper conditionals are near-ceiling and the "
                 "realized transfer is kappa~0.67 -> the head cannot deliver it.").format(
                     required_dcov_floor, required_dcov_attn),
    }


# ===========================================================================
# Section 6 -- #377/#379 reconciliation (deliverable 6)
# ===========================================================================

def reconcile_377() -> dict:
    """Round-trip: under the OLD 518.92 base + this harness, reproduce #373's +5.44 / #377's c>=0.9010.

    Proves the verdict flip is driven ENTIRELY by the base move 518.92 -> <=480.8, not a harness change."""
    priv = project(P_HEADLINE, RHO_PRIV)                          # 494.56
    resid = TARGET - priv                                         # 5.44
    req_central = required_dcov_for_private_500(P_HEADLINE, S_CENTRAL, RHO_PRIV, "regression")
    req_worst = required_dcov_for_private_500(P_HEADLINE, S_WORST, RHO_PRIV, "regression")
    repro_resid = abs(resid - RESIDUAL_544_377) < 0.05
    repro_delta_et = abs(req_central["delta_et_needed"] - DELTA_ET_CENTRAL_377) < 5e-4
    repro_dcov_central = abs(req_central["required_coverage_delta"] - DCOV_CENTRAL_377) < 5e-4
    repro_dcov_robust = abs(req_worst["required_coverage_delta"] - DCOV_ROBUST_377) < 5e-4
    repro_target = abs((COV_PRIOR + req_worst["required_coverage_delta"]) - REC_TARGET_377) < 5e-4
    return {
        "old_base_518_92": P_HEADLINE,
        "private_at_518_92_regression_central": priv,
        "residual_to_private_500_at_518_92": resid,
        "delta_et_at_518_92": req_central["delta_et_needed"],
        "dcov_central_at_518_92": req_central["required_coverage_delta"],
        "dcov_robust_at_518_92": req_worst["required_coverage_delta"],
        "recommended_target_at_518_92": COV_PRIOR + req_worst["required_coverage_delta"],
        "reproduces_377_residual_544": repro_resid,
        "reproduces_377_delta_et": repro_delta_et,
        "reproduces_377_dcov_central": repro_dcov_central,
        "reproduces_377_dcov_robust_0p0107": repro_dcov_robust,
        "reproduces_377_target_0p9010": repro_target,
        "reproduces_377_under_revival": bool(
            repro_resid and repro_dcov_central and repro_dcov_robust and repro_target),
    }


# ===========================================================================
# Section 7 -- verdict assembly (deliverable 5 fields)
# ===========================================================================

def build_verdict(bases: dict, recon: dict) -> dict:
    floor = bases["deployable_floor"]
    attn = bases["plus_attn_rebuild"]
    sl_floor = supply_lift_required(BASE_DEPLOYABLE_FLOOR)
    sl_attn = supply_lift_required(BASE_PLUS_ATTN_REBUILD)

    demand_reaches_floor = bool(floor["demand_reaches_500_at_full_budget_central"])
    demand_reaches_attn = bool(attn["demand_reaches_500_at_full_budget_central"])
    demand_reaches_attn_worst = bool(attn["demand_reaches_500_at_full_budget_worst"])
    # max private-strict reachable by demand ALONE on the floor base (full #336 budget, central joint).
    max_priv_floor = floor["max_private_at_full_budget_central"]
    # pilot on critical path? central-only closes it on SOME honest base => pilot would matter.
    central_closes_some_honest = demand_reaches_floor or demand_reaches_attn
    pilot_on_critical_path = bool(central_closes_some_honest and not demand_reaches_attn_worst)
    # supply lift to lead with: the floor base, joint (central) corner; robust = ET-only.
    supply_lift_first = sl_floor["supply_lift_required_joint_tps"]

    return {
        "demand_route_reaches_500_on_deployable_floor": demand_reaches_floor,
        "demand_route_reaches_500_with_attn_rebuild": demand_reaches_attn,
        "demand_route_reaches_500_with_attn_rebuild_worst_slope": demand_reaches_attn_worst,
        "max_private_strict_demand_only_floor": max_priv_floor,
        "max_private_strict_demand_only_floor_worst": floor["max_private_at_full_budget_worst"],
        "rho_to_1_cap_floor_no_et_credit": floor["rho_to_1_cap_no_et_credit"],
        "required_coverage_delta_floor": floor["required_coverage_delta_central"],
        "required_coverage_delta_attn_rebuild": attn["required_coverage_delta_central"],
        "required_dcov_floor_exceeds_336": floor["required_dcov_exceeds_336"],
        "required_dcov_attn_exceeds_336": attn["required_dcov_exceeds_336"],
        "supply_lift_required_first_tps": supply_lift_first,
        "supply_lift_required_first_tps_floor_et_only": sl_floor["supply_lift_required_et_only_tps"],
        "supply_lift_required_first_tps_attn_rebuild": sl_attn["supply_lift_required_joint_tps"],
        "attn_rebuild_alone_closes_supply_gap": sl_floor["attn_rebuild_alone_sufficient"],
        "pilot_on_critical_path": pilot_on_critical_path,
        "supply_lift_detail_floor": sl_floor,
        "supply_lift_detail_attn_rebuild": sl_attn,
        # the headline band for fern #357
        "demand_alone_standalone_go": bool(demand_reaches_floor or
                                           (demand_reaches_attn and demand_reaches_attn_worst)),
        "verdict_band": (
            "RED_demand_alone_insufficient_on_honest_base" if not (demand_reaches_floor or demand_reaches_attn)
            else "AMBER_demand_marginal_on_attn_rebuild_only"),
        "verdict_summary": (
            "Demand-alone does NOT reach PRIVATE-strict-500 on the honest deployable-strict base. On "
            "the floor base 469.68 the required coverage delta is +{:.4f} ({:.1f}x the #336 +0.031 "
            "budget); spending the FULL #336 budget the demand route tops out at private {:.1f} "
            "(central) / {:.1f} (worst) < 500. Even crediting the ~11-TPS attention rebuild (-> 480.8) "
            "the required delta is +{:.4f} ({:.1f}x budget) and the full-budget ceiling is {:.1f} < 500. "
            "The pure gap-closure cap (rho->1, no E[T] credit) is the public base itself ({:.1f}) < 500. "
            "Verdict FLIP vs #377 (+5.44 TPS, in budget) is driven ENTIRELY by the base move "
            "518.92 -> <=480.8 (round-trip reproduces #377 under 518.92 = {}). The demand route is "
            "NECESSARY-BUT-NOT-SUFFICIENT: a supply-side lm_head-BI lift of ~{:.1f} TPS (floor) / ~{:.1f} "
            "(attn-rebuild) must raise the public-strict base FIRST. Hand to fern #357: re-center on the "
            "supply-side lm_head-BI lever.").format(
                floor["required_coverage_delta_central"], floor["required_dcov_frac_of_336_central"],
                max_priv_floor, floor["max_private_at_full_budget_worst"],
                attn["required_coverage_delta_central"], attn["required_dcov_frac_of_336_central"],
                attn["max_private_at_full_budget_central"], floor["rho_to_1_cap_no_et_credit"],
                recon["reproduces_377_under_revival"],
                sl_floor["supply_lift_required_joint_tps"], sl_attn["supply_lift_required_joint_tps"]),
    }


# ===========================================================================
# Section 8 -- self-tests
# ===========================================================================

def run_self_tests(bases: dict, verdict: dict, recon: dict, headroom: dict) -> dict:
    c = {}
    floor, attn, revival = bases["deployable_floor"], bases["plus_attn_rebuild"], bases["eta_revival"]
    # a) transfer identities: regression rho=1 == naive proportional; anchor recovers mu_V.
    c["a_rho1_is_naive_prop"] = abs(project(P_HEADLINE, 1.0) - (1.0 - GAP_MEASURED) * P_HEADLINE) < 1e-9
    c["a_anchor_recovers_mu_v"] = abs(project(MU_P, RHO_PRIV) - MU_V) < 1e-9
    c["a_haircut_anchor_recovers_mu_v"] = abs(haircut(MU_P) - MU_V) < 1e-6
    # b) RECONCILIATION: under 518.92 reproduce #377 +5.44 / dcov / c>=0.9010 (the round-trip).
    c["b_reproduces_544"] = recon["reproduces_377_residual_544"]
    c["b_reproduces_delta_et"] = recon["reproduces_377_delta_et"]
    c["b_reproduces_dcov_central"] = recon["reproduces_377_dcov_central"]
    c["b_reproduces_dcov_robust_0p0107"] = recon["reproduces_377_dcov_robust_0p0107"]
    c["b_reproduces_target_0p9010"] = recon["reproduces_377_target_0p9010"]
    c["b_reproduces_377_under_revival"] = recon["reproduces_377_under_revival"]
    # c) honest bases are sub-500 and ORDER correctly (offtheshelf < floor < attn-rebuild < revival).
    c["c_floor_below_500"] = BASE_DEPLOYABLE_FLOOR < TARGET
    c["c_attn_rebuild_below_500"] = BASE_PLUS_ATTN_REBUILD < TARGET
    c["c_base_ordering"] = (BASE_DEPLOYABLE_OFFTHESHELF < BASE_DEPLOYABLE_FLOOR
                            < BASE_PLUS_ATTN_REBUILD < P_HEADLINE)
    c["c_attn_rebuild_roi_about_11"] = 9.0 < ATTN_REBUILD_ROI_TPS < 13.0
    # d) private at base < base (transfer is a haircut) and residual positive on honest bases.
    c["d_priv_below_base_floor"] = floor["private_strict_at_base_regression_central"] < BASE_DEPLOYABLE_FLOOR
    c["d_residual_positive_floor"] = floor["residual_to_private_500_regression_central"] > 0
    c["d_residual_positive_attn"] = attn["residual_to_private_500_regression_central"] > 0
    c["d_transfer_models_agree_floor"] = abs(
        floor["private_strict_at_base_regression_central"] - floor["private_strict_at_base_haircut_318"]) < 3.0
    # e) THE FLIP: required dcov on honest bases EXCEEDS the #336 budget (#377's did not).
    c["e_floor_dcov_exceeds_budget"] = floor["required_coverage_delta_central"] > COV_BUDGET_336
    c["e_attn_dcov_exceeds_budget"] = attn["required_coverage_delta_central"] > COV_BUDGET_336
    c["e_revival_dcov_within_budget"] = revival["required_coverage_delta_central"] <= COV_BUDGET_336
    c["e_dcov_monotone_in_base"] = (floor["required_coverage_delta_central"]
                                    > attn["required_coverage_delta_central"]
                                    > revival["required_coverage_delta_central"])
    # f) demand-alone does NOT clear 500 at full budget on the honest bases (the verdict).
    c["f_floor_no_clear_full_budget"] = not floor["demand_reaches_500_at_full_budget_central"]
    c["f_attn_worst_no_clear"] = not attn["demand_reaches_500_at_full_budget_worst"]
    c["f_max_private_floor_below_500"] = floor["max_private_at_full_budget_central"] < TARGET
    c["f_rho1_cap_floor_below_500"] = not floor["rho_to_1_cap_clears_500"]
    # g) supply lift positive & finite; attention-rebuild alone does not close the floor supply gap.
    c["g_supply_lift_floor_positive"] = verdict["supply_lift_required_first_tps"] > 0
    c["g_supply_lift_attn_le_floor"] = (verdict["supply_lift_required_first_tps_attn_rebuild"]
                                        <= verdict["supply_lift_required_first_tps"] + 1e-6)
    c["g_attn_rebuild_alone_insufficient"] = not verdict["attn_rebuild_alone_closes_supply_gap"]
    # h) #379 gap-decomp imports consistent: buckets sum ~100%, irreducible floor < total gap.
    c["h_buckets_sum_100"] = abs(GAP_BUCKET_ACCEPTANCE_PCT + GAP_BUCKET_CTXLEN_PCT - 100.0) < 1e-6
    c["h_irreducible_below_gap"] = GAP_IRREDUCIBLE_FLOOR_FRAC < GAP_MEASURED
    c["h_rho_max_below_1"] = RHO_MAX_DEMAND < 1.0
    # banked #379 slope (489.76448) == a1_marginal*K_cal to 6 sig figs; tolerate float provenance noise
    # (#379 a1_only_T2 vs #377 per_position_dEll_da[0] differ in the last ~2 digits).
    c["h_slope_379_is_a1_times_kcal"] = abs(SLOPE_379_TPS_PER_COV - A1_MARGINAL_289 * K_CAL) / SLOPE_379_TPS_PER_COV < 1e-5
    # i) a2/accept-headroom: required dcov on honest bases is NOT deliverable (False).
    c["i_floor_headroom_insufficient"] = (
        headroom["accept_headroom_sufficient_for_required_delta_floor"] != "sufficient")
    c["i_deep_positions_near_ceiling"] = headroom["deep_positions_near_ceiling"]
    # j) program secants below iid edge and worst<central (the #377 non-iid ordering).
    c["j_secants_ordered"] = S_WORST < S_CENTRAL
    c["j_kappa_realized_below_1"] = KAPPA_REALIZED_TRANSFER_377 < 1.0
    # k) numeric hygiene.
    flat = [floor["required_coverage_delta_central"], attn["required_coverage_delta_central"],
            floor["max_private_at_full_budget_central"], verdict["supply_lift_required_first_tps"],
            recon["private_at_518_92_regression_central"]]
    c["k_no_nan"] = all(v == v for v in flat)
    passes = all(c.values())
    return {"conditions": c, "n_checks": len(c), "passes": passes}


# ===========================================================================
# Section 9 -- report assembly + W&B + CLI
# ===========================================================================

def build_report() -> dict:
    bases = {
        "deployable_floor": price_base("deployable_floor", BASE_DEPLOYABLE_FLOOR),
        "deployable_offtheshelf": price_base("deployable_offtheshelf", BASE_DEPLOYABLE_OFFTHESHELF),
        "plus_attn_rebuild": price_base("plus_attn_rebuild", BASE_PLUS_ATTN_REBUILD),
        "eta_revival": price_base("eta_revival", P_HEADLINE),
    }
    recon = reconcile_377()
    headroom = accept_headroom_check(
        bases["deployable_floor"]["required_coverage_delta_central"],
        bases["plus_attn_rebuild"]["required_coverage_delta_central"])
    verdict = build_verdict(bases, recon)
    selftest = run_self_tests(bases, verdict, recon, headroom)
    return {
        "pr": 383, "agent": "denken", "kind": "demand-residual-honest-base",
        "analysis_only": True, "no_launch": True, "no_hf_job": True, "no_submission": True,
        "no_served_file_change": True, "gpu_used": False, "official_tps_expected": 0,
        "inputs": {
            "mu_p_public": MU_P, "mu_v_private": MU_V, "gap_measured": GAP_MEASURED, "k_cal": K_CAL,
            "e_t_realized": E_T_REALIZED, "rho_deployed_318": RHO_DEPLOYED_318,
            "rho_priv": RHO_PRIV, "rho_lb": RHO_LB,
            "base_deployable_floor": BASE_DEPLOYABLE_FLOOR,
            "base_deployable_offtheshelf": BASE_DEPLOYABLE_OFFTHESHELF,
            "base_plus_attn_rebuild": BASE_PLUS_ATTN_REBUILD, "eta_attn_378": ETA_ATTN_378,
            "attn_rebuild_roi_tps": ATTN_REBUILD_ROI_TPS, "base_eta_revival_377ref": P_HEADLINE,
            "cov_prior": COV_PRIOR, "identity_bar": IDENTITY_BAR, "cov_budget_336": COV_BUDGET_336,
            "cstar_central": CSTAR_CENTRAL, "cstar_worst": CSTAR_WORST,
            "s_central": S_CENTRAL, "s_worst": S_WORST, "a1_marginal_289": A1_MARGINAL_289,
            "slope_379_tps_per_cov": SLOPE_379_TPS_PER_COV,
            "gap_bucket_acceptance_pct": GAP_BUCKET_ACCEPTANCE_PCT,
            "gap_irreducible_floor_pct": GAP_IRREDUCIBLE_FLOOR_PCT, "rho_max_demand": RHO_MAX_DEMAND,
            "kappa_gap_377": KAPPA_GAP_377, "kappa_realized_transfer_377": KAPPA_REALIZED_TRANSFER_377,
            "residual_544_377": RESIDUAL_544_377, "dcov_central_377": DCOV_CENTRAL_377,
            "dcov_robust_377": DCOV_ROBUST_377, "rec_target_377": REC_TARGET_377,
            "p_deliver_central_380": P_DELIVER_CENTRAL_380, "p_deliver_robust_380": P_DELIVER_ROBUST_380,
            "pilot_gpu_hr_352": PILOT_GPU_HR_352,
            "source_378_run": "gghmgtk9", "source_377_run": "030uc5mk", "source_379_run": "5kpb73tb",
            "source_380_run": "00oijpwg", "source_373_run": "oqs8lddd",
        },
        "bases": bases, "reconciliation_377": recon, "accept_headroom": headroom, "verdict": verdict,
        # ----- card-required headline scalars (SENPAI-RESULT load-bearing) -----
        "demand_route_reaches_500_on_deployable_floor": verdict["demand_route_reaches_500_on_deployable_floor"],
        "demand_route_reaches_500_with_attn_rebuild": verdict["demand_route_reaches_500_with_attn_rebuild"],
        "max_private_strict_demand_only_floor": verdict["max_private_strict_demand_only_floor"],
        "required_coverage_delta_floor": verdict["required_coverage_delta_floor"],
        "required_coverage_delta_attn_rebuild": verdict["required_coverage_delta_attn_rebuild"],
        "supply_lift_required_first_tps": verdict["supply_lift_required_first_tps"],
        "pilot_on_critical_path": verdict["pilot_on_critical_path"],
        "reproduces_377_under_revival": recon["reproduces_377_under_revival"],
        "accept_headroom_sufficient_for_required_delta":
            headroom["accept_headroom_sufficient_for_required_delta_floor"],
        "verdict_band": verdict["verdict_band"],
        "primary_metric_max_private_strict_demand_only_floor": verdict["max_private_strict_demand_only_floor"],
        "self_test": selftest,
        "demand_residual_honest_base_self_test_passes": selftest["passes"],
    }


def log_to_wandb(report: dict, group: str, name: str) -> str:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"W&B import failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_unavailable"
    try:
        run = wandb.init(group=group, name=name, config=report["inputs"])
        v, recon = report["verdict"], report["reconciliation_377"]
        floor, attn = report["bases"]["deployable_floor"], report["bases"]["plus_attn_rebuild"]
        revival = report["bases"]["eta_revival"]
        wandb.log({
            "summary/demand_route_reaches_500_on_deployable_floor":
                float(v["demand_route_reaches_500_on_deployable_floor"]),
            "summary/demand_route_reaches_500_with_attn_rebuild":
                float(v["demand_route_reaches_500_with_attn_rebuild"]),
            "summary/demand_route_reaches_500_with_attn_rebuild_worst":
                float(v["demand_route_reaches_500_with_attn_rebuild_worst_slope"]),
            "summary/max_private_strict_demand_only_floor": v["max_private_strict_demand_only_floor"],
            "summary/max_private_strict_demand_only_floor_worst": v["max_private_strict_demand_only_floor_worst"],
            "summary/rho_to_1_cap_floor_no_et_credit": v["rho_to_1_cap_floor_no_et_credit"],
            "summary/required_coverage_delta_floor": v["required_coverage_delta_floor"],
            "summary/required_coverage_delta_attn_rebuild": v["required_coverage_delta_attn_rebuild"],
            "summary/required_dcov_floor_frac_of_336": floor["required_dcov_frac_of_336_central"],
            "summary/required_dcov_floor_379_a1slope": floor["required_coverage_delta_379_a1slope"],
            "summary/supply_lift_required_first_tps": v["supply_lift_required_first_tps"],
            "summary/supply_lift_required_first_tps_floor_et_only": v["supply_lift_required_first_tps_floor_et_only"],
            "summary/supply_lift_required_first_tps_attn_rebuild": v["supply_lift_required_first_tps_attn_rebuild"],
            "summary/attn_rebuild_alone_closes_supply_gap": float(v["attn_rebuild_alone_closes_supply_gap"]),
            "summary/pilot_on_critical_path": float(v["pilot_on_critical_path"]),
            "summary/demand_alone_standalone_go": float(v["demand_alone_standalone_go"]),
            "summary/private_strict_at_floor_central": floor["private_strict_at_base_regression_central"],
            "summary/private_strict_at_floor_haircut": floor["private_strict_at_base_haircut_318"],
            "summary/residual_to_private_500_floor": floor["residual_to_private_500_regression_central"],
            "summary/private_strict_at_attn_rebuild": attn["private_strict_at_base_regression_central"],
            "summary/max_private_at_full_budget_attn": attn["max_private_at_full_budget_central"],
            "summary/private_at_518_92_regression": recon["private_at_518_92_regression_central"],
            "summary/residual_to_private_500_at_518_92": recon["residual_to_private_500_at_518_92"],
            "summary/dcov_robust_at_518_92": recon["dcov_robust_at_518_92"],
            "summary/reproduces_377_under_revival": float(recon["reproduces_377_under_revival"]),
            "summary/revival_dcov_within_budget": float(revival["required_dcov_within_336_central"]),
            "summary/self_test_passes": float(report["self_test"]["passes"]),
        })
        for cond, val in report["self_test"]["conditions"].items():
            wandb.log({f"test/{cond}": float(bool(val))})
        run_id = run.id
        wandb.finish()
        return run_id
    except Exception as exc:  # noqa: BLE001
        print(f"W&B logging failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_failed"


def _fmt_base(tag: str, b: dict) -> str:
    return (f"  {tag:<22}: base={b['base_public_strict']:.2f}  "
            f"priv@cov0={b['private_strict_at_base_regression_central']:.2f}(reg)/"
            f"{b['private_strict_at_base_haircut_318']:.2f}(haircut)  "
            f"resid={b['residual_to_private_500_regression_central']:+.2f}  "
            f"req_dcov={b['required_coverage_delta_central']:.4f}"
            f"({b['required_dcov_frac_of_336_central']:.2f}x336)  "
            f"maxpriv@budget={b['max_private_at_full_budget_central']:.2f}"
            f"{'>=500' if b['demand_reaches_500_at_full_budget_central'] else '<500'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-price the demand residual on the honest deployable-strict base (PR #383).")
    ap.add_argument("--honest-base", action="store_true", help="(default) re-price on the #378 honest bases")
    ap.add_argument("--reconcile-377", action="store_true", help="(default) round-trip vs #377 under 518.92")
    ap.add_argument("--reanalyze", action="store_true", help="(default) full re-pricing + verdict")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--wandb_group", type=str, default="strict-bi-verify-gemm")
    ap.add_argument("--wandb_name", type=str, default="denken/demand-residual-honest-base")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", type=str,
                    default="research/validity/demand_residual_honest_base/"
                            "demand_residual_honest_base_results.json")
    args = ap.parse_args()

    report = build_report()
    bases, v, recon = report["bases"], report["verdict"], report["reconciliation_377"]
    hr = report["accept_headroom"]

    print("\n=== Re-price the demand residual on the HONEST deployable-strict base (PR #383) ===")
    print(f"public<->private anchor : {MU_P} -> {MU_V}  gap={GAP_MEASURED*100:.3f}%  K_cal={K_CAL:.3f}")
    print(f"transfer slopes         : program-secant central={S_CENTRAL:.2f} worst={S_WORST:.2f} E[T]/cov; "
          f"#379 a1-slope={A1_MARGINAL_289:.3f} (={SLOPE_379_TPS_PER_COV:.1f} TPS/cov)")
    print(f"gap decomp (#379)       : {GAP_BUCKET_ACCEPTANCE_PCT:.1f}% acceptance / {GAP_BUCKET_CTXLEN_PCT:.1f}% "
          f"ctxlen; irreducible floor {GAP_IRREDUCIBLE_FLOOR_PCT:.3f}% -> rho_max={RHO_MAX_DEMAND:.4f}")
    print("\nPer-base re-pricing (private at current coverage 0.8903; req_dcov to private-500; max-priv at full #336 budget):")
    print(_fmt_base("deployable_floor", bases["deployable_floor"]))
    print(_fmt_base("plus_attn_rebuild", bases["plus_attn_rebuild"]))
    print(_fmt_base("deployable_offtheshelf", bases["deployable_offtheshelf"]))
    print(_fmt_base("eta_revival(#377ref)", bases["eta_revival"]))
    print(f"\nRECONCILE #377 under 518.92 : priv={recon['private_at_518_92_regression_central']:.2f}  "
          f"resid={recon['residual_to_private_500_at_518_92']:+.2f}  "
          f"dcov_central={recon['dcov_central_at_518_92']:.5f}  dcov_robust={recon['dcov_robust_at_518_92']:.5f}  "
          f"target={recon['recommended_target_at_518_92']:.4f}")
    print(f"  reproduces_377_under_revival = {recon['reproduces_377_under_revival']}  "
          f"(544={recon['reproduces_377_residual_544']}, dcov_robust_0.0107={recon['reproduces_377_dcov_robust_0p0107']}, "
          f"target_0.9010={recon['reproduces_377_target_0p9010']})")
    print(f"\naccept-headroom (a2)    : floor={hr['accept_headroom_sufficient_for_required_delta_floor']}  "
          f"attn-rebuild={hr['accept_headroom_sufficient_for_required_delta_attn_rebuild']}  "
          f"(a1 room vs published {hr['a1_headroom_vs_published']:+.3f}, deep near-ceiling={hr['deep_positions_near_ceiling']})")
    print("\n--- VERDICT (load-bearing for fern #357) ---")
    print(f"  demand_route_reaches_500_on_deployable_floor : {v['demand_route_reaches_500_on_deployable_floor']}")
    print(f"  demand_route_reaches_500_with_attn_rebuild   : {v['demand_route_reaches_500_with_attn_rebuild']} "
          f"(worst-slope {v['demand_route_reaches_500_with_attn_rebuild_worst_slope']})")
    print(f"  max_private_strict_demand_only_floor         : {v['max_private_strict_demand_only_floor']:.2f} "
          f"(worst {v['max_private_strict_demand_only_floor_worst']:.2f}; rho->1 no-E[T] cap {v['rho_to_1_cap_floor_no_et_credit']:.2f}) < 500")
    print(f"  required_coverage_delta (floor / attn)       : +{v['required_coverage_delta_floor']:.4f} / "
          f"+{v['required_coverage_delta_attn_rebuild']:.4f}  (budget +{COV_BUDGET_336:.4f}; both exceed={v['required_dcov_floor_exceeds_336'] and v['required_dcov_attn_exceeds_336']})")
    print(f"  supply_lift_required_first_tps               : {v['supply_lift_required_first_tps']:.2f} (floor, joint) / "
          f"{v['supply_lift_required_first_tps_floor_et_only']:.2f} (floor, E[T]-only) / "
          f"{v['supply_lift_required_first_tps_attn_rebuild']:.2f} (attn-rebuild)")
    print(f"  attn_rebuild_alone_closes_supply_gap         : {v['attn_rebuild_alone_closes_supply_gap']}")
    print(f"  pilot_on_critical_path                       : {v['pilot_on_critical_path']}")
    print(f"  verdict_band                                 : {v['verdict_band']}")
    print(f"\n{v['verdict_summary']}")
    print(f"\nself-test: {report['self_test']['n_checks']} checks, passes={report['self_test']['passes']}")

    if args.self_test:
        return 0 if report["self_test"]["passes"] else 1
    if not args.no_wandb:
        report["wandb_run_id"] = log_to_wandb(report, args.wandb_group, args.wandb_name)
    else:
        report["wandb_run_id"] = None
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}  (W&B run {report.get('wandb_run_id')})")
    return 0 if report["self_test"]["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
