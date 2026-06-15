#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Honest cb3 supply lift through the MTP K=7 served loop (PR #392, denken).

THE QUESTION (the node fern #357's composite needs):
  lawine #388 (g5lfdpgw) measured the realized cb3-vs-int4-Marlin body-GEMM speedup
  (M=1 realistic 1.1234x) and translated it to a HEADLINE realized_strict_base_lift_tps =
  +38.34 via a single-forward `lift_factor` applied to the COMPLEMENT body fraction
  f_comp = 1 - f_attn - f_lmhead = 0.8825. That complement LUMPS the #378 `draft` fraction
  (0.1201 -- the spec drafter forward) into the cb3-shrinkable body. But the served stack is
  MTP K=7 spec-decode (PR #52): each step = 1 DRAFTER forward (a small model, NOT cb3-quantized
  -> un-shrunk) + 1 VERIFY forward (M=8, the cb3-shrinkable target body). A faster body GEMM
  speeds the verify forward but NOT the draft forward. So the honest end-to-end served-TPS lift
  must (a) credit cb3 ONLY to the verify-body fraction f_verify_body = 0.7624, and (b) compose
  through the acceptance-weighted step TPS = (1 + E[accepted]) / T_step where only T_verify shrinks.

  Does +38.3 survive once the draft is separated?

THE ANSWER (decision-critical, on #388's BANKED M=1 number -- does NOT depend on #391):
  YES, the supply route SURVIVES. The honest verify-body-only lift is +32.65 (off-the-shelf base)
  / +42.91 (floor base) at the M=1 tier -- still clears BOTH #383 supply targets (+17.22 floor,
  +23.75 robust) at every cell. #388's headline was +5.69 TPS OPTIMISTIC (the draft-fraction
  credit), ~15% of the +38.3. NB: #388's own *gate* already used the honest body-only number
  (32.647, cleared); only its *headline* carried the optimistic complement. This card makes the
  headline honest, composes it explicitly through the #289 acceptance loop, extends to the floor
  base + the M=8/roofline tier, and inverts (#387) for the combined supply+demand private-500 route:
  cb3 (M=1) on the floor base + a +0.0117 demand sliver (38% of the #336 budget) reaches private-500.

WHAT THIS IS / IS NOT:
  Pure-CPU analytic card (stdlib math). 0 GPU, 0 official TPS, 0 HF Job, NO served-file change,
  NO submission. BASELINE 481.53 TPS / PPL 2.3772 UNCHANGED. Reuses denken #383's floor-base
  translation harness + denken #387's reprice_anchor_band inversion + the #289 MTP per-depth
  conditional-accept ladder + #378's step decomposition. Treats cb3 as acceptance/identity-neutral
  (the #372 PPL-feasibility gate 2.3812 <= 2.42 is inherited from #388; greedy-identity of cb3-on-
  target is a SEPARATE gate, out of scope here -- this card composes the SPEED lift only).

REPRODUCE (0-GPU):
    cd target/ && .venv/bin/python research/validity/cb3_supply_lift_mtp_honest/\
cb3_supply_lift_mtp_honest.py --self-test
    cd target/ && .venv/bin/python research/validity/cb3_supply_lift_mtp_honest/\
cb3_supply_lift_mtp_honest.py \
      --wandb_group cb3-supply-lift-mtp-honest --wandb_name denken/cb3-supply-lift-mtp-honest
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ===========================================================================
# Section 0 -- banked anchors (imported EXACTLY from merged advisor-branch cards / PR #392 body)
# ===========================================================================

# ---- #372 mixed-precision body-shrink + #388 (g5lfdpgw) realized speedup tiers --------------------
INT4_BPW: float = 4.125                     # deployed int4-Marlin: 4b weight + bf16 g128 scale
CB3_BPW_EFF: float = 3.2368598382749325     # #372 mixed cb3 (RHT+VQ K=64) effective bpw
QTIP_BETA_BYTE_PROPORTIONAL: float = 0.51   # #388 QTIP batch=1 byte-proportional fraction (M=1 realistic)
MIXED_GATE_PPL_372: float = 2.3811966031692555  # #372 measured cb3 PPL; passes <= 2.42 (precondition)
PPL_GATE: float = 2.42
# #388 BANKED headline (the +38.3 to round-trip) -- qtip_empirical x complement x off-the-shelf base.
PUBLISHED_388_HEADLINE_TPS: float = 38.34161969078741    # cb3_kernel_realized_bw_results.json
PUBLISHED_388_HONEST_BODY_ONLY_OFF_TPS: float = 32.64721137113611   # #388's own gate driver (body-only)
PUBLISHED_388_HONEST_BODY_ONLY_FLOOR_TPS: float = 42.913354216  # #388 body-only delta_floor
PUBLISHED_388_S_QTIP_M1: float = 1.1233511704212635      # #388 realized_body_speedup_qtip_empirical
PUBLISHED_388_S_ROOFLINE: float = 1.2743832622046425     # #388 realized_body_speedup (roofline, M=8 upper)

# ---- #378 (gghmgtk9) served-strict step decomposition (fractions of the full MTP step; sum to 1) --
F_ATTN: float = 0.09506718019009251         # attention (un-pack; un-shrunk by cb3)
F_BODY_STRICT: float = 0.76240970145034     # verify-body GEMM weight read -- the HONEST cb3-shrinkable frac
F_LMHEAD: float = 0.022428229458960704      # lm_head (already-strict int4-Marlin, #384; un-shrunk by cb3)
F_DRAFT: float = 0.12009488890060672        # spec DRAFTER forward (small model; NOT cb3-quantized -> un-shrunk)
BAND_OFF_THE_SHELF: float = 357.32166269999993   # #378 worse-case VBI=1 strict base (off-the-shelf #326)
BAND_FLOOR: float = 469.6847174760462            # #378 better-case VBI=1 strict base (first-principles #327)

# ---- #289 (fi34s269) DEPLOYED MTP per-position conditional acceptance ladder a_1..a_7 (K=7) -------
# The per-DEPTH profile the deployed spec-tree exposes (conditional accept). Encodes the K=7 loop.
# PR #392 cites the 3-digit ladder [0.729,0.760,0.793,0.823,0.835,0.836,0.846]; these are the precise
# banked #289 values (== that ladder to 3 digits) used by denken #383/#387.
LADDER_289: list[float] = [
    0.7292532942898975, 0.759556697719242, 0.7929794882639035, 0.8228,
    0.8348727920920435, 0.8357919254658385, 0.8464932652113331,
]

# ---- #383 (t68af2yw) supply-side targets the body-shrink must clear (lifts above the floor base) --
SUPPLY_FLOOR_JOINT_TPS: float = 17.216386736379093     # joint (E[T] + gap co-benefit) supply lift required
SUPPLY_ROBUST_ET_ONLY_TPS: float = 23.74874176829968   # E[T]-channel-only (conservative/robust) supply lift

# ---- #387/#383 public<->private inversion plumbing (private-strict-500) ---------------------------
MU_P: float = 481.53                 # deployed public TPS (PR #52, 2x9fm2zx)
MU_V: float = 460.85                 # organizer private-verified TPS for the same submission
GAP_MEASURED: float = 1.0 - MU_V / MU_P              # 0.042946 public->private gap
K_CAL: float = 125.26795005202914                    # steps/s; official TPS = E[T] * K_cal (#344)
E_T_REALIZED: float = MU_P / K_CAL                   # 3.84438 realized accept length at deployed point
ET_PUBLIC_500: float = 500.0 / K_CAL                 # 3.99144 (E[T] at the speed-500 bar)
ET_DEPLOYED: float = MU_P / K_CAL                     # 3.84438 (E[T] at deployed 481.53)
RHO_PRIV: float = 0.9421             # #300/#310 central regression correlation
COV_PRIOR: float = 0.8902659519153152                # #336/#330 modeled top-4 coverage anchor c0
IDENTITY_BAR: float = 0.9213011665456927             # #336 greedy-identity coverage bar
COV_BUDGET_336: float = 0.031035214630377506         # #336 trainable coverage headroom (bar - prior)
CSTAR_CENTRAL: float = 0.9089                         # #340 c* central (program coverage->E[T] secant)
TARGET: float = 500.0
# #383/#387 program coverage->E[T] central secant (anchor-coupled at the central prior).
S_CENTRAL: float = (ET_PUBLIC_500 - ET_DEPLOYED) / (CSTAR_CENTRAL - COV_PRIOR)   # ~7.91 E[T]/cov
PUBLISHED_383_REQ_DCOV_FLOOR: float = 0.05716864498666053   # #383 demand-alone floor (busts +0.031 budget)


# ===========================================================================
# Section 1 -- cb3 speedup tiers (M=1 banked + M=8/roofline upper bound)
# ===========================================================================

def byte_ratio() -> float:
    """cb3 reads this fraction of int4-Marlin's weight bytes (== #372 0.785)."""
    return CB3_BPW_EFF / INT4_BPW


def roofline_speedup() -> float:
    """M=8 verify roofline: at equal BW-efficiency cb3 is 1/byte_ratio faster (the UPPER bound,
    pending lawine #391's measured M=8 tier; tight iff M=8 verify stays BW-bound)."""
    return INT4_BPW / CB3_BPW_EFF


def qtip_m1_speedup() -> float:
    """#388 BANKED M=1 realistic speedup: only the byte-proportional fraction (beta~=0.51) of the
    step shrinks with the weight bytes; the rest is fixed dequant/launch overhead.
    realized_time_ratio = byte_ratio*beta + (1-beta)."""
    r = byte_ratio()
    return 1.0 / (r * QTIP_BETA_BYTE_PROPORTIONAL + (1.0 - QTIP_BETA_BYTE_PROPORTIONAL))


# ===========================================================================
# Section 2 -- #388's single-forward lift_factor (the model we make HONEST + round-trip)
# ===========================================================================

def lift_factor(speedup: float, f_body: float) -> float:
    """#388's served-TPS multiplier when the step-time fraction `f_body` speeds up by `speedup`.

    new_step = (1 - f_body) + f_body/speedup ; multiplier = 1/new_step. Copied EXACTLY from #388 so
    the round-trip is faithful: f_body=complement reproduces #388's optimistic headline; f_body=
    F_BODY_STRICT is the honest verify-body-only lift."""
    r = 1.0 / speedup
    new_step = (1.0 - f_body) + f_body * r
    return 1.0 / new_step


def f_body_complement() -> float:
    """#388's PR-spec body fraction: complement of attention + lm_head. LUMPS the draft fraction in
    (= F_BODY_STRICT + F_DRAFT) -- the OPTIMISTIC credit this card removes."""
    return 1.0 - F_ATTN - F_LMHEAD


# ===========================================================================
# Section 3 -- E[accepted] from the #289 MTP ladder (the acceptance-weighted step)
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
    """E[T] = tokens emitted per step = 1 (the always-emitted target/verify token) + E[accepted]."""
    return 1.0 + expected_accepted(ladder)


# ===========================================================================
# Section 4 -- the HONEST MTP-loop composition (credit cb3 ONLY to the verify body)
# ===========================================================================

def honest_step_compose(base_tps: float, speedup: float, et: float) -> dict:
    """Compose the honest served-TPS lift through the acceptance-weighted MTP K=7 step.

    served TPS = (1 + E[accepted]) / T_step. The base served TPS already encodes E[T], so we back
    out the per-step wallclock T_step = et / base_tps (arbitrary unit), DECOMPOSE it into
    draft + verify-body + lm_head + attention via the #378 fractions, apply the cb3 speedup ONLY to
    the verify-body component, recompose, and re-divide by the (unchanged) E[T].

    This is algebraically base_tps * lift_factor(speedup, F_BODY_STRICT) -- E[T] cancels -- but the
    explicit decomposition makes the draft-separation transparent (the draft component is held fixed)."""
    t_step_old = et / base_tps
    t_draft = F_DRAFT * t_step_old              # un-shrunk (separate small drafter model)
    t_verify_body = F_BODY_STRICT * t_step_old  # cb3-shrinkable
    t_lmhead = F_LMHEAD * t_step_old            # un-shrunk (already-strict int4-Marlin, #384)
    t_attn = F_ATTN * t_step_old                # un-shrunk (un-pack attention, #378)
    t_verify_body_new = t_verify_body / speedup
    t_step_new = t_draft + t_verify_body_new + t_lmhead + t_attn
    tps_new = et / t_step_new
    return {
        "base_tps": base_tps,
        "speedup": speedup,
        "e_t_used": et,
        "t_step_old": t_step_old,
        "t_step_new": t_step_new,
        "t_draft_unshrunk": t_draft,
        "t_verify_body_old": t_verify_body,
        "t_verify_body_new": t_verify_body_new,
        "t_lmhead_unshrunk": t_lmhead,
        "t_attn_unshrunk": t_attn,
        "tps_new": tps_new,
        "honest_lift_tps": tps_new - base_tps,
        "lift_factor_check": base_tps * lift_factor(speedup, F_BODY_STRICT),
    }


# ===========================================================================
# Section 5 -- honest lift at both bases x both tiers + optimism gap (deliverables 1-3)
# ===========================================================================

def honest_lift_table(et: float) -> dict:
    """Honest verify-body-only lift on BOTH #378 bases at BOTH cb3 tiers (M=1 banked / roofline upper)."""
    s_m1 = qtip_m1_speedup()
    s_roof = roofline_speedup()
    out: dict[str, dict] = {}
    for base_name, base in (("off_the_shelf", BAND_OFF_THE_SHELF), ("floor", BAND_FLOOR)):
        for tier_name, S in (("m1_banked", s_m1), ("m8_roofline", s_roof)):
            comp = honest_step_compose(base, S, et)
            out[f"{base_name}__{tier_name}"] = {
                "base": base, "speedup": S,
                "honest_lifted_tps": comp["tps_new"],
                "honest_lift_tps": comp["honest_lift_tps"],
            }
    return out


def optimism_gap() -> dict:
    """#388's headline (credit-whole-complement, M=1, off-the-shelf) MINUS the honest verify-body-only
    lift at the same cell. Shows exactly how much of +38.3 was the draft-fraction optimism."""
    s_m1 = qtip_m1_speedup()
    headline = BAND_OFF_THE_SHELF * lift_factor(s_m1, f_body_complement()) - BAND_OFF_THE_SHELF
    honest = BAND_OFF_THE_SHELF * lift_factor(s_m1, F_BODY_STRICT) - BAND_OFF_THE_SHELF
    # the floor base counterpart (same complement-vs-body-only contrast, larger base)
    headline_floor = BAND_FLOOR * lift_factor(s_m1, f_body_complement()) - BAND_FLOOR
    honest_floor = BAND_FLOOR * lift_factor(s_m1, F_BODY_STRICT) - BAND_FLOOR
    return {
        "credit_whole_complement_headline_388_off": headline,
        "honest_verify_body_only_off": honest,
        "optimism_gap_tps": headline - honest,                       # off-the-shelf, M=1 (matches #388 headline)
        "credit_whole_complement_headline_388_floor": headline_floor,
        "honest_verify_body_only_floor": honest_floor,
        "optimism_gap_tps_floor": headline_floor - honest_floor,
        "optimism_gap_frac_of_headline": (headline - honest) / headline,
        "draft_fraction_pinned": F_DRAFT,
    }


# ===========================================================================
# Section 6 -- #383 supply gate (deliverable 4): does the honest lift clear +17.22 / +23.75?
# ===========================================================================

def supply_gate(lift_tps: float) -> dict:
    return {
        "lift_tps": lift_tps,
        "clears_383_floor": bool(lift_tps >= SUPPLY_FLOOR_JOINT_TPS),
        "clears_383_robust": bool(lift_tps >= SUPPLY_ROBUST_ET_ONLY_TPS),
    }


def gate_table() -> dict:
    """#383 floor/robust gate on the honest lift, at BOTH tiers. The floor base is the apples-to-apples
    comparison (#383's targets are lifts above the 469.68 floor); off-the-shelf is the conservative
    mismatch (a smaller-base lift vs a floor-relative target). The DECISION-CRITICAL headline uses the
    M=1 banked tier; the roofline tier is the pending-#391 upper sharpening."""
    s_m1 = qtip_m1_speedup()
    s_roof = roofline_speedup()
    rows: dict[str, dict] = {}
    for base_name, base in (("off_the_shelf", BAND_OFF_THE_SHELF), ("floor", BAND_FLOOR)):
        for tier_name, S in (("m1_banked", s_m1), ("m8_roofline", s_roof)):
            lift = base * lift_factor(S, F_BODY_STRICT) - base
            rows[f"{base_name}__{tier_name}"] = supply_gate(lift)
    # headline = M=1 tier, floor base (apples-to-apples, conservative tier).
    headline = rows["floor__m1_banked"]
    return {
        "rows": rows,
        "honest_lift_clears_383_floor": headline["clears_383_floor"],
        "honest_lift_clears_383_robust": headline["clears_383_robust"],
        "headline_cell": "floor__m1_banked",
        # also report the strictest cell (off-the-shelf, M=1) for transparency.
        "off_the_shelf_m1_clears_floor": rows["off_the_shelf__m1_banked"]["clears_383_floor"],
        "off_the_shelf_m1_clears_robust": rows["off_the_shelf__m1_banked"]["clears_383_robust"],
    }


# ===========================================================================
# Section 7 -- #387 inversion (deliverable 5): residual demand + combined private-500 route
# ===========================================================================

def public_for_private_500_regression(rho: float = RHO_PRIV, g: float = GAP_MEASURED) -> float:
    """#387: invert project(P; rho, g)=500 -> public ceiling needed for private-500 (anchor-indep)."""
    return MU_P + (TARGET / (1.0 - g) - MU_P) / rho


def required_dcov_program_secant(base: float, anchor: float = COV_PRIOR,
                                 cstar: float = CSTAR_CENTRAL, rho: float = RHO_PRIV) -> float:
    """#387/#383 model-I: Delta-coverage demand to reach private-500 from a public `base`, via the
    program coverage->E[T] secant. delta_et is anchor-independent; the secant is anchor-coupled."""
    pstar = public_for_private_500_regression(rho)
    delta_et = E_T_REALIZED * (pstar / base - 1.0)
    s = (ET_PUBLIC_500 - ET_DEPLOYED) / (cstar - anchor)
    return delta_et / s


def combined_route(et: float) -> dict:
    """Lift the floor base by the honest cb3 supply lift, then invert (#387) for the residual Delta-cov
    the demand side must still supply, and whether supply+demand reaches private-500 within +0.031."""
    s_m1 = qtip_m1_speedup()
    s_roof = roofline_speedup()
    out: dict[str, dict] = {}
    for tier_name, S in (("m1_banked", s_m1), ("m8_roofline", s_roof)):
        lift = BAND_FLOOR * lift_factor(S, F_BODY_STRICT) - BAND_FLOOR
        base = BAND_FLOOR + lift
        raw = required_dcov_program_secant(base)
        resid = max(0.0, raw)
        out[tier_name] = {
            "honest_supply_lift_tps": lift,
            "supply_lifted_base": base,
            "residual_demand_dcov_raw": raw,           # <0 => supply alone over-delivers private-500
            "residual_demand_dcov": resid,
            "residual_frac_of_336_budget": resid / COV_BUDGET_336,
            "combined_route_reaches_500": bool(resid <= COV_BUDGET_336),
            "supply_alone_reaches_500": bool(raw <= 0.0),
        }
    # headline = M=1 banked tier (conservative).
    head = out["m1_banked"]
    return {
        "tiers": out,
        "residual_demand_dcov_honest": head["residual_demand_dcov"],
        "combined_route_reaches_500_honest": head["combined_route_reaches_500"],
        "pstar_public_for_private_500": public_for_private_500_regression(),
        "383_floor_demand_alone_dcov": required_dcov_program_secant(BAND_FLOOR),  # round-trips +0.0572
    }


# ===========================================================================
# Section 8 -- self-tests (>=20 checks)
# ===========================================================================

def _finite(x: float) -> bool:
    return isinstance(x, (int, float)) and not (math.isnan(x) or math.isinf(x))


def run_self_tests(et: float, lifts: dict, gap: dict, gate: dict, comb: dict) -> dict:
    c: dict[str, bool] = {}
    s_m1 = qtip_m1_speedup()
    s_roof = roofline_speedup()

    # a) cb3 tiers reproduce #388's BANKED speedups (provenance).
    c["a_m1_speedup_matches_388"] = abs(s_m1 - PUBLISHED_388_S_QTIP_M1) < 1e-9
    c["a_roofline_speedup_matches_388"] = abs(s_roof - PUBLISHED_388_S_ROOFLINE) < 1e-9
    c["a_byte_ratio_is_372_0p785"] = round(byte_ratio(), 3) == 0.785
    c["a_roofline_ge_m1"] = s_roof >= s_m1 > 1.0   # M=8 roofline is the upper tier

    # b) ROUND-TRIP #388's +38.3 under the credit-whole-complement assumption (reproduces exactly).
    headline = BAND_OFF_THE_SHELF * lift_factor(s_m1, f_body_complement()) - BAND_OFF_THE_SHELF
    c["b_roundtrips_388_headline_exact"] = abs(headline - PUBLISHED_388_HEADLINE_TPS) < 1e-6
    c["b_roundtrips_388_headline_rounds_38p3"] = round(headline, 1) == 38.3
    # and the honest body-only number matches #388's own (already-honest) gate driver.
    honest_off = BAND_OFF_THE_SHELF * lift_factor(s_m1, F_BODY_STRICT) - BAND_OFF_THE_SHELF
    c["b_honest_matches_388_gate_body_only"] = abs(honest_off - PUBLISHED_388_HONEST_BODY_ONLY_OFF_TPS) < 1e-6
    c["b_optimism_gap_positive"] = gap["optimism_gap_tps"] > 0          # headline WAS optimistic
    c["b_optimism_gap_is_draft_credit"] = honest_off < headline          # body-only < complement

    # c) the explicit MTP-loop composition == the body-only lift_factor (E[T] cancels) at every cell.
    ok_compose = True
    for base in (BAND_OFF_THE_SHELF, BAND_FLOOR):
        for S in (s_m1, s_roof):
            comp = honest_step_compose(base, S, et)
            if abs(comp["tps_new"] - comp["lift_factor_check"]) > 1e-7:
                ok_compose = False
    c["c_mtp_compose_equals_body_only_liftfactor"] = ok_compose
    # the draft component is held FIXED across the cb3 shrink (the separation is real).
    comp_ref = honest_step_compose(BAND_FLOOR, s_m1, et)
    c["c_draft_component_unshrunk"] = comp_ref["t_draft_unshrunk"] > 0 and \
        comp_ref["t_step_new"] < comp_ref["t_step_old"]
    c["c_only_verify_body_shrinks"] = abs(comp_ref["t_verify_body_new"] - comp_ref["t_verify_body_old"] / s_m1) < 1e-12

    # d) #289 ladder: K=7, in (0,1), MONOTONE non-decreasing; E[T] consistent with E_T_REALIZED.
    c["d_ladder_len_7"] = len(LADDER_289) == 7
    c["d_ladder_in_unit"] = all(0.0 < a < 1.0 for a in LADDER_289)
    c["d_ladder_monotone_nondecreasing"] = all(LADDER_289[i] <= LADDER_289[i + 1]
                                               for i in range(len(LADDER_289) - 1))
    c["d_et_ladder_matches_realized"] = abs(et - E_T_REALIZED) / E_T_REALIZED < 0.01   # within ~0.2%
    c["d_et_in_1_to_8"] = 1.0 < et < 8.0

    # e) step fractions: sum to 1; verify-body = complement - draft (the honest separation identity).
    fsum = F_ATTN + F_BODY_STRICT + F_LMHEAD + F_DRAFT
    c["e_fractions_sum_1"] = abs(fsum - 1.0) < 1e-9
    c["e_verify_body_is_complement_minus_draft"] = abs(F_BODY_STRICT - (f_body_complement() - F_DRAFT)) < 1e-12
    c["e_draft_fraction_pinned_378"] = abs(F_DRAFT - 0.12009488890060672) < 1e-15

    # f) #383 gate: honest lift CLEARS floor + robust at the headline cell AND the strictest cell.
    c["f_clears_383_floor_headline"] = gate["honest_lift_clears_383_floor"]
    c["f_clears_383_robust_headline"] = gate["honest_lift_clears_383_robust"]
    c["f_clears_383_robust_off_m1_strictest"] = gate["off_the_shelf_m1_clears_robust"]
    c["f_supply_targets_ordered"] = SUPPLY_FLOOR_JOINT_TPS < SUPPLY_ROBUST_ET_ONLY_TPS

    # g) #383 floor demand-alone round-trip (base 469.68 -> +0.0572, busts the +0.031 budget).
    c["g_roundtrips_383_floor_demand_0p0572"] = abs(comb["383_floor_demand_alone_dcov"]
                                                    - PUBLISHED_383_REQ_DCOV_FLOOR) < 5e-4
    c["g_383_floor_demand_busts_budget"] = comb["383_floor_demand_alone_dcov"] > COV_BUDGET_336

    # h) combined route: residual demand fits the #336 budget; supply+demand reaches private-500.
    c["h_residual_within_budget"] = comb["residual_demand_dcov_honest"] <= COV_BUDGET_336
    c["h_combined_reaches_500"] = comb["combined_route_reaches_500_honest"]
    c["h_residual_smaller_than_383_demand_alone"] = (comb["residual_demand_dcov_honest"]
                                                     < comb["383_floor_demand_alone_dcov"])
    c["h_pstar_above_deployed"] = comb["pstar_public_for_private_500"] > MU_P

    # i) #372 PPL precondition passes (cb3 is PPL-feasible -- the body-shrink is admissible).
    c["i_ppl_372_passes"] = MIXED_GATE_PPL_372 <= PPL_GATE

    # j) numeric hygiene across all headline scalars.
    flat = [et, s_m1, s_roof, headline, honest_off, gap["optimism_gap_tps"],
            comb["residual_demand_dcov_honest"], comb["pstar_public_for_private_500"],
            lifts["floor__m1_banked"]["honest_lift_tps"], lifts["off_the_shelf__m1_banked"]["honest_lift_tps"]]
    c["j_no_nan_inf"] = all(_finite(v) for v in flat)

    passes = all(c.values())
    return {"conditions": c, "n_checks": len(c), "n_passed": sum(1 for v in c.values() if v), "passes": passes}


# ===========================================================================
# Section 9 -- report assembly + W&B + CLI
# ===========================================================================

def build_report() -> dict:
    et = expected_tokens_per_step(LADDER_289)
    lifts = honest_lift_table(et)
    gap = optimism_gap()
    gate = gate_table()
    comb = combined_route(et)
    selftest = run_self_tests(et, lifts, gap, gate, comb)
    return {
        "pr": 392, "agent": "denken", "kind": "cb3-supply-lift-mtp-honest",
        "analysis_only": True, "no_launch": True, "no_hf_job": True, "no_submission": True,
        "no_served_file_change": True, "gpu_used": False, "official_tps_expected": 0,
        "baseline_unchanged_tps": 481.53, "baseline_unchanged_ppl": 2.3772,
        "inputs": {
            "int4_bpw": INT4_BPW, "cb3_bpw_eff": CB3_BPW_EFF, "qtip_beta": QTIP_BETA_BYTE_PROPORTIONAL,
            "f_attn_378": F_ATTN, "f_body_strict_378": F_BODY_STRICT, "f_lmhead_378": F_LMHEAD,
            "f_draft_378": F_DRAFT, "band_off_the_shelf_378": BAND_OFF_THE_SHELF, "band_floor_378": BAND_FLOOR,
            "ladder_289": LADDER_289, "supply_floor_383": SUPPLY_FLOOR_JOINT_TPS,
            "supply_robust_383": SUPPLY_ROBUST_ET_ONLY_TPS, "mu_p": MU_P, "mu_v": MU_V, "k_cal": K_CAL,
            "rho_priv": RHO_PRIV, "cov_prior": COV_PRIOR, "identity_bar": IDENTITY_BAR,
            "cov_budget_336": COV_BUDGET_336, "cstar_central": CSTAR_CENTRAL, "s_central": S_CENTRAL,
            "published_388_headline": PUBLISHED_388_HEADLINE_TPS,
            "published_388_s_qtip_m1": PUBLISHED_388_S_QTIP_M1, "published_388_s_roofline": PUBLISHED_388_S_ROOFLINE,
            "ppl_gate": PPL_GATE, "mixed_gate_ppl_372": MIXED_GATE_PPL_372,
            "source_388_run": "g5lfdpgw", "source_378_run": "gghmgtk9", "source_383_run": "t68af2yw",
            "source_387_run": "z8osvif8", "source_289_run": "fi34s269", "source_384_run": "4f32ks1e",
            "source_391_run": "PENDING (cb3-m8-verify-body-speedup)",
        },
        "expected_tokens_per_step": et,
        "expected_accepted_draft": et - 1.0,
        "speedup_m1_banked": qtip_m1_speedup(),
        "speedup_m8_roofline": roofline_speedup(),
        "honest_lift_table": lifts,
        "optimism_gap": gap,
        "supply_gate_383": gate,
        "combined_route": comb,
        # ---- card-required headline scalars (SENPAI-RESULT load-bearing) ----
        "honest_supply_lift_tps_m1_off": lifts["off_the_shelf__m1_banked"]["honest_lift_tps"],
        "honest_supply_lift_tps_m1_floor": lifts["floor__m1_banked"]["honest_lift_tps"],
        "honest_supply_lift_tps_roofline_off": lifts["off_the_shelf__m8_roofline"]["honest_lift_tps"],
        "honest_supply_lift_tps_roofline_floor": lifts["floor__m8_roofline"]["honest_lift_tps"],
        "optimism_gap_tps": gap["optimism_gap_tps"],
        "honest_lift_clears_383_floor": gate["honest_lift_clears_383_floor"],
        "honest_lift_clears_383_robust": gate["honest_lift_clears_383_robust"],
        "residual_demand_dcov_honest": comb["residual_demand_dcov_honest"],
        "combined_route_reaches_500_honest": comb["combined_route_reaches_500_honest"],
        "self_test": selftest,
        "cb3_supply_lift_mtp_self_test_passes": selftest["passes"],
    }


def log_to_wandb(report: dict, group: str, name: str) -> str:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"W&B import failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_unavailable"
    try:
        run = wandb.init(group=group, name=name, config=report["inputs"])
        gap, gate, comb, lifts = (report["optimism_gap"], report["supply_gate_383"],
                                  report["combined_route"], report["honest_lift_table"])
        wandb.log({
            "summary/honest_supply_lift_tps_m1_off": report["honest_supply_lift_tps_m1_off"],
            "summary/honest_supply_lift_tps_m1_floor": report["honest_supply_lift_tps_m1_floor"],
            "summary/honest_supply_lift_tps_roofline_off": report["honest_supply_lift_tps_roofline_off"],
            "summary/honest_supply_lift_tps_roofline_floor": report["honest_supply_lift_tps_roofline_floor"],
            "summary/optimism_gap_tps": report["optimism_gap_tps"],
            "summary/optimism_gap_tps_floor": gap["optimism_gap_tps_floor"],
            "summary/optimism_gap_frac_of_headline": gap["optimism_gap_frac_of_headline"],
            "summary/credit_whole_complement_headline_388": gap["credit_whole_complement_headline_388_off"],
            "summary/honest_lift_clears_383_floor": float(report["honest_lift_clears_383_floor"]),
            "summary/honest_lift_clears_383_robust": float(report["honest_lift_clears_383_robust"]),
            "summary/off_the_shelf_m1_clears_robust": float(gate["off_the_shelf_m1_clears_robust"]),
            "summary/residual_demand_dcov_honest": report["residual_demand_dcov_honest"],
            "summary/residual_frac_of_336_budget": comb["tiers"]["m1_banked"]["residual_frac_of_336_budget"],
            "summary/combined_route_reaches_500_honest": float(report["combined_route_reaches_500_honest"]),
            "summary/combined_route_reaches_500_roofline": float(comb["tiers"]["m8_roofline"]["combined_route_reaches_500"]),
            "summary/supply_alone_reaches_500_roofline": float(comb["tiers"]["m8_roofline"]["supply_alone_reaches_500"]),
            "summary/expected_tokens_per_step": report["expected_tokens_per_step"],
            "summary/expected_accepted_draft": report["expected_accepted_draft"],
            "summary/speedup_m1_banked": report["speedup_m1_banked"],
            "summary/speedup_m8_roofline": report["speedup_m8_roofline"],
            "summary/383_floor_demand_alone_dcov": comb["383_floor_demand_alone_dcov"],
            "summary/pstar_public_for_private_500": comb["pstar_public_for_private_500"],
            "summary/self_test_passes": float(report["self_test"]["passes"]),
            "summary/self_test_n_checks": float(report["self_test"]["n_checks"]),
        })
        for cell, row in lifts.items():
            wandb.log({f"lift/{cell}/honest_lift_tps": row["honest_lift_tps"],
                       f"lift/{cell}/honest_lifted_tps": row["honest_lifted_tps"]})
        for tier, row in comb["tiers"].items():
            wandb.log({f"combined/{tier}/supply_lifted_base": row["supply_lifted_base"],
                       f"combined/{tier}/residual_demand_dcov": row["residual_demand_dcov"],
                       f"combined/{tier}/combined_reaches_500": float(row["combined_route_reaches_500"])})
        for cond, val in report["self_test"]["conditions"].items():
            wandb.log({f"test/{cond}": float(bool(val))})
        run_id = run.id
        wandb.finish()
        return run_id
    except Exception as exc:  # noqa: BLE001
        print(f"W&B logging failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_failed"


def print_report(r: dict) -> None:
    gap, gate, comb, lifts = (r["optimism_gap"], r["supply_gate_383"], r["combined_route"], r["honest_lift_table"])
    print("\n=== Honest cb3 supply lift through MTP K=7 (PR #392, denken) ===")
    print(f"cb3 tiers: M=1 banked {r['speedup_m1_banked']:.4f}x (#388) | M=8 roofline {r['speedup_m8_roofline']:.4f}x "
          f"(upper, #391 PENDING)   byte_ratio={byte_ratio():.4f}")
    print(f"#289 ladder E[accepted]={r['expected_accepted_draft']:.4f}  E[T]={r['expected_tokens_per_step']:.4f}  "
          f"(== deployed E_T_REALIZED {E_T_REALIZED:.4f} to {abs(r['expected_tokens_per_step']-E_T_REALIZED)/E_T_REALIZED*100:.2f}%)")
    print("\n-- DRAFT SEPARATION: credit cb3 ONLY to the verify-body fraction (0.7624), NOT the draft (0.1201) --")
    print(f"  #388 headline (credit-whole-complement, M=1, off-the-shelf) = +{gap['credit_whole_complement_headline_388_off']:.2f} TPS")
    print(f"  HONEST verify-body-only            (M=1, off-the-shelf)     = +{gap['honest_verify_body_only_off']:.2f} TPS")
    print(f"  optimism_gap_tps = +{gap['optimism_gap_tps']:.2f} TPS  ({gap['optimism_gap_frac_of_headline']*100:.1f}% of the +38.3 headline was draft-fraction credit)")
    print("\n-- honest_supply_lift_tps (verify-body-only, both bases x both tiers) --")
    for cell, row in lifts.items():
        print(f"    {cell:<26}: base {row['base']:.2f}  x{row['speedup']:.4f}  ->  {row['honest_lifted_tps']:.2f}  (+{row['honest_lift_tps']:.2f})")
    print("\n-- #383 supply gate (targets: +17.22 floor / +23.75 robust, above the 469.68 floor) --")
    for cell, row in gate["rows"].items():
        print(f"    {cell:<26}: +{row['lift_tps']:.2f}  clears_floor={row['clears_383_floor']}  clears_robust={row['clears_383_robust']}")
    print(f"  HEADLINE ({gate['headline_cell']}): honest_lift_clears_383_floor={gate['honest_lift_clears_383_floor']}  "
          f"honest_lift_clears_383_robust={gate['honest_lift_clears_383_robust']}")
    print("\n-- combined route to private-500 (#387 inversion at floor + honest supply lift) --")
    print(f"  pstar (public for private-500) = {comb['pstar_public_for_private_500']:.2f}   "
          f"#383 demand-alone @469.68 = +{comb['383_floor_demand_alone_dcov']:.4f} (busts +{COV_BUDGET_336:.4f})")
    for tier, row in comb["tiers"].items():
        tag = "SUPPLY-ALONE" if row["supply_alone_reaches_500"] else f"+{row['residual_demand_dcov']:.4f} dcov ({row['residual_frac_of_336_budget']*100:.0f}% budget)"
        print(f"    {tier:<12}: supply {BAND_FLOOR:.2f}->{row['supply_lifted_base']:.2f}  residual demand: {tag}  "
              f"reaches_500={row['combined_route_reaches_500']}")
    print(f"\n  residual_demand_dcov_honest = +{r['residual_demand_dcov_honest']:.4f}  "
          f"combined_route_reaches_500_honest = {r['combined_route_reaches_500_honest']}")
    print(f"\nself-test: {r['self_test']['n_passed']}/{r['self_test']['n_checks']} checks  "
          f"cb3_supply_lift_mtp_self_test_passes = {r['cb3_supply_lift_mtp_self_test_passes']}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Honest cb3 supply lift through MTP K=7 (PR #392).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="0-GPU analytic gate (PR #392 deliverables)")
    ap.add_argument("--reanalyze", action="store_true", help="0-GPU full re-analysis (alias of default)")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="cb3-supply-lift-mtp-honest")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default="denken/cb3-supply-lift-mtp-honest")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", type=str,
                    default="research/validity/cb3_supply_lift_mtp_honest/cb3_supply_lift_mtp_honest_results.json")
    args = ap.parse_args()

    report = build_report()
    print_report(report)

    if args.self_test:
        out = Path("research/validity/cb3_supply_lift_mtp_honest/cb3_supply_lift_mtp_honest_selftest.json")
        out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {out}")
        print(f"\ncb3_supply_lift_mtp_self_test_passes = {report['self_test']['passes']}")
        return 0 if report["self_test"]["passes"] else 1

    report["wandb_run_id"] = None if args.no_wandb else log_to_wandb(report, args.wandb_group, args.wandb_name)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}  (W&B run {report.get('wandb_run_id')})")

    print("\nSENPAI-RESULT " + json.dumps({
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [report["wandb_run_id"]] if report.get("wandb_run_id") else [],
        "no_hf_job": True, "official_tps": 0.0, "analysis_only": True,
        "honest_supply_lift_tps_m1_off": float(report["honest_supply_lift_tps_m1_off"]),
        "honest_supply_lift_tps_m1_floor": float(report["honest_supply_lift_tps_m1_floor"]),
        "honest_supply_lift_tps_roofline_band_floor": float(report["honest_supply_lift_tps_roofline_floor"]),
        "optimism_gap_tps": float(report["optimism_gap_tps"]),
        "honest_lift_clears_383_floor": bool(report["honest_lift_clears_383_floor"]),
        "honest_lift_clears_383_robust": bool(report["honest_lift_clears_383_robust"]),
        "residual_demand_dcov_honest": float(report["residual_demand_dcov_honest"]),
        "combined_route_reaches_500_honest": bool(report["combined_route_reaches_500_honest"]),
        "cb3_supply_lift_mtp_self_test_passes": bool(report["cb3_supply_lift_mtp_self_test_passes"]),
        "primary_metric": {"name": "cb3_supply_lift_mtp_self_test_passes",
                           "value": float(report["cb3_supply_lift_mtp_self_test_passes"])},
        "test_metric": {"name": "honest_supply_lift_tps_m1_floor",
                        "value": float(report["honest_supply_lift_tps_m1_floor"])},
    }))
    return 0 if report["self_test"]["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
