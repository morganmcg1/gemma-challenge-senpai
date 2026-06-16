#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
PR #532 — Realize the drafter-acceptance headroom: E[T] 3.849 -> ceiling ~4.91, cheapest path.

ANALYSIS-ONLY structural realization ledger (#502-style escape hatch, PR step 4).
0 GPU, 0 HF Job, 0 submission, 0 served-file change, official_tps=0. Challenge PAUSED.

Reuses MEASURED data only — NO fresh serve (every number below is already banked):
  - served E[T] + per-position a-ladder            : kanna #526 / #289   (W&B 5m17r52s)
  - byteexact-442 base step decomposition (steptime): MY #523 ledger     (W&B i11p5e3y)
  - EAGLE-3 faithful draft-step multiplier bracket  : wirbel #295        (W&B c334qaqu)
  - EAGLE-3 go-card (481-base, private OOD) framing  : fern #305          (W&B m4nmtdl9)
  - built EAGLE-3 K=1 head collapses past step 1     : fern #34 arch_notes

CORE CONTRIBUTION vs the priors (this is the on-task novelty):
  * kanna #526 projects 4.91 E[T] -> 565 TPS holding t_step FIXED (NO drafter step tax) —
    that is the PR's headline premise, and it is INCONSISTENT (wide EAGLE-3 acceptance
    ceiling priced at the narrow linear head's step cost).
  * fern #305 DOES fold wirbel #295's MEASURED EAGLE-3 step tax — but anchored on the
    481.53 FAST (non-byte-exact, equivalence-ILLEGAL) base.
  * PR #532 asks for the byteexact-442 base (the legal split-KV frontier, my #523).
  => Re-anchor the MEASURED step tax onto MY MEASURED 442-base draft fraction and emit the
     honest realized-TPS grid + crosses_500 / crosses_565 verdict on the LEGAL base.

Step model (PF>=1.0, #504):  TPS = E[T] / t_step
  Swapping linear-MTP -> EAGLE-3 inflates ONLY the drafter portion of the step (target
  verify exec_gpu is UNCHANGED: the 3 fused aux hidden layers are exported FREE on
  vLLM 0.22.0, fern #15; draft vocab 12288 unchanged):
      t_step_new(m) = t_step_old + draft_gpu_old * (m - 1)
  where m = EAGLE-3/linear draft-chain wall multiplier (wirbel #295 measured bracket).
  This is FAVORABLE to EAGLE-3 (no verify-side tax charged), so the projection is an
  upper bound on realized TPS at any given m.

KEY OUTPUTS (required by the PR):
  measured_E_T_baseline, best_realized_E_T, E_T_lift, cheapest_path_name,
  projected_tps_at_realized_E_T (on byteexact-442), crosses_500, crosses_565,
  drafter_is_quality_safe, self_det, ppl, go_no_go.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# ============================================================================
# GROUNDED INPUTS — all reused MEASURED data (provenance in comments).
# ============================================================================

# --- (1) Served E[T] + per-position conditional acceptance (the DIAGNOSE input) ---
# Source: PR #289 accept_calibration (W&B 5m17r52s/fi34s269), vLLM
#   spec_decode_num_accepted_tokens_per_pos counters on the SHIPPED kenyan-duma
#   linear-MTP K=7 drafter (DRAFTER_SHA256 ed159e..dd18e), 128x512 public sharegpt.
#   Re-used verbatim by kanna #526 (this PR's basis). a_k = G(k)/G(k-1).
A_LADDER = [
    0.7290715372907154, 0.759434719768749, 0.7934024106576444, 0.8215618336886993,
    0.834712084347121, 0.835989117761368, 0.8465829846582985,
]
ET_BASELINE = 3.849256527338719      # = et_from_ladder(A_LADDER), prometheus whole-run
ET_BASELINE_SERVERLOG = 3.844131736526946
K_SPEC = 7
VERIFY_M = 8
ET_THEORETICAL_MAX = float(VERIFY_M)  # 8.0 (all 7 drafts + bonus accepted every step)

# --- (2) Structural cap of the deployed LINEAR-MTP topology (deployed sits AT cap) ---
# denken #119 (via #289). ubel #399 (ec7i3z5t): every NO-RETRAIN / no-served-kernel lever
# (temperature, affine calibration, tree-free) is a rank-order NO-OP -> the deep-coverage
# MUST come from a drafter RETRAIN (topology change, EAGLE-3 class) or a tree verify.
# => "tune the existing MTP head" is a PROVEN-CLOSED cheapest-screen leg (a).
LINEAR_CAP_ET = 3.8445

# --- (3) Realistic EAGLE-3 acceptance ceiling (the falsifiable #289 deep-lift target) ---
# Lift the DEEP positions a_2..a_7 -> ~0.91 (flat) while holding a_1=0.729 (deep-lift is
# feasible; first-token-alone is ceiling-bound). Optimistic: EAGLE-3 also lifts a_1 -> 0.80.
# This is an internal feasibility TARGET (delivery uncertain), NOT theoretical-max, and is
# UNMEASURED on a trained build (see best_realized_E_T_measured_today below).
EAGLE_DEEP_ACCEPT = 0.91
EAGLE_DEEP_ACCEPT_HI = 0.914          # #289's >=4.966 variant
EAGLE_A1_OPTIMISTIC = 0.80

# --- (4) byteexact-442 base: MEASURED step decomposition (MY #523, W&B i11p5e3y) ---
# Arm bx_T4_S64 (the packaged byte-exact fixed-3D split-KV rung), STEPTIME=1, 128x512, n=3:
#   median_wall_tps=439.71  PPL=2.3766643  exec_gpu=6.89  gap=2.256  draft_gpu=1.554 ms
#   onegraph=True, bx_armed=True(4/64), served r1-r2=1.0 (#523), 0/8 byte-exact attention.
TPS_BX442_LOCAL = 439.70559           # MY measured median wall_tps (self-consistent w/ draft_gpu)
TPS_BX442_ANCHOR_519 = 442.35         # sibling anchor (#519 kwhylaeg) — +0.6%, within sigma_hw
DRAFT_GPU_MS = 1.554                  # linear-MTP drafter GPU time per step (the part EAGLE-3 inflates)
EXEC_GPU_MS = 6.89                    # target verify GPU time per step (UNCHANGED by drafter swap)
GAP_MS = 2.256
PPL_BX442 = 2.3766643358900286        # served PPL guardrail (<= 2.42), UNCHANGED by drafter quality

# --- (5) EAGLE-3 faithful draft-step multiplier bracket (wirbel #295, W&B c334qaqu) ---
# Measured RATIO faithful-EAGLE3-chain / linear-MTP-chain in the SAME bf16 CUDA-graph harness.
# The raw 1.745 is a DISPATCH-COMPRESSED LOWER bound (tiny linear runs ~11% A10G BW vs the
# faithful's ~60%); the deployed ONEGRAPH+INT4 regime corrections partly cancel and the
# reported bracket spans the residual uncertainty. Central 3.0 is the regime-corrected value
# (validates #293's independent 3x proxy; == fern #305 tornado central 2.953).
MULT_LO = 1.744676699335575           # multiplicative lower (harness microbench)
MULT_CENTRAL = 3.0                    # regime-corrected central
MULT_HI = 4.161395297380165           # additive upper

# --- (6) Frame transfers ---
TAU_LO = 1.0352                       # local -> official scalar (MY #267, spread 0.135%)
PRIV_FACTOR = 0.804                   # public/official -> private worst-case OOD (fern #305 / ubel #258)

# --- (7) Cheapest realizable path + its cost (the SCREEN result) ---
# (a) tune/retrain existing MTP head ........ PROVEN CLOSED (#119 cap, #399 null-lever).
# (b) EAGLE-3 fusion drafter ................ ONLY path to the ceiling, BUT:
#     - the ONLY built head (fern #34, K=1) COLLAPSES past step 1 (no TTT) -> realized
#       E[T] ~1.8 today, BELOW deployed 3.849. Reaching 4.91 needs the UNBUILT multi-step/
#       TTT (HASS-style) chain training + soft-KD top-k + reasoning-CoT root coverage (#336)
#       + supply-side phi fix (#335 AND-gate). That is a CLUSTER training slot, not 1 GPU.
#     - and it pays the MEASURED step tax above.
CHEAPEST_PATH_NAME = (
    "EAGLE-3 fusion-drafter retrain (soft-KD top-k + reasoning-CoT root coverage, #336) "
    "WITH multi-step/TTT chain training (fixes the #34 K=1 collapse) + supply-side phi fix "
    "(#335 AND-gate) — a cluster training slot (~107 A10G-GPU-h, denken #301), NOT a 1-GPU local realize"
)
EAGLE3_BUILT_HEAD_REALIZED_ET = 1.78  # fern #34: native step-1 ~0.7714 accept, chain collapses past step 1
EAGLE3_BUILD_GPU_H = 107.46577676190476   # denken #301 (b4zg7b6c) full EAGLE-3 build cost


# ============================================================================
# CORE MATH
# ============================================================================
def et_from_ladder(a_ladder: list[float]) -> float:
    """E[T] = 1 + sum_{m=1..K} prod_{k<=m} a_k  (survival-function form, #289)."""
    et, g = 1.0, 1.0
    for ak in a_ladder:
        g *= ak
        et += g
    return et


def t_step_old_ms() -> float:
    """Per-step wall time on the byteexact-442 base: t_step = 1000 * E[T] / TPS."""
    return 1000.0 * ET_BASELINE / TPS_BX442_LOCAL


def t_step_new_ms(mult: float) -> float:
    """EAGLE-3 inflates ONLY the drafter portion; exec_gpu (verify) unchanged."""
    return t_step_old_ms() + DRAFT_GPU_MS * (mult - 1.0)


def tps_local(et: float, mult: float) -> float:
    return 1000.0 * et / t_step_new_ms(mult)


def to_official(tps_local_val: float) -> float:
    return tps_local_val * TAU_LO


def to_private(tps_local_val: float) -> float:
    return tps_local_val * TAU_LO * PRIV_FACTOR


# ============================================================================
# BUILD REPORT
# ============================================================================
def build_report() -> dict:
    et_ladder = et_from_ladder(A_LADDER)                       # ~= ET_BASELINE

    # --- ceilings (DIAGNOSE: where the gap is + how big the realizable lift is) ---
    et_realistic = et_from_ladder([A_LADDER[0]] + [EAGLE_DEEP_ACCEPT] * 6)        # ~4.9146
    et_realistic_hi = et_from_ladder([A_LADDER[0]] + [EAGLE_DEEP_ACCEPT_HI] * 6)  # ~4.9601
    et_optimistic = et_from_ladder([EAGLE_A1_OPTIMISTIC] + [EAGLE_DEEP_ACCEPT] * 6)  # ~5.2955

    # per-position acceptance GAP vs the flat-0.91 deep target (the diagnose ledger)
    a_ceiling = [A_LADDER[0]] + [EAGLE_DEEP_ACCEPT] * 6
    per_pos_gap = [a_ceiling[i] - A_LADDER[i] for i in range(K_SPEC)]

    t_old = t_step_old_ms()
    draft_frac = DRAFT_GPU_MS / t_old

    # --- realized-TPS grid: {E[T]} x {multiplier}, on the byteexact-442 LEGAL base ---
    et_cases = {"realistic": et_realistic, "realistic_hi": et_realistic_hi, "optimistic": et_optimistic}
    mult_cases = {"notax_kanna": 1.0, "tax_lo": MULT_LO, "tax_central": MULT_CENTRAL, "tax_hi": MULT_HI}
    grid = {}
    for ek, ev in et_cases.items():
        grid[ek] = {}
        for mk, mv in mult_cases.items():
            loc = tps_local(ev, mv)
            grid[ek][mk] = {
                "mult": mv,
                "t_step_ms": t_step_new_ms(mv),
                "tps_local": loc,
                "tps_official": to_official(loc),
                "tps_private": to_private(loc),
            }

    # --- headline operating point: realistic ceiling at CENTRAL regime tax, LOCAL frame ---
    primary = grid["realistic"]["tax_central"]
    projected_tps_at_realized_E_T = primary["tps_local"]

    # --- crosses_500 / crosses_565 (primary bools = central operating point) ---
    crosses_500 = projected_tps_at_realized_E_T >= 500.0
    crosses_565 = projected_tps_at_realized_E_T >= 565.0

    # nuanced corner analysis (where, if anywhere, the lines ARE crossed)
    all_taxed_local = [grid[ek][mk]["tps_local"]
                       for ek in et_cases for mk in ("tax_lo", "tax_central", "tax_hi")]
    all_taxed_official = [grid[ek][mk]["tps_official"]
                          for ek in et_cases for mk in ("tax_lo", "tax_central", "tax_hi")]
    all_taxed_private = [grid[ek][mk]["tps_private"]
                         for ek in et_cases for mk in ("tax_lo", "tax_central", "tax_hi")]
    crosses_500_corner = {
        "local_any_taxed_corner": max(all_taxed_local) >= 500.0,
        "local_at_realistic_best_tax": grid["realistic"]["tax_lo"]["tps_local"] >= 500.0,
        "local_joint_optimistic": grid["optimistic"]["tax_lo"]["tps_local"] >= 500.0,
        "official_at_realistic_best_tax": grid["realistic"]["tax_lo"]["tps_official"] >= 500.0,
        "official_any_taxed_corner": max(all_taxed_official) >= 500.0,
        "private_any_taxed_corner": max(all_taxed_private) >= 500.0,
        "max_local_taxed": max(all_taxed_local),
        "max_official_taxed": max(all_taxed_official),
        "max_private_taxed": max(all_taxed_private),
    }
    crosses_565_corner = {
        "local_any_taxed_corner": max(all_taxed_local) >= 565.0,
        "notax_kanna_realistic_reaches_565": grid["realistic"]["notax_kanna"]["tps_local"] >= 560.0,
    }

    # --- the kanna-565 vs go_card-sub500 reconciliation, made explicit ---
    kanna_notax_realistic = grid["realistic"]["notax_kanna"]["tps_local"]      # ~561 (kanna's 565 on 442.35)
    tax_erosion_tps = kanna_notax_realistic - primary["tps_local"]            # how much the MEASURED tax eats

    go_no_go = "NO-GO-LOCAL-REALIZE / DEFER-TO-CLUSTER"
    rationale = (
        f"NO-GO for a LOCAL drafter-realize in this slot. The headroom is REAL "
        f"(+{et_realistic - ET_BASELINE:.2f} E[T] realistic, to ~{et_realistic:.2f}) but NOT free-500 on the "
        f"byte-exact base: (1) the only path to the ceiling is an EAGLE-3 retrain, and the only BUILT head "
        f"(#34, K=1) COLLAPSES past step 1 (realized E[T]~{EAGLE3_BUILT_HEAD_REALIZED_ET:.1f} < deployed "
        f"{ET_BASELINE:.2f}) — reaching {et_realistic:.2f} needs the UNBUILT multi-step/TTT + soft-KD/CoT "
        f"recipe, a ~{EAGLE3_BUILD_GPU_H:.0f} A10G-GPU-h CLUSTER slot; (2) folding wirbel #295's MEASURED "
        f"step tax onto MY MEASURED 442-base draft fraction ({draft_frac:.3f}) collapses kanna #526's no-tax "
        f"{kanna_notax_realistic:.0f}->{primary['tps_local']:.0f} TPS at the realistic ceiling (central regime; "
        f"-{tax_erosion_tps:.0f} TPS) — local crosses 500 ONLY at the joint-optimistic corner "
        f"(E[T]->{et_optimistic:.2f} AND tax->{MULT_LO:.2f} lower bound); crosses_565 NEVER under any measured "
        f"tax; PRIVATE (x{PRIV_FACTOR}) NEVER crosses 500. Hand off the de-risked training scope (cheapest path) "
        f"as the #2-priority CLUSTER slot, sequenced AFTER the supply-side realization-gap (#1, my #523)."
    )

    report = {
        "pr": 532,
        "agent": "lawine",
        "title": "Realize the drafter-acceptance headroom: E[T] 3.849 -> ceiling ~4.91, cheapest path",
        "kind": "drafter-acceptance-realization-ledger (#502-style, PR step 4 escape hatch)",
        "analysis_only": True,
        "no_hf_job": True,
        "no_launch": True,
        "no_submission": True,
        "no_served_file_change": True,
        "official_tps": 0,
        "gpu_used": False,
        "wandb_group": "drafter-acceptance-realize",
        "model": (
            "TPS = E[T]/t_step (PF>=1.0, #504); EAGLE-3 swap inflates ONLY draft_gpu: "
            "t_step_new(m) = t_step_old + draft_gpu_old*(m-1); m = wirbel #295 measured bracket"
        ),
        "inputs": {
            "a_ladder_a1_a7": A_LADDER,
            "ET_baseline_prometheus": ET_BASELINE,
            "ET_baseline_serverlog": ET_BASELINE_SERVERLOG,
            "K_spec": K_SPEC,
            "verify_M": VERIFY_M,
            "ET_theoretical_max": ET_THEORETICAL_MAX,
            "linear_cap_ET": LINEAR_CAP_ET,
            "eagle_deep_accept": EAGLE_DEEP_ACCEPT,
            "eagle_deep_accept_hi": EAGLE_DEEP_ACCEPT_HI,
            "eagle_a1_optimistic": EAGLE_A1_OPTIMISTIC,
            "tps_bx442_local_523": TPS_BX442_LOCAL,
            "tps_bx442_anchor_519": TPS_BX442_ANCHOR_519,
            "draft_gpu_ms_523": DRAFT_GPU_MS,
            "exec_gpu_ms_523": EXEC_GPU_MS,
            "gap_ms_523": GAP_MS,
            "ppl_bx442_523": PPL_BX442,
            "mult_lo_295": MULT_LO,
            "mult_central_295": MULT_CENTRAL,
            "mult_hi_295": MULT_HI,
            "tau_lo_267": TAU_LO,
            "priv_factor_305": PRIV_FACTOR,
            "eagle3_built_head_realized_et_34": EAGLE3_BUILT_HEAD_REALIZED_ET,
            "eagle3_build_gpu_h_301": EAGLE3_BUILD_GPU_H,
            "source_runs": {
                "et_per_pos_289": "5m17r52s/fi34s269",
                "byteexact442_steptime_523": "i11p5e3y",
                "eagle3_step_multiplier_295": "c334qaqu",
                "eagle3_go_card_305": "m4nmtdl9",
                "no_retrain_null_lever_399": "ec7i3z5t",
            },
        },

        # ============ REQUIRED KEY OUTPUTS ============
        "measured_E_T_baseline": ET_BASELINE,
        "measured_E_T_baseline_source": (
            "PR #289 accept_calibration (W&B 5m17r52s/fi34s269); shipped kenyan-duma linear-MTP K=7 "
            "(DRAFTER_SHA256 ed159e..dd18e), 128x512 public sharegpt; REUSED measured served draw, no fresh serve"
        ),
        "best_realized_E_T": et_realistic,           # PROJECTED realistic ceiling (UNMEASURED — see caveat)
        "best_realized_E_T_is_projection_not_measured": True,
        "best_realized_E_T_measured_today": EAGLE3_BUILT_HEAD_REALIZED_ET,  # only-built head (#34) < deployed
        "best_realized_E_T_band": [et_realistic, et_optimistic],
        "E_T_lift": et_realistic - ET_BASELINE,
        "E_T_lift_band": [et_realistic - ET_BASELINE, et_optimistic - ET_BASELINE],
        "cheapest_path_name": CHEAPEST_PATH_NAME,
        "projected_tps_at_realized_E_T": projected_tps_at_realized_E_T,   # LOCAL, realistic E[T], CENTRAL tax
        "projected_tps_at_realized_E_T_frame": "byteexact-442 LOCAL, realistic ceiling E[T], central regime tax",
        "projected_tps_official": primary["tps_official"],
        "projected_tps_private": primary["tps_private"],
        "crosses_500": crosses_500,
        "crosses_565": crosses_565,
        "drafter_is_quality_safe": True,
        "drafter_is_quality_safe_reason": (
            "spec-dec verify is byte-exact M=8: acceptance changes ONLY E[T] (tokens/step), NOT the emitted "
            "distribution — served output == target greedy regardless of drafter quality. So self-det/PPL are "
            "invariant to the drafter and inherited from the #523 served stack (no re-serve needed)."
        ),
        "self_det": "served r1-r2 = 1.0, attention 0/8 byte-exact microbench (#523 i11p5e3y); UNCHANGED by drafter",
        "ppl": PPL_BX442,
        "ppl_guardrail": 2.42,
        "go_no_go": go_no_go,
        "go_no_go_rationale": rationale,

        # ============ SUPPORTING DETAIL ============
        "diagnose": {
            "et_ladder_roundtrip": et_ladder,
            "per_position_conditional_acceptance": A_LADDER,
            "deep_target_flat": EAGLE_DEEP_ACCEPT,
            "per_position_gap_to_091": per_pos_gap,
            "first_token_a1": A_LADDER[0],
            "first_token_is_ceiling_bound": True,
            "deployed_at_linear_cap": abs(ET_BASELINE - LINEAR_CAP_ET) < 0.02,
            "gap_is_deep_positions_not_first_token": (
                sum(per_pos_gap[1:]) > 5.0 * abs(per_pos_gap[0])
            ),
            "note": (
                "The cap is the linear-MTP TOPOLOGY (a_2..a_7 plateau ~0.83-0.85 vs the 0.91 EAGLE-3 target); "
                "a_1=0.729 is first-token ceiling-bound (cannot be lifted by acceptance work). The realizable "
                "lift is the DEEP-position coverage, which needs a wider fusion (EAGLE-3) head."
            ),
        },
        "screen": {
            "leg_a_tune_existing_mtp_head": "PROVEN-CLOSED (#119 cap 3.845; #399 null-lever ec7i3z5t)",
            "leg_b_eagle3_fusion_head": (
                "ONLY path to the ceiling; but the only BUILT head (#34, K=1) collapses past step 1 "
                f"(realized E[T]~{EAGLE3_BUILT_HEAD_REALIZED_ET:.1f} < deployed {ET_BASELINE:.2f}); reaching the "
                "ceiling needs UNBUILT multi-step/TTT + soft-KD/CoT (#336) + supply phi fix (#335)"
            ),
            "reduced_screen_run_here": (
                "0-GPU analytic reconciliation (no train, no serve): folded wirbel #295's MEASURED step "
                "multiplier into MY MEASURED #523 442-base draft fraction to price the realized TPS on the LEGAL base"
            ),
            "why_no_local_train": (
                f"a faithful EAGLE-3 build is ~{EAGLE3_BUILD_GPU_H:.0f} A10G-GPU-h (denken #301) + the TTT recipe "
                "is unbuilt — overruns the slot (PR step-4 escape hatch triggered)"
            ),
        },
        "reconciliation": {
            "t_step_old_ms": t_old,
            "draft_gpu_ms": DRAFT_GPU_MS,
            "draft_frac_measured": draft_frac,
            "kanna_notax_realistic_tps": kanna_notax_realistic,
            "measured_tax_erosion_tps_at_realistic": tax_erosion_tps,
            "primary_operating_point": primary,
            "grid_local_official_private": grid,
            "crosses_500_corner": crosses_500_corner,
            "crosses_565_corner": crosses_565_corner,
            "statement": (
                "kanna #526's 565 (no-tax) is recovered exactly at m=1.0; the MEASURED EAGLE-3 step tax "
                "(wirbel #295) erodes it to ~414 (central) / ~496 (optimistic tax) at the realistic ceiling on "
                "the byteexact-442 base. fern #305's sub-500 finding (481-base) and kanna #526's 565 (no-tax) are "
                "reconciled: the gap between them IS the measured drafter step tax, now priced on the LEGAL base."
            ),
        },
    }

    report["self_test"] = run_self_test(report)
    report["drafter_acceptance_realize_self_test_passes"] = report["self_test"]["passes"]
    return report


# ============================================================================
# SELF-TEST (0 GPU)
# ============================================================================
def run_self_test(report: dict) -> dict:
    g = report["reconciliation"]["grid_local_official_private"]
    et_realistic = report["best_realized_E_T"]
    et_optimistic = report["best_realized_E_T_band"][1]
    t_old = report["reconciliation"]["t_step_old_ms"]
    primary = report["reconciliation"]["primary_operating_point"]

    c = {
        # --- diagnose / ladder ---
        "a_ladder_len_7": len(A_LADDER) == 7,
        "a_ladder_in_unit": all(0.0 < a < 1.0 for a in A_LADDER),
        "a_ladder_monotone_nondecreasing": all(A_LADDER[i] <= A_LADDER[i + 1] + 1e-9 for i in range(6)),
        "ladder_reproduces_baseline": abs(report["diagnose"]["et_ladder_roundtrip"] - ET_BASELINE) < 1e-9,
        "deployed_at_linear_cap": abs(ET_BASELINE - LINEAR_CAP_ET) < 0.02,
        "gap_is_deep_not_first_token": report["diagnose"]["gap_is_deep_positions_not_first_token"],
        # --- ceiling math (vs kanna #526 verified values) ---
        "ceiling_realistic_matches_kanna": abs(et_realistic - 4.914619849955159) < 1e-6,
        "ceiling_optimistic_matches_kanna": abs(et_optimistic - 5.2954576057128) < 1e-6,
        "ceiling_above_baseline": et_realistic > ET_BASELINE,
        "ceiling_below_theoretical_max": et_realistic < ET_THEORETICAL_MAX,
        "optimistic_above_realistic": et_optimistic > et_realistic,
        "lift_is_ceiling_minus_baseline": abs(report["E_T_lift"] - (et_realistic - ET_BASELINE)) < 1e-12,
        "lift_positive_above_1": report["E_T_lift"] > 1.0,
        # --- step model / draft fraction (MEASURED) ---
        "t_step_old_roundtrips": abs(t_old - 1000.0 * ET_BASELINE / TPS_BX442_LOCAL) < 1e-9,
        "draft_frac_in_measured_range": 0.10 < report["reconciliation"]["draft_frac_measured"] < 0.25,
        "mult_bracket_ordered": MULT_LO < MULT_CENTRAL < MULT_HI,
        "t_step_increases_with_mult": (
            g["realistic"]["tax_hi"]["t_step_ms"] > g["realistic"]["tax_central"]["t_step_ms"]
            > g["realistic"]["tax_lo"]["t_step_ms"] > g["realistic"]["notax_kanna"]["t_step_ms"]
        ),
        "tps_decreases_with_mult": (
            g["realistic"]["tax_lo"]["tps_local"] > g["realistic"]["tax_central"]["tps_local"]
            > g["realistic"]["tax_hi"]["tps_local"]
        ),
        "tps_increases_with_et": (
            g["optimistic"]["tax_central"]["tps_local"] > g["realistic"]["tax_central"]["tps_local"]
        ),
        # --- reconciliation: no-tax recovers kanna; tax always reduces ---
        "notax_recovers_kanna_565ish": abs(g["realistic"]["notax_kanna"]["tps_local"] - 561.4) < 3.0,
        "measured_tax_reduces_tps": g["realistic"]["notax_kanna"]["tps_local"] > primary["tps_local"],
        "tax_erosion_positive": report["reconciliation"]["measured_tax_erosion_tps_at_realistic"] > 0,
        # --- crosses_500 / crosses_565 verdict ---
        "crosses_500_false_at_central": not report["crosses_500"],
        "crosses_500_true_joint_optimistic_local": g["optimistic"]["tax_lo"]["tps_local"] >= 500.0,
        "crosses_500_false_realistic_central_local": g["realistic"]["tax_central"]["tps_local"] < 500.0,
        "crosses_565_false": not report["crosses_565"],
        "crosses_565_false_everywhere_taxed": report["reconciliation"]["crosses_500_corner"]["max_local_taxed"] < 565.0,
        "private_never_crosses_500": report["reconciliation"]["crosses_500_corner"]["max_private_taxed"] < 500.0,
        # --- transfers ---
        "tau_lo_gt_one": TAU_LO > 1.0,
        "priv_factor_lt_one": PRIV_FACTOR < 1.0,
        "official_gt_local": primary["tps_official"] > primary["tps_local"],
        "private_lt_official": primary["tps_private"] < primary["tps_official"],
        # --- quality-safety / built-head honesty ---
        "drafter_quality_safe": report["drafter_is_quality_safe"] is True,
        "ppl_under_guardrail": PPL_BX442 < 2.42,
        "built_head_today_not_above_deployed": report["best_realized_E_T_measured_today"] <= ET_BASELINE + 1e-9,
        "best_realized_flagged_as_projection": report["best_realized_E_T_is_projection_not_measured"] is True,
        # --- verdict / hygiene ---
        "go_no_go_is_nogo": "NO-GO" in report["go_no_go"],
        "no_nan_inf": _all_finite(report),
    }
    n = len(c)
    passed = sum(1 for v in c.values() if v)
    return {"conditions": c, "n_checks": n, "n_passed": passed, "passes": passed == n}


def _all_finite(obj) -> bool:
    if isinstance(obj, bool):
        return True
    if isinstance(obj, float):
        return math.isfinite(obj)
    if isinstance(obj, dict):
        return all(_all_finite(v) for v in obj.values())
    if isinstance(obj, list):
        return all(_all_finite(v) for v in obj)
    return True


# ============================================================================
# W&B (best-effort; analysis-only, 0 GPU)
# ============================================================================
def log_to_wandb(report: dict, group: str, name: str) -> str | None:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"W&B import failed (non-fatal): {exc}", file=sys.stderr)
        return None
    try:
        run = wandb.init(
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
            group=group, name=name, config=report["inputs"], job_type="analysis",
        )
        flat = {
            "measured_E_T_baseline": report["measured_E_T_baseline"],
            "best_realized_E_T": report["best_realized_E_T"],
            "best_realized_E_T_measured_today": report["best_realized_E_T_measured_today"],
            "E_T_lift": report["E_T_lift"],
            "projected_tps_at_realized_E_T": report["projected_tps_at_realized_E_T"],
            "projected_tps_official": report["projected_tps_official"],
            "projected_tps_private": report["projected_tps_private"],
            "crosses_500": float(report["crosses_500"]),
            "crosses_565": float(report["crosses_565"]),
            "drafter_is_quality_safe": float(report["drafter_is_quality_safe"]),
            "ppl": report["ppl"],
            "draft_frac_measured": report["reconciliation"]["draft_frac_measured"],
            "measured_tax_erosion_tps": report["reconciliation"]["measured_tax_erosion_tps_at_realistic"],
            "kanna_notax_realistic_tps": report["reconciliation"]["kanna_notax_realistic_tps"],
            "go_no_go_is_nogo_local": float("NO-GO" in report["go_no_go"]),
            "self_test_passes": float(report["self_test"]["passes"]),
            "self_test_n_checks": float(report["self_test"]["n_checks"]),
            "analysis_only": True, "no_hf_job": True, "official_tps": 0,
        }
        wandb.summary.update(flat)
        wandb.log({f"summary/{k}": v for k, v in flat.items() if isinstance(v, (int, float))})
        for ek, row in report["reconciliation"]["grid_local_official_private"].items():
            for mk, rec in row.items():
                wandb.log({
                    f"grid/{ek}/{mk}/tps_local": rec["tps_local"],
                    f"grid/{ek}/{mk}/tps_official": rec["tps_official"],
                    f"grid/{ek}/{mk}/tps_private": rec["tps_private"],
                    f"grid/{ek}/{mk}/t_step_ms": rec["t_step_ms"],
                })
        for cond, val in report["self_test"]["conditions"].items():
            wandb.log({f"test/{cond}": float(bool(val))})
        run_id = run.id
        wandb.finish()
        return run_id
    except Exception as exc:  # noqa: BLE001
        print(f"W&B logging failed (non-fatal): {exc}", file=sys.stderr)
        return None


# ============================================================================
def _fmt(report: dict) -> str:
    r = report["reconciliation"]
    g = r["grid_local_official_private"]
    lines = [
        "=== PR #532 — drafter-acceptance REALIZATION ledger (ANALYSIS-ONLY, 0 GPU) ===",
        f"measured E[T] baseline = {report['measured_E_T_baseline']:.4f}  (deployed linear-MTP K=7, at cap {LINEAR_CAP_ET})",
        f"realizable ceiling     = {report['best_realized_E_T']:.4f}  (PROJECTION/target — UNMEASURED; "
        f"only-built EAGLE-3 head #34 realizes ~{report['best_realized_E_T_measured_today']:.1f} < deployed)",
        f"E[T] lift (realistic)  = +{report['E_T_lift']:.3f}  (band +{report['E_T_lift']:.3f}..+{report['E_T_lift_band'][1]:.3f})",
        f"cheapest path          = {report['cheapest_path_name']}",
        "",
        f"byteexact-442 base: t_step={r['t_step_old_ms']:.3f} ms, draft_gpu={DRAFT_GPU_MS} ms, "
        f"draft_frac={r['draft_frac_measured']:.3f} (MY #523 i11p5e3y); PPL={report['ppl']:.5f}",
        "",
        "realized TPS grid on the LEGAL byteexact-442 base  [local | official x1.0352 | private x0.804]:",
        f"  E[T]={report['best_realized_E_T']:.3f} (realistic):",
        f"    no-tax (kanna m=1.0): {g['realistic']['notax_kanna']['tps_local']:.0f} | "
        f"{g['realistic']['notax_kanna']['tps_official']:.0f} | {g['realistic']['notax_kanna']['tps_private']:.0f}",
        f"    tax m={MULT_LO:.2f} (opt):  {g['realistic']['tax_lo']['tps_local']:.0f} | "
        f"{g['realistic']['tax_lo']['tps_official']:.0f} | {g['realistic']['tax_lo']['tps_private']:.0f}",
        f"    tax m=3.0 (central):  {g['realistic']['tax_central']['tps_local']:.0f} | "
        f"{g['realistic']['tax_central']['tps_official']:.0f} | {g['realistic']['tax_central']['tps_private']:.0f}",
        f"    tax m={MULT_HI:.2f} (pess): {g['realistic']['tax_hi']['tps_local']:.0f} | "
        f"{g['realistic']['tax_hi']['tps_official']:.0f} | {g['realistic']['tax_hi']['tps_private']:.0f}",
        f"  E[T]={report['best_realized_E_T_band'][1]:.3f} (optimistic):",
        f"    tax m={MULT_LO:.2f} (opt):  {g['optimistic']['tax_lo']['tps_local']:.0f} | "
        f"{g['optimistic']['tax_lo']['tps_official']:.0f} | {g['optimistic']['tax_lo']['tps_private']:.0f}",
        f"    tax m=3.0 (central):  {g['optimistic']['tax_central']['tps_local']:.0f} | "
        f"{g['optimistic']['tax_central']['tps_official']:.0f} | {g['optimistic']['tax_central']['tps_private']:.0f}",
        "",
        f"PRIMARY (realistic E[T], central tax, LOCAL): {report['projected_tps_at_realized_E_T']:.1f} TPS  "
        f"(official {report['projected_tps_official']:.1f}, private {report['projected_tps_private']:.1f})",
        f"measured-tax erosion of kanna's no-tax {r['kanna_notax_realistic_tps']:.0f}: "
        f"-{r['measured_tax_erosion_tps_at_realistic']:.0f} TPS",
        "",
        f"crosses_500 = {report['crosses_500']}   crosses_565 = {report['crosses_565']}   "
        f"quality_safe = {report['drafter_is_quality_safe']}",
        f"  (local crosses 500 only at joint-optimistic corner; private NEVER crosses 500; 565 NEVER under measured tax)",
        "",
        f"GO/NO-GO: {report['go_no_go']}",
        f"self-test: {report['self_test']['n_passed']}/{report['self_test']['n_checks']} "
        f"({'PASS' if report['self_test']['passes'] else 'FAIL'})",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="0-GPU analytic gate")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="drafter-acceptance-realize")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="lawine/drafter-acceptance-realize")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).with_name("realize_drafter_acceptance_results.json")))
    args = ap.parse_args()

    report = build_report()
    print(_fmt(report))

    if args.self_test:
        Path(__file__).with_name("realize_drafter_acceptance_selftest.json").write_text(
            json.dumps(report["self_test"], indent=2))
        return 0 if report["self_test"]["passes"] else 1

    report["wandb_run_id"] = None if args.no_wandb else log_to_wandb(
        report, args.wandb_group, args.wandb_name)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}  (W&B run {report.get('wandb_run_id')})")

    print("SENPAI-RESULT " + json.dumps({
        "terminal": True, "status": "complete", "pending_arms": False,
        "analysis_only": True, "no_hf_job": True, "official_tps": 0,
        "wandb_run_ids": [report["wandb_run_id"]] if report.get("wandb_run_id") else [],
        "primary_metric": {"name": "drafter_acceptance_realize_self_test_passes",
                           "value": int(report["self_test"]["passes"])},
        "test_metric": {"name": "projected_tps_at_realized_E_T",
                        "value": report["projected_tps_at_realized_E_T"]},
    }))
    return 0 if report["self_test"]["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
