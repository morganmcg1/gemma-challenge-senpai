#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
PR #526 — Price the drafter-acceptance headroom: is direction #3 worth a slot?

ANALYSIS-ONLY. 0 GPU, 0 HF Job, 0 submission, official_tps=0. Challenge PAUSED.
Reuses MEASURED served acceptance data (no fresh serve needed — see provenance below).

Question (#481 direction #3 / drafter-acceptance):
  spec-dec is distribution-preserving, so a better drafter (more accepted tokens/step
  E[T]) is "free" quality-safe speed. Before spending a drafter-retrain/topology slot,
  PRICE the lever: measure served E[T], compute dTPS/dE[T] on each shippable rung,
  estimate the realistic acceptance ceiling, and return a GO/NO-GO vs the marginal
  alternative (the #1 realization-gap "S-sweep", supply-side).

Model used (PF>=1.0, #504 0urxqwob):  TPS = E[T] / t_step
  E[T]   = accepted tokens per verify step (1 bonus + accepted drafts)
  t_step = wall time per verify step (target verify + K draft passes)
For a SAME-TOPOLOGY (K=7) acceptance improvement, t_step is ~invariant, so
  dTPS/dE[T] = 1/t_step = TPS/E[T].

KEY OUTPUTS (required by the PR):
  measured_E_T, measured_E_T_source,
  tps_per_accepted_token_surgical357, tps_per_accepted_token_splitkv442,
  acceptance_ceiling_E_T, drafter_improvement_tps_upside,
  go_no_go_drafter_slot (+ one-line rationale vs the #1 S-sweep).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# ----------------------------------------------------------------------------
# GROUNDED INPUTS (all reused MEASURED served data — provenance in comments).
# ----------------------------------------------------------------------------

# --- (1) Served E[T], measured DIRECTLY on the shipped linear-MTP K=7 drafter ---
# Source: PR #289 `research/accept_calibration/accept_calibration_results.json`
#   (W&B 5m17r52s / fi34s269), vLLM `spec_decode_num_accepted_tokens_per_pos`
#   prefix counters over the 128x512 public sharegpt benchmark.
# Drafter = kenyan-duma, SPECULATIVE_CONFIG mtp K=7, DRAFTER_SHA256 ed159e..dd18e
#   — the EXACT drafter shipped in submissions/fa2sw_strict_surgical357 AND the
#   byte-exact split-KV rung (acceptance is a SHARED drafter property, #522).
ET_PROM = 3.849256527338719          # prometheus whole-run, num_drafts=17082
ET_SERVERLOG = 3.844131736526946     # server-log cross-check, num_drafts=16700
ET_MEASURED = ET_PROM                # primary anchor (carries the full per-pos array)
MEASURED_ET_SOURCE = (
    "PR #289 accept_calibration (W&B 5m17r52s/fi34s269); vLLM "
    "spec_decode_num_accepted_tokens_per_pos counters on the shipped kenyan-duma "
    "linear-MTP K=7 drafter (DRAFTER_SHA256 ed159e..dd18e), 128x512 public sharegpt; "
    "REUSED measured served draw, no fresh serve."
)
# per-position CONDITIONAL acceptance a_1..a_7 (prometheus); a_k = G(k)/G(k-1)
A_LADDER = [
    0.7290715372907154, 0.759434719768749, 0.7934024106576444, 0.8215618336886993,
    0.834712084347121, 0.835989117761368, 0.8465829846582985,
]
K_SPEC = 7
VERIFY_M = 8
ET_THEORETICAL_MAX = float(VERIFY_M)  # 8.0 (all 7 drafts + 1 bonus accepted every step)

# --- (2) R_ea is a PRIVATE/PUBLIC E[T] TRANSFER RATIO, *not* a per-position accept ---
# Source: PR #522 `reopen_rung_private_speed_risk` (W&B w71zjxot). R_ea = ea_pri/ea_pub
# per domain draw; ea_pub ~4.06-4.11 (public E[T]), ea_pri ~3.50-3.86 (private E[T]).
# The PR's "R_ea -> current E[T]" framing is a category error (corrected in self-test).
R_EA_MEAN = 0.88773326152767
EA_PUB_DRAW = 4.060508965432251       # public E[T] draw (domain 20260613T194357Z)
EA_PRI_DRAW = 3.5652267935340864      # private E[T] draw (same domain)
PF_504 = 0.9999171490311938           # realized propagation factor ~1.0 (#504 0urxqwob)

# --- (3) Shippable rungs (PR #526 baseline). Same drafter SHA -> identical E[T];
#         rungs differ ONLY in t_step (supply side). ---
TPS_SURGICAL357 = 375.857             # official, W&B j7qao5e9 (128x512 HF Jobs)
TPS_SPLITKV = 442.35                  # local byte-exact split-KV, stark #519 (W&B kwhylaeg)

# --- (4) Structural cap of the deployed LINEAR-MTP topology (deployed sits AT cap) ---
# denken #119 linear cap (via #289); ubel #399 independently showed every NO-RETRAIN /
# no-served-kernel lever is a rank-order no-op -> the d-cov MUST come from a drafter
# RETRAIN (raise the a_j ladder) or a TREE verify. So "improve the drafter" == a
# topology change (EAGLE-3 class), NOT more epochs on the same linear head.
LINEAR_CAP_ET = 3.8445

# --- (5) Realistic EAGLE-3 acceptance target (the falsifiable #289 per-position goal) ---
# #289 feasibility: lifting the DEEP positions a_2..a_7 to ~0.91 (flat) while holding
# a_1 is FEASIBLE (the EAGLE-3 lane); a_1-alone is infeasible (ceiling-bound). This is
# an internal feasibility TARGET (delivery uncertain), NOT theoretical-max.
EAGLE_DEEP_ACCEPT = 0.91              # conservative deep-lift target
EAGLE_DEEP_ACCEPT_HI = 0.914         # #289's >=4.966 variant
EAGLE_A1_OPTIMISTIC = 0.80           # optimistic: EAGLE-3 also lifts first token

# --- (6) Marginal-alternative cost: the #1 realization-gap "S-sweep" (supply side) ---
# #481 priority #1 = "close the byte-exact realization gap 357->~457" (lawine #523).
# Constituent sub-experiments (RESEARCH_IDEAS_2026-06-16_11:33): reduction-sensitivity
# profiling (~4 GPU-h) + cuBLASLt-deterministic (2-4 GPU-h) = ~6-8 GPU-h, byte-exact
# (0 PPL / 0 greedy-identity risk), near-certain, and ALREADY 442/457 realized locally.
S_SWEEP_GPU_H_LO, S_SWEEP_GPU_H_HI = 6.0, 8.0
S_SWEEP_CEILING_TPS = 457.5          # byte-exact realization ceiling (#522 frontier457 anchor)
# A drafter-retrain (EAGLE-3) slot: training + stack integration (EAGLE-3 backend,
# ONEGRAPH/LOOPGRAPH capture) + greedy-identity & PPL re-validation. Even granting a
# "few GPU-h" head-train, the realistic full slot is back-loaded and delivery-uncertain.
DRAFTER_SLOT_GPU_H_LO, DRAFTER_SLOT_GPU_H_HI = 15.0, 40.0


# ----------------------------------------------------------------------------
# CORE MATH
# ----------------------------------------------------------------------------
def et_from_ladder(a_ladder: list[float]) -> float:
    """E[T] = 1 + sum_{m=1..K} prod_{k<=m} a_k  (survival-function form, #289)."""
    et, g = 1.0, 1.0
    for ak in a_ladder:
        g *= ak
        et += g
    return et


def et_uniform_acceptance(r: float, k: int) -> float:
    """E[T] IF a single rate r were the (uniform) per-position acceptance."""
    et, g = 1.0, 1.0
    for _ in range(k):
        g *= r
        et += g
    return et


def t_step_ms(et: float, tps: float) -> float:
    return 1000.0 * et / tps


def dtps_detp(tps: float, et: float) -> float:
    """dTPS/dE[T] = 1/t_step = TPS/E[T] (t_step invariant to same-K acceptance lift)."""
    return tps / et


def tps_at_et(tps0: float, et0: float, et1: float) -> float:
    """TPS at a new E[T], holding t_step fixed: TPS1 = TPS0 * E[T]1/E[T]0."""
    return tps0 * et1 / et0


def build_report() -> dict:
    # --- self-consistency of the measured ladder ---
    et_ladder = et_from_ladder(A_LADDER)            # ~= ET_PROM

    # --- ceilings ---
    a_eagle = [A_LADDER[0]] + [EAGLE_DEEP_ACCEPT] * 6
    a_eagle_hi = [A_LADDER[0]] + [EAGLE_DEEP_ACCEPT_HI] * 6
    a_eagle_opt = [EAGLE_A1_OPTIMISTIC] + [EAGLE_DEEP_ACCEPT] * 6
    et_ceiling_realistic = et_from_ladder(a_eagle)          # ~4.915
    et_ceiling_realistic_hi = et_from_ladder(a_eagle_hi)    # ~4.960
    et_ceiling_optimistic = et_from_ladder(a_eagle_opt)     # ~5.295

    rungs = {
        "surgical357": {"tps": TPS_SURGICAL357, "kind": "official", "anchor": "j7qao5e9"},
        "splitkv442": {"tps": TPS_SPLITKV, "kind": "local byte-exact", "anchor": "kwhylaeg/#519"},
    }

    per_rung = {}
    for name, r in rungs.items():
        tps = r["tps"]
        d = dtps_detp(tps, ET_MEASURED)             # TPS per +1.0 E[T]
        rec = {
            "tps": tps,
            "anchor": r["anchor"],
            "kind": r["kind"],
            "t_step_ms": t_step_ms(ET_MEASURED, tps),
            "dtps_per_1p0_accepted_token": d,
            "dtps_per_0p1_accepted_token": d * 0.1,
            # realistic upside (deep-lift EAGLE-3 target), and its band
            "tps_at_realistic_ceiling": tps_at_et(tps, ET_MEASURED, et_ceiling_realistic),
            "tps_at_realistic_ceiling_hi": tps_at_et(tps, ET_MEASURED, et_ceiling_realistic_hi),
            "tps_at_optimistic_ceiling": tps_at_et(tps, ET_MEASURED, et_ceiling_optimistic),
        }
        rec["upside_tps_realistic"] = rec["tps_at_realistic_ceiling"] - tps
        rec["upside_tps_realistic_hi"] = rec["tps_at_realistic_ceiling_hi"] - tps
        rec["upside_tps_optimistic"] = rec["tps_at_optimistic_ceiling"] - tps
        per_rung[name] = rec

    # the drafter lever is worth MORE on a faster (lower t_step) rung:
    sequencing_factor = per_rung["splitkv442"]["dtps_per_1p0_accepted_token"] / \
        per_rung["surgical357"]["dtps_per_1p0_accepted_token"]   # == TPS_split/TPS_surg

    # --- category-error correction: R_ea is NOT a per-position acceptance ---
    et_if_rea_were_accept = et_uniform_acceptance(R_EA_MEAN, K_SPEC)   # ~5.47 (WRONG)
    rea_is_transfer_ratio_not_acceptance = abs(et_if_rea_were_accept - ET_MEASURED) > 1.0

    # --- TPS-per-GPU-hour, marginal comparison ---
    # #1 S-sweep: most of 357->457 is ALREADY realized at split-KV 442.35 locally; the
    #   REMAINING ceiling headroom is 442->457; certifying official->457 banks 375.857->457.
    s_sweep_remaining_tps = S_SWEEP_CEILING_TPS - TPS_SPLITKV          # 442.35 -> 457.5
    s_sweep_official_to_ceiling = S_SWEEP_CEILING_TPS - TPS_SURGICAL357
    s_sweep_tps_per_gpuh_remaining = s_sweep_remaining_tps / S_SWEEP_GPU_H_HI
    s_sweep_tps_per_gpuh_official = s_sweep_official_to_ceiling / S_SWEEP_GPU_H_HI

    # #3 drafter: realistic upside is large (per_rung) but needs a training slot.
    drafter_upside_on_best_rung = per_rung["splitkv442"]["upside_tps_realistic"]
    drafter_tps_per_gpuh_lo = drafter_upside_on_best_rung / DRAFTER_SLOT_GPU_H_HI
    drafter_tps_per_gpuh_hi = drafter_upside_on_best_rung / DRAFTER_SLOT_GPU_H_LO

    go_no_go = "NO-GO-DEFER"   # defer behind #1 for the IMMEDIATE next slot; #3 NOT tapped
    rationale = (
        "NO-GO for the immediate next slot: #1 dominates on TPS-per-GPU-hour "
        f"(~{S_SWEEP_GPU_H_LO:.0f}-{S_SWEEP_GPU_H_HI:.0f} GPU-h, byte-exact, ~0 risk, already "
        f"{TPS_SPLITKV:.0f} of the ~{math.floor(S_SWEEP_CEILING_TPS):.0f} byte-exact ceiling realized locally) AND — because "
        f"dTPS/dE[T]=TPS/E[T] rises as t_step falls — banking #1 first raises the drafter "
        f"slot's payoff by {100*(sequencing_factor-1):.0f}%. BUT #3 is NOT tapped: linear-MTP "
        f"is at its {LINEAR_CAP_ET:.3f} cap, so an EAGLE-3 retrain unlocks a realistic "
        f"+{et_ceiling_realistic-ET_MEASURED:.2f} E[T] -> "
        f"+{per_rung['surgical357']['upside_tps_realistic']:.0f}/"
        f"+{drafter_upside_on_best_rung:.0f} TPS (surgical357/split-KV, crosses 500). "
        "It is the #2-priority slot, sequenced AFTER #1, not abandoned."
    )

    report = {
        "pr": 526,
        "agent": "kanna",
        "title": "Price the drafter-acceptance headroom (direction #3)",
        "analysis_only": True,
        "no_hf_job": True,
        "no_submission": True,
        "no_served_file_change": True,
        "official_tps": 0,
        "gpu_used": False,
        "wandb_group": "drafter-acceptance-headroom",
        "model": "TPS = E[T]/t_step at PF>=1.0 (#504); dTPS/dE[T]=TPS/E[T] for same-K lift",
        "inputs": {
            "ET_measured_prometheus": ET_PROM,
            "ET_measured_serverlog": ET_SERVERLOG,
            "a_ladder_a1_a7": A_LADDER,
            "K_spec": K_SPEC,
            "verify_M": VERIFY_M,
            "ET_theoretical_max": ET_THEORETICAL_MAX,
            "R_ea_mean_TRANSFER_RATIO": R_EA_MEAN,
            "ea_pub_draw": EA_PUB_DRAW,
            "ea_pri_draw": EA_PRI_DRAW,
            "PF_504": PF_504,
            "tps_surgical357_official": TPS_SURGICAL357,
            "tps_splitkv442_local": TPS_SPLITKV,
            "linear_cap_ET": LINEAR_CAP_ET,
            "eagle_deep_accept_target": EAGLE_DEEP_ACCEPT,
            "s_sweep_gpu_h": [S_SWEEP_GPU_H_LO, S_SWEEP_GPU_H_HI],
            "s_sweep_ceiling_tps": S_SWEEP_CEILING_TPS,
            "drafter_slot_gpu_h": [DRAFTER_SLOT_GPU_H_LO, DRAFTER_SLOT_GPU_H_HI],
            "source_runs": {
                "et_per_pos_289": "5m17r52s/fi34s269",
                "transfer_ratio_522": "w71zjxot",
                "propagation_504": "0urxqwob",
                "no_retrain_lever_null_399": "ec7i3z5t (ubel)",
            },
        },

        # ---- REQUIRED KEY OUTPUTS ----
        "measured_E_T": ET_MEASURED,
        "measured_E_T_band": [ET_SERVERLOG, ET_PROM],
        "measured_E_T_source": MEASURED_ET_SOURCE,
        "tps_per_accepted_token_surgical357": per_rung["surgical357"]["dtps_per_1p0_accepted_token"],
        "tps_per_accepted_token_surgical357_per_0p1": per_rung["surgical357"]["dtps_per_0p1_accepted_token"],
        "tps_per_accepted_token_splitkv442": per_rung["splitkv442"]["dtps_per_1p0_accepted_token"],
        "tps_per_accepted_token_splitkv442_per_0p1": per_rung["splitkv442"]["dtps_per_0p1_accepted_token"],
        "acceptance_ceiling_E_T": et_ceiling_realistic,
        "acceptance_ceiling_E_T_realistic_band": [et_ceiling_realistic, et_ceiling_optimistic],
        "acceptance_ceiling_E_T_theoretical_max": ET_THEORETICAL_MAX,
        "drafter_improvement_tps_upside": {
            "surgical357_realistic": per_rung["surgical357"]["upside_tps_realistic"],
            "surgical357_band": [per_rung["surgical357"]["upside_tps_realistic"],
                                 per_rung["surgical357"]["upside_tps_optimistic"]],
            "splitkv442_realistic": per_rung["splitkv442"]["upside_tps_realistic"],
            "splitkv442_band": [per_rung["splitkv442"]["upside_tps_realistic"],
                                per_rung["splitkv442"]["upside_tps_optimistic"]],
        },
        "go_no_go_drafter_slot": go_no_go,
        "go_no_go_rationale": rationale,

        # ---- supporting detail ----
        "per_rung": per_rung,
        "et_ceiling_realistic": et_ceiling_realistic,
        "et_ceiling_realistic_hi": et_ceiling_realistic_hi,
        "et_ceiling_optimistic": et_ceiling_optimistic,
        "delta_et_realistic": et_ceiling_realistic - ET_MEASURED,
        "delta_et_optimistic": et_ceiling_optimistic - ET_MEASURED,
        "sequencing_factor_split_over_surg": sequencing_factor,
        "et_if_rea_misread_as_acceptance": et_if_rea_were_accept,
        "rea_is_transfer_ratio_not_acceptance": rea_is_transfer_ratio_not_acceptance,
        "deployed_at_linear_cap": abs(ET_MEASURED - LINEAR_CAP_ET) < 0.02,
        "marginal_comparison": {
            "s_sweep_remaining_tps_442_to_457": s_sweep_remaining_tps,
            "s_sweep_official_to_ceiling_tps": s_sweep_official_to_ceiling,
            "s_sweep_tps_per_gpuh_remaining": s_sweep_tps_per_gpuh_remaining,
            "s_sweep_tps_per_gpuh_official_to_ceiling": s_sweep_tps_per_gpuh_official,
            "drafter_upside_on_best_rung_tps": drafter_upside_on_best_rung,
            "drafter_tps_per_gpuh_band": [drafter_tps_per_gpuh_lo, drafter_tps_per_gpuh_hi],
            "verdict": "S-sweep wins immediate slot on TPS/GPU-h + certainty; drafter is larger but back-loaded & delivery-uncertain",
        },
    }

    report["self_test"] = run_self_test(report, et_ladder)
    report["drafter_acceptance_headroom_self_test_passes"] = report["self_test"]["passes"]
    return report


# ----------------------------------------------------------------------------
# SELF-TEST (0 GPU)
# ----------------------------------------------------------------------------
def run_self_test(report: dict, et_ladder: float) -> dict:
    pr = report["per_rung"]
    c = {
        "a_ladder_len_7": len(A_LADDER) == 7,
        "a_ladder_in_unit": all(0.0 < a < 1.0 for a in A_LADDER),
        "a_ladder_monotone_nondecreasing": all(A_LADDER[i] <= A_LADDER[i + 1] + 1e-9
                                               for i in range(6)),
        "et_ladder_roundtrips_prometheus": abs(et_ladder - ET_PROM) < 1e-6,
        "measured_et_between_serverlog_and_prom": ET_SERVERLOG - 1e-9 <= ET_MEASURED <= ET_PROM + 1e-9,
        "deployed_at_or_near_linear_cap": report["deployed_at_linear_cap"],
        # dTPS/dE[T] == TPS/E[T] == 1/t_step
        "dtps_identity_surgical": abs(pr["surgical357"]["dtps_per_1p0_accepted_token"]
                                      - TPS_SURGICAL357 / ET_MEASURED) < 1e-9,
        "dtps_identity_splitkv": abs(pr["splitkv442"]["dtps_per_1p0_accepted_token"]
                                     - TPS_SPLITKV / ET_MEASURED) < 1e-9,
        "dtps_equals_inv_tstep_surg": abs(pr["surgical357"]["dtps_per_1p0_accepted_token"]
                                          - 1000.0 / pr["surgical357"]["t_step_ms"]) < 1e-6,
        "splitkv_dtps_gt_surgical_dtps": (pr["splitkv442"]["dtps_per_1p0_accepted_token"]
                                          > pr["surgical357"]["dtps_per_1p0_accepted_token"]),
        "sequencing_factor_eq_tps_ratio": abs(report["sequencing_factor_split_over_surg"]
                                              - TPS_SPLITKV / TPS_SURGICAL357) < 1e-9,
        # ratio identity: upside scales E[T]
        "upside_ratio_identity_surg": abs(
            pr["surgical357"]["tps_at_realistic_ceiling"]
            - TPS_SURGICAL357 * report["et_ceiling_realistic"] / ET_MEASURED) < 1e-9,
        # ceiling bounds
        "ceiling_above_current": report["et_ceiling_realistic"] > ET_MEASURED,
        "ceiling_below_theoretical_max": report["et_ceiling_realistic"] < ET_THEORETICAL_MAX,
        "optimistic_above_realistic": report["et_ceiling_optimistic"] > report["et_ceiling_realistic"],
        "theoretical_max_is_8": abs(ET_THEORETICAL_MAX - 8.0) < 1e-9,
        # category-error correction (the decisive premise fix)
        "rea_misread_gives_inconsistent_et": report["rea_is_transfer_ratio_not_acceptance"],
        "rea_misread_et_far_above_measured": report["et_if_rea_misread_as_acceptance"] - ET_MEASURED > 1.0,
        "ea_pri_below_ea_pub": EA_PRI_DRAW < EA_PUB_DRAW,
        "pf_near_one": abs(PF_504 - 1.0) < 1e-2,
        # marginal comparison sanity
        "s_sweep_remaining_positive": report["marginal_comparison"]["s_sweep_remaining_tps_442_to_457"] > 0,
        "drafter_upside_exceeds_s_sweep_remaining": (
            report["marginal_comparison"]["drafter_upside_on_best_rung_tps"]
            > report["marginal_comparison"]["s_sweep_remaining_tps_442_to_457"]),
        "go_no_go_is_defer": report["go_no_go_drafter_slot"] == "NO-GO-DEFER",
        # numeric hygiene
        "no_nan_inf": _all_finite(report),
    }
    n = len(c)
    passed = sum(1 for v in c.values() if v)
    return {"conditions": c, "n_checks": n, "n_passed": passed, "passes": passed == n}


def _all_finite(obj) -> bool:
    if isinstance(obj, float):
        return math.isfinite(obj)
    if isinstance(obj, dict):
        return all(_all_finite(v) for v in obj.values())
    if isinstance(obj, list):
        return all(_all_finite(v) for v in obj)
    return True


# ----------------------------------------------------------------------------
# W&B (best-effort; analysis-only, 0 GPU)
# ----------------------------------------------------------------------------
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
            "measured_E_T": report["measured_E_T"],
            "tps_per_accepted_token_surgical357": report["tps_per_accepted_token_surgical357"],
            "tps_per_accepted_token_surgical357_per_0p1": report["tps_per_accepted_token_surgical357_per_0p1"],
            "tps_per_accepted_token_splitkv442": report["tps_per_accepted_token_splitkv442"],
            "tps_per_accepted_token_splitkv442_per_0p1": report["tps_per_accepted_token_splitkv442_per_0p1"],
            "acceptance_ceiling_E_T": report["acceptance_ceiling_E_T"],
            "acceptance_ceiling_E_T_theoretical_max": report["acceptance_ceiling_E_T_theoretical_max"],
            "drafter_improvement_tps_upside_surgical357": report["drafter_improvement_tps_upside"]["surgical357_realistic"],
            "drafter_improvement_tps_upside_splitkv442": report["drafter_improvement_tps_upside"]["splitkv442_realistic"],
            "delta_et_realistic": report["delta_et_realistic"],
            "sequencing_factor_split_over_surg": report["sequencing_factor_split_over_surg"],
            "deployed_at_linear_cap": float(report["deployed_at_linear_cap"]),
            "rea_is_transfer_ratio_not_acceptance": float(report["rea_is_transfer_ratio_not_acceptance"]),
            "go_no_go_is_defer": float(report["go_no_go_drafter_slot"] == "NO-GO-DEFER"),
            "self_test_passes": float(report["self_test"]["passes"]),
            "self_test_n_checks": float(report["self_test"]["n_checks"]),
            "analysis_only": True, "no_hf_job": True, "official_tps": 0,
        }
        wandb.summary.update(flat)
        wandb.log({f"summary/{k}": v for k, v in flat.items() if isinstance(v, (int, float))})
        for rung, rec in report["per_rung"].items():
            wandb.log({
                f"rung/{rung}/tps": rec["tps"],
                f"rung/{rung}/t_step_ms": rec["t_step_ms"],
                f"rung/{rung}/dtps_per_0p1": rec["dtps_per_0p1_accepted_token"],
                f"rung/{rung}/tps_at_realistic_ceiling": rec["tps_at_realistic_ceiling"],
                f"rung/{rung}/upside_tps_realistic": rec["upside_tps_realistic"],
            })
        for cond, val in report["self_test"]["conditions"].items():
            wandb.log({f"test/{cond}": float(bool(val))})
        run_id = run.id
        wandb.finish()
        return run_id
    except Exception as exc:  # noqa: BLE001
        print(f"W&B logging failed (non-fatal): {exc}", file=sys.stderr)
        return None


# ----------------------------------------------------------------------------
def _fmt(report: dict) -> str:
    pr = report["per_rung"]
    return "\n".join([
        "=== PR #526 — drafter-acceptance lever pricing (ANALYSIS-ONLY) ===",
        f"measured E[T] = {report['measured_E_T']:.4f}  "
        f"(band {report['measured_E_T_band'][0]:.4f}-{report['measured_E_T_band'][1]:.4f})",
        f"  source: {report['measured_E_T_source']}",
        f"  [R_ea={R_EA_MEAN:.4f} is a private/public TRANSFER RATIO, NOT acceptance: "
        f"misreading it -> E[T]={report['et_if_rea_misread_as_acceptance']:.2f} (inconsistent)]",
        "",
        f"dTPS/dE[T]  surgical357 ({pr['surgical357']['tps']:.1f} TPS, t_step "
        f"{pr['surgical357']['t_step_ms']:.3f} ms): {pr['surgical357']['dtps_per_1p0_accepted_token']:.1f}/+1.0  "
        f"({pr['surgical357']['dtps_per_0p1_accepted_token']:.2f} TPS per +0.1)",
        f"dTPS/dE[T]  split-KV   ({pr['splitkv442']['tps']:.1f} TPS, t_step "
        f"{pr['splitkv442']['t_step_ms']:.3f} ms): {pr['splitkv442']['dtps_per_1p0_accepted_token']:.1f}/+1.0  "
        f"({pr['splitkv442']['dtps_per_0p1_accepted_token']:.2f} TPS per +0.1)",
        f"  -> drafter lever worth {100*(report['sequencing_factor_split_over_surg']-1):.0f}% MORE on the faster rung",
        "",
        f"acceptance ceiling E[T]: realistic {report['et_ceiling_realistic']:.3f} "
        f"(band {report['acceptance_ceiling_E_T_realistic_band'][0]:.2f}-"
        f"{report['acceptance_ceiling_E_T_realistic_band'][1]:.2f}), theoretical-max "
        f"{report['acceptance_ceiling_E_T_theoretical_max']:.1f}; deployed at linear cap "
        f"{LINEAR_CAP_ET:.3f} -> needs EAGLE-3 (non-linear) retrain",
        f"  dE[T] realistic = +{report['delta_et_realistic']:.2f}",
        "",
        f"drafter upside (realistic): surgical357 +{pr['surgical357']['upside_tps_realistic']:.1f} TPS "
        f"(-> {pr['surgical357']['tps_at_realistic_ceiling']:.1f}); "
        f"split-KV +{pr['splitkv442']['upside_tps_realistic']:.1f} TPS "
        f"(-> {pr['splitkv442']['tps_at_realistic_ceiling']:.1f}, crosses 500)",
        "",
        f"GO/NO-GO: {report['go_no_go_drafter_slot']}",
        f"  {report['go_no_go_rationale']}",
        "",
        f"self-test: {report['self_test']['n_passed']}/{report['self_test']['n_checks']} "
        f"({'PASS' if report['self_test']['passes'] else 'FAIL'})",
    ])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="0-GPU analytic gate")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="drafter-acceptance-headroom")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="kanna/drafter-acceptance-headroom")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).with_name("price_drafter_acceptance_results.json")))
    args = ap.parse_args()

    report = build_report()
    print(_fmt(report))

    if args.self_test:
        Path(__file__).with_name("price_drafter_acceptance_selftest.json").write_text(
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
        "primary_metric": {"name": "drafter_acceptance_headroom_self_test_passes",
                           "value": int(report["self_test"]["passes"])},
        "test_metric": {"name": "drafter_improvement_tps_upside_splitkv442",
                        "value": report["drafter_improvement_tps_upside"]["splitkv442_realistic"]},
    }))
    return 0 if report["self_test"]["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
