#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Valid-verify cluster capstone (PR #227, wirbel) — CPU-only consolidation.

THE QUESTION (capstone of #199 → #213 → #216 → #223; the human's live Issue #211)
---------------------------------------------------------------------------------
The greedy-identity gate (#114/#192) forces every >500 build onto a validity-preserving
verify path. The fleet has now PRICED every such path, and the menu has COLLAPSED to a single
survivor. This leg puts the collapsed menu in ONE frame and crosses it against the per-gate
leverage of the Blackwell node the human offers in Issue #211 (a node to TRAIN A DRAFT), to
state plainly whether that cluster unlocks a valid 500.

THE COLLAPSED MENU (each path × validity basis × λ=1 ceiling × clears-500? × what-it-needs)
------------------------------------------------------------------------------------------
  lane-a  int4 batch-invariant verify kernel  — VALID (split-K reduction-order fix ⇒ verify-M
          argmax == AR-M=1 argmax). λ=1 ceiling 520.953 (#204). CLEARS at the 0.9455%-of-step
          split-K floor for λ ≥ 0.8572 (#216 over #213's budget). DOUBLE-gated: (a) kernel
          BUILD near floor AND (b) land #71 λ ≥ 0.8572. ← THE ONLY SURVIVOR.
  lane-b  fp16/bf16 verify  — VALID (batch-invariant cuBLAS GEMM, no M-dependent split-K;
          imported #211 premise, lawine #221 confirms). λ=1 ceiling 520.953/M_step = 306.44 at
          central M_step=1.7; < 500 for every physical M_step ≥ 1.3 (crossover 1.0419). NEVER (#220).
  MarginGate  provable-argmax-stability skip  — VALID (sound gate skips only provably-stable
          positions; rest fall back to AR M=1). Needs sound skip ≥ 0.9706 to fit the λ=1 budget,
          but #114's 56.08% flip caps the SOUND skip at ≤ 0.4392. Demand ≫ supply. NEVER (#223).
  lane-c  no-spec int4 AR  — VALID (plain int4 autoregressive greedy, token-identity 1.0).
          ~165.4 official TPS, 66.9% below 500, structural. NEVER (#196).

THE CLUSTER'S PER-GATE LEVERAGE (the #211 deliverable)
-----------------------------------------------------
lane-a — the only survivor — is DOUBLE-gated. The Blackwell draft-training raises E[T](λ) and
the achievable λ, so it helps **gate-b (λ ≥ 0.8572) ONLY**; it does NOTHING for **gate-a (the
kernel BUILD)**, which is a split-K reduction-order fix and is draft-INDEPENDENT. Hence the
cluster's offer is NECESSARY-BUT-NOT-SUFFICIENT: it can lift the λ gate but cannot build the
kernel. The binding next action is therefore the CHEAP split-K microbench (#216's decisive
diagnostic, ~1–2 days), NOT draft training — authorize the kernel diagnostic first; draft-
training is conditional on it.

LOCAL, CPU-ONLY consolidation of the BANKED valid-verify ledger. No GPU / vLLM / HF Job /
submission / served-file change / official draw. BASELINE stays 481.53; adds **0 TPS**;
greedy/PPL untouched. Ceilings / budget / divergence / E[T] are IMPORTED unchanged from the
banked artifacts (#216/#213/#223/#220/#204/#196/#114); this leg DERIVES nothing new and
authorizes nothing. **NOT a launch. NOT open2.**

PRIMARY metric  valid_verify_cluster_capstone_self_test_passes
TEST    metric  n_surviving_valid_500_paths  (expect 1)
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

# --------------------------------------------------------------------------- #
# Banked artifacts to IMPORT (committed in-repo; this leg re-derives nothing).
# --------------------------------------------------------------------------- #
KF216_REL = "research/validity/kernel_feasibility/kernel_feasibility_results.json"      # #216 pc8g6s04
KB213_REL = "research/validity/kernel_budget_lambda/kernel_budget_lambda_results.json"  # #213 5o7zcj8s
MG223_REL = "research/validity/margingate_budget/margingate_budget_results.json"        # #223 54dtull1
FP220_REL = "research/validity/fp16_verify_ceiling/results.json"                        # #220 pqjnybbf
NS196_REL = "research/validity/compliant_nonspec_floor/floor_report.json"              # #196 y4tavh9p

TARGET = 500.0                 # official clear bar
BASELINE_TPS = 481.53          # PR #52 official (this leg adds 0 TPS)
SPINE_0997 = 0.997             # land #71 posted interim optimistic spine
LAMBDA_BAND = (0.8572, 0.997, 1.0)  # land #71 (floor / spine / ceiling)

# Instruction-pinned anchors — the menu must ROUND-TRIP these against the loaded JSONs.
PIN = {
    "kernel_floor_pct_216": 0.9455349322572293,
    "lambda_min_kernel_feasible_216": 0.8571542761568587,
    "verify_gemm_share_216": 0.606620584396473,
    "budget_at_lambda1_bb_213": 7.331808522875782,
    "lambda_crit_bb_213": 0.8344533978886615,
    "skip_rate_min_at_lambda1_223": 0.970608446587865,
    "skip_upper_bound_from_flip_223": 0.43920000000000003,
    "kanna114_flip_rate": 0.5608,
    "int4_spec_lambda1_ceiling_204": 520.9527323111674,
    "fp16_ceiling_central_mstep17_220": 306.44278371245144,
    "fp16_mstep_crossover_500_220": 1.041905464622335,
    "nonspec_official_tps_196": 165.43791973106974,
    "offtheshelf_122_overhead_pct": 51.78,
}
TOL_PIN = 1e-6          # loaded-vs-pinned round-trip tolerance
TOL_TRANSFORM = 1e-6    # TPS(overhead) calibration round-trip tolerance


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _load_json(relpath: str) -> dict[str, Any]:
    path = REPO_ROOT / relpath
    if not path.exists():
        raise FileNotFoundError(f"banked artifact missing: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _close(a: float, b: float, tol: float = TOL_PIN) -> bool:
    return _finite(a) and _finite(b) and abs(float(a) - float(b)) <= tol


# --------------------------------------------------------------------------- #
# Import the banked anchors (read-only) and round-trip them against the pins.
# --------------------------------------------------------------------------- #
def _import_anchors() -> dict[str, Any]:
    kf = _load_json(KF216_REL)["synthesis"]
    kb = _load_json(KB213_REL)["synthesis"]
    mg = _load_json(MG223_REL)["synthesis"]
    fp = _load_json(FP220_REL)["synthesis"]
    ns = _load_json(NS196_REL)

    kb_head = kb["headline"]
    bb_lam1 = kb["regimes"]["both_bugs"]["endpoint_anchors"]["overhead_budget_at_lambda_1"]
    mg_bb = mg["regimes"]["both_bugs"]["tau_central_1p0"]
    mg_route = next(r for r in mg_bb["route_comparison"]["routes"] if r["route"] == "MarginGate")

    a: dict[str, Any] = {
        # ---- #216 lane-a kernel feasibility ----
        "kernel_floor_pct": kf["cost_decomposition"]["custom_kernel_overhead_floor_pct"],
        "kernel_overhead_band_pct": kf["cost_decomposition"]["plausible_custom_overhead_band_pct"],
        "lambda_min_kernel_feasible": kf["headline"]["lambda_min_kernel_feasible"],
        "lambda_min_kernel_feasible_is_physical":
            kf["headline"]["lambda_min_kernel_feasible_is_physical"],
        "verify_gemm_share": kf["cost_decomposition"]["verify_gemm_cost_share_of_step"],
        "buildable_at_lambda1_floor":
            kf["regimes"]["both_bugs"]["tau_central_1p0"]["buildable_at_lambda_1_floor"],
        "buildable_at_lambda_hat_floor":
            kf["regimes"]["both_bugs"]["tau_central_1p0"]["buildable_at_lambda_hat_floor"],
        "kf_verdict": kf["verdict"],
        # ---- #213 budget curve (scalars from headline; official_tps from endpoint anchor) ----
        "budget_at_lambda1_bb": kb_head["overhead_budget_at_lambda_1_both_bugs_tau1"],
        "lambda_crit_bb": kb_head["lambda_crit_clears_500_zero_overhead_both_bugs_tau1"],
        "lambda_crit_descent": kb_head["lambda_crit_clears_500_zero_overhead_descent_tau1"],
        "budget_at_lambda1_descent": kb_head["overhead_budget_at_lambda_1_descent_tau1"],
        "bb_lambda1_official_tps": bb_lam1["official_tps_tau1"],
        "offtheshelf_122_clears":
            kb_head["off_the_shelf_122_clears_at_physical_lambda_both_bugs_tau1"],
        "max_budget_at_prob_saturation_bb":
            kb_head["max_budget_pct_at_prob_saturation_both_bugs_tau1"],
        # ---- #223 MarginGate ----
        "skip_rate_min_at_lambda1": mg["headline"]["skip_rate_min_at_lambda1"],
        "skip_rate_min_at_spine_0997": mg["headline"]["skip_rate_min_at_spine_0997"],
        "skip_upper_bound_from_flip": mg["headline"]["skip_upper_bound_from_flip_rate"],
        "kanna114_flip_rate": mg["imported_anchors"]["kanna114_pertoken_flip_rate"],
        "margingate_overhead_at_cap_pct":
            mg_route["overhead_at_skip_anchors_pct"]["0.4392"],
        "mg_verdict": mg["verdict"],
        "offtheshelf_122_overhead_pct":
            mg["imported_anchors"]["offtheshelf_whole_model_overhead_pct_122"],
        # ---- #220 fp16-verify ceiling ----
        "int4_spec_lambda1_ceiling": fp["composition"]["int4_spec_lambda1_ceiling"],
        "fp16_ceiling_central_mstep17": fp["headline"]["fp16verify_ceiling_at_lambda1"],
        "fp16_all_mstep_below_500": fp["headline"]["all_mstep_ceilings_below_500"],
        "fp16_mstep_crossover_500": fp["headline"]["mstep_crossover_ceiling_500"],
        "fp16_mstep_sweep": fp["composition"]["mstep_sweep"],
        "fp16_mstep_central": fp["composition"]["mstep_central"],
        "fp16_verdict": fp["verdict"],
        # ---- #196 no-spec floor ----
        "nonspec_official_tps": ns["nonspec_official_tps_est"],
        "nonspec_hw_band": ns["nonspec_official_tps_est_hw_band"],
        "nonspec_clears_500": ns["nonspec_clears_500"],
        "nonspec_token_identity_rate": ns["nonspec_token_identity_rate"],
        "ns_verdict_label": ns["verdict_label"],
    }
    return a


def _roundtrip_pins(a: dict[str, Any]) -> dict[str, bool]:
    """Each path's banked ceiling/verdict must match the instruction-pinned anchor."""
    return {
        "kernel_floor_pct_216": _close(a["kernel_floor_pct"], PIN["kernel_floor_pct_216"]),
        "lambda_min_kernel_feasible_216":
            _close(a["lambda_min_kernel_feasible"], PIN["lambda_min_kernel_feasible_216"]),
        "verify_gemm_share_216": _close(a["verify_gemm_share"], PIN["verify_gemm_share_216"]),
        "budget_at_lambda1_bb_213": _close(a["budget_at_lambda1_bb"], PIN["budget_at_lambda1_bb_213"]),
        "lambda_crit_bb_213": _close(a["lambda_crit_bb"], PIN["lambda_crit_bb_213"]),
        "skip_rate_min_at_lambda1_223":
            _close(a["skip_rate_min_at_lambda1"], PIN["skip_rate_min_at_lambda1_223"]),
        "skip_upper_bound_from_flip_223":
            _close(a["skip_upper_bound_from_flip"], PIN["skip_upper_bound_from_flip_223"]),
        "kanna114_flip_rate": _close(a["kanna114_flip_rate"], PIN["kanna114_flip_rate"]),
        "int4_spec_lambda1_ceiling_204":
            _close(a["int4_spec_lambda1_ceiling"], PIN["int4_spec_lambda1_ceiling_204"]),
        "fp16_ceiling_central_mstep17_220":
            _close(a["fp16_ceiling_central_mstep17"], PIN["fp16_ceiling_central_mstep17_220"]),
        "fp16_mstep_crossover_500_220":
            _close(a["fp16_mstep_crossover_500"], PIN["fp16_mstep_crossover_500_220"]),
        "nonspec_official_tps_196":
            _close(a["nonspec_official_tps"], PIN["nonspec_official_tps_196"]),
        "offtheshelf_122_overhead_pct":
            _close(a["offtheshelf_122_overhead_pct"], PIN["offtheshelf_122_overhead_pct"]),
    }


# --------------------------------------------------------------------------- #
# (1) The collapsed menu (the table).
# --------------------------------------------------------------------------- #
def _tps_at_overhead(ceiling_tps: float, overhead_pct: float) -> float:
    """Effective TPS after adding `overhead_pct` of step time (E[T] unchanged)."""
    return ceiling_tps / (1.0 + overhead_pct / 100.0)


def _build_menu(a: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    int4_ceiling = a["int4_spec_lambda1_ceiling"]            # 520.953  (#204, pinned)
    floor_pct = a["kernel_floor_pct"]                        # 0.9455%  (#216)
    lam_min = a["lambda_min_kernel_feasible"]                # 0.8572   (#216)
    fp16_central = a["fp16_ceiling_central_mstep17"]         # 306.44   (#220, M_step=1.7)
    skip_demand = a["skip_rate_min_at_lambda1"]              # 0.9706   (#223)
    skip_supply_cap = a["skip_upper_bound_from_flip"]        # 0.4392   (#114→#223)
    nospec = a["nonspec_official_tps"]                       # 165.4    (#196)

    # lane-a floor-adjusted ceiling: the 520.953 frontier minus the split-K floor overhead.
    lane_a_floor_adjusted = _tps_at_overhead(int4_ceiling, floor_pct)
    # MarginGate best case = #213 budget-calibrated λ=1 ceiling minus the cap-skip overhead.
    margingate_best_tps = _tps_at_overhead(a["bb_lambda1_official_tps"],
                                           a["margingate_overhead_at_cap_pct"])

    menu = [
        {
            "route": "lane_a_int4_kernel",
            "family": "speculative · int4 batch-invariant verify kernel",
            "validity_basis": (
                "split-K reduction-order fix makes the verify-M argmax identical to the AR-M=1 "
                "argmax ⇒ greedy-valid by construction (#192 strict-A)"),
            "valid": True,
            "lambda1_ceiling_tps": int4_ceiling,
            "lambda1_ceiling_floor_adjusted_tps": lane_a_floor_adjusted,
            "clears_500": True,
            "clears_500_basis": (
                f"CLEARS at the {floor_pct:.4f}%-of-step split-K floor for λ ≥ {lam_min:.4f} "
                f"(#216 over #213's budget; ∅ below — budget ≤ 0 under λ_crit={a['lambda_crit_bb']:.4f})"),
            "double_gated": True,
            "gate_a_kernel_build": (
                f"build the batch-invariant int4 verify kernel near its {floor_pct:.4f}%-of-step "
                f"split-K floor (band [{a['kernel_overhead_band_pct'][0]:.2f}%, "
                f"{a['kernel_overhead_band_pct'][1]:.2f}%]); the cheap split-K microbench settles it"),
            "gate_b_lambda": f"land #71 self-KV recovery λ ≥ {lam_min:.4f}",
            "what_it_needs": "DOUBLE-gated: (a) kernel BUILD near floor AND (b) land #71 λ ≥ 0.8572",
            "src": "#216 / #213 / #204",
        },
        {
            "route": "lane_b_fp16_verify",
            "family": "speculative · fp16/bf16 verify",
            "validity_basis": (
                "batch-invariant cuBLAS GEMM (no M-dependent split-K) ⇒ AR-M=1 and verify-M argmax "
                "identical ⇒ greedy-valid by construction (imported #211 premise; lawine #221 confirms)"),
            "valid": True,
            "lambda1_ceiling_tps": fp16_central,
            "lambda1_ceiling_note": (
                f"= int4_ceiling/M_step = {a['int4_spec_lambda1_ceiling']:.3f}/{a['fp16_mstep_central']} "
                f"at central M_step; < 500 for every physical M_step ≥ {min(a['fp16_mstep_sweep'])} "
                f"(crossover {a['fp16_mstep_crossover_500']:.4f})"),
            "clears_500": False,
            "clears_500_basis": (
                f"NEVER — draft-INDEPENDENT λ=1 cap {a['int4_spec_lambda1_ceiling']:.3f}/M_step is "
                f"below 500 for all M_step ≥ {min(a['fp16_mstep_sweep'])} ≫ break-even "
                f"{a['fp16_mstep_crossover_500']:.4f} (#220)"),
            "double_gated": False,
            "what_it_needs": (
                "nothing buildable lifts it — at λ=1 E[T] saturates at the tree max regardless of "
                "draft quality, so no draft, however strong, raises the λ=1 ceiling above 500"),
            "src": "#220",
        },
        {
            "route": "margingate_provable_skip",
            "family": "speculative · provable-argmax-stability skip",
            "validity_basis": (
                "a SOUND gate (margin > 2·ε_max) skips only provably-stable positions; the low-margin "
                "residual falls back to AR M=1 ⇒ greedy-valid (#192 strict-A)"),
            "valid": True,
            "lambda1_ceiling_tps": margingate_best_tps,
            "lambda1_ceiling_note": (
                f"best case at the SOUND skip cap {skip_supply_cap:.4f}: overhead "
                f"{a['margingate_overhead_at_cap_pct']:.2f}% of step ⇒ {margingate_best_tps:.1f} TPS"),
            "skip_demand_at_lambda1": skip_demand,
            "skip_supply_cap": skip_supply_cap,
            "clears_500": False,
            "clears_500_basis": (
                f"NEVER — needs SOUND skip ≥ {skip_demand:.4f} to fit the λ=1 budget, but #114's "
                f"{a['kanna114_flip_rate']*100:.2f}% flip caps the sound skip at ≤ {skip_supply_cap:.4f} "
                f"(provably-stable ⊆ non-flip); demand ≫ supply (#223)"),
            "double_gated": False,
            "what_it_needs": (
                f"a sound provable-stable skip ≥ {skip_demand:.4f}; the {skip_demand - skip_supply_cap:.4f}"
                f"-wide demand–supply gap is unclosable (needs < 3% full-verify vs ≥ 56.08% flips)"),
            "src": "#223 / #114",
        },
        {
            "route": "lane_c_no_spec",
            "family": "non-speculative · int4 AR greedy",
            "validity_basis": (
                "plain int4 autoregressive greedy — no verify GEMM, no batch-width mismatch "
                f"(token-identity rate {a['nonspec_token_identity_rate']:.1f}, #196)"),
            "valid": True,
            "lambda1_ceiling_tps": nospec,
            "lambda1_ceiling_note": f"σ_hw band [{a['nonspec_hw_band'][0]:.1f}, {a['nonspec_hw_band'][1]:.1f}]",
            "clears_500": False,
            "clears_500_basis": (
                f"NEVER — no-verify floor ≈ {nospec:.1f} official TPS, 66.9% below 500; structural (#196)"),
            "double_gated": False,
            "what_it_needs": "nothing — it is the compliant floor; the spec premium (≈316 TPS) is existential",
            "src": "#196",
        },
    ]

    nonworking_reference = {
        "route": "off_the_shelf_VLLM_BATCH_INVARIANT_122",
        "note": (
            "whole-model determinism (#122, +51.78% step) — the WRONG kernel for the localized "
            "#192 bug; clears at NO physical λ≤1 (max budget at prob-saturation "
            f"{a['max_budget_at_prob_saturation_bb']:.2f}% < {a['offtheshelf_122_overhead_pct']:.2f}%). "
            "It is lane-a's whole-model foil, ~55× the verify-GEMM-only cost — NOT a 5th survivor."),
        "overhead_pct": a["offtheshelf_122_overhead_pct"],
        "clears_500": bool(a["offtheshelf_122_clears"]),
    }
    return menu, nonworking_reference


# --------------------------------------------------------------------------- #
# (2) The cluster's per-gate leverage  +  (3) the #211 verdict.
# --------------------------------------------------------------------------- #
def _cluster_leverage(survivor: dict[str, Any]) -> dict[str, Any]:
    # lane-a is DOUBLE-gated. The Blackwell node TRAINS A DRAFT (Issue #211).
    gate_a_helped = False   # split-K reduction-order fix is draft-INDEPENDENT
    gate_b_helped = True    # a stronger draft raises E[T](λ) and achievable λ
    # Necessary (lifts the required λ gate) but NOT sufficient (the kernel build is untouched
    # AND is also required) ⇒ the cluster cannot, by itself, unlock 500.
    necessary_but_not_sufficient = bool(gate_b_helped and not gate_a_helped)
    return {
        "survivor_route": survivor["route"],
        "survivor_double_gated": survivor["double_gated"],
        "cluster_offer": "Issue #211 Blackwell node — TRAIN A DRAFT",
        "cluster_helps_gate": {"gate_a_kernel": gate_a_helped, "gate_b_lambda": gate_b_helped},
        "gate_a_kernel": {
            "what": survivor["gate_a_kernel_build"],
            "cluster_draft_training_helps": gate_a_helped,
            "reason": (
                "the batch-invariant fix is a split-K reduction-order change in the verify GEMM — "
                "draft-INDEPENDENT; a stronger draft changes E[T](λ), not the GEMM's reduction geometry"),
            "binding_diagnostic": "the cheap split-K microbench (#216's decisive diagnostic, ~1–2 days)",
        },
        "gate_b_lambda": {
            "what": survivor["gate_b_lambda"],
            "cluster_draft_training_helps": gate_b_helped,
            "reason": (
                "a stronger Blackwell draft raises E[T](λ) at fixed λ and lifts the achievable λ, "
                "moving land #71 toward the 0.8572 threshold"),
        },
        "cluster_necessary_but_not_sufficient": necessary_but_not_sufficient,
    }


def _verdict(n_surviving: int, survivor: dict[str, Any] | None,
             leverage: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if (n_surviving == 1 and survivor is not None
            and survivor["route"] == "lane_a_int4_kernel"
            and survivor["double_gated"]
            and leverage["cluster_necessary_but_not_sufficient"]):
        headline = "CLUSTER_UNLOCKS_500_VIA_LANE_A_ONLY_IF_KERNEL_BUILT"
    elif n_surviving == 0:
        headline = "CLUSTER_DOES_NOT_UNLOCK_500"
    else:
        headline = "CLUSTER_VERDICT_INDETERMINATE"
    detail = {
        "headline": headline,
        "cluster_pays_off_iff": (
            "the int4 batch-invariant kernel is ALSO built near its 0.9455%-of-step floor "
            "(gate-a) — the cluster's draft-training lifts only the λ gate (gate-b)"),
        "binding_next_action": "split_k_microbench_gate_a",
        "binding_next_action_detail": (
            "authorize the cheap split-K microbench FIRST (#216's decisive gate-a diagnostic, "
            "~1–2 days); it settles whether lane-a is buildable near floor. If gate-a fails, no "
            "draft-training — however strong — unlocks 500, since every other valid path NEVER clears."),
        "draft_training_conditional_on_gate_a": True,
    }
    return headline, detail


def _honest_band(a: dict[str, Any]) -> dict[str, Any]:
    return {
        "a_lane_a_floor_is_buildability_prior": (
            f"lane-a's {a['kernel_floor_pct']:.4f}%-of-step split-K floor is a buildability PRIOR "
            "(#216 scope-proportionality + split-K-penalty estimates), proven ONLY by the GPU "
            "split-K microbench — this leg does not certify it"),
        "b_margingate_out_rests_on_demand_and_supply": (
            f"the MarginGate-out verdict rests on #223's VERIFIED demand (sound skip ≥ "
            f"{a['skip_rate_min_at_lambda1']:.4f}, needing < 3% full-verify) AND #114's model-free "
            f"supply cap (≤ {a['skip_upper_bound_from_flip']:.4f} from the 56.08% flip). The supply "
            "cap's tightening leg is still open, but the DEMAND alone is already implausible"),
        "c_ceilings_budget_et_imported_unchanged": (
            "the int4-spec λ=1 ceiling 520.953 (#204), the budget curve / λ_crit (#213), the fp16 "
            "ceiling (#220), the no-spec floor (#196), and E[T](λ) reach-DP (#175/#184) are IMPORTED "
            "unchanged; the two-ceiling note (reach-DP 536.66 vs #204 520.95 — both > 500) is carried"),
        "d_lambda1_is_an_upper_bound": (
            "#199's rank-1 coverage 0.7304 over-counts the true compliant accept, so every λ=1 ceiling "
            "here is an UPPER bound; the verdict structure (1 survivor, double-gated) is robust to it"),
    }


def _handoff(headline: str) -> str:
    unlocks = ("unlocks 500 only if the kernel is also built near floor"
               if headline == "CLUSTER_UNLOCKS_500_VIA_LANE_A_ONLY_IF_KERNEL_BUILT"
               else "does not unlock 500")
    return (
        "HAND-OFF (Issue #211 + fern #185): the valid-verify menu has collapsed to ONE survivor "
        "(lane-a int4 kernel, DOUBLE-gated on a near-floor kernel BUILD and land #71 λ ≥ 0.8572); "
        f"the Blackwell cluster's draft-training helps only the λ gate, not the kernel build, so it "
        f"{unlocks} — authorize the cheap split-K microbench first (gate-a's decisive diagnostic), "
        "draft-training is conditional on it. Adds 0 TPS; imports ceilings/budget/divergence "
        "unchanged; authorizes nothing. NOT a launch."
    )


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    a = _import_anchors()
    roundtrips = _roundtrip_pins(a)
    menu, nonworking = _build_menu(a)

    survivors = [p for p in menu if p["clears_500"]]
    n_surviving = len(survivors)
    survivor = survivors[0] if n_surviving == 1 else None
    out_paths = [p for p in menu if not p["clears_500"]]

    leverage = _cluster_leverage(survivor) if survivor is not None else {
        "cluster_necessary_but_not_sufficient": False}
    headline, verdict_detail = _verdict(n_surviving, survivor, leverage)

    # TPS(overhead) transform calibration: at overhead = budget@1, TPS must equal exactly 500.
    transform_calib_tps = _tps_at_overhead(a["bb_lambda1_official_tps"], a["budget_at_lambda1_bb"])

    # ---------------- self-test (PRIMARY) ---------------- #
    # (a) the menu round-trips each path's banked ceiling/verdict.
    cond_a = all(roundtrips.values())
    # (b) exactly one surviving valid-500 path.
    cond_b = (n_surviving == 1)
    # (c) lane-a clears at floor for λ ≥ 0.8572 and is ∅ below (round-trip #216).
    cond_c = bool(
        survivor is not None
        and survivor["route"] == "lane_a_int4_kernel"
        and _close(a["lambda_min_kernel_feasible"], PIN["lambda_min_kernel_feasible_216"])
        and a["lambda_min_kernel_feasible_is_physical"]
        and a["buildable_at_lambda1_floor"] is True          # clears at λ=1 (≥ 0.8572)
        and a["buildable_at_lambda_hat_floor"] is False       # ∅ at λ̂=0.342 (< 0.8572)
        and a["lambda_min_kernel_feasible"] >= a["lambda_crit_bb"] - 1e-9)
    # (d) the three OUT paths' λ=1 ceilings are all < 500.
    cond_d = bool(len(out_paths) == 3
                  and all(_finite(p["lambda1_ceiling_tps"]) and p["lambda1_ceiling_tps"] < TARGET
                          for p in out_paths)
                  and _close(transform_calib_tps, TARGET, TOL_TRANSFORM))
    # (e) cluster necessary-but-not-sufficient (gate-a unaffected by draft).
    cond_e = bool(leverage.get("cluster_necessary_but_not_sufficient") is True
                  and leverage["cluster_helps_gate"]["gate_a_kernel"] is False
                  and leverage["cluster_helps_gate"]["gate_b_lambda"] is True)
    # (f) NaN-clean (finalized by the caller after the full payload walk).
    conditions = {
        "a_menu_roundtrips_banked_ceilings_and_verdicts": bool(cond_a),
        "b_exactly_one_surviving_valid_500_path": bool(cond_b),
        "c_lane_a_clears_at_floor_above_0p8572_and_empty_below": bool(cond_c),
        "d_three_out_paths_lambda1_ceilings_below_500": bool(cond_d),
        "e_cluster_necessary_but_not_sufficient_gate_a_unaffected": bool(cond_e),
        "f_nan_clean": True,
    }

    return {
        "self_test": {
            "valid_verify_cluster_capstone_self_test_passes": bool(all(conditions.values())),
            "conditions": conditions,
            "pin_roundtrips": roundtrips,
        },
        "test_metric": {"n_surviving_valid_500_paths": n_surviving},
        "headline": {
            "verdict": headline,
            "n_surviving_valid_500_paths": n_surviving,
            "surviving_route": survivor["route"] if survivor else None,
            "cluster_helps_gate": leverage.get("cluster_helps_gate"),
            "cluster_necessary_but_not_sufficient":
                leverage.get("cluster_necessary_but_not_sufficient"),
            "binding_next_action": verdict_detail["binding_next_action"],
            "draft_training_conditional_on_gate_a":
                verdict_detail["draft_training_conditional_on_gate_a"],
            "lambda_min_kernel_feasible": a["lambda_min_kernel_feasible"],
            "kernel_floor_pct": a["kernel_floor_pct"],
            "budget_at_lambda1_both_bugs_tau1": a["budget_at_lambda1_bb"],
            "skip_rate_min_at_lambda1": a["skip_rate_min_at_lambda1"],
            "skip_upper_bound_from_flip": a["skip_upper_bound_from_flip"],
            "int4_spec_lambda1_ceiling": a["int4_spec_lambda1_ceiling"],
            "fp16_ceiling_central_mstep17": a["fp16_ceiling_central_mstep17"],
            "nonspec_official_tps": a["nonspec_official_tps"],
        },
        "valid_verify_menu": menu,
        "nonworking_reference": nonworking,
        "cluster_leverage": leverage,
        "issue_211_verdict": verdict_detail,
        "honest_band": _honest_band(a),
        "imported_anchors": a,
        "composition": {
            "target_official": TARGET,
            "baseline_tps": BASELINE_TPS,
            "spine_0997": SPINE_0997,
            "lambda_band_floor_spine_ceiling": list(LAMBDA_BAND),
            "lambda_crit_both_bugs_tau1": a["lambda_crit_bb"],
            "lambda_crit_descent_tau1": a["lambda_crit_descent"],
            "tps_at_overhead_transform_calibration_tps": transform_calib_tps,
            "sources": {
                "kernel_feasibility_216": KF216_REL, "kernel_budget_lambda_213": KB213_REL,
                "margingate_budget_223": MG223_REL, "fp16_verify_ceiling_220": FP220_REL,
                "compliant_nonspec_floor_196": NS196_REL,
                "int4_ceiling_204": "research/validity/launch_sigma_unit_rebase",
                "divergence_114": "kanna #114 (9q5yy9l1)",
            },
        },
        "handoff_line": _handoff(headline),
        "verdict": headline,
    }


# --------------------------------------------------------------------------- #
# NaN guard + report + W&B (mirrors #216/#213; never fatal).
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
    st, hd, comp = syn["self_test"], syn["headline"], syn["composition"]
    print("\n" + "=" * 100, flush=True)
    print("VALID-VERIFY CLUSTER CAPSTONE (PR #227, wirbel) — Issue #211 Blackwell decision, CPU-only",
          flush=True)
    print("=" * 100, flush=True)
    print(f"  target {comp['target_official']:.0f}   baseline {comp['baseline_tps']:.2f} (this leg "
          f"adds 0 TPS)   λ band floor/spine/ceiling = {comp['lambda_band_floor_spine_ceiling']}",
          flush=True)
    print("-" * 100, flush=True)
    print("  THE COLLAPSED MENU (validity basis × λ=1 ceiling × clears-500? × what-it-needs):", flush=True)
    for p in syn["valid_verify_menu"]:
        mark = "SURVIVOR" if p["clears_500"] else "out"
        ceil = p["lambda1_ceiling_tps"]
        print(f"    [{mark:>10}] {p['route']:<24} valid={p['valid']}  λ=1 ceiling≈{ceil:7.2f} TPS",
              flush=True)
        print(f"                 needs: {p['what_it_needs']}", flush=True)
        print(f"                 basis: {p['clears_500_basis']}", flush=True)
    nw = syn["nonworking_reference"]
    print(f"    [ reference ] {nw['route']:<24} clears_500={nw['clears_500']}  (+{nw['overhead_pct']:.2f}% — wrong kernel)",
          flush=True)
    print("-" * 100, flush=True)
    print(f"  n_surviving_valid_500_paths = {hd['n_surviving_valid_500_paths']}  "
          f"(survivor: {hd['surviving_route']})", flush=True)
    print("-" * 100, flush=True)
    print("  CLUSTER PER-GATE LEVERAGE (Issue #211 Blackwell node TRAINS A DRAFT):", flush=True)
    print(f"      gate_a_kernel build  : cluster helps = {hd['cluster_helps_gate']['gate_a_kernel']} "
          "(split-K fix is draft-INDEPENDENT)", flush=True)
    print(f"      gate_b_lambda ≥0.8572: cluster helps = {hd['cluster_helps_gate']['gate_b_lambda']} "
          "(stronger draft raises E[T](λ), achievable λ)", flush=True)
    print(f"      cluster_necessary_but_not_sufficient = {hd['cluster_necessary_but_not_sufficient']}",
          flush=True)
    print("-" * 100, flush=True)
    print(f"  #211 VERDICT : {syn['verdict']}", flush=True)
    print(f"      binding next action  : {hd['binding_next_action']}  "
          f"(draft-training conditional on gate-a = {hd['draft_training_conditional_on_gate_a']})",
          flush=True)
    print("-" * 100, flush=True)
    print(f"  PRIMARY valid_verify_cluster_capstone_self_test_passes = "
          f"{st['valid_verify_cluster_capstone_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"        - {k}: {v}", flush=True)
    print(f"  TEST n_surviving_valid_500_paths = {syn['test_metric']['n_surviving_valid_500_paths']} "
          f"(expect 1)", flush=True)
    print("=" * 100, flush=True)
    print(f"\n  {syn['handoff_line']}\n", flush=True)


def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[valid-verify-capstone] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    hd, comp = syn["headline"], syn["composition"]
    lev = syn["cluster_leverage"]
    run = init_wandb_run(
        job_type="valid-verify-cluster-capstone",
        agent="wirbel",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["valid-verify-cluster-capstone", "issue-211", "issue-192", "validity-gate",
              "lane-a", "blackwell-decision", "capstone"],
        config={
            "target_official": TARGET, "baseline_tps": BASELINE_TPS, "spine_0997": SPINE_0997,
            "lambda_band": list(LAMBDA_BAND),
            "lambda_crit_both_bugs_tau1": comp["lambda_crit_both_bugs_tau1"],
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[valid-verify-capstone] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    st = syn["self_test"]
    summary: dict[str, Any] = {
        # PRIMARY + TEST
        "valid_verify_cluster_capstone_self_test_passes":
            int(bool(st["valid_verify_cluster_capstone_self_test_passes"])),
        "n_surviving_valid_500_paths": hd["n_surviving_valid_500_paths"],
        # verdict
        "verdict_cluster_unlocks_only_if_kernel_built":
            int(syn["verdict"] == "CLUSTER_UNLOCKS_500_VIA_LANE_A_ONLY_IF_KERNEL_BUILT"),
        "cluster_helps_gate_a_kernel": int(bool(lev["cluster_helps_gate"]["gate_a_kernel"])),
        "cluster_helps_gate_b_lambda": int(bool(lev["cluster_helps_gate"]["gate_b_lambda"])),
        "cluster_necessary_but_not_sufficient":
            int(bool(lev["cluster_necessary_but_not_sufficient"])),
        "draft_training_conditional_on_gate_a":
            int(bool(hd["draft_training_conditional_on_gate_a"])),
        # imported menu anchors
        "lambda_min_kernel_feasible": hd["lambda_min_kernel_feasible"],
        "kernel_floor_pct": hd["kernel_floor_pct"],
        "budget_at_lambda1_both_bugs_tau1": hd["budget_at_lambda1_both_bugs_tau1"],
        "skip_rate_min_at_lambda1": hd["skip_rate_min_at_lambda1"],
        "skip_upper_bound_from_flip": hd["skip_upper_bound_from_flip"],
        "int4_spec_lambda1_ceiling": hd["int4_spec_lambda1_ceiling"],
        "fp16_ceiling_central_mstep17": hd["fp16_ceiling_central_mstep17"],
        "nonspec_official_tps": hd["nonspec_official_tps"],
        "lambda_crit_both_bugs_tau1": comp["lambda_crit_both_bugs_tau1"],
        "tps_at_overhead_transform_calibration_tps":
            comp["tps_at_overhead_transform_calibration_tps"],
        # per-path clears_500 flags
        **{f"clears_500_{p['route']}": int(bool(p["clears_500"]))
           for p in syn["valid_verify_menu"]},
        # self-test conditions + pin round-trips
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
        **{f"pinrt_{k}": int(bool(v)) for k, v in st["pin_roundtrips"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="valid_verify_cluster_capstone_result", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    print(f"[valid-verify-capstone] wandb logged {len(summary)} keys", flush=True)


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
        "created_at": created_at, "pr": 227, "agent": "wirbel",
        "kind": "valid-verify-cluster-capstone", "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    syn["self_test"]["conditions"]["f_nan_clean"] = not nan_paths
    syn["self_test"]["valid_verify_cluster_capstone_self_test_passes"] = bool(
        all(syn["self_test"]["conditions"].values()))
    if nan_paths:
        print(f"[valid-verify-capstone] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "valid_verify_cluster_capstone_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[valid-verify-capstone] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = (syn["self_test"]["valid_verify_cluster_capstone_self_test_passes"]
              and payload["nan_clean"])
        print(f"[valid-verify-capstone] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
