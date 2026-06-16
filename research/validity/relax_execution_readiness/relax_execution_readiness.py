#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Relax-path execution-readiness skeleton (PR #465, land) — CPU-only analytic / spec-authoring.

THE QUESTION (turn a human "GO (relax strict)" into a zero-latency, fully-specified, reversible action)
-------------------------------------------------------------------------------------------------------
My #462 decision surface (sb1n4aa6) pre-wired the *verdict*: recommend(gain, ppl, same_kind, bar, k)
collapses stark #452's three pending numbers into GO / NO-GO / CI-AMBIGUOUS. But it stops at the
verdict. If the human reads the packet and says GO on #407, there is currently NO pre-staged
EXECUTION path — someone still has to figure out, under time pressure, *which* served-file knob to
flip, *how* to validate the post-relax config, and *how* to roll back. This card supplies the missing
EXECUTION complement: the analysis-only template that turns a "GO" into a fully-specified action.

  #462 says WHEN to go (the verdict surface).  THIS card says HOW to go SAFELY (the execution skeleton).

It is a SPEC / SKELETON ONLY — NO served-file change, NO submission, NO HF job. Applying the spec
remains a human-gated served-file change (operator Directive #3: served-file change + leaderboard
submission stay human-gated). stark #452's realized config drops into the empty <PENDING #452> slots,
exactly as #462's recommend() consumes #452's numbers.

STATUS UPDATE (2026-06-16): stark #452 LANDED and is NEGATIVE (runs daqrzr99 / 00ovtdnt, MERGED,
W&B-verified). The relax split-K re-partition was BUILT and benchmarked: realized 466.20 TPS
(DOMINATED — below deployed 481.53, gain -15.33; below the strict frontier 467.14 too) with a
cascading identity collapse (0.9966 -> 0.730, 3 -> 3317 flips; flip-KIND = free-running divergence,
NOT same-family near-ties). PPL barely moved (+0.001 -> 2.3782, still <= 2.42). Dropped into the
SHIPPED execution_verdict(), the realized config returns a bar-INVARIANT ROLLBACK (clause-a TPS
dominated AND clause-c new-KIND both fail; clause-b PPL and clause-d completion pass). The relax lane
is CLOSED. This card's value is now "the execution gate, validated on real data against the config
that tried and failed to clear it." The parameterized <PENDING #452> form is KEPT alongside the
realized overlay (in case the human wants the spec for a future lever).

FIVE PRODUCTS (this card produces; fern #357 folds the one-screen GO/NO-GO + "if GO, here's the action")
------------------------------------------------------------------------------------------------------
  (1) THE EXACT SERVED-FILE CHANGE SPEC (parameterized, NO edit). The realistic split-K relax-prize
      (+17.05 realistic / +29.34 ceiling over deployed 481.53) lives in the int4-Marlin W4A16 GEMM
      K-reduction (~85% of verify; ubel #450 c5oyb7gv roofline; stark #448 fn4iz0dz knob audit). The
      ONLY in-tree Python-selectable knob is `use_fp32_reduce` (default True) — but that is a sub-prize
      PROXY worth only +0.64 TPS (stark #448). The actual +17..+29 prize re-partitions the K-reduction
      (split-K / BLOCK_K / num_warps) inside the Marlin CUDA kernel — NO Python knob; it requires a
      kernel/source BUILD (a patched vLLM wheel or custom kernel shipped with the submission). Every
      realized geometry value stark #452 confirms is an explicit <PENDING #452> slot.
  (2) THE POST-RELAX VALIDATION HARNESS SPEC. The exact acceptance checklist the realized config must
      pass before it could ship: (a) measured TPS >= deployed 481.53 + human bar B, CI-clean at 1
      sigma_hw (#462); (b) measured PPL <= 2.42 (the hard gate, #462's most-sensitive input, margin
      0.0428); (c) flip-KIND is same-family accumulation-order near-ties (not a new quality-destroying
      mode, #462 clause-3); (d) 128/128 completion. Each check maps to the stark #452 metric AND the
      recommend() clause it resolves.
  (3) ROLLBACK CRITERION + REVERSIBILITY. Explicit rollback trigger (any of: PPL > 2.42, new-kind
      flips, TPS gain not CI-clean of bar B, < 128/128); single-knob byte-for-byte revert to the
      deployed 481.53 config (re-point the manifest's pinned wheel + restore the stock flag).
  (4) PRE-WIRED stark #452 -> EXECUTION. When stark #452 reports (realized TPS, PPL, flip-count,
      flip-kind), each fills a <PENDING #452> slot and the harness returns SHIP-READY / HOLD /
      ROLLBACK. This is a thin EXECUTION wrapper around #462's IMPORTED recommend() (GO -> SHIP-READY,
      CI-AMBIGUOUS -> HOLD, NO-GO -> ROLLBACK) plus the 128/128 precondition — NOT a re-implementation.
      Verdict logic is handed to fern #357; this card produces only the execution skeleton.
  (5) SELF-TEST + PPL anchor 2.3772.

NON-DUPLICATION: IMPORTS #462's recommend() verbatim (re-derives no decision logic); round-trips the
committed #462 / #458 / #457(=#450 roofline) / #448 result JSONs; re-derives no numbers. relax
identity/flips/PPL/TPS and the realized split-K geometry stay stark #452 PARAMETERIZED SLOTS.

LOCAL, CPU-ONLY, ANALYTIC. No GPU / vLLM / HF Job / submission / served-file change / official draw.
BASELINE stays 481.53; adds 0 TPS; greedy/PPL untouched (PPL anchor 2.3772).

PRIMARY metric  relax_exec_spec_pending_slots  (count of <PENDING #452> slots in the spec)
TEST    metric  ppl  (2.3772 anchor; this leg does not touch the served model)
HEADLINE        relax_change_is_single_knob_reversible, exec_skeleton_maps_to_recommend
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

# --- committed source JSONs round-tripped here (re-derives nothing) ----------------------------- #
DECISION_SURFACE_JSON = (
    REPO_ROOT / "research/validity/relax_decision_surface/relax_decision_surface_results.json"
)  # #462 land sb1n4aa6 — the verdict surface this card executes against
COST_LEDGER_JSON = (
    REPO_ROOT / "research/validity/relax_decision_cost_ledger/relax_decision_cost_ledger_results.json"
)  # #458 land uhhyec0q — cost ledger / gains / gate margin
CEILING_JSON = (
    REPO_ROOT / "research/validity/unified_absolute_ceiling/unified_absolute_ceiling_results.json"
)  # #457 land h0uggl9i — banks ubel #450 c5oyb7gv roofline split-K realistic/optimistic (the prize)
INT4_AUDIT_JSON = (
    REPO_ROOT
    / "research/validity/int4_gemm_kernel_config_audit/int4_gemm_kernel_config_audit_results.json"
)  # #448 stark fn4iz0dz — the served int4-GEMM KNOB surface (use_fp32_reduce; build-required prize)

# --- import #462's recommend() VERBATIM (the decision logic this skeleton executes; no re-derive) - #
sys.path.insert(0, str(REPO_ROOT / "research/validity/relax_decision_surface"))
import relax_decision_surface as ds  # noqa: E402

# anchors imported from the #462 module (single source of truth; never re-typed) ------------------ #
DEPLOYED_TPS = ds.DEPLOYED_TPS                      # 481.53 (PR #52 2x9fm2zx, non-equivalent 3 flips)
SIGMA_HW = ds.SIGMA_HW                              # 4.8153 (~1% of deployed)
PPL_ANCHOR = ds.PPL_ANCHOR                          # 2.3772 (PR #52 served PPL)
PPL_GATE = ds.PPL_GATE                              # 2.42 (reference + 5%)
RELAX_REALISTIC_TPS_GAIN = ds.RELAX_REALISTIC_TPS_GAIN  # +17.0499 (#450 realistic split-K over deployed)
RELAX_CEILING_TPS_GAIN = ds.RELAX_CEILING_TPS_GAIN      # +29.3424 (#450 optimistic ceiling over deployed)
RELAX_REALISTIC_TPS = ds.RELAX_REALISTIC_TPS       # 498.5799 (#450 realistic split-K hi, greedy-UNSAFE)
K_HEADLINE = ds.K_HEADLINE                          # 1 (sigma_hw multiplier; #462 CI-clean convention)
DEPLOYED_FLIPS = ds.DEPLOYED_FLIPS                  # 3 (deployed status-quo flip count)
BEST_BYTE_EXACT_LEVER_TPS = ds.BEST_BYTE_EXACT_LEVER_TPS  # +0.26 (best strict-safe lever; ref human bar)

PPL_GATE_MARGIN = PPL_GATE - PPL_ANCHOR            # 0.0428 (#462 most-sensitive thin margin)
DECISION_FLIP_TPS_THRESHOLD = ds.ci_clean_max_bar(RELAX_REALISTIC_TPS_GAIN, K_HEADLINE)  # 12.2346

# stark #448 knob-surface constants (round-tripped from INT4_AUDIT_JSON in load_banked) ----------- #
SERVED_KERNEL = "MarlinLinearKernel"               # uniquely selected on sm_86 (A10G)
SERVED_DEFAULT_FLAG = "use_fp32_reduce=True"       # the in-tree selectable proxy default
FP32OFF_PROXY_UB_TPS = 0.6416254072425431          # use_fp32_reduce=False UB (+0.64, sub-prize)

PENDING = "<PENDING #452>"                          # the sentinel for every value stark #452 fills
HUMAN_BAR = "<HUMAN BAR B>"                          # the human-set TPS bar (NOT a #452 slot)
TOL_RT = 1e-6
TOL_TIGHT = 1e-9

# served submission anchor (the deployed config the spec parameterizes; NOT edited) -------------- #
SERVED_SUBMISSION = "submissions/fa2sw_precache_kenyan"
SERVED_MANIFEST = f"{SERVED_SUBMISSION}/manifest.json"
SERVED_VLLM_WHEEL = "vllm-0.22.1rc1.dev307+g3e8afdf78 (manifest.dependencies[0], version-pinned)"

# --- stark #452 REALIZED relax-prize measurement (daqrzr99 / 00ovtdnt, MERGED, W&B-verified) ------- #
# The relax split-K re-partition was BUILT and benchmarked. Result is NEGATIVE: dominated TPS + a
# cascading identity collapse. These numbers FILL the <PENDING #452> slots (advisor 2026-06-16). The
# realized config returns ROLLBACK from the SHIPPED gate — real-data proof the GO-branch refuses a
# dominated, identity-breaking config. The parameterized PENDING form is KEPT alongside (future levers).
STARK452_LANDED = True
STARK452_RUN_IDS = ("daqrzr99", "00ovtdnt")
STRICT_FRONTIER_TPS = 467.14                       # denken #423 strict byte-exact frontier (gain framing)
DEPLOYED_IDENTITY = 0.9966                          # deployed served identity (PR #52) — for the ref delta
STARK452_REALIZED_TPS = 466.20                      # DOMINATED: below deployed 481.53 AND strict 467.14
STARK452_REALIZED_PPL = round(PPL_ANCHOR + 0.001, 4)  # 2.3782 — delta +0.001; PASSES the 2.42 gate
STARK452_REALIZED_IDENTITY = 0.730                  # collapsed from deployed 0.9966
STARK452_REALIZED_FLIPS = 3317                      # exploded from deployed 3
STARK452_REALIZED_SAME_KIND = False                 # cascading free-running divergence, NOT same-family
STARK452_REALIZED_COMPLETED = 128                   # valid run (W&B-verified, MERGED)


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# Load + round-trip the committed source JSONs (#462 / #458 / #457=#450 / #448). Re-derives nothing.
# --------------------------------------------------------------------------- #
def load_banked() -> dict[str, Any]:
    surf = json.loads(DECISION_SURFACE_JSON.read_text(encoding="utf-8"))["synthesis"]
    s_head, s_st = surf["headline"], surf["self_test"]

    led = json.loads(COST_LEDGER_JSON.read_text(encoding="utf-8"))["synthesis"]
    l_head, l_st = led["headline"], led["self_test"]

    ceil = json.loads(CEILING_JSON.read_text(encoding="utf-8"))["synthesis"]
    c_head = ceil["headline"]

    audit = json.loads(INT4_AUDIT_JSON.read_text(encoding="utf-8"))
    a_disp, a_req, a_cfg = audit["dispatch"], audit["required"], audit["config"]

    rt = {
        # --- #462 decision surface (the verdict surface this card executes against) ---
        "surf_decision_flip_threshold_resid": abs(
            s_head["decision_flip_tps_threshold"] - DECISION_FLIP_TPS_THRESHOLD),
        "surf_max_admissible_ppl_resid": abs(s_head["max_admissible_relax_ppl"] - PPL_GATE),
        "surf_ppl_margin_resid": abs(s_head["ppl_margin_from_deployed_anchor"] - PPL_GATE_MARGIN),
        "surf_realistic_k1_resid": abs(
            s_head["relax_realistic_max_ci_clean_bar_k1"] - DECISION_FLIP_TPS_THRESHOLD),
        # --- #458 cost ledger (the gains + gate margin this card cites) ---
        "led_realistic_gain_resid": abs(
            l_head["relax_realistic_tps_gain_over_deployed"] - RELAX_REALISTIC_TPS_GAIN),
        "led_ceiling_gain_resid": abs(
            l_head["relax_ceiling_tps_gain_over_deployed"] - RELAX_CEILING_TPS_GAIN),
        "led_gate_margin_resid": abs(l_head["deployed_ppl_gate_margin"] - PPL_GATE_MARGIN),
        # --- #457 ceiling = ubel #450 c5oyb7gv roofline split-K (the prize TPS) ---
        "ceil_realistic_tps_resid": abs(c_head["relax_prize_realistic_tps"] - RELAX_REALISTIC_TPS),
        "ceil_realistic_gain_resid": abs(
            c_head["headroom_deployed_to_ceiling_tps"] - RELAX_CEILING_TPS_GAIN),
        # --- #448 stark int4-GEMM knob audit (the served knob surface this spec names) ---
        "audit_proxy_ub_resid": abs(a_req["fp32reduce_off_upperbound_tps_delta"] - FP32OFF_PROXY_UB_TPS),
        "audit_ppl_resid": abs(a_req["ppl"] - PPL_ANCHOR),
        "audit_deployed_tps_resid": abs(a_cfg["deployed_tps"] - DEPLOYED_TPS),
    }
    max_resid = max(rt.values())
    return {
        "roundtrip_resid": rt,
        "max_roundtrip_resid": max_resid,
        "all_roundtrip_ok": bool(max_resid <= TOL_RT),
        # banked self-test gates of the parents (must be green for this card to stand on them)
        "decision_surface_self_test_passed": bool(s_st["decision_surface_self_test_passes"]),
        "cost_ledger_self_test_passed": bool(l_st["cost_ledger_self_test_passes"]),
        # banked knob-surface witnesses (#448): the surface is exactly what this spec parameterizes
        "served_kernel_banked": str(a_disp["selected_kernel"]),
        "served_default_config_banked": str(a_cfg["served_default_config"]),
        "fp32reduce_off_is_byteexact_banked": bool(a_req["fp32reduce_off_is_byteexact"]),
        "has_byteexact_selectable_headroom_banked": bool(a_req["has_byteexact_selectable_headroom"]),
        "faster_int4_gemm_requires_build_banked": bool(a_req["faster_int4_gemm_requires_build"]),
        "fp32off_proxy_ub_tps_banked": float(a_req["fp32reduce_off_upperbound_tps_delta"]),
    }


# --------------------------------------------------------------------------- #
# (1) The exact served-file change spec — parameterized, NO edit. file + symbol + current -> proposed.
# --------------------------------------------------------------------------- #
def served_change_spec() -> dict[str, Any]:
    """The realized split-K relax-prize knob(s). Two layers: the in-tree Python proxy (a sub-prize
    flag) and the build-level K-reduction re-partition (the actual +17..+29 prize, no Python knob).
    Every realized value stark #452 confirms is a <PENDING #452> slot."""
    return {
        "prize_location": {
            "kernel": SERVED_KERNEL,
            "reduction": "int4-Marlin W4A16 GEMM K-dim (contraction) accumulation",
            "frac_of_verify": 0.8509,
            "prize_tps_realistic": RELAX_REALISTIC_TPS,          # 498.58 (#450 c5oyb7gv)
            "prize_gain_realistic_over_deployed": RELAX_REALISTIC_TPS_GAIN,   # +17.05
            "prize_gain_ceiling_over_deployed": RELAX_CEILING_TPS_GAIN,       # +29.34
            "bw_slack": "~16% achieved-BW slack on the int4-Marlin GEMM (#450 roofline)",
            "provenance": "ubel #450 c5oyb7gv roofline + stark #448 fn4iz0dz knob audit",
        },
        "knob_A_in_tree_python_proxy": {
            "role": "SUB-PRIZE PROXY (proves the FP-reassociation hazard; NOT the prize)",
            "file": "vllm/model_executor/layers/quantization/utils/marlin_utils.py",
            "symbol": "apply_gptq_marlin_linear(..., use_fp32_reduce=<bool>) "
                      "(default sourced from envs.VLLM_MARLIN_USE_FP32_REDUCE)",
            "lives_in": f"the version-pinned served vLLM wheel: {SERVED_VLLM_WHEEL}",
            "current_value": "use_fp32_reduce = True  (served default; stark #448 served_default_config)",
            "proposed_value": "use_fp32_reduce = False  (re-orders the K-reduce accumulation)",
            "realized_endtoend_tps_delta": FP32OFF_PROXY_UB_TPS,   # +0.64 UB — BELOW the +2 bar
            "greedy_safety": "REASSOCIATING -> breaks byte-exactness on 3/4 shapes "
                             "(qkv, o_proj, down; gate_up stays bit-exact) — stark #448",
            "why_not_the_prize": ("only +0.64 TPS UB (sub-bar). The ~16% BW prize is NOT cashed by the "
                                  "accumulation-dtype flag alone; it needs the K-reduction re-partition "
                                  "below (knob B)."),
            "exact_realized_setting": PENDING,                     # SLOT 1: stark #452 confirms the flag it built with
        },
        "knob_B_build_level_split_k_repartition": {
            "role": "THE PRIZE (+17.05 realistic / +29.34 ceiling) — re-partitions the K-reduction",
            "kernel_source": "Marlin int4 W4A16 CUDA kernel (csrc/quantization/gptq_marlin/*) inside "
                             "the served vLLM wheel",
            "symbol": "split-K geometry (num_splits / max_par), BLOCK_K tile, num_warps — "
                      "AUTO-SELECTED in-kernel as f(M); NO exposed Python knob (stark #448 / kanna #122)",
            "current_value": "deployed in-kernel split-K geometry = f(M) auto-selection "
                             "(pinned to the deployed reduction order -> identity 0.9966, 3 flips)",
            "proposed_split_k_partition": PENDING,                 # SLOT 2
            "proposed_block_k": PENDING,                           # SLOT 3
            "proposed_num_warps": PENDING,                         # SLOT 4
            "requires_build": True,
            "build_artifact": ("a PATCHED vLLM Marlin wheel (or a custom kernel shipped in the "
                               "submission) realizing the re-partitioned K-reduction"),
            "realized_kernel_artifact_ref": PENDING,               # SLOT 5: the built wheel/kernel stark #452 produced
            "submission_anchor_that_would_change": {
                "file": SERVED_MANIFEST,
                "symbol": "dependencies[0] (the pinned vLLM wheel URL)",
                "current_value": SERVED_VLLM_WHEEL,
                "proposed_value": f"re-point to the patched-Marlin wheel ({PENDING})",  # references SLOT 5
            },
            "greedy_safety": "REASSOCIATING -> greedy-UNSAFE (realistic_splitk_greedy_safe=false, #450)",
        },
        "human_gated": {
            "directive": "operator Directive #3 — served-file change + leaderboard submission are "
                         "HUMAN-GATED actions",
            "required_before_apply": "an `Approval request: HF job for <submission-name>` GitHub issue "
                                     "(PR/branch, exact command, expected metric movement / PPL risk, "
                                     "quota/runtime risk, local checks, artifact paths) + explicit human "
                                     "approval in that issue, THEN the single allowed leaderboard "
                                     "submission",
            "this_card_does": "SPECIFIES the change (approval-ready); does NOT make it. "
                              "analysis_only=true, no_served_file_change=true.",
        },
        "human_bar_B": HUMAN_BAR,   # the required TPS-over-deployed bar; human-set, NOT a #452 slot
        "summary": (
            f"PRIZE = the int4-Marlin W4A16 K-reduction split-K re-partition (+{RELAX_REALISTIC_TPS_GAIN:.2f} "
            f"realistic / +{RELAX_CEILING_TPS_GAIN:.2f} ceiling over deployed {DEPLOYED_TPS}). NO Python "
            f"knob — needs a kernel BUILD (patched Marlin wheel). The in-tree `use_fp32_reduce=True->False` "
            f"flag is only a +{FP32OFF_PROXY_UB_TPS:.2f} sub-prize proxy. Every realized geometry value is a "
            f"{PENDING} slot; applying is human-gated (Directive #3)."),
    }


# --------------------------------------------------------------------------- #
# (2) The post-relax validation harness spec — 4 clauses; each -> stark #452 metric + recommend() clause.
# --------------------------------------------------------------------------- #
def validation_harness_spec() -> dict[str, Any]:
    clauses = [
        {
            "id": "a_tps_ci_clean_over_bar",
            "check": (f"measured relax TPS >= deployed {DEPLOYED_TPS} + human bar B, CI-CLEAN at "
                      f"1 sigma_hw (gain - {SIGMA_HW} >= B)"),
            "stark452_metric": "measured_relax_tps (-> gain over deployed)",
            "recommend_clause": "clause-1: (measured_gain_tps - k*sigma_hw) >= human_bar_tps  (CI-clean GO)",
            "decision_surface_anchor": (f"decision_flip_tps_threshold={DECISION_FLIP_TPS_THRESHOLD:.4f} "
                                        f"(largest CI-clean bar for the realistic +17.05; #462)"),
            "measured_slot": PENDING,                              # SLOT 6
        },
        {
            "id": "b_ppl_le_gate",
            "check": f"measured relax PPL <= {PPL_GATE} (HARD gate; #462 most-sensitive input, "
                     f"margin {PPL_GATE_MARGIN:.4f} from anchor {PPL_ANCHOR})",
            "stark452_metric": "measured_relax_ppl",
            "recommend_clause": "clause-2: measured_ppl > PPL_GATE -> NO-GO  (hard quality fail)",
            "decision_surface_anchor": f"max_admissible_relax_ppl={PPL_GATE} (the razor-thin 0.0428 margin)",
            "measured_slot": PENDING,                              # SLOT 7
        },
        {
            "id": "c_flip_kind_same_family",
            "check": ("flip-KIND is same-family accumulation-order near-ties (the deployed reduction-order "
                      "family), NOT a new quality-destroying mode"),
            "stark452_metric": "measured_relax_flip_kind  (the KIND, NEVER the count N — orthogonal)",
            "recommend_clause": "clause-3: not break_same_kind -> NO-GO  (new failure mode)",
            "decision_surface_anchor": ("#462 clause-3 / N-invariance: the count N is orthogonal; only the "
                                        "categorical KIND gates"),
            "measured_slot": PENDING,                              # SLOT 8 (flip-kind)
            "measured_count_slot_orthogonal": PENDING,            # SLOT 9 (flip-count N; reported, never gates)
        },
        {
            "id": "d_completed_128_of_128",
            "check": "128/128 public prompts completed (benchmark validity precondition)",
            "stark452_metric": "completed (== 128)",
            "recommend_clause": ("recommend() DOMAIN PRECONDITION: gain/ppl are only defined on a valid "
                                 "128/128 run; an incomplete run is invalid -> ROLLBACK (never reaches GO)"),
            "decision_surface_anchor": "challenge validity gate (128/128 completion; program.md)",
            "measured_slot": PENDING,                              # SLOT 10 (completed count)
        },
    ]
    return {
        "clauses": clauses,
        "n_clauses": len(clauses),
        "all_pass_means": "SHIP-READY (== recommend() GO on a valid 128/128 run)",
        "any_fail_means": "HOLD (CI-AMBIGUOUS gain) or ROLLBACK (hard fail) — see (3)/(4)",
        "summary": (f"4-clause checklist: (a) TPS CI-clean >= deployed+B at 1 sigma_hw -> recommend clause-1; "
                    f"(b) PPL <= {PPL_GATE} -> clause-2; (c) same-KIND flips -> clause-3; (d) 128/128 -> "
                    f"recommend domain precondition. Every check maps to a #462 recommend() clause."),
    }


# --------------------------------------------------------------------------- #
# (3) Rollback criterion + reversibility.
# --------------------------------------------------------------------------- #
def rollback_spec() -> dict[str, Any]:
    triggers = [
        {"trigger": "measured PPL > 2.42", "maps_to": "validation clause-b fail (hard quality gate)"},
        {"trigger": "new-KIND flips (not same accumulation-order family)",
         "maps_to": "validation clause-c fail (new failure mode)"},
        {"trigger": "TPS gain NOT CI-clean of human bar B (gain - sigma_hw < B)",
         "maps_to": "validation clause-a fail (gain below bar -> NO-GO; gain clears point-only -> HOLD)"},
        {"trigger": "< 128/128 completion", "maps_to": "validation clause-d fail (invalid run)"},
    ]
    return {
        "rollback_triggers_any_of": triggers,
        "revert_mechanism": {
            "single_knob": True,
            "how": (f"restore the deployed config: re-point {SERVED_MANIFEST} dependencies[0] back to the "
                    f"stock {SERVED_VLLM_WHEEL} (undo knob B) and restore use_fp32_reduce=True (undo the "
                    f"knob-A proxy). Both are a one-line manifest/flag revert."),
            "byte_for_byte_return_to_deployed": True,
            "deployed_target": f"{DEPLOYED_TPS} TPS / PPL {PPL_ANCHOR} / 128-128 (PR #52, fully version-pinned)",
            "why_byte_for_byte": ("the deployed config is fully specified and version-pinned (manifest "
                                  "dependency + env), so reverting reproduces the deployed 481.53 config "
                                  "exactly — no residual state."),
        },
        "relax_change_is_single_knob_reversible": True,
        "summary": (f"ROLLBACK if ANY of {{PPL > {PPL_GATE}; new-kind flips; TPS gain not CI-clean of B; "
                    f"< 128/128}}. Revert is a SINGLE-KNOB, byte-for-byte return to the deployed "
                    f"{DEPLOYED_TPS} config (re-point the pinned wheel + restore the stock flag)."),
    }


# --------------------------------------------------------------------------- #
# (4) Pre-wired stark #452 -> execution. SHIP-READY / HOLD / ROLLBACK, wrapping #462's recommend().
# --------------------------------------------------------------------------- #
def execution_verdict(measured_tps: float, measured_ppl: float, flip_kind_same: bool,
                      completed: int, human_bar_tps: float, k: int = K_HEADLINE) -> str:
    """Map stark #452's realized measurement -> the ACTION. Thin EXECUTION wrapper around #462's
    IMPORTED recommend() (re-derives no decision logic):

      < 128/128            -> ROLLBACK   (invalid run; recommend() domain precondition fails)
      recommend() == GO    -> SHIP-READY
      recommend() == CI-AMBIGUOUS -> HOLD (gain clears bar only within hw noise; re-measure / human call)
      recommend() == NO-GO -> ROLLBACK   (PPL breach / new-kind / gain below bar)
    """
    if completed < 128:                                   # benchmark-validity precondition
        return "ROLLBACK"
    gain = measured_tps - DEPLOYED_TPS                     # convert absolute TPS -> gain over deployed
    verdict = ds.recommend(gain, measured_ppl, flip_kind_same, human_bar_tps, k)  # #462 logic, verbatim
    return {"GO": "SHIP-READY", "CI-AMBIGUOUS": "HOLD", "NO-GO": "ROLLBACK"}[verdict]


def prewire_stark452() -> dict[str, Any]:
    one_to_one = [
        {"stark452_reports": "realized relax TPS", "fills_slot": "measured_relax_tps",
         "drives": "validation clause-a (CI-clean gain over deployed + human bar B)"},
        {"stark452_reports": "realized relax PPL", "fills_slot": "measured_relax_ppl",
         "drives": "validation clause-b (hard gate PPL <= 2.42)"},
        {"stark452_reports": "realized flip characterization (KIND)", "fills_slot": "measured_relax_flip_kind",
         "drives": "validation clause-c (same accumulation-order family)"},
        {"stark452_reports": "realized flip-count N", "fills_slot": "measured_relax_flip_count",
         "drives": "reported for transparency; ORTHOGONAL — never gates (the KIND does)"},
        {"stark452_reports": "completed count", "fills_slot": "completed",
         "drives": "validation clause-d (128/128 validity precondition)"},
    ]
    ref_bar = BEST_BYTE_EXACT_LEVER_TPS                    # illustrative human bar (best byte-exact lever)
    realistic_tps = DEPLOYED_TPS + RELAX_REALISTIC_TPS_GAIN   # 498.58 modeled realistic prize
    ceiling_tps = DEPLOYED_TPS + RELAX_CEILING_TPS_GAIN       # 510.87 modeled ceiling
    corners = [
        {"label": "modeled realistic gain, PPL-neutral (== anchor), same-kind, 128/128, ref bar +0.26",
         "tps": realistic_tps, "ppl": PPL_ANCHOR, "same_kind": True, "completed": 128, "bar": ref_bar,
         "verdict": execution_verdict(realistic_tps, PPL_ANCHOR, True, 128, ref_bar)},
        {"label": "modeled realistic gain, PPL BREACH (2.43), same-kind, 128/128, ref bar +0.26",
         "tps": realistic_tps, "ppl": 2.43, "same_kind": True, "completed": 128, "bar": ref_bar,
         "verdict": execution_verdict(realistic_tps, 2.43, True, 128, ref_bar)},
        {"label": "modeled realistic gain, PPL at gate (2.42), same-kind, 128/128, ref bar +0.26",
         "tps": realistic_tps, "ppl": PPL_GATE, "same_kind": True, "completed": 128, "bar": ref_bar,
         "verdict": execution_verdict(realistic_tps, PPL_GATE, True, 128, ref_bar)},
        {"label": "modeled realistic gain, PPL-neutral, NEW-kind break, 128/128, ref bar +0.26",
         "tps": realistic_tps, "ppl": PPL_ANCHOR, "same_kind": False, "completed": 128, "bar": ref_bar,
         "verdict": execution_verdict(realistic_tps, PPL_ANCHOR, False, 128, ref_bar)},
        {"label": "modeled realistic gain, PPL-neutral, same-kind, 128/128, bar +15 (in [12.23 CI, 17.05 pt])",
         "tps": realistic_tps, "ppl": PPL_ANCHOR, "same_kind": True, "completed": 128, "bar": 15.0,
         "verdict": execution_verdict(realistic_tps, PPL_ANCHOR, True, 128, 15.0)},
        {"label": "modeled realistic gain, PPL-neutral, same-kind, ONLY 120/128 completed, ref bar +0.26",
         "tps": realistic_tps, "ppl": PPL_ANCHOR, "same_kind": True, "completed": 120, "bar": ref_bar,
         "verdict": execution_verdict(realistic_tps, PPL_ANCHOR, True, 120, ref_bar)},
        {"label": "CEILING gain +29.34, PPL-neutral, same-kind, 128/128, aggressive bar +20 (24.53 CI >= 20)",
         "tps": ceiling_tps, "ppl": PPL_ANCHOR, "same_kind": True, "completed": 128, "bar": 20.0,
         "verdict": execution_verdict(ceiling_tps, PPL_ANCHOR, True, 128, 20.0)},
    ]
    return {
        "one_to_one_slot_map": one_to_one,
        "execution_verdict_signature": ("execution_verdict(measured_tps, measured_ppl, flip_kind_same, "
                                        "completed, human_bar_tps, k=1)"),
        "action_space": ["SHIP-READY", "HOLD", "ROLLBACK"],
        "wraps": "#462 recommend() (imported verbatim): GO->SHIP-READY, CI-AMBIGUOUS->HOLD, NO-GO->ROLLBACK; "
                 "plus the 128/128 validity precondition (< 128 -> ROLLBACK).",
        "handed_to_fern_357": ("fern #357 presents the one-screen GO/NO-GO + this skeleton as the "
                               "'if GO, here is the exact next action'. This card does NOT duplicate the "
                               "capstone verdict — it produces the execution wrapper only."),
        "worked_corners": corners,
        "live_status": ("the moment stark #452 reports (TPS, PPL, flip-kind, N, completed), each fills a "
                        "slot and execution_verdict() returns the ACTION. RESOLVED 2026-06-16: stark #452 "
                        "LANDED (466.20 TPS dominated, identity 0.730, cascading new-KIND) -> bar-INVARIANT "
                        "ROLLBACK (see realized_stark452). The generic skeleton would resolve to SHIP-READY "
                        "only on PPL <= 2.42 + same-kind + 128/128 at a human bar <= +12.23; the realized "
                        "config clears NONE of {TPS, KIND}, so the gate refuses it."),
    }


# --------------------------------------------------------------------------- #
# stark #452 LANDED — the realized relax config dropped into the slots. NEGATIVE -> gate ROLLBACK.
# The parameterized <PENDING> form (above) is KEPT; this is the real-data overlay that exercises it.
# --------------------------------------------------------------------------- #
def realized_stark452() -> dict[str, Any]:
    """stark #452 (daqrzr99 / 00ovtdnt, MERGED, W&B-verified) realized the relax split-K prize and it
    is DOMINATED + identity-breaking. Drop the realized numbers through the SHIPPED execution_verdict()
    and the gate returns a bar-INVARIANT ROLLBACK — the real-data proof the GO-branch refuses to ship a
    dominated config. NOTE: PPL (the #462 most-sensitive input) PASSES (2.3782 <= 2.42); it is the
    TPS-domination + KIND clauses that catch it. The advisor's note said 'HOLD'; mechanically the gate
    returns the HARD ROLLBACK because same_kind=False forecloses CI-AMBIGUOUS/HOLD — both sit inside the
    'never SHIP-READY' envelope, and ROLLBACK is the precise machine output."""
    gain_dep = STARK452_REALIZED_TPS - DEPLOYED_TPS              # -15.33 over deployed
    gain_strict = STARK452_REALIZED_TPS - STRICT_FRONTIER_TPS    # -0.94 over strict frontier
    ref_bar = BEST_BYTE_EXACT_LEVER_TPS                          # +0.26 illustrative human bar

    # the live verdict from the SAME execution_verdict() the skeleton ships — at the ref bar AND the
    # most-lenient bar B=0; both ROLLBACK (the verdict is bar-INVARIANT on a dominated, new-kind config).
    verdict_ref = execution_verdict(STARK452_REALIZED_TPS, STARK452_REALIZED_PPL,
                                    STARK452_REALIZED_SAME_KIND, STARK452_REALIZED_COMPLETED, ref_bar)
    verdict_b0 = execution_verdict(STARK452_REALIZED_TPS, STARK452_REALIZED_PPL,
                                   STARK452_REALIZED_SAME_KIND, STARK452_REALIZED_COMPLETED, 0.0)

    # per-clause evaluation of the (2) harness against the realized numbers (which clause caught it).
    clause_a_pass = bool((gain_dep - SIGMA_HW) >= ref_bar)      # TPS CI-clean over ref bar -> FALSE
    clause_b_pass = bool(STARK452_REALIZED_PPL <= PPL_GATE)     # PPL hard gate -> TRUE (passes!)
    clause_c_pass = bool(STARK452_REALIZED_SAME_KIND)           # same-KIND -> FALSE (cascading divergence)
    clause_d_pass = bool(STARK452_REALIZED_COMPLETED >= 128)    # 128/128 -> TRUE
    clauses_pass = {"a_tps_ci_clean_over_bar": clause_a_pass, "b_ppl_le_gate": clause_b_pass,
                    "c_flip_kind_same_family": clause_c_pass, "d_completed_128_of_128": clause_d_pass}

    return {
        "stark452_landed": STARK452_LANDED,
        "stark452_run_ids": list(STARK452_RUN_IDS),
        "realized_measurement": {
            "relax_tps": STARK452_REALIZED_TPS,
            "gain_over_deployed_481_53": round(gain_dep, 4),
            "gain_over_strict_frontier_467_14": round(gain_strict, 4),
            "relax_ppl": STARK452_REALIZED_PPL,
            "ppl_delta_from_anchor": round(STARK452_REALIZED_PPL - PPL_ANCHOR, 4),
            "identity": STARK452_REALIZED_IDENTITY,
            "deployed_identity_for_ref": DEPLOYED_IDENTITY,
            "flips": STARK452_REALIZED_FLIPS,
            "deployed_flips_for_ref": DEPLOYED_FLIPS,
            "flip_kind": "cascading free-running divergence (NOT same-family accumulation-order near-ties)",
            "same_kind": STARK452_REALIZED_SAME_KIND,
            "completed": STARK452_REALIZED_COMPLETED,
        },
        "slots_filled": {
            "spec.knob_B.realized_kernel_artifact_ref": "stark #452 split-K wheel (daqrzr99/00ovtdnt)",
            "harness.clauses[0].measured_slot__tps": STARK452_REALIZED_TPS,
            "harness.clauses[1].measured_slot__ppl": STARK452_REALIZED_PPL,
            "harness.clauses[2].measured_slot__kind": "cascading divergence (same_kind=False)",
            "harness.clauses[2].measured_count_slot_orthogonal__N": STARK452_REALIZED_FLIPS,
            "harness.clauses[3].measured_slot__completed": STARK452_REALIZED_COMPLETED,
        },
        "harness_clause_eval": clauses_pass,
        "harness_clauses_failed": [k for k, v in clauses_pass.items() if not v],
        "ppl_alone_would_pass": clause_b_pass,    # the validation point: PPL did NOT catch this
        "caught_by": ["clause-a (TPS dominated; gain -15.33 <= 0 < any bar; not CI-clean of any B)",
                      "clause-c (new-KIND cascading divergence; not same-family near-ties)"],
        "live_verdict": verdict_ref,                  # ROLLBACK
        "live_verdict_at_bar_zero": verdict_b0,       # ROLLBACK (bar-invariant)
        "verdict_is_bar_invariant_rollback": bool(verdict_ref == "ROLLBACK" and verdict_b0 == "ROLLBACK"),
        "never_ship_ready": bool(verdict_ref != "SHIP-READY" and verdict_b0 != "SHIP-READY"),
        "hold_vs_rollback": (
            "the gate returns the HARD ROLLBACK, strictly stronger than HOLD. HOLD (== recommend() "
            "CI-AMBIGUOUS) is IMPOSSIBLE here: CI-AMBIGUOUS requires same_kind=True, but the realized "
            "flip-KIND is cascading divergence (same_kind=False). The config also fails clause-a (gain "
            f"{gain_dep:.2f} <= 0, dominated). Two independent hard NO-GO clauses -> ROLLBACK. Both 'HOLD' "
            "and 'ROLLBACK' sit inside the advisor's 'never SHIP-READY' envelope; the precise machine "
            "output is ROLLBACK."),
        "relax_lane_status": "CLOSED",
        "interpretation": (
            "REAL-DATA VALIDATION of the execution gate: the relax split-K re-partition was built and "
            "benchmarked (stark #452) and came back DOMINATED (466.20 < deployed 481.53, gain -15.33; "
            "< strict 467.14 too) with a cascading identity collapse (0.9966 -> 0.730, 3 -> 3317 flips). "
            "The skeleton's execution_verdict() — the exact function it ships — returns ROLLBACK, refusing "
            "to ship a dominated, identity-breaking config, INVARIANT to the human bar B (even B=0). "
            "Notably PPL (the #462 most-sensitive input) PASSED (2.3782 <= 2.42): PPL alone would have "
            "waved it through. The TPS-domination + KIND clauses are what caught it — vindicating the "
            "multi-clause design and the count-vs-KIND orthogonality (3317 flips is a count; the KIND is "
            "what gates). The relax lane is CLOSED; this card is now 'the execution gate, validated "
            "against the config that tried and failed to clear it.'"),
    }


# --------------------------------------------------------------------------- #
# Slot accounting — count the <PENDING #452> sentinel leaves in the spec structure (PRIMARY metric).
# --------------------------------------------------------------------------- #
def _pending_slot_paths(node: Any, p: str = "spec") -> list[str]:
    paths: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            paths += _pending_slot_paths(v, f"{p}.{k}")
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            paths += _pending_slot_paths(v, f"{p}[{i}]")
    elif isinstance(node, str) and node == PENDING:
        paths.append(p)
    return paths


# --------------------------------------------------------------------------- #
# (5) Self-test (PRIMARY gate).
# --------------------------------------------------------------------------- #
def selftests(banked: dict, spec: dict, harness: dict, rollback: dict, wire: dict,
              realized: dict, slot_paths: list[str]) -> dict[str, Any]:
    # (a) every banked source number round-trips its committed JSON within tol.
    cond_a = bool(banked["all_roundtrip_ok"])

    # (b) the parent cards this skeleton stands on are themselves green.
    cond_b = bool(banked["decision_surface_self_test_passed"] and banked["cost_ledger_self_test_passed"])

    # (c) the knob surface this spec names matches the committed #448 audit: served kernel = Marlin,
    #     served default has use_fp32_reduce=True, NO byte-exact selectable headroom, prize needs a build.
    cond_c = bool(
        banked["served_kernel_banked"] == SERVED_KERNEL
        and "use_fp32_reduce=True" in banked["served_default_config_banked"]
        and banked["has_byteexact_selectable_headroom_banked"] is False
        and banked["faster_int4_gemm_requires_build_banked"] is True
        and banked["fp32reduce_off_is_byteexact_banked"] is False
    )

    # (d) the spec carries the in-tree proxy (knob A) AND the build-level prize (knob B), with the proxy
    #     correctly demoted (only +0.64 UB) and the prize correctly requiring a build.
    kA, kB = spec["knob_A_in_tree_python_proxy"], spec["knob_B_build_level_split_k_repartition"]
    cond_d = bool(
        "use_fp32_reduce = True" in kA["current_value"]
        and "use_fp32_reduce = False" in kA["proposed_value"]
        and abs(kA["realized_endtoend_tps_delta"] - FP32OFF_PROXY_UB_TPS) <= TOL_RT
        and kB["requires_build"] is True
        and kB["greedy_safety"].startswith("REASSOCIATING")
    )

    # (e) PRIMARY: exactly the enumerated <PENDING #452> slots, count > 0 and == len(enumerated).
    #     5 realized-config slots (the served-file change spec) + 5 measurement slots (the harness).
    expected_slots = {
        # served-file change spec: the realized geometry stark #452's build confirms
        "spec.knob_A_in_tree_python_proxy.exact_realized_setting",
        "spec.knob_B_build_level_split_k_repartition.proposed_split_k_partition",
        "spec.knob_B_build_level_split_k_repartition.proposed_block_k",
        "spec.knob_B_build_level_split_k_repartition.proposed_num_warps",
        "spec.knob_B_build_level_split_k_repartition.realized_kernel_artifact_ref",
        # validation harness: the values stark #452 measures on the realized config
        "harness.clauses[0].measured_slot",                  # measured_relax_tps
        "harness.clauses[1].measured_slot",                  # measured_relax_ppl
        "harness.clauses[2].measured_slot",                  # measured_relax_flip_kind
        "harness.clauses[2].measured_count_slot_orthogonal",  # measured_relax_flip_count N (orthogonal)
        "harness.clauses[3].measured_slot",                  # completed (128/128)
    }
    got_slots = set(slot_paths)
    n_slots = len(slot_paths)
    cond_e = bool(n_slots == len(expected_slots) and got_slots == expected_slots and n_slots > 0)

    # (f) the harness is exactly 4 clauses and each maps to a recommend() clause (the precondition counts).
    cl = harness["clauses"]
    cond_f = bool(
        harness["n_clauses"] == 4
        and all(isinstance(c.get("recommend_clause"), str) and c["recommend_clause"] for c in cl)
        and cl[0]["recommend_clause"].startswith("clause-1")
        and cl[1]["recommend_clause"].startswith("clause-2")
        and cl[2]["recommend_clause"].startswith("clause-3")
        and "PRECONDITION" in cl[3]["recommend_clause"]
    )

    # (g) execution_verdict wraps recommend() faithfully: the action is exactly the GO/AMBIG/NO-GO image
    #     plus the 128/128 precondition. Probe the full 3-way mapping + the precondition.
    ref = BEST_BYTE_EXACT_LEVER_TPS
    realistic_tps = DEPLOYED_TPS + RELAX_REALISTIC_TPS_GAIN
    ceiling_tps = DEPLOYED_TPS + RELAX_CEILING_TPS_GAIN
    cond_g = bool(
        execution_verdict(realistic_tps, PPL_ANCHOR, True, 128, ref) == "SHIP-READY"   # GO
        and execution_verdict(realistic_tps, 2.50, True, 128, ref) == "ROLLBACK"        # NO-GO (ppl)
        and execution_verdict(realistic_tps, PPL_ANCHOR, False, 128, ref) == "ROLLBACK" # NO-GO (kind)
        and execution_verdict(realistic_tps, PPL_ANCHOR, True, 128, 15.0) == "HOLD"      # CI-AMBIGUOUS
        and execution_verdict(realistic_tps, PPL_ANCHOR, True, 120, ref) == "ROLLBACK"   # < 128/128
        and execution_verdict(ceiling_tps, PPL_ANCHOR, True, 128, 20.0) == "SHIP-READY"  # GO ceiling@+20
    )

    # (h) the explicit cross-check that execution_verdict's GO/AMBIG/NO-GO image == ds.recommend() for a
    #     grid of (gain, ppl, kind, bar) at 128/128 — i.e. no decision logic was re-implemented.
    grid_ok = True
    for gain in (RELAX_REALISTIC_TPS_GAIN, RELAX_CEILING_TPS_GAIN, 5.0, 0.0):
        for ppl in (PPL_ANCHOR, PPL_GATE, 2.43):
            for kind in (True, False):
                for bar in (ref, 15.0, 20.0):
                    want = {"GO": "SHIP-READY", "CI-AMBIGUOUS": "HOLD", "NO-GO": "ROLLBACK"}[
                        ds.recommend(gain, ppl, kind, bar, K_HEADLINE)]
                    got = execution_verdict(DEPLOYED_TPS + gain, ppl, kind, 128, bar)
                    if want != got:
                        grid_ok = False
    cond_h = bool(grid_ok)

    # (i) rollback is single-knob byte-for-byte reversible, with all 4 triggers present.
    cond_i = bool(
        rollback["relax_change_is_single_knob_reversible"] is True
        and rollback["revert_mechanism"]["byte_for_byte_return_to_deployed"] is True
        and len(rollback["rollback_triggers_any_of"]) == 4
    )

    # (j) human-gating asserted (Directive #3) and analysis-only flags set; ppl anchor preserved.
    cond_j = bool(
        "Directive #3" in spec["human_gated"]["directive"]
        and abs(PPL_ANCHOR - 2.3772) <= TOL_RT and PPL_ANCHOR <= PPL_GATE
    )

    # (k) NaN-clean — set by the caller after the full payload walk.
    cond_k = True

    # (l) stark #452 LANDED: the realized (dominated + new-KIND) config returns ROLLBACK from the SHIPPED
    #     execution_verdict() at the ref bar AND at the most-lenient bar B=0 (bar-invariant ROLLBACK);
    #     never SHIP-READY; clause-a (TPS) and clause-c (KIND) FAIL while clause-b (PPL) and clause-d
    #     (completion) PASS — the gate refused a config PPL alone would have waved through.
    rz = realized
    he = rz["harness_clause_eval"]
    cond_l = bool(
        rz["live_verdict"] == "ROLLBACK"
        and rz["live_verdict_at_bar_zero"] == "ROLLBACK"
        and rz["verdict_is_bar_invariant_rollback"] is True
        and rz["never_ship_ready"] is True
        and he["a_tps_ci_clean_over_bar"] is False
        and he["c_flip_kind_same_family"] is False
        and he["b_ppl_le_gate"] is True
        and he["d_completed_128_of_128"] is True
        and rz["relax_lane_status"] == "CLOSED"
    )

    conditions = {
        "a_all_banked_numbers_roundtrip": cond_a,
        "b_parent_cards_462_458_self_tests_green": cond_b,
        "c_knob_surface_matches_448_audit": cond_c,
        "d_spec_has_proxy_and_build_prize_correctly_demoted": cond_d,
        "e_pending_slots_exactly_enumerated": cond_e,
        "f_harness_4_clauses_map_to_recommend": cond_f,
        "g_execution_verdict_wraps_recommend": cond_g,
        "h_execution_verdict_equals_recommend_on_grid": cond_h,
        "i_rollback_single_knob_byte_for_byte": cond_i,
        "j_human_gated_directive3_ppl_anchor": cond_j,
        "k_nan_clean": cond_k,
        "l_realized_stark452_bar_invariant_rollback": cond_l,
    }
    return {
        "conditions": conditions,
        "relax_exec_self_test_passes": bool(all(conditions.values())),
        "detail": {
            "max_roundtrip_resid": banked["max_roundtrip_resid"],
            "relax_exec_spec_pending_slots": n_slots,
            "pending_slot_paths": sorted(slot_paths),
            "validation_checklist_clauses": harness["n_clauses"],
            "relax_change_is_single_knob_reversible": rollback["relax_change_is_single_knob_reversible"],
            "realized_live_verdict": realized["live_verdict"],
            "relax_lane_status": realized["relax_lane_status"],
            "realized_harness_clauses_failed": realized["harness_clauses_failed"],
        },
    }


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    banked = load_banked()
    spec = served_change_spec()
    harness = validation_harness_spec()
    rollback = rollback_spec()
    wire = prewire_stark452()
    realized = realized_stark452()   # stark #452 LANDED — the realized overlay (gate -> ROLLBACK)

    # count <PENDING #452> slots over the spec + harness (the two structures that carry slots).
    slot_paths = _pending_slot_paths(spec, "spec") + _pending_slot_paths(harness, "harness")
    st = selftests(banked, spec, harness, rollback, wire, realized, slot_paths)

    n_slots = len(slot_paths)
    exec_maps_to_recommend = bool(st["conditions"]["f_harness_4_clauses_map_to_recommend"]
                                  and st["conditions"]["g_execution_verdict_wraps_recommend"]
                                  and st["conditions"]["h_execution_verdict_equals_recommend_on_grid"])

    headline = {
        "relax_exec_self_test_passes": bool(st["relax_exec_self_test_passes"]),       # PRIMARY gate
        "relax_exec_spec_pending_slots": n_slots,                                      # PRIMARY 10
        "relax_change_is_single_knob_reversible": rollback["relax_change_is_single_knob_reversible"],
        "validation_checklist_clauses": harness["n_clauses"],                          # 4
        "exec_skeleton_maps_to_recommend": exec_maps_to_recommend,                     # bool
        "decision_flip_tps_threshold": DECISION_FLIP_TPS_THRESHOLD,                    # 12.2346 (#462)
        "max_admissible_relax_ppl": PPL_GATE,                                          # 2.42
        "ppl_gate_margin": PPL_GATE_MARGIN,                                            # 0.0428
        "prize_gain_realistic": RELAX_REALISTIC_TPS_GAIN,                              # +17.05
        "prize_gain_ceiling": RELAX_CEILING_TPS_GAIN,                                  # +29.34
        "fp32off_proxy_ub_tps": FP32OFF_PROXY_UB_TPS,                                  # +0.64 sub-prize
        "ppl": PPL_ANCHOR,
        "analysis_only": True,
        "no_served_file_change": True,
        "official_tps": 0,
        # --- stark #452 LANDED (realized overlay; PENDING form kept) ---
        "stark452_landed": realized["stark452_landed"],                                # True
        "relax_lane_status": realized["relax_lane_status"],                            # CLOSED
        "realized_live_verdict": realized["live_verdict"],                             # ROLLBACK
        "realized_verdict_is_bar_invariant_rollback": realized["verdict_is_bar_invariant_rollback"],
        "realized_relax_tps": STARK452_REALIZED_TPS,                                   # 466.20 (dominated)
        "realized_relax_gain_over_deployed": round(STARK452_REALIZED_TPS - DEPLOYED_TPS, 4),  # -15.33
        "realized_relax_ppl": STARK452_REALIZED_PPL,                                   # 2.3782 (passes gate)
        "realized_relax_identity": STARK452_REALIZED_IDENTITY,                         # 0.730 (collapsed)
        "realized_relax_flips": STARK452_REALIZED_FLIPS,                               # 3317
        "realized_ppl_alone_would_pass": realized["ppl_alone_would_pass"],             # True
    }
    verdict = (
        f"RELAX-EXEC-READINESS-{n_slots}-PENDING452-SLOTS-SINGLE-KNOB-REVERSIBLE-"
        f"{harness['n_clauses']}-CLAUSE-HARNESS-WRAPS-462-recommend()-HUMAN-GATED-D3"
        f" || STARK452-LANDED-NEGATIVE-RELAX-LANE-CLOSED-GATE-RETURNS-{realized['live_verdict']}"
        f"-(466.20-DOMINATED,IDENTITY-0.730,CASCADING-NEW-KIND;PPL-2.3782-PASSED-BUT-TPS+KIND-REFUSED)"
    )
    handoff = (
        f"EXECUTION SKELETON (for fern #357, the 'if GO, here is the action'): the relax-prize is the "
        f"int4-Marlin K-reduction split-K re-partition (+{RELAX_REALISTIC_TPS_GAIN:.1f}..+{RELAX_CEILING_TPS_GAIN:.1f} "
        f"over deployed {DEPLOYED_TPS}); NO Python knob -> needs a patched-Marlin wheel BUILD (the in-tree "
        f"`use_fp32_reduce=True->False` flag is only a +{FP32OFF_PROXY_UB_TPS:.2f} proxy). {n_slots} <PENDING #452> "
        f"slots. VALIDATE: (a) TPS CI-clean >= deployed+B at 1 sigma_hw [recommend clause-1]; (b) PPL <= "
        f"{PPL_GATE} [clause-2]; (c) same-KIND flips [clause-3]; (d) 128/128 [precondition]. execution_verdict() "
        f"= imported recommend() (GO->SHIP-READY / CI-AMBIGUOUS->HOLD / NO-GO->ROLLBACK) + 128/128 gate. "
        f"ROLLBACK = single-knob byte-for-byte return to {DEPLOYED_TPS}. Applying is HUMAN-GATED (Directive #3). "
        f"|| RESOLVED 2026-06-16: stark #452 LANDED NEGATIVE (466.20 TPS dominated, identity 0.730, cascading "
        f"new-KIND; PPL 2.3782 passed). The SHIPPED execution_verdict() returns {realized['live_verdict']} "
        f"(bar-invariant) — the relax lane is CLOSED. PPL alone would have waved it through; the TPS-domination "
        f"+ KIND clauses caught it. fern reads this as 'the gate, validated against the config that failed it.'"
    )
    return {
        "headline": headline,
        "served_change_spec": spec,
        "validation_harness_spec": harness,
        "rollback_spec": rollback,
        "prewire_stark452": wire,
        "realized_stark452": realized,
        "banked_roundtrip": banked,
        "self_test": st,
        "constants": {
            "deployed_tps": DEPLOYED_TPS, "sigma_hw": SIGMA_HW, "ppl_anchor": PPL_ANCHOR,
            "ppl_gate": PPL_GATE, "ppl_gate_margin": PPL_GATE_MARGIN,
            "relax_realistic_tps": RELAX_REALISTIC_TPS,
            "relax_realistic_tps_gain": RELAX_REALISTIC_TPS_GAIN,
            "relax_ceiling_tps_gain": RELAX_CEILING_TPS_GAIN,
            "decision_flip_tps_threshold": DECISION_FLIP_TPS_THRESHOLD,
            "best_byte_exact_lever_tps": BEST_BYTE_EXACT_LEVER_TPS,
            "deployed_flips": DEPLOYED_FLIPS, "k_headline": K_HEADLINE,
            "served_kernel": SERVED_KERNEL, "served_default_flag": SERVED_DEFAULT_FLAG,
            "fp32off_proxy_ub_tps": FP32OFF_PROXY_UB_TPS,
            "served_submission": SERVED_SUBMISSION, "served_manifest": SERVED_MANIFEST,
            "strict_frontier_tps": STRICT_FRONTIER_TPS,
            "stark452_realized_tps": STARK452_REALIZED_TPS,
            "stark452_realized_ppl": STARK452_REALIZED_PPL,
            "stark452_realized_identity": STARK452_REALIZED_IDENTITY,
            "stark452_realized_flips": STARK452_REALIZED_FLIPS,
            "stark452_realized_same_kind": STARK452_REALIZED_SAME_KIND,
            "stark452_run_ids": list(STARK452_RUN_IDS),
        },
        "verdict": verdict,
        "handoff_line": handoff,
        "imports": {
            "provenance": (
                "land#462 sb1n4aa6 (recommend() IMPORTED verbatim; decision_flip_tps_threshold 12.2346, "
                "max_admissible_relax_ppl 2.42, ppl margin 0.0428, most-sensitive ppl) x land#458 uhhyec0q "
                "(gains +17.05/+29.34, gate margin 0.0428, deployed-off-strict reframe) x land#457 h0uggl9i "
                "(= ubel #450 c5oyb7gv roofline: realistic split-K 498.58, ceiling 510.87) x stark#448 "
                "fn4iz0dz (served int4-GEMM knob audit: Marlin unique on sm86, use_fp32_reduce=True default, "
                "no byte-exact selectable headroom, prize requires build, fp32off UB +0.64) x stark#452 "
                "daqrzr99/00ovtdnt (LANDED NEGATIVE: realized relax 466.20 TPS dominated, identity 0.730, "
                "3317 cascading new-KIND flips, PPL 2.3782 passed -> gate ROLLBACK, relax lane CLOSED). The "
                "parameterized <PENDING #452> form is KEPT alongside the realized overlay. All run-ids in "
                "wandb-applied-ai-team/gemma-challenge-senpai."),
            "machinery": "imports #462 recommend(); round-trips committed #462/#458/#457/#448 JSONs; "
                         "re-derives nothing",
        },
    }


# --------------------------------------------------------------------------- #
# NaN guard + report + W&B (mirrors #462; never fatal).
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
    spec, harness = syn["served_change_spec"], syn["validation_harness_spec"]
    rollback, wire, st = syn["rollback_spec"], syn["prewire_stark452"], syn["self_test"]
    realized = syn["realized_stark452"]
    h = syn["headline"]
    print("\n" + "=" * 100, flush=True)
    print("RELAX-PATH EXECUTION-READINESS SKELETON (PR #465, land) — GO-branch pre-stage, CPU-only",
          flush=True)
    print("=" * 100, flush=True)
    print("  (1) SERVED-FILE CHANGE SPEC (parameterized; NO edit):", flush=True)
    kA, kB = spec["knob_A_in_tree_python_proxy"], spec["knob_B_build_level_split_k_repartition"]
    print(f"      PRIZE: {spec['prize_location']['kernel']} K-reduction split-K re-partition "
          f"(+{spec['prize_location']['prize_gain_realistic_over_deployed']:.2f} realistic / "
          f"+{spec['prize_location']['prize_gain_ceiling_over_deployed']:.2f} ceiling over {DEPLOYED_TPS})",
          flush=True)
    print(f"      knob A (in-tree PROXY): {kA['symbol'].split('(')[0].strip()}  "
          f"[{kA['current_value']}] -> [{kA['proposed_value']}]  (+{kA['realized_endtoend_tps_delta']:.2f} "
          f"UB, sub-prize)", flush=True)
    print(f"      knob B (BUILD prize): split-K/BLOCK_K/num_warps re-partition (NO Python knob; "
          f"requires_build={kB['requires_build']}); realized geometry = {PENDING}", flush=True)
    print(f"      human-gated: {spec['human_gated']['directive']}", flush=True)
    print("-" * 100, flush=True)
    print(f"  (2) VALIDATION HARNESS ({harness['n_clauses']} clauses; each -> stark #452 metric + "
          f"recommend() clause):", flush=True)
    for c in harness["clauses"]:
        print(f"      [{c['id']:<26}] {c['stark452_metric']:<42} -> {c['recommend_clause'][:40]}", flush=True)
    print("-" * 100, flush=True)
    print(f"  (3) ROLLBACK: single-knob reversible = {rollback['relax_change_is_single_knob_reversible']} "
          f"(byte-for-byte -> {DEPLOYED_TPS}). Triggers (any of):", flush=True)
    for t in rollback["rollback_triggers_any_of"]:
        print(f"        - {t['trigger']}", flush=True)
    print("-" * 100, flush=True)
    print("  (4) PRE-WIRED stark #452 -> execution_verdict (SHIP-READY / HOLD / ROLLBACK):", flush=True)
    for cn in wire["worked_corners"]:
        print(f"        [{cn['verdict']:<11}] {cn['label']}", flush=True)
    print(f"      live: {wire['live_status'][:92]}", flush=True)
    print("-" * 100, flush=True)
    rm = realized["realized_measurement"]
    print(f"  (R) STARK #452 LANDED ({'/'.join(realized['stark452_run_ids'])}) -> RELAX LANE "
          f"{realized['relax_lane_status']}.  GATE VERDICT = {realized['live_verdict']} (bar-invariant)",
          flush=True)
    print(f"      realized: {rm['relax_tps']} TPS (gain {rm['gain_over_deployed_481_53']:+.2f} vs deployed, "
          f"{rm['gain_over_strict_frontier_467_14']:+.2f} vs strict) | PPL {rm['relax_ppl']} "
          f"(delta {rm['ppl_delta_from_anchor']:+.4f}) | identity {rm['identity']} (was "
          f"{rm['deployed_identity_for_ref']}) | flips {rm['flips']} (was {rm['deployed_flips_for_ref']}) "
          f"| same_kind={rm['same_kind']}", flush=True)
    print(f"      harness eval: " + ", ".join(
        f"{k.split('_')[0]}={'PASS' if v else 'FAIL'}" for k, v in realized["harness_clause_eval"].items()),
        flush=True)
    print(f"      caught by {realized['caught_by']}; PPL alone would PASS "
          f"({realized['ppl_alone_would_pass']}) -> proof the multi-clause gate works.", flush=True)
    print("-" * 100, flush=True)
    print(f"  (5) PRIMARY relax_exec_self_test_passes = {st['relax_exec_self_test_passes']}; "
          f"pending_slots = {h['relax_exec_spec_pending_slots']}; "
          f"maps_to_recommend = {h['exec_skeleton_maps_to_recommend']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"        - {k}: {v}", flush=True)
    print("=" * 100, flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print(f"\n  CAPSTONE HANDOFF: {syn['handoff_line']}\n", flush=True)


def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[exec-readiness] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    h, harness, rollback = syn["headline"], syn["validation_harness_spec"], syn["rollback_spec"]
    st = syn["self_test"]
    realized = syn["realized_stark452"]
    run = init_wandb_run(
        job_type="relax-execution-readiness",
        agent="land",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["relax-execution-readiness", "equivalence-escalation-anchors", "go-branch-pre-stage",
              "served-change-spec", "validation-harness", "rollback-reversibility", "stark452-prewire",
              "wraps-462-recommend", "capstone-anchor", "analysis-only", "bank-the-analysis",
              "stark452-landed-negative", "relax-lane-closed", "gate-rollback", "real-data-validation"],
        config={
            "deployed_tps": DEPLOYED_TPS, "sigma_hw": SIGMA_HW, "ppl_anchor": PPL_ANCHOR,
            "ppl_gate": PPL_GATE, "ppl_gate_margin": PPL_GATE_MARGIN,
            "relax_realistic_tps": RELAX_REALISTIC_TPS,
            "relax_realistic_tps_gain": RELAX_REALISTIC_TPS_GAIN,
            "relax_ceiling_tps_gain": RELAX_CEILING_TPS_GAIN,
            "decision_flip_tps_threshold": DECISION_FLIP_TPS_THRESHOLD,
            "fp32off_proxy_ub_tps": FP32OFF_PROXY_UB_TPS,
            "served_kernel": SERVED_KERNEL, "served_default_flag": SERVED_DEFAULT_FLAG,
            "served_submission": SERVED_SUBMISSION,
            "k_headline": K_HEADLINE,
            "wandb_group": args.wandb_group,
            "source_runs": "land#462 sb1n4aa6 (recommend imported), land#458 uhhyec0q, land#457 h0uggl9i "
                           "(=ubel#450 c5oyb7gv roofline), stark#448 fn4iz0dz (knob audit), "
                           "stark#452 daqrzr99/00ovtdnt (LANDED NEGATIVE: 466.20 dominated, id 0.730, "
                           "cascading new-KIND -> gate ROLLBACK, relax lane CLOSED)",
        },
    )
    if run is None:
        print("[exec-readiness] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "relax_exec_self_test_passes": int(bool(st["relax_exec_self_test_passes"])),       # PRIMARY gate
        "relax_exec_spec_pending_slots": h["relax_exec_spec_pending_slots"],               # PRIMARY
        "relax_change_is_single_knob_reversible": int(bool(h["relax_change_is_single_knob_reversible"])),
        "validation_checklist_clauses": h["validation_checklist_clauses"],                 # 4
        "exec_skeleton_maps_to_recommend": int(bool(h["exec_skeleton_maps_to_recommend"])),
        "decision_flip_tps_threshold": h["decision_flip_tps_threshold"],
        "max_admissible_relax_ppl": h["max_admissible_relax_ppl"],
        "ppl_gate_margin": h["ppl_gate_margin"],
        "prize_gain_realistic": h["prize_gain_realistic"],
        "prize_gain_ceiling": h["prize_gain_ceiling"],
        "fp32off_proxy_ub_tps": h["fp32off_proxy_ub_tps"],
        "deployed_tps": DEPLOYED_TPS,
        "sigma_hw_tps": SIGMA_HW,
        "max_roundtrip_resid": syn["banked_roundtrip"]["max_roundtrip_resid"],
        "ppl": PPL_ANCHOR,
        "ppl_gate": PPL_GATE,
        "analysis_only": 1,
        "no_served_file_change": 1,
        "official_tps": 0,
        # --- stark #452 LANDED (realized overlay; relax lane closed) ---
        "stark452_landed": int(bool(realized["stark452_landed"])),
        "relax_lane_closed": int(realized["relax_lane_status"] == "CLOSED"),
        "realized_rolls_back": int(realized["live_verdict"] == "ROLLBACK"),
        "realized_verdict_is_bar_invariant_rollback": int(bool(realized["verdict_is_bar_invariant_rollback"])),
        "realized_relax_tps": STARK452_REALIZED_TPS,
        "realized_relax_gain_over_deployed": round(STARK452_REALIZED_TPS - DEPLOYED_TPS, 4),
        "realized_relax_gain_over_strict": round(STARK452_REALIZED_TPS - STRICT_FRONTIER_TPS, 4),
        "realized_relax_ppl": STARK452_REALIZED_PPL,
        "realized_relax_identity": STARK452_REALIZED_IDENTITY,
        "realized_relax_flips": STARK452_REALIZED_FLIPS,
        "realized_relax_same_kind": int(bool(STARK452_REALIZED_SAME_KIND)),
        "realized_ppl_alone_would_pass": int(bool(realized["ppl_alone_would_pass"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    try:
        run.summary["verdict"] = syn["verdict"]
        run.summary["handoff_line"] = syn["handoff_line"]
        run.summary["realized_live_verdict"] = realized["live_verdict"]
        run.summary["relax_lane_status"] = realized["relax_lane_status"]
    except Exception:
        pass
    log_json_artifact(run, name="relax_execution_readiness_result", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    print(f"[exec-readiness] wandb logged {len(summary)} keys; run id {getattr(run, 'id', '?')}",
          flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="equivalence-escalation-anchors")
    args = ap.parse_args(argv)

    syn = synthesize()

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 465, "agent": "land",
        "kind": "relax-execution-readiness", "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _nan_paths(payload)
    payload["nan_clean"] = not nan_paths
    syn["self_test"]["conditions"]["k_nan_clean"] = not nan_paths
    syn["self_test"]["relax_exec_self_test_passes"] = bool(
        all(syn["self_test"]["conditions"].values()))
    syn["headline"]["relax_exec_self_test_passes"] = syn["self_test"]["relax_exec_self_test_passes"]
    if nan_paths:
        print(f"[exec-readiness] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "relax_execution_readiness_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[exec-readiness] wrote {out_path}", flush=True)

    st_path = out_dir / "relax_execution_readiness_selftest.json"
    with st_path.open("w", encoding="utf-8") as fh:
        json.dump(syn["self_test"]["conditions"], fh, indent=2, sort_keys=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = (syn["self_test"]["relax_exec_self_test_passes"] and payload["nan_clean"])
        print(f"[exec-readiness] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
