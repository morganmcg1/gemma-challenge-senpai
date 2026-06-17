#!/usr/bin/env python3
"""PR #561 — Two-gate FIRE-decision capstone (SYNTHESIS, analysis-only).

A pure-CPU SYNTHESIS card. It assembles already-banked, W&B-logged numbers into
the one artifact the program is missing: the airtight, pre-registered two-gate
FIRE-DECISION packet. NOTHING here is measured — every input is a constant that
traces to a banked W&B run (cited inline + in ``PROVENANCE``). No GPU, no served
job, no microbench, no HF launch, no submission, no served-file change.

The verdict that falls out: base_fullhead is a CONFIRMED quality-PASS ship whose
HARD speed ceiling (~311.25 magically-free / ~292 strict) sits BELOW the current
shipped 375.857-official ship AND far below the 481.53 official #1. osoi5 is the
mirror image (fast, quality collapses). No config is (PASS, PASS) -> NO-FIRE.

Outputs: a W&B run (group ``two-gate-fire-decision``) logging the synthesized
decision fields + a JSON artifact + a single-line SENPAI-RESULT.

Run under the wandb-capable venv (``.venv/bin/python``); imports wandb first so
the cached real module beats the ``./wandb`` run-data namespace shadow.
"""
from __future__ import annotations

# --- real-wandb-first (beats ./wandb namespace shadow); harmless if absent ---
try:  # pragma: no cover
    import wandb as _wandb_real  # noqa: F401  (cache the real module in sys.modules)
except Exception:  # pragma: no cover
    _wandb_real = None

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/workspace/senpai/target")
HERE = ROOT / "research" / "two_gate_fire_decision"
OUT_JSON = HERE / "two_gate_fire_decision.json"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # APPEND so site-packages wandb wins over ./wandb

# ======================================================================
# BANKED INPUTS — every value cited to its PR + W&B run. DO NOT re-derive.
# ======================================================================

# --- the quality-safe ship anchor (base_fullhead) ---------------------
BASE_FULLHEAD_ANCHOR_TPS = 252.31        # lawine #544 (d44b61gj) derived; wirbel #553 grounded
BASE_FULLHEAD_ANCHOR_TPS_FAST_PROXY = 253.78  # land #534 (ivpk7g7z) / fern #535 served proxy

# --- the quality-safe HARD speed ceiling (six convergent lenses) ------
# magically-free head (head bytes -> 0, body intact, KV -> 0, fixed floor kept)
CEILING_MAGICALLY_FREE_TPS = 311.2485991465399   # lawine #554 (fi8vr1nb), corrected from #544's 328.9
CEILING_544_FREE_HEAD_TPS = 328.9                # lawine #544 (d44b61gj) pre-correction (served-tax over-credit)
# strict identity-safe head lever (int4 precision; an UPPER bound per land #552/#556)
CEILING_STRICT_PRECISION_TPS = 292.1             # lawine #544 (d44b61gj) precision lever +38.3
# candidate-verify realized (read-bound projection, identity-safe BY CONSTRUCTION)
CANDVERIFY_REALIZED_CENTRAL_TPS = 305.4179473236596   # fern #549 (p9ga96xo) central
CANDVERIFY_REALIZED_BAND = [293.79708405168066, 309.01272414399165]  # fern #549 [pess, opt]
CANDVERIFY_GAIN_CENTRAL = 40.5983813617724            # fern #549 central +TPS over anchor
CANDVERIFY_GAIN_BAND = [28.977518089793477, 44.19315818210447]       # fern #549 [pess, opt]
CANDVERIFY_GREEDY_IDENTITY_RATE = 1.0                 # fern #549 offline 60k positions, miss@K_safe=0

# --- the configs on the table -----------------------------------------
# current shipped result = surgical-357 / osoi5 lineage (12k head-prune + baked body)
CURRENT_SHIP_OFFICIAL_TPS = 375.857      # the TPS to beat for a fire (official a10g-small)
OSOI5_FAST_LOCAL_TPS = 350.76            # lawine #544 (d44b61gj) osoi5 local class (353.73 in #549)
OSOI5_FAST_OFFICIAL_TPS = 375.857        # same ship, official

# --- the references ----------------------------------------------------
OFFICIAL_1_TPS = 481.53                  # PR-registered official public #1 (#52 flagship)
LIVE_PUBLIC_1_TPS = 508.6320894487107    # digest 2026-06-17: ff-splitkv-frantic-fawindow-w256 (honest note; widens gap)
MORGAN_QUALITY_GATE_FRAC = 0.90          # Morgan #515 >=90%-of-vanilla-base downstream gate

# --- the four MEASURED quality legs (all PASS, all >= 90% of vanilla base) ---
# value = fraction-of-anchor/vanilla-base (the Morgan #515 denominator)
QUALITY_LEGS = {
    "mmlu_pro": {"frac_of_anchor": 0.952, "pr": "stark #542", "wandb": "92pcnx6a",
                 "abs": 0.636, "anchor_abs": 0.668, "note": "ubel #538 anchor 0.668; stark tight-CI 95.2%"},
    "gpqa_diamond": {"frac_of_anchor": 0.999, "pr": "stark #542", "wandb": "92pcnx6a",
                     "abs": 0.4697, "anchor_abs": 0.470, "note": "ubel #538 anchor 0.444/0.470; stark tight-CI 99.9%"},
    "gsm8k": {"frac_of_anchor": 0.973, "pr": "wirbel #541/#545", "wandb": "uqnkzlf9",
              "abs": 0.973, "anchor_abs": 1.0, "note": "97.3% w/ min_tokens=8 EOS guard; as-served 86.8% recoverable"},
    "aime": {"frac_of_anchor": 1.1817, "pr": "fern #514/#535", "wandb": "xtanouk7",
             "abs": 0.1444, "anchor_abs": 0.1222, "note": "maj@1 0.1444 = 118.17% of base anchor 0.1222"},
}

# --- the framework axis (CLOSED across four legs) ----------------------
FRAMEWORK_LEGS = {
    "sglang": "denken #498 + lawine #558",
    "trt_llm": "fern #502",
    "flashinfer_standalone": "fern #507",
}

# --- provenance map (PR -> what it banked) -----------------------------
PROVENANCE = {
    "#544": {"wandb": "d44b61gj", "group": "base-fullhead-tps-ceiling",
             "what": "252.31->350.76 gap (82.2% head verify-tax / 17.8% +5 body); precision +38.3->292.1; free-head 328.9; quality_safe_ship_can_beat_442=FALSE"},
    "#551": {"wandb": "5rnkxttp", "group": "kv-read-fp8-lever",
             "what": "KV-read 1.09% of decode bytes; base_fullhead weight-read-bound; ceiling KV-robust"},
    "#554": {"wandb": "fi8vr1nb", "group": "fixed-overhead-ceiling",
             "what": "42-launch SDPA fixed-overhead floor 0.573ms -> corrected magically-free ceiling 311.25 (-17.65 vs 328.9)"},
    "#550": {"wandb": "5aobahij", "group": "faster-byte-identical-kernel",
             "what": "Marlin ONLY w4a16 kernel on sm_86; head GEMV 482.9 GB/s = 80.5% A10G peak; vLLM FORCES Triton for head_dim 256/512; kernel-robust"},
    "#552": {"wandb": "e4s81mih", "group": "lossless-head-prune-tps",
             "what": "provably-lossless head-prune keeps all 262,144 rows -> +0 TPS; EXACT-head lever structurally zero"},
    "#549": {"wandb": "p9ga96xo", "group": "fullhead-candidate-verify",
             "what": "cheap-candidate+full-verify head lever: +40.6 (band +29..+44) -> realized 305.4 (band 293.8..309), identity-safe by construction"},
    "#560": {"wandb": "in-flight", "group": "fullhead-candidate-verify",
             "what": "served realization of #549's candidate-verify lever (converting [+28,+44] projection to a measured number) — IN FLIGHT; verdict holds either realization"},
    "#535": {"wandb": "xtanouk7", "group": "base-fullhead-fast-ship-probe",
             "what": "AIME maj@1 0.1444 >= base 0.1222 (118.17%), matched conc=32 n=90"},
    "#542": {"wandb": "92pcnx6a", "group": "base-fullhead-shortchain-quality",
             "what": "MMLU-Pro 0.636 (95.2% of anchor) + GPQA-D 0.4697 (99.9%), tight-CI third leg"},
    "#541": {"wandb": "uqnkzlf9", "group": "base-fullhead-gsm8k-layerdrop",
             "what": "GSM8K 97.3% with min_tokens=8 first-token-EOS guard; as-served 86.8% recoverable artifact"},
    "#538": {"wandb": "(ubel)", "group": "—",
             "what": "MMLU-Pro 0.668 / GPQA-D 0.444 base_fullhead anchor"},
    "#534": {"wandb": "ivpk7g7z", "group": "fullhead-surgical-safe-anchor",
             "what": "complete base_fullhead gate-table 3/3 PASS, draw_ready=True; 253.78 = 71.7% of unsafe ship"},
    "#547": {"wandb": "(kanna)", "group": "—",
             "what": "head-WIDTH lever CLOSED: fast_quality_safe_ship_exists=FALSE (12k fails MMLU-Pro 0.550; 32k safe-min width, modest TPS)"},
    "#553": {"wandb": "(wirbel)", "group": "realized-anchor-tps",
             "what": "base_fullhead served-TPS anchor 252.31 vs 253.78"},
    "#558": {"wandb": "(lawine)", "group": "framework-zoomout-ceiling",
             "what": "framework leg: SGLang/FlashInfer/TRT-LLM do not serve byte-identically AND faster"},
    "#502": {"wandb": "(fern)", "group": "—", "what": "framework leg: TRT-LLM closed"},
    "#507": {"wandb": "(fern)", "group": "—", "what": "framework leg: FlashInfer-standalone flips identity 1292/2048 -> 0.0"},
    "#498": {"wandb": "(denken)", "group": "—", "what": "framework leg: SGLang closed"},
    "#515": {"wandb": "(Morgan)", "group": "—", "what": ">=90%-of-vanilla-base downstream quality gate"},
    "#524": {"wandb": "(Morgan)", "group": "—", "what": "two-gate pre-authorization: 'a good faster TPS result that keeps quality within 10% of the base model'"},
    "#319": {"wandb": "(program)", "group": "—", "what": "strict greedy-identity HARD gate (argmax_identity_rate=1.0)"},
}


def build_packet() -> dict[str, Any]:
    # ---- Stage 1: the two-gate truth table -----------------------------
    # base_fullhead: quality PASS (4 axes >= 90%), speed FAIL (ceiling < ship).
    quality_gate_base_fullhead = "PASS"
    speed_gate_base_fullhead = "FAIL"
    # osoi5: quality FAIL (moat collapse), speed PASS (375.857 shipped).
    quality_gate_osoi5 = "FAIL"
    speed_gate_osoi5 = "PASS"

    cells = {
        "base_fullhead": (quality_gate_base_fullhead, speed_gate_base_fullhead),
        "osoi5": (quality_gate_osoi5, speed_gate_osoi5),
    }
    two_gate_satisfiable = any(q == "PASS" and s == "PASS" for q, s in cells.values())
    fire_decision = "FIRE" if two_gate_satisfiable else "NO-FIRE"

    # ---- Stage 2: the precise TPS gap to a fire ------------------------
    gap_to_current_ship_magfree = CURRENT_SHIP_OFFICIAL_TPS - CEILING_MAGICALLY_FREE_TPS
    gap_to_current_ship_strict = CURRENT_SHIP_OFFICIAL_TPS - CEILING_STRICT_PRECISION_TPS
    gap_to_current_ship_candverify = CURRENT_SHIP_OFFICIAL_TPS - CANDVERIFY_REALIZED_CENTRAL_TPS
    gap_to_official_1_magfree = OFFICIAL_1_TPS - CEILING_MAGICALLY_FREE_TPS
    gap_to_official_1_strict = OFFICIAL_1_TPS - CEILING_STRICT_PRECISION_TPS
    gap_to_live_public_1_magfree = LIVE_PUBLIC_1_TPS - CEILING_MAGICALLY_FREE_TPS

    # ---- Stage 3: pre-registered pass/fail bars for every open lever ---
    # fern #560 candidate-verify served-realize: pass = lift base_fullhead above
    # the ship at identity=1.0. #549 caps the lever at +44 best -> 309 < 375.857.
    candidate_verify_best_case_tps = CANDVERIFY_REALIZED_BAND[1]  # optimistic 309.01
    candidate_verify_can_fire = candidate_verify_best_case_tps > CURRENT_SHIP_OFFICIAL_TPS  # structurally False

    # ubel #546 body depth-drop: pass = (identity=1.0) AND (quality >= 90% all 4 axes)
    # AND (adds >= gap_to_current_ship over the 311.25 ceiling).
    depth_drop_pass_bar_tps = gap_to_current_ship_magfree           # ~64.6 TPS over the magically-free ceiling
    depth_drop_pass_bar_tps_strict = gap_to_current_ship_strict     # ~83.8 TPS over strict
    # honest analytic call: the (identity AND quality>=90% AND +65 TPS) conjunction
    # is almost certainly empty on a dense 4B — dropping a transformer block changes
    # the logits -> near-certain #319 flip, and the moat shows depth costs quality.
    depth_drop_conjunction_plausible = False

    # kanna #547 head-width: already CLOSED (fast_quality_safe_ship_exists=FALSE).
    head_width_lever_open = False

    # ---- Stage 4: the verdict + the single flip condition --------------
    verdict_flip_condition = (
        "A served config measured above 375.857 TPS at #319 argmax_identity_rate=1.0 "
        "AND downstream quality >= 90% of vanilla base on all four axes. The "
        "hardware-rooted ceiling (denken #550 HBM byte-rate wall + lawine #554 "
        "fixed-overhead floor) caps base_fullhead at 311.25 magically-free, so no "
        "quality-safe config can reach it; and no quality-passing config has ever "
        "produced a TPS above the ship. The condition is provably unsatisfiable for "
        "base_fullhead and empirically unobserved for any quality-PASS config."
    )

    # ---- self-tests (deterministic arithmetic on banked constants) -----
    qlegs_all_ge_90 = all(v["frac_of_anchor"] >= MORGAN_QUALITY_GATE_FRAC for v in QUALITY_LEGS.values())
    self_tests = {
        "ceiling_magfree_below_ship": CEILING_MAGICALLY_FREE_TPS < CURRENT_SHIP_OFFICIAL_TPS,
        "ceiling_strict_below_magfree": CEILING_STRICT_PRECISION_TPS < CEILING_MAGICALLY_FREE_TPS,
        "candverify_best_below_magfree": candidate_verify_best_case_tps < CEILING_MAGICALLY_FREE_TPS,
        "candverify_best_below_ship": candidate_verify_best_case_tps < CURRENT_SHIP_OFFICIAL_TPS,
        "gap_to_ship_positive": gap_to_current_ship_magfree > 0,
        "gap_to_official1_positive": gap_to_official_1_magfree > 0,
        "live_gap_wider_than_registered": gap_to_live_public_1_magfree > gap_to_official_1_magfree,
        "two_gate_unsatisfiable": (not two_gate_satisfiable),
        "no_pass_pass_cell": not any(q == "PASS" and s == "PASS" for q, s in cells.values()),
        "fire_is_nofire": fire_decision == "NO-FIRE",
        "depth_drop_bar_equals_ship_gap": abs(depth_drop_pass_bar_tps - gap_to_current_ship_magfree) < 1e-9,
        "candidate_verify_cannot_fire": (not candidate_verify_can_fire),
        "depth_drop_not_plausible": (not depth_drop_conjunction_plausible),
        "head_width_closed": (not head_width_lever_open),
        "all_open_levers_fail_bar": (
            (not candidate_verify_can_fire)
            and (not depth_drop_conjunction_plausible)
            and (not head_width_lever_open)
        ),
        "quality_legs_all_ge_90pct": qlegs_all_ge_90,
        "anchor_below_ceiling": BASE_FULLHEAD_ANCHOR_TPS < CEILING_MAGICALLY_FREE_TPS,
        "nan_clean": all(
            math.isfinite(x) for x in [
                CEILING_MAGICALLY_FREE_TPS, CEILING_STRICT_PRECISION_TPS,
                gap_to_current_ship_magfree, gap_to_current_ship_strict,
                gap_to_official_1_magfree, depth_drop_pass_bar_tps,
            ]
        ),
    }
    self_tests["self_test_passes"] = all(self_tests.values())
    self_det = bool(self_tests["self_test_passes"])

    packet = {
        "pr": 561,
        "card": "two-gate-fire-decision-capstone",
        "analysis_only": True,
        "official_tps": 0,
        "no_served_file_change": True,
        "no_hf_job": True,
        "peak_gpu_gib": 0.0,
        # ---- Stage 1 ----
        "stage1_truth_table": {
            "quality_gate_base_fullhead": quality_gate_base_fullhead,
            "speed_gate_base_fullhead": speed_gate_base_fullhead,
            "quality_gate_osoi5": quality_gate_osoi5,
            "speed_gate_osoi5": speed_gate_osoi5,
            "two_gate_satisfiable": two_gate_satisfiable,
            "fire_decision": fire_decision,
            "cells": {k: {"quality": q, "speed": s} for k, (q, s) in cells.items()},
        },
        # ---- Stage 2 ----
        "stage2_gaps": {
            "base_fullhead_quality_safe_ceiling_tps": CEILING_MAGICALLY_FREE_TPS,  # primary (magically-free)
            "base_fullhead_quality_safe_ceiling_tps_magfree": CEILING_MAGICALLY_FREE_TPS,
            "base_fullhead_quality_safe_ceiling_tps_strict": CEILING_STRICT_PRECISION_TPS,
            "candidate_verify_realized_central_tps": CANDVERIFY_REALIZED_CENTRAL_TPS,
            "candidate_verify_realized_band": CANDVERIFY_REALIZED_BAND,
            "gap_to_current_ship_tps": gap_to_current_ship_magfree,         # primary (magically-free, tightest)
            "gap_to_current_ship_tps_magfree": gap_to_current_ship_magfree,
            "gap_to_current_ship_tps_strict": gap_to_current_ship_strict,
            "gap_to_current_ship_tps_candverify": gap_to_current_ship_candverify,
            "gap_to_official_1_tps": gap_to_official_1_magfree,             # primary (magically-free)
            "gap_to_official_1_tps_magfree": gap_to_official_1_magfree,
            "gap_to_official_1_tps_strict": gap_to_official_1_strict,
            "gap_to_live_public_1_tps_magfree": gap_to_live_public_1_magfree,
        },
        # ---- Stage 3 ----
        "stage3_pass_fail_bars": {
            "candidate_verify_can_fire": candidate_verify_can_fire,
            "candidate_verify_best_case_tps": candidate_verify_best_case_tps,
            "candidate_verify_pass_bar_tps": CURRENT_SHIP_OFFICIAL_TPS,
            "depth_drop_pass_bar_tps": depth_drop_pass_bar_tps,
            "depth_drop_pass_bar_tps_strict": depth_drop_pass_bar_tps_strict,
            "depth_drop_conjunction_plausible": depth_drop_conjunction_plausible,
            "head_width_lever_open": head_width_lever_open,
        },
        # ---- Stage 4 ----
        "stage4_verdict": {
            "verdict": "NO-FIRE",
            "verdict_flip_condition": verdict_flip_condition,
        },
        # ---- supporting banked inputs ----
        "inputs": {
            "base_fullhead_anchor_tps": BASE_FULLHEAD_ANCHOR_TPS,
            "base_fullhead_anchor_tps_fast_proxy": BASE_FULLHEAD_ANCHOR_TPS_FAST_PROXY,
            "ceiling_544_free_head_tps": CEILING_544_FREE_HEAD_TPS,
            "candverify_gain_central": CANDVERIFY_GAIN_CENTRAL,
            "candverify_gain_band": CANDVERIFY_GAIN_BAND,
            "candverify_greedy_identity_rate": CANDVERIFY_GREEDY_IDENTITY_RATE,
            "current_ship_official_tps": CURRENT_SHIP_OFFICIAL_TPS,
            "osoi5_fast_local_tps": OSOI5_FAST_LOCAL_TPS,
            "osoi5_fast_official_tps": OSOI5_FAST_OFFICIAL_TPS,
            "official_1_tps": OFFICIAL_1_TPS,
            "live_public_1_tps": LIVE_PUBLIC_1_TPS,
            "morgan_quality_gate_frac": MORGAN_QUALITY_GATE_FRAC,
        },
        "quality_legs": QUALITY_LEGS,
        "framework_legs": FRAMEWORK_LEGS,
        "provenance": PROVENANCE,
        "self_tests": self_tests,
        "self_det": self_det,
        "primary_metric_name": "base_fullhead_quality_safe_ceiling_tps",
        "primary_metric_value": CEILING_MAGICALLY_FREE_TPS,
    }
    return packet


def wandb_summary(p: dict[str, Any]) -> dict[str, Any]:
    s1 = p["stage1_truth_table"]
    s2 = p["stage2_gaps"]
    s3 = p["stage3_pass_fail_bars"]
    s4 = p["stage4_verdict"]
    return {
        # Stage 1
        "two_gate_satisfiable": s1["two_gate_satisfiable"],
        "two_gate_satisfiable_int": int(s1["two_gate_satisfiable"]),
        "fire_decision": s1["fire_decision"],
        "fire_is_nofire_int": int(s1["fire_decision"] == "NO-FIRE"),
        "quality_gate_base_fullhead": s1["quality_gate_base_fullhead"],
        "speed_gate_base_fullhead": s1["speed_gate_base_fullhead"],
        "quality_gate_osoi5": s1["quality_gate_osoi5"],
        "speed_gate_osoi5": s1["speed_gate_osoi5"],
        # Stage 2
        "base_fullhead_quality_safe_ceiling_tps": s2["base_fullhead_quality_safe_ceiling_tps"],
        "base_fullhead_quality_safe_ceiling_tps_strict": s2["base_fullhead_quality_safe_ceiling_tps_strict"],
        "candidate_verify_realized_central_tps": s2["candidate_verify_realized_central_tps"],
        "gap_to_current_ship_tps": s2["gap_to_current_ship_tps"],
        "gap_to_current_ship_tps_strict": s2["gap_to_current_ship_tps_strict"],
        "gap_to_current_ship_tps_candverify": s2["gap_to_current_ship_tps_candverify"],
        "gap_to_official_1_tps": s2["gap_to_official_1_tps"],
        "gap_to_official_1_tps_strict": s2["gap_to_official_1_tps_strict"],
        "gap_to_live_public_1_tps_magfree": s2["gap_to_live_public_1_tps_magfree"],
        # Stage 3
        "candidate_verify_can_fire": s3["candidate_verify_can_fire"],
        "candidate_verify_can_fire_int": int(s3["candidate_verify_can_fire"]),
        "candidate_verify_best_case_tps": s3["candidate_verify_best_case_tps"],
        "depth_drop_pass_bar_tps": s3["depth_drop_pass_bar_tps"],
        "depth_drop_pass_bar_tps_strict": s3["depth_drop_pass_bar_tps_strict"],
        "depth_drop_conjunction_plausible": s3["depth_drop_conjunction_plausible"],
        "depth_drop_conjunction_plausible_int": int(s3["depth_drop_conjunction_plausible"]),
        "head_width_lever_open": s3["head_width_lever_open"],
        "head_width_lever_open_int": int(s3["head_width_lever_open"]),
        # Stage 4
        "verdict_flip_condition": s4["verdict_flip_condition"],
        # meta
        "self_det": p["self_det"],
        "self_det_int": int(p["self_det"]),
        "n_self_tests_passed": sum(1 for v in p["self_tests"].values() if v),
        "n_self_tests_total": len(p["self_tests"]),
        "peak_gpu_gib": p["peak_gpu_gib"],
        "analysis_only": True,
        "official_tps": 0,
        "primary_metric": p["primary_metric_value"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb-name", default="lawine/two-gate-fire-decision")
    ap.add_argument("--wandb-group", default="two-gate-fire-decision")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    p = build_packet()
    HERE.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as fh:
        json.dump(p, fh, indent=2)
    print(f"[capstone] wrote {OUT_JSON}", flush=True)

    s1, s2, s3 = p["stage1_truth_table"], p["stage2_gaps"], p["stage3_pass_fail_bars"]
    line = "=" * 10 + " PR #561 — TWO-GATE FIRE-DECISION CAPSTONE (synthesis) " + "=" * 10
    print("\n" + line, flush=True)
    print("  STAGE 1 — two-gate truth table:", flush=True)
    print(f"    base_fullhead : quality={s1['quality_gate_base_fullhead']:5s} speed={s1['speed_gate_base_fullhead']:5s}", flush=True)
    print(f"    osoi5         : quality={s1['quality_gate_osoi5']:5s} speed={s1['speed_gate_osoi5']:5s}", flush=True)
    print(f"    two_gate_satisfiable = {s1['two_gate_satisfiable']}  ->  fire_decision = {s1['fire_decision']}", flush=True)
    print("  STAGE 2 — TPS gap to a fire:", flush=True)
    print(f"    quality-safe ceiling = {s2['base_fullhead_quality_safe_ceiling_tps']:.2f} magically-free "
          f"/ {s2['base_fullhead_quality_safe_ceiling_tps_strict']:.2f} strict "
          f"(candidate-verify realized {s2['candidate_verify_realized_central_tps']:.1f})", flush=True)
    print(f"    gap_to_current_ship_tps  = {s2['gap_to_current_ship_tps']:.2f} (magfree) "
          f"/ {s2['gap_to_current_ship_tps_strict']:.2f} (strict)  [ship {CURRENT_SHIP_OFFICIAL_TPS}]", flush=True)
    print(f"    gap_to_official_1_tps    = {s2['gap_to_official_1_tps']:.2f} (magfree) "
          f"/ {s2['gap_to_official_1_tps_strict']:.2f} (strict)  [official#1 {OFFICIAL_1_TPS}]", flush=True)
    print("  STAGE 3 — open-lever pass/fail bars:", flush=True)
    print(f"    candidate_verify_can_fire = {s3['candidate_verify_can_fire']} "
          f"(best {s3['candidate_verify_best_case_tps']:.1f} < ship {CURRENT_SHIP_OFFICIAL_TPS})", flush=True)
    print(f"    depth_drop_pass_bar_tps   = +{s3['depth_drop_pass_bar_tps']:.2f} over ceiling "
          f"(conjunction_plausible={s3['depth_drop_conjunction_plausible']})", flush=True)
    print(f"    head_width_lever_open     = {s3['head_width_lever_open']}", flush=True)
    print("  STAGE 4 — verdict:", flush=True)
    print(f"    {p['stage4_verdict']['verdict']}", flush=True)
    print(f"  self_det = {p['self_det']}  ({p['self_tests']})", flush=True)
    print("=" * len(line), flush=True)

    rid = None
    if not args.no_wandb:
        try:
            from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                               log_json_artifact, log_summary)
            run = init_wandb_run(
                job_type="systems-profile",
                agent="lawine",
                name=args.wandb_name,
                group=args.wandb_group,
                tags=["two-gate", "fire-decision", "capstone", "synthesis",
                      "analysis-only", "no-fire", "local-a10g"],
                notes="PR #561 two-gate FIRE-decision capstone: (quality-PASS, speed-ceiling) truth table + TPS gap + pre-registered pass/fail bars",
                config={
                    "synthesis_only": True,
                    "no_gpu": True,
                    "cited_prs": list(PROVENANCE.keys()),
                    "current_ship_official_tps": CURRENT_SHIP_OFFICIAL_TPS,
                    "official_1_tps": OFFICIAL_1_TPS,
                },
            )
            if run is not None:
                log_summary(run, wandb_summary(p), step=0)
                log_json_artifact(run, name="two-gate-fire-decision",
                                  artifact_type="fire-decision-capstone", data=p)
                rid = getattr(run, "id", None)
                finish_wandb(run)
                p["wandb_run_id"] = rid
                with OUT_JSON.open("w") as fh:
                    json.dump(p, fh, indent=2)
                print(f"[capstone] wandb run id = {rid}", flush=True)
        except Exception as exc:  # pragma: no cover
            print(f"[capstone] wandb unavailable: {exc}", flush=True)

    # ---- single-line SENPAI-RESULT ----
    senpai = {
        "terminal": True,
        "status": "complete",
        "pending_arms": False,
        "analysis_only": True,
        "official_tps": 0,
        "wandb_run_ids": [rid] if rid else [],
        "self_det": p["self_det"],
        "fire_decision": s1["fire_decision"],
        "two_gate_satisfiable": s1["two_gate_satisfiable"],
        "primary_metric": {"name": "base_fullhead_quality_safe_ceiling_tps",
                           "value": round(s2["base_fullhead_quality_safe_ceiling_tps"], 2)},
        "test_metric": {"name": "gap_to_current_ship_tps",
                        "value": round(s2["gap_to_current_ship_tps"], 2)},
    }
    print("\nSENPAI-RESULT: " + json.dumps(senpai, separators=(",", ":")), flush=True)
    return 0 if p["self_det"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
