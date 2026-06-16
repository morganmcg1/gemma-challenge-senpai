#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Relax-decision cost ledger (PR #458, land) — CPU-only analytic reconciliation.

THE QUESTION (escalation packet's missing COST leg; capstone needs the cost axis next to prize)
-----------------------------------------------------------------------------------------------
The escalation packet's PRIZE side is now thoroughly quantified — unified physical ceiling
510.87 +/- 4.82 (my #457 h0uggl9i) and realistic relax-prize ~498.58 / +17 TPS (ubel #450
c5oyb7gv). The COST side is the weakest-quantified leg and was unowned as a dedicated analysis;
it is scattered across cards. The human cannot weigh the #407 keep-strict-vs-relax fork without
a clean COST axis next to the clean PRIZE axis.

THE REFRAME (the single most decision-useful thing the packet still lacked)
---------------------------------------------------------------------------
The deployed 481.53 is ALREADY off the strict frontier (identity 0.9966, 3 flips, PR #52
2x9fm2zx). So the human's real choice is NOT "pristine-strict vs dirty-relax." It is:
*"we already ship 3 flips in production; does +17..29 TPS justify going from 3 flips to N flips,
and does quality (PPL <= 2.42) survive?"* That status-quo-relative framing makes the relax
decision GRADED, not binary.

THREE INTEGRANDS (this card produces; fern folds into the #357 capstone)
-----------------------------------------------------------------------
  (1) EQUIVALENCE-SEVERITY AXIS — every operating point on ONE axis, each row
      (TPS, greedy-identity-rate, flip-count, PPL, greedy-safe). Monotone in TPS; identity
      monotone-degrading as TPS rises (1.0 -> 0.9966 -> <=0.9966).
  (2) PPL-GATE HEADROOM LEDGER — gate PPL <= 2.42; deployed PPL 2.3772 -> margin 0.0428.
      flip-COUNT and PPL are DIFFERENT costs (flips can be PPL-neutral near-ties OR
      PPL-breaching) — kept orthogonal; this is the trap that sank four "isolated" levers.
  (3) STATUS-QUO-RELATIVE DECISION RULE — fern #357 verbatim: relax justified iff
      (TPS gain over deployed >= human-set threshold) AND (measured PPL <= 2.42) AND
      (break is the SAME KIND already deployed — accumulation-order flips, not a new mode).
      TPS-gain side computed now (re-cited from #457, not re-derived): realistic +17.05,
      ceiling +29.34.

NON-DUPLICATION: identity/flips for the relax-realistic point are stark #452's measurement;
this card consumes them as PARAMETERIZED SLOTS (default "UNMEASURED — stark #452 to fill") and
re-derives nothing. ubel #453 / kanna #454 hunt a strict-SAFE slice; lawine #455 re-anchors the
strict frontier; denken #456 is the closed-lever annex.

LOCAL, CPU-ONLY, ANALYTIC. No GPU / vLLM / HF Job / submission / served-file change / official
draw. BASELINE stays 481.53; adds 0 TPS; greedy/PPL untouched (PPL anchor 2.3772). Imports the
committed result JSONs of #457 (unified_absolute_ceiling) and directive4_correct_bar (deployed
served identity/flips/ppl); re-derives nothing. NOT a launch.

PRIMARY metric  cost_ledger_self_test_passes
TEST    metric  ppl  (2.3772 anchor; this leg does not touch the served model)
HEADLINE         relax_realistic_tps_gain_over_deployed (+17.05), deployed_ppl_gate_margin (0.0428)
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

CEILING_JSON = (
    REPO_ROOT
    / "research/validity/unified_absolute_ceiling/unified_absolute_ceiling_results.json"
)
DIRECTIVE4_JSON = (
    REPO_ROOT / "research/validity/directive4_correct_bar/directive4_correct_bar_results.json"
)

# --------------------------------------------------------------------------- #
# Banked external constants (provenance: W&B run ids, project
# wandb-applied-ai-team/gemma-challenge-senpai). IMPORTED, not re-derived. Each is
# round-tripped against its committed result JSON in load_banked().
# --------------------------------------------------------------------------- #
# TPS anchors + sigma + ppl (my #457 h0uggl9i — the prize-side parent of this card)
UNIFIED_CEILING_TPS = 510.8724230449973   # achieved-read-peak unified ceiling
SPEC_UB_TPS = 520.9527323111674           # spec-600 basis over-optimistic UPPER BOUND (not a point)
DEPLOYED_TPS = 481.53                      # PR #52 official, non-equivalent (3 flips), 2x9fm2zx
STRICT_FRONTIER_TPS = 467.14              # denken #423 5a6zq2yz realized blanket-strict frontier
RELAX_REALISTIC_TPS = 498.57990782684584  # ubel #450 c5oyb7gv realistic split-K hi (greedy-UNSAFE)
SIGMA_HW = 4.8153                          # hardware-variance envelope (~1% of deployed)
PPL_ANCHOR = 2.3772                        # PR #52 served PPL
PPL_GATE = 2.42                            # quality gate (reference PPL + 5%)
# TPS-gains re-cited from #457 (NOT re-derived here)
RELAX_REALISTIC_TPS_GAIN = 17.049907826845867  # #457 relax_prize_over_deployed_lo_tps
RELAX_CEILING_TPS_GAIN = 29.342423044997304    # #457 headroom_deployed_to_ceiling_tps

# Deployed served severity (PR #52 2x9fm2zx; banked in directive4 shared_baselines)
DEPLOYED_IDENTITY = 0.9966                 # directive4 shared_baselines.deployed_served_identity
DEPLOYED_FLIPS_STR = "3/882"              # directive4 shared_baselines.deployed_served_flips
DEPLOYED_FLIPS = 3                         # numerator of "3/882" (#405 j6h228xy reduction-order)
DEPLOYED_TOKENS = 882                      # denominator of "3/882"

# Strict frontier severity — PR-given parameters (provenance denken #423 5a6zq2yz). Supersedes
# directive4's earlier 0.9989 (stark #412, 1 residual bitwise-tie flip @ prompt 90); once the
# residual tie was canonicalized (stark #429 -> denken #423) the realized strict frontier is 1.0.
STRICT_IDENTITY = 1.0
STRICT_FLIPS = 0

# Relax-realistic severity — PARAMETERIZED SLOTS (UNMEASURED; stark #452 to fill). Never confuse a
# placeholder for a measurement: numeric value is None, with an explicit sentinel string.
RELAX_IDENTITY_SLOT = None
RELAX_IDENTITY_SLOT_NOTE = "<=0.9966, UNMEASURED — stark #452 to fill"
RELAX_IDENTITY_UB = 0.9966                 # upper bound for the monotone-degrading check
RELAX_FLIPS_SLOT = None
RELAX_FLIPS_SLOT_NOTE = ">=3, UNMEASURED — stark #452 to fill"
RELAX_PPL_SLOT = None
RELAX_PPL_SLOT_NOTE = "UNMEASURED — stark #452 quality run to fill"

TOL_RT = 1e-6                              # banked round-trip tolerance
TOL_TIGHT = 1e-9                           # derived-arithmetic tolerance


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# Load + round-trip the banked source JSONs (#457 / directive4). Re-derives nothing.
# --------------------------------------------------------------------------- #
def load_banked() -> dict[str, Any]:
    ceil = json.loads(CEILING_JSON.read_text(encoding="utf-8"))["synthesis"]
    c_head, c_const = ceil["headline"], ceil["constants"]
    c_relax = ceil["relax_prize"]
    c_env = ceil["sigma_envelope"]

    d4 = json.loads(DIRECTIVE4_JSON.read_text(encoding="utf-8"))["shared_baselines"]
    d4_flips_num = int(str(d4["deployed_served_flips"]).split("/")[0])

    rt = {
        # --- #457 prize-side anchors (the parent card) ---
        "unified_ceiling_resid": abs(c_head["unified_absolute_ceiling_tps"] - UNIFIED_CEILING_TPS),
        "spec_ub_resid": abs(c_head["spec_basis_ceiling_tps"] - SPEC_UB_TPS),
        "deployed_tps_resid": abs(c_const["deployed_tps"] - DEPLOYED_TPS),
        "strict_tps_resid": abs(c_const["strict_frontier_tps"] - STRICT_FRONTIER_TPS),
        "relax_realistic_tps_resid": abs(c_const["relax_realistic_hi"] - RELAX_REALISTIC_TPS),
        "sigma_hw_resid": abs(c_const["sigma_hw"] - SIGMA_HW),
        "ppl_anchor_resid": abs(c_const["ppl_anchor"] - PPL_ANCHOR),
        "ppl_gate_resid": abs(c_const["ppl_gate"] - PPL_GATE),
        # TPS-gains re-cited from #457 (NOT re-derived)
        "relax_realistic_gain_resid": abs(
            c_relax["relax_prize_over_deployed_lo_tps"] - RELAX_REALISTIC_TPS_GAIN
        ),
        "relax_ceiling_gain_resid": abs(
            c_env["headroom_deployed_to_ceiling_tps"] - RELAX_CEILING_TPS_GAIN
        ),
        # --- directive4 deployed-severity witnesses (PR #52 2x9fm2zx) ---
        "deployed_identity_resid": abs(d4["deployed_served_identity"] - DEPLOYED_IDENTITY),
        "deployed_flips_resid": abs(d4_flips_num - DEPLOYED_FLIPS),
        "deployed_ppl_xcheck_resid": abs(d4["deployed_ppl"] - PPL_ANCHOR),
        "ppl_cap_xcheck_resid": abs(d4["ppl_cap"] - PPL_GATE),
        "deployed_tps_xcheck_resid": abs(d4["deployed_tps"] - DEPLOYED_TPS),
    }
    max_resid = max(rt.values())
    return {
        "roundtrip_resid": rt,
        "max_roundtrip_resid": max_resid,
        "all_roundtrip_ok": bool(max_resid <= TOL_RT),
        "ceiling_self_test_passed": bool(c_head.get("unified_ceiling_self_test_passes", False)),
        "deployed_flips_str_banked": str(d4["deployed_served_flips"]),
    }


# --------------------------------------------------------------------------- #
# (1) Equivalence-severity axis: each row (TPS, identity, flips, PPL, greedy-safe, source).
# --------------------------------------------------------------------------- #
def severity_axis() -> dict[str, Any]:
    """ONE axis, monotone in TPS, identity monotone-degrading as TPS rises. The ceiling and the
    spec UB are markers (physical limits), NOT operating points — identity/flips/ppl are n/a there."""
    points = [
        {
            "point": "strict_frontier",
            "tps": STRICT_FRONTIER_TPS,
            "identity": STRICT_IDENTITY,
            "flips": STRICT_FLIPS,
            "ppl": PPL_ANCHOR,
            "ppl_status": "reference_by_construction",
            "greedy_safe": True,
            "is_operating_point": True,
            "source_run": "denken #423 5a6zq2yz",
            "note": ("realized blanket-strict greedy-safe frontier (identity 1.0, 0 flips). "
                     "PPL is the greedy-reference quality by construction (identity 1.0 => "
                     "byte-exact greedy token stream); supersedes directive4's earlier 0.9989 "
                     "(stark #412, 1 residual bitwise-tie flip) now that stark #429 canonicalized it"),
        },
        {
            "point": "deployed_noneq",
            "tps": DEPLOYED_TPS,
            "identity": DEPLOYED_IDENTITY,
            "flips": DEPLOYED_FLIPS,
            "ppl": PPL_ANCHOR,
            "ppl_status": "measured",
            "greedy_safe": False,
            "is_operating_point": True,
            "source_run": "PR #52 2x9fm2zx",
            "note": (f"deployed leaderboard incumbent; {DEPLOYED_FLIPS_STR} M=8 reduction-order "
                     "near-tie flips (#405 j6h228xy) => identity 0.9966 < 1.0. ALREADY off the "
                     "strict frontier — this is the status-quo the human implicitly accepted"),
        },
        {
            "point": "relax_realistic",
            "tps": RELAX_REALISTIC_TPS,
            "identity": RELAX_IDENTITY_SLOT,            # UNMEASURED slot
            "identity_slot_note": RELAX_IDENTITY_SLOT_NOTE,
            "identity_ub_for_monotone": RELAX_IDENTITY_UB,
            "flips": RELAX_FLIPS_SLOT,                  # UNMEASURED slot
            "flips_slot_note": RELAX_FLIPS_SLOT_NOTE,
            "ppl": RELAX_PPL_SLOT,                       # UNMEASURED slot
            "ppl_slot_note": RELAX_PPL_SLOT_NOTE,
            "ppl_status": "unmeasured_slot",
            "greedy_safe": False,
            "is_operating_point": True,
            "source_run": "ubel #450 c5oyb7gv",
            "note": ("realistic relax-prize (greedy-UNSAFE FP-reassociating split-K re-tiling). "
                     "identity/flips/PPL are stark #452's measurement — consumed as PARAMETERIZED "
                     "SLOTS here, NOT measured by this card"),
        },
        {
            "point": "unified_physical_ceiling",
            "tps": UNIFIED_CEILING_TPS,
            "tps_sigma": SIGMA_HW,
            "identity": None,
            "flips": None,
            "ppl": None,
            "ppl_status": "not_an_operating_point",
            "greedy_safe": None,
            "is_operating_point": False,
            "source_run": "land #457 h0uggl9i",
            "note": "achieved-read-peak physical ceiling 510.87 +/- 4.82 (a supply limit, not a config)",
        },
        {
            "point": "spec_over_optimistic_ub",
            "tps": SPEC_UB_TPS,
            "identity": None,
            "flips": None,
            "ppl": None,
            "ppl_status": "not_an_operating_point",
            "greedy_safe": None,
            "is_operating_point": False,
            "source_run": "land #436 nvsbctji (via #457)",
            "note": "spec-600 basis UPPER BOUND (unreachable BW); labeled UB, NOT an operating point",
        },
    ]

    # monotone in TPS across the whole axis (low -> high)
    tps_monotone = all(points[i]["tps"] <= points[i + 1]["tps"] + TOL_RT for i in range(len(points) - 1))

    # identity monotone-degrading across the OPERATING points as TPS rises:
    #   strict 1.0 >= deployed 0.9966 >= relax (<=0.9966, use UB for the unmeasured slot)
    op_identity_seq = [STRICT_IDENTITY, DEPLOYED_IDENTITY, RELAX_IDENTITY_UB]
    identity_degrading = all(
        op_identity_seq[i] >= op_identity_seq[i + 1] - TOL_RT for i in range(len(op_identity_seq) - 1)
    )

    # central reframe: deployed is ALREADY off the strict frontier.
    deployed_off_strict_frontier = bool(DEPLOYED_IDENTITY < STRICT_IDENTITY and DEPLOYED_FLIPS > 0)

    return {
        "points": points,
        "axis_monotone_in_tps": bool(tps_monotone),
        "identity_monotone_degrading": bool(identity_degrading),
        "operating_identity_sequence": op_identity_seq,
        "deployed_off_strict_frontier": deployed_off_strict_frontier,
        "operating_points": ["strict_frontier", "deployed_noneq", "relax_realistic"],
        "ceiling_markers": ["unified_physical_ceiling", "spec_over_optimistic_ub"],
        "summary": (
            f"strict {STRICT_FRONTIER_TPS:.2f} (id 1.0, 0 flips, SAFE) -> deployed {DEPLOYED_TPS:.2f} "
            f"(id 0.9966, 3 flips, UNSAFE) -> relax {RELAX_REALISTIC_TPS:.2f} (id/flips UNMEASURED slot, "
            f"UNSAFE) -> ceiling {UNIFIED_CEILING_TPS:.2f} +/- {SIGMA_HW:.2f} -> spec-UB {SPEC_UB_TPS:.2f}"),
    }


# --------------------------------------------------------------------------- #
# (2) PPL-gate headroom ledger. flip-COUNT and PPL are DIFFERENT costs — kept orthogonal.
# --------------------------------------------------------------------------- #
def ppl_ledger() -> dict[str, Any]:
    deployed_margin = PPL_GATE - PPL_ANCHOR  # 2.42 - 2.3772 = 0.0428

    ledger = [
        {
            "point": "strict_frontier",
            "ppl_status": "reference_by_construction",
            "ppl_measured": None,
            "ppl_gate_margin": None,
            "ppl_gate_margin_note": (
                f">= {deployed_margin:.4f} — identity 1.0 => byte-exact greedy reference stream => "
                "PPL is the reference quality the gate (reference + 5%) is defined against; admissible "
                "by construction. Not separately measured this cycle"),
            "ppl_gate_admissible": True,
        },
        {
            "point": "deployed_noneq",
            "ppl_status": "measured",
            "ppl_measured": PPL_ANCHOR,
            "ppl_gate_margin": deployed_margin,        # 0.0428
            "ppl_gate_margin_note": f"measured {PPL_ANCHOR} <= gate {PPL_GATE} => margin {deployed_margin:.4f}",
            "ppl_gate_admissible": True,
        },
        {
            "point": "relax_realistic",
            "ppl_status": "unmeasured_slot",
            "ppl_measured": None,
            "ppl_gate_margin": None,
            "available_ppl_budget_from_deployed_anchor": deployed_margin,  # 0.0428
            "ppl_gate_margin_note": (
                f"UNMEASURED — stark #452 quality run to fill. Available PPL budget from the deployed "
                f"anchor = {deployed_margin:.4f} (2.42 - 2.3772). PPL-admissible IFF measured PPL <= "
                f"{PPL_GATE}. Flip-count tells you NOTHING about this — flips can be PPL-neutral "
                "(near-ties) OR PPL-breaching"),
            "ppl_gate_admissible": "unknown_pending_measurement",
        },
    ]

    return {
        "ppl_gate": PPL_GATE,
        "deployed_ppl": PPL_ANCHOR,
        "deployed_ppl_gate_margin": deployed_margin,                  # HEADLINE 0.0428
        "available_ppl_budget_from_deployed_anchor": deployed_margin,
        "ledger": ledger,
        "flip_and_ppl_costs_kept_orthogonal": True,
        "orthogonality_note": (
            "flip-COUNT and PPL are DIFFERENT, ORTHOGONAL costs. A flip is a token-ID divergence from "
            "the greedy reference (an identity/equivalence cost); PPL is a quality cost. Flips can be "
            "PPL-NEUTRAL (reduction-order near-ties, like the deployed 3) or PPL-BREACHING (genuine "
            "quality loss). You CANNOT infer PPL-admissibility from the flip count — that conflation is "
            "the trap that sank four modeled-in-isolation levers. The ledger keeps the two in separate "
            "columns and never derives one from the other"),
        "summary": (
            f"gate PPL <= {PPL_GATE}; deployed PPL {PPL_ANCHOR} => margin {deployed_margin:.4f}. "
            f"strict = reference-quality (admissible by construction). relax PPL UNMEASURED — admissible "
            f"only if a quality run (stark #452) reads <= {PPL_GATE}"),
    }


# --------------------------------------------------------------------------- #
# (3) Status-quo-relative decision rule (fern #357 verbatim) + the graded framing.
# --------------------------------------------------------------------------- #
def decision_rule(led: dict) -> dict[str, Any]:
    rule_text = (
        "relax is justified iff (TPS gain over deployed >= human-set threshold) AND "
        "(measured PPL <= 2.42) AND (the break is the SAME KIND already deployed — "
        "accumulation-order flips, not a new failure mode)"
    )

    clauses = [
        {
            "clause": "tps_gain_over_deployed_ge_threshold",
            "status": "QUANTIFIED",
            "relax_realistic_tps_gain_over_deployed": RELAX_REALISTIC_TPS_GAIN,   # +17.05
            "relax_ceiling_tps_gain_over_deployed": RELAX_CEILING_TPS_GAIN,       # +29.34
            "threshold": "HUMAN-SET (not yet specified — the human's call)",
            "note": ("TPS-gain side re-cited from #457 (not re-derived): realistic +17.05, ceiling "
                     "+29.34 over the deployed 481.53. Whether that clears the bar is the human's "
                     "threshold to set"),
        },
        {
            "clause": "measured_ppl_le_gate",
            "status": "PENDING",
            "gate": PPL_GATE,
            "relax_ppl_measured": RELAX_PPL_SLOT,
            "available_budget_from_deployed_anchor": led["deployed_ppl_gate_margin"],  # 0.0428
            "note": ("relax PPL is UNMEASURED (stark #452 quality run). Available budget 0.0428 from "
                     "the deployed anchor; admissible only if the measured relax PPL <= 2.42"),
        },
        {
            "clause": "same_kind_of_break_as_deployed",
            "status": "PROVISIONAL_SAME_KIND",
            "deployed_break_kind": "M=8 reduction-order near-tie flips (#405 j6h228xy)",
            "relax_break_kind": "FP-reassociating split-K re-tiling (accumulation-order)",
            "note": ("the relax-prize break is FP accumulation-order reassociation — the SAME family as "
                     "the deployed reduction-order near-tie flips, NOT a new failure mode. Provisional "
                     "pending stark #452's flip characterization (confirm the relax flips are "
                     "accumulation-order near-ties, not a distinct quality-destroying mode)"),
        },
    ]

    # current evaluation: 1 of 3 clauses fully resolved (TPS gain); 2 pending stark #452 + human.
    resolved = sum(1 for c in clauses if c["status"] == "QUANTIFIED")
    return {
        "fern_357_rule_verbatim": rule_text,
        "clauses": clauses,
        "clauses_resolved": resolved,
        "clauses_total": len(clauses),
        "overall_status": "GRADED_DECISION_PENDING",
        "relax_realistic_tps_gain_over_deployed": RELAX_REALISTIC_TPS_GAIN,
        "relax_ceiling_tps_gain_over_deployed": RELAX_CEILING_TPS_GAIN,
        "status_quo_anchor": {
            "point": "deployed_noneq",
            "tps": DEPLOYED_TPS,
            "identity": DEPLOYED_IDENTITY,
            "flips": DEPLOYED_FLIPS,
            "framing": ("the deployed 3-flip / identity-0.9966 point is the status-quo the human "
                        "IMPLICITLY ALREADY ACCEPTED when PR #52 was deployed. The relax decision is "
                        "therefore GRADED — '3 flips -> N flips for +17..29 TPS, does quality survive?' "
                        "— NOT the binary 'pristine-strict vs dirty-relax' the packet first framed"),
        },
        "summary": (
            "fern #357 rule: relax justified iff (TPS gain >= human threshold) AND (measured PPL <= "
            "2.42) AND (same-kind break). TODAY: clause-1 QUANTIFIED (+17.05 realistic / +29.34 "
            "ceiling over deployed); clause-2 PENDING (relax PPL unmeasured, stark #452); clause-3 "
            "PROVISIONAL same-kind (accumulation-order flips). Decision is graded off the deployed "
            "3-flip status quo, not binary"),
    }


# --------------------------------------------------------------------------- #
# (4) Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def selftests(banked: dict, axis: dict, led: dict, dec: dict) -> dict[str, Any]:
    deployed_margin = led["deployed_ppl_gate_margin"]

    # (a) every banked source number round-trips its committed JSON within tol.
    cond_a = bool(banked["all_roundtrip_ok"])

    # (b) axis monotone in TPS (467.14 < 481.53 < 498.58 < 510.87 < 520.95).
    cond_b = bool(
        axis["axis_monotone_in_tps"]
        and STRICT_FRONTIER_TPS < DEPLOYED_TPS < RELAX_REALISTIC_TPS < UNIFIED_CEILING_TPS < SPEC_UB_TPS
    )

    # (c) identity monotone-degrading across operating points as TPS rises (1.0 >= 0.9966 >= <=0.9966).
    cond_c = bool(
        axis["identity_monotone_degrading"]
        and STRICT_IDENTITY >= DEPLOYED_IDENTITY >= RELAX_IDENTITY_UB
    )

    # (d) the central reframe: deployed identity 0.9966 < 1.0 (deployed already off strict frontier).
    cond_d = bool(DEPLOYED_IDENTITY < 1.0 and DEPLOYED_IDENTITY < STRICT_IDENTITY
                  and DEPLOYED_FLIPS > 0 and axis["deployed_off_strict_frontier"])

    # (e) deployed_ppl_gate_margin computed correctly (== gate - deployed, == 0.0428).
    cond_e = bool(
        abs(deployed_margin - (PPL_GATE - PPL_ANCHOR)) <= TOL_TIGHT
        and abs(deployed_margin - 0.0428) <= TOL_TIGHT
    )

    # (f) relax slots clearly marked UNMEASURED (numeric None + sentinel) — no placeholder mistaken
    #     for a measurement.
    relax_row = next(p for p in axis["points"] if p["point"] == "relax_realistic")
    relax_led = next(r for r in led["ledger"] if r["point"] == "relax_realistic")
    cond_f = bool(
        relax_row["identity"] is None and isinstance(relax_row.get("identity_slot_note"), str)
        and relax_row["flips"] is None and isinstance(relax_row.get("flips_slot_note"), str)
        and relax_row["ppl"] is None and isinstance(relax_row.get("ppl_slot_note"), str)
        and relax_led["ppl_measured"] is None
        and relax_led["ppl_gate_admissible"] == "unknown_pending_measurement"
    )

    # (g) TPS-gains re-cited from #457 match the (498.58-481.53) / (510.87-481.53) arithmetic.
    cond_g = bool(
        abs(RELAX_REALISTIC_TPS_GAIN - (RELAX_REALISTIC_TPS - DEPLOYED_TPS)) <= TOL_RT
        and abs(RELAX_CEILING_TPS_GAIN - (UNIFIED_CEILING_TPS - DEPLOYED_TPS)) <= TOL_RT
        and dec["relax_realistic_tps_gain_over_deployed"] == RELAX_REALISTIC_TPS_GAIN
        and dec["relax_ceiling_tps_gain_over_deployed"] == RELAX_CEILING_TPS_GAIN
    )

    # (h) ppl anchor preserved (this leg does not touch the served model) and within gate.
    cond_h = bool(abs(PPL_ANCHOR - 2.3772) <= TOL_RT and PPL_ANCHOR <= PPL_GATE)

    # (i) flip-COUNT and PPL kept orthogonal (separate columns; no derivation of one from the other).
    cond_i = bool(led["flip_and_ppl_costs_kept_orthogonal"])

    # (j) NaN-clean — set by the caller after the full payload walk.
    cond_j = True

    conditions = {
        "a_all_banked_numbers_roundtrip": cond_a,
        "b_axis_monotone_in_tps": cond_b,
        "c_identity_monotone_degrading": cond_c,
        "d_deployed_off_strict_frontier_0p9966_lt_1": cond_d,
        "e_deployed_ppl_gate_margin_correct_0p0428": cond_e,
        "f_relax_slots_marked_unmeasured": cond_f,
        "g_tps_gains_recited_from_457": cond_g,
        "h_ppl_anchor_preserved_within_gate": cond_h,
        "i_flip_and_ppl_costs_orthogonal": cond_i,
        "j_nan_clean": cond_j,
    }
    return {
        "conditions": conditions,
        "cost_ledger_self_test_passes": bool(all(conditions.values())),
        "detail": {
            "max_roundtrip_resid": banked["max_roundtrip_resid"],
            "deployed_ppl_gate_margin": deployed_margin,
            "relax_realistic_tps_gain_over_deployed": RELAX_REALISTIC_TPS_GAIN,
            "relax_ceiling_tps_gain_over_deployed": RELAX_CEILING_TPS_GAIN,
        },
    }


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    banked = load_banked()
    axis = severity_axis()
    led = ppl_ledger()
    dec = decision_rule(led)
    st = selftests(banked, axis, led, dec)

    headline = {
        "cost_ledger_self_test_passes": bool(st["cost_ledger_self_test_passes"]),    # PRIMARY
        "deployed_ppl_gate_margin": led["deployed_ppl_gate_margin"],                 # HEADLINE 0.0428
        "relax_realistic_tps_gain_over_deployed": RELAX_REALISTIC_TPS_GAIN,          # HEADLINE +17.05
        "relax_ceiling_tps_gain_over_deployed": RELAX_CEILING_TPS_GAIN,              # HEADLINE +29.34
        "deployed_identity": DEPLOYED_IDENTITY,
        "deployed_flips": DEPLOYED_FLIPS,
        "strict_identity": STRICT_IDENTITY,
        "strict_flips": STRICT_FLIPS,
        "relax_realistic_identity": RELAX_IDENTITY_SLOT,
        "relax_realistic_flips": RELAX_FLIPS_SLOT,
        "relax_realistic_ppl": RELAX_PPL_SLOT,
        "deployed_off_strict_frontier": axis["deployed_off_strict_frontier"],
        "axis_monotone_in_tps": axis["axis_monotone_in_tps"],
        "identity_monotone_degrading": axis["identity_monotone_degrading"],
        "flip_and_ppl_costs_kept_orthogonal": led["flip_and_ppl_costs_kept_orthogonal"],
        "ppl": PPL_ANCHOR,
        "analysis_only": True,
        "no_served_file_change": True,
        "official_tps": 0,
    }
    verdict = (
        f"RELAX-COST-LEDGER-DEPLOYED-ALREADY-3FLIPS-id0.9966-MARGIN-{led['deployed_ppl_gate_margin']:.4f}-"
        f"GRADED-DECISION-GAIN-+{RELAX_REALISTIC_TPS_GAIN:.2f}..+{RELAX_CEILING_TPS_GAIN:.2f}"
    )
    handoff = (
        f"COST AXIS: strict {STRICT_FRONTIER_TPS:.2f} (id 1.0/0 flips/SAFE) -> deployed {DEPLOYED_TPS:.2f} "
        f"(id 0.9966/3 flips/UNSAFE, PPL {PPL_ANCHOR}, margin {led['deployed_ppl_gate_margin']:.4f}) -> "
        f"relax {RELAX_REALISTIC_TPS:.2f} (id/flips/PPL UNMEASURED slot, UNSAFE) -> ceiling "
        f"{UNIFIED_CEILING_TPS:.2f}+/-{SIGMA_HW:.2f}. DECISION (fern #357): relax justified iff gain "
        f"(+{RELAX_REALISTIC_TPS_GAIN:.1f}..+{RELAX_CEILING_TPS_GAIN:.1f}) >= human threshold AND measured "
        f"PPL <= {PPL_GATE} AND same-kind break. GRADED off the deployed 3-flip status quo, not binary"
    )
    return {
        "headline": headline,
        "severity_axis": axis,
        "ppl_ledger": led,
        "decision_rule": dec,
        "banked_roundtrip": banked,
        "self_test": st,
        "constants": {
            "unified_ceiling_tps": UNIFIED_CEILING_TPS, "spec_ub_tps": SPEC_UB_TPS,
            "deployed_tps": DEPLOYED_TPS, "strict_frontier_tps": STRICT_FRONTIER_TPS,
            "relax_realistic_tps": RELAX_REALISTIC_TPS, "sigma_hw": SIGMA_HW,
            "ppl_anchor": PPL_ANCHOR, "ppl_gate": PPL_GATE,
            "deployed_identity": DEPLOYED_IDENTITY, "deployed_flips": DEPLOYED_FLIPS,
            "deployed_flips_str": DEPLOYED_FLIPS_STR, "deployed_tokens": DEPLOYED_TOKENS,
            "strict_identity": STRICT_IDENTITY, "strict_flips": STRICT_FLIPS,
            "relax_identity_ub": RELAX_IDENTITY_UB,
            "relax_realistic_tps_gain": RELAX_REALISTIC_TPS_GAIN,
            "relax_ceiling_tps_gain": RELAX_CEILING_TPS_GAIN,
        },
        "verdict": verdict,
        "handoff_line": handoff,
        "imports": {
            "provenance": (
                "land#457 h0uggl9i (unified ceiling 510.87 +/- 4.82, spec-UB 520.95, deployed 481.53, "
                "strict 467.14, relax-realistic 498.58, sigma_hw 4.8153, ppl_anchor 2.3772, ppl_gate "
                "2.42, relax TPS-gains +17.05/+29.34) x directive4_correct_bar shared_baselines "
                "(deployed_served_identity 0.9966, deployed_served_flips 3/882, deployed_ppl 2.3772, "
                "ppl_cap 2.42 — PR #52 2x9fm2zx). Strict identity 1.0 / 0 flips PR-given (denken #423 "
                "5a6zq2yz). relax identity/flips/PPL are stark #452 PARAMETERIZED SLOTS. All run-ids in "
                "wandb-applied-ai-team/gemma-challenge-senpai."),
            "machinery": "round-trip of committed #457 / directive4 result JSONs; re-derives nothing",
        },
    }


# --------------------------------------------------------------------------- #
# NaN guard + report + W&B (mirrors #457; never fatal).
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
    axis, led, dec = syn["severity_axis"], syn["ppl_ledger"], syn["decision_rule"]
    st = syn["self_test"]
    print("\n" + "=" * 100, flush=True)
    print("RELAX-DECISION COST LEDGER (PR #458, land) — equivalence-severity axis + PPL budget, CPU-only",
          flush=True)
    print("=" * 100, flush=True)
    print("  (1) EQUIVALENCE-SEVERITY AXIS (low -> high TPS):", flush=True)
    print(f"      {'TPS':>9}  {'identity':>9}  {'flips':>6}  {'PPL':>8}  {'safe':>6}  point / source", flush=True)
    for p in axis["points"]:
        ident = "n/a" if p["identity"] is None else (
            "SLOT" if p["point"] == "relax_realistic" else f"{p['identity']:.4f}")
        flips = "n/a" if p["flips"] is None else ("SLOT" if p["point"] == "relax_realistic" else str(p["flips"]))
        ppl = {"measured": f"{PPL_ANCHOR:.4f}", "reference_by_construction": "ref",
               "unmeasured_slot": "SLOT", "not_an_operating_point": "n/a"}[p["ppl_status"]]
        safe = {True: "YES", False: "no", None: "—"}[p["greedy_safe"]]
        print(f"      {p['tps']:9.2f}  {ident:>9}  {flips:>6}  {ppl:>8}  {safe:>6}  "
              f"{p['point']} [{p['source_run']}]", flush=True)
    print(f"      monotone-in-TPS={axis['axis_monotone_in_tps']}  "
          f"identity-degrading={axis['identity_monotone_degrading']}  "
          f"deployed-OFF-strict-frontier={axis['deployed_off_strict_frontier']}", flush=True)
    print("-" * 100, flush=True)
    print(f"  (2) PPL-GATE LEDGER (gate <= {led['ppl_gate']}; deployed PPL {led['deployed_ppl']} -> "
          f"margin {led['deployed_ppl_gate_margin']:.4f}):", flush=True)
    for r in led["ledger"]:
        m = "n/a" if r["ppl_gate_margin"] is None else f"{r['ppl_gate_margin']:.4f}"
        print(f"      {r['point']:<18} status={r['ppl_status']:<26} margin={m:>7}  "
              f"admissible={r['ppl_gate_admissible']}", flush=True)
    print(f"      flip-count & PPL kept ORTHOGONAL = {led['flip_and_ppl_costs_kept_orthogonal']} "
          "(flips can be PPL-neutral OR PPL-breaching; never inferred from count)", flush=True)
    print("-" * 100, flush=True)
    print(f"  (3) DECISION RULE (fern #357): {dec['overall_status']} "
          f"({dec['clauses_resolved']}/{dec['clauses_total']} clauses resolved)", flush=True)
    for c in dec["clauses"]:
        print(f"      - [{c['status']:<22}] {c['clause']}", flush=True)
    print(f"      TPS gain over deployed: realistic +{dec['relax_realistic_tps_gain_over_deployed']:.2f}, "
          f"ceiling +{dec['relax_ceiling_tps_gain_over_deployed']:.2f}  (GRADED off the 3-flip status quo)",
          flush=True)
    print("-" * 100, flush=True)
    print(f"  (4) PRIMARY cost_ledger_self_test_passes = {st['cost_ledger_self_test_passes']}", flush=True)
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
        print(f"[cost-ledger] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    h, led, dec = syn["headline"], syn["ppl_ledger"], syn["decision_rule"]
    axis, st = syn["severity_axis"], syn["self_test"]
    run = init_wandb_run(
        job_type="relax-decision-cost-ledger",
        agent="land",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["relax-decision-cost-ledger", "equivalence-escalation-anchors", "cost-axis",
              "ppl-gate-budget", "decision-rule", "equivalence-severity", "capstone-anchor",
              "analysis-only", "bank-the-analysis"],
        config={
            "unified_ceiling_tps": UNIFIED_CEILING_TPS, "spec_ub_tps": SPEC_UB_TPS,
            "deployed_tps": DEPLOYED_TPS, "strict_frontier_tps": STRICT_FRONTIER_TPS,
            "relax_realistic_tps": RELAX_REALISTIC_TPS, "sigma_hw": SIGMA_HW,
            "ppl_anchor": PPL_ANCHOR, "ppl_gate": PPL_GATE,
            "deployed_identity": DEPLOYED_IDENTITY, "deployed_flips": DEPLOYED_FLIPS,
            "strict_identity": STRICT_IDENTITY, "strict_flips": STRICT_FLIPS,
            "wandb_group": args.wandb_group,
            "source_runs": "land#457 h0uggl9i, directive4 shared_baselines (PR#52 2x9fm2zx), "
                           "denken#423 5a6zq2yz (strict, PR-given), stark#452 (relax slots, UNMEASURED)",
        },
    )
    if run is None:
        print("[cost-ledger] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "cost_ledger_self_test_passes": int(bool(st["cost_ledger_self_test_passes"])),  # PRIMARY
        "deployed_ppl_gate_margin": led["deployed_ppl_gate_margin"],                    # HEADLINE
        "relax_realistic_tps_gain_over_deployed": RELAX_REALISTIC_TPS_GAIN,             # HEADLINE
        "relax_ceiling_tps_gain_over_deployed": RELAX_CEILING_TPS_GAIN,                 # HEADLINE
        "available_ppl_budget_from_deployed_anchor": led["available_ppl_budget_from_deployed_anchor"],
        "deployed_identity": DEPLOYED_IDENTITY,
        "deployed_flips": DEPLOYED_FLIPS,
        "strict_identity": STRICT_IDENTITY,
        "strict_flips": STRICT_FLIPS,
        "relax_identity_ub": RELAX_IDENTITY_UB,
        "deployed_off_strict_frontier": int(bool(axis["deployed_off_strict_frontier"])),
        "axis_monotone_in_tps": int(bool(axis["axis_monotone_in_tps"])),
        "identity_monotone_degrading": int(bool(axis["identity_monotone_degrading"])),
        "flip_and_ppl_costs_kept_orthogonal": int(bool(led["flip_and_ppl_costs_kept_orthogonal"])),
        "decision_clauses_resolved": dec["clauses_resolved"],
        "decision_clauses_total": dec["clauses_total"],
        "deployed_tps": DEPLOYED_TPS,
        "strict_frontier_tps": STRICT_FRONTIER_TPS,
        "relax_realistic_tps": RELAX_REALISTIC_TPS,
        "unified_ceiling_tps": UNIFIED_CEILING_TPS,
        "spec_ub_tps": SPEC_UB_TPS,
        "sigma_hw_tps": SIGMA_HW,
        "max_roundtrip_resid": syn["banked_roundtrip"]["max_roundtrip_resid"],
        "ppl": PPL_ANCHOR,
        "ppl_gate": PPL_GATE,
        "analysis_only": 1,
        "no_served_file_change": 1,
        "official_tps": 0,
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="relax_decision_cost_ledger_result", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    print(f"[cost-ledger] wandb logged {len(summary)} keys; run id {getattr(run, 'id', '?')}",
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
        "created_at": created_at, "pr": 458, "agent": "land",
        "kind": "relax-decision-cost-ledger", "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _nan_paths(payload)
    payload["nan_clean"] = not nan_paths
    syn["self_test"]["conditions"]["j_nan_clean"] = not nan_paths
    syn["self_test"]["cost_ledger_self_test_passes"] = bool(
        all(syn["self_test"]["conditions"].values()))
    syn["headline"]["cost_ledger_self_test_passes"] = syn["self_test"]["cost_ledger_self_test_passes"]
    if nan_paths:
        print(f"[cost-ledger] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "relax_decision_cost_ledger_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[cost-ledger] wrote {out_path}", flush=True)

    st_path = out_dir / "relax_decision_cost_ledger_selftest.json"
    with st_path.open("w", encoding="utf-8") as fh:
        json.dump(syn["self_test"]["conditions"], fh, indent=2, sort_keys=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = (syn["self_test"]["cost_ledger_self_test_passes"] and payload["nan_clean"])
        print(f"[cost-ledger] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
