#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Relax-decision surface (PR #462, land) — CPU-only analytic.

THE QUESTION (turn #458's RULE into NUMERIC THRESHOLDS so the recommendation falls out automatically)
-----------------------------------------------------------------------------------------------------
My #458 cost-ledger (uhhyec0q) framed the human's #407 relax decision as a GRADED rule with
`GRADED_DECISION_PENDING` (1/3 clauses resolved): relax justified iff
  (TPS gain over deployed >= human-set bar B) AND (measured PPL <= 2.42) AND (same-kind break).
The TPS-gain clause is quantified (+17.05 realistic / +29.34 ceiling over the deployed 481.53);
the PPL + same-kind clauses are pending stark #452's measurement. The packet hands the human a
RULE but not the THRESHOLDS — so when stark #452's measured (TPS, PPL, flip-count) lands, someone
still has to turn it into GO / NO-GO by hand.

This card supplies the missing **decision surface**: the explicit decision-flip boundaries so the
recommendation falls out *automatically* the instant stark #452 reports. It is the analytic
complement to #458 (which states the rule) and a direct input to fern #357 (which composes the
one-screen GO/NO-GO packet). I build the decision MATH; fern PRESENTS it.

THREE PRODUCTS (this card produces; fern #357 folds the one-screen packet)
--------------------------------------------------------------------------
  (1) DECISION-FLIP THRESHOLDS — the boundaries at which the recommendation flips:
      (a) TPS-gain boundary: for a human bar B (required TPS over deployed 481.53), the largest B
          for which relax-realistic (+17.05) is a CI-CLEAN GO (gain - k*sigma_hw >= B), and the
          same for the ceiling (+29.34). The ceiling CI-clean at k=1 must EQUAL #457's banked
          headroom LCB 24.527 — a hard cross-check of the CI arithmetic against the parent card.
      (b) PPL boundary: max admissible relax-PPL (the gate 2.42) and its margin from the deployed
          anchor 2.3772 (= 0.0428). Rule: relax PPL-admissible iff measured PPL <= 2.42.
      (c) flip-count framing: "Δflips for ΔTPS" against the deployed 3-flip status quo, parameterized
          over stark #452's pending flip-count N. flip-count (equivalence) and PPL (quality) kept
          strictly ORTHOGONAL as in #458 — the decision gate is PPL + same-KIND, NEVER the count N.
  (2) SENSITIVITY / TORNADO — rank which pending input (measured TPS vs measured PPL vs measured
      flip-count) most swings the recommendation, by how close the deployed nominal sits to each
      axis's flip boundary. PPL is a hard gate with a razor-thin 0.0428 margin; the TPS-gain clause
      is robustly GO unless the human sets an aggressive bar; the flip-count is orthogonal (only the
      KIND gates, never the count). most-sensitive = PPL.
  (3) PRE-WIRED stark #452 -> RECOMMENDATION — the exact one-number-swap: when stark #452 reports
      (TPS, PPL, flip-count) each value resolves one clause, and a single recommend() collapses the
      three to GO / NO-GO / CI-AMBIGUOUS. Hand to fern #357; this card does NOT duplicate the
      capstone, it produces the surface fern reads.

NON-DUPLICATION: re-uses #458's committed anchors + #457's ceiling JSON (round-trip, re-derive
nothing). relax identity/flips/PPL stay stark #452 PARAMETERIZED SLOTS; this card never measures
them. It computes only the DECISION GEOMETRY on top of the already-banked numbers.

LOCAL, CPU-ONLY, ANALYTIC. No GPU / vLLM / HF Job / submission / served-file change / official
draw. BASELINE stays 481.53; adds 0 TPS; greedy/PPL untouched (PPL anchor 2.3772).

PRIMARY metric  decision_flip_tps_threshold  (largest bar B for which relax-realistic is a CI-clean
                GO, k=1 sigma_hw; == relax-realistic gain - sigma_hw == 12.2346)
TEST    metric  ppl  (2.3772 anchor; this leg does not touch the served model)
HEADLINE        most_sensitive_pending_input ("ppl"), max_admissible_relax_ppl (2.42)
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
COST_LEDGER_JSON = (
    REPO_ROOT
    / "research/validity/relax_decision_cost_ledger/relax_decision_cost_ledger_results.json"
)

# --------------------------------------------------------------------------- #
# Banked external constants (provenance: W&B run ids, project
# wandb-applied-ai-team/gemma-challenge-senpai). IMPORTED, not re-derived. Each is
# round-tripped against its committed result JSON in load_banked().
# --------------------------------------------------------------------------- #
# TPS anchors + sigma + ppl (my #457 h0uggl9i prize-side parent; #458 uhhyec0q cost sibling)
UNIFIED_CEILING_TPS = 510.8724230449973   # achieved-read-peak unified ceiling
SPEC_UB_TPS = 520.9527323111674           # spec-600 basis over-optimistic UPPER BOUND (not a point)
DEPLOYED_TPS = 481.53                      # PR #52 official, non-equivalent (3 flips), 2x9fm2zx
STRICT_FRONTIER_TPS = 467.14              # denken #423 5a6zq2yz realized blanket-strict frontier
RELAX_REALISTIC_TPS = 498.57990782684584  # ubel #450 c5oyb7gv realistic split-K hi (greedy-UNSAFE)
SIGMA_HW = 4.8153                          # hardware-variance envelope (~1% of deployed)
PPL_ANCHOR = 2.3772                        # PR #52 served PPL
PPL_GATE = 2.42                            # quality gate (reference PPL + 5%)
# TPS-gains re-cited from #457 / #458 (NOT re-derived here)
RELAX_REALISTIC_TPS_GAIN = 17.049907826845867  # #457 relax_prize_over_deployed_lo_tps
RELAX_CEILING_TPS_GAIN = 29.342423044997304    # #457 headroom_deployed_to_ceiling_tps
# #457 sigma-envelope LCB of the deployed->ceiling headroom (== gain - 1*sigma_hw). Used as a hard
# cross-check: my ceiling CI-clean threshold at k=1 MUST equal this banked number.
HEADROOM_DEPLOYED_TO_CEILING_LCB = 24.527123044997325  # #457 sigma_envelope (h0uggl9i)

# Deployed served severity (PR #52 2x9fm2zx; banked in directive4 shared_baselines)
DEPLOYED_IDENTITY = 0.9966                 # directive4 shared_baselines.deployed_served_identity
DEPLOYED_FLIPS_STR = "3/882"              # directive4 shared_baselines.deployed_served_flips
DEPLOYED_FLIPS = 3                         # numerator of "3/882" (#405 j6h228xy reduction-order)
DEPLOYED_TOKENS = 882                      # denominator of "3/882"

# Best realized byte-exact (greedy-SAFE) lever to date — the strongest STRICT alternative the relax
# move must beat to be worth going greedy-unsafe at all. PR-given baseline note (+0.26 TPS).
BEST_BYTE_EXACT_LEVER_TPS = 0.26

# Relax-realistic severity — PARAMETERIZED SLOTS (UNMEASURED; stark #452 to fill). Never confuse a
# placeholder for a measurement: numeric value is None, with an explicit sentinel string.
RELAX_IDENTITY_UB = 0.9966                 # upper bound for the monotone-degrading check
RELAX_FLIPS_SLOT_NOTE = ">=3, UNMEASURED — stark #452 to fill"
RELAX_PPL_SLOT_NOTE = "UNMEASURED — stark #452 quality run to fill"

# CI-clean confidence multipliers swept for the threshold table. k=1 is the headline (matches the
# #457 sigma-envelope LCB convention exactly); k=0 is the point boundary; k=2 is the 2-sigma guard.
K_LEVELS = (0, 1, 2)
K_HEADLINE = 1

TOL_RT = 1e-6                              # banked round-trip tolerance
TOL_TIGHT = 1e-9                           # derived-arithmetic tolerance


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# Load + round-trip the banked source JSONs (#457 / directive4 / #458). Re-derives nothing.
# --------------------------------------------------------------------------- #
def load_banked() -> dict[str, Any]:
    ceil = json.loads(CEILING_JSON.read_text(encoding="utf-8"))["synthesis"]
    c_head, c_const = ceil["headline"], ceil["constants"]
    c_relax, c_env = ceil["relax_prize"], ceil["sigma_envelope"]

    d4 = json.loads(DIRECTIVE4_JSON.read_text(encoding="utf-8"))["shared_baselines"]
    d4_flips_num = int(str(d4["deployed_served_flips"]).split("/")[0])

    led = json.loads(COST_LEDGER_JSON.read_text(encoding="utf-8"))["synthesis"]
    led_const = led["constants"]
    led_margin = led["ppl_ledger"]["deployed_ppl_gate_margin"]

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
        "relax_realistic_gain_resid": abs(
            c_relax["relax_prize_over_deployed_lo_tps"] - RELAX_REALISTIC_TPS_GAIN
        ),
        "relax_ceiling_gain_resid": abs(
            c_env["headroom_deployed_to_ceiling_tps"] - RELAX_CEILING_TPS_GAIN
        ),
        # #457 banked headroom LCB == gain - 1*sigma_hw. My ceiling CI-clean(k=1) must match this.
        "ceiling_lcb_resid": abs(
            c_env["headroom_deployed_to_ceiling_lcb_tps"] - HEADROOM_DEPLOYED_TO_CEILING_LCB
        ),
        # --- directive4 deployed-severity witnesses (PR #52 2x9fm2zx) ---
        "deployed_identity_resid": abs(d4["deployed_served_identity"] - DEPLOYED_IDENTITY),
        "deployed_flips_resid": abs(d4_flips_num - DEPLOYED_FLIPS),
        "deployed_ppl_xcheck_resid": abs(d4["deployed_ppl"] - PPL_ANCHOR),
        "ppl_cap_xcheck_resid": abs(d4["ppl_cap"] - PPL_GATE),
        "deployed_tps_xcheck_resid": abs(d4["deployed_tps"] - DEPLOYED_TPS),
        # --- #458 cost-ledger sibling (the RULE this card turns into thresholds) ---
        "led_relax_gain_resid": abs(led_const["relax_realistic_tps_gain"] - RELAX_REALISTIC_TPS_GAIN),
        "led_ceiling_gain_resid": abs(led_const["relax_ceiling_tps_gain"] - RELAX_CEILING_TPS_GAIN),
        "led_ppl_margin_resid": abs(led_margin - (PPL_GATE - PPL_ANCHOR)),
    }
    max_resid = max(rt.values())
    return {
        "roundtrip_resid": rt,
        "max_roundtrip_resid": max_resid,
        "all_roundtrip_ok": bool(max_resid <= TOL_RT),
        "ceiling_self_test_passed": bool(c_head.get("unified_ceiling_self_test_passes", False)),
        "cost_ledger_self_test_passed": bool(
            led["self_test"].get("cost_ledger_self_test_passes", False)
        ),
        "deployed_flips_str_banked": str(d4["deployed_served_flips"]),
        "banked_ceiling_lcb_tps": HEADROOM_DEPLOYED_TO_CEILING_LCB,
    }


# --------------------------------------------------------------------------- #
# (1a) TPS-gain decision-flip thresholds. CI-clean GO iff (gain - k*sigma_hw) >= bar B.
#      The largest CI-clean bar B is therefore exactly (gain - k*sigma_hw).
# --------------------------------------------------------------------------- #
def ci_clean_max_bar(gain: float, k: int) -> float:
    """Largest human bar B (TPS over deployed) for which `gain` is a CI-clean GO at k*sigma_hw.

    CI-clean GO  <=>  gain - k*sigma_hw >= B  <=>  B <= gain - k*sigma_hw.
    So the supremum admissible B is gain - k*sigma_hw (k=0 => the point boundary B = gain)."""
    return gain - k * SIGMA_HW


def tps_gain_thresholds() -> dict[str, Any]:
    realistic = {
        f"k{k}": ci_clean_max_bar(RELAX_REALISTIC_TPS_GAIN, k) for k in K_LEVELS
    }
    ceiling = {
        f"k{k}": ci_clean_max_bar(RELAX_CEILING_TPS_GAIN, k) for k in K_LEVELS
    }
    # headline: the bar above which relax-realistic STOPS being a CI-clean GO (k=1 sigma_hw).
    headline_threshold = realistic[f"k{K_HEADLINE}"]

    # Is relax-realistic a CI-clean GO against the strongest concrete STRICT alternative, the best
    # byte-exact (greedy-safe) lever B=+0.26? (i.e. is it even worth going greedy-unsafe at all?)
    ref_bar = BEST_BYTE_EXACT_LEVER_TPS
    realistic_ci_clean_vs_ref = bool(ci_clean_max_bar(RELAX_REALISTIC_TPS_GAIN, K_HEADLINE) >= ref_bar)

    # A small parameterized curve so fern can read GO/NO-GO at any candidate human bar B.
    bar_grid = [0.0, BEST_BYTE_EXACT_LEVER_TPS, 5.0, 10.0, headline_threshold, 15.0,
                RELAX_REALISTIC_TPS_GAIN, 20.0, 25.0, RELAX_CEILING_TPS_GAIN]
    bar_grid = sorted({round(b, 6) for b in bar_grid})
    curve = []
    for b in bar_grid:
        curve.append({
            "bar_B": b,
            "relax_realistic_point_go": bool(RELAX_REALISTIC_TPS_GAIN >= b),       # k=0
            "relax_realistic_ci_clean_go_k1": bool(ci_clean_max_bar(RELAX_REALISTIC_TPS_GAIN, 1) >= b),
            "relax_realistic_ci_clean_go_k2": bool(ci_clean_max_bar(RELAX_REALISTIC_TPS_GAIN, 2) >= b),
            "ceiling_ci_clean_go_k1": bool(ci_clean_max_bar(RELAX_CEILING_TPS_GAIN, 1) >= b),
        })

    return {
        "definition": ("CI-clean GO at confidence k <=> (gain - k*sigma_hw) >= human bar B. The "
                       "largest CI-clean bar is gain - k*sigma_hw. sigma_hw convention matches #457's "
                       "sigma-envelope (single sigma_hw on the headroom; deployed anchor treated fixed)."),
        "sigma_hw": SIGMA_HW,
        "relax_realistic_gain": RELAX_REALISTIC_TPS_GAIN,
        "ceiling_gain": RELAX_CEILING_TPS_GAIN,
        "relax_realistic_max_ci_clean_bar": realistic,    # {k0: 17.05, k1: 12.23, k2: 7.42}
        "ceiling_max_ci_clean_bar": ceiling,              # {k0: 29.34, k1: 24.53, k2: 19.71}
        "headline_k": K_HEADLINE,
        "decision_flip_tps_threshold": headline_threshold,  # PRIMARY, k=1 => 12.2346
        "reference_strict_bar_tps": ref_bar,                # best byte-exact lever +0.26
        "relax_realistic_ci_clean_of_threshold": realistic_ci_clean_vs_ref,  # vs ref bar +0.26
        "relax_realistic_ci_clean_of_threshold_bar": ref_bar,
        "bar_curve": curve,
        "ceiling_ci_clean_k1_equals_banked_lcb": bool(
            abs(ci_clean_max_bar(RELAX_CEILING_TPS_GAIN, 1) - HEADROOM_DEPLOYED_TO_CEILING_LCB)
            <= TOL_RT
        ),
        "note": ("relax-realistic +17.05 stays a CI-clean (k=1) GO for any human bar up to +12.23 "
                 "TPS; the ceiling +29.34 up to +24.53 (== #457 banked LCB). Both dwarf the best "
                 "byte-exact strict lever (+0.26), so the relax move clears every plausible bar "
                 "EXCEPT an aggressive >+12.23 (realistic) / >+24.53 (ceiling) one. The TPS clause is "
                 "NOT where this decision is tight."),
    }


# --------------------------------------------------------------------------- #
# (1b) PPL decision boundary. Hard gate; orthogonal to flip-count.
# --------------------------------------------------------------------------- #
def ppl_boundary() -> dict[str, Any]:
    margin = PPL_GATE - PPL_ANCHOR  # 2.42 - 2.3772 = 0.0428
    return {
        "max_admissible_relax_ppl": PPL_GATE,                 # HEADLINE 2.42
        "deployed_ppl_anchor": PPL_ANCHOR,
        "margin_from_deployed_anchor": margin,                # 0.0428
        "margin_pct_of_gate": 100.0 * margin / PPL_GATE,      # ~1.77%
        "rule": "relax is PPL-admissible iff stark #452's measured relax PPL <= 2.42",
        "note": ("the PPL gate is a HARD boundary at 2.42 with only 0.0428 of headroom above the "
                 "deployed anchor 2.3772 (~1.77% of the gate). A flip can be PPL-NEUTRAL (a "
                 "reduction-order near-tie, like the deployed 3) or PPL-BREACHING; the count tells "
                 "you NOTHING about which. PPL is the one axis that can flip GO->NO-GO outright."),
    }


# --------------------------------------------------------------------------- #
# (1c) flip-count framing: Δflips for ΔTPS vs the deployed 3-flip status quo, parameterized over N.
#      Decision is N-INVARIANT (orthogonal): the count is equivalence-severity, the GATE is PPL +
#      same-KIND, never the count.
# --------------------------------------------------------------------------- #
def flip_count_framing() -> dict[str, Any]:
    # A few illustrative N values stark #452 might report (N >= 3, since relax is off the strict
    # frontier at least as far as the deployed 3). delta_flips = N - deployed status quo (3).
    n_grid = [3, 4, 6, 10, 20]
    rows = []
    for n in n_grid:
        rows.append({
            "relax_flips_N": n,
            "delta_flips_vs_deployed_status_quo": n - DEPLOYED_FLIPS,  # N - 3
            "buys_delta_tps_realistic": RELAX_REALISTIC_TPS_GAIN,       # +17.05 (N-invariant)
            "decision_gate_depends_on_N": False,                       # ORTHOGONAL
            "gate_is": "PPL <= 2.42 AND same-KIND break (never the count N)",
        })
    return {
        "framing": ("relax buys ΔTPS = (measured gain over deployed) by moving the equivalence "
                    "severity from the deployed 3-flip status quo to N flips. The deployed 3 is the "
                    "status quo the human ALREADY accepted (#458 reframe) — so this is '3 -> N flips, "
                    "does quality survive?', graded, not binary."),
        "deployed_status_quo_flips": DEPLOYED_FLIPS,
        "parameterized_over_N": rows,
        "orthogonality": ("flip-COUNT (equivalence) and PPL (quality) are ORTHOGONAL. The decision "
                          "gate is PPL <= 2.42 AND same-KIND break; it does NOT read N. N is the "
                          "severity coordinate, reported for transparency, never a gate. Conflating "
                          "count with quality is the trap that sank four isolated levers (#458)."),
        "decision_is_N_invariant_given_kind_and_ppl": True,
    }


# --------------------------------------------------------------------------- #
# (3) Pre-wired stark #452 -> recommendation. Single collapse of (TPS, PPL, kind) -> verdict.
# --------------------------------------------------------------------------- #
def recommend(measured_gain_tps: float, measured_ppl: float, break_same_kind: bool,
              human_bar_tps: float, k: int = K_HEADLINE) -> str:
    """Collapse stark #452's three measurements + the human bar B into GO / NO-GO / CI-AMBIGUOUS.

    GO            : all three clauses hold AND the TPS gain is CI-clean of the bar.
    CI-AMBIGUOUS  : PPL + kind hold and the gain clears the bar at the POINT but not CI-clean
                    (B <= gain < B + k*sigma_hw) — the gain and the bar overlap within hw noise.
    NO-GO         : PPL breached, OR a new-kind break, OR the point gain is below the bar.
    """
    if measured_ppl > PPL_GATE:                       # clause-2 hard fail (quality)
        return "NO-GO"
    if not break_same_kind:                           # clause-3 hard fail (new failure mode)
        return "NO-GO"
    if (measured_gain_tps - k * SIGMA_HW) >= human_bar_tps:   # clause-1 CI-clean
        return "GO"
    if measured_gain_tps >= human_bar_tps:            # clause-1 point-only (within hw noise of bar)
        return "CI-AMBIGUOUS"
    return "NO-GO"


def prewire_stark452() -> dict[str, Any]:
    clause_map = [
        {
            "stark452_reports": "measured relax TPS (=> gain over deployed 481.53)",
            "resolves_clause": "clause-1 tps_gain_over_deployed_ge_bar",
            "rule": "CI-clean GO iff (gain - k*sigma_hw) >= human bar B; CI-AMBIGUOUS iff B <= gain < B + k*sigma_hw",
            "currently": "QUANTIFIED at the modeled +17.05 (realistic) / +29.34 (ceiling); only the human bar B is free",
        },
        {
            "stark452_reports": "measured relax PPL",
            "resolves_clause": "clause-2 measured_ppl_le_gate",
            "rule": "admissible iff measured PPL <= 2.42 (margin 0.0428 from deployed anchor 2.3772)",
            "currently": "PENDING — the decision-critical unknown (hard gate, thin margin)",
        },
        {
            "stark452_reports": "measured relax flip characterization (KIND of the N flips)",
            "resolves_clause": "clause-3 same_kind_of_break_as_deployed",
            "rule": "GO if the flips are accumulation-order near-ties (same family as the deployed 3); NO-GO if a new quality-destroying mode",
            "currently": "PROVISIONAL same-kind (FP split-K reassociation ~ deployed reduction-order near-ties); confirm via the KIND, not the count N",
        },
    ]

    # Worked corners: the verdict the instant stark #452 lands, for representative (gain, ppl, kind)
    # at the reference human bar B = best byte-exact lever (+0.26) and an aggressive bar (+20).
    ref_bar = BEST_BYTE_EXACT_LEVER_TPS
    aggressive_bar = 20.0
    corners = [
        {"label": "modeled gain, PPL-neutral (== deployed anchor), same-kind, ref bar +0.26",
         "gain": RELAX_REALISTIC_TPS_GAIN, "ppl": PPL_ANCHOR, "same_kind": True, "bar": ref_bar,
         "verdict": recommend(RELAX_REALISTIC_TPS_GAIN, PPL_ANCHOR, True, ref_bar)},
        {"label": "modeled gain, PPL just breaches gate (2.43), same-kind, ref bar +0.26",
         "gain": RELAX_REALISTIC_TPS_GAIN, "ppl": 2.43, "same_kind": True, "bar": ref_bar,
         "verdict": recommend(RELAX_REALISTIC_TPS_GAIN, 2.43, True, ref_bar)},
        {"label": "modeled gain, PPL at gate (2.42), same-kind, ref bar +0.26",
         "gain": RELAX_REALISTIC_TPS_GAIN, "ppl": PPL_GATE, "same_kind": True, "bar": ref_bar,
         "verdict": recommend(RELAX_REALISTIC_TPS_GAIN, PPL_GATE, True, ref_bar)},
        {"label": "modeled gain, PPL-neutral, NEW-kind break, ref bar +0.26",
         "gain": RELAX_REALISTIC_TPS_GAIN, "ppl": PPL_ANCHOR, "same_kind": False, "bar": ref_bar,
         "verdict": recommend(RELAX_REALISTIC_TPS_GAIN, PPL_ANCHOR, False, ref_bar)},
        {"label": "modeled gain, PPL-neutral, same-kind, bar +15 (in [12.23 k1-clean, 17.05 point])",
         "gain": RELAX_REALISTIC_TPS_GAIN, "ppl": PPL_ANCHOR, "same_kind": True, "bar": 15.0,
         "verdict": recommend(RELAX_REALISTIC_TPS_GAIN, PPL_ANCHOR, True, 15.0)},
        {"label": "modeled gain, PPL-neutral, same-kind, AGGRESSIVE bar +20 (> +17.05 point gain)",
         "gain": RELAX_REALISTIC_TPS_GAIN, "ppl": PPL_ANCHOR, "same_kind": True, "bar": aggressive_bar,
         "verdict": recommend(RELAX_REALISTIC_TPS_GAIN, PPL_ANCHOR, True, aggressive_bar)},
        {"label": "CEILING gain +29.34, PPL-neutral, same-kind, aggressive bar +20 (24.53 k1-clean >= 20)",
         "gain": RELAX_CEILING_TPS_GAIN, "ppl": PPL_ANCHOR, "same_kind": True, "bar": aggressive_bar,
         "verdict": recommend(RELAX_CEILING_TPS_GAIN, PPL_ANCHOR, True, aggressive_bar)},
    ]

    return {
        "one_number_swap": clause_map,
        "recommend_signature": "recommend(measured_gain_tps, measured_ppl, break_same_kind, human_bar_tps, k=1)",
        "verdict_space": ["GO", "CI-AMBIGUOUS", "NO-GO"],
        "live_status": ("TPS clause already a CI-clean GO for any bar <= +12.23 (realistic). The live "
                        "recommendation is GO-PENDING-PPL-AND-KIND: it resolves to GO the instant stark "
                        "#452 reads PPL <= 2.42 with same-kind flips (at any human bar <= +12.23), and "
                        "to NO-GO if PPL > 2.42 or the break is a new kind — regardless of the count N."),
        "worked_corners": corners,
    }


# --------------------------------------------------------------------------- #
# (2) Sensitivity / tornado over the three pending axes (measured TPS vs PPL vs flip-count).
#     Rank by how close the deployed nominal sits to each axis's decision-flip boundary, normalized
#     — the closer to its flip, the more a small measurement swings the recommendation.
# --------------------------------------------------------------------------- #
def tornado() -> dict[str, Any]:
    ref_bar = BEST_BYTE_EXACT_LEVER_TPS

    # PPL axis: nominal = deployed anchor 2.3772; flip at the gate 2.42. Distance-to-flip 0.0428,
    # normalized by the gate. Can flip GO->NO-GO within the plausible measured range. PIVOTAL.
    ppl_dist = PPL_GATE - PPL_ANCHOR                         # 0.0428
    ppl_norm = ppl_dist / PPL_GATE                           # ~0.0177

    # TPS axis: nominal gain = modeled realistic +17.05; CI-clean flip at bar + k*sigma_hw. At the
    # reference bar +0.26, the flip gain is 0.26 + 4.8153 = 5.0753. Distance-to-flip in gain space
    # = 17.05 - 5.08 = 11.97, normalized by the modeled gain. Far from flip => robust at ref bar.
    tps_flip_gain_at_ref = ref_bar + K_HEADLINE * SIGMA_HW   # 5.0753
    tps_dist = RELAX_REALISTIC_TPS_GAIN - tps_flip_gain_at_ref  # 11.9746
    tps_norm = tps_dist / RELAX_REALISTIC_TPS_GAIN           # ~0.702

    # flip-count axis: the decision is N-INVARIANT (orthogonal). No finite flip boundary in N =>
    # infinite distance-to-flip => zero pivotality. The KIND is a separate categorical gate (a flip
    # of the SAME family is admissible; a new family is not) — but that is the KIND, not the count.
    axes = [
        {
            "axis": "measured_ppl",
            "nominal": PPL_ANCHOR,
            "flip_boundary": PPL_GATE,
            "distance_to_flip": ppl_dist,
            "normalized_distance_to_flip": ppl_norm,        # smallest => most sensitive
            "can_flip_decision_in_plausible_range": True,
            "swing": "FULL (GO<->NO-GO): a hard gate; measured PPL in [2.3772, breach] straddles 2.42",
            "rank": 1,
            "why": ("razor-thin 0.0428 margin (~1.77% of the gate) and UNPREDICTABLE — flips can be "
                    "PPL-neutral OR PPL-breaching, so the measurement alone decides. The single most "
                    "decision-swinging pending input."),
        },
        {
            "axis": "measured_tps_gain",
            "nominal": RELAX_REALISTIC_TPS_GAIN,
            "flip_boundary": tps_flip_gain_at_ref,
            "flip_boundary_note": "CI-clean flip = human bar B + k*sigma_hw; shown at ref bar B=+0.26, k=1",
            "distance_to_flip": tps_dist,
            "normalized_distance_to_flip": tps_norm,
            "can_flip_decision_in_plausible_range": True,
            "swing": ("CONDITIONAL: flips NO-GO only if the measured gain craters below ~+5.08 OR the "
                      "human sets an aggressive bar > +12.23 (k1). Robustly GO at any modest bar."),
            "rank": 2,
            "why": ("modeled +17.05 sits ~12 TPS (2.5 sigma_hw) above the CI-clean flip at the "
                    "reference bar; the TPS clause is not where the decision is tight unless the "
                    "human's bar B is aggressive."),
        },
        {
            "axis": "measured_flip_count",
            "nominal": DEPLOYED_FLIPS,
            "flip_boundary": None,
            "distance_to_flip": None,                       # no finite flip boundary (orthogonal)
            "normalized_distance_to_flip": None,            # => least sensitive (sorts last)
            "can_flip_decision_in_plausible_range": False,
            "swing": "NONE on the COUNT: the decision is N-invariant (orthogonal). Only the KIND gates.",
            "rank": 3,
            "why": ("flip-COUNT is equivalence-severity, kept ORTHOGONAL to the PPL/quality gate "
                    "(#458). The count N never flips the recommendation; only the categorical KIND "
                    "(same accumulation-order family vs a new quality-destroying mode) does, and that "
                    "is captured by clause-3, not by N."),
        },
    ]
    # rank by normalized distance-to-flip ascending (closest to flip = most sensitive); the
    # orthogonal flip-count axis (None) has no finite boundary => sorts last (least sensitive).
    def _nd_key(a: dict) -> float:
        nd = a["normalized_distance_to_flip"]
        return math.inf if nd is None else nd
    ranked = sorted(axes, key=_nd_key)
    order = [a["axis"] for a in ranked]
    most_sensitive = order[0]
    monotone_rank_ok = bool(order == ["measured_ppl", "measured_tps_gain", "measured_flip_count"])
    return {
        "axes": axes,
        "ranked_axes_most_to_least_sensitive": order,
        "most_sensitive_pending_input": "ppl" if most_sensitive == "measured_ppl" else most_sensitive,
        "tornado_rank_matches_pivotality": monotone_rank_ok,
        "justified_region": {
            "description": ("relax is JUSTIFIED in the region: {measured PPL <= 2.42} AND {break is "
                            "same accumulation-order KIND} AND {measured gain - k*sigma_hw >= human "
                            "bar B}. It is a half-space below the PPL gate, intersected with the "
                            "same-KIND categorical set, intersected with the CI-clean TPS half-space. "
                            "The flip-count N does not bound the region (orthogonal)."),
            "ppl_halfspace": "measured_ppl <= 2.42",
            "kind_set": "break in {accumulation-order near-tie family}",
            "tps_halfspace_k1": "measured_gain >= human_bar_B + 4.8153",
            "n_flips": "free (does not bound the justified region)",
        },
        "summary": ("TORNADO (most -> least decision-swinging): PPL (hard gate, 0.0428 margin, "
                    "outcome-flipping) > TPS-gain (robust GO unless bar aggressive) > flip-count "
                    "(orthogonal; only the KIND gates). The decision turns almost entirely on stark "
                    "#452's PPL read."),
    }


# --------------------------------------------------------------------------- #
# (4) Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def selftests(banked: dict, tps: dict, ppl: dict, flips: dict, tor: dict, wire: dict) -> dict[str, Any]:
    # (a) every banked source number round-trips its committed JSON within tol.
    cond_a = bool(banked["all_roundtrip_ok"])

    # (b) ceiling CI-clean threshold at k=1 EQUALS #457's banked headroom LCB (24.527123). Hard
    #     cross-check that my CI arithmetic is the same convention as the parent card.
    cond_b = bool(
        tps["ceiling_ci_clean_k1_equals_banked_lcb"]
        and abs(ci_clean_max_bar(RELAX_CEILING_TPS_GAIN, 1) - HEADROOM_DEPLOYED_TO_CEILING_LCB) <= TOL_RT
    )

    # (c) decision_flip_tps_threshold == relax-realistic gain - sigma_hw, == 12.2346, and is nested
    #     k0 (point) >= k1 (headline) >= k2 (2-sigma) for BOTH realistic and ceiling.
    headline = tps["decision_flip_tps_threshold"]
    r = tps["relax_realistic_max_ci_clean_bar"]
    c = tps["ceiling_max_ci_clean_bar"]
    cond_c = bool(
        abs(headline - (RELAX_REALISTIC_TPS_GAIN - SIGMA_HW)) <= TOL_TIGHT
        and abs(headline - 12.234607826845867) <= 1e-6
        and r["k0"] >= r["k1"] >= r["k2"]
        and c["k0"] >= c["k1"] >= c["k2"]
    )

    # (d) max_admissible_relax_ppl == gate == 2.42; margin == 0.0428 == gate - anchor.
    cond_d = bool(
        ppl["max_admissible_relax_ppl"] == PPL_GATE
        and abs(ppl["margin_from_deployed_anchor"] - 0.0428) <= TOL_TIGHT
        and abs(ppl["margin_from_deployed_anchor"] - (PPL_GATE - PPL_ANCHOR)) <= TOL_TIGHT
    )

    # (e) tornado puts PPL first; most_sensitive == "ppl"; normalized distances strictly increasing
    #     ppl < tps < flips (PPL closest to its flip boundary).
    norm = {a["axis"]: a["normalized_distance_to_flip"] for a in tor["axes"]}
    cond_e = bool(
        tor["most_sensitive_pending_input"] == "ppl"
        and tor["tornado_rank_matches_pivotality"]
        and norm["measured_ppl"] < norm["measured_tps_gain"]   # PPL closest to its flip boundary
        and norm["measured_flip_count"] is None                # flip-count orthogonal (no boundary)
    )

    # (f) flip-count ORTHOGONAL: recommend() is invariant under N (the recommend() signature does not
    #     take N), and flip_count_framing declares N-invariance, and the tornado flip-count axis
    #     cannot flip the decision.
    flip_axis = next(a for a in tor["axes"] if a["axis"] == "measured_flip_count")
    cond_f = bool(
        flips["decision_is_N_invariant_given_kind_and_ppl"]
        and flip_axis["can_flip_decision_in_plausible_range"] is False
        and "measured_flip_count" not in recommend.__code__.co_varnames[:recommend.__code__.co_argcount]
    )

    # (g) recommend() monotonicity sanity:
    #     - PPL breach (> 2.42) => NO-GO regardless of gain / kind.
    #     - new-kind break => NO-GO regardless of gain / PPL.
    #     - holding PPL+kind admissible, raising the gain never turns GO into NO-GO.
    ref = BEST_BYTE_EXACT_LEVER_TPS
    mono_ppl = recommend(RELAX_CEILING_TPS_GAIN, 2.50, True, ref) == "NO-GO"
    mono_kind = recommend(RELAX_CEILING_TPS_GAIN, PPL_ANCHOR, False, ref) == "NO-GO"
    go_low = recommend(RELAX_REALISTIC_TPS_GAIN, PPL_ANCHOR, True, ref)
    go_hi = recommend(RELAX_CEILING_TPS_GAIN, PPL_ANCHOR, True, ref)
    mono_gain = (go_low == "GO" and go_hi == "GO")
    cond_g = bool(mono_ppl and mono_kind and mono_gain)

    # (h) worked corners land as expected (modeled+neutral+same-kind+ref-bar => GO; breach => NO-GO;
    #     gate-exact 2.42 admissible => GO; new-kind => NO-GO; aggressive bar +20 with +17.05 => not
    #     CI-clean => CI-AMBIGUOUS; ceiling +29.34 at +20 => GO).
    cmap = {cn["label"]: cn["verdict"] for cn in wire["worked_corners"]}
    cond_h = bool(
        cmap["modeled gain, PPL-neutral (== deployed anchor), same-kind, ref bar +0.26"] == "GO"
        and cmap["modeled gain, PPL just breaches gate (2.43), same-kind, ref bar +0.26"] == "NO-GO"
        and cmap["modeled gain, PPL at gate (2.42), same-kind, ref bar +0.26"] == "GO"
        and cmap["modeled gain, PPL-neutral, NEW-kind break, ref bar +0.26"] == "NO-GO"
        and cmap["modeled gain, PPL-neutral, same-kind, bar +15 (in [12.23 k1-clean, 17.05 point])"] == "CI-AMBIGUOUS"
        and cmap["modeled gain, PPL-neutral, same-kind, AGGRESSIVE bar +20 (> +17.05 point gain)"] == "NO-GO"
        and cmap["CEILING gain +29.34, PPL-neutral, same-kind, aggressive bar +20 (24.53 k1-clean >= 20)"] == "GO"
    )

    # (i) ppl anchor preserved (this leg does not touch the served model) and within gate.
    cond_i = bool(abs(PPL_ANCHOR - 2.3772) <= TOL_RT and PPL_ANCHOR <= PPL_GATE)

    # (j) NaN-clean — set by the caller after the full payload walk.
    cond_j = True

    conditions = {
        "a_all_banked_numbers_roundtrip": cond_a,
        "b_ceiling_ci_clean_k1_equals_banked_lcb_24p527": cond_b,
        "c_decision_flip_threshold_correct_and_k_nested": cond_c,
        "d_max_admissible_relax_ppl_2p42_margin_0p0428": cond_d,
        "e_tornado_ppl_most_sensitive": cond_e,
        "f_flip_count_orthogonal_decision_N_invariant": cond_f,
        "g_recommend_monotone_gates": cond_g,
        "h_worked_corners_as_expected": cond_h,
        "i_ppl_anchor_preserved_within_gate": cond_i,
        "j_nan_clean": cond_j,
    }
    return {
        "conditions": conditions,
        "decision_surface_self_test_passes": bool(all(conditions.values())),
        "detail": {
            "max_roundtrip_resid": banked["max_roundtrip_resid"],
            "decision_flip_tps_threshold": headline,
            "ceiling_ci_clean_k1": ci_clean_max_bar(RELAX_CEILING_TPS_GAIN, 1),
            "banked_ceiling_lcb": HEADROOM_DEPLOYED_TO_CEILING_LCB,
            "max_admissible_relax_ppl": PPL_GATE,
            "most_sensitive_pending_input": tor["most_sensitive_pending_input"],
        },
    }


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    banked = load_banked()
    tps = tps_gain_thresholds()
    ppl = ppl_boundary()
    flips = flip_count_framing()
    tor = tornado()
    wire = prewire_stark452()
    st = selftests(banked, tps, ppl, flips, tor, wire)

    headline = {
        "decision_surface_self_test_passes": bool(st["decision_surface_self_test_passes"]),  # PRIMARY gate
        "decision_flip_tps_threshold": tps["decision_flip_tps_threshold"],          # PRIMARY 12.2346 (k1)
        "relax_realistic_ci_clean_of_threshold": tps["relax_realistic_ci_clean_of_threshold"],  # vs +0.26
        "max_admissible_relax_ppl": ppl["max_admissible_relax_ppl"],                # 2.42
        "ppl_margin_from_deployed_anchor": ppl["margin_from_deployed_anchor"],      # 0.0428
        "most_sensitive_pending_input": tor["most_sensitive_pending_input"],        # "ppl"
        "relax_realistic_max_ci_clean_bar_k1": tps["relax_realistic_max_ci_clean_bar"]["k1"],  # 12.2346
        "ceiling_max_ci_clean_bar_k1": tps["ceiling_max_ci_clean_bar"]["k1"],       # 24.5271 == banked LCB
        "ceiling_ci_clean_k1_equals_banked_lcb": tps["ceiling_ci_clean_k1_equals_banked_lcb"],
        "decision_is_N_invariant": flips["decision_is_N_invariant_given_kind_and_ppl"],
        "ppl": PPL_ANCHOR,
        "analysis_only": True,
        "no_served_file_change": True,
        "official_tps": 0,
    }
    verdict = (
        f"RELAX-DECISION-SURFACE-FLIP@B={tps['decision_flip_tps_threshold']:.2f}-CIclean(k1)-"
        f"PPLgate{PPL_GATE}(margin{ppl['margin_from_deployed_anchor']:.4f})-"
        f"MOST-SENSITIVE={tor['most_sensitive_pending_input'].upper()}-N-INVARIANT"
    )
    handoff = (
        f"DECISION SURFACE (for fern #357): relax-realistic +{RELAX_REALISTIC_TPS_GAIN:.2f} is a "
        f"CI-clean (k1) GO for any human bar B <= +{tps['relax_realistic_max_ci_clean_bar']['k1']:.2f} "
        f"(ceiling +{RELAX_CEILING_TPS_GAIN:.2f} up to +{tps['ceiling_max_ci_clean_bar']['k1']:.2f} == "
        f"#457 banked LCB). PPL gate is the tight axis: max admissible {PPL_GATE}, only "
        f"{ppl['margin_from_deployed_anchor']:.4f} over the deployed anchor; MOST-SENSITIVE input = "
        f"stark #452 PPL. flip-count N is ORTHOGONAL (decision N-invariant; only the KIND gates). "
        f"recommend(gain, ppl, same_kind, bar, k=1): GO iff PPL<=2.42 AND same-kind AND gain-k*sigma>=B; "
        f"NO-GO if PPL>2.42 or new-kind; CI-AMBIGUOUS if gain clears B only within hw noise."
    )
    return {
        "headline": headline,
        "tps_gain_thresholds": tps,
        "ppl_boundary": ppl,
        "flip_count_framing": flips,
        "tornado": tor,
        "prewire_stark452": wire,
        "banked_roundtrip": banked,
        "self_test": st,
        "constants": {
            "unified_ceiling_tps": UNIFIED_CEILING_TPS, "spec_ub_tps": SPEC_UB_TPS,
            "deployed_tps": DEPLOYED_TPS, "strict_frontier_tps": STRICT_FRONTIER_TPS,
            "relax_realistic_tps": RELAX_REALISTIC_TPS, "sigma_hw": SIGMA_HW,
            "ppl_anchor": PPL_ANCHOR, "ppl_gate": PPL_GATE,
            "deployed_identity": DEPLOYED_IDENTITY, "deployed_flips": DEPLOYED_FLIPS,
            "deployed_flips_str": DEPLOYED_FLIPS_STR, "deployed_tokens": DEPLOYED_TOKENS,
            "relax_identity_ub": RELAX_IDENTITY_UB,
            "relax_realistic_tps_gain": RELAX_REALISTIC_TPS_GAIN,
            "relax_ceiling_tps_gain": RELAX_CEILING_TPS_GAIN,
            "headroom_deployed_to_ceiling_lcb": HEADROOM_DEPLOYED_TO_CEILING_LCB,
            "best_byte_exact_lever_tps": BEST_BYTE_EXACT_LEVER_TPS,
            "k_headline": K_HEADLINE,
        },
        "verdict": verdict,
        "handoff_line": handoff,
        "imports": {
            "provenance": (
                "land#457 h0uggl9i (unified ceiling 510.87 +/- 4.82, spec-UB 520.95, deployed 481.53, "
                "strict 467.14, relax-realistic 498.58, sigma_hw 4.8153, headroom LCB 24.527, ppl_anchor "
                "2.3772, ppl_gate 2.42, relax TPS-gains +17.05/+29.34) x land#458 uhhyec0q (cost-ledger "
                "RULE this card turns into thresholds; deployed_ppl_gate_margin 0.0428) x "
                "directive4_correct_bar shared_baselines (deployed_served_identity 0.9966, "
                "deployed_served_flips 3/882, deployed_ppl 2.3772, ppl_cap 2.42 — PR #52 2x9fm2zx). "
                "relax identity/flips/PPL are stark #452 PARAMETERIZED SLOTS. All run-ids in "
                "wandb-applied-ai-team/gemma-challenge-senpai."),
            "machinery": "round-trip of committed #457 / #458 / directive4 result JSONs; re-derives nothing",
        },
    }


# --------------------------------------------------------------------------- #
# NaN guard + report + W&B (mirrors #458; never fatal).
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
    tps, ppl, flips = syn["tps_gain_thresholds"], syn["ppl_boundary"], syn["flip_count_framing"]
    tor, wire, st = syn["tornado"], syn["prewire_stark452"], syn["self_test"]
    print("\n" + "=" * 100, flush=True)
    print("RELAX-DECISION SURFACE (PR #462, land) — decision-flip thresholds + tornado, CPU-only",
          flush=True)
    print("=" * 100, flush=True)
    print("  (1a) TPS-GAIN CI-CLEAN DECISION-FLIP BARS (largest human bar B for a CI-clean GO):", flush=True)
    print(f"       {'k(sigma_hw)':>12}  {'relax-realistic (+17.05)':>26}  {'ceiling (+29.34)':>18}", flush=True)
    for k in K_LEVELS:
        r = tps["relax_realistic_max_ci_clean_bar"][f"k{k}"]
        c = tps["ceiling_max_ci_clean_bar"][f"k{k}"]
        tag = "  <- HEADLINE (== #457 LCB for ceiling)" if k == K_HEADLINE else ""
        print(f"       {k:>12}  {r:>26.4f}  {c:>18.4f}{tag}", flush=True)
    print(f"       PRIMARY decision_flip_tps_threshold = {tps['decision_flip_tps_threshold']:.4f} "
          f"(relax-realistic, k=1); CI-clean vs best byte-exact bar +{tps['reference_strict_bar_tps']} "
          f"= {tps['relax_realistic_ci_clean_of_threshold']}", flush=True)
    print(f"       ceiling CI-clean(k1) == #457 banked LCB 24.527 : {tps['ceiling_ci_clean_k1_equals_banked_lcb']}",
          flush=True)
    print("-" * 100, flush=True)
    print(f"  (1b) PPL BOUNDARY: max admissible relax-PPL = {ppl['max_admissible_relax_ppl']} "
          f"(margin {ppl['margin_from_deployed_anchor']:.4f} from anchor {ppl['deployed_ppl_anchor']}, "
          f"~{ppl['margin_pct_of_gate']:.2f}% of gate)", flush=True)
    print(f"       rule: {ppl['rule']}", flush=True)
    print("-" * 100, flush=True)
    print(f"  (1c) FLIP-COUNT FRAMING (Δflips for ΔTPS vs deployed {flips['deployed_status_quo_flips']}-flip "
          f"status quo; N-invariant={flips['decision_is_N_invariant_given_kind_and_ppl']}):", flush=True)
    for row in flips["parameterized_over_N"]:
        print(f"       N={row['relax_flips_N']:>3}  Δflips={row['delta_flips_vs_deployed_status_quo']:>3}  "
              f"buys +{row['buys_delta_tps_realistic']:.2f} TPS  gate-depends-on-N={row['decision_gate_depends_on_N']}",
              flush=True)
    print("-" * 100, flush=True)
    print("  (2) TORNADO (most -> least decision-swinging pending input):", flush=True)
    for a in sorted(tor["axes"], key=lambda x: (math.inf if x["normalized_distance_to_flip"] is None
                                                else x["normalized_distance_to_flip"])):
        nd = a["normalized_distance_to_flip"]
        nd_s = "inf" if nd is None else f"{nd:.4f}"
        print(f"       #{a['rank']} {a['axis']:<20} norm-dist-to-flip={nd_s:>7}  "
              f"can-flip={a['can_flip_decision_in_plausible_range']}  swing={a['swing'][:46]}", flush=True)
    print(f"       most_sensitive_pending_input = {tor['most_sensitive_pending_input']}", flush=True)
    print("-" * 100, flush=True)
    print("  (3) PRE-WIRED stark #452 -> recommendation (one-number-swap):", flush=True)
    for cm in wire["one_number_swap"]:
        print(f"       {cm['stark452_reports']:<46} -> {cm['resolves_clause']}", flush=True)
    print(f"       live: {wire['live_status'][:92]}", flush=True)
    for cn in wire["worked_corners"]:
        print(f"         [{cn['verdict']:<12}] {cn['label']}", flush=True)
    print("-" * 100, flush=True)
    print(f"  (4) PRIMARY decision_surface_self_test_passes = {st['decision_surface_self_test_passes']}",
          flush=True)
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
        print(f"[decision-surface] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    tps, ppl, flips = syn["tps_gain_thresholds"], syn["ppl_boundary"], syn["flip_count_framing"]
    tor, st = syn["tornado"], syn["self_test"]
    run = init_wandb_run(
        job_type="relax-decision-surface",
        agent="land",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["relax-decision-surface", "equivalence-escalation-anchors", "decision-flip-thresholds",
              "tornado-sensitivity", "ci-clean-go", "ppl-gate-boundary", "stark452-prewire",
              "capstone-anchor", "analysis-only", "bank-the-analysis"],
        config={
            "unified_ceiling_tps": UNIFIED_CEILING_TPS, "spec_ub_tps": SPEC_UB_TPS,
            "deployed_tps": DEPLOYED_TPS, "strict_frontier_tps": STRICT_FRONTIER_TPS,
            "relax_realistic_tps": RELAX_REALISTIC_TPS, "sigma_hw": SIGMA_HW,
            "ppl_anchor": PPL_ANCHOR, "ppl_gate": PPL_GATE,
            "relax_realistic_tps_gain": RELAX_REALISTIC_TPS_GAIN,
            "relax_ceiling_tps_gain": RELAX_CEILING_TPS_GAIN,
            "headroom_deployed_to_ceiling_lcb": HEADROOM_DEPLOYED_TO_CEILING_LCB,
            "best_byte_exact_lever_tps": BEST_BYTE_EXACT_LEVER_TPS,
            "k_headline": K_HEADLINE,
            "wandb_group": args.wandb_group,
            "source_runs": "land#457 h0uggl9i, land#458 uhhyec0q, directive4 shared_baselines "
                           "(PR#52 2x9fm2zx), stark#452 (relax slots, UNMEASURED)",
        },
    )
    if run is None:
        print("[decision-surface] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "decision_surface_self_test_passes": int(bool(st["decision_surface_self_test_passes"])),  # PRIMARY gate
        "decision_flip_tps_threshold": tps["decision_flip_tps_threshold"],          # PRIMARY 12.2346
        "relax_realistic_ci_clean_of_threshold": int(bool(tps["relax_realistic_ci_clean_of_threshold"])),
        "relax_realistic_ci_clean_of_threshold_bar": tps["relax_realistic_ci_clean_of_threshold_bar"],
        "max_admissible_relax_ppl": ppl["max_admissible_relax_ppl"],                # 2.42
        "ppl_margin_from_deployed_anchor": ppl["margin_from_deployed_anchor"],      # 0.0428
        "ppl_margin_pct_of_gate": ppl["margin_pct_of_gate"],
        "relax_realistic_max_ci_clean_bar_k0": tps["relax_realistic_max_ci_clean_bar"]["k0"],
        "relax_realistic_max_ci_clean_bar_k1": tps["relax_realistic_max_ci_clean_bar"]["k1"],
        "relax_realistic_max_ci_clean_bar_k2": tps["relax_realistic_max_ci_clean_bar"]["k2"],
        "ceiling_max_ci_clean_bar_k1": tps["ceiling_max_ci_clean_bar"]["k1"],       # 24.5271
        "ceiling_ci_clean_k1_equals_banked_lcb": int(bool(tps["ceiling_ci_clean_k1_equals_banked_lcb"])),
        "decision_is_N_invariant": int(bool(flips["decision_is_N_invariant_given_kind_and_ppl"])),
        "tornado_rank_matches_pivotality": int(bool(tor["tornado_rank_matches_pivotality"])),
        "relax_realistic_tps_gain": RELAX_REALISTIC_TPS_GAIN,
        "relax_ceiling_tps_gain": RELAX_CEILING_TPS_GAIN,
        "deployed_tps": DEPLOYED_TPS,
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
    # most_sensitive_pending_input is a string headline — log it on the run summary directly too.
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    try:
        run.summary["most_sensitive_pending_input"] = tor["most_sensitive_pending_input"]
        run.summary["verdict"] = syn["verdict"]
    except Exception:
        pass
    log_json_artifact(run, name="relax_decision_surface_result", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    print(f"[decision-surface] wandb logged {len(summary)} keys; run id {getattr(run, 'id', '?')}",
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
        "created_at": created_at, "pr": 462, "agent": "land",
        "kind": "relax-decision-surface", "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _nan_paths(payload)
    payload["nan_clean"] = not nan_paths
    syn["self_test"]["conditions"]["j_nan_clean"] = not nan_paths
    syn["self_test"]["decision_surface_self_test_passes"] = bool(
        all(syn["self_test"]["conditions"].values()))
    syn["headline"]["decision_surface_self_test_passes"] = syn["self_test"]["decision_surface_self_test_passes"]
    if nan_paths:
        print(f"[decision-surface] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "relax_decision_surface_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[decision-surface] wrote {out_path}", flush=True)

    st_path = out_dir / "relax_decision_surface_selftest.json"
    with st_path.open("w", encoding="utf-8") as fh:
        json.dump(syn["self_test"]["conditions"], fh, indent=2, sort_keys=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = (syn["self_test"]["decision_surface_self_test_passes"] and payload["nan_clean"])
        print(f"[decision-surface] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
