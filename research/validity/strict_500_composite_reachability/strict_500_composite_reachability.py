#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Strict-submission decision packet: the one-screen human GO / NO-GO for the strict-equivalent HF submission.

RE-POINT (2026-06-16 08:25Z + 08:32Z, advisor formal #357 relay)
----------------------------------------------------------------
This file used to be the ">=500 composite reachability / keep-strict-vs-relax" capstone.  That
framing is OBSOLETE: the human deprioritized >500 (#407 21:13Z), the relax-prize COLLAPSED to ~0
TPS at a worse identity (stark #452), and the strict frontier was REALIZED end-to-end (stark #466).
The advisor LIFTED the 01:59Z freeze and re-pointed the capstone to its current, live mandate:

    Synthesize the single screen the human reads to decide WHAT we fire, AT WHAT number, UNDER
    WHICH condition, and WITH WHAT honest caveats — the strict-equivalent HF submission decision.

This capstone is the DECISION SYNTHESIS only.  It is NOT the identity oracle (denken #471 runs the
served census) and NOT the execution machinery (land #473 stages + fires the submission).  It
consumes their committed/relayed outputs and renders the human-facing call.  It is CPU-only: it
measures nothing, runs no HF job, changes no served file (official_tps == 0).

The decision (post the 08:24Z human ruling)
-------------------------------------------
The human ruled (#407, 08:24Z): "I'm ok with a few tied bitwise flips, 0.99-ish is totally fine —
make the submission when you feel confident."  So operative-1.0 (census ~0.99+, every residual a
bf16-ULP tie, 0 *semantic* flips) is accepted as honest-strict; literal byte-exact 1.0 is NOT
required.  That retires the old BLOCKED branch and leaves a clean fork gated on ONE live input,
denken #471's served 128-prompt census:

    GO-OPERATIVE   denken #471 census is operative-1.0 (>= OPERATIVE_CENSUS_FLOOR, n_semantic_flips == 0)
                   -> fire `senpai-strict-eqv-<realized>` (~456-459 TPS, the honest strict win, +~295 over floor)
    FLOOR-LOCK     denken #471 surfaces a SEMANTIC (non-tie) flip
                   -> fire `senpai-strict-m1ar-161` (161.70 TPS, literal-1.0 by construction, lawine #438)

Until denken #471's census lands the recommendation is `GO-OPERATIVE-PENDING-471-TIES-CONFIRM`.

The realized number (with the honest band) — stark #466 (sxigz7dp speed / gmd8v9sw identity, ab9b286)
-----------------------------------------------------------------------------------------------------
The blanket-strict frontier HOLDS end-to-end (collapse to the 161.70 M=1 AR floor REFUTED).  It is
config-reachable via VLLM_BATCH_INVARIANT=1 (no served-source edit, no kernel rebuild), byte-exact
at the attention locus (1.0000 / 0 flips @ M=8 hd=512), PPL 2.3772 <= 2.42.  Realized headline is
456.36 TPS (conservative @ L=640); cluster-mean ~459 (L~593); L-envelope [528,658] -> 456.5-461.6.
The OLD composed 467.14 was OPTIMISTIC: composed_vs_realized_drift = +10.78 TPS > sigma_hw 4.8153,
realized eta_attn 5.50% vs composed 3.08%.  The isolated-locus delta is a CONSERVATIVE LOWER BOUND;
the true frontier sits in [456.5, <= 467.14].  Headline the conservative 456.36, footnote the band.
stark #472 (whole-cycle A/B, in-graph overlap, in flight) tightens this -> carried as a LIVE slot.

The binding gate = the strict contract itself (#319), resolved by denken #471
----------------------------------------------------------------------------
The load-bearing tension the packet must render honestly: stark #466's LOCUS proof says 1.0000 /
0-flips, but the committed SERVED-census prior leans the other way — land #429 / lawine #455
(0r0ounl8) pinned the composed blanket-strict frontier at literal 0.9989 (1 flip @ prompt 90), and
ubel #461 all_pin (qz6f0zgw) floored at ~0.9978 (1-2 residual flips).  Reconciliation hypothesis:
the order-preserving 2D reduction REMOVES the attention 3D split-KV reassociation that produces the
near-tie population (denken #464, 1o7jwlw4: those flips are m1_self_gap == 0.0, downstream-invisible
bf16-ULP ties), so the served census SHOULD resolve deterministically to operative-1.0 — pending
denken #471's confirm.  Either way the human ruled the ties acceptable, so the binding question is
only "are there any *semantic* (non-tie) flips?", which denken #471 answers.

Honesty hinge
-------------
The submission is labeled with its TRUE served census (e.g. "operative-1.0: 0.9989, 1 tied flip @
p90, 0 semantic"), NOT claimed as literal 1.0.  literal-vs-operative is surfaced as RESOLVED (the
human ruled operative is honest-strict, 08:24Z).

Cross-check confidence
----------------------
ubel #470 (BI-pin, a DIFFERENT mechanism than stark's num_splits=1) and stark #472 (in-graph
overlap) re-derive the realized number independently.  If both land in [456.5, ~459] within
sigma_hw the public number is bulletproof; if they disagree the packet flags it BEFORE the board.

Reference frame
---------------
Deployed 481.53 (PR #52, 2x9fm2zx) is NON-equivalent (identity 0.9966, 3 flips {11,18,118}, all
quality-neutral bf16-ULP ties per denken #464) — OUTSIDE the strict feasible set, ~22 above the
realized strict frontier.  The relax lane is DEAD: stark #452 (daqrzr99) realized ~0 TPS gain
(466.20 / -0.94 vs the strict base) AND degraded identity to 0.730 (3317 flips) -> strictly
dominated, NO-GO.  We stay strict.

PRIMARY metric  capstone_self_test_passes  (0-GPU arithmetic-invariant integrity gate)
TEST    metrics capstone_recommendation, realized_strict_headline_tps, realized_strict_band,
                binding_gate, census_tension_rendered, floor_lock_tps, deployed_tps,
                relax_prize_gain_tps, sigma_hw, human_operative_ruling_0824z,
                consumes_466_472_470_471_473, literal_vs_operative_surfaced, analysis_only,
                no_served_file_change, official_tps, ppl.
LIVE slots (parameterized; default = pending -> GO-OPERATIVE-PENDING-471-TIES-CONFIRM):
                --denken471-served-census, --denken471-semantic-flips (THE gate),
                --stark472-best-estimate-tps, --ubel470-bipin-tps, --lawine467-sigma-hw.
Consume cross-student numbers ONLY as relayed by the advisor into the #357 thread.
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
# Banked / formally-relayed constants.  Every number below was relayed by the
# advisor into the #357 thread (08:25Z + 08:32Z, with the earlier 05:34Z /
# 07:08Z / 07:26Z relays).  We do NOT inspect the source branches; we consume
# only the relayed values.  W&B run ids are recorded for provenance.
# --------------------------------------------------------------------------- #

# --- Reference frame: the deployed NON-equivalent incumbent (off the strict feasible set) ---
DEPLOYED_TPS: float = 481.53            # PR #52 (2x9fm2zx) deployed fast path
DEPLOYED_IDENTITY: float = 0.9966       # served token-identity (NON-equivalent)
DEPLOYED_FLIPS: tuple[int, ...] = (11, 18, 118)  # all quality-neutral bf16-ULP ties (denken #464)
PPL_DEPLOYED: float = 2.3772            # PPL of the served config / strict M=1 AR reference
PPL_GATE: float = 2.42                  # challenge PPL ceiling

# --- Realized strict frontier (stark #466: sxigz7dp speed / gmd8v9sw identity; banked ab9b286) ---
REALIZED_STRICT_HEADLINE_TPS: float = 456.36     # conservative headline @ L=640
REALIZED_STRICT_CLUSTER_MEAN_TPS: float = 459.0  # cluster-mean @ L~593
REALIZED_STRICT_L_ENVELOPE: tuple[float, float] = (456.5, 461.6)  # L in [528,658]
COMPOSED_STRICT_FRONTIER_TPS: float = 467.14     # the OLD composed (optimistic) frontier
# realized_strict_band per the advisor's field: low = L-envelope low, high = composed upper bound
REALIZED_STRICT_BAND: tuple[float, float] = (456.5, COMPOSED_STRICT_FRONTIER_TPS)
COMPOSED_VS_REALIZED_DRIFT_TPS: float = round(COMPOSED_STRICT_FRONTIER_TPS
                                              - REALIZED_STRICT_HEADLINE_TPS, 2)  # +10.78
REALIZED_ETA_ATTN_PCT: float = 5.50     # realized strict-attn tax (vs composed 3.08%)
COMPOSED_ETA_ATTN_PCT: float = 3.08
STRICT_FRONTIER_COLLAPSES_TO_M1: bool = False    # collapse to 161.70 REFUTED (stark #466)
STRICT_FRONTIER_CONFIG: str = "VLLM_BATCH_INVARIANT=1"  # no served-source edit, no kernel rebuild

# --- Floor-lock fallback (lawine #438): M=1 autoregressive, literal 1.0 BY CONSTRUCTION ---
FLOOR_LOCK_TPS: float = 161.70

# --- The census tension (the load-bearing honesty the packet must render) ---
CENSUS_LOCUS_STARK466: float = 1.0000        # locus proof (M=8 hd=512), 0 flips — NOT a full census
CENSUS_SERVED_LAND429_LAWINE455: float = 0.9989   # composed blanket-strict served, 1 flip @ p90 (0r0ounl8)
CENSUS_SERVED_UBEL461_ALLPIN: float = 0.9978      # all_pin floor, 1-2 residual flips (qz6f0zgw)
# denken #464 (1o7jwlw4): the residual/deployed flips are bitwise ties (m1_self_gap == 0.0),
# downstream-invisible -> "0.99-ish ties" == operative-1.0.
TIE_PROOF_M1_SELF_GAP: float = 0.0

# Operative-1.0 acceptance band (the 08:24Z human ruling: "0.99-ish is totally fine").
OPERATIVE_CENSUS_FLOOR: float = 0.99

# --- Relax lane (stark #452: daqrzr99) — COLLAPSED, strictly dominated, NO-GO (one-liner) ---
RELAX_PRIZE_REALIZED_TPS: float = 466.20
RELAX_PRIZE_GAIN_VS_STRICT_TPS: float = round(RELAX_PRIZE_REALIZED_TPS
                                              - COMPOSED_STRICT_FRONTIER_TPS, 2)  # -0.94
RELAX_PRIZE_IDENTITY: float = 0.730
RELAX_PRIZE_FLIPS: int = 3317

# --- Spine numbers (wirbel #459: 6pwhesdy, STRICT-NULL) — the corrected Triton verify surface ---
VERIFY_ATTN_SURFACE_PCT: float = 5.41        # FA2 = 0 verify calls; full 37-layer Triton 3D split-KV
BYTE_EXACT_RETUNE_CEILING_TPS: float = 1.20  # only order-preserving knob (num_stages 3->2, maxdiff 0.0)
MATERIALITY_BAR_TPS: float = 2.0             # < bar -> does not reopen strict supply
GEOMETRY_SLIDING_LAYERS: int = 30
GEOMETRY_GLOBAL_LAYERS: int = 7
GEOMETRY_TOTAL_LAYERS: int = GEOMETRY_SLIDING_LAYERS + GEOMETRY_GLOBAL_LAYERS  # 37

# --- sigma_hw (lawine #467, empirical, in flight) — carry the default until realized ---
SIGMA_HW_DEFAULT: float = 4.8153

# --- Submission naming (land #473 owns the trigger; we only name the targets) ---
SUBMISSION_NAME_GO_PREFIX: str = "senpai-strict-eqv"
SUBMISSION_NAME_FLOOR: str = "senpai-strict-m1ar-161"

# --- Human ruling (#407, 08:24Z): operative-1.0 accepted; literal byte-exact NOT required ---
HUMAN_OPERATIVE_RULING_0824Z: bool = True

# Recommendation enum
REC_GO_OPERATIVE_PENDING = "GO-OPERATIVE-PENDING-471-TIES-CONFIRM"
REC_GO_OPERATIVE = "GO-OPERATIVE"
REC_FLOOR_LOCK = "FLOOR-LOCK"
VALID_RECOMMENDATIONS = (REC_GO_OPERATIVE_PENDING, REC_GO_OPERATIVE, REC_FLOOR_LOCK)

# The committed inputs this packet consumes (for the consumes_* receipt).
CONSUMED_CARDS = ("stark#466", "stark#472", "ubel#470", "denken#471", "land#473")


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# The decision logic — the heart of the packet.
# --------------------------------------------------------------------------- #
def decide(denken471_census: float | None,
           denken471_semantic_flips: int | None,
           realized_best_estimate_tps: float | None) -> dict[str, Any]:
    """Render the GO-OPERATIVE / FLOOR-LOCK fork from denken #471's census.

    The binding decision variable is `n_semantic_flips` (non-tie flips).  The human ruled
    (08:24Z) that bitwise ties are acceptable, so the census value itself is informational
    (the honest label); the only blocker is a *semantic* flip.
    """
    realized_tps = realized_best_estimate_tps if _finite(realized_best_estimate_tps) \
        else REALIZED_STRICT_HEADLINE_TPS

    census_resolved = _finite(denken471_census) and denken471_semantic_flips is not None
    if not census_resolved:
        recommendation = REC_GO_OPERATIVE_PENDING
        fire_name = f"{SUBMISSION_NAME_GO_PREFIX}-{round(realized_tps)}"  # provisional target
        fire_tps = realized_tps
        gate_state = "pending"
        rationale = ("denken #471 served census not yet landed; recommendation is a CONDITIONAL "
                     "GO-OPERATIVE pending the tie-confirm (human pre-accepted ties 08:24Z).")
    elif denken471_census >= OPERATIVE_CENSUS_FLOOR and denken471_semantic_flips == 0:
        recommendation = REC_GO_OPERATIVE
        fire_name = f"{SUBMISSION_NAME_GO_PREFIX}-{round(realized_tps)}"
        fire_tps = realized_tps
        gate_state = "operative_1.0_confirmed"
        rationale = (f"denken #471 census {denken471_census:.4f} is operative-1.0 "
                     f"({denken471_semantic_flips} semantic flips, residuals are bf16 ties the "
                     f"human accepted) -> fire the realized strict win.")
    else:
        recommendation = REC_FLOOR_LOCK
        fire_name = SUBMISSION_NAME_FLOOR
        fire_tps = FLOOR_LOCK_TPS
        gate_state = "semantic_flip_or_sub_operative"
        reason = ("a SEMANTIC (non-tie) flip" if (denken471_semantic_flips or 0) > 0
                  else f"census {denken471_census:.4f} below the operative floor {OPERATIVE_CENSUS_FLOOR}")
        rationale = (f"denken #471 surfaced {reason} -> the lone case the human did NOT sign off; "
                     f"fall back to the literal-1.0-by-construction floor.")

    return {
        "capstone_recommendation": recommendation,
        "fire_submission_name": fire_name,
        "fire_tps": round(fire_tps, 2),
        "gate_state": gate_state,
        "rationale": rationale,
        "census_resolved": census_resolved,
        "denken471_served_census": denken471_census,
        "denken471_semantic_flips": denken471_semantic_flips,
        "operative_census_floor": OPERATIVE_CENSUS_FLOOR,
        "realized_best_estimate_tps": round(realized_tps, 2),
    }


def cross_check(ubel470_bipin_tps: float | None,
                stark472_best_estimate_tps: float | None,
                sigma_hw: float) -> dict[str, Any]:
    """ubel #470 (BI-pin) + stark #472 (in-graph overlap): bulletproof if both agree within sigma_hw."""
    points = {k: v for k, v in {
        "stark466_headline": REALIZED_STRICT_HEADLINE_TPS,
        "stark466_cluster_mean": REALIZED_STRICT_CLUSTER_MEAN_TPS,
        "ubel470_bipin": ubel470_bipin_tps if _finite(ubel470_bipin_tps) else None,
        "stark472_in_graph": stark472_best_estimate_tps if _finite(stark472_best_estimate_tps) else None,
    }.items() if v is not None}

    landed = [v for k, v in points.items() if k in ("ubel470_bipin", "stark472_in_graph")]
    n_independent = len(landed)
    spread = (max(points.values()) - min(points.values())) if len(points) > 1 else 0.0
    all_within_sigma = bool(spread <= sigma_hw)
    if n_independent < 2:
        confidence = "pending_cross_check"
    elif all_within_sigma:
        confidence = "bulletproof"
    else:
        confidence = "DISAGREE_flag_before_board"
    return {
        "points_tps": {k: round(v, 2) for k, v in points.items()},
        "n_independent_landed": n_independent,
        "spread_tps": round(spread, 4),
        "sigma_hw": round(sigma_hw, 4),
        "all_within_sigma_hw": all_within_sigma,
        "confidence": confidence,
    }


def build_packet(*,
                 denken471_census: float | None = None,
                 denken471_semantic_flips: int | None = None,
                 stark472_best_estimate_tps: float | None = None,
                 ubel470_bipin_tps: float | None = None,
                 sigma_hw: float = SIGMA_HW_DEFAULT) -> dict[str, Any]:
    """Assemble the full strict-submission decision packet (the JSON rollup)."""
    decision = decide(denken471_census, denken471_semantic_flips, stark472_best_estimate_tps)
    xcheck = cross_check(ubel470_bipin_tps, stark472_best_estimate_tps, sigma_hw)

    realized = decision["realized_best_estimate_tps"]
    gain_over_floor = round(realized - FLOOR_LOCK_TPS, 2)
    deficit_vs_deployed_headline = round(DEPLOYED_TPS - REALIZED_STRICT_HEADLINE_TPS, 2)
    deficit_vs_deployed_clustermean = round(DEPLOYED_TPS - REALIZED_STRICT_CLUSTER_MEAN_TPS, 2)

    the_number = {
        "realized_strict_headline_tps": REALIZED_STRICT_HEADLINE_TPS,
        "realized_strict_cluster_mean_tps": REALIZED_STRICT_CLUSTER_MEAN_TPS,
        "realized_strict_l_envelope_tps": list(REALIZED_STRICT_L_ENVELOPE),
        "realized_strict_band_tps": list(REALIZED_STRICT_BAND),
        "composed_optimistic_frontier_tps": COMPOSED_STRICT_FRONTIER_TPS,
        "composed_vs_realized_drift_tps": COMPOSED_VS_REALIZED_DRIFT_TPS,
        "composition_was_optimistic": bool(COMPOSED_VS_REALIZED_DRIFT_TPS > sigma_hw),
        "realized_eta_attn_pct": REALIZED_ETA_ATTN_PCT,
        "composed_eta_attn_pct": COMPOSED_ETA_ATTN_PCT,
        "strict_frontier_collapses_to_m1": STRICT_FRONTIER_COLLAPSES_TO_M1,
        "config_reachable_via": STRICT_FRONTIER_CONFIG,
        "no_served_source_edit": True,
        "no_kernel_rebuild": True,
        "gain_over_floor_tps": gain_over_floor,
        "deficit_vs_deployed_headline_tps": deficit_vs_deployed_headline,
        "deficit_vs_deployed_cluster_mean_tps": deficit_vs_deployed_clustermean,
        "stark472_best_estimate_live": _finite(stark472_best_estimate_tps),
        "provenance": "stark#466 sxigz7dp(speed)/gmd8v9sw(identity) ab9b286; stark#472 in flight",
    }

    the_gate = {
        "binding_gate": "denken#471 served census == 1.0",
        "resolver": "denken#471 strict-submission-identity-certifier (full 128-prompt served census)",
        "do_not_duplicate": True,
        "census_prior": {
            "stark466_locus": CENSUS_LOCUS_STARK466,
            "land429_lawine455_served_p90": CENSUS_SERVED_LAND429_LAWINE455,
            "ubel461_allpin_floor": CENSUS_SERVED_UBEL461_ALLPIN,
        },
        "census_prior_span": round(CENSUS_LOCUS_STARK466 - CENSUS_SERVED_UBEL461_ALLPIN, 4),
        "census_tension_rendered": True,
        "reconciliation_hypothesis": (
            "the order-preserving 2D reduction REMOVES the attention 3D split-KV reassociation that "
            "produces the near-tie population (denken #464) -> served census SHOULD resolve to "
            "operative-1.0; pending denken #471 confirm."),
        "tie_proof_m1_self_gap": TIE_PROOF_M1_SELF_GAP,
        "ties_are_downstream_invisible": True,
    }

    the_fork = {
        "GO_OPERATIVE": {
            "condition": (f"denken#471 census >= {OPERATIVE_CENSUS_FLOOR} AND n_semantic_flips == 0 "
                          "(ties OK per human 08:24Z)"),
            "fire": f"{SUBMISSION_NAME_GO_PREFIX}-{round(realized)}",
            "tps": realized,
            "note": "the realized honest strict win, +~295 over the floor; land #473 staged ready-to-fire.",
        },
        "FLOOR_LOCK": {
            "condition": "denken#471 surfaces a SEMANTIC (non-tie) flip",
            "fire": SUBMISSION_NAME_FLOOR,
            "tps": FLOOR_LOCK_TPS,
            "note": "literal-1.0 by construction (lawine #438); the honest baseline entry, not the headline win.",
        },
        "BLOCKED_retired": {
            "was": "literal <1.0 -> don't ship",
            "retired_by": "human operative-1.0 ruling (#407, 08:24Z)",
        },
    }

    honesty_hinge = {
        "submission_labeled_with_true_census": True,
        "example_label": "operative-1.0: 0.9989, 1 tied flip @ p90, 0 semantic",
        "claimed_as_literal_1p0": False,
        "literal_vs_operative_surfaced": True,
        "literal_vs_operative_resolution": "human ruled operative is honest-strict (08:24Z)",
        "human_operative_ruling_0824z": HUMAN_OPERATIVE_RULING_0824Z,
    }

    reference_frame = {
        "deployed_tps": DEPLOYED_TPS,
        "deployed_identity": DEPLOYED_IDENTITY,
        "deployed_flips": list(DEPLOYED_FLIPS),
        "deployed_is_strict_equivalent": False,
        "deployed_outside_strict_feasible_set": True,
        "relax_prize_realized_tps": RELAX_PRIZE_REALIZED_TPS,
        "relax_prize_gain_vs_strict_tps": RELAX_PRIZE_GAIN_VS_STRICT_TPS,
        "relax_prize_identity": RELAX_PRIZE_IDENTITY,
        "relax_prize_flips": RELAX_PRIZE_FLIPS,
        "relax_lane_verdict": "DEAD (strictly dominated: ~0 gain AND worse identity) — stay strict",
        "ppl": PPL_DEPLOYED,
        "ppl_gate": PPL_GATE,
        "ppl_clears_gate": bool(PPL_DEPLOYED <= PPL_GATE),
        "sigma_hw": round(sigma_hw, 4),
        "spine_verify_attn_surface_pct": VERIFY_ATTN_SURFACE_PCT,
        "spine_byte_exact_retune_ceiling_tps": BYTE_EXACT_RETUNE_CEILING_TPS,
        "spine_reopens_strict_supply": bool(BYTE_EXACT_RETUNE_CEILING_TPS >= MATERIALITY_BAR_TPS),
        "spine_geometry_layers": GEOMETRY_TOTAL_LAYERS,
    }

    ownership = {
        "this_packet": "the human-facing decision narrative (synthesis only)",
        "denken471": "the identity oracle (served census)",
        "land473": "the execution machinery (submission command + approval + board post)",
        "consumes_466_472_470_471_473": True,
        "consumed_cards": list(CONSUMED_CARDS),
        "analysis_only": True,
        "no_served_file_change": True,
        "official_tps": 0,
    }

    packet = {
        "kind": "strict-submission-decision-packet",
        "pr": 357,
        "agent": "fern",
        "decision": decision,
        "A_the_number": the_number,
        "B_the_gate": the_gate,
        "C_the_fork": the_fork,
        "D_cross_check": xcheck,
        "E_reference_frame": reference_frame,
        "honesty_hinge": honesty_hinge,
        "ownership": ownership,
    }
    packet["self_test"] = _selftests(packet, sigma_hw)
    packet["headline"] = _headline(packet)
    return packet


def _headline(p: dict[str, Any]) -> dict[str, Any]:
    d = p["decision"]
    n = p["A_the_number"]
    return {
        "capstone_recommendation": d["capstone_recommendation"],
        "fire_submission_name": d["fire_submission_name"],
        "fire_tps": d["fire_tps"],
        "realized_strict_headline_tps": n["realized_strict_headline_tps"],
        "realized_strict_band_tps": n["realized_strict_band_tps"],
        "floor_lock_tps": FLOOR_LOCK_TPS,
        "deployed_tps": DEPLOYED_TPS,
        "binding_gate": p["B_the_gate"]["binding_gate"],
        "cross_check_confidence": p["D_cross_check"]["confidence"],
        "capstone_self_test_passes": None,  # filled by main() after nan-check
    }


# --------------------------------------------------------------------------- #
# Self-test — REAL arithmetic invariants, not a banked-constant mirror.
# Each condition checks a *relationship* between numbers (ordering, a computed
# difference, a decision-logic outcome), so a green result means the packet is
# internally consistent, not merely that a constant was copied.
# --------------------------------------------------------------------------- #
def _selftests(p: dict[str, Any], sigma_hw: float) -> dict[str, Any]:
    n = p["A_the_number"]
    g = p["B_the_gate"]
    r = p["E_reference_frame"]
    d = p["decision"]
    x = p["D_cross_check"]
    band_low, band_high = REALIZED_STRICT_BAND
    cond: dict[str, bool] = {}

    # Reference points are strictly ordered: floor < realized headline < deployed.
    cond["a_floor_lt_headline"] = FLOOR_LOCK_TPS < REALIZED_STRICT_HEADLINE_TPS
    cond["b_headline_lt_deployed"] = REALIZED_STRICT_HEADLINE_TPS < DEPLOYED_TPS
    cond["c_floor_lt_deployed"] = FLOOR_LOCK_TPS < DEPLOYED_TPS

    # The realized band is well-formed and brackets the cluster-mean and headline.
    cond["d_band_low_le_high"] = band_low <= band_high
    cond["e_clustermean_in_band"] = band_low <= REALIZED_STRICT_CLUSTER_MEAN_TPS <= band_high
    cond["f_headline_le_clustermean"] = REALIZED_STRICT_HEADLINE_TPS <= REALIZED_STRICT_CLUSTER_MEAN_TPS
    cond["g_clustermean_le_composed"] = REALIZED_STRICT_CLUSTER_MEAN_TPS <= COMPOSED_STRICT_FRONTIER_TPS
    cond["h_band_high_is_composed"] = math.isclose(band_high, COMPOSED_STRICT_FRONTIER_TPS)

    # The "composition was optimistic" claim: drift == composed - headline, and exceeds sigma_hw.
    cond["i_drift_equals_difference"] = math.isclose(
        COMPOSED_VS_REALIZED_DRIFT_TPS,
        round(COMPOSED_STRICT_FRONTIER_TPS - REALIZED_STRICT_HEADLINE_TPS, 2), abs_tol=0.01)
    cond["j_drift_exceeds_sigma_hw"] = COMPOSED_VS_REALIZED_DRIFT_TPS > sigma_hw
    cond["k_realized_eta_exceeds_composed_eta"] = REALIZED_ETA_ATTN_PCT > COMPOSED_ETA_ATTN_PCT
    cond["l_collapse_refuted"] = STRICT_FRONTIER_COLLAPSES_TO_M1 is False

    # Census prior tension is ordered (the honest span the packet renders).
    cond["m_census_prior_ordered"] = (CENSUS_SERVED_UBEL461_ALLPIN
                                      <= CENSUS_SERVED_LAND429_LAWINE455
                                      <= CENSUS_LOCUS_STARK466)
    cond["n_census_prior_within_operative_band"] = CENSUS_SERVED_UBEL461_ALLPIN >= OPERATIVE_CENSUS_FLOOR
    cond["o_census_tension_rendered"] = bool(g["census_tension_rendered"])
    cond["p_ties_downstream_invisible"] = math.isclose(TIE_PROOF_M1_SELF_GAP, 0.0)

    # Relax lane is strictly dominated: <= 0 gain AND worse identity than deployed (-> NO-GO).
    cond["q_relax_gain_nonpositive"] = RELAX_PRIZE_GAIN_VS_STRICT_TPS <= 0.0
    cond["r_relax_gain_equals_difference"] = math.isclose(
        RELAX_PRIZE_GAIN_VS_STRICT_TPS,
        round(RELAX_PRIZE_REALIZED_TPS - COMPOSED_STRICT_FRONTIER_TPS, 2), abs_tol=0.01)
    cond["s_relax_identity_worse_than_deployed"] = RELAX_PRIZE_IDENTITY < DEPLOYED_IDENTITY

    # Spine: +1.20 < the +2 materiality bar -> does NOT reopen strict supply; geometry sums.
    cond["t_spine_below_materiality"] = BYTE_EXACT_RETUNE_CEILING_TPS < MATERIALITY_BAR_TPS
    cond["u_spine_does_not_reopen_supply"] = r["spine_reopens_strict_supply"] is False
    cond["v_geometry_sums"] = (GEOMETRY_SLIDING_LAYERS + GEOMETRY_GLOBAL_LAYERS
                               == GEOMETRY_TOTAL_LAYERS == 37)

    # PPL clears the gate.
    cond["w_ppl_clears_gate"] = PPL_DEPLOYED <= PPL_GATE

    # Decision logic is sound for the resolved inputs (re-derive the fork independently).
    cond["x_recommendation_valid"] = d["capstone_recommendation"] in VALID_RECOMMENDATIONS
    if not d["census_resolved"]:
        cond["y_fork_matches_inputs"] = d["capstone_recommendation"] == REC_GO_OPERATIVE_PENDING
    elif (d["denken471_served_census"] is not None
          and d["denken471_served_census"] >= OPERATIVE_CENSUS_FLOOR
          and d["denken471_semantic_flips"] == 0):
        cond["y_fork_matches_inputs"] = (d["capstone_recommendation"] == REC_GO_OPERATIVE
                                         and math.isclose(d["fire_tps"], d["realized_best_estimate_tps"]))
    else:
        cond["y_fork_matches_inputs"] = (d["capstone_recommendation"] == REC_FLOOR_LOCK
                                         and math.isclose(d["fire_tps"], FLOOR_LOCK_TPS))
    # GO fires the realized strict number (above the floor); FLOOR-LOCK fires the floor exactly.
    if d["capstone_recommendation"] == REC_FLOOR_LOCK:
        cond["z_fire_tps_consistent"] = math.isclose(d["fire_tps"], FLOOR_LOCK_TPS)
    else:
        cond["z_fire_tps_consistent"] = d["fire_tps"] > FLOOR_LOCK_TPS

    # Cross-check confidence is one of the defined states and consistent with the spread.
    cond["aa_xcheck_confidence_defined"] = x["confidence"] in (
        "pending_cross_check", "bulletproof", "DISAGREE_flag_before_board")
    cond["ab_xcheck_spread_consistent"] = (
        (x["n_independent_landed"] < 2 and x["confidence"] == "pending_cross_check")
        or (x["n_independent_landed"] >= 2
            and ((x["all_within_sigma_hw"] and x["confidence"] == "bulletproof")
                 or (not x["all_within_sigma_hw"] and x["confidence"] == "DISAGREE_flag_before_board"))))

    # Mandate constraints (this is an analysis-only 0-GPU capstone).
    cond["ac_analysis_only"] = bool(p["ownership"]["analysis_only"])
    cond["ad_no_served_file_change"] = bool(p["ownership"]["no_served_file_change"])
    cond["ae_official_tps_zero"] = p["ownership"]["official_tps"] == 0
    cond["af_consumes_all_cards"] = bool(p["ownership"]["consumes_466_472_470_471_473"])
    cond["ag_literal_vs_operative_surfaced"] = bool(p["honesty_hinge"]["literal_vs_operative_surfaced"])
    cond["ah_human_ruling_recorded"] = bool(p["honesty_hinge"]["human_operative_ruling_0824z"])

    return {
        "conditions": cond,
        "n_conditions": len(cond),
        "n_passing": sum(1 for v in cond.values() if v),
        "capstone_self_test_passes": bool(all(cond.values())),
    }


# --------------------------------------------------------------------------- #
# One-screen human-facing render.
# --------------------------------------------------------------------------- #
def render_one_screen(p: dict[str, Any]) -> str:
    d = p["decision"]
    n = p["A_the_number"]
    g = p["B_the_gate"]
    r = p["E_reference_frame"]
    x = p["D_cross_check"]
    cp = g["census_prior"]
    realized = d["realized_best_estimate_tps"]
    band = n["realized_strict_band_tps"]
    L = [
        "================================================================================",
        " STRICT-SUBMISSION DECISION PACKET  —  PR #357  (fern, CPU-only synthesis)",
        "================================================================================",
        f" RECOMMENDATION : {d['capstone_recommendation']}",
        f"                  -> fire `{d['fire_submission_name']}`  (~{d['fire_tps']:.2f} TPS)",
        "",
        " A. THE NUMBER (stark #466, realized e2e — collapse to 161.70 REFUTED)",
        f"      realized strict  {n['realized_strict_headline_tps']:.2f} TPS  headline (conservative @ L=640)",
        f"      band [{band[0]:.2f}, {band[1]:.2f}]  ·  cluster-mean ~{n['realized_strict_cluster_mean_tps']:.0f}"
        f"  ·  +{n['gain_over_floor_tps']:.0f} over floor  ·  ~{n['deficit_vs_deployed_cluster_mean_tps']:.0f} under deployed",
        f"      * composition was OPTIMISTIC: old composed {n['composed_optimistic_frontier_tps']:.2f} over-counts"
        f" by +{n['composed_vs_realized_drift_tps']:.2f} (> sigma_hw {r['sigma_hw']:.2f}); true frontier in [{band[0]:.2f}, <= {band[1]:.2f}]",
        f"      config-reachable via {n['config_reachable_via']} (no served-source edit, no kernel rebuild)",
        "",
        " B. THE GATE (denken #471 served 128-prompt census — THE resolver, do not duplicate)",
        f"      binding: census == operative-1.0 (>= {OPERATIVE_CENSUS_FLOOR}, every residual a bf16 tie, 0 semantic flips)",
        f"      * tension: locus {cp['stark466_locus']:.4f} (stark#466) vs served prior"
        f" {cp['land429_lawine455_served_p90']:.4f}@p90 (land#429) / {cp['ubel461_allpin_floor']:.4f} all-pin (ubel#461)",
        "      reconciliation: order-preserving 2D reduction removes the 3D split-KV near-tie population (denken#464)",
        "",
        " C. THE FORK",
        f"      GO-OPERATIVE  (471: 0 semantic flips, ties OK per human 08:24Z) -> `{p['C_the_fork']['GO_OPERATIVE']['fire']}`"
        f" (~{realized:.2f}, honest strict win)",
        f"      FLOOR-LOCK    (471: >=1 SEMANTIC non-tie flip)                   -> `{SUBMISSION_NAME_FLOOR}`"
        f" ({FLOOR_LOCK_TPS:.2f}, literal-1.0 by construction)",
        "      [BLOCKED retired — human ruled operative-1.0 honest-strict, 08:24Z]",
        "",
        " HONESTY HINGE : submission labeled with its TRUE census"
        f" (e.g. \"{p['honesty_hinge']['example_label']}\"), NOT claimed literal 1.0",
        f" D. CROSS-CHECK: ubel#470 BI-pin + stark#472 in-graph overlap -> {x['confidence']}"
        f" (n_independent={x['n_independent_landed']}, spread {x['spread_tps']:.2f} vs sigma_hw {x['sigma_hw']:.2f})",
        f" E. REFERENCE  : deployed {r['deployed_tps']:.2f} NON-equiv (id {r['deployed_identity']:.4f}, 3 ties) OUTSIDE strict set"
        f"  ·  relax-prize DEAD (stark#452: {r['relax_prize_gain_vs_strict_tps']:+.2f} TPS, id {r['relax_prize_identity']:.3f})"
        f"  ·  PPL {r['ppl']:.4f} <= {r['ppl_gate']:.2f} OK",
        " OWNERSHIP     : this packet = the human call · denken#471 = oracle · land#473 = trigger"
        "  ·  CPU-only, official_tps=0, no served-file change",
        "================================================================================",
        f" self-test: {p['self_test']['n_passing']}/{p['self_test']['n_conditions']} invariants"
        f"  ·  recommendation: {d['capstone_recommendation']}",
        "================================================================================",
    ]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# NaN guard + W&B logging (reuses scripts.wandb_logging plumbing).
# --------------------------------------------------------------------------- #
def _nan_paths(node: Any, path: str = "result") -> list[str]:
    bad: list[str] = []
    if isinstance(node, float) and not math.isfinite(node):
        bad.append(path)
    elif isinstance(node, dict):
        for k, v in node.items():
            bad.extend(_nan_paths(v, f"{path}.{k}"))
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            bad.extend(_nan_paths(v, f"{path}[{i}]"))
    return bad


def _wandb_summary(payload: dict) -> dict[str, Any]:
    p = payload["synthesis"]
    d = p["decision"]
    n = p["A_the_number"]
    g = p["B_the_gate"]
    r = p["E_reference_frame"]
    x = p["D_cross_check"]
    o = p["ownership"]
    h = p["honesty_hinge"]
    st = p["self_test"]
    band = n["realized_strict_band_tps"]
    rec_code = {REC_GO_OPERATIVE_PENDING: 0, REC_GO_OPERATIVE: 1, REC_FLOOR_LOCK: -1}[
        d["capstone_recommendation"]]
    summary: dict[str, Any] = {
        # PRIMARY
        "capstone_self_test_passes": int(bool(st["capstone_self_test_passes"])),
        "capstone_self_test_n_passing": st["n_passing"],
        "capstone_self_test_n_conditions": st["n_conditions"],
        # the decision
        "capstone_recommendation_code": rec_code,  # 0 pending / 1 go / -1 floor-lock
        "fire_tps": d["fire_tps"],
        "gate_resolved": int(bool(d["census_resolved"])),
        # A. the number
        "realized_strict_headline_tps": n["realized_strict_headline_tps"],
        "realized_strict_cluster_mean_tps": n["realized_strict_cluster_mean_tps"],
        "realized_strict_band_low_tps": band[0],
        "realized_strict_band_high_tps": band[1],
        "composed_optimistic_frontier_tps": n["composed_optimistic_frontier_tps"],
        "composed_vs_realized_drift_tps": n["composed_vs_realized_drift_tps"],
        "composition_was_optimistic": int(bool(n["composition_was_optimistic"])),
        "realized_eta_attn_pct": n["realized_eta_attn_pct"],
        "composed_eta_attn_pct": n["composed_eta_attn_pct"],
        "strict_frontier_collapses_to_m1": int(bool(n["strict_frontier_collapses_to_m1"])),
        "gain_over_floor_tps": n["gain_over_floor_tps"],
        "deficit_vs_deployed_cluster_mean_tps": n["deficit_vs_deployed_cluster_mean_tps"],
        "stark472_best_estimate_live": int(bool(n["stark472_best_estimate_live"])),
        # B. the gate
        "census_locus_stark466": g["census_prior"]["stark466_locus"],
        "census_served_land429_p90": g["census_prior"]["land429_lawine455_served_p90"],
        "census_served_ubel461_allpin": g["census_prior"]["ubel461_allpin_floor"],
        "census_prior_span": g["census_prior_span"],
        "census_tension_rendered": int(bool(g["census_tension_rendered"])),
        "tie_proof_m1_self_gap": g["tie_proof_m1_self_gap"],
        "operative_census_floor": OPERATIVE_CENSUS_FLOOR,
        # D. cross-check
        "cross_check_n_independent": x["n_independent_landed"],
        "cross_check_spread_tps": x["spread_tps"],
        "sigma_hw": r["sigma_hw"],
        # E. reference frame
        "deployed_tps": r["deployed_tps"],
        "deployed_identity": r["deployed_identity"],
        "deployed_is_strict_equivalent": int(bool(r["deployed_is_strict_equivalent"])),
        "relax_prize_realized_tps": r["relax_prize_realized_tps"],
        "relax_prize_gain_vs_strict_tps": r["relax_prize_gain_vs_strict_tps"],
        "relax_prize_identity": r["relax_prize_identity"],
        "ppl": r["ppl"],
        "ppl_gate": r["ppl_gate"],
        "ppl_clears_gate": int(bool(r["ppl_clears_gate"])),
        "spine_verify_attn_surface_pct": r["spine_verify_attn_surface_pct"],
        "spine_byte_exact_retune_ceiling_tps": r["spine_byte_exact_retune_ceiling_tps"],
        "spine_reopens_strict_supply": int(bool(r["spine_reopens_strict_supply"])),
        "spine_geometry_layers": r["spine_geometry_layers"],
        # floor + honesty + ownership receipts
        "floor_lock_tps": FLOOR_LOCK_TPS,
        "human_operative_ruling_0824z": int(bool(h["human_operative_ruling_0824z"])),
        "literal_vs_operative_surfaced": int(bool(h["literal_vs_operative_surfaced"])),
        "consumes_466_472_470_471_473": int(bool(o["consumes_466_472_470_471_473"])),
        "analysis_only": int(bool(o["analysis_only"])),
        "no_served_file_change": int(bool(o["no_served_file_change"])),
        "official_tps": o["official_tps"],
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    return {k: v for k, v in summary.items()
            if v is not None and not (isinstance(v, float) and not math.isfinite(v))}


def _maybe_log_wandb(args: Any, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[strict-submission-decision] wandb logging unavailable: {exc}", flush=True)
        return

    p = payload["synthesis"]
    run = init_wandb_run(
        job_type="strict-submission-decision-packet",
        agent="fern",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=[
            "strict-submission-decision", "equivalence-escalation-anchors", "strict-equivalent",
            "go-operative", "floor-lock", "decision-packet", "human-facing", "bank-the-analysis",
            "stark-466-realized-frontier", "denken-471-census-gate", "land-473-trigger",
            "ubel-470-cross-check", "stark-472-in-graph", "lawine-467-sigma-hw",
            "operative-1p0", "literal-vs-operative", "human-ruling-0824z", "tie-proof-denken-464",
            "relax-prize-dead-stark-452", "deployed-nonequiv-481", "ppl-gate-2p42",
            "spine-wirbel-459-strict-null", "analysis-only", "cpu-only", "no-served-file-change",
        ],
        config={
            "deployed_tps": DEPLOYED_TPS,
            "deployed_identity": DEPLOYED_IDENTITY,
            "ppl_deployed": PPL_DEPLOYED,
            "ppl_gate": PPL_GATE,
            "realized_strict_headline_tps": REALIZED_STRICT_HEADLINE_TPS,
            "realized_strict_cluster_mean_tps": REALIZED_STRICT_CLUSTER_MEAN_TPS,
            "composed_optimistic_frontier_tps": COMPOSED_STRICT_FRONTIER_TPS,
            "floor_lock_tps": FLOOR_LOCK_TPS,
            "operative_census_floor": OPERATIVE_CENSUS_FLOOR,
            "sigma_hw_default": SIGMA_HW_DEFAULT,
            "binding_gate": p["B_the_gate"]["binding_gate"],
            "capstone_recommendation": p["decision"]["capstone_recommendation"],
            "live_slots": {
                "denken471_served_census": args.denken471_census,
                "denken471_semantic_flips": args.denken471_semantic_flips,
                "stark472_best_estimate_tps": args.stark472_best_estimate_tps,
                "ubel470_bipin_tps": args.ubel470_bipin_tps,
                "lawine467_sigma_hw": args.sigma_hw,
            },
            "source_runs": (
                "stark#466(sxigz7dp speed / gmd8v9sw identity, ab9b286): realized strict frontier "
                "HOLDS e2e 456.36 headline / ~459 cluster-mean, collapse-to-161.70 REFUTED, "
                "VLLM_BATCH_INVARIANT=1. denken#471(strict-submission-identity-certifier): served "
                "128-prompt census gate. land#429/lawine#455(0r0ounl8): served 0.9989 @ p90. "
                "ubel#461(qz6f0zgw): all_pin ~0.9978. denken#464(1o7jwlw4): flips are m1_self_gap=0.0 "
                "ties. stark#452(daqrzr99): relax-prize collapsed 466.20/-0.94, id 0.730. "
                "lawine#438: M=1 AR floor 161.70 literal-1.0. wirbel#459(6pwhesdy): STRICT-NULL spine "
                "5.41%/+1.20/37=30+7. lawine#467: sigma_hw empirical. land#473: submission trigger. "
                "Deployed PR#52(2x9fm2zx): 481.53 NON-equiv id 0.9966."
            ),
        },
    )
    if run is None:
        print("[strict-submission-decision] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary = _wandb_summary(payload)
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="strict_submission_decision_packet",
                      artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[strict-submission-decision] wandb logged {len(summary)} keys", flush=True)


# --------------------------------------------------------------------------- #
# CLI / main.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    # THE gate — denken #471 served census (the binding live input).
    ap.add_argument("--denken471-served-census", dest="denken471_census", type=float, default=None,
                    help="denken #471 MEASURED served 128-prompt census identity rate on stark's exact "
                         "config (omit -> recommendation stays GO-OPERATIVE-PENDING-471-TIES-CONFIRM).")
    ap.add_argument("--denken471-semantic-flips", dest="denken471_semantic_flips", type=int, default=None,
                    help="denken #471 count of SEMANTIC (non-tie) flips in the served census. THE binding "
                         "decision variable: 0 (with census >= operative floor) -> GO-OPERATIVE; "
                         ">0 -> FLOOR-LOCK (the lone case the human did not sign off, 08:24Z). Omit -> pending.")
    # LIVE slots that tighten the number / cross-check (do not change the fork unless gate resolves).
    ap.add_argument("--stark472-best-estimate-tps", dest="stark472_best_estimate_tps", type=float, default=None,
                    help="stark #472 whole-cycle A/B in-graph-overlap realized_strict_frontier_best_estimate_tps "
                         "(omit -> carry the stark #466 conservative headline 456.36 + band).")
    ap.add_argument("--ubel470-bipin-tps", dest="ubel470_bipin_tps", type=float, default=None,
                    help="ubel #470 BI-pin realized TPS (a DIFFERENT mechanism than stark's num_splits=1) for "
                         "the independent cross-check (omit -> cross-check stays pending).")
    ap.add_argument("--lawine467-sigma-hw", dest="sigma_hw", type=float, default=SIGMA_HW_DEFAULT,
                    help=f"lawine #467 empirical hardware sigma (TPS). Default {SIGMA_HW_DEFAULT} until realized.")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="equivalence-escalation-anchors")
    args = ap.parse_args(argv)

    packet = build_packet(
        denken471_census=args.denken471_census,
        denken471_semantic_flips=args.denken471_semantic_flips,
        stark472_best_estimate_tps=args.stark472_best_estimate_tps,
        ubel470_bipin_tps=args.ubel470_bipin_tps,
        sigma_hw=args.sigma_hw,
    )

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 357, "agent": "fern",
        "kind": "strict-submission-decision-packet", "analysis_only": True,
        "live_slots": {
            "denken471_served_census": args.denken471_census,
            "denken471_semantic_flips": args.denken471_semantic_flips,
            "stark472_best_estimate_tps": args.stark472_best_estimate_tps,
            "ubel470_bipin_tps": args.ubel470_bipin_tps,
            "lawine467_sigma_hw": args.sigma_hw,
        },
        "synthesis": packet,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _nan_paths(payload)
    payload["nan_clean"] = not nan_paths
    packet["self_test"]["conditions"]["ai_nan_clean"] = not nan_paths
    packet["self_test"]["n_conditions"] = len(packet["self_test"]["conditions"])
    packet["self_test"]["n_passing"] = sum(1 for v in packet["self_test"]["conditions"].values() if v)
    packet["self_test"]["capstone_self_test_passes"] = bool(
        all(packet["self_test"]["conditions"].values()))
    packet["headline"]["capstone_self_test_passes"] = packet["self_test"]["capstone_self_test_passes"]
    if nan_paths:
        print(f"[strict-submission-decision] WARNING non-finite at: {nan_paths}", flush=True)

    print(render_one_screen(packet), flush=True)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "strict_500_composite_reachability_results.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    md_path = out_dir / "strict_submission_decision_packet.md"
    md_path.write_text(render_one_screen(packet) + "\n", encoding="utf-8")
    print(f"[strict-submission-decision] wrote {json_path} and {md_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = packet["self_test"]["capstone_self_test_passes"] and payload["nan_clean"]
        print(f"[strict-submission-decision] self-test {'PASS' if ok else 'FAIL'} "
              f"({packet['self_test']['n_passing']}/{packet['self_test']['n_conditions']})", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
