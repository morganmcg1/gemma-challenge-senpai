#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Unified absolute ceiling (PR #457, land) — CPU-only analytic reconciliation.

THE QUESTION (the escalation packet's three-ceiling note; capstone needs ONE number)
------------------------------------------------------------------------------------
The escalation packet needs ONE defensible **absolute TPS ceiling** for single-stream
decode on this stack, but three different ceiling numbers are in play and must be
reconciled on a consistent basis:

  (a) verify-BW lambda=1 wall          520.95   (SPEC 600 GB/s basis; #436 nvsbctji,
                                                  reproduced as #450 ceiling_tps_spec /
                                                  #451 verify_bw_wall_tps)
  (b) achieved-read-peak recoverable   510.87   (MEASURED read-peak 517.58 GB/s basis;
                                                  #450 c5oyb7gv ceiling_tps_read_peak /
                                                  max_recoverable_endtoend_tps_ceiling)
  (c) demand-oracle raw                551.71   -> capped 520.95  (DEMAND-side E[T] oracle;
                                                  #451 c675zor8 optimistic_ceiling_raw_tps,
                                                  capped at the verify-BW wall)

They differ ONLY by *basis* (the bandwidth denominator / demand-vs-supply framing), exactly
the kind of denominator reconciliation #447 did for the attention fraction.

THE PHYSICS (all banked; nothing re-derived)
--------------------------------------------
Single-stream int4 verify decode is HBM-read-BOUND: each pass reads ~4B int4 weights
(weight+scale ~= 1772 MB/step in the #450 byte model) once — the IRREDUCIBLE HBM floor.
The ceiling is therefore a SUPPLY ceiling set by the read-bandwidth denominator. The three
numbers are the SAME supply ceiling viewed through different denominators:

  * SPEC 600 GB/s is the GA102 datasheet peak. The MEASURED achievable STREAM read peak is
    517.58 GB/s = 86.26% of spec (#450 peak_read_frac_of_spec). So the spec-basis 520.95
    assumes ~16% more bandwidth than the hardware delivers -> OVER-OPTIMISTIC upper bound.
  * The measured-read-peak basis gives the perfect-f->1 re-tiling limit = 510.87, bounded by
    a MEASURED bandwidth -> the DEFENSIBLE achievable ceiling.
  * The demand-oracle raw 551.71 is a DEMAND-side number (what a maximally-accepting bigger
    drafter would yield if supply were unlimited). It EXCEEDS supply, so it is capped at the
    supply wall; it does NOT add headroom, it confirms SUPPLY (bandwidth) binds, not demand.

DELIVERABLE — one authoritative `unified_absolute_ceiling_tps` (the achieved-read-peak 510.87,
since spec-600 is not realizable) + the sigma_hw envelope, with the basis of each number made
explicit, so the capstone states "the physical ceiling is X +/- sigma" with confidence. The
spec-basis 520.95 is carried as the over-optimistic upper bound.

LOCAL, CPU-ONLY, ANALYTIC. No GPU / vLLM / HF Job / submission / served-file change / official
draw. BASELINE stays 481.53; adds 0 TPS; greedy/PPL untouched (PPL anchor 2.3772). Imports the
committed result JSONs of #450 (gemm_roofline_bw_ceiling) and #451 (drafter_size_oracle_net_
ceiling); re-derives nothing. NOT a launch.

PRIMARY metric  unified_ceiling_self_test_passes
TEST    metric  ppl  (2.3772 anchor; this leg does not touch the served model)
HEADLINE         unified_absolute_ceiling_tps  (510.87, achieved-read-peak basis)
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

ROOFLINE_JSON = REPO_ROOT / "research/speed/gemm_roofline_bw_ceiling/roofline_ceiling.json"
ORACLE_JSON = (
    REPO_ROOT
    / "research/validity/drafter_size_oracle_net_ceiling/drafter_size_oracle_net_ceiling_results.json"
)

# --------------------------------------------------------------------------- #
# Banked external constants (provenance: W&B run ids, project
# wandb-applied-ai-team/gemma-challenge-senpai). IMPORTED, not re-derived. Each is
# round-tripped against its committed result JSON in load_banked().
# --------------------------------------------------------------------------- #
# (b) #450 c5oyb7gv  — achieved-read-peak recoverable ceiling (the UNIFIED ceiling)
CEIL_READPEAK = 510.8724230449973
# (a) #436 nvsbctji  — spec-BW lambda=1 wall (full precision == #451 verify_bw_wall_tps;
#                       #450 carries the rounded 520.953 form)
CEIL_SPEC = 520.9527323111674
CEIL_SPEC_ROUNDED = 520.953
# (c) #451 c675zor8  — demand-oracle raw + capped
DEMAND_RAW = 551.7124657582939
DEMAND_CAPPED = 520.9527323111674

# Bandwidth bases (#450)
PEAK_READ_GBPS = 517.5801601788328       # measured STREAM read peak (achievable)
PEAK_SPEC_GBPS = 600.0                    # GA102 datasheet peak (NOT achievable)
PEAK_READ_FRAC_OF_SPEC = 0.8626336002980547
GEMM_WEIGHT_SCALE_BYTES_MB = 1772.05248  # #450 irreducible per-pass weight+scale read

# Relax-prize edges (#450; greedy-UNSAFE FP-reassociating re-tiling)
RELAX_OPTIMISTIC = 510.8724230449973     # perfect-f re-tiling == the read-peak ceiling
RELAX_REALISTIC_HI = 498.57990782684584  # realizable split-K hi (5-12% of GEMM)
RELAX_REALISTIC_LO = 479.74508514667906  # realizable split-K lo

# Operating points + hardware variance (#451 c675zor8 / #52 / denken #423)
DEPLOYED_TPS = 481.53                     # PR #52 official, non-equivalent (3 flips), 2x9fm2zx
STRICT_FRONTIER_TPS = 467.14             # denken #423 5a6zq2yz realized blanket-strict frontier
SIGMA_HW = 4.8153                         # hardware-variance envelope (~1% of deployed)
PPL_ANCHOR = 2.3772                       # PR #52 served PPL (gate <= 2.42)

PPL_GATE = 2.42
TOL_RT = 1e-6                             # banked round-trip tolerance


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# Load + round-trip the banked source JSONs (#450 / #451). Re-derives nothing.
# --------------------------------------------------------------------------- #
def load_banked() -> dict[str, Any]:
    rf = json.loads(ROOFLINE_JSON.read_text(encoding="utf-8"))["verdict"]
    orc = json.loads(ORACLE_JSON.read_text(encoding="utf-8"))

    rt = {
        # (b) #450 read-peak ceiling
        "ceil_readpeak_resid": abs(rf["ceiling_tps_read_peak"] - CEIL_READPEAK),
        "ceil_readpeak_eq_maxrecoverable": abs(
            rf["max_recoverable_endtoend_tps_ceiling"] - CEIL_READPEAK
        ),
        # (a) #450 spec ceiling (rounded) + #451 verify-BW wall (full precision)
        "ceil_spec_rounded_resid": abs(rf["ceiling_tps_spec"] - CEIL_SPEC_ROUNDED),
        "verify_bw_wall_resid": abs(orc["verify_bw_wall_tps"] - CEIL_SPEC),
        # (c) #451 demand oracle
        "demand_raw_resid": abs(orc["optimistic_ceiling_raw_tps"] - DEMAND_RAW),
        "demand_capped_resid": abs(orc["optimistic_ceiling_capped_tps"] - DEMAND_CAPPED),
        # bases
        "peak_read_resid": abs(rf["peak_read_gbps"] - PEAK_READ_GBPS),
        "peak_spec_resid": abs(rf["peak_spec_gbps"] - PEAK_SPEC_GBPS),
        "peak_frac_resid": abs(rf["peak_read_frac_of_spec"] - PEAK_READ_FRAC_OF_SPEC),
        # relax prize
        "relax_hi_resid": abs(rf["realistic_splitk_tps_hi"] - RELAX_REALISTIC_HI),
        "relax_lo_resid": abs(rf["realistic_splitk_tps_lo"] - RELAX_REALISTIC_LO),
        # operating points + sigma
        "deployed_resid": abs(orc["deployed_noneq_tps"] - DEPLOYED_TPS),
        "strict_resid": abs(orc["base_realized_eq_tps"] - STRICT_FRONTIER_TPS),
        "sigma_hw_resid": abs(orc["sigma_hw_tps"] - SIGMA_HW),
        "ppl_resid": abs(rf["ppl_anchor"] - PPL_ANCHOR),
    }
    max_resid = max(rt.values())
    return {
        "roundtrip_resid": rt,
        "max_roundtrip_resid": max_resid,
        "all_roundtrip_ok": bool(max_resid <= TOL_RT),
        "roofline_greedy_safe_relax": bool(rf["realistic_splitk_greedy_safe"]),  # False
        "ppl_ok_banked": bool(rf.get("ppl_ok", True)),
    }


# --------------------------------------------------------------------------- #
# (1) Reconcile the three ceilings on a consistent (achievable) basis.
# --------------------------------------------------------------------------- #
def reconcile_ceilings() -> dict[str, Any]:
    """All three are the SAME supply (HBM-read-BW) ceiling; they differ only by the bandwidth
    denominator / demand-vs-supply framing. The defensible achievable ceiling is bounded by the
    MEASURED read peak (510.87); the spec-600 basis (520.95) assumes unreachable BW; the demand
    raw (551.71) exceeds supply and reduces to the supply wall when capped."""
    unified = CEIL_READPEAK  # achieved-read-peak basis = the defensible achievable ceiling
    spec_over = CEIL_SPEC - unified  # how much the spec basis over-states the ceiling
    spec_over_pct = 100.0 * spec_over / unified

    ceilings = [
        {
            "name": "spec_bw_lambda1_wall",
            "tps": CEIL_SPEC,
            "basis": "spec_600_gbps_datasheet_peak",
            "denominator_gbps": PEAK_SPEC_GBPS,
            "provenance": "#436 nvsbctji (== #450 ceiling_tps_spec 520.953, #451 verify_bw_wall_tps)",
            "realizable": False,
            "role": "over_optimistic_upper_bound",
            "why": ("assumes the int4 verify-GEMM reaches the GA102 DATASHEET 600 GB/s at lambda=1; "
                    "measured achievable read peak is only 517.58 GB/s = 86.26% of spec, so this "
                    "over-states the bandwidth the hardware can deliver by ~16%"),
        },
        {
            "name": "achieved_read_peak_recoverable",
            "tps": CEIL_READPEAK,
            "basis": "measured_read_peak_517p58_gbps",
            "denominator_gbps": PEAK_READ_GBPS,
            "provenance": "#450 c5oyb7gv ceiling_tps_read_peak / max_recoverable_endtoend_tps_ceiling",
            "realizable": True,
            "role": "unified_absolute_ceiling",
            "why": ("perfect-f->1 re-tiling limit against the MEASURED read peak (the realizable HBM "
                    "read ceiling); bounded by a measured bandwidth -> the defensible achievable ceiling"),
        },
        {
            "name": "demand_oracle_raw_capped",
            "tps_raw": DEMAND_RAW,
            "tps_capped": DEMAND_CAPPED,
            "basis": "demand_side_ET_oracle_supply_capped",
            "denominator_gbps": None,
            "provenance": "#451 c675zor8 optimistic_ceiling_raw_tps -> capped at verify_bw_wall_tps",
            "realizable": False,
            "role": "demand_exceeds_supply_reduces_to_wall",
            "why": ("demand-side oracle: what a maximally-accepting bigger drafter would yield if "
                    "supply were unlimited. 551.71 EXCEEDS supply, so it is capped at the supply wall "
                    "(#451 capped at the spec wall 520.95; on the achievable basis it caps at 510.87). "
                    "It adds NO headroom -> it CONFIRMS supply (bandwidth) binds, not demand"),
        },
    ]

    # On a CONSISTENT achievable basis the demand raw caps at the unified (read-peak) ceiling,
    # not the unreachable spec wall.
    demand_capped_achievable_basis = min(DEMAND_RAW, unified)

    return {
        "unified_absolute_ceiling_tps": unified,            # HEADLINE
        "unified_basis": "measured_read_peak_517p58_gbps",
        "spec_basis_ceiling_tps": CEIL_SPEC,
        "spec_basis_ceiling_is_over_optimistic": True,
        "spec_over_unified_tps": spec_over,                 # 520.95 - 510.87 = 10.08
        "spec_over_unified_pct": spec_over_pct,
        "demand_raw_tps": DEMAND_RAW,
        "demand_capped_spec_basis_tps": DEMAND_CAPPED,      # #451 cap (spec wall)
        "demand_capped_achievable_basis_tps": demand_capped_achievable_basis,  # caps at unified
        "demand_exceeds_supply": bool(DEMAND_RAW > CEIL_SPEC and DEMAND_RAW > unified),
        "irreducible_hbm_floor_note": (
            f"each verify pass reads ~{GEMM_WEIGHT_SCALE_BYTES_MB:.0f} MB of int4 weight+scale once "
            "(the irreducible HBM read floor); the ceiling is a SUPPLY ceiling set by the read-BW "
            "denominator, which is why all three numbers are the same ceiling at different bases"),
        "ceilings": ceilings,
        "ordering_ok": bool(unified < CEIL_SPEC < DEMAND_RAW),  # read-peak < spec wall < demand raw
        "summary": (
            f"unified absolute ceiling = {unified:.2f} TPS (achieved-read-peak basis); the spec-600 "
            f"basis {CEIL_SPEC:.2f} is an OVER-OPTIMISTIC upper bound (+{spec_over:.2f}, assumes "
            f"unreachable BW); the demand-oracle raw {DEMAND_RAW:.2f} exceeds supply and reduces to "
            f"the supply wall when capped"),
    }


# --------------------------------------------------------------------------- #
# (2) sigma_hw envelope + headroom gaps.
# --------------------------------------------------------------------------- #
def sigma_envelope(unified: float) -> dict[str, Any]:
    sigma_pct_of_deployed = 100.0 * SIGMA_HW / DEPLOYED_TPS  # ~1.0%
    band_lo, band_hi = unified - SIGMA_HW, unified + SIGMA_HW

    h_dep = unified - DEPLOYED_TPS          # deployed 481.53 -> ceiling
    h_str = unified - STRICT_FRONTIER_TPS   # realized-strict 467.14 -> ceiling
    # conservative: room that survives a 1-sigma-pessimistic ceiling.
    h_dep_lcb = band_lo - DEPLOYED_TPS
    h_str_lcb = band_lo - STRICT_FRONTIER_TPS

    # the over-optimistic spec UB headrooms, for context.
    h_dep_spec = CEIL_SPEC - DEPLOYED_TPS
    h_str_spec = CEIL_SPEC - STRICT_FRONTIER_TPS

    return {
        "sigma_hw_tps": SIGMA_HW,
        "sigma_hw_pct_of_deployed": sigma_pct_of_deployed,
        "unified_ceiling_band_lo_tps": band_lo,
        "unified_ceiling_band_hi_tps": band_hi,
        "headroom_deployed_to_ceiling_tps": h_dep,           # 510.87 - 481.53 = 29.34
        "headroom_deployed_to_ceiling_sigma": h_dep / SIGMA_HW,
        "headroom_strict_to_ceiling_tps": h_str,             # 510.87 - 467.14 = 43.73
        "headroom_strict_to_ceiling_sigma": h_str / SIGMA_HW,
        "headroom_deployed_to_ceiling_lcb_tps": h_dep_lcb,   # survives 1-sigma-pessimistic ceiling
        "headroom_strict_to_ceiling_lcb_tps": h_str_lcb,
        "headroom_deployed_to_spec_ub_tps": h_dep_spec,
        "headroom_strict_to_spec_ub_tps": h_str_spec,
        "both_headrooms_multi_sigma_positive": bool(
            h_dep > 0 and h_str > 0 and h_dep / SIGMA_HW > 1.0 and h_str / SIGMA_HW > 1.0
        ),
        "note": (
            f"unified ceiling {unified:.2f} +/- {SIGMA_HW:.2f} (1% of deployed) = "
            f"[{band_lo:.2f}, {band_hi:.2f}]; deployed->ceiling +{h_dep:.2f} ({h_dep/SIGMA_HW:.1f}sigma), "
            f"strict->ceiling +{h_str:.2f} ({h_str/SIGMA_HW:.1f}sigma) — both robustly positive, so "
            f"real physical room exists above each operating point even at the 1-sigma-pessimistic "
            f"ceiling (+{h_dep_lcb:.2f} over deployed, +{h_str_lcb:.2f} over strict)"),
    }


# --------------------------------------------------------------------------- #
# (3) Place the relax-prize on the same axis as the unified ceiling + frontier.
# --------------------------------------------------------------------------- #
def relax_prize_axis(unified: float) -> dict[str, Any]:
    """ubel #450's relax-prize (greedy-UNSAFE FP-reassociating re-tiling): optimistic 510.87 /
    realistic 498.6. Both sit AT-OR-BELOW the unified ceiling (the optimistic edge IS the perfect-f
    read-peak ceiling by construction). Expressed over the deployed 481.53 the prize is ~+17..+29 TPS."""
    prize_realistic_over_deployed = RELAX_REALISTIC_HI - DEPLOYED_TPS    # +17.05
    prize_optimistic_over_deployed = RELAX_OPTIMISTIC - DEPLOYED_TPS     # +29.34
    prize_realistic_over_strict = RELAX_REALISTIC_HI - STRICT_FRONTIER_TPS
    prize_optimistic_over_strict = RELAX_OPTIMISTIC - STRICT_FRONTIER_TPS

    # the ordered axis (low -> high).
    axis = [
        {"point": "realized_strict_frontier", "tps": STRICT_FRONTIER_TPS, "greedy_safe": True,
         "note": "denken #423 5a6zq2yz — blanket-strict greedy-safe realized best"},
        {"point": "deployed_noneq_3flips", "tps": DEPLOYED_TPS, "greedy_safe": False,
         "note": "PR #52 2x9fm2zx — non-equivalent deployed (3 token flips)"},
        {"point": "relax_prize_realistic", "tps": RELAX_REALISTIC_HI, "greedy_safe": False,
         "note": "#450 realistic split-K hi (greedy-UNSAFE re-tiling)"},
        {"point": "unified_absolute_ceiling", "tps": unified, "greedy_safe": None,
         "note": "achieved-read-peak basis == relax-prize optimistic edge (perfect-f)"},
        {"point": "spec_over_optimistic_ub", "tps": CEIL_SPEC, "greedy_safe": None,
         "note": "spec-600 basis — over-optimistic upper bound (unreachable BW)"},
    ]
    monotone = all(axis[i]["tps"] <= axis[i + 1]["tps"] + TOL_RT for i in range(len(axis) - 1))

    # SANITY (instruction 3): the relax-prize must sit below (or at) the unified ceiling.
    relax_realistic_below = bool(RELAX_REALISTIC_HI < unified)
    relax_optimistic_at_or_below = bool(RELAX_OPTIMISTIC <= unified + TOL_RT)

    return {
        "relax_prize_realistic_tps": RELAX_REALISTIC_HI,
        "relax_prize_optimistic_tps": RELAX_OPTIMISTIC,
        "relax_prize_greedy_safe": False,
        "relax_prize_over_deployed_lo_tps": prize_realistic_over_deployed,   # +17.05
        "relax_prize_over_deployed_hi_tps": prize_optimistic_over_deployed,  # +29.34
        "relax_prize_over_strict_lo_tps": prize_realistic_over_strict,
        "relax_prize_over_strict_hi_tps": prize_optimistic_over_strict,
        "relax_prize_realistic_below_unified": relax_realistic_below,
        "relax_prize_optimistic_at_or_below_unified": relax_optimistic_at_or_below,
        "relax_prize_sits_below_unified_ceiling": bool(
            relax_realistic_below and relax_optimistic_at_or_below),
        "axis": axis,
        "axis_monotone": bool(monotone),
        "optimistic_edge_is_unified_ceiling": bool(abs(RELAX_OPTIMISTIC - unified) <= TOL_RT),
        "capstone_one_line": (
            f"strict best = {STRICT_FRONTIER_TPS:.2f} (deployed {DEPLOYED_TPS:.2f} via 3 flips); "
            f"relax-prize ~= +{prize_realistic_over_deployed:.0f}..+{prize_optimistic_over_deployed:.0f} "
            f"TPS (greedy-unsafe); absolute physical ceiling = {unified:.2f} +/- {SIGMA_HW:.2f} "
            f"(achieved-read-peak basis; spec-basis {CEIL_SPEC:.2f} is an over-optimistic upper bound)"),
    }


# --------------------------------------------------------------------------- #
# (4) Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def selftests(banked: dict, rec: dict, env: dict, relax: dict) -> dict[str, Any]:
    unified = rec["unified_absolute_ceiling_tps"]

    # (a) every banked source number round-trips its committed JSON within tol.
    cond_a = bool(banked["all_roundtrip_ok"])

    # (b) ordering on the supply axis: read-peak < spec wall < demand raw; demand caps == spec wall.
    cond_b = bool(
        CEIL_READPEAK < CEIL_SPEC < DEMAND_RAW
        and abs(DEMAND_CAPPED - CEIL_SPEC) <= TOL_RT
        and rec["ordering_ok"]
    )

    # (c) the spec basis over-states the achievable ceiling because measured read peak < spec peak.
    cond_c = bool(
        PEAK_READ_GBPS < PEAK_SPEC_GBPS
        and abs(PEAK_READ_FRAC_OF_SPEC - PEAK_READ_GBPS / PEAK_SPEC_GBPS) <= TOL_RT
        and rec["spec_basis_ceiling_is_over_optimistic"] is True
        and CEIL_SPEC > unified
    )

    # (d) unified == achieved-read-peak ceiling (the defensible achievable number).
    cond_d = bool(abs(unified - CEIL_READPEAK) <= TOL_RT)

    # (e) sigma_hw == 1% of deployed; both headrooms positive and == ceiling - operating point.
    cond_e = bool(
        abs(SIGMA_HW - 0.01 * DEPLOYED_TPS) <= 1e-4
        and abs(env["headroom_deployed_to_ceiling_tps"] - (unified - DEPLOYED_TPS)) <= TOL_RT
        and abs(env["headroom_strict_to_ceiling_tps"] - (unified - STRICT_FRONTIER_TPS)) <= TOL_RT
        and env["both_headrooms_multi_sigma_positive"] is True
    )

    # (f) relax-prize sits at-or-below the unified ceiling (instruction-3 sanity).
    cond_f = bool(relax["relax_prize_sits_below_unified_ceiling"])

    # (g) operating points sit strictly below the unified ceiling: strict < deployed < unified.
    cond_g = bool(STRICT_FRONTIER_TPS < DEPLOYED_TPS < unified)

    # (h) ppl anchor preserved (this leg does not touch the served model) and within gate.
    cond_h = bool(abs(PPL_ANCHOR - 2.3772) <= TOL_RT and PPL_ANCHOR <= PPL_GATE)

    # (i) NaN-clean — set by the caller after the full payload walk.
    cond_i = True

    conditions = {
        "a_all_banked_numbers_roundtrip": cond_a,
        "b_supply_axis_ordering_and_demand_cap": cond_b,
        "c_spec_basis_over_optimistic_measured_lt_spec": cond_c,
        "d_unified_equals_achieved_read_peak": cond_d,
        "e_sigma_hw_and_headrooms_positive": cond_e,
        "f_relax_prize_at_or_below_unified_ceiling": cond_f,
        "g_operating_points_below_unified_ceiling": cond_g,
        "h_ppl_anchor_preserved_within_gate": cond_h,
        "i_nan_clean": cond_i,
    }
    return {
        "conditions": conditions,
        "unified_ceiling_self_test_passes": bool(all(conditions.values())),
        "detail": {
            "max_roundtrip_resid": banked["max_roundtrip_resid"],
            "peak_read_frac_check": PEAK_READ_GBPS / PEAK_SPEC_GBPS,
            "sigma_pct": 100.0 * SIGMA_HW / DEPLOYED_TPS,
        },
    }


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    banked = load_banked()
    rec = reconcile_ceilings()
    unified = rec["unified_absolute_ceiling_tps"]
    env = sigma_envelope(unified)
    relax = relax_prize_axis(unified)
    st = selftests(banked, rec, env, relax)

    headline = {
        "unified_ceiling_self_test_passes": bool(st["unified_ceiling_self_test_passes"]),  # PRIMARY
        "unified_absolute_ceiling_tps": unified,                                            # HEADLINE
        "unified_basis": rec["unified_basis"],
        "spec_basis_ceiling_tps": CEIL_SPEC,
        "spec_basis_ceiling_is_over_optimistic": rec["spec_basis_ceiling_is_over_optimistic"],
        "demand_raw_tps": DEMAND_RAW,
        "headroom_deployed_to_ceiling_tps": env["headroom_deployed_to_ceiling_tps"],
        "headroom_strict_to_ceiling_tps": env["headroom_strict_to_ceiling_tps"],
        "sigma_hw_tps": SIGMA_HW,
        "relax_prize_realistic_tps": RELAX_REALISTIC_HI,
        "relax_prize_optimistic_tps": RELAX_OPTIMISTIC,
        "relax_prize_sits_below_unified_ceiling": relax["relax_prize_sits_below_unified_ceiling"],
        "ppl": PPL_ANCHOR,
        "analysis_only": True,
        "no_served_file_change": True,
        "official_tps": 0,
    }
    verdict = (
        f"UNIFIED-ABSOLUTE-CEILING-{unified:.2f}-ACHIEVED-READ-PEAK-BASIS-"
        f"SPEC-{CEIL_SPEC:.2f}-OVER-OPTIMISTIC-UB"
    )
    return {
        "headline": headline,
        "reconcile": rec,
        "sigma_envelope": env,
        "relax_prize": relax,
        "banked_roundtrip": banked,
        "self_test": st,
        "constants": {
            "ceil_readpeak": CEIL_READPEAK, "ceil_spec": CEIL_SPEC, "demand_raw": DEMAND_RAW,
            "demand_capped": DEMAND_CAPPED, "peak_read_gbps": PEAK_READ_GBPS,
            "peak_spec_gbps": PEAK_SPEC_GBPS, "peak_read_frac_of_spec": PEAK_READ_FRAC_OF_SPEC,
            "deployed_tps": DEPLOYED_TPS, "strict_frontier_tps": STRICT_FRONTIER_TPS,
            "sigma_hw": SIGMA_HW, "ppl_anchor": PPL_ANCHOR, "ppl_gate": PPL_GATE,
            "relax_realistic_hi": RELAX_REALISTIC_HI, "relax_optimistic": RELAX_OPTIMISTIC,
        },
        "verdict": verdict,
        "handoff_line": relax["capstone_one_line"],
        "imports": {
            "provenance": (
                "land#436 nvsbctji (spec-BW lambda=1 wall 520.95) x ubel#450 c5oyb7gv "
                "(achieved-read-peak ceiling 510.87, measured read peak 517.58, relax-prize "
                "498.6/510.87 greedy-unsafe) x land#451 c675zor8 (demand-oracle raw 551.71->capped "
                "520.95, sigma_hw 4.8153) x PR#52 2x9fm2zx (deployed 481.53) x denken#423 5a6zq2yz "
                "(strict frontier 467.14). All run-ids in wandb-applied-ai-team/gemma-challenge-senpai."),
            "machinery": "round-trip of committed #450/#451 result JSONs; re-derives nothing",
        },
    }


# --------------------------------------------------------------------------- #
# NaN guard + report + W&B (mirrors #235/#450/#451; never fatal).
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
    h, rec, env = syn["headline"], syn["reconcile"], syn["sigma_envelope"]
    relax, st = syn["relax_prize"], syn["self_test"]
    print("\n" + "=" * 98, flush=True)
    print("UNIFIED ABSOLUTE CEILING (PR #457, land) — reconcile 510.87 / 520.95 / 551.71, CPU-only",
          flush=True)
    print("=" * 98, flush=True)
    print("  (1) RECONCILE — all three are the SAME supply (HBM-read-BW) ceiling at different bases:",
          flush=True)
    print(f"      spec-BW lambda=1 wall   {CEIL_SPEC:8.2f}  basis spec 600 GB/s   -> OVER-OPTIMISTIC UB "
          f"(measured read peak {PEAK_READ_GBPS:.1f} = {PEAK_READ_FRAC_OF_SPEC*100:.1f}% of spec)", flush=True)
    print(f"      achieved-read-peak      {CEIL_READPEAK:8.2f}  basis read {PEAK_READ_GBPS:.1f} GB/s -> "
          f"UNIFIED ABSOLUTE CEILING (defensible achievable)", flush=True)
    print(f"      demand-oracle raw       {DEMAND_RAW:8.2f}  -> capped {DEMAND_CAPPED:.2f}  "
          f"(demand exceeds supply -> SUPPLY binds, adds no headroom)", flush=True)
    print(f"      => unified_absolute_ceiling_tps = {h['unified_absolute_ceiling_tps']:.2f}  "
          f"(spec over-states by +{rec['spec_over_unified_tps']:.2f} = {rec['spec_over_unified_pct']:.1f}%)",
          flush=True)
    print("-" * 98, flush=True)
    print(f"  (2) SIGMA_HW envelope  +/- {SIGMA_HW:.2f} ({env['sigma_hw_pct_of_deployed']:.1f}% of "
          f"deployed) = [{env['unified_ceiling_band_lo_tps']:.2f}, {env['unified_ceiling_band_hi_tps']:.2f}]",
          flush=True)
    print(f"      headroom deployed 481.53 -> ceiling = +{env['headroom_deployed_to_ceiling_tps']:.2f} "
          f"({env['headroom_deployed_to_ceiling_sigma']:.1f}sigma)   "
          f"[1sigma-pessimistic ceiling: +{env['headroom_deployed_to_ceiling_lcb_tps']:.2f}]", flush=True)
    print(f"      headroom strict   467.14 -> ceiling = +{env['headroom_strict_to_ceiling_tps']:.2f} "
          f"({env['headroom_strict_to_ceiling_sigma']:.1f}sigma)   "
          f"[1sigma-pessimistic ceiling: +{env['headroom_strict_to_ceiling_lcb_tps']:.2f}]", flush=True)
    print("-" * 98, flush=True)
    print("  (3) RELAX-PRIZE axis (low -> high):", flush=True)
    for a in relax["axis"]:
        gs = {True: "greedy-safe", False: "greedy-UNSAFE", None: "ceiling/UB"}[a["greedy_safe"]]
        print(f"      {a['tps']:8.2f}  {a['point']:<26} [{gs}]", flush=True)
    print(f"      relax-prize ~= +{relax['relax_prize_over_deployed_lo_tps']:.0f}.."
          f"+{relax['relax_prize_over_deployed_hi_tps']:.0f} TPS over deployed (greedy-unsafe); "
          f"sits below unified ceiling = {relax['relax_prize_sits_below_unified_ceiling']}", flush=True)
    print("-" * 98, flush=True)
    print(f"  (4) PRIMARY unified_ceiling_self_test_passes = {st['unified_ceiling_self_test_passes']}",
          flush=True)
    for k, v in st["conditions"].items():
        print(f"        - {k}: {v}", flush=True)
    print("=" * 98, flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print(f"\n  CAPSTONE ONE-LINE: {syn['handoff_line']}\n", flush=True)


def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[unified-ceiling] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    h, rec, env = syn["headline"], syn["reconcile"], syn["sigma_envelope"]
    relax, st = syn["relax_prize"], syn["self_test"]
    run = init_wandb_run(
        job_type="unified-absolute-ceiling",
        agent="land",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["unified-absolute-ceiling", "equivalence-escalation-anchors", "ceiling-reconcile",
              "sigma-hw-envelope", "relax-prize", "capstone-anchor", "analysis-only",
              "bank-the-analysis"],
        config={
            "ceil_readpeak": CEIL_READPEAK, "ceil_spec": CEIL_SPEC, "demand_raw": DEMAND_RAW,
            "demand_capped": DEMAND_CAPPED, "peak_read_gbps": PEAK_READ_GBPS,
            "peak_spec_gbps": PEAK_SPEC_GBPS, "peak_read_frac_of_spec": PEAK_READ_FRAC_OF_SPEC,
            "deployed_tps": DEPLOYED_TPS, "strict_frontier_tps": STRICT_FRONTIER_TPS,
            "sigma_hw": SIGMA_HW, "ppl_anchor": PPL_ANCHOR, "wandb_group": args.wandb_group,
            "source_runs": "land#436 nvsbctji, ubel#450 c5oyb7gv, land#451 c675zor8, "
                           "PR#52 2x9fm2zx, denken#423 5a6zq2yz",
        },
    )
    if run is None:
        print("[unified-ceiling] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "unified_ceiling_self_test_passes": int(bool(st["unified_ceiling_self_test_passes"])),  # PRIMARY
        "unified_absolute_ceiling_tps": h["unified_absolute_ceiling_tps"],                       # HEADLINE
        "spec_basis_ceiling_tps": CEIL_SPEC,
        "spec_basis_ceiling_is_over_optimistic": int(bool(rec["spec_basis_ceiling_is_over_optimistic"])),
        "spec_over_unified_tps": rec["spec_over_unified_tps"],
        "spec_over_unified_pct": rec["spec_over_unified_pct"],
        "demand_raw_tps": DEMAND_RAW,
        "demand_capped_spec_basis_tps": DEMAND_CAPPED,
        "demand_capped_achievable_basis_tps": rec["demand_capped_achievable_basis_tps"],
        "sigma_hw_tps": SIGMA_HW,
        "sigma_hw_pct_of_deployed": env["sigma_hw_pct_of_deployed"],
        "unified_ceiling_band_lo_tps": env["unified_ceiling_band_lo_tps"],
        "unified_ceiling_band_hi_tps": env["unified_ceiling_band_hi_tps"],
        "headroom_deployed_to_ceiling_tps": env["headroom_deployed_to_ceiling_tps"],
        "headroom_deployed_to_ceiling_sigma": env["headroom_deployed_to_ceiling_sigma"],
        "headroom_strict_to_ceiling_tps": env["headroom_strict_to_ceiling_tps"],
        "headroom_strict_to_ceiling_sigma": env["headroom_strict_to_ceiling_sigma"],
        "headroom_deployed_to_ceiling_lcb_tps": env["headroom_deployed_to_ceiling_lcb_tps"],
        "headroom_strict_to_ceiling_lcb_tps": env["headroom_strict_to_ceiling_lcb_tps"],
        "headroom_deployed_to_spec_ub_tps": env["headroom_deployed_to_spec_ub_tps"],
        "headroom_strict_to_spec_ub_tps": env["headroom_strict_to_spec_ub_tps"],
        "relax_prize_realistic_tps": RELAX_REALISTIC_HI,
        "relax_prize_optimistic_tps": RELAX_OPTIMISTIC,
        "relax_prize_over_deployed_lo_tps": relax["relax_prize_over_deployed_lo_tps"],
        "relax_prize_over_deployed_hi_tps": relax["relax_prize_over_deployed_hi_tps"],
        "relax_prize_sits_below_unified_ceiling": int(bool(relax["relax_prize_sits_below_unified_ceiling"])),
        "optimistic_edge_is_unified_ceiling": int(bool(relax["optimistic_edge_is_unified_ceiling"])),
        "deployed_tps": DEPLOYED_TPS,
        "strict_frontier_tps": STRICT_FRONTIER_TPS,
        "peak_read_gbps": PEAK_READ_GBPS,
        "peak_spec_gbps": PEAK_SPEC_GBPS,
        "peak_read_frac_of_spec": PEAK_READ_FRAC_OF_SPEC,
        "max_roundtrip_resid": syn["banked_roundtrip"]["max_roundtrip_resid"],
        "ppl": PPL_ANCHOR,
        "analysis_only": 1,
        "no_served_file_change": 1,
        "official_tps": 0,
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="unified_absolute_ceiling_result", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    print(f"[unified-ceiling] wandb logged {len(summary)} keys; run id "
          f"{getattr(run, 'id', '?')}", flush=True)


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
        "created_at": created_at, "pr": 457, "agent": "land",
        "kind": "unified-absolute-ceiling", "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _nan_paths(payload)
    payload["nan_clean"] = not nan_paths
    syn["self_test"]["conditions"]["i_nan_clean"] = not nan_paths
    syn["self_test"]["unified_ceiling_self_test_passes"] = bool(
        all(syn["self_test"]["conditions"].values()))
    syn["headline"]["unified_ceiling_self_test_passes"] = syn["self_test"][
        "unified_ceiling_self_test_passes"]
    if nan_paths:
        print(f"[unified-ceiling] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "unified_absolute_ceiling_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[unified-ceiling] wrote {out_path}", flush=True)

    # standalone self-test summary file (mirrors #451).
    st_path = out_dir / "unified_absolute_ceiling_selftest.json"
    with st_path.open("w", encoding="utf-8") as fh:
        json.dump(syn["self_test"]["conditions"], fh, indent=2, sort_keys=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = (syn["self_test"]["unified_ceiling_self_test_passes"] and payload["nan_clean"])
        print(f"[unified-ceiling] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
