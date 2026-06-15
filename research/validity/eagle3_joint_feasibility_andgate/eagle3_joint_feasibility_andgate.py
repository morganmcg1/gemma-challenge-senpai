#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Joint compliant-500 AND-gate (PR #335, fern) -- CPU-only analytic synthesis.

THE GOVERNING QUESTION (one integrated GO/NO-GO over the whole EAGLE-3 compliant-500 lane)
-----------------------------------------------------------------------------------------
fern #325 (`xk1pghy4`) built the joint compliant-500 envelope -- central 520.95 / worst-case
492.87 -- but it stood on TWO load-bearing assumptions, each of which took a measured hit this
cycle. The envelope is an AND of an IDENTITY half (a strict-compliant verify kernel must exist)
and a DEMAND half (an EAGLE-3 E[T] lever must deliver). This card folds the two new measured
corners into the #325 envelope and produces the honest JOINT verdict, so the next human-facing
decision rests on one integrated GO/NO-GO rather than three separate cards.

  SUPPLY (identity half) -- denken #327 (`kcjlr5ny`): the batch-invariant bf16 lm_head + TRITON
  SDPA reduction-order FLOOR is 9.841 % of step > the lambda=1 budget 7.332 % -> the strict-
  compliant identity lane is (C) DEAD under the measured *linear* BW-gap law. It revives to (B)
  only if a deterministic SDPA recovers enough of the forgone split-KV parallelism; break-even
  is the kept-slack ratio phi* = 74.5 % (== a recovery fraction of 25.5 %). denken #332 (in-
  flight) is reconstructing the geometric phi to decide C <-> B.

  DEMAND (E[T] half) -- lawine #330 (`hfrscdai`): the official 128 "ShareGPT" eval is 100 %
  reasoning/STEM, so the honest unconditional top-4 coverage prior is 0.8903, which MISSES the
  0.9213 build bar by 0.031 (P(clears) ~ 0.06). It revives only with a better head delivering an
  aggregate per-source coverage lift Delta_cov >= 0.031.

The two halves are an AND-gate: compliant-500 needs BOTH to revive. If EITHER stays dead, the
EAGLE-3 compliant-500 lane is DEAD and should hand off to a fresh non-EAGLE-3 direction.

THE phi NOTATION (a deliberate reconciliation; read before the supply axis)
---------------------------------------------------------------------------
The PR overloads "phi". This card fixes ONE canonical free variable:

    phi  ==  the deterministic-SDPA split-KV parallelism RECOVERY fraction in [0, 1].

  - effective_floor(phi) = FLOOR_COMBINED * (1 - phi). phi=0 (no recovery, denken #327's measured
    LINEAR-LAW corner) -> the full 9.841 % floor; phi=1 (ideal recovery) -> 0 %.
  - revised identity ceiling(phi) = LAMBDA_CEIL * (1 - effective_floor(phi)). phi=0 -> 469.68
    (== denken #327 ceiling_at_floor); phi=1 -> 520.95 (== #325 identity ceiling).
  - SUPPLY revives to (B) <=> effective_floor(phi) <= budget <=> phi >= 1 - budget/floor
    = PHI_REC_STAR = 0.255. Equivalently, in kept-slack units psi = 1 - phi, supply revives
    <=> psi <= psi* = budget/floor = 0.745. So the SAME break-even number 0.745 the PR quotes is
    the kept-slack ratio psi*; the recovery break-even is 1 - 0.745 = 0.255.
  - The PR's 2x2 row spec "{C: phi>=0.745 fails, B: phi<0.745}" is stated in the kept-slack
    variable psi (small kept-slack == lots of recovery == alive). The PR's "phi=0 -> floor
    9.841 % -> C", "compliant ceiling at phi=0 = 469.68", and the self-test "(phi=0, Delta_cov=0)
    -> NO-GO" are all stated in the recovery variable phi (zero recovery == dead). Both are the
    SAME physical no-recovery corner: phi=0 <=> psi=1. This card carries BOTH and the report keys
    the 2x2 rows by the supply STATE {C, B} with both thresholds annotated, so it matches the PR
    letter-for-letter regardless of which symbol a reader has in mind.

THE LEDGER (everything imported; NOTHING re-measured)
-----------------------------------------------------
    compliant_tps(phi, demand, rho)
        = min( demand_public_tps(demand, rho),  revised_ceiling(phi) )
    revised_ceiling(phi) = LAMBDA_CEIL * (1 - FLOOR_COMBINED * (1 - phi))
    demand_public_tps(clear, rho) = HONEST_PUBLIC_611 * rho     # E[T] lever reaches the build target
    demand_public_tps(miss,  .)   = BASELINE_TPS = 481.53       # build fails the bar -> deployed frontier
    GREEN(phi, Delta_cov, rho) = G_supply(phi) AND G_demand(Delta_cov) AND (rho >= 0.8038)
                                 AND compliant_tps >= 500

stark #331 (`b48rmwjq`) pins the denominator: E[T]=6.11 top-4 salvage tree verifies at
M = W*K + 1 = 4*7 + 1 = 29 tokens = 2 Marlin thread_m_blocks = the SAME tile as #325's M=32
anchor, so the verify-GEMM denominator is UNCHANGED and the M=33 cliff (+16.98 % step) does NOT
bind. The #325 step (and hence the ledger) is valid for the salvage tree.

LOCAL, CPU-ONLY, ANALYTIC. 0 GPU, no model forward, no publish, no served-file change, no HF Job,
no submission, no build, no launch. BASELINE stays 481.53; this card adds 0 TPS -- it produces the
integrated feasibility verdict, not a speedup. Imports fern #325 (xk1pghy4), denken #327
(kcjlr5ny), lawine #330 (hfrscdai), stark #331 (b48rmwjq); re-derives nothing.

PRIMARY metric  joint_feasibility_andgate_self_test_passes
TEST    metric  p_both_revivals_land   (P(supply revives) * P(demand clears))
REPORT          binding_axis_at_measured_corner
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
# IMPORTED ANCHORS -- cited EXACTLY, UNCHANGED, with source. This card re-derives none.
# --------------------------------------------------------------------------- #
# fern #325 (xk1pghy4) -- the standing joint compliant-500 envelope + its composition law.
K_CAL = 125.26795005202914            # kanna #269 tokens/sec per E[T]
TAU = 1.218                           # denken #278 served-fraction frame
STEP_DEPLOYED = 1.2182                # deployed split-K M=32 step, normalized (1218.2us)
TAU_OVER_STEP = TAU / STEP_DEPLOYED   # ~0.99984
E_T_DEPLOYED = 3.844                  # deployed native accepted-tok/step (stark #266)
LAMBDA_CEIL = 520.9527323111674       # int4-spec batch-invariant verify ceiling (#204/#220)
HONEST_PUBLIC_611 = 622.080888        # = K_cal * realized_public_et (realized=4.966)
REALIZED_PUBLIC_ET = 4.966            # EAGLE-3 paper public acceptance length
E_T_BUILD_FREE = 6.1112149873699195   # free build target (wirbel #295)
RHO_CENTRAL = 0.9421228821714434      # #318 central private-tax
RHO_BREAKEVEN = 0.8037539966988988    # #318 break-even (= 500 / 622.08) -- the rho_priv floor 0.8038
RHO_WORST = 0.7922848664688427        # #318 PRIMARY worst-case cross-dataset
PRIV_TPS_CENTRAL_318 = 586.0766391463308   # #318 uncapped central (round-trip target)
PRIV_TPS_WORST_318 = 492.865273281899      # #318 uncapped worst (== capped worst; < ceiling)
ENVELOPE_CENTRAL_325 = 520.9527323111674   # #325 banked joint central (capped at LAMBDA_CEIL)
ENVELOPE_WORST_325 = 492.865273281899      # #325 banked joint worst

# denken #327 (kcjlr5ny) -- the batch-invariant bf16 lm_head+attn reduction FLOOR (supply axis).
# Re-derived here from the denken #291 measured exposed_us rows so the floor is reproduced, not
# hardcoded (the PR: "don't hardcode without the derivation").
SDPA_EXPOSED_US_291 = 505.43955671223955    # bf16 SDPA above-roofline exposed slack @ 34.9% BW
LMHEAD_EXPOSED_US_291 = 20.88292794270834   # bf16 tied lm_head exposed slack @ 83.4% BW
NORMS_EXPOSED_US_291 = 0.0                   # io/residual/norm catch-all (measured 0)
TOTAL_VERIFY_US_291 = 5348.1268310546875    # denken #291 full M=8 verify forward
SDPA_BW_UTIL = 0.34883864849061247          # bf16 SDPA bandwidth utilization (report only)
LMHEAD_BW_UTIL = 0.8344417980018903         # bf16 lm_head bandwidth utilization (report only)
BUDGET_LAMBDA1_FRAC = 0.07331808522875782   # wirbel #213 lambda=1 identity-overhead budget (7.332%)
# denken #327 banked headline values (drift cross-check, not the source of truth).
FLOOR_COMBINED_327 = 0.09841249119201488
PHI_STAR_KEPT_327 = 0.7450079186157905      # kept-slack break-even == budget/floor
RECOVERY_NEEDED_327 = 0.2549920813842095    # recovery break-even == 1 - budget/floor
CEILING_AT_FLOOR_327 = 469.6844761311386    # revised ceiling at phi=0 (full floor)
CONVEX_REVIVES_327 = True                   # #327 optimistic pi=(1-u)^2 sub-floor revives the lane

# lawine #330 (hfrscdai) -- the official-eval unconditional top-4 coverage PRIOR (demand axis).
COV_BAR = 0.9213011665456927          # lawine #316 regime-invariant build bar (uncond top-4)
# per-source unconditional top-4 (fern #34 gua9x68j, W&B-verified) + official-eval counts (57/57/14).
COV_PER_SOURCE = {"aime": 0.957005303537408,
                  "gpqa": 0.9175953770859131,
                  "mmlu_pro": 0.846544405293677}
OFFICIAL_EVAL_COUNTS = {"aime": 14, "gpqa": 57, "mmlu_pro": 57}   # by id prefix; sum = 128
N_EVAL_PROMPTS = 128
COV_PRIOR_330 = 0.8902659519153152    # banked live-weighted prior (drift cross-check)
P_DEMAND_CLEARS_330 = 0.06031894029725235   # banked P(official-eval uncond top-4 >= 0.9213)

# stark #331 (b48rmwjq) -- the sub-cliff verify tile (denominator pin).
TREE_WIDTH_W = 4                      # top-4 tree width
TREE_DEPTH_K = 7                      # K_spec depth (lawine #309 import)
MARLIN_THREAD_M_BLOCK = 16           # Marlin tiles M in groups of 16 rows
M_ANCHOR_325 = 32                    # #325's M=32 verify-GEMM anchor tile
CLIFF_STEP_PCT_M33 = 16.98           # stark #331 measured M=33 (3-tile) step cliff (%)

BASELINE_TPS = 481.53
TARGET = 500.0

TOL = 1e-9
TOL_REPRO = 1e-6
TOL_ANCHOR = 0.5     # deployed/honest-public reproduction tolerance (3.844 is rounded)


# --------------------------------------------------------------------------- #
# Composition primitives (all imported; nothing re-measured).
# --------------------------------------------------------------------------- #
def official_tps(et: float) -> float:
    """#325 composition law at the deployed step (tau/step ~ 1): mu = K_cal * E[T] * tau/step."""
    return K_CAL * et * TAU_OVER_STEP


def floor_combined() -> float:
    """denken #327 BW-gap floor = sum(locus exposed_us) / total_verify_us (SDPA + lm_head + norms)."""
    return (SDPA_EXPOSED_US_291 + LMHEAD_EXPOSED_US_291 + NORMS_EXPOSED_US_291) / TOTAL_VERIFY_US_291


def effective_floor(phi: float) -> float:
    """Identity-half step-inflation floor after a deterministic SDPA recovers fraction phi of the
    forgone split-KV parallelism. phi=0 -> full linear-law floor; phi=1 -> 0."""
    return floor_combined() * (1.0 - phi)


def revised_ceiling(phi: float) -> float:
    """#325 identity ceiling deflated by the #327 step-inflation floor at recovery phi."""
    return LAMBDA_CEIL * (1.0 - effective_floor(phi))


def coverage_prior() -> float:
    """lawine #330 official-eval unconditional top-4 prior = counts-weighted per-source coverage."""
    return sum(COV_PER_SOURCE[k] * OFFICIAL_EVAL_COUNTS[k] for k in COV_PER_SOURCE) / N_EVAL_PROMPTS


def marlin_tiles(m: int) -> int:
    """Marlin thread_m_blocks tile count for a verify width M (groups of 16 rows)."""
    return math.ceil(m / MARLIN_THREAD_M_BLOCK)


# Supply / demand gates (PRIMARY analytic objects).
def g_supply(phi: float) -> bool:
    """Identity half revives to (B) <=> the deterministic-SDPA floor fits the lambda=1 budget."""
    return effective_floor(phi) <= BUDGET_LAMBDA1_FRAC + TOL


def g_demand(delta_cov: float) -> bool:
    """Demand half clears <=> prior + Delta_cov reaches the build bar."""
    return coverage_prior() + delta_cov >= COV_BAR - TOL


def demand_public_tps(demand_clears: bool, rho: float) -> float:
    """E[T]-lever public TPS: clear -> 622.08*rho (build reaches target); miss -> deployed 481.53."""
    return HONEST_PUBLIC_611 * rho if demand_clears else BASELINE_TPS


def compliant_tps(phi: float, demand_clears: bool, rho: float) -> float:
    """The joint ledger: E[T]-lever public TPS capped at the floor-deflated identity ceiling."""
    return min(demand_public_tps(demand_clears, rho), revised_ceiling(phi))


# --------------------------------------------------------------------------- #
# (1) Re-state the standing #325 envelope from inputs + pin the stark #331 tile.
# --------------------------------------------------------------------------- #
def restate_envelope() -> dict[str, Any]:
    deployed_repro = official_tps(E_T_DEPLOYED)
    honest_public_repro = official_tps(REALIZED_PUBLIC_ET)
    central = min(HONEST_PUBLIC_611 * RHO_CENTRAL, LAMBDA_CEIL)   # capped -> 520.95
    worst = min(HONEST_PUBLIC_611 * RHO_WORST, LAMBDA_CEIL)       # uncapped -> 492.87
    breakeven = min(HONEST_PUBLIC_611 * RHO_BREAKEVEN, LAMBDA_CEIL)

    # stark #331 tile pin: the salvage tree verifies at M = W*K+1, same Marlin tile as #325's M=32.
    m_salvage = TREE_WIDTH_W * TREE_DEPTH_K + 1
    tiles_salvage = marlin_tiles(m_salvage)
    tiles_anchor = marlin_tiles(M_ANCHOR_325)
    tiles_cliff = marlin_tiles(M_ANCHOR_325 + 1)   # M=33
    denominator_unchanged = tiles_salvage == tiles_anchor
    cliff_does_not_bind = m_salvage <= M_ANCHOR_325 and tiles_cliff > tiles_anchor

    return {
        "source_card": "fern #325 (xk1pghy4)",
        "central_tps_reproduced": central,
        "worstcase_tps_reproduced": worst,
        "breakeven_tps_reproduced": breakeven,
        "central_matches_325_bank": bool(abs(central - ENVELOPE_CENTRAL_325) <= TOL_REPRO),
        "worstcase_matches_325_bank": bool(abs(worst - ENVELOPE_WORST_325) <= TOL_REPRO),
        "uncapped_central": HONEST_PUBLIC_611 * RHO_CENTRAL,
        "uncapped_worst": HONEST_PUBLIC_611 * RHO_WORST,
        "lambda_ceiling": LAMBDA_CEIL,
        "deployed_tps_reproduced": deployed_repro,
        "honest_public_611_reproduced": honest_public_repro,
        "deployed_repro_ok": bool(abs(deployed_repro - BASELINE_TPS) <= TOL_ANCHOR),
        "honest_public_repro_ok": bool(abs(honest_public_repro - HONEST_PUBLIC_611) <= 0.2),
        "tile_pin_331": {
            "source_card": "stark #331 (b48rmwjq)",
            "tree_width_W": TREE_WIDTH_W, "tree_depth_K": TREE_DEPTH_K,
            "m_salvage_tokens": m_salvage, "formula": "M = W*K + 1 = 4*7 + 1 = 29",
            "marlin_thread_m_block": MARLIN_THREAD_M_BLOCK,
            "tiles_at_salvage_M29": tiles_salvage,
            "tiles_at_anchor_M32": tiles_anchor,
            "tiles_at_cliff_M33": tiles_cliff,
            "verify_gemm_denominator_unchanged": bool(denominator_unchanged),
            "m33_cliff_step_pct": CLIFF_STEP_PCT_M33,
            "m33_cliff_does_not_bind": bool(cliff_does_not_bind),
            "note": (
                "E[T]=6.11 top-4 salvage tree verifies at M=W*K+1=29 tokens = {} Marlin "
                "thread_m_blocks = the SAME tile as #325's M=32 anchor (both ceil(M/16)=2), so the "
                "verify-GEMM denominator is UNCHANGED and the M=33 cliff (+{:.2f}% step, the 2->3 "
                "tile jump) does NOT bind. The #325 step -- and hence this ledger -- is valid for "
                "the salvage tree.".format(tiles_salvage, CLIFF_STEP_PCT_M33)),
        },
        "note": (
            "fern #325 joint envelope reproduced from inputs: central = min(622.08*{:.4f}, 520.95) = "
            "{:.2f} (CAPPED at the identity ceiling); worst = min(622.08*{:.4f}, 520.95) = {:.2f} "
            "(uncapped). These are the two numbers the supply/demand corners now deflate."
            .format(RHO_CENTRAL, central, RHO_WORST, worst)),
    }


# --------------------------------------------------------------------------- #
# (2) Supply axis -- G_supply(phi) + the budget arithmetic.
# --------------------------------------------------------------------------- #
def supply_axis() -> dict[str, Any]:
    floor = floor_combined()
    floor_sdpa = SDPA_EXPOSED_US_291 / TOTAL_VERIFY_US_291
    floor_lmhead = LMHEAD_EXPOSED_US_291 / TOTAL_VERIFY_US_291
    floor_norms = NORMS_EXPOSED_US_291 / TOTAL_VERIFY_US_291
    psi_star = BUDGET_LAMBDA1_FRAC / floor          # kept-slack break-even (the PR's "phi*=0.745")
    phi_rec_star = 1.0 - psi_star                   # recovery break-even (0.255)
    ceiling_at_phi0 = revised_ceiling(0.0)          # full-floor corner (469.68)
    ceiling_at_phi1 = revised_ceiling(1.0)          # full-recovery corner (520.95)
    ceiling_at_breakeven = revised_ceiling(phi_rec_star)   # B-edge (floor == budget)

    measured_corner_alive = g_supply(0.0)           # phi=0 -> linear law -> dead
    return {
        "source_card": "denken #327 (kcjlr5ny)",
        "free_variable": "phi = deterministic-SDPA split-KV parallelism RECOVERY fraction in [0,1]",
        "floor_combined": floor,
        "floor_b_sdpa_only": floor_sdpa,
        "floor_a_lm_head_only": floor_lmhead,
        "floor_norms_only": floor_norms,
        "sdpa_bw_util": SDPA_BW_UTIL,
        "lmhead_bw_util": LMHEAD_BW_UTIL,
        "budget_lambda1_frac": BUDGET_LAMBDA1_FRAC,
        "sdpa_alone_busts_budget": bool(floor_sdpa > BUDGET_LAMBDA1_FRAC),
        "phi_star_kept_slack": psi_star,            # 0.745
        "phi_rec_star_recovery": phi_rec_star,      # 0.255
        "ceiling_at_phi0_full_floor": ceiling_at_phi0,      # 469.68
        "ceiling_at_phi1_full_recovery": ceiling_at_phi1,   # 520.95
        "ceiling_at_breakeven_B_edge": ceiling_at_breakeven,
        "measured_corner_phi": 0.0,
        "measured_corner_kept_slack_psi": 1.0,
        "measured_corner_supply_alive": bool(measured_corner_alive),
        "measured_corner_state": "B" if measured_corner_alive else "C",
        "g_supply_rule": "G_supply(phi) = [effective_floor(phi) <= budget] = [phi >= 0.255] "
                         "= [kept-slack psi <= 0.745]",
        "convex_law_revives_327": CONVEX_REVIVES_327,
        "budget_arithmetic": (
            "bf16 SDPA @ {:.1f}% BW alone = {:.3f}% of step (BUSTS the {:.3f}% budget); lm_head @ "
            "{:.1f}% BW = {:.3f}% (cheap); norms = {:.3f}%; COMBINED floor = {:.3f}% -> at phi=0 (no "
            "recovery, denken #327's linear law) the compliant ceiling = 520.95*(1-{:.5f}) = {:.2f} "
            "TPS. Strict-compliant lane (C) DEAD at the measured corner; revives to (B) only if a "
            "deterministic SDPA recovers phi >= {:.1%} of the forgone split-KV parallelism (kept-slack "
            "psi <= {:.1%}). denken #332 (in-flight) fills phi."
            .format(SDPA_BW_UTIL * 100, floor_sdpa * 100, BUDGET_LAMBDA1_FRAC * 100,
                    LMHEAD_BW_UTIL * 100, floor_lmhead * 100, floor_norms * 100, floor * 100,
                    floor, ceiling_at_phi0, phi_rec_star, psi_star)),
    }


# --------------------------------------------------------------------------- #
# (3) Demand axis -- G_demand(Delta_cov) + per-source decomposition.
# --------------------------------------------------------------------------- #
def demand_axis() -> dict[str, Any]:
    prior = coverage_prior()
    delta_cov_star = COV_BAR - prior                 # 0.031
    binding_source = min(COV_PER_SOURCE, key=COV_PER_SOURCE.get)   # mmlu_pro
    weights = {k: OFFICIAL_EVAL_COUNTS[k] / N_EVAL_PROMPTS for k in COV_PER_SOURCE}
    measured_corner_clears = g_demand(0.0)           # prior alone -> miss
    return {
        "source_card": "lawine #330 (hfrscdai)",
        "free_variable": "Delta_cov = aggregate per-source unconditional top-4 coverage LIFT",
        "coverage_prior": prior,
        "build_bar": COV_BAR,
        "delta_cov_star": delta_cov_star,            # 0.031
        "per_source_top4": dict(COV_PER_SOURCE),
        "official_eval_counts": dict(OFFICIAL_EVAL_COUNTS),
        "official_eval_weights": weights,
        "binding_source": binding_source,            # mmlu_pro (reasoning-CoT breadth)
        "binding_source_cov": COV_PER_SOURCE[binding_source],
        "measured_corner_delta_cov": 0.0,
        "measured_corner_demand_clears": bool(measured_corner_clears),
        "measured_corner_state": "clear" if measured_corner_clears else "miss",
        "p_demand_clears_330": P_DEMAND_CLEARS_330,
        "g_demand_rule": "G_demand(Delta_cov) = [0.8903 + Delta_cov >= 0.9213] = [Delta_cov >= 0.031]",
        "note": (
            "Official-eval uncond top-4 prior = 14/57/14-weighted... (counts {} -> weights aime "
            "{:.3f} / gpqa {:.3f} / mmlu_pro {:.3f}) = {:.4f}, MISSING the {:.4f} bar by {:.4f}. The "
            "binding source is mmlu_pro ({:.4f}) -- the reasoning-CoT vocabulary breadth, NOT a "
            "near-absent answer-letter. Revival needs a better head delivering Delta_cov >= {:.4f}; "
            "P(clears) ~ {:.3f} (lawine #330)."
            .format(dict(OFFICIAL_EVAL_COUNTS), weights["aime"], weights["gpqa"], weights["mmlu_pro"],
                    prior, COV_BAR, delta_cov_star, COV_PER_SOURCE["mmlu_pro"], delta_cov_star,
                    P_DEMAND_CLEARS_330)),
    }


# --------------------------------------------------------------------------- #
# (4) Joint gate -- the 2x2 verdict table + the AND-gate.
# --------------------------------------------------------------------------- #
def joint_gate(sup: dict[str, Any], dem: dict[str, Any]) -> dict[str, Any]:
    # representative supply ceilings: C -> measured-corner (phi=0) 469.68; B -> #325 ceiling 520.95.
    ceil_C = revised_ceiling(0.0)
    ceil_B = revised_ceiling(1.0)
    rho_floor = RHO_BREAKEVEN

    def cell(supply_state: str, demand_state: str, rho: float) -> dict[str, Any]:
        clears = demand_state == "clear"
        d_tps = demand_public_tps(clears, rho)
        s_ceil = ceil_B if supply_state == "B" else ceil_C
        tps = min(d_tps, s_ceil)
        go = bool(supply_state == "B" and clears and rho >= rho_floor - TOL and tps >= TARGET - TOL)
        return {
            "supply": supply_state, "demand": demand_state, "rho": rho,
            "demand_public_tps": d_tps, "supply_ceiling": s_ceil,
            "compliant_tps": tps, "go": go,
            "verdict": "GO" if go else "NO-GO",
        }

    # 2x2 evaluated at the headline central rho (the rho-gate column is reported separately).
    table_central = {
        f"{s}_{d}": cell(s, d, RHO_CENTRAL)
        for s in ("C", "B") for d in ("miss", "clear")
    }
    # the (B,clear) cell at worst rho -- where the rho>=0.8038 gate flips it to NO-GO (mirrors #325 YELLOW).
    bclear_worst = cell("B", "clear", RHO_WORST)
    bclear_breakeven = cell("B", "clear", RHO_BREAKEVEN)

    go_cells = [k for k, c in table_central.items() if c["go"]]
    only_bclear_go = go_cells == ["B_clear"]

    # the measured corner today: supply C (phi=0) AND demand miss (Delta_cov=0).
    measured_cell = table_central["C_miss"]

    return {
        "ledger": "compliant_tps = min(demand_public_tps, revised_ceiling(phi)); "
                  "GREEN = G_supply AND G_demand AND rho>=0.8038 AND tps>=500",
        "supply_ceiling_C_phi0": ceil_C,
        "supply_ceiling_B_phi1": ceil_B,
        "rho_priv_floor": rho_floor,
        "rho_central": RHO_CENTRAL,
        "rho_worst": RHO_WORST,
        "table_central_rho": table_central,
        "bclear_at_worst_rho": bclear_worst,
        "bclear_at_breakeven_rho": bclear_breakeven,
        "go_cells_central": go_cells,
        "only_bclear_is_go": bool(only_bclear_go),
        "measured_corner_cell": measured_cell,
        "measured_corner_is_nogo": bool(not measured_cell["go"]),
        "rho_gate_central_passes": bool(RHO_CENTRAL >= rho_floor),
        "rho_gate_worst_passes": bool(RHO_WORST >= rho_floor),
        "note": (
            "2x2 over supply{{C: phi<0.255 i.e. kept-slack psi>=0.745; B: phi>=0.255 i.e. psi<0.745}} "
            "x demand{{miss: Delta_cov<0.031; clear: Delta_cov>=0.031}}, at central rho={:.4f}. ONLY "
            "(B,clear) clears 500 (=> {:.2f} TPS), and only while rho>=0.8038 -- at worst rho={:.4f} "
            "even (B,clear) lands {:.2f} < 500 (NO-GO), exactly #325's YELLOW. The measured corner "
            "(C,miss) = {:.2f} TPS = NO-GO.".format(
                RHO_CENTRAL, table_central["B_clear"]["compliant_tps"], RHO_WORST,
                bclear_worst["compliant_tps"], measured_cell["compliant_tps"])),
    }


# --------------------------------------------------------------------------- #
# (5) Honest bottom-line + binding axis at the measured corner.
# --------------------------------------------------------------------------- #
def bottom_line(sup: dict[str, Any], dem: dict[str, Any], jg: dict[str, Any]) -> dict[str, Any]:
    supply_dead = not sup["measured_corner_supply_alive"]
    demand_miss = not dem["measured_corner_demand_clears"]
    lane_dead_at_measured = supply_dead or demand_miss

    # binding axis -- "which axis is closer to its threshold today". Reported under several
    # normalizations because the two axes have different natural scales; the headline uses the
    # symmetric "fraction of improvement headroom consumed to reach threshold".
    floor = sup["floor_combined"]
    # supply: floor must drop from `floor` to `budget`; best floor = 0.
    supply_gap_headroom = (floor - BUDGET_LAMBDA1_FRAC) / (floor - 0.0)          # 0.255
    supply_gap_rel_to_budget = (floor - BUDGET_LAMBDA1_FRAC) / BUDGET_LAMBDA1_FRAC  # 0.342
    supply_gap_control_recovery = sup["phi_rec_star_recovery"]                   # 0.255 (recover this much)
    # demand: coverage must rise from prior to bar; best coverage = 1.0.
    prior = dem["coverage_prior"]
    demand_gap_headroom = (COV_BAR - prior) / (1.0 - prior)                      # 0.283
    demand_gap_rel_to_bar = (COV_BAR - prior) / COV_BAR                          # 0.034
    demand_gap_control_lift = (COV_BAR - prior) / prior                          # 0.035 (lift coverage this much)

    closer_by_headroom = "supply" if supply_gap_headroom < demand_gap_headroom else "demand"
    closer_by_rel = "supply" if supply_gap_rel_to_budget < demand_gap_rel_to_bar else "demand"

    return {
        "measured_corner_supply_state": "C_dead" if supply_dead else "B_alive",
        "measured_corner_demand_state": "miss" if demand_miss else "clear",
        "lane_dead_at_measured_corner": bool(lane_dead_at_measured),
        "verdict": "DEAD" if lane_dead_at_measured else "ALIVE",
        "handoff": (
            "EAGLE-3 compliant-500 is an AND-gate: it needs BOTH the identity-half supply revival "
            "(deterministic SDPA recovering phi>=0.255 of forgone split-KV parallelism) AND the "
            "demand-half coverage revival (a better head lifting Delta_cov>=0.031). At the measured "
            "corner BOTH fail (supply (C) DEAD under the linear law, demand MISSES the bar by 0.031), "
            "so the lane is DEAD today. If EITHER revival fails, EAGLE-3 compliant-500 stays DEAD and "
            "the lane should hand off to a fresh non-EAGLE-3 direction. The two remaining reads that "
            "could revive it are denken #332 (geometric phi -> supply C<->B) and a better-head "
            "coverage measurement (demand miss<->clear); GREEN requires BOTH to land AND rho>=0.8038."),
        "binding_axis_at_measured_corner": closer_by_headroom,
        "binding_axis_note": (
            "Headline metric = fraction of improvement headroom consumed to reach threshold: supply "
            "{:.3f} (floor 9.841%%->7.332%% of a 9.841%% headroom-to-zero) vs demand {:.3f} (coverage "
            "0.8903->0.9213 of a 0.1097 headroom-to-1.0). The two axes are nearly BALANCED (~25-28%% "
            "of their headroom each); '{}' is marginally closer. Under the relative-to-bar/budget "
            "metric the picture flips -- demand is only {:.1%} under its bar while supply is {:.1%} "
            "over its budget (closer = '{}') -- because the identity ceiling is a stiff function of "
            "the floor. DECISION-relevant: demand needs a +0.031 (+{:.1%}) coverage lift from a better "
            "head; supply needs a 25.5%% parallelism recovery in an UNBUILT deterministic kernel. "
            "Neither is near-free, and the AND-gate requires both."
            .format(supply_gap_headroom, demand_gap_headroom, closer_by_headroom,
                    demand_gap_rel_to_bar, supply_gap_rel_to_budget, closer_by_rel,
                    demand_gap_control_lift)),
        "gaps": {
            "supply_gap_headroom_fraction": supply_gap_headroom,
            "supply_gap_rel_to_budget": supply_gap_rel_to_budget,
            "supply_recovery_needed": supply_gap_control_recovery,
            "demand_gap_headroom_fraction": demand_gap_headroom,
            "demand_gap_rel_to_bar": demand_gap_rel_to_bar,
            "demand_coverage_lift_needed_rel": demand_gap_control_lift,
            "closer_axis_by_headroom": closer_by_headroom,
            "closer_axis_by_relative": closer_by_rel,
        },
    }


# --------------------------------------------------------------------------- #
# p_both_revivals_land (TEST metric) -- P(supply revives) * P(demand clears).
# --------------------------------------------------------------------------- #
def joint_revival_probability(p_supply_revives: float) -> dict[str, Any]:
    p_demand = P_DEMAND_CLEARS_330
    p_both = p_supply_revives * p_demand
    sweep = {f"p_supply_{ps:.2f}": ps * p_demand for ps in (0.0, 0.25, 0.5, 0.75, 1.0)}
    return {
        "p_demand_clears": p_demand,
        "p_supply_revives_input": p_supply_revives,
        "p_both_revivals_land": p_both,
        "p_both_revivals_land_upper_bound": p_demand,   # achieved iff supply revives with certainty
        "p_both_sweep_over_p_supply": sweep,
        "independence_assumed": True,
        "note": (
            "p_both = P(supply revives) * P(demand clears), independent (supply = kernel engineering; "
            "demand = head training). P(demand clears) = {:.4f} (banked lawine #330). P(supply "
            "revives) is UNMEASURED (denken #332 pending); the default uses the measured linear-law "
            "corner phi=0 -> supply DEAD -> P=0 -> p_both = {:.4f}. Even with CERTAIN supply revival "
            "the upper bound is only {:.4f} -- demand is the binding probabilistic gate. The lane is a "
            "<=6%% joint-revival bet today.".format(p_demand, p_both, p_demand)),
    }


# --------------------------------------------------------------------------- #
# Synthesis (pure: no time, no randomness -> deterministic).
# --------------------------------------------------------------------------- #
def synthesize(p_supply_revives: float = 0.0) -> dict[str, Any]:
    env = restate_envelope()
    sup = supply_axis()
    dem = demand_axis()
    jg = joint_gate(sup, dem)
    bl = bottom_line(sup, dem, jg)
    prob = joint_revival_probability(p_supply_revives)

    headline = {
        "verdict": bl["verdict"],
        "lane_dead_at_measured_corner": bl["lane_dead_at_measured_corner"],
        "binding_axis_at_measured_corner": bl["binding_axis_at_measured_corner"],
        "p_both_revivals_land": prob["p_both_revivals_land"],            # TEST
        "p_both_revivals_land_upper_bound": prob["p_both_revivals_land_upper_bound"],
        "envelope_central": env["central_tps_reproduced"],
        "envelope_worst": env["worstcase_tps_reproduced"],
        "measured_corner_tps": jg["measured_corner_cell"]["compliant_tps"],
        "bclear_central_tps": jg["table_central_rho"]["B_clear"]["compliant_tps"],
        "phi_star_kept_slack": sup["phi_star_kept_slack"],
        "phi_rec_star_recovery": sup["phi_rec_star_recovery"],
        "delta_cov_star": dem["delta_cov_star"],
        "ceiling_at_phi0": sup["ceiling_at_phi0_full_floor"],
        "supply_state_today": bl["measured_corner_supply_state"],
        "demand_state_today": bl["measured_corner_demand_state"],
    }
    return {
        "headline": headline,
        "restate_envelope": env,
        "supply_axis": sup,
        "demand_axis": dem,
        "joint_gate": jg,
        "bottom_line": bl,
        "revival_probability": prob,
        "imports": {
            "provenance": (
                "fern #325 xk1pghy4 (envelope central 520.95 / worst 492.87, K_cal 125.268, "
                "LAMBDA_CEIL 520.95, honest_public 622.08, rho central 0.9421 / breakeven 0.8038 / "
                "worst 0.7923) x denken #327 kcjlr5ny (floor 9.841% from #291 exposed_us, budget "
                "7.332%, kept-slack break-even 0.745 / recovery 0.255, ceiling@phi0 469.68) x lawine "
                "#330 hfrscdai (coverage prior 0.8903, bar 0.9213, Delta_cov* 0.031, P(clears) 0.06, "
                "mmlu_pro binding) x stark #331 b48rmwjq (M=W*K+1=29 = 2 Marlin tiles = M=32 anchor, "
                "M=33 cliff +16.98% does not bind). All run-ids in wandb-applied-ai-team/"
                "gemma-challenge-senpai."),
        },
    }


# --------------------------------------------------------------------------- #
# (6) Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def self_test(syn: dict[str, Any]) -> dict[str, Any]:
    env, sup, dem = syn["restate_envelope"], syn["supply_axis"], syn["demand_axis"]
    jg, bl, prob = syn["joint_gate"], syn["bottom_line"], syn["revival_probability"]
    c: dict[str, bool] = {}

    # --- envelope reproduced from #325 inputs (not hardcoded) ---
    c["01_envelope_central_reproduces_520p95"] = bool(
        env["central_matches_325_bank"]
        and abs(env["central_tps_reproduced"] - 520.9527323111674) <= TOL_REPRO)
    c["02_envelope_worst_reproduces_492p87"] = bool(
        env["worstcase_matches_325_bank"]
        and abs(env["worstcase_tps_reproduced"] - 492.865273281899) <= TOL_REPRO)
    c["03_deployed_481_and_honest_622_reproduced"] = bool(
        env["deployed_repro_ok"] and env["honest_public_repro_ok"])
    # central is CAPPED at the identity ceiling; worst is not.
    c["04_central_capped_worst_uncapped"] = bool(
        env["uncapped_central"] > LAMBDA_CEIL + TOL
        and env["uncapped_worst"] < LAMBDA_CEIL - TOL)

    # --- stark #331 tile pin ---
    tp = env["tile_pin_331"]
    c["05_tile_M29_eq_M32_two_blocks"] = bool(
        tp["m_salvage_tokens"] == 29 and tp["tiles_at_salvage_M29"] == 2
        and tp["tiles_at_anchor_M32"] == 2 and tp["verify_gemm_denominator_unchanged"])
    c["06_m33_cliff_does_not_bind"] = bool(
        tp["tiles_at_cliff_M33"] == 3 and tp["m33_cliff_does_not_bind"])

    # --- supply axis ---
    c["07_floor_reproduces_327_9p841"] = bool(
        abs(sup["floor_combined"] - FLOOR_COMBINED_327) <= TOL_REPRO)
    c["08_sdpa_alone_busts_budget"] = bool(
        sup["sdpa_alone_busts_budget"] and sup["floor_b_sdpa_only"] > BUDGET_LAMBDA1_FRAC)
    # phi* = 0.745 (kept-slack) AND recovery* = 0.255 thresholds recovered.
    c["09_phi_star_0p745_recovered"] = bool(
        abs(sup["phi_star_kept_slack"] - 0.745) <= 1e-3
        and abs(sup["phi_star_kept_slack"] - PHI_STAR_KEPT_327) <= TOL_REPRO)
    c["10_recovery_star_0p255_recovered"] = bool(
        abs(sup["phi_rec_star_recovery"] - 0.255) <= 1e-3
        and abs(sup["phi_rec_star_recovery"] - RECOVERY_NEEDED_327) <= TOL_REPRO)
    # compliant ceiling at phi=0 (full floor) = 469.68; at phi=1 = 520.95 (round-trip).
    c["11_ceiling_at_phi0_469p68"] = bool(
        abs(sup["ceiling_at_phi0_full_floor"] - CEILING_AT_FLOOR_327) <= TOL_REPRO
        and abs(sup["ceiling_at_phi1_full_recovery"] - LAMBDA_CEIL) <= TOL_REPRO)

    # --- demand axis ---
    c["12_demand_prior_reproduces_330_0p8903"] = bool(
        abs(dem["coverage_prior"] - COV_PRIOR_330) <= TOL_REPRO)
    c["13_delta_cov_star_0p031"] = bool(abs(dem["delta_cov_star"] - 0.031) <= 1e-3)
    c["14_binding_source_is_mmlu_pro"] = bool(dem["binding_source"] == "mmlu_pro")

    # --- gate monotonicity ---
    sup_seq = [g_supply(p) for p in (0.0, 0.20, 0.255, 0.30, 0.60, 1.0)]
    c["15_g_supply_monotone_in_phi"] = bool(
        all((not sup_seq[i]) or sup_seq[i + 1] for i in range(len(sup_seq) - 1))
        and sup_seq[0] is False and sup_seq[-1] is True)
    dem_seq = [g_demand(d) for d in (0.0, 0.01, 0.031, 0.05, 0.10)]
    c["16_g_demand_monotone_in_dcov"] = bool(
        all((not dem_seq[i]) or dem_seq[i + 1] for i in range(len(dem_seq) - 1))
        and dem_seq[0] is False and dem_seq[-1] is True)

    # --- measured corner -> NO-GO (the decisive check) ---
    c["17_measured_corner_phi0_dcov0_nogo"] = bool(
        (not sup["measured_corner_supply_alive"]) and (not dem["measured_corner_demand_clears"])
        and jg["measured_corner_is_nogo"] and bl["lane_dead_at_measured_corner"])

    # --- 2x2 internally consistent: ONLY (B,clear) is GO at central rho ---
    c["18_only_bclear_is_go"] = bool(jg["only_bclear_is_go"])
    tbl = jg["table_central_rho"]
    c["19_table_consistent_three_nogo"] = bool(
        (not tbl["C_miss"]["go"]) and (not tbl["C_clear"]["go"])
        and (not tbl["B_miss"]["go"]) and tbl["B_clear"]["go"]
        and tbl["B_clear"]["compliant_tps"] >= TARGET - TOL)

    # --- rho_priv floor applied: central passes the gate, worst fails it (mirrors #325 YELLOW) ---
    c["20_rho_floor_applied"] = bool(
        abs(jg["rho_priv_floor"] - 0.8037539966988988) <= TOL_REPRO
        and jg["rho_gate_central_passes"] and (not jg["rho_gate_worst_passes"])
        and (not jg["bclear_at_worst_rho"]["go"]))

    # --- determinism: two pure syntheses are identical ---
    a = json.dumps(synthesize(prob["p_supply_revives_input"]), sort_keys=True)
    b = json.dumps(synthesize(prob["p_supply_revives_input"]), sort_keys=True)
    c["21_determinism_two_runs_identical"] = bool(a == b)

    # --- p_both bounded by demand's P(clears) ---
    c["22_p_both_bounded_by_demand"] = bool(
        0.0 <= prob["p_both_revivals_land"] <= prob["p_both_revivals_land_upper_bound"] + TOL
        and abs(prob["p_both_revivals_land_upper_bound"] - P_DEMAND_CLEARS_330) <= TOL_REPRO)

    # --- NaN-clean (filled by caller) ---
    c["23_nan_clean"] = True

    gate = all(bool(v) for v in c.values())
    return {"joint_feasibility_andgate_self_test_passes": gate, "checks": c}


# --------------------------------------------------------------------------- #
# NaN-clean walk.
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


# --------------------------------------------------------------------------- #
# Console report.
# --------------------------------------------------------------------------- #
def _print_report(syn: dict, st: dict) -> None:
    env, sup, dem = syn["restate_envelope"], syn["supply_axis"], syn["demand_axis"]
    jg, bl, prob = syn["joint_gate"], syn["bottom_line"], syn["revival_probability"]
    print("\n" + "=" * 100, flush=True)
    print("JOINT COMPLIANT-500 AND-GATE (PR #335, fern) -- fold supply floor #327 + demand miss #330",
          flush=True)
    print("=" * 100, flush=True)
    print("  (1) STANDING #325 ENVELOPE (reproduced from inputs):", flush=True)
    print(f"      central = {env['central_tps_reproduced']:.2f} (CAPPED at {LAMBDA_CEIL:.2f})   "
          f"worst = {env['worstcase_tps_reproduced']:.2f}", flush=True)
    tp = env["tile_pin_331"]
    print(f"      stark #331 tile: M=W*K+1={tp['m_salvage_tokens']} -> {tp['tiles_at_salvage_M29']} "
          f"Marlin tiles == M=32 anchor ({tp['tiles_at_anchor_M32']}); M=33 -> "
          f"{tp['tiles_at_cliff_M33']} tiles (+{tp['m33_cliff_step_pct']:.2f}% cliff) does NOT bind",
          flush=True)
    print("-" * 100, flush=True)
    print("  (2) SUPPLY AXIS (denken #327; phi = SDPA parallelism recovery fraction):", flush=True)
    print(f"      floor: SDPA {sup['floor_b_sdpa_only']*100:.3f}% + lm_head "
          f"{sup['floor_a_lm_head_only']*100:.3f}% + norms {sup['floor_norms_only']*100:.3f}% = "
          f"{sup['floor_combined']*100:.3f}%  vs budget {sup['budget_lambda1_frac']*100:.3f}%",
          flush=True)
    print(f"      break-even: recovery phi*={sup['phi_rec_star_recovery']:.3f} (kept-slack "
          f"psi*={sup['phi_star_kept_slack']:.3f}); ceiling phi=0 -> "
          f"{sup['ceiling_at_phi0_full_floor']:.2f}, phi=1 -> "
          f"{sup['ceiling_at_phi1_full_recovery']:.2f}", flush=True)
    print(f"      measured corner phi=0 -> supply state {sup['measured_corner_state']} (DEAD under "
          f"linear law)", flush=True)
    print("-" * 100, flush=True)
    print("  (3) DEMAND AXIS (lawine #330; Delta_cov = coverage lift):", flush=True)
    print(f"      prior {dem['coverage_prior']:.4f} (aime {dem['per_source_top4']['aime']:.4f} / "
          f"gpqa {dem['per_source_top4']['gpqa']:.4f} / mmlu_pro "
          f"{dem['per_source_top4']['mmlu_pro']:.4f}, w 14/57/57) vs bar {dem['build_bar']:.4f}",
          flush=True)
    print(f"      Delta_cov* = {dem['delta_cov_star']:.4f}; binding source = {dem['binding_source']}; "
          f"measured corner -> {dem['measured_corner_state']} (P(clears)~"
          f"{dem['p_demand_clears_330']:.3f})", flush=True)
    print("-" * 100, flush=True)
    print(f"  (4) 2x2 VERDICT TABLE (central rho={jg['rho_central']:.4f}):", flush=True)
    tbl = jg["table_central_rho"]
    print(f"      {'':14s} {'demand=miss':>22s} {'demand=clear':>22s}", flush=True)
    for s in ("C", "B"):
        miss, clear = tbl[f"{s}_miss"], tbl[f"{s}_clear"]
        slabel = "supply=C(dead)" if s == "C" else "supply=B(alive)"
        print(f"      {slabel:14s} {miss['compliant_tps']:7.2f} [{miss['verdict']:>5s}]      "
              f"{clear['compliant_tps']:7.2f} [{clear['verdict']:>5s}]", flush=True)
    print(f"      (B,clear) at worst rho={jg['rho_worst']:.4f} -> "
          f"{jg['bclear_at_worst_rho']['compliant_tps']:.2f} "
          f"[{jg['bclear_at_worst_rho']['verdict']}] (rho<0.8038 gate) -- #325 YELLOW", flush=True)
    print("-" * 100, flush=True)
    print(f"  (5) BOTTOM LINE: lane is {bl['verdict']} at the measured corner "
          f"(supply {bl['measured_corner_supply_state']}, demand {bl['measured_corner_demand_state']})",
          flush=True)
    print(f"      binding_axis_at_measured_corner = {bl['binding_axis_at_measured_corner']}  "
          f"(supply headroom {bl['gaps']['supply_gap_headroom_fraction']:.3f} vs demand "
          f"{bl['gaps']['demand_gap_headroom_fraction']:.3f})", flush=True)
    print(f"      p_both_revivals_land = {prob['p_both_revivals_land']:.4f} (upper bound "
          f"{prob['p_both_revivals_land_upper_bound']:.4f}; demand-bound)", flush=True)
    print("-" * 100, flush=True)
    print(f"  PRIMARY joint_feasibility_andgate_self_test_passes = "
          f"{st['joint_feasibility_andgate_self_test_passes']}", flush=True)
    for k, v in st["checks"].items():
        print(f"        - {k}: {v}", flush=True)
    print("=" * 100, flush=True)
    print(f"\n  HAND-OFF: {bl['handoff']}\n", flush=True)


# --------------------------------------------------------------------------- #
# W&B logging.
# --------------------------------------------------------------------------- #
def _maybe_log_wandb(args, payload: dict) -> str | None:
    if getattr(args, "no_wandb", False) or not getattr(args, "wandb_name", None):
        return None
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[joint-andgate] wandb logging unavailable: {exc}", flush=True)
        return None

    syn, st = payload["synthesis"], payload["self_test"]
    env, sup, dem = syn["restate_envelope"], syn["supply_axis"], syn["demand_axis"]
    jg, bl, prob = syn["joint_gate"], syn["bottom_line"], syn["revival_probability"]
    h = syn["headline"]
    run = init_wandb_run(
        job_type="eagle3-joint-feasibility-andgate",
        agent="fern",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["eagle3-joint-feasibility-andgate", "issue-192", "eagle3", "and-gate",
              "supply-floor-327", "demand-coverage-330", "identity-ceiling", "validity-gate",
              "zero-tps", "bank-the-analysis"],
        config={
            "pr": 335, "analysis_only": True,
            "K_cal": K_CAL, "lambda_ceiling": LAMBDA_CEIL, "honest_public_611": HONEST_PUBLIC_611,
            "budget_lambda1_frac": BUDGET_LAMBDA1_FRAC, "cov_bar": COV_BAR,
            "rho_central": RHO_CENTRAL, "rho_breakeven": RHO_BREAKEVEN, "rho_worst": RHO_WORST,
            "p_supply_revives_input": prob["p_supply_revives_input"],
            "baseline_tps": BASELINE_TPS, "target": TARGET, "wandb_group": args.wandb_group,
            "source_runs": "fern#325 xk1pghy4, denken#327 kcjlr5ny, lawine#330 hfrscdai, "
                           "stark#331 b48rmwjq",
        },
    )
    if run is None:
        print("[joint-andgate] wandb: no run (no WANDB_API_KEY/mode) -- skipping", flush=True)
        return None

    summary: dict[str, Any] = {
        "joint_feasibility_andgate_self_test_passes":
            int(bool(st["joint_feasibility_andgate_self_test_passes"])),           # PRIMARY
        "p_both_revivals_land": prob["p_both_revivals_land"],                       # TEST
        "p_both_revivals_land_upper_bound": prob["p_both_revivals_land_upper_bound"],
        "p_demand_clears": prob["p_demand_clears"],
        "lane_dead_at_measured_corner": int(bool(bl["lane_dead_at_measured_corner"])),
        "verdict_dead": int(bl["verdict"] == "DEAD"),
        "binding_axis_supply": int(bl["binding_axis_at_measured_corner"] == "supply"),
        "binding_axis_demand": int(bl["binding_axis_at_measured_corner"] == "demand"),
        # envelope
        "envelope_central": env["central_tps_reproduced"],
        "envelope_worst": env["worstcase_tps_reproduced"],
        "lambda_ceiling": LAMBDA_CEIL,
        # supply
        "floor_combined": sup["floor_combined"],
        "floor_b_sdpa_only": sup["floor_b_sdpa_only"],
        "floor_a_lm_head_only": sup["floor_a_lm_head_only"],
        "budget_lambda1_frac": sup["budget_lambda1_frac"],
        "phi_star_kept_slack": sup["phi_star_kept_slack"],
        "phi_rec_star_recovery": sup["phi_rec_star_recovery"],
        "ceiling_at_phi0_full_floor": sup["ceiling_at_phi0_full_floor"],
        "ceiling_at_breakeven_B_edge": sup["ceiling_at_breakeven_B_edge"],
        "supply_alive_at_measured_corner": int(bool(sup["measured_corner_supply_alive"])),
        # demand
        "coverage_prior": dem["coverage_prior"],
        "build_bar": dem["build_bar"],
        "delta_cov_star": dem["delta_cov_star"],
        "demand_clears_at_measured_corner": int(bool(dem["measured_corner_demand_clears"])),
        # gate
        "measured_corner_tps": jg["measured_corner_cell"]["compliant_tps"],
        "bclear_central_tps": jg["table_central_rho"]["B_clear"]["compliant_tps"],
        "bclear_worst_tps": jg["bclear_at_worst_rho"]["compliant_tps"],
        "only_bclear_is_go": int(bool(jg["only_bclear_is_go"])),
        "rho_gate_central_passes": int(bool(jg["rho_gate_central_passes"])),
        "rho_gate_worst_passes": int(bool(jg["rho_gate_worst_passes"])),
        # binding-axis gaps
        "supply_gap_headroom_fraction": bl["gaps"]["supply_gap_headroom_fraction"],
        "demand_gap_headroom_fraction": bl["gaps"]["demand_gap_headroom_fraction"],
        "supply_gap_rel_to_budget": bl["gaps"]["supply_gap_rel_to_budget"],
        "demand_gap_rel_to_bar": bl["gaps"]["demand_gap_rel_to_bar"],
        # tile pin
        "tiles_at_salvage_M29": env["tile_pin_331"]["tiles_at_salvage_M29"],
        "tiles_at_anchor_M32": env["tile_pin_331"]["tiles_at_anchor_M32"],
        "m33_cliff_does_not_bind": int(bool(env["tile_pin_331"]["m33_cliff_does_not_bind"])),
        # hygiene
        "nan_clean": int(bool(payload["nan_clean"])), "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v)) for k, v in st["checks"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="eagle3_joint_feasibility_andgate_result",
                      artifact_type="validity", data=payload)
    rid = getattr(run, "id", None)
    print(f"[joint-andgate] wandb run: {getattr(run, 'url', rid)}", flush=True)
    finish_wandb(run)
    return rid


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--p-supply-revives", type=float, default=0.0,
                    help="P(supply revives) for p_both (default 0.0 = measured linear-law corner; "
                         "denken #332 pending)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="eagle3-joint-feasibility-andgate")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    syn = synthesize(p_supply_revives=args.p_supply_revives)
    st = self_test(syn)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 335, "agent": "fern",
        "kind": "eagle3-joint-feasibility-andgate", "analysis_only": True,
        "synthesis": syn, "self_test": st,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _nan_paths(payload)
    payload["nan_clean"] = not nan_paths
    st["checks"]["23_nan_clean"] = not nan_paths
    st["joint_feasibility_andgate_self_test_passes"] = all(bool(v) for v in st["checks"].values())
    syn["headline"]["joint_feasibility_andgate_self_test_passes"] = st[
        "joint_feasibility_andgate_self_test_passes"]
    if nan_paths:
        print(f"[joint-andgate] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn, st)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eagle3_joint_feasibility_andgate_results.json"

    rid = _maybe_log_wandb(args, payload)
    payload["wandb_run_id"] = rid
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[joint-andgate] wrote {out_path}  (wandb run {rid})", flush=True)

    print(f"  PRIMARY joint_feasibility_andgate_self_test_passes = "
          f"{st['joint_feasibility_andgate_self_test_passes']}", flush=True)
    print(f"  TEST p_both_revivals_land = {syn['headline']['p_both_revivals_land']:.4f}", flush=True)
    print(f"  REPORT binding_axis_at_measured_corner = "
          f"{syn['headline']['binding_axis_at_measured_corner']}", flush=True)

    if args.self_test:
        ok = st["joint_feasibility_andgate_self_test_passes"] and payload["nan_clean"]
        print(f"[joint-andgate] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
