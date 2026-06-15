#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #316 (student lawine) -- YELLOW-zone E[T]/TPS: what fusion rank-coverage still clears E[T]=6.11?

WHAT THIS CARD DOES (0-GPU, 0-TPS, no served-file change, no HF Job)
--------------------------------------------------------------------
lawine #313 (sw492nih, MERGED) collapsed #309's YELLOW to ONE measurable bar: the GREEN/YELLOW line at
fusion frac_true_beyond_top4 = 0.3935 (cov4* = 0.6065), with the deployed LINEAR spine at frac=0.3468
(GREEN, +0.0467 frac-headroom). #313's threshold lives on the RANK-COVERAGE axis. The GO/NO-GO,
however, is decided in TPS. This card BANKS the transfer function that converts the rank-coverage
threshold into the E[T]/TPS units the decision actually rides on -- #313's own follow-up #3.

It does THREE things, all exact arithmetic on banked anchors (re-derives none of them):

  1. REPRODUCE #313's salvage operator + GREEN/RED threshold to <= 1e-6 against the on-disk sw492nih
     artifact: c1_eff(a1)=a1+(1-a1)*cov_W, inverse a1_req=(T-cov_W)/(1-cov_W) (T=0.9213011665), and the
     W-invariant threshold cov4*=(T-d)/(1-d) / frac*=1-cov4* (GREEN d=0.80 -> cov4*=0.6065/frac*=0.3935;
     linear cov4=0.6532/frac=0.3468; RED only at cov4=0/frac=1).

  2. MAP THE YELLOW BAND TO E[T] AND TPS. The REALISTIC reading of "a slightly-worse-than-linear fusion
     draft": hold the raw draft a1 at the deployed cliff (raw a1 == today's linear spine), let the M=8
     tree salvage position-1 via cov4 = 1 - frac, and hold the deep spine a_{k>=2} at the BANKED
     DEPLOYED conditional ladder (lawine #300). Sweep frac across [linear 0.3468 -> GREEN edge 0.3935 ->
     toward 1.0 (RED)]; for each: cov4 = 1-frac, the salvaged effective c1_eff, the chain
     E[T] = 1 + sum_d prod_{j<=d} a_j, and the deployed-loopgraph TPS = K_cal*E[T] (== K_cal*(E[T]/step)
     *tau at the served step, tau/step ~= 1). At cov4=0 (frac=1) the chain reproduces the deployed
     operating point (E[T]~=3.85, TPS~=482, == the deployed 481.53 to 0.2%) -- the sanity anchor. The
     curve is MONOTONE: more rank-coverage (lower frac) -> higher E[T]/TPS. KEY WALL: with the deployed
     deep spine the chain CAPS at E[T]=4.910 (a1_eff=1.0, cov4=1), 1.20 BELOW the 6.11 build target --
     reproducing denken #304 ("kanna #289 spec yields 4.9196, falls 1.19 SHORT of 6.11"). Position-1
     rank-coverage salvage ALONE cannot reach the build target; the deep spine is the binding wall.

  3. SOLVE THE BUILD-TARGET BAR IN RANK-COVERAGE UNITS. To reach the step-corrected build target
     E[T]=6.11 (wirbel #295, central 6.1112) the deep spine MUST be lifted to the #297 build-uniform
     target (effective per-position == T). Under that build configuration, position-1 salvaged, the
     MAXIMUM fusion frac (worst rank-coverage) that still clears 6.11 is, as a function of the fusion
     draft's RAW a1:

        max_frac_clearing_611(raw_a1) = (1 - T) / (1 - raw_a1)            [solve c1_eff(raw_a1,cov4)=T]

     This single transfer function reproduces BOTH banked anchors exactly: raw_a1 = 0.7731 (#309's W4
     demand) -> 0.3468 == the LINEAR spine frac; raw_a1 = 0.80 (#313's GREEN cap) -> 0.3935 == #313's
     GREEN edge; raw_a1 = T -> 1.0 == the RED edge. The HEADLINE bar at the conservative deployed raw a1
     (0.72925, a fusion draft no better than linear at rank-1) is 0.2907 (cov4=0.7093) -- TIGHTER than
     the linear spine's 0.3468: a deployed-quality fusion draft needs BETTER rank-2+ coverage than the
     linear spine (cov4 0.6532 -> 0.7093, +5.6pp) OR a raw a1 >= 0.7731 to clear the build target.

  4. HONEST CAVEAT. The fusion cov_W is UNMEASURED (no fusion checkpoint). This maps the
     threshold -> E[T] -> TPS transfer function, parameterized by the still-to-be-measured fusion frac.
     The deep-spine a_{k>=2} are held at banked DEPLOYED values for the realistic curve (step 2) and at
     the build-uniform target for the build bar (step 3) -- a fusion draft may shift the deep spine too
     (flagged second-order axis). The deployed-deep-spine WALL (E[T] caps 4.910) is the realistic
     finding; the build bar assumes the deep spine is lifted.

LOCAL CPU-only analytic card. No GPU / vLLM / model forward / training / HF Job / submission /
served-file change. NOT a launch. BASELINE stays 481.53 (0 TPS). Greedy/PPL untouched.

PRIMARY metric  yellow_zone_et_self_test_passes
TEST    metrics max_frac_beyond_top4_clearing_611 (float)  +  linear_spine_clears_611_under_salvage (bool)

Reproduce:
    cd target/ && .venv/bin/python \\
        research/validity/eagle3_yellow_zone_et/eagle3_yellow_zone_et.py \\
        --self-test --wandb_group eagle3-yellow-zone-et --wandb_name lawine/eagle3-yellow-zone-et
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --------------------------------------------------------------------------- #
# Imported fleet anchors (cited EXACTLY, UNCHANGED; this card re-derives none).
# The self-test re-loads the on-disk banked artifacts and cross-checks these
# literals to <= 1e-6 so there is no silent drift.
# --------------------------------------------------------------------------- #
# denken #304 (dtf1ouml): the no-salvage hard line == the EFFECTIVE per-position target T for E[T]=6.11.
T_EFFECTIVE = 0.9213011665456927        # effective per-position acceptance for E[T]=6.11 (no tree salvage)
A1_DEPLOYED = 0.72925                    # deployed raw a1 cliff (#304); the conservative fusion raw-a1 anchor

# wirbel #295 (c334qaqu): the step-corrected build target E[T] (central regime point), == 500-by-construction.
ET_TARGET_611 = 6.1112149873699195

# lawine #300 (8t5q6sr0): deployed-effective per-position conditional ladder a_k (the deep spine for step 2).
A_K = [
    0.7292532942898975, 0.759556697719242, 0.7929794882639035, 0.8228,
    0.8348727920920435, 0.8357919254658385, 0.8464932652113331,
]
K_SPEC = 7                              # num_speculative_tokens (manifest); len(A_K) == K_SPEC
RAW_A1_DEPLOYED = A_K[0]                 # exact deployed-ladder rank-1 conditional == "raw a1 held deployed"

# wirbel #79 (z6wi4z4v): MEASURED rank-coverage on the deployed LINEAR spine (16,524 records, align_bad=0).
COV_W = {2: 0.4165047789261015, 3: 0.5714507731758489, 4: 0.6531976066516435}
FRAC_TRUE_BEYOND_TOP4 = 0.3468023933483565   # 1 - cov4 (irreducible width-4 miss mass, linear spine)
PRIMARY_W = 4

# lawine #309 (7tkn4d9x): the W4 raw-a1 demand at the LINEAR spine cov4 (the build-relevant raw a1 anchor).
A1_REQ_309_W4 = 0.7730729805683441

# lawine #313 (sw492nih): banked GREEN/RED threshold on the rank-coverage axis (this card reproduces it).
TRAINABLE_BAND_EDGE = 0.80             # GREEN edge on the RAW-a1 demand axis (#313 d=0.80)
DEPLOYED_PLUS_10PCT = A1_DEPLOYED * 1.10   # 0.802175 secondary GREEN edge (#309 verdict anchor)
COV4_STAR_GREEN_313 = 0.6065058327284635   # banked #313 cov4* at d=0.80
FRAC_STAR_GREEN_313 = 0.3934941672715365   # banked #313 frac* at d=0.80

# central composition convention (kanna #269 / #52): deployed-loopgraph TPS = K_cal * E[T].
K_CAL = 125.268                        # official = K_cal * E[T]  (125.268 * 3.844 == 481.53)
TAU = 1.218                            # served efficiency factor (central convention)
SERVED_STEP_US = 1218.2               # served decode step (us); tau/step ~= 1 at the served step
DEPLOYED_ET_CONVENTION = 3.844         # deployed E[T] central convention (#52/#269)
BASELINE_TPS = 481.53                  # current best summary.json:tps (unchanged; 0-TPS analytic)

# Banked artifacts (committed; read-only validation targets for the self-test).
PR313_RESULTS = (REPO_ROOT / "research" / "validity" / "eagle3_fusion_rankcov_probe"
                 / "eagle3_fusion_rankcov_probe_results.json")
PR309_RESULTS = (REPO_ROOT / "research" / "validity" / "eagle3_tree_salvage_a1"
                 / "eagle3_tree_salvage_a1_results.json")
PR295_RESULTS = (REPO_ROOT / "research" / "validity" / "eagle3_step_profile"
                 / "eagle3_step_profile_results.json")

# Report grid on the frac axis: linear (lower edge), 3 YELLOW-interior points, GREEN edge (upper edge),
# then toward RED (and frac=1 == deployed sanity anchor).
FRAC_GRID = [0.0, 0.20, FRAC_TRUE_BEYOND_TOP4, 0.36, 0.37, 0.38, FRAC_STAR_GREEN_313, 0.50, 0.70, 1.0]
RAW_A1_ANCHORS = [RAW_A1_DEPLOYED, 0.73, A1_REQ_309_W4, TRAINABLE_BAND_EDGE, DEPLOYED_PLUS_10PCT, T_EFFECTIVE]

TOL = 1e-9
TOL_REPRO = 1e-6


# --------------------------------------------------------------------------- #
# #313 / #309 salvage operator + its two inversions (independently re-implemented).
# --------------------------------------------------------------------------- #
def tree_recovered(base: float, cov: float) -> float:
    """Salvage operator: effective position-1 acceptance after a width-W verify tree.

    true token accepted if rank-1 (prob base) OR in the rank-2..W branch on a rank-1 miss
    (prob (1-base)*cov). Clipped to a valid acceptance [0,1].
    """
    return min(1.0, max(0.0, base + (1.0 - base) * cov))


def a1_draft_for_effective(a_eff_target: float, cov: float) -> float:
    """Invert the salvage in a1: raw draft a1 so that tree_recovered(a1, cov) == a_eff_target.

    a1 = (a_eff_target - cov)/(1 - cov); cov->0 returns a_eff_target (no salvage, reproduces #304).
    """
    if cov >= 1.0 - TOL:
        return 0.0
    return min(1.0, max(0.0, (a_eff_target - cov) / (1.0 - cov)))


def cov_for_demand(a1_demand: float, a_eff_target: float = T_EFFECTIVE) -> float:
    """Decision threshold: the cov_W at which the salvaged raw-a1 demand equals ``a1_demand``.

    cov* = (a_eff_target - a1_demand)/(1 - a1_demand). W-invariant on the cov axis.
    """
    if a1_demand >= 1.0 - TOL:
        return 0.0
    return (a_eff_target - a1_demand) / (1.0 - a1_demand)


# --------------------------------------------------------------------------- #
# E[T] chain machinery (reused from lawine #300/#309, UNCHANGED math).
# --------------------------------------------------------------------------- #
def survival(cond: list[float]) -> list[float]:
    """committed-survival S_d = prod_{j<=d} a_j for d=1..K (S_0=1 implicit)."""
    out, acc = [], 1.0
    for p in cond:
        acc *= float(p)
        out.append(acc)
    return out


def et_from_cond(cond: list[float]) -> float:
    """E[T] = 1 + sum_d S_d (1 base token + expected accepted draft tokens)."""
    return 1.0 + sum(survival(cond))


def tps_deployed_loopgraph(et: float) -> float:
    """deployed-loopgraph TPS = K_cal * E[T] == K_cal*(E[T]/step)*tau at the served step (tau/step~=1)."""
    return K_CAL * et


def max_frac_clearing_611(raw_a1: float, t_eff: float = T_EFFECTIVE) -> float:
    """Worst (max) fusion frac_true_beyond_top4 at which a verify tree still salvages raw_a1 up to the
    effective build target t_eff (so the build-uniform chain reaches E[T]=6.11). Solve
    tree_recovered(raw_a1, cov4) == t_eff -> cov4 = (t_eff-raw_a1)/(1-raw_a1) -> frac = 1 - cov4 =
    (1 - t_eff)/(1 - raw_a1). Clipped to [0,1]."""
    if raw_a1 >= 1.0 - TOL:
        return 0.0
    return min(1.0, max(0.0, (1.0 - t_eff) / (1.0 - raw_a1)))


# --------------------------------------------------------------------------- #
# Load banked artifacts (read-only) for the <= 1e-6 cross-checks.
# --------------------------------------------------------------------------- #
def load_banked() -> dict[str, Any]:
    def _load(p: Path) -> dict[str, Any]:
        return json.loads(p.read_text()) if p.exists() else {}
    return {"pr313": _load(PR313_RESULTS), "pr309": _load(PR309_RESULTS), "pr295": _load(PR295_RESULTS)}


# --------------------------------------------------------------------------- #
# Synthesis (steps 1-4).
# --------------------------------------------------------------------------- #
def synthesize(banked: dict[str, Any]) -> dict[str, Any]:
    d313 = banked.get("pr313", {})
    d309 = banked.get("pr309", {})
    d295 = banked.get("pr295", {})

    # ---- STEP 1: reproduce #313's operator + GREEN/RED threshold to <= 1e-6. ---- #
    cov_star_green = cov_for_demand(TRAINABLE_BAND_EDGE)
    frac_star_green = 1.0 - cov_star_green
    cov_star_red = cov_for_demand(T_EFFECTIVE)
    frac_star_red = 1.0 - cov_star_red
    cov_star_green_p10 = cov_for_demand(DEPLOYED_PLUS_10PCT)
    frac_star_green_p10 = 1.0 - cov_star_green_p10

    # cross-check vs the banked sw492nih artifact (#313 step2_decision_threshold).
    s2_313 = (d313.get("synthesis", {}) or {}).get("step2_decision_threshold", {}) or {}
    banked_cov_green = s2_313.get("cov_W_star_green")
    banked_frac_green = s2_313.get("frac_true_beyond_topW_star_green")
    cov_green_abs_diff = abs(cov_star_green - float(banked_cov_green)) if banked_cov_green is not None \
        else abs(cov_star_green - COV4_STAR_GREEN_313)
    frac_green_abs_diff = abs(frac_star_green - float(banked_frac_green)) if banked_frac_green is not None \
        else abs(frac_star_green - FRAC_STAR_GREEN_313)
    # the deployed linear spine's salvaged raw-a1 demand (== #309's W4 demand).
    demand_at_linear = a1_draft_for_effective(T_EFFECTIVE, COV_W[PRIMARY_W])
    demand_linear_abs_diff_vs_309 = abs(demand_at_linear - A1_REQ_309_W4)
    reproduces_313 = bool(cov_green_abs_diff <= TOL_REPRO and frac_green_abs_diff <= TOL_REPRO
                          and demand_linear_abs_diff_vs_309 <= TOL_REPRO
                          and abs(cov_star_red) <= TOL and abs(frac_star_red - 1.0) <= TOL)

    step1 = {
        "effective_target_T": T_EFFECTIVE,
        "cov4_star_green": cov_star_green,
        "frac_star_green": frac_star_green,
        "cov4_star_red": cov_star_red,
        "frac_star_red": frac_star_red,
        "secondary_green_edge_deployed_plus_10pct": DEPLOYED_PLUS_10PCT,
        "cov4_star_green_deployed_plus_10pct": cov_star_green_p10,
        "frac_star_green_deployed_plus_10pct": frac_star_green_p10,
        "linear_spine_cov4": COV_W[PRIMARY_W],
        "linear_spine_frac": FRAC_TRUE_BEYOND_TOP4,
        "salvaged_raw_a1_demand_at_linear": demand_at_linear,
        "cov_green_abs_diff_vs_313": cov_green_abs_diff,
        "frac_green_abs_diff_vs_313": frac_green_abs_diff,
        "demand_linear_abs_diff_vs_309": demand_linear_abs_diff_vs_309,
        "reproduces_313_operator_and_threshold": reproduces_313,
        "note": ("#313 operator c1_eff=a1+(1-a1)*cov_W and threshold cov4*=(T-d)/(1-d), frac*=1-cov4* "
                 "reproduced to <= 1e-6: GREEN d=0.80 -> cov4*={:.4f}/frac*={:.4f}; RED d=T -> "
                 "cov*=0/frac*=1; the linear spine salvaged demand {:.4f} == #309's W4 {:.4f}."
                 .format(cov_star_green, frac_star_green, demand_at_linear, A1_REQ_309_W4)),
    }

    # ---- STEP 2: map the YELLOW band to E[T] and TPS (realistic deployed-deep-spine chain). ---- #
    deep_spine = A_K[1:]                                   # a_{k>=2}, banked deployed conditional ladder
    # deep-spine survival multiplier (sum of survival from position-1's perspective): caps E[T] at a1_eff=1.
    deep_mult = sum(survival([1.0] + deep_spine))
    et_cap_deployed_deep = 1.0 + 1.0 * deep_mult           # ceiling at perfect position-1 (a1_eff=1, cov4=1)
    tps_cap_deployed_deep = tps_deployed_loopgraph(et_cap_deployed_deep)

    def chain_et_tps(frac: float) -> dict[str, float]:
        cov4 = 1.0 - frac
        a1_eff = tree_recovered(RAW_A1_DEPLOYED, cov4)
        chain = [a1_eff] + deep_spine
        et = et_from_cond(chain)
        return {"frac": frac, "cov4": cov4, "a1_eff": a1_eff, "et": et, "tps": tps_deployed_loopgraph(et)}

    curve = [chain_et_tps(f) for f in FRAC_GRID]
    # named report points.
    pt_linear = chain_et_tps(FRAC_TRUE_BEYOND_TOP4)
    pt_green = chain_et_tps(FRAC_STAR_GREEN_313)
    pt_deployed_anchor = chain_et_tps(1.0)                 # cov4=0 -> should reproduce the deployed point
    # deployed anchor reconciliation: chain at cov4=0 vs the deployed ladder / central convention.
    et_deployed_ladder = et_from_cond(A_K)
    deployed_anchor_et_abs_diff = abs(pt_deployed_anchor["et"] - et_deployed_ladder)
    deployed_anchor_tps = pt_deployed_anchor["tps"]
    deployed_conv_tps = tps_deployed_loopgraph(DEPLOYED_ET_CONVENTION)   # == 481.53
    deployed_conv_tps_abs_diff = abs(deployed_conv_tps - BASELINE_TPS)

    # monotonicity: as frac DECREASES (cov4 increases), E[T] and TPS strictly INCREASE.
    asc = sorted(curve, key=lambda r: r["frac"], reverse=True)   # frac high -> low
    et_monotone = all(asc[i]["et"] < asc[i + 1]["et"] - 1e-12 for i in range(len(asc) - 1))
    tps_monotone = all(asc[i]["tps"] < asc[i + 1]["tps"] - 1e-9 for i in range(len(asc) - 1))
    # the YELLOW band E[T]/TPS span (linear -> GREEN edge).
    yellow_et_span = [pt_green["et"], pt_linear["et"]]
    yellow_tps_span = [pt_green["tps"], pt_linear["tps"]]

    step2 = {
        "raw_a1_held": RAW_A1_DEPLOYED,
        "deep_spine_a_k_ge2": deep_spine,
        "deep_spine_survival_multiplier": deep_mult,
        "et_cap_deployed_deep_spine": et_cap_deployed_deep,
        "tps_cap_deployed_deep_spine": tps_cap_deployed_deep,
        "et_target_611": ET_TARGET_611,
        "deployed_deep_spine_caps_below_611": bool(et_cap_deployed_deep < ET_TARGET_611 - TOL),
        "et_611_shortfall_at_cap": ET_TARGET_611 - et_cap_deployed_deep,
        "curve": curve,
        "point_linear": pt_linear,
        "point_green_edge": pt_green,
        "point_deployed_anchor_cov4_0": pt_deployed_anchor,
        "yellow_band_et_span_green_to_linear": yellow_et_span,
        "yellow_band_tps_span_green_to_linear": yellow_tps_span,
        "deployed_ladder_et": et_deployed_ladder,
        "deployed_anchor_et_abs_diff_vs_ladder": deployed_anchor_et_abs_diff,
        "deployed_anchor_tps": deployed_anchor_tps,
        "deployed_convention_tps": deployed_conv_tps,
        "deployed_convention_tps_abs_diff_vs_baseline": deployed_conv_tps_abs_diff,
        "et_monotone_decreasing_in_frac": bool(et_monotone),
        "tps_monotone_decreasing_in_frac": bool(tps_monotone),
        "note": ("realistic chain: raw a1 held at deployed {:.5f}, deep spine at the banked deployed "
                 "ladder. YELLOW band (frac {:.4f}->{:.4f}) buys E[T] {:.4f}->{:.4f}, deployed-loopgraph "
                 "TPS {:.1f}->{:.1f}. At cov4=0 (frac=1) the chain reproduces the deployed point "
                 "(E[T]={:.4f}, TPS={:.1f}). The deployed deep spine CAPS E[T] at {:.4f} (a1_eff=1), "
                 "{:.3f} below the 6.11 build target -- position-1 rank-coverage salvage alone cannot "
                 "reach 6.11 (reproduces denken #304's 4.92 ceiling)."
                 .format(RAW_A1_DEPLOYED, FRAC_TRUE_BEYOND_TOP4, FRAC_STAR_GREEN_313, pt_linear["et"],
                         pt_green["et"], pt_linear["tps"], pt_green["tps"], pt_deployed_anchor["et"],
                         pt_deployed_anchor["tps"], et_cap_deployed_deep,
                         ET_TARGET_611 - et_cap_deployed_deep)),
    }

    # ---- STEP 3: solve the build-target bar in rank-coverage units. ---- #
    # Build configuration: deep spine lifted to the #297 build-uniform target (effective per-position == T),
    # position-1 salvaged. max_frac_clearing_611(raw_a1) = (1-T)/(1-raw_a1).
    bar_by_raw_a1 = {f"{ra:.6f}": {
        "raw_a1": ra,
        "max_frac_clearing_611": max_frac_clearing_611(ra),
        "min_cov4_clearing_611": 1.0 - max_frac_clearing_611(ra),
    } for ra in RAW_A1_ANCHORS}

    headline_max_frac = max_frac_clearing_611(RAW_A1_DEPLOYED)       # conservative deployed raw a1
    headline_min_cov4 = 1.0 - headline_max_frac
    # validation: the bar reproduces #309's linear frac (raw a1=0.7731) and #313's GREEN edge (raw a1=0.80).
    bar_at_309_w4 = max_frac_clearing_611(A1_REQ_309_W4)
    bar_at_309_abs_diff_vs_linear_frac = abs(bar_at_309_w4 - FRAC_TRUE_BEYOND_TOP4)
    bar_at_green = max_frac_clearing_611(TRAINABLE_BAND_EDGE)
    bar_at_green_abs_diff_vs_313 = abs(bar_at_green - FRAC_STAR_GREEN_313)
    bar_at_red = max_frac_clearing_611(T_EFFECTIVE)                 # raw a1 = T -> frac = 1 (RED)
    # does the LINEAR spine (frac=0.3468) clear 6.11 under salvage, at the conservative deployed raw a1?
    linear_spine_clears_611 = bool(FRAC_TRUE_BEYOND_TOP4 <= headline_max_frac + TOL)
    # rank-coverage headroom the BUILD has at deployed raw a1 (negative => needs better-than-linear cov).
    build_frac_headroom_at_deployed = headline_max_frac - FRAC_TRUE_BEYOND_TOP4
    build_cov4_gap_at_deployed = headline_min_cov4 - COV_W[PRIMARY_W]   # extra cov4 the fusion draft needs
    # the raw a1 at which the linear spine EXACTLY clears 6.11 (== #309's W4 demand, by construction).
    raw_a1_for_linear_clears = a1_draft_for_effective(T_EFFECTIVE, COV_W[PRIMARY_W])

    # round-trip: build the full uniform-T chain at the headline bar and confirm E[T]=6.11.
    cov4_head = headline_min_cov4
    a1_eff_head = tree_recovered(RAW_A1_DEPLOYED, cov4_head)
    build_chain_head = [a1_eff_head] + [T_EFFECTIVE] * (K_SPEC - 1)
    et_build_head = et_from_cond(build_chain_head)
    build_bar_roundtrips_611 = bool(abs(et_build_head - ET_TARGET_611) <= 1e-3)

    step3 = {
        "build_deep_spine_effective_per_position": T_EFFECTIVE,
        "max_frac_clearing_611_formula": "(1 - T) / (1 - raw_a1)",
        "max_frac_beyond_top4_clearing_611": headline_max_frac,
        "min_cov4_clearing_611": headline_min_cov4,
        "headline_raw_a1_deployed": RAW_A1_DEPLOYED,
        "bar_by_raw_a1": bar_by_raw_a1,
        "bar_at_309_w4_raw_a1": bar_at_309_w4,
        "bar_at_309_abs_diff_vs_linear_frac": bar_at_309_abs_diff_vs_linear_frac,
        "bar_at_green_cap_raw_a1": bar_at_green,
        "bar_at_green_abs_diff_vs_313_green_edge": bar_at_green_abs_diff_vs_313,
        "bar_at_red_raw_a1_T": bar_at_red,
        "linear_spine_clears_611_under_salvage": linear_spine_clears_611,
        "build_frac_headroom_at_deployed_raw_a1": build_frac_headroom_at_deployed,
        "build_cov4_gap_at_deployed_raw_a1": build_cov4_gap_at_deployed,
        "raw_a1_for_linear_spine_exact_clear": raw_a1_for_linear_clears,
        "build_bar_roundtrips_611": build_bar_roundtrips_611,
        "et_build_chain_at_headline_bar": et_build_head,
        "reconciliation_with_313": (
            "this does NOT contradict #313's GREEN verdict. #313 holds the fusion draft TRAINED to the "
            "salvage demand a1_req(cov4) and reads GREEN iff a1_req <= 0.80; at the linear spine cov4 the "
            "demand is 0.7731 <= 0.80 -> GREEN, frac-headroom +0.0467 to the 0.3935 edge. THIS card's "
            "headline fixes the OTHER axis: a fusion draft no better than the deployed spine at raw rank-1 "
            "(raw a1=0.7293). The two readings bracket the unmeasured fusion draft: OPTIMISTIC (trained "
            "raw a1>=0.7731) -> linear spine clears, max_frac=0.3468..0.3935; CONSERVATIVE (deployed raw "
            "a1) -> linear spine MISSES, max_frac=0.2907 (needs better-than-linear cov4 or trained-up raw "
            "a1). #313's GREEN is contingent on training raw a1 up to >= 0.7731."),
        "note": ("build bar max_frac=(1-T)/(1-raw_a1) reproduces #309's linear frac (raw a1={:.4f} -> "
                 "{:.4f}=={:.4f}) and #313's GREEN edge (raw a1=0.80 -> {:.4f}=={:.4f}). HEADLINE at the "
                 "conservative deployed raw a1 {:.5f}: max_frac={:.4f} (cov4>={:.4f}) -- TIGHTER than the "
                 "linear spine's {:.4f}, so a deployed-quality fusion draft does NOT clear 6.11 at the "
                 "linear spine's rank-coverage; it needs cov4 +{:.4f} (to {:.4f}) OR raw a1 >= {:.4f}."
                 .format(A1_REQ_309_W4, bar_at_309_w4, FRAC_TRUE_BEYOND_TOP4, bar_at_green,
                         FRAC_STAR_GREEN_313, RAW_A1_DEPLOYED, headline_max_frac, headline_min_cov4,
                         FRAC_TRUE_BEYOND_TOP4, build_cov4_gap_at_deployed, headline_min_cov4,
                         A1_REQ_309_W4)),
    }

    # ---- STEP 4: honest caveat. ---- #
    step4 = {
        "fusion_cov_W_measured": False,
        "fusion_checkpoint_exists": False,
        "deep_spine_held_deployed_for_step2": True,
        "deep_spine_lifted_to_build_uniform_for_step3": True,
        "deep_spine_is_second_order_axis": True,
        "deployed_deep_spine_is_the_binding_wall": bool(step2["deployed_deep_spine_caps_below_611"]),
        "caveat": ("the fusion draft's cov_W is UNMEASURED -- no fusion EAGLE-3 checkpoint exists -- so "
                   "this card maps the threshold -> E[T] -> TPS transfer function parameterized by the "
                   "still-to-be-measured fusion frac. Step 2 holds the deep spine a_{{k>=2}} at banked "
                   "DEPLOYED values (the realistic 'slightly-worse-than-linear' reading): under that "
                   "deep spine E[T] caps at {:.4f}, {:.3f} below the 6.11 build target -- position-1 "
                   "rank-coverage salvage ALONE cannot reach 6.11; the deep spine is the binding wall. "
                   "Step 3's build bar ASSUMES the deep spine is lifted to the #297 build-uniform target "
                   "(effective per-position == T); a {{2,21,39}}-fusion draft may shift the deep spine "
                   "too (a flagged second-order axis). RED (raw a1 == T -> frac=1) stays unreachable for "
                   "any positive rank-2+ transfer."
                   .format(step2["et_cap_deployed_deep_spine"],
                           ET_TARGET_611 - step2["et_cap_deployed_deep_spine"])),
        "what_flips_the_verdict": ("run the #79 RANKPROBE_W=4 probe on the fusion draft head; compare its "
                                   "measured frac_true_beyond_top4 to the {:.4f} build bar (deployed raw "
                                   "a1) and to #313's {:.4f} GREEN edge."
                                   .format(headline_max_frac, FRAC_STAR_GREEN_313)),
    }

    return {
        "step1_reproduce_313": step1,
        "step2_et_tps_curve": step2,
        "step3_build_bar": step3,
        "step4_caveat": step4,
        "test_metrics": {
            "max_frac_beyond_top4_clearing_611": headline_max_frac,
            "linear_spine_clears_611_under_salvage": linear_spine_clears_611,
        },
        "imported": {
            "T_effective_304": T_EFFECTIVE,
            "a1_deployed_304": A1_DEPLOYED,
            "et_target_611_295": ET_TARGET_611,
            "a_k_deployed_300": A_K,
            "cov_W_measured_79": COV_W,
            "frac_true_beyond_top4_79": FRAC_TRUE_BEYOND_TOP4,
            "a1_req_309_w4": A1_REQ_309_W4,
            "cov4_star_green_313": COV4_STAR_GREEN_313,
            "frac_star_green_313": FRAC_STAR_GREEN_313,
            "K_cal_269": K_CAL, "tau": TAU, "served_step_us": SERVED_STEP_US,
            "deployed_et_convention": DEPLOYED_ET_CONVENTION, "baseline_tps": BASELINE_TPS,
            "provenance": ("salvage operator + threshold lawine #313 (sw492nih) & #309 (7tkn4d9x); "
                           "measured cov_W + frac wirbel #79 (z6wi4z4v); deployed ladder lawine #300 "
                           "(8t5q6sr0); effective target T denken #304 (dtf1ouml); E[T]=6.11 wirbel #295 "
                           "(c334qaqu); K_cal kanna #269. E[T]/TPS transfer + build bar are this card "
                           "(#316)."),
        },
    }


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def self_test(syn: dict[str, Any], banked: dict[str, Any]) -> dict[str, Any]:
    s1 = syn["step1_reproduce_313"]
    s2 = syn["step2_et_tps_curve"]
    s3 = syn["step3_build_bar"]
    s4 = syn["step4_caveat"]
    c: dict[str, bool] = {}

    # (a) #313 operator + threshold reproduced to <= 1e-6.
    c["01_reproduces_313_green_threshold"] = bool(
        s1["cov_green_abs_diff_vs_313"] <= TOL_REPRO and s1["frac_green_abs_diff_vs_313"] <= TOL_REPRO)
    c["02_red_threshold_is_zero_cov_unit_frac"] = bool(
        abs(s1["cov4_star_red"]) <= TOL and abs(s1["frac_star_red"] - 1.0) <= TOL)
    c["03_linear_demand_reproduces_309_w4"] = bool(s1["demand_linear_abs_diff_vs_309"] <= TOL_REPRO)
    # GREEN threshold round-trips: plug cov4*_green back through the salvage inverse -> 0.80.
    rt_green = a1_draft_for_effective(T_EFFECTIVE, s1["cov4_star_green"])
    c["04_green_threshold_roundtrips_to_edge"] = bool(abs(rt_green - TRAINABLE_BAND_EDGE) <= TOL_REPRO)

    # (b) frac -> cov -> E[T] -> TPS curve monotone + reproduces the deployed anchor.
    c["05_et_monotone_decreasing_in_frac"] = bool(s2["et_monotone_decreasing_in_frac"])
    c["06_tps_monotone_decreasing_in_frac"] = bool(s2["tps_monotone_decreasing_in_frac"])
    # at cov4=0 (frac=1) the chain reproduces the deployed ladder E[T] exactly.
    c["07_deployed_anchor_reproduces_ladder"] = bool(
        s2["deployed_anchor_et_abs_diff_vs_ladder"] <= 1e-9)
    # the deployed convention TPS round-trips the baseline 481.53.
    c["08_deployed_convention_tps_is_baseline"] = bool(
        s2["deployed_convention_tps_abs_diff_vs_baseline"] <= 0.05)
    # the deployed deep spine caps E[T] below the 6.11 build target (the binding wall).
    c["09_deployed_deep_spine_caps_below_611"] = bool(s2["deployed_deep_spine_caps_below_611"])

    # (c) build-target (E[T]=6.11) max-frac bar solved + reproduces #309/#313 anchors.
    c["10_build_bar_reproduces_309_linear_frac"] = bool(
        s3["bar_at_309_abs_diff_vs_linear_frac"] <= TOL_REPRO)
    c["11_build_bar_reproduces_313_green_edge"] = bool(
        s3["bar_at_green_abs_diff_vs_313_green_edge"] <= TOL_REPRO)
    c["12_build_bar_red_at_raw_a1_T_is_unit_frac"] = bool(abs(s3["bar_at_red_raw_a1_T"] - 1.0) <= TOL)
    c["13_headline_max_frac_in_unit_interval"] = bool(
        0.0 < s3["max_frac_beyond_top4_clearing_611"] < 1.0)
    c["14_build_bar_roundtrips_611"] = bool(s3["build_bar_roundtrips_611"])
    # internal consistency: the linear-clears bool agrees with comparing linear frac to the headline bar.
    c["15_linear_clears_bool_consistent"] = bool(
        s3["linear_spine_clears_611_under_salvage"]
        == (FRAC_TRUE_BEYOND_TOP4 <= s3["max_frac_beyond_top4_clearing_611"] + TOL))

    # (d) imported constants match the on-disk banked artifacts to <= 1e-6 (no silent drift).
    consts_ok = True
    d313 = banked.get("pr313", {})
    d309 = banked.get("pr309", {})
    d295 = banked.get("pr295", {})
    if d313:
        s2_313 = (d313.get("synthesis", {}) or {}).get("step2_decision_threshold", {}) or {}
        if s2_313.get("effective_target_T") is not None:
            consts_ok = consts_ok and abs(float(s2_313["effective_target_T"]) - T_EFFECTIVE) <= TOL_REPRO
        if s2_313.get("cov_W_star_green") is not None:
            consts_ok = consts_ok and abs(float(s2_313["cov_W_star_green"]) - COV4_STAR_GREEN_313) <= TOL_REPRO
    if d309:
        imp = (d309.get("synthesis", {}) or {}).get("imported", {})
        if imp.get("a1_required_611_nosalvage") is not None:
            consts_ok = consts_ok and abs(float(imp["a1_required_611_nosalvage"]) - T_EFFECTIVE) <= TOL_REPRO
        if imp.get("frac_true_beyond_top4") is not None:
            consts_ok = consts_ok and abs(
                float(imp["frac_true_beyond_top4"]) - FRAC_TRUE_BEYOND_TOP4) <= TOL_REPRO
        req = ((d309.get("synthesis", {}) or {}).get("step3_invert", {})
               or {}).get("a1_draft_required_by_W", {}) or {}
        if req.get("4") is not None:
            consts_ok = consts_ok and abs(float(req["4"]) - A1_REQ_309_W4) <= TOL_REPRO
    if d295:
        rb = (d295.get("synthesis", {}) or {}).get("regime_bracket", {}) or {}
        if rb.get("corrected_central") is not None:
            consts_ok = consts_ok and abs(float(rb["corrected_central"]) - ET_TARGET_611) <= TOL_REPRO
    c["16_constants_match_banked_artifacts"] = bool(consts_ok and bool(d313) and bool(d309) and bool(d295))

    # (e) caveats carried (pre-registration honesty).
    c["17_caveats_carried"] = bool(
        s4["fusion_cov_W_measured"] is False and s4["fusion_checkpoint_exists"] is False
        and s4["deep_spine_is_second_order_axis"] is True
        and isinstance(s4["caveat"], str) and len(s4["caveat"]) > 120)

    gate = all(bool(v) for v in c.values())
    return {"yellow_zone_et_self_test_passes": gate, "checks": c}


# --------------------------------------------------------------------------- #
# NaN-clean walk.
# --------------------------------------------------------------------------- #
def assert_nan_clean(payload: dict) -> list[str]:
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

    walk(payload, "result")
    return bad


# --------------------------------------------------------------------------- #
# W&B logging (summary/ namespace; robust; never fatal).
# --------------------------------------------------------------------------- #
def maybe_log_wandb(args, payload: dict) -> str | None:
    if getattr(args, "no_wandb", False) or not getattr(args, "wandb_name", None):
        return None
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_json_artifact, log_summary
    except Exception as exc:  # noqa: BLE001
        print(f"[eagle3-yellow-zone-et] wandb logging unavailable: {exc}", flush=True)
        return None

    syn = payload["synthesis"]
    s1, s2, s3, s4 = (syn["step1_reproduce_313"], syn["step2_et_tps_curve"],
                      syn["step3_build_bar"], syn["step4_caveat"])
    st = payload["self_test"]
    tm = syn["test_metrics"]

    run = init_wandb_run(
        job_type="validity-analytic", agent="lawine", name=args.wandb_name, group=args.wandb_group,
        tags=["eagle3-yellow-zone-et", "validity-analytic", "rank-coverage", "tree-verify", "eagle3",
              "et-tps-transfer", "build-target", "bank-the-analysis"],
        config={
            "pr": 316, "effective_target_T": T_EFFECTIVE, "et_target_611": ET_TARGET_611,
            "a1_deployed": A1_DEPLOYED, "cov4_linear_79": COV_W[PRIMARY_W],
            "frac_true_beyond_top4_79": FRAC_TRUE_BEYOND_TOP4, "K_cal": K_CAL, "tau": TAU,
            "served_step_us": SERVED_STEP_US, "baseline_tps": BASELINE_TPS,
            "wandb_group": args.wandb_group, "imports": syn["imported"]["provenance"],
        },
    )
    if run is None:
        print("[eagle3-yellow-zone-et] wandb: no run (no WANDB_API_KEY/mode) -- skipping", flush=True)
        return None

    summary: dict[str, Any] = {
        # PRIMARY + TEST
        "yellow_zone_et_self_test_passes": int(bool(st["yellow_zone_et_self_test_passes"])),
        "max_frac_beyond_top4_clearing_611": tm["max_frac_beyond_top4_clearing_611"],
        "linear_spine_clears_611_under_salvage": int(bool(tm["linear_spine_clears_611_under_salvage"])),
        # step1 reproduction residuals
        "cov_green_abs_diff_vs_313": s1["cov_green_abs_diff_vs_313"],
        "frac_green_abs_diff_vs_313": s1["frac_green_abs_diff_vs_313"],
        "demand_linear_abs_diff_vs_309": s1["demand_linear_abs_diff_vs_309"],
        "cov4_star_green": s1["cov4_star_green"], "frac_star_green": s1["frac_star_green"],
        # step2 E[T]/TPS curve
        "et_at_linear": s2["point_linear"]["et"], "tps_at_linear": s2["point_linear"]["tps"],
        "et_at_green_edge": s2["point_green_edge"]["et"], "tps_at_green_edge": s2["point_green_edge"]["tps"],
        "et_cap_deployed_deep_spine": s2["et_cap_deployed_deep_spine"],
        "tps_cap_deployed_deep_spine": s2["tps_cap_deployed_deep_spine"],
        "et_611_shortfall_at_cap": s2["et_611_shortfall_at_cap"],
        "deployed_anchor_et": s2["point_deployed_anchor_cov4_0"]["et"],
        "deployed_anchor_tps": s2["deployed_anchor_tps"],
        "deployed_convention_tps": s2["deployed_convention_tps"],
        "et_monotone_decreasing_in_frac": int(bool(s2["et_monotone_decreasing_in_frac"])),
        "tps_monotone_decreasing_in_frac": int(bool(s2["tps_monotone_decreasing_in_frac"])),
        # step3 build bar
        "min_cov4_clearing_611": s3["min_cov4_clearing_611"],
        "bar_at_309_w4_raw_a1": s3["bar_at_309_w4_raw_a1"],
        "bar_at_green_cap_raw_a1": s3["bar_at_green_cap_raw_a1"],
        "build_frac_headroom_at_deployed_raw_a1": s3["build_frac_headroom_at_deployed_raw_a1"],
        "build_cov4_gap_at_deployed_raw_a1": s3["build_cov4_gap_at_deployed_raw_a1"],
        "et_build_chain_at_headline_bar": s3["et_build_chain_at_headline_bar"],
        # imported anchors
        "cov4_linear_79": COV_W[PRIMARY_W], "frac_true_beyond_top4_79": FRAC_TRUE_BEYOND_TOP4,
        "effective_target_T_304": T_EFFECTIVE, "et_target_611_295": ET_TARGET_611,
        # caveat flags
        "fusion_cov_W_measured": int(bool(s4["fusion_cov_W_measured"])),
        "deployed_deep_spine_is_the_binding_wall": int(bool(s4["deployed_deep_spine_is_the_binding_wall"])),
        # hygiene
        "nan_clean": int(bool(payload["nan_clean"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(val)) for k, val in st["checks"].items()},
    }
    summary = {k: val for k, val in summary.items()
               if not (isinstance(val, float) and not math.isfinite(val)) and val is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="eagle3_yellow_zone_et_result", artifact_type="validity", data=payload)
    rid = getattr(run, "id", None)
    print(f"[eagle3-yellow-zone-et] wandb run: {getattr(run, 'url', rid)}", flush=True)
    finish_wandb(run)
    return rid


# --------------------------------------------------------------------------- #
# Console report.
# --------------------------------------------------------------------------- #
def print_report(payload: dict) -> None:
    syn = payload["synthesis"]
    s1, s2, s3, s4 = (syn["step1_reproduce_313"], syn["step2_et_tps_curve"],
                      syn["step3_build_bar"], syn["step4_caveat"])
    st = payload["self_test"]
    print("\n" + "=" * 100, flush=True)
    print("EAGLE-3 YELLOW-ZONE E[T]/TPS (PR #316) -- what fusion rank-coverage still clears E[T]=6.11?",
          flush=True)
    print("=" * 100, flush=True)
    print("STEP 1 -- reproduce #313 operator + threshold:", flush=True)
    print(f"  GREEN edge d=0.80: cov4* = {s1['cov4_star_green']:.6f} frac* = {s1['frac_star_green']:.6f}  "
          f"(|diff| vs #313: cov {s1['cov_green_abs_diff_vs_313']:.2e} frac {s1['frac_green_abs_diff_vs_313']:.2e})",
          flush=True)
    print(f"  RED edge: cov4* = {s1['cov4_star_red']:.4f} frac* = {s1['frac_star_red']:.4f}; linear "
          f"salvaged demand {s1['salvaged_raw_a1_demand_at_linear']:.6f} (|diff| vs #309 W4 "
          f"{s1['demand_linear_abs_diff_vs_309']:.2e})", flush=True)
    print("-" * 100, flush=True)
    print("STEP 2 -- YELLOW band E[T]/TPS (raw a1 held deployed, deployed deep spine):", flush=True)
    print(f"  {'frac':>7} {'cov4':>7} {'a1_eff':>8} {'E[T]':>8} {'TPS':>9}", flush=True)
    for r in s2["curve"]:
        tag = ""
        if abs(r["frac"] - s1["linear_spine_frac"]) < 1e-9:
            tag = "  <- LINEAR"
        elif abs(r["frac"] - s1["frac_star_green"]) < 1e-9:
            tag = "  <- GREEN edge"
        elif abs(r["frac"] - 1.0) < 1e-9:
            tag = "  <- deployed anchor (cov4=0)"
        print(f"  {r['frac']:>7.4f} {r['cov4']:>7.4f} {r['a1_eff']:>8.4f} {r['et']:>8.4f} "
              f"{r['tps']:>9.2f}{tag}", flush=True)
    print(f"  deep-spine CAP: E[T] = {s2['et_cap_deployed_deep_spine']:.4f} (TPS "
          f"{s2['tps_cap_deployed_deep_spine']:.1f}) at a1_eff=1.0 -- {s2['et_611_shortfall_at_cap']:.3f} "
          f"BELOW the 6.11 build target. monotone E[T] {s2['et_monotone_decreasing_in_frac']} / TPS "
          f"{s2['tps_monotone_decreasing_in_frac']}", flush=True)
    print("-" * 100, flush=True)
    print("STEP 3 -- build-target bar in rank-coverage units (build-uniform deep spine):", flush=True)
    print(f"  max_frac_clearing_611(raw_a1) = (1-T)/(1-raw_a1):", flush=True)
    for _, b in sorted(s3["bar_by_raw_a1"].items(), key=lambda kv: kv[1]["raw_a1"]):
        print(f"    raw a1 = {b['raw_a1']:.6f} -> max_frac = {b['max_frac_clearing_611']:.6f} "
              f"(cov4 >= {b['min_cov4_clearing_611']:.6f})", flush=True)
    print(f"  validation: bar(0.7731)={s3['bar_at_309_w4_raw_a1']:.6f} vs linear frac "
          f"{s1['linear_spine_frac']:.6f} (|diff| {s3['bar_at_309_abs_diff_vs_linear_frac']:.2e}); "
          f"bar(0.80)={s3['bar_at_green_cap_raw_a1']:.6f} vs #313 GREEN {s1['frac_star_green']:.6f} "
          f"(|diff| {s3['bar_at_green_abs_diff_vs_313_green_edge']:.2e})", flush=True)
    print(f"  HEADLINE (deployed raw a1 {s3['headline_raw_a1_deployed']:.5f}): "
          f"max_frac_beyond_top4_clearing_611 = {s3['max_frac_beyond_top4_clearing_611']:.6f} "
          f"(cov4 >= {s3['min_cov4_clearing_611']:.6f})", flush=True)
    print(f"  linear_spine_clears_611_under_salvage = {s3['linear_spine_clears_611_under_salvage']}  "
          f"(build frac-headroom at deployed raw a1 = {s3['build_frac_headroom_at_deployed_raw_a1']:+.4f}, "
          f"cov4 gap +{s3['build_cov4_gap_at_deployed_raw_a1']:.4f})", flush=True)
    print("-" * 100, flush=True)
    print("STEP 4 -- caveat:", flush=True)
    print(f"  {s4['caveat']}", flush=True)
    print("-" * 100, flush=True)
    print(f"(PRIMARY) yellow_zone_et_self_test_passes = {st['yellow_zone_et_self_test_passes']}", flush=True)
    for k, val in st["checks"].items():
        print(f"   - {k}: {val}", flush=True)
    print(f"nan_clean = {payload['nan_clean']}   peak_mem_mib = {payload['peak_mem_mib']}", flush=True)
    print("=" * 100 + "\n", flush=True)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="eagle3-yellow-zone-et")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    banked = load_banked()
    syn = synthesize(banked)
    st = self_test(syn, banked)

    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "pr": 316, "agent": "lawine", "kind": "eagle3-yellow-zone-et",
        "eagle3_yellow_zone_et_analysis_only": True,
        "synthesis": syn, "self_test": st,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }

    nan_paths = assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[eagle3-yellow-zone-et] WARNING non-finite at: {nan_paths}", flush=True)
    gate = bool(st["yellow_zone_et_self_test_passes"] and payload["nan_clean"])
    st["yellow_zone_et_self_test_passes"] = gate

    print_report(payload)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eagle3_yellow_zone_et_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[eagle3-yellow-zone-et] wrote {out_path}", flush=True)

    rid = maybe_log_wandb(args, payload)
    payload["wandb_run_id"] = rid
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    tm = syn["test_metrics"]
    print(f"  PRIMARY yellow_zone_et_self_test_passes = {gate}", flush=True)
    print(f"  TEST max_frac_beyond_top4_clearing_611 = "
          f"{tm['max_frac_beyond_top4_clearing_611']:.6f}", flush=True)
    print(f"  TEST linear_spine_clears_611_under_salvage = "
          f"{tm['linear_spine_clears_611_under_salvage']}", flush=True)
    print(f"  wandb run = {rid}", flush=True)

    if args.self_test:
        print(f"[eagle3-yellow-zone-et] self-test {'PASS' if gate else 'FAIL'}", flush=True)
        return 0 if gate else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
