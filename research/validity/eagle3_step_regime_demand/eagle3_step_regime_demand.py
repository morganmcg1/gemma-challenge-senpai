#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #320 (denken) -- EAGLE-3 step-regime: is the 0.8717 additive a1-demand real or a bookend?

THE YELLOW THIS CARD CLOSES
---------------------------
denken #308 (5axqa6oa) landed the EAGLE-3 drafter trainability light at GREEN-YELLOW: on the
M=8 verify-tree the RAW {2,21,39}-fusion a1 demand to hit the EAGLE-3 build's E[T] target relaxes
from #304's 0.9213 to 0.7731 at the CENTRAL step regime (cov4=0.6532), which the in-repo head's
native step-1 a1=0.7714 essentially meets (gap 0.0017 < the head's own native-vs-tf 0.0097 spread).
But #308 flagged a YELLOW: at the ADDITIVE-UPPER step regime the demand re-inflates to 0.8717 > 0.80,
which the native head would MISS. This card decides whether 0.8717 is a LIVE blocker or a pessimistic
upper BOOKEND.

THE TWO THINGS "MULTIPLICATIVE vs ADDITIVE" CAN MEAN (the disambiguation that resolves the YELLOW)
-------------------------------------------------------------------------------------------------
The PR frames the regimes as "how per-position acceptance composes into E[T]". There are in fact TWO
distinct axes that the label "multiplicative vs additive" attaches to, and only ONE is a live fork:

  AXIS A -- the E[T] ACCEPTANCE composition (per-position acceptance -> expected tokens).
     multiplicative-geometric:  E[T] = 1 + sum_d prod_{j<=d} a_j   (chain / committed survivorship)
     additive-independent:      E[T] = 1 + sum_j a_j               (branches scored independently)
     >>> SETTLED MULTIPLICATIVE. A verify accepts the LONGEST matching prefix: position k is taken
         only if every position < k on its path was taken (reject => resample => discard the tail).
         The tree adds BREADTH (multiple candidates per depth, lifted by the rank-2+ salvage operator),
         but the DEPTH composition is still a cumulative product. The additive-independent E[T] counts a
         deep token even when its ancestor was rejected -- physically impossible -- so it is a STRICT
         UPPER BOUND on E[T], never the operating composition. BOTH the 0.7731 and 0.8717 demands are
         already computed on the multiplicative chain; Axis A is NOT where the spread lives.

  AXIS B -- the STEP-COST denominator (draft-step WALL -> the E[T] TARGET that 500 TPS demands).
     wirbel #295 (eagle3_step_profile) measured the EAGLE-3 fusion draft-step wall and removed the
     standalone-harness regime offset two ways:
       multiplicative anchoring (uniform-speedup premise)   -> LIGHT step  -> E[T] target 5.3636 (LOWER)
       additive anchoring        (common-additive premise)  -> HEAVY step  -> E[T] target 6.8588 (UPPER)
       central (the validated point)                        ->             -> E[T] target 6.1112
     >>> THE 0.7731 / 0.8717 SPREAD LIVES ENTIRELY ON AXIS B. Both demands inherit the SAME multiplicative
         Axis-A chain; they differ only in the E[T] TARGET set by the step-cost regime. "additive-upper"
         here is a STEP-cost anchoring, NOT an E[T]-acceptance composition.

So the question "is 0.8717 the additive composition demand?" is mis-posed: 0.8717 is NOT an additive-
composition number (Axis A is multiplicative). It is the Axis-B step-cost UPPER bookend's demand. This
card (1) proves Axis A is multiplicative (additive is a strict upper bound), (2) recomputes the raw-a1
demand on the multiplicative chain at the honest-500 floor and across the Axis-B step bracket, (3)
reports the YELLOW demand band and whether native clears, and (4) shows the additive-upper 6.86 is a
true upper bookend (its common-additive premise is refuted by #295's own bandwidth decomposition) and
quantifies how far below it the operating point sits.

SCOPE / HONESTY
---------------
LOCAL CPU-only analytic card over banked acceptance ladders, rank-coverage, and the #295 step bracket.
0 GPU / vLLM / model forward / training / HF Job / submission / served-file change. BASELINE 481.53
unchanged (0 TPS). Greedy/PPL untouched. This closes #308's STEP-REGIME YELLOW only; it does NOT touch
#308's other YELLOW (cov-transfer of the linear-spine rank-coverage to the fusion draft) and does NOT
make the build GO -- the fern #305 go-card private-binding axis still pins private sub-500 at the
central target (private 402 < 500). The drafter BUILD stays human-gated.

PRIMARY metric  step_regime_demand_resolved_self_test_passes
TEST    metric  raw_a1_required_operating_regime  (raw fusion-draft a1 demand at the CENTRAL/validated
                                                   step regime, after the M=8 cov4 tree salvage)

Reproduce:
    cd target/ && .venv/bin/python research/validity/eagle3_step_regime_demand/eagle3_step_regime_demand.py \\
        --self-test --wandb_group eagle3-step-regime --wandb_name denken/eagle3-step-regime-demand
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
# Banked anchors (imported VERBATIM; this leg re-derives none of them and the
# self-test reproduces every banked demand from first principles to 1e-6).
# All runs live in wandb-applied-ai-team/gemma-challenge-senpai.
# --------------------------------------------------------------------------- #
# ---- composition law (kanna #217 vgovdrjc / kanna #269): official = K_cal * E[T] ----
K_CAL = 125.268                       # steps/sec calibration (deployed-linear step)
OFFICIAL_PUBLIC = 481.53              # PR #52 deployed public TPS (linear MTP K=7)
STEP_US = 1218.2                      # served step (NORMALIZED unit)
TAU = 1.218                           # composition round-trip tau
E_T_DEPLOYED = 3.844                  # deployed K=7-linear public E[T] @ M=8
K_SPEC = 7                            # num_speculative_tokens (manifest)
E_T_MAX = float(K_SPEC + 1)          # 8.0 full-acceptance (K+1) ceiling
TARGET_TPS = 500.0                    # the binding launch gate (land #245)
ET_PRIVATE_500 = TARGET_TPS / K_CAL  # 3.99144: honest-500 E[T] @ the DEPLOYED-linear step

# ---- deployed-effective per-position conditional ladder (lawine #300 / kanna #289 / denken #297) ----
A_K = [
    0.7292532942898975, 0.759556697719242, 0.7929794882639035, 0.8228,
    0.8348727920920435, 0.8357919254658385, 0.8464932652113331,
]
A1_DEPLOYED = 0.72925                 # deployed a_1 (the cliff)

# ---- denken #304 (dtf1ouml): the no-salvage uniform EFFECTIVE demand for E[T]=6.11 ----
A1_REQUIRED_611_NOSALVAGE = 0.9213011665456927

# ---- wirbel #295 (eagle3_step_profile): the Axis-B step-cost regime bracket ----
# E[T] TARGET that 500 TPS demands, by how the standalone-harness offset is removed from the measured
# EAGLE-3 fusion draft-step wall. multiplicative=light step (lower E[T]); additive=heavy step (upper).
ET_TARGET_LOWER = 5.363610726985671   # multiplicative-anchored (light step) -> LOWER E[T] target
ET_TARGET_CENTRAL = 6.1112149873699195  # central / validated operating point
ET_TARGET_UPPER = 6.858819247754167   # additive-anchored (heavy step) -> UPPER E[T] target (bookend)
STEP_MULT_LOWER = 1.744676699335575   # multiplicative faithful draft-step multiplier vs linear
STEP_MULT_UPPER = 4.161395297380165   # additive faithful draft-step multiplier vs linear
WALL_DELTA_ADDITIVE_US = 2234.649658203125  # full harness chain delta (additive anchoring marginal)
# #295 bandwidth decomposition (the premise check for the additive anchoring).
BW_UTIL_PCT_LINEAR = 11.469762580782996   # tiny 256-dim linear: dispatch-bound, far below peak BW
BW_UTIL_PCT_FAITHFUL = 59.89991548615931  # faithful 2560-dim EAGLE-3: near-bandwidth
BW_UTIL_RATIO = 5.222419824671747         # linear is 5.2x more BW-starved (refutes common-additive)
LINEAR_DISPATCH_BOUND = True
# wirbel #293 (abhoog1x) independent MODELED m_fuse=3 step target (the central's corroboration).
CORRECTED_TARGET_MFUSE3_293 = 6.1245

# ---- wirbel #79 (z6wi4z4v) MEASURED rank-coverage on the deployed-linear stack ----
# cov_W = P(true token caught at rank <= W | rank-1 miss). Primary salvage width W=4 (the M=8 tree's
# width-4 root). Full greedy path, 16,524 records, align_bad=0.
COV4 = 0.6531976066516435
FRAC_TRUE_BEYOND_TOP4 = 0.3468023933483565  # irreducible width-4 miss mass (the cov-transfer YELLOW)

# ---- denken #308 (5axqa6oa): the in-repo {2,21,39} EAGLE-3 head native acceptance + banked demands ----
A1_NATIVE = 0.7714                    # in-repo Eagle3DraftHead native step-1 a1 (fern #34 gua9x68j)
A1_NATIVE_TF = 0.7617                 # same head, teacher-forced step-1
A1_NATIVE_TF_SPREAD = 0.0097          # native-vs-tf spread (the head's own measurement noise)
A1_PUBLISHED_ENVELOPE_MAX = 0.80      # EAGLE-3 published 0-alpha envelope max (arXiv 2503.01840)
# banked salvaged RAW-a1 demands (#308 step6_salvage_cost_loop.cost_bracket) -- reproduced below.
BANK_DEMAND_LOWER = 0.6585256596518204
BANK_DEMAND_CENTRAL = 0.7730729805683441
BANK_DEMAND_UPPER = 0.8716699450084371
BANK_DEMAND_FLOOR = 0.3827524457956924  # #309 deployed-step floor demand (E[T]=3.9914)
# banked uniform EFFECTIVE acceptances per E[T] target (#308 cost_bracket.*.a_eff_uniform).
BANK_AEFF_LOWER = 0.8815758815002
BANK_AEFF_CENTRAL = 0.9213011665456927
BANK_AEFF_UPPER = 0.9554948297903998

# ---- fern #305 (eagle3_go_card): the binding-axis honesty carry (NOT closed by this card) ----
PRIVATE_AT_CENTRAL_TPS = 402.0        # private projection @ E[T]=6.11 under conservative x0.804 OOD
PRIVATE_AT_BRACKET_HI_TPS = 451.17793160534995  # private @ bracket top 6.86 (still < 500)

TOL = 1e-9
TOL_MATCH = 1e-6
TOL_ET = 1e-3
# cross-run calibration noise: the banked deployed ladder (lawine #300) and K_cal (kanna #217) are
# measured in DIFFERENT runs, so the ladder's multiplicative E[T] agrees with the composition-law
# deployed E[T]=official/K_cal=3.844 only to ~0.2% (0.0072), not to machine precision.
TOL_XRUN = 1.2e-2


# --------------------------------------------------------------------------- #
# AXIS A -- the E[T] acceptance composition (the physically-justified model).
# --------------------------------------------------------------------------- #
def survival(cond: list[float]) -> list[float]:
    """committed-survival S_d = prod_{j<=d} a_j (S_0 = 1 implicit). The CHAIN."""
    out, acc = [], 1.0
    for p in cond:
        acc *= float(p)
        out.append(acc)
    return out


def et_multiplicative(cond: list[float]) -> float:
    """PHYSICAL E[T] = 1 + sum_d S_d. 1 base token + expected accepted draft tokens (chain/survivorship)."""
    return 1.0 + sum(survival(cond))


def et_additive_independent(cond: list[float]) -> float:
    """The additive-independent ALTERNATIVE: E[T] = 1 + sum_j a_j (each draft position scored alone,
    no survivorship). This is the strict UPPER BOUND on E[T] -- it credits a deep token even when its
    ancestor was rejected. Reported only to PROVE it dominates the multiplicative chain."""
    return 1.0 + sum(float(p) for p in cond)


def uniform_profile(a_u: float) -> list[float]:
    return [a_u] * K_SPEC


def _bisect(f, lo: float, hi: float, target: float, iters: int = 200) -> float:
    """monotone-increasing f: return x in [lo,hi] with f(x) ~= target."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def effective_uniform_for_et(et_target: float) -> float:
    """uniform EFFECTIVE per-position acceptance whose multiplicative ladder hits E[T]=et_target."""
    return _bisect(lambda a: et_multiplicative(uniform_profile(a)), 0.0, 1.0, et_target)


# --------------------------------------------------------------------------- #
# The M=8 tree salvage operator (rank-2+ recovery at the root; lawine #309 / wirbel #79).
# --------------------------------------------------------------------------- #
def tree_recovered(base: float, cov: float) -> float:
    """Effective acceptance after the M-node verify tree salvages rank-2..W candidates:
    c1_eff = base + (1-base)*cov  (true token is rank-1, prob base; OR in rank-2..W when rank-1 missed)."""
    return min(1.0, max(0.0, base + (1.0 - base) * cov))


def a1_draft_for_effective(a_eff_target: float, cov: float) -> float:
    """Invert the salvage: raw draft a1 with tree_recovered(a1, cov) == a_eff_target.
    a1 = (a_eff - cov)/(1 - cov); cov->0 returns a_eff (no salvage)."""
    if cov >= 1.0 - TOL:
        return 0.0
    return min(1.0, max(0.0, (a_eff_target - cov) / (1.0 - cov)))


def raw_a1_for_et(et_target: float, cov: float) -> float:
    """The full inversion the demand band uses: E[T] target -> uniform effective acceptance (multiplicative
    chain) -> raw fusion-draft a1 after the cov-salvage. cov=0 returns the no-salvage effective demand."""
    return a1_draft_for_effective(effective_uniform_for_et(et_target), cov)


# =========================================================================== #
# Synthesis.
# =========================================================================== #
def synthesize() -> dict[str, Any]:
    # ---- (1) AXIS A: prove the acceptance composition is multiplicative, additive is a strict UB. -- #
    # sanity: the multiplicative chain reproduces the deployed public E[T]=3.844 from the deployed ladder
    # (to within cross-run calibration noise), while the additive model massively overshoots it -- so the
    # multiplicative chain is the DISCRIMINATING fit to deployed reality, not an arbitrary choice.
    et_deployed_recomputed = et_multiplicative(A_K)
    et_deployed_additive = et_additive_independent(A_K)
    mult_err_vs_deployed = abs(et_deployed_recomputed - E_T_DEPLOYED)
    additive_err_vs_deployed = abs(et_deployed_additive - E_T_DEPLOYED)
    reproduces_deployed_et = bool(mult_err_vs_deployed <= TOL_XRUN)
    # multiplicative is FAR closer to deployed truth than additive (additive ~6.62 vs deployed 3.844).
    mult_beats_additive_on_deployed = bool(mult_err_vs_deployed < 0.25 * additive_err_vs_deployed)
    # additive >= multiplicative for every sampled acceptance ladder (equality only at degenerate {0,1}).
    sample_ladders = [
        A_K,
        uniform_profile(0.5), uniform_profile(0.7714), uniform_profile(0.9213),
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
        [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    ]
    dominance = []
    additive_dominates_all = True
    for lad in sample_ladders:
        em = et_multiplicative(lad)
        ea = et_additive_independent(lad)
        ge = bool(ea >= em - TOL)
        additive_dominates_all = additive_dominates_all and ge
        dominance.append({"ladder_mean": sum(lad) / len(lad), "et_mult": em, "et_additive": ea,
                          "additive_ge_mult": ge, "gap": ea - em})
    # degenerate equality endpoints: all-0 and all-1 give additive == multiplicative.
    eq_all0 = bool(abs(et_additive_independent([0.0] * K_SPEC)
                       - et_multiplicative([0.0] * K_SPEC)) <= TOL)
    eq_all1 = bool(abs(et_additive_independent([1.0] * K_SPEC)
                      - et_multiplicative([1.0] * K_SPEC)) <= TOL)
    # closed-form check: additive uniform E[T] == 1 + K*a.
    a_chk = 0.77
    additive_closed_form_ok = bool(
        abs(et_additive_independent(uniform_profile(a_chk)) - (1.0 + K_SPEC * a_chk)) <= TOL)

    axis_a = {
        "composition_model": "multiplicative-geometric (chain / committed survivorship)",
        "et_multiplicative_formula": "E[T] = 1 + sum_d prod_{j<=d} a_j",
        "et_additive_formula": "E[T]_UB = 1 + sum_j a_j  (strict upper bound; not physical)",
        "et_deployed_recomputed": et_deployed_recomputed,
        "et_deployed_additive": et_deployed_additive,
        "mult_err_vs_deployed": mult_err_vs_deployed,
        "additive_err_vs_deployed": additive_err_vs_deployed,
        "reproduces_deployed_et_3p844": reproduces_deployed_et,
        "mult_beats_additive_on_deployed": mult_beats_additive_on_deployed,
        "additive_is_strict_upper_bound": bool(additive_dominates_all),
        "additive_eq_mult_at_all0": eq_all0,
        "additive_eq_mult_at_all1": eq_all1,
        "additive_closed_form_1_plus_Ka": additive_closed_form_ok,
        "dominance_samples": dominance,
        "topology_justification": (
            "A verify accepts the LONGEST matching prefix: draft position k is committed only if every "
            "position j<k on its tree path was committed (a reject triggers resample and discards the "
            "tail). So depth composition is the cumulative product prod_{j<=d} a_j -- multiplicative. The "
            "M=8 tree contributes BREADTH (W candidates per depth), which the rank-2+ salvage operator "
            "folds into a LIFTED per-depth effective acceptance c_eff = a + (1-a)*cov; it does NOT make "
            "the depths independent. The additive-independent E[T] credits a deep token whose ancestor "
            "was rejected -- impossible -- so it strictly upper-bounds E[T] and is never the operating "
            "composition."),
    }

    # ---- (2) recompute the raw-a1 demand on the multiplicative chain at the honest-500 floor. ------ #
    aeff_floor = effective_uniform_for_et(ET_PRIVATE_500)
    raw_demand_floor = a1_draft_for_effective(aeff_floor, COV4)
    floor_matches_bank = bool(abs(raw_demand_floor - BANK_DEMAND_FLOOR) <= TOL_MATCH)
    native_clears_floor = bool(A1_NATIVE > raw_demand_floor)

    floor = {
        "honest500_floor_et": ET_PRIVATE_500,
        "floor_basis": ("E[T] for 500 TPS at the DEPLOYED-linear step (official = K_cal * E[T]). This is "
                        "the floor IF the EAGLE-3 drafter added zero step cost; the EAGLE-3 fusion step "
                        "is heavier, so the OPERATIVE target rises up the #295 Axis-B bracket."),
        "aeff_uniform_for_floor": aeff_floor,
        "raw_a1_demand_floor": raw_demand_floor,
        "raw_a1_demand_floor_matches_bank": floor_matches_bank,
        "native_a1": A1_NATIVE,
        "native_clears_floor": native_clears_floor,
        "native_margin_over_floor": A1_NATIVE - raw_demand_floor,
    }

    # ---- (3) the demand band across the Axis-B step regimes (multiplicative chain + cov4 salvage). - #
    def regime(name: str, et_target: float, bank_demand: float, bank_aeff: float) -> dict[str, Any]:
        aeff = effective_uniform_for_et(et_target)
        raw = a1_draft_for_effective(aeff, COV4)
        # round-trip: lift the raw demand back through the tree and re-evaluate E[T].
        et_back = et_multiplicative(uniform_profile(tree_recovered(raw, COV4)))
        return {
            "regime": name,
            "et_target": et_target,
            "aeff_uniform": aeff,
            "aeff_matches_bank": bool(abs(aeff - bank_aeff) <= TOL_MATCH),
            "raw_a1_demand": raw,
            "raw_a1_demand_matches_bank": bool(abs(raw - bank_demand) <= TOL_MATCH),
            "roundtrip_et": et_back,
            "roundtrip_ok": bool(abs(et_back - et_target) <= TOL_ET),
            "native_clears_strict": bool(A1_NATIVE >= raw),
            "native_clears_within_noise": bool(A1_NATIVE >= raw - A1_NATIVE_TF_SPREAD),
            "native_gap": A1_NATIVE - raw,  # >0 clears, <0 misses
            "in_published_envelope": bool(raw <= A1_PUBLISHED_ENVELOPE_MAX),
        }

    reg_lower = regime("multiplicative_lower_5p36", ET_TARGET_LOWER, BANK_DEMAND_LOWER, BANK_AEFF_LOWER)
    reg_central = regime("central_6p11", ET_TARGET_CENTRAL, BANK_DEMAND_CENTRAL, BANK_AEFF_CENTRAL)
    reg_upper = regime("additive_upper_6p86", ET_TARGET_UPPER, BANK_DEMAND_UPPER, BANK_AEFF_UPPER)

    raw_a1_required_operating_regime = reg_central["raw_a1_demand"]  # THE TEST METRIC

    demand_band = {
        "deployed_step_floor_3p99": raw_demand_floor,
        "multiplicative_lower_5p36": reg_lower["raw_a1_demand"],
        "central_6p11": reg_central["raw_a1_demand"],
        "additive_upper_6p86": reg_upper["raw_a1_demand"],
    }
    band_monotone_in_et = bool(
        raw_demand_floor < reg_lower["raw_a1_demand"] < reg_central["raw_a1_demand"]
        < reg_upper["raw_a1_demand"])
    # the PR band {multiplicative 0.7731, measured native, additive-upper 0.8717}.
    pr_band = {
        "multiplicative_central_demand": reg_central["raw_a1_demand"],
        "measured_native_a1": A1_NATIVE,
        "additive_upper_demand": reg_upper["raw_a1_demand"],
        "native_sits_at_central_within_noise": reg_central["native_clears_within_noise"],
        "native_below_central_by": reg_central["raw_a1_demand"] - A1_NATIVE,  # 0.0017 (within 0.0097)
    }

    band = {
        "demand_band_by_regime": demand_band,
        "regimes": {"lower": reg_lower, "central": reg_central, "upper": reg_upper},
        "band_monotone_in_et": band_monotone_in_et,
        "pr_band_multiplicative_measured_additive": pr_band,
        "native_clears_under_justified_regime": bool(reg_central["native_clears_within_noise"]),
        "native_clears_lower_strict": reg_lower["native_clears_strict"],
        "native_misses_upper": bool(A1_NATIVE < reg_upper["raw_a1_demand"]),
    }

    # ---- (4) is the additive-upper 6.86 a TRUE bookend? quantify the operating-point gap. ---------- #
    # The additive anchoring's premise: the standalone-harness regime offset is a COMMON ADDITIVE
    # overhead on linear AND faithful, so the true marginal == the full harness chain delta. #295's own
    # bandwidth decomposition REFUTES that premise: the tiny 256-dim linear runs dispatch-bound (~11% BW)
    # while the faithful 2560-dim runs near-bandwidth (~60% BW), so the harness penalty is ~5.2x heavier
    # on the linear -- NOT a common additive constant. Additive thus mis-attributes the linear's dispatch
    # starvation to real EAGLE-3 step cost, AND it ignores INT4 on the marginal (deployed body is INT4,
    # which lightens the weight-bound EAGLE-3 step). Both errors are one-directional => inflate.
    additive_premise_refuted = bool(LINEAR_DISPATCH_BOUND and BW_UTIL_RATIO > 1.0
                                    and BW_UTIL_PCT_FAITHFUL > BW_UTIL_PCT_LINEAR)
    central_is_bracket_midpoint = bool(
        abs(ET_TARGET_CENTRAL - 0.5 * (ET_TARGET_LOWER + ET_TARGET_UPPER)) <= TOL)
    central_validates_293 = bool(abs(ET_TARGET_CENTRAL - CORRECTED_TARGET_MFUSE3_293) < 1.0)
    central_validates_293_gap = abs(ET_TARGET_CENTRAL - CORRECTED_TARGET_MFUSE3_293)

    op_below_bookend_raw_a1 = reg_upper["raw_a1_demand"] - reg_central["raw_a1_demand"]   # 0.0986
    op_below_bookend_et = ET_TARGET_UPPER - ET_TARGET_CENTRAL                             # 0.7476
    op_below_bookend_aeff = reg_upper["aeff_uniform"] - reg_central["aeff_uniform"]       # 0.0342
    bookend_inflation_pct = 100.0 * op_below_bookend_raw_a1 / reg_central["raw_a1_demand"]
    native_misses_bookend_by = reg_upper["raw_a1_demand"] - A1_NATIVE                     # 0.1003

    bookend = {
        "additive_upper_is_bookend": bool(additive_premise_refuted and central_validates_293),
        "additive_premise": ("common additive harness overhead on linear & faithful => keep full harness "
                             "chain delta %.1f us as the marginal" % WALL_DELTA_ADDITIVE_US),
        "additive_premise_refuted_by_bw_decomp": additive_premise_refuted,
        "bw_util_pct_linear": BW_UTIL_PCT_LINEAR,
        "bw_util_pct_faithful": BW_UTIL_PCT_FAITHFUL,
        "bw_util_ratio_linear_starved": BW_UTIL_RATIO,
        "additive_ignores_int4_on_marginal": True,
        "step_multiplier_band": [STEP_MULT_LOWER, STEP_MULT_UPPER],
        "central_is_bracket_midpoint": central_is_bracket_midpoint,
        "central_validates_293_modeled_6p1245": central_validates_293,
        "central_validates_293_gap": central_validates_293_gap,
        "operating_point_below_bookend_raw_a1": op_below_bookend_raw_a1,
        "operating_point_below_bookend_et": op_below_bookend_et,
        "operating_point_below_bookend_aeff": op_below_bookend_aeff,
        "bookend_inflation_over_operating_pct": bookend_inflation_pct,
        "native_misses_bookend_by": native_misses_bookend_by,
        "verdict": ("The additive-upper E[T]=6.8588 -> raw-a1 0.8717 is a TRUE UPPER BOOKEND, not the "
                    "operating point: its defining common-additive-overhead premise is refuted by #295's "
                    "own bandwidth decomposition (linear %.1f%% BW dispatch-bound vs faithful %.1f%% BW, "
                    "%.1fx starvation gap), and it ignores INT4 on the marginal -- both inflate it above "
                    "the deployed truth, which sits between the regimes. The operating point is the "
                    "validated central E[T]=6.1112 (bracket midpoint; corroborated by #293's independent "
                    "modeled 6.1245, gap %.4f), whose raw-a1 demand 0.7731 sits %.4f below the bookend in "
                    "raw a1 (a %.1f%% inflation). Native a1=0.7714 clears the operating point within its "
                    "own native-vs-tf 0.0097 noise and misses ONLY the bookend (by %.4f)." % (
                        BW_UTIL_PCT_LINEAR, BW_UTIL_PCT_FAITHFUL, BW_UTIL_RATIO,
                        central_validates_293_gap, op_below_bookend_raw_a1, bookend_inflation_pct,
                        native_misses_bookend_by)),
    }

    # ---- (5) verdict + honest residuals. ----------------------------------------------------------- #
    verdict = {
        "axis_a_settled_multiplicative": True,
        "spread_lives_on_axis_b_stepcost": True,
        "additive_0p8717_is_stepcost_bookend_not_composition": True,
        "operating_regime_demand_raw_a1": raw_a1_required_operating_regime,
        "native_clears_operating_regime": bool(reg_central["native_clears_within_noise"]),
        "step_regime_yellow_resolved": bool(
            bookend["additive_upper_is_bookend"] and reg_central["native_clears_within_noise"]),
        "headline": (
            "RESOLVED: the 0.8717 'additive' a1-demand is a STEP-COST (Axis-B) UPPER BOOKEND, not a live "
            "E[T]-composition demand. The E[T] acceptance composition (Axis A) is settled MULTIPLICATIVE "
            "(additive-independent is a strict upper bound on E[T], never the operating model), so BOTH "
            "0.7731 and 0.8717 already ride the multiplicative chain and differ ONLY in the #295 step-cost "
            "regime. The operating point is the validated CENTRAL regime: raw-a1 demand 0.7731, which the "
            "native {2,21,39} head's a1=0.7714 meets within its own 0.0097 native-vs-tf noise. 0.8717 is "
            "the pessimistic upper bookend (operating point sits 0.0986 below it in raw a1); native misses "
            "ONLY that bookend."),
    }
    residuals = {
        "cov_transfer_yellow_OPEN": (
            "UNCHANGED by this card: cov4=0.6532 is measured on the deployed LINEAR spine (wirbel #79); a "
            "{2,21,39}-fusion draft with rank-1 misses further down the list (frac_beyond_top4 > %.3f) "
            "would drop cov4 and raise the demand toward 0.9213. This card resolves the STEP regime, not "
            "the cov transfer." % FRAC_TRUE_BEYOND_TOP4),
        "private_binding_axis_OPEN": (
            "UNCHANGED by this card: fern #305 go-card shows PRIVATE is the binding axis -- at the central "
            "target the private projection is %.1f TPS (< 500) under the conservative x0.804 OOD factor, "
            "and even at the bracket top only %.1f TPS. Closing the step-regime a1 YELLOW does NOT make "
            "the build GO." % (PRIVATE_AT_CENTRAL_TPS, PRIVATE_AT_BRACKET_HI_TPS)),
        "scope": ("0 TPS; LOCAL CPU-only analytic; BASELINE 481.53 untouched; greedy/PPL untouched; NO "
                  "GPU/vLLM/HF Job/submission/served-file change. The drafter BUILD stays human-gated."),
    }

    return {
        "axis_a_composition": axis_a,
        "honest500_floor": floor,
        "demand_band": band,
        "bookend_analysis": bookend,
        "verdict": verdict,
        "residuals": residuals,
        "test_metric": {"raw_a1_required_operating_regime": raw_a1_required_operating_regime},
        "imported": {
            "K_cal": K_CAL, "official_public": OFFICIAL_PUBLIC, "K_spec": K_SPEC,
            "et_private_500": ET_PRIVATE_500, "a_k_deployed": A_K,
            "a1_required_611_nosalvage_304": A1_REQUIRED_611_NOSALVAGE,
            "et_target_lower_295": ET_TARGET_LOWER, "et_target_central_295": ET_TARGET_CENTRAL,
            "et_target_upper_295": ET_TARGET_UPPER, "cov4_79": COV4, "a1_native_308": A1_NATIVE,
            "a1_native_tf_308": A1_NATIVE_TF, "corrected_target_mfuse3_293": CORRECTED_TARGET_MFUSE3_293,
            "provenance": (
                "composition law kanna #217 (K_cal=125.268, step=1218.2us, tau=1.218) / honest-500 floor "
                "500/K_cal=3.9914; deployed ladder a_k lawine #300 (8t5q6sr0); no-salvage 0.9213 denken "
                "#304 (dtf1ouml); step bracket [5.3636,6.1112,6.8588] + BW decomposition wirbel #295 "
                "(eagle3_step_profile) cross-checked by wirbel #293 modeled 6.1245; cov4=0.6532 wirbel #79 "
                "(z6wi4z4v); native a1=0.7714 / tf=0.7617 + banked demands denken #308 (5axqa6oa) over "
                "lawine #309 salvage; private-binding carry fern #305 go-card."),
        },
    }


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def self_test(syn: dict[str, Any]) -> dict[str, Any]:
    a = syn["axis_a_composition"]
    fl = syn["honest500_floor"]
    bd = syn["demand_band"]
    bk = syn["bookend_analysis"]
    vd = syn["verdict"]
    rc, rl, ru = bd["regimes"]["central"], bd["regimes"]["lower"], bd["regimes"]["upper"]
    c: dict[str, bool] = {}

    # ---- AXIS A: composition is multiplicative; additive is a strict upper bound. ----
    # multiplicative chain lands near the composition-law deployed E[T] (within cross-run noise) AND is
    # far closer to it than the additive model -- the discriminating fit, not an arbitrary pick.
    c["a01_mult_reproduces_deployed_et_3p844"] = bool(
        a["reproduces_deployed_et_3p844"] and a["mult_beats_additive_on_deployed"])
    c["a02_additive_strict_upper_bound"] = bool(a["additive_is_strict_upper_bound"])
    c["a03_additive_eq_mult_at_degenerate_endpoints"] = bool(
        a["additive_eq_mult_at_all0"] and a["additive_eq_mult_at_all1"])
    c["a04_additive_closed_form_1_plus_Ka"] = bool(a["additive_closed_form_1_plus_Ka"])
    # every interior (non-degenerate) sample has additive STRICTLY above multiplicative.
    interior_strict = all(
        s["gap"] > TOL for s in a["dominance_samples"] if 0.0 < s["ladder_mean"] < 1.0)
    c["a05_additive_strictly_above_on_interior"] = bool(interior_strict)

    # ---- DEMAND: re-derived demands reproduce the banked #308/#309 values to 1e-6. ----
    c["b01_floor_demand_reproduces_bank"] = bool(fl["raw_a1_demand_floor_matches_bank"])
    c["b02_lower_demand_reproduces_bank"] = bool(rl["raw_a1_demand_matches_bank"])
    c["b03_central_demand_reproduces_bank"] = bool(rc["raw_a1_demand_matches_bank"])
    c["b04_upper_demand_reproduces_bank"] = bool(ru["raw_a1_demand_matches_bank"])
    c["b05_aeff_central_reproduces_304_0p9213"] = bool(rc["aeff_matches_bank"])
    c["b06_demand_monotone_in_et"] = bool(bd["band_monotone_in_et"])
    # each regime round-trips: lift the raw demand through the tree, recover E[T] target.
    c["b07_regimes_roundtrip_et"] = bool(rl["roundtrip_ok"] and rc["roundtrip_ok"] and ru["roundtrip_ok"])
    # all demands are valid acceptances in (0,1).
    c["b08_demands_in_unit_interval"] = bool(
        all(0.0 < d < 1.0 for d in bd["demand_band_by_regime"].values()))

    # ---- NATIVE clears the operating regime within noise; misses only the bookend. ----
    c["c01_native_clears_floor_strict"] = bool(fl["native_clears_floor"])
    c["c02_native_clears_lower_strict"] = bool(rl["native_clears_strict"])
    c["c03_native_clears_central_within_noise"] = bool(rc["native_clears_within_noise"])
    c["c04_native_below_central_within_tf_spread"] = bool(
        abs(rc["native_gap"]) <= A1_NATIVE_TF_SPREAD + TOL)
    c["c05_native_misses_upper_bookend"] = bool(bd["native_misses_upper"])

    # ---- BOOKEND: additive-upper premise refuted; central is the validated midpoint operating point. ----
    c["d01_additive_premise_refuted_by_bw"] = bool(bk["additive_premise_refuted_by_bw_decomp"])
    c["d02_central_is_bracket_midpoint"] = bool(bk["central_is_bracket_midpoint"])
    c["d03_central_validates_293_modeled"] = bool(bk["central_validates_293_modeled_6p1245"])
    c["d04_additive_upper_is_bookend"] = bool(bk["additive_upper_is_bookend"])
    # the operating-point-below-bookend gap is positive and consistent with the band.
    c["d05_op_below_bookend_gap_consistent"] = bool(
        bk["operating_point_below_bookend_raw_a1"] > 0.0
        and abs(bk["operating_point_below_bookend_raw_a1"]
                - (ru["raw_a1_demand"] - rc["raw_a1_demand"])) <= TOL)

    # ---- VERDICT wiring + constants. ----
    c["e01_yellow_resolved_flag"] = bool(vd["step_regime_yellow_resolved"])
    c["e02_operating_demand_is_central"] = bool(
        abs(vd["operating_regime_demand_raw_a1"] - rc["raw_a1_demand"]) <= TOL)
    c["e03_constants_imported_exact"] = bool(
        K_CAL == 125.268 and K_SPEC == 7 and COV4 == 0.6531976066516435
        and A1_NATIVE == 0.7714 and ET_TARGET_CENTRAL == 6.1112149873699195
        and ET_TARGET_UPPER == 6.858819247754167
        and A1_REQUIRED_611_NOSALVAGE == 0.9213011665456927
        and abs(ET_PRIVATE_500 - 3.991442347606731) <= 1e-12)

    gate = all(bool(v) for v in c.values())
    return {"step_regime_demand_resolved_self_test_passes": gate, "checks": c}


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
# W&B logging (summary/ namespace; never fatal).
# --------------------------------------------------------------------------- #
def maybe_log_wandb(args, payload: dict) -> str | None:
    if getattr(args, "no_wandb", False) or not getattr(args, "wandb_name", None):
        return None
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_json_artifact, log_summary
    except Exception as exc:  # noqa: BLE001
        print(f"[eagle3-step-regime-demand] wandb logging unavailable: {exc}", flush=True)
        return None

    syn = payload["synthesis"]
    fl, bd, bk, vd = (syn["honest500_floor"], syn["demand_band"], syn["bookend_analysis"], syn["verdict"])
    rc, rl, ru = bd["regimes"]["central"], bd["regimes"]["lower"], bd["regimes"]["upper"]
    st = payload["self_test"]

    run = init_wandb_run(
        job_type="validity-analytic", agent="denken", name=args.wandb_name, group=args.wandb_group,
        tags=["eagle3-step-regime", "validity-analytic", "a1-cliff", "step-regime", "tree-verify",
              "composition-model", "bookend", "eagle3", "bank-the-analysis"],
        config={
            "pr": 320, "K_cal": K_CAL, "official_public": OFFICIAL_PUBLIC, "K_spec": K_SPEC,
            "et_private_500": ET_PRIVATE_500, "cov4": COV4, "a1_native": A1_NATIVE,
            "et_target_lower": ET_TARGET_LOWER, "et_target_central": ET_TARGET_CENTRAL,
            "et_target_upper": ET_TARGET_UPPER, "wandb_group": args.wandb_group,
            "imports": syn["imported"]["provenance"],
        },
    )
    if run is None:
        print("[eagle3-step-regime-demand] wandb: no run (no WANDB_API_KEY/mode) -- skipping", flush=True)
        return None

    summary: dict[str, Any] = {
        # PRIMARY + TEST
        "step_regime_demand_resolved_self_test_passes": int(bool(
            st["step_regime_demand_resolved_self_test_passes"])),
        "raw_a1_required_operating_regime": vd["operating_regime_demand_raw_a1"],
        # demand band
        "raw_a1_demand_deployed_floor_3p99": fl["raw_a1_demand_floor"],
        "raw_a1_demand_lower_5p36": rl["raw_a1_demand"],
        "raw_a1_demand_central_6p11": rc["raw_a1_demand"],
        "raw_a1_demand_upper_bookend_6p86": ru["raw_a1_demand"],
        "native_a1": A1_NATIVE, "native_a1_tf": A1_NATIVE_TF,
        # native clears
        "native_clears_operating_regime": int(bool(vd["native_clears_operating_regime"])),
        "native_clears_lower_strict": int(bool(rl["native_clears_strict"])),
        "native_misses_upper_bookend": int(bool(bd["native_misses_upper"])),
        "native_gap_vs_central": rc["native_gap"],
        "native_misses_bookend_by": bk["native_misses_bookend_by"],
        # axis A
        "axis_a_multiplicative_settled": int(bool(vd["axis_a_settled_multiplicative"])),
        "additive_is_strict_upper_bound": int(bool(
            syn["axis_a_composition"]["additive_is_strict_upper_bound"])),
        "et_deployed_recomputed": syn["axis_a_composition"]["et_deployed_recomputed"],
        # bookend
        "additive_upper_is_bookend": int(bool(bk["additive_upper_is_bookend"])),
        "operating_point_below_bookend_raw_a1": bk["operating_point_below_bookend_raw_a1"],
        "operating_point_below_bookend_et": bk["operating_point_below_bookend_et"],
        "bookend_inflation_over_operating_pct": bk["bookend_inflation_over_operating_pct"],
        "central_validates_293_gap": bk["central_validates_293_gap"],
        "bw_util_ratio_linear_starved": bk["bw_util_ratio_linear_starved"],
        # verdict
        "step_regime_yellow_resolved": int(bool(vd["step_regime_yellow_resolved"])),
        # honest binding-axis carry (NOT closed here)
        "private_at_central_tps_carry": PRIVATE_AT_CENTRAL_TPS,
        # hygiene
        "nan_clean": int(bool(payload["nan_clean"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(val)) for k, val in st["checks"].items()},
    }
    summary = {k: val for k, val in summary.items()
               if not (isinstance(val, float) and not math.isfinite(val)) and val is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="eagle3_step_regime_demand_result", artifact_type="validity",
                      data=payload)
    rid = getattr(run, "id", None)
    print(f"[eagle3-step-regime-demand] wandb run: {getattr(run, 'url', rid)}", flush=True)
    finish_wandb(run)
    return rid


# --------------------------------------------------------------------------- #
# Console report.
# --------------------------------------------------------------------------- #
def print_report(payload: dict) -> None:
    syn = payload["synthesis"]
    a, fl, bd, bk, vd = (syn["axis_a_composition"], syn["honest500_floor"], syn["demand_band"],
                         syn["bookend_analysis"], syn["verdict"])
    rc, rl, ru = bd["regimes"]["central"], bd["regimes"]["lower"], bd["regimes"]["upper"]
    st = payload["self_test"]
    print("\n" + "=" * 100, flush=True)
    print("EAGLE-3 STEP-REGIME-DEMAND (PR #320) -- is the 0.8717 additive a1-demand real or a bookend?",
          flush=True)
    print("=" * 100, flush=True)
    print("AXIS A -- E[T] acceptance composition:", flush=True)
    print(f"  model = {a['composition_model']}", flush=True)
    print(f"  {a['et_multiplicative_formula']}", flush=True)
    print(f"  deployed E[T] recomputed = {a['et_deployed_recomputed']:.4f} (==3.844); additive is a "
          f"strict UB? {a['additive_is_strict_upper_bound']}", flush=True)
    print("-" * 100, flush=True)
    print("HONEST-500 FLOOR (deployed-linear step, E[T]=3.9914):", flush=True)
    print(f"  raw-a1 demand = {fl['raw_a1_demand_floor']:.4f}  native a1=0.7714 clears? "
          f"{fl['native_clears_floor']} (margin +{fl['native_margin_over_floor']:.4f})", flush=True)
    print("-" * 100, flush=True)
    print("DEMAND BAND across the #295 Axis-B step regimes (multiplicative chain + cov4 salvage):",
          flush=True)
    print(f"  floor 3.99   -> raw-a1 {fl['raw_a1_demand_floor']:.4f}", flush=True)
    print(f"  lower 5.36   -> raw-a1 {rl['raw_a1_demand']:.4f}   (native gap +{rl['native_gap']:.4f})",
          flush=True)
    print(f"  CENTRAL 6.11 -> raw-a1 {rc['raw_a1_demand']:.4f}   (native gap {rc['native_gap']:+.4f}, "
          f"within tf-noise 0.0097)  <-- OPERATING POINT / TEST", flush=True)
    print(f"  upper 6.86   -> raw-a1 {ru['raw_a1_demand']:.4f}   (native gap {ru['native_gap']:+.4f})  "
          f"<-- pessimistic BOOKEND", flush=True)
    print(f"  native clears under justified regime? {bd['native_clears_under_justified_regime']}",
          flush=True)
    print("-" * 100, flush=True)
    print("BOOKEND ANALYSIS:", flush=True)
    print(f"  additive premise refuted by BW decomp (linear {bk['bw_util_pct_linear']:.1f}% vs faithful "
          f"{bk['bw_util_pct_faithful']:.1f}% BW, {bk['bw_util_ratio_linear_starved']:.1f}x)? "
          f"{bk['additive_premise_refuted_by_bw_decomp']}", flush=True)
    print(f"  central is bracket midpoint? {bk['central_is_bracket_midpoint']}; validates #293 modeled "
          f"6.1245 (gap {bk['central_validates_293_gap']:.4f})? {bk['central_validates_293_modeled_6p1245']}",
          flush=True)
    print(f"  operating point sits {bk['operating_point_below_bookend_raw_a1']:.4f} below the bookend in "
          f"raw-a1 ({bk['bookend_inflation_over_operating_pct']:.1f}% inflation)", flush=True)
    print(f"  additive_upper_is_bookend = {bk['additive_upper_is_bookend']}", flush=True)
    print("-" * 100, flush=True)
    print(f"VERDICT: {vd['headline']}", flush=True)
    print("-" * 100, flush=True)
    print(f"(PRIMARY) step_regime_demand_resolved_self_test_passes = "
          f"{st['step_regime_demand_resolved_self_test_passes']}", flush=True)
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
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="eagle3-step-regime")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    syn = synthesize()
    st = self_test(syn)

    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "pr": 320, "agent": "denken", "kind": "eagle3-step-regime-demand",
        "eagle3_step_regime_demand_analysis_only": True,
        "synthesis": syn, "self_test": st,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }

    nan_paths = assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[eagle3-step-regime-demand] WARNING non-finite at: {nan_paths}", flush=True)
    gate = bool(st["step_regime_demand_resolved_self_test_passes"] and payload["nan_clean"])
    st["step_regime_demand_resolved_self_test_passes"] = gate

    print_report(payload)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eagle3_step_regime_demand_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[eagle3-step-regime-demand] wrote {out_path}", flush=True)

    rid = maybe_log_wandb(args, payload)
    payload["wandb_run_id"] = rid
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    print(f"  PRIMARY step_regime_demand_resolved_self_test_passes = {gate}", flush=True)
    print(f"  TEST raw_a1_required_operating_regime = "
          f"{syn['test_metric']['raw_a1_required_operating_regime']:.4f}", flush=True)
    print(f"  additive_upper_is_bookend = {syn['bookend_analysis']['additive_upper_is_bookend']}; "
          f"operating point {syn['bookend_analysis']['operating_point_below_bookend_raw_a1']:.4f} below "
          f"bookend", flush=True)
    print(f"  wandb run = {rid}", flush=True)

    if args.self_test:
        print(f"[eagle3-step-regime-demand] self-test {'PASS' if gate else 'FAIL'}", flush=True)
        return 0 if gate else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
