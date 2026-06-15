#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #309 (student lawine) -- Does the M=8 verify-tree relax denken #304's a1-cliff-break demand?

THE RECONCILIATION (two merged results in apparent tension)
-----------------------------------------------------------
denken #304 (dtf1ouml): hitting wirbel #295's step-corrected E[T]=6.11 target DEMANDS breaking the
    position-1 cliff -- the uniform family needs a_uniform = 0.9213 (cliff-kept caps at 6.11 < 6.1112,
    so a_1 MUST rise from the deployed 0.72925 to ~0.9213). #304 reads this as a DRAFT-level demand.

lawine #300 (8t5q6sr0, mine): on the deployed LINEAR drafter, the M=8 verify-TREE already recovers
    the position-1 cliff. The deployed-effective collapse is position-1 HELD (c_1=1.0), residual
    c_deep=0.97135 on j>=2, BECAUSE the tree salvages the rank-2+ matches the width-1 spine rejects
    (ubel #258 / wirbel #79 z6wi4z4v: the raw 0.804 collapse is a j=1 discrimination loss; the
    rank-2..W branches catch cov_W of the rank-1 misses).

THE QUESTION
------------
#304's a_uniform=0.9213 is the EFFECTIVE per-position acceptance that enters E[T]. But a verify-TREE
lifts the EFFECTIVE position-1 acceptance ABOVE the raw draft top-1 via rank-2+ salvage:

        c1_eff(a1_draft) = a1_draft + (1 - a1_draft) * cov_W        [the salvage operator]

(standard tree-acceptance identity: true token is rank-1 (prob a1_draft) OR in the rank-2..W branch
when rank-1 missed (prob (1-a1_draft)*cov_W)). So the RAW draft a1 needed to reach the EFFECTIVE
0.9213 is LOWER than 0.9213 -- the tree does part of the cliff break. This card inverts the salvage to
report `a1_draft_required_after_tree_salvage` and compares it to #304's no-salvage 0.9213.

WHY THIS IS THE VERIFY-SIDE COMPLEMENT TO #304'S DRAFT-SIDE CARD
---------------------------------------------------------------
On PUBLIC prompts the deployed per-position conditionals (A_K) ~= the raw width-1 spine conditionals
(A_PUB_RAW, |diff| <= 0.0066, wirbel #79 cross-check), i.e. the deployed PUBLIC path is effectively
width-1: it does NOT salvage rank-2+. So adding the rank-2+ branch (the wirbel #79 measured
cov_W=0.6532 at W=4, the build-relevant verify tree) is a REAL new lift, not double-counting. The
verify-tree's salvage is the lever that turns #304's draft-level a1->0.92 into a raw-draft a1 well
below 0.92.

THE KEY CAVEAT (stated, not hidden)
-----------------------------------
cov_W is MEASURED on the deployed LINEAR spine's rank-2+ candidates (wirbel #79). A {2,21,39}-fusion
EAGLE-3 draft has a DIFFERENT rank-2+ candidate distribution and a distinct rejection geometry. If the
fusion draft's rank-1 misses land FURTHER down the rank list (higher frac_true_beyond_topW), cov_W
drops and the salvage weakens toward #304's hard demand. We bracket this honestly across W in {2,3,4}
and at zero-transfer (cov=0 reproduces #304's 0.9213 exactly).

LOCAL CPU-only analytic card over banked acceptance ladders + banked rank-coverage. No GPU / vLLM /
model forward / training / HF Job / submission / served-file change. NOT a launch. BASELINE stays
481.53 (this leg adds 0 TPS). Greedy/PPL untouched. PRIMARY = self-test.

PRIMARY metric  tree_salvage_a1_self_test_passes
TEST    metric  a1_draft_required_after_tree_salvage  (raw fusion-draft a1 to reach E[T]=6.11 after
                                                       the M=8 tree salvage; primary cov_W = cov4)

Reproduce:
    cd target/ && .venv/bin/python research/validity/eagle3_tree_salvage_a1/eagle3_tree_salvage_a1.py \\
        --self-test --wandb_group eagle3-tree-salvage-a1 --wandb_name lawine/eagle3-tree-salvage-a1
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
# Imported fleet anchors (imported EXACTLY, UNCHANGED; this leg re-derives none).
# --------------------------------------------------------------------------- #
K_CAL = 125.268                 # composition anchor: official = K_cal * E[T] (kanna #269)
OFFICIAL_PUBLIC = 481.53        # #52 deployed public TPS (linear MTP K=7)
K_SPEC = 7                      # num_speculative_tokens (manifest)
E_T_MAX = float(K_SPEC + 1)     # 8.0 theoretical ceiling at lambda=1
PRIVATE_BAR_TPS = 500.0         # the binding launch gate (land #245)
ET_PRIVATE_500 = PRIVATE_BAR_TPS / K_CAL  # 3.99146: E[T] for 500 TPS on the DEPLOYED-linear step

# Deployed-effective per-position conditional ladder a_k (lawine #300 / kanna #289 / denken #297).
A_K = [
    0.7292532942898975, 0.759556697719242, 0.7929794882639035, 0.8228,
    0.8348727920920435, 0.8357919254658385, 0.8464932652113331,
]

# ubel #258 BANKED raw width-1 spine ladders (public x private). The position-1 PRIVATE collapse the
# M=8 tree recovers in lawine #300 (c_1 raw ratio 0.822 -> 1.0).
A_PUB_RAW = [
    0.728739760479042, 0.7589764102641635, 0.7924989076194682, 0.821702519412012,
    0.8342716929825772, 0.8352594665096346, 0.8472621220149911,
]
A_PRIV_RAW = [
    0.5991304839661618, 0.6893085026081717, 0.7464222790849783, 0.7736626492721749,
    0.8477031613381011, 0.862379683274926, 0.885712826843481,
]

# wirbel #79 (run z6wi4z4v) MEASURED rank-coverage on the deployed-linear stack (the ubel #258
# rank-2+ structure the PR cites). cov_W = cumulative P(true token caught at rank <= W | rank-1 miss).
# Full greedy path, 16,524 records, align_bad=0, byteshark-cross-validated to 0.85% on rho2.
COV_W = {2: 0.4165047789261015, 3: 0.5714507731758489, 4: 0.6531976066516435}
RHO_MARGINAL = {2: 0.4165047789261015, 3: 0.2655480090557997, 4: 0.19075249320036264}
RANK1_TOP1_MEASURED = 0.7335390946502057      # raw rank-1 top-1 acceptance (deployed stack)
RHO2_BY_DEPTH = [                              # per-depth rho2 (depth 0 == j=1, the cliff position)
    0.3967749261866909, 0.4306826178747361, 0.4133545310015898, 0.4284603421461897,
    0.4351851851851852, 0.44471153846153844, 0.4095826893353941,
]
FRAC_TRUE_BEYOND_TOP4 = 0.3468023933483565    # irreducible width-4 miss mass on the linear spine

# denken #304 (run dtf1ouml): the no-salvage demand this card relaxes.
ET_TARGET_611 = 6.1112149873699195            # wirbel #295 step-corrected E[T] target (#304 central)
A1_REQUIRED_611_NOSALVAGE = 0.9213011665456927  # #304 uniform a required for 6.11 (no tree salvage)
A1_DEPLOYED = 0.72925                          # #304 deployed a_1 (the cliff)
ET_TARGET_6245 = 6.1245                        # #304 conservative (wirbel #293 honest step-overhead)
A1_REQUIRED_6245_NOSALVAGE = 0.9219520928012865

# lawine #300 carry (PR-requested log).
RHO_PRIV_E3 = 0.9421228821714434              # #300 EAGLE-3 deployed-effective private/public ratio

# Primary salvage width: the M=8 verify tree branches width-4 at the root (the cliff position, where
# wirbel #79 found every branch up to 4 clears its GEMM cost), so cov_W(primary) = cov4.
PRIMARY_W = 4
A1_SWEEP = [0.65, 0.70, 0.73, 0.75, 0.80, 0.85, 0.90, 0.92]

TOL = 1e-9
TOL_ET = 1e-3
TOL_RATIO = 1e-6


# --------------------------------------------------------------------------- #
# Core: survival ladder + E[T] (reused from lawine #300, UNCHANGED math).
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


def tps_at(et: float) -> float:
    """official = K_cal * E[T] (125.268 * 3.844 = 481.53)."""
    return K_CAL * et


def uniform_profile(a_u: float) -> list[float]:
    return [a_u] * K_SPEC


def _bisect(f, lo: float, hi: float, target: float, iters: int = 200) -> float:
    """monotone-increasing f: return x in [lo,hi] with f(x)~=target."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# The salvage operator (the one new piece of machinery).
# --------------------------------------------------------------------------- #
def tree_recovered(base: float, cov: float) -> float:
    """Effective acceptance after an M-node verify tree salvages rank-2..W candidates.

    base = raw top-1 (rank-1) acceptance; cov = P(true token at rank <=W | rank-1 missed). The true
    token is accepted at position-1 if it is rank-1 (prob base) OR in the rank-2..W branch when
    rank-1 missed (prob (1-base)*cov). Clipped to a valid acceptance [0,1].
    """
    return min(1.0, max(0.0, base + (1.0 - base) * cov))


def a1_draft_for_effective(a_eff_target: float, cov: float) -> float:
    """Invert the salvage: raw draft a1 so that tree_recovered(a1, cov) == a_eff_target.

    a1 = (a_eff_target - cov) / (1 - cov); clipped to [0,1]. cov->0 returns a_eff_target (no salvage,
    reproducing #304); cov->1 returns 0 (a perfect-coverage tree needs no draft acceptance at all).
    """
    if cov >= 1.0 - TOL:
        return 0.0
    return min(1.0, max(0.0, (a_eff_target - cov) / (1.0 - cov)))


def effective_uniform_for_et(et_target: float) -> float:
    """uniform EFFECTIVE per-position acceptance whose ladder hits E[T]=et_target (re-derives #304)."""
    return _bisect(lambda a: et_from_cond(uniform_profile(a)), 0.0, 1.0, et_target)


# --------------------------------------------------------------------------- #
# Synthesis (steps 1-4).
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    # ---- STEP 1: formalize the linear-path salvage; reproduce #300 c_1=1.0. -- #
    # The raw width-1 spine's position-1 PRIVATE/PUBLIC ratio (the cliff lawine #300 reports the tree
    # recovers): c1_raw_ratio = a_priv_raw / a_pub_raw.
    c1_raw_ratio = A_PRIV_RAW[0] / A_PUB_RAW[0]
    # rank-2+ private coverage REQUIRED at j=1 to recover the ratio to 1.0 (private a1 == public a1):
    #   a_priv_raw + (1 - a_priv_raw) * cov_priv_req = a_pub_raw
    cov_priv_req = (A_PUB_RAW[0] - A_PRIV_RAW[0]) / (1.0 - A_PRIV_RAW[0])
    # feed it back through the operator -> the recovered position-1 PRIVATE acceptance and its ratio.
    a1_priv_recovered = tree_recovered(A_PRIV_RAW[0], cov_priv_req)
    c1_recovered_ratio = a1_priv_recovered / A_PUB_RAW[0]   # MUST == 1.0 (reproduces #300 c_1=1.0)
    # grounding: the required private coverage must fit inside the MEASURED public rank-2 coverage
    # (private degrades OOD but cannot exceed public). Degradation factor it implies:
    ood_degradation_factor = cov_priv_req / COV_W[2]        # cov_priv_req / public cov2
    salvage_feasible_on_linear = bool(cov_priv_req <= COV_W[2] + TOL)

    step1 = {
        "c1_raw_private_public_ratio": c1_raw_ratio,
        "cov_priv_required_to_recover_c1": cov_priv_req,
        "a1_private_recovered": a1_priv_recovered,
        "c1_recovered_ratio": c1_recovered_ratio,
        "reproduces_300_c1_held": bool(abs(c1_recovered_ratio - 1.0) <= TOL_RATIO),
        "measured_public_cov2": COV_W[2],
        "ood_degradation_factor_implied": ood_degradation_factor,
        "salvage_feasible_on_linear_spine": salvage_feasible_on_linear,
        "rho_priv_e3_carry": RHO_PRIV_E3,
        "note": ("linear-path salvage: the raw width-1 spine drops position-1 to priv/pub ratio "
                 "{:.4f}; recovering it to 1.0 needs private rank-2+ coverage {:.4f} at j=1, which is "
                 "{:.4f} of the measured public cov2={:.4f} (OOD degradation >= {:.3f}) -- so the M=8 "
                 "tree's rank-2 branch alone recovers lawine #300's c_1=1.0 cliff hold."
                 .format(c1_raw_ratio, cov_priv_req, ood_degradation_factor, COV_W[2],
                         ood_degradation_factor)),
    }

    # ---- STEP 2: apply the salvage operator to a FUSION draft (a1 curve). ---- #
    # cross-check: re-derive #304's effective uniform a for E[T]=6.11 from our own ladder.
    a_eff_611 = effective_uniform_for_et(ET_TARGET_611)
    a_eff_611_matches_304 = bool(abs(a_eff_611 - A1_REQUIRED_611_NOSALVAGE) <= 1e-6)

    curves: dict[str, list[dict[str, float]]] = {}
    for w in (2, 3, 4):
        cov = COV_W[w]
        pts = []
        for a1 in A1_SWEEP:
            ceff = tree_recovered(a1, cov)
            pts.append({"a1_draft": a1, "c1_eff": ceff})
        curves[f"W{w}"] = pts
    # headline curve point: raw a1_draft = 0.73 (deployed cliff level), primary cov4.
    tree_recovered_c1_at_a1_073 = tree_recovered(0.73, COV_W[PRIMARY_W])
    curve_in_unit = all(
        0.0 <= p["c1_eff"] <= 1.0 for pts in curves.values() for p in pts
    )

    step2 = {
        "a_eff_uniform_for_611": a_eff_611,
        "a_eff_611_matches_304_0p9213": a_eff_611_matches_304,
        "salvage_operator": "c1_eff = a1_draft + (1 - a1_draft) * cov_W",
        "cov_W_used": COV_W,
        "primary_W": PRIMARY_W,
        "primary_cov_W": COV_W[PRIMARY_W],
        "effective_a1_curves": curves,
        "tree_recovered_c1_at_a1_073": tree_recovered_c1_at_a1_073,
        "curve_in_unit_interval": bool(curve_in_unit),
        "note": ("fusion-draft tree-recovered effective a1 = a1_draft + (1-a1_draft)*cov_W. At the "
                 "deployed cliff level raw a1=0.73 the primary (W=4, cov={:.4f}) tree lifts effective "
                 "a1 to {:.4f} -- already {:.4f} of #304's 0.9213 effective demand, from a raw draft "
                 "no better than today's linear spine."
                 .format(COV_W[PRIMARY_W], tree_recovered_c1_at_a1_073,
                         tree_recovered_c1_at_a1_073 / A1_REQUIRED_611_NOSALVAGE)),
    }

    # ---- STEP 3: invert -> raw a1_draft required after tree salvage. --------- #
    # target effective per-position acceptance for E[T]=6.11 is a_eff_611 (== #304's 0.9213).
    req_by_w = {w: a1_draft_for_effective(a_eff_611, COV_W[w]) for w in (2, 3, 4)}
    a1_draft_required_after_tree_salvage = req_by_w[PRIMARY_W]   # primary cov4 -> the TEST metric
    # zero-salvage must reproduce #304 exactly (cov=0).
    a1_req_zero_salvage = a1_draft_for_effective(a_eff_611, 0.0)
    reconciles_304_at_zero = bool(abs(a1_req_zero_salvage - A1_REQUIRED_611_NOSALVAGE) <= 1e-6)
    # round-trip: tree_recovered(req, cov4) back to the effective target and E[T]=6.11.
    eff_back = tree_recovered(a1_draft_required_after_tree_salvage, COV_W[PRIMARY_W])
    et_back = et_from_cond(uniform_profile(eff_back))
    roundtrip_et_611 = bool(abs(et_back - ET_TARGET_611) <= TOL_ET)

    # the deployed-step 500-TPS floor (E[T]=3.9914) variant, for completeness.
    a_eff_floor = effective_uniform_for_et(ET_PRIVATE_500)
    a1_draft_required_floor = a1_draft_for_effective(a_eff_floor, COV_W[PRIMARY_W])

    # the cliff-HELD reading: keep raw a1 at the deployed 0.73; the tree lifts effective a1 to 0.9064.
    # is E[T]=6.11 then reachable with a feasible (deepflat, <1) effective deep rate b?
    a1_eff_cliffheld = tree_recovered(0.73, COV_W[PRIMARY_W])
    # E[T] = 1 + a1_eff*(1 + b + ... + b^(K-1)); solve for b in (0,1].
    def et_cliffheld(b: float) -> float:
        geo = sum(b ** j for j in range(K_SPEC))   # 1 + b + ... + b^6
        return 1.0 + a1_eff_cliffheld * geo
    b_max = et_cliffheld(1.0)
    cliffheld_can_reach_611 = bool(b_max >= ET_TARGET_611 - TOL)
    b_required_cliffheld = (_bisect(et_cliffheld, 0.0, 1.0, ET_TARGET_611)
                            if cliffheld_can_reach_611 else float("nan"))

    salvage_relaxes_304_demand_by = A1_REQUIRED_611_NOSALVAGE - a1_draft_required_after_tree_salvage
    relaxes_by_by_w = {w: A1_REQUIRED_611_NOSALVAGE - req_by_w[w] for w in (2, 3, 4)}

    step3 = {
        "a1_draft_required_after_tree_salvage": a1_draft_required_after_tree_salvage,
        "a1_draft_required_by_W": req_by_w,
        "a1_required_611_nosalvage_304": A1_REQUIRED_611_NOSALVAGE,
        "a1_req_zero_salvage": a1_req_zero_salvage,
        "reconciles_304_at_zero_salvage": reconciles_304_at_zero,
        "roundtrip_effective_back": eff_back,
        "roundtrip_et_back": et_back,
        "roundtrip_reaches_611": roundtrip_et_611,
        "salvage_relaxes_304_demand_by": salvage_relaxes_304_demand_by,
        "salvage_relaxes_by_W": relaxes_by_by_w,
        "deployed_step_floor_et": ET_PRIVATE_500,
        "a_eff_uniform_for_floor": a_eff_floor,
        "a1_draft_required_floor": a1_draft_required_floor,
        "cliffheld_a1_eff_at_raw_073": a1_eff_cliffheld,
        "cliffheld_can_reach_611": cliffheld_can_reach_611,
        "cliffheld_b_required": b_required_cliffheld,
        "note": ("inverting the salvage: to reach #304's effective 0.9213 (E[T]=6.11) the raw fusion "
                 "draft needs a1={:.4f} (primary W=4) -- {:.4f} below #304's no-salvage 0.9213, and "
                 "only {:.4f} above the deployed linear's raw 0.73. Bracket: W=2 {:.4f}, W=3 {:.4f}, "
                 "W=4 {:.4f}; zero-salvage {:.4f}==#304. Cliff-HELD (raw a1=0.73): the tree lifts "
                 "effective a1 to {:.4f} and 6.11 is reachable at effective deep b={:.4f}<1."
                 .format(a1_draft_required_after_tree_salvage, salvage_relaxes_304_demand_by,
                         a1_draft_required_after_tree_salvage - 0.73, req_by_w[2], req_by_w[3],
                         req_by_w[4], a1_req_zero_salvage, a1_eff_cliffheld, b_required_cliffheld)),
    }

    # ---- STEP 4: verdict. --------------------------------------------------- #
    # the tree materially salvages a1 if the required raw draft a1 stays below #304's 0.9213 across
    # the whole width bracket (even the minimal W=2 root), i.e. the relaxation is not a knife-edge.
    salvages_at_primary = bool(a1_draft_required_after_tree_salvage < A1_REQUIRED_611_NOSALVAGE - 1e-6)
    salvages_across_bracket = bool(all(req_by_w[w] < A1_REQUIRED_611_NOSALVAGE - 1e-6 for w in (2, 3, 4)))
    tree_salvages_a1_for_eagle3 = bool(salvages_at_primary and salvages_across_bracket)

    verdicts = {
        "tree_salvages_a1_for_eagle3": tree_salvages_a1_for_eagle3,
        "salvages_at_primary_W4": salvages_at_primary,
        "salvages_across_width_bracket": salvages_across_bracket,
        "relaxed_demand_below_deployed_plus_10pct": bool(
            a1_draft_required_after_tree_salvage <= A1_DEPLOYED * 1.10),  # within 10% of deployed raw
        "cliffheld_breaks_via_tree_not_draft": cliffheld_can_reach_611,
    }
    step4 = {
        "verdicts": verdicts,
        "salvage_relaxes_304_demand_by": salvage_relaxes_304_demand_by,
        "caveat": ("cov_W is MEASURED on the deployed LINEAR spine (wirbel #79). A {{2,21,39}}-fusion "
                   "EAGLE-3 draft has a distinct rank-2+ candidate distribution and rejection "
                   "geometry; if its rank-1 misses fall further down the rank list (frac_beyond_top4 "
                   "> the linear spine's {:.3f}), cov_W drops and a1_draft_required rises toward "
                   "#304's 0.9213. The bracket [W2 {:.4f}, W4 {:.4f}] holds < 0.92 down to W=2; only "
                   "literal zero-transfer (cov=0 -> 0.9213) restores #304's hard demand."
                   .format(FRAC_TRUE_BEYOND_TOP4, req_by_w[2], req_by_w[4])),
        "handoff": ("GO/NO-GO: the M=8 verify-tree relaxes the EAGLE-3 drafter trainability demand "
                    "from RED (raw a1->0.9213, +26% over the deployed linear's 0.73) to GREEN-YELLOW "
                    "(raw a1->{:.4f}, just +{:.1f}% over deployed), buying {:.4f} in raw a1 under full "
                    "rank-2+ transfer (cov4=0.653); YELLOW pending confirmation that the {{2,21,39}} "
                    "fusion draft inherits the linear spine's rank-2+ coverage."
                    .format(a1_draft_required_after_tree_salvage,
                            100.0 * (a1_draft_required_after_tree_salvage - 0.73) / 0.73,
                            salvage_relaxes_304_demand_by)),
    }

    return {
        "step1_linear_salvage": step1,
        "step2_fusion_curve": step2,
        "step3_invert": step3,
        "step4_verdict": step4,
        "test_metric": {"a1_draft_required_after_tree_salvage": a1_draft_required_after_tree_salvage},
        "imported": {
            "K_cal": K_CAL, "official_public": OFFICIAL_PUBLIC, "K_spec": K_SPEC,
            "et_private_500": ET_PRIVATE_500, "a_k_deployed": A_K, "a_pub_raw": A_PUB_RAW,
            "a_priv_raw": A_PRIV_RAW, "cov_W_measured": COV_W, "rho_marginal": RHO_MARGINAL,
            "rank1_top1_measured": RANK1_TOP1_MEASURED, "rho2_by_depth": RHO2_BY_DEPTH,
            "frac_true_beyond_top4": FRAC_TRUE_BEYOND_TOP4, "et_target_611": ET_TARGET_611,
            "a1_required_611_nosalvage": A1_REQUIRED_611_NOSALVAGE, "a1_deployed": A1_DEPLOYED,
            "rho_priv_e3": RHO_PRIV_E3,
            "provenance": ("salvage operator over wirbel #79 (z6wi4z4v) measured rank-coverage "
                           "cov_W={cov}; raw spine ladders ubel #258; deployed a_k lawine #300 "
                           "(8t5q6sr0); E[T]=6.11 demand denken #304 (dtf1ouml) a_uniform=0.9213; "
                           "rho_priv_e3 carry lawine #300.".format(cov=COV_W)),
        },
    }


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY): the machinery's sanity anchors.
# --------------------------------------------------------------------------- #
def self_test(syn: dict[str, Any]) -> dict[str, Any]:
    s1, s2, s3, s4 = (syn["step1_linear_salvage"], syn["step2_fusion_curve"],
                      syn["step3_invert"], syn["step4_verdict"])
    c: dict[str, bool] = {}

    # (1) linear-path operator reproduces lawine #300's c_1=1.0 recovery EXACTLY (sanity anchor).
    c["01_linear_operator_reproduces_300_c1_1p0"] = bool(
        abs(s1["c1_recovered_ratio"] - 1.0) <= TOL_RATIO and s1["reproduces_300_c1_held"])

    # (2) the recovery is GROUNDED: required private cov2 fits inside the measured public cov2.
    c["02_c1_recovery_feasible_within_measured_cov2"] = bool(s1["salvage_feasible_on_linear_spine"])

    # (3) fusion extrapolation stays within [0,1] across the whole a1 sweep and all widths.
    c["03_fusion_curve_in_unit_interval"] = bool(s2["curve_in_unit_interval"])

    # (4) inverted a1 reconciles with #304 at ZERO salvage (cov=0 -> 0.9213).
    c["04_reconciles_304_at_zero_salvage"] = bool(s3["reconciles_304_at_zero_salvage"])

    # (5) our re-derived effective uniform for E[T]=6.11 matches #304's 0.9213 (cross-validation).
    c["05_eff_uniform_611_matches_304"] = bool(s2["a_eff_611_matches_304_0p9213"])

    # (6) primary inversion round-trips: tree_recovered(req, cov4) -> 0.9213 -> E[T]=6.11.
    c["06_primary_inversion_roundtrips_611"] = bool(s3["roundtrip_reaches_611"])

    # (7) salvage monotone: more coverage (W) => strictly smaller required raw a1.
    req = s3["a1_draft_required_by_W"]
    c["07_required_a1_monotone_in_width"] = bool(req[2] > req[3] > req[4] > 0.0)

    # (8) every required raw a1 (W=2..4) is a valid acceptance in (0,1) and below #304's 0.9213.
    c["08_required_a1_in_unit_and_below_304"] = bool(
        all(0.0 < req[w] < A1_REQUIRED_611_NOSALVAGE for w in (2, 3, 4)))

    # (9) the headline relaxation is positive and equals 0.9213 - required (consistency).
    c["09_relaxes_demand_positive_consistent"] = bool(
        s3["salvage_relaxes_304_demand_by"] > 0.0
        and abs(s3["salvage_relaxes_304_demand_by"]
                - (A1_REQUIRED_611_NOSALVAGE - s3["a1_draft_required_after_tree_salvage"])) <= TOL)

    # (10) tree_recovered_c1_at_a1_073 in [0,1] and below the effective target (sanity of the lift).
    c["10_c1_at_073_in_band"] = bool(
        0.73 <= s2["tree_recovered_c1_at_a1_073"] <= 1.0
        and s2["tree_recovered_c1_at_a1_073"] <= A1_REQUIRED_611_NOSALVAGE + 1e-9)

    # (11) carried rho_priv_e3 is the exact lawine #300 value (no silent drift).
    c["11_rho_priv_e3_carry_exact"] = bool(s1["rho_priv_e3_carry"] == RHO_PRIV_E3 == 0.9421228821714434)

    # (12) constants imported exact & unchanged.
    c["12_constants_imported_exact"] = bool(
        K_CAL == 125.268 and K_SPEC == 7 and A1_REQUIRED_611_NOSALVAGE == 0.9213011665456927
        and COV_W[4] == 0.6531976066516435 and A_K[3] == 0.8228
        and ET_TARGET_611 == 6.1112149873699195)

    gate = all(bool(v) for v in c.values())
    return {"tree_salvage_a1_self_test_passes": gate, "checks": c}


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
# W&B logging (summary/ namespace via log_summary; robust; never fatal).
# --------------------------------------------------------------------------- #
def maybe_log_wandb(args, payload: dict) -> str | None:
    if getattr(args, "no_wandb", False) or not getattr(args, "wandb_name", None):
        return None
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_json_artifact, log_summary
    except Exception as exc:  # noqa: BLE001
        print(f"[eagle3-tree-salvage-a1] wandb logging unavailable: {exc}", flush=True)
        return None

    syn = payload["synthesis"]
    s1, s2, s3, s4 = (syn["step1_linear_salvage"], syn["step2_fusion_curve"],
                      syn["step3_invert"], syn["step4_verdict"])
    st = payload["self_test"]
    v = s4["verdicts"]

    run = init_wandb_run(
        job_type="validity-analytic", agent="lawine", name=args.wandb_name, group=args.wandb_group,
        tags=["eagle3-tree-salvage-a1", "validity-analytic", "tree-verify", "a1-cliff", "eagle3",
              "rank-coverage", "reconciliation", "bank-the-analysis"],
        config={
            "pr": 309, "K_cal": K_CAL, "official_public": OFFICIAL_PUBLIC,
            "et_target_611": ET_TARGET_611, "a1_required_611_nosalvage": A1_REQUIRED_611_NOSALVAGE,
            "cov_W_measured": COV_W, "primary_W": PRIMARY_W, "a1_deployed": A1_DEPLOYED,
            "rho_priv_e3": RHO_PRIV_E3, "wandb_group": args.wandb_group,
            "imports": syn["imported"]["provenance"],
        },
    )
    if run is None:
        print("[eagle3-tree-salvage-a1] wandb: no run (no WANDB_API_KEY/mode) -- skipping", flush=True)
        return None

    summary: dict[str, Any] = {
        # PRIMARY + TEST
        "tree_salvage_a1_self_test_passes": int(bool(st["tree_salvage_a1_self_test_passes"])),
        "a1_draft_required_after_tree_salvage": s3["a1_draft_required_after_tree_salvage"],
        # PR-requested logs
        "tree_recovered_c1_at_a1_073": s2["tree_recovered_c1_at_a1_073"],
        "rho_priv_e3": RHO_PRIV_E3,
        # headline reconciliation
        "a1_required_611_nosalvage_304": A1_REQUIRED_611_NOSALVAGE,
        "salvage_relaxes_304_demand_by": s3["salvage_relaxes_304_demand_by"],
        "a1_draft_required_W2": s3["a1_draft_required_by_W"][2],
        "a1_draft_required_W3": s3["a1_draft_required_by_W"][3],
        "a1_draft_required_W4": s3["a1_draft_required_by_W"][4],
        "a1_req_zero_salvage": s3["a1_req_zero_salvage"],
        "a1_draft_required_floor_deployed_step": s3["a1_draft_required_floor"],
        "deployed_step_floor_et": ET_PRIVATE_500,
        # step-1 linear anchor
        "c1_raw_private_public_ratio": s1["c1_raw_private_public_ratio"],
        "cov_priv_required_to_recover_c1": s1["cov_priv_required_to_recover_c1"],
        "c1_recovered_ratio": s1["c1_recovered_ratio"],
        "ood_degradation_factor_implied": s1["ood_degradation_factor_implied"],
        "measured_public_cov2": COV_W[2],
        "measured_public_cov4": COV_W[4],
        # cliff-held reading
        "cliffheld_a1_eff_at_raw_073": s3["cliffheld_a1_eff_at_raw_073"],
        "cliffheld_b_required": s3["cliffheld_b_required"],
        "frac_true_beyond_top4_linear": FRAC_TRUE_BEYOND_TOP4,
        # hygiene
        "nan_clean": int(bool(payload["nan_clean"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"verdict_{k}": int(bool(val)) for k, val in v.items()},
        **{f"selftest_{k}": int(bool(val)) for k, val in st["checks"].items()},
    }
    summary = {k: val for k, val in summary.items()
               if not (isinstance(val, float) and not math.isfinite(val)) and val is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="eagle3_tree_salvage_a1_result", artifact_type="validity", data=payload)
    rid = getattr(run, "id", None)
    print(f"[eagle3-tree-salvage-a1] wandb run: {getattr(run, 'url', rid)}", flush=True)
    finish_wandb(run)
    return rid


# --------------------------------------------------------------------------- #
# Console report.
# --------------------------------------------------------------------------- #
def print_report(payload: dict) -> None:
    syn = payload["synthesis"]
    s1, s2, s3, s4 = (syn["step1_linear_salvage"], syn["step2_fusion_curve"],
                      syn["step3_invert"], syn["step4_verdict"])
    st = payload["self_test"]
    print("\n" + "=" * 100, flush=True)
    print("EAGLE-3 TREE-SALVAGE-A1 (PR #309) -- does the M=8 verify-tree relax #304's a1->0.92 demand?",
          flush=True)
    print("=" * 100, flush=True)
    print("STEP 1 -- linear-path salvage (reproduce lawine #300 c_1=1.0):", flush=True)
    print(f"  raw priv/pub pos-1 ratio = {s1['c1_raw_private_public_ratio']:.4f}; "
          f"cov_priv needed to recover -> 1.0 = {s1['cov_priv_required_to_recover_c1']:.4f}", flush=True)
    print(f"  recovered ratio = {s1['c1_recovered_ratio']:.6f} (==1.0); fits in measured public "
          f"cov2={s1['measured_public_cov2']:.4f}? {s1['salvage_feasible_on_linear_spine']}", flush=True)
    print("-" * 100, flush=True)
    print("STEP 2 -- fusion tree-recovered effective a1 = a1_draft + (1-a1_draft)*cov_W:", flush=True)
    print(f"  c1_eff at raw a1=0.73 (primary W={s2['primary_W']}, cov={s2['primary_cov_W']:.4f}) "
          f"= {s2['tree_recovered_c1_at_a1_073']:.4f}", flush=True)
    print("  a1_draft :   " + "  ".join(f"{p['a1_draft']:.2f}" for p in s2["effective_a1_curves"]["W4"]),
          flush=True)
    print("  c1_eff W4:   " + "  ".join(f"{p['c1_eff']:.2f}" for p in s2["effective_a1_curves"]["W4"]),
          flush=True)
    print("-" * 100, flush=True)
    print("STEP 3 -- invert: raw a1_draft required after tree salvage (target E[T]=6.11):", flush=True)
    req = s3["a1_draft_required_by_W"]
    print(f"  PRIMARY (W=4): a1_draft_required = {s3['a1_draft_required_after_tree_salvage']:.4f}  "
          f"vs #304 no-salvage 0.9213  (relaxes by {s3['salvage_relaxes_304_demand_by']:.4f})", flush=True)
    print(f"  bracket: W=2 {req[2]:.4f} | W=3 {req[3]:.4f} | W=4 {req[4]:.4f} | zero-salvage "
          f"{s3['a1_req_zero_salvage']:.4f} (==#304)", flush=True)
    print(f"  deployed-step floor (E[T]=3.9914): a1_draft_required = {s3['a1_draft_required_floor']:.4f}",
          flush=True)
    print(f"  cliff-HELD (raw a1=0.73): effective a1 -> {s3['cliffheld_a1_eff_at_raw_073']:.4f}; 6.11 "
          f"reachable at deep b={s3['cliffheld_b_required']:.4f}", flush=True)
    print("-" * 100, flush=True)
    print("STEP 4 -- verdict:", flush=True)
    for k, val in s4["verdicts"].items():
        print(f"  {k}: {val}", flush=True)
    print(f"  HAND-OFF: {s4['handoff']}", flush=True)
    print("-" * 100, flush=True)
    print(f"(PRIMARY) tree_salvage_a1_self_test_passes = {st['tree_salvage_a1_self_test_passes']}",
          flush=True)
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
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="eagle3-tree-salvage-a1")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    syn = synthesize()
    st = self_test(syn)

    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "pr": 309, "agent": "lawine", "kind": "eagle3-tree-salvage-a1",
        "eagle3_tree_salvage_a1_analysis_only": True,
        "synthesis": syn, "self_test": st,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }

    nan_paths = assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[eagle3-tree-salvage-a1] WARNING non-finite at: {nan_paths}", flush=True)
    gate = bool(st["tree_salvage_a1_self_test_passes"] and payload["nan_clean"])
    st["tree_salvage_a1_self_test_passes"] = gate

    print_report(payload)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eagle3_tree_salvage_a1_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[eagle3-tree-salvage-a1] wrote {out_path}", flush=True)

    rid = maybe_log_wandb(args, payload)
    payload["wandb_run_id"] = rid
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    print(f"  PRIMARY tree_salvage_a1_self_test_passes = {gate}", flush=True)
    print(f"  TEST a1_draft_required_after_tree_salvage = "
          f"{syn['step3_invert']['a1_draft_required_after_tree_salvage']:.4f}", flush=True)
    print(f"  salvage_relaxes_304_demand_by = {syn['step3_invert']['salvage_relaxes_304_demand_by']:.4f}",
          flush=True)
    print(f"  wandb run = {rid}", flush=True)

    if args.self_test:
        print(f"[eagle3-tree-salvage-a1] self-test {'PASS' if gate else 'FAIL'}", flush=True)
        return 0 if gate else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
