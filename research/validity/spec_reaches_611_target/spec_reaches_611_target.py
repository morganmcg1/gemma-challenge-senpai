#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #304 (denken) -- does hitting wirbel #295's 6.11 target require BREAKING the a1 cliff?

THE UNRECONCILED KEYSTONE
-------------------------
Two halves of the EAGLE-3 build target were settled in parallel and never
reconciled:

  * NUMERATOR (denken lane): kanna #289 (`fi34s269`) + denken #297 (`vo2ir6ca`)
    settled the per-position acceptance SPEC as "lift j>=2 conditional acceptance
    to ~0.91 while KEEPING a1 at the cliff (a1>=0.73)". That UNIFORM (prompt-
    invariant) spec yields E[T] ~ 4.92 (the a1-only ceiling 4.910, targeting
    fern #281's 4.966 / wirbel #290's 4.9029 free-lever step).

  * DENOMINATOR (wirbel lane): wirbel #295 (`c334qaqu`, MERGED) MEASURED the
    heavier EAGLE-3 fusion-draft step (regime-corrected multiplier ~2.95x,
    validating #293's conservative 6.1245 proxy) and settled the step-corrected
    E[T] TARGET at 6.1112 central, bracket [5.3636, 6.8588] -- NOT the optimistic
    ~5.0. The heavier fusion step RAISED the E[T] the build must hit from
    4.9029 (free-lever step) to 6.11 (fusion step).

THE PROBLEM: the #289 spec was written for the 4.9029/4.966 target. Its E[T] ~ 4.92
is ~1.19 SHORT of wirbel #295's 6.11. To hit 6.11 via the survival chain-product
E[T] = 1 + sum_{k=1..7} prod_{j=1..k} a_j, the chain-product survival-sum must
reach 5.11 -- which, at K=7, needs per-position acceptance ~0.921 UNIFORM across
ALL positions, INCLUDING a1. But #289's spec KEEPS a1 at the cliff (0.73) and
lifts only j>=2. So the heavier fusion step may DEMAND breaking the position-1
cliff (a1: 0.73 -> ~0.92), not merely keeping it -- a strictly harder drafter
than the #289 spec describes.

WHAT THIS LEG DOES (CPU-analytic chain-product inversion)
---------------------------------------------------------
1. Reproduce the #289 spec E[T] = 1 + sum cumprod([0.73, 0.91x6]) ~ 4.92,
   reconciling the 4.910 a1-only ceiling (self-test (a): the deployed a_k
   reproduce E[T] = 3.8512, denken #297 convention, resid < 1e-3).
2. INVERT the chain-product to the per-position acceptance REQUIRED to hit
   wirbel #295's targets {5.3636, 6.1112, 6.8588} (+ #293's 6.1245), under two
   policies:
     - Policy A (UNIFORM): all a_k = a*; solve sum_{k=1..7} a*^k = E*-1.
     - Policy B (CLIFF-KEPT): a1=0.73 fixed, lift a_{j>=2}=b*; solve
       E[T] = 1 + 0.73 * sum_{k=0..6} b*^k. The MAX at b*=1.0 is
       1 + 0.73*7 = 6.11 EXACTLY -> if the central target 6.1112 > 6.11, then
       6.11 is INFEASIBLE while keeping a1 at the cliff => the cliff MUST break.
3. Verdict: kanna289_spec_reaches_611 (NO), heavier_step_demands_a1_break (does
   6.11 REQUIRE a1 >> 0.73), a1_required_for_611 (the position-1 acceptance the
   6.11 target demands).

HONEST SCOPE
------------
0 TPS. This reconciles the per-position acceptance SPEC against the step-corrected
E[T] TARGET; it does NOT produce a built drafter and does NOT change the served
checkpoint (BASELINE stays 481.53). (a) 6.11 is wirbel #295's step-corrected
target -- conditional on the fusion-step profile, not a trained drafter; (b) the
required-acceptance is an ANALYTIC inversion of the chain-product (the per-position
vector that WOULD yield 6.11), NOT a measured drafter capability -- whether a1->0.92
is TRAINABLE is kanna #294's viability lane (we price WHAT 6.11 needs, kanna prices
WHETHER it's reachable); (c) the launch gate stays land #245's MEASURED >=500 at
lambda_hat>=0.9780 AND PPL<=2.42 AND VRAM<=24GB, human-approval-gated.
NOT a launch. NOT a build. NOT open2. No served-file change.

PRIMARY metric  spec_reaches_611_self_test_passes
TEST    metric  heavier_step_demands_a1_break
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

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Imported anchors -- DO NOT re-derive. Import EXACTLY, UNCHANGED.
# --------------------------------------------------------------------------- #
# wirbel #295 (`c334qaqu`) step-corrected target (the DENOMINATOR we reconcile to)
ET_TARGET_CENTRAL_295 = 6.1112149873699195      # corrected_central
ET_TARGET_LOWER_295 = 5.363610726985671         # corrected_multiplicative_lower
ET_TARGET_UPPER_295 = 6.858819247754167         # corrected_additive_upper
MULTIPLIER_FAITHFUL_295 = 1.745557050577551     # byte_ratio_x_bwfrac (regime ~2.95x)
# wirbel #293 (`abhoog1x`) conservative target (validated by #295)
ET_TARGET_CONSERVATIVE_293 = 6.1245

# kanna #289 (`fi34s269`) per-position acceptance SPEC (the NUMERATOR we test)
A_K_289_DEPLOYED = [0.72925, 0.75956, 0.79298, 0.82280, 0.83487, 0.83579, 0.84649]
CLIFF_VALUE_SPEC = 0.73          # "keep a1 >= 0.73" spec floor (position-1 cliff)
LIFT_VALUE_SPEC = 0.91           # "lift j>=2 -> ~0.91" spec lift
A1_ONLY_CEILING_289 = 4.910      # #289 a1-only lift ceiling (< 4.966)
PUBLIC_ET_FLOOR_281 = 4.966      # fern #281 public E[T] floor
STEP_BANKED_ET_290 = 4.9029      # wirbel #290 free-lever-step banked E[T]

# denken #297 (`vo2ir6ca`) chain-product anchors (the machinery + self-test target)
ET_DEPLOYED_297 = 3.851185944363104   # E[T] = 1 + sum cumprod(deployed a_k); rounds to 3.8512
ET_DEPLOYED_297_ROUND = 3.8512
A1_LOW_QUARTILE_297 = 0.6550          # low-quartile (hard-prompt) binding a1
LINEAR_CAP_119 = 3.8445               # denken #119 LINEAR drafter E[T] cap

# kanna #217 (`vgovdrjc`) official composition anchors (context)
OFFICIAL_TPS_217 = 481.53
E_T_ANCHOR_217 = 3.844
K_CAL_217 = 125.268
STEP_US_217 = 1218.2
TAU_217 = 1.218

K_SPEC = 7                # num_speculative_tokens (chain depth)
E_T_MAX = K_SPEC + 1      # 8.0 theoretical ceiling

# wirbel #295 banked whole-run counters that reproduce the deployed E[T] exactly
# (denken #282/#297 convention; step-1 exact cross-check, resid ~0).
BANKED_ACCEPTED_PER_POS = [12452.0, 9458.0, 7500.0, 6171.0, 5152.0, 4306.0, 3645.0]
BANKED_NUM_DRAFTS = 17075.0

TOL_RT = 1e-6     # inversion round-trip tolerance
TOL_REPRO = 1e-3  # deployed-E[T] reproduction tolerance


# --------------------------------------------------------------------------- #
# Chain-product E[T] (denken #297 convention): E[T] = 1 + sum_{k=1..K} prod a_j.
# --------------------------------------------------------------------------- #
def et_from_ak(a_k: list[float]) -> float:
    """Survival chain-product E[T] = 1 + sum_{k=1..K} prod_{j=1..k} a_j."""
    cp = np.cumprod(np.asarray(a_k, dtype=float))
    return 1.0 + float(cp.sum())


def et_from_counters(accepted_per_pos: list[float], num_drafts: float,
                     K: int = K_SPEC) -> float:
    """E[T] from raw per-position accepted counters (G(m)=app[m-1]/nd; exact)."""
    app = list(accepted_per_pos)
    G = [app[m - 1] / num_drafts for m in range(1, K + 1)]   # survival G(1..K) == cumprod
    return 1.0 + float(sum(G))


def et_cliffkept(a1: float, b: float, K: int = K_SPEC) -> float:
    """Cliff-KEPT E[T]: a1 fixed, a_{j>=2}=b. == 1 + a1 * sum_{k=0..K-1} b^k."""
    return 1.0 + a1 * float(sum(b ** k for k in range(K)))


def uniform_sum(a: float, K: int = K_SPEC) -> float:
    """sum_{k=1..K} a^k (geometric); E[T]_uniform = 1 + uniform_sum(a)."""
    return float(sum(a ** k for k in range(1, K + 1)))


def cliffkept_poly(b: float, K: int = K_SPEC) -> float:
    """sum_{k=0..K-1} b^k ; E[T]_cliffkept = 1 + a1 * cliffkept_poly(b)."""
    return float(sum(b ** k for k in range(K)))


def _bisect(fn, target: float, lo: float, hi: float, iters: int = 200) -> float:
    """Solve fn(x)=target for monotone-increasing fn on [lo, hi] via bisection."""
    flo, fhi = fn(lo) - target, fn(hi) - target
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    # expand hi if the target is above fn(hi) (used to expose b* > 1 infeasibility)
    grow = 0
    while fhi < 0.0 and grow < 80:
        hi *= 1.5
        fhi = fn(hi) - target
        grow += 1
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fmid = fn(mid) - target
        if fmid == 0.0:
            return mid
        if (fmid > 0.0) == (fhi > 0.0):
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# Inversion: per-position acceptance REQUIRED to hit a step-corrected E[T] target.
# --------------------------------------------------------------------------- #
def invert_uniform(et_star: float, K: int = K_SPEC) -> dict[str, Any]:
    """Policy A: all a_k = a*; solve sum_{k=1..K} a*^k = et_star - 1."""
    need = et_star - 1.0
    feasible = need <= K + 1e-12          # uniform_sum(1.0)=K is the max in [0,1]
    a_star = _bisect(lambda a: uniform_sum(a, K), need, 0.0, 1.0)
    et_rt = et_from_ak([a_star] * K)      # round-trip through the chain-product
    return {
        "policy": "uniform",
        "et_target": et_star,
        "survival_sum_needed": need,
        "a_uniform_required": a_star,
        "feasible_in_unit": bool(feasible and a_star <= 1.0 + 1e-9),
        "roundtrip_et": et_rt,
        "roundtrip_resid": abs(et_rt - et_star),
    }


def invert_cliffkept(et_star: float, a1: float = CLIFF_VALUE_SPEC,
                     K: int = K_SPEC) -> dict[str, Any]:
    """Policy B: a1 fixed at the cliff, lift a_{j>=2}=b*; solve for b*.

    E[T] = 1 + a1 * sum_{k=0..K-1} b^k. Max at b=1.0 is 1 + a1*K. If et_star
    exceeds that ceiling, b* > 1.0 (INFEASIBLE while keeping a1 at the cliff).
    """
    poly_needed = (et_star - 1.0) / a1
    max_et = et_cliffkept(a1, 1.0, K)               # 1 + a1*K
    b_star = _bisect(lambda b: cliffkept_poly(b, K), poly_needed, 0.0, 1.0)
    feasible = et_star <= max_et + 1e-12
    et_rt = et_cliffkept(a1, b_star, K)
    # minimum a1 that makes et_star feasible at the perfect b=1.0 corner
    min_a1_at_bfull = (et_star - 1.0) / K
    return {
        "policy": "cliff_kept",
        "et_target": et_star,
        "a1_fixed": a1,
        "cliff_poly_needed": poly_needed,            # sum_{k=0..K-1} b^k required
        "b_jge2_required": b_star,
        "max_et_cliffkept": max_et,                  # 1 + a1*K (b*=1.0 ceiling)
        "cliffkept_can_reach": bool(feasible),
        "b_star_le_one": bool(b_star <= 1.0 + 1e-9),
        "min_a1_at_bfull": min_a1_at_bfull,          # a1 needed at the b=1.0 corner
        "roundtrip_et": et_rt,
        "roundtrip_resid": abs(et_rt - et_star),
    }


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    # ---------- step 1: reproduce deployed + #289 spec E[T] ---------- #
    et_deployed_from_ak = et_from_ak(A_K_289_DEPLOYED)
    et_deployed_from_counters = et_from_counters(BANKED_ACCEPTED_PER_POS, BANKED_NUM_DRAFTS)
    spec289_ak = [CLIFF_VALUE_SPEC] + [LIFT_VALUE_SPEC] * (K_SPEC - 1)   # [0.73, 0.91x6]
    spec289_et = et_from_ak(spec289_ak)

    # ---------- step 2: invert to the step-corrected targets ---------- #
    targets = {
        "lower_5p3636": ET_TARGET_LOWER_295,
        "central_6p1112": ET_TARGET_CENTRAL_295,
        "conservative_6p1245_293": ET_TARGET_CONSERVATIVE_293,
        "upper_6p8588": ET_TARGET_UPPER_295,
    }
    uniform = {tag: invert_uniform(et) for tag, et in targets.items()}
    cliffkept = {tag: invert_cliffkept(et) for tag, et in targets.items()}
    # cliff-kept ceiling with the DEPLOYED a1 (0.72925) -- even lower than the spec 0.73
    max_et_cliffkept_deployed = et_cliffkept(A_K_289_DEPLOYED[0], 1.0)

    central = "central_6p1112"
    a_uniform_required_611 = uniform[central]["a_uniform_required"]
    max_et_cliffkept = cliffkept[central]["max_et_cliffkept"]       # 1 + 0.73*7 = 6.11
    cliffkept_can_reach_611 = cliffkept[central]["cliffkept_can_reach"]

    # ---------- step 3: verdicts ---------- #
    spec_shortfall_vs_611 = ET_TARGET_CENTRAL_295 - spec289_et
    kanna289_spec_reaches_611 = bool(spec289_et >= ET_TARGET_CENTRAL_295)
    # the heavier step DEMANDS breaking the a1 cliff iff 6.11 cannot be reached
    # while keeping a1 at the cliff (even at the perfect b*=1.0 corner) AND the
    # uniform-required a1 sits well above the deployed cliff.
    heavier_step_demands_a1_break = bool(
        (not cliffkept_can_reach_611)
        and (a_uniform_required_611 > CLIFF_VALUE_SPEC + 0.05))
    # the position-1 acceptance the 6.11 target demands = the uniform per-position
    # acceptance (a1 must rise to a* under a uniform lift).
    a1_required_for_611 = a_uniform_required_611

    verdict = _verdict(kanna289_spec_reaches_611, cliffkept_can_reach_611,
                       heavier_step_demands_a1_break)
    handoff = _handoff(spec289_et, spec_shortfall_vs_611, max_et_cliffkept,
                       cliffkept_can_reach_611, a1_required_for_611,
                       heavier_step_demands_a1_break)

    return {
        "step1_reproduce": {
            "et_deployed_from_ak": et_deployed_from_ak,
            "et_deployed_from_counters": et_deployed_from_counters,
            "et_deployed_resid_vs_297": abs(et_deployed_from_ak - ET_DEPLOYED_297),
            "et_counters_resid_vs_297": abs(et_deployed_from_counters - ET_DEPLOYED_297),
            "spec289_ak": spec289_ak,
            "spec289_et": spec289_et,
            "spec289_et_reconciles_a1ceiling_resid": abs(spec289_et - A1_ONLY_CEILING_289),
        },
        "step2_invert": {
            "targets": targets,
            "uniform": uniform,
            "cliff_kept": cliffkept,
            "max_et_cliffkept_spec073": max_et_cliffkept,
            "max_et_cliffkept_deployed_0p72925": max_et_cliffkept_deployed,
            "a_uniform_required_611": a_uniform_required_611,
            "b_jge2_required_611_cliffkept": cliffkept[central]["b_jge2_required"],
            "cliffkept_can_reach_611": cliffkept_can_reach_611,
            "min_a1_at_bfull_611": cliffkept[central]["min_a1_at_bfull"],
        },
        "step3_verdict": {
            "spec289_et": spec289_et,
            "spec_shortfall_vs_611": spec_shortfall_vs_611,
            "kanna289_spec_reaches_611": kanna289_spec_reaches_611,
            "cliffkept_can_reach_611": cliffkept_can_reach_611,
            "heavier_step_demands_a1_break": heavier_step_demands_a1_break,
            "a1_required_for_611": a1_required_for_611,
            "a1_deployed": A_K_289_DEPLOYED[0],
            "a1_low_quartile_297": A1_LOW_QUARTILE_297,
            "a1_lift_factor_vs_deployed": a1_required_for_611 / A_K_289_DEPLOYED[0],
        },
        "context": {
            "linear_cap_119": LINEAR_CAP_119,
            "public_et_floor_281": PUBLIC_ET_FLOOR_281,
            "step_banked_et_290": STEP_BANKED_ET_290,
            "multiplier_faithful_295": MULTIPLIER_FAITHFUL_295,
            "official_tps_217": OFFICIAL_TPS_217,
            "k_spec": K_SPEC,
            "e_t_max": E_T_MAX,
        },
        "verdict": verdict,
        "handoff_line": handoff,
    }


def _verdict(reaches: bool, cliffkept_reaches: bool, demands_break: bool) -> str:
    if reaches:
        return "SPEC-289-REACHES-611"                      # (not expected)
    if demands_break:
        return "HEAVIER-STEP-DEMANDS-A1-CLIFF-BREAK"        # cliff-kept caps below 6.11
    if cliffkept_reaches:
        return "SPEC-289-SHORT-BUT-CLIFF-KEPT-LIFT-REACHES-611"
    return "SPEC-289-SHORT-611-FEASIBILITY-AMBIGUOUS"


def _handoff(spec289_et: float, shortfall: float, max_cliffkept: float,
             cliffkept_reaches: bool, a1_req: float, demands_break: bool) -> str:
    reach = ("reaches" if spec289_et >= ET_TARGET_CENTRAL_295
             else f"falls {shortfall:.2f} short of")
    if demands_break:
        build = (f"a1->~{a1_req:.2f}, strictly harder than #289 (cliff-kept lift caps at "
                 f"{max_cliffkept:.4f} < {ET_TARGET_CENTRAL_295:.4f} even at b*=1.0)")
    elif cliffkept_reaches:
        build = "unchanged from #289 (cliff-kept lift feasibly reaches 6.11)"
    else:
        build = f"a1->~{a1_req:.2f} (cliff-kept feasibility ambiguous)"
    return (
        f"kanna #289's per-position spec (a1>=0.73, j>=2->0.91) yields E[T] ~ {spec289_et:.4f} "
        f"and {reach} wirbel #295's step-corrected 6.11 target; hitting 6.11 "
        f"{'REQUIRES breaking the a1 cliff to ~%.2f' % a1_req if demands_break else 'survives cliff-kept'} "
        f"(cliff-kept lift caps at {max_cliffkept:.4f}), so the EAGLE-3 build-spec under the "
        f"heavier fusion step is {build}, reconciling the numerator acceptance spec with the "
        f"denominator-corrected target. 0 TPS; 6.11 is wirbel #295's step-PROFILE target "
        f"(not a trained drafter); the required-acceptance is an ANALYTIC chain-product "
        f"inversion (kanna #294 prices whether a1->0.92 is TRAINABLE); gate stays land #245's "
        f"MEASURED >=500. NOT a launch. NOT a build. NOT open2."
    )


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def self_test(syn: dict[str, Any]) -> dict[str, Any]:
    s1, s2, s3 = syn["step1_reproduce"], syn["step2_invert"], syn["step3_verdict"]
    checks: dict[str, bool] = {}

    # (a) chain-product reproduces deployed E[T]=3.8512 from deployed a_k (resid < 1e-3)
    checks["a_deployed_ak_reproduces_3p8512"] = s1["et_deployed_resid_vs_297"] < TOL_REPRO
    checks["a_deployed_counters_reproduce_3p8512"] = s1["et_counters_resid_vs_297"] < TOL_REPRO

    # (b) #289 spec [0.73, 0.91x6] reconstructs ~4.92 (reconciles 4.910)
    checks["b_spec289_reconstructs_4p92"] = abs(s1["spec289_et"] - 4.92) < 0.01
    checks["b_spec289_near_a1_only_ceiling"] = s1["spec289_et_reconciles_a1ceiling_resid"] < 0.02

    # (c) max_et_cliffkept = 1 + 0.73*7 computed exactly; uniform & cliff-kept
    #     inversions round-trip (plug a*/b* back -> E[T]* within 1e-6).
    checks["c_max_et_cliffkept_is_1_plus_073x7"] = (
        abs(s2["max_et_cliffkept_spec073"] - (1.0 + CLIFF_VALUE_SPEC * K_SPEC)) < 1e-12)
    checks["c_uniform_roundtrips"] = all(
        s2["uniform"][t]["roundtrip_resid"] < TOL_RT for t in s2["uniform"])
    checks["c_cliffkept_roundtrips"] = all(
        s2["cliff_kept"][t]["roundtrip_resid"] < TOL_RT for t in s2["cliff_kept"])
    # cliff-kept closed form == chain-product through [a1, b x6]
    checks["c_cliffkept_matches_chainproduct"] = abs(
        et_cliffkept(CLIFF_VALUE_SPEC, 0.91)
        - et_from_ak([CLIFF_VALUE_SPEC] + [0.91] * (K_SPEC - 1))) < 1e-12

    # (d) all required a_k in [0,1] for FEASIBLE targets, flagged infeasible if > 1
    def _unit_or_flagged(inv: dict[str, Any], key: str, feasible_key: str) -> bool:
        v = inv[key]
        if inv[feasible_key]:
            return -1e-9 <= v <= 1.0 + 1e-9
        return v > 1.0 - 1e-9         # infeasible => the solved root sits at/above 1.0
    checks["d_uniform_unit_or_flagged"] = all(
        _unit_or_flagged(s2["uniform"][t], "a_uniform_required", "feasible_in_unit")
        for t in s2["uniform"])
    checks["d_cliffkept_unit_or_flagged"] = all(
        _unit_or_flagged(s2["cliff_kept"][t], "b_jge2_required", "b_star_le_one")
        for t in s2["cliff_kept"])

    # NaN-clean over the reported scalars
    scalars = [
        s1["et_deployed_from_ak"], s1["spec289_et"], s2["a_uniform_required_611"],
        s2["max_et_cliffkept_spec073"], s3["spec_shortfall_vs_611"],
        s3["a1_required_for_611"], s3["a1_lift_factor_vs_deployed"],
    ]
    checks["d_nan_clean"] = all(math.isfinite(float(x)) for x in scalars)

    # (e) imported anchors EXACT and UNCHANGED
    checks["e_constants_imported_exact"] = (
        ET_TARGET_CENTRAL_295 == 6.1112149873699195
        and ET_TARGET_LOWER_295 == 5.363610726985671
        and ET_TARGET_UPPER_295 == 6.858819247754167
        and ET_TARGET_CONSERVATIVE_293 == 6.1245
        and CLIFF_VALUE_SPEC == 0.73 and LIFT_VALUE_SPEC == 0.91
        and A1_ONLY_CEILING_289 == 4.910 and ET_DEPLOYED_297_ROUND == 3.8512
        and A1_LOW_QUARTILE_297 == 0.6550 and LINEAR_CAP_119 == 3.8445
        and STEP_BANKED_ET_290 == 4.9029 and PUBLIC_ET_FLOOR_281 == 4.966
        and OFFICIAL_TPS_217 == 481.53 and E_T_ANCHOR_217 == 3.844
        and K_SPEC == 7
    )

    # (f) the leg carries the 0-TPS + 6.11-is-a-profile-target +
    #     required-acceptance-is-analytic-not-trained caveats (verdict + handoff).
    checks["f_carries_caveats"] = bool(
        "0 TPS" in syn["handoff_line"]
        and "ANALYTIC" in syn["handoff_line"]
        and "NOT a launch" in syn["handoff_line"]
        and "NOT a build" in syn["handoff_line"])

    gate = bool(all(checks.values()))
    return {"spec_reaches_611_self_test_passes": gate, "checks": checks}


# --------------------------------------------------------------------------- #
def _assert_nan_clean(payload: Any, path: str = "result") -> list[str]:
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

    walk(payload, path)
    return bad


def _print_report(syn: dict, st: dict) -> None:
    s1, s2, s3 = syn["step1_reproduce"], syn["step2_invert"], syn["step3_verdict"]
    print("\n" + "=" * 92, flush=True)
    print("SPEC-REACHES-611-TARGET (PR #304, denken) -- chain-product inversion, CPU-only",
          flush=True)
    print("=" * 92, flush=True)
    print("  (1) REPRODUCE", flush=True)
    print(f"      deployed E[T] from a_k    = {s1['et_deployed_from_ak']:.6f}  "
          f"(297 {ET_DEPLOYED_297:.6f}; resid {s1['et_deployed_resid_vs_297']:.2e})", flush=True)
    print(f"      #289 spec [0.73,0.91x6]   = {s1['spec289_et']:.6f}  "
          f"(~4.92; a1-only ceiling 4.910, resid {s1['spec289_et_reconciles_a1ceiling_resid']:.4f})",
          flush=True)
    print("-" * 92, flush=True)
    print(f"  (2) INVERT  (chain-product survival-sum E[T]-1 -> per-position acceptance)", flush=True)
    print(f"      {'target':<26} {'E[T]*':>9} {'a*unif':>8} {'feas':>5}   "
          f"{'b*cliff':>8} {'cliffOK':>7}", flush=True)
    for tag in ("lower_5p3636", "central_6p1112", "conservative_6p1245_293", "upper_6p8588"):
        u, c = s2["uniform"][tag], s2["cliff_kept"][tag]
        print(f"      {tag:<26} {u['et_target']:>9.4f} {u['a_uniform_required']:>8.4f} "
              f"{str(u['feasible_in_unit']):>5}   {c['b_jge2_required']:>8.4f} "
              f"{str(c['cliffkept_can_reach']):>7}", flush=True)
    print(f"      max_et_cliffkept (a1=0.73, b*=1.0) = {s2['max_et_cliffkept_spec073']:.6f}  "
          f"(deployed a1=0.72925 -> {s2['max_et_cliffkept_deployed_0p72925']:.6f})", flush=True)
    print("-" * 92, flush=True)
    print("  (3) VERDICT", flush=True)
    print(f"      kanna289_spec_reaches_611      = {s3['kanna289_spec_reaches_611']}  "
          f"(shortfall {s3['spec_shortfall_vs_611']:.4f})", flush=True)
    print(f"      cliffkept_can_reach_611        = {s3['cliffkept_can_reach_611']}", flush=True)
    print(f"      heavier_step_demands_a1_break  = {s3['heavier_step_demands_a1_break']}", flush=True)
    print(f"      a1_required_for_611            = {s3['a1_required_for_611']:.4f}  "
          f"(deployed {s3['a1_deployed']:.4f}, low-q {s3['a1_low_quartile_297']:.4f}, "
          f"x{s3['a1_lift_factor_vs_deployed']:.3f})", flush=True)
    print("-" * 92, flush=True)
    print(f"  PRIMARY spec_reaches_611_self_test_passes = "
          f"{st['spec_reaches_611_self_test_passes']}", flush=True)
    for k, v in st["checks"].items():
        print(f"          - {k}: {v}", flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print("=" * 92, flush=True)
    print(f"\n  HAND-OFF: {syn['handoff_line']}\n", flush=True)


def _maybe_log_wandb(args, payload: dict) -> str | None:
    if not getattr(args, "wandb_name", None):
        return None
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[spec-reaches-611] wandb logging unavailable: {exc}", flush=True)
        return None

    syn, st = payload["synthesis"], payload["self_test"]
    s1, s2, s3 = syn["step1_reproduce"], syn["step2_invert"], syn["step3_verdict"]
    run = init_wandb_run(
        job_type="spec-reaches-611-target",
        agent="denken",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["spec-reaches-611-target", "eagle3-build-spec", "chain-product-inversion",
              "per-position-acceptance", "validity", "zero-tps"],
        config={
            "pr": 304, "analysis_only": True, "K_spec": K_SPEC,
            "et_target_central_295": ET_TARGET_CENTRAL_295,
            "et_target_lower_295": ET_TARGET_LOWER_295,
            "et_target_upper_295": ET_TARGET_UPPER_295,
            "et_target_conservative_293": ET_TARGET_CONSERVATIVE_293,
            "cliff_value_spec": CLIFF_VALUE_SPEC, "lift_value_spec": LIFT_VALUE_SPEC,
            "et_deployed_297": ET_DEPLOYED_297, "linear_cap_119": LINEAR_CAP_119,
            "public_et_floor_281": PUBLIC_ET_FLOOR_281, "step_banked_et_290": STEP_BANKED_ET_290,
            "official_tps_217": OFFICIAL_TPS_217, "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[spec-reaches-611] wandb: no run (no WANDB_API_KEY/mode) -- skipping", flush=True)
        return None

    central = s2["cliff_kept"]["central_6p1112"]
    summary: dict[str, Any] = {
        # PRIMARY + TEST
        "spec_reaches_611_self_test_passes":
            int(bool(st["spec_reaches_611_self_test_passes"])),
        "heavier_step_demands_a1_break": int(bool(s3["heavier_step_demands_a1_break"])),
        "kanna289_spec_reaches_611": int(bool(s3["kanna289_spec_reaches_611"])),
        "a1_required_for_611": s3["a1_required_for_611"],
        # step 1
        "et_deployed_from_ak": s1["et_deployed_from_ak"],
        "et_deployed_resid_vs_297": s1["et_deployed_resid_vs_297"],
        "spec289_et": s1["spec289_et"],
        # step 2
        "a_uniform_required_611": s2["a_uniform_required_611"],
        "b_jge2_required_611_cliffkept": s2["b_jge2_required_611_cliffkept"],
        "max_et_cliffkept_spec073": s2["max_et_cliffkept_spec073"],
        "max_et_cliffkept_deployed": s2["max_et_cliffkept_deployed_0p72925"],
        "cliffkept_can_reach_611": int(bool(s2["cliffkept_can_reach_611"])),
        "min_a1_at_bfull_611": s2["min_a1_at_bfull_611"],
        "a_uniform_required_lower": s2["uniform"]["lower_5p3636"]["a_uniform_required"],
        "a_uniform_required_upper": s2["uniform"]["upper_6p8588"]["a_uniform_required"],
        "cliffkept_can_reach_lower": int(bool(
            s2["cliff_kept"]["lower_5p3636"]["cliffkept_can_reach"])),
        "cliffkept_can_reach_upper": int(bool(
            s2["cliff_kept"]["upper_6p8588"]["cliffkept_can_reach"])),
        # step 3
        "spec_shortfall_vs_611": s3["spec_shortfall_vs_611"],
        "a1_lift_factor_vs_deployed": s3["a1_lift_factor_vs_deployed"],
        "a1_deployed": s3["a1_deployed"],
        "a1_low_quartile_297": s3["a1_low_quartile_297"],
        # verdict
        "verdict_demands_a1_break": int(
            syn["verdict"] == "HEAVIER-STEP-DEMANDS-A1-CLIFF-BREAK"),
        **{f"selftest_{k}": int(bool(v)) for k, v in st["checks"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="spec_reaches_611_target_result", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    rid = getattr(run, "id", None)
    print(f"[spec-reaches-611] wandb logged (run {rid}): {summary}", flush=True)
    return rid


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="spec-reaches-611-target")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    syn = synthesize()
    st = self_test(syn)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 304, "agent": "denken",
        "kind": "spec-reaches-611-target", "synthesis": syn, "self_test": st,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[spec-reaches-611] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn, st)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "spec_reaches_611_target_results.json"

    wid = None
    if not args.no_wandb:
        wid = _maybe_log_wandb(args, payload)
    payload["wandb_run_id"] = wid
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[spec-reaches-611] wrote {out_path}  (wandb run {wid})", flush=True)

    if args.self_test:
        ok = st["spec_reaches_611_self_test_passes"] and payload["nan_clean"]
        print(f"[spec-reaches-611] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
