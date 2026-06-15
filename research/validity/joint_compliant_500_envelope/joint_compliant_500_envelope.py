#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Joint compliant-500 envelope (PR #325, fern) — CPU-only analytic integration.

THE GOVERNING QUESTION (the integrator card on top of the #192 strict greedy-identity gate)
-------------------------------------------------------------------------------------------
A compliant >500 TPS result under the #192 gate needs BOTH halves of one path:
  1. IDENTITY half — a batch-invariant int4 verify kernel whose logits are M-independent so
     greedy tokens are bit-identical to the M=1 reference. wirbel #216 (pc8g6s04) / #227
     (o674wmna) priced this kernel's throughput ceiling at lambda = 520.95 (UNBUILT). The
     deployed 481.53 split-K verify GEMM diverges 0.73% at M=8 (denken #232 nxwv6pam) -> FAILS
     the gate, so 520.95 (not 481.53) is the relevant identity ceiling.
  2. SPEED half — an EAGLE-3 E[T]-lever (deployed E[T]=3.844 -> build target 6.11) raising
     tokens/step. fern #318 (xe8ff7hq) priced its private-tax envelope: central rho=0.9421 ->
     586.1 CLEAR, worst cross-dataset rho=0.7923 -> 492.9 (-1.4% underwater).

We priced each half in isolation but never COMPOSED them under one compliant ledger. This card
does exactly that and answers: does any point in (identity-kernel ceiling x E[T]-lever
realization) clear 500 while staying greedy-identical, and WHAT is the binding constraint?

THE LEDGER (composition law; K_cal/tau/step all imported, NOTHING re-measured)
-----------------------------------------------------------------------------
    official_compliant(E[T], kernel) = min( K_cal * (E[T] * tau / step_kernel),  lambda_ceiling )

  - K_cal = 125.268, tau = 1.218, deployed step = 1218.2us (=1.2182 normalized; denken #278).
    tau/step = 0.99984 ~ 1, so the law collapses to ~ K_cal * E[T]; deployed E[T]=3.844 -> 481.53.
  - step_kernel: the IDENTITY kernel INHERITS the deployed step. wirbel #216's kernel model
    (reconciled by #235 twoceiling): the split-K reduction-order fix changes argmax-ORDER
    (determinism) NOT the GEMM step-cost or topology. So step_kernel = step_deployed = 1.2182,
    and the int4-spec ceiling 520.95 is banked AT that step. The kernel therefore enters the
    ledger ONLY as the 520.95 throughput CAP, not as a heavier denominator. (Sensitivity: if the
    batch-invariant kernel were SLOWER than deployed, BOTH the denominator grows AND the realized
    ceiling drops below 520.95 -> the cap binds EARLIER, never later. So 520.95 / deployed-step is
    the OPTIMISTIC kernel corner; the verdict here is an UPPER bound on compliance.)
  - E[T]-lever realization (the SPEED input): fern #318's honest_public_611 = 622.08 = K_cal *
    realized_public_et (realized public E[T]=4.966 = EAGLE-3 paper public acceptance length, the
    wall-/rewrite-honest realization of the free 6.11 build target). private TPS = 622.08 * rho.
  - Wall-realization throttle (#298 xp974x58): only 47.7% of a composed free gain realizes on the
    host-to-host wall (realization_ratio_487 = 0.4769). Used in the binding-map sweep to convert
    FREE E[T] -> realized E[T]: realized = deployed + 0.477*(free - deployed).
  - Rewrite gate (#314 fwqbz7zf): the K_cal*E[T] composition is valid ONLY with the #312 loopgraph
    rewrite. The eager path floors TPS at 360-481 (<500) at every E[T]; rewrite worth +57..+140.
  - Coverage gate (#316 5lnz5jgb): reaching realized public E[T]=4.966 needs the build to clear
    max_frac_beyond_top4 <= 0.2907 (vs linear demand 0.3468); the DEPLOYED deep spine alone caps
    realized E[T] at 4.9097. A build-feasibility prerequisite on the SPEED half.

DELIVERABLES (PR instructions 1-4)
----------------------------------
  1. Joint compliant TPS at central / break-even / worst-case rho, cap applied; state whether the
     CAP or the E[T]-realization binds at each rho.
  2. Binding-constraint map over E[T] in [3.844, 6.11]: at each E[T] the limiter is one of
     (a) 520.95 kernel ceiling, (b) #298 47.7% wall-realization, (c) #316 coverage clip,
     (d) #318 private worst-case. Report E[T]* where binding switches E[T]-realization -> ceiling.
  3. Compliant-500 verdict: GREEN (clears worst-case AND <=ceiling), YELLOW (clears central not
     worst), RED (ceiling caps below 500). Headroom in TPS and %.
  4. The single cheapest measurement that flips YELLOW -> GREEN.

LOCAL, CPU-ONLY, ANALYTIC. No GPU / vLLM / HF Job / submission / served-file change / official
draw. BASELINE stays 481.53; adds 0 TPS; greedy/PPL untouched. Imports fern #318 (xe8ff7hq),
wirbel #216/#227/#235 (pc8g6s04/o674wmna; int4-spec ceiling 520.95), stark #298 (xp974x58),
wirbel #314 (fwqbz7zf), lawine #316 (5lnz5jgb), denken #278 / kanna #269 (K_cal/step/tau).
Re-derives nothing. NOT a launch. NOT a build.

PRIMARY metric  joint_compliant_500_envelope_self_test_passes
TEST    metric  joint_compliant_tps_worstcase  (min(622.08*rho_worst, 520.95))
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
# Composition law (denken #278 normalized unit / kanna #269 K_cal).
# --------------------------------------------------------------------------- #
K_CAL = 125.26795005202914        # kanna #269 / #148/#169
TAU = 1.218                       # PR #325 served-fraction frame (denken #278)
STEP_DEPLOYED = 1.2182            # deployed split-K M=32 step, normalized (1218.2us)
E_T_DEPLOYED = 3.844              # deployed native accepted-tok/step (stark #266)
TAU_OVER_STEP = TAU / STEP_DEPLOYED   # 0.99984 ~ 1

# --------------------------------------------------------------------------- #
# IDENTITY half — wirbel #216 (pc8g6s04) / #227 (o674wmna), reconciled #235 (twoceiling).
# --------------------------------------------------------------------------- #
LAMBDA_CEIL = 520.9527323111674   # int4-spec batch-invariant verify ceiling (#204/#220 pqjnybbf)
INT4_SPEC_ET_TAU1 = 5.0661371742562835   # #220 int4_anchor_et1 (ceiling*step/K_cal at tau=1)
STEP_KERNEL = STEP_DEPLOYED       # kernel inherits deployed step (#235/#216: argmax-order, not step)

# --------------------------------------------------------------------------- #
# SPEED half — fern #318 (xe8ff7hq) private-tax envelope; honest_public anchor from #310.
# --------------------------------------------------------------------------- #
HONEST_PUBLIC_611 = 622.080888               # = K_cal * realized_public_et (realized=4.966)
REALIZED_PUBLIC_ET = 4.966                   # EAGLE-3 paper public acceptance length (eagle3_public_et)
E_T_BUILD_FREE = 6.1112149873699195          # free build target (wirbel #295)
RHO_CENTRAL = 0.9421228821714434             # #318 central (linear deep fidelity inherited)
RHO_BREAKEVEN = 0.8037539966988988           # #318 break-even (= 500 / 622.08)
RHO_WORST = 0.7922848664688427               # #318 PRIMARY worst-case (EAGLE-3 worst cross-dataset)
PRIV_TPS_CENTRAL_318 = 586.0766391463308     # #318 banked central (round-trip target)
PRIV_TPS_WORST_318 = 492.865273281899        # #318 banked worst-case (round-trip target)

# --------------------------------------------------------------------------- #
# Wall-realization (#298 xp974x58), rewrite (#314 fwqbz7zf), coverage (#316 5lnz5jgb).
# --------------------------------------------------------------------------- #
WALL_REALIZE = 0.47691696793341565   # #298 realization_ratio_487 (47.7% of a free gain on the wall)
EAGER_FLOOR_LO = 360.4136756340405   # #314 eager TPS @ 6.11 (#295-reconciled 3x penalty)
EAGER_FLOOR_HI = 480.57110636599805  # #314 eager TPS @ 6.11 (iso-decode)
REWRITE_GAP_LO = 57.168740617160324  # #314 rewrite worth (optimistic 1x)
REWRITE_GAP_HI = 139.5863243659595   # #314 rewrite worth (central 3x)
LOOPGRAPH_TPS_AT_611 = 500.0         # #314 loopgraph-rewrite TPS @ 6.11
COV_MAX_FRAC_BAR = 0.2906732816855498   # #316 max_frac_beyond_top4 clearing 6.11 (the build bar)
COV_LINEAR_FRAC = 0.3468023933483565    # #316 linear-spine demand (frac_true_beyond_top4)
COV_DEPLOYED_ET_CAP = 4.909733376164471 # #316 deployed deep-spine E[T] cap (position-1 salvage only)

TARGET = 500.0
BASELINE_TPS = 481.53

TOL_RT = 1e-6
TOL_ANCHOR = 0.5     # deployed reproduction tolerance (3.844 is rounded; 3.8445 reproduces 481.59)


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def official_tps(et: float, step: float = STEP_KERNEL) -> float:
    """Composition law: mu = K_cal * (E[T] * tau / step). At step=STEP_DEPLOYED, tau/step~1."""
    return K_CAL * (et * TAU / step)


def realized_et(free_et: float) -> float:
    """#298 wall-realization: only 47.7% of the free E[T] gain over deployed reaches the wall."""
    return E_T_DEPLOYED + WALL_REALIZE * (free_et - E_T_DEPLOYED)


def joint_compliant_tps(public_tps: float) -> float:
    """The joint ledger: E[T]-lever public TPS, capped at the identity-kernel ceiling."""
    return min(public_tps, LAMBDA_CEIL)


# --------------------------------------------------------------------------- #
# (1) Joint compliant TPS at central / break-even / worst-case rho.
# --------------------------------------------------------------------------- #
def corners() -> dict[str, Any]:
    rows = {}
    for name, rho in (("central", RHO_CENTRAL),
                      ("breakeven", RHO_BREAKEVEN),
                      ("worstcase", RHO_WORST)):
        uncapped = HONEST_PUBLIC_611 * rho          # the E[T]-lever realization (pre-cap)
        capped = joint_compliant_tps(uncapped)      # cap at 520.95
        cap_binds = uncapped > LAMBDA_CEIL + 1e-9
        binder = "kernel_ceiling_520p95" if cap_binds else (
            "et_realization_private_tax_rho")
        rows[name] = {
            "rho": rho,
            "uncapped_et_lever_tps": uncapped,
            "joint_compliant_tps": capped,
            "cap_binds": bool(cap_binds),
            "binding_constraint": binder,
            "clears_500": bool(capped >= TARGET - 1e-9),
            "headroom_tps": capped - TARGET,
            "headroom_pct": 100.0 * (capped - TARGET) / TARGET,
            "et_lever_headroom_above_cap_tps": max(0.0, uncapped - LAMBDA_CEIL),
        }
    return rows


# --------------------------------------------------------------------------- #
# (2) Binding-constraint map over free E[T] in [3.844, 6.11].
# --------------------------------------------------------------------------- #
def _binding_at(free_et: float, rho: float) -> dict[str, Any]:
    r_et = realized_et(free_et)
    # coverage clip (c): the DEPLOYED deep spine cannot realize beyond 4.9097 without a deeper build.
    cov_clipped = r_et > COV_DEPLOYED_ET_CAP + 1e-9
    r_et_eff = min(r_et, COV_DEPLOYED_ET_CAP)
    public_tps = official_tps(r_et_eff)
    private_tps = public_tps * rho
    capped = joint_compliant_tps(private_tps)
    cap_binds = private_tps > LAMBDA_CEIL + 1e-9
    if cap_binds:
        limiter = "a_kernel_ceiling_520p95"
    elif rho <= RHO_BREAKEVEN + 1e-9:
        # below break-even rho the private worst-case is what holds the curve under 500/520.95
        limiter = "d_private_worstcase_rho"
    elif cov_clipped:
        limiter = "c_coverage_clip_4p9097"
    else:
        limiter = "b_wall_realization_298"
    return {
        "free_et": free_et,
        "realized_et": r_et,
        "realized_et_effective": r_et_eff,
        "coverage_clipped": bool(cov_clipped),
        "public_tps": public_tps,
        "private_tps": private_tps,
        "joint_compliant_tps": capped,
        "cap_binds": bool(cap_binds),
        "limiter": limiter,
    }


def binding_map() -> dict[str, Any]:
    grid = [E_T_DEPLOYED + i * (E_T_BUILD_FREE - E_T_DEPLOYED) / 24.0 for i in range(25)]
    central_rows = [_binding_at(e, RHO_CENTRAL) for e in grid]
    worst_rows = [_binding_at(e, RHO_WORST) for e in grid]

    # E[T]* (central rho) where the binding switches E[T]-realization -> kernel ceiling:
    # solve K_cal*(realized*tau/step)*rho = LAMBDA_CEIL  ->  realized*, then invert #298 to free.
    realized_star = LAMBDA_CEIL / (K_CAL * TAU_OVER_STEP * RHO_CENTRAL)
    free_star_central = E_T_DEPLOYED + (realized_star - E_T_DEPLOYED) / WALL_REALIZE
    switch_in_range = E_T_DEPLOYED <= free_star_central <= E_T_BUILD_FREE

    # at worst rho, does the cap ever bind inside [3.844, 6.11]?
    realized_star_w = LAMBDA_CEIL / (K_CAL * TAU_OVER_STEP * RHO_WORST)
    free_star_worst = E_T_DEPLOYED + (realized_star_w - E_T_DEPLOYED) / WALL_REALIZE
    cap_binds_worst_in_range = E_T_DEPLOYED <= free_star_worst <= E_T_BUILD_FREE

    return {
        "grid_free_et": grid,
        "central_rho_rows": central_rows,
        "worst_rho_rows": worst_rows,
        "realized_et_at_cap_central": realized_star,
        "free_et_star_central": free_star_central,
        "switch_et_realization_to_ceiling_in_range": bool(switch_in_range),
        "free_et_star_worst": free_star_worst,
        "cap_binds_at_worst_rho_in_range": bool(cap_binds_worst_in_range),
        "limiter_at_build_central": central_rows[-1]["limiter"],
        "limiter_at_build_worst": worst_rows[-1]["limiter"],
        "note": (
            "central rho: below free E[T]*={:.3f} the E[T]-realization (b, #298-throttled) binds; "
            "above it the 520.95 kernel ceiling (a) binds. worst rho: the cap NEVER binds in range "
            "(free E[T]*_worst={:.3f} > 6.11), so the private worst-case (d) is the limiter "
            "throughout. The #316 coverage clip (c) at realized 4.9097 sits ABOVE the cap at "
            "central rho, so (a) pre-empts (c).".format(free_star_central, free_star_worst)),
    }


# --------------------------------------------------------------------------- #
# (3) Compliant-500 feasibility verdict.
# --------------------------------------------------------------------------- #
def verdict(crn: dict[str, Any]) -> dict[str, Any]:
    central = crn["central"]
    worst = crn["worstcase"]
    ceiling_caps_below_500 = LAMBDA_CEIL < TARGET
    if ceiling_caps_below_500:
        v = "RED"
        why = "the identity-kernel ceiling caps below 500 -- no E[T] lever can clear it."
    elif worst["clears_500"]:
        v = "GREEN"
        why = "clears 500 at the worst-case private tax AND under the kernel ceiling."
    elif central["clears_500"]:
        v = "YELLOW"
        why = ("clears 500 at central rho (capped at the 520.95 kernel ceiling) but the worst-case "
               "private tax lands {:.2f} TPS ({:+.2f}%) below 500.".format(
                   worst["joint_compliant_tps"] - TARGET, worst["headroom_pct"]))
    else:
        v = "RED"
        why = "fails 500 even at central rho."
    return {
        "verdict": v,
        "why": why,
        "joint_central_tps": central["joint_compliant_tps"],
        "joint_breakeven_tps": crn["breakeven"]["joint_compliant_tps"],
        "joint_worstcase_tps": worst["joint_compliant_tps"],
        "central_headroom_tps": central["headroom_tps"],
        "central_headroom_pct": central["headroom_pct"],
        "worstcase_headroom_tps": worst["headroom_tps"],
        "worstcase_headroom_pct": worst["headroom_pct"],
        "central_binding": central["binding_constraint"],
        "worstcase_binding": worst["binding_constraint"],
        "cap_eats_at_central_tps": central["et_lever_headroom_above_cap_tps"],
        "bar_between_worst_and_cap": bool(
            worst["joint_compliant_tps"] < TARGET < LAMBDA_CEIL),
    }


# --------------------------------------------------------------------------- #
# (4) The single cheapest measurement that flips YELLOW -> GREEN.
# --------------------------------------------------------------------------- #
def cheapest_flip() -> dict[str, Any]:
    """The YELLOW is driven by rho_worst=0.7923 < break-even 0.8038 (gap in rho), worth ~-7 TPS.

    Two candidate flips:
      (A) 0-GPU: credit the organizer-verified M=8 tree a_1-recovery (c_1=1.0, #316/#323) into the
          0.792 cross-domain bound. But 0.792 is a RAW AGGREGATE tau-ratio (5.34/6.74) and EAGLE-3
          has NO per-depth alpha table to isolate a_1 vs deep. The MOST GENEROUS analytic a_1-credit
          (attribute 100% of the cross-domain degradation to DEEP positions, a_1 fully held) is
          fern #318's `implied_f_deep_a1_held` = 0.9083 for the 0.792 scenario -- still BELOW the
          break-even f_deep 0.9163. So the 0-GPU credit CANNOT flip GREEN on its own; it is blocked
          by the missing per-depth table.
      (B) checkpoint-gated: the #319 staged per-depth private-alpha read on a trained {2,21,39}
          fusion head. Directly measures rho_priv_e3 (and the a_1/deep split that unblocks (A)),
          replacing the 0.792 cross-DOMAIN literature bound. The within-task measured analogue is
          0.957 (Delta 4.3%, organizer-verified on the linear stack) -- ~5x inside break-even -- so
          the measured fusion rho would very likely clear 500 at worst-case -> GREEN.
    """
    rho_gap = RHO_BREAKEVEN - RHO_WORST                     # 0.0115
    tps_gap_worst = TARGET - PRIV_TPS_WORST_318             # ~7.13 TPS
    # (A) most-generous analytic a_1-credit (fern #318 implied_f_deep_a1_held for the 0.792 case):
    implied_f_deep_a1_held_worst = 0.9082968311305457       # fern #318 eagle3_worst_xdataset
    f_deep_breakeven = 0.9163111901482197                   # fern #318 breakeven_decomposition
    analytic_credit_flips = implied_f_deep_a1_held_worst >= f_deep_breakeven
    # how much rho would the measured within-task analogue deliver (the realistic challenge shift):
    rho_within_task = 0.9570535584491102                    # deployed linear public->private (4.3%)
    measured_flips = HONEST_PUBLIC_611 * rho_within_task >= TARGET
    return {
        "yellow_driver": "rho_worst 0.7923 < break-even 0.8038",
        "rho_gap_to_breakeven": rho_gap,
        "worstcase_tps_gap_to_500": tps_gap_worst,
        "optionA_0gpu_tree_a1_credit": {
            "cost": "0-GPU analytic (cheapest)",
            "implied_f_deep_a1_held_worst": implied_f_deep_a1_held_worst,
            "f_deep_breakeven": f_deep_breakeven,
            "flips_green_alone": bool(analytic_credit_flips),
            "blocker": ("0.792 is a raw aggregate tau-ratio; EAGLE-3 has no per-depth alpha table to "
                        "isolate a_1 vs deep. Even the max-generous credit (all degradation -> deep, "
                        "a_1 fully held) gives implied f_deep 0.9083 < break-even 0.9163 -> still "
                        "misses by ~7 TPS. Cannot flip GREEN without the per-depth read."),
        },
        "optionB_319_perdepth_read": {
            "cost": "checkpoint-gated (trained {2,21,39} head + private eval)",
            "within_task_rho_analogue": rho_within_task,
            "within_task_private_tps": HONEST_PUBLIC_611 * rho_within_task,
            "flips_green": bool(measured_flips),
            "why": ("directly measures rho_priv_e3 AND the a_1/deep split that unblocks (A); the "
                    "within-task analogue (0.957, Delta 4.3%) sits ~5x inside break-even, so the "
                    "measured fusion rho would very likely clear 500 at worst-case."),
        },
        "recommendation": (
            "The #319 staged per-depth private-alpha read on a trained {2,21,39} head is the single "
            "cheapest measurement that ACTUALLY flips YELLOW->GREEN. The 0-GPU tree-a_1-recovery "
            "credit is cheaper but CANNOT flip alone (blocked by the missing per-depth table: even "
            "the max-generous credit leaves f_deep 0.9083 < break-even 0.9163). #319 supplies that "
            "very table (unblocking the credit) AND directly measures rho -- so it is both the "
            "cheapest unblocking move and the decisive one. Reinforces issue #319."),
    }


# --------------------------------------------------------------------------- #
# (5) Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def _selftests(crn: dict, bmap: dict, vd: dict, flip: dict) -> dict[str, Any]:
    # (a) composition reproduces the deployed 481.53 (within the 3.844-rounding tolerance) and the
    #     #318 honest_public anchor 622.08 from realized public E[T]=4.966.
    dep = official_tps(E_T_DEPLOYED)
    honest = official_tps(REALIZED_PUBLIC_ET)
    cond_a = bool(abs(dep - BASELINE_TPS) <= TOL_ANCHOR
                  and abs(honest - HONEST_PUBLIC_611) <= 0.2)

    # (b) corners reproduce fern #318's banked private TPS (pre-cap), central & worst.
    cond_b = bool(abs(HONEST_PUBLIC_611 * RHO_CENTRAL - PRIV_TPS_CENTRAL_318) <= TOL_RT
                  and abs(HONEST_PUBLIC_611 * RHO_WORST - PRIV_TPS_WORST_318) <= TOL_RT
                  and abs(HONEST_PUBLIC_611 * RHO_BREAKEVEN - TARGET) <= 1e-6)

    # (c) the cap BINDS at central (586 > 520.95) and does NOT bind at worst (492.87 < 520.95).
    cond_c = bool(crn["central"]["cap_binds"] is True
                  and crn["worstcase"]["cap_binds"] is False
                  and abs(crn["central"]["joint_compliant_tps"] - LAMBDA_CEIL) <= TOL_RT
                  and abs(crn["worstcase"]["joint_compliant_tps"] - PRIV_TPS_WORST_318) <= TOL_RT)

    # (d) ceiling round-trips its banked E[T] basis: 520.95 = K_cal*(E[T]_int4*1/step) at tau=1.
    rt_ceiling = abs(K_CAL * (INT4_SPEC_ET_TAU1 / STEP_DEPLOYED) - LAMBDA_CEIL)
    cond_d = bool(rt_ceiling <= TOL_RT and LAMBDA_CEIL > TARGET)

    # (e) the E[T]* switch (central) is real and inside [3.844, 6.11]; cap never binds at worst.
    cond_e = bool(bmap["switch_et_realization_to_ceiling_in_range"] is True
                  and bmap["cap_binds_at_worst_rho_in_range"] is False
                  and bmap["limiter_at_build_central"] == "a_kernel_ceiling_520p95"
                  and bmap["limiter_at_build_worst"] == "d_private_worstcase_rho")

    # (f) joint TPS monotone non-decreasing in free E[T] up to the cap (central rho).
    seq = [r["joint_compliant_tps"] for r in bmap["central_rho_rows"]]
    cond_f = all(seq[i] <= seq[i + 1] + 1e-9 for i in range(len(seq) - 1))

    # (g) verdict is YELLOW and the 500 bar sits strictly between worst (492.87) and cap (520.95).
    cond_g = bool(vd["verdict"] == "YELLOW" and vd["bar_between_worst_and_cap"] is True)

    # (h) the 0-GPU credit does NOT flip alone; the #319 read does -> decisive measurement identified.
    cond_h = bool(flip["optionA_0gpu_tree_a1_credit"]["flips_green_alone"] is False
                  and flip["optionB_319_perdepth_read"]["flips_green"] is True)

    # (i) NaN-clean (set by caller).
    cond_i = True

    conditions = {
        "a_composition_reproduces_481_and_622": cond_a,
        "b_corners_reproduce_318_private_tps": cond_b,
        "c_cap_binds_central_not_worst": cond_c,
        "d_ceiling_roundtrips_int4_et_basis": cond_d,
        "e_et_star_switch_real_and_worst_uncapped": cond_e,
        "f_joint_tps_monotone_to_cap": cond_f,
        "g_verdict_yellow_bar_between_worst_and_cap": cond_g,
        "h_0gpu_credit_cannot_flip_319_can": cond_h,
        "i_nan_clean": cond_i,
    }
    return {
        "conditions": conditions,
        "joint_compliant_500_envelope_self_test_passes": bool(all(conditions.values())),
        "detail": {
            "deployed_repro_tps": dep, "honest_public_repro_tps": honest,
            "ceiling_roundtrip_err": rt_ceiling,
        },
    }


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    crn = corners()
    bmap = binding_map()
    vd = verdict(crn)
    flip = cheapest_flip()
    st = _selftests(crn, bmap, vd, flip)

    headline = {
        "joint_compliant_500_envelope_self_test_passes":
            bool(st["joint_compliant_500_envelope_self_test_passes"]),        # PRIMARY
        "joint_compliant_tps_worstcase": crn["worstcase"]["joint_compliant_tps"],   # TEST (primary)
        "joint_compliant_tps_central": crn["central"]["joint_compliant_tps"],       # TEST (test)
        "joint_compliant_tps_breakeven": crn["breakeven"]["joint_compliant_tps"],
        "verdict": vd["verdict"],
        "central_binding": vd["central_binding"],
        "worstcase_binding": vd["worstcase_binding"],
        "kernel_ceiling": LAMBDA_CEIL,
        "cap_eats_at_central_tps": vd["cap_eats_at_central_tps"],
        "central_headroom_pct": vd["central_headroom_pct"],
        "worstcase_headroom_pct": vd["worstcase_headroom_pct"],
        "free_et_star_central": bmap["free_et_star_central"],
        "bar_between_worst_and_cap": vd["bar_between_worst_and_cap"],
    }

    return {
        "headline": headline,
        "ledger": {
            "formula": "official_compliant(E[T],kernel) = min(K_cal*(E[T]*tau/step_kernel), lambda)",
            "K_cal": K_CAL, "tau": TAU, "step_deployed": STEP_DEPLOYED, "step_kernel": STEP_KERNEL,
            "tau_over_step": TAU_OVER_STEP, "lambda_ceiling": LAMBDA_CEIL,
            "honest_public_611": HONEST_PUBLIC_611, "realized_public_et": REALIZED_PUBLIC_ET,
            "e_t_deployed": E_T_DEPLOYED, "e_t_build_free": E_T_BUILD_FREE,
            "baseline_tps": BASELINE_TPS, "target": TARGET,
            "step_kernel_is_deployed_reasoning": (
                "wirbel #216/#227 reconciled by #235: the batch-invariant split-K fix changes "
                "argmax reduction-ORDER (determinism) NOT the GEMM step-cost; the int4-spec ceiling "
                "520.95 is banked AT the deployed step. So step_kernel = step_deployed and the "
                "kernel enters only as the 520.95 cap. OPTIMISTIC corner: a slower kernel lowers the "
                "ceiling and raises the denominator -> can only tighten the verdict."),
        },
        "corners": crn,
        "binding_map": bmap,
        "verdict_block": vd,
        "cheapest_flip": flip,
        "self_test": st,
        "imports": {
            "provenance": (
                "fern #318 xe8ff7hq (honest_public_611 622.08, rho central 0.9421 / breakeven 0.8038 "
                "/ worst 0.7923, private TPS 586.08/492.87, implied_f_deep 0.9083, breakeven f_deep "
                "0.9163, within-task rho 0.957) x wirbel #216 pc8g6s04 / #227 o674wmna / #235 "
                "twoceiling (int4-spec ceiling 520.95, E[T]_int4 5.0661, kernel=argmax-order-not-step) "
                "x stark #298 xp974x58 (wall-realization 0.4769) x wirbel #314 fwqbz7zf (eager 360-481, "
                "rewrite +57..+140, loopgraph@611=500) x lawine #316 5lnz5jgb (cov bar 0.2907 vs "
                "linear 0.3468, deployed-spine E[T] cap 4.9097) x denken #278 / kanna #269 (K_cal "
                "125.268, step 1.2182, tau 1.218). All run-ids in "
                "wandb-applied-ai-team/gemma-challenge-senpai."),
        },
    }


# --------------------------------------------------------------------------- #
# NaN guard + report + W&B.
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
    h, crn, bmap = syn["headline"], syn["corners"], syn["binding_map"]
    vd, flip, st = syn["verdict_block"], syn["cheapest_flip"], syn["self_test"]
    print("\n" + "=" * 98, flush=True)
    print("JOINT COMPLIANT-500 ENVELOPE (PR #325, fern) — identity ceiling x E[T]-lever, CPU-only",
          flush=True)
    print("=" * 98, flush=True)
    print("  LEDGER  official_compliant = min( K_cal*(E[T]*tau/step_kernel), lambda=520.95 )",
          flush=True)
    print(f"          K_cal={K_CAL:.5f}  tau={TAU}  step_kernel={STEP_KERNEL} (=deployed)  "
          f"tau/step={TAU_OVER_STEP:.5f}", flush=True)
    print("-" * 98, flush=True)
    print("  (1) JOINT COMPLIANT TPS (E[T]-lever public 622.08 * rho, capped at 520.95)", flush=True)
    for nm in ("central", "breakeven", "worstcase"):
        r = crn[nm]
        print(f"      {nm:<10} rho={r['rho']:.4f}  uncapped={r['uncapped_et_lever_tps']:7.2f}  "
              f"joint={r['joint_compliant_tps']:7.2f}  cap_binds={str(r['cap_binds']):>5}  "
              f"clears500={str(r['clears_500']):>5}  ({r['headroom_pct']:+.2f}%)  "
              f"[{r['binding_constraint']}]", flush=True)
    print("-" * 98, flush=True)
    print(f"  (2) BINDING MAP over free E[T] in [3.844, 6.11]", flush=True)
    print(f"      central rho: switch E[T]-realization -> kernel ceiling at free E[T]*="
          f"{bmap['free_et_star_central']:.3f} (realized {bmap['realized_et_at_cap_central']:.3f})",
          flush=True)
    print(f"      worst rho:   cap binds in range = {bmap['cap_binds_at_worst_rho_in_range']} "
          f"(free E[T]*_worst={bmap['free_et_star_worst']:.3f} > 6.11) -> (d) private worst-case "
          f"limits throughout", flush=True)
    print(f"      limiter @ build target: central=[{bmap['limiter_at_build_central']}]  "
          f"worst=[{bmap['limiter_at_build_worst']}]", flush=True)
    print("-" * 98, flush=True)
    print(f"  (3) VERDICT: {vd['verdict']} — {vd['why']}", flush=True)
    print(f"      joint central={vd['joint_central_tps']:.2f} ({vd['central_headroom_pct']:+.2f}%)  "
          f"break-even={vd['joint_breakeven_tps']:.2f}  worst={vd['joint_worstcase_tps']:.2f} "
          f"({vd['worstcase_headroom_pct']:+.2f}%)", flush=True)
    print(f"      cap eats {vd['cap_eats_at_central_tps']:.2f} TPS at central (586.08 -> 520.95); "
          f"500 bar sits between worst {vd['joint_worstcase_tps']:.2f} and cap {LAMBDA_CEIL:.2f}",
          flush=True)
    print("-" * 98, flush=True)
    print("  (4) CHEAPEST YELLOW->GREEN measurement:", flush=True)
    a = flip["optionA_0gpu_tree_a1_credit"]
    b = flip["optionB_319_perdepth_read"]
    print(f"      (A) 0-GPU tree-a1 credit: flips_alone={a['flips_green_alone']} "
          f"(implied f_deep {a['implied_f_deep_a1_held_worst']:.4f} < break-even "
          f"{a['f_deep_breakeven']:.4f})", flush=True)
    print(f"      (B) #319 per-depth read: flips={b['flips_green']} "
          f"(within-task rho 0.957 -> {b['within_task_private_tps']:.1f})", flush=True)
    print(f"      => {flip['recommendation'][:96]}...", flush=True)
    print("-" * 98, flush=True)
    print(f"  (5) PRIMARY self_test_passes = {st['joint_compliant_500_envelope_self_test_passes']}",
          flush=True)
    for k, v in st["conditions"].items():
        print(f"        - {k}: {v}", flush=True)
    print("=" * 98 + "\n", flush=True)


def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[joint-compliant-envelope] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    h, crn, bmap = syn["headline"], syn["corners"], syn["binding_map"]
    vd, st = syn["verdict_block"], syn["self_test"]
    run = init_wandb_run(
        job_type="joint-compliant-500-envelope",
        agent="fern",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["joint-compliant-500-envelope", "issue-192", "eagle3", "identity-kernel-ceiling",
              "et-lever", "private-tax", "validity-gate", "bank-the-analysis"],
        config={
            "K_cal": K_CAL, "tau": TAU, "step_deployed": STEP_DEPLOYED, "step_kernel": STEP_KERNEL,
            "lambda_ceiling": LAMBDA_CEIL, "honest_public_611": HONEST_PUBLIC_611,
            "realized_public_et": REALIZED_PUBLIC_ET, "e_t_build_free": E_T_BUILD_FREE,
            "rho_central": RHO_CENTRAL, "rho_breakeven": RHO_BREAKEVEN, "rho_worst": RHO_WORST,
            "wall_realize_298": WALL_REALIZE, "baseline_tps": BASELINE_TPS, "target": TARGET,
            "wandb_group": args.wandb_group,
            "source_runs": "fern#318 xe8ff7hq, wirbel#216 pc8g6s04/#227 o674wmna/#235, stark#298 "
                           "xp974x58, wirbel#314 fwqbz7zf, lawine#316 5lnz5jgb",
        },
    )
    if run is None:
        print("[joint-compliant-envelope] wandb: no run (no WANDB_API_KEY/mode) — skipping",
              flush=True)
        return

    summary: dict[str, Any] = {
        "joint_compliant_500_envelope_self_test_passes":
            int(bool(st["joint_compliant_500_envelope_self_test_passes"])),       # PRIMARY
        "joint_compliant_tps_worstcase": h["joint_compliant_tps_worstcase"],       # TEST (primary)
        "joint_compliant_tps_central": h["joint_compliant_tps_central"],           # TEST
        "joint_compliant_tps_breakeven": h["joint_compliant_tps_breakeven"],
        "uncapped_central_tps": crn["central"]["uncapped_et_lever_tps"],
        "uncapped_worstcase_tps": crn["worstcase"]["uncapped_et_lever_tps"],
        "kernel_ceiling": LAMBDA_CEIL,
        "cap_eats_at_central_tps": h["cap_eats_at_central_tps"],
        "central_headroom_pct": h["central_headroom_pct"],
        "worstcase_headroom_pct": h["worstcase_headroom_pct"],
        "central_cap_binds": int(bool(crn["central"]["cap_binds"])),
        "worstcase_cap_binds": int(bool(crn["worstcase"]["cap_binds"])),
        "free_et_star_central": bmap["free_et_star_central"],
        "cap_binds_at_worst_rho_in_range": int(bool(bmap["cap_binds_at_worst_rho_in_range"])),
        "verdict_yellow": int(vd["verdict"] == "YELLOW"),
        "verdict_green": int(vd["verdict"] == "GREEN"),
        "verdict_red": int(vd["verdict"] == "RED"),
        "bar_between_worst_and_cap": int(bool(vd["bar_between_worst_and_cap"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    # per-grid-point joint TPS (central + worst) as logged scalars.
    for i, (rc, rw) in enumerate(zip(bmap["central_rho_rows"], bmap["worst_rho_rows"])):
        summary[f"joint_central_free_et_{i:02d}"] = rc["joint_compliant_tps"]
        summary[f"joint_worst_free_et_{i:02d}"] = rw["joint_compliant_tps"]
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="joint_compliant_500_envelope_result", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    print(f"[joint-compliant-envelope] wandb logged {len(summary)} keys", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="eagle3-joint-compliant-envelope")
    args = ap.parse_args(argv)

    syn = synthesize()

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 325, "agent": "fern",
        "kind": "joint-compliant-500-envelope", "analysis_only": True, "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _nan_paths(payload)
    payload["nan_clean"] = not nan_paths
    syn["self_test"]["conditions"]["i_nan_clean"] = not nan_paths
    syn["self_test"]["joint_compliant_500_envelope_self_test_passes"] = bool(
        all(syn["self_test"]["conditions"].values()))
    syn["headline"]["joint_compliant_500_envelope_self_test_passes"] = syn["self_test"][
        "joint_compliant_500_envelope_self_test_passes"]
    if nan_paths:
        print(f"[joint-compliant-envelope] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "joint_compliant_500_envelope_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[joint-compliant-envelope] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = (syn["self_test"]["joint_compliant_500_envelope_self_test_passes"]
              and payload["nan_clean"])
        print(f"[joint-compliant-envelope] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
