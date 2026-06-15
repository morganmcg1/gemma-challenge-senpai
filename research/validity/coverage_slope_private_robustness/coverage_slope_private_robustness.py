#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #382 -- Does the coverage->gap slope (489.8 TPS/unit) survive private OOD? (#319)

WHAT THIS ANSWERS
-----------------
My #379 (`5kpb73tb`, merged) established the GREEN demand-side path: the 4.295%
public->private gap is ~85% acceptance (coverage-addressable), the irreducible floor
is 0.633% << 3.2%, and the coverage->gap lever has slope

    slope_public = dTPS/dcov = T2 * K_cal = 3.9097 * 125.268 ~ 489.76 TPS / unit cov

with coverage_target_for_3p2 = 0.9011 (within #336's +0.031 budget). fern #357 is about
to BANK that slope. BUT the slope's leverage T2 = 3.9097 was derived from the #289
a_1-only profile on the PUBLIC distribution. My #263 measured a private OOD acceptance
collapse. THE DECISIVE QUESTION: does the public-anchored slope LAND on the PRIVATE
distribution at 489.76, or does private OOD flatten it enough to push
coverage_target_for_3p2_private OUT of #336's +0.031 budget?

This is the public->private DISTRIBUTION-TRANSFER axis -- ORTHOGONAL to denken #380's
kappa (coverage->accept transfer degraded by int4-ct QUANTIZATION NOISE, a same-
distribution numerics effect). I own: does the delivered lift LAND on the private slice
at the assumed slope.

THE MECHANISM (why the slope can flatten -- and why a_1 is irrelevant to it)
---------------------------------------------------------------------------
Acceptance-length of a K-token draft chain with per-position CONDITIONAL acceptances
a_1..a_K (a_j = P(draft token j accepted | 1..j-1 accepted)):

    E[T] = 1 + sum_{k=1..K} prod_{j=1..k} a_j
         = 1 + a_1 * ( 1 + a_2 + a_2 a_3 + ... + a_2...a_K )
         = 1 + a_1 * T2                                         (EXACT identity)
    T2   = 1 + a_2 + a_2 a_3 + ... + a_2...a_K  =  dE[T]/da_1   (the DOWNSTREAM tail)

So the a_1-only leverage T2 is the DOWNSTREAM conditional-acceptance tail, and E[T] is
LINEAR in a_1 -> dE[T]/da_1 = T2 does NOT depend on a_1. CRITICAL CONSEQUENCE: the slope
flattening depends ONLY on how the DEEP (k>=2) conditional acceptances degrade on private
-- the (severe) first-token collapse is IRRELEVANT to the slope.

slope_private = T2_priv * K_cal ;  slope_flattening_ratio = T2_priv / T2_pub = slope_priv/slope_pub.

THE PRIVATE PROFILE IS DIRECTLY MEASURED (NOT modeled) -- #263
--------------------------------------------------------------
My #263 (`he7glotf`) did not only measure rank coverage; its rank-probe recorded the
per-DRAFT-POSITION conditional acceptance on a private-proxy slice AND a matched public
repro:
  PRIV a_k = conditional_rank1_acceptance_q  (private_proxy_sharegpt, the ADVERSARIAL slice)
  PUB  a_k = cross_check.conditional76        (matched public repro; == #289 to <1e-3)
The collapse is concentrated at the FIRST positions (a_1 0.729->0.598, a_2 0.759->0.691);
the DEEP positions HOLD UP or improve (a_5..a_7 are HIGHER on private -- a survivor effect:
on the harder private slice only well-tracked prompts reach depth). Because T2 is the deep
tail, the measured private slope only flattens MILDLY even on this adversarial slice.

MODELS (one directly-measured central + a conservative stress + a mild bound + a floor)
---------------------------------------------------------------------------------------
  central  (MEASURED) : T2_priv from #263 measured private per-position profile.
  conservative (STRESS): degrade every downstream a_k (k>=2) MULTIPLICATIVELY by the #263
                         mean rank-2+ marginal collapse (-34.5%). Over-reads the per-
                         position conditional degradation (marginal-mass collapse != per-
                         position conditional collapse) -> a deliberate worst case.
  mild     (survival)  : collapse the whole downstream survival SUM by -34.5% (Model A).
  floor    (adversarial): entire downstream pinned at the #289 low-tail constant alpha
                         (0.694) / decode-proxy regime.

BREAKEVEN (closed form): coverage_target exits the +0.031 budget when the coverage delta
needed exceeds the budget, i.e. when

    slope_private  <  tps_shrink_to_3p2 / COVERAGE_BUDGET
    flattening_breakeven = tps_shrink_to_3p2 / (COVERAGE_BUDGET * slope_public)

CPU-analytic over BANKED W&B numbers (my #379 slope/targets, #289 public profile, #263
measured private profile, #336 coverage budget). NO new GPU measurement, NO served-file
change, NO HF Job, NO --launch, NO submission. BASELINE stays 481.53; this leg adds 0 TPS.
The OPTIONAL local-A10G accept-gap leg is SKIPPED: the private per-position profile is
DIRECTLY MEASURED in #263 (not under-determined), and the breakeven sits far below every
plausible model -> the verdict does not need a fresh measurement.

PRIMARY self-test : slope_robustness_self_test_passes (bool).
Headline          : slope_private_oob, slope_flattening_ratio, coverage_target_for_3p2_private,
                    target_inflation, budget_margin_private, flattening_breakeven,
                    flattening_margin, slope_is_private_robust, demand_route_survives_private_oob.

Reproduce:
    cd target/ && python research/validity/coverage_slope_private_robustness/\
coverage_slope_private_robustness.py --private-oob-slope --anchor-263-collapse \
        --wandb_group strict-bi-verify-gemm --wandb_name ubel/coverage-slope-private-robustness
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

# research/validity/coverage_slope_private_robustness/this.py -> repo root is 3 up.
ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "research" / "validity" / "coverage_slope_private_robustness"
RESULTS_PATH = OUT_DIR / "coverage_slope_private_robustness_results.json"

# --------------------------------------------------------------------------- #
# Imported fleet anchors (DO NOT re-derive -- import EXACTLY, UNCHANGED)
# --------------------------------------------------------------------------- #
# --- my #379 (5kpb73tb) public-anchored slope + targets (the thing under test) ---
OFFICIAL_PUBLIC = 481.53            # PR #52 official frontier TPS (public)
PRIVATE_VALID = 460.85             # denken #373 private-VALID TPS (5k3px8p1)
KNIFE_EDGE_PCT = 3.2               # private-500 flips to GO iff gap < ~3.2%
K_CAL = 125.268                    # kanna #269: official = K_cal * E[T]
SLOPE_PUBLIC = 489.76448056537095  # #379 gap_shrink_contribution_per_coverage (= A1_ONLY_T2*K_cal)
COVERAGE_TARGET_PUBLIC = 0.9010623974566615  # #379 coverage_target_for_3p2 (public-anchored)
A1_ONLY_T2 = 3.909733376164471     # #289 a_1-only leverage T2 (= E_T_if_a1_perfect - 1)

# --- #336 coverage budget (top-4 root unconditional acceptance) ---
COVERAGE_BASELINE = 0.8903         # #336 aggregate_baseline
COVERAGE_BAR = 0.9213              # #336 bar (= baseline + budget)
COVERAGE_BUDGET = 0.031            # #336 soft-KD + reasoning-trace REACHABLE-MARGINAL envelope

# --- #289 (fi34s269) PUBLIC per-position conditional acceptance a_1..a_7 (K=7) ---
PUB_AK_289 = [
    0.7292532942898975, 0.759556697719242, 0.7929794882639035, 0.8228,
    0.8348727920920435, 0.8357919254658385, 0.8464932652113331,
]
ET_PUB_289 = 3.851185944363104     # #289 E_T from the public profile
PUB_LOW_TAIL_ALPHA_289 = 0.6940577578335754  # #289 low_tail_alpha_constant (adversarial floor)

# --- #263 (he7glotf) DIRECTLY-MEASURED per-draft-position conditional acceptance ---
# private-proxy slice (ADVERSARIAL: over-reads the benchmark gap 3-5x) + matched public repro.
PRIV_AK_263 = [   # analysis.conditional_rank1_acceptance_q
    0.5975381962737679, 0.6914075024046169, 0.7470438210062601, 0.7687771570453135,
    0.8445700444085588, 0.8582695984703633, 0.8916736285157337,
]
PUB_AK_263 = [    # cross_check.conditional76 (matched public repro; == #289 to <1e-3)
    0.728739760479042, 0.7589764102641635, 0.7924989076194682, 0.821702519412012,
    0.8342716929825772, 0.8352594665096346, 0.8472621220149911,
]
# #263 rank-2+ marginal coverage collapse vs public (per rank 2/3/4) and its mean:
RHO_COLLAPSE_PCT_263 = [-33.2, -34.5, -35.8]    # verdict_summary.rho_collapse_pct
RHO_COLLAPSE_MEAN_PCT_263 = -34.5               # verdict_summary.rho_collapse_mean_pct
PRIV_TOP1_263 = 0.5975381962737679              # private first-token acceptance
PUB_TOP1_263 = 0.728739760479042                # matched public first-token acceptance

# benchmark (NON-adversarial) private E[T] cross-check: rho_deployed * E_T_pub (#318/#379)
RHO_DEPLOYED_318 = 0.9570535584491102
E_T_PUB_DEPLOYED = 3.844
ET_PRIV_BENCHMARK = RHO_DEPLOYED_318 * E_T_PUB_DEPLOYED   # ~3.679 (the REALISTIC private E[T])

K_SPEC = 7                          # draft chain length (#257 k7 / #289 K=7 / #263 7 depths)


# --------------------------------------------------------------------------- #
# numeric helpers (no scipy in the analytic venv)
# --------------------------------------------------------------------------- #
def bisect(f: Callable[[float], float], lo: float, hi: float,
           tol: float = 1e-14, max_it: int = 600) -> float:
    """Robust bracketed root find; raises if [lo,hi] does not bracket a root."""
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        raise ValueError(f"bisect: no sign change on [{lo},{hi}] -> {flo},{fhi}")
    for _ in range(max_it):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < tol or (hi - lo) < tol:
            return mid
        if flo * fm < 0.0:
            hi = mid
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# the per-position acceptance algebra (E[T] = 1 + a_1 * T2 ; T2 = downstream tail)
# --------------------------------------------------------------------------- #
def t2_leverage(a_k: list[float]) -> float:
    """T2 = dE[T]/da_1 = 1 + a_2 + a_2 a_3 + ... + a_2...a_K  (the DOWNSTREAM tail).

    Independent of a_1 by construction (E[T] is linear in a_1)."""
    s, p = 0.0, 1.0
    for a in a_k[1:]:           # a_2 .. a_K
        p *= a
        s += p
    return 1.0 + s


def e_t(a_k: list[float]) -> float:
    """E[T] = 1 + sum_k prod_{j<=k} a_j = 1 + a_1 * T2."""
    return 1.0 + a_k[0] * t2_leverage(a_k)


def t2_constant_downstream(alpha: float, k_spec: int = K_SPEC) -> float:
    """T2 with every downstream a_k pinned at a constant alpha (adversarial floor)."""
    s, p = 0.0, 1.0
    for _ in range(k_spec - 1):     # K-1 downstream positions (a_2..a_K)
        p *= alpha
        s += p
    return 1.0 + s


def degrade_downstream_mult(a_k_pub: list[float], delta: float) -> list[float]:
    """Multiplicatively degrade DOWNSTREAM conditional acceptances (k>=2) by (1-delta).

    a_1 is left unchanged: it does not enter T2 (the slope). Clamped to [0,1]."""
    out = [a_k_pub[0]]
    for a in a_k_pub[1:]:
        out.append(min(1.0, max(0.0, a * (1.0 - delta))))
    return out


# --------------------------------------------------------------------------- #
# slope -> coverage-target / budget plumbing (the thing #379 banked)
# --------------------------------------------------------------------------- #
def tps_shrink_to_3p2() -> float:
    """TPS the private side must GAIN to reach the 3.2% knife-edge (fixed; measured)."""
    tps_priv_at_3p2 = OFFICIAL_PUBLIC * (1.0 - KNIFE_EDGE_PCT / 100.0)
    return tps_priv_at_3p2 - PRIVATE_VALID


def coverage_delta_for_3p2(slope: float) -> float:
    return tps_shrink_to_3p2() / slope


def coverage_target_for_3p2(slope: float) -> float:
    return COVERAGE_BASELINE + coverage_delta_for_3p2(slope)


def budget_margin(slope: float) -> float:
    """+0.031 - coverage_delta_needed ; >0 means within #336 budget."""
    return COVERAGE_BUDGET - coverage_delta_for_3p2(slope)


def slope_model(name: str, t2_priv: float, t2_pub: float, note: str) -> dict[str, Any]:
    ratio = t2_priv / t2_pub
    slope_priv = ratio * SLOPE_PUBLIC          # apply the #263-matched flattening to the banked slope
    slope_priv_direct = t2_priv * K_CAL        # cross-check (uses #263 pub anchor, not #289's)
    cov_delta = coverage_delta_for_3p2(slope_priv)
    cov_target = COVERAGE_BASELINE + cov_delta
    return {
        "name": name,
        "note": note,
        "t2_priv": t2_priv,
        "t2_pub": t2_pub,
        "slope_flattening_ratio": ratio,
        "slope_private_oob": slope_priv,
        "slope_private_direct_t2xkcal": slope_priv_direct,
        "coverage_delta_for_3p2_private": cov_delta,
        "coverage_target_for_3p2_private": cov_target,
        "target_inflation": cov_target - COVERAGE_TARGET_PUBLIC,
        "budget_margin_private": COVERAGE_BUDGET - cov_delta,
        "budget_fraction_used": cov_delta / COVERAGE_BUDGET,
        "within_336_budget": bool(cov_delta <= COVERAGE_BUDGET),
        "coverage_target_below_bar": bool(cov_target <= COVERAGE_BAR + 1e-12),
    }


# --------------------------------------------------------------------------- #
# build the report
# --------------------------------------------------------------------------- #
def build_report(conservative_delta: float = abs(RHO_COLLAPSE_MEAN_PCT_263) / 100.0) -> dict[str, Any]:
    t2_pub_263 = t2_leverage(PUB_AK_263)        # ~3.9034 (matched #263 public repro)
    t2_pub_289 = t2_leverage(PUB_AK_289)        # ~3.9097 (== imported A1_ONLY_T2)
    t2_priv_meas = t2_leverage(PRIV_AK_263)     # ~3.4849 (measured private)

    # ---- the FOUR models, all referenced to the #263 matched public T2 ----
    central = slope_model(
        "central_measured_263", t2_priv_meas, t2_pub_263,
        "DIRECTLY-MEASURED #263 private per-position profile (adversarial slice).")
    priv_cons = degrade_downstream_mult(PUB_AK_263, conservative_delta)
    conservative = slope_model(
        "conservative_mult_345", t2_leverage(priv_cons), t2_pub_263,
        f"STRESS: downstream a_k * (1-{conservative_delta:.3f}) [#263 mean rank-2+ collapse]; "
        "over-reads per-position degradation.")
    t2_priv_surv = 1.0 + (t2_pub_263 - 1.0) * (1.0 - conservative_delta)
    mild = slope_model(
        "mild_survival_sum_345", t2_priv_surv, t2_pub_263,
        f"survival-SUM collapse by {conservative_delta:.3f} (Model A).")
    t2_priv_floor = t2_constant_downstream(PUB_LOW_TAIL_ALPHA_289)
    adversarial = slope_model(
        "adversarial_lowtail_const", t2_priv_floor, t2_pub_263,
        f"downstream pinned at #289 low-tail alpha={PUB_LOW_TAIL_ALPHA_289:.4f}.")

    # ---- benchmark (REALISTIC, non-adversarial) cross-check ----
    # the benchmark private E[T] = rho*E_T_pub = 3.679 is only ~4.5% below public 3.851;
    # attributing that whole net E[T] degradation to the downstream tail (worst case for the
    # slope, since it ignores the a_1 share) bounds the realistic flattening from BELOW 1.0.
    # E[T] = 1 + a_1*T2 ; hold a_1 at public, solve T2_bench: T2 = (E_T_priv - 1)/a_1_pub
    a1_pub = PUB_AK_263[0]
    t2_bench_all_on_tail = (ET_PRIV_BENCHMARK - 1.0) / a1_pub
    bench = slope_model(
        "benchmark_realistic", min(t2_bench_all_on_tail, t2_pub_263), t2_pub_263,
        "REALISTIC: benchmark private E[T]=rho*E_T_pub, whole net E[T] loss charged to the tail.")

    # ---- breakeven: closed form + multiplicative-delta solve ----
    flattening_breakeven = tps_shrink_to_3p2() / (COVERAGE_BUDGET * SLOPE_PUBLIC)

    def ratio_mult(delta: float) -> float:
        return t2_leverage(degrade_downstream_mult(PUB_AK_263, delta)) / t2_pub_263

    # delta (multiplicative) at which the ratio hits breakeven (route exits budget)
    delta_breakeven = float("nan")
    try:
        if ratio_mult(0.0) > flattening_breakeven > ratio_mult(0.99):
            delta_breakeven = bisect(lambda d: ratio_mult(d) - flattening_breakeven, 0.0, 0.99)
    except ValueError:
        delta_breakeven = float("nan")

    central_delta_equiv = float("nan")   # the multiplicative-delta that reproduces the measured ratio
    try:
        if ratio_mult(0.0) > central["slope_flattening_ratio"] > ratio_mult(0.99):
            central_delta_equiv = bisect(
                lambda d: ratio_mult(d) - central["slope_flattening_ratio"], 0.0, 0.99)
    except ValueError:
        central_delta_equiv = float("nan")

    flattening_margin = central["slope_flattening_ratio"] - flattening_breakeven
    flattening_margin_conservative = conservative["slope_flattening_ratio"] - flattening_breakeven
    breakeven_multiple_of_263 = (delta_breakeven / conservative_delta
                                 if math.isfinite(delta_breakeven) and conservative_delta > 0
                                 else float("nan"))

    # ---- delta sweep (multiplicative model) for the W&B table ----
    sweep = []
    grid = sorted(set(
        [0.0, 0.10, abs((PRIV_TOP1_263 - PUB_TOP1_263) / PUB_TOP1_263), 0.25,
         conservative_delta, abs(min(RHO_COLLAPSE_PCT_263)) / 100.0, 0.45, 0.55, 0.65]
        + ([round(delta_breakeven, 6)] if math.isfinite(delta_breakeven) else [])))
    for d in grid:
        r = ratio_mult(d)
        sp = r * SLOPE_PUBLIC
        cd = coverage_delta_for_3p2(sp)
        sweep.append({
            "downstream_collapse_delta": d,
            "slope_flattening_ratio": r,
            "slope_private_tps_per_cov": sp,
            "coverage_delta_for_3p2_private": cd,
            "coverage_target_for_3p2_private": COVERAGE_BASELINE + cd,
            "budget_margin_private": COVERAGE_BUDGET - cd,
            "within_336_budget": bool(cd <= COVERAGE_BUDGET),
        })

    # ---- verdict ----
    slope_is_private_robust = bool(
        central["within_336_budget"] and conservative["within_336_budget"]
        and central["slope_flattening_ratio"] > flattening_breakeven
        and conservative["slope_flattening_ratio"] > flattening_breakeven)
    demand_route_survives = bool(central["within_336_budget"] and conservative["within_336_budget"])

    if slope_is_private_robust and demand_route_survives:
        band = "GREEN_slope_private_robust"
        recommended_action = (
            f"ROUTE CONFIRMED private-safe. fern #357 may BANK the coverage->gap slope. "
            f"Recommend banking the conservative private-anchored target "
            f"coverage~{conservative['coverage_target_for_3p2_private']:.4f} "
            f"(measured-central {central['coverage_target_for_3p2_private']:.4f}) rather than the "
            f"bare public {COVERAGE_TARGET_PUBLIC:.4f}; even the conservative target sits "
            f"{conservative['budget_margin_private']:.4f} inside #336's +0.031 budget. The slope "
            f"only breaks at flattening<{flattening_breakeven:.3f} (downstream collapse "
            f"delta~{delta_breakeven:.3f}, ~{breakeven_multiple_of_263:.1f}x the #263-measured "
            f"rank-2+ collapse) -- far below every plausible model.")
    else:
        band = "RED_slope_not_private_safe"
        recommended_action = (
            f"RE-SIZE NEEDED. The public 489.8 slope is NOT private-safe: "
            f"coverage_target_for_3p2_private={central['coverage_target_for_3p2_private']:.4f} "
            f"exits #336's +0.031 budget. fern #357 must bank the PRIVATE-anchored slope "
            f"{central['slope_private_oob']:.1f} TPS/unit and target "
            f"{central['coverage_target_for_3p2_private']:.4f}, NOT the public 489.8 / 0.9011. "
            f"Flag as the binding open item.")

    report = {
        "pr": 382, "issue": 319, "author": "ubel",
        "leg": "coverage->gap slope private-OOD robustness (#263-anchored)",
        "analysis_only": True, "no_hf_job": True, "no_launch": True, "no_submission": True,
        "no_served_file_change": True, "gpu_used": False, "tps_added_by_this_card": 0,
        "orthogonal_to": "denken #380 (kappa = int4-ct quantization-noise, same-distribution); "
                         "fern #357 (composite integrator). This leg = public->private "
                         "distribution-shift transfer of the coverage lever.",
        "optional_gpu_accept_leg": "SKIPPED -- the private per-position profile is DIRECTLY "
                                   "MEASURED in #263 (conditional_rank1_acceptance_q), not under-"
                                   "determined; and flattening_breakeven sits far below every "
                                   "plausible model. A fresh GPU E[l] measurement cannot move the "
                                   "verdict. CPU-analytic stands.",
        "measured_private_leverage": None,    # optional GPU leg not run
        "imported": {
            "official_public": OFFICIAL_PUBLIC, "private_valid": PRIVATE_VALID,
            "knife_edge_pct": KNIFE_EDGE_PCT, "K_cal": K_CAL,
            "slope_public": SLOPE_PUBLIC, "coverage_target_public_379": COVERAGE_TARGET_PUBLIC,
            "a1_only_T2_289": A1_ONLY_T2, "coverage_baseline_336": COVERAGE_BASELINE,
            "coverage_bar_336": COVERAGE_BAR, "coverage_budget_336": COVERAGE_BUDGET,
            "pub_ak_289": PUB_AK_289, "pub_ak_263": PUB_AK_263, "priv_ak_263": PRIV_AK_263,
            "rho_collapse_pct_263": RHO_COLLAPSE_PCT_263,
            "rho_collapse_mean_pct_263": RHO_COLLAPSE_MEAN_PCT_263,
            "priv_top1_263": PRIV_TOP1_263, "pub_top1_263": PUB_TOP1_263,
            "et_priv_benchmark": ET_PRIV_BENCHMARK, "k_spec": K_SPEC,
            "conservative_delta": conservative_delta,
        },
        # ---- leverage anchors ----
        "t2_public_289": t2_pub_289,
        "t2_public_263": t2_pub_263,
        "t2_private_measured_263": t2_priv_meas,
        "tps_shrink_to_3p2": tps_shrink_to_3p2(),
        "et_pub_263_profile": e_t(PUB_AK_263),
        "et_priv_263_profile": e_t(PRIV_AK_263),
        # ---- HEADLINE (central = measured #263 private profile) ----
        "slope_public": SLOPE_PUBLIC,
        "slope_private_oob": central["slope_private_oob"],
        "slope_flattening_ratio": central["slope_flattening_ratio"],
        "coverage_target_for_3p2_private": central["coverage_target_for_3p2_private"],
        "target_inflation": central["target_inflation"],
        "budget_margin_private": central["budget_margin_private"],
        "flattening_breakeven": flattening_breakeven,
        "flattening_margin": flattening_margin,
        "flattening_margin_conservative": flattening_margin_conservative,
        "delta_breakeven_multiplicative": delta_breakeven,
        "central_delta_equiv_multiplicative": central_delta_equiv,
        "breakeven_multiple_of_263_collapse": breakeven_multiple_of_263,
        # ---- verdict (PR-required booleans) ----
        "slope_is_private_robust": slope_is_private_robust,
        "demand_route_survives_private_oob": demand_route_survives,
        "recommended_action": recommended_action,
        "verdict_band": band,
        # ---- model corners + sweep ----
        "models": {
            "central_measured": central, "conservative_stress": conservative,
            "mild_survival": mild, "adversarial_floor": adversarial, "benchmark_realistic": bench,
        },
        "delta_sweep": sweep,
        "official_baseline_unchanged": OFFICIAL_PUBLIC,
    }

    report["self_test"], report["slope_robustness_self_test_passes"] = self_test(report)
    return report


# --------------------------------------------------------------------------- #
# self-test (PRIMARY)
# --------------------------------------------------------------------------- #
def self_test(r: dict[str, Any]) -> tuple[dict[str, bool], bool]:
    c = r["models"]["central_measured"]
    cons = r["models"]["conservative_stress"]
    checks: dict[str, bool] = {}

    # (a) public T2 round-trips the imported #289 a_1-only leverage
    checks["a_t2_pub_289_matches_import"] = abs(r["t2_public_289"] - A1_ONLY_T2) <= 1e-9
    # (a2) E[T] = 1 + a_1*T2 identity holds on the public profile (reproduces #289 E[T])
    checks["a_et_identity_pub"] = abs(e_t(PUB_AK_289) - ET_PUB_289) <= 1e-9
    checks["a_et_identity_decomposes"] = abs(
        e_t(PUB_AK_263) - (1.0 + PUB_AK_263[0] * t2_leverage(PUB_AK_263))) <= 1e-12

    # (b) public slope reconstructs the #379 banked 489.76 from T2_289*K_cal
    checks["b_slope_public_reconstructs_379"] = abs(A1_ONLY_T2 * K_CAL - SLOPE_PUBLIC) <= 1e-6
    # (b2) public coverage target reconstructs the #379 banked 0.9011
    checks["b_cov_target_public_reconstructs_379"] = abs(
        coverage_target_for_3p2(SLOPE_PUBLIC) - COVERAGE_TARGET_PUBLIC) <= 1e-9

    # (c) T2 is independent of a_1 (perturbing a_1 leaves T2 unchanged) -- the core mechanism
    pub_perturb = [min(1.0, PUB_AK_263[0] + 0.1)] + PUB_AK_263[1:]
    checks["c_t2_independent_of_a1"] = abs(
        t2_leverage(pub_perturb) - t2_leverage(PUB_AK_263)) <= 1e-12

    # (d) measured private flattening is in (0,1] and milder than the conservative stress
    checks["d_ratio_measured_in_unit"] = 0.0 < c["slope_flattening_ratio"] <= 1.0 + 1e-12
    checks["d_measured_milder_than_conservative"] = (
        c["slope_flattening_ratio"] >= cons["slope_flattening_ratio"] - 1e-12)
    # (d2) first-token collapse is severe but deep tail holds (survivor effect): priv a_7 >= pub a_7
    checks["d_deep_tail_holds_on_private"] = PRIV_AK_263[-1] >= PUB_AK_263[-1] - 1e-9

    # (e) breakeven closed form matches the multiplicative-delta solve
    be = r["flattening_breakeven"]
    db = r["delta_breakeven_multiplicative"]
    ratio_at_db = t2_leverage(degrade_downstream_mult(PUB_AK_263, db)) / r["t2_public_263"]
    checks["e_breakeven_closed_form_matches_sweep"] = (
        math.isfinite(db) and abs(ratio_at_db - be) <= 1e-6)
    # (e2) breakeven == coverage delta exactly exhausting the budget
    checks["e_breakeven_exhausts_budget"] = abs(
        coverage_delta_for_3p2(be * SLOPE_PUBLIC) - COVERAGE_BUDGET) <= 1e-9

    # (f) central + conservative both stay WITHIN the #336 budget (the GREEN bar)
    checks["f_central_within_budget"] = bool(c["within_336_budget"])
    checks["f_conservative_within_budget"] = bool(cons["within_336_budget"])
    checks["f_central_below_bar"] = bool(c["coverage_target_below_bar"])

    # (g) both central and conservative flattening sit strictly ABOVE breakeven (with margin)
    checks["g_central_above_breakeven"] = c["slope_flattening_ratio"] > be
    checks["g_conservative_above_breakeven"] = cons["slope_flattening_ratio"] > be
    checks["g_breakeven_needs_more_than_263"] = (
        math.isfinite(db) and db > abs(RHO_COLLAPSE_MEAN_PCT_263) / 100.0)

    # (h) sweep is monotone: more collapse -> lower ratio -> higher coverage target
    sw = r["delta_sweep"]
    checks["h_sweep_monotone_ratio"] = all(
        sw[i]["slope_flattening_ratio"] >= sw[i + 1]["slope_flattening_ratio"] - 1e-12
        for i in range(len(sw) - 1))
    checks["h_sweep_monotone_target"] = all(
        sw[i]["coverage_target_for_3p2_private"] <= sw[i + 1]["coverage_target_for_3p2_private"] + 1e-12
        for i in range(len(sw) - 1))

    # (i) tps_shrink_to_3p2 reconstructs the #379 +5.27 TPS
    checks["i_tps_shrink_matches_379"] = abs(r["tps_shrink_to_3p2"] - 5.271039999999914) <= 1e-3

    # (j) constants imported EXACT and UNCHANGED
    checks["j_constants_imported_exact"] = (
        OFFICIAL_PUBLIC == 481.53 and PRIVATE_VALID == 460.85 and K_CAL == 125.268
        and KNIFE_EDGE_PCT == 3.2 and COVERAGE_BUDGET == 0.031 and COVERAGE_BASELINE == 0.8903
        and SLOPE_PUBLIC == 489.76448056537095)

    # (k) NaN-clean across headline scalars
    scal = [r["slope_private_oob"], r["slope_flattening_ratio"],
            r["coverage_target_for_3p2_private"], r["target_inflation"],
            r["budget_margin_private"], r["flattening_breakeven"], r["flattening_margin"],
            r["delta_breakeven_multiplicative"], cons["slope_flattening_ratio"]]
    checks["k_nan_clean"] = all(math.isfinite(float(x)) for x in scal)

    gate = bool(
        checks["a_t2_pub_289_matches_import"] and checks["a_et_identity_pub"]
        and checks["a_et_identity_decomposes"] and checks["b_slope_public_reconstructs_379"]
        and checks["b_cov_target_public_reconstructs_379"] and checks["c_t2_independent_of_a1"]
        and checks["d_ratio_measured_in_unit"] and checks["d_measured_milder_than_conservative"]
        and checks["e_breakeven_closed_form_matches_sweep"] and checks["e_breakeven_exhausts_budget"]
        and checks["f_central_within_budget"] and checks["f_conservative_within_budget"]
        and checks["f_central_below_bar"] and checks["g_central_above_breakeven"]
        and checks["g_conservative_above_breakeven"] and checks["g_breakeven_needs_more_than_263"]
        and checks["h_sweep_monotone_ratio"] and checks["h_sweep_monotone_target"]
        and checks["i_tps_shrink_matches_379"] and checks["j_constants_imported_exact"]
        and checks["k_nan_clean"])
    return checks, gate


# --------------------------------------------------------------------------- #
# wandb
# --------------------------------------------------------------------------- #
def log_wandb(report: dict[str, Any], name: str, group: str) -> str | None:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[cspr] wandb unavailable ({exc})", flush=True)
        return None
    if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        print("[cspr] wandb disabled via env", flush=True)
        return None
    try:
        c = report["models"]["central_measured"]
        cons = report["models"]["conservative_stress"]
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            name=name, group=group, job_type="analysis",
            tags=["gemma-challenge", "analysis", "coverage-slope", "private-oob",
                  "slope-robustness", "demand-side", "issue-319", "pr-382"],
            config={
                "pr": 382, "issue": 319, "analysis_only": True, "wandb_group": group,
                "slope_public": SLOPE_PUBLIC, "coverage_target_public_379": COVERAGE_TARGET_PUBLIC,
                "coverage_budget_336": COVERAGE_BUDGET, "coverage_baseline_336": COVERAGE_BASELINE,
                "k_spec": K_SPEC, "conservative_delta": report["imported"]["conservative_delta"],
            },
        )
        flat = {
            "primary/slope_robustness_self_test_passes": int(report["slope_robustness_self_test_passes"]),
            "headline/slope_public": report["slope_public"],
            "headline/slope_private_oob": report["slope_private_oob"],
            "headline/slope_flattening_ratio": report["slope_flattening_ratio"],
            "headline/coverage_target_for_3p2_private": report["coverage_target_for_3p2_private"],
            "headline/coverage_target_public_379": COVERAGE_TARGET_PUBLIC,
            "headline/target_inflation": report["target_inflation"],
            "headline/budget_margin_private": report["budget_margin_private"],
            "headline/flattening_breakeven": report["flattening_breakeven"],
            "headline/flattening_margin": report["flattening_margin"],
            "headline/flattening_margin_conservative": report["flattening_margin_conservative"],
            "headline/delta_breakeven_multiplicative": report["delta_breakeven_multiplicative"],
            "headline/breakeven_multiple_of_263_collapse": report["breakeven_multiple_of_263_collapse"],
            "headline/slope_is_private_robust": int(report["slope_is_private_robust"]),
            "headline/demand_route_survives_private_oob": int(report["demand_route_survives_private_oob"]),
            "leverage/t2_public_289": report["t2_public_289"],
            "leverage/t2_public_263": report["t2_public_263"],
            "leverage/t2_private_measured_263": report["t2_private_measured_263"],
            "leverage/tps_shrink_to_3p2": report["tps_shrink_to_3p2"],
            "conservative/slope_flattening_ratio": cons["slope_flattening_ratio"],
            "conservative/slope_private_oob": cons["slope_private_oob"],
            "conservative/coverage_target_for_3p2_private": cons["coverage_target_for_3p2_private"],
            "conservative/budget_margin_private": cons["budget_margin_private"],
            "conservative/budget_fraction_used": cons["budget_fraction_used"],
            "central/budget_fraction_used": c["budget_fraction_used"],
            "tps_added_by_this_card": 0,
        }
        flat = {k: v for k, v in flat.items()
                if v is not None and not (isinstance(v, float) and math.isnan(v))}
        run.summary.update(flat)
        run.summary["verdict_band"] = report["verdict_band"]
        run.summary["recommended_action"] = report["recommended_action"]
        for k, v in report["self_test"].items():
            run.summary[f"selftest/{k}"] = int(bool(v))

        # model-corner table
        mtbl = wandb.Table(columns=["model", "t2_priv", "flattening_ratio", "slope_tps_per_cov",
                                    "cov_target_private", "budget_margin", "within_budget", "note"])
        for key in ["central_measured", "conservative_stress", "mild_survival",
                    "adversarial_floor", "benchmark_realistic"]:
            m = report["models"][key]
            mtbl.add_data(m["name"], m["t2_priv"], m["slope_flattening_ratio"],
                          m["slope_private_oob"], m["coverage_target_for_3p2_private"],
                          m["budget_margin_private"], int(m["within_336_budget"]), m["note"])
        run.log({"slope_models": mtbl})

        # delta sweep table
        stbl = wandb.Table(columns=["downstream_collapse_delta", "flattening_ratio",
                                    "slope_tps_per_cov", "cov_target_private", "budget_margin",
                                    "within_budget"])
        for s in report["delta_sweep"]:
            stbl.add_data(s["downstream_collapse_delta"], s["slope_flattening_ratio"],
                          s["slope_private_tps_per_cov"], s["coverage_target_for_3p2_private"],
                          s["budget_margin_private"], int(s["within_336_budget"]))
        run.log({"delta_sweep": stbl})

        # per-position profile table (public vs measured private)
        ptbl = wandb.Table(columns=["position_k", "a_k_public_263", "a_k_private_263",
                                    "private_minus_public"])
        for i in range(K_SPEC):
            ptbl.add_data(i + 1, PUB_AK_263[i], PRIV_AK_263[i], PRIV_AK_263[i] - PUB_AK_263[i])
        run.log({"per_position_profile": ptbl})

        rid = run.id
        print(f"[cspr] W&B run: {run.url}", flush=True)
        run.finish()
        return rid
    except Exception as exc:  # noqa: BLE001
        print(f"[cspr] wandb log failed ({exc})", flush=True)
        return None


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--private-oob-slope", action="store_true",
                    help="re-derive the coverage->gap slope under the private-OOD profile (default).")
    ap.add_argument("--anchor-263-collapse", action="store_true",
                    help="anchor the private profile on the #263 measured per-position profile (default on).")
    ap.add_argument("--conservative-delta", type=float, default=abs(RHO_COLLAPSE_MEAN_PCT_263) / 100.0,
                    help="downstream multiplicative collapse for the conservative stress model.")
    ap.add_argument("--reanalyze", action="store_true",
                    help="0-GPU reanalysis from banked constants (no wandb unless --force-wandb).")
    ap.add_argument("--force-wandb", action="store_true", help="log to wandb even under --reanalyze.")
    ap.add_argument("--self-test", action="store_true",
                    help="exit nonzero if the primary self-test fails.")
    # OPTIONAL local-A10G accept-gap leg (documented SKIP; profile directly measured in #263)
    ap.add_argument("--gpu", action="store_true", help="(optional) enable the GPU accept-gap leg.")
    ap.add_argument("--proxy", default="google/gemma-4-E4B-it-qat-w4a16-ct",
                    help="(optional) int4-ct proxy for the GPU accept-gap leg.")
    ap.add_argument("--measure-accept-gap", action="store_true",
                    help="(optional) measure per-distribution E[l] on the proxy.")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="strict-bi-verify-gemm")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="ubel/coverage-slope-private-robustness")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.gpu and args.measure_accept_gap:
        print("[cspr] NOTE: optional GPU accept-gap leg requested but SKIPPED by design -- the "
              "private per-DRAFT-POSITION conditional acceptance profile is DIRECTLY MEASURED in "
              "#263 (he7glotf: conditional_rank1_acceptance_q vs cross_check.conditional76). The "
              "private-OOD profile is NOT under-determined, and flattening_breakeven sits far below "
              "every plausible model. A fresh GPU E[l] measurement cannot move the verdict. "
              "CPU-analytic stands.", flush=True)

    report = build_report(conservative_delta=args.conservative_delta)

    wid = None
    do_wandb = (not args.no_wandb) and ((not args.reanalyze) or args.force_wandb)
    if do_wandb:
        wid = log_wandb(report, args.wandb_name, args.wandb_group)
    report["wandb_run_id"] = wid
    report["wandb_run_ids"] = [wid] if wid else []
    RESULTS_PATH.write_text(json.dumps(report, indent=2, default=str))

    c = report["models"]["central_measured"]
    cons = report["models"]["conservative_stress"]
    bar = "=" * 94
    print("\n" + bar, flush=True)
    print(" COVERAGE->GAP SLOPE PRIVATE-OOD ROBUSTNESS (PR #382, #319)", flush=True)
    print(bar, flush=True)
    print(f" slope_public (banked #379)         : {SLOPE_PUBLIC:.2f} TPS/unit cov  "
          f"(T2_pub={report['t2_public_289']:.4f} * K_cal={K_CAL})", flush=True)
    print(f" tps_shrink_to_3p2 (fixed)          : {report['tps_shrink_to_3p2']:.4f} TPS", flush=True)
    print(" --- PER-POSITION LEVERAGE (T2 = downstream tail; a_1 IRRELEVANT to slope) ---", flush=True)
    print(f"   T2_public (#263 matched repro)   : {report['t2_public_263']:.4f}", flush=True)
    print(f"   T2_private (#263 MEASURED)       : {report['t2_private_measured_263']:.4f}  "
          f"(a_1 collapse 0.729->0.598 but deep tail holds)", flush=True)
    print(" --- MODELS (flattening ratio | slope | cov_target_priv | budget_margin) ---", flush=True)
    for key in ["central_measured", "conservative_stress", "mild_survival",
                "adversarial_floor", "benchmark_realistic"]:
        m = report["models"][key]
        print(f"   {m['name']:<26s} {m['slope_flattening_ratio']:.4f} | "
              f"{m['slope_private_oob']:6.1f} | {m['coverage_target_for_3p2_private']:.4f} | "
              f"{m['budget_margin_private']:+.4f}  {'IN' if m['within_336_budget'] else 'OUT'}",
              flush=True)
    print(" --- HEADLINE (central = MEASURED #263 private profile) ---", flush=True)
    print(f"   slope_private_oob                : {report['slope_private_oob']:.2f} TPS/unit "
          f"(ratio {report['slope_flattening_ratio']:.4f})", flush=True)
    print(f"   coverage_target_for_3p2_private  : {report['coverage_target_for_3p2_private']:.4f} "
          f"(public {COVERAGE_TARGET_PUBLIC:.4f}; inflation +{report['target_inflation']:.4f})", flush=True)
    print(f"   budget_margin_private            : {report['budget_margin_private']:+.4f} "
          f"(uses {100*c['budget_fraction_used']:.1f}% of #336 +0.031)", flush=True)
    print(f"   conservative cov_target/margin   : {cons['coverage_target_for_3p2_private']:.4f} / "
          f"{cons['budget_margin_private']:+.4f} (uses {100*cons['budget_fraction_used']:.1f}%)", flush=True)
    print(" --- BREAKEVEN ---", flush=True)
    print(f"   flattening_breakeven (ratio)     : {report['flattening_breakeven']:.4f} "
          f"(downstream collapse delta {report['delta_breakeven_multiplicative']:.4f} = "
          f"{report['breakeven_multiple_of_263_collapse']:.1f}x the #263 -34.5% collapse)", flush=True)
    print(f"   flattening_margin (central)      : {report['flattening_margin']:+.4f}  "
          f"(conservative {report['flattening_margin_conservative']:+.4f})", flush=True)
    print(" --- VERDICT ---", flush=True)
    print(f"   slope_is_private_robust          : {report['slope_is_private_robust']}", flush=True)
    print(f"   demand_route_survives_private_oob: {report['demand_route_survives_private_oob']}", flush=True)
    print(f"   verdict_band                     : {report['verdict_band']}", flush=True)
    print(f"   PRIMARY slope_robustness_self_test: {report['slope_robustness_self_test_passes']}", flush=True)
    print(f"   wandb run                        : {wid}", flush=True)
    print(f"   artifacts                        : {RESULTS_PATH}", flush=True)
    print(bar + "\n", flush=True)

    marker = {
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": report["wandb_run_ids"],
        "primary_metric": {"name": "slope_robustness_self_test_passes",
                           "value": int(report["slope_robustness_self_test_passes"])},
        "test_metric": {"name": "slope_flattening_ratio",
                        "value": report["slope_flattening_ratio"]},
        "headline": {
            "slope_public": report["slope_public"],
            "slope_private_oob": report["slope_private_oob"],
            "slope_flattening_ratio": report["slope_flattening_ratio"],
            "coverage_target_for_3p2_private": report["coverage_target_for_3p2_private"],
            "target_inflation": report["target_inflation"],
            "budget_margin_private": report["budget_margin_private"],
            "flattening_breakeven": report["flattening_breakeven"],
            "flattening_margin": report["flattening_margin"],
            "slope_is_private_robust": report["slope_is_private_robust"],
            "demand_route_survives_private_oob": report["demand_route_survives_private_oob"],
            "verdict_band": report["verdict_band"]},
    }
    print("SENPAI-RESULT: " + json.dumps(marker), flush=True)

    if args.self_test and not report["slope_robustness_self_test_passes"]:
        failed = [k for k, v in report["self_test"].items() if not v]
        print(f"[cspr] SELF-TEST FAILED: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
