#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Kernel-overhead budget vs λ (PR #213, wirbel) — CPU-only analytic synthesis.

THE QUESTION (own follow-up #3 of #199; Issue #192 lane-a capstone)
------------------------------------------------------------------
My #199 (`compliant-spec-et-ceiling`, MERGED, `wdyqnx3g`) proved a token-identical
batch-invariant int4 *verify* CAN clear 500 — both-bugs ceiling 536.66 TPS (lower-CI
525.73 > 500) — but ONLY if the compliant verify kernel inflates per-step cost by
≤ 7.332% (both-bugs) / ≤ 4.12% (descent-only). That budget is the λ=1 (FULL self-KV
recovery) BEST CASE; at the realistic λ̂=0.342 the compliant FLOOR (416.31 TPS) misses 500
outright, so a *free* kernel already fails there. The kernel-overhead budget is therefore a
FUNCTION of the achieved self-KV-recovery λ, and land #71 needs the WHOLE curve, not the two
endpoints: how fast does ``max_kernel_overhead_pct(λ)`` open as λ improves 0.342 → 1.0?

THE MECHANISM (imported, NOT re-derived)
----------------------------------------
#199 banked two endpoint spines per regime: the realistic FLOOR spine (its
``floor_spine_at_lambda_hat`` at λ̂=0.342, reach-DP E[T] = 4.04848 both-bugs / 3.92943
descent) and the compliant CEILING spine (its rank-1-coverage ``ceiling_spine``, reach-DP
E[T] = 5.21888 / 5.06287). #178/#193's graded self-KV recovery scales the deep-rung
acceptance LINEARLY between two ladders (``spine_from_profile``: q_d(λ)=(1−λ)q_floor+λ q_full,
then the #175/#184 reach-DP pmf-mean). We carry the SAME mechanism over the segment the PR
frames — λ̂=0.342 → 1.0 — by linearly blending #199's banked FLOOR spine (anchored at
t=0 ⇔ λ=λ̂) into #199's banked CEILING spine (t=1 ⇔ λ=1):

    t(λ)   = (λ − λ̂) / (1 − λ̂)                       # λ ∈ [λ̂, 1] ↦ t ∈ [0, 1]
    spine  = (1 − t)·floor_spine + t·ceiling_spine     # per-depth linear self-KV recovery
    E[T](λ)= reach_DP_pmf_mean(spine)                  # #175/#184 — the SAME DP #199 used

This round-trips BOTH #199 endpoints EXACTLY (the reach-DP on the banked floor/ceiling spines
is bit-identical to #199's banked E[T]; verified by self-test) and is monotone increasing in
λ because ceiling_spine ≥ floor_spine at every depth. It does NOT re-derive 536.66, 416.31,
4.04848, 5.21888, K_cal, the step, or τ.

THE DELIVERABLE
---------------
official = K_cal·(E[T]/step)·τ ; K_cal=125.268, step=1.2182, τ∈{1.0, 0.9924}. A real
batch-invariant kernel inflates the verify step by (1+o): official = K_cal·E[T]/(step·(1+o))·τ
≥ 500 ⇔ o ≤ E[T]/bar(τ) − 1. So at each λ:

    max_kernel_overhead_pct(λ) = (E[T](λ)/bar(τ) − 1)·100      # the kernel-dev budget

Below ``lambda_crit_clears_500_zero_overhead`` (where zero-overhead TPS = 500 ⇔ E[T]=bar)
NO overhead budget exists — even a free kernel misses. kanna #122's off-the-shelf
``VLLM_BATCH_INVARIANT=1`` is a +51.78% NON-working reference; we report the λ (if any
physical λ≤1) at which the budget would reach it.

LOCAL, CPU-ONLY, ANALYTIC. No GPU / vLLM / HF Job / submission / kernel build / served-file
change. BASELINE stays 481.53; adds **0 TPS**; greedy/PPL untouched. The rank-1-coverage /
λ / zero-overhead optimisms are carried as a NOTED band (see #199), not a re-derivation.
**NOT a launch. NOT open2.** Directly serves Issue #192's only compliant 500-lane.

PRIMARY metric  kernel_budget_lambda_self_test_passes
TEST    metric  lambda_crit_clears_500_zero_overhead  (both-bugs, τ=1)
"""
from __future__ import annotations

import argparse
import importlib.util
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
# Import #199's compliant-spec machinery (path-based; do NOT re-derive it).
# Re-running C.synthesize() is the canonical "import #199's banked result": it is
# #199's OWN code on #199's OWN committed inputs → bit-identical banked spines/E[T].
# --------------------------------------------------------------------------- #
def _load(name: str, relpath: str):
    path = REPO_ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("compliant_spec_et", "research/validity/compliant_spec_et/compliant_spec_et.py")
ETSM = C.ETSM  # et_second_moment (same module instance #199 uses)

# Pinned launch composition (imported via #199 → #172/#148/#168/#181).
K_CAL = C.K_CAL                  # 125.26795005202914
STEP = C.STEP                    # 1.2182
TAU_CENTRAL = C.TAU_CENTRAL      # 1.0
TAU_CONS = C.TAU_CONS            # 0.9924
TAU_CORNERS = C.TAU_CORNERS      # (("tau_central_1p0", 1.0), ("tau_conservative_0p9924", .9924))
TARGET = C.TARGET                # 500.0
BENCH_TOKENS = C.BENCH_TOKENS    # 16384
Z95 = C.Z95                      # 1.959963984540054
KANNA122_OFFSHELF_OVERHEAD = C.KANNA122_OFFSHELF_OVERHEAD  # 0.5178 (NON-working ref)

# PR-specified λ grid (grid[0] == the realistic floor λ̂≈0.342; we anchor it at the
# precise λ̂ so the floor row round-trips #199's 4.04848 / 416.31 BIT-EXACTLY).
LAMBDA_GRID_NOMINAL = [0.342, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
REGIMES = ("both_bugs", "descent_only")

# #199 banked endpoint budgets / E[T] (round-trip targets for the self-test).
BANKED = {
    "both_bugs": {"et_floor": 4.048484687770039, "et_ceiling": 5.21887717841078,
                  "budget_floor_tau1": -16.738568764737806, "budget_ceiling_tau1": 7.331808522875782,
                  "tps_ceiling_tau1": 536.6590426143789, "tps_floor_tau1": 416.307156176311,
                  "ceiling_lcb_tau1": 525.7290377676009},
    "descent_only": {"et_floor": 3.9294296647453835, "et_ceiling": 5.062874725337895,
                     "budget_floor_tau1": -19.187063047728405, "budget_ceiling_tau1": 4.123450699935671,
                     "tps_ceiling_tau1": 520.6172534996784, "tps_floor_tau1": 404.06468476135797,
                     "ceiling_lcb_tau1": 509.76365791080394},
}

TOL_ROUNDTRIP = 1e-6   # endpoint budget/E[T]/TPS reproduction vs #199 banked values.


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# The unified per-regime E[T](λ) blend (imported DP; same object as #199).
# --------------------------------------------------------------------------- #
class LambdaCurve:
    """E[T](λ) over the λ̂→1 segment via a per-depth linear self-KV-recovery blend."""

    def __init__(self, floor_spine: list[float], ceil_spine: list[float], lam_hat: float):
        self.floor = list(floor_spine)
        self.ceil = list(ceil_spine)
        self.lam_hat = lam_hat
        # monotone precondition: ceiling acceptance ≥ floor acceptance at every depth.
        self.ceil_ge_floor_all_depths = all(
            c >= f - 1e-12 for f, c in zip(self.floor, self.ceil))

    def t_of_lambda(self, lam: float) -> float:
        return (lam - self.lam_hat) / (1.0 - self.lam_hat)

    def spine_at_t(self, t: float) -> list[float]:
        return [(1.0 - t) * f + t * c for f, c in zip(self.floor, self.ceil)]

    def spine_of_lambda(self, lam: float) -> list[float]:
        return self.spine_at_t(self.t_of_lambda(lam))

    def reachdp_at_t(self, t: float) -> dict[str, Any]:
        return C.et_via_reachdp(self.spine_at_t(t))

    def et_of_lambda(self, lam: float) -> float:
        return self.reachdp_at_t(self.t_of_lambda(lam))["et_pmf_mean"]

    def t_prob_saturation(self) -> float:
        """Smallest t>0 at which the binding depth's blended acceptance reaches 1.0 — the
        absolute physical wall of the blend (beyond it some prob would exceed 1)."""
        caps = [(1.0 - f) / (c - f) for f, c in zip(self.floor, self.ceil) if (c - f) > 1e-12]
        return min(caps) if caps else float("inf")


def budget_pct(et: float, tau: float) -> float:
    return (et / C.clear_bar(tau) - 1.0) * 100.0


# --------------------------------------------------------------------------- #
# Solvers: λ_crit (zero-overhead clear-500) and λ for an arbitrary budget target.
# --------------------------------------------------------------------------- #
def solve_lambda_for_et(curve: LambdaCurve, et_target: float,
                        lo: float, hi: float) -> float | None:
    """Monotone bisection on λ∈[lo,hi] for E[T](λ)=et_target. None if not bracketed."""
    f_lo = curve.et_of_lambda(lo) - et_target
    f_hi = curve.et_of_lambda(hi) - et_target
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    if (f_lo > 0.0) == (f_hi > 0.0):
        return None                                    # not bracketed in [lo,hi]
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (curve.et_of_lambda(mid) - et_target > 0.0) == (f_lo > 0.0):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def lambda_crit_clears_500(curve: LambdaCurve, tau: float) -> float | None:
    """Min λ at which zero-overhead compliant-spec TPS = 500 (E[T]=bar(τ))."""
    return solve_lambda_for_et(curve, C.clear_bar(tau), curve.lam_hat, 1.0)


def lambda_for_budget(curve: LambdaCurve, target_budget_pct: float,
                      tau: float) -> dict[str, Any]:
    """λ at which max_kernel_overhead_pct(λ)=target (e.g. kanna #122's +51.78%).

    The budget is monotone in λ, so this inverts E[T]=bar·(1+target/100). We allow t>1
    (λ>1, UNPHYSICAL — beyond full self-KV recovery) up to the prob-saturation wall, and
    flag physicality. If the target exceeds the budget even at the wall, it is unreachable.
    """
    bar = C.clear_bar(tau)
    et_target = bar * (1.0 + target_budget_pct / 100.0)
    t_wall = curve.t_prob_saturation()
    lam_wall = curve.lam_hat + t_wall * (1.0 - curve.lam_hat)
    et_wall = curve.reachdp_at_t(t_wall)["et_pmf_mean"] if math.isfinite(t_wall) else float("inf")
    budget_at_wall = budget_pct(et_wall, tau) if math.isfinite(et_wall) else float("inf")
    reachable = et_wall >= et_target
    lam = solve_lambda_for_et(curve, et_target, curve.lam_hat, lam_wall) if reachable else None
    return {
        "target_budget_pct": target_budget_pct,
        "et_target": et_target,
        "lambda_for_target": lam,
        "is_physical_lambda_le_1": bool(lam is not None and lam <= 1.0),
        "reachable_within_prob_saturation": bool(reachable),
        "lambda_prob_saturation_wall": lam_wall,
        "et_at_prob_saturation_wall": et_wall,
        "max_budget_pct_at_prob_saturation": budget_at_wall,
    }


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize(shard: Path, max_records: int | None = None) -> dict[str, Any]:
    # (0) Import #199's banked object (its OWN code on its OWN inputs) + provenance check.
    c199 = C.synthesize(shard, max_records)
    lam_hat = c199["lambda_hat"]
    banked_json = _load_banked_json()

    bar1 = C.clear_bar(TAU_CENTRAL)
    bar_cons = C.clear_bar(TAU_CONS)
    # grid[0] is the precise λ̂ (the PR's nominal 0.342 == λ̂, absorbed into it).
    grid = sorted(set([lam_hat] + [x for x in LAMBDA_GRID_NOMINAL if x > lam_hat + 1e-3]))

    regimes: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    for regime in REGIMES:
        br = c199["brackets"][regime]
        floor_spine = list(br["floor_spine_at_lambda_hat"])
        ceil_spine = list(br["ceiling_spine"])
        curve = LambdaCurve(floor_spine, ceil_spine, lam_hat)

        # provenance: reach-DP on the banked endpoint spines must reproduce #199 E[T].
        et_floor_dp = C.et_via_reachdp(floor_spine)["et_pmf_mean"]
        et_ceil_dp = curve.reachdp_at_t(1.0)["et_pmf_mean"]
        provenance[regime] = {
            "et_floor_reachdp": et_floor_dp,
            "et_floor_banked_199": br["et_compliant_floor"],
            "et_floor_resid": abs(et_floor_dp - br["et_compliant_floor"]),
            "et_ceiling_reachdp": et_ceil_dp,
            "et_ceiling_banked_199": br["et_compliant_ceiling"],
            "et_ceiling_resid": abs(et_ceil_dp - br["et_compliant_ceiling"]),
            "banked_json_et_floor": banked_json["brackets"][regime]["et_compliant_floor"]
            if banked_json else None,
            "banked_json_et_ceiling": banked_json["brackets"][regime]["et_compliant_ceiling"]
            if banked_json else None,
            "ceiling_ge_floor_all_depths": curve.ceil_ge_floor_all_depths,
        }

        # (1) the table: E[T](λ) → TPS(λ) → max_kernel_overhead_pct(λ), both τ corners.
        table = []
        prev_et = None
        monotone_et = True
        for lam in grid:
            rd = curve.reachdp_at_t(curve.t_of_lambda(lam))
            et = rd["et_pmf_mean"]
            if prev_et is not None and et < prev_et - 1e-12:
                monotone_et = False
            prev_et = et
            row = {
                "lambda": lam,
                "is_lambda_hat": bool(abs(lam - lam_hat) < 1e-12),
                "E_T": et,
                "sigma_L": rd["sigma_L"],
                "tau_central_1p0": {
                    "official_tps": C.official_tps(et, TAU_CENTRAL),
                    "max_kernel_overhead_pct": budget_pct(et, TAU_CENTRAL),
                    "clears_500_zero_overhead": bool(C.official_tps(et, TAU_CENTRAL) >= TARGET),
                },
                "tau_conservative_0p9924": {
                    "official_tps": C.official_tps(et, TAU_CONS),
                    "max_kernel_overhead_pct": budget_pct(et, TAU_CONS),
                    "clears_500_zero_overhead": bool(C.official_tps(et, TAU_CONS) >= TARGET),
                },
            }
            table.append(row)
        # budget monotone in λ ⇔ E[T] monotone (budget is linear-increasing in E[T]).
        monotone_budget = all(
            table[i]["tau_central_1p0"]["max_kernel_overhead_pct"]
            <= table[i + 1]["tau_central_1p0"]["max_kernel_overhead_pct"] + 1e-12
            for i in range(len(table) - 1))

        # (2) critical λ (zero-overhead clear-500) + λ for kanna #122's +51.78%.
        lam_crit = {tag: lambda_crit_clears_500(curve, tau) for tag, tau in TAU_CORNERS}
        lam_122 = {tag: lambda_for_budget(curve, KANNA122_OFFSHELF_OVERHEAD * 100.0, tau)
                   for tag, tau in TAU_CORNERS}

        # (3) endpoint anchors (must round-trip #199 EXACTLY).
        et_hat = curve.et_of_lambda(lam_hat)
        et_one = curve.et_of_lambda(1.0)
        anchors = {
            "overhead_budget_at_lambda_hat_0342": {
                "lambda": lam_hat, "E_T": et_hat,
                "tau_central_1p0": budget_pct(et_hat, TAU_CENTRAL),
                "tau_conservative_0p9924": budget_pct(et_hat, TAU_CONS),
                "official_tps_tau1": C.official_tps(et_hat, TAU_CENTRAL),
                "budget_le_zero_tau1": bool(budget_pct(et_hat, TAU_CENTRAL) <= 0.0),
                "roundtrips_199_floor_budget":
                    abs(budget_pct(et_hat, TAU_CENTRAL) - BANKED[regime]["budget_floor_tau1"]),
                "roundtrips_199_floor_tps":
                    abs(C.official_tps(et_hat, TAU_CENTRAL) - BANKED[regime]["tps_floor_tau1"]),
            },
            "overhead_budget_at_lambda_1": {
                "lambda": 1.0, "E_T": et_one,
                "tau_central_1p0": budget_pct(et_one, TAU_CENTRAL),
                "tau_conservative_0p9924": budget_pct(et_one, TAU_CONS),
                "official_tps_tau1": C.official_tps(et_one, TAU_CENTRAL),
                "roundtrips_199_ceiling_budget":
                    abs(budget_pct(et_one, TAU_CENTRAL) - BANKED[regime]["budget_ceiling_tau1"]),
                "roundtrips_199_ceiling_tps":
                    abs(C.official_tps(et_one, TAU_CENTRAL) - BANKED[regime]["tps_ceiling_tau1"]),
            },
        }

        # ceiling (λ=1) finite-sample CI — round-trips #199's LCB (richness anchor).
        ceil_ci = ETSM.finite_sample_tps_ci(
            et_one, curve.reachdp_at_t(1.0)["sigma_L"], BENCH_TOKENS, STEP, TAU_CENTRAL, Z95)

        regimes[regime] = {
            "floor_spine_at_lambda_hat": floor_spine,
            "ceiling_spine": ceil_spine,
            "overhead_budget_vs_lambda": table,
            "monotone_E_T_in_lambda": monotone_et,
            "monotone_budget_in_lambda": monotone_budget,
            "lambda_crit_clears_500_zero_overhead": lam_crit,
            "lambda_for_122_kernel_to_clear": lam_122,
            "endpoint_anchors": anchors,
            "ceiling_lambda1_finite_sample_ci_tau1": ceil_ci,
        }

    head = regimes["both_bugs"]
    lambda_crit_headline = head["lambda_crit_clears_500_zero_overhead"]["tau_central_1p0"]

    # ---------- self-test (PRIMARY) ---------- #
    bb, des = regimes["both_bugs"], regimes["descent_only"]
    cond_a = (bb["endpoint_anchors"]["overhead_budget_at_lambda_1"]["roundtrips_199_ceiling_budget"]
              <= TOL_ROUNDTRIP and
              bb["endpoint_anchors"]["overhead_budget_at_lambda_1"]["roundtrips_199_ceiling_tps"]
              <= TOL_ROUNDTRIP)
    cond_b = (des["endpoint_anchors"]["overhead_budget_at_lambda_1"]["roundtrips_199_ceiling_budget"]
              <= TOL_ROUNDTRIP)
    cond_c = all(
        regimes[r]["endpoint_anchors"]["overhead_budget_at_lambda_hat_0342"]["budget_le_zero_tau1"]
        and regimes[r]["endpoint_anchors"]["overhead_budget_at_lambda_hat_0342"][
            "roundtrips_199_floor_budget"] <= TOL_ROUNDTRIP
        for r in REGIMES)
    cond_d = all(regimes[r]["monotone_budget_in_lambda"] and regimes[r]["monotone_E_T_in_lambda"]
                 for r in REGIMES)
    cond_prov = all(
        provenance[r]["et_floor_resid"] <= 1e-9 and provenance[r]["et_ceiling_resid"] <= 1e-9
        and provenance[r]["ceiling_ge_floor_all_depths"] for r in REGIMES)
    conditions = {
        "a_lambda1_bothbugs_reproduces_199_ceiling_7p332_536p66": bool(cond_a),
        "b_lambda1_descent_reproduces_199_4p12": bool(cond_b),
        "c_lambda_hat_budget_le_zero_roundtrips_199_floor": bool(cond_c),
        "d_max_kernel_overhead_pct_monotone_increasing_in_lambda": bool(cond_d),
        "provenance_reachdp_on_banked_spines_reproduces_199_ET": bool(cond_prov),
        "e_nan_clean": True,   # set by the caller after the full payload walk.
    }

    handoff = _handoff(regimes, lam_hat, lambda_crit_headline)
    return {
        "self_test": {
            "kernel_budget_lambda_self_test_passes": bool(all(conditions.values())),
            "conditions": conditions,
        },
        "test_metric": {"lambda_crit_clears_500_zero_overhead": lambda_crit_headline},
        "headline": {
            "lambda_crit_clears_500_zero_overhead_both_bugs_tau1": lambda_crit_headline,
            "lambda_crit_clears_500_zero_overhead_descent_tau1":
                des["lambda_crit_clears_500_zero_overhead"]["tau_central_1p0"],
            "overhead_budget_at_lambda_hat_both_bugs_tau1":
                bb["endpoint_anchors"]["overhead_budget_at_lambda_hat_0342"]["tau_central_1p0"],
            "overhead_budget_at_lambda_1_both_bugs_tau1":
                bb["endpoint_anchors"]["overhead_budget_at_lambda_1"]["tau_central_1p0"],
            "overhead_budget_at_lambda_1_descent_tau1":
                des["endpoint_anchors"]["overhead_budget_at_lambda_1"]["tau_central_1p0"],
            "off_the_shelf_122_clears_at_physical_lambda_both_bugs_tau1":
                bb["lambda_for_122_kernel_to_clear"]["tau_central_1p0"]["is_physical_lambda_le_1"],
            "max_budget_pct_at_prob_saturation_both_bugs_tau1":
                bb["lambda_for_122_kernel_to_clear"]["tau_central_1p0"][
                    "max_budget_pct_at_prob_saturation"],
        },
        "regimes": regimes,
        "provenance": provenance,
        "lambda_hat": lam_hat,
        "lambda_grid": grid,
        "composition": {
            "K_cal": K_CAL, "step": STEP, "tau_central": TAU_CENTRAL,
            "tau_conservative": TAU_CONS, "target_official": TARGET,
            "clear500_bar_et_tau1": bar1, "clear500_bar_et_tau_cons": bar_cons,
            "bench_tokens": BENCH_TOKENS,
            "kanna122_offshelf_overhead_nonworking_ref": KANNA122_OFFSHELF_OVERHEAD,
        },
        "optimism_band_note": (
            "Three #199 optimisms carried as a NOTED band (not re-derived): (i) rank-1 "
            "coverage 0.7304 over-counts the true compliant accept (the rankprobe's true "
            "token is the batch-VARIANT int4 argmax, not a clean batch-invariant/AR greedy "
            "argmax), so the λ=1 ceiling is an UPPER bound; (ii) λ=1 is full self-KV recovery "
            "vs the realistic λ̂=0.342; (iii) zero kernel overhead. The budget CURVE inherits "
            "all three — read max_kernel_overhead_pct(λ) at the ACHIEVED λ, and treat the "
            "ceiling end as optimistic."),
        "verdict": _verdict(regimes, lam_hat),
        "handoff_line": handoff,
    }


def _load_banked_json() -> dict[str, Any] | None:
    path = REPO_ROOT / "research/validity/compliant_spec_et/compliant_spec_et_results.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)["synthesis"]


def _verdict(regimes: dict[str, Any], lam_hat: float) -> str:
    crit_bb = regimes["both_bugs"]["lambda_crit_clears_500_zero_overhead"]["tau_central_1p0"]
    if crit_bb is None:
        return "NO-LAMBDA-CLEARS-500"
    if crit_bb <= lam_hat:
        return "BUDGET-OPEN-AT-LAMBDA-HAT"
    return "BUDGET-OPENS-ONLY-ABOVE-LAMBDA-CRIT"     # the #199 picture: λ_crit > λ̂


def _handoff(regimes: dict[str, Any], lam_hat: float, lam_crit_bb: float | None) -> str:
    bb = regimes["both_bugs"]
    b_one = bb["endpoint_anchors"]["overhead_budget_at_lambda_1"]["tau_central_1p0"]
    crit_des = regimes["descent_only"]["lambda_crit_clears_500_zero_overhead"]["tau_central_1p0"]
    crit_bb_s = f"{lam_crit_bb:.4f}" if lam_crit_bb is not None else "NONE"
    crit_des_s = f"{crit_des:.4f}" if crit_des is not None else "NONE"
    return (
        f"COMPLIANT-SPEC KERNEL-OVERHEAD BUDGET vs λ (Issue #192 lane-a capstone): the "
        f"batch-invariant verify kernel's overhead budget opens from ≤0 at λ̂={lam_hat:.3f} "
        f"(the realistic FLOOR already misses 500 — even a free kernel fails) to "
        f"{b_one:.2f}% at λ=1 (full self-KV recovery, both-bugs). The zero-overhead compliant "
        f"path first clears 500 at λ_crit={crit_bb_s} (both-bugs) / {crit_des_s} (descent), "
        f"τ=1. So land #71 must BOTH build self-KV recovery λ above {crit_bb_s} AND hold the "
        f"batch-invariant verify kernel below max_kernel_overhead_pct(λ_achieved) — kanna "
        f"#122's off-the-shelf +51.78% clears at NO physical λ≤1 (the λ=1 budget is only "
        f"{b_one:.2f}%, ~{KANNA122_OFFSHELF_OVERHEAD * 100 / max(b_one, 1e-9):.1f}× over). "
        f"This is the kernel-dev target for the only compliant 500-lane. HONEST SCOPE: the "
        f"rank-1-coverage λ=1 end is an UPPER bound (#199's three optimisms, noted band); the "
        f"curve adds 0 TPS and authorizes nothing. NOT a launch. NOT open2."
    )


# --------------------------------------------------------------------------- #
# NaN guard + report + W&B (mirrors #199; never fatal).
# --------------------------------------------------------------------------- #
def _assert_nan_clean(payload: dict, path: str = "result") -> list[str]:
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


def _print_report(syn: dict) -> None:
    st, hd, comp = syn["self_test"], syn["headline"], syn["composition"]
    print("\n" + "=" * 96, flush=True)
    print("KERNEL-OVERHEAD BUDGET vs λ (PR #213, wirbel) — Issue #192 lane-a capstone, CPU-only",
          flush=True)
    print("=" * 96, flush=True)
    print(f"  λ̂={syn['lambda_hat']:.5f}   clear-500 bar E[T]={comp['clear500_bar_et_tau1']:.4f} "
          f"(τ=1) / {comp['clear500_bar_et_tau_cons']:.4f} (τ=cons)", flush=True)
    for regime in REGIMES:
        rg = syn["regimes"][regime]
        prov = syn["provenance"][regime]
        print("-" * 96, flush=True)
        print(f"  [{regime}]  provenance: reach-DP floor resid={prov['et_floor_resid']:.1e}  "
              f"ceiling resid={prov['et_ceiling_resid']:.1e}  "
              f"ceil≥floor-all-depths={prov['ceiling_ge_floor_all_depths']}", flush=True)
        print(f"      λ      E[T]     TPS@τ1    budget@τ1%   budget@cons%   clears500(0-ovh,τ1)",
              flush=True)
        for row in rg["overhead_budget_vs_lambda"]:
            tag = " (λ̂)" if row["is_lambda_hat"] else ""
            c1, cc = row["tau_central_1p0"], row["tau_conservative_0p9924"]
            print(f"     {row['lambda']:.4f}  {row['E_T']:7.4f}  {c1['official_tps']:8.2f}  "
                  f"{c1['max_kernel_overhead_pct']:+10.3f}  {cc['max_kernel_overhead_pct']:+11.3f}"
                  f"   {str(c1['clears_500_zero_overhead']):>5}{tag}", flush=True)
        crit = rg["lambda_crit_clears_500_zero_overhead"]
        l122 = rg["lambda_for_122_kernel_to_clear"]["tau_central_1p0"]
        crit1 = crit["tau_central_1p0"]
        critc = crit["tau_conservative_0p9924"]
        print(f"      λ_crit (zero-overhead clears 500): τ1="
              f"{crit1:.4f}" + (f"  τ_cons={critc:.4f}" if critc is not None else "  τ_cons=NONE"),
              flush=True)
        print(f"      kanna#122 +51.78%: physical λ≤1 clears={l122['is_physical_lambda_le_1']}  "
              f"max-budget@prob-saturation={l122['max_budget_pct_at_prob_saturation']:.2f}% "
              f"(<51.78 ⇒ never)", flush=True)
        ci = rg["ceiling_lambda1_finite_sample_ci_tau1"]
        print(f"      λ=1 ceiling CI(τ1): {ci['central_tps']:.2f} "
              f"[{ci['ci_lower_tps']:.2f}, {ci['ci_upper_tps']:.2f}]  LCB clears 500="
              f"{ci['lower_clears_500']}", flush=True)
    print("-" * 96, flush=True)
    print(f"  PRIMARY kernel_budget_lambda_self_test_passes = "
          f"{st['kernel_budget_lambda_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"        - {k}: {v}", flush=True)
    print(f"  TEST lambda_crit_clears_500_zero_overhead (both-bugs, τ1) = "
          f"{syn['test_metric']['lambda_crit_clears_500_zero_overhead']:.5f}", flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print("=" * 96, flush=True)
    print(f"\n  HAND-OFF: {syn['handoff_line']}\n", flush=True)


def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[kernel-budget-lambda] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    run = init_wandb_run(
        job_type="compliant-spec-et-ceiling",
        agent="wirbel",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["compliant-spec-et-ceiling", "issue-192", "batch-invariant", "validity-gate",
              "kernel-budget", "lambda-curve"],
        config={
            "K_cal": K_CAL, "step": STEP, "tau_central": TAU_CENTRAL,
            "tau_conservative": TAU_CONS, "target_official": TARGET,
            "lambda_hat": syn["lambda_hat"], "lambda_grid": syn["lambda_grid"],
            "bench_tokens": BENCH_TOKENS,
            "kanna122_offshelf_overhead": KANNA122_OFFSHELF_OVERHEAD,
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[kernel-budget-lambda] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    st, hd = syn["self_test"], syn["headline"]
    bb, des = syn["regimes"]["both_bugs"], syn["regimes"]["descent_only"]
    summary: dict[str, Any] = {
        # PRIMARY + TEST
        "kernel_budget_lambda_self_test_passes":
            int(bool(st["kernel_budget_lambda_self_test_passes"])),
        "lambda_crit_clears_500_zero_overhead": hd["lambda_crit_clears_500_zero_overhead_both_bugs_tau1"],
        # critical-λ both regimes / both τ
        "lambda_crit_both_bugs_tau1": bb["lambda_crit_clears_500_zero_overhead"]["tau_central_1p0"],
        "lambda_crit_both_bugs_tau_cons": bb["lambda_crit_clears_500_zero_overhead"]["tau_conservative_0p9924"],
        "lambda_crit_descent_tau1": des["lambda_crit_clears_500_zero_overhead"]["tau_central_1p0"],
        "lambda_crit_descent_tau_cons": des["lambda_crit_clears_500_zero_overhead"]["tau_conservative_0p9924"],
        # endpoint anchors (round-trip #199)
        "overhead_budget_at_lambda_hat_both_bugs_tau1": hd["overhead_budget_at_lambda_hat_both_bugs_tau1"],
        "overhead_budget_at_lambda_1_both_bugs_tau1": hd["overhead_budget_at_lambda_1_both_bugs_tau1"],
        "overhead_budget_at_lambda_1_descent_tau1": hd["overhead_budget_at_lambda_1_descent_tau1"],
        "overhead_budget_at_lambda_hat_descent_tau1":
            des["endpoint_anchors"]["overhead_budget_at_lambda_hat_0342"]["tau_central_1p0"],
        # off-the-shelf #122 reachability
        "off_the_shelf_122_clears_at_physical_lambda_both_bugs_tau1":
            int(bool(hd["off_the_shelf_122_clears_at_physical_lambda_both_bugs_tau1"])),
        "max_budget_pct_at_prob_saturation_both_bugs_tau1":
            hd["max_budget_pct_at_prob_saturation_both_bugs_tau1"],
        # ceiling CI round-trip (#199 LCB anchors)
        "ceiling_lambda1_lcb_tps_both_bugs": bb["ceiling_lambda1_finite_sample_ci_tau1"]["ci_lower_tps"],
        "ceiling_lambda1_lcb_tps_descent": des["ceiling_lambda1_finite_sample_ci_tau1"]["ci_lower_tps"],
        # monotonicity
        "monotone_budget_both_bugs": int(bool(bb["monotone_budget_in_lambda"])),
        "monotone_budget_descent": int(bool(des["monotone_budget_in_lambda"])),
        # provenance residuals
        "provenance_et_ceiling_resid_both_bugs": syn["provenance"]["both_bugs"]["et_ceiling_resid"],
        "provenance_et_floor_resid_both_bugs": syn["provenance"]["both_bugs"]["et_floor_resid"],
        # bars / composition
        "clear500_bar_et_tau1": syn["composition"]["clear500_bar_et_tau1"],
        "lambda_hat": syn["lambda_hat"],
        "verdict_budget_opens_only_above_lambda_crit":
            int(syn["verdict"] == "BUDGET-OPENS-ONLY-ABOVE-LAMBDA-CRIT"),
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    # per-λ budget curve (both-bugs, τ1) as logged scalars for a quick W&B view.
    for row in bb["overhead_budget_vs_lambda"]:
        key = f"budget_pct_bb_tau1_lambda_{row['lambda']:.3f}".replace(".", "p")
        summary[key] = row["tau_central_1p0"]["max_kernel_overhead_pct"]
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="kernel_budget_lambda_result", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    print(f"[kernel-budget-lambda] wandb logged {len(summary)} keys", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--shard", type=Path, default=C.RANKPROBE_SHARD,
                    help="in-scope PR#86 rankprobe shard (read-only; #199 source)")
    ap.add_argument("--max-records", type=int, default=None, help="debug: cap records parsed")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="compliant-spec-et-ceiling")
    args = ap.parse_args(argv)

    syn = synthesize(args.shard, args.max_records)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 213, "agent": "wirbel",
        "kind": "kernel-budget-lambda", "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    syn["self_test"]["conditions"]["e_nan_clean"] = not nan_paths
    syn["self_test"]["kernel_budget_lambda_self_test_passes"] = bool(
        all(syn["self_test"]["conditions"].values()))
    if nan_paths:
        print(f"[kernel-budget-lambda] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "kernel_budget_lambda_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[kernel-budget-lambda] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = syn["self_test"]["kernel_budget_lambda_self_test_passes"] and payload["nan_clean"]
        print(f"[kernel-budget-lambda] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
