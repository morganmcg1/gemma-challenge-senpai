#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""FP16-verify VALID-path TPS ceiling (PR #220, stark) — CPU-only analytic synthesis.

THE QUESTION (Issue #211 cluster decision; lane-b of the two valid >500 paths)
------------------------------------------------------------------------------
Our int4-spec frontier is greedy-INVALID (#114/#192: the int4 Marlin verify GEMM
picks its split-K reduction order as f(batch width M), so AR at M=1 and verify at
M=K+1 reduce in different float order ⇒ 56% token divergence vs the int4-AR-greedy
reference). There are exactly TWO mapped routes to a VALID >500:

  lane-a  a custom batch-invariant int4 verify kernel (keep int4 speed, fix the
          kernel) — wirbel #216 buildability; ceiling 536.66 @ ≤7.33% overhead (#199/#213).
  lane-b  (THIS leg) fp16/bf16 verify — standard cuBLAS GEMM has NO M-dependent
          split-K, so it is bit-identical across batch widths ⇒ greedy-valid BY
          CONSTRUCTION, no kernel work needed. The catch is speed: fp16 is slower
          per target-forward than int4 Marlin.

The single decision-relevant question: does the fp16-verify VALID path have ANY
achievable draft that clears 500, or is its ceiling capped below 500 so that no
draft (however strong, Blackwell-trained or not) can save it?

THE CONSTRUCTION (all imported; nothing re-derived)
---------------------------------------------------
fp16-verify shares the int4 draft tree (M=32 depth-9) and the SAME E[T](λ); only
the per-step target cost changes. For greedy validity BOTH the AR fallback and the
verify must share the fp16 numerics (an int4-AR / fp16-verify mix would match
NEITHER greedy reference), so the fp16 penalty applies to the WHOLE target forward:

    step_fp16          = step_int4 · M_step              (M_step swept; lawine #221 pins it)
    fp16verify_tps(λ)  = K_cal · (E[T](λ) / step_fp16) · τ
                       = int4spec_tps(λ) / M_step        (same K_cal, E[T], τ; only step ↑)
                       = (520.9527 / M_step) · g(λ)      g(λ) = E[T](λ)/E[T](1), g(1)=1

So the draft-INDEPENDENT λ=1 cap is

    fp16verify_ceiling_at_lambda1 = 520.9527 / M_step

(at λ=1 the reach-DP E[T] saturates at the tree max regardless of draft quality, so
this is the true "no draft saves it" bound). The crossover M_step at which the λ=1
ceiling equals 500 is M_step* = 520.9527/500 = 1.0419 — below even the most
optimistic "only the verify GEMM is fp16" estimate (≈1.3).

IMPORTS (do NOT re-derive)
--------------------------
  composition  K_cal=125.26795 (#148/#169), step_int4=1.2182 (#168), τ∈{1.0,0.9924} (#181)
  int4-spec λ=1 ceiling  520.9527323111674  (#204 launch_sigma_unit_rebase, lambda1_ceiling)
  E[T](λ)      #175/#184 reach-DP forward map (the SAME machinery #199/#213 used), on the
               banked both-bugs floor/ceiling spines (compliant_spec_et_results.json — the
               reach-DP needs NO rankprobe shard, only the anchor JSONs)
  achievable-λ band  λ̂=0.34186 realistic floor (#193) / λ≈0.997 spine (land #71) / 1.0 sat

VALIDITY PREMISE (imported from Issue #211's researcher dig, NOT certified here)
fp16/bf16 cuBLAS GEMM has no M-dependent split-K ⇒ batch-invariant ⇒ AR-M=1 and
verify-M=K+1 produce identical argmax ⇒ matches the fp16-AR-greedy reference ⇒
greedy-valid. The token-identity PROOF is lawine #221's separate local measurement.

LOCAL, CPU-ONLY, ANALYTIC. No GPU / vLLM / HF Job / submission / served-file change /
official draw. BASELINE stays 481.53; adds 0 TPS; greedy/PPL untouched. Do NOT measure
the fp16 cost here — lawine PR #221 (`fp16-verify-valid-cost`) MEASURES M_step and tests
the validity premise; this leg SWEEPS M_step and #221 pins the real column. NOT a launch.

PRIMARY metric  fp16_verify_ceiling_self_test_passes
TEST    metric  fp16verify_ceiling_at_lambda1   (central M_step=1.7, τ=1; expect ≪ 500)
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
# Import the #175/#184 reach-DP + launch composition (compliant_spec_et = module C).
# et_via_reachdp / official_tps / clear_bar use the committed anchor JSONs only; they
# do NOT read the (gitignored) rankprobe shard, so the import is shard-free.
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

K_CAL = C.K_CAL                  # 125.26795005202914 (#148/#169)
STEP = C.STEP                    # 1.2182 (int4 realized step, #168)
TAU_CENTRAL = C.TAU_CENTRAL      # 1.0
TAU_CONS = C.TAU_CONS            # 0.9924 (#181)
TAU_CORNERS = C.TAU_CORNERS      # (("tau_central_1p0",1.0),("tau_conservative_0p9924",.9924))
TARGET = C.TARGET                # 500.0
BENCH_TOKENS = C.BENCH_TOKENS    # 16384
INT4_DIVERGENCE = C.KANNA114_DIVERGENCE   # 0.5608 — 56% int4-Marlin verify token divergence

# The imported int4-spec λ=1 ceiling (#204). Loaded from its committed JSON with a
# literal fallback; the value is asserted against the literal in provenance.
CEIL_INT4_LAMBDA1_LITERAL = 520.9527323111674

# PR-specified fp16/int4 whole-step multiplier sweep (lawine #221 measures which is real):
#   ≈1.3–1.5  "only the verify portion is fp16"     ;  ≈2.0–2.3  "the whole target is fp16".
M_STEP_SWEEP = (1.3, 1.5, 1.7, 2.0, 2.3)
M_STEP_CENTRAL = 1.7
# Extended set for the self-test: straddles the crossover M* so lambda_min exercises the
# finite→∅ transition (1.0/1.03 clear at λ=1; 1.05 and the whole band miss).
M_STEP_SELFTEST = (1.0, 1.03, 1.05, 1.3, 1.5, 1.7, 2.0, 2.3)

LAMBDA_SPINE = 0.997             # land #71 measured spine (q[2..7]≈0.997)
N_LAMBDA_GRID = 60               # display + monotone-in-λ self-test grid
NOSPEC_AR_TPS = 165.0            # no-spec int4 AR (VALID but slow), lawine #196 (approx)

TOL_ROUNDTRIP = 1e-6             # import round-trip (520.9527 reproduction)
TOL_PROV = 1e-9                  # reach-DP on banked spines reproduces banked E[T]


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _load_int4_ceiling() -> tuple[float, str]:
    """Import the int4-spec λ=1 ceiling 520.9527 from #204's committed result JSON."""
    p = REPO_ROOT / "research/validity/launch_sigma_unit_rebase/launch_sigma_unit_rebase_results.json"
    if p.exists():
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            v = (j.get("imported_legs_201", {}).get("lambda1_ceiling")
                 or j.get("clean_trigger", {}).get("lambda1_ceiling_mu"))
            if v is not None and _finite(v):
                return float(v), str(p.relative_to(REPO_ROOT))
        except Exception:
            pass
    return CEIL_INT4_LAMBDA1_LITERAL, "literal(#204)"


def _load_banked_spines() -> dict[str, Any]:
    """Banked both-bugs / descent floor+ceiling spines + λ̂ from #199's committed JSON."""
    p = REPO_ROOT / "research/validity/compliant_spec_et/compliant_spec_et_results.json"
    syn = json.loads(p.read_text(encoding="utf-8"))["synthesis"]
    out: dict[str, Any] = {"lambda_hat": syn["lambda_hat"], "source": str(p.relative_to(REPO_ROOT)),
                           "regimes": {}}
    for regime in ("both_bugs", "descent_only"):
        br = syn["brackets"][regime]
        out["regimes"][regime] = {
            "floor_spine": list(br["floor_spine_at_lambda_hat"]),
            "ceil_spine": list(br["ceiling_spine"]),
            "et_floor_banked": br["et_compliant_floor"],
            "et_ceiling_banked": br["et_compliant_ceiling"],
        }
    return out


# --------------------------------------------------------------------------- #
# E[T](λ) shape via the #175/#184 reach-DP on a per-depth linear self-KV-recovery
# blend (the SAME machinery #199/#213 used). g(λ)=E[T](λ)/E[T](1) is the normalized
# SHAPE; the int4-spec LEVEL is set by the imported 520.9527 ceiling, not by this curve.
# --------------------------------------------------------------------------- #
class LambdaCurve:
    def __init__(self, floor_spine: list[float], ceil_spine: list[float], lam_hat: float):
        self.floor = list(floor_spine)
        self.ceil = list(ceil_spine)
        self.lam_hat = lam_hat
        self.ceil_ge_floor_all_depths = all(c >= f - 1e-12 for f, c in zip(self.floor, self.ceil))
        self._et1 = self._et_at(1.0)            # E[T](λ=1) == reach-DP on the ceiling spine

    def _t(self, lam: float) -> float:
        return (lam - self.lam_hat) / (1.0 - self.lam_hat)

    def _spine(self, lam: float) -> list[float]:
        t = self._t(lam)
        return [(1.0 - t) * f + t * c for f, c in zip(self.floor, self.ceil)]

    def _et_at(self, lam: float) -> float:
        return C.et_via_reachdp(self._spine(lam))["et_pmf_mean"]

    def et_of_lambda(self, lam: float) -> float:
        return self._et_at(lam)

    def g_of_lambda(self, lam: float) -> float:
        return self._et_at(lam) / self._et1     # g(1)=1 exactly

    @property
    def et1(self) -> float:
        return self._et1


# --------------------------------------------------------------------------- #
# fp16-verify TPS + ceiling + lambda_min.
# --------------------------------------------------------------------------- #
def fp16verify_tps(curve: LambdaCurve, anchor_et1: float, lam: float, m_step: float,
                   tau: float) -> float:
    """K_cal·(E_int4(λ)/step_fp16)·τ, with E_int4(λ)=g(λ)·anchor_et1, step_fp16=step·M_step."""
    et = curve.g_of_lambda(lam) * anchor_et1
    return C.official_tps(et, tau, step=STEP * m_step)


def fp16verify_ceiling_at_lambda1(ceil_int4: float, m_step: float, tau: float) -> float:
    """Draft-independent λ=1 cap = (520.9527·τ)/M_step (g(1)=1)."""
    return ceil_int4 * tau / m_step


def lambda_min_clears_500(curve: LambdaCurve, anchor_et1: float, m_step: float,
                          tau: float) -> float | None:
    """Smallest λ∈[λ̂,1] whose fp16verify_tps≥500. None if even λ=1 misses; λ̂ if the floor
    already clears (λ̂ is the floor of the achievable band). Monotone bisection on g(λ)."""
    et_bar = C.clear_bar(tau) * m_step                 # E_int4 must reach this to clear 500
    et_ceil = anchor_et1                               # g(1)=1
    et_floor = curve.g_of_lambda(curve.lam_hat) * anchor_et1
    if et_ceil < et_bar - 1e-12:
        return None                                    # even λ=1 misses
    if et_floor >= et_bar - 1e-12:
        return curve.lam_hat                           # clears even at the achievable floor
    g_target = et_bar / anchor_et1
    lo, hi = curve.lam_hat, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if curve.g_of_lambda(mid) < g_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    ceil_int4, ceil_src = _load_int4_ceiling()
    banked = _load_banked_spines()
    lam_hat = banked["lambda_hat"]

    # central shape = both-bugs (the int4-spec ceiling 520.9527 is on the both-bugs σ-regime,
    # #204 provenance); descent carried as a shape-band corner.
    reg = banked["regimes"]["both_bugs"]
    curve = LambdaCurve(reg["floor_spine"], reg["ceil_spine"], lam_hat)
    reg_d = banked["regimes"]["descent_only"]
    curve_d = LambdaCurve(reg_d["floor_spine"], reg_d["ceil_spine"], lam_hat)

    # int4-spec λ=1 E[T] anchor that yields 520.9527 under the composition (τ=1):
    #   ceil_int4 = K_cal·anchor_et1/step  ⇒  anchor_et1 = ceil_int4·step/(K_cal·τ_central)
    anchor_et1 = ceil_int4 * STEP / (K_CAL * TAU_CENTRAL)
    bar1 = C.clear_bar(TAU_CENTRAL)
    bar_cons = C.clear_bar(TAU_CONS)
    mstep_crossover = ceil_int4 / TARGET               # M* where the λ=1 ceiling == 500 (τ=1)

    # provenance: reach-DP on the banked endpoint spines must reproduce #199's banked E[T];
    # and the composition must round-trip the imported 520.9527 at the anchor.
    et_floor_dp = C.et_via_reachdp(reg["floor_spine"])["et_pmf_mean"]
    et_ceil_dp = C.et_via_reachdp(reg["ceil_spine"])["et_pmf_mean"]
    provenance = {
        "int4_ceiling_source": ceil_src,
        "int4_ceiling_value": ceil_int4,
        "int4_ceiling_resid_vs_literal": abs(ceil_int4 - CEIL_INT4_LAMBDA1_LITERAL),
        "int4_anchor_et1": anchor_et1,
        "composition_roundtrips_520p95": abs(K_CAL * anchor_et1 / STEP - ceil_int4),
        "et_floor_reachdp": et_floor_dp,
        "et_floor_banked_199": reg["et_floor_banked"],
        "et_floor_resid": abs(et_floor_dp - reg["et_floor_banked"]),
        "et_ceiling_reachdp": et_ceil_dp,
        "et_ceiling_banked_199": reg["et_ceiling_banked"],
        "et_ceiling_resid": abs(et_ceil_dp - reg["et_ceiling_banked"]),
        "g_at_floor_both_bugs": curve.g_of_lambda(lam_hat),
        "g_at_floor_descent": curve_d.g_of_lambda(lam_hat),
        "g_shape_floor_band_width": abs(curve.g_of_lambda(lam_hat) - curve_d.g_of_lambda(lam_hat)),
        "ceil_ge_floor_all_depths_both_bugs": curve.ceil_ge_floor_all_depths,
    }

    # ---------- (1) validity contrast (imported premise; instruction 1) ---------- #
    validity_contrast = {
        "int4_marlin_verify": {
            "valid": False, "reason": f"{INT4_DIVERGENCE*100:.0f}% token divergence vs int4-AR "
            "greedy (#114/#192: M-dependent split-K in the Marlin verify GEMM)",
            "approx_tps": "int4-spec frontier (fast but greedy-INVALID)"},
        "fp16_bf16_verify": {
            "valid": True, "reason": "batch-invariant cuBLAS GEMM (no M-dependent split-K) ⇒ "
            "AR-M=1 and verify-M=K+1 argmax identical ⇒ matches fp16-AR-greedy reference",
            "caveat": "IMPORTED first-principles premise (Issue #211 dig), NOT certified here; "
            "lawine #221's local token-identity-vs-fp16-AR measurement is the empirical confirmation",
            "approx_tps": "ceiling 520.9527/M_step (this leg)"},
        "nospec_int4_ar": {
            "valid": True, "reason": "plain int4 autoregressive greedy — no verify GEMM, no "
            "batch-width mismatch", "approx_tps": f"~{NOSPEC_AR_TPS:.0f} (lawine #196)"},
        "fp16_verify_valid_by_construction": True,
        "fp16_verify_valid_by_construction_caveat":
            "imported premise; token-identity proof is lawine #221's separate local measurement",
    }

    # ---------- (2) ceiling vs M_step (the deliverable) ---------- #
    ceiling_vs_mstep = []
    for m in M_STEP_SWEEP:
        c1 = fp16verify_ceiling_at_lambda1(ceil_int4, m, TAU_CENTRAL)
        cc = fp16verify_ceiling_at_lambda1(ceil_int4, m, TAU_CONS)
        ceiling_vs_mstep.append({
            "M_step": m,
            "step_fp16": STEP * m,
            "fp16verify_ceiling_at_lambda1_tau1": c1,
            "fp16verify_ceiling_at_lambda1_tau_cons": cc,
            "clears_500_at_lambda1_tau1": bool(c1 >= TARGET),
            "clears_500_at_lambda1_tau_cons": bool(cc >= TARGET),
            "ceiling_times_mstep_tau1": c1 * m,                       # self-test (e): == ceil_int4
            "lambda_min_fp16verify_clears_500_tau1":
                lambda_min_clears_500(curve, anchor_et1, m, TAU_CENTRAL),
            "lambda_min_fp16verify_clears_500_tau_cons":
                lambda_min_clears_500(curve, anchor_et1, m, TAU_CONS),
            # achievable-λ-band cross-check (instruction 3)
            "fp16verify_tps_at_spine_tau1": fp16verify_tps(curve, anchor_et1, LAMBDA_SPINE, m, TAU_CENTRAL),
            "fp16verify_tps_at_floor_tau1": fp16verify_tps(curve, anchor_et1, lam_hat, m, TAU_CENTRAL),
            "fp16verify_tps_at_spine_tau_cons": fp16verify_tps(curve, anchor_et1, LAMBDA_SPINE, m, TAU_CONS),
            "fp16verify_tps_at_floor_tau_cons": fp16verify_tps(curve, anchor_et1, lam_hat, m, TAU_CONS),
        })

    # ---------- (3) λ-curve at the central M_step (display + monotone-in-λ test) ---------- #
    grid = [lam_hat + i * (1.0 - lam_hat) / N_LAMBDA_GRID for i in range(N_LAMBDA_GRID + 1)]
    lambda_curve_central = [{
        "lambda": lam,
        "g": curve.g_of_lambda(lam),
        "E_int4": curve.g_of_lambda(lam) * anchor_et1,
        "fp16verify_tps_tau1": fp16verify_tps(curve, anchor_et1, lam, M_STEP_CENTRAL, TAU_CENTRAL),
    } for lam in grid]
    monotone_tps_in_lambda = all(
        lambda_curve_central[i + 1]["fp16verify_tps_tau1"]
        >= lambda_curve_central[i]["fp16verify_tps_tau1"] - 1e-12
        for i in range(len(lambda_curve_central) - 1))

    # ---------- (4) lambda_min over the extended (straddling) self-test set ---------- #
    lambda_min_table = []
    for m in M_STEP_SELFTEST:
        c1 = fp16verify_ceiling_at_lambda1(ceil_int4, m, TAU_CENTRAL)
        lm = lambda_min_clears_500(curve, anchor_et1, m, TAU_CENTRAL)
        lambda_min_table.append({
            "M_step": m, "ceiling_at_lambda1_tau1": c1,
            "clears_at_lambda1_tau1": bool(c1 >= TARGET),
            "lambda_min_tau1": lm, "lambda_min_is_none": lm is None,
        })

    # ---------- (5) verdict ---------- #
    all_below_500 = all(r["fp16verify_ceiling_at_lambda1_tau1"] < TARGET for r in ceiling_vs_mstep)
    if all_below_500:
        verdict = "FP16VERIFY_CEILING_BELOW_500_AT_ALL_MSTEP"
    else:
        clears = [r for r in ceiling_vs_mstep if r["fp16verify_ceiling_at_lambda1_tau1"] >= TARGET]
        m_below = max(r["M_step"] for r in clears)
        lam_x = min((r["lambda_min_fp16verify_clears_500_tau1"] for r in clears
                     if r["lambda_min_fp16verify_clears_500_tau1"] is not None), default=1.0)
        verdict = f"FP16VERIFY_CLEARS_500_ABOVE_LAMBDA_{lam_x:.4f}_FOR_MSTEP_BELOW_{m_below:.3f}"

    headline = {
        "fp16verify_ceiling_at_lambda1": fp16verify_ceiling_at_lambda1(
            ceil_int4, M_STEP_CENTRAL, TAU_CENTRAL),     # TEST (central M=1.7, τ1)
        "int4_spec_lambda1_ceiling_imported": ceil_int4,
        "mstep_crossover_ceiling_500": mstep_crossover,  # 1.0419 — break-even fp16 multiplier
        "all_mstep_ceilings_below_500": all_below_500,
        "min_mstep_in_sweep": min(M_STEP_SWEEP),
        "ceiling_at_min_mstep_tau1": fp16verify_ceiling_at_lambda1(ceil_int4, min(M_STEP_SWEEP), TAU_CENTRAL),
        "fp16verify_tps_at_spine_central_tau1":
            fp16verify_tps(curve, anchor_et1, LAMBDA_SPINE, M_STEP_CENTRAL, TAU_CENTRAL),
        "fp16verify_tps_at_floor_central_tau1":
            fp16verify_tps(curve, anchor_et1, lam_hat, M_STEP_CENTRAL, TAU_CENTRAL),
        "lambda_min_at_crossover_demo_mstep_1p0":     # finite control: M=1.0 DOES clear (ceiling 520.95)
            lambda_min_clears_500(curve, anchor_et1, 1.0, TAU_CENTRAL),
    }

    # ---------- (6) self-test (PRIMARY) ---------- #
    # (a) round-trip: at M_step=1.0, λ=1, the curve reproduces the imported 520.9527 exactly.
    rt = fp16verify_tps(curve, anchor_et1, 1.0, 1.0, TAU_CENTRAL)
    cond_a = (abs(rt - ceil_int4) <= TOL_ROUNDTRIP
              and provenance["composition_roundtrips_520p95"] <= TOL_ROUNDTRIP
              and provenance["int4_ceiling_resid_vs_literal"] <= TOL_ROUNDTRIP
              and provenance["et_floor_resid"] <= TOL_PROV
              and provenance["et_ceiling_resid"] <= TOL_PROV)
    # (b) fp16verify_ceiling_at_lambda1 monotone DECREASING in M_step.
    cei = [r["fp16verify_ceiling_at_lambda1_tau1"] for r in ceiling_vs_mstep]
    cond_b = all(cei[i] > cei[i + 1] for i in range(len(cei) - 1))
    # (c) fp16verify_tps(λ) monotone INCREASING in λ.
    cond_c = bool(monotone_tps_in_lambda and curve.ceil_ge_floor_all_depths)
    # (d) lambda_min monotone INCREASING in M_step (among finite) AND = ∅ iff ceiling<500.
    finite = [(r["M_step"], r["lambda_min_tau1"]) for r in lambda_min_table
              if r["lambda_min_tau1"] is not None]
    d_mono = all(finite[i][1] <= finite[i + 1][1] + 1e-9 for i in range(len(finite) - 1))
    d_none_iff = all((r["lambda_min_is_none"]) == (r["ceiling_at_lambda1_tau1"] < TARGET)
                     for r in lambda_min_table)
    d_has_finite = len(finite) >= 1                  # the control M=1.0 must be finite
    d_has_none = any(r["lambda_min_is_none"] for r in lambda_min_table)   # band must be ∅
    cond_d = bool(d_mono and d_none_iff and d_has_finite and d_has_none)
    # (e) fp16verify_ceiling_at_lambda1 · M_step == 520.9527 at every M_step.
    cond_e = all(abs(r["ceiling_times_mstep_tau1"] - ceil_int4) <= TOL_ROUNDTRIP
                 for r in ceiling_vs_mstep)
    conditions = {
        "a_mstep1_lambda1_reproduces_int4_ceiling_520p95": bool(cond_a),
        "b_ceiling_monotone_decreasing_in_mstep": bool(cond_b),
        "c_fp16verify_tps_monotone_increasing_in_lambda": bool(cond_c),
        "d_lambda_min_monotone_in_mstep_and_none_iff_ceiling_below_500": bool(cond_d),
        "e_ceiling_times_mstep_equals_int4_ceiling": bool(cond_e),
        "f_nan_clean": True,                         # set by caller after the payload walk
    }

    handoff = _handoff(ceil_int4, headline, mstep_crossover, all_below_500)
    return {
        "self_test": {
            "fp16_verify_ceiling_self_test_passes": bool(all(conditions.values())),
            "conditions": conditions,
        },
        "test_metric": {"fp16verify_ceiling_at_lambda1": headline["fp16verify_ceiling_at_lambda1"]},
        "headline": headline,
        "validity_contrast": validity_contrast,
        "ceiling_vs_mstep": ceiling_vs_mstep,
        "lambda_min_selftest_table": lambda_min_table,
        "lambda_curve_central_mstep_1p7": lambda_curve_central,
        "monotone_tps_in_lambda": monotone_tps_in_lambda,
        "verdict": verdict,
        "honest_band": _honest_band(ceil_int4, anchor_et1, curve, curve_d, lam_hat),
        "composition": {
            "K_cal": K_CAL, "step_int4": STEP, "tau_central": TAU_CENTRAL,
            "tau_conservative": TAU_CONS, "target_official": TARGET,
            "int4_spec_lambda1_ceiling": ceil_int4, "int4_anchor_et1": anchor_et1,
            "clear500_bar_et_tau1": bar1, "clear500_bar_et_tau_cons": bar_cons,
            "lambda_hat": lam_hat, "lambda_spine_land71": LAMBDA_SPINE,
            "mstep_sweep": list(M_STEP_SWEEP), "mstep_central": M_STEP_CENTRAL,
            "mstep_crossover_ceiling_500": mstep_crossover, "bench_tokens": BENCH_TOKENS,
        },
        "provenance": provenance,
        "lambda_hat": lam_hat,
        "handoff_line": handoff,
    }


def _honest_band(ceil_int4: float, anchor_et1: float, curve: LambdaCurve,
                 curve_d: LambdaCurve, lam_hat: float) -> dict[str, Any]:
    corners = {}
    for tag, m in (("best_mstep_1p3", 1.3), ("central_mstep_1p7", 1.7), ("worst_mstep_2p3", 2.3)):
        corners[tag] = {
            "M_step": m,
            "ceiling_at_lambda1_tau1": fp16verify_ceiling_at_lambda1(ceil_int4, m, TAU_CENTRAL),
            "tps_at_spine_tau1": fp16verify_tps(curve, anchor_et1, LAMBDA_SPINE, m, TAU_CENTRAL),
            "tps_at_floor_tau1": fp16verify_tps(curve, anchor_et1, lam_hat, m, TAU_CENTRAL),
        }
    return {
        "a_mstep_band_is_lawine_221s_to_pin":
            "M_step swept {1.3..2.3}; lawine PR #221 (fp16-verify-valid-cost) MEASURES it.",
        "b_fp16_validity_is_imported_premise":
            "fp16/bf16 batch-invariance ⇒ greedy-valid is a first-principles premise (Issue #211 "
            "dig); lawine #221's token-identity-vs-fp16-AR check is the empirical confirmation.",
        "c_et_and_520p95_imported_unchanged":
            "E[T](λ) reach-DP (#175/#184) and the int4-spec λ=1 ceiling 520.9527 (#204) imported, "
            "not re-derived.",
        "d_lambda1_ceiling_is_draft_independent":
            "this assumes the SAME draft tree (M=32 depth-9); a stronger Blackwell draft raises "
            "E[T](λ) at fixed λ (shifts the curve UP), BUT at λ=1 E[T] saturates at the tree max "
            "regardless of draft quality, so fp16verify_ceiling_at_lambda1=520.9527/M_step is the "
            "true draft-INDEPENDENT cap — no draft, however strong, can lift it above 500 once "
            "M_step>1.0419.",
        "e_shape_borrows_compliant_both_bugs_reachdp":
            "the sub-λ=1 shape g(λ)=E[T](λ)/E[T](1) borrows the #199/#213 both-bugs reach-DP "
            f"profile (descent shape differs by {abs(curve.g_of_lambda(lam_hat) - curve_d.g_of_lambda(lam_hat)):.2e} "
            "at the floor); the λ=1 ceiling and the verdict depend ONLY on the imported 520.9527, "
            "not on the shape.",
        "mstep_band_corners": corners,
    }


def _handoff(ceil_int4: float, headline: dict[str, Any], mstep_crossover: float,
             all_below_500: bool) -> str:
    c17 = headline["fp16verify_ceiling_at_lambda1"]
    clears_phrase = ("NEVER clears 500 at any λ" if all_below_500
                     else "clears 500 for a strong-enough draft")
    unlock = ("does NOT" if all_below_500 else "DOES")
    route = ("the ONLY" if all_below_500 else "the remaining")
    return (
        f"HAND-OFF (fern #185 + Issue #211 + wirbel #216): the fp16-verify VALID path "
        f"(greedy-valid by construction, no kernel needed) has λ=1 ceiling 520.9527/M_step = "
        f"{c17:.2f} TPS at the central M_step=1.7 (lawine #221 pins M_step), so it {clears_phrase} "
        f"— its draft-INDEPENDENT λ=1 cap is below 500 for every M_step ≥ 1.3 (≫ the {mstep_crossover:.4f} "
        f"break-even), meaning the Issue #211 Blackwell draft-training {unlock} unlock a valid 500 "
        f"via fp16-verify, and {route} valid-500 route is wirbel #216's batch-invariant int4 kernel. "
        f"This leg adds 0 TPS and authorizes nothing. NOT a launch."
    )


# --------------------------------------------------------------------------- #
# NaN guard + report + W&B (mirrors #213; never fatal).
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
    print("FP16-VERIFY VALID-PATH TPS CEILING (PR #220, stark) — Issue #211 lane-b, CPU-only",
          flush=True)
    print("=" * 96, flush=True)
    print(f"  int4-spec λ=1 ceiling (imported #204) = {comp['int4_spec_lambda1_ceiling']:.4f}   "
          f"anchor E[T](1)={comp['int4_anchor_et1']:.5f}   λ̂={comp['lambda_hat']:.5f}", flush=True)
    print(f"  clear-500 bar E[T]={comp['clear500_bar_et_tau1']:.4f} (τ1)   "
          f"crossover M*={comp['mstep_crossover_ceiling_500']:.4f} (ceiling==500 ⇔ M_step=M*)",
          flush=True)
    print("-" * 96, flush=True)
    print("  VALIDITY CONTRAST (imported premise):", flush=True)
    vc = syn["validity_contrast"]
    print(f"    int4 Marlin verify   : VALID={vc['int4_marlin_verify']['valid']}  "
          f"({vc['int4_marlin_verify']['reason'][:60]}…)", flush=True)
    print(f"    fp16/bf16 verify     : VALID={vc['fp16_bf16_verify']['valid']}  (by construction; "
          f"lawine #221 confirms token-identity)", flush=True)
    print(f"    no-spec int4 AR      : VALID={vc['nospec_int4_ar']['valid']}  "
          f"({vc['nospec_int4_ar']['approx_tps']})", flush=True)
    print("-" * 96, flush=True)
    print("    M_step  step_fp16  ceiling@λ1(τ1)  ceiling@λ1(τc)  clr500@λ1  TPS@spine  TPS@floor  λ_min",
          flush=True)
    for r in syn["ceiling_vs_mstep"]:
        lm = r["lambda_min_fp16verify_clears_500_tau1"]
        lm_s = f"{lm:.4f}" if lm is not None else "  ∅  "
        print(f"     {r['M_step']:.2f}   {r['step_fp16']:7.4f}   "
              f"{r['fp16verify_ceiling_at_lambda1_tau1']:11.2f}   "
              f"{r['fp16verify_ceiling_at_lambda1_tau_cons']:11.2f}   "
              f"{str(r['clears_500_at_lambda1_tau1']):>6}   "
              f"{r['fp16verify_tps_at_spine_tau1']:8.2f}   "
              f"{r['fp16verify_tps_at_floor_tau1']:8.2f}   {lm_s}", flush=True)
    print("-" * 96, flush=True)
    print("  λ_min self-test (straddling M*):", flush=True)
    for r in syn["lambda_min_selftest_table"]:
        lm = r["lambda_min_tau1"]
        lm_s = f"{lm:.4f}" if lm is not None else "∅"
        print(f"     M={r['M_step']:.3f}  ceiling@λ1={r['ceiling_at_lambda1_tau1']:7.2f}  "
              f"clears@λ1={str(r['clears_at_lambda1_tau1']):>5}  λ_min={lm_s}", flush=True)
    print("-" * 96, flush=True)
    print(f"  PRIMARY fp16_verify_ceiling_self_test_passes = "
          f"{st['fp16_verify_ceiling_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"        - {k}: {v}", flush=True)
    print(f"  TEST fp16verify_ceiling_at_lambda1 (central M=1.7, τ1) = "
          f"{syn['test_metric']['fp16verify_ceiling_at_lambda1']:.4f}", flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print("=" * 96, flush=True)
    print(f"\n  {syn['handoff_line']}\n", flush=True)


def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[fp16-verify-ceiling] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    comp = syn["composition"]
    run = init_wandb_run(
        job_type="fp16-verify-valid-ceiling",
        agent="stark",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["fp16-verify-valid-ceiling", "issue-211", "validity-gate", "lane-b",
              "ceiling", "mstep-sweep", "draft-independent-cap"],
        config={
            "K_cal": K_CAL, "step_int4": STEP, "tau_central": TAU_CENTRAL,
            "tau_conservative": TAU_CONS, "target_official": TARGET,
            "int4_spec_lambda1_ceiling": comp["int4_spec_lambda1_ceiling"],
            "int4_anchor_et1": comp["int4_anchor_et1"],
            "lambda_hat": comp["lambda_hat"], "lambda_spine_land71": comp["lambda_spine_land71"],
            "mstep_sweep": comp["mstep_sweep"], "mstep_central": comp["mstep_central"],
            "mstep_crossover_ceiling_500": comp["mstep_crossover_ceiling_500"],
            "bench_tokens": BENCH_TOKENS, "wandb_group": args.wandb_group,
            "companion_empirical_leg": "lawine PR #221 (fp16-verify-valid-cost)",
        },
    )
    if run is None:
        print("[fp16-verify-ceiling] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    st, hd = syn["self_test"], syn["headline"]
    summary: dict[str, Any] = {
        "fp16_verify_ceiling_self_test_passes":
            int(bool(st["fp16_verify_ceiling_self_test_passes"])),
        "fp16verify_ceiling_at_lambda1": hd["fp16verify_ceiling_at_lambda1"],
        "int4_spec_lambda1_ceiling_imported": hd["int4_spec_lambda1_ceiling_imported"],
        "mstep_crossover_ceiling_500": hd["mstep_crossover_ceiling_500"],
        "all_mstep_ceilings_below_500": int(bool(hd["all_mstep_ceilings_below_500"])),
        "ceiling_at_min_mstep_1p3_tau1": hd["ceiling_at_min_mstep_tau1"],
        "fp16verify_tps_at_spine_central_tau1": hd["fp16verify_tps_at_spine_central_tau1"],
        "fp16verify_tps_at_floor_central_tau1": hd["fp16verify_tps_at_floor_central_tau1"],
        "lambda_min_demo_mstep_1p0": hd["lambda_min_at_crossover_demo_mstep_1p0"],
        "fp16_verify_valid_by_construction":
            int(bool(syn["validity_contrast"]["fp16_verify_valid_by_construction"])),
        "verdict_ceiling_below_500_at_all_mstep":
            int(syn["verdict"] == "FP16VERIFY_CEILING_BELOW_500_AT_ALL_MSTEP"),
        "clear500_bar_et_tau1": comp["clear500_bar_et_tau1"],
        "lambda_hat": comp["lambda_hat"],
        "provenance_et_ceiling_resid": syn["provenance"]["et_ceiling_resid"],
        "provenance_composition_roundtrips_520p95": syn["provenance"]["composition_roundtrips_520p95"],
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    # per-M_step λ=1 ceiling + spine/floor TPS as logged scalars.
    for r in syn["ceiling_vs_mstep"]:
        tag = f"{r['M_step']:.1f}".replace(".", "p")
        summary[f"ceiling_lambda1_tau1_mstep_{tag}"] = r["fp16verify_ceiling_at_lambda1_tau1"]
        summary[f"tps_spine_tau1_mstep_{tag}"] = r["fp16verify_tps_at_spine_tau1"]
        summary[f"tps_floor_tau1_mstep_{tag}"] = r["fp16verify_tps_at_floor_tau1"]
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="fp16_verify_ceiling_result", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    print(f"[fp16-verify-ceiling] wandb logged {len(summary)} keys", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="fp16-verify-valid-ceiling")
    args = ap.parse_args(argv)

    syn = synthesize()

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 220, "agent": "stark",
        "kind": "fp16-verify-valid-ceiling", "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    syn["self_test"]["conditions"]["f_nan_clean"] = not nan_paths
    syn["self_test"]["fp16_verify_ceiling_self_test_passes"] = bool(
        all(syn["self_test"]["conditions"].values()))
    if nan_paths:
        print(f"[fp16-verify-ceiling] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[fp16-verify-ceiling] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = syn["self_test"]["fp16_verify_ceiling_self_test_passes"] and payload["nan_clean"]
        print(f"[fp16-verify-ceiling] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
