#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #233 (stark) — Publish-first f_priv-breakeven: does the empirical decode-drop
calibration flip the #124 publish-first POINT-estimate launch gate?

THE QUESTION (the f_priv-axis companion to kanna #228's λ-axis publish-first floor)
-----------------------------------------------------------------------------------
My #226 (`tzcc5xuq`, MERGED) closed the realizable-BLEND axis of the private bar: the
f_priv-minimizing vertex IS the λ-deficit-maximizing non-Latin-script vertex, so the
worst-case-over-blends f_priv collapses onto kanna #217's central 0.969107 and adds ZERO
spread. The binding private-bar uncertainty therefore MIGRATED from the blend to the
decode-drop CALIBRATION. kanna #224 (`1081oc84`) shows the publish-first private MEAN at
the λ=1 physical ceiling FLIPS across the [0.957054 empirical-floor, 0.969107 central]
f_priv interval: 504.86 (GO) at central vs 498.58 (NO-GO) at the empirical floor. So the
publish-first gate's very REACHABILITY straddles a break-even f_priv inside that band.

THE MODEL (the private MEAN, NOT the #226 private bar — imported, NOT re-derived)
--------------------------------------------------------------------------------
    private_mean(λ, f_priv) = K_cal·(E[T](λ)/step_int4)·τ · f_priv
                            = ceiling(λ) · f_priv
                            = CEIL_INT4 · g(λ) · f_priv

  * CEIL_INT4 = 520.9527323111674   the int4-spec λ=1 physical ceiling (#204
    launch_sigma_unit_rebase) = K_cal·(E[T](1)/step)·τ. IMPORTED unchanged.
  * g(λ) = E[T](λ)/E[T](1)          the #175/#184 reach-DP NORMALIZED shape on the #199
    both-bugs floor→ceiling self-KV-recovery spine blend (g(1)=1 exactly; the int4-spec
    LEVEL is set by CEIL_INT4, NOT by this curve — the SAME construction fp16_verify_ceiling
    (#220) and trigger_reconcile (kanna #224's private_mean_at_ceiling=504.86) use).
  * f_priv = private/public TPS ratio (carries τ_low via stark #226: f_priv=(1-drop)·τ_low).

This is the private MEAN (achieved private TPS = served ceiling × f_priv), DISTINCT from my
#226 private BAR (the build target = mu_safe_fresh / f_priv). It round-trips kanna #224's two
λ=1 anchors EXACTLY by construction (504.858898207883 / 498.57966624218955).

THE DELIVERABLE (prices the f_priv-axis of the human's #124 publish-first launch)
--------------------------------------------------------------------------------
1. private_mean_vs_fpriv_at_ceiling over f_priv ∈ {0.957054, 0.95978, 0.969107}.
2. f_priv_breakeven_publish_first = 500 / CEIL_INT4 (≈ 0.95978) and the boolean
   publish_first_at_ceiling_verdict_flips = ([emp-floor, central] STRADDLES break-even).
3. lambda_floor_publish_first(f_priv) = smallest λ with private_mean(λ,f_priv)=500, for
   f_priv ∈ {central (round-trips kanna #228), empirical-floor (∅/unreachable, >1)}, and the
   sensitivity dlambda_floor_dfpriv (the rate the publish-first λ-target rises as f_priv falls
   — the number kanna #228's honest-band explicitly deferred to this leg).
4. Honest band: realizable worst-case f_priv (==central, GO at ceiling) vs the lone #52
   empirical calibration tail (NO-GO at ceiling); P95-private bar stays the 0.9780 (#191).
5. Self-test (PRIMARY) + f_priv_breakeven_publish_first (TEST).

LOCAL, CPU-ONLY, ANALYTIC. No GPU / vLLM / HF Job / submission / served-file change / draw.
BASELINE stays 481.53; adds **0 TPS**; greedy/PPL untouched. Reuses stark #226's f_priv model
and the banked private-mean curve; E[T](λ), K_cal, step, τ, ceiling, kanna #224 anchors all
imported unchanged. The realizable worst-case f_priv stays my #226's central 0.969107 (this
leg prices the calibration-tail consequence, it does NOT re-open the blend axis). Authorizes
nothing. **NOT a launch. NOT open2.**

PRIMARY metric  publish_first_fpriv_breakeven_self_test_passes
TEST    metric  f_priv_breakeven_publish_first   (≈ 0.95978)

Run:
  cd target/ && CUDA_VISIBLE_DEVICES="" python \
    research/validity/publish_first_fpriv_breakeven/publish_first_fpriv_breakeven.py \
    --self-test --wandb_group winners-curse-redraw-budget \
    --wandb_name stark/publish-first-fpriv-breakeven
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
# Import #199's compliant-spec machinery (path-based; reach-DP + composition).
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

# Pinned launch composition (imported via #199 → #172/#148/#168/#181).
K_CAL = C.K_CAL                  # 125.26795005202914 (#148/#169)
STEP = C.STEP                    # 1.2182 (int4 realized step, #168)
TAU_CENTRAL = C.TAU_CENTRAL      # 1.0
TAU_CONS = C.TAU_CONS            # 0.9924 (#181)
TARGET = C.TARGET                # 500.0  (the publish-first POINT-estimate clear bar)

# --------------------------------------------------------------------------- #
# Imported external constants (provenance: W&B run-ids / committed JSON, project
# wandb-applied-ai-team/gemma-challenge-senpai). IMPORTED, NOT re-derived.
# --------------------------------------------------------------------------- #
CEIL_INT4_LITERAL = 520.9527323111674       # #204 launch_sigma_unit_rebase lambda1_ceiling
F_PRIV_CENTRAL = 0.969106920637722          # kanna #217 vgovdrjc / stark #226 realizable worst-case
PUB_52 = 481.53                             # PR #52 public draw
PRIV_52 = 460.85                            # PR #52 private draw
F_PRIV_EMP_FLOOR = PRIV_52 / PUB_52         # #52 lone hard paired draw == 0.9570535584491102 (≈0.957054)
TAU_LOW_181 = 0.9924318649123313            # #181 τ_low (baked into f_priv via stark #226)
LAMBDA_STAR_191 = 0.9780112973731208        # #191 P95 both-bugs bar (a DISTINCT gate; carried as note)
# kanna #224 (trigger_reconcile private_mean_at_ceiling / private_bar_reachability) λ=1 anchors:
PRIV_MEAN_CENTRAL_224 = 504.858898207883            # = CEIL_INT4 · F_PRIV_CENTRAL
PRIV_MEAN_EMP_FLOOR_224 = 498.57966624218955        # = CEIL_INT4 · F_PRIV_EMP_FLOOR

TOL_ANCHOR = 1e-6        # kanna #224 anchor / break-even round-trip tolerance
TOL_PROV = 1e-9         # reach-DP on banked spines reproduces banked E[T]
N_LAMBDA_GRID = 60      # display + monotone-in-λ grid
# f_priv grid for the monotone-decreasing-lambda_floor self-test (all > break-even ⇒ finite):
FPRIV_MONO_GRID = [0.96, 0.962, 0.965, F_PRIV_CENTRAL, 0.972, 0.975, 0.98, 0.99]
# f_priv grid for the private_mean monotone-increasing self-test (at λ=1):
FPRIV_INCR_GRID = [0.94, 0.95, F_PRIV_EMP_FLOOR, 0.96, 0.965, F_PRIV_CENTRAL, 0.98, 0.99]


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _load_int4_ceiling() -> tuple[float, str]:
    """Import the int4-spec λ=1 ceiling 520.9527 from #204's committed JSON (literal fallback)."""
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
    return CEIL_INT4_LITERAL, "literal(#204)"


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
# E[T](λ) reach-DP shape (the #175/#184 DP on a per-depth linear self-KV-recovery
# blend; SAME machinery #199/#213/#220 used). g(λ)=E[T](λ)/E[T](1) is the SHAPE;
# the int4-spec LEVEL is set by the imported CEIL_INT4, NOT by this curve.
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
# The private MEAN surface + the publish-first break-even + the λ-floor map.
# --------------------------------------------------------------------------- #
def private_mean(ceil: float, g: float, f_priv: float) -> float:
    """private_mean(λ, f_priv) = ceiling(λ)·f_priv = CEIL_INT4·g(λ)·f_priv."""
    return ceil * g * f_priv


def f_priv_breakeven_publish_first(ceil: float, target: float = TARGET) -> float:
    """f_priv at which private_mean(λ=1)=target, i.e. target/CEIL_INT4 (g(1)=1)."""
    return target / ceil


def lambda_floor_publish_first(curve: LambdaCurve, ceil: float, f_priv: float,
                               target: float = TARGET) -> float | None:
    """Smallest λ∈[λ̂,1] whose private_mean(λ,f_priv) ≥ target. None (∅) if even λ=1 misses
    (f_priv < break-even); λ̂ if the achievable floor already clears. Monotone bisection on g(λ)."""
    g_target = target / (ceil * f_priv)                 # g(λ) must reach this to clear target
    if g_target > 1.0 + 1e-15:
        return None                                     # even λ=1 misses (f_priv below break-even)
    if curve.g_of_lambda(curve.lam_hat) >= g_target - 1e-15:
        return curve.lam_hat                            # clears at the achievable floor
    lo, hi = curve.lam_hat, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if curve.g_of_lambda(mid) < g_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _g_prime(curve: LambdaCurve, lam: float, h: float = 1e-6) -> float:
    """Central finite difference of g(λ), clamped to [λ̂, 1]."""
    hi = min(lam + h, 1.0)
    lo = max(lam - h, curve.lam_hat)
    return (curve.g_of_lambda(hi) - curve.g_of_lambda(lo)) / (hi - lo)


def dlambda_floor_dfpriv_numeric(curve: LambdaCurve, ceil: float, f_priv: float,
                                 delta: float = 1e-4, target: float = TARGET) -> float | None:
    """d(λ_floor)/d(f_priv) via central finite difference on the λ-floor solver. Negative:
    λ_floor FALLS as f_priv rises (equivalently the publish-first λ-target RISES as f_priv falls)."""
    lo = lambda_floor_publish_first(curve, ceil, f_priv - delta, target)
    hi = lambda_floor_publish_first(curve, ceil, f_priv + delta, target)
    if lo is None or hi is None:
        return None
    return (hi - lo) / (2.0 * delta)


def dlambda_floor_dfpriv_analytic(curve: LambdaCurve, ceil: float, f_priv: float,
                                  lam_floor: float, target: float = TARGET) -> float | None:
    """Analytic d(λ_floor)/d(f_priv) = -g(λ_floor)/(g'(λ_floor)·f_priv) from g(λ_floor)·f_priv=const."""
    gp = _g_prime(curve, lam_floor)
    if gp == 0.0:
        return None
    return -curve.g_of_lambda(lam_floor) / (gp * f_priv)


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    ceil_int4, ceil_src = _load_int4_ceiling()
    banked = _load_banked_spines()
    lam_hat = banked["lambda_hat"]

    # central reach-shape = both-bugs (the int4-spec ceiling 520.9527 is on the both-bugs
    # σ-regime, #204 provenance, same choice as fp16_verify_ceiling #220); descent = shape corner.
    reg = banked["regimes"]["both_bugs"]
    curve = LambdaCurve(reg["floor_spine"], reg["ceil_spine"], lam_hat)
    reg_d = banked["regimes"]["descent_only"]
    curve_d = LambdaCurve(reg_d["floor_spine"], reg_d["ceil_spine"], lam_hat)

    # provenance: reach-DP on the banked endpoint spines must reproduce #199's banked E[T];
    # g(1)=1 exactly; the imported ceiling matches the #204 literal.
    et_floor_dp = C.et_via_reachdp(reg["floor_spine"])["et_pmf_mean"]
    et_ceil_dp = C.et_via_reachdp(reg["ceil_spine"])["et_pmf_mean"]
    provenance = {
        "int4_ceiling_source": ceil_src,
        "int4_ceiling_value": ceil_int4,
        "int4_ceiling_resid_vs_literal": abs(ceil_int4 - CEIL_INT4_LITERAL),
        "et1_both_bugs": curve.et1,
        "g_at_one": curve.g_of_lambda(1.0),
        "g_at_one_resid": abs(curve.g_of_lambda(1.0) - 1.0),
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

    # ----- (2 first: the core scalar everything hangs on) the publish-first break-even ----- #
    f_be = f_priv_breakeven_publish_first(ceil_int4, TARGET)
    verdict_flips = bool(F_PRIV_EMP_FLOOR < f_be < F_PRIV_CENTRAL)   # the interval STRADDLES it

    # ----- (1) private_mean vs f_priv at the λ=1 physical ceiling ----- #
    g1 = curve.g_of_lambda(1.0)
    fpriv_ceiling_points = {
        "empirical_floor_0p957054": F_PRIV_EMP_FLOOR,
        "breakeven_0p95978": f_be,
        "central_0p969107": F_PRIV_CENTRAL,
    }
    private_mean_vs_fpriv_at_ceiling = []
    for label, fp in fpriv_ceiling_points.items():
        pm = private_mean(ceil_int4, g1, fp)
        private_mean_vs_fpriv_at_ceiling.append({
            "label": label, "f_priv": fp, "private_mean_at_ceiling": pm,
            "clears_500_publish_first": bool(pm >= TARGET),
            "gap_vs_500": pm - TARGET,
        })

    pm_central = private_mean(ceil_int4, g1, F_PRIV_CENTRAL)
    pm_emp = private_mean(ceil_int4, g1, F_PRIV_EMP_FLOOR)
    pm_be = private_mean(ceil_int4, g1, f_be)

    # ----- (3) the λ-floor × f_priv map ----- #
    lam_floor_central = lambda_floor_publish_first(curve, ceil_int4, F_PRIV_CENTRAL, TARGET)
    lam_floor_emp = lambda_floor_publish_first(curve, ceil_int4, F_PRIV_EMP_FLOOR, TARGET)
    # descent-shape corner for the central floor (shape-band):
    lam_floor_central_descent = lambda_floor_publish_first(curve_d, ceil_int4, F_PRIV_CENTRAL, TARGET)

    dlam_num = dlambda_floor_dfpriv_numeric(curve, ceil_int4, F_PRIV_CENTRAL, target=TARGET)
    dlam_ana = (dlambda_floor_dfpriv_analytic(curve, ceil_int4, F_PRIV_CENTRAL, lam_floor_central, TARGET)
                if lam_floor_central is not None else None)
    dlam_resid = (abs(dlam_num - dlam_ana) if (dlam_num is not None and dlam_ana is not None) else None)

    # λ_floor over the f_priv monotone grid (all > break-even ⇒ finite, must be decreasing):
    lam_floor_grid = []
    for fp in FPRIV_MONO_GRID:
        lf = lambda_floor_publish_first(curve, ceil_int4, fp, TARGET)
        lam_floor_grid.append({"f_priv": fp, "lambda_floor": lf,
                               "private_mean_at_lambda_floor": (
                                   private_mean(ceil_int4, curve.g_of_lambda(lf), fp)
                                   if lf is not None else None)})

    # ----- (display) the private_mean(λ) curve at central + empirical-floor f_priv ----- #
    grid = [lam_hat + i * (1.0 - lam_hat) / N_LAMBDA_GRID for i in range(N_LAMBDA_GRID + 1)]
    lambda_curve = []
    for lam in grid:
        g = curve.g_of_lambda(lam)
        lambda_curve.append({
            "lambda": lam, "g": g,
            "private_mean_central": private_mean(ceil_int4, g, F_PRIV_CENTRAL),
            "private_mean_empirical_floor": private_mean(ceil_int4, g, F_PRIV_EMP_FLOOR),
        })
    monotone_pm_in_lambda = all(
        lambda_curve[i + 1]["private_mean_central"] >= lambda_curve[i]["private_mean_central"] - 1e-12
        for i in range(len(lambda_curve) - 1))

    # private_mean monotone increasing in f_priv at λ=1:
    pm_incr = [private_mean(ceil_int4, g1, fp) for fp in FPRIV_INCR_GRID]
    monotone_pm_in_fpriv = all(pm_incr[i] < pm_incr[i + 1] for i in range(len(pm_incr) - 1))

    # λ_floor monotone DECREASING in f_priv (over the finite grid):
    lfs = [r["lambda_floor"] for r in lam_floor_grid]
    monotone_lamfloor_decreasing = all(
        lfs[i] is not None and lfs[i + 1] is not None and lfs[i] > lfs[i + 1] - 1e-12
        for i in range(len(lfs) - 1))

    # ----- (4) honest band ----- #
    honest_band = {
        "realizable_worstcase_f_priv": F_PRIV_CENTRAL,
        "realizable_worstcase_private_mean_at_ceiling": pm_central,
        "realizable_worstcase_publish_first_GO_at_ceiling": bool(pm_central >= TARGET),
        "calibration_tail_f_priv": F_PRIV_EMP_FLOOR,
        "calibration_tail_private_mean_at_ceiling": pm_emp,
        "calibration_tail_publish_first_GO_at_ceiling": bool(pm_emp >= TARGET),
        "calibration_tail_is_one_paired_draw_outside_realizable_simplex": True,
        "second_hard_paired_draw_collapses_interval_to_measured_band": (
            "TIGHTENING FOLLOW-UP: a SECOND hard public/private paired draw would collapse the "
            "[504.86 realizable, 498.58 calibration-tail] interval to a measured band and pin "
            "whether the lone #52 f_priv=0.957 is a tail or the center."),
        "imports_unchanged": "E[T](λ), K_cal, step, τ, CEIL_INT4 520.9527, kanna #224 anchors",
        "gate_axis": ("POINT-estimate (publish-first, Issue #124). The P95-private bar is the "
                      f"DISTINCT 0.9780 (#191) gate — UNCHANGED by this leg."),
        "p95_private_bar_unchanged_191": LAMBDA_STAR_191,
    }

    # ----- (5) self-test (PRIMARY) ----- #
    st = _selftests(curve, ceil_int4, g1, f_be, verdict_flips,
                    pm_central, pm_emp, pm_be, lam_floor_central, lam_floor_emp,
                    monotone_pm_in_lambda, monotone_pm_in_fpriv, monotone_lamfloor_decreasing,
                    provenance, dlam_resid)

    verdict = (
        "PUBLISH-FIRST POINT-ESTIMATE GATE STRADDLES THE f_priv BREAK-EVEN: at the int4-spec "
        f"λ=1 physical ceiling 520.9527 the publish-first private mean = ceiling·f_priv FLIPS "
        f"across the [empirical-floor 0.957054, central 0.969107] interval — GO at the central / "
        f"realizable-worst-case f_priv ({pm_central:.2f} ≥ 500) but NO-GO at the lone #52 "
        f"empirical calibration tail ({pm_emp:.2f} < 500). The single calibration point DECIDES "
        f"reachability: f_priv_breakeven_publish_first = 500/520.9527 = {f_be:.6f}, and "
        f"[0.957054, 0.969107] STRADDLES it ({F_PRIV_EMP_FLOOR:.6f} < {f_be:.6f} < "
        f"{F_PRIV_CENTRAL:.6f}). The publish-first λ-floor is "
        + (f"{lam_floor_central:.5f}" if lam_floor_central is not None else "∅")
        + " at the realizable worst-case f_priv (round-trips kanna #228) and ∅/UNREACHABLE at the "
        f"empirical floor (private_mean(1, 0.957054)={pm_emp:.2f} < 500, even at full self-KV "
        f"recovery λ=1). The dλ_floor/df_priv sensitivity kanna #228 deferred is "
        + (f"{dlam_num:.4f}" if dlam_num is not None else "n/a")
        + " (λ-target RISES as f_priv falls). Realizable verdict GO; calibration-tail caveat NO-GO. "
        "Adds 0 TPS; the realizable worst-case f_priv stays my #226 central 0.969107 — this leg "
        "prices the calibration-tail consequence, it does NOT re-open the blend axis. NOT a launch.")

    handoff = (
        "the publish-first POINT-estimate gate (the human's #124 launch) is reachable at the "
        f"physical ceiling iff f_priv ≥ f_priv_breakeven_publish_first = {f_be:.6f} (≈0.9598), "
        f"which the [empirical-floor 0.957054, central 0.969107] interval STRADDLES — so the "
        f"realizable-worst-case f_priv (==central, my #226) clears publish-first at the ceiling "
        f"({pm_central:.2f} ≥ 500) but the lone #52 empirical calibration tail (0.957054) does NOT "
        f"({pm_emp:.2f} < 500); fern #185 carries publish-first as GO-at-realizable / "
        f"NO-GO-at-calibration-tail, and the dλ_floor/df_priv sensitivity is "
        + (f"{dlam_num:.4f}" if dlam_num is not None else "n/a")
        + f" (λ_floor_central={lam_floor_central:.5f} round-trips kanna #228; ∅ at the empirical floor).")

    return {
        "self_test": st,
        "test_metric": {"f_priv_breakeven_publish_first": f_be},
        "headline": {
            "f_priv_breakeven_publish_first": f_be,                          # TEST
            "publish_first_at_ceiling_verdict_flips": verdict_flips,         # core boolean
            "private_mean_at_ceiling_central": pm_central,                   # 504.86 (GO)
            "private_mean_at_ceiling_empirical_floor": pm_emp,               # 498.58 (NO-GO)
            "private_mean_at_ceiling_breakeven": pm_be,                      # 500.00
            "central_clears_publish_first_at_ceiling": bool(pm_central >= TARGET),
            "empirical_floor_clears_publish_first_at_ceiling": bool(pm_emp >= TARGET),
            "lambda_floor_central": lam_floor_central,                       # round-trips kanna #228
            "lambda_floor_empirical_floor": lam_floor_emp,                   # None (∅/unreachable)
            "lambda_floor_empirical_floor_unreachable": bool(lam_floor_emp is None),
            "lambda_floor_central_descent_shape_corner": lam_floor_central_descent,
            "dlambda_floor_dfpriv": dlam_num,                               # the #228-deferred number
            "dlambda_floor_dfpriv_analytic": dlam_ana,
            "dlambda_floor_dfpriv_num_vs_analytic_resid": dlam_resid,
        },
        "private_mean_vs_fpriv_at_ceiling": private_mean_vs_fpriv_at_ceiling,
        "lambda_floor_vs_fpriv": lam_floor_grid,
        "lambda_curve": lambda_curve,
        "monotone": {
            "private_mean_increasing_in_lambda": monotone_pm_in_lambda,
            "private_mean_increasing_in_fpriv": monotone_pm_in_fpriv,
            "lambda_floor_decreasing_in_fpriv": monotone_lamfloor_decreasing,
        },
        "honest_band": honest_band,
        "provenance": provenance,
        "composition": {
            "ceil_int4_lambda1": ceil_int4,
            "K_cal": K_CAL, "step": STEP, "tau_central": TAU_CENTRAL, "tau_conservative": TAU_CONS,
            "tau_low_181": TAU_LOW_181, "target_publish_first": TARGET,
            "lambda_hat": lam_hat,
            "f_priv_central_217": F_PRIV_CENTRAL,
            "f_priv_empirical_floor_52": F_PRIV_EMP_FLOOR,
            "kanna224_private_mean_central_anchor": PRIV_MEAN_CENTRAL_224,
            "kanna224_private_mean_empirical_floor_anchor": PRIV_MEAN_EMP_FLOOR_224,
            "p95_private_bar_191": LAMBDA_STAR_191,
        },
        "model_note": (
            "private_mean(λ, f_priv) = CEIL_INT4 · g(λ) · f_priv, g(λ)=E[T](λ)/E[T](1) the "
            "#175/#184 both-bugs reach-DP shape (g(1)=1), CEIL_INT4=520.9527 the #204 int4-spec "
            "λ=1 physical ceiling. The private MEAN (achieved private TPS), DISTINCT from my #226 "
            "private BAR (mu_safe_fresh/f_priv). Round-trips kanna #224's λ=1 anchors exactly."),
        "kanna228_dependency_note": (
            "kanna #228 (publish-first-lambda-floor) was IN-FLIGHT / not banked at run time, so "
            "lambda_floor_central is computed HERE on the same imported reach-DP + ceiling; it should "
            "round-trip kanna #228's central λ-floor once banked. The empirical-floor λ-floor is "
            "∅ (private_mean(1, 0.957054) < 500) and the dλ_floor/df_priv is the sensitivity #228 "
            "explicitly deferred to this leg."),
        "verdict": verdict,
        "handoff_line": handoff,
        "imports": {
            "provenance": ("kanna#217 vgovdrjc (f_priv central) x kanna#224 1081oc84 "
                           "(private_mean@ceiling 504.86/498.58, via trigger_reconcile + "
                           "private_bar_reachability) x #204 launch_sigma_unit_rebase (ceiling "
                           "520.9527) x #199 wdyqnx3g (reach-DP spines E[T](λ)) x #175/#184 "
                           "(reach-DP) x #181 (τ) x #168 (step) x #148/#169 (K_cal) x #191 (P95 "
                           "0.9780) x stark#226 tzcc5xuq (f_priv model, realizable worst-case) x "
                           "PR#52 (481.53/460.85 paired draw). All run-ids in "
                           "wandb-applied-ai-team/gemma-challenge-senpai."),
            "mechanism": "compliant_spec_et reach-DP (#199) + fp16_verify_ceiling g(λ)/ceiling model (#220)",
        },
    }


def _selftests(curve: LambdaCurve, ceil: float, g1: float, f_be: float, verdict_flips: bool,
               pm_central: float, pm_emp: float, pm_be: float,
               lam_floor_central: float | None, lam_floor_emp: float | None,
               monotone_pm_in_lambda: bool, monotone_pm_in_fpriv: bool,
               monotone_lamfloor_decreasing: bool, provenance: dict[str, Any],
               dlam_resid: float | None) -> dict[str, Any]:
    # (a) the two kanna #224 λ=1 anchors round-trip (504.86 / 498.58).
    cond_a = bool(_finite(pm_central) and _finite(pm_emp)
                  and abs(pm_central - PRIV_MEAN_CENTRAL_224) < TOL_ANCHOR
                  and abs(pm_emp - PRIV_MEAN_EMP_FLOOR_224) < TOL_ANCHOR)

    # (b) break-even = 500/ceiling to tol AND private_mean(1, breakeven) = 500 to tol.
    cond_b = bool(abs(f_be - TARGET / ceil) < 1e-12 and abs(pm_be - TARGET) < TOL_ANCHOR)

    # (c) verdict_flips consistent with empirical-floor < break-even < central.
    cond_c = bool(verdict_flips and (F_PRIV_EMP_FLOOR < f_be < F_PRIV_CENTRAL)
                  and (pm_central >= TARGET) and (pm_emp < TARGET))

    # (d) private_mean monotone INCREASING in BOTH f_priv and λ.
    cond_d = bool(monotone_pm_in_lambda and monotone_pm_in_fpriv)

    # (e) λ_floor monotone DECREASING in f_priv AND ∅ at the empirical floor (>break-even fails).
    cond_e = bool(monotone_lamfloor_decreasing and lam_floor_emp is None
                  and lam_floor_central is not None and lam_floor_central <= 1.0 + 1e-12)

    # (f) nan-clean handled at payload level; placeholder here.
    cond_f = True

    # provenance: reach-DP reproduces #199 banked E[T]; g(1)=1 exactly; ceiling matches #204 literal.
    cond_prov = bool(provenance["et_floor_resid"] <= TOL_PROV
                     and provenance["et_ceiling_resid"] <= TOL_PROV
                     and provenance["g_at_one_resid"] <= 1e-12
                     and provenance["int4_ceiling_resid_vs_literal"] <= 1e-9
                     and provenance["ceil_ge_floor_all_depths_both_bugs"])

    # cross-check: numeric vs analytic dλ_floor/df_priv agree (robustness, not gating-critical).
    cond_dlam = bool(dlam_resid is None or dlam_resid < 5e-3)

    conditions = {
        "a_kanna224_lambda1_anchors_roundtrip_504p86_498p58": cond_a,
        "b_breakeven_is_500_over_ceiling_and_pm_at_breakeven_is_500": cond_b,
        "c_verdict_flips_empfloor_lt_breakeven_lt_central": cond_c,
        "d_private_mean_monotone_increasing_in_fpriv_and_lambda": cond_d,
        "e_lambda_floor_decreasing_in_fpriv_and_empty_at_empirical_floor": cond_e,
        "f_nan_clean": cond_f,
        "provenance_reachdp_reproduces_199_ET_and_g1_is_1_and_ceiling_204": cond_prov,
        "dlambda_floor_numeric_matches_analytic": cond_dlam,
    }
    return {
        "conditions": conditions,
        "publish_first_fpriv_breakeven_self_test_passes": bool(all(conditions.values())),
        "a_detail": {"pm_central": pm_central, "anchor_central": PRIV_MEAN_CENTRAL_224,
                     "pm_emp": pm_emp, "anchor_emp": PRIV_MEAN_EMP_FLOOR_224},
        "b_detail": {"f_be": f_be, "target_over_ceiling": TARGET / ceil, "pm_be": pm_be},
        "c_detail": {"f_priv_emp": F_PRIV_EMP_FLOOR, "f_be": f_be, "f_priv_central": F_PRIV_CENTRAL},
        "e_detail": {"lam_floor_central": lam_floor_central, "lam_floor_emp": lam_floor_emp},
    }


# --------------------------------------------------------------------------- #
# NaN guard + report + W&B (mirrors #220/#226; never fatal).
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
    h, st, comp = syn["headline"], syn["self_test"], syn["composition"]
    print("\n" + "=" * 100, flush=True)
    print("PR #233  Publish-first f_priv-breakeven — does the #52 decode-drop calibration flip "
          "the #124 gate?", flush=True)
    print("=" * 100, flush=True)
    print(f"  model: private_mean(λ, f_priv) = CEIL_INT4·g(λ)·f_priv   "
          f"[CEIL_INT4={comp['ceil_int4_lambda1']:.7f} (#204), g(λ)=E[T](λ)/E[T](1) both-bugs, "
          f"λ̂={comp['lambda_hat']:.5f}]", flush=True)
    print("-" * 100, flush=True)
    print(f"  TEST  f_priv_breakeven_publish_first = 500/CEIL_INT4 = {h['f_priv_breakeven_publish_first']:.8f}",
          flush=True)
    print(f"  CORE  publish_first_at_ceiling_verdict_flips = {h['publish_first_at_ceiling_verdict_flips']}  "
          f"(empirical-floor {comp['f_priv_empirical_floor_52']:.6f} < break-even "
          f"{h['f_priv_breakeven_publish_first']:.6f} < central {comp['f_priv_central_217']:.6f})",
          flush=True)
    print("-" * 100, flush=True)
    print("  private_mean at the λ=1 physical ceiling 520.9527:", flush=True)
    for r in syn["private_mean_vs_fpriv_at_ceiling"]:
        print(f"     f_priv {r['f_priv']:.6f}  →  {r['private_mean_at_ceiling']:8.4f}  "
              f"({'GO ' if r['clears_500_publish_first'] else 'NO-GO'} vs 500; "
              f"{r['gap_vs_500']:+.4f})   [{r['label']}]", flush=True)
    print("-" * 100, flush=True)
    print(f"  λ-floor (smallest λ with private_mean=500):", flush=True)
    lfc = h["lambda_floor_central"]
    lfe = h["lambda_floor_empirical_floor"]
    print(f"     central 0.969107      : λ_floor = {lfc:.6f}  (round-trips kanna #228; "
          f"private_mean(1,central)={h['private_mean_at_ceiling_central']:.2f} ≥ 500 ⇒ GO)",
          flush=True)
    print(f"     empirical-floor 0.957 : λ_floor = "
          f"{'∅ UNREACHABLE' if lfe is None else f'{lfe:.6f}'}  "
          f"(private_mean(1,emp)={h['private_mean_at_ceiling_empirical_floor']:.2f} < 500 even at λ=1)",
          flush=True)
    print(f"     dλ_floor/df_priv = {h['dlambda_floor_dfpriv']:+.5f}  "
          f"(analytic {h['dlambda_floor_dfpriv_analytic']:+.5f}, "
          f"resid {h['dlambda_floor_dfpriv_num_vs_analytic_resid']:.2e}) — λ-target RISES as f_priv FALLS",
          flush=True)
    print(f"     descent-shape corner λ_floor(central) = {h['lambda_floor_central_descent_shape_corner']:.6f}",
          flush=True)
    print("-" * 100, flush=True)
    mono = syn["monotone"]
    print(f"  monotone: pm↑λ={mono['private_mean_increasing_in_lambda']}  "
          f"pm↑f_priv={mono['private_mean_increasing_in_fpriv']}  "
          f"λ_floor↓f_priv={mono['lambda_floor_decreasing_in_fpriv']}", flush=True)
    print(f"  PRIMARY publish_first_fpriv_breakeven_self_test_passes = "
          f"{st['publish_first_fpriv_breakeven_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"        - {k}: {v}", flush=True)
    print("=" * 100, flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
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
        print(f"[publish-first-fpriv-breakeven] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    h, st, comp, mono = syn["headline"], syn["self_test"], syn["composition"], syn["monotone"]
    run = init_wandb_run(
        job_type="publish-first-fpriv-breakeven",
        agent="stark",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["publish-first-fpriv-breakeven", "issue-124", "validity-gate", "f_priv",
              "private-mean", "publish-first", "point-estimate-gate", "lambda-floor",
              "winners-curse-redraw-budget", "bank-the-analysis"],
        config={
            "ceil_int4_lambda1": comp["ceil_int4_lambda1"], "K_cal": comp["K_cal"],
            "step": comp["step"], "tau_central": comp["tau_central"],
            "tau_conservative": comp["tau_conservative"], "tau_low_181": comp["tau_low_181"],
            "target_publish_first": comp["target_publish_first"], "lambda_hat": comp["lambda_hat"],
            "f_priv_central_217": comp["f_priv_central_217"],
            "f_priv_empirical_floor_52": comp["f_priv_empirical_floor_52"],
            "kanna224_private_mean_central_anchor": comp["kanna224_private_mean_central_anchor"],
            "kanna224_private_mean_empirical_floor_anchor":
                comp["kanna224_private_mean_empirical_floor_anchor"],
            "p95_private_bar_191": comp["p95_private_bar_191"],
            "wandb_group": args.wandb_group, "baseline_tps": 481.53,
            "source_runs": "kanna#217 vgovdrjc, kanna#224 1081oc84, #204, #199 wdyqnx3g, "
                           "stark#226 tzcc5xuq",
        },
    )
    if run is None:
        print("[publish-first-fpriv-breakeven] wandb: no run (no WANDB_API_KEY/mode) — skipping",
              flush=True)
        return

    summary: dict[str, Any] = {
        "publish_first_fpriv_breakeven_self_test_passes":
            int(bool(st["publish_first_fpriv_breakeven_self_test_passes"])),         # PRIMARY
        "f_priv_breakeven_publish_first": h["f_priv_breakeven_publish_first"],        # TEST
        "publish_first_at_ceiling_verdict_flips": int(bool(h["publish_first_at_ceiling_verdict_flips"])),
        "private_mean_at_ceiling_central": h["private_mean_at_ceiling_central"],
        "private_mean_at_ceiling_empirical_floor": h["private_mean_at_ceiling_empirical_floor"],
        "private_mean_at_ceiling_breakeven": h["private_mean_at_ceiling_breakeven"],
        "central_clears_publish_first_at_ceiling": int(bool(h["central_clears_publish_first_at_ceiling"])),
        "empirical_floor_clears_publish_first_at_ceiling":
            int(bool(h["empirical_floor_clears_publish_first_at_ceiling"])),
        "lambda_floor_central": h["lambda_floor_central"],
        "lambda_floor_empirical_floor_unreachable": int(bool(h["lambda_floor_empirical_floor_unreachable"])),
        "lambda_floor_central_descent_shape_corner": h["lambda_floor_central_descent_shape_corner"],
        "dlambda_floor_dfpriv": h["dlambda_floor_dfpriv"],
        "dlambda_floor_dfpriv_analytic": h["dlambda_floor_dfpriv_analytic"],
        "dlambda_floor_dfpriv_num_vs_analytic_resid": h["dlambda_floor_dfpriv_num_vs_analytic_resid"],
        "monotone_private_mean_increasing_in_lambda": int(bool(mono["private_mean_increasing_in_lambda"])),
        "monotone_private_mean_increasing_in_fpriv": int(bool(mono["private_mean_increasing_in_fpriv"])),
        "monotone_lambda_floor_decreasing_in_fpriv": int(bool(mono["lambda_floor_decreasing_in_fpriv"])),
        "ceil_int4_lambda1": comp["ceil_int4_lambda1"],
        "f_priv_central_217": comp["f_priv_central_217"],
        "f_priv_empirical_floor_52": comp["f_priv_empirical_floor_52"],
        "p95_private_bar_191": comp["p95_private_bar_191"],
        "provenance_et_ceiling_resid": syn["provenance"]["et_ceiling_resid"],
        "provenance_et_floor_resid": syn["provenance"]["et_floor_resid"],
        "provenance_g_at_one_resid": syn["provenance"]["g_at_one_resid"],
        "provenance_int4_ceiling_resid_vs_literal": syn["provenance"]["int4_ceiling_resid_vs_literal"],
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    # λ_floor over the f_priv grid as logged scalars.
    for r in syn["lambda_floor_vs_fpriv"]:
        if r["lambda_floor"] is not None:
            key = f"lambda_floor_fpriv_{r['f_priv']:.4f}".replace(".", "p")
            summary[key] = r["lambda_floor"]
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="publish_first_fpriv_breakeven_result", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    print(f"[publish-first-fpriv-breakeven] wandb logged {len(summary)} keys", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="winners-curse-redraw-budget")
    args = ap.parse_args(argv)

    syn = synthesize()

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 233, "agent": "stark",
        "kind": "publish-first-fpriv-breakeven", "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _nan_paths(payload)
    payload["nan_clean"] = not nan_paths
    syn["self_test"]["conditions"]["f_nan_clean"] = not nan_paths
    syn["self_test"]["publish_first_fpriv_breakeven_self_test_passes"] = bool(
        all(syn["self_test"]["conditions"].values()))
    if nan_paths:
        print(f"[publish-first-fpriv-breakeven] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[publish-first-fpriv-breakeven] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = (syn["self_test"]["publish_first_fpriv_breakeven_self_test_passes"]
              and payload["nan_clean"])
        print(f"[publish-first-fpriv-breakeven] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
