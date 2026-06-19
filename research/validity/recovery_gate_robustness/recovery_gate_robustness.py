#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Recovery gate-clearing robustness (PR #709) — is the measured 0.438 g32 AIME
ceiling a statistically ROBUST clear of the 0.420 gate, or a small-n knife-edge?

denken #706 (`c5obav63`, RECOVERY_SPEED_CONSTRAINED) closed the SPEED axis of the
int4-body recovery program: selectively upgrading ubel #700's top-N impact-energy
modules g128->g32 is speed-viable (sub-noise tax) but cannot deliver +10. That
Pareto's entire QUALITY conditional rested on a linear impact-energy->AIME proxy
anchored at the bf16 base 0.46 (`aime_base_ceiling`). But the MEASURED full-g32
ceiling is 0.438 (ubel #679 `1z5vq2ej`, `aime_g32_mean`=0.4375), NOT 0.46 — and
0.438 clears the 0.420 gate by only 0.018, THINNER than AIME's small-n Wilson
half-width. This card re-anchors the proxy on 0.438 and power-analyses whether the
gate-clearing is resolvable at ubel #702's planned n (5-seed x 60 ~= 300 trials).

PURE-CPU statistical model + pre-registration. analysis_only, official_tps=0,
no_hf_job=1, fires=0. NO HF Job / vLLM / submission / kernel build / served-file
change. Locked `int4_g128_lmhead`@126.378 untouched. NOT a fire, NOT an approval
trigger (peak RSS < 200 MB, runtime < 10 s).

------------------------------------------------------------------------------
THE OBJECTS
------------------------------------------------------------------------------
(1) Re-anchored proxy.  AIME(f) = floor + (ceiling - floor)*shape(f), where
    floor = 0.3467 (lawine #693 `6brpvz9x` #31-gate int4-body full-g128; == #706
    `aime_floor`; PR rounds 0.347), ceiling = 0.438 (ubel #679 measured full-g32,
    REPLACING #706's 0.46 bf16 base), f = cumulative impact-energy fraction over
    ubel #700's localized modules.  The gate-clearing cum-energy threshold
    f* = (0.420 - floor)/(ceiling - floor) and the module-count min-N that
    realizes it via ubel #700's energy curve.  Reported as a SHIFT vs #706's
    0.46-anchored f*=0.647 / min-N=14.

(2) Power analysis (CORE).  AIME is scored as a per-problem Bernoulli at the gate
    eval (lawine #693 / #31 basis, e.g. 24/60 Wilson in the fleet log).  For a
    true recovery rate p (full-g32 0.438; the re-anchored selective point) compute
    the Wilson + Clopper-Pearson CIs as a function of n in {30,60,120,240,480}.
    The gate convention is "Wilson CI-lo clears the bar" (two-sided 95%, z=1.96).
    PRIMARY  min_n_for_robust_gate_clear = the smallest n such that a true-0.438
    draw yields Wilson-lo > 0.420 at 95% power.  Also the 80% n and the at-the-
    point-estimate n (the best case: observe exactly 0.438).

(3) Pre-registration.  Point + band for ubel #702's three arms {full-g128 0.347,
    full-g32 0.438, selective-g32-on-48 (predicted)} BEFORE ubel reports, with the
    falsification rule and the proxy-error tolerance before the SPEED verdict flips.

(4) Proxy-shape sensitivity.  Linear vs concave (diminishing-returns) vs convex
    (threshold) shape(f) calibrated to the SAME two endpoints; how min-N modules
    and the clearing-n move across shapes.

PRIMARY metric  recovery_gate_robustness_self_test_passes  (the self-test gate)
KEY     metric  min_n_for_robust_gate_clear   (trials for true-0.438 Wilson-lo>0.420 @95%)
TEST    metric  reanchored_min_n_modules      (module min-N under 0.438 vs #706's 14)

Run:
    python -m research.validity.recovery_gate_robustness.recovery_gate_robustness \
        --self-test --wandb-name denken/recovery-gate-robustness \
        --wandb-group recovery-gate-robustness-denken
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

# =========================================================================== #
# VERIFIED ANCHORS (W&B-confirmed; cross-reads authorized #666).
# =========================================================================== #
# --- the AIME ladder on the #31 gate basis --------------------------------- #
AIME_FLOOR = 0.3467           # lawine #693 6brpvz9x full-g128 int4 body (== #706 aime_floor)
AIME_FLOOR_PR = 0.347         # PR-rounded floor (headline proxy)
AIME_G128_UBEL = 0.350        # ubel #679 1z5vq2ej aime_g128_mean (cross-check)
CEILING_G32 = 0.438           # ubel #679 aime_g32_mean=0.4375 -> the RE-ANCHOR target
CEILING_G64 = 0.4458          # ubel #679 aime_g64_mean=0.445833 (NON-MONOTONE: g64>g32)
CEILING_BF16 = 0.46           # #706 c5obav63 aime_base_ceiling (the OLD anchor)
GATE = 0.420                  # AIME gate = 0.90 x base ~0.467 (#515)
BASE_AIME = 0.467             # vanilla base AIME (gate denominator)

# --- ubel #679 measured g32/g64/g128 CIs (t-based, 4 sessions) ------------- #
UBEL679 = {
    "g32": {"mean": 0.4375, "lo95": 0.397725, "hi95": 0.477275, "n_sessions": 4, "std": 0.025},
    "g64": {"mean": 0.445833, "lo95": 0.334642, "hi95": 0.557024, "n_sessions": 4, "std": 0.069887},
    "g128": {"mean": 0.350000, "lo95": 0.258144, "hi95": 0.441856, "n_sessions": 4, "std": 0.057735},
}

# --- ubel #700 vjhzcvmu impact-energy distribution over the localized set --- #
# (a) summary-scalar topN cum-energy curve — the basis #706 used (n16->0.6719). #
ENERGY_TOPN_SCALAR = {1: 0.348452, 8: 0.586702, 16: 0.671918, 32: 0.770906}
# (b) pareto-table cum f_impact_energy (global 343-module normalization, reaches #
#     rank-48); frac=1.0 == full-g32 on ALL modules == the physical 0.438 anchor. #
ENERGY_PARETO_RANK = {1: 0.348452, 8: 0.529569, 16: 0.583424, 24: 0.663553,
                      32: 0.701038, 40: 0.726741, 48: 0.799552}
SUBSET_N_MODULES = 48         # ubel #700 clearing_subset_n_modules (40 ple + 3q + 3k + 2v)
PER_MODULE_TAX_TPS = 0.0137   # #706 g32_per_module_tax_avg_tps
G32_FULL_BODY_TAX_TPS = 4.699 # #706 g32_full_body_tax_tps
RECOVERY_TPS_BAND_HALF = 2.484  # #706 (128.844-123.8765)/2 ~ the +/- noise band on recovery TPS
LOCKED_ANCHOR_TPS = 126.378   # locked int4_g128_lmhead official TPS
PLUS10_BAR = 136.378          # the +10 bar (#706 plus10_bar)

# --- #706 reproduction targets --------------------------------------------- #
D706_FSTAR = 0.64695          # #706 need_cum_energy_to_clear (0.46-anchor)
D706_MIN_N_MODULES = 14       # #706 min_n_clearing_aime_gate
D706_MIN_N_INTERP = 13.656    # #706 min_n_clearing_aime_gate_interp
D706_SELECTIVE48 = 0.43404    # #706 aime_proj_n48 (energy basis, clamped at n32)

# --- ubel #702 planned design --------------------------------------------- #
UBEL702_SEEDS = 5
UBEL702_PROBLEMS = 60
UBEL702_N = UBEL702_SEEDS * UBEL702_PROBLEMS   # 300 effective trials

# --- statistical constants ------------------------------------------------- #
Z95 = 1.959963984540054       # two-sided 95% (the fleet gate convention; 1-sided 2.5%)
N_GRID = (30, 60, 120, 240, 480)
POWER_TARGETS = (0.80, 0.95)
N_SEARCH_CAP = 60000
UNRESOLVABLE_N = 1.0e9       # finite sentinel for "never clears at any feasible n" (cf. #187)
STABLE_RUN = 150             # consecutive-n stability window for the staircased power
VIF_GRID = (1.0, 1.5, 2.0)   # seed/session-correlation effective-N deflation (honest scope)


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# =========================================================================== #
# NUMERIC CORE — dependency-free (no scipy): regularized incomplete beta,
# beta quantile (bisection), Wilson CI, Clopper-Pearson, exact binomial tail.
# =========================================================================== #
def _betacf(a: float, b: float, x: float) -> float:
    """Lentz continued fraction for the incomplete beta (Numerical Recipes)."""
    MAXIT, EPS, FPMIN = 300, 3.0e-16, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a,b) in [0,1]."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def beta_quantile(p: float, a: float, b: float) -> float:
    """Inverse of I_x(a,b): smallest x with betai(a,b,x) >= p (bisection; monotone)."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if betai(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def binom_tail_ge(k: int, n: int, p: float) -> float:
    """Exact P(X >= k | X ~ Binom(n,p)) via the beta identity = I_p(k, n-k+1)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return betai(float(k), float(n - k + 1), p)


def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float, float]:
    """Two-sided Wilson score interval for k/n. Returns (center, lo, hi)."""
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n))) / denom
    return center, center - half, center + half


def wilson_lo_from_phat(phat: float, n: int, z: float = Z95) -> float:
    """Wilson lower bound for a continuous observed proportion phat at sample size n."""
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n))) / denom
    return center - half


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Clopper-Pearson) two-sided CI for k/n."""
    lo = 0.0 if k == 0 else beta_quantile(alpha / 2.0, k, n - k + 1)
    hi = 1.0 if k == n else beta_quantile(1.0 - alpha / 2.0, k + 1, n - k)
    return lo, hi


def smallest_k_clearing(n: int, bar: float, z: float = Z95) -> int:
    """Smallest integer k in [0,n] whose Wilson-lo(k,n) > bar (Wilson-lo monotone in k)."""
    lo, hi = 0, n + 1
    while lo < hi:
        mid = (lo + hi) // 2
        _, wlo, _ = wilson_ci(mid, n, z)
        if wlo > bar:
            hi = mid
        else:
            lo = mid + 1
    return lo  # == n+1 if unreachable


def power_clear(n: int, p_true: float, bar: float, z: float = Z95) -> float:
    """P(Wilson-lo(X,n) > bar | X ~ Binom(n, p_true)) — the gate-clear power at n."""
    k_min = smallest_k_clearing(n, bar, z)
    if k_min > n:
        return 0.0
    return binom_tail_ge(k_min, n, p_true)


# =========================================================================== #
# (1) RE-ANCHORED PROXY
# =========================================================================== #
def shape_fn(f: float, kind: str) -> float:
    """shape(f) on [0,1] -> [0,1], pinned shape(0)=0, shape(1)=1.
       linear: f ; concave (diminishing returns, early gains): sqrt(f) ;
       convex (threshold, late gains): f**2."""
    f = max(0.0, min(1.0, f))
    if kind == "linear":
        return f
    if kind == "concave":
        return math.sqrt(f)
    if kind == "convex":
        return f * f
    raise ValueError(kind)


def proxy_aime(f: float, floor: float, ceiling: float, kind: str = "linear") -> float:
    return floor + (ceiling - floor) * shape_fn(f, kind)


def fstar_clearing(floor: float, ceiling: float, bar: float, kind: str = "linear") -> float:
    """Cumulative-energy fraction f* at which proxy_aime(f*) == bar (clears the gate)."""
    if ceiling <= floor:
        return float("inf")
    s_star = (bar - floor) / (ceiling - floor)   # required shape(f*) value
    if s_star <= 0.0:
        return 0.0
    if s_star >= 1.0:
        return float("inf")                       # gate not reachable even at f=1
    if kind == "linear":
        return s_star
    if kind == "concave":
        return s_star * s_star                     # invert sqrt
    if kind == "convex":
        return math.sqrt(s_star)                   # invert square
    raise ValueError(kind)


def _interp_cum_energy(curve: dict[int, float], n_modules: float) -> float:
    """Cumulative energy at a (possibly fractional) module count via piecewise-linear
       interpolation of the logged cum-energy curve (extrapolation past the last logged
       point is linear and flagged by the caller; it OVER-states a concave tail)."""
    ranks = sorted(curve)
    if n_modules <= ranks[0]:
        return curve[ranks[0]] * (n_modules / ranks[0]) if ranks[0] > 0 else 0.0
    for i in range(len(ranks) - 1):
        r0, r1 = ranks[i], ranks[i + 1]
        if r0 <= n_modules <= r1:
            t = (n_modules - r0) / (r1 - r0)
            return curve[r0] + t * (curve[r1] - curve[r0])
    # extrapolate past the last point using the last segment slope
    r0, r1 = ranks[-2], ranks[-1]
    slope = (curve[r1] - curve[r0]) / (r1 - r0)
    return curve[r1] + slope * (n_modules - r1)


def modules_for_cum_energy(curve: dict[int, float], target: float) -> tuple[float, bool]:
    """Smallest module count whose cum-energy >= target, by inverting the curve.
       Returns (n_modules, reachable_within_logged_subset)."""
    ranks = sorted(curve)
    max_logged = curve[ranks[-1]]
    if target <= 0.0:
        return 0.0, True
    # bisection on a continuous module axis up to the subset size (then extrapolate)
    lo, hi = 0.0, float(ranks[-1])
    if _interp_cum_energy(curve, hi) < target:
        # not reachable within the logged subset; extrapolate the last slope
        r0, r1 = ranks[-2], ranks[-1]
        slope = (curve[r1] - curve[r0]) / (r1 - r0)
        if slope <= 0:
            return float("inf"), False
        n_ext = ranks[-1] + (target - max_logged) / slope
        return n_ext, False
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _interp_cum_energy(curve, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), True


def reanchor_block() -> dict[str, Any]:
    """Deliverable (1): re-anchor the proxy and report the min-N shift."""
    # f* on each anchor (linear) for the apples-to-apples shift.
    fstar_046 = fstar_clearing(AIME_FLOOR, CEILING_BF16, GATE, "linear")     # ~0.6470 (#706)
    fstar_0438 = fstar_clearing(AIME_FLOOR, CEILING_G32, GATE, "linear")     # ~0.8029
    fstar_g64 = fstar_clearing(AIME_FLOOR, CEILING_G64, GATE, "linear")      # ~0.7382 (non-mono xcheck)

    # min-N modules on the SAME basis #706 used (summary-scalar topN curve).
    m706_scalar, m706_reach = modules_for_cum_energy(ENERGY_TOPN_SCALAR, fstar_046)
    m0438_scalar, m0438_reach = modules_for_cum_energy(ENERGY_TOPN_SCALAR, fstar_0438)
    # min-N modules on the physical pareto/global basis (frac=1.0 == full-g32 == 0.438).
    m706_pareto, _ = modules_for_cum_energy(ENERGY_PARETO_RANK, fstar_046)
    m0438_pareto, m0438_pareto_reach = modules_for_cum_energy(ENERGY_PARETO_RANK, fstar_0438)
    mg64_pareto, mg64_pareto_reach = modules_for_cum_energy(ENERGY_PARETO_RANK, fstar_g64)

    # selective-on-48 predicted AIME (re-anchored) on each basis.
    sel48_scalar = proxy_aime(_interp_cum_energy(ENERGY_TOPN_SCALAR, SUBSET_N_MODULES),
                              AIME_FLOOR, CEILING_G32, "linear")
    sel48_pareto = proxy_aime(ENERGY_PARETO_RANK[48], AIME_FLOOR, CEILING_G32, "linear")
    # under the OLD 0.46-anchor the same 48-subset (param basis) predicted ~0.437.
    sel48_046_pareto = proxy_aime(ENERGY_PARETO_RANK[48], AIME_FLOOR, CEILING_BF16, "linear")

    return {
        "proxy_formula": "AIME(f) = floor + (ceiling - floor) * shape(f)",
        "floor": AIME_FLOOR, "floor_pr_rounded": AIME_FLOOR_PR,
        "ceiling_reanchored": CEILING_G32, "ceiling_old_706": CEILING_BF16,
        "ceiling_g64_nonmono_xcheck": CEILING_G64,
        "realizable_lift_0438": CEILING_G32 - AIME_FLOOR,         # 0.0913 (PR's 0.091)
        "realizable_lift_046_old": CEILING_BF16 - AIME_FLOOR,     # 0.1133
        "fstar_clearing_046_anchor": fstar_046,
        "fstar_clearing_0438_anchor": fstar_0438,
        "fstar_clearing_g64_anchor": fstar_0438 if False else fstar_g64,
        "fstar_shift_046_to_0438": fstar_0438 - fstar_046,
        # min-N modules (TEST metric on #706's summary-scalar basis).
        "min_n_modules_046_scalar_basis": m706_scalar,
        "min_n_modules_0438_scalar_basis": m0438_scalar,
        "min_n_modules_0438_scalar_reachable_in_subset": m0438_reach,
        # physical pareto/global basis (frac=1.0 == full body).
        "min_n_modules_046_pareto_basis": m706_pareto,
        "min_n_modules_0438_pareto_basis": m0438_pareto,
        "min_n_modules_0438_pareto_reachable_in_subset": m0438_pareto_reach,
        "min_n_modules_g64_pareto_basis": mg64_pareto,
        "min_n_modules_g64_reachable_in_subset": mg64_pareto_reach,
        "d706_min_n_modules": D706_MIN_N_MODULES,
        # selective-48 predicted AIME under re-anchoring.
        "selective48_pred_0438_scalar": sel48_scalar,
        "selective48_pred_0438_pareto": sel48_pareto,
        "selective48_pred_046_pareto": sel48_046_pareto,
        "selective48_clears_gate_0438": bool(min(sel48_scalar, sel48_pareto) > GATE),
        "note": (
            "Re-anchoring the asymptote 0.46->0.438 raises the cum-energy threshold "
            f"from f*={fstar_046:.4f} (#706) to f*={fstar_0438:.4f}. On the physical "
            "global/pareto basis (frac=1.0 == full-g32 == 0.438) the 48-module subset "
            f"captures only {ENERGY_PARETO_RANK[48]:.4f} < f*, so the selective fix "
            f"predicts AIME~{sel48_pareto:.4f} (ON the gate) vs the 0.46-anchored "
            f"{sel48_046_pareto:.4f} that cleared by ~0.017. The cheap-selective-fix "
            "margin is ERASED: min-N jumps from 14 modules to the entire localized subset."
        ),
    }


# =========================================================================== #
# (2) POWER ANALYSIS (CORE)
# =========================================================================== #
def power_table(p_points: dict[str, float], bar: float = GATE) -> list[dict[str, Any]]:
    """For each n in the grid and each labelled true rate, the Wilson/CP CI at the
       point estimate (observe exactly p) AND the gate-clear POWER."""
    rows = []
    for n in N_GRID:
        row: dict[str, Any] = {"n": n}
        for label, p in p_points.items():
            k = int(round(p * n))
            cen, wlo, whi = wilson_ci(k, n)
            cplo, cphi = clopper_pearson(k, n)
            row[label] = {
                "k_at_point": k, "phat": k / n,
                "wilson_lo": wlo, "wilson_hi": whi, "wilson_halfwidth": (whi - wlo) / 2.0,
                "cp_lo": cplo, "cp_hi": cphi,
                "point_clears_gate": bool(wlo > bar),
                "straddles_gate": bool(wlo <= bar <= whi),
                "power_clear_gate": power_clear(n, p, bar),
            }
        rows.append(row)
    return rows


def min_n_for_power(p_true: float, bar: float, target: float, z: float = Z95) -> dict[str, Any]:
    """Smallest n s.t. power_clear(n, p_true) >= target AND stays >= target over the
       next STABLE_RUN n (robust to the discrete staircase). Also the normal-approx
       cross-check. Returns the n and provenance."""
    if p_true <= bar:
        return {"min_n": UNRESOLVABLE_N, "reachable": False, "reason": "p_true<=bar (never clears)",
                "normal_approx": UNRESOLVABLE_N}
    # normal-approximation seed (Wilson-lo>bar ~ one-sided test at alpha=2.5%).
    z_alpha = z                                  # 1.96 (two-sided lo == one-sided 2.5%)
    z_beta = {0.80: 0.8416212335729143, 0.95: 1.6448536269514722,
              0.90: 1.2815515594465649}.get(round(target, 2), 1.6448536269514722)
    p0, p1 = bar, p_true
    num = z_alpha * math.sqrt(p0 * (1 - p0)) + z_beta * math.sqrt(p1 * (1 - p1))
    n_norm = (num / (p1 - p0)) ** 2
    # exact scan with a stability window, starting below the normal-approx seed.
    start = max(1, int(0.3 * n_norm))
    consec, run_start = 0, None
    n = start
    found = None
    while n <= N_SEARCH_CAP:
        if power_clear(n, p_true, bar, z) >= target:
            if consec == 0:
                run_start = n
            consec += 1
            if consec >= STABLE_RUN:
                found = run_start
                break
        else:
            consec, run_start = 0, None
        n += 1
    return {"min_n": float(found) if found is not None else UNRESOLVABLE_N,
            "reachable": found is not None,
            "normal_approx": n_norm,
            "power_target": target}


def min_n_point_estimate(p_true: float, bar: float, z: float = Z95) -> float:
    """Best case: smallest n at which OBSERVING exactly p_true gives Wilson-lo>bar.
       (The point estimate equals the truth — no sampling luck needed.)"""
    if p_true <= bar:
        return UNRESOLVABLE_N
    lo, hi = 1, N_SEARCH_CAP
    # monotone-ish: wilson_lo_from_phat(p,n) increases with n. bisect then verify.
    while lo < hi:
        mid = (lo + hi) // 2
        if wilson_lo_from_phat(p_true, mid, z) > bar:
            hi = mid
        else:
            lo = mid + 1
    return float(lo) if wilson_lo_from_phat(p_true, lo, z) > bar else UNRESOLVABLE_N


def power_block(reanchor: dict[str, Any]) -> dict[str, Any]:
    """Deliverable (2): the power table + the min-N-trials primary metric."""
    sel48 = reanchor["selective48_pred_0438_pareto"]   # the re-anchored selective point (~0.420)
    p_points = {
        "full_g32_0438": CEILING_G32,            # the measured ceiling (best case)
        "selective48_reanchored": sel48,         # the re-anchored selective prediction
    }
    table = power_table(p_points, GATE)

    # PRIMARY: min_n for true-0.438 Wilson-lo>0.420 at 95% (and 80%).
    mn95 = min_n_for_power(CEILING_G32, GATE, 0.95)
    mn80 = min_n_for_power(CEILING_G32, GATE, 0.80)
    mn_point = min_n_point_estimate(CEILING_G32, GATE)

    # the selective-48 point sits ~ON the gate -> effectively unresolvable.
    sel_clears = sel48 > GATE
    mn95_sel = min_n_for_power(sel48, GATE, 0.95) if sel_clears else {
        "min_n": UNRESOLVABLE_N, "reachable": False, "reason": "selective48<=gate (unresolvable)",
        "normal_approx": UNRESOLVABLE_N}

    # VIF (seed/session correlation) sensitivity on the planned-n Wilson half-width.
    vif_rows = []
    for vif in VIF_GRID:
        n_eff = UBEL702_N / vif
        k = int(round(CEILING_G32 * n_eff))
        _, wlo, whi = wilson_ci(k, int(round(n_eff)))
        vif_rows.append({"vif": vif, "n_eff": n_eff, "implied_seed_rho": (vif - 1) / (vif + 1),
                         "wilson_lo_at_0438": wlo, "wilson_halfwidth": (whi - wlo) / 2.0,
                         "clears_gate": bool(wlo > GATE)})

    # what does ubel #702's planned n=300 actually buy (at the 0.438 best case)?
    k300 = int(round(CEILING_G32 * UBEL702_N))
    cen300, wlo300, whi300 = wilson_ci(k300, UBEL702_N)
    planned = {
        "n": UBEL702_N, "phat": CEILING_G32, "wilson_lo": wlo300, "wilson_hi": whi300,
        "wilson_halfwidth": (whi300 - wlo300) / 2.0, "clears_gate_pointest": bool(wlo300 > GATE),
        "power_at_0438": power_clear(UBEL702_N, CEILING_G32, GATE),
        "shortfall_vs_min_n95": mn95["min_n"] / UBEL702_N if math.isfinite(mn95["min_n"]) else float("inf"),
    }

    return {
        "p_points": p_points,
        "power_table": table,
        "min_n_for_robust_gate_clear": mn95["min_n"],           # PRIMARY KEY metric (95%)
        "min_n_for_robust_gate_clear_80": mn80["min_n"],
        "min_n_point_estimate_best_case": mn_point,
        "min_n_95_detail": mn95, "min_n_80_detail": mn80,
        "selective48_point": sel48, "selective48_clears_gate": bool(sel_clears),
        "min_n_95_selective": mn95_sel,
        "vif_sensitivity": vif_rows,
        "ubel702_planned": planned,
        "ubel679_g32_empirical_straddle": bool(UBEL679["g32"]["lo95"] <= GATE),
        "note": (
            f"At the BEST case (true rate = the measured ceiling 0.438) the gate margin is "
            f"{CEILING_G32 - GATE:.3f}. Even observing exactly 0.438, the Wilson-lo clears 0.420 "
            f"only at n>={mn_point:.0f}; for a random 0.438 draw to clear at 95% power needs "
            f"n>={mn95['min_n']:.0f} ({mn95['min_n']/UBEL702_N:.0f}x ubel #702's planned {UBEL702_N}). "
            f"ubel #679's OWN measured g32 CI [{UBEL679['g32']['lo95']:.4f},{UBEL679['g32']['hi95']:.4f}] "
            f"already straddles 0.420 at 4 sessions — empirical confirmation of the knife-edge."
        ),
    }


# =========================================================================== #
# (3) PRE-REGISTRATION of ubel #702's three arms
# =========================================================================== #
def preregister_block(reanchor: dict[str, Any], power: dict[str, Any]) -> dict[str, Any]:
    """Deliverable (3): point + band per arm BEFORE ubel #702 reports, + falsification."""
    n = UBEL702_N
    sel48 = reanchor["selective48_pred_0438_pareto"]
    sel_lo_scalar = reanchor["selective48_pred_0438_scalar"]

    def band(point: float) -> dict[str, Any]:
        # measurement Wilson half-width at the planned n, centred on the predicted point.
        k = int(round(point * n))
        cen, wlo, whi = wilson_ci(k, n)
        return {"point": point, "n": n, "wilson_lo": wlo, "wilson_hi": whi,
                "wilson_halfwidth": (whi - wlo) / 2.0}

    arms = {
        "full_g128": {**band(AIME_FLOOR), "anchor": True, "basis": "lawine #693 0.347 fixed"},
        "full_g32": {**band(CEILING_G32), "anchor": True, "basis": "ubel #679 0.438 fixed"},
        "selective_g32_on_48": {
            **band(sel48), "anchor": False,
            "proxy_spread_lo": min(sel_lo_scalar, sel48),
            "proxy_spread_hi": CEILING_G32,    # if localization is perfect -> == full-g32
            "basis": "re-anchored linear proxy, global frac=0.7996",
        },
    }
    # the predicted SEPARATION ubel can/can't resolve: full-g32 vs full-g128 = 0.091.
    sep_g32_g128 = CEILING_G32 - AIME_FLOOR
    sep_sel_floor = sel48 - AIME_FLOOR
    # can ubel #702 at n=300 even separate the arms? (Wilson half-widths overlap?)
    hw = arms["full_g32"]["wilson_halfwidth"]
    falsify = {
        "rule": ("Reject the linear impact-energy proxy iff ubel #702's measured "
                 "selective-g32-on-48 lands OUTSIDE [point +/- (proxy_spread (+) Wilson "
                 "half-width)]. At n=300 the Wilson half-width is +/-%.3f, so the "
                 "falsification band is ~[%.3f, %.3f] — wide enough that almost any "
                 "plausible outcome is INSIDE it: the proxy is UNFALSIFIABLE at the "
                 "planned n." % (hw, max(0.0, sel48 - hw - 0.01), min(1.0, CEILING_G32 + hw))),
        "selective_falsification_halfwidth": hw,
        "n_to_falsify_selective": power["min_n_point_estimate_best_case"],
        "arms_separable_at_300": bool(sep_g32_g128 > 2 * hw),
    }
    tolerance = _speed_tolerance()
    return {"arms": arms, "predicted_separation_g32_minus_g128": sep_g32_g128,
            "predicted_separation_selective_minus_floor": sep_sel_floor,
            "falsification": falsify, "speed_verdict_tolerance": tolerance}


def _speed_tolerance() -> dict[str, Any]:
    """How wrong can the proxy be before the #706 SPEED verdict (sub-noise tax) flips?
       The selective tax is modules_needed x per-module-tax; it stays sub-noise as long
       as modules_needed <= noise_band / per_module_tax."""
    m_subnoise = RECOVERY_TPS_BAND_HALF / PER_MODULE_TAX_TPS
    tax_full_subset = SUBSET_N_MODULES * PER_MODULE_TAX_TPS
    return {
        "per_module_tax_tps": PER_MODULE_TAX_TPS,
        "noise_band_half_tps": RECOVERY_TPS_BAND_HALF,
        "modules_subnoise_ceiling": m_subnoise,
        "tax_full_48_subset_tps": tax_full_subset,
        "full_48_subset_subnoise": bool(tax_full_subset < RECOVERY_TPS_BAND_HALF),
        "g32_full_body_tax_tps": G32_FULL_BODY_TAX_TPS,
        "full_body_subnoise": bool(G32_FULL_BODY_TAX_TPS < RECOVERY_TPS_BAND_HALF),
        "note": (
            f"Even if the proxy is wrong and the selective fix needs the WHOLE 48-module "
            f"subset, the tax is {tax_full_subset:.2f} TPS < the +/-{RECOVERY_TPS_BAND_HALF:.2f} "
            f"noise band -> sub-noise, so the #706 SPEED verdict (selective-g32 speed-viable) "
            f"is ROBUST to the re-anchoring (which raises module count 14->~full-subset). The "
            f"SPEED verdict flips only if you must requant the FULL body (>{m_subnoise:.0f} "
            f"modules, tax {G32_FULL_BODY_TAX_TPS:.2f} TPS, NOT sub-noise). Re-anchoring breaks "
            f"the QUALITY margin, NOT the speed conclusion."
        ),
    }


# =========================================================================== #
# (4) PROXY-SHAPE SENSITIVITY
# =========================================================================== #
def shape_block() -> dict[str, Any]:
    """Deliverable (4): min-N modules + selective-48 prediction across shapes,
       all calibrated to the SAME endpoints (floor 0.3467, ceiling 0.438)."""
    rows = []
    for kind in ("linear", "concave", "convex"):
        fstar = fstar_clearing(AIME_FLOOR, CEILING_G32, GATE, kind)
        if math.isfinite(fstar):
            m_pareto, reach = modules_for_cum_energy(ENERGY_PARETO_RANK, fstar)
        else:
            m_pareto, reach = float("inf"), False
        sel48 = proxy_aime(ENERGY_PARETO_RANK[48], AIME_FLOOR, CEILING_G32, kind)
        rows.append({
            "shape": kind, "fstar_clearing": fstar,
            "min_n_modules_pareto": m_pareto, "reachable_in_48_subset": reach,
            "selective48_pred": sel48, "selective48_clears_gate": bool(sel48 > GATE),
        })
    # the POWER primary (min-N trials) is shape-INVARIANT for the 0.438 ceiling arm
    # (the endpoints are fixed); shape only moves the module-count and the selective point.
    finite_modules = [r["min_n_modules_pareto"] for r in rows if math.isfinite(r["min_n_modules_pareto"])]
    spread = (max(finite_modules) - min(finite_modules)) if finite_modules else float("inf")
    any_unreachable = any(not r["reachable_in_48_subset"] for r in rows)
    return {
        "shapes": rows,
        "module_min_n_spread_across_shapes": spread,
        "any_shape_unreachable_in_subset": any_unreachable,
        "power_min_n_is_shape_invariant": True,
        "note": (
            "Module-count gate-clearing-n is shape-FRAGILE (concave clears with fewer "
            "modules; convex needs more / unreachable in the 48-subset). But the PRIMARY "
            "power-n (trials for Wilson-lo>0.420 at the 0.438 ceiling) is shape-INVARIANT: "
            "it depends only on the fixed endpoints, so the ~thousands-of-trials requirement "
            "is trustworthy regardless of the impact-energy->AIME curvature."
        ),
    }


# =========================================================================== #
# SELF-TEST (PRIMARY) — validates the power/CI core against KNOWN values.
# =========================================================================== #
def _self_test(reanchor, power, shapes) -> dict[str, Any]:
    checks: dict[str, bool] = {}

    # (a) Wilson CI reproduces the fleet-logged 24/60 AIME read [0.2857, 0.5263].
    _, wlo, whi = wilson_ci(24, 60)
    checks["a_wilson_24_60_matches_log"] = (abs(wlo - 0.2857) < 5e-4 and abs(whi - 0.5263) < 5e-4)

    # (b) Wilson CI for the textbook 50/100 case == [0.4038, 0.5962].
    _, wlo2, whi2 = wilson_ci(50, 100)
    checks["b_wilson_50_100_textbook"] = (abs(wlo2 - 0.40383) < 1e-3 and abs(whi2 - 0.59617) < 1e-3)

    # (c) Clopper-Pearson endpoints: 0/10 upper == 0.30850, 10/10 lower == 0.69150.
    cp0_lo, cp0_hi = clopper_pearson(0, 10)
    cp10_lo, cp10_hi = clopper_pearson(10, 10)
    checks["c_cp_endpoints"] = (abs(cp0_hi - 0.308024) < 1e-3 and abs(cp10_lo - 0.691976) < 1e-3
                                and cp0_lo == 0.0 and cp10_hi == 1.0)

    # (d) Exact binomial tail identity: P(X>=1 | 10, 0.5) == 1 - 0.5^10 == 0.99902.
    t = binom_tail_ge(1, 10, 0.5)
    checks["d_binom_tail_exact"] = abs(t - (1.0 - 0.5 ** 10)) < 1e-9

    # (e) betai is a proper CDF: I_p(k,n-k+1) increasing in p; tail in [0,1].
    mono = all(binom_tail_ge(5, 20, p) <= binom_tail_ge(5, 20, p + 0.05) + 1e-12
               for p in [x / 20 for x in range(0, 19)])
    checks["e_betai_monotone_in_p"] = bool(mono and 0.0 <= binom_tail_ge(5, 20, 0.3) <= 1.0)

    # (f) min_n_point_estimate consistency: at that n Wilson-lo>gate, just below it does not.
    mnp = power["min_n_point_estimate_best_case"]
    if math.isfinite(mnp):
        ok_at = wilson_lo_from_phat(CEILING_G32, int(mnp)) > GATE
        ok_below = wilson_lo_from_phat(CEILING_G32, int(mnp) - 1) <= GATE
        checks["f_min_n_point_boundary"] = bool(ok_at and ok_below)
    else:
        checks["f_min_n_point_boundary"] = False

    # (g) power-n cross-checks the normal approximation within 25% (both are ~thousands).
    mn95 = power["min_n_95_detail"]
    if math.isfinite(mn95["min_n"]) and math.isfinite(mn95["normal_approx"]):
        rel = abs(mn95["min_n"] - mn95["normal_approx"]) / mn95["normal_approx"]
        checks["g_power_n_vs_normal_approx"] = bool(rel < 0.25)
    else:
        checks["g_power_n_vs_normal_approx"] = False

    # (h) at the computed 95% min_n the exact power really is >= 0.95 (and < 0.95 well below).
    if math.isfinite(mn95["min_n"]):
        p_at = power_clear(int(mn95["min_n"]), CEILING_G32, GATE)
        p_lo = power_clear(int(0.5 * mn95["min_n"]), CEILING_G32, GATE)
        checks["h_power_at_min_n_ge_target"] = bool(p_at >= 0.95 and p_lo < 0.95)
    else:
        checks["h_power_at_min_n_ge_target"] = False

    # (i) #706 reproduction: 0.46-anchor f* and min-N modules recover #706's 0.647 / 14.
    f046 = reanchor["fstar_clearing_046_anchor"]
    checks["i_reproduce_706_fstar_and_minN"] = (
        abs(f046 - D706_FSTAR) < 1e-3
        and abs(reanchor["min_n_modules_046_scalar_basis"] - D706_MIN_N_INTERP) < 0.5)

    # (j) re-anchoring strictly RAISES f* and the module min-N (the cost of 0.438<0.46).
    checks["j_reanchor_raises_minN"] = bool(
        reanchor["fstar_clearing_0438_anchor"] > reanchor["fstar_clearing_046_anchor"]
        and reanchor["min_n_modules_0438_scalar_basis"] > reanchor["min_n_modules_046_scalar_basis"])

    # (k) shape sensitivity present and the power-n shape-invariance flag holds.
    checks["k_shape_block_consistent"] = bool(
        len(shapes["shapes"]) == 3 and shapes["power_min_n_is_shape_invariant"])

    passes = all(checks.values())
    return {"recovery_gate_robustness_self_test_passes": passes, "conditions": checks,
            "evidence": {
                "wilson_24_60": [wlo, whi], "wilson_50_100": [wlo2, whi2],
                "cp_0_10_hi": cp0_hi, "cp_10_10_lo": cp10_lo, "binom_tail_1_10_p5": t,
                "min_n_95": mn95["min_n"], "normal_approx_95": mn95["normal_approx"],
                "fstar_046": f046}}


# =========================================================================== #
# VERDICT
# =========================================================================== #
def decide_verdict(power: dict, reanchor: dict, shapes: dict) -> tuple[str, str]:
    mn95 = power["min_n_for_robust_gate_clear"]
    planned = power["ubel702_planned"]["n"]
    knife = math.isfinite(mn95) and mn95 > planned
    shape_fragile = shapes["any_shape_unreachable_in_subset"]
    sel_on_gate = not power["selective48_clears_gate"]
    if knife or sel_on_gate:
        v = "RECOVERY_GATE_KNIFE_EDGE"
        msg = (
            f"RECOVERY_GATE_KNIFE_EDGE. Re-anchored on the MEASURED 0.438 ceiling, the gate "
            f"margin is only {CEILING_G32 - GATE:.3f}. For a true-0.438 recovery's Wilson-lo to "
            f"clear 0.420 at 95% power needs n>={mn95:.0f} trials ({mn95/planned:.0f}x ubel #702's "
            f"planned {planned}); the 80% n is {power['min_n_for_robust_gate_clear_80']:.0f} and even "
            f"OBSERVING exactly 0.438 needs n>={power['min_n_point_estimate_best_case']:.0f}. The "
            f"re-anchored selective-g32-on-48 prediction sits at ~{power['selective48_point']:.4f} "
            f"(ON the 0.420 gate) -> unresolvable at ANY finite n. ubel #679's own g32 CI "
            f"[{UBEL679['g32']['lo95']:.4f},{UBEL679['g32']['hi95']:.4f}] already straddles 0.420 at "
            f"4 sessions. ubel #702 MUST scale seeds (min_n_95~={mn95:.0f}) before spending GPU, "
            f"else its verdict CI straddles the gate (inconclusive). The QUALITY margin is a "
            f"knife-edge; the SPEED verdict (#706 sub-noise tax) stays robust. analysis_only, 0 TPS, "
            f"no fire."
        )
        if shape_fragile:
            msg += (" SECONDARY: module-count gate-clearing-n is also shape-fragile (convex shape "
                    "-> unreachable within the 48-subset), but the power-n is shape-invariant.")
        return v, msg
    if shape_fragile:
        return "RECOVERY_GATE_SHAPE_FRAGILE", (
            "RECOVERY_GATE_SHAPE_FRAGILE. The gate-clearing module-count is robust under the "
            "linear proxy but explodes / becomes unreachable under a plausible convex shape; the "
            "clearing claim is contingent on linearity, which only ubel #702's three-arm "
            "measurement can confirm.")
    return "RECOVERY_GATE_ROBUST", (
        "RECOVERY_GATE_ROBUST. Re-anchored on 0.438 the selective recovery's Wilson-lo clears "
        "0.420 at ubel #702's planned n under linear and non-linear shapes.")


# =========================================================================== #
# SYNTHESIS
# =========================================================================== #
def synthesize() -> dict[str, Any]:
    reanchor = reanchor_block()
    power = power_block(reanchor)
    prereg = preregister_block(reanchor, power)
    shapes = shape_block()
    st = _self_test(reanchor, power, shapes)
    verdict, verdict_msg = decide_verdict(power, reanchor, shapes)
    return {
        "self_test": st,
        "reanchor": reanchor,
        "power": power,
        "preregister": prereg,
        "shape_sensitivity": shapes,
        "verdict": verdict,
        "verdict_message": verdict_msg,
        "primary_metric": {"min_n_for_robust_gate_clear": power["min_n_for_robust_gate_clear"]},
        "test_metric": {"reanchored_min_n_modules": reanchor["min_n_modules_0438_scalar_basis"]},
    }


# =========================================================================== #
# NaN-clean walk.
# =========================================================================== #
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


def _sanitize(node: Any) -> Any:
    """Replace non-finite floats with string sentinels so the JSON file is strict-valid
       (inf appears legitimately for unreachable-n / convex-shape f*)."""
    if isinstance(node, dict):
        return {k: _sanitize(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_sanitize(v) for v in node]
    if isinstance(node, float) and not math.isfinite(node):
        return "inf" if node > 0 else ("-inf" if node < 0 else "nan")
    return node


# =========================================================================== #
# Console report.
# =========================================================================== #
def _print_report(syn: dict) -> None:
    ra, pw, pr, sh = syn["reanchor"], syn["power"], syn["preregister"], syn["shape_sensitivity"]
    st = syn["self_test"]
    print("\n" + "=" * 98, flush=True)
    print("RECOVERY GATE-CLEARING ROBUSTNESS (PR #709) — is 0.438 a robust clear or a knife-edge?",
          flush=True)
    print("=" * 98, flush=True)
    print("  (1) RE-ANCHOR proxy 0.46 -> measured 0.438:", flush=True)
    print(f"      proxy AIME(f)= {ra['floor']:.4f} + {ra['realizable_lift_0438']:.4f}*shape(f) "
          f"(was {ra['floor']:.4f}+{ra['realizable_lift_046_old']:.4f}*f at 0.46)", flush=True)
    print(f"      f* clearing 0.420:  0.46-anchor={ra['fstar_clearing_046_anchor']:.4f} (#706) "
          f"-> 0.438-anchor={ra['fstar_clearing_0438_anchor']:.4f}  (shift +{ra['fstar_shift_046_to_0438']:.4f})",
          flush=True)
    print(f"      min-N modules:  #706 0.46 = {ra['d706_min_n_modules']}  ->  re-anchored 0.438 = "
          f"{ra['min_n_modules_0438_scalar_basis']:.1f} (scalar) / "
          f"{ra['min_n_modules_0438_pareto_basis']:.1f} (pareto)  "
          f"[reach in 48-subset: {ra['min_n_modules_0438_pareto_reachable_in_subset']}]", flush=True)
    print(f"      selective-48 predicted AIME:  0.46-anchor={ra['selective48_pred_046_pareto']:.4f} "
          f"-> 0.438-anchor={ra['selective48_pred_0438_pareto']:.4f}  "
          f"(clears 0.420: {ra['selective48_clears_gate_0438']})", flush=True)
    print("-" * 98, flush=True)
    print("  (2) POWER — Wilson-lo>0.420 at n (best case p=0.438):", flush=True)
    for row in pw["power_table"]:
        g = row["full_g32_0438"]
        print(f"      n={row['n']:>4}: half=+/-{g['wilson_halfwidth']:.4f}  "
              f"Wilson=[{g['wilson_lo']:.4f},{g['wilson_hi']:.4f}]  "
              f"point-clears={int(g['point_clears_gate'])}  straddles={int(g['straddles_gate'])}  "
              f"power={g['power_clear_gate']:.3f}", flush=True)
    print(f"      PRIMARY min_n_for_robust_gate_clear (95%) = {pw['min_n_for_robust_gate_clear']:.0f}  "
          f"(80% = {pw['min_n_for_robust_gate_clear_80']:.0f}; point-est best case = "
          f"{pw['min_n_point_estimate_best_case']:.0f})", flush=True)
    print(f"      ubel #702 planned n={pw['ubel702_planned']['n']}: Wilson-lo={pw['ubel702_planned']['wilson_lo']:.4f} "
          f"(<0.420), power={pw['ubel702_planned']['power_at_0438']:.3f}, shortfall "
          f"{pw['ubel702_planned']['shortfall_vs_min_n95']:.0f}x", flush=True)
    print(f"      ubel #679 g32 empirical CI straddles 0.420: {pw['ubel679_g32_empirical_straddle']}",
          flush=True)
    print("-" * 98, flush=True)
    print("  (3) PRE-REGISTER ubel #702 three arms (n=300 Wilson bands):", flush=True)
    for name, a in pr["arms"].items():
        print(f"      {name:<22}: point={a['point']:.4f}  band=[{a['wilson_lo']:.4f},{a['wilson_hi']:.4f}]  "
              f"anchor={a['anchor']}", flush=True)
    print(f"      falsification: arms_separable_at_300={pr['falsification']['arms_separable_at_300']}  "
          f"n_to_falsify_selective>={pr['falsification']['n_to_falsify_selective']:.0f}", flush=True)
    print(f"      SPEED tolerance: full-48 tax={pr['speed_verdict_tolerance']['tax_full_48_subset_tps']:.2f} TPS "
          f"sub-noise={pr['speed_verdict_tolerance']['full_48_subset_subnoise']} -> SPEED verdict robust",
          flush=True)
    print("-" * 98, flush=True)
    print("  (4) SHAPE sensitivity (min-N modules, pareto basis):", flush=True)
    for r in sh["shapes"]:
        mm = f"{r['min_n_modules_pareto']:.1f}" if math.isfinite(r["min_n_modules_pareto"]) else "inf"
        print(f"      {r['shape']:<8}: f*={r['fstar_clearing']:.4f}  min-N={mm}  "
              f"reach48={r['reachable_in_48_subset']}  sel48={r['selective48_pred']:.4f} "
              f"clears={int(r['selective48_clears_gate'])}", flush=True)
    print(f"      power-n shape-invariant: {sh['power_min_n_is_shape_invariant']}", flush=True)
    print("-" * 98, flush=True)
    print(f"  PRIMARY self_test_passes = {st['recovery_gate_robustness_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"        - {k}: {v}", flush=True)
    print("-" * 98, flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print(f"  {syn['verdict_message']}", flush=True)
    print("=" * 98 + "\n", flush=True)


# =========================================================================== #
# W&B logging (mirrors lambda_built_ci.py; never fatal).
# =========================================================================== #
def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[recovery-gate-robustness] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    run = init_wandb_run(
        job_type="recovery-gate-robustness",
        agent="denken",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["recovery-gate-robustness", "analysis-only", "power-analysis", "aime-gate",
              "reanchor", "preregistration", "no-hf-job"],
        config={
            "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
            "aime_floor": AIME_FLOOR, "ceiling_reanchored": CEILING_G32,
            "ceiling_old_706": CEILING_BF16, "ceiling_g64": CEILING_G64, "gate": GATE,
            "ubel702_planned_n": UBEL702_N, "z95": Z95,
            "imports": "ubel#679 0.438 ceiling + ubel#700 energy dist + denken#706 0.46 proxy",
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[recovery-gate-robustness] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    st, ra, pw, pr, sh = (syn["self_test"], syn["reanchor"], syn["power"],
                          syn["preregister"], syn["shape_sensitivity"])
    # power-table flattened per n (best-case 0.438 arm).
    ptab = {}
    for row in pw["power_table"]:
        g = row["full_g32_0438"]
        ptab[f"power_clear_n{row['n']}"] = g["power_clear_gate"]
        ptab[f"wilson_lo_n{row['n']}"] = g["wilson_lo"]
        ptab[f"wilson_halfwidth_n{row['n']}"] = g["wilson_halfwidth"]
        ptab[f"straddles_gate_n{row['n']}"] = int(g["straddles_gate"])
    arms = {f"prereg_{name}_point": a["point"] for name, a in pr["arms"].items()}
    arms_lo = {f"prereg_{name}_wilson_lo": a["wilson_lo"] for name, a in pr["arms"].items()}
    arms_hi = {f"prereg_{name}_wilson_hi": a["wilson_hi"] for name, a in pr["arms"].items()}
    shp = {f"shape_{r['shape']}_min_n_modules": r["min_n_modules_pareto"]
           for r in sh["shapes"] if math.isfinite(r["min_n_modules_pareto"])}
    shp_sel = {f"shape_{r['shape']}_selective48": r["selective48_pred"] for r in sh["shapes"]}

    summary: dict[str, Any] = {
        # constraint scalars (EXPLICIT, required by the PR).
        "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        # primary / test metrics.
        "recovery_gate_robustness_self_test_passes": int(bool(
            st["recovery_gate_robustness_self_test_passes"])),
        "min_n_for_robust_gate_clear": pw["min_n_for_robust_gate_clear"],
        "min_n_for_robust_gate_clear_80": pw["min_n_for_robust_gate_clear_80"],
        "min_n_point_estimate_best_case": pw["min_n_point_estimate_best_case"],
        "reanchored_min_n_modules": ra["min_n_modules_0438_scalar_basis"],
        "reanchored_min_n_modules_pareto": ra["min_n_modules_0438_pareto_basis"],
        "d706_min_n_modules": ra["d706_min_n_modules"],
        # re-anchor.
        "fstar_046_anchor": ra["fstar_clearing_046_anchor"],
        "fstar_0438_anchor": ra["fstar_clearing_0438_anchor"],
        "fstar_shift": ra["fstar_shift_046_to_0438"],
        "realizable_lift_0438": ra["realizable_lift_0438"],
        "selective48_pred_0438": ra["selective48_pred_0438_pareto"],
        "selective48_pred_046": ra["selective48_pred_046_pareto"],
        "selective48_clears_gate_0438": int(ra["selective48_clears_gate_0438"]),
        # power context.
        "gate": GATE, "ceiling_g32": CEILING_G32, "gate_margin": CEILING_G32 - GATE,
        "ubel702_planned_n": pw["ubel702_planned"]["n"],
        "ubel702_planned_wilson_lo": pw["ubel702_planned"]["wilson_lo"],
        "ubel702_planned_power_at_0438": pw["ubel702_planned"]["power_at_0438"],
        "ubel702_shortfall_factor": pw["ubel702_planned"]["shortfall_vs_min_n95"],
        "ubel679_g32_empirical_straddle": int(pw["ubel679_g32_empirical_straddle"]),
        "normal_approx_min_n_95": pw["min_n_95_detail"]["normal_approx"],
        # shape.
        "module_min_n_spread_across_shapes": sh["module_min_n_spread_across_shapes"],
        "any_shape_unreachable_in_subset": int(sh["any_shape_unreachable_in_subset"]),
        "power_min_n_is_shape_invariant": int(sh["power_min_n_is_shape_invariant"]),
        # speed tolerance.
        "speed_tax_full_48_subset_tps": pr["speed_verdict_tolerance"]["tax_full_48_subset_tps"],
        "speed_full_48_subnoise": int(pr["speed_verdict_tolerance"]["full_48_subset_subnoise"]),
        # verdict.
        "verdict_knife_edge": int(syn["verdict"] == "RECOVERY_GATE_KNIFE_EDGE"),
        "verdict_robust": int(syn["verdict"] == "RECOVERY_GATE_ROBUST"),
        "verdict_shape_fragile": int(syn["verdict"] == "RECOVERY_GATE_SHAPE_FRAGILE"),
        "peak_mem_mib": payload["peak_mem_mib"],
        **ptab, **arms, **arms_lo, **arms_hi, **shp, **shp_sel,
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v))}
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="recovery_gate_robustness_result",
                      artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[recovery-gate-robustness] wandb logged verdict={syn['verdict']} "
          f"min_n95={pw['min_n_for_robust_gate_clear']:.0f}", flush=True)


# =========================================================================== #
# CLI.
# =========================================================================== #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="recovery-gate-robustness-denken")
    args = ap.parse_args(argv)

    syn = synthesize()
    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 709, "agent": "denken",
        "kind": "recovery-gate-robustness",
        "constraints": {"analysis_only": True, "official_tps": 0, "no_hf_job": 1, "fires": 0},
        "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _assert_nan_clean({k: v for k, v in payload.items()})
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[recovery-gate-robustness] NOTE non-finite (expected for unreachable n): {nan_paths}",
              flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "recovery_gate_robustness_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(_sanitize(payload), fh, indent=2, sort_keys=True)
    print(f"[recovery-gate-robustness] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = syn["self_test"]["recovery_gate_robustness_self_test_passes"]
        print(f"[recovery-gate-robustness] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
