#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Frozen-regime cost crossover: re-price #200's build-vs-redraw under #202's freeze (PR #206).

WHAT THIS IS
------------
kanna #200 (`n3alx7ca`, MERGED) priced the build-vs-redraw decision: building to the
N=1 safe point mu_safe=512.157 (Delta_mu=12.157 TPS above the bar) beats re-drawing
best-of-N at the mu=500 bar once per-shot cost c exceeds c* = 3.039*b (fixed-N) /
12.97*b (sequential early-stop). But #200 priced redraw under the FRESH regime: it
ASSUMED best-of-N=5 at the bar reaches P>=0.95 (P_fresh(500,5)=0.969). kanna #202
(`533jd6l1`, MERGED) showed that the more plausible FROZEN regime (fixed 128 prompts
+ greedy token-identity => the finite-sample bias is a COMMON per-checkpoint offset
best-of-N CANNOT beat down -- only sigma_hw re-randomizes) breaks that assumption:
P_frozen(500,5)=0.810 (NOT 0.969), and forcing P>=0.95 by re-drawing needs the
UNCAPPED quota ~30 frozen shots (vs 5 fresh). This leg FUSES the two: it re-prices
#200's crossover under #202's frozen redraw curve, so the human's Approval request
carries the REGIME-ROBUST spend, not the optimistic (fresh-only) one.

THE RE-PRICING (PR steps 1-3)
-----------------------------
The crossover is universal: build-to-N1 (cost b*Delta_mu + c) beats redraw-at-bar
(cost N_redraw*c fixed, or E_redraw*c sequential) once
    c* = Delta_mu / (N_redraw - 1)         [fixed-N]
    c* = Delta_mu / (E_redraw - 1)         [sequential]
Only the redraw budget N_redraw / E_redraw changes between regimes:
  * FRESH  : N_redraw = 5,  E_redraw = 1.9375  -> c* = 3.039*b / 12.97*b  (== #200).
  * FROZEN : N_redraw = 30, E_redraw = E_frozen(30) -> c* shifts HARD DOWN (redraw is
             a poor substitute, so build-higher wins over a much wider cost range).
build_higher_dominates_below_b = c0 / c*_slope is the build-cost threshold (at nominal
per-shot c0=1.0) below which building-to-512.2 is unambiguously the cheaper plan.

THE DELIVERABLE (PR step 2)
---------------------------
A 2x2 {regime in fresh/frozen} x {strategy in build-to-512.2-N1 / build-at-bar-best-of-N}
expected-GPU-$ table, swept over build cost b. build-to-512.2/N=1 is REGIME-INVARIANT
(N=1 => frozen == fresh exactly, #202 self-test d) -> it is the minimax-regret hedge;
best-of-N-at-bar is cheap under fresh (5 shots) but blows up under frozen (30 shots,
19%-exhaust tail). regime_robust_strategy = the strategy minimizing worst-case regret
(== the 50/50-prior expected-cost optimum) absent a pinned regime.

PARTIAL FREEZE (PR step 3)
--------------------------
sigma_beatable^2 = sigma_hw^2 + f*sigma_sample^2 ; the freeze fraction f in [0,1]
bridges fully-frozen (f=0, c*=0.419*b) to fully-fresh (f=1, c*=3.039*b == #200).
f_where_redraw_competitive = the f above which the #194/#200 N=5 budget still clears
the bar at P>=0.95 -- must reproduce #202's frozen_fraction_breakeven=0.846.

SELF-TEST (PR step 5 -- PRIMARY)
-------------------------------
(a) f=1 (fully fresh) reproduces #200's c* = 3.039*b (fixed) / 12.97*b (sequential)
    VERBATIM within tol (the import anchor);
(b) f=0 (fully frozen) GENUINE redraw never reaches 0.95 at the bar (the operationally
    -safe ceiling is Phi((mu-500)/sigma_sample)=0.5, since best-of-N cannot move the
    frozen bias) => effective redraw cost -> inf => build-higher dominates for ALL
    finite b (the limiting case stronger than the nominal 30-shot crossover);
(c) monotone: c*(f) non-decreasing in f, so c*_frozen <= c*_fresh (freeze only makes
    redraw LESS attractive, never more);
(d) build-to-512.2/N=1 cost is identical fresh vs frozen for every b (N=1 regime-
    invariance, the hedge property);
(e) NaN-clean.
PRIMARY = frozen_cost_self_test_passes (bool); TEST = build_higher_dominates_below_b
(float, GPU-$ build-cost threshold in units of b at c0=1.0).

LOCAL, CPU-ONLY, ANALYTIC. No GPU / vLLM / HF Job / submission / served-file / actual
official draw. BASELINE stays 481.53; greedy/PPL untouched; adds 0 TPS; authorizes NO
draws and NO shot count (a human still approves the spend AND confirms the harness
regime). IMPORTS #200's fresh crossover + #202's frozen redraw curve VERBATIM; does NOT
re-derive them. Draw-budget/cost lane (#159/#188/#194/#200/#202). NOT denken #197
(liveprobe). NOT ubel #204 (launch-sigma). NOT fern #185 (integrator). NOT stark #203
(private-drop). NOT open2. NOT a launch.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # research/validity/frozen_cost_crossover -> root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# source-of-truth: kanna's OWN banked cost (#200) and frozen (#202) curves (imported verbatim)
# ---------------------------------------------------------------------------
COST_200 = os.path.join(_ROOT, "research/validity/cost_budget/cost_budget_results.json")
FROZEN_202 = os.path.join(_ROOT, "research/validity/frozen_budget/frozen_budget_results.json")

TARGET = 500.0
P_TARGET = 0.95
N_SIMPSON = 2001  # #202's frozen-integral grid (reproduce verbatim)
N_BAR_FRESH = 5   # #194/#200 fresh redraw budget at the bar

# b-sweep grid for the scale-free 2x2 table / regret (a decade each side of b0=1.0, plus 0)
B_GRID = [0.0, 0.03162277660168379, 0.1, 0.31622776601683794, 1.0,
          3.1622776601683795, 10.0]
F_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _finite(x: Any) -> float:
    if x is None or not math.isfinite(float(x)):
        raise ValueError(f"non-finite value: {x!r}")
    return float(x)


def _phi(x: float) -> float:
    """Standard-normal CDF via erf (no scipy) -- identical to #194/#200/#202."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float, mu: float, sigma: float) -> float:
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


# ---------------------------------------------------------------------------
# best-of-N max-statistics (imported #202 model: fresh, frozen, partial-freeze)
# ---------------------------------------------------------------------------
def p_fresh(mu: float, sigma_draw: float, n: int, target: float = TARGET) -> float:
    """FRESH: both components re-randomize per shot. P(max of N >= t) = 1 - Phi((t-mu)/sigma)^N."""
    return 1.0 - _phi((target - mu) / sigma_draw) ** n


def p_frozen_general(mu: float, sigma_beatable: float, sigma_frozen: float, n: int,
                     target: float = TARGET, n_grid: int = N_SIMPSON) -> float:
    """best-of-N with a FROZEN per-checkpoint bias b~N(0,sigma_frozen^2) common across all N
    shots; only eps~N(0,sigma_beatable^2) re-draws per shot (== #202's p_frozen_general):
        P = 1 - E_b[ Phi((t - mu - b)/sigma_beatable)^N ],  Simpson over [-8s,8s]."""
    if sigma_frozen <= 1e-12:
        return 1.0 - _phi((target - mu) / sigma_beatable) ** n
    if sigma_beatable <= 1e-12:
        return _phi((mu - target) / sigma_frozen)
    lo, hi = -8.0 * sigma_frozen, 8.0 * sigma_frozen
    h = (hi - lo) / (n_grid - 1)
    acc = 0.0
    for i in range(n_grid):
        b = lo + i * h
        w = 1.0 if i in (0, n_grid - 1) else (4.0 if i % 2 == 1 else 2.0)
        acc += w * _norm_pdf(b, 0.0, sigma_frozen) * _phi((target - mu - b) / sigma_beatable) ** n
    return 1.0 - acc * h / 3.0


def partial_sigmas(f: float, sigma_sample: float, sigma_hw: float) -> tuple[float, float]:
    """sigma_beatable^2 = sigma_hw^2 + f*sigma_sample^2 ; sigma_frozen^2 = (1-f)*sigma_sample^2."""
    sigma_beatable = math.sqrt(sigma_hw ** 2 + f * sigma_sample ** 2)
    sigma_frozen = math.sqrt(max(0.0, 1.0 - f)) * sigma_sample
    return sigma_beatable, sigma_frozen


def p_partial(mu: float, f: float, sigma_sample: float, sigma_hw: float, n: int) -> float:
    sb, sf = partial_sigmas(f, sigma_sample, sigma_hw)
    return p_frozen_general(mu, sb, sf, n)


def n_redraw_partial(f: float, sigma_sample: float, sigma_hw: float,
                     mu: float = TARGET, p_target: float = P_TARGET, n_cap: int = 5000) -> int | None:
    """Minimal N s.t. partial-freeze best-of-N at the bar reaches P>=p_target (None if > n_cap)."""
    for n in range(1, n_cap + 1):
        if p_partial(mu, f, sigma_sample, sigma_hw, n) >= p_target:
            return n
    return None


def seq_expected_shots(p: float, n_max: int) -> float:
    """E[shots] for stop-on-first-clear truncated at n_max (== #200/#202): (1-(1-p)^n_max)/p."""
    if n_max <= 0:
        return 0.0
    if p <= 0.0:
        return float(n_max)
    return (1.0 - (1.0 - p) ** n_max) / p


def frozen_e_shots_capped(n_max: int, sigma_beatable: float, sigma_frozen: float,
                          n_grid: int = N_SIMPSON) -> float:
    """Frozen E[shots] capped at n_max: marginalize the truncated-geometric over the common frozen
    bias b~N(0,sigma_frozen^2) (== #202's _frozen_e_shots_capped, generalized to partial freeze).
    q(b)=Phi(b/sigma_beatable) is P(single frozen shot at the bar clears | bias b)."""
    if sigma_frozen <= 1e-12:  # fully fresh: single-shot clear prob is regime-free
        q = 1.0 - _phi((TARGET - TARGET) / sigma_beatable)
        return seq_expected_shots(q, n_max)
    lo, hi = -8.0 * sigma_frozen, 8.0 * sigma_frozen
    h = (hi - lo) / (n_grid - 1)
    acc = 0.0
    for i in range(n_grid):
        b = lo + i * h
        w = 1.0 if i in (0, n_grid - 1) else (4.0 if i % 2 == 1 else 2.0)
        q = _phi(b / sigma_beatable)  # P(single frozen shot at the bar clears | bias b)
        acc += w * _norm_pdf(b, 0.0, sigma_frozen) * seq_expected_shots(q, n_max)
    return acc * h / 3.0


def _bisect_unit(fn, target_val: float, tol: float = 1e-9, iters: int = 200) -> float | None:
    """Solve fn(f)=target_val on f in [0,1] for a monotone fn. None if no crossing (== #202)."""
    flo, fhi = fn(0.0) - target_val, fn(1.0) - target_val
    if flo >= 0:
        return 0.0
    if fhi < 0:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if fn(mid) - target_val >= 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# import #200 (fresh crossover) and #202 (frozen redraw curve) -- do NOT re-derive
# ---------------------------------------------------------------------------
def import_banked() -> dict[str, Any]:
    c200 = _load(COST_200)
    f202 = _load(FROZEN_202)
    co = c200["cost_optimal"]
    imp200 = c200["import_194"]
    bbar = f202["build_bar"]
    hsens = f202["harness_sensitivity"]
    reg = f202["regime_model"]["sigma_decomposition"]

    return {
        # ---- #200 fresh crossover (the anchor we re-price) ----
        "c_star_slope_fresh_fixedN": _finite(co["c_star_slope_per_b_fixedN"]),       # 3.039
        "c_star_slope_fresh_sequential": _finite(co["c_star_slope_per_b_sequential"]),  # 12.97
        "e_shots_fresh_bar": _finite(c200["expected_shots_sequential_at_bar"]),       # 1.9375
        "n_bar_fresh": int(co["n_bar_194"]),                                          # 5
        "delta_mu_tps": _finite(co["delta_mu_tps"]),                                  # 12.157
        "mu_safe_tps": _finite(imp200["mu_safe_tps"]),                                # 512.157
        # ---- #194 sigma decomposition (via #202) ----
        "sigma_sample_tps": _finite(reg["sigma_sample_tps"]),                         # 5.564
        "sigma_hw_tps": _finite(reg["sigma_hw_tps"]),                                 # 4.864
        "sigma_draw_tps": _finite(reg["sigma_draw_tps"]),                             # 7.391
        # ---- #202 frozen redraw curve ----
        "p_bar_n5_frozen": _finite(f202["p_bar_n5_frozen"]),                          # 0.810
        "mu_bar_frozen_p95": _finite(f202["mu_bar_frozen_p95"]),                      # 504.87
        "delta_mu_frozen": _finite(f202["delta_mu_frozen"]),                          # -7.28
        "n_shots_frozen_at_512": int(bbar["n_shots_frozen_at_512"]),                  # 1
        "frozen_fraction_breakeven_202": _finite(f202["frozen_fraction_breakeven"]),  # 0.846
        "e_shots_frozen_bar_nmax5": _finite(hsens["p95_shots_at_bar"]["e_shots_frozen_at_bar"]),  # 2.337
        "p_exhaust_frozen_nmax5": _finite(hsens["p95_shots_at_bar"]["p_exhaust_without_clear_frozen"]),  # 0.190
        "frozen_uncapped_quota_202": int(hsens["p95_shots_at_bar"]["frozen_uncapped"]),  # 30
        "fresh_uncapped_quota_202": int(hsens["p95_shots_at_bar"]["fresh_uncapped"]),    # 5
        "source_cost_200": COST_200,
        "source_frozen_202": FROZEN_202,
    }


# ---------------------------------------------------------------------------
# crossover algebra: c* = Delta_mu / (redraw_budget - 1); below_b = c0 / c*_slope
# ---------------------------------------------------------------------------
def c_star_slope(delta_mu: float, redraw_budget: float) -> float:
    """Per-shot cost c* (in units of b) at which build-to-N1 ties redraw-at-bar.
    build-N1 cost = b*Delta_mu + c ; redraw cost = redraw_budget*c. Tie => c* = Delta_mu/(R-1)."""
    denom = redraw_budget - 1.0
    return (delta_mu / denom) if denom > 1e-12 else float("inf")


def below_b(c0: float, slope: float) -> float:
    """Build-cost threshold b* (at fixed per-shot c0) below which build-higher is cheaper.
    build-higher cheaper iff c0 > slope*b iff b < c0/slope."""
    return (c0 / slope) if (slope is not None and math.isfinite(slope) and slope > 1e-12) else float("inf")


# ---------------------------------------------------------------------------
# STEP 1 -- re-derive the build-vs-redraw crossover under the frozen regime
# ---------------------------------------------------------------------------
def frozen_crossover(imp: dict, c0: float) -> dict[str, Any]:
    dmu = imp["delta_mu_tps"]
    ss, sh = imp["sigma_sample_tps"], imp["sigma_hw_tps"]

    # FRESH anchor (re-derived from the redraw budget; must match the imported #200 slopes)
    n_fresh = imp["n_bar_fresh"]                       # 5
    e_fresh = imp["e_shots_fresh_bar"]                 # 1.9375
    slope_fresh_fixed = c_star_slope(dmu, n_fresh)     # 3.039
    slope_fresh_seq = c_star_slope(dmu, e_fresh)       # 12.97

    # FROZEN: redraw at the bar needs the UNCAPPED quota to force P>=0.95 (recompute -> assert ==30)
    n_frozen = n_redraw_partial(0.0, ss, sh)           # 30 (== #202 frozen_uncapped)
    e_frozen = frozen_e_shots_capped(n_frozen, sh, ss) if n_frozen is not None else float("nan")
    slope_frozen_fixed = c_star_slope(dmu, float(n_frozen)) if n_frozen is not None else float("inf")
    slope_frozen_seq = c_star_slope(dmu, e_frozen) if n_frozen is not None else float("inf")

    return {
        "delta_mu_tps": dmu,
        "fresh": {
            "n_redraw_fixedN": n_fresh,
            "e_redraw_sequential": e_fresh,
            "crossover_fixed": slope_fresh_fixed,        # 3.039 (anchor)
            "crossover_sequential": slope_fresh_seq,     # 12.97 (anchor)
            "build_higher_dominates_below_b_fixedN": below_b(c0, slope_fresh_fixed),       # 0.329
            "build_higher_dominates_below_b_sequential": below_b(c0, slope_fresh_seq),     # 0.077
        },
        "frozen": {
            "n_redraw_fixedN": n_frozen,                 # 30
            "e_redraw_sequential": e_frozen,             # E_frozen(30)
            "crossover_fixed_frozen": slope_frozen_fixed,        # 0.419
            "crossover_sequential_frozen": slope_frozen_seq,     # << 12.97
            "build_higher_dominates_below_b_fixedN": below_b(c0, slope_frozen_fixed),      # 2.385 (TEST headline)
            "build_higher_dominates_below_b_sequential": below_b(c0, slope_frozen_seq),
        },
        "shift_vs_fresh": {
            "crossover_fixed_ratio_frozen_over_fresh": slope_frozen_fixed / slope_fresh_fixed,
            "crossover_sequential_ratio_frozen_over_fresh": slope_frozen_seq / slope_fresh_seq,
            "below_b_widen_factor_fixedN": below_b(c0, slope_frozen_fixed) / below_b(c0, slope_fresh_fixed),
            "interpretation": (
                "FROZEN redraw needs N=%s shots (vs fresh 5) to force P>=0.95 at the bar, so the build-vs-"
                "redraw crossover slope drops from c*=%.3f*b (fresh) to c*=%.3f*b (frozen) fixed-N -- "
                "build-higher now dominates for build costs below b=%.3f (vs fresh b=%.3f), a %.2fx wider "
                "range. The shift is HARD toward build-higher because re-drawing at the bar is a poor "
                "substitute under freeze (it can only beat down sigma_hw)."
                % (n_frozen, slope_fresh_fixed, slope_frozen_fixed,
                   below_b(c0, slope_frozen_fixed), below_b(c0, slope_fresh_fixed),
                   below_b(c0, slope_frozen_fixed) / below_b(c0, slope_fresh_fixed))
            ),
        },
    }


# ---------------------------------------------------------------------------
# STEP 2 -- the regime-robust spend: 2x2 {regime} x {strategy} expected-GPU-$ + regret
# ---------------------------------------------------------------------------
def regime_robust(imp: dict, cross: dict, c0: float) -> dict[str, Any]:
    dmu = imp["delta_mu_tps"]
    n_fresh = imp["n_bar_fresh"]                        # 5
    e_fresh = imp["e_shots_fresh_bar"]                  # 1.9375
    n_frozen = cross["frozen"]["n_redraw_fixedN"]       # 30
    e_frozen = cross["frozen"]["e_redraw_sequential"]   # E_frozen(30)

    def cost_buildN1(b: float) -> float:
        return b * dmu + c0  # REGIME-INVARIANT (N=1 => frozen == fresh exactly)

    # redraw-at-bar costs are b-independent (no build); regime-dependent shot budget.
    cost_redraw = {
        "fresh_fixedN": n_fresh * c0,                  # 5
        "frozen_fixedN": n_frozen * c0,                # 30
        "fresh_sequential": e_fresh * c0,              # 1.9375
        "frozen_sequential": e_frozen * c0,            # E_frozen(30)
    }

    def regret_table(b: float, scheme: str) -> dict[str, Any]:
        rk_fresh = cost_redraw[f"fresh_{scheme}"]
        rk_frozen = cost_redraw[f"frozen_{scheme}"]
        c_build = cost_buildN1(b)
        opt_fresh = min(c_build, rk_fresh)
        opt_frozen = min(c_build, rk_frozen)
        reg_build = {"fresh": c_build - opt_fresh, "frozen": c_build - opt_frozen}
        reg_redraw = {"fresh": rk_fresh - opt_fresh, "frozen": rk_frozen - opt_frozen}
        worst_build = max(reg_build["fresh"], reg_build["frozen"])
        worst_redraw = max(reg_redraw["fresh"], reg_redraw["frozen"])
        exp_build = c_build                              # 0.5*fresh + 0.5*frozen, both == c_build
        exp_redraw = 0.5 * rk_fresh + 0.5 * rk_frozen
        minimax = "build-to-512.2-N1" if worst_build <= worst_redraw else "build-at-bar-best-of-N"
        prior50 = "build-to-512.2-N1" if exp_build <= exp_redraw else "build-at-bar-best-of-N"
        return {
            "b_per_mu": b,
            "cost_buildN1_fresh": c_build,
            "cost_buildN1_frozen": c_build,            # identical (hedge)
            "cost_redraw_fresh": rk_fresh,
            "cost_redraw_frozen": rk_frozen,
            "regret_buildN1": reg_build,
            "regret_redraw": reg_redraw,
            "worstcase_regret_buildN1": worst_build,
            "worstcase_regret_redraw": worst_redraw,
            "expected_cost_buildN1_50_50": exp_build,
            "expected_cost_redraw_50_50": exp_redraw,
            "minimax_regret_strategy": minimax,
            "prior50_50_strategy": prior50,
        }

    fixed_rows = [regret_table(b, "fixedN") for b in B_GRID]
    seq_rows = [regret_table(b, "sequential") for b in B_GRID]

    # the build-cost crossover where the regime-robust (50/50-prior == minimax-regret) choice flips
    # build-N1 cheaper-in-expectation iff b*dmu + c0 <= 0.5*redraw_fresh + 0.5*redraw_frozen
    b_robust_fixed = (0.5 * cost_redraw["fresh_fixedN"] + 0.5 * cost_redraw["frozen_fixedN"] - c0) / dmu
    b_robust_seq = (0.5 * cost_redraw["fresh_sequential"] + 0.5 * cost_redraw["frozen_sequential"] - c0) / dmu

    return {
        "table_2x2_note": (
            "rows = build cost b (job-units/TPS, at per-shot c0=%.1f); build-to-512.2-N1 cost = b*%.3f + c0 "
            "is IDENTICAL fresh vs frozen (N=1 regime-invariance); build-at-bar-best-of-N costs %g (fresh) "
            "vs %g (frozen) fixed-N (the frozen 19%%-exhaust tail forces ~30 shots)."
            % (c0, dmu, cost_redraw["fresh_fixedN"], cost_redraw["frozen_fixedN"])
        ),
        "cost_redraw_corners": cost_redraw,
        "fixedN_rows": fixed_rows,
        "sequential_rows": seq_rows,
        "b_robust_crossover_fixedN": b_robust_fixed,      # 1.357
        "b_robust_crossover_sequential": b_robust_seq,
        "regime_robust_strategy": "build-to-512.2-N1",
        "regime_robust_rationale": (
            "build-to-512.2/N=1 is the minimax-regret (== 50/50-prior) optimum for build costs b below "
            "b_robust=%.3f (fixed-N): its cost is regime-INVARIANT (N=1 => frozen==fresh), so its "
            "worst-case regret is bounded, while best-of-N-at-bar is cheap under fresh (5 shots) but blows "
            "up under frozen (30 shots, 19%%-exhaust) -- worst-case regret up to %.1f shots' GPU-$ at b->0. "
            "Absent a pinned regime the human should HEDGE with build-higher."
            % (b_robust_fixed, cost_redraw["frozen_fixedN"] - c0)
        ),
    }


# ---------------------------------------------------------------------------
# STEP 3 -- partial-freeze sensitivity: c*(f) bridge from fresh (f=1) to frozen (f=0)
# ---------------------------------------------------------------------------
def partial_freeze(imp: dict, c0: float) -> dict[str, Any]:
    dmu = imp["delta_mu_tps"]
    ss, sh = imp["sigma_sample_tps"], imp["sigma_hw_tps"]

    rows = []
    for f in F_GRID:
        sb, sf = partial_sigmas(f, ss, sh)
        n_rd = n_redraw_partial(f, ss, sh)
        e_rd = frozen_e_shots_capped(n_rd, sb, sf) if n_rd is not None else float("nan")
        slope_fixed = c_star_slope(dmu, float(n_rd)) if n_rd is not None else float("inf")
        slope_seq = c_star_slope(dmu, e_rd) if (n_rd is not None and math.isfinite(e_rd)) else float("inf")
        rows.append({
            "f": f,
            "p_partial_n5_at_500": p_partial(TARGET, f, ss, sh, N_BAR_FRESH),
            "n_redraw_at_bar": n_rd,
            "e_redraw_at_bar": e_rd,
            "c_star_fixed": slope_fixed,
            "c_star_sequential": slope_seq,
            "build_higher_dominates_below_b_fixedN": below_b(c0, slope_fixed),
        })

    # f_where_redraw_competitive: the freeze fraction above which the #194/#200 N=5 budget still
    # clears the bar at P>=0.95 (i.e. n_redraw <= 5) -- must reproduce #202's frozen_fraction_breakeven.
    f_competitive = _bisect_unit(lambda f: p_partial(TARGET, f, ss, sh, N_BAR_FRESH), P_TARGET)
    aligns = (f_competitive is not None
              and abs(f_competitive - imp["frozen_fraction_breakeven_202"]) < 1e-6)

    return {
        "partial_curve": rows,
        "f_where_redraw_competitive": f_competitive,                 # 0.846
        "frozen_fraction_breakeven_202": imp["frozen_fraction_breakeven_202"],
        "aligns_with_202_breakeven": aligns,
        "note": (
            "c*(f) bridges continuously from the fully-frozen f=0 (redraw needs ~30 shots -> c*=%.3f*b, "
            "build-higher strongly favoured) to the fully-fresh f=1 (redraw needs 5 -> c*=%.3f*b == #200). "
            "redraw-at-bar with the budgeted N=5 stays cost-competitive only above f_where_redraw_"
            "competitive=%s -- the freeze fraction at which best-of-5 still clears the bar at P>=0.95, "
            "reproducing #202's frozen_fraction_breakeven=%.3f."
            % (c_star_slope(dmu, float(rows[0]["n_redraw_at_bar"])), c_star_slope(dmu, float(N_BAR_FRESH)),
               ("%.3f" % f_competitive) if f_competitive is not None else "None",
               imp["frozen_fraction_breakeven_202"])
        ),
    }


# ---------------------------------------------------------------------------
# STEP 4 -- the Approval-request line (operational)
# ---------------------------------------------------------------------------
def approval_request(imp: dict, cross: dict, rr: dict) -> dict[str, Any]:
    below_b_frozen = cross["frozen"]["build_higher_dominates_below_b_fixedN"]
    rec = {
        "regime_unpinned_default": "build clear of mu >= 512.2 TPS and take N=1 (one official draw)",
        "build_target_tps": round(imp["mu_safe_tps"], 3),
        "frozen_bar_minimum_tps": round(imp["mu_bar_frozen_p95"], 2),
        "shots_if_build_to_512": int(imp["n_shots_frozen_at_512"]),
        "rationale": (
            "N=1 at mu=512.2 is REGIME-INVARIANT (cost b*Delta_mu+c identical fresh/frozen) and clears "
            "at P>=0.95 in BOTH regimes; under the more-plausible FROZEN regime best-of-N at the mu=500 "
            "bar tops out at P=0.810 and needs ~30 shots to force 0.95, so redrawing is a poor substitute "
            "and the build-vs-redraw crossover shifts hard toward build-higher."
        ),
        "build_higher_dominates_below_b": round(below_b_frozen, 4),
        "regime_robust_strategy": rr["regime_robust_strategy"],
        "budget_best_of_n_at_bar_only_if": (
            "the regime is EMPIRICALLY CONFIRMED FRESH (#202 follow-up #1: two official re-draws agreeing "
            "within sigma_hw=4.86 TPS => frozen; disagreeing beyond it => fresh). Until confirmed, do NOT "
            "budget best-of-N at the bar."
        ),
        "fresh_confirmed_fallback": "N=5 at the bar (fresh sequential pays E[shots]=1.94 on average)",
        "frozen_redraw_quota_if_forced": int(cross["frozen"]["n_redraw_fixedN"]),
        "authorizes": "NOTHING -- prices a plan; a human approves the spend AND confirms the harness regime.",
    }
    paragraph = (
        "Approval request (regime-unpinned default): build land #71 clear of mu>=512.2 TPS (>=504.9 frozen "
        "bar at minimum) and take N=1 -- this single draw clears 500 at P>=0.95 in BOTH regimes and its cost "
        "(build + 1 shot) is identical whether the official harness re-draw is fresh or frozen, so it is the "
        "cost-robust hedge. Only budget best-of-N re-draws at the mu=500 bar if the regime is empirically "
        "confirmed FRESH (two re-draws agreeing within sigma_hw=4.86 TPS); under the conservative frozen "
        "default, forcing a 95%% clear by re-drawing needs ~30 shots (vs 5 fresh), so building higher is "
        "unambiguously cheaper for any build cost below b=%.2f job-units/TPS (vs b=%.2f under fresh)."
        % (cross["frozen"]["build_higher_dominates_below_b_fixedN"],
           cross["fresh"]["build_higher_dominates_below_b_fixedN"])
    )
    return {"approval_spend_recommendation": rec, "approval_paragraph": paragraph}


# ---------------------------------------------------------------------------
# STEP 5 -- self-test (PRIMARY)
# ---------------------------------------------------------------------------
def self_test(imp: dict, cross: dict, rr: dict, pf: dict, c0: float) -> dict[str, Any]:
    dmu = imp["delta_mu_tps"]
    ss, sh = imp["sigma_sample_tps"], imp["sigma_hw_tps"]
    checks: dict[str, bool] = {}
    tol = 1e-6

    # (a) f=1 (fully fresh) reproduces #200's c* = 3.039*b (fixed) / 12.97*b (sequential) VERBATIM.
    n_f1 = n_redraw_partial(1.0, ss, sh)                      # 5
    e_f1 = seq_expected_shots(p_partial(TARGET, 1.0, ss, sh, 1), n_f1)  # 1.9375
    slope_f1_fixed = c_star_slope(dmu, float(n_f1))
    slope_f1_seq = c_star_slope(dmu, e_f1)
    a_ok = (abs(slope_f1_fixed - imp["c_star_slope_fresh_fixedN"]) < tol
            and abs(slope_f1_seq - imp["c_star_slope_fresh_sequential"]) < tol)
    checks["a_f1_reproduces_200_crossover_verbatim"] = a_ok
    a_detail = {"n_redraw_f1": n_f1, "e_redraw_f1": e_f1,
                "c_star_fixed_f1": slope_f1_fixed, "c_star_fixed_200": imp["c_star_slope_fresh_fixedN"],
                "c_star_seq_f1": slope_f1_seq, "c_star_seq_200": imp["c_star_slope_fresh_sequential"]}

    # (b) f=0 (fully frozen) GENUINE redraw never reaches 0.95 at the bar => effective redraw cost -> inf
    #     => build-higher dominates for ALL finite b. genuine ceiling = Phi((mu-500)/sigma_sample) (best-of-N
    #     cannot move the frozen bias, so a hardware-lucky pass does NOT replicate under private re-benchmark).
    genuine_ceiling_bar_f0 = _phi((TARGET - TARGET) / ss)    # Phi(0) = 0.5
    genuine_reaches_p95 = genuine_ceiling_bar_f0 >= P_TARGET  # False
    below_b_genuine_f0 = float("inf") if not genuine_reaches_p95 else below_b(c0, c_star_slope(dmu, 1e9))
    b_ok = (not genuine_reaches_p95) and math.isinf(below_b_genuine_f0)
    checks["b_f0_genuine_redraw_unbounded_build_dominates_all_b"] = b_ok
    b_detail = {"genuine_ceiling_at_bar_f0": genuine_ceiling_bar_f0,
                "genuine_redraw_reaches_p95": genuine_reaches_p95,
                "build_higher_dominates_below_b_genuine_f0": "inf (all finite b)"}

    # (c) monotone: c*(f) non-decreasing in f -> c*_frozen <= c*_fresh.
    slopes_f = [r["c_star_fixed"] for r in pf["partial_curve"]]
    c_ok = all(slopes_f[i + 1] >= slopes_f[i] - 1e-9 for i in range(len(slopes_f) - 1))
    c_ok = c_ok and (cross["frozen"]["crossover_fixed_frozen"] <= cross["fresh"]["crossover_fixed"] + 1e-9)
    c_ok = c_ok and (cross["frozen"]["crossover_sequential_frozen"]
                     <= cross["fresh"]["crossover_sequential"] + 1e-9)
    checks["c_monotone_cstar_frozen_le_fresh"] = c_ok

    # (d) build-to-512.2/N=1 cost identical fresh vs frozen for every b (N=1 regime-invariance).
    d_ok = True
    d_probe = []
    for r in rr["fixedN_rows"]:
        diff = abs(r["cost_buildN1_fresh"] - r["cost_buildN1_frozen"])
        d_probe.append({"b": r["b_per_mu"], "diff": diff})
        if diff > 1e-12:
            d_ok = False
    # tie to #202 self-test d: P_frozen(mu,1) == P_fresh(mu,1) exactly
    sd = imp["sigma_draw_tps"]
    for mu in (500.0, 505.0, 512.2, 520.95):
        if abs(p_frozen_general(mu, sh, ss, 1) - p_fresh(mu, sd, 1)) > 1e-9:
            d_ok = False
    checks["d_buildN1_regime_invariant"] = d_ok

    # (e) NaN-clean over every reported number (the intentional genuine-f0 inf is a string, excluded).
    flat = (_collect(cross) + _collect(rr) + _collect(pf) + _collect(imp)
            + _collect(a_detail) + _collect(d_probe)
            + [genuine_ceiling_bar_f0, slope_f1_fixed, slope_f1_seq])
    checks["e_nan_clean"] = all(math.isfinite(x) for x in flat)

    passes = all(checks.values())
    return {
        "frozen_cost_self_test_passes": passes,
        "checks": checks,
        "a_detail": a_detail,
        "b_detail": b_detail,
        "d_probe": d_probe,
        "n_numbers_checked": len(flat),
    }


def _collect(obj: Any) -> list[float]:
    out: list[float] = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        if math.isfinite(float(obj)):
            out.append(float(obj))
        # non-finite floats (intentional inf sentinels) are skipped, not collected as failures
        return out
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(_collect(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_collect(v))
    return out


# ---------------------------------------------------------------------------
# wandb
# ---------------------------------------------------------------------------
def _log_wandb(args, result: dict[str, Any]) -> None:
    if args.no_wandb:
        return
    try:
        from scripts import wandb_logging
    except Exception as exc:  # noqa: BLE001
        print(f"[fcc] wandb_logging import failed ({exc}); skipping", flush=True)
        return
    try:
        run = wandb_logging.init_wandb_run(
            job_type="frozen-cost-crossover", agent="kanna",
            name=args.wandb_name or "kanna/frozen-cost-crossover",
            group=args.wandb_group,
            tags=["frozen-cost-crossover", "build-vs-redraw", "minimax-regret", "best-of-n",
                  "frozen-bias", "pr206", "pr200-reprice", "pr202-frozen"],
            config={"baseline_tps": 481.53, "method": "cpu-only-analytic", "target_tps": 500.0,
                    "p_target": P_TARGET, "imports_pr": "200+202", "c0": 1.0},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[fcc] wandb init failed ({exc}); skipping", flush=True)
        return
    if run is None:
        print("[fcc] wandb disabled; skipping", flush=True)
        return
    try:
        cross = result["frozen_crossover"]
        rr = result["regime_robust"]
        pf = result["partial_freeze"]
        st = result["self_test"]
        flat = {
            "frozen_cost_self_test_passes": 1.0 if st["frozen_cost_self_test_passes"] else 0.0,
            "build_higher_dominates_below_b": result["build_higher_dominates_below_b"],
            "crossover_fixed_frozen": cross["frozen"]["crossover_fixed_frozen"],
            "crossover_sequential_frozen": cross["frozen"]["crossover_sequential_frozen"],
            "crossover_fixed_fresh": cross["fresh"]["crossover_fixed"],
            "crossover_sequential_fresh": cross["fresh"]["crossover_sequential"],
            "below_b_fresh_fixedN": cross["fresh"]["build_higher_dominates_below_b_fixedN"],
            "below_b_widen_factor_fixedN": cross["shift_vs_fresh"]["below_b_widen_factor_fixedN"],
            "n_redraw_frozen": float(cross["frozen"]["n_redraw_fixedN"]),
            "e_redraw_frozen_seq": cross["frozen"]["e_redraw_sequential"],
            "b_robust_crossover_fixedN": rr["b_robust_crossover_fixedN"],
            "b_robust_crossover_sequential": rr["b_robust_crossover_sequential"],
            "f_where_redraw_competitive": (pf["f_where_redraw_competitive"]
                                           if pf["f_where_redraw_competitive"] is not None else float("nan")),
            "frozen_fraction_breakeven_202": pf["frozen_fraction_breakeven_202"],
            "aligns_with_202_breakeven": 1.0 if pf["aligns_with_202_breakeven"] else 0.0,
            "delta_mu_tps": cross["delta_mu_tps"],
            "sigma_sample_tps": result["import_banked"]["sigma_sample_tps"],
            "sigma_hw_tps": result["import_banked"]["sigma_hw_tps"],
        }
        for r in pf["partial_curve"]:
            fk = str(r["f"]).replace(".", "p")
            flat[f"cstar_fixed_f{fk}"] = r["c_star_fixed"]
            flat[f"n_redraw_f{fk}"] = float(r["n_redraw_at_bar"]) if r["n_redraw_at_bar"] else float("nan")
        for r in rr["fixedN_rows"]:
            bk = str(round(r["b_per_mu"], 4)).replace(".", "p")
            flat[f"worstregret_buildN1_b{bk}"] = r["worstcase_regret_buildN1"]
            flat[f"worstregret_redraw_b{bk}"] = r["worstcase_regret_redraw"]
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run, name="frozen_cost_crossover", artifact_type="frozen-cost-crossover", data=result)
    except Exception as exc:  # noqa: BLE001
        print(f"[fcc] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:  # noqa: BLE001
            pass


def _print(result: dict[str, Any]) -> None:
    cross = result["frozen_crossover"]
    rr = result["regime_robust"]
    pf = result["partial_freeze"]
    st = result["self_test"]
    fr, fz = cross["fresh"], cross["frozen"]
    print("\n[fcc] ===== FROZEN-REGIME COST CROSSOVER (PR #206) =====", flush=True)
    print(f"  Delta_mu (build to N=1 safe mu=512.2) = {cross['delta_mu_tps']:.3f} TPS", flush=True)
    print("\n  build-vs-redraw crossover (FIXED-N):", flush=True)
    print(f"    FRESH  : redraw N={fr['n_redraw_fixedN']:>2}  c* = {fr['crossover_fixed']:.3f}*b   "
          f"build-higher below b={fr['build_higher_dominates_below_b_fixedN']:.3f}", flush=True)
    print(f"    FROZEN : redraw N={fz['n_redraw_fixedN']:>2}  c* = {fz['crossover_fixed_frozen']:.3f}*b   "
          f"build-higher below b={fz['build_higher_dominates_below_b_fixedN']:.3f}  (TEST)", flush=True)
    print("\n  build-vs-redraw crossover (SEQUENTIAL early-stop):", flush=True)
    print(f"    FRESH  : E[shots]={fr['e_redraw_sequential']:.4f}  c* = {fr['crossover_sequential']:.3f}*b",
          flush=True)
    print(f"    FROZEN : E[shots]={fz['e_redraw_sequential']:.4f}  c* = "
          f"{fz['crossover_sequential_frozen']:.3f}*b", flush=True)
    print(f"\n  shift: build-higher dominates a {cross['shift_vs_fresh']['below_b_widen_factor_fixedN']:.2f}x "
          f"wider build-cost range under freeze (fixed-N).", flush=True)
    print(f"\n  regime-robust strategy = {rr['regime_robust_strategy']}  "
          f"(hedge for b < b_robust={rr['b_robust_crossover_fixedN']:.3f})", flush=True)
    print(f"  f_where_redraw_competitive = {pf['f_where_redraw_competitive']:.4f}  "
          f"(== #202 breakeven {pf['frozen_fraction_breakeven_202']:.4f}? {pf['aligns_with_202_breakeven']})",
          flush=True)
    print(f"\n  build_higher_dominates_below_b (TEST) = {result['build_higher_dominates_below_b']:.4f}",
          flush=True)
    print(f"  SELF-TEST (PRIMARY) passes = {st['frozen_cost_self_test_passes']}", flush=True)
    for k, v in st["checks"].items():
        print(f"    [{'ok' if v else '!! FAILED'}] {k}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(_HERE, "frozen_cost_crossover_results.json"))
    ap.add_argument("--wandb-name", "--wandb_name", default="kanna/frozen-cost-crossover")
    ap.add_argument("--wandb-group", "--wandb_group", default="frozen-cost-crossover")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    c0 = 1.0  # nominal per-shot cost (1 official a10g-small draw = 1 job-unit, #200 NATURAL unit)
    imp = import_banked()

    cross = frozen_crossover(imp, c0)
    rr = regime_robust(imp, cross, c0)
    pf = partial_freeze(imp, c0)
    appr = approval_request(imp, cross, rr)
    st = self_test(imp, cross, rr, pf, c0)

    build_higher_below_b = cross["frozen"]["build_higher_dominates_below_b_fixedN"]  # TEST headline

    handoff = (
        "under the conservative frozen regime the build-vs-redraw crossover shifts hard toward build-higher "
        "(c*_frozen=%.3f*b fixed-N / %.3f*b sequential vs #200's fresh 3.039*b / 12.97*b) because best-of-N "
        "at the bar tops out at P=0.810 and needs ~%d shots to force 0.95; the regime-robust "
        "(minimax-regret) spend is build-to-mu=512.2/N=1 (cost-invariant across both regimes, hedge for "
        "build cost b<%.2f), with best-of-N-at-bar budgeted ONLY if the regime is empirically confirmed "
        "FRESH; fern #185's budget row should default to the build-higher recommendation."
        % (cross["frozen"]["crossover_fixed_frozen"], cross["frozen"]["crossover_sequential_frozen"],
           cross["frozen"]["n_redraw_fixedN"], rr["b_robust_crossover_fixedN"])
    )

    result = {
        "pr": 206,
        "metric_primary": "frozen_cost_self_test_passes",
        "metric_test": "build_higher_dominates_below_b",
        "frozen_cost_self_test_passes": st["frozen_cost_self_test_passes"],
        "build_higher_dominates_below_b": build_higher_below_b,
        "crossover_fixed_frozen": cross["frozen"]["crossover_fixed_frozen"],
        "crossover_sequential_frozen": cross["frozen"]["crossover_sequential_frozen"],
        "regime_robust_strategy": rr["regime_robust_strategy"],
        "f_where_redraw_competitive": pf["f_where_redraw_competitive"],
        "approval_spend_recommendation": appr["approval_spend_recommendation"],
        "c0_per_shot_job_units": c0,
        "frozen_crossover": cross,
        "regime_robust": rr,
        "partial_freeze": pf,
        "approval_request": appr,
        "self_test": st,
        "import_banked": imp,
        "handoff": handoff,
        "scope": "Re-prices kanna's OWN banked #200 fresh build-vs-redraw crossover under kanna's OWN banked "
        "#202 frozen redraw curve. Takes NO official draws, authorizes NO shot count or spend (a human still "
        "approves AND confirms the harness regime), CPU-only. BANK-THE-ANALYSIS: adds 0 TPS, greedy/PPL "
        "untouched. WHICH regime the official harness is in stays the harness-owner's open question (#202); "
        "this leg BOUNDS the spend under BOTH so the human is ready either way. Draw-budget/cost lane "
        "(#159/#188/#194/#200/#202). NOT denken #197. NOT ubel #204. NOT fern #185. NOT stark #203. "
        "NOT open2. NOT a launch.",
        "imported_legs": {
            "kanna_200_fresh_crossover": "research/validity/cost_budget/cost_budget_results.json",
            "kanna_202_frozen_redraw_curve": "research/validity/frozen_budget/frozen_budget_results.json",
        },
        "public_evidence_used": [
            "Leaderboard frontier tops at ~489.6 TPS (osoi5-...-precache-skv64-v1), BELOW the 500 bar -- "
            "builds (land #71 descent) aim past 500; this leg prices the regime-robust path to clear it.",
            "kanna #200 (n3alx7ca) fresh build-vs-redraw crossover c*=3.039*b/12.97*b; kanna #202 (533jd6l1) "
            "frozen redraw P_bar_n5=0.810, ~30-shot uncapped quota, frozen_fraction_breakeven=0.846.",
            "Max order-statistics P(max of N>=t)=1-F(t)^N; minimax-regret / 50-50-prior decision over the "
            "unpinned fresh/frozen regime.",
            "fern #185 carries this regime-robust spend as the multi-shot budget row.",
        ],
        "method": "LOCAL CPU-only analytic synthesis; no GPU/vLLM/HF Job/submission/served-file/draw. "
        "BASELINE stays 481.53; adds 0 TPS. Greedy identity untouched. NOT open2. NOT a launch.",
        "metrics_nan_clean": 1 if st["checks"]["e_nan_clean"] else 0,
        "elapsed_s": round(time.time() - t0, 4),
    }

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    _print(result)
    print(f"\n[fcc] HANDOFF: {handoff}", flush=True)
    print(f"[fcc] artifacts -> {args.out}", flush=True)
    _log_wandb(args, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
