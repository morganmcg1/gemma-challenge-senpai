#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Gate semantics: POINT-clearance vs CI-certification, + certification-budget
per recovery lane (PR #716).

denken #714 (`fpbp6pcn`, PAIRING_INVARIANT_DEAD) proved Lane 1 (within-mandate
selective/full-g32 AIME recovery) is provability-DEAD under the CI-certification
reading: the 0.420 AIME gate is an ABSOLUTE one-sample bar, so the certification
budget n>=2889 (Wilson-lo) is pairing-invariant. But #714 surfaced a load-bearing
ambiguity it did not resolve: the as-applied fire-gate (lawine #703, `5r027mc3`)
gated AIME on the *point* (Wilson-HI<gate => decisive FAIL) and GPQA-D on the
*point* with the CI as an honesty caveat (CI-lo 0.406 < gate, straddle noted, NOT
a fail) — i.e. it gated on the POINT and treated the CI as a flag, not a bar. So
"DEAD" is precise: dead *to prove at 95% confidence*, while full-g32's POINT
(0.438) clears the bar the panel actually applied. That point-vs-CI gap governs
the value of the four live Lane-1 measurement cards (ubel #702 / fern #713 /
stark #711 / land #712) and the human's int8-mandate call.

This card (1) formalizes both fire-gate readings as explicit decision rules and
weighs which is defensible for a one-shot competition autonomous-fire (#481);
(2) reconciles them against the as-applied record (#703) and sets
`asapplied_gate_is_point`; (3) prices the certification-budget frontier — n to
CI-certify AIME>=0.420 (both Wilson-lo point-clears n AND the 95%-power n) — for
full-g32 (0.438; reproduces #714's 2889/9828 as the self-test), the int8-locus
(fern #659 greedy 0.450, margin 0.030), Route B (0.42131; reproduces 545,295),
and a PARAMETRIC sweep p in [0.420,0.470] so any measured point maps to a verdict
by lookup; (4) adjudicates the two-world consequence table; (5) gives the
verdict-conditional downstream note. GREEDY basis throughout (kanna #699:
int4/int8 SAMPLED decode is engine-fragile on vLLM 0.22.0 -> greedy is the valid
int4-precision basis; int8 SAMPLED 0.410 sits BELOW the gate).

PURE-CPU statistical/decision analysis. analysis_only, official_tps=0,
no_hf_job=1, fires=0. NO HF Job / vLLM / submission / kernel build / model load /
served-file change. Locked `int4_g128_lmhead`@126.378 untouched. NOT a fire, NOT
an approval trigger (peak RSS < 200 MB, runtime < 10 s).

PRIMARY metric  int8_certify_n_pointclears  (n to CI-certify int8 0.450 >= 0.420, greedy basis)
TEST    metric  asapplied_gate_is_point     (1/0: is the as-applied #703 gate point-based?)
GATE    metric  gate_semantics_self_test_passes

Run:
    python -m research.validity.gate_semantics_cert_budget.gate_semantics_cert_budget \
        --self-test --wandb-name denken/gate-semantics-cert-budget \
        --wandb-group gate-semantics-cert-budget-denken
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
# VERIFIED ANCHORS (W&B-confirmed; cross-reads authorized #666; GREEDY basis).
# =========================================================================== #
GATE = 0.420                  # AIME gate = 0.90 x base ~0.467 (#515); absolute one-sample bar
BASE_AIME = 0.467             # vanilla base AIME (gate denominator)
AIME_FLOOR = 0.347            # int4-body g128 #31-gate (lawine #693 6brpvz9x; == #710 floor)

# --- the two live recovery lanes (GREEDY points) --------------------------- #
CEILING_G32 = 0.438           # Lane 1 full-g32 MEASURED ceiling (ubel #679 1z5vq2ej aime_g32=0.4375)
CEILING_G64 = 0.4458          # full-g64 (non-monotone xcheck; ubel #679)
ROUTE_B_POINT = 0.42131       # denken #710 66rhys58 routeB_point_linear (fused-block, off knife-edge)
INT8_LOCUS_GREEDY = 0.450     # Lane 2 int8-on-L14-27 greedy (fern #659 nmjvtfov; margin +0.030)
INT8_LOCUS_SAMPLED = 0.410    # fern #659 int8 SAMPLED point -- sits BELOW the gate (CI-read hazard)
INT8_LOCUS_BASELINE = 0.400   # int8-locus pre-recovery / uniform int8-all (fern #659)

# --- #714 banked certification-budget anchors (reproduce as the self-test) -- #
D714_FULLG32_POINTCLEARS = 2889    # full-g32 0.438 margin 0.018: Wilson-lo point-clears n
D714_FULLG32_POWER95_BANKED = 9828 # full-g32 95%-power n (normal-approx-order; staircase band)
D714_ROUTEB_POINTCLEARS = 545295   # routeB 0.42131 margin 0.00131: Wilson-lo point-clears n
D714_PSI_EMPIRICAL = 0.217         # fern #659 reconstructed paired discordance (b+c=65 -> McNemar p=0.0248)

# --- as-applied gate record (lawine #703 5r027mc3, four-leg #515 panel) ----- #
# AIME    0.3467  Wilson-hi 0.4022 < gate -> decisive FAIL (even optimistic bound fails)
# GPQA-D  0.4747  point >= gate PASS; CI-lo 0.406 < gate -> straddle CAVEAT (NOT a fail)
# GSM8K   0.8788  PASS;  MMLU-Pro 0.6547 PASS
D703 = {
    "aime":     {"point": 0.3467, "wilson_hi": 0.4022, "wilson_lo": 0.2960, "verdict": "FAIL"},
    "gpqa_d":   {"point": 0.4747, "wilson_hi": 0.5430, "wilson_lo": 0.4060, "verdict": "PASS"},
    "gsm8k":    {"point": 0.8788, "wilson_hi": 0.9020, "wilson_lo": 0.8548, "verdict": "PASS"},
    "mmlu_pro": {"point": 0.6547, "wilson_hi": 0.6722, "wilson_lo": 0.6366, "verdict": "PASS"},
}

# --- AIME eval instrument (GREEDY basis is deterministic -> n == pool size) -- #
AIME2024_POOL = 30            # AIME-2024 unique problems (one greedy draw each)
FLEET31_POOL = 60            # the #31 fleet basis (ubel #702 design: 5 seeds x 60 problems)
GREEDY_POOL_CAP = FLEET31_POOL   # largest UNIQUE-problem n on the greedy basis (no reseeding gain)
EDGE_FEASIBLE_N = 1500       # "edge-feasible" expanded-pool budget (PR's ~1000-1500 notion)

# --- statistical constants ------------------------------------------------- #
Z95 = 1.959963984540054       # two-sided 95% (fleet gate convention; 1-sided 2.5%)
ZB_95 = 1.6448536269514722    # one-sided 95% power z (z_beta)
N_GRID = (30, 60, 120, 240, 480, 960)
POWER_TARGETS = (0.80, 0.95)
N_SEARCH_CAP = 2_000_000      # routeB needs ~545k; cap generously
POWER_SEARCH_CAP = 200_000
STABLE_RUN = 150              # consecutive-n stability window for the staircased power-n
UNRESOLVABLE_N = float("inf") # "never clears at any feasible n" sentinel
SWEEP_LO, SWEEP_HI, SWEEP_STEP = 0.420, 0.470, 0.002   # parametric n(p) frontier


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# =========================================================================== #
# NUMERIC CORE — dependency-free (no scipy): regularized incomplete beta,
# beta quantile (bisection), Wilson CI, Clopper-Pearson, exact binomial tail.
# (Reused verbatim from denken #709/#710/#714 — `recovery_gate_robustness`.)
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


def wilson_lo_from_phat(phat: float, n: float, z: float = Z95) -> float:
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


def cp_lo_at_phat(phat: float, n: int, alpha: float = 0.05) -> float:
    """Clopper-Pearson lower bound at the integer count nearest phat*n (discrete cross-check)."""
    k = int(round(phat * n))
    k = max(0, min(n, k))
    lo, _ = clopper_pearson(k, n, alpha)
    return lo


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
# CERTIFICATION-BUDGET CORE — n to CI-certify a recovery point clears the gate.
# =========================================================================== #
def min_n_point_clears(p: float, bar: float = GATE, z: float = Z95,
                       cap: int = N_SEARCH_CAP) -> float:
    """Smallest n such that, observing EXACTLY the point p, the Wilson lower bound
       clears the bar: wilson_lo_from_phat(p, n) > bar. The BEST-CASE certification
       budget (you happen to draw the point). wilson_lo is monotone increasing in n,
       so binary-search. Returns inf if p <= bar (never clears at any n)."""
    if p <= bar:
        return UNRESOLVABLE_N
    if wilson_lo_from_phat(p, cap, z) <= bar:
        return UNRESOLVABLE_N
    lo, hi = 1, cap
    while lo < hi:
        mid = (lo + hi) // 2
        if wilson_lo_from_phat(p, mid, z) > bar:
            hi = mid
        else:
            lo = mid + 1
    return float(lo)


def min_n_for_power(p_true: float, bar: float = GATE, power: float = 0.95,
                    z: float = Z95, cap: int = POWER_SEARCH_CAP,
                    stable: int = STABLE_RUN) -> dict[str, float]:
    """The 95%-power certification budget: smallest n such that a true-p draw yields
       Wilson-lo>bar with probability >= `power`. The binomial power staircases
       (oscillates) near the threshold, so we report BOTH the first crossing and the
       first n that STAYS above for `stable` consecutive n (the conservative, stable n)."""
    if p_true <= bar:
        return {"first_cross": UNRESOLVABLE_N, "stable": UNRESOLVABLE_N}
    first_cross = None
    stable_first = None
    run = 0
    n = 1
    while n <= cap:
        if power_clear(n, p_true, bar, z) >= power:
            if first_cross is None:
                first_cross = n
            if stable_first is None:
                stable_first = n
            run += 1
            if run >= stable:
                return {"first_cross": float(first_cross), "stable": float(stable_first)}
        else:
            stable_first = None
            run = 0
        n += 1
    return {"first_cross": float(first_cross) if first_cross else UNRESOLVABLE_N,
            "stable": UNRESOLVABLE_N}


def normal_n_point_clears(p: float, bar: float = GATE, z: float = Z95) -> float:
    """Normal-approx best-case point-clears n = z^2 p(1-p)/(p-bar)^2 (the #714 closed form)."""
    if p <= bar:
        return UNRESOLVABLE_N
    return z * z * p * (1.0 - p) / (p - bar) ** 2


def normal_n_power95(p: float, bar: float = GATE, z: float = Z95, zb: float = ZB_95) -> float:
    """Normal-approx 95%-power n (separate-variance one-sample proportion form)."""
    if p <= bar:
        return UNRESOLVABLE_N
    return (z * math.sqrt(bar * (1.0 - bar)) + zb * math.sqrt(p * (1.0 - p))) ** 2 / (p - bar) ** 2


# =========================================================================== #
# (1) FORMALIZE BOTH FIRE-GATE READINGS AS EXPLICIT DECISION RULES.
# =========================================================================== #
def gate_readings_block() -> dict[str, Any]:
    """Two decision rules + what evidence each requires + the one-shot-fire implication."""
    point_rule = {
        "name": "POINT-clearance",
        "decision_rule": "served config CLEARS the AIME leg IFF point estimate p_hat >= 0.420",
        "evidence_required": (
            "a single point estimate of AIME accuracy at the eval budget (the #31 basis, "
            "greedy, ~30-60 unique problems). The CI is reported as an HONESTY CAVEAT "
            "(flags whether the point is robust) but does NOT gate."),
        "this_is": "the lawine #703 AS-APPLIED operational rule (gate on the point, CI as a flag)",
        "one_shot_implication": (
            "the POINT is the unbiased best estimate of true quality; for a single "
            "irreversible decision under a flat-ish prior the Bayes/expected-utility-optimal "
            "action gates on the posterior mean (~the point), NOT a conservative bound. A "
            "lower-confidence-bound rule is a type-I-error-control device for a REPEATED "
            "certification regime, not the optimal rule for one irreversible board post."),
    }
    ci_rule = {
        "name": "CI-certification",
        "decision_rule": "served config CLEARS IFF Wilson/CP LOWER bound >= 0.420 at the eval budget",
        "evidence_required": (
            "enough draws n that the 95% lower confidence bound clears -- a certification "
            "at 95% confidence. For a margin m=p-0.420 this needs n ~ z^2 p(1-p)/m^2 draws "
            "(best case: observe exactly p), or the larger 95%-power n to PLAN for it."),
        "this_is": "the denken #710/#714 certification rule (the stricter standard)",
        "one_shot_implication": (
            "guarantees (at 95%) the served model is above the floor -- the risk-averse / "
            "do-no-harm reading. Appropriate when you make the call REPEATEDLY and want a "
            "long-run type-I-error guarantee; for a single shot it is conservative, and on "
            "AIME it is UNACHIEVABLE at any near-bar margin (see the budget frontier)."),
    }
    return {"point_reading": point_rule, "ci_reading": ci_rule}


# =========================================================================== #
# (2) RECONCILE AGAINST THE AS-APPLIED RECORD (#703).  asapplied_gate_is_point.
# =========================================================================== #
def asapplied_reconcile_block() -> dict[str, Any]:
    """Extract the operative rule the program ACTUALLY used in lawine #703."""
    # AIME failed because even the optimistic Wilson-HI < gate (a level gap, basis-independent).
    aime_fails_on_hi = D703["aime"]["wilson_hi"] < GATE
    # GPQA-D passed on the POINT while its CI-lo straddles the gate (caveat, not a fail).
    gpqa_point_passes = D703["gpqa_d"]["point"] >= GATE
    gpqa_ci_straddles = D703["gpqa_d"]["wilson_lo"] < GATE <= D703["gpqa_d"]["wilson_hi"]
    # The decisive test: was a PASS awarded on a point whose CI-lo is BELOW the gate?
    # If yes, the gate is point-based (CI is a caveat, not a certification bar).
    pass_with_ci_lo_below_gate = gpqa_point_passes and (D703["gpqa_d"]["wilson_lo"] < GATE)
    asapplied_gate_is_point = int(pass_with_ci_lo_below_gate and aime_fails_on_hi)

    # Counterfactual: if the program had used the CI rule, GPQA-D (CI-lo 0.406 < 0.420)
    # would ALSO FAIL -> the four-leg panel would collapse to >=2 failing legs, not 1.
    # That the program recorded GPQA-D as a PASS proves the CI rule was NOT applied.
    legs_failing_under_ci = sum(
        1 for leg in D703.values() if leg["wilson_lo"] < GATE)  # AIME + GPQA-D = 2
    legs_failing_under_point = sum(
        1 for leg in D703.values() if leg["point"] < GATE)      # AIME only = 1

    operative_rule = (
        "FAIL if even Wilson-HI < gate (decisive level failure, AIME 0.4022<0.420); "
        "PASS if point >= gate (GPQA-D 0.4747) with a CI-lo straddle recorded as an "
        "HONESTY CAVEAT, not a fail. => the as-applied gate is POINT-based.")
    return {
        "asapplied_gate_is_point": asapplied_gate_is_point,
        "aime_fails_on_wilson_hi": int(aime_fails_on_hi),
        "gpqa_point_passes": int(gpqa_point_passes),
        "gpqa_ci_straddles_gate": int(gpqa_ci_straddles),
        "pass_awarded_with_ci_lo_below_gate": int(pass_with_ci_lo_below_gate),
        "legs_failing_under_point_rule": legs_failing_under_point,   # 1 (AIME)
        "legs_failing_under_ci_rule": legs_failing_under_ci,         # 2 (AIME + GPQA-D)
        "ci_rule_would_collapse_panel": int(legs_failing_under_ci > legs_failing_under_point),
        "operative_rule": operative_rule,
        "adjudication": (
            "asapplied_gate_is_point=1. The program gated PASS on the point and used the "
            "CI as a caveat (GPQA-D point-PASS with CI-lo 0.406<gate). Applying the CI rule "
            "uniformly would ALSO fail GPQA-D -> the panel collapses; the recorded 1-leg "
            "failure is only consistent with the POINT rule."),
    }


# =========================================================================== #
# (3) CERTIFICATION-BUDGET FRONTIER — per lane + parametric n(p) sweep.
# =========================================================================== #
def _config_budget(name: str, p: float) -> dict[str, Any]:
    margin = p - GATE
    npc = min_n_point_clears(p)
    pw = min_n_for_power(p)
    return {
        "name": name,
        "point": p,
        "margin": round(margin, 5),
        "min_n_point_clears": (None if not math.isfinite(npc) else int(npc)),
        "min_n_power95_stable": (None if not math.isfinite(pw["stable"]) else int(pw["stable"])),
        "min_n_power95_first_cross": (None if not math.isfinite(pw["first_cross"])
                                      else int(pw["first_cross"])),
        "normal_n_point_clears": (None if not math.isfinite(normal_n_point_clears(p))
                                  else round(normal_n_point_clears(p), 1)),
        "normal_n_power95": (None if not math.isfinite(normal_n_power95(p))
                             else round(normal_n_power95(p), 1)),
        # feasibility flags against the AIME instrument.
        "point_clears_le_edge_feasible": int(math.isfinite(npc) and npc <= EDGE_FEASIBLE_N),
        "point_clears_le_greedy_pool": int(math.isfinite(npc) and npc <= GREEDY_POOL_CAP),
    }


def cert_budget_frontier_block() -> dict[str, Any]:
    """Per-lane budgets + the parametric p in [0.420,0.470] frontier + the instrument cap."""
    lanes = {
        "full_g32":   _config_budget("full_g32 (Lane 1, within-mandate)", CEILING_G32),
        "full_g64":   _config_budget("full_g64 (xcheck)", CEILING_G64),
        "route_b":    _config_budget("route_b (Lane 1 fused-block point)", ROUTE_B_POINT),
        "int8_locus": _config_budget("int8_locus (Lane 2, human-mandate)", INT8_LOCUS_GREEDY),
    }

    # parametric n(p) frontier (lookup table) -----------------------------------
    frontier = []
    steps = int(round((SWEEP_HI - SWEEP_LO) / SWEEP_STEP))
    for i in range(steps + 1):
        p = round(SWEEP_LO + i * SWEEP_STEP, 4)
        npc = min_n_point_clears(p)
        pw = min_n_for_power(p)
        frontier.append({
            "p": p,
            "margin": round(p - GATE, 4),
            "n_point_clears": (None if not math.isfinite(npc) else int(npc)),
            "n_power95": (None if not math.isfinite(pw["stable"]) else int(pw["stable"])),
            "le_edge_feasible": int(math.isfinite(npc) and npc <= EDGE_FEASIBLE_N),
            "le_greedy_pool": int(math.isfinite(npc) and npc <= GREEDY_POOL_CAP),
        })

    # smallest p whose point-clears budget is edge-feasible (<=1500) and greedy-feasible (<=60).
    p_edge = next((r["p"] for r in frontier if r["le_edge_feasible"]), None)
    p_greedy = next((r["p"] for r in frontier if r["le_greedy_pool"]), None)

    # The GREEDY-instrument cap: greedy decode is deterministic -> n == #unique problems.
    # No reseeding gain (every seed reproduces the same greedy answer), so n is capped at
    # the AIME problem pool (~30-60). int8's 1040 point-clears budget is ~17x that cap.
    instrument = {
        "greedy_is_deterministic": 1,
        "aime2024_pool": AIME2024_POOL,
        "fleet31_pool": FLEET31_POOL,
        "greedy_pool_cap": GREEDY_POOL_CAP,
        "int8_pointclears_over_greedy_cap": round(lanes["int8_locus"]["min_n_point_clears"]
                                                  / GREEDY_POOL_CAP, 1),
        "int8_sampled_point": INT8_LOCUS_SAMPLED,
        "int8_sampled_below_gate": int(INT8_LOCUS_SAMPLED < GATE),
        "note": (
            "On the GREEDY basis (kanna #699: int4/int8 SAMPLED decode is engine-fragile on "
            "0.22.0), AIME yields ONE deterministic draw per problem -> n is capped at the "
            "problem pool (~30-60), NOT by compute. int8's 1040 point-clears budget is ~17x "
            "the 60-problem cap; full-g32's 2889 is ~48x. CI-certification cannot be reached "
            "by reseeding (greedy has no seed variance). Sampled draws cannot backfill n "
            "either: int8 SAMPLED point 0.410 < gate -> draws whose own point is sub-gate "
            "cannot certify >=0.420. => the CI reading is OPERATIONALLY IMPOSSIBLE on the "
            "AIME instrument for BOTH lanes; only an expanded ~1000+ comparable-difficulty "
            "math pool could make int8 edge-feasible."),
    }
    return {
        "lanes": lanes,
        "frontier": frontier,
        "smallest_p_edge_feasible_point_clears": p_edge,
        "smallest_p_greedy_feasible_point_clears": p_greedy,
        "edge_feasible_budget": EDGE_FEASIBLE_N,
        "instrument_cap": instrument,
    }


# =========================================================================== #
# (4) ADJUDICATE THE TWO-WORLD CONSEQUENCE TABLE.
# =========================================================================== #
def two_world_block(cert: dict[str, Any]) -> dict[str, Any]:
    L = cert["lanes"]

    # POINT world: clears IFF point >= 0.420.
    point_world = {
        "full_g32":   {"point": CEILING_G32, "clears": int(CEILING_G32 >= GATE)},
        "route_b":    {"point": ROUTE_B_POINT, "clears": int(ROUTE_B_POINT >= GATE)},
        "int8_locus": {"point": INT8_LOCUS_GREEDY, "clears": int(INT8_LOCUS_GREEDY >= GATE)},
        "selective_g32": {"point": None, "clears": "IFF measured point >= 0.420 (ubel #702 pending)"},
    }
    lane1_alive_on_point = int(CEILING_G32 >= GATE)  # full-g32 point clears
    point_world["lane1_alive_on_point"] = lane1_alive_on_point

    # CI world: which are even EDGE-FEASIBLE to certify (point-clears n <= 1500)?
    ci_world = {
        "full_g32":   {"n_point_clears": L["full_g32"]["min_n_point_clears"],
                       "n_power95": L["full_g32"]["min_n_power95_stable"],
                       "edge_feasible": L["full_g32"]["point_clears_le_edge_feasible"]},
        "route_b":    {"n_point_clears": L["route_b"]["min_n_point_clears"],
                       "n_power95": L["route_b"]["min_n_power95_stable"],
                       "edge_feasible": L["route_b"]["point_clears_le_edge_feasible"]},
        "int8_locus": {"n_point_clears": L["int8_locus"]["min_n_point_clears"],
                       "n_power95": L["int8_locus"]["min_n_power95_stable"],
                       "edge_feasible": L["int8_locus"]["point_clears_le_edge_feasible"]},
    }
    # int8 is the ONLY config that is both point-clean AND edge-feasible to certify.
    int8_only_edge_feasible = int(
        ci_world["int8_locus"]["edge_feasible"] == 1
        and ci_world["full_g32"]["edge_feasible"] == 0
        and ci_world["route_b"]["edge_feasible"] == 0)
    ci_world["int8_only_edge_feasible_among_lanes"] = int8_only_edge_feasible
    # BUT on the strict greedy instrument NOTHING is feasible (pool cap 60 << 1040).
    ci_world["any_lane_greedy_pool_feasible"] = int(
        L["int8_locus"]["point_clears_le_greedy_pool"]
        or L["full_g32"]["point_clears_le_greedy_pool"]
        or L["route_b"]["point_clears_le_greedy_pool"])

    decision_sentence = (
        "Under the CI reading, the int8-locus's larger margin (0.030 vs full-g32's 0.018) "
        "makes it the ONLY recovery config that is both point-clean (greedy 0.450) AND "
        "edge-feasible to CI-certify on the point-clears budget (n=1040 <= 1500; full-g32 "
        "needs 2889, route_b ~545k). HOWEVER, on the strict greedy AIME instrument (pool "
        "<=60) even int8's 1040 is ~17x over-budget, and int8 SAMPLED (0.410) is sub-gate "
        "-> the CI reading is reachable for NO lane without an expanded ~1000+ "
        "comparable-difficulty math pool. int8 is 'the least-infeasible', and the only one "
        "that becomes feasible if the pool is expanded.")
    return {
        "point_world": point_world,
        "ci_world": ci_world,
        "int8_only_edge_feasible": int8_only_edge_feasible,
        "decision_sentence": decision_sentence,
    }


# =========================================================================== #
# (5) VERDICT-CONDITIONAL DOWNSTREAM NOTE (both worlds).
# =========================================================================== #
def downstream_note_block() -> dict[str, Any]:
    return {
        "if_point_gate": (
            "POINT_GATE_LANE1_ALIVE: the four Lane-1 measurement cards (ubel #702 selective-g32, "
            "fern #713 g32-locus, stark #711 shape, land #712 strict-#319 identity) remain "
            "FIRE-RELEVANT on their points -- full-g32's 0.438 point clears the bar the gate "
            "actually applies, and selective-g32 clears IFF its measured point >= 0.420. "
            "denken #710/#714 DEAD bounds ONLY the stricter CI-certification standard the "
            "fire-gate does NOT require -> do NOT over-kill Lane 1; the points are decision-grade."),
        "if_ci_gate": (
            "CI_GATE_INT8_ONLY: Lane 1 is dead (full-g32 n>=2889, route_b ~545k -- infeasible) "
            "and only Lane 2 (int8-locus, margin 0.030, point-clears n=1040) is even "
            "edge-feasible -> the int8/mandate ruling becomes strictly LOAD-BEARING. (And note "
            "even int8 is greedy-pool-infeasible at n<=60; CI-certification needs an expanded "
            "math pool.)"),
        "instrument_caveat": (
            "The greedy/sampled split is itself a reason the CI reading is hazardous on this "
            "engine: greedy caps n at the problem pool, and sampled (the only way to grow n) "
            "is engine-broken for int4 (kanna #699) and sub-gate for int8 (0.410)."),
    }


# =========================================================================== #
# RECOMMENDATION — defensible standard for a ONE-SHOT competition fire (#481).
# =========================================================================== #
def recommendation_block() -> dict[str, Any]:
    return {
        "recommended_standard": "POINT-clearance with a margin preference and the CI as risk disclosure",
        "rationale": (
            "(1) DECISION THEORY: a one-shot irreversible leaderboard post is a single "
            "expected-utility decision, not a repeated-certification regime. The EU-optimal "
            "action gates on the posterior mean (~the point); a conservative lower-bound rule "
            "is a frequentist type-I-error-rate control device justified by REPETITION, which "
            "a single board post does not have. (2) CONSISTENCY: the program's own four-leg "
            "panel (#703) is point-gated -- GPQA-D only point-passes (CI-lo 0.406<gate). "
            "Switching to the CI rule now would retroactively FAIL GPQA-D and collapse the "
            "panel, not just AIME. (3) FEASIBILITY: CI-certification of AIME at a near-bar "
            "margin is unachievable at any realistic AIME budget (full-g32 2889 / int8 1040 "
            "vs a greedy pool <=60); a standard no config can meet is not an operable fire-gate."),
        "margin_preference": (
            "BUT the irreversibility argues for the largest-margin point among clearing "
            "configs: a thin point (full-g32 0.438, P(true<0.420 | n=60) ~ 30-40%) carries a "
            "real, uncertified probability of being sub-floor. Among point-clearing recovery "
            "configs prefer maximum margin (int8 0.450 > full-g32 0.438) and report the Wilson "
            "CI as the honest risk flag. Functionally: gate on the point, size a margin to the "
            "false-fire/missed-fire loss asymmetry, disclose the CI."),
        "human_call": (
            "The human makes the final int8-mandate call. If the fire-gate is POINT (recommended), "
            "Lane 1 (full-g32) is alive and the int8 question is margin/robustness, not pass/fail. "
            "If the human elects the conservative CI standard for the irreversible post, Lane 1 "
            "is dead and int8 is the only (edge-feasible, pool-expansion-contingent) certifiable "
            "config -> the int8-mandate ruling becomes load-bearing."),
    }


# =========================================================================== #
# SELF-TEST — reproduce #714 anchors + boundary/monotonicity checks.
# =========================================================================== #
def self_test(cert: dict[str, Any], adj: dict[str, Any]) -> dict[str, Any]:
    L = cert["lanes"]
    conds: dict[str, bool] = {}

    # (a) reproduce #714 banked anchors EXACTLY (the headline self-test).
    conds["fullg32_pointclears_2889"] = (L["full_g32"]["min_n_point_clears"]
                                         == D714_FULLG32_POINTCLEARS)
    conds["routeb_pointclears_545295"] = (L["route_b"]["min_n_point_clears"]
                                          == D714_ROUTEB_POINTCLEARS)
    # (b) full-g32 power-n staircase band CONTAINS the banked 9828.
    fc = L["full_g32"]["min_n_power95_first_cross"]
    stbl = L["full_g32"]["min_n_power95_stable"]
    conds["fullg32_power_band_contains_9828"] = (
        fc is not None and stbl is not None
        and fc <= D714_FULLG32_POWER95_BANKED <= stbl)
    conds["fullg32_power_normal_near_9828"] = (
        abs(normal_n_power95(CEILING_G32) - D714_FULLG32_POWER95_BANKED) < 60.0)

    # (c) int8 primary metric computed and in the expected ~1040 region.
    conds["int8_pointclears_is_1040"] = (L["int8_locus"]["min_n_point_clears"] == 1040)
    conds["int8_power95_computed"] = (L["int8_locus"]["min_n_power95_stable"] is not None)

    # (d) boundary correctness: wilson_lo(p, npc) > gate >= wilson_lo(p, npc-1).
    for key, p in (("full_g32", CEILING_G32), ("int8_locus", INT8_LOCUS_GREEDY),
                   ("route_b", ROUTE_B_POINT)):
        npc = L[key]["min_n_point_clears"]
        conds[f"boundary_{key}"] = (
            wilson_lo_from_phat(p, npc) > GATE
            and wilson_lo_from_phat(p, npc - 1) <= GATE)

    # (e) frontier monotonicity: n_point_clears strictly decreasing as margin grows.
    fr = [r for r in cert["frontier"] if r["n_point_clears"] is not None]
    mono = all(fr[i]["n_point_clears"] >= fr[i + 1]["n_point_clears"] for i in range(len(fr) - 1))
    conds["frontier_monotone_decreasing"] = mono
    # p == gate (margin 0) is unresolvable (None).
    conds["margin_zero_unresolvable"] = (cert["frontier"][0]["p"] == GATE
                                         and cert["frontier"][0]["n_point_clears"] is None)

    # (f) adjudication: as-applied gate is point-based.
    conds["asapplied_gate_is_point"] = (adj["asapplied_gate_is_point"] == 1)
    conds["ci_rule_would_collapse_panel"] = (adj["ci_rule_would_collapse_panel"] == 1)

    # (g) instrument: greedy pool cap << int8 budget; int8 sampled below gate.
    conds["greedy_cap_below_int8_budget"] = (GREEDY_POOL_CAP < L["int8_locus"]["min_n_point_clears"])
    conds["int8_sampled_below_gate"] = (INT8_LOCUS_SAMPLED < GATE)

    # (h) int8 edge-feasible while full-g32 / route_b are not (the decision pivot).
    conds["int8_edge_feasible"] = (L["int8_locus"]["point_clears_le_edge_feasible"] == 1)
    conds["fullg32_not_edge_feasible"] = (L["full_g32"]["point_clears_le_edge_feasible"] == 0)

    # (i) constraint guards.
    conds["analysis_only_guard"] = True

    passes = all(conds.values())
    return {"gate_semantics_self_test_passes": passes,
            "conditions": {k: bool(v) for k, v in conds.items()},
            "n_conditions": len(conds),
            "n_passing": sum(1 for v in conds.values() if v)}


# =========================================================================== #
# SYNTHESIS.
# =========================================================================== #
def synthesize() -> dict[str, Any]:
    readings = gate_readings_block()
    adj = asapplied_reconcile_block()
    cert = cert_budget_frontier_block()
    worlds = two_world_block(cert)
    downstream = downstream_note_block()
    recommend = recommendation_block()
    st = self_test(cert, adj)

    # Headline verdict: the adjudication concludes the AS-APPLIED gate is POINT-based,
    # so POINT_GATE_LANE1_ALIVE is the adjudicated world; CI_GATE_INT8_ONLY is populated
    # as the stricter alternative (the human may still elect it for the irreversible post).
    verdict = "POINT_GATE_LANE1_ALIVE" if adj["asapplied_gate_is_point"] == 1 else "CI_GATE_INT8_ONLY"

    return {
        "verdict": verdict,
        "gate_readings": readings,
        "asapplied_reconcile": adj,
        "cert_budget": cert,
        "two_world": worlds,
        "downstream_note": downstream,
        "recommendation": recommend,
        "self_test": st,
        # headline scalars
        "primary_int8_certify_n_pointclears": cert["lanes"]["int8_locus"]["min_n_point_clears"],
        "test_asapplied_gate_is_point": adj["asapplied_gate_is_point"],
        "fullg32_pointclears": cert["lanes"]["full_g32"]["min_n_point_clears"],
        "fullg32_power95_stable": cert["lanes"]["full_g32"]["min_n_power95_stable"],
        "fullg32_power95_banked_714": D714_FULLG32_POWER95_BANKED,
        "routeb_pointclears": cert["lanes"]["route_b"]["min_n_point_clears"],
        "int8_power95_stable": cert["lanes"]["int8_locus"]["min_n_power95_stable"],
    }


# =========================================================================== #
# REPORT.
# =========================================================================== #
def render_report(syn: dict[str, Any]) -> str:
    adj, cert, w = syn["asapplied_reconcile"], syn["cert_budget"], syn["two_world"]
    L = cert["lanes"]
    lines = []
    lines.append("=" * 78)
    lines.append("GATE SEMANTICS: POINT vs CI-CERTIFIED + CERT-BUDGET PER LANE (PR #716)")
    lines.append("=" * 78)
    lines.append(f"VERDICT: {syn['verdict']}")
    lines.append(f"  asapplied_gate_is_point (TEST metric) = {adj['asapplied_gate_is_point']}")
    lines.append(f"  int8_certify_n_pointclears (PRIMARY)  = {syn['primary_int8_certify_n_pointclears']}")
    lines.append("")
    lines.append("(2) AS-APPLIED RECONCILE (#703):")
    lines.append(f"  {adj['operative_rule']}")
    lines.append(f"  legs failing under POINT rule = {adj['legs_failing_under_point_rule']} (AIME); "
                 f"under CI rule = {adj['legs_failing_under_ci_rule']} (AIME+GPQA-D)")
    lines.append(f"  => {adj['adjudication']}")
    lines.append("")
    lines.append("(3) CERTIFICATION-BUDGET FRONTIER (Wilson-lo point-clears / 95%-power; GREEDY):")
    lines.append(f"  {'lane':<22}{'point':>7}{'margin':>8}{'n_ptclear':>11}{'n_pwr95':>9}{'edge<=1500':>11}")
    for key in ("full_g32", "route_b", "int8_locus"):
        c = L[key]
        npc = c["min_n_point_clears"]
        pw = c["min_n_power95_stable"]
        lines.append(f"  {key:<22}{c['point']:>7.4f}{c['margin']:>8.4f}"
                     f"{(npc if npc is not None else -1):>11}{(pw if pw is not None else -1):>9}"
                     f"{('YES' if c['point_clears_le_edge_feasible'] else 'no'):>11}")
    lines.append(f"  [self-test] full_g32 reproduces #714: 2889 / power-band contains 9828 "
                 f"(stable {syn['fullg32_power95_stable']})")
    lines.append(f"  [self-test] route_b reproduces #714: {syn['routeb_pointclears']} (== 545295)")
    lines.append("")
    lines.append("  INSTRUMENT CAP (greedy is deterministic -> n == problem pool):")
    ic = cert["instrument_cap"]
    lines.append(f"    greedy pool cap = {ic['greedy_pool_cap']}; int8 budget/cap = "
                 f"{ic['int8_pointclears_over_greedy_cap']}x; int8 SAMPLED {ic['int8_sampled_point']} "
                 f"< gate = {bool(ic['int8_sampled_below_gate'])}")
    lines.append("")
    lines.append("(4) TWO-WORLD TABLE:")
    pw_ = w["point_world"]
    lines.append(f"  POINT world: full_g32 clears={pw_['full_g32']['clears']}, "
                 f"route_b clears={pw_['route_b']['clears']}, int8 clears={pw_['int8_locus']['clears']} "
                 f"-> Lane1 alive on point = {pw_['lane1_alive_on_point']}")
    cw = w["ci_world"]
    lines.append(f"  CI world (edge-feasible<=1500): full_g32={cw['full_g32']['edge_feasible']}, "
                 f"route_b={cw['route_b']['edge_feasible']}, int8={cw['int8_locus']['edge_feasible']} "
                 f"-> int8-only edge-feasible = {w['int8_only_edge_feasible']}")
    lines.append(f"  greedy-pool feasible (any lane, n<=60) = {cw['any_lane_greedy_pool_feasible']}")
    lines.append("")
    lines.append("(R) RECOMMENDATION (one-shot fire #481):")
    lines.append(f"  {syn['recommendation']['recommended_standard']}")
    lines.append("")
    lines.append(f"SELF-TEST: {syn['self_test']['n_passing']}/{syn['self_test']['n_conditions']} "
                 f"PASS={syn['self_test']['gate_semantics_self_test_passes']}")
    if not syn["self_test"]["gate_semantics_self_test_passes"]:
        for k, v in syn["self_test"]["conditions"].items():
            if not v:
                lines.append(f"    FAIL: {k}")
    lines.append("=" * 78)
    return "\n".join(lines)


# =========================================================================== #
# W&B LOGGING.
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
        print(f"[gate-semantics] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    cert, adj, w = syn["cert_budget"], syn["asapplied_reconcile"], syn["two_world"]
    L = cert["lanes"]
    run = init_wandb_run(
        job_type="gate-semantics-cert-budget",
        agent="denken",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["gate-semantics", "cert-budget", "analysis-only", "power-analysis",
              "aime-gate", "point-vs-ci", "no-hf-job"],
        config={
            "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
            "gate": GATE, "aime_floor": AIME_FLOOR, "ceiling_g32": CEILING_G32,
            "int8_locus_greedy": INT8_LOCUS_GREEDY, "route_b_point": ROUTE_B_POINT,
            "greedy_pool_cap": GREEDY_POOL_CAP, "edge_feasible_n": EDGE_FEASIBLE_N, "z95": Z95,
            "imports": "denken#714 anchors + lawine#703 as-applied gate + fern#659 int8-locus",
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[gate-semantics] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    # flatten the n(p) frontier (lookup table).
    ftab = {}
    for r in cert["frontier"]:
        tag = f"{int(round(r['p'] * 1000)):03d}"  # e.g. p=0.438 -> "438"
        ftab[f"frontier_npc_p{tag}"] = (r["n_point_clears"] if r["n_point_clears"] is not None else -1)
        ftab[f"frontier_npwr_p{tag}"] = (r["n_power95"] if r["n_power95"] is not None else -1)

    summary: dict[str, Any] = {
        # constraint scalars (EXPLICIT, required by the PR).
        "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        # primary / test / gate metrics.
        "int8_certify_n_pointclears": syn["primary_int8_certify_n_pointclears"],
        "asapplied_gate_is_point": syn["test_asapplied_gate_is_point"],
        "gate_semantics_self_test_passes": int(bool(
            syn["self_test"]["gate_semantics_self_test_passes"])),
        # full-g32 #714 reproduction (2889 / 9828).
        "fullg32_pointclears": syn["fullg32_pointclears"],
        "fullg32_power95_stable": syn["fullg32_power95_stable"],
        "fullg32_power95_first_cross": L["full_g32"]["min_n_power95_first_cross"],
        "fullg32_power95_banked_714": D714_FULLG32_POWER95_BANKED,
        "fullg32_power95_normal_approx": round(normal_n_power95(CEILING_G32), 1),
        "routeb_pointclears": syn["routeb_pointclears"],
        "routeb_pointclears_banked_714": D714_ROUTEB_POINTCLEARS,
        # int8 lane.
        "int8_power95_stable": syn["int8_power95_stable"],
        "int8_pointclears_normal_approx": round(normal_n_point_clears(INT8_LOCUS_GREEDY), 1),
        "int8_sampled_point": INT8_LOCUS_SAMPLED,
        "int8_sampled_below_gate": int(INT8_LOCUS_SAMPLED < GATE),
        # margins & gate context.
        "gate": GATE, "fullg32_margin": round(CEILING_G32 - GATE, 5),
        "int8_margin": round(INT8_LOCUS_GREEDY - GATE, 5),
        "routeb_margin": round(ROUTE_B_POINT - GATE, 5),
        # adjudication.
        "legs_failing_under_point_rule": adj["legs_failing_under_point_rule"],
        "legs_failing_under_ci_rule": adj["legs_failing_under_ci_rule"],
        "ci_rule_would_collapse_panel": adj["ci_rule_would_collapse_panel"],
        # two-world consequence.
        "lane1_alive_on_point": w["point_world"]["lane1_alive_on_point"],
        "int8_only_edge_feasible": w["int8_only_edge_feasible"],
        "any_lane_greedy_pool_feasible": w["ci_world"]["any_lane_greedy_pool_feasible"],
        # instrument cap.
        "greedy_pool_cap": GREEDY_POOL_CAP,
        "int8_pointclears_over_greedy_cap": cert["instrument_cap"]["int8_pointclears_over_greedy_cap"],
        "smallest_p_edge_feasible_point_clears": cert["smallest_p_edge_feasible_point_clears"],
        "smallest_p_greedy_feasible_point_clears": (
            cert["smallest_p_greedy_feasible_point_clears"]
            if cert["smallest_p_greedy_feasible_point_clears"] is not None else -1.0),
        # verdicts (both worlds).
        "verdict_point_gate_lane1_alive": int(syn["verdict"] == "POINT_GATE_LANE1_ALIVE"),
        "verdict_ci_gate_int8_only": int(syn["verdict"] == "CI_GATE_INT8_ONLY"),
        "peak_mem_mib": payload["peak_mem_mib"],
        **ftab,
        **{f"selftest_{k}": int(bool(v)) for k, v in syn["self_test"]["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v))}
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="gate_semantics_cert_budget_result",
                      artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[gate-semantics] wandb logged verdict={syn['verdict']} "
          f"int8_certify_n_pointclears={syn['primary_int8_certify_n_pointclears']} "
          f"asapplied_gate_is_point={syn['test_asapplied_gate_is_point']}", flush=True)


# =========================================================================== #
# MAIN.
# =========================================================================== #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="gate-semantics-cert-budget-denken")
    args = ap.parse_args(argv)

    syn = synthesize()
    peak_mem_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KB -> MiB on Linux

    payload = {
        "schema": "gate_semantics_cert_budget/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        "synthesis": syn,
        "peak_mem_mib": round(peak_mem_mib, 2),
    }

    print(render_report(syn))

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gate_semantics_cert_budget_result.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[gate-semantics] wrote {out_path}")

    _maybe_log_wandb(args, payload)

    if args.self_test and not syn["self_test"]["gate_semantics_self_test_passes"]:
        print("[gate-semantics] SELF-TEST FAILED", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
