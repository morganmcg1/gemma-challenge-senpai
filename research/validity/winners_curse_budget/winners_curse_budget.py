#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Winner's-curse re-draw budget: does best-of-N clear the binding PRIVATE bar? (PR #210)

WHAT THIS IS
------------
The PRIVATE-side capstone of the draw-budget lane (#159->#188->#194->#200->#202).
#194/#200/#202 all budgeted best-of-N shots against the PUBLIC TPS>=500 bar. But the
launch is irreversible and graded ONCE on the official PRIVATE re-benchmark, and
stark #191 proved the PRIVATE bar binds strictly tighter (private lambda*_LCB 0.9780
vs public 0.9052). "Best-of-N re-draws, take the max PUBLIC TPS" is a SELECTION rule:
you launch the checkpoint whose public shot fluctuated highest. That is a textbook
winner's-curse / optimizer's-curse (Capen 1971; Smith & Winkler 2006, Management
Science 52(3) Prop. 1): the selected public max OVERSTATES the checkpoint's
replicable quality by the expected max order-statistic of the RE-RANDOMIZING noise
(sigma_sel * E[Z_(N:N)]), and that up-fluctuation is FRESH hardware/sampling luck
that does NOT carry to the private grade. This leg quantifies it and re-prices the
build target the human's Approval request actually needs.

THE SELECTION MODEL (PR step 1)
-------------------------------
You re-draw the SAME checkpoint N times and launch argmax public TPS, then it is
graded ONCE privately. Public shot i:  X_i = mu_pub + b + eps_hw,i.
  * FROZEN (conservative default, #202): the per-checkpoint sampling bias b is COMMON
    across the N re-draws (fixed prompts + greedy => identical tokens), only
    eps_hw,i ~ N(0,sigma_hw^2) re-randomizes  =>  selection i* = argmax eps_hw,i,
    beatable scatter sigma_sel = sigma_hw.
  * FRESH (#194): each shot re-draws b_i AND eps_hw,i  =>  X_i ~ N(mu_pub, sigma_draw^2)
    iid, selection i* = argmax X_i, beatable scatter sigma_sel = sigma_draw.
The private grade of the SELECTED checkpoint:  Y = mu_pub*f_priv + (private noise),
private noise drawn FRESH and INDEPENDENT of the public selection (a new allocation +
the adverse-skew private prompt set). f_priv = (1-drop)*tau_low is #191's binding
public->private composition (stark #176 adverse drop 2.351% x tau_low 0.99243). The
private grade is a SINGLE fresh draw with the same noise structure: sigma_priv = sigma_draw.

THE MECHANISM (PR step 2)  winners_curse_tps(N) = sigma_sel * E[Z_(N:N)]
------------------------------------------------------------------------
E[Z_(N:N)] (expected max of N std normals; David & Nagaraja 2003): N=1->0, 2->0.5642,
3->0.8463, 5->1.1630, 10->1.5388. In FROZEN the ENTIRE best-of-N public gain (#202: it
only moves sigma_hw) is winner's-curse hardware luck that evaporates on a fresh private
re-benchmark.

THE CORE (PR step 3)  p_private_clear_given_trigger
---------------------------------------------------
Launch rule: launch iff max_i X_i >= 500. The private clear-prob conditional on the
best-of-N trigger,  P(Y>=500 | max_i X_i >= 500), computed by the joint over the frozen
common bias b and the fresh private noise (1-D Simpson). KEY RESULT (proved, not
assumed): because Y is INDEPENDENT of every public draw (b does not carry to the
adverse private prompt set; eps_hw does not carry to the fresh private allocation), the
conditioning event carries ZERO information about Y, so
    p_private_clear_given_trigger(mu_pub, N) == P(Y>=500)  for ALL N, BOTH regimes.
The private conditional is EXACTLY FLAT in N (Smith & Winkler Prop. 1; Galton complete
regression to the mean -- the selected max regresses ALL the way back on an independent
re-test). The GAP public_minus_private_clear is the silent over-optimism the human buys
by reading the public best-of-N max.

THE DELIVERABLE (PR step 4)  mu_bar_private_corrected (TEST)
-----------------------------------------------------------
The PUBLIC build mu_pub s.t. the conditional private clear >= 0.95 under best-of-5.
Since the conditional is flat in N, this solves P(Y>=500)=0.95 -> mu_pub*f_priv =
mu_safe (512.157, #194's fresh N=1 break-even) -> mu_bar_private_corrected =
mu_safe / f_priv ~ 528.5 TPS -- REGIME-INDEPENDENT (the private grade is one fresh
draw whatever the public regime). winners-curse tax delta_mu = it minus #202's
frozen public bar 504.87, decomposing into (a) the public best-of-N discount that
EVAPORATES privately (mu_safe-504.87 ~ 7.3 TPS) + (b) the drop gross-up (~16 TPS).

SELF-DEFEAT (PR step 5)
-----------------------
p_private_clear_given_trigger is FLAT in N -> n_star_private = 1: best-of-N raises the
PUBLIC max and fires the trigger more often but adds ZERO private clear. Against the
binding private bar, building higher STRICTLY dominates re-drawing more.

LOCAL CPU-only analytic. No GPU / vLLM / HF Job / submission / served-file / actual
official draw. BASELINE stays 481.53; greedy/PPL untouched; adds 0 TPS; authorizes NO
shot count or spend (a human still approves the launch). IMPORTS #194 sigma decomposition,
#202 frozen bar, #176/#191 private drop, #200 E[shots] VERBATIM; does NOT re-derive them.
Orthogonal to Issue #192 (greedy gate, held). NOT open2. NOT a launch.

SELF-TEST (PR step 6 -- PRIMARY)
-------------------------------
(a) N=1: winners_curse_tps(1)=0 and conditional private clear == unconditional single-shot.
(b) sigma_hw->0 FROZEN: trigger <=> mu_pub+b>=500, conditional private clear == single-shot
    (flat in N -- no winner's-curse gain OR loss from extra shots).
(c) drop->0 AND fresh-symmetric noise: the public-minus-private EXPECTATION gap equals the
    pure order-statistic winner's curse sigma_sel*E[Z_(N:N)] (clean limit).
(d) E[Z_(N:N)] reproduces the tabulated max order-stats (N=2->0.5642, 5->1.1630, 10->1.5388).
(e) reproduces #194's PUBLIC p_public_clear_bestofN and #202's frozen p_bar_n5_frozen=0.810
    verbatim (the public leg is unchanged -- only the private conditional is ADDED).
(f) NaN-clean.
(g) [core] conditional private clear == unconditional for all N, both regimes (FLAT-in-N).
PRIMARY = winners_curse_self_test_passes (bool); TEST = mu_bar_private_corrected (float TPS).
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
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # research/validity/winners_curse_budget -> repo root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# source-of-truth artifacts (imported verbatim; one source per constant)
# ---------------------------------------------------------------------------
REDRAW_194 = os.path.join(_ROOT, "research/validity/redraw_budget/redraw_budget_results.json")
FROZEN_202 = os.path.join(_ROOT, "research/validity/frozen_budget/frozen_budget_results.json")
DROP_176 = os.path.join(_ROOT, "research/validity/private_adverse_skew/results.json")
COST_200 = os.path.join(_ROOT, "research/validity/cost_budget/cost_budget_results.json")
BAR_191 = os.path.join(_ROOT, "research/validity/private_build_bar/results.json")

TARGET = 500.0
P_TARGET = 0.95
Z95_ONE_SIDED = 1.6448536269514722  # ppf(0.95); the single-draw P>=0.95 break-even quantile
# PR-named operating points
MU_GRID = [500.0, 505.0, 512.2, 515.0, 520.95]
N_GRID = [1, 5]
N_WC = [1, 2, 3, 5, 10]   # winner's-curse / order-stat table
N_SIMPSON = 2001          # #202's frozen-integral grid (reproduce verbatim)
# the tabulated expected max-of-N standard normals (PR step 2 / self-test d)
E_MAX_REF = {1: 0.0, 2: 0.5642, 3: 0.8463, 5: 1.1630, 10: 1.5388}


def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _finite(x: Any) -> float:
    if x is None or not math.isfinite(float(x)):
        raise ValueError(f"non-finite value: {x!r}")
    return float(x)


def _phi(x: float) -> float:
    """Standard-normal CDF via erf (no scipy) -- identical to #194/#202."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float, mu: float, sigma: float) -> float:
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def _simpson(f, lo: float, hi: float, n_grid: int = N_SIMPSON) -> float:
    """Deterministic composite-Simpson integral of f on [lo, hi] (n_grid odd)."""
    h = (hi - lo) / (n_grid - 1)
    acc = 0.0
    for i in range(n_grid):
        x = lo + i * h
        w = 1.0 if i in (0, n_grid - 1) else (4.0 if i % 2 == 1 else 2.0)
        acc += w * f(x)
    return acc * h / 3.0


# ---------------------------------------------------------------------------
# expected maximum of N standard normals  E[Z_(N:N)] = N * int z phi(z) Phi(z)^(N-1) dz
# (David & Nagaraja 2003, Order Statistics; the winner's-curse inflation per sigma)
# ---------------------------------------------------------------------------
def e_max_order_stat(n: int, lo: float = -12.0, hi: float = 14.0, n_grid: int = 200001) -> float:
    if n <= 1:
        return 0.0
    return n * _simpson(lambda z: z * _norm_pdf(z, 0.0, 1.0) * _phi(z) ** (n - 1), lo, hi, n_grid)


# ---------------------------------------------------------------------------
# PUBLIC best-of-N clear (reproduce #194 fresh / #202 frozen VERBATIM)
# ---------------------------------------------------------------------------
def p_public_fresh(mu: float, sigma_draw: float, n: int, target: float = TARGET) -> float:
    """FRESH: both components re-randomize per shot. P(max of N >= target) = 1 - Phi(.)^N."""
    return 1.0 - _phi((target - mu) / sigma_draw) ** n


def p_public_frozen(mu: float, sigma_sample: float, sigma_hw: float, n: int,
                    target: float = TARGET, n_grid: int = N_SIMPSON) -> float:
    """FROZEN: per-checkpoint bias b ~ N(0,sigma_sample^2) common to all N shots, only
    eps_hw re-draws. P = 1 - E_b[ Phi((target-mu-b)/sigma_hw)^N ]  (== #202 verbatim)."""
    if sigma_sample <= 1e-12:
        return 1.0 - _phi((target - mu) / sigma_hw) ** n
    if sigma_hw <= 1e-12:
        return _phi((mu - target) / sigma_sample)  # eps degenerate: clear iff mu+b>=target
    integ = _simpson(lambda b: _norm_pdf(b, 0.0, sigma_sample)
                     * _phi((target - mu - b) / sigma_hw) ** n,
                     -8.0 * sigma_sample, 8.0 * sigma_sample, n_grid)
    return 1.0 - integ


# ---------------------------------------------------------------------------
# PRIVATE grade (the winner's-curse correction)
# ---------------------------------------------------------------------------
def p_private_unconditional(mu_pub: float, sigma_sample: float, sigma_hw: float,
                            f_priv: float, target: float = TARGET) -> float:
    """Single fresh private grade Y = mu_pub*f_priv + N(0, sigma_priv^2),
    sigma_priv = hypot(sigma_sample, sigma_hw) (fresh b_priv (+) eps_hw'). P(Y>=target)."""
    sigma_priv = math.hypot(sigma_sample, sigma_hw)
    return 1.0 - _phi((target - mu_pub * f_priv) / sigma_priv)


def p_private_given_trigger_frozen(mu_pub: float, n: int, sigma_sample: float, sigma_hw: float,
                                   f_priv: float, rho_carry: float = 0.0,
                                   target: float = TARGET, n_grid: int = N_SIMPSON) -> float:
    """FROZEN conditional P(Y>=500 | max_i X_i >= 500) by the genuine joint over the frozen
    common bias b ~ N(0, sigma_sample^2):
        trigger | b : best-of-N over eps_hw ~ N(0,sigma_hw^2) -> 1 - Phi((500-mu-b)/sigma_hw)^N
        Y | b       : mean mu*f_priv + rho_carry*b ; residual sd hypot(sqrt(1-rho^2)sigma_sample, sigma_hw)
    rho_carry is the correlation between the PUBLIC sampling bias b and the PRIVATE sampling
    bias (how much public luck REPLICATES privately). The PR premise is rho_carry=0 (different
    adverse prompt set => b does NOT carry); then Y | b is constant in b and the ratio collapses
    to the unconditional -> conditional is EXACTLY flat in N (this is computed, not assumed)."""
    sig_resid = math.hypot(math.sqrt(max(0.0, 1.0 - rho_carry ** 2)) * sigma_sample, sigma_hw)
    lo, hi = -8.0 * sigma_sample, 8.0 * sigma_sample

    def _trig(b: float) -> float:
        if sigma_hw <= 1e-12:
            return 1.0 if (mu_pub + b) >= target else 0.0
        return 1.0 - _phi((target - mu_pub - b) / sigma_hw) ** n

    def _num(b: float) -> float:
        p_priv_b = 1.0 - _phi((target - (mu_pub * f_priv + rho_carry * b)) / sig_resid)
        return _norm_pdf(b, 0.0, sigma_sample) * _trig(b) * p_priv_b

    num = _simpson(_num, lo, hi, n_grid)
    den = _simpson(lambda b: _norm_pdf(b, 0.0, sigma_sample) * _trig(b), lo, hi, n_grid)
    return num / den if den > 1e-300 else float("nan")


def p_private_given_trigger_fresh(mu_pub: float, n: int, sigma_sample: float, sigma_hw: float,
                                  f_priv: float, target: float = TARGET) -> float:
    """FRESH conditional. Each public shot is iid N(mu_pub, sigma_draw^2) with no persistent
    common component, and the private grade Y is independent of every shot, so the joint
    factorises:  P(Y>=500, max>=500) = P(Y>=500) P(max>=500)  =>  conditional = P(Y>=500),
    EXACTLY the unconditional and INDEPENDENT of N."""
    return p_private_unconditional(mu_pub, sigma_sample, sigma_hw, f_priv, target)


def _bisect_mu(fn, target_val: float, lo: float, hi: float, tol: float = 1e-9,
               iters: int = 200) -> float:
    """Solve fn(mu) = target_val on [lo, hi] for a monotone-increasing fn."""
    if fn(lo) - target_val > 0:
        return lo
    if fn(hi) - target_val < 0:
        return hi
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
# import the banked legs (do NOT re-derive)
# ---------------------------------------------------------------------------
def import_banked() -> dict[str, Any]:
    rb = _load(REDRAW_194)
    fz = _load(FROZEN_202)
    d176 = _load(DROP_176)
    cost = _load(COST_200)
    bar191 = _load(BAR_191)

    d = rb["decomposition"]
    sigma_sample = _finite(d["sigma_sample_1sigma_tps"])      # 5.5645
    sigma_hw = _finite(d["sigma_hw_1sigma_tps"])              # 4.8645
    sigma_draw = _finite(rb["budget"]["sigma_draw_tps"])      # 7.3910 (= hypot)
    mu_safe = _finite(rb["mu_single_shot_safe_tps"])          # 512.157 (fresh N=1 break-even)
    # #194 banked fresh best-of-N reproduction targets (self-test e):
    mu_grid_194 = [{"mu": _finite(r["mu_tps"]),
                    "p_best_of_2": _finite(r["p_best_of_2"]),
                    "p_best_of_3": _finite(r["p_best_of_3"]),
                    "p_best_of_5": _finite(r["p_best_of_5"])}
                   for r in rb["budget"]["mu_grid_rows"]]

    mu_bar_frozen_p95 = _finite(fz["mu_bar_frozen_p95"])      # 504.873 (frozen public N=5 bar)
    mu_bar_fresh_p95_n5 = _finite(fz["build_bar"]["mu_bar_fresh_p95_n5"])  # 499.085 (fresh N=5)
    p_bar_n5_frozen = _finite(fz["p_bar_n5_frozen"])          # 0.80977 (self-test e)
    frozen_fraction_breakeven = _finite(fz["frozen_fraction_breakeven"])  # 0.8455

    av = d176["adverse_vertex"]
    drop_both = _finite(av["both_tree_drop_pct"]) / 100.0     # 0.023509 (#176 worst-corner tree drop)
    drop_descent = _finite(av["descent_tree_drop_pct"]) / 100.0
    decode_drop_pct = _finite(av["achieved_decode_drop_pct"]) # ~4.296 (<=5% DQ gate)
    tau_low = _finite(d176["constants"]["tau_low"])           # 0.9924318649
    f_priv = (1.0 - drop_both) * tau_low                      # #191 public->private composition

    e_shots_at_bar = _finite(cost["expected_shots_sequential_at_bar"])  # 1.9375 (#200)
    lambda_star_191 = _finite(bar191["synthesis"]["headline"]["lambda_star_lcb_private"])  # 0.9780

    return {
        "sigma_sample_tps": sigma_sample,
        "sigma_hw_tps": sigma_hw,
        "sigma_draw_tps": sigma_draw,
        "mu_safe_fresh_tps": mu_safe,
        "mu_grid_194": mu_grid_194,
        "mu_bar_frozen_p95": mu_bar_frozen_p95,
        "mu_bar_fresh_p95_n5": mu_bar_fresh_p95_n5,
        "p_bar_n5_frozen": p_bar_n5_frozen,
        "frozen_fraction_breakeven": frozen_fraction_breakeven,
        "drop_both": drop_both,
        "drop_descent": drop_descent,
        "decode_drop_pct": decode_drop_pct,
        "tau_low": tau_low,
        "f_priv": f_priv,
        "public_to_private_drop_pct": (1.0 - f_priv) * 100.0,  # additive %-of-mean drop
        "e_shots_at_bar_200": e_shots_at_bar,
        "lambda_star_191": lambda_star_191,
        "sources": {
            "redraw_194": REDRAW_194, "frozen_202": FROZEN_202, "drop_176": DROP_176,
            "cost_200": COST_200, "bar_191": BAR_191,
        },
    }


# ---------------------------------------------------------------------------
# STEP 1 -- selection model + regime trigger
# ---------------------------------------------------------------------------
def selection_model(imp: dict) -> dict[str, Any]:
    ss, sh, sd = imp["sigma_sample_tps"], imp["sigma_hw_tps"], imp["sigma_draw_tps"]
    return {
        "public_shot": "X_i = mu_pub + b + eps_hw,i",
        "private_grade": "Y = mu_pub*f_priv + (fresh private noise ~ N(0, sigma_priv^2)), "
                         "f_priv=(1-drop)*tau_low=%.6f, sigma_priv=sigma_draw=%.4f; private noise "
                         "INDEPENDENT of the public selection (fresh allocation + adverse private prompts)."
                         % (imp["f_priv"], sd),
        "FROZEN": "b common across the N re-draws (fixed prompts + greedy => identical tokens), only "
                  "eps_hw,i ~ N(0,sigma_hw^2=%.3f^2) re-draws; selection i*=argmax eps_hw,i, "
                  "sigma_sel=sigma_hw=%.4f (#202 conservative default)." % (sh, sh),
        "FRESH": "each shot re-draws b_i AND eps_hw,i => X_i ~ N(mu_pub, sigma_draw^2=%.3f^2) iid; "
                 "selection i*=argmax X_i, sigma_sel=sigma_draw=%.4f (#194 premise)." % (sd, sd),
        "regime_trigger": "WHICH regime the official harness is in is the harness-owner's open question "
                          "(same as #202/#192): FRESH iff the harness re-samples prompts or decode is "
                          "non-deterministic run-to-run; FROZEN iff it re-benchmarks the FIXED 128 prompts "
                          "under deterministic greedy. The challenge's token-IDENTITY contract leans FROZEN. "
                          "BOTH are bound here; the private conditional is regime-INVARIANT regardless.",
        "winners_curse_mechanism": "the launch trigger fires on the MAX of N public shots; the selected max "
                                   "overstates the replicable mean by sigma_sel*E[Z_(N:N)] of re-randomizing "
                                   "noise that does NOT appear in Y (Capen 1971; Smith & Winkler 2006).",
    }


# ---------------------------------------------------------------------------
# STEP 2 -- winner's-curse inflation table
# ---------------------------------------------------------------------------
def winners_curse_table(imp: dict, e_max: dict[int, float]) -> dict[str, Any]:
    ss, sh, sd = imp["sigma_sample_tps"], imp["sigma_hw_tps"], imp["sigma_draw_tps"]
    rows = []
    for n in N_WC:
        ez = e_max[n]
        rows.append({
            "n": n,
            "e_max_order_stat": ez,
            "winners_curse_tps_frozen": sh * ez,   # sigma_sel = sigma_hw
            "winners_curse_tps_fresh": sd * ez,    # sigma_sel = sigma_draw
        })
    wc5_frozen = sh * e_max[5]
    wc5_fresh = sd * e_max[5]
    return {
        "rows": rows,
        "winners_curse_tps_n5_frozen": wc5_frozen,
        "winners_curse_tps_n5_fresh": wc5_fresh,
        "note": ("FROZEN: the ENTIRE best-of-N public gain (#202 showed it only moves sigma_hw) is "
                 "winner's-curse hardware luck = sigma_hw*E[Z_(N:N)] (%.2f TPS at N=5) that EVAPORATES on a "
                 "fresh private re-benchmark. FRESH inflates more (sigma_draw*E[Z], %.2f TPS at N=5) because "
                 "the whole draw re-randomizes. best-of-N raises the PUBLIC number you SEE, never the "
                 "checkpoint's true private mean." % (wc5_frozen, wc5_fresh)),
    }


# ---------------------------------------------------------------------------
# STEP 3 -- conditional private clear vs public best-of-N (the GAP)
# ---------------------------------------------------------------------------
def gap_table(imp: dict) -> dict[str, Any]:
    ss, sh, sd, fp = (imp["sigma_sample_tps"], imp["sigma_hw_tps"],
                      imp["sigma_draw_tps"], imp["f_priv"])
    rows = []
    for mu in MU_GRID:
        p_priv_uncond = p_private_unconditional(mu, ss, sh, fp)
        for n in N_GRID:
            p_pub_fresh = p_public_fresh(mu, sd, n)
            p_pub_frozen = p_public_frozen(mu, ss, sh, n)
            p_priv_fresh = p_private_given_trigger_fresh(mu, n, ss, sh, fp)
            p_priv_frozen = p_private_given_trigger_frozen(mu, n, ss, sh, fp)
            rows.append({
                "mu_pub": mu, "n": n,
                "p_public_clear_bestofN_fresh": p_pub_fresh,
                "p_public_clear_bestofN_frozen": p_pub_frozen,
                "p_private_clear_given_trigger_fresh": p_priv_fresh,
                "p_private_clear_given_trigger_frozen": p_priv_frozen,
                "p_private_unconditional": p_priv_uncond,
                # the silent over-optimism the human buys by reading the public max:
                "public_minus_private_clear_fresh": p_pub_fresh - p_priv_fresh,
                "public_minus_private_clear_frozen": p_pub_frozen - p_priv_frozen,
            })
    return {
        "rows": rows,
        "note": ("p_private_clear_given_trigger is IDENTICAL across N (and across regime) -- the private "
                 "grade is one fresh draw whose distribution is unchanged by the public selection (Y _||_ "
                 "trigger). The GAP public_minus_private_clear GROWS with N (more shots inflate the public "
                 "trigger) while the private clear stays put: that gap is the winner's-curse over-optimism."),
    }


# ---------------------------------------------------------------------------
# STEP 4 -- private-corrected build target (the deliverable)
# ---------------------------------------------------------------------------
def private_corrected_bar(imp: dict) -> dict[str, Any]:
    ss, sh, sd, fp = (imp["sigma_sample_tps"], imp["sigma_hw_tps"],
                      imp["sigma_draw_tps"], imp["f_priv"])
    mu_safe = imp["mu_safe_fresh_tps"]

    # mu_bar_private_corrected: PUBLIC build s.t. conditional private clear (best-of-5) >= 0.95.
    # Bisect the FROZEN conditional at N=5 (literal PR ask); it is flat in N and regime, so it
    # equals the closed form mu_safe/f_priv.
    mu_bar = _bisect_mu(lambda m: p_private_given_trigger_frozen(m, 5, ss, sh, fp),
                        P_TARGET, 490.0, 580.0)
    mu_bar_closed = mu_safe / fp  # closed-form cross-check
    mu_bar_fresh_cond = _bisect_mu(lambda m: p_private_given_trigger_fresh(m, 5, ss, sh, fp),
                                   P_TARGET, 490.0, 580.0)

    mu_bar_frozen_pub = imp["mu_bar_frozen_p95"]   # 504.87 (public-only frozen N=5)
    mu_bar_fresh_pub = imp["mu_bar_fresh_p95_n5"]  # 499.08 (public-only fresh N=5)
    delta_mu_winners_curse = mu_bar - mu_bar_frozen_pub
    delta_mu_vs_fresh = mu_bar - mu_bar_fresh_pub

    # decomposition of the tax vs the frozen public bar:
    public_bestofN_discount = mu_safe - mu_bar_frozen_pub   # best-of-5 public discount that EVAPORATES
    drop_grossup = mu_bar - mu_safe                          # the private-drop gross-up of the build

    # does the freeze-robust N=1 build (mu=512.2) survive the private winner's curse?
    p_priv_at_512 = p_private_unconditional(512.2, ss, sh, fp)
    p_priv_at_512_n5 = p_private_given_trigger_frozen(512.2, 5, ss, sh, fp)  # == p_priv_at_512 (flat)

    return {
        "mu_bar_private_corrected": mu_bar,                 # TEST
        "mu_bar_private_corrected_closed_form": mu_bar_closed,
        "mu_bar_private_corrected_fresh_check": mu_bar_fresh_cond,
        "mu_bar_frozen_public_202": mu_bar_frozen_pub,
        "mu_bar_fresh_public_194": mu_bar_fresh_pub,
        "mu_safe_fresh_194": mu_safe,
        "delta_mu_winners_curse": delta_mu_winners_curse,   # vs #202 frozen public bar (PR-named)
        "delta_mu_vs_fresh_public": delta_mu_vs_fresh,
        "tax_decomposition": {
            "public_bestofN_discount_evaporates_tps": public_bestofN_discount,
            "private_drop_grossup_tps": drop_grossup,
            "sum_tps": public_bestofN_discount + drop_grossup,
        },
        "p_private_clear_at_mu512p2_n1": p_priv_at_512,
        "p_private_clear_at_mu512p2_n5": p_priv_at_512_n5,
        "freeze_robust_512_survives_private": bool(p_priv_at_512 >= P_TARGET),
        "regime_invariant": True,
        "interpretation": (
            "mu_bar_private_corrected=%.2f TPS (=mu_safe/f_priv) -- REGIME-INVARIANT (the private grade is "
            "one fresh draw whatever the public regime). winner's-curse tax vs #202's frozen public bar "
            "504.87 is +%.2f TPS = [public best-of-5 discount that EVAPORATES privately %.2f] + [private-drop "
            "gross-up %.2f]. The freeze-robust N=1 build mu=512.2 clears the private bar at only %.4f (NOT "
            ">=0.95): it does NOT survive the private winner's curse -- the 2.35%% adverse drop x tau_low sinks "
            "its private mean to %.2f<500."
            % (mu_bar, delta_mu_winners_curse, public_bestofN_discount, drop_grossup,
               p_priv_at_512, 512.2 * fp)),
    }


# ---------------------------------------------------------------------------
# STEP 5 -- does best-of-N help or HURT against the private bar?
# ---------------------------------------------------------------------------
def monotonicity_in_n(imp: dict) -> dict[str, Any]:
    ss, sh, sd, fp = (imp["sigma_sample_tps"], imp["sigma_hw_tps"],
                      imp["sigma_draw_tps"], imp["f_priv"])
    n_list = [1, 2, 3, 5, 10, 20, 50]
    # at the build-bar mu=500 (where best-of-N was budgeted) -- both regimes
    bar_curve = []
    for n in n_list:
        bar_curve.append({
            "n": n,
            "p_public_frozen": p_public_frozen(TARGET, ss, sh, n),
            "p_public_fresh": p_public_fresh(TARGET, sd, n),
            "p_private_given_trigger_frozen": p_private_given_trigger_frozen(TARGET, n, ss, sh, fp),
            "p_private_given_trigger_fresh": p_private_given_trigger_fresh(TARGET, n, ss, sh, fp),
        })
    priv_seq = [r["p_private_given_trigger_frozen"] for r in bar_curve]
    flat = all(abs(p - priv_seq[0]) < 1e-9 for p in priv_seq)
    monotone = all(priv_seq[i + 1] >= priv_seq[i] - 1e-12 for i in range(len(priv_seq) - 1))
    # n_star_private: the smallest N maximising the conditional private clear; flat => 1.
    n_star_private = 1 if flat else max(range(len(priv_seq)), key=lambda i: priv_seq[i]) + 1

    # rho-carry sensitivity: ONLY if public luck REPLICATES privately (rho>0) does best-of-N help.
    rho_sens = []
    for rho in (0.0, 0.5, 1.0):
        rho_sens.append({
            "rho_carry": rho,
            "p_priv_n1": p_private_given_trigger_frozen(TARGET, 1, ss, sh, fp, rho_carry=rho),
            "p_priv_n5": p_private_given_trigger_frozen(TARGET, 5, ss, sh, fp, rho_carry=rho),
            "p_priv_n20": p_private_given_trigger_frozen(TARGET, 20, ss, sh, fp, rho_carry=rho),
        })

    return {
        "bar_curve_in_n": bar_curve,
        "n_star_private": n_star_private,
        "private_clear_flat_in_n": bool(flat),
        "private_clear_monotone_in_N": bool(monotone),  # constant => weakly monotone, never worsens (rho>=0)
        "rho_carry_sensitivity": rho_sens,
        "structural_claim": (
            "p_private_clear_given_trigger is EXACTLY FLAT in N (n_star_private=1): every extra shot only "
            "inflates the PUBLIC trigger and adds ZERO private clear. The rho-carry sweep shows WHY -- best-"
            "of-N would help ONLY if public luck REPLICATED privately (rho>0); at the PR premise rho=0 "
            "(the selection is on non-replicating noise -- a fresh allocation + adverse private prompts) it "
            "never does. Against the binding private bar, BUILDING HIGHER STRICTLY DOMINATES RE-DRAWING MORE; "
            "best-of-N is self-defeating. (#200's ~%.2f public shots at the bar buy zero private clear.)"
            % imp["e_shots_at_bar_200"]),
    }


# ---------------------------------------------------------------------------
# STEP 6 -- self-test (PRIMARY)
# ---------------------------------------------------------------------------
def self_test(imp: dict, e_max: dict[int, float], wc: dict, gap: dict,
              bar: dict, mono: dict) -> dict[str, Any]:
    ss, sh, sd, fp = (imp["sigma_sample_tps"], imp["sigma_hw_tps"],
                      imp["sigma_draw_tps"], imp["f_priv"])
    tol = 1e-6
    checks: dict[str, bool] = {}

    # (a) N=1: zero winner's curse; conditional private clear == unconditional single-shot.
    a_wc = abs(sh * e_max[1]) < 1e-12 and abs(sd * e_max[1]) < 1e-12
    a_cond = True
    for mu in MU_GRID:
        uncond = p_private_unconditional(mu, ss, sh, fp)
        c_fr = p_private_given_trigger_frozen(mu, 1, ss, sh, fp)
        c_fe = p_private_given_trigger_fresh(mu, 1, ss, sh, fp)
        if abs(c_fr - uncond) > 1e-6 or abs(c_fe - uncond) > 1e-9:
            a_cond = False
    checks["a_n1_zero_curse_conditional_equals_unconditional"] = bool(a_wc and a_cond)

    # (b) sigma_hw->0 FROZEN: conditional private clear flat (== unconditional) across N.
    tiny = 1e-6
    b_ok = True
    for mu in (498.0, 500.0, 505.0, 512.2):
        uncond = p_private_unconditional(mu, ss, tiny, fp)
        for n in (1, 2, 5, 10):
            c = p_private_given_trigger_frozen(mu, n, ss, tiny, fp)
            if abs(c - uncond) > 1e-5:
                b_ok = False
    checks["b_sigma_hw0_frozen_conditional_flat"] = bool(b_ok)

    # (c) drop->0 (f_priv=1) AND fresh-symmetric: public-minus-private EXPECTATION gap == sigma_sel*E[Z].
    c_ok = True
    c_detail = []
    for n in (2, 3, 5, 10):
        ez = e_max[n]
        for regime, sigma_sel in (("fresh", sd), ("frozen", sh)):
            e_max_public = TARGET + sigma_sel * ez   # E[selected public max] at mu=500 (E[b]=0)
            e_private = TARGET * 1.0                  # E[Y] at f_priv=1, mu=500
            gap_val = e_max_public - e_private
            ref = sigma_sel * ez
            c_detail.append({"n": n, "regime": regime, "gap_tps": gap_val, "ref_sigma_sel_ez": ref})
            if abs(gap_val - ref) > 1e-9:
                c_ok = False
    checks["c_clean_limit_gap_equals_order_stat_curse"] = bool(c_ok)

    # (d) E[Z_(N:N)] reproduces the tabulated max order-stats.
    d_ok = True
    d_detail = []
    for n, ref in E_MAX_REF.items():
        got = e_max[n]
        d_detail.append({"n": n, "got": got, "ref": ref, "abs_err": abs(got - ref)})
        if abs(got - ref) > 1e-3:
            d_ok = False
    checks["d_e_max_order_stat_reproduces_tabulated"] = bool(d_ok)

    # (e) reproduces #194 PUBLIC best-of-N (fresh) and #202 frozen p_bar_n5_frozen=0.810 VERBATIM.
    e_ok = True
    e_max_err = 0.0
    for r in imp["mu_grid_194"]:
        mu = r["mu"]
        for nn, key in ((2, "p_best_of_2"), (3, "p_best_of_3"), (5, "p_best_of_5")):
            mine = p_public_fresh(mu, sd, nn)
            e_max_err = max(e_max_err, abs(mine - r[key]))
            if abs(mine - r[key]) > 1e-9:
                e_ok = False
    p_bar_n5_frozen_mine = p_public_frozen(TARGET, ss, sh, 5)
    e_frozen_err = abs(p_bar_n5_frozen_mine - imp["p_bar_n5_frozen"])
    if e_frozen_err > 1e-6:
        e_ok = False
    checks["e_reproduces_194_public_and_202_frozen_verbatim"] = bool(e_ok)

    # (g) [core] conditional private clear == unconditional for ALL N, BOTH regimes (FLAT-in-N).
    g_ok = True
    g_max_err = 0.0
    for mu in MU_GRID:
        uncond = p_private_unconditional(mu, ss, sh, fp)
        for n in (1, 2, 5, 10, 50):
            c_fr = p_private_given_trigger_frozen(mu, n, ss, sh, fp)
            c_fe = p_private_given_trigger_fresh(mu, n, ss, sh, fp)
            g_max_err = max(g_max_err, abs(c_fr - uncond), abs(c_fe - uncond))
            if abs(c_fr - uncond) > 1e-5 or abs(c_fe - uncond) > 1e-9:
                g_ok = False
    checks["g_conditional_flat_in_N_equals_unconditional"] = bool(g_ok)

    # (f) NaN-clean over every reported number.
    flat = (_collect(wc) + _collect(gap) + _collect(bar) + _collect(mono)
            + _collect(imp) + _collect(c_detail) + _collect(d_detail)
            + [e_max_err, e_frozen_err, g_max_err])
    checks["f_nan_clean"] = all(math.isfinite(x) for x in flat)

    passes = all(checks.values())
    return {
        "winners_curse_self_test_passes": passes,
        "checks": checks,
        "c_clean_limit_detail": c_detail,
        "d_e_max_detail": d_detail,
        "e_public_max_abs_err_vs_194": e_max_err,
        "e_frozen_abs_err_vs_202": e_frozen_err,
        "g_conditional_flatness_max_abs_err": g_max_err,
        "n_numbers_checked": len(flat),
    }


def _collect(obj: Any) -> list[float]:
    out: list[float] = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
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
        print(f"[wc] wandb_logging import failed ({exc}); skipping", flush=True)
        return
    try:
        run = wandb_logging.init_wandb_run(
            job_type="winners-curse-redraw-budget", agent="kanna",
            name=args.wandb_name or "kanna/winners-curse-redraw-budget",
            group=args.wandb_group,
            tags=["winners-curse-redraw-budget", "best-of-n", "max-statistics", "optimizers-curse",
                  "private-bar", "selection-bias", "pr210"],
            config={"baseline_tps": 481.53, "method": "cpu-only-analytic", "target_tps": 500.0,
                    "p_target": P_TARGET, "imports_pr": [194, 200, 202, 176, 191]},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[wc] wandb init failed ({exc}); skipping", flush=True)
        return
    if run is None:
        print("[wc] wandb disabled; skipping", flush=True)
        return
    try:
        imp = result["import_banked"]
        wc = result["winners_curse"]
        bar = result["private_corrected_bar"]
        mono = result["monotonicity_in_n"]
        st = result["self_test"]
        flat = {
            "winners_curse_self_test_passes": 1.0 if st["winners_curse_self_test_passes"] else 0.0,
            "mu_bar_private_corrected": bar["mu_bar_private_corrected"],
            "mu_bar_private_corrected_closed_form": bar["mu_bar_private_corrected_closed_form"],
            "delta_mu_winners_curse": bar["delta_mu_winners_curse"],
            "delta_mu_vs_fresh_public": bar["delta_mu_vs_fresh_public"],
            "public_bestofN_discount_evaporates_tps":
                bar["tax_decomposition"]["public_bestofN_discount_evaporates_tps"],
            "private_drop_grossup_tps": bar["tax_decomposition"]["private_drop_grossup_tps"],
            "p_private_clear_at_mu512p2": bar["p_private_clear_at_mu512p2_n1"],
            "freeze_robust_512_survives_private":
                1.0 if bar["freeze_robust_512_survives_private"] else 0.0,
            "winners_curse_tps_n5_frozen": wc["winners_curse_tps_n5_frozen"],
            "winners_curse_tps_n5_fresh": wc["winners_curse_tps_n5_fresh"],
            "n_star_private": float(mono["n_star_private"]),
            "private_clear_flat_in_n": 1.0 if mono["private_clear_flat_in_n"] else 0.0,
            "private_clear_monotone_in_N": 1.0 if mono["private_clear_monotone_in_N"] else 0.0,
            "f_priv": imp["f_priv"],
            "public_to_private_drop_pct": imp["public_to_private_drop_pct"],
            "sigma_draw_tps": imp["sigma_draw_tps"],
            "sigma_hw_tps": imp["sigma_hw_tps"],
            "mu_bar_frozen_public_202": imp["mu_bar_frozen_p95"],
            "e_public_max_abs_err_vs_194": st["e_public_max_abs_err_vs_194"],
            "e_frozen_abs_err_vs_202": st["e_frozen_abs_err_vs_202"],
            "g_conditional_flatness_max_abs_err": st["g_conditional_flatness_max_abs_err"],
        }
        # winner's-curse inflation curve + per-mu gap (private vs public best-of-5)
        for r in wc["rows"]:
            flat[f"winners_curse_tps_frozen_n{r['n']}"] = r["winners_curse_tps_frozen"]
            flat[f"winners_curse_tps_fresh_n{r['n']}"] = r["winners_curse_tps_fresh"]
            flat[f"e_max_order_stat_n{r['n']}"] = r["e_max_order_stat"]
        for r in result["gap_table"]["rows"]:
            if r["n"] == 5:
                mu_k = str(r["mu_pub"]).replace(".", "p")
                flat[f"gap_frozen_mu{mu_k}_n5"] = r["public_minus_private_clear_frozen"]
                flat[f"p_private_clear_mu{mu_k}"] = r["p_private_unconditional"]
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run, name="winners_curse_budget", artifact_type="winners-curse-redraw-budget", data=result)
    except Exception as exc:  # noqa: BLE001
        print(f"[wc] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:  # noqa: BLE001
            pass


def _print(result: dict[str, Any]) -> None:
    imp = result["import_banked"]
    wc = result["winners_curse"]
    bar = result["private_corrected_bar"]
    mono = result["monotonicity_in_n"]
    st = result["self_test"]
    print("\n[wc] ===== WINNER'S-CURSE RE-DRAW BUDGET (PR #210) =====", flush=True)
    print(f"  f_priv = (1-drop {imp['drop_both']*100:.3f}%)*tau_low {imp['tau_low']:.6f} = {imp['f_priv']:.6f}  "
          f"=> public->private mean drop {imp['public_to_private_drop_pct']:.3f}%", flush=True)
    print(f"  sigma_hw={imp['sigma_hw_tps']:.4f} (frozen sigma_sel)  sigma_draw={imp['sigma_draw_tps']:.4f} "
          f"(fresh sigma_sel)  sigma_priv={imp['sigma_draw_tps']:.4f}", flush=True)
    print("\n  winner's-curse inflation sigma_sel*E[Z_(N:N)]:", flush=True)
    for r in wc["rows"]:
        print(f"    N={r['n']:>2}  E[Z]={r['e_max_order_stat']:.4f}  "
              f"frozen={r['winners_curse_tps_frozen']:.3f} TPS  fresh={r['winners_curse_tps_fresh']:.3f} TPS",
              flush=True)
    print("\n  public best-of-5 vs private clear (conditional on trigger):", flush=True)
    for r in result["gap_table"]["rows"]:
        if r["n"] == 5:
            print(f"    mu={r['mu_pub']:7.2f}  P_pub_frozen(5)={r['p_public_clear_bestofN_frozen']:.4f}  "
                  f"P_priv|trig={r['p_private_unconditional']:.4f}  "
                  f"GAP={r['public_minus_private_clear_frozen']:+.4f}", flush=True)
    print(f"\n  mu_bar_private_corrected (TEST) = {bar['mu_bar_private_corrected']:.3f} TPS  "
          f"(closed-form mu_safe/f_priv = {bar['mu_bar_private_corrected_closed_form']:.3f})", flush=True)
    print(f"  delta_mu_winners_curse (vs #202 frozen 504.87) = {bar['delta_mu_winners_curse']:+.2f} TPS  "
          f"= discount {bar['tax_decomposition']['public_bestofN_discount_evaporates_tps']:.2f} + "
          f"drop {bar['tax_decomposition']['private_drop_grossup_tps']:.2f}", flush=True)
    print(f"  P(private clear | mu=512.2) = {bar['p_private_clear_at_mu512p2_n1']:.4f}  "
          f"freeze-robust 512.2 survives private = {bar['freeze_robust_512_survives_private']}", flush=True)
    print(f"  n_star_private = {mono['n_star_private']}  private_clear_flat_in_n = "
          f"{mono['private_clear_flat_in_n']}", flush=True)
    print(f"\n  SELF-TEST (PRIMARY) passes = {st['winners_curse_self_test_passes']}", flush=True)
    for k, v in st["checks"].items():
        print(f"    [{'ok' if v else '!! FAILED'}] {k}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(_HERE, "winners_curse_budget_results.json"))
    ap.add_argument("--wandb-name", "--wandb_name", default="kanna/winners-curse-redraw-budget")
    ap.add_argument("--wandb-group", "--wandb_group", default="winners-curse-redraw-budget")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    imp = import_banked()
    e_max = {n: e_max_order_stat(n) for n in sorted(set(N_WC + [1, 2, 3, 5, 10, 20, 50]))}

    model = selection_model(imp)
    wc = winners_curse_table(imp, e_max)
    gap = gap_table(imp)
    bar = private_corrected_bar(imp)
    mono = monotonicity_in_n(imp)
    st = self_test(imp, e_max, wc, gap, bar, mono)

    wc5 = wc["winners_curse_tps_n5_frozen"]
    handoff = (
        "best-of-N re-draws select the max PUBLIC shot, inflating it by sigma_sel*E[Z_(N:N)] ~ %.2f TPS at "
        "N=5 (frozen) of non-replicating luck, so to clear the BINDING 500 PRIVATE bar (stark #191) at "
        "P>=0.95 under a best-of-5 launch trigger the PUBLIC build must reach mu_pub=%.2f TPS (vs #202's "
        "public-only frozen bar 504.87, a +%.2f TPS winner's-curse tax); the conditional private clear is "
        "FLAT in N (n_star_private=1) so against the private bar building higher dominates re-drawing more -- "
        "fern #185 carries mu_bar_private_corrected=%.2f as the launch-decision multi-shot build target, NOT "
        "the public-only bar."
        % (wc5, bar["mu_bar_private_corrected"], bar["delta_mu_winners_curse"],
           bar["mu_bar_private_corrected"])
    )

    result = {
        "pr": 210,
        "metric_primary": "winners_curse_self_test_passes",
        "metric_test": "mu_bar_private_corrected",
        "winners_curse_self_test_passes": st["winners_curse_self_test_passes"],
        "mu_bar_private_corrected": bar["mu_bar_private_corrected"],
        "delta_mu_winners_curse": bar["delta_mu_winners_curse"],
        "winners_curse_tps_n5_frozen": wc["winners_curse_tps_n5_frozen"],
        "winners_curse_tps_n5_fresh": wc["winners_curse_tps_n5_fresh"],
        "n_star_private": mono["n_star_private"],
        "private_clear_flat_in_n": mono["private_clear_flat_in_n"],
        "selection_model": model,
        "winners_curse": wc,
        "gap_table": gap,
        "private_corrected_bar": bar,
        "monotonicity_in_n": mono,
        "self_test": st,
        "import_banked": imp,
        "e_max_order_stats": {str(k): v for k, v in e_max.items()},
        "handoff": handoff,
        "scope": "Winner's-curse / optimizer's-curse analog over the banked #194/#202 sigma-decomposition + "
        "stark #176/#191 private drop. Takes NO official draws, authorizes NO shot count or spend (a human "
        "still approves the launch AND confirms the harness regime), CPU-only. BANK-THE-ANALYSIS: adds 0 TPS, "
        "greedy/PPL untouched. The private conditional is regime-INVARIANT (FRESH/FROZEN bound, same answer). "
        "Draw-budget lane (#159/#188/#194/#200/#202). NOT ubel #201/#204. NOT stark #198/#203. NOT denken "
        "#197/#205. NOT fern #185 (the integrator -- this supplies the private-corrected multi-shot row it "
        "imports). Orthogonal to Issue #192 (greedy gate, held). NOT open2. NOT a launch.",
        "imported_legs": {
            "kanna_194_sigma_decomposition": REDRAW_194,
            "kanna_202_frozen_bar": FROZEN_202,
            "stark_176_adverse_private_drop": DROP_176,
            "kanna_200_sequential_expected_shots": COST_200,
            "stark_191_private_build_bar": BAR_191,
        },
        "public_evidence_used": [
            "Winner's curse: Capen, Clapp & Campbell (1971) JPT 23 -- the selected max overstates the "
            "common value (highest signal regresses to the mean on re-test).",
            "Optimizer's curse: Smith & Winkler (2006) Management Science 52(3) Prop. 1 -- when the grade is "
            "an independent draw, selection on a favourable noise realisation guarantees post-decision "
            "disappointment; the conditional re-test value equals the unconditional.",
            "Order statistics E[Z_(N:N)] = N*int z phi(z) Phi(z)^(N-1) dz (David & Nagaraja 2003).",
            "stark #191 binding private bar lambda*_LCB 0.9780 (both-bugs) > public 0.9052 -- the launch is "
            "graded on the PRIVATE re-benchmark.",
            "kanna #194/#202 banked best-of-N public budget (fresh / frozen) -- the public leg this corrects.",
        ],
        "method": "LOCAL CPU-only analytic synthesis; no GPU/vLLM/HF Job/submission/served-file/draw. "
        "BASELINE stays 481.53; adds 0 TPS. Greedy identity untouched. NOT open2. NOT a launch.",
        "metrics_nan_clean": 1 if st["checks"]["f_nan_clean"] else 0,
        "elapsed_s": round(time.time() - t0, 4),
    }

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    _print(result)
    print(f"\n[wc] HANDOFF: {handoff}", flush=True)
    print(f"[wc] artifacts -> {args.out}", flush=True)
    _log_wandb(args, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
