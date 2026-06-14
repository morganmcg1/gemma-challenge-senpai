#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Frozen-sampling re-draw budget: does best-of-N beat down all of sigma_draw, or
only sigma_hw? -- pin the conservative-regime build bar (PR #202).

WHAT THIS IS
------------
The load-bearing stress-test of kanna #194's (`mxm5q63j`, MERGED) official re-draw
budget. #194 budgeted N=5 official shots at the build-bar mu=500 for P(clear500)>=0.95
under the assumption that EVERY re-draw re-randomizes ALL of sigma_draw=7.391 TPS
(sigma_sample 5.564 iid (+) sigma_hw 4.864). #194 itself flagged -- twice (#194 5d,
#200 honest-scope) -- that this is the load-bearing assumption nobody has pinned: if
the official harness re-benchmarks the FIXED 128 public prompts under deterministic
greedy decode, then the finite-sample scatter sigma_sample is a FROZEN per-checkpoint
bias that best-of-N CANNOT beat down -- only sigma_hw (fresh HF-Job allocation)
re-randomizes per shot. This leg models BOTH regimes over #194's banked curve and
pins the conservative (FROZEN) build bar the human must clear.

THE TWO-REGIME MODEL (PR step 1)
--------------------------------
sigma_draw^2 = sigma_sample^2 + sigma_hw^2   (5.564^2 + 4.864^2 = 7.391^2).
  * FRESH  (#194 premise): each official shot draws X_n = mu + delta_n,
    delta_n ~ N(0, sigma_draw^2) iid -> best-of-N beats down the FULL sigma_draw.
    P_fresh(mu,N) = 1 - Phi((500-mu)/sigma_draw)^N.
  * FROZEN (#194 5d risk): a single per-checkpoint bias b ~ N(0, sigma_sample^2)
    is FIXED across re-draws (prompts fixed + greedy deterministic), only
    eps_hw,n ~ N(0, sigma_hw^2) re-randomizes per allocation:
    X_n = mu + b + eps_hw,n -> best-of-N beats down only sigma_hw (66% of variance).
    P_frozen(mu,N) = 1 - E_b[ Phi((500-mu-b)/sigma_hw)^N ]  (1-D Gauss-Hermite/Simpson).
WHICH regime the official harness is in is the harness-owner's open question (like
#192 enforcement). The challenge contract (fixed 128 prompts + greedy token-identity)
makes FROZEN the MORE PLAUSIBLE regime -- under deterministic greedy the same 128
prompts emit the same token counts every run, so sigma_sample does NOT re-randomize;
only the HF-Job timing (sigma_hw) does. We bound BOTH and flag FROZEN as the
conservative default; the human confirms the harness behavior before budgeting N.

THE DELIVERABLES (PR steps 2-4)
-------------------------------
  step 2  P_frozen(mu,N) for N in {1,2,5,10} at mu in {500,505,512.2,515,520.95};
          saturation N->inf (reported-max -> 1, but only via hardware luck; the
          OPERATIONALLY-safe ceiling is the sigma_sample-governed genuine-pass rate
          Phi((mu-500)/sigma_sample)); contrast vs P_fresh.
  step 3  mu_bar_frozen_p95 (TEST): build mu s.t. FROZEN best-of-5 hits P>=0.95;
          n_shots_frozen_at_512 (min N at mu=512.2, vs #194's N=1);
          delta_mu_frozen = mu_bar_frozen_p95 - mu_safe_fresh(512.157);
          p_bar_n5_frozen = P_frozen(500,5) (expect ~0.81 vs fresh 0.969).
  step 4  (a) frozen_fraction_breakeven f*: fraction of sigma_sample that must
              re-randomize (sigma_beatable^2 = sigma_hw^2 + f*sigma_sample^2) for
              N=5 to hit P>=0.95 -- at the bar mu=500 AND at the safe build mu=512.2.
          (b) p95_shots_at_bar: 95th-pct shots-paid under sequential early-stop at
              the bar, in both regimes (capped at N_max=5 and uncapped tail).

SELF-TEST (PR step 5 -- PRIMARY)
-------------------------------
(a) sigma_sample->0 (all variance fresh): frozen == fresh exactly, reproduces #194's
    N*(mu) at every mu;
(b) sigma_hw->0 (all variance frozen): best-of-N gives ZERO improvement
    (P_frozen(mu,N) == P_frozen(mu,1) for all N) and saturation = Phi((mu-500)/sigma_sample);
(c) monotone weakness: P_frozen(mu,N) <= P_fresh(mu,N) for all N>=2, mu;
(d) N=1: frozen == fresh exactly (single draw, same marginal sigma_draw);
(e) reproduces the ~0.81 frozen clear-prob at mu=500,N=5 (and #194's banked
    frozen_probe table VERBATIM within tol);
(f) NaN-clean.
PRIMARY = frozen_budget_self_test_passes (bool); TEST = mu_bar_frozen_p95 (float TPS).

LOCAL, CPU-ONLY, ANALYTIC. No GPU / vLLM / HF Job / submission / served-file / actual
official draw. BASELINE stays 481.53; greedy/PPL untouched; adds 0 TPS; authorizes NO
draws and NO shot count (a human still approves the spend AND confirms the harness
behavior). IMPORTS #194's sigma decomposition + frozen integral and #200's sequential
E[shots] VERBATIM; does NOT re-derive them. Draw-budget lane (#159/#188/#194/#200).
NOT ubel #201. NOT fern #185. NOT stark #198. NOT open2. NOT a launch.
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
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # research/validity/frozen_budget -> repo root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# source-of-truth: #194 banked re-draw budget (imported verbatim; one source/constant)
# ---------------------------------------------------------------------------
REDRAW_194 = os.path.join(_ROOT, "research/validity/redraw_budget/redraw_budget_results.json")

TARGET = 500.0
P_TARGET = 0.95
# normal quantiles (provenance: scipy.stats.norm.ppf) -- same constants as #194/#200
Z95_ONE_SIDED = 1.6448536269514722  # ppf(0.95); the N=1 break-even quantile
N_SIMPSON = 2001                    # #194's frozen-integral grid (reproduce verbatim)

# PR-named operating points
PR_MU_GRID = [500.0, 505.0, 512.2, 515.0, 520.95]
PR_N_GRID = [1, 2, 5, 10]
N_BAR = 5  # #194's banked budget at the bar


def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _finite(x: Any) -> float:
    if x is None or not math.isfinite(float(x)):
        raise ValueError(f"non-finite value: {x!r}")
    return float(x)


def _phi(x: float) -> float:
    """Standard-normal CDF via erf (no scipy) -- identical to #194/#200."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float, mu: float, sigma: float) -> float:
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


# ---------------------------------------------------------------------------
# best-of-N max-statistics: fresh, frozen, and the partial-freeze generalization
# ---------------------------------------------------------------------------
def p_fresh(mu: float, sigma_draw: float, n: int, target: float = TARGET) -> float:
    """FRESH regime: both components re-randomize per shot, draws ~ N(mu, sigma_draw^2).
    P(max of N >= target) = 1 - Phi((target-mu)/sigma_draw)^N. (== #194's p_best_of_n.)"""
    return 1.0 - _phi((target - mu) / sigma_draw) ** n


def p_frozen_general(mu: float, sigma_beatable: float, sigma_frozen: float, n: int,
                     target: float = TARGET, n_grid: int = N_SIMPSON) -> float:
    """best-of-N when a per-checkpoint bias b ~ N(0, sigma_frozen^2) is FROZEN across all
    N shots and only eps ~ N(0, sigma_beatable^2) re-draws per shot:
        P = 1 - E_b[ Phi((target - mu - b)/sigma_beatable)^N ],  b ~ N(0, sigma_frozen^2).
    Integrated on a deterministic Simpson grid over [-8 sigma_frozen, 8 sigma_frozen]
    (== #194's p_best_of_n_frozen with sigma_frozen=sigma_sample, sigma_beatable=sigma_hw).
    Degenerate limits handled exactly:
      * sigma_frozen->0  (all variance beatable): P = 1 - Phi((target-mu)/sigma_beatable)^N
        (pure best-of-N; == FRESH when sigma_beatable=sigma_draw);
      * sigma_beatable->0 (all variance frozen): P = Phi((mu-target)/sigma_frozen),
        N-INDEPENDENT (best-of-N is powerless -- the deterministic decode gives the
        same score every shot)."""
    if sigma_frozen <= 1e-12:
        return 1.0 - _phi((target - mu) / sigma_beatable) ** n
    if sigma_beatable <= 1e-12:
        # eps degenerate: clear iff mu + b >= target; b ~ N(0, sigma_frozen^2)
        return _phi((mu - target) / sigma_frozen)
    lo, hi = -8.0 * sigma_frozen, 8.0 * sigma_frozen
    h = (hi - lo) / (n_grid - 1)
    acc = 0.0
    for i in range(n_grid):
        b = lo + i * h
        w = 1.0 if i in (0, n_grid - 1) else (4.0 if i % 2 == 1 else 2.0)
        acc += w * _norm_pdf(b, 0.0, sigma_frozen) * _phi((target - mu - b) / sigma_beatable) ** n
    expect_fail = acc * h / 3.0
    return 1.0 - expect_fail


def n_star_fresh(mu: float, sigma_draw: float, p_target: float = P_TARGET,
                 target: float = TARGET, n_cap: int = 200) -> int | None:
    """Minimal N s.t. FRESH P(max of N >= target) >= p_target (== #194's n_star)."""
    p1 = 1.0 - _phi((target - mu) / sigma_draw)
    if p1 >= p_target:
        return 1
    if p1 <= 0.0:
        return None
    n = max(1, int(math.ceil(math.log(1.0 - p_target) / math.log(1.0 - p1))))
    return n if n <= n_cap else None


def n_star_frozen(mu: float, sigma_beatable: float, sigma_frozen: float,
                  p_target: float = P_TARGET, target: float = TARGET,
                  n_cap: int = 5000) -> int | None:
    """Minimal N s.t. FROZEN P(max of N >= target) >= p_target. None if > n_cap.
    P_frozen is monotone non-decreasing in N, so a linear scan is exact."""
    for n in range(1, n_cap + 1):
        if p_frozen_general(mu, sigma_beatable, sigma_frozen, n, target) >= p_target:
            return n
    return None


def seq_expected_shots(p: float, n_max: int) -> float:
    """E[shots] for stop-on-first-clear truncated at n_max (== #200's seq_expected_shots).
    shots = min(Geom(p), n_max); E = sum_{k=1}^{n_max} (1-p)^{k-1} = (1-(1-p)^n_max)/p."""
    if n_max <= 0:
        return 0.0
    if p <= 0.0:
        return float(n_max)
    return (1.0 - (1.0 - p) ** n_max) / p


def _bisect_mu(fn, target_val: float, lo: float, hi: float, tol: float = 1e-7,
               iters: int = 200) -> float:
    """Solve fn(mu) = target_val on [lo, hi] for a monotone-increasing fn."""
    flo, fhi = fn(lo) - target_val, fn(hi) - target_val
    if flo > 0:  # already above target at lo
        return lo
    if fhi < 0:  # never reaches target
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


def _bisect_unit(fn, target_val: float, tol: float = 1e-9, iters: int = 200) -> float | None:
    """Solve fn(f) = target_val on f in [0, 1] for a monotone fn. None if no crossing."""
    flo, fhi = fn(0.0) - target_val, fn(1.0) - target_val
    if flo >= 0:
        return 0.0  # already satisfied at full freeze
    if fhi < 0:
        return None  # not satisfied even fully fresh
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
# import #194's banked sigma decomposition (do NOT re-derive)
# ---------------------------------------------------------------------------
def import_194(rb: dict) -> dict[str, Any]:
    d = rb["decomposition"]
    sigma_sample = _finite(d["sigma_sample_1sigma_tps"])  # 5.5645 (iid finite-sample scatter)
    sigma_hw = _finite(d["sigma_hw_1sigma_tps"])          # 4.8645 (cross-allocation hardware)
    sigma_draw = _finite(rb["budget"]["sigma_draw_tps"])  # 7.3910 (= hypot(sample, hw))
    mu_safe = _finite(rb["mu_single_shot_safe_tps"])      # 512.1571 (N=1 fresh break-even)
    # #194's banked fresh mu-grid n_star and the frozen_probe (reproduce VERBATIM in self-test)
    mu_grid_n_star = [{"mu": _finite(r["mu_tps"]), "n_star": r["n_star_p95"]}
                      for r in rb["budget"]["mu_grid_rows"]]
    frozen_probe = [{"mu": _finite(r["mu"]), "n": int(r["n"]),
                     "p_iid": _finite(r["p_iid"]), "p_frozen": _finite(r["p_frozen"])}
                    for r in rb["self_test"]["frozen_probe"]]
    return {
        "sigma_sample_tps": sigma_sample,
        "sigma_hw_tps": sigma_hw,
        "sigma_draw_tps": sigma_draw,
        "mu_safe_fresh_tps": mu_safe,
        "variance_fraction_beatable_frozen": (sigma_hw ** 2) / (sigma_draw ** 2),  # 0.433
        "variance_fraction_unbeatable_frozen": (sigma_sample ** 2) / (sigma_draw ** 2),  # 0.567
        "sigma_fraction_beatable_frozen": sigma_hw / sigma_draw,  # 0.658 (the "66%" in the PR)
        "mu_grid_n_star_194": mu_grid_n_star,
        "frozen_probe_194": frozen_probe,
        "source_json": REDRAW_194,
    }


# ---------------------------------------------------------------------------
# STEP 1 -- two-regime model + regime trigger
# ---------------------------------------------------------------------------
def regime_model(imp: dict) -> dict[str, Any]:
    ss, sh, sd = imp["sigma_sample_tps"], imp["sigma_hw_tps"], imp["sigma_draw_tps"]
    identity_ok = abs(math.hypot(ss, sh) - sd) < 1e-9
    return {
        "sigma_decomposition": {
            "sigma_sample_tps": ss,
            "sigma_hw_tps": sh,
            "sigma_draw_tps": sd,
            "identity_check": "sigma_sample^2 + sigma_hw^2 == sigma_draw^2 (%.4f^2 + %.4f^2 = %.4f^2)"
                              % (ss, sh, sd),
            "identity_holds": identity_ok,
            "beatable_share_frozen": "best-of-N beats down only sigma_hw/sigma_draw = %.1f%% of the "
                                     "one-sigma scatter (sigma_hw^2/sigma_draw^2 = %.1f%% of variance)."
                                     % (100.0 * sh / sd, 100.0 * (sh ** 2) / (sd ** 2)),
        },
        "regime_trigger": {
            "FRESH": "official re-draw re-RANDOMIZES the finite-sample scatter: each shot is a fresh "
                     "128-prompt re-benchmark whose sampling deviation re-draws (the #194 premise). "
                     "best-of-N beats down the FULL sigma_draw=7.391. Trigger: official re-samples WHICH "
                     "prompts, or decode is non-deterministic so token counts vary run-to-run.",
            "FROZEN": "official re-draw RE-USES the fixed 128 public prompts under deterministic greedy "
                      "decode: the per-checkpoint sampling deviation b~N(0,sigma_sample^2) is a COMMON "
                      "bias across all shots (same prompts + same greedy tokens => same token counts => "
                      "same sampling bias), and only the HF-Job timing eps_hw~N(0,sigma_hw^2) re-draws "
                      "per allocation. best-of-N beats down ONLY sigma_hw=4.864.",
            "which_applies_is_open": "WHICH regime the official harness is in is the harness-owner's open "
                                     "question (UNMEASURED across official re-draws -- #188 measured sigma_hw "
                                     "across allocations, NOT sigma_sample's run-to-run behavior). Like #192 "
                                     "enforcement, this leg BOUNDS BOTH and does not resolve it.",
            "plausibility_note": "the challenge contract (fixed 128 prompts + greedy token-IDENTITY) leans "
                                 "FROZEN: under deterministic greedy the same prompts emit the same tokens "
                                 "every run, so the finite-sample scatter cannot re-randomize -- only HW "
                                 "timing does. FROZEN is therefore the conservative DEFAULT for budgeting "
                                 "until the human confirms the harness re-draw behavior.",
        },
    }


# ---------------------------------------------------------------------------
# STEP 2 -- frozen best-of-N table + saturation, contrasted vs fresh
# ---------------------------------------------------------------------------
def frozen_table(imp: dict) -> dict[str, Any]:
    ss, sh, sd = imp["sigma_sample_tps"], imp["sigma_hw_tps"], imp["sigma_draw_tps"]
    rows = []
    for mu in PR_MU_GRID:
        row: dict[str, Any] = {"mu_tps": mu}
        for n in PR_N_GRID:
            row[f"p_frozen_n{n}"] = p_frozen_general(mu, sh, ss, n)
            row[f"p_fresh_n{n}"] = p_fresh(mu, sd, n)
        # saturation: reported-max N->inf (-> 1 for sigma_hw>0, but log-slowly), shown at large N;
        # operationally-safe genuine ceiling governed by sigma_sample (the PR's "NOT 1").
        row["p_frozen_n100"] = p_frozen_general(mu, sh, ss, 100)
        row["p_frozen_n1000"] = p_frozen_general(mu, sh, ss, 1000)
        row["p_frozen_n10000"] = p_frozen_general(mu, sh, ss, 10000)
        row["genuine_clear_ceiling_sigma_sample"] = _phi((mu - TARGET) / ss)  # Phi((mu-500)/sigma_sample)
        row["reported_max_satN_inf"] = 1.0  # true limit for sigma_hw>0 (hardware-lucky, won't replicate)
        rows.append(row)
    return {
        "rows": rows,
        "saturation_note": (
            "REPORTED-MAX saturation P_frozen(mu,N->inf) -> 1.0 for sigma_hw>0 (max of N unbounded "
            "Gaussians diverges), but ONLY via a one-in-N HARDWARE-lucky allocation -- it grows like "
            "sigma_hw*sqrt(2 ln N), so the approach to 1 is logarithmically slow (see n100/n1000/n10000). "
            "A hardware-lucky pass does NOT replicate and would fail private re-benchmarking, so the "
            "OPERATIONALLY-safe ceiling is the sigma_sample-governed GENUINE-pass rate "
            "Phi((mu-500)/sigma_sample) -- the fraction of checkpoints whose FROZEN mean mu+b clears 500 "
            "on its own (best-of-N cannot move b). This is the PR's 'residual ceiling governed by "
            "sigma_sample, NOT 1' (exact in the sigma_hw->0 limit, self-test b)."
        ),
    }


# ---------------------------------------------------------------------------
# STEP 3 -- the frozen build bar (the deliverable)
# ---------------------------------------------------------------------------
def frozen_build_bar(imp: dict) -> dict[str, Any]:
    ss, sh, sd = imp["sigma_sample_tps"], imp["sigma_hw_tps"], imp["sigma_draw_tps"]
    mu_safe = imp["mu_safe_fresh_tps"]  # 512.157 (fresh N=1 break-even)

    # mu_bar_frozen_p95: build mu s.t. FROZEN best-of-5 clears at P>=0.95 (TEST metric)
    mu_bar_frozen_p95 = _bisect_mu(lambda mu: p_frozen_general(mu, sh, ss, N_BAR), P_TARGET, 490.0, 530.0)
    # apples-to-apples fresh N=5 bar (same shot budget, fresh regime)
    mu_bar_fresh_p95_n5 = _bisect_mu(lambda mu: p_fresh(mu, sd, N_BAR), P_TARGET, 480.0, 530.0)

    p_bar_n5_frozen = p_frozen_general(TARGET, sh, ss, N_BAR)  # P_frozen(500,5) ~ 0.81
    p_bar_n5_fresh = p_fresh(TARGET, sd, N_BAR)                # P_fresh(500,5) = 0.969

    n_shots_frozen_at_512 = n_star_frozen(512.2, sh, ss)      # min frozen N at mu=512.2 (vs #194 N=1)
    n_shots_fresh_at_512 = n_star_fresh(512.2, sd)

    delta_mu_frozen = mu_bar_frozen_p95 - mu_safe                       # PR-literal (vs fresh N=1 safe)
    delta_mu_frozen_vs_fresh_n5 = mu_bar_frozen_p95 - mu_bar_fresh_p95_n5  # apples-to-apples (same N=5)

    return {
        "mu_bar_frozen_p95": mu_bar_frozen_p95,            # TEST metric
        "mu_bar_fresh_p95_n5": mu_bar_fresh_p95_n5,
        "p_bar_n5_frozen": p_bar_n5_frozen,                # ~0.81
        "p_bar_n5_fresh": p_bar_n5_fresh,                  # 0.969
        "n_shots_frozen_at_512": n_shots_frozen_at_512,
        "n_shots_fresh_at_512": n_shots_fresh_at_512,
        "mu_safe_fresh_tps": mu_safe,
        "delta_mu_frozen": delta_mu_frozen,                # PR-named (vs mu_safe=512.157)
        "delta_mu_frozen_vs_fresh_n5": delta_mu_frozen_vs_fresh_n5,  # the real freeze-tax at fixed N=5
        "interpretation": (
            "At the mu=500 clear-bar, FROZEN best-of-5 gives P=%.4f (NOT fresh's %.4f) -- the #194 N=5 "
            "budget does NOT reach P>=0.95 under freeze. To restore P>=0.95 at N=5 the build must rise to "
            "mu_bar_frozen_p95=%.2f TPS, +%.2f TPS above the fresh-N=5 bar (%.2f) -- that is the freeze-tax "
            "at the same shot budget. RELATIVE to the fresh N=1 safe build mu_safe=%.3f, the frozen N=5 bar "
            "is %s it (delta_mu_frozen=%+.2f TPS): five frozen shots (beating down sigma_hw) still clear "
            "LOWER than a single fresh shot's safe point, so the conservative mu=512.2 build remains robust "
            "-- n_shots_frozen_at_512=%s (== #194's N=1, regime-invariant since a single shot at 512.2 "
            "already clears at 0.95)."
            % (p_bar_n5_frozen, p_bar_n5_fresh, mu_bar_frozen_p95, delta_mu_frozen_vs_fresh_n5,
               mu_bar_fresh_p95_n5, mu_safe,
               "below" if delta_mu_frozen < 0 else "above", delta_mu_frozen,
               n_shots_frozen_at_512)
        ),
    }


# ---------------------------------------------------------------------------
# STEP 4 -- harness-sensitivity (partial freeze) + risk-averse spend (p95 shots)
# ---------------------------------------------------------------------------
def harness_sensitivity(imp: dict) -> dict[str, Any]:
    ss, sh, sd = imp["sigma_sample_tps"], imp["sigma_hw_tps"], imp["sigma_draw_tps"]

    # (a) partial-freeze breakeven f*: fraction of sigma_sample that must re-randomize for N=5 P>=0.95.
    #     sigma_beatable^2 = sigma_hw^2 + f*sigma_sample^2 ; sigma_frozen^2 = (1-f)*sigma_sample^2.
    def p_partial(mu: float, f: float, n: int) -> float:
        sigma_beatable = math.sqrt(sh ** 2 + f * ss ** 2)
        sigma_frozen = math.sqrt(max(0.0, 1.0 - f)) * ss
        return p_frozen_general(mu, sigma_beatable, sigma_frozen, n)

    f_star_bar500 = _bisect_unit(lambda f: p_partial(TARGET, f, N_BAR), P_TARGET)
    f_star_safe512 = _bisect_unit(lambda f: p_partial(512.2, f, N_BAR), P_TARGET)
    partial_curve = [{"f": f, "p_partial_n5_at_500": p_partial(TARGET, f, N_BAR),
                      "p_partial_n5_at_512p2": p_partial(512.2, f, N_BAR)}
                     for f in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]

    # (b) p95 shots-paid under sequential early-stop at the bar (mu=500), both regimes.
    #     CDF of shots-to-first-clear within k draws == P(best-of-k clears) at mu=500.
    p1_fresh = p_fresh(TARGET, sd, 1)  # 0.5
    e_shots_fresh = seq_expected_shots(p1_fresh, N_BAR)  # 1.9375 (== #200)

    # frozen E[shots] capped at N_BAR: marginalize the truncated-geometric over b.
    def _frozen_e_shots_capped(n_max: int) -> float:
        lo, hi = -8.0 * ss, 8.0 * ss
        h = (hi - lo) / (N_SIMPSON - 1)
        acc = 0.0
        for i in range(N_SIMPSON):
            b = lo + i * h
            w = 1.0 if i in (0, N_SIMPSON - 1) else (4.0 if i % 2 == 1 else 2.0)
            q = 1.0 - _phi((TARGET - TARGET - b) / sh)  # P(single frozen shot clears | b) = Phi(b/sigma_hw)
            acc += w * _norm_pdf(b, 0.0, ss) * seq_expected_shots(q, n_max)
        return acc * h / 3.0

    e_shots_frozen = _frozen_e_shots_capped(N_BAR)

    def p95_capped(cdf_fn) -> int:
        for k in range(1, N_BAR + 1):
            cdf = 1.0 if k >= N_BAR else cdf_fn(k)  # capped: P(shots<=N_max)=1
            if cdf >= P_TARGET:
                return k
        return N_BAR

    p95_fresh_capped = p95_capped(lambda k: p_fresh(TARGET, sd, k))
    p95_frozen_capped = p95_capped(lambda k: p_frozen_general(TARGET, sh, ss, k))

    # uncapped p95 (true worst-case quota to force a 95% clear by re-drawing)
    def p95_uncapped(clear_fn, cap: int = 2000) -> int | None:
        for k in range(1, cap + 1):
            if clear_fn(k) >= P_TARGET:
                return k
        return None

    p95_fresh_uncapped = p95_uncapped(lambda k: p_fresh(TARGET, sd, k))
    p95_frozen_uncapped = p95_uncapped(lambda k: p_frozen_general(TARGET, sh, ss, k))

    # probability the N_BAR=5-shot plan EXHAUSTS without clearing (pays the cap, still fails)
    exhaust_fresh = 1.0 - p_fresh(TARGET, sd, N_BAR)            # 0.03125
    exhaust_frozen = 1.0 - p_frozen_general(TARGET, sh, ss, N_BAR)  # ~0.190

    return {
        "frozen_fraction_breakeven_at_bar500": f_star_bar500,
        "frozen_fraction_breakeven_at_safe512": f_star_safe512,
        "frozen_fraction_breakeven": f_star_bar500,  # headline = the bar (where N=5 was budgeted)
        "partial_freeze_curve": partial_curve,
        "p95_shots_at_bar": {
            "fresh_capped_nmax5": p95_fresh_capped,
            "frozen_capped_nmax5": p95_frozen_capped,
            "fresh_uncapped": p95_fresh_uncapped,
            "frozen_uncapped": p95_frozen_uncapped,
            "e_shots_fresh_at_bar": e_shots_fresh,
            "e_shots_frozen_at_bar": e_shots_frozen,
            "p_exhaust_without_clear_fresh": exhaust_fresh,
            "p_exhaust_without_clear_frozen": exhaust_frozen,
        },
        "note": (
            "(a) at the mu=500 bar, f*=%s of sigma_sample must re-randomize for N=5 to hit P>=0.95 "
            "(full freeze f=0 gives only %.3f); at the safe build mu=512.2, f*=%s -- N=5 clears for ANY "
            "freeze fraction (the safe build is freeze-robust). (b) capped at N_max=5 the worst-case quota "
            "is 5 shots in BOTH regimes, but the discriminator is the tail: under FROZEN the 5-shot plan "
            "pays E[shots]=%.3f (vs fresh %.3f) AND exhausts WITHOUT clearing %.1f%% of the time (vs fresh "
            "%.1f%%); to FORCE a 95%% clear by re-drawing, the uncapped quota is %s shots fresh vs %s shots "
            "frozen -- frozen-bad checkpoints can only be rescued by hardware luck, so re-drawing is a poor "
            "substitute for building higher."
            % (("%.3f" % f_star_bar500) if f_star_bar500 is not None else "None",
               partial_curve[0]["p_partial_n5_at_500"],
               ("%.3f" % f_star_safe512) if f_star_safe512 is not None else "0.000 (robust)",
               e_shots_frozen, e_shots_fresh, 100.0 * exhaust_frozen, 100.0 * exhaust_fresh,
               p95_fresh_uncapped, p95_frozen_uncapped)
        ),
    }


# ---------------------------------------------------------------------------
# STEP 5 -- self-test (PRIMARY)
# ---------------------------------------------------------------------------
def self_test(imp: dict, table: dict, bar: dict, sens: dict) -> dict[str, Any]:
    ss, sh, sd = imp["sigma_sample_tps"], imp["sigma_hw_tps"], imp["sigma_draw_tps"]
    tol = 1e-6
    checks: dict[str, bool] = {}

    # (a) sigma_sample->0 (all variance fresh, carried by the beatable component = sigma_draw):
    #     frozen == fresh exactly, and n_star reproduces #194's banked N*(mu) at every mu.
    a_ok = True
    a_detail = []
    for mu in (495.0, 500.0, 505.0, 510.0, 515.0, 520.0):
        p_fro0 = p_frozen_general(mu, sd, 0.0, 5)   # sigma_frozen=0, sigma_beatable=sigma_draw
        p_fre = p_fresh(mu, sd, 5)
        if abs(p_fro0 - p_fre) > 1e-12:
            a_ok = False
        a_detail.append({"mu": mu, "p_frozen_ss0_n5": p_fro0, "p_fresh_n5": p_fre})
    for row in imp["mu_grid_n_star_194"]:
        mu = row["mu"]
        n_ref = row["n_star"]
        # n_star under the frozen model with sigma_sample->0 must equal #194's fresh n_star
        n_mine = n_star_frozen(mu, sd, 0.0)
        if n_mine != n_ref:
            a_ok = False
        a_detail.append({"mu": mu, "n_star_frozen_ss0": n_mine, "n_star_194": n_ref})
    checks["a_sigma_sample0_recovers_fresh_and_194_nstar"] = a_ok

    # (b) sigma_hw->0 (all variance frozen): best-of-N gives ZERO improvement and the
    #     saturation = Phi((mu-500)/sigma_sample).
    b_ok = True
    b_detail = []
    for mu in (498.0, 500.0, 505.0, 512.2):
        p1 = p_frozen_general(mu, 0.0, sd, 1)   # sigma_beatable=0, sigma_frozen=sigma_draw
        pN = p_frozen_general(mu, 0.0, sd, 7)
        ceil_ss = _phi((mu - TARGET) / sd)      # with all variance in the frozen term = sigma_draw
        if abs(pN - p1) > 1e-12 or abs(p1 - ceil_ss) > 1e-9:
            b_ok = False
        b_detail.append({"mu": mu, "p_n1": p1, "p_n7": pN, "phi_ceiling": ceil_ss})
    checks["b_sigma_hw0_zero_improvement_sigma_sample_ceiling"] = b_ok

    # (c) monotone weakness: P_frozen(mu,N) <= P_fresh(mu,N) for all N>=2, mu (frozen beats
    #     down less variance -> weaker). N=1 equal (checked in d).
    c_ok = True
    c_probe = []
    for mu in (496.0, 498.0, 500.0, 503.0, 505.0, 510.0, 515.0):
        for n in (2, 3, 5, 8, 10):
            p_fro = p_frozen_general(mu, sh, ss, n)
            p_fre = p_fresh(mu, sd, n)
            c_probe.append({"mu": mu, "n": n, "p_frozen": p_fro, "p_fresh": p_fre})
            if p_fro > p_fre + 1e-9:
                c_ok = False
    checks["c_frozen_le_fresh_for_n_ge_2"] = c_ok

    # (d) N=1: frozen == fresh exactly (single draw shares the marginal sigma_draw).
    d_ok = True
    d_detail = []
    for mu in (496.0, 500.0, 505.0, 512.2, 520.95):
        p_fro = p_frozen_general(mu, sh, ss, 1)
        p_fre = p_fresh(mu, sd, 1)
        d_detail.append({"mu": mu, "p_frozen_n1": p_fro, "p_fresh_n1": p_fre})
        if abs(p_fro - p_fre) > 1e-9:
            d_ok = False
    checks["d_n1_frozen_equals_fresh"] = d_ok

    # (e) reproduces ~0.81 frozen clear-prob at mu=500,N=5; AND #194's banked frozen_probe VERBATIM.
    p500_5 = p_frozen_general(TARGET, sh, ss, 5)
    e_near_081 = abs(p500_5 - 0.81) < 0.01
    e_max_err = 0.0
    for pr in imp["frozen_probe_194"]:
        mine = p_frozen_general(pr["mu"], sh, ss, pr["n"])
        e_max_err = max(e_max_err, abs(mine - pr["p_frozen"]))
    e_verbatim = e_max_err < 1e-9
    checks["e_reproduces_081_and_194_frozen_probe_verbatim"] = e_near_081 and e_verbatim

    # (f) NaN-clean over every reported number.
    flat = (_collect(table) + _collect(bar) + _collect(sens) + _collect(imp)
            + _collect(a_detail) + _collect(b_detail) + _collect(c_probe) + _collect(d_detail)
            + [p500_5, e_max_err])
    checks["f_nan_clean"] = all(math.isfinite(x) for x in flat)

    passes = all(checks.values())
    return {
        "frozen_budget_self_test_passes": passes,
        "checks": checks,
        "a_detail": a_detail,
        "b_detail": b_detail,
        "d_detail": d_detail,
        "p_frozen_500_n5": p500_5,
        "frozen_probe_max_abs_err_vs_194": e_max_err,
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
        print(f"[frozen] wandb_logging import failed ({exc}); skipping", flush=True)
        return
    try:
        run = wandb_logging.init_wandb_run(
            job_type="frozen-sampling-redraw-budget", agent="kanna",
            name=args.wandb_name or "kanna/frozen-sampling-redraw-budget",
            group=args.wandb_group,
            tags=["frozen-sampling-redraw-budget", "best-of-n", "max-statistics", "frozen-bias",
                  "harness-sensitivity", "pr202", "pr194-stress-test"],
            config={"baseline_tps": 481.53, "method": "cpu-only-analytic", "target_tps": 500.0,
                    "p_target": P_TARGET, "imports_pr": 194, "n_bar": N_BAR},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[frozen] wandb init failed ({exc}); skipping", flush=True)
        return
    if run is None:
        print("[frozen] wandb disabled; skipping", flush=True)
        return
    try:
        bar = result["build_bar"]
        sens = result["harness_sensitivity"]
        st = result["self_test"]
        imp = result["import_194"]
        flat = {
            "frozen_budget_self_test_passes": 1.0 if st["frozen_budget_self_test_passes"] else 0.0,
            "mu_bar_frozen_p95": bar["mu_bar_frozen_p95"],
            "mu_bar_fresh_p95_n5": bar["mu_bar_fresh_p95_n5"],
            "p_bar_n5_frozen": bar["p_bar_n5_frozen"],
            "p_bar_n5_fresh": bar["p_bar_n5_fresh"],
            "delta_mu_frozen": bar["delta_mu_frozen"],
            "delta_mu_frozen_vs_fresh_n5": bar["delta_mu_frozen_vs_fresh_n5"],
            "n_shots_frozen_at_512": float(bar["n_shots_frozen_at_512"]),
            "frozen_fraction_breakeven_at_bar500": (sens["frozen_fraction_breakeven_at_bar500"]
                                                    if sens["frozen_fraction_breakeven_at_bar500"] is not None
                                                    else float("nan")),
            "frozen_fraction_breakeven_at_safe512": (sens["frozen_fraction_breakeven_at_safe512"]
                                                     if sens["frozen_fraction_breakeven_at_safe512"] is not None
                                                     else 0.0),
            "p95_shots_frozen_capped": float(sens["p95_shots_at_bar"]["frozen_capped_nmax5"]),
            "p95_shots_fresh_capped": float(sens["p95_shots_at_bar"]["fresh_capped_nmax5"]),
            "p95_shots_frozen_uncapped": float(sens["p95_shots_at_bar"]["frozen_uncapped"]),
            "p95_shots_fresh_uncapped": float(sens["p95_shots_at_bar"]["fresh_uncapped"]),
            "e_shots_frozen_at_bar": sens["p95_shots_at_bar"]["e_shots_frozen_at_bar"],
            "e_shots_fresh_at_bar": sens["p95_shots_at_bar"]["e_shots_fresh_at_bar"],
            "p_exhaust_without_clear_frozen": sens["p95_shots_at_bar"]["p_exhaust_without_clear_frozen"],
            "sigma_sample_tps": imp["sigma_sample_tps"],
            "sigma_hw_tps": imp["sigma_hw_tps"],
            "sigma_draw_tps": imp["sigma_draw_tps"],
            "sigma_fraction_beatable_frozen": imp["sigma_fraction_beatable_frozen"],
            "frozen_probe_max_abs_err_vs_194": st["frozen_probe_max_abs_err_vs_194"],
        }
        # frozen vs fresh curves per mu
        for row in result["frozen_table"]["rows"]:
            mu_k = str(row["mu_tps"]).replace(".", "p")
            for n in PR_N_GRID:
                flat[f"p_frozen_mu{mu_k}_n{n}"] = row[f"p_frozen_n{n}"]
                flat[f"p_fresh_mu{mu_k}_n{n}"] = row[f"p_fresh_n{n}"]
            flat[f"genuine_ceiling_mu{mu_k}"] = row["genuine_clear_ceiling_sigma_sample"]
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run, name="frozen_budget", artifact_type="frozen-sampling-redraw-budget", data=result)
    except Exception as exc:  # noqa: BLE001
        print(f"[frozen] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:  # noqa: BLE001
            pass


def _print(result: dict[str, Any]) -> None:
    imp = result["import_194"]
    bar = result["build_bar"]
    sens = result["harness_sensitivity"]
    st = result["self_test"]
    print("\n[frozen] ===== FROZEN-SAMPLING RE-DRAW BUDGET (PR #202) =====", flush=True)
    print(f"  sigma_sample={imp['sigma_sample_tps']:.4f} (FROZEN under fixed prompts + greedy)  "
          f"sigma_hw={imp['sigma_hw_tps']:.4f} (beatable)  sigma_draw={imp['sigma_draw_tps']:.4f}",
          flush=True)
    print(f"  best-of-N beats down only {100.0 * imp['sigma_fraction_beatable_frozen']:.1f}% of one-sigma "
          f"({100.0 * imp['variance_fraction_beatable_frozen']:.1f}% of variance) under freeze", flush=True)
    print("\n  P_frozen vs P_fresh (best-of-5):", flush=True)
    for row in result["frozen_table"]["rows"]:
        print(f"    mu={row['mu_tps']:7.2f}  P_frozen(5)={row['p_frozen_n5']:.4f}  "
              f"P_fresh(5)={row['p_fresh_n5']:.4f}  genuine_ceiling={row['genuine_clear_ceiling_sigma_sample']:.4f}",
              flush=True)
    print(f"\n  p_bar_n5_frozen (mu=500) = {bar['p_bar_n5_frozen']:.4f}  (fresh {bar['p_bar_n5_fresh']:.4f})",
          flush=True)
    print(f"  mu_bar_frozen_p95 (TEST) = {bar['mu_bar_frozen_p95']:.3f} TPS  "
          f"(fresh-N5 bar {bar['mu_bar_fresh_p95_n5']:.3f})", flush=True)
    print(f"  delta_mu_frozen (vs mu_safe 512.157) = {bar['delta_mu_frozen']:+.3f} TPS  |  "
          f"vs fresh-N5 bar = {bar['delta_mu_frozen_vs_fresh_n5']:+.3f} TPS", flush=True)
    print(f"  n_shots_frozen_at_512 = {bar['n_shots_frozen_at_512']}  (fresh {bar['n_shots_fresh_at_512']})",
          flush=True)
    fb = sens["frozen_fraction_breakeven_at_bar500"]
    print(f"\n  frozen_fraction_breakeven @bar500 = {fb:.3f}  @safe512 = "
          f"{sens['frozen_fraction_breakeven_at_safe512']} (robust)" , flush=True)
    ps = sens["p95_shots_at_bar"]
    print(f"  p95 shots @bar: fresh capped={ps['fresh_capped_nmax5']} uncapped={ps['fresh_uncapped']} | "
          f"frozen capped={ps['frozen_capped_nmax5']} uncapped={ps['frozen_uncapped']}", flush=True)
    print(f"  E[shots]@bar: fresh={ps['e_shots_fresh_at_bar']:.3f}  frozen={ps['e_shots_frozen_at_bar']:.3f}  "
          f"| P(exhaust w/o clear) frozen={ps['p_exhaust_without_clear_frozen']:.3f}", flush=True)
    print(f"\n  SELF-TEST (PRIMARY) passes = {st['frozen_budget_self_test_passes']}", flush=True)
    for k, v in st["checks"].items():
        print(f"    [{'ok' if v else '!! FAILED'}] {k}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(_HERE, "frozen_budget_results.json"))
    ap.add_argument("--wandb-name", "--wandb_name", default="kanna/frozen-sampling-redraw-budget")
    ap.add_argument("--wandb-group", "--wandb_group", default="frozen-sampling-redraw-budget")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    rb194 = _load(REDRAW_194)
    imp = import_194(rb194)

    model = regime_model(imp)
    table = frozen_table(imp)
    bar = frozen_build_bar(imp)
    sens = harness_sensitivity(imp)
    st = self_test(imp, table, bar, sens)

    handoff = (
        "if official re-draws re-randomize the 128 prompts (FRESH), #194's N=5 holds (P=%.3f at the "
        "mu=500 bar); if the harness re-benchmarks the FIXED prompts under greedy (FROZEN -- the more "
        "plausible regime under the challenge's token-identity contract), only sigma_hw=%.3f is beatable, "
        "N=5 at the bar gives P=%.3f (NOT 0.969) and to hold P>=0.95 at N=5 the build bar rises from the "
        "fresh-N=5 point %.2f to mu_bar_frozen_p95=%.2f TPS (+%.2f TPS) -- though that frozen bar is still "
        "BELOW the fresh N=1 safe point 512.2 (delta_mu_frozen=%+.2f), so building to mu=512.2 stays "
        "freeze-robust at N=1. The human must confirm the harness re-draw behavior before budgeting N=5 at "
        "the bar, and land #71 should aim the build clear of the FROZEN bar (>=%.1f) to be safe."
        % (bar["p_bar_n5_fresh"], imp["sigma_hw_tps"], bar["p_bar_n5_frozen"], bar["mu_bar_fresh_p95_n5"],
           bar["mu_bar_frozen_p95"], bar["delta_mu_frozen_vs_fresh_n5"], bar["delta_mu_frozen"],
           bar["mu_bar_frozen_p95"])
    )

    result = {
        "pr": 202,
        "metric_primary": "frozen_budget_self_test_passes",
        "metric_test": "mu_bar_frozen_p95",
        "frozen_budget_self_test_passes": st["frozen_budget_self_test_passes"],
        "mu_bar_frozen_p95": bar["mu_bar_frozen_p95"],
        "p_bar_n5_frozen": bar["p_bar_n5_frozen"],
        "delta_mu_frozen": bar["delta_mu_frozen"],
        "frozen_fraction_breakeven": sens["frozen_fraction_breakeven"],
        "regime_model": model,
        "frozen_table": table,
        "build_bar": bar,
        "harness_sensitivity": sens,
        "self_test": st,
        "import_194": imp,
        "handoff": handoff,
        "scope": "Models BOTH the FRESH (#194) and FROZEN (#194 5d risk) re-draw regimes over #194's "
        "banked sigma decomposition and pins the conservative FROZEN build bar. Takes NO official draws, "
        "authorizes NO shot count or spend (a human still approves AND confirms the harness behavior), "
        "CPU-only. BANK-THE-ANALYSIS: adds 0 TPS, greedy/PPL untouched. WHICH regime the official harness "
        "is in is the harness-owner's open question (like #192 enforcement); FROZEN is the conservative "
        "default under the challenge's token-identity contract. Draw-budget lane (#159/#188/#194/#200). "
        "NOT ubel #201. NOT fern #185. NOT stark #198. NOT open2. NOT a launch.",
        "imported_legs": {
            "kanna_194_sigma_decomposition_and_frozen_integral":
                "research/validity/redraw_budget/redraw_budget_results.json",
            "kanna_200_sequential_expected_shots":
                "research/validity/cost_budget/cost_budget_results.json",
        },
        "public_evidence_used": [
            "Leaderboard frontier tops at ~489.6 TPS (osoi5-...-precache-skv64-v1), BELOW the 500 bar -- "
            "builds (land #71 descent) aim to push past 500; this leg budgets the official re-draw shots "
            "and pins how the FROZEN regime raises that bar.",
            "kanna #194 (mxm5q63j) re-draw budget N=5@bar under the FRESH assumption it itself flagged as "
            "load-bearing (5d); this leg stress-tests that assumption.",
            "Max order-statistics P(max of N>=t)=1-F(t)^N (iid/fresh) vs the frozen-bias mixture "
            "1-E_b[Phi((t-mu-b)/sigma_hw)^N]; truncated-geometric early-stop E[shots]=(1-(1-p)^N)/p (#200).",
            "fern #185 carries this frozen-vs-fresh budget as the multi-shot row.",
        ],
        "method": "LOCAL CPU-only analytic synthesis; no GPU/vLLM/HF Job/submission/served-file/draw. "
        "BASELINE stays 481.53; adds 0 TPS. Greedy identity untouched. NOT open2. NOT a launch.",
        "metrics_nan_clean": 1 if st["checks"]["f_nan_clean"] else 0,
        "elapsed_s": round(time.time() - t0, 4),
    }

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    _print(result)
    print(f"\n[frozen] HANDOFF: {handoff}", flush=True)
    print(f"[frozen] artifacts -> {args.out}", flush=True)
    _log_wandb(args, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
