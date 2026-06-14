#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Cost-aware re-draw budget: the expected-cost-minimizing shot count (PR #200).

WHAT THIS IS
------------
PR #194 (`mxm5q63j`, kanna, MERGED) gave the official-shot budget N*(mu): N=5 at
the build-bar mu=500, N=1 at mu>=520.95, break-even mu=512.16 -- but treated every
official shot as FREE. Each official shot is a fresh HF a10g-small job, so N=5 = 5x
the per-draw GPU cost. There is an expected-cost tradeoff #194 never priced:
spending MORE on land #71's build (raising mu) BUYS a smaller N (fewer shots),
while building at the bar needs 5. This leg PRICES #194's banked curve: it computes
the expected-cost-minimizing (build-target mu, shot-count N) frontier so the human's
`Approval request` authorizes the CHEAPEST sufficient plan to clear 500 at P>=0.95,
not just a shot count.

IMPORTS (does NOT re-derive) #194's banked N*(mu) curve VERBATIM from
research/validity/redraw_budget/redraw_budget_results.json:
  * per_draw_sigma = 7.391 TPS (sigma_sample 5.564 iid + sigma_hw 4.864, both fresh
    per official run),
  * N=5 @ mu=500 (bar), N=1 @ mu=520.95 (lam=1), break-even mu_safe=512.157,
  * the tabulated best-of-N success probs (re-checked verbatim in self-test e).
The best-of-N identity P(max of N >= t) = 1 - Phi((t-mu)/sigma)^N is the universal
max-order-statistic; only the CONSTANTS are imported.

THE COST MODEL (PR step 1)
--------------------------
Two free inputs, swept as grids (no hidden constants):
  * c = per-shot GPU cost, in a10g-small job-units. NATURAL unit: one official draw
    = one a10g-small benchmark job = c=1.0 job-unit. Swept a decade each side:
    c in [0.1, 10] (log grid). The cost-min N at FIXED mu is c-INVARIANT (always the
    fewest feasible shots); c only matters when it can be TRADED against build mu.
  * b = build cost to raise mu by 1 TPS, in job-units/TPS. land #71's build-effort
    -> mu cost is NOT banked to this branch, so b is carried as a SWEPT parameter
    (b in {0} U [0.03, 32], log grid). The build-vs-redraw decision depends only on
    the ratio b/c, so the headline crossover is reported scale-free (in shot-units).

ASSUMPTIONS (explicit):
  (A1) Each official re-draw is an independent fresh HF Job -> sigma_sample AND
       sigma_hw both re-randomize per shot (the #194 fresh-re-benchmark premise; the
       FROZEN-sampling case breaks best-of-N and is the load-bearing risk, not cost).
  (A2) Build cost above the bar is incremental: cost_build(mu) = b * max(mu-500, 0).
       Building AT the bar (mu=500, land #71) is the common sunk reference (cost 0
       incremental); both plans pay it.
  (A3) The N=1 build target is the #194 break-even mu_safe=512.157 (the CHEAPEST
       build that achieves N=1), Delta_mu = mu_safe - 500 = 12.157 TPS. The PR's
       "mu=512.2" is this rounded; using the exact break-even is the fair best-case
       for the build-higher plan.
  (A4) c, b > 0 are GPU-$; the analysis prices a plan, it takes NO draws and
       authorizes none -- a human still approves the spend.

THE DELIVERABLES (PR steps 2-4)
-------------------------------
  step 2  expected_cost_fixedN(mu) = N*(mu) * c ; success 1 - Phi((500-mu)/sigma)^N*.
          expected_cost_sequential(mu) = c * E[shots], stop-on-first-clear,
          E[shots] = (1 - (1-p)^Nmax) / p  (geometric truncated at Nmax=N*(mu)).
  step 3  cost_optimal_n_at_bar (TEST): at mu=500 the cost-min feasible N = 5 (= #194).
          build_vs_redraw_crossover_cost: per-shot c* above which "build to 512.2, N=1"
          beats "build at 500, N=5". c* = b * Delta_mu / (5-1) = 3.039 * b. Scale-free:
          building higher wins iff reaching mu=512.2 costs < 4 official shots' GPU-$.
  step 4  expected_shots_sequential_at_bar = E[min(Geom(0.5), 5)] = 1.9375 (< 5 because
          early draws clear); $ saved vs fixed-5 = (5 - 1.9375)*c = 3.0625*c.

SELF-TEST (PR step 5 -- PRIMARY)
-------------------------------
(a) c->0 : cost-optimal N at FIXED mu = #194 P>=0.95 N at every mu (N=5 @ bar);
(b) expected cost monotone-increasing in c (fixed mu) and the shot-cost component
    monotone-decreasing in mu (more build -> fewer shots);
(c) the N=1 break-even reproduces #194's mu_safe=512.16 within tol;
(d) sequential expected-shots <= fixed-N at every mu (early-stop never costs more);
(e) fixed-N success prob reproduces #194's tabulated P(>=500, N) within tol;
(f) NaN-clean.
PRIMARY = cost_budget_self_test_passes (bool); TEST = cost_optimal_n_at_bar (int).

LOCAL, CPU-ONLY, ANALYTIC. No GPU / vLLM / HF Job / submission / served-file /
actual official draw. BASELINE stays 481.53; greedy/PPL untouched; adds 0 TPS;
authorizes NO spend (a human still approves). Models cost over #194's banked curve.
The build-effort->mu cost is a SWEPT assumption pending land #71's banked build cost.
Draw-budget lane (#159/#188/#194). NOT a launch. NOT open2.
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
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # research/validity/cost_budget -> repo root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# source-of-truth: #194's banked re-draw budget curve (imported verbatim)
# ---------------------------------------------------------------------------
REDRAW_194 = os.path.join(_ROOT, "research/validity/redraw_budget/redraw_budget_results.json")

TARGET = 500.0
P_TARGET = 0.95
# normal quantiles (provenance: scipy.stats.norm.ppf) -- same constants as #194
Z95_ONE_SIDED = 1.6448536269514722  # ppf(0.95); the N=1 break-even quantile

# PR step-2 operating points to price (mu in TPS); 512.2 is #194 break-even (rounded)
PR_MU_GRID = [500.0, 505.0, 512.2, 515.0, 520.95]


def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _finite(x: Any) -> float:
    if x is None or not math.isfinite(float(x)):
        raise ValueError(f"non-finite value: {x!r}")
    return float(x)


def _phi(x: float) -> float:
    """Standard-normal CDF via erf (no scipy) -- identical to #194."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _logspace(lo_exp: float, hi_exp: float, n: int) -> list[float]:
    if n == 1:
        return [10.0 ** lo_exp]
    return [10.0 ** (lo_exp + (hi_exp - lo_exp) * i / (n - 1)) for i in range(n)]


# ---------------------------------------------------------------------------
# best-of-N max-statistics (the #194 identity; constants imported, formula universal)
# ---------------------------------------------------------------------------
def p_single(mu: float, sigma: float, target: float = TARGET) -> float:
    """P(single draw >= target), draw ~ N(mu, sigma^2)."""
    return 1.0 - _phi((target - mu) / sigma)


def p_best_of_n(mu: float, sigma: float, n: int, target: float = TARGET) -> float:
    """P(at least one of N independent draws >= target)."""
    return 1.0 - _phi((target - mu) / sigma) ** n


def n_star(mu: float, sigma: float, p_target: float = P_TARGET,
           target: float = TARGET, n_cap: int = 200) -> int | None:
    """Minimal N such that P(max of N >= target) >= p_target. None if > n_cap."""
    p1 = p_single(mu, sigma, target)
    if p1 >= p_target:
        return 1
    if p1 <= 0.0:
        return None
    n = math.ceil(math.log(1.0 - p_target) / math.log(1.0 - p1))
    n = max(1, int(n))
    return n if n <= n_cap else None


def seq_expected_shots(p: float, n_max: int) -> float:
    """E[shots] for a stop-on-first-clear plan truncated at n_max draws.

    shots = min(Geom(p), n_max), support {1,...,n_max}.
    E[shots] = sum_{k=1}^{n_max} P(shots >= k) = sum_{k=1}^{n_max} (1-p)^{k-1}
             = (1 - (1-p)^{n_max}) / p   (for p > 0; = n_max for p -> 0).
    """
    if n_max <= 0:
        return 0.0
    q = 1.0 - p
    if p <= 0.0:
        return float(n_max)
    return (1.0 - q ** n_max) / p


# ---------------------------------------------------------------------------
# import #194's banked curve (do NOT re-derive)
# ---------------------------------------------------------------------------
def import_194(rb: dict) -> dict[str, Any]:
    sigma_draw = _finite(rb["budget"]["sigma_draw_tps"])             # 7.3910
    mu_safe = _finite(rb["mu_single_shot_safe_tps"])                 # 512.1571 (break-even, N=1)
    n_bar = int(rb["n_shots_for_p95_at_bar"])                        # 5
    n_lam1 = int(rb["n_shots_for_p95_at_lambda1"])                   # 1
    mu_bar = _finite(rb["budget"]["mu_bar_tps"])                     # 500.0001
    mu_lam1 = _finite(rb["budget"]["mu_lambda1_tps"])                # 520.9527
    sigma_sample = _finite(rb["decomposition"]["sigma_sample_1sigma_tps"])
    sigma_hw = _finite(rb["decomposition"]["sigma_hw_1sigma_tps"])
    # banked best-of-N table for self-test (e): reproduce these verbatim
    tabulated = []
    for r in rb["budget"]["mu_grid_rows"]:
        tabulated.append({
            "mu": _finite(r["mu_tps"]),
            "n_star": r["n_star_p95"],
            "p_best_of_2": _finite(r["p_best_of_2"]),
            "p_best_of_3": _finite(r["p_best_of_3"]),
            "p_best_of_5": _finite(r["p_best_of_5"]),
            "p_single_shot": _finite(r["p_single_shot"]),
        })
    return {
        "sigma_draw_tps": sigma_draw,
        "sigma_sample_tps": sigma_sample,
        "sigma_hw_tps": sigma_hw,
        "mu_safe_tps": mu_safe,
        "n_bar_194": n_bar,
        "n_lam1_194": n_lam1,
        "mu_bar_tps": mu_bar,
        "mu_lam1_tps": mu_lam1,
        "tabulated_best_of_n": tabulated,
        "source_json": REDRAW_194,
    }


# ---------------------------------------------------------------------------
# STEP 1 -- cost-model parametrization
# ---------------------------------------------------------------------------
def cost_model_params() -> dict[str, Any]:
    c_grid = _logspace(-1.0, 1.0, 13)       # 0.1 .. 10 job-units/shot (decade each side of 1.0)
    b_grid = [0.0] + _logspace(-1.5, 1.5, 13)  # 0 (free build) .. ~31.6 job-units/TPS
    return {
        "c_per_shot_job_units": {
            "grid": c_grid,
            "nominal_c0": 1.0,
            "unit": "a10g-small job-units; 1 official draw = 1 job-unit (NATURAL unit).",
            "sweep": "a decade each side of the realistic per-draw cost c0=1.0.",
        },
        "b_per_mu_job_units_per_tps": {
            "grid": b_grid,
            "nominal_b0": 1.0,
            "unit": "job-units to raise mu by 1 TPS (build effort, land #71 lane).",
            "sweep": "0 (free build) through ~31.6; land #71's build->mu cost is NOT "
                     "banked to this branch, so b is a SWEPT free parameter.",
            "note": "the build-vs-redraw decision depends only on the ratio b/c; the "
                    "headline crossover is reported scale-free (in shot-units).",
        },
        "assumptions": {
            "A1_independence": "each re-draw is a fresh HF Job -> sigma_sample AND sigma_hw "
                               "re-randomize per shot (the #194 fresh-re-benchmark premise).",
            "A2_incremental_build": "cost_build(mu) = b * max(mu-500, 0); the bar build is the "
                                    "common sunk reference (0 incremental).",
            "A3_n1_target": "N=1 build target = #194 break-even mu_safe=512.157 (cheapest N=1 "
                            "build); Delta_mu = 12.157 TPS.",
            "A4_no_authorization": "c,b are GPU-$; prices a plan, takes NO draws, authorizes none.",
        },
    }


# ---------------------------------------------------------------------------
# STEP 2 -- expected cost per mu (fixed-N vs sequential), swept over c
# ---------------------------------------------------------------------------
def expected_costs(imp: dict, c_grid: list[float], c0: float) -> dict[str, Any]:
    sigma = imp["sigma_draw_tps"]
    rows = []
    for mu in PR_MU_GRID:
        p = p_single(mu, sigma)
        nst = n_star(mu, sigma)
        e_shots = seq_expected_shots(p, nst) if nst is not None else float("nan")
        success = p_best_of_n(mu, sigma, nst) if nst is not None else float("nan")
        rows.append({
            "mu_tps": mu,
            "p_single": p,
            "n_star": nst,
            "success_at_n_star": success,
            "expected_shots_sequential": e_shots,
            "expected_cost_fixedN_at_c0": (nst * c0) if nst is not None else float("nan"),
            "expected_cost_sequential_at_c0": (c0 * e_shots) if nst is not None else float("nan"),
            "cost_fixedN_per_c_grid": [nst * c if nst is not None else float("nan") for c in c_grid],
            "cost_sequential_per_c_grid": [c * e_shots if nst is not None else float("nan") for c in c_grid],
        })
    return {"c0": c0, "rows": rows}


# ---------------------------------------------------------------------------
# STEP 3 -- cost-optimal plan + build-vs-redraw crossover (the deliverable)
# ---------------------------------------------------------------------------
def cost_optimal(imp: dict, c0: float, b0: float, b_grid: list[float]) -> dict[str, Any]:
    sigma = imp["sigma_draw_tps"]
    mu_safe = imp["mu_safe_tps"]
    n_bar = imp["n_bar_194"]      # 5
    delta_mu = mu_safe - TARGET   # 12.157

    # (i) cost-optimal N at the bar (mu=500 fixed): min feasible N = n_star(500) = 5.
    #     c-INVARIANT (N*c increasing in N at fixed mu) -> equals #194's N.
    n_at_bar = n_star(TARGET, sigma)

    # (ii) full (mu, N) frontier argmin at nominal (b0, c0), fixed-N and sequential.
    mu_fine = [500.0 + 0.05 * i for i in range(0, 501)]  # 500 .. 525 step 0.05
    def frontier_argmin(b: float, c: float, sequential: bool) -> dict[str, Any]:
        best = None
        for mu in mu_fine:
            nst = n_star(mu, sigma)
            if nst is None:
                continue
            build = b * max(mu - TARGET, 0.0)
            if sequential:
                shots = seq_expected_shots(p_single(mu, sigma), nst)
            else:
                shots = float(nst)
            total = build + c * shots
            if best is None or total < best["total_cost"] - 1e-12:
                best = {"mu_tps": round(mu, 4), "n_max": nst, "expected_shots": shots,
                        "build_cost": build, "shot_cost": c * shots, "total_cost": total}
        return best

    plan_fixedN_nominal = frontier_argmin(b0, c0, sequential=False)
    plan_sequential_nominal = frontier_argmin(b0, c0, sequential=True)
    # illustrate the regime shift: at LOW build cost the optimum jumps UP toward mu_safe;
    # at high build cost it stays at the bar. (b, c) corners, fixed-N.
    frontier_examples = []
    for b, c in ((0.03, 1.0), (0.1, 1.0), (0.3, 1.0), (1.0, 1.0), (0.1, 5.0)):
        frontier_examples.append({
            "b_per_mu": b, "c_per_shot": c,
            "fixedN": frontier_argmin(b, c, sequential=False),
            "sequential": frontier_argmin(b, c, sequential=True),
        })

    # (iii) build-vs-redraw crossover. Two named corner plans:
    #   Plan B (build at bar):  mu=500, N=5   -> cost = 0 + N_bar*c
    #   Plan A (build to N=1):  mu=mu_safe,N=1 -> cost = b*Delta_mu + 1*c
    # FIXED-N crossover: N_bar*c = b*Delta_mu + 1*c  ->  c* = b*Delta_mu/(N_bar-1).
    denom = float(n_bar - 1)  # 4
    c_star_per_b_fixedN = delta_mu / denom  # slope: c* = (Delta_mu/4) * b
    # SEQUENTIAL crossover: bar plan pays only E[shots] (early-stop), N=1 plan pays 1.
    e_shots_bar = seq_expected_shots(p_single(TARGET, sigma), n_bar)  # 1.9375
    denom_seq = e_shots_bar - 1.0  # 0.9375
    c_star_per_b_sequential = delta_mu / denom_seq if denom_seq > 0 else float("inf")

    # scale-free crossover: build-higher wins iff reaching mu_safe costs < (N_bar-1) shots.
    crossover_total_shots = denom               # 4 shots' worth of GPU-$
    crossover_shots_per_tps = denom / delta_mu  # b/c threshold (shot-units per TPS)

    # c*(b) tables over the swept build-cost grid
    cstar_table = []
    for b in b_grid:
        cstar_table.append({
            "b_per_mu": b,
            "build_cost_to_mu_safe": b * delta_mu,
            "c_star_fixedN": c_star_per_b_fixedN * b,
            "c_star_sequential": c_star_per_b_sequential * b,
        })

    return {
        "cost_optimal_n_at_bar": n_at_bar,                 # TEST metric (= 5)
        "delta_mu_tps": delta_mu,
        "n_bar_194": n_bar,
        "plan_fixedN_nominal": plan_fixedN_nominal,
        "plan_sequential_nominal": plan_sequential_nominal,
        "frontier_examples": frontier_examples,
        "nominal_b0": b0,
        "nominal_c0": c0,
        # headline crossover (per-shot c* AT nominal b0; scales linearly with b)
        "build_vs_redraw_crossover_cost": c_star_per_b_fixedN * b0,
        "build_vs_redraw_crossover_cost_sequential": c_star_per_b_sequential * b0,
        "c_star_slope_per_b_fixedN": c_star_per_b_fixedN,        # 3.039
        "c_star_slope_per_b_sequential": c_star_per_b_sequential,  # 12.97
        "crossover_total_shots": crossover_total_shots,         # 4 (scale-free, fixed-N)
        "crossover_shots_per_tps": crossover_shots_per_tps,     # 0.329 (b/c threshold)
        "e_shots_bar_sequential": e_shots_bar,
        "cstar_per_b_table": cstar_table,
        "crossover_note": (
            "FIXED-N: build-to-N1 (mu=%.2f) beats bar-N5 once per-shot cost c > %.3f*b "
            "(equivalently once reaching mu=%.1f costs < %d official shots' GPU-$). "
            "SEQUENTIAL early-stop makes the bar plan pay only E[shots]=%.4f, so the "
            "crossover RISES to c > %.3f*b -- early-stop substantially WEAKENS the case "
            "for building higher."
            % (mu_safe, c_star_per_b_fixedN, mu_safe, int(denom), e_shots_bar,
               c_star_per_b_sequential)
        ),
    }


# ---------------------------------------------------------------------------
# STEP 4 -- sequential (early-stop) savings at the bar
# ---------------------------------------------------------------------------
def sequential_savings(imp: dict, c0: float) -> dict[str, Any]:
    sigma = imp["sigma_draw_tps"]
    n_bar = imp["n_bar_194"]  # 5
    p_bar = p_single(TARGET, sigma)  # 0.5
    e_shots = seq_expected_shots(p_bar, n_bar)  # 1.9375
    saved_shots = n_bar - e_shots               # 3.0625
    # per-mu sequential vs fixed-N shots
    per_mu = []
    for mu in PR_MU_GRID:
        p = p_single(mu, sigma)
        nst = n_star(mu, sigma)
        e = seq_expected_shots(p, nst) if nst is not None else float("nan")
        per_mu.append({"mu_tps": mu, "n_star": nst, "expected_shots_sequential": e,
                       "saved_shots_vs_fixed": (nst - e) if nst is not None else float("nan")})
    return {
        "expected_shots_sequential_at_bar": e_shots,        # 1.9375
        "fixed_n_at_bar": n_bar,
        "p_single_at_bar": p_bar,
        "saved_shots_at_bar": saved_shots,                  # 3.0625
        "dollars_saved_at_bar_at_c0": saved_shots * c0,
        "dollars_saved_at_bar_per_c": "saved_shots * c = %.4f * c" % saved_shots,
        "per_mu_sequential": per_mu,
        "note": "best-of-5 at the bar pays only %.4f shots on average (not 5) because "
                "~half the draws clear on shot 1; this is the REALISTIC budget the human "
                "pays under stop-on-first-clear." % e_shots,
    }


# ---------------------------------------------------------------------------
# STEP 5 -- self-test (PRIMARY)
# ---------------------------------------------------------------------------
def self_test(imp: dict, ec: dict, co: dict, ss: dict, cmp: dict) -> dict[str, Any]:
    sigma = imp["sigma_draw_tps"]
    checks: dict[str, bool] = {}

    # (a) c->0: cost-optimal N at FIXED mu = #194 P>=0.95 N at every mu (N=5 @ bar).
    #     At fixed mu, min feasible N = n_star(mu) for ANY c>0 (and the c->0 limit).
    a_ok = True
    a_detail = []
    for t in imp["tabulated_best_of_n"]:
        mu = t["mu"]
        mine = n_star(mu, sigma)            # cost-optimal N at fixed mu (c-invariant)
        ref = t["n_star"]
        a_detail.append({"mu": mu, "cost_optimal_n": mine, "n_star_194": ref})
        if mine != ref:
            a_ok = False
    # and explicitly at the bar
    a_ok = a_ok and (co["cost_optimal_n_at_bar"] == imp["n_bar_194"] == 5)
    checks["a_c_to_zero_recovers_194_n"] = a_ok

    # (b) expected cost monotone-increasing in c (fixed mu) and shot-cost monotone-
    #     decreasing in mu (more build -> fewer shots).
    c_probe = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    mono_c = True
    for row in ec["rows"]:
        nst = row["n_star"]
        if nst is None:
            continue
        seq_costs = [c * row["expected_shots_sequential"] for c in c_probe]
        fix_costs = [c * nst for c in c_probe]
        mono_c = mono_c and all(seq_costs[i + 1] >= seq_costs[i] - 1e-12 for i in range(len(c_probe) - 1))
        mono_c = mono_c and all(fix_costs[i + 1] >= fix_costs[i] - 1e-12 for i in range(len(c_probe) - 1))
    # shot-cost (b=0) non-increasing in mu, both schemes
    fixn_seq = [r["n_star"] for r in ec["rows"] if r["n_star"] is not None]
    seqn_seq = [r["expected_shots_sequential"] for r in ec["rows"] if r["n_star"] is not None]
    mono_mu = (all(fixn_seq[i + 1] <= fixn_seq[i] + 1e-12 for i in range(len(fixn_seq) - 1))
               and all(seqn_seq[i + 1] <= seqn_seq[i] + 1e-12 for i in range(len(seqn_seq) - 1)))
    checks["b_cost_monotone_c_up_mu_down"] = mono_c and mono_mu

    # (c) N=1 break-even reproduces #194's mu_safe=512.16 within tol.
    mu_safe_recomputed = TARGET + Z95_ONE_SIDED * sigma
    checks["c_break_even_reproduces_194_mu_safe"] = abs(mu_safe_recomputed - imp["mu_safe_tps"]) < 1e-6

    # (d) sequential expected-shots <= fixed-N at every mu (early-stop never costs more).
    d_ok = True
    for row in ec["rows"]:
        if row["n_star"] is None:
            continue
        if row["expected_shots_sequential"] > row["n_star"] + 1e-12:
            d_ok = False
    checks["d_sequential_le_fixed_n"] = d_ok

    # (e) fixed-N success prob reproduces #194's tabulated P(>=500, N) verbatim.
    e_ok = True
    e_max_err = 0.0
    for t in imp["tabulated_best_of_n"]:
        mu = t["mu"]
        for n, key in ((2, "p_best_of_2"), (3, "p_best_of_3"), (5, "p_best_of_5")):
            mine = p_best_of_n(mu, sigma, n)
            err = abs(mine - t[key])
            e_max_err = max(e_max_err, err)
            if err > 1e-9:
                e_ok = False
        err1 = abs(p_single(mu, sigma) - t["p_single_shot"])
        e_max_err = max(e_max_err, err1)
        if err1 > 1e-9:
            e_ok = False
    checks["e_fixedN_success_reproduces_194"] = e_ok

    # (f) NaN-clean over every reported number.
    flat = (_collect(ec) + _collect(co) + _collect(ss) + _collect(cmp)
            + _collect(imp) + [mu_safe_recomputed, e_max_err])
    checks["f_nan_clean"] = all(math.isfinite(x) for x in flat)

    passes = all(checks.values())
    return {
        "cost_budget_self_test_passes": passes,
        "checks": checks,
        "a_detail": a_detail,
        "mu_safe_recomputed_tps": mu_safe_recomputed,
        "e_max_abs_err": e_max_err,
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
        print(f"[cost] wandb_logging import failed ({exc}); skipping", flush=True)
        return
    try:
        run = wandb_logging.init_wandb_run(
            job_type="cost-aware-redraw-budget", agent="kanna",
            name=args.wandb_name or "kanna/cost-aware-redraw-budget",
            group=args.wandb_group,
            tags=["cost-aware-redraw-budget", "best-of-n", "expected-cost", "sequential-early-stop",
                  "build-vs-redraw", "pr200", "pr194-extends"],
            config={"baseline_tps": 481.53, "method": "cpu-only-analytic", "target_tps": 500.0,
                    "p_target": P_TARGET, "imports_pr": 194},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[cost] wandb init failed ({exc}); skipping", flush=True)
        return
    if run is None:
        print("[cost] wandb disabled; skipping", flush=True)
        return
    try:
        co = result["cost_optimal"]
        ss = result["sequential_savings"]
        st = result["self_test"]
        flat = {
            "cost_budget_self_test_passes": 1.0 if st["cost_budget_self_test_passes"] else 0.0,
            "cost_optimal_n_at_bar": co["cost_optimal_n_at_bar"],
            "build_vs_redraw_crossover_cost": co["build_vs_redraw_crossover_cost"],
            "build_vs_redraw_crossover_cost_sequential": co["build_vs_redraw_crossover_cost_sequential"],
            "c_star_slope_per_b_fixedN": co["c_star_slope_per_b_fixedN"],
            "c_star_slope_per_b_sequential": co["c_star_slope_per_b_sequential"],
            "crossover_total_shots": co["crossover_total_shots"],
            "crossover_shots_per_tps": co["crossover_shots_per_tps"],
            "delta_mu_tps": co["delta_mu_tps"],
            "expected_shots_sequential_at_bar": ss["expected_shots_sequential_at_bar"],
            "saved_shots_at_bar": ss["saved_shots_at_bar"],
            "sigma_draw_tps": result["import_194"]["sigma_draw_tps"],
            "mu_safe_tps": result["import_194"]["mu_safe_tps"],
            "e_max_abs_err": st["e_max_abs_err"],
            "plan_fixedN_nominal_mu": co["plan_fixedN_nominal"]["mu_tps"],
            "plan_fixedN_nominal_n": co["plan_fixedN_nominal"]["n_max"],
            "plan_sequential_nominal_mu": co["plan_sequential_nominal"]["mu_tps"],
        }
        for row in result["expected_costs"]["rows"]:
            mu_k = str(row["mu_tps"]).replace(".", "p")
            flat[f"cost_fixedN_mu_{mu_k}"] = row["expected_cost_fixedN_at_c0"]
            flat[f"cost_seq_mu_{mu_k}"] = row["expected_cost_sequential_at_c0"]
            flat[f"n_star_mu_{mu_k}"] = float(row["n_star"]) if row["n_star"] is not None else float("nan")
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run, name="cost_budget", artifact_type="cost-aware-redraw-budget", data=result)
    except Exception as exc:  # noqa: BLE001
        print(f"[cost] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:  # noqa: BLE001
            pass


def _print(result: dict[str, Any]) -> None:
    imp = result["import_194"]
    ec = result["expected_costs"]
    co = result["cost_optimal"]
    ss = result["sequential_savings"]
    st = result["self_test"]
    print("\n[cost] ===== COST-AWARE RE-DRAW BUDGET (PR #200) =====", flush=True)
    print(f"  imported #194: sigma_draw={imp['sigma_draw_tps']:.4f}  mu_safe(N=1 break-even)="
          f"{imp['mu_safe_tps']:.3f}  N_bar={imp['n_bar_194']}", flush=True)
    print(f"\n  expected cost per mu (at c0={ec['c0']:.1f} job-units/shot):", flush=True)
    print("    mu      p_single  N*  E[shots]_seq  cost_fixedN  cost_seq", flush=True)
    for r in ec["rows"]:
        print(f"    {r['mu_tps']:7.2f}  {r['p_single']:.4f}    {str(r['n_star']):>2}   "
              f"{r['expected_shots_sequential']:.4f}        {r['expected_cost_fixedN_at_c0']:.3f}      "
              f"{r['expected_cost_sequential_at_c0']:.4f}", flush=True)
    print(f"\n  cost_optimal_n_at_bar (TEST) = {co['cost_optimal_n_at_bar']}  (= #194 N={imp['n_bar_194']})",
          flush=True)
    print(f"  Delta_mu (build to N=1) = {co['delta_mu_tps']:.3f} TPS", flush=True)
    print(f"  build-vs-redraw crossover (FIXED-N): c* = {co['c_star_slope_per_b_fixedN']:.3f} * b  "
          f"(build-higher wins iff reaching mu_safe costs < {int(co['crossover_total_shots'])} shots)",
          flush=True)
    print(f"  build-vs-redraw crossover (SEQUENTIAL): c* = {co['c_star_slope_per_b_sequential']:.3f} * b "
          f"(early-stop weakens build-higher)", flush=True)
    print(f"\n  sequential early-stop @ bar: E[shots] = {ss['expected_shots_sequential_at_bar']:.4f} "
          f"(< {ss['fixed_n_at_bar']}); saves {ss['saved_shots_at_bar']:.4f} shots = "
          f"{ss['saved_shots_at_bar']:.4f}*c GPU-$", flush=True)
    print(f"\n  SELF-TEST (PRIMARY) passes = {st['cost_budget_self_test_passes']}", flush=True)
    for k, v in st["checks"].items():
        flag = "ok" if v else "!! FAILED"
        print(f"    [{flag}] {k}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(_HERE, "cost_budget_results.json"))
    ap.add_argument("--wandb-name", "--wandb_name", default="kanna/cost-aware-redraw-budget")
    ap.add_argument("--wandb-group", "--wandb_group", default="cost-aware-redraw-budget")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    rb194 = _load(REDRAW_194)
    imp = import_194(rb194)

    params = cost_model_params()
    c_grid = params["c_per_shot_job_units"]["grid"]
    c0 = params["c_per_shot_job_units"]["nominal_c0"]
    b_grid = params["b_per_mu_job_units_per_tps"]["grid"]
    b0 = params["b_per_mu_job_units_per_tps"]["nominal_b0"]

    ec = expected_costs(imp, c_grid, c0)
    co = cost_optimal(imp, c0, b0, b_grid)
    ss = sequential_savings(imp, c0)
    st = self_test(imp, ec, co, ss, params)

    n_bar = co["cost_optimal_n_at_bar"]
    e_seq = ss["expected_shots_sequential_at_bar"]
    cstar = co["c_star_slope_per_b_fixedN"]
    handoff = (
        "pricing the shots at cost c, the expected-cost-minimizing plan to clear 500 at "
        "P>=0.95 is N=%d at the bar (sequential early-stop pays only %.2f shots on average); "
        "building to mu=512.2 for N=1 beats the 5-shot plan once per-shot cost exceeds "
        "c*=%.3f*b (i.e. once reaching mu=512.2 costs < 4 official shots' GPU-$) -- the human's "
        "Approval request should authorize the cheaper of {build-higher-N1, build-at-bar-N5}, "
        "and under early-stop the bar plan only pays ~%.2f shots so the build-higher crossover "
        "rises to c*=%.2f*b." % (n_bar, e_seq, cstar, e_seq, co["c_star_slope_per_b_sequential"])
    )

    result = {
        "pr": 200,
        "metric_primary": "cost_budget_self_test_passes",
        "metric_test": "cost_optimal_n_at_bar",
        "cost_budget_self_test_passes": st["cost_budget_self_test_passes"],
        "cost_optimal_n_at_bar": n_bar,
        "build_vs_redraw_crossover_cost": co["build_vs_redraw_crossover_cost"],
        "expected_shots_sequential_at_bar": e_seq,
        "cost_model_params": params,
        "import_194": imp,
        "expected_costs": ec,
        "cost_optimal": co,
        "sequential_savings": ss,
        "self_test": st,
        "handoff": handoff,
        "scope": "Prices #194's banked official-shot budget N*(mu) with a swept cost model. "
        "Computes the expected-cost-minimizing (build-target, shot-count) frontier and the "
        "build-vs-redraw crossover. Takes NO official draws, authorizes NO spend (a human still "
        "approves), CPU-only. BANK-THE-ANALYSIS: adds 0 TPS, greedy/PPL untouched. The "
        "build-effort->mu cost (b) is a SWEPT assumption pending land #71's banked build cost. "
        "Draw-budget lane (#159/#188/#194). NOT a launch. NOT open2.",
        "imported_legs": {
            "kanna_194_redraw_budget": "research/validity/redraw_budget/redraw_budget_results.json",
        },
        "public_evidence_used": [
            "Leaderboard frontier tops at ~489.6 TPS (osoi5-...-precache-skv64-v1), BELOW the 500 "
            "bar -- motivates pricing the cheapest path to clear 500.",
            "kanna #194 (mxm5q63j) official-shot budget N*(mu): N=5@bar, N=1@520.95, break-even "
            "mu=512.16 -- this leg prices that curve.",
            "Best-of-N / max order-statistics P(max of N>=t)=1-F(t)^N; truncated-geometric "
            "stop-on-first-clear expected shots E=(1-(1-p)^N)/p.",
            "fern #185 carries this cost-min plan as the multi-shot budget row.",
        ],
        "method": "LOCAL CPU-only analytic synthesis; no GPU/vLLM/HF Job/submission/served-file/draw. "
        "BASELINE stays 481.53; adds 0 TPS. Greedy identity untouched. NOT open2. NOT a launch.",
        "metrics_nan_clean": 1 if st["checks"]["f_nan_clean"] else 0,
        "elapsed_s": round(time.time() - t0, 4),
    }

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    _print(result)
    print(f"\n[cost] HANDOFF: {handoff}", flush=True)
    print(f"[cost] artifacts -> {args.out}", flush=True)
    _log_wandb(args, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
