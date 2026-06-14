#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Official re-draw budget: how many shots N for P(clear 500) >= 0.95? (PR #194)

WHAT THIS IS
------------
The MULTI-shot complement to fern #185's single-shot GO/NO-GO. kanna #188 pinned
the single-draw sigma_oneshot=4.86 (between-device, cross-allocation) and kanna
#159 flagged "best-of-2 -> P>=0.90" -- but #159's best-of-N only re-randomized
sigma_hw. The DOMINANT single-shot term is wirbel #175's finite-sample sampling
scatter (+/-10.906 TPS 95%-halfwidth, i.e. ~5.56 TPS one-sigma -- the 4.5x term
over lawine #168's input band), and that scatter is INDEPENDENT across official
re-draws (each official run re-benchmarks 128 prompts). best-of-N official
re-draws is far more powerful against an *independent* sampling term than against
a fixed bias. This leg computes the official-shot budget N*(mu): how many shots
clear 500 at P>=0.95, exploiting which uncertainty components re-randomize.

THE PER-DRAW MODEL (PR step 1)
------------------------------
Each official draw = mu + eps_sample + eps_hw.
  * eps_sample : iid finite-sample scatter. sigma_sample = 10.906/z95 = 5.564 TPS
    (wirbel #175 both-bugs, B=16384). FRESH every official run (re-benchmark).
  * eps_hw     : hardware draw. sigma_hw = sigma_between = 4.864 TPS (kanna #188,
    cross-allocation dominated). FRESH per device allocation; an official
    re-submission is a separate HF Job -> fresh a10g-small allocation -> eps_hw
    RE-DRAWS. (Only if multiple passes were taken within ONE allocation would
    eps_hw be frozen across them; the official re-draw is a fresh submission.)
  per_draw_sigma = hypot(sigma_sample, sigma_hw) = 7.391 TPS (both fresh).
CONSISTENCY: z95*per_draw_sigma = 14.49 reproduces denken #183's lam=1 both-bugs
LCB (central 535.43 - 14.49 = 520.95) to <0.01 TPS -> the convention is locked
(the +/-10.906 is a 95% half-width, not a one-sigma).

THE BUDGET (PR steps 2-3, the deliverable)
------------------------------------------
best-of-N max-statistics under independent re-draws:
  P(max of N >= 500) = 1 - Phi((500-mu)/sigma_draw)^N.
N*(mu) = minimal shots for P>=0.95, tabulated over mu and denken #183's build-bar
lambda points. At the build-bar (mu=500.0, the both-bugs lam*_LCB=0.9052 LCB) a
single official draw clears with probability EXACTLY 0.5 (mu sits on the bar), so
the budget is N=5 (1-0.5^5=0.969>=0.95) -- and this is sigma-INVARIANT (hence
ICC-invariant). At the lam=1 projection (mu=520.95 LCB) a single shot already
clears at P=0.998 -> N=1. Break-even mu_single_shot_safe = 500 + z_.95*sigma_draw
= 512.2 TPS: above it no re-draw budget is needed.

CORRELATION SENSITIVITY (PR step 4)
-----------------------------------
Re-run with wirbel #184/#190's ICC-inflated sampling half-width (accept_half x
sqrt(Deff), Deff=1+(mbar-1)*ICC, mbar=24.6 steps/prompt). AT THE BAR the budget
is ICC-INVARIANT (N=5 for any sigma, since P=0.5 at mu=500) -> the human can
budget the bar shot-count WITHOUT waiting for #190 to pin the ICC. ABOVE the bar
(lam=1) heavy ICC inflates the budget modestly (1 -> 3 at worst ICC=1) but does
NOT blow up: the sampling term STILL re-randomizes per official run, just with a
larger sigma. The budget only truly blows up in the FROZEN case (step 5d) -- if
the sampling term does NOT re-randomize across runs (same prompts, deterministic
decode), best-of-N cannot cure it; that, not ICC, is the load-bearing assumption.

LOCAL, CPU-ONLY, ANALYTIC. No GPU / vLLM / HF Job / submission / served-file /
actual official draw. BASELINE stays 481.53; greedy/PPL untouched; adds 0 TPS;
authorizes NO shot count (a human still approves N). PRIMARY = self-test.
IMPORTS the merged legs VERBATIM (wirbel #175 +/-10.906, kanna #188 sigma_oneshot
+ #159 sigma_hw decomposition, denken #183 forward map, wirbel #184/#190 ICC
bands, #159 P=0.791 + best-of-2 reproduction targets); does NOT re-derive them.
NOT open2. NOT a launch.

SELF-TEST (PR step 5 -- PRIMARY)
-------------------------------
(a) N=1 reproduces #159's single-draw P(clear500)=0.791 at the #159 operating
    point (mu=505.46, sigma=6.737) within tol;
(b) N=2 reproduces #159's best-of-2 -> P>=0.90 (0.9829, sigma_hw-only) within tol;
(c) N*(mu) monotone-decreasing in mu, and P(N) monotone-increasing in N;
(d) best-of-N with iid sampling is STRICTLY more effective than with a
    frozen-bias sampling term (P_N_iid >= P_N_frozen, strict for some N>=2);
(e) NaN-clean.
PRIMARY = redraw_budget_self_test_passes; TEST = n_shots_for_p95_at_bar (int).
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
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # research/validity/redraw_budget -> repo root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# source-of-truth artifacts (imported verbatim; one source per constant)
# ---------------------------------------------------------------------------
ET_175 = os.path.join(_ROOT, "research/oracle_readout/et_second_moment/et_second_moment_results.json")
ONESHOT_188 = os.path.join(_ROOT, "research/validity/oneshot_hw_bound/oneshot_hw_bound_results.json")
ENV_159 = os.path.join(_ROOT, "research/validity/hw_variance_envelope/envelope.json")
CARD_183 = os.path.join(_ROOT, "research/oracle_readout/lambda_acceptance_card/lambda_acceptance_card_results.json")
ICC_184 = os.path.join(_ROOT, "research/oracle_readout/lambda_robust_topology/lambda_robust_topology_results.json")

TARGET = 500.0
P_TARGET = 0.95
# normal quantiles (provenance: scipy.stats.norm.ppf)
Z95_TWO_SIDED = 1.959963984540054   # ppf(0.975); wirbel #175's z95 for the +/-10.906 half-width
Z95_ONE_SIDED = 1.6448536269514722  # ppf(0.95); the N=1 break-even quantile
MU_GRID = [495.0, 500.0, 505.0, 510.0, 515.0, 520.0]
N_CAP = 200  # max shots before we declare "budget does not close"


def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _finite(x: Any) -> float:
    if x is None or not math.isfinite(float(x)):
        raise ValueError(f"non-finite value: {x!r}")
    return float(x)


def _phi(x: float) -> float:
    """Standard-normal CDF via erf (no scipy)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float, mu: float, sigma: float) -> float:
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


# ---------------------------------------------------------------------------
# best-of-N max-statistics
# ---------------------------------------------------------------------------
def p_best_of_n(mu: float, sigma: float, n: int, target: float = TARGET) -> float:
    """P(at least one of N independent draws >= target), draws ~ N(mu, sigma^2)."""
    p_fail_single = _phi((target - mu) / sigma)  # P(single draw < target)
    return 1.0 - p_fail_single ** n


def n_star(mu: float, sigma: float, p_target: float = P_TARGET,
           target: float = TARGET, n_cap: int = N_CAP) -> int | None:
    """Minimal N such that P(max of N >= target) >= p_target. None if > n_cap."""
    p1 = 1.0 - _phi((target - mu) / sigma)  # P(single draw >= target)
    if p1 >= p_target:
        return 1
    if p1 <= 0.0:
        return None
    p_fail = 1.0 - p1
    n = math.ceil(math.log(1.0 - p_target) / math.log(p_fail))
    n = max(1, int(n))
    return n if n <= n_cap else None


def p_best_of_n_frozen(mu: float, sigma_sample: float, sigma_hw: float, n: int,
                       target: float = TARGET, n_grid: int = 2001) -> float:
    """best-of-N when eps_sample is FROZEN (common to all N draws) and only eps_hw
    re-draws.  P = 1 - E_{eps_s}[ Phi((target-mu-eps_s)/sigma_hw)^N ], integrated
    over eps_s ~ N(0, sigma_sample^2) on a deterministic grid (Simpson).  For N=1
    this equals the iid value (convolution identity); for N>=2 it is strictly
    smaller (the frozen draw correlates the N shots)."""
    lo, hi = -8.0 * sigma_sample, 8.0 * sigma_sample
    h = (hi - lo) / (n_grid - 1)
    acc = 0.0
    for i in range(n_grid):
        eps = lo + i * h
        w = 1.0 if i in (0, n_grid - 1) else (4.0 if i % 2 == 1 else 2.0)
        integrand = _norm_pdf(eps, 0.0, sigma_sample) * _phi((target - mu - eps) / sigma_hw) ** n
        acc += w * integrand
    expect_fail = acc * h / 3.0
    return 1.0 - expect_fail


# ---------------------------------------------------------------------------
# STEP 1 -- per-draw decomposition (which components re-randomize)
# ---------------------------------------------------------------------------
def decompose(et175: dict, one188: dict) -> dict[str, Any]:
    halfwidth_both = _finite(et175["tps_finite_sample_ci_halfwidth"])           # 10.906 (95% half-width)
    halfwidth_desc = _finite(et175["tps_finite_sample_ci_halfwidth_descent_only"])
    z95 = _finite(et175["model"]["z95"])
    e_t_both = _finite(et175["imported_160"]["both_bugs_E_T_anchor"])
    bench_tokens = _finite(et175["model"]["bench_tokens_primary"])
    n_steps = bench_tokens / e_t_both
    mbar = n_steps / 128.0

    sigma_sample = halfwidth_both / z95          # one-sigma sampling (both-bugs)
    sigma_hw = _finite(one188["sigma_oneshot"])  # 4.864 (== sigma_between dominated)
    sigma_within = _finite(one188["decomposition"]["decomposition"]["sigma_within_tps"])
    sigma_between = _finite(one188["decomposition"]["decomposition"]["sigma_between_tps"])
    per_draw_sigma = math.hypot(sigma_sample, sigma_hw)

    return {
        "sampling_halfwidth_95_both_bugs_tps": halfwidth_both,
        "sampling_halfwidth_95_descent_tps": halfwidth_desc,
        "z95_two_sided": z95,
        "sigma_sample_1sigma_tps": sigma_sample,
        "sigma_hw_1sigma_tps": sigma_hw,
        "sigma_within_tps": sigma_within,
        "sigma_between_tps": sigma_between,
        "per_draw_sigma_tps": per_draw_sigma,
        "n_steps_both_bugs": n_steps,
        "mean_steps_per_prompt_mbar": mbar,
        "redraw_independence_model": {
            "eps_sample": "iid; FRESH every official run (128-prompt re-benchmark) -> re-randomizes. "
            "sigma_sample = 10.906/z95 = %.4f TPS (wirbel #175 both-bugs, B=16384)." % sigma_sample,
            "eps_hw": "sigma_between=%.4f TPS (kanna #188, cross-allocation dominated, sigma_within=%.3f "
            "negligible). An official re-submission is a separate HF Job -> fresh a10g-small allocation "
            "-> eps_hw RE-DRAWS per shot. Frozen ONLY across multiple passes within one allocation."
            % (sigma_hw, sigma_within),
            "per_draw_sigma": "hypot(sigma_sample, sigma_hw) = %.4f TPS (both components fresh per shot)."
            % per_draw_sigma,
            "frozen_alternative": "If the official harness instead RE-USES the fixed 128 public prompts "
            "under deterministic greedy decode, eps_sample is a COMMON bias (does NOT re-randomize); "
            "best-of-N cannot cure it (see step-5d frozen comparison). This is the load-bearing "
            "assumption -- the PR premise is fresh re-benchmark per run.",
        },
    }


# ---------------------------------------------------------------------------
# denken #183 forward map -- pull the central + LCB at the build-bar and lam=1
# ---------------------------------------------------------------------------
def card_points(card: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for topo in ("both_bugs", "descent_only"):
        rows = card["synthesis"]["forward_map"][topo]["tau_central_1p0"]["rows"]
        bar = next(r for r in rows if r.get("is_lambda_star_lcb"))
        lam1 = next(r for r in rows if abs(_finite(r["lambda"]) - 1.0) < 1e-9)
        out[topo] = {
            "bar_lambda": _finite(bar["lambda"]),
            "bar_central_tps": _finite(bar["central_tps"]),
            "bar_lcb_tps": _finite(bar["predicted_lcb_tps"]),
            "lam1_central_tps": _finite(lam1["central_tps"]),
            "lam1_lcb_tps": _finite(lam1["predicted_lcb_tps"]),
        }
    return out


# ---------------------------------------------------------------------------
# STEP 2 + 3 -- the budget table and verdict
# ---------------------------------------------------------------------------
def budget_table(sigma_draw: float, points: dict) -> dict[str, Any]:
    grid_rows = []
    for mu in MU_GRID:
        grid_rows.append({
            "mu_tps": mu,
            "p_single_shot": 1.0 - _phi((TARGET - mu) / sigma_draw),
            "n_star_p95": n_star(mu, sigma_draw),
            "p_best_of_2": p_best_of_n(mu, sigma_draw, 2),
            "p_best_of_3": p_best_of_n(mu, sigma_draw, 3),
            "p_best_of_5": p_best_of_n(mu, sigma_draw, 5),
        })

    bar_rows = []
    for topo in ("both_bugs", "descent_only"):
        pt = points[topo]
        for label, mu in (
            ("bar_lcb", pt["bar_lcb_tps"]),
            ("bar_central", pt["bar_central_tps"]),
            ("lam1_lcb", pt["lam1_lcb_tps"]),
            ("lam1_central", pt["lam1_central_tps"]),
        ):
            bar_rows.append({
                "topology": topo,
                "point": label,
                "lambda": pt["bar_lambda"] if label.startswith("bar") else 1.0,
                "mu_tps": mu,
                "p_single_shot": 1.0 - _phi((TARGET - mu) / sigma_draw),
                "n_star_p95": n_star(mu, sigma_draw),
            })

    # PR step-3 headline operating points (both-bugs, the PR-named mu = LCB values)
    mu_bar = points["both_bugs"]["bar_lcb_tps"]      # 500.0 (the build-bar LCB)
    mu_lam1 = points["both_bugs"]["lam1_lcb_tps"]    # 520.95 (the lam=1 LCB)
    mu_safe = TARGET + Z95_ONE_SIDED * sigma_draw    # break-even: N=1 gives P>=0.95

    return {
        "sigma_draw_tps": sigma_draw,
        "mu_grid_rows": grid_rows,
        "build_bar_rows": bar_rows,
        "n_shots_for_p95_at_bar": n_star(mu_bar, sigma_draw),
        "n_shots_for_p95_at_lambda1": n_star(mu_lam1, sigma_draw),
        "mu_bar_tps": mu_bar,
        "mu_lambda1_tps": mu_lam1,
        "mu_single_shot_safe_tps": mu_safe,
        "bar_note": "At mu=500 (the bar) P(single>=500)=0.5 EXACTLY for any sigma -> N=5 "
        "(1-0.5^5=0.969>=0.95) is sigma-invariant.",
    }


# ---------------------------------------------------------------------------
# STEP 4 -- correlation (ICC) sensitivity, imported from wirbel #184/#190 bands
# ---------------------------------------------------------------------------
def icc_sensitivity(icc184: dict, sigma_hw: float, z95: float, points: dict) -> dict[str, Any]:
    bands = icc184["synthesis"]["neff_two_level"]["recommended"]["bands"]
    mu_bar = points["both_bugs"]["bar_lcb_tps"]      # 500.0
    mu_lam1 = points["both_bugs"]["lam1_lcb_tps"]    # 520.95
    rows = []
    for b in bands:
        icc = _finite(b["icc"])
        accept_half = _finite(b["accept_half_tps"])  # ICC-inflated sampling 95% half-width
        sigma_sample_icc = accept_half / z95
        sigma_draw_icc = math.hypot(sigma_sample_icc, sigma_hw)
        rows.append({
            "icc": icc,
            "design_effect": _finite(b["design_effect"]),
            "accept_half_tps": accept_half,
            "sigma_sample_icc_tps": sigma_sample_icc,
            "sigma_draw_icc_tps": sigma_draw_icc,
            "n_star_at_bar_mu500": n_star(mu_bar, sigma_draw_icc),
            "n_star_at_lambda1": n_star(mu_lam1, sigma_draw_icc),
            "p_single_at_lambda1": 1.0 - _phi((TARGET - mu_lam1) / sigma_draw_icc),
        })
    worst = rows[-1]  # ICC=1
    iid = rows[0]     # ICC=0
    n_at_bar_iid = iid["n_star_at_bar_mu500"]
    n_at_bar_worst = worst["n_star_at_bar_mu500"]
    n_lam1_iid = iid["n_star_at_lambda1"]
    n_lam1_worst = worst["n_star_at_lambda1"]
    bar_invariant = (n_at_bar_iid == n_at_bar_worst)
    return {
        "bands": rows,
        "n_shots_for_p95_icc": n_at_bar_worst,  # at the bar, worst-case ICC
        "n_shots_for_p95_at_bar_iid": n_at_bar_iid,
        "bar_budget_icc_invariant": bar_invariant,
        "n_lambda1_iid": n_lam1_iid,
        "n_lambda1_worst_icc": n_lam1_worst,
        "blow_up_flag": (n_lam1_worst is None) or (n_at_bar_worst is None),
        "verdict": (
            "AT THE BAR (mu=500) the budget is ICC-INVARIANT (N=%s for iid AND worst-case ICC=1): "
            "P=0.5 at the bar for any scatter, so the human can budget the bar shot-count WITHOUT "
            "waiting for #190 to pin the ICC. ABOVE the bar (lam=1, mu=520.95) heavy ICC inflates "
            "the budget %s->%s but does NOT blow up -- the sampling term STILL re-randomizes per "
            "official run (just with a larger sigma). The budget only blows up in the FROZEN case "
            "(step 5d), not under ICC." % (n_at_bar_iid, n_lam1_iid, n_lam1_worst)
        ),
    }


# ---------------------------------------------------------------------------
# #159 reproduction anchors (self-test a, b)
# ---------------------------------------------------------------------------
def reproduce_159(env: dict) -> dict[str, Any]:
    central = _finite(env["envelope"]["central_tps"])  # 505.46
    sc = env["propagation"]["scenarios"]["public_draw_cal+step+taurho"]
    sigma_full = _finite(sc["sigma_with_hw_tps"])       # 6.7366
    p159_single = _finite(sc["p_clear_500_with_hw"])    # 0.791323
    bon = env["propagation"]["redraw_budget_hardware"]["best_of_n"]
    p_single_hw = _finite(bon["p_single"])              # 0.869314 (sigma_hw-only headline)
    p159_best2 = _finite(bon["ladder"][1]["p_best_of_n"])  # 0.982921 (n=2)
    sigma_hw = _finite(env["propagation"]["sigma_hw_tps"])  # 4.864

    repro_single = p_best_of_n(central, sigma_full, 1)
    repro_best2 = 1.0 - (1.0 - p_single_hw) ** 2
    # also reproduce p_single_hw from central/sigma_hw as an extra check
    repro_p_single_hw = 1.0 - _phi((TARGET - central) / sigma_hw)
    return {
        "central_tps": central,
        "sigma_full_159_tps": sigma_full,
        "p159_single_published": p159_single,
        "p159_single_reproduced": repro_single,
        "p159_single_abs_err": abs(repro_single - p159_single),
        "sigma_hw_tps": sigma_hw,
        "p_single_hw_published": p_single_hw,
        "p_single_hw_reproduced": repro_p_single_hw,
        "p159_best2_published": p159_best2,
        "p159_best2_reproduced": repro_best2,
        "p159_best2_abs_err": abs(repro_best2 - p159_best2),
    }


# ---------------------------------------------------------------------------
# STEP 5 -- self-test (PRIMARY)
# ---------------------------------------------------------------------------
def self_test(decomp: dict, table: dict, repro: dict, points: dict,
              icc: dict) -> dict[str, Any]:
    tol = 1e-6
    checks: dict[str, bool] = {}

    # (a) N=1 reproduces #159 single-draw P=0.791 at the #159 operating point
    checks["a_reproduces_159_single_0p791"] = repro["p159_single_abs_err"] < tol

    # (b) N=2 reproduces #159 best-of-2 -> P>=0.90 (0.9829)
    checks["b_reproduces_159_best_of_2"] = (
        repro["p159_best2_abs_err"] < tol and repro["p159_best2_reproduced"] >= 0.90
    )

    # (c) N*(mu) monotone non-increasing in mu; P(N) monotone non-decreasing in N
    n_seq = [r["n_star_p95"] for r in table["mu_grid_rows"]]
    n_seq_clean = [N for N in n_seq if N is not None]
    mono_mu = all(n_seq_clean[i + 1] <= n_seq_clean[i] for i in range(len(n_seq_clean) - 1))
    sigma_draw = table["sigma_draw_tps"]
    p_seq = [p_best_of_n(505.0, sigma_draw, N) for N in range(1, 13)]
    mono_n = all(p_seq[i + 1] >= p_seq[i] - tol for i in range(len(p_seq) - 1))
    checks["c_n_star_monotone_decreasing_and_p_monotone_increasing"] = mono_mu and mono_n

    # (d) best-of-N iid STRICTLY more effective than frozen-bias (P_N_iid >= P_N_frozen)
    sigma_sample = decomp["sigma_sample_1sigma_tps"]
    sigma_hw = decomp["sigma_hw_1sigma_tps"]
    sd = math.hypot(sigma_sample, sigma_hw)
    ge_all = True
    strict_any = False
    frozen_probe = []
    for mu in (498.0, 500.0, 505.0, 510.0):
        for N in (1, 2, 3, 5, 8):
            p_iid = p_best_of_n(mu, sd, N)
            p_fro = p_best_of_n_frozen(mu, sigma_sample, sigma_hw, N)
            frozen_probe.append({"mu": mu, "n": N, "p_iid": p_iid, "p_frozen": p_fro})
            if p_iid < p_fro - 1e-7:
                ge_all = False
            if N >= 2 and p_iid > p_fro + 1e-4:
                strict_any = True
    checks["d_iid_ge_frozen_strict"] = ge_all and strict_any

    # (e) NaN-clean over the reported numbers
    flat = (_collect(table) + _collect(decomp) + _collect(repro)
            + _collect(points) + _collect(icc) + _collect(frozen_probe))
    checks["e_nan_clean"] = all(math.isfinite(x) for x in flat)

    # bonus consistency: sigma_draw reproduces #183 lam=1 both-bugs LCB
    lcb_reproduced = points["both_bugs"]["lam1_central_tps"] - Z95_TWO_SIDED * sd
    checks["consistency_reproduces_183_lam1_lcb"] = (
        abs(lcb_reproduced - points["both_bugs"]["lam1_lcb_tps"]) < 0.5
    )

    passes = all(checks.values())
    return {
        "redraw_budget_self_test_passes": passes,
        "checks": checks,
        "n_numbers_checked": len(flat),
        "frozen_probe": frozen_probe,
        "lcb_183_lam1_reproduced_tps": lcb_reproduced,
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
        print(f"[redraw] wandb_logging import failed ({exc}); skipping", flush=True)
        return
    try:
        run = wandb_logging.init_wandb_run(
            job_type="oneshot-redraw-budget", agent="kanna",
            name=args.wandb_name or "kanna/oneshot-redraw-budget",
            group=args.wandb_group,
            tags=["oneshot-redraw-budget", "best-of-n", "max-statistics", "sampling-iid",
                  "icc-band", "pr194"],
            config={"baseline_tps": 481.53, "method": "cpu-only-analytic", "target_tps": 500.0,
                    "p_target": P_TARGET},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[redraw] wandb init failed ({exc}); skipping", flush=True)
        return
    if run is None:
        print("[redraw] wandb disabled; skipping", flush=True)
        return
    try:
        d = result["decomposition"]
        tb = result["budget"]
        ic = result["icc_sensitivity"]
        st = result["self_test"]
        flat = {
            "sigma_sample_1sigma_tps": d["sigma_sample_1sigma_tps"],
            "sigma_hw_1sigma_tps": d["sigma_hw_1sigma_tps"],
            "per_draw_sigma_tps": d["per_draw_sigma_tps"],
            "mean_steps_per_prompt_mbar": d["mean_steps_per_prompt_mbar"],
            "n_shots_for_p95_at_bar": tb["n_shots_for_p95_at_bar"],
            "n_shots_for_p95_at_lambda1": tb["n_shots_for_p95_at_lambda1"],
            "mu_single_shot_safe_tps": tb["mu_single_shot_safe_tps"],
            "n_shots_for_p95_icc": ic["n_shots_for_p95_icc"],
            "n_lambda1_iid": ic["n_lambda1_iid"],
            "n_lambda1_worst_icc": ic["n_lambda1_worst_icc"],
            "bar_budget_icc_invariant": 1.0 if ic["bar_budget_icc_invariant"] else 0.0,
            "blow_up_flag": 1.0 if ic["blow_up_flag"] else 0.0,
            "p159_single_abs_err": result["reproduce_159"]["p159_single_abs_err"],
            "p159_best2_abs_err": result["reproduce_159"]["p159_best2_abs_err"],
            "redraw_budget_self_test_passes": 1.0 if st["redraw_budget_self_test_passes"] else 0.0,
        }
        # per-mu budget curve
        for r in tb["mu_grid_rows"]:
            ns = r["n_star_p95"]
            flat[f"n_star_mu_{int(r['mu_tps'])}"] = float(ns) if ns is not None else float("nan")
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run, name="redraw_budget", artifact_type="oneshot-redraw-budget", data=result)
    except Exception as exc:  # noqa: BLE001
        print(f"[redraw] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:  # noqa: BLE001
            pass


def _print(result: dict[str, Any]) -> None:
    d = result["decomposition"]
    tb = result["budget"]
    ic = result["icc_sensitivity"]
    st = result["self_test"]
    print("\n[redraw] ===== OFFICIAL RE-DRAW BUDGET N*(mu) (PR #194) =====", flush=True)
    print(f"  sigma_sample = {d['sigma_sample_1sigma_tps']:.4f} TPS (one-sigma; 10.906/z95, iid per run)",
          flush=True)
    print(f"  sigma_hw     = {d['sigma_hw_1sigma_tps']:.4f} TPS (kanna #188, fresh per allocation)",
          flush=True)
    print(f"  per_draw_sigma = {d['per_draw_sigma_tps']:.4f} TPS (both fresh)", flush=True)
    print("\n  N*(mu) budget (iid sampling):", flush=True)
    for r in tb["mu_grid_rows"]:
        print(f"    mu={r['mu_tps']:.0f}  P(single)={r['p_single_shot']:.4f}  N*={r['n_star_p95']}",
              flush=True)
    print(f"\n  n_shots_for_p95_at_bar (TEST) = {tb['n_shots_for_p95_at_bar']}  (mu={tb['mu_bar_tps']:.2f})",
          flush=True)
    print(f"  n_shots_for_p95_at_lambda1    = {tb['n_shots_for_p95_at_lambda1']}  (mu={tb['mu_lambda1_tps']:.2f})",
          flush=True)
    print(f"  mu_single_shot_safe (break-even N=1) = {tb['mu_single_shot_safe_tps']:.2f} TPS", flush=True)
    print(f"\n  ICC band: n_shots_for_p95_icc (worst ICC=1, at bar) = {ic['n_shots_for_p95_icc']}  "
          f"(bar ICC-invariant={ic['bar_budget_icc_invariant']}); lam=1 {ic['n_lambda1_iid']}->"
          f"{ic['n_lambda1_worst_icc']} across ICC", flush=True)
    print(f"\n  SELF-TEST (PRIMARY) passes = {st['redraw_budget_self_test_passes']}", flush=True)
    for k, v in st["checks"].items():
        flag = "ok" if v else "!! FAILED"
        print(f"    [{flag}] {k}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(_HERE, "redraw_budget_results.json"))
    ap.add_argument("--wandb-name", "--wandb_name", default="kanna/oneshot-redraw-budget")
    ap.add_argument("--wandb-group", "--wandb_group", default="oneshot-redraw-budget")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    et175 = _load(ET_175)
    one188 = _load(ONESHOT_188)
    env159 = _load(ENV_159)
    card183 = _load(CARD_183)
    icc184 = _load(ICC_184)

    decomp = decompose(et175, one188)
    points = card_points(card183)
    sigma_draw = decomp["per_draw_sigma_tps"]
    table = budget_table(sigma_draw, points)
    icc = icc_sensitivity(icc184, decomp["sigma_hw_1sigma_tps"], decomp["z95_two_sided"], points)
    repro = reproduce_159(env159)
    st = self_test(decomp, table, repro, points, icc)

    n_bar = table["n_shots_for_p95_at_bar"]
    mu_safe = table["mu_single_shot_safe_tps"]
    handoff = (
        "with sigma_oneshot=4.86 (+) iid sampling +/-10.9 (sigma_sample=%.2f) re-randomizing per "
        "official run, the both-bugs build needs N=%s official shots for P(clear 500)>=0.95 at the "
        "build-bar mu=500 LCB (N=1 once mu>=%.1f, the break-even); the human's Approval request "
        "should budget N=%s shots -- the bar budget is ICC-INVARIANT (robust to the unresolved "
        "#190 correlation axis), but FROZEN sampling (same prompts, no re-randomize) would break "
        "best-of-N; fern #185 carries this as the multi-shot row."
        % (decomp["sigma_sample_1sigma_tps"], n_bar, mu_safe, n_bar)
    )

    result = {
        "pr": 194,
        "metric_primary": "redraw_budget_self_test_passes",
        "metric_test": "n_shots_for_p95_at_bar",
        "redraw_budget_self_test_passes": st["redraw_budget_self_test_passes"],
        "n_shots_for_p95_at_bar": n_bar,
        "n_shots_for_p95_at_lambda1": table["n_shots_for_p95_at_lambda1"],
        "n_shots_for_p95_icc": icc["n_shots_for_p95_icc"],
        "mu_single_shot_safe_tps": mu_safe,
        "decomposition": decomp,
        "card_points_183": points,
        "budget": table,
        "icc_sensitivity": icc,
        "reproduce_159": repro,
        "self_test": st,
        "handoff": handoff,
        "scope": "Models the official-shot budget N*(mu) over banked sigma's (the MULTI-shot "
        "complement to fern #185's single-shot GO/NO-GO). Takes NO official draws, authorizes NO "
        "shot count (a human still approves N), CPU-only. BANK-THE-ANALYSIS: adds 0 TPS, greedy "
        "untouched. The realistic-ICC band (wirbel #184/#190) is the conservative band; the "
        "frozen-sampling case is the load-bearing assumption-risk. NOT open2. NOT a launch.",
        "imported_legs": {
            "wirbel_175_sampling_halfwidth": "research/oracle_readout/et_second_moment/et_second_moment_results.json",
            "kanna_188_sigma_oneshot": "research/validity/oneshot_hw_bound/oneshot_hw_bound_results.json",
            "kanna_159_hw_envelope_repro": "research/validity/hw_variance_envelope/envelope.json",
            "denken_183_forward_map": "research/oracle_readout/lambda_acceptance_card/lambda_acceptance_card_results.json",
            "wirbel_184_190_icc_bands": "research/oracle_readout/lambda_robust_topology/lambda_robust_topology_results.json",
        },
        "public_evidence_used": [
            "Best-of-N / max order-statistics: P(max of N >= t) = 1 - F(t)^N for iid draws.",
            "wirbel #175 finite-sample sampling 95% CI half-width +/-10.906 TPS (both-bugs, B=16384).",
            "kanna #188/#159 sigma_hw=4.86 TPS cross-allocation hardware draw.",
            "denken #183 finite-sample-LCB forward map (lambda -> central/LCB TPS).",
            "wirbel #184/#190 two-level ICC design-effect bands (within-prompt step correlation).",
        ],
        "method": "LOCAL CPU-only analytic synthesis; no GPU/vLLM/HF Job/submission/served-file/draw. "
        "BASELINE stays 481.53; adds 0 TPS. Greedy identity untouched. NOT open2. NOT a launch.",
        "metrics_nan_clean": 1 if st["checks"]["e_nan_clean"] else 0,
        "elapsed_s": round(time.time() - t0, 4),
    }

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    _print(result)
    print(f"\n[redraw] HANDOFF: {handoff}", flush=True)
    print(f"[redraw] artifacts -> {args.out}", flush=True)
    _log_wandb(args, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
