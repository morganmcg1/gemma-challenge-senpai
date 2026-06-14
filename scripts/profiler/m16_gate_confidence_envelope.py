#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Measured-500-gate CONFIDENCE ENVELOPE: CI + required-N around fern #142 (PR #146).

WHAT THIS IS
------------
fern #142's gate (`m16_measured_500_gate.measured_m16_to_official`) maps land #71's
MEASURED M=16 point estimate (accept_length E[T], branch-hit rho2, measured step,
tau) to ONE official-TPS go/no-go at the operative clear-500 bar E[T] >= 4.862
(lawine #136 measured step 1.2182). But land measures over a FINITE verify-step
budget (the banked oracle used 1024 steps: 391 salvage / 37 full). So the measured
E[T], branch-hit, and step-time each carry SAMPLING uncertainty. A point estimate
that clears 4.862 by a hair could be a lucky 1024-step draw -- and the team gets ONE
human-approved official shot.

This layer WRAPS fern #142's point gate (it does NOT duplicate it) and adds the
uncertainty quantification a skeptical human approver will demand:
  * E[T] sampling CI two ways -- CLT/normal (cross-check) and nonparametric
    bootstrap (the trustworthy one; the per-step accept-length law is skewed:
    mostly short walks + rare deep descents).
  * branch-hit rho2 Wilson CI + the required-N to separate rho2 from the 0.033
    chain-reject floor (the discriminator that the walk DESCENDS, not chain-rejects).
  * step-time CI (single-stream low-variance, carried through).
  * the COMPOSED TPS CI through fern's map official = K_cal*E[T]/step*tau, and the
    robust verdict: robust-GREEN iff the WHOLE TPS CI clears 500, robust-RED iff the
    WHOLE CI is below 500, INDETERMINATE (needs more N) otherwise.
  * the required verify-step count N that turns an INDETERMINATE point into a
    statistically robust verdict -- the prescription land #71 must hit on his live
    re-bench, and the uncertainty leg of the eventual `Approval request: HF job`
    evidence-line (pairs with fern #142 point gate + fern #145 facet decomp).

INPUT CONTRACT
--------------
The machinery consumes a PER-STEP accept-length vector a[1..N] (one accept length
per verify step, NOT just the mean) + the per-step branch-hit indicator + a
step-time band. land #71's live readout drops in via --measured-json. Until then it
runs ARMED/PENDING against the BANKED oracle samples (board tree-488-pw-fp32-v0):
the as-built fp32 ladder [0.674,0.350,0.203,0.131,0.089,0.060,0.037], E[T]=2.621,
branch-hit ~ rho2=0.4165, 391 salvage / 37 full over 1024 steps -- and the rho-
optimal target ladder (the descending regime where land's GOOD number would land).

PER-STEP RECONSTRUCTION (exact, not a fit)
------------------------------------------
For ANY committed-path cumulative acceptance ladder C (C[d] = P(committed path
reaches depth >= d)), the per-step accepted-depth D has P(D=d) = C[d]-C[d+1], and
the per-step accept length is T = 1 + D. This is the EXACT per-step law of the
greedy verify walk (linear or tree): E[T] = 1 + sum(C) and, for the tree,
C[d] = sum of reach-probabilities over depth-d nodes (mutually exclusive on a single
committed path). So we reconstruct the per-step sample directly from a ladder:
  * as-built oracle: the measured spine ladder (its variance source); the +0.077
    salvage residual (realized E[T]=2.621 vs spine 1+sum(C)=2.544) is folded as a
    uniform per-step offset so the sample mean is the authoritative 2.621 -- a shift
    leaves the sampling VARIANCE (the thing the CI needs) untouched.
  * rho-optimal ceiling / boundary: the committed-path ladder is read off fern's DP
    (treeshape_measured_accept) at the deployed rising spine with depth-1 = q1, so
    the descending-regime variance is the real tree-walk variance -- exactly the
    regime land's successful number lives in.

LOCAL, CPU-ONLY, ANALYTIC + STATISTICS. No GPU, no vLLM, no HF Job, no submission,
no kernel build. Produces a launch-READINESS decision artifact only -- it does NOT
authorize a launch (that still goes through the human-approved-issue gate). Greedy
identity untouched by construction (serves nothing). Rides on Issue #124 RESOLVED
(greedy-exact w.r.t. the tree's own M=32 verify; PPL <= 2.42 binding).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- fern #142's point gate, reused VERBATIM as the point-estimate core ----
gate = _load("m16_measured_500_gate", os.path.join(_HERE, "m16_measured_500_gate.py"))
measured_m16_to_official = gate.measured_m16_to_official
official_tps_map = gate.official_tps_map
accept_length_for_official = gate.accept_length_for_official
fern_self_test = gate.self_test

K_CAL = gate.K_CAL                        # 125.268
E_T_LINEAR = gate.E_T_LINEAR              # 3.844 linear-MTP floor
E_T_TREE = gate.E_T_TREE                  # 5.207 rho-optimal supply ceiling
FRONTIER_OFFICIAL = gate.FRONTIER_OFFICIAL  # 481.53
TARGET_OFFICIAL = gate.TARGET_OFFICIAL    # 500.0
RHO2_BRANCH_HIT = gate.RHO2_BRANCH_HIT    # 0.4165047789261015
TAU = gate.TAU                            # {"low":0.9983,"central":1.0,"high":1.0}

# ---- the operative MEASURED step (lawine #136): E[T]>=4.862 clears 500 here ----
STEP_MEASURED_DEPTH9 = 1.2182             # lawine #136 measured depth-9 step (+0.45% vs roofline)
STEP_ROOFLINE_DEPTH9 = gate.STEP_ROOFLINE_DEPTH9  # 1.2127 (fern's self-test anchors live here)

# ---- banked oracle readout of tree-488-pw-fp32-v0 (board 20260614-100550-487) ----
ORACLE_CUM_LADDER = [0.674, 0.350, 0.203, 0.131, 0.089, 0.060, 0.037]
ORACLE_E_T = 2.621                        # measured realized accept_length (incl. salvage)
ORACLE_STEPS = 1024                       # the finite verify-step budget land will also face
ORACLE_SALVAGES = 391
ORACLE_FULL = 37
CHAIN_REJECT_FLOOR = 0.033                # null branch-hit if the walk chain-rejects (no descent)

# ---- banked DP inputs (rho-optimal descending regime; reused as in #135) ----
ACCEPT_JSON = "research/accept_calibration/accept_calibration_results.json"
RANKCOV_JSON = "research/rank_coverage/rank_coverage_results.json"
RHO_OPT_JSON = "research/spec_cost_model/rho_optimal_topology_results.json"
DEPTH1_CEILING = 0.728739760479042        # rho-optimal q1 -> ET_tree=5.207 (supply ceiling)
W_DP = 4
MAXD_DP = 24

# ---- two-sided standard-normal quantiles (scipy-free; fixed confidence levels) ----
Z = {90: 1.6448536269514722, 95: 1.959963984540054, 99: 2.5758293035489004}

# ---- CI self-test anchors (the gate is valid iff the CI machinery classifies all) ----
ANCHOR_RED_ET = 2.621                     # whole TPS CI below 500 -> robust-RED
ANCHOR_GREEN_ET = 5.207                   # whole TPS CI above 500 -> robust-GREEN
ANCHOR_BORDERLINE_ET = None               # solved so official(.,step)=500 exactly -> INDETERMINATE


def _jd(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    raise TypeError(f"{type(o).__name__} not JSON serializable")


def _finite(x: float, default: float = 0.0) -> float:
    """NaN/inf -> default (every emitted metric must be NaN-clean)."""
    return float(x) if (x is not None and math.isfinite(x)) else default


# ===========================================================================
# Per-step accept-length law from a committed-path cumulative ladder
# ===========================================================================
def pmf_from_cumulative(C: list[float]) -> np.ndarray:
    """P(D=d), d=0..len(C), from the committed-path survival C[d]=P(D>=d), d>=1.

    The survival is C_full = [1.0, C[1], C[2], ..., C[K], 0.0] (P(D>=0)=1; the spine
    cannot exceed its built depth so it terminates). The marginal PMF is the negative
    first difference, clamped non-negative and renormalised against float noise."""
    surv = np.concatenate([[1.0], np.asarray(C, dtype=np.float64), [0.0]])
    pmf = np.clip(-np.diff(surv), 0.0, None)
    s = pmf.sum()
    return pmf / s if s > 0 else pmf


def samples_from_ladder(C: list[float], n: int, target_mean: float) -> np.ndarray:
    """Reconstruct n per-step accept-lengths (T=1+D) from ladder C, mean-pinned.

    Integer accept-lengths are laid down to match the ladder's marginal depth PMF at
    budget n; a single uniform offset then pins the sample mean to `target_mean`
    (for the oracle this absorbs the +0.077 salvage residual; elsewhere it is a
    ~1e-3 rounding correction). A uniform shift preserves the sampling VARIANCE --
    the only ladder property the mean-CI depends on."""
    pmf = pmf_from_cumulative(C)
    counts = np.round(pmf * n).astype(int)
    # reconcile rounding so the counts sum to exactly n (adjust the modal bin)
    drift = n - int(counts.sum())
    if drift != 0:
        counts[int(np.argmax(counts))] += drift
    counts = np.clip(counts, 0, None)
    depths = np.repeat(np.arange(len(counts)), counts)
    raw = 1.0 + depths.astype(np.float64)            # accept length = 1 + D
    raw += target_mean - raw.mean()                  # pin mean (documented offset)
    return raw


# ===========================================================================
# Committed-path ladder of the rho-optimal DESCENDING tree (read off fern's DP)
# ===========================================================================
class DescendingLadder:
    """Lazy holder for the rho-optimal tree DP; yields the committed-path ladder
    C[d] = sum of reach-probabilities over depth-d nodes, at a chosen depth-1 q1.
    C is the EXACT per-step accept-length survival of the greedy tree walk."""

    def __init__(self, accept_json: str, rankcov_json: str, rho_opt_json: str):
        from treeshape_measured_accept import (  # noqa: E402
            build_depth_pvecs_measured, load_measured, load_rank_coverage,
            score_tree_depthrank)
        from traversal_verify_et import load_m32_topology, tree_arrays  # noqa: E402
        self._build_pvecs = build_depth_pvecs_measured
        self._score = score_tree_depthrank
        meas = load_measured(accept_json, "server_log")
        rc = load_rank_coverage(rankcov_json)
        self.q_deployed = list(meas["q"])             # deployed rising conditional spine
        self.rho_cond = rc["rho_cond"]                # [0.4165, 0.2655, 0.1908]
        self.parent = load_m32_topology(rho_opt_json)
        children, depth_arr, _ = tree_arrays(self.parent)
        self.children = children
        self.depth_arr = depth_arr
        self.built_depth = int(max(depth_arr))

    def _pvecs(self, q1: float):
        qq = list(self.q_deployed)
        qq[0] = q1
        return self._build_pvecs(qq, self.rho_cond, W_DP, MAXD_DP, "flat")

    def et(self, q1: float) -> float:
        """E[T] of the rho-optimal descending walk with depth-1 overridden to q1."""
        return float(self._score(self.parent, self._pvecs(q1))[0])

    def ladder(self, q1: float) -> list[float]:
        """Committed-path cumulative ladder C[1..built_depth] at depth-1 q1."""
        pvecs = self._pvecs(q1)
        n = len(self.parent)
        pp = np.zeros(n)
        pp[0] = 1.0
        depth = [0] * n
        maxd = len(pvecs) - 1
        for par in range(n):
            for rank, c in enumerate(self.children[par], start=1):
                d = depth[par] + 1
                pv = pvecs[min(d, maxd)]
                r = rank if rank < len(pv) else len(pv) - 1
                pp[c] = pp[par] * pv[r]
                depth[c] = d
        C = np.zeros(self.built_depth + 1)            # C[0] unused
        for node in range(1, n):
            C[depth[node]] += pp[node]
        return [float(x) for x in C[1:]]               # C[1..built_depth]

    def q1_for_et(self, target_et: float, lo: float = 0.40, hi: float = 0.95) -> float:
        """Bisection for the depth-1 q1 that realises E[T]=target_et (et monotone in q1)."""
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if self.et(mid) < target_et:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


# ===========================================================================
# Confidence intervals
# ===========================================================================
def clt_ci(mean: float, sd: float, n: int, conf: int) -> dict:
    hw = Z[conf] * sd / math.sqrt(n)
    return {"lo": mean - hw, "hi": mean + hw, "half_width": hw}


def bootstrap_ci(samples: np.ndarray, conf: int, n_boot: int, rng) -> dict:
    n = len(samples)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = samples[idx].mean(axis=1)
    lo_p, hi_p = (100 - conf) / 2.0, 100 - (100 - conf) / 2.0
    lo = float(np.percentile(boot_means, lo_p))
    hi = float(np.percentile(boot_means, hi_p))
    return {"lo": lo, "hi": hi, "half_width": 0.5 * (hi - lo),
            "boot_mean": float(boot_means.mean()), "boot_sd": float(boot_means.std(ddof=1))}


def wilson_ci(k: int, n: int, conf: int) -> dict:
    """Wilson score interval for a binomial proportion (correct near p~0.4, finite n)."""
    z = Z[conf]
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return {"p_hat": p, "lo": center - margin, "hi": center + margin, "center": center}


def required_n_to_separate(p: float, floor: float, conf: int, n_max: int = 200_000) -> int | None:
    """Smallest N whose Wilson lower bound for proportion p strictly exceeds `floor`.

    Discriminator: at this N the branch-hit CI excludes the chain-reject floor, i.e.
    the data prove the walk DESCENDS rather than chain-rejecting."""
    if p <= floor:
        return None
    for n in range(2, n_max + 1):
        k = int(round(p * n))
        if wilson_ci(k, n, conf)["lo"] > floor:
            return n
    return None


def required_n_for_verdict(et_obs: float, sigma: float, bar_worst: float, bar_best: float,
                           conf: int, n_cap: int = 5_000_000) -> dict:
    """Smallest N whose E[T] CLT half-width clears the SYSTEMATIC band so the composed
    TPS CI no longer straddles 500 and the verdict is robust at `conf`.

    robust-GREEN needs the sampling lower bound above the worst-corner (step_hi,tau_low)
    bar: et_obs - z*sigma/sqrt(N) > bar_worst. robust-RED needs the upper bound below the
    best-corner (step_lo,tau_high) bar: et_obs + z*sigma/sqrt(N) < bar_best. A point INSIDE
    the systematic dead-band [bar_best, bar_worst] makes the step+tau band ALONE straddle
    500 -> NO finite N resolves it (unbounded / permanently INDETERMINATE). This is what
    keeps the required-N curve consistent with the robust_verdict self-test."""
    if et_obs > bar_worst:
        margin, side = et_obs - bar_worst, "GREEN"
    elif et_obs < bar_best:
        margin, side = bar_best - et_obs, "RED"
    else:
        return {"required_n": None, "unbounded": True, "margin_et": 0.0,
                "side": "INDETERMINATE", "feasible": False, "in_systematic_dead_band": True}
    if margin <= 1e-9:
        return {"required_n": None, "unbounded": True, "margin_et": margin,
                "side": "tie", "feasible": False}
    n_req = max(1, math.ceil((Z[conf] * sigma / margin) ** 2))
    feasible = n_req <= n_cap
    return {"required_n": int(n_req) if feasible else None,
            "unbounded": not feasible, "margin_et": margin, "side": side, "feasible": feasible}


# ===========================================================================
# Compose a per-step sample -> E[T] CI -> TPS CI -> robust verdict
# ===========================================================================
def effective_clear500_bar(step_central: float, step_rel_hw: float) -> dict:
    """The clear-500 E[T] bar and its worst-corner (step_hi, tau_low) shift.

    central bar  = 500*step_central/(K_cal*1.0)   (== fern's accept_length_for_official)
    worst bar    = 500*step_hi/(K_cal*tau_low)    (highest E[T] a robust-GREEN needs)
    """
    step_hi = step_central * (1.0 + step_rel_hw)
    step_lo = step_central * (1.0 - step_rel_hw)
    bar_central = accept_length_for_official(TARGET_OFFICIAL, step_central, TAU["central"])
    bar_worst = accept_length_for_official(TARGET_OFFICIAL, step_hi, TAU["low"])
    bar_best = accept_length_for_official(TARGET_OFFICIAL, step_lo, TAU["high"])
    return {"bar_central": bar_central, "bar_worst": bar_worst, "bar_best": bar_best,
            "step_central": step_central, "step_hi": step_hi, "step_lo": step_lo,
            "step_rel_half_width": step_rel_hw, "systematic_et_margin": bar_worst - bar_central}


def compose_tps_ci(et_ci: dict, step_central: float, step_rel_hw: float) -> dict:
    """Propagate an E[T] CI [et_lo, et_hi] + the step band + the tau band through
    official = K_cal*E[T]/step*tau to a TPS CI. The conservative TPS lower bound
    pairs E[T]_lo with step_hi and tau_low; the optimistic upper bound the reverse."""
    step_hi = step_central * (1.0 + step_rel_hw)
    step_lo = step_central * (1.0 - step_rel_hw)
    tps_lo = official_tps_map(et_ci["lo"], step_hi, TAU["low"])
    tps_hi = official_tps_map(et_ci["hi"], step_lo, TAU["high"])
    tps_mid = official_tps_map(0.5 * (et_ci["lo"] + et_ci["hi"]), step_central, TAU["central"])
    return {"tps_lo": tps_lo, "tps_hi": tps_hi, "tps_mid": tps_mid,
            "et_lo": et_ci["lo"], "et_hi": et_ci["hi"]}


def robust_verdict(tps_ci: dict) -> str:
    if tps_ci["tps_lo"] >= TARGET_OFFICIAL:
        return "robust-GREEN"
    if tps_ci["tps_hi"] < TARGET_OFFICIAL:
        return "robust-RED"
    return "INDETERMINATE"


def analyse_sample(name: str, samples: np.ndarray, branch_hit_k: int, n: int,
                   step_central: float, step_rel_hw: float, n_boot: int, rng,
                   point_et: float | None = None) -> dict:
    """Full envelope for one per-step accept-length sample: E[T] CLT+bootstrap CI,
    branch-hit Wilson CI, composed TPS CI, robust verdict, and required-N."""
    mean = float(samples.mean())
    sd = float(samples.std(ddof=1))
    et_point = point_et if point_et is not None else mean
    bars = effective_clear500_bar(step_central, step_rel_hw)

    et_clt = {c: clt_ci(mean, sd, n, c) for c in (95, 99)}
    et_boot = {c: bootstrap_ci(samples, c, n_boot, rng) for c in (95, 99)}

    # composed TPS CI (bootstrap is the trustworthy E[T] CI; CLT carried as cross-check)
    tps_ci = {c: compose_tps_ci(et_boot[c], step_central, step_rel_hw) for c in (95, 99)}
    tps_ci_clt = {c: compose_tps_ci(et_clt[c], step_central, step_rel_hw) for c in (95, 99)}
    verdict = {c: robust_verdict(tps_ci[c]) for c in (95, 99)}

    # required-N (CLT half-width vs the systematic dead-band) at this point estimate
    req_n = {c: required_n_for_verdict(et_point, sd, bars["bar_worst"], bars["bar_best"], c)
             for c in (95, 99)}

    # branch-hit Wilson CI + separation-from-floor required-N
    wilson = {c: wilson_ci(branch_hit_k, n, c) for c in (95, 99)}
    sep_n = {c: required_n_to_separate(RHO2_BRANCH_HIT, CHAIN_REJECT_FLOOR, c) for c in (95, 99)}
    branch_excludes_floor = bool(wilson[99]["lo"] > CHAIN_REJECT_FLOOR)
    branch_consistent_rho2 = bool(wilson[99]["lo"] <= RHO2_BRANCH_HIT <= wilson[99]["hi"])

    return {
        "name": name,
        "n_steps": n,
        "point_et": et_point,
        "sample_mean_et": mean,
        "per_step_sd": sd,
        "per_step_cv": sd / mean if mean > 0 else 0.0,
        "official_tps_point": official_tps_map(et_point, step_central, TAU["central"]),
        "clear500_bars": bars,
        "et_ci_clt": {str(c): et_clt[c] for c in (95, 99)},
        "et_ci_bootstrap": {str(c): et_boot[c] for c in (95, 99)},
        "et_ci_lo_95": et_boot[95]["lo"], "et_ci_hi_95": et_boot[95]["hi"],
        "et_ci_lo_99": et_boot[99]["lo"], "et_ci_hi_99": et_boot[99]["hi"],
        "tps_ci_bootstrap": {str(c): tps_ci[c] for c in (95, 99)},
        "tps_ci_clt": {str(c): tps_ci_clt[c] for c in (95, 99)},
        "robust_verdict_95": verdict[95],
        "robust_verdict_99": verdict[99],
        "required_n_for_verdict": {str(c): req_n[c] for c in (95, 99)},
        "branch_hit": {
            "k": branch_hit_k, "n": n, "p_hat": branch_hit_k / n,
            "rho2_target": RHO2_BRANCH_HIT, "chain_reject_floor": CHAIN_REJECT_FLOOR,
            "wilson_ci": {str(c): wilson[c] for c in (95, 99)},
            "excludes_chain_reject_floor_99": branch_excludes_floor,
            "consistent_with_rho2_99": branch_consistent_rho2,
            "required_n_to_separate_from_floor": {str(c): sep_n[c] for c in (95, 99)},
        },
    }


# ===========================================================================
# Required-N hand-off curve for land #71 (descending-regime sigma)
# ===========================================================================
def required_n_curve(sigma: float, et_bar_worst: float, et_bar_best: float,
                     et_points: list[float]) -> list[dict]:
    """For each candidate live measured E[T], the verify-step count land must hit for
    a robust verdict at 95% and 99% (using the descending-regime per-step sigma). Points
    inside the systematic dead-band [bar_best, bar_worst] return unbounded -- no N is
    enough, matching the robust_verdict self-test."""
    rows = []
    for et in et_points:
        rows.append({
            "measured_et": et,
            "official_tps_point": official_tps_map(et, STEP_MEASURED_DEPTH9, TAU["central"]),
            "required_n_95": required_n_for_verdict(et, sigma, et_bar_worst, et_bar_best, 95),
            "required_n_99": required_n_for_verdict(et, sigma, et_bar_worst, et_bar_best, 99),
        })
    return rows


def min_robust_et_at_n(sigma: float, et_bar_worst: float, n: int, conf: int) -> float:
    """The minimum measured E[T] that yields a robust-GREEN at budget n and `conf`
    (E[T]_lo > effective bar): et > bar_worst + z*sigma/sqrt(n)."""
    return et_bar_worst + Z[conf] * sigma / math.sqrt(n)


# ===========================================================================
# Main
# ===========================================================================
def _load_live(path: str | None) -> dict | None:
    """land #71's live per-step readout: {accept_lengths:[...], branch_hits:[...]
    (or branch_hit_count), ppl, greedy_token_ids_captured}."""
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        m = json.load(f)
    if "accept_lengths" not in m:
        return None
    a = np.asarray(m["accept_lengths"], dtype=np.float64)
    if "branch_hits" in m:
        bh = np.asarray(m["branch_hits"])
        k = int((bh > 0).sum())
    else:
        k = int(m.get("branch_hit_count", round(RHO2_BRANCH_HIT * len(a))))
    return {"accept_lengths": a, "branch_hit_k": k, "n": len(a),
            "ppl": m.get("ppl"), "greedy_token_ids_captured": m.get("greedy_token_ids_captured")}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measured-step", type=float, default=STEP_MEASURED_DEPTH9,
                    help="operative depth-9 step (lawine #136 measured 1.2182).")
    ap.add_argument("--step-rel-half-width", type=float, default=0.005,
                    help="ASSUMED relative half-width on the step (single-stream jitter + "
                         "roofline/measured 0.45%% gap), pending lawine #136 raw timing samples.")
    ap.add_argument("--n-steps", type=int, default=ORACLE_STEPS,
                    help="verify-step budget for the banked-sample CIs (oracle 1024).")
    ap.add_argument("--n-boot", type=int, default=20000, help="bootstrap resamples (>=10k).")
    ap.add_argument("--seed", type=int, default=146)
    ap.add_argument("--measured-json", default=None,
                    help="land #71's live per-step readout {accept_lengths:[...], "
                         "branch_hits:[...]|branch_hit_count, ppl, greedy_token_ids_captured}.")
    ap.add_argument("--accept-json", default=ACCEPT_JSON)
    ap.add_argument("--rankcov-json", default=RANKCOV_JSON)
    ap.add_argument("--rho-opt-json", default=RHO_OPT_JSON)
    ap.add_argument("--out", default="research/oracle_readout/m16_gate_confidence_envelope_results.json")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="gemma-challenge-senpai")
    ap.add_argument("--wandb-entity", default="wandb-applied-ai-team")
    ap.add_argument("--wandb-name", default="wirbel/measured-gate-confidence-envelope")
    ap.add_argument("--wandb-group", default="measured-gate-confidence-envelope")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    step = args.measured_step
    srhw = args.step_rel_half_width
    n = args.n_steps

    print("=" * 96)
    print("MEASURED-500-GATE CONFIDENCE ENVELOPE (PR #146) -- CI + required-N around fern #142")
    print("=" * 96)
    print(f"map: official = K_cal*E[T]/step*tau  (K_cal={K_CAL:.3f}, measured step={step:.4f} "
          f"+-{srhw*100:.2f}%, tau in [{TAU['low']},{TAU['high']}])", flush=True)

    bars = effective_clear500_bar(step, srhw)
    print(f"clear-500 bar: central E[T] >= {bars['bar_central']:.4f}  "
          f"(worst-corner {bars['bar_worst']:.4f}; systematic margin "
          f"+{bars['systematic_et_margin']:.4f} E[T])", flush=True)

    # ---- descending-regime ladders from fern's DP (rho-optimal + boundary) ----
    dl = DescendingLadder(args.accept_json, args.rankcov_json, args.rho_opt_json)
    et_ceiling = dl.et(DEPTH1_CEILING)
    assert abs(et_ceiling - E_T_TREE) < 0.02, (et_ceiling, E_T_TREE)
    # boundary anchor: solve q1 so official(E[T], step)=500 exactly (E[T]=bar_central)
    et_borderline = bars["bar_central"]
    q1_border = dl.q1_for_et(et_borderline)
    C_ceiling = dl.ladder(DEPTH1_CEILING)
    C_border = dl.ladder(q1_border)
    print(f"DP self-check: ET_tree(0.7287)={et_ceiling:.4f} (ceiling {E_T_TREE}); "
          f"boundary q1={q1_border:.4f} -> ET={dl.et(q1_border):.4f} (target {et_borderline:.4f})",
          flush=True)

    # ---- build the three banked per-step samples ----
    # oracle (as-built): measured spine ladder, mean pinned to the realized 2.621.
    s_oracle = samples_from_ladder(ORACLE_CUM_LADDER, n, ORACLE_E_T)
    # rho-optimal ceiling + boundary: committed-path ladders off the DP (descending var).
    s_ceiling = samples_from_ladder(C_ceiling, n, et_ceiling)
    s_border = samples_from_ladder(C_border, n, et_borderline)

    # branch-hit successes for each regime (proportion rho2 over n steps; oracle anchored
    # to the measured rho2). The descending regimes also descend -> same rho2 proportion.
    k_rho2 = int(round(RHO2_BRANCH_HIT * n))

    red = analyse_sample("as_built_oracle_E2.621", s_oracle, k_rho2, n, step, srhw,
                         args.n_boot, rng, point_et=ORACLE_E_T)
    green = analyse_sample("rho_optimal_ceiling_E5.207", s_ceiling, k_rho2, n, step, srhw,
                           args.n_boot, rng, point_et=et_ceiling)
    border = analyse_sample("clear500_boundary_E4.862", s_border, k_rho2, n, step, srhw,
                            args.n_boot, rng, point_et=et_borderline)

    # ---- self-test: the CI machinery must classify all three correctly ----
    red_ok = bool(red["robust_verdict_99"] == "robust-RED" and red["robust_verdict_95"] == "robust-RED")
    green_ok = bool(green["robust_verdict_99"] == "robust-GREEN" and green["robust_verdict_95"] == "robust-GREEN")
    border_ok = bool(border["robust_verdict_99"] == "INDETERMINATE")
    # cross-check: reproduce fern #142's POINT anchors (271 RED / 538 GREEN) as CI midpoints.
    fst = fern_self_test()
    fern_point_ok = bool(fst["passes"])
    gate_ci_self_test_passes = int(red_ok and green_ok and border_ok and fern_point_ok)

    # ---- required-N: TEST metric at the oracle point (proves MOOT: tiny N) ----
    req_n_oracle_99 = red["required_n_for_verdict"]["99"]
    required_n_for_robust_500_verdict = (req_n_oracle_99["required_n"]
                                         if req_n_oracle_99["feasible"] else None)

    # ---- the hand-off prescription for land #71 (descending-regime sigma) ----
    sigma_descend = green["per_step_sd"]
    curve_points = [4.862, 4.90, 4.95, 5.00, 5.05, 5.10, 5.131, 5.207]
    handoff_curve = required_n_curve(sigma_descend, bars["bar_worst"], bars["bar_best"], curve_points)
    min_et_1024_95 = min_robust_et_at_n(sigma_descend, bars["bar_worst"], n, 95)
    min_et_1024_99 = min_robust_et_at_n(sigma_descend, bars["bar_worst"], n, 99)
    req_n_ceiling_99 = green["required_n_for_verdict"]["99"]["required_n"]
    req_n_ceiling_95 = green["required_n_for_verdict"]["95"]["required_n"]

    # ---- live gate (if land #71's per-step vector is provided) ----
    live = _load_live(args.measured_json)
    live_out = None
    land_pending = live is None
    if live is not None:
        live_out = analyse_sample("LIVE_land71", live["accept_lengths"], live["branch_hit_k"],
                                  live["n"], step, srhw, args.n_boot, rng)
        # fern point gate on the live mean (preconditions wired through)
        live_out["fern_point_gate"] = measured_m16_to_official(
            float(live["accept_lengths"].mean()),
            live["branch_hit_k"] / live["n"], step, TAU["central"],
            ppl=live.get("ppl"), greedy_token_ids_captured=live.get("greedy_token_ids_captured"),
            step_is_roofline=False)

    # ---- top-line state ----
    if live_out is not None:
        gate_verdict = live_out["robust_verdict_99"]
        gate_state = "LIVE"
        gate_label = (f"LIVE land #71 E[T]={live_out['sample_mean_et']:.3f} (N={live['n']}) -> "
                      f"TPS CI99 [{live_out['tps_ci_bootstrap']['99']['tps_lo']:.1f}, "
                      f"{live_out['tps_ci_bootstrap']['99']['tps_hi']:.1f}] -> {gate_verdict}")
    else:
        gate_verdict = "ARMED"
        gate_state = "PENDING"
        gate_label = (f"ENVELOPE ARMED + VALIDATED (CI self-test "
                      f"{'PASS' if gate_ci_self_test_passes else 'FAIL'}); awaiting land #71's "
                      f"per-step a[1..N]. At N={n}, a robust-GREEN needs measured E[T] >= "
                      f"{min_et_1024_99:.3f} (99%) / {min_et_1024_95:.3f} (95%); the 5.207 "
                      f"ceiling clears it, anything in [{bars['bar_worst']:.3f}, "
                      f"{min_et_1024_95:.3f}] is INDETERMINATE at 1024 steps.")

    decision_input_line = (
        "M16-MEASURED-500-GATE CONFIDENCE ENVELOPE (decision input ONLY; does NOT authorize a "
        f"launch): wraps fern #142's point gate with the sampling CI. robust-GREEN iff the WHOLE "
        f"TPS CI clears 500 (E[T] CI lower bound > effective bar {bars['bar_worst']:.3f}). At the "
        f"measured step {step:.4f}+-{srhw*100:.2f}%, oracle E[T]=2.621 is robustly RED "
        f"(required-N {required_n_for_robust_500_verdict}, MOOT), the 5.207 ceiling robustly GREEN "
        f"(required-N {req_n_ceiling_99} @99%). Boundary 4.862 is INDETERMINATE at any finite N. "
        f"PRESCRIPTION for land #71: measure N >= {req_n_ceiling_99} steps AND E[T] >= "
        f"{min_et_1024_99:.3f} for a 99%-robust GREEN at N=1024.")

    out = {
        "primary_metric_name": "gate_ci_self_test_passes",
        "gate_ci_self_test_passes": gate_ci_self_test_passes,
        "test_metric_name": "required_n_for_robust_500_verdict",
        "required_n_for_robust_500_verdict": required_n_for_robust_500_verdict,
        "gate_state": gate_state,
        "gate_verdict": gate_verdict,
        "gate_label": gate_label,
        "land_measured_pending": land_pending,
        "decision_input_line": decision_input_line,
        "map": {
            "figure_of_merit": "official_TPS = K_cal * E[T] / step * tau",
            "K_cal": K_CAL, "measured_step": step, "step_rel_half_width": srhw,
            "step_band_is_assumed_pending_lawine136": True,
            "tau_band": TAU, "target_official": TARGET_OFFICIAL,
            "frontier_official": FRONTIER_OFFICIAL,
            "clear500_bar_central": bars["bar_central"],
            "clear500_bar_effective_worst": bars["bar_worst"],
        },
        "self_test": {
            "passes": bool(gate_ci_self_test_passes),
            "red_anchor_robust_RED": red_ok,
            "green_anchor_robust_GREEN": green_ok,
            "borderline_INDETERMINATE": border_ok,
            "fern142_point_anchors_reproduced": fern_point_ok,
            "fern142_anchor_red_tps": fst["anchor_red"]["gate_tps_central"],
            "fern142_anchor_green_tps": fst["anchor_green"]["gate_tps_central"],
            "note": ("CI self-test at the measured step 1.2182; fern #142's 271/538 point "
                     "anchors are reproduced at the 1.2127 roofline step (fern_self_test). "
                     "borderline = the exact clear-500 E[T] (official=500), which no finite N "
                     "can resolve -> INDETERMINATE by construction."),
        },
        "anchors": {"as_built_RED": red, "rho_optimal_GREEN": green, "boundary_INDETERMINATE": border},
        "required_n_test": {
            "oracle_point_E_T": ORACLE_E_T,
            "required_n_oracle_99": req_n_oracle_99,
            "required_n_oracle_95": red["required_n_for_verdict"]["95"],
            "interpretation": ("the oracle point is so far below the bar that ~a-few steps "
                               "already confirm RED -> the uncertainty lever is MOOT for the "
                               "oracle point; the live prescription is the deliverable."),
        },
        "handoff_land71": {
            "descending_regime_per_step_sd": sigma_descend,
            "effective_clear500_bar_et": bars["bar_worst"],
            "min_robust_green_et_at_N1024_99": min_et_1024_99,
            "min_robust_green_et_at_N1024_95": min_et_1024_95,
            "required_n_at_ceiling_5207_99": req_n_ceiling_99,
            "required_n_at_ceiling_5207_95": req_n_ceiling_95,
            "required_n_curve": handoff_curve,
            "prescription": ("on the live re-bench, measure a PER-STEP accept-length vector "
                             "a[1..N] (not just the mean). For a 99%-robust GREEN at N=1024, the "
                             f"measured E[T] must clear {min_et_1024_99:.3f}; nearer the 4.862 "
                             "boundary the required-N grows without bound, so a point estimate "
                             "that clears 4.862 by a hair is NOT a launch-safe GREEN."),
        },
        "live_gate": live_out,
        "banked_oracle": {
            "board": "20260614-100550-487", "package": "tree-488-pw-fp32-v0",
            "cumulative_ladder": ORACLE_CUM_LADDER, "E_T": ORACLE_E_T,
            "salvages": ORACLE_SALVAGES, "full": ORACLE_FULL, "steps": ORACLE_STEPS,
            "spine_only_E_T_1_plus_sumC": 1.0 + sum(ORACLE_CUM_LADDER),
            "salvage_residual": ORACLE_E_T - (1.0 + sum(ORACLE_CUM_LADDER)),
            "branch_hit_rho2": RHO2_BRANCH_HIT, "chain_reject_floor": CHAIN_REJECT_FLOOR,
        },
        "method": ("LOCAL CPU-only analytic + statistics; wraps fern #142's point gate (does NOT "
                   "duplicate it). Per-step law reconstructed EXACTLY from committed-path ladders; "
                   "CLT + nonparametric bootstrap E[T] CI, Wilson branch-hit CI, step+tau "
                   "propagation. Produces a launch-READINESS decision input only; does NOT "
                   "authorize a launch. Greedy identity untouched."),
        "provenance": ("extends fern #142 m16_measured_500_gate (point core, K_cal, tau, "
                       "271/538 anchors) + fern #100 lever_composition (K_cal=125.268) + wirbel "
                       "#135 bug2_salvage_descent DP (descending-regime committed-path ladder) + "
                       "lawine #136 measured step 1.2182. Oracle samples: board "
                       "20260614-100550-487 (tree-488-pw-fp32-v0)."),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=_jd)

    # ------------------------------- console -------------------------------
    print(f"\n[DP] descending-regime per-step sigma = {sigma_descend:.4f} "
          f"(CV {green['per_step_cv']:.3f}); oracle per-step sigma = {red['per_step_sd']:.4f}", flush=True)
    print(f"\n[ANCHORS] (measured step {step:.4f}, N={n}, bootstrap CI):")
    for r in (red, border, green):
        c99 = r["tps_ci_bootstrap"]["99"]
        print(f"  E[T]={r['point_et']:.3f} -> TPS {r['official_tps_point']:7.1f}  "
              f"CI99 [{c99['tps_lo']:6.1f}, {c99['tps_hi']:6.1f}]  -> "
              f"{r['robust_verdict_99']:>14s}  ({r['name']})", flush=True)
    print(f"\n[SELF-TEST] RED={red_ok}  GREEN={green_ok}  BORDERLINE-INDET={border_ok}  "
          f"fern-point(271/538)={fern_point_ok}  -> gate_ci_self_test_passes="
          f"{gate_ci_self_test_passes}", flush=True)
    print(f"\n[BRANCH-HIT] rho2={RHO2_BRANCH_HIT:.4f} vs chain-reject floor {CHAIN_REJECT_FLOOR}: "
          f"Wilson99 [{red['branch_hit']['wilson_ci']['99']['lo']:.4f}, "
          f"{red['branch_hit']['wilson_ci']['99']['hi']:.4f}]  excludes-floor="
          f"{red['branch_hit']['excludes_chain_reject_floor_99']}  "
          f"required-N-to-separate(99%)={red['branch_hit']['required_n_to_separate_from_floor']['99']}",
          flush=True)
    print(f"\n[TEST] required_n_for_robust_500_verdict (oracle point E[T]=2.621) = "
          f"{required_n_for_robust_500_verdict}  (MOOT -- oracle is decisively RED)", flush=True)
    print(f"\n[HAND-OFF land #71] descending sigma {sigma_descend:.3f}; "
          f"min robust-GREEN E[T] @N=1024: {min_et_1024_99:.3f} (99%) / {min_et_1024_95:.3f} (95%); "
          f"required-N @ceiling 5.207: {req_n_ceiling_99} (99%) / {req_n_ceiling_95} (95%)", flush=True)
    print(f"  {'measured E[T]':>13s} {'TPS pt':>8s} {'req-N 95':>9s} {'req-N 99':>9s}")
    for row in handoff_curve:
        n95 = row["required_n_95"]["required_n"]
        n99 = row["required_n_99"]["required_n"]
        print(f"  {row['measured_et']:13.3f} {row['official_tps_point']:8.1f} "
              f"{str(n95) if n95 is not None else 'unbounded':>9s} "
              f"{str(n99) if n99 is not None else 'unbounded':>9s}", flush=True)
    print(f"\n[GATE] {gate_verdict} / {gate_state} -- {gate_label}", flush=True)
    print(f"\nwrote {args.out}", flush=True)

    # ------------------------------- W&B -------------------------------
    if args.wandb and not args.no_wandb:
        try:
            log_wandb(args, out, red, green, border, handoff_curve)
        except Exception as e:  # noqa: BLE001
            print(f"[envelope] W&B logging failed (non-fatal): {e!r}", flush=True)


def log_wandb(args, out, red, green, border, handoff_curve):
    import wandb
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     name=args.wandb_name, group=args.wandb_group, job_type="analysis",
                     config={"gate": "measured-gate-confidence-envelope",
                             "method": "cpu-analytic-CI-wraps-fern142-point-gate",
                             "K_cal": K_CAL, "measured_step": args.measured_step,
                             "step_rel_half_width": args.step_rel_half_width,
                             "n_steps": args.n_steps, "n_boot": args.n_boot,
                             "tau_low": TAU["low"], "tau_central": TAU["central"],
                             "target_official": TARGET_OFFICIAL,
                             "clear500_bar_central": out["map"]["clear500_bar_central"],
                             "clear500_bar_effective_worst": out["map"]["clear500_bar_effective_worst"],
                             "branch_hit_rho2": RHO2_BRANCH_HIT,
                             "chain_reject_floor": CHAIN_REJECT_FLOOR})
    s = wandb.summary
    s["gate_ci_self_test_passes"] = out["gate_ci_self_test_passes"]
    rn = out["required_n_for_robust_500_verdict"]
    s["required_n_for_robust_500_verdict"] = _finite(rn, -1) if rn is not None else -1
    s["gate_state"] = out["gate_state"]
    s["gate_verdict"] = out["gate_verdict"]
    s["land_measured_pending"] = int(out["land_measured_pending"])
    # anchor verdicts + CI bounds
    for tag, r in (("red", red), ("green", green), ("border", border)):
        s[f"anchor_{tag}_point_et"] = r["point_et"]
        s[f"anchor_{tag}_verdict_99"] = r["robust_verdict_99"]
        s[f"anchor_{tag}_tps_ci99_lo"] = r["tps_ci_bootstrap"]["99"]["tps_lo"]
        s[f"anchor_{tag}_tps_ci99_hi"] = r["tps_ci_bootstrap"]["99"]["tps_hi"]
        s[f"anchor_{tag}_et_ci99_lo"] = r["et_ci_lo_99"]
        s[f"anchor_{tag}_et_ci99_hi"] = r["et_ci_hi_99"]
        s[f"anchor_{tag}_per_step_sd"] = r["per_step_sd"]
    s["descending_per_step_sd"] = out["handoff_land71"]["descending_regime_per_step_sd"]
    s["min_robust_green_et_at_N1024_99"] = out["handoff_land71"]["min_robust_green_et_at_N1024_99"]
    s["min_robust_green_et_at_N1024_95"] = out["handoff_land71"]["min_robust_green_et_at_N1024_95"]
    s["required_n_at_ceiling_5207_99"] = _finite(out["handoff_land71"]["required_n_at_ceiling_5207_99"], -1)
    s["required_n_at_ceiling_5207_95"] = _finite(out["handoff_land71"]["required_n_at_ceiling_5207_95"], -1)
    bh = red["branch_hit"]
    s["branch_hit_wilson99_lo"] = bh["wilson_ci"]["99"]["lo"]
    s["branch_hit_wilson99_hi"] = bh["wilson_ci"]["99"]["hi"]
    s["branch_hit_excludes_floor_99"] = int(bh["excludes_chain_reject_floor_99"])
    s["branch_hit_required_n_to_separate_99"] = _finite(
        bh["required_n_to_separate_from_floor"]["99"], -1)
    s["decision_input_line"] = out["decision_input_line"]

    # anchor table
    at = wandb.Table(columns=["anchor", "point_et", "tps_point", "et_ci99_lo", "et_ci99_hi",
                              "tps_ci99_lo", "tps_ci99_hi", "verdict_99", "per_step_sd"])
    for tag, r in (("RED_oracle", red), ("BORDERLINE", border), ("GREEN_ceiling", green)):
        c = r["tps_ci_bootstrap"]["99"]
        at.add_data(tag, r["point_et"], r["official_tps_point"], r["et_ci_lo_99"], r["et_ci_hi_99"],
                    c["tps_lo"], c["tps_hi"], r["robust_verdict_99"], r["per_step_sd"])
    wandb.log({"ci_anchors": at})
    # required-N hand-off curve
    ct = wandb.Table(columns=["measured_et", "tps_point", "required_n_95", "required_n_99"])
    for row in handoff_curve:
        n95 = row["required_n_95"]["required_n"]
        n99 = row["required_n_99"]["required_n"]
        ct.add_data(row["measured_et"], row["official_tps_point"],
                    n95 if n95 is not None else -1, n99 if n99 is not None else -1)
    wandb.log({"required_n_curve": ct})
    print(f"\nW&B run: {run.id}  ({run.url})", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
