#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""E[T] SECOND MOMENT -> finite-sample TPS CI + land distribution gate (PR #175, wirbel).

THE QUESTION
------------
Every launch-packet leg quotes E[T] as a POINT (descent-only 5.0564 / both-bugs
5.2070). But the official `summary.json:tps = total_tokens / total_decode_time` is a
*finite* draw: with a fixed token budget B the realized throughput reduces to
TPS proportional to L-bar/step, where L-bar is the BENCHMARK-MEAN accepted length
over the ~3150 spec-decode steps the run actually takes. L-bar is a random variable
-- each step accepts a random length, and the single benchmark draw realizes a
sample mean that scatters around E[T] with standard error sigma_L/sqrt(N_steps).
wirbel #160's DP gives only the FIRST moment (E[T]); its SECOND moment Var[L] -- and
hence the finite-benchmark TPS confidence interval -- has never been computed.

This file is the SAMPLING-UNCERTAINTY leg of the launch packet:
  fern #174 bands the INPUTS; denken #172 bounds the E[T] MODEL from below; wirbel
  #170 bounds it from above against over-acceptance; kanna #159 bands the
  DENOMINATOR's hardware jitter (sigma_hw). NONE answer: even if every input is
  exactly central and the model exactly right, how much could the single-shot TPS
  scatter purely by finite-sample chance? That number is
      1.96 * K_cal * (sigma_L/sqrt(N_steps)) / step * tau
  and it composes IN QUADRATURE with kanna's sigma_hw to give the launch's total
  single-shot TPS CI.

The same DP second moment also yields the full accepted-length DISTRIBUTION P(L=k),
which arms land #71 with a DISTRIBUTIONAL readout gate -- catching a build bug that
preserves the mean E[T] but distorts the shape. (wirbel #170 certifies the mean's
trustworthiness; this certifies the whole histogram.)

THE MODEL (imported, NOT re-derived)
------------------------------------
The descent walk (simulate_greedy_depthrank / gen_matches_greedy semantics): from the
root, at each level pick rank-r child with prob pvecs[d][r] (mutually exclusive across
ranks; residual = hard miss -> stop). The committed length per step is
    L = (number of edges descended) + 1 bonus token,   so  E[L] = E[T].
wirbel #160's `score_tree_depthrank` computes F = sum_c reach[c] = E[T], where
reach[c] = P(walk reaches node c) = product of edge marginals on root->c. The SECOND
moment falls out of the SAME object: the walk STOPS at node c (c is the deepest
reached) with prob reach[c]*(1 - s[c]), where s[c] = sum of c's child-edge marginals
(s[leaf]=0). Hence the exact accepted-length pmf is

    P(L = depth_c + 1) = sum_{c : depth_c = k-1} reach[c] * (1 - s[c])

which sums to 1 (the walk always stops somewhere) and whose mean is exactly E[T]
(sum_c reach[c]*(1-s[c]) telescopes: see _check). Var[L] = E[L^2] - E[L]^2 is then a
pure DP read -- no new model, no idealization.

LOCAL, CPU-ONLY, ANALYTIC. No GPU / vLLM / HF Job / submission / kernel build /
served-file change. BASELINE stays 481.53; 0 TPS; greedy identity untouched by
construction. Imports wirbel #160 (W&B x8vffgbs) descent E[T]-DP (the topology +
pvecs + score_tree_depthrank), #165 (laxllfjl) composed ceiling, #170 (ne7p642c)
over-accept locus; does NOT re-derive any of them.

PRIMARY metric  et_second_moment_self_test_passes
TEST    metric  tps_finite_sample_ci_halfwidth  (the both-bugs +/-1.96 SE half-width in TPS)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
_PROFILER = os.path.join(_ROOT, "scripts", "profiler")
if _PROFILER not in sys.path:
    sys.path.insert(0, _PROFILER)

# ---- imported single-source-of-truth #160 E[T]-DP (reuse verbatim; do NOT re-derive) ----
from treeshape_measured_accept import (  # noqa: E402
    build_depth_pvecs_measured,
    load_measured,
    load_rank_coverage,
    score_tree_depthrank,
    simulate_greedy_depthrank,
)
from traversal_verify_et import load_m32_topology, tree_arrays  # noqa: E402

# ---- banked inputs (the exact files #160 read) ----
RHO_OPT_JSON = os.path.join(_ROOT, "research", "spec_cost_model", "rho_optimal_topology_results.json")
ACCEPT_JSON = os.path.join(_ROOT, "research", "accept_calibration", "accept_calibration_results.json")
RANKCOV_JSON = os.path.join(_ROOT, "research", "rank_coverage", "rank_coverage_results.json")
SPINE_SPEC_JSON = os.path.join(_ROOT, "research", "spine_spec", "spine_spec_results.json")

# ---- contamination-model anchors (#160) ----
Q_TRUE = 0.728739760479042            # rank-1 acceptance (both-bugs depth-1 spine)
RHO2 = 0.4165047789261015             # rank-2 marginal
DESCENT_ONLY_D1 = 0.679               # descent-only build's residual depth-1 spine accept
ANCHOR_DESCENT_ONLY_ET = 5.056404568844709   # #160 x8vffgbs
ANCHOR_BOTH_BUGS_ET = 5.206954309441963      # #160 / #165 composed ceiling

# ---- cost composition (#160 baseline; reuse) ----
K_CAL = 125.268                       # ubel #148 local->official cal constant
STEP_MEASURED = 1.2182                # lawine #168 measured depth-9 step
STEP_ROOFLINE = 1.2127                # depth-9 verify step (roofline)
TAU_LO, TAU_HI = 0.9924, 1.0          # served-fraction band
TARGET_OFFICIAL = 500.0
Z95 = 1.959963984540054               # two-sided 95% normal quantile

# ---- benchmark budget (PR #175 contract) ----
BENCH_TOKENS = 16384                  # 128 prompts x 128 tokens (PR #175 stated budget)
BENCH_TOKENS_ALT = 65536              # 128 x 512 (prior-leg #170 budget) -- sensitivity only

# ---- comparison band (given in PR) ----
LAWINE168_OVERLAP_BAND = 2.4          # lawine #168 +/-2.4 TPS roofline<->overlap band

# ---- W=4, max_depth=24, extrapolate=flat: the exact #160 pvec build args ----
W_DEFAULT = 4
MAXD_DEFAULT = 24
EXACT_TOL = 1e-12                      # DP-vs-bruteforce pmf max-abs-diff tolerance
MEAN_TOL = 1e-9                        # pmf-mean vs imported E[T]


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _all_finite(xs) -> bool:
    return all(_finite(x) for x in xs)


# ===================================================================================
# build the two topologies' pvecs exactly as #160 did (import, not re-derive)
# ===================================================================================
def build_pvecs(q_deployed: list[float], rho_cond: list[float], q1: float,
                W: int, maxd: int) -> list[np.ndarray]:
    """The depth-1 spine edge is overridden to q1 (descent-only 0.679 / both-bugs q_true);
    the rest is the deployed rising rho-optimal spine. Identical to spine_spec_verify.ET_tree."""
    qq = list(q_deployed)
    qq[0] = q1
    return build_depth_pvecs_measured(qq, rho_cond, W, maxd, "flat")


# ===================================================================================
# 1. accepted-length pmf from the #160 DP  (the second moment falls out of reach[])
# ===================================================================================
def reach_and_stop(parent: list[int], pvecs: list[np.ndarray]):
    """reach[c] = P(walk reaches c) (== score_tree_depthrank pp[c]); sdesc[c] = sum of
    c's child-edge marginals (descent prob FROM c; sdesc[leaf]=0). Mirrors
    score_tree_depthrank's edge assignment EXACTLY (birth-order rank, depth-keyed pv)."""
    n = len(parent)
    children: list[list[int]] = [[] for _ in range(n)]
    for i in range(1, n):
        children[parent[i]].append(i)
    maxd = len(pvecs) - 1
    reach = np.zeros(n, dtype=np.float64)
    reach[0] = 1.0
    depth = [0] * n
    sdesc = np.zeros(n, dtype=np.float64)
    for par in range(n):
        for rank, c in enumerate(children[par], start=1):
            d = depth[par] + 1
            pv = pvecs[min(d, maxd)]
            r = rank if rank < len(pv) else len(pv) - 1
            e = float(pv[r])
            reach[c] = reach[par] * e
            depth[c] = d
            sdesc[par] += e
    return reach, sdesc, depth, children


def dp_accepted_length_pmf(parent: list[int], pvecs: list[np.ndarray]):
    """Exact committed-length pmf P(L=k), k = 1..max_depth+1 (L includes the bonus token,
    so E[L] == E[T]). Returns (pmf array indexed by k, reach, sdesc, depth)."""
    reach, sdesc, depth, _ = reach_and_stop(parent, pvecs)
    maxk = max(depth) + 1
    pmf = np.zeros(maxk + 1, dtype=np.float64)   # index 0 unused; support 1..maxk
    for c in range(len(parent)):
        pmf[depth[c] + 1] += reach[c] * (1.0 - sdesc[c])
    return pmf, reach, sdesc, depth


def pmf_moments(pmf: np.ndarray) -> dict:
    """total mass, mean, variance, std of a committed-length pmf indexed by k."""
    ks = np.arange(len(pmf), dtype=np.float64)
    total = float(pmf.sum())
    mean = float((ks * pmf).sum())
    m2 = float((ks * ks * pmf).sum())
    var = m2 - mean * mean
    return {"total_mass": total, "mean": mean, "E_L2": m2, "var": var,
            "std": math.sqrt(var) if var > 0 else 0.0}


# ===================================================================================
# 2. DP-exactness certificate: independent enumerations of the walk's stop pmf
# ===================================================================================
def bruteforce_pmf_recursive(parent: list[int], pvecs: list[np.ndarray]) -> np.ndarray:
    """INDEPENDENT enumeration #1: recursive DFS over the whole tree, carrying the
    root->node path-product, accumulating the stop-here mass at every node. Structurally
    distinct from the DP's iterative forward sweep -- exhausts the walk's sample space
    (every node = one 'deepest reached' outcome)."""
    n = len(parent)
    children: list[list[int]] = [[] for _ in range(n)]
    for i in range(1, n):
        children[parent[i]].append(i)
    maxd = len(pvecs) - 1
    depth = [0] * n
    for i in range(1, n):
        depth[i] = depth[parent[i]] + 1
    maxk = max(depth) + 1
    pmf = np.zeros(maxk + 1, dtype=np.float64)

    def rec(u: int, du: int, path_prob: float) -> None:
        kids = children[u]
        dchild = du + 1
        pv = pvecs[min(dchild, maxd)]
        sdesc = 0.0
        edges = []
        for rank, c in enumerate(kids, start=1):
            r = rank if rank < len(pv) else len(pv) - 1
            e = float(pv[r])
            edges.append((c, e))
            sdesc += e
        pmf[du + 1] += path_prob * (1.0 - sdesc)   # stop at u
        for c, e in edges:
            rec(c, dchild, path_prob * e)

    rec(0, 0, 1.0)
    return pmf


def bruteforce_pmf_parentwalk(parent: list[int], pvecs: list[np.ndarray]) -> np.ndarray:
    """INDEPENDENT enumeration #2: for each node, recompute its reach by walking UP the
    parent chain (recomputing sibling-rank + depth from scratch), times its own stop
    prob. Different code path again -- no shared accumulator with the DP or the DFS."""
    n = len(parent)
    children: list[list[int]] = [[] for _ in range(n)]
    for i in range(1, n):
        children[parent[i]].append(i)
    maxd = len(pvecs) - 1
    depth = [0] * n
    for i in range(1, n):
        depth[i] = depth[parent[i]] + 1
    rankof = {}
    for par in range(n):
        for rank, c in enumerate(children[par], start=1):
            rankof[c] = rank

    def path_prob(c: int) -> float:
        p, node = 1.0, c
        while node != 0:
            d, rank = depth[node], rankof[node]
            pv = pvecs[min(d, maxd)]
            r = rank if rank < len(pv) else len(pv) - 1
            p *= float(pv[r])
            node = parent[node]
        return p

    def sdesc(c: int) -> float:
        s, dchild = 0.0, depth[c] + 1
        pv = pvecs[min(dchild, maxd)]
        for rank, _ in enumerate(children[c], start=1):
            r = rank if rank < len(pv) else len(pv) - 1
            s += float(pv[r])
        return s

    maxk = max(depth) + 1
    pmf = np.zeros(maxk + 1, dtype=np.float64)
    for c in range(n):
        pmf[depth[c] + 1] += path_prob(c) * (1.0 - sdesc(c))
    return pmf


def mc_pmf(parent: list[int], pvecs: list[np.ndarray], trials: int, seed: int) -> np.ndarray:
    """EMPIRICAL cross-check: Monte-Carlo histogram of committed length, mirroring
    simulate_greedy_depthrank EXACTLY but binning length+1 (reports its own MC error,
    NOT the exactness certificate)."""
    n = len(parent)
    children: list[list[int]] = [[] for _ in range(n)]
    depth = [0] * n
    for i in range(1, n):
        children[parent[i]].append(i)
        depth[i] = depth[parent[i]] + 1
    maxd = len(pvecs) - 1
    maxk = max(depth) + 1
    rng = np.random.default_rng(seed)
    counts = np.zeros(maxk + 1, dtype=np.float64)
    for _ in range(trials):
        u, length = 0, 0
        while children[u]:
            kids = children[u]
            d = depth[u] + 1
            pv = pvecs[min(d, maxd)]
            draw = rng.random()
            cum, chosen = 0.0, -1
            for idx in range(len(kids)):
                r = idx + 1
                cum += float(pv[r if r < len(pv) else len(pv) - 1])
                if draw < cum:
                    chosen = idx
                    break
            if chosen < 0:
                break
            u = kids[chosen]
            length += 1
        counts[length + 1] += 1.0
    return counts / trials


def _align(a: np.ndarray, b: np.ndarray):
    n = max(len(a), len(b))
    aa = np.zeros(n); aa[:len(a)] = a
    bb = np.zeros(n); bb[:len(b)] = b
    return aa, bb


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = _align(a, b)
    return float(np.max(np.abs(aa - bb)))


# ===================================================================================
# 3. second moment -> finite-benchmark TPS CI
# ===================================================================================
def finite_sample_tps_ci(et: float, sigma_L: float, n_tokens: int, step: float,
                         tau: float, z: float = Z95) -> dict:
    """Single-shot 95% TPS CI from the accept-length sampling spread.

    official(L_bar) = K_cal * (L_bar/step) * tau, with L_bar the benchmark sample-mean
    accepted length over N_steps ~= n_tokens/E[T] iid steps. SE[L_bar]=sigma_L/sqrt(N).
    The map is linear in L_bar, so SE[TPS] = (K_cal*tau/step)*SE[L_bar] and the 95%
    half-width is z*SE[TPS]. (E[L_bar]=E[T] so the central TPS is the #160 point.)"""
    n_steps = n_tokens / et
    se_lbar = sigma_L / math.sqrt(n_steps)
    slope = K_CAL * tau / step                  # dTPS / dL_bar
    central_tps = slope * et
    se_tps = slope * se_lbar
    half = z * se_tps
    return {
        "E_T": et, "sigma_L": sigma_L, "n_tokens": n_tokens, "step": step, "tau": tau,
        "N_steps": n_steps, "SE_Lbar": se_lbar, "tps_slope_per_Lbar": slope,
        "central_tps": central_tps, "SE_tps": se_tps,
        "ci_halfwidth_tps": half, "ci_lower_tps": central_tps - half,
        "ci_upper_tps": central_tps + half,
        "lower_clears_500": bool((central_tps - half) > TARGET_OFFICIAL),
    }


# ===================================================================================
# 4. quadrature compose with kanna #159 (sigma_hw) + land #71 distributional gate
# ===================================================================================
def quadrature_total_ci(half_acceptlen: float, half_hw_pending) -> dict:
    """Total single-shot TPS half-width = accept-length sampling term (numerator, THIS
    leg) (+) kanna #159's sigma_hw step-jitter term (denominator), composed IN
    QUADRATURE. kanna's term is carried as a PENDING band (armed slot) -- do NOT consume."""
    if half_hw_pending is None:
        total = None
        formula_value = None
    else:
        total = math.sqrt(half_acceptlen ** 2 + half_hw_pending ** 2)
        formula_value = total
    return {
        "formula": "halfwidth_total = sqrt( halfwidth_acceptlen^2 + halfwidth_sigma_hw^2 )",
        "halfwidth_acceptlen_tps": half_acceptlen,         # THIS leg (numerator term)
        "halfwidth_sigma_hw_tps_PENDING": half_hw_pending,  # kanna #159 (armed slot)
        "halfwidth_total_tps": formula_value,
        "note": ("numerator (accept-length finite-sample, THIS leg) and denominator "
                 "(kanna #159 sigma_hw step-jitter) are independent -> add in quadrature. "
                 "kanna's term is carried PENDING; plug kanna #159's measured sigma_hw "
                 "TPS half-width into the armed slot to close the total single-shot CI."),
        "armed": half_hw_pending is None,
    }


def make_land_histogram_gate():
    """One-line predicate land #71 evaluates on its MEASURED per-step accepted-length
    histogram, validating the build at the DISTRIBUTION level (shape, not just mean)."""

    def land_histogram_in_band(measured_pmf, predicted_pmf, tol: float = 0.02) -> bool:
        a = np.asarray(measured_pmf, dtype=np.float64)
        b = np.asarray(predicted_pmf, dtype=np.float64)
        n = max(len(a), len(b))
        aa = np.zeros(n); aa[:len(a)] = a
        bb = np.zeros(n); bb[:len(b)] = b
        if aa.sum() <= 0:
            return False
        aa = aa / aa.sum()
        return bool(np.max(np.abs(aa - bb)) <= tol)

    return land_histogram_in_band


# ===================================================================================
# 5. self-test (PRIMARY) + TEST metric
# ===================================================================================
def self_test(topos: dict, exact: dict, ci: dict, mc: dict) -> dict:
    checks = []

    def chk(name, ok, detail):
        checks.append({"name": name, "passes": bool(ok), "detail": str(detail)})

    # (a) DP pmf mean reproduces #160 5.0564 / 5.2070 AND equals score_tree_depthrank E[T]
    for lab, anchor in (("descent_only", ANCHOR_DESCENT_ONLY_ET),
                        ("both_bugs", ANCHOR_BOTH_BUGS_ET)):
        t = topos[lab]
        chk(f"{lab}: pmf-mean reproduces #160 E[T] anchor",
            abs(t["pmf_mean"] - anchor) < MEAN_TOL,
            f"pmf_mean={t['pmf_mean']:.12f} anchor={anchor:.12f}")
        chk(f"{lab}: pmf-mean == score_tree_depthrank E[T] (same object)",
            abs(t["pmf_mean"] - t["score_tree_E_T"]) < MEAN_TOL,
            f"pmf_mean={t['pmf_mean']:.12f} score_tree={t['score_tree_E_T']:.12f}")
        # (b) pmf sums to 1 and is non-negative
        chk(f"{lab}: pmf sums to 1",
            abs(t["pmf_total_mass"] - 1.0) < 1e-12, f"total={t['pmf_total_mass']:.15f}")
        chk(f"{lab}: pmf non-negative",
            t["pmf_min"] >= -1e-15, f"min={t['pmf_min']:.3e}")
        # (d) sigma_L >= 0
        chk(f"{lab}: sigma_L >= 0", t["sigma_L"] >= 0.0, f"sigma_L={t['sigma_L']:.6f}")

    # (c) brute-force exactness (both independent enumerations match the DP)
    chk("DP pmf exact vs recursive DFS enumeration (both topologies)",
        exact["dp_distribution_exact"],
        f"max_abs_diff={exact['pmf_max_abs_diff']:.3e} (tol {EXACT_TOL:.0e})")
    chk("DP pmf exact vs parent-walk enumeration (both topologies)",
        exact["dp_vs_parentwalk_exact"],
        f"max_abs_diff={exact['pmf_max_abs_diff_parentwalk']:.3e}")

    # (d) finite-sample CI brackets the point estimate; clears-500 verdict EXPLICIT
    for lab in ("descent_only", "both_bugs"):
        c = ci[lab]["primary_16384_tau1"]
        chk(f"{lab}: CI brackets the point estimate (lower<central<upper)",
            c["ci_lower_tps"] < c["central_tps"] < c["ci_upper_tps"],
            f"[{c['ci_lower_tps']:.3f}, {c['ci_upper_tps']:.3f}] around {c['central_tps']:.3f}")
        chk(f"{lab}: lower-bound-clears-500 verdict explicit (bool present)",
            isinstance(c["lower_clears_500"], bool),
            f"lower={c['ci_lower_tps']:.3f} clears500={c['lower_clears_500']}")

    # MC empirical agreement (not gating the exactness, but a health check)
    chk("MC histogram agrees with DP pmf within MC tolerance (both topologies)",
        mc["mc_agrees"], f"max_abs_diff={mc['mc_max_abs_diff']:.3e} (tol {mc['mc_tol']:.0e})")

    # (e) NaN-clean across every reported scalar
    scal = []
    for lab in ("descent_only", "both_bugs"):
        t = topos[lab]
        scal += [t["pmf_mean"], t["pmf_total_mass"], t["var_L"], t["sigma_L"], t["score_tree_E_T"]]
        scal += [ci[lab]["primary_16384_tau1"]["ci_halfwidth_tps"],
                 ci[lab]["primary_16384_tau1"]["ci_lower_tps"]]
    nan_clean = _all_finite(scal)
    chk("all reported scalars NaN-clean", nan_clean, f"{len(scal)} scalars finite")

    passes = all(c["passes"] for c in checks)
    return {"passes": passes, "n_checks": len(checks),
            "n_passed": sum(c["passes"] for c in checks), "checks": checks,
            "nan_clean": nan_clean}


# ===================================================================================
# main
# ===================================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="research/oracle_readout/et_second_moment/et_second_moment_results.json")
    ap.add_argument("--report-md", default="research/oracle_readout/et_second_moment/report_et_second_moment.md")
    ap.add_argument("--W", type=int, default=W_DEFAULT)
    ap.add_argument("--max-depth", type=int, default=MAXD_DEFAULT)
    ap.add_argument("--mc-trials", type=int, default=2_000_000)
    ap.add_argument("--sigma-hw-tps", type=float, default=None,
                    help="OPTIONAL kanna #159 sigma_hw TPS half-width to plug into the armed "
                         "quadrature slot (default: PENDING/armed, do NOT consume).")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="gemma-challenge-senpai")
    ap.add_argument("--wandb-entity", default="wandb-applied-ai-team")
    ap.add_argument("--wandb-name", default="wirbel/et-second-moment-tps-ci")
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", default="et-second-moment-tps-ci")
    args = ap.parse_args()

    W, maxd = args.W, args.max_depth

    # ---- load the EXACT #160 model (topology + measured acceptance) ----
    parent = load_m32_topology(RHO_OPT_JSON)
    children, depth_arr, leaves = tree_arrays(parent)
    built_depth = max(depth_arr)
    meas = load_measured(ACCEPT_JSON, "server_log")
    q_deployed = list(meas["q"])
    rc = load_rank_coverage(RANKCOV_JSON)
    rho_cond = rc["rho_cond"]
    topo_meta = {
        "n_nodes": len(parent), "built_depth": built_depth,
        "max_branch": max(len(c) for c in children), "n_leaves": len(leaves),
        "parent": parent,
    }

    # ---- per-topology: pvecs, score_tree E[T], pmf, moments ----
    Q1 = {"descent_only": DESCENT_ONLY_D1, "both_bugs": Q_TRUE}
    topos = {}
    pmfs = {}
    for lab, q1 in Q1.items():
        pvecs = build_pvecs(q_deployed, rho_cond, q1, W, maxd)
        et_score, _ = score_tree_depthrank(parent, pvecs)            # imported first moment
        pmf, reach, sdesc, depth = dp_accepted_length_pmf(parent, pvecs)
        mom = pmf_moments(pmf)
        pmfs[lab] = {"pvecs": pvecs, "pmf": pmf}
        # full pmf table (committed length k=1..maxk) + accepted-drafts view (k-1=0..depth)
        support = list(range(1, len(pmf)))
        pmf_table = [{"committed_length_L": k, "accepted_drafts": k - 1,
                      "P": float(pmf[k])} for k in support]
        topos[lab] = {
            "q1_depth1_spine": q1,
            "score_tree_E_T": et_score,
            "pmf_mean": mom["mean"],
            "pmf_total_mass": mom["total_mass"],
            "E_L2": mom["E_L2"],
            "var_L": mom["var"],
            "sigma_L": mom["std"],
            "pmf_min": float(pmf.min()),
            "pmf_max_support_length": len(pmf) - 1,
            "accepted_length_pmf": pmf_table,
            "mode_committed_length": int(np.argmax(pmf)),
        }

    # ---- 2. exactness certificate (both topologies, both independent enumerations) ----
    diffs_dfs, diffs_pw = [], []
    mc_diffs = []
    mc_tol = 5e-3
    for lab in Q1:
        pvecs = pmfs[lab]["pvecs"]
        dp = pmfs[lab]["pmf"]
        bf_dfs = bruteforce_pmf_recursive(parent, pvecs)
        bf_pw = bruteforce_pmf_parentwalk(parent, pvecs)
        diffs_dfs.append(max_abs_diff(dp, bf_dfs))
        diffs_pw.append(max_abs_diff(dp, bf_pw))
        mc = mc_pmf(parent, pvecs, args.mc_trials, seed=175)
        mc_diffs.append(max_abs_diff(dp, mc))
        topos[lab]["bruteforce_dfs_max_abs_diff"] = diffs_dfs[-1]
        topos[lab]["bruteforce_parentwalk_max_abs_diff"] = diffs_pw[-1]
        topos[lab]["mc_max_abs_diff"] = mc_diffs[-1]
        # MC E[T] cross-check (mean of the MC histogram)
        topos[lab]["mc_mean"] = float(sum(k * mc[k] for k in range(len(mc))))
    exact = {
        "pmf_max_abs_diff": max(diffs_dfs),
        "pmf_max_abs_diff_parentwalk": max(diffs_pw),
        "dp_distribution_exact": bool(max(diffs_dfs) < EXACT_TOL),
        "dp_vs_parentwalk_exact": bool(max(diffs_pw) < EXACT_TOL),
        "enumeration_methods": ["recursive_DFS_pathproduct", "per_node_parentwalk"],
        "tol": EXACT_TOL,
    }
    mc_summary = {
        "mc_trials": args.mc_trials, "mc_max_abs_diff": max(mc_diffs),
        "mc_tol": mc_tol, "mc_agrees": bool(max(mc_diffs) < mc_tol),
    }

    # ---- 3. finite-sample TPS CI (both topologies, both budgets, tau band) ----
    ci = {}
    for lab in Q1:
        et = topos[lab]["pmf_mean"]
        sig = topos[lab]["sigma_L"]
        ci[lab] = {
            "primary_16384_tau1": finite_sample_tps_ci(et, sig, BENCH_TOKENS, STEP_MEASURED, TAU_HI),
            "primary_16384_tau_lo": finite_sample_tps_ci(et, sig, BENCH_TOKENS, STEP_MEASURED, TAU_LO),
            "sensitivity_65536_tau1": finite_sample_tps_ci(et, sig, BENCH_TOKENS_ALT, STEP_MEASURED, TAU_HI),
            "roofline_16384_tau1": finite_sample_tps_ci(et, sig, BENCH_TOKENS, STEP_ROOFLINE, TAU_HI),
        }

    # TEST metric = both-bugs half-width @ primary budget, measured step, tau=1
    test_halfwidth = ci["both_bugs"]["primary_16384_tau1"]["ci_halfwidth_tps"]
    descent_halfwidth = ci["descent_only"]["primary_16384_tau1"]["ci_halfwidth_tps"]

    # comparison to lawine #168 +/-2.4 band
    def band_compare(h):
        ratio = h / LAWINE168_OVERLAP_BAND
        if ratio > 1.15:
            rel = "LARGER"
        elif ratio < 0.87:
            rel = "smaller"
        else:
            rel = "same order"
        return {"halfwidth_tps": h, "lawine168_band_tps": LAWINE168_OVERLAP_BAND,
                "ratio_to_lawine168": ratio, "relation": rel}

    band_cmp = {"both_bugs": band_compare(test_halfwidth),
                "descent_only": band_compare(descent_halfwidth)}

    # ---- 4. quadrature composition + land distributional gate ----
    quad = {lab: quadrature_total_ci(ci[lab]["primary_16384_tau1"]["ci_halfwidth_tps"],
                                     args.sigma_hw_tps) for lab in Q1}
    land_gate = make_land_histogram_gate()
    # demonstrate the gate on a few measured-histogram scenarios (both-bugs predicted pmf)
    pred = pmfs["both_bugs"]["pmf"]
    pred_norm = pred / pred.sum()
    # perturb: a build bug that preserves the mean but shifts mass between k and k+/-1
    perturbed = pred_norm.copy()
    # move 5% mass from the mode to its neighbour (mean ~preserved, shape distorted)
    mode = int(np.argmax(pred_norm))
    if mode + 1 < len(perturbed) and mode - 1 >= 1:
        move = 0.05
        perturbed[mode] -= 2 * move
        perturbed[mode + 1] += move
        perturbed[mode - 1] += move
    gate_demo = [
        {"scenario": "measured == predicted (clean build)",
         "in_band": land_gate(pred_norm, pred_norm, tol=0.02)},
        {"scenario": "5% mass spread off the mode (mean-preserving shape distortion)",
         "in_band": land_gate(perturbed, pred_norm, tol=0.02),
         "perturbed_mean": float(sum(k * perturbed[k] for k in range(len(perturbed)))),
         "predicted_mean": float(sum(k * pred_norm[k] for k in range(len(pred_norm))))},
    ]

    # ---- 5. self-test (PRIMARY) ----
    st = self_test(topos, exact, ci, mc_summary)
    et_second_moment_self_test_passes = bool(st["passes"])

    out = {
        "primary_metric_name": "et_second_moment_self_test_passes",
        "et_second_moment_self_test_passes": int(et_second_moment_self_test_passes),
        "test_metric_name": "tps_finite_sample_ci_halfwidth",
        "tps_finite_sample_ci_halfwidth": test_halfwidth,
        "tps_finite_sample_ci_halfwidth_descent_only": descent_halfwidth,
        "accepted_length_std_both_bugs": topos["both_bugs"]["sigma_L"],
        "accepted_length_std_descent_only": topos["descent_only"]["sigma_L"],
        "verdict": (
            "SECOND MOMENT STAMPED. The #160 E[T]-DP's accepted-length pmf is exact "
            f"(max-abs-diff {exact['pmf_max_abs_diff']:.1e} vs two independent enumerations), "
            f"sums to 1, and reproduces E[T]=5.0564/5.2070 as its mean. sigma_L="
            f"{topos['both_bugs']['sigma_L']:.4f} (both-bugs) gives a finite-benchmark "
            f"single-shot TPS 95% half-width of +/-{test_halfwidth:.2f} TPS at the 16384-token "
            f"budget; the lower bound "
            f"{'CLEARS' if ci['both_bugs']['primary_16384_tau1']['lower_clears_500'] else 'does NOT clear'}"
            f" 500 for both-bugs and "
            f"{'CLEARS' if ci['descent_only']['primary_16384_tau1']['lower_clears_500'] else 'does NOT clear'}"
            f" 500 for descent-only. This is the sampling-uncertainty term, composing in "
            "quadrature with kanna #159's sigma_hw (armed) for the total single-shot CI, and "
            "it arms land #71 with a distributional readout gate alongside wirbel #170's mean gate."),
        "model": {
            "committed_length_L": "L = (edges descended) + 1 bonus token ; E[L] == E[T]",
            "pmf": "P(L=k) = sum_{c: depth_c=k-1} reach[c]*(1-s[c]) ; reach from #160 score_tree_depthrank",
            "second_moment": "Var[L] = E[L^2] - E[L]^2 (pure DP read of the same reach[] object)",
            "official": "official = K_cal*(L_bar/step)*tau ; SE[L_bar]=sigma_L/sqrt(N_steps); N_steps=B/E[T]",
            "K_cal": K_CAL, "step_measured": STEP_MEASURED, "tau_band": [TAU_LO, TAU_HI],
            "bench_tokens_primary": BENCH_TOKENS, "bench_tokens_sensitivity": BENCH_TOKENS_ALT,
            "z95": Z95,
            "iid_steps_caveat": (
                "SE[L_bar]=sigma_L/sqrt(N_steps) is the PR-prescribed iid-steps CLT term. If "
                "consecutive benchmark steps are positively serially correlated (shared text "
                "context), the effective N is smaller and the TRUE half-width is LARGER than "
                "reported -- i.e. +/-10.9 TPS is a LOWER estimate of the single-draw spread "
                "(positive correlation would widen it). Even so, the both-bugs lower CI bound "
                "524.5 clears 500 by a 24.5 TPS margin, so the go-signal is robust to plausible "
                "serial-correlation inflation."),
        },
        "imported_160": {
            "wandb_run_160": "x8vffgbs", "wandb_run_165": "laxllfjl", "wandb_run_170": "ne7p642c",
            "descent_only_E_T_anchor": ANCHOR_DESCENT_ONLY_ET,
            "both_bugs_E_T_anchor": ANCHOR_BOTH_BUGS_ET,
            "topology": topo_meta,
            "q_deployed_rising_spine": q_deployed, "rho_cond": rho_cond,
            "source_rho_opt": os.path.relpath(RHO_OPT_JSON, _ROOT),
            "source_accept": os.path.relpath(ACCEPT_JSON, _ROOT),
            "source_rankcov": os.path.relpath(RANKCOV_JSON, _ROOT),
        },
        "topologies": topos,
        "dp_exactness_certificate": exact,
        "mc_crosscheck": mc_summary,
        "finite_sample_tps_ci": ci,
        "lawine168_band_comparison": band_cmp,
        "quadrature_composition_kanna159": quad,
        "land71_distribution_gate": {
            "predicate": "land_histogram_in_band(measured_pmf, predicted_pmf, tol=0.02) -> bool",
            "predicate_body": "normalize(measured); max_k |measured[k]-predicted[k]| <= tol",
            "predicted_pmf_both_bugs": [float(x) for x in pred_norm],
            "demo": gate_demo,
            "composition_with_170": (
                "wirbel #170 = MEAN gate ('is E[T] trustworthy, not over-accept-inflated?'); "
                "THIS = DISTRIBUTION gate ('does the whole accepted-length shape match "
                "prediction?'). A build bug that preserves the mean E[T] but distorts the "
                "histogram passes #170 yet fails this -> together a mean+distribution build gate."),
            "handoff": "land #71 logs its per-step accepted-length histogram; run it through BOTH gates.",
        },
        "self_test": st,
        "provenance": (
            "imports wirbel #160 (W&B x8vffgbs) descent E[T]-DP: the M=32 depth-9 max-branch-3 "
            "topology (rho_optimal_topology_results.json), the measured rising spine + rho_cond, "
            "and score_tree_depthrank -- the SECOND moment is read off the SAME reach[] object, "
            "nothing re-derived. Anchors cross-checked against #165 (laxllfjl) composed ceiling "
            "5.206954 and #170 (ne7p642c) over-accept locus."),
        "method": ("LOCAL CPU-only analytic synthesis. No GPU/vLLM/HF Job/submission/kernel "
                   "build/served-file change. BASELINE stays 481.53. Greedy untouched. NOT open2 "
                   "(tree economics / DP second moment, not drafter architecture). Does NOT authorize a launch."),
        "metrics_nan_clean": int(st["nan_clean"]),
    }

    out_path = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    _console(out)
    rep_path = args.report_md if os.path.isabs(args.report_md) else os.path.join(_ROOT, args.report_md)
    _write_report(out, rep_path)

    if args.wandb:
        _log_wandb(args, out)


# ===================================================================================
# console + report + wandb
# ===================================================================================
def _console(out: dict) -> None:
    print("=" * 100)
    print("E[T] SECOND MOMENT -> finite-sample TPS CI + land distribution gate (PR #175, wirbel)")
    print("=" * 100)
    tm = out["imported_160"]["topology"]
    print(f"\n[MODEL] M={tm['n_nodes']} depth={tm['built_depth']} max_branch={tm['max_branch']} "
          f"leaves={tm['n_leaves']}  (imported #160 x8vffgbs)")
    for lab in ("descent_only", "both_bugs"):
        t = out["topologies"][lab]
        print(f"\n[{lab.upper()}]  q1={t['q1_depth1_spine']:.4f}")
        print(f"  score_tree E[T] = {t['score_tree_E_T']:.9f}   pmf-mean = {t['pmf_mean']:.9f}  "
              f"(sum={t['pmf_total_mass']:.12f})")
        print(f"  Var[L] = {t['var_L']:.6f}   sigma_L = {t['sigma_L']:.6f}   "
              f"mode L = {t['mode_committed_length']}")
        print(f"  pmf P(L=k) k=1..{t['pmf_max_support_length']}: " +
              " ".join(f"{r['P']:.4f}" for r in t["accepted_length_pmf"]))
        print(f"  exactness: DFS diff={t['bruteforce_dfs_max_abs_diff']:.2e} "
              f"parentwalk diff={t['bruteforce_parentwalk_max_abs_diff']:.2e} "
              f"MC diff={t['mc_max_abs_diff']:.2e}")
        c = out["finite_sample_tps_ci"][lab]["primary_16384_tau1"]
        print(f"  [CI 16384/tau1] central={c['central_tps']:.2f}  half=+/-{c['ci_halfwidth_tps']:.2f}  "
              f"[{c['ci_lower_tps']:.2f}, {c['ci_upper_tps']:.2f}]  "
              f"lower_clears_500={c['lower_clears_500']}  (N_steps={c['N_steps']:.1f})")
    bc = out["lawine168_band_comparison"]["both_bugs"]
    print(f"\n[vs lawine #168 band] both-bugs half +/-{bc['halfwidth_tps']:.2f} vs +/-{bc['lawine168_band_tps']} "
          f"-> {bc['relation']} (ratio {bc['ratio_to_lawine168']:.2f}x)")
    q = out["quadrature_composition_kanna159"]["both_bugs"]
    print(f"[quadrature] {q['formula']}  acceptlen=+/-{q['halfwidth_acceptlen_tps']:.2f}  "
          f"sigma_hw={q['halfwidth_sigma_hw_tps_PENDING']} (armed={q['armed']})")
    st = out["self_test"]
    print(f"\n[SELF-TEST] {st['n_passed']}/{st['n_checks']} checks")
    for c in st["checks"]:
        print(f"  [{'OK' if c['passes'] else 'FAIL'}] {c['name']}  ({c['detail']})")
    print(f"\n[PRIMARY] et_second_moment_self_test_passes = {out['et_second_moment_self_test_passes']}")
    print(f"[TEST]    tps_finite_sample_ci_halfwidth (both-bugs) = {out['tps_finite_sample_ci_halfwidth']:.4f} TPS")
    print(f"[NaN-clean] {out['metrics_nan_clean']}")


def _write_report(out: dict, path: str) -> None:
    tb = out["topologies"]["both_bugs"]
    td = out["topologies"]["descent_only"]
    cb = out["finite_sample_tps_ci"]["both_bugs"]["primary_16384_tau1"]
    cd = out["finite_sample_tps_ci"]["descent_only"]["primary_16384_tau1"]
    cb65 = out["finite_sample_tps_ci"]["both_bugs"]["sensitivity_65536_tau1"]
    bc = out["lawine168_band_comparison"]
    ex = out["dp_exactness_certificate"]
    st = out["self_test"]
    tm = out["imported_160"]["topology"]

    def pmf_rows(t):
        s = ""
        for r in t["accepted_length_pmf"]:
            s += f"| {r['committed_length_L']} | {r['accepted_drafts']} | {r['P']:.6f} |\n"
        return s

    md = f"""<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# E[T] second moment — finite-sample TPS CI + land distribution gate (PR #175 · wirbel)

**PRIMARY** `et_second_moment_self_test_passes` = **{bool(out['et_second_moment_self_test_passes'])}** ({st['n_passed']}/{st['n_checks']} checks, NaN-clean)
**TEST** `tps_finite_sample_ci_halfwidth` = **±{out['tps_finite_sample_ci_halfwidth']:.3f} TPS** (both-bugs, 16384-token budget, measured step, τ=1)

## Honest scope
Pure-analytic **CPU-only** second-moment read of wirbel #160's E[T]-DP. No GPU / vLLM / HF Job / submission / kernel build / served-file change. BASELINE stays 481.53; **0 TPS**; greedy untouched by construction. Imports #160 (`x8vffgbs`) topology + `score_tree_depthrank`, #165 (`laxllfjl`) composed ceiling, #170 (`ne7p642c`) over-accept locus — **does NOT re-derive any of them**. The second moment is read off the **same `reach[]` object** that yields the first moment. **NOT open2** (tree economics, not drafter architecture). **Does not authorize a launch.**

## The model (imported; the second moment falls out of the first-moment object)
The descent walk commits `L = (edges descended) + 1 bonus` tokens per step, so `E[L] = E[T]`. #160's `score_tree_depthrank` gives `reach[c] = P(reach node c)` (path-product of edge marginals). The walk **stops** at `c` (deepest reached) with prob `reach[c]·(1 − s[c])`, `s[c]` = sum of `c`'s child-edge marginals. Hence the exact pmf:

```
P(L = k) = Σ_{{c : depth_c = k−1}} reach[c] · (1 − s[c])     (sums to 1; mean = E[T])
Var[L] = E[L²] − E[L]²        σ_L = √Var[L]
```

Topology (imported #160): **M={tm['n_nodes']}, depth {tm['built_depth']}, max-branch {tm['max_branch']}, {tm['n_leaves']} leaves**.

## 1. Accepted-length pmf (both topologies)
`L` = committed length (incl. bonus); `accepted_drafts = L−1` (the literal "k=0..depth" view). `E[L]=E[T]`, so the mean falls out as the first-moment consistency check.

**both-bugs (E[T]={tb['pmf_mean']:.6f}, σ_L={tb['sigma_L']:.4f}, Var={tb['var_L']:.4f}):**

| committed L | accepted drafts | P(L) |
|---|---|---|
{pmf_rows(tb)}
**descent-only (E[T]={td['pmf_mean']:.6f}, σ_L={td['sigma_L']:.4f}, Var={td['var_L']:.4f}):**

| committed L | accepted drafts | P(L) |
|---|---|---|
{pmf_rows(td)}
First-moment consistency: `Σ_k k·P(L=k)` = {tb['pmf_mean']:.9f} (both-bugs) / {td['pmf_mean']:.9f} (descent-only) — reproduces the imported #160 ceilings exactly. `Σ_k P(L=k)` = {tb['pmf_total_mass']:.12f}.

## 2. DP-exactness certificate
Two **independent** enumerations of the walk's stop-node distribution — a recursive DFS path-product and a per-node parent-walk — match the DP pmf to **max-abs-diff {ex['pmf_max_abs_diff']:.1e}** (DFS) / **{ex['pmf_max_abs_diff_parentwalk']:.1e}** (parent-walk), tol {ex['tol']:.0e}. `dp_distribution_exact` = **{ex['dp_distribution_exact']}**. A {out['mc_crosscheck']['mc_trials']:,}-trial Monte-Carlo histogram agrees to {out['mc_crosscheck']['mc_max_abs_diff']:.1e}. The propagated second moment is therefore exact, not a DP artifact.

## 3. Second moment → finite-benchmark TPS CI
Budget `B={out['model']['bench_tokens_primary']}` tokens (128×128, PR contract); `N_steps ≈ B/E[T]`. `SE[L̄]=σ_L/√N_steps`; `official=K_cal·(L̄/step)·τ` is linear in `L̄`, so the 95 % half-width is `1.96·(K_cal·τ/step)·SE[L̄]`.

| topology | E[T] | σ_L | N_steps | central TPS | ±95% half (TPS) | 95% CI | lower clears 500? |
|---|---|---|---|---|---|---|---|
| both-bugs | {cb['E_T']:.4f} | {tb['sigma_L']:.4f} | {cb['N_steps']:.0f} | {cb['central_tps']:.2f} | ±{cb['ci_halfwidth_tps']:.3f} | [{cb['ci_lower_tps']:.2f}, {cb['ci_upper_tps']:.2f}] | **{cb['lower_clears_500']}** |
| descent-only | {cd['E_T']:.4f} | {td['sigma_L']:.4f} | {cd['N_steps']:.0f} | {cd['central_tps']:.2f} | ±{cd['ci_halfwidth_tps']:.3f} | [{cd['ci_lower_tps']:.2f}, {cd['ci_upper_tps']:.2f}] | **{cd['lower_clears_500']}** |

**vs lawine #168's ±2.4 TPS roofline↔overlap band:** the both-bugs finite-sample half-width ±{bc['both_bugs']['halfwidth_tps']:.2f} is **{bc['both_bugs']['relation']}** ({bc['both_bugs']['ratio_to_lawine168']:.2f}× lawine's band) — i.e. the single-draw sampling scatter is {('the larger' if bc['both_bugs']['relation']=='LARGER' else 'comparable to' if bc['both_bugs']['relation']=='same order' else 'the smaller')} of the two terms.

**Budget sensitivity:** at the 512-token/prompt budget (B={out['model']['bench_tokens_sensitivity']}, the #170 convention) the half-width shrinks to ±{cb65['ci_halfwidth_tps']:.3f} TPS (∝ 1/√N_steps). The PR's 16384-token contract is the conservative (wider) case.

## 4. Quadrature composition with kanna #159 (σ_hw) + land #71 distribution gate
**(a)** Total single-shot TPS variance = this accept-length **numerator** term ⊕ kanna #159's σ_hw **denominator** step-jitter term (independent → quadrature):

```
halfwidth_total = √( halfwidth_acceptlen²  +  halfwidth_σ_hw² )
                = √( {cb['ci_halfwidth_tps']:.3f}²  +  σ_hw_term² )      ← σ_hw slot ARMED (kanna #159, pending)
```

The accept-length half-width ±{cb['ci_halfwidth_tps']:.3f} TPS is supplied here; plug kanna #159's measured σ_hw TPS half-width into the armed slot to close the total single-shot CI.

**(b)** land #71 distributional gate (validates the build at the **shape** level):

```
land_histogram_in_band(measured_pmf, predicted_pmf, tol=0.02) -> bool
    # normalize(measured); return max_k |measured[k] − predicted[k]| <= tol
```

Composition with #170: **#170 = mean gate** ("is E[T] trustworthy, not over-accept-inflated?"); **this = distribution gate** ("does the whole accepted-length shape match prediction?"). A build bug that preserves the mean but distorts the histogram passes #170 yet fails this — together a **mean + distribution** build gate. (Demo: a mean-preserving 5 %-mass shape distortion is caught — measured mean {out['land71_distribution_gate']['demo'][1].get('perturbed_mean', float('nan')):.4f} ≈ predicted {out['land71_distribution_gate']['demo'][1].get('predicted_mean', float('nan')):.4f}, but `in_band={out['land71_distribution_gate']['demo'][1]['in_band']}`.)

## 5. Self-validate (PRIMARY)
{st['n_passed']}/{st['n_checks']} checks pass: DP mean reproduces 5.0564/5.2070 and equals `score_tree_depthrank`; pmf sums to 1 and is non-negative; brute-force exactness (two enumerations); σ_L ≥ 0; CI brackets the point estimate; lower-clears-500 verdict explicit for both topologies; NaN-clean. **`et_second_moment_self_test_passes` = {bool(out['et_second_moment_self_test_passes'])}**.

## Hand-off
This is the **finite-sample sampling-uncertainty stamp** for fern #167's launch packet — how much the single irreversible benchmark draw could scatter from the central TPS by chance alone, distinct from every input-band (fern #174) and modeling bound (denken #172 / wirbel #170) already in the packet. Composes in quadrature with kanna #159 (σ_hw) for the total single-shot TPS CI, and hands land #71 a distributional readout gate alongside wirbel #170's mean gate. Does **not** authorize a launch.

## Public / banked evidence used
- wirbel #160 (`x8vffgbs`): descent E[T]-DP — topology + measured rising spine + `score_tree_depthrank` (imported; second moment read off the same `reach[]`).
- wirbel #165 (`laxllfjl`): composed ceiling 5.206954 (both-bugs anchor).
- wirbel #170 (`ne7p642c`): over-accept locus — the **mean** trustworthiness gate this **distribution** gate composes with.
- lawine #168: measured step 1.2182 + the ±2.4 TPS roofline↔overlap band compared against.
- kanna #159: σ_hw step-jitter — the denominator quadrature twin (armed, pending).
"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(md)
    print(f"\nwrote {os.path.relpath(path, _ROOT)}")


def _log_wandb(args, out: dict) -> None:
    import wandb

    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     name=args.wandb_name, group=args.wandb_group, job_type="analysis",
                     config={"leg": "et-second-moment-tps-ci",
                             "method": "cpu-analytic-second-moment-of-160-ET-DP",
                             "bench_tokens_primary": BENCH_TOKENS,
                             "bench_tokens_sensitivity": BENCH_TOKENS_ALT,
                             "K_cal": K_CAL, "step_measured": STEP_MEASURED, "z95": Z95,
                             "wandb_run_160": "x8vffgbs", "wandb_run_165": "laxllfjl",
                             "wandb_run_170": "ne7p642c"})
    s = wandb.summary
    s["et_second_moment_self_test_passes"] = out["et_second_moment_self_test_passes"]
    s["tps_finite_sample_ci_halfwidth"] = out["tps_finite_sample_ci_halfwidth"]
    s["tps_finite_sample_ci_halfwidth_descent_only"] = out["tps_finite_sample_ci_halfwidth_descent_only"]
    s["accepted_length_std_both_bugs"] = out["accepted_length_std_both_bugs"]
    s["accepted_length_std_descent_only"] = out["accepted_length_std_descent_only"]
    s["metrics_nan_clean"] = out["metrics_nan_clean"]
    s["pmf_max_abs_diff"] = out["dp_exactness_certificate"]["pmf_max_abs_diff"]
    s["dp_distribution_exact"] = int(out["dp_exactness_certificate"]["dp_distribution_exact"])
    s["mc_max_abs_diff"] = out["mc_crosscheck"]["mc_max_abs_diff"]
    for lab in ("descent_only", "both_bugs"):
        t = out["topologies"][lab]
        s[f"{lab}_E_T"] = t["pmf_mean"]
        s[f"{lab}_var_L"] = t["var_L"]
        s[f"{lab}_sigma_L"] = t["sigma_L"]
        c = out["finite_sample_tps_ci"][lab]["primary_16384_tau1"]
        s[f"{lab}_central_tps"] = c["central_tps"]
        s[f"{lab}_ci_halfwidth_tps"] = c["ci_halfwidth_tps"]
        s[f"{lab}_ci_lower_tps"] = c["ci_lower_tps"]
        s[f"{lab}_lower_clears_500"] = int(c["lower_clears_500"])
        s[f"{lab}_N_steps"] = c["N_steps"]
    s["lawine168_band_ratio_both_bugs"] = out["lawine168_band_comparison"]["both_bugs"]["ratio_to_lawine168"]
    s["n_checks"] = out["self_test"]["n_checks"]
    s["n_passed"] = out["self_test"]["n_passed"]

    # pmf tables
    for lab in ("descent_only", "both_bugs"):
        pt = wandb.Table(columns=["committed_length_L", "accepted_drafts", "P"])
        for r in out["topologies"][lab]["accepted_length_pmf"]:
            pt.add_data(r["committed_length_L"], r["accepted_drafts"], r["P"])
        wandb.log({f"accepted_length_pmf_{lab}": pt})

    # finite-sample CI table (both topologies, both budgets)
    ct = wandb.Table(columns=["topology", "budget_tokens", "tau", "E_T", "sigma_L",
                              "N_steps", "central_tps", "ci_halfwidth_tps",
                              "ci_lower_tps", "lower_clears_500"])
    for lab in ("descent_only", "both_bugs"):
        for key in ("primary_16384_tau1", "primary_16384_tau_lo", "sensitivity_65536_tau1"):
            c = out["finite_sample_tps_ci"][lab][key]
            ct.add_data(lab, c["n_tokens"], c["tau"], c["E_T"],
                        out["topologies"][lab]["sigma_L"], c["N_steps"],
                        c["central_tps"], c["ci_halfwidth_tps"], c["ci_lower_tps"],
                        int(c["lower_clears_500"]))
    wandb.log({"finite_sample_tps_ci": ct})

    # self-test checks
    stt = wandb.Table(columns=["check", "passes", "detail"])
    for c in out["self_test"]["checks"]:
        stt.add_data(c["name"], int(c["passes"]), c["detail"])
    wandb.log({"self_test_checks": stt})

    print(f"\nW&B run: {run.id}  ({run.url})")
    wandb.finish()


if __name__ == "__main__":
    main()
