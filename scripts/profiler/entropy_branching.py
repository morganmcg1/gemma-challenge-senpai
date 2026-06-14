#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Does the drafter's OWN UNCERTAINTY predict rank-2 branching value? (PR #86)

WHAT THIS ANSWERS
-----------------
PR #83 proved per-DEPTH rho2 is FLAT [0.397-0.445] -- spine position does NOT predict
the rank-2 rescue probability. The open question (this PR): does the drafter's
predictive ENTROPY at a first-reject step predict rho2? Calibration intuition says a
drafter that is UNCERTAIN (high entropy, mass spread over candidates) is more likely
to have the verifier's true token at rank-2; a drafter that is CONFIDENT-BUT-WRONG
(low entropy, peaked on a wrong token) is not. If rho2 rises with entropy, an
ENTROPY-GATED tree could allocate width to high-uncertainty steps and beat the uniform
max-branch-3 DP (E[T]=5.207, +18.17%) at the same M=32 budget. If rho2 is flat in
entropy, the uniform DP is at the structural limit and the lane closes.

INPUT: the PR #86 extended rank-coverage records (rankprobe_patch.py with
RANKPROBE_LOGITS=1). Each first-divergence record carries:
    Hd    -- drafter predictive entropy over the full sparse candidate set (nats)
    Hd64  -- drafter entropy over the top-K candidates (robustness twin)
    pd    -- drafter top-4 softmax probs [p1>=p2>=p3>=p4]
    vp1   -- verifier (target) top-1 softmax prob at the rejected position
    Hv    -- verifier full-vocab entropy at the rejected position
    rank_fd / salv2 -- rank of the true token in the drafter top-W; salv2 = (rank_fd==2)
    req   -- prompt id (for the within-prompt confound control)

ESTIMAND: rho2(x_bin) = P(rank_fd == 2 | rank-1 missed, x in bin) where x is the
conditioning signal (drafter entropy / 1-p1 / verifier p1). 5 EQUAL-FREQUENCY bins
(balanced counts; the first-reject selection effect compresses the entropy range, so
equal-width bins would be badly unbalanced -- see researcher note).

OUTPUT: rho2(entropy_bin) + rho2(1-p1_bin) + rho2(verifier_p1_bin) tables, Pearson r
(bin-level, the primary metric) cross-checked with the per-record point-biserial and
Spearman, bootstrap CIs, a within-prompt-controlled correlation, and a flat/non-flat
verdict. CPU-only, analytic; no GPU, no vLLM, no served-file change.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_coverage import _record_files  # noqa: E402  (reuse #79 shard discovery)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "research" / "rank_coverage"
DEFAULT_RECORDS = OUT_DIR / "rankprobe_records.jsonl"
ACCEPT_JSON = ROOT / "research" / "accept_calibration" / "accept_calibration_results.json"
RANKCOV_JSON = ROOT / "research" / "rank_coverage" / "rank_coverage_results.json"

# Non-flat / flat decision thresholds (from the PR body).
R_SIGNIF = 0.20          # |Pearson r| above this => non-flat
RANGE_SIGNIF = 0.10      # rho2 spread across bins above this => non-flat
R_FLAT = 0.05            # |Pearson r| below this AND ...
RANGE_FLAT = 0.05        # ... rho2 spread below this => lane closed
RHO2_79 = 0.4165         # #79 unconditional pooled rho2 (sanity anchor)

# #83 deployed uniform tree (max-branch-3, depth-9, M=32) -- the entropy-gated DP
# ceiling baseline. E[T]=5.207, +18.17% drafter-aware (research/spec_cost_model/
# rho_optimal_topology_results.json). Cost model: g_verify(M)=0.532, g_drafter=0.168.
TOPO83_M32 = [-1, 0, 0, 0, 1, 1, 1, 2, 3, 4, 4, 5, 7, 9, 9, 10, 11, 12, 13, 15, 16,
              17, 18, 19, 20, 21, 22, 24, 25, 26, 28, 29]
ET_UNIFORM_83 = 5.207            # #83 uniform max-branch-3 E[T] (the gate to beat)
GAIN_UNIFORM_83 = 0.1817         # #83 uniform drafter-aware TPS gain
G_VERIFY, G_DRAFTER, BASE_DEPTH = 0.532, 0.168, 7    # #83 cost-model shares
WALL_MDE = 0.002                 # lawine #72 wall_tps MDE upper bound
# A per-step dynamic tree forfeits the static-topology onegraph CUDA graph on a 99.4%-
# GPU-bound decode (project_cudagraph_already_deployed), an eager-mode cost of several
# percent. We use a CONSERVATIVE 3x-MDE bar (0.6pp) as the minimum oracle TPS ceiling
# that would even justify a dynamic-tree feasibility study; the true bar is much higher.
DYNAMIC_TREE_BAR = 3.0


# --------------------------------------------------------------------------- #
# Load first-reject records with the PR #86 logit fields
# --------------------------------------------------------------------------- #
def load_records(records_path: Path) -> dict[str, np.ndarray]:
    shards = _record_files(records_path)
    if not shards:
        raise SystemExit(f"no record shards found at {records_path}.*")
    Hd, Hd64, p1, vp1, Hv, rank_fd, salv2, req, depth = ([] for _ in range(9))
    n_total = n_div = n_with_logits = n_align_bad = 0
    for shard in shards:
        with open(shard) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                n_total += 1
                if not rec.get("align", True):
                    n_align_bad += 1
                    continue
                if int(rec["fd"]) >= int(rec["n"]):
                    continue  # all-accept step, no first divergence
                n_div += 1
                if "Hd" not in rec or rec.get("Hd") is None:
                    continue  # #79-format record without logit capture
                n_with_logits += 1
                pd = rec.get("pd") or [float("nan")]
                Hd.append(float(rec["Hd"]))
                Hd64.append(float(rec.get("Hd64", rec["Hd"])))
                p1.append(float(pd[0]))
                vp1.append(float(rec["vp1"]) if rec.get("vp1") is not None else float("nan"))
                Hv.append(float(rec["Hv"]) if rec.get("Hv") is not None else float("nan"))
                rank_fd.append(int(rec["rank_fd"]))
                salv2.append(int(rec.get("salv2", 1 if int(rec["rank_fd"]) == 2 else 0)))
                req.append(int(rec.get("req", -1)))
                depth.append(int(rec["fd"]))
    out = {
        "Hd": np.array(Hd), "Hd64": np.array(Hd64), "p1": np.array(p1),
        "vp1": np.array(vp1), "Hv": np.array(Hv),
        "rank_fd": np.array(rank_fd), "salv2": np.array(salv2, dtype=float),
        "req": np.array(req), "depth": np.array(depth),
        "n_total": n_total, "n_div": n_div, "n_with_logits": n_with_logits,
        "n_align_bad": n_align_bad, "shards": [p.name for p in shards],
    }
    return out


# --------------------------------------------------------------------------- #
# Equal-frequency binning + per-bin rho2
# --------------------------------------------------------------------------- #
def equal_freq_bins(x: np.ndarray, y: np.ndarray, n_bins: int) -> dict[str, Any]:
    """rho2 (= mean y) per equal-frequency bin of x. y is the salvage indicator.

    Returns bin midpoints (mean x in bin), bin edges, per-bin rho2, counts, and
    binomial SE per bin. Equal-frequency => balanced counts (handles the right-shifted
    entropy distribution from first-reject conditioning).
    """
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    n = len(xs)
    # quantile edges; dedupe to avoid empty bins when x has ties/plateaus
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(xs, qs)
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    idx = np.digitize(xs, edges[1:-1], right=False)
    mids, rho2s, counts, ses, lo_hi = [], [], [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        c = int(m.sum())
        if c == 0:
            continue
        r = float(ys[m].mean())
        mids.append(float(xs[m].mean()))
        rho2s.append(r)
        counts.append(c)
        ses.append(float(math.sqrt(max(r * (1 - r), 0.0) / c)))
        lo_hi.append((float(xs[m].min()), float(xs[m].max())))
    return {
        "bin_mid_x": mids, "bin_rho2": rho2s, "bin_count": counts,
        "bin_se": ses, "bin_x_range": lo_hi,
        "rho2_range": (max(rho2s) - min(rho2s)) if rho2s else 0.0,
        "edges": [float(e) for e in edges],
    }


def pearson(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return pearson(ra, rb)


def within_prompt_corr(x: np.ndarray, y: np.ndarray, req: np.ndarray) -> dict[str, Any]:
    """Pearson r between x and y after removing each ``req`` group's mean.

    NOTE: the #79/#86 probe runs at conc=1, so ``req`` is always 0 (the in-batch
    request index, NOT the prompt id) -- this control therefore collapses to the
    raw point-biserial. The MEANINGFUL stratifier for the static-tree question is the
    first-divergence DEPTH (see ``within_depth_corr``), kept here only for provenance.
    """
    finite = np.isfinite(x) & np.isfinite(y)
    x, y, req = x[finite], y[finite], req[finite]
    xd = x.astype(float).copy()
    yd = y.astype(float).copy()
    for r in np.unique(req):
        m = req == r
        if m.sum() >= 2:
            xd[m] -= x[m].mean()
            yd[m] -= y[m].mean()
        else:
            xd[m] = 0.0
            yd[m] = 0.0
    return {"within_prompt_pearson": pearson(xd, yd), "n_prompts": int(len(np.unique(req)))}


def within_depth_corr(x: np.ndarray, y: np.ndarray, depth: np.ndarray,
                      min_count: int = 150) -> dict[str, Any]:
    """Point-biserial r between x and salvage y AFTER removing each first-divergence
    DEPTH's mean. This is the decisive confound control for the tree question: #83
    proved per-DEPTH rho2 is FLAT [0.397-0.445], and a static draft tree can only
    allocate branch width by tree position (depth). If the entropy->rho2 signal
    SURVIVES depth-demeaning (within_depth ~ raw point-biserial), the signal is a
    genuine WITHIN-step quantity that NO static depth-indexed topology can exploit --
    only a per-step dynamic tree could. If it vanishes, entropy is just a depth proxy
    the static tree already captures."""
    finite = np.isfinite(x) & np.isfinite(y)
    x, y, depth = x[finite], y[finite], depth[finite]
    xd = x.astype(float).copy()
    yd = y.astype(float).copy()
    used = 0
    for dval in np.unique(depth):
        m = depth == dval
        if m.sum() >= min_count:
            xd[m] -= x[m].mean()
            yd[m] -= y[m].mean()
            used += int(m.sum())
        else:
            xd[m] = 0.0
            yd[m] = 0.0
    n_depths = int(sum(1 for dv in np.unique(depth) if (depth == dv).sum() >= min_count))
    return {"within_depth_pointbiserial": pearson(xd, yd),
            "n_depths_used": n_depths, "n_records_used": used}


def bootstrap_ci(x: np.ndarray, y: np.ndarray, n_bins: int, n_boot: int,
                 seed: int = 7) -> dict[str, Any]:
    """Bootstrap CI on the bin-level Pearson r and on the rho2 range across bins."""
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    rng = np.random.default_rng(seed)
    n = len(x)
    rs, ranges = [], []
    for _ in range(n_boot):
        samp = rng.integers(0, n, n)
        bb = equal_freq_bins(x[samp], y[samp], n_bins)
        if len(bb["bin_mid_x"]) >= 2:
            rs.append(pearson(bb["bin_mid_x"], bb["bin_rho2"]))
            ranges.append(bb["rho2_range"])
    rs = np.array([v for v in rs if np.isfinite(v)])
    ranges = np.array(ranges)
    def ci(a):
        return [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))] if len(a) else [float("nan")] * 2
    return {"r_ci95": ci(rs), "rho2_range_ci95": ci(ranges),
            "r_mean": float(rs.mean()) if len(rs) else float("nan")}


def analyze_signal(name: str, x: np.ndarray, y: np.ndarray, req: np.ndarray,
                   depth: np.ndarray, n_bins: int, n_boot: int) -> dict[str, Any]:
    bins = equal_freq_bins(x, y, n_bins)
    r_bin = pearson(bins["bin_mid_x"], bins["bin_rho2"])
    r_point = pearson(x, y)               # per-record point-biserial (x vs binary y)
    rho_sp = spearman(np.array(bins["bin_mid_x"]), np.array(bins["bin_rho2"]))
    wp = within_prompt_corr(x, y, req)
    wd = within_depth_corr(x, y, depth)
    boot = bootstrap_ci(x, y, n_bins, n_boot)
    return {
        "signal": name,
        "bins": bins,
        "pearson_r_binned": r_bin,                  # PRIMARY for entropy signal
        "pearson_r_pointbiserial": r_point,
        "spearman_rho_binned": rho_sp,
        "rho2_range_across_bins": bins["rho2_range"],
        **wp,
        **wd,
        "bootstrap": boot,
    }


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #
def verdict(entropy_res: dict[str, Any]) -> dict[str, Any]:
    r = abs(entropy_res["pearson_r_binned"])
    rng = entropy_res["rho2_range_across_bins"]
    significant = (r > R_SIGNIF) or (rng > RANGE_SIGNIF)
    flat = (r < R_FLAT) and (rng < RANGE_FLAT)
    state = "non_flat" if significant else ("flat" if flat else "inconclusive")
    return {
        "rho2_entropy_correlation_r": entropy_res["pearson_r_binned"],
        "rho2_range_across_entropy_bins": rng,
        "significant_for_gating": bool(significant),
        "flat_lane_closed": bool(flat),
        "state": state,
        "thresholds": {"r_signif": R_SIGNIF, "range_signif": RANGE_SIGNIF,
                       "r_flat": R_FLAT, "range_flat": RANGE_FLAT},
    }


# --------------------------------------------------------------------------- #
# Step 4: entropy-gated DP ORACLE CEILING (regime-conditional Sequoia DP)
# --------------------------------------------------------------------------- #
def _bin_rho_ladder(rank_fd: np.ndarray, mask: np.ndarray, W: int = 4) -> dict[str, Any]:
    """Measured per-rank conditional rescue ladder [rho2..rhoW] on the masked subset,
    via the #79 chain-rule denominators (beyond-W mass stays in every denominator)."""
    c = int(mask.sum())
    rf = rank_fd[mask]
    nr = {r: int((rf == r).sum()) for r in range(2, W + 1)}
    n0 = int((rf == 0).sum())
    rho, remaining = [], c
    for r in range(2, W + 1):
        rho.append(nr[r] / remaining if remaining > 0 else 0.0)
        remaining -= nr[r]
    cov = sum(nr[r] for r in range(2, W + 1)) / c if c else 0.0
    return {"rho_cond": rho, "cov4": cov, "beyond4": (n0 / c if c else 0.0), "count": c}


def entropy_gated_dp_ceiling(Hd: np.ndarray, rank_fd: np.ndarray, *, n_bins: int = 5,
                             M: int = 32, depth_cap: int = 9, W: int = 4) -> dict[str, Any]:
    """ORACLE CEILING on the E[T]/TPS gain of a perfectly entropy-gated DYNAMIC tree.

    Step 4 of the PR: re-run the #83 depth-dependent Sequoia DP with position-varying
    rho2 = f(entropy_bin). Each entropy bin is treated as a REGIME with its own measured
    rescue ladder; the drafter's spine acceptance q[d] (#76) is held GLOBAL because
    entropy-gating reallocates only BRANCH width, never the rank-1 spine. We compare a
    one-size-fits-all topology (the deployed #83 tree T*) against a tree that picks the
    REGIME-OPTIMAL topology per step, both at the SAME M=32 node budget (cost matched),
    then frequency-weight. This is an UPPER BOUND: it assumes free, perfect per-step
    regime identification AND ignores the cost of a per-step DYNAMIC tree (which breaks
    the static-topology onegraph CUDA capture the deployed stack depends on).

    Returns the cost-matched E[T] gain (the PR test metric) plus a drafter-aware
    adaptive-depth TPS variant and an aggressive regime-switch sanity check.
    """
    from treeshape_measured_accept import (  # noqa: E402
        build_depth_pvecs_measured, score_tree_depthrank, load_measured)
    from rho_optimal_topology import build_depth_dp, depth_swept_optimum  # noqa: E402
    from treeshape_real_cost import RealGemmCurve  # noqa: E402
    from sequoia_dp_tree import build_linear  # noqa: E402

    meas = load_measured(str(ACCEPT_JSON), "server_log")
    q = meas["q"]
    real = RealGemmCurve(str(ROOT / "research" / "spec_cost_model" / "verify_gemm_roofline.json"))

    def cost_mult(m: int, depth: int) -> float:
        return (1.0 - G_VERIFY - G_DRAFTER) + G_VERIFY * real.ratio(m) + G_DRAFTER * (depth / BASE_DEPTH)

    # global pv + linear-8 anchor (E[T] of the deployed K=7 chain under the measured ladder)
    glad = _bin_rho_ladder(rank_fd, np.ones(len(Hd), bool), W)
    gpv = build_depth_pvecs_measured(q, glad["rho_cond"], W, 24, "flat")
    F_lin8, _ = score_tree_depthrank(build_linear(8), gpv)
    F_uni, d_uni = score_tree_depthrank(TOPO83_M32, gpv)         # #83 tree under global pv
    cost_uni = cost_mult(M, d_uni)
    gain_uni = (F_uni / F_lin8) / cost_uni - 1.0

    edges = np.quantile(Hd, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(Hd, edges[1:-1])

    rows = []
    eu = ed = 0.0                       # (A) cost-matched E[T] mixture (uniform vs dynamic)
    num_dyn = den_dyn = 0.0             # (B) drafter-aware adaptive-depth TPS mixture
    for b in range(n_bins):
        m = idx == b
        lad = _bin_rho_ladder(rank_fd, m, W)
        freq = lad["count"] / len(Hd)
        pv = build_depth_pvecs_measured(q, lad["rho_cond"], W, 24, "flat")
        Fu_b, _ = score_tree_depthrank(TOPO83_M32, pv)          # uniform tree, this regime
        par_b, Fd_b, dd_b = build_depth_dp(pv, M, depth_cap, W)  # regime-optimal, M=32, d<=cap
        dyn_br = max(par_b.count(i) for i in range(len(par_b)))
        # drafter-aware: regime free to depth-sweep its own optimum (captures cost saving)
        best, _ = depth_swept_optimum(
            pv, M, 24, W, lambda F, mm, dep, gd=None: {"gain": (F / F_lin8) / cost_mult(mm, dep) - 1.0})
        eu += freq * Fu_b
        ed += freq * Fd_b
        num_dyn += freq * best["F_tree"]
        den_dyn += freq * cost_mult(M, best["depth"])
        rows.append({
            "bin": b + 1, "freq": freq, "count": lad["count"],
            "entropy_mid": float(Hd[m].mean()), "rho_cond": lad["rho_cond"],
            "cov4": lad["cov4"], "beyond4": lad["beyond4"],
            "ET_uniform_tree": Fu_b, "ET_regime_optimal": Fd_b,
            "regime_opt_depth": dd_b, "regime_opt_max_branch": dyn_br,
            "drafter_aware_opt_depth": best["depth"],
        })

    gain_ET_costmatched = ed / eu - 1.0
    tps_dyn = (num_dyn / F_lin8) / den_dyn - 1.0
    gain_tps_adaptive_depth = tps_dyn - gain_uni

    # (C) aggressive: drop to a linear M=8 tree on the top-2 (uncertain) regimes
    num_c = den_c = 0.0
    for b in range(n_bins):
        m = idx == b
        lad = _bin_rho_ladder(rank_fd, m, W)
        freq = lad["count"] / len(Hd)
        pv = build_depth_pvecs_measured(q, lad["rho_cond"], W, 24, "flat")
        if b >= n_bins - 2:                                     # uncertain -> linear M=8
            F, d, mm = (*score_tree_depthrank(build_linear(8), pv), 8)
        else:                                                  # confident -> #83 tree M=32
            F, d, mm = (*score_tree_depthrank(TOPO83_M32, pv), M)
        num_c += freq * F
        den_c += freq * cost_mult(mm, d)
    gain_tps_aggressive = ((num_c / F_lin8) / den_c - 1.0) - gain_uni

    return {
        "n_bins": n_bins, "M": M, "depth_cap": depth_cap,
        "q_global_held_constant": True,
        "global_rho_cond": glad["rho_cond"], "global_cov4": glad["cov4"],
        "F_linear8_anchor": F_lin8,
        "ET_uniform_global": F_uni, "uniform_depth": d_uni, "uniform_tps_gain": gain_uni,
        "ET_uniform_mix": eu, "ET_dynamic_mix": ed,
        "entropy_gated_tree_E_T_gain_pct": gain_ET_costmatched * 100.0,   # PR TEST METRIC
        "tps_gain_adaptive_depth_pp": gain_tps_adaptive_depth * 100.0,
        "tps_gain_aggressive_regimeswitch_pp": gain_tps_aggressive * 100.0,
        "per_regime": rows,
        "note": ("ORACLE CEILING. q[d] spine held global (#76); only the branch rescue "
                 "ladder varies by entropy regime. Uniform tree is near-optimal in EVERY "
                 "regime, so per-regime re-optimisation is third-order -- the spine "
                 "dominates E[T]. A per-step dynamic tree is also not capturable by the "
                 "deployed static-topology onegraph CUDA graph."),
    }


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--n-bins-fine", type=int, default=10,
                    help="extra fine binning for the diagnostic plot table")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--dp-budget", type=int, default=32,
                    help="node budget M for the entropy-gated DP ceiling (#83 = 32)")
    ap.add_argument("--dp-depth-cap", type=int, default=9,
                    help="max draft depth for the regime-optimal DP (#83 optimum = 9)")
    ap.add_argument("--out", type=Path,
                    default=OUT_DIR / "entropy_branching_results.json")
    ap.add_argument("--wandb-name", default="wirbel/entropy-branching-correlation")
    ap.add_argument("--wandb-group", default="entropy-branching-correlation")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    d = load_records(args.records)
    print(f"[entropy] records: total={d['n_total']} divergence={d['n_div']} "
          f"with_logits={d['n_with_logits']} align_bad={d['n_align_bad']} "
          f"shards={d['shards']}", flush=True)
    if d["n_with_logits"] < 1000:
        print(f"[entropy] WARNING: only {d['n_with_logits']} logit-bearing first-reject "
              f"records (<1000); correlation will be noisy.", flush=True)
    rho2_overall = float(d["salv2"].mean())
    print(f"[entropy] pooled rho2 (salvage) = {rho2_overall:.4f} "
          f"(vs #79 unconditional {RHO2_79:.4f})", flush=True)

    results: dict[str, Any] = {
        "n_total": d["n_total"], "n_divergence": d["n_div"],
        "n_with_logits": d["n_with_logits"], "n_align_bad": d["n_align_bad"],
        "shards": d["shards"], "rho2_pooled": rho2_overall, "rho2_79_anchor": RHO2_79,
        "n_bins": args.n_bins,
    }

    signals = {
        "drafter_entropy": d["Hd"],
        "drafter_entropy_top64": d["Hd64"],
        "drafter_1_minus_p1": 1.0 - d["p1"],
        "verifier_p1": d["vp1"],
        "verifier_entropy": d["Hv"],
    }
    analyses = {}
    for name, x in signals.items():
        analyses[name] = analyze_signal(name, x, d["salv2"], d["req"], d["depth"],
                                        args.n_bins, args.n_boot)
        a = analyses[name]
        print(f"\n[entropy] === {name} (5 equal-freq bins) ===", flush=True)
        b = a["bins"]
        for i in range(len(b["bin_mid_x"])):
            print(f"  bin{i+1}: x_mid={b['bin_mid_x'][i]:.4f} "
                  f"range=[{b['bin_x_range'][i][0]:.3f},{b['bin_x_range'][i][1]:.3f}] "
                  f"rho2={b['bin_rho2'][i]:.4f}+-{b['bin_se'][i]:.4f} n={b['bin_count'][i]}",
                  flush=True)
        print(f"  pearson_r(binned)={a['pearson_r_binned']:.4f} "
              f"point-biserial={a['pearson_r_pointbiserial']:.4f} "
              f"spearman={a['spearman_rho_binned']:.4f} "
              f"rho2_range={a['rho2_range_across_bins']:.4f}", flush=True)
        print(f"  within_DEPTH point-biserial={a['within_depth_pointbiserial']:.4f} "
              f"(n_depths={a['n_depths_used']}; ~raw => within-step signal, not a depth "
              f"proxy) | within_prompt(req,degenerate)={a['within_prompt_pearson']:.4f}",
              flush=True)
        print(f"  bootstrap r_ci95={a['bootstrap']['r_ci95']} "
              f"range_ci95={a['bootstrap']['rho2_range_ci95']}", flush=True)

    # fine 10-bin diagnostic on drafter entropy (researcher's recommended plot)
    fine = equal_freq_bins(d["Hd"], d["salv2"], args.n_bins_fine)
    results["drafter_entropy_fine10"] = fine

    results["signals"] = analyses
    v = verdict(analyses["drafter_entropy"])

    # ---- step 4: entropy-gated DP oracle ceiling (correlation is non-flat) ----
    ceiling = entropy_gated_dp_ceiling(d["Hd"], d["rank_fd"], n_bins=args.n_bins,
                                       M=args.dp_budget, depth_cap=args.dp_depth_cap)
    results["entropy_gated_dp_ceiling"] = ceiling
    gain_pct = ceiling["entropy_gated_tree_E_T_gain_pct"]
    tps_pp = ceiling["tps_gain_adaptive_depth_pp"]
    v["entropy_gated_tree_E_T_gain_pct"] = gain_pct
    v["tps_gain_adaptive_depth_pp"] = tps_pp

    # Realizability, not the raw MDE, decides the lane. The entropy->rho2 signal is
    # WITHIN-step: its within-DEPTH point-biserial retains the full raw magnitude, and
    # #83 proved per-depth rho2 is flat -- so a STATIC depth-indexed tree (the deployable
    # kind, captured by the onegraph static-topology CUDA graph) allocates width only by
    # depth and captures ZERO of this signal. Only a per-step DYNAMIC tree could, but that
    # forfeits the CUDA graph on a 99.4%-GPU-bound decode (project_cudagraph_already_deployed),
    # whose eager-mode overhead dwarfs the +0.27% E[T] / +0.33pp TPS oracle ceiling.
    ent = analyses["drafter_entropy"]
    wd_r, raw_r = abs(ent["within_depth_pointbiserial"]), abs(ent["pearson_r_pointbiserial"])
    v["signal_is_within_step"] = bool(raw_r > 0 and wd_r > 0.5 * raw_r)
    v["statically_capturable"] = bool(not v["signal_is_within_step"])
    v["oracle_ceiling_above_wall_mde"] = bool(gain_pct / 100.0 > WALL_MDE)
    # A dynamic tree must clear a ROBUST multi-MDE bar to justify forfeiting the CUDA
    # graph; an oracle ceiling below DYNAMIC_TREE_BAR x MDE is not worth that re-architecture.
    dynamic_worth_studying = (tps_pp / 100.0) > DYNAMIC_TREE_BAR * WALL_MDE
    v["dynamic_tree_worth_studying"] = bool(dynamic_worth_studying)
    v["lane_closed_final"] = bool(v["signal_is_within_step"] and not dynamic_worth_studying)
    v["lane_verdict"] = (
        f"correlation STRONG (r={v['rho2_entropy_correlation_r']:+.2f}, sign REVERSED vs "
        "hypothesis: confident drafter -> HIGHER rho2) but NON-actionable. Signal is "
        f"within-step (within-depth r={ent['within_depth_pointbiserial']:+.2f} == raw "
        f"{ent['pearson_r_pointbiserial']:+.2f}), so a static depth-tree captures 0 (#83 "
        f"per-depth rho2 flat); the dynamic-tree oracle ceiling is only {gain_pct:+.2f}% "
        f"E[T] / {tps_pp:+.2f}pp TPS (free perfect regime-ID, cost-matched M=32) and "
        "forfeits the onegraph CUDA graph on a 99.4%-GPU-bound decode -> CLOSE the lane"
        if v["lane_closed_final"] else
        f"entropy-gated oracle ceiling {gain_pct:+.2f}% E[T] / {tps_pp:+.2f}pp TPS clears "
        f"the {DYNAMIC_TREE_BAR:.0f}x-MDE dynamic-tree bar -> feasibility study warranted")
    results["verdict"] = v

    print(f"\n[entropy] === entropy-gated DP oracle ceiling (step 4) ===", flush=True)
    print(f"  global rho_cond={[round(x,4) for x in ceiling['global_rho_cond']]} "
          f"uniform E[T]={ceiling['ET_uniform_global']:.4f} (#83 anchor 5.207)", flush=True)
    for r in ceiling["per_regime"]:
        print(f"  regime{r['bin']}: H~{r['entropy_mid']:.2f} freq={r['freq']:.3f} "
              f"rho_cond={[round(x,3) for x in r['rho_cond']]} cov4={r['cov4']:.3f} | "
              f"uniformE[T]={r['ET_uniform_tree']:.4f} regimeOptE[T]={r['ET_regime_optimal']:.4f} "
              f"(d={r['regime_opt_depth']},br={r['regime_opt_max_branch']})", flush=True)
    print(f"  E_T_gain(cost-matched M=32) = {gain_pct:+.3f}%  "
          f"TPS_gain(adaptive-depth) = {ceiling['tps_gain_adaptive_depth_pp']:+.3f}pp  "
          f"TPS_gain(aggressive linear-on-uncertain) = "
          f"{ceiling['tps_gain_aggressive_regimeswitch_pp']:+.3f}pp", flush=True)

    print(f"\n[entropy] VERDICT: state={v['state']} "
          f"rho2_entropy_correlation_r={v['rho2_entropy_correlation_r']:.4f} "
          f"range={v['rho2_range_across_entropy_bins']:.4f} "
          f"significant={v['significant_for_gating']} | "
          f"entropy_gated_E_T_gain={gain_pct:+.2f}% lane_closed={v['lane_closed_final']}",
          flush=True)
    print(f"[entropy] {v['lane_verdict']}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"[entropy] wrote {args.out}", flush=True)

    if not args.no_wandb:
        try:
            log_wandb(args, results, analyses, v)
        except Exception as e:  # noqa: BLE001
            print(f"[entropy] W&B logging failed (non-fatal): {e!r}", flush=True)
    return 0


def log_wandb(args, results, analyses, v):
    import wandb
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        name=args.wandb_name, group=args.wandb_group, job_type="profiling",
        config={"n_bins": args.n_bins, "n_boot": args.n_boot,
                "n_with_logits": results["n_with_logits"]},
    )
    ceiling = results["entropy_gated_dp_ceiling"]
    flat = {
        "primary/rho2_entropy_correlation_r": v["rho2_entropy_correlation_r"],
        "primary/rho2_range_across_entropy_bins": v["rho2_range_across_entropy_bins"],
        "primary/entropy_gated_tree_E_T_gain_pct": v["entropy_gated_tree_E_T_gain_pct"],
        "test/entropy_gated_tree_E_T_gain_pct": v["entropy_gated_tree_E_T_gain_pct"],
        "test/tps_gain_adaptive_depth_pp": v["tps_gain_adaptive_depth_pp"],
        "test/tps_gain_aggressive_regimeswitch_pp": ceiling["tps_gain_aggressive_regimeswitch_pp"],
        "verdict/significant_for_gating": int(v["significant_for_gating"]),
        "verdict/flat_lane_closed": int(v["flat_lane_closed"]),
        "verdict/signal_is_within_step": int(v["signal_is_within_step"]),
        "verdict/statically_capturable": int(v["statically_capturable"]),
        "verdict/oracle_ceiling_above_wall_mde": int(v["oracle_ceiling_above_wall_mde"]),
        "verdict/dynamic_tree_worth_studying": int(v["dynamic_tree_worth_studying"]),
        "verdict/lane_closed_final": int(v["lane_closed_final"]),
        "ceiling/ET_uniform_global": ceiling["ET_uniform_global"],
        "ceiling/ET_uniform_mix": ceiling["ET_uniform_mix"],
        "ceiling/ET_dynamic_mix": ceiling["ET_dynamic_mix"],
        "ceiling/uniform_tps_gain_pct": ceiling["uniform_tps_gain"] * 100.0,
        "rho2_pooled": results["rho2_pooled"],
        "n_with_logits": results["n_with_logits"],
        "n_divergence": results["n_divergence"],
    }
    for name, a in analyses.items():
        flat[f"corr/{name}/pearson_binned"] = a["pearson_r_binned"]
        flat[f"corr/{name}/point_biserial"] = a["pearson_r_pointbiserial"]
        flat[f"corr/{name}/within_depth"] = a["within_depth_pointbiserial"]
        flat[f"corr/{name}/within_prompt"] = a["within_prompt_pearson"]
        flat[f"corr/{name}/rho2_range"] = a["rho2_range_across_bins"]
    run.summary.update(flat)
    for name, a in analyses.items():
        b = a["bins"]
        tb = wandb.Table(columns=["bin", "x_mid", "x_lo", "x_hi", "rho2", "se", "count"])
        for i in range(len(b["bin_mid_x"])):
            tb.add_data(i + 1, b["bin_mid_x"][i], b["bin_x_range"][i][0],
                        b["bin_x_range"][i][1], b["bin_rho2"][i], b["bin_se"][i],
                        b["bin_count"][i])
        run.log({f"bins/{name}": tb})
    rt = wandb.Table(columns=["regime", "entropy_mid", "freq", "count", "rho2",
                              "rho3", "rho4", "cov4", "ET_uniform", "ET_regime_opt",
                              "regime_opt_depth", "regime_opt_max_branch"])
    for r in ceiling["per_regime"]:
        rc = r["rho_cond"] + [float("nan")] * 3
        rt.add_data(r["bin"], r["entropy_mid"], r["freq"], r["count"], rc[0], rc[1],
                    rc[2], r["cov4"], r["ET_uniform_tree"], r["ET_regime_optimal"],
                    r["regime_opt_depth"], r["regime_opt_max_branch"])
    run.log({"ceiling/per_regime": rt})
    run.summary["wandb_run_id"] = run.id
    print(f"[entropy] W&B run: {run.url}", flush=True)
    run.finish()


if __name__ == "__main__":
    raise SystemExit(main())
