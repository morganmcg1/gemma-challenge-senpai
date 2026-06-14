#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Tightened private-safe 500-frontier + land #71 min-recovery BUILD GATE (PR #162).

WHAT THIS IS
------------
fern #149 mapped the (spread lambda x width mu) recovery square at the clear-500 bar
E[T]>=4.862 with NO private axis -- only a ~3% top-right corner (green-area 0.0300)
cleared 500 publicly. Two MERGED legs moved that corner in OPPOSITE directions and
were never folded into the frontier:

  * ubel #154  LOWERED the clear-500 bar to E[T]>=4.808-4.820 (decode-path scatter +
    LogitsProcessor-wrapper avoidance shaved 0.86-1.11% off the depth-9 step; greedy-
    token-identical, equivalence_rate=1.0). A lower bar WIDENS the green corner.
  * stark #151 showed the public projection is only PRIVATE-MARGINAL: the descent-only
    topology absorbs a 5.89% private drop, the both-bugs topology 9.88%, against a
    documented 4-9% private band. The private haircut SHRINKS the safe corner.

This instrument re-draws #149's frontier under the NEW bar, adds stark's private-
stability as a SECOND axis, and -- the actionable deliverable -- INVERTS the frontier
into the minimum per-bug realized recovery (lambda_min, mu_min) land #71's descent
kernel must reach so the realized point lands PRIVATE-SAFE-GREEN at P(clear-500)>=0.5
(and separately P>=0.9). It is the inverse of #149: #149 answered "given (lambda,mu),
do we clear?"; this answers "to clear privately, how good must (lambda,mu) be?" -- a
PRE-build de-risk gate land checks the instant its kernel assembles, not after.

PURE-ANALYTIC, CPU-ONLY. No GPU / vLLM / HF Job / submission / served-file change.
BASELINE stays 481.53; adds 0 TPS -- a build-DE-RISKING decision leg. IMPORTS the
committed leg outputs (does NOT re-derive them), exactly as the #155 consolidator does.

THE THREE COMPOSED LEGS (one source of truth per constant)
----------------------------------------------------------
  (1) #149 build-recovery surface  : (lambda,mu) -> public tree E[T] (joint_et) ->
      public TPS = K_cal*E[T]/step. lambda = deep-spine-SPREAD recovery (q[2:] meas->
      rho-opt), mu = branch-WIDTH recovery (rho_cond 0->opt); depth-1 held at rho-opt
      (q1=0.7287, the both-bugs/BUG-1-fixed value). Imported VERBATIM.
  (2) ubel #154 bar-lowering = step reduction : bar E[T] = 500*step/K_cal, so a smaller
      step lowers the E[T] bar. Imported recoverable_step_pct {conservative 0.857%,
      realistic 1.108%} off the lawine #136 step 1.2182 -> steps {4.862:1.2182,
      4.820:1.2078,4.808:1.2047}.
  (3) stark #151 private haircut : a private linear-drop d retains a fraction r_tree(d)
      of the PUBLIC tree E[T] (E_priv = E_pub*r_tree(d)). r_tree(d) is piecewise-linear
      through stark's BANKED anchors per topology, so the breakevens reproduce EXACTLY:
        both_bugs  : r(0)=1, r(0.043)=0.9717, r(0.0988)=bar/5.207=0.9338  (be 9.88%)
        descent_only: r(0)=1, r(0.043)=0.9721, r(0.0589)=bar/5.056=0.9616 (be 5.89%)
      The retention is ~topology-independent at a given d (GT: 0.9717 vs 0.9721, 0.04%).

THE PROBABILITY MODEL (fed to / from the #155 consolidator -- one source of truth)
---------------------------------------------------------------------------------
P(clear 500) on the PRIVATE projection uses the consolidator's exact quadrature:
    proj = K_cal * E_pub(lambda,mu) * r_tree(d) / step
    combined_rel = sqrt(sampling^2 + calibration^2 + step_anchor^2)   (#146/#148/#136)
    P = Phi( (proj - 500) / (combined_rel*proj) )
P>=0.5  <=> proj>=500 (sigma-independent boundary).
P>=0.9  <=> proj*(1 - z90*combined_rel) >= 500, z90 = Phi^-1(0.9) = 1.281552.

THE INVERSION (the build gate)
------------------------------
private-safe-GREEN(d,P) = { (lambda,mu) : P(clear 500 | private drop d) >= P }. Its
lower-left boundary is the iso-contour lambda_min(mu): the MINIMUM spread recovery at
each width recovery. Reuses #149's extract_iso_contour / green_area_fraction VERBATIM
on the private (P=0.5) and lower-confidence-bound (P=0.9) TPS grids. The headline pair
lambda_mu_min_private_safe = (lambda_min@mu=1, mu_min@lambda=1) -- each axis intercept
sits EXACTLY on the P=0.5 boundary by construction.

Distinct from stark #156 (which MEASURES the private drop this constraint USES via
--private-drop), wirbel #160 (the BUG-1 depth-1 spine build spec), denken #158 (per-
token exactness), kanna #159 (sigma_hw). Serves nothing -> greedy identity untouched.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import sys

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


# ============================================================================
# Import the merged legs VERBATIM (one source of truth per constant).
# ============================================================================
decomp = _load("deep_spine_width_spread_decomp",
               os.path.join(_HERE, "deep_spine_width_spread_decomp.py"))   # fern #149
cons = _load("approval_projection_consolidator",
             os.path.join(_HERE, "approval_projection_consolidator.py"))   # fern #155

# #149 surface machinery (reused, never re-derived)
load_banked = decomp.load_banked
build_lattice = decomp.build_lattice
joint_et = decomp.joint_et
joint_tps = decomp.joint_tps
joint_frontier_surface = decomp.joint_frontier_surface
extract_iso_contour = decomp.extract_iso_contour
green_area_fraction = decomp.green_area_fraction
official_tps_map = decomp.official_tps_map
accept_length_for_official = decomp.accept_length_for_official

K_CAL = decomp.K_CAL                              # 125.268
TARGET_OFFICIAL = decomp.TARGET_OFFICIAL          # 500.0
FRONTIER_OFFICIAL = decomp.FRONTIER_OFFICIAL      # 481.53
TARGET_530 = decomp.TARGET_530                    # 530.0
STEP_MEASURED_DEPTH9 = decomp.STEP_MEASURED_DEPTH9  # 1.2182 (lawine #136 merged)
STEP_ROOFLINE_DEPTH9 = decomp.STEP_ROOFLINE_DEPTH9  # 1.2127
TAU = decomp.TAU                                  # {"low":0.9924..,"central":1,"high":1}
E_T_LINEAR = decomp.E_T_LINEAR                    # 3.844
E_T_TREE = decomp.E_T_TREE                        # 5.207
RHO2_BRANCH_HIT = decomp.RHO2_BRANCH_HIT          # 0.4165
DEPTH1_FP32_TARGET = decomp.DEPTH1_FP32_TARGET    # 0.7287 (both-bugs/rho-opt depth-1)

# #155 consolidator probability model (reused, never re-derived)
SamplingModel = cons.SamplingModel
load_kcal_band = cons.load_kcal_band
build_joint_b_dict = cons.build_joint_b_dict
_phi = cons._phi
validity_gate = cons.validity_gate
ORACLE_E_T = cons.ORACLE_E_T                      # 2.621
STEP_REL_1SIGMA_DEFAULT = cons.STEP_REL_1SIGMA_DEFAULT  # 0.005

# committed artifacts (the two legs this PR folds in)
UBEL154_JSON = os.path.join(_ROOT, "research/spec_cost_model/step_denominator_reduction_audit.json")
STARK151_JSON = os.path.join(_ROOT, "research/validity/tree_private_acceptance_gap/results.json")

# one-sided z for P>=0.9 (standard-normal 0.9 quantile; Phi(z)=0.9)
Z_P90_ONESIDED = 1.2815515594457412
DEPTH1_DESCENT_ONLY = 0.679                        # stark #151 BUG-1-unfixed depth-1

# stark #156's in-flight pinned-drop candidates (reported; --private-drop overrides).
STARK156_CANDIDATE_DROPS = [0.043, 0.113, 0.196]


def _finite(x, default: float = 0.0) -> float:
    try:
        return float(x) if (x is not None and math.isfinite(float(x))) else default
    except (TypeError, ValueError):
        return default


# ============================================================================
# Leg import 1 -- ubel #154 bar-lowering (committed recoverable_step_pct).
# ============================================================================
def load_ubel154_bars(path: str = UBEL154_JSON,
                      base_step: float = STEP_MEASURED_DEPTH9) -> dict:
    """The three operative (bar, step) points: original 4.862 plus ubel #154's
    conservative/realistic step reductions. bar_E[T] = 500*step/K_cal."""
    with open(path) as f:
        u = json.load(f)
    rp_cons = float(u["recoverable_step_pct"]) / 100.0           # 0.00857
    rp_real = float(u["recoverable_step_pct_realistic"]) / 100.0  # 0.01108
    pts = {}
    for label, frac in (("original", 0.0), ("conservative", rp_cons), ("realistic", rp_real)):
        step = base_step * (1.0 - frac)
        pts[label] = {"step": step, "bar_et": accept_length_for_official(TARGET_OFFICIAL, step, 1.0),
                      "step_reduction_pct": frac * 100.0}
    return {
        "points": pts,
        "recoverable_step_pct_conservative": rp_cons * 100.0,
        "recoverable_step_pct_realistic": rp_real * 100.0,
        "base_step": base_step,
        "source_headline": u.get("headline", ""),
    }


# ============================================================================
# Leg import 2 -- stark #151 private retention r_tree(d) (committed breakevens).
# ============================================================================
def load_stark151_retention(path: str = STARK151_JSON,
                            step: float = STEP_MEASURED_DEPTH9) -> dict:
    """r_tree(d) = retained fraction of PUBLIC tree E[T] after a private linear-drop d,
    piecewise-linear through stark's BANKED anchors per topology so the breakevens
    reproduce exactly. Anchors per topology: d=0 -> 1; d=GT (4.3%) -> 1 - et_drop_pct;
    d=breakeven -> bar/E_pub (E[T] sits on the 500 bar)."""
    with open(path) as f:
        s = json.load(f)
    bar = accept_length_for_official(TARGET_OFFICIAL, step, 1.0)   # 4.86238 @ 1.2182
    pub = s["public_descent_walk_reference"]
    v = s["verdict"]
    head = s["headline"]
    et_both = float(pub["et_both_bugs"])               # 5.20695
    et_desc = float(pub["et_descent_only"])            # 5.05640
    be_both = float(v["breakeven_private_drop_both_bugs"])     # 0.09880
    be_desc = float(v["breakeven_private_drop_descent_only"])  # 0.05891
    gt = float(head["ground_truth_drop"])              # 0.043
    drop_both_gt = float(head["both_bugs_et_drop_pct_vs_public"]) / 100.0     # 0.02828
    drop_desc_gt = float(head["descent_only_et_drop_pct_vs_public"]) / 100.0  # 0.02786
    knots = {
        "both_bugs": [(0.0, 1.0), (gt, 1.0 - drop_both_gt), (be_both, bar / et_both)],
        "descent_only": [(0.0, 1.0), (gt, 1.0 - drop_desc_gt), (be_desc, bar / et_desc)],
    }
    return {
        "knots": knots,
        "breakeven_both_bugs": be_both, "breakeven_descent_only": be_desc,
        "ground_truth_drop": gt,
        "et_both_public": et_both, "et_descent_public": et_desc,
        "official_both_public": float(pub["official_both_bugs"]),
        "official_descent_public": float(pub["official_descent_only"]),
        "tree_private_tps_proj": float(head["tree_private_tps_proj"]),
        "bar_et_at_step": bar,
        "band_low": 0.04, "band_high": 0.09,
    }


def r_tree(d: float, topo: str, knots: dict) -> float:
    """Piecewise-linear retention through stark's banked anchors; linearly extrapolated
    (clamped monotone-decreasing, in [0,1]) beyond the breakeven knot."""
    pts = knots[topo]
    if d <= pts[0][0]:
        return 1.0
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        if d <= x1:
            return _finite(y0 + (y1 - y0) * (d - x0) / (x1 - x0), 1.0)
    # extrapolate with the last segment's slope, clamp to [0,1].
    x0, y0 = pts[-2]
    x1, y1 = pts[-1]
    slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0.0
    return max(0.0, min(1.0, _finite(y1 + slope * (d - x1), y1)))


# ============================================================================
# b_dict per depth-1 state (BUG-1 fixed = rho-opt spine; BUG-1 unfixed = descent-only).
# ============================================================================
def b_dict_for_depth1(b_base: dict, depth1: float) -> dict:
    """Copy the #149 banked b_dict with depth-1 overridden. joint_spine uses q76[0] as
    depth-1, so this is the only field that changes the BUG-1 state of the surface."""
    b = copy.deepcopy(b_base)
    b["q76"] = [float(depth1)] + list(b_base["q76"][1:])
    b["_L"] = build_lattice(b)
    return b


# ============================================================================
# ET surface (step-INVARIANT) -- computed once, scaled to any (step, retention).
# ============================================================================
def et_surface(b_dict: dict, n: int) -> dict:
    """The (lambda,mu) public tree-E[T] grid via #149's surface at step=1 (so its
    tps_grid == K_cal*E[T]); E[T] itself is step-invariant."""
    surf = joint_frontier_surface(b_dict, 1.0, 1.0, n)   # tps_grid_here = K_cal*E[T]
    return surf


def tps_grid_from_et(et_surf: dict, step: float, retention: float = 1.0) -> dict:
    """Scale the cached K_cal*E[T] grid to official TPS at `step` with private
    `retention`: tps = K_cal*E[T]*retention/step. Returns an extract_iso_contour-
    compatible mini-surface."""
    base = et_surf["tps_grid"]   # == K_cal * E[T]
    factor = retention / step
    grid = [[_finite(v * factor) for v in row] for row in base]
    return {"lambda_axis": et_surf["lambda_axis"], "mu_axis": et_surf["mu_axis"],
            "tps_grid": grid, "n": et_surf["n"]}


def lcb_grid_p90(et_surf: dict, step: float, retention: float, sampling, calib_rel: float,
                 step_rel: float) -> dict:
    """Lower-confidence-bound grid for P>=0.9: LCB = proj*(1 - z90*combined_rel), with
    per-cell combined_rel from the consolidator's quadrature on the PRIVATE E[T]. The
    P>=0.9 private-safe region is exactly {LCB >= 500}."""
    base = et_surf["tps_grid"]            # K_cal*E[T]
    lam_axis, mu_axis = et_surf["lambda_axis"], et_surf["mu_axis"]
    grid = []
    for row in base:
        out = []
        for v in row:
            et_priv = (v / K_CAL) * retention            # private E[T]
            proj = _finite(v * retention / step)         # private proj TPS
            samp = sampling.samp_rel_1sigma(et_priv)
            crel = math.sqrt(samp ** 2 + calib_rel ** 2 + step_rel ** 2)
            out.append(_finite(proj * (1.0 - Z_P90_ONESIDED * crel)))
        grid.append(out)
    return {"lambda_axis": lam_axis, "mu_axis": mu_axis, "tps_grid": grid, "n": et_surf["n"]}


# ============================================================================
# Private P(clear-500) at a single (lambda,mu) -- consolidator quadrature verbatim.
# ============================================================================
def private_p_clear(b_dict, lam, mu, d, step, retention_knots, topo, sampling,
                    calib_rel, step_rel):
    et_pub = joint_et(lam, mu, b_dict)
    rt = r_tree(d, topo, retention_knots)
    et_priv = et_pub * rt
    proj = _finite(K_CAL * et_priv / step)
    samp = sampling.samp_rel_1sigma(et_priv)
    crel = math.sqrt(samp ** 2 + calib_rel ** 2 + step_rel ** 2)
    sigma = crel * proj
    p = _phi((proj - TARGET_OFFICIAL) / sigma) if sigma > 0 else (1.0 if proj >= TARGET_OFFICIAL else 0.0)
    return {"et_public": _finite(et_pub), "r_tree": _finite(rt), "et_private": _finite(et_priv),
            "private_proj_tps": proj, "combined_rel": _finite(crel),
            "p_clear_500": _finite(p)}


# ============================================================================
# The INVERSION -- lambda_min(mu) build-gate contour + axis intercepts.
# ============================================================================
def invert_build_gate(et_surf, step, d, retention_knots, topo, sampling, calib_rel, step_rel):
    """Reuse #149's iso-contour/green-area on the private (P=0.5) and LCB (P=0.9) grids.
    lambda_min(mu) is the minimum spread recovery at each width recovery; the axis
    intercepts are the per-bug minima (each assumes the OTHER facet fully recovered)."""
    rt = r_tree(d, topo, retention_knots)
    # P>=0.5: private proj>=500  <=>  public (K_cal*E[T]) >= 500*step/retention.
    priv_surface = tps_grid_from_et(et_surf, step, rt)
    iso_p50 = extract_iso_contour(priv_surface, TARGET_OFFICIAL)
    area_p50 = green_area_fraction(priv_surface, TARGET_OFFICIAL)
    # P>=0.9: LCB>=500.
    lcb_surface = lcb_grid_p90(et_surf, step, rt, sampling, calib_rel, step_rel)
    iso_p90 = extract_iso_contour(lcb_surface, TARGET_OFFICIAL)
    area_p90 = green_area_fraction(lcb_surface, TARGET_OFFICIAL)
    return {
        "private_drop": d, "r_tree": _finite(rt), "topo": topo, "step": step,
        "P50": {
            "lambda_min_at_mu1": iso_p50["lambda_intercept_at_mu1"],
            "mu_min_at_lambda1": iso_p50["mu_intercept_at_lambda1"],
            "lambda_min_of_mu": iso_p50["lambda_iso_of_mu"],
            "private_safe_area_fraction": _finite(area_p50),
            "rows_with_crossing": iso_p50["rows_with_crossing"], "rows_total": iso_p50["rows_total"],
        },
        "P90": {
            "lambda_min_at_mu1": iso_p90["lambda_intercept_at_mu1"],
            "mu_min_at_lambda1": iso_p90["mu_intercept_at_lambda1"],
            "lambda_min_of_mu": iso_p90["lambda_iso_of_mu"],
            "private_safe_area_fraction": _finite(area_p90),
            "rows_with_crossing": iso_p90["rows_with_crossing"], "rows_total": iso_p90["rows_total"],
        },
    }


# ============================================================================
# Self-test (PRIMARY) -- the four PR assertions (a)-(d).
# ============================================================================
def self_test(et_surf_both, et_surf_desc, b_both, ubel, stark, sampling, calib_rel, step_rel):
    knots = stark["knots"]
    step_real = ubel["points"]["realistic"]["step"]
    step_orig = ubel["points"]["original"]["step"]

    # (a) at mu=1, the private drop d* at which lambda_min(mu=1)=1.0 reproduces stark's
    #     both-bugs breakeven (~9.88%). By the r_tree calibration, lambda=mu=1 (E_pub=
    #     5.207) hits exactly 500 when r_tree(d)=bar/5.207, i.e. d = breakeven. Verify the
    #     corner's private proj == 500 at d = breakeven (within tol).
    be_both = stark["breakeven_both_bugs"]
    corner = private_p_clear(b_both, 1.0, 1.0, be_both, step_orig, knots, "both_bugs",
                             sampling, calib_rel, step_rel)
    a_proj_at_be = corner["private_proj_tps"]
    a_ok = bool(abs(a_proj_at_be - TARGET_OFFICIAL) <= 1.0)            # corner sits on 500 bar
    # cross-check: lambda_min(mu=1) at d=breakeven is 1.0 (full spine needed).
    inv_be = invert_build_gate(et_surf_both, step_orig, be_both, knots, "both_bugs",
                               sampling, calib_rel, step_rel)
    lam_min_be = inv_be["P50"]["lambda_min_at_mu1"]
    a_lambda_ok = bool(lam_min_be is not None and lam_min_be >= 0.99)
    assert_a = bool(a_ok and a_lambda_ok)

    # (b) updated PUBLIC green-area >= #149's 0.0300 (lower bar widens); private-safe
    #     sub-region <= 0.0300 (private constraint shrinks). Quantify the net.
    pub_real = tps_grid_from_et(et_surf_both, step_real, 1.0)
    green_pub_real = green_area_fraction(pub_real, TARGET_OFFICIAL)
    pub_orig = tps_grid_from_et(et_surf_both, step_orig, 1.0)
    green_pub_orig = green_area_fraction(pub_orig, TARGET_OFFICIAL)   # == #149's 0.0300
    inv_gt_real = invert_build_gate(et_surf_both, step_real, stark["ground_truth_drop"],
                                    knots, "both_bugs", sampling, calib_rel, step_rel)
    green_priv = inv_gt_real["P50"]["private_safe_area_fraction"]
    b_widens = bool(green_pub_real >= 0.0300 - 1e-6)
    b_shrinks = bool(green_priv <= 0.0300 + 1e-6)
    assert_b = bool(b_widens and b_shrinks)

    # (c) the 3 bracketing anchors still land RED / GREEN / INDETERMINATE under the
    #     updated bar. boundary anchor tracks the updated bar (E[T]=updated bar -> 500).
    bar_real = ubel["points"]["realistic"]["bar_et"]
    def verdict_at(et, step):
        proj = official_tps_map(et, step, 1.0)
        proj_lo = official_tps_map(et, step, TAU["low"])
        if proj_lo >= TARGET_OFFICIAL:
            return "GREEN", proj
        if proj < TARGET_OFFICIAL:
            return "RED", proj
        return "INDETERMINATE", proj
    v_oracle, p_oracle = verdict_at(ORACLE_E_T, step_real)
    v_both, p_both = verdict_at(E_T_TREE, step_real)
    v_bound, p_bound = verdict_at(bar_real, step_real)          # updated-bar boundary
    assert_c = bool(v_oracle == "RED" and v_both == "GREEN" and v_bound == "INDETERMINATE")

    # (d) the (lambda_min,mu_min) gate intercepts map to exactly P=0.5 (proj==500) by
    #     construction. Evaluate the realistic-bar / GT-drop intercepts.
    lam_int = inv_gt_real["P50"]["lambda_min_at_mu1"]
    mu_int = inv_gt_real["P50"]["mu_min_at_lambda1"]
    pc_lam = private_p_clear(b_both, lam_int, 1.0, stark["ground_truth_drop"], step_real,
                             knots, "both_bugs", sampling, calib_rel, step_rel) if lam_int is not None else None
    pc_mu = private_p_clear(b_both, 1.0, mu_int, stark["ground_truth_drop"], step_real,
                            knots, "both_bugs", sampling, calib_rel, step_rel) if mu_int is not None else None
    d_lam_ok = bool(pc_lam is not None and abs(pc_lam["p_clear_500"] - 0.5) <= 0.02)
    d_mu_ok = bool(pc_mu is not None and abs(pc_mu["p_clear_500"] - 0.5) <= 0.02)
    assert_d = bool(d_lam_ok and d_mu_ok)

    passes = bool(assert_a and assert_b and assert_c and assert_d)
    return {
        "passes": passes,
        "assert_a_breakeven_reproduced": {
            "ok": assert_a, "breakeven_both_bugs": be_both,
            "corner_proj_at_breakeven": a_proj_at_be, "expected_proj": TARGET_OFFICIAL,
            "lambda_min_at_mu1_at_breakeven": lam_min_be, "expected_lambda_min": 1.0,
            "note": "at d=stark both-bugs breakeven, the (1,1) corner sits on the 500 bar "
                    "and lambda_min(mu=1)=1 (full spine needed) -> reproduces 9.88%."},
        "assert_b_green_area_moves_opposite": {
            "ok": assert_b, "public_green_area_orig_149": _finite(green_pub_orig),
            "public_green_area_realistic_bar": _finite(green_pub_real),
            "private_safe_green_area_gt_realistic": _finite(green_priv),
            "public_widens": b_widens, "private_shrinks": b_shrinks,
            "net_vs_149_public": _finite(green_pub_real - green_pub_orig),
            "net_vs_149_private": _finite(green_priv - green_pub_orig),
            "note": "lower bar widens public (>=0.0300); private haircut shrinks safe "
                    "sub-region (<=0.0300) -- the two legs move in opposite directions."},
        "assert_c_bracketing_anchors": {
            "ok": assert_c,
            "oracle_2621": {"verdict": v_oracle, "proj": _finite(p_oracle), "expect": "RED"},
            "both_bugs_5207": {"verdict": v_both, "proj": _finite(p_both), "expect": "GREEN"},
            "boundary_updated_bar": {"bar_et": _finite(bar_real), "verdict": v_bound,
                                     "proj": _finite(p_bound), "expect": "INDETERMINATE"},
            "note": "boundary anchor tracks the updated (realistic) bar 4.808; oracle/both-"
                    "bugs keep RED/GREEN -- the bracketing structure survives the lower bar."},
        "assert_d_gate_on_p50": {
            "ok": assert_d, "lambda_min_at_mu1": lam_int, "mu_min_at_lambda1": mu_int,
            "p_clear_at_lambda_intercept": (pc_lam["p_clear_500"] if pc_lam else None),
            "p_clear_at_mu_intercept": (pc_mu["p_clear_500"] if pc_mu else None),
            "note": "each axis intercept of the build gate sits on the P=0.5 contour "
                    "(private proj==500) by construction."},
    }


# ============================================================================
# Hand-off (step 5) -- is the BUG-1 depth-1 spine fix mandatory for the private shot?
# ============================================================================
def bug1_mandatory_analysis(et_surf_both, et_surf_desc, b_both, b_desc, ubel, stark,
                            sampling, calib_rel, step_rel, drops):
    """Does ubel #154's lower bar ALONE (without the both-bugs depth-1 spine) ever reach
    private-safe-GREEN? Compare the descent-only (BUG-1 unfixed, q1=0.679) full-recovery
    corner (lambda=mu=1) against the both-bugs corner across the private band, at the
    realistic lower bar. If the descent-only corner clears at a given drop, BUG-1 is NOT
    mandatory there; if only the both-bugs corner clears, BUG-1 IS mandatory."""
    knots = stark["knots"]
    rows = []
    for label in ("original", "realistic"):
        step = ubel["points"][label]["step"]
        for d in drops:
            pc_both = private_p_clear(b_both, 1.0, 1.0, d, step, knots, "both_bugs",
                                      sampling, calib_rel, step_rel)
            pc_desc = private_p_clear(b_desc, 1.0, 1.0, d, step, knots, "descent_only",
                                      sampling, calib_rel, step_rel)
            rows.append({
                "bar_label": label, "step": _finite(step), "private_drop_pct": _finite(d * 100.0),
                "descent_only_corner_proj": pc_desc["private_proj_tps"],
                "descent_only_corner_p_clear": pc_desc["p_clear_500"],
                "descent_only_clears_p50": bool(pc_desc["private_proj_tps"] >= TARGET_OFFICIAL),
                "both_bugs_corner_proj": pc_both["private_proj_tps"],
                "both_bugs_corner_p_clear": pc_both["p_clear_500"],
                "both_bugs_clears_p50": bool(pc_both["private_proj_tps"] >= TARGET_OFFICIAL),
                "bug1_mandatory_here": bool(pc_both["private_proj_tps"] >= TARGET_OFFICIAL
                                            and pc_desc["private_proj_tps"] < TARGET_OFFICIAL),
            })
    # the binding question: at the realistic bar, is there ANY band drop where descent-
    # only alone clears? and is the both-bugs spine required at the 9% ceiling?
    real_rows = [r for r in rows if r["bar_label"] == "realistic"]
    desc_clears_gt = next((r for r in real_rows
                           if abs(r["private_drop_pct"] - stark["ground_truth_drop"] * 100) < 1e-6), None)
    band_ceiling = next((r for r in real_rows if abs(r["private_drop_pct"] - 9.0) < 1e-6), None)
    bug1_mandatory_at_ceiling = bool(band_ceiling is not None and band_ceiling["bug1_mandatory_here"])
    bug1_mandatory_at_gt = bool(desc_clears_gt is not None and desc_clears_gt["bug1_mandatory_here"])
    return {
        "rows": rows,
        "descent_only_clears_at_gt_realistic": bool(desc_clears_gt is not None
                                                    and desc_clears_gt["descent_only_clears_p50"]),
        "bug1_mandatory_at_gt_realistic": bug1_mandatory_at_gt,
        "bug1_mandatory_at_band_ceiling_realistic": bug1_mandatory_at_ceiling,
        "verdict": ("BUG-1 spine (wirbel #160) is MANDATORY across the private band: even with"
                    " the lower bar + full descent, the BUG-1-unfixed corner falls below 500 at"
                    " the band ceiling." if bug1_mandatory_at_ceiling else
                    "BUG-1 spine is NOT strictly mandatory at the realistic bar for low band"
                    " drops: a full descent + the lower bar alone reaches private-safe-GREEN at"
                    " the ground-truth drop; it becomes mandatory only toward the band ceiling."),
    }


# ============================================================================
# Driver.
# ============================================================================
def run(args) -> dict:
    n = max(101, int(args.grid_n))
    step_rel = args.step_rel_half_width

    ubel = load_ubel154_bars(base_step=args.base_step)
    stark = load_stark151_retention(step=args.base_step)
    kcal_band = load_kcal_band()
    calib_rel = kcal_band["calib_downside_rel"]
    sampling = SamplingModel(n_steps=args.n_steps, n_boot=args.n_boot, seed=args.seed,
                             step=args.base_step, step_rel_hw=step_rel)

    # banked #149 b_dicts: BUG-1-fixed (depth-1 rho-opt) and BUG-1-unfixed (0.679).
    b_both = build_joint_b_dict(args.rho_json, args.oracle_json)
    b_desc = b_dict_for_depth1(b_both, DEPTH1_DESCENT_ONLY)
    et_surf_both = et_surface(b_both, n)   # step-invariant K_cal*E[T] grid (depth1 fixed)
    et_surf_desc = et_surface(b_desc, n)   # depth1 = 0.679

    knots = stark["knots"]

    # ---- (step 1) re-draw the public frontier at the three bars ----
    bars = {}
    for label, pt in ubel["points"].items():
        pub = tps_grid_from_et(et_surf_both, pt["step"], 1.0)
        bars[label] = {
            "step": _finite(pt["step"]), "bar_et": _finite(pt["bar_et"]),
            "step_reduction_pct": _finite(pt["step_reduction_pct"]),
            "public_green_area_fraction": _finite(green_area_fraction(pub, TARGET_OFFICIAL)),
            "iso500_lambda_intercept_mu1": extract_iso_contour(pub, TARGET_OFFICIAL)["lambda_intercept_at_mu1"],
            "iso500_mu_intercept_lambda1": extract_iso_contour(pub, TARGET_OFFICIAL)["mu_intercept_at_lambda1"],
        }
    green_149 = bars["original"]["public_green_area_fraction"]   # reproduces 0.0300

    # ---- (steps 2-3) private axis + inversion at each (bar x drop) ----
    report_drops = sorted(set([args.private_drop, stark["ground_truth_drop"], 0.09]
                              + STARK156_CANDIDATE_DROPS))
    gates = {}
    for bar_label in ("original", "conservative", "realistic"):
        step = ubel["points"][bar_label]["step"]
        gates[bar_label] = {}
        for d in report_drops:
            inv = invert_build_gate(et_surf_both, step, d, knots, "both_bugs",
                                    sampling, calib_rel, step_rel)
            gates[bar_label][f"{d*100:.1f}pct"] = inv

    # ---- headline build gate: realistic bar, --private-drop (default GT 4.3%), P>=0.5 ----
    hd_step = ubel["points"]["realistic"]["step"]
    hd = invert_build_gate(et_surf_both, hd_step, args.private_drop, knots, "both_bugs",
                           sampling, calib_rel, step_rel)
    lam_min_star = hd["P50"]["lambda_min_at_mu1"]      # min spread at full width
    mu_min_star = hd["P50"]["mu_min_at_lambda1"]       # min width at full spread
    lambda_mu_min_private_safe = [lam_min_star, mu_min_star]

    # ---- (step 4) self-test ----
    st = self_test(et_surf_both, et_surf_desc, b_both, ubel, stark, sampling, calib_rel, step_rel)
    tightened_frontier_self_test_passes = int(st["passes"])

    # ---- (step 5) is BUG-1 mandatory? ----
    bug1 = bug1_mandatory_analysis(et_surf_both, et_surf_desc, b_both, b_desc, ubel, stark,
                                   sampling, calib_rel, step_rel,
                                   drops=[stark["ground_truth_drop"], 0.06, 0.09])

    # ---- net green-area move (the (b) headline) ----
    green_pub_real = bars["realistic"]["public_green_area_fraction"]
    green_priv_real_gt = gates["realistic"][f"{stark['ground_truth_drop']*100:.1f}pct"]["P50"]["private_safe_area_fraction"]

    # ---- feed-the-consolidator block: what replaces #149's 0.0300 input ----
    consolidator_feed = {
        "replaces_fern149_green_area_input": green_149,
        "updated_public_green_area_realistic_bar": _finite(green_pub_real),
        "updated_private_safe_green_area_gt_realistic": _finite(green_priv_real_gt),
        "lambda_mu_min_build_gate_realistic_gt": lambda_mu_min_private_safe,
        "bracketing_anchors_under_updated_bar": st["assert_c_bracketing_anchors"],
        "note": ("the consolidator's decision-geometry leg compares land #71's measured "
                 "(spread_lambda,width_mu) against lambda_min(mu); the public green-area "
                 "input 0.0300 is replaced by the updated-bar public area and carved by "
                 "the private haircut."),
    }

    state = "ARMED" if st["passes"] else "SELF-TEST-FAIL"
    label = (
        f"TIGHTENED PRIVATE-500 FRONTIER {'VALIDATED' if st['passes'] else 'FAILED'}; build "
        f"gate @ realistic bar {ubel['points']['realistic']['bar_et']:.3f} + private drop "
        f"{args.private_drop*100:.1f}%: land #71 must reach (lambda_min={lam_min_star}, "
        f"mu_min={mu_min_star}) for P(clear-500)>=0.5. Public green-area {green_149*100:.2f}% "
        f"(#149) -> {green_pub_real*100:.2f}% (lower bar) -> {green_priv_real_gt*100:.2f}% "
        f"(private-safe). BUG-1 spine mandatory @ band ceiling: "
        f"{bug1['bug1_mandatory_at_band_ceiling_realistic']}.")

    out = {
        "primary_metric_name": "tightened_frontier_self_test_passes",
        "tightened_frontier_self_test_passes": tightened_frontier_self_test_passes,
        "test_metric_name": "lambda_mu_min_private_safe",
        "lambda_mu_min_private_safe": lambda_mu_min_private_safe,
        "lambda_min_at_mu1_realistic_gt": _finite(lam_min_star, -1.0) if lam_min_star is not None else -1.0,
        "mu_min_at_lambda1_realistic_gt": _finite(mu_min_star, -1.0) if mu_min_star is not None else -1.0,
        "gate_state": state, "gate_label": label,
        "self_test": st,
        "bars_public_frontier": bars,
        "green_area_149_reproduced": green_149,
        "build_gate_headline": {
            "bar_label": "realistic", "bar_et": _finite(ubel["points"]["realistic"]["bar_et"]),
            "step": _finite(hd_step), "private_drop_pct": _finite(args.private_drop * 100.0),
            "r_tree": hd["P50"].get("r_tree", _finite(r_tree(args.private_drop, "both_bugs", knots))),
            "lambda_min_at_mu1": lam_min_star, "mu_min_at_lambda1": mu_min_star,
            "lambda_min_of_mu_curve_P50": hd["P50"]["lambda_min_of_mu"],
            "lambda_min_at_mu1_P90": hd["P90"]["lambda_min_at_mu1"],
            "mu_min_at_lambda1_P90": hd["P90"]["mu_min_at_lambda1"],
            "private_safe_area_P50": hd["P50"]["private_safe_area_fraction"],
            "private_safe_area_P90": hd["P90"]["private_safe_area_fraction"],
        },
        "build_gates_all_bars_drops": gates,
        "green_area_net_move": {
            "fern149_public": green_149,
            "updated_public_realistic": _finite(green_pub_real),
            "private_safe_realistic_gt": _finite(green_priv_real_gt),
            "net_public_minus_149": _finite(green_pub_real - green_149),
            "net_private_minus_149": _finite(green_priv_real_gt - green_149),
        },
        "bug1_mandatory_handoff": bug1,
        "consolidator_feed": consolidator_feed,
        "private_retention_model": {
            "r_tree_knots": {k: [[_finite(x), _finite(y)] for x, y in v] for k, v in knots.items()},
            "r_tree_at_report_drops_both_bugs": {f"{d*100:.1f}pct": _finite(r_tree(d, "both_bugs", knots))
                                                 for d in report_drops},
            "r_tree_at_report_drops_descent_only": {f"{d*100:.1f}pct": _finite(r_tree(d, "descent_only", knots))
                                                    for d in report_drops},
            "breakeven_both_bugs": stark["breakeven_both_bugs"],
            "breakeven_descent_only": stark["breakeven_descent_only"],
            "stark156_candidate_drops": STARK156_CANDIDATE_DROPS,
            "note": ("E_private = E_public * r_tree(d); r_tree piecewise-linear through stark "
                     "#151 banked anchors -> breakevens reproduce exactly. ~topology-independent "
                     "at a given drop (GT 0.9717 both-bugs vs 0.9721 descent-only)."),
        },
        "uncertainty_model": {
            "combined_rel_formula": "sqrt(sampling^2 + calibration^2 + step_anchor^2)",
            "calibration_rel_148": _finite(calib_rel),
            "step_anchor_rel_136": _finite(step_rel),
            "sigma_descend_146": _finite(sampling.sigma_descend),
            "n_steps": sampling.n_steps,
            "z_p90_one_sided": Z_P90_ONESIDED,
            "source": "consolidator #155 quadrature (wirbel #146 sampling + ubel #148 "
                      "calibration + lawine #136 step-anchor), applied to the PRIVATE projection.",
        },
        "map": {
            "figure_of_merit": "private_official = K_cal*E_public*r_tree(d)/step",
            "K_cal": K_CAL, "base_step_136": args.base_step,
            "ubel154_recoverable_step_pct_conservative": ubel["recoverable_step_pct_conservative"],
            "ubel154_recoverable_step_pct_realistic": ubel["recoverable_step_pct_realistic"],
            "bar_et_original": _finite(ubel["points"]["original"]["bar_et"]),
            "bar_et_conservative": _finite(ubel["points"]["conservative"]["bar_et"]),
            "bar_et_realistic": _finite(ubel["points"]["realistic"]["bar_et"]),
            "target_official": TARGET_OFFICIAL, "frontier_official": FRONTIER_OFFICIAL,
            "private_band_low": stark["band_low"], "private_band_high": stark["band_high"],
            "tree_private_tps_proj_151": stark["tree_private_tps_proj"],
        },
        "provenance": (
            "INVERTS fern #149's joint (spread x width) frontier into land #71's minimum-"
            "recovery build gate. Imports VERBATIM: fern #149 surface/iso-contour/green-area, "
            "ubel #154 recoverable_step_pct (bar-lowering), stark #151 breakeven tolerances "
            "(private retention), fern #155 consolidator quadrature (P-levels). One source of "
            "truth per constant. depth-1 held at rho-opt 0.7287 (BUG-1 fixed) for the main "
            "surface; 0.679 (BUG-1 unfixed) for the mandatory-spine hand-off."),
        "method": ("LOCAL CPU-only analytic decision-geometry inversion; no GPU/vLLM/HF Job/"
                   "submission/kernel build. BASELINE stays 481.53; adds 0 TPS -- a pre-build "
                   "de-risk gate. Greedy identity untouched by construction."),
        "metrics_nan_clean": 1,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    _print_console(out, ubel, stark, args)

    if args.wandb and not args.no_wandb:
        try:
            _log_wandb(args, out, ubel, stark, sampling)
        except Exception as e:  # noqa: BLE001
            print(f"[tightened-frontier] W&B logging failed (non-fatal): {e!r}", flush=True)

    return out


def _print_console(out, ubel, stark, args):
    print("=" * 100)
    print("TIGHTENED PRIVATE-SAFE 500-FRONTIER + land #71 MIN-RECOVERY BUILD GATE (PR #162)")
    print("=" * 100)
    print(f"\nmap: private_official = K_cal*E_public*r_tree(d)/step  (K_cal={K_CAL:.3f}, "
          f"base step={args.base_step:.4f})\n")

    print("[STEP 1] PUBLIC frontier re-drawn at the three bars (lower bar widens green):")
    for label in ("original", "conservative", "realistic"):
        b = out["bars_public_frontier"][label]
        tag = "  (== #149 0.0300)" if label == "original" else ""
        print(f"  {label:13s} bar E[T]={b['bar_et']:.4f} step={b['step']:.4f} "
              f"(-{b['step_reduction_pct']:.3f}%)  public green-area={b['public_green_area_fraction']*100:.2f}%{tag}")

    print(f"\n[STEP 2] PRIVATE retention r_tree(d) (stark #151 banked; breakevens reproduce):")
    rt = out["private_retention_model"]["r_tree_at_report_drops_both_bugs"]
    for k, v in rt.items():
        print(f"  drop {k:>7s} -> r_tree(both-bugs)={v:.4f}")
    print(f"  breakeven both-bugs={stark['breakeven_both_bugs']*100:.3f}%  "
          f"descent-only={stark['breakeven_descent_only']*100:.3f}%  band [4%,9%]")

    print(f"\n[STEP 3] INVERSION -- build gate (realistic bar, drop {args.private_drop*100:.1f}%):")
    h = out["build_gate_headline"]
    print(f"  lambda_min(mu=1) = {h['lambda_min_at_mu1']}  (min SPREAD at full width)")
    print(f"  mu_min(lambda=1) = {h['mu_min_at_lambda1']}  (min WIDTH at full spread)")
    print(f"  P>=0.5 private-safe area = {h['private_safe_area_P50']*100:.2f}%; "
          f"P>=0.9 area = {h['private_safe_area_P90']*100:.2f}%")
    print(f"  lambda_min(mu) curve (P>=0.5):")
    for r in h["lambda_min_of_mu_curve_P50"]:
        if r["mu"] in (0.6, 0.7, 0.8, 0.9, 1.0):
            li = "n/a (unreachable)" if r["lambda_iso"] is None else f"{r['lambda_iso']:.3f}"
            print(f"      mu={r['mu']:.2f} -> lambda_min = {li}")

    print(f"\n[STEP 4] SELF-TEST (the four assertions):")
    st = out["self_test"]
    print(f"  (a) breakeven 9.88% reproduced @ (1,1)         -> {'OK' if st['assert_a_breakeven_reproduced']['ok'] else 'FAIL'}")
    bb = st["assert_b_green_area_moves_opposite"]
    print(f"  (b) public widens ({bb['public_green_area_orig_149']*100:.2f}%->"
          f"{bb['public_green_area_realistic_bar']*100:.2f}%), private shrinks "
          f"(->{bb['private_safe_green_area_gt_realistic']*100:.2f}%) -> {'OK' if bb['ok'] else 'FAIL'}")
    cc = st["assert_c_bracketing_anchors"]
    print(f"  (c) anchors RED/GREEN/INDET under updated bar   -> {'OK' if cc['ok'] else 'FAIL'} "
          f"({cc['oracle_2621']['verdict']}/{cc['both_bugs_5207']['verdict']}/{cc['boundary_updated_bar']['verdict']})")
    dd = st["assert_d_gate_on_p50"]
    print(f"  (d) gate intercepts on P=0.5 (proj==500)        -> {'OK' if dd['ok'] else 'FAIL'} "
          f"(p_lam={dd['p_clear_at_lambda_intercept']}, p_mu={dd['p_clear_at_mu_intercept']})")
    print(f"  => tightened_frontier_self_test_passes = {out['tightened_frontier_self_test_passes']}")

    print(f"\n[STEP 5] IS BUG-1 (depth-1 spine, wirbel #160) MANDATORY?")
    bug1 = out["bug1_mandatory_handoff"]
    print(f"  descent-only (BUG-1 unfixed) corner clears @ GT/realistic: "
          f"{bug1['descent_only_clears_at_gt_realistic']}")
    print(f"  BUG-1 mandatory @ band ceiling (realistic): {bug1['bug1_mandatory_at_band_ceiling_realistic']}")
    print(f"  -> {bug1['verdict']}")

    print(f"\n[PRIMARY] tightened_frontier_self_test_passes = {out['tightened_frontier_self_test_passes']}")
    print(f"[TEST]    lambda_mu_min_private_safe = {out['lambda_mu_min_private_safe']}")
    print(f"\n[STATE] {out['gate_state']} -- {out['gate_label']}")
    print(f"\nwrote {args.out}")


def _log_wandb(args, out, ubel, stark, sampling):
    import wandb
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     name=args.wandb_name, group=args.wandb_group, job_type="analysis",
                     config={"instrument": "tightened-private-500-frontier",
                             "method": "cpu-analytic-frontier-inversion-149-154-151-155",
                             "K_cal": K_CAL, "base_step": args.base_step,
                             "grid_n": max(101, int(args.grid_n)),
                             "private_drop": args.private_drop,
                             "ubel154_recoverable_pct_conservative": ubel["recoverable_step_pct_conservative"],
                             "ubel154_recoverable_pct_realistic": ubel["recoverable_step_pct_realistic"],
                             "stark151_breakeven_both": stark["breakeven_both_bugs"],
                             "stark151_breakeven_descent": stark["breakeven_descent_only"],
                             "target_official": TARGET_OFFICIAL, "frontier_official": FRONTIER_OFFICIAL,
                             "z_p90_one_sided": Z_P90_ONESIDED})
    s = wandb.summary
    s["tightened_frontier_self_test_passes"] = out["tightened_frontier_self_test_passes"]
    s["lambda_min_at_mu1_realistic_gt"] = out["lambda_min_at_mu1_realistic_gt"]
    s["mu_min_at_lambda1_realistic_gt"] = out["mu_min_at_lambda1_realistic_gt"]
    s["green_area_149_reproduced"] = out["green_area_149_reproduced"]
    s["public_green_area_conservative"] = out["bars_public_frontier"]["conservative"]["public_green_area_fraction"]
    s["public_green_area_realistic"] = out["bars_public_frontier"]["realistic"]["public_green_area_fraction"]
    s["private_safe_area_P50_realistic_gt"] = out["build_gate_headline"]["private_safe_area_P50"]
    s["private_safe_area_P90_realistic_gt"] = out["build_gate_headline"]["private_safe_area_P90"]
    s["net_public_minus_149"] = out["green_area_net_move"]["net_public_minus_149"]
    s["net_private_minus_149"] = out["green_area_net_move"]["net_private_minus_149"]
    s["bug1_mandatory_at_band_ceiling"] = int(out["bug1_mandatory_handoff"]["bug1_mandatory_at_band_ceiling_realistic"])
    s["descent_only_clears_at_gt"] = int(out["bug1_mandatory_handoff"]["descent_only_clears_at_gt_realistic"])
    for k in ("assert_a_breakeven_reproduced", "assert_b_green_area_moves_opposite",
              "assert_c_bracketing_anchors", "assert_d_gate_on_p50"):
        s[f"selftest_{k}"] = int(out["self_test"][k]["ok"])
    s["gate_state"] = out["gate_state"]
    s["gate_label"] = out["gate_label"]

    # bar-sweep table
    bt = wandb.Table(columns=["bar", "bar_et", "step", "step_reduction_pct", "public_green_area"])
    for label in ("original", "conservative", "realistic"):
        b = out["bars_public_frontier"][label]
        bt.add_data(label, b["bar_et"], b["step"], b["step_reduction_pct"], b["public_green_area_fraction"])
    wandb.log({"bar_sweep_public_frontier": bt})

    # build-gate curve table (realistic, headline drop, P50)
    ct = wandb.Table(columns=["mu", "lambda_min_P50"])
    for r in out["build_gate_headline"]["lambda_min_of_mu_curve_P50"]:
        ct.add_data(r["mu"], (-1.0 if r["lambda_iso"] is None else r["lambda_iso"]))
    wandb.log({"build_gate_lambda_min_of_mu": ct})

    # bar x drop gate table
    gt = wandb.Table(columns=["bar", "drop_pct", "r_tree", "lambda_min_mu1_P50", "mu_min_lam1_P50",
                              "private_safe_area_P50", "lambda_min_mu1_P90", "private_safe_area_P90"])
    for bar_label in ("original", "conservative", "realistic"):
        for dk, inv in out["build_gates_all_bars_drops"][bar_label].items():
            gt.add_data(bar_label, dk, inv["r_tree"],
                        (-1.0 if inv["P50"]["lambda_min_at_mu1"] is None else inv["P50"]["lambda_min_at_mu1"]),
                        (-1.0 if inv["P50"]["mu_min_at_lambda1"] is None else inv["P50"]["mu_min_at_lambda1"]),
                        inv["P50"]["private_safe_area_fraction"],
                        (-1.0 if inv["P90"]["lambda_min_at_mu1"] is None else inv["P90"]["lambda_min_at_mu1"]),
                        inv["P90"]["private_safe_area_fraction"])
    wandb.log({"build_gates_bar_x_drop": gt})

    # bug-1 mandatory table
    mt = wandb.Table(columns=["bar", "drop_pct", "descent_only_proj", "descent_clears",
                              "both_bugs_proj", "both_clears", "bug1_mandatory"])
    for r in out["bug1_mandatory_handoff"]["rows"]:
        mt.add_data(r["bar_label"], r["private_drop_pct"], r["descent_only_corner_proj"],
                    int(r["descent_only_clears_p50"]), r["both_bugs_corner_proj"],
                    int(r["both_bugs_clears_p50"]), int(r["bug1_mandatory_here"]))
    wandb.log({"bug1_mandatory_band": mt})
    print(f"\nW&B run: {run.id}  ({run.url})", flush=True)
    wandb.finish()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-step", type=float, default=STEP_MEASURED_DEPTH9,
                    help="lawine #136 merged depth-9 step (1.2182); ubel #154 reductions apply to it.")
    ap.add_argument("--private-drop", type=float, default=0.043,
                    help="private aggregate drop d (fraction). Default = stark #151 ground-truth "
                         "4.3%%; stark #156's pinned value drops straight in here when it lands.")
    ap.add_argument("--grid-n", type=int, default=101, help="surface grid per axis (>=101).")
    ap.add_argument("--step-rel-half-width", type=float, default=STEP_REL_1SIGMA_DEFAULT,
                    help="lawine #136/#147 step-anchor 1-sigma relative (default 0.5%%).")
    ap.add_argument("--n-steps", type=int, default=cons.env.ORACLE_STEPS,
                    help="verify-step budget for the sampling CI (oracle 1024).")
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="bootstrap resamples for the #146 sampling model (sigma_descend is "
                         "bootstrap-independent, so a small value suffices).")
    ap.add_argument("--seed", type=int, default=162)
    ap.add_argument("--rho-json", default=cons.RHO_OPT_JSON)
    ap.add_argument("--oracle-json", default=cons.ORACLE_LIVE_JSON)
    ap.add_argument("--out", default="research/spec_cost_model/tightened_private_500_frontier_results.json")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", "--wandb_project",
                    default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb-entity", "--wandb_entity",
                    default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb-name", "--wandb_name", default="fern/tightened-private-500-frontier")
    ap.add_argument("--wandb-group", "--wandb_group", default="tightened-private-500-frontier")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
