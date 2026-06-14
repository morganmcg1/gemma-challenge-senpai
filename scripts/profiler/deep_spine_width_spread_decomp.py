#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Deep-spine WIDTH-vs-SPREAD decomposition: the 537.8-vs-376 watched risk (PR #145).

WHAT THIS IS
------------
A single CPU-only diagnostic that splits land #71's measured per-depth conditional-
accept ladder q[d] (d=1..9) into the THREE TPS-attributable facets of the tree's
realized E[T], so the fleet sees EXACTLY which facet recovered on the re-bench and
whether the both-bugs-fixed cell lands at the FULL ceiling (537.8) or collapses to
the WIDTH-ONLY floor (376.3) -- the 161.6-TPS swing that is the one number the fleet
watches (fern #134's load-bearing sensitivity).

My #142 gate returns a correct SCALAR go/no-go from land's measured E[T]; it does NOT
decompose WHERE the ladder falls short. This is that decomposition. It does NOT
authorize a launch (BASELINE stays 481.53); it is a decision diagnostic only.

THE THREE FACETS (of realized E[T] = 1 bonus + Sum_d path-survival(d))
---------------------------------------------------------------------
Toggling each facet from its as-built/measured state to its rho-optimal state, on the
banked M=32 / depth-9 / max-branch-3 topology, via wirbel's exact E[T] DP:
  (a) depth-1 spine        q1: 0.674 (measured) -> 0.7287 (denken #133 fp32 target).
  (b) branch-width         the rank>=2 branch RESCUE at depths 2..9 (rho_cond
                           [0.4165,0.2655,0.1908]) -- the salvage the DESCENT fix
                           re-seeds. "width" == wirbel #135's width facet (C-A).
  (c) deep-spine-spread    the deeper rank-1 SPINE at depths 2..9 (measured DECLINING
                           q_meas[1:] -> rho-opt RISING q76[1:]) -- the "easy run"
                           the descending walk must keep. "spread" == wirbel #135's
                           spread facet (B-A). This facet IS the 161.6-TPS band.

NAMING (read carefully -- aligned to wirbel #135 + this PR's own hand-off semantics):
  WIDTH  = branch rescue (rank>=2 re-seeding). A width FAILURE = "descent not
           re-seeding rescued nodes" (salvages fire but stay terminal leaves).
  SPREAD = deep rank-1 spine (depths 2..9). A spread FAILURE = "deep-spine decaying
           faster than rho-opt" (the walk loses the easy run once it descends).
  The width-only COLLAPSE (376.3) = depth-1 fixed + width restored, spread LOST.

THE TWO ENDPOINTS (banked anchors -- the gate is valid iff it reproduces BOTH +-2%)
----------------------------------------------------------------------------------
  FULL ladder (a,b,c all rho-opt)         -> E[T]=5.207 -> 537.8  (fern #134 both-bugs)
  WIDTH-ONLY (a,b rho-opt, c=measured)    -> E[T]=3.643 -> 376.3  (fern #134 Fb collapse)
The band [376.3, 537.8] differs ONLY in facet (c): the deep-spine-spread is the binding
risk. The continuous map interpolates land's measured deep spine q[2:] -> realized TPS.

THE MAP (the #142 gate compose, reused verbatim -- one source of truth):
  official_TPS = K_cal * E[T] / step_time * tau   (K_cal=125.268; tau in [0.9983,1.0])
  step_time: lawine #136 MEASURED depth-9 step 1.2182 (merged) for the live band; the
  1.2127 roofline for the banked 537.8/376.3 anchors (where #134/#125 defined them).

LOCAL, CPU-ONLY, ANALYTIC. No GPU, no vLLM, no HF Job, no submission, no kernel build.
Reuses the banked #100 K_cal, the #142 gate compose, and wirbel's E[T] DP
(build_depth_pvecs_measured / score_tree_depthrank) verbatim. Serves nothing -> greedy
identity untouched by construction. Pairs with my #134 harness (the richer per-position
readout) and my #142 scalar gate (the go/no-go).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- banked models reused verbatim (one source of truth per constant) -------
lc = _load("lever_composition", os.path.join(_HERE, "lever_composition.py"))
gate = _load("m16_measured_500_gate", os.path.join(_HERE, "m16_measured_500_gate.py"))
tma = _load("treeshape_measured_accept", os.path.join(_HERE, "treeshape_measured_accept.py"))

K_CAL = lc.K_CAL                          # 125.268 (= 481.53 / 3.844; #100 compose)
E_T_LINEAR = lc.E_T_LINEAR               # 3.844  linear-MTP floor
E_T_TREE = lc.E_T_TREE                   # 5.207  rho-optimal supply ceiling
FRONTIER_OFFICIAL = lc.FRONTIER_OFFICIAL  # 481.53
TARGET_OFFICIAL = lc.TARGET_OFFICIAL     # 500.0
TARGET_530 = gate.TARGET_530             # 530.0

# the #142 gate compose, reused verbatim (the figure of merit + verdict logic).
official_tps_map = gate.official_tps_map
accept_length_for_official = gate.accept_length_for_official
_tps_verdict = gate._tps_verdict
TAU = gate.TAU                           # {low:0.9983, central:1.0, high:1.0}

# depth-9 decode step. lawine #136 MEASURED 1.2182 is now MERGED -> the operative
# live step (CURRENT_RESEARCH_STATE.md cycle-47, EXPERIMENTS_LOG.md). The 1.2127
# roofline (fern #125) is the step at which the 537.8/376.3 anchors were DEFINED
# (#134/#125), so the self-test reproduces them there.
STEP_MEASURED_DEPTH9 = 1.2182            # lawine #136 (merged): measured depth-9 step
STEP_ROOFLINE_DEPTH9 = gate.STEP_ROOFLINE_DEPTH9  # 1.2127483746822987 (#125 roofline)

# ---- banked watched-risk anchors (fern #134 sensitivity; roofline-step numbers) ----
ANCHOR_FULL_TPS = 537.8                  # both-bugs-fixed (full rho-opt ladder); fern #134
ANCHOR_WIDTHONLY_TPS = 376.3            # width restored, deep-spine-spread=0; fern #134 Fb
ANCHOR_TOL = 0.02                        # +-2% (the gate-validity tolerance)

# ---- depth-1 spine anchors (denken #133 lane) -------------------------------
DEPTH1_MEAS = 0.674                      # as-built C[0] (ladder-consistent depth-1)
DEPTH1_MEAS_ALT = 0.679                 # separately-cited depth-1 (robustness alt)
DEPTH1_FP32_TARGET = 0.728739760479042  # rho-optimal q1 (denken #133 recoverable target)

PPL_GATE = gate.PPL_GATE                 # 2.42 quality gate (carried for the live verdict)
RHO2_BRANCH_HIT = gate.RHO2_BRANCH_HIT   # 0.4165 measured rank-2 branch-hit (land's gate)


def _finite(x: float) -> float:
    """NaN/inf -> 0.0 (keep every emitted metric NaN-clean)."""
    return float(x) if (x is not None and math.isfinite(x)) else 0.0


# ----------------------------------------------------------------------------
# Banked tree model: the M=32/depth-9/max-branch-3 topology + rho-optimal ladders.
# ----------------------------------------------------------------------------
def load_banked(rho_path: str, oracle_path: str) -> dict:
    with open(rho_path) as f:
        rho = json.load(f)
    parent = rho["per_budget"]["32"]["optimal"]["parent"]
    rho_cond_opt = [float(x) for x in rho["inputs"]["rho_cond_measured"]]   # [0.4165,...]
    q76 = [float(x) for x in rho["inputs"]["depth_q_76"]]                    # rho-opt rising spine
    W = int(rho["config"]["W"])
    max_depth = int(rho["config"]["max_depth"])

    with open(oracle_path) as f:
        oracle = json.load(f)
    cum = [float(x) for x in oracle["per_position_cumulative_accept"]]
    q_meas = _cum_to_conditional(cum)                                       # as-built declining spine
    return {
        "parent": parent, "rho_cond_opt": rho_cond_opt, "q76": q76,
        "W": W, "max_depth": max_depth,
        "as_built_cumulative": cum, "q_meas": q_meas,
        "as_built_et": float(oracle.get("accept_length", 1.0 + sum(cum))),
        "depth1_meas": float(oracle.get("per_position_cumulative_accept", [DEPTH1_MEAS])[0]),
    }


def _cum_to_conditional(cum: list[float]) -> list[float]:
    """q[d] = C[d]/C[d-1] -- the conditional spine ladder behind a cumulative profile."""
    if not cum:
        return []
    q = [cum[0]]
    for i in range(1, len(cum)):
        prev = cum[i - 1]
        q.append(cum[i] / prev if prev > 1e-12 else 0.0)
    return q


def tree_et(q_spine: list[float], rho_cond: list[float], b: dict) -> float:
    """E[T] of the banked M=32 topology under spine ladder q_spine + branch rho_cond,
    via wirbel's exact DP (the same DP that produced F_tree=5.207)."""
    pv = tma.build_depth_pvecs_measured(q_spine, rho_cond, b["W"], b["max_depth"], "flat")
    F, _ = tma.score_tree_depthrank(b["parent"], pv)
    return _finite(F)


# ----------------------------------------------------------------------------
# The three-facet lattice. Each facet toggles measured/as-built <-> rho-optimal.
#   a = depth-1 spine     q1 in {DEPTH1_MEAS, DEPTH1_FP32_TARGET}
#   b = branch-width      rho_cond in {none, rho_cond_opt}     (rank>=2 rescue)
#   c = deep-spine-spread q[2:] in {q_meas[1:], q76[1:]}       (deeper rank-1 spine)
# Reproduces BOTH banked anchors at the b-on corner:
#   (1,1,0) = WIDTH-ONLY collapse -> 376.3 ; (1,1,1) = FULL -> 537.8.
# ----------------------------------------------------------------------------
def facet_spine(a: int, c: int, b_dict: dict) -> list[float]:
    """Assemble the rank-1 spine ladder for lattice point (a, *, c)."""
    d1 = DEPTH1_FP32_TARGET if a else b_dict["depth1_meas"]
    deeper = b_dict["q76"][1:] if c else b_dict["q_meas"][1:]
    return [d1] + list(deeper)


def lattice_et(a: int, b: int, c: int, b_dict: dict) -> float:
    rho_cond = b_dict["rho_cond_opt"] if b else [0.0] * len(b_dict["rho_cond_opt"])
    return tree_et(facet_spine(a, c, b_dict), rho_cond, b_dict)


def build_lattice(b_dict: dict) -> dict:
    L = {}
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                L[(a, b, c)] = lattice_et(a, b, c, b_dict)
    return L


# ----------------------------------------------------------------------------
# Decomposition: nested (a->b->c) marginals + Shapley-symmetric attribution.
# Both sum EXACTLY to the total recovery (000 -> 111). The nested order places the
# width-only anchor (1,1,0) on the b-on/c-off corner so the (c) marginal IS the band.
# ----------------------------------------------------------------------------
FACET_NAMES = {0: "depth1_spine", 1: "branch_width", 2: "deep_spine_spread"}


def nested_decomposition(L: dict, step: float, tau: float) -> dict:
    base = L[(0, 0, 0)]
    m_a = L[(1, 0, 0)] - L[(0, 0, 0)]            # +depth-1 (at b off, c meas)
    m_b = L[(1, 1, 0)] - L[(1, 0, 0)]            # +branch-width (at a fix, c meas) -> 376.3
    m_c = L[(1, 1, 1)] - L[(1, 1, 0)]            # +deep-spine-spread -> 537.8  (THE BAND)
    full = L[(1, 1, 1)]
    facets = {
        "depth1_spine": {"d_et": _finite(m_a), "d_tps": _finite(official_tps_map(m_a, step, tau))},
        "branch_width": {"d_et": _finite(m_b), "d_tps": _finite(official_tps_map(m_b, step, tau))},
        "deep_spine_spread": {"d_et": _finite(m_c), "d_tps": _finite(official_tps_map(m_c, step, tau))},
    }
    return {
        "order": ["depth1_spine", "branch_width", "deep_spine_spread"],
        "baseline_et": _finite(base),
        "full_et": _finite(full),
        "total_recovery_et": _finite(full - base),
        "facets": facets,
        "sum_check_et": _finite(m_a + m_b + m_c),
        "additive": bool(abs((m_a + m_b + m_c) - (full - base)) < 1e-9),
    }


def shapley_decomposition(L: dict, step: float, tau: float) -> dict:
    import itertools
    base = L[(0, 0, 0)]
    full = L[(1, 1, 1)]
    sh = {0: 0.0, 1: 0.0, 2: 0.0}
    perms = list(itertools.permutations([0, 1, 2]))
    for perm in perms:
        st = [0, 0, 0]
        for f in perm:
            before = L[tuple(st)]
            st[f] = 1
            sh[f] += (L[tuple(st)] - before) / len(perms)
    facets = {FACET_NAMES[f]: {"d_et": _finite(sh[f]),
                               "d_tps": _finite(official_tps_map(sh[f], step, tau))}
              for f in (0, 1, 2)}
    return {
        "baseline_et": _finite(base),
        "full_et": _finite(full),
        "total_recovery_et": _finite(full - base),
        "facets": facets,
        "sum_check_et": _finite(sum(sh.values())),
        "additive": bool(abs(sum(sh.values()) - (full - base)) < 1e-9),
    }


# ----------------------------------------------------------------------------
# Width-vs-spread band + the continuous spread-recovery map.
# ----------------------------------------------------------------------------
def width_spread_band(b_dict: dict, step: float, tau: float) -> dict:
    full_et = b_dict["_L"][(1, 1, 1)]
    widthonly_et = b_dict["_L"][(1, 1, 0)]
    full_tps = official_tps_map(full_et, step, tau)
    widthonly_tps = official_tps_map(widthonly_et, step, tau)
    return {
        "full_endpoint_et": _finite(full_et),
        "widthonly_endpoint_et": _finite(widthonly_et),
        "full_endpoint_tps": _finite(full_tps),
        "widthonly_endpoint_tps": _finite(widthonly_tps),
        "band_tps": [_finite(widthonly_tps), _finite(full_tps)],
        "swing_tps": _finite(full_tps - widthonly_tps),
        "binding_facet": "deep_spine_spread",
        "step_time": step,
    }


def spread_recovery_map(b_dict: dict, step: float, tau: float, n: int = 11) -> dict:
    """Continuous map: deep-spine-spread recovery fraction lambda in [0,1] (depth-1 and
    branch-width held at rho-opt) -> realized E[T] and official TPS, from the width-only
    floor (lambda=0, 376.3) to the full ceiling (lambda=1, 537.8). lambda=0 keeps the
    deeper spine at measured; lambda=1 lifts it to the rho-optimal rising ladder."""
    q_meas, q76 = b_dict["q_meas"], b_dict["q76"]
    rows = []
    cross_500 = None
    for i in range(n):
        lam = i / (n - 1)
        deeper = [q_meas[1 + j] + lam * (q76[1 + j] - q_meas[1 + j])
                  for j in range(len(q76) - 1)]
        q_spine = [DEPTH1_FP32_TARGET] + deeper
        et = tree_et(q_spine, b_dict["rho_cond_opt"], b_dict)
        tps = official_tps_map(et, step, tau)
        rows.append({"lambda_spread_recovery": round(lam, 4),
                     "E_T": _finite(et), "official_tps": _finite(tps),
                     "clears_500": bool(tps >= TARGET_OFFICIAL)})
        if cross_500 is None and tps >= TARGET_OFFICIAL:
            cross_500 = lam
    return {
        "axis": "deep_spine_spread_recovery_fraction",
        "step_time": step,
        "rows": rows,
        "spread_recovery_to_clear_500": cross_500,
        "note": ("lambda = how far land's re-benched deeper spine q[2:] recovers from the "
                 "measured declining ladder toward the rho-optimal rising ladder, with "
                 "depth-1 and branch-width at rho-opt. lambda below "
                 + (f"{cross_500:.2f}" if cross_500 is not None else "n/a")
                 + " => sub-500 even with width fully restored => a SPREAD failure."),
    }


def width_recovery_map(b_dict: dict, step: float, tau: float, n: int = 11) -> dict:
    """Cross-check map: branch-width (rank>=2 rescue) recovery fraction mu in [0,1], with
    depth-1 and deep-spine-spread held at rho-opt -> TPS. mu=0 zeroes the branch rescue
    (pure rho-opt spine, 431.5); mu=1 is the full rho-opt branch ladder (537.8)."""
    rho_opt = b_dict["rho_cond_opt"]
    rows = []
    cross_500 = None
    for i in range(n):
        mu = i / (n - 1)
        rho_cond = [mu * r for r in rho_opt]
        et = tree_et(b_dict["q76"], rho_cond, b_dict)
        tps = official_tps_map(et, step, tau)
        rows.append({"mu_width_recovery": round(mu, 4), "E_T": _finite(et),
                     "official_tps": _finite(tps), "clears_500": bool(tps >= TARGET_OFFICIAL)})
        if cross_500 is None and tps >= TARGET_OFFICIAL:
            cross_500 = mu
    return {"axis": "branch_width_recovery_fraction", "step_time": step, "rows": rows,
            "width_recovery_to_clear_500": cross_500}


# ============================================================================
# JOINT (spread x width) CLEARS-500 FRONTIER (PR #149)
# ----------------------------------------------------------------------------
# Upgrades the two 1-D recovery slices (#145 spread_recovery_map / width_recovery_
# map) into the full 2-D decision surface over (lambda, mu) in [0,1]^2:
#   lambda = deep-spine-SPREAD recovery fraction (q_meas[1:] -> q76[1:]),
#   mu     = branch-WIDTH recovery fraction      (rho_cond 0 -> rho_cond_opt),
# with depth-1 held at rho-opt (q76[0] == DEPTH1_FP32_TARGET, verified). This is
# the honest go/no-go for land #71's likely PARTIAL-BOTH-FACET landing -- the
# (lambda, mu) INTERIOR where NEITHER 1-D slice applies (each slice silently
# assumes the OTHER facet is fully rho-opt). By construction the edges reproduce
# the #145 slices exactly: the mu=1 column IS spread_recovery_map (lambda-intercept
# 0.90) and the lambda=1 row IS width_recovery_map (mu-intercept 0.70). TPS is
# monotone-nondecreasing in both axes (more recovery -> higher E[T]), so every
# iso-TPS contour is single-valued lambda_iso(mu).
# ============================================================================
def joint_spine(lam: float, b_dict: dict) -> list[float]:
    """Rank-1 spine ladder at spread-recovery lambda, depth-1 held at rho-opt.
    lambda=0 keeps the deeper spine at land's measured DECLINING ladder; lambda=1
    lifts it to the rho-optimal RISING ladder. Same construction as #145
    spread_recovery_map so the mu=1 column reproduces that slice verbatim."""
    q_meas, q76 = b_dict["q_meas"], b_dict["q76"]
    deeper = [q_meas[1 + j] + lam * (q76[1 + j] - q_meas[1 + j]) for j in range(len(q76) - 1)]
    return [q76[0]] + deeper


def joint_rho(mu: float, b_dict: dict) -> list[float]:
    """Branch rescue ladder at width-recovery mu. mu=0 zeroes the rank>=2 rescue;
    mu=1 is the full rho-optimal branch ladder. Same as #145 width_recovery_map."""
    return [mu * r for r in b_dict["rho_cond_opt"]]


def joint_et(lam: float, mu: float, b_dict: dict) -> float:
    """E[T] of the banked M=32 topology at joint recovery (lambda, mu)."""
    return tree_et(joint_spine(lam, b_dict), joint_rho(mu, b_dict), b_dict)


def joint_tps(lam: float, mu: float, b_dict: dict, step: float, tau: float) -> float:
    """official TPS at (lambda, mu): the #142 compose on the joint E[T]. NaN-clean."""
    return _finite(official_tps_map(joint_et(lam, mu, b_dict), step, tau))


def joint_frontier_surface(b_dict: dict, step: float, tau: float, n: int = 101) -> dict:
    """Sweep (lambda, mu) on an n x n grid -> the official-TPS surface (NaN-clean).
    Precomputes per-lambda spines and per-mu rho ladders so each cell is one E[T] DP.
    Composes through the #142 gate map (== measured_m16_to_official's core)."""
    axis = [i / (n - 1) for i in range(n)]
    spines = [joint_spine(lam, b_dict) for lam in axis]          # depends on lambda only
    rhos = [joint_rho(mu, b_dict) for mu in axis]                # depends on mu only
    et_grid, tps_grid = [], []
    nan_cells = 0
    for j in range(n):                                            # rows = mu
        et_row, tps_row = [], []
        rho = rhos[j]
        for i in range(n):                                        # cols = lambda
            et = tree_et(spines[i], rho, b_dict)
            tps = official_tps_map(et, step, tau)
            if not (math.isfinite(et) and math.isfinite(tps)):
                nan_cells += 1
            et_row.append(_finite(et))
            tps_row.append(_finite(tps))
        et_grid.append(et_row)
        tps_grid.append(tps_row)
    return {
        "lambda_axis": axis, "mu_axis": axis, "n": n,
        "et_grid": et_grid, "tps_grid": tps_grid,
        "nan_cells": nan_cells, "nan_clean": bool(nan_cells == 0),
        "step_time": step, "tau": tau,
        "tps_min": _finite(min(min(r) for r in tps_grid)),
        "tps_max": _finite(max(max(r) for r in tps_grid)),
    }


def _interp_cross(axis: list[float], vals: list[float], level: float) -> float | None:
    """First x where a monotone-nondecreasing `vals` crosses up through `level`,
    linearly interpolated. None if no crossing inside [axis[0], axis[-1]]."""
    if vals[0] >= level:
        return axis[0]
    for i in range(1, len(vals)):
        if vals[i] >= level:
            y0, y1 = vals[i - 1], vals[i]
            if y1 == y0:
                return axis[i]
            return axis[i - 1] + (axis[i] - axis[i - 1]) * (level - y0) / (y1 - y0)
    return None


def extract_iso_contour(surface: dict, level: float) -> dict:
    """The iso-`level` contour as lambda_iso(mu) (scan each mu row over lambda) and
    its inverse mu_iso(lambda) (scan each lambda column over mu). Single-valued
    because TPS is monotone in both axes."""
    lam_axis, mu_axis, grid = surface["lambda_axis"], surface["mu_axis"], surface["tps_grid"]
    lam_of_mu, mu_of_lam = [], []
    for j, mu in enumerate(mu_axis):
        c = _interp_cross(lam_axis, grid[j], level)
        lam_of_mu.append({"mu": round(mu, 4), "lambda_iso": (None if c is None else round(c, 4))})
    for i, lam in enumerate(lam_axis):
        col = [grid[j][i] for j in range(len(mu_axis))]
        c = _interp_cross(mu_axis, col, level)
        mu_of_lam.append({"lambda": round(lam, 4), "mu_iso": (None if c is None else round(c, 4))})
    # contour length / coverage: how many mu rows actually have a crossing.
    covered = sum(1 for r in lam_of_mu if r["lambda_iso"] is not None)
    return {
        "level_tps": level,
        "lambda_iso_of_mu": lam_of_mu,
        "mu_iso_of_lambda": mu_of_lam,
        "lambda_intercept_at_mu1": lam_of_mu[-1]["lambda_iso"],   # continuous lambda* at mu=1
        "mu_intercept_at_lambda1": mu_of_lam[-1]["mu_iso"],       # continuous mu* at lambda=1
        "rows_with_crossing": covered,
        "rows_total": len(mu_axis),
    }


def green_area_fraction(surface: dict, level: float = TARGET_OFFICIAL) -> float:
    """Fraction of the (lambda, mu) grid with official TPS >= `level` (midpoint
    rule over the n x n grid). NaN cells count as not-green."""
    grid = surface["tps_grid"]
    tot = sum(len(r) for r in grid)
    green = sum(1 for r in grid for v in r if math.isfinite(v) and v >= level)
    return _finite(green / tot) if tot else 0.0


def coarse_axis_intercept(b_dict: dict, step: float, tau: float, axis_kind: str, n: int = 11) -> dict:
    """#145-faithful coarse 0.1-grid FIRST-CLEARING fraction (the value #145 reported):
    'spread' varies lambda at mu=1 (reproduces spread_recovery_map -> 0.90);
    'width'  varies mu at lambda=1 (reproduces width_recovery_map -> 0.70)."""
    cross, rows = None, []
    for i in range(n):
        f = i / (n - 1)
        tps = joint_tps(f, 1.0, b_dict, step, tau) if axis_kind == "spread" \
            else joint_tps(1.0, f, b_dict, step, tau)
        clears = bool(tps >= TARGET_OFFICIAL)
        rows.append({"frac": round(f, 4), "official_tps": _finite(tps), "clears_500": clears})
        if cross is None and clears:
            cross = round(f, 4)
    return {"first_clearing_frac": cross, "rows": rows}


def joint_self_test(b_dict: dict, surface_measured: dict) -> dict:
    """PRIMARY validity: the 2-D map must reproduce the #145 anchors.
      (1) corner (lambda=1, mu=1) -> 537.8 within +-2% at the 1.2127 roofline,
      (2) corner (lambda=0, mu=1) -> 376.3 within +-2% at the roofline,
      (3) lambda-intercept at mu=1 reproduces 0.90 (coarse 0.1 grid, measured step),
      (4) mu-intercept at lambda=1 reproduces 0.70 (coarse 0.1 grid, measured step),
      (5) the edge VALUE profiles match #145's reported slice TPS within +-1%,
      (6) the surface is NaN-clean."""
    step_m, tau_m = surface_measured["step_time"], surface_measured["tau"]

    def _rel(v, ref):
        return abs(v - ref) / ref if ref else float("inf")

    # (1)-(2) corners priced at the roofline step (where 537.8/376.3 were defined).
    full_roof = official_tps_map(joint_et(1.0, 1.0, b_dict), STEP_ROOFLINE_DEPTH9, TAU["central"])
    width_roof = official_tps_map(joint_et(0.0, 1.0, b_dict), STEP_ROOFLINE_DEPTH9, TAU["central"])
    full_rel, width_rel = _rel(full_roof, ANCHOR_FULL_TPS), _rel(width_roof, ANCHOR_WIDTHONLY_TPS)
    full_ok, width_ok = bool(full_rel <= ANCHOR_TOL), bool(width_rel <= ANCHOR_TOL)

    # (3)-(4) axis intercepts on the #145 coarse 0.1 grid at the MEASURED step.
    sp = coarse_axis_intercept(b_dict, step_m, tau_m, "spread")
    wd = coarse_axis_intercept(b_dict, step_m, tau_m, "width")
    lam_int, mu_int = sp["first_clearing_frac"], wd["first_clearing_frac"]
    INTERCEPT_TOL = 0.01
    lam_int_ok = bool(lam_int is not None and abs(lam_int - 0.90) <= INTERCEPT_TOL)
    mu_int_ok = bool(mu_int is not None and abs(mu_int - 0.70) <= INTERCEPT_TOL)

    # (5) strong edge-profile reproduction: #145 slice VALUES within +-1%.
    edge_specs = [(0.8, 1.0, 492.27, "spread@lambda0.8"), (0.9, 1.0, 512.94, "spread@lambda0.9"),
                  (1.0, 0.6, 494.86, "width@mu0.6"), (1.0, 0.7, 505.23, "width@mu0.7")]
    edge_checks = []
    for lam, mu, ref145, tag in edge_specs:
        v = joint_tps(lam, mu, b_dict, step_m, tau_m)
        rel = _rel(v, ref145)
        edge_checks.append({"point": tag, "tps": _finite(v), "ref_145": ref145,
                            "rel_err": _finite(rel), "within_1pct": bool(rel <= 0.01)})
    edges_ok = bool(all(e["within_1pct"] for e in edge_checks))

    nan_clean = bool(surface_measured["nan_clean"])
    passes = bool(full_ok and width_ok and lam_int_ok and mu_int_ok and edges_ok and nan_clean)
    return {
        "passes": passes,
        "corner_full": {"lambda": 1.0, "mu": 1.0, "reconstructed_tps_roofline": _finite(full_roof),
                        "expected_tps": ANCHOR_FULL_TPS, "rel_err": _finite(full_rel),
                        "within_2pct": full_ok, "et": _finite(joint_et(1.0, 1.0, b_dict))},
        "corner_widthonly": {"lambda": 0.0, "mu": 1.0, "reconstructed_tps_roofline": _finite(width_roof),
                             "expected_tps": ANCHOR_WIDTHONLY_TPS, "rel_err": _finite(width_rel),
                             "within_2pct": width_ok, "et": _finite(joint_et(0.0, 1.0, b_dict))},
        "lambda_intercept_at_mu1": {"coarse_first_clearing": lam_int, "expected": 0.90,
                                    "within_1pct": lam_int_ok},
        "mu_intercept_at_lambda1": {"coarse_first_clearing": mu_int, "expected": 0.70,
                                    "within_1pct": mu_int_ok},
        "edge_profile_checks": edge_checks, "edges_within_1pct": edges_ok,
        "nan_clean": nan_clean, "tolerance_corner": ANCHOR_TOL, "tolerance_intercept": INTERCEPT_TOL,
        "step_roofline": STEP_ROOFLINE_DEPTH9, "step_measured": step_m,
        "note": ("corners priced at the 1.2127 roofline (where 537.8/376.3 were defined, fern "
                 "#134/#125); intercepts + edge profiles at the measured 1.2182 (where #145 "
                 "reported 0.90/0.70). The mu=1 column IS spread_recovery_map and the lambda=1 "
                 "row IS width_recovery_map by construction, so the slices reproduce exactly."),
    }


def derive_lambda_mu_from_ladder(q_land, branch_hit, b_dict: dict) -> dict:
    """Project land's measured per-depth conditional ladder q[1..9] + branch-hit rho2
    onto the surface coordinates (lambda, mu).
      mu     = branch_hit / rho-opt rank-2 hit (clamped [0,1]) -- width recovery.
      lambda = mean over depths >=2 of (q_land[d]-q_meas[d])/(q76[d]-q_meas[d]),
               clamped [0,1] -- deep-spine-spread recovery.
    Depth-1 is the SEPARATE denken #133 facet (the surface holds it at rho-opt); a
    short land depth-1 is flagged but does NOT move (lambda, mu)."""
    q_meas, q76 = b_dict["q_meas"], b_dict["q76"]
    mu = None
    if branch_hit is not None and RHO2_BRANCH_HIT > 1e-9:
        mu = max(0.0, min(1.0, branch_hit / RHO2_BRANCH_HIT))
    fracs = []
    if q_land and len(q_land) > 1:
        ndeep = min(len(q_land) - 1, len(q76) - 1)
        for j in range(ndeep):
            meas, opt, val = q_meas[1 + j], q76[1 + j], q_land[1 + j]
            if abs(opt - meas) > 1e-9:
                fracs.append((val - meas) / (opt - meas))
    lam = max(0.0, min(1.0, sum(fracs) / len(fracs))) if fracs else None
    depth1_land = (q_land[0] if q_land else None)
    return {
        "lambda": lam, "mu": mu,
        "per_depth_lambda_fracs": [_finite(f) for f in fracs],
        "depth1_land": depth1_land, "depth1_rho_opt": q76[0],
        "depth1_short_of_rho_opt": (bool(depth1_land < q76[0] - 1e-6)
                                    if depth1_land is not None else None),
        "branch_hit": branch_hit, "rho2_rho_opt": RHO2_BRANCH_HIT,
        "note": ("lambda from the deeper-spine recovery fraction (depths >=2); mu from the "
                 "rank-2 branch-hit vs rho-opt. Depth-1 is the separate denken #133 facet, "
                 "held at rho-opt on this surface."),
    }


def realized_point_reader(b_dict: dict, step: float, tau: float, lam: float, mu: float,
                          surface: dict, derived: dict | None = None) -> dict:
    """Locate (lambda, mu) on the surface -> GO / NO-GO / MARGINAL vs iso-500, name
    the BINDING facet (which axis is limiting), and the CHEAPEST direction to GREEN
    (the surface gradient, in TPS-per-unit-recovery)."""
    n = surface["n"]
    h = 1.0 / (n - 1)

    def T(l, m):
        return joint_tps(l, m, b_dict, step, tau)

    def T_low(l, m):
        return _finite(official_tps_map(joint_et(l, m, b_dict), step, TAU["low"]))

    tps_c, tps_lo = T(lam, mu), T_low(lam, mu)
    verdict_tps = _tps_verdict(tps_c, tps_lo, TARGET_OFFICIAL)
    go_no_go = {"GREEN": "GO", "AMBER": "MARGINAL", "RED": "NO-GO"}[verdict_tps]

    # surface gradients (TPS per unit recovery), central diff clamped to [0,1].
    def grad(axis):
        if axis == "lambda":
            lo, hi = max(0.0, lam - h), min(1.0, lam + h)
            return _finite((T(hi, mu) - T(lo, mu)) / (hi - lo)) if hi > lo else 0.0
        lo, hi = max(0.0, mu - h), min(1.0, mu + h)
        return _finite((T(lam, hi) - T(lam, lo)) / (hi - lo)) if hi > lo else 0.0

    g_lam, g_mu = grad("lambda"), grad("mu")

    # TPS still on the table from each facet's shortfall (push to 1.0, hold the other).
    left_lambda = _finite(T(1.0, mu) - tps_c)
    left_mu = _finite(T(lam, 1.0) - tps_c)
    binding = "deep_spine_spread" if left_lambda >= left_mu else "branch_width"

    # cheapest single-facet path to 500: smallest feasible recovery delta (invert the
    # 1-D slice through the point); None if that facet alone cannot reach 500 by 1.0.
    def delta_to_500(axis, steps=400):
        cur = lam if axis == "lambda" else mu
        for k in range(steps + 1):
            f = cur + (1.0 - cur) * k / steps
            t = T(f, mu) if axis == "lambda" else T(lam, f)
            if t >= TARGET_OFFICIAL:
                return _finite(f - cur), round(f, 4)
        return None, None

    d_lam, lam_star = delta_to_500("lambda")
    d_mu, mu_star = delta_to_500("mu")
    feas = [("deep_spine_spread", d_lam, g_lam), ("branch_width", d_mu, g_mu)]
    feas_ok = [f for f in feas if f[1] is not None]
    if tps_c >= TARGET_OFFICIAL:
        cheapest = None
    elif feas_ok:
        cheapest = min(feas_ok, key=lambda f: f[1])[0]
    else:                                       # neither facet alone reaches 500 -> steepest
        cheapest = "deep_spine_spread" if g_lam >= g_mu else "branch_width"

    return {
        "lambda": _finite(lam), "mu": _finite(mu),
        "official_tps_central": tps_c, "official_tps_taulow": tps_lo,
        "tps_verdict": verdict_tps, "go_no_go": go_no_go,
        "margin_to_500": _finite(tps_c - TARGET_OFFICIAL),
        "clears_500_central": bool(tps_c >= TARGET_OFFICIAL),
        "clears_500_conservative": bool(tps_lo >= TARGET_OFFICIAL),
        "gradient_tps_per_unit_recovery": {"d_tps_d_lambda_spread": g_lam, "d_tps_d_mu_width": g_mu},
        "tps_left_on_table": {"deep_spine_spread": left_lambda, "branch_width": left_mu},
        "binding_facet": binding,
        "cheapest_direction_to_green": cheapest,
        "recovery_delta_to_500": {
            "deep_spine_spread": {"delta_lambda": d_lam, "lambda_star": lam_star,
                                  "feasible_by_1.0": bool(d_lam is not None)},
            "branch_width": {"delta_mu": d_mu, "mu_star": mu_star,
                             "feasible_by_1.0": bool(d_mu is not None)},
        },
        "depth1_caveat": (derived.get("depth1_short_of_rho_opt") if derived else None),
        "step_time": step, "tau": tau,
        "reader_note": ("verdict vs iso-500 with the tau band: GO = clears 500 robustly "
                        "(central and tau-low corner >= 500); MARGINAL = central clears but "
                        "the conservative corner does not (knife-edge); NO-GO = below 500. "
                        "BINDING = the facet leaving the most TPS on the table; CHEAPEST = "
                        "the facet that reaches 500 with the smallest recovery (or steepest "
                        "gradient if neither alone suffices)."),
    }


# ----------------------------------------------------------------------------
# Self-test: the gate is valid iff it reproduces BOTH banked anchors within +-2%.
# Anchors are roofline-step numbers (fern #134/#125), so the self-test prices them
# at the 1.2127 roofline; the live band re-prices at lawine #136's measured step.
# ----------------------------------------------------------------------------
def self_test(b_dict: dict) -> dict:
    L = b_dict["_L"]
    full_tps = official_tps_map(L[(1, 1, 1)], STEP_ROOFLINE_DEPTH9, TAU["central"])
    widthonly_tps = official_tps_map(L[(1, 1, 0)], STEP_ROOFLINE_DEPTH9, TAU["central"])

    def _rel(v, ref):
        return abs(v - ref) / ref if ref else float("inf")

    full_rel = _rel(full_tps, ANCHOR_FULL_TPS)
    width_rel = _rel(widthonly_tps, ANCHOR_WIDTHONLY_TPS)
    full_ok = bool(full_rel <= ANCHOR_TOL)
    width_ok = bool(width_rel <= ANCHOR_TOL)

    # additivity: both decompositions must sum to the total recovery, NaN-clean.
    nd = nested_decomposition(L, STEP_ROOFLINE_DEPTH9, TAU["central"])
    sd = shapley_decomposition(L, STEP_ROOFLINE_DEPTH9, TAU["central"])
    additive_ok = bool(nd["additive"] and sd["additive"])
    nan_clean = all(math.isfinite(v) for v in
                    [full_tps, widthonly_tps, nd["sum_check_et"], sd["sum_check_et"]])

    passes = bool(full_ok and width_ok and additive_ok and nan_clean)
    return {
        "passes": passes,
        "anchor_full": {"reconstructed_tps": _finite(full_tps), "expected_tps": ANCHOR_FULL_TPS,
                        "rel_err": _finite(full_rel), "within_2pct": full_ok,
                        "et": _finite(L[(1, 1, 1)])},
        "anchor_widthonly": {"reconstructed_tps": _finite(widthonly_tps),
                             "expected_tps": ANCHOR_WIDTHONLY_TPS, "rel_err": _finite(width_rel),
                             "within_2pct": width_ok, "et": _finite(L[(1, 1, 0)])},
        "additive_ok": additive_ok,
        "nan_clean": bool(nan_clean),
        "tolerance": ANCHOR_TOL,
        "step_time_anchor": STEP_ROOFLINE_DEPTH9,
        "note": ("anchors 537.8/376.3 are roofline-step numbers (fern #134/#125); the "
                 "self-test reproduces them at the 1.2127 roofline. The live band/map "
                 "re-price at lawine #136's measured 1.2182 (+0.45%, within +-2%)."),
    }


# ----------------------------------------------------------------------------
# Live gate: land #71's measured per-depth ladder -> realized TPS + binding-facet flag.
# ----------------------------------------------------------------------------
def evaluate_live_ladder(q_land: list[float], b_dict: dict, step: float,
                         rho_cond_land: list[float] | None = None,
                         branch_hit: float | None = None,
                         ppl: float | None = None,
                         greedy_token_ids_captured: bool | None = None) -> dict:
    """Decompose land's MEASURED per-depth conditional ladder q[1..9] into the three
    facets, return realized official TPS, the verdict vs 500, and the binding shortfall
    facet (width failure vs spread failure) when the verdict is < GREEN."""
    q76, q_meas = b_dict["q76"], b_dict["q_meas"]
    rho_opt = b_dict["rho_cond_opt"]
    # land's branch rescue ladder: explicit rho_cond, else scale rho-opt by the measured
    # branch-hit/rho2 (rho2 is the rank-2 hit; below rho-opt => weaker rescue), else rho-opt.
    if rho_cond_land is not None:
        rho_land = list(rho_cond_land)
    elif branch_hit is not None and RHO2_BRANCH_HIT > 1e-9:
        scale = max(0.0, branch_hit / RHO2_BRANCH_HIT)
        rho_land = [scale * r for r in rho_opt]
    else:
        rho_land = list(rho_opt)

    realized_et = tree_et(q_land, rho_land, b_dict)
    realized_central = official_tps_map(realized_et, step, TAU["central"])
    realized_taulow = official_tps_map(realized_et, step, TAU["low"])
    verdict = _tps_verdict(realized_central, realized_taulow, TARGET_OFFICIAL)

    # per-facet "TPS left on the table": recover each facet to rho-opt, hold the others
    # at land's measured values -> the marginal TPS that facet's shortfall is costing.
    d1_land = q_land[0] if q_land else b_dict["depth1_meas"]
    deeper_land = q_land[1:] if len(q_land) > 1 else q_meas[1:]

    et_fix_depth1 = tree_et([DEPTH1_FP32_TARGET] + list(deeper_land), rho_land, b_dict)
    et_fix_width = tree_et(list(q_land), rho_opt, b_dict)
    et_fix_spread = tree_et([d1_land] + list(q76[1:]), rho_land, b_dict)

    left = {
        "depth1_spine": _finite(official_tps_map(et_fix_depth1 - realized_et, step, TAU["central"])),
        "branch_width": _finite(official_tps_map(et_fix_width - realized_et, step, TAU["central"])),
        "deep_spine_spread": _finite(official_tps_map(et_fix_spread - realized_et, step, TAU["central"])),
    }
    binding = max(left, key=left.get)
    # facet recovery fractions (0 = measured, 1 = rho-opt) for the hand-off readout.
    def _frac(meas, opt, val):
        return _finite((val - meas) / (opt - meas)) if abs(opt - meas) > 1e-9 else 1.0
    spread_meas_et = tree_et([d1_land] + list(q_meas[1:]), rho_land, b_dict)
    recovery = {
        "depth1_spine_frac": _frac(b_dict["depth1_meas"], DEPTH1_FP32_TARGET, d1_land),
        "deep_spine_spread_frac": _frac(spread_meas_et, et_fix_spread, realized_et),
        "branch_width_rho2": branch_hit,
    }

    failure_flag = None
    if verdict != "GREEN":
        if binding == "deep_spine_spread":
            failure_flag = ("SPREAD FAILURE: deep-spine decaying faster than rho-opt "
                            "(q[2:] short of the rising ladder); the descending walk is "
                            "losing the easy run. Fix the deep-spine descent, not the branches.")
        elif binding == "branch_width":
            failure_flag = ("WIDTH FAILURE: descent not re-seeding rescued nodes (branch-hit / "
                            "rho2 below rho-opt); salvages fire but stay terminal leaves. "
                            "Fix the rank>=2 re-seed, not the deep spine.")
        else:
            failure_flag = ("DEPTH-1 SPINE shortfall (q1 plumbing, denken #133): a separate "
                            "fix from the descent walk; only the 522->538 margin.")

    # validity preconditions (carried, mirroring the #142 gate)
    above_floor = bool(realized_et > E_T_LINEAR)
    ppl_within = (bool(ppl <= PPL_GATE) if ppl is not None else None)
    greedy_ok = (bool(greedy_token_ids_captured) if greedy_token_ids_captured is not None else None)
    preconds_pass = bool(above_floor and (ppl_within in (None, True)) and (greedy_ok in (None, True)))

    go_no_go = "GO" if (verdict == "GREEN" and above_floor and ppl_within and greedy_ok) else "NO-GO"
    return {
        "measured_ladder": [round(x, 4) for x in q_land],
        "branch_rescue_ladder": [round(x, 4) for x in rho_land],
        "realized_et": _finite(realized_et),
        "realized_official_tps_central": _finite(realized_central),
        "realized_official_tps_taulow": _finite(realized_taulow),
        "verdict": verdict,
        "clears_500_central": bool(realized_central >= TARGET_OFFICIAL),
        "clears_500_conservative": bool(realized_taulow >= TARGET_OFFICIAL),
        "margin_to_500": _finite(realized_central - TARGET_OFFICIAL),
        "tps_left_on_table_by_facet": left,
        "binding_shortfall_facet": binding,
        "facet_recovery": recovery,
        "failure_flag": failure_flag,
        "where_on_band": ("FULL (537.8)" if realized_central >= ANCHOR_FULL_TPS - 5 else
                          "WIDTH-ONLY floor (376.3)" if realized_central <= ANCHOR_WIDTHONLY_TPS + 5 else
                          "between width-only and full"),
        "preconditions": {"tok_per_step_above_linear_floor": above_floor,
                          "ppl_within_gate": ppl_within, "greedy_token_ids_captured": greedy_ok,
                          "all_pass": preconds_pass},
        "go_no_go": go_no_go,
        "step_time": step,
    }


def _load_live_ladder(args) -> dict | None:
    if args.measured_json and os.path.exists(args.measured_json):
        with open(args.measured_json) as f:
            m = json.load(f)
        q = m.get("q_ladder") or m.get("per_position_conditional_accept")
        if q is None and m.get("per_position_cumulative_accept"):
            q = _cum_to_conditional([float(x) for x in m["per_position_cumulative_accept"]])
        return {
            "q_ladder": [float(x) for x in q] if q else None,
            "rho_cond_land": m.get("rho_cond") or m.get("rho_cond_land"),
            "branch_hit": m.get("per_position_branch_hit", m.get("branch_hit")),
            "ppl": m.get("ppl"),
            "greedy_token_ids_captured": m.get("greedy_token_ids_captured"),
        }
    if args.measured_q_ladder:
        return {"q_ladder": [float(x) for x in args.measured_q_ladder],
                "rho_cond_land": None, "branch_hit": args.measured_branch_hit,
                "ppl": args.measured_ppl, "greedy_token_ids_captured": args.measured_greedy_captured}
    return None


def _nearest_idx(axis: list[float], val: float) -> int:
    return min(range(len(axis)), key=lambda i: abs(axis[i] - val))


def run_joint_frontier(args, b_dict: dict, step: float, step_is_measured: bool, tau: float) -> dict:
    """PR #149 --joint-frontier: the full 2-D (spread x width) clears-500 decision
    surface, its iso-contours + geometry, GREEN-region area, a self-test against the
    #145 anchors, and a realized-point reader for land #71's measured (lambda, mu)."""
    n = max(101, int(args.joint_grid_n))                          # PR floor: >= 101 x 101

    # ---- the surface (measured step = the operative live band) ----
    surface = joint_frontier_surface(b_dict, step, tau, n)

    # ---- PRIMARY self-test (reproduce the #145 anchors + slices) ----
    st = joint_self_test(b_dict, surface)
    joint_frontier_self_test_passes = int(st["passes"])

    # ---- contours: iso-500 (the bar), iso-481.53 (frontier parity), iso-530 (context) ----
    iso_500 = extract_iso_contour(surface, TARGET_OFFICIAL)
    iso_481 = extract_iso_contour(surface, FRONTIER_OFFICIAL)
    iso_530 = extract_iso_contour(surface, TARGET_530)

    # ---- TEST: GREEN-region (clears-500) area fraction ----
    green_region_area_fraction = green_area_fraction(surface, TARGET_OFFICIAL)
    # roofline-step area for context (the anchors' step).
    surface_roof = joint_frontier_surface(b_dict, STEP_ROOFLINE_DEPTH9, tau, n)
    green_area_roofline = green_area_fraction(surface_roof, TARGET_OFFICIAL)

    # ---- corner geometry: the 4 corners + the interior trade-curve ----
    lam_axis, mu_axis = surface["lambda_axis"], surface["mu_axis"]
    corners = {
        "lambda0_mu0": joint_tps(0.0, 0.0, b_dict, step, tau),
        "lambda1_mu0": joint_tps(1.0, 0.0, b_dict, step, tau),
        "lambda0_mu1": joint_tps(0.0, 1.0, b_dict, step, tau),
        "lambda1_mu1": joint_tps(1.0, 1.0, b_dict, step, tau),
    }
    # coarse (#145-faithful) + continuous axis intercepts.
    coarse_spread = coarse_axis_intercept(b_dict, step, tau, "spread")
    coarse_width = coarse_axis_intercept(b_dict, step, tau, "width")
    # trade-curve: how much width-recovery (mu) buys back a spread (lambda) shortfall
    # along the iso-500 contour, and the reverse. Sample at a few mu / lambda.
    trade_lambda_of_mu = [iso_500["lambda_iso_of_mu"][_nearest_idx(mu_axis, m)]
                          for m in (0.6, 0.7, 0.8, 0.9, 1.0)]
    trade_mu_of_lambda = [iso_500["mu_iso_of_lambda"][_nearest_idx(lam_axis, l)]
                          for l in (0.85, 0.9, 0.95, 1.0)]
    # local exchange slope d(lambda)/d(mu) on the iso-500 contour (extra mu per unit
    # lambda bought back): finite diff over the contour's covered mu rows.
    contour_pts = [(r["mu"], r["lambda_iso"]) for r in iso_500["lambda_iso_of_mu"]
                   if r["lambda_iso"] is not None]
    exchange_slope = None
    if len(contour_pts) >= 2:
        (m0, l0), (m1, l1) = contour_pts[0], contour_pts[-1]
        exchange_slope = _finite((l1 - l0) / (m1 - m0)) if (m1 - m0) != 0 else None

    corner_geometry = {
        "corners_tps_measured_step": {k: _finite(v) for k, v in corners.items()},
        "lambda_intercept_at_mu1": {
            "coarse_first_clearing_145": coarse_spread["first_clearing_frac"],
            "continuous": iso_500["lambda_intercept_at_mu1"], "expected_145": 0.90},
        "mu_intercept_at_lambda1": {
            "coarse_first_clearing_145": coarse_width["first_clearing_frac"],
            "continuous": iso_500["mu_intercept_at_lambda1"], "expected_145": 0.70},
        "trade_curve_iso500_lambda_of_mu": trade_lambda_of_mu,
        "trade_curve_iso500_mu_of_lambda": trade_mu_of_lambda,
        "exchange_slope_dlambda_dmu_iso500": exchange_slope,
        "interpretation": ("along iso-500, lowering mu (less width recovery) demands more "
                           "lambda (more spread recovery) to hold 500, and vice versa. The "
                           "two axis intercepts (0.90 spread @ full width, 0.70 width @ full "
                           "spread) bound the interior trade."),
    }

    # ---- realized-point reader (direct (lambda, mu) OR derived from land's ladder) ----
    realized, derived = None, None
    lam_in, mu_in = args.realized_lambda, args.realized_mu
    if lam_in is None or mu_in is None:
        live_in = _load_live_ladder(args)
        if live_in and (live_in.get("q_ladder") or live_in.get("branch_hit") is not None):
            derived = derive_lambda_mu_from_ladder(
                live_in.get("q_ladder"), live_in.get("branch_hit"), b_dict)
            if lam_in is None:
                lam_in = derived["lambda"]
            if mu_in is None:
                mu_in = derived["mu"]
    if lam_in is not None and mu_in is not None:
        lam_in = max(0.0, min(1.0, float(lam_in)))
        mu_in = max(0.0, min(1.0, float(mu_in)))
        realized = realized_point_reader(b_dict, step, tau, lam_in, mu_in, surface, derived)

    # ---- top-line state ----
    if realized is not None:
        state, go = realized["tps_verdict"], realized["go_no_go"]
        label = (f"REALIZED (lambda={realized['lambda']:.3f}, mu={realized['mu']:.3f}) -> official "
                 f"{realized['official_tps_central']:.1f} -> {state} / {go}; BINDING="
                 f"{realized['binding_facet']}, CHEAPEST-to-GREEN={realized['cheapest_direction_to_green']}")
    else:
        state, go = "ARMED", "PENDING"
        label = (f"JOINT FRONTIER ARMED + VALIDATED (self-test {'PASS' if st['passes'] else 'FAIL'}); "
                 f"awaiting land #71's (lambda, mu). GREEN-region area "
                 f"{green_region_area_fraction*100:.1f}% @ measured step {step:.4f}; iso-500 from "
                 f"(lambda*={iso_500['lambda_intercept_at_mu1']}, mu=1) to (lambda=1, mu*="
                 f"{iso_500['mu_intercept_at_lambda1']}). Coarse intercepts reproduce 0.90 / 0.70.")

    handoff = (
        "HAND-OFF to land #71: feed measured (lambda, mu) -- spread-recovery lambda (deeper "
        "spine q[2:] vs rho-opt) and width-recovery mu (branch-hit rho2 vs rho-opt) -- OR the "
        "raw q[1..9] + rho2 (auto-projected). The realized-point reader returns the JOINT "
        "GO/NO-GO, the BINDING facet, and the CHEAPEST recovery direction to cross 500. This "
        "is strictly more honest than either 1-D slice for a partial-both-facet landing: each "
        "slice assumes the OTHER facet is fully rho-opt, which the real descent will not be. "
        "Pairs with #142 (scalar gate), #145 (1-D slices), wirbel #146 (sampling CI), ubel "
        "#148 (calibration band).")

    # store a 1-dp surface (compact) + the axes; full-precision stays in-memory.
    tps_grid_store = [[round(v, 1) for v in row] for row in surface["tps_grid"]]

    out = {
        "primary_metric_name": "joint_frontier_self_test_passes",
        "joint_frontier_self_test_passes": joint_frontier_self_test_passes,
        "test_metric_name": "green_region_area_fraction",
        "green_region_area_fraction": green_region_area_fraction,
        "green_region_area_fraction_roofline": green_area_roofline,
        "gate_state": state, "gate_go_no_go": go, "gate_label": label,
        "land_measured_pending": bool(realized is None),
        "self_test": st,
        "surface": {
            "n": n, "axis_lambda": [round(x, 4) for x in lam_axis],
            "axis_mu": [round(x, 4) for x in mu_axis],
            "tps_grid_measured_step": tps_grid_store,
            "tps_min": surface["tps_min"], "tps_max": surface["tps_max"],
            "nan_cells": surface["nan_cells"], "nan_clean": surface["nan_clean"],
            "row_axis": "mu (width recovery)", "col_axis": "lambda (spread recovery)",
            "step_time": step, "tau": tau,
        },
        "iso_contours": {"iso_500": iso_500, "iso_481_53_frontier_parity": iso_481, "iso_530": iso_530},
        "green_region": {
            "area_fraction_measured_step": green_region_area_fraction,
            "area_fraction_roofline_step": green_area_roofline,
            "definition": "fraction of (lambda, mu) in [0,1]^2 with official TPS >= 500",
        },
        "corner_geometry": corner_geometry,
        "realized_point_reader": realized,
        "realized_derivation": derived,
        "handoff_land71": handoff,
        "map": {
            "figure_of_merit": "official_TPS = K_cal * E[T] / step_time * tau  (== #142 measured_m16_to_official core)",
            "K_cal": K_CAL, "tau_band": TAU, "operative_step": step,
            "operative_step_is_lawine136_merged": step_is_measured,
            "step_measured_depth9_lawine136": STEP_MEASURED_DEPTH9,
            "step_roofline_depth9_125": STEP_ROOFLINE_DEPTH9,
            "frontier_official": FRONTIER_OFFICIAL, "target_official": TARGET_OFFICIAL,
            "target_530": TARGET_530,
        },
        "axes_spec": {
            "lambda_spread_recovery": ("deep-spine-SPREAD recovery fraction: deeper rank-1 spine "
                                       "q[2:] from measured declining (lambda=0) to rho-opt rising "
                                       "(lambda=1); depth-1 held at rho-opt. == #145 spread axis."),
            "mu_width_recovery": ("branch-WIDTH recovery fraction: rank>=2 branch rescue rho_cond "
                                  "from 0 (mu=0) to rho-optimal (mu=1). == #145 width axis."),
            "edges": ("mu=1 column reproduces #145 spread_recovery_map (lambda-intercept 0.90); "
                      "lambda=1 row reproduces #145 width_recovery_map (mu-intercept 0.70)."),
        },
        "provenance": (
            "2-D joint frontier extending fern #145's two 1-D slices. Reuses #145's facet-recovery "
            "maps verbatim (joint_spine/joint_rho == spread/width slice construction), the #100 "
            "K_cal (125.268), the #142 gate compose (official_tps_map == measured_m16_to_official "
            "core), and wirbel's E[T] DP. depth-1 held at rho-opt (q76[0]=0.7287=DEPTH1_FP32_TARGET, "
            "verified). Anchors: (1,1)->537.8, (0,1)->376.3 (roofline); intercepts 0.90/0.70 "
            "(measured 1.2182). Rides on Issue #124 RESOLVED."),
        "method": ("LOCAL CPU-only analytic decision-geometry diagnostic; no GPU/vLLM/HF Job/"
                   "submission/kernel build. Does NOT authorize a launch; BASELINE stays 481.53. "
                   "Greedy identity untouched by construction."),
    }

    os.makedirs(os.path.dirname(args.joint_out), exist_ok=True)
    with open(args.joint_out, "w") as f:
        json.dump(out, f, indent=2)

    # ------------------------------- console -------------------------------
    print("=" * 100)
    print("JOINT (spread x width) CLEARS-500 FRONTIER -- the 2-D decision surface (PR #149)")
    print("=" * 100)
    print(f"\nmap: official = K_cal*E[T]/step*tau  (K_cal={K_CAL:.3f}, measured step={step:.4f}"
          f"{' [lawine #136 MERGED]' if step_is_measured else ' [land-supplied]'}, tau={tau:.4f}, "
          f"grid {n}x{n})")

    print(f"\n[SELF-TEST] joint map valid iff it reproduces the #145 anchors:")
    cf, cw = st["corner_full"], st["corner_widthonly"]
    print(f"  corner (1,1) FULL       E[T]={cf['et']:.3f} -> {cf['reconstructed_tps_roofline']:.1f} "
          f"(exp {cf['expected_tps']:.1f}, err {cf['rel_err']*100:.2f}%) @ roofline  "
          f"-> {'OK' if cf['within_2pct'] else 'FAIL'}")
    print(f"  corner (0,1) WIDTH-ONLY E[T]={cw['et']:.3f} -> {cw['reconstructed_tps_roofline']:.1f} "
          f"(exp {cw['expected_tps']:.1f}, err {cw['rel_err']*100:.2f}%) @ roofline  "
          f"-> {'OK' if cw['within_2pct'] else 'FAIL'}")
    print(f"  lambda-intercept @ mu=1 = {st['lambda_intercept_at_mu1']['coarse_first_clearing']} "
          f"(exp 0.90)  -> {'OK' if st['lambda_intercept_at_mu1']['within_1pct'] else 'FAIL'}")
    print(f"  mu-intercept @ lambda=1 = {st['mu_intercept_at_lambda1']['coarse_first_clearing']} "
          f"(exp 0.70)  -> {'OK' if st['mu_intercept_at_lambda1']['within_1pct'] else 'FAIL'}")
    print(f"  edge profiles within 1%: {st['edges_within_1pct']}  nan_clean: {st['nan_clean']}")
    print(f"  => joint_frontier_self_test_passes = {joint_frontier_self_test_passes}")

    print(f"\n[SURFACE] official TPS over (lambda, mu) in [0,1]^2  (min {surface['tps_min']:.1f}, "
          f"max {surface['tps_max']:.1f}):")
    print(f"  4 corners @ measured step: (0,0)={corners['lambda0_mu0']:.1f}  "
          f"(1,0)={corners['lambda1_mu0']:.1f}  (0,1)={corners['lambda0_mu1']:.1f}  "
          f"(1,1)={corners['lambda1_mu1']:.1f}")
    print(f"  GREEN-region (clears-500) area fraction = {green_region_area_fraction*100:.2f}% "
          f"(measured step), {green_area_roofline*100:.2f}% (roofline)")

    print(f"\n[ISO-500 CONTOUR] lambda_iso(mu) (spread recovery needed at each width recovery):")
    for m in (0.6, 0.7, 0.8, 0.9, 1.0):
        r = iso_500["lambda_iso_of_mu"][_nearest_idx(mu_axis, m)]
        li = "n/a (unreachable)" if r["lambda_iso"] is None else f"{r['lambda_iso']:.3f}"
        print(f"  mu={r['mu']:.2f}  ->  lambda_iso = {li}")
    print(f"  axis intercepts: lambda*(mu=1) continuous={iso_500['lambda_intercept_at_mu1']} "
          f"(coarse 0.90); mu*(lambda=1) continuous={iso_500['mu_intercept_at_lambda1']} (coarse 0.70)")

    if realized is not None:
        print(f"\n[REALIZED-POINT READER] (lambda={realized['lambda']:.3f}, mu={realized['mu']:.3f}):")
        print(f"  official {realized['official_tps_central']:.1f} (taulow "
              f"{realized['official_tps_taulow']:.1f})  verdict {realized['tps_verdict']}  "
              f"-> {realized['go_no_go']}  (margin {realized['margin_to_500']:+.1f})")
        g = realized["gradient_tps_per_unit_recovery"]
        print(f"  gradient: dTPS/dlambda(spread)={g['d_tps_d_lambda_spread']:+.1f}, "
              f"dTPS/dmu(width)={g['d_tps_d_mu_width']:+.1f} TPS per unit recovery")
        print(f"  BINDING facet = {realized['binding_facet']}  |  CHEAPEST to GREEN = "
              f"{realized['cheapest_direction_to_green']}")
        if derived and derived.get("depth1_short_of_rho_opt"):
            print(f"  NOTE: land depth-1 {derived['depth1_land']} < rho-opt {derived['depth1_rho_opt']:.4f} "
                  f"-- a SEPARATE denken #133 facet (off this (lambda, mu) surface).")
    else:
        print(f"\n[REALIZED-POINT READER] PENDING -- no land #71 (lambda, mu) yet. "
              f"green_region_area_fraction = {green_region_area_fraction*100:.2f}%.")

    print(f"\n[PRIMARY] joint_frontier_self_test_passes = {joint_frontier_self_test_passes}")
    print(f"[TEST]    green_region_area_fraction = {green_region_area_fraction:.4f}")
    print(f"\n[STATE] {state} / {go} -- {label}")
    print(f"\nwrote {args.joint_out}")

    # ------------------------------- W&B -------------------------------
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=args.wandb_name, group=args.wandb_group, job_type="analysis",
                         config={"diagnostic": "joint-spread-width-500-frontier",
                                 "method": "cpu-analytic-2d-frontier-extends-145-142",
                                 "K_cal": K_CAL, "operative_step": step,
                                 "step_is_lawine136_merged": step_is_measured,
                                 "step_roofline_125": STEP_ROOFLINE_DEPTH9,
                                 "tau_low": TAU["low"], "tau_central": TAU["central"],
                                 "grid_n": n, "frontier_official": FRONTIER_OFFICIAL,
                                 "target_official": TARGET_OFFICIAL, "target_530": TARGET_530,
                                 "land_measured_pending": bool(realized is None)})
        s = wandb.summary
        s["joint_frontier_self_test_passes"] = joint_frontier_self_test_passes
        s["green_region_area_fraction"] = green_region_area_fraction
        s["green_region_area_fraction_roofline"] = green_area_roofline
        s["selftest_corner_full_tps"] = st["corner_full"]["reconstructed_tps_roofline"]
        s["selftest_corner_full_relerr"] = st["corner_full"]["rel_err"]
        s["selftest_corner_widthonly_tps"] = st["corner_widthonly"]["reconstructed_tps_roofline"]
        s["selftest_corner_widthonly_relerr"] = st["corner_widthonly"]["rel_err"]
        s["selftest_lambda_intercept_mu1"] = (st["lambda_intercept_at_mu1"]["coarse_first_clearing"]
                                              if st["lambda_intercept_at_mu1"]["coarse_first_clearing"] is not None else -1.0)
        s["selftest_mu_intercept_lambda1"] = (st["mu_intercept_at_lambda1"]["coarse_first_clearing"]
                                             if st["mu_intercept_at_lambda1"]["coarse_first_clearing"] is not None else -1.0)
        s["surface_tps_min"] = surface["tps_min"]
        s["surface_tps_max"] = surface["tps_max"]
        s["corner_lambda1_mu1_tps"] = corners["lambda1_mu1"]
        s["corner_lambda0_mu1_tps"] = corners["lambda0_mu1"]
        s["corner_lambda1_mu0_tps"] = corners["lambda1_mu0"]
        s["corner_lambda0_mu0_tps"] = corners["lambda0_mu0"]
        s["iso500_lambda_intercept_mu1_continuous"] = (iso_500["lambda_intercept_at_mu1"]
                                                       if iso_500["lambda_intercept_at_mu1"] is not None else -1.0)
        s["iso500_mu_intercept_lambda1_continuous"] = (iso_500["mu_intercept_at_lambda1"]
                                                      if iso_500["mu_intercept_at_lambda1"] is not None else -1.0)
        s["gate_state"] = state
        s["gate_go_no_go"] = go
        s["gate_label"] = label
        if realized is not None:
            s["realized_lambda"] = realized["lambda"]
            s["realized_mu"] = realized["mu"]
            s["realized_official_tps_central"] = realized["official_tps_central"]
            s["realized_tps_verdict"] = realized["tps_verdict"]
            s["realized_go_no_go"] = realized["go_no_go"]
            s["realized_binding_facet"] = realized["binding_facet"]
            s["realized_cheapest_direction"] = realized["cheapest_direction_to_green"]
        # downsampled surface heatmap table (~21x21 to keep the table light).
        stride = max(1, n // 21)
        hm = wandb.Table(columns=["lambda", "mu", "official_tps", "clears_500"])
        for j in range(0, n, stride):
            for i in range(0, n, stride):
                v = surface["tps_grid"][j][i]
                hm.add_data(round(lam_axis[i], 4), round(mu_axis[j], 4), round(v, 2),
                            int(v >= TARGET_OFFICIAL))
        wandb.log({"joint_tps_surface": hm})
        # iso-500 contour table.
        ct = wandb.Table(columns=["mu", "lambda_iso_500"])
        for r in iso_500["lambda_iso_of_mu"]:
            ct.add_data(r["mu"], (-1.0 if r["lambda_iso"] is None else r["lambda_iso"]))
        wandb.log({"iso_500_contour": ct})
        # coarse intercept slices (the #145 reproduction).
        it = wandb.Table(columns=["axis", "frac", "official_tps", "clears_500"])
        for r in coarse_spread["rows"]:
            it.add_data("spread_lambda_at_mu1", r["frac"], r["official_tps"], int(r["clears_500"]))
        for r in coarse_width["rows"]:
            it.add_data("width_mu_at_lambda1", r["frac"], r["official_tps"], int(r["clears_500"]))
        wandb.log({"axis_intercept_slices": it})
        # corner table.
        cg = wandb.Table(columns=["corner", "lambda", "mu", "official_tps", "clears_500"])
        for key, (l, m) in {"lambda0_mu0": (0, 0), "lambda1_mu0": (1, 0),
                            "lambda0_mu1": (0, 1), "lambda1_mu1": (1, 1)}.items():
            v = corners[key]
            cg.add_data(key, l, m, round(v, 2), int(v >= TARGET_OFFICIAL))
        wandb.log({"corner_geometry": cg})
        print(f"\nW&B run: {run.id}  ({run.url})")
        wandb.finish()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rho", default="research/spec_cost_model/rho_optimal_topology_results.json")
    ap.add_argument("--oracle-json",
                    default="research/oracle_readout/oracle_live_tree488_fp32_20260614.json",
                    help="banked as-built fp32 oracle (the measured cumulative ladder).")
    ap.add_argument("--measured-step", type=float, default=None,
                    help="land #71's MEASURED depth-9 step. Omit to use lawine #136's "
                         "merged 1.2182.")
    ap.add_argument("--tau", type=float, default=TAU["central"])
    # land's measured per-depth ladder (optional; until it lands the gate runs ARMED).
    ap.add_argument("--measured-json", default=None,
                    help="land #71's measured readout {q_ladder (or per_position_cumulative_"
                         "accept), rho_cond, per_position_branch_hit, ppl, greedy_token_ids_captured}.")
    ap.add_argument("--measured-q-ladder", type=float, nargs="+", default=None,
                    help="land's measured per-depth CONDITIONAL accept ladder q[1..9].")
    ap.add_argument("--measured-branch-hit", type=float, default=None)
    ap.add_argument("--measured-ppl", type=float, default=None)
    ap.add_argument("--measured-greedy-captured", action="store_true", default=None)
    ap.add_argument("--out", default="research/oracle_readout/deep_spine_width_spread_decomp_results.json")
    ap.add_argument("--sample-out",
                    default="research/oracle_readout/deep_spine_decomp_input_sample.json")
    # ---- PR #149: joint (spread x width) clears-500 frontier mode ----
    ap.add_argument("--joint-frontier", action="store_true",
                    help="PR #149: emit the full 2-D (lambda spread x mu width) clears-500 "
                         "decision surface, contours, GREEN-area, self-test, realized-point reader.")
    ap.add_argument("--joint-grid-n", type=int, default=101,
                    help="joint surface grid resolution per axis (>= 101; clamped up to 101).")
    ap.add_argument("--joint-out",
                    default="research/oracle_readout/joint_spread_width_500_frontier_results.json",
                    help="output JSON for the joint-frontier mode.")
    ap.add_argument("--realized-lambda", type=float, default=None,
                    help="land #71's realized deep-spine-SPREAD recovery fraction (0..1).")
    ap.add_argument("--realized-mu", type=float, default=None,
                    help="land #71's realized branch-WIDTH recovery fraction (0..1).")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="gemma-challenge-senpai")
    ap.add_argument("--wandb-entity", default="wandb-applied-ai-team")
    ap.add_argument("--wandb-name", default="fern/deep-spine-width-spread-decomp")
    ap.add_argument("--wandb-group", default="deep-spine-width-spread-decomp")
    args = ap.parse_args()

    # operative live step (lawine #136 merged, unless land hands a fresher measured step).
    step = args.measured_step if args.measured_step is not None else STEP_MEASURED_DEPTH9
    step_is_measured_anchor = args.measured_step is None  # True => the merged lawine #136 step
    tau = args.tau

    b_dict = load_banked(args.rho, args.oracle_json)
    b_dict["_L"] = build_lattice(b_dict)
    L = b_dict["_L"]

    # PR #149: joint (spread x width) frontier mode -- self-contained; reuses the
    # banked facet-recovery maps and leaves the #145 default flow untouched.
    if args.joint_frontier:
        run_joint_frontier(args, b_dict, step, step_is_measured_anchor, tau)
        return

    # ---- PRIMARY: self-test (reproduce BOTH anchors within +-2%, additive, NaN-clean) ----
    st = self_test(b_dict)
    decomp_self_test_passes = int(st["passes"])

    # ---- decompositions (nested + Shapley), at the operative live step ----
    nested = nested_decomposition(L, step, tau)
    shapley = shapley_decomposition(L, step, tau)

    # ---- width-vs-spread band + continuous maps (live step) ----
    band = width_spread_band(b_dict, step, tau)
    band_roofline = width_spread_band(b_dict, STEP_ROOFLINE_DEPTH9, tau)  # the named 537.8/376.3
    spread_map = spread_recovery_map(b_dict, step, tau)
    width_map = width_recovery_map(b_dict, step, tau)

    # ---- TEST: band armed (self-test passes AND both endpoints finite + ordered) ----
    width_vs_spread_band_armed = int(
        st["passes"] and math.isfinite(band["full_endpoint_tps"])
        and math.isfinite(band["widthonly_endpoint_tps"])
        and band["full_endpoint_tps"] > band["widthonly_endpoint_tps"])

    # ---- live gate (if land's measured per-depth ladder is provided) ----
    live_in = _load_live_ladder(args)
    live = None
    land_pending = not (live_in and live_in.get("q_ladder"))
    if not land_pending:
        live = evaluate_live_ladder(
            live_in["q_ladder"], b_dict, step,
            rho_cond_land=live_in.get("rho_cond_land"),
            branch_hit=live_in.get("branch_hit"),
            ppl=live_in.get("ppl"),
            greedy_token_ids_captured=live_in.get("greedy_token_ids_captured"))

    # ---- top-line state ----
    if live is not None:
        gate_state = live["verdict"]
        gate_go = live["go_no_go"]
        gate_label = (
            f"LIVE: land q[1..9] -> E[T]={live['realized_et']:.3f} -> official "
            f"{live['realized_official_tps_central']:.1f} ({live['where_on_band']}); "
            f"{gate_state} / {gate_go}"
            + (f"  BINDING={live['binding_shortfall_facet']}" if live['failure_flag'] else ""))
    else:
        gate_state, gate_go = "ARMED", "PENDING"
        gate_label = (
            f"DECOMP ARMED + VALIDATED (self-test {'PASS' if st['passes'] else 'FAIL'}); "
            f"awaiting land #71's measured per-depth ladder. Width-vs-spread band "
            f"[{band['widthonly_endpoint_tps']:.1f} (width-only, spread=0) .. "
            f"{band['full_endpoint_tps']:.1f} (full)] @ measured step {step:.4f}; the "
            f"{band['swing_tps']:.1f}-TPS swing is ALL deep-spine-spread. Spread recovery "
            f">= {spread_map['spread_recovery_to_clear_500']} clears 500.")

    handoff = (
        "HAND-OFF to land #71 (on the re-bench): feed your measured per-depth conditional "
        "ladder q[1..9] (+ branch-hit rho2). A sub-GREEN E[T] is decomposed to a BINDING "
        "facet: a WIDTH failure (descent not re-seeding rescued nodes -> branch-hit/rho2 "
        "below rho-opt) vs a SPREAD failure (deep-spine decaying faster than rho-opt -> "
        "q[2:] short). The 537.8-vs-376.3 swing is ENTIRELY deep-spine-spread (facet c): "
        "width restored alone lands 376.3 (RED); only full deep-spine recovery reaches "
        "537.8. Pairs with my #134 per-position harness and #142 scalar gate.")

    out = {
        "primary_metric_name": "decomp_self_test_passes",
        "decomp_self_test_passes": decomp_self_test_passes,
        "test_metric_name": "width_vs_spread_band_armed",
        "width_vs_spread_band_armed": width_vs_spread_band_armed,
        "gate_state": gate_state,
        "gate_go_no_go": gate_go,
        "gate_label": gate_label,
        "land_measured_pending": bool(land_pending),
        "self_test": st,
        "decomposition_nested": nested,
        "decomposition_shapley": shapley,
        "width_vs_spread_band_measured_step": band,
        "width_vs_spread_band_roofline_anchors": band_roofline,
        "spread_recovery_map": spread_map,
        "width_recovery_map": width_map,
        "lattice_et": {f"a{a}b{b}c{c}": _finite(L[(a, b, c)])
                       for a in (0, 1) for b in (0, 1) for c in (0, 1)},
        "live_gate": live,
        "handoff_land71": handoff,
        "map": {
            "figure_of_merit": "official_TPS = K_cal * E[T] / step_time * tau",
            "K_cal": K_CAL, "tau_band": TAU,
            "step_measured_depth9_lawine136": STEP_MEASURED_DEPTH9,
            "step_roofline_depth9_125": STEP_ROOFLINE_DEPTH9,
            "operative_step": step, "operative_step_is_lawine136_merged": step_is_measured_anchor,
            "frontier_official": FRONTIER_OFFICIAL, "target_official": TARGET_OFFICIAL,
            "target_530": TARGET_530, "linear_floor_et": E_T_LINEAR, "supply_ceiling_et": E_T_TREE,
        },
        "facets_spec": {
            "depth1_spine": "q1: 0.674 measured -> 0.7287 fp32 target (denken #133; BUG-1 plumbing).",
            "branch_width": ("rank>=2 branch RESCUE at depths 2..9 (rho_cond [0.4165,0.2655,0.1908]); "
                             "the salvage the DESCENT fix re-seeds. == wirbel #135 width facet."),
            "deep_spine_spread": ("deeper rank-1 SPINE at depths 2..9 (measured declining -> rho-opt "
                                  "rising); the easy run the descending walk must keep. == wirbel #135 "
                                  "spread facet. THIS facet IS the 161.6-TPS band."),
            "naming_note": ("WIDTH = branch rescue (rank>=2 re-seeding); SPREAD = deep rank-1 spine "
                            "(depths 2..9). Aligned to wirbel #135 and this PR's hand-off semantics; "
                            "the width-only collapse (376.3) keeps width, loses spread."),
        },
        "measured_input_schema": {
            "q_ladder": "list[float] -- land's MEASURED per-depth CONDITIONAL accept q[1..9]",
            "per_position_cumulative_accept": "list[float] -- alt: cumulative ladder (converted)",
            "rho_cond": "list[float] -- optional measured rank-2..W rescue ratios",
            "per_position_branch_hit": "float -- measured rank-2 branch-hit rho2 (~0.4165)",
            "ppl": "float -- captured PPL (<= 2.42)",
            "greedy_token_ids_captured": "bool -- decode-audit IDs captured",
        },
        "provenance": (
            "Three-facet decomposition layer on fern #134's recovery matrix + #142 scalar "
            "gate. Reuses the #100 K_cal (125.268), the #142 gate compose verbatim, and "
            "wirbel's E[T] DP (build_depth_pvecs_measured / score_tree_depthrank) on the "
            "rho-optimal M=32/depth-9/max-branch-3 topology. Anchors: FULL E[T]=5.207->537.8 "
            "and WIDTH-ONLY E[T]=3.643->376.3 (fern #134 Fb sensitivity), reproduced within "
            "+-2% at the 1.2127 roofline. Live band re-prices at lawine #136's merged 1.2182. "
            "depth-1 target 0.7287 (denken #133); rho_cond [0.4165,0.2655,0.1908] (#79/#86)."),
        "method": ("LOCAL CPU-only analytic decomposition; no GPU/vLLM/HF Job/submission/"
                   "kernel build. Decision diagnostic only -- does NOT authorize a launch; "
                   "BASELINE stays 481.53. Greedy identity untouched by construction."),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # measured-input template for land #71 to fill (the rho-optimal ladder as EXAMPLE).
    os.makedirs(os.path.dirname(args.sample_out), exist_ok=True)
    sample = {
        "q_ladder": [round(x, 4) for x in (b_dict["q76"] + [b_dict["q76"][-1]] * 2)][:9],
        "per_position_branch_hit": RHO2_BRANCH_HIT,
        "ppl": 2.39,
        "greedy_token_ids_captured": True,
        "_note": ("EXAMPLE (the rho-optimal rising ladder -> the 537.8 full ceiling). Replace "
                  "with land #71's MEASURED per-depth conditional q[1..9]. Pass land's measured "
                  "depth-9 step via --measured-step (default = lawine #136's merged 1.2182)."),
    }
    with open(args.sample_out, "w") as f:
        json.dump(sample, f, indent=2)

    # ------------------------------- console -------------------------------
    print("=" * 100)
    print("DEEP-SPINE WIDTH-vs-SPREAD DECOMPOSITION -- the 537.8-vs-376 watched risk (PR #145)")
    print("=" * 100)
    print(f"\nmap: official = K_cal*E[T]/step*tau  (K_cal={K_CAL:.3f}, operative step="
          f"{step:.4f}{' [lawine #136 MEASURED, merged]' if step_is_measured_anchor else ' [land-supplied]'}, "
          f"tau_central={tau:.4f})")

    print(f"\n[SELF-TEST] decomp valid iff it reproduces BOTH banked anchors within +-2% "
          f"(@ roofline {STEP_ROOFLINE_DEPTH9:.4f}):")
    af, aw = st["anchor_full"], st["anchor_widthonly"]
    print(f"  FULL      E[T]={af['et']:.3f} -> {af['reconstructed_tps']:.1f} "
          f"(exp {af['expected_tps']:.1f}, err {af['rel_err']*100:.2f}%)  "
          f"-> {'OK' if af['within_2pct'] else 'FAIL'}")
    print(f"  WIDTH-ONLY E[T]={aw['et']:.3f} -> {aw['reconstructed_tps']:.1f} "
          f"(exp {aw['expected_tps']:.1f}, err {aw['rel_err']*100:.2f}%)  "
          f"-> {'OK' if aw['within_2pct'] else 'FAIL'}")
    print(f"  additive={st['additive_ok']}  nan_clean={st['nan_clean']}  "
          f"=> decomp_self_test_passes = {decomp_self_test_passes}")

    print(f"\n[DECOMPOSITION] three facets (marginal E[T] -> marginal TPS @ step {step:.4f}):")
    print(f"  {'facet':<20s} {'nested dE[T]':>12s} {'nested dTPS':>12s} {'Shapley dE[T]':>14s} {'Shapley dTPS':>13s}")
    for fac in ("depth1_spine", "branch_width", "deep_spine_spread"):
        nf, sf = nested["facets"][fac], shapley["facets"][fac]
        tag = "  <-- THE BAND" if fac == "deep_spine_spread" else ""
        print(f"  {fac:<20s} {nf['d_et']:>+12.4f} {nf['d_tps']:>+12.1f} "
              f"{sf['d_et']:>+14.4f} {sf['d_tps']:>+13.1f}{tag}")
    print(f"  baseline E[T]={nested['baseline_et']:.3f} -> full E[T]={nested['full_et']:.3f} "
          f"(total recovery {nested['total_recovery_et']:.4f})")

    print(f"\n[WIDTH-vs-SPREAD BAND] (the watched risk):")
    print(f"  banked anchors @ roofline {STEP_ROOFLINE_DEPTH9:.4f}: "
          f"width-only {band_roofline['widthonly_endpoint_tps']:.1f} .. full "
          f"{band_roofline['full_endpoint_tps']:.1f}  (swing {band_roofline['swing_tps']:.1f})")
    print(f"  live re-price @ measured {step:.4f}:               "
          f"width-only {band['widthonly_endpoint_tps']:.1f} .. full "
          f"{band['full_endpoint_tps']:.1f}  (swing {band['swing_tps']:.1f})")
    print(f"  binding facet = deep-spine-spread; spread recovery >= "
          f"{spread_map['spread_recovery_to_clear_500']} clears 500.")

    print(f"\n[SPREAD-RECOVERY MAP] deep-spine recovery fraction -> official (width+depth1 at rho-opt):")
    for r in spread_map["rows"]:
        flag = "YES" if r["clears_500"] else " no"
        print(f"  lambda={r['lambda_spread_recovery']:.2f}  E[T]={r['E_T']:.3f}  "
              f"official={r['official_tps']:7.1f}  clears500 {flag}")

    if live is not None:
        print(f"\n[LIVE GATE] land q[1..9] -> E[T]={live['realized_et']:.3f} -> official "
              f"{live['realized_official_tps_central']:.1f} (taulow "
              f"{live['realized_official_tps_taulow']:.1f})  verdict {live['verdict']}  "
              f"-> {live['go_no_go']}  [{live['where_on_band']}]")
        print(f"  TPS left on the table: depth1 {live['tps_left_on_table_by_facet']['depth1_spine']:+.1f}, "
              f"width {live['tps_left_on_table_by_facet']['branch_width']:+.1f}, "
              f"spread {live['tps_left_on_table_by_facet']['deep_spine_spread']:+.1f}  "
              f"-> BINDING = {live['binding_shortfall_facet']}")
        if live["failure_flag"]:
            print(f"  FLAG: {live['failure_flag']}")
    else:
        print(f"\n[LIVE GATE] PENDING -- no land #71 measured per-depth ladder yet. "
              f"width_vs_spread_band_armed = {width_vs_spread_band_armed}.")

    print(f"\n[PRIMARY] decomp_self_test_passes = {decomp_self_test_passes}")
    print(f"[TEST]    width_vs_spread_band_armed = {width_vs_spread_band_armed}")
    print(f"\n[STATE] {gate_state} / {gate_go} -- {gate_label}")
    print(f"\nwrote {args.out}")
    print(f"wrote {args.sample_out} (measured-input template for land #71)")

    # ------------------------------- W&B -------------------------------
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=args.wandb_name, group=args.wandb_group, job_type="analysis",
                         config={"diagnostic": "deep-spine-width-spread-decomp",
                                 "method": "cpu-analytic-3facet-decomp-extends-134-142",
                                 "K_cal": K_CAL, "operative_step": step,
                                 "step_measured_lawine136": STEP_MEASURED_DEPTH9,
                                 "step_roofline_125": STEP_ROOFLINE_DEPTH9,
                                 "tau_low": TAU["low"], "tau_central": TAU["central"],
                                 "frontier_official": FRONTIER_OFFICIAL,
                                 "target_official": TARGET_OFFICIAL, "target_530": TARGET_530,
                                 "depth1_fp32_target": DEPTH1_FP32_TARGET,
                                 "rho2_branch_hit": RHO2_BRANCH_HIT,
                                 "land_measured_pending": bool(land_pending)})
        s = wandb.summary
        s["decomp_self_test_passes"] = decomp_self_test_passes
        s["width_vs_spread_band_armed"] = width_vs_spread_band_armed
        s["selftest_anchor_full_tps"] = st["anchor_full"]["reconstructed_tps"]
        s["selftest_anchor_full_relerr"] = st["anchor_full"]["rel_err"]
        s["selftest_anchor_widthonly_tps"] = st["anchor_widthonly"]["reconstructed_tps"]
        s["selftest_anchor_widthonly_relerr"] = st["anchor_widthonly"]["rel_err"]
        s["band_full_tps"] = band["full_endpoint_tps"]
        s["band_widthonly_tps"] = band["widthonly_endpoint_tps"]
        s["band_swing_tps"] = band["swing_tps"]
        s["band_full_tps_roofline"] = band_roofline["full_endpoint_tps"]
        s["band_widthonly_tps_roofline"] = band_roofline["widthonly_endpoint_tps"]
        s["spread_recovery_to_clear_500"] = (spread_map["spread_recovery_to_clear_500"]
                                             if spread_map["spread_recovery_to_clear_500"] is not None else -1.0)
        for fac in ("depth1_spine", "branch_width", "deep_spine_spread"):
            s[f"nested_{fac}_d_et"] = nested["facets"][fac]["d_et"]
            s[f"nested_{fac}_d_tps"] = nested["facets"][fac]["d_tps"]
            s[f"shapley_{fac}_d_et"] = shapley["facets"][fac]["d_et"]
            s[f"shapley_{fac}_d_tps"] = shapley["facets"][fac]["d_tps"]
        s["gate_state"] = gate_state
        s["gate_go_no_go"] = gate_go
        s["gate_label"] = gate_label
        if live is not None:
            s["live_realized_et"] = live["realized_et"]
            s["live_realized_official_tps_central"] = live["realized_official_tps_central"]
            s["live_verdict"] = live["verdict"]
            s["live_binding_shortfall_facet"] = live["binding_shortfall_facet"]
            s["live_go_no_go"] = live["go_no_go"]
        # facet decomposition table
        ft = wandb.Table(columns=["facet", "nested_d_et", "nested_d_tps",
                                  "shapley_d_et", "shapley_d_tps"])
        for fac in ("depth1_spine", "branch_width", "deep_spine_spread"):
            ft.add_data(fac, nested["facets"][fac]["d_et"], nested["facets"][fac]["d_tps"],
                        shapley["facets"][fac]["d_et"], shapley["facets"][fac]["d_tps"])
        wandb.log({"facet_decomposition": ft})
        # spread-recovery map table
        sm = wandb.Table(columns=["lambda_spread_recovery", "E_T", "official_tps", "clears_500"])
        for r in spread_map["rows"]:
            sm.add_data(r["lambda_spread_recovery"], r["E_T"], r["official_tps"], int(r["clears_500"]))
        wandb.log({"spread_recovery_map": sm})
        # width-recovery cross-check table
        wm = wandb.Table(columns=["mu_width_recovery", "E_T", "official_tps", "clears_500"])
        for r in width_map["rows"]:
            wm.add_data(r["mu_width_recovery"], r["E_T"], r["official_tps"], int(r["clears_500"]))
        wandb.log({"width_recovery_map": wm})
        # band endpoints table
        bt = wandb.Table(columns=["endpoint", "step", "E_T", "official_tps"])
        bt.add_data("widthonly_roofline", STEP_ROOFLINE_DEPTH9, band_roofline["widthonly_endpoint_et"],
                    band_roofline["widthonly_endpoint_tps"])
        bt.add_data("full_roofline", STEP_ROOFLINE_DEPTH9, band_roofline["full_endpoint_et"],
                    band_roofline["full_endpoint_tps"])
        bt.add_data("widthonly_measured", step, band["widthonly_endpoint_et"], band["widthonly_endpoint_tps"])
        bt.add_data("full_measured", step, band["full_endpoint_et"], band["full_endpoint_tps"])
        wandb.log({"width_vs_spread_band": bt})
        print(f"\nW&B run: {run.id}  ({run.url})")
        wandb.finish()


if __name__ == "__main__":
    main()
