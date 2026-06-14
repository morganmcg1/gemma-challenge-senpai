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
