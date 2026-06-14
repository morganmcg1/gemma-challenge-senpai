#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Approval-issue projection-CI CONSOLIDATOR (PR #155).

WHAT THIS IS
------------
The one human-approved 500-shot is irreversible, and the evidence-line that
justifies it is scattered across five separate instruments the fleet built:
  * fern #142  m16_measured_500_gate           -- the scalar go/no-go GATE
  * wirbel #146 m16_gate_confidence_envelope    -- the sampling CI + required-N
  * ubel #148  kcal_tree_transfer_band.json     -- the calibration band (tau_tree)
  * fern #149  deep_spine_width_spread_decomp   -- the joint (spread x width) frontier
  * denken #150 validity preflight              -- PPL<=2.42 / boots / 128/128

When land #71's kernel emits its measured tuple, NO single artifact turns it into
the one thing the human approver needs:
    "the tree will score X +- Y TPS, P(clear 500) = Z%, validity = READY/NOT-READY".

This is NOT a seventh instrument. It IMPORTS and COLLAPSES the five legs above into
ONE call that emits a single all-uncertainty-propagated GO/NO-GO decision block --
the block that goes verbatim into the eventual `Approval request: HF job` issue. It
does NOT authorize a launch (the actual launch still goes through the human-approved
issue gate).

THE ONE ENTRY POINT
-------------------
    consolidate(E_T, branch_hit, spread_lambda, width_mu, step, ppl, tau)
      -> {proj_tps, ci_lo, ci_hi, p_clear_500, validity_gate, binding_leg, ...}

THE UNCERTAINTY MODEL (PR step 3)
---------------------------------
Central projection (the #142 map at tau_central=1):
    proj_tps = K_cal * E_T / step * tau_central          (K_cal=125.268)
Combined 1-sigma RELATIVE uncertainty, in quadrature over the three INDEPENDENT
legs (tau is NOT a separate term -- the tree-class clock-exposure floor lives
INSIDE the #148 calibration leg, so adding a tau term would double-count it):
    combined = sqrt(sampling^2 + calibration^2 + step_anchor^2)
  - sampling     : wirbel #146 bootstrap E[T] CI (tau/step-INVARIANT), or the
                   descending-regime per-step sigma / sqrt(N) handoff for a live
                   E[T]-only point.
  - calibration  : ubel #148 one-sided downward band (0.787%, tree-class tau floor
                   0.9924 + scorer amortization already folded in).
  - step_anchor  : lawine #136/#147 measured-step residual (0.5% half-width).
Decision-geometry fold-in (#149): locate (lambda, mu) on the joint frontier; a
PARTIAL-recovery point deep in the RED interior caps the clear-probability toward
0 regardless of the central CI. The headline P(clear 500) is the CONSERVATIVE
union of the direct-readout CI and the (lambda, mu)-geometry membership:
    p_clear_500 = min( Phi((proj_tps-500)/sigma), Phi((joint_tps(lam,mu)-500)/sigma) )
Validity gate (#150): NOT-READY (PPL>2.42, or boots fail, or completed!=128)
=> NO-GO irrespective of TPS. The binding leg names which source -- sampling,
calibration, step_anchor, decision_geometry, or validity -- flips the verdict.

SELF-VALIDATION (PR step 4 -- PRIMARY)
--------------------------------------
On the three bracketing anchors (priced at the measured step 1.2182, the operative
live band; the #142 271/538 point anchors live at the 1.2127 roofline):
  (a) oracle           E[T]=2.621, as-built (lam=0,mu=1)  -> robust-RED,   p~0
  (b) both-bugs-fixed  E[T]=5.207, (lam=mu=1)             -> robust-GREEN,  p~1
  (c) clear-500 bound  E[T]=4.862, full width (lam*,mu=1) -> INDETERMINATE, p~0.5
AND bit-match the imported legs' own outputs so the consolidator is provably a
faithful UNION, not a re-derivation:
  * wirbel #146 anchor TPS CI99 (tau-invariant hi + E[T] CI exact; the lo carries
    the SAME tree-class tau-floor correction PR #155 applied to #142, ~0.6% down)
  * ubel #148 K_cal band [124.282, 125.268]
  * fern #149 green-region area fraction 0.0300

LOCAL, CPU-ONLY, ANALYTIC. No GPU, no vLLM, no HF Job, no submission, no kernel
build. Imports the five legs verbatim (one source of truth per constant). Serves
nothing -> greedy identity untouched by construction. Rides on Issue #124 RESOLVED
(greedy-exact; PPL <= 2.42 binding).
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


# ============================================================================
# Import the five legs VERBATIM (does NOT duplicate them).
# ============================================================================
gate = _load("m16_measured_500_gate", os.path.join(_HERE, "m16_measured_500_gate.py"))
env = _load("m16_gate_confidence_envelope", os.path.join(_HERE, "m16_gate_confidence_envelope.py"))
decomp = _load("deep_spine_width_spread_decomp", os.path.join(_HERE, "deep_spine_width_spread_decomp.py"))

# fern #142 point gate (the central map + the official->E[T] inversion + tau band)
measured_m16_to_official = gate.measured_m16_to_official
official_tps_map = gate.official_tps_map
accept_length_for_official = gate.accept_length_for_official
K_CAL = gate.K_CAL                                  # 125.268
E_T_LINEAR = gate.E_T_LINEAR                        # 3.844 hard floor
E_T_TREE = gate.E_T_TREE                            # 5.207 rho-optimal ceiling
TARGET_OFFICIAL = gate.TARGET_OFFICIAL             # 500.0
FRONTIER_OFFICIAL = gate.FRONTIER_OFFICIAL         # 481.53
RHO2_BRANCH_HIT = gate.RHO2_BRANCH_HIT             # 0.4165 measured rank-2 branch-hit
PPL_GATE = gate.PPL_GATE                            # 2.42
TAU_GATE = gate.TAU                                 # {"low":0.9924...,"central":1,"high":1} (PR #155 fix)

# wirbel #146 confidence envelope (sampling CI + descending-regime sigma)
STEP_MEASURED_DEPTH9 = env.STEP_MEASURED_DEPTH9    # 1.2182 (lawine #136 measured step)
STEP_ROOFLINE_DEPTH9 = env.STEP_ROOFLINE_DEPTH9    # 1.2127 (#125 roofline; #142 anchors live here)
ORACLE_CUM_LADDER = env.ORACLE_CUM_LADDER
ORACLE_E_T = env.ORACLE_E_T                         # 2.621
ORACLE_STEPS = env.ORACLE_STEPS                     # 1024
DEPTH1_CEILING = env.DEPTH1_CEILING                # rho-optimal q1 -> 5.207
Z = env.Z                                           # {90:..,95:1.96,99:2.5758}
Z99 = Z[99]

# ubel #148 calibration band (read the committed artifact -- one source of truth)
KCAL_BAND_JSON = os.path.join(_ROOT, "research/kcal_tree_transfer/kcal_tree_transfer_band.json")

# default banked inputs for the #149 joint-frontier b_dict
RHO_OPT_JSON = os.path.join(_ROOT, "research/spec_cost_model/rho_optimal_topology_results.json")
ORACLE_LIVE_JSON = os.path.join(_ROOT, "research/oracle_readout/oracle_live_tree488_fp32_20260614.json")

# step-anchor residual (lawine #136 measured 1.2182 vs #125 roofline 1.2127 + jitter;
# wirbel #146's assumed step relative half-width). Used as the step_anchor 1-sigma.
STEP_REL_1SIGMA_DEFAULT = 0.005

# the three self-test anchors (measured-step operative band).
ANCHOR_BOUNDARY_ET = accept_length_for_official(TARGET_OFFICIAL, STEP_MEASURED_DEPTH9, 1.0)  # 4.8624


def _finite(x: float, default: float = 0.0) -> float:
    """NaN/inf -> default (every emitted metric must be NaN-clean)."""
    try:
        return float(x) if (x is not None and math.isfinite(float(x))) else default
    except (TypeError, ValueError):
        return default


def _phi(x: float) -> float:
    """Standard-normal CDF (scipy-free, via erf), clamped NaN-clean to [0, 1]."""
    if x is None or not math.isfinite(x):
        return 0.0
    return min(1.0, max(0.0, 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))))


def _jd(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    raise TypeError(f"{type(o).__name__} not JSON serializable")


# ============================================================================
# Leg A -- calibration band (ubel #148), read from the committed artifact.
# ============================================================================
def load_kcal_band(path: str = KCAL_BAND_JSON) -> dict:
    with open(path) as f:
        b = json.load(f)
    band = b["kcal_tree_transfer_band"]
    legA = band["legs"]["A_clock_exposure"]
    k_central = float(band["K_cal_central"])
    k_lo = float(band["K_cal_lo"])
    downside_pct = float(band["kcal_tree_transfer_band_width_pct"])
    return {
        "K_cal_central": k_central,
        "K_cal_lo": k_lo,
        "calib_downside_rel": downside_pct / 100.0,          # 0.00787 one-sided down (1-sigma)
        "calib_downside_pct": downside_pct,
        "tau_tree_floor": float(legA["scale_floor"]),         # 0.99243 (tree-class clock exposure)
        "band": [k_lo, k_central],
        "one_sided_downward": bool(band["band_is_one_sided_downward"]),
    }


# ============================================================================
# Leg C -- the #149 joint (spread x width) frontier b_dict (one DP build, reused).
# ============================================================================
def build_joint_b_dict(rho_path: str = RHO_OPT_JSON, oracle_path: str = ORACLE_LIVE_JSON) -> dict:
    b = decomp.load_banked(rho_path, oracle_path)
    b["_L"] = decomp.build_lattice(b)
    return b


def joint_tps_at(b_dict: dict, lam: float, mu: float, step: float, tau: float = 1.0) -> float:
    """official TPS at recovery point (lambda, mu) on the #149 surface. NaN-clean."""
    return _finite(decomp.joint_tps(lam, mu, b_dict, step, tau))


def lambda_intercept_at_mu1(b_dict: dict, step: float, tau: float = 1.0,
                            lo: float = 0.0, hi: float = 1.0) -> float:
    """Continuous lambda* where joint_tps(lambda*, mu=1) == 500 (bisection; TPS is
    monotone-nondecreasing in lambda). The on-contour boundary point for the
    INDETERMINATE anchor so its geometry membership is ~0.5."""
    if joint_tps_at(b_dict, hi, 1.0, step, tau) < TARGET_OFFICIAL:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if joint_tps_at(b_dict, mid, 1.0, step, tau) < TARGET_OFFICIAL:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ============================================================================
# Leg B -- sampling sigma (wirbel #146): one descending-regime per-step sigma,
# reproduced by CALLING #146's machinery (not re-deriving it).
# ============================================================================
class SamplingModel:
    """Holds wirbel #146's descending-regime per-step sigma and reproduces its three
    anchor CIs by calling analyse_sample VERBATIM. The per-step sigma drives the
    sampling 1-sigma for an E[T]-only live point via the CLT handoff
    samp_rel(E_T, N) = sigma_descend / sqrt(N) / E_T (== #146 min_robust_et_at_n)."""

    def __init__(self, n_steps: int = ORACLE_STEPS, n_boot: int = 20000, seed: int = 146,
                 step: float = STEP_MEASURED_DEPTH9, step_rel_hw: float = STEP_REL_1SIGMA_DEFAULT):
        self.n_steps = n_steps
        self.step = step
        self.step_rel_hw = step_rel_hw
        rng = np.random.default_rng(seed)
        dl = env.DescendingLadder(env.ACCEPT_JSON, env.RANKCOV_JSON, env.RHO_OPT_JSON)
        et_ceiling = dl.et(DEPTH1_CEILING)
        bars = env.effective_clear500_bar(step, step_rel_hw)
        et_border = bars["bar_central"]
        q1_border = dl.q1_for_et(et_border)
        C_ceiling = dl.ladder(DEPTH1_CEILING)
        C_border = dl.ladder(q1_border)
        s_oracle = env.samples_from_ladder(ORACLE_CUM_LADDER, n_steps, ORACLE_E_T)
        s_ceiling = env.samples_from_ladder(C_ceiling, n_steps, et_ceiling)
        s_border = env.samples_from_ladder(C_border, n_steps, et_border)
        k_rho2 = int(round(RHO2_BRANCH_HIT * n_steps))
        self.anchor_red = env.analyse_sample("as_built_oracle", s_oracle, k_rho2, n_steps,
                                              step, step_rel_hw, n_boot, rng, point_et=ORACLE_E_T)
        self.anchor_green = env.analyse_sample("rho_optimal_ceiling", s_ceiling, k_rho2, n_steps,
                                               step, step_rel_hw, n_boot, rng, point_et=et_ceiling)
        self.anchor_border = env.analyse_sample("clear500_boundary", s_border, k_rho2, n_steps,
                                                step, step_rel_hw, n_boot, rng, point_et=et_border)
        # descending-regime per-step sigma (the launch-relevant regime; #146 handoff)
        self.sigma_descend = float(self.anchor_green["per_step_sd"])
        self.et_ceiling = float(et_ceiling)
        self.et_border = float(et_border)

    def samp_rel_1sigma(self, E_T: float, n_steps: int | None = None) -> float:
        """Sampling 1-sigma RELATIVE on the projected TPS at E[T] (== relative 1-sigma
        on the mean accept-length): sigma_descend / sqrt(N) / E_T. tau/step-invariant."""
        n = n_steps if n_steps is not None else self.n_steps
        if E_T is None or E_T <= 0 or n <= 0:
            return 0.0
        return _finite(self.sigma_descend / math.sqrt(n) / E_T)

    @staticmethod
    def et_ci_rel_1sigma_from_anchor(anchor: dict, conf: int = 99) -> float:
        """tau-invariant sampling 1-sigma from a #146 anchor's bootstrap E[T] CI:
        (et_hi - et_lo) / (2 * z_conf * et_mid). Used to bit-match the anchors."""
        ci = anchor["et_ci_bootstrap"][str(conf)]
        et_mid = anchor["point_et"]
        hw = 0.5 * (ci["hi"] - ci["lo"])
        return _finite(hw / (Z[conf] * et_mid)) if et_mid > 0 else 0.0


# ============================================================================
# Validity gate (denken #150 contract, implemented inline -- PPL & boots & 128/128).
# ============================================================================
def validity_gate(ppl: float | None, boots: bool | None, completed: int | None) -> dict:
    ppl_ok = (ppl is not None) and bool(ppl <= PPL_GATE)
    boots_ok = (boots is True)
    completed_ok = (completed is not None) and (int(completed) == 128)
    ready = bool(ppl_ok and boots_ok and completed_ok)
    reasons = []
    if ppl is None:
        reasons.append("ppl_not_captured")
    elif not ppl_ok:
        reasons.append(f"ppl_{ppl:.4f}_exceeds_{PPL_GATE}")
    if boots is None:
        reasons.append("boots_unknown")
    elif not boots_ok:
        reasons.append("boots_failed")
    if completed is None:
        reasons.append("completed_unknown")
    elif not completed_ok:
        reasons.append(f"completed_{completed}_not_128")
    return {
        "gate": "READY" if ready else "NOT-READY",
        "ready": ready,
        "ppl": ppl, "ppl_gate": PPL_GATE, "ppl_within_gate": ppl_ok,
        "boots": boots, "boots_ok": boots_ok,
        "completed": completed, "completed_128": completed_ok,
        "blocking_reasons": reasons,
    }


# ============================================================================
# THE CONSOLIDATOR -- one call collapses all five legs into one GO/NO-GO.
# ============================================================================
def consolidate(E_T: float,
                branch_hit: float | None,
                spread_lambda: float | None,
                width_mu: float | None,
                step: float = STEP_MEASURED_DEPTH9,
                ppl: float | None = None,
                tau: float = 1.0,
                *,
                boots: bool | None = None,
                completed: int | None = None,
                n_steps: int | None = None,
                conf: int = 99,
                sampling: "SamplingModel | None" = None,
                b_dict: dict | None = None,
                kcal_band: dict | None = None) -> dict:
    """Collapse land #71's measured tuple into one all-uncertainty-propagated GO/NO-GO.

    Free inputs (land's measured readout -- the SAME tuple #142/#146/#150 consume):
      E_T            MEASURED accept-length E[T] (the numerator)
      branch_hit     measured rank-2 branch-hit rho2 (land's local topology gate)
      spread_lambda  deep-spine-spread recovery fraction lambda in [0,1] (#149 axis)
      width_mu       branch-width recovery fraction mu in [0,1] (#149 axis)
      step           measured depth-9 decode step (lawine #136; default 1.2182)
      ppl            captured PPL (validity gate; <= 2.42)
      tau            local->official transfer central (pinned 1.0; the tree-class
                     floor lives in the #148 calibration leg, not here)
    Validity (denken #150 contract): boots, completed (128 required).

    Returns the approval-issue projection block: proj_tps, ci_lo, ci_hi,
    p_clear_500, validity_gate, binding_leg (+ rich detail).
    """
    sampling = sampling or SamplingModel(step=step)
    b_dict = b_dict or build_joint_b_dict()
    kcal_band = kcal_band or load_kcal_band()
    z = Z[conf]

    # ---------- central projection (the #142 map at tau_central) ----------
    k_central = kcal_band["K_cal_central"]
    proj_tps = _finite(k_central * E_T / step * tau)

    # ---------- the three independent uncertainty legs (1-sigma RELATIVE) ----------
    samp_rel = sampling.samp_rel_1sigma(E_T, n_steps)          # wirbel #146 (tau/step-invariant)
    calib_rel = kcal_band["calib_downside_rel"]                 # ubel #148 (tree-class tau folded in)
    step_rel = sampling.step_rel_hw                            # lawine #136/#147 measured-step residual
    combined_rel = _finite(math.sqrt(samp_rel ** 2 + calib_rel ** 2 + step_rel ** 2))
    sigma_tps = _finite(combined_rel * proj_tps)

    # ---------- combined CI band (E[T]-direct, all three legs in quadrature) ----------
    ci_lo = _finite(proj_tps * (1.0 - z * combined_rel))
    ci_hi = _finite(proj_tps * (1.0 + z * combined_rel))

    # ---------- CI-based clear probability ----------
    p_ci = _phi((proj_tps - TARGET_OFFICIAL) / sigma_tps) if sigma_tps > 0 else \
        (1.0 if proj_tps >= TARGET_OFFICIAL else 0.0)

    # ---------- decision-geometry fold-in (#149): green membership of (lambda, mu) ----------
    geom_known = (spread_lambda is not None and width_mu is not None)
    if geom_known:
        lam = max(0.0, min(1.0, float(spread_lambda)))
        mu = max(0.0, min(1.0, float(width_mu)))
        geom_tps = joint_tps_at(b_dict, lam, mu, step, tau)
        sigma_geom = _finite(combined_rel * geom_tps)
        p_geom = _phi((geom_tps - TARGET_OFFICIAL) / sigma_geom) if sigma_geom > 0 else \
            (1.0 if geom_tps >= TARGET_OFFICIAL else 0.0)
    else:
        lam = mu = None
        geom_tps = proj_tps                      # no geometry -> CI-only (geometry cannot suppress)
        p_geom = p_ci

    # CONSERVATIVE union: a point deep in the RED interior caps p toward 0 regardless
    # of the central CI; geometry can only SUPPRESS, never inflate beyond the CI.
    p_clear_500 = _finite(min(p_ci, p_geom))

    # ---------- validity gate (#150) ----------
    vg = validity_gate(ppl, boots, completed)
    validity_ready = vg["ready"]

    # ---------- robust verdict (GO / NO-GO / HOLD) ----------
    geom_clears = (p_geom >= 0.99)
    geom_dead = (p_geom <= 0.01)
    if not validity_ready:
        tps_verdict = "robust-RED" if (ci_hi < TARGET_OFFICIAL or geom_dead) else "INDETERMINATE"
        verdict = "robust-GREEN" if (ci_lo >= TARGET_OFFICIAL and geom_clears) else tps_verdict
        go_no_go = "NO-GO"        # validity NOT-READY => NO-GO irrespective of TPS
    else:
        if ci_lo >= TARGET_OFFICIAL and geom_clears:
            verdict = "robust-GREEN"
            go_no_go = "GO"
        elif ci_hi < TARGET_OFFICIAL or geom_dead:
            verdict = "robust-RED"
            go_no_go = "NO-GO"
        else:
            verdict = "INDETERMINATE"
            go_no_go = "HOLD"     # needs more N / re-bench before the irreversible shot

    # ---------- hard-floor sanity (the #142 linear-MTP floor) ----------
    above_linear_floor = bool(E_T > E_T_LINEAR)

    # ---------- name the BINDING leg (what flips/limits the verdict) ----------
    leg_down_tps = {
        "sampling": _finite(proj_tps * z * samp_rel),
        "calibration": _finite(proj_tps * z * calib_rel),
        "step_anchor": _finite(proj_tps * z * step_rel),
    }
    geom_caps = bool(geom_known and (p_geom < p_ci - 0.02))
    if not validity_ready:
        binding_leg = "validity"
    elif not above_linear_floor:
        binding_leg = "linear_floor_hard_abort"
    elif geom_caps:
        binding_leg = "decision_geometry"          # the (lambda, mu) recovery state limits it
    elif verdict == "robust-RED":
        binding_leg = "central_projection_below_500"
    else:
        binding_leg = max(leg_down_tps, key=leg_down_tps.get)
    verdict_robust_to_legs = bool(verdict in ("robust-GREEN", "robust-RED"))

    return {
        "inputs": {
            "E_T": E_T, "branch_hit": branch_hit,
            "spread_lambda": spread_lambda, "width_mu": width_mu,
            "step": step, "ppl": ppl, "tau": tau,
            "boots": boots, "completed": completed,
            "n_steps": n_steps if n_steps is not None else sampling.n_steps,
        },
        # ---- the six PR-mandated outputs ----
        "proj_tps": proj_tps,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_clear_500": p_clear_500,
        "validity_gate": vg["gate"],
        "binding_leg": binding_leg,
        # ---- decision + verdict ----
        "verdict": verdict,
        "go_no_go": go_no_go,
        "confidence_pct": conf,
        # ---- uncertainty decomposition (quadrature legs) ----
        "uncertainty": {
            "combined_rel_1sigma": combined_rel,
            "sigma_tps": sigma_tps,
            "legs_rel_1sigma": {"sampling": samp_rel, "calibration": calib_rel, "step_anchor": step_rel},
            "legs_downside_tps_at_conf": leg_down_tps,
            "quadrature_formula": "combined = sqrt(sampling^2 + calibration^2 + step_anchor^2)",
            "tau_note": ("tau is NOT a separate quadrature term -- the tree-class clock-exposure "
                         "floor 0.9924 is folded INSIDE the #148 calibration leg (adding tau would "
                         "double-count)."),
        },
        # ---- CI-vs-geometry probabilities ----
        "probabilities": {
            "p_ci_direct_readout": p_ci,
            "p_geom_membership": p_geom,
            "p_clear_500_conservative_union": p_clear_500,
            "combination": "p_clear_500 = min(p_ci, p_geom) (geometry can only suppress)",
        },
        # ---- decision-geometry detail (#149) ----
        "decision_geometry": {
            "spread_lambda": lam, "width_mu": mu,
            "geom_tps": geom_tps, "geom_clears_500": bool(geom_known and geom_tps >= TARGET_OFFICIAL),
            "geom_membership_prob": p_geom, "geom_caps_verdict": geom_caps,
        },
        # ---- validity detail (#150) ----
        "validity_detail": vg,
        # ---- preconditions ----
        "preconditions": {
            "above_linear_floor_3p844": above_linear_floor,
            "branch_hit": branch_hit, "branch_hit_target_rho2": RHO2_BRANCH_HIT,
            "verdict_robust_to_all_legs": verdict_robust_to_legs,
        },
        "binding_leg_detail": {
            "binding_leg": binding_leg,
            "geometry_caps": geom_caps,
            "leg_downside_tps": leg_down_tps,
            "what_it_means": ("the source that, if it grew, would first flip the verdict; "
                              "'validity' = NOT-READY blocks GO; 'decision_geometry' = the "
                              "(lambda,mu) recovery state caps the clear-prob below the central CI."),
        },
        "map": {
            "figure_of_merit": "official_TPS = K_cal * E_T / step * tau",
            "K_cal_central": k_central, "K_cal_lo": kcal_band["K_cal_lo"],
            "step": step, "tau_central": tau, "target_official": TARGET_OFFICIAL,
        },
    }


# ============================================================================
# Self-validation (PR step 4 -- PRIMARY metric).
# ============================================================================
def _rel_err(v, ref):
    return abs(v - ref) / ref if ref else float("inf")


def build_self_test(sampling: SamplingModel, b_dict: dict, kcal_band: dict, step: float) -> dict:
    """Run the consolidator on the three bracketing anchors AND bit-match the imported
    legs. PRIMARY metric = consolidator_self_test_passes; TEST = p_clear_500_at_oracle."""
    # on-contour boundary lambda so the INDETERMINATE anchor's geometry membership ~ 0.5
    lam_star = lambda_intercept_at_mu1(b_dict, step, 1.0)

    common = dict(step=step, ppl=2.39, tau=1.0, boots=True, completed=128,
                  sampling=sampling, b_dict=b_dict, kcal_band=kcal_band)
    # (a) oracle: as-built (lambda=0 spread, mu=1 full measured width); E[T]=2.621.
    oracle = consolidate(ORACLE_E_T, RHO2_BRANCH_HIT, 0.0, 1.0, **common)
    # (b) both-bugs-fixed: full recovery (lambda=mu=1); E[T]=5.207.
    both = consolidate(E_T_TREE, RHO2_BRANCH_HIT, 1.0, 1.0, **common)
    # (c) clear-500 boundary: full width (mu=1), on-contour lambda*; E[T]=4.862.
    border = consolidate(ANCHOR_BOUNDARY_ET, RHO2_BRANCH_HIT, lam_star, 1.0, **common)

    # ---- anchor verdict + p_clear expectations ----
    oracle_ok = bool(oracle["verdict"] == "robust-RED" and oracle["p_clear_500"] <= 0.02)
    both_ok = bool(both["verdict"] == "robust-GREEN" and both["p_clear_500"] >= 0.98)
    border_ok = bool(border["verdict"] == "INDETERMINATE" and abs(border["p_clear_500"] - 0.5) <= 0.10)
    anchors_ok = bool(oracle_ok and both_ok and border_ok)

    # ---- bit-match leg outputs (faithful UNION, not re-derivation) ----
    # wirbel #146: tau-INVARIANT hi (tau_high=1.0) bit-matches the published values; the
    # E[T] CI bit-matches exactly; the lo carries the SAME tree-class tau-floor correction
    # PR #155 applied to #142 (#146 imports gate.TAU) -> ~0.6% down, NO verdict flip.
    pub146 = {"red": (253.1, 286.1), "green": (506.8, 563.6), "border": (471.7, 528.1)}
    bm146 = {}
    for tag, anc in (("red", sampling.anchor_red), ("green", sampling.anchor_green),
                     ("border", sampling.anchor_border)):
        c = anc["tps_ci_bootstrap"]["99"]
        lo_pub, hi_pub = pub146[tag]
        bm146[tag] = {
            "tps_ci99_lo_reproduced": _finite(c["tps_lo"]), "tps_ci99_lo_published_prefix": lo_pub,
            "tps_ci99_hi_reproduced": _finite(c["tps_hi"]), "tps_ci99_hi_published": hi_pub,
            "hi_bitmatch_exact": bool(_rel_err(c["tps_hi"], hi_pub) <= 0.001),
            "lo_within_tau_correction_1pct": bool(_rel_err(c["tps_lo"], lo_pub) <= 0.01),
            "robust_verdict_99": anc["robust_verdict_99"],
        }
    v146 = (sampling.anchor_red["robust_verdict_99"] == "robust-RED"
            and sampling.anchor_green["robust_verdict_99"] == "robust-GREEN"
            and sampling.anchor_border["robust_verdict_99"] == "INDETERMINATE")
    bitmatch_146 = bool(all(bm146[t]["hi_bitmatch_exact"] and bm146[t]["lo_within_tau_correction_1pct"]
                            for t in bm146) and v146)

    # ubel #148: K_cal band [124.282, 125.268] (exact read of the committed artifact).
    bitmatch_148 = bool(_rel_err(kcal_band["K_cal_lo"], 124.282) <= 1e-4
                        and _rel_err(kcal_band["K_cal_central"], 125.268) <= 1e-4)

    # fern #149: green-region area fraction 0.0300 (reproduce by calling the leg's code).
    surface = decomp.joint_frontier_surface(b_dict, step, 1.0, n=101)
    green_area = _finite(decomp.green_area_fraction(surface, TARGET_OFFICIAL))
    bitmatch_149 = bool(abs(green_area - 0.0300) <= 0.0005)

    bitmatch_ok = bool(bitmatch_146 and bitmatch_148 and bitmatch_149)
    passes = bool(anchors_ok and bitmatch_ok)

    return {
        "passes": passes,
        "anchors_ok": anchors_ok,
        "bitmatch_ok": bitmatch_ok,
        "lambda_star_boundary": lam_star,
        "anchors": {
            "oracle_RED": {"E_T": ORACLE_E_T, "lambda": 0.0, "mu": 1.0,
                           "proj_tps": oracle["proj_tps"], "ci": [oracle["ci_lo"], oracle["ci_hi"]],
                           "p_clear_500": oracle["p_clear_500"], "verdict": oracle["verdict"],
                           "go_no_go": oracle["go_no_go"], "binding_leg": oracle["binding_leg"],
                           "expected": "robust-RED / p~0", "ok": oracle_ok},
            "both_bugs_GREEN": {"E_T": E_T_TREE, "lambda": 1.0, "mu": 1.0,
                                "proj_tps": both["proj_tps"], "ci": [both["ci_lo"], both["ci_hi"]],
                                "p_clear_500": both["p_clear_500"], "verdict": both["verdict"],
                                "go_no_go": both["go_no_go"], "binding_leg": both["binding_leg"],
                                "expected": "robust-GREEN / p~1", "ok": both_ok},
            "boundary_INDETERMINATE": {"E_T": ANCHOR_BOUNDARY_ET, "lambda": lam_star, "mu": 1.0,
                                       "proj_tps": border["proj_tps"], "ci": [border["ci_lo"], border["ci_hi"]],
                                       "p_clear_500": border["p_clear_500"], "verdict": border["verdict"],
                                       "go_no_go": border["go_no_go"], "binding_leg": border["binding_leg"],
                                       "expected": "INDETERMINATE / p~0.5", "ok": border_ok},
        },
        "bitmatch": {
            "wirbel_146_sampling": {"ok": bitmatch_146, "anchors": bm146,
                                    "note": ("hi (tau_high=1.0) + E[T] CI bit-match exactly; lo carries "
                                             "the SAME tree-class tau-floor correction PR #155 applied to "
                                             "#142 (#146 imports gate.TAU) -> ~0.6% down, no verdict flip.")},
            "ubel_148_calibration": {"ok": bitmatch_148, "K_cal_band": kcal_band["band"],
                                     "expected": [124.282, 125.268]},
            "fern_149_green_area": {"ok": bitmatch_149, "green_region_area_fraction": green_area,
                                    "expected": 0.0300},
        },
        "_anchor_objs": {"oracle": oracle, "both": both, "border": border},
    }


# ============================================================================
# Hand-off: the exact tuple land #71 must report (same readout #142/#146/#150 use).
# ============================================================================
LAND_TUPLE_SPEC = {
    "E_T": "MEASURED accept-length E[T] (numerator) -- the SAME number #142/#146 consume",
    "branch_hit": "measured rank-2 conditional branch-hit rho2 (~0.4165; land's local gate)",
    "spread_lambda": "deep-spine-spread recovery fraction lambda in [0,1] (#149 axis; or derive "
                     "from the per-depth conditional ladder q[2:] via #149 derive_lambda_mu_from_ladder)",
    "width_mu": "branch-width recovery fraction mu in [0,1] (#149 axis; = branch_hit/rho2 clamped)",
    "step": "lawine #136 MEASURED depth-9 decode step (s); default 1.2182",
    "ppl": "captured PPL (validity; must be <= 2.42)",
    "boots": "submission boots cleanly (validity)",
    "completed": "benchmark completed-prompt count (validity; must be 128)",
    "_note": ("NO new measurement is required beyond what #142 / #146 / #150 already consume -- "
              "the consolidator ASSEMBLES, it does not add a measurement requirement."),
}


def _load_live(path: str | None) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        m = json.load(f)
    if "E_T" not in m and "accept_length" not in m:
        return None
    return {
        "E_T": float(m.get("E_T", m.get("accept_length"))),
        "branch_hit": m.get("branch_hit", m.get("per_position_branch_hit")),
        "spread_lambda": m.get("spread_lambda"),
        "width_mu": m.get("width_mu"),
        "step": float(m.get("step", STEP_MEASURED_DEPTH9)),
        "ppl": m.get("ppl"),
        "boots": m.get("boots"),
        "completed": m.get("completed"),
        "n_steps": m.get("n_steps"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measured-step", type=float, default=STEP_MEASURED_DEPTH9,
                    help="operative depth-9 step (lawine #136 measured 1.2182).")
    ap.add_argument("--step-rel-half-width", type=float, default=STEP_REL_1SIGMA_DEFAULT,
                    help="step-anchor 1-sigma relative (lawine #136/#147 residual; default 0.5%%).")
    ap.add_argument("--n-steps", type=int, default=ORACLE_STEPS,
                    help="verify-step budget for the sampling CI (oracle 1024).")
    ap.add_argument("--n-boot", type=int, default=20000, help="bootstrap resamples for #146 anchors.")
    ap.add_argument("--seed", type=int, default=146)
    ap.add_argument("--measured-json", default=None,
                    help="land #71's live tuple {E_T, branch_hit, spread_lambda, width_mu, step, "
                         "ppl, boots, completed}.")
    ap.add_argument("--rho-json", default=RHO_OPT_JSON)
    ap.add_argument("--oracle-json", default=ORACLE_LIVE_JSON)
    ap.add_argument("--kcal-band-json", default=KCAL_BAND_JSON)
    ap.add_argument("--out", default="research/spec_cost_model/approval_projection_consolidator_results.json")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="gemma-challenge-senpai")
    ap.add_argument("--wandb-entity", default="wandb-applied-ai-team")
    ap.add_argument("--wandb-name", default="fern/approval-projection-consolidator")
    ap.add_argument("--wandb-group", default="approval-projection-consolidator")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    step = args.measured_step

    print("=" * 96)
    print("APPROVAL-ISSUE PROJECTION-CI CONSOLIDATOR (PR #155)")
    print("=" * 96)
    print(f"map: official = K_cal*E_T/step*tau  (K_cal={K_CAL:.3f}, measured step={step:.4f}, "
          f"tau_central=1.0)", flush=True)
    print("legs: fern#142 gate + wirbel#146 sampling-CI + ubel#148 calibration + fern#149 "
          "joint-frontier + denken#150 validity\n", flush=True)

    # ---- assemble the legs once (one DP build / one bootstrap) ----
    sampling = SamplingModel(n_steps=args.n_steps, n_boot=args.n_boot, seed=args.seed,
                             step=step, step_rel_hw=args.step_rel_half_width)
    b_dict = build_joint_b_dict(args.rho_json, args.oracle_json)
    kcal_band = load_kcal_band(args.kcal_band_json)

    print(f"[LEGS] #146 descending-regime per-step sigma = {sampling.sigma_descend:.4f}; "
          f"#148 calib downside = {kcal_band['calib_downside_pct']:.4f}% "
          f"(tree-class tau floor {kcal_band['tau_tree_floor']:.6f}); step-anchor 1sigma = "
          f"{args.step_rel_half_width*100:.2f}%", flush=True)
    # consistency: PR #155 aligned #142's tau-low with #148's tree-class floor.
    tau_aligned = abs(TAU_GATE["low"] - kcal_band["tau_tree_floor"]) <= 5e-4
    print(f"[FIX #155] #142 TAU.low = {TAU_GATE['low']:.6f}  vs  #148 tree-class floor "
          f"{kcal_band['tau_tree_floor']:.6f}  -> aligned={tau_aligned} (was SplitK 0.9983)\n", flush=True)

    # ---- self-validation (PRIMARY) ----
    st = build_self_test(sampling, b_dict, kcal_band, step)
    consolidator_self_test_passes = int(st["passes"])
    p_clear_500_at_oracle = _finite(st["anchors"]["oracle_RED"]["p_clear_500"])

    print("[SELF-TEST] three bracketing anchors (measured step):")
    for key, a in st["anchors"].items():
        print(f"  {key:24s} E[T]={a['E_T']:.3f} (lam={a['lambda']:.3f},mu={a['mu']:.3f}) -> "
              f"proj {a['proj_tps']:6.1f}  CI[{a['ci'][0]:6.1f},{a['ci'][1]:6.1f}]  "
              f"p(>=500)={a['p_clear_500']:.3f}  {a['verdict']:>14s}  bind={a['binding_leg']}  "
              f"-> {'OK' if a['ok'] else 'FAIL'}", flush=True)
    print(f"\n[BIT-MATCH] faithful-union proof (imports, not re-derivation):", flush=True)
    b = st["bitmatch"]
    print(f"  wirbel #146 sampling CIs : {'OK' if b['wirbel_146_sampling']['ok'] else 'FAIL'}  "
          f"(hi+E[T] exact; lo carries the #155 tree-tau correction)", flush=True)
    for tag in ("red", "green", "border"):
        m = b["wirbel_146_sampling"]["anchors"][tag]
        print(f"      {tag:6s} CI99 reproduced [{m['tps_ci99_lo_reproduced']:6.1f},"
              f"{m['tps_ci99_hi_reproduced']:6.1f}]  published-prefix "
              f"[{m['tps_ci99_lo_published_prefix']:6.1f},{m['tps_ci99_hi_published']:6.1f}]  "
              f"{m['robust_verdict_99']}", flush=True)
    print(f"  ubel #148 K_cal band     : {'OK' if b['ubel_148_calibration']['ok'] else 'FAIL'}  "
          f"{[round(x,3) for x in b['ubel_148_calibration']['K_cal_band']]} (exp [124.282,125.268])",
          flush=True)
    print(f"  fern #149 green-area     : {'OK' if b['fern_149_green_area']['ok'] else 'FAIL'}  "
          f"{b['fern_149_green_area']['green_region_area_fraction']:.4f} (exp 0.0300)", flush=True)
    print(f"\n[PRIMARY] consolidator_self_test_passes = {consolidator_self_test_passes}", flush=True)
    print(f"[TEST]    p_clear_500_at_oracle = {p_clear_500_at_oracle:.4f} (expect ~0)\n", flush=True)

    # ---- live consolidation (if land #71's tuple is provided) ----
    live = _load_live(args.measured_json)
    live_out = None
    land_pending = live is None
    if live is not None:
        live_out = consolidate(live["E_T"], live["branch_hit"], live["spread_lambda"],
                               live["width_mu"], live["step"], live["ppl"], 1.0,
                               boots=live["boots"], completed=live["completed"],
                               n_steps=live["n_steps"], sampling=sampling, b_dict=b_dict,
                               kcal_band=kcal_band)

    # ---- top-line state + the projection block for the Approval-request issue ----
    if live_out is not None:
        proj_block = (
            f"PROJECTION (consolidated; decision input ONLY, does NOT authorize a launch): "
            f"land #71 E[T]={live['E_T']:.3f} (lam={live_out['decision_geometry']['spread_lambda']}, "
            f"mu={live_out['decision_geometry']['width_mu']}) -> official "
            f"{live_out['proj_tps']:.1f} TPS, CI{args.n_steps and '99'} "
            f"[{live_out['ci_lo']:.1f}, {live_out['ci_hi']:.1f}], "
            f"P(clear 500) = {live_out['p_clear_500']*100:.1f}%, validity = {live_out['validity_gate']}, "
            f"binding leg = {live_out['binding_leg']} -> {live_out['verdict']} / {live_out['go_no_go']}.")
        gate_state, gate_verdict, gate_go = "LIVE", live_out["verdict"], live_out["go_no_go"]
    else:
        proj_block = (
            "PROJECTION-CI CONSOLIDATOR ARMED + VALIDATED "
            f"(self-test {'PASS' if st['passes'] else 'FAIL'}); awaiting land #71's measured tuple "
            "(E_T, branch_hit, spread_lambda, width_mu, step, ppl, boots, completed). The instant "
            "his readout lands, ONE call emits the human-readable GO/NO-GO with its CI, its "
            "P(clear 500), its named binding leg, and its validity stamp -- the verbatim "
            "projection block of the `Approval request: HF job` issue.")
        gate_state, gate_verdict, gate_go = "PENDING", "ARMED", "PENDING"

    out = {
        "primary_metric_name": "consolidator_self_test_passes",
        "consolidator_self_test_passes": consolidator_self_test_passes,
        "test_metric_name": "p_clear_500_at_oracle",
        "p_clear_500_at_oracle": p_clear_500_at_oracle,
        "gate_state": gate_state,
        "gate_verdict": gate_verdict,
        "gate_go_no_go": gate_go,
        "land_measured_pending": land_pending,
        "projection_block": proj_block,
        "tau_floor_fix_155": {
            "fern142_tau_low_after": TAU_GATE["low"],
            "fern142_tau_low_before_splitk": 0.9983,
            "ubel148_tree_class_floor": kcal_band["tau_tree_floor"],
            "aligned": bool(tau_aligned),
            "central_clear500_bar_et": accept_length_for_official(TARGET_OFFICIAL, step, 1.0),
            "conservative_clear500_bar_et_before": accept_length_for_official(TARGET_OFFICIAL, step, 0.9983),
            "conservative_clear500_bar_et_after": accept_length_for_official(TARGET_OFFICIAL, step, TAU_GATE["low"]),
        },
        "self_test": {k: v for k, v in st.items() if k != "_anchor_objs"},
        "uncertainty_model": {
            "quadrature_formula": "combined = sqrt(sampling^2 + calibration^2 + step_anchor^2)",
            "sampling_source": "wirbel #146 bootstrap E[T] CI / descending-regime sigma (tau-invariant)",
            "calibration_source": "ubel #148 one-sided downward band (0.787%; tree-class tau folded in)",
            "step_anchor_source": "lawine #136/#147 measured-step residual (0.5% half-width)",
            "decision_geometry_source": "fern #149 joint (spread x width) frontier green membership",
            "validity_source": "denken #150 contract PPL<=2.42 & boots & 128/128 (inline)",
            "p_clear_500_combination": "min(p_ci_direct, p_geom_membership) (conservative union)",
        },
        "legs_imported": {
            "fern_142_gate": "measured_m16_to_official / official_tps_map (corrected tau-low floor)",
            "wirbel_146_envelope": "analyse_sample / DescendingLadder / per-step sigma",
            "ubel_148_calibration": kcal_band["band"],
            "fern_149_frontier": "joint_tps / joint_frontier_surface / green_area_fraction",
            "denken_150_validity": "PPL<=2.42 & boots & 128/128 (inline contract)",
        },
        "land_tuple_spec": LAND_TUPLE_SPEC,
        "live_consolidation": live_out,
        "provenance": ("consolidates fern #142 (point gate + #155 tau-low fix), wirbel #146 "
                       "(sampling CI / required-N), ubel #148 (K_cal tree-transfer band), fern #149 "
                       "(joint spread x width frontier), denken #150 (validity). One source of truth "
                       "per constant -- imports, does not re-derive. K_cal=125.268; clear-500 bar "
                       "E[T]=4.862 (measured step 1.2182)."),
        "method": ("LOCAL CPU-only analytic synthesis + a #142 tau-floor bug-fix; no GPU/vLLM/HF "
                   "Job/submission/kernel build. Produces the Approval-request PROJECTION BLOCK only; "
                   "does NOT authorize a launch. Greedy identity untouched."),
        "metrics_nan_clean": 1,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=_jd)

    print(f"[STATE] {gate_state} / {gate_verdict} / {gate_go}", flush=True)
    print(f"\n{proj_block}\n", flush=True)
    print(f"wrote {args.out}", flush=True)

    if args.wandb and not args.no_wandb:
        try:
            _log_wandb(args, out, st, kcal_band, sampling, live_out)
        except Exception as e:  # noqa: BLE001
            print(f"[consolidator] W&B logging failed (non-fatal): {e!r}", flush=True)


def _log_wandb(args, out, st, kcal_band, sampling, live_out):
    import wandb
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     name=args.wandb_name, group=args.wandb_group, job_type="analysis",
                     config={"consolidator": "approval-projection-consolidator",
                             "method": "cpu-analytic-union-142-146-148-149-150",
                             "K_cal": K_CAL, "measured_step": args.measured_step,
                             "step_rel_half_width": args.step_rel_half_width,
                             "n_steps": args.n_steps, "n_boot": args.n_boot,
                             "tau_central": 1.0, "fern142_tau_low_fixed": TAU_GATE["low"],
                             "fern142_tau_low_before": 0.9983,
                             "ubel148_tree_floor": kcal_band["tau_tree_floor"],
                             "sigma_descend": sampling.sigma_descend,
                             "calib_downside_pct": kcal_band["calib_downside_pct"],
                             "target_official": TARGET_OFFICIAL})
    s = wandb.summary
    s["consolidator_self_test_passes"] = out["consolidator_self_test_passes"]
    s["p_clear_500_at_oracle"] = out["p_clear_500_at_oracle"]
    s["self_test_anchors_ok"] = int(st["anchors_ok"])
    s["self_test_bitmatch_ok"] = int(st["bitmatch_ok"])
    s["gate_state"] = out["gate_state"]
    s["gate_verdict"] = out["gate_verdict"]
    s["land_measured_pending"] = int(out["land_measured_pending"])
    s["tau_floor_aligned_148"] = int(out["tau_floor_fix_155"]["aligned"])
    s["conservative_bar_et_before"] = out["tau_floor_fix_155"]["conservative_clear500_bar_et_before"]
    s["conservative_bar_et_after"] = out["tau_floor_fix_155"]["conservative_clear500_bar_et_after"]
    s["green_region_area_fraction"] = st["bitmatch"]["fern_149_green_area"]["green_region_area_fraction"]
    s["kcal_lo"] = kcal_band["K_cal_lo"]
    s["kcal_central"] = kcal_band["K_cal_central"]
    s["projection_block"] = out["projection_block"]
    # anchor table
    at = wandb.Table(columns=["anchor", "E_T", "lambda", "mu", "proj_tps", "ci_lo", "ci_hi",
                              "p_clear_500", "verdict", "go_no_go", "binding_leg", "ok"])
    for key, a in st["anchors"].items():
        at.add_data(key, a["E_T"], a["lambda"], a["mu"], a["proj_tps"], a["ci"][0], a["ci"][1],
                    a["p_clear_500"], a["verdict"], a["go_no_go"], a["binding_leg"], int(a["ok"]))
    wandb.log({"self_test_anchors": at})
    # bit-match table
    bt = wandb.Table(columns=["leg", "reproduced_lo", "reproduced_hi", "published_lo", "published_hi",
                              "verdict", "ok"])
    for tag in ("red", "green", "border"):
        m = st["bitmatch"]["wirbel_146_sampling"]["anchors"][tag]
        bt.add_data(f"146_{tag}", m["tps_ci99_lo_reproduced"], m["tps_ci99_hi_reproduced"],
                    m["tps_ci99_lo_published_prefix"], m["tps_ci99_hi_published"],
                    m["robust_verdict_99"], int(m["hi_bitmatch_exact"] and m["lo_within_tau_correction_1pct"]))
    wandb.log({"bitmatch_146": bt})
    if live_out is not None:
        s["live_proj_tps"] = live_out["proj_tps"]
        s["live_ci_lo"] = live_out["ci_lo"]
        s["live_ci_hi"] = live_out["ci_hi"]
        s["live_p_clear_500"] = live_out["p_clear_500"]
        s["live_validity_gate"] = live_out["validity_gate"]
        s["live_binding_leg"] = live_out["binding_leg"]
        s["live_go_no_go"] = live_out["go_no_go"]
    print(f"\nW&B run: {run.id}  ({run.url})", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
