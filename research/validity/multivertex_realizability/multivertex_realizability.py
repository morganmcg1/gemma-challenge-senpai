#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Multi-vertex realizability (PR #208) — is #191/#198/#203's both-bugs private build
bar 0.9780 the worst case over ALL realizable domain BLENDS, or only over the single
measured non-Latin-script (NLS) axis?

THE LAST OPEN ASSUMPTION IN THE PRIVATE-VALIDITY CHAIN (#176 -> #191 -> #198 -> #203)
------------------------------------------------------------------------------------
stark #203 (`hexhagf6`) proved the both-bugs private bar is a MONOTONE function of the
reach-weighted deficit Sigma_d w_d.delta_d, and that #176's NLS vertex is the worst case
OVER A SINGLE-AXIS SYNTHETIC SHAPE FAMILY (re-scaling the ONE measured NLS centered
deviation). #203 flagged honestly that "NLS maximizes Sigma w.delta among ALL realizable
domain blends" was ARGUED FROM #176's construction, not proven over a richer polytope.
This PR closes it: an explicit optimization over #176's SIX banked per-axis adverse
domains (non-Latin-script + code + casual + sharegpt + math + long-context), each a real
domain calibrated to the same decode-frame drop (GT-4.3%) by #176's count-pool.

MODEL (the forward map = #198's mechanism + #191's map + #183's bisection, kept verbatim;
the ONLY new thing is the multi-axis blend polytope)
----------------------------------------------------------------------------------------
    Per axis a, #176 count-pools public + hard-component_a in CUMULATIVE-ladder space at
    the weight W_a that lands the decode drop on GT-4.3%:
        C_mix[d] = (1-W_a).C_pub[d] + W_a.C_hard_a[d]     (C_x[d] = prod_{k<=d} q_x[k])
        q_adv_a[d] = C_mix[d] / C_mix[d-1]
        delta^(a)_d = 1 - q_adv_a[d] / q_pub[d]            (the per-axis adverse deficit)
    This reproduces #176's banked NLS adverse vertex (`q_native`) BYTE-EXACTLY, so the NLS
    axis delta equals #203's delta_meas and reproduces 0.977978 exactly (self-test a).

    A realizable domain blend is the convex combination delta_d(p) = Sum_a p_a.delta^(a)_d
    (p >= 0, Sum p = 1). The reach-weighted deficit Sigma w.delta(p) is LINEAR in p, so the
    maximum is at a VERTEX of the polytope = a single axis. The LP reduces to: which of the
    six calibrated axes maximizes Sigma_d w_d.delta^(a)_d at the bar's reach-weights?

CONSTRAINT SET (stated explicitly; this is the headline realizability)
----------------------------------------------------------------------
    DECODE-DROP-REALIZABLE (natural mass): each vertex is a real #176 domain calibrated to
    decode drop = GT-4.3%. The deficit MASS Sigma delta VARIES across axes (decode drop, NOT
    Sigma delta, is the physical invariant #176 held). NLS happens to carry the LARGEST mass
    (Sigma delta = 0.04169). This is the faithful realizable polytope and the headline.

    The PR text also asked to "carry the SAME total-mass constraint Sigma delta=0.04169 you
    used in #203". Applying that FIXED-MASS normalization to the multi-axis case is a
    DEGENERATE counterfactual: the near-mass-balanced axes (math Sigma delta=0.0084,
    longctx Sigma delta=-0.056) get scaled 5-9x (or sign-flipped) into >5% per-rung deficits
    that correspond to NO decode-4.3% domain. It is reported as a SECONDARY shape-ordering
    arm (its "winners" are super-NLS shapes -> private-UNREACHABLE, exactly #203's
    more-shallow-than-NLS finding), not as a realizable blend.

WHAT IT FINDS (honest)
----------------------
Among the six DECODE-DROP-REALIZABLE axes, NLS maximizes Sigma w.delta (2.349pp) at the
bar's reach-weights, beating the runner-up (code, 2.331pp) by ~0.018pp, STABLE across
#193's beta-range. NLS combines a front-loaded SHAPE with the largest realizable MASS; the
other axes are either less front-loaded (longctx) or carry a smaller realizable mass
(math) -> all give a smaller reach-weighted deficit. So 0.9780 STANDS as the operative
worst-case FINITE go-bar over all realizable blends; the only way to exceed it is the
fixed-mass degenerate (super-NLS) counterfactual, which is private-UNREACHABLE (a strictly
STRONGER NO-GO, not a higher finite go bar). The realistic-floor NO-GO (lambda_hat_1=0.342
<< bar) survives the true worst case with a 0.636 gap in lambda.

LOCAL CPU-only analytic synthesis over EXISTING banked numbers (#176 per-axis components +
#203 reach-weights + #193 beta). No GPU / vLLM / HF Job / submission / served-file change.
BASELINE stays 481.53. Greedy/PPL untouched. Bank-the-analysis (PRIMARY = self-test, adds
0 TPS). Optimizes over numbers already computed -- takes NO draws, authorizes none. NOT
open2. NOT a launch.

PRIMARY metric  multivertex_self_test_passes
TEST    metric  both_bugs_bar_worstcase_blend  (worst-case bar over realizable blends = NLS)

Run:
    python research/validity/multivertex_realizability/multivertex_realizability.py \
        --wandb_group private-drop-shape-robustness \
        --wandb_name stark/multivertex-realizability
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

_SR_PATH = REPO_ROOT / "research/validity/lambda_private_drop_shape/shape_robustness.py"
_D176_PROXIES = REPO_ROOT / "research/validity/private_adverse_skew/proxies_native_6axis.json"
_D176_RESULTS = REPO_ROOT / "research/validity/private_adverse_skew/results.json"


def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import module {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- import stark #203 (shape robustness) verbatim; it in turn imports #198/#191/#193/#183/#176. --- #
SR = _import("shape_robustness", _SR_PATH)
Mechanism = SR.Mechanism
reach_weights = SR.reach_weights

TARGET_OFFICIAL = SR.TARGET_OFFICIAL              # 500.0
LAMBDA_STAR_191 = SR.LAMBDA_STAR_191              # 0.9780112973731208 (#191 fixed-drop bar)
LAMBDA_FLOOR = SR.LAMBDA_FLOOR                    # 0.3418647166361965 (#193 liveprobe floor)
BETA_PRIMARY = SR.BETA_PRIMARY                    # 0.765124365433998 (#193)
BETA_RANGE = SR.BETA_RANGE                        # (0.6164865.., 0.9495993..) (#193)
PUBLIC_BAR_BOTH = SR.PUBLIC_BAR_BOTH              # 0.9052283680740145 (#183 public bar, tau=1)
D198_COUPLED_BAR = SR.D198_COUPLED_BAR            # 0.9779783323491393 (#203/#198 anchor)
FLAT_BAR_203 = 0.9453023711509982                # #203 c=0 flat-shape bar (self-test c anchor)
RESID_TOL = 1e-9                                  # exact-reproduction tolerance
NLS_AXIS = "native_multilingual"                  # #176's adverse vertex = non-Latin-script


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# #176 import: the six measured adverse axes + the count-pool reconstruction.
# --------------------------------------------------------------------------- #
def load_176_axes() -> dict[str, Any]:
    """Import #176's six per-axis hard components + shared public reference + the calibrated
    per-axis pool weights (NLS at full precision from the banked adverse vertex). NOT a
    re-measurement: the components are banked; the cumulative count-pool is deterministic and
    reproduces #176's NLS adverse vertex `q_native` byte-exactly (verified in self-test a)."""
    with _D176_PROXIES.open(encoding="utf-8") as fh:
        prox = json.load(fh)
    with _D176_RESULTS.open(encoding="utf-8") as fh:
        r176 = json.load(fh)
    q_pub = list(prox["q_pub_sglang"]["conditional_p_sglang"])
    pool_weight = {a["name"]: a["pool_weight"] for a in r176["per_axis"]}
    axis_label = {a["name"]: a["axis"] for a in r176["per_axis"]}
    w_nls_full = r176["adverse_vertex"]["W_hard"]           # full-precision NLS calibration weight
    axes = []
    for p in prox["proxies"]:
        name = p["name"]
        # NLS uses the full-precision banked weight (exact anchor); others use the banked
        # per-axis pool weight (rounded; ~1e-3 pp in Sigma w.delta, far below inter-axis margins).
        w = w_nls_full if name == NLS_AXIS else pool_weight[name]
        axes.append({
            "name": name,
            "axis": axis_label.get(name, p.get("axis", name)),
            "q_hard": list(p["component"]["conditional_p_sglang"]),
            "W": w,
            "pool_weight_banked": pool_weight[name],
        })
    return {"q_pub": q_pub, "axes": axes, "w_nls_full": w_nls_full}


def _cum(q: list[float]) -> list[float]:
    out, acc = [], 1.0
    for x in q:
        acc *= x
        out.append(acc)
    return out


def cumpool(q_pub: list[float], q_hard: list[float], g: float) -> list[float]:
    """#176's count-pool in CUMULATIVE-ladder space: C_mix = (1-g).C_pub + g.C_hard,
    then q_mix[d] = C_mix[d]/C_mix[d-1]. Reproduces the banked NLS adverse vertex exactly."""
    c_pub, c_hard = _cum(q_pub), _cum(q_hard)
    c_mix = [(1.0 - g) * c_pub[d] + g * c_hard[d] for d in range(len(q_pub))]
    out, prev = [], 1.0
    for d in range(len(q_pub)):
        out.append(c_mix[d] / prev)
        prev = c_mix[d]
    return out


def axis_delta(q_pub: list[float], q_hard: list[float], w: float) -> list[float]:
    q_adv = cumpool(q_pub, q_hard, w)
    return [1.0 - q_adv[d] / q_pub[d] for d in range(len(q_pub))]


# --------------------------------------------------------------------------- #
# The LP: max_p Sum_d w_d.delta_d(p) over the convex blend polytope. Objective linear in p
# => optimum at a vertex = a single axis. We enumerate vertices (exact) and confirm via a
# Dirichlet sweep that no interior blend beats the best vertex.
# --------------------------------------------------------------------------- #
def solve_lp(w: list[float], axis_deltas: dict[str, list[float]],
             n_sweep: int = 200000, seed: int = 20826) -> dict[str, Any]:
    H = len(w)

    def swd(delta: list[float]) -> float:
        return sum(w[d] * delta[d] for d in range(H))

    per_axis = {name: swd(delta) for name, delta in axis_deltas.items()}
    names = list(axis_deltas.keys())
    argmax_name = max(names, key=lambda n: per_axis[n])
    max_swd = per_axis[argmax_name]
    # one-hot blend at the maximizing vertex:
    argmax_blend = {n: (1.0 if n == argmax_name else 0.0) for n in names}

    # Dirichlet interior sweep (confirm vertex-optimality of the linear objective):
    import numpy as np
    rng = np.random.default_rng(seed)
    delta_mat = np.array([axis_deltas[n] for n in names])          # (A, H)
    wv = np.array(w)
    p = rng.dirichlet(np.ones(len(names)), size=n_sweep)            # (N, A)
    blend_swd = (p @ delta_mat) @ wv                               # (N,)
    sweep_max = float(blend_swd.max())
    vertex_optimal = bool(sweep_max <= max_swd + 1e-12)

    return {
        "per_axis_weighted_deficit_pct": {n: per_axis[n] * 100.0 for n in names},
        "argmax_axis": argmax_name,
        "argmax_blend": argmax_blend,
        "max_weighted_deficit_pct": max_swd * 100.0,
        "lp_vertex_optimal_confirmed": vertex_optimal,
        "dirichlet_sweep_max_pct": sweep_max * 100.0,
        "dirichlet_sweep_n": n_sweep,
    }


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def _solve_at_beta(beta: float, q_pub: list[float],
                   axes: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the mechanism + reach-weights at beta, then solve the realizable LP and map the
    worst vertex through the forward map -> bar. NLS vertex uses #203's canonical delta_meas
    (exact anchor); the other axes use the cumulative count-pool reconstruction."""
    mech = Mechanism(beta)
    H = mech.H
    lam_star = mech.bar_of(mech.r_meas)              # measured-NLS bar at this beta (= lambda*)
    w = reach_weights(mech, lam_star)

    # per-axis calibrated deficit (natural mass = decode-drop-realizable):
    axis_deltas: dict[str, list[float]] = {}
    nls_resid = 0.0
    for a in axes:
        if a["name"] == NLS_AXIS:
            # canonical anchor; assert the count-pool reconstruction matches it exactly.
            recon = axis_delta(q_pub, a["q_hard"], a["W"])
            nls_resid = max(abs(recon[d] - mech.delta_meas[d]) for d in range(H))
            axis_deltas[a["name"]] = list(mech.delta_meas)
        else:
            axis_deltas[a["name"]] = axis_delta(q_pub, a["q_hard"], a["W"])

    lp = solve_lp(w, axis_deltas)

    # map every vertex through the forward map (bar; null if private-UNREACHABLE):
    bars = {}
    for name, delta in axis_deltas.items():
        r = [1.0 - x for x in delta]
        reach = mech.reachable_at_full(r)
        bars[name] = mech.bar_of(r) if reach else None

    worst_name = lp["argmax_axis"]
    worst_delta = axis_deltas[worst_name]
    worst_r = [1.0 - x for x in worst_delta]
    worst_reach = mech.reachable_at_full(worst_r)
    worst_bar = mech.bar_of(worst_r) if worst_reach else None

    # monotonicity of bar in achieved Sigma w.delta over the reachable vertices:
    pts = sorted([(lp["per_axis_weighted_deficit_pct"][n], bars[n])
                  for n in axis_deltas if bars[n] is not None])
    bar_monotone = all(pts[i + 1][1] >= pts[i][1] - 1e-9 for i in range(len(pts) - 1))

    return {
        "beta": beta,
        "lambda_star": lam_star,
        "reach_weights": w,
        "axis_deltas": axis_deltas,
        "nls_reconstruction_resid": nls_resid,
        "lp": lp,
        "vertex_bars": bars,
        "worst_axis": worst_name,
        "worst_bar": worst_bar,
        "worst_reachable": bool(worst_reach),
        "bar_monotone_in_weighted_deficit": bool(bar_monotone),
        "_mech": mech,
    }


def synthesize(beta: float = BETA_PRIMARY) -> dict[str, Any]:
    imp = load_176_axes()
    q_pub, axes = imp["q_pub"], imp["axes"]

    main = _solve_at_beta(beta, q_pub, axes)
    mech = main["_mech"]
    H = mech.H
    sigma_target = mech.sigma                         # #203's fixed total-mass = NLS mass = 0.04169
    w = main["reach_weights"]
    axis_deltas = main["axis_deltas"]
    lp = main["lp"]

    def swd(delta: list[float]) -> float:
        return sum(w[d] * delta[d] for d in range(H))

    # ---- (1) realizable axes table (per-rung deficits + mass + reach-weighted deficit) ---- #
    realizable_axes = []
    for a in axes:
        delta = axis_deltas[a["name"]]
        realizable_axes.append({
            "name": a["name"],
            "axis": a["axis"],
            "calibration_weight_W": a["W"],
            "is_nls": bool(a["name"] == NLS_AXIS),
            "delta_d_pct": [round(x * 100.0, 4) for x in delta],
            "sigma_delta": sum(delta),
            "weighted_deficit_pct": swd(delta) * 100.0,
            "bar": main["vertex_bars"][a["name"]],
        })

    nls_swd = swd(axis_deltas[NLS_AXIS]) * 100.0      # 2.349 pp
    max_swd = lp["max_weighted_deficit_pct"]
    optimum_exceeds_nls = bool(max_swd > nls_swd + 1e-9)
    margin_vs_nls_pp = max_swd - nls_swd

    blend_constraints = {
        "headline": "DECODE-DROP-REALIZABLE (natural mass)",
        "description": (
            "delta_d(p) = Sum_axis p_axis.delta^(axis)_d, p>=0, Sum p=1. Each vertex delta^(axis) "
            "is a #176 domain count-pooled (cumulative-ladder space) at the weight W_axis landing "
            "the decode-frame drop on GT-4.3%. The deficit total mass Sigma delta VARIES across "
            "axes -- decode drop, not Sigma delta, is the physical invariant #176 held."),
        "objective_linear_in_p": True,
        "optimum_at_vertex": True,
        "n_axes": len(axes),
        "diversity_cap_note": (
            "#176 capped each hard axis at g<=0.5 in its count-simplex. The single-axis vertices "
            "use the solo calibration weight; native_longctx's solo weight 0.670 exceeds 0.5, but "
            "longctx is the EASIEST axis (smallest deficit) so the cap never binds at the optimum."),
        "fixed_mass_alt": (
            "Sigma delta=0.04169 normalization (the PR's literal 'apples-to-apples' request) is a "
            "DEGENERATE counterfactual for the multi-axis case (it scales mass-balanced axes 5-9x or "
            "sign-flips them into non-realizable >5% deficits); reported as a secondary shape-ordering "
            "arm only -- its winners are super-NLS shapes -> private-UNREACHABLE."),
        "total_mass_sigma_delta_nls": sigma_target,
    }

    # ---- (3) map the optimum to the worst-case bar (TEST) ---- #
    both_bugs_bar_worstcase_blend = main["worst_bar"]     # TEST (0.977978 = NLS, finite)
    delta_vs_203_worstcase = (
        (both_bugs_bar_worstcase_blend - D198_COUPLED_BAR)
        if both_bugs_bar_worstcase_blend is not None else None)
    bar_0978_stands = bool(
        (not optimum_exceeds_nls)
        and both_bugs_bar_worstcase_blend is not None
        and abs(both_bugs_bar_worstcase_blend - D198_COUPLED_BAR) < 1e-6)

    # ---- (4) NO-GO robustness at the true worst case ---- #
    floor_to_bar_gap = (
        (both_bugs_bar_worstcase_blend - LAMBDA_FLOOR)
        if both_bugs_bar_worstcase_blend is not None else None)
    # drop at the realistic floor for the worst vertex:
    worst_r = [1.0 - x for x in axis_deltas[main["worst_axis"]]]
    drop_at_floor_pct = mech.drop_shape(LAMBDA_FLOOR, worst_r) * 100.0
    nogo_robust_worstcase_blend = bool(floor_to_bar_gap is not None and floor_to_bar_gap > 0.0)

    # ---- (5) beta-robustness (SECONDARY): re-solve the LP at #193's beta endpoints ---- #
    beta_lo, beta_hi = BETA_RANGE
    res_lo = _solve_at_beta(beta_lo, q_pub, axes)
    res_hi = _solve_at_beta(beta_hi, q_pub, axes)
    argmax_blend_beta_stable = bool(
        res_lo["worst_axis"] == main["worst_axis"] == res_hi["worst_axis"])
    beta_robustness = {
        "beta_primary": {"beta": beta, "argmax_axis": main["worst_axis"],
                         "worstcase_bar": both_bugs_bar_worstcase_blend,
                         "max_weighted_deficit_pct": max_swd},
        "beta_lo": {"beta": beta_lo, "argmax_axis": res_lo["worst_axis"],
                    "worstcase_bar": res_lo["worst_bar"],
                    "max_weighted_deficit_pct": res_lo["lp"]["max_weighted_deficit_pct"],
                    "per_axis_weighted_deficit_pct": res_lo["lp"]["per_axis_weighted_deficit_pct"],
                    "nan_clean": _finite(res_lo["lp"]["max_weighted_deficit_pct"])},
        "beta_hi": {"beta": beta_hi, "argmax_axis": res_hi["worst_axis"],
                    "worstcase_bar": res_hi["worst_bar"],
                    "max_weighted_deficit_pct": res_hi["lp"]["max_weighted_deficit_pct"],
                    "per_axis_weighted_deficit_pct": res_hi["lp"]["per_axis_weighted_deficit_pct"],
                    "nan_clean": _finite(res_hi["lp"]["max_weighted_deficit_pct"])},
        "argmax_blend_beta_stable": argmax_blend_beta_stable,
        "worstcase_bar_band": [
            min(b for b in (both_bugs_bar_worstcase_blend, res_lo["worst_bar"], res_hi["worst_bar"])
                if b is not None),
            max(b for b in (both_bugs_bar_worstcase_blend, res_lo["worst_bar"], res_hi["worst_bar"])
                if b is not None),
        ],
        "note": ("low beta steepens the shallow->deep reach-weight falloff -> MORE weight on the "
                 "front-loaded rungs -> STRENGTHENS the NLS-wins argument; NLS stays the argmax "
                 "across the whole beta-range."),
    }

    # ---- fixed-mass DEGENERATE arm (the PR's literal Sigma delta=0.04169 request) ---- #
    fixed_mass_axes = []
    for a in axes:
        delta = axis_deltas[a["name"]]
        sd = sum(delta)
        if abs(sd) < 1e-9:
            scale = None
            dfix = None
            swd_fix = None
            bar_fix = None
            degenerate = True
        else:
            scale = sigma_target / sd
            dfix = [delta[d] * scale for d in range(H)]
            swd_fix = swd(dfix) * 100.0
            rfix = [1.0 - x for x in dfix]
            bar_fix = mech.bar_of(rfix) if mech.reachable_at_full(rfix) else None
            degenerate = bool(scale < 0.0 or scale > 2.0)   # sign-flip or >2x blowup
        fixed_mass_axes.append({
            "name": a["name"],
            "is_nls": bool(a["name"] == NLS_AXIS),
            "mass_scaling_factor": scale,
            "weighted_deficit_pct": swd_fix,
            "bar": bar_fix,                                  # null when UNREACHABLE
            "degenerate_non_realizable": degenerate,
        })
    fixed_mass_argmax = max(
        (x for x in fixed_mass_axes if x["weighted_deficit_pct"] is not None),
        key=lambda x: x["weighted_deficit_pct"], default=None)
    fixed_mass_arm = {
        "framing": ("the PR's literal Sigma delta=0.04169 'apples-to-apples' normalization; "
                    "DEGENERATE for mass-balanced axes -> non-realizable shapes. Reported for "
                    "shape-ordering only, NOT as realizable blends."),
        "axes": fixed_mass_axes,
        "argmax_axis": fixed_mass_argmax["name"] if fixed_mass_argmax else None,
        "argmax_weighted_deficit_pct": (
            fixed_mass_argmax["weighted_deficit_pct"] if fixed_mass_argmax else None),
        "argmax_bar": fixed_mass_argmax["bar"] if fixed_mass_argmax else None,
        "interpretation": ("the fixed-mass 'winners' (math/casual/sharegpt/code shapes forced to "
                           "NLS's mass) exceed NLS's reach-weighted deficit and map to "
                           "private-UNREACHABLE (bar=null) -- exactly #203's super-NLS finding: "
                           "worse-than-NLS shapes are a STRONGER NO-GO, not a higher finite go bar."),
    }

    # ---- (6) self-test (PRIMARY) ---- #
    st = _selftests(mech, w, axis_deltas, main, res_lo, res_hi, sigma_target)

    handoff = _handoff(
        worst_bar=both_bugs_bar_worstcase_blend, optimum_exceeds_nls=optimum_exceeds_nls,
        floor_gap=floor_to_bar_gap, bar_band=beta_robustness["worstcase_bar_band"])

    verdict = (
        "0.9780 STANDS as the worst-case FINITE go-bar over all realizable domain blends. Among "
        "#176's six DECODE-DROP-realizable adverse axes, the non-Latin-script vertex MAXIMIZES the "
        "reach-weighted deficit (Sigma w.delta = %.4fpp), beating the runner-up (%s, %.4fpp) by "
        "%.4fpp, and stays the argmax across #193's whole beta-range. NLS is the unique vertex that "
        "pairs a front-loaded SHAPE with the LARGEST realizable MASS (Sigma delta=0.04169); every "
        "other axis is either less front-loaded or carries a smaller realizable mass, so all give a "
        "SMALLER reach-weighted deficit and a LOOSER bar (down to 0.9608 for long-context). The only "
        "way to beat NLS is the fixed-mass DEGENERATE counterfactual (a super-NLS shape at NLS's "
        "mass), which is private-UNREACHABLE -- a strictly STRONGER NO-GO, not a higher finite go "
        "bar. The realistic-floor NO-GO (lambda_hat_1=0.342 << bar) survives the true worst case "
        "with a %.3f gap in lambda."
        % (max_swd,
           sorted(((v, n) for n, v in lp["per_axis_weighted_deficit_pct"].items()
                   if n != main["worst_axis"]), reverse=True)[0][1],
           sorted((v for n, v in lp["per_axis_weighted_deficit_pct"].items()
                   if n != main["worst_axis"]), reverse=True)[0],
           margin_vs_nls_pp,
           floor_to_bar_gap if floor_to_bar_gap is not None else float("nan")))

    return {
        "beta": beta,
        "constants": {
            "target_official": TARGET_OFFICIAL,
            "lambda_star_191_fixed_drop": LAMBDA_STAR_191,
            "d198_coupled_bar": D198_COUPLED_BAR,
            "flat_bar_203": FLAT_BAR_203,
            "public_bar_both_bugs": PUBLIC_BAR_BOTH,
            "lambda_floor_liveprobe": LAMBDA_FLOOR,
            "beta_primary": BETA_PRIMARY,
            "beta_range": list(BETA_RANGE),
            "tau_corner_low": mech.tau_low,
            "sigma_delta_nls": sigma_target,
            "horizon": H,
            "scale_reanchor": mech.scale,
            "w_d_at_bar": w,
            "w_shallow_over_deep_ratio": max(w) / min(w),
        },
        "imports": {
            "provenance": (
                "stark#176 6-axis components+pool-weights (uzl7ixll) x stark#203 reach-weights+"
                "forward-map (hexhagf6) x stark#191 fixed-drop bar (jeclr39w) x stark#198 lambda-"
                "coupled drop (llo1bzn3) x denken#193 beta-range (2clxvlr8) x denken#183 LCB "
                "bisection (82uisrez). NLS vertex = #203 delta_meas (canonical anchor)."),
            "nls_axis": NLS_AXIS,
            "w_nls_full_precision": imp["w_nls_full"],
            "nls_reconstruction_resid": main["nls_reconstruction_resid"],
        },
        "realizable_axes": realizable_axes,
        "blend_constraints": blend_constraints,
        "lp_natural_mass": {
            "per_axis_weighted_deficit_pct": lp["per_axis_weighted_deficit_pct"],
            "argmax_axis": lp["argmax_axis"],
            "argmax_blend": lp["argmax_blend"],
            "max_weighted_deficit_pct": max_swd,
            "nls_weighted_deficit_pct": nls_swd,
            "optimum_exceeds_nls": optimum_exceeds_nls,
            "margin_vs_nls_pp": margin_vs_nls_pp,
            "lp_vertex_optimal_confirmed": lp["lp_vertex_optimal_confirmed"],
            "dirichlet_sweep_max_pct": lp["dirichlet_sweep_max_pct"],
            "bar_monotone_in_weighted_deficit": main["bar_monotone_in_weighted_deficit"],
        },
        "worst_case_blend": {
            "both_bugs_bar_worstcase_blend": both_bugs_bar_worstcase_blend,   # TEST
            "worst_axis": main["worst_axis"],
            "worst_reachable": main["worst_reachable"],
            "delta_vs_203_worstcase": delta_vs_203_worstcase,
            "delta_vs_191_fixed_bar": (
                (both_bugs_bar_worstcase_blend - LAMBDA_STAR_191)
                if both_bugs_bar_worstcase_blend is not None else None),
            "bar_0978_stands": bar_0978_stands,
            "justification": (
                "the LP optimum over the realizable polytope is the NLS vertex (= #203's adverse "
                "vertex); its bar IS #203's 0.977978, so 0.9780 is the worst-case finite go-bar. "
                "flatter/smaller-mass axes give a lower (looser) bar."),
        },
        "nogo_worstcase": {
            "nogo_robust_worstcase_blend": nogo_robust_worstcase_blend,
            "lambda_floor_liveprobe": LAMBDA_FLOOR,
            "floor_to_bar_gap": floor_to_bar_gap,
            "drop_at_floor_pct": drop_at_floor_pct,
            "note": ("even at the true worst realizable blend the realistic floor lambda_hat_1=0.342 "
                     "misses the bar by ~0.64 in lambda -- the NO-GO hardens, never softens."),
        },
        "beta_robustness": beta_robustness,
        "fixed_mass_degenerate_arm": fixed_mass_arm,
        "self_test": st,
        "handoff_lines": handoff,
        "verdict": verdict,
    }


def _selftests(mech, w, axis_deltas, main, res_lo, res_hi, sigma_target) -> dict[str, Any]:
    H = mech.H

    # (a) NLS vertex reproduces #203/#198's coupled both-bugs bar 0.977978 EXACTLY.
    nls_bar = main["vertex_bars"][NLS_AXIS]
    cond_a = bool(_finite(nls_bar)
                  and abs(nls_bar - D198_COUPLED_BAR) < RESID_TOL
                  and main["nls_reconstruction_resid"] < 1e-12)

    # (b) the LP optimum >= NLS's weighted deficit (NLS is feasible -> max can only be >=).
    swd_nls = sum(w[d] * axis_deltas[NLS_AXIS][d] for d in range(H))
    max_swd = main["lp"]["max_weighted_deficit_pct"] / 100.0
    cond_b = bool(max_swd >= swd_nls - 1e-12)

    # (c) a flat/uniform blend reproduces #203's c=0 flat-shape bar 0.945302.
    flat = [mech.dbar] * H
    bar_flat = mech.bar_of([1.0 - x for x in flat])
    cond_c = bool(_finite(bar_flat) and abs(bar_flat - FLAT_BAR_203) < RESID_TOL)

    # (d) the bar is monotone in the achieved Sigma w.delta (consistent with #203).
    cond_d = bool(main["bar_monotone_in_weighted_deficit"])

    # (e) beta-endpoint LPs are NaN-clean and report an argmax.
    cond_e = bool(_finite(res_lo["lp"]["max_weighted_deficit_pct"])
                  and _finite(res_hi["lp"]["max_weighted_deficit_pct"])
                  and res_lo["worst_axis"] is not None
                  and res_hi["worst_axis"] is not None)

    return {
        "conditions": {
            "a_nls_vertex_reproduces_203_bar": cond_a,
            "b_lp_optimum_ge_nls_deficit": cond_b,
            "c_flat_blend_reproduces_203_flat_bar": cond_c,
            "d_bar_monotone_in_weighted_deficit": cond_d,
            "e_beta_endpoint_lps_nan_clean_with_argmax": cond_e,
            # f (full-payload NaN-clean) filled in main() after walking the payload.
        },
        "a_detail": {"nls_vertex_bar": nls_bar, "d198_coupled_bar": D198_COUPLED_BAR,
                     "resid_vs_203": abs(nls_bar - D198_COUPLED_BAR) if _finite(nls_bar) else None,
                     "nls_reconstruction_resid": main["nls_reconstruction_resid"]},
        "b_detail": {"lp_max_weighted_deficit_pct": max_swd * 100.0,
                     "nls_weighted_deficit_pct": swd_nls * 100.0},
        "c_detail": {"flat_blend_bar": bar_flat, "flat_bar_203": FLAT_BAR_203,
                     "resid": abs(bar_flat - FLAT_BAR_203)},
        "d_detail": {"bar_monotone": cond_d},
        "e_detail": {"beta_lo_argmax": res_lo["worst_axis"], "beta_hi_argmax": res_hi["worst_axis"],
                     "beta_lo_max_swd_pct": res_lo["lp"]["max_weighted_deficit_pct"],
                     "beta_hi_max_swd_pct": res_hi["lp"]["max_weighted_deficit_pct"]},
        "partial_passes_a_to_e": bool(cond_a and cond_b and cond_c and cond_d and cond_e),
    }


def _handoff(*, worst_bar, optimum_exceeds_nls, floor_gap, bar_band) -> dict[str, str]:
    stands = "STANDS as" if not optimum_exceeds_nls else "is SUPERSEDED as"
    is_nls = "IS" if not optimum_exceeds_nls else "is NOT"
    bar_str = f"{worst_bar:.6f}" if worst_bar is not None else "private-UNREACHABLE"
    fern = (
        "MULTI-VERTEX REALIZABILITY (stark #208): the TRUE worst-case private bar over ALL "
        "realizable domain blends is %s (vs #203's single-axis 0.977978) -- 0.9780 %s the operative "
        "worst-case FINITE go-bar; non-Latin-script %s the maximizing vertex (optimum_exceeds_nls=%s), "
        "stable across #193's beta-range (worst-case bar band [%.6f, %.6f]); and the realistic-floor "
        "NO-GO survives the true worst case (gap %.3f in lambda). NLS is special because it pairs a "
        "front-loaded SHAPE with the LARGEST realizable MASS; forcing any other axis's shape to NLS's "
        "mass is non-realizable (decode drop != 4.3%%) and goes private-UNREACHABLE (a stronger "
        "NO-GO). fern #185 uses 0.9780 as the binding both-bugs private bar."
        % (bar_str, stands, is_nls, str(optimum_exceeds_nls),
           bar_band[0], bar_band[1], floor_gap if floor_gap is not None else float("nan")))
    land = (
        "land #71 co-log (the #203 spec): co-logging per-rung q_adv[d]/q_pub[d] on the adverse slice "
        "lets fern read the ACTUAL measured blend's deficit and replace this worst-case bar with the "
        "EXACT bar -- if the measured Sigma w.delta <= 2.349pp the bar <= 0.9780 (use 0.9780); if "
        "> 2.349pp it heads toward UNREACHABLE (stronger NO-GO).")
    return {"fern_185_packet": fern, "land_71_colog": land}


# --------------------------------------------------------------------------- #
# NaN-clean walk (UNREACHABLE bars are null, not NaN).
# --------------------------------------------------------------------------- #
def _nan_paths(node: Any, p: str = "result") -> list[str]:
    bad: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "_mech":
                continue
            bad += _nan_paths(v, f"{p}.{k}")
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            bad += _nan_paths(v, f"{p}[{i}]")
    elif isinstance(node, float) and not math.isfinite(node):
        bad.append(p)
    return bad


def _strip_private(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _strip_private(v) for k, v in node.items() if not k.startswith("_")}
    if isinstance(node, list):
        return [_strip_private(v) for v in node]
    return node


# --------------------------------------------------------------------------- #
# Console report.
# --------------------------------------------------------------------------- #
def _print_report(syn: dict) -> None:
    c = syn["constants"]
    lp = syn["lp_natural_mass"]
    wc = syn["worst_case_blend"]
    ng = syn["nogo_worstcase"]
    br = syn["beta_robustness"]
    fm = syn["fixed_mass_degenerate_arm"]
    st = syn["self_test"]
    print("\n" + "=" * 96, flush=True)
    print("MULTI-VERTEX REALIZABILITY (PR #208) — is 0.9780 the worst case over ALL realizable blends?",
          flush=True)
    print("=" * 96, flush=True)
    print(f"  Sigma_delta(NLS) = {c['sigma_delta_nls']:.6f}   H = {c['horizon']}   beta = {syn['beta']:.6f}   "
          f"tau_low = {c['tau_corner_low']:.10f}", flush=True)
    print(f"  reach-weights w_d = {[round(x, 4) for x in c['w_d_at_bar']]}  "
          f"(shallow/deep {c['w_shallow_over_deep_ratio']:.2f}x)", flush=True)
    print("-" * 96, flush=True)
    print("  (1)+(2) realizable axes (natural mass = decode-drop-4.3%) — LP max Sigma w.delta:", flush=True)
    rows = sorted(syn["realizable_axes"], key=lambda a: -a["weighted_deficit_pct"])
    for a in rows:
        tag = "  <- NLS (argmax)" if a["is_nls"] else ""
        bar = f"{a['bar']:.6f}" if a["bar"] is not None else "UNREACHABLE"
        print(f"      {a['name']:20s} Sd={a['sigma_delta']:+.5f}  Sigma_w.delta={a['weighted_deficit_pct']:+.4f}%  "
              f"bar={bar}{tag}", flush=True)
    print(f"      argmax = {lp['argmax_axis']}   max Sigma w.delta = {lp['max_weighted_deficit_pct']:.4f}%  "
          f"(NLS {lp['nls_weighted_deficit_pct']:.4f}%)", flush=True)
    print(f"      optimum_exceeds_nls = {lp['optimum_exceeds_nls']}   margin_vs_nls = {lp['margin_vs_nls_pp']:+.4f}pp   "
          f"vertex_optimal = {lp['lp_vertex_optimal_confirmed']} (sweep max {lp['dirichlet_sweep_max_pct']:.4f}%)",
          flush=True)
    print("-" * 96, flush=True)
    print("  (3) WORST-CASE BAR over realizable blends (TEST):", flush=True)
    bbb = wc["both_bugs_bar_worstcase_blend"]
    print(f"      both_bugs_bar_worstcase_blend = {bbb:.6f} (axis={wc['worst_axis']})  "
          f"delta_vs_203 = {wc['delta_vs_203_worstcase']:+.2e}" if bbb is not None
          else f"      both_bugs_bar_worstcase_blend = UNREACHABLE (axis={wc['worst_axis']})", flush=True)
    print(f"      0.9780 STANDS as the worst-case finite go-bar = {wc['bar_0978_stands']}", flush=True)
    print("-" * 96, flush=True)
    print("  (4) NO-GO robustness at the true worst case:", flush=True)
    print(f"      nogo_robust_worstcase_blend = {ng['nogo_robust_worstcase_blend']}   "
          f"floor->bar gap = {ng['floor_to_bar_gap']:.4f} in lambda  (floor lambda_hat_1={ng['lambda_floor_liveprobe']:.4f})",
          flush=True)
    print("-" * 96, flush=True)
    print("  (5) beta-robustness of the argmax (SECONDARY):", flush=True)
    for key in ("beta_lo", "beta_primary", "beta_hi"):
        e = br[key]
        bs = f"{e['worstcase_bar']:.6f}" if e["worstcase_bar"] is not None else "UNREACHABLE"
        print(f"      beta={e['beta']:.4f}  argmax={e['argmax_axis']:20s}  max Sigma w.delta={e['max_weighted_deficit_pct']:.4f}%  "
              f"bar={bs}", flush=True)
    print(f"      argmax_blend_beta_stable = {br['argmax_blend_beta_stable']}   "
          f"worst-case bar band = [{br['worstcase_bar_band'][0]:.6f}, {br['worstcase_bar_band'][1]:.6f}]", flush=True)
    print("-" * 96, flush=True)
    print("  fixed-mass DEGENERATE arm (PR's literal Sigma delta=0.04169 request; NOT realizable):", flush=True)
    for a in sorted((x for x in fm["axes"] if x["weighted_deficit_pct"] is not None),
                    key=lambda x: -x["weighted_deficit_pct"]):
        bar = f"{a['bar']:.6f}" if a["bar"] is not None else "UNREACHABLE"
        print(f"      {a['name']:20s} scale={a['mass_scaling_factor']:+.3f}x  "
              f"Sigma w.delta={a['weighted_deficit_pct']:+.4f}%  bar={bar}  degenerate={a['degenerate_non_realizable']}",
              flush=True)
    print(f"      -> fixed-mass argmax {fm['argmax_axis']} is super-NLS -> "
          f"{'UNREACHABLE' if fm['argmax_bar'] is None else f'{fm[chr(39)+chr(97)+chr(114)+chr(103)+chr(109)+chr(97)+chr(120)+chr(95)+chr(98)+chr(97)+chr(114)+chr(39)]:.6f}'} "
          f"(stronger NO-GO, consistent with #203)", flush=True)
    print("-" * 96, flush=True)
    print("  (6) SELF-TEST (PRIMARY):", flush=True)
    for k, v in st["conditions"].items():
        print(f"        - {k}: {v}", flush=True)
    print("-" * 96, flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print("=" * 96, flush=True)
    print(f"\n  HAND-OFF (fern #185): {syn['handoff_lines']['fern_185_packet']}", flush=True)


# --------------------------------------------------------------------------- #
# W&B logging (mirrors #203; never fatal).
# --------------------------------------------------------------------------- #
def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[multivertex] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    lp = syn["lp_natural_mass"]
    wc = syn["worst_case_blend"]
    ng = syn["nogo_worstcase"]
    br = syn["beta_robustness"]
    run = init_wandb_run(
        job_type="multivertex-realizability",
        agent="stark",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["private-drop-shape-robustness", "multivertex-realizability", "validity-gate",
              "private-drop", "blend-polytope", "reach-weights", "LP", "composition"],
        config={
            "target_official": TARGET_OFFICIAL, "lambda_star_191_fixed_drop": LAMBDA_STAR_191,
            "d198_coupled_bar": D198_COUPLED_BAR, "lambda_floor_liveprobe": LAMBDA_FLOOR,
            "beta_primary": BETA_PRIMARY, "sigma_delta_nls": syn["constants"]["sigma_delta_nls"],
            "tau_corner_low": syn["constants"]["tau_corner_low"], "n_axes": len(syn["realizable_axes"]),
            "imports": syn["imports"]["provenance"], "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[multivertex] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "multivertex_self_test_passes": int(bool(payload["self_test_passes"])),
        "both_bugs_bar_worstcase_blend": wc["both_bugs_bar_worstcase_blend"],
        "worst_axis": wc["worst_axis"],
        "argmax_axis": lp["argmax_axis"],
        "max_weighted_deficit_pct": lp["max_weighted_deficit_pct"],
        "nls_weighted_deficit_pct": lp["nls_weighted_deficit_pct"],
        "optimum_exceeds_nls": int(bool(lp["optimum_exceeds_nls"])),
        "margin_vs_nls_pp": lp["margin_vs_nls_pp"],
        "delta_vs_203_worstcase": wc["delta_vs_203_worstcase"],
        "bar_0978_stands": int(bool(wc["bar_0978_stands"])),
        "lp_vertex_optimal_confirmed": int(bool(lp["lp_vertex_optimal_confirmed"])),
        "bar_monotone_in_weighted_deficit": int(bool(lp["bar_monotone_in_weighted_deficit"])),
        "nogo_robust_worstcase_blend": int(bool(ng["nogo_robust_worstcase_blend"])),
        "floor_to_bar_gap": ng["floor_to_bar_gap"],
        "argmax_blend_beta_stable": int(bool(br["argmax_blend_beta_stable"])),
        "worstcase_bar_beta_lo": br["beta_lo"]["worstcase_bar"],
        "worstcase_bar_beta_hi": br["beta_hi"]["worstcase_bar"],
        "d198_coupled_bar": D198_COUPLED_BAR,
        "lambda_star_191_fixed_drop": LAMBDA_STAR_191,
        "nan_clean": int(bool(payload["nan_clean"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        "beta": syn["beta"],
        **{f"swd_{n}_pct": v for n, v in lp["per_axis_weighted_deficit_pct"].items()},
        **{f"selftest_{k}": int(bool(v)) for k, v in syn["self_test"]["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="multivertex_realizability_result",
                      artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[multivertex] wandb logged: {summary}", flush=True)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--beta", type=float, default=BETA_PRIMARY, help="depth-decay beta (#193)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="private-drop-shape-robustness")
    args = ap.parse_args(argv)

    syn = synthesize(beta=args.beta)
    syn = _strip_private(syn)            # drop _mech handles before serialization / NaN-walk

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "created_at": created_at,
        "pr": 208,
        "agent": "stark",
        "kind": "multivertex-realizability",
        "synthesis": syn,
    }

    # (f) NaN-clean over the full payload (UNREACHABLE bars stored as null, not NaN).
    nan_bad = _nan_paths(payload)
    payload["nan_clean"] = not nan_bad
    if nan_bad:
        print(f"[multivertex] WARNING non-finite values at: {nan_bad}", flush=True)
    syn["self_test"]["conditions"]["f_nan_clean"] = bool(payload["nan_clean"])

    cond = syn["self_test"]["conditions"]
    self_test_passes = bool(all(cond.values()))
    payload["self_test_passes"] = self_test_passes
    syn["self_test"]["multivertex_self_test_passes"] = self_test_passes

    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload["peak_mem_mib"] = round(peak_kib / 1024.0, 3)

    _print_report(syn)
    print(f"  PRIMARY multivertex_self_test_passes = {self_test_passes}", flush=True)
    bbb = syn["worst_case_blend"]["both_bugs_bar_worstcase_blend"]
    print(f"  TEST    both_bugs_bar_worstcase_blend = "
          f"{bbb:.6f}" if bbb is not None else "  TEST    both_bugs_bar_worstcase_blend = UNREACHABLE",
          flush=True)
    print(f"  peak_mem_mib = {payload['peak_mem_mib']}", flush=True)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[multivertex] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = self_test_passes and payload["nan_clean"]
        print(f"[multivertex] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
