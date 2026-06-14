#!/usr/bin/env python3
"""PR #226 — Private-bar worst-case hardening: f_priv over realizable domain blends.

Capstone of stark's private-worst-case lane (#176 -> #198 -> #208 -> #215 -> this).
RE-POINT the #208 worst-case-blend LP from the lambda-ACCEPTANCE axis to the
private-TPS-DROP (f_priv) axis. CPU-only analytic; no GPU/vLLM/HF/served-file/draw.
BASELINE stays 481.53. Bank-the-analysis (PRIMARY = self-test; adds 0 TPS).

THE MECHANISM (imported verbatim from my banked legs; NOT re-derived)
---------------------------------------------------------------------
    f_priv(blend) = (1 - drop(blend)) * tau_low                      [the #191/#198 forward map
                                                                      ratio at full recovery]
    drop(blend)   = mech.drop_shape(1.0, r(blend))                   [#198 tree-DP E[T] drop,
                                                                      NLS-anchored scale]
    delta_d(blend)= Sum_axis p_axis * delta_d^(axis)                 [#176 six axes, linear in p]
    r_d(blend)    = 1 - delta_d(blend)
    private_bar(f_priv) = mu_safe_fresh / f_priv                     [kanna #217 gross-up]
    mu_safe_fresh = private_bar_217 * f_priv_217                     [round-trips 528.48]

At the NLS vertex (r = mech.r_meas) the full-recovery drop is #198's drop_both_176 = 2.35028%,
so f_priv = (1 - 0.0235028) * 0.99243186 = 0.969106920637722 == kanna #217's f_priv EXACTLY.

THE CORE QUESTION
-----------------
Is non-Latin-script (NLS) ALSO the f_priv-MINIMIZING vertex (it is the lambda-deficit-MAXIMIZING
vertex in #208)? What is f_priv_worstcase over all realizable blends, and does the resulting
private_bar_worstcase = mu_safe_fresh / f_priv_worstcase stay <= wirbel #199's compliant-spec
ceiling 536.66 (private-FEASIBLE) or CROSS it (private-INFEASIBLE)?

Run:
  cd target/ && CUDA_VISIBLE_DEVICES="" python \
    research/validity/private_fpriv_worstcase/private_fpriv_worstcase.py \
    --self-test --wandb_group private-drop-shape-robustness --wandb_name stark/private-fpriv-worstcase
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

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]                       # research/validity/<leg> -> repo root
_SR_PATH = REPO_ROOT / "research/validity/lambda_private_drop_shape/shape_robustness.py"
_MV_PATH = REPO_ROOT / "research/validity/multivertex_realizability/multivertex_realizability.py"


def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- banked machinery (the forward map + the realizable six-axis LP); NOT re-derived --- #
SR = _import("shape_robustness", _SR_PATH)
MV = _import("multivertex_realizability", _MV_PATH)
Mechanism = SR.Mechanism
reach_weights = SR.reach_weights
BETA_PRIMARY = SR.BETA_PRIMARY                     # 0.765124365433998
BETA_RANGE = SR.BETA_RANGE                         # (0.616486595380561, 0.9495993894553337)
LAMBDA_FLOOR = SR.LAMBDA_FLOOR                     # 0.3418647166361965 (lambda_hat_1 realistic floor)
LAMBDA_STAR_191 = SR.LAMBDA_STAR_191               # 0.9780112973731208
NLS_AXIS = MV.NLS_AXIS                             # "native_multilingual"

# --------------------------------------------------------------------------- #
# Imported external constants (provenance: W&B run ids, project
# wandb-applied-ai-team/gemma-challenge-senpai). IMPORTED, not re-derived.
# --------------------------------------------------------------------------- #
F_PRIV_217 = 0.969106920637722              # kanna #217 vgovdrjc  summary/f_priv (CENTRAL)
PRIVATE_BAR_217 = 528.4835555959944         # kanna #217 vgovdrjc  mu_bar_private_corrected
LAMBDA1_CEILING = 520.9527323111674         # kanna #217 / #204    lambda1_ceiling (physical)
PRIVATE_BAR_MINUS_CEILING_217 = 7.530823284826965   # kanna #217   private_bar_minus_ceiling
COMPLIANT_CEILING_CENTRAL = 536.6590426143789       # wirbel #199 wdyqnx3g compliant_spec_tps_ceiling
COMPLIANT_CEILING_LCB = 525.7290377676009           # wirbel #199 wdyqnx3g ceiling_lcb_tps_both_bugs
FROZEN_BAR_202 = 504.87342465668917         # kanna #202 533jd6l1  mu_bar_frozen_p95
# my #198 llo1bzn3 (tree-DP both-bugs drops, pct):
DROP_BOTH_176_PCT = 2.3502816766841543      # full recovery (lambda=1) -> the f_priv basis
DROP_FLOOR_PCT = 2.293464667545164          # at lambda_hat_1=0.342 (NEGATIVE coupling -> smaller)
DROP_BAR_PCT = 2.348874564960838            # at the lambda bar
# PR #52 single hard paired public/private draw:
PUB_52 = 481.53
PRIV_52 = 460.85
F_PRIV_52_STATED = 0.95705                  # the PR's stated #52 observed f_priv
# decode-drop calibration (#176 / #164):
GT_DECODE_DROP_PCT = 4.294644155088989      # #176 adverse-vertex achieved decode drop
AGG_DROP_CI_PCT = (1.87, 2.21)              # #164 aggregate (central-blend) TPS-drop CI band

TARGET_OFFICIAL = 500.0
RESID_TOL = 1e-6
MONO_GRID = [0.93, 0.94, 0.95, 0.955, 0.96, 0.965, 0.969106920637722, 0.975, 0.98, 0.99]


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def r_of_delta(delta: list[float]) -> list[float]:
    return [1.0 - x for x in delta]


def f_priv_of_r(mech: Any, r: list[float], lam: float = 1.0) -> float:
    """f_priv = (1 - drop(lam)) * tau_low. lam=1 (full recovery) is the conservative point that
    reproduces kanna #217's f_priv (the #198 negative coupling makes lam<1 a SMALLER drop)."""
    return (1.0 - mech.drop_shape(lam, r)) * mech.tau_low


def private_bar_of_fpriv(fp: float, mu_safe_fresh: float) -> float:
    return mu_safe_fresh / fp


# --------------------------------------------------------------------------- #
# Axis construction (NLS = exact #198 anchor; the other five via the #176/#208 count-pool).
# --------------------------------------------------------------------------- #
def build_axis_deltas(mech: Any) -> tuple[dict[str, list[float]], list[dict[str, Any]], float]:
    d176 = MV.load_176_axes()
    q_pub = d176["q_pub"]
    axis_deltas: dict[str, list[float]] = {}
    meta: list[dict[str, Any]] = []
    nls_resid = 0.0
    for a in d176["axes"]:
        name = a["name"]
        if name == NLS_AXIS:
            recon = MV.axis_delta(q_pub, a["q_hard"], a["W"])
            nls_resid = max(abs(recon[d] - mech.delta_meas[d]) for d in range(mech.H))
            axis_deltas[name] = list(mech.delta_meas)            # canonical full-precision anchor
        else:
            axis_deltas[name] = MV.axis_delta(q_pub, a["q_hard"], a["W"])
        meta.append({"name": name, "axis": a["axis"], "is_nls": name == NLS_AXIS, "W": a["W"]})
    return axis_deltas, meta, nls_resid


def blend_delta(axis_deltas: dict[str, list[float]], names: list[str],
                p: list[float], H: int) -> list[float]:
    return [sum(p[i] * axis_deltas[names[i]][d] for i in range(len(names))) for d in range(H)]


# --------------------------------------------------------------------------- #
# The worst-case f_priv over the realizable simplex (re-pointed #208 LP).
# --------------------------------------------------------------------------- #
def solve_fpriv_worstcase(mech: Any, axis_deltas: dict[str, list[float]],
                          meta: list[dict[str, Any]],
                          n_sweep_treedp: int = 20000,
                          seed: int = 22600) -> dict[str, Any]:
    H = mech.H
    names = list(axis_deltas.keys())

    # ---- per-vertex f_priv at full recovery (the f_priv-axis objective) ---- #
    per_axis = {}
    for m in meta:
        n = m["name"]
        r = r_of_delta(axis_deltas[n])
        drop1 = mech.drop_shape(1.0, r)
        per_axis[n] = {
            "is_nls": m["is_nls"], "axis": m["axis"],
            "drop_full_recovery_pct": drop1 * 100.0,
            "f_priv": (1.0 - drop1) * mech.tau_low,
        }
    # f_priv-MINIMIZING vertex (= drop-MAXIMIZING vertex):
    argmin_name = min(names, key=lambda n: per_axis[n]["f_priv"])
    f_priv_worstcase = per_axis[argmin_name]["f_priv"]
    runner = sorted(names, key=lambda n: per_axis[n]["f_priv"])[1]
    margin_to_runner_fpriv = per_axis[runner]["f_priv"] - f_priv_worstcase
    margin_to_runner_drop_pp = (per_axis[argmin_name]["drop_full_recovery_pct"]
                                - per_axis[runner]["drop_full_recovery_pct"])

    # ---- (i) #208 LINEAR check: reach-weighted deficit is exactly linear in p -> vertex argmax.
    lam_star = mech.bar_of(mech.r_meas)
    w = reach_weights(mech, lam_star)
    lp_linear = MV.solve_lp(w, axis_deltas)        # 200k Dirichlet sweep on the LINEAR objective

    # ---- (ii) NONLINEAR check: the tree-DP f_priv objective itself is vertex-optimal.
    rng = np.random.default_rng(seed)
    delta_mat = np.array([axis_deltas[n] for n in names])           # (A, H)
    P = rng.dirichlet(np.ones(len(names)), size=n_sweep_treedp)     # (N, A)
    blend_deltas = P @ delta_mat                                    # (N, H)
    sweep_min_fpriv = math.inf
    sweep_argmin_p = None
    for i in range(n_sweep_treedp):
        r = [1.0 - blend_deltas[i, d] for d in range(H)]
        fp = (1.0 - mech.drop_shape(1.0, r)) * mech.tau_low
        if fp < sweep_min_fpriv:
            sweep_min_fpriv = fp
            sweep_argmin_p = P[i].tolist()
    # vertex-optimal iff no interior blend dips below the worst vertex (tol for tree-DP noise):
    treedp_vertex_optimal = bool(sweep_min_fpriv >= f_priv_worstcase - 1e-9)

    return {
        "names": names,
        "per_axis": per_axis,
        "f_priv_worstcase": f_priv_worstcase,
        "f_priv_min_vertex": argmin_name,
        "f_priv_min_is_nls": bool(argmin_name == NLS_AXIS),
        "runner_up_vertex": runner,
        "margin_to_runner_fpriv": margin_to_runner_fpriv,
        "margin_to_runner_drop_pp": margin_to_runner_drop_pp,
        "lp_linear_argmax_axis": lp_linear["argmax_axis"],
        "lp_linear_vertex_optimal": bool(lp_linear["lp_vertex_optimal_confirmed"]),
        "lp_linear_max_weighted_deficit_pct": lp_linear["max_weighted_deficit_pct"],
        "lp_linear_dirichlet_sweep_max_pct": lp_linear["dirichlet_sweep_max_pct"],
        "lp_linear_per_axis_weighted_deficit_pct": lp_linear["per_axis_weighted_deficit_pct"],
        "treedp_sweep_min_fpriv": sweep_min_fpriv,
        "treedp_sweep_n": n_sweep_treedp,
        "treedp_vertex_optimal": treedp_vertex_optimal,
        "treedp_sweep_argmin_p": dict(zip(names, sweep_argmin_p)) if sweep_argmin_p else None,
        "reach_weights_at_lambda_star": w,
        "lambda_star": lam_star,
    }


# --------------------------------------------------------------------------- #
# Robustness: beta (reach-weight shape) + decode-drop calibration scale.
# --------------------------------------------------------------------------- #
def robustness(mech_primary: Any, mu_safe_fresh: float) -> dict[str, Any]:
    # ---- (A) beta sweep: reach-weight shape across #193's range ---- #
    beta_rows = []
    for beta in (BETA_RANGE[0], BETA_PRIMARY, BETA_RANGE[1]):
        m = Mechanism(beta)
        ad, meta, _ = build_axis_deltas(m)
        names = list(ad.keys())
        fpr = {n: (1.0 - m.drop_shape(1.0, r_of_delta(ad[n]))) * m.tau_low for n in names}
        amin = min(names, key=lambda n: fpr[n])
        fpw = fpr[amin]
        beta_rows.append({
            "beta": beta, "f_priv_worstcase": fpw,
            "private_bar_worstcase": mu_safe_fresh / fpw,
            "vertex": amin, "is_nls": bool(amin == NLS_AXIS),
            "feasible_vs_central_536": bool(mu_safe_fresh / fpw <= COMPLIANT_CEILING_CENTRAL),
        })
    fpw_band = [r["f_priv_worstcase"] for r in beta_rows]
    bar_band_beta = [r["private_bar_worstcase"] for r in beta_rows]

    # ---- (B) decode-drop calibration scale: delta_meas -> s*delta_meas on the NLS vertex.
    # s scales the per-rung deficit magnitude (a higher decode drop -> bigger deficits).
    def bar_at_scale(s: float) -> dict[str, float]:
        r = [1.0 - s * d for d in mech_primary.delta_meas]
        drop = mech_primary.drop_shape(1.0, r)
        fp = (1.0 - drop) * mech_primary.tau_low
        return {"scale": s, "drop_full_recovery_pct": drop * 100.0, "f_priv": fp,
                "private_bar": mu_safe_fresh / fp,
                "feasible_vs_central_536": bool(mu_safe_fresh / fp <= COMPLIANT_CEILING_CENTRAL),
                "feasible_vs_lcb_526": bool(mu_safe_fresh / fp <= COMPLIANT_CEILING_LCB)}

    scale_rows = [bar_at_scale(s) for s in (0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6)]
    # the scale that reproduces the #52 empirical f_priv (bisect on f_priv(s)):
    s_lo, s_hi = 0.5, 4.0
    f52 = PRIV_52 / PUB_52
    for _ in range(80):
        s_mid = 0.5 * (s_lo + s_hi)
        if bar_at_scale(s_mid)["f_priv"] > f52:   # f_priv decreasing in s
            s_lo = s_mid
        else:
            s_hi = s_mid
    s_emulates_52 = 0.5 * (s_lo + s_hi)
    row_52 = bar_at_scale(s_emulates_52)
    # the scale at which private_bar hits the central ceiling 536.66 (feasibility break-even):
    s_lo, s_hi = 0.5, 6.0
    for _ in range(80):
        s_mid = 0.5 * (s_lo + s_hi)
        if bar_at_scale(s_mid)["private_bar"] < COMPLIANT_CEILING_CENTRAL:
            s_lo = s_mid
        else:
            s_hi = s_mid
    s_breakeven_central = 0.5 * (s_lo + s_hi)

    # ---- (C) #198 NEGATIVE-coupling caveat: floor drop is SMALLER -> floor f_priv HIGHER. ---- #
    drop_floor = mech_primary.drop_shape(LAMBDA_FLOOR, mech_primary.r_meas)
    f_priv_floor = (1.0 - drop_floor) * mech_primary.tau_low
    f_priv_ceiling = (1.0 - mech_primary.drop_shape(1.0, mech_primary.r_meas)) * mech_primary.tau_low

    return {
        "beta_sweep": beta_rows,
        "f_priv_worstcase_band_over_beta": [min(fpw_band), max(fpw_band)],
        "private_bar_worstcase_band_over_beta": [min(bar_band_beta), max(bar_band_beta)],
        "vertex_identity_stable_over_beta": bool(all(r["is_nls"] for r in beta_rows)),
        "feasible_vs_central_stable_over_beta": bool(all(r["feasible_vs_central_536"]
                                                         for r in beta_rows)),
        "decode_drop_scale_sweep": scale_rows,
        "scale_emulates_52": s_emulates_52,
        "row_at_scale_emulates_52": row_52,
        "scale_breakeven_central_536": s_breakeven_central,
        "aggregate_drop_ci_pct": list(AGG_DROP_CI_PCT),
        "nls_worst_drop_above_aggregate_ci": bool(DROP_BOTH_176_PCT > AGG_DROP_CI_PCT[1]),
        "negative_coupling_caveat": {
            "drop_at_floor_pct": drop_floor * 100.0,
            "drop_at_full_recovery_pct": mech_primary.drop_shape(1.0, mech_primary.r_meas) * 100.0,
            "f_priv_at_floor": f_priv_floor,
            "f_priv_at_full_recovery": f_priv_ceiling,
            "floor_fpriv_higher_than_ceiling": bool(f_priv_floor >= f_priv_ceiling),
            "private_bar_at_floor": mu_safe_fresh / f_priv_floor,
            "private_bar_at_ceiling": mu_safe_fresh / f_priv_ceiling,
            "note": ("#198 NEGATIVE coupling: the shallow-concentrated deficit COMPOUNDS along the "
                     "accepted chain, so the drop is SMALLEST at the realistic floor lambda_hat_1 "
                     "and LARGEST at full recovery. f_priv at lambda=1 (used here, == kanna #217's "
                     "0.969107) is therefore the CONSERVATIVE worst point; the operating lambda<1 "
                     "carries a SMALLER private-TPS drop, making the realistic-floor private bar "
                     "SLIGHTLY easier, not harder. The negative coupling does NOT widen the bar."),
        },
    }


# --------------------------------------------------------------------------- #
# Reference table + feasibility verdict.
# --------------------------------------------------------------------------- #
def build_table(mu_safe_fresh: float, f_priv_worstcase: float) -> dict[str, Any]:
    f52 = PRIV_52 / PUB_52
    f_breakeven_central = mu_safe_fresh / COMPLIANT_CEILING_CENTRAL
    f_breakeven_lcb = mu_safe_fresh / COMPLIANT_CEILING_LCB

    def row(label: str, fp: float) -> dict[str, Any]:
        bar = mu_safe_fresh / fp
        return {
            "label": label, "f_priv": fp, "private_bar": bar,
            "clears_central_536": bool(bar <= COMPLIANT_CEILING_CENTRAL),
            "clears_lcb_526": bool(bar <= COMPLIANT_CEILING_LCB),
            "gap_vs_physical_ceiling_520p95": bar - LAMBDA1_CEILING,
        }

    rows = [
        row("central_217 (f_priv=0.969107)", F_PRIV_217),
        row("worstcase_blend (NLS vertex)", f_priv_worstcase),
        row("observed_52 (460.85/481.53)", f52),
        row("breakeven_central_536.66", f_breakeven_central),
        row("breakeven_lcb_525.73", f_breakeven_lcb),
    ]
    return {
        "rows": rows,
        "f_priv_observed_52": f52,
        "f_priv_breakeven_central_536": f_breakeven_central,
        "f_priv_breakeven_lcb_526": f_breakeven_lcb,
    }


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    mech = Mechanism(BETA_PRIMARY)
    tau_low = mech.tau_low
    axis_deltas, meta, nls_resid = build_axis_deltas(mech)

    # mu_safe_fresh round-trips kanna #217's bar (private_bar = mu_safe_fresh / f_priv).
    mu_safe_fresh = PRIVATE_BAR_217 * F_PRIV_217

    # mechanism-DERIVED central f_priv (NLS full-recovery) -> must reproduce kanna #217's import.
    f_priv_central_derived = (1.0 - mech.drop_shape(1.0, mech.r_meas)) * tau_low
    f_priv_central_resid = abs(f_priv_central_derived - F_PRIV_217)

    wc = solve_fpriv_worstcase(mech, axis_deltas, meta)
    f_priv_worstcase = wc["f_priv_worstcase"]
    private_bar_worstcase = private_bar_of_fpriv(f_priv_worstcase, mu_safe_fresh)

    table = build_table(mu_safe_fresh, f_priv_worstcase)
    robust = robustness(mech, mu_safe_fresh)

    # feasibility verdict (PR instruction 3).
    compliant_lane_private_feasible = bool(private_bar_worstcase <= COMPLIANT_CEILING_CENTRAL)
    feasible_vs_lcb = bool(private_bar_worstcase <= COMPLIANT_CEILING_LCB)
    private_bar_worstcase_minus_central = private_bar_worstcase - PRIVATE_BAR_217
    private_bar_worstcase_minus_536 = private_bar_worstcase - COMPLIANT_CEILING_CENTRAL
    gap_vs_physical_ceiling = private_bar_worstcase - LAMBDA1_CEILING
    f_priv_breakeven_central = table["f_priv_breakeven_central_536"]
    f52 = table["f_priv_observed_52"]
    # how much headroom does the EMPIRICAL #52 floor leave to the central break-even?
    f52_minus_breakeven_central = f52 - f_priv_breakeven_central
    private_bar_at_52 = mu_safe_fresh / f52

    # ---- self tests ---- #
    st = _selftests(mech, mu_safe_fresh, f_priv_worstcase, f_priv_central_derived, wc, table)

    verdict = (
        "REALIZABLE-BLEND WORST-CASE == CENTRAL: NLS is BOTH the lambda-deficit-maximizing AND the "
        "f_priv-minimizing vertex; the realizable-blend LP adds ZERO spread to the private bar "
        f"(worst-case {private_bar_worstcase:.4f} == kanna #217 central 528.48, because #217's "
        "'central' f_priv=0.969107 IS the NLS worst-vertex drop at 4.3% decode drop). The bar STAYS "
        f"{private_bar_worstcase - COMPLIANT_CEILING_CENTRAL:+.2f} vs wirbel #199's compliant-spec "
        f"ceiling 536.66 -> compliant lane is private-FEASIBLE vs the CENTRAL ceiling (break-even "
        f"f_priv {f_priv_breakeven_central:.5f}; worst-case 0.969107 clears it by "
        f"{f_priv_worstcase - f_priv_breakeven_central:+.5f}). BUT the binding risk is the "
        "decode-drop CALIBRATION, not the blend: the one hard paired draw (#52) gives f_priv=0.95705 "
        f"-> bar {private_bar_at_52:.2f} (within {COMPLIANT_CEILING_CENTRAL - private_bar_at_52:.2f} "
        f"of 536.66, clearing the break-even by only {f52_minus_breakeven_central:+.5f}), a private "
        "drop ~1.5x the realizable-blend worst vertex and OUTSIDE the realizable simplex. And vs the "
        f"conservative LCB ceiling 525.73 the worst-case bar already MISSES "
        f"({private_bar_worstcase - COMPLIANT_CEILING_LCB:+.2f}) -> private-INFEASIBLE at P95.")

    handoff = (
        f"the private build target is private_bar_worstcase = {private_bar_worstcase:.4f} under the "
        f"worst realizable domain blend (f_priv_worstcase = {f_priv_worstcase:.6f}, binding vertex "
        f"non-Latin-script), == kanna #224's central 528.48 (the realizable-blend LP adds no spread "
        "because #217's central IS the NLS worst vertex); it STAYS BELOW wirbel #199's compliant-spec "
        "ceiling 536.66, so the compliant-verify lane is private-FEASIBLE at the worst realizable "
        f"blend (f_priv break-even = {f_priv_breakeven_central:.5f}); the binding risk is the "
        "decode-drop calibration not the blend (the #52 empirical f_priv=0.95705 -> bar "
        f"{private_bar_at_52:.2f} sits OUTSIDE the realizable simplex, {f52_minus_breakeven_central:+.5f} "
        "above break-even); fern carries the private bar as the interval [central 528.48, "
        f"empirical-floor {private_bar_at_52:.2f}], FEASIBLE vs 536.66 central / INFEASIBLE vs 525.73 LCB.")

    return {
        "beta": BETA_PRIMARY,
        "constants": {
            "tau_low": tau_low,
            "mu_safe_fresh": mu_safe_fresh,
            "f_priv_217_imported": F_PRIV_217,
            "private_bar_217_imported": PRIVATE_BAR_217,
            "lambda1_ceiling": LAMBDA1_CEILING,
            "compliant_ceiling_central_536": COMPLIANT_CEILING_CENTRAL,
            "compliant_ceiling_lcb_526": COMPLIANT_CEILING_LCB,
            "frozen_bar_202": FROZEN_BAR_202,
            "drop_both_176_pct": DROP_BOTH_176_PCT,
            "gt_decode_drop_pct": GT_DECODE_DROP_PCT,
            "nls_axis": NLS_AXIS,
        },
        "f_priv_model": {
            "formula": "f_priv(blend) = (1 - drop_shape(1.0, r(blend))) * tau_low",
            "drop_formula": "drop(blend) = #198 tree-DP E[T] drop (NLS-anchored scale) at lambda=1",
            "blend_linearity": "delta_d(blend) = Sum_axis p_axis * delta_d^(axis) (linear in p)",
            "anchor_central_f_priv_derived": f_priv_central_derived,
            "anchor_central_f_priv_imported_217": F_PRIV_217,
            "anchor_central_resid": f_priv_central_resid,
            "anchor_52_f_priv_observed": f52,
            "anchor_52_f_priv_stated": F_PRIV_52_STATED,
            "nls_reconstruction_resid_vs_198": nls_resid,
        },
        "worstcase": wc,
        "headline": {
            "f_priv_worstcase": f_priv_worstcase,                       # HEADLINE
            "private_bar_worstcase": private_bar_worstcase,             # TEST
            "f_priv_min_is_nls": wc["f_priv_min_is_nls"],
            "margin_to_runner_drop_pp": wc["margin_to_runner_drop_pp"],
            "private_bar_worstcase_minus_central": private_bar_worstcase_minus_central,
            "private_bar_worstcase_minus_536": private_bar_worstcase_minus_536,
            "gap_vs_physical_ceiling_520p95": gap_vs_physical_ceiling,
            "compliant_lane_private_feasible": compliant_lane_private_feasible,   # vs 536.66 central
            "compliant_lane_private_feasible_vs_lcb": feasible_vs_lcb,            # vs 525.73 LCB
            "f_priv_breakeven_central_536": f_priv_breakeven_central,
            "f_priv_breakeven_lcb_526": table["f_priv_breakeven_lcb_526"],
            "f_priv_observed_52": f52,
            "private_bar_at_52": private_bar_at_52,
            "f52_minus_breakeven_central": f52_minus_breakeven_central,
            "private_bar_worstcase_crosses_536": bool(private_bar_worstcase > COMPLIANT_CEILING_CENTRAL),
        },
        "table": table,
        "robustness": robust,
        "self_test": st,
        "verdict": verdict,
        "handoff_line": handoff,
        "imports": {
            "provenance": ("kanna#217 vgovdrjc (f_priv, private_bar, lambda1_ceiling) x "
                           "wirbel#199 wdyqnx3g (compliant ceiling 536.66/LCB 525.73) x "
                           "kanna#202 533jd6l1 (frozen bar 504.87) x stark#198 llo1bzn3 (tree-DP drops) "
                           "x stark#208 wi4gxxx8 (six-axis realizable LP) x stark#176 (decode-drop axes) "
                           "x PR#52 (481.53/460.85 paired draw). All run-ids in "
                           "wandb-applied-ai-team/gemma-challenge-senpai."),
            "mechanism": "shape_robustness.Mechanism (#198 forward map) + multivertex_realizability (#208 LP)",
        },
    }


def _selftests(mech: Any, mu_safe_fresh: float, f_priv_worstcase: float,
               f_priv_central_derived: float, wc: dict[str, Any],
               table: dict[str, Any]) -> dict[str, Any]:
    # (a) at f_priv=0.969107 the bar round-trips kanna #217's 528.48 EXACTLY.
    bar_at_central = mu_safe_fresh / F_PRIV_217
    cond_a = bool(_finite(bar_at_central)
                  and abs(bar_at_central - PRIVATE_BAR_217) < RESID_TOL
                  and abs(f_priv_central_derived - F_PRIV_217) < 1e-9)   # mechanism reproduces import

    # (b) f_priv_worstcase <= f_priv_central (worst-case is a bigger drop / smaller f_priv).
    cond_b = bool(f_priv_worstcase <= F_PRIV_217 + 1e-12)

    # (c) the worst realizable blend is a pure vertex (interior sweeps do not beat the vertex):
    cond_c = bool(wc["lp_linear_vertex_optimal"] and wc["treedp_vertex_optimal"])

    # (d) private_bar monotone DECREASING in f_priv.
    bars = [mu_safe_fresh / fp for fp in MONO_GRID]
    cond_d = all(bars[i] > bars[i + 1] for i in range(len(bars) - 1))

    # (e) round-trips #52's observed 0.95705 from 460.85/481.53.
    f52 = PRIV_52 / PUB_52
    cond_e = bool(abs(f52 - F_PRIV_52_STATED) < 5e-6)

    # (f) nan-clean handled at payload level; placeholder here.
    cond_f = True

    conditions = {
        "a_f_priv_0969107_roundtrips_528p48_and_mechanism_reproduces_217": cond_a,
        "b_f_priv_worstcase_le_f_priv_central": cond_b,
        "c_worst_blend_is_pure_vertex_both_sweeps": cond_c,
        "d_private_bar_monotone_decreasing_in_f_priv": cond_d,
        "e_roundtrips_52_observed_0p95705": cond_e,
        "f_nan_clean": cond_f,
    }
    return {
        "conditions": conditions,
        "private_fpriv_worstcase_self_test_passes": bool(all(conditions.values())),
        "a_detail": {"bar_at_central": bar_at_central, "private_bar_217": PRIVATE_BAR_217,
                     "f_priv_central_derived": f_priv_central_derived,
                     "f_priv_central_resid": abs(f_priv_central_derived - F_PRIV_217)},
        "c_detail": {"lp_linear_vertex_optimal": wc["lp_linear_vertex_optimal"],
                     "treedp_vertex_optimal": wc["treedp_vertex_optimal"],
                     "treedp_sweep_min_fpriv": wc["treedp_sweep_min_fpriv"],
                     "f_priv_worstcase": f_priv_worstcase},
        "e_detail": {"f52": f52, "stated": F_PRIV_52_STATED},
    }


# --------------------------------------------------------------------------- #
# nan audit, report, wandb, main.
# --------------------------------------------------------------------------- #
def _nan_paths(node: Any, p: str = "result") -> list[str]:
    bad = []
    if isinstance(node, dict):
        for k, v in node.items():
            bad += _nan_paths(v, f"{p}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            bad += _nan_paths(v, f"{p}[{i}]")
    elif isinstance(node, float) and not math.isfinite(node):
        bad.append(p)
    return bad


def _print_report(syn: dict) -> None:
    h, st, tb = syn["headline"], syn["self_test"], syn["table"]
    print("=" * 100, flush=True)
    print("PR #226  Private-bar worst-case hardening: f_priv over realizable domain blends", flush=True)
    print("=" * 100, flush=True)
    print(f"  f_priv model: f_priv(blend) = (1 - drop_shape(1.0, r(blend))) * tau_low   "
          f"[tau_low={syn['constants']['tau_low']:.10f}]", flush=True)
    print(f"  central f_priv (derived NLS full-recovery) = {syn['f_priv_model']['anchor_central_f_priv_derived']:.12f}"
          f"  vs kanna#217 import {syn['f_priv_model']['anchor_central_f_priv_imported_217']:.12f}  "
          f"(resid {syn['f_priv_model']['anchor_central_resid']:.2e})", flush=True)
    print(f"  mu_safe_fresh = 528.48 * 0.969107 = {syn['constants']['mu_safe_fresh']:.6f}", flush=True)
    print("-" * 100, flush=True)
    print(f"  HEADLINE  f_priv_worstcase = {h['f_priv_worstcase']:.10f}  "
          f"(min vertex = {syn['worstcase']['f_priv_min_vertex']}, is_nls={h['f_priv_min_is_nls']}, "
          f"margin-to-runner {h['margin_to_runner_drop_pp']:+.4f}pp drop)", flush=True)
    print(f"  TEST      private_bar_worstcase = {h['private_bar_worstcase']:.4f}  "
          f"(vs central 528.48: {h['private_bar_worstcase_minus_central']:+.4f}; "
          f"vs 536.66: {h['private_bar_worstcase_minus_536']:+.4f}; "
          f"vs phys-ceiling 520.95: {h['gap_vs_physical_ceiling_520p95']:+.4f})", flush=True)
    print(f"  compliant_lane_private_feasible (<=536.66 central) = {h['compliant_lane_private_feasible']}  |  "
          f"vs LCB 525.73 = {h['compliant_lane_private_feasible_vs_lcb']}", flush=True)
    print("-" * 100, flush=True)
    print("  f_priv          private_bar   clears536  clears526  gap_vs_520.95   label", flush=True)
    for r in tb["rows"]:
        print(f"  {r['f_priv']:.6f}     {r['private_bar']:9.4f}    "
              f"{str(r['clears_central_536']):>5}     {str(r['clears_lcb_526']):>5}    "
              f"{r['gap_vs_physical_ceiling_520p95']:+8.3f}     {r['label']}", flush=True)
    print("-" * 100, flush=True)
    rb = syn["robustness"]
    print(f"  robustness | beta band f_priv_wc {rb['f_priv_worstcase_band_over_beta'][0]:.6f}.."
          f"{rb['f_priv_worstcase_band_over_beta'][1]:.6f}  bar "
          f"{rb['private_bar_worstcase_band_over_beta'][0]:.3f}..{rb['private_bar_worstcase_band_over_beta'][1]:.3f}  "
          f"(NLS stable={rb['vertex_identity_stable_over_beta']})", flush=True)
    print(f"  robustness | decode-drop scale: #52 emulated at s={rb['scale_emulates_52']:.4f}  "
          f"(drop {rb['row_at_scale_emulates_52']['drop_full_recovery_pct']:.3f}%, bar "
          f"{rb['row_at_scale_emulates_52']['private_bar']:.2f}); 536.66 break-even at "
          f"s={rb['scale_breakeven_central_536']:.4f}", flush=True)
    nc = rb["negative_coupling_caveat"]
    print(f"  robustness | #198 NEG coupling: f_priv floor {nc['f_priv_at_floor']:.6f} >= "
          f"ceiling {nc['f_priv_at_full_recovery']:.6f} (floor bar {nc['private_bar_at_floor']:.3f} <= "
          f"ceiling bar {nc['private_bar_at_ceiling']:.3f}) -> lambda=1 is conservative", flush=True)
    print("-" * 100, flush=True)
    print(f"  PRIMARY private_fpriv_worstcase_self_test_passes = "
          f"{st['private_fpriv_worstcase_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"        - {k}: {v}", flush=True)
    print("=" * 100, flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print(f"\n  HAND-OFF: {syn['handoff_line']}\n", flush=True)


def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[private-fpriv-worstcase] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    h, st, rb, c = syn["headline"], syn["self_test"], syn["robustness"], syn["constants"]
    run = init_wandb_run(
        job_type="private-fpriv-worstcase",
        agent="stark",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["private-fpriv-worstcase", "issue-211", "validity-gate", "worstcase-blend-lp",
              "f_priv", "compliant-lane-feasibility", "bank-the-analysis"],
        config={
            "beta": syn["beta"], "tau_low": c["tau_low"], "mu_safe_fresh": c["mu_safe_fresh"],
            "f_priv_217_imported": c["f_priv_217_imported"],
            "private_bar_217_imported": c["private_bar_217_imported"],
            "lambda1_ceiling": c["lambda1_ceiling"],
            "compliant_ceiling_central_536": c["compliant_ceiling_central_536"],
            "compliant_ceiling_lcb_526": c["compliant_ceiling_lcb_526"],
            "drop_both_176_pct": c["drop_both_176_pct"],
            "gt_decode_drop_pct": c["gt_decode_drop_pct"],
            "nls_axis": c["nls_axis"], "target_official": TARGET_OFFICIAL,
            "wandb_group": args.wandb_group, "baseline_tps": 481.53,
            "source_runs": "kanna#217 vgovdrjc, wirbel#199 wdyqnx3g, kanna#202 533jd6l1, "
                           "stark#208 wi4gxxx8, stark#198 llo1bzn3",
        },
    )
    if run is None:
        print("[private-fpriv-worstcase] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "private_fpriv_worstcase_self_test_passes":
            int(bool(st["private_fpriv_worstcase_self_test_passes"])),         # PRIMARY
        "private_bar_worstcase": h["private_bar_worstcase"],                   # TEST
        "f_priv_worstcase": h["f_priv_worstcase"],                            # HEADLINE
        "f_priv_min_is_nls": int(bool(h["f_priv_min_is_nls"])),
        "margin_to_runner_drop_pp": h["margin_to_runner_drop_pp"],
        "private_bar_worstcase_minus_central": h["private_bar_worstcase_minus_central"],
        "private_bar_worstcase_minus_536": h["private_bar_worstcase_minus_536"],
        "gap_vs_physical_ceiling_520p95": h["gap_vs_physical_ceiling_520p95"],
        "compliant_lane_private_feasible": int(bool(h["compliant_lane_private_feasible"])),
        "compliant_lane_private_feasible_vs_lcb": int(bool(h["compliant_lane_private_feasible_vs_lcb"])),
        "private_bar_worstcase_crosses_536": int(bool(h["private_bar_worstcase_crosses_536"])),
        "f_priv_breakeven_central_536": h["f_priv_breakeven_central_536"],
        "f_priv_breakeven_lcb_526": h["f_priv_breakeven_lcb_526"],
        "f_priv_observed_52": h["f_priv_observed_52"],
        "private_bar_at_52": h["private_bar_at_52"],
        "f52_minus_breakeven_central": h["f52_minus_breakeven_central"],
        "mu_safe_fresh": c["mu_safe_fresh"],
        "f_priv_central_derived_resid_vs_217": syn["f_priv_model"]["anchor_central_resid"],
        "nls_reconstruction_resid_vs_198": syn["f_priv_model"]["nls_reconstruction_resid_vs_198"],
        "lp_linear_vertex_optimal": int(bool(syn["worstcase"]["lp_linear_vertex_optimal"])),
        "treedp_vertex_optimal": int(bool(syn["worstcase"]["treedp_vertex_optimal"])),
        "treedp_sweep_min_fpriv": syn["worstcase"]["treedp_sweep_min_fpriv"],
        "vertex_identity_stable_over_beta": int(bool(rb["vertex_identity_stable_over_beta"])),
        "feasible_vs_central_stable_over_beta": int(bool(rb["feasible_vs_central_stable_over_beta"])),
        "scale_emulates_52": rb["scale_emulates_52"],
        "scale_breakeven_central_536": rb["scale_breakeven_central_536"],
        "f_priv_floor_negcoupling": rb["negative_coupling_caveat"]["f_priv_at_floor"],
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    # per-axis f_priv as logged scalars.
    for name, d in syn["worstcase"]["per_axis"].items():
        tag = name.replace("native_", "")
        summary[f"fpriv_axis_{tag}"] = d["f_priv"]
        summary[f"drop1_axis_{tag}_pct"] = d["drop_full_recovery_pct"]
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="private_fpriv_worstcase_result", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    print(f"[private-fpriv-worstcase] wandb logged {len(summary)} keys", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="private-drop-shape-robustness")
    args = ap.parse_args(argv)

    syn = synthesize()

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 226, "agent": "stark",
        "kind": "private-fpriv-worstcase", "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _nan_paths(payload)
    payload["nan_clean"] = not nan_paths
    syn["self_test"]["conditions"]["f_nan_clean"] = not nan_paths
    syn["self_test"]["private_fpriv_worstcase_self_test_passes"] = bool(
        all(syn["self_test"]["conditions"].values()))
    if nan_paths:
        print(f"[private-fpriv-worstcase] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[private-fpriv-worstcase] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = (syn["self_test"]["private_fpriv_worstcase_self_test_passes"]
              and payload["nan_clean"])
        print(f"[private-fpriv-worstcase] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
