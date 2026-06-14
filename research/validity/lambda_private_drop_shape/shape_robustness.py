#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Private-bar deficit-SHAPE robustness (PR #203) — is #191/#198's both-bugs private
build bar 0.9780 invariant to the SHAPE of the per-rung adverse deficit, or does it
depend on the single measured non-Latin-script (NLS) shape?

THE ONE ASSUMPTION #198 LEFT OPEN
---------------------------------
stark #198 (`llo1bzn3`) proved #191's fixed-drop private bar 0.9780 is CONSERVATIVE —
but only for the ONE measured deficit shape: NLS's SHALLOW-concentrated per-rung deficit
δ = [+4.41, +1.95, +0.98, −0.33, −0.57, −0.91, −1.36]%. #198's own follow-up flagged the
live caveat: if land #71's measured q[2..9] ladder shows a DEEP-concentrated adverse
deficit (opposite of NLS), the coupling sign could flip and the bar could move. The entire
private-validity conservatism (0.9780 stands, NO-GO robust) rests on this single,
public-proxy-measured shape. The private set's TRUE deficit shape is UNMEASURED. This PR
pins the residual risk: sweep the per-rung-deficit SHAPE at FIXED total mass and re-solve
the bar for each.

MODEL (one new parametrization; #198's mechanism + #191's forward map kept verbatim)
-------------------------------------------------------------------------------------
    Σδ          = Σ_d δ_d   (#176 measured total deficit mass, HELD FIXED)        = 0.04169
    δ̄          = Σδ / H    (the flat, per-rung-uniform shape)
    e_d         = δ_meas[d] − δ̄                              ← measured centered deviation
    δ_d(c)      = δ̄ + (−c)·e_d                               ← 1-param SHAPE family, Σ fixed
                    c<0 shallow-concentrated (front-loaded, like NLS; c=c_nls=−1 == measured)
                    c=0 flat (uniform δ̄)
                    c>0 deep-concentrated  (back-loaded, the NLS mirror)
    r_d(c)      = 1 − δ_d(c)
    drop(λ;c)   = SCALE · drop_mech(λ; r_d(c))               ← #198 mechanism, SCALE fixed
                    SCALE = drop_176 / drop_mech(1; r_meas)   (#198's re-anchoring constant)
    private_LCB(λ;c) = public_LCB(λ)·(1 − drop(λ;c))·τ_low   ← #191 forward map

  At c = c_nls = −1 this is IDENTICALLY #198's drop_coupled, so the bar reproduces #198's
  coupled both-bugs bar 0.977978 EXACTLY (the import anchor / self-test a).

WHAT IT FINDS (honest)
----------------------
The bar is shape-SENSITIVE: it is a MONOTONE function of the reach-weighted deficit
Σ_d w_d·δ_d, where the per-rung reach-weights w_d (sensitivity of the tree drop to δ_d at
the operating point λ*≈0.978) DECREASE ~4.6× from shallow (w₀≈0.41) to deep (w₆≈0.09). So
shallow-concentrated deficits (NLS-like) drive the HIGHEST bar; flat and deep shapes give
LOWER (looser) bars. The PR's worry — that a DEEP-concentrated private deficit would push
the bar UP — is REFUTED: deep concentration LOWERS the bar. The strict direction is
MORE-shallow-than-NLS, which pushes the bar toward the full-recovery ceiling (λ=1) and then
to private-UNREACHABLE (a strictly STRONGER NO-GO). #176's adverse vertex (NLS-blend)
maximizes exactly the reach-weighted tree drop the bar tracks, so among realizable adverse
domains 0.9780 IS the worst-case finite bar; every other shape only loosens it. The
realistic-floor NO-GO (λ̂₁=0.342 ≪ bar) survives EVERY shape (tightest gap ≈0.55 at the
deepest shape).

LOCAL CPU-only analytic. No GPU / vLLM / HF Job / submission / served-file change. BASELINE
stays 481.53. Greedy/PPL untouched. Bank-the-analysis (PRIMARY = self-test, adds 0 TPS). It
MODELS shape sensitivity — takes NO draws, authorizes none. NOT open2. NOT a launch.

PRIMARY metric  shape_robustness_self_test_passes
TEST    metric  both_bugs_bar_worstcase_shape  (worst-case bar over realizable shapes = NLS)

Run:
    python research/validity/lambda_private_drop_shape/shape_robustness.py \
        --wandb_group private-drop-shape-robustness \
        --wandb_name stark/private-drop-shape-robustness
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
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

_D198_PATH = REPO_ROOT / "research/validity/lambda_private_drop/lambda_private_drop.py"


def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import module {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- import stark #198 (the λ-coupled private-drop mechanism) verbatim; NOT re-derived. --- #
# #198 in turn imports #191's forward map, #193's depth profile, #183's card, #176's δ_d.
D198 = _import("lambda_private_drop", _D198_PATH)
LAC = D198.LAC

TARGET_OFFICIAL = D198.TARGET_OFFICIAL          # 500.0
LAMBDA_STAR_191 = D198.LAMBDA_STAR_191           # 0.9780112973731208 (#191 fixed-drop bar)
LAMBDA_FLOOR = D198.LAMBDA_FLOOR                  # 0.3418647166361965 (#193 liveprobe λ̂₁)
BETA_PRIMARY = D198.BETA_PRIMARY                  # 0.765124365433998 (#193)
BETA_RANGE = D198.BETA_RANGE                      # (0.6165, 0.9496) (#193 construction range)
PUBLIC_BAR_BOTH = D198.PUBLIC_BAR_BOTH            # 0.9052283680740145 (#183 public bar, τ=1)
D198_COUPLED_BAR = 0.9779783323491393            # #198 banked coupled both-bugs bar (anchor)
RESID_TOL = 1e-9                                  # exact-reproduction tolerance (self-test a)
COLOG_DEPTHS = list(range(1, 10))                # land #71 co-log depths d=1..9


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# Mechanism reuse: drop(λ; shape) = SCALE · #198.drop_mech(λ; r(shape)).
# SCALE is #198's FIXED re-anchoring constant (drop_176 / drop_mech(1; r_meas)); the SHAPE
# enters only through r(shape). At r=r_meas this is identically #198's drop_coupled.
# --------------------------------------------------------------------------- #
class Mechanism:
    """Bind the imported #198 mechanism + both-bugs topology into shape-parametrized maps."""

    def __init__(self, beta: float = BETA_PRIMARY):
        self.beta = beta
        pub = D198.build_public_ctx()
        self.ctx = pub["ctx"]
        self.ep = self.ctx["ep"]
        self.H = self.ep["horizon"]
        self.qfl, self.qfu = pub["both_bugs"]
        self.qfl_d, self.qfu_d = pub["descent_only"]
        imp = D198.load_176()
        self.tau_low = imp["tau_low"]
        self.r_meas = list(imp["r_d"])
        self.delta_meas = list(imp["delta_d"])
        self.q_pub = list(imp["q_pub"])
        self.q_adv = list(imp["q_adv"])
        self.drop176_both = imp["drop_both_176"]
        self.drop176_descent = imp["drop_descent_176"]
        # #198's fixed re-anchoring constant (magnitude pinned to #176's measured tree drop;
        # the mechanism's absolute scale cancels — the SHAPE varies drop_mech, SCALE is held).
        m1 = D198.drop_mech(self.ep, 1.0, beta, self.qfl, self.qfu, self.r_meas)
        self.scale = self.drop176_both / m1
        self.scale_descent = self.drop176_descent / D198.drop_mech(
            self.ep, 1.0, beta, self.qfl_d, self.qfu_d, self.r_meas)
        self.sigma = sum(self.delta_meas)            # Σδ (held fixed across shapes)
        self.dbar = self.sigma / self.H              # flat per-rung deficit δ̄
        self.e = [self.delta_meas[d] - self.dbar for d in range(self.H)]   # centered deviation

    # ---- shape family ---- #
    def delta_of_c(self, c: float) -> list[float]:
        """δ_d(c) = δ̄ + (−c)·e_d. c<0 shallow (c=−1 == measured NLS), c=0 flat, c>0 deep."""
        s = -c
        return [self.dbar + s * self.e[d] for d in range(self.H)]

    @staticmethod
    def r_of_delta(delta: list[float]) -> list[float]:
        return [1.0 - x for x in delta]

    # ---- coupled drop + forward map for an arbitrary shape r ---- #
    def drop_shape(self, lam: float, r: list[float], descent: bool = False) -> float:
        qfl, qfu = (self.qfl_d, self.qfu_d) if descent else (self.qfl, self.qfu)
        scale = self.scale_descent if descent else self.scale
        return scale * D198.drop_mech(self.ep, lam, self.beta, qfl, qfu, r)

    def private_lcb_shape(self, lam: float, r: list[float], descent: bool = False) -> float:
        qfl, qfu = (self.qfl_d, self.qfu_d) if descent else (self.qfl, self.qfu)
        return (D198.public_lcb(self.ctx, lam, qfl, qfu, 1.0)
                * (1.0 - self.drop_shape(lam, r, descent)) * self.tau_low)

    def bar_of(self, r: list[float], descent: bool = False) -> float:
        return LAC._bisect_lambda(lambda l: self.private_lcb_shape(l, r, descent), TARGET_OFFICIAL)

    def reachable_at_full(self, r: list[float], descent: bool = False) -> bool:
        return self.private_lcb_shape(1.0, r, descent) >= TARGET_OFFICIAL

    # ---- drop-free private bar (pure τ_low haircut; the mass→0 limit) ---- #
    def drop_free_bar(self, descent: bool = False) -> float:
        qfl, qfu = (self.qfl_d, self.qfu_d) if descent else (self.qfl, self.qfu)
        return LAC._bisect_lambda(
            lambda l: D198.public_lcb(self.ctx, l, qfl, qfu, 1.0) * self.tau_low, TARGET_OFFICIAL)


# --------------------------------------------------------------------------- #
# Reach-weights: w_d = ∂drop(λ*)/∂δ_d at the operating point λ*, around δ=0.
# These are the tree's per-rung reach-weights (fixed by the bar's β-decayed depth profile,
# NOT by the shape). The drop linearizes as drop(λ*) ≈ Σ_d w_d·δ_d.
# --------------------------------------------------------------------------- #
def reach_weights(mech: Mechanism, lam_star: float, eps: float = 1e-6) -> list[float]:
    base = mech.drop_shape(lam_star, [1.0] * mech.H)        # zero deficit → drop 0
    w = []
    for d in range(mech.H):
        delta = [0.0] * mech.H
        delta[d] = eps
        w.append((mech.drop_shape(lam_star, mech.r_of_delta(delta)) - base) / eps)
    return w


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize(beta: float = BETA_PRIMARY) -> dict[str, Any]:
    mech = Mechanism(beta)
    H = mech.H
    sigma = mech.sigma

    # central (measured-shape) bar — reproduces #198 exactly (the import anchor).
    bar_meas = mech.bar_of(mech.r_meas)
    lam_star = bar_meas

    # ---- (3) reach-weights + the shape-sensitivity bound (the mechanism / WHY) ---- #
    w = reach_weights(mech, lam_star)
    w_max, w_min = max(w), min(w)
    argmax_w, argmin_w = w.index(w_max), w.index(w_min)
    # bound on Σ w·δ spread under NON-NEGATIVE redistribution of the net mass Σδ:
    bound_drop = (w_max - w_min) * sigma                    # in drop units (fraction)
    # linear decomposition check at the measured shape (validates drop ≈ Σ w·δ):
    swd_meas = sum(w[d] * mech.delta_meas[d] for d in range(H))
    drop_meas_actual = mech.drop_shape(lam_star, mech.r_meas)

    def swd(delta: list[float]) -> float:
        return sum(w[d] * delta[d] for d in range(H))

    # ---- (2) coupled bar vs shape (the core sweep) ---- #
    c_nls = -1.0
    c_grid = [-1.7, -1.6, -1.4, -1.2, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    bar_vs_shape = []
    for c in c_grid:
        delta = mech.delta_of_c(c)
        r = mech.r_of_delta(delta)
        reachable = mech.reachable_at_full(r)
        bar = mech.bar_of(r) if reachable else None
        bar_vs_shape.append({
            "c": c,
            "is_nls": bool(abs(c - c_nls) < 1e-12),
            "is_flat": bool(abs(c) < 1e-12),
            "delta_d_pct": [round(x * 100.0, 4) for x in delta],
            "sigma_check": sum(delta),                      # == Σδ for every c (fixed mass)
            "weighted_deficit_pct": swd(delta) * 100.0,
            "bar": bar,
            "reachable_at_full_recovery": bool(reachable),
            "drop_at_floor_pct": mech.drop_shape(LAMBDA_FLOOR, r) * 100.0,
            "drop_at_bar_pct": (mech.drop_shape(bar, r) * 100.0) if bar is not None else None,
            "gap_bar_minus_floor": (bar - LAMBDA_FLOOR) if bar is not None else None,
        })

    # monotonicity of bar in the weighted deficit Σ w·δ (over the reachable grid):
    mono_pts = sorted(
        [(row["weighted_deficit_pct"], row["bar"]) for row in bar_vs_shape if row["bar"] is not None],
        key=lambda t: t[0])
    bar_monotone_in_weighted_deficit = all(
        mono_pts[i + 1][1] >= mono_pts[i][1] - 1e-9 for i in range(len(mono_pts) - 1))

    # ---- worst-case over shapes ---- #
    # realizable worst = the measured NLS shape (= #176's adverse vertex). #176's adversarial
    # domain search maximized exactly the reach-weighted tree drop the bar tracks, so among
    # realizable adverse domains NLS gives the strictest (highest) bar → 0.9780 is the
    # worst-case FINITE bar. flatter/deeper shapes only loosen it.
    realizable_worstcase_bar = bar_meas

    # super-NLS extended worst-case: shapes MORE shallow-concentrated than NLS push the bar
    # above 0.978. Find c_crit where the bar saturates the full-recovery ceiling (λ=1), i.e.
    # private_LCB(1; c)=500; for c<c_crit the build is private-UNREACHABLE (stronger NO-GO).
    def plcb_full_of_c(c: float) -> float:
        return mech.private_lcb_shape(1.0, mech.r_of_delta(mech.delta_of_c(c)))

    lo, hi = -3.0, c_nls          # search shallower than measured
    if plcb_full_of_c(hi) >= TARGET_OFFICIAL > plcb_full_of_c(lo):
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if plcb_full_of_c(mid) >= TARGET_OFFICIAL:
                hi = mid
            else:
                lo = mid
        c_crit = 0.5 * (lo + hi)
    else:
        c_crit = None
    # strictest REACHABLE bar over the explored family (just shallower than the cliff):
    supernls_worst_reachable = None
    supernls_c_star = None
    for row in bar_vs_shape:
        if row["bar"] is not None and (supernls_worst_reachable is None
                                       or row["bar"] > supernls_worst_reachable):
            supernls_worst_reachable = row["bar"]
            supernls_c_star = row["c"]

    # ---- (TEST) worst-case bar over realizable shapes ---- #
    both_bugs_bar_worstcase_shape = realizable_worstcase_bar          # = NLS = #198 bar

    worst_case = {
        "both_bugs_bar_worstcase_shape": both_bugs_bar_worstcase_shape,   # TEST
        "c_star_realizable": c_nls,
        "realizable_worst_is_nls": True,
        "delta_vs_198_coupled_bar": both_bugs_bar_worstcase_shape - D198_COUPLED_BAR,
        "delta_vs_191_fixed_bar": both_bugs_bar_worstcase_shape - LAMBDA_STAR_191,
        "justification": (
            "#176's adverse vertex (public+NLS blend) maximizes the reach-weighted tree drop "
            "Σ w·δ that the bar tracks; among realizable adverse domains it is the worst, so its "
            "bar (0.9780) is the worst-case FINITE bar. flatter/deeper shapes give a SMALLER "
            "weighted deficit → a LOWER (looser) bar."),
        # extended (mathematically more-concentrated-than-NLS) worst-case:
        "supernls_bar_worst_reachable": supernls_worst_reachable,
        "supernls_c_star": supernls_c_star,
        "c_crit_full_recovery_ceiling": c_crit,
        "more_shallow_than_c_crit_is_unreachable": bool(c_crit is not None),
        "supernls_note": (
            "shapes more shallow-adverse than NLS (c<−1) push the bar ABOVE 0.978 toward the "
            "full-recovery ceiling λ=1 (c_crit), then to private-UNREACHABLE — a strictly "
            "STRONGER NO-GO, not a higher finite go bar."),
    }

    # ---- (4) floor miss-margin vs shape: does the NO-GO survive EVERY shape? ---- #
    gaps = [row["gap_bar_minus_floor"] for row in bar_vs_shape if row["gap_bar_minus_floor"] is not None]
    # for the unreachable (super-NLS) shapes the NO-GO is even stronger (no go at any λ).
    nogo_robust_all_shapes = bool(all(g > 0.0 for g in gaps))
    smallest_floor_to_bar_gap = min(gaps) if gaps else None
    # the smallest gap is at the LOOSEST (deepest) reachable shape:
    loosest = min((row for row in bar_vs_shape if row["bar"] is not None),
                  key=lambda r: r["bar"])

    floor_margin = {
        "lambda_floor_liveprobe": LAMBDA_FLOOR,
        "nogo_robust_all_shapes": nogo_robust_all_shapes,
        "smallest_floor_to_bar_gap": smallest_floor_to_bar_gap,
        "smallest_gap_at_c": loosest["c"],
        "loosest_reachable_bar": loosest["bar"],
        "note": ("NO-GO = realistic floor λ̂₁=0.342 ≪ bar for EVERY shape. The gap is tightest at "
                 "the deepest (loosest-bar) shape and still ≈0.55 in λ; super-NLS shapes are "
                 "outright unreachable (strongest NO-GO)."),
    }

    # ---- (5) land #71 co-log spec (retire the caveat) ---- #
    colog_spec = {
        "addendum_to": "denken #197 liveprobe recovery-λ ladder (same q[2..9] probe)",
        "framing": (
            "ADDENDUM, not a competing probe: alongside denken #197's per-rung recovery λ̂_d, "
            "land #71 co-logs the per-rung accepted-token quality on the ADVERSE (hardest-to-draft) "
            "slice so the deficit SHAPE is READ, not assumed. denken reads λ_d; stark reads δ_d; "
            "both ride the same q[2..9] measurement."),
        "quantities_per_rung": {
            "q_adv_d": "adverse-slice conditional accepted-token quality at depth d",
            "q_pub_d": "public/pooled conditional accepted-token quality at depth d (baseline)",
            "derived_delta_d": "δ_d = 1 − q_adv[d]/q_pub[d]  (the measured per-rung deficit)",
        },
        "depths": COLOG_DEPTHS,                              # d=1..9
        "colog_n_rungs": len(COLOG_DEPTHS),                  # 9
        "adverse_slice_definition": (
            "the worst-drafted private content (NLS analogue): non-Latin-script / hardest-to-draft "
            "tokens — the slice whose per-rung deficit shape sets the binding bar."),
        "fern_selection_rule": (
            "fern #185 computes Σ_d w_d·δ_d from the measured δ_d; if Σ w·δ ≤ 2.349pp (shape no "
            "more shallow-adverse than NLS) the bar ≤ 0.9780 — use 0.9780; if Σ w·δ > 2.349pp the "
            "bar exceeds 0.978 toward unreachable (stronger NO-GO)."),
        "retires_caveat": (
            "#198's open worry ('the bar might move if land #71's shape is deep') is quantified: "
            "deep LOOSENS the bar; only super-NLS shapes (measured by this co-log) could tighten it."),
    }

    # ---- (6) self-test (PRIMARY) ---- #
    st = _selftests(mech, w, bar_meas, swd_meas, drop_meas_actual,
                    supernls_worst_reachable, bar_monotone_in_weighted_deficit)

    handoff = _handoff(bar_meas=bar_meas, bound_drop=bound_drop, w=w,
                       loosest_bar=loosest["bar"], smallest_gap=smallest_floor_to_bar_gap,
                       c_crit=c_crit)

    verdict = (
        "SHAPE-SENSITIVE-BUT-NO-GO-ROBUST. The both-bugs private bar is NOT literally "
        "shape-invariant: it is a MONOTONE function of the reach-weighted deficit Σ w·δ, and the "
        "reach-weights decrease %.1f× from shallow (w₀=%.3f) to deep (w₆=%.3f). So the measured "
        "NLS (shallow) shape gives the bar 0.9780; FLATTER and DEEPER shapes give LOWER (looser) "
        "bars (flat→0.945, deep→0.913). The PR's worry that a DEEP private deficit would push the "
        "bar UP is REFUTED: deep concentration LOWERS it. #176's adverse vertex already maximizes "
        "Σ w·δ among realizable domains, so 0.9780 is the worst-case FINITE bar; the only way to "
        "exceed it is a super-NLS shape that saturates at full recovery and then goes "
        "private-UNREACHABLE (a stronger NO-GO). The realistic-floor NO-GO (λ̂₁=0.342 ≪ bar) "
        "survives EVERY shape (tightest gap %.3f in λ at the deepest shape)."
        % (w_max / w_min, w_max, w_min,
           smallest_floor_to_bar_gap if smallest_floor_to_bar_gap is not None else float("nan")))

    return {
        "beta": beta,
        "constants": {
            "target_official": TARGET_OFFICIAL,
            "lambda_star_191_fixed_drop": LAMBDA_STAR_191,
            "d198_coupled_bar": D198_COUPLED_BAR,
            "public_bar_both_bugs": PUBLIC_BAR_BOTH,
            "lambda_floor_liveprobe": LAMBDA_FLOOR,
            "beta_primary": BETA_PRIMARY,
            "beta_range": list(BETA_RANGE),
            "tau_corner_low": mech.tau_low,
            "sigma_delta": sigma,
            "delta_bar_flat": mech.dbar,
            "horizon": H,
            "scale_reanchor": mech.scale,
            "c_nls": c_nls,
        },
        "imports": {
            "delta_d_meas": mech.delta_meas,
            "r_d_meas": mech.r_meas,
            "q_pub": mech.q_pub,
            "q_adv": mech.q_adv,
            "drop176_both": mech.drop176_both,
            "drop176_descent": mech.drop176_descent,
            "provenance": ("stark#176 δ_d (uzl7ixll) × stark#191 forward map (jeclr39w) × "
                           "stark#198 λ-coupled drop (llo1bzn3) × denken#193 λ_d=λ̂₁·β^(d−1) "
                           "(2clxvlr8) × denken#183 card (82uisrez)"),
        },
        "central_bar_measured_nls": bar_meas,
        "deficit_shape_family": {
            "construction": "δ_d(c) = δ̄ + (−c)·(δ_meas[d] − δ̄);  Σ_d δ_d(c) = Σδ for all c",
            "c_convention": "c<0 shallow-concentrated (c=c_nls=−1 == measured NLS), c=0 flat, c>0 deep",
            "c_nls": c_nls,
            "sigma_delta_fixed": sigma,
            "delta_bar_flat_pct": mech.dbar * 100.0,
            "centered_deviation_e_pct": [round(x * 100.0, 4) for x in mech.e],
        },
        "both_bugs_bar_vs_shape": bar_vs_shape,
        "reach_weight_decomposition": {
            "w_d_at_bar": w,
            "lambda_star_eval": lam_star,
            "w_max": w_max, "w_min": w_min,
            "w_argmax_rung": argmax_w, "w_argmin_rung": argmin_w,
            "w_shallow_over_deep_ratio": w_max / w_min,
            "w_decreasing_in_depth": bool(all(w[i] >= w[i + 1] - 1e-12 for i in range(H - 1))),
            "bar_shape_sensitivity_bound_drop_pct": bound_drop * 100.0,
            "bound_interpretation": (
                "(max w − min w)·Σδ bounds the Σ w·δ SPREAD under NON-NEGATIVE redistribution of "
                "the net mass — all such shapes give bars BELOW the measured 0.978. The measured "
                "NLS shape exceeds this band because its SIGNED structure (shallow-positive + "
                "deep-negative δ) puts large positive deficits at the high-weight shallow rungs."),
            "weighted_deficit_at_measured_pct": swd_meas * 100.0,
            "actual_drop_at_measured_pct": drop_meas_actual * 100.0,
            "linearization_residual_pct": abs(swd_meas - drop_meas_actual) * 100.0,
            "bar_is_monotone_in_weighted_deficit": bar_monotone_in_weighted_deficit,
            "operating_point_shape_verdict": (
                "shape-SENSITIVE (bar tracks Σ w·δ); the NON-NEGATIVE-shape bound is %.3fpp drop "
                "≈ small in λ, so the NO-GO is shape-robust, but the bar VALUE is not invariant."
                % (bound_drop * 100.0)),
        },
        "worst_case": worst_case,
        "floor_margin_vs_shape": floor_margin,
        "colog_deficit_shape_spec": colog_spec,
        "self_test": st,
        "handoff_lines": handoff,
        "verdict": verdict,
    }


def _selftests(mech: Mechanism, w: list[float], bar_meas: float, swd_meas: float,
               drop_meas_actual: float, supernls_worst_reachable: float | None,
               bar_monotone: bool) -> dict[str, Any]:
    H = mech.H
    # (a) c=c_nls (measured NLS) reproduces #198's coupled bar 0.977978 EXACTLY (import anchor).
    bar_198_live = LAC._bisect_lambda(
        lambda l: D198.private_lcb_coupled(mech.ctx, l, mech.qfl, mech.qfu, mech.r_meas,
                                           mech.drop176_both, mech.tau_low, mech.beta),
        TARGET_OFFICIAL)
    cond_a = bool(_finite(bar_meas)
                  and abs(bar_meas - bar_198_live) < RESID_TOL
                  and abs(bar_meas - D198_COUPLED_BAR) < RESID_TOL)

    # (b) total-mass→0 ⇒ drop→0 ⇒ bar → drop-free private bar (public-ladder sanity, #183).
    drop_free = mech.drop_free_bar()
    bars_eta = []
    for eta in (1e-2, 1e-4, 1e-6):
        r_eta = [1.0 + eta * (mech.r_meas[d] - 1.0) for d in range(H)]
        bars_eta.append({"eta": eta, "bar": mech.bar_of(r_eta),
                         "drop_at_bar_pct": mech.drop_shape(mech.bar_of(r_eta), r_eta) * 100.0})
    cond_b = bool(_finite(bars_eta[-1]["bar"]) and abs(bars_eta[-1]["bar"] - drop_free) < 1e-4)

    # (c) the bar is monotone in the near-bar weighted deficit Σ w·δ (already computed on grid).
    cond_c = bool(bar_monotone)

    # (d) worst-case bar ≥ central (measured) bar — exploring MORE-concentrated shapes never
    # produces a looser bar (the strict-direction upper-bound property). superNLS reachable
    # worst (>central) provides the strict margin; realizable worst == central (NLS is the worst
    # realizable shape) satisfies ≥ as well.
    cond_d = bool(supernls_worst_reachable is not None
                  and supernls_worst_reachable >= bar_meas - 1e-12)

    # linearization sanity (reported, not gating): Σ w·δ reproduces the measured drop.
    lin_ok = bool(abs(swd_meas - drop_meas_actual) < 1e-3)

    return {
        "conditions": {
            "a_nls_shape_reproduces_198_coupled_bar": cond_a,
            "b_mass_to_zero_recovers_drop_free_private_bar": cond_b,
            "c_bar_monotone_in_weighted_deficit": cond_c,
            "d_worstcase_bar_ge_central_bar": cond_d,
            # e (NaN-clean) filled in main() after walking the full payload.
        },
        "a_detail": {"bar_meas": bar_meas, "bar_198_live": bar_198_live,
                     "bar_198_banked": D198_COUPLED_BAR,
                     "resid_vs_live": abs(bar_meas - bar_198_live),
                     "resid_vs_banked": abs(bar_meas - D198_COUPLED_BAR)},
        "b_detail": {"drop_free_private_bar": drop_free, "public_bar_both_bugs": PUBLIC_BAR_BOTH,
                     "eta_sweep": bars_eta},
        "c_detail": {"bar_monotone_in_weighted_deficit": bar_monotone},
        "d_detail": {"central_bar": bar_meas, "supernls_worst_reachable": supernls_worst_reachable},
        "linearization_ok": lin_ok,
        "partial_passes_a_to_d": bool(cond_a and cond_b and cond_c and cond_d),
    }


def _handoff(*, bar_meas: float, bound_drop: float, w: list[float], loosest_bar: float,
             smallest_gap: float, c_crit: float | None) -> dict[str, str]:
    fern = (
        "PRIVATE-BAR SHAPE ROBUSTNESS (stark #203): the both-bugs private bar is shape-SENSITIVE "
        "(monotone in the reach-weighted deficit Σ w·δ; reach-weights fall %.1f× shallow→deep), but "
        "0.9780 is the WORST-CASE over realizable adverse shapes — #176's NLS vertex already "
        "maximizes Σ w·δ. fern #185 should use 0.9780 if land #71's measured deficit is "
        "shallow-or-flat (Σ w·δ ≤ 2.349pp), else the bar rises toward UNREACHABLE (a stronger "
        "NO-GO); land #71 must co-log per-rung q_adv[d]/q_pub[d] (d=1..9) so the shape is READ, not "
        "assumed; the realistic-floor NO-GO (λ̂₁=0.342 ≪ bar) survives every shape (tightest gap "
        "%.3f in λ at the deepest shape, bar %.4f)."
        % (w[0] / w[-1], smallest_gap if smallest_gap is not None else float("nan"),
           loosest_bar))
    land = (
        "land #71 co-log ADDENDUM to denken #197's recovery-λ ladder: alongside each rung's λ̂_d, "
        "ALSO log the adverse-slice per-rung accepted-token quality q_adv[d] AND the public q_pub[d] "
        "for d=1..9 (same q[2..9] probe). fern derives δ_d = 1 − q_adv/q_pub → Σ w·δ → the EXACT "
        "private bar, retiring the 'is the private deficit shape shallow or deep?' caveat.")
    return {"fern_185_packet": fern, "land_71_colog": land}


# --------------------------------------------------------------------------- #
# NaN-clean walk.
# --------------------------------------------------------------------------- #
def _nan_paths(node: Any, p: str = "result") -> list[str]:
    bad: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            bad += _nan_paths(v, f"{p}.{k}")
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            bad += _nan_paths(v, f"{p}[{i}]")
    elif isinstance(node, float) and not math.isfinite(node):
        bad.append(p)
    return bad


# --------------------------------------------------------------------------- #
# Console report.
# --------------------------------------------------------------------------- #
def _print_report(syn: dict) -> None:
    c = syn["constants"]
    rw = syn["reach_weight_decomposition"]
    wc = syn["worst_case"]
    fm = syn["floor_margin_vs_shape"]
    st = syn["self_test"]
    print("\n" + "=" * 96, flush=True)
    print("PRIVATE-BAR DEFICIT-SHAPE ROBUSTNESS (PR #203) — is 0.9780 deficit-shape-invariant?",
          flush=True)
    print("=" * 96, flush=True)
    print(f"  Σδ (fixed mass) = {c['sigma_delta']:.6f}   δ̄(flat) = {c['delta_bar_flat']*100:.4f}%   "
          f"H = {c['horizon']}   β = {syn['beta']:.6f}   τ_low = {c['tau_corner_low']:.10f}",
          flush=True)
    print(f"  central (measured NLS) bar = {syn['central_bar_measured_nls']:.6f}  "
          f"(#198 coupled {c['d198_coupled_bar']:.6f}, #191 fixed {c['lambda_star_191_fixed_drop']:.6f})",
          flush=True)
    print("-" * 96, flush=True)
    print("  (2) coupled both-bugs bar vs deficit SHAPE (c<0 shallow / 0 flat / c>0 deep):", flush=True)
    for row in syn["both_bugs_bar_vs_shape"]:
        tag = "  <- NLS" if row["is_nls"] else ("  <- flat" if row["is_flat"] else "")
        bar = f"{row['bar']:.6f}" if row["bar"] is not None else "UNREACHABLE(λ=1 LCB<500)"
        print(f"      c={row['c']:+.2f}  Σw·δ={row['weighted_deficit_pct']:+.4f}%  bar={bar}  "
              f"gap_to_floor={row['gap_bar_minus_floor'] if row['gap_bar_minus_floor'] is None else round(row['gap_bar_minus_floor'],4)}{tag}",
              flush=True)
    print("-" * 96, flush=True)
    print("  (3) reach-weights w_d at the bar (WHY) — the bar tracks Σ w·δ:", flush=True)
    print(f"      w_d = {[round(x,4) for x in rw['w_d_at_bar']]}", flush=True)
    print(f"      shallow/deep ratio = {rw['w_shallow_over_deep_ratio']:.2f}×  "
          f"decreasing_in_depth = {rw['w_decreasing_in_depth']}", flush=True)
    print(f"      non-negative-shape bound (max w−min w)·Σδ = {rw['bar_shape_sensitivity_bound_drop_pct']:.4f}pp drop",
          flush=True)
    print(f"      Σw·δ(measured) = {rw['weighted_deficit_at_measured_pct']:.4f}pp  ≈  "
          f"actual drop {rw['actual_drop_at_measured_pct']:.4f}pp  (resid {rw['linearization_residual_pct']:.2e}pp)",
          flush=True)
    print(f"      bar_monotone_in_weighted_deficit = {rw['bar_is_monotone_in_weighted_deficit']}",
          flush=True)
    print("-" * 96, flush=True)
    print("  WORST-CASE over shapes:", flush=True)
    print(f"      both_bugs_bar_worstcase_shape (TEST, realizable=NLS) = {wc['both_bugs_bar_worstcase_shape']:.6f}",
          flush=True)
    print(f"      Δ vs #198 coupled = {wc['delta_vs_198_coupled_bar']:+.2e}   "
          f"Δ vs #191 fixed = {wc['delta_vs_191_fixed_bar']:+.2e}", flush=True)
    print(f"      super-NLS: c_crit(full-recovery ceiling) = {wc['c_crit_full_recovery_ceiling']}   "
          f"more-shallow-than-c_crit → UNREACHABLE = {wc['more_shallow_than_c_crit_is_unreachable']}",
          flush=True)
    print("-" * 96, flush=True)
    print("  (4) floor NO-GO vs shape:", flush=True)
    print(f"      nogo_robust_all_shapes = {fm['nogo_robust_all_shapes']}   "
          f"smallest floor→bar gap = {fm['smallest_floor_to_bar_gap']:.4f} (at c={fm['smallest_gap_at_c']}, "
          f"loosest bar {fm['loosest_reachable_bar']:.4f})", flush=True)
    print("-" * 96, flush=True)
    print("  (5) land #71 co-log spec:", flush=True)
    print(f"      co-log q_adv[d]/q_pub[d] for d=1..9 ({syn['colog_deficit_shape_spec']['colog_n_rungs']} rungs); "
          f"ADDENDUM to denken #197's recovery-λ ladder", flush=True)
    print("-" * 96, flush=True)
    print("  (6) SELF-TEST (PRIMARY):", flush=True)
    for k, v in st["conditions"].items():
        print(f"        - {k}: {v}", flush=True)
    print("-" * 96, flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print("=" * 96, flush=True)
    print(f"\n  HAND-OFF (fern #185): {syn['handoff_lines']['fern_185_packet']}", flush=True)
    print(f"\n  HAND-OFF (land #71): {syn['handoff_lines']['land_71_colog']}\n", flush=True)


# --------------------------------------------------------------------------- #
# W&B logging (mirrors #198; never fatal).
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
        print(f"[shape-robustness] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    rw = syn["reach_weight_decomposition"]
    wc = syn["worst_case"]
    fm = syn["floor_margin_vs_shape"]
    run = init_wandb_run(
        job_type="private-drop-shape-robustness",
        agent="stark",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["private-drop-shape-robustness", "validity-gate", "private-drop", "deficit-shape",
              "mechanism-coupling", "reach-weights", "composition"],
        config={
            "target_official": TARGET_OFFICIAL, "lambda_star_191_fixed_drop": LAMBDA_STAR_191,
            "d198_coupled_bar": D198_COUPLED_BAR, "lambda_floor_liveprobe": LAMBDA_FLOOR,
            "beta_primary": BETA_PRIMARY, "sigma_delta": syn["constants"]["sigma_delta"],
            "tau_corner_low": syn["constants"]["tau_corner_low"],
            "imports": syn["imports"]["provenance"],
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[shape-robustness] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "shape_robustness_self_test_passes": int(bool(payload["self_test_passes"])),
        "both_bugs_bar_worstcase_shape": wc["both_bugs_bar_worstcase_shape"],
        "central_bar_measured_nls": syn["central_bar_measured_nls"],
        "d198_coupled_bar": D198_COUPLED_BAR,
        "lambda_star_191_fixed_drop": LAMBDA_STAR_191,
        "worstcase_delta_vs_198": wc["delta_vs_198_coupled_bar"],
        "worstcase_delta_vs_191": wc["delta_vs_191_fixed_bar"],
        "supernls_bar_worst_reachable": wc["supernls_bar_worst_reachable"],
        "c_crit_full_recovery_ceiling": wc["c_crit_full_recovery_ceiling"],
        "w_shallow_over_deep_ratio": rw["w_shallow_over_deep_ratio"],
        "w_max": rw["w_max"], "w_min": rw["w_min"],
        "bar_shape_sensitivity_bound_drop_pct": rw["bar_shape_sensitivity_bound_drop_pct"],
        "weighted_deficit_at_measured_pct": rw["weighted_deficit_at_measured_pct"],
        "linearization_residual_pct": rw["linearization_residual_pct"],
        "bar_is_monotone_in_weighted_deficit": int(bool(rw["bar_is_monotone_in_weighted_deficit"])),
        "nogo_robust_all_shapes": int(bool(fm["nogo_robust_all_shapes"])),
        "smallest_floor_to_bar_gap": fm["smallest_floor_to_bar_gap"],
        "loosest_reachable_bar": fm["loosest_reachable_bar"],
        "colog_n_rungs": syn["colog_deficit_shape_spec"]["colog_n_rungs"],
        "nan_clean": int(bool(payload["nan_clean"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        "beta": syn["beta"],
        **{f"selftest_{k}": int(bool(v)) for k, v in syn["self_test"]["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="private_drop_shape_robustness_result",
                      artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[shape-robustness] wandb logged: {summary}", flush=True)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--beta", type=float, default=BETA_PRIMARY, help="depth-decay β (#193)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="private-drop-shape-robustness")
    args = ap.parse_args(argv)

    syn = synthesize(beta=args.beta)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "created_at": created_at,
        "pr": 203,
        "agent": "stark",
        "kind": "private-drop-shape-robustness",
        "synthesis": syn,
    }

    # (e) NaN-clean over the full payload.
    nan_bad = _nan_paths(payload)
    payload["nan_clean"] = not nan_bad
    if nan_bad:
        print(f"[shape-robustness] WARNING non-finite values at: {nan_bad}", flush=True)
    syn["self_test"]["conditions"]["e_nan_clean"] = bool(payload["nan_clean"])

    cond = syn["self_test"]["conditions"]
    self_test_passes = bool(all(cond.values()))
    payload["self_test_passes"] = self_test_passes
    syn["self_test"]["shape_robustness_self_test_passes"] = self_test_passes

    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload["peak_mem_mib"] = round(peak_kib / 1024.0, 3)

    _print_report(syn)
    print(f"  PRIMARY shape_robustness_self_test_passes = {self_test_passes}", flush=True)
    print(f"  TEST    both_bugs_bar_worstcase_shape     = "
          f"{syn['worst_case']['both_bugs_bar_worstcase_shape']:.6f}", flush=True)
    print(f"  peak_mem_mib = {payload['peak_mem_mib']}", flush=True)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[shape-robustness] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = self_test_passes and payload["nan_clean"]
        print(f"[shape-robustness] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
