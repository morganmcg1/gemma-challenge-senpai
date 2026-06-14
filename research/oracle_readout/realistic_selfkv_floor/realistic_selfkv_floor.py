#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Realistic self-KV E[T] floor (PR #178) — graded recovery curve, CPU-only.

denken #172 (`gh8pa4f3`) triple-confirmed the descent central **E[T]=5.0564** but left a
BINARY caveat: the adversarial self-KV-starvation floor (openevolve cause #2, 100%
depth>0 starvation) is **E[T]=3.5346 → ~363 TPS, FAILS 500**. So the descent-only 520
projection rests on cause #2 being a *fixable build defect*, and #172 could only say
"fixable → 520, unfixable → 363." That binary is too coarse for the launch reviewer:
depth>0 self-KV is **partially** recovered in reality, not all-or-nothing.

This PR converts #172's binary floor into a **graded realistic-recovery curve E[T](λ)**,
anchored to live evidence. It is a *synthesis*: it IMPORTS denken #172's descent E[T]-DP
machinery (`et_backward` / `et_pathenum` / the committed endpoint spines) and openevolve's
liveprobe numbers, and propagates them analytically. It does **not** re-derive 5.0564,
3.5346, 5.2070, K_cal, or the step. No GPU / vLLM / HF Job / submission / served-file
change. BASELINE stays 481.53. Greedy untouched. Adds **0 TPS** — it grades the NUMERATOR
caveat the descent-only path hangs on.

------------------------------------------------------------------------------
(1) PER-DEPTH SELF-KV RECOVERY MAP  q_d(λ_d)
------------------------------------------------------------------------------
λ_d ∈ [0,1] = fraction of the depth-d self-KV deficit recovered. Linear per-depth map
between two COMMITTED #172 endpoint spines:

    q_d(λ_d) = (1 − λ_d)·q_floor[d]  +  λ_d·q_full[d]

  * q_full  (λ=1, full self-KV)  = #172 ``spine_central`` = [0.679, rising q_deployed[2..]]
            → ``et_backward(q_full)`` == #172 central **5.0564**.
  * q_floor (λ=0, full self-KV starvation) = #172 ``q_meas_cond`` = the oracle's MEASURED
            declining conditional ladder [0.674, 0.519, 0.580, 0.645, 0.679, 0.674, 0.617]
            → ``et_backward(q_floor)`` == #172 adversarial floor **3.5346**.

Both endpoints reproduce #172 EXACTLY (``endpoints_reproduce``). Unlike #172's λ-knob —
which pinned depth-1 at 0.679 and only swept depth≥2, so its λ=0 was 3.5445, NOT the true
floor — this map interpolates depth-1 too (0.674↔0.679), so all-λ=0 reproduces 3.5346.

------------------------------------------------------------------------------
(2) ANCHOR λ TO THE LIVE LIVEPROBE (openevolve, board import)
------------------------------------------------------------------------------
The liveprobe measured the depth-1 deficit on the as-built stack:
``walk_topw0_hit=0.6927`` (tree-walk depth-1 top-1 hit) **<** ``linear_top1=0.7287``
(linear chain depth-1 = full self-KV). The recovery fraction is the fraction of the
maximum self-KV deficit the as-built stack has CLOSED:

    λ̂_1 = (walk_topw0_hit − q_floor_d1) / (linear_top1 − q_floor_d1)
         = (0.6927 − 0.674) / (0.7287 − 0.674)  ≈  0.342

q_floor_d1 = 0.674 is the oracle's self-KV-starved depth-1 (the λ=0 floor); linear_top1 is
the liveprobe's own full-self-KV reference. λ̂_1 is a dimensionless deficit-closure fraction
that transfers onto the model's per-depth segments. The model's depth-1 full endpoint
(0.679, BUG-1-unfixed descent-only) is held distinct from the liveprobe's full (0.7287,
BUG-1-fixed) — that gap is the separate BUG-1 axis, NOT part of λ.

``main0_accept=0.4974`` and ``tok/step=2.583`` are the as-built *realized* (compounded)
rates — both BELOW the λ=0 floor 3.5346 because the as-built walk also lacks the descent
re-seeding topology (consistent with #172 ``ORACLE_E_T``=2.621). They are reported as
context, NOT used as the conditional λ anchor.

Deeper-depth profile (only depth-1 is measured — the rest is MODELLED, stated):
  * **constant-λ (PRIMARY):** λ_d = λ̂_1 ∀d. Minimal assumption — carry the one measured
    point flat. This is the *optimistic-among-realistic* choice (self-KV starvation in fact
    worsens with depth), so a miss here is robust to the assumption.
  * **geometric decay (conservative sensitivity):** λ_d = λ̂_1·γ^(d−1), γ<1, since deeper
    nodes accumulate more missing self-context. Reported as a lower band.

------------------------------------------------------------------------------
(3) REALISTIC-FLOOR E[T] + CLEAR-500 VERDICT
------------------------------------------------------------------------------
official = K_cal·(E[T]/step)·τ ; K_cal=125.268, step=1.2182, τ∈{1.0 central, 0.9924
conservative}. clear-500 bar E[T] = 500·step/(K_cal·τ). Report ``descent_only_realistic
_floor_E_T`` (TEST, constant-λ at λ̂), its TPS + clear-500 verdict at BOTH τ corners, the
full E_T(λ) curve, and the **λ-threshold λ*** (bisection) — the minimum self-KV recovery
the build must achieve to clear 500.

------------------------------------------------------------------------------
(4) BOTH-BUGS CROSS-CHECK + HAND-OFF
------------------------------------------------------------------------------
Repeat at the both-bugs anchor (depth-1 fixed 0.7287 → 5.2070 at λ=1). The SAME self-KV λ
governs depth≥2 (BUG-2 recovery is a kernel property shared by both paths; the both-bugs
delta is only the depth-1 BUG-1 fix). Report both-bugs realistic E[T], clear-500 verdict,
and λ*_bb — the safer-first-shot threshold for fern #174's GO path.

------------------------------------------------------------------------------
(5) SELF-TEST (PRIMARY) ``realistic_selfkv_floor_self_test_passes``
------------------------------------------------------------------------------
(a) endpoints reproduce 5.0564 / 3.5346; (b) E_T(λ) monotone in λ and bracketed by the two
endpoints; (c) λ̂ ∈ [0,1] and its clear-500 verdict explicit at both τ corners; (d) λ*
reported; (e) NaN-clean.

Run:
    python -m research.oracle_readout.realistic_selfkv_floor.realistic_selfkv_floor \
        --self-test --wandb-name denken/realistic-selfkv-floor \
        --wandb-group descent-realistic-selfkv-floor
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

# --------------------------------------------------------------------------- #
# Import denken #172's descent E[T]-DP machinery (do NOT re-derive).
# --------------------------------------------------------------------------- #
_D172_PATH = REPO_ROOT / "research/validity/descent_et_audit/descent_et_dp_audit.py"


def _import_d172():
    spec = importlib.util.spec_from_file_location("descent_et_dp_audit", _D172_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import #172 module at {_D172_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D172 = _import_d172()

# ---- launch composition constants (committed; imported from #172) ---- #
K_CAL = D172.K_CAL                  # ubel #148 / #100
STEP = D172.STEP_OVERLAP            # 1.2182 lawine #168 launch-realized step
TAU_CENTRAL = 1.0
TAU_CONSERVATIVE = 0.9924           # conservative-τ corner
TAU_CORNERS = (("tau_central_1p0", TAU_CENTRAL), ("tau_conservative_0p9924", TAU_CONSERVATIVE))
TARGET_OFFICIAL = D172.TARGET_OFFICIAL  # 500.0

# ---- openevolve liveprobe (board import; the measured depth-1 anchor) ---- #
LIVEPROBE_WALK_TOPW0_HIT = 0.6927   # as-built tree-walk depth-1 top-1 hit
LIVEPROBE_LINEAR_TOP1 = 0.7287      # linear chain depth-1 = full self-KV reference
LIVEPROBE_MAIN0_ACCEPT = 0.4974     # realized (compounded) main-path accept — context only
LIVEPROBE_TOK_PER_STEP = 2.583      # realized as-built walk E[T] — context only

# ---- committed #172 endpoints (cross-check targets) ---- #
IMPORTED_CENTRAL_5p0564 = D172.IMPORTED_DESCENT_ONLY_0679   # 5.056404568844709
IMPORTED_FLOOR_3p5346 = D172.IMPORTED_CONFIG_C              # 3.534580633373862
IMPORTED_BOTH_BUGS_5p2070 = D172.IMPORTED_BOTH_BUGS         # 5.206954309441963

TOL_ENDPOINT = 1e-6        # endpoints must reproduce #172 to this tolerance.
GEOMETRIC_GAMMAS = (1.0, 0.9, 0.8, 0.7)   # constant-λ (1.0) + conservative decay band.


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# Endpoint spines + per-depth λ interpolation.
# --------------------------------------------------------------------------- #
def build_endpoints(anchors: dict[str, Any]) -> dict[str, Any]:
    """Two committed #172 endpoint spines + tree structure (imported, not re-derived)."""
    parent = anchors["parent"]
    children, depth = D172.build_children(parent)
    q_deployed = anchors["q_deployed"]
    q_meas_cond = D172.cum_to_conditional(anchors["oracle_cum_ladder"])

    # λ=1 full self-KV = #172 spine_central (depth-1 0.679, rising deep spine).
    q_full = list(q_deployed)
    q_full[0] = D172.ORACLE_DEPTH1_ALT          # 0.679
    # λ=0 full self-KV starvation = #172 q_meas_cond (declining incl depth-1 0.674).
    q_floor = list(q_meas_cond)
    horizon = max(len(q_full), len(q_floor))
    return {
        "parent": parent,
        "children": children,
        "depth": depth,
        "rho_cond": anchors["rho_cond"],
        "W": D172.W_DEFAULT,
        "q_full": q_full,
        "q_floor": q_floor,
        "q_deployed": list(q_deployed),
        "horizon": horizon,
    }


def _at(vec: list[float], i: int) -> float:
    return vec[i] if i < len(vec) else vec[-1]


def spine_from_profile(ep: dict[str, Any], lam_per_depth: list[float],
                       q_floor: list[float], q_full: list[float]) -> list[float]:
    """q_d(λ_d) = (1−λ_d)·q_floor[d] + λ_d·q_full[d], per depth-index (0 == depth 1)."""
    H = ep["horizon"]
    return [(1.0 - lam_per_depth[d]) * _at(q_floor, d) + lam_per_depth[d] * _at(q_full, d)
            for d in range(H)]


def et_of_spine(ep: dict[str, Any], spine: list[float]) -> float:
    return D172.et_backward(ep["parent"], ep["children"], ep["depth"], spine,
                            ep["rho_cond"], ep["W"])


def et_pathenum_of_spine(ep: dict[str, Any], spine: list[float]) -> float:
    return D172.et_pathenum(ep["children"], ep["depth"], spine, ep["rho_cond"], ep["W"])


def constant_lambda(H: int, lam: float) -> list[float]:
    return [lam] * H


def geometric_lambda(H: int, lam1: float, gamma: float) -> list[float]:
    """λ_d = λ1·γ^(d−1); index 0 == depth 1, so exponent == index."""
    return [lam1 * (gamma ** d) for d in range(H)]


# --------------------------------------------------------------------------- #
# Propagation E[T] -> official TPS + clear-500.
# --------------------------------------------------------------------------- #
def official_tps(et: float, tau: float) -> float:
    return D172.official_tps(et, STEP, K_CAL, tau)


def clear500_bar(tau: float) -> float:
    return D172.clear500_bar(STEP, K_CAL, tau, TARGET_OFFICIAL)


def propagate(et: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tag, tau in TAU_CORNERS:
        bar = clear500_bar(tau)
        tps = official_tps(et, tau)
        out[tag] = {
            "tau": tau,
            "official_tps": tps,
            "clear500_bar_et": bar,
            "clears_500": bool(tps >= TARGET_OFFICIAL),
            "et_margin_over_bar": et - bar,
            "tps_margin_over_500": tps - TARGET_OFFICIAL,
        }
    return out


def lambda_star(et_at_lambda: Callable[[float], float], tau: float) -> float:
    """Minimum constant-λ that clears 500 at this τ corner (bisection)."""
    bar = clear500_bar(tau)
    lo, hi = 0.0, 1.0
    if et_at_lambda(hi) < bar:
        return float("nan")          # cannot clear even fully recovered.
    if et_at_lambda(lo) >= bar:
        return 0.0                   # clears even at the floor.
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if et_at_lambda(mid) < bar:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize(anchors: dict[str, Any]) -> dict[str, Any]:
    ep = build_endpoints(anchors)
    H = ep["horizon"]
    q_full, q_floor = ep["q_full"], ep["q_floor"]

    # ---------- (1) endpoint reproduction ---------- #
    et_full = et_of_spine(ep, q_full)
    et_floor = et_of_spine(ep, q_floor)
    endpoints_reproduce = bool(
        abs(et_full - IMPORTED_CENTRAL_5p0564) <= TOL_ENDPOINT
        and abs(et_floor - IMPORTED_FLOOR_3p5346) <= TOL_ENDPOINT
    )

    # descent-only constant-λ E[T](λ).
    def et_descent_const(lam: float) -> float:
        return et_of_spine(ep, spine_from_profile(ep, constant_lambda(H, lam), q_floor, q_full))

    # ---------- (2) liveprobe λ̂ ---------- #
    q_floor_d1 = q_floor[0]                       # 0.674 (oracle self-KV-starved depth-1)
    lam_hat_1 = (LIVEPROBE_WALK_TOPW0_HIT - q_floor_d1) / (LIVEPROBE_LINEAR_TOP1 - q_floor_d1)
    lam_hat_in_unit = bool(0.0 <= lam_hat_1 <= 1.0)

    # primary profile = constant-λ at λ̂_1; conservative band = geometric decay.
    lambda_profiles: dict[str, dict[str, Any]] = {}
    for gamma in GEOMETRIC_GAMMAS:
        lam_list = (constant_lambda(H, lam_hat_1) if gamma == 1.0
                    else geometric_lambda(H, lam_hat_1, gamma))
        spine = spine_from_profile(ep, lam_list, q_floor, q_full)
        et = et_of_spine(ep, spine)
        key = "constant" if gamma == 1.0 else f"geom_gamma_{gamma:.2f}".replace(".", "p")
        lambda_profiles[key] = {
            "gamma": gamma,
            "lambda_per_depth": lam_list,
            "E_T": et,
            "official_tps_tau1": official_tps(et, TAU_CENTRAL),
            "official_tps_tau_cons": official_tps(et, TAU_CONSERVATIVE),
            "clears_500_tau1": bool(official_tps(et, TAU_CENTRAL) >= TARGET_OFFICIAL),
            "clears_500_tau_cons": bool(official_tps(et, TAU_CONSERVATIVE) >= TARGET_OFFICIAL),
        }

    descent_only_realistic_floor_E_T = lambda_profiles["constant"]["E_T"]   # TEST (primary profile)
    prop_realistic = propagate(descent_only_realistic_floor_E_T)
    descent_only_clears_500_realistic = {
        tag: prop_realistic[tag]["clears_500"] for tag, _ in TAU_CORNERS
    }

    # ---------- (3) E_T(λ) curve + λ-threshold ---------- #
    lam_grid = [0.0, 0.1, 0.2, 0.25, round(lam_hat_1, 5), 0.3, 0.4, 0.5, 0.6, 0.7, 0.75,
                0.8, 0.9, 0.95, 1.0]
    lam_grid = sorted(set(lam_grid))
    et_curve = []
    prev = None
    monotone = True
    for lam in lam_grid:
        et = et_descent_const(lam)
        if prev is not None and et < prev - 1e-12:
            monotone = False
        prev = et
        et_curve.append({
            "lambda": lam,
            "E_T": et,
            "official_tps_tau1": official_tps(et, TAU_CENTRAL),
            "official_tps_tau_cons": official_tps(et, TAU_CONSERVATIVE),
        })
    bracketed = bool(all(et_floor - 1e-9 <= row["E_T"] <= et_full + 1e-9 for row in et_curve))

    lam_star = {tag: lambda_star(et_descent_const, tau) for tag, tau in TAU_CORNERS}

    # ---------- (4) both-bugs cross-check ---------- #
    # depth-1 fixed at 0.7287 (BUG-1 fixed); depth≥2 governed by the SAME self-KV λ.
    q_full_bb = list(ep["q_deployed"])                       # depth-1 0.7287, rising
    q_floor_bb = list(q_floor); q_floor_bb[0] = ep["q_deployed"][0]   # depth-1 0.7287, declining
    et_bb_full = et_of_spine(ep, q_full_bb)
    et_bb_floor = et_of_spine(ep, q_floor_bb)

    def et_bb_const(lam: float) -> float:
        return et_of_spine(ep, spine_from_profile(ep, constant_lambda(H, lam), q_floor_bb, q_full_bb))

    et_bb_realistic = et_bb_const(lam_hat_1)
    prop_bb_realistic = propagate(et_bb_realistic)
    bothbugs_clears_500_realistic = {
        tag: prop_bb_realistic[tag]["clears_500"] for tag, _ in TAU_CORNERS
    }
    lam_star_bb = {tag: lambda_star(et_bb_const, tau) for tag, tau in TAU_CORNERS}
    # robust across realistic λ range == clears at the both-bugs floor (λ=0).
    bothbugs_robust_across_realistic = {
        tag: bool(official_tps(et_bb_floor, tau) >= TARGET_OFFICIAL) for tag, tau in TAU_CORNERS
    }

    # ---------- cross-method M1==M2 spot-check at the realistic anchor ---------- #
    realistic_spine = spine_from_profile(ep, constant_lambda(H, lam_hat_1), q_floor, q_full)
    xmethod_resid = abs(et_of_spine(ep, realistic_spine) - et_pathenum_of_spine(ep, realistic_spine))

    # ---------- (5) self-test (PRIMARY) ---------- #
    cond_endpoints = endpoints_reproduce
    cond_monotone_bracketed = bool(monotone and bracketed)
    cond_lam_hat_unit_and_verdict = bool(
        lam_hat_in_unit
        and all(isinstance(v, bool) for v in descent_only_clears_500_realistic.values())
    )
    cond_lam_star_reported = bool(all(_finite(v) for v in lam_star.values()))
    self_test_passes = bool(
        cond_endpoints and cond_monotone_bracketed and cond_lam_hat_unit_and_verdict
        and cond_lam_star_reported
    )

    handoff = _handoff_line(
        lam_hat=lam_hat_1,
        et_realistic=descent_only_realistic_floor_E_T,
        tps_realistic=prop_realistic["tau_central_1p0"]["official_tps"],
        clears=descent_only_clears_500_realistic["tau_central_1p0"],
        margin=prop_realistic["tau_central_1p0"]["tps_margin_over_500"],
        lam_star=lam_star["tau_central_1p0"],
        bb_lam_star=lam_star_bb["tau_central_1p0"],
        bb_tps=prop_bb_realistic["tau_central_1p0"]["official_tps"],
        bb_clears=bothbugs_clears_500_realistic["tau_central_1p0"],
    )

    return {
        "self_test": {
            "realistic_selfkv_floor_self_test_passes": self_test_passes,
            "conditions": {
                "endpoints_reproduce_5p0564_3p5346": cond_endpoints,
                "E_T_monotone_and_bracketed": cond_monotone_bracketed,
                "lambda_hat_in_unit_and_clear500_explicit": cond_lam_hat_unit_and_verdict,
                "lambda_star_reported": cond_lam_star_reported,
            },
        },
        "endpoints": {
            "endpoints_reproduce": endpoints_reproduce,
            "q_full_lambda1": q_full,
            "q_floor_lambda0": q_floor,
            "E_T_full_lambda1": et_full,
            "E_T_floor_lambda0": et_floor,
            "imported_central_5p0564": IMPORTED_CENTRAL_5p0564,
            "imported_floor_3p5346": IMPORTED_FLOOR_3p5346,
            "resid_full_vs_imported": abs(et_full - IMPORTED_CENTRAL_5p0564),
            "resid_floor_vs_imported": abs(et_floor - IMPORTED_FLOOR_3p5346),
        },
        "liveprobe_anchor": {
            "walk_topw0_hit": LIVEPROBE_WALK_TOPW0_HIT,
            "linear_top1": LIVEPROBE_LINEAR_TOP1,
            "q_floor_depth1": q_floor_d1,
            "lambda_hat_depth1": lam_hat_1,
            "lambda_hat_in_unit": lam_hat_in_unit,
            "lambda_hat_formula": "(walk_topw0_hit - q_floor_d1)/(linear_top1 - q_floor_d1)",
            "main0_accept_context": LIVEPROBE_MAIN0_ACCEPT,
            "tok_per_step_context": LIVEPROBE_TOK_PER_STEP,
            "lambda_profile_primary": "constant-lambda (lam_d = lambda_hat_1 for all d)",
            "lambda_profiles": lambda_profiles,
            "xmethod_resid_M1_M2_at_anchor": xmethod_resid,
        },
        "realistic_floor": {
            "descent_only_realistic_floor_E_T": descent_only_realistic_floor_E_T,
            "propagate": prop_realistic,
            "descent_only_clears_500_realistic": descent_only_clears_500_realistic,
            "E_T_lambda_curve": et_curve,
            "lambda_star_crossing_500": lam_star,
            "monotone": monotone,
            "bracketed_by_endpoints": bracketed,
        },
        "both_bugs_crosscheck": {
            "q_full_bb_lambda1": q_full_bb,
            "q_floor_bb_lambda0": q_floor_bb,
            "E_T_bb_full_lambda1": et_bb_full,
            "E_T_bb_floor_lambda0": et_bb_floor,
            "E_T_bb_realistic": et_bb_realistic,
            "propagate_bb_realistic": prop_bb_realistic,
            "bothbugs_clears_500_realistic": bothbugs_clears_500_realistic,
            "lambda_star_bb_crossing_500": lam_star_bb,
            "bothbugs_robust_across_realistic_range": bothbugs_robust_across_realistic,
            "imported_both_bugs_5p2070": IMPORTED_BOTH_BUGS_5p2070,
            "resid_bb_full_vs_imported": abs(et_bb_full - IMPORTED_BOTH_BUGS_5p2070),
        },
        "descent_only_realistic_floor_E_T": descent_only_realistic_floor_E_T,
        "lambda_hat_depth1": lam_hat_1,
        "verdict": _verdict(descent_only_clears_500_realistic, bothbugs_clears_500_realistic),
        "handoff_line": handoff,
        "composition": {
            "K_cal": K_CAL, "step": STEP,
            "tau_central": TAU_CENTRAL, "tau_conservative": TAU_CONSERVATIVE,
            "clear500_bar_tau1": clear500_bar(TAU_CENTRAL),
            "clear500_bar_tau_cons": clear500_bar(TAU_CONSERVATIVE),
        },
    }


def _verdict(descent_clears: dict[str, bool], bb_clears: dict[str, bool]) -> str:
    d = any(descent_clears.values())
    b = any(bb_clears.values())
    if d and b:
        return "REALISTIC-FLOOR-CLEARS-BOTH"
    if b and not d:
        return "REALISTIC-FLOOR-BOTHBUGS-ONLY"
    return "REALISTIC-FLOOR-MISSES-BOTH"


def _handoff_line(*, lam_hat: float, et_realistic: float, tps_realistic: float, clears: bool,
                  margin: float, lam_star: float, bb_lam_star: float, bb_tps: float,
                  bb_clears: bool) -> str:
    verb = "clears" if clears else "misses"
    bbverb = "clears" if bb_clears else "misses"
    return (
        f"GRADED REALISTIC-FLOOR (replaces #172's binary fixable/unfixable): descent-only "
        f"clears 500 iff built deep-spine self-KV recovery λ ≥ {lam_star:.3f}; the "
        f"liveprobe-anchored realistic estimate is λ̂={lam_hat:.3f} → E[T]={et_realistic:.4f} "
        f"→ {verb} 500 by {abs(margin):.0f} TPS ({tps_realistic:.1f}). Both-bugs (depth-1 "
        f"BUG-1 fix on top) needs λ ≥ {bb_lam_star:.3f} and at λ̂ {bbverb} 500 "
        f"({bb_tps:.1f}) — the safer first shot (lower threshold) but NOT robust at the "
        f"measured floor; the binding constraint for BOTH paths is self-KV recovery, not "
        f"BUG-1. HONEST SCOPE: liveprobe is one measured depth-1 point; the depth-1→depth>0 "
        f"transfer and deeper-depth λ are MODELLED (constant-λ primary; geometric-decay only "
        f"lowers E[T]). The true closure remains land #71's built-kernel ladder q[2..9]. "
        f"NOT open2. NOT a launch."
    )


# --------------------------------------------------------------------------- #
# W&B logging (mirrors #172 / scripts/wandb_logging.py; never fatal).
# --------------------------------------------------------------------------- #
def _maybe_log_wandb(args, payload: dict, anchors: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb,
            init_wandb_run,
            log_json_artifact,
            log_summary,
        )
    except Exception as exc:
        print(f"[realistic-floor] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    run = init_wandb_run(
        job_type="realistic-selfkv-floor",
        agent="denken",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["realistic-selfkv-floor", "validity-gate", "numerator-bound", "graded-floor"],
        config={
            "K_cal": K_CAL,
            "step": STEP,
            "tau_central": TAU_CENTRAL,
            "tau_conservative": TAU_CONSERVATIVE,
            "n_nodes": anchors["n_nodes"],
            "imported_central_5p0564": IMPORTED_CENTRAL_5p0564,
            "imported_floor_3p5346": IMPORTED_FLOOR_3p5346,
            "imported_both_bugs_5p2070": IMPORTED_BOTH_BUGS_5p2070,
            "liveprobe_walk_topw0_hit": LIVEPROBE_WALK_TOPW0_HIT,
            "liveprobe_linear_top1": LIVEPROBE_LINEAR_TOP1,
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[realistic-floor] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    st = syn["self_test"]
    rf = syn["realistic_floor"]
    bb = syn["both_bugs_crosscheck"]
    la = syn["liveprobe_anchor"]
    ep = syn["endpoints"]
    summary: dict[str, Any] = {
        # PRIMARY + TEST
        "realistic_selfkv_floor_self_test_passes": int(bool(st["realistic_selfkv_floor_self_test_passes"])),
        "descent_only_realistic_floor_E_T": rf["descent_only_realistic_floor_E_T"],
        # anchor
        "lambda_hat_depth1": la["lambda_hat_depth1"],
        "endpoints_reproduce": int(bool(ep["endpoints_reproduce"])),
        "E_T_full_lambda1": ep["E_T_full_lambda1"],
        "E_T_floor_lambda0": ep["E_T_floor_lambda0"],
        # descent-only realistic clear-500
        "descent_realistic_tps_tau1": rf["propagate"]["tau_central_1p0"]["official_tps"],
        "descent_realistic_tps_tau_cons": rf["propagate"]["tau_conservative_0p9924"]["official_tps"],
        "descent_clears_500_tau1": int(bool(rf["descent_only_clears_500_realistic"]["tau_central_1p0"])),
        "descent_clears_500_tau_cons": int(bool(rf["descent_only_clears_500_realistic"]["tau_conservative_0p9924"])),
        "lambda_star_tau1": rf["lambda_star_crossing_500"]["tau_central_1p0"],
        "lambda_star_tau_cons": rf["lambda_star_crossing_500"]["tau_conservative_0p9924"],
        "descent_tps_margin_over_500_tau1": rf["propagate"]["tau_central_1p0"]["tps_margin_over_500"],
        # both-bugs cross-check
        "bothbugs_realistic_E_T": bb["E_T_bb_realistic"],
        "bothbugs_realistic_tps_tau1": bb["propagate_bb_realistic"]["tau_central_1p0"]["official_tps"],
        "bothbugs_clears_500_tau1": int(bool(bb["bothbugs_clears_500_realistic"]["tau_central_1p0"])),
        "bothbugs_clears_500_tau_cons": int(bool(bb["bothbugs_clears_500_realistic"]["tau_conservative_0p9924"])),
        "lambda_star_bb_tau1": bb["lambda_star_bb_crossing_500"]["tau_central_1p0"],
        "bothbugs_robust_across_realistic_tau1": int(bool(bb["bothbugs_robust_across_realistic_range"]["tau_central_1p0"])),
        # geometric-decay conservative band
        "geom_gamma0p9_E_T": la["lambda_profiles"]["geom_gamma_0p90"]["E_T"],
        "geom_gamma0p8_E_T": la["lambda_profiles"]["geom_gamma_0p80"]["E_T"],
        # bars
        "clear500_bar_tau1": syn["composition"]["clear500_bar_tau1"],
        "clear500_bar_tau_cons": syn["composition"]["clear500_bar_tau_cons"],
        "xmethod_resid_M1_M2_at_anchor": la["xmethod_resid_M1_M2_at_anchor"],
        "K_cal": K_CAL,
        "verdict_misses_both": int(syn["verdict"] == "REALISTIC-FLOOR-MISSES-BOTH"),
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items() if not (isinstance(v, float) and not math.isfinite(v))}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="realistic_selfkv_floor_result", artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[realistic-floor] wandb logged: {summary}", flush=True)


# --------------------------------------------------------------------------- #
# Reporting + CLI.
# --------------------------------------------------------------------------- #
def _assert_nan_clean(payload: dict, path: str = "result") -> list[str]:
    bad: list[str] = []

    def walk(node: Any, p: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{p}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f"{p}[{i}]")
        elif isinstance(node, float) and not math.isfinite(node):
            bad.append(p)

    walk(payload, path)
    return bad


def _print_report(syn: dict) -> None:
    ep, la, rf, bb, st = (syn["endpoints"], syn["liveprobe_anchor"], syn["realistic_floor"],
                          syn["both_bugs_crosscheck"], syn["self_test"])
    print("\n" + "=" * 84, flush=True)
    print("REALISTIC SELF-KV E[T] FLOOR (PR #178) — graded recovery curve", flush=True)
    print("=" * 84, flush=True)
    print(f"  (1) ENDPOINTS reproduce #172: full(λ=1)={ep['E_T_full_lambda1']:.6f} "
          f"(5.0564, resid {ep['resid_full_vs_imported']:.1e})  "
          f"floor(λ=0)={ep['E_T_floor_lambda0']:.6f} (3.5346, resid "
          f"{ep['resid_floor_vs_imported']:.1e})  reproduce={ep['endpoints_reproduce']}", flush=True)
    print("-" * 84, flush=True)
    print(f"  (2) LIVEPROBE ANCHOR  λ̂_1 = (walk {la['walk_topw0_hit']} − floor "
          f"{la['q_floor_depth1']}) / (linear {la['linear_top1']} − floor "
          f"{la['q_floor_depth1']}) = {la['lambda_hat_depth1']:.5f}", flush=True)
    print(f"      (context: main0_accept={la['main0_accept_context']}, "
          f"tok/step={la['tok_per_step_context']} — realized, below the λ=0 floor)", flush=True)
    print("      deeper-depth profiles (λ̂ carried forward):", flush=True)
    for key, pr in la["lambda_profiles"].items():
        print(f"        {key:<16} γ={pr['gamma']:.2f}  E[T]={pr['E_T']:.4f}  "
              f"TPS@τ1={pr['official_tps_tau1']:.1f}  clears={pr['clears_500_tau1']}", flush=True)
    print("-" * 84, flush=True)
    print(f"  (3) REALISTIC FLOOR (constant-λ PRIMARY)  E[T]="
          f"{rf['descent_only_realistic_floor_E_T']:.4f}", flush=True)
    for tag, _ in TAU_CORNERS:
        p = rf["propagate"][tag]
        print(f"        {tag:<24} TPS={p['official_tps']:.2f}  bar={p['clear500_bar_et']:.4f}  "
              f"clears500={p['clears_500']}  margin={p['tps_margin_over_500']:+.2f}", flush=True)
    print(f"      λ* (min recovery to clear 500): τ1={rf['lambda_star_crossing_500']['tau_central_1p0']:.4f}  "
          f"τ_cons={rf['lambda_star_crossing_500']['tau_conservative_0p9924']:.4f}", flush=True)
    print("      E_T(λ) curve:", flush=True)
    for row in rf["E_T_lambda_curve"]:
        print(f"        λ={row['lambda']:.5f}  E[T]={row['E_T']:.4f}  TPS@τ1={row['official_tps_tau1']:.1f}", flush=True)
    print("-" * 84, flush=True)
    print(f"  (4) BOTH-BUGS CROSS-CHECK  full(λ=1)={bb['E_T_bb_full_lambda1']:.4f} (5.2070)  "
          f"floor(λ=0)={bb['E_T_bb_floor_lambda0']:.4f}", flush=True)
    print(f"      realistic λ̂: E[T]={bb['E_T_bb_realistic']:.4f}  "
          f"TPS@τ1={bb['propagate_bb_realistic']['tau_central_1p0']['official_tps']:.2f}  "
          f"clears500={bb['bothbugs_clears_500_realistic']['tau_central_1p0']}", flush=True)
    print(f"      λ*_bb: τ1={bb['lambda_star_bb_crossing_500']['tau_central_1p0']:.4f}  "
          f"τ_cons={bb['lambda_star_bb_crossing_500']['tau_conservative_0p9924']:.4f}  "
          f"robust-across-range(τ1)={bb['bothbugs_robust_across_realistic_range']['tau_central_1p0']}", flush=True)
    print("-" * 84, flush=True)
    print(f"  (5) PRIMARY realistic_selfkv_floor_self_test_passes = "
          f"{st['realistic_selfkv_floor_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"          - {k}: {v}", flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print("=" * 84, flush=True)
    print(f"\n  HAND-OFF: {syn['handoff_line']}\n", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--bug2-anchor", type=Path, default=D172.DEFAULT_BUG2_ANCHOR)
    ap.add_argument("--topo-json", type=Path, default=D172.DEFAULT_TOPO_JSON)
    ap.add_argument("--accept-json", type=Path, default=D172.DEFAULT_ACCEPT_JSON)
    ap.add_argument("--rankcov-json", type=Path, default=D172.DEFAULT_RANKCOV_JSON)
    ap.add_argument("--decomp-json", type=Path, default=D172.DEFAULT_DECOMP_JSON)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="descent-realistic-selfkv-floor")
    args = ap.parse_args(argv)

    anchors = D172.load_anchors(
        args.bug2_anchor, args.topo_json, args.accept_json, args.rankcov_json, args.decomp_json
    )
    syn = synthesize(anchors)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at,
        "pr": 178,
        "agent": "denken",
        "kind": "realistic-selfkv-floor",
        "anchors": {k: v for k, v in anchors.items() if k != "_paths"},
        "anchor_paths": anchors.get("_paths"),
        "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }

    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[realistic-floor] WARNING non-finite values at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "realistic_selfkv_floor_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[realistic-floor] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload, anchors)

    if args.self_test:
        ok = syn["self_test"]["realistic_selfkv_floor_self_test_passes"] and payload["nan_clean"]
        print(f"[realistic-floor] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
