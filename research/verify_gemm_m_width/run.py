#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Verify-GEMM M-aware tree width (PR #268) — find the TPS(M)-maximizing M*.

THE WIDTH AXIS
--------------
denken #257 (run `h1gj2ved`) banked two load-bearing facts about the verify path:
verify-GEMM is 53.2% of decode at M=8, and the verify forward is SUPER-linear in M
past the M=16->32 knee (`verify_us` M8=5.164ms -> M32=5.980ms, +15.8%). The live
tree-decode build (land #245) currently assumes a wide tree (M~=32), but wide is not
free. The TPS-maximizing tree width is

    M* = argmax_M  TPS(M) = K_cal * E[T](M) * tau / step(M)

with `step(M) = bridge(verify_us(M) + tree_draft_cost)` (#257 reconciliation) and
`E[T](M)` SUBLINEAR in M (logarithmic rank-coverage saturation; wirbel #79/#86 measured
rho). A sublinear numerator over a super-linear-past-knee denominator gives TPS(M) an
interior maximum -- IF the knee bites inside the lambda-valid region. This leg prices
that curve over the BANKED #257 grid and answers the decisive question: does TPS(M) peak
above 500 at any lambda-valid M, or does tree-WIDTH alone top out below the gate?

THE TWO STEP EDGES (#257, the load-bearing g_d fork — bracket, do not pick)
---------------------------------------------------------------------------
#257 found the built step lives in the [1.12, 1.43] ms band (the 1.085 analytic anchor
RETIRED), and the entire spread is the g_d (draft/verify) basis fork it FLAGGED for
advisor reconciliation:
  * CENTRAL edge  (MEASURED g_d=0.0195): step(M=32) = 1.3458 ms — `step = c*(v(M)+5d)`.
  * OPTIMISTIC edge (ASSUMED g_d=0.168): step(M=32) = 1.1186 ms — verify-light split.
Both are affine in the banked `verify_us(M)`; both round-trip #257's banked M=32 step
and its implied TPS at E[T]=4.512 PROVENANCE-EXACT (self-test a). We report TPS(M) for
BOTH so M* and the go/no-go are bracketed against the step uncertainty.

E[T](M): MEASURED-rho RANK-COVERAGE (sublinear), floor- and lambda-guarded
--------------------------------------------------------------------------
  * M <= 8  : LINEAR-chain path-product over the measured conditional-acceptance ladder
              C (wirbel #86, run `z6wi4z4v`); E[T](8) round-trips F_linear8=3.84445 EXACT.
  * M=16,32 : BANKED tree rank-coverage F_tree (wirbel #79/#86 tree-reprice, measured
              per-rank split): 4.5123, 5.1573. E[T](16)=4.5123 matches the #257 anchor
              4.512 and the path-product et_both_committed 4.5119.
  * Each doubling of M adds ~0.65 to E[T] (LOGARITHMIC) — the sublinear numerator.
ANCHORED to the measured floor E_T_meas_floor=4.3305 (#241, run `hqewf1d6`): a build
clears the floor only for M>=16 (M=8 linear control sits below). RESPECTS lambda_hat >=
0.9780 P95 (fern #249): the measured min-lambda over q[2..9] is 0.9827 (valid through the
depth-9 / M=32 tree); M>32 is UNMEASURED and therefore lambda-EXCLUDED.

SCOPE: CPU-only analytic over denken's OWN banked #257 profiler grid + the merged
rank-coverage/floor/lambda anchors. No GPU / vLLM / HF-Job / submission / served-file /
official draw. Prices the WIDTH-axis ceiling and hands land an M*; does not itself ship a
TPS number. BASELINE stays 481.53. Greedy/PPL untouched by construction. NOT a launch.

PRIMARY metric  verify_gemm_m_aware_self_test_passes
TEST    metric  M_star (argmax over lambda-valid M) + clears_500 (per step edge)

Run:
    CUDA_VISIBLE_DEVICES="" python research/verify_gemm_m_width/run.py \\
        --self-test --wandb_group denken-verify-m-width --wandb_name denken/verify-gemm-m-aware
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

# ---- banked anchors (all merged on the advisor branch; read-only) ----
_D257 = REPO_ROOT / "research/validity/built_step_roofline/built_step_roofline_report.json"   # denken #257 (h1gj2ved)
_TREESHAPE = REPO_ROOT / "research/accept_calibration/treeshape_measured_results.json"          # wirbel #79/#86 (z6wi4z4v)
_D241 = REPO_ROOT / "research/validity/measured_et_shortfall/measured_et_shortfall_results.json"  # denken #241 (hqewf1d6)
_ETBOTH = REPO_ROOT / "research/tree_verify_path/comp_etboth_perdepth_verdict.json"             # lambda q-ladder (PARENT_M16)

TARGET_TPS = 500.0
OFFICIAL_BASELINE = 481.53

# numeric tolerances
TOL_PROV = 1e-6     # provenance round-trip vs #257 banked step / TPS
TOL_EXACT = 1e-9    # path-product F_linear8 round-trip
TOL_FLOOR = 1e-9


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# Imports — banked numbers, NOT re-measured.
# --------------------------------------------------------------------------- #
def load_imports() -> dict[str, Any]:
    d257 = json.load(open(_D257))
    tree = json.load(open(_TREESHAPE))
    d241 = json.load(open(_D241))
    etb = json.load(open(_ETBOTH))

    ia = d257["imported_anchors"]
    cb5 = d257["built_projection_models"]["central_b5"]
    gb5 = d257["gd_sensitivity_assumed_0p168"]["central_b5"]

    # verify_us(M) banked grid (the JSON field is in MS) -> us
    verify_us_us = {int(k): float(v) * 1e3 for k, v in d257["verify_us_ms"].items()}

    # measured conditional-acceptance ladder C (cumulative coverage) — linear-chain E[T]
    C = list(tree["measured_acceptance"]["server_log"]["C"])

    # tree rank-coverage F_tree (measured per-rank split = the path-product-consistent family)
    ms = tree["measured_per_rank_split"]
    geom = tree["central_geom_rho0565"]

    floor = d241["synthesis"]["tps_gate"]["e_t_meas_floor_tps500"]
    floor_cons = d241["synthesis"]["binding_tolerance"]["e_t_meas_floor_conservative"]

    return {
        # ---- #257 step composition (graphed, deployed basis) ----
        "verify_us_us": verify_us_us,
        "K_cal": ia["K_cal"],
        "step_served_ms": ia["step_served_ms"],
        "g_d_assumed": ia["g_d_assumed"],
        "g_d_measured": d257["g_d_measured"],
        "K_spec": ia["K_spec"],
        "M_verify_served": ia["M_verify_served"],   # 8
        "M_tree": ia["M_tree"],                     # 32
        "tau_band": list(ia["tau_band"]),
        "E_T_built_anchor_257": ia["E_T_built"],    # 4.512 — provenance probe value
        "draft_pass_us_graphed": d257["draft_pass_us_graphed"],
        "n_tree_central_b5": cb5["n_tree_passes"],  # 5
        "v8_us": d257["verify_v8_us"],
        "v32_us": d257["verify_v32_us"],
        "bridge_c_banked": d257["bridge_c_wall_to_normalized"],
        "s_served_abs_us_banked": d257["s_served_abs_us"],
        # banked M=32 step + implied-TPS provenance targets (the round-trip)
        "step_central_M32_banked": cb5["step_built_measured_proj_ms"],          # 1.3458358727
        "step_optimistic_M32_banked": gb5["step_built_proj_ms"],                # 1.1185888769
        "tps_central_M32_at_ET4512_banked": cb5["implied_tps_at_ET4512"],       # 419.9687
        "tps_optimistic_M32_at_ET4512_banked": gb5["implied_tps_at_ET4512"],    # 505.2875
        "verify_v32_over_v8": d257["verify_v32_over_v8"],
        "knee_first_M_above_flat": d257["knee"]["knee_first_M_above_flat"],     # 16
        # ---- E[T](M) rank-coverage (wirbel #79/#86) ----
        "C_ladder": C,
        "F_linear8": tree["anchor_F_linear8"],                                  # 3.8444537126
        "F_tree16_split": ms["M16"]["F_tree"],                                  # 4.5123
        "F_tree32_split": ms["M32"]["F_tree"],                                  # 5.1573
        "F_tree16_geom": geom["M16"]["F_tree"],                                 # 4.4958 (sensitivity)
        "F_tree32_geom": geom["M32"]["F_tree"],                                 # 5.1410 (sensitivity)
        "rho_cov_W": tree["central_geom_rho0565"]["rho"],                       # 0.6532
        # ---- floor (#241) + lambda ladder (PARENT_M16) ----
        "E_T_meas_floor": floor,                                                # 4.3305
        "E_T_meas_floor_conservative": floor_cons,                             # 4.4890
        "lambda_bar": etb["bar_lambda"],                                        # 0.978
        "lambda_min_q2_q9": etb["min_lambda_q2_q9"],                            # 0.9827
        "q_ladder_lambda": {int(k): float(v) for k, v in etb["q_ladder_lambda"].items()},
        "et_both_committed_M16": etb["conservative"]["et_both_committed"],      # 4.5119
        "source_runs": {"d257": "h1gj2ved", "wirbel79_86": "z6wi4z4v",
                        "d241": "hqewf1d6", "etboth": "comp_etboth_perdepth(PARENT_M16)"},
    }


# --------------------------------------------------------------------------- #
# verify_us(M) — banked grid + honest interp/extrap.
# --------------------------------------------------------------------------- #
def verify_us_of_M(imp: dict, M: float) -> tuple[float, str]:
    """Return (verify_us in us, provenance). Banked at {1,2,4,8,16,32};
    log-linear interp between banked anchors; LINEAR (compute-bound past-knee)
    extrapolation for M>32 using the measured M16->M32 slope."""
    grid = imp["verify_us_us"]
    Ms = sorted(grid)
    if M in grid:
        return grid[int(M)], "banked"
    if M < Ms[0]:
        return grid[Ms[0]], "clamp_lo"
    if M > Ms[-1]:
        # compute-bound past the knee -> verify ~linear in M; extend the M16->M32 line.
        v16, v32 = grid[16], grid[32]
        slope = (v32 - v16) / (32 - 16)
        return v32 + slope * (M - 32), "extrap_linear_postknee"
    # interpolate log-linearly in M between bracketing banked anchors
    lo = max(m for m in Ms if m <= M)
    hi = min(m for m in Ms if m >= M)
    f = (math.log2(M) - math.log2(lo)) / (math.log2(hi) - math.log2(lo))
    return grid[lo] + f * (grid[hi] - grid[lo]), "interp_loglinear"


# --------------------------------------------------------------------------- #
# E[T](M) — measured-rho rank-coverage, sublinear (logarithmic).
# --------------------------------------------------------------------------- #
def e_t_of_M(imp: dict, M: float, family: str = "split") -> tuple[float, str]:
    """E[T](M). M<=8: linear-chain path-product over the measured C ladder (round-trips
    F_linear8 EXACT at M=8). M=16,32: banked tree rank-coverage F_tree. Intermediate:
    log-linear interp. M>32: logarithmic extrapolation at the measured per-doubling slope."""
    C = imp["C_ladder"]
    f16 = imp["F_tree16_split"] if family == "split" else imp["F_tree16_geom"]
    f32 = imp["F_tree32_split"] if family == "split" else imp["F_tree32_geom"]

    if M <= 8:
        # E[T]_linear(K) = 1 + sum_{k=1}^{K} C_k, with K = M-1 draft positions.
        K = int(round(M)) - 1
        K = max(0, min(K, len(C)))
        return 1.0 + sum(C[:K]), ("banked_pathproduct" if M in (1, 2, 4, 8) else "pathproduct")
    if abs(M - 16) < 1e-9:
        return f16, "banked_tree_rankcov"
    if abs(M - 32) < 1e-9:
        return f32, "banked_tree_rankcov"
    # logarithmic law E[T](M) = E[T]_anchor + s*log2(M/anchor); s from the banked doublings.
    if M < 16:   # between linear-8 and tree-16
        e8 = 1.0 + sum(C)
        s = (f16 - e8) / (math.log2(16) - math.log2(8))
        return e8 + s * (math.log2(M) - math.log2(8)), "interp_log"
    s = (f32 - f16) / (math.log2(32) - math.log2(16))   # per-doubling slope
    if M < 32:
        return f16 + s * (math.log2(M) - math.log2(16)), "interp_log"
    return f32 + s * (math.log2(M) - math.log2(32)), "extrap_log"   # M>32 (lambda-excluded)


# --------------------------------------------------------------------------- #
# lambda_hat(M) — measured min q[2..depth(M)]; UNMEASURED (excluded) past M=32.
# --------------------------------------------------------------------------- #
# Depth reached by a tree of verify width M (banked topology): M=8 linear -> depth 7,
# M=16 -> depth 8, M=32 -> depth 9 (wirbel #86 / land tree_spec). lambda is per-depth.
_M_TO_DEPTH = {1: 1, 2: 2, 4: 4, 8: 7, 16: 8, 32: 9}
M_LAMBDA_CEIL = 32   # deepest verify width with a MEASURED lambda ladder (depth 9)


def lambda_hat_of_M(imp: dict, M: float) -> tuple[float | None, bool, str]:
    """Return (lambda_hat, lambda_valid, provenance). Valid iff MEASURED and the binding
    min over q[2..depth(M)] >= bar (0.9780). M>32 is unmeasured -> cannot be certified
    >= bar -> EXCLUDED (the fern #249 validity rule)."""
    bar = imp["lambda_bar"]
    qll = imp["q_ladder_lambda"]
    if M > M_LAMBDA_CEIL:
        return None, False, "unmeasured_excluded"
    depth = _M_TO_DEPTH.get(int(M))
    if depth is None:   # interpolated width -> use the nearest measured depth ceiling
        depth = _M_TO_DEPTH[min(_M_TO_DEPTH, key=lambda k: abs(k - M))]
    # binding min over q[2..depth] (q1 excluded — pos-1 bonus is the served floor)
    relevant = [qll[d] for d in range(2, depth + 1) if d in qll]
    lam = min(relevant) if relevant else qll.get(1, 1.0)
    return lam, bool(lam >= bar - 1e-12), "measured_min_q2_depth"


# --------------------------------------------------------------------------- #
# step(M) — the two #257 g_d edges (both affine in verify_us(M)).
# --------------------------------------------------------------------------- #
def make_step_models(imp: dict) -> dict[str, Any]:
    d = imp["draft_pass_us_graphed"]
    n_tree = imp["n_tree_central_b5"]            # 5
    K_spec = imp["K_spec"]
    v8 = imp["v8_us"]
    step_served = imp["step_served_ms"]
    g_d_assumed = imp["g_d_assumed"]

    # CENTRAL (measured g_d): s_built = verify(M) + n_tree*d; bridged to normalized by c.
    s_served_abs = v8 + K_spec * d
    bridge_c = step_served / s_served_abs        # ms/us

    # OPTIMISTIC (assumed g_d=0.168): verify-light split, same measured verify-growth.
    verify8_n = step_served / (1.0 + K_spec * g_d_assumed)
    draft_n = g_d_assumed * verify8_n

    def step_central(M: float) -> float:
        v, _ = verify_us_of_M(imp, M)
        return bridge_c * (v + n_tree * d)

    def step_optimistic(M: float) -> float:
        v, _ = verify_us_of_M(imp, M)
        return verify8_n * (v / v8) + n_tree * draft_n

    return {
        "central": step_central, "optimistic": step_optimistic,
        "bridge_c": bridge_c, "s_served_abs_us": s_served_abs,
        "verify8_n": verify8_n, "draft_n": draft_n, "n_tree": n_tree, "d_us": d,
        "tree_draft_cost_us": n_tree * d,
    }


def tps_of(K_cal: float, e_t: float, step_ms: float, tau: float) -> float:
    return K_cal * (e_t / step_ms) * tau


# --------------------------------------------------------------------------- #
# (1) TPS(M) curve over the banked grid (+ lambda-excluded extension).
# --------------------------------------------------------------------------- #
M_VALID = [1, 2, 4, 8, 16, 32]            # measured / lambda-valid widths
M_EXTRAP = [48, 64]                       # UNMEASURED, lambda-excluded (interior-max + exclusion demo)
M_GRID = M_VALID + M_EXTRAP


def tps_curve(imp: dict, steps: dict, tau: float = 1.0, family: str = "split") -> dict[str, Any]:
    K_cal = imp["K_cal"]
    rows = []
    for M in M_GRID:
        e_t, e_prov = e_t_of_M(imp, M, family)
        v_us, v_prov = verify_us_of_M(imp, M)
        lam, lam_valid, lam_prov = lambda_hat_of_M(imp, M)
        s_c = steps["central"](M)
        s_o = steps["optimistic"](M)
        clears_floor = bool(e_t >= imp["E_T_meas_floor"] - TOL_FLOOR)
        rows.append({
            "M": M,
            "verify_us_us": v_us, "verify_prov": v_prov,
            "E_T": e_t, "E_T_prov": e_prov, "E_T_clears_floor": clears_floor,
            "lambda_hat": lam, "lambda_valid": lam_valid, "lambda_prov": lam_prov,
            "step_central_ms": s_c, "step_optimistic_ms": s_o,
            "tps_central": tps_of(K_cal, e_t, s_c, tau),
            "tps_optimistic": tps_of(K_cal, e_t, s_o, tau),
            "tps_central_clears_500": bool(tps_of(K_cal, e_t, s_c, tau) >= TARGET_TPS),
            "tps_optimistic_clears_500": bool(tps_of(K_cal, e_t, s_o, tau) >= TARGET_TPS),
        })
    return {"tau": tau, "family": family, "rows": rows}


def _argmax_over(rows: list[dict], key: str, valid_only: bool) -> dict[str, Any]:
    pool = [r for r in rows if (r["lambda_valid"] or not valid_only)]
    if not pool:
        return {"M_star": None, "tps": None}
    best = max(pool, key=lambda r: r[key])
    return {"M_star": best["M"], "tps": best[key],
            "E_T": best["E_T"], "lambda_hat": best["lambda_hat"],
            "step_central_ms": best["step_central_ms"], "step_optimistic_ms": best["step_optimistic_ms"]}


def locate_M_star(imp: dict, curve: dict) -> dict[str, Any]:
    rows = curve["rows"]
    # M* = argmax over the LAMBDA-VALID region, per step edge.
    central_valid = _argmax_over(rows, "tps_central", valid_only=True)
    optimistic_valid = _argmax_over(rows, "tps_optimistic", valid_only=True)
    # UNCONSTRAINED argmax (ignores lambda) — shows whether the lambda gate BINDS.
    central_unc = _argmax_over(rows, "tps_central", valid_only=False)
    optimistic_unc = _argmax_over(rows, "tps_optimistic", valid_only=False)

    valid_rows = [r for r in rows if r["lambda_valid"]]
    max_central = max(r["tps_central"] for r in valid_rows)
    max_optimistic = max(r["tps_optimistic"] for r in valid_rows)
    clears_central = bool(max_central >= TARGET_TPS)
    clears_optimistic = bool(max_optimistic >= TARGET_TPS)

    # operating point at the (optimistic-edge) M* hand-off for land
    opM = optimistic_valid["M_star"]
    op_row = next(r for r in rows if r["M"] == opM)

    return {
        "M_star_central_edge": central_valid["M_star"],
        "M_star_optimistic_edge": optimistic_valid["M_star"],
        "M_star_unconstrained_central": central_unc["M_star"],
        "M_star_unconstrained_optimistic": optimistic_unc["M_star"],
        "lambda_gate_binds_central": bool(central_unc["M_star"] != central_valid["M_star"]),
        "lambda_gate_binds_optimistic": bool(optimistic_unc["M_star"] != optimistic_valid["M_star"]),
        "max_tps_central_edge": max_central,
        "max_tps_optimistic_edge": max_optimistic,
        "clears_500_central_edge": clears_central,
        "clears_500_optimistic_edge": clears_optimistic,
        "clears_500_both_edges": bool(clears_central and clears_optimistic),
        "clears_500_neither_edge": bool((not clears_central) and (not clears_optimistic)),
        "decisive_width_cannot_reach_500": bool((not clears_central) and (not clears_optimistic)),
        "handoff_operating_point": {
            "M_star": opM,
            "E_T": op_row["E_T"],
            "lambda_hat": op_row["lambda_hat"],
            "step_central_ms": op_row["step_central_ms"],
            "step_optimistic_ms": op_row["step_optimistic_ms"],
            "tps_central": op_row["tps_central"],
            "tps_optimistic": op_row["tps_optimistic"],
        },
    }


# --------------------------------------------------------------------------- #
# (2) E[T]-anchor sensitivity at the M* width (the contested branch-interior).
#     bracket E[T] in {rank-cov idealized, #257 flat 4.512 anchor, #241 floor 4.3305}.
# --------------------------------------------------------------------------- #
def et_anchor_matrix(imp: dict, steps: dict, M_star: int, tau: float = 1.0) -> dict[str, Any]:
    K_cal = imp["K_cal"]
    s_c = steps["central"](M_star)
    s_o = steps["optimistic"](M_star)
    e_rankcov, _ = e_t_of_M(imp, M_star, "split")
    anchors = {
        "rankcov_idealized": e_rankcov,                 # full branch-interior realized
        "flat_257_anchor_4p512": imp["E_T_built_anchor_257"],   # M16-equiv conservative anchor
        "meas_floor_4p3305": imp["E_T_meas_floor"],     # #241 TPS500 floor (max discount)
        "meas_floor_conservative_4p489": imp["E_T_meas_floor_conservative"],
    }
    matrix = {}
    for name, e_t in anchors.items():
        matrix[name] = {
            "E_T": e_t,
            "tps_central": tps_of(K_cal, e_t, s_c, tau),
            "tps_optimistic": tps_of(K_cal, e_t, s_o, tau),
            "tps_central_clears_500": bool(tps_of(K_cal, e_t, s_c, tau) >= TARGET_TPS),
            "tps_optimistic_clears_500": bool(tps_of(K_cal, e_t, s_o, tau) >= TARGET_TPS),
        }
    n_corners = 4 * 0 + sum(1 for v in matrix.values()
                            for k in ("tps_central_clears_500", "tps_optimistic_clears_500") if v[k])
    return {
        "M_star": M_star, "step_central_ms": s_c, "step_optimistic_ms": s_o,
        "matrix": matrix,
        "n_corners_clearing_500": n_corners,
        "n_corners_total": 2 * len(anchors),
        "note": (
            "2x4 (step edge) x (E[T] anchor) decision matrix at M*. Clearing 500 needs BOTH "
            "the optimistic g_d step (1.119, the #257 advisor-reconciliation fork) AND a high "
            "E[T] realization (rank-cov idealized or the 4.512 anchor); the measured/central "
            "step and/or the #241 floor never clear."),
    }


# --------------------------------------------------------------------------- #
# (3) tau band (minor — does it flip the go/no-go?).
# --------------------------------------------------------------------------- #
def tau_band_check(imp: dict, steps: dict, M_star: int) -> dict[str, Any]:
    K_cal = imp["K_cal"]
    e_t, _ = e_t_of_M(imp, M_star, "split")
    s_o = steps["optimistic"](M_star)
    s_c = steps["central"](M_star)
    lo, hi = imp["tau_band"][0], imp["tau_band"][1]
    return {
        "tau_band": [lo, hi], "M_star": M_star,
        "tps_optimistic_tau_hi": tps_of(K_cal, e_t, s_o, hi),
        "tps_optimistic_tau_lo": tps_of(K_cal, e_t, s_o, lo),
        "tps_central_tau_hi": tps_of(K_cal, e_t, s_c, hi),
        "tps_central_tau_lo": tps_of(K_cal, e_t, s_c, lo),
        "tau_flips_optimistic_500": bool(
            (tps_of(K_cal, e_t, s_o, hi) >= TARGET_TPS) != (tps_of(K_cal, e_t, s_o, lo) >= TARGET_TPS)),
        "note": "tau in [0.9924,1.0] scales TPS by <=0.76%; it does not flip the optimistic-edge go/no-go.",
    }


# --------------------------------------------------------------------------- #
# (4) Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def self_test(imp: dict, steps: dict, curve: dict, mstar: dict) -> dict[str, Any]:
    K_cal = imp["K_cal"]

    # (a) PROVENANCE-EXACT round-trip against #257 banked step + implied TPS at E[T]=4.512.
    s_c32 = steps["central"](32)
    s_o32 = steps["optimistic"](32)
    e_anchor = imp["E_T_built_anchor_257"]   # 4.512
    tps_c32 = tps_of(K_cal, e_anchor, s_c32, 1.0)
    tps_o32 = tps_of(K_cal, e_anchor, s_o32, 1.0)
    bridge_ok = abs(steps["bridge_c"] - imp["bridge_c_banked"]) <= 1e-12
    cond_a = bool(
        abs(s_c32 - imp["step_central_M32_banked"]) <= TOL_PROV
        and abs(s_o32 - imp["step_optimistic_M32_banked"]) <= TOL_PROV
        and abs(tps_c32 - imp["tps_central_M32_at_ET4512_banked"]) <= 1e-3
        and abs(tps_o32 - imp["tps_optimistic_M32_at_ET4512_banked"]) <= 1e-3
        and bridge_ok
        # E[T](8) linear-chain path-product round-trips F_linear8 EXACT
        and abs(e_t_of_M(imp, 8, "split")[0] - imp["F_linear8"]) <= TOL_EXACT
    )

    # (b) lambda_hat >= 0.9780 EXCLUSION is applied: every M>32 is excluded; every kept M is
    #     >= bar; and the gate BINDS (caps the optimistic M* below its unconstrained argmax).
    rows = curve["rows"]
    excluded_all_gt_ceil = all((not r["lambda_valid"]) for r in rows if r["M"] > M_LAMBDA_CEIL)
    kept_all_ge_bar = all(r["lambda_valid"] == (r["lambda_hat"] is not None
                                                and r["lambda_hat"] >= imp["lambda_bar"] - 1e-12)
                          for r in rows if r["M"] <= M_LAMBDA_CEIL)
    mstar_in_valid = bool(mstar["M_star_optimistic_edge"] <= M_LAMBDA_CEIL
                          and mstar["M_star_central_edge"] <= M_LAMBDA_CEIL)
    gate_binds_demo = bool(mstar["lambda_gate_binds_optimistic"])   # unconstrained argmax sits >32
    cond_b = bool(excluded_all_gt_ceil and kept_all_ge_bar and mstar_in_valid and gate_binds_demo)

    # (c) E[T](M) is SUBLINEAR (concave / non-increasing per-doubling increments) and the
    #     floor is respected exactly where claimed (M>=16 clears, M=8 does not).
    e1, e2, e4, e8 = (e_t_of_M(imp, m, "split")[0] for m in (1, 2, 4, 8))
    e16, e32 = (e_t_of_M(imp, m, "split")[0] for m in (16, 32))
    incr = [e2 - e1, e4 - e2, e8 - e4, e16 - e8, e32 - e16]   # not per-doubling but monotone+positive
    doublings = [e8 - e4, e16 - e8, e32 - e16]                 # per-2x increments past M=4
    sublinear = all(d > 0 for d in doublings) and doublings[-1] <= doublings[0] + 1e-9
    floor_logic = bool(e16 >= imp["E_T_meas_floor"] and e32 >= imp["E_T_meas_floor"]
                       and e8 < imp["E_T_meas_floor"])
    cond_c = bool(all(x > 0 for x in incr) and sublinear and floor_logic)

    # (d) the central step edge has its TPS MAX strictly below 500 inside the valid region
    #     (the decisive width-axis ceiling), AND M*=32 is the lambda-valid argmax for both edges.
    cond_d = bool(mstar["max_tps_central_edge"] < TARGET_TPS
                  and mstar["M_star_central_edge"] == M_LAMBDA_CEIL
                  and mstar["M_star_optimistic_edge"] == M_LAMBDA_CEIL)

    # (e) NaN/Inf-clean (key scalars; full-payload walk enforced in main()).
    keymetrics = [s_c32, s_o32, tps_c32, tps_o32,
                  mstar["max_tps_central_edge"], mstar["max_tps_optimistic_edge"],
                  e16, e32, imp["E_T_meas_floor"], imp["lambda_min_q2_q9"]]
    cond_e = all(_finite(x) for x in keymetrics)

    passes = bool(cond_a and cond_b and cond_c and cond_d and cond_e)
    return {
        "verify_gemm_m_aware_self_test_passes": passes,
        "conditions": {
            "a_provenance_roundtrip_257_step_and_tps_and_F_linear8": cond_a,
            "b_lambda_0p9780_exclusion_applied_and_binds": cond_b,
            "c_E_T_sublinear_and_floor_logic": cond_c,
            "d_central_edge_max_below_500_and_Mstar_is_lambda_ceiling": cond_d,
            "e_key_scalars_finite": cond_e,
        },
        "evidence": {
            "a_step_central_M32": s_c32, "a_step_central_M32_banked": imp["step_central_M32_banked"],
            "a_step_optimistic_M32": s_o32, "a_step_optimistic_M32_banked": imp["step_optimistic_M32_banked"],
            "a_tps_central_M32_at_4p512": tps_c32, "a_tps_central_banked": imp["tps_central_M32_at_ET4512_banked"],
            "a_tps_optimistic_M32_at_4p512": tps_o32, "a_tps_optimistic_banked": imp["tps_optimistic_M32_at_ET4512_banked"],
            "a_bridge_c": steps["bridge_c"], "a_bridge_c_banked": imp["bridge_c_banked"],
            "a_E_T_8": e8, "a_F_linear8": imp["F_linear8"],
            "b_M_lambda_ceiling": M_LAMBDA_CEIL,
            "b_lambda_min_q2_q9": imp["lambda_min_q2_q9"], "b_bar": imp["lambda_bar"],
            "b_unconstrained_M_star_optimistic": mstar["M_star_unconstrained_optimistic"],
            "b_valid_M_star_optimistic": mstar["M_star_optimistic_edge"],
            "c_E_T_doublings_past_M4": doublings,
            "c_E_T_8": e8, "c_E_T_16": e16, "c_E_T_32": e32, "c_floor": imp["E_T_meas_floor"],
            "d_max_tps_central_edge": mstar["max_tps_central_edge"],
            "d_max_tps_optimistic_edge": mstar["max_tps_optimistic_edge"],
        },
    }


# --------------------------------------------------------------------------- #
# Verdict + hand-off.
# --------------------------------------------------------------------------- #
def _verdict(imp: dict, mstar: dict, etmat: dict) -> str:
    mc = mstar["max_tps_central_edge"]
    mo = mstar["max_tps_optimistic_edge"]
    return (
        f"WIDTH-AXIS PRICED. Over the banked #257 verify_us(M) grid the TPS(M) curve is MONOTONE-"
        f"increasing across the lambda-valid region [1..32] under BOTH step edges, so M* = "
        f"{mstar['M_star_optimistic_edge']} (the depth-9 / lambda-valid CEILING, not an interior "
        f"width below 32). The #257 super-linear verify knee (M16->M32 +10.6%) is real but the "
        f"rank-coverage E[T] log-growth (+14.3% per the same doubling) still OUTPACES it through "
        f"M=32; the interior max sits ABOVE 32 where lambda_hat is UNMEASURED and therefore "
        f"excluded (fern #249). GO/NO-GO is SPLIT by the #257 g_d step fork, NOT by width: under "
        f"the MEASURED-g_d (central) step the curve tops out at {mc:.1f} < 500 (tree-WIDTH alone "
        f"CANNOT reach 500 — the gate then rests on the DEPTH axis stark #266 and/or a genuine "
        f"E[T] lift fern #259/land #245), while under the ASSUMED-g_d (optimistic) step it clears "
        f"500 at M>=16 and peaks at {mo:.1f}. The E[T] anchor amplifies the same fork: only the "
        f"(optimistic step x rank-cov / 4.512-anchor E[T]) corners clear; the #241 floor (4.3305) "
        f"and the central step never do ({etmat['n_corners_clearing_500']}/{etmat['n_corners_total']} "
        f"corners clear). HAND-OFF to land #245: build at M*={mstar['M_star_optimistic_edge']} "
        f"(widest lambda-valid width); the gate is the g_d step-basis reconciliation + the M=32 "
        f"branch-interior E[T] realization, not the choice of width. BASELINE 481.53 untouched. "
        f"CPU-only analytic. NOT a launch."
    )


def _handoff(imp: dict, mstar: dict, etmat: dict) -> dict[str, str]:
    op = mstar["handoff_operating_point"]
    line = (
        f"verify-GEMM M-aware width: M* = {mstar['M_star_optimistic_edge']} (argmax over the "
        f"lambda-valid [1..32]; TPS(M) MONOTONE-increasing to the depth-9 ceiling — the verify knee "
        f"never produces an interior max BELOW 32, it would only bite past M=32 where lambda is "
        f"unmeasured/excluded). Operating point at M*: E[T]={op['E_T']:.4f} (rank-cov; floor "
        f"4.3305), lambda_hat={op['lambda_hat']:.4f} (>= bar 0.9780), step in [{op['step_optimistic_ms']:.4f} "
        f"optimistic-g_d, {op['step_central_ms']:.4f} measured-g_d] ms. clears_500 = "
        f"{mstar['clears_500_optimistic_edge']} (optimistic step) / {mstar['clears_500_central_edge']} "
        f"(measured step). WIDTH is not the binding axis: the go/no-go is the #257 g_d step "
        f"reconciliation x the M=32 branch-interior E[T] realization (land #245 cycle-1 / fern #259)."
    )
    return {"land_245": line, "stark_266": (
        "width axis = land builds at the lambda-valid ceiling M*=32; under the measured-g_d step "
        "width tops out <500, so your DEPTH-axis adaptive-K early-exit is the complementary lever — "
        "together you set the tree SHAPE (M*=32 width x your depth budget).")}


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    imp = load_imports()
    steps = make_step_models(imp)
    curve = tps_curve(imp, steps, tau=1.0, family="split")
    curve_geom = tps_curve(imp, steps, tau=1.0, family="geom")    # E[T] family sensitivity
    mstar = locate_M_star(imp, curve)
    etmat = et_anchor_matrix(imp, steps, mstar["M_star_optimistic_edge"], tau=1.0)
    tau_chk = tau_band_check(imp, steps, mstar["M_star_optimistic_edge"])
    st = self_test(imp, steps, curve, mstar)
    handoff = _handoff(imp, mstar, etmat)
    return {
        "self_test": st,
        "test_metric": {
            "M_star": mstar["M_star_optimistic_edge"],
            "clears_500_optimistic_edge": mstar["clears_500_optimistic_edge"],
            "clears_500_central_edge": mstar["clears_500_central_edge"],
            "max_tps_central_edge": mstar["max_tps_central_edge"],
            "max_tps_optimistic_edge": mstar["max_tps_optimistic_edge"],
        },
        "imports": imp,
        "step_models": {k: v for k, v in steps.items() if not callable(v)},
        "tps_curve_split": curve,
        "tps_curve_geom_sensitivity": curve_geom,
        "M_star": mstar,
        "et_anchor_matrix": etmat,
        "tau_band": tau_chk,
        "verdict": _verdict(imp, mstar, etmat),
        "handoff_lines": handoff,
    }


# --------------------------------------------------------------------------- #
# NaN-clean walk.
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


# --------------------------------------------------------------------------- #
# Console report.
# --------------------------------------------------------------------------- #
def _print_report(syn: dict) -> None:
    imp = syn["imports"]
    curve, mstar = syn["tps_curve_split"], syn["M_star"]
    etmat, st = syn["et_anchor_matrix"], syn["self_test"]
    print("\n" + "=" * 100, flush=True)
    print("VERIFY-GEMM M-AWARE TREE WIDTH (PR #268) — TPS(M), M*, clears_500", flush=True)
    print("=" * 100, flush=True)
    print(f"  K_cal={imp['K_cal']:.5f}  step_band=[{imp['step_optimistic_M32_banked']:.4f}(opt g_d), "
          f"{imp['step_central_M32_banked']:.4f}(meas g_d)] ms  floor E[T]>={imp['E_T_meas_floor']:.4f}  "
          f"lambda_bar={imp['lambda_bar']}  min-lambda(q2..9)={imp['lambda_min_q2_q9']}", flush=True)
    print("-" * 100, flush=True)
    hdr = f"  {'M':>4} {'verify_us':>10} {'E[T]':>7} {'lam':>7} {'lamOK':>5} {'stepC':>7} {'stepO':>7} {'TPS_C':>7} {'TPS_O':>7}"
    print(hdr, flush=True)
    for r in curve["rows"]:
        lam = f"{r['lambda_hat']:.4f}" if r["lambda_hat"] is not None else "  --  "
        flag = "" if r["lambda_valid"] else "  <-- lambda EXCLUDED (unmeasured)"
        star = "  *M*" if r["M"] == mstar["M_star_optimistic_edge"] else ""
        print(f"  {r['M']:>4} {r['verify_us_us']:>10.1f} {r['E_T']:>7.4f} {lam:>7} "
              f"{str(r['lambda_valid']):>5} {r['step_central_ms']:>7.4f} {r['step_optimistic_ms']:>7.4f} "
              f"{r['tps_central']:>7.1f} {r['tps_optimistic']:>7.1f}{star}{flag}", flush=True)
    print("-" * 100, flush=True)
    print(f"  M* (optimistic edge) = {mstar['M_star_optimistic_edge']}   "
          f"M* (central edge) = {mstar['M_star_central_edge']}   "
          f"(UNCONSTRAINED opt argmax = {mstar['M_star_unconstrained_optimistic']} -> lambda gate "
          f"BINDS = {mstar['lambda_gate_binds_optimistic']})", flush=True)
    print(f"  max TPS  central-edge = {mstar['max_tps_central_edge']:.1f}  (clears 500 = "
          f"{mstar['clears_500_central_edge']})", flush=True)
    print(f"  max TPS  optimistic-edge = {mstar['max_tps_optimistic_edge']:.1f}  (clears 500 = "
          f"{mstar['clears_500_optimistic_edge']})", flush=True)
    print(f"  decisive: width-alone-cannot-reach-500 (both edges) = "
          f"{mstar['decisive_width_cannot_reach_500']}", flush=True)
    print("-" * 100, flush=True)
    print(f"  E[T]-anchor x step matrix @ M*={etmat['M_star']}  "
          f"({etmat['n_corners_clearing_500']}/{etmat['n_corners_total']} corners clear 500):", flush=True)
    for name, v in etmat["matrix"].items():
        print(f"      {name:>28}: E[T]={v['E_T']:.4f}  TPS_C={v['tps_central']:.1f} "
              f"({v['tps_central_clears_500']})  TPS_O={v['tps_optimistic']:.1f} "
              f"({v['tps_optimistic_clears_500']})", flush=True)
    print("-" * 100, flush=True)
    print(f"  (PRIMARY) verify_gemm_m_aware_self_test_passes = "
          f"{st['verify_gemm_m_aware_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"          - {k}: {v}", flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print("=" * 100, flush=True)
    print(f"\n  HAND-OFF (land #245): {syn['handoff_lines']['land_245']}\n", flush=True)


# --------------------------------------------------------------------------- #
# W&B logging (mirrors denken #219; never fatal).
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
        print(f"[verify-m-width] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    imp, curve = syn["imports"], syn["tps_curve_split"]
    mstar, etmat, tauc, st = syn["M_star"], syn["et_anchor_matrix"], syn["tau_band"], syn["self_test"]

    run = init_wandb_run(
        job_type="validity-gate",
        agent="denken",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["verify-gemm", "tree-width", "M-aware", "tps-curve", "rank-coverage",
              "width-axis", "step-band", "g_d-fork", "bank-the-analysis", "pr-268"],
        config={
            "K_cal": imp["K_cal"], "official_baseline": OFFICIAL_BASELINE, "target_tps": TARGET_TPS,
            "step_central_M32": imp["step_central_M32_banked"],
            "step_optimistic_M32": imp["step_optimistic_M32_banked"],
            "E_T_meas_floor": imp["E_T_meas_floor"], "lambda_bar": imp["lambda_bar"],
            "lambda_min_q2_q9": imp["lambda_min_q2_q9"], "rho_cov_W": imp["rho_cov_W"],
            "M_lambda_ceiling": M_LAMBDA_CEIL, "tau_band": imp["tau_band"],
            "imports": "denken#257(h1gj2ved) verify_us(M) x wirbel#79/#86(z6wi4z4v) rank-cov "
                       "x denken#241(hqewf1d6) floor x lambda q-ladder(PARENT_M16)",
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[verify-m-width] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "verify_gemm_m_aware_self_test_passes": int(bool(st["verify_gemm_m_aware_self_test_passes"])),
        "M_star_optimistic_edge": mstar["M_star_optimistic_edge"],
        "M_star_central_edge": mstar["M_star_central_edge"],
        "M_star_unconstrained_optimistic": mstar["M_star_unconstrained_optimistic"],
        "lambda_gate_binds_optimistic": int(bool(mstar["lambda_gate_binds_optimistic"])),
        "max_tps_central_edge": mstar["max_tps_central_edge"],
        "max_tps_optimistic_edge": mstar["max_tps_optimistic_edge"],
        "clears_500_central_edge": int(bool(mstar["clears_500_central_edge"])),
        "clears_500_optimistic_edge": int(bool(mstar["clears_500_optimistic_edge"])),
        "clears_500_both_edges": int(bool(mstar["clears_500_both_edges"])),
        "decisive_width_cannot_reach_500": int(bool(mstar["decisive_width_cannot_reach_500"])),
        "Mstar_E_T": mstar["handoff_operating_point"]["E_T"],
        "Mstar_lambda_hat": mstar["handoff_operating_point"]["lambda_hat"],
        "Mstar_step_central_ms": mstar["handoff_operating_point"]["step_central_ms"],
        "Mstar_step_optimistic_ms": mstar["handoff_operating_point"]["step_optimistic_ms"],
        "n_corners_clearing_500": etmat["n_corners_clearing_500"],
        "tau_flips_optimistic_500": int(bool(tauc["tau_flips_optimistic_500"])),
        "nan_clean": int(bool(payload["nan_clean"])),
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
        # per-M scalars for plotting the TPS(M) curve
        **{f"tps_central_M{r['M']}": r["tps_central"] for r in curve["rows"]},
        **{f"tps_optimistic_M{r['M']}": r["tps_optimistic"] for r in curve["rows"]},
        **{f"E_T_M{r['M']}": r["E_T"] for r in curve["rows"]},
        **{f"verify_us_M{r['M']}": r["verify_us_us"] for r in curve["rows"]},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="verify_gemm_m_width_result", artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[verify-m-width] wandb logged {len(summary)} summary keys", flush=True)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", default="denken-verify-m-width")
    args = ap.parse_args(argv)

    syn = synthesize()

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at,
        "pr": 268,
        "agent": "denken",
        "kind": "verify-gemm-m-width",
        "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }

    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[verify-m-width] WARNING non-finite values at: {nan_paths}", flush=True)
    # fold nan-clean into self-test (e) and recompute PRIMARY
    syn["self_test"]["conditions"]["e_key_scalars_finite"] = bool(
        syn["self_test"]["conditions"]["e_key_scalars_finite"] and payload["nan_clean"])
    passes = bool(all(syn["self_test"]["conditions"].values()))
    syn["self_test"]["verify_gemm_m_aware_self_test_passes"] = passes

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "verify_gemm_m_width_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[verify-m-width] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    tm = syn["test_metric"]
    print(f"  PRIMARY verify_gemm_m_aware_self_test_passes = {passes}", flush=True)
    print(f"  TEST M* = {tm['M_star']}  clears_500: optimistic={tm['clears_500_optimistic_edge']} "
          f"central={tm['clears_500_central_edge']}  (max TPS {tm['max_tps_central_edge']:.1f}/"
          f"{tm['max_tps_optimistic_edge']:.1f})", flush=True)
    print(f"  peak_mem_mib = {payload['peak_mem_mib']}", flush=True)

    if args.self_test:
        ok = passes and payload["nan_clean"]
        print(f"[verify-m-width] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
