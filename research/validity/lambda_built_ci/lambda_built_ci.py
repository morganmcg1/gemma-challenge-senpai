#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Margin-aware λ̂_built measurement-CI (PR #187) — the GATE-INPUT resolvability stamp.

denken #183 (`82uisrez`) made the GO gate crisp: the both-bugs build must demonstrate
per-depth self-KV recovery λ ≥ 0.9052 (the finite-sample-LCB-clears-500 bar). wirbel
#175 (`zh1accmi`) priced the noise on the *projected TPS* (±10.9, OUTPUT side). But the
gate's INPUT is itself a *finite-sample measurement*: land #71 reports `λ̂_built` inferred
from a measured per-depth accepted-rate ladder `q[2..9]`, each `q̂_d` a binomial over the
verify-positions that REACH depth d across a 128-prompt × 512-token bench. So **λ̂_built
carries its own sampling CI** — and if that CI straddles 0.9052, the single-run GO is a
coin-flip on the bar, not a decision.

This PR is the INPUT-side dual of #175: it prices the measurement noise on the *measured
λ̂* that drives the gate, and converts it into a measurement protocol — given N prompts at
output_len 512, what is the half-width on λ̂_built, and how many prompts does land #71 need
so its implied λ̂ cleanly resolves (clears OR fails) 0.9052 at 95%? It is a SYNTHESIS: it
IMPORTS denken #183's λ-card (`q_d(λ)`, the inverse map, λ*_LCB=0.9052), #178's forward
map / recovery geometry, and wirbel #175's accepted-length pmf (the depth-thinning survival
that sets the per-depth trial counts). It does NOT re-derive `q_d(λ)`, the 0.9052 bar,
5.0564/5.2070, K_cal, the step, σ_L, the ±10.906 CI, or σ_hw.

LOCAL CPU-only analytic. No GPU / vLLM / HF Job / submission / kernel build / served-file
change. BASELINE stays 481.53. Greedy untouched. Bank-the-analysis (PRIMARY = self-test,
adds 0 TPS). NOT open2 — the acceptance / forward-map lane. NOT a launch.

------------------------------------------------------------------------------
THE MEASUREMENT MODEL (the new object; everything else imported)
------------------------------------------------------------------------------
Per depth d, land's measured conditional accept is a binomial rate:
    q̂_d ~ Binomial(n_d, q_d) / n_d ,   Var[q̂_d] = q_d(1−q_d)/n_d
where n_d = the number of verify-positions that REACH depth d across the bench. The chain
is greedy (mutually-exclusive siblings, #172): a step tests the depth-d spine token iff its
accepted chain reached depth d−1, i.e. iff committed length L ≥ d. Hence the per-depth
trial ladder is the SURVIVAL of wirbel #175's accepted-length pmf:
    n_d = N_steps · S(d),   S(d) = P(L ≥ d) = Σ_{k≥d} pmf[k],   N_steps = n_prompts·512 / E[T]
Depths thin out exactly as the pmf tail decays (depth-9 has far fewer trials than depth-2).

Inversion (denken #183's per-depth deficit-closure map):
    λ̂_d = (q̂_d − q_floor[d]) / (q_full[d] − q_floor[d]) = (q̂_d − q_floor[d]) / span_d
    λ̂_built = pool_d λ̂_d        (OLS simple-mean = #183 inverse map ; WLS/MLE = inverse-variance)
Delta-method: Var[λ̂_d] = Var[q̂_d]/span_d² ; the pooled half-width is z95·√Var[λ̂_built].

PRIMARY metric  lambda_built_ci_self_test_passes
TEST    metric  lambda_built_halfwidth   (both-bugs, λ̂≈0.905 operating point, N=128 prompts)

Run:
    python -m research.validity.lambda_built_ci.lambda_built_ci --self-test \
        --wandb-name denken/lambda-built-ci --wandb-group lambda-built-ci
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

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Import denken #183 (λ-card: q_d(λ), inverse map, λ*_LCB) which itself imports
# #178 (forward map / recovery geometry) and #175 (accepted-length pmf). Nothing
# re-derived; the chain D183 -> D178 -> D172 + D175 is reached transitively.
# --------------------------------------------------------------------------- #
_D183_PATH = REPO_ROOT / "research/oracle_readout/lambda_acceptance_card/lambda_acceptance_card.py"


def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import module {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D183 = _import("lambda_acceptance_card", _D183_PATH)
D178 = D183.D178
D175 = D183.D175
D172 = D183.D172

# ---- composition / CI constants (committed; imported, not re-derived) ---- #
K_CAL = D183.K_CAL                       # 125.268
STEP = D183.STEP                         # 1.2182
Z95 = D183.Z95                           # 1.959963984540054
SIGMA_HW = D183.SIGMA_HW                 # 4.86 (kanna #159 denominator leg)
TARGET_OFFICIAL = D183.TARGET_OFFICIAL   # 500.0
MAXD = D183.MAXD                         # 24 (pvec build horizon)
TAU_CENTRAL = D183.TAU_CENTRAL           # 1.0

# ---- the gate's two operating points (imported from #183) ---- #
BOTH_BUGS_LAMBDA_STAR_LCB = 0.905229     # #183 TEST metric (both-bugs LCB bar)
DESCENT_LAMBDA_STAR_LCB = 0.975          # #183 descent-only LCB bar (context)

# ---- the bench contract for the q[2..9] ladder (PR #187) ---- #
DEFAULT_N_PROMPTS = 128
DEFAULT_OUTPUT_LEN = 512                  # 128 × 512 ladder bench (== #175 BENCH_TOKENS_ALT)
D_LO, D_HI = 2, 9                         # the self-KV-governed depths (depth-1 = BUG-1 axis)
N_DEPTHS = D_HI - D_LO + 1               # 8

# ---- resolvability protocol (PR #187) ---- #
RESOLVE_TRUE_LAMBDAS = (0.86, 0.88, 0.905, 0.93, 0.95)
RESOLVE_N_CAP = 1.0e9                     # clamp (keeps NaN-clean if a build sits ON the bar)

# ---- serial-correlation sensitivity (honest scope; mirrors #183) ---- #
SERIAL_CORR_VIF = (1.0, 1.5, 2.0)         # effective-N inflation: n_eff = n_d / VIF

# ---- self-test recovery anchors ---- #
LIVEPROBE_LAMBDA = 0.342                  # #178 liveprobe depth-1 deficit closure
RECOVER_TOL = 1e-9
SCALING_TOL = 1e-6                        # 1/√N scaling constant tolerance


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# Topology endpoints (imported from #183 / #178). both-bugs is the operating
# topology; descent-only carried for cross-checks.
# --------------------------------------------------------------------------- #
def build_ctx(anchors: dict[str, Any]) -> dict[str, Any]:
    ctx = D183.build_topologies(anchors)
    topo = ctx["topo"]
    ctx["qfl_b"], ctx["qfu_b"] = topo["both_bugs"]["q_floor"], topo["both_bugs"]["q_full"]
    ctx["qfl_d"], ctx["qfu_d"] = topo["descent_only"]["q_floor"], topo["descent_only"]["q_full"]
    return ctx


def operating_spine(ctx: dict, lam: float, q_floor: list[float], q_full: list[float]) -> list[float]:
    """The constant-λ spine at the operating point (7-entry horizon spine)."""
    return D178.spine_from_profile(ctx["ep"], D178.constant_lambda(len(q_full), lam), q_floor, q_full)


def extended_spine_9(spine: list[float]) -> list[float]:
    """Un-tie the flat-extrapolated deep depths into an explicit depth-1..9 vector so each
    measured depth (incl. 8,9) is an independent input. Setting entries 8,9 to the depth-7
    clamp value is a NO-OP for E[T]/pmf (qd_at clamps identically) but lets the compose-step
    perturb depths 8,9 independently."""
    return [D172.qd_at(spine, d) for d in range(1, D_HI + 1)]


# --------------------------------------------------------------------------- #
# (1) Measurement model: per-depth trial ladder n_d from the #175 survival.
# --------------------------------------------------------------------------- #
def survival_and_et(ctx: dict, spine: list[float]) -> tuple[dict[int, float], float, np.ndarray]:
    """S(d) = P(L ≥ d) reach-survival of wirbel #175's accepted-length pmf at this spine,
    plus E[T] (= pmf mean, the provenance lock). n_d is set by S(d)."""
    ep = ctx["ep"]
    pvecs = D175.build_depth_pvecs_measured(spine, ep["rho_cond"], ep["W"], MAXD, "flat")
    pmf, _, _, _ = D175.dp_accepted_length_pmf(ep["parent"], pvecs)
    et = float(D175.pmf_moments(pmf)["mean"])
    survival = {d: float(pmf[d:].sum()) for d in range(1, len(pmf))}
    return survival, et, pmf


def per_depth_table(ctx: dict, lam: float, q_floor: list[float], q_full: list[float],
                    n_prompts: int = DEFAULT_N_PROMPTS, output_len: int = DEFAULT_OUTPUT_LEN,
                    vif: float = 1.0) -> dict[str, Any]:
    """Per-depth measurement model at recovery λ: true q_d, span_d, survival S(d), trial
    count n_d = N_steps·S(d), binomial Var[q̂_d], and propagated Var[λ̂_d]."""
    spine = operating_spine(ctx, lam, q_floor, q_full)
    survival, et, _ = survival_and_et(ctx, spine)
    n_steps = n_prompts * output_len / et
    rows = []
    for d in range(D_LO, D_HI + 1):
        q_d = D172.qd_at(spine, d)
        q_fl = D172.qd_at(q_floor, d)
        q_fu = D172.qd_at(q_full, d)
        span = q_fu - q_fl
        s_d = survival.get(d, 0.0)
        n_d = n_steps * s_d / vif                          # effective trials (VIF deflates)
        var_q = q_d * (1.0 - q_d) / n_d if n_d > 0 else float("inf")
        var_lam_d = var_q / (span * span) if abs(span) > 1e-12 else float("inf")
        rows.append({
            "depth": d, "q_true": q_d, "q_floor": q_fl, "q_full": q_fu, "span": span,
            "survival_reach": s_d, "n_trials": n_d, "var_qhat": var_q,
            "var_lambda_hat_d": var_lam_d,
            "flat_extrapolated": bool(d > len(spine)),
        })
    return {"lambda": lam, "E_T": et, "N_steps": n_steps, "n_prompts": n_prompts,
            "output_len": output_len, "vif": vif, "rows": rows,
            "n_ladder": [r["n_trials"] for r in rows],
            "survival_ladder": [r["survival_reach"] for r in rows]}


# --------------------------------------------------------------------------- #
# (2) Propagate q̂ -> λ̂_built CI. Two estimators of the pooled λ̂_built:
#   OLS  (simple mean = #183's lambda_from_measured_ladder, what land's code does)
#   WLS  (inverse-variance / MLE, the optimal "least-squares fit across depths")
# --------------------------------------------------------------------------- #
def pool_lambda_built(per: dict[str, Any]) -> dict[str, Any]:
    rows = per["rows"]
    var_d = [r["var_lambda_hat_d"] for r in rows]
    D = len(rows)
    # OLS simple mean (denken #183 inverse map): equal weight 1/D.
    var_ols = sum(var_d) / (D * D)
    hw_ols = Z95 * math.sqrt(var_ols)
    # WLS / MLE: inverse-variance weights.
    inv = [1.0 / v if v > 0 and math.isfinite(v) else 0.0 for v in var_d]
    sum_inv = sum(inv)
    var_wls = 1.0 / sum_inv if sum_inv > 0 else float("inf")
    hw_wls = Z95 * math.sqrt(var_wls)
    weights_wls = [w / sum_inv if sum_inv > 0 else 0.0 for w in inv]
    # per-depth variance contribution shares (OLS pooling: (1/D²)·var_d / var_ols).
    contrib_ols = [(v / (D * D)) / var_ols if var_ols > 0 else float("nan") for v in var_d]
    return {
        "var_ols": var_ols, "halfwidth_ols": hw_ols,
        "var_wls": var_wls, "halfwidth_wls": hw_wls,
        "wls_weights": weights_wls,
        "ols_var_contribution_share": contrib_ols,
        "per_depth_var": var_d,
    }


def lambda_built_ci(ctx: dict, lam: float, q_floor: list[float], q_full: list[float],
                    n_prompts: int = DEFAULT_N_PROMPTS, output_len: int = DEFAULT_OUTPUT_LEN,
                    vif: float = 1.0, estimator: str = "wls") -> dict[str, Any]:
    """Half-width on λ̂_built and the CI [λ̂−hw, λ̂+hw] at recovery λ, N prompts."""
    per = per_depth_table(ctx, lam, q_floor, q_full, n_prompts, output_len, vif)
    pool = pool_lambda_built(per)
    hw = pool["halfwidth_wls"] if estimator == "wls" else pool["halfwidth_ols"]
    return {"lambda": lam, "estimator": estimator, "halfwidth": hw,
            "ci": [lam - hw, lam + hw], "per": per, "pool": pool}


# --------------------------------------------------------------------------- #
# Inverse map (imported #183) — recover λ̂ from a measured ladder, both poolings.
# --------------------------------------------------------------------------- #
def recover_lambda(ctx: dict, q_meas_2to9: list[float], q_floor: list[float],
                   q_full: list[float], var_d: list[float] | None = None) -> dict[str, float]:
    """OLS recovery via #183's lambda_from_measured_ladder (simple mean) + WLS recovery."""
    ols = D183.lambda_from_measured_ladder(ctx, q_meas_2to9, q_floor, q_full)["lambda_hat_built"]
    lam_d = []
    for i, q_m in enumerate(q_meas_2to9):
        d = D_LO + i
        span = D172.qd_at(q_full, d) - D172.qd_at(q_floor, d)
        lam_d.append((q_m - D172.qd_at(q_floor, d)) / span if abs(span) > 1e-12 else float("nan"))
    if var_d is not None:
        inv = [1.0 / v if v > 0 and math.isfinite(v) else 0.0 for v in var_d]
        s = sum(inv)
        wls = sum(w * l for w, l in zip(inv, lam_d)) / s if s > 0 else float("nan")
    else:
        wls = sum(lam_d) / len(lam_d)
    return {"ols": ols, "wls": wls}


# --------------------------------------------------------------------------- #
# (3) Resolvability gate.  n_prompts so a build at true-λ resolves vs the bar at 95%.
#   half-width scales as 1/√N  ->  hw(N) = hw(N0)·√(N0/N).
#   resolve iff hw(N) ≤ margin  ->  N ≥ N0·(hw(N0)/margin)².
# --------------------------------------------------------------------------- #
def n_prompts_to_resolve(ctx: dict, true_lambda: float, q_floor: list[float], q_full: list[float],
                         bar: float, n0: int = DEFAULT_N_PROMPTS, output_len: int = DEFAULT_OUTPUT_LEN,
                         vif: float = 1.0, estimator: str = "wls",
                         hw_ref: float | None = None) -> dict[str, Any]:
    # Operating-point half-width at the true build quality — auditable, but it varies with the
    # operating point (q_d(1−q_d) and span move with λ), so it is NOT used for the resolvability
    # curve, which must be a pure function of margin.
    res_op = lambda_built_ci(ctx, true_lambda, q_floor, q_full, n0, output_len, vif, estimator)
    hw_op = res_op["halfwidth"]
    # Reference noise for the resolvability curve = the at-the-bar half-width (the gate's noise
    # floor AT the decision boundary). With a fixed reference, N(margin)=n0·(hw_ref/margin)² is a
    # pure, monotone-decreasing function of margin. Defaults to the bar half-width if not supplied.
    if hw_ref is None:
        hw_ref = lambda_built_ci(ctx, bar, q_floor, q_full, n0, output_len, vif, estimator)["halfwidth"]
    margin = abs(true_lambda - bar)
    if margin < 1e-9:
        n_need = RESOLVE_N_CAP
    else:
        n_need = n0 * (hw_ref / margin) ** 2
    n_need = min(n_need, RESOLVE_N_CAP)
    return {
        "true_lambda": true_lambda, "bar": bar, "margin": margin,
        "side": ("GO" if true_lambda > bar else "NO-GO" if true_lambda < bar else "ON-BAR"),
        "halfwidth_at_n0": hw_ref, "halfwidth_at_true_lambda": hw_op, "n0": n0,
        "n_prompts_to_resolve": math.ceil(n_need) if math.isfinite(n_need) else RESOLVE_N_CAP,
        "n_prompts_to_resolve_raw": n_need,
        "decisive_at_n0": bool(hw_ref <= margin),
        "on_bar_unresolvable": bool(n_need >= RESOLVE_N_CAP),
        "estimator": estimator,
    }


# --------------------------------------------------------------------------- #
# (4) Compose input-CI (this) ⊕ output-CI (#175). Double-count audit: both are
# functions of the SAME finite-sample accept events, so on a shared bench the
# λ̂-noise partly lives inside #175's L̄-draw -> quadrature double-counts.
# --------------------------------------------------------------------------- #
def _dEt_dq(ctx: dict, ext_spine: list[float], d: int, eps: float = 1e-6) -> float:
    """∂E[T]/∂q_d via central difference on the imported #172 backward DP, perturbing the
    depth-d conditional independently (ext_spine is the un-tied depth-1..9 vector)."""
    ep = ctx["ep"]
    sp_hi = list(ext_spine); sp_hi[d - 1] += eps
    sp_lo = list(ext_spine); sp_lo[d - 1] -= eps
    et_hi = D172.et_backward(ep["parent"], ep["children"], ep["depth"], sp_hi, ep["rho_cond"], ep["W"])
    et_lo = D172.et_backward(ep["parent"], ep["children"], ep["depth"], sp_lo, ep["rho_cond"], ep["W"])
    return (et_hi - et_lo) / (2.0 * eps)


def _dEt_dlambda(ctx: dict, lam: float, q_floor: list[float], q_full: list[float],
                 eps: float = 1e-6) -> float:
    hi = D178.et_of_spine(ctx["ep"], operating_spine(ctx, lam + eps, q_floor, q_full))
    lo = D178.et_of_spine(ctx["ep"], operating_spine(ctx, lam - eps, q_floor, q_full))
    return (hi - lo) / (2.0 * eps)


def compose_input_output(ctx: dict, lam: float, q_floor: list[float], q_full: list[float],
                         n_prompts: int = DEFAULT_N_PROMPTS, output_len: int = DEFAULT_OUTPUT_LEN,
                         tau: float = TAU_CENTRAL, estimator: str = "wls") -> dict[str, Any]:
    """Overlap audit between the λ̂-route (INPUT) and #175's L̄-route (OUTPUT) TPS CIs.

    Both are linear functionals of the SAME per-depth accept fluctuations δq̂_d:
      δλ̂_built = Σ_d w_λ,d δq̂_d ,  w_λ,d = (estimator weight_d)/span_d
      δL̄(rate-driven) = Σ_d w_L,d δq̂_d ,  w_L,d = ∂E[T]/∂q_d   (#175's L̄ shares this part)
    overlap = squared correlation ρ² in the Var[q̂]-metric ⟨u,v⟩ = Σ_d u_d v_d Var[q̂_d]."""
    per = per_depth_table(ctx, lam, q_floor, q_full, n_prompts, output_len)
    pool = pool_lambda_built(per)
    rows = per["rows"]
    spine = operating_spine(ctx, lam, q_floor, q_full)
    ext = extended_spine_9(spine)

    D = len(rows)
    inv = [1.0 / r["var_lambda_hat_d"] if r["var_lambda_hat_d"] > 0 else 0.0 for r in rows]
    sum_inv = sum(inv)
    w_lambda, w_L, var_q = [], [], []
    for i, r in enumerate(rows):
        d = r["depth"]
        weight = (inv[i] / sum_inv) if estimator == "wls" and sum_inv > 0 else (1.0 / D)
        w_lambda.append(weight / r["span"] if abs(r["span"]) > 1e-12 else 0.0)
        w_L.append(_dEt_dq(ctx, ext, d))
        var_q.append(r["var_qhat"])

    def dot(u, v):
        return sum(ui * vi * wi for ui, vi, wi in zip(u, v, var_q))

    num = dot(w_L, w_lambda)
    den = math.sqrt(dot(w_L, w_L) * dot(w_lambda, w_lambda))
    rho = num / den if den > 0 else 0.0
    overlap = rho * rho

    # TPS half-widths for the two routes (at the 128×512 bench).
    slope_tps = K_CAL * tau / STEP
    et = per["E_T"]
    # INPUT route: forward-map slope d(central_tps)/dλ × hw_λ.
    fwd_slope = slope_tps * _dEt_dlambda(ctx, lam, q_floor, q_full)
    hw_lambda = (Z95 * math.sqrt(pool["var_wls"])) if estimator == "wls" else (Z95 * math.sqrt(pool["var_ols"]))
    h_in_tps = abs(fwd_slope) * hw_lambda
    # OUTPUT route: #175's σ_L/√N_steps × slope (recomputed on the 128×512 bench).
    _, _, pmf = survival_and_et(ctx, spine)
    sigma_L = float(D175.pmf_moments(pmf)["std"])
    n_steps = per["N_steps"]
    se_lbar = sigma_L / math.sqrt(n_steps)
    h_out_tps = Z95 * slope_tps * se_lbar
    # composition: naive quadrature vs overlap-corrected (same-bench removes the redundant part).
    quad = math.sqrt(h_in_tps ** 2 + h_out_tps ** 2)
    corrected_same_bench = math.sqrt(h_out_tps ** 2 + (1.0 - overlap) * h_in_tps ** 2)
    verdict = "independent-quadrature" if overlap < 0.02 else "partial-overlap"
    return {
        "verdict": verdict,
        "overlap_fraction": overlap,
        "rho_input_output": rho,
        "h_in_tps_lambda_route": h_in_tps,
        "h_out_tps_lbar_route_175": h_out_tps,
        "sigma_L": sigma_L,
        "forward_map_slope_tps_per_lambda": fwd_slope,
        "quadrature_sum_tps": quad,
        "overlap_corrected_same_bench_tps": corrected_same_bench,
        "w_lambda": w_lambda, "w_L_dEt_dq": w_L, "var_qhat": var_q,
        "regime_note": (
            "INDEPENDENT BENCHES (ladder bench != official TPS bench): the λ̂-rate uncertainty "
            "and #175's conditional L̄-scatter decompose by law of total variance -> add in "
            f"quadrature ({quad:.2f} TPS). SAME BENCH (one run yields both q̂[2..9] and L̄): the "
            f"realized λ̂ and L̄ are the SAME draw, sharing overlap_fraction={overlap:.3f} of "
            "variance; quadrature double-counts -> use the overlap-corrected "
            f"{corrected_same_bench:.2f} TPS (the integrator fern #185 must not stack independent "
            "CIs on a shared bench)."),
    }


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize(anchors: dict[str, Any], n_prompts: int = DEFAULT_N_PROMPTS,
               output_len: int = DEFAULT_OUTPUT_LEN, estimator: str = "wls") -> dict[str, Any]:
    ctx = build_ctx(anchors)
    qfl_b, qfu_b = ctx["qfl_b"], ctx["qfu_b"]
    qfl_d, qfu_d = ctx["qfl_d"], ctx["qfu_d"]
    bar = BOTH_BUGS_LAMBDA_STAR_LCB
    lam_op = bar                                            # operating point ≈ 0.905 (the bar)

    # ---------- (1) measurement model + n_d ladder (both-bugs, λ=bar) ---------- #
    per_op = per_depth_table(ctx, lam_op, qfl_b, qfu_b, n_prompts, output_len)
    n_ladder = per_op["n_ladder"]

    # ---------- (2) λ̂_built CI at the operating point ---------- #
    ci_op = lambda_built_ci(ctx, lam_op, qfl_b, qfu_b, n_prompts, output_len, estimator=estimator)
    lambda_built_halfwidth = ci_op["halfwidth"]            # TEST metric
    pool_op = ci_op["pool"]
    # estimator-independent: dominant raw per-depth variance contributor (self-test e).
    var_d = pool_op["per_depth_var"]
    dominant_idx = int(np.argmax(var_d))
    dominant_depth = D_LO + dominant_idx

    # ---------- (3) resolvability gate / resolve_table ---------- #
    resolve_rows = [n_prompts_to_resolve(ctx, tl, qfl_b, qfu_b, bar, n_prompts, output_len,
                                         estimator=estimator, hw_ref=lambda_built_halfwidth)
                    for tl in RESOLVE_TRUE_LAMBDAS]
    # monotone-in-margin check (self-test d): N decreasing as margin grows.
    by_margin = sorted(resolve_rows, key=lambda r: r["margin"])
    resolve_monotone = all(by_margin[i]["n_prompts_to_resolve_raw"]
                           >= by_margin[i + 1]["n_prompts_to_resolve_raw"] - 1e-6
                           for i in range(len(by_margin) - 1))

    # ---------- (4) compose input ⊕ output ---------- #
    compose = compose_input_output(ctx, lam_op, qfl_b, qfu_b, n_prompts, output_len,
                                   estimator=estimator)

    # ---------- serial-correlation sensitivity (conservative inflation) ---------- #
    sens = []
    for vif in SERIAL_CORR_VIF:
        r = lambda_built_ci(ctx, lam_op, qfl_b, qfu_b, n_prompts, output_len, vif=vif,
                            estimator=estimator)
        sens.append({"vif": vif, "implied_lag1_rho": (vif - 1.0) / (vif + 1.0),
                     "halfwidth": r["halfwidth"]})

    # ---------- (5) self-test (PRIMARY) ---------- #
    st = _self_test(ctx, qfl_b, qfu_b, qfl_d, qfu_d, bar, per_op, pool_op, ci_op,
                    resolve_rows, resolve_monotone, dominant_depth, var_d,
                    n_prompts, output_len, estimator)

    handoff = _handoff(bar=bar, hw=lambda_built_halfwidth, resolve_rows=resolve_rows,
                       n_prompts=n_prompts, estimator=estimator, ci_op=ci_op,
                       overlap=compose["overlap_fraction"])

    return {
        "self_test": st,
        "test_metric": {"lambda_built_halfwidth": lambda_built_halfwidth},
        "operating_point": {
            "topology": "both_bugs", "lambda": lam_op, "bar_lambda_star_lcb": bar,
            "n_prompts": n_prompts, "output_len": output_len, "estimator": estimator,
            "E_T": per_op["E_T"], "N_steps": per_op["N_steps"],
        },
        "measurement_model": {
            "formula_n_d": "n_d = N_steps · S(d) ; S(d)=P(L>=d)=sum_{k>=d} pmf[k] ; N_steps=n_prompts·output_len/E[T]",
            "formula_var_qhat": "Var[q̂_d] = q_d(1-q_d)/n_d  (binomial)",
            "formula_var_lambda": "Var[λ̂_d] = Var[q̂_d]/span_d² ; span_d = q_full[d]-q_floor[d]",
            "n_ladder": n_ladder,
            "survival_ladder": per_op["survival_ladder"],
            "per_depth": per_op["rows"],
        },
        "lambda_built_ci": {
            "lambda_built_halfwidth": lambda_built_halfwidth,
            "lambda_built_ci": ci_op["ci"],
            "halfwidth_ols_simple_mean_183": pool_op["halfwidth_ols"],
            "halfwidth_wls_mle": pool_op["halfwidth_wls"],
            "estimator_primary": estimator,
            "dominant_variance_depth": dominant_depth,
            "per_depth_var_lambda": var_d,
            "ols_var_contribution_share": pool_op["ols_var_contribution_share"],
            "wls_weights": pool_op["wls_weights"],
            "note": ("OLS simple-mean is the literal denken #183 inverse map (what land's code "
                     "does today); WLS/MLE is the optimal inverse-variance pooling the PR names "
                     "— it down-weights the noisy deep depths and needs fewer prompts. PRIMARY "
                     "estimator = " + estimator + "."),
        },
        "resolvability": {
            "bar": bar,
            "resolve_table": resolve_rows,
            "resolve_monotone_in_margin": resolve_monotone,
            "punchline": _resolve_punchline(resolve_rows, n_prompts),
        },
        "input_output_compose": compose,
        "serial_correlation_sensitivity": sens,
        "verdict": _verdict(lambda_built_halfwidth, resolve_rows, bar, n_prompts,
                            compose["overlap_fraction"]),
        "handoff_lines": handoff,
    }


def _resolve_punchline(resolve_rows: list[dict], n_prompts: int) -> str:
    by_lam = {r["true_lambda"]: r for r in resolve_rows}
    r93 = by_lam.get(0.93)
    if r93 is None:
        return ""
    return (f"to make a true-λ=0.93 build read as a DECISIVE GO vs the 0.9052 bar, measure "
            f"q[2..9] over N≥{r93['n_prompts_to_resolve']} prompts at output_len 512 "
            f"(decisive at the default {n_prompts}: {r93['decisive_at_n0']}).")


def _verdict(hw: float, resolve_rows: list[dict], bar: float, n_prompts: int, overlap: float) -> str:
    by_lam = {r["true_lambda"]: r for r in resolve_rows}
    near = by_lam.get(0.905)
    far = by_lam.get(0.93)
    near_msg = (f"a build at λ≈0.905 (ON the bar) needs N≈{near['n_prompts_to_resolve']:.0f} prompts "
                f"— effectively unresolvable") if near else ""
    far_msg = (f"a true-λ=0.93 build resolves decisively at N≥{far['n_prompts_to_resolve']} "
               f"({'already' if far['decisive_at_n0'] else 'NOT'} at the default {n_prompts})") if far else ""
    return (
        f"INPUT-CI-STAMPED. At the both-bugs 0.9052 bar the measured λ̂_built half-width is "
        f"±{hw:.4f} over the default {n_prompts}-prompt × 512-token bench. RESOLVABILITY: "
        f"{far_msg}; {near_msg}. The gate INPUT is decisive only with margin from the bar — a "
        f"point λ̂ within ~±{hw:.3f} of 0.9052 is an indecisive GO. Input/output CIs are "
        f"partial-overlap (overlap_fraction={overlap:.3f} on a shared bench — quadrature would "
        f"double-count). This prices the GATE INPUT noise; it does NOT move the bar (#183 owns "
        f"it) or authorize a launch. NOT open2. NOT a launch."
    )


def _handoff(*, bar: float, hw: float, resolve_rows: list[dict], n_prompts: int,
             estimator: str, ci_op: dict, overlap: float) -> dict[str, str]:
    by_lam = {r["true_lambda"]: r for r in resolve_rows}
    r93 = by_lam.get(0.93)
    n93 = r93["n_prompts_to_resolve"] if r93 else "?"
    land = (
        f"land #71 measurement protocol (denken #187): measure your q[2..9] ladder over "
        f"N≥{n93} prompts at output_len 512 so your implied λ̂_built resolves the 0.9052 bar "
        f"DECISIVELY at 95% — otherwise a point λ̂ near the bar (half-width ±{hw:.4f} at the "
        f"default {n_prompts}) is an indecisive GO, not a decision. Pool the per-depth λ̂_d by "
        f"inverse-variance ({estimator}); the deep depths thin out and dominate the raw "
        f"variance. Report q̂[2..9] WITH their per-depth trial counts n_d so the CI is auditable."
    )
    fern = (
        f"fern #185 integrator note (denken #187): the INPUT-side λ̂_built CI (half-width "
        f"±{hw:.4f} at N={n_prompts}) and wirbel #175's OUTPUT-side ±10.9 TPS CI are NOT "
        f"independent on a shared bench — they share overlap_fraction={overlap:.3f} of variance "
        f"(both are functions of the same accept draw). Compose in quadrature ONLY if the ladder "
        f"and official-TPS benches are independent; on a shared bench subtract the overlap."
    )
    return {"land_71_protocol": land, "fern_185_integrator": fern}


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def _self_test(ctx, qfl_b, qfu_b, qfl_d, qfu_d, bar, per_op, pool_op, ci_op,
               resolve_rows, resolve_monotone, dominant_depth, var_d,
               n_prompts, output_len, estimator) -> dict[str, Any]:
    # (a) liveprobe ladder λ=0.342 inverts back to 0.342 (both poolings, both topologies).
    rec_live = {}
    for lab, (qfl, qfu) in (("both_bugs", (qfl_b, qfu_b)), ("descent_only", (qfl_d, qfu_d))):
        spine = operating_spine(ctx, LIVEPROBE_LAMBDA, qfl, qfu)
        ladder = [D172.qd_at(spine, d) for d in range(D_LO, D_HI + 1)]
        per = per_depth_table(ctx, LIVEPROBE_LAMBDA, qfl, qfu, n_prompts, output_len)
        rec = recover_lambda(ctx, ladder, qfl, qfu, [r["var_lambda_hat_d"] for r in per["rows"]])
        rec_live[lab] = rec
    cond_a = all(abs(r["ols"] - LIVEPROBE_LAMBDA) < RECOVER_TOL
                 and abs(r["wls"] - LIVEPROBE_LAMBDA) < RECOVER_TOL for r in rec_live.values())

    # (b) λ=1 ladder inverts to 1.0 (and λ=0 -> 0.0) — endpoint recovery.
    rec_one, rec_zero = {}, {}
    for lab, (qfl, qfu) in (("both_bugs", (qfl_b, qfu_b)), ("descent_only", (qfl_d, qfu_d))):
        sp1 = operating_spine(ctx, 1.0, qfl, qfu)
        sp0 = operating_spine(ctx, 0.0, qfl, qfu)
        l1 = [D172.qd_at(sp1, d) for d in range(D_LO, D_HI + 1)]
        l0 = [D172.qd_at(sp0, d) for d in range(D_LO, D_HI + 1)]
        rec_one[lab] = recover_lambda(ctx, l1, qfl, qfu)
        rec_zero[lab] = recover_lambda(ctx, l0, qfl, qfu)
    cond_b = all(abs(rec_one[l]["ols"] - 1.0) < RECOVER_TOL and abs(rec_one[l]["wls"] - 1.0) < RECOVER_TOL
                 and abs(rec_zero[l]["ols"]) < RECOVER_TOL and abs(rec_zero[l]["wls"]) < RECOVER_TOL
                 for l in rec_one)

    # (c) half-width shrinks as 1/√N: hw(N)·√N is constant; monotone decreasing in N.
    scaling = []
    for npr in (32, 64, 128, 256, 512, 1024):
        r = lambda_built_ci(ctx, bar, qfl_b, qfu_b, npr, output_len, estimator=estimator)
        scaling.append({"n_prompts": npr, "halfwidth": r["halfwidth"],
                        "hw_times_sqrt_n": r["halfwidth"] * math.sqrt(npr)})
    const0 = scaling[0]["hw_times_sqrt_n"]
    cond_c = (all(scaling[i]["halfwidth"] > scaling[i + 1]["halfwidth"] for i in range(len(scaling) - 1))
              and all(abs(s["hw_times_sqrt_n"] - const0) < SCALING_TOL * const0 for s in scaling))

    # (d) resolve_table monotone in margin.
    cond_d = bool(resolve_monotone)

    # (e) deep-depth dominant raw variance contributor.
    cond_e = bool(dominant_depth >= 5 and var_d[D_HI - D_LO] > var_d[0])  # deepest > shallowest

    # (f) NaN-clean handled at payload level; here check key scalars finite.
    key_scalars = [ci_op["halfwidth"], pool_op["halfwidth_ols"], pool_op["halfwidth_wls"],
                   per_op["E_T"], per_op["N_steps"]] + var_d + per_op["n_ladder"]
    cond_f = all(_finite(x) for x in key_scalars)

    passes = bool(cond_a and cond_b and cond_c and cond_d and cond_e and cond_f)
    return {
        "lambda_built_ci_self_test_passes": passes,
        "conditions": {
            "a_liveprobe_0p342_recovers": cond_a,
            "b_endpoints_recover_1_and_0": cond_b,
            "c_halfwidth_scales_1_over_sqrt_N": cond_c,
            "d_resolve_table_monotone_in_margin": cond_d,
            "e_deep_depth_dominates_variance": cond_e,
            "f_key_scalars_finite": cond_f,
        },
        "evidence": {
            "recover_liveprobe": rec_live, "recover_one": rec_one, "recover_zero": rec_zero,
            "scaling_1_over_sqrt_N": scaling,
            "dominant_variance_depth": dominant_depth,
            "per_depth_var_lambda": var_d,
        },
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
    st = syn["self_test"]
    op = syn["operating_point"]
    mm = syn["measurement_model"]
    lc = syn["lambda_built_ci"]
    print("\n" + "=" * 96, flush=True)
    print("MARGIN-AWARE λ̂_built MEASUREMENT-CI (PR #187) — the gate-INPUT resolvability stamp", flush=True)
    print("=" * 96, flush=True)
    print(f"  operating point: both-bugs  λ={op['lambda']:.6f} (the 0.9052 LCB bar)  "
          f"E[T]={op['E_T']:.4f}  N_steps={op['N_steps']:.0f}  ({op['n_prompts']}×{op['output_len']} bench)",
          flush=True)
    print("-" * 96, flush=True)
    print("  (1) measurement model: per-depth trial ladder n_d = N_steps·S(d)  (S = #175 pmf survival)", flush=True)
    for r in mm["per_depth"]:
        print(f"      depth {r['depth']}: q*={r['q_true']:.4f}  span={r['span']:.4f}  "
              f"S(d)={r['survival_reach']:.4f}  n_d={r['n_trials']:.0f}  "
              f"Var[q̂]={r['var_qhat']:.2e}  Var[λ̂_d]={r['var_lambda_hat_d']:.2e}", flush=True)
    print(f"      n_ladder = [{', '.join(f'{n:.0f}' for n in mm['n_ladder'])}]", flush=True)
    print("-" * 96, flush=True)
    print("  (2) λ̂_built CI at the bar:", flush=True)
    print(f"      lambda_built_halfwidth (TEST, {lc['estimator_primary']}) = ±{lc['lambda_built_halfwidth']:.6f}",
          flush=True)
    print(f"      CI = [{lc['lambda_built_ci'][0]:.4f}, {lc['lambda_built_ci'][1]:.4f}]   "
          f"OLS(#183 simple-mean)=±{lc['halfwidth_ols_simple_mean_183']:.6f}  "
          f"WLS(MLE)=±{lc['halfwidth_wls_mle']:.6f}", flush=True)
    print(f"      dominant raw-variance depth = {lc['dominant_variance_depth']} (deep-thinning)", flush=True)
    print("-" * 96, flush=True)
    print("  (3) resolvability — n_prompts so a true-λ build resolves vs 0.9052 at 95%:", flush=True)
    for r in syn["resolvability"]["resolve_table"]:
        tag = "unresolvable(ON-BAR)" if r["on_bar_unresolvable"] else f"N≥{r['n_prompts_to_resolve']}"
        print(f"      true-λ={r['true_lambda']:.3f} ({r['side']:<5} margin={r['margin']:.4f}): "
              f"{tag}   decisive@{r['n0']}={r['decisive_at_n0']}", flush=True)
    print(f"      PUNCHLINE: {syn['resolvability']['punchline']}", flush=True)
    print("-" * 96, flush=True)
    cmp = syn["input_output_compose"]
    print(f"  (4) compose input⊕output: {cmp['verdict']}  overlap_fraction={cmp['overlap_fraction']:.4f} "
          f"(ρ={cmp['rho_input_output']:.4f})", flush=True)
    print(f"      H_in(λ̂-route)=±{cmp['h_in_tps_lambda_route']:.3f} TPS  "
          f"H_out(#175 L̄-route)=±{cmp['h_out_tps_lbar_route_175']:.3f} TPS  "
          f"quad={cmp['quadrature_sum_tps']:.3f}  same-bench-corrected={cmp['overlap_corrected_same_bench_tps']:.3f}",
          flush=True)
    print("-" * 96, flush=True)
    print("  serial-correlation sensitivity (effective-N deflation):", flush=True)
    for s in syn["serial_correlation_sensitivity"]:
        print(f"      VIF={s['vif']:.1f} (ρ≈{s['implied_lag1_rho']:.2f})  halfwidth=±{s['halfwidth']:.6f}",
              flush=True)
    print("-" * 96, flush=True)
    print(f"  (5) PRIMARY lambda_built_ci_self_test_passes = {st['lambda_built_ci_self_test_passes']}",
          flush=True)
    for k, v in st["conditions"].items():
        print(f"          - {k}: {v}", flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print("=" * 96, flush=True)
    print(f"\n  HAND-OFF (land #71): {syn['handoff_lines']['land_71_protocol']}", flush=True)
    print(f"\n  HAND-OFF (fern #185): {syn['handoff_lines']['fern_185_integrator']}\n", flush=True)


# --------------------------------------------------------------------------- #
# W&B logging (mirrors #183; never fatal).
# --------------------------------------------------------------------------- #
def _maybe_log_wandb(args, payload: dict) -> None:
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
        print(f"[lambda-built-ci] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    run = init_wandb_run(
        job_type="lambda-built-ci",
        agent="denken",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["lambda-built-ci", "validity-gate", "input-ci", "resolvability", "measurement-ci"],
        config={
            "K_cal": K_CAL, "step": STEP, "z95": Z95, "sigma_hw_tps": SIGMA_HW,
            "bar_lambda_star_lcb": BOTH_BUGS_LAMBDA_STAR_LCB,
            "n_prompts": args.n_prompts, "output_len": args.output_len, "estimator": args.estimator,
            "imports": "denken#183 q_d(λ)+inverse-map + denken#178 forward-map + wirbel#175 pmf-survival",
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[lambda-built-ci] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    st = syn["self_test"]
    lc = syn["lambda_built_ci"]
    cmp = syn["input_output_compose"]
    op = syn["operating_point"]
    resolve = {f"resolve_n_lambda_{str(r['true_lambda']).replace('.', 'p')}": r["n_prompts_to_resolve"]
               for r in syn["resolvability"]["resolve_table"]}
    resolve_decisive = {f"decisive128_lambda_{str(r['true_lambda']).replace('.', 'p')}": int(r["decisive_at_n0"])
                        for r in syn["resolvability"]["resolve_table"]}
    summary: dict[str, Any] = {
        "lambda_built_ci_self_test_passes": int(bool(st["lambda_built_ci_self_test_passes"])),
        "lambda_built_halfwidth": lc["lambda_built_halfwidth"],
        "lambda_built_halfwidth_ols": lc["halfwidth_ols_simple_mean_183"],
        "lambda_built_halfwidth_wls": lc["halfwidth_wls_mle"],
        "lambda_built_ci_lo": lc["lambda_built_ci"][0],
        "lambda_built_ci_hi": lc["lambda_built_ci"][1],
        "dominant_variance_depth": lc["dominant_variance_depth"],
        "operating_lambda": op["lambda"], "operating_E_T": op["E_T"], "N_steps": op["N_steps"],
        "n_depth_2_trials": syn["measurement_model"]["n_ladder"][0],
        "n_depth_9_trials": syn["measurement_model"]["n_ladder"][-1],
        # compose
        "input_output_overlap_fraction": cmp["overlap_fraction"],
        "input_output_rho": cmp["rho_input_output"],
        "h_in_tps_lambda_route": cmp["h_in_tps_lambda_route"],
        "h_out_tps_lbar_route_175": cmp["h_out_tps_lbar_route_175"],
        "compose_quadrature_tps": cmp["quadrature_sum_tps"],
        "compose_corrected_same_bench_tps": cmp["overlap_corrected_same_bench_tps"],
        # sensitivity
        "halfwidth_vif2": syn["serial_correlation_sensitivity"][-1]["halfwidth"],
        "resolve_monotone_in_margin": int(bool(syn["resolvability"]["resolve_monotone_in_margin"])),
        "K_cal": K_CAL, "sigma_hw_tps": SIGMA_HW,
        "verdict_stamped": int(syn["verdict"].startswith("INPUT-CI-STAMPED")),
        **resolve, **resolve_decisive,
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items() if not (isinstance(v, float) and not math.isfinite(v))}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="lambda_built_ci_result", artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[lambda-built-ci] wandb logged: {summary}", flush=True)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--n-prompts", type=int, default=DEFAULT_N_PROMPTS)
    ap.add_argument("--output-len", type=int, default=DEFAULT_OUTPUT_LEN)
    ap.add_argument("--estimator", choices=("wls", "ols"), default="wls")
    ap.add_argument("--bug2-anchor", type=Path, default=D172.DEFAULT_BUG2_ANCHOR)
    ap.add_argument("--topo-json", type=Path, default=D172.DEFAULT_TOPO_JSON)
    ap.add_argument("--accept-json", type=Path, default=D172.DEFAULT_ACCEPT_JSON)
    ap.add_argument("--rankcov-json", type=Path, default=D172.DEFAULT_RANKCOV_JSON)
    ap.add_argument("--decomp-json", type=Path, default=D172.DEFAULT_DECOMP_JSON)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="lambda-built-ci")
    args = ap.parse_args(argv)

    anchors = D172.load_anchors(
        args.bug2_anchor, args.topo_json, args.accept_json, args.rankcov_json, args.decomp_json
    )
    syn = synthesize(anchors, n_prompts=args.n_prompts, output_len=args.output_len,
                     estimator=args.estimator)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at,
        "pr": 187,
        "agent": "denken",
        "kind": "lambda-built-ci",
        "anchors": {k: v for k, v in anchors.items() if k != "_paths"},
        "anchor_paths": anchors.get("_paths"),
        "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }

    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[lambda-built-ci] WARNING non-finite values at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "lambda_built_ci_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[lambda-built-ci] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = syn["self_test"]["lambda_built_ci_self_test_passes"] and payload["nan_clean"]
        print(f"[lambda-built-ci] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
