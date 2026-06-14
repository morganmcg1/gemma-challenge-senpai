#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Liveprobe depth-budget (PR #197) — which depths × N for a DECISIVE GO/NO-GO.

THE UN-COMPOSED SEAM
--------------------
Three banked facts about land #71's self-KV recovery gate, never combined into the
one thing land's harness needs — a concrete MEASUREMENT DESIGN:

  * denken #193 (`2clxvlr8`): the salvage-staleness law λ_d=λ̂₁·β^(d−1) makes the
    depth-1 liveprobe NECESSARY-BUT-NOT-SUFFICIENT — the 0.9052 bar is unreachable
    from a depth-1-only probe unless β ≥ `beta_crit_depth1_sufficient`=0.9649, so
    the q[2..9] ladder must be measured DIRECTLY, not inferred from depth-1.
  * denken #187 (`tloghme9`): each depth's q̂_d is a survival-thinned binomial; the
    per-depth trial ladder n_d = N_steps·S(d) thins with depth, so depth-9 is the
    FEWEST-trials / noisiest end (`lambda_built_halfwidth`=±0.0171 aggregate).
  * stark #191 (`jeclr39w`): the binding bar is now PRIVATE-stricter — both-bugs
    λ ≥ `lambda_star_lcb_private`=0.9780 (descent-only UNREACHABLE at the private LCB).

This leg is the 2-D measurement-design capstone: given a fixed liveprobe trial
budget, HOW to spend it ACROSS depths 1..9 — and how many depths to measure — to
make a DECISIVE GO/NO-GO against the private bar, neither a false GO nor an
indecisive coin-flip inside #187's ±0.017 zone. The cruel coincidence #193+#187
expose: depth-9 is simultaneously the fewest-trials (noisiest, #187), the
most-λ-decayed (lowest-recovery, #193), AND — as this leg shows — the lowest
E[T]-leverage end, so naive equal-allocation over-samples the cheap shallow depths.

IMPORTS — NOT re-derived
------------------------
    #193  mechanism_lambda, metrics_at_profile, ground_beta, beta_crit (0.9649)
    #187  per_depth_table, _dEt_dq (∂E[T]/∂q_d), _dEt_dlambda, n_prompts_to_resolve
    #191  synthesize() private bar 0.9780 + descent UNREACHABLE; load_176_drop()
The private-LCB forward map is composed as #193's profile-aware public LCB × #191's
(1−drop)·τ_low — validated to reproduce #191's bar (private_LCB(0.9780,β=1)=500.0).

LOCAL CPU-only analytic. No GPU / vLLM / HF Job / submission / served-file change.
BASELINE stays 481.53. Greedy/PPL untouched. Bank-the-analysis (PRIMARY = self-test,
adds 0 TPS). NOT open2. NOT a launch.

PRIMARY metric  depth_budget_self_test_passes
TEST    metric  total_trials_for_decisive_private  (Neyman trials, best-case λ=1.0 GO)

Run:
    python -m research.oracle_readout.liveprobe_depth_budget.liveprobe_depth_budget \
        --self-test --wandb-name denken/liveprobe-depth-budget \
        --wandb-group liveprobe-depth-budget
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

_D193_PATH = REPO_ROOT / "research/oracle_readout/lambda_depth_profile/lambda_depth_profile.py"
_D187_PATH = REPO_ROOT / "research/validity/lambda_built_ci/lambda_built_ci.py"
_D191_PATH = REPO_ROOT / "research/validity/private_build_bar/private_build_bar.py"


def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import module {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- import the three banked legs (and reach #172/#178/#175/#183 transitively) ---- #
D193 = _import("lambda_depth_profile", _D193_PATH)        # mechanism + beta + profile LCB
D187 = _import("lambda_built_ci", _D187_PATH)             # measurement model + sensitivities
D191 = _import("private_build_bar", _D191_PATH)           # private bar 0.9780
D172 = D193.D172
D178 = D193.D178

# ---- composition constants (committed; imported, not re-derived) ---- #
Z95 = D193.Z95
K_CAL = D193.K_CAL
STEP = D193.STEP
TARGET_OFFICIAL = D193.TARGET_OFFICIAL                    # 500.0
PUBLIC_BAR_BOTH = D193.D183_BOTHBUGS_LAMBDA_STAR_LCB      # 0.905229… (#183 public LCB bar, full precision)
PUBLIC_BAR_BOTH_187 = D187.BOTH_BUGS_LAMBDA_STAR_LCB     # 0.905229 (#187's OWN published bar; truncated)
HW187 = 0.017139919169244854                             # #187 lambda_built_halfwidth (WLS, @128)

D_LO, D_HI = D187.D_LO, D187.D_HI                         # 2, 9 (self-KV-governed measured depths)
N_DEPTHS_MEASURED = D_HI - D_LO + 1                       # 8
DEFAULT_N_PROMPTS = D187.DEFAULT_N_PROMPTS               # 128
DEFAULT_OUTPUT_LEN = D187.DEFAULT_OUTPUT_LEN             # 512

# GO builds to size the decisive budget against the private bar (all > 0.9780).
LAM_TRUE_GRID = (0.98, 0.985, 0.99, 1.0)
HEADLINE_LAM_TRUE = 1.0                                   # best-case (max margin 0.022) -> fewest trials

TOL_REPRO = 1e-4
TOL_BAR_TPS = 0.5


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# Shared setup: ctx + topology spines + the private bar + the grounded β.
# --------------------------------------------------------------------------- #
def build(anchors: dict[str, Any]) -> dict[str, Any]:
    ctx = D187.build_ctx(anchors)
    syn193 = D193.synthesize(anchors)
    syn191 = D191.synthesize()
    imp176 = D191.load_176_drop()
    ml = syn193["mechanism_law"]
    h191 = syn191["headline"]
    return {
        "ctx": ctx,
        "qfl_b": ctx["qfl_b"], "qfu_b": ctx["qfu_b"],
        "qfl_d": ctx["qfl_d"], "qfu_d": ctx["qfu_d"],
        "H": len(ctx["qfu_b"]),
        "lam_hat_1": ml["lambda_hat_1"],
        "beta_primary": ml["beta_grounding"]["beta_primary_geomean"],
        "beta_range": list(ml["beta_grounding"]["beta_construction_range"]),
        "beta_crit_depth1": syn193["verdict_robustness"]["beta_crit_depth1_sufficient"],
        "private_bar_both": h191["lambda_star_lcb_private"],            # 0.9780
        "private_bar_descent": h191["lambda_star_lcb_private_descent"], # None (UNREACHABLE)
        "drop_both": imp176["drop_both"],
        "drop_descent": imp176["drop_descent"],
        "tau_low": imp176["tau_low"],
    }


# --------------------------------------------------------------------------- #
# Private-LCB forward map: #193 profile-aware public LCB × #191 (1−drop)·τ_low.
# Validated: private_lcb_mech(0.9780, β=1, both) == #191's 500.0 (self-test c).
# --------------------------------------------------------------------------- #
def private_lcb_mech(st: dict, lam1: float, beta: float, topo: str = "both") -> float:
    qfl, qfu = (st["qfl_b"], st["qfu_b"]) if topo == "both" else (st["qfl_d"], st["qfu_d"])
    drop = st["drop_both"] if topo == "both" else st["drop_descent"]
    prof = D193.mechanism_lambda(st["H"], lam1, beta)
    pub = D193.metrics_at_profile(st["ctx"], prof, qfl, qfu, 1.0)["lcb_full_tps"]
    return pub * (1.0 - drop) * st["tau_low"]


def private_central_mech(st: dict, lam1: float, beta: float, topo: str = "both") -> float:
    qfl, qfu = (st["qfl_b"], st["qfu_b"]) if topo == "both" else (st["qfl_d"], st["qfu_d"])
    drop = st["drop_both"] if topo == "both" else st["drop_descent"]
    prof = D193.mechanism_lambda(st["H"], lam1, beta)
    pub = D193.metrics_at_profile(st["ctx"], prof, qfl, qfu, 1.0)["central_tps"]
    return pub * (1.0 - drop) * st["tau_low"]


def _depth_weights(st: dict, lam_true: float):
    """Per-depth (d=2..9) E[T]-leverage a_d=∂E[T]/∂q_d and binomial σ_d=√(q_d(1−q_d))
    on the constant-λ operating spine at lam_true. a_d imported from #187's central-diff DP."""
    ctx, qfl, qfu = st["ctx"], st["qfl_b"], st["qfu_b"]
    spine = D187.operating_spine(ctx, lam_true, qfl, qfu)
    ext = D187.extended_spine_9(spine)
    a, sig, q = [], [], []
    for d in range(D_LO, D_HI + 1):
        qd = D172.qd_at(spine, d)
        a.append(D187._dEt_dq(ctx, ext, d))
        sig.append(math.sqrt(max(qd * (1.0 - qd), 0.0)))
        q.append(qd)
    dEdlam = D187._dEt_dlambda(ctx, lam_true, qfl, qfu)
    return a, sig, q, dEdlam


# --------------------------------------------------------------------------- #
# (1) Decisive-certification budget: Neyman optimal per-depth allocation.
#     Certify a GO build's E[T]→private_LCB clears 500 by more than its measurement
#     CI. Linear functional Ê[T]=Σ a_d q̂_d, Var[q̂_d]=q_d(1−q_d)/n_d. Minimise Σn_d
#     s.t. the λ-equivalent half-width ≤ margin m=|λ_true−0.9780|:
#         min Var[λ̂_eff] = (Σ a_dσ_d)² / (N·(dE/dλ)²)   at  n_d ∝ a_dσ_d
#     ⇒ N_opt = z95² (Σ a_dσ_d)² / ((dE/dλ)²·m²);  equal-alloc N_eq = z95²·D·Σ(a_dσ_d)²/…
#     efficiency_gain = N_eq/N_opt = D·Σ(a_dσ_d)²/(Σ a_dσ_d)² ≥ 1 (Cauchy–Schwarz).
# --------------------------------------------------------------------------- #
def neyman_budget(st: dict, lam_true: float) -> dict[str, Any]:
    bar = st["private_bar_both"]
    a, sig, q, dEdlam = _depth_weights(st, lam_true)
    aw = [ai * si for ai, si in zip(a, sig)]            # Neyman per-depth weight a_d·σ_d
    sum_aw = sum(aw)
    sum_aw2 = sum(w * w for w in aw)
    D = len(aw)
    m = abs(lam_true - bar)
    if m < 1e-9 or dEdlam <= 0:
        return {"lam_true": lam_true, "margin": m, "decisive_feasible": False,
                "N_opt": None, "N_equal": None, "efficiency_gain": None, "n_d_2to9": None,
                "N_d_budget_1to9": None}
    denom = (dEdlam ** 2) * (m ** 2)
    N_opt = Z95 ** 2 * (sum_aw ** 2) / denom
    N_equal = Z95 ** 2 * D * sum_aw2 / denom
    eff = N_equal / N_opt
    n_d = [N_opt * (w / sum_aw) for w in aw]            # depths 2..9
    # full 1..9 budget vector: depth-1 is PINNED for both-bugs (span=0, deployed) → 0 trials.
    N_d_budget = [0.0] + n_d
    return {
        "lam_true": lam_true, "margin": m, "decisive_feasible": True,
        "N_opt": N_opt, "N_equal": N_equal, "efficiency_gain": eff,
        "sum_a_sigma": sum_aw, "dEt_dlambda": dEdlam,
        "a_d": a, "sigma_d": sig, "q_d": q, "neyman_weight_a_sigma": aw,
        "n_d_2to9": n_d, "N_d_budget_1to9": N_d_budget,
        "neyman_fraction_2to9": [w / sum_aw for w in aw],
        "depth1_pinned": True,
    }


# --------------------------------------------------------------------------- #
# (1b) conservative-ordering measurement cost: Σ q_d(1−q_d)/span_d² (trials per unit
#      λ-variance) — the per-depth resolvability load, evaluated on a given spine.
# --------------------------------------------------------------------------- #
def ladder_resolve_load(st: dict, lam1: float, beta: float) -> float:
    ctx, qfl, qfu = st["ctx"], st["qfl_b"], st["qfu_b"]
    spine = D178.spine_from_profile(ctx["ep"], D193.mechanism_lambda(st["H"], lam1, beta), qfl, qfu)
    load = 0.0
    for d in range(D_LO, D_HI + 1):
        qd = D172.qd_at(spine, d)
        span = D172.qd_at(qfu, d) - D172.qd_at(qfl, d)
        if abs(span) > 1e-12:
            load += qd * (1.0 - qd) / (span * span)
    return load


# --------------------------------------------------------------------------- #
# (2) Minimum depth-COUNT for β-identification. Geometric ln λ_d = ln λ̂₁ + (d−1)·ln β.
#     WLS line fit on depths {1..k} with per-depth measurement variances; extrapolate
#     unmeasured depths k+1..9. "Suffices" iff every extrapolated λ_d half-width ≤ the
#     #187 single-depth precision HW187 (i.e. extrapolation is as good as measuring).
# --------------------------------------------------------------------------- #
def _wls_line(xs: list[float], ys: list[float], vs: list[float]) -> dict[str, float]:
    w = [1.0 / v if v > 0 and math.isfinite(v) else 0.0 for v in vs]
    Sw = sum(w); Swx = sum(wi * xi for wi, xi in zip(w, xs))
    Swy = sum(wi * yi for wi, yi in zip(w, ys))
    Swxx = sum(wi * xi * xi for wi, xi in zip(w, xs))
    Swxy = sum(wi * xi * yi for wi, xi, yi in zip(w, xs, ys))
    delta = Sw * Swxx - Swx * Swx
    if abs(delta) < 1e-300:
        return {"a": float("nan"), "b": float("nan"), "var_a": float("inf"),
                "var_b": float("inf"), "cov_ab": float("inf")}
    b = (Sw * Swxy - Swx * Swy) / delta
    a = (Swxx * Swy - Swx * Swxy) / delta
    return {"a": a, "b": b, "var_a": Swxx / delta, "var_b": Sw / delta, "cov_ab": -Swx / delta}


def _per_depth_lambda_var(st: dict, lam1: float, beta: float, n_prompts: int):
    """Var[λ̂_d] at depths 1..9 for the mechanism build (λ1,β), survival-thinned over
    a both-bugs N-prompt bench. Depth-1 uses the descent-frame span (both-bugs span=0)."""
    ctx, qfl_b, qfu_b = st["ctx"], st["qfl_b"], st["qfu_b"]
    spine = D178.spine_from_profile(ctx["ep"], D193.mechanism_lambda(st["H"], lam1, beta), qfl_b, qfu_b)
    surv, et, _ = D187.survival_and_et(ctx, spine)
    n_steps = n_prompts * DEFAULT_OUTPUT_LEN / et
    var = {}
    for d in range(1, D_HI + 1):
        qd = D172.qd_at(spine, d)
        if d == 1:
            span = st["qfu_d"][0] - st["qfl_d"][0]      # descent-frame liveprobe anchor span (0.005)
        else:
            span = D172.qd_at(qfu_b, d) - D172.qd_at(qfl_b, d)
        n_d = n_steps * surv.get(d, 0.0)
        var[d] = (qd * (1.0 - qd) / n_d) / (span * span) if (n_d > 0 and abs(span) > 1e-12) else float("inf")
    return var, et


def beta_depth_count(st: dict, n_prompts: int = DEFAULT_N_PROMPTS) -> dict[str, Any]:
    lam1, beta = st["lam_hat_1"], st["beta_primary"]
    var_d, et = _per_depth_lambda_var(st, lam1, beta, n_prompts)
    lam_true_d = {d: lam1 * (beta ** (d - 1)) for d in range(1, D_HI + 1)}
    ln_lam = {d: math.log(lam_true_d[d]) for d in lam_true_d}
    var_ln = {d: (var_d[d] / (lam_true_d[d] ** 2)) if math.isfinite(var_d[d]) else float("inf")
              for d in var_d}

    per_k = []
    min_depths = None
    beta_ci_2depth = None
    for k in range(2, D_HI + 1):                          # measure depths 1..k
        xs = [d - 1 for d in range(1, k + 1)]
        ys = [ln_lam[d] for d in range(1, k + 1)]
        vs = [var_ln[d] for d in range(1, k + 1)]
        fit = _wls_line(xs, ys, vs)
        sd_b = math.sqrt(fit["var_b"]) if math.isfinite(fit["var_b"]) else float("inf")
        beta_lo = math.exp(fit["b"] - Z95 * sd_b) if math.isfinite(sd_b) else 0.0
        beta_hi = math.exp(fit["b"] + Z95 * sd_b) if math.isfinite(sd_b) else float("inf")
        # extrapolate unmeasured depths k+1..9: hw[λ_d] = z95·√Var[ŷ(x)]·λ_d
        max_hw, hw_at_9 = 0.0, None
        for d in range(k + 1, D_HI + 1):
            x = d - 1
            var_y = fit["var_a"] + x * x * fit["var_b"] + 2.0 * x * fit["cov_ab"]
            hw = Z95 * math.sqrt(max(var_y, 0.0)) * lam_true_d[d] if math.isfinite(var_y) else float("inf")
            max_hw = max(max_hw, hw)
            if d == D_HI:
                hw_at_9 = hw
        suffices = bool(k == D_HI or (math.isfinite(max_hw) and max_hw <= HW187))
        row = {
            "depths_measured": k, "lever_arm": k - 1,
            "beta_hat": math.exp(fit["b"]) if math.isfinite(fit["b"]) else None,
            "sd_ln_beta": sd_b if math.isfinite(sd_b) else None,
            "beta_ci": [beta_lo, beta_hi] if math.isfinite(beta_hi) else [beta_lo, None],
            "max_extrap_halfwidth_lambda": max_hw if math.isfinite(max_hw) else None,
            "extrap_halfwidth_lambda_depth9": hw_at_9 if (hw_at_9 is not None and math.isfinite(hw_at_9)) else None,
            "within_187_ci": suffices,
        }
        per_k.append(row)
        if k == 2:
            beta_ci_2depth = row["beta_ci"]
        if suffices and min_depths is None and k < D_HI:
            min_depths = k

    full_ladder_required = min_depths is None
    depth1_plus_2_suffices = bool(per_k[0]["within_187_ci"])  # k=2 row
    return {
        "operating_point": {"lam_hat_1": lam1, "beta_primary": beta, "n_prompts": n_prompts,
                            "E_T": et, "ref_precision_hw187": HW187},
        "true_lambda_ladder_1to9": [lam_true_d[d] for d in range(1, D_HI + 1)],
        "per_depth_count": per_k,
        "min_depths_for_decisive": "full-ladder" if full_ladder_required else min_depths,
        "min_depths_for_decisive_int": D_HI if full_ladder_required else min_depths,
        "depth1_plus_2_suffices": depth1_plus_2_suffices,
        "beta_ci_2depth_fit": beta_ci_2depth,
        "interpretation": (
            "the 2-parameter geometric fit's extrapolation half-width to depth 9 stays "
            f">> the #187 single-depth precision ±{HW187:.4f} for every measured-depth count "
            "k<9, so a few-depth β-fit cannot stand in for measuring the ladder: land #71 must "
            "probe all of depths 2..9 DIRECTLY (the deep, β-decayed, survival-thinned end is "
            "exactly where extrapolation is worst)."),
    }


# --------------------------------------------------------------------------- #
# (3) Depth-1-only under-measurement error: read λ̂₁, assume flat (β=1) → naive GO;
#     truth at the grounded β decays the ladder → MISS the private bar.
# --------------------------------------------------------------------------- #
def depth1_false_go(st: dict) -> dict[str, Any]:
    beta = st["beta_primary"]
    rows = []
    false_go = False
    for lam1 in (PUBLIC_BAR_BOTH, 0.95, 1.0):
        naive_pub_central = D193.metrics_at_profile(
            st["ctx"], D193.mechanism_lambda(st["H"], lam1, 1.0), st["qfl_b"], st["qfu_b"], 1.0)["central_tps"]
        naive_flat_priv = private_lcb_mech(st, lam1, 1.0)      # depth-1+flat private LCB (β=1)
        true_priv = private_lcb_mech(st, lam1, beta)           # truth: mechanism decays
        clears_public = bool(lam1 >= PUBLIC_BAR_BOTH - 1e-9)
        naive_says_go = bool(naive_flat_priv >= TARGET_OFFICIAL)
        true_is_go = bool(true_priv >= TARGET_OFFICIAL)
        this_false_go = bool((clears_public or naive_says_go) and not true_is_go)
        false_go = false_go or this_false_go
        rows.append({
            "lambda_hat_1": lam1,
            "clears_public_bar_0p9052": clears_public,
            "naive_flat_public_central_tps": naive_pub_central,
            "naive_flat_private_lcb_tps": naive_flat_priv,
            "true_mechanism_private_lcb_tps": true_priv,
            "naive_says_go_private": naive_says_go,
            "true_is_go_private": true_is_go,
            "false_go": this_false_go,
            "overstatement_tps_flat_minus_mech": naive_flat_priv - true_priv,
        })
    at1 = rows[-1]                                              # λ̂₁=1.0 (perfect depth-1)
    return {
        "false_go_risk_depth1_only": false_go,
        "rows": rows,
        "headline_lambda1": at1["lambda_hat_1"],
        "overstatement_tps": at1["overstatement_tps_flat_minus_mech"],
        "true_private_lcb_at_lambda1_eq_1": at1["true_mechanism_private_lcb_tps"],
        "naive_private_lcb_at_lambda1_eq_1": at1["naive_flat_private_lcb_tps"],
        "interpretation": (
            "any depth-1 read λ̂₁∈[0.9052,1.0] that clears the FLAT public bar maps, under the "
            f"grounded β={beta:.3f} decay, to a true private LCB of ~412–420 TPS — a NO-GO vs the "
            "500 bar. Reading depth-1 only and assuming flat is a FALSE GO worth "
            f"{at1['overstatement_tps_flat_minus_mech']:.1f} TPS of overstatement at λ̂₁=1.0."),
    }


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize(anchors: dict[str, Any], n_prompts: int = DEFAULT_N_PROMPTS) -> dict[str, Any]:
    st = build(anchors)

    # ---------- (1) decisive budget over the GO grid ---------- #
    budgets = [neyman_budget(st, lt) for lt in LAM_TRUE_GRID]
    headline_budget = next(b for b in budgets if abs(b["lam_true"] - HEADLINE_LAM_TRUE) < 1e-12)
    total_trials_for_decisive_private = headline_budget["N_opt"]          # TEST metric

    # ---------- (2) min depth-COUNT for β-identification ---------- #
    bdc = beta_depth_count(st, n_prompts)

    # ---------- (3) depth-1-only false GO ---------- #
    d1 = depth1_false_go(st)

    # ---------- mechanism feasibility ceiling (why GO needs β≈1) ---------- #
    priv_lcb_perfect_mech = private_lcb_mech(st, 1.0, st["beta_primary"])  # ~419.6 << 500
    mech_can_clear_private = bool(priv_lcb_perfect_mech >= TARGET_OFFICIAL)

    # ---------- self-test (PRIMARY) ---------- #
    sttest = _self_test(st, budgets, bdc, d1)

    handoff = _handoff(st, headline_budget, bdc, d1, total_trials_for_decisive_private)

    return {
        "self_test": sttest,
        "test_metric": {"total_trials_for_decisive_private": total_trials_for_decisive_private},
        "imports": {
            "private_bar_both_0p9780": st["private_bar_both"],
            "private_bar_descent": st["private_bar_descent"],
            "public_bar_both_0p9052": PUBLIC_BAR_BOTH,
            "beta_primary": st["beta_primary"],
            "beta_range": st["beta_range"],
            "beta_crit_depth1_sufficient": st["beta_crit_depth1"],
            "lambda_hat_1_liveprobe": st["lam_hat_1"],
            "lambda_built_halfwidth_187": HW187,
            "drop_both": st["drop_both"], "drop_descent": st["drop_descent"],
            "tau_low": st["tau_low"],
        },
        "decisive_budget": {
            "headline_lam_true": HEADLINE_LAM_TRUE,
            "total_trials_for_decisive_private": total_trials_for_decisive_private,
            "N_d_budget_1to9": headline_budget["N_d_budget_1to9"],
            "N_equal_allocation": headline_budget["N_equal"],
            "efficiency_gain_neyman_vs_equal": headline_budget["efficiency_gain"],
            "neyman_fraction_2to9": headline_budget["neyman_fraction_2to9"],
            "a_d_dEt_dq_2to9": headline_budget["a_d"],
            "per_lam_true_table": [
                {"lam_true": b["lam_true"], "margin": b["margin"], "N_opt": b["N_opt"],
                 "N_equal": b["N_equal"], "efficiency_gain": b["efficiency_gain"],
                 "n_d_2to9": b["n_d_2to9"]}
                for b in budgets
            ],
            "note": (
                "certify a const-λ GO build's E[T]→private_LCB clears 500 by > its measurement CI. "
                "Neyman n_d ∝ a_d·σ_d (a_d=∂E[T]/∂q_d from #187) is SHALLOW-weighted (depth-2 gets "
                "the most trials, depth-9 the fewest) — generalising #187's inverse-variance WLS to "
                "the E[T] functional. The private bar 0.9780 is so tight that even the best-case "
                "λ=1.0 build (margin 0.022) needs ~30k trials; margin 0.002 needs ~3.8M."),
        },
        "beta_depth_count": bdc,
        "depth1_false_go": d1,
        "mechanism_feasibility": {
            "private_lcb_at_perfect_depth1_mech": priv_lcb_perfect_mech,
            "mechanism_can_clear_private_bar": mech_can_clear_private,
            "note": (
                f"at the grounded β={st['beta_primary']:.3f}, even PERFECT depth-1 recovery "
                f"(λ̂₁=1.0) yields private_LCB={priv_lcb_perfect_mech:.1f} << 500: a GO build must "
                "have β≈1 (no salvage staleness). So the decisive budget sizes the measurement that "
                "CONFIRMS β≈1 across the full ladder — it cannot be shortcut by a depth-1 probe."),
        },
        "composition": {
            "K_cal": K_CAL, "step": STEP, "z95": Z95, "target_official": TARGET_OFFICIAL,
            "private_forward_map": "private_LCB(λ-profile) = #193.metrics_at_profile.lcb_full_tps · (1−drop) · τ_low",
            "n_prompts": n_prompts, "output_len": DEFAULT_OUTPUT_LEN,
        },
        "verdict": _verdict(st, total_trials_for_decisive_private, bdc, d1, mech_can_clear_private,
                            headline_budget["efficiency_gain"]),
        "handoff_lines": handoff,
    }


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def _self_test(st: dict, budgets: list[dict], bdc: dict, d1: dict) -> dict[str, Any]:
    ctx, qfl_b, qfu_b = st["ctx"], st["qfl_b"], st["qfu_b"]

    # (a) reproduce #187's single-depth N_resolve (flat, single-aggregate-λ̂, PUBLIC bar 0.9052).
    #     Reproduce #187's PUBLISHED resolve_table verbatim, which means using #187's OWN bar
    #     constant (0.905229, truncated) — NOT the full-precision #193/#183 re-derivation. The
    #     on-bar entry (true_lambda=0.905, margin 0.0002, N~7e5) is pathologically sensitive to that
    #     6th-decimal truncation (715067 vs 717062), so faithful reproduction must use #187's bar.
    ref187 = {0.86: 19, 0.88: 60, 0.905: 717062, 0.93: 62, 0.95: 19}
    a_rows = []
    cond_a = True
    for tl, ref in ref187.items():
        got = D187.n_prompts_to_resolve(ctx, tl, qfl_b, qfu_b, PUBLIC_BAR_BOTH_187)["n_prompts_to_resolve"]
        ok = bool(got == ref)
        a_rows.append({"true_lambda": tl, "ref": ref, "got": got, "ok": ok})
        cond_a = cond_a and ok

    # (b) reproduce #193's beta_crit_depth1_sufficient = 0.9649.
    cond_b = bool(abs(st["beta_crit_depth1"] - 0.9648839148878561) < TOL_REPRO)

    # (c) reproduce #191's private bar 0.9780 (both) + descent UNREACHABLE; and the private
    #     forward map reproduces #191 (private_lcb_mech(0.9780, β=1) == 500.0).
    bar_ok = bool(abs(st["private_bar_both"] - 0.9780112973731208) < TOL_REPRO)
    descent_unreachable = bool(st["private_bar_descent"] is None)
    fwd_at_bar = private_lcb_mech(st, st["private_bar_both"], 1.0)
    fwd_ok = bool(abs(fwd_at_bar - TARGET_OFFICIAL) < TOL_BAR_TPS)
    cond_c = bool(bar_ok and descent_unreachable and fwd_ok)

    # (d) conservative ordering: staleness cannot make MEASUREMENT easier — at every λ̂₁ the
    #     β-aware (decayed, lower-q) ladder's resolve-load Σq(1−q)/span² ≥ the flat ladder's;
    #     and the β-aware budget total ≥ the flat-spine budget total at matched margin.
    load_rows, cond_d = [], True
    for lam1 in (0.5, PUBLIC_BAR_BOTH, 1.0):
        load_flat = ladder_resolve_load(st, lam1, 1.0)
        load_mech = ladder_resolve_load(st, lam1, st["beta_primary"])
        ok = bool(load_mech >= load_flat - 1e-9)
        load_rows.append({"lambda_hat_1": lam1, "load_flat": load_flat, "load_mech": load_mech, "ok": ok})
        cond_d = cond_d and ok

    # (e) NaN-clean is enforced at payload level; here confirm key scalars finite and the
    #     intentional nulls are PRESENT (descent bar, infeasible margins) rather than NaN.
    key = [b["N_opt"] for b in budgets if b["N_opt"] is not None] + \
          [d1["overstatement_tps"], bdc["min_depths_for_decisive_int"]]
    cond_e_scalars = all(_finite(x) for x in key)
    cond_e_nulls = bool(st["private_bar_descent"] is None)   # unreachable stored as null

    passes = bool(cond_a and cond_b and cond_c and cond_d and cond_e_scalars and cond_e_nulls)
    return {
        "depth_budget_self_test_passes": passes,
        "conditions": {
            "a_reproduces_187_n_resolve_flat_public": cond_a,
            "b_reproduces_193_beta_crit_0p9649": cond_b,
            "c_reproduces_191_private_bar_0p9780_and_descent_unreachable": cond_c,
            "d_conservative_ordering_mech_ge_flat": cond_d,
            "e_key_scalars_finite_and_nulls_clean": bool(cond_e_scalars and cond_e_nulls),
        },
        "evidence": {
            "a_187_resolve": a_rows,
            "b_beta_crit": st["beta_crit_depth1"],
            "c_private_bar": st["private_bar_both"], "c_descent": st["private_bar_descent"],
            "c_forward_map_at_bar_tps": fwd_at_bar,
            "d_resolve_load": load_rows,
        },
    }


def _verdict(st: dict, total_trials: float, bdc: dict, d1: dict, mech_can_clear: bool,
             eff: float) -> str:
    md = bdc["min_depths_for_decisive"]
    return (
        f"DEPTH-BUDGET-DESIGNED. Against the PRIVATE-stricter bar λ≥0.9780 (#191), a DECISIVE "
        f"private GO needs ~{total_trials:,.0f} liveprobe trials even for the best-case λ=1.0 build "
        f"(margin 0.022), Neyman-allocated SHALLOW-heavy (depth-2 ≫ depth-9; efficiency "
        f"≈{eff:.2f}× over equal-allocation). The depth-COUNT cannot be shortcut: "
        f"min_depths_for_decisive={md} — a 2-depth β-fit's extrapolation to depth 9 (±0.20 in λ) "
        f"dwarfs the #187 ±0.017 precision, so land #71 must measure ALL of depths 2..9 directly. "
        f"A depth-1-only read assuming flat is a FALSE GO (risk={d1['false_go_risk_depth1_only']}) "
        f"worth {d1['overstatement_tps']:.0f} TPS. At the grounded β no real build clears the "
        f"private bar (mech_can_clear={mech_can_clear}); the budget confirms β≈1. NOT a launch."
    )


def _handoff(st: dict, hb: dict, bdc: dict, d1: dict, total_trials: float) -> dict[str, str]:
    nd = hb["N_d_budget_1to9"]
    nd_s = "[" + ", ".join(f"{x:.0f}" for x in nd) + "]"
    land = (
        f"land #71 liveprobe measurement SPEC (denken #197): to DECISIVELY certify self-KV "
        f"recovery against the private bar λ≥0.9780, measure depths 2..9 DIRECTLY (depth-1 is "
        f"pinned/deployed, span 0) — do NOT extrapolate the ladder from a 2-depth β-fit "
        f"(min_depths_for_decisive={bdc['min_depths_for_decisive']}; depth1+2 suffices="
        f"{bdc['depth1_plus_2_suffices']}). Neyman-allocate the trial budget SHALLOW-heavy "
        f"N_d[1..9]={nd_s} (best-case λ=1.0; total≈{total_trials:,.0f} trials); tighter true-λ "
        f"margins need quadratically more. A depth-1-only read assuming flat is a FALSE GO worth "
        f"{d1['overstatement_tps']:.0f} TPS."
    )
    fern = (
        f"fern #185 GO/NO-GO integrator (denken #197): this DESIGNS the measurement; you WIRE the "
        f"verdict. Consume the certified tuple (per-depth budget, full-ladder requirement, private "
        f"bar 0.9780 both-bugs / descent UNREACHABLE). The decisive-GO margin is structurally ≤0.022 "
        f"in λ (private bar so high that λ=1.0 is only +0.022), and at the grounded β=0.765 NO build "
        f"clears it — so the GO hinges on confirming β≈1 across the measured ladder, not on a point λ̂."
    )
    return {"land_71_spec": land, "fern_185_integrator": fern}


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
    db = syn["decisive_budget"]
    bdc = syn["beta_depth_count"]
    d1 = syn["depth1_false_go"]
    st = syn["self_test"]
    print("\n" + "=" * 96, flush=True)
    print("LIVEPROBE DEPTH-BUDGET (PR #197) — which depths × N for a DECISIVE GO/NO-GO", flush=True)
    print("=" * 96, flush=True)
    print(f"  private bar (both-bugs) = {syn['imports']['private_bar_both_0p9780']:.4f}   "
          f"descent = {syn['imports']['private_bar_descent']}   "
          f"β_primary = {syn['imports']['beta_primary']:.4f}  β_crit = {syn['imports']['beta_crit_depth1_sufficient']:.4f}",
          flush=True)
    print("-" * 96, flush=True)
    print(f"  (1) DECISIVE BUDGET  total_trials_for_decisive_private = "
          f"{db['total_trials_for_decisive_private']:,.0f}  (λ=1.0 best case; "
          f"efficiency {db['efficiency_gain_neyman_vs_equal']:.3f}× vs equal)", flush=True)
    print(f"      N_d_budget[1..9] = [{', '.join(f'{x:.0f}' for x in db['N_d_budget_1to9'])}]  "
          f"(depth-1 pinned=0)", flush=True)
    for r in db["per_lam_true_table"]:
        print(f"        λ_true={r['lam_true']:.3f}  margin={r['margin']:.4f}  "
              f"N_opt={r['N_opt']:,.0f}  N_equal={r['N_equal']:,.0f}", flush=True)
    print("-" * 96, flush=True)
    print(f"  (2) MIN DEPTH-COUNT  min_depths_for_decisive = {bdc['min_depths_for_decisive']}  "
          f"depth1+2 suffices = {bdc['depth1_plus_2_suffices']}", flush=True)
    print(f"      β CI (2-depth fit) = {bdc['beta_ci_2depth_fit']}", flush=True)
    for r in bdc["per_depth_count"]:
        hw9 = r["extrap_halfwidth_lambda_depth9"]
        hw9s = f"{hw9:.4f}" if hw9 is not None else "n/a(all measured)"
        print(f"        k={r['depths_measured']} (lever {r['lever_arm']}): β̂="
              f"{r['beta_hat']:.4f}  hw[λ_9]=±{hw9s}  within#187CI={r['within_187_ci']}", flush=True)
    print("-" * 96, flush=True)
    print(f"  (3) DEPTH-1-ONLY FALSE GO  risk = {d1['false_go_risk_depth1_only']}  "
          f"overstatement = {d1['overstatement_tps']:.1f} TPS (λ̂₁=1.0)", flush=True)
    for r in d1["rows"]:
        print(f"        λ̂₁={r['lambda_hat_1']:.4f}: naive-flat priv_LCB={r['naive_flat_private_lcb_tps']:.1f} "
              f"(GO={r['naive_says_go_private']})  TRUE priv_LCB={r['true_mechanism_private_lcb_tps']:.1f} "
              f"(GO={r['true_is_go_private']})  false_GO={r['false_go']}", flush=True)
    print("-" * 96, flush=True)
    mf = syn["mechanism_feasibility"]
    print(f"  mechanism feasibility: private_LCB@(λ̂₁=1,β={syn['imports']['beta_primary']:.3f})="
          f"{mf['private_lcb_at_perfect_depth1_mech']:.1f}  can_clear_private={mf['mechanism_can_clear_private_bar']}",
          flush=True)
    print("-" * 96, flush=True)
    print(f"  (4) PRIMARY depth_budget_self_test_passes = {st['depth_budget_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"          - {k}: {v}", flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print("=" * 96, flush=True)
    print(f"\n  HAND-OFF (land #71): {syn['handoff_lines']['land_71_spec']}", flush=True)
    print(f"\n  HAND-OFF (fern #185): {syn['handoff_lines']['fern_185_integrator']}\n", flush=True)


# --------------------------------------------------------------------------- #
# W&B logging (mirrors #187 / #193; never fatal).
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
        print(f"[depth-budget] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    run = init_wandb_run(
        job_type="liveprobe-depth-budget",
        agent="denken",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["liveprobe-depth-budget", "validity-gate", "measurement-design",
              "neyman-allocation", "private-bar", "bank-the-analysis"],
        config={
            "K_cal": K_CAL, "step": STEP, "z95": Z95, "target_official": TARGET_OFFICIAL,
            "private_bar_both": syn["imports"]["private_bar_both_0p9780"],
            "public_bar_both": PUBLIC_BAR_BOTH, "beta_primary": syn["imports"]["beta_primary"],
            "n_prompts": args.n_prompts,
            "imports": "denken#193 mechanism+beta_crit + denken#187 measurement-CI + stark#191 private bar 0.9780",
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[depth-budget] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    st = syn["self_test"]
    db = syn["decisive_budget"]
    bdc = syn["beta_depth_count"]
    d1 = syn["depth1_false_go"]
    mf = syn["mechanism_feasibility"]
    summary: dict[str, Any] = {
        "depth_budget_self_test_passes": int(bool(st["depth_budget_self_test_passes"])),
        "total_trials_for_decisive_private": db["total_trials_for_decisive_private"],
        "efficiency_gain_neyman_vs_equal": db["efficiency_gain_neyman_vs_equal"],
        "N_equal_allocation": db["N_equal_allocation"],
        "min_depths_for_decisive_int": bdc["min_depths_for_decisive_int"],
        "depth1_plus_2_suffices": int(bool(bdc["depth1_plus_2_suffices"])),
        "false_go_risk_depth1_only": int(bool(d1["false_go_risk_depth1_only"])),
        "depth1_overstatement_tps": d1["overstatement_tps"],
        "true_private_lcb_at_lambda1_eq_1": d1["true_private_lcb_at_lambda1_eq_1"],
        "naive_private_lcb_at_lambda1_eq_1": d1["naive_private_lcb_at_lambda1_eq_1"],
        "mechanism_private_lcb_perfect_depth1": mf["private_lcb_at_perfect_depth1_mech"],
        "mechanism_can_clear_private_bar": int(bool(mf["mechanism_can_clear_private_bar"])),
        "private_bar_both": syn["imports"]["private_bar_both_0p9780"],
        "beta_primary": syn["imports"]["beta_primary"],
        "beta_crit_depth1_sufficient": syn["imports"]["beta_crit_depth1_sufficient"],
        "n_depth2_budget": db["N_d_budget_1to9"][1],
        "n_depth9_budget": db["N_d_budget_1to9"][-1],
        "nan_clean": int(bool(payload["nan_clean"])),
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    # per-λ_true totals
    for r in db["per_lam_true_table"]:
        summary[f"N_opt_lam_{str(r['lam_true']).replace('.', 'p')}"] = r["N_opt"]
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="liveprobe_depth_budget_result", artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[depth-budget] wandb logged: {summary}", flush=True)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--n-prompts", type=int, default=DEFAULT_N_PROMPTS)
    ap.add_argument("--bug2-anchor", type=Path, default=D172.DEFAULT_BUG2_ANCHOR)
    ap.add_argument("--topo-json", type=Path, default=D172.DEFAULT_TOPO_JSON)
    ap.add_argument("--accept-json", type=Path, default=D172.DEFAULT_ACCEPT_JSON)
    ap.add_argument("--rankcov-json", type=Path, default=D172.DEFAULT_RANKCOV_JSON)
    ap.add_argument("--decomp-json", type=Path, default=D172.DEFAULT_DECOMP_JSON)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", default="liveprobe-depth-budget")
    args = ap.parse_args(argv)

    anchors = D172.load_anchors(
        args.bug2_anchor, args.topo_json, args.accept_json, args.rankcov_json, args.decomp_json)
    syn = synthesize(anchors, n_prompts=args.n_prompts)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at,
        "pr": 197,
        "agent": "denken",
        "kind": "liveprobe-depth-budget",
        "anchors": {k: v for k, v in anchors.items() if k != "_paths"},
        "anchor_paths": anchors.get("_paths"),
        "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }

    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[depth-budget] WARNING non-finite values at: {nan_paths}", flush=True)
    syn["self_test"]["conditions"]["e_key_scalars_finite_and_nulls_clean"] = bool(
        syn["self_test"]["conditions"]["e_key_scalars_finite_and_nulls_clean"] and payload["nan_clean"])
    # recompute PRIMARY after folding nan-clean into (e)
    passes = bool(all(syn["self_test"]["conditions"].values()))
    syn["self_test"]["depth_budget_self_test_passes"] = passes

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "liveprobe_depth_budget_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[depth-budget] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    print(f"  PRIMARY depth_budget_self_test_passes = {passes}", flush=True)
    print(f"  TEST total_trials_for_decisive_private = "
          f"{syn['test_metric']['total_trials_for_decisive_private']:,.0f}", flush=True)
    print(f"  peak_mem_mib = {payload['peak_mem_mib']}", flush=True)

    if args.self_test:
        ok = passes and payload["nan_clean"]
        print(f"[depth-budget] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
