#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Deep-tail build-bar budget (PR #215) — convert the #208-certified scalar both-bugs
private go-bar 0.977978 into a per-DEPTH build target.

THE ONE NUMBER THIS LEG PRODUCES
--------------------------------
stark #208 (`wi4gxxx8`) certified the THRESHOLD: 0.977978 is the worst-case both-bugs
private go-bar over ALL realizable domain blends. The build CLEARS iff the depth-aggregate
reach-weighted acceptance λ̂ = Σ_d ŵ_d·λ_d ≥ 0.977978. land #71 has now POSTED an interim
shallow-mid spine `lambda_spine_min_q2_q7 = 0.997` (q[2..7], ≫ the bar) but the DEEP tail
q[8..9] — where acceptance falls off (denken #193's λ(depth) staleness curve) — is still
UNMEASURED. So the single launch-decision-relevant number is the bar's SLACK decomposed by
depth: holding the measured shallow spine, what is the MINIMUM reach-weighted deep-tail
acceptance over q[8..9] at which the aggregate drops exactly to the certified 0.977978 bar?
Below it the build MISSES even with a perfect shallow spine; at/above it the build CLEARS.

MODEL (imports — NOT re-derived)
--------------------------------
    w_d_at_bar  = #208/#203 reach-weights (depths 1..7, the draft horizon)  [IMPORTED]
    β           = #193 salvage-staleness depth decay 0.765124               [IMPORTED]
    bar         = #208 worst-case-blend both-bugs private bar 0.977978      [IMPORTED]
  The horizon-7 forward map does not reach depths 8..9, so the deep-tail reach mass is
  EXTENDED with #193's β: w_8 = w_7·β, w_9 = w_8·β. This is mechanism-consistent — #203's
  reach-weights already decay at geomean ratio ≈0.775/rung ≈ β — and is exactly the #193
  λ(depth) curve the PR invokes for the deep-tail projection.

    ŵ_d         = w_d / Σ_{d∈q[2..9]} w_d           ← NORMALIZED reach-weighted MEAN (Σŵ=1)
    λ̂           = Σ_{d=2}^{9} ŵ_d·λ_d
                = W_shallow·λ_spine + W_deep·λ_deeptail        (q[2..7] held | q[8..9] unknown)
  Depth-1 is the liveprobe head/anchor (λ̂_1, #193), measured separately; it is EXCLUDED
  from the q[2..9] ladder aggregate (the PR's two bands q[2..7] ∪ q[8..9] = q[2..9]). The
  depth-1-INCLUSIVE aggregate is reported as a sensitivity arm.

CLOSED FORM
-----------
    min_deeptail_lambda_q8q9_clears_bar = (bar − W_shallow·λ_spine) / W_deep
                                        = λ_spine + (bar − λ_spine) / W_deep
    ∂λ̂/∂λ_deeptail = W_deep ;   ∂budget/∂λ_spine = −W_shallow / W_deep   (buys down the tail)
    spine_value_where_budget_hits_zero = bar / W_shallow

LOCAL CPU-only analytic synthesis. No GPU / vLLM / HF Job / submission / served-file change /
official draw. Does NOT re-measure λ (the deep tail is land #71's measurement to make; this
leg gives only the THRESHOLD it must clear). BASELINE stays 481.53. Greedy/PPL untouched.
Bank-the-analysis (PRIMARY = self-test, adds 0 TPS). NOT open2. NOT a launch.

PRIMARY metric  deeptail_bar_budget_self_test_passes
TEST    metric  min_deeptail_lambda_q8q9_clears_bar

Run:
    python research/validity/deeptail_bar_budget/deeptail_bar_budget.py \
        --wandb_group deeptail-bar-budget --wandb_name stark/deeptail-bar-budget
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

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

# ---- banked artifacts (import-not-rederive) ---- #
D208_PATH = REPO_ROOT / "research/validity/multivertex_realizability/results.json"
D193_PATH = REPO_ROOT / "research/oracle_readout/lambda_depth_profile/lambda_depth_profile_results.json"

# land #71 POSTED interim shallow-mid spine (q[2..7]); a PARAMETER, swept as a band (step 3).
LAMBDA_SPINE_INTERIM = 0.997                 # marker 2026-06-14T17:31:23Z, NON-TERMINAL
SPINE_SWEEP = [0.990, 0.995, 0.997, 0.999, 1.000]
ROUNDTRIP_TOL = 1e-9


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Imports (read the banked JSON; do not re-derive).
# --------------------------------------------------------------------------- #
def load_imports() -> dict[str, Any]:
    d208 = _load(D208_PATH)["synthesis"]
    d193 = _load(D193_PATH)["synthesis"]

    w7 = list(d208["constants"]["w_d_at_bar"])                         # depths 1..7
    if len(w7) != 7:
        raise ValueError(f"expected 7 reach-weights, got {len(w7)}")
    bar = float(d208["constants"]["d198_coupled_bar"])                # 0.9779783323491393
    bar_band = list(d208["beta_robustness"]["worstcase_bar_band"])    # [lo, hi]
    beta = float(d208["constants"]["beta_primary"])                   # 0.765124365433998
    beta_range = list(d208["constants"]["beta_range"])                # [0.61649, 0.94960]
    lambda_floor = float(d208["constants"]["lambda_floor_liveprobe"]) # 0.3418647166361965
    lam191 = float(d208["constants"]["lambda_star_191_fixed_drop"])   # 0.9780112973731208

    lod = d193["mechanism_law"]["lambda_of_depth"]
    pml = list(lod["pure_mechanism_lambda_unclamped"])                # depths 1..9
    dp_clamped = list(lod["effective_lambda_DP_clamped"])             # depths 1..9
    lambda_hat_1 = float(d193["mechanism_law"]["lambda_hat_1"])       # 0.3418647166361965
    if len(pml) != 9:
        raise ValueError(f"expected 9-depth λ(depth), got {len(pml)}")

    return {
        "w7": w7, "bar": bar, "bar_band": bar_band, "beta": beta, "beta_range": beta_range,
        "lambda_floor": lambda_floor, "lambda_star_191": lam191,
        "pure_mechanism_lambda": pml, "dp_clamped_lambda": dp_clamped,
        "lambda_hat_1": lambda_hat_1,
        "provenance": (
            "stark#208 w_d_at_bar + certified bar (wi4gxxx8) × stark#203 reach-weights/forward "
            "map (hexhagf6) × stark#191 fixed-drop bar (jeclr39w) × denken#193 λ_d=λ̂₁·β^(d−1) "
            "(2clxvlr8). land #71 interim spine λ(q[2..7])=0.997 (PARAMETER, swept)."),
    }


# --------------------------------------------------------------------------- #
# Reach-weight profile over the q[2..9] ladder (depths 2..9), β-extended to 8,9.
# --------------------------------------------------------------------------- #
def build_weights(w7: list[float], beta: float) -> dict[str, Any]:
    w8 = w7[-1] * beta                       # depth-8 reach mass via #193 β-staleness
    w9 = w8 * beta                           # depth-9
    w_full = list(w7) + [w8, w9]             # depths 1..9 (raw reach mass)

    # PRIMARY aggregate axis = q[2..9] (depth-1 head excluded). bands partition it exactly.
    w_q2q9 = w_full[1:9]                      # depths 2..9 (8 weights)
    norm = sum(w_q2q9)
    wh = [x / norm for x in w_q2q9]           # normalized (Σ=1)
    # band masses (indices into wh): spine q[2..7] = wh[0:6]; deep tail q[8..9] = wh[6:8].
    w_mass_shallow_q2q7 = sum(wh[0:6])
    w_mass_deeptail_q8q9 = sum(wh[6:8])

    # depth-1-INCLUSIVE alt axis = q[1..9] (sensitivity arm).
    norm_full = sum(w_full)
    wh_full = [x / norm_full for x in w_full]
    w_mass_head_q1_full = wh_full[0]
    w_mass_shallow_q2q7_full = sum(wh_full[1:7])
    w_mass_deeptail_q8q9_full = sum(wh_full[7:9])

    return {
        "w_full_raw": w_full, "w8": w8, "w9": w9,
        "w_q2q9_raw": w_q2q9, "norm_q2q9": norm, "wh_q2q9": wh,
        "w_mass_shallow_q2q7": w_mass_shallow_q2q7,
        "w_mass_deeptail_q8q9": w_mass_deeptail_q8q9,
        "wh_full_q1q9": wh_full,
        "w_mass_head_q1_full": w_mass_head_q1_full,
        "w_mass_shallow_q2q7_full": w_mass_shallow_q2q7_full,
        "w_mass_deeptail_q8q9_full": w_mass_deeptail_q8q9_full,
        # deep-tail raw weights (for mechanism reach-weighting): depths 8,9.
        "deeptail_raw_w": [w_full[7], w_full[8]],
    }


def budget_for(bar: float, w_shallow: float, w_deep: float, lam_spine: float) -> float:
    """min deep-tail reach-weighted acceptance s.t. aggregate == bar (closed form)."""
    return (bar - w_shallow * lam_spine) / w_deep


def reach_weighted_mean(vals: list[float], raw_w: list[float]) -> float:
    s = sum(raw_w)
    return sum(v * w for v, w in zip(vals, raw_w)) / s


def _beta_crit_for_budget(lam_spine: float, raw_w: list[float], budget: float) -> float | None:
    """β at which the coherent (β-from-spine) deep-tail projection exactly meets the budget.
    proj(β) = λ_spine·(w8·β + w9·β²)/(w8+w9) = budget  ⇒  w9·β² + w8·β − budget·(w8+w9)/λ_spine = 0."""
    w8, w9 = raw_w
    W = w8 + w9
    target = budget * W / lam_spine
    disc = w8 * w8 + 4.0 * w9 * target
    if disc < 0.0 or w9 == 0.0:
        return None
    return (-w8 + math.sqrt(disc)) / (2.0 * w9)


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize(lam_spine: float = LAMBDA_SPINE_INTERIM) -> dict[str, Any]:
    imp = load_imports()
    bar = imp["bar"]
    beta = imp["beta"]
    W = build_weights(imp["w7"], beta)
    w_sh = W["w_mass_shallow_q2q7"]
    w_dt = W["w_mass_deeptail_q8q9"]

    # ---- (1) depth-decomposed aggregate ---- #
    def agg(ls: float, ldt: float) -> float:
        return w_sh * ls + w_dt * ldt

    lambda_hat_shallow_only = agg(lam_spine, 0.0)          # deep tail → 0 (absolute floor)
    lambda_hat_uniform_spine = agg(lam_spine, lam_spine)   # uniform build (== lam_spine)

    # ---- (2) the deep-tail budget (the deliverable) ---- #
    budget = budget_for(bar, w_sh, w_dt, lam_spine)        # min deep-tail to clear
    d_lambdahat_d_deeptail = w_dt                          # ∂λ̂/∂λ_deeptail
    # cross-check vs the #193 mechanism (literal absolute import for d∈{8,9}):
    pml = imp["pure_mechanism_lambda"]
    deeptail_raw_w = W["deeptail_raw_w"]
    deeptail_lambda_mechanism_proj = reach_weighted_mean([pml[7], pml[8]], deeptail_raw_w)
    mechanism_clears_bar = bool(deeptail_lambda_mechanism_proj >= budget)
    mechanism_margin_lambda = deeptail_lambda_mechanism_proj - budget
    # honest secondary: the #193 ABSOLUTE proj is anchored at the pessimistic floor λ̂₁=0.342,
    # which is REGIME-INCOHERENT with land #71's measured spine 0.997. The coherent reading
    # continues #193's per-step β decay FROM the measured spine (d7 anchor → d8,d9).
    proj_rel_vals = [lam_spine * beta, lam_spine * beta * beta]
    deeptail_lambda_mechanism_proj_rel = reach_weighted_mean(proj_rel_vals, deeptail_raw_w)
    mechanism_clears_bar_rel = bool(deeptail_lambda_mechanism_proj_rel >= budget)
    mechanism_margin_lambda_rel = deeptail_lambda_mechanism_proj_rel - budget
    # the coherent projection's GO/NO-GO verdict FLIPS on the deep-tail decay rate β (the very
    # thing land #71 must measure) — sweep #193's β construction range to show the crossover.
    beta_lo, beta_hi = imp["beta_range"]
    mechanism_proj_beta_band = []
    for b in (beta_lo, beta, beta_hi):
        proj_b = reach_weighted_mean([lam_spine * b, lam_spine * b * b], deeptail_raw_w)
        mechanism_proj_beta_band.append({
            "beta": b,
            "deeptail_proj_relative_from_spine": proj_b,
            "clears_bar": bool(proj_b >= budget),
            "margin_lambda": proj_b - budget,
        })
    # β at which the coherent projection exactly meets the budget (GO/NO-GO crossover).
    beta_crit = _beta_crit_for_budget(lam_spine, deeptail_raw_w, budget)

    # ---- (3) robustness to the interim spine value ---- #
    budget_vs_spine = []
    for ls in SPINE_SWEEP:
        b = budget_for(bar, w_sh, w_dt, ls)
        budget_vs_spine.append({
            "lambda_spine": ls,
            "min_deeptail_lambda_q8q9_clears_bar": b,
            "budget_le_zero_deeptail_free": bool(b <= 0.0),    # deep tail can fully collapse
            "budget_gt_one_infeasible": bool(b > 1.0),         # even perfect tail can't clear
            "lambda_hat_shallow_only": agg(ls, 0.0),
        })
    # slope of budget vs spine (≈ linear): d(budget)/d(spine) = −W_shallow/W_deep.
    budget_vs_spine_slope = -w_sh / w_dt
    weight_ratio_shallow_over_deep = w_sh / w_dt
    # spine at which the deep tail becomes irrelevant (budget == 0): bar / W_shallow.
    spine_value_where_deeptail_budget_hits_zero = bar / w_sh
    spine_zero_exists_le_1 = bool(spine_value_where_deeptail_budget_hits_zero <= 1.0)

    # ---- depth-1-INCLUSIVE sensitivity arm (q[1..9] aggregate) ---- #
    w_sh_full = W["w_mass_head_q1_full"] + W["w_mass_shallow_q2q7_full"]  # head+spine held
    w_dt_full = W["w_mass_deeptail_q8q9_full"]
    budget_full = budget_for(bar, w_sh_full, w_dt_full, lam_spine)
    depth1_inclusive = {
        "axis": "q[1..9] (depth-1 head INCLUDED in the held/shallow band at λ_spine)",
        "w_mass_head_q1": W["w_mass_head_q1_full"],
        "w_mass_shallow_q2q7": W["w_mass_shallow_q2q7_full"],
        "w_mass_deeptail_q8q9": w_dt_full,
        "w_mass_held_q1q7": w_sh_full,
        "min_deeptail_lambda_q8q9_clears_bar": budget_full,
        "lambda_hat_shallow_only": w_sh_full * lam_spine,
        "spine_value_where_budget_hits_zero": bar / w_sh_full,
        "note": ("including the depth-1 head (largest reach mass) DILUTES the deep-tail mass "
                 "→ LOWER required deep tail; primary axis excludes it (it is the separately "
                 "anchored liveprobe head λ̂₁, and the PR's bands partition q[2..9])."),
    }

    # ---- weight-extension sensitivity (the second-biggest modeling lever) ---- #
    w7 = imp["w7"]
    ext_variants = {}
    for tag, (e8, e9) in {
        "beta_primary": (w7[-1] * beta, w7[-1] * beta * beta),
        "flat_w7": (w7[-1], w7[-1]),
        "beta_squared": (w7[-1] * beta * beta, w7[-1] * beta ** 4),
        "beta_lo_0p6165": (w7[-1] * 0.616486595380561, w7[-1] * 0.616486595380561 ** 2),
        "beta_hi_0p9496": (w7[-1] * 0.9495993894553337, w7[-1] * 0.9495993894553337 ** 2),
    }.items():
        wf = list(w7) + [e8, e9]
        wq = wf[1:9]
        nz = sum(wq)
        wdt = (e8 + e9) / nz
        wsh = 1.0 - wdt
        ext_variants[tag] = {
            "w_mass_deeptail_q8q9": wdt,
            "min_deeptail_lambda_q8q9_clears_bar": budget_for(bar, wsh, wdt, lam_spine),
        }

    # ---- certified-bar β-band sensitivity (the bar itself moves [0.977978, 0.978015]) ---- #
    bar_band = imp["bar_band"]
    budget_at_bar_band = {
        "bar_lo": bar_band[0], "bar_hi": bar_band[1],
        "budget_at_bar_lo": budget_for(bar_band[0], w_sh, w_dt, lam_spine),
        "budget_at_bar_hi": budget_for(bar_band[1], w_sh, w_dt, lam_spine),
    }

    # ---- (4) self-test (PRIMARY) ---- #
    st = _selftests(agg, bar, lam_spine, w_sh, w_dt, lambda_hat_shallow_only,
                    budget, budget_vs_spine)

    # ---- (5) hand-off line ---- #
    handoff = (
        "DEEP-TAIL BUILD-BAR BUDGET (stark #215): given land #71's measured shallow-mid spine "
        "λ(q[2..7])=%.3f, the depth-aggregate clears the #208-certified %.6f bar iff the deep "
        "tail q[8..9] holds reach-weighted acceptance ≥ %.4f (deep tail carries only %.1f%% of "
        "the q[2..9] reach mass, so each unit of deep-tail acceptance buys %.4f in λ̂; the "
        "shallow spine alone leaves λ̂=%.4f, BELOW the bar, so the deep tail is NOT a free pass). "
        "#193's mechanism, read coherently as per-step β-decay from the measured spine, projects "
        "deep tail ≈%.3f ⇒ %s by %.3f in λ — a CLOSE call; the literal #193 absolute floor proj "
        "(%.3f, regime-incoherent with the 0.997 spine) would say miss by %.3f. fern #185 carries "
        "%.4f as the deep-tail GO threshold so the instant land #71 posts the measured q[8..9] the "
        "GO/NO-GO reads off directly; a perfect spine (1.0) still needs deep tail ≥ %.4f (budget "
        "hits zero only at spine %.4f > 1, so the deep tail is never irrelevant)."
        % (lam_spine, bar, budget, w_dt * 100.0, d_lambdahat_d_deeptail,
           lambda_hat_shallow_only,
           deeptail_lambda_mechanism_proj_rel,
           "CLEARS" if mechanism_clears_bar_rel else "MISSES",
           abs(mechanism_margin_lambda_rel),
           deeptail_lambda_mechanism_proj, abs(mechanism_margin_lambda),
           budget, budget_for(bar, w_sh, w_dt, 1.0),
           spine_value_where_deeptail_budget_hits_zero))

    verdict = (
        "DEEP-TAIL BUDGET = %.4f. Holding land #71's measured shallow-mid spine λ(q[2..7])=%.3f, "
        "the build clears the #208-certified %.6f both-bugs private bar iff the reach-weighted "
        "deep-tail acceptance over q[8..9] is ≥ %.4f. The deep tail carries only %.1f%% of the "
        "q[2..9] reach mass (reach decays ≈β=%.3f/rung), so the budget is FAR below the spine "
        "(0.997): there is large slack. But the deep tail is NOT free — with it collapsed to 0 "
        "the aggregate is %.4f, BELOW the bar. The launch hinges on the unmeasured q[8..9]: #193's "
        "coherent (β-from-spine) projection ≈%.3f sits JUST %s the budget (margin %.3f in λ), so "
        "land #71's actual q[8..9] measurement is decisive."
        % (budget, lam_spine, bar, budget, w_dt * 100.0, beta, lambda_hat_shallow_only,
           deeptail_lambda_mechanism_proj_rel,
           "above" if mechanism_clears_bar_rel else "below",
           abs(mechanism_margin_lambda_rel)))

    return {
        "lambda_spine_interim": lam_spine,
        "constants": {
            "bar_certified_208": bar,
            "bar_band_208": bar_band,
            "beta_primary_193": beta,
            "lambda_floor_liveprobe": imp["lambda_floor"],
            "lambda_hat_1_193": imp["lambda_hat_1"],
            "lambda_star_191_fixed_drop": imp["lambda_star_191"],
            "w_d_at_bar_imported_d1_d7": imp["w7"],
            "pure_mechanism_lambda_d1_d9": pml,
            "dp_clamped_lambda_d1_d9": imp["dp_clamped_lambda"],
            "lambda_spine_sweep": SPINE_SWEEP,
        },
        "imports": {"provenance": imp["provenance"],
                    "d208_path": str(D208_PATH.relative_to(REPO_ROOT)),
                    "d193_path": str(D193_PATH.relative_to(REPO_ROOT))},
        "reach_weight_profile": {
            "axis": "q[2..9] (depth-1 head excluded; bands q[2..7] ∪ q[8..9] partition it)",
            "w_full_raw_d1_d9": W["w_full_raw"],
            "w8_beta_extended": W["w8"], "w9_beta_extended": W["w9"],
            "deeptail_extension_rule": "w_8 = w_7·β, w_9 = w_8·β (β=#193 salvage-staleness decay)",
            "wh_q2q9_normalized": W["wh_q2q9"],
            "w_mass_shallow_q2q7": w_sh,
            "w_mass_deeptail_q8q9": w_dt,
            "w_mass_shallow_over_deeptail_ratio": weight_ratio_shallow_over_deep,
            "weights_sum_to_one": abs(sum(W["wh_q2q9"]) - 1.0) < 1e-12,
        },
        "depth_decomposed_aggregate": {
            "aggregate_form": "λ̂ = W_shallow·λ_spine + W_deep·λ_deeptail  (normalized, Σŵ=1)",
            "lambda_hat_shallow_only": lambda_hat_shallow_only,
            "lambda_hat_shallow_only_clears_bar": bool(lambda_hat_shallow_only >= bar),
            "lambda_hat_uniform_spine": lambda_hat_uniform_spine,
            "deeptail_carries_build_risk": bool(lambda_hat_shallow_only < bar),
        },
        "deeptail_budget": {
            "min_deeptail_lambda_q8q9_clears_bar": budget,         # TEST
            "d_lambdahat_d_deeptail": d_lambdahat_d_deeptail,
            "deeptail_lambda_mechanism_proj": deeptail_lambda_mechanism_proj,  # #193 absolute
            "mechanism_clears_bar": mechanism_clears_bar,
            "mechanism_margin_lambda": mechanism_margin_lambda,
            "deeptail_lambda_mechanism_proj_relative_from_spine": deeptail_lambda_mechanism_proj_rel,
            "mechanism_clears_bar_relative": mechanism_clears_bar_rel,
            "mechanism_margin_lambda_relative": mechanism_margin_lambda_rel,
            "mechanism_proj_beta_band": mechanism_proj_beta_band,
            "mechanism_proj_beta_crit_meets_budget": beta_crit,
            "mechanism_proj_beta_crit_note": (
                "the coherent projection's GO/NO-GO FLIPS at β≈%.3f: for deep-tail per-step "
                "retention β ≥ %.3f the build CLEARS, below it MISSES. #193's primary β=%.3f "
                "(range [%.3f, %.3f]) straddles this — so the verdict genuinely hinges on land "
                "#71's measured deep-tail decay, not on this analysis."
                % (beta_crit if beta_crit is not None else float("nan"),
                   beta_crit if beta_crit is not None else float("nan"),
                   beta, beta_lo, beta_hi)),
            "mechanism_proj_note": (
                "the #193 ABSOLUTE proj (%.4f) is anchored at the pessimistic floor λ̂₁=%.3f and "
                "is REGIME-INCOHERENT with land #71's measured spine 0.997 (a build that accepts "
                "0.997 shallow is NOT in the floor regime); the COHERENT reading continues #193's "
                "β per-step decay from the measured spine → %.4f. Both miss the budget, but the "
                "coherent reading misses by only %.3f vs %.3f — the call is genuinely close."
                % (deeptail_lambda_mechanism_proj, imp["lambda_hat_1"],
                   deeptail_lambda_mechanism_proj_rel,
                   abs(mechanism_margin_lambda_rel), abs(mechanism_margin_lambda))),
        },
        "robustness_vs_spine": {
            "budget_vs_spine": budget_vs_spine,
            "budget_vs_spine_slope_d_per_d": budget_vs_spine_slope,
            "weight_ratio_shallow_over_deeptail": weight_ratio_shallow_over_deep,
            "spine_value_where_deeptail_budget_hits_zero": spine_value_where_deeptail_budget_hits_zero,
            "spine_zero_exists_le_1": spine_zero_exists_le_1,
            "interpretation": (
                "a higher measured spine BUYS DOWN the required deep tail roughly linearly: "
                "slope = −W_shallow/W_deep = −%.2f (each +0.001 spine lowers the deep-tail budget "
                "by ≈%.4f). budget hits zero (deep tail irrelevant) only at spine=%.4f; since "
                "that %s ≤ 1, even a PERFECT spine still requires deep tail ≥ %.4f."
                % (weight_ratio_shallow_over_deep, weight_ratio_shallow_over_deep * 0.001,
                   spine_value_where_deeptail_budget_hits_zero,
                   "exceeds" if not spine_zero_exists_le_1 else "is",
                   budget_for(bar, w_sh, w_dt, 1.0))),
        },
        "sensitivity": {
            "depth1_inclusive_arm": depth1_inclusive,
            "weight_extension_arm": ext_variants,
            "bar_band_arm": budget_at_bar_band,
        },
        "self_test": st,
        "handoff_line": handoff,
        "verdict": verdict,
    }


def _selftests(agg, bar, lam_spine, w_sh, w_dt, lambda_hat_shallow_only,
               budget, budget_vs_spine) -> dict[str, Any]:
    # (a) uniform λ=0.997 across all q[2..9] depths → aggregate ≥ bar (and == λ_spine exactly).
    agg_uniform = agg(lam_spine, lam_spine)
    cond_a = bool(_finite(agg_uniform) and abs(agg_uniform - lam_spine) < 1e-12
                  and agg_uniform >= bar)

    # (b) deep tail = 0 → aggregate == lambda_hat_shallow_only EXACTLY.
    agg_dt0 = agg(lam_spine, 0.0)
    cond_b = bool(_finite(agg_dt0) and abs(agg_dt0 - lambda_hat_shallow_only) < 1e-12)

    # (c) plug the budget back → aggregate == bar to tol (round-trip the #208 threshold).
    agg_at_budget = agg(lam_spine, budget)
    cond_c = bool(_finite(agg_at_budget) and abs(agg_at_budget - bar) < ROUNDTRIP_TOL)

    # (d) aggregate monotone INCREASING in λ_deeptail AND budget monotone DECREASING in spine.
    mono_up = agg(lam_spine, 0.6) < agg(lam_spine, 0.6 + 1e-6)
    seq = [r["min_deeptail_lambda_q8q9_clears_bar"] for r in budget_vs_spine]
    budget_decreasing_in_spine = all(seq[i + 1] < seq[i] - 1e-12 for i in range(len(seq) - 1))
    cond_d = bool(mono_up and budget_decreasing_in_spine)

    return {
        "conditions": {
            "a_uniform_spine_aggregate_eq_spine_and_clears": cond_a,
            "b_deeptail_zero_eq_shallow_only_exact": cond_b,
            "c_budget_roundtrips_208_bar_exact": cond_c,
            "d_aggregate_up_in_deeptail_and_budget_down_in_spine": cond_d,
            # e (NaN-clean) filled in main().
        },
        "a_detail": {"agg_uniform": agg_uniform, "lambda_spine": lam_spine, "bar": bar},
        "b_detail": {"agg_deeptail_zero": agg_dt0,
                     "lambda_hat_shallow_only": lambda_hat_shallow_only},
        "c_detail": {"agg_at_budget": agg_at_budget, "bar": bar,
                     "roundtrip_resid": abs(agg_at_budget - bar)},
        "d_detail": {"aggregate_increasing_in_deeptail": bool(mono_up),
                     "budget_decreasing_in_spine": bool(budget_decreasing_in_spine),
                     "budget_sweep_seq": seq},
        "partial_passes_a_to_d": bool(cond_a and cond_b and cond_c and cond_d),
    }


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
    rw = syn["reach_weight_profile"]
    da = syn["depth_decomposed_aggregate"]
    db = syn["deeptail_budget"]
    rb = syn["robustness_vs_spine"]
    st = syn["self_test"]
    print("\n" + "=" * 96, flush=True)
    print("DEEP-TAIL BUILD-BAR BUDGET (PR #215) — min q[8..9] acceptance to clear the certified bar",
          flush=True)
    print("=" * 96, flush=True)
    print(f"  certified bar (#208) = {c['bar_certified_208']:.6f}  band {c['bar_band_208']}   "
          f"β(#193) = {c['beta_primary_193']:.6f}   λ_spine(land #71) = {syn['lambda_spine_interim']:.3f}",
          flush=True)
    print("-" * 96, flush=True)
    print("  (1) reach-weight decomposition over q[2..9] (depth-1 head excluded):", flush=True)
    print(f"      w_full(d1..d9) = {[round(x,4) for x in rw['w_full_raw_d1_d9']]}", flush=True)
    print(f"      deep-tail extension: {rw['deeptail_extension_rule']}", flush=True)
    print(f"      w_mass_shallow_q2q7 = {rw['w_mass_shallow_q2q7']:.4f}   "
          f"w_mass_deeptail_q8q9 = {rw['w_mass_deeptail_q8q9']:.4f}   "
          f"(ratio {rw['w_mass_shallow_over_deeptail_ratio']:.2f}×)", flush=True)
    print(f"      lambda_hat_shallow_only (deep tail→0) = {da['lambda_hat_shallow_only']:.6f}  "
          f"clears_bar = {da['lambda_hat_shallow_only_clears_bar']}  "
          f"(deep tail carries build risk = {da['deeptail_carries_build_risk']})", flush=True)
    print("-" * 96, flush=True)
    print("  (2) THE DEEP-TAIL BUDGET:", flush=True)
    print(f"      min_deeptail_lambda_q8q9_clears_bar = {db['min_deeptail_lambda_q8q9_clears_bar']:.6f}  (TEST)",
          flush=True)
    print(f"      ∂λ̂/∂λ_deeptail = {db['d_lambdahat_d_deeptail']:.6f}", flush=True)
    print(f"      #193 mechanism proj (absolute) = {db['deeptail_lambda_mechanism_proj']:.4f}  "
          f"clears = {db['mechanism_clears_bar']}  margin = {db['mechanism_margin_lambda']:+.4f}", flush=True)
    print(f"      #193 mechanism proj (β-from-spine, coherent) = "
          f"{db['deeptail_lambda_mechanism_proj_relative_from_spine']:.4f}  "
          f"clears = {db['mechanism_clears_bar_relative']}  "
          f"margin = {db['mechanism_margin_lambda_relative']:+.4f}", flush=True)
    bc = db["mechanism_proj_beta_crit_meets_budget"]
    print(f"      coherent-proj GO/NO-GO crossover at β_crit = "
          f"{bc:.4f}" if bc is not None else "      β_crit = None", flush=True)
    for r in db["mechanism_proj_beta_band"]:
        print(f"        β={r['beta']:.4f} → deep-tail proj {r['deeptail_proj_relative_from_spine']:.4f} "
              f"clears={r['clears_bar']}", flush=True)
    print("-" * 96, flush=True)
    print("  (3) robustness vs interim spine:", flush=True)
    for r in rb["budget_vs_spine"]:
        print(f"      λ_spine={r['lambda_spine']:.3f} → budget={r['min_deeptail_lambda_q8q9_clears_bar']:.6f}",
              flush=True)
    print(f"      slope d(budget)/d(spine) = {rb['budget_vs_spine_slope_d_per_d']:.3f}   "
          f"spine where budget=0: {rb['spine_value_where_deeptail_budget_hits_zero']:.4f} "
          f"(exists≤1 = {rb['spine_zero_exists_le_1']})", flush=True)
    print("-" * 96, flush=True)
    print("  (4) SELF-TEST (PRIMARY):", flush=True)
    for k, v in st["conditions"].items():
        print(f"        - {k}: {v}", flush=True)
    print("-" * 96, flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print("=" * 96, flush=True)
    print(f"\n  HAND-OFF (land #71 + fern #185): {syn['handoff_line']}\n", flush=True)


# --------------------------------------------------------------------------- #
# W&B logging (mirrors #203/#208; never fatal).
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
        print(f"[deeptail-bar-budget] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    c = syn["constants"]
    rw = syn["reach_weight_profile"]
    da = syn["depth_decomposed_aggregate"]
    db = syn["deeptail_budget"]
    rb = syn["robustness_vs_spine"]
    run = init_wandb_run(
        job_type="deeptail-bar-budget",
        agent="stark",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["deeptail-bar-budget", "validity-gate", "private-drop", "reach-weights",
              "depth-aggregate", "build-bar", "composition", "bank-the-analysis"],
        config={
            "bar_certified_208": c["bar_certified_208"],
            "beta_primary_193": c["beta_primary_193"],
            "lambda_spine_interim": syn["lambda_spine_interim"],
            "lambda_floor_liveprobe": c["lambda_floor_liveprobe"],
            "lambda_star_191_fixed_drop": c["lambda_star_191_fixed_drop"],
            "imports": syn["imports"]["provenance"],
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[deeptail-bar-budget] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "deeptail_bar_budget_self_test_passes": int(bool(payload["self_test_passes"])),
        "min_deeptail_lambda_q8q9_clears_bar": db["min_deeptail_lambda_q8q9_clears_bar"],
        "d_lambdahat_d_deeptail": db["d_lambdahat_d_deeptail"],
        "w_mass_shallow_q2q7": rw["w_mass_shallow_q2q7"],
        "w_mass_deeptail_q8q9": rw["w_mass_deeptail_q8q9"],
        "w_mass_shallow_over_deeptail_ratio": rw["w_mass_shallow_over_deeptail_ratio"],
        "lambda_hat_shallow_only": da["lambda_hat_shallow_only"],
        "lambda_hat_shallow_only_clears_bar": int(bool(da["lambda_hat_shallow_only_clears_bar"])),
        "deeptail_carries_build_risk": int(bool(da["deeptail_carries_build_risk"])),
        "deeptail_lambda_mechanism_proj": db["deeptail_lambda_mechanism_proj"],
        "mechanism_clears_bar": int(bool(db["mechanism_clears_bar"])),
        "mechanism_margin_lambda": db["mechanism_margin_lambda"],
        "deeptail_lambda_mechanism_proj_relative_from_spine":
            db["deeptail_lambda_mechanism_proj_relative_from_spine"],
        "mechanism_clears_bar_relative": int(bool(db["mechanism_clears_bar_relative"])),
        "mechanism_margin_lambda_relative": db["mechanism_margin_lambda_relative"],
        "mechanism_proj_beta_crit_meets_budget": db["mechanism_proj_beta_crit_meets_budget"],
        "budget_vs_spine_slope": rb["budget_vs_spine_slope_d_per_d"],
        "spine_value_where_deeptail_budget_hits_zero":
            rb["spine_value_where_deeptail_budget_hits_zero"],
        "spine_zero_exists_le_1": int(bool(rb["spine_zero_exists_le_1"])),
        "bar_certified_208": c["bar_certified_208"],
        "beta_primary_193": c["beta_primary_193"],
        "lambda_spine_interim": syn["lambda_spine_interim"],
        "depth1_incl_budget": syn["sensitivity"]["depth1_inclusive_arm"][
            "min_deeptail_lambda_q8q9_clears_bar"],
        "nan_clean": int(bool(payload["nan_clean"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"budget_at_spine_{str(r['lambda_spine']).replace('.', 'p')}":
           r["min_deeptail_lambda_q8q9_clears_bar"] for r in rb["budget_vs_spine"]},
        **{f"selftest_{k}": int(bool(v)) for k, v in syn["self_test"]["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="deeptail_bar_budget_result",
                      artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[deeptail-bar-budget] wandb logged: {summary}", flush=True)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--lambda-spine", type=float, default=LAMBDA_SPINE_INTERIM,
                    help="land #71 interim shallow-mid spine λ(q[2..7]) (default 0.997)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="deeptail-bar-budget")
    args = ap.parse_args(argv)

    syn = synthesize(lam_spine=args.lambda_spine)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "created_at": created_at,
        "pr": 215,
        "agent": "stark",
        "kind": "deeptail-bar-budget",
        "synthesis": syn,
    }

    # (e) NaN-clean over the full payload.
    nan_bad = _nan_paths(payload)
    payload["nan_clean"] = not nan_bad
    if nan_bad:
        print(f"[deeptail-bar-budget] WARNING non-finite values at: {nan_bad}", flush=True)
    syn["self_test"]["conditions"]["e_nan_clean"] = bool(payload["nan_clean"])

    cond = syn["self_test"]["conditions"]
    self_test_passes = bool(all(cond.values()))
    payload["self_test_passes"] = self_test_passes
    syn["self_test"]["deeptail_bar_budget_self_test_passes"] = self_test_passes

    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload["peak_mem_mib"] = round(peak_kib / 1024.0, 3)

    _print_report(syn)
    print(f"  PRIMARY deeptail_bar_budget_self_test_passes = {self_test_passes}", flush=True)
    print(f"  TEST    min_deeptail_lambda_q8q9_clears_bar  = "
          f"{syn['deeptail_budget']['min_deeptail_lambda_q8q9_clears_bar']:.6f}", flush=True)
    print(f"  peak_mem_mib = {payload['peak_mem_mib']}", flush=True)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[deeptail-bar-budget] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = self_test_passes and payload["nan_clean"]
        print(f"[deeptail-bar-budget] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
