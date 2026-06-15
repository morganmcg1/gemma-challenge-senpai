#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #289 -- per-position acceptance decay: the BUILT-raise target to E[T]>=4.97.

WHAT THIS ANSWERS
-----------------
fern #281 closed Path-A on all three axes and left exactly ONE re-open: a BUILT
public-E[T] raise to >=4.966 at the deployed step (acceptance-per-candidate, not
width). lawine #282 proved the headroom is real but concentrated in the low tail
and found NO free prompt lever; denken #119 proved the LINEAR drafter's E[T] caps
at 3.8445 even at perfect capacity. The LOAD-BEARING gap nobody had decomposed:
the measured aggregate E[T]=3.844 is a SUM over a per-position acceptance profile
a_1..a_7 (the probability draft position k is accepted GIVEN positions 1..k-1
were). This leg decomposes the deployed E[T] into that profile, locates the
acceptance cliff, and prices the EXACT per-position lift a BUILT drafter must
deliver to reach fern's 4.966 floor -- the quantitative per-position target that
operationalizes the sole remaining >500 path (EAGLE-3).

THE DECOMPOSITION (exact, from banked counters -- no new measurement)
--------------------------------------------------------------------
In vLLM speculative decoding, the verifier accepts a PREFIX of the K=7 draft
tokens: rejection sampling stops at the first reject, so draft position k is
accepted only if positions 0..k-1 were all accepted. The Prometheus counter
`vllm:spec_decode_num_accepted_tokens_per_pos[k]` therefore counts the number of
spec steps in which positions 0..k were ALL accepted == the number of steps with
(# accepted draft tokens j) >= k+1. lawine #282 (`2j0e8xgg`) banked that vector
whole-run:

    accepted_per_pos = [12452, 9458, 7500, 6171, 5152, 4306, 3645]  (k=0..6)
    num_drafts       = 17075   (spec steps)
    num_accepted     = 48684   (= sum accepted_per_pos)

That vector IS the survival function of j (number of accepted draft tokens):

    G(0) = P(j>=0) = 1
    G(m) = P(j>=m) = accepted_per_pos[m-1] / num_drafts ,  m = 1..7

A spec step commits T = j + 1 tokens (j accepted draft + 1 always-emitted bonus),
so the deployed per-step E[T] is the survival sum (the standard "expectation of a
non-negative integer = sum of its survival function"):

    E[T] = 1 + E[j] = 1 + sum_{m=1..7} G(m) = 1 + 48684/17075 = 3.8512

(reproduces kanna #217's 3.844 anchor to within 0.007 = 0.18% measurement noise).
The per-position CONDITIONAL acceptance is the survival ratio

    a_k = P(accept draft pos k | accepted pos 1..k-1) = G(k) / G(k-1) ,  k = 1..7

and the cumulative product Pi_{j<=k} a_j == G(k) round-trips the survival exactly.

CPU analytic over BANKED W&B numbers (lawine #282 per-pos counters + per-prompt
accepted-length deltas, fern #281 4.966 floor, denken #119 3.8445 linear cap).
NO new GPU measurement, NO served-file change, NO HF Job, NO submission. NOT
open2. BASELINE stays 481.53; this leg adds 0 TPS (it DECOMPOSES the banked
E[T]). Analysis-only: per_position_acceptance_decay_analysis_only = True.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

# research/validity/per_position_acceptance_decay/this.py -> repo root is 3 up.
ROOT = Path(__file__).resolve().parents[3]

# --------------------------------------------------------------------------- #
# Imported fleet anchors (DO NOT re-derive -- import EXACTLY, UNCHANGED)
# --------------------------------------------------------------------------- #
OFFICIAL = 481.53                 # #52 official linear TPS (this leg adds 0)
CEILING_LAMBDA1 = 520.95          # lambda=1 ceiling
K_CAL = 125.268                   # kanna #269 anchor: official = K_cal * E[T]
E_T_ANCHOR = 3.844                # kanna #217 deployed linear served E[T]
STEP_US = 1218.2                  # kanna #217 per-forward-pass time (microseconds)
STEP_MS = 1.2182                  # same, milliseconds
TAU = 1.218                       # composition tau
K_SPEC = 7                        # num_speculative_tokens (linear MTP depth)
E_T_MAX = K_SPEC + 1              # 8.0 -- theoretical E[T] ceiling (all 7 accepted + bonus)
HONEST_500_FLOOR = 3.9914         # honest-500 E[T] floor (= 500/K_cal, private-side)
LINEAR_CAP = 3.8445               # denken #119: LINEAR drafter E[T] cap at perfect capacity
PUBLIC_ET_TARGET = 4.966          # fern #281: public E[T] needed @ deployed step (priv 0.804)
PUBLIC_ET_TARGET_ROUND = 4.97     # fern #281 rounded
PRIVATE_FACTOR = 0.804            # ubel #263 canonical private factor (fern #281)
PRIVATE_VERIFIED = 460.85         # private-verified reference (PR baseline)
TOP1_LINEAR_ACCEPT_ANCHOR = 0.728739760479042  # #76 accept_calibration top-1 (cross-check a_1)

# lawine #282 distribution anchors (context only; imported, not re-derived)
LAWINE_ET_MEAN = 3.8512
LAWINE_ET_MEDIAN = 3.9088
LAWINE_ET_P5 = 2.930
LAWINE_ET_STD = 0.8013

# lawine #282 banked whole-run spec counters (W&B-verified; cross-check input)
BANKED_ACCEPTED_PER_POS = [12452.0, 9458.0, 7500.0, 6171.0, 5152.0, 4306.0, 3645.0]
BANKED_NUM_DRAFTS = 17075.0
BANKED_NUM_ACCEPTED = 48684.0

OUT_DIR = ROOT / "research" / "validity" / "per_position_acceptance_decay"
RESULTS_PATH = OUT_DIR / "per_position_acceptance_decay_results.json"
DEFAULT_BANKED = ROOT / "research" / "validity" / "et_prompt_distribution" / "measured_result.json"


# --------------------------------------------------------------------------- #
# numeric helpers (no scipy in the analytic venv)
# --------------------------------------------------------------------------- #
def bisect(f: Callable[[float], float], lo: float, hi: float,
           tol: float = 1e-13, max_it: int = 300) -> float:
    """Robust bracketed root find; raises if [lo,hi] does not bracket a root."""
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        raise ValueError(f"bisect: no sign change on [{lo},{hi}] -> {flo},{fhi}")
    for _ in range(max_it):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < tol or (hi - lo) < tol:
            return mid
        if flo * fm < 0.0:
            hi = mid
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def tps_from_et(et: float) -> float:
    """official = K_cal * E[T] (125.268 * 3.844 = 481.53)."""
    return K_CAL * et


def et_of_profile(a: list[float]) -> float:
    """E[T] = 1 + sum_{k=1..K} Pi_{j<=k} a_j (survival sum of committed length)."""
    s, prod = 0.0, 1.0
    for ak in a:
        prod *= ak
        s += prod
    return 1.0 + s


# --------------------------------------------------------------------------- #
# Input: banked per-position counters (lawine #282)
# --------------------------------------------------------------------------- #
def load_banked(banked_path: Path) -> dict[str, Any]:
    """Load lawine #282 whole-run per-position accepted-token counters + per-prompt
    deltas from the committed measured_result.json, and cross-check the per-pos
    vector against the documented W&B-verified constants (fail loudly on drift)."""
    src = json.loads(banked_path.read_text())
    base = src["baseline_counters"]
    fin = src["final_counters"]
    app = [f - b for f, b in zip(fin["accepted_per_pos"], base["accepted_per_pos"])]
    num_drafts = fin["num_drafts"] - base["num_drafts"]
    num_accepted = fin["num_accepted_tokens"] - base["num_accepted_tokens"]

    # cross-check against documented banked vector (W&B-verified provenance)
    if not np.allclose(app, BANKED_ACCEPTED_PER_POS, atol=1e-6):
        raise ValueError(f"banked accepted_per_pos {app} != documented "
                         f"{BANKED_ACCEPTED_PER_POS}; source file changed")
    if abs(num_drafts - BANKED_NUM_DRAFTS) > 1e-6 or abs(num_accepted - BANKED_NUM_ACCEPTED) > 1e-6:
        raise ValueError("banked num_drafts/num_accepted drifted from documented constants")
    if abs(sum(app) - num_accepted) > 1e-6:
        raise ValueError("per-position counts do not sum to num_accepted (prefix invariant broken)")

    return {
        "accepted_per_pos": app,
        "num_drafts": num_drafts,
        "num_accepted": num_accepted,
        "per_prompt": src["per_prompt"],
        "sum_per_prompt_drafts": src["sum_per_prompt_drafts"],
        "sum_per_prompt_accepted": src["sum_per_prompt_accepted"],
        "banked_source": str(banked_path),
    }


# --------------------------------------------------------------------------- #
# Step 1 -- decompose E[T] into the per-position acceptance profile
# --------------------------------------------------------------------------- #
def decompose(banked: dict[str, Any]) -> dict[str, Any]:
    app = banked["accepted_per_pos"]
    nd = banked["num_drafts"]
    K = K_SPEC

    # survival of j (# accepted draft tokens): G(m)=P(j>=m), m=0..K
    survival = [1.0] + [app[m - 1] / nd for m in range(1, K + 1)]
    # per-position conditional acceptance a_k = G(k)/G(k-1)
    a = [survival[m] / survival[m - 1] for m in range(1, K + 1)]
    # cumulative product Pi_{j<=k} a_j (== G(k), the round-trip check)
    cumprod = list(np.cumprod(a))

    e_j = sum(survival[1:])          # E[# accepted draft tokens]
    et = 1.0 + e_j                   # E[T] = 1 + E[j]

    # committed-length survival S(L>=t) for L=j+1 in {1..8} (the PR's "len" view):
    # S(1)=1; S(t)=G(t-1) for t=2..8 -> E[T]=sum_{t=1..8} S(t)
    committed_survival = [1.0] + survival[1:] + [0.0]  # index t=1..8 -> [1, G1..G7]; pad
    committed_survival = [1.0] + survival[1:]          # S(1..8) = [1, G(1)..G(7)]

    return {
        "K": K,
        "survival_j": survival,                       # G(0..7)
        "a_k": a,                                      # a_1..a_7 (conditional)
        "cumprod_a": cumprod,                          # Pi a (== G(1..7))
        "committed_survival": committed_survival,      # S(L>=1..8)
        "E_j": e_j,
        "E_T": et,
        "roundtrip_resid": et - (1.0 + sum(survival[1:])),  # 0 by construction
        "roundtrip_resid_vs_anchor": et - E_T_ANCHOR,
        "a1_vs_top1_anchor_resid": a[0] - TOP1_LINEAR_ACCEPT_ANCHOR,
        "marginal_accept_per_pos": [app[i] / nd for i in range(K)],  # = G(1..7), the "j>=k collapse"
    }


# --------------------------------------------------------------------------- #
# Step 2 -- identify the acceptance cliff (per-position token-loss table)
# --------------------------------------------------------------------------- #
def cliff(dec: dict[str, Any]) -> dict[str, Any]:
    G = dec["survival_j"]
    a = dec["a_k"]
    K = dec["K"]

    # First-rejection forfeit attribution: a step whose FIRST reject is at position
    # k commits L=k tokens and forfeits (K+1-k). P(first reject at k)=G(k-1)-G(k).
    # sum_k loss_k == (K+1) - E[T] (total expected forfeit vs the L=8 ideal).
    token_loss = [(G[k - 1] - G[k]) * ((K + 1) - k) for k in range(1, K + 1)]
    abs_survival_drop = [G[k - 1] - G[k] for k in range(1, K + 1)]
    # conditional-acceptance decay a_{k-1}-a_k (a_0 := 1 = always-emitted bonus)
    cond_drop = [(1.0 if k == 1 else a[k - 2]) - a[k - 1] for k in range(1, K + 1)]

    cliff_by_loss = int(np.argmax(token_loss)) + 1
    cliff_by_min_a = int(np.argmin(a)) + 1
    cliff_by_abs_drop = int(np.argmax(abs_survival_drop)) + 1
    cliff_by_cond_drop = int(np.argmax(cond_drop)) + 1

    return {
        "token_loss_per_pos": token_loss,             # ΔE[T]_k (forfeit attribution)
        "token_loss_total": float(sum(token_loss)),   # == 8 - E[T]
        "abs_survival_drop_per_pos": abs_survival_drop,
        "cond_accept_drop_per_pos": cond_drop,
        "acceptance_cliff_position": cliff_by_loss,    # PRIMARY cliff (max token loss)
        "cliff_position_by_min_conditional": cliff_by_min_a,
        "cliff_position_by_abs_survival_drop": cliff_by_abs_drop,
        "cliff_position_by_conditional_decay": cliff_by_cond_drop,
        "cliff_agrees_across_measures": (
            cliff_by_loss == cliff_by_min_a == cliff_by_abs_drop == cliff_by_cond_drop
        ),
        "conditional_acceptance_increases_with_depth": all(
            a[i] <= a[i + 1] + 1e-9 for i in range(len(a) - 1)
        ),
    }


# --------------------------------------------------------------------------- #
# Step 3 -- price the BUILT-raise target profile to public E[T] >= 4.966
# --------------------------------------------------------------------------- #
def price(dec: dict[str, Any], target: float) -> dict[str, Any]:
    a = dec["a_k"]
    G = dec["survival_j"]
    K = dec["K"]
    need_sum = target - 1.0  # need sum_{k=1..K} Pi a = target - 1

    # (0) raise a_1 ONLY (highest leverage: multiplies all downstream survival
    #     terms). T2 = sum_{k} Pi_{2<=j<=k} a_j  -> sum G(k) = a_1 * T2.
    T2 = sum(float(np.prod(a[1:m])) for m in range(1, K + 1))  # m=1 -> empty prod = 1
    a1_needed = need_sum / T2
    et_if_a1_perfect = 1.0 + 1.0 * T2  # a_1 = 1, others fixed
    a1_only_feasible = a1_needed <= 1.0

    # (a) uniform MULTIPLICATIVE lift a_k -> c * a_k
    c_mult = bisect(lambda c: sum((c ** m) * G[m] for m in range(1, K + 1)) - need_sum, 1.0, 4.0)
    a_mult = [c_mult * x for x in a]
    mult_feasible = all(x <= 1.0 + 1e-12 for x in a_mult)
    mult_max = max(a_mult)

    # (b) uniform ADDITIVE lift a_k -> a_k + delta  (headline per-position lift)
    delta_add = bisect(lambda d: et_of_profile([x + d for x in a]) - target, 0.0, 1.0)
    a_add = [x + delta_add for x in a]
    add_feasible = all(x <= 1.0 + 1e-12 for x in a_add)
    add_max = max(a_add)

    # (b2) deep-flat-only (the PR's literal "lift the cliff positions to a flat-decay
    #      profile holding early fixed"): HOLD a_1 at its measured value and set the
    #      deep positions a_2..a_K to a single flat conditional rate a_d (geometric /
    #      "flat-decay" survival), solving a_d to hit the target. This is EAGLE-3's
    #      exact lane -- it raises DEEP-position (j>=2) acceptance via multi-layer
    #      hidden-state fidelity (ubel #263) without touching the first-token bottleneck.
    a1 = a[0]
    n_deep = K - 1

    def _deep(ad: float) -> float:
        # E[T] = 1 + a1 * sum_{i=0..n_deep} ad^i
        geom = sum(ad ** i for i in range(n_deep + 1))
        return 1.0 + a1 * geom
    deep_geom_target = (need_sum / a1)  # sum_{i=0..n_deep} ad^i required
    deep_only_feasible = deep_geom_target <= (n_deep + 1)  # ad<=1 cap (geom max = n_deep+1)
    if deep_only_feasible:
        a_d = bisect(lambda ad: _deep(ad) - target, 0.0, 1.0)
    else:
        a_d = 1.0
    deep_flat_profile = [a1] + [a_d] * n_deep
    deep_lift_per_pos = [a_d - a[i] for i in range(1, K)]  # lift on each deep position

    # (c) cliff-targeted: lift the cliff (a_1) to perfect, then a uniform additive
    #     top-up on the deep positions a_2..a_K to reach the target (a "flat-decay"
    #     top-up holding the cliff at its ceiling).
    cliff_targeted: dict[str, Any]
    if et_if_a1_perfect >= target:
        cliff_targeted = {
            "scheme": "a_1 -> 1.0 alone overshoots target",
            "feasible": True, "a_k_star": [1.0] + a[1:],
            "E_T_star": et_if_a1_perfect,
        }
    else:
        def _f(d: float) -> float:
            prof = [1.0] + [min(a[i] + d, 1.0) for i in range(1, K)]
            return et_of_profile(prof) - target
        d_top = bisect(_f, 0.0, 1.0)
        prof = [1.0] + [min(a[i] + d_top, 1.0) for i in range(1, K)]
        cliff_targeted = {
            "scheme": "a_1 -> 1.0 + uniform additive top-up on a_2..a_K",
            "deep_topup_delta": d_top,
            "a_k_star": prof,
            "E_T_star": et_of_profile(prof),
            "feasible": all(x <= 1.0 + 1e-12 for x in prof),
        }

    return {
        "public_et_target": target,
        "need_survival_sum": need_sum,
        "current_survival_sum": dec["E_j"],
        "gap_in_et": target - dec["E_T"],
        # a_1-only (the high-leverage single position)
        "a1_only_needed": a1_needed,
        "a1_only_feasible": a1_only_feasible,
        "E_T_if_a1_perfect": et_if_a1_perfect,
        "E_T_if_a1_perfect_shortfall": target - et_if_a1_perfect,
        # uniform multiplicative
        "uniform_mult_factor": c_mult,
        "uniform_mult_profile": a_mult,
        "uniform_mult_max_ak": mult_max,
        "uniform_mult_feasible": mult_feasible,
        # uniform additive (HEADLINE min per-position lift)
        "uniform_add_delta": delta_add,
        "uniform_add_profile": a_add,
        "uniform_add_max_ak": add_max,
        "uniform_add_feasible": add_feasible,
        "min_per_position_acceptance_lift_to_497": delta_add,
        # deep-flat-only (PR's literal (b); EAGLE-3 lane)
        "deep_flat_rate_a_d": a_d,
        "deep_flat_profile": deep_flat_profile,
        "deep_flat_lift_per_pos": deep_lift_per_pos,
        "deep_flat_max_lift": max(deep_lift_per_pos) if deep_lift_per_pos else 0.0,
        "deep_only_feasible": deep_only_feasible,
        "deep_only_E_T": _deep(a_d),
        # cliff-targeted
        "cliff_targeted": cliff_targeted,
        # the load-bearing asymmetry: deep-only works, a1-only does not
        "feasibility_asymmetry_deep_yes_a1_no": (deep_only_feasible and not a1_only_feasible),
    }


# --------------------------------------------------------------------------- #
# Step 4 -- cross the linear cap (denken #119)
# --------------------------------------------------------------------------- #
def linear_cap(dec: dict[str, Any], target: float, target_round: float) -> dict[str, Any]:
    gap_precise = target - LINEAR_CAP
    gap_round = target_round - LINEAR_CAP
    deployed_at_cap = abs(dec["E_T"] - LINEAR_CAP) <= 0.02  # within measurement noise
    return {
        "linear_cap": LINEAR_CAP,
        "public_et_target": target,
        "linear_cap_gap_precise": gap_precise,          # 4.966 - 3.8445
        "linear_cap_gap_round": gap_round,              # 4.97 - 3.8445 = 1.126
        "deployed_et": dec["E_T"],
        "deployed_at_or_above_linear_cap": deployed_at_cap,
        "built_raise_requires_nonlinear_drafter": gap_round > 0.0 and deployed_at_cap,
    }


# --------------------------------------------------------------------------- #
# Step 5 -- per-prompt cliff structure (low-tail vs top-tail)
# --------------------------------------------------------------------------- #
def fit_constant_alpha(et_target: float, K: int) -> float:
    """alpha s.t. constant-conditional-acceptance survival reproduces E[T]:
    1 + sum_{k=1..K} alpha^k = E[T].  Single-parameter (max-entropy) fit to a mean."""
    et_target = min(max(et_target, 1.0 + 1e-9), float(K + 1) - 1e-9)
    return bisect(lambda al: 1.0 + sum(al ** k for k in range(1, K + 1)) - et_target, 1e-9, 1.0 - 1e-12)


def tail_structure(banked: dict[str, Any], dec: dict[str, Any]) -> dict[str, Any]:
    """Per-prompt MEANS are banked (lawine #282), but per-prompt PER-POSITION counts
    are NOT (the Prometheus per-pos counter was read whole-run only). We therefore
    split prompts into bottom/top quartile by their banked E[T]_p, fit a
    single-parameter survival per quartile, and report the implied cliff. Two
    models bracket the inference; both place the cliff at the same position, so the
    cliff POSITION is model-robust. Detecting a genuine cliff-position SHIFT between
    tails would require a per-prompt per-position remeasure (flagged as follow-up)."""
    K = dec["K"]
    per = banked["per_prompt"]
    ets = np.array([p["E_T"] for p in per], dtype=float)
    accs = np.array([p["delta_accepted"] for p in per], dtype=float)
    drf = np.array([p["delta_drafts"] for p in per], dtype=float)
    order = np.argsort(ets)
    q = len(per) // 4
    low_idx, top_idx = order[:q], order[-q:]

    def pooled_et(idx: np.ndarray) -> float:
        return 1.0 + float(accs[idx].sum() / drf[idx].sum())

    def cliff_pos_of_profile(a: list[float]) -> int:
        G = [1.0]
        for ak in a:
            G.append(G[-1] * ak)
        loss = [(G[k - 1] - G[k]) * ((K + 1) - k) for k in range(1, K + 1)]
        return int(np.argmax(loss)) + 1

    et_low, et_top = pooled_et(low_idx), pooled_et(top_idx)

    # Model A: constant conditional acceptance alpha per quartile.
    alpha_low, alpha_top = fit_constant_alpha(et_low, K), fit_constant_alpha(et_top, K)
    cliff_low_A = cliff_pos_of_profile([alpha_low] * K)
    cliff_top_A = cliff_pos_of_profile([alpha_top] * K)

    # Model B: shape-transfer -- each quartile shares the aggregate CONDITIONAL
    # shape, multiplicatively scaled to match its pooled E[T] (a_k -> s*a_k).
    a_agg = dec["a_k"]
    G_agg = dec["survival_j"]

    def fit_scale(et_target: float) -> float:
        need = et_target - 1.0
        return bisect(lambda s: sum((s ** m) * G_agg[m] for m in range(1, K + 1)) - need,
                      1e-6, 1.0 / max(a_agg))
    s_low, s_top = fit_scale(et_low), fit_scale(et_top)
    cliff_low_B = cliff_pos_of_profile([min(s_low * x, 1.0) for x in a_agg])
    cliff_top_B = cliff_pos_of_profile([min(s_top * x, 1.0) for x in a_agg])

    return {
        "quartile_size": q,
        "low_tail_pooled_et": et_low,
        "top_tail_pooled_et": et_top,
        "low_tail_indices": [int(i) for i in low_idx],
        "top_tail_indices": [int(i) for i in top_idx],
        # model A (constant alpha)
        "low_tail_alpha_constant": alpha_low,
        "top_tail_alpha_constant": alpha_top,
        "low_tail_cliff_position": cliff_low_A,        # reported headline
        "top_tail_cliff_position": cliff_top_A,
        # model B (shape transfer)
        "low_tail_scale_shapexfer": s_low,
        "top_tail_scale_shapexfer": s_top,
        "low_tail_cliff_position_modelB": cliff_low_B,
        "top_tail_cliff_position_modelB": cliff_top_B,
        # verdicts
        "cliff_position_invariant_across_tails": (
            cliff_low_A == cliff_top_A == cliff_low_B == cliff_top_B
        ),
        "low_tail_cliff_is_deeper_not_later": (
            cliff_low_A == cliff_top_A and alpha_low < alpha_top
        ),
        "per_prompt_per_position_banked": False,  # only per-prompt MEANS are banked
    }


# --------------------------------------------------------------------------- #
# Step 6 -- self-test (PRIMARY)
# --------------------------------------------------------------------------- #
def self_test(dec: dict[str, Any], clf: dict[str, Any], pr: dict[str, Any],
              lc: dict[str, Any], tail: dict[str, Any]) -> dict[str, Any]:
    K = dec["K"]
    G = dec["survival_j"]
    a = dec["a_k"]
    checks: dict[str, bool] = {}

    # (a) a_k decomposition round-trips E[T] (sum survival == E[T]-1; reproduces 3.844)
    sum_surv = sum(G[1:])
    checks["a_roundtrips_E_T"] = abs((1.0 + sum_surv) - dec["E_T"]) <= 1e-9
    checks["a_reproduces_3844_anchor"] = abs(dec["E_T"] - E_T_ANCHOR) <= 0.05
    # cumulative product matches survival exactly
    checks["a_cumprod_matches_survival"] = all(
        abs(dec["cumprod_a"][i] - G[i + 1]) <= 1e-9 for i in range(K)
    )

    # (b) survival monotone non-increasing
    checks["b_survival_monotone"] = all(G[i] >= G[i + 1] - 1e-12 for i in range(len(G) - 1))

    # (c) every a_k in [0,1]
    checks["c_a_k_in_unit_interval"] = all(-1e-12 <= x <= 1.0 + 1e-12 for x in a)

    # (d) BUILT-raise target reproduces E[T]=4.97 within tol (uniform additive solve)
    et_add = et_of_profile([x + pr["uniform_add_delta"] for x in a])
    checks["d_built_raise_reproduces_497"] = abs(et_add - pr["public_et_target"]) <= 1e-6
    et_mult = 1.0 + sum((pr["uniform_mult_factor"] ** m) * G[m] for m in range(1, K + 1))
    checks["d_built_raise_mult_reproduces_497"] = abs(et_mult - pr["public_et_target"]) <= 1e-6

    # (e) linear-cap gap reproduces 4.97 - 3.8445 = 1.126
    checks["e_linear_cap_gap_1p126"] = abs(lc["linear_cap_gap_round"] - 1.126) <= 1e-3

    # (f) NaN-clean across reported scalars
    scalar_keys = [
        dec["E_T"], dec["E_j"], clf["token_loss_total"], pr["uniform_add_delta"],
        pr["uniform_mult_factor"], pr["E_T_if_a1_perfect"], pr["a1_only_needed"],
        lc["linear_cap_gap_round"], lc["linear_cap_gap_precise"],
        tail["low_tail_pooled_et"], tail["top_tail_pooled_et"],
        tail["low_tail_alpha_constant"], tail["top_tail_alpha_constant"],
    ]
    checks["f_nan_clean"] = all(math.isfinite(float(x)) for x in scalar_keys) and \
        all(math.isfinite(float(x)) for x in a) and all(math.isfinite(float(x)) for x in G)

    # (g) imported constants EXACT and UNCHANGED
    checks["g_constants_imported_exact"] = (
        OFFICIAL == 481.53 and CEILING_LAMBDA1 == 520.95 and K_CAL == 125.268
        and E_T_ANCHOR == 3.844 and STEP_US == 1218.2 and K_SPEC == 7
        and HONEST_500_FLOOR == 3.9914 and LINEAR_CAP == 3.8445
    )

    # bonus: token-loss attribution sums to the total forfeit (8 - E[T])
    checks["h_token_loss_sums_to_forfeit"] = abs(
        clf["token_loss_total"] - (E_T_MAX - dec["E_T"])
    ) <= 1e-9
    # bonus: a_1 reproduces #76 top-1 accept anchor (independent measurement cross-check)
    checks["i_a1_matches_top1_anchor"] = abs(a[0] - TOP1_LINEAR_ACCEPT_ANCHOR) <= 0.01

    gate = bool(
        checks["a_roundtrips_E_T"] and checks["a_reproduces_3844_anchor"]
        and checks["a_cumprod_matches_survival"] and checks["b_survival_monotone"]
        and checks["c_a_k_in_unit_interval"] and checks["d_built_raise_reproduces_497"]
        and checks["d_built_raise_mult_reproduces_497"] and checks["e_linear_cap_gap_1p126"]
        and checks["f_nan_clean"] and checks["g_constants_imported_exact"]
        and checks["h_token_loss_sums_to_forfeit"]
    )
    return {"checks": checks, "per_position_acceptance_decay_self_test_passes": gate}


# --------------------------------------------------------------------------- #
# assemble report
# --------------------------------------------------------------------------- #
def build_report(banked: dict[str, Any]) -> dict[str, Any]:
    dec = decompose(banked)
    clf = cliff(dec)
    pr = price(dec, PUBLIC_ET_TARGET)
    lc = linear_cap(dec, PUBLIC_ET_TARGET, PUBLIC_ET_TARGET_ROUND)
    tail = tail_structure(banked, dec)
    st = self_test(dec, clf, pr, lc, tail)

    # reconciliation: honest-500 floor / private factor ~= public target
    public_from_floor = HONEST_500_FLOOR / PRIVATE_FACTOR

    report = {
        "pr": 289,
        "leg": "per-position acceptance decay: BUILT-raise target to E[T]>=4.97",
        "per_position_acceptance_decay_analysis_only": True,
        "banked_source": banked["banked_source"],
        "imported": {
            "official": OFFICIAL, "ceiling_lambda1": CEILING_LAMBDA1, "K_cal": K_CAL,
            "E_T_anchor": E_T_ANCHOR, "step_us": STEP_US, "step_ms": STEP_MS, "tau": TAU,
            "K_spec": K_SPEC, "E_T_max": E_T_MAX, "honest_500_floor": HONEST_500_FLOOR,
            "linear_cap_denken119": LINEAR_CAP, "public_et_target_fern281": PUBLIC_ET_TARGET,
            "public_et_target_round": PUBLIC_ET_TARGET_ROUND, "private_factor": PRIVATE_FACTOR,
            "private_verified": PRIVATE_VERIFIED, "top1_linear_accept_anchor": TOP1_LINEAR_ACCEPT_ANCHOR,
            "lawine282_et_mean": LAWINE_ET_MEAN, "lawine282_et_median": LAWINE_ET_MEDIAN,
            "lawine282_et_p5": LAWINE_ET_P5, "lawine282_et_std": LAWINE_ET_STD,
            "banked_accepted_per_pos": BANKED_ACCEPTED_PER_POS,
            "banked_num_drafts": BANKED_NUM_DRAFTS, "banked_num_accepted": BANKED_NUM_ACCEPTED,
        },
        "decomposition": dec,
        "cliff": clf,
        "pricing": pr,
        "linear_cap": lc,
        "tail_structure": tail,
        "self_test": st["checks"],
        # headline metrics
        "per_position_acceptance_decay_self_test_passes": st["per_position_acceptance_decay_self_test_passes"],
        "acceptance_cliff_position": clf["acceptance_cliff_position"],
        "min_per_position_acceptance_lift_to_497": pr["min_per_position_acceptance_lift_to_497"],
        "built_raise_requires_nonlinear_drafter": lc["built_raise_requires_nonlinear_drafter"],
        # reconciliation diagnostics
        "public_et_from_honest_floor_over_private": public_from_floor,
        "public_et_reconciliation_resid": public_from_floor - PUBLIC_ET_TARGET,
        "composition_anchor_roundtrip_tps": tps_from_et(E_T_ANCHOR),
    }
    return report


# --------------------------------------------------------------------------- #
# wandb
# --------------------------------------------------------------------------- #
def log_wandb(report: dict[str, Any], name: str, group: str) -> str | None:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[ppad] wandb unavailable ({exc})", flush=True)
        return None
    try:
        dec, clf, pr, lc, tail = (report["decomposition"], report["cliff"],
                                  report["pricing"], report["linear_cap"], report["tail_structure"])
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            name=name, group=group, job_type="analysis",
            config={
                "pr": 289, "analysis_only": True, "banked_source": report["banked_source"],
                "K_spec": K_SPEC, "K_cal": K_CAL, "official": OFFICIAL,
                "E_T_anchor": E_T_ANCHOR, "linear_cap": LINEAR_CAP,
                "public_et_target": PUBLIC_ET_TARGET, "private_factor": PRIVATE_FACTOR,
                "honest_500_floor": HONEST_500_FLOOR,
            },
        )
        flat = {
            "primary/per_position_acceptance_decay_self_test_passes":
                report["per_position_acceptance_decay_self_test_passes"],
            "test/acceptance_cliff_position": report["acceptance_cliff_position"],
            "test/min_per_position_acceptance_lift_to_497":
                report["min_per_position_acceptance_lift_to_497"],
            "built_raise_requires_nonlinear_drafter": report["built_raise_requires_nonlinear_drafter"],
            "E_T_decomposed": dec["E_T"],
            "E_T_roundtrip_resid_vs_anchor": dec["roundtrip_resid_vs_anchor"],
            "a1_vs_top1_anchor_resid": dec["a1_vs_top1_anchor_resid"],
            "conditional_acceptance_increases_with_depth": clf["conditional_acceptance_increases_with_depth"],
            "cliff_agrees_across_measures": clf["cliff_agrees_across_measures"],
            "token_loss_total": clf["token_loss_total"],
            "a1_only_needed": pr["a1_only_needed"],
            "a1_only_feasible": pr["a1_only_feasible"],
            "E_T_if_a1_perfect": pr["E_T_if_a1_perfect"],
            "E_T_if_a1_perfect_shortfall": pr["E_T_if_a1_perfect_shortfall"],
            "uniform_add_delta": pr["uniform_add_delta"],
            "uniform_add_feasible": pr["uniform_add_feasible"],
            "uniform_add_max_ak": pr["uniform_add_max_ak"],
            "uniform_mult_factor": pr["uniform_mult_factor"],
            "uniform_mult_feasible": pr["uniform_mult_feasible"],
            "uniform_mult_max_ak": pr["uniform_mult_max_ak"],
            "deep_flat_rate_a_d": pr["deep_flat_rate_a_d"],
            "deep_only_feasible": pr["deep_only_feasible"],
            "deep_flat_max_lift": pr["deep_flat_max_lift"],
            "feasibility_asymmetry_deep_yes_a1_no": pr["feasibility_asymmetry_deep_yes_a1_no"],
            "gap_in_et": pr["gap_in_et"],
            "linear_cap_gap_round": lc["linear_cap_gap_round"],
            "linear_cap_gap_precise": lc["linear_cap_gap_precise"],
            "deployed_at_or_above_linear_cap": lc["deployed_at_or_above_linear_cap"],
            "low_tail_pooled_et": tail["low_tail_pooled_et"],
            "top_tail_pooled_et": tail["top_tail_pooled_et"],
            "low_tail_cliff_position": tail["low_tail_cliff_position"],
            "top_tail_cliff_position": tail["top_tail_cliff_position"],
            "low_tail_alpha_constant": tail["low_tail_alpha_constant"],
            "top_tail_alpha_constant": tail["top_tail_alpha_constant"],
            "cliff_position_invariant_across_tails": tail["cliff_position_invariant_across_tails"],
            "public_et_reconciliation_resid": report["public_et_reconciliation_resid"],
        }
        run.summary.update(flat)

        # per-position decomposition table
        ptbl = wandb.Table(columns=[
            "position_k", "conditional_accept_a_k", "marginal_survival_G_k",
            "cumprod_a", "token_loss_forfeit", "abs_survival_drop",
            "uniform_add_a_k_star", "uniform_mult_a_k_star",
        ])
        for k in range(K_SPEC):
            ptbl.add_data(
                k + 1, dec["a_k"][k], dec["survival_j"][k + 1], dec["cumprod_a"][k],
                clf["token_loss_per_pos"][k], clf["abs_survival_drop_per_pos"][k],
                pr["uniform_add_profile"][k], pr["uniform_mult_profile"][k],
            )
        run.log({"per_position_profile": ptbl})

        # survival-curve table (includes G(0)=1)
        stbl = wandb.Table(columns=["m", "survival_G_m_P_j_ge_m"])
        for m, g in enumerate(dec["survival_j"]):
            stbl.add_data(m, g)
        run.log({"survival_curve": stbl})

        # pricing-scenarios table
        sc = wandb.Table(columns=["scenario", "E_T_reached", "max_a_k", "feasible", "note"])
        sc.add_data("deployed", dec["E_T"], max(dec["a_k"]), True, "banked baseline")
        sc.add_data("a1_only_perfect", pr["E_T_if_a1_perfect"], 1.0, True,
                    f"a1->1 short by {pr['E_T_if_a1_perfect_shortfall']:.4f}")
        sc.add_data("uniform_additive", PUBLIC_ET_TARGET, pr["uniform_add_max_ak"],
                    pr["uniform_add_feasible"], f"delta=+{pr['uniform_add_delta']:.4f}/pos")
        sc.add_data("uniform_multiplicative", PUBLIC_ET_TARGET, pr["uniform_mult_max_ak"],
                    pr["uniform_mult_feasible"], f"factor={pr['uniform_mult_factor']:.4f}")
        sc.add_data("deep_flat_only_hold_a1", pr["deep_only_E_T"], max(pr["deep_flat_profile"]),
                    pr["deep_only_feasible"],
                    f"a_2..a_7->{pr['deep_flat_rate_a_d']:.4f} flat (EAGLE-3 lane), a1 held")
        sc.add_data("cliff_targeted", pr["cliff_targeted"]["E_T_star"],
                    max(pr["cliff_targeted"]["a_k_star"]), pr["cliff_targeted"]["feasible"],
                    pr["cliff_targeted"]["scheme"])
        run.log({"pricing_scenarios": sc})

        rid = run.id
        print(f"[ppad] W&B run: {run.url}", flush=True)
        run.finish()
        return rid
    except Exception as exc:  # noqa: BLE001
        print(f"[ppad] wandb log failed ({exc})", flush=True)
        return None


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="run the decomposition + pricing + self-test (PRIMARY) over banked numbers.")
    ap.add_argument("--banked", type=Path, default=DEFAULT_BANKED,
                    help="path to lawine #282 measured_result.json (banked per-pos counters).")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="per-position-acceptance-decay")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="kanna/per-position-acceptance-decay")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    banked = load_banked(args.banked.resolve())
    report = build_report(banked)

    wid = None
    if not args.no_wandb:
        wid = log_wandb(report, args.wandb_name, args.wandb_group)
    report["wandb_run_id"] = wid
    RESULTS_PATH.write_text(json.dumps(report, indent=2))

    dec, clf, pr, lc, tail = (report["decomposition"], report["cliff"], report["pricing"],
                              report["linear_cap"], report["tail_structure"])
    print("\n========== PER-POSITION ACCEPTANCE DECAY (PR #289) ==========", flush=True)
    print(f"E[T] decomposed       : {dec['E_T']:.5f}  (anchor 3.844; resid "
          f"{dec['roundtrip_resid_vs_anchor']:+.5f})", flush=True)
    print(f"a_1..a_7 (conditional): " + " ".join(f"{x:.4f}" for x in dec["a_k"]), flush=True)
    print(f"survival G(1..7)      : " + " ".join(f"{x:.4f}" for x in dec["survival_j"][1:]), flush=True)
    print(f"cond accept INCREASES : {clf['conditional_acceptance_increases_with_depth']} "
          f"(survivorship)", flush=True)
    print(f"token-loss/pos        : " + " ".join(f"{x:.3f}" for x in clf["token_loss_per_pos"]), flush=True)
    print(f"acceptance_cliff_pos  : {clf['acceptance_cliff_position']} "
          f"(agrees across measures: {clf['cliff_agrees_across_measures']})", flush=True)
    print(f"--- pricing to public E[T]={pr['public_et_target']} (gap {pr['gap_in_et']:+.4f}) ---", flush=True)
    print(f"a1-only needed        : {pr['a1_only_needed']:.5f}  feasible={pr['a1_only_feasible']} "
          f"(a1=1 -> E[T]={pr['E_T_if_a1_perfect']:.4f}, short {pr['E_T_if_a1_perfect_shortfall']:.4f})",
          flush=True)
    print(f"uniform ADD delta     : +{pr['uniform_add_delta']:.5f}/pos  feasible={pr['uniform_add_feasible']} "
          f"(max a_k {pr['uniform_add_max_ak']:.4f})  <- min_per_position_lift", flush=True)
    print(f"uniform MULT factor   : x{pr['uniform_mult_factor']:.5f}  feasible={pr['uniform_mult_feasible']} "
          f"(max a_k {pr['uniform_mult_max_ak']:.4f})", flush=True)
    print(f"deep-flat-only (a1 held): a_2..a_7->{pr['deep_flat_rate_a_d']:.4f}  feasible={pr['deep_only_feasible']} "
          f"(EAGLE-3 lane; max deep lift +{pr['deep_flat_max_lift']:.4f})", flush=True)
    print(f"ASYMMETRY deep-yes/a1-no: {pr['feasibility_asymmetry_deep_yes_a1_no']}  "
          f"(deep headroom reachable, first-token ceiling-bound)", flush=True)
    print(f"linear-cap gap (round): {lc['linear_cap_gap_round']:.4f}  (4.97-3.8445=1.126)", flush=True)
    print(f"requires nonlinear    : {lc['built_raise_requires_nonlinear_drafter']} "
          f"(deployed {lc['deployed_et']:.4f} at/above cap {LINEAR_CAP})", flush=True)
    print(f"tail cliff low/top    : {tail['low_tail_cliff_position']}/{tail['top_tail_cliff_position']} "
          f"(alpha {tail['low_tail_alpha_constant']:.4f} vs {tail['top_tail_alpha_constant']:.4f}; "
          f"deeper-not-later={tail['low_tail_cliff_is_deeper_not_later']})", flush=True)
    print(f"PRIMARY self_test     : {report['per_position_acceptance_decay_self_test_passes']}", flush=True)
    print(f"wandb run             : {wid}", flush=True)
    print(f"artifacts             : {RESULTS_PATH}", flush=True)

    if args.self_test and not report["per_position_acceptance_decay_self_test_passes"]:
        failed = [k for k, v in report["self_test"].items() if not v]
        print(f"[ppad] SELF-TEST FAILED: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
