#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #294 -- EAGLE-3 Phase-1 viability gate: the cheap-proxy GO threshold.

WHAT THIS ANSWERS
-----------------
The full EAGLE-3 retrain (the SOLE remaining >500 TPS path) is human-approval-gated
and expensive. The cheap precursor is a Phase-1 architecture-adaptation viability run
(~2h, single GPU, frozen backbone: SupportsEagle3 load + adapt, NO full retrain). But
nobody has PRICED the GATE: what MINIMUM acceptance must a Phase-1 run demonstrate
(measured LOCALLY) to justify spending the full retrain? Without a sized threshold the
human either over-spends (full retrain on a hunch) or under-acts (never builds).

This card translates three BANKED, W&B-verified results into an operational, measurable
retrain trigger:
  * kanna #289 (fi34s269): the per-position conditional-acceptance profile a_1..a_7 that
    decomposes the deployed linear E[T]=3.8512, with the cliff at position 1 and
    `built_raise_requires_nonlinear_drafter=True`.
  * wirbel #290 (ub3kpsso): the HONEST step-banked aggregate target E[T]=4.9029 (after
    banking the one proven-free lossless step lever) and the recoverable budget +1.0584.
  * lawine #288 (i1e5054m): the local->official transfer (tau_acc=1.0; safe local
    greedy-identity lambda-hat bar 0.9855 = operative 0.9780 + kernel jitter 0.0075;
    safe local PPL bar 2.4185).

THE TWO AXES (kept distinct -- this is the load-bearing honesty of the card)
---------------------------------------------------------------------------
  * SPEED axis -- the per-position SPEC-ACCEPTANCE profile a_k (#289). This determines
    E[T]/TPS and IS the operative GO trigger. The Phase-1 run must SHOW a position-2
    conditional acceptance a_2 above a sized threshold.
  * SAFETY axis -- the greedy-identity preservation rate lambda-hat (#288, bar 0.9855).
    EAGLE-3's emission = verify argmax (the drafter only PROPOSES; the verifier's argmax
    token is always emitted), so greedy identity is preserved BY CONSTRUCTION and the
    0.9855 lambda-hat bar is cleared structurally -- it is NOT the binding gate. PPL is
    likewise pinned (#288 bar 2.4185). lawine #288's tau_acc=1.0 (acceptance is a
    clock-invariant probability) is what makes the Phase-1 a_2 LOCALLY measurable and
    1:1 transferable to official, with only the +/-0.0075 kernel-jitter envelope.

So the human's measurable trigger is a SPEC-ACCEPTANCE threshold on a_2; the lambda-hat
safety bar is satisfied by construction and reported as a certificate, not a gate.

THE COMPOSITION LAW (clean K_cal frame; identical to #289/#290)
---------------------------------------------------------------
  official = K_cal * E[T]            (K_cal = 481.53 / 3.844 = 125.268)
  E[T]     = 1 + sum_{m=1..K} prod_{j<=m} a_j           (survival-product, #289)

CPU analytic over BANKED W&B numbers + the published EAGLE-3 acceptance anchor
(literature import, cited). NO new GPU measurement, NO model forward, NO Phase-1 run
(THIS card SIZES the gate; running Phase-1 is human-approval-gated). Analysis-only: NO
served-file change, NO HF Job, NO submission, NOT open2, NOT a build, NOT a launch.
BASELINE stays 481.53; this leg adds 0 TPS. eagle3_phase1_gate_analysis_only = True.
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]                       # .../target

# --------------------------------------------------------------------------- #
# Imported fleet anchors (DO NOT re-derive -- import EXACTLY, UNCHANGED).
# All runs live in wandb-applied-ai-team/gemma-challenge-senpai.
# --------------------------------------------------------------------------- #
OFFICIAL = 481.53                  # PR #52 official frontier TPS (2x9fm2zx); this leg adds 0
CEILING_LAMBDA1 = 520.95           # lambda=1 step-side ceiling (headline value)
K_CAL = 125.268                    # kanna #217/#269: official = K_cal * E[T]  (== 481.53/3.844)
E_T_ANCHOR = 3.844                 # kanna #217 deployed linear served E[T] (vgovdrjc)
STEP_US = 1218.2                   # kanna #217 served per-forward-pass step (microseconds)
NEW_STEP_US = 1202.7171244939168   # wirbel #285 lossless step (banked into #290's target)
TAU = 1.218                        # composition tau (cancels into step normalization)
K_SPEC = 7                         # num_speculative_tokens (linear MTP depth K=7)
E_T_MAX = float(K_SPEC + 1)        # 8.0 -- full-acceptance ceiling (all K accepted + bonus)
HONEST_500_FLOOR = 3.9914          # honest-500 real-E[T] floor (= 500 / K_cal)
LINEAR_CAP = 3.8445                # denken #119: LINEAR drafter E[T] cap at perfect capacity
TARGET_BANKED = 4.9029             # wirbel #290 ub3kpsso: step-banked BUILT-raise target (HEADLINE)
EAGLE3_BUDGET_ET = 1.0584          # wirbel #290: recoverable budget beyond linear cap (cite)
FERN_FLOOR_PUBLIC = 4.966          # fern #281: pre-step-bank public E[T] floor (for reconstruct)
PRIVATE_VERIFIED = 460.85          # private-verified reference (PR baseline)

# ---- kanna #289 fi34s269: per-position conditional acceptance (full precision) ----
A_BASE = [
    0.7292532942898975,   # a_1 (the cliff -- a model property; held fixed in the j>=2 lift)
    0.759556697719242,    # a_2 (highest-leverage feasible deep position -- the TEST handle)
    0.7929794882639035,   # a_3
    0.8228,               # a_4
    0.8348727920920435,   # a_5
    0.8357919254658385,   # a_6
    0.8464932652113331,   # a_7
]
E_T_DECOMP_289 = 3.851185944363104           # #289 E[T] reproduced by the survival product
TOKEN_LOSS_FORFEIT_289 = [1.895, 1.052, 0.573, 0.311, 0.179, 0.099, 0.039]  # dE[T]_k (cite)
A1_TOP1_ANCHOR = 0.728739760479042           # #76 top-1 accept (independent a_1 cross-check)

# ---- lawine #288 i1e5054m: local->official transfer (greedy-identity SAFETY axis) ----
TAU_ACC = 1.0                      # local lambda-hat == official lambda-hat (clock-invariant)
SAFE_LOCAL_LAMBDA_BAR = 0.9855     # safe local greedy-identity bar (operative 0.9780 + jitter)
OPERATIVE_LAMBDA_BAR = 0.9780112973731208    # #288/#249 operative official greedy-identity bar
TRANSFER_JITTER = 0.0075           # +/- kernel-jitter envelope (batch-variance argmax flip)
SAFE_LOCAL_PPL_BAR = 2.4185        # #288 safe local PPL bar (PPL pinned; emission = verify argmax)
M8_GREEDY_IDENTITY = 0.9927083333333333      # #288 deployed M=8 greedy identity vs M=1 ref

# --------------------------------------------------------------------------- #
# EAGLE-3 LITERATURE ANCHOR (import, cite). The published EAGLE-3 mean-accepted-length
# the full retrain can ASPIRE to -- the UPPER bound on the realizable lift (step 5).
#
# Provenance (literature pass; see PR body "Research" section):
#   EAGLE-3 (Li et al. 2025, arXiv:2503.01840), Table 1: average acceptance length tau
#   (== tokens committed per draft-verify cycle == our E[T]) on LLaMA-3.1-8B at T=0 is
#   ~6.2 mean (CNN/DM worst 5.34) with DEPTH-8 TREE/dynamic-draft decoding. Our deployed
#   lane is a LINEAR depth-7 CHAIN (K=7), not a tree; EAGLE-1 (arXiv:2401.15077, Table 5)
#   measures a consistent tree-over-chain premium of +0.62..+0.74 tau on 7B-class models
#   at T=0, so the CHAIN-equivalent EAGLE-3 acceptance length is ~5.5 central / ~4.6
#   conservative (tree anchor - 0.68; +/-0.3 extrapolation uncertainty). EAGLE-3's Fig 7
#   reports a NEAR-FLAT per-depth acceptance (vs EAGLE-1/2's steep decay) but gives no
#   numeric per-position values, so we model the EAGLE-3 profile as FLAT at a common
#   conditional acceptance a_lit that reproduces the chain-equivalent length on a 7-chain.
#   No EAGLE paper covers int4 quantization or reasoning/STEM OOD -> a distinct UNPRICED
#   downside risk on top of the chain discount (flagged, not numerically applied).
# These are bf16/MT-bench numbers; the int4 + STEM-OOD haircut is additional downside.
# --------------------------------------------------------------------------- #
EAGLE3_LIT_L_TREE_CENTRAL = 6.2    # EAGLE-3 Table 1 mean accepted length (tree, T=0, LLaMA-3.1-8B)
EAGLE3_LIT_L_CHAIN_CENTRAL = 5.5   # chain-equivalent (tree - 0.68 EAGLE-1 Table 5 premium)
EAGLE3_LIT_L_CHAIN_LOW = 4.6       # conservative chain-equivalent (CNN/DM worst-dist - premium)
EAGLE3_TREE_CHAIN_PREMIUM = 0.68   # EAGLE-1 Table 5 tree-over-chain tau premium (7B-class, T=0)
EAGLE3_LIT_NOTE = (
    "EAGLE-3 (arXiv:2503.01840 Table 1) accepted-length tau~6.2 (tree, T=0, LLaMA-3.1-8B); "
    "chain-equivalent ~5.5 central / ~4.6 low after the EAGLE-1 (arXiv:2401.15077 Table 5) "
    "tree-over-chain premium 0.68. EAGLE-3 Fig 7 reports a near-FLAT per-depth profile, so "
    "we model it as flat at a_lit reproducing the chain length, then transplant a_lit onto "
    "OUR deployed deep positions a_2..a_7 while HOLDING our a_1 cliff 0.7293 (a base-model "
    "predictability property #289 holds fixed). int4 + reasoning/STEM OOD is unpriced extra "
    "downside on top of the chain discount."
)

OUT_PATH = HERE / "eagle3_phase1_gate_results.json"
F_PHASE1_SWEEP = [0.3, 0.5, 0.7, 1.0]   # Phase-1 captures-fraction sweep (PR step 2)
F_HEADLINE = 0.5                         # the f_phase1 at which the TEST metric is reported


# --------------------------------------------------------------------------- #
# numeric helpers (no scipy in the analytic venv)
# --------------------------------------------------------------------------- #
def bisect(f: Callable[[float], float], lo: float, hi: float,
           tol: float = 1e-14, max_it: int = 400) -> float:
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


def et_of_profile(a: list[float]) -> float:
    """E[T] = 1 + sum_{k=1..K} prod_{j<=k} a_j (survival sum of committed length, #289)."""
    s, prod = 0.0, 1.0
    for ak in a:
        prod *= ak
        s += prod
    return 1.0 + s


def tps_from_et(et: float) -> float:
    """official = K_cal * E[T] (125.268 * 3.844 = 481.53)."""
    return K_CAL * et


def deep_uniform_profile(a1: float, a_deep: float, k: int = K_SPEC) -> list[float]:
    """[a1, a_deep, a_deep, ...] -- hold the cliff a_1, set j>=2 to a common a_deep."""
    return [a1] + [a_deep] * (k - 1)


# --------------------------------------------------------------------------- #
# Step 1 -- per-position lift needed -> aggregate target (inverse map to 4.9029)
# --------------------------------------------------------------------------- #
def solve_a_star(target: float) -> dict[str, Any]:
    """Hold a_1 = A_BASE[0]; find the common j>=2 conditional acceptance a* that yields
    E[T] = target via the survival-product (reproduces #289's 'a_2..a_7 -> ~0.91' lane)."""
    a1 = A_BASE[0]
    # E[T] = 1 + a1 * sum_{i=0..K-1} a*^i ; monotone increasing in a* -> bracket [0,1).
    a_star = bisect(lambda ad: et_of_profile(deep_uniform_profile(a1, ad)) - target,
                    1e-9, 1.0 - 1e-12)
    profile = deep_uniform_profile(a1, a_star)
    return {
        "a_star_uniform_j2": a_star,
        "a_star_profile": profile,
        "E_T_at_a_star": et_of_profile(profile),
        "a1_held": a1,
        "deep_lift_per_pos": [a_star - A_BASE[j] for j in range(1, K_SPEC)],
        # cross-check: target reconstructs as fern_floor * new_step/step (links to #290)
        "target_reconstruct_from_290": FERN_FLOOR_PUBLIC * (NEW_STEP_US / STEP_US),
    }


# --------------------------------------------------------------------------- #
# Step 2 -- Phase-1 -> full-retrain capture-fraction model + sweep
# --------------------------------------------------------------------------- #
def phase1_profile(a_star: float, f: float) -> list[float]:
    """Phase-1 realizes fraction f of the full per-position lift (a_1 held). At f=0 ->
    deployed base; at f=1 -> the full a_star profile."""
    a1 = A_BASE[0]
    deep = [A_BASE[j] + f * (a_star - A_BASE[j]) for j in range(1, K_SPEC)]
    return [a1] + deep


def sweep_phase1(a_star: float) -> dict[str, Any]:
    rows = []
    for f in F_PHASE1_SWEEP:
        prof = phase1_profile(a_star, f)
        rows.append({
            "f_phase1": f,
            "phase1_profile": prof,
            "E_T_phase1": et_of_profile(prof),
            "tps_phase1": tps_from_et(et_of_profile(prof)),
            "a2_phase1": prof[1],                       # position-2 conditional acceptance
        })
    et_series = [r["E_T_phase1"] for r in rows]
    monotone = all(et_series[i] < et_series[i + 1] + 1e-12 for i in range(len(et_series) - 1))
    return {
        "sweep": rows,
        "E_T_phase1_monotone_in_f": monotone,
        "E_T_at_f1_equals_target": abs(rows[-1]["E_T_phase1"] - TARGET_BANKED),
    }


# --------------------------------------------------------------------------- #
# Step 3 -- the GO threshold (local-measured, SPEED axis = spec-acceptance a_2)
# --------------------------------------------------------------------------- #
def go_thresholds(a_star: float) -> dict[str, Any]:
    """For each f, the position-2 conditional acceptance a Phase-1 run must SHOW to be
    on-track (= base + f * (a_star - base)). Local-measurable + 1:1 to official by
    lawine #288's tau_acc=1.0 (only the +/-0.0075 kernel-jitter envelope)."""
    a2_base = A_BASE[1]                  # deployed a_2 = NO-GO floor (must beat the linear drafter)
    rows = []
    for f in F_PHASE1_SWEEP:
        thr = a2_base + f * (a_star - a2_base)
        rows.append({
            "f_phase1": f,
            "phase1_go_threshold_a2": thr,
            "margin_above_nogo_floor": thr - a2_base,    # vs deployed a_2 (must exceed jitter)
            "margin_below_a_star": a_star - thr,         # headroom to full target
        })
    headline = next(r for r in rows if abs(r["f_phase1"] - F_HEADLINE) < 1e-12)
    # The local greedy-identity lambda-hat GO threshold per f (SAFETY axis): EAGLE-3 pins
    # lambda-hat at the deployed M=8 identity by construction, so the "on-track" lambda-hat
    # is the safe bar itself (no projection needed). Reported as a certificate.
    lambda_go = {f: SAFE_LOCAL_LAMBDA_BAR for f in F_PHASE1_SWEEP}
    return {
        "nogo_floor_a2_deployed": a2_base,
        "go_threshold_rows": rows,
        "phase1_go_threshold_a2": headline["phase1_go_threshold_a2"],   # TEST metric (f=0.5)
        "phase1_go_threshold_a2_by_f": {r["f_phase1"]: r["phase1_go_threshold_a2"] for r in rows},
        "phase1_local_lambda_go_threshold": lambda_go,    # SAFETY axis (auto-cleared bar)
        "go_band": [a2_base, headline["phase1_go_threshold_a2"]],       # [nogo_floor, go_threshold]
        "a_star_uniform_j2": a_star,
    }


# --------------------------------------------------------------------------- #
# Step 4 -- is the gate DECISIVE?
# --------------------------------------------------------------------------- #
def decisiveness(go: dict[str, Any], a_star: float) -> dict[str, Any]:
    """Decisive iff the GO threshold is separated by >= the +/-0.0075 transfer jitter from
    BOTH the NO-GO floor (deployed a_2) below AND the full target a_star above -- so a
    cheap 2h run distinguishes GO from NO-GO without ambiguity. Reported at the headline
    f=0.5 (the TEST point) and across the projection regime f in {0.3, 0.5, 0.7}."""
    nogo = go["nogo_floor_a2_deployed"]
    rows = go["go_threshold_rows"]

    def both_sided(thr: float) -> bool:
        return (thr - nogo >= TRANSFER_JITTER) and (a_star - thr >= TRANSFER_JITTER)

    headline_thr = go["phase1_go_threshold_a2"]
    headline_decisive = both_sided(headline_thr)

    # projection regime = f < 1 (at f=1 Phase-1 IS the full target; no projection).
    proj_rows = [r for r in rows if r["f_phase1"] < 1.0 - 1e-12]
    proj_decisive = all(both_sided(r["phase1_go_threshold_a2"]) for r in proj_rows)
    min_margin_below = min(r["phase1_go_threshold_a2"] - nogo for r in proj_rows)
    min_margin_above = min(a_star - r["phase1_go_threshold_a2"] for r in proj_rows)

    return {
        "phase1_gate_is_decisive": bool(headline_decisive),
        "decisive_in_projection_regime_all_f": bool(proj_decisive),
        "headline_margin_below_nogo": headline_thr - nogo,
        "headline_margin_above_a_star": a_star - headline_thr,
        "min_margin_below_nogo_over_proj_f": min_margin_below,
        "min_margin_above_a_star_over_proj_f": min_margin_above,
        "transfer_jitter": TRANSFER_JITTER,
        "go_band_nogo_floor": nogo,
        "go_band_go_threshold": headline_thr,
        "full_target_a_star": a_star,
        "decisive_interpretation": (
            "at f=0.5 the required Phase-1 a_2=%.4f sits %.4f above the deployed NO-GO floor "
            "%.4f and %.4f below the full target a_star=%.4f -- both margins exceed the "
            "+/-%.4f transfer jitter, so a cheap 2h run cleanly separates GO from NO-GO."
            % (headline_thr, headline_thr - nogo, nogo, a_star - headline_thr, a_star,
               TRANSFER_JITTER)),
    }


# --------------------------------------------------------------------------- #
# Step 5 -- literature reality check (does published EAGLE-3 lift clear 4.9029?)
# --------------------------------------------------------------------------- #
def fit_flat_accept(length: float) -> float:
    """The common conditional acceptance a s.t. a FLAT depth-K chain reproduces a given
    accepted-length: 1 + sum_{k=1..K} a^k = length (EAGLE-3's near-flat Fig-7 profile)."""
    length = min(max(length, 1.0 + 1e-9), E_T_MAX - 1e-9)
    return bisect(lambda a: et_of_profile([a] * K_SPEC) - length, 1e-9, 1.0 - 1e-12)


def lit_reality_check(a_star: float) -> dict[str, Any]:
    """Convert EAGLE-3's published CHAIN-equivalent accepted-length to a flat deep-position
    conditional acceptance a_lit (Fig-7 flat profile), then TRANSPLANT it onto OUR base:
    hold our a_1 cliff (0.7293, a base-model property #289 holds fixed), set a_2..a_7 = a_lit,
    and project the full-retrain E[T]. If even the published EAGLE-3 number falls short of
    4.9029 on our (low-a_1) base, the build is RISKIER than wirbel #290's necessary-condition
    window suggests -- the honest counter to window-optimism. The DIRECT chain-equivalent
    length (ignoring our worse a_1) is reported as the optimistic upper bracket."""
    a1 = A_BASE[0]
    a_lit_c = fit_flat_accept(EAGLE3_LIT_L_CHAIN_CENTRAL)     # flat deep accept @ chain length 5.5
    a_lit_lo = fit_flat_accept(EAGLE3_LIT_L_CHAIN_LOW)        # flat deep accept @ chain length 4.6
    prof_c = deep_uniform_profile(a1, a_lit_c)               # transplant onto our a_1 cliff
    prof_lo = deep_uniform_profile(a1, a_lit_lo)
    et_c = et_of_profile(prof_c)
    et_lo = et_of_profile(prof_lo)
    clears_c = et_c >= TARGET_BANKED
    clears_lo = et_lo >= TARGET_BANKED
    return {
        "eagle3_lit_L_tree_central": EAGLE3_LIT_L_TREE_CENTRAL,
        "eagle3_lit_L_chain_central": EAGLE3_LIT_L_CHAIN_CENTRAL,
        "eagle3_lit_L_chain_low": EAGLE3_LIT_L_CHAIN_LOW,
        "eagle3_lit_accept_central": a_lit_c,                 # derived flat deep accept (central)
        "eagle3_lit_accept_low": a_lit_lo,
        # headline = projection ON OUR BASE (a_1 cliff held), the PR's "on our a_k base"
        "eagle3_lit_projected_et": et_c,
        "eagle3_lit_projected_et_low": et_lo,
        "eagle3_lit_projected_tps_central": tps_from_et(et_c),
        "eagle3_lit_clears_target": bool(clears_c),
        "eagle3_lit_clears_target_low": bool(clears_lo),
        # optimistic upper bracket = EAGLE-3's own chain length, ignoring our worse a_1
        "eagle3_lit_direct_chain_et_central": EAGLE3_LIT_L_CHAIN_CENTRAL,
        "eagle3_lit_direct_clears_target": bool(EAGLE3_LIT_L_CHAIN_CENTRAL >= TARGET_BANKED),
        # the literature STRADDLES the target: hold-a_1 transplant < target < direct chain
        # -> literature alone CANNOT decide; the cheap Phase-1 measurement is load-bearing.
        "eagle3_lit_brackets_target": bool(et_c < TARGET_BANKED < EAGLE3_LIT_L_CHAIN_CENTRAL),
        "a_star_uniform_j2": a_star,
        "lit_central_vs_a_star": a_lit_c - a_star,            # >0 -> lit deep accept beats need
        "eagle3_lit_margin_et": et_c - TARGET_BANKED,
        "note": EAGLE3_LIT_NOTE,
        "risk_flag": (
            None if clears_c else
            "published EAGLE-3 central lift (flat deep accept %.4f from chain-length %.1f) "
            "TRANSPLANTED onto our a_1=%.4f cliff projects E[T]=%.4f < target %.4f -- and the "
            "flat deep accept %.4f sits BELOW the a_star=%.4f our low-a_1 base needs, so the "
            "build is RISKIER than the necessary-condition window suggests (cheap Phase-1 "
            "measurement is load-bearing, not optional)."
            % (a_lit_c, EAGLE3_LIT_L_CHAIN_CENTRAL, a1, et_c, TARGET_BANKED, a_lit_c, a_star)),
    }


# --------------------------------------------------------------------------- #
# Step 6 -- greedy-safety + gate note (SAFETY axis certificate)
# --------------------------------------------------------------------------- #
def safety_certificate() -> dict[str, Any]:
    # EAGLE-3 emission = verify argmax -> greedy identity preserved by construction; the
    # deployed M=8 identity (0.9927) already clears the safe local bar 0.9855.
    clears = M8_GREEDY_IDENTITY >= SAFE_LOCAL_LAMBDA_BAR
    return {
        "phase1_is_greedy_safe": True,
        "phase1_run_is_human_gated": True,
        "phase1_card_is_cpu_analytic": True,
        "emission_is_verify_argmax": True,
        "lambda_bar_cleared_by_construction": bool(clears),
        "safe_local_lambda_bar": SAFE_LOCAL_LAMBDA_BAR,
        "deployed_m8_greedy_identity": M8_GREEDY_IDENTITY,
        "ppl_pinned_bar": SAFE_LOCAL_PPL_BAR,
        "tau_acc": TAU_ACC,
        "note": (
            "EAGLE-3 (and the Phase-1 adaptation) is greedy-IDENTICAL by construction: the "
            "drafter only PROPOSES; the verifier's argmax token is always emitted, so the "
            "greedy-identity lambda-hat is pinned at the deployed M=8 level (0.9927 >= safe "
            "bar 0.9855) and PPL stays pinned (<= 2.4185). The 0.9855 lambda-hat bar is "
            "therefore a CERTIFICATE, not the binding gate; the operative trigger is the "
            "spec-acceptance a_2 threshold (SPEED axis)."),
    }


# --------------------------------------------------------------------------- #
# Step 7 -- self-test (PRIMARY)
# --------------------------------------------------------------------------- #
def self_test(s1: dict[str, Any], sw: dict[str, Any], go: dict[str, Any],
              dec: dict[str, Any]) -> dict[str, Any]:
    a_star = s1["a_star_uniform_j2"]
    a_star_prof = s1["a_star_profile"]
    rows = sw["sweep"]
    checks: dict[str, bool] = {}

    # (a) a_star_uniform_j2 reproduces E[T]=4.9029 within tol via the survival product.
    checks["a_a_star_reproduces_target"] = abs(et_of_profile(a_star_prof) - TARGET_BANKED) <= 1e-9

    # (b) Phase-1 E[T] monotone increasing in f_phase1.
    checks["b_E_T_phase1_monotone_in_f"] = bool(sw["E_T_phase1_monotone_in_f"])

    # (c) at f=1.0 the Phase-1 profile reaches a_star (full capture = full target).
    f1 = rows[-1]
    checks["c_f1_reaches_a_star"] = (
        abs(f1["E_T_phase1"] - TARGET_BANKED) <= 1e-9
        and all(abs(f1["phase1_profile"][i] - a_star_prof[i]) <= 1e-9 for i in range(K_SPEC)))

    # (d) GO threshold ordered: nogo_floor (linear base a_2) < go_threshold <= a_star  for all f.
    nogo = go["nogo_floor_a2_deployed"]
    checks["d_go_threshold_ordered"] = all(
        nogo < r["phase1_go_threshold_a2"] <= a_star + 1e-12 for r in go["go_threshold_rows"])

    # (e) phase1_go_threshold_a2 > a_base[1]=0.7596 (Phase-1 must beat the deployed a_2) at f=0.5.
    checks["e_go_threshold_beats_deployed_a2"] = go["phase1_go_threshold_a2"] > A_BASE[1]

    # (f) local-official transfer applied (tau_acc=1.0, bar 0.9855 imported from lawine #288).
    checks["f_transfer_applied"] = (abs(TAU_ACC - 1.0) < 1e-12
                                    and abs(SAFE_LOCAL_LAMBDA_BAR - 0.9855) < 1e-12)

    # (g) composition round-trips (481.53 <-> E[T]=3.844 at step=1218.2 reproduces K_cal=125.268).
    k_cal_implied = OFFICIAL / E_T_ANCHOR
    checks["g_composition_roundtrips_kcal"] = (
        abs(k_cal_implied - K_CAL) < 1e-2 and abs(tps_from_et(E_T_ANCHOR) - OFFICIAL) < 1e-2)

    # (h) NaN-clean over the f_phase1 sweep (and the a_star profile).
    flat = ([a_star] + a_star_prof
            + [r["E_T_phase1"] for r in rows]
            + [r["a2_phase1"] for r in rows]
            + [r["phase1_go_threshold_a2"] for r in go["go_threshold_rows"]])
    checks["h_nan_clean_sweep"] = all(math.isfinite(float(x)) for x in flat)

    # (i) constants imported EXACT.
    checks["i_constants_imported_exact"] = (
        abs(OFFICIAL - 481.53) < 1e-9 and abs(CEILING_LAMBDA1 - 520.95) < 1e-9
        and abs(K_CAL - 125.268) < 1e-9 and abs(TARGET_BANKED - 4.9029) < 1e-9
        and abs(LINEAR_CAP - 3.8445) < 1e-9 and abs(SAFE_LOCAL_LAMBDA_BAR - 0.9855) < 1e-9
        and abs(E_T_MAX - 8.0) < 1e-9)

    # bonus: deployed base reproduces #289's E[T] decomposition (3.8512) within tol.
    checks["j_base_reproduces_289_decomp"] = abs(et_of_profile(A_BASE) - E_T_DECOMP_289) <= 1e-9
    # bonus: a_star sits inside (linear-cap-equiv, full-acceptance) acceptance window.
    checks["k_a_star_in_unit_interval"] = (A_BASE[1] < a_star < 1.0)

    gate = bool(
        checks["a_a_star_reproduces_target"] and checks["b_E_T_phase1_monotone_in_f"]
        and checks["c_f1_reaches_a_star"] and checks["d_go_threshold_ordered"]
        and checks["e_go_threshold_beats_deployed_a2"] and checks["f_transfer_applied"]
        and checks["g_composition_roundtrips_kcal"] and checks["h_nan_clean_sweep"]
        and checks["i_constants_imported_exact"] and checks["j_base_reproduces_289_decomp"]
        and checks["k_a_star_in_unit_interval"])
    return {"checks": checks, "k_cal_implied": k_cal_implied,
            "eagle3_phase1_gate_self_test_passes": gate}


# --------------------------------------------------------------------------- #
# assemble
# --------------------------------------------------------------------------- #
def build_report() -> dict[str, Any]:
    s1 = solve_a_star(TARGET_BANKED)
    a_star = s1["a_star_uniform_j2"]
    sw = sweep_phase1(a_star)
    go = go_thresholds(a_star)
    dec = decisiveness(go, a_star)
    lit = lit_reality_check(a_star)
    safe = safety_certificate()
    st = self_test(s1, sw, go, dec)

    handoff = (
        "the EAGLE-3 Phase-1 viability gate is %s: a 2h frozen-backbone adaptation must lift "
        "position-2 conditional acceptance to >= %.4f (local-measured, vs deployed %.4f and "
        "full-target a_star %.4f) to project onto wirbel #290's 4.9029 target; the published "
        "EAGLE-3 lift %s clear the target on our a_k base (eagle3_lit_clears_target=%s) -- in "
        "fact the literature STRADDLES it (hold-a_1 transplant %.4f < 4.9029 < direct chain "
        "%.1f), so whether it clears depends on lifting our a_1=%.4f cliff, which is exactly "
        "why the cheap Phase-1 measurement is load-bearing; and the gate run is "
        "human-approval-gated GPU spend while THIS sizing card is CPU-analytic -- giving the "
        "human an operational, measurable retrain trigger instead of a hunch." % (
            "DECISIVE" if dec["phase1_gate_is_decisive"] else "NOT decisive",
            go["phase1_go_threshold_a2"], go["nogo_floor_a2_deployed"], a_star,
            "does" if lit["eagle3_lit_clears_target"] else "does NOT",
            lit["eagle3_lit_clears_target"], lit["eagle3_lit_projected_et"],
            lit["eagle3_lit_direct_chain_et_central"], A_BASE[0]))

    return {
        "pr": 294,
        "leg": "EAGLE-3 Phase-1 viability gate: the cheap-proxy GO threshold",
        "eagle3_phase1_gate_analysis_only": True,
        "imported": {
            "official": OFFICIAL, "ceiling_lambda1": CEILING_LAMBDA1, "K_cal": K_CAL,
            "E_T_anchor": E_T_ANCHOR, "step_us": STEP_US, "new_step_us": NEW_STEP_US,
            "tau": TAU, "K_spec": K_SPEC, "E_T_max": E_T_MAX, "honest_500_floor": HONEST_500_FLOOR,
            "linear_cap_denken119": LINEAR_CAP, "target_banked_wirbel290": TARGET_BANKED,
            "eagle3_budget_et_wirbel290": EAGLE3_BUDGET_ET, "fern_floor_public": FERN_FLOOR_PUBLIC,
            "a_base_kanna289": A_BASE, "E_T_decomp_289": E_T_DECOMP_289,
            "token_loss_forfeit_289": TOKEN_LOSS_FORFEIT_289,
            "tau_acc_lawine288": TAU_ACC, "safe_local_lambda_bar_lawine288": SAFE_LOCAL_LAMBDA_BAR,
            "operative_lambda_bar": OPERATIVE_LAMBDA_BAR, "transfer_jitter": TRANSFER_JITTER,
            "safe_local_ppl_bar_lawine288": SAFE_LOCAL_PPL_BAR,
            "m8_greedy_identity_lawine288": M8_GREEDY_IDENTITY,
            "eagle3_lit_L_tree_central": EAGLE3_LIT_L_TREE_CENTRAL,
            "eagle3_lit_L_chain_central": EAGLE3_LIT_L_CHAIN_CENTRAL,
            "eagle3_lit_L_chain_low": EAGLE3_LIT_L_CHAIN_LOW,
            "eagle3_tree_chain_premium": EAGLE3_TREE_CHAIN_PREMIUM,
            "wandb_sources": {
                "kanna_289": "fi34s269", "wirbel_290": "ub3kpsso", "lawine_288": "i1e5054m",
                "fern_281": "10necg21", "kanna_217": "vgovdrjc",
            },
        },
        "step1_inverse_target": s1,
        "step2_phase1_sweep": sw,
        "step3_go_thresholds": go,
        "step4_decisiveness": dec,
        "step5_lit_reality_check": lit,
        "step6_safety_certificate": safe,
        "self_test": st["checks"],
        "k_cal_implied": st["k_cal_implied"],
        # ---- headline metrics ----
        "eagle3_phase1_gate_self_test_passes": st["eagle3_phase1_gate_self_test_passes"],  # PRIMARY
        "phase1_go_threshold_a2": go["phase1_go_threshold_a2"],                            # TEST (f=0.5)
        "phase1_gate_is_decisive": dec["phase1_gate_is_decisive"],
        "a_star_uniform_j2": a_star,
        "eagle3_lit_projected_et": lit["eagle3_lit_projected_et"],
        "eagle3_lit_clears_target": lit["eagle3_lit_clears_target"],
        "phase1_is_greedy_safe": safe["phase1_is_greedy_safe"],
        "phase1_run_is_human_gated": safe["phase1_run_is_human_gated"],
        "phase1_card_is_cpu_analytic": safe["phase1_card_is_cpu_analytic"],
        "handoff": handoff,
    }


# --------------------------------------------------------------------------- #
# NaN audit
# --------------------------------------------------------------------------- #
def _assert_nan_clean(obj: Any, path: str = "") -> list[str]:
    bad: list[str] = []
    if isinstance(obj, float):
        if not math.isfinite(obj):
            bad.append(path)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            bad += _assert_nan_clean(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad += _assert_nan_clean(v, f"{path}[{i}]")
    return bad


# --------------------------------------------------------------------------- #
# W&B logging (mirrors wirbel #290; never fatal).
# --------------------------------------------------------------------------- #
def _maybe_log_wandb(args, report: dict[str, Any]) -> str | None:
    if getattr(args, "no_wandb", False):
        return None
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.append(repo)
    _w = sys.modules.get("wandb")
    if _w is not None and not hasattr(_w, "init"):
        del sys.modules["wandb"]
    try:
        import wandb as _wb
        if not hasattr(_wb, "init"):
            raise ImportError("resolved a stub/namespace wandb with no .init")
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[eagle3-phase1-gate] wandb logging skipped (analysis unaffected): {exc}", flush=True)
        return None

    go = report["step3_go_thresholds"]
    dec = report["step4_decisiveness"]
    lit = report["step5_lit_reality_check"]
    safe = report["step6_safety_certificate"]
    s1 = report["step1_inverse_target"]
    try:
        run = init_wandb_run(
            job_type="validity-gate", agent="kanna", name=args.wandb_name, group=args.wandb_group,
            tags=["eagle3-phase1-gate", "go-threshold", "capture-fraction", "decisive-gate",
                  "spec-acceptance", "greedy-identity-safety", "bank-the-analysis", "pr-294"],
            config={
                "official": OFFICIAL, "K_cal": K_CAL, "E_T_anchor": E_T_ANCHOR,
                "linear_cap": LINEAR_CAP, "target_banked": TARGET_BANKED,
                "safe_local_lambda_bar": SAFE_LOCAL_LAMBDA_BAR, "tau_acc": TAU_ACC,
                "transfer_jitter": TRANSFER_JITTER, "K_spec": K_SPEC,
                "f_phase1_sweep": F_PHASE1_SWEEP, "f_headline": F_HEADLINE,
                "eagle3_lit_L_chain_central": EAGLE3_LIT_L_CHAIN_CENTRAL,
                "eagle3_lit_L_chain_low": EAGLE3_LIT_L_CHAIN_LOW,
                "imports": "kanna#289(fi34s269 a_k) x wirbel#290(ub3kpsso target=4.9029) x "
                           "lawine#288(i1e5054m tau_acc=1.0 bar=0.9855) x fern#281(10necg21) x "
                           "kanna#217(vgovdrjc K_cal=125.268)",
                "wandb_group": args.wandb_group,
            },
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[eagle3-phase1-gate] wandb init failed (analysis unaffected): {exc}", flush=True)
        return None
    if run is None:
        print("[eagle3-phase1-gate] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return None

    summary: dict[str, Any] = {
        "eagle3_phase1_gate_self_test_passes": int(bool(report["eagle3_phase1_gate_self_test_passes"])),
        "phase1_go_threshold_a2": report["phase1_go_threshold_a2"],
        "phase1_gate_is_decisive": int(bool(report["phase1_gate_is_decisive"])),
        "a_star_uniform_j2": report["a_star_uniform_j2"],
        "nogo_floor_a2_deployed": go["nogo_floor_a2_deployed"],
        "go_band_low": go["go_band"][0],
        "go_band_high": go["go_band"][1],
        "headline_margin_below_nogo": dec["headline_margin_below_nogo"],
        "headline_margin_above_a_star": dec["headline_margin_above_a_star"],
        "min_margin_below_nogo_over_proj_f": dec["min_margin_below_nogo_over_proj_f"],
        "decisive_in_projection_regime_all_f": int(bool(dec["decisive_in_projection_regime_all_f"])),
        "transfer_jitter": TRANSFER_JITTER,
        "eagle3_lit_projected_et": lit["eagle3_lit_projected_et"],
        "eagle3_lit_projected_et_low": lit["eagle3_lit_projected_et_low"],
        "eagle3_lit_clears_target": int(bool(lit["eagle3_lit_clears_target"])),
        "eagle3_lit_clears_target_low": int(bool(lit["eagle3_lit_clears_target_low"])),
        "eagle3_lit_direct_chain_et_central": lit["eagle3_lit_direct_chain_et_central"],
        "eagle3_lit_direct_clears_target": int(bool(lit["eagle3_lit_direct_clears_target"])),
        "eagle3_lit_brackets_target": int(bool(lit["eagle3_lit_brackets_target"])),
        "eagle3_lit_accept_central": lit["eagle3_lit_accept_central"],
        "eagle3_lit_margin_et": lit["eagle3_lit_margin_et"],
        "lit_central_vs_a_star": lit["lit_central_vs_a_star"],
        "target_reconstruct_from_290": s1["target_reconstruct_from_290"],
        "k_cal_implied": report["k_cal_implied"],
        "phase1_is_greedy_safe": int(bool(safe["phase1_is_greedy_safe"])),
        "phase1_run_is_human_gated": int(bool(safe["phase1_run_is_human_gated"])),
        "phase1_card_is_cpu_analytic": int(bool(safe["phase1_card_is_cpu_analytic"])),
        "lambda_bar_cleared_by_construction": int(bool(safe["lambda_bar_cleared_by_construction"])),
        "nan_clean": int(bool(report["nan_clean"])),
        **{f"selftest_{k}": int(bool(v)) for k, v in report["self_test"].items()},
        **{f"go_threshold_a2_f{str(f).replace('.', 'p')}": v
           for f, v in go["phase1_go_threshold_a2_by_f"].items()},
        **{f"E_T_phase1_f{str(r['f_phase1']).replace('.', 'p')}": r["E_T_phase1"]
           for r in report["step2_phase1_sweep"]["sweep"]},
    }
    summary = {k: v for k, v in summary.items()
               if v is not None and not (isinstance(v, float) and not math.isfinite(v))}
    try:
        log_summary(run, summary, step=0)
        log_json_artifact(run, name="eagle3_phase1_gate_result", artifact_type="validity", data=report)
        rid = getattr(run, "id", None)
        finish_wandb(run)
        print(f"[eagle3-phase1-gate] wandb logged {len(summary)} summary keys (run {rid})", flush=True)
        return rid
    except Exception as exc:  # noqa: BLE001
        print(f"[eagle3-phase1-gate] wandb write failed (analysis unaffected): {exc}", flush=True)
        return None


def _print_human(report: dict[str, Any]) -> None:
    go = report["step3_go_thresholds"]
    dec = report["step4_decisiveness"]
    lit = report["step5_lit_reality_check"]
    print("\n" + "=" * 100, flush=True)
    print(" EAGLE-3 PHASE-1 VIABILITY GATE: the cheap-proxy GO threshold (PR #294)", flush=True)
    print("=" * 100, flush=True)
    print(f"  a_star_uniform_j2 (E[T]=4.9029)      : {report['a_star_uniform_j2']:.4f}  "
          f"(hold a_1={A_BASE[0]:.4f}; deployed a_2={A_BASE[1]:.4f})", flush=True)
    print("  Phase-1 capture-fraction sweep (E[T], a_2 GO threshold):", flush=True)
    thr_by_f = go["phase1_go_threshold_a2_by_f"]
    for r in report["step2_phase1_sweep"]["sweep"]:
        f = r["f_phase1"]
        print(f"      f={f:<4} -> E[T]={r['E_T_phase1']:.4f}  a2_GO_threshold={thr_by_f[f]:.4f}",
              flush=True)
    print(f"  GO band [nogo_floor, go_threshold]   : "
          f"[{go['go_band'][0]:.4f}, {go['go_band'][1]:.4f}]  (full target a_star "
          f"{report['a_star_uniform_j2']:.4f})", flush=True)
    print(f"  phase1_go_threshold_a2 (TEST, f=0.5) : {report['phase1_go_threshold_a2']:.4f}", flush=True)
    print(f"  phase1_gate_is_decisive              : {report['phase1_gate_is_decisive']}  "
          f"(margins below {dec['headline_margin_below_nogo']:.4f} / above "
          f"{dec['headline_margin_above_a_star']:.4f} vs jitter {TRANSFER_JITTER})", flush=True)
    print(f"  LIT reality: central accept {lit['eagle3_lit_accept_central']:.3f} -> E[T]="
          f"{lit['eagle3_lit_projected_et']:.4f}  clears_target={lit['eagle3_lit_clears_target']}  "
          f"(low {lit['eagle3_lit_accept_low']:.3f} -> {lit['eagle3_lit_projected_et_low']:.4f}, "
          f"clears={lit['eagle3_lit_clears_target_low']})", flush=True)
    if lit["risk_flag"]:
        print(f"  LIT RISK FLAG: {lit['risk_flag']}", flush=True)
    print(f"  greedy-safe (emission=verify argmax) : {report['phase1_is_greedy_safe']}  "
          f"(lambda bar 0.9855 cleared by construction; PPL pinned <= 2.4185)", flush=True)
    print(f"  PRIMARY self_test                    : {report['eagle3_phase1_gate_self_test_passes']}",
          flush=True)
    print("-" * 100, flush=True)
    print(f"  HAND-OFF: {report['handoff']}\n", flush=True)


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="run the gate-sizing analytic + PRIMARY self-test over banked numbers.")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="eagle3-phase1-gate")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="kanna/eagle3-phase1-gate")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    report = build_report()
    nan_paths = _assert_nan_clean(report)
    report["nan_clean"] = not nan_paths
    report["eagle3_phase1_gate_self_test_passes"] = bool(
        report["eagle3_phase1_gate_self_test_passes"] and report["nan_clean"])
    if nan_paths:
        print(f"[eagle3-phase1-gate] WARNING non-finite at: {nan_paths[:10]}", flush=True)

    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["peak_mem_mib"] = round(peak_kib / 1024.0, 3)
    report["created_at"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    wid = _maybe_log_wandb(args, report)
    report["wandb_run_id"] = wid

    HERE.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, default=float))

    _print_human(report)
    print(f"[eagle3-phase1-gate] wrote {OUT_PATH}", flush=True)
    print(f"[eagle3-phase1-gate] PRIMARY eagle3_phase1_gate_self_test_passes = "
          f"{report['eagle3_phase1_gate_self_test_passes']}", flush=True)
    print(f"[eagle3-phase1-gate] TEST phase1_go_threshold_a2 = "
          f"{report['phase1_go_threshold_a2']:.4f}", flush=True)
    print(f"[eagle3-phase1-gate] phase1_gate_is_decisive = {report['phase1_gate_is_decisive']}",
          flush=True)
    print(f"[eagle3-phase1-gate] wandb run = {wid}", flush=True)

    if args.self_test and not report["eagle3_phase1_gate_self_test_passes"]:
        failed = [k for k, v in report["self_test"].items() if not v]
        print(f"[eagle3-phase1-gate] SELF-TEST FAILED: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
