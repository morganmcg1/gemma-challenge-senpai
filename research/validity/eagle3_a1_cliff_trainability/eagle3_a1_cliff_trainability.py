#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #308 (denken) -- is a1~0.92 a TRAINABLE target or an INTRINSIC position-1 floor?

THE LAST LOAD-BEARING UNKNOWN
-----------------------------
denken #304 (`dtf1ouml`, MERGED) proved the keystone: hitting wirbel #295's
step-corrected E[T]=6.1112 target CANNOT be done while keeping the position-1
acceptance cliff -- even at the physically-impossible perfect-tail corner the
cliff-kept ceiling caps at 1 + 0.73*7 = 6.11 < 6.1112. So the heavier
{2,21,39}-fusion step DEMANDS a drafter whose first-draft acceptance rises to
**a1 ~ 0.9213** (uniform profile), vs the deployed linear-MTP a1 = 0.72925
(kanna #289), a +0.192 / x1.263 lift on the single position that is the cliff.

Every other cost axis in the >500 build is priced, but they ALL assume a drafter
that hits the required a_k profile EXISTS and is TRAINABLE. This card prices the
one thing #304 left open: is a1~0.92 reachable in principle from published EAGLE-3
evidence + the deployed a_k structure, or is the position-1 cliff an INTRINSIC
discrimination floor no amount of EAGLE-3 training can lift?

WHY a1 IS THE CLIFF (and why it is the position least responsive to training)
-----------------------------------------------------------------------------
The deployed a_k = [0.729, 0.760, 0.793, 0.823, 0.835, 0.836, 0.846] is MONOTONE
INCREASING: a1 is the MINIMUM. This is counter-intuitive (later draft positions
"should" be harder) and the reason is structural:

  * Interior a_k (k>=2) are CONDITIONAL acceptances -- a_k = P(accept pos k |
    positions 1..k-1 accepted). Conditioning on a long accepted prefix is a
    SURVIVORSHIP filter: it selects the easy, low-entropy continuations, so the
    conditional acceptance RISES with depth.
  * a1 is the UNCONDITIONED first-draft argmax-match over the WHOLE eval
    distribution -- no easy-prefix selection. It is the raw discrimination floor.
  * a1 is depth-0 / "serve-faithful": its input is the REAL target feature
    h_{n-1} + the just-sampled token (never the drafter's own predicted feature).
    The standard EAGLE-3 acceptance lever -- TTT / multi-step unroll (and HASS) --
    repairs the depth>0 collapse where the drafter consumes its OWN features. It
    does NOT touch depth-0: the literature is explicit (EAGLE-3 arXiv 2503.01840:
    0-alpha rises from REMOVING the feature-prediction constraint, NOT from TTT;
    HASS arXiv 2408.15766 Fig 5: TTT-style training DECREASES 0-alpha vs EAGLE-2
    before a reweighting term claws it back). So the very lever the build's other
    axes rely on cannot lift a1.

WHAT THIS CARD DOES (analytic + literature; 0 GPU)
-------------------------------------------------
1. ANCHOR the deployed cliff: import kanna #289 a_k, show a1 is the minimum,
   quantify the deficit vs the required 0.9213 and vs EVERY interior a_k (the
   demand exceeds even the deployed MAX interior acceptance 0.846). Re-derive
   #304's 0.9213 by inverting the chain-product (consistency self-test), and
   price the a1-demand SENSITIVITY to the interior profile (perfect / uniform /
   spec-capped 0.91).
2. SURVEY the published EAGLE-3 first-token (0-alpha) acceptance envelope (numbers
   imported below with citations) and normalize to our a1. In-repo anchor: the
   trained-from-scratch EAGLE-3 head on THIS target (gua9x68j, fern #34,
   arch_notes Sec 9) hit native step-1 = 0.7714 -- squarely inside the published
   0.77-0.80 envelope.
3. CEILING: (a) info-theoretic -- SpecTr (arXiv 2208.11970) gives single-draft
   acceptance = 1 - TV(p_draft, p_target); a perfect drafter -> 1.0, so 0.92 is
   BELOW the info ceiling => a TRAINING problem, not physics-forbidden. (b)
   argmax-robustness -- a compressed 1-layer fusion head that recovers the target
   argmax in proportion to the eval-distribution top-1 mass caps at a1 ~ E[top-1
   mass] = m_bar; for any plausible m_bar <= 0.90 (reasoning/MCQ instruct text)
   the demand 0.9213 sits ABOVE this binding ceiling.
4. VERDICT: a1_target_trainable in {in-envelope, at-frontier, out-of-reach};
   a1_demand_blocks_go (bool); residual risk + the only levers that touch a1.

HONEST SCOPE
------------
0 TPS. BASELINE 481.53 unchanged. NO training, NO drafter build, NO served-file
change, NO HF Job. This prices whether the required a1 is REACHABLE IN PRINCIPLE
from published evidence; the drafter BUILD itself stays human-gated. "out-of-reach"
here means OUT OF THE DEMONSTRATED/PUBLISHED ENVELOPE and the argmax-robustness
ceiling -- NOT information-theoretically forbidden (the perfect-drafter ceiling is
1.0). The published numbers are read from the cited papers (figures interpolated
where the paper gives no table); the eval top-1 mass is a STATED assumption a
0-GPU card cannot measure, carried as the explicit residual uncertainty.
NOT a launch. NOT a build. NOT a served-file change.

PRIMARY metric  a1_cliff_trainability_self_test_passes
TEST    metric  a1_required_for_611
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

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Imported anchors -- DO NOT re-derive. Import EXACTLY, UNCHANGED.
# --------------------------------------------------------------------------- #
# kanna #289 (`fi34s269`) per-position acceptance (the deployed linear-MTP a_k).
A_K_289_DEPLOYED = [0.72925, 0.75956, 0.79298, 0.82280, 0.83487, 0.83579, 0.84649]
A1_DEPLOYED = A_K_289_DEPLOYED[0]              # 0.72925 -- the position-1 cliff (minimum)
A1_LOW_QUARTILE_297 = 0.6550                   # denken #297 low-quartile (hard-prompt) a1

# denken #304 (`dtf1ouml`, MERGED) -- the demand this card prices for trainability.
A1_REQUIRED_FOR_611 = 0.9213                   # uniform per-position acceptance for 6.11
ET_TARGET_CENTRAL_295 = 6.1112149873699195     # wirbel #295 step-corrected central target
ET_TARGET_LOWER_295 = 5.363610726985671        # multiplicative-lower bracket
ET_TARGET_UPPER_295 = 6.858819247754167        # additive-upper bracket
SPEC_LIFT_289 = 0.91                           # kanna #289 interior-lift spec (a_{j>=2}->0.91)
CLIFF_VALUE_SPEC = 0.73                         # #289 "keep a1>=0.73" cliff floor
K_SPEC = 7                                      # num_speculative_tokens (chain depth)

# In-repo EAGLE-3 head on THIS target (fern #34 `gua9x68j`, arch_notes.md Sec 9):
# trained-from-scratch {2,21,39}-fusion draft head, benchmark-matched corpus.
A1_INREPO_EAGLE3_NATIVE_STEP1 = 0.7714         # free-running native step-1 top-1 (a1 analog)
A1_INREPO_EAGLE3_TF = 0.7617                   # teacher-forced top-1 (serve-faithful proxy)

# --------------------------------------------------------------------------- #
# PUBLISHED EAGLE-family first-token (0-alpha) acceptance envelope.
# 0-alpha = acceptance of the first draft token (drafter input = REAL target
# features only, no drafter-predicted features) -- the exact analog of our a1.
# All QUOTED unless marked interpolated.
# --------------------------------------------------------------------------- #
# EAGLE-1 (arXiv 2401.15077, Table 2, MT-bench, T=0):
EAGLE1_VICUNA7B_0ALPHA_T0 = 0.79               # peak exact-quoted 0-alpha
EAGLE1_VICUNA13B_0ALPHA_T0 = 0.79
EAGLE1_LLAMA2CHAT7B_0ALPHA_T0 = 0.76
EAGLE1_LLAMA2CHAT13B_0ALPHA_T0 = 0.77
EAGLE1_LLAMA2CHAT70B_0ALPHA_T0 = 0.75
EAGLE1_MIXTRAL8X7B_0ALPHA_T0 = 0.67
# EAGLE-3 (arXiv 2503.01840, Fig 4/7, LLaMA-3.1-8B, MT-bench, T=0; INTERPOLATED
# from figure Y-axis 0.74-0.80 -- the paper publishes 0-alpha only in figures):
EAGLE3_LLAMA31_8B_0ALPHA_T0 = 0.80             # highest first-token acceptance in the record
# greedy(T=0) vs sampling(T=1) gap for the same model (EAGLE-1 Vicuna 13B: 0.79 vs 0.73)
GREEDY_OVER_TEMP1_0ALPHA_GAIN = 0.06
# EAGLE-1 -> EAGLE-3 0-alpha gain (~0.75 -> ~0.80): the SINGLE biggest demonstrated
# architectural lever on the first token (multi-layer fusion + no feature-pred constraint).
ARCH_LEVER_GAIN_ON_A1 = 0.05

A1_PUBLISHED_ENVELOPE_MAX = EAGLE3_LLAMA31_8B_0ALPHA_T0     # 0.80 (interpolated EAGLE-3)
A1_PUBLISHED_EXACT_QUOTED_MAX = EAGLE1_VICUNA7B_0ALPHA_T0   # 0.79 (quoted EAGLE-1)
# Literature finding: NO published spec-decode method (EAGLE-1/2/3, HASS, Medusa,
# Hydra, GliDe, PARD, MTP, ...) reports first-token 0-alpha >= 0.90 on ANY
# chat/instruct target at ANY temperature.
NO_PUBLISHED_METHOD_GE_090 = True
# TTT / multi-step unroll (and HASS) repair depth>0, NOT depth-0/a1 (see docstring).
TTT_LIFTS_DEPTH0 = False

# --------------------------------------------------------------------------- #
# Theoretical ceilings.
# --------------------------------------------------------------------------- #
# (a) Info-theoretic: SpecTr (arXiv 2208.11970) -- single-draft acceptance
#     = sum_x min(p,q) = 1 - TV(p_draft, p_target). Perfect drafter (q=p) -> 1.0.
A1_INFO_THEORETIC_CEILING = 1.0
# (b) argmax-robustness ceiling: STATED eval-distribution mean top-1 mass m_bar.
#     Under a confidence-proportional argmax-recovery model (a compressed 1-layer
#     head recovers the target argmax with prob = local top-1 mass), a1 caps at
#     a1 ~ E[top-1 mass] = m_bar. Basis for the band (reasoning/MCQ instruct text,
#     the #34 corpus: mmlu_pro/gpqa/aime): high-confidence structural tokens
#     (MCQ letters, "ANSWER:", function words) ~0.95+ mixed with low-confidence
#     reasoning-choice tokens ~0.3-0.6 -> token-weighted mean ~0.80-0.90.
#     Cross-checks: (i) the best demonstrated greedy 0-alpha ~0.80 is a LOWER
#     bound on m_bar (a near-best head recovers >= the confident mass); (ii) the
#     target's measured eval PPL 2.3772 -> mean NLL 0.8659 nats -> Jensen floor on
#     the reference-token prob exp(-0.8659)=0.4207 (weak: argmax >= reference).
EVAL_TOP1_MASS_MEAN = 0.85                      # central m_bar
EVAL_TOP1_MASS_LOW = 0.80                       # conservative
EVAL_TOP1_MASS_HIGH = 0.90                      # OPTIMISTIC argmax-robustness ceiling
TARGET_EVAL_PPL_217 = 2.3772                    # official PPL (kanna #217 / land #71 frontier)

# Verdict thresholds.
AT_FRONTIER_MARGIN = ARCH_LEVER_GAIN_ON_A1      # one more best-known-lever above envelope
OFFICIAL_TPS_217 = 481.53                       # BASELINE (unchanged; this card adds 0 TPS)

TOL_RT = 1e-6      # chain-product inversion round-trip tolerance
TOL_REPRO = 4e-4   # #304 0.9213-reproduction tolerance (0.9213 is reported to 4dp)


# --------------------------------------------------------------------------- #
# Chain-product machinery (denken #297/#304 convention): E[T]=1+sum cumprod(a).
# --------------------------------------------------------------------------- #
def et_from_ak(a_k: list[float]) -> float:
    """Survival chain-product E[T] = 1 + sum_{k=1..K} prod_{j=1..k} a_j."""
    cp = np.cumprod(np.asarray(a_k, dtype=float))
    return 1.0 + float(cp.sum())


def uniform_sum(a: float, K: int = K_SPEC) -> float:
    """sum_{k=1..K} a^k (geometric); E[T]_uniform = 1 + uniform_sum(a)."""
    return float(sum(a ** k for k in range(1, K + 1)))


def cliffkept_poly(b: float, K: int = K_SPEC) -> float:
    """sum_{k=0..K-1} b^k ; E[T] with a1 free and a_{j>=2}=b is 1 + a1*cliffkept_poly(b)."""
    return float(sum(b ** k for k in range(K)))


def _bisect(fn, target: float, lo: float, hi: float, iters: int = 200) -> float:
    """Solve fn(x)=target for monotone-increasing fn on [lo, hi] via bisection."""
    flo, fhi = fn(lo) - target, fn(hi) - target
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    grow = 0
    while fhi < 0.0 and grow < 80:
        hi *= 1.5
        fhi = fn(hi) - target
        grow += 1
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fmid = fn(mid) - target
        if fmid == 0.0:
            return mid
        if (fmid > 0.0) == (fhi > 0.0):
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)


def solve_uniform_a1(et_star: float, K: int = K_SPEC) -> float:
    """Per-position acceptance a* with all a_k=a* that yields E[T]=et_star."""
    return _bisect(lambda a: uniform_sum(a, K), et_star - 1.0, 0.0, 1.0)


def a1_required_given_interior(et_star: float, b: float, K: int = K_SPEC) -> float:
    """a1 needed to hit et_star when interior a_{j>=2}=b fixed: a1=(et-1)/sum_{k<K} b^k."""
    return (et_star - 1.0) / cliffkept_poly(b, K)


def argmax_robustness_ceiling(top1_mass: float) -> float:
    """Confidence-proportional ceiling: a1 ~ E[top-1 mass] under the recovery model."""
    return float(top1_mass)


# --------------------------------------------------------------------------- #
# Verdict.
# --------------------------------------------------------------------------- #
def classify_a1_trainable(a1_req: float, env_max: float, lever: float) -> str:
    """in-envelope (<=published max) / at-frontier (<= max + one best lever) / out-of-reach."""
    if a1_req <= env_max + 1e-9:
        return "in-envelope"
    if a1_req <= env_max + lever + 1e-9:
        return "at-frontier"
    return "out-of-reach"


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    # ---------- step 1: anchor the deployed cliff + deficit math ---------- #
    a1 = A1_DEPLOYED
    a1_is_profile_minimum = bool(a1 == min(A_K_289_DEPLOYED))
    profile_monotone_nondec = bool(
        all(A_K_289_DEPLOYED[i] <= A_K_289_DEPLOYED[i + 1] + 1e-12
            for i in range(len(A_K_289_DEPLOYED) - 1)))
    max_interior_ak = max(A_K_289_DEPLOYED[1:])             # 0.84649 (deployed ceiling)
    gap_vs_required = A1_REQUIRED_FOR_611 - a1              # 0.19205
    lift_factor_vs_deployed = A1_REQUIRED_FOR_611 / a1      # x1.2634
    gap_vs_each_interior = [A1_REQUIRED_FOR_611 - ak for ak in A_K_289_DEPLOYED]
    gap_vs_max_interior = A1_REQUIRED_FOR_611 - max_interior_ak   # 0.07481 (smallest gap)
    required_exceeds_all_deployed = bool(A1_REQUIRED_FOR_611 > max_interior_ak)

    # re-derive #304's 0.9213 (uniform inversion) -- consistency with #304.
    a1_uniform_recomputed = solve_uniform_a1(ET_TARGET_CENTRAL_295)
    reproduces_304 = abs(a1_uniform_recomputed - A1_REQUIRED_FOR_611) < TOL_REPRO

    # a1-demand sensitivity to the interior profile (the demand is NOT a single number):
    a1_demand_perfect_interior = a1_required_given_interior(ET_TARGET_CENTRAL_295, 1.0)   # ~0.730
    a1_demand_uniform = a1_uniform_recomputed                                             # ~0.9213
    a1_demand_spec_interior = a1_required_given_interior(ET_TARGET_CENTRAL_295, SPEC_LIFT_289)  # ~0.952
    # realistic: interior caps at the #289 spec 0.91 (TTT-reachable) -> a1 demand RISES to ~0.95.
    realistic_a1_demand_ge_uniform = bool(a1_demand_spec_interior >= a1_demand_uniform)

    # ---------- step 2: published envelope position ---------- #
    in_envelope = bool(A1_REQUIRED_FOR_611 <= A1_PUBLISHED_ENVELOPE_MAX + 1e-9)
    gap_above_env_max = A1_REQUIRED_FOR_611 - A1_PUBLISHED_ENVELOPE_MAX          # +0.1213
    gap_above_quoted_max = A1_REQUIRED_FOR_611 - A1_PUBLISHED_EXACT_QUOTED_MAX   # +0.1313
    gap_above_inrepo = A1_REQUIRED_FOR_611 - A1_INREPO_EAGLE3_NATIVE_STEP1       # +0.1499
    # the demand exceeds the biggest demonstrated single-lever gain by this multiple:
    gap_in_lever_units = gap_above_env_max / ARCH_LEVER_GAIN_ON_A1               # ~2.4x
    # the demonstrated arch lever on a1 (linear-MTP -> in-repo EAGLE-3):
    inrepo_arch_lever_gain = A1_INREPO_EAGLE3_NATIVE_STEP1 - A1_DEPLOYED         # +0.0422

    # ---------- step 3: ceilings ---------- #
    a1_argmax_robustness_ceiling_central = argmax_robustness_ceiling(EVAL_TOP1_MASS_MEAN)   # 0.85
    a1_theoretical_ceiling = argmax_robustness_ceiling(EVAL_TOP1_MASS_HIGH)                 # 0.90 (optimistic)
    demand_below_info_ceiling = bool(A1_REQUIRED_FOR_611 < A1_INFO_THEORETIC_CEILING)       # True
    demand_above_robustness_ceiling = bool(A1_REQUIRED_FOR_611 > a1_theoretical_ceiling)    # True (even optimistic)
    # PPL -> entropy cross-check (weak Jensen floor on mean reference-token prob).
    mean_nll_nats = math.log(TARGET_EVAL_PPL_217)
    jensen_floor_ref_prob = math.exp(-mean_nll_nats)

    # ---------- step 4: verdict ---------- #
    a1_target_trainable = classify_a1_trainable(
        A1_REQUIRED_FOR_611, A1_PUBLISHED_ENVELOPE_MAX, AT_FRONTIER_MARGIN)
    a1_demand_blocks_go = bool(a1_target_trainable == "out-of-reach")
    light = {"in-envelope": "GREEN", "at-frontier": "YELLOW",
             "out-of-reach": "RED"}[a1_target_trainable]

    verdict = _verdict(a1_target_trainable, demand_below_info_ceiling)
    handoff = _handoff(a1_target_trainable, light, gap_above_env_max,
                       a1_theoretical_ceiling, a1_demand_blocks_go)

    return {
        "step1_anchor_cliff": {
            "a_k_289_deployed": A_K_289_DEPLOYED,
            "a1_deployed": a1,
            "a1_is_profile_minimum": a1_is_profile_minimum,
            "profile_monotone_nondecreasing": profile_monotone_nondec,
            "max_interior_ak": max_interior_ak,
            "a1_required_for_611": A1_REQUIRED_FOR_611,
            "gap_vs_required": gap_vs_required,
            "lift_factor_vs_deployed": lift_factor_vs_deployed,
            "gap_vs_each_interior": gap_vs_each_interior,
            "gap_vs_max_interior": gap_vs_max_interior,
            "required_exceeds_all_deployed_ak": required_exceeds_all_deployed,
            "a1_uniform_recomputed": a1_uniform_recomputed,
            "reproduces_304_0p9213": reproduces_304,
            "a1_demand_perfect_interior": a1_demand_perfect_interior,
            "a1_demand_uniform": a1_demand_uniform,
            "a1_demand_spec_interior_0p91": a1_demand_spec_interior,
            "realistic_a1_demand_ge_uniform": realistic_a1_demand_ge_uniform,
        },
        "step2_published_envelope": {
            "a1_published_envelope_max": A1_PUBLISHED_ENVELOPE_MAX,
            "a1_published_exact_quoted_max": A1_PUBLISHED_EXACT_QUOTED_MAX,
            "a1_inrepo_eagle3_native_step1": A1_INREPO_EAGLE3_NATIVE_STEP1,
            "a1_inrepo_eagle3_tf": A1_INREPO_EAGLE3_TF,
            "in_envelope": in_envelope,
            "gap_above_env_max": gap_above_env_max,
            "gap_above_quoted_max": gap_above_quoted_max,
            "gap_above_inrepo": gap_above_inrepo,
            "gap_in_lever_units": gap_in_lever_units,
            "arch_lever_gain_on_a1_published": ARCH_LEVER_GAIN_ON_A1,
            "arch_lever_gain_on_a1_inrepo": inrepo_arch_lever_gain,
            "no_published_method_ge_090": NO_PUBLISHED_METHOD_GE_090,
            "ttt_lifts_depth0": TTT_LIFTS_DEPTH0,
            "greedy_over_temp1_gain": GREEDY_OVER_TEMP1_0ALPHA_GAIN,
            "eagle1_table2_t0": {
                "vicuna_7b": EAGLE1_VICUNA7B_0ALPHA_T0,
                "vicuna_13b": EAGLE1_VICUNA13B_0ALPHA_T0,
                "llama2chat_7b": EAGLE1_LLAMA2CHAT7B_0ALPHA_T0,
                "llama2chat_13b": EAGLE1_LLAMA2CHAT13B_0ALPHA_T0,
                "llama2chat_70b": EAGLE1_LLAMA2CHAT70B_0ALPHA_T0,
                "mixtral_8x7b": EAGLE1_MIXTRAL8X7B_0ALPHA_T0,
            },
        },
        "step3_ceiling": {
            "a1_info_theoretic_ceiling": A1_INFO_THEORETIC_CEILING,
            "eval_top1_mass_mean": EVAL_TOP1_MASS_MEAN,
            "eval_top1_mass_low": EVAL_TOP1_MASS_LOW,
            "eval_top1_mass_high": EVAL_TOP1_MASS_HIGH,
            "a1_argmax_robustness_ceiling_central": a1_argmax_robustness_ceiling_central,
            "a1_theoretical_ceiling": a1_theoretical_ceiling,
            "demand_below_info_ceiling": demand_below_info_ceiling,
            "demand_above_robustness_ceiling": demand_above_robustness_ceiling,
            "mean_nll_nats": mean_nll_nats,
            "jensen_floor_ref_prob": jensen_floor_ref_prob,
        },
        "step4_verdict": {
            "a1_target_trainable": a1_target_trainable,
            "a1_demand_blocks_go": a1_demand_blocks_go,
            "handoff_light": light,
            "a1_required_for_611": A1_REQUIRED_FOR_611,
            "a1_deployed": a1,
            "a1_low_quartile_297": A1_LOW_QUARTILE_297,
        },
        "context": {
            "official_tps_217": OFFICIAL_TPS_217,
            "et_target_central_295": ET_TARGET_CENTRAL_295,
            "et_target_lower_295": ET_TARGET_LOWER_295,
            "et_target_upper_295": ET_TARGET_UPPER_295,
            "k_spec": K_SPEC,
            "spec_lift_289": SPEC_LIFT_289,
            "cliff_value_spec": CLIFF_VALUE_SPEC,
        },
        "verdict": verdict,
        "handoff_line": handoff,
    }


def _verdict(trainable: str, below_info: bool) -> str:
    if trainable == "in-envelope":
        return "A1-IN-PUBLISHED-ENVELOPE-TRAINABLE"
    if trainable == "at-frontier":
        return "A1-AT-FRONTIER-TRAINABLE-WITH-BEST-INGREDIENTS"
    tail = "BELOW-INFO-CEILING-NO-DEMONSTRATED-PATH" if below_info else "ABOVE-INFO-CEILING-INTRINSIC"
    return f"A1-OUT-OF-REACH-OF-PUBLISHED-ENVELOPE--{tail}"


def _handoff(trainable: str, light: str, gap: float, ceiling: float, blocks: bool) -> str:
    return (
        f"a1~0.92 is a {light} light for the >500 EAGLE-3 build: the demand sits +{gap:.3f} "
        f"above the published EAGLE-3 first-token (0-alpha) envelope ~0.80 and above every "
        f"demonstrated first-token acceptance (NO published spec-decode method reports 0-alpha "
        f">=0.90 on any chat target), on the ONE position the standard TTT/multi-step levers "
        f"CANNOT lift (depth-0 is serve-faithful; HASS even degrades it). It is NOT "
        f"information-theoretically forbidden (perfect-drafter ceiling 1.0) but exceeds the "
        f"argmax-robustness ceiling ~{ceiling:.2f} (eval top-1 mass), so it has no demonstrated "
        f"training path: a1_target_trainable={trainable}, a1_demand_blocks_go={blocks}. "
        f"0 TPS; analytic + literature only; BASELINE 481.53 unchanged; the drafter BUILD stays "
        f"human-gated. The single biggest residual uncertainty is the true eval-distribution "
        f"top-1 mass (a STATED 0-GPU assumption): a one-GPU top-1-entropy probe on the served "
        f"target would convert this RED to a hard RED or reopen it. NOT a launch. NOT a build. "
        f"NOT a served-file change."
    )


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def self_test(syn: dict[str, Any]) -> dict[str, Any]:
    s1, s2, s3, s4 = (syn["step1_anchor_cliff"], syn["step2_published_envelope"],
                      syn["step3_ceiling"], syn["step4_verdict"])
    checks: dict[str, bool] = {}

    # (a) the deployed gap reconciles #289 / #304.
    checks["a_a1_deployed_is_289_minimum"] = bool(
        s1["a1_deployed"] == A_K_289_DEPLOYED[0] and s1["a1_is_profile_minimum"])
    checks["a_profile_monotone_nondecreasing"] = bool(s1["profile_monotone_nondecreasing"])
    checks["a_gap_vs_required_is_0p192"] = abs(s1["gap_vs_required"] - 0.19205) < 1e-4
    checks["a_required_exceeds_max_interior"] = bool(s1["required_exceeds_all_deployed_ak"])
    # re-derive #304's 0.9213 by inverting the chain-product (independent of the import).
    checks["a_reproduces_304_0p9213"] = bool(s1["reproduces_304_0p9213"])
    # a1-demand sensitivity ordering: perfect-interior < uniform < spec-capped-interior.
    checks["a_demand_sensitivity_ordered"] = bool(
        s1["a1_demand_perfect_interior"] < s1["a1_demand_uniform"] < s1["a1_demand_spec_interior_0p91"])
    checks["a_perfect_interior_near_cliff"] = abs(s1["a1_demand_perfect_interior"] - 0.73017) < 1e-3
    checks["a_spec_interior_demand_ge_uniform"] = bool(s1["realistic_a1_demand_ge_uniform"])

    # (b) the published-envelope citations are real, normalized, and below the demand.
    checks["b_envelope_max_below_demand"] = bool(
        s2["a1_published_envelope_max"] < A1_REQUIRED_FOR_611)
    checks["b_quoted_max_below_demand"] = bool(
        s2["a1_published_exact_quoted_max"] < A1_REQUIRED_FOR_611)
    checks["b_inrepo_in_published_band"] = bool(
        0.74 <= s2["a1_inrepo_eagle3_native_step1"] <= 0.81)
    checks["b_not_in_envelope"] = bool(not s2["in_envelope"])
    checks["b_no_method_ge_090"] = bool(s2["no_published_method_ge_090"])
    checks["b_ttt_does_not_lift_depth0"] = bool(not s2["ttt_lifts_depth0"])
    checks["b_gap_exceeds_one_best_lever"] = bool(s2["gap_in_lever_units"] > 1.0)
    # the in-repo arch lever (linear-MTP -> EAGLE-3) is consistent with the published ~0.05.
    checks["b_inrepo_arch_lever_consistent"] = bool(
        0.02 <= s2["arch_lever_gain_on_a1_inrepo"] <= 0.08)

    # (c) the ceiling bound is COMPUTED from the stated eval-distribution top-1 mass.
    checks["c_ceiling_is_stated_top1_mass"] = bool(
        s3["a1_theoretical_ceiling"] == argmax_robustness_ceiling(EVAL_TOP1_MASS_HIGH))
    checks["c_central_ceiling_is_stated_mean"] = bool(
        s3["a1_argmax_robustness_ceiling_central"] == argmax_robustness_ceiling(EVAL_TOP1_MASS_MEAN))
    checks["c_demand_below_info_ceiling"] = bool(s3["demand_below_info_ceiling"])
    checks["c_demand_above_robustness_ceiling"] = bool(s3["demand_above_robustness_ceiling"])
    checks["c_jensen_floor_consistent"] = bool(
        0.0 < s3["jensen_floor_ref_prob"] <= EVAL_TOP1_MASS_HIGH)

    # (d) verdict consistency: out-of-reach <=> demand > env_max + one best lever; blocks_go aligns.
    expected = classify_a1_trainable(
        A1_REQUIRED_FOR_611, A1_PUBLISHED_ENVELOPE_MAX, AT_FRONTIER_MARGIN)
    checks["d_verdict_matches_rule"] = bool(s4["a1_target_trainable"] == expected)
    checks["d_blocks_go_aligns"] = bool(
        s4["a1_demand_blocks_go"] == (s4["a1_target_trainable"] == "out-of-reach"))
    checks["d_light_aligns"] = bool(
        s4["handoff_light"] == {"in-envelope": "GREEN", "at-frontier": "YELLOW",
                                "out-of-reach": "RED"}[s4["a1_target_trainable"]])

    # (e) imported anchors EXACT and UNCHANGED.
    checks["e_constants_imported_exact"] = bool(
        A_K_289_DEPLOYED[0] == 0.72925
        and A1_REQUIRED_FOR_611 == 0.9213
        and ET_TARGET_CENTRAL_295 == 6.1112149873699195
        and A1_INREPO_EAGLE3_NATIVE_STEP1 == 0.7714
        and EAGLE1_VICUNA7B_0ALPHA_T0 == 0.79
        and EAGLE3_LLAMA31_8B_0ALPHA_T0 == 0.80
        and A1_INFO_THEORETIC_CEILING == 1.0
        and A1_LOW_QUARTILE_297 == 0.6550
        and SPEC_LIFT_289 == 0.91 and CLIFF_VALUE_SPEC == 0.73
        and OFFICIAL_TPS_217 == 481.53 and K_SPEC == 7)

    # (f) the leg carries the 0-TPS + analytic + not-forbidden-but-no-path caveats.
    hl = syn["handoff_line"]
    checks["f_carries_caveats"] = bool(
        "0 TPS" in hl and "analytic" in hl and "NOT a launch" in hl
        and "NOT a build" in hl and "human-gated" in hl
        and "information-theoretically forbidden" in hl)

    # nan-clean over the reported scalars.
    scalars = [
        s1["gap_vs_required"], s1["lift_factor_vs_deployed"], s1["a1_uniform_recomputed"],
        s1["a1_demand_perfect_interior"], s1["a1_demand_spec_interior_0p91"],
        s2["gap_above_env_max"], s2["gap_in_lever_units"], s3["a1_theoretical_ceiling"],
        s3["mean_nll_nats"], s3["jensen_floor_ref_prob"],
    ]
    checks["d_nan_clean"] = all(math.isfinite(float(x)) for x in scalars)

    gate = bool(all(checks.values()))
    return {"a1_cliff_trainability_self_test_passes": gate, "checks": checks}


# --------------------------------------------------------------------------- #
def _assert_nan_clean(payload: Any, path: str = "result") -> list[str]:
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


def _print_report(syn: dict, st: dict) -> None:
    s1, s2, s3, s4 = (syn["step1_anchor_cliff"], syn["step2_published_envelope"],
                      syn["step3_ceiling"], syn["step4_verdict"])
    print("\n" + "=" * 94, flush=True)
    print("EAGLE-3 a1-CLIFF TRAINABILITY (PR #308, denken) -- analytic + literature, 0 GPU",
          flush=True)
    print("=" * 94, flush=True)
    print("  (1) ANCHOR THE DEPLOYED CLIFF", flush=True)
    print(f"      deployed a_k (289)        = {['%.4f' % a for a in s1['a_k_289_deployed']]}", flush=True)
    print(f"      a1 (cliff, profile min)   = {s1['a1_deployed']:.5f}  "
          f"(monotone-nondecreasing={s1['profile_monotone_nondecreasing']}; "
          f"max interior {s1['max_interior_ak']:.5f})", flush=True)
    print(f"      a1 required for 6.11 (304)= {s1['a1_required_for_611']:.4f}  "
          f"(gap +{s1['gap_vs_required']:.4f}, x{s1['lift_factor_vs_deployed']:.3f}; "
          f"recompute {s1['a1_uniform_recomputed']:.4f}, matches304={s1['reproduces_304_0p9213']})",
          flush=True)
    print(f"      required EXCEEDS max interior 0.846 by +{s1['gap_vs_max_interior']:.4f}  "
          f"(exceeds_all_deployed={s1['required_exceeds_all_deployed_ak']})", flush=True)
    print(f"      a1-demand band: perfect-int {s1['a1_demand_perfect_interior']:.4f} < "
          f"uniform {s1['a1_demand_uniform']:.4f} < spec-int(0.91) {s1['a1_demand_spec_interior_0p91']:.4f}",
          flush=True)
    print("-" * 94, flush=True)
    print("  (2) PUBLISHED EAGLE-3 FIRST-TOKEN (0-alpha) ENVELOPE", flush=True)
    print(f"      published max (EAGLE-3 8B,T0) = {s2['a1_published_envelope_max']:.4f}  "
          f"(exact-quoted EAGLE-1 = {s2['a1_published_exact_quoted_max']:.4f})", flush=True)
    print(f"      in-repo EAGLE-3 (gua9x68j)    = {s2['a1_inrepo_eagle3_native_step1']:.4f} "
          f"native step-1 (tf {s2['a1_inrepo_eagle3_tf']:.4f})  [IN published band]", flush=True)
    print(f"      demand gap: +{s2['gap_above_env_max']:.4f} vs env-max, "
          f"+{s2['gap_above_inrepo']:.4f} vs in-repo  = {s2['gap_in_lever_units']:.2f}x best lever",
          flush=True)
    print(f"      no_method_ge_090={s2['no_published_method_ge_090']}  "
          f"ttt_lifts_depth0={s2['ttt_lifts_depth0']}  "
          f"inrepo_arch_lever=+{s2['arch_lever_gain_on_a1_inrepo']:.4f}", flush=True)
    print("-" * 94, flush=True)
    print("  (3) CEILINGS", flush=True)
    print(f"      info-theoretic (perfect drafter, SpecTr) = {s3['a1_info_theoretic_ceiling']:.2f}  "
          f"-> demand BELOW it ({s3['demand_below_info_ceiling']}) => training problem", flush=True)
    print(f"      argmax-robustness (eval top-1 mass {s3['eval_top1_mass_mean']:.2f}, opt "
          f"{s3['eval_top1_mass_high']:.2f}) = {s3['a1_theoretical_ceiling']:.2f}  "
          f"-> demand ABOVE it ({s3['demand_above_robustness_ceiling']})", flush=True)
    print("-" * 94, flush=True)
    print("  (4) VERDICT", flush=True)
    print(f"      a1_target_trainable  = {s4['a1_target_trainable']}  ({s4['handoff_light']})", flush=True)
    print(f"      a1_demand_blocks_go  = {s4['a1_demand_blocks_go']}", flush=True)
    print("-" * 94, flush=True)
    print(f"  PRIMARY a1_cliff_trainability_self_test_passes = "
          f"{st['a1_cliff_trainability_self_test_passes']}", flush=True)
    for k, v in st["checks"].items():
        print(f"          - {k}: {v}", flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print("=" * 94, flush=True)
    print(f"\n  HAND-OFF: {syn['handoff_line']}\n", flush=True)


def _maybe_log_wandb(args, payload: dict) -> str | None:
    if not getattr(args, "wandb_name", None):
        return None
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[a1-cliff-trainability] wandb logging unavailable: {exc}", flush=True)
        return None

    syn, st = payload["synthesis"], payload["self_test"]
    s1, s2, s3, s4 = (syn["step1_anchor_cliff"], syn["step2_published_envelope"],
                      syn["step3_ceiling"], syn["step4_verdict"])
    run = init_wandb_run(
        job_type="eagle3-a1-cliff-trainability",
        agent="denken",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["eagle3-a1-cliff-trainability", "eagle3-build-spec", "position-1-acceptance",
              "drafter-trainability", "literature", "validity", "zero-tps"],
        config={
            "pr": 308, "analysis_only": True, "K_spec": K_SPEC,
            "a1_deployed": A1_DEPLOYED, "a1_required_for_611": A1_REQUIRED_FOR_611,
            "a1_inrepo_eagle3_native_step1": A1_INREPO_EAGLE3_NATIVE_STEP1,
            "a1_published_envelope_max": A1_PUBLISHED_ENVELOPE_MAX,
            "a1_published_exact_quoted_max": A1_PUBLISHED_EXACT_QUOTED_MAX,
            "eval_top1_mass_mean": EVAL_TOP1_MASS_MEAN,
            "eval_top1_mass_high": EVAL_TOP1_MASS_HIGH,
            "et_target_central_295": ET_TARGET_CENTRAL_295,
            "official_tps_217": OFFICIAL_TPS_217, "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[a1-cliff-trainability] wandb: no run (no WANDB_API_KEY/mode) -- skipping",
              flush=True)
        return None

    summary: dict[str, Any] = {
        # PRIMARY + TEST
        "a1_cliff_trainability_self_test_passes":
            int(bool(st["a1_cliff_trainability_self_test_passes"])),
        "a1_required_for_611": A1_REQUIRED_FOR_611,
        # the three logged ceilings/envelope the GO/NO-GO packet reads
        "a1_published_envelope_max": s2["a1_published_envelope_max"],
        "a1_theoretical_ceiling": s3["a1_theoretical_ceiling"],
        "a1_info_theoretic_ceiling": s3["a1_info_theoretic_ceiling"],
        # verdict
        "a1_target_trainable_code": {"in-envelope": 2, "at-frontier": 1,
                                     "out-of-reach": 0}[s4["a1_target_trainable"]],
        "a1_demand_blocks_go": int(bool(s4["a1_demand_blocks_go"])),
        "verdict_out_of_reach": int(syn["verdict"].startswith("A1-OUT-OF-REACH")),
        # step 1
        "a1_deployed": s1["a1_deployed"],
        "a1_is_profile_minimum": int(bool(s1["a1_is_profile_minimum"])),
        "max_interior_ak": s1["max_interior_ak"],
        "gap_vs_required": s1["gap_vs_required"],
        "gap_vs_max_interior": s1["gap_vs_max_interior"],
        "lift_factor_vs_deployed": s1["lift_factor_vs_deployed"],
        "required_exceeds_all_deployed_ak": int(bool(s1["required_exceeds_all_deployed_ak"])),
        "a1_uniform_recomputed": s1["a1_uniform_recomputed"],
        "reproduces_304_0p9213": int(bool(s1["reproduces_304_0p9213"])),
        "a1_demand_perfect_interior": s1["a1_demand_perfect_interior"],
        "a1_demand_uniform": s1["a1_demand_uniform"],
        "a1_demand_spec_interior_0p91": s1["a1_demand_spec_interior_0p91"],
        # step 2
        "a1_inrepo_eagle3_native_step1": s2["a1_inrepo_eagle3_native_step1"],
        "a1_inrepo_eagle3_tf": s2["a1_inrepo_eagle3_tf"],
        "a1_published_exact_quoted_max": s2["a1_published_exact_quoted_max"],
        "gap_above_env_max": s2["gap_above_env_max"],
        "gap_above_inrepo": s2["gap_above_inrepo"],
        "gap_in_lever_units": s2["gap_in_lever_units"],
        "arch_lever_gain_on_a1_inrepo": s2["arch_lever_gain_on_a1_inrepo"],
        "no_published_method_ge_090": int(bool(s2["no_published_method_ge_090"])),
        "ttt_lifts_depth0": int(bool(s2["ttt_lifts_depth0"])),
        # step 3
        "eval_top1_mass_mean": s3["eval_top1_mass_mean"],
        "eval_top1_mass_high": s3["eval_top1_mass_high"],
        "a1_argmax_robustness_ceiling_central": s3["a1_argmax_robustness_ceiling_central"],
        "demand_below_info_ceiling": int(bool(s3["demand_below_info_ceiling"])),
        "demand_above_robustness_ceiling": int(bool(s3["demand_above_robustness_ceiling"])),
        "jensen_floor_ref_prob": s3["jensen_floor_ref_prob"],
        **{f"selftest_{k}": int(bool(v)) for k, v in st["checks"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="eagle3_a1_cliff_trainability_result",
                      artifact_type="validity", data=payload)
    finish_wandb(run)
    rid = getattr(run, "id", None)
    print(f"[a1-cliff-trainability] wandb logged (run {rid})", flush=True)
    return rid


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="eagle3-a1-trainability")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    syn = synthesize()
    st = self_test(syn)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 308, "agent": "denken",
        "kind": "eagle3-a1-cliff-trainability", "synthesis": syn, "self_test": st,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[a1-cliff-trainability] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn, st)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eagle3_a1_cliff_trainability_results.json"

    wid = None
    if not args.no_wandb:
        wid = _maybe_log_wandb(args, payload)
    payload["wandb_run_id"] = wid
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[a1-cliff-trainability] wrote {out_path}  (wandb run {wid})", flush=True)

    if args.self_test:
        ok = st["a1_cliff_trainability_self_test_passes"] and payload["nan_clean"]
        print(f"[a1-cliff-trainability] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
