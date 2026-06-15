#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #336 (student lawine) -- What head recipe could lift fusion top-4 coverage +0.031 to clear the 0.9213 bar?

WHAT THIS CARD DOES (0-GPU, 0-TPS, no training, no model forward, no served-file change, no HF Job)
---------------------------------------------------------------------------------------------------
lawine #330 (hfrscdai, MERGED) established the DEMAND-half blocker for the EAGLE-3 compliant-500 lane:
the official 128 "ShareGPT" eval is a misnomer -- it is 100% reasoning/STEM (mmlu_pro 57 / gpqa 57 /
aime 14), so the honest unconditional top-4 ROOT coverage prior for the fern #34 (gua9x68j) {2,21,39}
fusion head is 0.8903, which MISSES the 0.9213 build bar (P(clears) ~ 0.06). The drag is the mmlu_pro
reasoning-CoT vocabulary breadth (per-source coverage 0.8465), NOT the answer-letter tail (~0.0002 of
tokens). The lane revives ONLY if a better-trained head delivers >= +0.031 aggregate top-4 coverage.

This card answers the open question lawine #330 left: IS +0.031 reachable with a realistic head-training
recipe, and WHICH recipe? It is the demand-side revival lever (the fern #335 joint AND-gate folds in its
verdict). CPU + literature only -- no head training, no checkpoint, no model forward. It does FOUR
things on banked anchors (it re-derives none of the upstream coverage numbers):

  1. PER-SOURCE LIFT-TARGET TABLE. The aggregate gap to the bar is 0.9213 - 0.8903 = 0.031 (weighted-
     sum lift 0.031 * 128 = 3.97). Decompose that aggregate lift into the per-source coverage gain each
     source must deliver under THREE allocation policies:
       (a) mmlu_pro-only  -- concentrate all lift on the binding source (mmlu_pro 0.8465 -> ~0.9162,
           single-source Delta ~ 0.070);
       (b) uniform        -- the same Delta on all three sources (Delta = gap = 0.031 since weights
           sum to 1);
       (c) proportional-to-gap -- lift each source toward 1.0 in proportion to its current shortfall
           (1 - cov), i.e. Delta_i = k*(1 - cov_i) with a single common k.
     Emit each policy's per-source Delta vector, the min and max single-source lift it demands, and the
     headline minimax fact: uniform is the allocation that MINIMIZES the hardest single-source lift
     (its max single-source lift == the aggregate gap 0.031); concentrating (a) demands the largest
     single jump (~0.070). Verify each policy re-aggregates to EXACTLY the 0.9213 bar.

  2. DIAGNOSE THE mmlu_pro DRAG. From lawine #330's decomposition, characterize WHY mmlu_pro top-4
     coverage is low: reasoning-CoT token breadth (multi-domain technical vocabulary, LaTeX/symbolic
     spans, numerals) vs. the near-absent answer-letter tail. State which token classes the head misses
     and whether the shortfall is head-CAPACITY-limited or training-OBJECTIVE/DATA-limited.

  3. LITERATURE-GROUNDED RECIPE RANKING. Rank candidate head-training recipes by expected aggregate
     top-4 ROOT-coverage lift, each with a literature citation and a plausibility band:
       - soft-KD / top-k logit distillation (match teacher top-k, not just argmax) -- PRIMARY tail lever;
       - more on-distribution reasoning data in the head's train mix -- targets the binding source;
       - deeper/wider fusion head -- secondary capacity, VRAM/latency cost;
       - on-policy TTT -- RANKED LOW: lawine #316 found TTT lifts depth>=2 acceptance, NOT the ROOT
         token, and the 0.9213 bar is a top-4 ROOT bar, so TTT is the wrong axis here.
     Total-order them, give each an expected Delta-cov band, and flag which COMBINATION plausibly clears
     +0.031 (the soft-KD + reasoning-data pair, under a conservative non-additivity haircut).

  4. FEASIBILITY VERDICT. State plainly whether +0.031 is REACHABLE, REACHABLE-MARGINAL, or
     OUT-OF-REACH. If reachable, name the cheapest recipe that clears the bar and the cost to validate
     it (head retrain -> re-measure coverage, NOT a TPS run).

LOCAL CPU-only analytic card. No GPU / vLLM / model forward / training / checkpoint / HF Job /
submission / served-file change / publish. NOT a launch. BASELINE stays 481.53 (0 TPS). Greedy/PPL
untouched. This adds 0 TPS; it scopes the head-training target the fern #335 joint AND-gate consumes.

PRIMARY metric  coverage_lift_target_self_test_passes
TEST    metric  min_aggregate_lift_required  (float; the aggregate top-4 coverage lift the bar demands ~ 0.031)
REPORT          feasibility_verdict          (REACHABLE | REACHABLE-MARGINAL | OUT-OF-REACH)

Reproduce:
    cd target/ && .venv/bin/python \\
        research/validity/eagle3_head_coverage_lift_target/eagle3_head_coverage_lift_target.py \\
        --self-test --wandb_group eagle3-head-coverage-lift-target \\
        --wandb_name lawine/eagle3-head-coverage-lift-target
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --------------------------------------------------------------------------- #
# Imported fleet anchors (cited EXACTLY, UNCHANGED; this card re-derives none).
# Cross-checked at runtime against the on-disk banked lawine #330 artifact so there is no silent drift.
# --------------------------------------------------------------------------- #
# lawine #316 (5lnz5jgb) via lawine #323 (ceddxj20): the regime-invariant build bar (unconditional top-4
# ROOT coverage target solving 1 + sum_{j=1..7} T^j = 6.11).
T_EFFECTIVE = 0.9213011665456927        # build bar in UNCONDITIONAL top-4 units (regime-invariant)

# fern #34 (gua9x68j): the ONLY trained {2,21,39} fusion EAGLE-3 head. W&B-verified eval/* summary,
# teacher-forced on the 240-record benchmark-matched reasoning holdout. Per-source UNCONDITIONAL top-4
# coverage -- these ARE the official-eval per-source curve (holdout source mix matches official 57/57/14).
# (lawine #330 step2 / synthesis.imported.fern34_per_source_top4.)
FERN_SRC_TOP4 = {
    "aime":     0.957005303537408,
    "gpqa":     0.9175953770859131,
    "mmlu_pro": 0.846544405293677,
}
# Official 128 eval source counts (by id prefix; lawine #330 loaded live from eval_prompts_sharegpt.json).
# ignore_eos=true -> uniform 512 completion tokens/prompt -> token-weight == prompt-weight == these counts.
OFFICIAL_SRC_COUNTS = {"aime": 14, "gpqa": 57, "mmlu_pro": 57}
N_PROMPTS = 128

# Published cross-paper estimate for a FULLY-trained head (a1~0.765): aggregate uncond top-4 band (#323).
# Even this lit-central MISSES the 0.9213 reasoning bar -> the clear is at the optimistic edge of training.
LIT_TOP4_CENTRAL = 0.913
LIT_TOP4_RANGE = [0.899, 0.929]

# lawine #330 banked aggregate (official-eval-weighted) -- cross-check target for the reproduction.
BANKED_330_AGGREGATE = 0.8902659519153152
BANKED_330_GAP = -0.031035214630377506   # aggregate - bar (negative == miss)

BASELINE_TPS = 481.53          # current best summary.json:tps (unchanged; 0-TPS analytic)

# Banked on-disk lawine #330 result (read-only) for the constant-drift cross-check.
PR330_RESULTS = (REPO_ROOT / "research" / "validity" / "eagle3_sharegpt_coverage_prior"
                 / "eagle3_sharegpt_coverage_prior_results.json")

TOL = 1e-9
TOL_REPRO = 1e-6

# --------------------------------------------------------------------------- #
# Recipe ranking model (deliverable 3). EXPECTED-VALUE PRIORS with literature-grounded bands -- NOT
# measurements. Each Delta-cov is the expected lift in AGGREGATE unconditional top-4 ROOT coverage. The
# self-test only checks STRUCTURAL properties (strict total order, soft-KD #1, TTT last, the combination
# clears); the band magnitudes are honest priors a retrain would confirm, not facts this card measures.
# --------------------------------------------------------------------------- #
RECIPES = {
    "soft_kd_topk_distill": {
        "rank_intended": 1,
        "delta_cov_central": 0.030,
        "delta_cov_band": [0.015, 0.045],
        "targets": "the top-4 TAIL directly (ranks 2..4)",
        "citation": (
            "DistillSpec (Zhou et al., ICLR 2024, arXiv:2310.08461): KL-divergence knowledge "
            "distillation of the draft to the target lifts acceptance rate / block efficiency (reports "
            "10-45% speedup), with on-policy data generation central. Online Speculative Decoding (Liu et "
            "al., ICML 2024, arXiv:2310.07177): KD-based online adaptation raises aggregate token "
            "acceptance by 0.10-0.65. Medusa (Cai et al., ICLR 2024, arXiv:2401.10774): self-distillation "
            "head training. NOTE: these establish general-purpose draft KD; the top-4-TAIL-specific "
            "mechanism below is a REASONED INFERENCE (richer soft-target gradient than hard-CE), NOT a "
            "per-paper ablation of top-4 ROOT coverage."),
        "why": (
            "The bar is a top-4 (not top-1) coverage metric -- it is literally the drafter's top-k tail "
            "accuracy. fern #34 was trained with hard cross-entropy (argmax target), which gives NO "
            "gradient to the rank-2..4 mass, leaving the tail uncalibrated by construction; matching the "
            "teacher top-k softmax trains exactly that tail. This makes soft-KD/top-k distillation the "
            "single most directly-targeted lever for THIS metric, hence rank 1 -- a mechanism inference "
            "(general draft-KD gains are measured in the lit; the per-rank attribution is ours)."),
    },
    "more_reasoning_root_data": {
        "rank_intended": 2,
        "delta_cov_central": 0.025,
        "delta_cov_band": [0.010, 0.040],
        "targets": "the BINDING source (mmlu_pro, weight 0.445, lowest coverage 0.8465)",
        "citation": (
            "EAGLE-3 (Li et al., 2025, 'Scaling up Inference Acceleration of LLMs via Training-Time "
            "Test', arXiv:2503.01840): removes EAGLE/EAGLE-2's feature-prediction (feature-regression) "
            "constraint and switches to direct token prediction specifically so the draft head 'fully "
            "benefits from scaling up training data' (its stated finding); ~1.4x over EAGLE-2."),
        "why": (
            "The mmlu_pro drag is reasoning-CoT vocabulary BREADTH (law/philosophy/business/health/...), "
            "a training-data-mix property. Adding mmlu_pro/gpqa-style CoT to the head's train mix "
            "directly covers the binding source's breadth. High leverage (heaviest, lowest source), but "
            "ranked below soft-KD because data-mix shifts raise top-1 more reliably than the top-4 tail "
            "per se, and the lift saturates as the head's tail fills in."),
    },
    "deeper_wider_head": {
        "rank_intended": 3,
        "delta_cov_central": 0.012,
        "delta_cov_band": [0.005, 0.020],
        "targets": "the ceiling (multi-domain capacity), NOT the binding objective gap",
        "citation": (
            "Medusa (Cai et al., ICLR 2024, arXiv:2401.10774; Medusa-1 frozen vs Medusa-2 joint-finetune "
            "-> a capacity-vs-latency tradeoff) and EAGLE-3 (arXiv:2503.01840, capacity gains attributed "
            "mainly to the architecture change, not raw head-width scaling). DIRECTIONAL support only -- "
            "neither paper publishes a head width/depth ablation, so the magnitude band below is a "
            "reasoned estimate, not a citable quantitative claim."),
        "why": (
            "Capacity is NOT the primary binding constraint: the SAME head already reaches 0.957 on aime, "
            "proving it can hit high coverage when tokens are well-covered. A wider/deeper head lifts the "
            "ceiling for the broadest domains but costs per-step draft latency + VRAM (ubel #299/#306: the "
            "current head fits 24 GB at 20.16 GiB / 3.84 GiB headroom; a bigger head eats that headroom "
            "and slows the draft forward, partly offsetting realized TPS even if E[T] rises). Secondary."),
    },
    "on_policy_ttt": {
        "rank_intended": 4,
        "delta_cov_central": 0.002,
        "delta_cov_band": [0.000, 0.005],
        "targets": "depth>=2 continuation acceptance -- the WRONG axis for a top-4 ROOT bar",
        "citation": (
            "lawine #316 (INTERNAL MEASUREMENT on this exact fusion head): on-policy TTT lifts depth>=2 "
            "(continuation) acceptance, NOT the ROOT (position-1) token. This per-position split is OUR "
            "measured result, NOT a general literature claim -- the external spec-decode lit (Online "
            "Speculative Decoding, arXiv:2310.07177; EAGLE-3 training-time-test) reports only AGGREGATE "
            "acceptance gains and does not ablate root-vs-continuation, so #316 is the load-bearing "
            "evidence here. (Note: EAGLE-3's 'TTT' = training-time test, a positive TRAINING technique, "
            "distinct from this on-policy test-time-training recipe.)"),
        "why": (
            "RANKED LOWEST. The 0.9213 bar is a top-4 ROOT (position-1) coverage bar. lawine #316 MEASURED "
            "that TTT improves the deep-spine continuation (E[T] length given the root is accepted), which "
            "is ORTHOGONAL to root coverage -- so for THIS bar TTT's expected lift on root top-4 is ~0. "
            "(TTT would help E[T] realization downstream, a different gate, not this one.)"),
    },
}
# Conservative non-additivity factor for stacking the two training levers (soft-KD + reasoning-data):
# they partly target the same mmlu_pro shortfall and the head saturates, so the combined lift is LESS
# than the naive sum. 0.70 == retain 70% of the sum, lose 30% to overlap/saturation. An explicit modeling
# assumption, not a measured number.
COMBO_NON_ADDITIVITY = 0.70

FEASIBILITY_THRESHOLDS = {
    "reachable_combo_central_clears_and_single_lever_clears": "REACHABLE",
    "reachable_combo_central_clears_single_lever_marginal": "REACHABLE-MARGINAL",
    "out_of_reach_combo_central_misses": "OUT-OF-REACH",
}


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #
def weights() -> dict[str, float]:
    return {k: OFFICIAL_SRC_COUNTS[k] / N_PROMPTS for k in OFFICIAL_SRC_COUNTS}


def aggregate(cov: dict[str, float]) -> float:
    """Token-weighted (== prompt-weighted, uniform 512 tokens) aggregate unconditional top-4 coverage."""
    w = weights()
    return sum(cov[k] * w[k] for k in cov)


def load_banked_330() -> dict[str, Any]:
    return json.loads(PR330_RESULTS.read_text()) if PR330_RESULTS.exists() else {}


# --------------------------------------------------------------------------- #
# Deliverable 1: per-source lift-target table under three allocation policies.
# --------------------------------------------------------------------------- #
def lift_target_table() -> dict[str, Any]:
    w = weights()
    cov0 = dict(FERN_SRC_TOP4)
    agg0 = aggregate(cov0)
    gap = T_EFFECTIVE - agg0                    # the aggregate lift the bar demands (>0 == miss)
    weighted_sum_lift = gap * N_PROMPTS         # 0.031 * 128 ~ 3.97

    def policy_result(deltas: dict[str, float]) -> dict[str, Any]:
        new_cov = {k: cov0[k] + deltas[k] for k in cov0}
        new_agg = aggregate(new_cov)
        nz = [deltas[k] for k in deltas if deltas[k] > TOL]
        return {
            "delta_per_source": deltas,
            "new_cov_per_source": new_cov,
            "new_aggregate": new_agg,
            "reaches_bar": bool(abs(new_agg - T_EFFECTIVE) <= TOL),
            "max_single_source_lift": max(deltas.values()),
            "min_single_source_lift": min(deltas.values()),
            "min_nonzero_single_source_lift": (min(nz) if nz else 0.0),
            "all_cov_le_one": bool(all(v <= 1.0 + TOL for v in new_cov.values())),
            "binding_source": max(deltas, key=lambda k: deltas[k]),
        }

    # (a) mmlu_pro-only: all the weighted lift on the binding source. w_m * Delta_m = gap.
    d_a = {k: 0.0 for k in cov0}
    d_a["mmlu_pro"] = gap / w["mmlu_pro"]
    pol_a = policy_result(d_a)

    # (b) uniform: same Delta on all three. sum_i w_i*Delta = Delta * sum w_i = Delta = gap.
    d_b = {k: gap for k in cov0}
    pol_b = policy_result(d_b)

    # (c) proportional-to-gap: Delta_i = k*(1 - cov_i). sum_i w_i*k*(1-cov_i) = k*(1 - agg0) = gap.
    k_prop = gap / (1.0 - agg0)
    d_c = {k: k_prop * (1.0 - cov0[k]) for k in cov0}
    pol_c = policy_result(d_c)

    # minimax fact: uniform's max single-source lift == gap; no allocation that raises a weighted mean
    # (weights summing to 1) by `gap` can keep every element below `gap`, so uniform is minimax-optimal.
    uniform_is_minimax = bool(
        pol_b["max_single_source_lift"] <= pol_a["max_single_source_lift"] + TOL
        and pol_b["max_single_source_lift"] <= pol_c["max_single_source_lift"] + TOL)

    return {
        "baseline_cov_per_source": cov0,
        "weights": w,
        "aggregate_baseline": agg0,
        "bar": T_EFFECTIVE,
        "aggregate_gap": gap,
        "weighted_sum_lift": weighted_sum_lift,
        "min_aggregate_lift_required": gap,    # policy-independent: cannot clear with less aggregate lift
        "binding_source_overall": min(cov0, key=lambda k: cov0[k]),
        "policies": {
            "a_mmlu_pro_only": pol_a,
            "b_uniform": pol_b,
            "c_proportional_to_gap": pol_c,
        },
        "k_proportional": k_prop,
        "uniform_is_minimax_optimal": uniform_is_minimax,
        "note": (
            "Aggregate gap to bar = {:.4f} (weighted-sum lift {:.4f}*128 = {:.2f}). The MINIMUM AGGREGATE "
            "lift to clear is {:.4f} regardless of allocation. How that maps to per-source demands depends "
            "on the policy: (a) mmlu_pro-only concentrates all lift on the binding source -> a single "
            "{:.4f} jump (0.8465 -> {:.4f}); (b) uniform asks {:.4f} of EVERY source (Delta == gap, since "
            "weights sum to 1) and is MINIMAX-OPTIMAL (smallest possible hardest single-source lift); "
            "(c) proportional-to-gap lifts each toward 1.0 by k={:.4f}*shortfall -> mmlu_pro {:.4f}, gpqa "
            "{:.4f}, aime {:.4f}. All three re-aggregate to EXACTLY the 0.9213 bar."
            .format(gap, gap, weighted_sum_lift, gap, pol_a["max_single_source_lift"],
                    pol_a["new_cov_per_source"]["mmlu_pro"], gap, k_prop,
                    d_c["mmlu_pro"], d_c["gpqa"], d_c["aime"])),
    }


# --------------------------------------------------------------------------- #
# Deliverable 2: diagnose the mmlu_pro drag.
# --------------------------------------------------------------------------- #
def mmlu_drag_diagnosis() -> dict[str, Any]:
    cov0 = dict(FERN_SRC_TOP4)
    binding = min(cov0, key=lambda k: cov0[k])
    aime_minus_mmlu = cov0["aime"] - cov0["mmlu_pro"]
    return {
        "binding_source": binding,
        "binding_cov": cov0[binding],
        "aime_minus_mmlu_gap": aime_minus_mmlu,
        "token_classes_missed": [
            "multi-domain technical vocabulary (law / philosophy / business / health / psychology / "
            "economics / engineering -- mmlu_pro's 14 subject domains, the highest-entropy token mix)",
            "rare / long-tail domain terms outside the head's training distribution",
            "LaTeX / symbolic spans and numerals interleaved in the CoT",
        ],
        "NOT_the_drag": (
            "the MCQ answer-letter tail: lawine #330 measured the literal committed answer-letter token at "
            "~0.0002 (0.02%) of all generated tokens (only ~7.8% of 512-token chains even reach 'ANSWER:'), "
            "~420x below the level the letter-contamination model would need to matter. The drag is the "
            "WHOLE reasoning-CoT body, not a near-absent letter."),
        "capacity_vs_data_verdict": "TRAINING-OBJECTIVE/DATA-LIMITED (capacity secondary)",
        "capacity_vs_data_rationale": (
            "Primarily TRAINING-limited, not head-capacity-limited, on three grounds: (1) OBJECTIVE -- "
            "top-4 (not top-1) coverage is the drafter's top-k TAIL; fern #34's hard-CE objective optimizes "
            "only the argmax and provides NO gradient to ranks 2..4, so the tail is uncalibrated BY "
            "CONSTRUCTION (soft-KD fixes this). (2) CAPACITY HEADROOM -- the SAME 1-layer fusion head "
            "already reaches 0.957 on aime; the shortfall tracks domain BREADTH (a data-mix property), not "
            "an inherent ceiling, so the head demonstrably HAS the capacity to cover a domain when its "
            "tokens are trained. (3) UNDERTRAINING -- fern #34 is K=1, hard-CE, no soft-KD, no TTT (the only "
            "trained head). Capacity is a SECONDARY ceiling at the very top: even the published fully-"
            "trained lit-central head (0.913) still misses 0.9213 on a 100%-reasoning eval, leaving a thin "
            "residual a deeper/wider head would address -- which is why capacity is the #3 lever, not #1."),
    }


# --------------------------------------------------------------------------- #
# Deliverable 3: literature-grounded recipe ranking.
# --------------------------------------------------------------------------- #
def recipe_ranking(gap: float) -> dict[str, Any]:
    # strict total order by expected central Delta-cov (descending).
    ordered = sorted(RECIPES.items(), key=lambda kv: -kv[1]["delta_cov_central"])
    ranking = []
    for rank, (name, r) in enumerate(ordered, start=1):
        ranking.append({
            "rank": rank,
            "recipe": name,
            "delta_cov_central": r["delta_cov_central"],
            "delta_cov_band": r["delta_cov_band"],
            "central_clears_alone": bool(r["delta_cov_central"] >= gap - TOL),
            "high_band_clears_alone": bool(r["delta_cov_band"][1] >= gap - TOL),
            "targets": r["targets"],
            "citation": r["citation"],
            "why": r["why"],
        })
    # strict-total-order check: all central values distinct -> strictly descending.
    centrals = [r["delta_cov_central"] for _, r in ordered]
    strict_total_order = all(centrals[i] > centrals[i + 1] for i in range(len(centrals) - 1))

    # the recommended COMBINATION: soft-KD + reasoning-data (the two training levers; no arch change).
    kd = RECIPES["soft_kd_topk_distill"]
    rd = RECIPES["more_reasoning_root_data"]
    combo_naive_central = kd["delta_cov_central"] + rd["delta_cov_central"]
    combo_central = combo_naive_central * COMBO_NON_ADDITIVITY
    combo_low = (kd["delta_cov_band"][0] + rd["delta_cov_band"][0]) * COMBO_NON_ADDITIVITY
    combo_high = (kd["delta_cov_band"][1] + rd["delta_cov_band"][1]) * COMBO_NON_ADDITIVITY
    combo = {
        "recipe": "soft_kd_topk_distill + more_reasoning_root_data",
        "non_additivity_factor": COMBO_NON_ADDITIVITY,
        "delta_cov_naive_sum": combo_naive_central,
        "delta_cov_central": combo_central,
        "delta_cov_band": [combo_low, combo_high],
        "central_clears": bool(combo_central >= gap - TOL),
        "low_band_clears": bool(combo_low >= gap - TOL),
        "high_band_clears": bool(combo_high >= gap - TOL),
        "single_lever_soft_kd_central": kd["delta_cov_central"],
        "single_lever_soft_kd_central_clears": bool(kd["delta_cov_central"] >= gap - TOL),
        "note": (
            "The cheapest CLEAR is the two TRAINING levers stacked on the existing {{2,21,39}} arch (no "
            "capacity change -> same VRAM, same deploy/latency path). soft-KD central {:.3f} alone lands "
            "JUST UNDER the gap {:.3f} (marginal); reasoning-data central {:.3f} also alone-marginal. "
            "Stacked, naive sum {:.3f} -> with a conservative {:.2f} non-additivity haircut central "
            "{:.4f}, which CLEARS the gap {:.4f} with a thin margin. Band [{:.4f}, {:.4f}]: the high and "
            "central clear, the low (both levers at their floors) MISSES -> not guaranteed. If the two "
            "training levers land low, add the deeper/wider head (#3, +{:.3f} central) to buy margin and "
            "lift the residual ceiling the lit-central 0.913 still leaves under 0.9213."
            .format(kd["delta_cov_central"], gap, rd["delta_cov_central"], combo_naive_central,
                    COMBO_NON_ADDITIVITY, combo_central, gap, combo_low, combo_high,
                    RECIPES["deeper_wider_head"]["delta_cov_central"])),
    }
    return {
        "ranking": ranking,
        "strict_total_order": strict_total_order,
        "top_recipe": ranking[0]["recipe"],
        "bottom_recipe": ranking[-1]["recipe"],
        "ttt_ranked_last": bool(ranking[-1]["recipe"] == "on_policy_ttt"),
        "ttt_rationale_is_316_root_not_deep": RECIPES["on_policy_ttt"]["why"],
        "recommended_combination": combo,
    }


# --------------------------------------------------------------------------- #
# Deliverable 4: feasibility verdict.
# --------------------------------------------------------------------------- #
def feasibility_verdict(gap: float, ranking: dict[str, Any]) -> dict[str, Any]:
    combo = ranking["recommended_combination"]
    soft_kd_alone_clears = combo["single_lever_soft_kd_central_clears"]
    combo_central_clears = combo["central_clears"]
    if not combo_central_clears:
        verdict = "OUT-OF-REACH"
    elif soft_kd_alone_clears:
        verdict = "REACHABLE"
    else:
        verdict = "REACHABLE-MARGINAL"
    return {
        "feasibility_verdict": verdict,
        "min_aggregate_lift_required": gap,
        "cheapest_clearing_recipe": combo["recipe"],
        "cheapest_recipe_keeps_arch": True,
        "validation_cost": (
            "ONE cluster drafter retrain (soft-KD top-k distillation + reasoning-CoT data augmentation on "
            "the existing {2,21,39} fusion arch; open instructions/training-request.md, NOT a TPS run) -> "
            "re-measure per-source UNCONDITIONAL top-4 coverage on the 240-record benchmark-matched "
            "reasoning holdout via the fern #34 / lawine #330 RANKPROBE_W=4 protocol (CPU/1-GPU coverage "
            "eval, NO HF Job, 0 TPS). Gate: retrained aggregate >= 0.9213 flips the demand-half GO for the "
            "fern #335 joint AND-gate. No architecture change -> the ubel #299/#306 24 GB VRAM fit and the "
            "deployed latency path are preserved; only the drafter weights change."),
        "rationale": (
            "{} +0.031 is reachable with the soft-KD + reasoning-data COMBINATION (central {:.4f} clears the "
            "{:.4f} gap), but at the OPTIMISTIC EDGE: a single lever (soft-KD central {:.3f}) lands just "
            "under, the combination's low band ({:.4f}) misses, and even the published fully-trained "
            "lit-central head (0.913) still misses this 100%-reasoning eval -- so the clear is not "
            "guaranteed and a deeper/wider head may be needed for margin. NOT out of reach, NOT a "
            "slam-dunk. The cheapest clear keeps the architecture (same VRAM/latency), so the validation "
            "spend is one cluster retrain + one CPU/1-GPU coverage re-measure, 0 TPS, 0 HF quota."
            .format("REACHABLE-MARGINAL:" if verdict == "REACHABLE-MARGINAL" else verdict + ":",
                    combo["delta_cov_central"], gap, combo["single_lever_soft_kd_central"],
                    combo["delta_cov_band"][0])),
        "caveats": (
            "PRIOR / FEASIBILITY SCOPING, NOT a measurement. The per-recipe Delta-cov bands are honest "
            "literature-grounded expected-value priors a retrain would confirm; they are NOT measured here "
            "(no training, no model forward). The 0.9213 bar is position-1 ROOT coverage: necessary, NOT "
            "sufficient -- the deployed deep spine still caps E[T] (#316), and the private-tax robustness "
            "(fern #325 pass-a) is a SEPARATE gate. This card scopes the DEMAND-half revival target only; "
            "the fern #335 joint AND-gate folds in both halves. 0 TPS; greedy/PPL untouched; BASELINE "
            "481.53 unchanged; no checkpoint, no publish, no HF Job."),
    }


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    table = lift_target_table()
    gap = table["aggregate_gap"]
    drag = mmlu_drag_diagnosis()
    ranking = recipe_ranking(gap)
    verdict = feasibility_verdict(gap, ranking)
    return {
        "step1_lift_target_table": table,
        "step2_mmlu_drag_diagnosis": drag,
        "step3_recipe_ranking": ranking,
        "step4_feasibility_verdict": verdict,
        "test_metrics": {
            "coverage_lift_target_self_test_passes": None,   # filled by main
            "min_aggregate_lift_required": gap,
            "feasibility_verdict": verdict["feasibility_verdict"],
            "aggregate_baseline_top4": table["aggregate_baseline"],
        },
        "imported": {
            "T_effective": T_EFFECTIVE,
            "fern34_per_source_top4": FERN_SRC_TOP4,
            "official_src_counts": OFFICIAL_SRC_COUNTS,
            "lit_top4_central": LIT_TOP4_CENTRAL, "lit_top4_range": LIT_TOP4_RANGE,
            "provenance": (
                "build bar lawine #316 (5lnz5jgb) via lawine #323 (ceddxj20); per-source unconditional "
                "top-4 + aggregate fern #34 (gua9x68j, W&B-verified eval/* summary); official 128 eval "
                "source mix + the 0.8903 aggregate / 0.031 gap banked by lawine #330 (hfrscdai, on-disk "
                "eagle3_sharegpt_coverage_prior_results.json); TTT root-vs-deep finding lawine #316. "
                "Lift-target decomposition, drag diagnosis, literature recipe ranking, and feasibility "
                "verdict are this card (#336)."),
        },
    }


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY). >= 15 checks, 0 GPU.
# --------------------------------------------------------------------------- #
def self_test(syn: dict[str, Any], banked330: dict[str, Any]) -> dict[str, Any]:
    t = syn["step1_lift_target_table"]
    drag = syn["step2_mmlu_drag_diagnosis"]
    rk = syn["step3_recipe_ranking"]
    fv = syn["step4_feasibility_verdict"]
    gap = t["aggregate_gap"]
    c: dict[str, bool] = {}

    # (a) reproduction of the upstream anchors.
    c["01_aggregate_0p8903_reproduced"] = bool(abs(t["aggregate_baseline"] - BANKED_330_AGGREGATE) <= TOL)
    c["02_gap_0p031_recovered"] = bool(abs(gap - (-BANKED_330_GAP)) <= TOL and gap > 0.0)
    c["03_weighted_sum_lift_3p97"] = bool(abs(t["weighted_sum_lift"] - 3.9725074727) <= 1e-4)
    c["04_min_aggregate_lift_eq_gap"] = bool(abs(t["min_aggregate_lift_required"] - gap) <= TOL)

    # (b) the three allocation policies are internally consistent (each re-aggregates to EXACTLY the bar).
    pol = t["policies"]
    c["05_policy_a_reaches_bar"] = bool(pol["a_mmlu_pro_only"]["reaches_bar"])
    c["06_policy_b_reaches_bar"] = bool(pol["b_uniform"]["reaches_bar"])
    c["07_policy_c_reaches_bar"] = bool(pol["c_proportional_to_gap"]["reaches_bar"])
    c["08_all_policies_reach_bar"] = bool(
        abs(pol["a_mmlu_pro_only"]["new_aggregate"] - T_EFFECTIVE) <= TOL
        and abs(pol["b_uniform"]["new_aggregate"] - T_EFFECTIVE) <= TOL
        and abs(pol["c_proportional_to_gap"]["new_aggregate"] - T_EFFECTIVE) <= TOL)
    # no policy demands a coverage > 1.0 (every per-source target is physically reachable).
    c["09_no_policy_demands_cov_gt_one"] = bool(
        pol["a_mmlu_pro_only"]["all_cov_le_one"] and pol["b_uniform"]["all_cov_le_one"]
        and pol["c_proportional_to_gap"]["all_cov_le_one"])
    # mmlu_pro-only single-source jump ~ 0.070 (matches the PR's stated Delta).
    c["10_policy_a_mmlu_jump_0p070"] = bool(
        abs(pol["a_mmlu_pro_only"]["delta_per_source"]["mmlu_pro"] - 0.0696938) <= 1e-3)
    # uniform Delta == gap on every source.
    c["11_policy_b_uniform_eq_gap"] = bool(
        all(abs(v - gap) <= TOL for v in pol["b_uniform"]["delta_per_source"].values()))

    # (c) mmlu_pro is the binding source, and uniform is minimax-optimal.
    c["12_mmlu_pro_binding"] = bool(
        t["binding_source_overall"] == "mmlu_pro" and drag["binding_source"] == "mmlu_pro"
        and pol["a_mmlu_pro_only"]["binding_source"] == "mmlu_pro"
        and pol["c_proportional_to_gap"]["binding_source"] == "mmlu_pro")
    c["13_uniform_minimax_optimal"] = bool(
        t["uniform_is_minimax_optimal"]
        and abs(pol["b_uniform"]["max_single_source_lift"] - gap) <= TOL
        and pol["a_mmlu_pro_only"]["max_single_source_lift"] > gap + 1e-4)

    # (d) drag diagnosis: capacity-vs-data verdict present and is training-limited.
    c["14_drag_training_limited"] = bool(
        "TRAINING" in drag["capacity_vs_data_verdict"]
        and len(drag["capacity_vs_data_rationale"]) > 200
        and len(drag["token_classes_missed"]) >= 3)

    # (e) recipe ranking: strict total order, soft-KD #1, TTT last with the #316 root-vs-deep rationale.
    c["15_recipe_ranking_total_ordered"] = bool(rk["strict_total_order"] and len(rk["ranking"]) == 4)
    c["16_soft_kd_ranked_first"] = bool(rk["top_recipe"] == "soft_kd_topk_distill")
    c["17_ttt_ranked_last"] = bool(rk["ttt_ranked_last"] and rk["bottom_recipe"] == "on_policy_ttt")
    c["18_ttt_rationale_root_not_deep"] = bool(
        "ROOT" in rk["ttt_rationale_is_316_root_not_deep"]
        and "#316" in rk["ttt_rationale_is_316_root_not_deep"]
        and ("depth>=2" in rk["ttt_rationale_is_316_root_not_deep"]
             or "deep-spine" in rk["ttt_rationale_is_316_root_not_deep"]))
    # every recipe carries a literature citation.
    c["19_every_recipe_has_citation"] = bool(
        all(isinstance(r["citation"], str) and len(r["citation"]) > 40 for r in rk["ranking"]))

    # (f) the combination clears, the single lever is marginal -> the verdict is internally consistent.
    combo = rk["recommended_combination"]
    c["20_combination_central_clears"] = bool(combo["central_clears"])
    c["21_single_lever_marginal"] = bool(
        not combo["single_lever_soft_kd_central_clears"] and not combo["low_band_clears"])
    c["22_feasibility_verdict_marginal"] = bool(
        fv["feasibility_verdict"] == "REACHABLE-MARGINAL"
        and fv["feasibility_verdict"] in {"REACHABLE", "REACHABLE-MARGINAL", "OUT-OF-REACH"})

    # (g) banked-constant drift guard: imported per-source top-4 match the on-disk lawine #330 artifact.
    drift_ok = True
    if banked330:
        ps330 = (((banked330.get("synthesis", {}) or {}).get("step2_composition_to_coverage", {}) or {})
                 .get("per_source_top4", {}) or {})
        for k in FERN_SRC_TOP4:
            if ps330.get(k) is not None:
                drift_ok = drift_ok and abs(float(ps330[k]) - FERN_SRC_TOP4[k]) <= TOL_REPRO
        agg330 = (((banked330.get("synthesis", {}) or {}).get("step2_composition_to_coverage", {}) or {})
                  .get("point_estimate_uncond_top4"))
        if agg330 is not None:
            drift_ok = drift_ok and abs(float(agg330) - t["aggregate_baseline"]) <= TOL_REPRO
    c["23_constants_match_banked_330"] = bool(drift_ok and bool(banked330))

    gate = all(bool(v) for v in c.values())
    return {"coverage_lift_target_self_test_passes": gate, "checks": c}


# --------------------------------------------------------------------------- #
# NaN-clean walk.
# --------------------------------------------------------------------------- #
def assert_nan_clean(payload: dict) -> list[str]:
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

    walk(payload, "result")
    return bad


# --------------------------------------------------------------------------- #
# Determinism: two independent synthesize() passes must serialize identically.
# --------------------------------------------------------------------------- #
def determinism_ok() -> bool:
    a = synthesize()
    b = synthesize()
    a["test_metrics"]["coverage_lift_target_self_test_passes"] = True
    b["test_metrics"]["coverage_lift_target_self_test_passes"] = True
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# --------------------------------------------------------------------------- #
# W&B logging (summary/ namespace; robust; never fatal).
# --------------------------------------------------------------------------- #
def maybe_log_wandb(args, payload: dict) -> str | None:
    if getattr(args, "no_wandb", False) or not getattr(args, "wandb_name", None):
        return None
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_json_artifact, log_summary
    except Exception as exc:  # noqa: BLE001
        print(f"[eagle3-head-coverage-lift-target] wandb logging unavailable: {exc}", flush=True)
        return None

    syn = payload["synthesis"]
    t = syn["step1_lift_target_table"]
    rk = syn["step3_recipe_ranking"]
    fv = syn["step4_feasibility_verdict"]
    st = payload["self_test"]
    tm = syn["test_metrics"]
    pol = t["policies"]
    combo = rk["recommended_combination"]

    run = init_wandb_run(
        job_type="validity-analytic", agent="lawine", name=args.wandb_name, group=args.wandb_group,
        tags=["eagle3-head-coverage-lift-target", "validity-analytic", "coverage-lift", "eagle3",
              "demand-half", "recipe-ranking", "soft-kd", "feasibility", "bank-the-analysis"],
        config={
            "pr": 336, "build_bar_unconditional_top4": T_EFFECTIVE,
            "aggregate_baseline_top4": t["aggregate_baseline"],
            "official_src_counts": OFFICIAL_SRC_COUNTS,
            "fern34_per_source_top4": FERN_SRC_TOP4,
            "lit_top4_central": LIT_TOP4_CENTRAL, "baseline_tps": BASELINE_TPS,
            "combo_non_additivity_factor": COMBO_NON_ADDITIVITY,
            "wandb_group": args.wandb_group, "imports": syn["imported"]["provenance"],
        },
    )
    if run is None:
        print("[eagle3-head-coverage-lift-target] wandb: no run (no WANDB_API_KEY/mode) -- skipping",
              flush=True)
        return None

    summary: dict[str, Any] = {
        # PRIMARY + TEST + REPORT
        "coverage_lift_target_self_test_passes": int(bool(
            st["coverage_lift_target_self_test_passes"])),
        "min_aggregate_lift_required": tm["min_aggregate_lift_required"],
        "feasibility_verdict_reachable_marginal": int(fv["feasibility_verdict"] == "REACHABLE-MARGINAL"),
        # step1 lift target
        "aggregate_baseline_top4": t["aggregate_baseline"],
        "aggregate_gap": t["aggregate_gap"], "weighted_sum_lift": t["weighted_sum_lift"],
        "policy_a_mmlu_only_single_jump": pol["a_mmlu_pro_only"]["delta_per_source"]["mmlu_pro"],
        "policy_b_uniform_delta": pol["b_uniform"]["delta_per_source"]["mmlu_pro"],
        "policy_c_prop_mmlu_delta": pol["c_proportional_to_gap"]["delta_per_source"]["mmlu_pro"],
        "policy_c_prop_aime_delta": pol["c_proportional_to_gap"]["delta_per_source"]["aime"],
        "k_proportional": t["k_proportional"],
        "uniform_is_minimax_optimal": int(bool(t["uniform_is_minimax_optimal"])),
        # step3 recipe ranking
        "rank1_soft_kd_central": RECIPES["soft_kd_topk_distill"]["delta_cov_central"],
        "rank2_reasoning_data_central": RECIPES["more_reasoning_root_data"]["delta_cov_central"],
        "rank3_deeper_head_central": RECIPES["deeper_wider_head"]["delta_cov_central"],
        "rank4_ttt_central": RECIPES["on_policy_ttt"]["delta_cov_central"],
        "recipe_ranking_total_ordered": int(bool(rk["strict_total_order"])),
        "ttt_ranked_last": int(bool(rk["ttt_ranked_last"])),
        "combo_delta_cov_central": combo["delta_cov_central"],
        "combo_delta_cov_low": combo["delta_cov_band"][0],
        "combo_delta_cov_high": combo["delta_cov_band"][1],
        "combo_central_clears": int(bool(combo["central_clears"])),
        "single_lever_soft_kd_clears": int(bool(combo["single_lever_soft_kd_central_clears"])),
        # bars / context
        "build_bar_unconditional_top4": T_EFFECTIVE, "lit_top4_central": LIT_TOP4_CENTRAL,
        "baseline_tps": BASELINE_TPS,
        # hygiene
        "nan_clean": int(bool(payload["nan_clean"])),
        "determinism_ok": int(bool(payload["determinism_ok"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(val)) for k, val in st["checks"].items()},
    }
    summary = {k: val for k, val in summary.items()
               if not (isinstance(val, float) and not math.isfinite(val)) and val is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="eagle3_head_coverage_lift_target_result", artifact_type="validity",
                      data=payload)
    rid = getattr(run, "id", None)
    print(f"[eagle3-head-coverage-lift-target] wandb run: {getattr(run, 'url', rid)}", flush=True)
    finish_wandb(run)
    return rid


# --------------------------------------------------------------------------- #
# Console report.
# --------------------------------------------------------------------------- #
def print_report(payload: dict) -> None:
    syn = payload["synthesis"]
    t = syn["step1_lift_target_table"]
    drag = syn["step2_mmlu_drag_diagnosis"]
    rk = syn["step3_recipe_ranking"]
    fv = syn["step4_feasibility_verdict"]
    st = payload["self_test"]
    pol = t["policies"]
    print("\n" + "=" * 100, flush=True)
    print("EAGLE-3 HEAD COVERAGE LIFT TARGET (PR #336) -- what recipe lifts top-4 coverage +0.031 to 0.9213?",
          flush=True)
    print("=" * 100, flush=True)
    print("STEP 1 -- per-source lift-target table:", flush=True)
    print(f"  baseline per-source top-4: {{k: round(v,4)}} -> "
          f"{ {k: round(v, 4) for k, v in t['baseline_cov_per_source'].items()} }", flush=True)
    print(f"  aggregate {t['aggregate_baseline']:.4f}  bar {t['bar']:.4f}  GAP {t['aggregate_gap']:+.4f}  "
          f"(weighted-sum lift {t['weighted_sum_lift']:.2f})", flush=True)
    print(f"  min_aggregate_lift_required = {t['min_aggregate_lift_required']:.4f}", flush=True)
    for pk, pv in pol.items():
        dd = ", ".join(f"{k}:{v:+.4f}" for k, v in pv["delta_per_source"].items())
        print(f"   [{pk}] Delta[{dd}]  max {pv['max_single_source_lift']:.4f}  -> new_agg "
              f"{pv['new_aggregate']:.4f} (reaches_bar={pv['reaches_bar']})", flush=True)
    print(f"  uniform_is_minimax_optimal = {t['uniform_is_minimax_optimal']}", flush=True)
    print("-" * 100, flush=True)
    print("STEP 2 -- mmlu_pro drag diagnosis:", flush=True)
    print(f"  binding={drag['binding_source']} cov={drag['binding_cov']:.4f}  "
          f"aime-mmlu gap {drag['aime_minus_mmlu_gap']:+.4f}", flush=True)
    print(f"  VERDICT: {drag['capacity_vs_data_verdict']}", flush=True)
    print("-" * 100, flush=True)
    print("STEP 3 -- recipe ranking (by expected aggregate top-4 ROOT-coverage lift):", flush=True)
    for r in rk["ranking"]:
        print(f"   #{r['rank']} {r['recipe']}: central {r['delta_cov_central']:+.3f} "
              f"band [{r['delta_cov_band'][0]:+.3f},{r['delta_cov_band'][1]:+.3f}]  "
              f"clears_alone={r['central_clears_alone']}", flush=True)
    combo = rk["recommended_combination"]
    print(f"  COMBINATION {combo['recipe']}: central {combo['delta_cov_central']:+.4f} "
          f"band [{combo['delta_cov_band'][0]:+.4f},{combo['delta_cov_band'][1]:+.4f}]  "
          f"clears={combo['central_clears']} (low_clears={combo['low_band_clears']})", flush=True)
    print("-" * 100, flush=True)
    print(f"STEP 4 -- FEASIBILITY VERDICT: {fv['feasibility_verdict']}", flush=True)
    print(f"  cheapest clearing recipe: {fv['cheapest_clearing_recipe']}", flush=True)
    print(f"  {fv['rationale']}", flush=True)
    print("-" * 100, flush=True)
    print(f"(PRIMARY) coverage_lift_target_self_test_passes = "
          f"{st['coverage_lift_target_self_test_passes']}", flush=True)
    for k, val in st["checks"].items():
        print(f"   - {k}: {val}", flush=True)
    print(f"nan_clean = {payload['nan_clean']}   determinism_ok = {payload['determinism_ok']}   "
          f"peak_mem_mib = {payload['peak_mem_mib']}", flush=True)
    print("=" * 100 + "\n", flush=True)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="eagle3-head-coverage-lift-target")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    banked330 = load_banked_330()
    syn = synthesize()
    st = self_test(syn, banked330)
    syn["test_metrics"]["coverage_lift_target_self_test_passes"] = bool(
        st["coverage_lift_target_self_test_passes"])

    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "pr": 336, "agent": "lawine", "kind": "eagle3-head-coverage-lift-target",
        "eagle3_head_coverage_lift_target_analysis_only": True,
        "synthesis": syn, "self_test": st,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }

    nan_paths = assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[eagle3-head-coverage-lift-target] WARNING non-finite at: {nan_paths}", flush=True)
    payload["determinism_ok"] = determinism_ok()

    gate = bool(st["coverage_lift_target_self_test_passes"] and payload["nan_clean"]
                and payload["determinism_ok"])
    st["coverage_lift_target_self_test_passes"] = gate
    syn["test_metrics"]["coverage_lift_target_self_test_passes"] = gate

    print_report(payload)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eagle3_head_coverage_lift_target_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[eagle3-head-coverage-lift-target] wrote {out_path}", flush=True)

    rid = maybe_log_wandb(args, payload)
    payload["wandb_run_id"] = rid
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    tm = syn["test_metrics"]
    print(f"  PRIMARY coverage_lift_target_self_test_passes = {gate}", flush=True)
    print(f"  TEST min_aggregate_lift_required = {tm['min_aggregate_lift_required']:.4f}", flush=True)
    print(f"  REPORT feasibility_verdict = {tm['feasibility_verdict']}", flush=True)
    print(f"  wandb run = {rid}", flush=True)

    if args.self_test:
        print(f"[eagle3-head-coverage-lift-target] self-test {'PASS' if gate else 'FAIL'}", flush=True)
        return 0 if gate else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
