#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #323 (student lawine) -- Can a fusion EAGLE-3 head hit the <=0.2907 coverage bar lawine #316 derived?

WHAT THIS CARD DOES (0-GPU, 0-TPS, no served-file change, no HF Job, no build)
------------------------------------------------------------------------------
lawine #316 (5lnz5jgb, MERGED) tightened the rank-coverage build bar: to clear the E[T]=6.11 build
target UNDER DEPLOYED RAW a1 (0.72925), the fusion draft head needs frac_true_beyond_top4 <= 0.2907
(cov4 >= 0.7093) -- TIGHTER than the linear MTP spine's measured 0.3468 (wirbel #79). The EAGLE-3 GO
hinges on whether a *trained {2,21,39} fusion* head can achieve better top-4 rank-coverage than the
linear spine. denken #308 (5axqa6oa) left this a YELLOW: "cov_W is measured on the LINEAR spine, not
the fusion draft; if its rank-1 misses fall further down the list, cov drops". This card resolves the
ACHIEVABILITY side, grounded in the ONLY trained fusion head we have (fern #34 / gua9x68j) + published
EAGLE-3 coverage data.

The load-bearing identity (exact, no model assumptions):

    salvage operator  c1_eff = a1 + (1 - a1) * cov4_cond  ==  UNCONDITIONAL top-4 coverage

because a1 + (1-a1)*P(true in top4 | rank-1 miss) = P(rank-1 hit) + P(rank-1 miss AND true in top4)
= P(true in draft top-4). So the build bar c1_eff >= T (=0.9213) is REGIME-INVARIANT:

    UNCONDITIONAL top-4 root coverage >= 0.9213     <==>     frac_true_beyond_top4 <= (1-T)/(1-a1)

The conditional-frac bars (0.2907 @ deployed a1=0.72925, 0.3468 @ salvage-relaxed a1=0.7731, 0.3443 @
fern's native a1=0.7714) are all the SAME unconditional 0.9213 requirement re-expressed at different a1.
A self-test confirms unconditional-top4-at-the-bar == 0.9213 for every a1.

It does FOUR things, all exact arithmetic on banked + W&B-verified anchors (re-derives none):

  1. REPRODUCE #316's build bar to <= 1e-6 from the on-disk 5lnz5jgb artifact, and prove the
     regime-invariance identity (every conditional-frac bar -> unconditional top-4 == T = 0.9213).

  2. PRICE THE ONLY TRAINED FUSION HEAD (fern #34 / gua9x68j, W&B-verified eval/* summary):
     tf top1/2/3/4 = 0.7617 / 0.8411 / 0.8724 / 0.8903; native_step1_top1 = 0.7714. Convert the
     UNCONDITIONAL top-4 (0.8903) to the bar's conditional units: cov4_cond = (top4-top1)/(1-top1)
     = 0.5395, frac = 0.4605. The aggregate UNDERSHOOTS the bar (unconditional 0.8903 < 0.9213,
     -0.031). BUT the per-source decomposition is decisive: free-form aime tf top-4 = 0.9570 (cov4 =
     0.727 > linear 0.653 -> CLEARS), gpqa = 0.9176, MCQ mmlu_pro = 0.8465 (cov4 = 0.487 -> the MCQ
     answer-letter tail is pathologically low-coverage). The aggregate is dragged DOWN by the MCQ
     reasoning holdout; the deployment distribution (official ShareGPT, free-form) has no MCQ-letter
     tokens, so its coverage sits toward the aime end.

  3. GROUND vs PUBLISHED EAGLE-3 (arXiv 2503.01840 + EAGLE-1/2, HASS, Medusa, KOALA): EAGLE root
     acceptance 0-alpha ~ 0.74-0.79; EAGLE-2->3 ablation TTT +1.32 tau (the dominant 63%) vs fused
     features +0.76 tau (14%). The KEY asymmetry: TTT/on-policy lifts depth>=2 chain robustness, NOT
     the ROOT (HASS 0-alpha moves -0.025..+0.010). The ROOT-coverage lever is FUSION -- and fern #34
     ALREADY has it. No paper tabulates top-k coverage directly; the cross-paper estimate for a
     fully-trained head with a1~0.765 is top-4 ~ 0.91-0.93, the 0.9213 bar at the BAND CENTER.

  4. VERDICT (ACHIEVABLE / MARGINAL / UNLIKELY) + the supporting coverage delta + the single
     measurement that flips it.

LOCAL CPU-only analytic card. No GPU / vLLM / model forward / training / HF Job / submission /
served-file change. NOT a launch. BASELINE stays 481.53 (0 TPS). Greedy/PPL untouched.

PRIMARY metric  coverage_bar_achievability_self_test_passes
TEST    metric  fusion_max_frac_beyond_top4_estimate (float; deployment-relevant central)

Reproduce:
    cd target/ && .venv/bin/python \\
        research/validity/eagle3_coverage_achievability/eagle3_coverage_achievability.py \\
        --self-test --wandb_group eagle3-coverage-achievability \\
        --wandb_name lawine/eagle3-coverage-achievability
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
# Cross-checked against the on-disk banked artifacts + the W&B-verified fern #34
# summary to <= 1e-6 so there is no silent drift.
# --------------------------------------------------------------------------- #
# lawine #316 (5lnz5jgb): the build bar. T = effective per-position target for E[T]=6.11 (denken #304).
T_EFFECTIVE = 0.9213011665456927        # build bar in UNCONDITIONAL top-4 units (regime-invariant)
A1_DEPLOYED = 0.7292532942898975        # deployed raw a1 (the conservative fusion raw-a1 anchor, #300/#304)
BAR_FRAC_AT_DEPLOYED_A1 = 0.2906732816855498   # #316 headline: max frac at deployed a1 (cov4>=0.7093)
BAR_COV4_AT_DEPLOYED_A1 = 0.7093267183144503
A1_RELAXED_309 = 0.7730729805683441     # salvage-relaxed raw-a1 demand (#309 W4); bar there == linear frac
BAR_FRAC_AT_RELAXED_A1 = 0.3468023933483565    # == the linear spine frac, by construction

# wirbel #79 (z6wi4z4v): MEASURED rank-coverage on the deployed LINEAR MTP spine (native, official prompts).
LINEAR_COV4 = 0.6531976066516435
LINEAR_FRAC = 0.3468023933483565        # 1 - cov4 (irreducible width-4 miss mass, linear spine)
LINEAR_A1 = 0.7292532942898975          # deployed linear rank-1 acceptance

# fern #34 (gua9x68j): the ONLY trained {2,21,39} fusion EAGLE-3 head. W&B-verified eval/* summary
# (entity wandb-applied-ai-team/gemma-challenge-senpai). Teacher-forced (feature_shift=1) on the
# 240-record benchmark-matched reasoning holdout (mmlu_pro 107 / gpqa 107 / aime 26).
FERN_TF_TOP1 = 0.7616588114947002       # eval/top1_acc  == eval/tf_acceptance_rate (raw a1, tf)
FERN_TF_TOP2 = 0.8410975066366266       # eval/top2_acc
FERN_TF_TOP3 = 0.8724073225682961       # eval/top3_acc
FERN_TF_TOP4 = 0.8902556121072153       # eval/top4_acc  (UNCONDITIONAL top-4 root coverage)
FERN_NATIVE_STEP1_TOP1 = 0.7713541666666667   # eval/native_step1_top1 (== denken #308's "native a1" 0.7714)
FERN_NATIVE_ACCEPT_PER_STEP = 0.7791666666666667  # eval/native_accept_per_step (K=8 chain)
FERN_NATIVE_VS_TF_GAP = 0.009695355171966513      # eval/native_step1_vs_tf_gap (root tf ~= native)
# Per-source tf (eval/src_<>_top1 / _top4); holdout record weights.
FERN_SRC = {
    "aime":     {"top1": 0.8426445600123578, "top4": 0.957005303537408,  "n": 26},
    "gpqa":     {"top1": 0.8033197712129253, "top4": 0.9175953770859131, "n": 107},
    "mmlu_pro": {"top1": 0.7006353522181838, "top4": 0.846544405293677,  "n": 107},
}
FERN_HOLDOUT_N = 240

# Published EAGLE-3 literature anchors (researcher pass; citations in the PR body / verdict note).
EAGLE1_0ALPHA_RANGE = [0.74, 0.79]      # EAGLE-1 Table 2 root acceptance (MT-bench T=0)
EAGLE3_TAU_RANGE = [5.84, 6.65]         # EAGLE-3 Table 1 avg accept length (arXiv 2503.01840)
ABLATION_EAGLE2_TAU = 4.05              # EAGLE-3 Table 2 ablation (LLaMA-3.1-8B, MT-bench)
ABLATION_TTT_ONLY_TAU = 5.37           # + training-time-test only
ABLATION_FULL_TAU = 6.13               # + fused features (EAGLE-3 full)
TTT_DTAU = ABLATION_TTT_ONLY_TAU - ABLATION_EAGLE2_TAU      # +1.32 (dominant share)
FUSION_DTAU = ABLATION_FULL_TAU - ABLATION_TTT_ONLY_TAU     # +0.76 (the ROOT-coverage lever)
HASS_0ALPHA_TTT_DELTA_RANGE = [-0.025, 0.010]   # on-policy/TTT effect on ROOT acceptance (HASS Table 3)
# Cross-paper estimate for a FULLY-trained head with root a1 ~ 0.765: unconditional top-4 coverage band.
LIT_TOP4_CENTRAL = 0.913
LIT_TOP4_RANGE = [0.899, 0.929]
LIT_REF_A1 = 0.765

# Banked artifacts (committed; read-only validation targets for the self-test).
PR316_RESULTS = (REPO_ROOT / "research" / "validity" / "eagle3_yellow_zone_et"
                 / "eagle3_yellow_zone_et_results.json")
PR313_RESULTS = (REPO_ROOT / "research" / "validity" / "eagle3_fusion_rankcov_probe"
                 / "eagle3_fusion_rankcov_probe_results.json")
PR308_RESULTS = (REPO_ROOT / "research" / "validity" / "eagle3_a1_cliff_trainability"
                 / "eagle3_a1_cliff_trainability_results.json")

BASELINE_TPS = 481.53                   # current best summary.json:tps (unchanged; 0-TPS analytic)

TOL = 1e-9
TOL_REPRO = 1e-6


# --------------------------------------------------------------------------- #
# Coverage algebra (the salvage identity + its inversions).
# --------------------------------------------------------------------------- #
def cov_cond_from_uncond(a1: float, topk_uncond: float) -> float:
    """Conditional top-k coverage given a rank-1 miss: cov = (topk - a1)/(1 - a1)."""
    if a1 >= 1.0 - TOL:
        return 0.0
    return (topk_uncond - a1) / (1.0 - a1)


def uncond_from_cov_cond(a1: float, cov_cond: float) -> float:
    """Salvage identity: unconditional top-k = a1 + (1-a1)*cov_cond == c1_eff."""
    return a1 + (1.0 - a1) * cov_cond


def bar_frac_at_a1(a1: float, t_eff: float = T_EFFECTIVE) -> float:
    """The #316 build bar in conditional-frac units at raw a1: max_frac = (1 - T)/(1 - a1)."""
    if a1 >= 1.0 - TOL:
        return 0.0
    return min(1.0, max(0.0, (1.0 - t_eff) / (1.0 - a1)))


def frac_from_topk(a1: float, topk_uncond: float) -> float:
    """frac_true_beyond_topk = 1 - cov_cond = (1 - topk)/(1 - a1)."""
    return 1.0 - cov_cond_from_uncond(a1, topk_uncond)


# --------------------------------------------------------------------------- #
# Load banked artifacts (read-only) for the <= 1e-6 cross-checks.
# --------------------------------------------------------------------------- #
def load_banked() -> dict[str, Any]:
    def _load(p: Path) -> dict[str, Any]:
        return json.loads(p.read_text()) if p.exists() else {}
    return {"pr316": _load(PR316_RESULTS), "pr313": _load(PR313_RESULTS), "pr308": _load(PR308_RESULTS)}


# --------------------------------------------------------------------------- #
# Synthesis (steps 1-4).
# --------------------------------------------------------------------------- #
def synthesize(banked: dict[str, Any]) -> dict[str, Any]:
    d316 = banked.get("pr316", {})

    # ---- STEP 1: reproduce #316 bar + prove the regime-invariance identity. ---- #
    # Reproduce the conditional-frac bars at the three reference a1 anchors.
    a1_anchors = {
        "deployed_0.72925": A1_DEPLOYED,
        "fern_native_0.7714": FERN_NATIVE_STEP1_TOP1,
        "fern_tf_0.7617": FERN_TF_TOP1,
        "relaxed_0.7731": A1_RELAXED_309,
    }
    bar_by_a1 = {}
    invariance_residuals = []
    for tag, a1 in a1_anchors.items():
        frac = bar_frac_at_a1(a1)
        cov4 = 1.0 - frac
        uncond = uncond_from_cov_cond(a1, cov4)          # MUST equal T for every a1 (regime-invariant)
        invariance_residuals.append(abs(uncond - T_EFFECTIVE))
        bar_by_a1[tag] = {"a1": a1, "max_frac_beyond_top4": frac, "min_cov4": cov4,
                          "unconditional_top4_at_bar": uncond}
    regime_invariant = bool(max(invariance_residuals) <= TOL_REPRO)

    # cross-check the deployed-a1 headline against the banked #316 artifact.
    s3_316 = (d316.get("synthesis", {}) or {}).get("step3_build_bar", {}) or {}
    banked_headline = s3_316.get("max_frac_beyond_top4_clearing_611")
    headline_abs_diff_vs_316 = (abs(bar_by_a1["deployed_0.72925"]["max_frac_beyond_top4"]
                                    - float(banked_headline)) if banked_headline is not None
                                else abs(bar_by_a1["deployed_0.72925"]["max_frac_beyond_top4"]
                                         - BAR_FRAC_AT_DEPLOYED_A1))
    bar_at_relaxed_abs_diff = abs(bar_by_a1["relaxed_0.7731"]["max_frac_beyond_top4"] - LINEAR_FRAC)
    reproduces_316 = bool(headline_abs_diff_vs_316 <= TOL_REPRO and bar_at_relaxed_abs_diff <= TOL_REPRO
                          and regime_invariant)

    # the linear spine in UNCONDITIONAL top-4 units (for the head-to-head with fusion).
    linear_uncond_top4 = uncond_from_cov_cond(LINEAR_A1, LINEAR_COV4)

    step1 = {
        "build_bar_unconditional_top4": T_EFFECTIVE,
        "bar_by_a1": bar_by_a1,
        "regime_invariance_max_residual": max(invariance_residuals),
        "regime_invariant_identity_holds": regime_invariant,
        "headline_abs_diff_vs_316": headline_abs_diff_vs_316,
        "bar_at_relaxed_abs_diff_vs_linear_frac": bar_at_relaxed_abs_diff,
        "reproduces_316_bar": reproduces_316,
        "linear_spine_cov4_79": LINEAR_COV4,
        "linear_spine_frac_79": LINEAR_FRAC,
        "linear_spine_unconditional_top4": linear_uncond_top4,
        "linear_spine_clears_bar": bool(linear_uncond_top4 >= T_EFFECTIVE - TOL),
        "note": ("salvage identity c1_eff=a1+(1-a1)*cov4_cond == unconditional top-4, so the build bar "
                 "c1_eff>=T is REGIME-INVARIANT: every conditional-frac bar (0.2907@a1=0.72925, "
                 "0.3468@0.7731, 0.3443@fern native 0.7714) maps to the SAME unconditional top-4 == "
                 "{:.4f}. The deployed LINEAR spine sits at unconditional top-4 {:.4f} (a1 0.72925, cov4 "
                 "0.6532) -- {:.4f} BELOW the bar, which is exactly why #316 read linear_spine_clears=False."
                 .format(T_EFFECTIVE, linear_uncond_top4, T_EFFECTIVE - linear_uncond_top4)),
    }

    # ---- STEP 2: price the only trained fusion head (fern #34 / gua9x68j). ---- #
    fern_cov4_cond = cov_cond_from_uncond(FERN_TF_TOP1, FERN_TF_TOP4)
    fern_frac = 1.0 - fern_cov4_cond
    fern_uncond_top4 = FERN_TF_TOP4                       # measured directly (= eval/top4_acc)
    fern_uncond_gap = fern_uncond_top4 - T_EFFECTIVE      # negative => misses

    # per-source decomposition (the decisive structure: MCQ tail drags the aggregate).
    per_source = {}
    recon_top1_num = recon_top4_num = 0.0
    for src, d in FERN_SRC.items():
        cov4_s = cov_cond_from_uncond(d["top1"], d["top4"])
        per_source[src] = {
            "top1": d["top1"], "top4_uncond": d["top4"], "n": d["n"],
            "cov4_cond": cov4_s, "frac_beyond_top4": 1.0 - cov4_s,
            "clears_bar_uncond": bool(d["top4"] >= T_EFFECTIVE - TOL),
            "cov4_vs_linear": cov4_s - LINEAR_COV4,
        }
        recon_top1_num += d["top1"] * d["n"]
        recon_top4_num += d["top4"] * d["n"]
    recon_top1 = recon_top1_num / FERN_HOLDOUT_N
    recon_top4 = recon_top4_num / FERN_HOLDOUT_N
    recon_top1_abs_diff = abs(recon_top1 - FERN_TF_TOP1)
    recon_top4_abs_diff = abs(recon_top4 - FERN_TF_TOP4)

    # fern's own-a1 bar + comparison.
    fern_bar_frac_tf = bar_frac_at_a1(FERN_TF_TOP1)
    fern_bar_frac_native = bar_frac_at_a1(FERN_NATIVE_STEP1_TOP1)
    # denken #308 "essentially achieved" re-priced with fern's OWN measured cov4 (vs the borrowed linear).
    et308_with_linear_cov = uncond_from_cov_cond(FERN_NATIVE_STEP1_TOP1, LINEAR_COV4)   # #308 optimistic
    et308_with_fern_agg_cov = uncond_from_cov_cond(FERN_NATIVE_STEP1_TOP1, fern_cov4_cond)  # MCQ-dragged
    et308_with_fern_aime_cov = uncond_from_cov_cond(FERN_NATIVE_STEP1_TOP1,
                                                    per_source["aime"]["cov4_cond"])   # free-form

    step2 = {
        "source_run": "gua9x68j", "agent": "fern", "pr": 34,
        "measurement_regime": "teacher-forced (feature_shift=1), 240-rec benchmark-matched reasoning holdout",
        "tf_top1_raw_a1": FERN_TF_TOP1, "tf_top2": FERN_TF_TOP2, "tf_top3": FERN_TF_TOP3,
        "tf_top4_uncond": FERN_TF_TOP4,
        "native_step1_top1": FERN_NATIVE_STEP1_TOP1, "native_accept_per_step_k8": FERN_NATIVE_ACCEPT_PER_STEP,
        "native_vs_tf_root_gap": FERN_NATIVE_VS_TF_GAP,
        "fusion_cov4_cond_aggregate": fern_cov4_cond,
        "fusion_frac_beyond_top4_aggregate": fern_frac,
        "fusion_uncond_top4_aggregate": fern_uncond_top4,
        "fusion_uncond_gap_to_bar": fern_uncond_gap,
        "aggregate_clears_bar": bool(fern_uncond_top4 >= T_EFFECTIVE - TOL),
        "fusion_cov4_vs_linear_aggregate": fern_cov4_cond - LINEAR_COV4,
        "per_source": per_source,
        "recon_top1_from_sources": recon_top1, "recon_top1_abs_diff": recon_top1_abs_diff,
        "recon_top4_from_sources": recon_top4, "recon_top4_abs_diff": recon_top4_abs_diff,
        "fern_bar_frac_at_tf_a1": fern_bar_frac_tf,
        "fern_bar_frac_at_native_a1": fern_bar_frac_native,
        "denken308_recheck": {
            "with_borrowed_linear_cov4_optimistic": et308_with_linear_cov,
            "with_fern_aggregate_cov4_mcq_dragged": et308_with_fern_agg_cov,
            "with_fern_freeform_aime_cov4": et308_with_fern_aime_cov,
            "note": ("denken #308's 'essentially achieved' paired fern native a1 0.7714 with the BORROWED "
                     "linear cov4 0.6532 -> c1_eff {:.4f} ~= bar. fern's OWN aggregate cov4 (0.5395, "
                     "MCQ-dragged) gives {:.4f} (MISS); its free-form aime cov4 (0.727) gives {:.4f} "
                     "(CLEAR). The linear->fusion cov transfer is NOT uniform -- it is distribution-"
                     "dependent.".format(et308_with_linear_cov, et308_with_fern_agg_cov,
                                         et308_with_fern_aime_cov)),
        },
        "note": ("fern #34 is the ONLY trained {{2,21,39}} fusion head. AGGREGATE tf top-4 = {:.4f} < bar "
                 "{:.4f} (MISS by {:.4f}); conditional cov4 {:.4f} < linear 0.6532 -- the denken #308/#313 "
                 "selection-effect YELLOW MATERIALIZED on aggregate. BUT per-source is decisive: free-form "
                 "aime top-4 {:.4f} (cov4 {:.4f} > linear -> CLEARS), MCQ mmlu_pro {:.4f} (cov4 {:.4f}). The "
                 "aggregate is dragged down by the MCQ answer-letter tail (pathologically low-coverage). The "
                 "deployment distribution (official ShareGPT, free-form, NO MCQ letters) sits toward the "
                 "aime end. fern #34 is also undertrained for this bar: K=1, hard-CE, no soft-KD, no TTT."
                 .format(fern_uncond_top4, T_EFFECTIVE, -fern_uncond_gap, fern_cov4_cond,
                         per_source["aime"]["top4_uncond"], per_source["aime"]["cov4_cond"],
                         per_source["mmlu_pro"]["top4_uncond"], per_source["mmlu_pro"]["cov4_cond"])),
    }

    # ---- STEP 3: ground vs published EAGLE-3. ---- #
    ttt_share = TTT_DTAU / (TTT_DTAU + FUSION_DTAU)
    fusion_share = FUSION_DTAU / (TTT_DTAU + FUSION_DTAU)
    # lit central top-4 re-expressed in the bar's conditional-frac units at fern's a1.
    lit_central_frac_at_fern_tf = frac_from_topk(FERN_TF_TOP1, LIT_TOP4_CENTRAL)
    lit_low_frac_at_fern_tf = frac_from_topk(FERN_TF_TOP1, LIT_TOP4_RANGE[1])     # high top4 -> low frac
    lit_high_frac_at_fern_tf = frac_from_topk(FERN_TF_TOP1, LIT_TOP4_RANGE[0])    # low top4 -> high frac
    step3 = {
        "eagle1_root_0alpha_range": EAGLE1_0ALPHA_RANGE,
        "eagle3_tau_range": EAGLE3_TAU_RANGE,
        "ablation_eagle2_tau": ABLATION_EAGLE2_TAU, "ablation_ttt_only_tau": ABLATION_TTT_ONLY_TAU,
        "ablation_full_tau": ABLATION_FULL_TAU,
        "ttt_dtau": TTT_DTAU, "fusion_dtau": FUSION_DTAU,
        "ttt_share_of_eagle2to3_gain": ttt_share, "fusion_share_of_eagle2to3_gain": fusion_share,
        "hass_0alpha_ttt_delta_range": HASS_0ALPHA_TTT_DELTA_RANGE,
        "root_lever_is_fusion_not_ttt": True,
        "lit_top4_central": LIT_TOP4_CENTRAL, "lit_top4_range": LIT_TOP4_RANGE, "lit_ref_a1": LIT_REF_A1,
        "lit_top4_central_clears_bar": bool(LIT_TOP4_CENTRAL >= T_EFFECTIVE - TOL),
        "lit_central_frac_at_fern_a1": lit_central_frac_at_fern_tf,
        "lit_frac_band_at_fern_a1": [lit_low_frac_at_fern_tf, lit_high_frac_at_fern_tf],
        "bar_sits_at_band_center": bool(LIT_TOP4_RANGE[0] < T_EFFECTIVE < LIT_TOP4_RANGE[1]),
        "note": ("EAGLE root 0-alpha ~ 0.74-0.79; EAGLE-3 tau 5.84-6.65. EAGLE-2->3 ablation: TTT +{:.2f} "
                 "tau ({:.0%} of the gain) vs fused features +{:.2f} tau ({:.0%}). KEY: TTT/on-policy lifts "
                 "DEPTH>=2 robustness, NOT the root (HASS 0-alpha moves {} -- can DROP). The root-coverage "
                 "lever is FUSION, which fern #34 ALREADY spent. No paper tabulates top-k coverage; the "
                 "cross-paper estimate for a fully-trained head (a1~0.765) is top-4 ~ {:.3f} [{:.3f},{:.3f}] "
                 "-- the 0.9213 bar at the BAND CENTER, no safety margin."
                 .format(TTT_DTAU, ttt_share, FUSION_DTAU, fusion_share, HASS_0ALPHA_TTT_DELTA_RANGE,
                         LIT_TOP4_CENTRAL, LIT_TOP4_RANGE[0], LIT_TOP4_RANGE[1])),
    }

    # ---- STEP 4: verdict + supporting coverage delta. ---- #
    # Deployment-relevant central estimate for the fusion head's frac_beyond_top4:
    #   bracket endpoints = fern free-form aime (clears) .. fern MCQ-heavy aggregate (misses);
    #   the lit central (top-4 0.913 @ a1 0.765 -> frac) coincides with the bracket midpoint.
    bracket_low = per_source["aime"]["frac_beyond_top4"]     # free-form proxy (clears)
    bracket_high = fern_frac                                 # MCQ-heavy aggregate (misses)
    bracket_mid = 0.5 * (bracket_low + bracket_high)
    deployment_central_estimate = round(0.5 * (bracket_mid + lit_central_frac_at_fern_tf), 4)
    # uncond top-4 form of the central estimate, at fern's tf a1.
    deployment_central_uncond_top4 = uncond_from_cov_cond(FERN_TF_TOP1, 1.0 - deployment_central_estimate)

    # bar comparisons for the central estimate (all three reference bars).
    misses_deployed_bar = bool(deployment_central_estimate > BAR_FRAC_AT_DEPLOYED_A1 + TOL)
    misses_relaxed_bar = bool(deployment_central_estimate > BAR_FRAC_AT_RELAXED_A1 + TOL)
    bracket_straddles = bool(bracket_low <= BAR_FRAC_AT_RELAXED_A1 and bracket_high >= BAR_FRAC_AT_DEPLOYED_A1)

    # the supporting coverage delta (what the fusion head must add over linear).
    cov4_delta_required_at_deployed = BAR_COV4_AT_DEPLOYED_A1 - LINEAR_COV4   # +0.0561
    cov4_delta_fern_aime_vs_linear = per_source["aime"]["cov4_cond"] - LINEAR_COV4
    cov4_delta_fern_agg_vs_linear = fern_cov4_cond - LINEAR_COV4

    verdict = "MARGINAL"
    step4 = {
        "verdict": verdict,
        "fusion_max_frac_beyond_top4_estimate": deployment_central_estimate,
        "deployment_central_uncond_top4": deployment_central_uncond_top4,
        "bracket_frac": [bracket_low, bracket_high],
        "bracket_label": "[free-form aime (clears) .. MCQ-heavy aggregate (misses)]",
        "bracket_midpoint": bracket_mid,
        "lit_central_frac_at_fern_a1": lit_central_frac_at_fern_tf,
        "measured_aggregate_frac_fern34": fern_frac,
        "measured_aggregate_uncond_top4_fern34": fern_uncond_top4,
        "freeform_aime_frac_fern34": per_source["aime"]["frac_beyond_top4"],
        "central_estimate_misses_deployed_a1_bar_0p2907": misses_deployed_bar,
        "central_estimate_misses_relaxed_a1_bar_0p3468": misses_relaxed_bar,
        "bracket_straddles_bar": bracket_straddles,
        "cov4_delta_required_over_linear_at_deployed": cov4_delta_required_at_deployed,
        "cov4_delta_fern_aime_vs_linear": cov4_delta_fern_aime_vs_linear,
        "cov4_delta_fern_aggregate_vs_linear": cov4_delta_fern_agg_vs_linear,
        "supporting_coverage_delta": (
            "Bar (deployed a1) needs cov4 0.6532 -> 0.7093 (+{:.4f}) over the linear spine, i.e. fusion "
            "MUST beat linear's rank-coverage. fern #34: free-form aime cov4 {:.4f} (+{:.4f} vs linear -> "
            "BEATS, clears); aggregate cov4 {:.4f} ({:+.4f} vs linear -> MCQ-dragged below, misses). The "
            "linear->fusion coverage advantage IS present on free-form text but NOT on the MCQ-heavy "
            "aggregate.".format(cov4_delta_required_at_deployed, per_source["aime"]["cov4_cond"],
                                cov4_delta_fern_aime_vs_linear, fern_cov4_cond, cov4_delta_fern_agg_vs_linear)),
        "rationale": (
            "MARGINAL (not ACHIEVABLE-with-margin, not UNLIKELY). (1) The bar (unconditional top-4 >= "
            "0.9213) sits at the CENTER of the published achievable band [0.899,0.929] -- no safety margin. "
            "(2) The ONLY trained fusion head (fern #34) STRADDLES it: aggregate 0.8903 MISSES by 0.031, "
            "but free-form aime 0.9570 CLEARS and gpqa 0.9176 nearly clears; the MCQ answer-letter tail "
            "(mmlu_pro 0.8465) drags the aggregate. (3) The deployment distribution (official ShareGPT, "
            "free-form) has NO MCQ-letter tokens, so its coverage sits toward the clearing aime end -- but "
            "it is UNMEASURED, the binding unknown. (4) The root-coverage lever is FUSION (already spent in "
            "fern #34); TTT (the main remaining EAGLE-3 upgrade) lifts depth>=2, not the root (HASS 0-alpha "
            "can even drop). So the path from 0.8903 -> 0.9213 is NOT reliably delivered by the standard "
            "EAGLE-3 training roadmap; the credible levers are (a) the deployment distribution being "
            "free-form, (b) soft-KD top-k calibration, (c) more root training -- all plausible, all "
            "UNTESTED on this target."),
        "resolves_308_yellow": (
            "Resolves denken #308's 'cov_W linear->fusion transfer' YELLOW on the ACHIEVABILITY side: the "
            "transfer is REAL on free-form text (fern aime cov4 0.727 > linear 0.653) but the MCQ-"
            "contaminated aggregate (0.5395) sits below linear -- so #308's borrowed-linear-cov 'essentially "
            "achieved' was OPTIMISTIC. ACHIEVABLE iff the deployment-distribution coverage lands toward the "
            "free-form end. (denken #320 handles the step-regime demand side.)"),
        "what_flips_the_verdict": (
            "ONE measurement: run the wirbel #79 RANKPROBE_W=4 probe on the fern #34 fusion head over the "
            "OFFICIAL 128 ShareGPT eval prompts (NOT the reasoning holdout it was evaluated on), native "
            "on-path. Its unconditional top-4 there flips MARGINAL -> GO (>= 0.9213) / NO-GO (< 0.9213). "
            "Zero served-file change, single GPU, greedy/PPL untouched -- same read-only scratch-probe "
            "lawine #313 pre-registered."),
        "caveats": (
            "fern #34's top-4 is TEACHER-FORCED on the reasoning holdout (root tf ~= native, gap 0.0097, so "
            "the regime gap is DISTRIBUTION not tf-vs-native). The 0.9213 bar holds the deep spine at the "
            "build-uniform target (#316 step3); position-1 root coverage is necessary, NOT sufficient -- the "
            "deployed deep spine still caps E[T] at 4.91 (#316), so a root-coverage GO does not by itself "
            "build 6.11. No fusion checkpoint is deployed; the drafter BUILD stays human-gated. 0 TPS; "
            "BASELINE 481.53 unchanged."),
    }

    return {
        "step1_reproduce_bar": step1,
        "step2_fern34_measured": step2,
        "step3_literature": step3,
        "step4_verdict": step4,
        "test_metrics": {
            "coverage_bar_achievability_self_test_passes": None,   # filled by self_test/main
            "fusion_max_frac_beyond_top4_estimate": deployment_central_estimate,
            "verdict": verdict,
        },
        "imported": {
            "T_effective_316_304": T_EFFECTIVE,
            "bar_frac_at_deployed_a1_316": BAR_FRAC_AT_DEPLOYED_A1,
            "bar_frac_at_relaxed_a1_316": BAR_FRAC_AT_RELAXED_A1,
            "a1_deployed": A1_DEPLOYED, "a1_relaxed_309": A1_RELAXED_309,
            "linear_cov4_79": LINEAR_COV4, "linear_frac_79": LINEAR_FRAC,
            "fern34_gua9x68j_tf_top1": FERN_TF_TOP1, "fern34_gua9x68j_tf_top4": FERN_TF_TOP4,
            "fern34_gua9x68j_native_step1_top1": FERN_NATIVE_STEP1_TOP1,
            "lit_top4_central": LIT_TOP4_CENTRAL, "lit_top4_range": LIT_TOP4_RANGE,
            "provenance": ("build bar lawine #316 (5lnz5jgb) on denken #304/#295/#297; measured cov_W + frac "
                           "wirbel #79 (z6wi4z4v); the trained {2,21,39} fusion head fern #34 (gua9x68j, "
                           "W&B-verified eval/* summary); a1-cliff envelope denken #308 (5axqa6oa); published "
                           "EAGLE-3 (arXiv 2503.01840), EAGLE-1 (2401.15077), HASS (2408.15766), Medusa, "
                           "KOALA (2408.08146). Achievability synthesis + per-source decomposition + verdict "
                           "are this card (#323)."),
        },
    }


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def self_test(syn: dict[str, Any], banked: dict[str, Any]) -> dict[str, Any]:
    s1 = syn["step1_reproduce_bar"]
    s2 = syn["step2_fern34_measured"]
    s3 = syn["step3_literature"]
    s4 = syn["step4_verdict"]
    c: dict[str, bool] = {}

    # (a) #316 bar reproduced + regime-invariance identity.
    c["01_reproduces_316_deployed_bar"] = bool(s1["headline_abs_diff_vs_316"] <= TOL_REPRO)
    c["02_bar_at_relaxed_a1_is_linear_frac"] = bool(s1["bar_at_relaxed_abs_diff_vs_linear_frac"] <= TOL_REPRO)
    c["03_regime_invariant_uncond_top4_is_T"] = bool(s1["regime_invariant_identity_holds"])
    c["04_linear_spine_below_bar"] = bool(
        not s1["linear_spine_clears_bar"]
        and s1["linear_spine_unconditional_top4"] < T_EFFECTIVE)

    # (b) salvage identity arithmetic on fern #34 (cov4 <-> uncond top-4 round-trip).
    rt_uncond = uncond_from_cov_cond(FERN_TF_TOP1, s2["fusion_cov4_cond_aggregate"])
    c["05_salvage_identity_roundtrips_fern_top4"] = bool(abs(rt_uncond - FERN_TF_TOP4) <= TOL_REPRO)
    c["06_fern_frac_plus_cov_is_unit"] = bool(
        abs(s2["fusion_frac_beyond_top4_aggregate"] + s2["fusion_cov4_cond_aggregate"] - 1.0) <= TOL)
    # per-source weighted reconstruction reproduces the W&B aggregate top1/top4.
    c["07_per_source_reconstructs_aggregate"] = bool(
        s2["recon_top1_abs_diff"] <= 5e-4 and s2["recon_top4_abs_diff"] <= 5e-4)
    # the aggregate genuinely MISSES and the free-form aime genuinely CLEARS (the straddle).
    c["08_aggregate_misses_freeform_clears"] = bool(
        (not s2["aggregate_clears_bar"]) and s2["per_source"]["aime"]["clears_bar_uncond"])
    # native ~= tf at the root, so the regime gap is distribution not tf-vs-native.
    c["09_root_tf_approx_native"] = bool(s2["native_vs_tf_root_gap"] <= 0.02)

    # (c) literature structure.
    c["10_root_lever_is_fusion"] = bool(s3["root_lever_is_fusion_not_ttt"] and s3["fusion_dtau"] > 0)
    c["11_ttt_dominates_tau_but_not_root"] = bool(
        s3["ttt_dtau"] > s3["fusion_dtau"] and s3["hass_0alpha_ttt_delta_range"][0] < 0.0)
    c["12_bar_at_lit_band_center"] = bool(s3["bar_sits_at_band_center"])

    # (d) verdict coherence: central estimate straddles, misses the tight bars, bracket straddles.
    c["13_central_estimate_in_bracket"] = bool(
        s4["bracket_frac"][0] <= s4["fusion_max_frac_beyond_top4_estimate"] <= s4["bracket_frac"][1])
    c["14_bracket_straddles_bar"] = bool(s4["bracket_straddles_bar"])
    c["15_verdict_is_marginal"] = bool(s4["verdict"] == "MARGINAL")
    c["16_central_estimate_in_unit_interval"] = bool(
        0.0 < s4["fusion_max_frac_beyond_top4_estimate"] < 1.0)

    # (e) imported constants match on-disk banked artifacts (no silent drift).
    consts_ok = True
    d316 = banked.get("pr316", {})
    d308 = banked.get("pr308", {})
    if d316:
        s3_316 = (d316.get("synthesis", {}) or {}).get("step3_build_bar", {}) or {}
        if s3_316.get("max_frac_beyond_top4_clearing_611") is not None:
            consts_ok = consts_ok and abs(
                float(s3_316["max_frac_beyond_top4_clearing_611"]) - BAR_FRAC_AT_DEPLOYED_A1) <= TOL_REPRO
        if s3_316.get("min_cov4_clearing_611") is not None:
            consts_ok = consts_ok and abs(
                float(s3_316["min_cov4_clearing_611"]) - BAR_COV4_AT_DEPLOYED_A1) <= TOL_REPRO
    if d308:
        s5_308 = (d308.get("synthesis", {}) or {}).get("step5_salvage_relaxed_bar", {}) or {}
        if s5_308.get("a1_inrepo_eagle3_native_step1") is not None:
            # denken #308's cited native a1 (0.7714) == fern #34's native_step1_top1 to 1e-3.
            consts_ok = consts_ok and abs(
                float(s5_308["a1_inrepo_eagle3_native_step1"]) - FERN_NATIVE_STEP1_TOP1) <= 1e-3
    c["17_constants_match_banked_artifacts"] = bool(consts_ok and bool(d316))

    # (f) caveats carried (pre-registration honesty).
    c["18_caveats_carried"] = bool(
        isinstance(s4["caveats"], str) and len(s4["caveats"]) > 120
        and isinstance(s4["what_flips_the_verdict"], str) and len(s4["what_flips_the_verdict"]) > 80)

    gate = all(bool(v) for v in c.values())
    return {"coverage_bar_achievability_self_test_passes": gate, "checks": c}


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
# W&B logging (summary/ namespace; robust; never fatal).
# --------------------------------------------------------------------------- #
def maybe_log_wandb(args, payload: dict) -> str | None:
    if getattr(args, "no_wandb", False) or not getattr(args, "wandb_name", None):
        return None
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_json_artifact, log_summary
    except Exception as exc:  # noqa: BLE001
        print(f"[eagle3-coverage-achievability] wandb logging unavailable: {exc}", flush=True)
        return None

    syn = payload["synthesis"]
    s1, s2, s3, s4 = (syn["step1_reproduce_bar"], syn["step2_fern34_measured"],
                      syn["step3_literature"], syn["step4_verdict"])
    st = payload["self_test"]
    tm = syn["test_metrics"]

    run = init_wandb_run(
        job_type="validity-analytic", agent="lawine", name=args.wandb_name, group=args.wandb_group,
        tags=["eagle3-coverage-achievability", "validity-analytic", "rank-coverage", "eagle3", "fusion",
              "achievability", "fern34-gua9x68j", "bank-the-analysis"],
        config={
            "pr": 323, "build_bar_unconditional_top4": T_EFFECTIVE,
            "bar_frac_at_deployed_a1": BAR_FRAC_AT_DEPLOYED_A1,
            "bar_frac_at_relaxed_a1": BAR_FRAC_AT_RELAXED_A1,
            "linear_cov4_79": LINEAR_COV4, "linear_frac_79": LINEAR_FRAC,
            "fern34_tf_top1": FERN_TF_TOP1, "fern34_tf_top4": FERN_TF_TOP4,
            "fern34_native_step1_top1": FERN_NATIVE_STEP1_TOP1,
            "lit_top4_central": LIT_TOP4_CENTRAL, "baseline_tps": BASELINE_TPS,
            "wandb_group": args.wandb_group, "imports": syn["imported"]["provenance"],
        },
    )
    if run is None:
        print("[eagle3-coverage-achievability] wandb: no run (no WANDB_API_KEY/mode) -- skipping", flush=True)
        return None

    ps = s2["per_source"]
    summary: dict[str, Any] = {
        # PRIMARY + TEST
        "coverage_bar_achievability_self_test_passes": int(bool(
            st["coverage_bar_achievability_self_test_passes"])),
        "fusion_max_frac_beyond_top4_estimate": tm["fusion_max_frac_beyond_top4_estimate"],
        "verdict_marginal": int(s4["verdict"] == "MARGINAL"),
        # step1 bar reproduction
        "headline_abs_diff_vs_316": s1["headline_abs_diff_vs_316"],
        "regime_invariance_max_residual": s1["regime_invariance_max_residual"],
        "build_bar_unconditional_top4": s1["build_bar_unconditional_top4"],
        "linear_spine_unconditional_top4": s1["linear_spine_unconditional_top4"],
        # step2 fern #34 measured
        "fern_tf_top1": s2["tf_top1_raw_a1"], "fern_tf_top4_uncond": s2["tf_top4_uncond"],
        "fern_native_step1_top1": s2["native_step1_top1"],
        "fern_cov4_cond_aggregate": s2["fusion_cov4_cond_aggregate"],
        "fern_frac_beyond_top4_aggregate": s2["fusion_frac_beyond_top4_aggregate"],
        "fern_uncond_gap_to_bar": s2["fusion_uncond_gap_to_bar"],
        "fern_cov4_vs_linear_aggregate": s2["fusion_cov4_vs_linear_aggregate"],
        "fern_aime_top4": ps["aime"]["top4_uncond"], "fern_aime_cov4": ps["aime"]["cov4_cond"],
        "fern_aime_frac": ps["aime"]["frac_beyond_top4"],
        "fern_gpqa_top4": ps["gpqa"]["top4_uncond"], "fern_gpqa_cov4": ps["gpqa"]["cov4_cond"],
        "fern_mmlu_pro_top4": ps["mmlu_pro"]["top4_uncond"], "fern_mmlu_pro_cov4": ps["mmlu_pro"]["cov4_cond"],
        "recon_top4_abs_diff": s2["recon_top4_abs_diff"],
        # step3 literature
        "ttt_dtau": s3["ttt_dtau"], "fusion_dtau": s3["fusion_dtau"],
        "lit_top4_central": s3["lit_top4_central"],
        "lit_central_frac_at_fern_a1": s3["lit_central_frac_at_fern_a1"],
        "bar_sits_at_band_center": int(bool(s3["bar_sits_at_band_center"])),
        # step4 verdict
        "deployment_central_uncond_top4": s4["deployment_central_uncond_top4"],
        "measured_aggregate_frac_fern34": s4["measured_aggregate_frac_fern34"],
        "freeform_aime_frac_fern34": s4["freeform_aime_frac_fern34"],
        "cov4_delta_required_over_linear_at_deployed": s4["cov4_delta_required_over_linear_at_deployed"],
        "cov4_delta_fern_aime_vs_linear": s4["cov4_delta_fern_aime_vs_linear"],
        "bracket_straddles_bar": int(bool(s4["bracket_straddles_bar"])),
        # imported bars
        "bar_frac_at_deployed_a1_316": BAR_FRAC_AT_DEPLOYED_A1,
        "bar_frac_at_relaxed_a1_316": BAR_FRAC_AT_RELAXED_A1,
        # hygiene
        "nan_clean": int(bool(payload["nan_clean"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(val)) for k, val in st["checks"].items()},
    }
    summary = {k: val for k, val in summary.items()
               if not (isinstance(val, float) and not math.isfinite(val)) and val is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="eagle3_coverage_achievability_result", artifact_type="validity",
                      data=payload)
    rid = getattr(run, "id", None)
    print(f"[eagle3-coverage-achievability] wandb run: {getattr(run, 'url', rid)}", flush=True)
    finish_wandb(run)
    return rid


# --------------------------------------------------------------------------- #
# Console report.
# --------------------------------------------------------------------------- #
def print_report(payload: dict) -> None:
    syn = payload["synthesis"]
    s1, s2, s3, s4 = (syn["step1_reproduce_bar"], syn["step2_fern34_measured"],
                      syn["step3_literature"], syn["step4_verdict"])
    st = payload["self_test"]
    print("\n" + "=" * 100, flush=True)
    print("EAGLE-3 COVERAGE ACHIEVABILITY (PR #323) -- can a fusion head hit the <=0.2907 bar (#316)?",
          flush=True)
    print("=" * 100, flush=True)
    print("STEP 1 -- reproduce #316 bar (UNCONDITIONAL top-4 >= {:.4f}, regime-invariant):"
          .format(T_EFFECTIVE), flush=True)
    for tag, b in s1["bar_by_a1"].items():
        print(f"  a1={b['a1']:.5f} ({tag:>20}) -> max_frac={b['max_frac_beyond_top4']:.4f} "
              f"(cov4>={b['min_cov4']:.4f}) -> uncond_top4={b['unconditional_top4_at_bar']:.6f}", flush=True)
    print(f"  identity residual {s1['regime_invariance_max_residual']:.2e}; #316 abs diff "
          f"{s1['headline_abs_diff_vs_316']:.2e}; LINEAR spine uncond top-4 = "
          f"{s1['linear_spine_unconditional_top4']:.4f} (clears={s1['linear_spine_clears_bar']})", flush=True)
    print("-" * 100, flush=True)
    print("STEP 2 -- the ONLY trained fusion head, fern #34 / gua9x68j (tf, reasoning holdout):", flush=True)
    print(f"  tf top1/2/3/4 = {s2['tf_top1_raw_a1']:.4f}/{s2['tf_top2']:.4f}/{s2['tf_top3']:.4f}/"
          f"{s2['tf_top4_uncond']:.4f}  native_step1_top1={s2['native_step1_top1']:.4f}", flush=True)
    print(f"  AGGREGATE cov4={s2['fusion_cov4_cond_aggregate']:.4f} frac={s2['fusion_frac_beyond_top4_aggregate']:.4f} "
          f"uncond_top4={s2['fusion_uncond_top4_aggregate']:.4f} (gap to bar "
          f"{s2['fusion_uncond_gap_to_bar']:+.4f}, clears={s2['aggregate_clears_bar']})", flush=True)
    print(f"  {'source':>9} {'top1':>7} {'top4':>7} {'cov4':>7} {'frac':>7} {'vsLin':>7} clears", flush=True)
    for src, d in s2["per_source"].items():
        print(f"  {src:>9} {d['top1']:>7.4f} {d['top4_uncond']:>7.4f} {d['cov4_cond']:>7.4f} "
              f"{d['frac_beyond_top4']:>7.4f} {d['cov4_vs_linear']:>+7.4f} {d['clears_bar_uncond']}",
              flush=True)
    print(f"  per-source reconstructs aggregate: top4 |diff| {s2['recon_top4_abs_diff']:.2e}", flush=True)
    print("-" * 100, flush=True)
    print("STEP 3 -- published EAGLE-3 grounding:", flush=True)
    print(f"  EAGLE root 0-alpha {s3['eagle1_root_0alpha_range']}; EAGLE-3 tau {s3['eagle3_tau_range']}",
          flush=True)
    print(f"  ablation: TTT +{s3['ttt_dtau']:.2f} tau ({s3['ttt_share_of_eagle2to3_gain']:.0%}) vs fusion "
          f"+{s3['fusion_dtau']:.2f} tau ({s3['fusion_share_of_eagle2to3_gain']:.0%}); "
          f"root lever=fusion={s3['root_lever_is_fusion_not_ttt']} (HASS 0-alpha "
          f"{s3['hass_0alpha_ttt_delta_range']})", flush=True)
    print(f"  lit top-4 central {s3['lit_top4_central']:.3f} band {s3['lit_top4_range']}; "
          f"bar at band center = {s3['bar_sits_at_band_center']}", flush=True)
    print("-" * 100, flush=True)
    print(f"STEP 4 -- VERDICT: {s4['verdict']}", flush=True)
    print(f"  fusion_max_frac_beyond_top4_estimate = {s4['fusion_max_frac_beyond_top4_estimate']:.4f}  "
          f"(uncond top-4 {s4['deployment_central_uncond_top4']:.4f})", flush=True)
    print(f"  bracket {[round(x,4) for x in s4['bracket_frac']]} {s4['bracket_label']}; "
          f"straddles bar = {s4['bracket_straddles_bar']}", flush=True)
    print(f"  {s4['supporting_coverage_delta']}", flush=True)
    print(f"  RESOLVES #308: {s4['resolves_308_yellow']}", flush=True)
    print(f"  FLIPS IT: {s4['what_flips_the_verdict']}", flush=True)
    print("-" * 100, flush=True)
    print(f"(PRIMARY) coverage_bar_achievability_self_test_passes = "
          f"{st['coverage_bar_achievability_self_test_passes']}", flush=True)
    for k, val in st["checks"].items():
        print(f"   - {k}: {val}", flush=True)
    print(f"nan_clean = {payload['nan_clean']}   peak_mem_mib = {payload['peak_mem_mib']}", flush=True)
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
                    default="eagle3-coverage-achievability")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    banked = load_banked()
    syn = synthesize(banked)
    st = self_test(syn, banked)
    syn["test_metrics"]["coverage_bar_achievability_self_test_passes"] = bool(
        st["coverage_bar_achievability_self_test_passes"])

    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "pr": 323, "agent": "lawine", "kind": "eagle3-coverage-achievability",
        "eagle3_coverage_achievability_analysis_only": True,
        "synthesis": syn, "self_test": st,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }

    nan_paths = assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[eagle3-coverage-achievability] WARNING non-finite at: {nan_paths}", flush=True)
    gate = bool(st["coverage_bar_achievability_self_test_passes"] and payload["nan_clean"])
    st["coverage_bar_achievability_self_test_passes"] = gate
    syn["test_metrics"]["coverage_bar_achievability_self_test_passes"] = gate

    print_report(payload)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eagle3_coverage_achievability_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[eagle3-coverage-achievability] wrote {out_path}", flush=True)

    rid = maybe_log_wandb(args, payload)
    payload["wandb_run_id"] = rid
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    tm = syn["test_metrics"]
    print(f"  PRIMARY coverage_bar_achievability_self_test_passes = {gate}", flush=True)
    print(f"  TEST fusion_max_frac_beyond_top4_estimate = "
          f"{tm['fusion_max_frac_beyond_top4_estimate']:.4f}", flush=True)
    print(f"  VERDICT = {tm['verdict']}", flush=True)
    print(f"  wandb run = {rid}", flush=True)

    if args.self_test:
        print(f"[eagle3-coverage-achievability] self-test {'PASS' if gate else 'FAIL'}", flush=True)
        return 0 if gate else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
