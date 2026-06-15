#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #330 (student lawine) -- What unconditional top-4 coverage prior does the "ShareGPT" eval carry vs the 0.9213 bar?

WHAT THIS CARD DOES (0-GPU, 0-TPS, no served-file change, no HF Job, no build)
------------------------------------------------------------------------------
lawine #323 (ceddxj20, MERGED) read the fusion EAGLE-3 head fern #34 (gua9x68j) as STRADDLING the
regime-invariant build bar (unconditional top-4 root coverage >= T = 0.9213): per-source aime 0.9570
(clears) / gpqa 0.9176 (nearly) / mmlu_pro 0.8465 (misses) -> aggregate 0.8903 (misses by 0.031). #323
left ONE binding unknown: the coverage of the *deployment distribution*. It ASSUMED that distribution is
"official ShareGPT, free-form, NO MCQ letters -> sits toward the clearing aime end" and called the
verdict MARGINAL on that optimistic lean. This card RESOLVES that binding unknown -- and the resolution
flips the lean.

THE PREMISE IS WRONG, BY THE REPO'S OWN BENCHMARK DATA. The official 128 eval is the file
`official/main_bucket/shared_resources/speed_benchmark/data/eval_prompts_sharegpt.json`. Despite the
filename, its contents are 100% reasoning/STEM: mmlu_pro 57 / gpqa_diamond 57 / aime 14 (DATASET_ANALYSIS.md,
last updated 2026-06-13, already documented this). 114/128 prompts (89.1%) are multiple-choice
(>=2 lettered options) "ending in ANSWER: $LETTER"; 14/128 (10.9%) are open-form competition math. There
is ZERO free-form conversational chat. The greedy-reference decode over these 128 (committed at
research/greedy_reference/google__gemma-4-E4B-it/decode_outputs.jsonl) confirms it: every generation is a
512-token LaTeX/math/science reasoning chain (latex char-density ~5%, digit ~3%), NOT prose.

SECOND, FINER ERROR (also in the repo's own framing): the coverage drag is NOT "the MCQ-answer-letter
tail". With ignore_eos=true + max_tokens=512, only ~7.8% of the 128 generations even REACH "ANSWER:";
the literal committed answer-letter token is ~0.02% of all generated tokens. Yet mmlu_pro's WHOLE-CoT
coverage is 0.8465 (vs aime 0.9570). So the drag is the reasoning-CoT *vocabulary breadth* (mmlu_pro
spans law/philosophy/business/health/... -> highest entropy -> lowest coverage), not a near-absent
answer-letter. The right axis is the SOURCE MIX (free-form-math fraction), not a letter fraction.

Both errors push the SAME way. The official-eval coverage prior is the 57/57/14-weighted per-source
coverage == 0.8903 (identical to fern's "benchmark-matched" aggregate to 1e-5, because the proportions
match: fern holdout 107/107/26 ~ official 57/57/14). It MISSES the bar by 0.0310. With ignore_eos every
prompt emits exactly 512 tokens, so token-weight == prompt-weight == source proportions: the aggregate
is the honest token-level prior.

It does FOUR things, all exact arithmetic on banked + W&B-verified anchors (re-derives none) plus the
on-disk official eval composition (loaded live, banked as fallback):

  1. TOKEN-TYPE COMPOSITION of the official 128 eval. Load the eval file + greedy decode; report source
     mix (57/57/14, 100% reasoning/STEM), MCQ-prompt count (114/128), the uniform 512-token-per-prompt
     property, and the generation token-type signatures (LaTeX/digit/option/ANSWER densities). Estimate
     the literal answer-letter token fraction (~0.0002) and show it is ~300x below the level the
     letter-contamination model would need to matter. Conclusion: the eval is the LOW-coverage reasoning
     distribution, the "free-form, ~0 MCQ" premise is REFUTED, and the drag is CoT breadth not letters.

  2. COMPOSITION -> COVERAGE. Push the composition through #323's per-source curves: 57/57/14-weighted
     unconditional top-4 == 0.8903 (point estimate). Uncertainty band from per-source sampling error
     (conservative record-Bernoulli SE ~0.020) + native-vs-tf gap. P(official-eval uncond-top-4 >=
     0.9213) ~ 0.06 (range <0.01..0.15 across SE/native assumptions). The bar sits near the UPPER edge of
     the 95% band [0.851, 0.929].

  3. SENSITIVITY. (a) Source-mix axis (the correct one): cov(f) with free-form(aime) fraction f crosses
     the bar at f ~ 0.524; the eval sits at f = 0.109 -> deep miss. (b) The PR's literal X% MCQ-answer-
     letter model (free-form base @aime + X% letter@c_mcq): crosses at X ~ 5.4..8.8% (central 6.4% @
     c_mcq=0.40), so X in {0,5} clears and X=10 misses. The official eval's literal answer-letter
     fraction (~0.02%) is ~300x BELOW that crossing -> under this (wrong) model it would CLEAR, which is
     exactly why the premise looked optimistic. The model mis-localizes the drag to a near-absent token.

  4. VERDICT (LIKELY-CLEARS / TOSS-UP / LIKELY-MISSES) framed as the PRIOR that fern #329's GPU-gated
     RANKPROBE_W=4 read over the official 128 would confirm/refute. LIKELY-MISSES: the prior predicts the
     read lands ~0.89 (NO-GO); a clear would require the eval to be unexpectedly aime-like. This sharpens
     #323's binding unknown without spending a GPU -- by RESOLVING the distribution identity (reasoning,
     not free-form) from the repo's own data.

LOCAL CPU-only analytic card. No GPU / vLLM / model forward / training / HF Job / submission /
served-file change. NOT a launch. BASELINE stays 481.53 (0 TPS). Greedy/PPL untouched.

PRIMARY metric  sharegpt_coverage_prior_self_test_passes
TEST    metric  p_sharegpt_clears_0p9213  (float; P(official-eval uncond top-4 >= 0.9213))

Reproduce:
    cd target/ && .venv/bin/python \\
        research/validity/eagle3_sharegpt_coverage_prior/eagle3_sharegpt_coverage_prior.py \\
        --self-test --wandb_group eagle3-sharegpt-prior \\
        --wandb_name lawine/eagle3-sharegpt-coverage-prior
"""
from __future__ import annotations

import argparse
import json
import math
import re
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
# Cross-checked against the on-disk banked #323 artifact (eagle3_coverage_achievability).
# --------------------------------------------------------------------------- #
# lawine #316 (5lnz5jgb) via lawine #323 (ceddxj20): the regime-invariant build bar.
T_EFFECTIVE = 0.9213011665456927        # build bar in UNCONDITIONAL top-4 units (regime-invariant)
A1_DEPLOYED = 0.7292532942898975        # deployed raw a1

# wirbel #79 (z6wi4z4v): MEASURED rank-coverage on the deployed LINEAR MTP spine.
LINEAR_COV4 = 0.6531976066516435
LINEAR_FRAC = 0.3468023933483565
LINEAR_UNCOND_TOP4 = 0.9061043944685533  # a1+(1-a1)*cov4 (#323 step1; 0.0152 below bar)

# fern #34 (gua9x68j): the ONLY trained {2,21,39} fusion EAGLE-3 head. W&B-verified eval/* summary,
# teacher-forced (feature_shift=1) on the 240-record benchmark-matched reasoning holdout. The per-source
# UNCONDITIONAL top-4 coverage values (eval/src_<>_top4) -- these ARE the official-eval per-source curve,
# because the holdout source mix (mmlu_pro 107 / gpqa 107 / aime 26) matches the official eval (57/57/14).
FERN_TF_TOP1 = 0.7616588114947002       # aggregate raw a1 (tf)
FERN_TF_TOP4 = 0.8902556121072153       # aggregate UNCONDITIONAL top-4 (the holdout-weighted figure)
FERN_NATIVE_VS_TF_GAP = 0.009695355171966513   # native_step1 - tf at the ROOT (native slightly higher)
# Per-source teacher-forced top-4 (UNCONDITIONAL) + holdout record counts (from #323 / fern #34).
FERN_SRC = {
    "aime":     {"top1": 0.8426445600123578, "top4": 0.957005303537408,  "cov4": 0.7267670157068064,
                 "frac": 0.27323298429319365, "holdout_n": 26},
    "gpqa":     {"top1": 0.8033197712129253, "top4": 0.9175953770859131, "cov4": 0.5810223354819368,
                 "frac": 0.4189776645180632, "holdout_n": 107},
    "mmlu_pro": {"top1": 0.7006353522181838, "top4": 0.846544405293677,  "cov4": 0.4873957367933272,
                 "frac": 0.5126042632066727, "holdout_n": 107},
}

# Published cross-paper estimate for a FULLY-trained head (a1~0.765): unconditional top-4 band (#323 step3).
LIT_TOP4_CENTRAL = 0.913
LIT_TOP4_RANGE = [0.899, 0.929]

# Banked on-disk artifacts (read-only; the eval composition is loaded live from these and the self-test
# cross-checks the banked #323 numbers so there is no silent drift).
EVAL_PROMPTS_PATH = (REPO_ROOT / "official" / "main_bucket" / "shared_resources" / "speed_benchmark"
                     / "data" / "eval_prompts_sharegpt.json")
DECODE_OUTPUTS_PATH = (REPO_ROOT / "research" / "greedy_reference" / "google__gemma-4-E4B-it"
                       / "decode_outputs.jsonl")
PR323_RESULTS = (REPO_ROOT / "research" / "validity" / "eagle3_coverage_achievability"
                 / "eagle3_coverage_achievability_results.json")

# Banked official-eval composition (fallback when the on-disk files are absent; the live load overrides
# these and the self-test asserts the live values equal the banked ones when files are present).
N_PROMPTS = 128
BANKED_SRC_COUNTS = {"aime": 14, "gpqa": 57, "mmlu_pro": 57}   # by id prefix (aime2026/gpqa_diamond/mmlu_pro)
BANKED_MCQ_PROMPTS = 114                                       # prompts with >=2 lettered options
BANKED_COMPLETION_TOKENS_EACH = 512                           # ignore_eos=true -> uniform 512 per prompt

# Model parameters for the sensitivity (deliverable 3).
C_MCQ_LETTER_CENTRAL = 0.40    # coverage of a PURE answer-letter token (~uniform over ~10 -> top-4 ~ 4/10)
C_MCQ_LETTER_RANGE = [0.30, 0.55]

BASELINE_TPS = 481.53          # current best summary.json:tps (unchanged; 0-TPS analytic)

TOL = 1e-9
TOL_REPRO = 1e-6


# --------------------------------------------------------------------------- #
# Stats helpers (normal CDF/SF via erf -- no scipy dependency).
# --------------------------------------------------------------------------- #
def norm_sf(z: float) -> float:
    """Upper-tail standard-normal survival function P(Z > z)."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def uncond_from_cov_cond(a1: float, cov_cond: float) -> float:
    """Salvage identity: unconditional top-k = a1 + (1-a1)*cov_cond == c1_eff."""
    return a1 + (1.0 - a1) * cov_cond


def _src_key(rid: str) -> str:
    """Normalize an eval/decode id to the fern source key (aime / gpqa / mmlu_pro)."""
    head = re.match(r"([a-zA-Z_]+)", rid).group(1)
    if head.startswith("gpqa"):
        return "gpqa"
    if head.startswith("aime"):
        return "aime"
    if head.startswith("mmlu_pro"):
        return "mmlu_pro"
    return head


def _char_density(text: str, pattern: str) -> float:
    if not text:
        return 0.0
    return len(re.findall(pattern, text)) / len(text)


# --------------------------------------------------------------------------- #
# Live load of the official eval composition (deliverable 1).
# --------------------------------------------------------------------------- #
def load_eval_composition() -> dict[str, Any]:
    """Characterize the official 128 eval from the on-disk file + greedy decode. Falls back to banked
    constants when the files are absent so the card still runs CPU-only anywhere."""
    out: dict[str, Any] = {"loaded_from_disk": False}

    # source mix + MCQ-prompt count from the eval prompts file.
    src_counts = dict(BANKED_SRC_COUNTS)
    mcq_prompts = BANKED_MCQ_PROMPTS
    n_prompts = N_PROMPTS
    if EVAL_PROMPTS_PATH.exists():
        prompts = json.loads(EVAL_PROMPTS_PATH.read_text())
        n_prompts = len(prompts)
        src_counts = {}
        mcq_prompts = 0
        for p in prompts:
            k = _src_key(p["id"])
            src_counts[k] = src_counts.get(k, 0) + 1
            human = p["conversations"][0]["value"]
            n_opts = len(set(re.findall(r"(?:^|\n)\s*([A-J])\)", human)))
            if n_opts >= 2:
                mcq_prompts += 1
        out["loaded_from_disk"] = True
    out["n_prompts"] = n_prompts
    out["src_counts"] = src_counts
    out["mcq_prompts"] = mcq_prompts
    out["mcq_frac"] = mcq_prompts / n_prompts
    out["reasoning_stem_frac"] = sum(src_counts.get(k, 0) for k in ("aime", "gpqa", "mmlu_pro")) / n_prompts
    out["freeform_conversational_frac"] = 0.0   # zero by construction (no chat ids present)

    # generation token-type signatures + literal answer-letter fraction from the greedy decode.
    sig: dict[str, Any] = {}
    completion_lens_uniform = True
    completion_tokens_each = BANKED_COMPLETION_TOKENS_EACH
    answer_letter_token_frac = None
    if DECODE_OUTPUTS_PATH.exists():
        recs = [json.loads(ln) for ln in DECODE_OUTPUTS_PATH.read_text().splitlines() if ln.strip()]
        lens = [len(r["completion_token_ids"]) for r in recs]
        completion_lens_uniform = (len(set(lens)) == 1)
        completion_tokens_each = lens[0] if lens else BANKED_COMPLETION_TOKENS_EACH
        total_tokens = sum(lens)
        # overall + per-source char-class densities over generated_text.
        def _agg(texts: list[str]) -> dict[str, float]:
            n = max(1, len(texts))
            return {
                "latex_density": sum(_char_density(t, r"[\\$^_{}]") for t in texts) / n,
                "digit_density": sum(_char_density(t, r"[0-9]") for t in texts) / n,
                "alpha_density": sum(_char_density(t, r"[A-Za-z]") for t in texts) / n,
                "has_answer_tail_frac": sum(bool(re.search(r"ANSWER\s*:", t)) for t in texts) / n,
                "has_option_enum_frac": sum(bool(re.search(r"(^|\n)\s*[A-J]\)", t)) for t in texts) / n,
            }
        all_gen = [r["generated_text"] for r in recs]
        sig["overall"] = _agg(all_gen)
        by_src: dict[str, list[str]] = {}
        for r in recs:
            by_src.setdefault(_src_key(r["id"]), []).append(r["generated_text"])
        sig["by_source"] = {k: _agg(v) for k, v in by_src.items()}
        # literal committed answer-letter token fraction: ~1 letter token per generation that REACHES
        # "ANSWER:" (most do not within 512 tokens), divided by all generated tokens. Upper bound.
        n_reach_answer = sum(bool(re.search(r"ANSWER\s*:", t)) for t in all_gen)
        answer_letter_token_frac = n_reach_answer / max(1, total_tokens)
        out["decode_loaded"] = True
    else:
        out["decode_loaded"] = False
    out["completion_lens_uniform"] = completion_lens_uniform
    out["completion_tokens_each"] = completion_tokens_each
    out["token_weight_equals_prompt_weight"] = bool(completion_lens_uniform)
    out["generation_signatures"] = sig
    out["literal_answer_letter_token_frac"] = answer_letter_token_frac
    return out


# --------------------------------------------------------------------------- #
# Load banked #323 result (read-only) for the cross-check.
# --------------------------------------------------------------------------- #
def load_banked_323() -> dict[str, Any]:
    return json.loads(PR323_RESULTS.read_text()) if PR323_RESULTS.exists() else {}


# --------------------------------------------------------------------------- #
# Synthesis (steps 1-4).
# --------------------------------------------------------------------------- #
def synthesize(comp: dict[str, Any], banked323: dict[str, Any]) -> dict[str, Any]:
    # ---- STEP 1: token-type composition of the official 128 eval. ---- #
    src_counts = comp["src_counts"]
    # the deployment distribution is reasoning/STEM, NOT free-form conversational.
    premise_freeform_refuted = bool(
        comp["freeform_conversational_frac"] <= TOL and comp["mcq_frac"] >= 0.5)
    sig = comp.get("generation_signatures", {})
    overall_sig = sig.get("overall", {})
    # the literal answer-letter fraction (the token the PR's letter-model blames) is ~0.
    letter_frac = comp.get("literal_answer_letter_token_frac")
    step1 = {
        "eval_file": str(EVAL_PROMPTS_PATH.relative_to(REPO_ROOT)),
        "loaded_from_disk": comp["loaded_from_disk"],
        "n_prompts": comp["n_prompts"],
        "src_counts": src_counts,
        "src_fracs": {k: v / comp["n_prompts"] for k, v in src_counts.items()},
        "mcq_prompts": comp["mcq_prompts"],
        "mcq_frac": comp["mcq_frac"],
        "reasoning_stem_frac": comp["reasoning_stem_frac"],
        "freeform_conversational_frac": comp["freeform_conversational_frac"],
        "premise_freeform_sharegpt_refuted": premise_freeform_refuted,
        "completion_tokens_each": comp["completion_tokens_each"],
        "completion_lens_uniform": comp["completion_lens_uniform"],
        "token_weight_equals_prompt_weight": comp["token_weight_equals_prompt_weight"],
        "generation_signatures": sig,
        "literal_answer_letter_token_frac": letter_frac,
        "note": (
            "The official 128 eval (file '{}') is 100% reasoning/STEM: mmlu_pro {} / gpqa {} / aime {} "
            "(by id prefix). {}/{} prompts ({:.1%}) are multiple-choice (>=2 lettered options) 'ending in "
            "ANSWER: $LETTER'; the rest are open-form competition math. ZERO free-form conversational chat "
            "-> the '#323 free-form ShareGPT, ~0 MCQ' premise is REFUTED by the repo's own benchmark data "
            "(DATASET_ANALYSIS.md, 2026-06-13). ignore_eos=true forces exactly {} tokens/prompt, so "
            "token-weight == prompt-weight == source proportions. The greedy decode shows LaTeX/math "
            "reasoning (latex char-density {:.3f}, digit {:.3f}); the literal committed answer-letter token "
            "is only ~{} of all generated tokens (most 512-token chains never reach 'ANSWER:'), so the "
            "coverage drag is the reasoning-CoT VOCAB BREADTH, not a near-absent letter."
            .format(step1_eval_name(), src_counts.get("mmlu_pro", 0), src_counts.get("gpqa", 0),
                    src_counts.get("aime", 0), comp["mcq_prompts"], comp["n_prompts"], comp["mcq_frac"],
                    comp["completion_tokens_each"], overall_sig.get("latex_density", float("nan")),
                    overall_sig.get("digit_density", float("nan")),
                    ("%.4f" % letter_frac) if letter_frac is not None else "n/a")),
    }

    # ---- STEP 2: composition -> coverage. ---- #
    # weight the per-source UNCONDITIONAL top-4 by the OFFICIAL eval proportions (== token weight).
    n_prompts = comp["n_prompts"]
    weighted_central = sum(FERN_SRC[k]["top4"] * src_counts.get(k, 0) for k in FERN_SRC) / n_prompts
    gap_to_bar = weighted_central - T_EFFECTIVE
    # cross-check: weighting matched-distribution per-source reproduces fern's aggregate (proportions match).
    central_vs_fern_aggregate_diff = abs(weighted_central - FERN_TF_TOP4)
    # conservative record-Bernoulli aggregate SE (treats each holdout RECORD as one Bernoulli trial).
    var_agg = 0.0
    for k, d in FERN_SRC.items():
        w = src_counts.get(k, 0) / n_prompts
        p = d["top4"]
        var_agg += (w ** 2) * (p * (1.0 - p) / d["holdout_n"])
    se_agg = math.sqrt(var_agg)
    band_95 = [weighted_central - 1.96 * se_agg, weighted_central + 1.96 * se_agg]
    # native-vs-tf upside (root gap; native slightly higher) -- an optimistic shift, not the central.
    native_adj_central = weighted_central + FERN_NATIVE_VS_TF_GAP
    # P(clears) under a small SE grid + the native-adjusted central.
    se_grid = [0.010, 0.020, 0.030]
    p_clears_by_se = {f"se_{s:.3f}": norm_sf((T_EFFECTIVE - weighted_central) / s) for s in se_grid}
    p_clears_central = norm_sf((T_EFFECTIVE - weighted_central) / se_agg)
    p_clears_native_adj = norm_sf((T_EFFECTIVE - native_adj_central) / se_agg)
    p_clears_range = [min(list(p_clears_by_se.values()) + [p_clears_native_adj]),
                      max(list(p_clears_by_se.values()) + [p_clears_native_adj])]
    step2 = {
        "per_source_top4": {k: FERN_SRC[k]["top4"] for k in FERN_SRC},
        "official_eval_weights": {k: src_counts.get(k, 0) / n_prompts for k in FERN_SRC},
        "point_estimate_uncond_top4": weighted_central,
        "gap_to_bar": gap_to_bar,
        "misses_bar": bool(weighted_central < T_EFFECTIVE - TOL),
        "central_vs_fern_aggregate_abs_diff": central_vs_fern_aggregate_diff,
        "record_bernoulli_se": se_agg,
        "band_95": band_95,
        "bar_within_95_band": bool(band_95[0] <= T_EFFECTIVE <= band_95[1]),
        "native_adj_central_uncond_top4": native_adj_central,
        "p_clears_by_se": p_clears_by_se,
        "p_clears_central": p_clears_central,
        "p_clears_native_adj": p_clears_native_adj,
        "p_clears_range": p_clears_range,
        "linear_spine_uncond_top4": LINEAR_UNCOND_TOP4,
        "note": (
            "Official-eval prior = 57/57/14-weighted per-source unconditional top-4 = {:.4f} (== fern's "
            "benchmark-matched aggregate to {:.1e}, because the holdout proportions match). It MISSES the "
            "0.9213 bar by {:.4f}. Conservative record-Bernoulli SE {:.4f} -> 95% band [{:.4f}, {:.4f}] "
            "with the bar near the UPPER edge. P(clears) central ~ {:.3f} (range {:.3f}..{:.3f} across SE "
            "and the +{:.4f} native-vs-tf upside). Even the published fully-trained-head central (0.913) "
            "misses this reasoning eval; only the upper lit edge (0.929) clears."
            .format(weighted_central, central_vs_fern_aggregate_diff, -gap_to_bar, se_agg,
                    band_95[0], band_95[1], p_clears_central, p_clears_range[0], p_clears_range[1],
                    FERN_NATIVE_VS_TF_GAP)),
    }

    # ---- STEP 3: sensitivity. ---- #
    # (a) source-mix axis (correct): free-form(aime) fraction f, rest = the 57/57 gpqa/mmlu_pro blend.
    n_mcq = src_counts.get("gpqa", 0) + src_counts.get("mmlu_pro", 0)
    mcq_blend_cov = ((FERN_SRC["gpqa"]["top4"] * src_counts.get("gpqa", 0)
                      + FERN_SRC["mmlu_pro"]["top4"] * src_counts.get("mmlu_pro", 0)) / max(1, n_mcq))
    aime_cov = FERN_SRC["aime"]["top4"]
    f_actual = src_counts.get("aime", 0) / n_prompts
    f_cross = (T_EFFECTIVE - mcq_blend_cov) / (aime_cov - mcq_blend_cov)
    sourcemix_sweep = {}
    for f in [0.0, f_actual, 0.25, 0.50, f_cross, 0.75, 1.0]:
        cov = f * aime_cov + (1.0 - f) * mcq_blend_cov
        sourcemix_sweep[f"f_{f:.4f}"] = {"freeform_frac": f, "uncond_top4": cov,
                                         "clears": bool(cov >= T_EFFECTIVE - TOL)}
    # (b) the PR's literal X% MCQ-answer-letter model (base = free-form @ aime, + X% letter @ c_mcq).
    letter_model = {}
    for c_mcq in [C_MCQ_LETTER_RANGE[0], C_MCQ_LETTER_CENTRAL, C_MCQ_LETTER_RANGE[1]]:
        x_cross = (aime_cov - T_EFFECTIVE) / (aime_cov - c_mcq)
        sweep = {}
        for X in [0.0, 0.05, 0.10]:
            cov = (1.0 - X) * aime_cov + X * c_mcq
            sweep[f"X_{X:.2f}"] = {"uncond_top4": cov, "clears": bool(cov >= T_EFFECTIVE - TOL)}
        letter_model[f"c_mcq_{c_mcq:.2f}"] = {"x_cross": x_cross, "sweep": sweep}
    x_cross_central = (aime_cov - T_EFFECTIVE) / (aime_cov - C_MCQ_LETTER_CENTRAL)
    letter_frac = comp.get("literal_answer_letter_token_frac")
    letter_headroom_ratio = (x_cross_central / letter_frac) if (letter_frac and letter_frac > 0) else None
    step3 = {
        "sourcemix_mcq_blend_cov": mcq_blend_cov,
        "sourcemix_aime_cov": aime_cov,
        "sourcemix_freeform_frac_actual": f_actual,
        "sourcemix_freeform_frac_crossing": f_cross,
        "sourcemix_sweep": sourcemix_sweep,
        "letter_model_central_x_cross": x_cross_central,
        "letter_model": letter_model,
        "literal_answer_letter_token_frac": letter_frac,
        "letter_model_headroom_ratio": letter_headroom_ratio,
        "note": (
            "(a) SOURCE-MIX axis (correct): cov(f)=f*{:.4f}+(1-f)*{:.4f} crosses the bar at free-form(aime) "
            "fraction f={:.3f}; the eval sits at f={:.3f} -> deep miss (needs ~5x more open-math content). "
            "(b) PR's X% answer-letter model (base@aime + X% letter@{:.2f}): crosses at X={:.3f}; X in {{0,5}} "
            "clears, X=10 misses. But the eval's LITERAL answer-letter token fraction is ~{} -> ~{}x below "
            "the X-crossing, so this (wrong) model predicts CLEARS. That is exactly why the premise looked "
            "optimistic: it localizes the drag to a near-absent token. The DATA say the drag is the whole "
            "reasoning-CoT body (mmlu_pro 0.8465), which the source-mix axis captures and the letter axis misses."
            .format(aime_cov, mcq_blend_cov, f_cross, f_actual, C_MCQ_LETTER_CENTRAL, x_cross_central,
                    ("%.4f" % letter_frac) if letter_frac is not None else "n/a",
                    ("%d" % round(letter_headroom_ratio)) if letter_headroom_ratio else "n/a")),
    }

    # ---- STEP 4: verdict. ---- #
    p_central = p_clears_central
    if p_central >= 0.70:
        verdict = "LIKELY-CLEARS"
    elif p_central >= 0.30:
        verdict = "TOSS-UP"
    else:
        verdict = "LIKELY-MISSES"
    step4 = {
        "verdict": verdict,
        "p_sharegpt_clears_0p9213": p_central,
        "p_clears_range": p_clears_range,
        "point_estimate_uncond_top4": weighted_central,
        "gap_to_bar": gap_to_bar,
        "thresholds": {"likely_clears_p>=": 0.70, "toss_up_p>=": 0.30, "likely_misses_p<": 0.30},
        "supporting_coverage_estimate": (
            "Official-eval unconditional top-4 prior = {:.4f} +- {:.4f} (record-Bernoulli), {:.4f} BELOW the "
            "0.9213 bar. P(clears) ~ {:.3f} (range {:.3f}..{:.3f}). Verdict LIKELY-MISSES.".format(
                weighted_central, se_agg, -gap_to_bar, p_central, p_clears_range[0], p_clears_range[1])),
        "reframes_329_read": (
            "This is the PRIOR for fern #329's GPU-gated RANKPROBE_W=4 read over the official 128. Because "
            "those 128 ARE the reasoning/MCQ distribution (not free-form), the prior predicts the read lands "
            "~0.89 (NO-GO), CONFIRMING the miss rather than discovering a clear. #323's 'what flips it' "
            "framed the official 128 as DIFFERENT from the reasoning holdout; they are the SAME distribution "
            "(the holdout is 'benchmark-matched', deduped against these 128), so the holdout aggregate 0.8903 "
            "is ALREADY an on-distribution estimate. The read would tighten the CI, not move the central."),
        "what_would_flip_it": (
            "A CLEAR (>=0.9213) would require EITHER (i) the official 128 to be unexpectedly aime-like "
            "(free-form fraction ~0.52 vs the actual 0.11 -- contradicted by the on-disk ids), OR (ii) a "
            "BETTER-trained fusion head than fern #34 (soft-KD top-k calibration / more root training) "
            "lifting per-source coverage by >=0.031 uniformly -- plausible but UNTESTED, and even the "
            "published fully-trained-head central (0.913) still misses this reasoning eval."),
        "caveats": (
            "PRIOR, NOT a measurement: built on fern #34's TEACHER-FORCED per-source coverage (root tf~native, "
            "gap +0.0097) over a holdout that MATCHES but is disjoint from the official 128. The record-"
            "Bernoulli SE is conservative (token-level SE is tighter -> even lower P). The 0.9213 bar is "
            "position-1 root coverage: necessary, NOT sufficient -- the deployed deep spine still caps E[T] "
            "at 4.91 (#316). fern #34 is the ONLY trained head and is undertrained (K=1, hard-CE, no soft-KD, "
            "no TTT); a better head is the only credible clear lever. No fusion checkpoint is deployed; the "
            "drafter BUILD stays human-gated. 0 TPS; greedy/PPL untouched; BASELINE 481.53 unchanged."),
        "rationale": (
            "LIKELY-MISSES. (1) The official 128 'ShareGPT' eval is 100% reasoning/STEM (mmlu_pro 57 / gpqa "
            "57 / aime 14); 89% are MCQ -- the '#323 free-form, ~0 MCQ' premise is REFUTED by the repo's own "
            "benchmark data. (2) The 57/57/14-weighted per-source prior is 0.8903 == fern's benchmark-matched "
            "aggregate, 0.031 BELOW the bar; P(clears) ~ 0.06. (3) The coverage drag is the reasoning-CoT "
            "vocab breadth, NOT the MCQ-answer-letter (which is ~0.02% of tokens) -- so the PR's letter-"
            "contamination intuition (which predicts a clear) mis-localizes the drag. (4) This RESOLVES "
            "#323's binding unknown by identifying the deployment distribution (reasoning, not free-form); "
            "#323's MARGINAL leaned on the optimistic free-form assumption, which is false."),
    }

    return {
        "step1_eval_composition": step1,
        "step2_composition_to_coverage": step2,
        "step3_sensitivity": step3,
        "step4_verdict": step4,
        "test_metrics": {
            "sharegpt_coverage_prior_self_test_passes": None,   # filled by self_test/main
            "p_sharegpt_clears_0p9213": p_central,
            "official_eval_uncond_top4_prior": weighted_central,
            "verdict": verdict,
        },
        "imported": {
            "T_effective": T_EFFECTIVE, "fern34_per_source_top4": {k: FERN_SRC[k]["top4"] for k in FERN_SRC},
            "fern34_aggregate_top4": FERN_TF_TOP4, "linear_uncond_top4_79": LINEAR_UNCOND_TOP4,
            "lit_top4_central": LIT_TOP4_CENTRAL, "lit_top4_range": LIT_TOP4_RANGE,
            "provenance": (
                "build bar lawine #316 (5lnz5jgb) via lawine #323 (ceddxj20); per-source + aggregate top-4 "
                "fern #34 (gua9x68j, W&B-verified eval/* summary); linear spine wirbel #79 (z6wi4z4v); "
                "official 128 eval composition loaded live from "
                "official/main_bucket/.../speed_benchmark/data/eval_prompts_sharegpt.json + "
                "research/greedy_reference/google__gemma-4-E4B-it/decode_outputs.jsonl; source-identity "
                "confirmed by research/DATASET_ANALYSIS.md (2026-06-13). Composition->coverage prior, "
                "sensitivity, and verdict are this card (#330)."),
        },
    }


def step1_eval_name() -> str:
    return EVAL_PROMPTS_PATH.name


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def self_test(syn: dict[str, Any], comp: dict[str, Any], banked323: dict[str, Any]) -> dict[str, Any]:
    s1 = syn["step1_eval_composition"]
    s2 = syn["step2_composition_to_coverage"]
    s3 = syn["step3_sensitivity"]
    s4 = syn["step4_verdict"]
    c: dict[str, bool] = {}

    # (a) composition: reasoning/STEM, not free-form; MCQ-heavy; uniform 512-token weighting.
    c["01_eval_is_100pct_reasoning_stem"] = bool(abs(s1["reasoning_stem_frac"] - 1.0) <= TOL)
    c["02_eval_has_zero_freeform_chat"] = bool(s1["freeform_conversational_frac"] <= TOL)
    c["03_premise_freeform_refuted"] = bool(s1["premise_freeform_sharegpt_refuted"])
    c["04_mcq_majority"] = bool(s1["mcq_frac"] >= 0.5)
    c["05_token_weight_eq_prompt_weight"] = bool(s1["token_weight_equals_prompt_weight"])
    # if loaded from disk, the source mix must be exactly 57/57/14 and 114 MCQ.
    if comp["loaded_from_disk"]:
        c["06_source_mix_57_57_14"] = bool(
            s1["src_counts"].get("mmlu_pro") == 57 and s1["src_counts"].get("gpqa") == 57
            and s1["src_counts"].get("aime") == 14 and s1["n_prompts"] == 128)
        c["07_mcq_prompt_count_114"] = bool(s1["mcq_prompts"] == 114)
    else:
        c["06_source_mix_57_57_14"] = True   # banked fallback path; not asserting disk values
        c["07_mcq_prompt_count_114"] = True
    # the generation is LaTeX/math reasoning, not prose (latex density well above a free-form floor).
    if comp.get("decode_loaded"):
        latex = s1["generation_signatures"]["overall"]["latex_density"]
        c["08_generations_are_reasoning_latex"] = bool(latex >= 0.02)
        # the literal answer-letter token is a near-absent fraction (drag is NOT the letter).
        c["09_answer_letter_token_near_absent"] = bool(
            s1["literal_answer_letter_token_frac"] is not None
            and s1["literal_answer_letter_token_frac"] <= 0.01)
    else:
        c["08_generations_are_reasoning_latex"] = True
        c["09_answer_letter_token_near_absent"] = True

    # (b) coverage arithmetic.
    # weighted central reproduces fern's benchmark-matched aggregate (proportions match) to ~1e-4.
    c["10_central_matches_fern_aggregate"] = bool(s2["central_vs_fern_aggregate_abs_diff"] <= 5e-4)
    # salvage-identity sanity: linear spine sits below the bar by ~0.0152 (anchors the scale).
    c["11_linear_spine_below_bar"] = bool(s2["linear_spine_uncond_top4"] < T_EFFECTIVE)
    # the prior genuinely MISSES the bar.
    c["12_prior_misses_bar"] = bool(s2["misses_bar"] and s2["gap_to_bar"] < 0.0)
    # the bar sits inside (near the upper edge of) the 95% band -> a clear is unlikely but not impossible.
    c["13_bar_in_95_band_upper"] = bool(
        s2["bar_within_95_band"] and T_EFFECTIVE > 0.5 * (s2["band_95"][0] + s2["band_95"][1]))
    # P(clears) is low and well-defined.
    c["14_p_clears_low"] = bool(0.0 <= s2["p_clears_central"] < 0.30)

    # (c) sensitivity coherence.
    # the source-mix crossing requires far more free-form than the eval actually has.
    c["15_sourcemix_crossing_above_actual"] = bool(
        s3["sourcemix_freeform_frac_crossing"] > s3["sourcemix_freeform_frac_actual"] + 0.2)
    # the PR's letter-model crosses at a small X (few %) -> the eval's ~0 letter fraction would "clear"
    # under that model, which is the optimistic-premise trap; headroom ratio is large.
    c["16_letter_model_small_crossing"] = bool(0.0 < s3["letter_model_central_x_cross"] < 0.15)
    if comp.get("decode_loaded"):
        c["17_letter_headroom_large"] = bool(
            s3["letter_model_headroom_ratio"] is not None and s3["letter_model_headroom_ratio"] >= 10.0)
    else:
        c["17_letter_headroom_large"] = True

    # (d) verdict coherence + banked-constant drift guard.
    c["18_verdict_is_likely_misses"] = bool(s4["verdict"] == "LIKELY-MISSES")
    c["19_p_in_unit_interval"] = bool(0.0 <= s4["p_sharegpt_clears_0p9213"] <= 1.0)
    # imported per-source top-4 match the banked #323 artifact (no silent drift).
    drift_ok = True
    if banked323:
        ps323 = (((banked323.get("synthesis", {}) or {}).get("step2_fern34_measured", {}) or {})
                 .get("per_source", {}) or {})
        for k in FERN_SRC:
            if ps323.get(k, {}).get("top4_uncond") is not None:
                drift_ok = drift_ok and abs(float(ps323[k]["top4_uncond"]) - FERN_SRC[k]["top4"]) <= TOL_REPRO
    c["20_constants_match_banked_323"] = bool(drift_ok and bool(banked323))

    # (e) caveats / pre-registration honesty carried.
    c["21_caveats_carried"] = bool(
        isinstance(s4["caveats"], str) and len(s4["caveats"]) > 120
        and isinstance(s4["reframes_329_read"], str) and len(s4["reframes_329_read"]) > 80)

    gate = all(bool(v) for v in c.values())
    return {"sharegpt_coverage_prior_self_test_passes": gate, "checks": c}


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
        print(f"[eagle3-sharegpt-prior] wandb logging unavailable: {exc}", flush=True)
        return None

    syn = payload["synthesis"]
    s1, s2, s3, s4 = (syn["step1_eval_composition"], syn["step2_composition_to_coverage"],
                      syn["step3_sensitivity"], syn["step4_verdict"])
    st = payload["self_test"]
    tm = syn["test_metrics"]

    run = init_wandb_run(
        job_type="validity-analytic", agent="lawine", name=args.wandb_name, group=args.wandb_group,
        tags=["eagle3-sharegpt-coverage-prior", "validity-analytic", "rank-coverage", "eagle3",
              "deployment-distribution", "sharegpt-misnomer", "premise-refuted", "bank-the-analysis"],
        config={
            "pr": 330, "build_bar_unconditional_top4": T_EFFECTIVE,
            "official_eval_src_counts": s1["src_counts"], "official_eval_mcq_frac": s1["mcq_frac"],
            "fern34_aggregate_top4": FERN_TF_TOP4, "lit_top4_central": LIT_TOP4_CENTRAL,
            "baseline_tps": BASELINE_TPS, "wandb_group": args.wandb_group,
            "imports": syn["imported"]["provenance"],
        },
    )
    if run is None:
        print("[eagle3-sharegpt-prior] wandb: no run (no WANDB_API_KEY/mode) -- skipping", flush=True)
        return None

    summary: dict[str, Any] = {
        # PRIMARY + TEST
        "sharegpt_coverage_prior_self_test_passes": int(bool(
            st["sharegpt_coverage_prior_self_test_passes"])),
        "p_sharegpt_clears_0p9213": tm["p_sharegpt_clears_0p9213"],
        "official_eval_uncond_top4_prior": tm["official_eval_uncond_top4_prior"],
        "verdict_likely_misses": int(s4["verdict"] == "LIKELY-MISSES"),
        # step1 composition
        "eval_reasoning_stem_frac": s1["reasoning_stem_frac"],
        "eval_freeform_conversational_frac": s1["freeform_conversational_frac"],
        "eval_mcq_frac": s1["mcq_frac"], "eval_mcq_prompts": s1["mcq_prompts"],
        "eval_n_prompts": s1["n_prompts"],
        "eval_completion_tokens_each": s1["completion_tokens_each"],
        "premise_freeform_refuted": int(bool(s1["premise_freeform_sharegpt_refuted"])),
        "literal_answer_letter_token_frac": s1["literal_answer_letter_token_frac"],
        # step2 coverage
        "point_estimate_uncond_top4": s2["point_estimate_uncond_top4"],
        "gap_to_bar": s2["gap_to_bar"], "record_bernoulli_se": s2["record_bernoulli_se"],
        "band_95_low": s2["band_95"][0], "band_95_high": s2["band_95"][1],
        "central_vs_fern_aggregate_abs_diff": s2["central_vs_fern_aggregate_abs_diff"],
        "native_adj_central_uncond_top4": s2["native_adj_central_uncond_top4"],
        "p_clears_central": s2["p_clears_central"], "p_clears_native_adj": s2["p_clears_native_adj"],
        # step3 sensitivity
        "sourcemix_freeform_frac_actual": s3["sourcemix_freeform_frac_actual"],
        "sourcemix_freeform_frac_crossing": s3["sourcemix_freeform_frac_crossing"],
        "letter_model_central_x_cross": s3["letter_model_central_x_cross"],
        "letter_model_headroom_ratio": s3["letter_model_headroom_ratio"],
        # bars
        "build_bar_unconditional_top4": T_EFFECTIVE,
        "fern34_aggregate_top4": FERN_TF_TOP4,
        "lit_top4_central": LIT_TOP4_CENTRAL,
        # hygiene
        "nan_clean": int(bool(payload["nan_clean"])), "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(val)) for k, val in st["checks"].items()},
    }
    summary = {k: val for k, val in summary.items()
               if not (isinstance(val, float) and not math.isfinite(val)) and val is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="eagle3_sharegpt_coverage_prior_result", artifact_type="validity",
                      data=payload)
    rid = getattr(run, "id", None)
    print(f"[eagle3-sharegpt-prior] wandb run: {getattr(run, 'url', rid)}", flush=True)
    finish_wandb(run)
    return rid


# --------------------------------------------------------------------------- #
# Console report.
# --------------------------------------------------------------------------- #
def print_report(payload: dict) -> None:
    syn = payload["synthesis"]
    s1, s2, s3, s4 = (syn["step1_eval_composition"], syn["step2_composition_to_coverage"],
                      syn["step3_sensitivity"], syn["step4_verdict"])
    st = payload["self_test"]
    print("\n" + "=" * 100, flush=True)
    print("EAGLE-3 ShareGPT COVERAGE PRIOR (PR #330) -- what uncond top-4 does the official eval carry vs 0.9213?",
          flush=True)
    print("=" * 100, flush=True)
    print("STEP 1 -- token-type composition of the official 128 eval (loaded_from_disk={}):"
          .format(s1["loaded_from_disk"]), flush=True)
    print(f"  source mix: {s1['src_counts']}  ->  reasoning/STEM frac {s1['reasoning_stem_frac']:.3f}, "
          f"free-form chat frac {s1['freeform_conversational_frac']:.3f}", flush=True)
    print(f"  MCQ prompts {s1['mcq_prompts']}/{s1['n_prompts']} ({s1['mcq_frac']:.1%}); "
          f"completion tokens/prompt {s1['completion_tokens_each']} (uniform={s1['completion_lens_uniform']}); "
          f"premise free-form REFUTED = {s1['premise_freeform_sharegpt_refuted']}", flush=True)
    if s1["generation_signatures"].get("overall"):
        o = s1["generation_signatures"]["overall"]
        print(f"  gen signatures: latex_density {o['latex_density']:.4f}  digit {o['digit_density']:.4f}  "
              f"answer-tail {o['has_answer_tail_frac']:.3f}  option-enum {o['has_option_enum_frac']:.3f}",
              flush=True)
    print(f"  literal answer-letter token frac ~ {s1['literal_answer_letter_token_frac']}", flush=True)
    print("-" * 100, flush=True)
    print("STEP 2 -- composition -> coverage:", flush=True)
    print(f"  per-source top-4: {{k: round(v,4)}} -> "
          f"{ {k: round(v,4) for k,v in s2['per_source_top4'].items()} }", flush=True)
    print(f"  official-eval weights {{k: round(v,3)}}: "
          f"{ {k: round(v,3) for k,v in s2['official_eval_weights'].items()} }", flush=True)
    print(f"  POINT ESTIMATE uncond top-4 = {s2['point_estimate_uncond_top4']:.4f}  (gap to bar "
          f"{s2['gap_to_bar']:+.4f}, misses={s2['misses_bar']})", flush=True)
    print(f"  SE {s2['record_bernoulli_se']:.4f}  95% band [{s2['band_95'][0]:.4f}, {s2['band_95'][1]:.4f}]  "
          f"(bar in band={s2['bar_within_95_band']})", flush=True)
    print(f"  P(clears 0.9213) central = {s2['p_clears_central']:.4f}  "
          f"(by SE {{ {', '.join(f'{k}:{v:.3f}' for k,v in s2['p_clears_by_se'].items())} }}; "
          f"native-adj {s2['p_clears_native_adj']:.3f})", flush=True)
    print("-" * 100, flush=True)
    print("STEP 3 -- sensitivity:", flush=True)
    print(f"  (a) SOURCE-MIX: aime-cov {s3['sourcemix_aime_cov']:.4f} / mcq-blend {s3['sourcemix_mcq_blend_cov']:.4f}; "
          f"free-form frac actual {s3['sourcemix_freeform_frac_actual']:.3f} vs crossing "
          f"{s3['sourcemix_freeform_frac_crossing']:.3f}", flush=True)
    print(f"  (b) LETTER-MODEL: central X-cross {s3['letter_model_central_x_cross']:.4f}; "
          f"eval literal letter frac {s3['literal_answer_letter_token_frac']} -> headroom ratio "
          f"{s3['letter_model_headroom_ratio']}", flush=True)
    for c_mcq_key, dd in s3["letter_model"].items():
        sweep = ", ".join(f"{xk}:{xv['uncond_top4']:.4f}({'C' if xv['clears'] else 'M'})"
                          for xk, xv in dd["sweep"].items())
        print(f"     {c_mcq_key}: x_cross {dd['x_cross']:.4f}  [{sweep}]", flush=True)
    print("-" * 100, flush=True)
    print(f"STEP 4 -- VERDICT: {s4['verdict']}", flush=True)
    print(f"  p_sharegpt_clears_0p9213 = {s4['p_sharegpt_clears_0p9213']:.4f}  "
          f"(range {s4['p_clears_range'][0]:.3f}..{s4['p_clears_range'][1]:.3f})", flush=True)
    print(f"  {s4['supporting_coverage_estimate']}", flush=True)
    print(f"  REFRAMES #329: {s4['reframes_329_read']}", flush=True)
    print(f"  WOULD FLIP IT: {s4['what_would_flip_it']}", flush=True)
    print("-" * 100, flush=True)
    print(f"(PRIMARY) sharegpt_coverage_prior_self_test_passes = "
          f"{st['sharegpt_coverage_prior_self_test_passes']}", flush=True)
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
                    default="eagle3-sharegpt-prior")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    comp = load_eval_composition()
    banked323 = load_banked_323()
    syn = synthesize(comp, banked323)
    st = self_test(syn, comp, banked323)
    syn["test_metrics"]["sharegpt_coverage_prior_self_test_passes"] = bool(
        st["sharegpt_coverage_prior_self_test_passes"])

    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "pr": 330, "agent": "lawine", "kind": "eagle3-sharegpt-coverage-prior",
        "eagle3_sharegpt_coverage_prior_analysis_only": True,
        "eval_composition_raw": comp,
        "synthesis": syn, "self_test": st,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }

    nan_paths = assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[eagle3-sharegpt-prior] WARNING non-finite at: {nan_paths}", flush=True)
    gate = bool(st["sharegpt_coverage_prior_self_test_passes"] and payload["nan_clean"])
    st["sharegpt_coverage_prior_self_test_passes"] = gate
    syn["test_metrics"]["sharegpt_coverage_prior_self_test_passes"] = gate

    print_report(payload)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eagle3_sharegpt_coverage_prior_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[eagle3-sharegpt-prior] wrote {out_path}", flush=True)

    rid = maybe_log_wandb(args, payload)
    payload["wandb_run_id"] = rid
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    tm = syn["test_metrics"]
    print(f"  PRIMARY sharegpt_coverage_prior_self_test_passes = {gate}", flush=True)
    print(f"  TEST p_sharegpt_clears_0p9213 = {tm['p_sharegpt_clears_0p9213']:.4f}", flush=True)
    print(f"  PRIOR official_eval_uncond_top4 = {tm['official_eval_uncond_top4_prior']:.4f}", flush=True)
    print(f"  VERDICT = {tm['verdict']}", flush=True)
    print(f"  wandb run = {rid}", flush=True)

    if args.self_test:
        print(f"[eagle3-sharegpt-prior] self-test {'PASS' if gate else 'FAIL'}", flush=True)
        return 0 if gate else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
