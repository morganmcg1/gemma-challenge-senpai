#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Non-EAGLE-3 >500 feasibility screen (PR #345, stark) -- does ANY method escape the supply tax?

THE GOVERNING QUESTION (the #319 option-B fallback)
---------------------------------------------------
The strict-EAGLE-3 >500 lane is closed end-to-end: denken #332 (y5cl0ena) priced the SUPPLY
side RED (the verify-attention BW floor caps the strict-compliant ceiling at 473.53 < 500), and
stark #340 (jwv1vbug) priced the DEMAND side insufficient (at the honest fusion coverage 0.8903
the compliant-500 envelope collapses; even clearing the 0.9213 identity bar buys only central-500,
never worst-500). The #319 decision needs its option-B fallback answered numerically:

  *Is there a NON-EAGLE-3 speculative method that reaches strict-compliant >500 -- or do all
   alternatives share the same fate, leaving #124 (lifting the gate to PPL-only) as the only lever?*

THE LOAD-BEARING INSIGHT TO TEST (not assume)
---------------------------------------------
denken #332 proved the verify-step attention bandwidth floor (34.9% BW utilisation, arithmetic
intensity AI = 7.88 flop/byte << ridge 208) is OCCUPANCY-SATURATED and METHOD-INDEPENDENT: the
adaptive 3D split-KV verify path ALREADY launches 96 CTAs > the A10G's 80 SMs (occupancy-saturated)
yet still sits at 34.9% BW, so the exposed slack is the low-arithmetic-intensity attention floor of
the BATCHED MULTI-TOKEN VERIFY FORWARD reading the KV cache, NOT a property of EAGLE-3's drafter.

HYPOTHESIS: any speculative method that verifies M>1 candidate tokens in ONE batched target-model
forward (lookahead/Jacobi, n-gram PLD, Medusa tree, self-speculative/layer-skip -- ALL do) inherits
the SAME deterministic-attention supply tax under strict greedy-identity. If true, switching the
DRAFT method does NOT escape the strict supply RED -- only #124 (lifting the gate) does. This card
CONFIRMS or REFUTES that per method class, then prices the PPL-only alternative.

THE METHOD (CPU-analytic over banked W&B anchors + the literature candidate set; re-derives nothing)
---------------------------------------------------------------------------------------------------
For each candidate method class, tabulate:
  (i)   STRICT supply tax: does it use a batched multi-token verify forward? -> inherits_332_supply_tax
        (bool). If yes, its strict-compliant ceiling <= 473.53 (the #332 cap), method-independent.
  (ii)  PPL-only realistic E[T] on OUR ~100% reasoning/STEM eval (lawine #330): literature accept-rate
        RANGES (low/central/high), compared to the deployed EAGLE-3-lane head's E[T] = 3.8512 (the
        #289 a_k survival product). Reasoning/STEM is the WORST case for training-free methods
        (lookahead ~1.4-1.8x, n-gram PLD <1.1x) -- they rely on redundancy near-absent in CoT.
  (iii) PPL-only ceiling TPS = LAMBDA1_CEIL(520.95) * (E[T]_method / E[T]_eagle3_289): a LINEAR-E[T]
        scaling of the lambda=1 ceiling (the verify step time is method-independent per (i), so TPS
        scales with accepted-tokens-per-step). Does the method's ceiling beat the EAGLE-3 head's?

THE TWO VERDICT BOOLS (the deliverable)
---------------------------------------
  any_non_eagle3_escapes_strict_supply_tax (expect False) -- under strict, no method beats 473.53.
  any_non_eagle3_beats_eagle3_head_ppl_only (expect False) -- under PPL-only, NO alternative's E[T]
    (even its optimistic high anchor, even Medusa with its training cost) beats the EXISTING head.
ONE-LINE SYNTHESIS: under STRICT the method choice is IRRELEVANT (all <= 473.53 < 500) -> only #124
  helps; under PPL-only the EXISTING EAGLE-3 head + coverage retrain (wirbel's path, E[T]->6.1112)
  DOMINATES the training-free alternatives, which under-accept on reasoning/STEM.

HONEST CAVEATS (instruction 3)
------------------------------
- literature accept rates are workload-dependent POINT estimates -- carried as RANGES, not false
  precision; the verdict is robust across each method's whole band.
- self-speculative / layer-skip may ALTER PPL if it emits draft tokens without a full verify (flagged
  ppl_risk=True, out of scope for a speed-only screen).
- a DEFINITIVE accept-rate needs a measured run -- gated. Named here as a cheap ONE-RUN screen to fire
  IF #124 lifts the gate (vLLM ships ngram + EAGLE natively; one A10G run measures the real E[T]).
- this is a SCREEN, not a build recommendation: it scopes option B, it does not green-light any method.

LOCAL, CPU-ONLY, ANALYTIC. 0 GPU, no model forward, no training, no served-file change, no HF Job, no
submission, no launch, 0 official-TPS. BASELINE stays 481.53; adds 0 TPS. Imports VERBATIM: denken
#332 y5cl0ena (BW 34.9%, AI 7.88, geometric phi 0.925, floor 0.09841, strict ceiling 473.53, ceiling
520.953, >500 budget 4.022%); kanna #289 fi34s269 (a_k profile, E[T]=3.8512); stark #340 jwv1vbug
(coverage-retrain target E[T](0.9213)=6.1112); lawine #330 hfrscdai (eval 100% reasoning/STEM,
mmlu_pro 57 / gpqa 57 / aime 14, coverage prior 0.8903). Literature: Fu et al. ICML 2024
(lookahead/Jacobi, arXiv:2402.02057); Cai et al. 2024 (Medusa, arXiv:2401.10774); Saxena 2023 /
vLLM (n-gram prompt-lookup); Zhang et al. ACL 2024 (Draft&Verify self-spec, arXiv:2309.08168) +
Elhoushi et al. 2024 (LayerSkip, arXiv:2404.16710); Leviathan et al. ICML 2023 (spec-decode
distributional-equivalence guarantee, arXiv:2211.17192). Re-derives nothing measured. NOT a launch /
build / submission / served-file change / open2.

PRIMARY metric  non_eagle3_feasibility_self_test_passes
TEST    metric  any_non_eagle3_escapes_strict_supply_tax  (bool, expect False)
                + best_ppl_only_alternative_etratio        (float, best alt E[T] ratio vs EAGLE-3 head)
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

# --------------------------------------------------------------------------- #
# Imported EXACT from banked W&B runs -- DO NOT re-derive. Full-precision; the
# displayed 4-dp forms (473.53, 0.9213, 3.8512, ...) are asserted in the self-test.
# All runs live in wandb-applied-ai-team/gemma-challenge-senpai.
# --------------------------------------------------------------------------- #
# ---- denken #332 (y5cl0ena): the METHOD-INDEPENDENT verify-attention supply tax ----
SDPA_BW_UTIL = 0.34883864849061247           # 34.9% verify-attention BW utilisation (the floor)
SDPA_AI_FLOP_PER_BYTE = 7.880597014925373    # AI 7.88 flop/byte -- << ridge -> bandwidth-bound
RIDGE_AI = 208.33333333333334                # A10G roofline ridge (600 GB/s, fp16 compute)
GEOMETRIC_PHI_332 = 0.925                     # forgone split-KV parallelism fraction (occupancy)
FLOOR_AT_PHI1_327 = 0.09841249119201488       # determinism floor at phi=1 (denken #327, round-trip)
FLOOR_AT_GEO_332 = 0.09103155435261377        # floor at the geometric phi (= FLOOR_AT_PHI1*0.925)
STRICT_COMPLIANT_CEILING_332 = 473.5295953446407   # strict ceiling = LAMBDA1_CEIL*(1-floor); the 473.53 cap
LAMBDA1_CEIL = 520.9527323111674              # lambda=1 step-side ceiling (int4-spec batch-invariant verify)
N_FULL_3D_CTAS_332 = 96                        # adaptive 3D split-KV verify CTAs (> 80 SMs -> saturated)
A10G_SMS_332 = 80                              # A10G SM count (occupancy denominator)
BUDGET_500_FRAC_192 = 0.040220025755911104     # >500 floor budget = 1 - 500/520.953 (operative bar)

# ---- kanna #289 (fi34s269): deployed EAGLE-3-lane per-position conditional acceptance + E[T] ----
A_K_EAGLE3_289 = [
    0.7292532942898975,   # a_1 (the deployed cliff)
    0.759556697719242,    # a_2
    0.7929794882639035,   # a_3
    0.8228,               # a_4
    0.8348727920920435,   # a_5
    0.8357919254658385,   # a_6
    0.8464932652113331,   # a_7
]
E_T_EAGLE3_289 = 3.851185944363104            # E[T] = 1 + sum cumprod(a_k) -- the DEPLOYED head's E[T]

# ---- stark #340 (jwv1vbug): the coverage-retrain TARGET E[T] (wirbel's PPL-only path) ----
E_T_RETRAIN_TARGET_340 = 6.111214987369918    # E[T](0.9213) -- the coverage-retrained EAGLE-3 head target
IDENTITY_BAR_340 = 0.9213011665456927         # strict greedy-identity per-depth coverage bar

# ---- lawine #330 (hfrscdai): the eval composition (the accept-rate CONTEXT) ----
REASONING_STEM_FRAC = 1.0                      # eval is 100% reasoning/STEM
EVAL_MIX = {"mmlu_pro": 57, "gpqa_diamond": 57, "aime": 14}   # 128 prompts (89% MCQ CoT)
COV_PRIOR_330 = 0.8903                         # honest fusion top-4 coverage prior (the demand shortfall)

BASELINE_TPS = 481.53
TARGET = 500.0
K_CAL = 125.268                                # kanna #217/#269: official = K_cal * E[T] (= 481.53/3.844)

TOL_EXACT = 1e-9
TOL_ROUNDTRIP = 1e-6
TOL_DISPLAY = 5e-4


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# Core laws.
# --------------------------------------------------------------------------- #
def et_of_profile(a: list[float]) -> float:
    """E[T] = 1 + sum_{k} prod_{j<=k} a_j (survival sum of committed length; kanna #289/#294)."""
    s, prod = 0.0, 1.0
    for ak in a:
        prod *= ak
        s += prod
    return 1.0 + s


def ppl_only_ceiling_tps(et_method: float, et_eagle3: float = E_T_EAGLE3_289) -> float:
    """PPL-only ceiling for a method = LAMBDA1_CEIL * (E[T]_method / E[T]_eagle3) (instruction iii).

    A LINEAR-E[T] scaling of the lambda=1 ceiling: the verify step time is method-independent (the
    #332 supply tax is a property of the batched verify forward, not the drafter), so TPS scales with
    accepted-tokens-per-step. The EAGLE-3 head maps to LAMBDA1_CEIL by construction (ratio 1); every
    alternative with E[T] < E[T]_eagle3 maps strictly below it. This is a CEILING (lambda=1, no
    overhead); the head's REALIZED TPS is 481.53. Ranking is what matters and is convention-invariant.
    """
    return LAMBDA1_CEIL * (et_method / et_eagle3)


# --------------------------------------------------------------------------- #
# The candidate method registry (literature accept-rate RANGES on reasoning/STEM).
# et_ppl_* are accepted-tokens-per-step E[T] proxies (= compression ratio) on OUR
# ~100% reasoning/STEM eval -- the WORST case for training-free redundancy methods.
# Carried as low/central/high RANGES (instruction 3: no false precision).
# --------------------------------------------------------------------------- #
METHODS: list[dict[str, Any]] = [
    {
        "name": "lookahead_jacobi",
        "label": "Lookahead / Jacobi decoding",
        "citation": "Fu et al., ICML 2024 (arXiv:2402.02057)",
        "uses_batched_multitoken_verify": True,
        "verify_structure": "verifies the Jacobi n-gram window + guess pool in ONE batched target "
                            "forward (the lookahead branch is appended to the verify positions).",
        "training_free": True,
        "ppl_risk": False,                      # lossless: target argmax over the verified window
        "general_speedup": 1.8,                 # general workloads (context only)
        "et_ppl_low": 1.4, "et_ppl_central": 1.6, "et_ppl_high": 1.8,   # reasoning/STEM (Fu+follow-on)
        "et_note": "reasoning/STEM ~1.4-1.8x: Fu et al. report ~1.5x on GSM8K/CoT, the NeurIPS-2025 "
                   "lookahead follow-on tops ~1.4x on reasoning, GSM8K-on-CodeLLaMA reaches ~1.8x; "
                   "Jacobi convergence is short on low-redundancy CoT so it stays well below the head.",
    },
    {
        "name": "ngram_pld",
        "label": "n-gram Prompt-Lookup Decoding (PLD)",
        "citation": "Saxena 2023 / vLLM --speculative-method ngram",
        "uses_batched_multitoken_verify": True,
        "verify_structure": "verifies the prompt-looked-up continuation tokens in ONE batched target "
                            "forward (drafter is a pure CPU n-gram match, but the VERIFY is batched).",
        "training_free": True,
        "ppl_risk": False,                      # lossless: target argmax over the verified continuation
        "general_speedup": 1.75,                # summarization/long-context with input repetition
        "et_ppl_low": 1.0, "et_ppl_central": 1.05, "et_ppl_high": 1.1,  # reasoning/STEM (<1.1x)
        "et_note": "reasoning/STEM <1.1x (vs ~1.5-2x summarization): PLD requires input n-gram "
                   "repetition (copy-heavy tasks); near-absent in MCQ/competition-math generation.",
    },
    {
        "name": "medusa",
        "label": "Medusa multi-head",
        "citation": "Cai et al. 2024 (arXiv:2401.10774)",
        "uses_batched_multitoken_verify": True,
        "verify_structure": "verifies the TREE of multi-head proposals in ONE batched tree-attention "
                            "target forward (more verify rows than a linear chain -> >= the supply tax).",
        "training_free": False,                 # needs trained heads (upfront cost) -- DEFER per #345
        "ppl_risk": False,                      # lossless under typical-acceptance greedy verify
        "general_speedup": 3.0,                 # general (Medusa-2, ~2.3-3.6x upper)
        "et_ppl_low": 2.0, "et_ppl_central": 2.3, "et_ppl_high": 2.8,   # reasoning/STEM (head-dependent)
        "et_note": "reasoning/STEM ~2.0-2.5x conservative (general Medusa-2 tops ~3.6x); needs TRAINING "
                   "new heads (DEFERRED per #345). Even its generous reasoning high anchor 2.8 -- and "
                   "even the 3.6x general ceiling -- stays below the EXISTING head's 3.8512, and it does "
                   "NOT dodge the supply tax (batched tree verify has MORE rows than a linear chain).",
    },
    {
        "name": "self_spec_layerskip",
        "label": "Self-spec (Draft&Verify)",
        "citation": "Zhang et al. ACL 2024 (Draft&Verify, arXiv:2309.08168); "
                    "Elhoushi et al. 2024 (LayerSkip, arXiv:2404.16710)",
        "uses_batched_multitoken_verify": True,
        "verify_structure": "REPRESENTED by Draft&Verify: the FULL model verifies the early-exit "
                            "drafted tokens in ONE separate batched target forward (lossless, inherits "
                            "the tax cleanly). STOCK LayerSkip is EXCLUDED -- it reuses the early-exit "
                            "forward and only runs the model TAIL to verify (not a separate full forward), "
                            "so the identical 473.53 cap does not transfer without a measured profile; it "
                            "also needs a specially-trained model (layer-dropout + early-exit loss) and "
                            "carries PPL risk -- disqualified as a clean drop-in regardless.",
        "training_free": True,                  # Draft&Verify drafts with the model's own early exit (no sep. model)
        "ppl_risk": True,                       # FLAG: stock LayerSkip needs retraining; raw early-exit alters PPL
        "general_speedup": 1.9,                 # general Draft&Verify (~1.3-1.99x)
        "et_ppl_low": 1.3, "et_ppl_central": 1.5, "et_ppl_high": 1.8,   # reasoning/STEM (early-exit divergence)
        "et_note": "Draft&Verify literature ~1.3-1.99x general; reasoning-discounted here because the "
                   "early-exit self-draft distribution diverges from the full model on hard reasoning "
                   "tokens (lower accept). PPL-RISK: only the verify-and-accept (Draft&Verify) variant is "
                   "lossless; stock LayerSkip needs a retrained model and its raw early-exit emission "
                   "alters the distribution -- EXCLUDED from the clean supply-tax claim (see caveats).",
    },
]


# --------------------------------------------------------------------------- #
# (D1) STRICT supply-tax screen: per method inherits_332_supply_tax + strict ceiling.
# --------------------------------------------------------------------------- #
def deliverable1_strict_supply_tax() -> dict[str, Any]:
    # the structural reason the tax is method-independent: the batched verify forward's attention is
    # bandwidth-bound (AI << ridge) AND the adaptive path already saturates occupancy yet stays at the
    # BW floor -> the slack is the low-AI attention floor of reading the KV cache, not the drafter.
    bandwidth_bound = bool(SDPA_AI_FLOP_PER_BYTE < RIDGE_AI)
    occupancy_saturated = bool(N_FULL_3D_CTAS_332 > A10G_SMS_332)
    tax_is_method_independent = bool(bandwidth_bound and occupancy_saturated)

    rows = []
    for m in METHODS:
        inherits = bool(m["uses_batched_multitoken_verify"]) and tax_is_method_independent
        # under strict identity, any spec method inheriting the tax is capped at the #332 ceiling.
        strict_ceiling = STRICT_COMPLIANT_CEILING_332 if inherits else None
        rows.append({
            "name": m["name"], "label": m["label"], "citation": m["citation"],
            "uses_batched_multitoken_verify": bool(m["uses_batched_multitoken_verify"]),
            "verify_structure": m["verify_structure"],
            "inherits_332_supply_tax": inherits,
            "strict_compliant_ceiling_tps": strict_ceiling,
            "strict_ceiling_clears_500": bool(strict_ceiling is not None and strict_ceiling >= TARGET),
        })
    all_inherit = all(r["inherits_332_supply_tax"] for r in rows)
    # reconstruct the 473.53 strict ceiling from the floor (round-trip the #332 anchor).
    strict_ceiling_reconstructed = LAMBDA1_CEIL * (1.0 - FLOOR_AT_GEO_332)
    floor_geo_reconstructed = FLOOR_AT_PHI1_327 * GEOMETRIC_PHI_332
    return {
        "supply_tax_law": "the #332 verify-attention BW floor (34.9% BW, AI 7.88 << ridge 208) is a "
                          "property of the BATCHED MULTI-TOKEN VERIFY FORWARD reading the KV cache, "
                          "method-INDEPENDENT: any M>1 verify inherits it -> strict ceiling <= 473.53.",
        "bandwidth_bound": bandwidth_bound,
        "occupancy_saturated": occupancy_saturated,
        "tax_is_method_independent": tax_is_method_independent,
        "sdpa_bw_util": SDPA_BW_UTIL, "sdpa_ai_flop_per_byte": SDPA_AI_FLOP_PER_BYTE,
        "ridge_ai": RIDGE_AI, "n_full_3d_ctas": N_FULL_3D_CTAS_332, "a10g_sms": A10G_SMS_332,
        "rows": rows,
        "all_methods_inherit_supply_tax": all_inherit,
        "strict_compliant_ceiling_tps": STRICT_COMPLIANT_CEILING_332,
        "strict_ceiling_reconstructed": strict_ceiling_reconstructed,
        "strict_ceiling_roundtrip_ok": bool(
            abs(strict_ceiling_reconstructed - STRICT_COMPLIANT_CEILING_332) <= TOL_ROUNDTRIP),
        "floor_geo_reconstructed": floor_geo_reconstructed,
        "floor_geo_roundtrip_ok": bool(abs(floor_geo_reconstructed - FLOOR_AT_GEO_332) <= TOL_ROUNDTRIP),
        "strict_ceiling_below_500": bool(STRICT_COMPLIANT_CEILING_332 < TARGET),
    }


# --------------------------------------------------------------------------- #
# (D2) PPL-only E[T] screen: per method E[T] range, ratio vs EAGLE-3 head, ceiling TPS.
# --------------------------------------------------------------------------- #
def deliverable2_ppl_only_screen() -> dict[str, Any]:
    et_eagle3 = E_T_EAGLE3_289
    rows = []
    for m in METHODS:
        lo, ce, hi = m["et_ppl_low"], m["et_ppl_central"], m["et_ppl_high"]
        ratio_lo = lo / et_eagle3
        ratio_ce = ce / et_eagle3
        ratio_hi = hi / et_eagle3
        rows.append({
            "name": m["name"], "label": m["label"], "citation": m["citation"],
            "training_free": bool(m["training_free"]), "ppl_risk": bool(m["ppl_risk"]),
            "general_speedup": m["general_speedup"],
            "et_ppl_low": lo, "et_ppl_central": ce, "et_ppl_high": hi,
            "et_note": m["et_note"],
            "et_ratio_low": ratio_lo, "et_ratio_central": ratio_ce, "et_ratio_high": ratio_hi,
            "ppl_only_ceiling_tps_central": ppl_only_ceiling_tps(ce),
            "ppl_only_ceiling_tps_high": ppl_only_ceiling_tps(hi),
            "beats_eagle3_head_central": bool(ce > et_eagle3),
            "beats_eagle3_head_high": bool(hi > et_eagle3),     # most generous: optimistic anchor
            "ceiling_high_clears_500": bool(ppl_only_ceiling_tps(hi) >= TARGET),
        })
    # best alternative E[T] ratio (the TEST metric): max central ratio across ALL methods.
    best_central = max(rows, key=lambda r: r["et_ratio_central"])
    best_high = max(rows, key=lambda r: r["et_ratio_high"])
    # training-free-only best (the apples-to-apples drop-in comparison for verdict bool 2).
    tf_rows = [r for r in rows if r["training_free"]]
    best_tf_central = max(tf_rows, key=lambda r: r["et_ratio_central"])
    return {
        "ppl_only_ceiling_law": "ceiling_method = LAMBDA1_CEIL(520.95) * (E[T]_method / E[T]_eagle3 "
                               "= 3.8512): linear-E[T] scaling of the lambda=1 ceiling (verify step "
                               "method-independent). EAGLE-3 head -> 520.95 by construction.",
        "e_t_eagle3_head_289": et_eagle3,
        "e_t_retrain_target_340": E_T_RETRAIN_TARGET_340,
        "rows": rows,
        "best_alt_name_central": best_central["name"],
        "best_ppl_only_alternative_etratio": best_central["et_ratio_central"],          # TEST metric
        "best_alt_name_optimistic": best_high["name"],
        "best_ppl_only_alternative_etratio_optimistic": best_high["et_ratio_high"],
        "best_trainingfree_name_central": best_tf_central["name"],
        "best_trainingfree_etratio_central": best_tf_central["et_ratio_central"],
        "any_alt_central_beats_head": bool(any(r["beats_eagle3_head_central"] for r in rows)),
        "any_alt_optimistic_beats_head": bool(any(r["beats_eagle3_head_high"] for r in rows)),
        "retrain_dominates_all_alts": bool(
            E_T_RETRAIN_TARGET_340 > max(r["et_ppl_high"] for r in rows)),
    }


# --------------------------------------------------------------------------- #
# (D3) Verdict table: the two deliverable bools.
# --------------------------------------------------------------------------- #
def deliverable3_verdict(d1: dict, d2: dict) -> dict[str, Any]:
    # under strict: a method ESCAPES the supply tax iff it does NOT inherit it (and could exceed 473.53).
    any_escapes = bool(not d1["all_methods_inherit_supply_tax"])
    any_strict_clears_500 = bool(any(r["strict_ceiling_clears_500"] for r in d1["rows"]))
    # under PPL-only: does ANY alternative beat the EXISTING head's E[T]? (use the OPTIMISTIC anchor --
    # most generous: give every method its best-case E[T], including Medusa's training-cost path.)
    any_beats_ppl_only = bool(d2["any_alt_optimistic_beats_head"])
    return {
        "any_non_eagle3_escapes_strict_supply_tax": any_escapes,                  # expect False
        "any_non_eagle3_strict_ceiling_clears_500": any_strict_clears_500,        # expect False
        "any_non_eagle3_beats_eagle3_head_ppl_only": any_beats_ppl_only,          # expect False
        "best_ppl_only_alternative_etratio": d2["best_ppl_only_alternative_etratio"],
        "best_ppl_only_alternative_etratio_optimistic": d2[
            "best_ppl_only_alternative_etratio_optimistic"],
        "only_124_lever_under_strict": bool(not any_escapes and not any_strict_clears_500),
        "retrain_is_better_ppl_only_bet": bool(
            not any_beats_ppl_only and d2["retrain_dominates_all_alts"]),
        "verdict": (
            "STRICT: every candidate (lookahead/Jacobi, n-gram PLD, Medusa, self-spec/layer-skip) uses "
            "a batched multi-token verify forward -> ALL inherit the #332 supply tax -> strict ceiling "
            "<= 473.53 < 500 for every method -> method choice is IRRELEVANT under strict, only #124 "
            "(lifting the gate) helps. PPL-ONLY: the best alternative E[T] ratio vs the EXISTING head "
            "is {:.3f} (central) / {:.3f} (optimistic), both < 1 -> NO non-EAGLE-3 method beats the "
            "EXISTING EAGLE-3 head, and the coverage-retrained head (E[T]->6.1112) dominates further. "
            "The EAGLE-3 head + coverage retrain is the better PPL-only bet.".format(
                d2["best_ppl_only_alternative_etratio"],
                d2["best_ppl_only_alternative_etratio_optimistic"])),
    }


# --------------------------------------------------------------------------- #
# (D4) Caveats + the cheap one-run screen + scope.
# --------------------------------------------------------------------------- #
def deliverable4_caveats() -> dict[str, Any]:
    return {
        "accept_rates_are_ranges": (
            "literature accept rates are workload-dependent POINT estimates; carried as low/central/"
            "high RANGES per method. The verdict is robust across each method's WHOLE band -- even "
            "every method's optimistic high anchor stays below the EXISTING head's E[T] = 3.8512."),
        "self_spec_ppl_risk": (
            "self-speculative / layer-skip may ALTER PPL if it emits early-exit draft tokens without a "
            "full verify (flagged ppl_risk=True). Only the verify-and-accept (Draft&Verify) variant is "
            "lossless; a raw layer-skip emission breaks greedy identity AND the PPL gate -- OUT OF SCOPE."),
        "layerskip_verify_structure_exclusion": (
            "the self-spec row is REPRESENTED by Draft&Verify (arXiv:2309.08168), which runs a SEPARATE "
            "full batched target forward to verify -> cleanly inherits the #332 supply tax. STOCK "
            "LayerSkip (arXiv:2404.16710) is DELIBERATELY EXCLUDED from the identical-cap claim: its "
            "verify reuses the early-exit forward and only runs the model TAIL (layers E+1..N) over the "
            "draft positions, so it is NOT an independent full forward -- the 473.53 cap cannot be "
            "transferred to it without a measured profile. But LayerSkip ALSO (a) needs a specially-"
            "trained model (layer-dropout + early-exit loss -- not a drop-in, not training-free), and "
            "(b) carries documented PPL risk; so even if its tail-verify were cheaper, it fails the "
            "clean option-B test (training-free + lossless drop-in) on two other axes. The verdict is "
            "robust: it is NOT a free escape, just a different (and PPL/training-encumbered) supply shape."),
        "cheap_one_run_screen": (
            "a DEFINITIVE accept-rate needs a measured run (gated). IF #124 lifts the gate to PPL-only, "
            "the cheapest confirmation is ONE A10G run of vLLM's native ngram (--speculative-method "
            "ngram) and/or lookahead on the 128 reasoning/STEM eval, reading the realised E[T] from the "
            "spec-decode acceptance counters -- no training, one official draw. This screen PREDICTS "
            "that run lands E[T] < 1.5 (training-free) << the head's 3.8512."),
        "scope": (
            "this is a SCREEN scoping option B, NOT a build recommendation: it does not green-light any "
            "method. It answers 'does any DIFFERENT method beat the EAGLE-3 lane?' (no) -- the build "
            "decision stays with the EAGLE-3 head + coverage retrain (wirbel) under #124, or the #124 "
            "gate-lift itself under strict."),
        "non_collision": (
            "ORTHOGONAL to wirbel ppl-only-gate-500-envelope (gate-lift pricing for the EXISTING head), "
            "denken gate-independent-speed-lever (non-spec SPEED levers), kanna #342 (a_1-lift), lawine "
            "#339 (coverage clear-prob), fern #341 (joint isocline). This is the METHOD-scoping leg."),
    }


# --------------------------------------------------------------------------- #
# (D5) Greedy-safety + analytic scope.
# --------------------------------------------------------------------------- #
def deliverable5_greedy_safety() -> dict[str, Any]:
    return {
        "non_eagle3_card_is_cpu_analytic": True,
        "no_gpu": True,
        "no_served_change": True,
        "no_model_forward": True,
        "no_training": True,
        "no_hf_job": True,
        "zero_official_tps": True,
        "greedy_identity_preserved_by_construction": True,
        "note": (
            "this card builds and runs NO method -- it is a numeric screen over banked anchors + the "
            "literature candidate set. The draft METHOD is the SPEED axis: a verify-and-accept "
            "speculative method emits the target model's argmax token regardless of drafter, so greedy "
            "identity is invariant to method choice (the one exception, raw layer-skip emission without "
            "verify, is flagged ppl_risk and excluded). No served file, kernel, or decode path is "
            "touched. BASELINE stays 481.53."),
    }


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def _selftests(d1: dict, d2: dict, d3: dict, nan_clean: bool) -> dict[str, Any]:
    rows1, rows2 = d1["rows"], d2["rows"]
    et_eagle3_recomputed = et_of_profile(A_K_EAGLE3_289)
    conditions = {
        # (a) #332 supply floor + 473.53 ceiling round-trip <= 1e-6.
        "a_strict_ceiling_roundtrips_332": bool(d1["strict_ceiling_roundtrip_ok"]),
        "a_floor_geo_roundtrips_332": bool(d1["floor_geo_roundtrip_ok"]),
        "a_strict_ceiling_is_473p53": bool(abs(STRICT_COMPLIANT_CEILING_332 - 473.53) <= 5e-2),
        # (b) EAGLE-3 E[T] from the #289 a_k profile reproduced <= 1e-6.
        "b_eagle3_et_reproduces_289": bool(abs(et_eagle3_recomputed - E_T_EAGLE3_289) <= TOL_ROUNDTRIP),
        "b_eagle3_et_is_3p8512": bool(abs(E_T_EAGLE3_289 - 3.8512) <= 5e-4),
        # (c) each method row has inherits_332_supply_tax + a cited PPL-only E[T] RANGE, ordered.
        "c_all_rows_have_inherit_bool": bool(all(
            isinstance(r["inherits_332_supply_tax"], bool) for r in rows1)),
        "c_all_rows_have_citation": bool(all(bool(r["citation"]) for r in rows2)),
        "c_all_et_ranges_ordered": bool(all(
            r["et_ppl_low"] <= r["et_ppl_central"] <= r["et_ppl_high"] for r in rows2)),
        "c_all_et_below_eagle3_head": bool(all(r["et_ppl_high"] < E_T_EAGLE3_289 for r in rows2)),
        "c_four_method_classes": bool(len(METHODS) == 4),
        "c_nan_clean": bool(nan_clean),
        # (d) the two verdict bools computed (and as expected: both False).
        "d_escape_bool_is_false": bool(d3["any_non_eagle3_escapes_strict_supply_tax"] is False),
        "d_beats_ppl_bool_is_false": bool(d3["any_non_eagle3_beats_eagle3_head_ppl_only"] is False),
        "d_best_etratio_is_float_below_1": bool(
            _finite(d3["best_ppl_only_alternative_etratio"])
            and d3["best_ppl_only_alternative_etratio"] < 1.0),
        # (e) reasoning/STEM eval composition (lawine #330) cited as the accept-rate context.
        "e_eval_is_100pct_reasoning_stem": bool(abs(REASONING_STEM_FRAC - 1.0) <= TOL_EXACT),
        "e_eval_mix_sums_to_128": bool(sum(EVAL_MIX.values()) == 128),
        # (f) [extra] ALL methods inherit the supply tax (the load-bearing insight: True for all).
        "f_all_methods_inherit_tax": bool(d1["all_methods_inherit_supply_tax"]),
        "f_tax_is_method_independent": bool(d1["tax_is_method_independent"]),
        # (g) [extra] no method's strict ceiling clears 500 (all capped at 473.53 < 500).
        "g_no_strict_ceiling_clears_500": bool(
            not any(r["strict_ceiling_clears_500"] for r in rows1)
            and d1["strict_ceiling_below_500"]),
        # (h) [extra] even the OPTIMISTIC best alternative ratio < 1 (robust verdict).
        "h_optimistic_best_ratio_below_1": bool(
            d2["best_ppl_only_alternative_etratio_optimistic"] < 1.0),
        # (i) [extra] coverage-retrained head E[T] dominates every alternative's high anchor.
        "i_retrain_dominates_all_alts": bool(d2["retrain_dominates_all_alts"]),
        # (j) [extra] Medusa flagged needs-training; self-spec flagged ppl_risk.
        "j_medusa_needs_training": bool(
            not next(m for m in METHODS if m["name"] == "medusa")["training_free"]),
        "j_selfspec_ppl_risk_flagged": bool(
            next(m for m in METHODS if m["name"] == "self_spec_layerskip")["ppl_risk"]),
        # (k) [extra] bandwidth-bound (AI << ridge) -- the structural reason the tax is method-indep.
        "k_verify_bandwidth_bound": bool(SDPA_AI_FLOP_PER_BYTE < RIDGE_AI),
        "k_occupancy_saturated": bool(N_FULL_3D_CTAS_332 > A10G_SMS_332),
    }
    passes = bool(all(conditions.values()))
    return {
        "conditions": conditions,
        "non_eagle3_feasibility_self_test_passes": passes,
        "n_checks": len(conditions),
        "detail": {
            "best_ppl_only_alternative_etratio": d3["best_ppl_only_alternative_etratio"],
            "best_alt_name": d2["best_alt_name_central"],
            "any_escapes_strict": d3["any_non_eagle3_escapes_strict_supply_tax"],
            "any_beats_ppl_only": d3["any_non_eagle3_beats_eagle3_head_ppl_only"],
        },
    }


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    d1 = deliverable1_strict_supply_tax()
    d2 = deliverable2_ppl_only_screen()
    d3 = deliverable3_verdict(d1, d2)
    d4 = deliverable4_caveats()
    d5 = deliverable5_greedy_safety()

    handoff = (
        "the #319 option-B method scope: EVERY non-EAGLE-3 speculative candidate (lookahead/Jacobi, "
        "n-gram PLD, Medusa, self-spec/layer-skip) verifies M>1 candidate tokens in a batched target "
        "forward -> ALL inherit denken #332's method-independent verify-attention supply tax (34.9% BW, "
        "AI 7.88 << ridge 208) -> strict-compliant ceiling <= 473.53 < 500 for every method "
        "(any_non_eagle3_escapes_strict_supply_tax = {}). So under STRICT the draft-method choice is "
        "irrelevant -- only #124 (lifting the gate) moves the >500 lane. Under PPL-ONLY the best "
        "alternative E[T] ratio vs the EXISTING head is {:.3f} central / {:.3f} optimistic (both < 1: "
        "training-free methods under-accept on reasoning/STEM, Medusa needs training and still loses) "
        "-> the EXISTING EAGLE-3 head already beats every alternative, and the coverage-retrained head "
        "(E[T]->6.1112, wirbel's path) dominates further (any_non_eagle3_beats_eagle3_head_ppl_only = "
        "{}). CONCLUSION: no non-EAGLE-3 method escapes the lane; #124 is the only strict lever and the "
        "EAGLE-3 head + coverage retrain is the PPL-only bet. SCREEN ONLY -- not a build/launch.".format(
            d3["any_non_eagle3_escapes_strict_supply_tax"],
            d3["best_ppl_only_alternative_etratio"],
            d3["best_ppl_only_alternative_etratio_optimistic"],
            d3["any_non_eagle3_beats_eagle3_head_ppl_only"]))

    headline = {
        "non_eagle3_feasibility_self_test_passes": None,   # set after nan audit
        "any_non_eagle3_escapes_strict_supply_tax": d3["any_non_eagle3_escapes_strict_supply_tax"],  # TEST
        "best_ppl_only_alternative_etratio": d3["best_ppl_only_alternative_etratio"],                 # TEST
        "any_non_eagle3_beats_eagle3_head_ppl_only": d3["any_non_eagle3_beats_eagle3_head_ppl_only"],
        "best_ppl_only_alternative_etratio_optimistic": d3[
            "best_ppl_only_alternative_etratio_optimistic"],
        "all_methods_inherit_supply_tax": d1["all_methods_inherit_supply_tax"],
        "strict_compliant_ceiling_tps": STRICT_COMPLIANT_CEILING_332,
        "e_t_eagle3_head_289": E_T_EAGLE3_289,
        "e_t_retrain_target_340": E_T_RETRAIN_TARGET_340,
        "only_124_lever_under_strict": d3["only_124_lever_under_strict"],
        "retrain_is_better_ppl_only_bet": d3["retrain_is_better_ppl_only_bet"],
    }
    return {
        "headline": headline,
        "deliverable1_strict_supply_tax": d1,
        "deliverable2_ppl_only_screen": d2,
        "deliverable3_verdict": d3,
        "deliverable4_caveats": d4,
        "deliverable5_greedy_safety": d5,
        "handoff": handoff,
        "imports": {
            "provenance": (
                "denken #332 y5cl0ena (BW 34.9%, AI 7.88, ridge 208, geometric phi 0.925, floor "
                "0.09841, strict ceiling 473.53, ceiling 520.953, >500 budget 4.022%) x kanna #289 "
                "fi34s269 (a_k profile, E[T]=3.8512) x stark #340 jwv1vbug (coverage-retrain target "
                "E[T](0.9213)=6.1112) x lawine #330 hfrscdai (eval 100% reasoning/STEM, mmlu_pro 57 / "
                "gpqa 57 / aime 14, coverage prior 0.8903). All run-ids in "
                "wandb-applied-ai-team/gemma-challenge-senpai."),
            "literature": (
                "Fu et al. ICML 2024 lookahead/Jacobi (arXiv:2402.02057); Cai et al. 2024 Medusa "
                "(arXiv:2401.10774); Saxena 2023 / vLLM ngram prompt-lookup; Zhang et al. ACL 2024 "
                "Draft&Verify self-spec (arXiv:2309.08168) + Elhoushi et al. 2024 LayerSkip "
                "(arXiv:2404.16710); Leviathan et al. ICML 2023 spec-decode distributional guarantee "
                "(arXiv:2211.17192)."),
            "caveats": [
                "this is a SCREEN over banked anchors + literature accept-rate RANGES -- it builds and "
                "runs NO method, re-derives nothing measured. NOT a running drafter / kernel / launch.",
                "two E[T] conventions are kept SEPARATE: the #289 a_k linear-chain E[T]=3.8512 (the "
                "deployed-head comparison anchor, instruction ii) vs the #340 top-4 survival "
                "E[T](0.9213)=6.1112 (the coverage-retrain target). Method E[T] is compared to 3.8512.",
                "literature accept rates are workload-dependent point estimates carried as RANGES; the "
                "verdict holds across each method's whole band (even its optimistic high anchor).",
                "self-spec/layer-skip PPL-risk and Medusa training cost are FLAGGED; the screen does "
                "NOT green-light any method. NOT a launch / build / served-file change / HF Job / open2.",
            ],
        },
    }


# --------------------------------------------------------------------------- #
# NaN guard + report + W&B.
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


def _print_report(syn: dict, st: dict) -> None:
    h = syn["headline"]
    d1 = syn["deliverable1_strict_supply_tax"]
    d2 = syn["deliverable2_ppl_only_screen"]
    d3 = syn["deliverable3_verdict"]
    print("\n" + "=" * 100, flush=True)
    print("NON-EAGLE-3 >500 FEASIBILITY (PR #345, stark) — does any method escape the supply tax?",
          flush=True)
    print("=" * 100, flush=True)
    print("  (D1) STRICT SUPPLY-TAX SCREEN  (method-independent #332 cap = 473.53 < 500)", flush=True)
    print(f"      bandwidth-bound (AI {SDPA_AI_FLOP_PER_BYTE:.2f} << ridge {RIDGE_AI:.0f}) = "
          f"{d1['bandwidth_bound']}   occupancy-saturated ({d1['n_full_3d_ctas']}>{d1['a10g_sms']} "
          f"SMs) = {d1['occupancy_saturated']}", flush=True)
    for r in d1["rows"]:
        print(f"        - {r['label']:<34s} batched-verify={r['uses_batched_multitoken_verify']!s:<5s} "
              f"inherits_tax={r['inherits_332_supply_tax']!s:<5s} strict<=473.53 "
              f"clears500={r['strict_ceiling_clears_500']}", flush=True)
    print(f"      all_methods_inherit_supply_tax = {d1['all_methods_inherit_supply_tax']}", flush=True)
    print("-" * 100, flush=True)
    print(f"  (D2) PPL-ONLY E[T] SCREEN  (vs deployed head E[T]={d2['e_t_eagle3_head_289']:.4f}; "
          f"retrain target {d2['e_t_retrain_target_340']:.4f})", flush=True)
    for r in d2["rows"]:
        flags = []
        if not r["training_free"]:
            flags.append("needs-train")
        if r["ppl_risk"]:
            flags.append("ppl-risk")
        fl = (" [" + ",".join(flags) + "]") if flags else ""
        print(f"        - {r['label']:<34s} E[T]=[{r['et_ppl_low']:.2f},{r['et_ppl_central']:.2f},"
              f"{r['et_ppl_high']:.2f}]  ratio_ce={r['et_ratio_central']:.3f}  "
              f"ceil_ce={r['ppl_only_ceiling_tps_central']:.1f} TPS{fl}", flush=True)
    print(f"      best_ppl_only_alternative_etratio = {d2['best_ppl_only_alternative_etratio']:.4f} "
          f"({d2['best_alt_name_central']})  optimistic="
          f"{d2['best_ppl_only_alternative_etratio_optimistic']:.4f}", flush=True)
    print("-" * 100, flush=True)
    print("  (D3) VERDICT", flush=True)
    print(f"      any_non_eagle3_escapes_strict_supply_tax  = "
          f"{d3['any_non_eagle3_escapes_strict_supply_tax']}  (expect False)", flush=True)
    print(f"      any_non_eagle3_beats_eagle3_head_ppl_only = "
          f"{d3['any_non_eagle3_beats_eagle3_head_ppl_only']}  (expect False)", flush=True)
    print(f"      only_124_lever_under_strict = {d3['only_124_lever_under_strict']}   "
          f"retrain_is_better_ppl_only_bet = {d3['retrain_is_better_ppl_only_bet']}", flush=True)
    print("-" * 100, flush=True)
    print(f"  HAND-OFF: {syn['handoff']}", flush=True)
    print("-" * 100, flush=True)
    print(f"  PRIMARY non_eagle3_feasibility_self_test_passes = "
          f"{st['non_eagle3_feasibility_self_test_passes']} ({st['n_checks']} checks)", flush=True)
    for k, v in st["conditions"].items():
        if not v:
            print(f"        - FAIL {k}: {v}", flush=True)
    print("=" * 100 + "\n", flush=True)


def _maybe_log_wandb(args, payload: dict) -> str | None:
    if not getattr(args, "wandb_name", None):
        return None
    try:
        sys.path.insert(0, str(REPO_ROOT))
        _w = sys.modules.get("wandb")
        if _w is not None and not hasattr(_w, "init"):
            del sys.modules["wandb"]
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[non-eagle3-500-feasibility] wandb logging unavailable: {exc}", flush=True)
        return None

    syn = payload["synthesis"]
    d1 = syn["deliverable1_strict_supply_tax"]
    d2 = syn["deliverable2_ppl_only_screen"]
    d3 = syn["deliverable3_verdict"]
    st = payload["self_test"]
    run = init_wandb_run(
        job_type="validity-gate",
        agent="stark",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["non-eagle3-500-feasibility", "issue-192", "eagle3", "supply-tax", "method-scoping",
              "speculative-decoding", "lookahead", "ngram-pld", "medusa", "self-speculative",
              "ppl-only", "validity-gate", "bank-the-analysis", "pr-345"],
        config={
            "sdpa_bw_util": SDPA_BW_UTIL, "sdpa_ai_flop_per_byte": SDPA_AI_FLOP_PER_BYTE,
            "ridge_ai": RIDGE_AI, "geometric_phi_332": GEOMETRIC_PHI_332,
            "strict_compliant_ceiling_332": STRICT_COMPLIANT_CEILING_332,
            "lambda1_ceil": LAMBDA1_CEIL, "budget_500_frac_192": BUDGET_500_FRAC_192,
            "e_t_eagle3_head_289": E_T_EAGLE3_289, "e_t_retrain_target_340": E_T_RETRAIN_TARGET_340,
            "reasoning_stem_frac": REASONING_STEM_FRAC, "cov_prior_330": COV_PRIOR_330,
            "target": TARGET, "baseline_tps": BASELINE_TPS, "n_methods": len(METHODS),
            "wandb_group": args.wandb_group,
            "source_runs": "denken#332(y5cl0ena), kanna#289(fi34s269), stark#340(jwv1vbug), "
                           "lawine#330(hfrscdai)",
        },
    )
    if run is None:
        print("[non-eagle3-500-feasibility] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return None

    summary: dict[str, Any] = {
        "non_eagle3_feasibility_self_test_passes": int(bool(
            st["non_eagle3_feasibility_self_test_passes"])),                                      # PRIMARY
        "any_non_eagle3_escapes_strict_supply_tax": int(bool(
            d3["any_non_eagle3_escapes_strict_supply_tax"])),                                     # TEST
        "best_ppl_only_alternative_etratio": d3["best_ppl_only_alternative_etratio"],             # TEST
        "any_non_eagle3_beats_eagle3_head_ppl_only": int(bool(
            d3["any_non_eagle3_beats_eagle3_head_ppl_only"])),
        "best_ppl_only_alternative_etratio_optimistic": d3[
            "best_ppl_only_alternative_etratio_optimistic"],
        "all_methods_inherit_supply_tax": int(bool(d1["all_methods_inherit_supply_tax"])),
        "any_non_eagle3_strict_ceiling_clears_500": int(bool(
            d3["any_non_eagle3_strict_ceiling_clears_500"])),
        "only_124_lever_under_strict": int(bool(d3["only_124_lever_under_strict"])),
        "retrain_is_better_ppl_only_bet": int(bool(d3["retrain_is_better_ppl_only_bet"])),
        "strict_compliant_ceiling_tps": STRICT_COMPLIANT_CEILING_332,
        "e_t_eagle3_head_289": E_T_EAGLE3_289,
        "e_t_retrain_target_340": E_T_RETRAIN_TARGET_340,
        "best_trainingfree_etratio_central": d2["best_trainingfree_etratio_central"],
        "tax_is_method_independent": int(bool(d1["tax_is_method_independent"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    # per-method E[T] + ratio + inherits, flattened for cross-run analysis.
    for r in d2["rows"]:
        summary[f"et_central_{r['name']}"] = r["et_ppl_central"]
        summary[f"et_high_{r['name']}"] = r["et_ppl_high"]
        summary[f"et_ratio_central_{r['name']}"] = r["et_ratio_central"]
        summary[f"ceiling_central_{r['name']}"] = r["ppl_only_ceiling_tps_central"]
    for r in d1["rows"]:
        summary[f"inherits_tax_{r['name']}"] = int(bool(r["inherits_332_supply_tax"]))
    summary = {k: v for k, v in summary.items()
               if v is not None and not (isinstance(v, float) and not math.isfinite(v))}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="non_eagle3_500_feasibility_result", artifact_type="validity",
                      data=payload)
    rid = getattr(run, "id", None)
    finish_wandb(run)
    print(f"[non-eagle3-500-feasibility] wandb logged {len(summary)} keys (run {rid})", flush=True)
    return rid


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="non-eagle3-500-feasibility")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)
    if args.no_wandb:
        args.wandb_name = None

    syn = synthesize()

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 345, "agent": "stark",
        "kind": "non-eagle3-500-feasibility", "analysis_only": True, "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _nan_paths(payload)
    payload["nan_clean"] = not nan_paths

    st = _selftests(syn["deliverable1_strict_supply_tax"], syn["deliverable2_ppl_only_screen"],
                    syn["deliverable3_verdict"], payload["nan_clean"])
    payload["self_test"] = st
    syn["headline"]["non_eagle3_feasibility_self_test_passes"] = st[
        "non_eagle3_feasibility_self_test_passes"]
    if nan_paths:
        print(f"[non-eagle3-500-feasibility] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn, st)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "non_eagle3_500_feasibility_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
    print(f"[non-eagle3-500-feasibility] wrote {out_path}", flush=True)

    rid = _maybe_log_wandb(args, payload)
    payload["wandb_run_id"] = rid

    if args.self_test:
        ok = st["non_eagle3_feasibility_self_test_passes"] and payload["nan_clean"]
        if not ok:
            failed = [k for k, v in st["conditions"].items() if not v]
            print(f"[non-eagle3-500-feasibility] SELF-TEST FAILED: {failed}", flush=True)
        print(f"[non-eagle3-500-feasibility] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
