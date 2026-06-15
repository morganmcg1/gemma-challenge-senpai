#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #359 (kanna) -- Strict identity-preserving step-shave stack: close the gap to 500?

THE STRATEGIC QUESTION
----------------------
Under the #319 strict-lock, the ONLY lever family that touches neither quantization
nor speculation -- so strict greedy-token-identity is preserved BY CONSTRUCTION -- is
identity-preserving step-shaving on the deployed batch=1 step:

  (1) CUDA-graph capture            (launch / host-gap removal)
  (2) kernel fusion                 (norm / residual / activation epilogue folds)
  (3) attention-backend selection   (FA2-SW <-> FlashInfer / Triton kernel tuning)

denken #344 (`sxltbech`, gate_independent_speed_lever) gave the step waterfall
(normalized 1218.2us): gate_up 43.6% / down_proj 21.8% / draft-K7 12.0% / attention
9.5% / qkv 6.5%. The GEMM terms are weight-read-BW-bound (immovable without quant,
identity + physics), but the launch/host/fusion overhead and the attention-kernel
choice are the candidate identity-safe shaves. Question: how much can the STACKED
identity-preserving step-shave move the 1218us step, and does it close the residual
~23-TPS gap to 500 on top of wirbel #354's custom-kernel-compliant ~477?

THE ANSWER (short)
------------------
NO. The decomposition looks like it has ~6.3% of "shaveable overhead" (the attention
exposed slack 6.19% + host 0.10% + fusion epilogues), which is where a naive ~3-6%
prior comes from. But almost all of it is NOT identity-shaveable:

  - the int4 GEMM body (76.2% of step) + lm_head GEMM (2.2%) + draft chain (12.0%) +
    the attention roofline floor (3.3%) are BW-bound and IMMOVABLE -> 93.7% of step
    cannot be touched without quant/spec (out of the strict-lock scope).
  - the attention EXPOSED slack (6.19% of step) is occupancy-saturated (denken #332:
    96 CTAs > 80 sm_86 SMs) -> NOT removable by a backend swap. The FlashInfer-vs-
    FA2/Triton prior at batch=1 Ampere sm_86 is ~PARITY (0%, range -10%..+5%; the
    kernels converge to an HBM-BW GEMV, FlashInfer's edge is Hopper/large-batch only,
    and a backend switch can even ADD host-dispatch overhead + break CUDA-graph capture
    -> vLLM #9471). [researcher-agent: arXiv 2501.01005 / 2511.11581 / vLLM #9471]
  - the CUDA-graph / host lever is ALREADY BANKED: ONEGRAPH=1 fuses the decode step
    into one captured graph (ubel #306 capture survives, 20.158 GiB peak, 3.84 GiB
    headroom); the residual host overhead is ~0.10% of the normalized step and does
    NOT clear materiality (ubel #284).
  - the norm/residual/lm_head EPILOGUE folds are ALREADY CAPTURED in the deployed
    baseline (wirbel #285: lm_head incremental 0.0us, norms incremental 0.0us).

So the realizable identity-preserving step-shave COLLAPSES to wirbel #285's MEASURED
lossless micro-lever envelope: 15.483us (the SDPA num_stages 3->2 kernel tune, bit-
identical 0/128) = +1.2873% -> 487.73 TPS, which the merged card already proved does
NOT clear 500. The optimistic MAX (granting FlashInfer its literature-optimistic +5%
edge on the post-tune attention + full host recovery) reaches only ~+1.79% -> ~490.3.

VERDICT
-------
max_identity_step_shave_frac ~ 1.27% (measured floor, #285) .. 1.79% (optimistic max),
NOT the ~3-6% prior. Stacked on wirbel #354's custom-kernel-compliant 477:
  realizable  477 x 1.012873 = 483.1 < 500
  optimistic  477 x 1.018187 = 485.7 < 500
=> step_shave_closes_500_gap = False (decisively, not borderline). The ~3% residual
to 500 lives OFF the lossless step axis -- on the E[T]/coverage axis (#285 handoff),
which is the gated retrain, NOT an identity-safe step shave. The single highest-yield
identity-safe shave is the verify-SDPA num_stages 3->2 tune (sliding-h256 variant
captures 87.0% of the saving). The ~3-6% prior is REFUTED: it implicitly treats the
attention exposed slack as kernel-removable, but #332 (occupancy-saturated) and the
FlashInfer-parity literature both show it is a hardware floor.

HONEST SCOPE
------------
0 GPU, 0 TPS, BASELINE 481.53 UNCHANGED. CPU-analytic over MERGED banked numbers +
literature priors; NO build, NO model forward, NO served-file change, NO HF Job, NO
submission, NO launch. wirbel #354's 477 is an EXTERNAL anchor supplied by the PR body
(its branch is not merged into this advisor branch) -- used as the stacking substrate,
NOT re-derived. The optimistic attention-backend ceiling is an UPPER bound the
literature does not support (prior is parity). The roofline floor split is the #344
banked decomposition. Touching the GEMM body (quant) or draft chain (spec) is out of
the strict-lock scope by construction.

PRIMARY metric  strict_step_shave_stack_self_test_passes
KEY metrics     max_identity_step_shave_frac, tps_after_shave, step_shave_closes_500_gap,
                highest_yield_shave
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
# Imported anchors -- DO NOT re-derive. Import EXACTLY, UNCHANGED, with source.
# --------------------------------------------------------------------------- #
# Baseline / target (PR #52; twoceiling; #257/#278).
OFFICIAL_TPS = 481.53
TARGET_TPS = 500.0
STEP_NORM_US = 1218.2                       # deployed NORMALIZED single-token step (#257/#278)

# denken #344 (`gate_independent_speed_lever`, sxltbech) -- the step waterfall basis.
# WALL per-term (graphed, M=8 verify) + K=7 draft chain; STEP_NORM = bridge * wall_total.
VERIFY_BODY_US_M8 = 4474.193849563599        # 37-layer int4 body GEMMs (graphed)
VERIFY_ATTN_US_M8 = 557.9008138179779        # 37-layer attention (graphed)
VERIFY_LMHEAD_US_M8 = 131.6198444366455      # lm_head GEMM (graphed)
DRAFT_PASS_US_GRAPHED = 100.6822395324707    # one drafter pass (graphed)
K_SPEC = 7                                   # draft chain length
BRIDGE_344 = 0.2075832048263608              # STEP_NORM_US / wall_total (#257/#344)
SDPA_BW_UTIL_332 = 0.34883864849061247       # verify-attention BW utilization (#332/#344)
DOMINANT_BODY_PCT_344 = 94.3                 # int4 body = 94.3% of step HBM bytes (#344)
DOMINANT_AI_M1_344 = 4.0                     # body AI at batch=1 (M=1) << ridge 208.3 (#344)
RIDGE_AI_344 = 208.33333333333334            # A10G ridge = 125TFLOP / 600GB/s (#344)
# denken #278 (`linear_step_decomposition`) GeluAndMul-fold honesty (CUSTOM-kernel fusion).
FUSION_GEMM_EPILOGUE_HONEST_PCT_278 = 0.9113547874137429   # +0.91% (Model B), naive +4.39% refuted
FUSION_GEMM_EPILOGUE_NAIVE_PCT_278 = 4.390896003290608

# wirbel #285 (`lossless_micro_lever_envelope`, 97b57hhe) -- the MEASURED identity-safe
# step-shave envelope on the deployed step (A10G sm_86, ONEGRAPH-faithful, M=8 verify,
# bit-identical 0/128, ppl 2.3772, verify-side bridge=1.0 -> NO discount).
LOSSLESS_ENVELOPE_SAVING_US_285 = 15.482875506083142     # total greedy-safe lossless shave
LOSSLESS_ENVELOPE_TPS_GAIN_PCT_285 = 1.2873247741107985  # -> +1.2873%
LOSSLESS_ENVELOPE_TPS_285 = 487.72885498477575           # -> 487.73 TPS
LOSSLESS_ENVELOPE_NEW_STEP_US_285 = 1202.7171244939168   # 1218.2 - 15.483
LOSSLESS_SDPA_FULL_SAVING_US_285 = 15.482875506083142    # SDPA num_stages 3->2 (the ONLY incr lever)
LOSSLESS_SDPA_SLIDING_ONLY_SAVING_US_285 = 13.475842475891088
LOSSLESS_SDPA_SLIDING_ONLY_CAPTURES_PCT_285 = 87.03707829076389
LOSSLESS_LMHEAD_INCREMENTAL_US_285 = 0.0                 # already fused (FUSED_SPARSE_ARGMAX)
LOSSLESS_NORMS_INCREMENTAL_US_285 = 0.0                  # already captured (ONEGRAPH + fused add+rmsnorm)
LOSSLESS_ENVELOPE_CLEARS_500_285 = False
LOSSLESS_RESIDUAL_GAP_TO_500_US_285 = 29.517432493916886
LOSSLESS_STEP_NEEDED_FOR_500_US_285 = 1173.199692
LOSSLESS_SDPA_GLOBAL_SPEEDUP_285 = 1.0194039514193418
LOSSLESS_SDPA_SLIDING_SPEEDUP_285 = 1.0884567272425576

# ubel #284 (`decode_host_overhead`) -- the non-model per-step host overhead, MEASURED.
HOST_OVERHEAD_FRAC_WALL_284 = 0.0049893975302482225      # 0.499% of the WALL cycle
HOST_RECOVERABLE_TPS_284 = 0.4990282123962925            # +0.499 TPS (normalized basis, over-credit-discounted)
HOST_CLEARS_MATERIALITY_284 = False                      # < 2% materiality gate; largely irreducible
HOST_ONEGRAPH_BANKED_284 = True                          # ONEGRAPH=1 already fuses decode into one CUDA graph

# ubel #306 (`eagle3_capture_peak`) -- CUDA-graph capture feasibility (supports host lever banked).
CAPTURE_PEAK_GB_306 = 20.158                             # runtime peak fits <=24 with 3.84 GiB headroom
CAPTURE_SURVIVES_306 = True                              # the deployed M=8 graph is captured & fits VRAM

# researcher-agent (FlashInfer vs Triton/FA2 attention, batch=1 Ampere sm_86) -- literature prior.
#   arXiv 2501.01005 (FlashInfer; edge is Hopper/large-batch/FP8, not Ampere batch=1)
#   arXiv 2511.11581 (vLLM Triton attn already 98.6-105.9% of FA3 -> little room over Triton)
#   vLLM #9471 (FlashInfer ~16% SLOWER at batch=64 H100 FP8 -> no universal dominance)
ATTN_BACKEND_SWAP_PRIOR_FRAC = 0.0           # PARITY at batch=1 Ampere sm_86 (recommended prior)
ATTN_BACKEND_SWAP_OPTIMISTIC_FRAC = 0.05     # optimistic CEILING (+5% on the attention kernel)
ATTN_BACKEND_SWAP_RISK = "negative_possible"  # backend switch can add host-dispatch + break graph capture

# wirbel #354 (custom-kernel-compliant ceiling) -- EXTERNAL anchor from the PR body.
# Its branch is NOT merged into this advisor branch; used as the stacking substrate only,
# NOT inspected or re-derived (launch-isolation honored).
CUSTOM_KERNEL_COMPLIANT_CEILING_TPS_354 = 477.0
CUSTOM_KERNEL_CEILING_SOURCE_354 = "PR #359 body (external, unmerged; advisor-supplied ~477)"

# denken #332 -- the attention exposed slack is occupancy-saturated (hardware floor).
SDPA_OCCUPANCY_CTAS_332 = 96                 # 96 CTAs > 80 sm_86 SMs -> saturated, NOT removable
SM_COUNT_A10G_332 = 80

TOL_ROUNDTRIP = 1.0e-6
TOL_TPS = 1.0e-3
TOL_US = 1.0e-6


# --------------------------------------------------------------------------- #
# TPS <-> step conversions (deployed-substrate basis; PR-specified formula)
# --------------------------------------------------------------------------- #
def tps_after_shave(step_after_us: float) -> float:
    """PR formula: tps_after_shave = 481.53 * 1218.2 / step_after_shave_us."""
    return OFFICIAL_TPS * STEP_NORM_US / step_after_us


def step_for_tps(tps: float) -> float:
    return OFFICIAL_TPS * STEP_NORM_US / tps


# --------------------------------------------------------------------------- #
# Step 1: decompose the 1218.2us step into BW-bound (immovable) vs shaveable
# --------------------------------------------------------------------------- #
def step1_decompose() -> dict[str, Any]:
    bridge = BRIDGE_344
    # Normalized per-term shares (each WALL term x bridge; sums to 1218.2us, #344 basis).
    body_us = VERIFY_BODY_US_M8 * bridge          # int4 GEMM body (gate_up/down/qkv/o)
    attn_us = VERIFY_ATTN_US_M8 * bridge          # SDPA attention
    lmhead_us = VERIFY_LMHEAD_US_M8 * bridge      # lm_head GEMM
    draft_us = DRAFT_PASS_US_GRAPHED * K_SPEC * bridge  # K=7 drafter chain
    waterfall_sum = body_us + attn_us + lmhead_us + draft_us
    waterfall_resid = abs(waterfall_sum - STEP_NORM_US)

    # Attention term split: roofline floor (immovable, BW-bound) vs exposed slack.
    attn_floor_us = SDPA_BW_UTIL_332 * attn_us            # 34.9% -> the immovable KV/act read
    attn_exposed_slack_us = (1.0 - SDPA_BW_UTIL_332) * attn_us  # 65.1% -> "looks shaveable"

    # --- BW-bound GEMM time: IMMOVABLE (identity + physics, strict-lock out-of-scope) --- #
    #   int4 body GEMMs (94.3% of HBM bytes, AI 4.0 << ridge 208.3); lm_head GEMM
    #   (bf16, near-roofline 83.4% BW); draft chain (drafter weight reads -- touching it
    #   = speculation); attention roofline floor (KV + activation read).
    bw_bound_gemm_us = body_us + lmhead_us + draft_us          # the pure-GEMM immovable terms
    bw_bound_total_immovable_us = bw_bound_gemm_us + attn_floor_us  # + attention floor
    immovable_pct = 100.0 * bw_bound_total_immovable_us / STEP_NORM_US

    # --- shaveable overhead (the candidate identity-safe surface) --- #
    #   attention exposed slack + host/launch gaps + fusion-epilogue round-trips.
    #   (NOTE: "shaveable" here is the APPARENT surface; Step 2 prices how little of it
    #    is actually identity-removable.)
    host_overhead_norm_us = HOST_RECOVERABLE_TPS_284 / OFFICIAL_TPS * STEP_NORM_US  # ~1.26us normalized
    shaveable_surface_us = attn_exposed_slack_us + host_overhead_norm_us
    shaveable_surface_pct = 100.0 * shaveable_surface_us / STEP_NORM_US

    rows = [
        {"term": "gate_up_proj+down_proj+qkv+o (int4 body)", "norm_us": body_us,
         "pct": 100.0 * body_us / STEP_NORM_US, "class": "bw_bound_gemm", "shaveable_us": 0.0,
         "why_immovable": "int4 weight read 94.3% of HBM, AI 4.0 << ridge 208.3; quant=out-of-scope"},
        {"term": "lm_head (bf16 GEMM)", "norm_us": lmhead_us,
         "pct": 100.0 * lmhead_us / STEP_NORM_US, "class": "bw_bound_gemm", "shaveable_us": 0.0,
         "why_immovable": "bf16 weight read near-roofline 83.4% BW; fused epilogue already banked (#285)"},
        {"term": "draft_chain (K=7 drafter)", "norm_us": draft_us,
         "pct": 100.0 * draft_us / STEP_NORM_US, "class": "bw_bound_gemm", "shaveable_us": 0.0,
         "why_immovable": "drafter weight read; touching it = speculation (strict-lock out-of-scope)"},
        {"term": "attention roofline floor", "norm_us": attn_floor_us,
         "pct": 100.0 * attn_floor_us / STEP_NORM_US, "class": "bw_bound_floor", "shaveable_us": 0.0,
         "why_immovable": "KV + activation HBM read (34.9% BW); physical floor"},
        {"term": "attention exposed slack", "norm_us": attn_exposed_slack_us,
         "pct": 100.0 * attn_exposed_slack_us / STEP_NORM_US, "class": "shaveable_surface",
         "shaveable_us": attn_exposed_slack_us,
         "why_immovable": "65.1% above roofline -- APPARENT shave; Step 2 prices the realizable part"},
        {"term": "host / launch gaps", "norm_us": host_overhead_norm_us,
         "pct": 100.0 * host_overhead_norm_us / STEP_NORM_US, "class": "shaveable_surface",
         "shaveable_us": host_overhead_norm_us,
         "why_immovable": "ONEGRAPH already fuses decode -> ~0.10% residual, irreducible (#284)"},
    ]

    gemm_terms_shave_sum = sum(r["shaveable_us"] for r in rows if r["class"] in ("bw_bound_gemm", "bw_bound_floor"))

    return {
        "bridge": bridge,
        "waterfall_sum_us": waterfall_sum,
        "waterfall_resid_us": waterfall_resid,
        "body_us": body_us, "attn_us": attn_us, "lmhead_us": lmhead_us, "draft_us": draft_us,
        "attn_floor_us": attn_floor_us, "attn_exposed_slack_us": attn_exposed_slack_us,
        "attn_exposed_slack_pct": 100.0 * attn_exposed_slack_us / STEP_NORM_US,
        "host_overhead_norm_us": host_overhead_norm_us,
        "bw_bound_gemm_us": bw_bound_gemm_us,
        "bw_bound_total_immovable_us": bw_bound_total_immovable_us,
        "immovable_pct": immovable_pct,
        "shaveable_surface_us": shaveable_surface_us,
        "shaveable_surface_pct": shaveable_surface_pct,
        "gemm_terms_shave_sum_us": gemm_terms_shave_sum,
        "rows": rows,
        "note": ("step decomposes into 93.7% BW-bound immovable (int4 body + lm_head GEMM + "
                 "draft chain + attention roofline floor) and ~6.3% APPARENT shaveable surface "
                 "(attention exposed slack 6.19% + host 0.10%). Step 2 prices how little of the "
                 "surface is IDENTITY-removable."),
    }


# --------------------------------------------------------------------------- #
# Step 2: price each identity-safe lever on the shaveable surface (merged priors)
# --------------------------------------------------------------------------- #
def step2_levers(s1: dict[str, Any]) -> dict[str, Any]:
    attn_us = s1["attn_us"]
    attn_exposed_slack_us = s1["attn_exposed_slack_us"]

    # (1) attention-kernel tune: SDPA num_stages 3->2 (#285, MEASURED, bit-identical 0/128).
    #     This is the ONE incremental free lever; it recovers part of the exposed slack.
    sdpa_tune_us = LOSSLESS_SDPA_FULL_SAVING_US_285
    sdpa_tune_frac_of_slack = sdpa_tune_us / attn_exposed_slack_us
    # post-tune attention term (what a backend swap would further attack).
    attn_post_tune_us = attn_us - sdpa_tune_us

    # (2) attention-BACKEND swap (FA2-SW <-> FlashInfer): literature prior = parity at
    #     batch=1 Ampere sm_86. Realizable = 0; optimistic CEILING = +5% on post-tune attn.
    attn_backend_realizable_us = ATTN_BACKEND_SWAP_PRIOR_FRAC * attn_post_tune_us       # 0.0
    attn_backend_optimistic_us = ATTN_BACKEND_SWAP_OPTIMISTIC_FRAC * attn_post_tune_us  # ceiling
    # residual exposed slack the backend swap would NEED to recover but cannot (#332).
    attn_residual_saturated_us = attn_exposed_slack_us - sdpa_tune_us

    # (3) kernel fusion (norm / residual / lm_head epilogue): ALREADY BANKED (#285).
    fusion_epilogue_us = LOSSLESS_LMHEAD_INCREMENTAL_US_285 + LOSSLESS_NORMS_INCREMENTAL_US_285  # 0.0

    # (4) CUDA-graph / host launch: ALREADY BANKED (ONEGRAPH, #284/#306). Optimistic =
    #     full host recovery (does not clear materiality, largely irreducible).
    host_realizable_us = 0.0
    host_optimistic_us = s1["host_overhead_norm_us"]

    levers = {
        "attn_kernel_tune_sdpa_numstages": {
            "family": "attention-backend-selection (kernel tune, same backend)",
            "realizable_us": sdpa_tune_us, "optimistic_us": sdpa_tune_us,
            "identity": "bit-identical 0/128 maxdiff 0.0 (wirbel #285)",
            "source": "wirbel #285 lossless_micro_lever_envelope (MEASURED)",
            "note": (f"SDPA num_stages 3->2; recovers {100*sdpa_tune_frac_of_slack:.1f}% of the "
                     f"attention exposed slack; sliding-h256 variant captures "
                     f"{LOSSLESS_SDPA_SLIDING_ONLY_CAPTURES_PCT_285:.1f}%"),
        },
        "attn_backend_swap_flashinfer": {
            "family": "attention-backend-selection (backend swap)",
            "realizable_us": attn_backend_realizable_us, "optimistic_us": attn_backend_optimistic_us,
            "identity": "must be greedy-identical by construction (no quant/spec)",
            "source": "researcher-agent: arXiv 2501.01005 / 2511.11581 / vLLM #9471",
            "note": (f"batch=1 Ampere sm_86 prior = PARITY (0%); optimistic ceiling +5% on the "
                     f"post-tune attention; risk={ATTN_BACKEND_SWAP_RISK}. The residual "
                     f"{attn_residual_saturated_us:.1f}us exposed slack is occupancy-saturated "
                     f"(#332: {SDPA_OCCUPANCY_CTAS_332} CTAs > {SM_COUNT_A10G_332} SMs) -> a "
                     f"hardware floor, not backend-recoverable"),
        },
        "kernel_fusion_norm_residual_lmhead_epilogue": {
            "family": "kernel-fusion (norm/residual/activation epilogue)",
            "realizable_us": fusion_epilogue_us, "optimistic_us": fusion_epilogue_us,
            "identity": "bit-identical (wirbel #285)",
            "source": "wirbel #285 (lm_head incr 0.0, norms incr 0.0 -- already_captured)",
            "note": ("already banked in the deployed baseline (FUSED_SPARSE_ARGMAX + ONEGRAPH + "
                     "vLLM fused add+rmsnorm). NB the GEMM+activation epilogue (#278 +0.91%) is a "
                     "CUSTOM kernel -> it lives in wirbel #354's custom-kernel ceiling, NOT in this "
                     "no-custom-kernel identity stack"),
        },
        "cuda_graph_host_launch": {
            "family": "cuda-graph-capture (launch/host-gap removal)",
            "realizable_us": host_realizable_us, "optimistic_us": host_optimistic_us,
            "identity": "host-side, emission-neutral",
            "source": "ubel #284 (host 0.499% wall, +0.499 TPS, clears_materiality=False) + #306",
            "note": ("ONEGRAPH=1 already fuses decode into one captured graph (#306 capture "
                     "survives, 20.158 GiB, 3.84 GiB headroom); residual ~0.10% of normalized "
                     "step, largely irreducible"),
        },
    }

    # naive ARITHMETIC sum of optimistic lever fractions (for the composed<sum self-test).
    naive_sum_us = sum(lv["optimistic_us"] for lv in levers.values())
    naive_sum_frac = naive_sum_us / STEP_NORM_US

    return {
        "levers": levers,
        "sdpa_tune_us": sdpa_tune_us,
        "sdpa_tune_frac_of_slack": sdpa_tune_frac_of_slack,
        "attn_post_tune_us": attn_post_tune_us,
        "attn_backend_realizable_us": attn_backend_realizable_us,
        "attn_backend_optimistic_us": attn_backend_optimistic_us,
        "attn_residual_saturated_us": attn_residual_saturated_us,
        "fusion_epilogue_us": fusion_epilogue_us,
        "host_realizable_us": host_realizable_us,
        "host_optimistic_us": host_optimistic_us,
        "naive_sum_us": naive_sum_us,
        "naive_sum_frac": naive_sum_frac,
    }


def _compose_multiplicative(step_us: float, saving_us_list: list[float]) -> float:
    """Compose overlapping step-savings multiplicatively (NOT naive sum): each lever
    shaves a fraction f_i = s_i/step of the step, so step_after = step * prod(1 - f_i).
    For overlapping levers this is strictly less aggressive than the arithmetic sum
    step - sum(s_i), which is exactly the composed<sum property the self-test asserts."""
    prod = 1.0
    for s in saving_us_list:
        prod *= (1.0 - s / step_us)
    return step_us * prod


# --------------------------------------------------------------------------- #
# Step 3: compose the stacked identity-preserving shave (compose, NOT sum)
# --------------------------------------------------------------------------- #
def step3_compose(s2: dict[str, Any]) -> dict[str, Any]:
    # Realizable stack: SDPA tune (#285) + fusion (0) + host (0) + backend (0).
    realizable_savings = [s2["sdpa_tune_us"], s2["fusion_epilogue_us"],
                          s2["host_realizable_us"], s2["attn_backend_realizable_us"]]
    step_after_realizable = _compose_multiplicative(STEP_NORM_US, realizable_savings)
    tps_realizable = tps_after_shave(step_after_realizable)
    frac_realizable = (STEP_NORM_US - step_after_realizable) / STEP_NORM_US

    # Optimistic MAX stack: SDPA tune + fusion + host(optimistic) + backend(+5% ceiling).
    optimistic_savings = [s2["sdpa_tune_us"], s2["fusion_epilogue_us"],
                          s2["host_optimistic_us"], s2["attn_backend_optimistic_us"]]
    step_after_optimistic = _compose_multiplicative(STEP_NORM_US, optimistic_savings)
    tps_optimistic = tps_after_shave(step_after_optimistic)
    frac_optimistic = (STEP_NORM_US - step_after_optimistic) / STEP_NORM_US

    # composed (multiplicative) total vs the naive arithmetic sum -- composed < sum.
    composed_optimistic_frac = frac_optimistic
    composed_lt_naive_sum = bool(composed_optimistic_frac < s2["naive_sum_frac"] + 1e-12)

    # The NAIVE THEORETICAL ceiling that produces the ~3-6% prior: if the FULL attention
    # exposed slack were kernel-removable (+ host). REFUTED by #332 + FlashInfer parity.
    # (attn_exposed_slack is recomputed here from the same constants for transparency.)
    return {
        "realizable": {
            "savings_us": realizable_savings, "step_after_us": step_after_realizable,
            "tps_after": tps_realizable, "shave_frac": frac_realizable,
            "tps_gain_pct": 100.0 * (tps_realizable / OFFICIAL_TPS - 1.0),
        },
        "optimistic": {
            "savings_us": optimistic_savings, "step_after_us": step_after_optimistic,
            "tps_after": tps_optimistic, "shave_frac": frac_optimistic,
            "tps_gain_pct": 100.0 * (tps_optimistic / OFFICIAL_TPS - 1.0),
        },
        "naive_sum_frac": s2["naive_sum_frac"],
        "composed_lt_naive_sum": composed_lt_naive_sum,
        # cross-check: the realizable stack reproduces wirbel #285's measured envelope.
        "matches_285_envelope": bool(
            abs(step_after_realizable - LOSSLESS_ENVELOPE_NEW_STEP_US_285) < 1e-6
            and abs(tps_realizable - LOSSLESS_ENVELOPE_TPS_285) < 1e-3),
    }


# --------------------------------------------------------------------------- #
# Step 4: stack on wirbel #354's custom-kernel-compliant ceiling -> close 500?
# --------------------------------------------------------------------------- #
def step4_verdict(s3: dict[str, Any]) -> dict[str, Any]:
    ceiling = CUSTOM_KERNEL_COMPLIANT_CEILING_TPS_354
    gain_realizable = s3["realizable"]["tps_after"] / OFFICIAL_TPS          # 1.012873
    gain_optimistic = s3["optimistic"]["tps_after"] / OFFICIAL_TPS          # 1.018187
    stacked_realizable = ceiling * gain_realizable
    stacked_optimistic = ceiling * gain_optimistic
    # the shave fraction the gap from 477 to 500 would REQUIRE.
    required_frac_from_354 = TARGET_TPS / ceiling - 1.0                     # +4.82%
    # advisor naive cross-check: the +5%-step prior the PR flagged "borderline".
    advisor_naive_5pct_stacked = ceiling * 1.05

    closes = bool(stacked_optimistic >= TARGET_TPS)
    return {
        "custom_kernel_compliant_ceiling_354": ceiling,
        "ceiling_source": CUSTOM_KERNEL_CEILING_SOURCE_354,
        "stacked_realizable_tps": stacked_realizable,
        "stacked_optimistic_tps": stacked_optimistic,
        "required_shave_frac_from_354_to_500": required_frac_from_354,
        "max_identity_step_shave_frac_realizable": s3["realizable"]["shave_frac"],
        "max_identity_step_shave_frac_optimistic": s3["optimistic"]["shave_frac"],
        "advisor_naive_5pct_stacked_tps": advisor_naive_5pct_stacked,
        "advisor_naive_5pct_borderline": bool(abs(advisor_naive_5pct_stacked - TARGET_TPS) < 2.0),
        "step_shave_closes_500_gap": closes,
        "residual_to_500_tps": TARGET_TPS - stacked_optimistic,
        "verdict_note": (
            "the realizable identity-preserving shave (+1.29%, #285 measured) stacks 477->483.1; "
            "the optimistic max (+1.79%, grants FlashInfer its literature-optimistic +5% edge + "
            "full host recovery) stacks 477->485.7. Both < 500. Closing the gap needs +4.82% from "
            "477, but the lossless step axis tops out at ~+1.8%; the residual ~3% lives OFF this "
            "axis (E[T]/coverage, #285 handoff). The advisor's ~3-6% prior is REFUTED -- it treats "
            "the attention exposed slack (6.19% of step) as kernel-removable, but #332 "
            "(occupancy-saturated) + the FlashInfer-parity literature show it is a hardware floor."),
    }


# --------------------------------------------------------------------------- #
# Synthesis + verdict
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    s1 = step1_decompose()
    s2 = step2_levers(s1)
    s3 = step3_compose(s2)
    s4 = step4_verdict(s3)

    # KEY DELIVERABLES.
    max_identity_step_shave_frac = s3["optimistic"]["shave_frac"]    # the MAX (optimistic ceiling)
    tps_after = s3["optimistic"]["tps_after"]                        # on the deployed substrate
    step_after = s3["optimistic"]["step_after_us"]
    closes_500 = s4["step_shave_closes_500_gap"]
    highest_yield = "verify_sdpa_num_stages_3to2_sliding_h256"

    handoff = (
        "PR #359 hand-off (the identity-preserving step-shave closure for the strict >500 lane): "
        "the deployed batch=1 step (1218.2us) is 93.7% BW-bound IMMOVABLE (int4 GEMM body 76.2% + "
        "lm_head GEMM 2.2% + draft chain 12.0% + attention roofline floor 3.3%; touching them = "
        "quant/spec, out of the strict-lock scope). The ~6.3% APPARENT shaveable surface is almost "
        "entirely a HARDWARE FLOOR: the attention exposed slack (6.19% of step) is occupancy-"
        "saturated (#332: 96 CTAs > 80 sm_86 SMs) and the FlashInfer-vs-FA2/Triton prior at batch=1 "
        "Ampere is PARITY (0%, range -10..+5%; FlashInfer's edge is Hopper/large-batch only; a "
        "backend switch can ADD host-dispatch + break CUDA-graph capture, vLLM #9471). The host/"
        "CUDA-graph lever is already banked (ONEGRAPH, ubel #284/#306; ~0.10% irreducible) and the "
        "norm/residual/lm_head epilogue folds are already captured (#285 incr 0.0). So the "
        "realizable identity-safe step-shave COLLAPSES to wirbel #285's MEASURED lossless envelope "
        "(SDPA num_stages 3->2): 15.483us = +1.2873% -> 487.73, which already does NOT clear 500. "
        f"The optimistic MAX reaches only +{100*max_identity_step_shave_frac:.2f}% -> {tps_after:.2f}. "
        f"Stacked on wirbel #354's custom-kernel-compliant 477: realizable 483.1, optimistic 485.7 -- "
        f"step_shave_closes_500_gap = {closes_500}. The single highest-yield identity-safe shave is "
        "the verify-SDPA num_stages 3->2 tune (sliding-h256 captures 87.0%). The ~3% residual to 500 "
        "is structurally OFF the lossless step axis -- it lives on the E[T]/coverage axis (the gated "
        "retrain), confirming #344's closure that #124 (not a step lever) is the sole >500 gate. "
        "0 GPU, 0 TPS, BASELINE 481.53 unchanged; analytic over MERGED priors; NOT a launch."
    )

    verdict = (
        f"IDENTITY-PRESERVING STEP-SHAVE DOES NOT CLOSE THE GAP TO 500. "
        f"max_identity_step_shave_frac = {100*max_identity_step_shave_frac:.2f}% (optimistic max) / "
        f"{100*s3['realizable']['shave_frac']:.2f}% (measured floor, #285) -- NOT the ~3-6% prior. "
        f"tps_after_shave = {tps_after:.2f} (deployed substrate). Stacked on wirbel #354's 477: "
        f"realizable {s4['stacked_realizable_tps']:.1f} / optimistic {s4['stacked_optimistic_tps']:.1f} "
        f"< 500 => step_shave_closes_500_gap = {closes_500}. The 93.7% BW-bound GEMM/floor time is "
        f"immovable (identity + physics); the attention exposed slack is occupancy-saturated (#332) "
        f"and backend-inert (FlashInfer parity at batch=1 Ampere sm_86); host + epilogue fusion are "
        f"already banked (ONEGRAPH/#285). Highest-yield identity-safe shave = SDPA num_stages 3->2 "
        f"(sliding-h256, 87.0% of the saving). The residual to 500 lives off the lossless step axis. "
        f"BASELINE 481.53 untouched; CPU-analytic; 0 TPS; NOT a launch."
    )

    return {
        "official_tps": OFFICIAL_TPS, "target_tps": TARGET_TPS, "step_norm_us": STEP_NORM_US,
        "step1_decompose": s1, "step2_levers": s2, "step3_compose": s3, "step4_verdict": s4,
        # KEY DELIVERABLES (PR #359).
        "max_identity_step_shave_frac": max_identity_step_shave_frac,
        "max_identity_step_shave_frac_measured_floor": s3["realizable"]["shave_frac"],
        "step_after_shave_us": step_after,
        "tps_after_shave": tps_after,
        "step_shave_closes_500_gap": closes_500,
        "highest_yield_shave": highest_yield,
        "verdict": verdict, "handoff": handoff,
    }


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY: strict_step_shave_stack_self_test_passes)
# --------------------------------------------------------------------------- #
def self_test(syn: dict[str, Any]) -> dict[str, Any]:
    s1, s2, s3, s4 = (syn["step1_decompose"], syn["step2_levers"],
                      syn["step3_compose"], syn["step4_verdict"])
    c: dict[str, bool] = {}

    # (a) waterfall sums to 1218.2us (the #344 decomposition basis).
    c["a_waterfall_sums_to_1218p2"] = bool(s1["waterfall_resid_us"] <= 1e-3)
    c["a_immovable_is_dominant"] = bool(s1["immovable_pct"] > 90.0)

    # (b) BW-bound GEMM time is NOT shaved (identity + physics): every GEMM / floor term
    #     has shaveable_us == 0, and the body is BW-bound (AI << ridge).
    c["b_gemm_terms_not_shaved"] = bool(s1["gemm_terms_shave_sum_us"] == 0.0)
    c["b_body_is_bw_bound"] = bool(DOMINANT_AI_M1_344 < RIDGE_AI_344 and DOMINANT_BODY_PCT_344 > 90.0)
    c["b_lmhead_fusion_already_banked"] = bool(LOSSLESS_LMHEAD_INCREMENTAL_US_285 == 0.0
                                               and LOSSLESS_NORMS_INCREMENTAL_US_285 == 0.0)

    # (c) the realizable stack reproduces wirbel #285's MEASURED envelope (15.483us -> 487.73).
    c["c_realizable_matches_285"] = bool(s3["matches_285_envelope"])
    c["c_realizable_tps_roundtrips_285"] = bool(
        abs(s3["realizable"]["tps_after"] - LOSSLESS_ENVELOPE_TPS_285) < 1e-3)
    c["c_285_envelope_does_not_clear_500"] = bool(LOSSLESS_ENVELOPE_CLEARS_500_285 is False
                                                  and LOSSLESS_ENVELOPE_TPS_285 < TARGET_TPS)

    # (d) composed shave < naive arithmetic sum (overlapping levers, not additive).
    c["d_composed_lt_naive_sum"] = bool(s3["composed_lt_naive_sum"])
    c["d_max_frac_below_prior_band"] = bool(syn["max_identity_step_shave_frac"] < 0.03)  # << 3-6% prior

    # (e) round-trip 481.53 at ZERO shave (PR formula sanity).
    c["e_zero_shave_roundtrips_481p53"] = bool(abs(tps_after_shave(STEP_NORM_US) - OFFICIAL_TPS) <= TOL_TPS)
    c["e_tps_after_formula_consistent"] = bool(
        abs(syn["tps_after_shave"] - OFFICIAL_TPS * STEP_NORM_US / syn["step_after_shave_us"]) <= TOL_TPS)

    # (f) attention-backend prior is parity (the load-bearing literature input).
    c["f_attn_backend_prior_is_parity"] = bool(ATTN_BACKEND_SWAP_PRIOR_FRAC == 0.0)
    c["f_attn_slack_positive"] = bool(s1["attn_exposed_slack_us"] > 0.0)
    c["f_attn_residual_saturated_positive"] = bool(s2["attn_residual_saturated_us"] > 0.0)

    # (g) verdict logic: closes_500 == (stacked_optimistic >= 500), and is False here.
    closes = s4["step_shave_closes_500_gap"]
    c["g_verdict_logic_consistent"] = bool(closes == (s4["stacked_optimistic_tps"] >= TARGET_TPS))
    c["g_verdict_is_false"] = bool(closes is False)
    c["g_stacked_below_500"] = bool(s4["stacked_realizable_tps"] < TARGET_TPS
                                    and s4["stacked_optimistic_tps"] < TARGET_TPS)
    c["g_required_frac_exceeds_max"] = bool(
        s4["required_shave_frac_from_354_to_500"] > syn["max_identity_step_shave_frac"])

    # (h) constants imported exact (no silent re-derivation).
    c["h_constants_exact"] = bool(
        OFFICIAL_TPS == 481.53 and STEP_NORM_US == 1218.2
        and abs(LOSSLESS_ENVELOPE_SAVING_US_285 - 15.482875506083142) < 1e-9
        and CUSTOM_KERNEL_COMPLIANT_CEILING_TPS_354 == 477.0)

    passed = all(c.values())
    return {"checks": c, "strict_step_shave_stack_self_test_passes": passed}


# --------------------------------------------------------------------------- #
# NaN-clean + report + wandb + main
# --------------------------------------------------------------------------- #
def _assert_nan_clean(obj: Any, path: str = "") -> list[str]:
    bad: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            bad += _assert_nan_clean(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            bad += _assert_nan_clean(v, f"{path}[{i}]")
    elif isinstance(obj, float) and not math.isfinite(obj):
        bad.append(path)
    return bad


def _print_report(syn: dict[str, Any], st: dict[str, Any]) -> None:
    s1, s3, s4 = syn["step1_decompose"], syn["step3_compose"], syn["step4_verdict"]
    print("=" * 78, flush=True)
    print("PR #359  Strict identity-preserving step-shave stack -- close the gap to 500?", flush=True)
    print("=" * 78, flush=True)
    print(f"  step decomposition (sums to {s1['waterfall_sum_us']:.4f}us, "
          f"resid {s1['waterfall_resid_us']:.2e}):", flush=True)
    for r in s1["rows"]:
        tag = "IMMOVABLE" if r["class"].startswith("bw_bound") else "shaveable?"
        print(f"    {r['term']:<42} {r['norm_us']:7.2f}us ({r['pct']:5.2f}%) [{tag}]", flush=True)
    print(f"  immovable BW-bound = {s1['immovable_pct']:.1f}% of step; "
          f"apparent shaveable surface = {s1['shaveable_surface_pct']:.2f}%", flush=True)
    print("  identity-safe levers (priced on merged anchors):", flush=True)
    print(f"    SDPA num_stages 3->2 (#285 MEASURED) : {syn['step2_levers']['sdpa_tune_us']:6.2f}us "
          f"(realizable)", flush=True)
    print(f"    attn-backend swap (FlashInfer)       : "
          f"{syn['step2_levers']['attn_backend_realizable_us']:6.2f}us realizable / "
          f"{syn['step2_levers']['attn_backend_optimistic_us']:6.2f}us optimistic (parity prior)", flush=True)
    print(f"    norm/residual/lm_head epilogue (#285): {syn['step2_levers']['fusion_epilogue_us']:6.2f}us "
          f"(already banked)", flush=True)
    print(f"    CUDA-graph/host (#284/#306)          : {syn['step2_levers']['host_realizable_us']:6.2f}us "
          f"realizable / {syn['step2_levers']['host_optimistic_us']:6.2f}us optimistic (ONEGRAPH banked)",
          flush=True)
    print(f"  realizable stack: step {s3['realizable']['step_after_us']:.2f}us -> "
          f"{s3['realizable']['tps_after']:.2f} TPS (+{s3['realizable']['tps_gain_pct']:.3f}%) "
          f"[matches #285 = {s3['matches_285_envelope']}]", flush=True)
    print(f"  optimistic MAX : step {s3['optimistic']['step_after_us']:.2f}us -> "
          f"{s3['optimistic']['tps_after']:.2f} TPS (+{s3['optimistic']['tps_gain_pct']:.3f}%)", flush=True)
    print(f"  composed < naive sum: {s3['composed_lt_naive_sum']} "
          f"(composed {100*s3['optimistic']['shave_frac']:.3f}% < sum {100*s3['naive_sum_frac']:.3f}%)",
          flush=True)
    print(f"  STACK on #354 ceiling {s4['custom_kernel_compliant_ceiling_354']:.1f}: "
          f"realizable {s4['stacked_realizable_tps']:.1f} / optimistic {s4['stacked_optimistic_tps']:.1f} "
          f"(need +{100*s4['required_shave_frac_from_354_to_500']:.2f}% for 500)", flush=True)
    print(f"  KEY  max_identity_step_shave_frac = {100*syn['max_identity_step_shave_frac']:.3f}% "
          f"(measured floor {100*syn['max_identity_step_shave_frac_measured_floor']:.3f}%)", flush=True)
    print(f"  KEY  tps_after_shave = {syn['tps_after_shave']:.2f}  "
          f"step_after_shave_us = {syn['step_after_shave_us']:.2f}", flush=True)
    print(f"  KEY  step_shave_closes_500_gap = {syn['step_shave_closes_500_gap']}", flush=True)
    print(f"  KEY  highest_yield_shave = {syn['highest_yield_shave']}", flush=True)
    print(f"  SELF-TEST  strict_step_shave_stack_self_test_passes = "
          f"{st['strict_step_shave_stack_self_test_passes']}", flush=True)
    if not st["strict_step_shave_stack_self_test_passes"]:
        for k, v in st["checks"].items():
            if not v:
                print(f"    FAIL: {k}", flush=True)
    print("=" * 78, flush=True)


def _maybe_log_wandb(args, payload: dict) -> str | None:
    if not getattr(args, "wandb_name", None):
        return None
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[step-shave] wandb logging unavailable: {exc}", flush=True)
        return None

    syn, st = payload["synthesis"], payload["self_test"]
    s1, s3, s4 = syn["step1_decompose"], syn["step3_compose"], syn["step4_verdict"]
    run = init_wandb_run(
        job_type="strict-step-shave-stack",
        agent="kanna",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["strict-step-shave-stack", "strict-frontier", "identity-preserving",
              "step-shave", "cuda-graph", "kernel-fusion", "attention-backend",
              "flashinfer-parity", "roofline", "validity", "zero-tps"],
        config={
            "pr": 359, "analysis_only": True,
            "official_tps": OFFICIAL_TPS, "target_tps": TARGET_TPS, "step_norm_us": STEP_NORM_US,
            "custom_kernel_compliant_ceiling_354": CUSTOM_KERNEL_COMPLIANT_CEILING_TPS_354,
            "attn_backend_swap_prior_frac": ATTN_BACKEND_SWAP_PRIOR_FRAC,
            "attn_backend_swap_optimistic_frac": ATTN_BACKEND_SWAP_OPTIMISTIC_FRAC,
            "lossless_envelope_saving_us_285": LOSSLESS_ENVELOPE_SAVING_US_285,
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[step-shave] wandb: no run (no WANDB_API_KEY/mode) -- skipping", flush=True)
        return None

    summary: dict[str, Any] = {
        # PRIMARY + KEY deliverables.
        "strict_step_shave_stack_self_test_passes": int(bool(
            st["strict_step_shave_stack_self_test_passes"])),
        "max_identity_step_shave_frac": syn["max_identity_step_shave_frac"],
        "max_identity_step_shave_frac_measured_floor": syn["max_identity_step_shave_frac_measured_floor"],
        "tps_after_shave": syn["tps_after_shave"],
        "step_after_shave_us": syn["step_after_shave_us"],
        "step_shave_closes_500_gap": int(bool(syn["step_shave_closes_500_gap"])),
        "highest_yield_shave": syn["highest_yield_shave"],
        # step1 decomposition.
        "immovable_pct": s1["immovable_pct"],
        "bw_bound_total_immovable_us": s1["bw_bound_total_immovable_us"],
        "shaveable_surface_pct": s1["shaveable_surface_pct"],
        "attn_exposed_slack_pct": s1["attn_exposed_slack_pct"],
        "waterfall_resid_us": s1["waterfall_resid_us"],
        # step2/3 levers + composition.
        "sdpa_tune_us": syn["step2_levers"]["sdpa_tune_us"],
        "sdpa_tune_frac_of_slack": syn["step2_levers"]["sdpa_tune_frac_of_slack"],
        "attn_backend_optimistic_us": syn["step2_levers"]["attn_backend_optimistic_us"],
        "attn_residual_saturated_us": syn["step2_levers"]["attn_residual_saturated_us"],
        "naive_sum_frac": s3["naive_sum_frac"],
        "composed_lt_naive_sum": int(bool(s3["composed_lt_naive_sum"])),
        "matches_285_envelope": int(bool(s3["matches_285_envelope"])),
        "realizable_tps_after": s3["realizable"]["tps_after"],
        "realizable_shave_frac": s3["realizable"]["shave_frac"],
        "optimistic_tps_after": s3["optimistic"]["tps_after"],
        "optimistic_shave_frac": s3["optimistic"]["shave_frac"],
        # step4 verdict / stacking.
        "custom_kernel_compliant_ceiling_354": s4["custom_kernel_compliant_ceiling_354"],
        "stacked_realizable_tps": s4["stacked_realizable_tps"],
        "stacked_optimistic_tps": s4["stacked_optimistic_tps"],
        "required_shave_frac_from_354_to_500": s4["required_shave_frac_from_354_to_500"],
        "advisor_naive_5pct_stacked_tps": s4["advisor_naive_5pct_stacked_tps"],
        "residual_to_500_tps": s4["residual_to_500_tps"],
        "attn_backend_swap_prior_frac": ATTN_BACKEND_SWAP_PRIOR_FRAC,
        "tps_added_by_this_leg": 0.0,
        **{f"selftest_{k}": int(bool(v)) for k, v in st["checks"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="strict_step_shave_stack_result",
                      artifact_type="validity", data=payload)
    finish_wandb(run)
    rid = getattr(run, "id", None)
    print(f"[step-shave] wandb logged (run {rid})", flush=True)
    return rid


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", default="strict-frontier")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    syn = synthesize()
    st = self_test(syn)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 359, "agent": "kanna",
        "kind": "strict-step-shave-stack", "synthesis": syn, "self_test": st,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[step-shave] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn, st)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "strict_step_shave_stack_results.json"

    wid = None
    if not args.no_wandb:
        wid = _maybe_log_wandb(args, payload)
    payload["wandb_run_id"] = wid
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[step-shave] wrote {out_path}  (wandb run {wid})", flush=True)

    if args.self_test:
        ok = st["strict_step_shave_stack_self_test_passes"] and payload["nan_clean"]
        print(f"[step-shave] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
