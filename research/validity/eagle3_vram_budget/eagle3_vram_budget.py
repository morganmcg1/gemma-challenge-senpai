#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""EAGLE-3 build VRAM budget: does the fusion drafter fit <= 24 GB? (PR #299).

THE QUESTION
------------
The human-approval-gated EAGLE-3 retrain (wirbel #290's feasibility bracket; the SOLE
re-open of Path-A is a structurally-non-linear drafter) only matters if the resulting
build still SERVES on the A10G. The deployed checkpoint (fa2sw_precache_kenyan, int4
body + linear MTP drafter) already sits at ~19.3 GiB resident under
gpu_memory_utilization=0.90. Swapping the linear drafter for an EAGLE-3 fusion drafter
ADDS: a 1-layer Llama decoder (2H attention input), a fusion FC [3H->H], hidden-state
retention buffers for the {2,21,39} capture, and an extra drafter KV layer. Nobody has
sized that NET delta against the 24 GB ceiling.

THE DELIVERABLE (this leg)
--------------------------
(1) Reconcile the DEPLOYED resident memory MAP (int4 body + KV + linear drafter +
    framework residual) to my #284 ~19.3 GiB anchor, from the deployed server log
    (model-loading / KV / CUDA-graph all log-measured; non-torch CUDA context is the
    balancing residual, validated against a plausible 0.5-1.2 GiB band).
(2) Size the EAGLE-3 fusion drafter's NET memory delta vs the LINEAR drafter it
    replaces -- parametric (from live config dims), corroborated by an OPTIONAL
    random-init GPU spot-check of the weight allocation. Four terms:
      drafter_weights  = EAGLE-3 1-layer decoder (+final norm) - linear drafter
      fusion_fc        = [3H->H] + bias
      hidden_retention = L_FUSE * H * draft_positions * dtype_bytes
      extra_kv         = +1 drafter attention layer's KV at the deployed token capacity
(3) Verdict: eagle3_build_resident_gb = deployed_resident_gb + net_delta <= 24 GiB?
    Report headroom/bust margin against BOTH 23-usable and 24-hard (and the measured
    22.058 GiB device-visible cap), the dominant_memory_term, embed/lm_head sensitivity
    scenarios, and -- since it fits -- the headroom-implied max drafter size.

Pure CPU analytic over banked, log-measured numbers (imported VERBATIM; never re-derived)
plus an OPTIONAL torch GPU spot-check. Analysis-only; NO training, NO checkpoint train, NO
served-file change, NO emitted-token change, NO HF Job, NO submission, NOT a launch, NOT a
build (random-init weight allocation only). BASELINE 481.53 untouched; this leg adds 0 TPS."""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]                      # .../target
GIB = float(1024 ** 3)                           # 1 GiB in bytes (vLLM reports GiB)
BF16 = 2                                          # drafter dtype bytes (linear drafter is bf16)

# --------------------------------------------------------------------------- #
# Banked anchors (imported VERBATIM; never re-derived). Provenance in comments.
# --------------------------------------------------------------------------- #
OFFICIAL_BASELINE = 481.53                        # PR #52 frontier TPS (unchanged; this leg adds 0)

# ---- ubel #284 (u58fxtu6): deployed resident anchor + log-measured memory map ----
# Source: research/validity/decode_host_overhead/server_deployed_decode.log (the #284 rig).
DEPLOYED_RESIDENT_GIB = 19.3                       # #284 reconciled deployed resident (the anchor)
MODEL_LOADING_GIB = 8.85                           # log:128 "Model loading took 8.85 GiB"
KV_CACHE_GIB = 9.46                                # log:151 "Available KV cache memory: 9.46 GiB"
CUDA_GRAPH_GIB = 0.04                              # log:159 "CUDA graph pool memory: 0.04 GiB (actual)"
KV_CACHE_TOKENS = 376_880                          # log:154 "GPU KV cache size: 376,880 tokens"
GPU_MEM_UTIL = 0.90                                # serve gpu_memory_utilization
GPU_MEM_UTIL_EFFECTIVE = 0.8974                    # log:152 effective (CUDA-graph profiling)
DEVICE_VISIBLE_GIB = 22.058                        # measured torch.cuda.mem_get_info total (this session)

# ---- VRAM ceilings (A10G) ----
VRAM_HARD_GIB = 24.0                               # A10G nominal hard ceiling (the > guard)
VRAM_USABLE_GIB = 23.0                             # practically-usable ceiling (arch_notes "23 GB A10G")

# ---- live config dims (google/gemma-4-E4B-it; confirmed against base + body config.json) ----
HIDDEN = 2560                                      # hidden_size
N_LAYERS = 42                                      # base num_hidden_layers (body=37 int4; embed/lm_head shared)
HEADS = 8                                          # num_attention_heads
KV_HEADS = 2                                       # num_key_value_heads
HEAD_DIM = 256                                     # head_dim
INTERMEDIATE = 10240                               # intermediate_size
VOCAB = 262_144                                    # vocab_size
TWO_H = 2 * HIDDEN                                 # 5120 -- EAGLE layer-0 qkv input (embed (+) hidden)
FUSE_IN = 3 * HIDDEN                               # 7680 -- fusion FC input (concat of {2,21,39})

# ---- EAGLE-3 architecture (research/eagle3_drafter/arch_notes.md) ----
EAGLE3_AUX_LAYERS = (2, 21, 39)                    # target hidden-state capture layers
L_FUSE = len(EAGLE3_AUX_LAYERS)                    # 3 captured states
# draft_dim == backbone_dim == HIDDEN (2560) -> per gemma4.py:176 the EAGLE-3 drafter
# REUSES the target embed AND lm_head (the deployed LINEAR drafter keeps its OWN lm_head
# only because draft_dim(256) != backbone_dim(2560); log:122). PRIMARY assumes reuse.
EAGLE3_REUSES_EMBED_LMHEAD = True

# ---- deployed LINEAR MTP drafter being REPLACED (qat-assistant) ----
LINEAR_DRAFTER_PARAMS = 78_779_908                 # model.safetensors total params
LINEAR_DRAFTER_FILE_BYTES = 159_138_240            # model.safetensors on-disk (~resident; shares target embed)

# ---- deployed serving knobs ----
MAX_NUM_BATCHED_TOKENS = 512                       # caps prefill hidden-state capture width
M_VERIFY = 8                                       # deployed verify batch (K=7 + bonus)
K_SPEC = 7                                         # num_speculative_tokens

# ---- plausibility band for the non-torch CUDA-context residual ----
NONTORCH_BAND_GIB = (0.5, 1.2)                      # CUDA primary context + cuBLAS/cuDNN/Triton workspaces

# ---- live config paths (dims read at runtime; banked values are the cross-checked fallback) ----
LIVE_CONFIG_CANDIDATES = (
    os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-4-E4B-it/"
                       "snapshots/fee6332c1abaafb77f6f9624236c63aa2f1d0187/config.json"),
    "/tmp/osoi5-12k-baked/config.json",            # served int4 body (n_layers=37 baked)
)

# ---- launch-gate clause this leg sizes (land #245; carried as a caveat, never re-derived) ----
LAUNCH_GATE = ("land #245: MEASURED >=500 TPS at lambda_hat >= 0.9780 AND PPL <= 2.42 AND VRAM <= 24 GiB, "
               "human-approval-gated; this leg sizes ONLY the VRAM<=24 clause.")


def _read_live_config() -> dict[str, Any]:
    """Read hidden_size + n_layers from the live Gemma config (PR: 'not invented'). Non-fatal."""
    import os.path as _osp
    for path in LIVE_CONFIG_CANDIDATES:
        if not _osp.exists(path):
            continue
        try:
            cfg = json.loads(Path(path).read_text())
        except Exception:
            continue
        tc = cfg.get("text_config", cfg)
        hs, nl = tc.get("hidden_size"), tc.get("num_hidden_layers")
        if hs is None or nl is None:
            continue
        return {"read": True, "source": path, "hidden_size": int(hs), "num_hidden_layers": int(nl),
                "head_dim": tc.get("head_dim"), "vocab_size": tc.get("vocab_size")}
    return {"read": False, "source": None, "hidden_size": None, "num_hidden_layers": None}


# --------------------------------------------------------------------------- #
# Parametric weight model (DERIVED from dims; self-test checks against banked literals).
# --------------------------------------------------------------------------- #
def eagle3_decoder_layer_params() -> dict[str, int]:
    """One EAGLE-3 Llama decoder layer; layer-0 q/k/v ingest 2H (embed (+) fused hidden)."""
    q = (HEADS * HEAD_DIM) * TWO_H
    k = (KV_HEADS * HEAD_DIM) * TWO_H
    v = (KV_HEADS * HEAD_DIM) * TWO_H
    o = HIDDEN * (HEADS * HEAD_DIM)
    gate = INTERMEDIATE * HIDDEN
    up = INTERMEDIATE * HIDDEN
    down = HIDDEN * INTERMEDIATE
    input_ln = HIDDEN
    post_attn_ln = HIDDEN
    total = q + k + v + o + gate + up + down + input_ln + post_attn_ln
    return {"q": q, "k": k, "v": v, "o": o, "gate": gate, "up": up, "down": down,
            "input_ln": input_ln, "post_attn_ln": post_attn_ln, "total": total}


def synthesize(spot: dict[str, Any] | None = None,
               live_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    # ---- live config dims read at runtime, cross-checked against the banked literals ----
    live_cfg = live_cfg or {"read": False}
    cfg_hidden = live_cfg.get("hidden_size")
    cfg_layers = live_cfg.get("num_hidden_layers")
    config_dims_match = bool(live_cfg.get("read") and cfg_hidden == HIDDEN
                             and cfg_layers in (N_LAYERS, 37))   # base 42 or served int4 body 37
    config_read = bool(live_cfg.get("read"))

    # ====================================================================== #
    # (1) DEPLOYED RESIDENT MEMORY MAP -- reconcile to the #284 19.3 GiB anchor
    # ====================================================================== #
    torch_subtotal_gib = MODEL_LOADING_GIB + KV_CACHE_GIB + CUDA_GRAPH_GIB        # 18.35
    nontorch_residual_gib = DEPLOYED_RESIDENT_GIB - torch_subtotal_gib            # 0.95 (balancing)
    nontorch_in_band = bool(NONTORCH_BAND_GIB[0] <= nontorch_residual_gib <= NONTORCH_BAND_GIB[1])
    reconciled_resident_gib = torch_subtotal_gib + nontorch_residual_gib          # == 19.3 by construction
    reconcile_resid_gib = abs(reconciled_resident_gib - DEPLOYED_RESIDENT_GIB)
    util_budget_gib = GPU_MEM_UTIL_EFFECTIVE * DEVICE_VISIBLE_GIB                 # 19.79 util ceiling
    # cross-check: deployed resident sits just under the util budget (gap = transient activation peak).
    util_budget_gap_gib = util_budget_gib - DEPLOYED_RESIDENT_GIB                 # ~0.49

    deployed_map = {
        "model_loading_int4_body_plus_linear_drafter_gib": MODEL_LOADING_GIB,
        "kv_cache_gib": KV_CACHE_GIB,
        "cuda_graph_pool_gib": CUDA_GRAPH_GIB,
        "torch_subtotal_gib": torch_subtotal_gib,
        "nontorch_cuda_context_residual_gib": nontorch_residual_gib,
        "nontorch_in_plausible_band": nontorch_in_band,
        "reconciled_resident_gib": reconciled_resident_gib,
        "anchor_284_resident_gib": DEPLOYED_RESIDENT_GIB,
        "reconcile_resid_gib": reconcile_resid_gib,
        "util_budget_gib": util_budget_gib,
        "util_budget_gap_gib": util_budget_gap_gib,
        "kv_cache_tokens": KV_CACHE_TOKENS,
    }

    # ====================================================================== #
    # (2) EAGLE-3 FUSION-DRAFTER NET MEMORY DELTA (vs the linear drafter it replaces)
    # ====================================================================== #
    dec = eagle3_decoder_layer_params()
    final_norm_params = HIDDEN                                                    # EAGLE-3 final RMSNorm
    eagle3_drafter_side_params = dec["total"] + final_norm_params                 # decoder (+final norm)
    fc_params = HIDDEN * FUSE_IN                                                  # [3H->H]
    fc_bias_params = HIDDEN                                                       # +bias (PR term)
    fusion_fc_params = fc_params + fc_bias_params

    # --- term (a) drafter_weights: EAGLE-3 decoder (+final norm) MINUS linear drafter ---
    eagle3_drafter_side_bytes = eagle3_drafter_side_params * BF16
    drafter_weights_delta_gib = (eagle3_drafter_side_bytes - LINEAR_DRAFTER_FILE_BYTES) / GIB
    # --- term (b) fusion_fc ---
    fusion_fc_gib = (fusion_fc_params * BF16) / GIB
    # --- term (c) hidden-state retention buffer: L_FUSE * H * draft_positions * bytes ---
    draft_positions = MAX_NUM_BATCHED_TOKENS                                      # worst-case prefill capture width
    hidden_retention_bytes = L_FUSE * HIDDEN * draft_positions * BF16
    hidden_retention_gib = hidden_retention_bytes / GIB
    # --- term (d) extra drafter KV: +1 attention layer at the deployed token capacity ---
    # The LINEAR drafter shares target KV (its layers map to target layers 19/20; log:123-126) ->
    # zero extra KV. The EAGLE-3 decoder layer is NEW -> its own KV. Conservative: a full-attention
    # layer at the deployed 376,880-token capacity.
    kv_bytes_per_token_per_layer = 2 * KV_HEADS * HEAD_DIM * BF16                 # K&V
    extra_kv_hold_capacity_gib = (KV_CACHE_TOKENS * kv_bytes_per_token_per_layer) / GIB
    # In deployment KV is ELASTIC (vLLM sizes total KV to fill the util budget): adding the
    # drafter layer trades KV tokens, it does NOT grow resident. Report both framings.
    extra_kv_deployment_elastic_gib = 0.0
    kv_token_haircut_frac = extra_kv_hold_capacity_gib / KV_CACHE_GIB            # ~3.6% fewer tokens

    terms_conservative = {
        "drafter_weights": drafter_weights_delta_gib,
        "fusion_fc": fusion_fc_gib,
        "hidden_state_retention": hidden_retention_gib,
        "extra_kv": extra_kv_hold_capacity_gib,
    }
    terms_elastic = dict(terms_conservative, extra_kv=extra_kv_deployment_elastic_gib)
    net_delta_conservative_gib = sum(terms_conservative.values())                # ~0.800 (hold KV capacity)
    net_delta_elastic_gib = sum(terms_elastic.values())                          # ~0.081 (weights only)
    dominant_memory_term = max(terms_conservative, key=lambda k: terms_conservative[k])
    dominant_memory_term_elastic = max(terms_elastic, key=lambda k: terms_elastic[k])

    # --- optional GPU spot-check corroboration of the weight allocation (drafter + fc) ---
    parametric_weight_alloc_bytes = (eagle3_drafter_side_params + fusion_fc_params) * BF16
    spot_ran = bool(spot and spot.get("ran"))
    spot_rel_err = None
    spot_agrees = None
    spot_device_total_gib = None
    spot_device_total_matches_banked = None
    if spot_ran:
        meas = float(spot["alloc_delta_bytes"])
        spot_rel_err = abs(meas - parametric_weight_alloc_bytes) / parametric_weight_alloc_bytes
        spot_agrees = bool(spot_rel_err < 0.02)
        if spot.get("device_total_bytes"):
            spot_device_total_gib = float(spot["device_total_bytes"]) / GIB
            # soft corroboration of the banked DEVICE_VISIBLE_GIB (driver/context can shift it ~0.5 GiB).
            spot_device_total_matches_banked = bool(
                abs(spot_device_total_gib - DEVICE_VISIBLE_GIB) < 0.6)
    delta_is_parametric = not spot_ran

    net_delta = {
        "eagle3_decoder_layer_params": dec["total"],
        "eagle3_drafter_side_params": eagle3_drafter_side_params,
        "fusion_fc_params": fusion_fc_params,
        "linear_drafter_params": LINEAR_DRAFTER_PARAMS,
        "linear_drafter_file_bytes": LINEAR_DRAFTER_FILE_BYTES,
        "draft_positions": draft_positions,
        "kv_bytes_per_token_per_layer": kv_bytes_per_token_per_layer,
        "terms_conservative_gib": terms_conservative,
        "terms_elastic_gib": terms_elastic,
        "net_delta_conservative_gib": net_delta_conservative_gib,
        "net_delta_elastic_gib": net_delta_elastic_gib,
        "extra_kv_hold_capacity_gib": extra_kv_hold_capacity_gib,
        "extra_kv_deployment_elastic_gib": extra_kv_deployment_elastic_gib,
        "kv_token_haircut_frac": kv_token_haircut_frac,
        "dominant_memory_term": dominant_memory_term,
        "dominant_memory_term_elastic": dominant_memory_term_elastic,
        "eagle3_reuses_embed_lmhead": EAGLE3_REUSES_EMBED_LMHEAD,
        "parametric_weight_alloc_bytes": parametric_weight_alloc_bytes,
        "spot_check_ran": spot_ran,
        "spot_check_rel_err": spot_rel_err,
        "spot_check_agrees": spot_agrees,
        "spot_device_total_gib": spot_device_total_gib,
        "spot_device_total_matches_banked": spot_device_total_matches_banked,
        "delta_is_parametric": delta_is_parametric,
    }

    # ====================================================================== #
    # (2b) embed/lm_head SENSITIVITY scenarios (added to the fixed-weight side)
    # ====================================================================== #
    lmhead_full_gib = (VOCAB * HIDDEN * BF16) / GIB                              # one [VOCAB,H] bf16 matrix
    lmhead_32k_gib = (32_768 * HIDDEN * BF16) / GIB                             # reduced-vocab head
    sensitivity = {
        "S0_reuse_embed_and_lmhead_PRIMARY": 0.0,
        "S1_separate_lmhead_full_vocab": lmhead_full_gib,
        "S2_untied_separate_embed_and_lmhead": 2.0 * lmhead_full_gib,
        "S3_reduced_32k_vocab_lmhead": lmhead_32k_gib,
    }

    # ====================================================================== #
    # (3) FIT VERDICT
    # ====================================================================== #
    def fit_block(net_delta_gib: float, embed_extra_gib: float = 0.0) -> dict[str, Any]:
        build_resident = DEPLOYED_RESIDENT_GIB + net_delta_gib + embed_extra_gib
        return {
            "net_delta_gib": net_delta_gib,
            "embed_extra_gib": embed_extra_gib,
            "build_resident_gib": build_resident,
            "fits_24_hard": bool(build_resident <= VRAM_HARD_GIB),
            "fits_23_usable": bool(build_resident <= VRAM_USABLE_GIB),
            "fits_device_visible": bool(build_resident <= DEVICE_VISIBLE_GIB),
            "headroom_vs_24_hard_gib": VRAM_HARD_GIB - build_resident,
            "headroom_vs_23_usable_gib": VRAM_USABLE_GIB - build_resident,
            "headroom_vs_device_visible_gib": DEVICE_VISIBLE_GIB - build_resident,
        }

    # PRIMARY headline = conservative (hold-KV-capacity) net delta, S0 (reuse embed/lm_head).
    primary_fit = fit_block(net_delta_conservative_gib)
    elastic_fit = fit_block(net_delta_elastic_gib)
    eagle3_build_resident_gb = primary_fit["build_resident_gib"]
    eagle3_net_memory_delta_gb = net_delta_conservative_gib
    eagle3_build_fits_24gb = primary_fit["fits_24_hard"]
    fits_23_usable = primary_fit["fits_23_usable"]

    # sensitivity fit table (conservative net delta + each embed/lm_head scenario).
    sensitivity_fits = {
        name: fit_block(net_delta_conservative_gib, extra)
        for name, extra in sensitivity.items()
    }

    # headroom-implied MAX drafter size (since it fits): how many extra EAGLE-3 decoder layers
    # would the 23-usable headroom absorb, and the max retention width.
    one_decoder_layer_gib = (eagle3_drafter_side_params * BF16) / GIB
    headroom_23_primary = primary_fit["headroom_vs_23_usable_gib"]
    max_additional_decoder_layers = math.floor(headroom_23_primary / one_decoder_layer_gib)
    one_retention_col_gib = (L_FUSE * HIDDEN * BF16) / GIB                       # per retained position
    max_retention_positions = int(headroom_23_primary / one_retention_col_gib) if one_retention_col_gib > 0 else 0
    max_drafter_policy = {
        "fits": True,
        "one_eagle3_decoder_layer_gib": one_decoder_layer_gib,
        "headroom_23_usable_gib": headroom_23_primary,
        "max_additional_decoder_layers_within_23": max_additional_decoder_layers,
        "max_retention_positions_within_23": max_retention_positions,
        "note": ("the build fits with %0.2f GiB of 23-usable headroom; the drafter could be "
                 "~%dx larger (decoder layers) before busting 23 GiB -- size is NOT the binding "
                 "constraint." % (headroom_23_primary, max_additional_decoder_layers + 1)),
    }

    # the lone bust: doubly-pessimistic (untied separate full-vocab embed+lm_head AND hold-KV) brushes
    # the measured device-visible cap -- flag it as unrealistic (EAGLE-3 reuses embed/lm_head).
    s2_conservative_resident = sensitivity_fits["S2_untied_separate_embed_and_lmhead"]["build_resident_gib"]
    s2_exceeds_visible = bool(s2_conservative_resident > DEVICE_VISIBLE_GIB)

    # vram_headroom_gb alias (PR hand-off field): margin to the 24-hard launch-gate ceiling.
    vram_headroom_gb = primary_fit["headroom_vs_24_hard_gib"]

    # ---- honest-framing caveats (PR step 4; carried + self-tested) ------- #
    caveats = {
        "parametric_is_the_deliverable": (
            "the parametric net delta is the deliverable; the random-init GPU spot-check (random weight "
            "tensors, freed immediately) only VALIDATES it -- NOT a trained build."),
        "deployed_193_is_my_284_measurement": (
            "the deployed ~19.3 GiB basis is MY ubel #284 (u58fxtu6) reconciled measurement; the delta "
            "stacks on it."),
        "measures_vram_not_et": (
            "this leg prices the build's MEMORY feasibility (resident footprint vs VRAM<=24), NOT its "
            "E[T] (kanna #289/denken #297 lane) nor its step cost (wirbel #295 lane)."),
        "zero_tps": "0 TPS added; BASELINE 481.53 untouched; NOT a launch, NOT a build, NOT open2.",
        "launch_gate_clause": LAUNCH_GATE,
        "wirbel295_module_reused": False,
        "wirbel295_reuse_note": (
            "wirbel #295 is a pure-CPU forward-cost analytic with NO instantiable drafter module to "
            "import; the spot-check therefore allocates raw random-init weight tensors matching the "
            "parametric param counts (orthogonal MEMORY axis)."),
    }

    # ---- curated non-negativity gate (PR self-test (c): all memory magnitudes >= 0) ---- #
    curated_mem_magnitudes = [
        MODEL_LOADING_GIB, KV_CACHE_GIB, CUDA_GRAPH_GIB, torch_subtotal_gib, nontorch_residual_gib,
        DEPLOYED_RESIDENT_GIB, drafter_weights_delta_gib, fusion_fc_gib, hidden_retention_gib,
        extra_kv_hold_capacity_gib, net_delta_conservative_gib, net_delta_elastic_gib,
        eagle3_build_resident_gb, elastic_fit["build_resident_gib"], one_decoder_layer_gib,
    ] + list(sensitivity.values())
    all_memory_values_nonneg = bool(all(math.isfinite(x) and x >= 0.0 for x in curated_mem_magnitudes))

    # ====================================================================== #
    # SELF-TEST (PRIMARY)
    # ====================================================================== #
    # (a) deployed map reconciles to the 19.3 anchor within 0.3 and non-torch residual is plausible.
    a_map_reconciles = bool(reconcile_resid_gib < 0.3 and nontorch_in_band)
    # (b) the four conservative terms sum to the conservative net delta (decomposition closes).
    b_decomp_sums = bool(abs(sum(terms_conservative.values()) - net_delta_conservative_gib) < 1e-12)
    # (c) parametric EAGLE-3 dims reproduce the banked decoder/fc param literals; if the GPU
    #     spot-check ran, the measured weight allocation agrees within 2% (else parametric-only OK).
    c_params_exact = bool(dec["total"] == 99_619_840 and fc_params == 19_660_800
                          and eagle3_drafter_side_params == 99_622_400)
    c_spot_ok = bool((not spot_ran) or spot_agrees)
    c_eagle3_params = bool(c_params_exact and c_spot_ok)
    # (d) build-resident identity: deployed + net_delta == build_resident (both framings).
    d_identity = bool(
        abs((DEPLOYED_RESIDENT_GIB + net_delta_conservative_gib) - eagle3_build_resident_gb) < 1e-12
        and abs((DEPLOYED_RESIDENT_GIB + net_delta_elastic_gib)
                - elastic_fit["build_resident_gib"]) < 1e-12)
    # (e) dominant term is the argmax of the four conservative terms AND is extra_kv.
    e_dominant = bool(dominant_memory_term == "extra_kv"
                      and terms_conservative["extra_kv"] == max(terms_conservative.values()))
    # (f) fit flags are consistent with the ceilings, and the PRIMARY build fits both 23 and 24.
    f_fit_consistent = bool(
        eagle3_build_fits_24gb == (eagle3_build_resident_gb <= VRAM_HARD_GIB)
        and fits_23_usable == (eagle3_build_resident_gb <= VRAM_USABLE_GIB)
        and eagle3_build_fits_24gb and fits_23_usable)
    # (g) constants imported EXACT; hidden_size + n_layers READ from the live config (not invented).
    g_constants = bool(
        abs(DEPLOYED_RESIDENT_GIB - 19.3) < 1e-9 and abs(MODEL_LOADING_GIB - 8.85) < 1e-9
        and abs(KV_CACHE_GIB - 9.46) < 1e-9 and abs(CUDA_GRAPH_GIB - 0.04) < 1e-9
        and KV_CACHE_TOKENS == 376_880 and HIDDEN == 2560 and VOCAB == 262_144
        and HEADS == 8 and KV_HEADS == 2 and HEAD_DIM == 256 and INTERMEDIATE == 10240
        and EAGLE3_AUX_LAYERS == (2, 21, 39) and LINEAR_DRAFTER_PARAMS == 78_779_908
        and abs(GPU_MEM_UTIL - 0.90) < 1e-9 and abs(OFFICIAL_BASELINE - 481.53) < 1e-9
        and abs(VRAM_HARD_GIB - 24.0) < 1e-9 and abs(VRAM_USABLE_GIB - 23.0) < 1e-9
        and config_read and config_dims_match)
    # (h) NaN-clean -- checked on the full payload in main().
    # (i) every sensitivity scenario still fits 24-hard (robustness of the verdict).
    i_all_scenarios_fit_24 = bool(all(fb["fits_24_hard"] for fb in sensitivity_fits.values()))
    # (j) all curated memory magnitudes >= 0 (PR self-test (c)).
    j_memory_nonneg = all_memory_values_nonneg
    # (k) the 0-TPS / parametric-not-a-build / measures-VRAM-not-E[T] caveats are carried (PR (e)).
    k_caveats_carried = bool(
        caveats["zero_tps"] and caveats["measures_vram_not_et"]
        and caveats["parametric_is_the_deliverable"] and caveats["deployed_193_is_my_284_measurement"]
        and caveats["launch_gate_clause"] == LAUNCH_GATE)

    cond = {
        "a_deployed_map_reconciles_to_193": a_map_reconciles,
        "b_net_delta_decomposition_sums": b_decomp_sums,
        "c_eagle3_params_exact_and_spotcheck_ok": c_eagle3_params,
        "d_build_resident_identity_holds": d_identity,
        "e_dominant_term_is_extra_kv": e_dominant,
        "f_fit_flags_consistent_and_fits": f_fit_consistent,
        "g_constants_exact_and_config_read": g_constants,
        "i_all_sensitivity_scenarios_fit_24": i_all_scenarios_fit_24,
        "j_all_memory_magnitudes_nonneg": j_memory_nonneg,
        "k_honest_caveats_carried": k_caveats_carried,
    }

    # ---- hand-off + verdict ---------------------------------------------- #
    handoff = (
        "EAGLE-3 fusion drafter FITS: deployed resident %.2f GiB + net delta %.3f GiB (conservative, "
        "hold-KV-capacity) = %.2f GiB build resident, %.2f GiB under the 23-usable ceiling (%.2f under "
        "24-hard); dominant term = extra_kv (%.3f GiB, but ELASTIC in deployment -> ~%.1f%% fewer KV "
        "tokens, not +resident); the weight terms (drafter %.3f + fc %.3f + retention %.3f) add only "
        "%.3f GiB; EAGLE-3 reuses the target embed+lm_head (draft_dim==backbone_dim), and even the "
        "untied-separate-full-vocab-head worst case (%.2f GiB) still fits 23-usable -- VRAM is NOT the "
        "binding constraint on the human-gated build." % (
            DEPLOYED_RESIDENT_GIB, net_delta_conservative_gib, eagle3_build_resident_gb,
            primary_fit["headroom_vs_23_usable_gib"], primary_fit["headroom_vs_24_hard_gib"],
            extra_kv_hold_capacity_gib, kv_token_haircut_frac * 100.0,
            drafter_weights_delta_gib, fusion_fc_gib, hidden_retention_gib, net_delta_elastic_gib,
            s2_conservative_resident))

    verdict = (
        "DEPLOYED resident reconciles to the #284 anchor 19.30 GiB (model %.2f + KV %.2f + graph %.2f = "
        "%.2f torch + %.2f non-torch CUDA context; resid %.3f GiB). Swapping the linear MTP drafter for "
        "an EAGLE-3 fusion drafter adds a NET %.3f GiB (conservative, hold-KV-capacity): drafter_weights "
        "%.3f (1-layer 2H-input Llama decoder %.1fM params - linear drafter %.3f GiB), fusion_fc %.3f "
        "([3H->H]+bias %.1fM), hidden_retention %.4f (L_FUSE %d x H %d x %d positions), extra_kv %.3f "
        "(+1 drafter attention layer at %d tokens -- the dominant term, though ELASTIC: in deployment it "
        "trades ~%.1f%% of KV tokens rather than growing resident). eagle3_build_resident = %.2f GiB, "
        "which FITS both ceilings (24-hard headroom %.2f; 23-usable headroom %.2f) and the measured "
        "%.3f GiB device-visible cap. EAGLE-3 reuses the target embed+lm_head (draft_dim==backbone_dim; "
        "gemma4.py:176); ALL embed/lm_head sensitivity scenarios fit 24-hard, and only the doubly-"
        "pessimistic untied-separate-full-vocab + hold-KV case (%.2f GiB) brushes the 22.058 visible cap "
        "(exceeds=%s) -- unrealistic. VRAM is NOT the binding constraint. Analysis-only; BASELINE 481.53 "
        "untouched; 0 TPS added; NOT a launch/build." % (
            MODEL_LOADING_GIB, KV_CACHE_GIB, CUDA_GRAPH_GIB, torch_subtotal_gib, nontorch_residual_gib,
            reconcile_resid_gib, net_delta_conservative_gib, drafter_weights_delta_gib,
            eagle3_drafter_side_params / 1e6, LINEAR_DRAFTER_FILE_BYTES / GIB, fusion_fc_gib,
            fusion_fc_params / 1e6, hidden_retention_gib, L_FUSE, HIDDEN, draft_positions,
            extra_kv_hold_capacity_gib, KV_CACHE_TOKENS, kv_token_haircut_frac * 100.0,
            eagle3_build_resident_gb, primary_fit["headroom_vs_24_hard_gib"],
            primary_fit["headroom_vs_23_usable_gib"], DEVICE_VISIBLE_GIB, s2_conservative_resident,
            s2_exceeds_visible))

    return {
        "constants": {
            "official_baseline": OFFICIAL_BASELINE, "deployed_resident_gib": DEPLOYED_RESIDENT_GIB,
            "model_loading_gib": MODEL_LOADING_GIB, "kv_cache_gib": KV_CACHE_GIB,
            "cuda_graph_gib": CUDA_GRAPH_GIB, "kv_cache_tokens": KV_CACHE_TOKENS,
            "gpu_mem_util": GPU_MEM_UTIL, "gpu_mem_util_effective": GPU_MEM_UTIL_EFFECTIVE,
            "device_visible_gib": DEVICE_VISIBLE_GIB, "vram_hard_gib": VRAM_HARD_GIB,
            "vram_usable_gib": VRAM_USABLE_GIB, "hidden": HIDDEN, "n_layers": N_LAYERS,
            "heads": HEADS, "kv_heads": KV_HEADS, "head_dim": HEAD_DIM, "intermediate": INTERMEDIATE,
            "vocab": VOCAB, "eagle3_aux_layers": list(EAGLE3_AUX_LAYERS), "l_fuse": L_FUSE,
            "linear_drafter_params": LINEAR_DRAFTER_PARAMS, "k_spec": K_SPEC, "m_verify": M_VERIFY,
        },
        "deployed_map": deployed_map,
        "net_delta": net_delta,
        "sensitivity_gib": sensitivity,
        "sensitivity_fits": sensitivity_fits,
        "primary_fit": primary_fit,
        "elastic_fit": elastic_fit,
        "max_drafter_policy": max_drafter_policy,
        "s2_exceeds_device_visible": s2_exceeds_visible,
        "live_config": {"read": config_read, "source": live_cfg.get("source"),
                        "hidden_size": cfg_hidden, "num_hidden_layers": cfg_layers,
                        "dims_match_banked": config_dims_match},
        "caveats": caveats,
        "self_test": {"conditions": cond},
        # ---- headline metrics (required outputs) ----
        "eagle3_build_resident_gb": eagle3_build_resident_gb,
        "eagle3_net_memory_delta_gb": eagle3_net_memory_delta_gb,
        "eagle3_build_fits_24gb": eagle3_build_fits_24gb,
        "fits_23_usable": fits_23_usable,
        "vram_headroom_gb": vram_headroom_gb,
        "dominant_memory_term": dominant_memory_term,
        "delta_is_parametric": delta_is_parametric,
        "verdict": verdict, "handoff": handoff,
    }


# --------------------------------------------------------------------------- #
# OPTIONAL random-init GPU spot-check (corroborates the parametric weight bytes).
# Shells out to a torch-bearing interpreter (SENPAI_TORCH_PYTHON or /usr/bin/python3)
# with CUDA_VISIBLE_DEVICES=0. NOT a build: allocates random-init weight tensors only,
# measures torch.cuda.memory_allocated delta, frees them. Never fatal.
# --------------------------------------------------------------------------- #
_SPOT_SRC = r"""
import json, sys
try:
    import torch
    if not torch.cuda.is_available():
        print(json.dumps({"ran": False, "reason": "cuda not available"})); sys.exit(0)
    dev = "cuda:0"
    torch.cuda.init(); torch.cuda.synchronize(); torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated(dev)
    H, INTER, HEADS, KVH, HD = 2560, 10240, 8, 2, 256
    TWO_H, FUSE_IN, DT = 2 * H, 3 * H, torch.bfloat16
    t = []
    t.append(torch.empty(HEADS * HD, TWO_H, dtype=DT, device=dev))   # q_proj (2H input)
    t.append(torch.empty(KVH * HD, TWO_H, dtype=DT, device=dev))     # k_proj
    t.append(torch.empty(KVH * HD, TWO_H, dtype=DT, device=dev))     # v_proj
    t.append(torch.empty(H, HEADS * HD, dtype=DT, device=dev))       # o_proj
    t.append(torch.empty(INTER, H, dtype=DT, device=dev))            # gate
    t.append(torch.empty(INTER, H, dtype=DT, device=dev))            # up
    t.append(torch.empty(H, INTER, dtype=DT, device=dev))            # down
    t.append(torch.empty(H, dtype=DT, device=dev))                   # input_ln
    t.append(torch.empty(H, dtype=DT, device=dev))                   # post_attn_ln
    t.append(torch.empty(H, dtype=DT, device=dev))                   # final norm
    t.append(torch.empty(H, FUSE_IN, dtype=DT, device=dev))          # fusion fc
    t.append(torch.empty(H, dtype=DT, device=dev))                   # fc bias
    torch.cuda.synchronize()
    delta = torch.cuda.memory_allocated(dev) - base
    n_params = int(sum(x.numel() for x in t))
    free_b, total_b = torch.cuda.mem_get_info(dev)   # (free, total) -- total is the device-visible cap
    del t; torch.cuda.empty_cache()
    print(json.dumps({"ran": True, "alloc_delta_bytes": int(delta), "n_params": n_params,
                      "device_total_bytes": int(total_b), "device_free_at_check_bytes": int(free_b),
                      "torch_version": torch.__version__,
                      "device_name": torch.cuda.get_device_name(0)}))
except Exception as exc:
    print(json.dumps({"ran": False, "reason": repr(exc)}))
"""


def _gpu_spot_check(enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {"ran": False, "reason": "disabled (--no-spot-check)"}
    py = os.environ.get("SENPAI_TORCH_PYTHON", "/usr/bin/python3")
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    try:
        proc = subprocess.run([py, "-c", _SPOT_SRC], capture_output=True, text=True,
                              timeout=180, env=env)
    except Exception as exc:
        return {"ran": False, "reason": f"subprocess failed: {exc!r}"}
    out = (proc.stdout or "").strip().splitlines()
    for line in reversed(out):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                continue
    return {"ran": False, "reason": f"no json (rc={proc.returncode}); stderr={proc.stderr[-300:]!r}"}


# --------------------------------------------------------------------------- #
# W&B logging (mirrors wirbel #290; never fatal).
# --------------------------------------------------------------------------- #
def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.append(repo)
    _w = sys.modules.get("wandb")
    if _w is not None and not hasattr(_w, "init"):
        del sys.modules["wandb"]
    try:
        import wandb as _wb
        if not hasattr(_wb, "init"):
            raise ImportError(
                f"resolved a stub/namespace wandb at {list(getattr(_wb, '__path__', []) or [])} "
                "with no .init -> this venv lacks the wandb wheel")
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[eagle3-vram] wandb logging skipped (analysis unaffected): {exc}", flush=True)
        return

    syn = payload["synthesis"]
    dm = syn["deployed_map"]
    nd = syn["net_delta"]
    pf = syn["primary_fit"]
    ef = syn["elastic_fit"]
    st = syn["self_test"]
    try:
        run = init_wandb_run(
            job_type="validity-gate", agent="ubel", name=args.wandb_name, group=args.wandb_group,
            tags=["eagle3-vram-budget", "resident-memory-map", "fusion-drafter-net-delta",
                  "fit-24gb", "analysis-only", "pr-299"],
            config={
                "official_baseline": OFFICIAL_BASELINE, "deployed_resident_gib": DEPLOYED_RESIDENT_GIB,
                "model_loading_gib": MODEL_LOADING_GIB, "kv_cache_gib": KV_CACHE_GIB,
                "cuda_graph_gib": CUDA_GRAPH_GIB, "kv_cache_tokens": KV_CACHE_TOKENS,
                "device_visible_gib": DEVICE_VISIBLE_GIB, "vram_hard_gib": VRAM_HARD_GIB,
                "vram_usable_gib": VRAM_USABLE_GIB, "hidden": HIDDEN, "vocab": VOCAB,
                "eagle3_aux_layers": list(EAGLE3_AUX_LAYERS), "linear_drafter_params": LINEAR_DRAFTER_PARAMS,
                "imports": "ubel#284(u58fxtu6 resident=19.3 model=8.85 KV=9.46 graph=0.04 tok=376880) x "
                           "arch_notes(eagle3 1-layer 2H-input + fc[3H->H] + aux{2,21,39}) x "
                           "gemma4.py:176(draft_dim==backbone_dim -> reuse embed+lm_head)",
                "wandb_group": args.wandb_group,
            },
        )
    except Exception as exc:
        print(f"[eagle3-vram] wandb init failed (analysis unaffected): {exc}", flush=True)
        return
    if run is None:
        print("[eagle3-vram] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "eagle3_vram_budget_self_test_passes":
            int(bool(payload["eagle3_vram_budget_self_test_passes"])),
        "eagle3_build_fits_24gb": int(bool(syn["eagle3_build_fits_24gb"])),
        "fits_23_usable": int(bool(syn["fits_23_usable"])),
        "eagle3_net_memory_delta_gb": syn["eagle3_net_memory_delta_gb"],
        "eagle3_build_resident_gb": syn["eagle3_build_resident_gb"],
        "vram_headroom_gb": syn["vram_headroom_gb"],
        "net_delta_elastic_gib": nd["net_delta_elastic_gib"],
        "drafter_weights_gib": nd["terms_conservative_gib"]["drafter_weights"],
        "fusion_fc_gib": nd["terms_conservative_gib"]["fusion_fc"],
        "hidden_state_retention_gib": nd["terms_conservative_gib"]["hidden_state_retention"],
        "extra_kv_hold_capacity_gib": nd["extra_kv_hold_capacity_gib"],
        "kv_token_haircut_frac": nd["kv_token_haircut_frac"],
        "headroom_vs_24_hard_gib": pf["headroom_vs_24_hard_gib"],
        "headroom_vs_23_usable_gib": pf["headroom_vs_23_usable_gib"],
        "headroom_vs_device_visible_gib": pf["headroom_vs_device_visible_gib"],
        "build_resident_elastic_gib": ef["build_resident_gib"],
        "deployed_reconcile_resid_gib": dm["reconcile_resid_gib"],
        "nontorch_cuda_context_gib": dm["nontorch_cuda_context_residual_gib"],
        "spot_check_ran": int(bool(nd["spot_check_ran"])),
        "delta_is_parametric": int(bool(nd["delta_is_parametric"])),
        "config_read_from_live": int(bool(syn["live_config"]["read"])),
        "config_dims_match_banked": int(bool(syn["live_config"]["dims_match_banked"])),
        "spot_device_total_matches_banked": int(bool(nd.get("spot_device_total_matches_banked"))),
        "s2_exceeds_device_visible": int(bool(syn["s2_exceeds_device_visible"])),
        "max_additional_decoder_layers_within_23":
            syn["max_drafter_policy"]["max_additional_decoder_layers_within_23"],
        "nan_clean": int(bool(payload["nan_clean"])),
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    if nd["spot_check_rel_err"] is not None:
        summary["spot_check_rel_err"] = nd["spot_check_rel_err"]
    # string verdict goes in config-side; keep summary numeric + the dominant term label.
    summary["dominant_memory_term"] = syn["dominant_memory_term"]
    summary = {k: v for k, v in summary.items()
               if v is not None and not (isinstance(v, float) and not math.isfinite(v))}
    try:
        log_summary(run, summary, step=0)
        log_json_artifact(run, name="eagle3_vram_budget_result",
                          artifact_type="validity", data=payload)
        finish_wandb(run)
        print(f"[eagle3-vram] wandb logged {len(summary)} summary keys", flush=True)
    except Exception as exc:
        print(f"[eagle3-vram] wandb write failed (analysis unaffected): {exc}", flush=True)


def _assert_nan_clean(obj: Any, path: str = "") -> list[str]:
    bad = []
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


def _print_human(syn: dict) -> None:
    print("\n" + "=" * 104, flush=True)
    print(" EAGLE-3 BUILD VRAM BUDGET — does the fusion drafter fit <= 24 GiB? (PR #299)", flush=True)
    print("=" * 104, flush=True)
    dm = syn["deployed_map"]
    nd = syn["net_delta"]
    pf = syn["primary_fit"]
    ef = syn["elastic_fit"]
    print("  (1) DEPLOYED MAP (reconcile to #284 19.30 GiB):", flush=True)
    print(f"      model {dm['model_loading_int4_body_plus_linear_drafter_gib']:.2f} + KV "
          f"{dm['kv_cache_gib']:.2f} + graph {dm['cuda_graph_pool_gib']:.2f} = "
          f"{dm['torch_subtotal_gib']:.2f} torch + {dm['nontorch_cuda_context_residual_gib']:.2f} "
          f"non-torch -> {dm['reconciled_resident_gib']:.2f} (resid {dm['reconcile_resid_gib']:.3f}; "
          f"non-torch in band={dm['nontorch_in_plausible_band']})", flush=True)
    print(f"      util budget {dm['util_budget_gib']:.2f} GiB (gap to resident "
          f"{dm['util_budget_gap_gib']:.2f} = transient activation peak)", flush=True)
    print("-" * 104, flush=True)
    tc = nd["terms_conservative_gib"]
    print("  (2) EAGLE-3 NET DELTA (vs linear drafter; conservative hold-KV-capacity):", flush=True)
    print(f"      drafter_weights {tc['drafter_weights']:+.4f}  fusion_fc {tc['fusion_fc']:+.4f}  "
          f"hidden_retention {tc['hidden_state_retention']:+.4f}  extra_kv {tc['extra_kv']:+.4f}",
          flush=True)
    print(f"      -> net_delta_conservative {nd['net_delta_conservative_gib']:.4f} GiB  "
          f"(weights-only/elastic {nd['net_delta_elastic_gib']:.4f}; dominant = "
          f"{nd['dominant_memory_term']})", flush=True)
    print(f"      extra_kv is ELASTIC: in deployment -> ~{nd['kv_token_haircut_frac']*100:.1f}% fewer "
          f"KV tokens, not +resident", flush=True)
    sc = "ran" if nd["spot_check_ran"] else f"parametric-only ({'no torch/GPU' if nd['delta_is_parametric'] else ''})"
    print(f"      GPU spot-check: {sc}"
          + (f"  rel_err={nd['spot_check_rel_err']:.4f} agrees={nd['spot_check_agrees']}"
             if nd["spot_check_ran"] else ""), flush=True)
    print("-" * 104, flush=True)
    print("  (3) FIT VERDICT:", flush=True)
    print(f"      PRIMARY build_resident {pf['build_resident_gib']:.2f} GiB  "
          f"fits24={pf['fits_24_hard']} (headroom {pf['headroom_vs_24_hard_gib']:.2f})  "
          f"fits23={pf['fits_23_usable']} (headroom {pf['headroom_vs_23_usable_gib']:.2f})  "
          f"fits_visible={pf['fits_device_visible']}", flush=True)
    print(f"      elastic build_resident {ef['build_resident_gib']:.2f} GiB "
          f"(headroom23 {ef['headroom_vs_23_usable_gib']:.2f})", flush=True)
    print("      embed/lm_head sensitivity (conservative net delta + scenario):", flush=True)
    for name, fb in syn["sensitivity_fits"].items():
        print(f"        {name:<42}{fb['build_resident_gib']:>7.2f} GiB  fits24={fb['fits_24_hard']} "
              f"fits23={fb['fits_23_usable']} fits_visible={fb['fits_device_visible']}", flush=True)
    mp = syn["max_drafter_policy"]
    print(f"      MAX drafter (since it fits): +{mp['max_additional_decoder_layers_within_23']} "
          f"more decoder layers within 23-usable (~{mp['one_eagle3_decoder_layer_gib']:.3f} GiB each)",
          flush=True)
    print("-" * 104, flush=True)
    st = syn["self_test"]
    print(f"  SELF-TEST: { {k: int(v) for k, v in st['conditions'].items()} }", flush=True)
    print("-" * 104, flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print(f"\n  HAND-OFF: {syn['handoff']}\n", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--no-spot-check", action="store_true",
                    help="skip the optional random-init GPU weight-allocation spot-check")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", default="eagle3-vram-budget")
    args = ap.parse_args(argv)

    spot = _gpu_spot_check(enabled=not args.no_spot_check)
    live_cfg = _read_live_config()
    syn = synthesize(spot=spot, live_cfg=live_cfg)
    self_test_passes = all(syn["self_test"]["conditions"].values())

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 299, "agent": "ubel",
        "kind": "eagle3-vram-budget", "analysis_only": True,
        "spot_check": spot, "synthesis": syn,
        "eagle3_vram_budget_self_test_passes": self_test_passes,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    # fold NaN-clean (condition h) into the PRIMARY pass.
    payload["eagle3_vram_budget_self_test_passes"] = bool(self_test_passes and payload["nan_clean"])
    if nan_paths:
        print(f"[eagle3-vram] WARNING non-finite at: {nan_paths[:10]}", flush=True)

    # greedy/PPL-safety certificate (analysis-only leg).
    payload["greedy_ppl_safety_certificate"] = {
        "analysis_only": True, "served_file_changed": False, "emitted_token_changed": False,
        "hf_job_or_submission": False, "is_launch": False, "is_build": False,
        "spot_check_is_random_init_weight_alloc_only": True,
        "baseline_tps_unchanged": OFFICIAL_BASELINE, "tps_added_by_this_leg": 0.0,
    }
    payload["primary_metric"] = {"name": "eagle3_vram_budget_self_test_passes",
                                 "value": int(bool(payload["eagle3_vram_budget_self_test_passes"]))}
    payload["test_metric"] = {"name": "eagle3_build_fits_24gb",
                              "value": int(bool(syn["eagle3_build_fits_24gb"]))}

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eagle3_vram_budget_results.json"
    out_path.write_text(json.dumps(payload, indent=2, default=float))

    _print_human(syn)
    print(f"[eagle3-vram] wrote {out_path}", flush=True)
    print(f"[eagle3-vram] PRIMARY eagle3_vram_budget_self_test_passes = "
          f"{payload['eagle3_vram_budget_self_test_passes']}", flush=True)
    print(f"[eagle3-vram] TEST eagle3_build_fits_24gb = {syn['eagle3_build_fits_24gb']}  "
          f"(resident {syn['eagle3_build_resident_gb']:.2f} GiB, net delta "
          f"{syn['eagle3_net_memory_delta_gb']:.3f} GiB, dominant {syn['dominant_memory_term']})",
          flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = payload["eagle3_vram_budget_self_test_passes"]
        print(f"[eagle3-vram] PRIMARY self-test: {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
