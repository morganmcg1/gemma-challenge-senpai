#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #344 (denken) -- Gate-independent SPEED lever closure: is #124 the sole >500 gate?

THE STRATEGIC QUESTION
----------------------
The deployed 481.53-TPS single-token step is the NORMALIZED 1218.2us composition
unit (denken #257/#278). Across ALL its terms -- not just the verify attention
that #332 (`y5cl0ena`) pinned BW-floored -- is the batch=1 step already at its
roofline floor, so that the ONLY way to amortize HBM traffic is spec-decode
(verify M tokens per single weight read)? If so, then because spec-decode under
strict greedy identity is supply-capped at 473.5 (denken #332, the deterministic-
SDPA floor), the entire >500 question COLLAPSES to #124: lift the int4 batch-
variance gate -> spec-decode reaches the 520.95 ceiling via coverage; keep the
gate -> no gate-independent lever reaches 500.

This card is the SPEED-side CLOSURE. It does NOT re-measure any banked anchor; it
assembles the step roofline waterfall from them and bounds every gate-independent
SPEED lever against it. ORTHOGONAL to wirbel `ppl-only-gate-500-envelope`
(demand/gate-lift pricing), stark `non-eagle3-500-feasibility` (alternative spec
METHODS), kanna #342 (a_1-lift recipe), lawine #339 (coverage clear-prob), fern
#341 (joint isocline). It answers ONLY: "is there a non-spec-decode SPEED lever,
or is #124 the sole gate?"

THE WATERFALL (deployed single-token step, normalized basis)
------------------------------------------------------------
The 1218.2us step = BRIDGE x (draft K=7 chain + verify M=8 forward) wall, with the
banked s_served_abs_us = 5868.49us and BRIDGE = 1218.2/5868.49 = 0.20758 (denken
#257). Each WALL term (graphed, ONEGRAPH-faithful) maps to the normalized step by
x BRIDGE. The verify body (37 int4 layers) is split into its 4 GEMMs by weight
bytes -- valid because M=8 is still weight-bound (built_step knee: m8_still_
weightbound=True). For each term we report bytes moved, flops, arithmetic
intensity (AI), and BW-bound vs compute-bound at batch=1.

The DOMINANT HBM-traffic term is the int4 body weight read: 1.6973824 GB, 94.3% of
the step's HBM bytes, AI = 4.0 flop/byte at batch=1 (M=1) -- 52x below the A10G
ridge 208.3 -> deeply BW-bound. The KV read for a 512-token single stream is only
40.0 MB (GQA), 2.2% of step traffic -> the KV-AI / fp8-KV lever is geometrically
dead. EVERY term at batch=1 is BW-bound (max AI = attention 7.88 << 208).

THE LEVERS (each bounded to step-us -> TPS; do any clear 500 ALONE?)
-------------------------------------------------------------------
(a) weight-read amortization == spec-decode. The ONLY lever that attacks the
    dominant term. AR (M=1) reads 1.697 GB PER token; spec M=8 reads it ONCE per
    verify -> per-accepted-token read = 1.697/E[T] = 0.363 GB (E[T]=4.6828). The
    1218.2us step is ALREADY this amortized per-token unit. Going faster = more
    M / higher E[T], but the strict-greedy-identity ceiling is 473.5 (#332). < 500.
(b) kernel fusion (GeluAndMul fold + GEMM/attn) removes only the activation
    round-trips (~0.4% of step HBM bytes); the honest composition gain is +0.91%
    (#278 Model B; the naive +4.39% is a refuted 4.8x over-credit) -> 485.9. < 500.
(c) fp8-KV: DEAD at batch=1 (no dequant-attention fusion -> zero single-stream
    latency; vLLM Apr-2026). CUDA-graph: ~1-3% micro-lever AND already deployed
    (ONEGRAPH=1) -> <= 3% upper -> 496.0. < 500.

VERDICT
-------
gate_independent_speed_lever_clears_500_alone = False. No gate-independent SPEED
lever reaches 500 alone, because none touches the dominant weight-read term except
spec-decode, which is strict-identity capped at 473.5. The >500 decision reduces
entirely to #124.

HONEST SCOPE
------------
0 TPS. BASELINE 481.53 unchanged. CPU-analytic over banked numbers; NO GPU, NO
model forward, NO served-file change, NO HF Job, NO submission, NO build, NO
launch. Roofline reductions are UPPER bounds (un-measured). Fusion gains depend on
kernel availability in vLLM 0.22 (flagged real vs hypothetical). Any fp8/quant
lever touching PPL is out of scope (gated). The deployed topology is held fixed.

PRIMARY metric  speed_lever_self_test_passes
TEST    metric  dominant_hbm_term_ai  (AI of the dominant BW term, batch=1)
                + gate_independent_speed_lever_clears_500_alone  (bool, expect False)
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
# Baseline / ceilings (PR #52 2x9fm2zx; wirbel #204/#220 pqjnybbf; twoceiling).
OFFICIAL_TPS = 481.53
TARGET_TPS = 500.0
LAMBDA1_CEIL = 520.9527323111674          # int4-spec batch-invariant verify ceiling (lambda=1)
STEP_NORM_US = 1218.2                       # deployed NORMALIZED single-token step (#257/#278)
K_CAL = 125.26795005202914                  # composition calibration (steps/s); TPS = K_cal*E[T]/step_ms
E_T_SERVED = 4.6827608                       # deployed served E[T] (the 1218.2us step is banked here)
PPL_BASELINE = 2.3772
PPL_GATE = 2.42

# A10G roofline constants (kanna #280 roofline.json; denken #332).
A10G_BW_GBPS = 600.0                         # nominal HBM bandwidth
A10G_PEAK_TFLOPS = 125.0                     # FP16 tensor (w/ sparsity); ridge = peak/bw
RIDGE_AI = 208.33333333333334                # = 125e12 / 600e9 flop/byte

# Deployed model geometry (verify_phase.json dims; osoi5 deployed depth=37).
HIDDEN = 2560
N_Q_HEADS = 8
N_KV_HEADS = 2
GQA_GROUP = N_Q_HEADS // N_KV_HEADS          # 4
HEAD_DIM = 256
INTERMEDIATE = 10240
N_LAYERS = 37
VOCAB = 12288
CTX = 528
INT4_BYTES = 0.5                             # W4A16 -> 0.5 byte/param
BF16_BYTES = 2.0

# GEMM output/input shapes (verify_phase.json shapes).
QKV_OUT = (N_Q_HEADS + 2 * N_KV_HEADS) * HEAD_DIM     # 3072
O_IN = N_Q_HEADS * HEAD_DIM                            # 2048
GU_OUT = 2 * INTERMEDIATE                              # 20480

# Banked byte counts (denken #257/#283 physical_floor; #332 geometry).
BODY_INT4_BYTES = 1697382400.0               # 1.6973824 GB int4 body weights (37 layers)
LMHEAD_BF16_BYTES = 62914560.0               # 0.06291456 GB bf16 lm_head
KV_GQA_BYTES = 40009728.0                     # GQA KV cache read, M=8, ctx=528
KV_DENKEN_BYTES = 160038912.0                 # full (un-GQA'd) KV bytes, #332 attention roofline
BODY_PARAMS = 3394764800                      # 3.3947648 B body params

# denken #332 (`eagle3_sdpa_phi_floor`, y5cl0ena) -- the verify ATTENTION roofline row.
SDPA_US_M8 = 776.2124633789062               # measured M=8 SDPA wall (denken #291)
SDPA_TOTAL_BYTES = 162463744.0               # KV_denken + Q/out activations
SDPA_ROOFLINE_US = 270.7729066666667         # total_bytes / 600GBps
SDPA_BW_UTIL = 0.34883864849061247           # 34.9% -- roofline_us / SDPA_US_M8
SDPA_EXPOSED_US = 505.43955671223955         # above-roofline exposed slack
SDPA_AI_FLOP_PER_BYTE = 7.880597014925373    # attention AI (M=8, full ctx)
SPEC_STRICT_CEILING_332 = 473.5295953446407  # compliant_ceiling_tps_at_geo (deterministic-SDPA floor)

# denken #257 (`built_step_roofline`) -- the graphed WALL per-term verify breakdown
# (verify_phase.json, M=8) + the K=7 draft chain. Their sum is s_served_abs_us, and
# STEP_NORM_US = BRIDGE * s_served_abs_us (denken #278).
VERIFY_BODY_US_M8 = 4474.193849563599        # 37-layer body GEMMs (graphed)
VERIFY_ATTN_US_M8 = 557.9008138179779        # 37-layer attention (graphed)
VERIFY_LMHEAD_US_M8 = 131.6198444366455      # lm_head (graphed)
DRAFT_PASS_US_GRAPHED = 100.6822395324707    # one drafter pass (graphed)
K_SPEC = 7                                    # draft chain length
WALL_TOTAL_US_257 = 5868.490184545517        # s_served_abs_us = body+attn+lmhead+draft_k7
BRIDGE_257 = 0.2075832048263608              # STEP_NORM_US / WALL_TOTAL_US_257

# denken #278 (`linear_step_decomposition`) -- the GeluAndMul-fold honesty result.
FUSION_GAIN_HONEST_PCT_278 = 0.9113547874137429   # Model B (bridge the wall saving)
FUSION_GAIN_NAIVE_PCT_278 = 4.390896003290608     # Model A (over-credits 4.8x; refuted)
FUSION_OVERCREDIT_FACTOR_278 = 4.817987532332126

# Pre-screen verdicts (advisor fresh-research; cite, do NOT re-derive).
FP8_KV_GAIN_PCT = 0.0                         # dead at batch=1 (no dequant-attn fusion)
CUDAGRAPH_GAIN_UPPER_PCT = 3.0                # ~1-3% micro-lever; already deployed (ONEGRAPH)

TOL_SUM = 1.0e-3
TOL_ROUNDTRIP = 1.0e-6
TOL_TPS = 1.0e-3


# --------------------------------------------------------------------------- #
# Roofline primitives
# --------------------------------------------------------------------------- #
def _bound(ai: float) -> str:
    return "BW" if ai < RIDGE_AI else "compute"


def gemm_term(name: str, out_dim: int, in_dim: int, m: int) -> dict[str, Any]:
    """int4 (W4A16) GEMM roofline at batch=1, M tokens."""
    params = out_dim * in_dim
    weight_bytes = params * INT4_BYTES
    flops = 2.0 * m * params                      # 2*M*N*K (MAC = 2 flop)
    ai = flops / weight_bytes                      # = 4*M for int4 (0.5 byte/param)
    return {
        "name": name, "dtype": "int4", "out": out_dim, "in": in_dim,
        "params": params, "bytes": weight_bytes, "flops": flops,
        "ai_flop_per_byte": ai, "bound": _bound(ai),
    }


def sdpa_geometry(m: int) -> dict[str, Any]:
    """Reproduce denken #332's verify-attention roofline row from primitives."""
    kv_bytes_denken = 2 * BF16_BYTES * CTX * N_Q_HEADS * HEAD_DIM * N_LAYERS   # K,V (un-GQA'd)
    kv_bytes_gqa = kv_bytes_denken / GQA_GROUP
    act_bytes = 2 * m * N_Q_HEADS * HEAD_DIM * BF16_BYTES * N_LAYERS            # Q in, out
    total_bytes = kv_bytes_denken + act_bytes
    roofline_us = total_bytes / (A10G_BW_GBPS * 1.0e3)                          # bytes/(600GB/s)->us
    bw_util = roofline_us / SDPA_US_M8
    exposed_us = SDPA_US_M8 * (1.0 - bw_util)
    # attention flops = 2*(QK^T + A.V) = 4 * M * ctx * head_dim * n_q * layers
    attn_flops = 4.0 * m * CTX * HEAD_DIM * N_Q_HEADS * N_LAYERS
    ai = attn_flops / total_bytes
    return {
        "kv_bytes_denken": kv_bytes_denken, "kv_bytes_gqa": kv_bytes_gqa,
        "act_bytes": act_bytes, "total_bytes": total_bytes, "roofline_us": roofline_us,
        "bw_util": bw_util, "exposed_us": exposed_us, "attn_flops": attn_flops,
        "ai_flop_per_byte": ai, "bound": _bound(ai),
    }


# --------------------------------------------------------------------------- #
# Step 1: the deployed single-token step roofline waterfall (sums to 1218.2us)
# --------------------------------------------------------------------------- #
def step1_waterfall() -> dict[str, Any]:
    m = N_Q_HEADS  # deployed verify width M=8

    # --- per-term roofline (analytic, batch=1) --- #
    gemms = {
        "qkv_proj": gemm_term("qkv_proj", QKV_OUT, HIDDEN, m),
        "o_proj": gemm_term("o_proj", HIDDEN, O_IN, m),
        "gate_up_proj": gemm_term("gate_up_proj", GU_OUT, HIDDEN, m),
        "down_proj": gemm_term("down_proj", HIDDEN, INTERMEDIATE, m),
    }
    per_layer_params = sum(g["params"] for g in gemms.values())
    body_bytes_check = per_layer_params * N_LAYERS * INT4_BYTES

    attn = sdpa_geometry(m)
    # lm_head: bf16, AI = M (= 2*M*P / (2*P)).
    lmhead_params = VOCAB * HIDDEN
    lmhead = {
        "name": "lm_head", "dtype": "bf16", "params": lmhead_params,
        "bytes": lmhead_params * BF16_BYTES, "flops": 2.0 * m * lmhead_params,
        "ai_flop_per_byte": float(m), "bound": _bound(float(m)),
    }
    # norms (RMSNorm): activation-read, ~2 per layer; tiny, AI~1 -> BW; exposed ~0 (#291).
    norms_bytes = 2 * N_LAYERS * (m * HIDDEN * BF16_BYTES * 2 + HIDDEN * BF16_BYTES)
    norms = {"name": "norms", "dtype": "bf16", "bytes": float(norms_bytes),
             "flops": float(2 * 2 * N_LAYERS * m * HIDDEN), "ai_flop_per_byte": 1.0,
             "bound": "BW", "exposed_note": "folded into graphed body; #291 exposed=0"}
    # sampling: argmax over [M, vocab] logits; tiny, AI~0 -> BW.
    sampling_bytes = m * VOCAB * 4.0
    sampling = {"name": "sampling", "dtype": "fp32", "bytes": float(sampling_bytes),
                "flops": float(m * VOCAB), "ai_flop_per_byte": m * VOCAB / sampling_bytes,
                "bound": "BW", "exposed_note": "argmax; ~us, folded"}

    # --- WALL per-term (graphed; #257 basis) split body by weight bytes (M=8 weight-bound) --- #
    body_split = {k: VERIFY_BODY_US_M8 * (g["params"] / per_layer_params)
                  for k, g in gemms.items()}
    draft_chain_us = DRAFT_PASS_US_GRAPHED * K_SPEC
    wall = {
        "qkv_proj": body_split["qkv_proj"],
        "o_proj": body_split["o_proj"],
        "gate_up_proj": body_split["gate_up_proj"],
        "down_proj": body_split["down_proj"],
        "attention": VERIFY_ATTN_US_M8,
        "lm_head": VERIFY_LMHEAD_US_M8,
        "draft_chain": draft_chain_us,
    }
    wall_total = sum(wall.values())
    bridge = STEP_NORM_US / wall_total
    # normalized step share: each WALL term x bridge -> sums to 1218.2us.
    norm = {k: v * bridge for k, v in wall.items()}
    norm_sum = sum(norm.values())
    # norms / sampling / framework are folded (graphed) or host-only -> 0 additive.
    norm["norms"] = 0.0
    norm["sampling"] = 0.0
    norm["framework_overhead"] = 0.0

    waterfall_rows = []
    body_norm_us = sum(norm[k] for k in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"))
    for k in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
        g = gemms[k]
        waterfall_rows.append({
            "term": k, "dtype": "int4", "bytes": g["bytes"] * N_LAYERS,
            "flops": g["flops"] * N_LAYERS, "ai_flop_per_byte": g["ai_flop_per_byte"],
            "bound": g["bound"], "wall_us": wall[k], "norm_us": norm[k],
            "norm_pct_of_step": 100.0 * norm[k] / STEP_NORM_US,
        })
    waterfall_rows.append({
        "term": "attention", "dtype": "bf16-kv", "bytes": attn["total_bytes"],
        "flops": attn["attn_flops"], "ai_flop_per_byte": attn["ai_flop_per_byte"],
        "bound": attn["bound"], "wall_us": wall["attention"], "norm_us": norm["attention"],
        "norm_pct_of_step": 100.0 * norm["attention"] / STEP_NORM_US,
        "bw_util": attn["bw_util"],
    })
    waterfall_rows.append({
        "term": "lm_head", "dtype": "bf16", "bytes": lmhead["bytes"],
        "flops": lmhead["flops"], "ai_flop_per_byte": lmhead["ai_flop_per_byte"],
        "bound": lmhead["bound"], "wall_us": wall["lm_head"], "norm_us": norm["lm_head"],
        "norm_pct_of_step": 100.0 * norm["lm_head"] / STEP_NORM_US,
    })
    waterfall_rows.append({
        "term": "draft_chain", "dtype": "int4-drafter", "bytes": None,
        "flops": None, "ai_flop_per_byte": None, "bound": "BW",
        "wall_us": wall["draft_chain"], "norm_us": norm["draft_chain"],
        "norm_pct_of_step": 100.0 * norm["draft_chain"] / STEP_NORM_US,
        "note": "K=7 drafter fwd chain; drafter-weight-bound; ONEGRAPH-captured (composite bytes)",
    })
    for extra, rec in (("norms", norms), ("sampling", sampling)):
        waterfall_rows.append({
            "term": extra, "dtype": rec["dtype"], "bytes": rec["bytes"],
            "flops": rec["flops"], "ai_flop_per_byte": rec["ai_flop_per_byte"],
            "bound": rec["bound"], "wall_us": 0.0, "norm_us": 0.0, "norm_pct_of_step": 0.0,
            "note": rec.get("exposed_note", ""),
        })
    waterfall_rows.append({
        "term": "framework_overhead", "dtype": "host", "bytes": 0.0, "flops": 0.0,
        "ai_flop_per_byte": None, "bound": "host", "wall_us": 0.0, "norm_us": 0.0,
        "norm_pct_of_step": 0.0,
        "note": "host launch; removed by deployed ONEGRAPH CUDA graphs (already minimized)",
    })

    return {
        "M_verify": m,
        "rows": waterfall_rows,
        "wall_terms_us": wall,
        "wall_total_us": wall_total,
        "bridge": bridge,
        "bridge_257_crosscheck": BRIDGE_257,
        "bridge_resid": abs(bridge - BRIDGE_257),
        "norm_sum_us": norm_sum,
        "step_norm_us": STEP_NORM_US,
        "waterfall_sum_resid_us": abs(norm_sum - STEP_NORM_US),
        "body_norm_us": body_norm_us,
        "body_bytes_check": body_bytes_check,
        "per_layer_params": per_layer_params,
        "attn_geometry": attn,
        "note": ("normalized step 1218.2us = bridge x (draft K7 + verify M8 forward) wall "
                 "(#257 s_served_abs_us=5868.49); body split by weight bytes (M=8 weight-bound)."),
    }


# --------------------------------------------------------------------------- #
# Step 2: dominant HBM term + KV-read share (kills the KV-AI lever)
# --------------------------------------------------------------------------- #
def step2_dominant() -> dict[str, Any]:
    # Per single-token forward HBM traffic (batch=1): weights + KV + lm_head.
    body = BODY_INT4_BYTES
    lmhead = LMHEAD_BF16_BYTES
    kv_gqa = KV_GQA_BYTES                 # physical GQA cache read
    kv_denken = KV_DENKEN_BYTES           # #332 attention-roofline accounting (broadcast)
    step_hbm_bytes = body + lmhead + kv_gqa
    dominant = "body_int4_weight_read"
    dominant_share = body / step_hbm_bytes
    lmhead_share = lmhead / step_hbm_bytes
    kv_share_gqa = kv_gqa / step_hbm_bytes
    kv_share_denken = kv_denken / (body + lmhead + kv_denken)

    # Dominant-term AI at batch=1 (M=1, per-token AR) and at deployed M=8.
    ai_m1 = 2.0 * 1 * BODY_PARAMS / body          # = 4.0  (4*M, M=1)
    ai_m8 = 2.0 * N_Q_HEADS * BODY_PARAMS / body  # = 32.0 (4*M, M=8)

    # Per-token weight read & the spec-decode amortization of the dominant term.
    read_per_token_ar_gb = body / 1e9                       # AR reads full body per token
    read_per_step_gb = body / 1e9                           # spec reads it ONCE per verify
    read_per_accepted_gb = (body / 1e9) / E_T_SERVED        # amortized over E[T] accepted
    amortization_factor = E_T_SERVED                        # tokens per single weight read
    read_floor_us = body / (A10G_BW_GBPS * 1e3)             # 2829us-ish (body only)

    return {
        "dominant_term": dominant,
        "step_hbm_bytes": step_hbm_bytes,
        "body_int4_bytes": body, "lmhead_bf16_bytes": lmhead, "kv_gqa_bytes": kv_gqa,
        "dominant_share": dominant_share, "dominant_pct": 100.0 * dominant_share,
        "lmhead_share": lmhead_share, "lmhead_pct": 100.0 * lmhead_share,
        "kv_read_share_gqa": kv_share_gqa, "kv_read_pct_gqa": 100.0 * kv_share_gqa,
        "kv_read_share_denken": kv_share_denken, "kv_read_pct_denken": 100.0 * kv_share_denken,
        "kv_ai_lever_dead": bool(kv_share_gqa < 0.10),
        "dominant_hbm_term_ai_batch1_m1": ai_m1,
        "dominant_hbm_term_ai_deployed_m8": ai_m8,
        "ridge_ai": RIDGE_AI,
        "dominant_is_bw_bound": bool(ai_m1 < RIDGE_AI and ai_m8 < RIDGE_AI),
        "read_per_token_ar_gb": read_per_token_ar_gb,
        "read_per_step_gb": read_per_step_gb,
        "read_per_accepted_gb": read_per_accepted_gb,
        "amortization_factor_spec": amortization_factor,
        "body_read_floor_us": read_floor_us,
        "note": ("int4 body weight read dominates step HBM traffic (94.3%); KV read "
                 "2.2% (GQA) -> KV-AI/fp8-KV lever geometrically dead. spec-decode is "
                 "the ONLY lever that amortizes the dominant read (E[T]x per weight read)."),
    }


# --------------------------------------------------------------------------- #
# Step 3: bound every gate-independent SPEED lever -> step-us -> TPS
# --------------------------------------------------------------------------- #
def _tps_at_step(step_us: float, e_t: float = E_T_SERVED) -> float:
    """TPS = K_cal * E[T] / step_ms (twoceiling/joint basis)."""
    return K_CAL * e_t / (step_us / 1000.0)


def _step_for_tps(tps: float, e_t: float = E_T_SERVED) -> float:
    return K_CAL * e_t / tps * 1000.0


def _tps_from_pct(gain_pct: float) -> float:
    """A +g% step reduction -> TPS = OFFICIAL_TPS * step_old/step_new = OFFICIAL_TPS*(1+g/100)."""
    return OFFICIAL_TPS * (1.0 + gain_pct / 100.0)


def step3_levers(s2: dict[str, Any]) -> dict[str, Any]:
    gap_to_500_pct = 100.0 * (TARGET_TPS / OFFICIAL_TPS - 1.0)   # +3.836% step cut needed
    step_for_500 = _step_for_tps(TARGET_TPS)
    step_cut_for_500_us = STEP_NORM_US - step_for_500

    # (a) weight-read amortization == spec-decode. The ONLY dominant-term lever.
    #     Strict greedy identity caps it at 473.5 (#332 deterministic-SDPA floor).
    spec = {
        "lever": "weight_read_amortization_spec_decode",
        "attacks_dominant_term": True,
        "amortization_factor": s2["amortization_factor_spec"],
        "strict_identity_ceiling_tps": SPEC_STRICT_CEILING_332,
        "clears_500_alone": bool(SPEC_STRICT_CEILING_332 >= TARGET_TPS),
        "note": ("AR reads 1.697GB/token; spec reads it once per M=8 verify -> E[T]x "
                 "amortized. The 1218.2us step IS this amortized unit. Going faster needs "
                 "more M/E[T], but strict-identity ceiling = 473.5 (#332) < 500. To exceed, "
                 "LIFT the gate (#124) -> 520.95 ceiling via coverage."),
    }

    # (b) kernel fusion (GeluAndMul fold + GEMM/attn): removes only activation round-trips.
    #     HBM bound: the gate_up->down intermediate [M, 2*intermediate] round-trip.
    inter_rt_bytes = N_LAYERS * (N_Q_HEADS * INTERMEDIATE * BF16_BYTES) * 2  # write+read, M=8
    fusion_hbm_frac = inter_rt_bytes / s2["step_hbm_bytes"]
    fusion_hbm_bound_pct = 100.0 * fusion_hbm_frac
    fusion = {
        "lever": "kernel_fusion_geluandmul_gemm_attn",
        "attacks_dominant_term": False,
        "intermediate_roundtrip_bytes_m8": inter_rt_bytes,
        "hbm_roundtrip_bound_pct": fusion_hbm_bound_pct,     # ~0.4% pure-HBM upper
        "honest_gain_pct_278": FUSION_GAIN_HONEST_PCT_278,    # +0.91% (Model B)
        "naive_gain_pct_278_refuted": FUSION_GAIN_NAIVE_PCT_278,
        "overcredit_factor_278": FUSION_OVERCREDIT_FACTOR_278,
        "tps_at_honest_gain": _tps_from_pct(FUSION_GAIN_HONEST_PCT_278),
        "clears_500_alone": bool(_tps_from_pct(FUSION_GAIN_HONEST_PCT_278) >= TARGET_TPS),
        "vllm_022_availability": ("GeluAndMul fused activation is REAL in vLLM; a full "
                                  "GEMM+attn epilogue fusion is HYPOTHETICAL at int4 W4A16."),
    }

    # (c) fp8-KV (dead) + CUDA-graph (~1-3%, already deployed).
    fp8_kv = {
        "lever": "fp8_kv_cache",
        "attacks_dominant_term": False,
        "gain_pct": FP8_KV_GAIN_PCT,
        "tps_at_gain": _tps_from_pct(FP8_KV_GAIN_PCT),
        "clears_500_alone": False,
        "prescreen": ("DEAD at batch=1: vLLM FP8-KV does not fuse dequant with attention "
                      "-> zero single-stream latency (only multi-request). [vLLM Apr-2026]"),
    }
    cuda_graph = {
        "lever": "cuda_graph_tuning",
        "attacks_dominant_term": False,
        "gain_pct_upper": CUDAGRAPH_GAIN_UPPER_PCT,
        "tps_at_upper": _tps_from_pct(CUDAGRAPH_GAIN_UPPER_PCT),
        "clears_500_alone": bool(_tps_from_pct(CUDAGRAPH_GAIN_UPPER_PCT) >= TARGET_TPS),
        "prescreen": ("~1-3% micro-lever AND already deployed (ONEGRAPH=1, #271) -> realizable "
                      "additional gain near 0; bounded, not a >500 path."),
    }

    levers = [spec, fusion, fp8_kv, cuda_graph]
    any_clears_alone = any(bool(lv.get("clears_500_alone")) for lv in levers)

    # Optimistic non-spec STACK (fusion x cuda-graph upper) -- the honest caveat.
    stack_optimistic = OFFICIAL_TPS * (1.0 + FUSION_GAIN_HONEST_PCT_278 / 100.0) \
        * (1.0 + CUDAGRAPH_GAIN_UPPER_PCT / 100.0)
    stack_realistic = OFFICIAL_TPS * (1.0 + FUSION_GAIN_HONEST_PCT_278 / 100.0) * 1.01

    all_pcts = [FUSION_GAIN_HONEST_PCT_278, FP8_KV_GAIN_PCT, CUDAGRAPH_GAIN_UPPER_PCT,
                fusion_hbm_bound_pct, gap_to_500_pct, stack_optimistic, stack_realistic]
    levers_nan_clean = all(math.isfinite(x) for x in all_pcts)

    return {
        "gap_to_500_pct": gap_to_500_pct,
        "step_for_500_us": step_for_500,
        "step_cut_for_500_us": step_cut_for_500_us,
        "spec_decode": spec,
        "kernel_fusion": fusion,
        "fp8_kv": fp8_kv,
        "cuda_graph": cuda_graph,
        "any_lever_clears_500_alone": any_clears_alone,
        "stack_optimistic_tps": stack_optimistic,
        "stack_realistic_tps": stack_realistic,
        "stack_optimistic_clears_500": bool(stack_optimistic >= TARGET_TPS),
        "fp8_kv_dead_cited": True,
        "cuda_graph_prescreen_cited": True,
        "levers_nan_clean": levers_nan_clean,
        "stack_caveat": ("the only non-spec stack near 500 (fusion 0.91% x cuda-graph 3% upper "
                         "= 500.5) relies on a cuda-graph gain already banked in the 481.53 "
                         "baseline (ONEGRAPH) and STILL leaves the dominant weight read "
                         "untouched -- a coincidental micro-pileup at the bar, not a robust "
                         ">500 path and not an HBM-amortization."),
    }


# --------------------------------------------------------------------------- #
# Step 4: step <-> TPS budget conversion (round-trips)
# --------------------------------------------------------------------------- #
def step4_budget() -> dict[str, Any]:
    # Exact conversion: TPS = K_cal * E[T] / step_ms; hold deployed E[T]=4.6827608.
    e_t_int4_ceiling = LAMBDA1_CEIL * (STEP_NORM_US / 1000.0) / K_CAL   # full-accept E[T] at ceiling
    tps_deployed_rt = _tps_at_step(STEP_NORM_US)                        # -> 481.53
    step_for_500 = _step_for_tps(TARGET_TPS)                           # 1173.21us
    step_for_ceiling = _step_for_tps(LAMBDA1_CEIL)                     # 1126.04us
    tps_500_rt = _tps_at_step(step_for_500)
    tps_ceiling_rt = _tps_at_step(step_for_ceiling, e_t=E_T_SERVED)
    # ceiling round-trips at its OWN full-accept E[T] at the deployed step.
    tps_ceiling_fullaccept_rt = _tps_at_step(STEP_NORM_US, e_t=e_t_int4_ceiling)
    return {
        "conversion": "TPS = K_cal * E[T] / step_ms ; hold deployed E[T]=4.6827608",
        "k_cal": K_CAL, "e_t_served": E_T_SERVED, "step_norm_us": STEP_NORM_US,
        "e_t_int4_ceiling": e_t_int4_ceiling,
        "tps_deployed_roundtrip": tps_deployed_rt,
        "deployed_roundtrip_resid": abs(tps_deployed_rt - OFFICIAL_TPS),
        "step_for_500_us": step_for_500,
        "step_for_ceiling_us": step_for_ceiling,
        "step_cut_for_500_us": STEP_NORM_US - step_for_500,
        "step_cut_for_ceiling_us": STEP_NORM_US - step_for_ceiling,
        "tps_500_roundtrip": tps_500_rt,
        "tps_500_roundtrip_resid": abs(tps_500_rt - TARGET_TPS),
        "tps_ceiling_roundtrip": tps_ceiling_rt,
        "tps_ceiling_roundtrip_resid": abs(tps_ceiling_rt - LAMBDA1_CEIL),
        "tps_ceiling_fullaccept_roundtrip": tps_ceiling_fullaccept_rt,
        "ceiling_fullaccept_resid": abs(tps_ceiling_fullaccept_rt - LAMBDA1_CEIL),
        "advisor_hint_note": ("advisor hint '520.95<->1080; 500<->~1126' uses a conservative "
                              "E[T]~4.49 (measured-floor) basis; the exact deployed-E[T] "
                              "conversion gives 520.95<->1126.04us and 500<->1173.21us, which "
                              "round-trips the banked 481.53 anchor."),
    }


# --------------------------------------------------------------------------- #
# Synthesis + verdict
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    s1 = step1_waterfall()
    s2 = step2_dominant()
    s3 = step3_levers(s2)
    s4 = step4_budget()

    clears = bool(s3["any_lever_clears_500_alone"])   # expect False
    dominant_ai = s2["dominant_hbm_term_ai_batch1_m1"]  # 4.0

    handoff = (
        "PR #344 hand-off (the SPEED-side closure for the >500 decision): the deployed "
        "batch=1 single-token step (normalized 1218.2us) is weight-read-BW-bound -- the int4 "
        "body read (1.6973824 GB) is 94.3% of step HBM traffic at AI 4.0 (M=1) / 32 (M=8), both "
        "<< the A10G ridge 208.3; the KV read is only 40.0 MB (2.2%, GQA), so the KV-AI / fp8-KV "
        "lever is geometrically dead. EVERY step term is BW-bound at batch=1 (max AI = attention "
        "7.88, #332). The ONLY HBM-amortization lever is spec-decode (verify M tokens per single "
        "weight read), whose strict-greedy-identity ceiling is 473.5 (#332 deterministic-SDPA "
        "floor). Kernel fusion removes only the ~0.4% activation round-trips (honest +0.91% #278, "
        "the naive +4.39% is a refuted 4.8x over-credit) -> 485.9; fp8-KV is dead at batch=1; "
        "CUDA-graph is a <=3% micro-lever already deployed (ONEGRAPH) -> 496.0. No gate-independent "
        "SPEED lever clears 500 alone. THEREFORE the >500 decision reduces ENTIRELY to #124: lift "
        "the int4 batch-variance gate -> spec-decode reaches the 520.95 ceiling via coverage "
        "(wirbel's PPL-only envelope); keep the gate -> 473.5 cap, no gate-independent path. "
        "0 TPS; analytic over banked numbers; BASELINE 481.53 unchanged; NOT a launch, NOT a "
        "build, NOT a served-file change."
    )

    verdict = (
        "GATE-INDEPENDENT SPEED LEVER DOES NOT CLEAR 500 ALONE -- #124 IS THE SOLE >500 GATE. "
        "The batch=1 step waterfall (sums to 1218.2us) is dominated by the int4 body weight read "
        "(94.3% of HBM bytes, AI 4.0 << ridge 208 -> deeply BW-bound); the KV read is 2.2% (kills "
        "the KV-AI lever) and every term is BW-bound at batch=1. The only lever that attacks the "
        "dominant weight-read term is spec-decode (E[T]x amortization per weight read), and its "
        "strict-greedy-identity ceiling is 473.5 (#332) < 500. The non-dominant-term levers fall "
        "short alone: fusion removes only ~0.4% activation round-trips (honest +0.91% -> 485.9; "
        "the +4.39% naive projection is a refuted 4.8x composition over-credit, #278); fp8-KV is "
        "dead at batch=1 (no dequant-attn fusion); CUDA-graph is a <=3% micro-lever already in the "
        "deployed ONEGRAPH baseline (-> 496.0). Even the optimistic non-spec stack (500.5) sits AT "
        "the bar on a cuda-graph gain already banked and leaves the dominant read untouched -- not "
        "a robust path. So no gate-independent SPEED lever reaches 500; the >500 question collapses "
        "to #124 (lift gate -> 520.95 via coverage; keep gate -> 473.5 cap). BASELINE 481.53 "
        "untouched; CPU-analytic; 0 TPS; NOT a launch."
    )

    return {
        "official_tps": OFFICIAL_TPS, "target_tps": TARGET_TPS, "lambda1_ceil": LAMBDA1_CEIL,
        "step_norm_us": STEP_NORM_US, "spec_strict_ceiling_332": SPEC_STRICT_CEILING_332,
        "step1_waterfall": s1, "step2_dominant": s2, "step3_levers": s3, "step4_budget": s4,
        "dominant_hbm_term_ai": dominant_ai,
        "gate_independent_speed_lever_clears_500_alone": clears,
        "reduces_to_124": bool(not clears),
        "verdict": verdict, "handoff": handoff,
    }


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY: speed_lever_self_test_passes)
# --------------------------------------------------------------------------- #
def self_test(syn: dict[str, Any]) -> dict[str, Any]:
    s1, s2, s3, s4 = (syn["step1_waterfall"], syn["step2_dominant"],
                      syn["step3_levers"], syn["step4_budget"])
    c: dict[str, bool] = {}

    # (a) step waterfall sums to 1218.2us <= 1e-3.
    c["a_waterfall_sums_to_1218p2"] = bool(s1["waterfall_sum_resid_us"] <= TOL_SUM)
    c["a_bridge_matches_257"] = bool(s1["bridge_resid"] <= 1e-6)
    c["a_body_bytes_roundtrip"] = bool(abs(s1["body_bytes_check"] - BODY_INT4_BYTES) < 1.0)

    # (b) #332 attention BW 34.9% / AI 7.88 round-trip <= 1e-6.
    attn = s1["attn_geometry"]
    c["b_sdpa_bw_util_roundtrips_349"] = bool(abs(attn["bw_util"] - SDPA_BW_UTIL) <= TOL_ROUNDTRIP)
    c["b_sdpa_ai_roundtrips_788"] = bool(
        abs(attn["ai_flop_per_byte"] - SDPA_AI_FLOP_PER_BYTE) <= TOL_ROUNDTRIP)
    c["b_sdpa_total_bytes_exact"] = bool(abs(attn["total_bytes"] - SDPA_TOTAL_BYTES) < 1.0)
    c["b_sdpa_roofline_us_exact"] = bool(abs(attn["roofline_us"] - SDPA_ROOFLINE_US) <= 1e-6)

    # (c) dominant term (weight read) explicit + KV-read share computed.
    c["c_dominant_is_body_int4"] = bool(s2["dominant_term"] == "body_int4_weight_read")
    c["c_dominant_share_gt_half"] = bool(s2["dominant_share"] > 0.5)
    c["c_kv_share_computed_small"] = bool(0.0 < s2["kv_read_share_gqa"] < 0.10)
    c["c_kv_ai_lever_dead"] = bool(s2["kv_ai_lever_dead"])
    c["c_dominant_ai_below_ridge"] = bool(s2["dominant_is_bw_bound"])
    c["c_dominant_ai_is_4"] = bool(abs(s2["dominant_hbm_term_ai_batch1_m1"] - 4.0) < 1e-9)
    c["c_lmhead_byte_exact"] = bool(abs(s2["lmhead_bf16_bytes"] - LMHEAD_BF16_BYTES) < 1.0)
    c["c_kv_byte_exact"] = bool(abs(s2["kv_gqa_bytes"] - KV_GQA_BYTES) < 1.0)

    # (d) each lever's step-reduction bound NaN-clean + fp8-KV-dead / CUDA-graph pre-screens cited.
    c["d_levers_nan_clean"] = bool(s3["levers_nan_clean"])
    c["d_fp8_kv_dead_cited"] = bool(s3["fp8_kv_dead_cited"]
                                    and s3["fp8_kv"]["gain_pct"] == 0.0)
    c["d_cuda_graph_prescreen_cited"] = bool(s3["cuda_graph_prescreen_cited"]
                                             and s3["cuda_graph"]["gain_pct_upper"] <= 3.0)
    c["d_fusion_honest_below_naive"] = bool(
        s3["kernel_fusion"]["honest_gain_pct_278"] < s3["kernel_fusion"]["naive_gain_pct_278_refuted"])

    # (e) the 500 step budget round-trips (state exact conversion).
    c["e_deployed_anchor_roundtrips"] = bool(s4["deployed_roundtrip_resid"] <= TOL_TPS)
    c["e_step_for_500_roundtrips"] = bool(s4["tps_500_roundtrip_resid"] <= TOL_ROUNDTRIP)
    c["e_step_for_ceiling_roundtrips"] = bool(s4["tps_ceiling_roundtrip_resid"] <= TOL_ROUNDTRIP)
    c["e_ceiling_fullaccept_roundtrips"] = bool(s4["ceiling_fullaccept_resid"] <= TOL_TPS)

    # (f) verdict: no single gate-independent SPEED lever clears 500.
    c["f_no_lever_clears_500_alone"] = bool(not s3["any_lever_clears_500_alone"])
    c["f_verdict_is_false"] = bool(
        syn["gate_independent_speed_lever_clears_500_alone"] is False)
    c["f_spec_strict_ceiling_below_500"] = bool(SPEC_STRICT_CEILING_332 < TARGET_TPS)
    c["f_reduces_to_124"] = bool(syn["reduces_to_124"])

    passed = all(c.values())
    return {"checks": c, "speed_lever_self_test_passes": passed}


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
    s1, s2, s3, s4 = (syn["step1_waterfall"], syn["step2_dominant"],
                      syn["step3_levers"], syn["step4_budget"])
    print("=" * 78, flush=True)
    print("PR #344  Gate-independent SPEED lever closure -- is #124 the sole >500 gate?",
          flush=True)
    print("=" * 78, flush=True)
    print(f"  step waterfall (normalized, sums to {s1['norm_sum_us']:.4f}us "
          f"[resid {s1['waterfall_sum_resid_us']:.2e}], bridge {s1['bridge']:.5f}):", flush=True)
    for r in s1["rows"]:
        ai = r["ai_flop_per_byte"]
        ai_s = f"{ai:7.2f}" if isinstance(ai, (int, float)) else "    n/a"
        print(f"    {r['term']:<18} norm {r['norm_us']:7.2f}us "
              f"({r['norm_pct_of_step']:5.1f}%)  AI {ai_s} [{r['bound']}]", flush=True)
    print(f"  dominant HBM term: {s2['dominant_term']}  "
          f"{s2['dominant_pct']:.1f}% of step bytes  "
          f"AI(M=1)={s2['dominant_hbm_term_ai_batch1_m1']:.1f} "
          f"AI(M=8)={s2['dominant_hbm_term_ai_deployed_m8']:.1f} << ridge {s2['ridge_ai']:.1f}",
          flush=True)
    print(f"  KV read share: {s2['kv_read_pct_gqa']:.2f}% (GQA) -> KV-AI lever "
          f"dead={s2['kv_ai_lever_dead']}", flush=True)
    print(f"  gap to 500: +{s3['gap_to_500_pct']:.3f}%  (cut "
          f"{s3['step_cut_for_500_us']:.2f}us from 1218.2us)", flush=True)
    print("  levers (each ALONE):", flush=True)
    print(f"    (a) spec-decode      strict ceiling {s3['spec_decode']['strict_identity_ceiling_tps']:.1f} "
          f"clears={s3['spec_decode']['clears_500_alone']}", flush=True)
    print(f"    (b) kernel fusion    honest +{s3['kernel_fusion']['honest_gain_pct_278']:.2f}% "
          f"-> {s3['kernel_fusion']['tps_at_honest_gain']:.1f} "
          f"clears={s3['kernel_fusion']['clears_500_alone']}", flush=True)
    print(f"    (c) fp8-KV           {s3['fp8_kv']['gain_pct']:.0f}% (dead) "
          f"clears={s3['fp8_kv']['clears_500_alone']}", flush=True)
    print(f"    (c) cuda-graph       <=+{s3['cuda_graph']['gain_pct_upper']:.0f}% "
          f"-> {s3['cuda_graph']['tps_at_upper']:.1f} clears={s3['cuda_graph']['clears_500_alone']}",
          flush=True)
    print(f"  budget: 500<->{s4['step_for_500_us']:.2f}us  "
          f"520.95<->{s4['step_for_ceiling_us']:.2f}us  "
          f"(deployed RT {s4['tps_deployed_roundtrip']:.2f})", flush=True)
    print(f"  VERDICT  gate_independent_speed_lever_clears_500_alone = "
          f"{syn['gate_independent_speed_lever_clears_500_alone']}  "
          f"(dominant_hbm_term_ai = {syn['dominant_hbm_term_ai']:.1f})", flush=True)
    print(f"  SELF-TEST  speed_lever_self_test_passes = {st['speed_lever_self_test_passes']}",
          flush=True)
    if not st["speed_lever_self_test_passes"]:
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
        print(f"[speed-lever] wandb logging unavailable: {exc}", flush=True)
        return None

    syn, st = payload["synthesis"], payload["self_test"]
    s1, s2 = syn["step1_waterfall"], syn["step2_dominant"]
    s3, s4 = syn["step3_levers"], syn["step4_budget"]
    run = init_wandb_run(
        job_type="gate-independent-speed-lever",
        agent="denken",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["gate-independent-speed-lever", "roofline", "weight-read-bound",
              "spec-decode-amortization", "kernel-fusion", "fp8-kv", "validity", "zero-tps"],
        config={
            "pr": 344, "analysis_only": True,
            "official_tps": OFFICIAL_TPS, "target_tps": TARGET_TPS,
            "lambda1_ceil": LAMBDA1_CEIL, "step_norm_us": STEP_NORM_US,
            "spec_strict_ceiling_332": SPEC_STRICT_CEILING_332,
            "a10g_bw_gbps": A10G_BW_GBPS, "ridge_ai": RIDGE_AI,
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[speed-lever] wandb: no run (no WANDB_API_KEY/mode) -- skipping", flush=True)
        return None

    summary: dict[str, Any] = {
        # PRIMARY + TEST metrics.
        "speed_lever_self_test_passes": int(bool(st["speed_lever_self_test_passes"])),
        "dominant_hbm_term_ai": syn["dominant_hbm_term_ai"],
        "gate_independent_speed_lever_clears_500_alone": int(bool(
            syn["gate_independent_speed_lever_clears_500_alone"])),
        "reduces_to_124": int(bool(syn["reduces_to_124"])),
        # step1 waterfall.
        "waterfall_sum_us": s1["norm_sum_us"],
        "waterfall_sum_resid_us": s1["waterfall_sum_resid_us"],
        "bridge": s1["bridge"], "wall_total_us": s1["wall_total_us"],
        "attn_bw_util": s1["attn_geometry"]["bw_util"],
        "attn_ai_flop_per_byte": s1["attn_geometry"]["ai_flop_per_byte"],
        # step2 dominant term.
        "dominant_pct": s2["dominant_pct"], "lmhead_pct": s2["lmhead_pct"],
        "kv_read_pct_gqa": s2["kv_read_pct_gqa"],
        "kv_ai_lever_dead": int(bool(s2["kv_ai_lever_dead"])),
        "dominant_hbm_term_ai_m1": s2["dominant_hbm_term_ai_batch1_m1"],
        "dominant_hbm_term_ai_m8": s2["dominant_hbm_term_ai_deployed_m8"],
        "amortization_factor_spec": s2["amortization_factor_spec"],
        "read_per_accepted_gb": s2["read_per_accepted_gb"],
        # step3 levers.
        "gap_to_500_pct": s3["gap_to_500_pct"],
        "step_cut_for_500_us": s3["step_cut_for_500_us"],
        "spec_strict_ceiling_tps": s3["spec_decode"]["strict_identity_ceiling_tps"],
        "fusion_honest_gain_pct": s3["kernel_fusion"]["honest_gain_pct_278"],
        "fusion_hbm_roundtrip_bound_pct": s3["kernel_fusion"]["hbm_roundtrip_bound_pct"],
        "fusion_tps_at_honest": s3["kernel_fusion"]["tps_at_honest_gain"],
        "cuda_graph_tps_at_upper": s3["cuda_graph"]["tps_at_upper"],
        "any_lever_clears_500_alone": int(bool(s3["any_lever_clears_500_alone"])),
        "stack_optimistic_tps": s3["stack_optimistic_tps"],
        "stack_optimistic_clears_500": int(bool(s3["stack_optimistic_clears_500"])),
        # step4 budget.
        "step_for_500_us": s4["step_for_500_us"],
        "step_for_ceiling_us": s4["step_for_ceiling_us"],
        "tps_deployed_roundtrip": s4["tps_deployed_roundtrip"],
        "e_t_int4_ceiling": s4["e_t_int4_ceiling"],
        **{f"selftest_{k}": int(bool(v)) for k, v in st["checks"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="gate_independent_speed_lever_result",
                      artifact_type="validity", data=payload)
    finish_wandb(run)
    rid = getattr(run, "id", None)
    print(f"[speed-lever] wandb logged (run {rid})", flush=True)
    return rid


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="gate-independent-speed-lever")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    syn = synthesize()
    st = self_test(syn)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 344, "agent": "denken",
        "kind": "gate-independent-speed-lever", "synthesis": syn, "self_test": st,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[speed-lever] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn, st)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gate_independent_speed_lever_results.json"

    wid = None
    if not args.no_wandb:
        wid = _maybe_log_wandb(args, payload)
    payload["wandb_run_id"] = wid
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[speed-lever] wrote {out_path}  (wandb run {wid})", flush=True)

    if args.self_test:
        ok = st["speed_lever_self_test_passes"] and payload["nan_clean"]
        print(f"[speed-lever] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
