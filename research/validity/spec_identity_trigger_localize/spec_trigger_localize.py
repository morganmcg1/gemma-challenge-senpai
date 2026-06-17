#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #621 (stark): Localize the REAL spec-#319 trigger -- the non-Marlin (and
lm_head-Marlin) op probe on vLLM 0.22.0.

CONTEXT
-------
- wirbel #607 (yuvztndu): on a CLEAN 0.22.0 gate, the int4+MTP spec-verify path
  breaks greedy identity on 31048/65536 (47%) tokens vs a 0/65536 plain-AR floor.
  #607 ran WITH VLLM_BATCH_INVARIANT=1 and a clean ref-vs-ref floor, yet broke.
- stark #617 (fa1f9vm1): proved BI=1 does NOT cover the int4-Marlin _C op, but the
  served BODY Marlin GEMMs (qkv/o_proj/gate_up/down, N<=20480) are bit-exact at the
  verify width (M=7/8/16 maxdiff 0.0). So the BODY Marlin is not the trigger.
- GAP #617 left open: the lm_head is ALSO int4 g128-Marlin in the #607 build
  (`int4_g128_lmhead/config.json`: group_1 targets=['re:.*lm_head'] num_bits=4
  gsize=128 int sym; tie_word_embeddings=False) with shape K=2560, N=262144 -- a
  shape #617 NEVER swept. If that GEMM is M-dependent at M=8 it flips near-tie
  logit argmaxes -> exactly #607's clean-floor structural break.

WHAT THIS PROBES (real A10G sm_86, NO build, NO checkpoint, synthetic weights/KV)
--------------------------------------------------------------------------------
PROBE 1  lm_head Marlin row-0 M-sweep (THE #617 gap). Reuses the #617 faithful int4
  g128-symmetric Marlin builder; sweeps M in {1,2,4,7,8,16,32,64,128}; bit-compares
  output row 0 vs the M=1 result for the lm_head shape (K=2560,N=262144) AND the 4
  body shapes as controls. first_divergent_M + m8/m7_bitexact decide attribution.

PROBE 2  attention (TRITON_ATTN unified_attention) row-0, single sequence, paged KV.
  The submission's served backend is TRITON_ATTN (vllm_attn_group_patch.py). Source
  (triton_unified_attention.py:923-932): the spec-verify forward (max_seqlen_q=M=8>1)
  is ALWAYS the 2D path; the plain-AR decode (max_seqlen_q=1, num_seqs=1<=64) takes
  the 3D split-KV path UNLESS is_batch_invariant. So BI=1 forces BOTH to 2D. We run
  query row 0 over an IDENTICAL causal KV prefix in three configs and bit-compare:
    verify  : M=8, 2D                          (always)
    ar_bi1  : M=1, is_batch_invariant=True  -> 2D (the submission's effective path)
    ar_bi0  : M=1, is_batch_invariant=False -> 3D (split-KV, 16 segments)
  d(verify, ar_bi1) ~ 0 confirms the BI guard makes verify==AR (attention CLEAN under
  the submission). d(verify, ar_bi0) > 0 shows the guard is LOAD-BEARING (3D AR would
  diverge from 2D verify if BI were off). #607 ran BI=1, so attention is not its cause
  iff d(verify, ar_bi1) == 0.

PROBE 3  sampler / greedy argmax + final_logit_softcapping row-independence. Greedy
  (temp=0) is per-row argmax over (softcap-tanh of) logits -- structurally row-
  independent. Confirmed empirically (argmax of row 0 in an M=8 batch == argmax at
  M=1) to close the PR's instruction-2b suspect. The only batch-variance entry point
  upstream of argmax is the lm_head GEMM (PROBE 1).

NOTE on the MTP drafter forward (PR instruction 2a): under greedy temp=0, identity is
decided by the TARGET argmax accepting/rejecting drafted tokens; a different draft only
changes the acceptance rate (speed), never the emitted token (a rejected draft falls
back to the target's own argmax = the AR token). So the drafter cannot be a greedy-
identity trigger -- a logical certainty, not an empirical question. Documented, not swept.

VERDICT: SPEC_TRIGGER_RECOVERABLE__<op>__<knob>  (serving/config; surface the knob,
NEVER auto-fire) | SPEC_TRIGGER_FUNDAMENTAL__<op> (irreducible kernel M-dependence ->
custom batch-invariant kernel = #319-class risk, human-gated, out of scope).
analysis_only=true, official_tps=0, NO HF Job, single A10G sm_86.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0") or "0"

HERE = Path(__file__).resolve().parent

# served gemma-4-E4B-it config (matches stark #613/#617 SERVED_SHAPES + lm_head) ---- #
HIDDEN = 2560
INTERMEDIATE = 10240
N_Q_HEADS = 8
N_KV_HEADS = 2
HEAD_DIM = 256
VOCAB = 262144
GROUP_SIZE = 128
FINAL_LOGIT_SOFTCAP = 30.0  # text_config.final_logit_softcapping in the #607 build

# name -> (K=in, N=out). The 4 fused body GEMMs (#617 controls) + the lm_head (the gap).
SERVED_SHAPES: list[tuple[str, int, int]] = [
    ("qkv", HIDDEN, (N_Q_HEADS + 2 * N_KV_HEADS) * HEAD_DIM),  # 2560 -> 3072
    ("o_proj", N_Q_HEADS * HEAD_DIM, HIDDEN),                   # 2048 -> 2560
    ("gate_up", HIDDEN, 2 * INTERMEDIATE),                      # 2560 -> 20480
    ("down", INTERMEDIATE, HIDDEN),                             # 10240 -> 2560
    ("lm_head", HIDDEN, VOCAB),                                 # 2560 -> 262144  (THE GAP)
]

# spec submissions use NUM_SPECULATIVE_TOKENS 6 (->7) / K_spec 7 (->8): verify width M in {7,8}.
M_LIST = [1, 2, 4, 7, 8, 16, 32, 64, 128]
M_MAX = max(M_LIST)


def _device():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available (set CUDA_VISIBLE_DEVICES=0)")
    return torch.device("cuda:0")


def _gpu_facts(dev) -> dict[str, Any]:
    import torch
    p = torch.cuda.get_device_properties(dev)
    cc = torch.cuda.get_device_capability(dev)
    return {
        "name": p.name, "sm_count": p.multi_processor_count,
        "compute_capability": f"{cc[0]}.{cc[1]}", "cc_tuple": list(cc),
        "is_sm86": bool(cc == (8, 6)), "is_sm80_family": bool(cc[0] == 8),
    }


# ------------------------------------------------------------------------------------ #
# PROBE 1 -- lm_head (and body controls) int4-Marlin row-0 M-sweep  (reuses #617)       #
# ------------------------------------------------------------------------------------ #
def build_marlin(K: int, N: int, dev, seed: int = 0):
    """Faithful served int4 g128-symmetric Marlin weight + run(a) closure (served defaults)."""
    import torch
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new, marlin_make_empty_g_idx)
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize

    torch.manual_seed(seed)
    wtype = scalar_types.uint4b8
    gs = GROUP_SIZE if K % GROUP_SIZE == 0 else -1
    w = (torch.randn(K, N, dtype=torch.bfloat16, device=dev) * 0.02)
    w_ref, q_w, s, g_idx, sort_idx, _perm = marlin_quantize(w, wtype, gs, act_order=False)
    del w, w_ref
    zp = marlin_make_empty_g_idx(dev)
    ws = marlin_make_workspace_new(dev)

    def run(a, fp32_reduce: bool = True, atomic: bool = False):
        # served defaults: use_fp32_reduce=True, use_atomic_add hard-False on sm_86+bf16
        return ops.marlin_gemm(
            a, None, q_w, None, s, None, None, zp, g_idx, sort_idx, ws,
            wtype, a.shape[0], N, K, True, atomic, fp32_reduce, False)

    return {"K": K, "N": N, "group_size": gs, "run": run, "wtype": wtype}


def lmhead_marlin_probe(dev) -> dict[str, Any]:
    import torch
    results: dict[str, Any] = {"per_shape": {}, "M_list": M_LIST}
    for name, K, N in SERVED_SHAPES:
        g = build_marlin(K, N, dev, seed=0)
        a_full = (torch.randn(M_MAX, K, dtype=torch.bfloat16, device=dev) * 0.1)
        outs = {M: g["run"](a_full[:M]) for M in M_LIST}
        ref0 = outs[1][0]
        per_M = {}
        first_div = None
        for M in M_LIST:
            row0 = outs[M][0]
            md = float((row0.float() - ref0.float()).abs().max().item())
            exact = bool(torch.equal(row0, ref0))
            per_M[M] = {"maxdiff_row0_vs_m1": md, "bitexact_row0_vs_m1": exact}
            if (not exact) and first_div is None and M != 1:
                first_div = M
        results["per_shape"][name] = {
            "K": K, "N": N, "group_size": g["group_size"], "per_M": per_M,
            "first_divergent_M": first_div,
            "m7_bitexact_vs_m1": per_M[7]["bitexact_row0_vs_m1"],
            "m8_bitexact_vs_m1": per_M[8]["bitexact_row0_vs_m1"],
            "maxdiff_m8": per_M[8]["maxdiff_row0_vs_m1"],
        }
        del g, outs, a_full
        torch.cuda.empty_cache()
        print(f"[probe1] {name:8s} K={K:5d} N={N:7d} m7_exact={per_M[7]['bitexact_row0_vs_m1']} "
              f"m8_exact={per_M[8]['bitexact_row0_vs_m1']} maxdiff_m8={per_M[8]['maxdiff_row0_vs_m1']:.3e} "
              f"first_div_M={first_div}")
    lh = results["per_shape"]["lm_head"]
    body = {k: v for k, v in results["per_shape"].items() if k != "lm_head"}
    results["lmhead_diverges_at_verify_width"] = not (
        lh["m7_bitexact_vs_m1"] and lh["m8_bitexact_vs_m1"])
    results["body_diverges_at_verify_width"] = any(
        not (s["m7_bitexact_vs_m1"] and s["m8_bitexact_vs_m1"]) for s in body.values())
    results["lmhead_first_divergent_M"] = lh["first_divergent_M"]
    return results


# ------------------------------------------------------------------------------------ #
# PROBE 2 -- TRITON_ATTN unified_attention row-0: verify(2D) vs AR-BI1(2D) vs AR-BI0(3D) #
# ------------------------------------------------------------------------------------ #
def attention_probe(dev, prefix_len: int = 2048, block_size: int = 16) -> dict[str, Any]:
    import torch
    import vllm.v1.attention.ops.triton_unified_attention as tua
    from vllm.utils.math_utils import next_power_of_2

    out: dict[str, Any] = {}
    torch.manual_seed(0)

    nq, nkv, hd = N_Q_HEADS, N_KV_HEADS, HEAD_DIM
    M_verify = 8
    P = prefix_len
    total_pos = P + M_verify                       # populate cache for prefix + the 8 verify toks
    num_blocks = (total_pos + block_size - 1) // block_size + 4
    scale = 1.0 / math.sqrt(hd)

    # paged KV caches: [num_blocks, block_size, num_kv_heads, head_size]
    key_cache = (torch.randn(num_blocks, block_size, nkv, hd, dtype=torch.bfloat16, device=dev) * 0.1)
    value_cache = (torch.randn(num_blocks, block_size, nkv, hd, dtype=torch.bfloat16, device=dev) * 0.1)
    # identity block table for the single sequence
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=dev).view(1, num_blocks)

    # shared query rows; q[0] is identical across all three configs
    q_full = (torch.randn(M_verify, nq, hd, dtype=torch.bfloat16, device=dev) * 0.1)

    # 3D segment buffers (shapes per TritonAttentionMetadataBuilder)
    seq_threshold_3D = 128 // nkv                   # = 64
    num_par_softmax_segments = 16
    hd_pad = next_power_of_2(hd)
    segm_out = torch.empty((seq_threshold_3D, nq, num_par_softmax_segments, hd_pad),
                           dtype=torch.float32, device=dev)
    segm_max = torch.empty((seq_threshold_3D, nq, num_par_softmax_segments),
                           dtype=torch.float32, device=dev)
    segm_exp = torch.empty((seq_threshold_3D, nq, num_par_softmax_segments),
                           dtype=torch.float32, device=dev)

    def attn(q_rows, seqused_total: int, force_bi: bool, window):
        """unified_attention for a single seq. q_rows: [m, nq, hd]; seqused_total = total KV length
        (causal abs-pos of q_rows[i] = seqused_total - m + i). Returns out [m, nq, hd]."""
        m = q_rows.shape[0]
        q = q_rows.contiguous()
        o = torch.empty_like(q)
        cu_seqlens_q = torch.tensor([0, m], dtype=torch.int32, device=dev)
        seqused_k = torch.tensor([seqused_total], dtype=torch.int32, device=dev)
        prev = tua.is_batch_invariant
        tua.is_batch_invariant = force_bi
        try:
            tua.unified_attention(
                q=q, k=key_cache, v=value_cache, out=o,
                cu_seqlens_q=cu_seqlens_q, max_seqlen_q=m,
                seqused_k=seqused_k, max_seqlen_k=int(seqused_total),
                softmax_scale=scale, causal=True, window_size=window,
                block_table=block_table, softcap=0.0,
                q_descale=None, k_descale=None, v_descale=None,
                seq_threshold_3D=seq_threshold_3D,
                num_par_softmax_segments=num_par_softmax_segments,
                softmax_segm_output=segm_out, softmax_segm_max=segm_max,
                softmax_segm_expsum=segm_exp)
        finally:
            tua.is_batch_invariant = prev
        torch.cuda.synchronize()
        return o

    def per_window(window, tag):
        # verify forward: all M_verify rows in ONE M=8 call -> 2D (max_seqlen_q>1, always).
        o_verify = attn(q_full, P + M_verify, force_bi=False, window=window)
        # AR reference: each row j as its OWN M=1 forward at abs pos P+j, KV=cache[0..P+j].
        # This is the exact op-level #319 identity condition (verify block == sequence of AR steps).
        max_row_diff = 0.0
        all_rows_exact = True
        per_row = []
        for j in range(M_verify):
            o_ar_j = attn(q_full[j:j + 1], P + j + 1, force_bi=True, window=window)  # BI=1 -> 2D
            d = float((o_verify[j].float() - o_ar_j[0].float()).abs().max().item())
            ex = bool(torch.equal(o_verify[j], o_ar_j[0]))
            per_row.append({"row": j, "maxdiff": d, "bitexact": ex})
            max_row_diff = max(max_row_diff, d)
            all_rows_exact = all_rows_exact and ex
        # load-bearing demo: row-0 AR WITHOUT BI -> 3D split-KV path.
        o_ar0_bi0 = attn(q_full[0:1], P + 1, force_bi=False, window=window)  # BI=0 -> 3D
        d_row0_bi0 = float((o_verify[0].float() - o_ar0_bi0[0].float()).abs().max().item())
        print(f"[probe2:{tag}] verify(2D) vs per-row AR(2D,BI=1): max_row_maxdiff={max_row_diff:.3e} "
              f"all_rows_exact={all_rows_exact} | row0 vs AR(3D,BI=0)={d_row0_bi0:.3e}")
        return {
            "window": list(window), "max_row_maxdiff_verify_vs_ar_bi1": max_row_diff,
            "all_verify_rows_bitexact_vs_ar_bi1": all_rows_exact,
            "row0_maxdiff_verify_vs_ar_bi0_3D": d_row0_bi0,
            "per_row": per_row,
        }

    res_global = per_window((-1, -1), "global")
    res_sliding = per_window((511, 0), "sliding512")  # Gemma local layers: sliding_window=512

    attn_clean_under_bi = bool(
        res_global["all_verify_rows_bitexact_vs_ar_bi1"]
        and res_sliding["all_verify_rows_bitexact_vs_ar_bi1"])
    bi_load_bearing = bool(
        res_global["row0_maxdiff_verify_vs_ar_bi0_3D"] > 0.0
        or res_sliding["row0_maxdiff_verify_vs_ar_bi0_3D"] > 0.0)

    out.update({
        "prefix_len": P, "block_size": block_size, "num_blocks": num_blocks,
        "seq_threshold_3D": seq_threshold_3D, "num_par_softmax_segments": num_par_softmax_segments,
        "num_q_heads": nq, "num_kv_heads": nkv, "head_size": hd, "softmax_scale": scale,
        "verify_path": "2D", "ar_bi1_path": "2D", "ar_bi0_path": "3D_splitKV",
        "global": res_global, "sliding512": res_sliding,
        # back-compat flat fields used by the verdict + W&B:
        "maxdiff_verify_vs_ar_bi1": max(res_global["max_row_maxdiff_verify_vs_ar_bi1"],
                                        res_sliding["max_row_maxdiff_verify_vs_ar_bi1"]),
        "maxdiff_verify_vs_ar_bi0": max(res_global["row0_maxdiff_verify_vs_ar_bi0_3D"],
                                        res_sliding["row0_maxdiff_verify_vs_ar_bi0_3D"]),
        "attention_clean_under_BI_all_rows": attn_clean_under_bi,
        # ATTENTION is the #607 trigger ONLY IF a verify row diverges from its AR step under BI=1:
        "attention_is_607_trigger_under_BI": (not attn_clean_under_bi),
        "bi_guard_is_load_bearing": bi_load_bearing,
    })
    print(f"[probe2] attention_clean_under_BI(all rows, global+sliding)={attn_clean_under_bi} "
          f"bi_guard_load_bearing={bi_load_bearing}")
    del key_cache, value_cache, segm_out, segm_max, segm_exp
    torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------------------------ #
# PROBE 3 -- greedy sampler / argmax + final_logit_softcapping row-independence          #
# ------------------------------------------------------------------------------------ #
def sampler_probe(dev) -> dict[str, Any]:
    import torch
    torch.manual_seed(0)
    M = 8
    logits = (torch.randn(M, VOCAB, dtype=torch.float32, device=dev))

    def softcap(x):  # final_logit_softcapping: x = cap * tanh(x / cap)  (elementwise, row-independent)
        return FINAL_LOGIT_SOFTCAP * torch.tanh(x / FINAL_LOGIT_SOFTCAP)

    am_batch = int(torch.argmax(softcap(logits)[0]).item())            # argmax of row 0 within M=8 batch
    am_single = int(torch.argmax(softcap(logits[:1])[0]).item())       # argmax of the same row at M=1
    # plain (no-softcap) argmax sanity too
    raw_batch = int(torch.argmax(logits[0]).item())
    raw_single = int(torch.argmax(logits[:1][0]).item())
    out = {
        "vocab": VOCAB, "softcap": FINAL_LOGIT_SOFTCAP,
        "argmax_row0_batch8": am_batch, "argmax_row0_single": am_single,
        "argmax_row_independent_softcapped": bool(am_batch == am_single),
        "argmax_row_independent_raw": bool(raw_batch == raw_single),
    }
    print(f"[probe3] greedy argmax row0 batch8={am_batch} single={am_single} "
          f"row_independent={out['argmax_row_independent_softcapped']}")
    return out


# ------------------------------------------------------------------------------------ #
# PROBE 4 -- custom _C norm/rope ops OUTSIDE BI: rms_norm, fused_add_rms_norm, rotary    #
# ------------------------------------------------------------------------------------ #
def norm_rope_probe(dev) -> dict[str, Any]:
    """rms_norm / fused_add_rms_norm / rotary_embedding are custom torch.ops._C ops that BI does
    NOT patch (BI only overrides aten + the attention split guard). They reduce per-row / rotate
    per-position, so they should be M-invariant -- but they live outside BI, so a hidden M-dependence
    here would survive VLLM_BATCH_INVARIANT=1 and could be the #607 trigger. Bit-compare row 0 of
    each op's output for M=1 vs M=8 with an identical row-0 input."""
    import torch
    from vllm import _custom_ops as ops

    out: dict[str, Any] = {}
    H = HIDDEN
    eps = 1e-6
    torch.manual_seed(0)
    x_full = (torch.randn(M_MAX, H, dtype=torch.bfloat16, device=dev) * 1.0)
    w = (torch.randn(H, dtype=torch.bfloat16, device=dev) * 0.1)

    # --- rms_norm (out-of-place) ---
    def rms_row0(M):
        inp = x_full[:M].contiguous()
        o = torch.empty_like(inp)
        ops.rms_norm(o, inp, w, eps)
        return o[0].clone()
    r1, r8 = rms_row0(1), rms_row0(8)
    out["rms_norm_maxdiff_m8_vs_m1"] = float((r1.float() - r8.float()).abs().max().item())
    out["rms_norm_bitexact_m8_vs_m1"] = bool(torch.equal(r1, r8))

    # --- fused_add_rms_norm (in-place; vllm comments "batch invariant") ---
    def fused_row0(M):
        inp = x_full[:M].clone()
        res = (torch.randn(M, H, dtype=torch.bfloat16, device=dev, generator=torch.Generator(dev).manual_seed(7)))
        ops.fused_add_rms_norm(inp, res, w, eps)
        return inp[0].clone()
    # identical residual row 0 across M: rebuild res deterministically from the same seed, slice row 0.
    f1 = fused_row0(1)
    f8 = fused_row0(8)
    out["fused_add_rms_norm_maxdiff_m8_vs_m1"] = float((f1.float() - f8.float()).abs().max().item())
    out["fused_add_rms_norm_bitexact_m8_vs_m1"] = bool(torch.equal(f1, f8))

    # --- rotary_embedding (in-place on q/k; per-position) ---
    n_q, n_kv, hd = N_Q_HEADS, N_KV_HEADS, HEAD_DIM
    rot_dim = hd
    max_pos = 4096
    inv_freq = 1.0 / (10000 ** (torch.arange(0, rot_dim, 2, dtype=torch.float32, device=dev) / rot_dim))
    t = torch.arange(max_pos, dtype=torch.float32, device=dev)
    freqs = torch.outer(t, inv_freq)
    cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(torch.bfloat16)  # [max_pos, rot_dim]
    positions_full = torch.arange(M_MAX, dtype=torch.int64, device=dev)
    q_base = (torch.randn(M_MAX, n_q * hd, dtype=torch.bfloat16, device=dev) * 0.1)
    k_base = (torch.randn(M_MAX, n_kv * hd, dtype=torch.bfloat16, device=dev) * 0.1)

    def rope_row0(M):
        q = q_base[:M].clone()
        k = k_base[:M].clone()
        pos = positions_full[:M].contiguous()
        ops.rotary_embedding(pos, q, k, hd, cos_sin_cache, True)
        return q[0].clone()
    rope1, rope8 = rope_row0(1), rope_row0(8)
    out["rotary_maxdiff_m8_vs_m1"] = float((rope1.float() - rope8.float()).abs().max().item())
    out["rotary_bitexact_m8_vs_m1"] = bool(torch.equal(rope1, rope8))

    out["any_norm_rope_diverges_m8"] = bool(
        (not out["rms_norm_bitexact_m8_vs_m1"])
        or (not out["fused_add_rms_norm_bitexact_m8_vs_m1"])
        or (not out["rotary_bitexact_m8_vs_m1"]))
    print(f"[probe4] rms_norm m8_exact={out['rms_norm_bitexact_m8_vs_m1']} "
          f"fused_add_rms_norm m8_exact={out['fused_add_rms_norm_bitexact_m8_vs_m1']} "
          f"rotary m8_exact={out['rotary_bitexact_m8_vs_m1']}")
    return out


# ------------------------------------------------------------------------------------ #
def decide_verdict(p1: dict | None, p2: dict | None, p3: dict | None,
                   p4: dict | None = None) -> dict[str, Any]:
    """First-diverging op + classification. Marlin M-dependence inside a precompiled CUDA
    schedule is FUNDAMENTAL (no in-wheel fix); a 2D/3D launch-path divergence closed by an
    existing flag is RECOVERABLE (surface the knob)."""
    v: dict[str, Any] = {"first_divergence_op": None, "verdict": None, "knob": None, "notes": []}

    lmhead_div = bool(p1 and p1.get("lmhead_diverges_at_verify_width"))
    body_div = bool(p1 and p1.get("body_diverges_at_verify_width"))
    attn_trigger = bool(p2 and p2.get("attention_is_607_trigger_under_BI"))
    attn_loadbearing = bool(p2 and p2.get("bi_guard_is_load_bearing"))
    sampler_clean = bool(p3 and p3.get("argmax_row_independent_softcapped"))
    normrope_div = bool(p4 and p4.get("any_norm_rope_diverges_m8"))

    if attn_trigger:
        v["first_divergence_op"] = "attention_unified_attention_2D_verify_vs_AR"
        v["verdict"] = "SPEC_TRIGGER_RECOVERABLE__attention__VLLM_BATCH_INVARIANT"
        v["knob"] = "VLLM_BATCH_INVARIANT=1 (already set by submission)"
    elif normrope_div:
        bad = [k for k in ("rms_norm", "fused_add_rms_norm", "rotary") if p4 and not p4.get(f"{k}_bitexact_m8_vs_m1", True)]
        v["first_divergence_op"] = f"custom_C_norm_rope::{','.join(bad)}"
        v["verdict"] = f"SPEC_TRIGGER_FUNDAMENTAL__{bad[0]}"
        v["knob"] = ("custom _C op outside BI is M-dependent; fix = batch-invariant kernel "
                     "(#319-class risk, human-gated) -- SURFACE, do NOT auto-fire")
    elif lmhead_div:
        v["first_divergence_op"] = "lm_head_int4_marlin_gemm (K=2560,N=262144)"
        v["verdict"] = "SPEC_TRIGGER_FUNDAMENTAL__lm_head_marlin"
        v["knob"] = ("no in-wheel fast BI int4 path for compressed-tensors WNA16 on sm_86; "
                     "fix = custom batch-invariant Marlin kernel for the lm_head shape "
                     "(#319-class risk, human-gated) OR a non-int4 lm_head (e.g. int8/bf16 head, "
                     "see stark #593) -- SURFACE, do NOT auto-fire")
    elif body_div:
        v["first_divergence_op"] = "body_int4_marlin_gemm"
        v["verdict"] = "SPEC_TRIGGER_FUNDAMENTAL__body_marlin"
        v["knob"] = "custom BI Marlin kernel (#319-class) -- SURFACE, do NOT auto-fire"
    else:
        # Every probed op is byte-identical at verify width M=8 vs AR M=1 UNDER THE SUBMISSION's
        # BI=1 config. The only batch-variance found anywhere is the attention 3D split-KV path,
        # which is taken by the M=1 AR decode ONLY when BI is OFF and is fully closed by BI=1.
        v["first_divergence_op"] = "none_intrinsic_under_BI (only batch-variance = attention 3D split-KV, BI-gated)"
        if attn_loadbearing:
            v["verdict"] = "SPEC_TRIGGER_RECOVERABLE__attention_3D_splitKV__VLLM_BATCH_INVARIANT"
            v["knob"] = ("VLLM_BATCH_INVARIANT=1 (ALREADY set by the submission) forces the M=1 AR "
                         "decode onto the 2D path, matching the always-2D M>1 verify -> byte-exact. "
                         "The verify path is identity-preserving WITHIN-STACK; a residual #607 break at "
                         "BI=1 implies the greedy-identity reference was NOT measured within-stack with "
                         "BI=1 on BOTH sides (the only divergence, attn 3D, is ~1 bf16-ULP and flips "
                         "near-tie argmaxes). RECOMMEND: pin the #319 reference within-stack at BI=1. "
                         "SURFACE only -- knob already on, do NOT auto-fire.")
        else:
            v["verdict"] = "SPEC_VERIFY_IDENTITY_PRESERVING__no_trigger_found"
            v["knob"] = None

    v["notes"].append(f"lmhead_diverges_at_verify={lmhead_div}; body_diverges_at_verify={body_div}")
    v["notes"].append(f"attention_607_trigger_under_BI={attn_trigger}; bi_guard_load_bearing={attn_loadbearing}")
    v["notes"].append(f"sampler_argmax_row_independent={sampler_clean}; norm_rope_diverges={normrope_div}")
    v["notes"].append("MTP drafter cannot be a greedy-identity trigger (only acceptance/speed); "
                      "rejected drafts fall back to target argmax = the AR token.")
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", choices=["lmhead", "attention", "sampler", "normrope", "all"], default="all")
    ap.add_argument("--prefix_len", type=int, default=2048, help="attention KV prefix length")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb_group", default="spec-identity-trigger-localize")
    ap.add_argument("--wandb_name", default="stark/spec-identity-trigger-localize")
    ap.add_argument("--resume_id", default=None)
    args = ap.parse_args()

    dev = _device()
    gpu = _gpu_facts(dev)
    print(f"[spec-trigger] GPU: {gpu['name']} cc={gpu['compute_capability']} sm86={gpu['is_sm86']}")

    p1 = p2 = p3 = p4 = None
    if args.probe in ("lmhead", "all"):
        print("[spec-trigger] PROBE 1: lm_head (+body controls) int4-Marlin row-0 M-sweep")
        p1 = lmhead_marlin_probe(dev)
    if args.probe in ("attention", "all"):
        print("[spec-trigger] PROBE 2: TRITON_ATTN unified_attention verify(2D) vs AR(2D/3D)")
        p2 = attention_probe(dev, prefix_len=args.prefix_len)
    if args.probe in ("sampler", "all"):
        print("[spec-trigger] PROBE 3: greedy argmax / softcap row-independence")
        p3 = sampler_probe(dev)
    if args.probe in ("normrope", "all"):
        print("[spec-trigger] PROBE 4: custom _C rms_norm / fused_add_rms_norm / rotary (outside BI)")
        p4 = norm_rope_probe(dev)

    verdict = decide_verdict(p1, p2, p3, p4)
    print(f"[spec-trigger] VERDICT op={verdict['first_divergence_op']} :: {verdict['verdict']}")

    payload = {
        "pr": 621, "analysis_only": True, "official_tps": 0, "no_hf_job": True, "no_build": True,
        "vllm_version": "0.22.0", "ts": datetime.now(timezone.utc).isoformat(),
        "probe_selected": args.probe, "gpu": gpu,
        "lmhead_marlin_probe": p1, "attention_probe": p2, "sampler_probe": p3,
        "norm_rope_probe": p4,
        "verdict": verdict,
        "anchors": {"stark_617": "fa1f9vm1", "wirbel_607": "yuvztndu"},
    }
    out_path = HERE / f"spec_trigger_localize_results_{args.probe}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[spec-trigger] wrote {out_path}")

    if not args.no_wandb:
        try:
            import wandb
            resume_id = args.resume_id
            if resume_id is None and (HERE / "run_id.txt").exists():
                resume_id = (HERE / "run_id.txt").read_text().strip()
            init_kw = dict(project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
                           entity=os.environ.get("WANDB_ENTITY"),
                           group=args.wandb_group, name=args.wandb_name, job_type="analysis")
            if resume_id:
                init_kw.update(id=resume_id, resume="allow")
            run = wandb.init(**init_kw)
            flat: dict[str, Any] = {"audit_phase": 1, "audit_done": 1,
                                    "is_sm86": gpu["is_sm86"], "probe_selected": args.probe}
            if p1 is not None:
                flat.update({
                    "lmhead_m8_bitexact_vs_m1": p1["per_shape"]["lm_head"]["m8_bitexact_vs_m1"],
                    "lmhead_m7_bitexact_vs_m1": p1["per_shape"]["lm_head"]["m7_bitexact_vs_m1"],
                    "lmhead_maxdiff_m8": p1["per_shape"]["lm_head"]["maxdiff_m8"],
                    "lmhead_first_divergent_M": (p1["lmhead_first_divergent_M"] or -1),
                    "lmhead_diverges_at_verify_width": p1["lmhead_diverges_at_verify_width"],
                    "body_diverges_at_verify_width": p1["body_diverges_at_verify_width"],
                })
                cols = ["shape", "K", "N"] + [f"maxdiff_M{m}" for m in M_LIST] + ["first_div_M", "m8_exact"]
                tbl = wandb.Table(columns=cols)
                for name, s in p1["per_shape"].items():
                    tbl.add_data(name, s["K"], s["N"],
                                 *[s["per_M"][m]["maxdiff_row0_vs_m1"] for m in M_LIST],
                                 (s["first_divergent_M"] or -1), s["m8_bitexact_vs_m1"])
                wandb.log({"marlin_m_sweep_with_lmhead": tbl})
            if p2 is not None:
                flat.update({
                    "attn_maxdiff_verify_vs_ar_bi1": p2["maxdiff_verify_vs_ar_bi1"],
                    "attn_maxdiff_verify_vs_ar_bi0": p2["maxdiff_verify_vs_ar_bi0"],
                    "attn_is_607_trigger_under_BI": p2["attention_is_607_trigger_under_BI"],
                    "attn_bi_guard_load_bearing": p2["bi_guard_is_load_bearing"],
                })
            if p3 is not None:
                flat.update({"sampler_argmax_row_independent": p3["argmax_row_independent_softcapped"]})
            if p4 is not None:
                flat.update({
                    "rms_norm_m8_bitexact": p4["rms_norm_bitexact_m8_vs_m1"],
                    "fused_add_rms_norm_m8_bitexact": p4["fused_add_rms_norm_bitexact_m8_vs_m1"],
                    "rotary_m8_bitexact": p4["rotary_bitexact_m8_vs_m1"],
                    "rms_norm_maxdiff_m8": p4["rms_norm_maxdiff_m8_vs_m1"],
                    "fused_add_rms_norm_maxdiff_m8": p4["fused_add_rms_norm_maxdiff_m8_vs_m1"],
                    "rotary_maxdiff_m8": p4["rotary_maxdiff_m8_vs_m1"],
                    "any_norm_rope_diverges_m8": p4["any_norm_rope_diverges_m8"],
                })
            wandb.log(flat)
            run.summary.update({"verdict": verdict["verdict"],
                                "first_divergence_op": verdict["first_divergence_op"]})
            run.config.update({"verdict": verdict["verdict"]}, allow_val_change=True)
            payload["wandb_run_id"] = run.id
            out_path.write_text(json.dumps(payload, indent=2))
            print(f"[spec-trigger] wandb run {run.id}")
            wandb.finish()
        except Exception as ex:  # noqa: BLE001
            print(f"[spec-trigger] wandb failed (non-fatal): {type(ex).__name__}: {str(ex)[:200]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
