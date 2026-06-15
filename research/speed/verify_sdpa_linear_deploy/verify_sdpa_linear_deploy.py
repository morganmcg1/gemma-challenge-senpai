#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Verify SDPA linear-deploy: does wirbel #270's num_stages=3->2 bit-identical
TRITON_ATTN tune (1.097x at M=8 *2D tree-verify*) transfer to the DEPLOYED
linear-path verify SDPA? (PR #279, wirbel). LOCAL GPU micro-profiling + CPU
analytic. Analysis-only: NO served-file change, NO HF Job, NO submission, NOT a
launch. BASELINE stays 481.53.

THE QUESTION (and the premise correction this leg makes)
--------------------------------------------------------
#270 (iwwcmvez, MERGED) found the deployed TRITON_ATTN kernel_unified_attention
is a bare @triton.jit (triton-default num_warps=4 / num_stages=3, NOT autotuned)
and that forcing num_stages=2 nets ~+9.7% bit-identical (maxdiff=0.0) at M=8.
BUT #270 measured that on the **2D** path (use_3d = M==1, so M=8 -> 2D, which
selects TILE_SIZE_PREFILL=32). PR #279 asks: does that win transfer to the
**deployed linear-path verify** SDPA?

Reading the served submission (fa2sw_precache_kenyan) settles the shape, and it
is NOT the #270 2D M=8 shape:

  * MAX_NUM_SEQS=1 -> the deployment serves a SINGLE sequence, not 8 concurrent
    requests. The "batch=8" (lawine #246) is the K+1 = 8 query rows of the
    linear-chain verify (SPECULATIVE_CONFIG num_speculative_tokens=7), for ONE
    sequence -- not 8 batch sequences. (PR #279's premise said "8 concurrent
    sequences"; the real source of M=8 is the chain length.)
  * SPLITKV_VERIFY=1 (splitkv_verify_patch.py) REDIRECTS the M=8 verify
    attention to the 3D split-KV (FlashDecoding) path by faking max_seqlen_q->1.
    So the deployed verify SDPA runs IS_3D=True with tile_size=TILE_SIZE_DECODE
    (line 965 of triton_unified_attention.py), and for Gemma-4
    (sliding_window=512 != 1024 -> NOT _is_gemma3_attention) TILE_SIZE_DECODE=16.
    #270's 2D M=8 used TILE_SIZE_PREFILL=32. => DIFFERENT kernel specialization.

So `linear_sdpa_kernel_matches_tree_shape` is the central instruction-1 boolean:
same kernel entry point + same num_stages=3 default (TRUE), but a DIFFERENT
launch configuration (3D split-KV TILE=16 vs 2D TILE=32). We therefore RE-MEASURE
the num_stages tune at the ACTUAL deployed 3D split-KV M=8 shape.

WHICH TARGET LAYERS THE VERIFY TUNE TOUCHES (gemma-4-E4B, osoi5-v0-baked)
------------------------------------------------------------------------
37 target layers: 30 sliding_attention (head_dim=256) + 7 full_attention
(global_head_dim=512). fa_sliding_patch flips the 16 head-256 sliding layers in
idx 0..18 (non-KV-shared, not share-sources 19/20) to FA2 (NOT tunable). The
remaining 21 attention layers keep TRITON_ATTN and ARE tunable by num_stages:
  * 7 full head-512 layers (idx 2,8,14,20,26,32,36);
  * 14 sliding head-256 layers (idx 19 + the 13 KV-shared sliding in 21..35).
All 21 run M=8 -> 3D split-KV (M=8 <= SPLITKV_VERIFY_MAX_Q=64). 8 q heads, 2 kv
heads, BLOCK_M=16, BLOCK_Q=4. The verify-SDPA tune saving = sum over these 21.

PRICING (composition official = K_cal*E[T]*tau/step; tau=STEP_US so the deployed
point reproduces 481.53 = K_cal*E[T]). The verify SDPA tune is a STEP reduction
(verify forward shrinks), NOT a draft-pass cut -> NO fern #274 phi-discount; but
it is a fraction of the step:
  verify_sdpa_saving_us = total_verify_sdpa_deployed_us - total_verify_sdpa_tuned_us
  new_step_us           = STEP_US - verify_sdpa_saving_us
  projected_tps_gain_pct= (STEP_US/new_step_us - 1)*100   (E[T] unchanged: bit-ident)
  honest_projected_tps_after = 481.53 * STEP_US/new_step_us
Standalone per-call replay OVER-states the in-graph SDPA (no graph overlap) -> the
projected gain is an UPPER BOUND; the realistic gain is <=. Cross-reference fern
#274 honest static-K=4 = 493.96 (the draft-cut ceiling this verify lever must beat
to be "the one post-phi lever above 500").

SELF-TEST (`verify_sdpa_linear_deploy_self_test_passes`, PRIMARY)
----------------------------------------------------------------
(a) clone fidelity: launch_tuned at the deployed config == the REAL
    unified_attention at every shape (3D & 2D) -> we time the deployed kernel;
(b) shape match reported with evidence: would_redirect(M=8)=True (verify IS 3D
    split-KV) AND attn_is_shape_specialized==False AND the deployed 3D tile==16;
(c) both kernel-level and full-verify-pass speedups measured via CUDA events;
(d) 128-draw bit-identity gate on the deployed 3D verify shapes
    (linear_sdpa_tune_greedy_identical reported, divergent count reported);
(e) composition round-trips: K_cal*E[T] == 481.53 (within tol);
(f) BASELINE 481.53 / 520.95-private-Δ / K_cal 125.268 / step 1218.2 / E[T] 3.844
    imported EXACTLY and UNCHANGED;
(g) NaN-clean; (h) speedup sane (>= 1-eps; deployed in grid);
(i) LYNCHPIN: the REAL wrapper (triton-default warps/stages) wall-clock matches
    our forced-(w4,s3) deployed baseline at every shape -> default IS s3.
TEST metrics: `projected_tps_gain_pct` (float, 0.0 if NULL) and
`linear_sdpa_tune_greedy_identical` (bool).

Requires the deployed senpai vLLM wheel venv (vllm 0.22.1 + triton 3.6).
No serve change, no HF Job, no submission. NOT a launch. NOT open2.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0") or "0"

import torch  # noqa: E402

assert torch.cuda.is_available(), "CUDA unavailable (set CUDA_VISIBLE_DEVICES=0)"
torch.zeros(1, device="cuda")
torch.cuda.synchronize()

from vllm.triton_utils import triton  # noqa: E402
from vllm.v1.attention.ops.triton_unified_attention import (  # noqa: E402
    kernel_unified_attention,
    reduce_segments,
    unified_attention,
    _get_tile_size,
    _is_gemma3_attention,
)

# --- import the DEPLOYED splitkv-verify predicate (proves the M=8 routing) -----
_SUB = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                    "submissions", "fa2sw_precache_kenyan")
sys.path.insert(0, os.path.abspath(_SUB))
try:
    from splitkv_verify_patch import would_redirect as _would_redirect  # noqa: E402
    from splitkv_verify_patch import (SPLITKV_VERIFY_MAX_Q  # noqa: E402
                                      as _SPLITKV_MAX_Q)
    _HAVE_SPLITKV = True
except Exception as _exc:  # noqa: BLE001
    print(f"[verify-sdpa] WARN: could not import splitkv_verify_patch: {_exc!r}",
          flush=True)
    _HAVE_SPLITKV = False
    _SPLITKV_MAX_Q = 64

    def _would_redirect(**kw):  # type: ignore
        return None

# ---- IMPORTED, UNCHANGED (this leg moves nothing) ----------------------------
FRONTIER_TPS = 481.53      # PR #52 official a10g-small frontier (BASELINE)
PRIVATE_TPS = 460.85       # private-verified reference (Δ 4.3% <= 5%; PPL 2.3777)
K_CAL = 125.268            # composition calibration (kanna #217 vgovdrjc)
STEP_US = 1218.2           # served decode step (kanna #217 / #260)
ET_DEPLOYED = 3.844        # accepted tok/step (kanna #217 vgovdrjc)
K_DEPLOYED = 7             # num_speculative_tokens (manifest SPECULATIVE_CONFIG)
PPL_PINNED = 2.3772        # PR #52 official PPL (bit-identical => unchanged)
FERN274_HONEST_K4_TPS = 493.96   # fern #274 brnmnl60: honest phi-corrected static-K=4
                                 # (the draft-cut ceiling this verify lever must beat)
VERIFY_FORWARD_US = 511.0  # verify-forward anchor (~42% of 1218.2us step; PR #279
                           # prose / denken #278). Used ONLY for the full-verify-
                           # pass speedup% diagnostic; the STEP saving is computed
                           # directly from the measured SDPA times, anchor-free.

# Triton defaults (confirmed bare-jit, no autotune; lynchpin re-validated here)
DEPLOYED_NUM_WARPS = 4
DEPLOYED_NUM_STAGES = 3
NUM_PAR_SOFTMAX_SEGMENTS = 16
MIN_LAUNCH_GRID_SIZE_2D = 128

# Target verify TRITON_ATTN layer composition (config.json osoi5-v0-baked +
# fa_sliding_patch idx logic; derived & checked in PR #279 task-1):
N_VERIFY_GLOBAL_H512 = 7    # full_attention head-512 layers (all TRITON_ATTN)
N_VERIFY_SLIDING_H256 = 14  # KV-shared/source sliding head-256 (TRITON_ATTN)
N_VERIFY_FA2_H256 = 16      # sliding head-256 flipped to FA2 (NOT tunable)

# Realistic decode context for HONEST pricing. ctx=2048 GROSSLY over-states: <1%
# of real decode steps reach it. Measured on the 128 served speed-benchmark
# prompts (official/main_bucket/shared_resources/speed_benchmark/data/
# eval_prompts_sharegpt.json) tokenized with the deployed osoi5-v0-baked
# tokenizer: prompt tokens median=221 / mean=264 / p90=373 / max=2419. The
# benchmark generates OUTPUT_LEN=512 tok/prompt (decode_outputs.py), so the
# TIME-AVERAGED decode context over a generation is prompt + OUT/2 = median 477 /
# mean 520; end-of-generation median 733 / mean 776. Only 0.8% of decode steps
# exceed ctx 2048, 3.9% exceed 1024. => price the realistic average (512), with
# 768 (~mean end-of-gen) and 2048 (loose UB) as the band. At ctx=2048 the
# standalone SDPA sum exceeds the 511us verify forward (physically impossible
# in-graph) -> a clear over-statement flagged by standalone_share_exceeds_verify.
PRICE_CTX_REALISTIC = 512        # time-averaged decode ctx (prompt~221 + OUT/2~256)
PRICE_CTX_BAND = (512, 768, 2048)  # realistic / near-worst-typical / loose UB

GRID_WARPS = [2, 4, 8]
GRID_STAGES = [1, 2, 3, 4]
GRID_TILE_BITIDENT = [16]        # deployed 3D decode tile; bit-identical headline
GRID_TILE_FPTOL = [32, 64]       # TILE change = re-tiles softmax (fp-tol partition)
FP_TOL = 1e-2


# --------------------------------------------------------------------------- #
# Realistic paged-KV input construction (mirrors triton_attn.py forward).      #
# --------------------------------------------------------------------------- #
def make_inputs(M, head_size, num_heads, num_kv_heads, context_len, sliding,
                block_size=16, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    seq_len = context_len + M
    num_blocks = (seq_len + block_size - 1) // block_size + 1
    q = torch.randn(M, num_heads, head_size, device=dev, dtype=torch.bfloat16) * 0.1
    out = torch.empty(M, num_heads, head_size, device=dev, dtype=torch.bfloat16)
    kv = torch.randn(num_blocks, 2, block_size, num_kv_heads, head_size,
                     device=dev, dtype=torch.bfloat16) * 0.1
    k_cache, v_cache = kv.unbind(1)
    block_table = torch.arange(num_blocks, device=dev, dtype=torch.int32).view(1, -1)
    cu_seqlens_q = torch.tensor([0, M], device=dev, dtype=torch.int32)
    seqused_k = torch.tensor([seq_len], device=dev, dtype=torch.int32)
    window = (sliding - 1, sliding - 1) if sliding else (-1, -1)
    return dict(q=q, k=k_cache, v=v_cache, out=out, cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=M, seqused_k=seqused_k, max_seqlen_k=seq_len,
                softmax_scale=head_size ** -0.5, causal=True, window_size=window,
                block_table=block_table, softcap=0.0,
                q_descale=None, k_descale=None, v_descale=None)


def make_segm(num_tokens, num_heads, head_size, num_kv_heads):
    seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // num_kv_heads
    n = max(num_tokens, seq_threshold_3D)
    hp = triton.next_power_of_2(head_size)
    return dict(
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=NUM_PAR_SOFTMAX_SEGMENTS,
        softmax_segm_output=torch.empty((n, num_heads, NUM_PAR_SOFTMAX_SEGMENTS, hp),
                                        dtype=torch.float32, device="cuda"),
        softmax_segm_max=torch.empty((n, num_heads, NUM_PAR_SOFTMAX_SEGMENTS),
                                     dtype=torch.float32, device="cuda"),
        softmax_segm_expsum=torch.empty((n, num_heads, NUM_PAR_SOFTMAX_SEGMENTS),
                                        dtype=torch.float32, device="cuda"),
    )


def call_deployed_wrapper(inp, segm, use_3d, force_redirect=False):
    """The REAL deployed unified_attention. For the verify path the deployed
    serve redirects via SPLITKV_VERIFY (max_seqlen_q->1) to select 3D split-KV
    while leaving M>1 query rows intact; force_redirect reproduces that."""
    kw = dict(inp)
    kw.update(segm)
    if force_redirect:
        kw["max_seqlen_q"] = 1  # the splitkv redirect: pick 3D, rows from cu_seqlens
    elif not use_3d:
        # 2D stock path: strip the segm tensors so the wrapper's use_3d gate is False
        for kk in ("softmax_segm_output", "softmax_segm_max", "softmax_segm_expsum"):
            kw.pop(kk, None)
    unified_attention(**kw)
    return inp["out"]


def launch_tuned(inp, segm, tile_size, num_warps, num_stages, block_m, use_3d):
    """Faithful clone of unified_attention's launch with overridable knobs,
    calling the EXACT deployed kernel_unified_attention."""
    q, k, v, out = inp["q"], inp["k"], inp["v"], inp["out"]
    cu_seqlens_q, seqused_k = inp["cu_seqlens_q"], inp["seqused_k"]
    block_table = inp["block_table"]; window_size = inp["window_size"]
    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]; num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]
    BLOCK_M = block_m
    BLOCK_Q = max(1, BLOCK_M // num_queries_per_kv)
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
    head_size_padded = triton.next_power_of_2(head_size)
    if use_3d:
        segm_output_ptr = segm["softmax_segm_output"]
        segm_max_ptr = segm["softmax_segm_max"]
        segm_expsum_ptr = segm["softmax_segm_expsum"]
        num_segments = segm["num_par_softmax_segments"]
        grid = (total_num_q_blocks, num_kv_heads, num_segments)
    else:
        segm_output_ptr = segm_max_ptr = segm_expsum_ptr = out
        num_segments = 1
        grid = (total_num_q_blocks, num_kv_heads)
    kernel_unified_attention[grid](
        output_ptr=out, segm_output_ptr=segm_output_ptr, segm_max_ptr=segm_max_ptr,
        segm_expsum_ptr=segm_expsum_ptr, query_ptr=q, key_cache_ptr=k,
        value_cache_ptr=v, sink_ptr=None, block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k, alibi_slopes_ptr=None, qq_bias_ptr=None,
        k_scale_cache_ptr=k, v_scale_cache_ptr=v, scale=inp["softmax_scale"],
        q_scale=None, k_scale=None, v_scale=None, out_scale=1.0, softcap=0.0,
        num_query_heads=num_query_heads, num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0), query_stride_0=q.stride(0),
        query_stride_1=q.stride(1), output_stride_0=out.stride(0),
        output_stride_1=out.stride(1), qq_bias_stride_0=0, BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size, HEAD_SIZE=head_size, HEAD_SIZE_PADDED=head_size_padded,
        USE_ALIBI_SLOPES=False, USE_ALIBI_SQRT=False, USE_QQ_BIAS=False,
        USE_SOFTCAP=False, USE_SINKS=False, USE_MM_PREFIX=False, MAX_MM_RANGES=0,
        mm_prefix_range_ptr=None, SLIDING_WINDOW=(1 + window_size[0]),
        stride_k_cache_0=k.stride(0), stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2), stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0), stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2), stride_v_cache_3=v.stride(3),
        stride_ks_blk=0, stride_ks_slot=0, stride_ks_head=0,
        stride_vs_blk=0, stride_vs_slot=0, stride_vs_head=0,
        query_start_len_ptr=cu_seqlens_q, BLOCK_Q=BLOCK_Q, num_seqs=num_seqs,
        BLOCK_M=BLOCK_M, NUM_SEGMENTS_PER_SEQ=num_segments, USE_FP8=False,
        IS_3D=use_3d, KV_QUANT_MODE=0, CHUNK_LOOKBACK=-1, CHUNK_SIZE=-1,
        USE_TD=False, USE_TD_QO=False, num_warps=num_warps, num_stages=num_stages,
    )
    if use_3d:
        reduce_segments[(q.shape[0], num_query_heads)](
            output_ptr=out, segm_output_ptr=segm["softmax_segm_output"],
            segm_max_ptr=segm["softmax_segm_max"],
            segm_expsum_ptr=segm["softmax_segm_expsum"], seq_lens_ptr=seqused_k,
            num_seqs=num_seqs, num_query_heads=num_query_heads, out_scale_inv=1.0,
            output_stride_0=out.stride(0), output_stride_1=out.stride(1),
            block_table_stride=block_table.stride(0), TILE_SIZE=tile_size,
            HEAD_SIZE=head_size, HEAD_SIZE_PADDED=head_size_padded,
            query_start_len_ptr=cu_seqlens_q, BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=segm["num_par_softmax_segments"], USE_FP8=False,
        )
    return out


def deployed_tile(head_size, sliding, use_3d):
    """The tile_size the deployed wrapper passes: 3D path -> TILE_SIZE_DECODE
    (line 965), 2D path -> TILE_SIZE_PREFILL (line 962)."""
    sw = (sliding) if sliding else 0
    return _get_tile_size(head_size, sw, 2, is_prefill=(not use_3d))


# --------------------------------------------------------------------------- #
# CUDA-graph replay timing (served ONEGRAPH basis; launch overhead amortized). #
# --------------------------------------------------------------------------- #
def graph_time(run, iters, warmup):
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(5):
                run()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(g):
            run()
        for _ in range(max(10, warmup)):
            g.replay()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        ms = e0.elapsed_time(e1) / iters
        del g
        return ms * 1e3, True  # us
    except Exception:  # noqa: BLE001
        with torch.inference_mode():
            for _ in range(warmup):
                run()
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(iters):
                run()
            e1.record()
            torch.cuda.synchronize()
        return e0.elapsed_time(e1) / iters * 1e3, False


def official_tps(step_us):
    """official = K_cal*E[T]*tau/step with tau=STEP_US -> reproduces 481.53 at
    step=STEP_US (E[T] held fixed: the verify tune is bit-identical)."""
    return K_CAL * ET_DEPLOYED * (STEP_US / step_us)


# --------------------------------------------------------------------------- #
# Per-shape deployed measurement + focused num_stages sweep.                   #
# --------------------------------------------------------------------------- #
def sweep_shape(label, M, head_size, num_heads, num_kv_heads, context_len, sliding,
                use_3d, iters, warmup, verbose=True):
    num_queries_per_kv = num_heads // num_kv_heads
    dep_block_m = 16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    dep_tile = deployed_tile(head_size, sliding, use_3d)
    force_redirect = bool(use_3d and M > 1)  # verify: deployed selects 3D via redirect

    # Is this shape (if a verify batch) actually routed to 3D by the deployed
    # SPLITKV_VERIFY predicate? (instruction-1 evidence)
    seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // num_kv_heads
    routed_3d = None
    if M > 1:
        routed_3d = _would_redirect(
            q_rows=M, max_seqlen_q=M, segm_rows=max(M, seq_threshold_3D),
            seq_threshold_3D=seq_threshold_3D, num_seqs=1)

    # reference = REAL deployed wrapper (3D verify uses the splitkv redirect)
    inp_ref = make_inputs(M, head_size, num_heads, num_kv_heads, context_len, sliding)
    segm_ref = make_segm(M, num_heads, head_size, num_kv_heads)
    ref = call_deployed_wrapper(inp_ref, segm_ref, use_3d,
                                force_redirect=force_redirect).clone()
    nan_clean = bool(torch.isfinite(ref).all().item())

    # clone fidelity: launch_tuned at deployed config == wrapper (self-test a)
    inp_c = make_inputs(M, head_size, num_heads, num_kv_heads, context_len, sliding)
    segm_c = make_segm(M, num_heads, head_size, num_kv_heads)
    clone = launch_tuned(inp_c, segm_c, dep_tile, DEPLOYED_NUM_WARPS,
                         DEPLOYED_NUM_STAGES, dep_block_m, use_3d).clone()
    clone_bitident = bool(torch.equal(ref, clone))

    inp_t = make_inputs(M, head_size, num_heads, num_kv_heads, context_len, sliding)
    segm_t = make_segm(M, num_heads, head_size, num_kv_heads)

    def time_config(tile, warps, stages, block_m):
        def run():
            launch_tuned(inp_t, segm_t, tile, warps, stages, block_m, use_3d)
        return graph_time(run, iters, warmup)

    deployed_us, _ = time_config(dep_tile, DEPLOYED_NUM_WARPS, DEPLOYED_NUM_STAGES,
                                 dep_block_m)

    # LYNCHPIN (self-test i): the REAL wrapper (triton-default warps/stages) matches
    # our forced-(w4,s3) baseline -> the deployed default IS num_stages=3.
    inp_w = make_inputs(M, head_size, num_heads, num_kv_heads, context_len, sliding)
    segm_w = make_segm(M, num_heads, head_size, num_kv_heads)
    wrapper_us, _ = graph_time(
        lambda: call_deployed_wrapper(inp_w, segm_w, use_3d,
                                      force_redirect=force_redirect), iters, warmup)
    wrapper_matches_deployed = bool(abs(wrapper_us - deployed_us) <= 0.06 * deployed_us)

    # ---- focused sweep: num_stages headline (bit-ident TILE) + fp-tol TILEs ----
    rows = []
    n_compile_fail = 0
    tiles = GRID_TILE_BITIDENT + GRID_TILE_FPTOL
    for tile in tiles:
        for warps in GRID_WARPS:
            for stages in GRID_STAGES:
                bm = dep_block_m
                inp_v = make_inputs(M, head_size, num_heads, num_kv_heads,
                                    context_len, sliding)
                segm_v = make_segm(M, num_heads, head_size, num_kv_heads)
                try:
                    outv = launch_tuned(inp_v, segm_v, tile, warps, stages, bm,
                                        use_3d).clone()
                except Exception:  # noqa: BLE001
                    n_compile_fail += 1
                    continue
                if not torch.isfinite(outv).all().item():
                    n_compile_fail += 1
                    continue
                bit = bool(torch.equal(ref, outv))
                maxdiff = float((ref.float() - outv.float()).abs().max().item())
                us, captured = time_config(tile, warps, stages, bm)
                rows.append({
                    "tile": tile, "warps": warps, "stages": stages, "block_m": bm,
                    "us": us, "bit_identical": bit, "maxdiff": maxdiff,
                    "tile_changed": tile != dep_tile, "captured": captured,
                    "is_deployed": (tile == dep_tile and warps == DEPLOYED_NUM_WARPS
                                    and stages == DEPLOYED_NUM_STAGES),
                })

    bitident = [r for r in rows if r["bit_identical"]]
    best_biti = min(bitident, key=lambda r: r["us"]) if bitident else None
    # the specific num_stages=2 candidate at deployed tile/warps (the PR headline)
    s2 = [r for r in rows if r["tile"] == dep_tile and r["warps"] == DEPLOYED_NUM_WARPS
          and r["stages"] == 2]
    s2_row = s2[0] if s2 else None
    fptol = [r for r in rows if r["maxdiff"] <= FP_TOL]
    best_fptol = min(fptol, key=lambda r: r["us"]) if fptol else None

    biti_us = best_biti["us"] if best_biti else deployed_us
    speedup_biti = deployed_us / biti_us if biti_us > 0 else 1.0
    s2_us = s2_row["us"] if s2_row else deployed_us
    s2_speedup = deployed_us / s2_us if s2_us > 0 else 1.0
    s2_bitident = bool(s2_row["bit_identical"]) if s2_row else False
    s2_maxdiff = s2_row["maxdiff"] if s2_row else 0.0
    n_tile_diverged = sum(1 for r in rows if r["tile_changed"] and not r["bit_identical"])
    n_bitident_rejected = sum(1 for r in rows if not r["bit_identical"])

    res = {
        "label": label, "M": M, "head_size": head_size, "num_heads": num_heads,
        "num_kv_heads": num_kv_heads, "context_len": context_len,
        "sliding_window": sliding, "use_3d": use_3d, "force_redirect": force_redirect,
        "routed_3d_by_splitkv": routed_3d,
        "deployed_tile": dep_tile, "deployed_block_m": dep_block_m,
        "deployed_num_warps": DEPLOYED_NUM_WARPS,
        "deployed_num_stages": DEPLOYED_NUM_STAGES,
        "attn_us_deployed": deployed_us, "attn_us_wrapper_real": wrapper_us,
        "wrapper_matches_deployed": wrapper_matches_deployed,
        "attn_us_bitident": biti_us, "attn_speedup_bitident": speedup_biti,
        "best_bitident_config": {k: best_biti[k] for k in ("tile", "warps", "stages")}
                                if best_biti else None,
        # the explicit num_stages=3->2 candidate (the PR headline)
        "s2_us": s2_us, "s2_speedup": s2_speedup, "s2_bit_identical": s2_bitident,
        "s2_maxdiff": s2_maxdiff,
        "best_fptol_us": best_fptol["us"] if best_fptol else deployed_us,
        "best_fptol_config": {k: best_fptol[k] for k in ("tile", "warps", "stages")}
                             if best_fptol else None,
        "best_fptol_bit_identical": bool(best_fptol["bit_identical"]) if best_fptol else True,
        "best_fptol_maxdiff": best_fptol["maxdiff"] if best_fptol else 0.0,
        "clone_bitident": clone_bitident, "nan_clean": nan_clean,
        "n_configs": len(rows), "n_compile_fail": n_compile_fail,
        "n_bitident_rejected": n_bitident_rejected, "n_tile_diverged": n_tile_diverged,
        "rows": rows,
    }
    if verbose:
        bc = res["best_bitident_config"]
        print(f"[verify-sdpa] {label:26s} M={M:2d} h{head_size} {'3D' if use_3d else '2D'} "
              f"tile{dep_tile} routed3d={routed_3d}: wrap={wrapper_us:6.2f}us"
              f"(match={wrapper_matches_deployed}) dep={deployed_us:6.2f}us "
              f"s2={s2_us:6.2f}us({s2_speedup:.3f}x bit={s2_bitident} "
              f"md={s2_maxdiff:.1e}) biti_best={biti_us:6.2f}us{bc} "
              f"clone_ok={clone_bitident} cfail={n_compile_fail}", flush=True)
    return res


# --------------------------------------------------------------------------- #
# 128-draw bit-identity gate (the deployed greedy-safety certificate).         #
# --------------------------------------------------------------------------- #
def bitident_128(label, M, head_size, num_heads, num_kv_heads, sliding, use_3d,
                 n_draws=128, ctx_choices=(256, 512, 1024, 2048, 3072, 4000)):
    """Run the deployed 3D verify SDPA (num_stages=3) vs num_stages=2 over n_draws
    distinct realistic input draws; count non-bit-identical outputs. Bit-identity
    (torch.equal) => greedy-token identity by construction (the entire downstream
    sampler/accept is a deterministic function of these outputs)."""
    dep_tile = deployed_tile(head_size, sliding, use_3d)
    bm = 16 if (num_heads // num_kv_heads) <= 16 else triton.next_power_of_2(num_heads // num_kv_heads)
    divergent = 0
    max_md = 0.0
    for i in range(n_draws):
        ctx = ctx_choices[i % len(ctx_choices)]
        inp_a = make_inputs(M, head_size, num_heads, num_kv_heads, ctx, sliding,
                            seed=1000 + i)
        seg_a = make_segm(M, num_heads, head_size, num_kv_heads)
        outa = launch_tuned(inp_a, seg_a, dep_tile, DEPLOYED_NUM_WARPS,
                            DEPLOYED_NUM_STAGES, bm, use_3d).clone()
        inp_b = make_inputs(M, head_size, num_heads, num_kv_heads, ctx, sliding,
                            seed=1000 + i)
        seg_b = make_segm(M, num_heads, head_size, num_kv_heads)
        outb = launch_tuned(inp_b, seg_b, dep_tile, DEPLOYED_NUM_WARPS, 2, bm,
                            use_3d).clone()
        if not torch.equal(outa, outb):
            divergent += 1
            md = float((outa.float() - outb.float()).abs().max().item())
            max_md = max(max_md, md)
    print(f"[verify-sdpa] 128-gate {label:26s}: divergent={divergent}/{n_draws} "
          f"max_maxdiff={max_md:.2e}", flush=True)
    return {"label": label, "n_draws": n_draws, "divergent": divergent,
            "max_maxdiff": max_md, "bit_identical_all": bool(divergent == 0)}


# --------------------------------------------------------------------------- #
# Lean per-context deployed-vs-(num_stages=2) measurement (for the price band). #
# --------------------------------------------------------------------------- #
def measure_ctx(M, head_size, num_heads, num_kv_heads, context_len, sliding, use_3d,
                iters, warmup):
    """Measure the deployed (warps4/stages3) and the num_stages=2 candidate at a
    single context; return (deployed_us, s2_us, s2_bit_identical). Reuses the
    exact launch_tuned/graph_time machinery (no full grid)."""
    nqpkv = num_heads // num_kv_heads
    bm = 16 if nqpkv <= 16 else triton.next_power_of_2(nqpkv)
    dep_tile = deployed_tile(head_size, sliding, use_3d)
    inp_r = make_inputs(M, head_size, num_heads, num_kv_heads, context_len, sliding)
    seg_r = make_segm(M, num_heads, head_size, num_kv_heads)
    ref = launch_tuned(inp_r, seg_r, dep_tile, DEPLOYED_NUM_WARPS,
                       DEPLOYED_NUM_STAGES, bm, use_3d).clone()
    inp_s = make_inputs(M, head_size, num_heads, num_kv_heads, context_len, sliding)
    seg_s = make_segm(M, num_heads, head_size, num_kv_heads)
    outs2 = launch_tuned(inp_s, seg_s, dep_tile, DEPLOYED_NUM_WARPS, 2, bm,
                         use_3d).clone()
    s2_bit = bool(torch.equal(ref, outs2))
    inp_t = make_inputs(M, head_size, num_heads, num_kv_heads, context_len, sliding)
    seg_t = make_segm(M, num_heads, head_size, num_kv_heads)
    dep_us, _ = graph_time(
        lambda: launch_tuned(inp_t, seg_t, dep_tile, DEPLOYED_NUM_WARPS,
                             DEPLOYED_NUM_STAGES, bm, use_3d), iters, warmup)
    s2_us, _ = graph_time(
        lambda: launch_tuned(inp_t, seg_t, dep_tile, DEPLOYED_NUM_WARPS, 2, bm,
                             use_3d), iters, warmup)
    return dep_us, s2_us, s2_bit


def _price_one(g_dep, g_tuned, s_dep, s_tuned):
    """Compose the per-step verify-SDPA saving -> projected (upper-bound) TPS."""
    dep_total = N_VERIFY_GLOBAL_H512 * g_dep + N_VERIFY_SLIDING_H256 * s_dep
    tuned_total = N_VERIFY_GLOBAL_H512 * g_tuned + N_VERIFY_SLIDING_H256 * s_tuned
    saving = max(0.0, dep_total - tuned_total)
    new_step = STEP_US - saving
    gain = max(0.0, 100.0 * (STEP_US / new_step - 1.0))
    tps = FRONTIER_TPS * STEP_US / new_step
    return {
        "total_verify_sdpa_deployed_us": dep_total,
        "total_verify_sdpa_tuned_us": tuned_total,
        "verify_sdpa_saving_us": saving, "new_step_us": new_step,
        "projected_tps_gain_pct": gain, "honest_projected_tps_after": tps,
        "sdpa_share_of_verify": dep_total / VERIFY_FORWARD_US,
        "standalone_share_exceeds_verify": bool(dep_total > VERIFY_FORWARD_US),
        "clears_500": bool(tps >= 500.0),
    }


def price_band(contexts, iters, warmup):
    """Pricing band over the two dominant 3D verify shapes (global head-512 +
    sliding head-256) at each context. The realistic ctx (~512) is the HONEST
    headline; ctx=2048 is a loose upper bound (<1% of real decode steps)."""
    band = {}
    for ctx in contexts:
        g_dep, g_s2, g_bit = measure_ctx(8, 512, 8, 2, ctx, 0, True, iters, warmup)
        s_dep, s_s2, s_bit = measure_ctx(8, 256, 8, 2, ctx, 512, True, iters, warmup)
        g_tuned = g_s2 if g_bit else g_dep
        s_tuned = s_s2 if s_bit else s_dep
        priced = _price_one(g_dep, g_tuned, s_dep, s_tuned)
        priced.update({
            "context_len": ctx,
            "global_h512_deployed_us": g_dep, "global_h512_s2_us": g_s2,
            "global_h512_s2_bitident": g_bit,
            "sliding_h256_deployed_us": s_dep, "sliding_h256_s2_us": s_s2,
            "sliding_h256_s2_bitident": s_bit,
            "global_h512_s2_speedup": (g_dep / g_s2) if g_s2 > 0 else 1.0,
            "sliding_h256_s2_speedup": (s_dep / s_s2) if s_s2 > 0 else 1.0,
        })
        band[ctx] = priced
        print(f"[verify-sdpa] price ctx={ctx:5d}: sdpa={priced['total_verify_sdpa_deployed_us']:6.1f}us "
              f"({priced['sdpa_share_of_verify']*100:4.0f}% of {VERIFY_FORWARD_US:.0f}us"
              f"{' OVER-STATES' if priced['standalone_share_exceeds_verify'] else ''}) "
              f"saving={priced['verify_sdpa_saving_us']:5.2f}us -> "
              f"{priced['projected_tps_gain_pct']:+.2f}% -> "
              f"{priced['honest_projected_tps_after']:.1f}TPS "
              f"(clears500={priced['clears_500']})", flush=True)
        gc.collect(); torch.cuda.empty_cache()
    return band


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--context-len", type=int, default=PRICE_CTX_REALISTIC,
                    help="primary sweep context = realistic time-averaged decode "
                         "ctx (default 512); ctx=2048 over-states (<1% of steps)")
    ap.add_argument("--price-contexts", default="512,768,2048",
                    help="pricing band contexts (realistic/near-worst/loose-UB)")
    ap.add_argument("--gate-draws", type=int, default=128)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--output",
                    default="research/speed/verify_sdpa_linear_deploy/results.json")
    ap.add_argument("--wandb_project",
                    default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity",
                    default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="verify-sdpa-linear-deploy")
    ap.add_argument("--wandb_name", default="wirbel/verify-sdpa-linear-deploy")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    dev = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"[verify-sdpa] device {dev} sm_{cap[0]}{cap[1]} torch {torch.__version__} "
          f"triton {triton.__version__}", flush=True)
    torch.cuda.reset_peak_memory_stats()

    kname = type(kernel_unified_attention).__name__
    attn_is_shape_specialized = (kname != "JITFunction")
    print(f"[verify-sdpa] kernel jit_type={kname} autotuned={attn_is_shape_specialized}"
          f"; splitkv_imported={_HAVE_SPLITKV} max_q<={_SPLITKV_MAX_Q}; "
          f"is_gemma3_attn(512,0)={_is_gemma3_attention(512, 0)} "
          f"is_gemma3_attn(256,512)={_is_gemma3_attention(256, 512)}", flush=True)

    iters = 60 if args.quick else args.iters
    warmup = 15 if args.quick else args.warmup
    gate_draws = 16 if args.quick else args.gate_draws
    ctx = args.context_len

    # (label, M, head, num_heads, num_kv_heads, context, sliding, use_3d)
    # DEPLOYED verify path = 3D split-KV M=8 (the real target). 2D M=8 = #270's
    # measurement (cross-ref, shows the shape difference). M=1 = decode reference.
    shapes = [
        ("verify_global_h512_M8_3d", 8, 512, 8, 2, ctx, 0, True),
        ("verify_sliding_h256_M8_3d", 8, 256, 8, 2, ctx, 512, True),
        ("verify_global_h512_M8_2d_ref270", 8, 512, 8, 2, ctx, 0, False),
        ("verify_sliding_h256_M8_2d_ref270", 8, 256, 8, 2, ctx, 512, False),
        ("decode_global_h512_M1_3d", 1, 512, 8, 2, ctx, 0, True),
    ]
    results = {}
    for (label, M, hs, nh, nkv, c, sw, u3) in shapes:
        results[label] = sweep_shape(label, M, hs, nh, nkv, c, sw, u3, iters, warmup)
        gc.collect(); torch.cuda.empty_cache()

    # ---- 128-draw bit-identity gate on the DEPLOYED 3D verify shapes ----------
    gates = {}
    for label, M, hs, nh, nkv, sw, u3 in [
        ("verify_global_h512_M8_3d", 8, 512, 8, 2, 0, True),
        ("verify_sliding_h256_M8_3d", 8, 256, 8, 2, 512, True),
    ]:
        gates[label] = bitident_128(label, M, hs, nh, nkv, sw, u3, n_draws=gate_draws)
        gc.collect(); torch.cuda.empty_cache()
    total_divergent = sum(g["divergent"] for g in gates.values())
    linear_sdpa_tune_greedy_identical = bool(total_divergent == 0)
    linear_sdpa_tune_divergent_prompts = total_divergent

    # ---- instruction 1: kernel-shape match -----------------------------------
    g3d = results["verify_global_h512_M8_3d"]
    s3d = results["verify_sliding_h256_M8_3d"]
    g2d = results["verify_global_h512_M8_2d_ref270"]
    # The deployed linear verify uses the SAME kernel + SAME num_stages=3 default,
    # but a DIFFERENT launch config than #270's 2D M=8: 3D split-KV TILE=16 vs 2D
    # TILE=32. "Matches" is TRUE only if the deployed verify is the same shape as
    # the #270 measurement -> it is NOT (different dispatch path + tile).
    deployed_is_3d_verify = bool(g3d["routed_3d_by_splitkv"] is True)
    deployed_tile_16 = bool(g3d["deployed_tile"] == 16 and s3d["deployed_tile"] == 16)
    same_kernel_entry = True  # kernel_unified_attention for both 2D & 3D (verified)
    same_stages_default = bool(g3d["wrapper_matches_deployed"]
                               and attn_is_shape_specialized is False)
    # tree(#270) shape was 2D tile=32; deployed linear is 3D tile=16 -> mismatch
    linear_sdpa_kernel_matches_tree_shape = bool(
        same_kernel_entry and same_stages_default
        and (g3d["use_3d"] == g2d["use_3d"])           # same dispatch path? (False)
        and (g3d["deployed_tile"] == g2d["deployed_tile"]))  # same tile? (False)

    # ---- pricing band over contexts (realistic 512 headline; 2048 = loose UB) -
    price_contexts = [int(x) for x in str(args.price_contexts).split(",") if x.strip()]
    if PRICE_CTX_REALISTIC not in price_contexts:
        price_contexts = [PRICE_CTX_REALISTIC] + price_contexts
    band = price_band(sorted(set(price_contexts)), iters, warmup)
    band_real = band[PRICE_CTX_REALISTIC]
    band_ub = band.get(2048) or band[max(band)]

    # ---- instruction 2: kernel + full-verify-pass speedups (3D verify) --------
    # kernel-level (num_stages=3->2 at the deployed 3D shape, bit-ident gated)
    g3d_s2_ok = bool(g3d["s2_bit_identical"])
    s3d_s2_ok = bool(s3d["s2_bit_identical"])
    # use the s2 candidate iff bit-identical, else no saving for that layer
    g3d_speedup = g3d["s2_speedup"] if g3d_s2_ok else 1.0
    s3d_speedup = s3d["s2_speedup"] if s3d_s2_ok else 1.0
    sdpa_kernel_linear_speedup = g3d_speedup  # headline = global head-512 layer
    g3d_dep = g3d["attn_us_deployed"]; g3d_tuned = g3d["s2_us"] if g3d_s2_ok else g3d_dep
    s3d_dep = s3d["attn_us_deployed"]; s3d_tuned = s3d["s2_us"] if s3d_s2_ok else s3d_dep

    # pricing TOTALS from the realistic-context band (the HONEST headline; the
    # main-sweep per-shape us above are the full-grid measurement at the same ctx)
    total_verify_sdpa_deployed_us = band_real["total_verify_sdpa_deployed_us"]
    total_verify_sdpa_tuned_us = band_real["total_verify_sdpa_tuned_us"]
    verify_sdpa_saving_us = band_real["verify_sdpa_saving_us"]
    sdpa_share_of_verify = band_real["sdpa_share_of_verify"]
    standalone_share_exceeds_verify = bool(band_real["standalone_share_exceeds_verify"])
    # full-verify-pass speedup (over the 511us anchor; diagnostic)
    verify_forward_after_us = VERIFY_FORWARD_US - verify_sdpa_saving_us
    verify_forward_linear_speedup_pct = (
        100.0 * (VERIFY_FORWARD_US / verify_forward_after_us - 1.0)
        if verify_forward_after_us > 0 else 0.0)

    # ---- instruction 4: composition-honest TPS (realistic headline + UB band) -
    new_step_us = band_real["new_step_us"]
    projected_tps_gain_pct = band_real["projected_tps_gain_pct"]
    honest_projected_tps_after = band_real["honest_projected_tps_after"]
    clears_500 = bool(honest_projected_tps_after >= 500.0)
    beats_fern274 = bool(honest_projected_tps_after > FERN274_HONEST_K4_TPS)
    # loose upper bound (ctx=2048; <1% of real decode steps reach it)
    projected_tps_gain_pct_upper_ctx2048 = band_ub["projected_tps_gain_pct"]
    honest_projected_tps_after_upper_ctx2048 = band_ub["honest_projected_tps_after"]
    clears_500_even_at_ub = bool(band_ub["honest_projected_tps_after"] >= 500.0)
    # null the gain if the bit-identity gate failed (invalid lever)
    if not linear_sdpa_tune_greedy_identical:
        projected_tps_gain_pct = 0.0
        honest_projected_tps_after = FRONTIER_TPS
        projected_tps_gain_pct_upper_ctx2048 = 0.0
        honest_projected_tps_after_upper_ctx2048 = FRONTIER_TPS

    # ---- self-test conditions -------------------------------------------------
    st_a = all(r["clone_bitident"] for r in results.values())
    st_b = bool(deployed_is_3d_verify and (attn_is_shape_specialized is False)
                and deployed_tile_16)
    st_c = bool(g3d["attn_us_deployed"] > 0 and g3d["s2_us"] > 0
                and s3d["attn_us_deployed"] > 0)  # CUDA-event timings present
    st_d = bool(gates["verify_global_h512_M8_3d"]["n_draws"] >= (16 if args.quick else 128)
                and gates["verify_sliding_h256_M8_3d"]["n_draws"]
                >= (16 if args.quick else 128))
    st_e = bool(abs(K_CAL * ET_DEPLOYED - FRONTIER_TPS) < 1e-2)  # composition round-trip
    st_f = bool(FRONTIER_TPS == 481.53 and K_CAL == 125.268 and STEP_US == 1218.2
                and ET_DEPLOYED == 3.844 and K_DEPLOYED == 7)
    st_g = all(r["nan_clean"] for r in results.values())
    st_h = all(r["attn_speedup_bitident"] >= 1.0 - 0.05 for r in results.values())
    st_i = all(r["wrapper_matches_deployed"] for r in results.values())
    self_test_passes = bool(st_a and st_b and st_c and st_d and st_e and st_f
                            and st_g and st_h and st_i)

    peak_vram_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)

    handoff = (
        f"the deployed linear batch=8 verify SDPA "
        f"{'matches' if linear_sdpa_kernel_matches_tree_shape else 'does NOT match'} "
        f"the #270 M=8 tree shape (MAX_NUM_SEQS=1: the M=8 is the K+1 chain rows of "
        f"ONE sequence, and SPLITKV_VERIFY routes it to 3D split-KV TILE=16, not the "
        f"2D TILE=32 #270 measured; same kernel + same num_stages=3 default); applying "
        f"num_stages=3->2 gives {sdpa_kernel_linear_speedup:.3f}x kernel speedup "
        f"(global head-512 3D) and {verify_forward_linear_speedup_pct:+.2f}% full-"
        f"verify-pass speedup, bit-identical "
        f"{'yes' if linear_sdpa_tune_greedy_identical else str(linear_sdpa_tune_divergent_prompts) + ' diverge'}, "
        f"-> honest realistic-ctx (~{PRICE_CTX_REALISTIC}) {projected_tps_gain_pct:+.2f}% / "
        f"{honest_projected_tps_after:.1f} TPS (loose ctx=2048 UB "
        f"{projected_tps_gain_pct_upper_ctx2048:+.2f}% / "
        f"{honest_projected_tps_after_upper_ctx2048:.1f} TPS), vs fern #274's "
        f"{FERN274_HONEST_K4_TPS} static-K=4 honest ceiling, so the verify SDPA tune "
        f"{'IS' if clears_500_even_at_ub else 'is NOT'} the one post-phi lever above 500 "
        f"(even at the loose UB).")

    verdict = {
        "verify_sdpa_linear_deploy_self_test_passes": self_test_passes,
        "projected_tps_gain_pct": projected_tps_gain_pct,
        "linear_sdpa_tune_greedy_identical": linear_sdpa_tune_greedy_identical,
        "linear_sdpa_tune_divergent_prompts": linear_sdpa_tune_divergent_prompts,
        "linear_sdpa_kernel_matches_tree_shape": linear_sdpa_kernel_matches_tree_shape,
        "honest_projected_tps_after": honest_projected_tps_after,
        "clears_500": clears_500, "beats_fern274_493_96": beats_fern274,
        # realistic-ctx headline vs loose ctx=2048 upper bound (instruction 4 band)
        "price_ctx_realistic": PRICE_CTX_REALISTIC,
        "projected_tps_gain_pct_upper_ctx2048": projected_tps_gain_pct_upper_ctx2048,
        "honest_projected_tps_after_upper_ctx2048": honest_projected_tps_after_upper_ctx2048,
        "clears_500_even_at_ub": clears_500_even_at_ub,
        "standalone_share_exceeds_verify": standalone_share_exceeds_verify,
        "pricing_band": {str(c): {k: b[k] for k in (
            "projected_tps_gain_pct", "honest_projected_tps_after",
            "verify_sdpa_saving_us", "total_verify_sdpa_deployed_us",
            "sdpa_share_of_verify", "standalone_share_exceeds_verify", "clears_500",
            "global_h512_deployed_us", "global_h512_s2_us", "global_h512_s2_bitident",
            "sliding_h256_deployed_us", "sliding_h256_s2_us", "sliding_h256_s2_bitident")}
                          for c, b in band.items()},
        # kernel + full-pass speedups (instruction 2)
        "sdpa_kernel_linear_speedup": sdpa_kernel_linear_speedup,
        "sdpa_kernel_linear_speedup_global_h512": g3d_speedup,
        "sdpa_kernel_linear_speedup_sliding_h256": s3d_speedup,
        "verify_forward_linear_speedup_pct": verify_forward_linear_speedup_pct,
        "sdpa_share_of_verify": sdpa_share_of_verify,
        # shape-match evidence (instruction 1)
        "deployed_verify_is_3d_split_kv": deployed_is_3d_verify,
        "deployed_verify_tile": g3d["deployed_tile"],
        "ref270_2d_verify_tile": g2d["deployed_tile"],
        "deployed_verify_use_3d": g3d["use_3d"],
        "ref270_2d_verify_use_3d": g2d["use_3d"],
        "attn_is_shape_specialized": attn_is_shape_specialized,
        "kernel_jit_type": kname,
        # per-shape deployed/s2
        "global_h512_3d_deployed_us": g3d_dep, "global_h512_3d_s2_us": g3d["s2_us"],
        "global_h512_3d_s2_bitident": g3d_s2_ok, "global_h512_3d_s2_maxdiff": g3d["s2_maxdiff"],
        "sliding_h256_3d_deployed_us": s3d_dep, "sliding_h256_3d_s2_us": s3d["s2_us"],
        "sliding_h256_3d_s2_bitident": s3d_s2_ok, "sliding_h256_3d_s2_maxdiff": s3d["s2_maxdiff"],
        "ref270_2d_global_h512_s2_speedup": g2d["s2_speedup"],
        "ref270_2d_global_h512_s2_bitident": g2d["s2_bit_identical"],
        # pricing internals
        "n_verify_global_h512": N_VERIFY_GLOBAL_H512,
        "n_verify_sliding_h256": N_VERIFY_SLIDING_H256,
        "n_verify_fa2_h256": N_VERIFY_FA2_H256,
        "total_verify_sdpa_deployed_us": total_verify_sdpa_deployed_us,
        "total_verify_sdpa_tuned_us": total_verify_sdpa_tuned_us,
        "verify_sdpa_saving_us": verify_sdpa_saving_us,
        "verify_forward_us_anchor": VERIFY_FORWARD_US,
        "new_step_us": new_step_us, "step_us": STEP_US,
        # safety
        "ppl_pinned": PPL_PINNED, "nan_clean": st_g,
        "peak_vram_gib": peak_vram_gib, "vram_ok": bool(peak_vram_gib <= 24.0),
        # imported, unchanged
        "frontier_tps": FRONTIER_TPS, "private_tps": PRIVATE_TPS, "k_cal": K_CAL,
        "et_deployed": ET_DEPLOYED, "k_deployed": K_DEPLOYED,
        "fern274_honest_k4_tps": FERN274_HONEST_K4_TPS,
        "self_test_conditions": {
            "a_clone_fidelity": st_a, "b_shape_match_evidence": st_b,
            "c_cuda_event_timings": st_c, "d_128_gate": st_d,
            "e_composition_roundtrip": st_e, "f_constants_unchanged": st_f,
            "g_nan_clean": st_g, "h_speedup_sane": st_h,
            "i_lynchpin_wrapper_s3": st_i},
        "handoff_line": handoff,
    }

    print("\n[verify-sdpa] ===== VERDICT =====", flush=True)
    print(f"  shape-match (deployed linear verify vs #270 2D M=8): "
          f"{linear_sdpa_kernel_matches_tree_shape} "
          f"(deployed=3D-splitkv tile{g3d['deployed_tile']}, #270=2D tile{g2d['deployed_tile']}; "
          f"routed_3d={g3d['routed_3d_by_splitkv']})", flush=True)
    print(f"  kernel num_stages=3->2 (3D verify): global-h512 {g3d_speedup:.3f}x "
          f"(bit={g3d_s2_ok} md={g3d['s2_maxdiff']:.1e}), sliding-h256 {s3d_speedup:.3f}x "
          f"(bit={s3d_s2_ok} md={s3d['s2_maxdiff']:.1e}); #270 2D-ref global "
          f"{g2d['s2_speedup']:.3f}x (bit={g2d['s2_bit_identical']})", flush=True)
    print(f"  128-gate: divergent={total_divergent} -> greedy_identical="
          f"{linear_sdpa_tune_greedy_identical}", flush=True)
    print(f"  verify SDPA @ctx{PRICE_CTX_REALISTIC}: deployed={total_verify_sdpa_deployed_us:.1f}us "
          f"({sdpa_share_of_verify*100:.0f}% of {VERIFY_FORWARD_US:.0f}us verify"
          f"{' OVER-STATES' if standalone_share_exceeds_verify else ''}) "
          f"saving={verify_sdpa_saving_us:.2f}us -> full-verify-pass "
          f"{verify_forward_linear_speedup_pct:+.2f}%", flush=True)
    print(f"  PRICE realistic ctx~{PRICE_CTX_REALISTIC}: step {STEP_US:.1f}->{new_step_us:.1f}us "
          f"-> {projected_tps_gain_pct:+.2f}% -> {honest_projected_tps_after:.1f} TPS "
          f"(clears_500={clears_500}, beats_fern274={beats_fern274}); "
          f"loose ctx=2048 UB -> {projected_tps_gain_pct_upper_ctx2048:+.2f}% / "
          f"{honest_projected_tps_after_upper_ctx2048:.1f} TPS "
          f"(clears_500_at_UB={clears_500_even_at_ub})", flush=True)
    print(f"  self_test={self_test_passes}  conditions={verdict['self_test_conditions']}",
          flush=True)
    print(f"  HANDOFF: {handoff}", flush=True)

    payload = {
        "config": {"device": dev, "sm": f"{cap[0]}{cap[1]}", "torch": torch.__version__,
                   "triton": triton.__version__, "iters": iters, "warmup": warmup,
                   "context_len": ctx, "gate_draws": gate_draws, "quick": args.quick,
                   "note": "Deployed linear-path verify SDPA (fa2sw_precache_kenyan, "
                           "MAX_NUM_SEQS=1, SPLITKV_VERIFY=1, ONEGRAPH=1) micro-"
                           "profiled at the real 3D split-KV M=8 shape; num_stages=3->2 "
                           "bit-identity-gated. No serve change, no HF Job, no submission."},
        "shapes": {lab: {k: v for k, v in r.items() if k != "rows"}
                   for lab, r in results.items()},
        "sweep_rows": {lab: r["rows"] for lab, r in results.items()},
        "pricing_band": {str(c): b for c, b in band.items()},
        "gates": gates, "verdict": verdict,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[verify-sdpa] wrote {args.output}", flush=True)

    if not args.no_wandb:
        try:
            _log_wandb(args, payload, results, gates)
        except Exception as exc:  # noqa: BLE001
            print(f"[verify-sdpa] W&B logging failed (non-fatal): {exc!r}", flush=True)

    gc.collect(); torch.cuda.empty_cache()
    return 0 if self_test_passes else 1


def _log_wandb(args, payload, results, gates):
    import wandb
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type="profiling", config=payload["config"])
    v = payload["verdict"]
    vt = wandb.Table(columns=["shape", "M", "head", "use_3d", "routed_3d", "tile",
                              "wrapper_us", "deployed_us", "s2_us", "s2_speedup",
                              "s2_bit_identical", "s2_maxdiff", "biti_best_us",
                              "n_bitident_rejected"])
    for lab, r in results.items():
        vt.add_data(lab, r["M"], r["head_size"], r["use_3d"], r["routed_3d_by_splitkv"],
                    r["deployed_tile"], r["attn_us_wrapper_real"], r["attn_us_deployed"],
                    r["s2_us"], r["s2_speedup"], r["s2_bit_identical"], r["s2_maxdiff"],
                    r["attn_us_bitident"], r["n_bitident_rejected"])
    run.log({"verdict_table": vt})
    st = wandb.Table(columns=["shape", "M", "tile", "warps", "stages", "us",
                              "bit_identical", "maxdiff", "is_deployed"])
    for lab, r in results.items():
        for row in r["rows"]:
            st.add_data(lab, r["M"], row["tile"], row["warps"], row["stages"],
                        row["us"], row["bit_identical"], row["maxdiff"],
                        row["is_deployed"])
    run.log({"sweep_configs": st})
    gt = wandb.Table(columns=["shape", "n_draws", "divergent", "max_maxdiff",
                              "bit_identical_all"])
    for lab, g in gates.items():
        gt.add_data(lab, g["n_draws"], g["divergent"], g["max_maxdiff"],
                    g["bit_identical_all"])
    run.log({"gate_128": gt})
    pb = wandb.Table(columns=["context_len", "global_h512_dep_us", "global_h512_s2_us",
                              "global_h512_s2_bit", "sliding_h256_dep_us",
                              "sliding_h256_s2_us", "sliding_h256_s2_bit",
                              "total_sdpa_us", "sdpa_share_of_verify",
                              "standalone_exceeds_verify", "saving_us",
                              "projected_tps_gain_pct", "honest_tps_after", "clears_500"])
    for c, b in payload["pricing_band"].items():
        pb.add_data(int(c), b["global_h512_deployed_us"], b["global_h512_s2_us"],
                    b["global_h512_s2_bitident"], b["sliding_h256_deployed_us"],
                    b["sliding_h256_s2_us"], b["sliding_h256_s2_bitident"],
                    b["total_verify_sdpa_deployed_us"], b["sdpa_share_of_verify"],
                    b["standalone_share_exceeds_verify"], b["verify_sdpa_saving_us"],
                    b["projected_tps_gain_pct"], b["honest_projected_tps_after"],
                    b["clears_500"])
    run.log({"pricing_band": pb})
    run.summary.update({k: val for k, val in v.items()
                        if isinstance(val, (int, float, bool, str))})
    run.finish()
    print(f"[verify-sdpa] W&B run: {run.url}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
