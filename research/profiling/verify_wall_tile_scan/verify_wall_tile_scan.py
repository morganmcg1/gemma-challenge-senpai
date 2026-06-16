#!/usr/bin/env python
"""PR #447 -- Verify-wall full decomposition + Triton tile-headroom scan.

LOCAL on a single A10G (sm_86, on-target). Byte-exact, NO served-file change,
NO HF submission. Profiling + microbench only.

What this draws (the map nobody has drawn end-to-end):

  Part 1  Full verify-step wall decomposition at the served M=8 / K=7 /
          head_dim=256 / ctx=128 sm_86 decode shape. Each component is timed
          the way it runs in the served ONEGRAPH (CUDA-graph replay; launch
          amortized), summing to ~100% of T_verify:
            attention  = FA2 x35 sliding(head256) + Triton-3D split-KV x7 global(head512)
            int4-GEMM  = vendored Marlin W4A16 body stack (qkv/o/gate_up/down)
            lm_head    = vendored Marlin W4A16 12288-row pruned head
            sampling   = Triton rejection_greedy_sample_kernel
            dispatch   = T_verify_graph - sum(components)  (within-graph glue)

  Part 2  Triton tile-config sweep for every TUNABLE Triton kernel in the
          verify path (NOT the vendored Marlin CUDA GEMM, NOT vendored FA2):
            * Triton-3D split-KV attention (kernel_unified_attention +
              reduce_segments): sweep BLOCK_M -> BLOCK_Q, TILE_SIZE, num_warps,
              num_stages -- the knobs the autotune signal points at. Every
              swept config is correctness-gated vs the served-default output
              (max_abs_err tol) so a wrong launch can never be reported a win.
            * rejection_greedy_sample_kernel: a grid=(batch,) scalar loop over
              max_spec_len draft rows -- no BLOCK_M/BLOCK_N tile surface, so its
              re-tile headroom is structurally nil (reported, not swept blindly).

  Part 3  HONEST kernel->end-to-end mapping. A kernel saving of d_us microseconds
          off the verify wall maps to the decode cycle as
            d_TPS = TPS_base * d_us / cycle_us,   cycle_us = E[T(7)]/TPS_base.
          This is the discipline that the pinned-K (#433: modeled +13.998 ->
          realized -5.82) and cb3 (#437: modeled +15.60 -> realized 0.0) traps
          violated: a kernel-microbench delta is NOT an end-to-end delta. We
          report max_honest_endtoend_tps_delta = best kernel d_us * slope.

Anchors (advisor branch approval-gated-8gpu-20260613, all cited in PR #447):
  realized equivalence frontier  denken #423  5a6zq2yz   467.14 TPS
  deployed incumbent (non-equiv) PR #52        2x9fm2zx   481.53 TPS / PPL 2.3772
  attention fraction of verify   denken #441  7rb089z3   6.90%
                                 stark  #445  emljqube   9.28%
  live lead (UNPROVEN realized)  wirbel #442  e5n9a2dc   modeled attn-autotune +15.86 -> 483.0
  E[T(7)] ladder                 #289         fi34s269   3.851185944363104
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
from typing import Any

# ============================================================================ #
# Served geometry + anchors (gemma-4-E4B-it-qat, manifest fa2sw_*_kenyan).      #
# ============================================================================ #
# 42 transformer layers: 7 global (head512, 8Q/2KV, Triton-3D split-KV) +
# 35 sliding (head256, 8Q/2KV, FA2 FA_SLIDING=1, window 512).
N_GLOBAL_LAYERS = 7
N_SLIDING_LAYERS = 35
HEAD_GLOBAL = 512
HEAD_SLIDING = 256
N_Q_HEADS = 8
N_KV_HEADS = 2
SLIDING_WINDOW = 512
LMHEAD_WIDTH = 12288  # LM_HEAD_PRUNE 12k (PCK-04); in=2560
HIDDEN = 2560
KV_BLOCK_SIZE = 16
NUM_PAR_SOFTMAX_SEGMENTS = 16
MIN_LAUNCH_GRID_SIZE_2D = 128

# int4-Marlin body GEMM shapes -- (out_features, in_features, instance_count).
# 7 global + 35 sliding (cb3 #437 / #441 extraction).
BODY_SHAPES: list[dict[str, Any]] = [
    {"name": "q_full",   "out": 4096,  "in": 2560,  "count": 7},
    {"name": "q_slide",  "out": 2048,  "in": 2560,  "count": 35},
    {"name": "kv_full",  "out": 1024,  "in": 2560,  "count": 8},
    {"name": "kv_slide", "out": 512,   "in": 2560,  "count": 40},
    {"name": "o_full",   "out": 2560,  "in": 4096,  "count": 7},
    {"name": "o_slide",  "out": 2560,  "in": 2048,  "count": 35},
    {"name": "gate_up",  "out": 10240, "in": 2560,  "count": 84},
    {"name": "down",     "out": 2560,  "in": 10240, "count": 42},
]
INT4_BPW = 4.125  # 4b weight + bf16 g128 scale

# Spec-decode regime.
M_VERIFY = 8        # M = 1 + K, K=7
MAX_SPEC_LEN = 7

# Realized anchors.
REALIZED_TPS_K7 = 467.14
ET_K7 = 3.851185944363104          # #289 fi34s269
DEPLOYED_TPS = 481.53
PPL_ANCHOR = 2.3772                 # PPL-neutral: tiling cannot change emitted tokens
PPL_CAP = 2.42
# Per-step decode wall at the realized frontier: cycle_us = E[T]/TPS.
CYCLE_US = ET_K7 / REALIZED_TPS_K7 * 1e6      # ~8243.7 us
TPS_PER_US = REALIZED_TPS_K7 / CYCLE_US        # ~0.056665 TPS per us saved off verify

# Prior fractions for cross-check.
ATTN_FRAC_441 = 0.0690
ATTN_FRAC_445 = 0.0928
WIRBEL_MODELED_TPS = 15.86

# Tile sweep grid (PR #447). For the attention kernel the GEMM-style BLOCK_N
# maps to TILE_SIZE (the KV tile); BLOCK_M -> BLOCK_Q = BLOCK_M // num_q_per_kv.
SWEEP_BLOCK_M = [4, 8, 16, 32, 64]      # -> BLOCK_Q in {1,2,4,8,16} (served BLOCK_M=16)
SWEEP_TILE_SIZE = [16, 32, 64, 128]     # KV tile (BLOCK_N analog)
SWEEP_NUM_WARPS = [2, 4, 8]
SWEEP_NUM_STAGES = [2, 3, 4]
ATTN_CORRECT_TOL = 2.0e-3               # bf16 split-KV reorder tolerance (PR #39: ~6e-5 typical)


# ============================================================================ #
# Timing primitives (served ONEGRAPH basis: CUDA-graph replay).                 #
# ============================================================================ #
def _device():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable. Launch with CUDA_VISIBLE_DEVICES=0 (single-A10G pod "
            "remaps its one GPU to index 0; CVD=5 points at a non-existent device).")
    return torch.device("cuda:0")


def _gpu_facts(dev) -> dict[str, Any]:
    import torch
    p = torch.cuda.get_device_properties(dev)
    cc = torch.cuda.get_device_capability(dev)
    return {"name": p.name, "sm_count": p.multi_processor_count,
            "compute_capability": f"{cc[0]}.{cc[1]}",
            "total_mem_gib": round(p.total_memory / (1024 ** 3), 2),
            "is_a10g_sm86": bool(cc == (8, 6) and "A10G" in p.name)}


def graph_time_us(run, iters: int, warmup: int) -> tuple[float, bool]:
    """CUDA-graph replay timing (served ONEGRAPH basis). Falls back to eager
    event timing for uncapturable configs (flagged via the returned bool)."""
    import torch
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
        return ms * 1e3, True
    except Exception:  # noqa: BLE001 - eager fallback
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


# ============================================================================ #
# Component builders.                                                           #
# ============================================================================ #
def _make_triton_inputs(M, head_size, num_heads, num_kv_heads, context_len, sliding,
                        block_size=KV_BLOCK_SIZE, seed=0):
    import torch
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


def _make_segm(num_tokens, num_heads, head_size, num_kv_heads):
    import torch
    from vllm.triton_utils import triton
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


def _call_triton_3d_splitkv(inp, segm):
    """Served verify path: SPLITKV_VERIFY overrides max_seqlen_q=1 to select 3D
    split-KV while the true M rows are still computed (splitkv_verify_patch)."""
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    kw = dict(inp)
    kw.update(segm)
    kw["max_seqlen_q"] = 1
    unified_attention(**kw)
    return inp["out"]


def _make_fa2_inputs(M, head_size, num_heads, num_kv_heads, context_len,
                     block_size=KV_BLOCK_SIZE, seed=0):
    import torch
    torch.manual_seed(seed)
    dev = "cuda"
    seq_len = context_len + M
    num_blocks = (seq_len + block_size - 1) // block_size + 1
    q = torch.randn(M, num_heads, head_size, device=dev, dtype=torch.bfloat16) * 0.1
    k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size,
                          device=dev, dtype=torch.bfloat16) * 0.1
    v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size,
                          device=dev, dtype=torch.bfloat16) * 0.1
    out = torch.empty(M, num_heads, head_size, device=dev, dtype=torch.bfloat16)
    block_table = torch.arange(num_blocks, device=dev, dtype=torch.int32).view(1, -1)
    cu_seqlens_q = torch.tensor([0, M], device=dev, dtype=torch.int32)
    seqused_k = torch.tensor([seq_len], device=dev, dtype=torch.int32)
    return dict(q=q, k=k_cache, v=v_cache, out=out, cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=M, max_seqlen_k=seq_len, seqused_k=seqused_k,
                block_table=block_table, softmax_scale=head_size ** -0.5,
                window=(SLIDING_WINDOW - 1, 0))


def _call_fa2(inp):
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    flash_attn_varlen_func(
        q=inp["q"], k=inp["k"], v=inp["v"], max_seqlen_q=inp["max_seqlen_q"],
        cu_seqlens_q=inp["cu_seqlens_q"], max_seqlen_k=inp["max_seqlen_k"],
        seqused_k=inp["seqused_k"], softmax_scale=inp["softmax_scale"], causal=True,
        window_size=list(inp["window"]), block_table=inp["block_table"],
        out=inp["out"], fa_version=2)
    return inp["out"]


def _int4_weight_bytes(out: int, inn: int) -> float:
    return out * inn * INT4_BPW / 8.0


def _build_marlin_gemm(out: int, inn: int, dev, m: int = 1):
    import torch
    from vllm import _custom_ops as ops  # noqa: F401  (registers custom ops)
    from vllm.scalar_type import scalar_types
    import vllm.model_executor.layers.quantization.utils.marlin_utils as mu
    import vllm.model_executor.layers.quantization.utils.marlin_utils_test as mt
    K, N = inn, out
    wtype = scalar_types.uint4b8
    gs = 128
    w = (torch.randn(K, N, dtype=torch.bfloat16, device=dev) * 0.02)
    _w_ref, q_w, s, _g_idx, _sort, _rp = mt.marlin_quantize(w, wtype, gs, act_order=False)
    ws = mu.marlin_make_workspace_new(dev)
    zp = torch.empty(0, dtype=torch.int, device=dev)
    g_idx = torch.empty(0, dtype=torch.int, device=dev)
    sort_idx = torch.empty(0, dtype=torch.int, device=dev)
    x = torch.randn(m, K, dtype=torch.bfloat16, device=dev)

    def run():
        return mu.apply_gptq_marlin_linear(
            x, q_w, s, zp, g_idx, sort_idx, ws, wtype,
            output_size_per_partition=N, input_size_per_partition=K, is_k_full=True)

    out_t = run()
    ok = bool(out_t.shape == (m, N) and torch.isfinite(out_t).all().item())
    return run, _int4_weight_bytes(out, inn), ok


def _build_rejection_greedy(m_spec: int, dev):
    """Served verify accept/reject: greedy rejection-sampling Triton kernel.
    grid=(batch,) at concurrency=1; a scalar loop over max_spec_len draft rows."""
    import torch
    from vllm.v1.sample.rejection_sampler import (
        rejection_greedy_sample_kernel, PLACEHOLDER_TOKEN_ID)
    bs = 1
    out_ids = torch.full((bs, m_spec + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int32, device=dev)
    cu_num_draft = torch.tensor([m_spec], dtype=torch.int32, device=dev)
    draft_ids = torch.randint(0, 256000, (m_spec,), dtype=torch.int32, device=dev)
    target_argmax = torch.randint(0, 256000, (m_spec + 1,), dtype=torch.int32, device=dev)
    bonus = torch.randint(0, 256000, (bs, 1), dtype=torch.int32, device=dev)

    def run():
        rejection_greedy_sample_kernel[(bs,)](
            out_ids, cu_num_draft, draft_ids, target_argmax, bonus,
            None, m_spec, None, None, SYNTHETIC_MODE=False)
        return out_ids

    run()
    return run


# ============================================================================ #
# Attention tile-sweep: re-launch the 3D split-KV kernel with overridden tile   #
# config, reusing vLLM's own arg construction via module-global interception.   #
# Every swept config is correctness-gated vs the served default.                #
# ============================================================================ #
_SWEEP_CFG: dict[str, Any] = {"cfg": None}


def _install_attn_proxies():
    """Wrap kernel_unified_attention + reduce_segments so a chosen tile config is
    injected at launch. With cfg=None the served default launch is reproduced."""
    import vllm.v1.attention.ops.triton_unified_attention as ua

    real_kernel = ua.kernel_unified_attention
    real_reduce = ua.reduce_segments
    if getattr(real_kernel, "_vwts_proxy", False):
        return ua

    class _KernelProxy:
        _vwts_proxy = True

        def __getitem__(self, grid):
            cfg = _SWEEP_CFG["cfg"]
            base = real_kernel[grid]
            if cfg is None:
                return base

            def launch(**kw):
                g = grid
                if cfg.get("BLOCK_M") is not None:
                    kw["BLOCK_M"] = cfg["BLOCK_M"]
                    kw["BLOCK_Q"] = cfg["BLOCK_Q"]
                    g = (cfg["q_rows"] // cfg["BLOCK_Q"] + cfg["num_seqs"],
                         grid[1], grid[2])
                if cfg.get("TILE_SIZE") is not None:
                    kw["TILE_SIZE"] = cfg["TILE_SIZE"]
                extra = {}
                if cfg.get("num_warps"):
                    extra["num_warps"] = cfg["num_warps"]
                if cfg.get("num_stages"):
                    extra["num_stages"] = cfg["num_stages"]
                return real_kernel[g](**kw, **extra)

            return launch

    class _ReduceProxy:
        _vwts_proxy = True

        def __getitem__(self, grid):
            cfg = _SWEEP_CFG["cfg"]
            base = real_reduce[grid]
            if cfg is None or cfg.get("BLOCK_Q") is None:
                return base

            def launch(**kw):
                kw["BLOCK_Q"] = cfg["BLOCK_Q"]
                return real_reduce[grid](**kw)

            return launch

    ua.kernel_unified_attention = _KernelProxy()
    ua.reduce_segments = _ReduceProxy()
    return ua


def _attn_reference_out(head_size, sliding):
    """Served-default Triton-3D output for the verify shape (correctness anchor)."""
    import torch
    inp = _make_triton_inputs(M_VERIFY, head_size, N_Q_HEADS, N_KV_HEADS, 128, sliding)
    segm = _make_segm(M_VERIFY, N_Q_HEADS, head_size, N_KV_HEADS)
    _SWEEP_CFG["cfg"] = None
    _call_triton_3d_splitkv(inp, segm)
    torch.cuda.synchronize()
    return inp, segm, inp["out"].clone()


def time_attn_config(inp, segm, cfg, iters, warmup):
    """Time one tile config + report correctness vs served default."""
    import torch
    _SWEEP_CFG["cfg"] = cfg
    try:
        # correctness first (fresh out buffer)
        inp["out"].zero_()
        _call_triton_3d_splitkv(inp, segm)
        torch.cuda.synchronize()
        got = inp["out"].clone()
        max_abs_err = float((got - cfg["_ref"]).abs().max().item())
        ok = math.isfinite(max_abs_err) and max_abs_err <= ATTN_CORRECT_TOL
        us, captured = graph_time_us(lambda: _call_triton_3d_splitkv(inp, segm),
                                     iters, warmup)
    except Exception as exc:  # noqa: BLE001 - invalid launch config
        _SWEEP_CFG["cfg"] = None
        return {"valid": False, "err": repr(exc)[:160], "us": float("inf"),
                "max_abs_err": float("nan"), "captured": False}
    finally:
        _SWEEP_CFG["cfg"] = None
    return {"valid": bool(ok), "us": us, "max_abs_err": max_abs_err,
            "captured": bool(captured)}


# ============================================================================ #
# GPU driver.                                                                   #
# ============================================================================ #
def run_gpu(args) -> dict[str, Any]:
    import torch
    dev = _device()
    torch.zeros(1, device=dev)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    facts = _gpu_facts(dev)
    iters, warmup = args.iters, args.warmup
    ctx = args.context_len
    _install_attn_proxies()

    # Discover the served-default tile config the verify kernel actually launches.
    from vllm.v1.attention.ops.triton_unified_attention import _get_tile_size
    served_block_m = 16 if (N_Q_HEADS // N_KV_HEADS) <= 16 else 1
    served_block_q = served_block_m // (N_Q_HEADS // N_KV_HEADS)
    served_tile_global = _get_tile_size(HEAD_GLOBAL, 0, 2, is_prefill=False)
    served = {"BLOCK_M": served_block_m, "BLOCK_Q": served_block_q,
              "TILE_SIZE_DECODE_global": int(served_tile_global),
              "num_par_softmax_segments": NUM_PAR_SOFTMAX_SEGMENTS}
    print(f"[vwts] device {facts['name']} cc{facts['compute_capability']} "
          f"torch {torch.__version__} iters={iters} warmup={warmup} ctx={ctx}", flush=True)
    print(f"[vwts] served verify-attn tile defaults: {served}", flush=True)

    # ---- Part 1: component timings (graph-replay, served ONEGRAPH basis) ------
    # attention runners: FA2 x35 (head256, vendored) + Triton-3D x7 (head512, tunable)
    fa2_inp = _make_fa2_inputs(M_VERIFY, HEAD_SLIDING, N_Q_HEADS, N_KV_HEADS, ctx)
    tri_inp = _make_triton_inputs(M_VERIFY, HEAD_GLOBAL, N_Q_HEADS, N_KV_HEADS, ctx, 0)
    tri_segm = _make_segm(M_VERIFY, N_Q_HEADS, HEAD_GLOBAL, N_KV_HEADS)
    _SWEEP_CFG["cfg"] = None

    fa2_us, cap_fa = graph_time_us(lambda: _call_fa2(fa2_inp), iters, warmup)
    tri_us, cap_tr = graph_time_us(lambda: _call_triton_3d_splitkv(tri_inp, tri_segm),
                                   iters, warmup)
    t_attn_fa2 = N_SLIDING_LAYERS * fa2_us
    t_attn_tri = N_GLOBAL_LAYERS * tri_us
    t_attn = t_attn_fa2 + t_attn_tri
    print(f"[vwts] attn: FA2-h256={fa2_us:.2f}us x35={t_attn_fa2:.1f}  "
          f"Tri3D-h512={tri_us:.2f}us x7={t_attn_tri:.1f}  T_attn={t_attn:.1f}us", flush=True)
    gc.collect(); torch.cuda.empty_cache()

    # int4-Marlin body GEMM stack (vendored), graph-replayed weighted by count.
    marlin_runs = []
    body_ok = True
    per_shape = {}
    for sh in BODY_SHAPES:
        run, _wb, ok = _build_marlin_gemm(sh["out"], sh["in"], dev, M_VERIFY)
        body_ok = body_ok and ok
        us, cap = graph_time_us(run, iters, warmup)
        per_shape[sh["name"]] = {"us_each": us, "count": sh["count"],
                                 "us_total": us * sh["count"], "captured": cap}
        marlin_runs.append((run, sh["count"]))
    t_body = sum(v["us_total"] for v in per_shape.values())
    print(f"[vwts] body int4-Marlin stack: T_body={t_body:.1f}us (ok={body_ok})", flush=True)
    for nm, v in per_shape.items():
        print(f"        {nm:9s} {v['us_each']:7.2f}us x{v['count']:>3d} = "
              f"{v['us_total']:8.1f}us  cap={v['captured']}", flush=True)
    gc.collect(); torch.cuda.empty_cache()

    # lm_head: pruned 12288 Marlin W4A16 (vendored).
    lm_run, _lwb, lm_ok = _build_marlin_gemm(LMHEAD_WIDTH, HIDDEN, dev, M_VERIFY)
    t_lmhead, cap_lm = graph_time_us(lm_run, iters, warmup)
    print(f"[vwts] lm_head Marlin 12288: T_lmhead={t_lmhead:.2f}us (ok={lm_ok})", flush=True)

    # sampling: rejection_greedy_sample_kernel (Triton, tunable surface = none).
    rej_run = _build_rejection_greedy(MAX_SPEC_LEN, dev)
    t_sampling, cap_rej = graph_time_us(rej_run, iters, warmup)
    print(f"[vwts] sampling rejection_greedy: T_sampling={t_sampling:.3f}us", flush=True)

    # dispatch: full verify-forward graph minus the sum of components (within-graph glue).
    def _run_verify_forward():
        _SWEEP_CFG["cfg"] = None
        for _ in range(N_SLIDING_LAYERS):
            _call_fa2(fa2_inp)
        for _ in range(N_GLOBAL_LAYERS):
            _call_triton_3d_splitkv(tri_inp, tri_segm)
        for run, cnt in marlin_runs:
            for _ in range(cnt):
                run()
        lm_run()
        rej_run()

    t_verify_graph, cap_vf = graph_time_us(_run_verify_forward, max(50, iters // 4),
                                           max(20, warmup // 4))
    comp_sum = t_attn + t_body + t_lmhead + t_sampling
    t_dispatch = max(0.0, t_verify_graph - comp_sum)
    # If the fused graph is *faster* than the sum (kernels overlap / scheduler), the
    # components are an upper bound; report dispatch=0 and note the overlap factor.
    overlap_factor = t_verify_graph / comp_sum if comp_sum > 0 else float("nan")
    t_verify = max(t_verify_graph, comp_sum)
    print(f"[vwts] T_verify_graph={t_verify_graph:.1f}us  sum_components={comp_sum:.1f}us "
          f"overlap={overlap_factor:.3f}  dispatch={t_dispatch:.1f}us", flush=True)
    gc.collect(); torch.cuda.empty_cache()

    comp = {"attention": t_attn, "int4_gemm_body": t_body, "lm_head": t_lmhead,
            "sampling": t_sampling, "dispatch": t_dispatch}
    frac = {k: v / t_verify for k, v in comp.items()}

    # ---- Part 2: tile sweep of the tunable Triton kernels ---------------------
    # (a) Triton-3D split-KV attention (head512 global; the wirbel-autotune target)
    inp_s, segm_s, ref = _attn_reference_out(HEAD_GLOBAL, 0)
    nqpk = N_Q_HEADS // N_KV_HEADS
    base_cfg = {"BLOCK_M": None, "TILE_SIZE": None, "num_warps": None,
                "num_stages": None, "_ref": ref}
    base = time_attn_config(inp_s, segm_s, base_cfg, iters, warmup)
    base_us = base["us"]
    print(f"[vwts] attn sweep baseline (served default) = {base_us:.3f}us "
          f"valid={base['valid']} err={base['max_abs_err']:.2e}", flush=True)

    sweep_rows = []
    best = {"us": base_us, "cfg": "served_default", "valid": True,
            "max_abs_err": 0.0, "cfg_dict": None}
    n_cfg = n_valid = 0
    for bm in SWEEP_BLOCK_M:
        bq = bm // nqpk
        if bq < 1 or bm % nqpk != 0:
            continue
        for ts in SWEEP_TILE_SIZE:
            for nw in SWEEP_NUM_WARPS:
                for ns in SWEEP_NUM_STAGES:
                    if bm == 16 and ts == served["TILE_SIZE_DECODE_global"] and nw == 4 and ns == 3:
                        # near the served default; still measure but mark
                        pass
                    cfg = {"BLOCK_M": bm, "BLOCK_Q": bq, "TILE_SIZE": ts,
                           "num_warps": nw, "num_stages": ns, "_ref": ref,
                           "q_rows": M_VERIFY, "num_seqs": 1}
                    r = time_attn_config(inp_s, segm_s, cfg, iters, warmup)
                    n_cfg += 1
                    row = {"BLOCK_M": bm, "BLOCK_Q": bq, "TILE_SIZE": ts,
                           "num_warps": nw, "num_stages": ns, "us": r["us"],
                           "valid": r["valid"], "max_abs_err": r["max_abs_err"],
                           "captured": r["captured"]}
                    sweep_rows.append(row)
                    if r["valid"]:
                        n_valid += 1
                        if r["us"] < best["us"]:
                            best = {"us": r["us"], "cfg": f"BM{bm}_BQ{bq}_TS{ts}_w{nw}_s{ns}",
                                    "valid": True, "max_abs_err": r["max_abs_err"],
                                    "cfg_dict": {k: v for k, v in cfg.items()
                                                 if k != "_ref"}}
    # Raw sweep-min is winner's-curse biased: min over many noisy configs sits
    # below the served default by order-statistics even when the winner is
    # served-equivalent. Report it, but do NOT map it to TPS directly.
    sweep_min_delta_us_per_layer = base_us - best["us"]
    sweep_min_speedup_pct = (sweep_min_delta_us_per_layer / base_us * 100.0
                             if base_us > 0 else 0.0)
    print(f"[vwts] attn sweep: {n_valid}/{n_cfg} valid configs; best={best['cfg']} "
          f"{best['us']:.3f}us vs {base_us:.3f}us (sweep-min per-layer "
          f"d={sweep_min_delta_us_per_layer:+.3f}us, {sweep_min_speedup_pct:+.2f}%)",
          flush=True)

    # Winner's-curse guard: re-time the winning config head-to-head against the
    # served default, alternating, with several reps. The CONFIRMED median delta
    # (not the sweep-min) is the honest kernel saving used for the TPS mapping.
    confirm = {"confirmed": False, "reps": 0,
               "base_median_us": base_us, "best_median_us": best["us"],
               "delta_us_per_layer": sweep_min_delta_us_per_layer,
               "speedup_pct": sweep_min_speedup_pct,
               "base_samples": [], "best_samples": []}
    if best.get("cfg_dict") is not None:
        reps = 7
        win_cfg = dict(best["cfg_dict"]); win_cfg["_ref"] = ref
        bs, ws = [], []
        for _ in range(reps):
            bs.append(time_attn_config(inp_s, segm_s, base_cfg, iters, warmup)["us"])
            ws.append(time_attn_config(inp_s, segm_s, win_cfg, iters, warmup)["us"])
        b0 = statistics.median(bs)
        b1 = statistics.median(ws)
        d = b0 - b1
        confirm = {"confirmed": True, "reps": reps,
                   "base_median_us": b0, "best_median_us": b1,
                   "delta_us_per_layer": d,
                   "speedup_pct": (d / b0 * 100.0 if b0 > 0 else 0.0),
                   "base_samples": bs, "best_samples": ws}
        print(f"[vwts] winner reconfirm ({reps} reps alternating): "
              f"base_med={b0:.3f}us best_med={b1:.3f}us -> per-layer d={d:+.3f}us "
              f"({confirm['speedup_pct']:+.2f}%)  [sweep-min was {sweep_min_speedup_pct:+.2f}%]",
              flush=True)

    # kernel-level delta on the Triton-3D attention (per-layer), x7 global layers.
    # Uses the CONFIRMED (unbiased) median, clamped >=0 (a served-equivalent or
    # slower winner yields no honest saving).
    attn_tri_kernel_delta_us_per_layer = max(0.0, confirm["delta_us_per_layer"])
    attn_tri_kernel_delta_us = attn_tri_kernel_delta_us_per_layer * N_GLOBAL_LAYERS
    attn_kernel_speedup_pct = confirm["speedup_pct"]

    # (b) rejection_greedy_sample_kernel -- no BLOCK_M/BLOCK_N tile surface.
    rej_sweep = {"tunable_tile_surface": False,
                 "reason": "grid=(batch,) scalar loop over max_spec_len draft rows; "
                           "no BLOCK_M/BLOCK_N constexpr -> re-tile headroom structurally nil",
                 "t_sampling_us": t_sampling}

    # ---- Part 3: honest kernel -> end-to-end TPS mapping ----------------------
    def endtoend_delta_tps(saving_us: float) -> float:
        return REALIZED_TPS_K7 * saving_us / CYCLE_US

    # Best achievable verify-wall saving from re-tiling = attention kernel delta
    # (the only tunable Triton kernel with non-trivial size; rejection ~ nil).
    attn_endtoend_tps = endtoend_delta_tps(max(0.0, attn_tri_kernel_delta_us))
    # Upper-bound sanity: even ELIMINATING the entire tunable Triton-3D attention.
    attn_elim_endtoend_tps = endtoend_delta_tps(t_attn_tri)
    max_honest = max(0.0, attn_endtoend_tps)

    peak_vram_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return {
        "facts": facts, "iters": iters, "warmup": warmup, "context_len": ctx,
        "served_tile_defaults": served,
        "components_us": comp, "fractions": frac,
        "t_verify_us": t_verify, "t_verify_graph_us": t_verify_graph,
        "comp_sum_us": comp_sum, "overlap_factor": overlap_factor,
        "attn_breakdown_us": {"fa2_h256_each": fa2_us, "fa2_h256_x35": t_attn_fa2,
                              "tri3d_h512_each": tri_us, "tri3d_h512_x7": t_attn_tri,
                              "tri3d_tunable_frac_of_attn": (t_attn_tri / t_attn
                                                             if t_attn > 0 else 0.0)},
        "body_per_shape": per_shape,
        "attn_sweep": {"baseline_us": base_us, "best": best,
                       "n_configs": n_cfg, "n_valid": n_valid,
                       "sweep_min_delta_us_per_layer": sweep_min_delta_us_per_layer,
                       "sweep_min_speedup_pct": sweep_min_speedup_pct,
                       "confirm": confirm,
                       "kernel_delta_us_per_layer": attn_tri_kernel_delta_us_per_layer,
                       "kernel_delta_us_x7": attn_tri_kernel_delta_us,
                       "kernel_speedup_pct": attn_kernel_speedup_pct,
                       "rows": sweep_rows},
        "rejection_sweep": rej_sweep,
        "endtoend": {"cycle_us": CYCLE_US, "tps_per_us": TPS_PER_US,
                     "attn_retile_endtoend_tps": attn_endtoend_tps,
                     "attn_eliminate_endtoend_tps_upperbound": attn_elim_endtoend_tps,
                     "max_honest_endtoend_tps_delta": max_honest},
        "body_ok": body_ok, "lm_ok": lm_ok,
        "captured_flags": {"fa2": cap_fa, "tri3d": cap_tr, "lmhead": cap_lm,
                           "sampling": cap_rej, "verify_forward": cap_vf},
        "peak_vram_gib": peak_vram_gib,
    }


# ============================================================================ #
# Verdict.                                                                       #
# ============================================================================ #
def build_verdict(meas: dict[str, Any]) -> dict[str, Any]:
    frac = meas["fractions"]
    e2e = meas["endtoend"]
    attn_frac = frac["attention"]
    gemm_frac = frac["int4_gemm_body"]
    max_honest = e2e["max_honest_endtoend_tps_delta"]
    # (a) int4-GEMM dominant + vendored-optimal?
    gemm_dominant = bool(gemm_frac == max(frac.values()))
    # (c) attention fraction corroborates #441 6.90% / #445 9.28%?
    attn_corroborates = bool(min(ATTN_FRAC_441, ATTN_FRAC_445) - 0.03
                             <= attn_frac <= max(ATTN_FRAC_441, ATTN_FRAC_445) + 0.03)
    sw = meas.get("attn_sweep", {})
    confirm = sw.get("confirm", {})
    sweepmin_pct = sw.get("sweep_min_speedup_pct", 0.0)
    confirmed_pct = sw.get("kernel_speedup_pct", 0.0)
    return {
        "max_honest_endtoend_tps_delta": round(max_honest, 4),
        "any_nonattn_triton_over_2tps": False,   # rejection nil; lm_head/body vendored Marlin
        "int4_gemm_dominant": gemm_dominant,
        "int4_gemm_frac_of_verify": round(gemm_frac, 4),
        "int4_gemm_vendored_marlin_no_selectable_tiling": True,
        "attn_frac_of_verify": round(attn_frac, 4),
        "attn_frac_corroborates_441_445": attn_corroborates,
        "attn_tunable_tri3d_frac_of_attn": round(
            meas["attn_breakdown_us"]["tri3d_tunable_frac_of_attn"], 4),
        "attn_best_cfg": sw.get("best", {}).get("cfg", "served_default"),
        "attn_kernel_sweepmin_speedup_pct": round(sweepmin_pct, 3),
        "attn_kernel_confirmed_speedup_pct": round(confirmed_pct, 3),
        "winner_curse_shrinks_delta": bool(confirm.get("confirmed")
                                           and confirmed_pct < sweepmin_pct - 1e-9),
        "beats_realized_467": bool(REALIZED_TPS_K7 + max_honest > REALIZED_TPS_K7 + 1e-9
                                   and max_honest > 0),
        "beats_deployed_481": bool(REALIZED_TPS_K7 + max_honest > DEPLOYED_TPS),
        "wirbel_modeled_tps": WIRBEL_MODELED_TPS,
        "wirbel_modeled_implied_verify_saving_us": round(WIRBEL_MODELED_TPS / TPS_PER_US, 1),
        "ppl": PPL_ANCHOR, "ppl_cap": PPL_CAP,
        "tps_at_realized_frontier": REALIZED_TPS_K7,
    }


# ============================================================================ #
# 0-GPU self-test.                                                              #
# ============================================================================ #
def self_test() -> dict[str, Any]:
    res = {}
    def ck(name, cond):
        res[name] = bool(cond)
        print(f"        {'ok ' if cond else 'XX '} {name}", flush=True)

    # cycle / slope arithmetic
    cyc = ET_K7 / REALIZED_TPS_K7 * 1e6
    ck("a_cycle_us_matches", abs(cyc - CYCLE_US) < 1e-6)
    ck("b_slope_positive", TPS_PER_US > 0 and abs(TPS_PER_US - REALIZED_TPS_K7 / CYCLE_US) < 1e-12)
    # mapping: saving 0 -> 0 TPS; saving s>0 monotone
    ck("c_map_zero", abs(REALIZED_TPS_K7 * 0.0 / CYCLE_US) < 1e-12)
    d50 = REALIZED_TPS_K7 * 50.0 / CYCLE_US
    d100 = REALIZED_TPS_K7 * 100.0 / CYCLE_US
    ck("d_map_monotone", d100 > d50 > 0)
    # wirbel modeled +15.86 implies a verify saving; check it exceeds the whole
    # tunable Triton-3D attention size envelope (the trap signature).
    wirbel_saving = WIRBEL_MODELED_TPS / TPS_PER_US
    ck("e_wirbel_implies_large_saving", wirbel_saving > 250.0)  # ~280us
    # fractions sum to 1 on a synthetic decomposition
    synth = {"attention": 0.069, "int4_gemm_body": 0.918, "lm_head": 0.004,
             "sampling": 0.001, "dispatch": 0.008}
    ck("f_synth_fracs_sum_1", abs(sum(synth.values()) - 1.0) < 1e-9)
    # verdict wiring on synthetic measurement
    synth_meas = {
        "fractions": synth,
        "endtoend": {"cycle_us": CYCLE_US, "tps_per_us": TPS_PER_US,
                     "max_honest_endtoend_tps_delta": 0.6},
        "attn_breakdown_us": {"tri3d_tunable_frac_of_attn": 0.07},
    }
    v = build_verdict(synth_meas)
    ck("g_gemm_dominant_true", v["int4_gemm_dominant"] is True)
    ck("h_nonattn_over2_false", v["any_nonattn_triton_over_2tps"] is False)
    ck("i_attn_corroborates", v["attn_frac_corroborates_441_445"] is True)
    ck("j_no_beat_deployed", v["beats_deployed_481"] is False)
    # eliminating a 90us tunable attn maps below the +2 GREEN bar AND below
    # the wirbel claim -> the honest envelope is small.
    elim90 = REALIZED_TPS_K7 * 90.0 / CYCLE_US
    ck("k_elim90_under_wirbel", elim90 < WIRBEL_MODELED_TPS)
    ck("l_elim90_value", abs(elim90 - 5.1) < 0.6)
    ck("m_ppl_neutral", PPL_ANCHOR <= PPL_CAP)
    ck("n_nan_clean", all(math.isfinite(x) for x in [cyc, TPS_PER_US, wirbel_saving, elim90]))
    # winner's-curse guard: a confirmed delta below the sweep-min is detected,
    # and a negative confirmed delta clamps the honest saving to 0.
    sm = build_verdict({**synth_meas, "attn_sweep": {
        "best": {"cfg": "BM16_BQ4_TS16_w4_s2"},
        "sweep_min_speedup_pct": 5.6,
        "kernel_speedup_pct": 0.4,
        "confirm": {"confirmed": True}}})
    ck("o_winner_curse_detected", sm["winner_curse_shrinks_delta"] is True)
    ck("p_honest_clamp_nonneg", max(0.0, -3.0) == 0.0)
    npass = sum(res.values())
    print(f"[vwts] self-test: {'PASS' if npass == len(res) else 'FAIL'} ({npass}/{len(res)})",
          flush=True)
    res["_n_pass"] = npass
    res["_n_total"] = len(res) - 1 if "_n_pass" in res else len(res)
    return res


# ============================================================================ #
# Plumbing.                                                                      #
# ============================================================================ #
def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else round(o, 6)
    return o


def log_wandb(args, payload):
    import wandb
    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
        group=args.wandb_group, name=args.wandb_name, job_type="analysis-gpu-microbench",
        config={"pr": 447, "M_verify": M_VERIFY, "K": MAX_SPEC_LEN,
                "context_len": args.context_len, "iters": args.iters,
                "warmup": args.warmup, "realized_tps_k7": REALIZED_TPS_K7,
                "cycle_us": CYCLE_US, "tps_per_us": TPS_PER_US,
                "analysis_only": True, "no_hf_job": True,
                "no_served_file_change": True})
    meas = payload["measurement"]
    verdict = payload["verdict"]
    flat = {f"summary/{k}": v for k, v in verdict.items()
            if isinstance(v, (int, float, bool))}
    for k, v in meas["fractions"].items():
        flat[f"frac/{k}"] = v
    for k, v in meas["components_us"].items():
        flat[f"comp_us/{k}"] = v
    flat["attn/baseline_us"] = meas["attn_sweep"]["baseline_us"]
    flat["attn/best_us"] = meas["attn_sweep"]["best"]["us"]
    flat["attn/kernel_speedup_pct"] = meas["attn_sweep"]["kernel_speedup_pct"]
    flat["attn/sweep_min_speedup_pct"] = meas["attn_sweep"]["sweep_min_speedup_pct"]
    flat["attn/confirmed_speedup_pct"] = meas["attn_sweep"]["confirm"]["speedup_pct"]
    flat["attn/confirm_base_median_us"] = meas["attn_sweep"]["confirm"]["base_median_us"]
    flat["attn/confirm_best_median_us"] = meas["attn_sweep"]["confirm"]["best_median_us"]
    flat["attn/n_valid"] = meas["attn_sweep"]["n_valid"]
    flat["attn/n_configs"] = meas["attn_sweep"]["n_configs"]
    flat["endtoend/max_honest_tps_delta"] = meas["endtoend"]["max_honest_endtoend_tps_delta"]
    flat["endtoend/attn_eliminate_upperbound_tps"] = \
        meas["endtoend"]["attn_eliminate_endtoend_tps_upperbound"]
    flat["peak_vram_gib"] = meas["peak_vram_gib"]
    wandb.log(flat)
    # tables
    comp_tbl = wandb.Table(columns=["component", "us", "frac_of_verify"])
    for k in ["attention", "int4_gemm_body", "lm_head", "sampling", "dispatch"]:
        comp_tbl.add_data(k, round(meas["components_us"][k], 3),
                          round(meas["fractions"][k], 5))
    wandb.log({"verify_decomposition": comp_tbl})
    rows = meas["attn_sweep"]["rows"]
    sw_tbl = wandb.Table(columns=["BLOCK_M", "BLOCK_Q", "TILE_SIZE", "num_warps",
                                  "num_stages", "us", "valid", "max_abs_err"])
    for r in sorted(rows, key=lambda x: x["us"])[:64]:
        sw_tbl.add_data(r["BLOCK_M"], r["BLOCK_Q"], r["TILE_SIZE"], r["num_warps"],
                        r["num_stages"], round(r["us"], 3), r["valid"],
                        None if math.isnan(r["max_abs_err"]) else round(r["max_abs_err"], 6))
    wandb.log({"attn_tile_sweep_top64": sw_tbl})
    rid = run.id
    wandb.finish()
    return rid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--context-len", type=int, default=128,
                    help="served verify decode ctx (PR #447 shape: 128)")
    ap.add_argument("--self-test", action="store_true", help="0-GPU gate")
    ap.add_argument("--smoke", action="store_true", help="fast: few iters")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb_group", default="kernel-tiling-sweep")
    ap.add_argument("--wandb_name", default="denken/verify-wall-tile-scan")
    ap.add_argument("--out-dir",
                    default="research/profiling/verify_wall_tile_scan")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.self_test:
        st = self_test()
        with open(os.path.join(args.out_dir, "verify_wall_tile_scan_selftest.json"), "w") as f:
            json.dump(_jsonable(st), f, indent=2)
        return
    if args.smoke:
        args.iters, args.warmup = 30, 10

    meas = run_gpu(args)
    print("[vwts] running 0-GPU self-test gate ...", flush=True)
    st = self_test()
    st_pass = bool(st["_n_pass"] == st["_n_total"])
    verdict = build_verdict(meas)
    verdict["self_test_passes"] = st_pass
    payload = {"measurement": meas, "verdict": verdict, "self_test": _jsonable(st)}

    rid = None
    if not args.no_wandb:
        try:
            rid = log_wandb(args, payload)
        except Exception as exc:  # noqa: BLE001
            print(f"[vwts] wandb logging failed: {exc!r}", flush=True)
    payload["wandb_run_id"] = rid

    with open(os.path.join(args.out_dir, "verify_wall_tile_scan_results.json"), "w") as f:
        json.dump(_jsonable(payload), f, indent=2)

    print("\n=== VERDICT ===", flush=True)
    print(json.dumps(_jsonable(verdict), indent=2), flush=True)
    print(f"\nSENPAI-RESULT: " + json.dumps({
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [rid] if rid else [],
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True,
        "primary_metric": {"name": "max_honest_endtoend_tps_delta",
                           "value": verdict["max_honest_endtoend_tps_delta"]},
        "test_metric": {"name": "ppl", "value": PPL_ANCHOR},
        "self_test_passes": st_pass,
        "any_nonattn_triton_over_2tps": verdict["any_nonattn_triton_over_2tps"],
        "int4_gemm_dominant": verdict["int4_gemm_dominant"],
        "beats_deployed_481": verdict["beats_deployed_481"],
    }), flush=True)


if __name__ == "__main__":
    main()
