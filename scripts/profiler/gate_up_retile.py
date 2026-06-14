#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""gate_up tile-shape re-tiling micro-bench + roofline (PR #130).

THE QUESTION
------------
denken #117 (MERGED, 🔴 RED) found the SplitK verify-GEMM lever physically caps at
1.56% net because the dominant fused-MLP `gate_up_proj` GEMM (in=2560 -> out=20480,
int4 W4A16 Marlin, **54% of verify time**) is **CTA-saturated**: at M=8 it tiles N
into 128-wide column blocks -> N/128 = **160 CTAs = exactly 2 full waves** on the
A10G's 80 SMs, so a K-reduction SplitK cannot add occupancy -> it manufactures ~0
extra bandwidth there. The named single ceiling-breaker: **re-tile `gate_up` so it is
NOT CTA-saturated** -- either (a) a tile whose CTA count leaves a remainder SplitK can
exploit, or (b) raise arithmetic intensity toward the A10G ridge (107; current AI~=28).

This is the ONLY un-capped path to lift the tree-free #123 491.8 ceiling toward 500,
and it also reduces the tree's M=32 verify step_time -> higher realized tree TPS.

WHAT THIS MEASURES (LOCAL micro-bench, no HF Job, no submission, lossless by
construction -- isolated GEMM/stream timing never touches the token stream):

  Part A -- deployed Marlin gate_up BASELINE (the bar). Synthetic int4 W4A16 gate_up
            GEMM (value-independent; Marlin time depends only on M,in,out,group), timed
            launch-free (CUDA-graph replay, the deployed serve mechanism), at M=8 AND
            M=32. Achieved GB/s, %HBM, AI, implied CTA count / wave quantization.
            Confirms denken #117's 160-CTA / 2-wave / AI~=28 characterization.

  Part B -- the PHYSICAL HBM streaming ceiling (the decisive, kernel-AGNOSTIC test). A
            pure read-stream of the gate_up weight byte volume (26.2 MB) swept over CTA
            count (40..1280 blocks = 0.5..16 waves). If achieved read BW plateaus by 160
            CTAs (2 waves) and does not rise with more blocks, then NO re-tile and NO
            SplitK can manufacture bandwidth on this GEMM -> denken #117's wall is hard.
            The (ceiling_BW - marlin_BW)/marlin_BW gap is the UPPER BOUND on any re-tile.

  Part C -- a TUNABLE Triton W4A16 GEMM tile sweep (the PR's literal ask: sweep
            {BLOCK_M, BLOCK_N, num_warps, num_stages, SPLIT_K}; BLOCK_K pinned to the
            group size for scale correctness). Tests the re-tile MECHANISM directly: does
            a smaller N-tile (more CTAs) + SplitK beat the 160-CTA config? Triton is
            ~15-25% below hand-tuned Marlin at small M, so this is read as a FRACTION of
            the Part-B ceiling (does ANY tile reach >79.2% HBM?), not raw-vs-Marlin.

  Part D -- the Marlin knobs actually exposed from Python (no recompile): SplitK reduce
            parallelism (workspace max_blocks_per_sm), use_atomic_add, use_fp32_reduce.
            Best vs the default = the realizable Marlin-path re-tile speedup.

  Part E -- project the best realized gate_up speedup through the committed decode budget
            (gate_up = 54% of verify, verify = 53% of M=8 decode; ship model #105/#123,
            tau folded into K_cal) -> tree-free official TPS vs the 481.53 bar / 491.8
            ceiling / 500 target, and tree M=32 vs fern #125's ~538. GREEN/AMBER/RED.

Greedy identity: a tile-shape change is a SHAPE change (not a precision change); it is
numerically lossless if the reduction result is unchanged. We verify the Triton kernel
bit-profile/output vs a reference dequant matmul; the Marlin path is byte-identical to
deployed. Inherits NO #114 divergence (orthogonal to the batch-invariance question).
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import statistics
import time

# Must be set before importing torch/vllm (container inherits host CUDA id; see
# verify_gemm_roofline.py / scale_palette_bw_probe.py / project_local_a10g_gpu_env).
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

# ---- A10G (AWS g5, GA102, sm_86) roofline ceilings -------------------------
A10G_HBM_GBS = 600.0
A10G_FP16_TENSOR_TFLOPS = 70.0
N_SM = 80
GPTQ_MARLIN_TILE = 16          # Marlin 16x16 mma tile; M-tile granularity
MARLIN_THREAD_N_SMALL = 128    # research #130: M<=16 -> thread_n=128 (N/128 CTAs)
MARLIN_THREAD_N_LARGE = 256    # research #130: M>16  -> thread_n=256 (N/256 CTAs)

# gate_up_proj fused MLP shape (in -> out), 42 text decoder layers (#68).
GATE_UP_IN = 2560
GATE_UP_OUT = 20480

# ---- ship-model anchors (committed; tree_free_500_ceiling.py / #117 / #123) --
GATE_UP_SHARE_OF_VERIFY = 0.54   # #117: gate_up = 54% of verify-GEMM time
VERIFY_FRAC_M8 = 0.53            # #30/#105: verify-GEMM = 53% of M=8 decode step
FRONTIER_OFFICIAL = 481.53       # lawine #52 official best (private-verified VALID)
TREE_FREE_CEILING_123 = 491.8    # denken #123 central tree-free ceiling (SplitK wall)
TARGET_OFFICIAL = 500.0          # theykk human directive
TREE_SUPPLY_125 = 538.0          # fern #125 tree supply at W*=M=32/d9 (PR-stated)
TAU_OFFICIAL_MULT = 1.06019      # lawine #99 local->official (folded into K_cal)


# --------------------------------------------------------------------------- #
# launch-free timing (median-of-repeats CUDA-graph replay; from #110)
# --------------------------------------------------------------------------- #
def graph_time(fn, iters, warmup, repeats):
    """Median/min/max launch-free per-call ms of fn() via CUDA-graph replay."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.inference_mode():
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(g):
        fn()
    for _ in range(max(10, warmup)):
        g.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(repeats):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        out.append(e0.elapsed_time(e1) / iters)
    del g
    return statistics.median(out), min(out), max(out)


def burn_in(fn, seconds=4.0):
    """Drive the GPU to steady state before any timing.

    On a cold A10G the FIRST timed config reads ~8% slow (62.9us) vs its warm
    steady state (57.8us) even though SM clocks are pinned at 1710MHz the whole
    time -- a power/cache ramp transient, not frequency scaling (verified: 8
    consecutive warm graph_time calls land at 57.79-57.80us, <0.1% spread). If
    Part A is measured cold and Part D warm, the identical default config looks
    ~8% faster in D -> a phantom 're-tile speedup'. A single sustained warm-up
    here removes that artifact so every Part is measured in the same regime."""
    t0 = time.time()
    with torch.inference_mode():
        while time.time() - t0 < seconds:
            for _ in range(200):
                fn()
            torch.cuda.synchronize()


# --------------------------------------------------------------------------- #
# byte / roofline helpers
# --------------------------------------------------------------------------- #
def gemm_bytes(in_f, out_f, M, group, fp16_act=True):
    w = 0.5 * in_f * out_f
    n_groups = 1 if group == -1 else math.ceil(in_f / group)
    sc = 2.0 * out_f * n_groups
    act = 2.0 * M * in_f
    o = 2.0 * M * out_f
    return {"w": w, "scale": sc, "act": act, "out": o, "total": w + sc + act + o}


def roofline(in_f, out_f, M, group, t_us):
    t = t_us / 1e6
    b = gemm_bytes(in_f, out_f, M, group)
    flops = 2.0 * M * in_f * out_f
    gbs = b["total"] / t / 1e9
    return {
        "t_us": t_us, "gbytes_s": gbs, "pct_hbm": 100.0 * gbs / A10G_HBM_GBS,
        "gflops_s": flops / t / 1e9,
        "pct_compute": 100.0 * (flops / t / 1e9) / (A10G_FP16_TENSOR_TFLOPS * 1e3),
        "ai": flops / b["total"], "weight_MB": b["w"] / 1e6, "scale_MB": b["scale"] / 1e6,
        "total_MB": b["total"] / 1e6,
    }


def marlin_cta_count(out_f, M):
    """Implied Marlin CTA count + wave quantization (research #130 config table)."""
    thread_n = MARLIN_THREAD_N_SMALL if M <= 16 else MARLIN_THREAD_N_LARGE
    n_tiles = math.ceil(out_f / thread_n)
    m_tiles = math.ceil(M / GPTQ_MARLIN_TILE)
    ctas = n_tiles * m_tiles
    waves = ctas / N_SM
    return {"thread_n": thread_n, "n_tiles": n_tiles, "m_tiles": m_tiles, "ctas": ctas,
            "waves": waves, "cta_saturated": abs(waves - round(waves)) < 1e-9 and waves >= 1.0}


# =========================================================================== #
# Part A -- deployed Marlin gate_up baseline
# =========================================================================== #
def build_marlin_weight(in_f, out_f, group, dev):
    import vllm.model_executor.layers.quantization.utils.marlin_utils as mu
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
    from vllm.scalar_type import scalar_types
    w = (torch.randn(in_f, out_f, device=dev, dtype=torch.float16) * 0.05)
    w_ref, q_w, s, g_idx, sort_idx, _ = marlin_quantize(w, scalar_types.uint4b8, group, False)
    zp = torch.empty(0, dtype=torch.int32, device=dev)
    return {"q_w": q_w, "s": s, "zp": zp, "g_idx": g_idx, "sort_idx": sort_idx}


def marlin_call(packed, x, out_f, in_f, workspace, use_atomic_add, use_fp32_reduce):
    """Mirror apply_gptq_marlin_linear -> ops.marlin_gemm with forced knobs."""
    import vllm._custom_ops as ops
    from vllm.scalar_type import scalar_types
    return ops.marlin_gemm(
        x, None, packed["q_w"], None, packed["s"], None, None, packed["zp"],
        packed["g_idx"], packed["sort_idx"], workspace, scalar_types.uint4b8,
        size_m=x.shape[0], size_n=out_f, size_k=in_f, is_k_full=True,
        use_atomic_add=use_atomic_add, use_fp32_reduce=use_fp32_reduce, is_zp_float=False)


def part_a_marlin_baseline(in_f, out_f, groups, m_list, iters, warmup, repeats, dev):
    import vllm.model_executor.layers.quantization.utils.marlin_utils as mu
    ws = mu.marlin_make_workspace_new(dev)  # default max_blocks_per_sm=1 -> numel=80
    rows = []
    packed_by_g = {}
    burned = False
    for group in groups:
        if in_f % group != 0:
            continue
        packed = build_marlin_weight(in_f, out_f, group, dev)
        packed_by_g[group] = packed
        if not burned:  # warm the GPU to steady state BEFORE the first timed config
            xb = torch.randn(max(m_list), in_f, device=dev, dtype=torch.float16)
            burn_in(lambda: marlin_call(packed, xb, out_f, in_f, ws, False, True))
            burned = True
        for M in m_list:
            x = torch.randn(M, in_f, device=dev, dtype=torch.float16)
            fn = lambda packed=packed, x=x: marlin_call(packed, x, out_f, in_f, ws, False, True)
            med, lo, hi = graph_time(fn, iters, warmup, repeats)
            rm = roofline(in_f, out_f, M, group, med * 1000.0)
            rm.update({"M": M, "group": group, "t_us_min": lo * 1000.0, "t_us_max": hi * 1000.0,
                       **{f"cta_{k}": v for k, v in marlin_cta_count(out_f, M).items()}})
            rows.append(rm)
            print(f"[A] marlin g={group:4d} M={M:2d}: {rm['t_us']:7.1f}us  {rm['gbytes_s']:5.0f}GB/s "
                  f"({rm['pct_hbm']:4.1f}%HBM)  AI={rm['ai']:5.1f}  CTAs={rm['cta_ctas']:4d} "
                  f"({rm['cta_waves']:.2f}w, sat={rm['cta_cta_saturated']})", flush=True)
    return rows, ws, packed_by_g


# =========================================================================== #
# Part B -- physical HBM streaming ceiling (kernel-agnostic wall)
# =========================================================================== #
@triton.jit
def _stream_read_kernel(x_ptr, out_ptr, n_elem, ELEMS_PER_PROG: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * ELEMS_PER_PROG
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for off in range(0, ELEMS_PER_PROG, BLOCK):
        idx = base + off + tl.arange(0, BLOCK)
        acc += tl.load(x_ptr + idx, mask=idx < n_elem, other=0.0)
    tl.store(out_ptr + pid, tl.sum(acc))


def part_b_streaming_ceiling(weight_bytes, grid_sweep, block_elems, iters, warmup, repeats, dev):
    n_elem = int(weight_bytes // 4)                  # stream the 26.2 MB weight as fp32
    rows = []
    for G in grid_sweep:
        epp = math.ceil(n_elem / G)
        epp = ((epp + block_elems - 1) // block_elems) * block_elems  # pad to BLOCK
        n_pad = G * epp
        x = torch.randn(n_pad, device=dev, dtype=torch.float32)
        out = torch.zeros(G, device=dev, dtype=torch.float32)
        fn = lambda x=x, out=out, epp=epp: _stream_read_kernel[(G,)](
            x, out, x.numel(), ELEMS_PER_PROG=epp, BLOCK=block_elems)
        med, lo, hi = graph_time(fn, iters, warmup, repeats)
        gbs = (n_pad * 4) / (med / 1e3) / 1e9
        waves = G / N_SM
        rows.append({"grid": G, "waves": waves, "t_us": med * 1000.0, "gbytes_s": gbs,
                     "pct_hbm": 100.0 * gbs / A10G_HBM_GBS, "bytes_MB": n_pad * 4 / 1e6})
        print(f"[B] stream grid={G:5d} ({waves:5.2f}w): {med*1000:7.1f}us  {gbs:5.0f}GB/s "
              f"({100.0*gbs/A10G_HBM_GBS:4.1f}%HBM)", flush=True)
        del x, out
    # torch reference ceilings (well-optimized; sanity)
    big = torch.randn(n_elem, device=dev, dtype=torch.float32)
    fn_sum = lambda: big.sum()
    med_s, _, _ = graph_time(fn_sum, iters, warmup, repeats)
    torch_sum_gbs = (n_elem * 4) / (med_s / 1e3) / 1e9
    dst = torch.empty_like(big)
    fn_cp = lambda: dst.copy_(big)
    med_c, _, _ = graph_time(fn_cp, iters, warmup, repeats)
    torch_copy_gbs = (n_elem * 4 * 2) / (med_c / 1e3) / 1e9   # read+write
    del big, dst
    best = max(rows, key=lambda r: r["gbytes_s"])
    return {"rows": rows, "best_stream_gbs": best["gbytes_s"], "best_stream_pct_hbm": best["pct_hbm"],
            "best_grid": best["grid"], "torch_sum_gbs": torch_sum_gbs, "torch_copy_gbs": torch_copy_gbs}


# =========================================================================== #
# Part C -- tunable Triton W4A16 GEMM tile sweep (BLOCK_K pinned to group=128)
# =========================================================================== #
@triton.jit
def _w4a16_kernel(
    a_ptr, qw_ptr, s_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_qk, stride_qn, stride_sk, stride_sn, stride_cm, stride_cn,
    GROUP: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SPLIT_K: tl.constexpr,
):
    # BLOCK_K == GROUP (one group per K-chunk). qw is [K//8, N] int32 (8 int4 / int32
    # along K). 8-dot-along-K unpack: nibble j contributes the k = kbase + i*8 + j cols.
    pid = tl.program_id(0)
    pid_sk = tl.program_id(1)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    BK8: tl.constexpr = GROUP // 8
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_groups = tl.cdiv(K, GROUP)
    for gi in range(pid_sk, n_groups, SPLIT_K):
        kbase = gi * GROUP
        offs_k8 = (kbase // 8) + tl.arange(0, BK8)
        qw = tl.load(qw_ptr + offs_k8[:, None] * stride_qk + offs_n[None, :] * stride_qn,
                     mask=offs_n[None, :] < N, other=0)
        s = tl.load(s_ptr + gi * stride_sk + offs_n * stride_sn, mask=offs_n < N, other=0.0)
        s = s.to(tl.float32)
        for j in range(8):
            q_j = ((qw >> (4 * j)) & 0xF).to(tl.float32) - 8.0     # [BK8, BLOCK_N]
            w_j = (q_j * s[None, :]).to(tl.float16)
            offs_ak = kbase + tl.arange(0, BK8) * 8 + j
            a_j = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_ak[None, :] * stride_ak,
                          mask=offs_m[:, None] < M, other=0.0).to(tl.float16)
            acc += tl.dot(a_j, w_j)
    c = acc.to(tl.float16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=mask)
    else:
        tl.atomic_add(c_ptrs, acc, mask=mask)   # fp32 reduce across split-k blocks


def _pack_int4_k(w_fp16, group):
    """Pack [K,N] fp16 -> (qw [K//8,N] int32, scale [K//G,N] fp16). uint4b8 (bias 8)."""
    K, N = w_fp16.shape
    ng = K // group
    wg = w_fp16.reshape(ng, group, N)
    scale = (wg.abs().amax(dim=1) / 7.0).clamp_min(1e-8)               # [ng, N]
    q = torch.round(wg / scale[:, None, :]).clamp(-8, 7).to(torch.int32) + 8   # [ng,group,N] in [0,15]
    q = q.reshape(K, N)
    qw = torch.zeros(K // 8, N, dtype=torch.int32, device=w_fp16.device)
    for j in range(8):
        qw |= (q[j::8, :] & 0xF) << (4 * j)
    return qw, scale.to(torch.float16)


def _triton_gemm(a, qw, scale, N, K, group, BLOCK_M, BLOCK_N, SPLIT_K, num_warps, num_stages):
    M = a.shape[0]
    c = torch.zeros(M, N, device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), SPLIT_K)
    _w4a16_kernel[grid](
        a, qw, scale, c, M, N, K,
        a.stride(0), a.stride(1), qw.stride(0), qw.stride(1),
        scale.stride(0), scale.stride(1), c.stride(0), c.stride(1),
        GROUP=group, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, SPLIT_K=SPLIT_K,
        num_warps=num_warps, num_stages=num_stages)
    return c


def part_c_triton_sweep(in_f, out_f, group, m_list, configs, iters, warmup, repeats, dev,
                        marlin_us_by_m):
    # build reference weight + packing
    w = (torch.randn(in_f, out_f, device=dev, dtype=torch.float16) * 0.05)
    qw, scale = _pack_int4_k(w, group)
    # dequant reference for correctness
    ng = in_f // group
    q = torch.zeros(in_f, out_f, dtype=torch.int32, device=dev)
    for j in range(8):
        q[j::8, :] = (qw >> (4 * j)) & 0xF
    w_deq = ((q.reshape(ng, group, out_f).to(torch.float16) - 8.0) * scale[:, None, :]).reshape(in_f, out_f)
    results = {}
    for M in m_list:
        a = torch.randn(M, in_f, device=dev, dtype=torch.float16)
        ref = (a.to(torch.float32) @ w_deq.to(torch.float32))
        rows = []
        best = None
        for (BM, BN, SK, nw, ns) in configs:
            if M > BM and triton.cdiv(M, BM) > 1 and BM < 16:
                continue
            try:
                c = _triton_gemm(a, qw, scale, out_f, in_f, group, BM, BN, SK, nw, ns)
                rel = (c.to(torch.float32) - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
                ok = rel < 0.08    # int4 dequant + fp16 accum tolerance
                fn = lambda a=a, BM=BM, BN=BN, SK=SK, nw=nw, ns=ns: _triton_gemm(
                    a, qw, scale, out_f, in_f, group, BM, BN, SK, nw, ns)
                med, lo, hi = graph_time(fn, iters, warmup, repeats)
                rm = roofline(in_f, out_f, M, group, med * 1000.0)
                n_tiles = triton.cdiv(out_f, BN)
                m_tiles = triton.cdiv(M, BM)
                ctas = n_tiles * m_tiles * SK
                rec = {"BLOCK_M": BM, "BLOCK_N": BN, "SPLIT_K": SK, "num_warps": nw,
                       "num_stages": ns, "ctas": ctas, "waves": ctas / N_SM, "rel_err": rel,
                       "correct": ok, "t_us": rm["t_us"], "gbytes_s": rm["gbytes_s"],
                       "pct_hbm": rm["pct_hbm"]}
                rows.append(rec)
                if ok and (best is None or rm["t_us"] < best["t_us"]):
                    best = rec
            except Exception as exc:  # noqa: BLE001
                rows.append({"BLOCK_M": BM, "BLOCK_N": BN, "SPLIT_K": SK, "num_warps": nw,
                             "num_stages": ns, "error": repr(exc)[:120]})
        mu = marlin_us_by_m.get(M)
        if best is not None:
            print(f"[C] triton M={M:2d} BEST: BM={best['BLOCK_M']} BN={best['BLOCK_N']} "
                  f"SK={best['SPLIT_K']} w{best['num_warps']} s{best['num_stages']}: "
                  f"{best['t_us']:7.1f}us {best['gbytes_s']:5.0f}GB/s ({best['pct_hbm']:4.1f}%HBM) "
                  f"CTAs={best['ctas']} | marlin {mu:.1f}us", flush=True)
        results[M] = {"rows": rows, "best": best, "marlin_us": mu}
    del w, qw, scale, w_deq, q
    gc.collect(); torch.cuda.empty_cache()
    return results


# =========================================================================== #
# Part D -- exposed Marlin knobs (no recompile)
# =========================================================================== #
def part_d_marlin_knobs(in_f, out_f, group, m_list, packed, iters, warmup, repeats, dev,
                        marlin_us_by_m):
    import vllm.model_executor.layers.quantization.utils.marlin_utils as mu
    knobs = [("default", 1, False, True)]
    for mbps in (2, 4):
        knobs.append((f"blocks_per_sm={mbps}", mbps, False, True))
    knobs.append(("atomic_add", 1, True, True))
    knobs.append(("fp16_reduce", 1, False, False))
    results = {}
    for M in m_list:
        x = torch.randn(M, in_f, device=dev, dtype=torch.float16)
        base_us = marlin_us_by_m.get(M)   # Part-A warm baseline (cross-check only)
        rows = []
        best = None
        default_us = None
        for (name, mbps, atomic, fp32r) in knobs:
            ws = mu.marlin_make_workspace_new(dev, max_blocks_per_sm=mbps)
            try:
                fn = lambda x=x, ws=ws, atomic=atomic, fp32r=fp32r: marlin_call(
                    packed, x, out_f, in_f, ws, atomic, fp32r)
                med, lo, hi = graph_time(fn, iters, warmup, repeats)
                t_us = med * 1000.0
                if name == "default":
                    default_us = t_us
                # Knob speedup is measured vs the in-loop default (same warm regime,
                # same Part) so it isolates the KNOB, immune to any Part-to-Part drift.
                ref_us = default_us if default_us else base_us
                spd = 100.0 * (ref_us - t_us) / ref_us if ref_us else 0.0
                spd_vs_a = 100.0 * (base_us - t_us) / base_us if base_us else 0.0
                rec = {"knob": name, "max_blocks_per_sm": mbps, "use_atomic_add": atomic,
                       "use_fp32_reduce": fp32r, "t_us": t_us, "speedup_pct": spd,
                       "speedup_vs_partA_pct": spd_vs_a}
                rows.append(rec)
                if best is None or rec["t_us"] < best["t_us"]:
                    best = rec
                print(f"[D] marlin M={M:2d} {name:18s}: {t_us:7.1f}us  ({spd:+.2f}% vs in-loop default"
                      f", {spd_vs_a:+.2f}% vs PartA)", flush=True)
            except Exception as exc:  # noqa: BLE001
                rows.append({"knob": name, "error": repr(exc)[:120]})
        results[M] = {"rows": rows, "best": best, "base_us": base_us, "default_us": default_us}
    return results


# =========================================================================== #
# Part E -- project through the decode budget (ship model #105/#123)
# =========================================================================== #
def _load_ship_model():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "tree_free_500_ceiling.py")
    spec = importlib.util.spec_from_file_location("tree_free_500_ceiling", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def part_e_project(delta_m8, delta_m32, splitk_headroom_m8):
    """delta_* = realized gate_up per-step speedup (fraction). splitk_headroom_m8 =
    optimistic UPPER-BOUND speedup from hitting the Part-B streaming ceiling."""
    out = {"delta_m8": delta_m8, "delta_m32": delta_m32,
           "splitk_headroom_m8": splitk_headroom_m8}

    def treefree_alone(delta):
        step_red = VERIFY_FRAC_M8 * GATE_UP_SHARE_OF_VERIFY * delta   # absolute step units
        return FRONTIER_OFFICIAL / (1.0 - step_red)

    # tree-free: re-tile applied to the deployed 481.53 bar (clean, single-lever)
    out["treefree_alone_realized"] = treefree_alone(delta_m8)
    out["treefree_alone_ceiling"] = treefree_alone(splitk_headroom_m8)   # if re-tile hit the wall

    # tree-free stacked on the #123 491.8 build-complete ceiling (compose model)
    try:
        sm = _load_ship_model()
        p = sm.point("central")
        base = sm.compose(0.0156, p)               # denken #123 central (SplitK net 1.56%)
        step0 = base["step_time"]
        d_step = VERIFY_FRAC_M8 * GATE_UP_SHARE_OF_VERIFY * delta_m8
        stacked = base["official_tps"] * step0 / (step0 - d_step)
        out["treefree_stacked_123_base"] = base["official_tps"]
        out["treefree_stacked_123_with_retile"] = stacked
        d_step_c = VERIFY_FRAC_M8 * GATE_UP_SHARE_OF_VERIFY * splitk_headroom_m8
        out["treefree_stacked_123_ceiling"] = base["official_tps"] * step0 / (step0 - d_step_c)
    except Exception as exc:  # noqa: BLE001
        out["ship_model_error"] = repr(exc)[:160]

    # tree M=32: gate_up re-tile reduces the tree's verify step_time(W*). Apply the same
    # gate_up=54%-of-verify share to the tree's verify slice. The tree's verify share of
    # its M=32 step differs from M=8's 0.53; we use 0.53 as a documented approximation
    # (the M=32 verify is a LARGER share, so this is a conservative under-statement).
    step_red_tree = VERIFY_FRAC_M8 * GATE_UP_SHARE_OF_VERIFY * delta_m32
    out["tree_125_with_retile"] = TREE_SUPPLY_125 / (1.0 - step_red_tree)
    out["tree_125_base"] = TREE_SUPPLY_125
    return out


def decide_gate(primary_speedup_pct, proj):
    tf_real = proj.get("treefree_stacked_123_with_retile", proj["treefree_alone_realized"])
    tree_real = proj["tree_125_with_retile"]
    tree_gain = tree_real - TREE_SUPPLY_125
    if tf_real >= TARGET_OFFICIAL or tree_gain >= 5.0:
        return "GREEN", (f"re-tile projects tree-free {tf_real:.1f}>=500 OR tree +{tree_gain:.1f}>=+5 "
                         f"-> bank the tile shape, hand to land #71 / tree-free stack")
    if primary_speedup_pct > 0.3:    # a real, measurable speedup below the 500-closer
        return "AMBER", (f"measurable +{primary_speedup_pct:.2f}% gate_up speedup that composes "
                         f"(tree-free {tf_real:.1f}<500, tree +{tree_gain:.1f}<+5) -> bank as stacking lever")
    return "RED", ("every tile shape is <= the deployed Marlin tiling within noise AND the streaming "
                   "ceiling shows no exploitable BW headroom -> gate_up is re-tile-proof at our shapes; "
                   "denken #117's CTA-saturation ceiling is confirmed HARD")


# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="128,32", help="marlin group sizes to bench (deployed ambiguity)")
    ap.add_argument("--m-list", default="8,32")
    ap.add_argument("--grid-sweep", default="40,80,120,160,200,240,320,480,640,1280")
    ap.add_argument("--stream-block", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--skip-triton", action="store_true")
    ap.add_argument("--output", default="research/gate_up_retile/gate_up_retile.json")
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="gate-up-retile")
    ap.add_argument("--wandb_name", default="wirbel/gate-up-retile")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    groups = [int(x) for x in args.groups.split(",") if x.strip()]
    m_list = [int(x) for x in args.m_list.split(",") if x.strip()]
    grid_sweep = [int(x) for x in args.grid_sweep.split(",") if x.strip()]
    dev = torch.device("cuda")
    print(f"[retile] A10G {torch.cuda.get_device_name(0)} SMs={N_SM} | gate_up {GATE_UP_IN}->{GATE_UP_OUT} "
          f"| groups={groups} M={m_list}", flush=True)

    # Part A
    print("\n[retile] === Part A: deployed Marlin gate_up baseline ===", flush=True)
    a_rows, ws, packed_by_g = part_a_marlin_baseline(
        GATE_UP_IN, GATE_UP_OUT, groups, m_list, args.iters, args.warmup, args.repeats, dev)
    # Deployed checkpoint group reconciled to g=32: #68's 62.9us = 475GB/s/79% byte
    # count (29.86MB) matches ONLY g=32's scale volume (3.28MB); g=128 would read
    # 27.40MB -> 436GB/s at the same time. Confirmed: warm g=32 M=8 = 79.5%HBM/AI=28.
    deployed_group = 32 if 32 in packed_by_g else groups[0]
    marlin_us_by_m = {r["M"]: r["t_us"] for r in a_rows if r["group"] == deployed_group}
    marlin_bw_by_m = {r["M"]: r["gbytes_s"] for r in a_rows if r["group"] == deployed_group}

    # Part B
    print("\n[retile] === Part B: physical HBM streaming ceiling ===", flush=True)
    wbytes = 0.5 * GATE_UP_IN * GATE_UP_OUT
    part_b = part_b_streaming_ceiling(wbytes, grid_sweep, args.stream_block,
                                      args.iters, args.warmup, args.repeats, dev)
    # headroom = how much faster a hypothetical re-tile could read the weight vs Marlin
    headroom_m8 = max(0.0, (part_b["best_stream_gbs"] - marlin_bw_by_m.get(8, 1)) / marlin_bw_by_m.get(8, 1))
    print(f"[B] streaming ceiling {part_b['best_stream_gbs']:.0f}GB/s ({part_b['best_stream_pct_hbm']:.1f}%HBM) "
          f"@grid={part_b['best_grid']} | torch.sum {part_b['torch_sum_gbs']:.0f} copy {part_b['torch_copy_gbs']:.0f}"
          f" | marlin M=8 {marlin_bw_by_m.get(8,0):.0f}GB/s -> headroom {100*headroom_m8:.2f}%", flush=True)

    # Part C
    part_c = {}
    if not args.skip_triton:
        print("\n[retile] === Part C: tunable Triton W4A16 tile sweep ===", flush=True)
        BMs, BNs, SKs = [16, 32, 64], [64, 128, 256, 512], [1, 2, 4, 8]
        configs = [(bm, bn, sk, nw, ns)
                   for bm in BMs for bn in BNs for sk in SKs
                   for nw in (4, 8) for ns in (3, 4)]
        # Triton kernel contracts tl.dot over GROUP//8; tensor-core tl.dot needs K>=16
        # -> GROUP>=128. Part C builds its OWN weight and is a group-agnostic MECHANISM
        # probe (does smaller-N-tile + SplitK lift %HBM above the wall?), not a deployed
        # replica, so pin GROUP=128 regardless of the deployed g=32.
        tg = 128
        part_c = part_c_triton_sweep(GATE_UP_IN, GATE_UP_OUT, tg, m_list, configs,
                                     args.iters, args.warmup, args.repeats, dev, marlin_us_by_m)

    # Part D
    print("\n[retile] === Part D: exposed Marlin knobs (no recompile) ===", flush=True)
    part_d = part_d_marlin_knobs(GATE_UP_IN, GATE_UP_OUT, deployed_group, m_list,
                                 packed_by_g[deployed_group], args.iters, args.warmup,
                                 args.repeats, dev, marlin_us_by_m)

    # ---- realized best gate_up speedup (the primary metric) -------------------
    def best_realized_speedup(M):
        base = marlin_us_by_m.get(M)
        cands = [0.0]
        d = part_d.get(M, {}).get("best")
        if d:
            cands.append(d["speedup_pct"])
        c = part_c.get(M, {}).get("best") if part_c else None
        if c and c.get("t_us") and base:
            cands.append(100.0 * (base - c["t_us"]) / base)   # triton vs marlin (usually negative)
        return max(cands), {"marlin_knob_best_pct": d["speedup_pct"] if d else None,
                            "triton_best_us": c["t_us"] if c else None}

    spd8, det8 = best_realized_speedup(8)
    spd32, det32 = best_realized_speedup(32)
    primary_speedup_pct = spd8        # M=8 = the tree-free / linear-MTP verify width
    delta_m8 = max(0.0, spd8) / 100.0
    delta_m32 = max(0.0, spd32) / 100.0

    # Part E
    print("\n[retile] === Part E: project through decode budget ===", flush=True)
    proj = part_e_project(delta_m8, delta_m32, headroom_m8)
    verdict, reason = decide_gate(primary_speedup_pct, proj)
    test_metric = proj.get("treefree_stacked_123_with_retile", proj["treefree_alone_realized"])

    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"\n[retile] ===== VERDICT: {verdict} =====", flush=True)
    print(f"  primary gate_up_retile_per_step_speedup_pct (M=8) = {primary_speedup_pct:+.3f}%", flush=True)
    print(f"  test    gate_up_retile_projected_official_tps      = {test_metric:.2f} (tree-free, stacked)", flush=True)
    print(f"  tree-free alone (on 481.53): {proj['treefree_alone_realized']:.2f} | "
          f"ceiling-if-hit-wall {proj['treefree_alone_ceiling']:.2f}", flush=True)
    print(f"  tree-free stacked on 491.8:  {proj.get('treefree_stacked_123_with_retile', float('nan')):.2f} | "
          f"ceiling {proj.get('treefree_stacked_123_ceiling', float('nan')):.2f}", flush=True)
    print(f"  tree M=32 (on {TREE_SUPPLY_125}): {proj['tree_125_with_retile']:.2f}", flush=True)
    print(f"  reason: {reason}", flush=True)
    print(f"  peak GPU mem: {peak_mem:.2f} GiB", flush=True)

    payload = {
        "config": {
            "device": torch.cuda.get_device_name(0), "n_sm": N_SM,
            "vllm": __import__("vllm").__version__, "torch": torch.__version__,
            "triton": triton.__version__, "gate_up_in": GATE_UP_IN, "gate_up_out": GATE_UP_OUT,
            "groups": groups, "m_list": m_list, "grid_sweep": grid_sweep,
            "iters": args.iters, "warmup": args.warmup, "repeats": args.repeats,
            "deployed_group": deployed_group, "peak_gpu_mem_gib": peak_mem,
            "gate_up_share_of_verify": GATE_UP_SHARE_OF_VERIFY, "verify_frac_m8": VERIFY_FRAC_M8,
            "note": "isolated Marlin/Triton/stream timing; synthetic value-independent weights; "
                    "launch-free CUDA-graph replay; no served-file/token-stream change; lossless.",
        },
        "part_a_marlin_baseline": a_rows,
        "part_b_streaming_ceiling": part_b,
        "part_c_triton_sweep": {str(M): part_c[M] for M in part_c} if part_c else {},
        "part_d_marlin_knobs": {str(M): part_d[M] for M in part_d},
        "streaming_headroom_m8": headroom_m8,
        "realized_speedup": {"m8_pct": spd8, "m32_pct": spd32, "m8_detail": det8, "m32_detail": det32},
        "projection": proj,
        "verdict": {"gate": verdict, "reason": reason,
                    "primary_gate_up_retile_per_step_speedup_pct": primary_speedup_pct,
                    "test_gate_up_retile_projected_official_tps": test_metric},
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(f"[retile] wrote {args.output}", flush=True)

    if not args.no_wandb:
        try:
            _log_wandb(args, payload)
        except Exception as exc:  # noqa: BLE001
            print(f"[retile] W&B logging failed: {exc!r}", flush=True)

    gc.collect(); torch.cuda.empty_cache()


def _log_wandb(args, payload):
    import wandb
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                     group=args.wandb_group, name=args.wandb_name, job_type="profiling",
                     config=payload["config"])
    # Part A table
    cols_a = ["group", "M", "t_us", "gbytes_s", "pct_hbm", "ai", "cta_ctas", "cta_waves",
              "cta_cta_saturated", "cta_thread_n"]
    ta = wandb.Table(columns=cols_a)
    for r in payload["part_a_marlin_baseline"]:
        ta.add_data(*[r.get(c) for c in cols_a])
    run.log({"marlin_baseline": ta})
    # Part B streaming line
    for r in payload["part_b_streaming_ceiling"]["rows"]:
        run.log({"stream_grid": r["grid"], "stream_waves": r["waves"], "stream_gbytes_s": r["gbytes_s"],
                 "stream_pct_hbm": r["pct_hbm"]})
    v = payload["verdict"]
    pr = payload["projection"]
    summary = {
        "gate_up_retile_per_step_speedup_pct": v["primary_gate_up_retile_per_step_speedup_pct"],
        "gate_up_retile_projected_official_tps": v["test_gate_up_retile_projected_official_tps"],
        "verdict_gate": v["gate"],
        "streaming_ceiling_gbs": payload["part_b_streaming_ceiling"]["best_stream_gbs"],
        "streaming_ceiling_pct_hbm": payload["part_b_streaming_ceiling"]["best_stream_pct_hbm"],
        "streaming_headroom_m8_pct": 100.0 * payload["streaming_headroom_m8"],
        "treefree_alone_realized": pr["treefree_alone_realized"],
        "treefree_stacked_123_with_retile": pr.get("treefree_stacked_123_with_retile"),
        "tree_125_with_retile": pr["tree_125_with_retile"],
        "realized_speedup_m8_pct": payload["realized_speedup"]["m8_pct"],
        "realized_speedup_m32_pct": payload["realized_speedup"]["m32_pct"],
        "peak_gpu_mem_gib": payload["config"]["peak_gpu_mem_gib"],
    }
    run.summary.update({k: val for k, val in summary.items() if val is not None})
    run.finish()
    print(f"[retile] W&B run: {run.url}", flush=True)


if __name__ == "__main__":
    main()
