#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #721 (stark) -- custom non-Marlin int4 g128 M=1 decode GEMV kernel (Triton).

LOCAL A10G (sm_86). analysis_only=1, official_tps=0, no_hf_job=1, fires=0.
NO served-file change, NO HF Job, NO --launch, NO submission.

The kernel: y[1,N] = x[1,K] @ W[K,N], W int4 W4A16 group_size=128 along K, bf16
per-group scales, symmetric (uint4b8: stored nibble in {0..15} -> signed {-8..7}).
dequant(k,n) = (nibble[k,n] - 8) * scale[k//128, n]; this EQUALS w_ref = w_q*scale,
so the kernel computes the SAME mathematical product as Marlin (differs only in fp
accumulation order). At M=1 this is a memory-bandwidth-bound GEMV.

Purpose-built for M=1: NO wasted MMA tile rows (true vector reduction, not tl.dot),
fine N-tiling for occupancy on the small body shapes (qkv/o) where Marlin under-
saturates (#602: qkv 59.1% / o 65.7% of read-peak vs gate_up 92.9% / down 86.7%).

Weight layout (pack-along-K, bandwidth-optimal -- each int32 read ONCE, 8 nibbles):
  qpacked[K//8, N] int32: qpacked[kp, n] packs k = kp*8 + j (j=0..7) in nibble j.
  scales[K//128, N] bf16. For k in [kp*8, kp*8+8) the group index is kp//16 (=GROUP_KP).
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # local A10G is index 0 (inherited =1 -> 0 GPUs)

import torch
import triton
import triton.language as tl

PACK = 8          # int4 per int32
GROUP_SIZE = 128  # quant group along K
GROUP_KP = GROUP_SIZE // PACK  # =16 packed-rows per scale group


# --------------------------------------------------------------------------- #
# packing                                                                      #
# --------------------------------------------------------------------------- #
def pack_int4_k8(w_q: torch.Tensor) -> torch.Tensor:
    """w_q [K,N] UNSIGNED nibble in [0,15] (uint4b8 storage; true value = (w_q-8)*scale)
    -> qpacked [K//8, N] int32 (8 K-rows / int32). The kernel applies the -8 bias."""
    K, N = w_q.shape
    assert K % PACK == 0, f"K={K} must be divisible by {PACK}"
    nib = w_q.to(torch.int32) & 0xF                # [K,N] in {0..15}, raw stored nibble
    nib = nib.reshape(K // PACK, PACK, N)          # [KP, 8, N]
    packed = torch.zeros(K // PACK, N, dtype=torch.int32, device=w_q.device)
    for j in range(PACK):
        packed |= (nib[:, j, :] << (4 * j))
    return packed.contiguous()


# --------------------------------------------------------------------------- #
# single-pass GEMV (full K reduction per program; one program-tile owns BLOCK_N)
# --------------------------------------------------------------------------- #
def _configs():
    cfgs = []
    for bn in (16, 32, 64, 128, 256):
        for bkp in (32, 64, 128):
            for w in (2, 4, 8):
                for s in (2, 3, 4):
                    cfgs.append(triton.Config(
                        {"BLOCK_N": bn, "BLOCK_KP": bkp}, num_warps=w, num_stages=s))
    return cfgs


@triton.autotune(configs=_configs(), key=["K", "N"])
@triton.jit
def _gemv_k8_kernel(
    x_ptr, qw_ptr, sc_ptr, y_ptr,
    K, N,
    GROUP_KP: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_KP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = offs_n < N
    acc = tl.zeros([BLOCK_N], tl.float32)
    KP = K // 8
    for kp0 in range(0, KP, BLOCK_KP):
        offs_kp = kp0 + tl.arange(0, BLOCK_KP)               # [BLOCK_KP]
        kpmask = offs_kp < KP
        qw = tl.load(qw_ptr + offs_kp[:, None] * N + offs_n[None, :],
                     mask=kpmask[:, None] & nmask[None, :], other=0)        # int32 [BKP,BN]
        grp = offs_kp // GROUP_KP
        sc = tl.load(sc_ptr + grp[:, None] * N + offs_n[None, :],
                     mask=kpmask[:, None] & nmask[None, :], other=0.0).to(tl.float32)
        for j in range(8):
            q = ((qw >> (4 * j)) & 0xF).to(tl.float32) - 8.0               # [BKP,BN]
            xj = tl.load(x_ptr + offs_kp * 8 + j, mask=kpmask, other=0.0).to(tl.float32)
            acc += tl.sum(xj[:, None] * q * sc, axis=0)
    tl.store(y_ptr + offs_n, acc, mask=nmask)


def gemv_int4_g128(x: torch.Tensor, qpacked: torch.Tensor, scales: torch.Tensor,
                   N: int, K: int, out_dtype=torch.bfloat16) -> torch.Tensor:
    """x [K] or [1,K] bf16; qpacked [K//8,N] int32; scales [K//128,N]. Returns y [1,N]."""
    x = x.reshape(-1).contiguous()
    y = torch.empty(N, dtype=torch.float32, device=x.device)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    _gemv_k8_kernel[grid](x, qpacked, scales, y, K, N, GROUP_KP)
    return y.to(out_dtype).reshape(1, N)


# --------------------------------------------------------------------------- #
# deterministic split-K GEMV (for small-N shapes that under-fill 80 SMs).
# pass 1: partials[SPLIT, N]; pass 2: fixed-order sum over SPLIT.
# --------------------------------------------------------------------------- #
@triton.jit
def _gemv_k8_splitk_kernel(
    x_ptr, qw_ptr, sc_ptr, part_ptr,
    K, N, SPLIT,
    GROUP_KP: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_KP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = offs_n < N
    acc = tl.zeros([BLOCK_N], tl.float32)
    KP = K // 8
    kp_per = tl.cdiv(KP, SPLIT)
    kp_start = pid_s * kp_per
    kp_end = tl.minimum(kp_start + kp_per, KP)
    for kp0 in range(kp_start, kp_end, BLOCK_KP):
        offs_kp = kp0 + tl.arange(0, BLOCK_KP)
        kpmask = offs_kp < kp_end
        qw = tl.load(qw_ptr + offs_kp[:, None] * N + offs_n[None, :],
                     mask=kpmask[:, None] & nmask[None, :], other=0)
        grp = offs_kp // GROUP_KP
        sc = tl.load(sc_ptr + grp[:, None] * N + offs_n[None, :],
                     mask=kpmask[:, None] & nmask[None, :], other=0.0).to(tl.float32)
        for j in range(8):
            q = ((qw >> (4 * j)) & 0xF).to(tl.float32) - 8.0
            xj = tl.load(x_ptr + offs_kp * 8 + j, mask=kpmask, other=0.0).to(tl.float32)
            acc += tl.sum(xj[:, None] * q * sc, axis=0)
    tl.store(part_ptr + pid_s * N + offs_n, acc, mask=nmask)


@triton.jit
def _reduce_partials_kernel(part_ptr, y_ptr, N, SPLIT, BLOCK_N: tl.constexpr):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = offs_n < N
    acc = tl.zeros([BLOCK_N], tl.float32)
    for s in range(SPLIT):                       # fixed order -> deterministic
        acc += tl.load(part_ptr + s * N + offs_n, mask=nmask, other=0.0)
    tl.store(y_ptr + offs_n, acc, mask=nmask)


def gemv_int4_g128_splitk(x, qpacked, scales, N, K, split=8,
                          block_n=64, block_kp=64, num_warps=4, num_stages=3,
                          out_dtype=torch.bfloat16):
    x = x.reshape(-1).contiguous()
    part = torch.empty(split * N, dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(N, block_n), split)
    _gemv_k8_splitk_kernel[grid](x, qpacked, scales, part, K, N, split,
                                 GROUP_KP, BLOCK_N=block_n, BLOCK_KP=block_kp,
                                 num_warps=num_warps, num_stages=num_stages)
    y = torch.empty(N, dtype=torch.float32, device=x.device)
    rgrid = (triton.cdiv(N, 256),)
    _reduce_partials_kernel[rgrid](part, y, N, split, BLOCK_N=256)
    return y.to(out_dtype).reshape(1, N)


# --------------------------------------------------------------------------- #
# self-test: correctness vs dequant reference                                  #
# --------------------------------------------------------------------------- #
def _self_test():
    from vllm.model_executor.layers.quantization.utils import marlin_utils_test as mt
    from vllm.scalar_type import scalar_types
    dev = "cuda:0"
    ok = True
    for (K, N) in [(256, 64), (2560, 3072), (2048, 2560), (10240, 256), (2560, 512)]:
        torch.manual_seed(0)
        w = (torch.randn(K, N, dtype=torch.bfloat16, device=dev) * 0.02)
        w_ref, w_q, w_s, _zp = mt.quantize_weights(w.float(), scalar_types.uint4b8, GROUP_SIZE,
                                                   zero_points=False)
        # reconstruct identity: w_ref == w_q * scale (per group)
        scales = w_s.to(torch.bfloat16).contiguous()       # [K//128, N]
        qpacked = pack_int4_k8(w_q)
        x = torch.randn(1, K, dtype=torch.bfloat16, device=dev) * 0.1
        # reference in fp32 (the mathematical product)
        y_ref = (x.float() @ w_ref.float()).reshape(N)
        y_tri = gemv_int4_g128(x, qpacked, scales, N, K, out_dtype=torch.float32).reshape(N)
        y_spk = gemv_int4_g128_splitk(x, qpacked, scales, N, K, split=8,
                                      out_dtype=torch.float32).reshape(N)
        rel = (y_tri - y_ref).abs().max().item() / (y_ref.abs().max().item() + 1e-9)
        rel_spk = (y_spk - y_ref).abs().max().item() / (y_ref.abs().max().item() + 1e-9)
        am_ref = int(y_ref.argmax()); am_tri = int(y_tri.argmax()); am_spk = int(y_spk.argmax())
        # identity (argmax/greedy token) is the #319-relevant gate; rel is fp-order noise
        shape_ok = (am_ref == am_tri == am_spk) and rel < 3e-3 and rel_spk < 3e-3
        ok &= shape_ok
        print(f"[self-test] K={K:5d} N={N:6d} rel_max={rel:.2e} rel_splitk={rel_spk:.2e} "
              f"argmax ref/tri/spk={am_ref}/{am_tri}/{am_spk} -> {'OK' if shape_ok else 'FAIL'}",
              flush=True)
    print(f"[self-test] {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _self_test() else 1)
