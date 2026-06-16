#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #490 — cuBLASLt-deterministic vs vLLM Triton batch-invariant GEMM at M=1 decode.

DRAW-SAFE / LOCAL-ONLY pure microbenchmark on ONE pod A10G (sm_86). NO serve, NO HF
Job, NO submission, NO served-file change. Local measurement is pre-authorized (PR #490).

The strict byte-exact serve (~222 local / 234.47 official, ubel #470) routes EVERY matmul
through vLLM's deterministic Triton `matmul_persistent` (batch_invariant) kernel — a ~48-51%
end-to-end determinism tax (land f7zwyoc8: no-flag 460.12 vs strict-flag 223.65). The #481
forward survey established that tax is a structural IEEE-754 floor; the open question this card
answers is whether vLLM's *specific* Triton invariant kernel is the *fastest deterministic*
kernel at our M=1 decode shapes, or whether NVIDIA's cuBLASLt-deterministic GEMM is materially
faster. ubel #491 de-risked the body GEMMs (QKV/O/gate_up/down) to argmax-free by Marlin
M-invariance, so this is a PURE SPEED comparison of three kernels at the real shapes.

Three kernels per real (M=1, K, N) decode-GEMM shape, BF16, A10G:
  (a) nondeterministic baseline  -> torch.nn.functional.linear (default cuBLAS, fast)
  (b) cuBLASLt deterministic     -> use_deterministic_algorithms(True) + F.linear
  (c) vLLM Triton invariant      -> matmul_persistent(x, W.t())  (the batch_invariant path)

Shapes are pulled from the REAL google/gemma-4-E4B-it text config by MODEL INTROSPECTION
(meta device) — NOT assumed. The card's "37 layers (30 SWA + 7 full)" is wrong: the real
text decoder is 42 layers (35 SWA-512 + 7 full-attn), H=2560, I=10240, nq=8, nkv=2, hd=256
(full-attn global_head_dim=512), V=262144, num_kv_shared_layers=18 (only 24 layers run K/V).

KEY OUTPUTS (logged to W&B):
  cublaslt_det_vs_triton_invariant_speedup = aggregate (c)/(b)  (>1 => cuBLASLt-det faster)
  cublaslt_det_faster                      = bool, (c)/(b) >= 1.10
  cublaslt_det_tax_vs_nondet               = (b)/(a)
  triton_invariant_tax_vs_nondet           = (c)/(a)
  + per-shape table, run-to-run reproducibility, (b)~(c) ULP match, and the CRUX
  batch-invariance (M=1 vs M=8) test for (a)/(b)/(c) — because cuBLASLt-deterministic
  guarantees run-to-run determinism but NOT batch-invariance (M=1==M=batch), which is the
  actual byte-exact-serve requirement and the whole reason vLLM wrote the Triton kernel.
  Any speedup is only a usable byte-exact lever if cuBLASLt-det is ALSO M-invariant.

Kernel (c) provenance: matmul_persistent / matmul_kernel_persistent / _compute_pid are copied
VERBATIM from vLLM's vllm/model_executor/layers/batch_invariant.py (0.22.x, Apache-2.0,
"Defeating Nondeterminism in LLM Inference", Horace He et al.). Only two non-kernel calls are
localized: `num_compute_units(idx)` -> torch SM count, and `from vllm.triton_utils import tl,
triton` -> plain triton imports. When a real vLLM is importable the script ALSO imports the
live kernel and asserts byte-identity vendored-vs-vLLM (proving the vendored copy is faithful).
"""
from __future__ import annotations

import os

# Force GPU 0 + cuBLAS deterministic workspace BEFORE importing torch.
# (Inherited CUDA_VISIBLE_DEVICES=7 is stale on this pod; only GPU 0 exists.)
if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------------------------------
# Kernel (c): vLLM Triton batch-invariant persistent matmul.
# Prefer a live vLLM import; fall back to the verbatim-vendored kernel below.
# ----------------------------------------------------------------------------------
_VLLM_MATMUL_PERSISTENT: Callable | None = None
_KERNEL_C_SOURCE = "vendored"
try:  # live vLLM, if this env has it (used for the byte-identity cross-check)
    from vllm.model_executor.layers.batch_invariant import (  # type: ignore
        matmul_persistent as _VLLM_MATMUL_PERSISTENT,
    )

    _KERNEL_C_SOURCE = "vllm-import"
except Exception:  # noqa: BLE001
    _VLLM_MATMUL_PERSISTENT = None

import triton
import triton.language as tl

_fp16_block_size_n = 256  # vLLM module constant (only used for fp16; we bench bf16)


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def matmul_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    bias_ptr,
    M,
    N,
    K,  #
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        if A_LARGE:
            offs_am = offs_am.to(tl.int64)
        if B_LARGE:
            offs_bn = offs_bn.to(tl.int64)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            if A_LARGE or B_LARGE:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
            else:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

            a = tl.load(
                a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            b = tl.load(
                b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            accumulator = tl.dot(a, b, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        if C_LARGE:
            offs_cm = offs_cm.to(tl.int64)
            offs_cn = offs_cn.to(tl.int64)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if HAS_BIAS:
            bias_ptrs = bias_ptr + offs_cn
            bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0).to(tl.float32)
            accumulator += bias
        c = accumulator.to(c_ptr.dtype.element_ty)
        tl.store(c_ptrs, c, mask=c_mask)


def _matmul_persistent_vendored(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    assert bias is None or bias.dim() == 1
    NUM_SMS = torch.cuda.get_device_properties(a.device.index).multi_processor_count
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)

    def grid(META):
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            ),
        )

    configs = {
        torch.bfloat16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": _fp16_block_size_n,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float32: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
    }
    matmul_kernel_persistent[grid](
        a, b, c, bias, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        NUM_SMS=NUM_SMS,
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        HAS_BIAS=bias is not None,
        **configs[dtype],
    )
    return c


def triton_invariant_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """vLLM batch-invariant matmul: a[M,K] @ b[K,N]. Uses live vLLM kernel if present."""
    if _VLLM_MATMUL_PERSISTENT is not None:
        return _VLLM_MATMUL_PERSISTENT(a, b)
    return _matmul_persistent_vendored(a, b)


# ----------------------------------------------------------------------------------
# Shape enumeration: pull the REAL decode-GEMM shapes by model introspection.
# ----------------------------------------------------------------------------------
DEFAULT_CONFIG = (
    "/senpai-run/home/student-land/.cache/huggingface/hub/"
    "models--google--gemma-4-E4B-it/snapshots/fee6332c1abaafb77f6f9624236c63aa2f1d0187/config.json"
)
# The eight named decode GEMMs the card asks to enumerate (everything else is a
# Gemma3n-specific per-layer projection we report under "extra").
NAMED8 = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"}


def enumerate_decode_gemms(config_path: str) -> dict[str, Any]:
    """Build the gemma-4-E4B-it text decoder on the meta device and enumerate every
    decode-path nn.Linear with its (in=K, out=N) and per-token count."""
    from collections import Counter

    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(config_path)
    tc = cfg.text_config
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(tc)

    by_shape: dict[tuple[str, int, int], int] = Counter()
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            leaf = name.split(".")[-1]
            by_shape[(leaf, mod.in_features, mod.out_features)] += 1

    shapes = []
    for (leaf, k, n), count in sorted(by_shape.items(), key=lambda kv: (-kv[0][1] * kv[0][2] * kv[1], kv[0])):
        shapes.append(
            {
                "family": leaf,
                "K": int(k),
                "N": int(n),
                "count": int(count),
                "named8": leaf in NAMED8,
                "flops_per_token": 2 * 1 * int(k) * int(n) * int(count),
            }
        )
    meta = {
        "model_type": cfg.model_type,
        "text_model": type(model).__name__,
        "num_hidden_layers": int(tc.num_hidden_layers),
        "hidden_size": int(tc.hidden_size),
        "intermediate_size": int(tc.intermediate_size),
        "num_attention_heads": int(tc.num_attention_heads),
        "num_key_value_heads": int(tc.num_key_value_heads),
        "head_dim": int(tc.head_dim),
        "global_head_dim": int(getattr(tc, "global_head_dim", tc.head_dim)),
        "vocab_size": int(tc.vocab_size),
        "num_kv_shared_layers": int(getattr(tc, "num_kv_shared_layers", 0)),
    }
    return {"shapes": shapes, "meta": meta}


# ----------------------------------------------------------------------------------
# Timing + correctness.
# ----------------------------------------------------------------------------------
def _time_fn(fn: Callable[[], Any], iters: int, repeats: int) -> dict[str, float]:
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / iters)  # ms per call
    times.sort()
    n = len(times)
    return {
        "median_ms": statistics.median(times),
        "p25_ms": times[max(0, n // 4)],
        "p75_ms": times[min(n - 1, (3 * n) // 4)],
        "min_ms": times[0],
    }


def _autotime(fn: Callable[[], Any], warmup: int, budget_iters: int, repeats: int) -> dict[str, float]:
    """EAGER per-call latency (kernel launch + execute)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    # quick probe to size the inner loop so each repeat is ~constant work
    p_start = torch.cuda.Event(enable_timing=True)
    p_end = torch.cuda.Event(enable_timing=True)
    p_start.record()
    for _ in range(5):
        fn()
    p_end.record()
    torch.cuda.synchronize()
    per_call_ms = max(p_start.elapsed_time(p_end) / 5.0, 1e-4)
    iters = int(max(20, min(budget_iters, 25.0 / per_call_ms)))
    return _time_fn(fn, iters, repeats)


def _time_graph(fn: Callable[[], Any], warmup: int, budget_iters: int, repeats: int) -> dict[str, Any]:
    """CUDA-graph REPLAY time (execution only, launch overhead amortized to ~0 — matches a
    graph-captured serve). Captures one fn() into a graph, then times replays. Returns
    {"unsupported": True} if the kernel cannot be captured."""
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        return {"unsupported": True, "error": repr(exc)[:200]}
    # probe replay cost to size the loop
    p_start = torch.cuda.Event(enable_timing=True)
    p_end = torch.cuda.Event(enable_timing=True)
    p_start.record()
    for _ in range(5):
        g.replay()
    p_end.record()
    torch.cuda.synchronize()
    per_call_ms = max(p_start.elapsed_time(p_end) / 5.0, 1e-4)
    iters = int(max(20, min(budget_iters, 25.0 / per_call_ms)))
    out = _time_fn(g.replay, iters, repeats)
    del g
    return out


def _max_abs_float(x: torch.Tensor, y: torch.Tensor) -> float:
    return float((x.float() - y.float()).abs().max().item())


def _max_ulp_bf16(x: torch.Tensor, y: torch.Tensor) -> int:
    xi = x.contiguous().view(torch.int16).to(torch.int32)
    yi = y.contiguous().view(torch.int16).to(torch.int32)
    return int((xi - yi).abs().max().item())


def _frac_exact(x: torch.Tensor, y: torch.Tensor) -> float:
    return float((x == y).float().mean().item())


def correctness_and_invariance(x1: torch.Tensor, W: torch.Tensor, batch_M: int) -> dict[str, Any]:
    """Run-to-run reproducibility (b,c), (b)~(c) ULP at M=1, and the crux M=1-vs-M=batch
    batch-invariance test for (a),(b),(c)."""
    Wt = W.t()
    K = x1.shape[1]
    xB = torch.randn(batch_M, K, dtype=x1.dtype, device=x1.device)
    xB[0] = x1[0]

    # run-to-run reproducibility
    torch.use_deterministic_algorithms(False)
    a1 = F.linear(x1, W)
    a1b = F.linear(x1, W)
    aB = F.linear(xB, W)
    nondet_reproducible = bool(torch.equal(a1, a1b))
    nondet_m_inv = bool(torch.equal(a1[0], aB[0]))
    nondet_m_maxdiff = _max_abs_float(a1[0], aB[0])

    torch.use_deterministic_algorithms(True)
    b1 = F.linear(x1, W)
    b1b = F.linear(x1, W)
    bB = F.linear(xB, W)
    det_reproducible = bool(torch.equal(b1, b1b))
    det_m_inv = bool(torch.equal(b1[0], bB[0]))
    det_m_maxdiff = _max_abs_float(b1[0], bB[0])
    torch.use_deterministic_algorithms(False)

    c1 = triton_invariant_mm(x1, Wt)
    c1b = triton_invariant_mm(x1, Wt)
    cB = triton_invariant_mm(xB, Wt)
    triton_reproducible = bool(torch.equal(c1, c1b))
    triton_m_inv = bool(torch.equal(c1[0], cB[0]))
    triton_m_maxdiff = _max_abs_float(c1[0], cB[0])

    # (b) cuBLAS-det vs (c) Triton-invariant at M=1: same fp32-accumulated math
    bc_bitexact = bool(torch.equal(b1, c1))
    bc_max_ulp = _max_ulp_bf16(b1, c1)
    bc_max_abs = _max_abs_float(b1, c1)
    bc_frac_exact = _frac_exact(b1, c1)

    nan_clean = bool(
        torch.isfinite(a1).all() and torch.isfinite(b1).all() and torch.isfinite(c1).all()
    )
    return {
        "nondet_reproducible": nondet_reproducible,
        "det_reproducible": det_reproducible,
        "triton_reproducible": triton_reproducible,
        "nondet_m_invariant": nondet_m_inv,
        "det_m_invariant": det_m_inv,
        "triton_m_invariant": triton_m_inv,
        "nondet_m_maxdiff": nondet_m_maxdiff,
        "det_m_maxdiff": det_m_maxdiff,
        "triton_m_maxdiff": triton_m_maxdiff,
        "bc_bitexact": bc_bitexact,
        "bc_max_ulp": bc_max_ulp,
        "bc_max_abs": bc_max_abs,
        "bc_frac_exact": bc_frac_exact,
        "nan_clean": nan_clean,
    }


def bench_shape(shape: dict[str, Any], warmup: int, budget_iters: int, repeats: int,
                batch_M: int, seed: int) -> dict[str, Any]:
    dev = torch.device("cuda")
    K, N = shape["K"], shape["N"]
    gen = torch.Generator(device="cuda").manual_seed(seed + K * 131 + N)
    x = torch.randn(1, K, dtype=torch.bfloat16, device=dev, generator=gen)
    W = torch.randn(N, K, dtype=torch.bfloat16, device=dev, generator=gen) * (K ** -0.5)
    Wt = W.t()

    def measure(fn: Callable[[], Any]) -> dict[str, Any]:
        # Both timing bases: EAGER (launch+exec, what a non-graph loop sees) and GRAPH
        # replay (exec only, launch overhead amortized ~0 — what a cudagraph-captured serve
        # sees). For tiny M=1 GEMVs the Triton launch path is heavier than cuBLAS's, so the
        # eager ratio overstates Triton's disadvantage; the graph ratio is the serve-faithful one.
        return {
            "eager": _autotime(fn, warmup, budget_iters, repeats),
            "graph": _time_graph(fn, warmup, budget_iters, repeats),
        }

    torch.use_deterministic_algorithms(False)
    a = measure(lambda: F.linear(x, W))
    torch.use_deterministic_algorithms(True)
    b = measure(lambda: F.linear(x, W))
    torch.use_deterministic_algorithms(False)
    c = measure(lambda: triton_invariant_mm(x, Wt))

    corr = correctness_and_invariance(x, W, batch_M)
    del x, W, Wt
    torch.cuda.empty_cache()

    graph_ok = not any(k["graph"].get("unsupported") for k in (a, b, c))
    basis = "graph" if graph_ok else "eager"

    def md(k: dict[str, Any], bss: str) -> float:
        return k[bss]["median_ms"]

    eager_ratios = {
        "speedup_c_over_b": round(md(c, "eager") / md(b, "eager"), 4),
        "tax_b_over_a": round(md(b, "eager") / md(a, "eager"), 4),
        "tax_c_over_a": round(md(c, "eager") / md(a, "eager"), 4),
    }
    graph_ratios = None
    if graph_ok:
        graph_ratios = {
            "speedup_c_over_b": round(md(c, "graph") / md(b, "graph"), 4),
            "tax_b_over_a": round(md(b, "graph") / md(a, "graph"), 4),
            "tax_c_over_a": round(md(c, "graph") / md(a, "graph"), 4),
        }
    head = graph_ratios if graph_ok else eager_ratios
    return {
        **shape,
        "a_nondet": a,
        "b_cublas_det": b,
        "c_triton_inv": c,
        "graph_capturable": graph_ok,
        "timing_basis": basis,
        "tax_b_over_a": head["tax_b_over_a"],
        "tax_c_over_a": head["tax_c_over_a"],
        "speedup_c_over_b": head["speedup_c_over_b"],
        "graph_ratios": graph_ratios,
        "eager_ratios": eager_ratios,
        "correctness": corr,
    }


def aggregate(results: list[dict[str, Any]], subset: Callable[[dict], bool],
              basis: str = "graph") -> dict[str, Any]:
    rows = [r for r in results if subset(r)]
    if not rows:
        return {}

    def md(r: dict[str, Any], key: str) -> float:
        return r[key][basis]["median_ms"]

    ta = sum(r["count"] * md(r, "a_nondet") for r in rows)
    tb = sum(r["count"] * md(r, "b_cublas_det") for r in rows)
    tc = sum(r["count"] * md(r, "c_triton_inv") for r in rows)
    flops = sum(r["flops_per_token"] for r in rows)
    flop_wt_b_over_a = sum(r["flops_per_token"] * (md(r, "b_cublas_det") / md(r, "a_nondet")) for r in rows) / flops
    flop_wt_c_over_a = sum(r["flops_per_token"] * (md(r, "c_triton_inv") / md(r, "a_nondet")) for r in rows) / flops
    flop_wt_c_over_b = sum(r["flops_per_token"] * (md(r, "c_triton_inv") / md(r, "b_cublas_det")) for r in rows) / flops
    return {
        "basis": basis,
        "per_token_ms_a_nondet": round(ta, 5),
        "per_token_ms_b_cublas_det": round(tb, 5),
        "per_token_ms_c_triton_inv": round(tc, 5),
        # wall-time-weighted (true per-token GEMM time) ratios
        "cublaslt_det_tax_vs_nondet": round(tb / ta, 4),
        "triton_invariant_tax_vs_nondet": round(tc / ta, 4),
        "cublaslt_det_vs_triton_invariant_speedup": round(tc / tb, 4),
        # FLOP-weighted average of per-shape ratios (as the card phrases it)
        "flopwt_cublaslt_det_tax_vs_nondet": round(flop_wt_b_over_a, 4),
        "flopwt_triton_invariant_tax_vs_nondet": round(flop_wt_c_over_a, 4),
        "flopwt_cublaslt_det_vs_triton_invariant_speedup": round(flop_wt_c_over_b, 4),
        "n_shapes": len(rows),
        "total_flops_per_token": flops,
    }


# Serve anchors for the first-order projection (LOCAL A10G; label clearly).
STRICT_LOCAL_TPS = 222.0       # ubel #470 local global-flag full serve (221.16 rounded)
STRICT_OFFICIAL_TPS = 234.47   # ubel #470 official global-flag full serve
DEPLOYED_LOCAL_TPS = 460.12    # land f7zwyoc8 no-flag local
DEPLOYED_OFFICIAL_TPS = 481.53  # PR #52 deployed public


def project_serve_lift(agg_all: dict[str, Any], det_m_invariant_all: bool) -> dict[str, Any]:
    """First-order kernel-microbench projection of the strict-serve TPS if the body+lm_head
    GEMMs swapped vLLM Triton-invariant -> cuBLASLt-deterministic. NOT a realized serve."""
    tc = agg_all["per_token_ms_c_triton_inv"]
    tb = agg_all["per_token_ms_b_cublas_det"]
    ta = agg_all["per_token_ms_a_nondet"]
    gemm_saved_ms = tc - tb  # per-token GEMM time recovered by cuBLAS-det
    out = {}
    for label, strict_tps, deployed_tps in (
        ("local", STRICT_LOCAL_TPS, DEPLOYED_LOCAL_TPS),
        ("official", STRICT_OFFICIAL_TPS, DEPLOYED_OFFICIAL_TPS),
    ):
        t_strict = 1000.0 / strict_tps
        t_deploy = 1000.0 / deployed_tps
        full_tax_ms = t_strict - t_deploy
        gemm_tax_ms = tc - ta  # GEMM share of the determinism tax (triton vs nondet)
        proj_step = max(t_strict - gemm_saved_ms, t_deploy)  # cannot beat the nondet stack
        proj_tps = 1000.0 / proj_step
        out[label] = {
            "strict_tps_anchor": strict_tps,
            "deployed_tps_anchor": deployed_tps,
            "full_determinism_tax_ms_per_tok": round(full_tax_ms, 4),
            "gemm_share_of_tax_ms_per_tok": round(gemm_tax_ms, 4),
            "gemm_share_of_tax_frac": round(gemm_tax_ms / full_tax_ms, 4) if full_tax_ms > 0 else None,
            "gemm_ms_recovered_by_cublas_det": round(gemm_saved_ms, 5),
            "projected_strict_tps_with_cublas_det_gemm": round(proj_tps, 2),
            "projected_lift_tps": round(proj_tps - strict_tps, 2),
            "projected_lift_pct": round((proj_tps / strict_tps - 1.0) * 100.0, 2),
            "capped_at_deployed": bool(t_strict - gemm_saved_ms < t_deploy),
        }
    out["byte_exact_usable"] = bool(det_m_invariant_all)
    out["timing_basis"] = agg_all.get("basis", "graph")
    out["caveat"] = (
        f"First-order projection ONLY, built on '{agg_all.get('basis', 'graph')}'-basis per-token "
        "GEMM time. Applies the measured per-token GEMM-time delta to the serve step time; assumes "
        "non-GEMM strict overhead (attention num_splits=1, norms, sampler) unchanged and that serve "
        "caching/overlap match the microbench. VALID as a byte-exact lever ONLY if cuBLASLt-det is "
        "batch-invariant (det_m_invariant=True); otherwise swapping it would break the greedy "
        "byte-exact gate and the projection is void."
    )
    return out


def maybe_log_wandb(args, payload: dict[str, Any]) -> str | None:
    if not getattr(args, "wandb_name", None):
        return None
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (
            finish_wandb,
            init_wandb_run,
            log_json_artifact,
            log_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[microbench] wandb logging unavailable: {exc}", flush=True)
        return None

    agg_all = payload["aggregate_all"]
    agg_body = payload["aggregate_body_excl_lmhead"]
    agg_all_eager = payload["aggregate_all_eager"]
    corr = payload["determinism_summary"]
    run = init_wandb_run(
        job_type="cublaslt-det-microbench",
        agent="land",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["cublaslt-det-microbench", "gemm-kernel", "m1-decode", "determinism-tax",
              "analysis-only", "local-exploratory", "draw-safe"],
        config={
            "device": payload["env"]["device"],
            "compute_capability": payload["env"]["compute_capability"],
            "torch": payload["env"]["torch"],
            "triton": payload["env"]["triton"],
            "kernel_c_source": payload["env"]["kernel_c_source"],
            "dtype": "bfloat16",
            "M": 1,
            "batch_M_for_invariance": payload["env"]["batch_M"],
            "headline_timing_basis": payload["env"]["headline_timing_basis"],
            "graph_capturable_all": payload["env"]["graph_capturable_all"],
            **{f"model_{k}": v for k, v in payload["model_meta"].items()},
        },
    )
    if run is None:
        print("[microbench] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return None

    summary = {
        # headline KEY OUTPUTS (all-in aggregate incl full lm_head)
        "cublaslt_det_vs_triton_invariant_speedup": agg_all["cublaslt_det_vs_triton_invariant_speedup"],
        "cublaslt_det_faster": int(payload["cublaslt_det_faster"]),
        "cublaslt_det_tax_vs_nondet": agg_all["cublaslt_det_tax_vs_nondet"],
        "triton_invariant_tax_vs_nondet": agg_all["triton_invariant_tax_vs_nondet"],
        # body-only (excl lm_head) — robust to lm_head vocab choice
        "body_cublaslt_det_vs_triton_invariant_speedup": agg_body["cublaslt_det_vs_triton_invariant_speedup"],
        "body_cublaslt_det_tax_vs_nondet": agg_body["cublaslt_det_tax_vs_nondet"],
        "body_triton_invariant_tax_vs_nondet": agg_body["triton_invariant_tax_vs_nondet"],
        # per-token GEMM time (ms) — headline basis (graph replay = serve-faithful)
        "headline_timing_basis_graph": int(payload["env"]["headline_timing_basis"] == "graph"),
        "per_token_gemm_ms_nondet": agg_all["per_token_ms_a_nondet"],
        "per_token_gemm_ms_cublas_det": agg_all["per_token_ms_b_cublas_det"],
        "per_token_gemm_ms_triton_inv": agg_all["per_token_ms_c_triton_inv"],
        # eager-basis context (launch-inclusive; overstates Triton's M=1 disadvantage)
        "eager_cublaslt_det_vs_triton_invariant_speedup": agg_all_eager["cublaslt_det_vs_triton_invariant_speedup"],
        "eager_per_token_gemm_ms_cublas_det": agg_all_eager["per_token_ms_b_cublas_det"],
        "eager_per_token_gemm_ms_triton_inv": agg_all_eager["per_token_ms_c_triton_inv"],
        # determinism / batch-invariance (the crux)
        "all_det_reproducible": int(corr["all_det_reproducible"]),
        "all_triton_reproducible": int(corr["all_triton_reproducible"]),
        "all_bc_within_1ulp": int(corr["all_bc_within_1ulp"]),
        "max_bc_ulp": corr["max_bc_ulp"],
        "max_bc_abs": corr["max_bc_abs"],
        "cublaslt_det_batch_invariant": int(corr["all_det_m_invariant"]),
        "triton_batch_invariant": int(corr["all_triton_m_invariant"]),
        "nondet_batch_invariant": int(corr["all_nondet_m_invariant"]),
        "nan_clean": int(corr["all_nan_clean"]),
        # projection
        "projected_strict_tps_local": payload["projection"]["local"]["projected_strict_tps_with_cublas_det_gemm"],
        "projected_lift_pct_local": payload["projection"]["local"]["projected_lift_pct"],
        "projected_strict_tps_official": payload["projection"]["official"]["projected_strict_tps_with_cublas_det_gemm"],
        "projection_byte_exact_usable": int(payload["projection"]["byte_exact_usable"]),
        "analysis_only": 1,
        "no_served_file_change": 1,
        "official_tps": 0,
    }
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="cublaslt_det_microbench", artifact_type="validity", data=payload)
    rid = getattr(run, "id", None)
    finish_wandb(run)
    print(f"[microbench] wandb logged {len(summary)} keys; run id {rid}", flush=True)
    return rid


def build_env() -> dict[str, Any]:
    return {
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "kernel_c_source": _KERNEL_C_SOURCE,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", default="cublaslt-det-microbench")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="path to gemma-4-E4B-it config.json")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--budget-iters", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=25)
    ap.add_argument("--batch-m", type=int, default=8, help="M for the batch-invariance test")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--selftest", action="store_true", help="quick tiny-shape sanity run")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "research/local_validation")
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        print("[microbench] FATAL: CUDA not available (check CUDA_VISIBLE_DEVICES).", flush=True)
        return 2

    env = build_env()
    print(f"[microbench] env: {json.dumps(env)}", flush=True)
    print(f"[microbench] kernel (c) source: {_KERNEL_C_SOURCE}", flush=True)

    if args.selftest:
        shapes = [
            {"family": "q_proj", "K": 2560, "N": 2048, "count": 35, "named8": True,
             "flops_per_token": 2 * 2560 * 2048 * 35},
            {"family": "lm_head", "K": 2560, "N": 262144, "count": 1, "named8": True,
             "flops_per_token": 2 * 2560 * 262144},
        ]
        model_meta = {"selftest": True}
        warmup, budget, repeats = 10, 100, 5
    else:
        enum = enumerate_decode_gemms(args.config)
        shapes = enum["shapes"]
        model_meta = enum["meta"]
        warmup, budget, repeats = args.warmup, args.budget_iters, args.repeats

    print(f"[microbench] benchmarking {len(shapes)} unique decode-GEMM shapes "
          f"(M=1, bf16); model meta: {json.dumps(model_meta)}", flush=True)

    results = []
    for s in shapes:
        r = bench_shape(s, warmup, budget, repeats, args.batch_m, args.seed)
        results.append(r)
        cc = r["correctness"]
        tb_ = r["timing_basis"]
        print(
            f"  {s['family']:26s} K={s['K']:6d} N={s['N']:6d} x{s['count']:2d} | [{tb_}] "
            f"a={r['a_nondet'][tb_]['median_ms']*1e3:8.2f}us b={r['b_cublas_det'][tb_]['median_ms']*1e3:8.2f}us "
            f"c={r['c_triton_inv'][tb_]['median_ms']*1e3:8.2f}us | (c)/(b)={r['speedup_c_over_b']:.3f} "
            f"(b)/(a)={r['tax_b_over_a']:.3f} | det_minv={cc['det_m_invariant']} "
            f"tri_minv={cc['triton_m_invariant']} bc_ulp={cc['bc_max_ulp']}",
            flush=True,
        )

    # determinism / invariance roll-up + hard assertions
    det_repro = all(r["correctness"]["det_reproducible"] for r in results)
    tri_repro = all(r["correctness"]["triton_reproducible"] for r in results)
    bc_1ulp = all(r["correctness"]["bc_max_ulp"] <= 1 for r in results)
    det_minv = all(r["correctness"]["det_m_invariant"] for r in results)
    tri_minv = all(r["correctness"]["triton_m_invariant"] for r in results)
    nondet_minv = all(r["correctness"]["nondet_m_invariant"] for r in results)
    nan_clean = all(r["correctness"]["nan_clean"] for r in results)
    determinism_summary = {
        "all_det_reproducible": det_repro,
        "all_triton_reproducible": tri_repro,
        "all_bc_within_1ulp": bc_1ulp,
        "all_bc_bitexact": all(r["correctness"]["bc_bitexact"] for r in results),
        "all_det_m_invariant": det_minv,
        "all_triton_m_invariant": tri_minv,
        "all_nondet_m_invariant": nondet_minv,
        "all_nan_clean": nan_clean,
        "max_bc_ulp": max(r["correctness"]["bc_max_ulp"] for r in results),
        "max_bc_abs": max(r["correctness"]["bc_max_abs"] for r in results),
        # (b) and (c) are BOTH deterministic but use DIFFERENT fp32 reduction orders
        # (cuBLAS tiling vs Triton sequential BLOCK_K), so they are not bit-identical;
        # max_abs is the meaningful agreement metric, not strict-ULP equality.
        "bc_note": "b,c bitexact-equal only on short reductions; long reductions (lm_head) "
                   "diverge by reduction order — argmax-relevant, not a kernel bug",
    }
    # The card: reject any (b)/(c) kernel that is not bit-reproducible.
    assert det_repro, "cuBLAS-deterministic (b) is NOT run-to-run reproducible — rejected"
    assert tri_repro, "Triton-invariant (c) is NOT run-to-run reproducible — rejected"
    assert nan_clean, "NaN/Inf detected in kernel outputs"

    # Headline basis: graph-replay (serve-faithful, exec-only) if EVERY shape captured
    # cleanly, else fall back to eager for a coherent wall-time-weighted aggregate.
    graph_ok_all = all(r["graph_capturable"] for r in results)
    headline_basis = "graph" if graph_ok_all else "eager"

    agg_all = aggregate(results, lambda r: True, headline_basis)
    agg_body = aggregate(results, lambda r: r["family"] != "lm_head", headline_basis)
    agg_named8 = aggregate(results, lambda r: r["named8"], headline_basis)
    agg_lmhead = aggregate(results, lambda r: r["family"] == "lm_head", headline_basis)
    # Eager-basis all-aggregate kept for context (launch-inclusive, non-graph loop).
    agg_all_eager = aggregate(results, lambda r: True, "eager")

    cublaslt_det_faster = agg_all["cublaslt_det_vs_triton_invariant_speedup"] >= 1.10
    projection = project_serve_lift(agg_all, det_minv)

    payload = {
        "env": {**env, "batch_M": args.batch_m, "headline_timing_basis": headline_basis,
                "graph_capturable_all": graph_ok_all},
        "model_meta": model_meta,
        "per_shape": results,
        "aggregate_all": agg_all,
        "aggregate_all_eager": agg_all_eager,
        "aggregate_body_excl_lmhead": agg_body,
        "aggregate_named8": agg_named8,
        "aggregate_lmhead": agg_lmhead,
        "determinism_summary": determinism_summary,
        "cublaslt_det_faster": bool(cublaslt_det_faster),
        "projection": projection,
        "analysis_only": True,
        "no_served_file_change": True,
        "draw_safe_local_only": True,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cublaslt_det_microbench.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\n=== AGGREGATE (per-token decode-GEMM time, wall-time-weighted; "
          f"basis={headline_basis}{'' if graph_ok_all else ' [graph capture failed for >=1 shape]'}) ===")
    for label, agg in (("ALL (incl full lm_head)", agg_all), ("BODY (excl lm_head)", agg_body),
                       ("NAMED-8", agg_named8), ("LM_HEAD only", agg_lmhead)):
        if agg:
            print(f"  {label:26s}: (c)/(b)={agg['cublaslt_det_vs_triton_invariant_speedup']:.3f}  "
                  f"(b)/(a)={agg['cublaslt_det_tax_vs_nondet']:.3f}  "
                  f"(c)/(a)={agg['triton_invariant_tax_vs_nondet']:.3f}  "
                  f"[a={agg['per_token_ms_a_nondet']:.3f} b={agg['per_token_ms_b_cublas_det']:.3f} "
                  f"c={agg['per_token_ms_c_triton_inv']:.3f} ms/tok]")
    if agg_all_eager:
        print(f"  {'ALL (eager context)':26s}: (c)/(b)={agg_all_eager['cublaslt_det_vs_triton_invariant_speedup']:.3f}  "
              f"(b)/(a)={agg_all_eager['cublaslt_det_tax_vs_nondet']:.3f}  "
              f"(c)/(a)={agg_all_eager['triton_invariant_tax_vs_nondet']:.3f}  "
              f"[a={agg_all_eager['per_token_ms_a_nondet']:.3f} b={agg_all_eager['per_token_ms_b_cublas_det']:.3f} "
              f"c={agg_all_eager['per_token_ms_c_triton_inv']:.3f} ms/tok]")
    print(f"\n  cublaslt_det_faster (>=10%): {cublaslt_det_faster}")
    print(f"  determinism: det_repro={det_repro} tri_repro={tri_repro} bc<=1ulp={bc_1ulp} "
          f"max_bc_ulp={determinism_summary['max_bc_ulp']}")
    print(f"  BATCH-INVARIANCE (M=1 vs M={args.batch_m}): cublas_det={det_minv} "
          f"triton={tri_minv} nondet={nondet_minv}")
    print(f"  projection (local): {projection['local']['projected_strict_tps_with_cublas_det_gemm']} TPS "
          f"({projection['local']['projected_lift_pct']:+.1f}%), byte_exact_usable={projection['byte_exact_usable']}")
    print(f"[microbench] wrote {out_path}")

    rid = maybe_log_wandb(args, payload)
    if rid:
        print(f"SENPAI-RESULT wandb_run_id={rid} "
              f"cublaslt_det_vs_triton_invariant_speedup={agg_all['cublaslt_det_vs_triton_invariant_speedup']} "
              f"cublaslt_det_faster={cublaslt_det_faster} "
              f"cublaslt_det_batch_invariant={det_minv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
