#!/usr/bin/env python
"""PR #736 wirbel — M-invariant int4 Marlin GEMV microbench (route (a)).

Analysis-only kernel microbench on the assigned local A10G. NO HF Job, NO submission
change. Group: wirbel-minvariant-int4-gemv.

Goal (the kernel read denken #733's route (b) card only does qualitatively):
  (1) LOCUS: reproduce the int4 W4A16 Marlin GEMV at M=1 vs M=K+1 (K in {5,6}) and show
      the reduction order differs -> row-0 of the M=K+1 batched GEMM is NOT bit-identical
      to the M=1 GEMV of the SAME activation row. This is the byte-divergence source that
      #728's AR-vs-AR control (M=1 run-to-run 128/128 identical, BI=1) localizes 100% to
      this kernel. Also show it is DETERMINISTIC (same M -> same bits, run-to-run), i.e.
      a fixed schedule difference (repairable in principle), not atomicAdd noise.
  (2) OVERHEAD bracket for an M-INVARIANT variant (fixed, M-independent reduction order):
      time the stock Marlin GEMM at M in {1,6,7,8} on the real Gemma-4-E4B verify shapes.
      The M-dependent reduction is a *compute/reduction-order* effect; the dominant cost
      at M<=8 is the single int4 weight read from HBM. If the GEMM is memory-bandwidth
      bound (t(M=7) ~= t(M=1), achieved BW near A10G peak), then an M-invariant schedule
      that preserves the single weight read adds only a small fraction (the reduction is a
      sliver of a BW-bound kernel) -> overhead bracket. We also report the weight-read
      memory floor to bound the worst case.

Mirrors production exactly: vllm.model_executor.kernels.linear.mixed_precision.marlin
uses marlin_make_workspace_new(device) + apply_gptq_marlin_linear(..., use_fp32_reduce=
USE_FP32_REDUCE_DEFAULT=True); use_atomic_add is computed internally and is False on A10G
(sm86<90 + bf16, and n>=2048 for every verify shape). We call that exact production fn.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    GPTQ_MARLIN_MAX_PARALLEL,
    GPTQ_MARLIN_MIN_THREAD_N,
    USE_FP32_REDUCE_DEFAULT,
    apply_gptq_marlin_linear,
    marlin_make_empty_g_idx,
    marlin_make_workspace_new,
    should_use_atomic_add_reduce,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    marlin_quantize,
)
from vllm.scalar_type import scalar_types

HERE = Path(__file__).resolve().parent

# A10G (sm86) spec: 24 GB GDDR6, 600 GB/s peak HBM bandwidth (datasheet).
A10G_PEAK_BW_GBPS = 600.0

# Gemma-4-E4B-it verify GEMM shapes (size_k = contraction, size_n = output).
# hidden=2560, intermediate=10240, vocab=262144, q=8*256=2048, kv=2*256=512.
# All N>=2048 -> should_use_atomic_add_reduce == False (deterministic global reduce).
SHAPES = [
    # name,           size_k, size_n
    ("qkv_proj",       2560,   3072),    # q(2048)+k(512)+v(512)
    ("o_proj",         2048,   2560),
    ("gate_up_proj",   2560,  20480),    # 2*intermediate, biggest MLP weight read
    ("down_proj",     10240,   2560),    # largest K (most K-splitting)
    ("lm_head",        2560, 262144),    # logit producer, largest weight read
]

GROUP_SIZE = 128
WTYPE = scalar_types.uint4b8  # GPTQ-style int4, symmetric (no zero-point)


def _bf16_bytes_for_int4_weight(size_k: int, size_n: int, group_size: int) -> int:
    """HBM bytes read for one int4 W4A16 GEMM: 4-bit packed weights + bf16 group scales."""
    w_bytes = size_k * size_n * 4 // 8          # 4-bit packed
    s_bytes = (size_k // group_size) * size_n * 2  # bf16 scales, one per group per col
    return w_bytes + s_bytes


def build_layer(size_k: int, size_n: int, device: torch.device, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(size_k, size_n, generator=g, dtype=torch.float32) * 0.05
    w = w.to(device)
    w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, _rand_perm = marlin_quantize(
        w, WTYPE, GROUP_SIZE, act_order=False
    )
    workspace = marlin_make_workspace_new(device)
    empty = marlin_make_empty_g_idx(device)
    return {
        "w_ref": w_ref, "q": marlin_q_w, "s": marlin_s,
        "g_idx": empty, "sort": empty, "zp": empty, "workspace": workspace,
    }


def run_gemm(layer, a: torch.Tensor, size_k: int, size_n: int) -> torch.Tensor:
    return apply_gptq_marlin_linear(
        input=a,
        weight=layer["q"],
        weight_scale=layer["s"],
        weight_zp=layer["zp"],
        g_idx=layer["g_idx"],
        g_idx_sort_indices=layer["sort"],
        workspace=layer["workspace"],
        wtype=WTYPE,
        output_size_per_partition=size_n,
        input_size_per_partition=size_k,
        is_k_full=True,
        bias=None,
        use_fp32_reduce=USE_FP32_REDUCE_DEFAULT,
    )


def time_gemm(layer, a, size_k, size_n, reps=50, warmup=10) -> float:
    for _ in range(warmup):
        run_gemm(layer, a, size_k, size_n)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_gemm(layer, a, size_k, size_n)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def locus_and_overhead(size_k, size_n, name, device, ms, seed=0):
    layer = build_layer(size_k, size_n, device, seed)
    gx = torch.Generator(device="cpu").manual_seed(seed + 99)
    x_row = (torch.randn(size_k, generator=gx, dtype=torch.float32) * 1.0).to(
        device, dtype=torch.bfloat16
    )

    atomic = should_use_atomic_add_reduce(
        m=max(ms), n=size_n, k=size_k, device=device, dtype=torch.bfloat16
    )

    # --- M=1 baseline output (the proven-deterministic AR path) ---
    a1 = x_row.view(1, size_k).contiguous()
    out1 = run_gemm(layer, a1, size_k, size_n)[0].float().clone()

    per_m = {}
    for M in ms:
        # all M rows identical -> any row of the batched output is the same activation's
        # logit; row 0 should equal the M=1 GEMV if the kernel were M-invariant.
        a = x_row.view(1, size_k).expand(M, size_k).contiguous()
        outM = run_gemm(layer, a, size_k, size_n)
        row0 = outM[0].float()
        diff = (row0 - out1).abs()
        # run-to-run determinism at this M
        row0_b = run_gemm(layer, a, size_k, size_n)[0].float()
        det_bitexact = bool(torch.equal(row0, row0_b))
        # cross-row consistency within the batched output (rows are identical inputs)
        cross = (outM[0].float() - outM[-1].float()).abs().max().item() if M > 1 else 0.0
        n_bitdiff = int((row0 != out1).sum().item())
        per_m[M] = {
            "n_elem": size_n,
            "n_bitdiff_vs_m1": n_bitdiff,
            "frac_bitdiff_vs_m1": n_bitdiff / size_n,
            "max_abs_diff_vs_m1": diff.max().item(),
            "mean_abs_diff_vs_m1": diff.mean().item(),
            "rel_diff_vs_m1": (diff.max() / (out1.abs().max() + 1e-9)).item(),
            "row_run_to_run_bitexact": det_bitexact,
            "cross_row_max_abs": cross,
        }

    # --- timing sweep (verify-step cost vs M) ---
    weight_bytes = _bf16_bytes_for_int4_weight(size_k, size_n, GROUP_SIZE)
    mem_floor_s = weight_bytes / (A10G_PEAK_BW_GBPS * 1e9)
    timings = {}
    for M in [1] + [m for m in ms if m != 1]:
        a = x_row.view(1, size_k).expand(M, size_k).contiguous()
        t = time_gemm(layer, a, size_k, size_n)
        timings[M] = {
            "median_s": t,
            "achieved_bw_gbps": weight_bytes / t / 1e9,
            "bw_util_vs_peak": (weight_bytes / t / 1e9) / A10G_PEAK_BW_GBPS,
            "t_over_memfloor": t / mem_floor_s,
        }

    return {
        "name": name, "size_k": size_k, "size_n": size_n,
        "use_atomic_add": atomic,
        "weight_read_bytes": weight_bytes,
        "mem_floor_s": mem_floor_s,
        "locus": per_m,
        "timings": timings,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ks", default="5,6", help="comma list of K (num_spec); M=K+1 verify")
    ap.add_argument("--extra-m", default="8", help="extra M values to time")
    ap.add_argument("--out", type=Path, default=HERE / "microbench_report.json")
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False  # ieee, match BI determinism

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    verify_ms = sorted({k + 1 for k in ks} | {int(x) for x in args.extra_m.split(",") if x.strip()})
    ms = [1] + verify_ms

    print(f"[mb] device={torch.cuda.get_device_name(device)} "
          f"cap={torch.cuda.get_device_capability(device)}", flush=True)
    print(f"[mb] Ks={ks} -> verify M=K+1 in {verify_ms}; timing M in {ms}", flush=True)

    results = []
    for name, sk, sn in SHAPES:
        print(f"[mb] === {name}: size_k={sk} size_n={sn} ===", flush=True)
        r = locus_and_overhead(sk, sn, name, device, ms, seed=hash(name) % 10000)
        results.append(r)
        # console summary
        for M in verify_ms:
            lm = r["locus"][M]
            print(f"[mb]   M={M}: bitdiff_vs_M1={lm['n_bitdiff_vs_m1']}/{lm['n_elem']} "
                  f"({lm['frac_bitdiff_vs_m1']*100:.2f}%) max|d|={lm['max_abs_diff_vs_m1']:.3e} "
                  f"rel={lm['rel_diff_vs_m1']:.2e} det_run2run={lm['row_run_to_run_bitexact']}",
                  flush=True)
        t1 = r["timings"][1]["median_s"]
        for M in verify_ms:
            tM = r["timings"][M]["median_s"]
            print(f"[mb]   t(M={M})={tM*1e6:.1f}us  t(M=1)={t1*1e6:.1f}us  "
                  f"ratio={tM/t1:.3f}  BWutil(M={M})={r['timings'][M]['bw_util_vs_peak']*100:.0f}% "
                  f"t/memfloor={r['timings'][M]['t_over_memfloor']:.2f}", flush=True)

    report = {
        "pr": 736, "analysis_only": True, "official_tps": 0,
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "a10g_peak_bw_gbps": A10G_PEAK_BW_GBPS,
        "group_size": GROUP_SIZE, "wtype": "uint4b8",
        "ks": ks, "verify_ms": verify_ms,
        "use_fp32_reduce": USE_FP32_REDUCE_DEFAULT,
        "results": results,
    }
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"[mb] report -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
