#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Is the served int4 verify-GEMM the FASTEST selectable config for sm_86 M=8? (PR #448, stark)

THE QUESTION
------------
denken #441 (7rb089z3) decomposed verify: T_body (int4-Marlin GEMM) = 16935.9us of T_verify =
18270.6us -> the GEMM body is ~92.7% of verify (attention only 6.90%, M-invariant). So the verify
wall IS the int4 GEMM. This card asks: is the SERVED Marlin kernel+config the fastest AVAILABLE for
the real sm_86 / M=8 / head_dim=256 / GQA-8:2 shape, or is there SELECTABLE headroom -- a Marlin
config knob, or an ALTERNATIVE int4 GEMM already compiled into the vLLM-0.22 wheel (Machete / Cutlass
/ AllSpark / Exllama)? SELECTION + CONFIG ONLY, NO new kernel BUILD (a build is stark #440's NO-GO).

WHAT THE SERVED STACK DISPATCHES (resolved from vLLM's own selector, not assumed)
---------------------------------------------------------------------------------
Served checkpoint = google/gemma-4-E4B-it-qat-w4a16-ct baked: int4 W4A16, compressed-tensors
pack-quantized, group_size=128 SYMMETRIC (uint4b8), bf16 activations. choose_mp_linear_kernel() on
sm_86 EXCLUDES every alternative and selects Marlin UNIQUELY:
  * CutlassW4A8 / Machete : require compute capability 90 (Hopper) -> excluded on sm_86.
  * AllSpark (min_cap 80) : "For Ampere GPU, AllSpark does not support group_size=128" -> excluded.
  * Conch                 : not installed in the wheel -> excluded.
  * Exllama (min_cap 60)  : "Exllama only supports float16 activations" (served is bf16) -> excluded.
  * Marlin   (min_cap 75) : can_implement=True -> SELECTED for qkv/o/gate_up/down.
=> Marlin is the ONLY in-wheel int4 path for the served numerics on sm_86. (kanna #132: sub-4-bit
   schemes Machete=Hopper / Q-Palette=Ada have n_subbit_servable_in_wheel=0 -- consistent.)

THE SELECTABLE SURFACE Marlin EXPOSES (apply_gptq_marlin_linear -> ops.marlin_gemm)
----------------------------------------------------------------------------------
  * use_atomic_add : should_use_atomic_add_reduce() returns False for ALL our shapes -- HARD-OFF on
    sm8x+bf16 ("sm8x doesn't support atomicAdd + bfloat16 natively") AND n>=2048. NOT selectable.
  * use_fp32_reduce: served default = USE_FP32_REDUCE_DEFAULT = True. Setting False uses a lower-
    precision split-K reduction -> FASTER only where the kernel splits K, but CHANGES THE OUTPUT BITS
    (reduction-order/precision change) -> byte-exact identity RISK. This is the one real speed knob.
  * tile config (thread_k/thread_n/num_stages/m_blocks): AUTO-SELECTED inside the CUDA kernel by
    (m,n,k); vLLM exposes NO Python override -> changing it is a kernel BUILD = NO-GO (stark #440).
  * group_size / W4A8 input-dtype: change the NUMERICS (PPL + bits) -> not a byte-exact "selection".

WHAT THIS MEASURES (real A10G sm_86, byte-exact, NO served-file change, NO build, NO HF job)
-------------------------------------------------------------------------------------------
For each served fused shape (qkv K2560/N3072, o K2048/N2560, gate_up K2560/N20480, down K10240/N2560)
built from a faithful int4 g128-symmetric Marlin quantization:
  1. Served-default Marlin latency at M=8 (CUDA-event kernel-time + CUDA-graph replay), and M=1 for
     the M-invariance cross-check (#441: body is weight-read bound, +0.47% over M=1->8).
  2. The selectable sweep: use_fp32_reduce True(served) vs False -- per-shape speedup AND bit-exact
     test (lawine #438: candidate-vs-default on the SAME size-8 path so any FP artifact cancels).
     use_atomic_add forced True (verify it is ignored/guarded on sm8x+bf16).
  3. Honest end-to-end map via the SAME Amdahl translation #437/#433/#440 used (f_body=0.7624 body
     fraction of the realized decode step, base 467.14 realized frontier): a body speedup `penalty`
     -> TPS = base / (1 - f_body + f_body/penalty). The byte-exact-SAFE penalty is 1.0 (no faster
     byte-exact config) -> DeltaTPS=0; the fp32_reduce=False UPPER BOUND is mapped too, to show even
     the identity-breaking option is sub-threshold. This is the pinned-K (#433 +13.998 -> -5.82) /
     cb3 (#437 +15.60 -> 0.0) discipline: a microbench ratio is NOT an end-to-end delta.

VERDICT: is there >+2 TPS HONEST byte-exact end-to-end headroom in int4-GEMM SELECTION, or is the
served Marlin already optimal for this shape (and any faster int4 GEMM a BUILD = NO-GO)? ppl anchored
2.3772 (a kernel-selection change with byte-exact output is PPL-neutral; verify is the byte-exact
arbiter, land #420). analysis_only, no_hf_job, no_served_file_change.

PUBLIC/BANKED EVIDENCE: denken #441 7rb089z3 (verify decomposition, body=92.7%); denken #423 5a6zq2yz
(467.14 realized base); deployed PR #52 2x9fm2zx (481.53 non-equiv); stark #437/#440 (f_body=0.7624,
Marlin HBM eff 0.6418, fused-build NO-GO); kanna #132 (sub-4-bit unservable on sm_86); wirbel #442
e5n9a2dc (attention-autotune lead, orthogonal). vLLM 0.22 wheel (.venv).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# Single-A10G pod: force device 0 unless caller overrides (the #358/#363 2nd-GPU gotcha).
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0") or "0"

# ============================================================================ #
# Anchors (banked, do NOT re-measure) -- self-contained, robust to sibling moves.
# ============================================================================ #
REALIZED_TPS = 467.14     # denken #423 5a6zq2yz blanket-strict realized equivalence frontier (base)
DEPLOYED_TPS = 481.53     # PR #52 2x9fm2zx deployed (non-equivalent) incumbent -- the bar to beat
PPL_ANCHORED = 2.3772     # ubel #422 / deployed; a byte-exact kernel-selection change is PPL-neutral
F_BODY = 0.7624097014503400   # body-GEMM weight-read fraction of the realized step (stark #437/#440)
T_BODY_441_US = 16935.9   # denken #441 7rb089z3 T_body(M=8) CUDA-event sum (basis cross-check note)
MARLIN_HBM_EFF_437 = 0.6417907155742067   # stark #437 M=8 Marlin peak-copy HBM efficiency
MATERIALITY_TPS = 2.0     # PR bar: ">+2 TPS HONEST end-to-end" to call it real headroom

# ---- served gemma-4-E4B-it int4 config (from /tmp/osoi5-v0-baked/config.json) -------------- #
HIDDEN = 2560
INTERMEDIATE = 10240
N_Q_HEADS = 8
N_KV_HEADS = 2
HEAD_DIM = 256
N_LAYERS = 37             # 30 sliding + 7 full (the sliding/full split is ATTENTION-only; the
                         # Linear GEMM shapes are identical across all 37 decode layers)
GROUP_SIZE = 128
VOCAB_PRUNED = 12000     # LM_HEAD_PRUNE 12k served width (#441 lmhead 74.75us -- minor)

# Served fused GEMM shapes vLLM produces (QKVParallelLinear + MergedColumnParallelLinear):
#   name -> (K=in, N=out, per-layer count)
# NB: lm_head (channel-wise int4, pruned ~12k) is NOT part of T_body -- #441 separates T_lmhead
# (74.75us, 0.4% of verify) from T_body and it is a different (per-channel) quant path; excluded.
SERVED_SHAPES: list[tuple[str, int, int, int]] = [
    ("qkv",     HIDDEN, (N_Q_HEADS + 2 * N_KV_HEADS) * HEAD_DIM, N_LAYERS),   # 2560 -> 3072
    ("o_proj",  N_Q_HEADS * HEAD_DIM, HIDDEN, N_LAYERS),                       # 2048 -> 2560
    ("gate_up", HIDDEN, 2 * INTERMEDIATE, N_LAYERS),                           # 2560 -> 20480
    ("down",    INTERMEDIATE, HIDDEN, N_LAYERS),                               # 10240 -> 2560
]


# ============================================================================ #
# Amdahl end-to-end translation (the SAME one #437/#433/#440 used).
# ============================================================================ #
def amdahl_tps(penalty: float, f_body: float = F_BODY, base: float = REALIZED_TPS) -> float:
    """If the body GEMM ran `penalty`x faster (>1) and were deployed: new_step = old_step *
    (1 - f_body + f_body/penalty); TPS = base/that. penalty=1 => base (no change)."""
    if penalty <= 0:
        return float("nan")
    factor = (1.0 - f_body) + f_body / penalty
    return base / factor if factor > 0 else float("nan")


# ============================================================================ #
# GPU helpers
# ============================================================================ #
def _device():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available (set CUDA_VISIBLE_DEVICES)")
    return torch.device("cuda:0")


def _gpu_facts(dev) -> dict[str, Any]:
    import torch
    p = torch.cuda.get_device_properties(dev)
    cc = torch.cuda.get_device_capability(dev)
    return {
        "name": p.name, "sm_count": p.multi_processor_count,
        "compute_capability": f"{cc[0]}.{cc[1]}", "cc_tuple": list(cc),
        "total_mem_gib": round(p.total_memory / 1024**3, 2),
        "is_sm86": bool(cc == (8, 6)),
    }


def _cuda_event_us(fn: Callable[[], Any], iters: int, warmup: int, reps: int = 3) -> float:
    """Median per-call kernel time in microseconds (CUDA events; excludes Python dispatch)."""
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    means = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        means.append(s.elapsed_time(e) * 1000.0 / iters)  # ms -> us
    return float(statistics.median(means))


def _cuda_graph_us(fn: Callable[[], Any], iters: int, warmup: int) -> float:
    """Per-call time when the op is captured in a CUDA graph and replayed -- mirrors the served
    ONEGRAPH path (launch overhead amortized). Returns us/call or nan if capture fails."""
    import torch
    try:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        for _ in range(warmup):
            g.replay()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        return float(s.elapsed_time(e) * 1000.0 / iters)
    except Exception:  # noqa: BLE001
        return float("nan")


# ============================================================================ #
# Build a faithful served int4 g128-symmetric Marlin GEMM for one shape.
# ============================================================================ #
def build_marlin_gemm(K: int, N: int, dev, seed: int = 0):
    """Returns a dict with the repacked int4 weight + scales + a runner closure run(M, fp32, atomic)."""
    import torch
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new, marlin_make_empty_g_idx)
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize

    torch.manual_seed(seed)
    wtype = scalar_types.uint4b8
    gs = GROUP_SIZE if (K % GROUP_SIZE == 0 and N != VOCAB_PRUNED) else -1  # lm_head is channel-wise
    w = (torch.randn(K, N, dtype=torch.bfloat16, device=dev) * 0.02)
    w_ref, q_w, s, g_idx, sort_idx, _perm = marlin_quantize(w, wtype, gs, act_order=False)
    zp = marlin_make_empty_g_idx(dev)
    ws = marlin_make_workspace_new(dev)

    def make_input(M: int):
        return torch.randn(M, K, dtype=torch.bfloat16, device=dev) * 0.1

    def run(a, fp32_reduce: bool, atomic: bool):
        return ops.marlin_gemm(
            a, None, q_w, None, s, None, None, zp, g_idx, sort_idx, ws,
            wtype, a.shape[0], N, K, True, atomic, fp32_reduce, False)

    return {"K": K, "N": N, "w_ref": w_ref, "make_input": make_input, "run": run, "group_size": gs}


# ============================================================================ #
# Resolve dispatch via vLLM's OWN selector (authoritative, not assumed).
# ============================================================================ #
def resolve_dispatch(dev) -> dict[str, Any]:
    import torch
    from vllm.scalar_type import scalar_types
    from vllm.model_executor.kernels.linear import choose_mp_linear_kernel, _POSSIBLE_KERNELS
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import MPLinearLayerConfig
    from vllm.platforms import PlatformEnum

    def cfg(K, N):
        return MPLinearLayerConfig(
            full_weight_shape=(K, N), partition_weight_shape=(K, N),
            weight_type=scalar_types.uint4b8, act_type=torch.bfloat16,
            group_size=GROUP_SIZE, zero_points=False, has_g_idx=False)

    selected = {}
    for name, K, N, _c in SERVED_SHAPES:
        if name == "lm_head":
            continue
        selected[name] = choose_mp_linear_kernel(cfg(K, N)).__name__

    # per-kernel verdict for the representative gate_up shape
    c = cfg(HIDDEN, 2 * INTERMEDIATE)
    verdicts = []
    for kcls in _POSSIBLE_KERNELS[PlatformEnum.CUDA]:
        ok, why = kcls.can_implement(c)
        verdicts.append({"kernel": kcls.__name__, "min_capability": kcls.get_min_capability(),
                         "can_implement": bool(ok), "reason": (why or "")[:120]})

    uniq = sorted(set(selected.values()))
    return {
        "selected_per_shape": selected,
        "selected_kernel": uniq[0] if len(uniq) == 1 else "MIXED:" + ",".join(uniq),
        "marlin_is_unique": bool(uniq == ["MarlinLinearKernel"]),
        "cuda_kernel_priority": [k.__name__ for k in _POSSIBLE_KERNELS[PlatformEnum.CUDA]],
        "per_kernel_verdict_gate_up": verdicts,
    }


def probe_machete(dev) -> dict[str, Any]:
    """Confirm Machete is Hopper-only on this device: schedules may register, but it is never
    SELECTED (sm90 gate) and machete_mm does not run the served numerics on sm_86."""
    import torch
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types
    out: dict[str, Any] = {}
    try:
        sch = ops.machete_supported_schedules(
            torch.bfloat16, scalar_types.uint4b8, group_scales_type=torch.bfloat16)
        out["machete_supported_schedules_n"] = len(sch) if sch else 0
        out["machete_schedules_are_tma_hopper"] = bool(sch and all("Tma" in s for s in sch))
    except Exception as ex:  # noqa: BLE001
        out["machete_supported_schedules_err"] = f"{type(ex).__name__}: {str(ex)[:120]}"
    cc = torch.cuda.get_device_capability(dev)
    out["device_is_hopper_sm90"] = bool(cc[0] >= 9)
    out["machete_selectable_here"] = bool(cc[0] >= 9)
    return out


# ============================================================================ #
# Microbench one shape: served default + selectable sweep + byte-exact test.
# ============================================================================ #
def bench_shape(name: str, K: int, N: int, count: int, dev, iters: int, warmup: int) -> dict[str, Any]:
    import torch
    g = build_marlin_gemm(K, N, dev)
    a8 = g["make_input"](8)
    a1 = g["make_input"](1)

    # served default: fp32_reduce=True, atomic=False
    served8 = _cuda_event_us(lambda: g["run"](a8, True, False), iters, warmup)
    served1 = _cuda_event_us(lambda: g["run"](a1, True, False), iters, warmup)
    served8_graph = _cuda_graph_us(lambda: g["run"](a8, True, False), iters, warmup)

    # selectable knob: fp32_reduce=False (the one real speed lever) -- eager AND graph (served basis)
    fp32off8 = _cuda_event_us(lambda: g["run"](a8, False, False), iters, warmup)
    fp32off8_graph = _cuda_graph_us(lambda: g["run"](a8, False, False), iters, warmup)

    # byte-exact test: candidate vs served default on the SAME size-8 path (lawine #438)
    o_served = g["run"](a8, True, False)
    o_fp32off = g["run"](a8, False, False)
    bitexact_fp32off = bool(torch.equal(o_served, o_fp32off))
    max_abs_delta = float((o_served.float() - o_fp32off.float()).abs().max().item())

    # atomic_add forced True -> verify it is guarded/ignored (output unchanged) on sm8x+bf16
    try:
        o_atomic = g["run"](a8, True, True)
        atomic_bitexact = bool(torch.equal(o_served, o_atomic))
        atomic_err = ""
    except Exception as ex:  # noqa: BLE001
        atomic_bitexact = None
        atomic_err = f"{type(ex).__name__}: {str(ex)[:80]}"

    speedup_fp32off = served8 / fp32off8 if fp32off8 > 0 else float("nan")
    m_invariance = served8 / served1 if served1 > 0 else float("nan")
    del g, a8, a1, o_served, o_fp32off
    gc.collect(); torch.cuda.empty_cache()

    speedup_fp32off_graph = (served8_graph / fp32off8_graph
                             if fp32off8_graph and fp32off8_graph == fp32off8_graph and fp32off8_graph > 0
                             else float("nan"))
    return {
        "name": name, "K": K, "N": N, "count": count, "params_per_layer": K * N,
        "served_us_m8": served8, "served_us_m1": served1, "served_graph_us_m8": served8_graph,
        "fp32off_us_m8": fp32off8, "fp32off_graph_us_m8": fp32off8_graph,
        "speedup_fp32off": speedup_fp32off, "speedup_fp32off_graph": speedup_fp32off_graph,
        "m8_over_m1": m_invariance,
        "fp32off_bitexact_vs_served": bitexact_fp32off, "fp32off_max_abs_delta": max_abs_delta,
        "atomic_true_bitexact_vs_served": atomic_bitexact, "atomic_true_err": atomic_err,
    }


# ============================================================================ #
# Compose: T_body, honest map, verdict.
# ============================================================================ #
def compose(gpu, dispatch, machete, rows: list[dict[str, Any]]) -> dict[str, Any]:
    # T_body = sum over shapes of count * latency (served default), M=8 vs M=1.
    body_rows = [r for r in rows if r["name"] != "lm_head"]
    t_body_m8 = sum(r["count"] * r["served_us_m8"] for r in body_rows)
    t_body_m1 = sum(r["count"] * r["served_us_m1"] for r in body_rows)
    t_body_fp32off_m8 = sum(r["count"] * r["fp32off_us_m8"] for r in body_rows)
    body_m_invariance = t_body_m8 / t_body_m1 if t_body_m1 else float("nan")

    # latency-weighted body speedup if fp32_reduce=False were applied to the WHOLE body.
    # GRAPH basis is the FAITHFUL one (served stack is ONEGRAPH -> launch overhead amortized; the
    # small qkv/o GEMMs are overhead-bound in eager). Eager kept as a conservative cross-check.
    t_body_graph_m8 = sum(r["count"] * r["served_graph_us_m8"] for r in body_rows)
    t_body_graph_fp32off_m8 = sum(r["count"] * r["fp32off_graph_us_m8"] for r in body_rows)
    body_speedup_fp32off_eager = t_body_m8 / t_body_fp32off_m8 if t_body_fp32off_m8 else float("nan")
    body_speedup_fp32off_graph = (t_body_graph_m8 / t_body_graph_fp32off_m8
                                  if t_body_graph_fp32off_m8 else float("nan"))
    # headline = graph basis (faithful); fall back to eager if graph capture failed.
    body_speedup_fp32off = (body_speedup_fp32off_graph
                            if body_speedup_fp32off_graph == body_speedup_fp32off_graph
                            else body_speedup_fp32off_eager)

    # byte-exact-SAFE selectable surface: which candidates keep bit-identical output?
    all_fp32off_bitexact = all(r["fp32off_bitexact_vs_served"] for r in body_rows)
    # served default is trivially byte-exact vs itself -> the only byte-exact-safe config is served.
    byteexact_safe_penalty = 1.0   # no faster byte-exact-safe config exists (proven per-shape below)
    byteexact_safe_tps = amdahl_tps(byteexact_safe_penalty)
    byteexact_safe_delta = byteexact_safe_tps - REALIZED_TPS   # = 0.0

    # UPPER BOUND: if we SACRIFICED byte-exactness and ran fp32_reduce=False everywhere
    fp32off_tps = amdahl_tps(body_speedup_fp32off)
    fp32off_delta = fp32off_tps - REALIZED_TPS
    fp32off_clears_materiality = bool(fp32off_delta > MATERIALITY_TPS)
    fp32off_is_byteexact = all_fp32off_bitexact   # if even the "fast" knob is byte-exact, free win

    # the PR primary metric
    if fp32off_is_byteexact:
        # fp32_reduce=False didn't change bits anywhere -> it's a FREE byte-exact speedup
        max_honest_byteexact_delta = fp32off_delta
        best_byteexact_penalty = body_speedup_fp32off
    else:
        max_honest_byteexact_delta = byteexact_safe_delta  # 0.0
        best_byteexact_penalty = 1.0
    max_honest_byteexact_tps = REALIZED_TPS + max_honest_byteexact_delta

    has_byteexact_headroom = bool(max_honest_byteexact_delta > MATERIALITY_TPS)

    verdict = (
        f"On sm_86 the served int4 W4A16 group-128 GEMM dispatches UNIQUELY to Marlin "
        f"(choose_mp_linear_kernel: Cutlass/Machete=Hopper-only, AllSpark=no-g128-on-Ampere, "
        f"Conch=absent, Exllama=fp16-only -> only Marlin can_implement). Marlin's tile config is "
        f"auto-selected inside the kernel (NO Python knob); the only selectable lever is "
        f"use_fp32_reduce (use_atomic_add is hard-off for sm8x+bf16). Measured at M=8: "
        f"use_fp32_reduce=False is {body_speedup_fp32off:.4f}x on the body "
        f"({(body_speedup_fp32off-1)*100:+.2f}%) but "
        f"{'KEEPS' if all_fp32off_bitexact else 'BREAKS'} byte-exactness "
        f"(per-shape bit-flip on the split-K layers). Body is M-invariant "
        f"(T_body M8/M1={body_m_invariance:.4f}, reproducing #441's weight-read-bound finding). "
        f"Honest Amdahl map (f_body={F_BODY:.4f}, base {REALIZED_TPS}): byte-exact-safe headroom = "
        f"{byteexact_safe_delta:+.2f} TPS; even the identity-BREAKING fp32_reduce=False upper bound "
        f"= {fp32off_delta:+.2f} TPS ({'CLEARS' if fp32off_clears_materiality else 'BELOW'} the "
        f"+{MATERIALITY_TPS:.0f} bar). VERDICT: "
        f"{'SELECTABLE HEADROOM EXISTS' if has_byteexact_headroom else 'NO byte-exact selectable headroom -- served Marlin is already optimal for this shape'}; "
        f"any faster int4 GEMM requires a kernel/source BUILD = NO-GO scope (stark #440). "
        f"ppl anchored {PPL_ANCHORED} (byte-exact selection is PPL-neutral)."
    )

    required = {
        "selected_kernel": dispatch["selected_kernel"],
        "marlin_is_unique_on_sm86": dispatch["marlin_is_unique"],
        "machete_selectable_here": machete.get("machete_selectable_here", False),
        "t_body_us_m8": t_body_m8, "t_body_us_m1": t_body_m1,
        "body_m_invariance_m8_over_m1": body_m_invariance,
        "body_speedup_fp32reduce_off": body_speedup_fp32off,
        "fp32reduce_off_is_byteexact": fp32off_is_byteexact,
        "byteexact_safe_endtoend_tps_delta": byteexact_safe_delta,
        "fp32reduce_off_upperbound_tps_delta": fp32off_delta,
        "max_honest_endtoend_tps_delta": max_honest_byteexact_delta,
        "max_honest_endtoend_tps": max_honest_byteexact_tps,
        "has_byteexact_selectable_headroom": has_byteexact_headroom,
        "faster_int4_gemm_requires_build": True,
        "ppl": PPL_ANCHORED,
    }
    config = {
        "agent": "stark", "pr": 448, "kind": "int4-gemm-kernel-config-audit",
        "hidden": HIDDEN, "intermediate": INTERMEDIATE, "head_dim": HEAD_DIM,
        "n_q_heads": N_Q_HEADS, "n_kv_heads": N_KV_HEADS, "n_layers": N_LAYERS,
        "group_size": GROUP_SIZE, "f_body": F_BODY, "realized_base_tps": REALIZED_TPS,
        "deployed_tps": DEPLOYED_TPS, "materiality_tps": MATERIALITY_TPS,
        "served_default_config": "use_fp32_reduce=True, use_atomic_add=False, is_k_full=True, W4A16",
    }
    wandb_metrics = {k: v for k, v in required.items() if isinstance(v, (int, float, bool))}
    wandb_metrics.update({
        "t_body_fp32off_us_m8": t_body_fp32off_m8,
        "t_body_graph_us_m8": t_body_graph_m8,
        "body_speedup_fp32off_graph": body_speedup_fp32off_graph,
        "body_speedup_fp32off_eager": body_speedup_fp32off_eager,
        "fp32off_endtoend_tps": fp32off_tps,
        "deployed_tps": DEPLOYED_TPS, "realized_base_tps": REALIZED_TPS,
        "t_body_441_us_ref": T_BODY_441_US,
    })
    return {
        "gpu": gpu, "dispatch": dispatch, "machete": machete, "per_shape": rows,
        "required": required, "verdict": verdict, "config": config, "wandb_metrics": wandb_metrics,
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
        "timestamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    }


# ============================================================================ #
# Self-test (0-GPU): the Amdahl arithmetic + verdict logic.
# ============================================================================ #
def self_test() -> dict[str, Any]:
    c: dict[str, bool] = {}
    c["amdahl_penalty1_is_base"] = abs(amdahl_tps(1.0) - REALIZED_TPS) < 1e-9
    c["amdahl_faster_raises_tps"] = amdahl_tps(1.10) > REALIZED_TPS
    c["amdahl_slower_lowers_tps"] = amdahl_tps(0.90) < REALIZED_TPS
    # SENSITIVITY: f_body is large, so the map is steep -- a 0.2% body speedup is sub-threshold
    # (+0.71 TPS) but a 1% body speedup clears it (+3.55 TPS). The threshold body speedup is ~0.55%.
    c["tiny_body_speedup_subthreshold"] = (amdahl_tps(1.002) - REALIZED_TPS) < MATERIALITY_TPS
    c["onepct_body_speedup_clears"] = (amdahl_tps(1.01) - REALIZED_TPS) > MATERIALITY_TPS
    # infinite body speedup ceiling is finite and sane
    ceil = amdahl_tps(1e9)
    c["body_free_ceiling_finite"] = REALIZED_TPS < ceil < 3000
    c["base_below_deployed"] = REALIZED_TPS < DEPLOYED_TPS
    c["four_body_shapes"] = len([s for s in SERVED_SHAPES if s[0] != "lm_head"]) == 4
    c["qkv_shape"] = SERVED_SHAPES[0][1:3] == (2560, 3072)
    c["gate_up_shape"] = SERVED_SHAPES[2][1:3] == (2560, 20480)
    c["down_shape"] = SERVED_SHAPES[3][1:3] == (10240, 2560)
    c["f_body_in_unit"] = 0.0 < F_BODY < 1.0
    return {"self_test_passes": all(c.values()), "checks": c}


# ============================================================================ #
# Main
# ============================================================================ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=80)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb_group", type=str, default="kernel-tiling-sweep")
    ap.add_argument("--wandb_name", type=str, default="stark/int4-gemm-kernel-config-audit")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if args.self_test:
        st = self_test()
        (here / "selftest.json").write_text(json.dumps(st, indent=2))
        print(f"[self-test] passes={st['self_test_passes']}")
        for k, v in st["checks"].items():
            if not v:
                print(f"  FAIL: {k}")
        sys.exit(0 if st["self_test_passes"] else 1)

    dev = _device()
    gpu = _gpu_facts(dev)
    print(f"[audit] device={gpu['name']} cc={gpu['compute_capability']} sm={gpu['sm_count']}")
    dispatch = resolve_dispatch(dev)
    print(f"[audit] selected kernel: {dispatch['selected_kernel']} (marlin_unique={dispatch['marlin_is_unique']})")
    machete = probe_machete(dev)

    rows = []
    for name, K, N, count in SERVED_SHAPES:
        r = bench_shape(name, K, N, count, dev, args.iters, args.warmup)
        rows.append(r)
        print(f"[audit] {name:8s} K={K:5d} N={N:5d}: served(m8)={r['served_us_m8']:.2f}us "
              f"graph={r['served_graph_us_m8']:.2f}us fp32off={r['fp32off_us_m8']:.2f}us "
              f"speedup={r['speedup_fp32off']:.4f} bitexact={r['fp32off_bitexact_vs_served']} "
              f"m8/m1={r['m8_over_m1']:.4f}")

    payload = compose(gpu, dispatch, machete, rows)
    payload["self_test"] = self_test()
    payload["required"]["self_test_passes"] = payload["self_test"]["self_test_passes"]

    (here / "int4_gemm_kernel_config_audit_results.json").write_text(json.dumps(payload, indent=2))
    print(f"\n[audit] VERDICT: {payload['verdict']}")
    print(f"[audit] max_honest_endtoend_tps_delta = {payload['required']['max_honest_endtoend_tps_delta']:+.3f} TPS")

    if not args.no_wandb:
        try:
            import wandb
            run = wandb.init(project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
                             group=args.wandb_group, name=args.wandb_name,
                             config=payload["config"], job_type="analysis")
            wandb.log(payload["wandb_metrics"])
            cols = ["shape", "K", "N", "count", "served_us_m8", "graph_us_m8", "fp32off_us_m8",
                    "fp32off_graph_us_m8", "speedup_fp32off_eager", "speedup_fp32off_graph",
                    "fp32off_bitexact", "m8_over_m1"]
            tbl = wandb.Table(columns=cols)
            for r in rows:
                tbl.add_data(r["name"], r["K"], r["N"], r["count"], r["served_us_m8"],
                             r["served_graph_us_m8"], r["fp32off_us_m8"], r["fp32off_graph_us_m8"],
                             r["speedup_fp32off"], r["speedup_fp32off_graph"],
                             r["fp32off_bitexact_vs_served"], r["m8_over_m1"])
            wandb.log({"per_shape_gemm": tbl})
            dcols = ["kernel", "min_capability", "can_implement", "reason"]
            dtbl = wandb.Table(columns=dcols)
            for v in dispatch["per_kernel_verdict_gate_up"]:
                dtbl.add_data(v["kernel"], v["min_capability"], v["can_implement"], v["reason"])
            wandb.log({"kernel_dispatch": dtbl})
            payload["wandb_run_id"] = run.id
            (here / "int4_gemm_kernel_config_audit_results.json").write_text(json.dumps(payload, indent=2))
            wandb.finish()
            print(f"[audit] wandb run {run.id}")
        except Exception as ex:  # noqa: BLE001
            print(f"[audit] wandb failed (non-fatal): {type(ex).__name__}: {str(ex)[:160]}")

    sys.exit(0)


if __name__ == "__main__":
    main()
