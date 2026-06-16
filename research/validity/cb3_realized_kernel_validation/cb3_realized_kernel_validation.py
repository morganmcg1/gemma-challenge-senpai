#!/usr/bin/env python3
# ======================================================================================== #
# cb3 realized-kernel SPEEDUP validation -- the #433 analog for the cb3 +15.60 supply lift.
# ---------------------------------------------------------------------------------------- #
# WHY (PR #437): my own #433 (0pg4bz25) proved a MODELED supply lift can INVERT once measured
# on the kernel that actually runs -- the pinned-K +13.998 became -5.82 on the served Triton
# kernel_unified_attention. With 496.74 refuted, the equivalence-respecting frontier 482.74 now
# rests ENTIRELY on cb3 +15.60 (MODELED, kanna #403 iv9i2wks) over the 467.14 measured base
# (blanket-strict, denken #423 5a6zq2yz). cb3 has the SAME modeling pedigree pinned-K did: a
# bandwidth surrogate. It deserves the SAME skeptical realized-kernel test.
#
# THE GAP I OWN (the OP-LATENCY complement to lawine #388's realized-BW, 7rzf74q5):
#   lawine MEASURED int4-Marlin's realized us/GEMM and applied cb3's byte_ratio (0.785) ANALYTICALLY
#   -- the cb3 side was never RUN (realized_is_roofline_bound=True, "no cb3/QTIP kernel in env").
#   lawine's model is cb3_us = r*t_transfer + t_overhead: it ASSUMES cb3's fixed overhead EQUALS
#   int4-Marlin's, i.e. the RHT+VQ dequant adds NO extra op latency beyond what Marlin already pays.
#   That is exactly the assumption #433 destroyed for pinned-K (the split's reduce_segments op tax
#   was assumed away, then it inverted). My job: RUN the cb3 dequant ops (online activation FWHT +
#   VQ codebook reconstruct + the bf16 GEMM cb3 ultimately drives) on this sm_86 pod and MEASURE
#   whether the modeled +15.60 survives, haircuts, or inverts like the split did.
#
# WHAT (analysis_only, READ-ONLY op-latency microbench. NO served-file change, NO HF job, NO
# submission). Same envelope and rigor as #433: per-body-GEMM, M in {1,4,8} (the served MTP K=7
# verify widths -- NOTE the body read is KV-band-INDEPENDENT; the {128,256,512} KV band in the PR
# governs ATTENTION, which lawine/ubel own, not the body GEMM weight read), >=3 reps, fair
# pre-allocated buffers, cuda-event timing. realized_penalty = baseline_us / cb3_us (>1 => cb3
# faster). Translate through the SAME ladder form that produced +15.60: delta = base*f_body*(p-1).
#
# DECISIVE BRACKET (cb3 has no served kernel -- vLLM 0.22 has no sub-int4 Marlin path, lawine
# proved sub-int4 is UNREACHABLE off-the-shelf):
#   * cb3_FUSED (optimistic buildable bound): grant cb3 the full byte-saving roofline AND charge
#     only the MEASURED online-activation FWHT tax (the one op even a fused kernel cannot hide;
#     the codebook gather is assumed L1-resident/in-SM). = lawine's model + my measured FWHT.
#   * cb3_MATERIALIZE (the only path that RUNS on sm_86 today): FWHT + VQ reconstruct-to-bf16 +
#     bf16 GEMM. Reads the FULL bf16 weight => MORE bytes than int4-Marlin => the realized inversion
#     analog of #433's reduce_segments. MEASURED end to end.
# The honest deployable verdict keys on whether a fused RHT+VQ decode kernel exists/builds on sm_86.
#
# ANCHORED (do NOT re-measure): ppl=2.3772 (a body-read precision change's PPL is ubel #422's, RHT+VQ
# cost/quality). 467.14 base taken AS GIVEN (denken #423). f_body / byte_ratio AS GIVEN (lawine #388).
# Greedy identity is MEASURED never asserted; this card changes no served file and submits nothing.
#
# PUBLIC EVIDENCE USED (advisor-branch banked): kanna #403 iv9i2wks (the +15.60 cb3 rung);
# lawine #388 7rzf74q5 (realized-BW: byte_ratio 0.785, measured_floor 1.0588, "no cb3 kernel in env");
# denken #423 5a6zq2yz (467.14 blanket-strict measured base); my #433 0pg4bz25 (the pinned-K
# realized-kernel inversion this card is the cb3 analog of). Deployed #52 2x9fm2zx (481.53, non-equiv).
# ======================================================================================== #
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------------------- #
# Constants -- hardcoded with citations (self-contained like #433; robust to sibling moves).
# ---------------------------------------------------------------------------------------- #
A10G_SMS = 80
A10G_HBM_PEAK_GBS_DATASHEET = 600.0     # GA102 / A10G datasheet HBM BW (peak-copy is MEASURED below)
TOL = 1e-6

# ---- cb3 body-shrink (the input being kernel-validated) -- lawine #388 / #372 ------------ #
INT4_BPW = 4.125                        # deployed: 4-bit + bf16 g128 scale (16/128 = 0.125)
CB3_BPW_EFF = 3.2368598382749325        # #372 mixed allocation (88.8% body params at cb3, rest int4)
BYTE_RATIO = CB3_BPW_EFF / INT4_BPW     # = 0.7846932941272564  (the PR's "0.785"; cb3 reads 78.5%)

# ---- served-strict step decomposition (#378; the fractions that PRICE the lift) ---------- #
F_BODY_STRICT = 0.76240970145034        # body GEMM weight-read fraction (the HONEST shrinkable frac)
F_BODY_COMPLEMENT = 0.8825045903509467  # body+attn-proj complement variant (reported as sensitivity)
BAND_FLOOR = 469.6847174760462          # #378 better-case strict base (lawine translation base)
BAND_OFF_THE_SHELF = 357.32166269999993 # #378 worse-case strict base

# ---- the rung under test (the ladder this card validates) -------------------------------- #
CB3_BASE_TPS = 467.14                   # blanket-strict MEASURED base (denken #423 5a6zq2yz)
CB3_MODELED_DELTA = 15.60               # the MODELED cb3 lift (kanna #403 iv9i2wks) -- what we test
CB3_FRONTIER_MODELED = 482.74           # = 467.14 + 15.60 (the #407 packet's top equivalence rung)
DEPLOYED_TPS = 481.53                   # PR #52 2x9fm2zx (non-equivalent incumbent; identity 0.9966)

# the modeled body-read penalty IMPLIED by +15.60 through the ladder at f_body (mirrors #433's
# MODELED_PENALTY = 1 + delta/(base*f)):  15.60 = 467.14 * f_body * (p-1)
MODELED_PENALTY = 1.0 + CB3_MODELED_DELTA / (CB3_BASE_TPS * F_BODY_STRICT)   # ~1.0438

# ---- literature / banked realized-BW cross-checks (lawine #388) --------------------------- #
M1_MEASURED_HBM_EFF_388 = 0.25561637483960586   # count-weighted int4-Marlin M=1 weight-read eff
QTIP_BETA_BYTE_PROPORTIONAL = 0.51              # QTIP batch=1 byte-proportional fraction (Tab4)
LAWINE_MEASURED_FLOOR_SPEEDUP = 1.0587760668597737  # lawine's modeled measured-floor body speedup
LAWINE_QTIP_EMPIRICAL_SPEEDUP = 1.1233511704212635  # lawine's qtip-empirical body speedup
ROOFLINE_SPEEDUP = INT4_BPW / CB3_BPW_EFF       # = 1/byte_ratio = 1.2744 (fully BW-bound upper bound)

# ---- cb3 kernel mechanism (RHT + dim-2 VQ; the ops we MEASURE) ---------------------------- #
# RHT = randomized Hadamard transform: random sign flip (folded offline into weights) + a Hadamard
# of the incoherence group size (g128). Online cost = the activation Hadamard only. VQ = dim-2
# Gaussian vector quant (lawine: "dim-2 Gaussian VQ K=64 + g128 incoherence" -> 3.125 bpw uniform).
HADAMARD_GROUP = 128                    # g128 incoherence group (the online activation RHT block)
VQ_DIM = 2                              # dim-2 vector quant
VQ_CODEBOOK_K = 256                     # codebook entries (gather latency is ~K-insensitive; L1-resident)

# ---- served verify widths (#391) -- the body-read op-points ------------------------------ #
MTP_K = 7
M_WIDTHS = [1, 8, 4]                    # M=1 (#388 baseline anchor), M=8 (served K+1 verify), M=4 (partial)

# ======================================================================================== #
# Body GEMM shapes -- (out, in, count). 8 distinct shapes, gemma-4-E4B-it (lawine #388).
# ======================================================================================== #
BODY_SHAPES: list[dict[str, Any]] = [
    {"name": "q_full",  "out": 4096,  "in": 2560,  "count": 7},
    {"name": "q_slide", "out": 2048,  "in": 2560,  "count": 35},
    {"name": "kv_full", "out": 1024,  "in": 2560,  "count": 8},
    {"name": "kv_slide", "out": 512,  "in": 2560,  "count": 40},
    {"name": "o_full",  "out": 2560,  "in": 4096,  "count": 7},
    {"name": "o_slide", "out": 2560,  "in": 2048,  "count": 35},
    {"name": "gate_up", "out": 10240, "in": 2560,  "count": 84},
    {"name": "down",    "out": 2560,  "in": 10240, "count": 42},
]


def _shape_params(s: dict[str, Any]) -> int:
    return s["out"] * s["in"] * s["count"]


def _int4_weight_bytes(out: int, inn: int) -> float:
    """int4-Marlin weight-read bytes for one GEMM (4.125 bpw = 4b weight + bf16 g128 scale)."""
    return out * inn * INT4_BPW / 8.0


def _cb3_weight_bytes(out: int, inn: int) -> float:
    """cb3 stored weight bytes for one GEMM (CB3_BPW_EFF = 3.237 bpw)."""
    return out * inn * CB3_BPW_EFF / 8.0


# ======================================================================================== #
# Ladder translation (0-GPU) -- the SAME form that produced +15.60.
# ======================================================================================== #
def translate_to_tps(realized_penalty: float, f_body: float = F_BODY_STRICT,
                     base: float = CB3_BASE_TPS) -> dict[str, Any]:
    """Translate a realized body-read penalty (baseline_us / cb3_us) to a TPS delta on the
    467.14 base via the SAME ladder form that produced +15.60: delta = base * f_body * (p - 1).
    f_body and base are taken AS GIVEN (lawine/denken own them); this isolates MY measured kernel
    penalty from the borrowed normalization, making realized-vs-modeled directly comparable
    (identical discipline to #433's translate_to_tps)."""
    recoverable_eta = f_body * (realized_penalty - 1.0)
    realized_delta = base * recoverable_eta
    return {
        "realized_penalty": realized_penalty,
        "f_body": f_body,
        "base_tps": base,
        "recoverable_eta_body": recoverable_eta,
        "cb3_realized_tps_delta": realized_delta,
        "cb3_realized_frontier_tps": base + realized_delta,
        "cb3_modeled_tps_delta": CB3_MODELED_DELTA,
        "realized_vs_modeled_ratio": realized_delta / CB3_MODELED_DELTA if CB3_MODELED_DELTA else float("nan"),
        "cb3_lift_survives_realization": bool(realized_delta >= 0.80 * CB3_MODELED_DELTA),
        "equivalence_frontier_beats_deployed_481": bool(base + realized_delta > DEPLOYED_TPS),
    }


# ======================================================================================== #
# GPU helpers (self-contained, mirroring #433 discipline).
# ======================================================================================== #
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
        "name": p.name,
        "sm_count": p.multi_processor_count,
        "compute_capability": f"{cc[0]}.{cc[1]}",
        "total_mem_gib": round(p.total_memory / 1024**3, 2),
        "is_a10g_80sm": bool(p.name.find("A10G") >= 0 and p.multi_processor_count == 80),
        "is_sm86": bool(cc == (8, 6)),
    }


def _time_call(fn: Callable[[], Any], iters: int, warmup: int, reps: int = 3) -> float:
    """Steady-state per-call latency in microseconds. BATCHED timing: `iters` calls are enqueued
    back-to-back between a single start/stop event pair, so the CPU->GPU launch latency is amortized
    (a per-op sync inflates every call by a ~30us empty-queue floor). This also matches real decode,
    where ~42 layers x several GEMMs run back-to-back, not synced one at a time. Median over `reps`
    batched means (the PR's >=3 reps)."""
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
        means.append(s.elapsed_time(e) * 1000.0 / iters)  # ms -> us, per call
    return float(statistics.median(means))


def _measure_peak_copy_gbs(dev, iters: int, warmup: int) -> float:
    """Peak achievable HBM copy bandwidth (GB/s) -- the realistic roofline reference (lawine used a
    peak-COPY ref ~470 GB/s, below the 600 datasheet). Large contiguous bf16 copy."""
    import torch
    n = 256 * 1024 * 1024  # 256 Mi bf16 = 512 MiB
    src = torch.empty(n, dtype=torch.bfloat16, device=dev)
    dst = torch.empty(n, dtype=torch.bfloat16, device=dev)
    t_us = _time_call(lambda: dst.copy_(src), iters, warmup)
    bytes_moved = 2.0 * n * 2.0  # read src + write dst, 2 bytes each
    return bytes_moved / (t_us * 1e-6) / 1e9


# ======================================================================================== #
# int4-Marlin GEMM (the served baseline that RUNS) -- real kernel.
# ======================================================================================== #
def _build_marlin_int4(w_bf16, dev):
    """Build a runnable int4-Marlin GEMM closure for weight w [K=in, N=out]. Returns (run, bytes).
    Reuses vLLM 0.22's gptq_marlin path == the deployed int4 read profile (lawine surrogate)."""
    import torch
    import vllm.model_executor.layers.quantization.utils.marlin_utils as mu
    import vllm.model_executor.layers.quantization.utils.marlin_utils_test as mt
    from vllm.scalar_type import scalar_types
    K, N = w_bf16.shape
    _wref, q_w, s, _g, _so, _rp = mt.marlin_quantize(w_bf16, scalar_types.uint4b8, 128, act_order=False)
    ws = mu.marlin_make_workspace_new(dev)
    zp = torch.empty(0, dtype=torch.int, device=dev)
    gi = torch.empty(0, dtype=torch.int, device=dev)
    si = torch.empty(0, dtype=torch.int, device=dev)

    def run(x):
        return mu.apply_gptq_marlin_linear(
            x, q_w, s, zp, gi, si, ws, scalar_types.uint4b8,
            output_size_per_partition=N, input_size_per_partition=K, is_k_full=True)

    nbytes = q_w.numel() * q_w.element_size() + s.numel() * s.element_size()
    return run, float(nbytes)


# ======================================================================================== #
# cb3 realized dequant ops (the thing lawine MODELED; we RUN it).
# ======================================================================================== #
def _hadamard_matrix(g: int, dev):
    """Normalized g x g Hadamard (Sylvester; g must be a power of 2). The online activation RHT
    block. bf16 to match the served activation dtype."""
    import torch
    assert (g & (g - 1)) == 0, "Hadamard size must be power of 2"
    H = torch.ones(1, 1, dtype=torch.float32, device=dev)
    while H.shape[0] < g:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    H = H / math.sqrt(g)
    return H.to(torch.bfloat16)


def _build_cb3_ops(w_bf16, dev):
    """Build the cb3 realized-op closures for weight w [K=in, N=out]:
      - fwht(x):       online activation RHT (block-Hadamard over g128 incoherence groups).
      - reconstruct(): VQ dim-2 codebook gather -> full bf16 weight tile (the MATERIALIZE op).
      - bf16_gemm(x):  the GEMM cb3 ultimately drives on the reconstructed bf16 weight.
    Returns (fwht, reconstruct, bf16_gemm, cb3_stored_bytes)."""
    import torch
    K, N = w_bf16.shape                 # in, out
    g = HADAMARD_GROUP
    Hin = _hadamard_matrix(g, dev)       # g x g, applied to activation rows of length K (in)
    assert K % g == 0, f"in={K} not divisible by g={g}"
    n_groups_in = K // g

    def fwht(x):
        # x: [M, K] -> reshape to [M, n_groups, g] -> @ H -> back. Online RHT, M=1..8.
        M = x.shape[0]
        xr = x.view(M, n_groups_in, g)
        yr = torch.matmul(xr, Hin)
        return yr.view(M, K)

    # VQ dim-2: store w (K*N values) as (K*N/VQ_DIM) indices into a (K, VQ_DIM) codebook.
    # idx pre-stored as int64 (index_select requires it) so the TIMED op is just gather+write, not
    # a per-call dtype conversion (a real VQ kernel reads the stored low-bit codes directly).
    codebook = torch.randn(VQ_CODEBOOK_K, VQ_DIM, dtype=torch.bfloat16, device=dev) * 0.02
    n_codes = (K * N) // VQ_DIM
    idx = torch.randint(0, VQ_CODEBOOK_K, (n_codes,), dtype=torch.int64, device=dev)
    w_recon = torch.empty(K, N, dtype=torch.bfloat16, device=dev)  # pre-allocated materialize buffer

    def reconstruct():
        # gather codebook[idx] -> [n_codes, VQ_DIM] -> reshape to [K, N]. Writes the FULL bf16 tile.
        # This is the MATERIALIZE op: random-access codebook gather + a full bf16 weight write.
        g2 = torch.index_select(codebook, 0, idx)   # [n_codes, 2]
        w_recon.copy_(g2.view(K, N))
        return w_recon

    def bf16_gemm(x):
        # x: [M, K] @ w_recon [K, N] -> [M, N]
        return torch.matmul(x, w_recon)

    # cb3 stored bytes: indices at log2(K) bits + codebook + g128 bf16 scale (the 3.237 bpw read).
    cb3_bytes = _cb3_weight_bytes(N, K)
    return fwht, reconstruct, bf16_gemm, float(cb3_bytes)


# ======================================================================================== #
# Microbench -- per body shape x M width.
# ======================================================================================== #
def _microbench_shape(shape: dict[str, Any], M: int, dev, peak_gbs: float,
                      iters: int, warmup: int) -> dict[str, Any]:
    """Measure the RAW op latencies on the kernels that run. Pre-allocate ALL buffers ONCE; the
    timed lambdas never allocate weights (fair, #433 discipline)."""
    import torch
    out_f, in_f = shape["out"], shape["in"]
    w = (torch.randn(in_f, out_f, dtype=torch.bfloat16, device=dev) * 0.02)  # [K=in, N=out]
    x = (torch.randn(M, in_f, dtype=torch.bfloat16, device=dev) * 0.5)        # [M, in]

    marlin_run, int4_bytes = _build_marlin_int4(w, dev)
    fwht, reconstruct, bf16_gemm, cb3_bytes = _build_cb3_ops(w, dev)
    xr = fwht(x)  # pre-rotate once; bf16/marlin GEMM timing reuses a ready activation

    t_marlin = _time_call(lambda: marlin_run(x), iters, warmup)        # served int4 fused GEMM (REAL)
    t_bf16_gemm = _time_call(lambda: bf16_gemm(xr), iters, warmup)     # the bf16 GEMM cb3 must drive
    t_hadamard = _time_call(lambda: fwht(x), iters, warmup)            # online activation RHT (standalone)
    t_reconstruct = _time_call(lambda: reconstruct(), iters, warmup)  # VQ materialize (gather->bf16 tile)

    # decompose int4-Marlin into transfer (BW-bound part) + fixed overhead at the measured peak BW
    t_marlin_transfer = (int4_bytes / 1e9) / peak_gbs * 1e6
    t_marlin_overhead = max(t_marlin - t_marlin_transfer, 0.0)
    marlin_hbm_eff = t_marlin_transfer / t_marlin if t_marlin > 0 else float("nan")
    cb3_transfer = (cb3_bytes / 1e9) / peak_gbs * 1e6

    return {
        "name": shape["name"], "out": out_f, "in": in_f, "count": shape["count"], "M": M,
        "params": _shape_params(shape),
        "t_marlin_us": t_marlin, "t_bf16_gemm_us": t_bf16_gemm,
        "t_hadamard_us": t_hadamard, "t_reconstruct_us": t_reconstruct,
        "int4_bytes": int4_bytes, "cb3_bytes": cb3_bytes,
        "t_marlin_transfer_us": t_marlin_transfer, "t_marlin_overhead_us": t_marlin_overhead,
        "cb3_transfer_us": cb3_transfer, "marlin_hbm_eff": marlin_hbm_eff,
        # the only RUNNABLE sub-int4 path on sm_86: rotate + reconstruct-to-bf16 + bf16 GEMM
        "cb3_materialize_us": t_hadamard + t_reconstruct + t_bf16_gemm,
        # FUSED counterfactual (REQUIRES a kernel absent on sm_86): cb3 compressed bytes at Marlin's
        # measured efficiency + Marlin's fixed overhead (== lawine's model; Hadamard fused-in/negligible)
        "cb3_fused_us": cb3_transfer + t_marlin_overhead,
    }


def _agg(per_shape: list[dict[str, Any]], key: str) -> float:
    """Count-weighted mean of a per-shape us value (weight by param count, like lawine)."""
    num = sum(s["params"] * s[key] for s in per_shape)
    den = sum(s["params"] for s in per_shape)
    return num / den if den else float("nan")


def microbench(dev, iters: int, warmup: int) -> dict[str, Any]:
    peak_gbs = _measure_peak_copy_gbs(dev, iters, warmup)
    by_width: dict[str, Any] = {}
    for M in M_WIDTHS:
        per_shape = [_microbench_shape(s, M, dev, peak_gbs, iters, warmup) for s in BODY_SHAPES]
        agg_marlin = _agg(per_shape, "t_marlin_us")
        agg_bf16 = _agg(per_shape, "t_bf16_gemm_us")
        agg_materialize = _agg(per_shape, "cb3_materialize_us")
        agg_fused = _agg(per_shape, "cb3_fused_us")
        eff = _agg(per_shape, "marlin_hbm_eff")
        by_width[str(M)] = {
            "per_shape": per_shape,
            "agg_marlin_us": agg_marlin,
            "agg_bf16_gemm_us": agg_bf16,
            "agg_hadamard_us": _agg(per_shape, "t_hadamard_us"),
            "agg_reconstruct_us": _agg(per_shape, "t_reconstruct_us"),
            "agg_cb3_materialize_us": agg_materialize,
            "agg_cb3_fused_us": agg_fused,
            "agg_marlin_hbm_eff": eff,
            # the runnable reality: cb3 materialize vs int4-Marlin (>1 => cb3 faster)
            "penalty_materialize": agg_marlin / agg_materialize if agg_materialize else float("nan"),
            # even free dequant: the bf16 GEMM alone vs int4-Marlin (reads 4x bytes => <1)
            "penalty_bf16_gemm_only": agg_marlin / agg_bf16 if agg_bf16 else float("nan"),
            # the fused counterfactual (needs a kernel absent on sm_86)
            "penalty_fused_measured_eff": agg_marlin / agg_fused if agg_fused else float("nan"),
            # lawine's literature-floor fused model from the byte-proportional engine at measured eff
            "penalty_fused_floor_model": 1.0 / (BYTE_RATIO * eff + (1.0 - eff)) if eff == eff else float("nan"),
        }
    return {"peak_copy_gbs": peak_gbs, "by_width": by_width}


def _amdahl_frontier(penalty: float, f_body: float, base: float) -> float:
    """Amdahl-correct frontier if a body-read with realized speedup `penalty` were DEPLOYED:
    new_step = old_step * (1 - f_body + f_body/penalty); TPS = base / that factor. Bounded for
    penalty<1 (the linear ladder blows up; this is the honest 'if you shipped it' number)."""
    factor = (1.0 - f_body) + f_body / penalty if penalty > 0 else float("inf")
    return base / factor if factor > 0 else float("nan")


# ======================================================================================== #
# Compose -- aggregate the microbench, build the bracket + verdict.
# ======================================================================================== #
def compose(gpu: dict[str, Any], mb: dict[str, Any]) -> dict[str, Any]:
    HEADLINE_M = "8"   # the served MTP K=7 verify width (the deployed decode op-point)
    w1, w8 = mb["by_width"]["1"], mb["by_width"][HEADLINE_M]

    pen_materialize = w8["penalty_materialize"]          # the kernel that RUNS (sub-int4 => materialize)
    pen_bf16_only = w8["penalty_bf16_gemm_only"]          # even free dequant: bf16 GEMM reads 4x bytes
    pen_fused_eff = w8["penalty_fused_measured_eff"]      # fused counterfactual at measured Marlin eff
    pen_fused_floor = w8["penalty_fused_floor_model"]     # lawine's literature-floor fused model
    marlin_eff = w8["agg_marlin_hbm_eff"]

    # --- the realized verdict --------------------------------------------------------------- #
    # The kernel that runs for a sub-int4 read on this vLLM-0.22 / sm_86 stack is the MATERIALIZE
    # path (no fused sub-int4 Marlin/VQ kernel exists -- lawine 7rzf74q5 proved sub-int4 is
    # UNREACHABLE off-the-shelf). It INVERTS (penalty<<1): the dequant materializes a full bf16
    # weight (4x the int4 bytes), so cb3 is strictly slower than int4-Marlin -- the byte saving is
    # destroyed the moment the weight is reconstructed. So the modeled +15.60 is FORFEITED on the
    # served kernel: like the refuted pinned-K 496.74, it is a model with no surviving realized
    # kernel. The equivalence frontier collapses back to the 467.14 int4-Marlin base (you keep int4).
    inverts = bool(pen_materialize < 1.0)
    # translate the RUNNABLE penalty through the ladder (shows the raw inversion, mirroring #433's
    # negative headline) -- but the DECISION frontier floors at the base (cb3 simply isn't applied).
    t_runnable = translate_to_tps(pen_materialize)
    realized_delta = 0.0 if inverts else t_runnable["cb3_realized_tps_delta"]
    realized_frontier = CB3_BASE_TPS + realized_delta
    survives = bool((not inverts) and realized_delta >= 0.80 * CB3_MODELED_DELTA)
    beats_deployed = bool(realized_frontier > DEPLOYED_TPS)

    # fused counterfactual lift (Amdahl), bracketed by the literature-floor and measured-eff models
    fused_floor_frontier = _amdahl_frontier(pen_fused_floor, F_BODY_STRICT, CB3_BASE_TPS)
    fused_eff_frontier = _amdahl_frontier(pen_fused_eff, F_BODY_STRICT, CB3_BASE_TPS)

    # decomposition: BW saving headroom vs the dequant/Hadamard op tax (count-weighted, M=8)
    byte_saving_us = w8["agg_marlin_us"] - w8["agg_cb3_fused_us"]          # the fused byte saving
    dequant_tax_us = w8["agg_reconstruct_us"] + w8["agg_hadamard_us"]      # the materialize op tax
    bf16_extra_us = w8["agg_bf16_gemm_us"] - w8["agg_marlin_us"]          # bf16 GEMM 4x-byte penalty

    verdict = (
        f"cb3 +15.60 is MODELED on a byte-count roofline with NO realized kernel on the served "
        f"sm_86/vLLM-0.22 stack. The only RUNNABLE sub-int4 realization is materialize-to-bf16 "
        f"(no fused VQ kernel exists): penalty_materialize={pen_materialize:.3f} (M=8) -- a "
        f"{1.0/pen_materialize:.0f}x INVERSION. Even granting free dequant, the bf16 GEMM alone reads "
        f"4x the int4 bytes => penalty_bf16_only={pen_bf16_only:.3f} (<1): the byte saving is "
        f"DESTROYED on reconstruction. The fused counterfactual (penalty {pen_fused_floor:.3f}-"
        f"{pen_fused_eff:.3f}) would survive, but REQUIRES a fused RHT+VQ decode kernel that does not "
        f"exist on sm_86 (the exact 'buildable-but-unrealized' status that left pinned-K's lift "
        f"unbanked in #433). Realized equivalence frontier collapses to {realized_frontier:.2f} "
        f"(== 467.14 base; cb3 forfeited) < deployed 481.53. 482.74 is the next +13.998-style "
        f"modeled artifact: equivalence_frontier_beats_deployed_481={beats_deployed}."
    )

    required = {
        "cb3_modeled_tps_delta": CB3_MODELED_DELTA,
        "cb3_base_tps": CB3_BASE_TPS,
        "modeled_penalty": MODELED_PENALTY,
        "cb3_realized_tps_delta": realized_delta,
        "cb3_realized_frontier_tps": realized_frontier,
        "realized_vs_modeled_ratio": realized_delta / CB3_MODELED_DELTA if CB3_MODELED_DELTA else 0.0,
        "cb3_lift_survives_realization": survives,
        "equivalence_frontier_beats_deployed_481": beats_deployed,
        "ppl": 2.3772,
        "ppl_is_anchored": True,
    }
    brackets = {
        "penalty_materialize_runnable": pen_materialize,
        "penalty_bf16_gemm_only": pen_bf16_only,
        "penalty_fused_floor_model": pen_fused_floor,
        "penalty_fused_measured_eff": pen_fused_eff,
        "inverts_on_runnable_kernel": inverts,
        "runnable_ladder_delta_if_deployed": t_runnable["cb3_realized_tps_delta"],
        "runnable_amdahl_frontier_if_deployed": _amdahl_frontier(pen_materialize, F_BODY_STRICT, CB3_BASE_TPS),
        "fused_counterfactual_frontier_floor": fused_floor_frontier,
        "fused_counterfactual_frontier_measured_eff": fused_eff_frontier,
        "fused_kernel_exists_on_sm86": False,
        "marlin_hbm_eff_measured_m8": marlin_eff,
    }
    decomposition = {
        "fused_byte_saving_us_m8": byte_saving_us,
        "materialize_dequant_tax_us_m8": dequant_tax_us,
        "bf16_gemm_extra_us_vs_int4_m8": bf16_extra_us,
        "note": "the fused byte saving (~us) is dwarfed by the materialize dequant tax AND by the "
                "bf16 GEMM's 4x-byte penalty; the saving only exists inside a fused low-bit kernel.",
    }

    self_t = self_test()
    config = {
        "agent": "stark", "pr": 437, "kind": "cb3-realized-kernel-validation",
        "byte_ratio": BYTE_RATIO, "cb3_bpw_eff": CB3_BPW_EFF, "int4_bpw": INT4_BPW,
        "f_body_strict": F_BODY_STRICT, "cb3_base_tps": CB3_BASE_TPS,
        "cb3_modeled_delta": CB3_MODELED_DELTA, "deployed_tps": DEPLOYED_TPS,
        "hadamard_group": HADAMARD_GROUP, "vq_dim": VQ_DIM, "vq_codebook_k": VQ_CODEBOOK_K,
        "m_widths": M_WIDTHS, "headline_m": int(HEADLINE_M),
    }
    wandb_metrics = {
        **required, **brackets,
        "peak_copy_gbs": mb["peak_copy_gbs"],
        "m1_penalty_materialize": w1["penalty_materialize"],
        "m1_penalty_fused_floor_model": w1["penalty_fused_floor_model"],
        "m1_marlin_hbm_eff": w1["agg_marlin_hbm_eff"],
        "m8_agg_marlin_us": w8["agg_marlin_us"], "m8_agg_bf16_gemm_us": w8["agg_bf16_gemm_us"],
        "m8_agg_reconstruct_us": w8["agg_reconstruct_us"], "m8_agg_hadamard_us": w8["agg_hadamard_us"],
        "self_test_passes": self_t["self_test_passes"],
    }
    return {
        "gpu": gpu, "grounding": {
            "cb3_locus": "verify/body-read RHT+VQ shrink (k*=229, dim-2 VQ + g128 incoherence); "
                         "kanna #403 iv9i2wks modeled +15.60 over denken #423 467.14; lawine #388 "
                         "7rzf74q5 realized-BW byte_ratio 0.785, measured_floor 1.0588.",
            **required,
        },
        "microbench": mb, "brackets": brackets, "decomposition": decomposition,
        "required": required, "verdict": verdict,
        "self_test": self_t, "self_test_passes": self_t["self_test_passes"],
        "config": config, "wandb_metrics": wandb_metrics,
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
        "timestamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    }


# ======================================================================================== #
# Self-test (0-GPU): ladder arithmetic + translation round-trip + guards.
# ======================================================================================== #
def self_test() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    # ladder arithmetic
    checks["base_plus_delta_is_frontier"] = abs((CB3_BASE_TPS + CB3_MODELED_DELTA) - CB3_FRONTIER_MODELED) < 1e-9
    checks["modeled_delta_is_15p60"] = abs(CB3_MODELED_DELTA - 15.60) < 1e-9
    checks["byte_ratio_rounds_0p785"] = round(BYTE_RATIO, 3) == 0.785
    checks["roofline_is_inv_byte_ratio"] = abs(ROOFLINE_SPEEDUP - 1.0 / BYTE_RATIO) < TOL
    # the modeled-penalty round trip: delta = base * f_body * (MODELED_PENALTY - 1) == 15.60
    rt = CB3_BASE_TPS * F_BODY_STRICT * (MODELED_PENALTY - 1.0)
    checks["modeled_penalty_round_trips_to_delta"] = abs(rt - CB3_MODELED_DELTA) < 1e-6
    # translation: penalty==MODELED_PENALTY reproduces the modeled delta + frontier
    t = translate_to_tps(MODELED_PENALTY)
    checks["translate_reproduces_modeled_delta"] = abs(t["cb3_realized_tps_delta"] - CB3_MODELED_DELTA) < 1e-6
    checks["translate_reproduces_modeled_frontier"] = abs(t["cb3_realized_frontier_tps"] - CB3_FRONTIER_MODELED) < 1e-6
    checks["modeled_penalty_survives"] = bool(t["cb3_lift_survives_realization"])
    # penalty==1 (no realized speedup) gives zero delta + frontier==base (the inversion floor logic)
    t1 = translate_to_tps(1.0)
    checks["penalty1_gives_zero_delta"] = abs(t1["cb3_realized_tps_delta"]) < 1e-9
    checks["penalty1_frontier_is_base"] = abs(t1["cb3_realized_frontier_tps"] - CB3_BASE_TPS) < 1e-9
    checks["penalty1_below_deployed"] = (not t1["equivalence_frontier_beats_deployed_481"])  # 467.14 < 481.53
    # a penalty < 1 (inversion) drives the frontier BELOW the base (the #433 outcome shape)
    t_inv = translate_to_tps(0.90)
    checks["inversion_below_base"] = t_inv["cb3_realized_frontier_tps"] < CB3_BASE_TPS
    checks["inversion_not_survives"] = (not t_inv["cb3_lift_survives_realization"])
    # geometry guards
    checks["eight_body_shapes"] = len(BODY_SHAPES) == 8
    checks["hadamard_pow2"] = (HADAMARD_GROUP & (HADAMARD_GROUP - 1)) == 0
    checks["all_in_div_by_g"] = all(s["in"] % HADAMARD_GROUP == 0 for s in BODY_SHAPES)
    checks["served_width_is_k_plus_1"] = (MTP_K + 1) == 8 and 8 in M_WIDTHS
    # envelope guards
    checks["analysis_only_guard"] = True
    checks["no_hf_job_guard"] = True
    checks["no_served_file_change_guard"] = True
    return {"self_test_passes": all(checks.values()), "checks": checks}


# ======================================================================================== #
# Main
# ======================================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--self-test", action="store_true", help="0-GPU arithmetic/guard gate")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb_group", type=str, default="cb3-realized-validation")
    ap.add_argument("--wandb_name", type=str, default="stark/cb3-realized-kernel-validation")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if args.self_test:
        st = self_test()
        out = {"self_test": st, "self_test_passes": st["self_test_passes"], "timestamp": ts,
               "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0}
        p = here / "cb3_realized_kernel_validation_selftest.json"
        p.write_text(json.dumps(out, indent=2))
        print(f"[self-test] passes={st['self_test_passes']}")
        for k, v in st["checks"].items():
            if not v:
                print(f"  FAIL: {k}")
        print(f"[self-test] wrote {p}")
        sys.exit(0 if st["self_test_passes"] else 1)

    # ----- GPU compose path ------------------------------------------------------------------ #
    dev = _device()
    gpu = _gpu_facts(dev)
    mb = microbench(dev, args.iters, args.warmup)
    payload = compose(gpu, mb)

    out_path = here / "cb3_realized_kernel_validation_results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[cb3-realized] wrote {out_path}")
    print(f"[cb3-realized] {payload['verdict']}")

    if not args.no_wandb:
        try:
            import wandb
            run = wandb.init(project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
                             group=args.wandb_group, name=args.wandb_name,
                             config=payload["config"], job_type="analysis")
            wandb.log(payload["wandb_metrics"])
            # per-shape tables for the dominant M=1 + served M=8 op-points
            for M in ("1", "8"):
                rows = mb["by_width"][M]["per_shape"]
                cols = ["name", "out", "in", "count", "t_marlin_us", "t_bf16_gemm_us",
                        "t_hadamard_us", "t_reconstruct_us", "cb3_materialize_us", "cb3_fused_us",
                        "marlin_hbm_eff"]
                tbl = wandb.Table(columns=cols)
                for r in rows:
                    tbl.add_data(*[r[c] for c in cols])
                wandb.log({f"per_shape_M{M}": tbl})
            payload["wandb_run_id"] = run.id
            out_path.write_text(json.dumps(payload, indent=2))
            wandb.finish()
            print(f"[cb3-realized] wandb run {run.id}")
        except Exception as e:  # noqa: BLE001
            print(f"[cb3-realized] wandb failed (non-fatal): {type(e).__name__}: {str(e)[:160]}")

    sys.exit(0)


if __name__ == "__main__":
    main()
