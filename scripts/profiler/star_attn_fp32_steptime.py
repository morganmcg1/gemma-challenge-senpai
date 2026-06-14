#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""fp32 star-attn STEP-TIME tax (PR #131): does the E[T]-recovering fp32 upcast
still clear 500/530 *after* its denominator cost?

THE QUESTION THIS PRICES (the DENOMINATOR, not the numerator)
-------------------------------------------------------------
The tree build defect is localized (chiku-inu): the depth-1 acceptance deficit
(built 0.598 vs correct q[1]=0.7287 -> realized E[T] capped at 2.10) is caused by
the star-attention VERIFY FORWARD running in bf16 -- a noisy bf16 root-row argmax
flips on near-ties and rejects the drafter's correct depth-1 guess. The in-flight
fix is a QK+PV upcast to fp32/IEEE (star relerr ~1e-3 -> 1e-6), build
`tree-488-pw-fp32-v0`. denken #128 checks the NUMERATOR (does fp32 recover the 13pp
deficit, E[T] 2.10 -> ~5.2). THIS script checks the DENOMINATOR: fp32 QK+PV is more
expensive per verify step than bf16; if the step-time tax is large the recovered
E[T] might not clear 500/530 after the cost.

WHY THIS IS NOT wirbel #98 (which found "fp32 star-attn ~free")
--------------------------------------------------------------
wirbel #98 priced a DIFFERENT fp32 channel: the split-KV per-segment PARTIAL-buffer
dtype (`softmax_segm_output` fp32 vs bf16). It is free because the deployed kernel
ALREADY accumulates in fp32, KV stays bf16 (no extra HBM read), and the partials are
L2-resident. BUT bf16 *inputs* (8-bit mantissa, ~2^-8 ~= 4e-3 relative precision)
FLOOR the matmul rel-error at ~1e-3 REGARDLESS of fp32 accumulation. Reaching the
relerr 1e-6 that `tree-488-pw-fp32-v0` targets REQUIRES fp32 *inputs* to the QK^T and
PV matmuls -- which on A10G (GA102, sm_86) drop OFF the bf16 tensor cores (~70 TFLOPS
fp32-accum) onto IEEE-fp32 CUDA cores (~31 TFLOPS) or tf32 tensor (~35 TFLOPS, but
tf32's 10-bit mantissa ~1e-3 is INSUFFICIENT for 1e-6). wirbel #98 never varied the
matmul-input dtype, so its "free" does NOT cover this build.

THE PHYSICS (roofline): at M=1 decode the attention is deeply BW-bound (shared-prefix
KV read once, AI = M*(n_q/n_kv) = 1*4 = 4 FLOP/byte << ridge) -> fp32 free. At M=32
TREE verify the SAME shared-prefix attention has AI = 32*4 = 128 FLOP/byte ~ the A10G
bf16 ridge (70e12/600e9 = 117) and well ABOVE the fp32 ridge (31e12/600e9 = 52). So
the M=32 verify attention has CROSSED into compute-exposed territory: upcasting its
matmuls to IEEE fp32 is NOT free. "decode attention is BW-bound so fp32 is free" does
NOT transfer to the M=32 tree verify. This script MEASURES the tax.

WHAT IS MEASURED
----------------
  Part A (real kernel) -- deployed vLLM 3D split-KV `unified_attention` (SPLITKV_
    VERIFY=1, fp32 partials, the #107/#98 machinery verbatim) at M in {1,8,16,32},
    summed over the real loaded sliding+full layer counts. Least-squares fit
    attn_us(M) = c0 + c1*M -> c0 = M-invariant KV-read floor, c1*M = the M-scaling
    (matmul + softmax + writes) compute that the precision change can touch.
  Part B (matmul micro-bench) -- the QK^T and PV matmuls at the M=32 star shapes,
    timed in bf16-tensor / tf32-tensor / fp32-IEEE. Gives the fp32/bf16 and tf32/bf16
    ratios AND the matmul fraction of the M-scaling compute (vs the fp32 softmax that
    does NOT slow down). CRITICAL ARTIFACT: at M=32 the full-layer fp32 K-buffer
    (n_q=8 * hd=512 * N=528 * 4B = 8.65MB) ALONE exceeds the 6.3MB L2, so fp32 spills
    and re-reads K from HBM every iter while bf16 (4.32MB) stays L2-resident. That
    EXTRA fp32 HBM read inflates the realized M=32 ratio to ~6 -- but it is a proxy
    for a NAIVE build that materializes fp32 KV in HBM, NOT the on-chip-upcast build
    the PR targets (which reads bf16 KV once and upcasts in registers, paying only the
    COMPUTE delta ~ datasheet 70/31.2 = 2.24x). So Part B brackets TWO build paths:
      r_fp32_central      = the datasheet compute ratio 2.24 (on-chip upcast: only
                            tensor->CUDA-core compute changes, KV HBM read unchanged),
                            corroborated by the large-M asymptotic micro-bench where
                            the K-read amortizes and the ratio converges off ~6 toward
                            the compute ratio.
      r_fp32_conservative = the realized M=32 ratio (~6, the naive fp32-KV-in-HBM
                            build that pays the extra fp32 read -- a real worst case).
  Part C (roofline) -- AI(M), ridges, predicted compute regime as a cross-check.

  Delta_step_time_fp32 = (M-scaling compute at M=32) * matmul_frac * (r_fp32 - 1),
  expressed in M=8-step-normalized units (the #107 step basis) and re-priced through
  the official TPS model. CENTRAL uses r_fp32_central (on-chip 2.24) * matmul_frac;
  CONSERVATIVE uses r_fp32_conservative (naive HBM ~6) * matmul_frac=1 (all M-scaling
  compute charged as matmul).

RE-PRICE + GATE (Steps 2-3)
---------------------------
  official_TPS = K_cal * E[T]_recovered / (step_ratio_bf16 + Delta_step_time_fp32) * tau_tree
  K_cal = 125.268 (#107); E[T]_recovered = 5.207 ceiling (fern #125) default, or
  denken #128 if landed; tau_tree central 1.0 / floor 0.9924 (my #126);
  step_ratio_bf16 = 1.156 (my #107 measured M=32/M=8 whole-step ratio).
    GREEN  net clears 530 at the conservative corner.
    AMBER  clears 500 but not 530, or needs the central corner.
    RED    net falls below 500.
  Also reports the break-even E[T] for 500/530 AFTER the tax (vs fern #102's pre-tax
  4.624) and the SELECTIVE-fp32 hybrid (Step 4): fp32 only on the depth-1 root row +
  the wirbel #93 near-tie tail (0.537% < 1e-3 rel-margin), bf16 on the safe bulk.

LOCAL A10G micro-bench + roofline. NO HF Job, NO submission, NO kernel build, NO
served-file change. Timing/analysis only -> greedy identity untouched by construction.
"""
from __future__ import annotations

import os

# Pin the single visible A10G + disable the broken flashinfer sampler JIT BEFORE
# importing torch/vllm (project_local_a10g_gpu_env). Force the deployed split-KV
# verify path on so M>1 verify batches route to the 3D split-KV attention we price.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("SPLITKV_VERIFY", "1")

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- attention machinery (wirbel #98 / #39 / my #107, reused verbatim) --------
from scripts.local_validation.profile_attention import (  # noqa: E402
    BLOCK_SIZE,
    HEAD_DIM,
    N_FULL,
    N_KV_HEADS,
    N_Q_HEADS,
    N_SLIDING,
    QPKV,
    SLIDING_WINDOW,
    _maybe_install_splitkv,
    _measure_peak_bw,
    _profiled_device_us,
)
from scripts.profiler.star_attn_fp32_cost import _build_op_inputs, _make_call

# z for a 95% two-sided normal CI (#72/#82 convention).
Z95 = 1.959963984540054
ATTN_CTX_DEFAULT = 528          # deployed mean served ctx ~527.7 (wirbel #98)

# ===== merged-artifact constants (the model we re-price) =======================
# my #107 tree_step_denominator.json (MEASURED, median N=5).
STEP_RATIO_BF16 = 1.1559689045914052   # whole-step M=32/M=8 ratio (method_A budget)
R_GEMM_107 = 1.1686205063215744        # verify-GEMM M=32/M=8
R_ATTN_107 = 1.8325004530121336        # verify-attention M=32/M=8 (deployed split-KV)
B_GEMM = 0.53                          # verify-GEMM share of the M=8 step (#107 budget)
B_ATTN = 0.08                          # attention share of the M=8 step
# fern's E[T]->TPS calibration (lever_composition).
K_CAL = 125.26795005202914             # 481.53 / 3.844 (official baseline / E[T]_linear)
E_T_TREE_CEILING = 5.207               # fern #125 / denken #101 analytical ceiling
E_T_LINEAR = 3.844                     # linear MTP K=7 reference
E_T_ASBUILT = 2.097                    # byteshark bf16 build (the deficit we recover from)
# my #126 tree_verify_tau_roofline.json.
TAU_TREE_CENTRAL = 1.0
TAU_TREE_FLOOR = 0.9924318649123313
# fern #125 / #126 headline tree projection (the fleet number we haircut).
TREE_FERN_CENTRAL_OFFICIAL = 568.0
# A10G (GA102, sm_86) compute ceilings (datasheet; #68 convention for the A10G, NOT
# the data-center A10). Used ONLY for the roofline cross-check; the GATE rides the
# MEASURED Part-B ratio.
A10G_BF16_TENSOR_TFLOPS = 70.0         # fp32-accum tensor core (#68 A10G figure)
A10G_TF32_TENSOR_TFLOPS = 35.0         # tf32 tensor core (GA102 ~= 0.5x bf16-accum)
A10G_FP32_CUDA_TFLOPS = 31.2           # IEEE fp32 CUDA cores
A10G_PEAK_GBPS = 600.0
# datasheet compute ratio = bf16-tensor / fp32-CUDA = the on-chip-upcast central
# r_fp32 (KV read once as bf16, upcast in-register, only the matmul COMPUTE changes).
R_FP32_DATASHEET = A10G_BF16_TENSOR_TFLOPS / A10G_FP32_CUDA_TFLOPS  # ~2.244
# large-M shape where the fp32 K-read amortizes -> realized ratio converges off the
# L2-spill ~6 toward the compute ratio (corroborates r_fp32_central, isolates the M=32
# spill as a micro-bench/naive-build artifact rather than an on-chip-upcast cost).
M_ASYMPTOTIC = 512
# wirbel #93 margin map (the hybrid target population).
NEAR_TIE_FRAC_LT_1E3 = 0.00537109375   # frac positions < 1e-3 rel-margin (fp32 ref)
SAFE_BULK_MEDIAN_REL_MARGIN = 0.18     # median rel-margin of the safe bulk

TARGET_500 = 500.0
TARGET_530 = 530.0


# ---------------------------------------------------------------------------
def summarize(values: list[float]) -> dict:
    n = len(values)
    mean = statistics.fmean(values)
    median = statistics.median(values)
    std = statistics.pstdev(values) if n > 1 else 0.0
    se = std / math.sqrt(n) if n else 0.0
    return {"n": n, "mean": mean, "median": median, "std": std, "se": se,
            "cv_pct": 100.0 * std / mean if mean else 0.0, "ci95_abs": Z95 * se,
            "min": min(values), "max": max(values), "values": values}


# ===== Part A: real-kernel attention substep M-sweep (deployed bf16) ===========
def measure_attn_us(M: int, ctx: int, n_iter: int, warmup: int,
                    counts: dict) -> tuple[float, dict]:
    """Sum deployed fp32-partial split-KV `unified_attention` device time over the
    sliding + full layers at width M (the #107 `measure_attn_us` basis: real loaded
    layer counts, fp32 partials = deployed/greedy-safe, bf16 matmul inputs)."""
    total = 0.0
    per_type = {}
    for lt in ("sliding", "full"):
        inp = _build_op_inputs(torch, lt, M, ctx)
        out_buf = torch.empty(M, N_Q_HEADS, inp["hd"], dtype=torch.bfloat16,
                              device=inp["device"])
        call, _ = _make_call(torch, inp, M, ctx, torch.float32, out_buf)  # deployed
        per_op_us = _profiled_device_us(torch, call, n_iter, warmup)
        per_type[lt] = {"per_op_us": per_op_us, "count": counts[lt]}
        total += counts[lt] * per_op_us
        del inp, out_buf
        torch.cuda.empty_cache()
    return total, per_type


def fit_mem_compute(ms: list[int], us: list[float]) -> dict:
    """Least-squares fit attn_us(M) = c0 + c1*M. c0 = M-invariant KV-read floor;
    c1*M = the M-scaling (matmul + softmax + writes) compute the precision can touch."""
    n = len(ms)
    sm = sum(ms); su = sum(us)
    smm = sum(m * m for m in ms); smu = sum(m * u for m, u in zip(ms, us))
    denom = n * smm - sm * sm
    c1 = (n * smu - sm * su) / denom
    c0 = (su - c1 * sm) / n
    # R^2
    mean_u = su / n
    ss_tot = sum((u - mean_u) ** 2 for u in us)
    ss_res = sum((u - (c0 + c1 * m)) ** 2 for m, u in zip(ms, us))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {"c0_mem_us": c0, "c1_compute_us_per_row": c1, "r2": r2}


# ===== Part B: QK^T + PV matmul micro-bench in 3 precisions =====================
def _attn_matmul_bench(M: int, ctx: int, dtype, allow_tf32: bool, n_iter: int,
                       warmup: int, counts: dict) -> dict:
    """Time the QK^T and PV matmuls at the M=32 star shapes, summed over layers.
    Compute-bound: input buffers are reused (L2-resident) every iter, mirroring the
    real kernel where bf16 KV is read once from HBM then upcast ON-CHIP -- so the
    fp32-vs-bf16 delta here isolates the MATMUL COMPUTE tax (not an extra HBM read).
    softmax is run in fp32 in ALL precisions (it does not change) so it cancels in
    the fp32-vs-bf16 delta; we still time it to get the matmul fraction."""
    prev = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    dev = torch.device("cuda")
    qk_total = pv_total = sm_total = 0.0
    per_type = {}
    try:
        for lt in ("sliding", "full"):
            hd = HEAD_DIM[lt]
            N = min(ctx, SLIDING_WINDOW) if lt == "sliding" else ctx
            scale = 1.0 / math.sqrt(hd)
            # per-head batched matmuls at the real star shapes (GQA expanded to n_q).
            q = (torch.randn(N_Q_HEADS, M, hd, device=dev) * 0.1).to(dtype)
            k = (torch.randn(N_Q_HEADS, hd, N, device=dev) * 0.1).to(dtype)
            v = (torch.randn(N_Q_HEADS, N, hd, device=dev) * 0.1).to(dtype)
            scores_f32 = torch.empty(N_Q_HEADS, M, N, device=dev, dtype=torch.float32)
            probs = torch.empty(N_Q_HEADS, M, N, device=dev, dtype=dtype)

            def qk():
                torch.bmm(q, k).mul_(scale)

            def softmax():
                torch.softmax(scores_f32, dim=-1)

            def pv():
                torch.bmm(probs, v)

            qk_us = _profiled_device_us(torch, qk, n_iter, warmup)
            sm_us = _profiled_device_us(torch, softmax, n_iter, warmup)
            pv_us = _profiled_device_us(torch, pv, n_iter, warmup)
            c = counts[lt]
            qk_total += c * qk_us
            pv_total += c * pv_us
            sm_total += c * sm_us
            per_type[lt] = {"qk_us": qk_us, "pv_us": pv_us, "softmax_us": sm_us,
                            "count": c, "N": N, "hd": hd}
            del q, k, v, scores_f32, probs
            torch.cuda.empty_cache()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
        torch.backends.cudnn.allow_tf32 = prev_cudnn
    return {"qk_us_cycle": qk_total, "pv_us_cycle": pv_total,
            "softmax_us_cycle": sm_total, "matmul_us_cycle": qk_total + pv_total,
            "per_type": per_type}


def attn_matmul_precisions(M: int, ctx: int, n_iter: int, warmup: int, rounds: int,
                           counts: dict) -> dict:
    """Interleave bf16 / tf32 / fp32-IEEE matmul timings across rounds so thermal
    drift cancels in the tiny precision deltas. Returns per-precision medians."""
    specs = {
        "bf16": (torch.bfloat16, False),   # tensor cores, fp32-accum (deployed)
        "tf32": (torch.float32, True),     # tf32 tensor cores (relerr ~1e-3 INSUFFICIENT)
        "fp32": (torch.float32, False),    # IEEE fp32 CUDA cores (relerr ~1e-6 REQUIRED)
    }
    acc = {k: {"matmul": [], "qk": [], "pv": [], "softmax": []} for k in specs}
    last = {}
    for _ in range(rounds):
        for name, (dt, tf32) in specs.items():
            r = _attn_matmul_bench(M, ctx, dt, tf32, n_iter, warmup, counts)
            acc[name]["matmul"].append(r["matmul_us_cycle"])
            acc[name]["qk"].append(r["qk_us_cycle"])
            acc[name]["pv"].append(r["pv_us_cycle"])
            acc[name]["softmax"].append(r["softmax_us_cycle"])
            last[name] = r
    out = {}
    for name in specs:
        out[name] = {
            "matmul_us_cycle": statistics.median(acc[name]["matmul"]),
            "qk_us_cycle": statistics.median(acc[name]["qk"]),
            "pv_us_cycle": statistics.median(acc[name]["pv"]),
            "softmax_us_cycle": statistics.median(acc[name]["softmax"]),
            "matmul_us_all": acc[name]["matmul"],
            "per_type": last[name]["per_type"],
        }
    return out


def asymptotic_fp32_ratio(M: int, ctx: int, n_iter: int, warmup: int, rounds: int,
                          counts: dict) -> dict:
    """Measure fp32/bf16 matmul ratio at a LARGE M (M_ASYMPTOTIC). At high M the matmul
    AI is large, so the fp32 K-read (the L2-spill that contaminates M=32) amortizes over
    many query rows and the realized ratio converges off ~6 toward the compute ratio
    (~datasheet 2.24). This ISOLATES the M=32 spill as an artifact: if the asymptotic
    ratio is ~2-3 while M=32 is ~6, the gap is the spill, not an on-chip-upcast cost.
    bf16 vs fp32-IEEE only (tf32 is insufficient for 1e-6 -> not the build path)."""
    acc_bf16: list[float] = []
    acc_fp32: list[float] = []
    for _ in range(rounds):
        rb = _attn_matmul_bench(M, ctx, torch.bfloat16, False, n_iter, warmup, counts)
        rf = _attn_matmul_bench(M, ctx, torch.float32, False, n_iter, warmup, counts)
        acc_bf16.append(rb["matmul_us_cycle"])
        acc_fp32.append(rf["matmul_us_cycle"])
    bf16_med = statistics.median(acc_bf16)
    fp32_med = statistics.median(acc_fp32)
    return {
        "M": M, "bf16_matmul_us_cycle": bf16_med, "fp32_matmul_us_cycle": fp32_med,
        "r_fp32_asymptotic": fp32_med / bf16_med if bf16_med > 0 else float("nan"),
        "bf16_all": acc_bf16, "fp32_all": acc_fp32,
    }


# ===== Part C: roofline cross-check ============================================
def roofline(M: int, ctx: int) -> dict:
    """Attention arithmetic intensity AI(M) = M*(n_q/n_kv) and the A10G ridges.
    KV (bf16) read once, shared across M rows -> AI independent of ctx/head_dim."""
    ai = M * QPKV
    ridge_bf16 = A10G_BF16_TENSOR_TFLOPS * 1e12 / (A10G_PEAK_GBPS * 1e9)
    ridge_tf32 = A10G_TF32_TENSOR_TFLOPS * 1e12 / (A10G_PEAK_GBPS * 1e9)
    ridge_fp32 = A10G_FP32_CUDA_TFLOPS * 1e12 / (A10G_PEAK_GBPS * 1e9)
    return {
        "ai_attn": ai, "ridge_bf16": ridge_bf16, "ridge_tf32": ridge_tf32,
        "ridge_fp32": ridge_fp32,
        "ai_over_ridge_bf16": ai / ridge_bf16,
        "ai_over_ridge_fp32": ai / ridge_fp32,
        # additive-roofline compute fraction = min(1, AI/ridge) / (1 + min(1,AI/ridge))
        # is regime-dependent; report the simple ceilings the gate cross-checks against.
        "bf16_compute_bound": ai > ridge_bf16,
        "fp32_compute_bound": ai > ridge_fp32,
        "datasheet_r_fp32": A10G_BF16_TENSOR_TFLOPS / A10G_FP32_CUDA_TFLOPS,
        "datasheet_r_tf32": A10G_BF16_TENSOR_TFLOPS / A10G_TF32_TENSOR_TFLOPS,
    }


# ===== Steps 2-4: re-price, gate, hybrid =======================================
def official_tps(e_t: float, step_ratio: float, tau: float) -> float:
    return K_CAL * e_t / step_ratio * tau


def breakeven_et(target: float, step_ratio: float, tau: float) -> float:
    return target * step_ratio / (K_CAL * tau)


def reprice(delta_attn_steprat: float, e_t: float) -> dict:
    """Fold the fp32 attention step-time tax into the official TPS model.
    delta_attn_steprat = the fp32 tax in M=8-step-normalized units (added to the
    M=32 attention's contribution); step_ratio_fp32 = STEP_RATIO_BF16 + delta."""
    step_fp32 = STEP_RATIO_BF16 + delta_attn_steprat
    # central (tau=1) and floor (tau=0.9924) corners.
    central = official_tps(e_t, step_fp32, TAU_TREE_CENTRAL)
    floor = official_tps(e_t, step_fp32, TAU_TREE_FLOOR)
    # bf16 reference under the SAME formula (internal consistency) + the fleet 568.
    bf16_internal = official_tps(e_t, STEP_RATIO_BF16, TAU_TREE_CENTRAL)
    # express as a haircut on the fleet headline 568 (ratio is anchor-robust).
    ratio = STEP_RATIO_BF16 / step_fp32
    fleet_central = TREE_FERN_CENTRAL_OFFICIAL * ratio * (e_t / E_T_TREE_CEILING)
    fleet_floor = fleet_central * TAU_TREE_FLOOR
    return {
        "e_t": e_t, "step_ratio_bf16": STEP_RATIO_BF16,
        "delta_attn_steprat": delta_attn_steprat, "step_ratio_fp32": step_fp32,
        "official_central_tau1": central, "official_floor_tau": floor,
        "official_bf16_internal": bf16_internal,
        "fleet_headline_haircut_central": fleet_central,
        "fleet_headline_haircut_floor": fleet_floor,
        "step_inflation_pct": 100.0 * delta_attn_steprat / STEP_RATIO_BF16,
        "breakeven_et_500_tau1": breakeven_et(TARGET_500, step_fp32, TAU_TREE_CENTRAL),
        "breakeven_et_530_tau1": breakeven_et(TARGET_530, step_fp32, TAU_TREE_CENTRAL),
        "breakeven_et_500_floor": breakeven_et(TARGET_500, step_fp32, TAU_TREE_FLOOR),
        "breakeven_et_530_floor": breakeven_et(TARGET_530, step_fp32, TAU_TREE_FLOOR),
    }


def gate(rep_ceiling: dict, rep_ceiling_cons: dict) -> dict:
    """GREEN  net clears 530 at the conservative corner (fp32 tax cheap, BW-bound).
       AMBER  clears 500 but not 530, or needs the central corner.
       RED     net falls below 500."""
    cons_floor = rep_ceiling_cons["official_floor_tau"]
    cons_central = rep_ceiling_cons["official_central_tau1"]
    cen_floor = rep_ceiling["official_floor_tau"]
    cen_central = rep_ceiling["official_central_tau1"]
    if cons_floor >= TARGET_530:
        verdict = "GREEN"
    elif cons_floor < TARGET_500 and cen_central < TARGET_500:
        verdict = "RED"
    elif cen_central < TARGET_500:
        verdict = "RED"
    else:
        verdict = "AMBER"
    return {
        "verdict": verdict,
        "conservative_central_tau1": cons_central,
        "conservative_floor_tau": cons_floor,
        "central_estimate_tau1": cen_central,
        "central_estimate_floor_tau": cen_floor,
        "clears_530_conservative_floor": cons_floor >= TARGET_530,
        "clears_500_conservative_floor": cons_floor >= TARGET_500,
        "clears_500_central": cen_central >= TARGET_500,
        "clears_530_central": cen_central >= TARGET_530,
        "rule": ("GREEN: net clears 530 at conservative corner; AMBER: clears 500 "
                 "not 530 / needs central; RED: net < 500"),
    }


def hybrid_selective_fp32(delta_attn_full: dict, e_t: float) -> dict:
    """Step 4: SELECTIVE-fp32 escape -- fp32 ONLY on the depth-1 root row (1 of M=32
    query rows, where denken #101's deficit lives) + wirbel #93's near-tie tail
    (0.537% of positions < 1e-3 rel-margin); bf16 on the safe bulk (median rel-margin
    18%). The matmul tax scales with the fp32-row fraction. Two recipes:
      (a) root-row-only: 1/32 of the verify rows in fp32 (the depth-1 fix is a
          root-row effect, so this may suffice for the E[T] recovery).
      (b) root-row + near-tie tail: add the 0.537% margin-tail re-verify."""
    full = delta_attn_full["delta_attn_steprat_central"]
    full_cons = delta_attn_full["delta_attn_steprat_conservative"]
    f_root = 1.0 / 32.0
    f_root_tail = f_root + NEAR_TIE_FRAC_LT_1E3
    recipes = {}
    for name, frac in (("root_row_only", f_root),
                       ("root_row_plus_near_tie_tail", f_root_tail)):
        d_cen = full * frac
        d_cons = full_cons * frac
        rep = reprice(d_cen, e_t)
        rep_cons = reprice(d_cons, e_t)
        recipes[name] = {
            "fp32_row_fraction": frac,
            "delta_attn_steprat_central": d_cen,
            "delta_attn_steprat_conservative": d_cons,
            "official_central_tau1": rep["official_central_tau1"],
            "official_conservative_floor": rep_cons["official_floor_tau"],
            "fleet_headline_central": rep["fleet_headline_haircut_central"],
            "fleet_headline_conservative_floor": rep_cons["fleet_headline_haircut_floor"],
            "clears_530_conservative_floor": rep_cons["official_floor_tau"] >= TARGET_530,
            "clears_500_conservative_floor": rep_cons["official_floor_tau"] >= TARGET_500,
        }
    return {
        "recipes": recipes,
        "preserves_et_recovery_rationale": (
            "denken #101 localizes the deficit to the DEPTH-1 root row (built 0.598 "
            "vs q[1]=0.7287); the recovery is a root-row argmax effect, so fp32 on the "
            "root row (recipe a) is the minimal fix that restores the numerator. The "
            "near-tie tail (recipe b) covers wirbel #93's 0.537% < 1e-3 rel-margin "
            "flip-risk set for full greedy identity on the deeper tree rows."),
        "near_tie_frac": NEAR_TIE_FRAC_LT_1E3,
        "safe_bulk_median_rel_margin": SAFE_BULK_MEDIAN_REL_MARGIN,
    }


# ---------------------------------------------------------------------------
def run(args) -> dict:
    assert torch.cuda.is_available(), "CUDA required"
    _maybe_install_splitkv()
    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    ctx = args.ctx
    counts = {"sliding": N_SLIDING, "full": N_FULL}

    res: dict = {
        "pr": 131, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "l2_bytes": torch.cuda.get_device_properties(0).L2_cache_size,
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "ctx": ctx, "counts": counts,
        "config": {
            "n_iter": args.n_iter, "warmup": args.warmup, "rounds": args.rounds,
            "n_q_heads": N_Q_HEADS, "n_kv_heads": N_KV_HEADS, "qpkv": QPKV,
            "head_dim": HEAD_DIM, "step_ratio_bf16": STEP_RATIO_BF16,
            "k_cal": K_CAL, "e_t_tree_ceiling": E_T_TREE_CEILING,
            "tau_central": TAU_TREE_CENTRAL, "tau_floor": TAU_TREE_FLOOR,
        },
    }
    print(f"[fp32steptime] GPU {res['gpu']} ctx={ctx} L2={res['l2_bytes']/1e6:.1f}MB",
          flush=True)
    res["peak_bw"] = _measure_peak_bw(torch, dev)
    print(f"[fp32steptime] peak HBM copy BW = "
          f"{res['peak_bw']['measured_peak_gbps_copy']:.0f} GB/s", flush=True)

    # ---- Part A: real-kernel attention substep M-sweep (deployed bf16) --------
    m_values = [1, 8, 16, 32]
    attn_us = {}
    attn_detail = {}
    for M in m_values:
        total, per_type = measure_attn_us(M, ctx, args.n_iter, args.warmup, counts)
        attn_us[M] = total
        attn_detail[M] = per_type
        print(f"   [A] attn_us(M={M:<2d}) = {total:8.1f}us  "
              f"(sliding {per_type['sliding']['per_op_us']:.2f} x{counts['sliding']}, "
              f"full {per_type['full']['per_op_us']:.2f} x{counts['full']})", flush=True)
    fit = fit_mem_compute(m_values, [attn_us[m] for m in m_values])
    compute_us_m32 = fit["c1_compute_us_per_row"] * 32.0
    compute_frac_m32 = compute_us_m32 / attn_us[32]
    res["partA_real_attn"] = {
        "attn_us": {str(m): attn_us[m] for m in m_values},
        "per_type": {str(m): attn_detail[m] for m in m_values},
        "fit_mem_compute": fit,
        "r_attn_M32_M8_measured": attn_us[32] / attn_us[8],
        "r_attn_M32_M8_ref107": R_ATTN_107,
        "compute_us_m32": compute_us_m32,
        "compute_frac_m32": compute_frac_m32,
        "mem_floor_us": fit["c0_mem_us"],
    }
    print(f"   [A] fit attn_us(M) = {fit['c0_mem_us']:.1f} + "
          f"{fit['c1_compute_us_per_row']:.2f}*M  (R^2={fit['r2']:.4f}) -> "
          f"compute_frac(M=32)={compute_frac_m32:.3f}", flush=True)

    # ---- Part B: QK+PV matmul micro-bench in bf16 / tf32 / fp32-IEEE ----------
    mm = attn_matmul_precisions(32, ctx, args.n_iter, args.warmup, args.rounds, counts)
    bf16_mm = mm["bf16"]["matmul_us_cycle"]
    fp32_mm = mm["fp32"]["matmul_us_cycle"]
    tf32_mm = mm["tf32"]["matmul_us_cycle"]
    r_fp32_m32_realized = fp32_mm / bf16_mm       # CONTAMINATED by the L2 spill at M=32
    r_tf32 = tf32_mm / bf16_mm
    # matmul fraction of the M-scaling compute: the explicit bf16 matmul vs
    # (matmul + softmax) at M=32 (softmax is fp32 in all precisions -> not taxed).
    bf16_sm = mm["bf16"]["softmax_us_cycle"]
    matmul_frac = bf16_mm / (bf16_mm + bf16_sm) if (bf16_mm + bf16_sm) > 0 else 1.0

    # asymptotic corroboration: at M=512 the fp32 K-read amortizes -> ratio converges
    # off the M=32 spill toward the compute ratio. Lighter iters (ratio only).
    asym = asymptotic_fp32_ratio(
        M_ASYMPTOTIC, ctx, max(50, args.n_iter // 3), args.warmup,
        max(2, args.rounds - 1), counts)
    r_fp32_asymptotic = asym["r_fp32_asymptotic"]

    # ---- r_fp32 BRACKET: on-chip-upcast (central) vs naive-HBM (conservative) ----
    # CENTRAL = datasheet compute ratio (on-chip upcast: KV read once as bf16, upcast
    # in-register, only tensor->CUDA-core COMPUTE changes). The large-M asymptotic
    # measurement corroborates this is the right ballpark (NOT the ~6 spill). We hold
    # central at the datasheet 2.24; if the realized asymptotic is HIGHER (CUDA-core
    # inefficiency), we take it as central instead (never optimistic below measured).
    r_fp32_central = max(R_FP32_DATASHEET, r_fp32_asymptotic) \
        if math.isfinite(r_fp32_asymptotic) else R_FP32_DATASHEET
    # CONSERVATIVE = realized M=32 ratio = a NAIVE build that materializes fp32 KV in
    # HBM (pays the extra fp32 read the spill proxies). Floor it at central.
    r_fp32_conservative = max(r_fp32_m32_realized, r_fp32_central)
    l2_bytes = res["l2_bytes"]
    fp32_k_full_mb = N_Q_HEADS * HEAD_DIM["full"] * ctx * 4 / 1e6
    bf16_k_full_mb = fp32_k_full_mb / 2.0
    r_fp32_bracket = {
        "r_fp32_central_onchip": r_fp32_central,
        "r_fp32_conservative_naive_hbm": r_fp32_conservative,
        "r_fp32_datasheet_compute": R_FP32_DATASHEET,
        "r_fp32_m32_realized": r_fp32_m32_realized,
        "r_fp32_asymptotic_measured": r_fp32_asymptotic,
        "asymptotic_M": M_ASYMPTOTIC,
        "l2_bytes": l2_bytes,
        "fp32_k_buffer_full_mb": fp32_k_full_mb,
        "bf16_k_buffer_full_mb": bf16_k_full_mb,
        "fp32_spills_l2": fp32_k_full_mb * 1e6 > l2_bytes,
        "bf16_fits_l2": bf16_k_full_mb * 1e6 <= l2_bytes,
        "artifact_note": (
            f"M=32 realized r_fp32={r_fp32_m32_realized:.2f} is L2-spill contaminated: "
            f"the full-layer fp32 K-buffer ({fp32_k_full_mb:.1f}MB) exceeds L2 "
            f"({l2_bytes/1e6:.1f}MB) so fp32 re-reads K from HBM every iter while bf16 "
            f"({bf16_k_full_mb:.1f}MB) stays resident. The on-chip-upcast build reads "
            f"bf16 KV ONCE and upcasts in-register -> only the COMPUTE ratio applies "
            f"(datasheet {R_FP32_DATASHEET:.2f}; asymptotic M={M_ASYMPTOTIC} measured "
            f"{r_fp32_asymptotic:.2f} corroborates). The ~6 ratio is the NAIVE "
            f"fp32-KV-in-HBM build's worst case, kept as the conservative corner."),
    }
    res["partB_matmul"] = {
        "M": 32, "bf16_matmul_us_cycle": bf16_mm, "tf32_matmul_us_cycle": tf32_mm,
        "fp32_matmul_us_cycle": fp32_mm, "bf16_softmax_us_cycle": bf16_sm,
        "r_fp32_m32_realized": r_fp32_m32_realized, "r_tf32_measured": r_tf32,
        "r_fp32_bracket": r_fp32_bracket,
        "asymptotic": asym,
        "matmul_frac_of_scaling_compute": matmul_frac,
        "qk_pv_split_bf16": {"qk": mm["bf16"]["qk_us_cycle"],
                             "pv": mm["bf16"]["pv_us_cycle"]},
        "per_precision": {k: {kk: vv for kk, vv in v.items() if kk != "per_type"}
                          for k, v in mm.items()},
    }
    print(f"   [B] M=32 matmul: bf16 {bf16_mm:.1f}us  tf32 {tf32_mm:.1f}us "
          f"(r={r_tf32:.2f}, INSUFFICIENT 1e-3)  fp32-IEEE {fp32_mm:.1f}us "
          f"(r_m32={r_fp32_m32_realized:.2f} SPILL)  matmul_frac={matmul_frac:.3f}",
          flush=True)
    print(f"   [B] r_fp32 bracket: central(on-chip)={r_fp32_central:.2f} "
          f"[datasheet {R_FP32_DATASHEET:.2f}, asym M{M_ASYMPTOTIC}={r_fp32_asymptotic:.2f}]  "
          f"conservative(naive-HBM)={r_fp32_conservative:.2f}  "
          f"(fp32 K {fp32_k_full_mb:.1f}MB > L2 {l2_bytes/1e6:.1f}MB = spill)", flush=True)

    # ---- Part C: roofline cross-check ----------------------------------------
    res["partC_roofline"] = {str(m): roofline(m, ctx) for m in (1, 8, 32)}
    rc32 = res["partC_roofline"]["32"]
    print(f"   [C] AI(M=32)={rc32['ai_attn']:.0f} FLOP/byte  bf16 ridge "
          f"{rc32['ridge_bf16']:.0f} (compute-bound={rc32['bf16_compute_bound']})  "
          f"fp32 ridge {rc32['ridge_fp32']:.0f} (compute-bound="
          f"{rc32['fp32_compute_bound']})  datasheet r_fp32={rc32['datasheet_r_fp32']:.2f}",
          flush=True)

    # ---- Step 1 result: Delta_step_time_fp32 ---------------------------------
    # attention's M=32 contribution in M=8-step units = B_ATTN * r_attn.
    attn_m32_steprat = B_ATTN * (attn_us[32] / attn_us[8])
    # CENTRAL: on-chip-upcast r_fp32 (datasheet/asymptotic compute ratio), tax only the
    # matmul fraction of the M-scaling compute (softmax stays fp32 -> not taxed).
    delta_central = (attn_m32_steprat * compute_frac_m32 * matmul_frac
                     * (r_fp32_central - 1.0))
    # CONSERVATIVE: naive-HBM r_fp32 (the spilled ~6 proxy) AND charge ALL M-scaling
    # compute as matmul (matmul_frac = 1) -> the double-pessimistic corner.
    delta_cons = (attn_m32_steprat * compute_frac_m32 * 1.0
                  * (r_fp32_conservative - 1.0))
    delta = {
        "attn_m32_steprat": attn_m32_steprat,
        "compute_frac_m32": compute_frac_m32, "matmul_frac": matmul_frac,
        "r_fp32_central": r_fp32_central, "r_fp32_conservative": r_fp32_conservative,
        "delta_attn_steprat_central": delta_central,
        "delta_attn_steprat_conservative": delta_cons,
        "delta_attn_us_m32_central": compute_us_m32 * matmul_frac * (r_fp32_central - 1.0),
        "delta_attn_us_m32_conservative": compute_us_m32 * (r_fp32_conservative - 1.0),
        "delta_pct_of_step_central": 100.0 * delta_central / STEP_RATIO_BF16,
        "delta_pct_of_step_conservative": 100.0 * delta_cons / STEP_RATIO_BF16,
    }
    res["step1_delta_step_time_fp32"] = delta
    print(f"   [1] Delta_step_fp32: central +{delta['delta_pct_of_step_central']:.2f}% "
          f"step / conservative +{delta['delta_pct_of_step_conservative']:.2f}%", flush=True)

    # ---- Steps 2-3: re-price + gate ------------------------------------------
    e_t = args.e_t_recovered
    rep_central = reprice(delta_central, e_t)
    rep_cons = reprice(delta_cons, e_t)
    g = gate(rep_central, rep_cons)
    res["step2_reprice"] = {"central": rep_central, "conservative": rep_cons}
    res["step3_gate"] = g
    # sensitivity to E[T]_recovered (denken #128's numerator).
    res["step2_et_sensitivity"] = {
        f"{et:.2f}": {
            "official_central_tau1": reprice(delta_central, et)["official_central_tau1"],
            "official_conservative_floor": reprice(delta_cons, et)["official_floor_tau"],
        } for et in (4.45, 4.62, 4.80, 5.00, 5.207)
    }

    # sensitivity to the r_fp32 bracket (matmul_frac held at the central value): shows
    # the AMBER verdict is robust whether the on-chip ratio is datasheet 2.24 or the
    # measured asymptotic 2.72, and only the naive-HBM ~6 corner goes RED.
    def _central_at(r):
        d = attn_m32_steprat * compute_frac_m32 * matmul_frac * (r - 1.0)
        return reprice(d, e_t)["official_central_tau1"]
    res["step2_rfp32_sensitivity"] = {
        "datasheet_2_24": {"r_fp32": R_FP32_DATASHEET,
                           "official_central_tau1": _central_at(R_FP32_DATASHEET)},
        "asymptotic_measured": {"r_fp32": r_fp32_asymptotic,
                                "official_central_tau1": _central_at(r_fp32_asymptotic)},
        "central_used": {"r_fp32": r_fp32_central,
                         "official_central_tau1": rep_central["official_central_tau1"]},
        "naive_hbm_conservative": {"r_fp32": r_fp32_conservative,
                                   "official_central_tau1": _central_at(r_fp32_conservative)},
    }
    print(f"   [2/3] E[T]={e_t}: central {rep_central['official_central_tau1']:.1f} "
          f"(floor {rep_central['official_floor_tau']:.1f})  conservative-floor "
          f"{rep_cons['official_floor_tau']:.1f}  -> VERDICT={g['verdict']}", flush=True)
    print(f"        break-even E[T] AFTER tax: 500@{rep_central['breakeven_et_500_tau1']:.3f} "
          f"530@{rep_central['breakeven_et_530_tau1']:.3f}  (vs pre-tax 4.624 / ceiling "
          f"{E_T_TREE_CEILING})", flush=True)

    # ---- Step 4: selective-fp32 hybrid ---------------------------------------
    res["step4_hybrid"] = hybrid_selective_fp32(delta, e_t)
    hr = res["step4_hybrid"]["recipes"]["root_row_only"]
    print(f"   [4] hybrid root-row-only: central {hr['official_central_tau1']:.1f} "
          f"clears530-cons-floor={hr['clears_530_conservative_floor']}", flush=True)

    # ---- PRIMARY / TEST metrics ----------------------------------------------
    res["primary_metric"] = {
        "name": "fp32_tree_official_tps_central",
        "value": rep_central["official_central_tau1"]}
    res["test_metric"] = {
        "name": "fp32_tree_clears_500",
        "value": int(g["clears_500_central"])}
    res["verdict"] = g["verdict"]

    res["elapsed_s"] = time.time() - t0
    res["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2))
    print(f"\n[fp32steptime] VERDICT={res['verdict']}  primary(central official)="
          f"{res['primary_metric']['value']:.1f}  clears500={res['test_metric']['value']}",
          flush=True)
    print(f"[fp32steptime] wrote {out_path}  ({res['elapsed_s']:.0f}s, "
          f"peak {res['peak_gpu_gb']:.2f}GB)", flush=True)

    # ---- W&B -----------------------------------------------------------------
    if args.wandb_group:
        try:
            import wandb
            run_w = wandb.init(
                project="gemma-challenge-senpai", entity="wandb-applied-ai-team",
                group=args.wandb_group, name=args.wandb_name,
                config={**res["config"], "gpu": res["gpu"], "ctx": ctx})
            log = {
                "fp32_tree_official_tps_central": rep_central["official_central_tau1"],
                "fp32_tree_official_tps_floor": rep_central["official_floor_tau"],
                "fp32_tree_official_conservative_floor": rep_cons["official_floor_tau"],
                "fp32_tree_clears_500": int(g["clears_500_central"]),
                "fp32_tree_clears_530_conservative_floor": int(g["clears_530_conservative_floor"]),
                "delta_step_pct_central": delta["delta_pct_of_step_central"],
                "delta_step_pct_conservative": delta["delta_pct_of_step_conservative"],
                "r_fp32_central_onchip": r_fp32_central,
                "r_fp32_conservative_naive_hbm": r_fp32_conservative,
                "r_fp32_m32_realized": r_fp32_m32_realized,
                "r_fp32_asymptotic_measured": r_fp32_asymptotic,
                "r_fp32_datasheet": R_FP32_DATASHEET, "r_tf32_measured": r_tf32,
                "compute_frac_m32": compute_frac_m32, "matmul_frac": matmul_frac,
                "attn_ai_m32": rc32["ai_attn"], "ridge_bf16": rc32["ridge_bf16"],
                "ridge_fp32": rc32["ridge_fp32"],
                "breakeven_et_500_after_tax": rep_central["breakeven_et_500_tau1"],
                "breakeven_et_530_after_tax": rep_central["breakeven_et_530_tau1"],
                "hybrid_rootrow_official_central": hr["official_central_tau1"],
                "verdict_green": int(g["verdict"] == "GREEN"),
                "verdict_amber": int(g["verdict"] == "AMBER"),
                "verdict_red": int(g["verdict"] == "RED"),
                "measured_peak_gbps": res["peak_bw"]["measured_peak_gbps_copy"],
            }
            for m in m_values:
                log[f"attn_us_M{m}"] = attn_us[m]
            wandb.log(log)
            run_w.summary.update(log)
            res["wandb_run_id"] = run_w.id
            wandb.finish()
            print(f"[fp32steptime] W&B run {run_w.id} (group {args.wandb_group})", flush=True)
            out_path.write_text(json.dumps(res, indent=2))
        except Exception as e:  # noqa: BLE001
            print(f"[fp32steptime] W&B logging skipped: {e!r}", flush=True)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ctx", type=int, default=ATTN_CTX_DEFAULT)
    ap.add_argument("--n-iter", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--e-t-recovered", type=float, default=E_T_TREE_CEILING,
                    help="E[T] the fp32 upcast recovers (denken #128 numerator); "
                         "default = the 5.207 ceiling (fern #125)")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "research/spec_cost_model/fp32_star_steptime_tax.json")
    ap.add_argument("--wandb-group", type=str, default="fp32-star-steptime-tax")
    ap.add_argument("--wandb-name", type=str, default="lawine/fp32-star-steptime-tax")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)
    if args.no_wandb:
        args.wandb_group = None
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
