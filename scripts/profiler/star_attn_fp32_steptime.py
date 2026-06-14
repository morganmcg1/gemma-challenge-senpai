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


# ===========================================================================
# PR #136: MEASURED STEP-ANCHOR for the depth-9 verify step (the DENOMINATOR)
# ===========================================================================
# fern #129's whole-fleet go/no-go denominator is the depth-9 W* step = 1.2127
# (my #125 roofline: gemm_mult 1.098 + drafter_add 0.048 + attn_add 0.0666, ALL
# device-time). But the live oracle run `tree-488-pw-fp32-v0` (openevolve, board
# 20260614-100550-487) ran EAGER star-attn with attn_py_calls/step=37 -- a KNOWN
# flagged build blocker ("eager-dispatch overhead, graph path needed", byteshark
# board ~05:53Z). Eager dispatch of 37 attention kernels/step exposes GPU-idle
# launch gaps that the GRAPHED M=8 linear baseline (step=1.0; decode 99.41%
# GPU-bound, ~0 launch headroom, #65) does NOT pay. So the realized EAGER step
# >= the 1.2127 fused roofline. This section MEASURES that gap locally on the
# deployed unified_attention kernel (same path as Part A) and brackets
# measured_depth9_step_time, then re-prices the #131 root-row recipe + fern's
# operative clear-500/clear-530 bars at the MEASURED (not roofline) step.
#
# NOTE (chiku-inu board 20260614-104247-994): the fp32 build is a NUMERATOR
# failure -- E[T] stuck ~2.07-2.62 (BUG-1 depth-1 deficit is STRUCTURAL, not
# bf16 precision; denken #133/kanna #134 own that). The step-anchor here is
# still load-bearing: it pins the DENOMINATOR every 500-verdict divides by, for
# ANY future build that fixes the numerator. The step is no longer the binding
# lever, but it IS the denominator. (lawine owns the denominator; not the E[T].)

# fern #125/#129 depth-9 composition constants (the denominator we anchor).
STEP_WSTAR_DEPTH9 = 1.2127483746822987      # roofline depth-9 W* step (M=8-norm)
STEP_WSTAR_GEMM = 1.098148338441328         # Marlin staircase M=32 (denken #68)
STEP_WSTAR_DRAFTER_ADD = 0.048              # drafter expansion depth-9 (wirbel)
STEP_WSTAR_ATTN_ADD = 0.06660003624097069   # bf16 tree-mask attn tax (#107 1.83x)
CLEAR500_BAR_ROOFLINE = 4.840617149792076   # fern #129 operative clear-500 bar
TAU_FERN_CENTRAL = 1.0
TAU_FERN_LOW = 0.9983                        # fern #129 tau band low (lawine #116)
# oracle-measured E[T] numerators (NOT my lane -- only for the realized cross).
E_T_OPENEVOLVE_ORACLE = 2.621               # openevolve A10G readout (board 100550)
E_T_CHIKU_RUN = 2.07                        # chiku-inu's own run (board 104247)
# M=8 linear-baseline step wall time: K_cal == steps/sec at step=1.0, so 1 step
# unit == 1/K_cal seconds. Converts a measured us idle into normalized step units.
STEP_M8_US = 1.0e6 / K_CAL                   # ~7982.86 us
PR131_TAX_JSON = ROOT / "research/spec_cost_model/fp32_star_steptime_tax.json"
TARGET_530_F = 530.0
# filler bf16 NxN GEMM ~ per-layer non-attention GPU work, used to give the GPU
# something to overlap the eager attention-launch CPU dispatch with (Part D2).
GEMM_FILLER_N = 2048


def _load_pr131_fp32_tax() -> dict:
    """The fp32 QK+PV upcast device-compute tax from my merged #131, in M=8-step
    units -- the SAME normalization as fern's 1.2127 (both are fractions of the
    M=8 linear step). The tax is a kernel-COMPUTE property, invariant to eager-vs-
    graphed dispatch, so we reuse it verbatim and add it onto the MEASURED step."""
    d = json.loads(PR131_TAX_JSON.read_text())
    delta = d["step1_delta_step_time_fp32"]
    return {
        "delta_full_central": delta["delta_attn_steprat_central"],
        "delta_full_conservative": delta["delta_attn_steprat_conservative"],
        "r_fp32_central": delta["r_fp32_central"],
        "r_fp32_conservative": delta["r_fp32_conservative"],
        "source_utc": d.get("utc"), "source_wandb": d.get("wandb_run_id"),
    }


def measure_eager_dispatch_overhead(M, ctx, n_passes, warmup, counts, n_iter):
    """Part D: the eager-dispatch GPU-idle gap for ONE depth-9 verify step's worth
    of attention = counts['sliding']+counts['full'] = 37 calls (the oracle's
    attn_py_calls/step). Three timings of the SAME 37-call sequence on the deployed
    split-KV unified_attention (cold-KV rotation defeats L2, as Part A):
      device_busy_us  -- profiler self_device_time (NO gaps; GPU saturated across
                         reps) = the GRAPHED/fused floor.
      eager_steady_us -- CUDA-event span over N back-to-back steps / N (CPU
                         pipelines launches across steps; realistic continuous-
                         decode lower bound on the idle).
      eager_cold_us   -- CUDA-event span over ONE step, sync before each (cold
                         launch queue per step; pessimistic per-step upper bound).
    exposed_idle = max(0, eager - device_busy). This captures ONLY the 37
    attention launches' idle; the full eager step ALSO pays the tree's Python
    control flow (drafter + salvage walk) which we do NOT measure -> this idle is
    a LOWER bound on the total eager step inflation. openevolve's wall_tps would
    capture the full step; we request it on the board."""
    dev = torch.device("cuda")
    inp_s = _build_op_inputs(torch, "sliding", M, ctx)
    inp_f = _build_op_inputs(torch, "full", M, ctx)
    out_s = torch.empty(M, N_Q_HEADS, inp_s["hd"], dtype=torch.bfloat16, device=dev)
    out_f = torch.empty(M, N_Q_HEADS, inp_f["hd"], dtype=torch.bfloat16, device=dev)
    call_s, _ = _make_call(torch, inp_s, M, ctx, torch.float32, out_s)
    call_f, _ = _make_call(torch, inp_f, M, ctx, torch.float32, out_f)
    ns, nf = counts["sliding"], counts["full"]

    def one_step():
        for _ in range(ns):
            call_s()
        for _ in range(nf):
            call_f()

    # (1) device-busy floor (no gaps) -- profiler self-device-time of 37 calls.
    device_busy_us = _profiled_device_us(torch, one_step, n_iter, warmup)

    # (2) eager steady-state: events around N back-to-back steps (CPU pipelines).
    for _ in range(warmup):
        one_step()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(n_passes):
        one_step()
    ev1.record()
    torch.cuda.synchronize()
    eager_steady_us = ev0.elapsed_time(ev1) * 1e3 / n_passes

    # (3) eager cold: sync before each single step (empty launch queue per step).
    cold = []
    for _ in range(max(3, warmup // 4)):
        one_step()
        torch.cuda.synchronize()
    for _ in range(n_passes):
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        one_step()
        e1.record()
        torch.cuda.synchronize()
        cold.append(e0.elapsed_time(e1) * 1e3)
    eager_cold_us = statistics.median(cold)

    idle_steady = max(0.0, eager_steady_us - device_busy_us)
    idle_cold = max(0.0, eager_cold_us - device_busy_us)
    n_calls = ns + nf
    del inp_s, inp_f, out_s, out_f
    torch.cuda.empty_cache()
    return {
        "M": M, "n_attn_calls_per_step": n_calls,
        "device_busy_us": device_busy_us,
        "eager_steady_us": eager_steady_us, "eager_cold_us": eager_cold_us,
        "exposed_idle_steady_us": idle_steady, "exposed_idle_cold_us": idle_cold,
        "per_call_idle_steady_us": idle_steady / n_calls,
        "per_call_idle_cold_us": idle_cold / n_calls,
        "cold_samples": cold,
        "method": ("device_busy = profiler self-time (graphed floor); eager_* = "
                   "CUDA-event GPU-timeline span incl. launch idle; idle = eager - "
                   "busy (attention-launch component only, LOWER bound on full step)"),
    }


def measure_overlap_hidden_idle(M, ctx, n_passes, warmup, counts, n_iter, gemm_n):
    """Part D2: the eager attention-launch idle that SURVIVES realistic GEMM overlap
    -- the credible eager penalty. Part D times 37 back-to-back attention launches
    with NO other GPU work, so the Triton CPU launch starves the GPU and OVER-states
    the exposed idle (a no-overlap UPPER bound). The real depth-9 step interleaves
    each layer's attention with that layer's GEMM GPU work (gemm_mult + drafter ~=
    1.146 step-units ~= 9150us/step ~= 247us/layer). Issuing a filler bf16 NxN GEMM
    before each attention call gives the GPU work that hides the attention CPU
    dispatch; the idle that REMAINS is what an eager (un-graphed) verify step pays
    on the ATTENTION path. NOTE: the tree's drafter + salvage Python control flow is
    NOT modeled here -> attention-path penalty only; openevolve's full-step wall_tps
    still anchors the total. A graphed build (blocker #2) drives this idle to ~0."""
    dev = torch.device("cuda")
    inp_s = _build_op_inputs(torch, "sliding", M, ctx)
    inp_f = _build_op_inputs(torch, "full", M, ctx)
    out_s = torch.empty(M, N_Q_HEADS, inp_s["hd"], dtype=torch.bfloat16, device=dev)
    out_f = torch.empty(M, N_Q_HEADS, inp_f["hd"], dtype=torch.bfloat16, device=dev)
    call_s, _ = _make_call(torch, inp_s, M, ctx, torch.float32, out_s)
    call_f, _ = _make_call(torch, inp_f, M, ctx, torch.float32, out_f)
    ns, nf = counts["sliding"], counts["full"]
    n_calls = ns + nf
    a = torch.randn(gemm_n, gemm_n, dtype=torch.bfloat16, device=dev)
    b = torch.randn(gemm_n, gemm_n, dtype=torch.bfloat16, device=dev)
    c = torch.empty(gemm_n, gemm_n, dtype=torch.bfloat16, device=dev)

    # size the per-layer filler to ~= the step's non-attention GPU work / 37 layers.
    filler_us = _profiled_device_us(torch, lambda: torch.mm(a, b, out=c), n_iter, warmup)
    target_per_layer_us = ((STEP_WSTAR_GEMM + STEP_WSTAR_DRAFTER_ADD)
                           * STEP_M8_US / n_calls)
    gemm_per_call = max(1, round(target_per_layer_us / filler_us))

    def one_step():
        for _ in range(ns):
            for _ in range(gemm_per_call):
                torch.mm(a, b, out=c)
            call_s()
        for _ in range(nf):
            for _ in range(gemm_per_call):
                torch.mm(a, b, out=c)
            call_f()

    device_busy_us = _profiled_device_us(torch, one_step, n_iter, warmup)
    for _ in range(warmup):
        one_step()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(n_passes):
        one_step()
    ev1.record()
    torch.cuda.synchronize()
    eager_span_us = ev0.elapsed_time(ev1) * 1e3 / n_passes
    idle = max(0.0, eager_span_us - device_busy_us)
    gpu_nonattn_us = filler_us * gemm_per_call * n_calls
    del inp_s, inp_f, out_s, out_f, a, b, c
    torch.cuda.empty_cache()
    return {
        "M": M, "gemm_n": gemm_n, "gemm_per_call": gemm_per_call,
        "filler_us_each": filler_us,
        "target_per_layer_us": target_per_layer_us,
        "gpu_nonattn_us_per_step": gpu_nonattn_us,
        "device_busy_us": device_busy_us, "eager_span_us": eager_span_us,
        "exposed_idle_overlap_us": idle,
        "per_call_idle_overlap_us": idle / n_calls,
        "method": ("interleave a per-layer filler GEMM before each of the 37 "
                   "attention calls; idle = eager span - device-busy floor = the "
                   "attention-path eager penalty that survives realistic GEMM "
                   "overlap (drafter/salvage Python NOT modeled)"),
    }


def fern_official(e_t: float, step: float, tau: float) -> float:
    """fern #129 compose: official = K_cal * E[T] / step * tau."""
    return K_CAL * e_t / step * tau


def fern_clear_bar(target: float, step: float, tau: float) -> float:
    """E[T] needed to clear `target` official at (step, tau). Rises with step."""
    return target * step / (K_CAL * tau)


def _price_recipe_at_step(step_bf16: float, tax: float) -> dict:
    """Price one recipe (bf16 tax=0 / root-row tax=full/32 / full tax=full) at a
    given bf16 measured step. Reports its own step + the clear-500/530 E[T] bars
    (central + tau-low) and whether each bar sits under the 5.207 supply ceiling."""
    step = step_bf16 + tax
    out = {"recipe_step": step, "fp32_tax_step_units": tax}
    for tname, tau in (("central", TAU_FERN_CENTRAL), ("taulow", TAU_FERN_LOW)):
        bar500 = fern_clear_bar(TARGET_500, step, tau)
        bar530 = fern_clear_bar(TARGET_530_F, step, tau)
        out[tname] = {
            "tau": tau,
            "clear500_bar_et": bar500, "clear530_bar_et": bar530,
            "clears500_under_ceiling": bar500 <= E_T_TREE_CEILING,
            "clears530_under_ceiling": bar530 <= E_T_TREE_CEILING,
            "official_at_ceiling": fern_official(E_T_TREE_CEILING, step, tau),
        }
    return out


def run_measured_anchor(args) -> dict:
    """PR #136 Steps 1-4: measure the eager step, bracket it, re-price the root-row
    recipe + fern's operative bars at the MEASURED step, gate."""
    assert torch.cuda.is_available(), "CUDA required"
    _maybe_install_splitkv()
    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    ctx = args.ctx
    counts = {"sliding": N_SLIDING, "full": N_FULL}
    tax131 = _load_pr131_fp32_tax()

    res: dict = {
        "pr": 136, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "l2_bytes": torch.cuda.get_device_properties(0).L2_cache_size,
        "ctx": ctx, "counts": counts,
        "anchors": {
            "step_wstar_depth9_roofline": STEP_WSTAR_DEPTH9,
            "step_decomp": {"gemm_mult": STEP_WSTAR_GEMM,
                            "drafter_add": STEP_WSTAR_DRAFTER_ADD,
                            "attn_add_bf16_treemask": STEP_WSTAR_ATTN_ADD},
            "k_cal": K_CAL, "step_m8_us": STEP_M8_US,
            "clear500_bar_roofline": CLEAR500_BAR_ROOFLINE,
            "e_t_tree_ceiling": E_T_TREE_CEILING,
            "tau_central": TAU_FERN_CENTRAL, "tau_low": TAU_FERN_LOW,
            "pr131_fp32_tax": tax131,
            "oracle_e_t_openevolve": E_T_OPENEVOLVE_ORACLE,
            "oracle_e_t_chiku": E_T_CHIKU_RUN,
        },
        "config": {"n_iter": args.n_iter, "warmup": args.warmup,
                   "eager_passes": args.eager_passes,
                   "openevolve_wall_tps": args.openevolve_wall_tps,
                   "openevolve_e_t": args.openevolve_e_t},
    }
    print(f"[anchor] GPU {res['gpu']} ctx={ctx} step_M8={STEP_M8_US:.1f}us "
          f"roofline_step={STEP_WSTAR_DEPTH9:.4f}", flush=True)

    # ---- Part A: real attn fit (device-busy; cross-checks #131, used for the floor)
    m_values = [1, 8, 16, 32]
    attn_us = {}
    for M in m_values:
        total, _ = measure_attn_us(M, ctx, args.n_iter, args.warmup, counts)
        attn_us[M] = total
    fit = fit_mem_compute(m_values, [attn_us[m] for m in m_values])
    res["partA_attn_fit"] = {
        "attn_us": {str(m): attn_us[m] for m in m_values}, "fit": fit,
        "fit_str": f"attn_us(M) = {fit['c0_mem_us']:.1f} + "
                   f"{fit['c1_compute_us_per_row']:.2f}*M",
        "r_attn_M32_M8": attn_us[32] / attn_us[8]}
    print(f"   [A] {res['partA_attn_fit']['fit_str']} (R^2={fit['r2']:.3f}) "
          f"r_attn(32/8)={attn_us[32]/attn_us[8]:.3f}", flush=True)

    # ---- Part D: eager dispatch overhead (the NEW measurement) ----------------
    eager = measure_eager_dispatch_overhead(
        32, ctx, args.eager_passes, args.warmup, counts, args.n_iter)
    res["partD_eager_overhead"] = eager
    print(f"   [D] 37-call step: device_busy={eager['device_busy_us']:.1f}us  "
          f"eager_steady={eager['eager_steady_us']:.1f}us  "
          f"eager_cold={eager['eager_cold_us']:.1f}us  "
          f"idle_steady={eager['exposed_idle_steady_us']:.1f}us "
          f"({eager['per_call_idle_steady_us']:.2f}/call)  "
          f"idle_cold={eager['exposed_idle_cold_us']:.1f}us "
          f"({eager['per_call_idle_cold_us']:.2f}/call)", flush=True)

    # ---- Part D2: idle that SURVIVES realistic GEMM overlap (the credible one) -
    overlap = measure_overlap_hidden_idle(
        32, ctx, args.eager_passes, args.warmup, counts, args.n_iter,
        args.gemm_filler_n)
    res["partD2_overlap_hidden"] = overlap
    print(f"   [D2] interleaved (filler {overlap['gemm_n']}^3 x{overlap['gemm_per_call']}"
          f"/call, {overlap['filler_us_each']:.0f}us each): "
          f"device_busy={overlap['device_busy_us']:.1f}us  "
          f"eager_span={overlap['eager_span_us']:.1f}us  "
          f"idle_overlap={overlap['exposed_idle_overlap_us']:.1f}us "
          f"({overlap['per_call_idle_overlap_us']:.2f}/call)", flush=True)

    # ---- Step 1: bracket the measured depth-9 step ----------------------------
    # Three regimes (NOT steady-vs-cold; both of those are the no-overlap pessimist):
    #   graphed floor  = roofline (graph path built -> attn launch idle fully hidden)
    #   overlap central= roofline + idle that SURVIVES per-layer GEMM overlap (D2);
    #                    the realistic as-built eager ATTENTION-path penalty
    #   isolation high = roofline + Part-D no-overlap idle (GPU-starved upper bound)
    idle_overlap_step = overlap["exposed_idle_overlap_us"] / STEP_M8_US
    idle_isolation_step = eager["exposed_idle_steady_us"] / STEP_M8_US
    step_graphed = STEP_WSTAR_DEPTH9
    step_overlap = STEP_WSTAR_DEPTH9 + idle_overlap_step
    step_isolation = STEP_WSTAR_DEPTH9 + idle_isolation_step
    # central = realistic-overlap (the as-run config); anchored by openevolve's
    # wall_tps if supplied (their localizer E[T] / wall_tps -> the FULL-step number).
    anchored = args.openevolve_wall_tps is not None and args.openevolve_wall_tps > 0
    if anchored:
        e_t_anchor = args.openevolve_e_t or E_T_OPENEVOLVE_ORACLE
        step_anchored = K_CAL * e_t_anchor / (args.openevolve_wall_tps * TAU_FERN_CENTRAL)
        measured_step = step_anchored
    else:
        step_anchored = None
        measured_step = step_overlap
    res["step1_measured_step"] = {
        "roofline_step": STEP_WSTAR_DEPTH9,
        "step_graphed_floor": step_graphed,
        "step_overlap_central": step_overlap,
        "step_isolation_high": step_isolation,
        "idle_overlap_step_units": idle_overlap_step,
        "idle_isolation_step_units": idle_isolation_step,
        "idle_isolation_steady_us": eager["exposed_idle_steady_us"],
        "idle_isolation_cold_us": eager["exposed_idle_cold_us"],
        "idle_overlap_us": overlap["exposed_idle_overlap_us"],
        "anchored": anchored, "step_anchored": step_anchored,
        "measured_depth9_step_time": measured_step,
        "delta_vs_roofline_abs": measured_step - STEP_WSTAR_DEPTH9,
        "delta_vs_roofline_pct": 100.0 * (measured_step - STEP_WSTAR_DEPTH9)
        / STEP_WSTAR_DEPTH9,
        "bracket_low_high": [step_graphed, step_isolation],
        "bracket_note": ("LOW = graphed floor (roofline, idle hidden); CENTRAL = "
                         "overlap-survivor (realistic eager attn path); HIGH = "
                         "no-overlap isolation (GPU-starved upper bound). The "
                         "isolation corner strips the per-layer GEMM GPU work that "
                         "hides the attention CPU dispatch -> over-states the idle."),
    }
    print(f"   [1] measured step = {measured_step:.4f} "
          f"(bracket [graphed {step_graphed:.4f}, isolation {step_isolation:.4f}], "
          f"overlap-central {step_overlap:.4f}, roofline {STEP_WSTAR_DEPTH9:.4f}, "
          f"delta +{res['step1_measured_step']['delta_vs_roofline_pct']:.2f}%)"
          f"{'  [ANCHORED]' if anchored else '  [overlap-central, openevolve wall_tps pending]'}",
          flush=True)

    # ---- Step 2: re-price the recipes at each step regime ---------------------
    tax_full = tax131["delta_full_central"]
    tax_full_cons = tax131["delta_full_conservative"]
    tax_root = tax_full / 32.0                # 1/32 of verify rows (depth-1 root)
    tax_root_cons = tax_full_cons / 32.0
    recipes = {}
    for stepname, step_bf16 in (("graphed_floor", step_graphed),
                                ("overlap_central", step_overlap),
                                ("measured", measured_step),
                                ("isolation_high", step_isolation)):
        recipes[stepname] = {
            "bf16_internal": _price_recipe_at_step(step_bf16, 0.0),
            "root_row_central": _price_recipe_at_step(step_bf16, tax_root),
            "root_row_conservative": _price_recipe_at_step(step_bf16, tax_root_cons),
            "full_fp32_central": _price_recipe_at_step(step_bf16, tax_full),
            "full_fp32_conservative": _price_recipe_at_step(step_bf16, tax_full_cons),
        }
    res["step2_recipes_at_step"] = {
        "tax_full_central": tax_full, "tax_root_central": tax_root,
        "tax_full_conservative": tax_full_cons, "tax_root_conservative": tax_root_cons,
        "tax_reduction_root_vs_full": tax_full / tax_root if tax_root else None,
        "by_step": recipes}
    mm = recipes["measured"]
    print(f"   [2] at measured step: bf16 step {mm['bf16_internal']['recipe_step']:.4f}  "
          f"root-row {mm['root_row_central']['recipe_step']:.4f}  "
          f"full-fp32 {mm['full_fp32_central']['recipe_step']:.4f}  "
          f"(root tax {tax_root:.5f} = full/{tax_full/tax_root:.0f})", flush=True)

    # ---- Step 3: knife-edge -- clear-500/530 bars + realized-official cross ----
    rr = mm["root_row_central"]["central"]
    rootrow_clears_530 = rr["clears530_under_ceiling"]
    # realized official at the measured step for the oracle-measured E[T] (cross).
    realized = {
        "e_t_openevolve_2.621": {
            "official_at_measured_step": fern_official(E_T_OPENEVOLVE_ORACLE, measured_step, 1.0),
            "official_at_roofline": fern_official(E_T_OPENEVOLVE_ORACLE, STEP_WSTAR_DEPTH9, 1.0)},
        "e_t_chiku_2.07": {
            "official_at_measured_step": fern_official(E_T_CHIKU_RUN, measured_step, 1.0),
            "official_at_roofline": fern_official(E_T_CHIKU_RUN, STEP_WSTAR_DEPTH9, 1.0)},
    }
    bar530_graphed = recipes["graphed_floor"]["root_row_central"]["central"]["clear530_bar_et"]
    bar530_overlap = recipes["overlap_central"]["root_row_central"]["central"]["clear530_bar_et"]
    bar530_isolation = recipes["isolation_high"]["root_row_central"]["central"]["clear530_bar_et"]
    res["step3_knife_edge"] = {
        "rootrow_clears_530_at_measured_step": int(rootrow_clears_530),
        "rootrow_clear530_bar_at_measured": rr["clear530_bar_et"],
        "rootrow_clear530_bar_graphed": bar530_graphed,
        "rootrow_clear530_bar_overlap_central": bar530_overlap,
        "rootrow_clear530_bar_isolation_high": bar530_isolation,
        "rootrow_clears_530_graphed": int(recipes["graphed_floor"]["root_row_central"]["central"]["clears530_under_ceiling"]),
        "rootrow_clears_530_overlap": int(recipes["overlap_central"]["root_row_central"]["central"]["clears530_under_ceiling"]),
        "rootrow_clears_530_isolation": int(recipes["isolation_high"]["root_row_central"]["central"]["clears530_under_ceiling"]),
        "supply_ceiling_e_t": E_T_TREE_CEILING,
        "full_fp32_clears_530_at_measured": int(mm["full_fp32_central"]["central"]["clears530_under_ceiling"]),
        "bf16_clears_530_at_measured": int(mm["bf16_internal"]["central"]["clears530_under_ceiling"]),
        "realized_official_cross": realized,
        "note": ("clears530 == clear-530 E[T] bar <= 5.207 supply ceiling. The "
                 "oracle-measured E[T]=2.621 FAILS 500 at any of these steps; the "
                 "bar question is conditional on a future build that fixes BUG-1/2."),
    }
    print(f"   [3] root-row clear-530 bar: graphed {bar530_graphed:.4f}  "
          f"overlap-central {bar530_overlap:.4f}  isolation {bar530_isolation:.4f} "
          f"(ceiling {E_T_TREE_CEILING}) -> clears530@measured={rootrow_clears_530}",
          flush=True)
    print(f"       realized official @measured: E[T]=2.621 -> "
          f"{realized['e_t_openevolve_2.621']['official_at_measured_step']:.1f}  "
          f"E[T]=2.07 -> {realized['e_t_chiku_2.07']['official_at_measured_step']:.1f} "
          f"(both FAIL 500 -- numerator bug, not denominator)", flush=True)

    # ---- Step 4: operative-bar shift -> hand to fern --------------------------
    bar500_measured = fern_clear_bar(TARGET_500, measured_step, TAU_FERN_CENTRAL)
    bar500_overlap = fern_clear_bar(TARGET_500, step_overlap, TAU_FERN_CENTRAL)
    bar500_isolation = fern_clear_bar(TARGET_500, step_isolation, TAU_FERN_CENTRAL)
    res["step4_operative_bar_shift"] = {
        "clear500_bar_roofline": CLEAR500_BAR_ROOFLINE,
        "clear500_bar_measured": bar500_measured,
        "clear500_bar_overlap_central": bar500_overlap,
        "clear500_bar_isolation_high": bar500_isolation,
        "shift_vs_roofline": bar500_measured - CLEAR500_BAR_ROOFLINE,
        "direction": "RISES (denominator bigger)" if bar500_measured > CLEAR500_BAR_ROOFLINE
        else "drops",
        "demand_floor_denken123": 4.624,
        "clear500_bar_still_under_ceiling": bar500_measured <= E_T_TREE_CEILING,
        "handoff": ("fern #129: replace the 1.2127-roofline clear-500 bar 4.841 with "
                    "the measured-step bar; the eager star-attn denominator raises it "
                    "(overlap-central is the realistic operative bar; isolation is the "
                    "no-overlap pessimist). A graphed build (blocker #2) recovers the "
                    "roofline bar."),
    }
    print(f"   [4] fern operative clear-500 bar: roofline 4.841 -> measured "
          f"{bar500_measured:.4f} ({res['step4_operative_bar_shift']['direction']}, "
          f"+{bar500_measured - CLEAR500_BAR_ROOFLINE:.4f} E[T])", flush=True)

    # ---- Gate (PR #136) -------------------------------------------------------
    clears_overlap = recipes["overlap_central"]["root_row_central"]["central"]["clears530_under_ceiling"]
    clears_isolation = recipes["isolation_high"]["root_row_central"]["central"]["clears530_under_ceiling"]
    clears_graphed = recipes["graphed_floor"]["root_row_central"]["central"]["clears530_under_ceiling"]
    materially_worse = res["step1_measured_step"]["delta_vs_roofline_pct"] > 2.0
    if anchored:
        verdict = "GREEN" if rootrow_clears_530 else "RED"
        verdict_reason = ("anchored by openevolve wall_tps; root-row "
                          + ("clears" if rootrow_clears_530 else "MISSES") + " 530")
    elif not clears_overlap and materially_worse:
        verdict = "RED"
        verdict_reason = ("even the realistic overlap-central eager step pushes the "
                          "root-row clear-530 bar above the 5.207 ceiling -> route 530 "
                          "elsewhere (graph the verify or root-row won't clear 530)")
    else:
        verdict = "AMBER"
        verdict_reason = ("openevolve full-step wall_tps not yet returned; reporting the "
                          "measured bracket [graphed, isolation] + operative-bar "
                          "sensitivity. root-row clears 530 at "
                          + ("overlap-central" if clears_overlap else "neither")
                          + (" but NOT the no-overlap isolation pessimist"
                             if (clears_overlap and not clears_isolation) else "")
                          + "; a graphed build recovers the roofline bar.")
    res["gate"] = {
        "verdict": verdict, "reason": verdict_reason,
        "anchored": anchored,
        "rootrow_clears_530_graphed": int(clears_graphed),
        "rootrow_clears_530_overlap_central": int(clears_overlap),
        "rootrow_clears_530_isolation_high": int(clears_isolation),
        "materially_worse_than_roofline": materially_worse,
        "rule": ("GREEN = step ANCHORED (openevolve full-step wall_tps) AND root-row "
                 "clears 530 at the measured step; AMBER = wall_tps pending, report "
                 "bracket + bar sensitivity; RED = realistic overlap-central step is "
                 "materially worse AND root-row clear-530 bar > 5.207 ceiling there"),
    }

    # ---- PRIMARY / TEST metrics ----------------------------------------------
    res["primary_metric"] = {"name": "measured_depth9_step_time", "value": measured_step}
    res["test_metric"] = {"name": "rootrow_clears_530_at_measured_step",
                          "value": int(rootrow_clears_530)}
    res["verdict"] = verdict
    res["elapsed_s"] = time.time() - t0
    res["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2))
    print(f"\n[anchor] VERDICT={verdict}  measured_step={measured_step:.4f}  "
          f"rootrow_clears_530={int(rootrow_clears_530)}", flush=True)
    print(f"[anchor] {verdict_reason}", flush=True)
    print(f"[anchor] wrote {out_path}  ({res['elapsed_s']:.0f}s, "
          f"peak {res['peak_gpu_gb']:.2f}GB)", flush=True)

    # ---- W&B -----------------------------------------------------------------
    if args.wandb_group:
        try:
            import wandb
            run_w = wandb.init(
                project="gemma-challenge-senpai", entity="wandb-applied-ai-team",
                group=args.wandb_group, name=args.wandb_name,
                config={**res["config"], **res["anchors"], "gpu": res["gpu"], "ctx": ctx})
            log = {
                "measured_depth9_step_time": measured_step,
                "rootrow_clears_530_at_measured_step": int(rootrow_clears_530),
                "step_roofline": STEP_WSTAR_DEPTH9,
                "step_graphed_floor": step_graphed,
                "step_overlap_central": step_overlap,
                "step_isolation_high": step_isolation,
                "delta_vs_roofline_pct": res["step1_measured_step"]["delta_vs_roofline_pct"],
                "eager_idle_isolation_steady_us": eager["exposed_idle_steady_us"],
                "eager_idle_isolation_cold_us": eager["exposed_idle_cold_us"],
                "eager_device_busy_us": eager["device_busy_us"],
                "overlap_idle_us": overlap["exposed_idle_overlap_us"],
                "overlap_device_busy_us": overlap["device_busy_us"],
                "overlap_per_call_idle_us": overlap["per_call_idle_overlap_us"],
                "per_call_idle_isolation_steady_us": eager["per_call_idle_steady_us"],
                "rootrow_clear530_bar_measured": rr["clear530_bar_et"],
                "rootrow_clear530_bar_graphed": bar530_graphed,
                "rootrow_clear530_bar_overlap_central": bar530_overlap,
                "rootrow_clear530_bar_isolation_high": bar530_isolation,
                "clear500_bar_roofline": CLEAR500_BAR_ROOFLINE,
                "clear500_bar_measured": bar500_measured,
                "clear500_bar_shift": bar500_measured - CLEAR500_BAR_ROOFLINE,
                "realized_official_et2621_measured": realized["e_t_openevolve_2.621"]["official_at_measured_step"],
                "supply_ceiling_e_t": E_T_TREE_CEILING,
                "verdict_green": int(verdict == "GREEN"),
                "verdict_amber": int(verdict == "AMBER"),
                "verdict_red": int(verdict == "RED"),
            }
            wandb.log(log)
            run_w.summary.update(log)
            res["wandb_run_id"] = run_w.id
            wandb.finish()
            print(f"[anchor] W&B run {run_w.id} (group {args.wandb_group})", flush=True)
            out_path.write_text(json.dumps(res, indent=2))
        except Exception as e:  # noqa: BLE001
            print(f"[anchor] W&B logging skipped: {e!r}", flush=True)
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
    # ---- PR #136 measured-anchor mode -------------------------------------
    ap.add_argument("--measured-anchor", action="store_true",
                    help="PR #136: measure the eager depth-9 step + re-price the "
                         "root-row recipe at the MEASURED (not roofline) step")
    ap.add_argument("--eager-passes", type=int, default=200,
                    help="back-to-back depth-9 steps timed for the eager idle gap")
    ap.add_argument("--gemm-filler-n", type=int, default=GEMM_FILLER_N,
                    help="Part D2 filler bf16 NxN GEMM size (per-layer non-attn GPU "
                         "proxy that hides the eager attention-launch CPU dispatch)")
    ap.add_argument("--openevolve-wall-tps", type=float, default=None,
                    help="if set, ANCHOR the step to openevolve's measured wall_tps "
                         "for tree-488-pw-fp32-v0 (step = K_cal*E[T]/wall_tps)")
    ap.add_argument("--openevolve-e-t", type=float, default=E_T_OPENEVOLVE_ORACLE,
                    help="oracle-measured E[T] to pair with --openevolve-wall-tps")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--wandb-group", type=str, default="fp32-star-steptime-tax")
    ap.add_argument("--wandb-name", type=str, default="lawine/fp32-star-steptime-tax")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)
    if args.no_wandb:
        args.wandb_group = None
    if args.output is None:
        args.output = ROOT / ("research/spec_cost_model/"
                              + ("fp32_star_steptime_measured_anchor.json"
                                 if args.measured_anchor
                                 else "fp32_star_steptime_tax.json"))
    if args.measured_anchor:
        run_measured_anchor(args)
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
