#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Verify-step(M) cost curve (PR #153): is the depth-9 verify step FLAT in M?

THE QUESTION (the DENOMINATOR shape wirbel #152 needs)
------------------------------------------------------
lawine #136 measured the depth-9 verify step at the deployed M=32 tree = 1.2182
(M=8-step-normalized units; +0.45% over the 1.2127 roofline) and showed the step
is GPU-bound. But it was a SINGLE point. The verify GEMM processes M candidate
rows in one int4-Marlin batch; at conc=1 decode the path is weight-read-dominated
(BASELINE profiler ~92% weight-GEMM). IF the GEMM is weight-read-bound and M=32
sits inside a single HBM/occupancy wave, step(M) should be ~FLAT in M up to some
M_crit -- adding candidate rows reuses the same weight read until the GEMM
saturates compute/occupancy. If FLAT, a bigger draft tree (more nodes -> more
E[T] coverage) is nearly FREE and descent-only could clear 530. If RISING, M~=32
is already near the cost-optimal knee. This script MEASURES step(M) and hands
wirbel #152 the real curve.

WHAT IS MEASURED (reuses my #136 + denken #68 + my #107 methodology verbatim)
-----------------------------------------------------------------------------
For each M in {8,16,24,32,48,64,96,128} (M=8 = linear-MTP 481.53 frontier,
M=32 = deployed tree / my #136 anchor, up to 128):
  * verify-GEMM(M)  -- real int4 W4A16 Marlin weight GEMMs (qkv/o/gate_up/down)
                       summed over the loaded decoder layers, launch-free CUDA
                       graph (the true kernel time). denken #68 found Marlin
                       16-row tile cliffs at M=33/49 -- this resolves the staircase.
  * verify-attn(M)  -- deployed fp32-partial split-KV unified_attention summed
                       over the sliding+full layers (wirbel #98 / my #107 basis).
  * eager-idle(M)   -- the eager star-attn launch idle that SURVIVES per-layer
                       GEMM overlap (my #136 INTERLEAVED method, NOT isolation;
                       isolation over-reads ~80x). ~M-invariant: 37 launches/step
                       regardless of M, and wider M gives more GPU work to hide
                       behind -> idle flat/shrinking.
  * drafter-fill(M) -- the Gemma4 MTP K=7 drafter per-pass GEMM chain at the
                       per-pass frontier b(M)=ceil(M/K), x K passes (denken #69 /
                       drafter_forward_roofline basis). The tree grows by BRANCHING
                       (wider frontier) not deeper rollout, so the drafter weight
                       read amortizes and the cost grows much slower than verify.

COMPOSITION (reproduces my #136 1.2182 at M=32 BY CONSTRUCTION)
--------------------------------------------------------------
fern #129's depth-9 roofline decomposes (M=8 step == 1.0) as
  STEP_WSTAR_DEPTH9 = 1.2127 = GEMM(1.0981) + DRAFTER_ADD(0.048) + ATTN_ADD(0.0666)
where the DRAFTER/ATTN ADDs are the TREE deltas over the M=8 linear chain (0 at
M=8). I keep those exact M=32 anchors and let the MEASURED raw curves drive the
M-shape (pinned to 0 at M=8 and to fern's constant at M=32, interpolated/extra-
polated by the measured excess):
  dGEMM(M)  = (STEP_WSTAR_GEMM-1) * (g[M]-g[8]) / (g[32]-g[8])
  dATTN(M)  = STEP_WSTAR_ATTN_ADD * (a[M]-a[8]) / (a[32]-a[8])
  dDRAFT(M) = STEP_WSTAR_DRAFTER_ADD * (dft[M]-dft[8]) / (dft[32]-dft[8])
  didle(M)  = idle_overlap_us(M) / STEP_M8_US        [freshly measured each M]
  step_norm(M) = 1 + dGEMM + dATTN + dDRAFT + didle
At M=32 this is 1.2127 + idle(~43us)/7982.86 ~= 1.2182 (my #136), the hard
cross-check. At M=8 it is 1.000 (the linear 481.53 frontier), the 2nd cross-check.

DELIVERABLES
------------
  primary  verify_step_flat_M_ceiling = largest M with step(M) <= 1.02*step(32)
  test     step_M128_rel_increase_pct = (step(128)/step(32)-1)*100
  hand-off research/oracle_readout/verify_step_m_curve.json with
           {M, step_ms, step_rel_m32, drafter_fill_pct} -- drops into wirbel
           #152's --step-m-json. Plus raw measured component us for transparency.

LOCAL A10G micro-bench. NO HF Job, NO submission, NO kernel build, NO served-file
change. Timing only -> greedy identity untouched by construction. BASELINE 481.53.
"""
from __future__ import annotations

import os

# Pin the single visible A10G + disable the broken flashinfer sampler JIT BEFORE
# importing torch/vllm (project_local_a10g_gpu_env). Force the deployed split-KV
# verify path so M>1 verify batches route to the 3D split-KV attention we price.
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

# --- measurement primitives, reused VERBATIM from the merged profilers ---------
from scripts.profiler.verify_gemm_roofline import (  # noqa: E402
    DEFAULT_MODEL,
    build_llm,
    collect_gemm_instances,
    find_decoder_layers,
    find_runner,
    uniquify,
)
from scripts.profiler.tree_step_denominator import (  # noqa: E402
    layer_counts_from_gemm,
    measure_attn_us,
    measure_gemm_us,
)
from scripts.profiler.star_attn_fp32_steptime import (  # noqa: E402
    K_CAL,
    R_ATTN_107,
    R_GEMM_107,
    STEP_M8_US,
    STEP_WSTAR_ATTN_ADD,
    STEP_WSTAR_DEPTH9,
    STEP_WSTAR_DRAFTER_ADD,
    STEP_WSTAR_GEMM,
    measure_overlap_hidden_idle,
)
from scripts.local_validation.profile_attention import _maybe_install_splitkv  # noqa: E402

Z95 = 1.959963984540054
ATTN_CTX_DEFAULT = 528          # deployed mean served ctx ~527.7 (wirbel #98)
M_ANCHOR = 32                   # my #136 anchor (deployed tree)
M_LINEAR = 8                    # linear-MTP 481.53 frontier
DEFAULT_M_SWEEP = "8,16,24,32,48,64,96,128"
DEFAULT_DRAFTER = "/tmp/qat-assistant"
K_SPEC = 7                      # num_speculative_tokens (deployed MTP K=7)
# denken #69-corrected drafter-forward share of the deployed (M=32) decode step.
DRAFTER_SHARE_M32_LO = 0.155
DRAFTER_SHARE_M32_HI = 0.181
DRAFTER_SHARE_M32_CENTRAL = 0.5 * (DRAFTER_SHARE_M32_LO + DRAFTER_SHARE_M32_HI)
FLAT_TOL = 1.02                 # step(M) <= 1.02*step(32) == "free-M" (<= +2%)


def summarize(values):
    n = len(values)
    if n == 0:
        return {"n": 0, "median": float("nan"), "mean": float("nan"), "cv_pct": 0.0}
    mean = statistics.fmean(values)
    median = statistics.median(values)
    std = statistics.pstdev(values) if n > 1 else 0.0
    se = std / math.sqrt(n) if n else 0.0
    return {"n": n, "mean": mean, "median": median, "std": std, "se": se,
            "cv_pct": 100.0 * std / mean if mean else 0.0, "ci95_abs": Z95 * se,
            "min": min(values), "max": max(values), "values": values}


# ===== drafter per-pass chain (denken #69 / drafter_forward_roofline) ==========
def measure_drafter_passcost(drafter_dir, b_values, iters, warmup):
    """Time the FULL Gemma4-MTP drafter per-pass GEMM chain (one onegraph replay of
    every per-pass weight GEMM) at each frontier width b. Returns {b: per_pass_us}.
    The drafter is built, timed, and FREED before the big int4 model loads so peak
    memory stays low. Guarded: a drafter failure leaves the verify deliverable intact."""
    from scripts.profiler.drafter_forward_roofline import (
        build_drafter_gemms, time_pass_chain_graph,
    )
    gemms, per_pass_multiset = build_drafter_gemms(drafter_dir)
    lookup = {(u["in"], u["out"]): u["module"] for u in gemms}
    modules_in_order = [(lookup[(inn, out)], inn) for (inn, out) in per_pass_multiset
                        if (inn, out) in lookup]
    dense_w_mb = sum(2.0 * u["in"] * u["out"] * u["count"] for u in gemms) / 1e6
    # clock-ramp + JIT warmup over the frontier range (same A10G ramp caveat as #107)
    for _ in range(3):
        for b in b_values:
            time_pass_chain_graph(modules_in_order, b, max(40, warmup), warmup)
    torch.cuda.synchronize()
    out = {}
    for b in b_values:
        ms_g, ms_e, captured = time_pass_chain_graph(modules_in_order, b, iters, warmup)
        out[b] = {"per_pass_us": ms_g * 1000.0, "per_pass_us_eager": ms_e * 1000.0,
                  "captured": bool(captured)}
    # free the drafter modules before the big model load
    for u in gemms:
        u["module"].to("cpu")
    del gemms, modules_in_order, lookup
    torch.cuda.empty_cache()
    return out, {"per_pass_gemm_count": len(per_pass_multiset),
                 "dense_w_mb_per_pass": dense_w_mb, "n_unique_gemms": len(out)}


def drafter_total_us(passcost, b):
    """K=7 sequential MTP passes at per-pass frontier b -> total drafter-fill us."""
    return K_SPEC * passcost[b]["per_pass_us"]


# ===== verify sweep ============================================================
def sweep_verify_once(uniq, counts, ctx, m_list, gemm_iters, gemm_warmup,
                      attn_iters, attn_warmup):
    g, a = {}, {}
    gmeta = {}
    for M in m_list:
        gus, gm = measure_gemm_us(uniq, M, gemm_iters, gemm_warmup)
        aus, _ = measure_attn_us(M, ctx, attn_iters, attn_warmup, counts)
        g[M] = gus
        a[M] = aus
        gmeta[M] = gm["all_graphed"]
    return g, a, gmeta


def run(args) -> dict:
    assert torch.cuda.is_available(), "CUDA required"
    t0 = time.time()
    ctx = args.ctx
    m_list = [int(x) for x in args.m_sweep.split(",") if x.strip()]
    assert M_LINEAR in m_list and M_ANCHOR in m_list, "sweep must include M=8 and M=32 anchors"
    torch.cuda.reset_peak_memory_stats()

    res: dict = {
        "pr": 153, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "l2_bytes": torch.cuda.get_device_properties(0).L2_cache_size,
        "ctx": ctx, "m_sweep": m_list,
        "anchors": {
            "step_wstar_depth9_m32": STEP_WSTAR_DEPTH9,
            "step_wstar_gemm_m32": STEP_WSTAR_GEMM,
            "step_wstar_drafter_add_m32": STEP_WSTAR_DRAFTER_ADD,
            "step_wstar_attn_add_m32": STEP_WSTAR_ATTN_ADD,
            "measured_step_m32_136": 1.2182, "step_m8_us": STEP_M8_US,
            "k_cal": K_CAL, "k_spec": K_SPEC,
            "drafter_share_m32_central_69": DRAFTER_SHARE_M32_CENTRAL,
        },
        "config": {"gemm_iters": args.gemm_iters, "gemm_warmup": args.gemm_warmup,
                   "attn_iters": args.attn_iters, "attn_warmup": args.attn_warmup,
                   "repeats": args.repeats, "eager_passes": args.eager_passes,
                   "drafter_dir": args.drafter_dir, "flat_tol": FLAT_TOL},
    }
    print(f"[mcurve] GPU {res['gpu']} ctx={ctx} sweep={m_list}", flush=True)

    # ---- Phase 0: drafter per-pass staircase (built + freed BEFORE big model) --
    b_for_M = {M: max(1, math.ceil(M / K_SPEC)) for M in m_list}
    b_values = sorted(set(b_for_M.values()))
    drafter = None
    try:
        print(f"[mcurve] drafter per-pass chain @ frontier b={b_values} "
              f"(K={K_SPEC} passes/step) ...", flush=True)
        passcost, dmeta = measure_drafter_passcost(
            args.drafter_dir, b_values, args.gemm_iters, args.gemm_warmup)
        drafter = {"passcost": passcost, "meta": dmeta, "b_for_M": b_for_M}
        for b in b_values:
            print(f"   [drafter] b={b:3d}: per-pass {passcost[b]['per_pass_us']:7.1f}us "
                  f"(graphed={passcost[b]['captured']})", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[mcurve] drafter phase FAILED ({e!r}); drafter-fill modeled FLAT "
              f"(weight-read-bound fallback)", flush=True)
        drafter = None

    # ---- Phase 1: load int4 model + introspect verify GEMMs (denken #68) -------
    print(f"[mcurve] building LLM {args.model} ...", flush=True)
    llm = build_llm(args.model, args.max_ctx)
    runner = find_runner(llm)
    if runner is None:
        raise RuntimeError("could not locate GPUModelRunner")
    layers = find_decoder_layers(runner.model)
    uniq = uniquify(collect_gemm_instances(layers))
    counts = layer_counts_from_gemm(uniq)
    print(f"[mcurve] LLM ready in {time.time()-t0:.0f}s | layers={len(layers)} | "
          f"attn sliding={counts['sliding']} full={counts['full']} | "
          f"GEMM shapes={len(uniq)}", flush=True)
    _maybe_install_splitkv()

    # ---- clock-ramp + JIT warmup over EVERY swept M (MANDATORY; #107 caveat) ----
    # A10G base->boost ramp (~1.9x) + Triton/Marlin first-use JIT: the FIRST timed
    # kernel at each shape runs ~2x slow. Sustained throwaway pass over ALL M sheds
    # it so the staircase is not corrupted by order-dependent clock state.
    print(f"[mcurve] warmup: ramping clocks + JIT over all {len(m_list)} widths ...",
          flush=True)
    for _ in range(2):
        for M in m_list:
            measure_gemm_us(uniq, M, max(60, args.gemm_warmup), args.gemm_warmup)
            measure_attn_us(M, ctx, max(60, args.attn_warmup), args.attn_warmup, counts)
    torch.cuda.synchronize()
    print(f"[mcurve] warmup done ({time.time()-t0:.0f}s); timed repeats x{args.repeats}",
          flush=True)

    # ---- Phase 2: N fresh repeats of the GEMM + attn sweep (median-of-N) --------
    g_runs = {M: [] for M in m_list}
    a_runs = {M: [] for M in m_list}
    graphed_all = True
    for r in range(args.repeats):
        g, a, gmeta = sweep_verify_once(
            uniq, counts, ctx, m_list, args.gemm_iters, args.gemm_warmup,
            args.attn_iters, args.attn_warmup)
        for M in m_list:
            g_runs[M].append(g[M])
            a_runs[M].append(a[M])
            graphed_all = graphed_all and gmeta[M]
        print(f"   [rep {r+1}/{args.repeats}] "
              + "  ".join(f"M{M}:g{g[M]/1000:.2f}+a{a[M]/1000:.2f}ms" for M in m_list),
              flush=True)
    g_us = {M: statistics.median(g_runs[M]) for M in m_list}
    a_us = {M: statistics.median(a_runs[M]) for M in m_list}
    g_stat = {M: summarize(g_runs[M]) for M in m_list}
    a_stat = {M: summarize(a_runs[M]) for M in m_list}

    # ---- Phase 3: eager-idle that survives GEMM overlap, per M (#136 interleaved)
    idle_us = {}
    for M in m_list:
        ov = measure_overlap_hidden_idle(
            M, ctx, args.eager_passes, args.attn_warmup, counts, args.attn_iters,
            args.gemm_filler_n)
        idle_us[M] = ov["exposed_idle_overlap_us"]
        print(f"   [idle] M={M:3d}: overlap-survivor {idle_us[M]:6.1f}us "
              f"({ov['per_call_idle_overlap_us']:.2f}/call)", flush=True)

    # ---- Phase 4: compose step_norm(M) (reproduces 1.2182 at M=32) --------------
    g8, g32 = g_us[M_LINEAR], g_us[M_ANCHOR]
    a8, a32 = a_us[M_LINEAR], a_us[M_ANCHOR]
    gemm_excess_ref = STEP_WSTAR_GEMM - 1.0  # 0.09815 (fern GEMM contribution @ M=32)

    # drafter raw fill us @ each M (K passes at frontier b(M)); fallback FLAT
    if drafter is not None:
        dft_us = {M: drafter_total_us(drafter["passcost"], b_for_M[M]) for M in m_list}
    else:
        dft_us = {M: 1.0 for M in m_list}  # flat -> dDRAFT==0, drafter_pct from #69 only
    dft8, dft32 = dft_us[M_LINEAR], dft_us[M_ANCHOR]

    def frac_excess(x, x8, x32):
        denom = x32 - x8
        return (x - x8) / denom if abs(denom) > 1e-9 else 0.0

    # drafter tree-expansion add: pin to fern's anchors (0 at M=8, 0.048 at M=32)
    # and extend past M=32 by the MEASURED per-pass staircase. frac_excess on the
    # near-FLAT drafter (per-pass ~107us across b=2..5) is ill-conditioned (noise
    # denominator), so drive the expansion by a robust monotone scale instead:
    #   M<=8 -> 0 ; 8<M<=32 -> linear ramp (M-8)/24 ; M>32 -> per_pass(b(M))/per_pass(b(32))
    # (the drafter grows the tree by BRANCHING/wider frontier, not deeper rollout;
    # measured per-pass flatness => the >M32 extension stays ~flat -- the PR's point).
    def drafter_scale(M):
        if M <= M_LINEAR:
            return 0.0
        if M <= M_ANCHOR:
            return (M - M_LINEAR) / (M_ANCHOR - M_LINEAR)
        if drafter is not None:
            b32 = b_for_M[M_ANCHOR]
            return drafter["passcost"][b_for_M[M]]["per_pass_us"] / \
                drafter["passcost"][b32]["per_pass_us"]
        return 1.0  # flat fallback

    curve = {}
    for M in m_list:
        d_gemm = gemm_excess_ref * frac_excess(g_us[M], g8, g32)
        d_attn = STEP_WSTAR_ATTN_ADD * frac_excess(a_us[M], a8, a32)
        d_draft = STEP_WSTAR_DRAFTER_ADD * drafter_scale(M)
        d_idle = idle_us[M] / STEP_M8_US
        # graphed floor = GEMM+attn+drafter roofline (no eager idle); =1.0 at M=8
        # (the 481.53 frontier roofline floor) and =1.2127 at M=32 (#136 roofline).
        # eager step (step_norm) = graphed + the surviving eager attn-launch idle;
        # =1.2182 at M=32 (#136 anchor). The 481.53 K_cal calibration absorbs the
        # M=8 eager idle, so the M=8 cross-check uses the graphed floor (=1.0).
        step_graphed = 1.0 + d_gemm + d_attn + d_draft
        step_norm = step_graphed + d_idle
        curve[M] = {
            "M": M, "step_norm": step_norm, "step_graphed": step_graphed,
            "d_gemm": d_gemm, "d_attn": d_attn, "d_draft_expansion": d_draft,
            "d_idle": d_idle,
            "gemm_us": g_us[M], "attn_us": a_us[M], "idle_us": idle_us[M],
            "drafter_fill_us": dft_us[M], "drafter_frontier_b": b_for_M[M],
            "gemm_rel_m32": g_us[M] / g32, "gemm_rel_m8": g_us[M] / g8,
            "attn_rel_m32": a_us[M] / a32,
            "gemm_cv_pct": g_stat[M]["cv_pct"], "attn_cv_pct": a_stat[M]["cv_pct"],
        }
    step32 = curve[M_ANCHOR]["step_norm"]
    step8 = curve[M_LINEAR]["step_norm"]

    # total drafter share for wirbel's verify+drafter pricing: anchor MAGNITUDE to
    # denken #69's deployed-M32 share, scale SHAPE by the measured per-pass staircase.
    for M in m_list:
        if drafter is not None:
            shape = dft_us[M] / dft32
        else:
            shape = 1.0
        drafter_total_norm = DRAFTER_SHARE_M32_CENTRAL * step32 * shape
        curve[M]["drafter_total_norm"] = drafter_total_norm
        curve[M]["drafter_fill_pct"] = 100.0 * drafter_total_norm / curve[M]["step_norm"]
        curve[M]["step_rel_m32"] = curve[M]["step_norm"] / step32
        curve[M]["step_ms"] = curve[M]["step_norm"] * STEP_M8_US / 1000.0

    # ---- Phase 5: M_crit flat ceiling + verdict --------------------------------
    flat_ceiling = max((M for M in m_list if curve[M]["step_norm"] <= FLAT_TOL * step32),
                       default=M_ANCHOR)
    m128 = max(m_list)
    step_m128_rel_increase_pct = 100.0 * (curve[m128]["step_norm"] / step32 - 1.0)
    # is drafter the binding term? compare drafter staircase vs verify-GEMM staircase
    gemm_grow_32_to_max = g_us[m128] / g32
    draft_grow_32_to_max = (dft_us[m128] / dft32) if drafter is not None else 1.0
    drafter_is_binding = draft_grow_32_to_max > gemm_grow_32_to_max

    # cross-check #1 (M=32): eager step reproduces #136's 1.2182 (validates idle).
    m32_reproduces = abs(step32 - 1.2182) <= 0.01  # within +/-0.01 step-units
    # cross-check #2 (M=8): graphed floor = 1.0 (481.53 frontier roofline) by
    # construction; the MEANINGFUL validation is the measured M=32/M=8 GEMM + attn
    # ratios matching my #107 denominator (r_gemm 1.1686, r_attn 1.8325).
    step_graphed8 = curve[M_LINEAR]["step_graphed"]
    m8_graphed_reproduces = abs(step_graphed8 - 1.0) <= 1e-6
    r_gemm_meas = g32 / g8
    r_attn_meas = a32 / a8
    r_gemm_matches_107 = abs(r_gemm_meas - R_GEMM_107) / R_GEMM_107 <= 0.05  # within 5%
    r_attn_matches_107 = abs(r_attn_meas - R_ATTN_107) / R_ATTN_107 <= 0.08  # within 8%

    res["raw"] = {
        "gemm_us": g_us, "attn_us": a_us, "idle_us": idle_us, "drafter_fill_us": dft_us,
        "gemm_stat": g_stat, "attn_stat": a_stat, "all_graphed": graphed_all,
    }
    res["drafter"] = drafter if drafter is not None else {"status": "FLAT_FALLBACK"}
    res["curve"] = {str(M): curve[M] for M in m_list}
    res["analysis"] = {
        "step_norm_m8_eager": step8, "step_graphed_m8": step_graphed8,
        "step_norm_m32_eager": step32, "step_graphed_m32": curve[M_ANCHOR]["step_graphed"],
        "m32_reproduces_1p2182": bool(m32_reproduces),
        "m8_graphed_reproduces_1p0": bool(m8_graphed_reproduces),
        "r_gemm_meas_m32_over_m8": r_gemm_meas, "r_gemm_107": R_GEMM_107,
        "r_gemm_matches_107": bool(r_gemm_matches_107),
        "r_attn_meas_m32_over_m8": r_attn_meas, "r_attn_107": R_ATTN_107,
        "r_attn_matches_107": bool(r_attn_matches_107),
        "verify_step_flat_M_ceiling": flat_ceiling,
        "step_M128_rel_increase_pct": step_m128_rel_increase_pct,
        "gemm_staircase_rel_m8": {str(M): g_us[M] / g8 for M in m_list},
        "gemm_staircase_rel_m32": {str(M): g_us[M] / g32 for M in m_list},
        "gemm_grow_m32_to_max": gemm_grow_32_to_max,
        "drafter_grow_m32_to_max": draft_grow_32_to_max,
        "drafter_is_binding_term": bool(drafter_is_binding),
        "flat_tol_pct": 100.0 * (FLAT_TOL - 1.0),
    }
    # verdict: FLAT (free tree-growth headroom past 32) vs KNEE (M~=32 near optimal)
    if flat_ceiling >= 64:
        verdict = "FLAT"
        verdict_reason = (f"step(M) <= +2% out to M={flat_ceiling}; growing M past 32 "
                          f"buys descent-only E[T] toward 530 nearly free "
                          f"(M=128 only +{step_m128_rel_increase_pct:.1f}%)")
    elif flat_ceiling >= 48:
        verdict = "SOFT_KNEE"
        verdict_reason = (f"step(M) flat to M={flat_ceiling}, then rises; modest free "
                          f"headroom (M=48) above the M=32 deployed tree "
                          f"(M=128 +{step_m128_rel_increase_pct:.1f}%)")
    else:
        verdict = "KNEE_AT_32"
        first_over = min((m for m in m_list if m > M_ANCHOR), default=M_ANCHOR)
        verdict_reason = (f"step(M) leaves the flat regime by M={first_over}; "
                          f"M~=32 is already near the cost-optimal knee "
                          f"(M=128 +{step_m128_rel_increase_pct:.1f}%)")
    res["verdict"] = verdict
    res["verdict_reason"] = verdict_reason
    res["primary_metric"] = {"name": "verify_step_flat_M_ceiling", "value": flat_ceiling}
    res["test_metric"] = {"name": "step_M128_rel_increase_pct",
                          "value": step_m128_rel_increase_pct}
    res["elapsed_s"] = time.time() - t0
    res["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9

    # ---- wirbel #152 hand-off JSON (the --step-m-json drop-in) ------------------
    handoff = {
        "_source": "lawine PR#153 verify_step_m_curve.py",
        "_basis": ("step_ms = K_cal-normalized official-basis step (1 unit = "
                   "STEP_M8_US us = 1/K_cal s); step_rel_m32 normalized to M=32=1.0; "
                   "reproduces lawine #136 1.2182 at M=32 and 1.000 at M=8."),
        "M": m_list,
        "step_ms": [curve[M]["step_ms"] for M in m_list],
        "step_rel_m32": [curve[M]["step_rel_m32"] for M in m_list],
        "step_norm_m8basis": [curve[M]["step_norm"] for M in m_list],
        "drafter_fill_pct": [curve[M]["drafter_fill_pct"] for M in m_list],
        "verify_step_flat_M_ceiling": flat_ceiling,
        "step_M128_rel_increase_pct": step_m128_rel_increase_pct,
        "verdict": verdict,
    }
    res["wirbel152_handoff"] = handoff

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2))
    handoff_path = out_path.parent / "verify_step_m_curve_handoff.json"
    handoff_path.write_text(json.dumps(handoff, indent=2))

    print(f"\n[mcurve] === step(M) curve (M=8-norm; M=32 anchor {step32:.4f}) ===",
          flush=True)
    for M in m_list:
        c = curve[M]
        print(f"   M={M:3d} | step {c['step_norm']:.4f} (rel32 {c['step_rel_m32']:.4f}, "
              f"{c['step_ms']:.3f}ms) | gemm {c['gemm_us']/1000:6.3f}ms "
              f"(rel32 {c['gemm_rel_m32']:.3f}) | attn {c['attn_us']/1000:.3f}ms | "
              f"idle {c['idle_us']:5.0f}us | draft_pct {c['drafter_fill_pct']:.1f}%",
              flush=True)
    print(f"[mcurve] VERDICT={verdict}  flat_M_ceiling={flat_ceiling}  "
          f"M{m128} +{step_m128_rel_increase_pct:.1f}%  "
          f"m32_repro_1.2182={m32_reproduces}({step32:.4f}) "
          f"m8_graphed_repro_1.0={m8_graphed_reproduces}({step_graphed8:.4f}) "
          f"r_gemm={r_gemm_meas:.4f}(107={R_GEMM_107},ok={r_gemm_matches_107}) "
          f"r_attn={r_attn_meas:.4f}(107={R_ATTN_107},ok={r_attn_matches_107})",
          flush=True)
    print(f"[mcurve] {verdict_reason}", flush=True)
    print(f"[mcurve] wrote {out_path} + {handoff_path}  "
          f"({res['elapsed_s']:.0f}s, peak {res['peak_gpu_gb']:.3f}GB)", flush=True)

    # ---- W&B -------------------------------------------------------------------
    if not args.no_wandb:
        try:
            import wandb
            run_w = wandb.init(
                project=args.wandb_project, entity=args.wandb_entity,
                group=args.wandb_group, name=args.wandb_name,
                config={**res["config"], **res["anchors"], "gpu": res["gpu"],
                        "ctx": ctx, "m_sweep": m_list})
            # per-M curve as a wandb Table for plotting
            cols = ["M", "step_norm", "step_rel_m32", "step_ms", "gemm_us", "attn_us",
                    "idle_us", "drafter_fill_us", "drafter_fill_pct", "gemm_rel_m32"]
            tbl = wandb.Table(columns=cols)
            for M in m_list:
                c = curve[M]
                tbl.add_data(*[c[k] for k in cols])
                # also log per-M as stepped metrics for line plots over M
                wandb.log({f"step_norm": c["step_norm"], "step_rel_m32": c["step_rel_m32"],
                           "step_ms": c["step_ms"], "gemm_us": c["gemm_us"],
                           "attn_us": c["attn_us"], "idle_us": c["idle_us"],
                           "drafter_fill_pct": c["drafter_fill_pct"],
                           "gemm_rel_m32": c["gemm_rel_m32"], "M": M})
            summary = {
                "verify_step_flat_M_ceiling": flat_ceiling,
                "step_M128_rel_increase_pct": step_m128_rel_increase_pct,
                "step_norm_m32": step32, "step_norm_m8": step8,
                "m32_reproduces_1p2182": int(m32_reproduces),
                "m8_graphed_reproduces_1p0": int(m8_graphed_reproduces),
                "r_gemm_meas_m32_over_m8": r_gemm_meas,
                "r_gemm_matches_107": int(r_gemm_matches_107),
                "r_attn_meas_m32_over_m8": r_attn_meas,
                "r_attn_matches_107": int(r_attn_matches_107),
                "gemm_grow_m32_to_max": gemm_grow_32_to_max,
                "drafter_grow_m32_to_max": draft_grow_32_to_max,
                "drafter_is_binding_term": int(drafter_is_binding),
                "all_graphed": int(graphed_all),
                "verdict_flat": int(verdict == "FLAT"),
                "verdict_soft_knee": int(verdict == "SOFT_KNEE"),
                "verdict_knee_at_32": int(verdict == "KNEE_AT_32"),
                "peak_gpu_gb": res["peak_gpu_gb"], "elapsed_s": res["elapsed_s"],
            }
            wandb.log({"curve_table": tbl})
            wandb.log(summary)
            run_w.summary.update(summary)
            res["wandb_run_id"] = run_w.id
            wandb.finish()
            print(f"[mcurve] W&B run {run_w.id} (group {args.wandb_group})", flush=True)
            out_path.write_text(json.dumps(res, indent=2))
        except Exception as e:  # noqa: BLE001
            print(f"[mcurve] W&B logging skipped: {e!r}", flush=True)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--m-sweep", default=DEFAULT_M_SWEEP)
    ap.add_argument("--ctx", type=int, default=ATTN_CTX_DEFAULT)
    ap.add_argument("--max-ctx", type=int, default=256, help="vLLM max ctx for the GEMM load")
    ap.add_argument("--drafter-dir", default=DEFAULT_DRAFTER)
    ap.add_argument("--gemm-iters", type=int, default=200)
    ap.add_argument("--gemm-warmup", type=int, default=40)
    ap.add_argument("--attn-iters", type=int, default=300)
    ap.add_argument("--attn-warmup", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=3, help="fresh repeats (median-of-N)")
    ap.add_argument("--eager-passes", type=int, default=120,
                    help="passes for the interleaved eager-idle measurement")
    ap.add_argument("--gemm-filler-n", type=int, default=2048)
    ap.add_argument("--output", default="research/oracle_readout/verify_step_m_curve.json")
    ap.add_argument("--wandb-project", default="gemma-challenge-senpai")
    ap.add_argument("--wandb-entity", default="wandb-applied-ai-team")
    ap.add_argument("--wandb-name", default="lawine/verify-step-m-cost-curve")
    ap.add_argument("--wandb-group", default="verify-step-m-cost-curve")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
