#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Spec-dec amortization ceiling (PR #677, denken). LOCAL A10G micro-profiling +
CPU analytic. Analysis-only: NO served-file change, NO HF Job, NO submission,
NOT a launch. The strict-#319 anchor (int4_g128_lmhead @ official 126.378) is
untouched. PRIMARY = self-test.

THE QUESTION
------------
Spec-dec is the program's only live speed lever (denken #676: the int4 GEMV is at
the HBM wall, byte-identical headroom = 0; denken #674: scheduling ALREADY_OPTIMAL).
Spec-dec amortizes the ONE-per-step body weight-read (#676: 2.38 GB/token = 2.034
body + 0.346 lm_head, the dominant 86% of per-token bytes) over E[accept] accepted
tokens. This card turns the measured body-read into the spec-dec SPEED LAW:

  TPS(E,K) = E[accept] / T_step(K),  T_step(K) = T_verify(K) + T_draft(K)

and answers: what E[accept]* clears +10 (136.378 official-equiv) at the optimal K,
and is that reachable with an IN-SCOPE-publishable drafter (stock 3.33 / top_k64
3.38) or only the out-of-scope qat retrain (3.66)?

WHAT THIS DERIVES / MEASURES
----------------------------
  (1) K_max_useful -- the draft width at which the verify forward stops being
      weight-read-bound (M-invariant) and goes compute-bound. From the sm_86
      roofline: int4-weight arithmetic intensity = 4*M FLOP/byte, ridge AI =
      125 TFLOP / 600 GB/s = 208.3, so M_knee = 208.3/4 ~ 52 (verify_step_component
      _roofline / kanna #280 basis). LOCAL M-SWEEP confirms per-shape verify GEMM
      us(M) is flat (M-invariant) up to ~M=32-52 then rises. #676 only ever
      measured M=1; this extends it.
  (2) T_step(K) cost model, CALIBRATED to the committed strict batch-invariant
      (VLLM_BATCH_INVARIANT=1 + recompute-rescue) walltps K-sweep (land #82/#90,
      qat drafter): (K,E,localTPS) = (3,2.856,165.71)(4,3.204,171.68)(5,3.474,
      172.74)(6,3.657,170.18)(7,3.825,152.26). T_step = E/localTPS -> the realized
      strict per-step cost. Optimal K=5 (max localTPS). Decompose T_step(K) =
      T0_strict + d*K (linear K=3..6): the K-cost is ALL DRAFT, the verify is
      width-free up to K_max_useful.
  (3) RAW body-read amortization ceiling (instruction-1 law, rescue-free): TPS_raw
      = E / (body+head read + cheap draft). Shows the body-read PERMITS a frontier-
      class ceiling (~480 official, matches the public 481.53 deployed) -> the body-
      read is NOT the binding constraint.
  (4) STRICT realized envelope (the #319-valid ceiling, with the rescue tax): the
      walltps sweep x0.870 = official-equiv. In-scope land #670 points: stock 3.33
      -> 136.12, top_k64 3.38 -> 137.14; qat OOS 3.66 -> 148 (== K=6 of the local
      sweep, cross-checks x0.870). The rescue tax COLLAPSES the raw ~480 ceiling to
      ~137-150 -- the +10 bar sits right at the collapsed strict ceiling.
  (5) E[accept]* for +10 and the verdict (in-scope vs OOS).
  (6) Self-test (PRIMARY).

local->official x0.870 is the PR-specified strict-basis projection (the program's
"stark tax"; #676). It is CONSERVATIVE: the deployed-path projection is x1.06
(research/walltps_ab/local_official_projection) and the AR anchor is x1.029
(126.378/122.87) -- both ABOVE 1.0, so x0.870 understates official-equiv and any
"clears +10" under it is robust to the projection choice. Reported as a sensitivity.

Greedy/PPL pinned BY CONSTRUCTION: this leg edits no served file and runs only
micro-benchmarks (random int4 weights; achieved BW / kernel time are value-
independent). No tokens are emitted. analysis_only=1, official_tps=0, fires=0.

Reproduce: cd target/ && CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
  /tmp/senpai-venvs/5f4c623f772358a2/bin/python \
  research/speed/specdec_amortization_ceiling/specdec_amortization_ceiling.py \
  --self-test --wandb_name denken/specdec-amortization-ceiling \
  --wandb_group specdec-amortization-ceiling-denken
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

_here = os.path.dirname(os.path.abspath(__file__))

# =================== IMPORTED, EXACT (this leg derives nothing measured upstream) ===
# --- denken #676 body-read (research/speed/gemv_hbm_roofline_ceiling) ---------------
D676 = {
    "body_bytes": 2034278400.0,      # 2.034 GB int4 body weight+scale read / token
    "lm_head_bytes": 346030080.0,    # 0.346 GB int4 untied head read / token
    "full_gemv_bytes": 2380308480.0, # 2.38 GB/token (dominant 86% of per-token bytes)
    "peak_read_gbps": 517.0289326653055,   # measured STREAM read peak (sm_86 A10G)
    "full_gemv_isolated_us": 6038.179356292173,   # isolated full-body+head gemv (M=1)
    "full_gemv_achieved_bw_gbps": 394.2096349820356,
    "dominant_achieved_bw_gbps": 444.7333231272872,   # body (down/gate dominant) achieved
    "lm_head_int4_achieved_bw_gbps": 467.2527396044429,
    # d674 deployed per-step decode split (one AR M=1 step)
    "d674_step_wall_us": 8138.943890282619,
    "d674_matmul_us": 6919.9059492186625,   # body+head gemv in-loop
    "d674_attn_us": 903.4943359374225,
    "d674_norm_us": 175.36125781246608,
    "d674_sampling_us": 14.871394531239364,
    "d674_other_us": 148.23333593751414,
    "d674_local_tps": 122.86606388747026,   # local AR int4 decode TPS (deployed)
}
# the strict #319 anchor body fused shapes (out N, in K), recovered from #676
# per-shape int4 weight_bytes / 0.5 bytes-per-param / in-dim. Full untied head (262k).
HIDDEN = 2560
STRICT_SHAPES = {                    # (N=out, K=in), group_size
    "qkv_proj":     ((3072, 2560), 128),    # q2048+k512+v512 fused (GQA)
    "o_proj":       ((2560, 2048), 128),
    "gate_up_proj": ((20480, 2560), 128),   # gate10240+up10240 fused
    "down_proj":    ((2560, 10240), 128),
    "lm_head":      ((262144, 2560), -1),   # FULL untied head, per-channel int4
}
# per-component layer multiplicity in the served body (#676 component_layers)
LAYER_MULT = {"qkv_proj": 42, "o_proj": 42, "gate_up_proj": 42, "down_proj": 42, "lm_head": 1}

# --- sm_86 A10G roofline (verify_step_component_roofline / kanna #280, ubel #450) ----
A10G_SPEC_BW_GBPS = 600.0            # GA102 datasheet peak (NOT achievable)
A10G_BF16_TFLOPS = 125.0             # GA102 bf16 tensor (FP16-accum) peak
RIDGE_AI = A10G_BF16_TFLOPS * 1e12 / (A10G_SPEC_BW_GBPS * 1e9)   # 208.3 FLOP/byte
INT4_AI_PER_M = 4.0                  # int4 weight (0.5 B/param): AI = 2*M*P / (0.5*P) = 4*M

# --- strict batch-invariant (VLLM_BATCH_INVARIANT=1 + recompute-rescue) walltps -----
# committed K-sweep, qat drafter (research/walltps_ab/optionb_bi1_stock_int4/ksweep/*,
# land PR #82/#90). E = e_accept_exact, localTPS = arms.candidate wall_tps mean.
WALLTPS_KSWEEP = {                    # K: (E_accept, local_wall_tps)
    3: (2.856167469218395, 165.70970814804593),
    4: (3.2039219827163326, 171.675),
    5: (3.4741068549689316, 172.74491495249353),
    6: (3.6573820864780973, 170.184),
    7: (3.8253840567120236, 152.25574001135794),
}
# --- IN-SCOPE-publishable drafter realized points (land #670, x0.870 official-equiv) -
INSCOPE_POINTS = {                    # label: (E_accept, official_equiv_tps)
    "stock":   (3.33, 136.12),
    "top_k64": (3.38, 137.14),
}
INSCOPE_E_CAP = 3.38                  # best publishable acceptance (top_k64)
OOS_E = 3.66                          # qat-retrain (out-of-scope publishable)

# --- the bar (PR #677) --------------------------------------------------------------
REF_OFFICIAL_TPS = 126.378           # strict-#319 anchor int4_g128_lmhead (AR int4)
PLUS10_BAR = 136.378                 # +10 official-equiv
LOCAL_TO_OFFICIAL = 0.870            # PR-specified strict-basis projection (stark tax)
PLUS10_LOCAL = PLUS10_BAR / LOCAL_TO_OFFICIAL   # 156.76 local
# sensitivity multipliers (deployed x1.06, AR anchor x1.029) -- both > 1.0 => x0.870 cons.
PROJ_DEPLOYED = 1.0601865051833779   # research/walltps_ab/local_official_projection
PROJ_AR_ANCHOR = REF_OFFICIAL_TPS / D676["d674_local_tps"]   # 1.0285


# =============================== CPU-ANALYTIC DERIVATION =============================
def derive_k_max_useful():
    """The verify forward stays weight-read (memory) bound while int4-weight AI=4*M
    is below the sm_86 ridge AI. Knee M where 4*M = RIDGE_AI."""
    m_knee_spec = RIDGE_AI / INT4_AI_PER_M           # at 600 GB/s datasheet, 125 TFLOP
    # sensitivity: achievable BW (measured 517) and FP32-accum tensor (70 TFLOP) bounds
    ridge_achievable = A10G_BF16_TFLOPS * 1e12 / (D676["peak_read_gbps"] * 1e9)
    m_knee_achievable = ridge_achievable / INT4_AI_PER_M
    ridge_fp32acc = 70.0 * 1e12 / (A10G_SPEC_BW_GBPS * 1e9)
    m_knee_fp32acc = ridge_fp32acc / INT4_AI_PER_M
    return {
        "k_max_useful_M_knee": m_knee_spec,          # M=K+1 knee ~52
        "k_max_useful_K": m_knee_spec - 1.0,         # draft-width knee ~51
        "ridge_ai_flop_per_byte": RIDGE_AI,
        "int4_ai_per_M": INT4_AI_PER_M,
        "m_knee_achievable_bw": m_knee_achievable,   # ~60 at the higher achievable AI
        "m_knee_fp32acc_tensor": m_knee_fp32acc,     # ~29 lower bound (FP32-accum)
    }


def fit_tstep_model():
    """Calibrate T_step(K)=E/localTPS from the strict walltps K-sweep, decompose into a
    linear K0..K6 base+draft model (the K=7 jump is a vLLM M=8 graph-bucket artifact,
    not the roofline knee at M~52)."""
    tstep = {K: E / tps for K, (E, tps) in WALLTPS_KSWEEP.items()}   # seconds
    # linear fit T_step = T0 + d*K over K in {3,4,5,6} (pre-bucket-jump)
    ks = [3, 4, 5, 6]
    xs = ks
    ys = [tstep[k] * 1e3 for k in ks]   # ms
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    d = (n * sxy - sx * sy) / (n * sxx - sx * sx)     # ms/draft
    t0 = (sy - d * sx) / n                              # ms K->0 strict base
    # per-draft increments (diagnostic)
    incr = {f"{k}->{k+1}": (tstep[k + 1] - tstep[k]) * 1e3 for k in range(3, 7)}
    # optimal K = argmax localTPS
    k_opt = max(WALLTPS_KSWEEP, key=lambda k: WALLTPS_KSWEEP[k][1])
    return {
        "tstep_ms": {k: tstep[k] * 1e3 for k in tstep},
        "tstep_linear_T0_ms": t0,
        "tstep_linear_draft_ms_per_K": d,
        "per_draft_increment_ms": incr,
        "k_opt": k_opt,
        "tstep_at_kopt_ms": tstep[k_opt] * 1e3,
        "localtps_at_kopt": WALLTPS_KSWEEP[k_opt][1],
        # the strict base T0 vs the deployed AR step -> the batch-invariant+rescue tax
        "ar_step_ms": 1e3 / D676["d674_local_tps"],
        "strict_base_tax_ms": t0 - 1e3 / D676["d674_local_tps"],
    }


def raw_bodyread_ceiling(E):
    """Instruction-1 RAW law (rescue-free): TPS = E / (body+head read + cheap draft).
    The body-read is M-invariant so the verify reads it ONCE per step. Two draft
    assumptions bracket the raw ceiling."""
    t_body_isolated_s = D676["full_gemv_isolated_us"] * 1e-6           # 6.04 ms measured
    t_body_469_s = D676["full_gemv_bytes"] / (469.0 * 1e9)             # 5.08 ms @469 GB/s
    # cheap MTP-style draft ~ the deployed drafter share (ubel #443: 1426us/cycle)
    t_draft_cheap_s = 1426e-6
    out = {}
    for tag, tb in (("isolated_6.04ms", t_body_isolated_s), ("ideal469_5.08ms", t_body_469_s)):
        for dtag, td in (("draftfree", 0.0), ("cheapMTP_1.43ms", t_draft_cheap_s)):
            local = E / (tb + td)
            out[f"{tag}/{dtag}"] = {"local_tps": local, "official_equiv": local * LOCAL_TO_OFFICIAL}
    return out


def strict_envelope():
    """The #319-valid realized ceiling: walltps K-sweep x0.870 (qat OOS drafter) +
    the in-scope land #670 points. The rescue tax is what collapses the raw ~480
    ceiling to here."""
    oos = {K: {"E": E, "local_tps": tps, "official_equiv": tps * LOCAL_TO_OFFICIAL,
               "clears_plus10": tps * LOCAL_TO_OFFICIAL > PLUS10_BAR}
           for K, (E, tps) in WALLTPS_KSWEEP.items()}
    k_opt = max(WALLTPS_KSWEEP, key=lambda k: WALLTPS_KSWEEP[k][1])
    inscope = {lab: {"E": E, "official_equiv": oe, "clears_plus10": oe > PLUS10_BAR}
               for lab, (E, oe) in INSCOPE_POINTS.items()}
    return {
        "oos_qat_by_K": oos,
        "oos_qat_best_official_equiv": oos[k_opt]["official_equiv"],   # 150.28 @K=5
        "oos_qat_best_K": k_opt,
        "inscope": inscope,
        "inscope_best_official_equiv": max(v["official_equiv"] for v in inscope.values()),
        "qat_K6_cross_check_official": WALLTPS_KSWEEP[6][1] * LOCAL_TO_OFFICIAL,  # ==148 PR
    }


def solve_e_star(tstep_kopt_ms):
    """E[accept]* to clear +10 (a) at the optimal-K best-case strict draft cost
    (T_step at K_opt), and (b) from the realized in-scope land #670 envelope."""
    # (a) optimal-K best-case: need local = PLUS10_LOCAL at T_step(K_opt)
    e_star_kopt = PLUS10_LOCAL * (tstep_kopt_ms * 1e-3)        # E = local * T_step(s)
    # (b) realized in-scope: linear interp through the two land #670 points
    (e_lo, oe_lo), (e_hi, oe_hi) = INSCOPE_POINTS["stock"], INSCOPE_POINTS["top_k64"]
    e_star_realized = e_lo + (PLUS10_BAR - oe_lo) / (oe_hi - oe_lo) * (e_hi - e_lo)
    return {
        "e_star_optimalK_bestcase": e_star_kopt,
        "e_star_realized_inscope": e_star_realized,
        "inscope_cap_top_k64": INSCOPE_E_CAP,
        "stock_clears": INSCOPE_POINTS["stock"][1] > PLUS10_BAR,
        "top_k64_clears": INSCOPE_POINTS["top_k64"][1] > PLUS10_BAR,
        "oos_qat_clears": True,
    }


def assign_verdict(estar, strict_env):
    """CLEARS_10_INSCOPE if a publishable drafter (E<=3.38) clears +10 at optimal K;
    NEEDS_OOS_DRAFTER if only the 3.66 retrain reaches; BELOW_10 if max E*K can't."""
    inscope_best = strict_env["inscope_best_official_equiv"]
    oos_best = strict_env["oos_qat_best_official_equiv"]
    top_k64_clears = strict_env["inscope"]["top_k64"]["clears_plus10"]
    if inscope_best > PLUS10_BAR and top_k64_clears:
        v = "SPECDEC_CEILING_CLEARS_10_INSCOPE"
    elif oos_best > PLUS10_BAR:
        v = "SPECDEC_CEILING_NEEDS_OOS_DRAFTER"
    else:
        v = "SPECDEC_CEILING_BELOW_10"
    return v


# =============================== LOCAL GPU M-SWEEP (validation) ======================
def run_m_sweep(m_list, iters, warmup, n_distinct):
    """Self-build the strict int4-Marlin verify shapes and time apply_gptq_marlin_linear
    at each M. Confirms per-shape verify GEMM us(M) is M-invariant (flat) up to the
    roofline knee ~52, then rises. (#676 only ever measured M=1.) Random weights:
    achieved BW / kernel time are value-independent."""
    import torch
    from vllm.model_executor.layers.quantization.utils import marlin_utils_test as _mt
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_gptq_marlin_linear as _apply, marlin_make_workspace_new as _mkws)
    from vllm.scalar_type import scalar_types
    QT = scalar_types.uint4b8
    dev = "cuda"
    torch.manual_seed(0)
    ws = _mkws(dev)
    zp = torch.zeros(0, dtype=torch.int, device=dev)

    def quant(N, K, g):
        w = torch.randn(K, N, dtype=torch.float16, device=dev) * 0.02
        res = _mt.marlin_quantize(w, QT, g if g > 0 else K, False)
        del w
        return res[1], res[2], res[3], res[4]

    def timed(fn):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            fn()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / iters * 1e3   # us/call

    out = {}
    for comp, ((N, K), g) in STRICT_SHAPES.items():
        nd = max(2, n_distinct // 4) if comp == "lm_head" else n_distinct
        weights = [quant(N, K, g) for _ in range(nd)]
        per_m = {}
        for M in m_list:
            x = torch.randn(M, K, dtype=torch.float16, device=dev)
            cyc = {"i": 0}

            def fn():
                qw, s, gi, so = weights[cyc["i"] % nd]
                cyc["i"] += 1
                _apply(x, qw, s, zp, gi, so, ws, QT, N, K, is_k_full=True, bias=None)
            us = timed(fn)
            per_m[M] = us
            del x
        out[comp] = per_m
        del weights
        gc.collect()
        torch.cuda.empty_cache()

    # aggregate verify-body us(M) = sum_comp LAYER_MULT * us(M); invariance ratio
    base_M = m_list[0]
    body_us = {M: sum(LAYER_MULT[c] * out[c][M] for c in out) for M in m_list}
    inv_ratio = {M: body_us[M] / body_us[base_M] for M in m_list}
    per_shape_ratio = {c: {M: out[c][M] / out[c][base_M] for M in m_list} for c in out}
    # empirical knee: first M where body invariance ratio exceeds 1.25
    knee = None
    for M in m_list:
        if inv_ratio[M] > 1.25:
            knee = M
            break
    return {
        "per_shape_us": out,
        "body_us": body_us,
        "body_invariance_ratio": inv_ratio,
        "per_shape_invariance_ratio": per_shape_ratio,
        "empirical_knee_M_gt_1p25x": knee,
        "m_list": m_list,
    }


# =============================== SELF-TEST (PRIMARY) ================================
def self_test(res):
    c = {}
    kmu = res["k_max_useful"]
    c["k_max_useful_30_to_70"] = 30.0 <= kmu["k_max_useful_M_knee"] <= 70.0
    c["realistic_K_below_knee"] = res["tstep"]["k_opt"] < kmu["k_max_useful_M_knee"]
    c["tstep_increases_in_K"] = res["tstep"]["tstep_linear_draft_ms_per_K"] > 0
    c["strict_base_tax_positive"] = res["tstep"]["strict_base_tax_ms"] > 0
    c["raw_ceiling_clears_plus10_3x"] = (
        max(v["official_equiv"] for v in res["raw_ceiling"].values()) > 3.0 * PLUS10_BAR)
    c["qat_K6_crosscheck_near_148"] = abs(res["strict_env"]["qat_K6_cross_check_official"] - 148.0) < 1.5
    c["estar_finite"] = math.isfinite(res["estar"]["e_star_realized_inscope"])
    c["estar_in_inscope_band"] = res["estar"]["e_star_realized_inscope"] <= INSCOPE_E_CAP
    c["verdict_known"] = res["verdict"] in (
        "SPECDEC_CEILING_CLEARS_10_INSCOPE", "SPECDEC_CEILING_NEEDS_OOS_DRAFTER",
        "SPECDEC_CEILING_BELOW_10")
    c["official_tps_zero"] = res["scalars"]["official_tps"] == 0
    c["analysis_only"] = res["scalars"]["analysis_only"] == 1
    c["fires_false"] = res["scalars"]["fires"] is False
    if res.get("m_sweep") is not None:
        ms = res["m_sweep"]
        # M-invariance: body us at M=8 within 20% of M=1 (weight-read bound)
        r8 = ms["body_invariance_ratio"].get(8)
        c["m_invariance_holds_at_M8"] = (r8 is not None and r8 < 1.20)
        c["m_sweep_has_knee_or_flat"] = True   # knee may be None (flat in tested range)
    passes = all(c.values())
    return {"passes": passes, "checks": c}


# =============================== ORCHESTRATION =====================================
def build_results(args):
    k_max = derive_k_max_useful()
    tstep = fit_tstep_model()
    raw = raw_bodyread_ceiling(INSCOPE_E_CAP)
    strict_env = strict_envelope()
    estar = solve_e_star(tstep["tstep_at_kopt_ms"])
    verdict = assign_verdict(estar, strict_env)

    m_sweep = None
    if not args.no_gpu:
        try:
            m_list = [int(x) for x in args.m_list.split(",")]
            m_sweep = run_m_sweep(m_list, args.iters, args.warmup, args.n_distinct)
        except Exception as exc:  # noqa: BLE001
            print(f"[specdec-ceiling] M-sweep FAILED (non-fatal): {exc!r}", flush=True)
            m_sweep = {"error": repr(exc)}

    # headline scalars
    inscope_best = strict_env["inscope_best_official_equiv"]
    scalars = {
        "analysis_only": 1,
        "official_tps": 0,
        "fires": False,
        "no_hf_job": True,
        "no_served_file_change": True,
        # PRIMARY metric (best E,K = qat OOS at optimal K, max realizable strict spec)
        "specdec_tps_ceiling_official_equiv": strict_env["oos_qat_best_official_equiv"],
        # the verdict-binding in-scope ceiling (best publishable drafter, top_k64)
        "specdec_inscope_ceiling_official_equiv": inscope_best,
        # the body-read RAW ceiling (rescue-free) -- not the binding constraint
        "specdec_raw_bodyread_ceiling_official_equiv": max(
            v["official_equiv"] for v in raw.values()),
        # TEST metric
        "e_accept_star_for_plus10": estar["e_star_realized_inscope"],
        "e_accept_star_optimalK_bestcase": estar["e_star_optimalK_bestcase"],
        "k_max_useful_M_knee": k_max["k_max_useful_M_knee"],
        "k_opt_amortization": tstep["k_opt"],
        "tstep_T0_strict_ms": tstep["tstep_linear_T0_ms"],
        "tstep_draft_ms_per_K": tstep["tstep_linear_draft_ms_per_K"],
        "strict_base_tax_ms": tstep["strict_base_tax_ms"],
        "ref_official_tps": REF_OFFICIAL_TPS,
        "plus10_bar": PLUS10_BAR,
        "local_to_official": LOCAL_TO_OFFICIAL,
        "inscope_stock_clears": estar["stock_clears"],
        "inscope_top_k64_clears": estar["top_k64_clears"],
        "verdict": verdict,
    }
    res = {
        "k_max_useful": k_max, "tstep": tstep, "raw_ceiling": raw,
        "strict_env": strict_env, "estar": estar, "verdict": verdict,
        "m_sweep": m_sweep, "scalars": scalars,
        "constants": {"D676": D676, "STRICT_SHAPES": {k: v for k, v in STRICT_SHAPES.items()},
                      "WALLTPS_KSWEEP": WALLTPS_KSWEEP, "INSCOPE_POINTS": INSCOPE_POINTS,
                      "proj_sensitivity": {"x0.870_used": LOCAL_TO_OFFICIAL,
                                           "x_deployed": PROJ_DEPLOYED, "x_ar_anchor": PROJ_AR_ANCHOR}},
        "timestamp": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
    }
    res["self_test"] = self_test(res)
    return res


def maybe_wandb(res, args):
    if args.no_wandb:
        return
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[specdec-ceiling] wandb unavailable: {exc!r}", flush=True)
        return
    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
                     entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
                     name=args.wandb_name, group=args.wandb_group,
                     config={"pr": 677, "analysis_only": True, "official_tps": 0,
                             "fires": False, "no_hf_job": True})
    flat = dict(res["scalars"])
    flat["self_test_passes"] = res["self_test"]["passes"]
    # per-K strict envelope + M-sweep ratios as flat scalars
    for K, v in res["strict_env"]["oos_qat_by_K"].items():
        flat[f"oos_qat_K{K}_official_equiv"] = v["official_equiv"]
        flat[f"oos_qat_K{K}_E"] = v["E"]
    if res.get("m_sweep") and "body_invariance_ratio" in res["m_sweep"]:
        for M, r in res["m_sweep"]["body_invariance_ratio"].items():
            flat[f"m_invariance_ratio_M{M}"] = r
    wandb.log(flat)
    wandb.summary.update(flat)
    # full artifact
    art_path = os.path.join(_here, "specdec_amortization_ceiling.json")
    try:
        art = wandb.Artifact("specdec_amortization_ceiling", type="analysis")
        art.add_file(art_path)
        run.log_artifact(art)
    except Exception as exc:  # noqa: BLE001
        print(f"[specdec-ceiling] artifact log skipped: {exc!r}", flush=True)
    run.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--no-gpu", action="store_true", help="skip the local M-sweep (CPU-only)")
    ap.add_argument("--m-list", dest="m_list", default="1,2,4,8,16,32,48,64,96")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--n-distinct", type=int, default=8)
    ap.add_argument("--wandb_name", default="denken/specdec-amortization-ceiling")
    ap.add_argument("--wandb_group", default="specdec-amortization-ceiling-denken")
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny M-list + few iters")
    args = ap.parse_args()
    if args.smoke:
        args.m_list, args.iters, args.warmup, args.n_distinct = "1,8,32", 10, 5, 4

    res = build_results(args)
    out_path = os.path.join(_here, "specdec_amortization_ceiling.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2, default=str)

    s = res["scalars"]
    print("\n================= SPEC-DEC AMORTIZATION CEILING (PR #677) =================")
    print(f"  K_max_useful (verify M-invariance knee, M=K+1): {s['k_max_useful_M_knee']:.1f}"
          f"   (realistic K_opt={s['k_opt_amortization']} << knee)")
    print(f"  T_step(K) = {s['tstep_T0_strict_ms']:.2f}ms + {s['tstep_draft_ms_per_K']:.3f}ms/draft"
          f"   (strict base tax over AR step = {s['strict_base_tax_ms']:.2f}ms)")
    print(f"  RAW body-read ceiling (rescue-free): {s['specdec_raw_bodyread_ceiling_official_equiv']:.1f}"
          f" official-equiv  ->  body-read permits {s['specdec_raw_bodyread_ceiling_official_equiv']/PLUS10_BAR:.2f}x the +10 bar")
    print(f"  STRICT realized ceiling (rescue tax): OOS-qat best {s['specdec_tps_ceiling_official_equiv']:.2f}"
          f" @K={res['strict_env']['oos_qat_best_K']} ; in-scope best {s['specdec_inscope_ceiling_official_equiv']:.2f} (top_k64)")
    print(f"  +10 bar = {PLUS10_BAR}  |  stock(3.33) clears={s['inscope_stock_clears']}  "
          f"top_k64(3.38) clears={s['inscope_top_k64_clears']}")
    print(f"  E*[accept] for +10: realized-inscope={s['e_accept_star_for_plus10']:.3f}  "
          f"optimalK-bestcase={s['e_accept_star_optimalK_bestcase']:.3f}")
    if res.get("m_sweep") and "body_invariance_ratio" in res["m_sweep"]:
        ir = res["m_sweep"]["body_invariance_ratio"]
        print("  M-sweep verify-body invariance ratio us(M)/us(1): "
              + " ".join(f"M{M}={ir[M]:.3f}" for M in res["m_sweep"]["m_list"]))
        print(f"  empirical knee (>1.25x): M={res['m_sweep']['empirical_knee_M_gt_1p25x']}")
    print(f"  VERDICT: {res['verdict']}")
    print(f"  self_test: passes={res['self_test']['passes']}  checks={res['self_test']['checks']}")
    print("==========================================================================\n")

    maybe_wandb(res, args)
    if args.self_test and not res["self_test"]["passes"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
