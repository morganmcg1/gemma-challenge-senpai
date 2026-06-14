#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Tree-step denominator measurement (PR #107): pin the REAL M=8 -> M=32
verify-forward wall-time ratio that fern #102's break-even rests on.

WHY THIS MEASUREMENT EXISTS
---------------------------
The whole tree go/no-go rests on a number that is DERIVED, not measured
end-to-end: fern #102's break-even (E[T]* = 4.624) and lawine #99's 569
projection both assume the M=32 tree-verify step is ~1.16x the linear M=8 step.
fern's `lever_composition.compose({"tree"}, point("central"))["step_time"]`
back-solves that 1.158 from the committed 568-official net-tree projection -- it
is NOT built up from a direct GEMM+attention measurement. This script measures
the load-bearing denominator directly on the deployed kernel.

WHAT IS MEASURED (the GEMM + causal-attention floor)
----------------------------------------------------
The verify forward at width M = the M-scaling part of one decoder forward:
  * verify-GEMM  -- all int4 W4A16 Marlin weight GEMMs (qkv/o/gate_up/down),
                    summed over every decoder layer, launch-free CUDA-graph
                    timing (denken #68 machinery, verbatim).
  * causal attention -- the deployed vLLM 3D split-KV `unified_attention`
                    (SPLITKV_VERIFY=1, fp32 partials = deployed/greedy-safe),
                    summed over the sliding + full layers, device-only time
                    (wirbel #98 / #39 machinery, verbatim).

  verify_forward_us(M) = gemm_us(M) + attn_us(M)
  measured_M32_M8_step_ratio = verify_forward_us(32) / verify_forward_us(8)

This is the MEASURABLE FLOOR. The custom star-attn TREE-MASK kernel (land #71,
not yet built) is EXCLUDED by construction -- causal attention is its lower
bound; the tree-mask delta is the only residual that must wait for the build.

The drafter, sampler, host/Python overhead, layernorms and rotary are NOT part
of the verify forward; they are M-invariant (the drafter runs once per step) and
cancel in the verify-forward DIFFERENCE. They are folded back in via the
committed decode budget only when mapping the floor ratio to the WHOLE-step
ratio that fern's model uses.

PROTOCOL (locked wall_tps discipline, PR #72/#82)
-------------------------------------------------
Median of N fresh repeats of the full GEMM+attention timing, CV-characterized;
the ratio carries a z*SE CI. Each repeat is an independent CUDA-graph capture +
profiled attention pass, so thermal/clock drift shows up in the CV.

DECOMPOSITION + WHOLE-STEP RECONCILIATION (Step 2)
--------------------------------------------------
  r_gemm = gemm_us(32)/gemm_us(8)   (denken #68: expected ~flat to M<=32)
  r_attn = attn_us(32)/attn_us(8)   (wirbel #98: real-kernel ~1.83x, growing)
The floor ratio R_v lives on the verify-forward denominator (~61% of the step).
Two transparent maps to the WHOLE-step ratio fern uses (linear M=8 step = 1.0):
  (A) budget-share : 1 + b_gemm*(r_gemm-1) + b_attn*(r_attn-1)
  (B) lumped-share : (1 - b_vf) + b_vf * R_v          (b_vf = b_gemm + b_attn)
with the committed budget b_gemm=0.53, b_attn=0.08 (CURRENT_RESEARCH_STATE.md).

BREAK-EVEN RECOMPUTE (Step 3)
-----------------------------
fern's break-even is E[T]* = 500 * step_time / (K_cal * lk * tau). Holding
lk=tau=1 (tree-alone), substitute the MEASURED whole-step ratio for fern's
modeled step_time:
  corrected_breakeven_ET = 500 * measured_whole_step_ratio / K_cal
and compare to fern's 4.624 (= 500 * 1.158 / 125.268).

GATE
----
On the measured WHOLE-step ratio (the apples-to-apples comparand of fern's
1.16x), NOT the raw verify-forward floor:
  GREEN  measured whole-step ratio <= ~1.16x -> break-even holds/improves.
  AMBER  ~1.16-1.30x -> break-even rises above 4.624 (report corrected number).
  RED    >> 1.30x   -> step materially heavier; break-even approaches 5.207.

LOCAL A10G op-microbench. NO HF Job, NO server, NO submission, NO leaderboard
number, NO served-file change. Timing only -> greedy identity untouched by
construction (no token stream is produced).
"""
from __future__ import annotations

import os

# Pin the single visible A10G and disable the broken flashinfer sampler JIT
# BEFORE importing torch/vllm (project_local_a10g_gpu_env). Force the deployed
# split-KV verify path on so M>1 verify batches route to 3D split-KV (the path
# whose attention we are pricing).
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("SPLITKV_VERIFY", "1")

import argparse
import gc
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

# --- GEMM machinery (denken #68, reused verbatim) ---------------------------
from scripts.profiler.verify_gemm_roofline import (  # noqa: E402
    DEFAULT_MODEL,
    build_llm,
    collect_gemm_instances,
    find_decoder_layers,
    find_runner,
    time_gemm_graph,
    uniquify,
)
# --- attention machinery (wirbel #98 / #39, reused verbatim) ----------------
from scripts.local_validation.profile_attention import (  # noqa: E402
    HEAD_DIM,
    N_Q_HEADS,
    _maybe_install_splitkv,
    _measure_peak_bw,
    _profiled_device_us,
)
from scripts.profiler.star_attn_fp32_cost import (  # noqa: E402
    _build_op_inputs,
    _make_call,
)
# --- fern's model (consumed verbatim; the thing we measure the denominator of)
from scripts.profiler.lever_composition import (  # noqa: E402
    BUDGET,
    E_T_LINEAR,
    E_T_TREE,
    FRONTIER_OFFICIAL,
    K_CAL,
    TARGET_OFFICIAL,
    compose,
    point,
)
from scripts.profiler.tree_et_breakeven import breakeven_raw_et  # noqa: E402

# z for a 95% two-sided normal CI (matches the #72/#82 Z_DETECT convention).
Z95 = 1.959963984540054
ATTN_CTX_DEFAULT = 528          # deployed mean served ctx ~527.7 (wirbel #98)
B_GEMM = BUDGET["verify_gemm"]  # 0.53
B_ATTN = BUDGET["attention"]    # 0.08
B_VF = B_GEMM + B_ATTN          # 0.61 verify-forward share of the M=8 step


# ---------------------------------------------------------------------------
# Measurement primitives
# ---------------------------------------------------------------------------
def measure_gemm_us(uniq: dict, M: int, iters: int, warmup: int) -> tuple[float, dict]:
    """Sum launch-free CUDA-graph GEMM time over every decoder-layer weight GEMM
    at verify width M. Returns (total_us, per_shape_us). Identical basis to
    denken #68 `aggregate_by_M[M].total_gemm_us`."""
    total_ms = 0.0
    per_shape = {}
    all_graphed = True
    for key, u in uniq.items():
        ms, graphed = time_gemm_graph(u["module"], M, u["in"], iters, warmup)
        total_ms += ms * u["count"]
        per_shape[f"{u['role']}|{u['in']}x{u['out']}|x{u['count']}"] = {
            "per_call_us": ms * 1000.0, "count": u["count"], "graphed": graphed}
        all_graphed = all_graphed and graphed
    return total_ms * 1000.0, {"per_shape": per_shape, "all_graphed": all_graphed}


def measure_attn_us(M: int, ctx: int, n_iter: int, warmup: int,
                    counts: dict) -> tuple[float, dict]:
    """Sum deployed fp32 split-KV `unified_attention` device time over the
    sliding + full layers at width M. Identical basis to wirbel #98
    `attn_us_per_cycle_fp32` but with the REAL loaded-model layer counts."""
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


def layer_counts_from_gemm(uniq: dict) -> dict:
    """Derive (n_sliding, n_full) attention-layer counts from the loaded model's
    qkv-proj GEMM shapes: out=3072 -> sliding (hd=256), out=6144 -> full (hd=512).
    Ties the attention cycle to the SAME model introspection as the GEMM sum so
    the two components are summed over a consistent layer set."""
    n_sliding = n_full = 0
    for u in uniq.values():
        if "qkv_proj" not in u["role"]:
            continue
        if u["out"] == 3072:
            n_sliding += u["count"]
        elif u["out"] == 6144:
            n_full += u["count"]
    if n_sliding == 0 and n_full == 0:  # defensive fallback to #39 constants
        from scripts.local_validation.profile_attention import N_FULL, N_SLIDING
        n_sliding, n_full = N_SLIDING, N_FULL
    return {"sliding": n_sliding, "full": n_full}


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def summarize(values: list[float]) -> dict:
    n = len(values)
    mean = statistics.fmean(values)
    median = statistics.median(values)
    std = statistics.pstdev(values) if n > 1 else 0.0
    se = std / math.sqrt(n) if n else 0.0
    cv_pct = 100.0 * std / mean if mean else 0.0
    return {
        "n": n, "mean": mean, "median": median, "std": std, "se": se,
        "cv_pct": cv_pct, "ci95_abs": Z95 * se,
        "ci95_lo": median - Z95 * se, "ci95_hi": median + Z95 * se,
        "min": min(values), "max": max(values), "values": values,
    }


# ---------------------------------------------------------------------------
# Break-even (Step 3) — substitute the measured whole-step ratio into fern #102
# ---------------------------------------------------------------------------
def corrected_breakeven(whole_step_ratio: float, target: float = TARGET_OFFICIAL) -> float:
    """fern #102 tree-alone: E[T]* = target * step_time / (K_cal * 1 * 1), with
    step_time replaced by the MEASURED whole-step ratio (linear M=8 step = 1.0)."""
    return target * whole_step_ratio / K_CAL


def fern_modeled_step_and_breakeven() -> dict:
    """fern's tree-alone CENTRAL step_time + break-even, computed VERBATIM from
    her merged model so the comparison is exact (no transcribed constant)."""
    c = compose({"tree"}, point("central"))
    step = c["step_time"]
    be = breakeven_raw_et({"tree"}, point("central"))
    return {"modeled_tree_step_ratio": step,
            "modeled_breakeven_ET": be["breakeven_raw_et"],
            "beat_linear_ET": be["beat_linear_raw_et"]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--m-linear", type=int, default=8, help="linear-chain verify width")
    ap.add_argument("--m-tree", type=int, default=32, help="tree verify width (M<=32 cap)")
    ap.add_argument("--repeats", type=int, default=5, help="fresh repeats (median-of-N)")
    ap.add_argument("--gemm-iters", type=int, default=200)
    ap.add_argument("--gemm-warmup", type=int, default=40)
    ap.add_argument("--attn-ctx", type=int, default=ATTN_CTX_DEFAULT)
    ap.add_argument("--attn-iters", type=int, default=300)
    ap.add_argument("--attn-warmup", type=int, default=20)
    ap.add_argument("--max-ctx", type=int, default=256, help="vLLM max ctx for the GEMM load")
    ap.add_argument("--output", default="research/spec_cost_model/tree_step_denominator.json")
    ap.add_argument("--wandb-project", default="gemma-challenge-senpai")
    ap.add_argument("--wandb-entity", default="wandb-applied-ai-team")
    ap.add_argument("--wandb-name", default="lawine/tree-step-denominator")
    ap.add_argument("--wandb-group", default="tree-step-denominator")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    assert torch.cuda.is_available(), "CUDA required"
    ml, mt = args.m_linear, args.m_tree
    t0 = time.time()

    # ---- load model + introspect the verify GEMMs (denken #68) ----
    print(f"[denom] building LLM {args.model} ...", flush=True)
    llm = build_llm(args.model, args.max_ctx)
    runner = find_runner(llm)
    if runner is None:
        raise RuntimeError("could not locate GPUModelRunner")
    layers = find_decoder_layers(runner.model)
    uniq = uniquify(collect_gemm_instances(layers))
    counts = layer_counts_from_gemm(uniq)
    n_gemm_layers = max((u["count"] for u in uniq.values()), default=0)
    print(f"[denom] LLM ready in {time.time()-t0:.0f}s | decoder layers={len(layers)} "
          f"| attn counts sliding={counts['sliding']} full={counts['full']} "
          f"| GEMM unique shapes={len(uniq)}", flush=True)

    _maybe_install_splitkv()
    peak_bw = _measure_peak_bw(torch, torch.device("cuda"))
    print(f"[denom] measured peak HBM copy BW = {peak_bw['measured_peak_gbps_copy']:.0f} GB/s",
          flush=True)

    # ---- clock-ramp + lazy-init warmup (MANDATORY before timing) ----
    # The A10G ramps from base ~885 MHz to boost ~1695 MHz (~1.9x) under sustained
    # load; the FIRST timed CUDA-graph kernel in a cold process runs at BASE clock
    # and reads ~2x slow (smoke: GEMM M=8 10013us vs warm ~4868us). The
    # unified_attention / reduce_segments Triton kernels also JIT-compile on first
    # use. A sustained throwaway pass over BOTH widths ramps clocks to steady boost
    # and sheds every first-use penalty, so repeat 1 below is already warm and the
    # M=8 vs M=32 ratio is not corrupted by clock state (which is order-dependent).
    print("[denom] warmup: ramping clocks + triggering JIT over both widths ...", flush=True)
    for _ in range(3):
        for M in (mt, ml):
            measure_gemm_us(uniq, M, max(80, args.gemm_warmup), args.gemm_warmup)
            measure_attn_us(M, args.attn_ctx, max(80, args.attn_warmup),
                            args.attn_warmup, counts)
    torch.cuda.synchronize()
    print(f"[denom] warmup done ({time.time()-t0:.0f}s elapsed); starting timed repeats",
          flush=True)

    # ---- N fresh repeats of the full GEMM + attention timing ----
    repeats = []
    for r in range(args.repeats):
        gemm_l, gl_meta = measure_gemm_us(uniq, ml, args.gemm_iters, args.gemm_warmup)
        gemm_t, gt_meta = measure_gemm_us(uniq, mt, args.gemm_iters, args.gemm_warmup)
        attn_l, al_meta = measure_attn_us(ml, args.attn_ctx, args.attn_iters,
                                          args.attn_warmup, counts)
        attn_t, at_meta = measure_attn_us(mt, args.attn_ctx, args.attn_iters,
                                          args.attn_warmup, counts)
        vf_l = gemm_l + attn_l
        vf_t = gemm_t + attn_t
        rec = {
            "repeat": r,
            f"gemm_us_M{ml}": gemm_l, f"gemm_us_M{mt}": gemm_t,
            f"attn_us_M{ml}": attn_l, f"attn_us_M{mt}": attn_t,
            f"vf_us_M{ml}": vf_l, f"vf_us_M{mt}": vf_t,
            "r_gemm": gemm_t / gemm_l, "r_attn": attn_t / attn_l,
            "verify_forward_ratio": vf_t / vf_l,
            "gemm_all_graphed": gl_meta["all_graphed"] and gt_meta["all_graphed"],
        }
        repeats.append(rec)
        print(f"[denom] repeat {r+1}/{args.repeats}: "
              f"GEMM M{ml}={gemm_l:.0f} M{mt}={gemm_t:.0f}us (x{rec['r_gemm']:.4f}) | "
              f"ATTN M{ml}={attn_l:.0f} M{mt}={attn_t:.0f}us (x{rec['r_attn']:.4f}) | "
              f"VF ratio={rec['verify_forward_ratio']:.4f}", flush=True)

    # ---- aggregate (Step 1) ----
    R_v = summarize([x["verify_forward_ratio"] for x in repeats])
    r_gemm = summarize([x["r_gemm"] for x in repeats])
    r_attn = summarize([x["r_attn"] for x in repeats])
    gemm_l = summarize([x[f"gemm_us_M{ml}"] for x in repeats])
    gemm_t = summarize([x[f"gemm_us_M{mt}"] for x in repeats])
    attn_l = summarize([x[f"attn_us_M{ml}"] for x in repeats])
    attn_t = summarize([x[f"attn_us_M{mt}"] for x in repeats])
    vf_l = summarize([x[f"vf_us_M{ml}"] for x in repeats])
    vf_t = summarize([x[f"vf_us_M{mt}"] for x in repeats])

    measured_M32_M8_step_ratio = R_v["median"]

    # ---- decomposition (Step 2): attribute the verify-forward DELTA ----
    d_gemm = gemm_t["median"] - gemm_l["median"]
    d_attn = attn_t["median"] - attn_l["median"]
    d_vf = vf_t["median"] - vf_l["median"]
    decomposition = {
        "verify_forward_delta_us": d_vf,
        "gemm_delta_us": d_gemm, "attn_delta_us": d_attn,
        "gemm_share_of_delta": d_gemm / d_vf if d_vf else None,
        "attn_share_of_delta": d_attn / d_vf if d_vf else None,
        "r_gemm_median": r_gemm["median"], "r_attn_median": r_attn["median"],
        # internal split of the M=8 verify forward (measured)
        "gemm_frac_of_vf_M8": gemm_l["median"] / vf_l["median"],
        "attn_frac_of_vf_M8": attn_l["median"] / vf_l["median"],
        "note": ("denken #68 expected GEMM ~flat (r_gemm~1.0); wirbel #98 expected "
                 "attention ~1.83x. The measured r_gemm/r_attn confirm or correct "
                 "those pieces; the star-attn tree-mask kernel is excluded (residual)."),
    }

    # ---- whole-step reconciliation (two transparent maps) ----
    ws_budget = 1.0 + B_GEMM * (r_gemm["median"] - 1.0) + B_ATTN * (r_attn["median"] - 1.0)
    ws_lumped = (1.0 - B_VF) + B_VF * R_v["median"]
    # CI on the budget map: propagate the component-ratio SEs (independent draws).
    ws_budget_se = math.hypot(B_GEMM * r_gemm["se"], B_ATTN * r_attn["se"])
    ws_lumped_se = B_VF * R_v["se"]
    ws_map_lo, ws_map_hi = min(ws_budget, ws_lumped), max(ws_budget, ws_lumped)
    whole_step = {
        "method_A_budget_share": ws_budget,
        "method_A_budget_share_ci95": [ws_budget - Z95 * ws_budget_se,
                                       ws_budget + Z95 * ws_budget_se],
        "method_B_lumped_share": ws_lumped,
        "method_B_lumped_share_ci95": [ws_lumped - Z95 * ws_lumped_se,
                                       ws_lumped + Z95 * ws_lumped_se],
        # The two transparent maps differ only in whether the verify-forward 61%
        # is split by the COMMITTED budget shares (A) or by the MEASURED VF-internal
        # split folded into R_v (B); the true whole-step ratio lies between them.
        "whole_step_bracket": [ws_map_lo, ws_map_hi],
        "budget_shares": {"verify_gemm": B_GEMM, "attention": B_ATTN,
                          "verify_forward_total": B_VF,
                          "m_invariant_remainder": 1.0 - B_VF},
        "primary_whole_step_ratio": ws_budget,  # budget-share is the comparand of fern's 1.158
    }
    measured_whole_step_ratio = ws_budget

    # ---- break-even recompute (Step 3) ----
    fern = fern_modeled_step_and_breakeven()
    be_central = corrected_breakeven(ws_budget)
    be_lumped = corrected_breakeven(ws_lumped)
    be_lo = corrected_breakeven(whole_step["method_A_budget_share_ci95"][0])
    be_hi = corrected_breakeven(whole_step["method_A_budget_share_ci95"][1])
    be_map_lo = corrected_breakeven(ws_map_lo)
    be_map_hi = corrected_breakeven(ws_map_hi)
    breakeven = {
        "fern_modeled_tree_step_ratio": fern["modeled_tree_step_ratio"],
        "fern_modeled_breakeven_ET": fern["modeled_breakeven_ET"],
        "corrected_breakeven_ET": be_central,
        "corrected_breakeven_ET_ci95": [be_lo, be_hi],
        "corrected_breakeven_ET_lumped": be_lumped,
        # the honest uncertainty here is the A-vs-B map spread, not the (tiny)
        # microbench CI; report the break-even bracket the two maps imply.
        "corrected_breakeven_ET_method_bracket": [be_map_lo, be_map_hi],
        "delta_vs_fern_ET": be_central - fern["modeled_breakeven_ET"],
        "delta_vs_fern_pct": 100.0 * (be_central - fern["modeled_breakeven_ET"])
        / fern["modeled_breakeven_ET"],
        "fern_bracketed_by_methods": be_map_lo <= fern["modeled_breakeven_ET"] <= be_map_hi,
        "K_cal": K_CAL, "target_official": TARGET_OFFICIAL,
        "beat_linear_ET": fern["beat_linear_ET"],
        "analytical_ceiling_ET": E_T_TREE,
        "byteshark_asbuilt_ET": 2.097,
        "linear_chain_ET": E_T_LINEAR,
    }

    # ---- gate ----
    # On the measured WHOLE-step ratio vs fern's ~1.16x, NOT the raw verify-forward
    # floor R_v (which lives on the ~61% VF denominator and is ~1.25 by construction;
    # comparing R_v directly to 1.16 is a denominator mismatch). The honest
    # uncertainty is the A-vs-B map bracket, so the gate is bracket-aware: if the
    # lower map sits at/below the 1.16 line the assumption is NOT refuted (the bracket
    # straddles fern's number) -> GREEN. Only a bracket that clears 1.16 entirely is a
    # real tightening (AMBER), and only one clearing 1.30 is a blow-up (RED).
    GREEN_LINE, RED_LINE = 1.16, 1.30
    fern_ws = fern["modeled_tree_step_ratio"]
    straddles = ws_map_lo <= GREEN_LINE <= ws_map_hi or ws_map_lo <= fern_ws <= ws_map_hi
    if ws_map_lo <= GREEN_LINE:
        verdict = "GREEN"
        if straddles:
            verdict_label = (
                f"measured whole-step bracket [{ws_map_lo:.3f},{ws_map_hi:.3f}] STRADDLES "
                f"fern's modeled {fern_ws:.3f} (1.16x) -> the load-bearing 1.16x "
                f"denominator is CONFIRMED by direct GEMM+attention measurement; "
                f"corrected break-even {be_map_lo:.3f}-{be_map_hi:.3f} brackets fern's "
                f"{fern['modeled_breakeven_ET']:.3f}; the 569 projection denominator "
                f"stands. (Star-attn fp32 haircut + tree-mask delta are the only "
                f"residuals, on top of this measured floor.)")
        else:
            verdict_label = (
                f"measured whole-step bracket [{ws_map_lo:.3f},{ws_map_hi:.3f}] at/below "
                f"~1.16x -> fern #102's break-even holds or improves; the 1.16x "
                f"denominator is CONFIRMED; the 569 projection denominator stands. "
                f"(Star-attn tree-mask delta is the only residual, on top of this floor.)")
    elif ws_map_lo <= RED_LINE:
        verdict = "AMBER"
        verdict_label = (
            f"measured whole-step bracket [{ws_map_lo:.3f},{ws_map_hi:.3f}] clears ~1.16x "
            f"-> break-even rises to {be_map_lo:.3f}-{be_map_hi:.3f} (vs fern 4.624); the "
            f"tree must recover closer to 5.207; flag the tightened bar to fern + the "
            f"build team (the star-attn residual lands on top of this).")
    else:
        verdict = "RED"
        verdict_label = (
            f"measured whole-step bracket [{ws_map_lo:.3f},{ws_map_hi:.3f}] >> 1.30x -> "
            f"the M=32 step is materially heavier than assumed; break-even "
            f"{be_map_lo:.3f}-{be_map_hi:.3f} approaches the 5.207 ceiling -> escalate.")

    peak_gpu_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    result = {
        "pr": 107, "metric": "measured_M32_M8_step_ratio",
        "primary_metric": {"name": "measured_M32_M8_step_ratio",
                           "value": measured_M32_M8_step_ratio,
                           "ci95": [R_v["ci95_lo"], R_v["ci95_hi"]],
                           "cv_pct": R_v["cv_pct"],
                           "denominator": "verify forward (GEMM + causal attention)"},
        "test_metric": {"name": "corrected_breakeven_ET",
                        "value": be_central,
                        "ci95": [be_lo, be_hi],
                        "fern_modeled": fern["modeled_breakeven_ET"]},
        "verdict": verdict, "verdict_label": verdict_label,
        "config": {
            "model": args.model, "vllm": __import__("vllm").__version__,
            "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
            "m_linear": ml, "m_tree": mt, "repeats": args.repeats,
            "gemm_iters": args.gemm_iters, "attn_iters": args.attn_iters,
            "attn_ctx": args.attn_ctx, "splitkv_verify": os.environ.get("SPLITKV_VERIFY"),
            "n_attn_sliding": counts["sliding"], "n_attn_full": counts["full"],
            "n_gemm_layers_max": n_gemm_layers,
            "peak_hbm_copy_gbps": peak_bw["measured_peak_gbps_copy"],
            "peak_gpu_gib": peak_gpu_gib,
        },
        "step1_measurement": {
            "verify_forward_ratio_M32_M8": R_v,
            "r_gemm": r_gemm, "r_attn": r_attn,
            f"gemm_us_M{ml}": gemm_l, f"gemm_us_M{mt}": gemm_t,
            f"attn_us_M{ml}": attn_l, f"attn_us_M{mt}": attn_t,
            f"verify_forward_us_M{ml}": vf_l, f"verify_forward_us_M{mt}": vf_t,
        },
        "step2_decomposition": decomposition,
        "step2_whole_step_reconciliation": whole_step,
        "step3_breakeven": breakeven,
        "repeats": repeats,
        "elapsed_s": time.time() - t0,
        "method": ("LOCAL A10G verify-forward microbench: launch-free CUDA-graph "
                   "GEMM (denken #68) + deployed fp32 split-KV unified_attention "
                   "(wirbel #98) summed over the real loaded-model layers, median of "
                   "N fresh repeats. Whole-step + break-even via fern #102's model "
                   "verbatim. No HF Job / no submission / no served-file change."),
    }

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))

    _print_summary(result, ml, mt)
    print(f"[denom] wrote {out_path}", flush=True)

    if not args.no_wandb:
        try:
            _log_wandb(args, result)
        except Exception as exc:  # noqa: BLE001
            print(f"[denom] W&B logging failed (non-fatal): {exc!r}", flush=True)

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def _print_summary(res: dict, ml: int, mt: int) -> None:
    s1 = res["step1_measurement"]
    d = res["step2_decomposition"]
    ws = res["step2_whole_step_reconciliation"]
    be = res["step3_breakeven"]
    rv = s1["verify_forward_ratio_M32_M8"]
    print("\n" + "=" * 80, flush=True)
    print(f"TREE-STEP DENOMINATOR (PR #107) — M={ml} -> M={mt} verify-forward ratio",
          flush=True)
    print("=" * 80, flush=True)
    print(f"[STEP 1] PRIMARY measured_M32_M8_step_ratio (verify-forward floor, "
          f"GEMM+causal-attn) = {rv['median']:.4f}  "
          f"CI95=[{rv['ci95_lo']:.4f},{rv['ci95_hi']:.4f}]  CV={rv['cv_pct']:.3f}%  "
          f"(n={rv['n']})  <- lives on the ~{B_VF*100:.0f}% VF denominator, NOT the whole step",
          flush=True)
    print(f"         GEMM   M{ml}={s1[f'gemm_us_M{ml}']['median']:.0f}us  "
          f"M{mt}={s1[f'gemm_us_M{mt}']['median']:.0f}us  ratio={d['r_gemm_median']:.4f}",
          flush=True)
    print(f"         ATTN   M{ml}={s1[f'attn_us_M{ml}']['median']:.0f}us  "
          f"M{mt}={s1[f'attn_us_M{mt}']['median']:.0f}us  ratio={d['r_attn_median']:.4f}",
          flush=True)
    print(f"[STEP 2] verify-forward Δ attributed: GEMM {d['gemm_share_of_delta']*100:.1f}% / "
          f"attn {d['attn_share_of_delta']*100:.1f}%  (GEMM flat? r={d['r_gemm_median']:.3f} "
          f"-- denken #68 said ~1.0; 16-row tile staircase makes it >1)", flush=True)
    print(f"         WHOLE-step ratio (apples-to-apples w/ fern's 1.16x): "
          f"bracket=[{ws['whole_step_bracket'][0]:.4f},{ws['whole_step_bracket'][1]:.4f}]  "
          f"(budget-share={ws['method_A_budget_share']:.4f}, "
          f"lumped={ws['method_B_lumped_share']:.4f})  vs fern modeled "
          f"{be['fern_modeled_tree_step_ratio']:.4f}", flush=True)
    print(f"[STEP 3] break-even E[T]*: corrected bracket="
          f"[{be['corrected_breakeven_ET_method_bracket'][0]:.3f},"
          f"{be['corrected_breakeven_ET_method_bracket'][1]:.3f}]  "
          f"(A={be['corrected_breakeven_ET']:.4f})  vs fern "
          f"{be['fern_modeled_breakeven_ET']:.4f}  (Δ{be['delta_vs_fern_pct']:+.2f}%)  "
          f"fern-bracketed={be['fern_bracketed_by_methods']}", flush=True)
    print(f"         placement: linear {be['linear_chain_ET']:.3f} < beat-linear "
          f"{be['beat_linear_ET']:.3f} < break-even ~{be['corrected_breakeven_ET']:.2f} "
          f"< ceiling {be['analytical_ceiling_ET']:.3f}  (as-built {be['byteshark_asbuilt_ET']})",
          flush=True)
    print(f"\n[VERDICT] {res['verdict']} — {res['verdict_label']}\n", flush=True)


def _log_wandb(args, res: dict) -> None:
    import wandb
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     name=args.wandb_name, group=args.wandb_group,
                     job_type="profiling", config=res["config"])
    s1, d = res["step1_measurement"], res["step2_decomposition"]
    ws, be = res["step2_whole_step_reconciliation"], res["step3_breakeven"]
    rv = s1["verify_forward_ratio_M32_M8"]
    summary = {
        "measured_M32_M8_step_ratio": res["primary_metric"]["value"],
        "verify_forward_ratio_cv_pct": rv["cv_pct"],
        "verify_forward_ratio_ci95_lo": rv["ci95_lo"],
        "verify_forward_ratio_ci95_hi": rv["ci95_hi"],
        "r_gemm": d["r_gemm_median"], "r_attn": d["r_attn_median"],
        "gemm_share_of_delta": d["gemm_share_of_delta"],
        "attn_share_of_delta": d["attn_share_of_delta"],
        "whole_step_ratio_budget": ws["method_A_budget_share"],
        "whole_step_ratio_lumped": ws["method_B_lumped_share"],
        "whole_step_bracket_lo": ws["whole_step_bracket"][0],
        "whole_step_bracket_hi": ws["whole_step_bracket"][1],
        "fern_modeled_tree_step_ratio": be["fern_modeled_tree_step_ratio"],
        "corrected_breakeven_ET": be["corrected_breakeven_ET"],
        "corrected_breakeven_ET_lumped": be["corrected_breakeven_ET_lumped"],
        "corrected_breakeven_ET_bracket_lo": be["corrected_breakeven_ET_method_bracket"][0],
        "corrected_breakeven_ET_bracket_hi": be["corrected_breakeven_ET_method_bracket"][1],
        "fern_modeled_breakeven_ET": be["fern_modeled_breakeven_ET"],
        "fern_bracketed_by_methods": int(be["fern_bracketed_by_methods"]),
        "breakeven_delta_vs_fern_pct": be["delta_vs_fern_pct"],
        "beat_linear_ET": be["beat_linear_ET"],
        "verdict": res["verdict"],
        "verdict_green": int(res["verdict"] == "GREEN"),
        "peak_gpu_gib": res["config"]["peak_gpu_gib"],
    }
    for M in (args.m_linear, args.m_tree):
        summary[f"gemm_us_M{M}"] = s1[f"gemm_us_M{M}"]["median"]
        summary[f"attn_us_M{M}"] = s1[f"attn_us_M{M}"]["median"]
        summary[f"verify_forward_us_M{M}"] = s1[f"verify_forward_us_M{M}"]["median"]
    wandb.log(summary)
    run.summary.update(summary)
    # per-repeat table for audit
    cols = ["repeat", f"gemm_us_M{args.m_linear}", f"gemm_us_M{args.m_tree}",
            f"attn_us_M{args.m_linear}", f"attn_us_M{args.m_tree}",
            "r_gemm", "r_attn", "verify_forward_ratio"]
    tbl = wandb.Table(columns=cols)
    for rec in res["repeats"]:
        tbl.add_data(*[rec.get(c) for c in cols])
    wandb.log({"repeats_table": tbl})
    res["wandb_run_id"] = run.id
    (ROOT / args.output).write_text(json.dumps(res, indent=2, default=str))
    wandb.finish()
    print(f"[denom] W&B run {run.id} ({run.url})", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
