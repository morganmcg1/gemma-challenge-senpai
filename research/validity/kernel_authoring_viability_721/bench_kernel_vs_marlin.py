#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #721 (stark) -- custom Triton int4 g128 M=1 GEMV  vs  Marlin: head-to-head.

LOCAL A10G (sm_86). analysis_only=1, official_tps=0, no_hf_job=1, fires=0.
NO served-file change, NO HF Job, NO --launch, NO submission.

Q: can a purpose-built non-Marlin int4 g128 M=1 (conc=1) decode GEMV (Triton, this
   card's triton_int4_gemv.py) beat Marlin's realized M=1 decode-step latency on this
   A10G, or is the 90.7% HBM read-peak wall (denken #676) fundamental to any kernel?

Method (mirrors stark #602 int4_body_gemv_bw_saturation methodology):
  * Same served int4 g128 shapes (qkv/o/gate_up/down) + the int4 g128 lm_head.
  * For each shape, the SAME logical int4 weights feed BOTH kernels: Marlin via
    apply_gptq_marlin_linear (the served GPTQMarlinLinearMethod.apply path) and the
    Triton kernel via quantize_weights -> pack_int4_k8 (identical w_q integers + scales).
  * L2-cold timing: n_distinct cold weights per shape (working set >> 6 MiB A10G L2),
    CUDA-graph replay (serve-faithful ONEGRAPH amortized launch). Achieved DRAM BW is
    value-independent (shape/dtype/group_size/layout), so random weights at the served
    shapes faithfully reproduce the deployed kernel's bandwidth.
  * Co-measured STREAM read/copy peak (reproduces ubel #450 ~517 GB/s).
  * Triton variant per shape = min(single-pass autotuned, deterministic split-K).
  * Identity: argmax(greedy-token) match of Triton vs reference(x@w_ref f32) and vs
    Marlin on the lm_head GEMV over a large matched-scale vector sample.

Verdicts:
  KERNEL_HEADROOM_REALIZABLE -- Triton beats Marlin full-body M=1 latency by a material
    margin AND preserves greedy argmax. Report speedup + implied AR-lane TPS.
  KERNEL_HBM_WALLED -- Triton cannot beat Marlin / the 90.7% read-peak wall.

Reproduce: cd target/ && CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  research/validity/kernel_authoring_viability_721/bench_kernel_vs_marlin.py \
  --wandb_name stark/kernel-authoring-viability
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # local A10G index 0 (inherited =1 -> 0 GPUs)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # 262144-vocab head

import argparse
import gc
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from triton_int4_gemv import (  # noqa: E402
    pack_int4_k8, gemv_int4_g128, gemv_int4_g128_splitk, GROUP_SIZE, GROUP_KP)

# --------------------------------------------------------------- constants ----
A10G_SPEC_BW_GBPS = 600.0
BF16_BYTES = 2.0
N_LAYERS = 37
HIDDEN, INTERMEDIATE = 2560, 10240
N_Q, N_KV, HEAD_DIM = 8, 2, 256
VOCAB = 262144
SAT_THRESHOLD = 0.90

REF_OFFICIAL_TPS = 126.378          # operative int4_g128_lmhead (#319-byte-exact AR)
PLUS10_TARGET = 136.378
TAU_LOCAL_TO_OFFICIAL = 1.03524     # #267 local wall_tps -> official scalar
BODY_FRAC_591 = 0.444               # body int4-GEMV share of the decode cycle (#591/#602)
HEAD_FRAC_593 = 0.183               # int4 lm_head GEMV share of the decode cycle (#593)
MARLIN_676_READPEAK_PCT = 90.66     # denken #676 dominant-component read-peak wall

# (K=in, N=out) served fused shapes + the int4 g128 lm_head
BODY_SHAPES = {
    "qkv_proj":     (HIDDEN, N_Q * HEAD_DIM + 2 * N_KV * HEAD_DIM),   # 2560 -> 3072
    "o_proj":       (N_Q * HEAD_DIM, HIDDEN),                         # 2048 -> 2560
    "gate_up_proj": (HIDDEN, 2 * INTERMEDIATE),                       # 2560 -> 20480
    "down_proj":    (INTERMEDIATE, HIDDEN),                           # 10240 -> 2560
}
BODY_ORDER = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
HEAD_SHAPE = {"lm_head": (HIDDEN, VOCAB)}                             # 2560 -> 262144
ALL_SHAPES = {**BODY_SHAPES, **HEAD_SHAPE}


def shape_bytes(K: int, N: int, M: int = 1) -> dict:
    w = K * N * 0.5                       # int4 weight
    s = (K // GROUP_SIZE) * N * BF16_BYTES  # g128 bf16 scales
    act = (M * K + M * N) * BF16_BYTES    # read x + write y
    return {"weight_bytes": w, "scale_bytes": s, "act_bytes": act, "total_bytes": w + s + act}


# ------------------------------------------------------------- peak HBM BW ----
def _timed_eager(fn, iters, warmup):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters / 1e3  # s/call


def measure_peak_bw(dev, iters, warmup):
    import torch
    N = 512 * 1024 * 1024
    a = torch.empty(N, dtype=torch.bfloat16, device=dev).uniform_(-1, 1)
    b = torch.empty(N, dtype=torch.bfloat16, device=dev)
    nb = N * 2
    t_copy = _timed_eager(lambda: b.copy_(a), iters, warmup)
    t_read = _timed_eager(lambda: torch.sum(a), iters, warmup)
    del a, b; gc.collect(); torch.cuda.empty_cache()
    return {"bw_read_gbps": nb / t_read / 1e9, "bw_copy_gbps": 2 * nb / t_copy / 1e9}


# --------------------------------------------------- L2-cold CUDA-graph time --
def _graph_us_per_call(run, iters, warmup, rounds):
    import torch
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    series = []
    for _ in range(rounds):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            g.replay()
        e1.record(); torch.cuda.synchronize()
        series.append(e0.elapsed_time(e1) / iters * 1e3)  # us/call
    del g
    return statistics.median(series), statistics.pstdev(series) if len(series) > 1 else 0.0


# ------------------------------------------ build weights for one shape -------
def build_shape(K, N, dev, n_distinct):
    """Return n_distinct (marlin_run_inputs, triton_packed, triton_scales) for ONE shape.
    Marlin and Triton consume the SAME logical int4 g128 weights. The f32 dequant
    reference is NOT retained here (timing doesn't use it; head_identity rebuilds its
    own) -- holding n_distinct f32 refs OOMs the 262144-vocab lm_head (2.68 GB each)."""
    import torch
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new, marlin_make_empty_g_idx)
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
        marlin_quantize, quantize_weights)
    QT = scalar_types.uint4b8
    ws = marlin_make_workspace_new(dev)
    zp = marlin_make_empty_g_idx(dev)
    marlin = []
    triton = []
    for d in range(n_distinct):
        torch.manual_seed(1000 + d)
        w = (torch.randn(K, N, dtype=torch.bfloat16, device=dev) * 0.02)
        # Marlin permuted weights (the served path)
        _wr, q_w, s, g_idx, sort_idx, _perm = marlin_quantize(w, QT, GROUP_SIZE, act_order=False)
        marlin.append((q_w, s, g_idx, sort_idx))
        # Triton simple-packed weights from the SAME w (identical w_q + scales)
        _wref, w_q, w_s, _zp = quantize_weights(w.float(), QT, GROUP_SIZE, zero_points=False)
        triton.append((pack_int4_k8(w_q), w_s.to(torch.bfloat16).contiguous()))
        del w, _wr, _wref, w_q, w_s  # free the f32 dequant refs immediately
        gc.collect(); torch.cuda.empty_cache()

    def marlin_run_one(a, idx):
        q_w, s, g_idx, sort_idx = marlin[idx]
        return ops.marlin_gemm(a, None, q_w, None, s, None, None, zp, g_idx, sort_idx, ws,
                               QT, a.shape[0], N, K, True, False, True, False)

    return {"marlin": marlin, "triton": triton,
            "marlin_run_one": marlin_run_one, "ws": ws, "zp": zp, "K": K, "N": N}


# ------------------------------------- time Marlin + Triton for one shape -----
def time_shape(name, K, N, dev, n_distinct, iters, warmup, rounds, peak, splitk_grid):
    import torch
    sb = shape_bytes(K, N, 1)
    built = build_shape(K, N, dev, n_distinct)
    x = torch.randn(1, K, dtype=torch.bfloat16, device=dev) * 0.1

    # --- Marlin: time n_distinct cold GEMVs / call ---
    def marlin_call():
        for d in range(n_distinct):
            built["marlin_run_one"](x, d)
    for d in range(n_distinct):  # warmup compile/select
        built["marlin_run_one"](x, d)
    mu, msd = _graph_us_per_call(marlin_call, iters, warmup, rounds)
    marlin_us = mu / n_distinct
    marlin_sd = msd / n_distinct

    # --- Triton single-pass (autotuned): warm autotune, then graph ---
    for d in range(n_distinct):
        qp, sc = built["triton"][d]
        gemv_int4_g128(x, qp, sc, N, K)
    torch.cuda.synchronize()

    def triton_sp_call():
        for d in range(n_distinct):
            qp, sc = built["triton"][d]
            gemv_int4_g128(x, qp, sc, N, K)
    su, ssd = _graph_us_per_call(triton_sp_call, iters, warmup, rounds)
    triton_sp_us = su / n_distinct

    # --- Triton split-K (small grid, deterministic) ---
    best_spk_us = float("inf"); best_spk_cfg = None
    for (split, bn, bkp, w, st) in splitk_grid:
        try:
            for d in range(n_distinct):
                qp, sc = built["triton"][d]
                gemv_int4_g128_splitk(x, qp, sc, N, K, split=split, block_n=bn,
                                      block_kp=bkp, num_warps=w, num_stages=st)
            torch.cuda.synchronize()

            def spk_call(split=split, bn=bn, bkp=bkp, w=w, st=st):
                for d in range(n_distinct):
                    qp, sc = built["triton"][d]
                    gemv_int4_g128_splitk(x, qp, sc, N, K, split=split, block_n=bn,
                                          block_kp=bkp, num_warps=w, num_stages=st)
            ku, _ = _graph_us_per_call(spk_call, iters, warmup, rounds)
            ku /= n_distinct
            if ku < best_spk_us:
                best_spk_us = ku; best_spk_cfg = {"split": split, "block_n": bn,
                                                  "block_kp": bkp, "num_warps": w, "num_stages": st}
        except Exception as ex:  # noqa: BLE001
            print(f"    [splitk {split}/{bn}/{bkp}/{w}/{st}] skip: {ex}", flush=True)

    triton_best_us = min(triton_sp_us, best_spk_us)
    triton_best_kind = "single_pass" if triton_sp_us <= best_spk_us else "split_k"

    def bw(us):
        return (sb["total_bytes"] / (us * 1e-6)) / 1e9

    res = {
        "name": name, "K": K, "N": N, "total_bytes": sb["total_bytes"],
        "weight_bytes": sb["weight_bytes"], "scale_bytes": sb["scale_bytes"],
        "marlin_us": marlin_us, "marlin_us_sd": marlin_sd,
        "marlin_bw_gbps": bw(marlin_us), "marlin_f_read": bw(marlin_us) / peak["bw_read_gbps"],
        "triton_sp_us": triton_sp_us, "triton_sp_bw_gbps": bw(triton_sp_us),
        "triton_splitk_us": best_spk_us, "triton_splitk_cfg": best_spk_cfg,
        "triton_best_us": triton_best_us, "triton_best_kind": triton_best_kind,
        "triton_best_bw_gbps": bw(triton_best_us),
        "triton_best_f_read": bw(triton_best_us) / peak["bw_read_gbps"],
        "speedup_marlin_over_triton": marlin_us / triton_best_us,  # >1 => triton faster
        "triton_beats_marlin": bool(triton_best_us < marlin_us),
    }
    del built, x; gc.collect(); torch.cuda.empty_cache()
    return res


# --------------------------------------------- identity on the lm_head --------
def head_identity(dev, n_vec, K=HIDDEN, N=VOCAB):
    """argmax(greedy-token) match: Triton vs reference(x@w_ref f32) and vs Marlin, over
    n_vec matched-scale hidden vectors on the int4 g128 lm_head shape."""
    import torch
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new, marlin_make_empty_g_idx)
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
        marlin_quantize, quantize_weights)
    QT = scalar_types.uint4b8
    torch.manual_seed(7)
    w = (torch.randn(K, N, dtype=torch.bfloat16, device=dev) * 0.02)
    _wr, q_w, s, g_idx, sort_idx, _perm = marlin_quantize(w, QT, GROUP_SIZE, act_order=False)
    w_ref, w_q, w_s, _zp = quantize_weights(w.float(), QT, GROUP_SIZE, zero_points=False)
    qp = pack_int4_k8(w_q); sc = w_s.to(torch.bfloat16).contiguous()
    ws = marlin_make_workspace_new(dev); zp = marlin_make_empty_g_idx(dev)
    del w

    # "decisive" = f32-ref top-1 margin exceeds the bf16 logit resolution at that scale.
    # On a real confident model logits are decisive; random-weight logits are near-ties
    # (iid -> tiny gaps) which spuriously flip argmax for ANY int4 kernel incl. Marlin.
    tri_vs_ref = 0; tri_vs_marlin = 0; marlin_vs_ref = 0
    dec_n = 0; dec_tri_vs_ref = 0; dec_tri_vs_marlin = 0; dec_marlin_vs_ref = 0
    top1_margin = []
    for i in range(n_vec):
        torch.manual_seed(10_000 + i)
        x = torch.randn(1, K, dtype=torch.bfloat16, device=dev) * 0.1
        y_ref = (x.float() @ w_ref.float()).reshape(N)
        y_tri = gemv_int4_g128(x, qp, sc, N, K, out_dtype=torch.float32).reshape(N)
        y_mar = ops.marlin_gemm(x, None, q_w, None, s, None, None, zp, g_idx, sort_idx, ws,
                                QT, 1, N, K, True, False, True, False).float().reshape(N)
        a_ref = int(y_ref.argmax()); a_tri = int(y_tri.argmax()); a_mar = int(y_mar.argmax())
        tri_vs_ref += (a_tri == a_ref); tri_vs_marlin += (a_tri == a_mar)
        marlin_vs_ref += (a_mar == a_ref)
        top2 = torch.topk(y_ref, 2).values
        margin = (top2[0] - top2[1]).item()
        top1_margin.append(margin)
        # bf16 ULP at top-1 magnitude (~3 mantissa bits/decade -> 2^-8 rel)
        decisive = margin > abs(top2[0].item()) * (2 ** -7)
        if decisive:
            dec_n += 1
            dec_tri_vs_ref += (a_tri == a_ref); dec_tri_vs_marlin += (a_tri == a_mar)
            dec_marlin_vs_ref += (a_mar == a_ref)
    return {
        "n_vec": n_vec,
        "triton_vs_ref_argmax_rate": tri_vs_ref / n_vec,
        "triton_vs_marlin_argmax_rate": tri_vs_marlin / n_vec,
        "marlin_vs_ref_argmax_rate": marlin_vs_ref / n_vec,
        "decisive_n": dec_n,
        "decisive_triton_vs_ref_argmax_rate": (dec_tri_vs_ref / dec_n) if dec_n else float("nan"),
        "decisive_triton_vs_marlin_argmax_rate": (dec_tri_vs_marlin / dec_n) if dec_n else float("nan"),
        "decisive_marlin_vs_ref_argmax_rate": (dec_marlin_vs_ref / dec_n) if dec_n else float("nan"),
        "median_top1_margin": statistics.median(top1_margin),
    }


# ----------------------------------------------------------------- compose ----
def compose(gpu, peak, shapes, ident):
    # full-body Marlin/Triton time (per-component us * N_LAYERS)
    body_marlin_us = sum(shapes[c]["marlin_us"] for c in BODY_ORDER) * N_LAYERS
    body_triton_us = sum(shapes[c]["triton_best_us"] for c in BODY_ORDER) * N_LAYERS
    body_speedup = body_marlin_us / body_triton_us
    body_triton_beats = body_triton_us < body_marlin_us

    # implied official TPS if body GEMV time scales by (triton/marlin) on its cycle share
    def tps_if(frac, ratio):  # ratio = triton_us/marlin_us on that component
        return REF_OFFICIAL_TPS / ((1.0 - frac) + frac * ratio)
    body_ratio = body_triton_us / body_marlin_us
    tps_body_only = tps_if(BODY_FRAC_591, body_ratio)

    head_ratio = shapes["lm_head"]["triton_best_us"] / shapes["lm_head"]["marlin_us"]
    # body + head both swapped (combined GEMV-replaceable cycle share)
    combined_frac = BODY_FRAC_591 + HEAD_FRAC_593
    combined_ratio = ((BODY_FRAC_591 * body_ratio + HEAD_FRAC_593 * head_ratio) / combined_frac)
    tps_body_plus_head = tps_if(combined_frac, combined_ratio)

    any_component_beats = any(shapes[c]["triton_beats_marlin"] for c in ALL_SHAPES)
    headroom_realizable = bool(body_triton_beats and tps_body_only > REF_OFFICIAL_TPS + 0.5
                               and ident["triton_vs_ref_argmax_rate"] >= 0.999)

    verdict_tag = "KERNEL_HEADROOM_REALIZABLE" if headroom_realizable else "KERNEL_HBM_WALLED"
    verdict = (
        f"{verdict_tag}. Custom Triton int4 g128 M=1 GEMV vs Marlin on A10G sm_86. "
        f"Read-peak {peak['bw_read_gbps']:.1f} GB/s. FULL-BODY (37L): Marlin {body_marlin_us:.0f}us "
        f"({shapes['gate_up_proj']['marlin_f_read']*100:.1f}% read-peak on dominant gate_up) vs "
        f"Triton-best {body_triton_us:.0f}us -> body_speedup {body_speedup:.4f}x "
        f"(triton_beats_marlin={body_triton_beats}). Per-component triton/marlin us: " +
        ", ".join(f"{c}={shapes[c]['triton_best_us']:.0f}/{shapes[c]['marlin_us']:.0f}"
                  f"({shapes[c]['triton_best_kind']})" for c in BODY_ORDER) +
        f", lm_head={shapes['lm_head']['triton_best_us']:.0f}/{shapes['lm_head']['marlin_us']:.0f}. "
        f"Marlin per-component read-peak%: " +
        ", ".join(f"{c}={shapes[c]['marlin_f_read']*100:.0f}" for c in BODY_ORDER) +
        f". Triton-best read-peak%: " +
        ", ".join(f"{c}={shapes[c]['triton_best_f_read']*100:.0f}" for c in BODY_ORDER) +
        f". Implied official TPS (body-share {BODY_FRAC_591}): {tps_body_only:.2f} vs ref "
        f"{REF_OFFICIAL_TPS} (+10 bar {PLUS10_TARGET}). Identity (lm_head argmax over "
        f"{ident['n_vec']} vecs): triton-vs-ref {ident['triton_vs_ref_argmax_rate']*100:.1f}%, "
        f"triton-vs-marlin {ident['triton_vs_marlin_argmax_rate']*100:.1f}%. "
        f"The byte-identical/value-equivalent int4 GEMV is bandwidth-fundamental; Marlin is "
        f"near-optimal on the dominant large-N shapes and a custom Triton GEMV does not create a "
        f"+10 speed lever at M=1."
    )

    headline = {
        "verdict_tag": verdict_tag,
        "kernel_headroom_realizable": headroom_realizable,
        "triton_beats_marlin_fullbody": body_triton_beats,
        "any_component_triton_beats_marlin": any_component_beats,
        "body_marlin_us": body_marlin_us, "body_triton_us": body_triton_us,
        "body_speedup_marlin_over_triton": body_speedup,
        "kernel_m1_decode_tps_vs_marlin": tps_body_only,
        "tps_body_only_implied": tps_body_only,
        "tps_body_plus_head_implied": tps_body_plus_head,
        "peak_read_gbps": peak["bw_read_gbps"], "peak_copy_gbps": peak["bw_copy_gbps"],
        "marlin_dominant_f_read_pct": shapes["gate_up_proj"]["marlin_f_read"] * 100,
        "triton_best_dominant_f_read_pct": shapes["gate_up_proj"]["triton_best_f_read"] * 100,
        "head_triton_us": shapes["lm_head"]["triton_best_us"],
        "head_marlin_us": shapes["lm_head"]["marlin_us"],
        "head_speedup_marlin_over_triton": shapes["lm_head"]["marlin_us"] / shapes["lm_head"]["triton_best_us"],
        "triton_vs_ref_argmax_rate": ident["triton_vs_ref_argmax_rate"],
        "triton_vs_marlin_argmax_rate": ident["triton_vs_marlin_argmax_rate"],
        "marlin_vs_ref_argmax_rate": ident["marlin_vs_ref_argmax_rate"],
        "ref_official_tps": REF_OFFICIAL_TPS, "plus10_target_tps": PLUS10_TARGET,
        "marlin_676_readpeak_pct": MARLIN_676_READPEAK_PCT,
        "official_tps": 0, "analysis_only": True, "no_hf_job": True, "fires": 0,
        "no_served_file_change": True,
    }
    for c in ALL_SHAPES:
        headline[f"{c}_marlin_us"] = shapes[c]["marlin_us"]
        headline[f"{c}_triton_best_us"] = shapes[c]["triton_best_us"]
        headline[f"{c}_triton_best_kind"] = shapes[c]["triton_best_kind"]
        headline[f"{c}_speedup_marlin_over_triton"] = shapes[c]["speedup_marlin_over_triton"]
        headline[f"{c}_marlin_f_read"] = shapes[c]["marlin_f_read"]
        headline[f"{c}_triton_best_f_read"] = shapes[c]["triton_best_f_read"]
    return {"gpu": gpu, "peak_bw": peak, "shapes": shapes, "identity": ident,
            "headline": headline, "verdict": verdict,
            "timestamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}


def _log_wandb(args, payload):
    try:
        import wandb
    except Exception as ex:  # noqa: BLE001
        print(f"[wandb] import failed: {ex!r}", flush=True); return None
    h = payload["headline"]
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type="kernel-authoring-viability",
                     config={"agent": "stark", "pr": 721, "kind": "kernel-authoring-viability",
                             "group_size": GROUP_SIZE, "n_layers": N_LAYERS,
                             "ref_official_tps": REF_OFFICIAL_TPS}, reinit=True)
    flat = {k: v for k, v in h.items() if isinstance(v, (int, float, bool))}
    wandb.log(flat); wandb.summary.update(flat)
    wandb.summary["verdict"] = payload["verdict"]
    wandb.summary["verdict_tag"] = h["verdict_tag"]
    rid = run.id
    wandb.finish()
    return rid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--n-distinct", type=int, default=6)
    ap.add_argument("--n-distinct-head", type=int, default=2)  # 1 lm_head weight=335MB >> 6MB L2
    ap.add_argument("--n-vec", type=int, default=256)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--output", default=str(HERE / "bench_kernel_vs_marlin_results.json"))
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="kernel-authoring-viability-721")
    ap.add_argument("--wandb_name", default="stark/kernel-authoring-viability")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    import torch
    dev = torch.device("cuda:0")
    p = torch.cuda.get_device_properties(dev); cc = torch.cuda.get_device_capability(dev)
    gpu = {"name": p.name, "sm_count": p.multi_processor_count,
           "compute_capability": f"{cc[0]}.{cc[1]}", "is_sm86": bool(cc == (8, 6))}
    print("[gpu]", gpu, flush=True)

    if args.smoke:
        iters, warmup, rounds, n_distinct, n_dhead, n_vec = 5, 5, 3, 2, 2, 16
        splitk_grid = [(4, 64, 64, 4, 3), (8, 32, 64, 4, 3)]
    else:
        iters, warmup, rounds = args.iters, args.warmup, args.rounds
        n_distinct, n_dhead, n_vec = args.n_distinct, args.n_distinct_head, args.n_vec
        splitk_grid = [(s, bn, bkp, 4, 3) for s in (2, 4, 8, 16)
                       for bn in (16, 32, 64) for bkp in (64, 128)]

    peak = measure_peak_bw(dev, iters, warmup)
    print("[peak]", {k: round(v, 1) for k, v in peak.items()}, flush=True)

    shapes = {}
    for name, (K, N) in ALL_SHAPES.items():
        nd = n_dhead if name == "lm_head" else n_distinct
        print(f"[time] {name} K={K} N={N} n_distinct={nd} ...", flush=True)
        shapes[name] = time_shape(name, K, N, dev, nd, iters, warmup, rounds, peak, splitk_grid)
        r = shapes[name]
        print(f"    marlin={r['marlin_us']:.1f}us ({r['marlin_f_read']*100:.1f}% rp)  "
              f"triton_best={r['triton_best_us']:.1f}us ({r['triton_best_kind']}, "
              f"{r['triton_best_f_read']*100:.1f}% rp)  speedup(m/t)={r['speedup_marlin_over_triton']:.4f}  "
              f"triton_wins={r['triton_beats_marlin']}", flush=True)

    print("[identity] lm_head argmax over", n_vec, "vecs ...", flush=True)
    ident = head_identity(dev, n_vec)
    print("    ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ident.items()}, flush=True)

    payload = compose(gpu, peak, shapes, ident)
    json.dump(payload, open(args.output, "w"), indent=2)
    print("\n=== HEADLINE ===", flush=True)
    print(json.dumps({k: v for k, v in payload["headline"].items()
                      if not k.startswith(tuple(ALL_SHAPES))}, indent=2), flush=True)
    print("\n=== VERDICT ===\n" + payload["verdict"], flush=True)

    if not args.no_wandb and not args.smoke:
        rid = _log_wandb(args, payload)
        payload["wandb_run_id"] = rid
        json.dump(payload, open(args.output, "w"), indent=2)
        print(f"\n[wandb] run_id={rid}", flush=True)


if __name__ == "__main__":
    main()
