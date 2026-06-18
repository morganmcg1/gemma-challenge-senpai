#!/usr/bin/env python
"""PR #676 (denken) -- int4-GEMV HBM-roofline CEILING.

Does the shipped int4_g128_lmhead AR (spec-OFF, M=1) decode GEMV run at the HBM
bandwidth WALL (-> AR speed leg fundamentally capped, lawine #675 kernel sweep is
bounded, refocus on spec-dec), or WELL BELOW peak because of a dequant-ALU tax
(-> real reclaimable headroom, green light for the kernel axis)?

LOCAL A10G (sm_86) microbench. analysis_only=True, official_tps=0, NO served-file
change, NO HF Job. Greedy/PPL pinned by construction (no served change; achieved
DRAM BW is value-independent -> random weights at the served shapes reproduce the
deployed kernel's bandwidth).

This OWNS the ROOFLINE CEILING (denken #676); lawine #675 owns WHICH byte-identical
kernel/config is fastest. Different deliverable, different method. Reuses the
#674 anchor (matmul bucket 6919.9 us = 85% of the 8161.9 us GPU-busy step) and the
SAME apply_gptq_marlin_linear the served GPTQMarlinLinearMethod.apply calls.

Deliverables:
  1. EMPIRICAL achievable HBM peak (STREAM read/copy + DtoD memcpy) -- the real
     denominator, NOT the 600 GB/s GA102 datasheet.
  2. EXACT per-token int4 GEMV bytes from the served safetensors header (body 42
     decode layers w/ Gemma-3n KV-sharing + full-vocab int4 lm_head + PLE proj).
  3. achieved int4-GEMV GB/s (isolated kernel) + gemv_pct_of_hbm_peak; reconciled
     against the in-loop #674 matmul bucket.
  4. BW-vs-ALU split: bf16-WEIGHT GEMV control at the SAME shapes (no dequant) ->
     isolate the int4->bf16 dequant-ALU tax. + M-invariance (latency vs bandwidth).
  5. official-equiv AR TPS ceiling via stark tax 0.870 (GEMV at empirical peak).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import struct
import sys
from collections import defaultdict
from datetime import datetime, timezone

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # local A10G is index 0

import torch

SERVED = "/workspace/gemma_build/int4_g128_lmhead/model.safetensors"
A10G_SPEC_BW_GBPS = 600.0          # GA102 datasheet peak (NOT achievable)
GROUP_SIZE = 128
HIDDEN = 2560
VOCAB = 262144
STARK_TAX = 0.870                  # local -> official-equiv (PR #676 baseline)
REF_OFFICIAL_TPS = 126.378
PLUS10 = 136.378
# #674 anchor (FULL_AND_PIECEWISE, BI=1, AR M=1, the matched anchor for this card)
D674 = {
    "gpu_busy_us": 8161.866273437305,
    "clean_wall_us": 8138.943890282619,
    "matmul_us": 6919.9059492186625,
    "attn_us": 903.4943359374225,
    "norm_us": 175.36125781246608,
    "sampling_us": 14.871394531239364,
    "other_us": 148.23333593751414,
    "local_tps": 122.86606388747026,
    "local_ar_anchor_ref": 126.94,
}

# canonical SERVED-dispatch fused shapes (K=in_features, N=out_features)
SHAPES = {
    "qkv_proj":     (2560, 3072),    # fused q(2048)+k(512)+v(512) in kv-producing layers
    "o_proj":       (2048, 2560),
    "gate_up_proj": (2560, 20480),   # fused gate(10240)+up(10240)
    "down_proj":    (10240, 2560),
    "ple_proj":     (256, 2560),     # per_layer_projection / input_gate (small)
    "lm_head":      (2560, 262144),  # AR full-vocab int4 head
}
BF16_BYTES = 2.0


# ----------------------------------------------------- exact byte model --------
def _hdr(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def _nbytes(v):
    el = 1
    for s in v["shape"]:
        el *= s
    bpe = {"I32": 4, "I64": 8, "BF16": 2, "F16": 2, "F32": 4, "I8": 1, "U8": 1}[v["dtype"]]
    return el * bpe


def exact_byte_model():
    """Per-token int4 weight+scale bytes streamed by the GEMV path, summed EXACTLY
    over the served safetensors (Gemma-3n: q/o on 42 layers, k/v on 24 layers w/
    KV-sharing, full-vocab int4 lm_head). Returns component -> bytes + the map onto
    the canonical timed shapes."""
    import re
    hdr = _hdr(SERVED)
    comp_b = defaultdict(float)
    comp_layers = defaultdict(set)
    for k, v in hdr.items():
        m = re.match(r"model\.language_model\.layers\.(\d+)\.(.+?)\.(weight_packed|weight_scale)$", k)
        if m:
            comp_b[m.group(2)] += _nbytes(v)
            comp_layers[m.group(2)].add(int(m.group(1)))
    # lm_head
    lm = 0.0
    for k, v in hdr.items():
        if k.startswith("lm_head.") and k.endswith(("weight_packed", "weight_scale")):
            lm += _nbytes(v)
    # map onto canonical timed shapes (BW is shape-keyed; bytes are exact)
    qkv = comp_b["self_attn.q_proj"] + comp_b["self_attn.k_proj"] + comp_b["self_attn.v_proj"]
    gate_up = comp_b["mlp.gate_proj"] + comp_b["mlp.up_proj"]
    ple = comp_b.get("per_layer_projection", 0.0) + comp_b.get("per_layer_input_gate", 0.0)
    mapped = {
        "qkv_proj": qkv,
        "o_proj": comp_b["self_attn.o_proj"],
        "gate_up_proj": gate_up,
        "down_proj": comp_b["mlp.down_proj"],
        "ple_proj": ple,
        "lm_head": lm,
    }
    body = mapped["qkv_proj"] + mapped["o_proj"] + mapped["gate_up_proj"] + mapped["down_proj"] + mapped["ple_proj"]
    return {
        "per_component_mb": {k: v / 1e6 for k, v in comp_b.items()},
        "component_layers": {k: len(v) for k, v in comp_layers.items()},
        "mapped_bytes": mapped,
        "body_bytes": body,
        "lm_head_bytes": mapped["lm_head"],
        "full_gemv_bytes": body + mapped["lm_head"],
    }


# ----------------------------------------------------- empirical HBM peak ------
def _timed_eager(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters / 1e3   # s/call


def measure_peak_bw(dev, iters=50, warmup=40):
    N = 512 * 1024 * 1024
    a = torch.empty(N, dtype=torch.bfloat16, device=dev).uniform_(-1, 1)
    b = torch.empty(N, dtype=torch.bfloat16, device=dev)
    nb = N * 2
    t_copy = _timed_eager(lambda: b.copy_(a), iters, warmup)
    t_read = _timed_eager(lambda: torch.sum(a), iters, warmup)
    # explicit DtoD memcpy (2x: read+write), independent of the reduction kernel
    t_memcpy = _timed_eager(lambda: b.copy_(a, non_blocking=True), iters, warmup)
    del a, b
    gc.collect(); torch.cuda.empty_cache()
    return {
        "bw_read_gbps": nb / t_read / 1e9, "read_us": t_read * 1e6,
        "bw_copy_gbps": 2 * nb / t_copy / 1e9, "copy_us": t_copy * 1e6,
        "bw_memcpy_dtod_gbps": 2 * nb / t_memcpy / 1e9,
    }


# ----------------------------------------------------- L2-cold graph timing ----
def _graph_us_per_call(run, calls_in_graph, iters, warmup, rounds):
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
        series.append(e0.elapsed_time(e1) / iters / calls_in_graph * 1e3)  # us/call
    del g
    return statistics.median(series), series


def _per_call_int4_bytes(K, N):
    w = (K * N) / 2.0                      # int4 packed
    s = N * (K / GROUP_SIZE) * BF16_BYTES  # bf16 group scales
    return w, s


# ----------------------------------------------------- int4 GEMV per shape -----
def time_int4_shape(dev, K, N, M, n_distinct, calls, iters, warmup, rounds):
    from vllm.model_executor.layers.quantization.utils import marlin_utils_test as mt
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_gptq_marlin_linear as apply_marlin, marlin_make_workspace_new as mk_ws)
    from vllm.scalar_type import scalar_types
    QT = scalar_types.uint4b8
    ws = mk_ws(dev); zp = torch.zeros(0, dtype=torch.int, device=dev)
    wl = []
    for _ in range(n_distinct):
        w = torch.randn(K, N, dtype=torch.float16, device=dev) * 0.02
        _, q_w, s, g_idx, sort_idx, _perm = mt.marlin_quantize(w, QT, GROUP_SIZE, False)
        del w; wl.append((q_w, s, g_idx, sort_idx))
    x = torch.randn(M, K, dtype=torch.float16, device=dev)

    def run():
        for c in range(calls):
            q_w, s, g_idx, sort_idx = wl[c % len(wl)]
            apply_marlin(x, q_w, s, zp, g_idx, sort_idx, ws, QT, N, K, is_k_full=True, bias=None)

    us, series = _graph_us_per_call(run, calls, iters, warmup, rounds)
    del wl, x
    gc.collect(); torch.cuda.empty_cache()
    wb, sb = _per_call_int4_bytes(K, N)
    act = M * K * BF16_BYTES + M * N * BF16_BYTES
    total = wb + sb + act
    return {"us": us, "weight_bytes": wb, "scale_bytes": sb, "act_bytes": act,
            "total_bytes": total, "achieved_bw_gbps": (total / (us * 1e-6)) / 1e9,
            "series_cv": (statistics.pstdev(series) / us if us else 0.0)}


# ----------------------------------------------------- bf16-weight control -----
def time_bf16_shape(dev, K, N, M, n_distinct, calls, iters, warmup, rounds):
    """Same shape, bf16 weights, NO dequant -- isolates the dequant-ALU tax. At M=1
    AI<<ridge so this is also memory-bound; if int4_bw ~ bf16_bw the dequant is
    hidden under the weight read (bandwidth-bound); if int4_bw << bf16_bw the
    dequant-ALU is the bottleneck."""
    import torch.nn.functional as Fnn
    wl = [torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.02 for _ in range(n_distinct)]
    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)

    def run():
        for c in range(calls):
            Fnn.linear(x, wl[c % len(wl)])

    us, series = _graph_us_per_call(run, calls, iters, warmup, rounds)
    del wl, x
    gc.collect(); torch.cuda.empty_cache()
    wb = N * K * BF16_BYTES
    act = M * K * BF16_BYTES + M * N * BF16_BYTES
    total = wb + act
    return {"us": us, "weight_bytes": wb, "act_bytes": act, "total_bytes": total,
            "achieved_bw_gbps": (total / (us * 1e-6)) / 1e9,
            "series_cv": (statistics.pstdev(series) / us if us else 0.0)}


# --------------------------------------------------------------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=1)
    ap.add_argument("--m_sweep", type=str, default="1,2,4,8")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--calls", type=int, default=42)     # layers per body shape
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--wandb_name", type=str, default="denken/gemv-hbm-roofline-ceiling")
    ap.add_argument("--wandb_group", type=str, default="gemv-hbm-roofline-denken")
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.iters, args.warmup, args.rounds, args.calls = 8, 5, 4, 6
        args.m_sweep = "1"

    dev = torch.device("cuda:0")
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    print(f"[denken#676] {name} sm_{cc[0]}{cc[1]} torch {torch.__version__}", flush=True)

    bm = exact_byte_model()
    print(f"[byte-model] body={bm['body_bytes']/1e9:.4f} GB  lm_head={bm['lm_head_bytes']/1e9:.4f} GB"
          f"  full={bm['full_gemv_bytes']/1e9:.4f} GB", flush=True)
    print(f"[byte-model] component layers: {bm['component_layers']}", flush=True)

    peak = measure_peak_bw(dev)
    print(f"[peak] read={peak['bw_read_gbps']:.1f} copy={peak['bw_copy_gbps']:.1f} "
          f"memcpy={peak['bw_memcpy_dtod_gbps']:.1f} GB/s "
          f"(read is {100*peak['bw_read_gbps']/A10G_SPEC_BW_GBPS:.1f}% of 600 spec)", flush=True)
    READ = peak["bw_read_gbps"]

    # n_distinct per shape so the cold working set exceeds the 6 MiB A10G L2
    def n_dist(K, N, is_bf16):
        wb = N * K * (BF16_BYTES if is_bf16 else 0.5)
        return max(2, min(8, math.ceil(28e6 / wb)))

    m_list = [int(x) for x in args.m_sweep.split(",")]
    int4 = {}
    bf16 = {}
    msweep = defaultdict(dict)
    for sh, (K, N) in SHAPES.items():
        calls = 1 if sh == "lm_head" else args.calls
        nd = n_dist(K, N, False)
        r = time_int4_shape(dev, K, N, args.M, nd, calls, args.iters, args.warmup, args.rounds)
        r["f_vs_read"] = r["achieved_bw_gbps"] / READ
        r["f_vs_spec"] = r["achieved_bw_gbps"] / A10G_SPEC_BW_GBPS
        r["n_distinct"] = nd
        int4[sh] = r
        ndb = n_dist(K, N, True)
        rb = time_bf16_shape(dev, K, N, args.M, ndb, calls, args.iters, args.warmup, args.rounds)
        rb["f_vs_read"] = rb["achieved_bw_gbps"] / READ
        bf16[sh] = rb
        print(f"[{sh:13s}] K={K:5d} N={N:6d} | int4 {r['us']:8.2f}us {r['achieved_bw_gbps']:6.1f}GB/s "
              f"({100*r['f_vs_read']:.1f}% read) | bf16 {rb['us']:8.2f}us {rb['achieved_bw_gbps']:6.1f}GB/s "
              f"({100*rb['f_vs_read']:.1f}%) | dequant_tax {100*(1-r['achieved_bw_gbps']/rb['achieved_bw_gbps']):+.1f}%",
              flush=True)
        # M-invariance only on the dominant gate_up
        if sh == "gate_up_proj":
            for M in m_list:
                rm = time_int4_shape(dev, K, N, M, nd, calls, args.iters, args.warmup, args.rounds)
                msweep[sh][str(M)] = {"M": M, "us": rm["us"], "bw": rm["achieved_bw_gbps"],
                                      "f_vs_read": rm["achieved_bw_gbps"] / READ}

    # ---- aggregate body + full int4 GEMV using EXACT bytes, per-shape BW --------
    def comp_time_us(sh, exact_bytes):
        # bandwidth-bound: time = exact_bytes / achieved_bw(shape)
        return exact_bytes / (int4[sh]["achieved_bw_gbps"] * 1e9) * 1e6

    def comp_time_us_bf16(sh, exact_bytes):
        return exact_bytes / (bf16[sh]["achieved_bw_gbps"] * 1e9) * 1e6

    body_shapes = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj", "ple_proj"]
    # dominant weight-streaming GEMVs (the "is it at the bandwidth wall" axis);
    # ple_proj is a tiny latency-bound tail (K=256) reported but not headline.
    dominant_shapes = ["gate_up_proj", "down_proj", "lm_head"]
    body_us = sum(comp_time_us(sh, bm["mapped_bytes"][sh]) for sh in body_shapes)
    lm_us = comp_time_us("lm_head", bm["mapped_bytes"]["lm_head"])
    full_us = body_us + lm_us
    body_bw = bm["body_bytes"] / (body_us * 1e-6) / 1e9
    full_bw = bm["full_gemv_bytes"] / (full_us * 1e-6) / 1e9
    body_us_bf16 = sum(comp_time_us_bf16(sh, bm["mapped_bytes"][sh]) for sh in body_shapes)
    body_bw_bf16 = bm["body_bytes"] / (body_us_bf16 * 1e-6) / 1e9
    # dominant-shape aggregates (byte-weighted)
    dom_bytes = sum(bm["mapped_bytes"][sh] for sh in dominant_shapes)
    dom_us = sum(comp_time_us(sh, bm["mapped_bytes"][sh]) for sh in dominant_shapes)
    dom_us_bf16 = sum(comp_time_us_bf16(sh, bm["mapped_bytes"][sh]) for sh in dominant_shapes)
    dom_bw = dom_bytes / (dom_us * 1e-6) / 1e9
    dom_bw_bf16 = dom_bytes / (dom_us_bf16 * 1e-6) / 1e9
    dom_pct = 100 * dom_bw / READ
    dom_dequant_tax = 1 - dom_bw / dom_bw_bf16

    # ---- reconcile vs in-loop #674 matmul bucket (6919.9 us) -------------------
    inloop_bucket_us = D674["matmul_us"]
    inloop_bw = bm["full_gemv_bytes"] / (inloop_bucket_us * 1e-6) / 1e9   # bytes / full bucket
    iso_vs_inloop_gap_us = inloop_bucket_us - full_us                      # in-loop overhead (NOT bandwidth)

    # ---- official-equiv AR TPS ceiling (GEMV at empirical read peak) -----------
    full_floor_us = bm["full_gemv_bytes"] / (READ * 1e9) * 1e6   # if GEMV ran at read-peak
    # ceiling replaces the measured GEMV streaming portion with the read-peak floor,
    # keeping attn/norm/sampling/other and the irreducible in-loop overhead fixed.
    saved_us = full_us - full_floor_us                            # reclaimable IF kernel hit peak
    step_now_us = D674["clean_wall_us"]
    step_ceiling_us = step_now_us - saved_us
    local_tps_now = 1e6 / step_now_us
    local_tps_ceiling = 1e6 / step_ceiling_us
    speedup = local_tps_ceiling / local_tps_now
    off_tps_now = local_tps_now * STARK_TAX
    off_tps_ceiling = local_tps_ceiling * STARK_TAX
    # tax convention is ambiguous (local 122.9 x0.870=106.9 != live 126.378); also
    # project the measured local SPEEDUP onto the live official rung as a 2nd basis.
    off_tps_ceiling_live_basis = REF_OFFICIAL_TPS * speedup

    # ---- verdict --------------------------------------------------------------
    gemv_pct = 100 * full_bw / READ
    body_pct = 100 * body_bw / READ
    dequant_tax = 1 - body_bw / body_bw_bf16          # BW-weighted over body shapes
    msw = msweep.get("gate_up_proj", {})
    m_inv = (msw.get("8", {}).get("us", 1) / msw.get("1", {}).get("us", 1)) if "8" in msw and "1" in msw else None
    # Operational discriminator (mirrors stark #602): the GEMV is AT THE WALL when
    #   (a) the dominant weight-streaming shares (gate_up+down+lm_head, ~86% of
    #       bytes) run >=85% of empirical read-peak with ~0 dequant-ALU tax, i.e.
    #       the gap is unavoidable BW/silicon -- NOT the dequant-ALU headroom case;
    #   AND (b) any sub-peak residual (small qkv/o occupancy) is NOT byte-identically
    #       recoverable (stark #602: only-Marlin on sm_86; fp32_reduce breaks bits).
    # GEMV_HAS_HEADROOM requires achieved<<peak AND the gap be dequant-ALU.
    dominant_at_wall = (dom_pct >= 85.0) and (dom_dequant_tax < 0.08)
    m_invariant = (m_inv is None) or (m_inv < 1.10)   # flat -> bandwidth-bound not latency
    byte_identical_headroom = False                   # stark #602 proof (no sm_86 lever)
    at_wall = dominant_at_wall and m_invariant and (not byte_identical_headroom)
    verdict = "GEMV_AT_HBM_WALL" if at_wall else "GEMV_HAS_HEADROOM"

    out = {
        "gpu": {"name": name, "sm": f"sm_{cc[0]}{cc[1]}", "torch": torch.__version__},
        "peak_bw": peak,
        "byte_model": {k: bm[k] for k in ("body_bytes", "lm_head_bytes", "full_gemv_bytes",
                                          "component_layers")} | {"mapped_mb": {k: v/1e6 for k, v in bm["mapped_bytes"].items()},
                                                                   "per_component_mb": bm["per_component_mb"]},
        "int4_per_shape": int4,
        "bf16_per_shape": bf16,
        "m_sweep_gate_up": dict(msweep),
        "aggregate": {
            "body_us": body_us, "lm_head_us": lm_us, "full_us": full_us,
            "body_achieved_bw_gbps": body_bw, "full_achieved_bw_gbps": full_bw,
            "body_achieved_bw_gbps_bf16": body_bw_bf16,
            "body_pct_of_read_peak": body_pct,
            "gemv_achieved_bw_gbps": full_bw,
            "gemv_pct_of_hbm_peak": gemv_pct,
            "gemv_pct_of_spec": 100 * full_bw / A10G_SPEC_BW_GBPS,
            "dequant_alu_tax_frac": dequant_tax,
            "m_invariance_m8_over_m1": m_inv,
            "dominant_bytes_gb": dom_bytes / 1e9,
            "dominant_frac_of_full": dom_bytes / bm["full_gemv_bytes"],
            "dominant_achieved_bw_gbps": dom_bw,
            "dominant_pct_of_read_peak": dom_pct,
            "dominant_achieved_bw_gbps_bf16": dom_bw_bf16,
            "dominant_dequant_alu_tax_frac": dom_dequant_tax,
        },
        "inloop_reconcile": {
            "d674_matmul_bucket_us": inloop_bucket_us,
            "inloop_bucket_bw_gbps": inloop_bw,
            "inloop_bucket_pct_of_read_peak": 100 * inloop_bw / READ,
            "inloop_bucket_pct_of_spec": 100 * inloop_bw / A10G_SPEC_BW_GBPS,
            "isolated_full_gemv_us": full_us,
            "iso_vs_inloop_gap_us": iso_vs_inloop_gap_us,
            "iso_vs_inloop_gap_frac_of_bucket": iso_vs_inloop_gap_us / inloop_bucket_us,
        },
        "ceiling": {
            "stark_tax": STARK_TAX,
            "full_gemv_floor_us_at_read_peak": full_floor_us,
            "saved_us_if_kernel_hit_peak": saved_us,
            "local_tps_now": local_tps_now, "local_tps_ceiling": local_tps_ceiling,
            "speedup_if_kernel_hit_peak": speedup,
            "official_equiv_tps_now": off_tps_now, "official_equiv_tps_ceiling": off_tps_ceiling,
            "official_equiv_tps_ceiling_live_basis": off_tps_ceiling_live_basis,
            "ref_official_tps": REF_OFFICIAL_TPS, "plus10_bar": PLUS10,
            "ceiling_clears_ref_tax_basis": off_tps_ceiling > REF_OFFICIAL_TPS,
            "ceiling_clears_plus10_tax_basis": off_tps_ceiling > PLUS10,
            "ceiling_clears_plus10_live_basis": off_tps_ceiling_live_basis > PLUS10,
            "headroom_is_byte_identical": False,   # stark #602: no byte-identical lever on sm_86
        },
        "verdict": {
            "verdict": verdict,
            "gemv_achieved_bw_gbps": full_bw,
            "gemv_pct_of_hbm_peak": gemv_pct,
            "body_pct_of_read_peak": body_pct,
            "dequant_alu_tax_frac": dequant_tax,
            "headroom_margin_pct_below_peak": 100 - body_pct,
            "official_equiv_tps_ceiling": off_tps_ceiling,
            "analysis_only": True, "official_tps": 0, "fires": False,
            "no_served_file_change": True, "no_hf_job": True,
            "ref_official_tps": REF_OFFICIAL_TPS,
        },
        "anchors": {"d674": D674, "stark602_body_bw": 436.3, "stark602_read_peak": 517.9,
                    "gemm_roofline_read_peak": 517.6},
        "args": vars(args),
        "timestamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    }

    # ---- self-test ------------------------------------------------------------
    st = {
        "byte_model_body_gt_1p5GB": bm["body_bytes"] > 1.5e9,
        "lm_head_near_int4_floor": abs(bm["lm_head_bytes"] - (VOCAB * HIDDEN / 2 + VOCAB * (HIDDEN / GROUP_SIZE) * 2)) / bm["lm_head_bytes"] < 0.02,
        "read_peak_below_spec": READ < A10G_SPEC_BW_GBPS,
        "gemv_pct_in_unit": 0 < gemv_pct < 100,
        "body_pct_in_unit": 0 < body_pct < 100,
        "dequant_tax_finite": math.isfinite(dequant_tax),
        "ceiling_above_now": off_tps_ceiling > off_tps_now,
        "official_tps_zero": out["verdict"]["official_tps"] == 0,
    }
    out["self_test"] = {"passes": all(st.values()), "checks": st}

    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    jp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      f"roofline_ceiling{'_smoke' if args.smoke else ''}.json")
    with open(jp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[VERDICT] {verdict}", flush=True)
    print(f"  DOMINANT (gate_up+down+lm_head, {100*dom_bytes/bm['full_gemv_bytes']:.0f}% of bytes) = "
          f"{dom_bw:.1f} GB/s = {dom_pct:.1f}% read-peak | dequant-ALU tax {100*dom_dequant_tax:+.1f}% | M8/M1={m_inv} -> {'AT WALL' if dominant_at_wall else 'sub-peak'}", flush=True)
    print(f"  full path        = {full_bw:.1f} GB/s = {gemv_pct:.1f}% read-peak / {100*full_bw/A10G_SPEC_BW_GBPS:.1f}% spec", flush=True)
    print(f"  body-only        = {body_bw:.1f} GB/s = {body_pct:.1f}% read-peak  (dequant-ALU tax {100*dequant_tax:+.1f}%)", flush=True)
    print(f"  in-loop #674 bucket = {inloop_bw:.1f} GB/s = {100*inloop_bw/READ:.1f}% read-peak "
          f"(gap {iso_vs_inloop_gap_us:.0f}us = {100*iso_vs_inloop_gap_us/inloop_bucket_us:.0f}% of bucket is in-loop overhead, NOT bandwidth)", flush=True)
    print(f"  ceiling (GEMV@peak, NOT byte-identically realizable): speedup {speedup:.3f}x | "
          f"local {local_tps_now:.1f}->{local_tps_ceiling:.1f} | "
          f"official-equiv x0.870={off_tps_ceiling:.1f} | live-basis={off_tps_ceiling_live_basis:.1f} (ref {REF_OFFICIAL_TPS}, +10 {PLUS10})", flush=True)
    print(f"  self_test={out['self_test']['passes']}  wrote {jp}", flush=True)

    if not args.no_wandb:
        try:
            import wandb
            run = wandb.init(project="gemma-challenge-senpai", entity="wandb-applied-ai-team",
                             name=args.wandb_name, group=args.wandb_group, job_type="analysis",
                             config={"pr": 676, "kind": "gemv-hbm-roofline-ceiling", "agent": "denken",
                                     "analysis_only": True, "official_tps": 0, "fires": False,
                                     **{f"shape_{k}": v for k, v in SHAPES.items()}})
            flat = {
                "analysis_only": 1, "official_tps": 0, "fires": 0,
                "gemv_achieved_bw_gbps": full_bw, "gemv_pct_of_hbm_peak": gemv_pct,
                "gemv_pct_of_spec": 100 * full_bw / A10G_SPEC_BW_GBPS,
                "body_achieved_bw_gbps": body_bw, "body_pct_of_read_peak": body_pct,
                "body_achieved_bw_gbps_bf16": body_bw_bf16, "dequant_alu_tax_frac": dequant_tax,
                "dominant_achieved_bw_gbps": dom_bw, "dominant_pct_of_read_peak": dom_pct,
                "dominant_dequant_alu_tax_frac": dom_dequant_tax,
                "lm_head_pct_of_read_peak": int4["lm_head"]["f_vs_read"] * 100,
                "m_invariance_m8_over_m1": m_inv if m_inv else 0.0,
                "empirical_read_peak_gbps": READ, "empirical_copy_peak_gbps": peak["bw_copy_gbps"],
                "read_peak_pct_of_spec": 100 * READ / A10G_SPEC_BW_GBPS,
                "body_bytes_gb": bm["body_bytes"] / 1e9, "lm_head_bytes_gb": bm["lm_head_bytes"] / 1e9,
                "full_gemv_bytes_gb": bm["full_gemv_bytes"] / 1e9,
                "inloop_bucket_bw_gbps": inloop_bw, "inloop_bucket_pct_of_read_peak": 100 * inloop_bw / READ,
                "iso_vs_inloop_gap_us": iso_vs_inloop_gap_us,
                "isolated_full_gemv_us": full_us, "d674_matmul_bucket_us": inloop_bucket_us,
                "local_tps_ceiling": local_tps_ceiling, "official_equiv_tps_ceiling": off_tps_ceiling,
                "official_equiv_tps_ceiling_live_basis": off_tps_ceiling_live_basis,
                "ceiling_speedup_if_kernel_hit_peak": speedup,
                "ceiling_clears_ref": int(off_tps_ceiling > REF_OFFICIAL_TPS),
                "ceiling_clears_plus10": int(off_tps_ceiling > PLUS10),
                "ceiling_clears_plus10_live_basis": int(off_tps_ceiling_live_basis > PLUS10),
                "headroom_byte_identical_realizable": 0,
                "headroom_margin_pct_below_peak": 100 - body_pct,
                "self_test_passes": int(out["self_test"]["passes"]),
                "verdict_at_wall": int(verdict == "GEMV_AT_HBM_WALL"),
            }
            wandb.log(flat)
            for k, v in flat.items():
                run.summary[k] = v
            run.summary["verdict"] = verdict
            out["wandb_run_id"] = run.id
            with open(jp, "w") as f:
                json.dump(out, f, indent=2)
            print(f"[wandb] {run.id} logged", flush=True)
            wandb.finish()
        except Exception as e:
            print(f"[wandb] FAILED: {e}", flush=True)

    return 0 if out["self_test"]["passes"] else 1


if __name__ == "__main__":
    sys.exit(main())
