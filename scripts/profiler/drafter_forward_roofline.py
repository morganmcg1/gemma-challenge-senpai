#!/usr/bin/env python
"""Drafter-forward roofline + 7-pass decomposition (PR #75).

WHAT THIS MEASURES
------------------
The deployed `fa2sw_precache_kenyan` stack speculative-decodes with a Gemma4 MTP
drafter at `num_speculative_tokens=7` (manifest SPECULATIVE_CONFIG). wirbel #69
recomposed the decode step (attention harvested to 7.6%), which makes the
**drafter forward (~15.5-18.1%) the #2 decode block after the 53% verify-GEMM**
(denken #68, merged) -- and it has had no roofline verdict.

stark #70 ("int4 drafter weights") assumes the drafter is **weight-bandwidth-bound
at M=1 x K=7**, so cutting weight bytes 4x (bf16 -> int4) would speed it up. This
script is the decisive check of that premise, in the same Step-0 roofline form as
denken #68 (verify_gemm_roofline.py).

THE DRAFTER (gemma-4-E4B-it-qat-q4_0-unquantized-assistant, ft-v1-epoch_001)
---------------------------------------------------------------------------
A Gemma4 MTP head, all weights **bf16** (NOT int4 -- this is exactly what stark
would change). 78.78M params / 157.3MB total, but **85% (134MB) is the tied
[262144, 256] embed/lm_head table, which is only single-row gathered per pass,
NOT read densely.** Architecture (config.json):
  hidden_size=256 (drafter internal), backbone_hidden_size=2560,
  num_hidden_layers=4 (3 sliding + 1 full), intermediate_size=2048,
  num_attention_heads=4, head_dim=256 (sliding)/512 (full, global),
  num_kv_heads=2, ALL 4 layers KV-shared (Q-only), vocab=262144,
  use_ordered_embeddings=True (centroid sparse sampler: num_centroids=2048,
  centroid_intermediate_top_k=32).

Per draft pass, `Gemma4MultiTokenPredictor.forward` runs these **dense** weight
GEMMs at M=1 (read each weight once to produce 1 row):
  pre_projection 5120->256, then x4 layers {q_proj, o_proj, gate_up, down},
  then post_projection 256->2560; plus the centroid sampler GEMM 256->2048.
= 19 tiny (256-wide) GEMMs/pass, ~25MB dense weight bytes/pass, x7 = ~175MB/step.
The K=7 passes are **strictly sequential** (constant_draft_positions; each pass
consumes the previous pass's sampled draft token), so drafter latency = 7 x
per-pass latency.

DEPLOYED EXECUTION MODE (from the submission code, not assumed)
--------------------------------------------------------------
ONEGRAPH=1 (manifest): the whole 7-pass propose() is captured into ONE CUDA graph
and served via `graph.replay()` (sitecustomize.py propose_onegraph), and the
centroid sampler is separately CUDA-graphed (gemma4.py _setup_centroids_cuda_graphs,
sizes 1..64). So the deployed drafter forward is **launch-free** -- it does NOT pay
the ~55us/call eager launch floor #68 measured. The launch-free CUDA-graph timing
here is therefore the deployed-representative basis; eager is reported only as the
without-onegraph contrast.

THE ROOFLINE QUESTION (three regimes, not two)
----------------------------------------------
At M=1 every drafter GEMM has arithmetic intensity ~= 1 FLOP/byte (bf16: 2 FLOP /
2 weight-bytes), far below the A10G ridge (~117). The naive roofline says
"bandwidth-bound", but that is only decisive if the kernel actually SATURATES HBM.
At hidden=256, M=1, each GEMM is a tiny GEMV that may be **latency/occupancy-bound**
(time set by kernel-issue + memory latency, not by weight bytes). The discriminator
is the *achieved* %HBM peak:
  - achieved %HBM peak HIGH (near peak) -> genuinely bandwidth-bound -> int4 cuts
    weight bytes ~4x -> drafter forward ~4x faster -> stark #70 premise VALIDATED.
  - achieved %HBM peak LOW (far below peak) -> latency/occupancy-bound -> int4
    keeps ~same time (bytes are not the bottleneck) -> no TPS win -> stark #70
    premise REFUTED (flag for stark before he builds it).
Primary metric: `drafter_forward_pct_hbm_peak_at_M1K7`.

This is a PURE isolated-kernel microbenchmark (#68 methodology): exact drafter GEMM
shapes + real bf16 weights, synthetic [M, in] activations, fixed weights, M swept.
No serve-path change, no token-stream change, no HF Job -> lossless by construction.
JSON dump + optional W&B (group drafter-forward-roofline).
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
import time

# Must be set before importing torch. The container exposes one A10G as index 0
# but inherits CUDA_VISIBLE_DEVICES=5 (host id); see project_local_a10g_gpu_env.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch  # noqa: E402

DEFAULT_DRAFTER = "/tmp/qat-assistant"

# ---- A10G (AWS g5, GA102, sm_86) roofline ceilings (identical to #68) -------
A10G_HBM_GBS = 600.0
A10G_FP16_TENSOR_TFLOPS = 70.0    # A10G dense FP16/BF16 tensor (datasheet)
A10_FP16_TENSOR_TFLOPS = 125.0    # data-center A10 (reference only)
# stark #70 int4 hypothetical: 4-bit weights + fp16 group scales.
INT4_WEIGHT_BITS = 4
INT4_SCALE_BYTES = 2
INT4_GROUP_SIZE = 32
BF16_BYTES = 2.0


# --------------------------------------------------------------------------- #
# Drafter weight introspection: parse the real safetensors header for exact   #
# GEMM shapes, then build the per-pass GEMM multiset.                          #
# --------------------------------------------------------------------------- #
def read_safetensors_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def load_tensor(path: str, name: str) -> torch.Tensor:
    """Load a single tensor from a safetensors file (bf16-aware, no extra deps)."""
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def build_drafter_gemms(drafter_dir: str):
    """Return (gemms, per_pass_multiset).

    gemms: list of dicts {role, in, out, count, module} -- one per UNIQUE
           (role-family, in, out); module is a real-weight bf16 nn.Linear on cuda.
    per_pass_multiset: list of (in, out) repeated per the real per-pass GEMM count.
    """
    st_path = os.path.join(drafter_dir, "model.safetensors")
    hdr = read_safetensors_header(st_path)

    def shape_of(name):
        return tuple(hdr[name]["shape"]) if name in hdr else None

    # Discover layer count + per-layer shapes from the header.
    layer_ids = sorted({
        int(k.split(".layers.")[1].split(".")[0])
        for k in hdr if ".layers." in k
    })

    specs = []  # (role, in_features, out_features, weight_tensor_or_None)

    # pre_projection: Linear(2*backbone -> hidden); weight [out, in]
    w = load_tensor(st_path, "pre_projection.weight")
    specs.append(("pre_projection", w.shape[1], w.shape[0], w))

    for i in layer_ids:
        qw = load_tensor(st_path, f"model.layers.{i}.self_attn.q_proj.weight")
        specs.append((f"layer{i}.q_proj", qw.shape[1], qw.shape[0], qw))
        ow = load_tensor(st_path, f"model.layers.{i}.self_attn.o_proj.weight")
        specs.append((f"layer{i}.o_proj", ow.shape[1], ow.shape[0], ow))
        # gate_up fused (vLLM MergedColumnParallelLinear): [2*inter, hidden]
        gw = load_tensor(st_path, f"model.layers.{i}.mlp.gate_proj.weight")
        uw = load_tensor(st_path, f"model.layers.{i}.mlp.up_proj.weight")
        guw = torch.cat([gw, uw], dim=0)
        specs.append((f"layer{i}.gate_up", guw.shape[1], guw.shape[0], guw))
        dw = load_tensor(st_path, f"model.layers.{i}.mlp.down_proj.weight")
        specs.append((f"layer{i}.down_proj", dw.shape[1], dw.shape[0], dw))

    # post_projection: Linear(hidden -> backbone)
    w = load_tensor(st_path, "post_projection.weight")
    specs.append(("post_projection", w.shape[1], w.shape[0], w))

    # centroid sampler GEMM (per-pass greedy sampling): Linear(hidden -> num_centroids)
    cw_name = "masked_embedding.centroids.weight"
    if cw_name in hdr:
        cw = load_tensor(st_path, cw_name)
        specs.append(("centroids_sampler", cw.shape[1], cw.shape[0], cw))

    # Build real-weight bf16 modules; bucket unique (role-family, in, out).
    def family(role):
        # collapse layer index so sliding/full variants bucket by shape
        return role.split(".", 1)[1] if "." in role else role

    per_pass_multiset = [(s[1], s[2]) for s in specs]

    uniq: dict[tuple, dict] = {}
    for role, inn, out, wt in specs:
        key = (family(role), inn, out)
        if key not in uniq:
            lin = torch.nn.Linear(inn, out, bias=False).to("cuda", torch.bfloat16)
            with torch.no_grad():
                lin.weight.copy_(wt.to("cuda", torch.bfloat16))
            uniq[key] = {"role": family(role), "in": inn, "out": out,
                         "count": 0, "module": lin, "example": role}
        uniq[key]["count"] += 1
    return list(uniq.values()), per_pass_multiset


# --------------------------------------------------------------------------- #
# Timing (verbatim methodology from #68 verify_gemm_roofline.py)              #
# --------------------------------------------------------------------------- #
def time_gemm(module, M, in_features, iters, warmup):
    """Median GPU ms for `module(x)` over `iters` EAGER calls (carries the
    ~55us/call launch+dispatch floor -- constant in M, the contrast to graph)."""
    x = torch.randn(M, in_features, device="cuda", dtype=torch.bfloat16)
    ev = lambda: torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        for _ in range(warmup):
            module(x)
        torch.cuda.synchronize()
        e0 = [ev() for _ in range(iters)]
        e1 = [ev() for _ in range(iters)]
        torch.cuda.synchronize()
        for i in range(iters):
            e0[i].record()
            module(x)
            e1[i].record()
        torch.cuda.synchronize()
    ms = [e0[i].elapsed_time(e1[i]) for i in range(iters)]
    return statistics.median(ms)


def time_gemm_graph(module, M, in_features, iters, warmup):
    """Launch-free per-call ms via CUDA-graph replay (the true kernel time, and
    the deployed onegraph-representative basis). Falls back to eager on failure."""
    x = torch.randn(M, in_features, device="cuda", dtype=torch.bfloat16)
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(5):
                module(x)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(g):
            module(x)
        for _ in range(max(10, warmup)):
            g.replay()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        ms = e0.elapsed_time(e1) / iters
        del g
        return ms, True
    except Exception as exc:  # noqa: BLE001
        print(f"[drafter-roofline]   graph capture failed (M={M}, in={in_features}): "
              f"{exc!r}; eager fallback", flush=True)
        return time_gemm(module, M, in_features, iters, warmup), False


def time_pass_chain_graph(modules_in_order, M, iters, warmup):
    """Launch-free latency of the WHOLE per-pass GEMM chain captured in ONE CUDA
    graph (mirrors the deployed onegraph: many tiny sequential kernels, one
    replay). modules_in_order: list of (module, in_features). Each GEMM gets a
    persistent [M, in] input buffer; we run them back-to-back inside one capture.
    Returns (ms_per_pass_graph, ms_per_pass_eager, captured)."""
    bufs = [torch.randn(M, inf, device="cuda", dtype=torch.bfloat16)
            for (_, inf) in modules_in_order]

    def run_chain():
        for (mod, _), b in zip(modules_in_order, bufs):
            mod(b)

    # eager chain (with launch floor)
    with torch.inference_mode():
        for _ in range(warmup):
            run_chain()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            run_chain()
        e1.record()
        torch.cuda.synchronize()
        ms_eager = e0.elapsed_time(e1) / iters

    # graphed chain (launch-free)
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(5):
                run_chain()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(g):
            run_chain()
        for _ in range(max(10, warmup)):
            g.replay()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        ms_graph = e0.elapsed_time(e1) / iters
        del g
        return ms_graph, ms_eager, True
    except Exception as exc:  # noqa: BLE001
        print(f"[drafter-roofline]   chain graph capture failed: {exc!r}", flush=True)
        return ms_eager, ms_eager, False


def roofline_metrics(in_f, out_f, M, ms, peak_tflops, w_bytes_per_elem=BF16_BYTES,
                     group_size=INT4_GROUP_SIZE):
    """Roofline row for a GEMM. Reports the CURRENT (bf16) bytes/AI and the
    int4-HYPOTHETICAL (stark #70) bytes so the byte-cut ratio is explicit."""
    t = ms / 1000.0
    flops = 2.0 * M * out_f * in_f
    w_bytes = w_bytes_per_elem * out_f * in_f
    act_bytes = 2.0 * M * in_f
    out_bytes = 2.0 * M * out_f
    total_bytes = w_bytes + act_bytes + out_bytes
    # stark int4 hypothetical weight bytes (same GEMM, 4-bit packed + fp16 scales)
    w_bytes_int4 = (INT4_WEIGHT_BITS / 8.0) * out_f * in_f + \
        INT4_SCALE_BYTES * out_f * math.ceil(in_f / group_size)
    total_bytes_int4 = w_bytes_int4 + act_bytes + out_bytes
    gflops_s = flops / t / 1e9
    gbytes_s = total_bytes / t / 1e9
    return {
        "M": M, "in": in_f, "out": out_f, "t_us": ms * 1000.0,
        "flops": flops, "w_bytes": w_bytes, "act_bytes": act_bytes,
        "out_bytes": out_bytes, "total_bytes": total_bytes,
        "w_bytes_int4": w_bytes_int4, "total_bytes_int4": total_bytes_int4,
        "int4_byte_ratio": total_bytes_int4 / total_bytes,
        "gflops_s": gflops_s, "gbytes_s": gbytes_s,
        "ai_flop_per_byte": flops / total_bytes,
        "pct_hbm_peak": 100.0 * gbytes_s / A10G_HBM_GBS,
        "pct_compute_peak": 100.0 * gflops_s / (peak_tflops * 1000.0),
    }


def compute_ceiling_probe(module, in_f, out_f, M_list, iters, warmup):
    best = {"gflops_s": 0.0}
    rows = []
    for M in M_list:
        ms, graphed = time_gemm_graph(module, M, in_f, max(50, iters // 2), warmup)
        g = 2.0 * M * out_f * in_f / (ms / 1000.0) / 1e9
        rows.append({"M": M, "t_us": ms * 1000.0, "gflops_s": g, "graphed": graphed})
        if g > best["gflops_s"]:
            best = {"M": M, "t_us": ms * 1000.0, "gflops_s": g}
    return best, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter-dir", default=DEFAULT_DRAFTER)
    ap.add_argument("--m-sweep", default="1,2,4,8",
                    help="draft row counts; M=1 is the deployed width-1 pass")
    ap.add_argument("--ceiling-m", default="256,512,1024,2048",
                    help="large M to read off the realizable FP16 tensor peak")
    ap.add_argument("--k", type=int, default=7, help="num_speculative_tokens (deployed=7)")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--decode-step-ms", type=float, default=11.6,
                    help="reference int4 verify decode-step latency (ms); #51/#68")
    ap.add_argument("--drafter-budget-pct-lo", type=float, default=15.5,
                    help="wirbel #69-corrected drafter-forward share of decode step (low)")
    ap.add_argument("--drafter-budget-pct-hi", type=float, default=18.1)
    ap.add_argument("--frontier-tps", type=float, default=481.53)
    ap.add_argument("--output", default="research/spec_cost_model/drafter_forward_roofline.json")
    ap.add_argument("--introspect-only", action="store_true")
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="drafter-forward-roofline")
    ap.add_argument("--wandb_name", default="drafter-forward-roofline-bf16")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--log-only", default=None,
                    help="skip timing; load this JSON payload and log it to W&B "
                         "(run from a non-cwd dir / venv that has wandb installed)")
    args = ap.parse_args()

    if args.log_only:
        with open(args.log_only) as fh:
            payload = json.load(fh)
        _log_wandb(args, payload)
        return

    m_sweep = [int(x) for x in args.m_sweep.split(",") if x.strip()]
    ceiling_m = [int(x) for x in args.ceiling_m.split(",") if x.strip()]

    print(f"[drafter-roofline] device: {torch.cuda.get_device_name(0)}", flush=True)
    t0 = time.time()
    gemms, per_pass_multiset = build_drafter_gemms(args.drafter_dir)
    print(f"[drafter-roofline] built {len(gemms)} unique drafter GEMMs from "
          f"{args.drafter_dir} in {time.time()-t0:.1f}s; per-pass GEMM count="
          f"{len(per_pass_multiset)}", flush=True)
    print("[drafter-roofline] per-pass GEMM multiset (role | in->out | xcount):", flush=True)
    for u in sorted(gemms, key=lambda u: (u["in"], u["out"])):
        wmb = BF16_BYTES * u["in"] * u["out"] / 1e6
        print(f"    {u['role']:>16s} | {u['in']:5d} -> {u['out']:5d} | x{u['count']:1d} "
              f"| {wmb:6.2f}MB bf16", flush=True)
    dense_w_mb = sum(BF16_BYTES * u["in"] * u["out"] * u["count"] for u in gemms) / 1e6
    print(f"[drafter-roofline] dense weight bytes/pass = {dense_w_mb:.2f}MB; "
          f"x{args.k} passes = {dense_w_mb*args.k:.1f}MB/step", flush=True)

    if args.introspect_only:
        print("[drafter-roofline] --introspect-only; exiting before timing.", flush=True)
        return

    # --- realizable compute-ceiling probe (largest GEMM by in*out) ----------
    big = max(gemms, key=lambda u: u["in"] * u["out"])
    print(f"[drafter-roofline] compute-ceiling probe on {big['role']} "
          f"{big['in']}->{big['out']} at M={ceiling_m}", flush=True)
    ceil_best, ceil_rows = compute_ceiling_probe(
        big["module"], big["in"], big["out"], ceiling_m, args.iters, args.warmup)
    measured_peak_tflops = ceil_best["gflops_s"] / 1000.0
    print(f"[drafter-roofline] measured realizable FP16/BF16 tensor peak ~= "
          f"{measured_peak_tflops:.1f} TFLOPS (M={ceil_best['M']}); "
          f"datasheet A10G={A10G_FP16_TENSOR_TFLOPS}", flush=True)
    peak_for_pct = max(measured_peak_tflops, 1.0)

    # --- per-GEMM M-sweep roofline ------------------------------------------
    rows = []
    for u in sorted(gemms, key=lambda u: (u["in"], u["out"])):
        for M in m_sweep:
            ms, graphed = time_gemm_graph(u["module"], M, u["in"], args.iters, args.warmup)
            ms_eager = time_gemm(u["module"], M, u["in"], max(40, args.iters // 4), args.warmup)
            rm = roofline_metrics(u["in"], u["out"], M, ms, peak_for_pct)
            rm.update({"role": u["role"], "count": u["count"], "graphed": graphed,
                       "t_us_eager": ms_eager * 1000.0})
            rows.append(rm)
        r1 = next(r for r in rows if r["role"] == u["role"] and r["in"] == u["in"]
                  and r["out"] == u["out"] and r["M"] == 1)
        print(f"[drafter-roofline] {u['role']:>16s} {u['in']:5d}->{u['out']:5d} x{u['count']:1d}  "
              f"M=1: {r1['t_us']:6.1f}us (eager {r1['t_us_eager']:6.1f})  "
              f"{r1['gbytes_s']:5.0f}GB/s ({r1['pct_hbm_peak']:4.1f}% BW)  "
              f"{r1['gflops_s']/1000:5.2f}TF/s ({r1['pct_compute_peak']:4.1f}% comp)  "
              f"AI={r1['ai_flop_per_byte']:.2f}", flush=True)

    # --- aggregate per-pass + x K decode step -------------------------------
    by_role = {}
    for r in rows:
        by_role.setdefault((r["role"], r["in"], r["out"], r["count"]), {})[r["M"]] = r
    agg = {}
    for M in m_sweep:
        tot_us = sum(d[M]["t_us"] * cnt for (role, i, o, cnt), d in by_role.items())
        tot_us_eager = sum(d[M]["t_us_eager"] * cnt for (role, i, o, cnt), d in by_role.items())
        tot_flops = sum(d[M]["flops"] * cnt for (role, i, o, cnt), d in by_role.items())
        tot_bytes = sum(d[M]["total_bytes"] * cnt for (role, i, o, cnt), d in by_role.items())
        tot_bytes_int4 = sum(d[M]["total_bytes_int4"] * cnt for (role, i, o, cnt), d in by_role.items())
        t = tot_us / 1e6
        agg[M] = {
            "M": M, "per_pass_gemm_us": tot_us, "per_pass_gemm_us_eager": tot_us_eager,
            "agg_gflops_s": tot_flops / t / 1e9, "agg_gbytes_s": tot_bytes / t / 1e9,
            "agg_ai": tot_flops / tot_bytes,
            "agg_pct_hbm_peak": 100.0 * (tot_bytes / t / 1e9) / A10G_HBM_GBS,
            "agg_pct_compute_peak": 100.0 * (tot_flops / t / 1e9) / (peak_for_pct * 1000.0),
            "decode_step_gemm_us": tot_us * args.k,
            "decode_step_gemm_us_eager": tot_us_eager * args.k,
            "int4_byte_ratio": tot_bytes_int4 / tot_bytes,
            "per_pass_total_bytes": tot_bytes, "per_pass_total_bytes_int4": tot_bytes_int4,
        }

    # --- full per-pass GEMM chain in ONE graph (deployed onegraph proxy) -----
    chain_modules = []
    for inn, out in per_pass_multiset:
        m = next(u["module"] for u in gemms if u["in"] == inn and u["out"] == out)
        chain_modules.append((m, inn))
    ms_chain_g, ms_chain_e, chain_cap = time_pass_chain_graph(
        chain_modules, 1, args.iters, args.warmup)
    chain_bytes = sum(BF16_BYTES * inn * out + 2.0 * 1 * inn + 2.0 * 1 * out
                      for inn, out in per_pass_multiset)
    chain = {
        "per_pass_chain_us_graph": ms_chain_g * 1000.0,
        "per_pass_chain_us_eager": ms_chain_e * 1000.0,
        "captured": chain_cap,
        "decode_step_chain_us_graph": ms_chain_g * 1000.0 * args.k,
        "decode_step_chain_us_eager": ms_chain_e * 1000.0 * args.k,
        "per_pass_total_bytes": chain_bytes,
        "chain_gbytes_s_graph": chain_bytes / (ms_chain_g / 1000.0) / 1e9,
        "chain_pct_hbm_peak_graph": 100.0 * (chain_bytes / (ms_chain_g / 1000.0) / 1e9) / A10G_HBM_GBS,
    }

    # --- VERDICT -------------------------------------------------------------
    a1 = agg[1]
    decode_step_us = args.decode_step_ms * 1000.0
    budget_lo_us = decode_step_us * args.drafter_budget_pct_lo / 100.0
    budget_hi_us = decode_step_us * args.drafter_budget_pct_hi / 100.0
    # primary metric: achieved %HBM peak of the deployed-representative (graph)
    # 7-pass drafter GEMM chain at M=1, K=7.
    primary_pct_hbm = chain["chain_pct_hbm_peak_graph"]
    ridge_measured = peak_for_pct * 1e12 / (A10G_HBM_GBS * 1e9)
    if primary_pct_hbm >= 50.0:
        regime = "BANDWIDTH-BOUND"
        stark_verdict = "VALIDATED: int4 drafter weights should help (~4x fewer weight bytes)."
    elif primary_pct_hbm < 25.0:
        regime = "LATENCY/OCCUPANCY-BOUND"
        stark_verdict = ("REFUTED: drafter GEMMs are NOT bandwidth-saturated at M=1; "
                         "int4 weights cut bytes but not the latency-floored kernel time "
                         "-> no TPS win. FLAG FOR STARK #70.")
    else:
        regime = "PARTIALLY BANDWIDTH-BOUND"
        stark_verdict = ("MIXED: partial bandwidth pressure; int4 may give a fraction of the "
                         "naive 4x. Quantify before building.")

    peak_mem_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print("\n[drafter-roofline] ===== PER-PASS DRAFTER GEMM COST vs M (launch-free graph) =====", flush=True)
    print("  M | per_pass_gemm_us (graph/eager) | agg GB/s (%BW) | agg TF/s (%comp) | AI", flush=True)
    for M in sorted(m_sweep):
        a = agg[M]
        print(f"  {M:2d} | {a['per_pass_gemm_us']:7.1f} / {a['per_pass_gemm_us_eager']:7.1f} | "
              f"{a['agg_gbytes_s']:5.0f} ({a['agg_pct_hbm_peak']:4.1f}%) | "
              f"{a['agg_gflops_s']/1000:5.2f} ({a['agg_pct_compute_peak']:4.1f}%) | {a['agg_ai']:.2f}",
              flush=True)
    print(f"\n[drafter-roofline] full per-pass GEMM CHAIN (one graph, K-pass proxy): "
          f"{chain['per_pass_chain_us_graph']:.1f}us graph / {chain['per_pass_chain_us_eager']:.1f}us eager "
          f"-> {chain['chain_gbytes_s_graph']:.0f} GB/s ({chain['chain_pct_hbm_peak_graph']:.1f}% HBM peak)",
          flush=True)
    print(f"[drafter-roofline] decode-step drafter GEMM (chain x{args.k}): "
          f"{chain['decode_step_chain_us_graph']:.0f}us graph / {chain['decode_step_chain_us_eager']:.0f}us eager",
          flush=True)
    print(f"[drafter-roofline] drafter-forward budget ({args.drafter_budget_pct_lo}-"
          f"{args.drafter_budget_pct_hi}% of {args.decode_step_ms}ms step) = "
          f"{budget_lo_us:.0f}-{budget_hi_us:.0f}us", flush=True)
    print(f"\n[drafter-roofline] PRIMARY METRIC drafter_forward_pct_hbm_peak_at_M1K7 = "
          f"{primary_pct_hbm:.1f}%", flush=True)
    print(f"[drafter-roofline] regime: {regime}", flush=True)
    print(f"[drafter-roofline] ridge(measured) = {ridge_measured:.0f} FLOP/byte; "
          f"drafter AI @ M=1 = {a1['agg_ai']:.2f}", flush=True)
    print(f"[drafter-roofline] STARK #70 PREMISE: {stark_verdict}", flush=True)
    print(f"[drafter-roofline] peak GPU mem: {peak_mem_gib:.2f} GiB", flush=True)

    verdict = {
        "primary_metric_name": "drafter_forward_pct_hbm_peak_at_M1K7",
        "drafter_forward_pct_hbm_peak_at_M1K7": primary_pct_hbm,
        "regime": regime,
        "stark70_premise_verdict": stark_verdict,
        "agg_ai_at_M1": a1["agg_ai"],
        "ridge_measured_flop_per_byte": ridge_measured,
        "agg_pct_hbm_peak_at_M1_isolated": a1["agg_pct_hbm_peak"],
        "agg_pct_compute_peak_at_M1": a1["agg_pct_compute_peak"],
        "per_pass_gemm_us_graph": a1["per_pass_gemm_us"],
        "per_pass_chain_us_graph": chain["per_pass_chain_us_graph"],
        "decode_step_chain_us_graph": chain["decode_step_chain_us_graph"],
        "decode_step_chain_us_eager": chain["decode_step_chain_us_eager"],
        "drafter_budget_lo_us": budget_lo_us, "drafter_budget_hi_us": budget_hi_us,
        "int4_byte_ratio_at_M1": a1["int4_byte_ratio"],
        "measured_peak_tflops": measured_peak_tflops,
        "k": args.k,
    }

    payload = {
        "config": {
            "drafter_dir": args.drafter_dir, "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0), "m_sweep": m_sweep,
            "ceiling_m": ceiling_m, "k": args.k, "iters": args.iters,
            "warmup": args.warmup, "decode_step_ms": args.decode_step_ms,
            "drafter_budget_pct": [args.drafter_budget_pct_lo, args.drafter_budget_pct_hi],
            "frontier_tps": args.frontier_tps, "A10G_HBM_GBS": A10G_HBM_GBS,
            "A10G_FP16_TENSOR_TFLOPS": A10G_FP16_TENSOR_TFLOPS,
            "dense_weight_mb_per_pass": dense_w_mb,
            "dense_weight_mb_per_step": dense_w_mb * args.k,
            "drafter_dtype": "bfloat16 (unquantized; stark #70 would make int4)",
            "peak_gpu_mem_gib": peak_mem_gib,
            "note": "isolated bf16 GEMM timing, real drafter weights, synthetic [M,in] "
                    "activations; lossless. Drafter linears are unquantized "
                    "(quant_config=None) -> F.linear/cuBLAS, identical to the served path. "
                    "Deployed drafter is launch-free (ONEGRAPH=1 + centroid CUDA graphs).",
        },
        "gemms": [{"role": u["role"], "in": u["in"], "out": u["out"], "count": u["count"]}
                  for u in gemms],
        "per_pass_multiset": per_pass_multiset,
        "rows": rows, "aggregate_by_M": agg, "chain": chain,
        "compute_ceiling": {"best": ceil_best, "rows": ceil_rows,
                            "measured_peak_tflops": measured_peak_tflops},
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[drafter-roofline] wrote {args.output}", flush=True)

    if not args.no_wandb:
        try:
            _log_wandb(args, payload)
        except Exception as exc:  # noqa: BLE001
            print(f"[drafter-roofline] W&B logging failed: {exc!r}", flush=True)

    gc.collect()
    torch.cuda.empty_cache()


def _log_wandb(args, payload):
    import wandb
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type="profiling", config=payload["config"])
    cols = ["role", "M", "in", "out", "count", "t_us", "t_us_eager", "gflops_s",
            "gbytes_s", "ai_flop_per_byte", "pct_hbm_peak", "pct_compute_peak"]
    tbl = wandb.Table(columns=cols)
    for r in payload["rows"]:
        tbl.add_data(r["role"], r["M"], r["in"], r["out"], r["count"], r["t_us"],
                     r["t_us_eager"], r["gflops_s"], r["gbytes_s"],
                     r["ai_flop_per_byte"], r["pct_hbm_peak"], r["pct_compute_peak"])
    run.log({"drafter_roofline_table": tbl})
    for M in sorted(payload["aggregate_by_M"], key=int):
        a = payload["aggregate_by_M"][M]
        run.log({"M": a["M"], "per_pass_gemm_us": a["per_pass_gemm_us"],
                 "agg_gbytes_s": a["agg_gbytes_s"], "agg_pct_hbm_peak": a["agg_pct_hbm_peak"],
                 "agg_pct_compute_peak": a["agg_pct_compute_peak"], "agg_ai": a["agg_ai"]})
    run.summary.update({k: v for k, v in payload["verdict"].items() if v is not None})
    run.summary.update({"chain_per_pass_us_graph": payload["chain"]["per_pass_chain_us_graph"],
                        "chain_per_pass_us_eager": payload["chain"]["per_pass_chain_us_eager"]})
    run.finish()
    print(f"[drafter-roofline] W&B run: {run.url}", flush=True)


if __name__ == "__main__":
    main()
