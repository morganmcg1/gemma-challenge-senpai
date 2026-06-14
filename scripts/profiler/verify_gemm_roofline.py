#!/usr/bin/env python
"""Isolated int4-Marlin verify-GEMM roofline + M-sweep audit (PR #68).

WHAT THIS MEASURES
------------------
The deployed `fa2sw_precache_kenyan` stack speculative-decodes with an MTP drafter
at `num_speculative_tokens=7`, so every verify step forwards **M = K+1 = 8** query
positions for one sequence through the int4 W4A16 (compressed-tensors / Marlin)
weight GEMMs. #30 profiling attributes **53.2% of the decode step** to this
verify-GEMM block; #51/#28/#37 mapped the whole-forward staircase (flat to M~=32,
Marlin 16-row tile cliffs at M=33/49).

This script answers the ONE roofline question the prior whole-forward profiling
could not isolate: **at M=8, is the int4 Marlin verify-GEMM weight-bandwidth-bound
(each weight tile read serves only 8 rows, so verifying MORE candidate rows per
weight-read is nearly free until the tile cliff / compute roofline), or already
compute/tile-bound (no free headroom)?**

It is a PURE isolated-kernel microbenchmark: it loads the real quantized model,
extracts the actual Marlin linear submodules (qkv / o / gate_up / down) for each
decoder layer, and times `module(x)` with SYNTHETIC bf16 activations [M, in] while
sweeping M. Weights are fixed (the real packed int4 weights); only the activation
row count M changes. No drafter, no greedy gate, no serve-path change, no HF Job
-> lossless by construction (it never touches the emitted token stream).

ROOFLINE FRAMING (the crux)
---------------------------
Marlin W4A16 dequantizes the int4 weights to FP16 on-chip and runs FP16xFP16
tensor-core MACs (arXiv:2408.11743 Sec.3). So the relevant COMPUTE ceiling is the
A10G FP16 tensor peak (~70 TFLOPS dense; the 125 TFLOPS figure is the data-center
A10, not the A10G), NOT the int4 tensor peak (~280 TOPS) -- the int4 path is never
used; 4-bit is only a weight-STORAGE format that cuts HBM traffic. Bandwidth
ceiling is the ~600 GB/s GDDR6 HBM. For a W4A16 GEMM reading each int4 weight
(0.5 byte) once and contributing 2 FLOPs/row, arithmetic intensity ~= 4*M
FLOP/byte; ridge point ~= 70e12/600e9 ~= 117 FLOP/byte -> crossover at M ~= 29.
At M=8 AI ~= 32, ~3.6x below the ridge => expected deeply memory-bound.

OUTPUT
------
- per-(shape, M) roofline rows: achieved GFLOP/s, GB/s, arithmetic intensity,
  fraction of FP16 compute peak and of HBM bandwidth peak.
- aggregate verify-body GEMM cost per decode step (sum over all 42 layers'
  projections) vs M, and the MARGINAL GEMM cost per extra verified row (us/token).
- a measured compute-ceiling probe (largest GEMM at large M) to resolve the
  realizable FP16 tensor peak on THIS card instead of trusting a datasheet number.
- the M=8 bandwidth-vs-compute verdict.
JSON dump + optional W&B (group verify-gemm-m8-audit).
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

# Must be set before importing torch/vllm. See project_local_a10g_gpu_env memory /
# research/spec_cost_model/report.md: the container exposes one A10G as index 0 but
# inherits CUDA_VISIBLE_DEVICES=5 (host id) which makes torch.cuda unavailable; the
# flashinfer sampler JIT fails on the incomplete CUDA dev headers (curand.h).
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch

DEFAULT_MODEL = "google/gemma-4-E4B-it-qat-w4a16-ct"

# ---- A10G (AWS g5, GA102, sm_86) roofline ceilings -------------------------
# Memory bandwidth is unambiguous (GDDR6, 600 GB/s). The FP16 tensor peak differs
# between the data-center A10 (125 TFLOPS) and the AWS A10G (~70 TFLOPS, fewer
# enabled tensor cores); we ALSO measure the realizable peak empirically (a big-M
# GEMM) so the verdict does not hinge on the datasheet ambiguity.
A10G_HBM_GBS = 600.0
A10G_FP16_TENSOR_TFLOPS = 70.0    # A10G dense FP16/BF16 tensor (datasheet)
A10_FP16_TENSOR_TFLOPS = 125.0    # data-center A10 (reference only)
A10G_INT4_TENSOR_TOPS = 280.0     # IRRELEVANT for Marlin W4A16 (FP16 MACs); ref only
WEIGHT_BITS = 4
SCALE_BYTES = 2                   # fp16 group scales
DEFAULT_GROUP_SIZE = 32          # compressed-tensors uint4b8 group_size for this ckpt


def find_runner(obj, depth=0, seen=None):
    """Walk the in-process engine object graph to the GPUModelRunner."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    if seen is None:
        seen = set()
    if id(obj) in seen or depth > 12:
        return None
    seen.add(id(obj))
    if isinstance(obj, GPUModelRunner):
        return obj
    for attr in ("llm_engine", "engine_core", "engine", "model_executor", "executor",
                 "driver_worker", "worker", "model_runner", "core", "engines"):
        child = getattr(obj, attr, None)
        if child is not None:
            r = find_runner(child, depth + 1, seen)
            if r is not None:
                return r
    return None


def build_llm(model: str, max_ctx: int):
    from vllm import LLM
    return LLM(
        model=model,
        quantization="compressed-tensors",
        dtype="bfloat16",
        max_model_len=max(4096, max_ctx + 128),
        gpu_memory_utilization=0.90,
        max_num_batched_tokens=2048,
        max_num_seqs=1,
        enforce_eager=True,         # isolated module timing: no graph capture needed
        trust_remote_code=True,
        disable_log_stats=True,
        seed=0,
    )


def _size_in(m):
    return int(getattr(m, "input_size_per_partition", None) or getattr(m, "input_size"))


def _size_out(m):
    return int(getattr(m, "output_size_per_partition", None) or getattr(m, "output_size"))


def find_decoder_layers(model):
    """Return [(name, module), ...] for transformer decoder layers (have self_attn+mlp)."""
    layers = []
    for name, mod in model.named_modules():
        if hasattr(mod, "self_attn") and hasattr(mod, "mlp"):
            layers.append((name, mod))
    return layers


def collect_linears(parent):
    """vLLM quantized linear submodules directly under `parent` (qkv/o or gate_up/down)."""
    out = []
    for name, child in parent.named_children():
        if type(child).__name__.endswith("Linear") and (
            hasattr(child, "input_size") or hasattr(child, "input_size_per_partition")
        ):
            out.append((name, child))
    return out


def collect_gemm_instances(layers):
    """All weight-GEMM instances across layers as (role, in, out, module, layer_name)."""
    inst = []
    for lname, layer in layers:
        for parent_attr in ("self_attn", "mlp"):
            parent = getattr(layer, parent_attr, None)
            if parent is None:
                continue
            for cname, child in collect_linears(parent):
                inst.append((f"{parent_attr}.{cname}", _size_in(child), _size_out(child),
                             child, lname))
    return inst


def uniquify(instances):
    """Bucket GEMM instances by (role, in, out) -> representative module + count."""
    uniq: dict[tuple, dict] = {}
    for role, inn, out, module, lname in instances:
        key = (role, inn, out)
        if key not in uniq:
            uniq[key] = {"role": role, "in": inn, "out": out, "module": module,
                         "count": 0, "example_layer": lname}
        uniq[key]["count"] += 1
    return uniq


def time_gemm(module, M, in_features, iters, warmup):
    """Median GPU ms for `module(x)` over `iters` back-to-back EAGER calls.

    Eager timing carries a fixed ~55-60us/call launch+dispatch floor (kernel
    launch + Marlin workspace + Python). This floor dominates the small attention
    GEMMs at M=8 and inflates the absolute aggregate, but it is CONSTANT in M, so
    the M-marginal (a difference of totals) cancels it. Reported as a cross-check;
    the launch-free CUDA-graph time is the primary roofline basis."""
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
    """Launch-free per-call GEMM ms via CUDA-graph replay (the true kernel time).

    Captures one `module(x)` into a CUDA graph (the same mechanism vLLM uses to
    serve these layers) and times many replays. Graph launch is ~1-2us vs ~55us
    eager, so this strips the launch/dispatch floor and exposes the genuine
    compute/bandwidth-bound kernel time. Falls back to eager if capture fails."""
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
        print(f"[roofline]   graph capture failed (M={M}, in={in_features}): "
              f"{exc!r}; eager fallback", flush=True)
        return time_gemm(module, M, in_features, iters, warmup), False


def roofline_metrics(in_f, out_f, M, ms, group_size, peak_tflops):
    t = ms / 1000.0
    flops = 2.0 * M * out_f * in_f
    w_bytes = (WEIGHT_BITS / 8.0) * out_f * in_f + SCALE_BYTES * out_f * math.ceil(in_f / group_size)
    act_bytes = 2.0 * M * in_f
    out_bytes = 2.0 * M * out_f
    total_bytes = w_bytes + act_bytes + out_bytes
    gflops_s = flops / t / 1e9
    gbytes_s = total_bytes / t / 1e9
    return {
        "M": M, "in": in_f, "out": out_f, "t_us": ms * 1000.0,
        "flops": flops, "w_bytes": w_bytes, "act_bytes": act_bytes,
        "out_bytes": out_bytes, "total_bytes": total_bytes,
        "gflops_s": gflops_s, "gbytes_s": gbytes_s,
        "ai_flop_per_byte": flops / total_bytes,
        "ai_weight_only": flops / w_bytes,
        "pct_hbm_peak": 100.0 * gbytes_s / A10G_HBM_GBS,
        "pct_compute_peak": 100.0 * gflops_s / (peak_tflops * 1000.0),
    }


def compute_ceiling_probe(module, in_f, out_f, M_list, iters, warmup):
    """Time the largest GEMM at large M (launch-free) to read off the realizable
    FP16 tensor peak on THIS card -- resolves the A10G-70 vs A10-125 ambiguity."""
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
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--m-sweep", default="1,8,12,16,24,32,33,48,49",
                    help="verify row counts to sweep (M=8 is the deployed verify width)")
    ap.add_argument("--ceiling-m", default="256,512,1024",
                    help="large M values to read off the realizable FP16 tensor peak")
    ap.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--max-ctx", type=int, default=256)
    ap.add_argument("--decode-step-ms", type=float, default=11.6,
                    help="reference verify decode-step latency (ms) for %%-of-step; "
                         "#51 graph-mode int4 verify step at M~=8 ~= 11.6 ms")
    ap.add_argument("--frontier-tps", type=float, default=481.53,
                    help="deployed frontier output TPS for per-emitted-token budget context")
    ap.add_argument("--output", default="research/spec_cost_model/verify_gemm_roofline.json")
    ap.add_argument("--introspect-only", action="store_true",
                    help="load, print discovered GEMM shapes, and exit (no timing)")
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--wandb_group", default="verify-gemm-m8-audit")
    ap.add_argument("--wandb_name", default="verify-gemm-roofline-int4")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    m_sweep = [int(x) for x in args.m_sweep.split(",") if x.strip()]
    ceiling_m = [int(x) for x in args.ceiling_m.split(",") if x.strip()]

    t0 = time.time()
    print(f"[roofline] building LLM {args.model} ...", flush=True)
    llm = build_llm(args.model, args.max_ctx)
    runner = find_runner(llm)
    if runner is None:
        raise RuntimeError("could not locate GPUModelRunner")
    model = runner.model
    print(f"[roofline] LLM ready in {time.time()-t0:.1f}s", flush=True)

    layers = find_decoder_layers(model)
    instances = collect_gemm_instances(layers)
    uniq = uniquify(instances)
    print(f"[roofline] decoder layers found: {len(layers)}; "
          f"GEMM instances: {len(instances)}; unique shapes: {len(uniq)}", flush=True)
    print("[roofline] discovered unique verify-GEMM shapes (role | in -> out | count | class):",
          flush=True)
    for key, u in sorted(uniq.items(), key=lambda kv: (-kv[1]["count"], kv[1]["role"])):
        print(f"    {u['role']:>18s} | {u['in']:5d} -> {u['out']:6d} | x{u['count']:2d} | "
              f"{type(u['module']).__name__}", flush=True)

    if args.introspect_only:
        print("[roofline] --introspect-only set; exiting before timing.", flush=True)
        return

    # --- realizable compute-ceiling probe on the largest GEMM (gate_up) ---------
    big = max(uniq.values(), key=lambda u: u["in"] * u["out"])
    print(f"[roofline] compute-ceiling probe on largest GEMM "
          f"{big['role']} {big['in']}->{big['out']} at M={ceiling_m}", flush=True)
    ceil_best, ceil_rows = compute_ceiling_probe(
        big["module"], big["in"], big["out"], ceiling_m, args.iters, args.warmup)
    measured_peak_tflops = ceil_best["gflops_s"] / 1000.0
    print(f"[roofline] measured realizable FP16 tensor peak ~= {measured_peak_tflops:.1f} "
          f"TFLOPS (at M={ceil_best['M']}); datasheet A10G={A10G_FP16_TENSOR_TFLOPS}, "
          f"A10={A10_FP16_TENSOR_TFLOPS}", flush=True)
    peak_for_pct = max(measured_peak_tflops, 1.0)

    # --- per-shape M-sweep + roofline ------------------------------------------
    rows = []
    for key, u in sorted(uniq.items(), key=lambda kv: (kv[1]["role"], kv[1]["in"], kv[1]["out"])):
        for M in m_sweep:
            ms, graphed = time_gemm_graph(u["module"], M, u["in"], args.iters, args.warmup)
            ms_eager = time_gemm(u["module"], M, u["in"], max(40, args.iters // 4), args.warmup)
            rm = roofline_metrics(u["in"], u["out"], M, ms, args.group_size, peak_for_pct)
            rm.update({"role": u["role"], "count": u["count"], "graphed": graphed,
                       "t_us_eager": ms_eager * 1000.0})
            rows.append(rm)
        r8 = next(r for r in rows if r["role"] == u["role"] and r["in"] == u["in"]
                  and r["out"] == u["out"] and r["M"] == 8)
        print(f"[roofline] {u['role']:>18s} {u['in']:5d}->{u['out']:6d} x{u['count']:2d}  "
              f"M=8: {r8['t_us']:7.1f}us  {r8['gflops_s']/1000:5.1f}TF/s "
              f"({r8['pct_compute_peak']:4.1f}% peak)  {r8['gbytes_s']:5.0f}GB/s "
              f"({r8['pct_hbm_peak']:4.1f}% BW)  AI={r8['ai_flop_per_byte']:5.1f}", flush=True)

    # --- aggregate verify-body GEMM cost per decode step vs M ------------------
    by_role = {}
    for r in rows:
        by_role.setdefault((r["role"], r["in"], r["out"], r["count"]), {})[r["M"]] = r
    agg = {}
    for M in m_sweep:
        tot_us = sum(d[M]["t_us"] * cnt for (role, i, o, cnt), d in by_role.items())
        tot_us_eager = sum(d[M]["t_us_eager"] * cnt for (role, i, o, cnt), d in by_role.items())
        tot_flops = sum(d[M]["flops"] * cnt for (role, i, o, cnt), d in by_role.items())
        tot_bytes = sum(d[M]["total_bytes"] * cnt for (role, i, o, cnt), d in by_role.items())
        t = tot_us / 1e6
        agg[M] = {
            "M": M, "total_gemm_us": tot_us, "total_gemm_us_eager": tot_us_eager,
            "agg_gflops_s": tot_flops / t / 1e9,
            "agg_gbytes_s": tot_bytes / t / 1e9,
            "agg_ai": tot_flops / tot_bytes,
            "agg_pct_hbm_peak": 100.0 * (tot_bytes / t / 1e9) / A10G_HBM_GBS,
            "agg_pct_compute_peak": 100.0 * (tot_flops / t / 1e9) / (peak_for_pct * 1000.0),
        }

    # marginal GEMM cost per extra verified row between consecutive sweep points
    marginal = {}
    ms_sorted = sorted(m_sweep)
    for i in range(1, len(ms_sorted)):
        a, b = ms_sorted[i - 1], ms_sorted[i]
        d_us = agg[b]["total_gemm_us"] - agg[a]["total_gemm_us"]
        per_row = d_us / (b - a)
        per_row_eager = (agg[b]["total_gemm_us_eager"] - agg[a]["total_gemm_us_eager"]) / (b - a)
        marginal[b] = {
            "from_M": a, "to_M": b, "delta_us": d_us, "us_per_row": per_row,
            "us_per_row_eager": per_row_eager,
            "pct_of_M8_gemm": 100.0 * per_row / agg[8]["total_gemm_us"] if 8 in agg else None,
            "pct_of_decode_step": 100.0 * per_row / (args.decode_step_ms * 1000.0),
            "pct_of_emitted_token_budget": 100.0 * per_row / (1e6 / args.frontier_tps),
        }

    ridge_measured = peak_for_pct * 1e12 / (A10G_HBM_GBS * 1e9)
    ridge_datasheet = A10G_FP16_TENSOR_TFLOPS * 1e12 / (A10G_HBM_GBS * 1e9)
    a8 = agg.get(8, {})
    verdict = {
        "operating_M": 8,
        "agg_ai_at_M8": a8.get("agg_ai"),
        "ridge_measured_flop_per_byte": ridge_measured,
        "ridge_datasheet_flop_per_byte": ridge_datasheet,
        "agg_pct_hbm_peak_at_M8": a8.get("agg_pct_hbm_peak"),
        "agg_pct_compute_peak_at_M8": a8.get("agg_pct_compute_peak"),
        "bandwidth_bound_at_M8": (a8.get("agg_ai", 1e9) < ridge_measured),
        "M8_to_M16_free": None,
        "measured_peak_tflops": measured_peak_tflops,
    }
    if 8 in agg and 16 in agg:
        verdict["M8_to_M16_free"] = {
            "gemm_us_M8": agg[8]["total_gemm_us"],
            "gemm_us_M16": agg[16]["total_gemm_us"],
            "delta_us": agg[16]["total_gemm_us"] - agg[8]["total_gemm_us"],
            "delta_pct": 100.0 * (agg[16]["total_gemm_us"] - agg[8]["total_gemm_us"]) / agg[8]["total_gemm_us"],
            "extra_rows_verified": 8,
        }

    peak_mem_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print("\n[roofline] ===== AGGREGATE VERIFY-GEMM COST vs M (launch-free CUDA-graph) =====", flush=True)
    print("  M | gemm_us (graph / eager) | agg GB/s (%BW) | agg TF/s (%peak) | AI", flush=True)
    for M in ms_sorted:
        a = agg[M]
        print(f"  {M:3d} | {a['total_gemm_us']:8.1f} / {a['total_gemm_us_eager']:8.1f} | "
              f"{a['agg_gbytes_s']:5.0f} ({a['agg_pct_hbm_peak']:4.1f}%) | {a['agg_gflops_s']/1000:5.1f} "
              f"({a['agg_pct_compute_peak']:4.1f}%) | {a['agg_ai']:5.1f}", flush=True)
    print("\n[roofline] marginal GEMM cost per extra verified row:", flush=True)
    for b in sorted(marginal):
        m = marginal[b]
        print(f"  M {m['from_M']:2d}->{m['to_M']:2d}: {m['us_per_row']:7.2f} us/row "
              f"({m['pct_of_decode_step']:4.2f}% of {args.decode_step_ms}ms step)", flush=True)
    print(f"\n[roofline] VERDICT @ M=8: AI={verdict['agg_ai_at_M8']:.1f} FLOP/byte vs ridge "
          f"{ridge_measured:.0f} -> {'BANDWIDTH-BOUND' if verdict['bandwidth_bound_at_M8'] else 'COMPUTE-BOUND'}; "
          f"{a8.get('agg_pct_hbm_peak', 0):.0f}% of HBM BW, "
          f"{a8.get('agg_pct_compute_peak', 0):.0f}% of FP16 compute peak", flush=True)
    if verdict["M8_to_M16_free"]:
        f = verdict["M8_to_M16_free"]
        print(f"[roofline] M=8->16 (verify 8 extra rows): +{f['delta_us']:.1f}us "
              f"({f['delta_pct']:+.1f}% GEMM) -> {'~FREE' if abs(f['delta_pct']) < 5 else 'NOT free'}",
              flush=True)
    print(f"[roofline] peak GPU mem: {peak_mem_gib:.2f} GiB", flush=True)

    payload = {
        "config": {
            "model": args.model, "vllm": __import__("vllm").__version__,
            "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
            "m_sweep": m_sweep, "ceiling_m": ceiling_m, "group_size": args.group_size,
            "iters": args.iters, "warmup": args.warmup, "max_ctx": args.max_ctx,
            "decode_step_ms": args.decode_step_ms, "frontier_tps": args.frontier_tps,
            "A10G_HBM_GBS": A10G_HBM_GBS,
            "A10G_FP16_TENSOR_TFLOPS": A10G_FP16_TENSOR_TFLOPS,
            "A10_FP16_TENSOR_TFLOPS": A10_FP16_TENSOR_TFLOPS,
            "A10G_INT4_TENSOR_TOPS_irrelevant": A10G_INT4_TENSOR_TOPS,
            "marlin_compute_dtype": "fp16 tensor-core MACs (int4 weight dequantized on-chip)",
            "peak_gpu_mem_gib": peak_mem_gib,
            "note": "isolated Marlin W4A16 GEMM timing; weights fixed, synthetic bf16 "
                    "activations; lossless (no token-stream change). GEMM shapes are "
                    "architecture-determined and identical to the PLE-folded deployed "
                    "osoi5 weights (folding changes values, not shapes/dtype).",
        },
        "unique_shapes": [
            {"role": u["role"], "in": u["in"], "out": u["out"], "count": u["count"],
             "class": type(u["module"]).__name__}
            for u in uniq.values()
        ],
        "rows": rows,
        "aggregate_by_M": agg,
        "marginal_per_row": marginal,
        "compute_ceiling": {"best": ceil_best, "rows": ceil_rows,
                            "measured_peak_tflops": measured_peak_tflops},
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[roofline] wrote {args.output}", flush=True)

    if not args.no_wandb:
        try:
            _log_wandb(args, payload)
        except Exception as exc:  # noqa: BLE001
            print(f"[roofline] W&B logging failed: {exc!r}", flush=True)

    del llm
    gc.collect()
    torch.cuda.empty_cache()


def _log_wandb(args, payload):
    import wandb
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type="profiling", config=payload["config"])
    cols = ["role", "M", "in", "out", "count", "t_us", "gflops_s", "gbytes_s",
            "ai_flop_per_byte", "pct_hbm_peak", "pct_compute_peak"]
    tbl = wandb.Table(columns=cols)
    for r in payload["rows"]:
        tbl.add_data(r["role"], r["M"], r["in"], r["out"], r["count"], r["t_us"],
                     r["gflops_s"], r["gbytes_s"], r["ai_flop_per_byte"],
                     r["pct_hbm_peak"], r["pct_compute_peak"])
    run.log({"roofline_table": tbl})
    # aggregate per-M line series
    for M in sorted(payload["aggregate_by_M"], key=int):
        a = payload["aggregate_by_M"][M]
        run.log({"M": a["M"], "total_gemm_us": a["total_gemm_us"],
                 "agg_gbytes_s": a["agg_gbytes_s"], "agg_gflops_s": a["agg_gflops_s"],
                 "agg_ai": a["agg_ai"], "agg_pct_hbm_peak": a["agg_pct_hbm_peak"],
                 "agg_pct_compute_peak": a["agg_pct_compute_peak"]})
    v = payload["verdict"]
    summary = {
        "verdict_bandwidth_bound_at_M8": bool(v["bandwidth_bound_at_M8"]),
        "agg_ai_at_M8": v["agg_ai_at_M8"],
        "ridge_measured_flop_per_byte": v["ridge_measured_flop_per_byte"],
        "agg_pct_hbm_peak_at_M8": v["agg_pct_hbm_peak_at_M8"],
        "agg_pct_compute_peak_at_M8": v["agg_pct_compute_peak_at_M8"],
        "measured_peak_tflops": v["measured_peak_tflops"],
        "peak_gpu_mem_gib": payload["config"]["peak_gpu_mem_gib"],
    }
    if v.get("M8_to_M16_free"):
        summary["M8_to_M16_delta_pct"] = v["M8_to_M16_free"]["delta_pct"]
        summary["M8_to_M16_delta_us"] = v["M8_to_M16_free"]["delta_us"]
    for b, m in payload["marginal_per_row"].items():
        summary[f"marginal_us_per_row_to_M{b}"] = m["us_per_row"]
    run.summary.update({k: val for k, val in summary.items() if val is not None})
    run.finish()
    print(f"[roofline] W&B run: {run.url}", flush=True)


if __name__ == "__main__":
    main()
