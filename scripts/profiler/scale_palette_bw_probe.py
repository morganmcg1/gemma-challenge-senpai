#!/usr/bin/env python
"""Phase-2 BW-critical-path gate: are int4-Marlin SCALE bytes on the verify-GEMM
bandwidth-critical path, or hidden by the weight-prefetch pipeline / L2? (PR #110)

THE GATE
--------
PR #104 proved the core-7 verify-GEMM FP16 scales (53.70 MB, g=128) take only
1,009 distinct values -> a lossless 10-bit palette saves ~37.5% (per-tensor 9-bit
~43%) of the scale bytes (Phase 1, scale_palette_lut.py). The build-or-kill
question is whether removing those scale bytes moves wall_tps at all, or whether
the scales are already overlapped/cached so the saving is invisible.

THE CONTROLLED PROBE
--------------------
Marlin int4 GEMM wall-time is value-INDEPENDENT (fixed work per element); it
depends only on (M, in, out, group_size). The group_size directly controls scale
bytes while leaving the int4 WEIGHT bytes byte-identical:

    weight bytes  = in*out*0.5                    (CONSTANT across group_size)
    scale bytes   = out*ceil(in/g)*2              (g=-1 per-channel -> ~zero;
                                                   g=128 deployed; g=64,32 -> more)

So timing the SAME synthetic GEMM at g in {-1(per-channel), 128, 64, 32}, M=8,
launch-free (CUDA-graph replay), isolates the wall-clock cost of scale-load:

    f = (T_g128 - T_perchannel) / T_g128   = scale-load share of the verify GEMM.

  * f ~= analytical byte share (~3%)  -> scales FULLY on the DRAM critical path -> YES
  * f ~= 0 (within noise)             -> scales hidden/cached                    -> NO  (clean negative)
  * 0 < f < byte share               -> partially overlapped/cached             -> PARTIAL

CAVEAT (conservative direction): smaller g also raises the group-boundary dequant
frequency (a compute/branch effect), so the measured slope/f is an UPPER BOUND on
the true scale-BANDWIDTH cost. If even this upper bound predicts a tiny palette
gain, the negative verdict is robust.

PALETTE WALL_TPS PREDICTION
---------------------------
The palette removes `save_frac` of the g128 scale bytes (0.375 global / 0.43
per-tensor). If scale-time is linear in scale-bytes (validated by the g32/g64/g128
points), the verify-body saving is save_frac*(T_g128 - T_perchannel), and:

    palette_wall_tps_gain_pct ~= 100 * save_frac * f * verify_gemm_frac

Isolated Marlin timing only; weights synthetic random; NO served-file change, NO
HF Job, NO token-stream change -> lossless by construction. JSON + W&B
(group scale-palette-lut).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import time

# Must be set before importing torch/vllm (container inherits host CUDA id; see
# verify_gemm_roofline.py / project_local_a10g_gpu_env).
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")     # use the cached snapshot

import torch  # noqa: E402

# Shape discovery uses the PUBLIC QAT model (same Gemma-4-E4B architecture as the
# deployed PLE-folded osoi5-v0-baked: folding changes weight VALUES, not GEMM
# dims). The deployed checkpoint itself will not load via vLLM's standard path (a
# baked-head vocab assertion), and we only need the served fused GEMM shapes +
# per-layer counts -- which are architecture-determined and identical. Built-in
# fidelity check: the aggregate g128 scale payload reproduced from these shapes is
# compared to the deployed 53.70 MB (#104). NOTE the public model is natively g32,
# so the real-module cross-check anchors the synthetic g32 point.
DEFAULT_MODEL = "google/gemma-4-E4B-it-qat-w4a16-ct"
PUBLIC_NATIVE_GROUP = 32

A10G_HBM_GBS = 600.0
VERIFY_GEMM_FRAC = 0.532       # #30: verify-GEMM = 53.2% of conc=1 decode step
WALL_TPS = 454.338             # lawine #90 locked linear-chain local reference
OFFICIAL_PROJ = 1.06019        # lawine #99 local->official projection multiplier
DEPLOYED_OFFICIAL = 481.53
# Phase-1 lossless palette savings (scale_palette_lut.py):
SAVE_FRAC_GLOBAL = 0.375
SAVE_FRAC_PERTENSOR = 0.43


# --------------------------------------------------------------------------- #
# model load + shape discovery (mirrors verify_gemm_roofline.py)
# --------------------------------------------------------------------------- #
def find_runner(obj, depth=0, seen=None):
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
    return LLM(model=model, quantization="compressed-tensors", dtype="bfloat16",
               max_model_len=max(4096, max_ctx + 128), gpu_memory_utilization=0.90,
               max_num_batched_tokens=2048, max_num_seqs=1, enforce_eager=True,
               trust_remote_code=True, disable_log_stats=True, seed=0)


def _size_in(m):
    return int(getattr(m, "input_size_per_partition", None) or getattr(m, "input_size"))


def _size_out(m):
    return int(getattr(m, "output_size_per_partition", None) or getattr(m, "output_size"))


def find_decoder_layers(model):
    return [(n, m) for n, m in model.named_modules()
            if hasattr(m, "self_attn") and hasattr(m, "mlp")]


def collect_gemm_instances(layers):
    inst = []
    for lname, layer in layers:
        for parent_attr in ("self_attn", "mlp"):
            parent = getattr(layer, parent_attr, None)
            if parent is None:
                continue
            for cname, child in parent.named_children():
                if type(child).__name__.endswith("Linear") and (
                    hasattr(child, "input_size") or hasattr(child, "input_size_per_partition")):
                    inst.append((f"{parent_attr}.{cname}", _size_in(child), _size_out(child), child, lname))
    return inst


def uniquify(instances):
    uniq = {}
    for role, inn, out, module, lname in instances:
        key = (role, inn, out)
        if key not in uniq:
            uniq[key] = {"role": role, "in": inn, "out": out, "module": module,
                         "count": 0, "example_layer": lname}
        uniq[key]["count"] += 1
    return uniq


# --------------------------------------------------------------------------- #
# launch-free timing
# --------------------------------------------------------------------------- #
def _graph_time(fn, iters, warmup, repeats):
    """Median-of-`repeats` launch-free per-call ms of `fn()` via CUDA-graph replay."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.inference_mode():
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(g):
        fn()
    for _ in range(max(10, warmup)):
        g.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(repeats):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        out.append(e0.elapsed_time(e1) / iters)
    del g
    return statistics.median(out), min(out), max(out)


def scale_bytes(in_f, out_f, g):
    n_groups = 1 if g == -1 else math.ceil(in_f / g)
    return out_f * n_groups * 2


def weight_bytes(in_f, out_f):
    return in_f * out_f * 0.5


# --------------------------------------------------------------------------- #
def main():
    import vllm.model_executor.layers.quantization.utils.marlin_utils as mu
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
    from vllm.scalar_type import scalar_types

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--m-sweep", default="1,8,16")
    ap.add_argument("--group-sweep", default="-1,128,64,32",
                    help="-1=per-channel(~zero scale bytes), 128=deployed, 64/32=more scale bytes")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-ctx", type=int, default=256)
    ap.add_argument("--decode-step-ms", type=float, default=11.6,
                    help="#51 graph-mode int4 verify step at M~=8 ~= 11.6 ms (for %%-of-step)")
    ap.add_argument("--output", default="research/scale_palette_lut/scale_palette_bw_probe.json")
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="scale-palette-lut")
    ap.add_argument("--wandb_name", default="wirbel/scale-palette-bw-probe-phase2")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    m_sweep = [int(x) for x in args.m_sweep.split(",") if x.strip()]
    g_sweep = [int(x) for x in args.group_sweep.split(",") if x.strip()]
    gate_M = 8 if 8 in m_sweep else m_sweep[0]
    dev = torch.device("cuda")

    t0 = time.time()
    print(f"[bwprobe] building LLM {args.model} ...", flush=True)
    llm = build_llm(args.model, args.max_ctx)
    runner = find_runner(llm)
    if runner is None:
        raise RuntimeError("could not locate GPUModelRunner")
    layers = find_decoder_layers(runner.model)
    uniq = uniquify(collect_gemm_instances(layers))
    print(f"[bwprobe] LLM ready in {time.time()-t0:.1f}s | decoder layers={len(layers)} | "
          f"unique verify-GEMM shapes={len(uniq)}", flush=True)
    for key, u in sorted(uniq.items(), key=lambda kv: (-kv[1]["count"], kv[1]["role"])):
        sc128 = scale_bytes(u["in"], u["out"], 128)
        wb = weight_bytes(u["in"], u["out"])
        print(f"    {u['role']:>14s} | {u['in']:5d}->{u['out']:6d} | x{u['count']:2d} | "
              f"wbytes={wb/1e6:6.2f}MB scale@g128={sc128/1e3:6.1f}KB ({100*sc128/wb:.2f}% of w)", flush=True)

    # ---- per-shape group-size sweep (synthetic, launch-free) -------------------
    rows = []
    real_xcheck = {}
    for key, u in sorted(uniq.items(), key=lambda kv: (kv[1]["role"], kv[1]["in"], kv[1]["out"])):
        in_f, out_f = u["in"], u["out"]
        w = (torch.randn(in_f, out_f, device=dev, dtype=torch.float16) * 0.05)
        zp = torch.empty(0, dtype=torch.int32, device=dev)
        ws = mu.marlin_make_workspace_new(dev)
        built = {}
        for g in g_sweep:
            if g != -1 and in_f % g != 0:
                continue
            w_ref, q_w, s, g_idx, sort_idx, _ = marlin_quantize(w, scalar_types.uint4b8, g, False)
            built[g] = (q_w, s, g_idx, sort_idx)
        for M in m_sweep:
            x = torch.randn(M, in_f, device=dev, dtype=torch.float16)
            for g, (q_w, s, g_idx, sort_idx) in built.items():
                fn = lambda q_w=q_w, s=s, g_idx=g_idx, sort_idx=sort_idx, x=x: \
                    mu.apply_gptq_marlin_linear(x, q_w, s, zp, g_idx, sort_idx, ws,
                        scalar_types.uint4b8, out_f, in_f, is_k_full=True, use_fp32_reduce=True)
                med, lo, hi = _graph_time(fn, args.iters, args.warmup, args.repeats)
                rows.append({"role": u["role"], "in": in_f, "out": out_f, "count": u["count"],
                             "M": M, "group_size": g, "t_us": med * 1000.0,
                             "t_us_min": lo * 1000.0, "t_us_max": hi * 1000.0,
                             "scale_bytes": scale_bytes(in_f, out_f, g),
                             "weight_bytes": weight_bytes(in_f, out_f)})
        # real-module cross-check at gate_M (deployed g128 served path)
        try:
            xr = torch.randn(gate_M, in_f, device=dev, dtype=torch.float16).to(torch.bfloat16)
            medr, _, _ = _graph_time(lambda: u["module"](xr), args.iters, args.warmup, args.repeats)
            real_xcheck[key] = medr * 1000.0
        except Exception as exc:  # noqa: BLE001
            real_xcheck[key] = None
            print(f"[bwprobe]   real xcheck failed {u['role']} {in_f}->{out_f}: {exc!r}", flush=True)
        del w, built
        gc.collect(); torch.cuda.empty_cache()

    # ---- aggregate verify-body GEMM time vs group_size at gate_M --------------
    def agg_at(M):
        out = {}
        for g in g_sweep:
            tot_us = 0.0
            tot_scale = 0.0
            tot_w = 0.0
            covered = True
            for key, u in uniq.items():
                r = next((x for x in rows if x["role"] == u["role"] and x["in"] == u["in"]
                          and x["out"] == u["out"] and x["M"] == M and x["group_size"] == g), None)
                if r is None:
                    covered = False
                    continue
                tot_us += r["t_us"] * u["count"]
                tot_scale += r["scale_bytes"] * u["count"]
                tot_w += r["weight_bytes"] * u["count"]
            out[g] = {"total_gemm_us": tot_us, "total_scale_MB": tot_scale / 1e6,
                      "total_weight_MB": tot_w / 1e6, "covered": covered}
        return out

    agg = {M: agg_at(M) for M in m_sweep}

    # ---- gate computation at gate_M -------------------------------------------
    # PRIMARY signal = GROUPED slope across {g128,g64,g32}. All three use the SAME
    # grouped Marlin kernel, so dT/d(scale_byte) over them isolates scale-load cost
    # without a kernel-variant confound. Per-channel (g=-1) uses a DIFFERENT kernel
    # path (group_size=size_k), so T_g128 - T_perchannel conflates scale bytes with
    # the kernel choice -> kept only as a SECONDARY, confounded anchor (NOT gated on).
    # Conservative direction: smaller g also raises group-boundary dequant frequency,
    # so the grouped slope is an UPPER BOUND on the true scale-BANDWIDTH cost; if even
    # this upper bound implies a tiny palette gain, the negative is robust.
    a = agg[gate_M]
    t_g128 = a.get(128, {}).get("total_gemm_us")
    t_pc = a.get(-1, {}).get("total_gemm_us")
    scale_mb_g128 = a.get(128, {}).get("total_scale_MB")
    weight_mb = a.get(128, {}).get("total_weight_MB")
    scale_bytes_g128 = scale_mb_g128 * 1e6 if scale_mb_g128 else None
    analytical_byte_share = (scale_mb_g128 / (scale_mb_g128 + weight_mb)) if scale_mb_g128 else None

    def band_us(g):
        gr = [r for r in rows if r["M"] == gate_M and r["group_size"] == g]
        return sum((r["t_us_max"] - r["t_us_min"]) * uniq[(r["role"], r["in"], r["out"])]["count"]
                   for r in gr) / 2.0

    noise_us = band_us(128)  # g128 count-weighted half-band (per-call jitter proxy)

    # --- PRIMARY: grouped slope over covered grouped points (exclude per-channel) ---
    grouped_gs = [g for g in g_sweep if g != -1 and g in a and a[g]["covered"]]
    grouped_pts = sorted((a[g]["total_scale_MB"] * 1e6, a[g]["total_gemm_us"], g) for g in grouped_gs)
    slope_grouped = None
    if len(grouped_pts) >= 2:
        xs = [p[0] for p in grouped_pts]; ys = [p[1] for p in grouped_pts]
        mx = statistics.mean(xs); my = statistics.mean(ys)
        denom = sum((x - mx) ** 2 for x in xs)
        slope_grouped = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
                         if denom else None)  # us per scale-byte
    # implied scale-load share at g128 from the grouped slope (extrapolate to 0 bytes)
    scale_load_us_grouped = (slope_grouped * scale_bytes_g128
                             if (slope_grouped is not None and scale_bytes_g128) else None)
    f_grouped = (scale_load_us_grouped / t_g128
                 if (scale_load_us_grouped is not None and t_g128) else None)
    # measured grouped span (most scale bytes minus fewest) + its endpoint noise
    grouped_span_us = (grouped_pts[-1][1] - grouped_pts[0][1]) if len(grouped_pts) >= 2 else None
    span_noise_us = ((band_us(grouped_pts[0][2]) + band_us(grouped_pts[-1][2]))
                     if len(grouped_pts) >= 2 else noise_us)

    # --- SECONDARY (confounded): per-channel kernel-variant anchor ---
    scale_load_us_pc = (t_g128 - t_pc) if (t_g128 and t_pc) else None
    f_perchannel = (scale_load_us_pc / t_g128) if (scale_load_us_pc is not None and t_g128) else None

    # fidelity: aggregate g128 scale payload reproduced from these shapes vs the
    # deployed 53.70 MB (#104). A close match validates the shape/count discovery.
    deployed_scale_mb = 53.70
    fidelity_ratio = (scale_mb_g128 / deployed_scale_mb) if scale_mb_g128 else None

    # harness cross-check: real-module (public g32) vs synthetic g32, count-weighted.
    real_tot = 0.0; syn_g32_tot = 0.0; xcheck_ok = True
    for key, u in uniq.items():
        rv = real_xcheck.get(key)
        sr = next((x for x in rows if x["role"] == u["role"] and x["in"] == u["in"]
                   and x["out"] == u["out"] and x["M"] == gate_M and x["group_size"] == PUBLIC_NATIVE_GROUP), None)
        if rv is None or sr is None:
            xcheck_ok = False
            continue
        real_tot += rv * u["count"]
        syn_g32_tot += sr["t_us"] * u["count"]
    real_vs_syn_g32_pct = (100.0 * (real_tot - syn_g32_tot) / syn_g32_tot
                           if (xcheck_ok and syn_g32_tot) else None)

    # palette wall_tps prediction (gate on the GROUPED f, both save fracs)
    def palette_gain_pct(save_frac):
        if f_grouped is None:
            return None
        return 100.0 * save_frac * f_grouped * VERIFY_GEMM_FRAC

    gain_global = palette_gain_pct(SAVE_FRAC_GLOBAL)
    gain_pertensor = palette_gain_pct(SAVE_FRAC_PERTENSOR)
    # slope-based cross-check: saved bytes * grouped slope -> us -> % of decode step
    decode_step_us = (t_g128 / VERIFY_GEMM_FRAC) if t_g128 else None
    gain_pertensor_slope = None
    if slope_grouped is not None and scale_bytes_g128 and decode_step_us:
        saved_us = slope_grouped * (SAVE_FRAC_PERTENSOR * scale_bytes_g128)
        gain_pertensor_slope = 100.0 * saved_us / decode_step_us

    # verdict: gate on the GROUPED slope (clean kernel), not the per-channel anchor
    significant = (grouped_span_us is not None and span_noise_us is not None
                   and grouped_span_us > 2.0 * span_noise_us and (f_grouped or 0) > 0.005)
    if not significant:
        verdict = "no"
    elif f_grouped is not None and analytical_byte_share and f_grouped >= 0.5 * analytical_byte_share:
        verdict = "yes"
    else:
        verdict = "partial"

    peak_mem_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print(f"\n[bwprobe] ===== VERIFY-BODY GEMM TIME vs GROUP_SIZE (M={gate_M}, launch-free) =====", flush=True)
    print("  group |  scale_MB | weight_MB | total_gemm_us | vs perchannel", flush=True)
    for g in g_sweep:
        if g not in a:
            continue
        d = a[g]
        dvs = (d["total_gemm_us"] - t_pc) if (t_pc is not None) else float("nan")
        tag = "deployed" if g == 128 else ("per-channel~0scale" if g == -1 else "")
        print(f"  {g:5d} | {d['total_scale_MB']:8.2f} | {d['total_weight_MB']:8.1f} | "
              f"{d['total_gemm_us']:11.1f} | {dvs:+8.1f}us  {tag}", flush=True)
    print(f"\n[bwprobe] fidelity: aggregate scale@g128 = {scale_mb_g128:.2f} MB vs deployed "
          f"53.70 MB (ratio {fidelity_ratio:.3f})", flush=True)
    if real_vs_syn_g32_pct is not None:
        print(f"[bwprobe] harness xcheck: real-module(g32) vs synthetic-g32 = "
              f"{real_vs_syn_g32_pct:+.1f}% (synthetic faithful if small)", flush=True)
    print(f"\n[bwprobe] PRIMARY grouped slope {{g128,g64,g32}} = "
          f"{(slope_grouped or 0)*1e3:+.4f} us/KB-scale", flush=True)
    print(f"[bwprobe]   grouped span (T_smallg - T_g128) = {(grouped_span_us or 0):+.1f} us "
          f"(noise +-{span_noise_us:.1f} us)  significant={significant}", flush=True)
    print(f"[bwprobe]   implied scale-load @g128 = {(scale_load_us_grouped or 0):+.1f} us  "
          f"f_grouped = {(f_grouped or 0)*100:.3f}%   vs analytical byte share "
          f"{(analytical_byte_share or 0)*100:.2f}%", flush=True)
    print(f"[bwprobe] SECONDARY per-channel anchor (CONFOUNDED by kernel variant): "
          f"T_g128 - T_perchannel = {(scale_load_us_pc if scale_load_us_pc is not None else float('nan')):+.1f} us  "
          f"f_perchannel = {(f_perchannel or 0)*100:.3f}%", flush=True)
    print(f"[bwprobe] VERDICT scale_bytes_on_critical_path = {verdict.upper()}", flush=True)
    print(f"[bwprobe] predicted palette wall_tps gain (grouped f): global(37.5%)={gain_global:+.3f}%  "
          f"per-tensor(43%)={gain_pertensor:+.3f}%", flush=True)
    print(f"[bwprobe] peak GPU mem {peak_mem_gib:.2f} GiB", flush=True)

    payload = {
        "config": {
            "model": args.model, "vllm": __import__("vllm").__version__,
            "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
            "m_sweep": m_sweep, "group_sweep": g_sweep, "gate_M": gate_M,
            "iters": args.iters, "warmup": args.warmup, "repeats": args.repeats,
            "decode_step_ms": args.decode_step_ms, "verify_gemm_frac": VERIFY_GEMM_FRAC,
            "wall_tps": WALL_TPS, "official_proj_mult": OFFICIAL_PROJ,
            "save_frac_global": SAVE_FRAC_GLOBAL, "save_frac_pertensor": SAVE_FRAC_PERTENSOR,
            "peak_gpu_mem_gib": peak_mem_gib,
            "note": "value-independent isolated Marlin timing; synthetic random weights; "
                    "group_size varies scale bytes at CONSTANT int4 weight bytes; "
                    "launch-free CUDA-graph replay. No served-file/token-stream change.",
        },
        "unique_shapes": [{"role": u["role"], "in": u["in"], "out": u["out"], "count": u["count"]}
                          for u in uniq.values()],
        "rows": rows,
        "aggregate_by_M": {str(M): agg[M] for M in m_sweep},
        "real_xcheck_us": {f"{k[0]}:{k[1]}->{k[2]}": v for k, v in real_xcheck.items()},
        "gate": {
            "gate_M": gate_M,
            "t_g128_us": t_g128, "t_perchannel_us": t_pc,
            # PRIMARY grouped-slope signal {g128,g64,g32} (clean kernel)
            "slope_grouped_us_per_scale_byte": slope_grouped,
            "scale_load_us_grouped": scale_load_us_grouped,
            "f_grouped_share": f_grouped,
            "grouped_span_us": grouped_span_us, "span_noise_us": span_noise_us,
            "significant": significant,
            # SECONDARY per-channel anchor (CONFOUNDED by kernel variant; not gated on)
            "scale_load_us_perchannel": scale_load_us_pc,
            "f_perchannel_share": f_perchannel,
            "noise_us_g128": noise_us,
            "analytical_byte_share": analytical_byte_share,
            "total_scale_MB_g128": scale_mb_g128, "total_weight_MB": weight_mb,
            "fidelity_scale_mb_vs_deployed_ratio": fidelity_ratio,
            "real_vs_synthetic_g32_pct": real_vs_syn_g32_pct,
            "scale_bytes_on_critical_path": verdict,
            "palette_served_wall_tps_gain_pct_global": gain_global,
            "palette_served_wall_tps_gain_pct_pertensor": gain_pertensor,
            "palette_gain_pct_pertensor_slope_xcheck": gain_pertensor_slope,
        },
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[bwprobe] wrote {args.output}", flush=True)

    if not args.no_wandb:
        try:
            _log_wandb(args, payload)
        except Exception as exc:  # noqa: BLE001
            print(f"[bwprobe] W&B logging failed: {exc!r}", flush=True)

    del llm
    gc.collect(); torch.cuda.empty_cache()


def _log_wandb(args, payload):
    import wandb
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type="profiling", config=payload["config"])
    cols = ["role", "in", "out", "count", "M", "group_size", "t_us", "scale_bytes", "weight_bytes"]
    tbl = wandb.Table(columns=cols)
    for r in payload["rows"]:
        tbl.add_data(*[r[c] for c in cols])
    run.log({"bw_sweep_table": tbl})
    gate_M = payload["gate"]["gate_M"]
    for g, d in payload["aggregate_by_M"][str(gate_M)].items():
        run.log({"group_size": g, "total_gemm_us": d["total_gemm_us"],
                 "total_scale_MB": d["total_scale_MB"]})
    g = payload["gate"]
    run.summary.update({k: v for k, v in g.items() if v is not None and not isinstance(v, str)})
    run.summary.update({
        "scale_bytes_on_critical_path": g["scale_bytes_on_critical_path"],
        "verdict_scales_hidden": int(g["scale_bytes_on_critical_path"] == "no"),
        "peak_gpu_mem_gib": payload["config"]["peak_gpu_mem_gib"],
    })
    run.finish()
    print(f"[bwprobe] W&B run: {run.url}", flush=True)


if __name__ == "__main__":
    main()
