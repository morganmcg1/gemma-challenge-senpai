#!/usr/bin/env python
"""PR #798 — int4head decode GEMM attribution via synthetic-weight Marlin microbench.

WHY SYNTHETIC (no model load)
-----------------------------
This pod has ~7.4 GB free disk; the merged int4head checkpoint
(`/workspace/gemma_build/bi0_int4head_g32`, ~10 GB) is NOT built here and cannot be
built (source ckpt is already 11 GB cached; build needs source+output simultaneously).
So a full *served* re-profile of the int4head stack is impossible on this node.

But the quantity PR #798 asks for — the per-token GEMM cost split (body-Marlin verify
GEMM vs lm_head GEMV) — is **value-independent** in the conc=1 decode regime: a
W4A16-Marlin / bf16-cuBLAS GEMM's wall time depends only on (M, N, K, group_size),
not the weight VALUES (it is weight-bandwidth-bound; M=7 is ~4x below the FP16 ridge,
see #117/#130/#108 and verify_gemm_roofline.py). So we time the EXACT decode GEMM
shapes with the EXACT serving kernel path (`apply_gptq_marlin_linear` -> `ops.marlin_gemm`,
the same call vLLM's CompressedTensorsW4A16 scheme runs) using synthetic random
weights of the right shape. The shape list + multiplicities are read from the cached
source checkpoint's safetensors header and reproduce the known body byte budget
(1.986 GB packed + 0.248 GB scales = 2.234 GB/forward) exactly.

The non-GEMM terms (attention, sampling, MTP drafter, KV read, norms) are UNCHANGED by
the lm_head quant (only the lm_head bytes changed), so they are carried from the #781
served attribution + the #786/#789 STEPTIME absolutes for the before/after table.

OUTPUT
------
- lm_head GEMV: bf16 cuBLAS vs int4-g32 Marlin, ms at M in {1,7,8,...} (the new lever).
- body verify-GEMM: per-shape + aggregate (sum over 343 Linears) ms at the M-sweep,
  with the M-flatness that decides the rung-3 1-wave-saturation verdict.
- JSON dump + W&B (group bi0-int4head-reprofile).
LOCAL ONLY. No model download, no HF job, no token-stream touch.
"""
from __future__ import annotations

import os

# Must precede torch/vllm import (see verify_gemm_roofline.py / project_local_pod_gpu_index
# memory): the pod exposes one A10G as index 0 but may inherit a stale CUDA_VISIBLE_DEVICES.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import argparse
import gc
import json
import math
import statistics
import time

import torch

from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_gptq_marlin_linear,
    marlin_make_empty_g_idx,
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    marlin_quantize,
)

# ---- A10G (AWS g5, GA102, sm_86) roofline ceilings (match verify_gemm_roofline.py) ----
A10G_HBM_GBS = 600.0
A10G_FP16_TENSOR_TFLOPS = 70.0
WEIGHT_BITS = 4
SCALE_BYTES = 2
GROUP_SIZE = 32
HIDDEN = 2560
VOCAB = 262144

# Body GEMM shapes (out, in) -> count, as the FUSED layers vLLM actually serves
# (gemma4.py: MergedColumnParallelLinear gate_up, QKVParallelLinear qkv). The checkpoint
# stores separate q/k/v; vLLM packs them into one qkv GEMM per layer. KV-sharing
# (num_kv_shared_layers=18) shares the KV *cache*, not the projection — but the checkpoint
# only carries k/v for the 24 non-shared layers, so the 18 shared layers project q only.
# Full-attn layers (7: idx 5,11,17,23,29,35,41) use global_head_dim=512 (q=4096, kv=1024);
# 35 sliding use head_dim=256 (q=2048, kv=512). Non-shared: 20 sliding + 4 full; shared:
# 15 sliding + 3 full. This regrouping preserves the byte total (2.234 GB/forward = 1.986
# packed + 0.248 scales) verified against the separate-tensor enumeration.
BODY_SHAPES = [
    # (out, in, count, role)
    (20480, 2560, 42, "mlp.gate_up_proj(fused)"),    # gate+up fused, all 42 layers
    (2560, 10240, 42, "mlp.down_proj"),              # down, all 42 layers
    (3072, 2560, 20, "attn.qkv_proj.sliding(fused)"),  # q2048+k512+v512, non-shared sliding
    (6144, 2560, 4,  "attn.qkv_proj.full(fused)"),     # q4096+k1024+v1024, non-shared full
    (2048, 2560, 15, "attn.q_proj.sliding(kvshared)"),  # q only (KV cache shared)
    (4096, 2560, 3,  "attn.q_proj.full(kvshared)"),
    (2560, 2048, 35, "attn.o_proj.sliding"),
    (2560, 4096, 7,  "attn.o_proj.full"),
    (256, 2560, 42,  "per_layer_input_gate"),        # ReplicatedLinear, NOT fused (N=256)
    (2560, 256, 42,  "per_layer_projection"),
    (10752, 2560, 1, "per_layer_model_projection"),
]
LM_HEAD = (VOCAB, HIDDEN)  # (out, in)


def build_marlin(out_f: int, in_f: int, device, group_size: int = GROUP_SIZE):
    """Pack a synthetic [in_f, out_f] weight to W4A16-g32 Marlin (uint4b8, no act_order).

    Returns the tensors `apply_gptq_marlin_linear` consumes — the exact objects the
    CompressedTensorsW4A16 serving scheme holds after process_weights_after_loading."""
    w = torch.randn(in_f, out_f, device=device, dtype=torch.float16)  # [size_k, size_n]
    wtype = scalar_types.uint4b8
    _w_ref, marlin_q_w, marlin_s, _g_idx, _sort, _perm = marlin_quantize(
        w, wtype, group_size, act_order=False
    )
    # Group-symmetric (no act_order) serving path: empty g_idx / sort_indices / zp.
    g_idx = marlin_make_empty_g_idx(device)
    sort_indices = marlin_make_empty_g_idx(device)
    zp = marlin_make_empty_g_idx(device)
    workspace = marlin_make_workspace_new(device)
    return {
        "marlin_q_w": marlin_q_w, "marlin_s": marlin_s, "g_idx": g_idx,
        "sort_indices": sort_indices, "zp": zp, "workspace": workspace,
        "out": out_f, "in": in_f, "wtype": wtype,
    }


def marlin_call(packed, x):
    return apply_gptq_marlin_linear(
        input=x,
        weight=packed["marlin_q_w"],
        weight_scale=packed["marlin_s"],
        weight_zp=packed["zp"],
        g_idx=packed["g_idx"],
        g_idx_sort_indices=packed["sort_indices"],
        workspace=packed["workspace"],
        wtype=packed["wtype"],
        output_size_per_partition=packed["out"],
        input_size_per_partition=packed["in"],
        is_k_full=True,
    )


def time_call_graph(fn, iters, warmup):
    """Launch-free per-call ms via CUDA-graph replay (true kernel time, no launch floor).

    Mirrors verify_gemm_roofline.time_gemm_graph but for a no-arg closure. Falls back to
    eager (with the ~55us launch floor, flagged) if capture fails."""
    try:
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
        with torch.inference_mode():
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
        print(f"[gemm]   graph capture failed ({packed_desc(fn)}): {exc!r}; eager",
              flush=True)
        return e0.elapsed_time(e1) / iters, False


def packed_desc(_fn):
    return "closure"


def gemm_bytes(out_f, in_f, M, w4: bool):
    if w4:
        w = (WEIGHT_BITS / 8.0) * out_f * in_f + SCALE_BYTES * out_f * math.ceil(in_f / GROUP_SIZE)
    else:
        w = 2.0 * out_f * in_f  # bf16 weight
    return w + 2.0 * M * in_f + 2.0 * M * out_f  # + act in + act out


def roofline(out_f, in_f, M, ms, w4: bool):
    t = ms / 1000.0
    flops = 2.0 * M * out_f * in_f
    tb = gemm_bytes(out_f, in_f, M, w4)
    g_s = flops / t / 1e9
    gb_s = tb / t / 1e9
    return {
        "gflops_s": g_s, "gbytes_s": gb_s, "total_bytes": tb, "flops": flops,
        "pct_hbm_peak": 100.0 * gb_s / A10G_HBM_GBS,
        "pct_compute_peak": 100.0 * g_s / (A10G_FP16_TENSOR_TFLOPS * 1000.0),
        "ai_flop_per_byte": flops / tb,
    }


def time_lmhead_bf16(M, in_f, out_f, iters, warmup, device):
    """bf16 cuBLAS lm_head GEMV: x[M,in] @ W[in,out]. The pre-int4head head path."""
    W = torch.randn(in_f, out_f, device=device, dtype=torch.bfloat16)
    x = torch.randn(M, in_f, device=device, dtype=torch.bfloat16)
    out = torch.empty(M, out_f, device=device, dtype=torch.bfloat16)
    fn = lambda: torch.mm(x, W, out=out)
    return time_call_graph(fn, iters, warmup)


def time_marlin(packed, M, iters, warmup, device):
    x = torch.randn(M, packed["in"], device=device, dtype=torch.bfloat16)
    fn = lambda: marlin_call(packed, x)
    return time_call_graph(fn, iters, warmup)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m-sweep", default="1,2,4,7,8,16,29,32",
                    help="verify row counts; M=7 is the deployed verify width (K=6 -> K+1)")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--verify-width", type=int, default=7)
    ap.add_argument("--output",
                    default="research/bi0_int4head_reprofile/gemm_attrib.json")
    ap.add_argument("--wandb_project",
                    default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--wandb_group", default="bi0-int4head-reprofile")
    ap.add_argument("--wandb_name", default="stark/int4head-gemm-attrib")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    m_sweep = [int(x) for x in args.m_sweep.split(",") if x.strip()]
    device = torch.device("cuda:0")
    assert torch.cuda.is_available(), "no CUDA device (check CUDA_VISIBLE_DEVICES=0)"
    print(f"[gemm] device={torch.cuda.get_device_name(0)} vllm={__import__('vllm').__version__} "
          f"torch={torch.__version__}", flush=True)

    # Init W&B early (liveness + the run is the deliverable's artifact).
    run = None
    if not args.no_wandb:
        try:
            import wandb
            run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                             group=args.wandb_group, name=args.wandb_name,
                             job_type="profiling",
                             config={"m_sweep": m_sweep, "iters": args.iters,
                                     "warmup": args.warmup, "group_size": GROUP_SIZE,
                                     "verify_width": args.verify_width,
                                     "device": torch.cuda.get_device_name(0),
                                     "A10G_HBM_GBS": A10G_HBM_GBS,
                                     "A10G_FP16_TENSOR_TFLOPS": A10G_FP16_TENSOR_TFLOPS,
                                     "note": "synthetic-weight Marlin GEMM microbench; "
                                             "value-independent decode-GEMM timing for "
                                             "int4head re-attribution (model build disk-blocked)"})
            print(f"[gemm] W&B run: {run.url}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[gemm] W&B init failed: {exc!r}", flush=True)

    t0 = time.time()

    # ---- lm_head: bf16 cuBLAS vs int4-g32 Marlin -----------------------------
    print("[gemm] building int4-g32 Marlin lm_head ...", flush=True)
    lm_packed = build_marlin(LM_HEAD[0], LM_HEAD[1], device)
    lmhead = {"bf16": {}, "int4": {}}
    for M in m_sweep:
        ms_bf, gbf = time_lmhead_bf16(M, LM_HEAD[1], LM_HEAD[0], args.iters, args.warmup, device)
        ms_i4, gi4 = time_marlin(lm_packed, M, args.iters, args.warmup, device)
        rb = roofline(LM_HEAD[0], LM_HEAD[1], M, ms_bf, w4=False)
        ri = roofline(LM_HEAD[0], LM_HEAD[1], M, ms_i4, w4=True)
        lmhead["bf16"][M] = {"ms": ms_bf, "graphed": gbf, **rb}
        lmhead["int4"][M] = {"ms": ms_i4, "graphed": gi4, **ri}
        print(f"[gemm] lm_head M={M:3d}: bf16 {ms_bf:7.3f}ms ({rb['pct_hbm_peak']:4.0f}%BW) | "
              f"int4 {ms_i4:7.3f}ms ({ri['pct_hbm_peak']:4.0f}%BW) | "
              f"saving {ms_bf-ms_i4:6.3f}ms ({ms_bf/max(ms_i4,1e-9):4.2f}x)", flush=True)
    del lm_packed
    gc.collect(); torch.cuda.empty_cache()

    # ---- body verify-GEMM: per-shape + aggregate over 343 Linears -------------
    print("[gemm] building + timing body Marlin shapes ...", flush=True)
    body_rows = []  # per (shape, M)
    for (out_f, in_f, count, role) in BODY_SHAPES:
        packed = build_marlin(out_f, in_f, device)
        for M in m_sweep:
            ms, graphed = time_marlin(packed, M, args.iters, args.warmup, device)
            r = roofline(out_f, in_f, M, ms, w4=True)
            body_rows.append({"role": role, "out": out_f, "in": in_f, "count": count,
                              "M": M, "ms": ms, "graphed": graphed, **r})
        r7 = next(x for x in body_rows if x["role"] == role and x["M"] == args.verify_width)
        print(f"[gemm] {role:28s} {in_f:5d}->{out_f:6d} x{count:3d}  "
              f"M={args.verify_width}: {r7['ms']*1000:7.1f}us  "
              f"{r7['gbytes_s']:5.0f}GB/s ({r7['pct_hbm_peak']:4.0f}%BW) "
              f"{r7['gflops_s']/1000:5.1f}TF/s ({r7['pct_compute_peak']:4.0f}%pk)", flush=True)
        del packed
        gc.collect(); torch.cuda.empty_cache()

    # aggregate body GEMM per decode step (sum ms*count) at each M
    body_agg = {}
    for M in m_sweep:
        tot_ms = sum(r["ms"] * r["count"] for r in body_rows if r["M"] == M)
        tot_bytes = sum(r["total_bytes"] * r["count"] for r in body_rows if r["M"] == M)
        tot_flops = sum(r["flops"] * r["count"] for r in body_rows if r["M"] == M)
        t = tot_ms / 1e3
        body_agg[M] = {
            "M": M, "body_gemm_ms": tot_ms,
            "agg_gbytes_s": tot_bytes / t / 1e9,
            "agg_pct_hbm_peak": 100.0 * (tot_bytes / t / 1e9) / A10G_HBM_GBS,
            "agg_pct_compute_peak": 100.0 * (tot_flops / t / 1e9) / (A10G_FP16_TENSOR_TFLOPS * 1000.0),
            "agg_ai": tot_flops / tot_bytes,
        }

    vw = args.verify_width
    # M-flatness: body GEMM ms relative to M=1 (1-wave saturation check)
    flat = {M: body_agg[M]["body_gemm_ms"] / body_agg[1]["body_gemm_ms"] for M in m_sweep}

    print("\n[gemm] ===== BODY verify-GEMM aggregate (343 Linears, CUDA-graph) =====", flush=True)
    print("  M  | body_gemm_ms | agg GB/s (%BW) | %compute | ms/ms(M=1)", flush=True)
    for M in m_sweep:
        a = body_agg[M]
        print(f"  {M:3d} | {a['body_gemm_ms']:11.3f} | {a['agg_gbytes_s']:5.0f} "
              f"({a['agg_pct_hbm_peak']:4.0f}%) | {a['agg_pct_compute_peak']:6.1f}% | {flat[M]:5.3f}",
              flush=True)

    body_ms_vw = body_agg[vw]["body_gemm_ms"]
    lm_bf_vw = lmhead["bf16"][vw]["ms"]
    lm_i4_vw = lmhead["int4"][vw]["ms"]
    print(f"\n[gemm] @ verify width M={vw}:", flush=True)
    print(f"  body verify-GEMM  : {body_ms_vw:7.3f} ms", flush=True)
    print(f"  lm_head bf16 GEMV : {lm_bf_vw:7.3f} ms   (pre-int4head)", flush=True)
    print(f"  lm_head int4 GEMV : {lm_i4_vw:7.3f} ms   (int4head; saving {lm_bf_vw-lm_i4_vw:.3f} ms)",
          flush=True)
    peak_mem_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"[gemm] peak GPU mem: {peak_mem_gib:.2f} GiB; elapsed {time.time()-t0:.1f}s", flush=True)

    payload = {
        "config": {
            "device": torch.cuda.get_device_name(0),
            "vllm": __import__("vllm").__version__, "torch": torch.__version__,
            "m_sweep": m_sweep, "iters": args.iters, "warmup": args.warmup,
            "group_size": GROUP_SIZE, "verify_width": vw, "hidden": HIDDEN, "vocab": VOCAB,
            "A10G_HBM_GBS": A10G_HBM_GBS, "A10G_FP16_TENSOR_TFLOPS": A10G_FP16_TENSOR_TFLOPS,
            "peak_gpu_mem_gib": peak_mem_gib,
            "body_shapes": [{"out": o, "in": i, "count": c, "role": r}
                            for (o, i, c, r) in BODY_SHAPES],
            "note": "synthetic-weight Marlin/cuBLAS GEMM microbench; value-independent "
                    "decode-GEMM timing. Model build disk-blocked; non-GEMM terms carried "
                    "from #781 served attribution + #786/#789 STEPTIME for re-attribution.",
        },
        "lmhead": {k: {str(M): v for M, v in d.items()} for k, d in lmhead.items()},
        "body_rows": body_rows,
        "body_aggregate_by_M": {str(M): v for M, v in body_agg.items()},
        "body_flatness_vs_M1": {str(M): flat[M] for M in m_sweep},
        "verify_width_summary": {
            "M": vw, "body_gemm_ms": body_ms_vw,
            "lmhead_bf16_ms": lm_bf_vw, "lmhead_int4_ms": lm_i4_vw,
            "lmhead_saving_ms": lm_bf_vw - lm_i4_vw,
            "lmhead_speedup_x": lm_bf_vw / max(lm_i4_vw, 1e-9),
        },
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[gemm] wrote {args.output}", flush=True)

    if run is not None:
        try:
            cols = ["kind", "role", "M", "in", "out", "count", "ms", "gbytes_s",
                    "pct_hbm_peak", "pct_compute_peak", "graphed"]
            tbl = wandb.Table(columns=cols)
            for kind, d in lmhead.items():
                for M, v in d.items():
                    tbl.add_data(f"lm_head_{kind}", "lm_head", M, HIDDEN, VOCAB, 1,
                                 v["ms"], v["gbytes_s"], v["pct_hbm_peak"],
                                 v["pct_compute_peak"], v["graphed"])
            for r in body_rows:
                tbl.add_data("body", r["role"], r["M"], r["in"], r["out"], r["count"],
                             r["ms"], r["gbytes_s"], r["pct_hbm_peak"],
                             r["pct_compute_peak"], r["graphed"])
            run.log({"gemm_table": tbl})
            for M in m_sweep:
                run.log({"M": M, "body_gemm_ms": body_agg[M]["body_gemm_ms"],
                         "body_agg_pct_hbm_peak": body_agg[M]["agg_pct_hbm_peak"],
                         "body_flatness_vs_M1": flat[M],
                         "lmhead_bf16_ms": lmhead["bf16"][M]["ms"],
                         "lmhead_int4_ms": lmhead["int4"][M]["ms"]})
            run.summary.update({
                "verify_width": vw, "body_gemm_ms_at_vw": body_ms_vw,
                "lmhead_bf16_ms_at_vw": lm_bf_vw, "lmhead_int4_ms_at_vw": lm_i4_vw,
                "lmhead_saving_ms": lm_bf_vw - lm_i4_vw,
                "lmhead_speedup_x": lm_bf_vw / max(lm_i4_vw, 1e-9),
                "body_flatness_M1_to_vw": flat[vw],
                "peak_gpu_mem_gib": peak_mem_gib,
            })
            run.finish()
        except Exception as exc:  # noqa: BLE001
            print(f"[gemm] W&B log failed: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
