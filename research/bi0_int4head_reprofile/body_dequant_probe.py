#!/usr/bin/env python
"""PR #798 follow-on — body GEMM int4-Marlin vs bf16-cuBLAS, per shape (the de-quant lever).

The main attribution (gemm_attrib.py) times every body Linear as int4 W4A16 Marlin (the
shipped path). This probe answers a SEPARATE question the rung-3 verdict turns on: for
which body shapes is int4-Marlin actually SLOWER than plain bf16 cuBLAS at the conc=1
decode width (M=7)?

WHY THIS CAN HAPPEN: int4-Marlin's win is a smaller weight READ (bandwidth). But its
fixed tile grid needs enough output columns (N) to fill the SMs. At N=256
(per_layer_input_gate, 2560->256) the grid is starved (~3% HBM BW) while bf16 cuBLAS
runs the same tiny GEMV near its small-shape ceiling — so int4 is a NET LOSS there. At
large N (mlp/attn/lm_head) int4's read saving dominates and it wins big. This probe
measures the crossover so the "de-quant per_layer_input_gate -> bf16" lever is a measured
number, not an assertion.

Value-independent (same argument as gemm_attrib.py): at M=7 both kernels are ~4x below
the FP16 ridge, so wall time is shape-bound, not weight-value-bound -> synthetic weights
are faithful. LOCAL ONLY, no model load, no HF job.

Reuses the EXACT serving kernel paths from gemm_attrib.py (apply_gptq_marlin_linear for
int4; torch.mm bf16 cuBLAS for the de-quantized counterfactual).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

# gemm_attrib sets CUDA_VISIBLE_DEVICES=0 etc. at import (before torch); reuse it.
from gemm_attrib import (  # noqa: E402
    BODY_SHAPES,
    A10G_HBM_GBS,
    GROUP_SIZE,
    build_marlin,
    roofline,
    time_lmhead_bf16,
    time_marlin,
)

import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m-sweep", default="1,7,8,16",
                    help="M=7 is the deployed verify width; 1/8/16 bracket the crossover")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--verify-width", type=int, default=7)
    ap.add_argument("--output",
                    default="research/bi0_int4head_reprofile/body_dequant_probe.json")
    ap.add_argument("--wandb_project",
                    default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--wandb_group", default="bi0-int4head-reprofile")
    ap.add_argument("--wandb_name", default="stark/int4head-body-dequant-probe")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    m_sweep = [int(x) for x in args.m_sweep.split(",") if x.strip()]
    vw = args.verify_width
    device = torch.device("cuda:0")
    assert torch.cuda.is_available(), "no CUDA device (check CUDA_VISIBLE_DEVICES=0)"
    print(f"[probe] device={torch.cuda.get_device_name(0)} "
          f"vllm={__import__('vllm').__version__} torch={torch.__version__}", flush=True)

    run = None
    if not args.no_wandb:
        try:
            import wandb
            run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                             group=args.wandb_group, name=args.wandb_name,
                             job_type="profiling",
                             config={"m_sweep": m_sweep, "iters": args.iters,
                                     "warmup": args.warmup, "group_size": GROUP_SIZE,
                                     "verify_width": vw,
                                     "device": torch.cuda.get_device_name(0),
                                     "note": "body int4-Marlin vs bf16-cuBLAS per shape; "
                                             "measures the per_layer_input_gate de-quant lever"})
            print(f"[probe] W&B run: {run.url}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[probe] W&B init failed: {exc!r}", flush=True)

    t0 = time.time()
    rows = []  # per (shape, M): int4 + bf16 side by side
    print("[probe] timing body shapes int4-Marlin vs bf16-cuBLAS ...", flush=True)
    for (out_f, in_f, count, role) in BODY_SHAPES:
        packed = build_marlin(out_f, in_f, device)
        for M in m_sweep:
            ms_i4, g_i4 = time_marlin(packed, M, args.iters, args.warmup, device)
            ms_bf, g_bf = time_lmhead_bf16(M, in_f, out_f, args.iters, args.warmup, device)
            ri = roofline(out_f, in_f, M, ms_i4, w4=True)
            rb = roofline(out_f, in_f, M, ms_bf, w4=False)
            rows.append({
                "role": role, "out": out_f, "in": in_f, "count": count, "M": M,
                "int4_ms": ms_i4, "bf16_ms": ms_bf,
                "int4_graphed": g_i4, "bf16_graphed": g_bf,
                "int4_pct_hbm_peak": ri["pct_hbm_peak"],
                "bf16_pct_hbm_peak": rb["pct_hbm_peak"],
                # >1 => int4 SLOWER than bf16 (de-quant would help); <1 => int4 wins.
                "int4_over_bf16": ms_i4 / max(ms_bf, 1e-9),
                # per-step block ms (x layer count) for both, + the de-quant saving.
                "int4_block_ms": ms_i4 * count,
                "bf16_block_ms": ms_bf * count,
                "dequant_saving_block_ms": (ms_i4 - ms_bf) * count,
            })
        r7 = next(r for r in rows if r["role"] == role and r["M"] == vw)
        flag = "  <-- int4 SLOWER (de-quant lever)" if r7["int4_over_bf16"] > 1.0 else ""
        print(f"[probe] {role:30s} {in_f:5d}->{out_f:6d} x{count:3d}  M={vw}: "
              f"int4 {r7['int4_ms']*1000:6.1f}us ({r7['int4_pct_hbm_peak']:3.0f}%BW) | "
              f"bf16 {r7['bf16_ms']*1000:6.1f}us ({r7['bf16_pct_hbm_peak']:3.0f}%BW) | "
              f"int4/bf16 {r7['int4_over_bf16']:4.2f}x{flag}", flush=True)
        del packed
        gc.collect(); torch.cuda.empty_cache()

    # de-quant lever totals at the verify width: which shapes are int4-slower, and the
    # total per-step ms we'd recover by serving those few shapes in bf16 instead.
    vw_rows = [r for r in rows if r["M"] == vw]
    slower = [r for r in vw_rows if r["int4_over_bf16"] > 1.0]
    dequant_total_ms = sum(r["dequant_saving_block_ms"] for r in slower)
    print(f"\n[probe] @ M={vw}: {len(slower)} of {len(vw_rows)} body shapes are int4-SLOWER", flush=True)
    for r in sorted(slower, key=lambda r: -r["dequant_saving_block_ms"]):
        print(f"  {r['role']:30s} x{r['count']:3d}: int4 {r['int4_block_ms']:.3f}ms vs "
              f"bf16 {r['bf16_block_ms']:.3f}ms  -> save {r['dequant_saving_block_ms']:.3f}ms/step",
              flush=True)
    print(f"[probe] total recoverable by de-quanting int4-slower shapes -> bf16: "
          f"{dequant_total_ms:.3f} ms/step", flush=True)

    peak_mem_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"[probe] peak GPU mem: {peak_mem_gib:.2f} GiB; elapsed {time.time()-t0:.1f}s",
          flush=True)

    payload = {
        "config": {
            "device": torch.cuda.get_device_name(0),
            "vllm": __import__("vllm").__version__, "torch": torch.__version__,
            "m_sweep": m_sweep, "iters": args.iters, "warmup": args.warmup,
            "group_size": GROUP_SIZE, "verify_width": vw,
            "peak_gpu_mem_gib": peak_mem_gib,
            "note": "body int4-Marlin vs bf16-cuBLAS per shape; value-independent at M=7. "
                    "int4_over_bf16>1 marks shapes where the int4 Marlin tile grid is "
                    "starved (small N) and bf16 cuBLAS is faster -> a de-quant lever.",
        },
        "rows": rows,
        "dequant_lever_at_vw": {
            "verify_width": vw,
            "n_shapes_int4_slower": len(slower),
            "shapes_int4_slower": [r["role"] for r in slower],
            "total_recoverable_ms_per_step": dequant_total_ms,
        },
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[probe] wrote {args.output}", flush=True)

    if run is not None:
        try:
            cols = ["role", "M", "in", "out", "count", "int4_ms", "bf16_ms",
                    "int4_pct_hbm_peak", "bf16_pct_hbm_peak", "int4_over_bf16",
                    "dequant_saving_block_ms"]
            tbl = wandb.Table(columns=cols)
            for r in rows:
                tbl.add_data(r["role"], r["M"], r["in"], r["out"], r["count"],
                             r["int4_ms"], r["bf16_ms"], r["int4_pct_hbm_peak"],
                             r["bf16_pct_hbm_peak"], r["int4_over_bf16"],
                             r["dequant_saving_block_ms"])
            run.log({"body_dequant_table": tbl})
            run.summary.update({
                "verify_width": vw,
                "n_shapes_int4_slower": len(slower),
                "dequant_total_recoverable_ms_per_step": dequant_total_ms,
                "peak_gpu_mem_gib": peak_mem_gib,
            })
            run.finish()
        except Exception as exc:  # noqa: BLE001
            print(f"[probe] W&B log failed: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
