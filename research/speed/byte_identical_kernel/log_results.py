#!/usr/bin/env python3
"""PR #550: consolidate the byte-identical-kernel verdict + log to W&B.

Reads enumerate_and_roofline.json (the GPU enumeration + per-op roofline +
self_det), composes the per-axis byte-identical-kernel verdict, and logs a
single analysis run to W&B group `faster-byte-identical-kernel`.

No GPU. analysis_only=true, official_tps=0.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

BASE_FULLHEAD_TPS = 253.78  # served (graphed + spec-on) local baseline, lawine #544


def build_verdict(rf: dict) -> dict:
    head = rf["head_roofline"]
    body = rf.get("body_roofline", {})
    kern = rf["current_kernels"]

    # --- per-axis byte-identical-kernel availability on A10G sm_86 ---
    axes = {
        "int4_body_gemm": {
            "dispatched_kernel": "MarlinLinearKernel (CompressedTensorsWNA16, w4a16 group_size=32)",
            "faster_byte_identical_alt_on_sm86": False,
            "argmax_identity_rate": 1.0,   # no swap; Marlin is byte-exact (#501 all-0-flip census)
            "tps_delta_vs_253": 0.0,
            "why": (
                "Marlin is the ONLY w4a16 kernel dispatchable on sm_86: Machete is Hopper-only "
                "(sm_90 wgmma+TMA; vLLM choose_mp_linear_kernel gates capability>=90), Marlin24 "
                "needs 2:4-sparse weights, ExLlamaV2 needs GPTQ format. The deployed Marlin matmul "
                "is already M-invariant/byte-exact (#501 census: 0 flips M=1-vs-M=8; fp32 "
                "global-reduce since atomic-add+bf16 is unsupported pre-SM90) and exposes no Python "
                "split-K knob. VLLM_MARLIN_USE_ATOMIC_ADD is unsupported for bf16 on sm_86 AND would "
                "change reduction order (break identity). At M=1 a single proj is launch-bound "
                "(62.5us vs 7.4us byte floor) -> flat-in-M (#18); the served stack's CUDA graphs "
                "(ONEGRAPH) already amortize that launch overhead -- it is not a kernel-swap lever."
            ),
            "measured_us_m1": body.get("measured_us"),
            "byte_floor_us": body.get("floor_us"),
        },
        "bf16_262k_head": {
            "dispatched_kernel": "bf16 dense ParallelLMHead / UnquantizedEmbeddingMethod [262144x2560] (cuBLAS GEMM at M=1, bf16 in/out)",
            "faster_byte_identical_alt_on_sm86": False,
            "argmax_identity_rate": 1.0,   # no swap; current path is the reference
            "tps_delta_vs_253": 0.0,
            "why": (
                f"Bandwidth-bound: reads the 1.342 GB head weight once per token. Measured "
                f"{head['measured_us']:.0f}us vs {head['floor_us']:.0f}us @600GB/s floor = "
                f"{head['bw_realized_GBs']:.0f} GB/s realized = {head['bw_realized_GBs']/6.0:.1f}% of "
                "the A10G 600 GB/s peak (i.e. AT the practical HBM ceiling; real achievable BW is "
                "~80-85% of peak). A GEMV-vs-GEMM swap is bandwidth-equal at M=1 (both read the "
                "matrix once) AND changes the K=2560 reduction tiling -> changes bf16 rounding -> "
                "breaks identity. The only head speedups move fewer BYTES: lower precision (lawine "
                "#544's tested lever, +38 TPS -> 292.1 ceiling) or row-prune (quality-unsafe 12k "
                "head) -- neither is a kernel lever; both change the argmax bits."
            ),
            "measured_us_m1": head["measured_us"],
            "byte_floor_us": head["floor_us"],
            "bw_realized_GBs": head["bw_realized_GBs"],
            "pct_of_peak_bw": head["bw_realized_GBs"] / 6.0,
        },
        "attention": {
            "dispatched_kernel": f"{kern['attention']['impl_class']} / {kern['attention']['backend']} (FORCED by vLLM for heterogeneous head_dim 256/512)",
            "faster_byte_identical_alt_on_sm86": False,
            "argmax_identity_rate": 0.0,   # FlashInfer default flips (fern #507: 1292/2048) -> fails gate
            "tps_delta_vs_253": 0.0,
            "why": (
                "vLLM FORCES TRITON_ATTN here ('Gemma4 has heterogeneous head dimensions head_dim=256/"
                "global_head_dim=512; Forcing TRITON_ATTN to prevent mixed-backend numerical "
                "divergence') -- so FlashAttention is both identity-risky AND prevented by vLLM for "
                "correctness. FlashInfer (fern #507, same A10G sm_86 box): (a) version-skewed -- "
                "JIT-only on torch-2.11+cu130, CUDA-13 cubin risk -> not a clean drop-in to the serve "
                "stack; (b) default split-KV decode is batch-VARIANT (1292/2048 flips M=1-vs-M=8 @"
                "L=8192) -> argmax_identity<1.0, fails #319 gate; (c) the invariant fixed_split_size "
                "knob is 1.2-4.7x SLOWER at M=1; (d) head_dim-512 full-attn layers have NO tensor-core/"
                "fixed-split path on Ampere. The served split-KV (lawine #496) already reproduces "
                "FlashInfer's byte-exact primitive, faster."
            ),
        },
    }

    any_green = any(a["faster_byte_identical_alt_on_sm86"] for a in axes.values())
    verdict = {
        "pr": 550,
        "analysis_only": True,
        "official_tps": 0,
        "hardware": f"{rf.get('device')} sm_{rf.get('sm')}",
        "current_kernels": {
            "int4_body_gemm": axes["int4_body_gemm"]["dispatched_kernel"],
            "bf16_262k_head": axes["bf16_262k_head"]["dispatched_kernel"],
            "attention": axes["attention"]["dispatched_kernel"],
        },
        "axes": axes,
        "kernel_lever_is_green": any_green,
        "fastest_byte_identical_tps": BASE_FULLHEAD_TPS,
        "tps_delta_vs_253": 0.0,
        "stacked_ceiling_tps": None,   # Stage 2 gated on a Stage-1 win; not reached
        "stage2_reached": False,
        "lm_head_full_ok": rf.get("lm_head_full_ok"),
        "lm_head_full_rows": rf.get("lm_head_full_rows"),
        "self_det": rf.get("self_det"),
        "base_fullhead_m1ar_eager_tps": rf.get("base_fullhead_m1ar_tps", {}).get("median_tps"),
        "head_measured_us": head["measured_us"],
        "head_floor_us": head["floor_us"],
        "head_bw_realized_GBs": head["bw_realized_GBs"],
        "head_pct_of_peak_bw": head["bw_realized_GBs"] / 6.0,
        "body_measured_us": body.get("measured_us"),
        "body_floor_us": body.get("floor_us"),
        "peak_vram_gib": rf.get("peak_vram_gib"),
        "self_test_passes": rf.get("self_test_passes"),
        "primary_metric_name": "fastest_byte_identical_tps",
        "primary_metric_value": BASE_FULLHEAD_TPS,
        "verdict": (
            "NO-GO: the current fast stack (Marlin int4 body + bf16 dense 262k head + vLLM-forced "
            "TritonAttention) is already the fastest byte-identical kernel set on A10G sm_86. The "
            "dominant byte-read ops are HBM-bandwidth-bound at ~80% of peak; every faster alternative "
            "(atomic-add Marlin, GEMV head, FlashInfer/FlashAttention) changes reduction order and "
            "breaks the #319 byte-identity gate, or is not dispatchable on sm_86. This HARDENS lawine "
            "#544's 292.1 precision-lever ceiling as KERNEL-ROBUST: at M=1 kernel selection is not a "
            "free TPS lever; the only identity-preserving lever is bytes-read (precision)."
        ),
    }
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(HERE / "enumerate_and_roofline.json"))
    ap.add_argument("--out", default=str(HERE / "verdict.json"))
    ap.add_argument("--wandb_name", default="denken/faster-byte-identical-kernel")
    ap.add_argument("--wandb_group", default="faster-byte-identical-kernel")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    rf = json.loads(Path(args.inp).read_text())
    verdict = build_verdict(rf)
    Path(args.out).write_text(json.dumps(verdict, indent=2))
    print(json.dumps({k: v for k, v in verdict.items()
                      if k not in ("axes", "current_kernels", "verdict")}, indent=2))
    print("\nVERDICT:", verdict["verdict"])

    run_id = None
    if not args.no_wandb:
        if str(REPO_ROOT) not in sys.path:
            sys.path.append(str(REPO_ROOT))
        try:
            from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                               log_json_artifact, log_summary)
            run = init_wandb_run(
                job_type="analysis", agent="denken",
                name=args.wandb_name, group=args.wandb_group,
                notes="PR #550 byte-identical-kernel axis: enumerate the kernels base_fullhead "
                      "dispatches at M=1 decode + roofline the dominant byte-read ops. NO-GO -- the "
                      "current Marlin/bf16-dense-head/forced-TritonAttention stack is already the "
                      "fastest byte-identical kernel set on A10G sm_86; faster alts break the #319 "
                      "argmax-identity gate or are not sm_86-dispatchable. Hardens lawine #544's "
                      "precision ceiling as kernel-robust. LOCAL analysis_only, no HF job.",
                tags=["byte-exact", "kernel-selection", "marlin", "lm-head-roofline",
                      "flashinfer", "negative", "pr-550", "kernel-robust"],
                config={"pr": 550, "wandb_group": args.wandb_group,
                        "model_id": "google/gemma-4-E4B-it (int4 w4a16 base_fullhead)",
                        "hardware": verdict["hardware"], "analysis_only": True,
                        "official_tps": 0, "baseline_base_fullhead_tps": BASE_FULLHEAD_TPS},
            )
            if run is not None:
                log_summary(run, verdict, step=0)
                log_json_artifact(run, name="byte_identical_kernel_verdict",
                                  artifact_type="analysis", data=verdict)
                log_json_artifact(run, name="enumerate_and_roofline",
                                  artifact_type="analysis", data=rf)
                run_id = run.id
                print(f"\n[wandb] run id: {run_id}")
                finish_wandb(run)
        except Exception as exc:  # noqa: BLE001
            print(f"[wandb] logging skipped: {type(exc).__name__}: {exc}")
    print(f"\nWANDB_RUN_ID={run_id}")


if __name__ == "__main__":
    main()
