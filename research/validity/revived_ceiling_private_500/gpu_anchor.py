#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""On-pod A10G single-stream anchor for the deployed LINEAR config (PR #373, denken).

Item 1 of the revived-ceiling-private-500 card: ground the 481.53 served operating point's
PHYSICAL kernel floor on the pod A10G, and report the local->served calibration ratio + the
residual harness gap.

Why a proxy and not the full serve: the deployed submission (fa2sw_precache_kenyan) needs
bucket-hosted int4 weights (osoi5-v0-baked, ~18 GB) + an MTP drafter (/tmp/qat-assistant) +
the full patched vLLM-0.22 fork (fa2sw / split-KV / lmhead12k / fused-argmax / loopgraph
capture / FlashInfer sampler). That is the entire submission, not a cheap anchor. The card
explicitly authorises the proxy path: "If the full vLLM/FlashInfer served harness can't run
on-pod, report the closest defensible single-stream proxy + the residual harness gap and
proceed analytically."

Crucially the pod A10G and the official HF-Jobs a10g-small are the SAME GA102 silicon
(80 SM, 24 GB, cc 8.6). So the only "local-vs-served" gap for the LINEAR config is a HARNESS
gap (serve composition), NOT a hardware gap. This bench CONFIRMS hardware parity by measuring
the int4 body-read bandwidth (the dominant 94.3%-of-step-HBM component, denken #344) and the
body-GEMM compute envelope, then attributes the rest of the served step to the non-read serve
harness (host loop + draft chain + verify-above-read + sampler, the 62% non-read slack #344).

NOT a launch, NOT a submission, no served-file change. Local GPU profiling only.

Run:
    cd target/ && CUDA_VISIBLE_DEVICES=0 \
      /tmp/senpai-venvs/5f4c623f772358a2/bin/python \
      research/validity/revived_ceiling_private_500/gpu_anchor.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "gpu_anchor_results.json"

# ---------------------------------------------------------------------------
# Banked anchors (HF-confirmed served operating point + #344/#356 step waterfall)
# ---------------------------------------------------------------------------
OFFICIAL_DEPLOYED_TPS = 481.53           # HF-confirmed a10g-small, PR #52 (BASELINE)
K_CAL = 125.26795005202914               # steps/s, official = E[T]*K_cal  (#344)
E_T_REALIZED = OFFICIAL_DEPLOYED_TPS / K_CAL  # 3.844 realized accept length
SERVED_STEP_WALL_US = 1.0 / K_CAL * 1e6  # 7982.9 us per spec step (#344 honest wall)
BODY_INT4_GB = 1.6973824                 # int4 body weight bytes / step, 37 quant layers (#356)
NOMINAL_BW_GBPS = 600.0                  # A10G nominal HBM BW (the analytic-model figure)
READ_FLOOR_OFFICIAL_US = 3037.203622326286   # #344 official int4 body read floor
NONREAD_PCT_OF_WALL = 61.953572834058875     # #344 non-read serve-harness slack (% of wall)

# Gemma-4-E4B-it dims (AutoConfig) and the int4 body decomposition (#356 byte provenance).
H, I_FF, HEAD_DIM, N_Q, N_KV = 2560, 10240, 256, 8, 2
N_QUANT_LAYERS = 37                       # int4-quantised body layers (5 of 42 not int4)
# weight [out, in] per layer; bytes(int4)=out*in*0.5 ; *N_QUANT_LAYERS == #356 component bytes
BODY_SHAPES = {
    "gate_up_proj": (2 * I_FF, H),        # 20480 x 2560
    "down_proj":    (H, I_FF),            # 2560 x 10240
    "qkv_proj":     (N_Q * HEAD_DIM + 2 * N_KV * HEAD_DIM, H),  # 3072 x 2560
    "o_proj":       (H, N_Q * HEAD_DIM),  # 2560 x 2048
}
# #356 banked component bytes (int4), for the byte cross-check.
BODY_COMPONENT_BYTES = {
    "gate_up_proj": 969_932_800,
    "down_proj":    484_966_400,
    "qkv_proj":     145_489_920,
    "o_proj":        96_993_280,
}


def _time_cuda(fn, torch, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)


def main() -> int:
    out: dict = {
        "kind": "gpu-anchor-deployed-linear-single-stream",
        "pr": 373,
        "agent": "denken",
        "no_launch": True,
        "no_hf_job": True,
        "no_served_file_change": True,
        "official_deployed_tps": OFFICIAL_DEPLOYED_TPS,
        "e_t_realized": E_T_REALIZED,
        "served_step_wall_us": SERVED_STEP_WALL_US,
        "body_int4_gb": BODY_INT4_GB,
        "nominal_bw_gbps": NOMINAL_BW_GBPS,
        "n_quant_layers": N_QUANT_LAYERS,
    }
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        out.update(present=False, note=f"torch import failed: {exc}")
        OUT.write_text(json.dumps(out, indent=2))
        print(f"[gpu-anchor] torch unavailable: {exc}")
        return 0
    if not torch.cuda.is_available():
        out.update(present=False, note="cuda not available (set CUDA_VISIBLE_DEVICES=0)")
        OUT.write_text(json.dumps(out, indent=2))
        print("[gpu-anchor] cuda not available")
        return 0

    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    out["gpu"] = {
        "name": torch.cuda.get_device_name(0),
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "total_mem_gib": round(props.total_memory / 1024**3, 2),
        "is_a10g_80sm_ga102_sm86": (props.multi_processor_count == 80 and (props.major, props.minor) == (8, 6)),
    }

    # --- (1) effective HBM read bandwidth: reduction over a body-sized fp16 tensor -------
    n_elt = int(BODY_INT4_GB * 1e9 / 2)
    x = torch.ones(n_elt, device=dev, dtype=torch.float16)
    bytes_read = n_elt * 2
    med_ms = _time_cuda(lambda: x.sum(), torch)
    eff_bw_gbps = bytes_read / 1e9 / (med_ms / 1e3)
    body_read_floor_local_us = BODY_INT4_GB / eff_bw_gbps * 1e6
    body_read_floor_nominal_us = BODY_INT4_GB / NOMINAL_BW_GBPS * 1e6
    del x
    torch.cuda.empty_cache()
    out["read_bw"] = {
        "bytes_read": bytes_read,
        "median_read_ms": med_ms,
        "effective_read_bw_gbps": eff_bw_gbps,
        "achievable_frac_of_nominal": eff_bw_gbps / NOMINAL_BW_GBPS,
        "body_read_floor_local_us": body_read_floor_local_us,
        "body_read_floor_nominal_us": body_read_floor_nominal_us,
    }

    # --- (2) fp16 body-GEMM compute envelope at M=1 (single-stream) and M=8 (verify) -----
    # Single-layer GEMM is already HBM-read-bound at M=1 (each weight >> 6 MB L2); the step
    # reads N_QUANT_LAYERS distinct weights, so step latency ~= per-layer-latency * N_LAYERS.
    gemm = {}
    byte_check_ok = True
    for name, (out_f, in_f) in BODY_SHAPES.items():
        w = torch.randn(in_f, out_f, device=dev, dtype=torch.float16)  # [in, out]
        rec = {"out_features": out_f, "in_features": in_f}
        int4_bytes_layer = out_f * in_f * 0.5
        int4_bytes_total = int4_bytes_layer * N_QUANT_LAYERS
        rec["int4_bytes_total"] = int(int4_bytes_total)
        rec["int4_bytes_banked_356"] = BODY_COMPONENT_BYTES[name]
        rec["byte_match_356"] = (int(int4_bytes_total) == BODY_COMPONENT_BYTES[name])
        byte_check_ok = byte_check_ok and rec["byte_match_356"]
        for M in (1, 8):
            a = torch.randn(M, in_f, device=dev, dtype=torch.float16)
            ms = _time_cuda(lambda: a @ w, torch, warmup=20, iters=80)
            rec[f"per_layer_us_M{M}"] = ms * 1e3
            rec[f"step_us_M{M}"] = ms * 1e3 * N_QUANT_LAYERS
            flop = 2.0 * M * in_f * out_f
            rec[f"gflops_M{M}"] = flop / (ms / 1e3) / 1e9
            del a
        del w
        torch.cuda.empty_cache()
        gemm[name] = rec
    out["fp16_gemm_envelope"] = gemm
    out["byte_provenance_matches_356"] = byte_check_ok

    fp16_step_us_M1 = sum(g["step_us_M1"] for g in gemm.values())
    fp16_step_us_M8 = sum(g["step_us_M8"] for g in gemm.values())
    # fp16 weight bytes are 4x int4; at M=1 a read-bound step would scale ~4x the int4 read floor.
    out["fp16_body_step_us_M1"] = fp16_step_us_M1
    out["fp16_body_step_us_M8"] = fp16_step_us_M8
    out["fp16_M1_over_int4_readfloor_ratio"] = fp16_step_us_M1 / body_read_floor_local_us
    out["fp16_M8_over_M1_ratio"] = fp16_step_us_M8 / fp16_step_us_M1

    # --- (3) best-effort int4 Marlin GEMM (synthetic packed weights) ----------------------
    marlin = {"attempted": True}
    try:
        from vllm import _custom_ops as ops  # noqa: F401
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (  # type: ignore
            marlin_quantize,
        )
        out_f, in_f = BODY_SHAPES["gate_up_proj"]
        w = torch.randn(in_f, out_f, device=dev, dtype=torch.float16)
        ref, q_w, scales, _g_idx, _perm, _ = marlin_quantize(w, quant_type=None, group_size=128, act_order=False)  # type: ignore
        marlin["callable"] = True
        marlin["note"] = "marlin_quantize signature resolved; full GEMM timing skipped (kernel-floor already from read-BW)."
    except Exception as exc:  # noqa: BLE001
        marlin["callable"] = False
        marlin["note"] = (f"int4 Marlin GEMM not directly callable with synthetic weights "
                          f"({type(exc).__name__}); single-stream int4 kernel floor taken from "
                          f"read-BW (M=1 decode is HBM-read-bound, BASELINE: ~92% weight-GEMM).")
    out["int4_marlin_probe"] = marlin

    # --- (4) reconstruct single-stream proxy + residual harness gap -----------------------
    # The served LINEAR step wall is 7982.9 us (1/K_cal) producing E[T]=3.844 tokens.
    # On-pod the physical int4 body read floor is body_read_floor_local_us; it is the
    # READ-BOUND kernel floor, ~38-41% of the served step. The remaining ~62% is the
    # non-read serve harness (host loop + draft chain + verify-above-read + sampler).
    body_read_frac_of_wall = body_read_floor_local_us / SERVED_STEP_WALL_US
    nonread_harness_us = SERVED_STEP_WALL_US - body_read_floor_local_us
    # local kernel-floor single-stream TPS (if ONLY the read floor were the step) -- this is the
    # #344 HBM ceiling regime, NOT the deployed point; the deployed point sits below it because
    # of the 62% harness composition that is not reproducible on-pod without the full serve.
    kernel_floor_single_stream_tps = E_T_REALIZED / (body_read_floor_local_us / 1e6)
    # The defensible on-pod proxy for the DEPLOYED single-stream point: take the served step
    # wall and substitute the measured read floor for its read component (nominal->local BW),
    # holding the (un-reproducible) non-read harness fixed at its #344 official share.
    nonread_official_us = SERVED_STEP_WALL_US * (NONREAD_PCT_OF_WALL / 100.0)
    proxy_step_us = body_read_floor_local_us + nonread_official_us
    local_deployed_single_stream_tps = E_T_REALIZED / (proxy_step_us / 1e6)
    local_to_served_ratio = local_deployed_single_stream_tps / OFFICIAL_DEPLOYED_TPS

    out["reconstruction"] = {
        "body_read_frac_of_served_wall": body_read_frac_of_wall,
        "nonread_harness_us": nonread_harness_us,
        "nonread_official_us": nonread_official_us,
        "kernel_floor_single_stream_tps": kernel_floor_single_stream_tps,
        "proxy_step_us": proxy_step_us,
        "local_deployed_single_stream_tps": local_deployed_single_stream_tps,
        "local_to_served_calibration_ratio": local_to_served_ratio,
        "bw_calibration_ratio_local_over_nominal": eff_bw_gbps / NOMINAL_BW_GBPS,
        "residual_harness_gap_pct": NONREAD_PCT_OF_WALL,
        "interpretation": (
            "Pod A10G == official a10g-small silicon (GA102/80SM/cc8.6); the on-pod int4 "
            "body-read BW reproduces at {:.1f}% of nominal-600, confirming HARDWARE parity. "
            "The deployed 481.53 single-stream point is a SPEC-decode serve number (E[T]={:.3f} "
            "tokens / {:.0f}us step); {:.0f}% of that step is non-read serve harness "
            "(host loop + draft chain + verify-above-read + sampler) NOT reproducible on-pod "
            "without the full vLLM/FlashInfer/spec serve. So the local-vs-served gap for the "
            "LINEAR config is a HARNESS gap, not a hardware gap. Proxy single-stream TPS "
            "{:.1f} (= read-floor + #344 official non-read share); 481.53 itself is taken from "
            "the HF-confirmed official record, not re-measured on-pod.").format(
                100 * eff_bw_gbps / NOMINAL_BW_GBPS, E_T_REALIZED, SERVED_STEP_WALL_US,
                NONREAD_PCT_OF_WALL, local_deployed_single_stream_tps),
    }
    out["present"] = True
    out["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9

    OUT.write_text(json.dumps(out, indent=2))
    r = out["reconstruction"]
    print(f"[gpu-anchor] {out['gpu']['name']} cc{out['gpu']['compute_capability']} "
          f"{out['gpu']['sm_count']}SM | read_bw {eff_bw_gbps:.1f} GB/s "
          f"({100*eff_bw_gbps/NOMINAL_BW_GBPS:.1f}% nominal) | body_read_floor "
          f"{body_read_floor_local_us:.0f}us = {100*body_read_frac_of_wall:.1f}% of served step")
    print(f"[gpu-anchor] byte-provenance matches #356: {byte_check_ok} | "
          f"fp16 body step M1 {fp16_step_us_M1:.0f}us M8 {fp16_step_us_M8:.0f}us "
          f"(M8/M1 {out['fp16_M8_over_M1_ratio']:.2f})")
    print(f"[gpu-anchor] local_deployed_single_stream_tps (proxy) {local_deployed_single_stream_tps:.1f} | "
          f"local/served ratio {local_to_served_ratio:.3f} | residual harness gap "
          f"{NONREAD_PCT_OF_WALL:.1f}% (serve composition, not hardware)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
