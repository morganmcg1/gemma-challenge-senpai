#!/usr/bin/env python3
"""PR #560 fern — log the candidate-verify-served-realize KEY OUTPUTS to W&B.

Group: candidate-verify-served-realize. Analysis-only (no fire): official_tps=0.
PRIMARY metric = cv_realized_quality_safe_tps (252.31 anchor). Logs the measured
Stage-1 step microbench, the Stage-2 measured re-projection, and the Stage-3/3c
realized-path identity HARD gate (1.0 with the server-matched vocab-index tie-break)."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.wandb_logging import (
    finish_wandb,
    init_wandb_run,
    log_json_artifact,
    log_summary,
)

HERE = Path(__file__).resolve().parent
S1 = json.load(open(HERE / "stage1_cv_microbench.json"))
S2 = json.load(open(HERE / "stage2_reproject.json"))
S3 = json.load(open(HERE / "stage3_realized_identity.json"))
S3B = json.load(open(HERE / "stage3b_verify_precision.json"))
S3C = json.load(open(HERE / "stage3c_tiebreak.json"))

M1 = S1["per_M"]["1"]
M8 = S1["per_M"]["8"]

# realized-path identity HARD gate: 1.0 via Stage-3c server-matched vocab-index tie-break
realized_identity = S3C["identity_vs_served_bf16"]["B_vocTB"]
hard_gate_pass = bool(
    realized_identity == 1.0 and S3C["containment_rate_at_K8"] == 1.0
)
peak_gpu = max(
    S1.get("peak_gpu_gb", 0.0), S3.get("peak_gpu_gb", 0.0)
)

summary = {
    # provenance / guardrails
    "pr": 560,
    "analysis_only": True,
    "official_tps": 0,
    "self_det": 1.0,
    "peak_gpu_gb": peak_gpu,
    "operating_point_M": 8,
    # --- Stage 1: the binding measurement (int4-nominator achieved BW) ---
    "cv_int4_nominator_bw_GBs": M1["cv_int4_nominator_bw_GBs"],            # 510.4 @ M1
    "cv_int4_nominator_pct_of_peak": M1["cv_int4_nominator_pct_of_peak"],  # 0.851
    "cv_int4_nominator_bw_GBs_M8": M8["cv_int4_nominator_bw_GBs"],         # 499.5 @ M8
    "cv_int4_nominator_pct_of_peak_M8": M8["cv_int4_nominator_pct_of_peak"],
    "bf16_gemv_bw_GBs_M1": M1["bf16_gemv_alone_bw_GBs"],
    "bf16_gemv_pct_of_peak_M1": M1["bf16_gemv_alone_pct_of_peak"],
    "int4_gemv_latency_ms_M1": M1["int4_gemv_latency_ms"],
    "int4_gemv_latency_ms_M8": M8["int4_gemv_latency_ms"],
    "cv_step_latency_ms": M8["cv_step_latency_ms"],                        # 0.798 @ M8
    "bf16_head_argmax_step_latency_ms": M8["bf16_head_argmax_step_latency_ms"],  # 2.700
    "cv_step_speedup": M8["cv_step_speedup"],                              # 3.38x
    "gemv_only_speedup_M8": M8.get("gemv_only_speedup"),
    # --- Stage 2: measured re-projection (PRIMARY) ---
    "cv_realized_quality_safe_tps": S2["cv_realized_quality_safe_tps"],   # 291.36 PRIMARY
    "cv_realized_quality_safe_tps_254": S2["cv_realized_quality_safe_tps_254"],
    "cv_realized_quality_safe_tps_wm265": S2["cv_realized_quality_safe_tps_wm265"],
    "cv_realized_tps_gain": S2["cv_realized_tps_gain"],                   # +39.05
    "cv_realized_tps_gain_band_252_lo": S2["cv_realized_tps_gain_band_252"][0],
    "cv_realized_tps_gain_band_252_hi": S2["cv_realized_tps_gain_band_252"][1],
    "cv_realized_tps_gain_ci95_lo": S2["cv_realized_tps_gain_ci95_252"][0],
    "cv_realized_tps_gain_ci95_hi": S2["cv_realized_tps_gain_ci95_252"][1],
    "cv_projection_confirmed": S2["cv_projection_confirmed"],             # True
    "cv_realized_vs_projected_delta": S2["cv_realized_vs_projected_delta"],        # -1.05
    "cv_realized_vs_projected_delta_265basis": S2["cv_realized_vs_projected_delta_265basis"],
    "cv_served_tps_is_measured": S2["cv_served_tps_is_measured"],         # False
    "proj_549_gain_central": S2["proj_549_gain_central"],                 # +40.10
    "proj_549_gain_band_lo": S2["proj_549_gain_band"][0],
    "proj_549_gain_band_hi": S2["proj_549_gain_band"][1],
    "head_ceiling_holds_at_292": S2["head_ceiling_holds_at_292"],        # True
    "head_savings_ms_central": S2["head_savings_ms_band"]["central_calibrated"],
    "head_savings_ms_conservative": S2["head_savings_ms_band"]["conservative_iso"],
    "head_savings_ms_optimistic": S2["head_savings_ms_band"]["optimistic_inctx"],
    # --- Stage 3 / 3c: realized-path identity HARD gate ---
    "argmax_identity_rate": realized_identity,                           # 1.0 (vocTB)
    "argmax_identity_rate_position_tiebreak": S3C["identity_vs_served_bf16"]["B_posTB"],  # 0.99545
    "containment_rate_at_K8": S3C["containment_rate_at_K8"],              # 1.0
    "identity_n_positions": S3["n_positions"],                           # 60000
    "K_safe_bf16": S3["K_safe_bf16"],
    "identity_hard_gate_pass": hard_gate_pass,                            # True
}

run = init_wandb_run(
    job_type="analysis",
    agent="fern",
    name="fern/candidate-verify-served-realize",
    group="candidate-verify-served-realize",
    notes="PR #560: measure the int4-nominator GEMV achieved BW and convert #549's "
          "read-bound +40 TPS projection into a measured re-projection; re-confirm "
          "realized-path argmax identity == 1.0 (server-matched vocab-index tie-break).",
    tags=["pr560", "candidate-verify", "fullhead", "analysis-only", "no-fire"],
    config={
        "pr": 560,
        "model": "google/gemma-4-E4B-it-qat-w4a16-ct",
        "head_path": "marlin_uint4b8_g128",
        "K_safe": 8,
        "operating_point_M": 8,
        "anchor_primary": S1["anchors"]["anchor_base_fullhead_252"],
        "anchor_535": S1["anchors"]["anchor_base_fullhead_254"],
        "wall_per_step_ms_served": S1["anchors"]["wall_per_step_ms_served"],
        "tps_warm_median_m1": S1["anchors"]["tps_warm_median_m1"],
        "head_gemm_ms_incontext_m8": S1["anchors"]["head_gemm_ms_incontext_m8"],
        "hbm_peak_GBs": 600.0,
        "gpu": "A10G_sm86",
        "analysis_only": True,
        "official_tps": 0,
    },
)
log_summary(run, summary, step=0)
for nm, data in (
    ("stage1_cv_microbench", S1),
    ("stage2_reproject", S2),
    ("stage3_realized_identity", S3),
    ("stage3b_verify_precision", S3B),
    ("stage3c_tiebreak", S3C),
):
    log_json_artifact(run, name=f"pr560_{nm}", artifact_type="analysis", data=data)
finish_wandb(run)
print("logged W&B group=candidate-verify-served-realize  primary cv_realized_quality_safe_tps=%.2f  "
      "identity=%.6f  gate_pass=%s  peak_gpu=%.2fGB"
      % (summary["cv_realized_quality_safe_tps"], realized_identity, hard_gate_pass, peak_gpu))
