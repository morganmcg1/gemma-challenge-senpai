#!/usr/bin/env python3
"""PR #625 -- log the QuaRot body re-quant Frobenius pre-screen + verdict to ONE W&B run.

analysis_only, official_tps=0, NO HF Job. Reads the airtight bf16-master screen
(frobenius_screen_bf16.json) and the consolidated verdict (verdict.json) and logs the
go/no-go mechanism result under group `quarot-body-requant-recovery`.

Run:  .venvs/vllm022/bin/python research/quarot_body_requant/log_wandb.py [--no-wandb]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="quarot-body-requant-recovery")
    ap.add_argument("--name", default="wirbel/quarot-body-requant-frobenius-screen")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    screen = json.load(open(HERE / "frobenius_screen_bf16.json"))
    screen_g32 = json.load(open(HERE / "frobenius_screen.json")) if (HERE / "frobenius_screen.json").exists() else None
    verdict = json.load(open(HERE / "verdict.json"))

    g = screen["global"]
    dG = verdict["gpqa_deficit_targeted"]

    flat = {
        # ---- VERDICT (the terminal go/no-go) ----
        "verdict": verdict["verdict"],
        "rotation_recovers_reasoning": verdict["rotation_recovers_reasoning"],
        "gate": verdict["gate"],
        "gate_outcome": verdict["gate_outcome"],
        "deficit_closed_frac": verdict["deficit_closed_frac"],
        # ---- required PR terminal fields ----
        "gpqa_probe_naive_int4": verdict["gpqa_probe_naive_int4"],
        "gpqa_probe_rotated_requant": verdict["gpqa_probe_rotated_requant"],  # None: see gate_rationale
        # ---- mechanism: int4 weight-error screen on the TRUE bf16 master ----
        "int4_incoherent_floor": verdict["int4_incoherent_floor"],
        "global_rel_naive_qat": g["global_rel_naive"],
        "global_rel_rotated_hadamard_g128": g["global_rel_had_g128blk"],
        "global_rel_rotated_randorth_full": g["global_rel_randorth_full"],
        "sse_reduction_full_rotation_NOT_zerotps": g["sse_reduction_had_g128blk"],
        "sse_reduction_randorth_full_NOT_zerotps": g["sse_reduction_randorth_full"],
        "sse_reduction_best_zerotps_rotation_R2_oproj": g["sse_reduction_r2_oproj_only"],
        # ---- the only offline/zero-TPS-foldable rotation (R2 V->O), both touched matrices ----
        "r2_o_proj_naive": screen["per_type"]["o_proj"]["rel_naive"],
        "r2_o_proj_rotated_head256": screen["per_type"]["o_proj"]["rel_extra"],
        "r2_v_proj_naive": screen["per_type"]["v_proj"]["rel_naive"],
        "r2_v_proj_rotated_Vout_head256": screen["per_type"]["v_proj"]["rel_extra"],
        # ---- the one above-floor (non-reasoning, non-foldable) type rotation HELPS ----
        "per_layer_input_gate_naive": screen["per_type"]["per_layer_input_gate"]["rel_naive"],
        "per_layer_input_gate_rotated_had": screen["per_type"]["per_layer_input_gate"]["rel_had_g128blk"],
        # ---- faithfulness of the bf16 master vs shipped int4_g128 ----
        "faithfulness_frac_int32_word_match_min": min(
            v["frac_int32_equal"] for v in screen["faithfulness_vs_shipped_g128"].values()),
        "faithfulness_frac_int32_word_match_max": max(
            v["frac_int32_equal"] for v in screen["faithfulness_vs_shipped_g128"].values()),
        # ---- GPQA deficit this PR targeted (reused #598 n4ro7bzk, NO new eval) ----
        "gpqa_base_bf16_body_mean_acc": dG["base_bf16_body_mean_acc"],
        "gpqa_int4_body_mean_acc": dG["int4_body_mean_acc"],
        "gpqa_raw_deficit_mt3072": dG["raw_deficit_mt3072"],
        "gpqa_residual_deficit_mt6144": dG["residual_deficit_after_truncation_fix_mt6144"],
        "gpqa_genuine_reasoning_n01_cells": dG["genuine_reasoning_n01_cells"],
        # ---- bookkeeping ----
        "n_language_modules": screen["n_language_modules"],
        "analysis_only": True,
        "official_tps": 0,
        "no_hf_job": True,
    }
    if screen_g32 is not None:
        flat["sse_reduction_R2_oproj_g32dequant_source_caveat"] = screen_g32["global"].get("sse_reduction_r2_oproj_only")

    print("=== PR #625 QuaRot body re-quant -- Frobenius go/no-go ===")
    for k, v in flat.items():
        print(f"  {k} = {v}")

    if not args.no_wandb:
        import wandb
        run = wandb.init(
            entity="wandb-applied-ai-team",
            project="gemma-challenge-senpai",
            group=args.group,
            name=args.name,
            job_type="quant-scheme-prescreen",
            config={
                "pr": 625,
                "hypothesis": "Hadamard-rotation re-quant of the int4 body recovers the land #619 GPQA reasoning deficit at zero TPS",
                "source": screen["source"],
                "quantizer": screen["quantizer"],
                "gate": "frobenius weight-error pre-screen (orthogonal rotation preserves Frobenius norm -> apples-to-apples int4-group-quant error of W vs W.R^T)",
                "rotations_tested": ["naive_qat", "hadamard_block128", "randorth_full_indim",
                                     "R2_head256_oproj_offline_foldable", "R2_Vout_head256_vproj"],
                "analysis_only": True,
                "official_tps": 0,
                "no_hf_job": True,
            },
        )
        run.summary.update(flat)
        # per-type table for audit
        run.summary["per_type_rel_err"] = screen["per_type"]
        run.summary["architectural_finding"] = verdict["architectural_finding"]
        run.summary["gate_rationale"] = verdict["gate_rationale"]
        run.summary["mechanism"] = verdict["mechanism"]
        print(f"\n[wandb] logged run {run.id} (group={args.group})")
        run.finish()
        print(f"WANDB_RUN_ID {run.id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
