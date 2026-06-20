#!/usr/bin/env python3
"""Log the PR #788 lm_head-bytes result to W&B (0-GPU, wandb-capable venv = /usr/bin/python3).

One run in group ``bi0-lmhead-bytes`` carrying the full fewer-weight-bytes ladder for the
bf16 lm_head of merged bi0: bf16 control -> int8 W8A16 channelwise (Arm A) -> int4 W4A16
g32 (Arm B). Reads the local artifacts under research/_int8head_smoke/ (prevalidate
summaries, eager op-profiles, GSM8K greedy paired evals, greedy-divergence, modalities)
and emits TPS / PPL / 128-completion / per-token lm_head GEMV / GSM8K quality / divergence
for each arm. analysis_only=1, no_hf_job=1 (local A10G exploratory TPS; PPL + greedy
token_ids + GSM8K are hardware-independent). Primary metric = int4 local decode TPS;
test metric = int4 GSM8K greedy accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent


def load(p: str) -> dict:
    return json.loads((HERE / p).read_text())


def gemv_per_token(breakdown: str, kernel_substr: str) -> float:
    d = load(breakdown)
    gt = d["gen_tokens"]
    for k in d["top_kernels"]:
        if kernel_substr in k["kernel"] and k["count"] == gt:
            return k["cuda_ms"] / gt
    raise SystemExit(f"kernel {kernel_substr!r} count=={gt} not found in {breakdown}")


def marlin_op_ms(breakdown: str) -> tuple[float, int]:
    d = load(breakdown)
    for k in d["top_kernels"]:
        if k["kernel"] == "_C::marlin_gemm":
            return k["cuda_ms"], k["count"]
    raise SystemExit(f"_C::marlin_gemm not found in {breakdown}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="gemma-challenge-senpai")
    ap.add_argument("--entity", default="wandb-applied-ai-team")
    ap.add_argument("--group", default="bi0-lmhead-bytes")
    ap.add_argument("--name", default="ubel/bi0-lmhead-bytes")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    # --- prevalidate (TPS / PPL / completed) ---
    ctrl = load("prevalidate_bf16_control/local_summary.json")
    i8 = load("prevalidate_int8_candidate/local_summary.json")
    i4 = load("prevalidate_int4_candidate/local_summary.json")

    # --- GSM8K greedy (paired, n=200, seed 1234) ---
    g_ctrl = load("gsm8k/bf16control_greedy.json")
    g_i8 = load("gsm8k/int8head_greedy.json")
    g_i4 = load("gsm8k/int4head_greedy.json")

    # --- greedy divergence vs bf16 control (evidence only) ---
    div8 = load("divergence_int8_vs_bf16.json")
    div4 = load("divergence_int4_vs_bf16.json")

    # --- modalities (all-4) ---
    mod8 = load("modalities_int8.json")
    mod4 = load("modalities_int4.json")

    # --- per-token lm_head GEMV (eager, M=1, spec OFF) ---
    gemv_bf16 = gemv_per_token("profile_bf16head/profile_breakdown.json", "gemv2T_kernel_val")
    gemv_i8 = gemv_per_token("profile_int8head/profile_breakdown.json", "ampere_hgemm_W8A16")
    # int4 lm_head folds into the body's Marlin template -> isolate by the +gen_tokens op-count
    # delta vs the byte-identical int8 body (lm_head is AllSpark in the int8 arm, NOT Marlin).
    m4_ms, m4_n = marlin_op_ms("profile_int4head/profile_breakdown.json")
    m8_ms, m8_n = marlin_op_ms("profile_int8head/profile_breakdown.json")
    gt = load("profile_int4head/profile_breakdown.json")["gen_tokens"]
    assert m4_n - m8_n == gt, f"expected +{gt} marlin calls (lm_head/token), got {m4_n - m8_n}"
    gemv_i4 = (m4_ms - m8_ms) / gt

    def qd(name, val, base):  # quality delta vs base
        return round((val - base) / base * 100, 3)

    config = {
        "pr": 788, "phase": "bi0_lmhead_bytes",
        "base_submission": "submissions/int4_mtp_bi0_surgattn",
        "base_model_id": "google/gemma-4-E4B-it-qat-w4a16-ct (int4 body, bf16 lm_head, tied)",
        "lever": "quantize the bf16 lm_head GEMV; int4 body byte-identical",
        "spec_method": "mtp (gemma4_assistant drafter)", "num_speculative_tokens": 6,
        "vllm": "0.22.0", "gpu": "A10G sm_86 (local, exploratory TPS)",
        "sampler": "VLLM_USE_FLASHINFER_SAMPLER=0 (native; greedy/PPL-identical)",
        "decode_num_prompts": 128, "output_len": 512, "ppl_gate": 2.42,
        "gsm8k_n": g_i4["n_problems"], "gsm8k_seed": g_i4["seed"], "gsm8k_nshot": g_i4["n_shot"],
        "base_gsm8k_anchor": 0.867, "gsm8k_official_gate": 0.807,
        "analysis_only": 1, "no_hf_job": 1, "fires": 0, "official_tps": 0,
    }

    run = wandb.init(
        project=args.project, entity=args.entity, group=args.group,
        name=args.name, id=args.run_id, resume=("allow" if args.run_id else None),
        config=config,
    )

    summary = {
        "analysis_only": 1, "no_hf_job": 1, "fires": 0, "official_tps": 0,
        # primary + test metric (int4 = the bigger win)
        "primary_metric_int4_local_decode_tps": i4["tps"],
        "test_metric_int4_gsm8k_greedy_acc": g_i4["accuracy"],
        # --- decode TPS (local exploratory) ---
        "tps_bf16_control": ctrl["tps"], "tps_int8_head": i8["tps"], "tps_int4_head": i4["tps"],
        "tps_int8_gain_pct": qd("tps", i8["tps"], ctrl["tps"]),
        "tps_int4_gain_pct": qd("tps", i4["tps"], ctrl["tps"]),
        # --- PPL (hardware-independent; gate <= 2.42) ---
        "ppl_bf16_control": ctrl["ppl"], "ppl_int8_head": i8["ppl"], "ppl_int4_head": i4["ppl"],
        "ppl_int8_within_gate": int(i8["ppl"] <= 2.42), "ppl_int4_within_gate": int(i4["ppl"] <= 2.42),
        # --- 128/128 completion ---
        "completed_bf16_control": ctrl["completed"], "completed_int8_head": i8["completed"],
        "completed_int4_head": i4["completed"],
        # --- per-token lm_head GEMV (eager) ---
        "gemv_ms_per_tok_bf16": round(gemv_bf16, 4), "gemv_ms_per_tok_int8": round(gemv_i8, 4),
        "gemv_ms_per_tok_int4": round(gemv_i4, 4),
        "gemv_int8_speedup_x": round(gemv_bf16 / gemv_i8, 3),
        "gemv_int4_speedup_x": round(gemv_bf16 / gemv_i4, 3),
        # --- lm_head bytes/token (1.342 GB bf16) ---
        "lmhead_bytes_gb_bf16": 1.3422, "lmhead_bytes_gb_int8": 0.6711, "lmhead_bytes_gb_int4": 0.3775,
        "byte_reduction_x_int8": 2.00, "byte_reduction_x_int4": 3.56,
        "quant_rel_err_int8": 0.00966, "quant_rel_err_int4": 0.06743,
        # --- GSM8K greedy paired (within-5%-of-base gate) ---
        "gsm8k_acc_bf16_control": g_ctrl["accuracy"], "gsm8k_acc_int8_head": g_i8["accuracy"],
        "gsm8k_acc_int4_head": g_i4["accuracy"],
        "gsm8k_int8_delta_pct_vs_ctrl": qd("acc", g_i8["accuracy"], g_ctrl["accuracy"]),
        "gsm8k_int4_delta_pct_vs_ctrl": qd("acc", g_i4["accuracy"], g_ctrl["accuracy"]),
        "gsm8k_int8_within_5pct": int(g_i8["accuracy"] >= 0.95 * g_ctrl["accuracy"]),
        "gsm8k_int4_within_5pct": int(g_i4["accuracy"] >= 0.95 * g_ctrl["accuracy"]),
        # --- greedy divergence vs bf16 control (EVIDENCE, not a gate) ---
        "divergence_frac_int8": div8["frac_diverged"], "divergence_frac_int4": div4["frac_diverged"],
        "div_mean_prefix_match_int8": div8["mean_prefix_match_frac"],
        "div_mean_prefix_match_int4": div4["mean_prefix_match_frac"],
        # --- all-4-modalities (presence tier) ---
        "modalities_all_loaded_int8": int(mod8["all_modalities_loaded"]),
        "modalities_all_loaded_int4": int(mod4["all_modalities_loaded"]),
        # verdicts
        "int8_clears_bar": int(i8["tps"] > ctrl["tps"] and i8["ppl"] <= 2.42 and i8["completed"] == 128
                               and g_i8["accuracy"] >= 0.95 * g_ctrl["accuracy"] and mod8["all_modalities_loaded"]),
        "int4_clears_bar": int(i4["tps"] > ctrl["tps"] and i4["ppl"] <= 2.42 and i4["completed"] == 128
                               and g_i4["accuracy"] >= 0.95 * g_ctrl["accuracy"] and mod4["all_modalities_loaded"]),
    }
    run.summary.update(summary)

    run.log({"arms": wandb.Table(
        columns=["arm", "lmhead_dtype", "kernel", "bytes_gb", "tps", "tps_gain_pct", "ppl",
                 "completed", "gemv_ms_tok", "gsm8k_acc", "gsm8k_delta_pct", "divergence_frac",
                 "modalities_all"],
        data=[
            ["bf16_control", "bf16", "cuBLAS gemv2T", 1.3422, ctrl["tps"], 0.0, ctrl["ppl"],
             ctrl["completed"], round(gemv_bf16, 4), g_ctrl["accuracy"], 0.0, 0.0, None],
            ["int8_head", "int8 W8A16 ch", "AllSpark", 0.6711, i8["tps"],
             summary["tps_int8_gain_pct"], i8["ppl"], i8["completed"], round(gemv_i8, 4),
             g_i8["accuracy"], summary["gsm8k_int8_delta_pct_vs_ctrl"], div8["frac_diverged"],
             int(mod8["all_modalities_loaded"])],
            ["int4_head_g32", "int4 W4A16 g32", "Marlin", 0.3775, i4["tps"],
             summary["tps_int4_gain_pct"], i4["ppl"], i4["completed"], round(gemv_i4, 4),
             g_i4["accuracy"], summary["gsm8k_int4_delta_pct_vs_ctrl"], div4["frac_diverged"],
             int(mod4["all_modalities_loaded"])],
        ])})

    # consolidated artifact (single source of truth for the PR comment)
    results = {"config": config, "summary": summary}
    (HERE / "results_788.json").write_text(json.dumps(results, indent=2))

    print(f"[wandb] run {run.id} group={args.group}")
    print(f"[wandb] TPS ctrl={ctrl['tps']:.2f} int8={i8['tps']:.2f}(+{summary['tps_int8_gain_pct']}%) "
          f"int4={i4['tps']:.2f}(+{summary['tps_int4_gain_pct']}%)")
    print(f"[wandb] GEMV ms/tok bf16={gemv_bf16:.3f} int8={gemv_i8:.3f} int4={gemv_i4:.3f}")
    print(f"[wandb] GSM8K ctrl={g_ctrl['accuracy']:.4f} int8={g_i8['accuracy']:.4f} int4={g_i4['accuracy']:.4f}")
    print(f"[wandb] clears_bar int8={summary['int8_clears_bar']} int4={summary['int4_clears_bar']}")
    run.finish()
    print(f"RUN_ID={run.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
