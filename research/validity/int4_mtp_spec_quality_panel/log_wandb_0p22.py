#!/usr/bin/env python
"""PR #629 (fern) — log the Option-B quality panel on vLLM 0.22.0 to W&B.

Single-variable A/B vs the #624 dev307 panel: same int4_g128_lmhead+MTP-K7 spec stack,
BI=1, gb6144 greedy gates + GPQA-D >=10 seeds. ONLY change = engine dev307 -> 0.22.0.
Answers whether Option-B is stack-robust or inherits the #547/#618 int4 0.22.0 crater.

LOCAL ONLY, analysis_only=True, official_tps=0. Group `optionb-quality-0p22-10seed`.
This config is NOT #319 byte-identical (spec) -> SURFACE-to-human, never auto-fire.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import wandb

GROUP = "optionb-quality-0p22-10seed"
HERE = Path(__file__).resolve().parent
ENTITY = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
PROJECT = os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai")

COMMON = {
    "config_name": "int4_g128_lmhead + MTP-K7 spec (fern #597)",
    "model_id": "/workspace/gemma_build/int4_g128_lmhead",
    "drafter": "/tmp/qat-assistant",
    "num_speculative_tokens": 7,
    "vllm_batch_invariant": 1,
    "engine": "manifest (0.22.0)",
    "vllm_version": "0.22.0",
    "max_model_len": 8192,
    "max_num_seqs": 16,
    "min_tokens": 8,
    "max_tokens": 6144,
    "greedy_decode": "temp=0, top_p=1, top_k off",
    "gpqa_sampling": "T=1.0, top_p=0.95, top_k=64",
    "is_319_identical": False,
    "analysis_only": True,
    "official_tps": 0,
    "comparison_baseline": "dev307 #624 panel",
}


def main() -> int:
    panel = json.loads((HERE / "panel_0p22.json").read_text())
    gates = panel["gates"]

    run = wandb.init(
        project=PROJECT, entity=ENTITY, name="fern/optionb-quality-0p22-panel",
        group=GROUP, job_type="quality-panel", reinit=True,
        config={**COMMON, "bars": panel["bars"],
                "crater_fl_threshold": panel["crater_fl_threshold"]},
    )

    log = {}
    for leg in ("mmlu_pro", "gsm8k", "aime", "gpqa_diamond"):
        gl = gates.get(leg)
        if not gl:
            continue
        log[f"{leg}/accuracy"] = gl["accuracy"]
        log[f"{leg}/bar"] = gl["bar"]
        log[f"{leg}/pass"] = int(gl["pass"])
        log[f"{leg}/margin_over_bar"] = gl["accuracy"] - gl["bar"]
        log[f"{leg}/finish_length_rate"] = gl.get("finish_length_rate")
        log[f"{leg}/dev307_accuracy"] = gl.get("dev307_acc")
        log[f"{leg}/delta_vs_dev307"] = (
            gl["accuracy"] - gl["dev307_acc"] if gl.get("dev307_acc") is not None else None)
        if leg == "gpqa_diamond":
            log["gpqa_diamond/n_seeds"] = gl["n_seeds"]
            log["gpqa_diamond/ci95_lo_wilson"] = gl["ci95_wilson"][0]
            log["gpqa_diamond/ci95_hi_wilson"] = gl["ci95_wilson"][1]
            log["gpqa_diamond/ci_lo_clears_bar"] = int(gl["gpqa_ci_lo_clears_bar"])
            log["gpqa_diamond/n_scored"] = gl["n"]
            # de-confounded (excl gb6144 ctx-fit overflow item), mirroring #624
            if gl.get("accuracy_excl_request_error") is not None:
                log["gpqa_diamond/accuracy_deconf"] = gl["accuracy_excl_request_error"]
                log["gpqa_diamond/n_scored_deconf"] = gl.get("n_scored_excl_request_error")
                log["gpqa_diamond/n_request_error"] = gl.get("n_request_error")
                dcw = gl.get("ci95_wilson_excl_request_error") or [None, None]
                log["gpqa_diamond/ci95_lo_wilson_deconf"] = dcw[0]
                log["gpqa_diamond/ci95_hi_wilson_deconf"] = dcw[1]
                log["gpqa_diamond/ci_lo_clears_bar_deconf"] = int(gl.get("gpqa_ci_lo_clears_bar_deconf", False))

    # GPQA per-seed table
    gp = gates.get("gpqa_diamond")
    if gp and gp.get("per_seed_acc"):
        tbl = wandb.Table(columns=["seed", "accuracy", "n_correct", "n_scored"])
        for r in gp["per_seed_acc"]:
            tbl.add_data(str(r["seed"]), r["accuracy"], r["n_correct"], r["n_scored"])
        log["gpqa_diamond/per_seed"] = tbl

    # verdict booleans
    log["verdict/is_crater"] = int(panel["is_crater"])
    log["verdict/panel_all_gates_pass"] = int(panel["panel_all_gates_pass"])
    log["verdict/gpqa_ci_lo_clears_bar"] = int(panel["gpqa_ci_lo_clears_bar"])
    log["verdict/optionb_healthy_on_0p22"] = int(panel["optionb_healthy_on_0p22"])
    log["verdict/serves_on_0p22"] = int(panel["serves_on_0p22"])

    wandb.log(log)
    run.summary["verdict"] = panel["verdict"]
    run.summary["optionb_healthy_on_0p22"] = panel["optionb_healthy_on_0p22"]
    run.summary["serves_on_0p22"] = panel["serves_on_0p22"]
    run.summary["gpqa_ci_lo_clears_bar"] = panel["gpqa_ci_lo_clears_bar"]
    run.summary["is_crater"] = panel["is_crater"]
    run.summary["crater_legs"] = panel["crater_legs"]
    run.summary["surface_to_human"] = True  # non-#319-identical: never auto-fire
    for leg in ("mmlu_pro", "gsm8k", "aime", "gpqa_diamond"):
        if leg in gates:
            run.summary[f"{leg}_accuracy"] = gates[leg]["accuracy"]
            run.summary[f"{leg}_finish_length_rate"] = gates[leg].get("finish_length_rate")
    rid = run.id
    run.finish()
    print(f"[wandb] logged panel run id={rid} verdict={panel['verdict']}")
    print(f"[wandb] group={GROUP} project={ENTITY}/{PROJECT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
