#!/usr/bin/env python3
"""Persistent W&B run for the option-C sub-int4 Pareto (PR #611).

One run is created and then *resumed* across the separate stage processes
(build / feasibility / proxy-tps / ppl / quality-gate) so all evidence lands on
a single comparable run. Run id is persisted to ``run_id.txt`` beside this file.

LOCAL / analysis_only / NO FIRE.  Use system python3 (it has a working wandb).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_ID_FILE = HERE / "run_id.txt"
PROJECT = os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai")
ENTITY = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
GROUP = "optionc-subint4-compression-pareto"
NAME = "stark/optionc-subint4-pareto"


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=HERE, text=True,
                              capture_output=True, check=False).stdout.strip()
    except Exception:
        return ""


def init_or_resume(extra_config: dict | None = None):
    import wandb
    rid = RUN_ID_FILE.read_text().strip() if RUN_ID_FILE.exists() else wandb.util.generate_id()
    run = wandb.init(
        project=PROJECT, entity=ENTITY, id=rid, resume="allow",
        name=NAME, group=GROUP, job_type="optionc-subint4-pareto",
        tags=["gemma-challenge", "optionc", "subint4", "analysis-only", "pr-611",
              "weight-compression", "pareto", "no-fire"],
        config={
            "pr": 611, "analysis_only": True, "official_tps": 0, "no_fire": True,
            "engine": "vllm-dev307 (0.22.1rc1.dev307+g3e8afdf78)",
            "baseline_tps": 126.378, "baseline_ppl": 2.019,
            "baseline_bytes_per_token_gib": 9.85,
            "gate_ppl_max": 2.42,
            "gate_bars": {"gsm8k": 0.807, "mmlu_pro": 0.605,
                          "gpqa_diamond": 0.471, "aime": 0.090},
            "gate_base_refs": {"gsm8k": 0.8967, "mmlu_pro": 0.6727,
                               "gpqa_diamond": 0.5236, "aime": 0.100},
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("branch", "--show-current"),
            **(extra_config or {}),
        },
    )
    RUN_ID_FILE.write_text(run.id)
    return run


if __name__ == "__main__":
    run = init_or_resume()
    run.summary["phase0/disk_free_gb_at_pickup"] = 132
    run.summary["phase0/disk_threshold_gb"] = 170
    run.summary["phase0/proceeding"] = True
    run.summary["phase0/reason"] = (
        "source+int4 body pre-built; no concurrent writers; incremental builds")
    run.summary["env/dense_source"] = "/workspace/gemma_build/qat_unq (15.9GB)"
    run.summary["env/int4_body"] = "/workspace/gemma_build/int4_g128_lmhead (10.3GB)"
    run.summary["env/mlp_modules"] = 126
    run.summary["env/attn_modules"] = 132
    run.summary["env/ple_modules"] = 85
    run.log({"global_step": 0, "stage": 0, "event/name": "liveness_init"})
    print(json.dumps({"run_id": run.id, "url": run.url, "group": GROUP}))
    run.finish()
