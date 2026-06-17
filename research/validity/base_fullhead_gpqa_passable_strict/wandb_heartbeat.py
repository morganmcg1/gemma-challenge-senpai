#!/usr/bin/env python3
"""Early W&B heartbeat for PR #574 (liveness) — inits the run, logs config + a running
status, writes the run id to wandb_run_id.txt so analyze.py can resume the SAME run to
attach final metrics. LOCAL analysis_only, NO FIRE."""
import os
import sys
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
HERE = ROOT / "research/validity/base_fullhead_gpqa_passable_strict"
# A wandb OUTPUT dir lives at the target root and shadows the real `wandb` package if
# ROOT is on the FRONT of sys.path. Append ROOT (so site-packages' real wandb wins for
# `import wandb`, while `scripts.*` still resolves) and steer run files into HERE.
os.environ.setdefault("WANDB_DIR", str(HERE))
sys.path.append(str(ROOT))
from scripts.wandb_logging import init_wandb_run, log_event, finish_wandb  # noqa: E402

run = init_wandb_run(
    job_type="analysis",
    agent="stark",
    name="stark/base-fullhead-gpqa-passable-strict",
    group="base-fullhead-gpqa-passable-strict",
    notes="PR #574: replace the un-passable GPQA Wilson CI-lb lens (#564) with the standard "
          "small-n PAIRED replacements — a paired bootstrap CI on the gate margin and a "
          "Beta-posterior on the clear-probability — computed on the banked #564 (7bi4e2ne) "
          "per-item GPQA-Diamond paired data (198 items). 0-GPU re-analysis. analysis_only, NO FIRE.",
    tags=["pr-574", "analysis-only", "no-fire", "gpqa", "passable-strict", "bootstrap",
          "beta-posterior", "greedy-protocol"],
    config={
        "pr": 574,
        "analysis_only": True,
        "official_tps": 0,
        "no_hf_job": True,
        "zero_gpu_reanalysis": True,
        "data_source_run": "7bi4e2ne",
        "data_source_pr": 564,
        "vllm_build": "vllm-0.22.1rc1.dev307+g3e8afdf78 (banked, not re-served)",
        "checkpoint": "google/gemma-4-E4B-it-qat-w4a16-ct (stock int4, native 262k head)",
        "wandb_group": "base-fullhead-gpqa-passable-strict",
        "protocol": "greedy temp=0/top_p=1/top_k=0 (program-record protocol; kanna #563 owns sampling)",
        "morgan_515_gate": "base_fullhead >= 0.90 x vanilla base",
        "gpqa_diamond_full_n": 198,
        "paired_2x2_both_correct": 75,
        "paired_2x2_fullhead_only": 20,
        "paired_2x2_ple_fold_only": 23,
        "paired_2x2_both_wrong": 80,
        "base_fullhead_k": 95,
        "ple_fold_k": 98,
        "denominator_primary_ple_fold": 0.494949,
        "denominator_ubel_511": 0.470,
        "denominator_base_gpqa_json": 0.4444,
        "beta_pass_threshold": 0.95,
        "cites": ["#564 7bi4e2ne (data + Wilson miss)", "#557 yw6vwk1w (ple_fold denom)",
                  "#542 92pcnx6a (z-GO)", "#515 Morgan gate", "#511 ubel anchor",
                  "#563 kanna sampling-axis (disjoint)"],
    },
)
if run is None:
    print("[wandb] init returned None (no key/disabled)")
    raise SystemExit(0)

log_event(run, "started", step=0, metrics={"status_code": 3})
(HERE / "wandb_run_id.txt").write_text(run.id)
print(f"[wandb] run id={run.id} group=base-fullhead-gpqa-passable-strict")
finish_wandb(run)
