#!/usr/bin/env python3
"""Early W&B heartbeat for PR #557 (liveness) — inits the run, logs config + a
running status, writes the run id to wandb_run_id.txt so the aggregator can
resume the SAME run to attach final metrics. LOCAL analysis_only, NO FIRE."""
import sys
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))
from scripts.wandb_logging import init_wandb_run, log_event, finish_wandb  # noqa: E402

HERE = ROOT / "research/validity/vanilla_base_serve_regression"

run = init_wandb_run(
    job_type="analysis",
    agent="stark",
    name="stark/vanilla-base-serve-regression",
    group="vanilla-base-serve-regression",
    notes="PR #557: root-cause the dev307 Gemma4 forced-TRITON_ATTN vanilla-serve "
          "regression and recover a healthy quality denominator. analysis_only, NO FIRE.",
    tags=["pr-557", "analysis-only", "no-fire", "denominator-integrity"],
    config={
        "pr": 557,
        "analysis_only": True,
        "official_tps": 0,
        "no_hf_job": True,
        "vllm_build": "vllm-0.22.1rc1.dev307+g3e8afdf78",
        "checkpoint": "google/gemma-4-E4B-it-qat-w4a16-ct (stock int4, native 262k head)",
        "wandb_group": "vanilla-base-serve-regression",
        "regression_cause_hypothesis": "Gemma4Config.verify_and_update_config forces "
            "TRITON_ATTN for heterogeneous head dims (head_dim=256 sliding / "
            "global_head_dim=512 full); TRITON degenerates long-CoT decode.",
        "anchor_mmlu": 0.668,
        "anchor_gpqa": 0.470,
        "broken_base_mmlu_542": 0.432,
        "broken_base_gpqa_542": 0.3131,
    },
)
if run is None:
    print("[wandb] init returned None (no key/disabled)")
    raise SystemExit(0)

log_event(run, "started", step=0, metrics={"status_code": 3})
(HERE / "wandb_run_id.txt").write_text(run.id)
print(f"[wandb] run id={run.id} group=vanilla-base-serve-regression")
finish_wandb(run)
