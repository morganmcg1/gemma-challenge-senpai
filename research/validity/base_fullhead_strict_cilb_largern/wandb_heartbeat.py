#!/usr/bin/env python3
"""Early W&B heartbeat for PR #564 (liveness) — inits the run, logs config + a running
status, writes the run id to wandb_run_id.txt so aggregate.py can resume the SAME run to
attach final metrics. LOCAL analysis_only, NO FIRE."""
import sys
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))
from scripts.wandb_logging import init_wandb_run, log_event, finish_wandb  # noqa: E402

HERE = ROOT / "research/validity/base_fullhead_strict_cilb_largern"

run = init_wandb_run(
    job_type="analysis",
    agent="stark",
    name="stark/base-fullhead-strict-cilb-largern",
    group="base-fullhead-strict-cilb-largern",
    notes="PR #564: settle the #557 strict-Wilson-CI-lb miss by re-measuring base_fullhead "
          "(surgical 2D attn + PLE fold) AND the recovered ple_fold vanilla base at LARGER "
          "MMLU-Pro n, recompute whether base_fullhead CI-lb clears 0.90x the ple_fold point, "
          "and test (paired McNemar) whether the residual point gap is surgical-attention cost "
          "or sampling noise. analysis_only, NO FIRE.",
    tags=["pr-564", "analysis-only", "no-fire", "strict-cilb", "ci-width", "greedy-protocol"],
    config={
        "pr": 564,
        "analysis_only": True,
        "official_tps": 0,
        "no_hf_job": True,
        "vllm_build": "vllm-0.22.1rc1.dev307+g3e8afdf78",
        "checkpoint": "google/gemma-4-E4B-it-qat-w4a16-ct (stock int4, native 262k head)",
        "wandb_group": "base-fullhead-strict-cilb-largern",
        "protocol": "greedy temp=0/top_p=1/top_k=0 (the program-record protocol; kanna #563 owns sampling)",
        "arm_base_fullhead": "FA_SLIDING=1 + SURGICAL_ATTN_USE_3D_OFF=1 + PLE_FOLD_EMBED_SCALE=1 (surgical 2D attn + fold)",
        "arm_ple_fold": "PLE_FOLD_EMBED_SCALE=1 only (plain vanilla 3D-default TRITON + fold)",
        "strict_cilb_rule": "base_fullhead Wilson CI-lb >= 0.90 x ple_fold point estimate",
        # #557 n=500/198 starting margins this card re-measures at larger n:
        "ref_557_mmlu_base_fullhead": 0.636,
        "ref_557_mmlu_ple_fold": 0.662,
        "ref_557_mmlu_cilb": 0.593,
        "ref_557_mmlu_gate_0p90x": 0.596,
        "ref_557_gpqa_base_fullhead": 0.4697,
        "ref_557_gpqa_ple_fold": 0.4848,
        "ref_557_gpqa_cilb": 0.401,
        "ref_557_gpqa_gate_0p90x": 0.436,
        "gpqa_diamond_full_n": 198,
        "gpqa_ci_untightenable_at_dataset_ceiling": True,
        "morgan_515_gate": "base_fullhead >= 0.90 x vanilla base",
        "cites": ["#557 yw6vwk1w", "#542 92pcnx6a", "#515 Morgan gate", "#511 ubel anchor", "#563 kanna protocol-axis"],
    },
)
if run is None:
    print("[wandb] init returned None (no key/disabled)")
    raise SystemExit(0)

log_event(run, "started", step=0, metrics={"status_code": 3})
(HERE / "wandb_run_id.txt").write_text(run.id)
print(f"[wandb] run id={run.id} group=base-fullhead-strict-cilb-largern")
finish_wandb(run)
