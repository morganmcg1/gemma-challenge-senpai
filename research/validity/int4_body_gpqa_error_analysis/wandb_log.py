#!/usr/bin/env python3
"""PR #619 -- log the int4-body gpqa_main failure-mode analysis + max_tokens recovery
to W&B. Reads breakdown.json (mandatory) and recover_aggregate.json (if the recovery
re-run ran). Group = int4-body-gpqa-error-analysis. analysis_only, NO FIRE. LOCAL."""
import json
import os
import sys
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
HERE = ROOT / "research/validity/int4_body_gpqa_error_analysis"
os.environ.setdefault("WANDB_DIR", str(HERE))
sys.path.append(str(ROOT))

from scripts.wandb_logging import (  # noqa: E402
    init_wandb_run, log_event, log_json_artifact, finish_wandb,
)

bd = json.load(open(HERE / "breakdown.json"))
rec_path = HERE / "recover_aggregate.json"
rec = json.load(open(rec_path)) if rec_path.exists() else None

run = init_wandb_run(
    job_type="analysis",
    agent="land",
    name="land/int4-body-gpqa-error-analysis",
    group="int4-body-gpqa-error-analysis",
    notes="PR #619: WHY does the int4 body regress gpqa_main (#598: McNemar p=0.0009, "
          "n01=309 base-right/int4-wrong)? Read the McNemar-discordant cells and tag each "
          "failure mode: (i) truncated, (ii) first-token-EOS, (iii) extraction artifact, "
          "(iv) genuine reasoning error, (v) domain clustering. THEN test the dominant "
          "recoverable knob (max_tokens 3072->6144) on the truncated cells, both arms, to "
          "see if the deficit shrinks. Verdict: serving-recoverable vs fundamental precision "
          "loss. Reuses #598 run n4ro7bzk artifacts. analysis_only, NO FIRE.",
    tags=["pr-619", "analysis-only", "no-fire", "int4-body-gpqa-error-analysis",
          "gpqa-main", "mcnemar", "failure-mode", "truncation", "max-tokens-recovery",
          "base-fullhead", "quality-gate"],
    config={
        "pr": 619, "analysis_only": True, "official_tps": 0, "no_hf_job": True,
        "wandb_group": "int4-body-gpqa-error-analysis",
        "source_run": bd["source_run"], "instrument": bd["instrument"],
        "n_items": bd["n_items"], "K_seeds": bd["K_seeds"],
        "base_checkpoint": "google/gemma-4-E4B-it (UNQUANTIZED bf16, gate denominator)",
        "int4_checkpoint": "google/gemma-4-E4B-it-qat-w4a16-ct (int4 W4A16 g32 body, native head)",
        "protocol_598": {"temperature": 1.0, "top_p": 0.95, "top_k": 64,
                         "min_tokens": 8, "max_tokens": 3072, "max_model_len": 6144,
                         "dataset_seed": 12345},
        "recovery_knob": (rec["knob"] if rec else None),
        "vllm_build": "vllm-0.22.1rc1.dev307+g3e8afdf78",
        "inspect_ai": "0.3.240", "inspect_evals": "0.14.0",
    },
)

# scalar metrics from the breakdown
metrics = {}
for k, v in bd["category_counts_n01"].items():
    metrics[f"category_n01/{k}"] = v
for k, v in bd["truncation"].items():
    metrics[f"truncation/{k}"] = v
for k, v in bd["verbosity_n01"].items():
    metrics[f"verbosity/{k}"] = v
for k, v in bd["domain_n01_rate_per_1000_shared"].items():
    metrics[f"domain_n01_rate/{k}"] = v
metrics["mcnemar/n01_base_right_int4_wrong"] = bd["n01_base_right_int4_wrong"]
metrics["mcnemar/n10_base_wrong_int4_right"] = bd["n10_base_wrong_int4_right"]
metrics["mcnemar/net_margin"] = bd["net_mcnemar_margin"]
log_event(run, "int4_gpqa_failure_breakdown", step=0, metrics=metrics)

if rec is not None:
    rm = {}
    for phase in ("BEFORE_3072", "AFTER_6144"):
        for k, v in rec[phase].items():
            if isinstance(v, (int, float)):
                rm[f"{phase}/{k}"] = v
    rm["recovery/deficit_before"] = rec["deficit_before"]
    rm["recovery/deficit_after"] = rec["deficit_after"]
    rm["recovery/deficit_shrink"] = rec["deficit_shrink"]
    rm["recovery/deficit_shrink_pct"] = rec["deficit_shrink_pct"]
    for arm in ("recovery_base", "recovery_int4"):
        for k, v in rec[arm].items():
            rm[f"{arm}/{k}"] = v
    log_event(run, "int4_gpqa_max_tokens_recovery", step=1, metrics=rm)

if run is not None:
    import wandb

    run.summary["n01_base_right_int4_wrong"] = bd["n01_base_right_int4_wrong"]
    run.summary["n10_base_wrong_int4_right"] = bd["n10_base_wrong_int4_right"]
    for k, v in bd["category_counts_n01"].items():
        run.summary[f"category_n01_{k}"] = v
    run.summary["int4_truncated_pct_wrong"] = bd["truncation"]["int4_truncated_pct_wrong"]
    run.summary["net_truncation_pct_of_margin"] = bd["truncation"]["pct_of_net_margin"]
    if rec is not None:
        run.summary["deficit_before_3072"] = rec["deficit_before"]
        run.summary["deficit_after_6144"] = rec["deficit_after"]
        run.summary["deficit_shrink"] = rec["deficit_shrink"]
        run.summary["deficit_shrink_pct"] = rec["deficit_shrink_pct"]
        run.summary["mcnemar_p_after_6144"] = rec["AFTER_6144"]["mcnemar_p_exact"]
        run.summary["mcnemar_p_after_gt_0p05"] = rec["mcnemar_p_after_gt_0p05"]

    # n01 category table
    ct = wandb.Table(columns=["category", "n01_cells", "recoverable"])
    rec_map = {"i_truncated": "yes (budget)", "ii_empty": "yes (min_tokens)",
               "iii_extraction": "yes (parser)", "iv_genuine": "no (fundamental)"}
    for k, v in bd["category_counts_n01"].items():
        ct.add_data(k, v, rec_map.get(k, "?"))
    run.log({"n01_category_table": ct})

    if rec is not None:
        bt = wandb.Table(columns=["phase", "base_acc", "int4_acc", "deficit",
                                  "mcnemar_n01", "mcnemar_n10", "mcnemar_p"])
        for phase in ("BEFORE_3072", "AFTER_6144"):
            d = rec[phase]
            bt.add_data(phase, d["base_mean_acc"], d["int4_mean_acc"], d["deficit"],
                        d["n01_base_right_int4_wrong"], d["n10_base_wrong_int4_right"],
                        d["mcnemar_p_exact"])
        run.log({"recovery_before_after_table": bt})
        log_json_artifact(run, name="int4_gpqa_recover_aggregate", artifact_type="analysis",
                          data=rec)

    log_json_artifact(run, name="int4_gpqa_failure_breakdown", artifact_type="analysis", data=bd)

print("wandb run:", getattr(run, "id", None), getattr(run, "name", None))
if run is not None:
    (HERE / "wandb_run_id.txt").write_text(str(run.id) + "\n")
finish_wandb(run)
