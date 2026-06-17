#!/usr/bin/env python3
"""PR #589 — log the base_fullhead GPQA-D sampling CI-tighten verdict to W&B.
Reads aggregate.json and emits config + summary metrics + per-seed table.
Group = gpqa-ci-tighten. analysis_only, NO FIRE (official_tps=0). LOCAL."""
import json
import os
import sys
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
HERE = ROOT / "research/validity/base_fullhead_gpqa_ci_tighten"
os.environ.setdefault("WANDB_DIR", str(HERE))
sys.path.append(str(ROOT))  # append (not prepend) so site-packages wandb beats the ./wandb output dir

from scripts.wandb_logging import init_wandb_run, log_event, finish_wandb  # noqa: E402

agg = json.load(open(HERE / "aggregate.json"))

run = init_wandb_run(
    job_type="analysis",
    agent="land",
    name="land/gpqa-ci-tighten",
    group="gpqa-ci-tighten",
    notes="PR #589: is base_fullhead's GPQA-Diamond gate margin (point 0.4798 vs >=0.471, "
          "+0.009, the thinnest of 4 quality gates) CI-robust under the actual downstream "
          "SAMPLING protocol (generation_config.json temp=1.0/top_p=0.95/top_k=64, lewtun #31) "
          "across K seeds, min_tokens=8 (#541)? Verdict gpqa_ci_lb_clears_0471. analysis_only, NO FIRE.",
    tags=["pr-589", "analysis-only", "no-fire", "gpqa-ci-tighten", "sampling-protocol",
          "quality-gate", "base-fullhead"],
    config={
        "pr": 589,
        "analysis_only": True,
        "official_tps": 0,
        "no_hf_job": True,
        "wandb_group": "gpqa-ci-tighten",
        "gate_abs": agg["gate_abs"],
        "task": "gpqa_diamond",
        "n_items": agg["n_items"],
        "K_seeds": agg["K_seeds"],
        "dataset_seed": 12345,
        "protocol": agg["protocol"],
        "serve": agg["serve"],
        "checkpoint": "google/gemma-4-E4B-it-qat-w4a16-ct (int4 W4A16 g32 body, native 262k bf16 lm_head)",
        "vllm_build": "vllm-0.22.1rc1.dev307+g3e8afdf78",
        "inspect_ai": "0.3.240",
        "inspect_evals": "0.14.0",
        "n_prompt_mismatch": agg["n_prompt_mismatch"],
    },
)

# scalar metrics (everything numeric from aggregate.json)
metrics = {}
for k, v in agg.items():
    if isinstance(v, bool):
        metrics[f"verdict/{k}"] = 1 if v else 0
    elif isinstance(v, (int, float)):
        metrics[f"gpqa/{k}"] = v
log_event(run, "ci_tighten_verdict", step=0, metrics=metrics,
          data={"gpqa_ci_lb_clears_0471": agg["gpqa_ci_lb_clears_0471"],
                "primary_cilb_basis": agg["primary_cilb_basis"]})

# pin headline numbers + verdict bools into run.summary
if run is not None:
    for k in ("mean_acc", "std_acc", "min_seed_acc", "max_seed_acc",
              "wilson_cilb_n198", "bootstrap_cilb_items", "wilson_cilb_pooled",
              "seed_mean_cilb", "primary_cilb", "n_seeds_below_gate",
              "n_for_wilson_cilb_pass_at_point", "gate_abs", "K_seeds"):
        run.summary[k] = agg[k]
    for k in ("gpqa_ci_lb_clears_0471", "worst_seed_clears_0471",
              "wilson_n198_clears_0471", "wilson_pooled_clears_0471",
              "seed_mean_clears_0471", "point_clears_0471",
              "ci_untightenable_on_diamond"):
        run.summary[k] = agg[k]

    import wandb
    tbl = wandb.Table(columns=["sampling_seed", "accuracy", "n_correct", "n_scored",
                               "n_empty", "min_tokens", "below_gate_0471"])
    for r in agg["seed_table"]:
        tbl.add_data(r["sampling_seed"], r["accuracy"], r["n_correct"], r["n_scored"],
                     r["n_empty"], r["min_tokens"], r["below_gate"])
    run.log({"per_seed_table": tbl})

print("wandb run:", getattr(run, "id", None), getattr(run, "name", None))
if run is not None:
    (HERE / "wandb_run_id.txt").write_text(str(run.id) + "\n")
finish_wandb(run)
