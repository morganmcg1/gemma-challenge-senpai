#!/usr/bin/env python3
"""PR #598 -- log the gpqa_main (larger-instrument n=448) GPQA CI-robustness + paired
no-regression verdicts to W&B. Reads aggregate.json and emits config + summary metrics
+ per-seed tables (both configs) + a verdict table + the raw aggregate as an artifact.
Group = gpqa-larger-ci. analysis_only, NO FIRE (official_tps=0). LOCAL."""
import json
import os
import sys
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
HERE = ROOT / "research/validity/gpqa_larger_instrument_ci"
os.environ.setdefault("WANDB_DIR", str(HERE))
sys.path.append(str(ROOT))  # append so site-packages wandb beats the ./wandb output dir

from scripts.wandb_logging import (  # noqa: E402
    init_wandb_run, log_event, log_json_artifact, finish_wandb,
)

agg = json.load(open(HERE / "aggregate.json"))

run = init_wandb_run(
    job_type="analysis",
    agent="land",
    name="land/gpqa-larger-ci",
    group="gpqa-larger-ci",
    notes="PR #598: can the GPQA gate be made CI-robust on a LARGER instrument "
          "(gpqa_main n=448, 2.26x Diamond) by re-anchoring at 0.90x vanilla-base on "
          "that instrument -- does base_fullhead's 95% CI-lb clear? AND is base_fullhead "
          "(int4 body) a regression vs vanilla base under paired McNemar on shared "
          "sampling seeds (only the model differs)? K=5 seeds, generation_config.json "
          "sampling (temp=1.0/top_p=0.95/top_k=64, lewtun #31), min_tokens=8 (#541). "
          "Head-invariant (stark #536) -> certifies the int4-body family. analysis_only, NO FIRE.",
    tags=["pr-598", "analysis-only", "no-fire", "gpqa-larger-ci", "sampling-protocol",
          "quality-gate", "base-fullhead", "mcnemar", "ci-robustness", "larger-instrument"],
    config={
        "pr": 598,
        "analysis_only": True,
        "official_tps": 0,
        "no_hf_job": True,
        "wandb_group": "gpqa-larger-ci",
        "task": "gpqa_main",
        "instrument_n": agg["instrument_n"],
        "gpqa_main_ceiling_n": agg["gpqa_main_ceiling_n"],
        "gpqa_extended_ceiling_n": agg["gpqa_extended_ceiling_n"],
        "quality_frac": agg["quality_frac"],
        "reanchored_gate": agg["reanchored_gate"],
        "K_seeds_base": agg["K_seeds_base"],
        "K_seeds_base_fullhead": agg["K_seeds_base_fullhead"],
        "dataset_seed": 12345,
        "protocol": agg["protocol"],
        "serve": agg["serve"],
        "base_checkpoint": "google/gemma-4-E4B-it (UNQUANTIZED bf16, the gate denominator)",
        "base_fullhead_checkpoint": (
            "google/gemma-4-E4B-it-qat-w4a16-ct (int4 W4A16 g32 body, native 262k bf16 lm_head)"),
        "gpqa_mirror": "Wanfq/gpqa (verbatim original CSVs, 78-col schema incl. Canary String; "
                       "SHA256-pinned; main=448/extended=546/diamond=198 row-exact)",
        "vllm_build": "vllm-0.22.1rc1.dev307+g3e8afdf78",
        "inspect_ai": "0.3.240",
        "inspect_evals": "0.14.0",
        "n_prompt_mismatch": agg["n_prompt_mismatch"],
    },
)

# scalar metrics: everything numeric/bool from aggregate.json
metrics = {}
for k, v in agg.items():
    if isinstance(v, bool):
        metrics[f"verdict/{k}"] = 1 if v else 0
    elif isinstance(v, (int, float)):
        metrics[f"gpqa/{k}"] = v
log_event(run, "gpqa_larger_ci_verdict", step=0, metrics=metrics,
          data={
              "gpqa_main_ci_lb_clears_90pct_base": agg["gpqa_main_ci_lb_clears_90pct_base"],
              "base_fullhead_not_regression_sampling": agg["base_fullhead_not_regression_sampling"],
              "primary_cilb_basis": agg["primary_cilb_basis"],
              "mcnemar_direction": agg["mcnemar_direction"],
          })

if run is not None:
    import wandb

    # pin headline numbers + the two primary verdict bools into run.summary
    for k in (
        "instrument_n", "K_seeds_base", "K_seeds_base_fullhead", "n_prompt_mismatch",
        "base_mean_acc", "base_std_acc", "base_min_acc", "base_max_acc",
        "base_fullhead_mean_acc", "base_fullhead_std_acc",
        "base_fullhead_min_acc", "base_fullhead_max_acc",
        "base_point_item_mean", "base_fullhead_point_item_mean",
        "reanchored_gate",
        "fh_primary_cilb", "fh_bootstrap_cilb_items", "fh_bootstrap_cihi_items",
        "fh_wilson_cilb_n448", "fh_wilson_cihi_n448",
        "fh_primary_cilb_margin", "fh_wilson_cilb_margin",
        "worst_seed_margin", "point_margin",
        "base_fullhead_n_seeds_below_gate",
        "mcnemar_shared_cells", "mcnemar_n01_base_right_fh_wrong",
        "mcnemar_n10_base_wrong_fh_right", "mcnemar_n00", "mcnemar_n11",
        "mcnemar_p_exact", "mcnemar_p_chi2_cc", "mcnemar_chi2_cc",
        "sign_test_n_base_gt", "sign_test_n_fh_gt", "sign_test_n_tie", "sign_test_p",
        "mean_item_diff_base_minus_fh",
        "n_for_wilson_cilb_pass_at_point",
    ):
        run.summary[k] = agg[k]
    for k in (
        "gpqa_main_ci_lb_clears_90pct_base",   # PRIMARY verdict
        "base_fullhead_not_regression_sampling",  # NO-REGRESSION verdict
        "fh_bootstrap_clears", "fh_wilson_n448_clears", "worst_seed_clears",
        "point_clears", "fh_cilb_ge_0p90_base_cilb",
        "mcnemar_direction",
        "ci_untightenable_on_gpqa_main", "ci_untightenable_on_any_gpqa",
    ):
        run.summary[k] = agg[k]

    # per-seed tables, both configs
    for cfg, tbl_key in (("base", "base_seed_table"),
                         ("base_fullhead", "base_fullhead_seed_table")):
        t = wandb.Table(columns=["config", "sampling_seed", "accuracy", "n_correct",
                                 "n_scored", "n_empty", "empty_rate", "min_tokens",
                                 "below_reanchored_gate"])
        for r in agg[tbl_key]:
            t.add_data(cfg, r["sampling_seed"], r["accuracy"], r["n_correct"],
                       r["n_scored"], r["n_empty"], r.get("empty_rate"),
                       r.get("min_tokens"),
                       bool(r["accuracy"] < agg["reanchored_gate"]))
        run.log({f"per_seed_table_{cfg}": t})

    # compact verdict / CI-lens summary table
    vt = wandb.Table(columns=["lens", "value", "gate", "clears"])
    vt.add_data("primary_clustered_bootstrap_cilb", agg["fh_primary_cilb"],
                agg["reanchored_gate"], agg["fh_bootstrap_clears"])
    vt.add_data("item_wilson_cilb_n448", agg["fh_wilson_cilb_n448"],
                agg["reanchored_gate"], agg["fh_wilson_n448_clears"])
    vt.add_data("worst_single_seed", agg["base_fullhead_min_acc"],
                agg["reanchored_gate"], agg["worst_seed_clears"])
    vt.add_data("point_mean", agg["base_fullhead_mean_acc"],
                agg["reanchored_gate"], agg["point_clears"])
    run.log({"ci_lens_table": vt})

    # leave behind the full raw aggregate as a rich artifact
    log_json_artifact(run, name="gpqa_larger_ci_aggregate", artifact_type="analysis",
                      data=agg)

print("wandb run:", getattr(run, "id", None), getattr(run, "name", None))
if run is not None:
    (HERE / "wandb_run_id.txt").write_text(str(run.id) + "\n")
finish_wandb(run)
