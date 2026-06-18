#!/usr/bin/env python
"""PR #634 (fern) — log the GPQA-D dev307 leg + paired A/B vs #629's 0.22.0 to W&B.

Single-variable (engine) A/B: same Option-B stack (int4_g128_lmhead + MTP-K7, BI=1,
max_model_len 8192, max_tokens 6144, conc 16, T=1.0/top_p0.95/top_k64/min_tokens8),
SAME 10 dataset seeds, SAME 198 GPQA-D questions/seed. ONLY change = engine 0.22.0->dev307.
Disambiguates: GPQA model-limited (fails on both engines) vs engine-specific (dev307 lifts).

LOCAL ONLY, analysis_only=True, official_tps=0. Group `optionb-gpqa-dev307-10seed`.
This config is NOT #319 byte-identical (spec) and dev307 has served-nondeterminism
(lawine #606/#610) -> any dev307>0.22.0 delta is determinism-caveated, SURFACE-to-human.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import wandb

GROUP = "optionb-gpqa-dev307-10seed"
HERE = Path(__file__).resolve().parent
ENTITY = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
PROJECT = os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai")

CONFIG = {
    "config_name": "int4_g128_lmhead + MTP-K7 spec (fern #597)",
    "model_id": "/workspace/gemma_build/int4_g128_lmhead",
    "drafter": "/tmp/qat-assistant",
    "num_speculative_tokens": 7,
    "vllm_batch_invariant": 1,
    "engine": "dev307 (0.22.1rc1.dev307+g3e8afdf78)",
    "vllm_version": "0.22.1rc1.dev307",
    "max_model_len": 8192,
    "max_num_seqs": 16,
    "min_tokens": 8,
    "max_tokens": 6144,
    "gpqa_sampling": "T=1.0, top_p=0.95, top_k=64",
    "gpqa_sampling_seed": 0,
    "task": "gpqa_diamond",
    "n_seeds": 10,
    "is_319_identical": False,
    "analysis_only": True,
    "official_tps": 0,
    "comparison_baseline": "fern #629 0.22.0 10-seed GPQA leg (run 2jhhk0u3)",
    "bar_gpqa": 0.471,
}


def main() -> int:
    d = json.loads((HERE / "paired_dev307_vs_0p22.json").read_text())

    run = wandb.init(
        project=PROJECT, entity=ENTITY, name="fern/optionb-gpqa-dev307-10seed",
        group=GROUP, job_type="gpqa-engine-ab", reinit=True,
        config={**CONFIG, "seeds": d["seeds"]},
    )

    log = {}
    # dev307 headline
    log["gpqa/dev307_acc"] = d["gpqa_dev307_10seed_acc"]
    log["gpqa/dev307_n_correct"] = d["dev307_n_correct"]
    log["gpqa/dev307_n_scored"] = d["dev307_n_scored"]
    log["gpqa/dev307_ci_lo_wilson"] = d["dev307_ci95_wilson"][0]
    log["gpqa/dev307_ci_hi_wilson"] = d["dev307_ci95_wilson"][1]
    log["gpqa/dev307_ci_lo_clears_bar"] = int(d["gpqa_dev307_ci_lo_clears_bar"])
    log["gpqa/dev307_point_clears_bar"] = int(d["dev307_point_clears_bar"])
    log["gpqa/dev307_margin_over_bar"] = d["gpqa_dev307_10seed_acc"] - d["bar"]
    # de-confounded
    log["gpqa/dev307_acc_deconf"] = d["de_confounded_acc"]
    log["gpqa/dev307_n_scored_deconf"] = d["de_confounded_n"]
    log["gpqa/dev307_n_request_error"] = d["de_confounded_n_request_error"]
    log["gpqa/dev307_deconf_pass"] = int(d["de_confounded_pass"])
    log["gpqa/dev307_ci_lo_wilson_deconf"] = d["de_confounded_ci95_wilson"][0]
    log["gpqa/dev307_ci_hi_wilson_deconf"] = d["de_confounded_ci95_wilson"][1]
    # 0.22.0 banked A-arm
    log["gpqa/p0p22_acc"] = d["gpqa_0p22_10seed_acc"]
    log["gpqa/p0p22_ci_lo_wilson"] = d["p0p22_ci95_wilson"][0]
    log["gpqa/p0p22_ci_hi_wilson"] = d["p0p22_ci95_wilson"][1]
    # seed-level paired t-test
    log["paired/mean_delta_dev_minus_0p22"] = d["paired_mean_delta_dev307_minus_0p22"]
    log["paired/delta_se"] = d["paired_delta_se"]
    log["paired/delta_t"] = d["paired_delta_t"]
    log["paired/delta_p_two_sided"] = d["paired_delta_p_two_sided"]
    log["paired/delta_ci_lo"] = d["paired_delta_ci95"][0]
    log["paired/delta_ci_hi"] = d["paired_delta_ci95"][1]
    log["paired/delta_ci_significant"] = int(d["paired_delta_ci_significant"])
    log["paired/delta_sig_positive"] = int(d["paired_delta_sig_positive"])
    # item-level McNemar
    log["mcnemar/paired_items"] = d["mcnemar_paired_items"]
    log["mcnemar/b_dev_correct_0p22_wrong"] = d["mcnemar_b_dev_correct_0p22_wrong"]
    log["mcnemar/c_0p22_correct_dev_wrong"] = d["mcnemar_c_0p22_correct_dev_wrong"]
    log["mcnemar/both_correct"] = d["mcnemar_both_correct"]
    log["mcnemar/both_wrong"] = d["mcnemar_both_wrong"]
    log["mcnemar/item_delta"] = d["mcnemar_item_delta_dev_minus_0p22"]
    log["mcnemar/p_two_sided"] = d["mcnemar_p_two_sided"]
    log["mcnemar/significant"] = int(d["mcnemar_significant"])
    log["mcnemar/prompt_sha_mismatch"] = d["prompt_sha_mismatch"]
    # bar-sensitivity
    log["bar_sensitivity/implied_base_for_0p22_clear"] = d["implied_base_for_0p22_clear"]
    log["bar_sensitivity/implied_base_for_dev307_clear"] = d["implied_base_for_dev307_clear"]

    # per-seed paired table
    tbl = wandb.Table(columns=["seed", "acc_dev307", "acc_0p22", "delta",
                               "n_correct_dev307", "n_scored_dev307"])
    for r in d["per_seed"]:
        tbl.add_data(str(r["seed"]), r["acc_dev307"], r["acc_0p22"],
                     r["delta_dev_minus_0p22"], r["n_correct_dev307"], r["n_scored_dev307"])
    log["gpqa/per_seed_paired"] = tbl

    wandb.log(log)

    run.summary["VERDICT"] = d["VERDICT"]
    run.summary["gpqa_dev307_10seed_acc"] = d["gpqa_dev307_10seed_acc"]
    run.summary["gpqa_dev307_ci_lo_clears_bar"] = d["gpqa_dev307_ci_lo_clears_bar"]
    run.summary["paired_mean_delta_dev307_minus_0p22"] = d["paired_mean_delta_dev307_minus_0p22"]
    run.summary["paired_delta_ci_significant"] = d["paired_delta_ci_significant"]
    run.summary["mcnemar_p_two_sided"] = d["mcnemar_p_two_sided"]
    run.summary["de_confounded_acc"] = d["de_confounded_acc"]
    run.summary["implied_base_for_0p22_clear"] = d["implied_base_for_0p22_clear"]
    run.summary["implied_base_for_dev307_clear"] = d["implied_base_for_dev307_clear"]
    run.summary["surface_to_human"] = True
    run.summary["official_tps"] = 0
    run.summary["analysis_only"] = True
    rid = run.id
    run.finish()
    print(f"[wandb] logged dev307 GPQA A/B run id={rid} VERDICT={d['VERDICT']}")
    print(f"[wandb] group={GROUP} project={ENTITY}/{PROJECT}")
    print(f"WANDB_RUN_ID={rid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
