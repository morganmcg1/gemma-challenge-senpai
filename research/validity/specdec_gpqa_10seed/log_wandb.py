#!/usr/bin/env python
"""PR #656 — log the AR-vs-spec GPQA-D 10-seed contrast to W&B.

Two runs in group `specdec-gpqa-10seed-wirbel` (so SENPAI-RESULT cites both ids):
  - wirbel/specdec-gpqa-10seed-ar   : AR (M=1, drafter OFF) per-seed + pooled
  - wirbel/specdec-gpqa-10seed-spec : SPEC (MTP K=6 shipped) per-seed + pooled +
                                      the SPEC-AR contrast (delta is primary_metric)

LOCAL ONLY, analysis_only=True, official_tps=0. dev307+spec is not #319 byte-
identical and has served-nondeterminism -> multi-seed mean is the valid estimand.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent
GROUP = "specdec-gpqa-10seed-wirbel"
ENTITY = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
PROJECT = os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai")

COMMON = {
    "model_id": "/workspace/gemma_build/int4_g128_lmhead",
    "body_head": "int4_g128_lmhead (g128 int4 body + untied int4 g128 lm_head)",
    "engine": "dev307 (0.22.1rc1.dev307+g3e8afdf78)",
    "vllm_batch_invariant": 1,
    "max_model_len": 8192,
    "max_num_seqs": 16,
    "max_tokens": 6144,
    "min_tokens": 8,
    "task": "gpqa_diamond",
    "gpqa_sampling": "T=1.0, top_p=0.95, top_k=64",
    "n_seeds": 10,
    "n_per_seed": 198,
    "n_pooled": 1980,
    "base_gpqa_sampled": 0.5404,
    "bar_90pct": 0.4864,
    "analysis_only": True,
    "official_tps": 0,
}


def add_seed_table(arm):
    tbl = wandb.Table(columns=["seed", "accuracy", "n_correct", "n_scored", "n_error", "n_length"])
    for r in arm["per_seed"]:
        tbl.add_data(str(r["seed"]), r["accuracy"], r["n_correct"], r["n_scored"],
                     r["n_error"], r["n_length"])
    return tbl


def log_arm(name, arm, kind, contrast=None):
    cfg = {**COMMON, "arm": kind}
    if kind == "ar":
        cfg["num_speculative_tokens"] = 0
        cfg["drafter"] = "OFF (speculative_config=None, M=1)"
        cfg["is_319_identical"] = "n/a (sampled, M=1)"
    else:
        cfg["num_speculative_tokens"] = 6
        cfg["drafter"] = "/tmp/qat-assistant (MTP, shipped int4_mtp_batchinv manifest)"
        cfg["is_319_identical"] = False
    run = wandb.init(project=PROJECT, entity=ENTITY, name=name, group=GROUP,
                     job_type="gpqa-ar-vs-spec", reinit=True, config=cfg)
    log = {
        f"gpqa/{kind}_pooled_acc": arm["p"],
        f"gpqa/{kind}_n_correct": arm["c"],
        f"gpqa/{kind}_n_scored": arm["n"],
        f"gpqa/{kind}_ci_lo_wilson": arm["lo"],
        f"gpqa/{kind}_ci_hi_wilson": arm["hi"],
        f"gpqa/{kind}_perseed_mean": arm["mean"],
        f"gpqa/{kind}_perseed_sd": arm["sd"],
        f"gpqa/{kind}_perseed_min": arm["min"],
        f"gpqa/{kind}_perseed_max": arm["max"],
        f"gpqa/{kind}_pct_of_base": arm["p"] / COMMON["base_gpqa_sampled"] * 100,
        f"gpqa/{kind}_pass_bar": int(arm["p"] >= COMMON["bar_90pct"]),
        f"gpqa/{kind}_per_seed": add_seed_table(arm),
    }
    if contrast:
        ct = contrast
        log.update({
            "contrast/spec_minus_ar_delta": ct["delta_spec_minus_ar"],
            "contrast/z": ct["z"], "contrast/p_unpaired": ct["p"], "contrast/se": ct["se"],
            "contrast/mcnemar_b_AR_correct_SPEC_wrong": ct["mcnemar"]["b_AR_correct_SPEC_wrong"],
            "contrast/mcnemar_c_AR_wrong_SPEC_correct": ct["mcnemar"]["c_AR_wrong_SPEC_correct"],
            "contrast/mcnemar_n_discordant": ct["mcnemar"]["n_discordant"],
            "contrast/mcnemar_p": ct["mcnemar"]["mcnemar_p"],
            "contrast/net_total_b_minus_c": ct["net_total_b_minus_c"],
            "contrast/n_questions_nonzero_net": ct["n_questions_nonzero_net"],
            "contrast/top5_share_abs_flip": ct["top5_share_abs_flip"],
        })
    wandb.log(log)
    run.summary[f"{kind}_pooled_acc"] = arm["p"]
    run.summary[f"{kind}_pct_of_base"] = arm["p"] / COMMON["base_gpqa_sampled"] * 100
    run.summary[f"{kind}_pass_bar"] = int(arm["p"] >= COMMON["bar_90pct"])
    run.summary["analysis_only"] = True
    run.summary["official_tps"] = 0
    if contrast:
        run.summary["gpqa_sampled_spec_minus_ar_delta"] = contrast["delta_spec_minus_ar"]
        run.summary["primary_metric"] = contrast["delta_spec_minus_ar"]
        run.summary["VERDICT"] = contrast.get("verdict", "?")
        run.summary["surface_to_human"] = True
    rid = run.id
    run.finish()
    print(f"WANDB_RUN_ID_{kind.upper()}={rid}")
    return rid


def main():
    d = json.loads((HERE / "contrast.json").read_text())
    ids = {}
    if "ar" in d:
        ids["ar"] = log_arm("wirbel/specdec-gpqa-10seed-ar", d["ar"], "ar")
    if "spec" in d:
        contrast = dict(d.get("contrast", {}))
        contrast["verdict"] = d.get("verdict", "?")
        ids["spec"] = log_arm("wirbel/specdec-gpqa-10seed-spec", d["spec"], "spec",
                              contrast=contrast if "contrast" in d else None)
    print(f"[wandb] group={GROUP} project={ENTITY}/{PROJECT}")
    print(f"[wandb] run_ids={ids}")
    (HERE / "wandb_run_ids.json").write_text(json.dumps(ids, indent=2))


if __name__ == "__main__":
    main()
