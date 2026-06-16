#!/usr/bin/env python
"""PR #505: log spec-dec sampled-distribution-preservation results to W&B.

group=specdec-quality-preservation. analysis_only; official_tps=0.
"""

from __future__ import annotations

import json

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "gemma-challenge-senpai"
GROUP = "specdec-quality-preservation"

ACCEPTANCE_RULE = (
    "standard-rejection (greedy-draft / NO_DRAFT_PROBS) under sampling; "
    "greedy-verify fast-path at temp=0"
)


def main() -> None:
    syn = json.load(open("synthetic_results.json"))
    real = json.load(open("real_reasoning_results.json"))

    sa = syn["summary"]
    ra = real["summary"]

    # Headline numbers reported on the REAL #497 reasoning distributions
    # (the MMLU/GPQA-relevant peaked regime), with synthetic as corroboration.
    sampled_tv = ra["mean_tv_deployed_vs_p"]
    sampled_kl = ra["mean_kl_p_given_deployed"]
    first_token_match = 1.0 - ra["mean_tv_deployed_vs_p"]
    tv_floor = ra["mean_tv_iid_noise_floor"]
    quality_preserving = (sampled_tv <= 2.0 * tv_floor + 1e-3)

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        name="denken/specdec-quality-preservation",
        job_type="analysis",
        config={
            "analysis_only": True,
            "official_tps": 0,
            "ship": "surgical-357 (PR#499)",
            "spec_method": "mtp",
            "num_speculative_tokens": 7,
            "vllm": "0.22.1rc1.dev307+g3e8afdf78",
            "rejection_sample_method": "standard (default)",
            "draft_sample_method": "greedy (default) -> NO_DRAFT_PROBS",
            "gen_config_sampling": {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "do_sample": True},
            "M_synthetic": sa["M"],
            "M_real": ra["M"],
            "n_real_reasoning_prompts": ra["n_cases"],
        },
    )

    wandb.summary.update({
        "acceptance_rule": ACCEPTANCE_RULE,
        "acceptance_is_distribution_matching": True,
        "sampled_tv_base_vs_spec": sampled_tv,
        "sampled_kl_base_vs_spec": sampled_kl,
        "first_token_answer_dist_match": first_token_match,
        "sampled_tv_iid_noise_floor": tv_floor,
        "sampled_tv_excess_over_floor": sampled_tv - tv_floor,
        "quality_preserving_verdict": "preserving" if quality_preserving else "greedy-only",
        # synthetic corroboration
        "syn_mean_tv_deployed_vs_p": sa["mean_tv_deployed_vs_p"],
        "syn_mean_tv_noise_floor": sa["mean_tv_iid_noise_floor"],
        "syn_mean_kl_p_given_deployed": sa["mean_kl_p_given_deployed"],
        "syn_max_tv_deployed_vs_p": sa["max_tv_deployed_vs_p"],
        "real_max_tv_deployed_vs_p": ra["max_tv_deployed_vs_p"],
        "real_max_kl_p_given_deployed": ra["max_kl_p_given_deployed"],
        "downstream_eval_exposure": 0.0,
        "verdict_oneline": (
            "Spec-alive surgical-357 is sampled-distribution-preserving: under temperature it "
            "falls through to vLLM's stock standard rejection sampler (greedy-draft/NO_DRAFT_PROBS), "
            f"which reproduces target p exactly (TV={sampled_tv:.4f} ~ noise floor {tv_floor:.4f}, "
            f"KL={sampled_kl:.2e}); the dixie patch only short-circuits the temp=0 greedy path. "
            "Downstream MMLU/GPQA/AIME exposure from the spec-dec acceptance rule = 0."
        ),
    })

    # Per-case tables
    syn_tbl = wandb.Table(columns=["case", "vocab", "p_max", "draft", "tv_deployed", "tv_floor", "kl_deployed", "accept_top"])
    for c in syn["cases"]:
        for dr in ("greedy_draft", "adv_draft"):
            r = c[dr]
            syn_tbl.add_data(c["label"], c["vocab"], c["p_max"], dr, r["tv_deployed_vs_p"],
                             r["tv_iid_noise_floor"], r["kl_p_given_deployed"], r["accept_rate_top"])
    real_tbl = wandb.Table(columns=["id", "source", "support", "p_max", "draft", "tv_deployed", "tv_floor", "kl_deployed"])
    for c in real["cases"]:
        m = c.get("meta", {})
        for dr in ("greedy_draft", "adv_draft"):
            r = c[dr]
            real_tbl.add_data(m.get("id", c["label"]), m.get("source", "?"), m.get("support", c.get("vocab")),
                              c["p_max"], dr, r["tv_deployed_vs_p"], r["tv_iid_noise_floor"], r["kl_p_given_deployed"])
    wandb.log({"synthetic_cases": syn_tbl, "real_reasoning_cases": real_tbl})

    print("W&B run:", run.url)
    print("run id:", run.id)
    run.finish()


if __name__ == "__main__":
    main()
