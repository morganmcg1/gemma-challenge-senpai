#!/usr/bin/env python
"""Log the PR #620 matched-arm spec-vs-AR distribution-preservation result to W&B.

Group `spec-distribution-preservation-matched-arm`. ANALYSIS-ONLY, official_tps=0.
One run (job_type=matched-arm-verdict) carrying the paired GPQA-D + GSM8K evidence:
per-arm acc + cluster-bootstrap CI, paired delta + CI, McNemar, truncation, verdict.
Headline metrics are ALSO mirrored under a `summary/` prefix (project convention).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent
GROUP = "spec-distribution-preservation-matched-arm"
ENTITY = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
PROJECT = os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai")

# Cross-run anchors from the PR body (DIFFERENT runs/seeds/substrate-states — the
# whole point of the matched arm is to isolate spec-degradation from this noise).
ANCHORS = {
    "stark_605_spec_gpqa_abs": 0.4141,   # 3 seeds, absolute spec GPQA-D
    "kanna_610_ar_gpqa_abs": 0.4414,     # 10 seeds, absolute AR int4 GPQA-D
    "denken_609_ngram_gsm8k_delta": 0.006,  # ngram matched-arm GSM8K (distribution-preserving)
}

CONFIG = {
    "design": "matched-arm paired (spec=int4+MTP-K7 ON vs ar=int4 spec OFF, SAME body)",
    "model_id": "/workspace/gemma_build/int4_g128_lmhead",
    "drafter": "/tmp/qat-assistant",
    "num_speculative_tokens": 7,
    "vllm_batch_invariant": 1,
    "vllm_version": "0.22.0",
    "max_model_len": 6144,
    "min_tokens": 8,
    "gpqa_max_tokens": 4096,
    "gsm8k_max_tokens": 512,
    "sampling": "T=1.0 top_p=0.95 top_k=64 (lewtun #31)",
    "gpqa_seeds": [12345, 23456, 34567, 45678, 56789],
    "gsm8k_sampling_seeds": [1234, 2345, 3456, 4567, 5678],
    "primary_eval": "gpqa_diamond",
    "analysis_only": True,
    "official_tps": 0,
    "is_319_greedy_identical": False,
    "claim_axis": "SAMPLED downstream distribution (NOT the greedy #319 byte-exact gate)",
}


def _flat_eval(prefix: str, e: dict) -> dict:
    cb = e.get("cluster_bootstrap", {})
    mc = e.get("mcnemar_seeditem", {})
    cl = e.get("cluster_level_paired_diff", {})
    tr = e.get("truncation", {})
    out = {
        f"{prefix}/n_paired_units": e.get("n_paired_units"),
        f"{prefix}/n_clusters": cb.get("n_clusters"),
        f"{prefix}/prompt_sha_mismatch": e.get("n_prompt_sha_mismatch"),
        f"{prefix}/spec_acc": cb.get("spec_acc"),
        f"{prefix}/ar_acc": cb.get("ar_acc"),
        f"{prefix}/paired_delta": cb.get("delta"),
        f"{prefix}/paired_delta_ci_lo": (cb.get("delta_ci95") or [None, None])[0],
        f"{prefix}/paired_delta_ci_hi": (cb.get("delta_ci95") or [None, None])[1],
        f"{prefix}/paired_delta_se_boot": cb.get("delta_se_boot"),
        f"{prefix}/mcnemar_b_spec_wins": mc.get("b"),
        f"{prefix}/mcnemar_c_ar_wins": mc.get("c"),
        f"{prefix}/mcnemar_p_exact": mc.get("p_exact"),
        f"{prefix}/cluster_paired_diff_mean": cl.get("mean_paired_diff"),
        f"{prefix}/spec_trunc_length_rate": (tr.get("spec") or {}).get("finish_length_rate"),
        f"{prefix}/ar_trunc_length_rate": (tr.get("ar") or {}).get("finish_length_rate"),
        f"{prefix}/verdict": e.get("verdict"),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-json", type=Path, default=HERE / "results" / "analysis.json")
    ap.add_argument("--name", default="denken/spec-distribution-preservation-matched-arm")
    args = ap.parse_args()
    data = json.loads(args.analysis_json.read_text())

    run = wandb.init(project=PROJECT, entity=ENTITY, name=args.name, group=GROUP,
                     job_type="matched-arm-verdict", reinit=True,
                     config={**CONFIG, "anchors": ANCHORS})

    gp = _flat_eval("gpqa_diamond", data.get("gpqa_diamond", {}))
    gs = _flat_eval("gsm8k", data.get("gsm8k", {}))
    headline = data.get("headline_verdict")

    metrics = {**{k: v for k, v in gp.items() if not isinstance(v, str)},
               **{k: v for k, v in gs.items() if not isinstance(v, str)}}
    metrics.update({
        "anchor/stark605_spec_gpqa_abs": ANCHORS["stark_605_spec_gpqa_abs"],
        "anchor/kanna610_ar_gpqa_abs": ANCHORS["kanna_610_ar_gpqa_abs"],
    })
    # summary/ mirror of the headline numbers (project convention).
    gpd = data.get("gpqa_diamond", {}).get("cluster_bootstrap", {})
    gsd = data.get("gsm8k", {}).get("cluster_bootstrap", {})
    metrics.update({
        "summary/analysis_only": 1,
        "summary/official_tps": 0,
        "summary/headline_verdict_preserving": int(headline == "SPEC_DISTRIBUTION_PRESERVING"),
        "summary/gpqa_spec_acc": gpd.get("spec_acc"),
        "summary/gpqa_ar_acc": gpd.get("ar_acc"),
        "summary/gpqa_paired_delta": gpd.get("delta"),
        "summary/gsm8k_spec_acc": gsd.get("spec_acc"),
        "summary/gsm8k_ar_acc": gsd.get("ar_acc"),
        "summary/gsm8k_paired_delta": gsd.get("delta"),
    })
    wandb.log(metrics)

    run.summary["headline_verdict"] = headline
    run.summary["gpqa_verdict"] = data.get("gpqa_diamond", {}).get("verdict")
    run.summary["gsm8k_verdict"] = data.get("gsm8k", {}).get("verdict")
    run.summary["gpqa_paired_delta"] = gpd.get("delta")
    run.summary["gpqa_paired_delta_ci95"] = gpd.get("delta_ci95")
    run.summary["gsm8k_paired_delta"] = gsd.get("delta")
    run.summary["analysis_only"] = True
    run.summary["official_tps"] = 0
    print(f"[wandb] logged run {run.id} ({run.name}) group={GROUP}")
    print(f"[wandb] headline_verdict={headline}")
    run.finish()
    print(f"WANDB_RUN_ID={run.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
