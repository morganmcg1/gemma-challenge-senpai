#!/usr/bin/env python
"""Log per-arm batch-invariant greedy-identity results to W&B.

One run per arm (group ``int4-mtp-batchinv``). Reads the FLIPRATE_JSON line that
flip_rate.py wrote to ``<outdir>/<arm>_fliprate.txt`` and logs the verdict +
per-token flip rate (censored-geometric MLE) + per-prompt stats, tagged with the
arm's VLLM_BATCH_INVARIANT state so the ON vs OFF arms are directly comparable.

The decisive PR-#19 question: does enabling batch-invariant kernels drive the
int4+MTP spec-decode flip rate to 0 (GREEDY_IDENTICAL)? OFF is the control
(reproduce PR #5's ~0.33%/tok); ON is the target.

Local AWS A10G greedy-identity diagnostic only -- NOT an official a10g-small run.

Usage:
  log_arm_wandb.py --outdir /tmp/arms_bi \
    --arm int4_off:0 --arm int4_on:1
Each --arm is  label:batch_invariant  (batch_invariant in {0,1}).
"""
from __future__ import annotations

import argparse
import json
import os


def load_fliprate(path: str) -> dict | None:
    try:
        with open(path) as fh:
            for line in fh:
                if line.startswith("FLIPRATE_JSON "):
                    return json.loads(line[len("FLIPRATE_JSON "):])
    except FileNotFoundError:
        return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/tmp/arms_bi")
    ap.add_argument("--arm", action="append", default=[],
                    help="label:batch_invariant(0/1)")
    ap.add_argument("--group", default="int4-mtp-batchinv")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--num-prompts", type=int, default=32)
    ap.add_argument("--target-model-id",
                    default="google/gemma-4-E4B-it-qat-w4a16-ct")
    ap.add_argument("--drafter",
                    default="google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant")
    args = ap.parse_args()

    try:
        import wandb
    except ModuleNotFoundError:
        print("wandb not installed; skipping")
        return

    run_ids = []
    for spec in args.arm:
        label, inv = (spec.split(":", 1) + ["1"])[:2]
        batch_invariant = bool(int(inv))
        # Per-arm target precision/model derived from the label prefix so the
        # bf16 positive-control arm (all-aten, no Marlin) is not mislabeled int4.
        if label.startswith("bf16"):
            precision, target_model_id, covers_target_gemm = (
                "bf16", "google/gemma-4-E4B-it", True)
        elif label.startswith("fp8"):
            precision, target_model_id, covers_target_gemm = (
                "fp8", "google/gemma-4-E4B-it", False)
        else:
            precision, target_model_id, covers_target_gemm = (
                "int4-w4a16", args.target_model_id, False)
        fr = load_fliprate(os.path.join(args.outdir, f"{label}_fliprate.txt"))
        if fr is None:
            print(f"[{label}] no fliprate json found; skipping")
            continue
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "senpai-v1"),
            entity=os.environ.get("WANDB_ENTITY") or None,
            group=args.group,
            name=f"kanna/batchinv-{label}",
            job_type="greedy-identity-batchinv",
            reinit=True,
            config={
                "arm": label,
                "batch_invariant": batch_invariant,
                "vllm_batch_invariant_env": inv,
                "target_precision": precision,
                "target_gemm_aten_covered": covers_target_gemm,
                "target_model_id": target_model_id,
                "drafter": args.drafter,
                "num_speculative_tokens": args.k,
                "engine": "vllm==0.22.0",
                "enforce_eager": True,
                "num_prompts": args.num_prompts,
                "output_len": 512,
                "seed": 1,
                "ignore_eos": True,
                "gpu": "A10G (local diagnostic)",
                "spec_method": "mtp",
                "attn_backend": "TRITON_ATTN",
                "attn_group_patch": "num_heads {8,4} backport (PR #5)",
            },
        )
        verdict = fr.get("verdict")
        log = {
            "greedy_identical": int(verdict == "GREEDY_IDENTICAL"),
            "flip_rate_per_token": fr.get("flip_rate_per_token"),
            "flip_rate_ci95_lo": (fr.get("flip_rate_ci95") or [None, None])[0],
            "flip_rate_ci95_hi": (fr.get("flip_rate_ci95") or [None, None])[1],
            "flip_events": fr.get("flip_events"),
            "geom_trials": fr.get("geom_trials"),
            "prompts_identical": fr.get("identical"),
            "prompts_divergent": fr.get("divergent"),
            "prompts_total": fr.get("prompts"),
            "mean_first_divergence_index": fr.get("mean_first_divergence_index"),
            "raw_cascade_divergent_fraction": fr.get("raw_cascade_divergent_fraction"),
            "total_tokens_compared": fr.get("total_tokens_compared"),
        }
        wandb.log(log)
        run.summary.update(log)
        run.summary["verdict"] = verdict
        print(f"[{label}] batch_invariant={batch_invariant} {verdict} "
              f"flip_rate={fr.get('flip_rate_per_token')} -> {run.url}  id={run.id}")
        run_ids.append(run.id)
        run.finish()

    print("WANDB_RUN_IDS " + json.dumps(run_ids))


if __name__ == "__main__":
    main()
