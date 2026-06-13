#!/usr/bin/env python
"""Log per-arm greedy-identity precision-localization results to W&B.

One run per arm (group ``int4-mtp-greedy-precision``). Reads the FLIPRATE_JSON
line that flip_rate.py wrote to ``<outdir>/<arm>_fliprate.txt`` and logs the
verdict + per-token flip rate + per-prompt stats, tagged with the arm's target
precision so the three arms are directly comparable in the W&B table.

Local AWS A10G greedy-identity diagnostic only -- NOT an official a10g-small run.

Usage:
  log_greedy_arm_wandb.py --outdir /tmp/arms \
    --arm int4:google/gemma-4-E4B-it-qat-w4a16-ct: \
    --arm bf16:google/gemma-4-E4B-it: \
    --arm fp8:google/gemma-4-E4B-it:fp8
Each --arm is  label:target_model_id:quant  (quant empty for native dtype).
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
    ap.add_argument("--outdir", default="/tmp/arms")
    ap.add_argument("--arm", action="append", default=[],
                    help="label:target_model_id:quant")
    ap.add_argument("--group", default="int4-mtp-greedy-precision")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--num-prompts", type=int, default=32)
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
        label, model_id, quant = (spec.split(":", 2) + ["", "", ""])[:3]
        fr = load_fliprate(os.path.join(args.outdir, f"{label}_fliprate.txt"))
        if fr is None:
            print(f"[{label}] no fliprate json found; skipping")
            continue
        if quant:
            precision = quant
        elif "qat-w4a16" in model_id:
            precision = "int4-w4a16"
        else:
            precision = "bf16"
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "senpai-v1"),
            entity=os.environ.get("WANDB_ENTITY") or None,
            group=args.group,
            name=f"kanna/greedy-{label}",
            job_type="greedy-identity-precision",
            reinit=True,
            config={
                "arm": label,
                "target_precision": precision,
                "target_model_id": model_id,
                "target_quantization": quant or "none",
                "drafter": args.drafter,
                "num_speculative_tokens": args.k,
                "engine": "vllm==0.22.0",
                "enforce_eager": True,
                "num_prompts": args.num_prompts,
                "output_len": 512,
                "seed": 1,
                "gpu": "A10G (local diagnostic)",
                "spec_method": "mtp",
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
        print(f"[{label}] {verdict} flip_rate={fr.get('flip_rate_per_token')} "
              f"-> {run.url}  id={run.id}")
        run_ids.append(run.id)
        run.finish()

    print("WANDB_RUN_IDS " + json.dumps(run_ids))


if __name__ == "__main__":
    main()
