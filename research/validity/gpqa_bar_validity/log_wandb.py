#!/usr/bin/env python3
"""PR #614 -- log the GPQA bar-validity audit to W&B (group gpqa-bar-validity).

Logs the bars_verdict.py summary as config + summary metrics plus a per-seed table,
so the truncation/regime audit leaves a rich, queryable record. analysis_only run.
"""
from __future__ import annotations

import argparse
import json
import os

import wandb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="bars_verdict.py output json")
    ap.add_argument("--name", default="ubel/gpqa-bar-validity")
    ap.add_argument("--group", default="gpqa-bar-validity")
    ap.add_argument("--entity", default="wandb-applied-ai-team")
    ap.add_argument("--project", default="gemma-challenge-senpai")
    args = ap.parse_args()

    s = json.load(open(args.summary))

    run = wandb.init(
        entity=args.entity, project=args.project,
        name=args.name, group=args.group,
        job_type="quality-bar-audit",
        config={
            "pr": 614,
            "analysis_only": True,
            "official_tps": 0,
            "engine": s.get("engine"),
            "model": s.get("model"),
            "n_questions": s.get("n_questions"),
            "max_tokens_primary": s.get("max_tokens_primary"),
            "max_model_len": s.get("max_model_len"),
            "decode_protocol_sampled": s.get("decode_protocol_sampled"),
            "sampled_n_seeds": s.get("sampled_n_seeds"),
            "prior_bar_581": s.get("prior_bar_581"),
            "prior_sampled_base_581": s.get("prior_sampled_base_581"),
            "prior_greedy_anchor_581": s.get("prior_greedy_anchor_581"),
        },
    )

    # scalar summary metrics (everything numeric / bool / str that isn't a list/dict)
    skip = {"sampled_per_seed_accuracy", "decode_protocol_sampled"}
    for k, v in s.items():
        if k in skip:
            continue
        if isinstance(v, (int, float, bool, str)) or v is None:
            run.summary[k] = v

    # per-seed sampled accuracy table
    per_seed = s.get("sampled_per_seed_accuracy") or []
    if per_seed:
        tbl = wandb.Table(columns=["seed_idx", "accuracy"])
        for i, a in enumerate(per_seed):
            tbl.add_data(i, a)
        run.log({"sampled_per_seed_accuracy": tbl})

    print(f"[wandb] logged run id={run.id} name={args.name} group={args.group}")
    print(f"[wandb] url={run.url}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
