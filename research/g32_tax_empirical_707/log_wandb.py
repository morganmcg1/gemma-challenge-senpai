#!/usr/bin/env python
"""Standalone W&B logger for PR #707 g32-tax-empirical.

Run as a FILE (not -m) with a wandb-capable python so sys.path[0] is THIS dir
(no repo-root wandb/ data-dir shadow), e.g.:
  /workspace/senpai/target/.venv/bin/python \
    research/g32_tax_empirical_707/log_wandb.py \
    --results research/g32_tax_empirical_707/g32_tax_opbench_results.json
"""
from __future__ import annotations

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="g32-tax-empirical-land")
    ap.add_argument("--wandb_name", default="land/g32-tax-empirical")
    args = ap.parse_args()

    payload = json.load(open(args.results))
    h = payload["headline"]

    import wandb
    assert hasattr(wandb, "init"), f"wandb shadowed: {wandb.__file__}"
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type="g32-tax-empirical", config=payload.get("config", {}))
    flat = {k: v for k, v in h.items() if isinstance(v, (int, float, bool))}
    for c, cv in payload["per_component"].items():
        flat[f"comp_{c}_tax_frac"] = cv["tax_frac"]
        flat[f"comp_{c}_delta_us"] = cv["delta_us"]
        flat[f"comp_{c}_us_g128"] = cv["us_g128"]
        flat[f"comp_{c}_us_g32"] = cv["us_g32"]
    # byte-model totals for the record
    bm = payload["byte_model_total"]
    for k in ("weight_bytes", "scale_bytes_g128", "scale_bytes_g32",
              "ws_bytes_g128", "ws_bytes_g32", "byte_tax_frac_ws"):
        flat[f"bytemodel_{k}"] = bm[k]
    wandb.log(flat)
    wandb.summary.update(flat)
    wandb.summary["verdict"] = h["verdict"]
    wandb.summary["note"] = payload["note"]
    rid = run.id
    wandb.finish()
    print(f"[wandb] run_id={rid}")
    # persist the run id back into the results json
    payload["wandb_run_id"] = rid
    json.dump(payload, open(args.results, "w"), indent=2)


if __name__ == "__main__":
    main()
