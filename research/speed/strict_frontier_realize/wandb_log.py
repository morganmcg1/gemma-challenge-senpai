#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Standalone W&B logger for the strict-frontier-realize card (PR #466, stark).

The GPU tool-venv that runs the vLLM Triton unified_attention microbench has NO
usable wandb (empty PEP-420 namespace shadow). This logger reads the harness JSON
and logs it to W&B from a python that HAS wandb (the repo .venv). Pure json+wandb --
no torch/vllm import -- so it is venv-agnostic. One run per JSON.

Reproduce: cd target/ && .venv/bin/python \
  research/speed/strict_frontier_realize/wandb_log.py \
  --json research/speed/strict_frontier_realize/strict_frontier_realize.json \
  --wandb_group strict-frontier-realize --wandb_name stark/strict-frontier-realize
"""
from __future__ import annotations

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="strict-frontier-realize")
    ap.add_argument("--wandb_name", default="stark/strict-frontier-realize")
    ap.add_argument("--job_type", default="profiling")
    args = ap.parse_args()

    with open(args.json) as fh:
        payload = json.load(fh)
    import wandb

    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type=args.job_type, config=payload.get("config", {}))

    verdict = payload.get("verdict", {})
    run.summary.update({k: v for k, v in verdict.items() if isinstance(v, (int, float, bool, str))})

    # per-L sweep: permissive 3D vs strict 2D serial vs strict 3D num_par=1, per cycle.
    per_L = payload.get("per_L")
    if isinstance(per_L, dict):
        t = wandb.Table(columns=["L", "perm_3d_us", "strict_2d_us", "strict_3d_ns1_us",
                                 "added_us_2d", "added_us_sigma_2d", "realized_strict_2d_tps",
                                 "realized_strict_3d_ns1_tps", "strict_2d_captured"])
        for L, d in sorted(per_L.items(), key=lambda kv: int(kv[0])):
            m = d.get("median_us", {})
            t.add_data(int(L), m.get("permissive_3d"), m.get("strict_2d"), m.get("strict_3d_ns1"),
                       d.get("added_us_strict_2d"), d.get("added_us_sigma_2d"),
                       d.get("realized_strict_2d_tps"), d.get("realized_strict_3d_ns1_tps"),
                       d.get("captured", {}).get("strict_2d"))
        run.log({"per_L_sweep": t})

    # identity vs the per-row M=1 sequential canonical (is the strict variant truly strict?).
    ident = payload.get("identity")
    if isinstance(ident, dict) and isinstance(ident.get("per_arm"), dict):
        t = wandb.Table(columns=["arm", "byte_identity_min", "byte_identity_mean",
                                 "argmax_identity_min", "argmax_identity_mean", "max_abs_diff"])
        for arm, d in ident["per_arm"].items():
            t.add_data(arm, d.get("byte_identity_min"), d.get("byte_identity_mean"),
                       d.get("argmax_identity_min"), d.get("argmax_identity_mean"),
                       d.get("max_abs_diff"))
        run.log({"identity_vs_canonical": t})

    print(f"[wandb_log] logged {args.json} -> {run.url}  id={run.id}", flush=True)
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
