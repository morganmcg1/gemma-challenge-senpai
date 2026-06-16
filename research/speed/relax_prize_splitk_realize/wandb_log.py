#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Standalone W&B logger for the relax-prize arms (PR #452, stark).

The GPU/vLLM microbench runs on the senpai tool-venv (vllm dev build #450 pinned),
which has NO usable wandb (empty PEP-420 namespace shadow). This logger reads a
harness JSON and logs it to W&B from a python that HAS wandb (the repo .venv). Pure
json+wandb -- no torch/vllm import -- so it is venv-agnostic. One run per JSON.

Reproduce: cd target/ && .venv/bin/python \
  research/speed/relax_prize_splitk_realize/wandb_log.py \
  --json research/speed/relax_prize_splitk_realize/relax_prize_splitk_realize.json \
  --wandb_group relax-equivalence-prize --wandb_name stark/relax-prize-splitk-realize
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
    ap.add_argument("--wandb_group", default="relax-equivalence-prize")
    ap.add_argument("--wandb_name", default="stark/relax-prize-splitk-realize")
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

    # reduction-order bit-exactness table (if present)
    bits = payload.get("bit_exactness")
    if isinstance(bits, dict):
        t = wandb.Table(columns=["component", "N", "K", "max_abs_delta", "byteexact"])
        for c, d in bits.items():
            t.add_data(c, d.get("N"), d.get("K"), d.get("max_abs_delta"), d.get("byteexact"))
        run.log({"reduction_byteexact": t})

    # Triton split-K confirmation table (if present)
    tri = payload.get("triton_splitk")
    if isinstance(tri, dict) and isinstance(tri.get("per_shape"), dict):
        t = wandb.Table(columns=["component", "marlin_us", "triton_best_us",
                                 "triton_best_pct_hbm", "triton_vs_marlin_pct", "triton_faster"])
        for c, d in tri["per_shape"].items():
            t.add_data(c, d.get("marlin_us"), d.get("triton_best_us"),
                       d.get("triton_best_pct_hbm"), d.get("triton_vs_marlin_pct"),
                       d.get("triton_faster"))
        run.log({"triton_splitk_confirm": t})

    # identity arm table (if present)
    ident = payload.get("identity")
    if isinstance(ident, dict) and isinstance(ident.get("per_prompt"), list):
        t = wandb.Table(columns=["idx", "n_tokens", "n_flips", "identical",
                                 "strict_first_flip_pos"])
        for r in ident["per_prompt"]:
            t.add_data(r.get("idx"), r.get("n_tokens"), r.get("n_flips"),
                       r.get("identical"), r.get("first_flip_pos"))
        run.log({"identity_per_prompt": t})

    print(f"[wandb_log] logged {args.json} -> {run.url}  id={run.id}", flush=True)
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
