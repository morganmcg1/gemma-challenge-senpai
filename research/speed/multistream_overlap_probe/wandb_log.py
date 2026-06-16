#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Standalone W&B logger for the multi-stream overlap probe (PR #477, lawine).

The GPU tool-venv that runs the vLLM Triton unified_attention + Marlin microbench has NO
usable wandb (empty PEP-420 namespace shadow). This logger reads the harness JSON and logs
it to W&B from a python that HAS wandb (the repo .venv). Pure json+wandb -- no torch/vllm
import -- so it is venv-agnostic. One run per JSON.

Reproduce: cd target/ && .venv/bin/python \
  research/speed/multistream_overlap_probe/wandb_log.py \
  --json research/speed/multistream_overlap_probe/multistream_overlap_probe.json \
  --wandb_group equivalence-escalation-anchors --wandb_name lawine/multistream-overlap-probe
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
    ap.add_argument("--wandb_group", default="equivalence-escalation-anchors")
    ap.add_argument("--wandb_name", default="lawine/multistream-overlap-probe")
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

    # per-L two-stream overlap sweep
    per_L = payload.get("per_L")
    if isinstance(per_L, dict):
        t = wandb.Table(columns=["L", "body_us", "iso_strict_us", "iso_perm_us", "two_strict_us",
                                 "two_perm_us", "two_body_us", "exposed_strict_us", "exposed_perm_us",
                                 "multistream_strict_added_us", "overlap_fraction_strict",
                                 "symmetric_overlap_speedup", "gemm_dram_bw_util",
                                 "attn_bw_frac_of_peak", "multistream_hideable_us",
                                 "multistream_strict_tps_ceiling"])
        for L, d in sorted(per_L.items(), key=lambda kv: int(kv[0])):
            if "error" in d:
                continue
            m = d.get("median_us", {})
            t.add_data(int(L), m.get("body"), m.get("iso_strict"), m.get("iso_perm"),
                       m.get("two_strict"), m.get("two_perm"), m.get("two_body"),
                       d.get("exposed_strict_us"), d.get("exposed_perm_us"),
                       d.get("multistream_strict_added_us"), d.get("overlap_fraction_strict"),
                       d.get("symmetric_overlap_speedup"), d.get("gemm_dram_bw_util"),
                       d.get("attn_bw_frac_of_peak"), d.get("multistream_hideable_us"),
                       d.get("multistream_strict_tps_ceiling"))
        run.log({"per_L_overlap_sweep": t})

    # strict byte-identity vs the per-row M=1 sequential canonical
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
