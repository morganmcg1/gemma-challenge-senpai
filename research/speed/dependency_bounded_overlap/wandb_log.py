#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Standalone W&B logger for the dependency-bounded multi-stream overlap probe (PR #482, lawine).

The GPU tool-venv that runs the vLLM Triton unified_attention + Marlin microbench has NO usable
wandb (empty PEP-420 namespace shadow). This logger reads the harness JSON and logs it to W&B from
a python that HAS wandb (the repo .venv). Pure json+wandb -- no torch/vllm import -- so it is
venv-agnostic. One run per JSON.

Reproduce: cd target/ && .venv/bin/python \
  research/speed/dependency_bounded_overlap/wandb_log.py \
  --json research/speed/dependency_bounded_overlap/dependency_bounded_overlap.json \
  --wandb_group equivalence-escalation-anchors --wandb_name lawine/dependency-bounded-overlap
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
    ap.add_argument("--wandb_name", default="lawine/dependency-bounded-overlap")
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

    # nested self-test conditions -> flat scalar summary (selftest/<name>)
    st = verdict.get("self_test_conditions")
    if isinstance(st, dict):
        run.summary.update({f"selftest/{k}": bool(v) for k, v in st.items()})

    # peak BW / barrier micro-bench scalars
    for blk in ("peak_bw", "barrier"):
        d = payload.get(blk)
        if isinstance(d, dict):
            run.summary.update({f"{blk}/{k}": v for k, v in d.items()
                                if isinstance(v, (int, float, bool, str))})

    # per-L arm sweep: every captured arm wall + the three paired taxes + realizable/ceiling fractions
    per_L = payload.get("per_L")
    if isinstance(per_L, dict):
        t = wandb.Table(columns=[
            "L", "body_us", "serial_strict_us", "serial_perm_us",
            "pipe_dep_strict_us", "pipe_dep_perm_us", "pipe_indep_strict_us", "pipe_indep_perm_us",
            "indep_mono_strict_us", "serial_tax_us", "dep_tax_us", "ceil_tax_us",
            "exposed_mono_strict_us", "realizable_overlap_fraction", "ceiling_overlap_fraction",
            "dependency_bounded_strict_tps", "ceiling_strict_tps", "single_stream_floor_tps",
            "multistream_overhead_us", "pipeline_fill_drain_us"])
        for L, d in sorted(per_L.items(), key=lambda kv: int(kv[0])):
            if "error" in d:
                continue
            m = d.get("median_us", {})
            t.add_data(int(L), m.get("body"), m.get("serial_strict"), m.get("serial_perm"),
                       m.get("pipe_dep_strict"), m.get("pipe_dep_perm"), m.get("pipe_indep_strict"),
                       m.get("pipe_indep_perm"), m.get("indep_mono_strict"),
                       d.get("serial_tax_us"), d.get("dep_tax_us"), d.get("ceil_tax_us"),
                       d.get("exposed_mono_strict_us"), d.get("realizable_overlap_fraction"),
                       d.get("ceiling_overlap_fraction"), d.get("dependency_bounded_strict_tps"),
                       d.get("ceiling_strict_tps"), d.get("single_stream_floor_tps"),
                       d.get("multistream_overhead_us"), d.get("pipeline_fill_drain_us"))
        run.log({"per_L_dependency_sweep": t})

    # strict 2D byte-identity survives the pipelined capture+replay (== #466/#472/#477)
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
