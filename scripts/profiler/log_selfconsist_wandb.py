#!/usr/bin/env python
"""Log a PR #720 self-consistency result.json to W&B (analysis-only).

Run under a python that has wandb (system python3 here; the serve venvs don't).
One run per config; prints ``WANDB_RUN_ID=<id>`` for the SENPAI-RESULT marker.

#319 GPU directive flags (advisor): analysis_only=1, official_tps=0,
no_hf_job=1, fires=0 -- this is a LOCAL measurement, nothing fires.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ENTITY = "wandb-applied-ai-team"
PROJECT = "gemma-challenge-senpai"


def flatten(prefix: str, d: dict, out: dict) -> None:
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flatten(key + "/", v, out)
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[key] = v
        else:
            out[key] = json.dumps(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path, required=True, help="result.json from the profiler")
    ap.add_argument("--complement", type=Path, default=None, help="optional complement.json (vs-anchor)")
    ap.add_argument("--name", required=True, help="wandb run name, e.g. land/selfconsist-g32-locus")
    ap.add_argument("--group", default="pr720-selfconsist-319")
    args = ap.parse_args()

    import wandb

    res = json.loads(args.result.read_text())
    comp = json.loads(args.complement.read_text()) if args.complement and args.complement.exists() else {}

    config = {
        "pr": res.get("pr", 720),
        "analysis_only": int(bool(res.get("analysis_only", 1))),
        "official_tps": res.get("official_tps", 0),
        "no_hf_job": int(bool(res.get("no_hf_job", 1))),
        "fires": int(bool(res.get("fires", 0))),
        "label": res.get("label") or res.get("config"),
        "config_dir": res.get("config_dir"), "substrate": res.get("substrate"),
        "num_prompts": res.get("num_prompts"), "output_len": res.get("output_len"),
        "seed": res.get("seed"), "bi": res.get("bi", True),
        "legs": json.dumps(res.get("legs", [])),
        "verdict": res.get("verdict"),
        "strict_literal_holds": res.get("strict_literal_holds"),
        "floor_clean": res.get("floor_clean"),
    }
    os.environ.setdefault("WANDB_SILENT", "true")
    run = wandb.init(entity=ENTITY, project=PROJECT, name=args.name, group=args.group,
                     job_type="selfconsist_319", config=config, reinit=True)

    summary = {"verdict": res.get("verdict")}
    # comparison blocks (dual_substrate: dev307_vs_ar/...; k5: floor_ab/spec_vs_ar)
    for k in ("dev307_vs_ar", "floor_vs_ar", "dev307graph_vs_eager", "dev307eager_vs_ar",
              "floor_ab", "spec_vs_ar"):
        if isinstance(res.get(k), dict):
            flatten(f"{k}/", res[k], summary)
    for k in ("dev307_self_consistent", "strict_literal_holds", "floor_clean",
              "all_residual_flips_known_ties", "confident_genuine_flips",
              "gap_tau", "max_residual_gap_nat"):
        if k in res:
            summary[k] = res[k]
    # gap-probe scalars (the advisor fire metric: confident_genuine_flips@0.3);
    # skip the bulky per-flip `records` array, keep the small confident set.
    gp = res.get("gap_probe")
    if isinstance(gp, dict):
        for gk in ("tau", "n_probed", "n_errors", "confident_genuine_flips", "max_gap"):
            if gk in gp:
                summary[f"gap_probe/{gk}"] = gp[gk]
        if gp.get("confident_records") is not None:
            summary["gap_probe/confident_records"] = json.dumps(gp["confident_records"])
    for k in ("break_attribution", "residual_flips_not_known_ties"):
        if res.get(k) is not None:
            summary[k] = json.dumps(res[k])
    for leg, m in (res.get("legs_meta") or {}).items():
        flatten(f"leg/{leg}/", m, summary)
    if comp:
        flatten("complement/", comp, summary)
    run.summary.update(summary)
    rid = run.id
    run.finish()
    print(f"WANDB_RUN_ID={rid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
