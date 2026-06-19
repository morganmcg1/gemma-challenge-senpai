#!/usr/bin/env python
"""Log PR #708 mixed-grid-servability op-bench results to W&B.

Runs in BASE python (working wandb 0.27.0); the dev307 vLLM venv ships a broken
wandb stub (no .init), which is why the GPU op-bench writes JSON and this logs it.
Uploads SOURCE_READ.md (decisive source-code path + line refs) as an artifact.
"""
from __future__ import annotations

import argparse
import json
import os

VERDICT = "MIXED_GRID_SERVABLE_STANDALONE_FUSED_BLOCKED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(__file__), "mixed_grid_results.json"))
    ap.add_argument("--source-read", default=os.path.join(os.path.dirname(__file__), "SOURCE_READ.md"))
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="mixed-grid-marlin-servability-land")
    ap.add_argument("--wandb_name", default="land/mixed-grid-marlin-servability")
    args = ap.parse_args()

    payload = json.load(open(args.results))
    h = payload["headline"]

    import wandb
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type="mixed-grid-marlin-servability", config=payload.get("config", {}))

    flat = {k: v for k, v in h.items() if isinstance(v, (int, float, bool))}
    # per-component isolated tax + delta
    for n, r in payload.get("iso", {}).items():
        flat[f"iso_{n}_tax_frac"] = r["tax_frac"]
        flat[f"iso_{n}_delta_us"] = r["per_call_delta_us"]
        flat[f"iso_{n}_us_g128"] = r["per_call_us_g128"]
        flat[f"iso_{n}_us_g32"] = r["per_call_us_g32"]
    # full-body measurements
    for lbl, b in payload.get("bodies", {}).items():
        flat[f"body_{lbl}_us"] = b["us"]
        flat[f"body_{lbl}_us_ci"] = b["us_ci"]

    wandb.log(flat)
    wandb.summary.update(flat)
    wandb.summary["verdict"] = VERDICT
    wandb.summary["mixed_grid_servable"] = h["mixed_grid_servable"]
    wandb.summary["mixed_grid_opbench_tps"] = h["mixed_grid_opbench_tps"]

    # 3-anchor comparison table
    tbl = wandb.Table(columns=["config", "opbench_tps", "delta_vs_g128_tps", "note"])
    g128 = h["tps_g128_anchor"]
    rows = [
        ("all_g128_anchor", g128, 0.0, "land #707 AR rung"),
        ("all_g32_full_measured", h["tps_g32_full_measured"], h["tps_g32_full_measured"] - g128, "refined full-g32 floor"),
        ("all_g32_full_707", h["tps_g32_full_707"], h["tps_g32_full_707"] - g128, "land #707 37-layer model"),
        ("proj_706_linear", h["proj_706"], h["proj_706"] - g128, "denken #706 projection target"),
        ("mix_servable_40plig", h["tps_mix_servable_40plig"], h["tps_mix_servable_40plig"] - g128, "SERVABLE ship path"),
        ("fakequant48_ideal", h["tps_fakequant48_ideal"], h["tps_fakequant48_ideal"] - g128, "NOT servable (8 attn fused)"),
        ("mix_wholeqkv3", h["tps_mix_wholeqkv3"], h["tps_mix_wholeqkv3"] - g128, "servable attn route, 3 qkv blocks"),
        ("mix_wholeqkv8", h["tps_mix_wholeqkv8"], h["tps_mix_wholeqkv8"] - g128, "servable attn route, 8 qkv blocks"),
    ]
    for r in rows:
        tbl.add_data(*r)
    wandb.log({"anchor_comparison": tbl})

    if os.path.exists(args.source_read):
        art = wandb.Artifact("mixed_grid_source_read_708", type="source-read")
        art.add_file(args.source_read)
        art.add_file(args.results)
        run.log_artifact(art)

    rid = run.id
    print(f"[wandb] run_id={rid} url={run.url}")
    wandb.finish()
    payload["wandb_run_id"] = rid
    json.dump(payload, open(args.results, "w"), indent=2)


if __name__ == "__main__":
    main()
