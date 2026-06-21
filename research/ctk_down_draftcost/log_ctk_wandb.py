"""Log the PR #824 CENTROID_TOP_K DOWN-sweep to W&B (group bi0-ctk-down-draftcost).

Reads ``sweep/ctk_sweep_results.json`` produced by ctk_sweep.py and logs ONE run
per ctk arm (config.ctk + summary metrics -> metric-vs-ctk plottable across the
group), plus one ``curve`` overview run carrying a table of all arms and the full
results JSON artifact. Run under the repo .venv (the serve venv has no wandb and a
local ./wandb dir shadows the import).

Usage: .venv/bin/python research/ctk_down_draftcost/log_ctk_wandb.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))
from scripts.wandb_logging import (  # noqa: E402
    finish_wandb,
    init_wandb_run,
    log_json_artifact,
    log_summary,
)

GROUP = "bi0-ctk-down-draftcost"
RESULTS = ROOT / "research" / "ctk_down_draftcost" / "sweep" / "ctk_sweep_results.json"


def _arm_summary(r: dict, full: dict) -> dict:
    out = {
        "ctk": r.get("ctk"),
        "active_tokens": r.get("active_tokens"),
        "drafter_gpu_ms": r.get("drafter_gpu_ms"),
        "verify_gpu_ms": r.get("verify_gpu_ms"),
        "e_accept": r.get("e_accept"),
        "wall_tps_mean": r.get("wall_tps_mean"),
        "wall_tps_std": r.get("wall_tps_std"),
        "wall_tps_cv_pct": r.get("wall_tps_cv_pct"),
        "steady_tps_mean": r.get("steady_tps_mean"),
        "steady_tps_std": r.get("steady_tps_std"),
        "wall_tps_delta_pct": r.get("wall_tps_delta_pct"),
        "steady_tps_delta_pct": r.get("steady_tps_delta_pct"),
        "ppl": r.get("ppl"),
        "num_drafts": full.get("num_drafts"),
    }
    for k, v in (r.get("per_position_accept_rate") or {}).items():
        out[f"accept_rate_{k}"] = v
    recs = r.get("decode_records") or []
    out["decode_records_min"] = min(recs) if recs else None
    out["reps"] = len(recs)
    return out


def main() -> int:
    data = json.loads(RESULTS.read_text())
    agg = data["aggregate"]
    arms_full = {a["label"]: a for a in data.get("arms_full", [])}
    arm_rows = agg["arms"]

    run_ids = []
    for r in arm_rows:
        full = arms_full.get(r["label"], {})
        summary = _arm_summary(r, full)
        run = init_wandb_run(
            job_type="ctk-sweep",
            agent="lawine",
            name=f"lawine/ctk-down-{r['label']}",
            group=GROUP,
            project="gemma-challenge-senpai",
            entity="wandb-applied-ai-team",
            tags=["pr824", "ctk", "drafter", "draftcost", "down-sweep",
                  "control" if str(r["label"]).endswith("control") else "variant"],
            notes=(
                "PR #824 CENTROID_TOP_K down-sweep arm. Local A10G exploratory "
                "(not official a10g-small TPS). Greedy temp=0 is lossless -> PPL/"
                "128-of-128 ctk-independent (measured on control). ctk lever: "
                "num_selected = ctk*128 draft-head rows/step."
            ),
            config={
                "pr": 824, "ctk": r.get("ctk"), "active_tokens": r.get("active_tokens"),
                "K": agg.get("K", 6), "workload": "128x512", "conc": 1,
                "drafter": "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant",
                "is_control": str(r["label"]).endswith("control"),
            },
        )
        if run is None:
            print("WANDB not initialized (no api key / disabled).", flush=True)
            return 1
        log_summary(run, summary, step=0)
        print(f"WANDB_ARM {r['label']} ctk={r.get('ctk')} id={run.id} url={run.url}", flush=True)
        run_ids.append(run.id)
        finish_wandb(run)

    # Overview run: table of the whole curve + full results artifact.
    run = init_wandb_run(
        job_type="ctk-sweep-curve",
        agent="lawine",
        name="lawine/ctk-down-curve",
        group=GROUP,
        project="gemma-challenge-senpai",
        entity="wandb-applied-ai-team",
        tags=["pr824", "ctk", "down-sweep", "curve", "summary"],
        notes="PR #824 ctk DOWN-sweep curve overview (drafter_gpu_ms / E_accept / TPS vs ctk).",
        config={"pr": 824, "arms": [r.get("ctk") for r in arm_rows], "K": agg.get("K", 6)},
    )
    if run is not None:
        try:
            import wandb

            cols = ["label", "ctk", "active_tokens", "drafter_gpu_ms", "verify_gpu_ms",
                    "e_accept", "wall_tps_mean", "wall_tps_std", "wall_tps_delta_pct",
                    "steady_tps_mean", "steady_tps_delta_pct", "ppl"]
            tbl = wandb.Table(columns=cols)
            for r in arm_rows:
                tbl.add_data(*[r.get(c) for c in cols])
            run.log({"ctk_curve": tbl})
        except Exception as exc:  # noqa: BLE001
            print(f"[wandb] table log failed ({exc})", flush=True)
        log_json_artifact(run, name="ctk_sweep_results", artifact_type="results", data=data)
        print(f"WANDB_CURVE id={run.id} url={run.url}", flush=True)
        run_ids.append(run.id)
        finish_wandb(run)

    print("WANDB_RUN_IDS " + ",".join(run_ids), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
