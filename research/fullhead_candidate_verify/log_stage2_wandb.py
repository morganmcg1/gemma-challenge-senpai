#!/usr/bin/env python3
"""Log the EXISTING stage2_report.json (offline candidate miss-rate / K_safe) to
W&B. Re-uses the measured report verbatim — no recompute. Runs under SYSTEM
python3; imports wandb first to beat the ./wandb namespace shadow under ROOT."""
from __future__ import annotations

try:
    import wandb as _wandb_real  # noqa: F401  cache real module
except Exception:
    _wandb_real = None

import json
import sys
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
HERE = ROOT / "research" / "fullhead_candidate_verify"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.wandb_logging import (finish_wandb, init_wandb_run,  # noqa: E402
                                   log_json_artifact, log_summary)

S2 = HERE / "stage2_report.json"


def main() -> int:
    rep = json.loads(S2.read_text())
    run = init_wandb_run(
        job_type="systems-profile",
        agent="fern",
        name="fern/fullhead-stage2-missrate",
        group="fullhead-candidate-verify",
        tags=["fullhead", "candidate-verify", "stage2", "miss-rate", "K-safe",
              "local-a10g", "analysis-only"],
        notes="PR #549 Stage 2: offline candidate miss-rate(K) -> K_safe over 60000 held-out positions",
        config={
            "n_positions": rep["n_positions"],
            "n_rows_required": rep["n_rows_required"],
            "vocab": rep["vocab"], "hidden": rep["hidden"],
            "full_head_bytes": rep["full_head_bytes"],
        },
    )
    if run is None:
        print("[stage2-log] wandb init returned None (no API key / disabled)", flush=True)
        return 1

    # miss_rate(K) curves per scheme, logged along a shared K step axis.
    Ks = rep["Ks"]
    for i, K in enumerate(Ks):
        payload = {"global_step": K, "K": K}
        for sname, sc in rep["schemes"].items():
            mr = sc["miss_rate_by_K_conservative"].get(str(K))
            if mr is not None:
                payload[f"missrate/{sname}"] = mr
        run.log(payload)

    summary = {
        "n_positions": rep["n_positions"],
        "rows_ok": rep["rows_ok"],
        "gold_tie_frac": rep["gold_tie_frac"],
        "gold_fp32_vs_bf16_disagreements": rep["gold_fp32_vs_bf16_disagreements"],
        "K_safe_int4_g128": rep["schemes"]["int4_g128"]["K_safe_conservative"],
        "K_safe_int4_perrow": rep["schemes"]["int4_perrow"]["K_safe_conservative"],
        "K_safe_fp8_e4m3": rep["schemes"]["fp8_e4m3"]["K_safe_conservative"],
        "int4_g128_head_read_gb": rep["schemes"]["int4_g128"]["served_head_read_bytes"] / 1e9,
        "int4_perrow_head_read_gb": rep["schemes"]["int4_perrow"]["served_head_read_bytes"] / 1e9,
        "fp8_e4m3_head_read_gb": rep["schemes"]["fp8_e4m3"]["served_head_read_bytes"] / 1e9,
        "int4_g128_miss_at_1": rep["schemes"]["int4_g128"]["miss_rate_by_K_conservative"]["1"],
        "fp8_e4m3_miss_at_1": rep["schemes"]["fp8_e4m3"]["miss_rate_by_K_conservative"]["1"],
        "analysis_only": True, "official_tps": 0,
        "primary_metric": rep["schemes"]["int4_g128"]["K_safe_conservative"],
    }
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="fullhead-stage2-report",
                      artifact_type="stage2-report", data=rep)
    rid = getattr(run, "id", None)
    finish_wandb(run)
    print(f"[stage2-log] wandb run id = {rid}", flush=True)

    rep["wandb_run_id"] = rid
    S2.write_text(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
