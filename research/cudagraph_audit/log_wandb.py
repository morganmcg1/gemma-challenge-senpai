#!/usr/bin/env python3
"""Log the PR #787 bi0 CUDA-graph coverage audit to W&B.

Local-only audit (no HF Job): records the M=K+1=7 verify-pass cudagraph hit rate
plus the graph-on / graph-off (enforce-eager) TPS comparison as two runs grouped
under ``bi0-cudagraph-audit``. Run from the repo root with the repo ``.venv``
python (which has wandb + the WANDB_* env). Idempotent enough for a one-shot
audit; re-running creates fresh runs.

Evidence for the hardcoded audit verdict (see logs/pr787/serve_on_metrics.log):
the ``--cudagraph-metrics`` table shows ``| 7 | 7 | 0 | FULL | ~670 |`` in every
~10s window (30/30 windows, ALL M=7 rows FULL, 0 paddings; NONE rows are prefill
chunks only). Mechanism: vLLM 0.22.0 sets uniform_decode_query_len = 1 + K = 7
and captures a dedicated FULL graph for that shape.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import wandb

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "research/cudagraph_audit"
PROJECT = os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai")
ENTITY = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
GROUP = "bi0-cudagraph-audit"

# Shared config: the bi0 serving shape under audit.
SHARED_CONFIG = {
    "submission": "int4_mtp_bi0_surgattn",
    "pr": 787,
    "model_id": "google/gemma-4-E4B-it-qat-w4a16-ct",
    "num_speculative_tokens": 6,
    "max_num_seqs": 1,
    "max_num_batched_tokens": 512,
    "max_model_len": 4096,
    "verify_pass_tokens_M": 7,  # K+1
    "uniform_decode_query_len": 7,
    "vllm_version": "0.22.0",
}

# Audit verdict (graph-on arm), established from the --cudagraph-metrics table.
AUDIT_VERDICT = {
    "m7_verify_runtime_mode": "FULL",
    "m7_verify_num_paddings": 0,
    "m7_verify_hit_fraction": 1.0,
    "m7_non_full_rows": 0,
    "cudagraph_mode": "FULL_AND_PIECEWISE",
    "cudagraph_capture_sizes": [1, 2, 4, 8],
    "hypothesis": "verify pass (M=7) falls back to eager -> add to capture list",
    "verdict": "REFUTED: M=7 already 100% FULL-captured via uniform_decode_query_len",
}


def _summary(path: Path) -> dict:
    return json.loads(path.read_text())


def _log_run(name: str, config: dict, summary: dict) -> str:
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        name=name,
        job_type="cudagraph-audit",
        config=config,
        reinit=True,
    )
    run.summary.update(summary)
    run_id = run.id
    run.finish()
    return run_id


def main() -> None:
    on = _summary(AUDIT / "on/local_summary.json")
    eager_decode = _summary(AUDIT / "eager/decode_summary.json")
    eager_tps = (eager_decode["num_completion_tokens"] / eager_decode["duration_s"]) if eager_decode.get("duration_s") else 0.0

    on_tps = float(on["tps"])
    speedup_pct = (on_tps / eager_tps - 1.0) * 100.0 if eager_tps else 0.0

    on_id = _log_run(
        "lawine/bi0-cudagraph-audit-graph-on",
        {**SHARED_CONFIG, **AUDIT_VERDICT, "enforce_eager": False, "cudagraph": "on"},
        {
            "tps_local": on_tps,
            "ppl": float(on["ppl"]),
            "ppl_num_records": on["ppl_num_records"],
            "completed": on["completed"],
            "decode_num_records": on["decode_num_records"],
            "decode_duration_s": on["decode_duration_s"],
            "graph_speedup_vs_eager_pct": speedup_pct,
        },
    )

    eager_id = _log_run(
        "lawine/bi0-cudagraph-audit-eager",
        {**SHARED_CONFIG, "enforce_eager": True, "cudagraph": "off",
         "note": "negative control: graphs+inductor-compile OFF (custom_ops path)"},
        {
            "tps_local": eager_tps,
            "decode_num_records": eager_decode["num_records"],
            "decode_duration_s": eager_decode["duration_s"],
        },
    )

    print(json.dumps({
        "graph_on_run_id": on_id,
        "graph_on_tps": round(on_tps, 4),
        "eager_run_id": eager_id,
        "eager_tps": round(eager_tps, 4),
        "graph_speedup_vs_eager_pct": round(speedup_pct, 2),
        "project": f"{ENTITY}/{PROJECT}",
        "group": GROUP,
    }, indent=2))


if __name__ == "__main__":
    main()
