"""Log measure_drafter result.json files to W&B (group bi0-drafter-gemm).

Run via the project venv (`uv run python ...`) which has wandb; the serve venv
(vllm022) does not. Reads runs/<label>/result.json and logs config + summary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent
GROUP = "bi0-drafter-gemm"


def decode_tps(result: dict) -> float | None:
    ds = result.get("decode_summary") or {}
    n, dur = ds.get("num_completion_tokens"), ds.get("duration_s")
    if isinstance(n, (int, float)) and isinstance(dur, (int, float)) and dur > 0:
        return n / dur
    return None


def main() -> int:
    labels = sys.argv[1:]
    if not labels:
        labels = [p.parent.name for p in (HERE / "runs").glob("*/result.json")]
    for label in labels:
        rp = HERE / "runs" / label / "result.json"
        if not rp.exists():
            print(f"[skip] {rp} missing")
            continue
        r = json.loads(rp.read_text())
        ds = r.get("decode_summary") or {}
        sm = r.get("spec_metrics") or {}
        tp = r.get("tps_probe") or {}
        run = wandb.init(
            entity="wandb-applied-ai-team", project="gemma-challenge-senpai",
            group=GROUP, name=f"stark/drafter-{label}", job_type="drafter-gemm-format",
            config={
                "pr": 786, "label": label,
                "spec_quant": r.get("spec_quant"),
                "num_prompts": r.get("num_prompts"), "output_len": r.get("output_len"),
                "server_started": r.get("server_started"),
                "baseline_submission": "int4_mtp_bi0_surgattn",
            },
            reinit=True,
        )
        summ = {
            "server_started": r.get("server_started"),
            "startup_s": r.get("startup_s"),
            "decode_tps_steady": decode_tps(r),
            "warm_probe_decode_tps": tp.get("decode_tps_single_stream"),
            "num_completion_tokens": ds.get("num_completion_tokens"),
            "num_records": ds.get("num_records"),
            "verify_gpu_ms": r.get("verify_gpu_ms"),
            "drafter_gpu_ms": r.get("drafter_gpu_ms"),
            "drafter_over_verify_ratio": r.get("drafter_over_verify_ratio"),
            "drafter_frac_of_gpu_busy": r.get("drafter_frac_of_gpu_busy"),
            "draft_acceptance_rate": sm.get("draft_acceptance_rate"),
            "e_accept_mean_acceptance_length": sm.get("e_accept_mean_acceptance_length"),
            "num_accepted_tokens": sm.get("num_accepted_tokens"),
            "num_draft_tokens": sm.get("num_draft_tokens"),
            "greedy_verdict": r.get("greedy_verdict"),
            "ppl": (r.get("ppl_summary") or {}).get("ppl"),
            "error": r.get("error"),
        }
        run.summary.update({k: v for k, v in summ.items() if v is not None})
        print(f"[wandb] {label}: run {run.id}  decode_tps={summ['decode_tps_steady']} "
              f"started={summ['server_started']} verdict={summ['greedy_verdict']}")
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
