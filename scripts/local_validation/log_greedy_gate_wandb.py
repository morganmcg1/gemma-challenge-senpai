"""Log a served-vs-served greedy-gate comparison to W&B.

The greedy gate for a spec/fold submission is a 3-stage served-vs-served pipeline
(``gen_greedy_reference`` for an exact reference, again for the candidate, then
``greedy_gate`` to compare) — see ``research/validity/lf29cap444_pupa_check``.
That pipeline writes a ``greedy_report.json`` (the official
``ComparisonReport.to_dict()``) but does not itself touch W&B. This reads that
report (plus the reference/candidate ``meta.json`` for provenance) and logs the
verdict, the per-token and per-prompt flip rates, and the divergence-onset
distribution as a single ``greedy-gate`` run, mirroring
``validate_submission._maybe_log_wandb``.

    python -m scripts.local_validation.log_greedy_gate_wandb \\
        --report research/validity/<sub>/greedy_gate/greedy_report.json \\
        --reference-meta research/validity/<sub>/greedy_gate/reference_m1ar_exactffn/meta.json \\
        --candidate-meta research/validity/<sub>/greedy_gate/candidate_m1ar_foldon/meta.json \\
        --wandb-name wirbel/lf29cap444-greedy-gate \\
        --wandb-group fa2sw-precache-validate-and-lf29-check \\
        --flip-rate-metric pupa_lf29_greedy_flip_rate_per_token
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text())


def summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    """Derive the gate's headline metrics from a ComparisonReport dict."""
    total_tokens = report.get("total_tokens_compared", 0) or 0
    total_divergent = report.get("total_divergent_tokens", 0) or 0
    num_prompts = report.get("num_prompts_compared", 0) or 0
    num_divergent = report.get("num_divergent", 0) or 0

    onsets = sorted(
        p["first_divergence_index"]
        for p in report.get("per_prompt", [])
        if not p.get("identical") and p.get("first_divergence_index") is not None
    )
    summary: dict[str, Any] = {
        "greedy_verdict": report.get("verdict"),
        "num_prompts_compared": num_prompts,
        "num_identical": report.get("num_identical", 0),
        "num_divergent": num_divergent,
        "total_tokens_compared": total_tokens,
        "total_divergent_tokens": total_divergent,
        "flip_rate_per_token": (total_divergent / total_tokens) if total_tokens else 0.0,
        "flip_rate_per_prompt": (num_divergent / num_prompts) if num_prompts else 0.0,
    }
    if onsets:
        summary["divergence_onset_min"] = onsets[0]
        summary["divergence_onset_median"] = int(statistics.median(onsets))
        summary["divergence_onset_max"] = onsets[-1]
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", type=Path, required=True, help="greedy_report.json (ComparisonReport.to_dict())")
    ap.add_argument("--reference-meta", type=Path, default=None, help="reference stage meta.json")
    ap.add_argument("--candidate-meta", type=Path, default=None, help="candidate stage meta.json")
    ap.add_argument("--submission", default=None, help="submission name/path for run config")
    ap.add_argument("--wandb-name", required=True, help="W&B run name")
    ap.add_argument("--wandb-group", default=None, help="W&B group tag")
    ap.add_argument(
        "--flip-rate-metric",
        default="greedy_flip_rate_per_token",
        help="extra summary key to alias flip_rate_per_token under (matches the PR's primary_metric name)",
    )
    args = ap.parse_args(argv)

    report = _load_json(args.report)
    if not report:
        raise SystemExit(f"no report found at {args.report}")
    ref_meta = _load_json(args.reference_meta)
    cand_meta = _load_json(args.candidate_meta)

    summary = summarize_report(report)
    # Alias under the PR's primary_metric name so the dashboard/marker line up.
    summary[args.flip_rate_metric] = summary["flip_rate_per_token"]

    print(
        f"[log-greedy-gate] verdict={summary['greedy_verdict']} "
        f"flip_rate_per_token={summary['flip_rate_per_token']:.6f} "
        f"({summary['total_divergent_tokens']}/{summary['total_tokens_compared']} tok) "
        f"divergent_prompts={summary['num_divergent']}/{summary['num_prompts_compared']}",
        flush=True,
    )

    try:
        from scripts.wandb_logging import (
            finish_wandb,
            init_wandb_run,
            log_file_artifact,
            log_summary,
        )
    except Exception as exc:  # pragma: no cover - logging must never break the report
        print(f"[log-greedy-gate] wandb logging unavailable: {exc}", flush=True)
        return 0

    run = init_wandb_run(
        job_type="greedy-gate",
        agent="senpai",
        name=args.wandb_name,
        tags=["greedy-gate", *([args.wandb_group] if args.wandb_group else [])],
        config={
            "submission": args.submission,
            "reference_kind": ref_meta.get("reference_kind"),
            "reference_ref_env": ref_meta.get("ref_env"),
            "candidate_ref_env": cand_meta.get("ref_env"),
            "model_id": cand_meta.get("model_id") or ref_meta.get("model_id"),
            "output_len": cand_meta.get("output_len") or ref_meta.get("output_len"),
            "num_records": cand_meta.get("num_records"),
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[log-greedy-gate] W&B run not created (no creds/disabled); printed summary only", flush=True)
        return 0
    log_summary(run, summary, step=0)
    log_file_artifact(run, path=Path(args.report), name="greedy_report", artifact_type="greedy-gate-report")
    finish_wandb(run)
    print(f"[log-greedy-gate] logged run {args.wandb_name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
