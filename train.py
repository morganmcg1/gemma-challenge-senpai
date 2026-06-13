#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from scripts.common import ROOT, agent_id, load_dotenv, run_prefix, submission_name, submission_prefix
from scripts.poll_run import poll_run
from scripts.run_hf_job import launch_job
from scripts.upload_submission import upload_submission
from scripts.wandb_logging import (
    finish_wandb,
    log_event,
    log_file_artifact,
    log_json_artifact,
    init_wandb_run,
)


def print_result(summary: dict, run: str) -> None:
    tps = summary.get("tps") or summary.get("output_tps") or 0
    ppl = summary.get("ppl") or 0
    completed = summary.get("completed") or summary.get("completed_requests") or summary.get("num_completed_requests") or "unknown"
    print(f"SENPAI-RESULT tps={tps} ppl={ppl} completed={completed} run_prefix={run}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Senpai entrypoint for Gemma challenge benchmark experiments.")
    parser.add_argument("--agent", "--agent-id", dest="agent_id", default=None)
    parser.add_argument("--submission", default="submissions/vllm_baseline")
    parser.add_argument("--name", default=None)
    parser.add_argument("--method", default="")
    parser.add_argument("--launch", action="store_true", help="Launch the challenge HF Job after upload.")
    parser.add_argument("--wait", action="store_true", help="Wait for the HF Job and print SENPAI-RESULT.")
    parser.add_argument("--run-prefix", default=None)
    parser.add_argument("--interval-s", type=int, default=30)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--wandb-notes", default=os.environ.get("WANDB_NOTES", ""))
    args = parser.parse_args()

    load_dotenv()
    agent = agent_id(args.agent_id)
    path = (ROOT / args.submission).resolve()
    name = submission_name(path, args.name)
    run = args.run_prefix or run_prefix(agent, name)
    wandb_run = init_wandb_run(
        job_type="experiment",
        agent=agent,
        name=f"{agent}-{name}",
        notes=args.wandb_notes,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        tags=["submission", "hf-job" if args.launch else "upload-only"],
        config={
            "submission_path": str(path),
            "submission_name": name,
            "method": args.method,
            "launch": args.launch,
            "wait": args.wait,
            "run_prefix": run,
            "interval_s": args.interval_s,
            "timeout_s": args.timeout_s,
        },
    )

    try:
        step = 0
        log_event(wandb_run, "upload_start", step=step, data={"submission_path": str(path)})
        dest = upload_submission(path, agent=agent, name=name)
        print(f"uploaded_submission={dest}")
        step += 1
        log_event(
            wandb_run,
            "upload_complete",
            step=step,
            metrics={"submission/uploaded": 1},
            data={"submission_uri": dest},
        )
        log_file_artifact(
            wandb_run,
            path=path / "manifest.json",
            name=f"{agent}-{name}-manifest",
            artifact_type="submission-manifest",
        )

        if not args.launch:
            print("upload complete; pass --launch to run the HF benchmark")
            return

        step += 1
        submission = submission_prefix(agent, name)
        response = launch_job(agent=agent, submission=submission, run=run)
        log_event(
            wandb_run,
            "launch_complete",
            step=step,
            metrics={"job/launched": 1},
            data={"submission_prefix": submission, "run_prefix": run},
        )
        log_json_artifact(
            wandb_run,
            name=f"{agent}-{name}-launch-response",
            artifact_type="hf-job-response",
            data=response,
        )

        if args.wait:
            summary = poll_run(
                agent,
                run,
                wait=True,
                interval_s=args.interval_s,
                timeout_s=args.timeout_s,
                wandb_run=wandb_run,
                initial_step=step + 1,
            )
            if summary:
                print_result(summary, run)
        else:
            print(f"launched run_prefix={run}")
    finally:
        finish_wandb(wandb_run)


if __name__ == "__main__":
    main()
