#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.common import DEFAULT_API, agent_id, load_dotenv, post_json, require_hf_token, run_prefix, submission_prefix
from scripts.wandb_logging import finish_wandb, init_wandb_run, log_event, log_json_artifact


def launch_job(
    *,
    agent: str,
    submission: str,
    run: str,
    api: str = DEFAULT_API,
    token: str | None = None,
) -> dict:
    token = token or require_hf_token()
    payload = {
        "agent_id": agent,
        "submission_prefix": submission,
        "run_prefix": run,
    }
    response = post_json(f"{api}/v1/jobs:run", payload, token=token)
    print(response)
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Gemma challenge org-credit HF Job.")
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--submission-name", default="vllm-baseline")
    parser.add_argument("--submission-prefix", default=None)
    parser.add_argument("--run-prefix", default=None)
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--wandb-notes", default="")
    args = parser.parse_args()

    load_dotenv()
    agent = agent_id(args.agent_id)
    submission = args.submission_prefix or submission_prefix(agent, args.submission_name)
    run = args.run_prefix or run_prefix(agent, args.submission_name)
    print(f"submission_prefix={submission}")
    print(f"run_prefix={run}")
    wandb_run = init_wandb_run(
        job_type="launch",
        agent=agent,
        name=f"{agent}-{args.submission_name}-launch",
        notes=args.wandb_notes,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        tags=["hf-job"],
        config={
            "submission_name": args.submission_name,
            "submission_prefix": submission,
            "run_prefix": run,
            "api": args.api,
        },
    )
    try:
        log_event(
            wandb_run,
            "launch_start",
            step=0,
            data={"submission_prefix": submission, "run_prefix": run},
        )
        response = launch_job(agent=agent, submission=submission, run=run, api=args.api)
        log_event(
            wandb_run,
            "launch_complete",
            step=1,
            metrics={"job/launched": 1},
            data={"run_prefix": run},
        )
        log_json_artifact(
            wandb_run,
            name=f"{agent}-{run}-launch-response",
            artifact_type="hf-job-response",
            data=response,
        )
    finally:
        finish_wandb(wandb_run)


if __name__ == "__main__":
    main()
