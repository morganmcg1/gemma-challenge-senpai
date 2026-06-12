#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.common import DEFAULT_API, agent_id, post_json, require_hf_token, run_prefix, submission_prefix


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
    args = parser.parse_args()

    agent = agent_id(args.agent_id)
    submission = args.submission_prefix or submission_prefix(agent, args.submission_name)
    run = args.run_prefix or run_prefix(agent, args.submission_name)
    print(f"submission_prefix={submission}")
    print(f"run_prefix={run}")
    launch_job(agent=agent, submission=submission, run=run, api=args.api)


if __name__ == "__main__":
    main()
