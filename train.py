#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.common import ROOT, agent_id, load_dotenv, run_prefix, submission_name, submission_prefix
from scripts.poll_run import poll_run
from scripts.run_hf_job import launch_job
from scripts.upload_submission import upload_submission


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
    args = parser.parse_args()

    load_dotenv()
    agent = agent_id(args.agent_id)
    path = (ROOT / args.submission).resolve()
    name = submission_name(path, args.name)
    dest = upload_submission(path, agent=agent, name=name)
    print(f"uploaded_submission={dest}")

    if not args.launch:
        print("upload complete; pass --launch to run the HF benchmark")
        return

    run = args.run_prefix or run_prefix(agent, name)
    launch_job(agent=agent, submission=submission_prefix(agent, name), run=run)

    if args.wait:
        summary = poll_run(agent, run, wait=True, interval_s=args.interval_s, timeout_s=args.timeout_s)
        if summary:
            print_result(summary, run)
    else:
        print(f"launched run_prefix={run}")


if __name__ == "__main__":
    main()
