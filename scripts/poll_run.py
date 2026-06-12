#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.common import agent_id, hf_bucket_uri, hf_json, load_dotenv


TERMINAL = {"completed", "error", "timed_out", "cancelled", "failed"}


def read_status(agent: str, run_prefix: str) -> dict:
    return hf_json(hf_bucket_uri(agent, f"{run_prefix}/job_status.json"))


def read_summary(agent: str, run_prefix: str) -> dict:
    return hf_json(hf_bucket_uri(agent, f"{run_prefix}/summary.json"))


def poll_run(agent: str, run_prefix: str, *, wait: bool, interval_s: int, timeout_s: int) -> dict | None:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            status = read_status(agent, run_prefix)
        except Exception as exc:
            if not wait:
                raise
            print(f"status unavailable yet: {exc}")
            status = {"status": "missing"}
        state = str(status.get("status") or status.get("stage") or "unknown")
        print(f"{run_prefix}: {state}")

        if state in TERMINAL or not wait:
            break
        if time.monotonic() >= deadline:
            raise SystemExit(f"timed out waiting for {run_prefix}")
        time.sleep(interval_s)

    if state == "completed":
        summary = read_summary(agent, run_prefix)
        print(summary)
        return summary
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll a Gemma challenge HF Job run in the scratch bucket.")
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--interval-s", type=int, default=30)
    parser.add_argument("--timeout-s", type=int, default=1800)
    args = parser.parse_args()

    load_dotenv()
    poll_run(agent_id(args.agent_id), args.run_prefix, wait=args.wait, interval_s=args.interval_s, timeout_s=args.timeout_s)


if __name__ == "__main__":
    main()
