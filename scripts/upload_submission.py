#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.common import ROOT, agent_id, hf_bucket_uri, load_dotenv, run, submission_name, submission_prefix


def upload_submission(path: Path, *, agent: str, name: str, delete: bool = True) -> str:
    if not (path / "manifest.json").exists() or not (path / "serve.py").exists():
        raise SystemExit(f"submission must contain manifest.json and serve.py: {path}")
    prefix = submission_prefix(agent, name)
    dest = hf_bucket_uri(agent, prefix)
    command = ["hf", "buckets", "sync", str(path), dest]
    if delete:
        command.append("--delete")
    run(command)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a submission directory to the senpai HF bucket.")
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--name", default=None, help="Submission name under submissions/<agent-id>/")
    parser.add_argument("--path", default="submissions/vllm_baseline")
    parser.add_argument("--no-delete", action="store_true", help="Do not delete remote files missing locally.")
    args = parser.parse_args()

    load_dotenv()
    path = (ROOT / args.path).resolve()
    agent = agent_id(args.agent_id)
    name = submission_name(path, args.name)
    dest = upload_submission(path, agent=agent, name=name, delete=not args.no_delete)
    print(dest)


if __name__ == "__main__":
    main()
