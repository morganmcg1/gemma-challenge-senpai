#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.common import ROOT, agent_id, hf, hf_bucket_uri, load_dotenv, run, submission_name, submission_prefix
from scripts.wandb_logging import finish_wandb, init_wandb_run, log_event, log_file_artifact


def upload_submission(path: Path, *, agent: str, name: str, delete: bool = True) -> str:
    if not (path / "manifest.json").exists() or not (path / "serve.py").exists():
        raise SystemExit(f"submission must contain manifest.json and serve.py: {path}")
    prefix = submission_prefix(agent, name)
    dest = hf_bucket_uri(agent, prefix)
    command = hf("buckets", "sync", str(path), dest)
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
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--wandb-notes", default="")
    args = parser.parse_args()

    load_dotenv()
    path = (ROOT / args.path).resolve()
    agent = agent_id(args.agent_id)
    name = submission_name(path, args.name)
    wandb_run = init_wandb_run(
        job_type="upload",
        agent=agent,
        name=f"{agent}-{name}-upload",
        notes=args.wandb_notes,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        tags=["submission"],
        config={
            "submission_path": str(path),
            "submission_name": name,
            "delete_remote_missing": not args.no_delete,
        },
    )
    try:
        log_event(wandb_run, "upload_start", step=0, data={"submission_path": str(path)})
        dest = upload_submission(path, agent=agent, name=name, delete=not args.no_delete)
        log_event(
            wandb_run,
            "upload_complete",
            step=1,
            metrics={"submission/uploaded": 1},
            data={"submission_uri": dest},
        )
        log_file_artifact(
            wandb_run,
            path=path / "manifest.json",
            name=f"{agent}-{name}-manifest",
            artifact_type="submission-manifest",
        )
        print(dest)
    finally:
        finish_wandb(wandb_run)


if __name__ == "__main__":
    main()
