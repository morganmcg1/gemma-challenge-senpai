#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.common import ROOT, agent_id, hf, hf_bucket_uri, load_dotenv, run, submission_name, submission_prefix
from scripts.wandb_logging import finish_wandb, init_wandb_run, log_event, log_file_artifact


REMOTE_SUBMISSION_DIR = Path("/submission")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _model_reference_errors(label: str, value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, str) or not value.strip():
        return [f"{label} must be a non-empty string when set."]

    model_id = value.strip()
    model_path = Path(model_id)
    if model_path.is_absolute() and not _is_relative_to(model_path, REMOTE_SUBMISSION_DIR):
        return [
            (
                f"{label} points at local absolute path {model_id!r}. "
                "HF Jobs upload only the submission directory; use a Hub model id, "
                "or package the artifact inside the submission and reference it with a relative path."
            )
        ]
    if not model_path.is_absolute() and ".." in model_path.parts:
        return [
            (
                f"{label} escapes the submission directory with {model_id!r}. "
                "Package model artifacts inside the submission directory or publish them as a Hub model id."
            )
        ]
    return []


def validate_submission_package(path: Path) -> dict:
    manifest_path = path / "manifest.json"
    serve_path = path / "serve.py"
    errors = []

    if not manifest_path.exists():
        errors.append(f"missing manifest.json: {manifest_path}")
    if not serve_path.exists():
        errors.append(f"missing serve.py: {serve_path}")
    if errors:
        raise SystemExit("invalid submission package:\n- " + "\n- ".join(errors))

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid manifest JSON: {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise SystemExit(f"manifest must be a JSON object: {manifest_path}")

    errors.extend(_model_reference_errors("manifest.model_id", manifest.get("model_id")))
    env = manifest.get("env") or {}
    if not isinstance(env, dict):
        errors.append("manifest.env must be an object when set.")
    else:
        errors.extend(_model_reference_errors("manifest.env.MODEL_ID", env.get("MODEL_ID")))

    if errors:
        raise SystemExit("submission is not remote-loadable:\n- " + "\n- ".join(errors))
    return manifest


def upload_submission(path: Path, *, agent: str, name: str, delete: bool = True) -> str:
    validate_submission_package(path)
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
