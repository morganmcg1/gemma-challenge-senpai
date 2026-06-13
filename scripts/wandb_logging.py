from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from scripts.common import ROOT


STATUS_VALUE = {
    "missing": 0,
    "queued": 1,
    "pending": 1,
    "starting": 2,
    "running": 3,
    "completed": 4,
    "complete": 4,
    "error": -1,
    "failed": -1,
    "timed_out": -2,
    "cancelled": -3,
}


def safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "-" for char in value)
    return cleaned.strip("-") or "gemma-run"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _numeric(value: Any) -> float | int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return value
    return None


def flatten_numeric(prefix: str, data: dict[str, Any]) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    for key, value in data.items():
        name = f"{prefix}/{key}"
        numeric = _numeric(value)
        if numeric is not None:
            metrics[name] = numeric
        elif isinstance(value, dict):
            metrics.update(flatten_numeric(name, value))
    return metrics


def git_info() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        ).stdout.strip()

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "git_dirty": bool(git("status", "--porcelain")),
    }


def init_wandb_run(
    *,
    job_type: str,
    agent: str,
    name: str,
    notes: str = "",
    project: str | None = None,
    entity: str | None = None,
    mode: str | None = None,
    tags: list[str] | None = None,
    config: dict[str, Any] | None = None,
):
    if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        return None

    resolved_mode = mode or os.environ.get("WANDB_MODE")
    if resolved_mode and resolved_mode.lower() == "disabled":
        return None
    if not resolved_mode and not os.environ.get("WANDB_API_KEY"):
        return None

    try:
        import wandb
    except ImportError:
        return None

    run_config = {
        "agent_id": agent,
        "notes": notes,
        "repo_root": str(ROOT),
        **git_info(),
        **(config or {}),
    }

    run = wandb.init(
        project=project or os.environ.get("WANDB_PROJECT") or "gemma-challenge-senpai",
        entity=entity or os.environ.get("WANDB_ENTITY") or None,
        name=name,
        group=agent,
        job_type=job_type,
        tags=["gemma-challenge", job_type, *(tags or [])],
        notes=notes or None,
        config=_jsonable(run_config),
        mode=resolved_mode,
    )
    run.define_metric("global_step")
    run.define_metric("*", step_metric="global_step")
    return run


def log_event(
    run: Any,
    event: str,
    *,
    step: int,
    metrics: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> None:
    if run is None:
        return

    payload = {
        "global_step": step,
        "event/name": event,
        **(metrics or {}),
    }
    if data:
        payload.update({f"event/{key}": _jsonable(value) for key, value in data.items()})
    run.log(payload)


def log_status(run: Any, status: dict[str, Any], *, step: int, elapsed_s: float, attempt: int) -> None:
    state = str(status.get("status") or status.get("stage") or "unknown")
    metrics = {
        "poll/attempt": attempt,
        "poll/elapsed_s": elapsed_s,
        "job/status_code": STATUS_VALUE.get(state, -9),
    }
    metrics.update(flatten_numeric("job_status", status))
    log_event(run, "poll_status", step=step, metrics=metrics, data={"job_status": state})


def log_summary(run: Any, summary: dict[str, Any], *, step: int, run_prefix: str | None = None) -> None:
    if run is None:
        return

    metrics = flatten_numeric("summary", summary)
    log_event(run, "summary", step=step, metrics=metrics, data={"run_prefix": run_prefix or ""})
    for key, value in summary.items():
        if isinstance(value, (dict, list)):
            continue
        run.summary[f"summary/{key}"] = _jsonable(value)
    if run_prefix:
        run.summary["run_prefix"] = run_prefix


def log_json_artifact(run: Any, *, name: str, artifact_type: str, data: dict[str, Any]) -> None:
    if run is None:
        return

    import wandb

    artifact = wandb.Artifact(safe_name(name), type=artifact_type)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
        json.dump(_jsonable(data), handle, indent=2, sort_keys=True)
        path = Path(handle.name)
    try:
        artifact.add_file(str(path), name=f"{safe_name(name)}.json")
        run.log_artifact(artifact)
    finally:
        path.unlink(missing_ok=True)


def log_file_artifact(run: Any, *, path: Path, name: str, artifact_type: str) -> None:
    if run is None or not path.exists():
        return

    import wandb

    artifact = wandb.Artifact(safe_name(name), type=artifact_type)
    artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact)


def finish_wandb(run: Any) -> None:
    if run is not None:
        run.finish()
