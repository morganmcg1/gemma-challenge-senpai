#!/usr/bin/env python3
"""Locally pre-validate a Gemma-challenge submission on the assigned A10G.

This is the LOCAL, exploratory counterpart to the official HF Jobs harness
(``official/main_bucket/shared_resources/speed_benchmark/scripts/hf_bucket_single_job.py``).
It never launches an HF Job and never touches the org-credit ``/v1/jobs:run``
path. It exists so a submission can be sanity-checked end to end on the local
GPU *before* spending HF Jobs quota:

* stand up the submission endpoint (``serve.py`` with the manifest ``env``),
* prove the vLLM-compatible PPL contract (integer-token prompt +
  ``prompt_logprobs`` + ``add_special_tokens: false``) by running the official
  ``ppl_endpoint.py`` scorer and producing a ``ppl_summary.json``,
* prove the decode/token-ID contract (``return_token_ids: true`` ->
  ``choices[0].token_ids``) by running the official ``decode_outputs.py`` and
  producing ``decode_summary.json``,
* print ``tps`` / ``ppl`` / ``completed`` and write a combined
  ``local_summary.json``.

The reported ``tps`` is a single-stream *local* decode-throughput proxy
(``decode num_completion_tokens / duration_s``); it is exploratory only and is
NOT the official a10g-small leaderboard number. PPL and the contract checks are
hardware-independent and DO carry over to the official run.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "official/main_bucket/shared_resources/speed_benchmark"
PPL_SCRIPT = HARNESS / "scripts/ppl_endpoint.py"
DECODE_SCRIPT = HARNESS / "scripts/decode_outputs.py"
PPL_DATASET = HARNESS / "data/ppl_ground_truth_tokens.jsonl"
EVAL_DATASET = HARNESS / "data/eval_prompts_sharegpt.json"

DEFAULT_VENV_PYTHON = "/tmp/server-venv/bin/python"
DEFAULT_MODEL_ID = "google/gemma-4-E4B-it"
DEFAULT_SERVED_MODEL_NAME = "gemma-4-e4b-it"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submission", default="submissions/vllm_baseline",
                        help="submission dir (relative to repo root or absolute)")
    parser.add_argument("--base-url", default=None,
                        help="attach to an already-running endpoint instead of starting serve.py")
    parser.add_argument("--venv-python", default=DEFAULT_VENV_PYTHON,
                        help="python that has the submission runtime deps (vllm, transformers)")
    parser.add_argument("--create-venv", action="store_true",
                        help="create the server venv with uv and install manifest deps if missing")
    parser.add_argument("--python", default="3.12", help="python version for --create-venv")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--startup-timeout-s", type=int, default=900)
    parser.add_argument("--output-dir", default=None,
                        help="where to write artifacts (default: research/local_validation/<submission>)")
    parser.add_argument("--server-log", default="/tmp/local_prevalidate_serve.log")
    parser.add_argument("--ppl-records", type=int, default=0,
                        help="limit PPL to the first N records (0 = all 128, the validity gate)")
    parser.add_argument("--decode-num-prompts", type=int, default=16,
                        help="prompts for the decode/TPS pass (128 = full audit set)")
    parser.add_argument("--decode-output-len", type=int, default=512)
    parser.add_argument("--no-ppl", action="store_true")
    parser.add_argument("--no-decode", action="store_true")
    parser.add_argument("--request-timeout-s", type=int, default=180)
    parser.add_argument("--wandb-name", default=None,
                        help="log the local pre-validation summary to W&B under this run name")
    parser.add_argument("--wandb-group", default=None,
                        help="experiment group tag (e.g. int4-channel-lmhead-sweep)")
    parser.add_argument("--wandb-project", default=None,
                        help="W&B project (default: canonical wandb-applied-ai-team/gemma-challenge-senpai)")
    return parser.parse_args()


def _maybe_log_wandb(args: argparse.Namespace, summary: dict[str, Any]) -> str | None:
    """Best-effort W&B log of the local pre-validation summary; no-op without creds."""
    if not getattr(args, "wandb_name", None):
        return None
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_summary
    except Exception as exc:  # pragma: no cover - logging must never break validation
        print(f"[prevalidate] wandb logging unavailable: {exc}", flush=True)
        return None
    run = init_wandb_run(
        job_type="local-prevalidate",
        agent="senpai",
        name=args.wandb_name,
        project=args.wandb_project,
        tags=["local-prevalidate", *([args.wandb_group] if args.wandb_group else [])],
        config={
            "submission": summary.get("submission"),
            "served_model_name": summary.get("served_model_name"),
            "decode_num_prompts": args.decode_num_prompts,
            "decode_output_len": args.decode_output_len,
            "ppl_records": args.ppl_records,
            "wandb_group": args.wandb_group,
            "local_exploratory": True,
        },
    )
    if run is None:
        return None
    log_summary(run, summary, step=0)
    run_id = getattr(run, "id", None)
    print(f"[prevalidate] wandb run id={run_id}", flush=True)
    finish_wandb(run)
    return run_id


def load_manifest(submission_dir: Path) -> dict[str, Any]:
    data = json.loads((submission_dir / "manifest.json").read_text())
    if not isinstance(data, dict) or not data.get("serve"):
        raise ValueError(f"invalid manifest in {submission_dir}")
    data.setdefault("dependencies", [])
    return data


def ensure_venv(venv_python: Path, manifest: dict[str, Any], python_version: str) -> None:
    if venv_python.exists():
        return
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv not found; cannot create venv. Pre-create it or install uv.")
    venv_dir = venv_python.parent.parent
    print(f"[prevalidate] creating venv {venv_dir} (python {python_version})", flush=True)
    subprocess.run([uv, "venv", str(venv_dir), "--python", python_version], check=True)
    deps = manifest.get("dependencies") or []
    if deps:
        print(f"[prevalidate] installing manifest deps: {deps}", flush=True)
        subprocess.run([uv, "pip", "install", "--python", str(venv_python), *deps], check=True)


def server_env(manifest: dict[str, Any], submission_dir: Path, venv_python: Path, port: int) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in (manifest.get("env") or {}).items():
        env[str(key)] = str(value)
    venv_bin = str(venv_python.parent)
    env["VIRTUAL_ENV"] = str(venv_python.parent.parent)
    env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("MODEL_ID", str(manifest.get("model_id", DEFAULT_MODEL_ID)))
    env.setdefault("SERVED_MODEL_NAME", str(manifest.get("served_model_name", DEFAULT_SERVED_MODEL_NAME)))
    env.setdefault("HOST", "0.0.0.0")
    env["PORT"] = str(port)
    return env


def build_serve_cmd(manifest: dict[str, Any], venv_python: Path) -> list[str]:
    cmd = list(manifest["serve"])
    if cmd and cmd[0] in {"python", "python3"}:
        cmd[0] = str(venv_python)
    return cmd


def wait_for_models(base_url: str, timeout_s: int, proc: subprocess.Popen | None) -> None:
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited before readiness (code {proc.returncode}); see server log")
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5.0) as resp:
                if resp.status == 200:
                    return
                last = f"status={resp.status}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = str(exc)
        time.sleep(5)
    raise RuntimeError(f"endpoint not ready at {base_url}/v1/models: {last}")


def run_script(venv_python: Path, script: Path, args: list[str]) -> int:
    cmd = [str(venv_python), str(script), *args]
    print("[prevalidate] running:", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=False).returncode


def truncate_ppl_dataset(limit: int, dest: Path) -> Path:
    lines = [l for l in PPL_DATASET.read_text().splitlines() if l.strip()][:limit]
    dest.write_text("\n".join(lines) + "\n")
    return dest


def terminate(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()


def main() -> int:
    args = parse_args()
    submission_dir = Path(args.submission)
    if not submission_dir.is_absolute():
        submission_dir = (ROOT / submission_dir).resolve()
    manifest = load_manifest(submission_dir)
    served_model_name = str(manifest.get("served_model_name", DEFAULT_SERVED_MODEL_NAME))
    venv_python = Path(args.venv_python)

    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "research/local_validation" / submission_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    base_url = args.base_url or f"http://{args.host}:{args.port}"
    server_proc: subprocess.Popen | None = None

    if args.base_url is None:
        if args.create_venv:
            ensure_venv(venv_python, manifest, args.python)
        if not venv_python.exists():
            raise FileNotFoundError(f"venv python not found: {venv_python} (use --create-venv or --base-url)")
        env = server_env(manifest, submission_dir, venv_python, args.port)
        serve_cmd = build_serve_cmd(manifest, venv_python)
        log = open(args.server_log, "w")
        print(f"[prevalidate] starting server: {' '.join(serve_cmd)} (log: {args.server_log})", flush=True)
        server_proc = subprocess.Popen(
            serve_cmd, cwd=submission_dir, env=env,
            stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid,
        )

    summary: dict[str, Any] = {
        "submission": str(submission_dir),
        "served_model_name": served_model_name,
        "base_url": base_url,
        "local_exploratory": True,
        "note": "Local A10G numbers are exploratory; only a10g-small HF Jobs runs are official.",
    }
    try:
        print(f"[prevalidate] waiting for {base_url}/v1/models ...", flush=True)
        wait_for_models(base_url, args.startup_timeout_s, server_proc)
        print("[prevalidate] endpoint ready", flush=True)

        if not args.no_ppl:
            ppl_dataset = PPL_DATASET
            if args.ppl_records and args.ppl_records > 0:
                ppl_dataset = truncate_ppl_dataset(args.ppl_records, out_dir / "ppl_subset.jsonl")
            ppl_summary_file = out_dir / "ppl_summary.json"
            rc = run_script(venv_python, PPL_SCRIPT, [
                "--base-url", base_url, "--model", served_model_name,
                "--dataset-path", str(ppl_dataset),
                "--output-file", str(out_dir / "ppl_results.jsonl"),
                "--summary-file", str(ppl_summary_file),
                "--request-timeout-s", str(args.request_timeout_s),
            ])
            if rc != 0:
                raise RuntimeError(f"PPL stage failed (rc={rc}) -- this is the validity gate; see logs above")
            ppl_summary = json.loads(ppl_summary_file.read_text())
            summary["ppl"] = ppl_summary["ppl"]
            summary["ppl_num_records"] = ppl_summary["num_records"]
            summary["ppl_num_tokens"] = ppl_summary["num_tokens"]
            summary["ppl_summary_file"] = str(ppl_summary_file)

        if not args.no_decode:
            decode_summary_file = out_dir / "decode_summary.json"
            rc = run_script(venv_python, DECODE_SCRIPT, [
                "--base-url", base_url, "--model", served_model_name,
                "--dataset-path", str(EVAL_DATASET),
                "--output-file", str(out_dir / "decode_outputs.jsonl"),
                "--summary-file", str(decode_summary_file),
                "--num-prompts", str(args.decode_num_prompts),
                "--output-len", str(args.decode_output_len),
                "--request-timeout-s", str(args.request_timeout_s),
            ])
            if rc != 0:
                raise RuntimeError(f"decode/token-ID contract stage failed (rc={rc}); see logs above")
            decode_summary = json.loads(decode_summary_file.read_text())
            dur = decode_summary.get("duration_s") or 0.0
            toks = decode_summary.get("num_completion_tokens") or 0
            summary["decode_num_records"] = decode_summary["num_records"]
            summary["decode_num_completion_tokens"] = toks
            summary["decode_duration_s"] = dur
            summary["decode_token_id_sources"] = decode_summary.get("token_id_sources", {})
            summary["tps"] = (toks / dur) if dur else 0.0
            summary["decode_summary_file"] = str(decode_summary_file)

        # "completed" tracks the PPL validity-gate completion (all 128 GT records); the
        # decode pass is a separate, configurable TPS sample (reported as decode_num_records).
        summary["completed"] = int(
            summary.get("ppl_num_records")
            or summary.get("decode_num_records")
            or 0
        )

        (out_dir / "local_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
        run_id = _maybe_log_wandb(args, summary)
        if run_id:
            summary["wandb_run_id"] = run_id
            (out_dir / "local_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
        tps = summary.get("tps", 0.0)
        ppl = summary.get("ppl", 0.0)
        print("\n[prevalidate] LOCAL pre-validation summary", flush=True)
        print(f"SENPAI-LOCAL tps={tps:.4f} ppl={ppl:.4f} completed={summary['completed']} "
              f"decode_sample={summary.get('decode_num_records', 0)} "
              f"submission={submission_dir.name} (exploratory; not official a10g-small)", flush=True)
        print(f"[prevalidate] artifacts in {out_dir}", flush=True)
        return 0
    finally:
        terminate(server_proc)


if __name__ == "__main__":
    raise SystemExit(main())
