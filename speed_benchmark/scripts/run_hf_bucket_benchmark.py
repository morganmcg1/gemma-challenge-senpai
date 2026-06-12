#!/usr/bin/env python
"""Launch a bucket-backed single HF Job for the Gemma endpoint benchmark."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import (
    HfApi,
    JobStage,
    Volume,
    batch_bucket_files,
    download_bucket_files,
    get_token,
    run_job,
)


DEFAULT_BUCKET = "gemma-challenge/gemma-main-bucket"
DEFAULT_HARNESS_PREFIX = "shared_resources/speed_benchmark"
DEFAULT_FLAVOR = "a10g-small"
DEFAULT_IMAGE = "vllm/vllm-openai"
DEFAULT_TIMEOUT = "4h"
ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
SINGLE_JOB_SCRIPT = Path(__file__).resolve().parent / "hf_bucket_single_job.py"
PPL_SCRIPT = Path(__file__).resolve().parent / "ppl_endpoint.py"
DECODE_SCRIPT = Path(__file__).resolve().parent / "decode_outputs.py"
DATASET_PATH = ARTIFACT_ROOT / "data/eval_prompts_sharegpt.json"
PPL_DATASET_PATH = ARTIFACT_ROOT / "data/ppl_ground_truth_tokens.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=os.environ.get("HF_BUCKET", DEFAULT_BUCKET))
    parser.add_argument("--submission-bucket", default=os.environ.get("HF_SUBMISSION_BUCKET"))
    parser.add_argument("--harness-bucket", default=os.environ.get("HF_HARNESS_BUCKET"))
    parser.add_argument("--run-bucket", default=os.environ.get("HF_RUN_BUCKET"))
    parser.add_argument("--submission-prefix", required=True)
    parser.add_argument("--manifest-name", default="manifest.json")
    parser.add_argument("--harness-prefix", default=DEFAULT_HARNESS_PREFIX)
    parser.add_argument("--run-prefix")
    parser.add_argument("--namespace", default=os.environ.get("HF_NAMESPACE"))
    parser.add_argument("--flavor", "--server-flavor", dest="flavor", default=DEFAULT_FLAVOR)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--timeout", default=DEFAULT_TIMEOUT)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout-s", type=int, default=900)
    parser.add_argument("--python", default="3.12")
    parser.add_argument(
        "--enable-ppl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the endpoint PPL stage after the benchmark (default: on; use --no-enable-ppl to skip)",
    )
    parser.add_argument("--ppl-dataset-path", default="/harness/data/ppl_ground_truth_tokens.jsonl")
    parser.add_argument("--ppl-output-file", default="/state/ppl_results.jsonl")
    parser.add_argument("--ppl-summary-file", default="/state/ppl_summary.json")
    parser.add_argument("--sync-harness", action="store_true")
    parser.add_argument("--skip-harness-sync", action="store_true")
    parser.add_argument("--wait", action="store_true")
    return parser.parse_args()


def clean_prefix(prefix: str) -> str:
    return prefix.strip("/")


def join_prefix(*parts: str) -> str:
    return "/".join(clean_prefix(part) for part in parts if clean_prefix(part))


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slug_from_prefix(prefix: str) -> str:
    slug = clean_prefix(prefix).replace("/", "-").replace("_", "-")
    return "".join(ch for ch in slug if ch.isalnum() or ch == "-")[:80] or "submission"


def load_manifest(bucket: str, submission_prefix: str, manifest_name: str) -> dict[str, Any]:
    remote_path = join_prefix(submission_prefix, manifest_name)
    with tempfile.TemporaryDirectory() as tmp:
        local_path = Path(tmp) / manifest_name
        download_bucket_files(bucket, [(remote_path, local_path)])
        manifest = json.loads(local_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest must be a JSON object: hf://buckets/{bucket}/{remote_path}")
    if not manifest.get("serve"):
        raise ValueError("manifest must define a non-empty 'serve' command list")
    if not isinstance(manifest["serve"], list) or not all(isinstance(x, str) for x in manifest["serve"]):
        raise ValueError("manifest 'serve' must be a list of strings")
    dependencies = manifest.get("dependencies", [])
    if dependencies is None:
        dependencies = []
    if not isinstance(dependencies, list) or not all(isinstance(x, str) for x in dependencies):
        raise ValueError("manifest 'dependencies' must be a list of strings")
    return manifest


def sync_harness(bucket: str, harness_prefix: str) -> None:
    files = [
        (DATASET_PATH, join_prefix(harness_prefix, "data/eval_prompts_sharegpt.json")),
        (PPL_DATASET_PATH, join_prefix(harness_prefix, "data/ppl_ground_truth_tokens.jsonl")),
        (SINGLE_JOB_SCRIPT, join_prefix(harness_prefix, "scripts/hf_bucket_single_job.py")),
        (DECODE_SCRIPT, join_prefix(harness_prefix, "scripts/decode_outputs.py")),
        (PPL_SCRIPT, join_prefix(harness_prefix, "scripts/ppl_endpoint.py")),
    ]
    missing = [str(local) for local, _ in files if not local.exists()]
    if missing:
        raise FileNotFoundError(f"missing harness files: {', '.join(missing)}")
    batch_bucket_files(bucket, add=[(local, remote) for local, remote in files])


def write_run_request(bucket: str, run_prefix: str, payload: dict[str, Any]) -> None:
    batch_bucket_files(
        bucket,
        add=[
            (
                json.dumps(payload, indent=2, sort_keys=True).encode(),
                join_prefix(run_prefix, "run_request.json"),
            )
        ],
    )


def token_secret() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or get_token()
    return {"HF_TOKEN": token} if token else {}


def stage_value(stage: Any) -> str:
    return getattr(stage, "value", str(stage))


def wait_for_job(api: HfApi, job_id: str, namespace: str | None) -> str:
    terminal = {
        JobStage.COMPLETED.value,
        JobStage.CANCELED.value,
        JobStage.ERROR.value,
        JobStage.DELETED.value,
    }
    while True:
        info = api.inspect_job(job_id=job_id, namespace=namespace)
        stage = stage_value(info.status.stage)
        print(f"job {job_id}: {stage}", flush=True)
        if stage in terminal:
            return stage
        time.sleep(30)


def maybe_print_summary(bucket: str, run_prefix: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        local_path = Path(tmp) / "summary.json"
        try:
            download_bucket_files(
                bucket,
                [(join_prefix(run_prefix, "summary.json"), local_path)],
                raise_on_missing_files=True,
            )
        except Exception:
            return
        print(local_path.read_text(), flush=True)


def main() -> int:
    args = parse_args()
    default_bucket = args.bucket
    submission_bucket = args.submission_bucket or default_bucket
    harness_bucket = args.harness_bucket or default_bucket
    run_bucket = args.run_bucket or submission_bucket
    submission_prefix = clean_prefix(args.submission_prefix)
    harness_prefix = clean_prefix(args.harness_prefix)
    run_prefix = clean_prefix(args.run_prefix or f"runs/{slug_from_prefix(submission_prefix)}-{timestamp()}")
    manifest_path = f"/submission/{args.manifest_name}"
    manifest = load_manifest(submission_bucket, submission_prefix, args.manifest_name)

    # The job runs the harness script mounted from the central bucket at /harness, so
    # local edits only take effect once pushed there. Sync by default; maintainers with
    # write access keep the canonical copy current. Participants (read-only on the central
    # bucket) are skipped gracefully -- the job just uses the existing canonical harness.
    if not args.skip_harness_sync:
        print(f"Syncing fixed harness to hf://buckets/{harness_bucket}/{harness_prefix}", flush=True)
        try:
            sync_harness(harness_bucket, harness_prefix)
        except Exception as exc:
            print(
                f"Skipping harness sync (no write access to {harness_bucket}?): {exc}",
                flush=True,
            )

    write_run_request(
        run_bucket,
        run_prefix,
        {
            "default_bucket": default_bucket,
            "submission_bucket": submission_bucket,
            "submission_prefix": submission_prefix,
            "manifest_name": args.manifest_name,
            "harness_bucket": harness_bucket,
            "harness_prefix": harness_prefix,
            "run_bucket": run_bucket,
            "run_prefix": run_prefix,
            "flavor": args.flavor,
            "port": args.port,
            "decode_capture": {
                "enabled": True,
                "script": "/harness/scripts/decode_outputs.py",
                "output_file": "/state/decode_outputs.jsonl",
                "summary_file": "/state/decode_summary.json",
                "requires": {
                    "request": "return_token_ids: true on /v1/completions",
                    "response": "choices[0].token_ids",
                },
            },
            "ppl": {
                "enabled": args.enable_ppl,
                "dataset_path": args.ppl_dataset_path,
                "output_file": args.ppl_output_file,
                "summary_file": args.ppl_summary_file,
            },
            "created_at": timestamp(),
            "manifest": manifest,
        },
    )

    volumes = [
        Volume(
            type="bucket",
            source=submission_bucket,
            path=submission_prefix,
            mount_path="/submission",
            read_only=True,
        ),
        Volume(
            type="bucket",
            source=harness_bucket,
            path=harness_prefix,
            mount_path="/harness",
            read_only=True,
        ),
        Volume(
            type="bucket",
            source=run_bucket,
            path=run_prefix,
            mount_path="/state",
            read_only=False,
        ),
    ]

    labels = {
        "task": "gemma-openai-endpoint-benchmark",
        "submission": slug_from_prefix(submission_prefix),
    }

    print("Launching benchmark job", flush=True)
    # Run the harness script mounted read-only at /harness (from the central bucket) via
    # run_job. Unlike run_uv_job, this uploads nothing, so it does NOT create a
    # `{namespace}/jobs-artifacts` bucket and needs no personal-namespace write. The job's
    # data volumes (/submission, /harness, /state) all live in the org bucket already.
    job = run_job(
        image=args.image,
        command=[
            "python3",
            "/harness/scripts/hf_bucket_single_job.py",
            "--submission-dir",
            "/submission",
            "--manifest",
            manifest_path,
            "--state-dir",
            "/state",
            "--dataset-path",
            "/harness/data/eval_prompts_sharegpt.json",
            "--port",
            str(args.port),
            "--startup-timeout-s",
            str(args.startup_timeout_s),
            "--python",
            args.python,
            "--output-file",
            "/state/benchmark.jsonl",
            "--summary-file",
            "/state/summary.json",
            "--enable-decode-capture",
            "--decode-script",
            "/harness/scripts/decode_outputs.py",
            "--decode-output-file",
            "/state/decode_outputs.jsonl",
            "--decode-summary-file",
            "/state/decode_summary.json",
            "--ppl-dataset-path",
            args.ppl_dataset_path,
            "--ppl-output-file",
            args.ppl_output_file,
            "--ppl-summary-file",
            args.ppl_summary_file,
            *(
                ["--enable-ppl", "--ppl-script", "/harness/scripts/ppl_endpoint.py"]
                if args.enable_ppl
                else ["--no-enable-ppl"]
            ),
        ],
        flavor=args.flavor,
        timeout=args.timeout,
        labels=labels,
        volumes=volumes,
        namespace=args.namespace,
        secrets=token_secret(),
    )

    print(f"job_id={job.id}", flush=True)
    print(f"job_url={job.url}", flush=True)
    print(f"run_bucket=hf://buckets/{run_bucket}/{run_prefix}", flush=True)
    print(f"summary=hf://buckets/{run_bucket}/{join_prefix(run_prefix, 'summary.json')}", flush=True)

    if not args.wait:
        return 0

    stage = wait_for_job(HfApi(), job.id, args.namespace)
    maybe_print_summary(run_bucket, run_prefix)
    return 0 if stage == JobStage.COMPLETED.value else 1


if __name__ == "__main__":
    raise SystemExit(main())
