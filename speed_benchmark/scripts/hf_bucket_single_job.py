#!/usr/bin/env python
"""Run a participant server and the fixed benchmark in one HF Job.

The submission bucket prefix is mounted at /submission and contains only the
participant-owned files: manifest.json, serve.py, and optional model artifacts.
This job creates two independent virtualenvs:

* /tmp/server-venv for participant dependencies from manifest.json
* /tmp/bench-venv for pinned benchmark dependencies

The benchmark talks to the participant endpoint over localhost, so no tunnel or
job-to-job networking is required.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "google/gemma-4-E4B-it"
DEFAULT_SERVED_MODEL_NAME = "gemma-4-e4b-it"
DEFAULT_PORT = 8000
TOKENIZER = "google/gemma-4-E4B-it"
NUM_PROMPTS = 128
OUTPUT_LEN = 512
MAX_CONCURRENCY = 1
REQUEST_RATE = "inf"
WARMUP_REQUESTS = 4
SEED = 1
BENCHMARK_DEPENDENCIES = [
    "sglang==0.5.2",
    "transformers==5.9.0",
    "jinja2==3.1.6",
    "pybase64==1.4.3",
    "pydantic==2.13.4",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", default="/submission")
    parser.add_argument("--manifest", default="/submission/manifest.json")
    parser.add_argument("--state-dir", default="/state")
    parser.add_argument("--dataset-path", default="/harness/eval_prompts_sharegpt.json")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--startup-timeout-s", type=int, default=900)
    parser.add_argument("--python", default="3.12")
    parser.add_argument("--server-venv", default="/tmp/server-venv")
    parser.add_argument("--bench-venv", default="/tmp/bench-venv")
    parser.add_argument("--output-file", default="/state/benchmark.jsonl")
    parser.add_argument("--summary-file", default="/state/summary.json")
    parser.add_argument(
        "--enable-decode-capture",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="capture generated text and token IDs after the speed benchmark (default: on)",
    )
    parser.add_argument("--decode-script", default="/harness/scripts/decode_outputs.py")
    parser.add_argument("--decode-output-file", default="/state/decode_outputs.jsonl")
    parser.add_argument("--decode-summary-file", default="/state/decode_summary.json")
    parser.add_argument(
        "--enable-ppl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the endpoint PPL stage after the benchmark (default: on; use --no-enable-ppl to skip)",
    )
    parser.add_argument("--ppl-script", default="/harness/scripts/ppl_endpoint.py")
    parser.add_argument("--ppl-dataset-path", default="/harness/data/ppl_ground_truth_tokens.jsonl")
    parser.add_argument("--ppl-output-file", default="/state/ppl_results.jsonl")
    parser.add_argument("--ppl-summary-file", default="/state/ppl_summary.json")
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")
    if not data.get("serve"):
        raise ValueError("manifest must define a non-empty 'serve' command list")
    if not isinstance(data["serve"], list) or not all(isinstance(x, str) for x in data["serve"]):
        raise ValueError("manifest 'serve' must be a list of strings")
    dependencies = data.get("dependencies", [])
    if dependencies is None:
        dependencies = []
    if not isinstance(dependencies, list) or not all(isinstance(x, str) for x in dependencies):
        raise ValueError("manifest 'dependencies' must be a list of strings")
    data["dependencies"] = dependencies
    return data


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def create_venv(path: Path, python: str, dependencies: list[str]) -> Path:
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is required in the HF Job image")

    run([uv, "venv", str(path), "--python", python])
    python_path = path / "bin/python"
    if dependencies:
        run([uv, "pip", "install", "--python", str(python_path), *dependencies])
    return python_path


def resolve_model_id(value: str, submission_dir: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    candidate = submission_dir / value
    if candidate.exists():
        return str(candidate)
    return value


def participant_env(
    manifest: dict[str, Any],
    submission_dir: Path,
    server_venv: Path,
    port: int,
) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in (manifest.get("env") or {}).items():
        env[str(key)] = str(value)

    env["VIRTUAL_ENV"] = str(server_venv)
    env["PATH"] = f"{server_venv / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault(
        "MODEL_ID",
        resolve_model_id(str(manifest.get("model_id", DEFAULT_MODEL_ID)), submission_dir),
    )
    env.setdefault("SERVED_MODEL_NAME", str(manifest.get("served_model_name", DEFAULT_SERVED_MODEL_NAME)))
    env.setdefault("HOST", "0.0.0.0")
    env["PORT"] = str(port)
    return env


def benchmark_env(bench_venv: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(bench_venv)
    env["PATH"] = f"{bench_venv / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def build_serve_command(command: list[str], submission_dir: Path, server_python: Path) -> list[str]:
    cmd = list(command)
    if cmd[0] in {"python", "python3"}:
        cmd[0] = str(server_python)
    elif cmd[0].endswith(".py"):
        script = Path(cmd[0])
        if not script.is_absolute():
            script = submission_dir / script
        cmd = [str(server_python), str(script), *cmd[1:]]
    else:
        executable = Path(cmd[0])
        if not executable.is_absolute() and (submission_dir / executable).exists():
            cmd[0] = str(submission_dir / executable)
    return cmd


def stream_output(proc: subprocess.Popen[str]) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"[server] {line.rstrip()}", flush=True)


def wait_for_models(base_url: str, timeout_s: int, proc: subprocess.Popen[str] | None = None) -> None:
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited before readiness with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5.0) as response:
                if response.status == 200:
                    return
                body = response.read(200).decode("utf-8", "replace")
                last_error = f"status={response.status} body={body}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        time.sleep(5)
    raise RuntimeError(f"endpoint did not become ready at {base_url}/v1/models: {last_error}")


def run_benchmark(
    bench_python: Path,
    bench_env: dict[str, str],
    *,
    base_url: str,
    model: str,
    dataset_path: Path,
    output_file: Path,
) -> int:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(bench_python),
        "-m",
        "sglang.bench_serving",
        "--backend",
        "vllm-chat",
        "--base-url",
        base_url.rstrip("/"),
        "--model",
        model,
        "--tokenizer",
        TOKENIZER,
        "--dataset-name",
        "sharegpt",
        "--dataset-path",
        str(dataset_path),
        "--sharegpt-output-len",
        str(OUTPUT_LEN),
        "--num-prompts",
        str(NUM_PROMPTS),
        "--max-concurrency",
        str(MAX_CONCURRENCY),
        "--request-rate",
        REQUEST_RATE,
        "--warmup-requests",
        str(WARMUP_REQUESTS),
        "--seed",
        str(SEED),
        "--extra-request-body",
        json.dumps({"ignore_eos": True}),
        "--output-file",
        str(output_file),
        "--output-details",
        "--disable-stream",
        "--disable-tqdm",
    ]
    print("Running:", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=False, env=bench_env).returncode


def run_decode_capture(
    bench_python: Path,
    bench_env: dict[str, str],
    *,
    decode_script: Path,
    base_url: str,
    model: str,
    dataset_path: Path,
    output_file: Path,
    summary_file: Path,
) -> int:
    if not decode_script.exists():
        raise FileNotFoundError(f"decode capture script not found: {decode_script}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"benchmark dataset not found: {dataset_path}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(bench_python),
        str(decode_script),
        "--base-url",
        base_url.rstrip("/"),
        "--model",
        model,
        "--dataset-path",
        str(dataset_path),
        "--output-file",
        str(output_file),
        "--summary-file",
        str(summary_file),
        "--tokenizer",
        TOKENIZER,
        "--num-prompts",
        str(NUM_PROMPTS),
        "--output-len",
        str(OUTPUT_LEN),
        "--seed",
        str(SEED),
    ]
    print("Running:", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=False, env=bench_env).returncode


def run_ppl(
    bench_python: Path,
    bench_env: dict[str, str],
    *,
    ppl_script: Path,
    base_url: str,
    model: str,
    dataset_path: Path,
    output_file: Path,
    summary_file: Path,
) -> int:
    if not ppl_script.exists():
        raise FileNotFoundError(f"PPL script not found: {ppl_script}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"PPL dataset not found: {dataset_path}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(bench_python),
        str(ppl_script),
        "--base-url",
        base_url.rstrip("/"),
        "--model",
        model,
        "--dataset-path",
        str(dataset_path),
        "--output-file",
        str(output_file),
        "--summary-file",
        str(summary_file),
    ]
    print("Running:", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=False, env=bench_env).returncode


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def write_summary(
    summary_file: Path,
    result: dict[str, Any],
    *,
    base_url: str,
    model: str,
    output_file: Path,
    server_dependencies: list[str],
) -> dict[str, Any]:
    total_tps = (result["total_input_tokens"] + result["total_output_tokens"]) / result["duration"]
    summary = {
        "tps": result["output_throughput"],
        "output_tps": result["output_throughput"],
        "total_tps": total_tps,
        "completed": result["completed"],
        "duration_s": result["duration"],
        "request_throughput_req_s": result["request_throughput"],
        "mean_e2e_latency_ms": result["mean_e2e_latency_ms"],
        "p99_e2e_latency_ms": result["p99_e2e_latency_ms"],
        "max_concurrency": result["max_concurrency"],
        "num_prompts": NUM_PROMPTS,
        "output_len": OUTPUT_LEN,
        "model": model,
        "base_url": base_url,
        "benchmark_jsonl": str(output_file),
        "benchmark_dependencies": BENCHMARK_DEPENDENCIES,
        "server_dependencies": server_dependencies,
        "job_id": os.environ.get("JOB_ID"),
    }
    write_json(summary_file, summary)
    return summary


def terminate(proc: subprocess.Popen[str] | None) -> None:
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
    submission_dir = Path(args.submission_dir)
    state_dir = Path(args.state_dir)
    dataset_path = Path(args.dataset_path)
    manifest = load_manifest(Path(args.manifest))
    port = args.port or int(manifest.get("port", DEFAULT_PORT))
    base_url = f"http://127.0.0.1:{port}"

    if not dataset_path.exists():
        raise FileNotFoundError(f"benchmark dataset not found: {dataset_path}")

    server_venv = Path(args.server_venv)
    bench_venv = Path(args.bench_venv)
    print(f"Creating participant server venv at {server_venv}", flush=True)
    server_python = create_venv(server_venv, args.python, manifest["dependencies"])
    print(f"Creating pinned benchmark venv at {bench_venv}", flush=True)
    bench_python = create_venv(bench_venv, args.python, BENCHMARK_DEPENDENCIES)

    server_env = participant_env(manifest, submission_dir, server_venv, port)
    serve_cmd = build_serve_command(manifest["serve"], submission_dir, server_python)
    bench_env = benchmark_env(bench_venv)
    output_file = Path(args.output_file)
    summary_file = Path(args.summary_file)
    decode_script = Path(args.decode_script)
    decode_output_file = Path(args.decode_output_file)
    decode_summary_file = Path(args.decode_summary_file)
    ppl_script = Path(args.ppl_script)
    ppl_dataset_path = Path(args.ppl_dataset_path)
    ppl_output_file = Path(args.ppl_output_file)
    ppl_summary_file = Path(args.ppl_summary_file)

    write_json(
        state_dir / "run_environment.json",
        {
            "base_url": base_url,
            "benchmark_dependencies": BENCHMARK_DEPENDENCIES,
            "decode_capture": {
                "enabled": args.enable_decode_capture,
                "script": str(decode_script),
                "output_file": str(decode_output_file),
                "summary_file": str(decode_summary_file),
                "requires": {
                    "request": "return_token_ids: true on /v1/completions",
                    "response": "choices[0].token_ids",
                },
            },
            "ppl": {
                "enabled": args.enable_ppl,
                "script": str(ppl_script),
                "dataset_path": str(ppl_dataset_path),
                "output_file": str(ppl_output_file),
                "summary_file": str(ppl_summary_file),
            },
            "server_dependencies": manifest["dependencies"],
            "server_venv": str(server_venv),
            "bench_venv": str(bench_venv),
            "manifest": manifest,
        },
    )

    server_proc: subprocess.Popen[str] | None = None

    def handle_signal(signum: int, _frame: object) -> None:
        print(f"Received signal {signum}; stopping server", flush=True)
        terminate(server_proc)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        print("Starting participant server:", " ".join(serve_cmd), flush=True)
        server_proc = subprocess.Popen(
            serve_cmd,
            cwd=submission_dir,
            env=server_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )
        threading.Thread(target=stream_output, args=(server_proc,), daemon=True).start()
        wait_for_models(base_url, args.startup_timeout_s, server_proc)
        write_json(
            state_dir / "server.json",
            {
                "base_url": base_url,
                "port": port,
                "served_model_name": server_env["SERVED_MODEL_NAME"],
                "job_id": os.environ.get("JOB_ID"),
                "ready_at_unix": time.time(),
            },
        )
        print(f"Server ready at {base_url}", flush=True)

        rc = run_benchmark(
            bench_python,
            bench_env,
            base_url=base_url,
            model=server_env["SERVED_MODEL_NAME"],
            dataset_path=dataset_path,
            output_file=output_file,
        )
        if rc != 0:
            return rc

        result = json.loads(output_file.read_text().strip().splitlines()[-1])
        summary = write_summary(
            summary_file,
            result,
            base_url=base_url,
            model=server_env["SERVED_MODEL_NAME"],
            output_file=output_file,
            server_dependencies=manifest["dependencies"],
        )
        print("\nSummary", flush=True)
        print(f"TPS={summary['tps']:.4f}", flush=True)
        print(f"total_tps={summary['total_tps']:.4f}", flush=True)
        print(f"completed={summary['completed']}", flush=True)

        if args.enable_decode_capture:
            rc = run_decode_capture(
                bench_python,
                bench_env,
                decode_script=decode_script,
                base_url=base_url,
                model=server_env["SERVED_MODEL_NAME"],
                dataset_path=dataset_path,
                output_file=decode_output_file,
                summary_file=decode_summary_file,
            )
            if rc != 0:
                return rc
            decode_summary = json.loads(decode_summary_file.read_text())
            summary["decode_outputs_file"] = str(decode_output_file)
            summary["decode_summary_file"] = str(decode_summary_file)
            summary["decode_num_records"] = decode_summary["num_records"]
            summary["decode_num_completion_tokens"] = decode_summary["num_completion_tokens"]
            summary["decode_token_ids_required"] = True
            write_json(summary_file, summary)
            print(f"decode_records={summary['decode_num_records']}", flush=True)
            print(f"decode_completion_tokens={summary['decode_num_completion_tokens']}", flush=True)

        if args.enable_ppl:
            rc = run_ppl(
                bench_python,
                bench_env,
                ppl_script=ppl_script,
                base_url=base_url,
                model=server_env["SERVED_MODEL_NAME"],
                dataset_path=ppl_dataset_path,
                output_file=ppl_output_file,
                summary_file=ppl_summary_file,
            )
            if rc != 0:
                return rc
            ppl_summary = json.loads(ppl_summary_file.read_text())
            summary["ppl"] = ppl_summary["ppl"]
            summary["ppl_num_tokens"] = ppl_summary["num_tokens"]
            summary["ppl_summary_file"] = str(ppl_summary_file)
            summary["ppl_results_file"] = str(ppl_output_file)
            write_json(summary_file, summary)
            print(f"PPL={summary['ppl']:.4f}", flush=True)

        print(f"summary_file={summary_file}", flush=True)
        return 0
    finally:
        terminate(server_proc)


if __name__ == "__main__":
    raise SystemExit(main())
