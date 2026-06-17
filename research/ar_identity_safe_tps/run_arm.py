#!/usr/bin/env python
"""Serve int4_g128_lmhead via the canonical api_server config (optionally + one
identity-safe knob), capture the official 128x512 greedy decode, record wall_tps.

Every arm goes through the SAME launcher and differs ONLY by the extra flags/env
passed on the command line, so a variant-vs-ref comparison isolates the knob.
The canonical flags below are exactly what submissions/int4_g128_lmhead/serve.py
execs (it just runs vllm.entrypoints.openai.api_server with these), so an arm
with no extra flags reproduces the shipped reference serve.

  run_arm.py --arm-name ref --out-dir <dir>/ref
  run_arm.py --arm-name fullcg --out-dir <dir>/fullcg \
      --extra-flag --compilation-config --extra-flag '{"cudagraph_mode":"FULL"}'
  run_arm.py --arm-name mnbt1024 --out-dir <dir>/mnbt1024 \
      --extra-env MAX_NUM_BATCHED_TOKENS=1024   # (env only used if a flag reads it)

Run with the server venv python from the repo root:
  /tmp/senpai-venvs/<hash>/bin/python -m research.ar_identity_safe_tps.run_arm ...
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.local_validation import harness, paths  # noqa: E402

MODEL_DIR = ROOT / "submissions" / "int4_g128_lmhead" / "model"


def canonical_serve_cmd(py: str, model: str, port: int, max_num_batched_tokens: str) -> list[str]:
    # Mirrors submissions/int4_g128_lmhead/serve.py verbatim.
    return [
        py, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--served-model-name", "gemma-4-e4b-it",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--dtype", "bfloat16",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.90",
        "--trust-remote-code",
        "--no-enable-log-requests",
        "--max-num-batched-tokens", max_num_batched_tokens,
    ]


def wait_ready(proc: subprocess.Popen, base_url: str, timeout_s: int = 1200) -> None:
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited rc={proc.returncode} before ready")
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5.0) as r:
                if r.status == 200:
                    return
                last = f"status={r.status}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = str(exc)
        time.sleep(5)
    raise RuntimeError(f"not ready at {base_url}: {last}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-name", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--max-num-batched-tokens", default="512")
    ap.add_argument("--extra-flag", action="append", default=[],
                    help="extra vLLM CLI token; repeat for multi-token flags")
    ap.add_argument("--extra-env", action="append", default=[],
                    help="KEY=VALUE env override; repeatable")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    model = str(MODEL_DIR)
    base_url = f"http://127.0.0.1:{args.port}"

    env = os.environ.copy()
    for note in paths.prepare_local_gpu_env():
        print(f"[arm:{args.arm_name}] {note}", flush=True)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = env.get("VLLM_USE_FLASHINFER_SAMPLER", "0")
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for kv in args.extra_env:
        k, v = kv.split("=", 1)
        env[k] = v

    cmd = canonical_serve_cmd(py, model, args.port, args.max_num_batched_tokens) + args.extra_flag
    log_path = out_dir / "server.log"
    print(f"[arm:{args.arm_name}] serve: {' '.join(cmd)}", flush=True)
    print(f"[arm:{args.arm_name}] extra_env: {args.extra_env}", flush=True)

    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                text=True, preexec_fn=os.setsid)
        result: dict = {"arm": args.arm_name, "extra_flag": args.extra_flag,
                        "extra_env": args.extra_env, "cmd": cmd}
        try:
            wait_ready(proc, base_url)
            ready_s = time.time() - t0
            print(f"[arm:{args.arm_name}] ready in {ready_s:.0f}s", flush=True)
            out_file = out_dir / "decode_outputs.jsonl"
            summary_file = out_dir / "decode_summary.json"
            summary = harness.capture_decode(
                Path(py), base_url=base_url, model="gemma-4-e4b-it",
                out_file=out_file, summary_file=summary_file,
                num_prompts=args.num_prompts, output_len=args.output_len,
                seed=paths.SEED, tokenizer=model,
            )
            wall_tps = summary["num_completion_tokens"] / summary["duration_s"]
            result.update({
                "ready_s": ready_s,
                "num_completion_tokens": summary["num_completion_tokens"],
                "duration_s": summary["duration_s"],
                "wall_tps": wall_tps,
                "tau_official_proj": wall_tps * 1.03524,
                "decode_jsonl": str(out_file),
            })
            print(f"[arm:{args.arm_name}] wall_tps={wall_tps:.3f} "
                  f"official_proj={wall_tps*1.03524:.3f} "
                  f"({summary['num_completion_tokens']} tok / {summary['duration_s']:.1f}s)",
                  flush=True)
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=30)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
    (out_dir / "arm_result.json").write_text(json.dumps(result, indent=2))
    print(f"[arm:{args.arm_name}] wrote {out_dir / 'arm_result.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
