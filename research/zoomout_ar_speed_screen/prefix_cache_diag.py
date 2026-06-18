#!/usr/bin/env python
"""PR #630 self-determinism mechanism diagnostic (analysis-only, LOCAL).

The main census found stock vLLM 0.22.0 serving int4_g128_lmhead at M=1 greedy is
NOT self-deterministic across two back-to-back passes on ONE warm server:
``stock r1 vs r2 = 34/64`` completion-token mismatch (prompt_sha parity holds; some
prompts diverge at decode position 0). All three ``*_r1`` passes were byte-identical
because each was the FIRST pass on a fresh (cold-cache) server -> the cross-pass
state is the driver, not the attention backend (all resolved to TRITON_ATTN).

Prime suspect: ``enable_prefix_caching=True`` (vLLM V1 default). r1 does a cold full
chunked prefill; r2 hits the prefix cache from r1 and prefills only the uncached
remainder under a DIFFERENT chunk-boundary alignment -> slightly different KV (FP)
-> int4-Marlin grid-tie flips at greedy argmax (same class as #616's 0.43% and the
#607/#621 batch-variant-reduction family, but here on the prefix-cache-state axis).

This diagnostic stands up a STANDALONE vLLM server with the EXACT stock flags and a
single toggle (``--prefix-cache off`` adds ``--no-enable-prefix-caching``), then runs
the SAME 64x512 seed-1 decode twice and compares r1 vs r2. It does NOT touch the
submission serve.py / official path. Prefix caching gives ~0 benefit at unique-prompt
M=1, so if disabling it restores r1==r2 byte-exactness, that is a speed-neutral,
#319-stabilising config -- reported as a SURFACE for the advisor, not a fire.

Usage (server venv auto-detected; reuses serve_census decode worker + census):
  python research/zoomout_ar_speed_screen/prefix_cache_diag.py --prefix-cache off --port 8011
  python research/zoomout_ar_speed_screen/prefix_cache_diag.py --prefix-cache on  --port 8012   # control
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:] = [p for p in sys.path if p not in ("", str(Path(__file__).resolve().parent))]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.zoomout_ar_speed_screen.serve_census import _aggregate_tps, _census_compare  # noqa: E402
from scripts.local_validation import harness, paths  # noqa: E402

BUILT_CKPT = Path("/workspace/gemma_build/int4_g128_lmhead")
OUT_ROOT = ROOT / "research" / "zoomout_ar_speed_screen" / "out"
SELF_PY = str(Path(ROOT / "research" / "zoomout_ar_speed_screen" / "serve_census.py").resolve())

NUM_PROMPTS = 64
OUTPUT_LEN = 512
SEED = 1
WARMUP = 4


def _wait_ready(base_url: str, timeout_s: int = 1800) -> None:
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5.0) as r:
                if r.status == 200:
                    return
                last = f"status={r.status}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = str(exc)
        time.sleep(5)
    raise RuntimeError(f"server not ready at {base_url}: {last}")


def _warm(base_url: str, model: str, n: int = WARMUP) -> None:
    payload = {"model": model, "prompt": "Warm up the server.", "max_tokens": 16,
               "temperature": 0.0, "stream": False, "ignore_eos": True}
    body = json.dumps(payload).encode()
    for _ in range(n):
        req = urllib.request.Request(f"{base_url}/v1/completions", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                r.read()
        except Exception as exc:  # noqa: BLE001
            print(f"[diag] warmup failed (non-fatal): {exc!r}", flush=True)


def _decode_pass(server_python: Path, worker_env: dict, base_url: str, model: str,
                 out_file: Path) -> dict:
    cmd = [
        str(server_python), SELF_PY, "--decode-worker",
        "--base-url", base_url, "--model", model,
        "--dataset-path", str(paths.EVAL_PROMPTS), "--tokenizer", paths.TOKENIZER,
        "--num-prompts", str(NUM_PROMPTS), "--output-len", str(OUTPUT_LEN),
        "--seed", str(SEED), "--out-file", str(out_file), "--request-timeout-s", "600",
    ]
    print(f"[diag] decode pass -> {out_file.name}", flush=True)
    subprocess.run(cmd, check=True, timeout=7200, env=worker_env)
    return json.loads(out_file.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-cache", choices=["on", "off"], required=True)
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--served-name", default="gemma-4-e4b-it")
    args = ap.parse_args()

    for note in paths.prepare_local_gpu_env():
        print(f"[diag] {note}", flush=True)
    if not BUILT_CKPT.exists():
        print(f"[diag] FAIL: built checkpoint missing at {BUILT_CKPT}", flush=True)
        return 1

    manifest = harness.load_manifest(ROOT / "submissions" / "int4_g128_lmhead")
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    base_url = f"http://127.0.0.1:{args.port}"
    tag = f"pcache_{args.prefix_cache}"
    log_path = OUT_ROOT / f"server_{tag}.log"

    # EXACT stock flags (mirrors submissions/int4_g128_lmhead/serve.py + harness
    # --max-num-batched-tokens 512), plus the single prefix-cache toggle.
    serve_cmd = [
        str(server_python), "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(BUILT_CKPT), "--served-model-name", args.served_name,
        "--host", "127.0.0.1", "--port", str(args.port),
        "--dtype", "bfloat16", "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.90", "--trust-remote-code",
        "--no-enable-log-requests", "--max-num-batched-tokens", "512",
    ]
    if args.prefix_cache == "off":
        serve_cmd.append("--no-enable-prefix-caching")

    # worker subprocess must run under the SERVER venv (needs transformers tokenizer)
    worker_env = os.environ.copy()
    worker_env.pop("PYTHONPATH", None)
    worker_env["VIRTUAL_ENV"] = str(server_python.parent.parent)
    worker_env["PATH"] = f"{server_python.parent}{os.pathsep}{worker_env.get('PATH', '')}"
    worker_env["PYTHONDONTWRITEBYTECODE"] = "1"
    worker_env["PYTHONSAFEPATH"] = "1"

    server_env = os.environ.copy()
    server_env["MODEL_ID"] = str(BUILT_CKPT)

    print(f"[diag] serve: {' '.join(serve_cmd)}", flush=True)
    log = open(log_path, "w")
    proc = subprocess.Popen(serve_cmd, env=server_env, stdout=log,
                            stderr=subprocess.STDOUT, text=True, preexec_fn=os.setsid)
    result: dict = {"prefix_cache": args.prefix_cache, "port": args.port}
    try:
        _wait_ready(base_url)
        print(f"[diag] server ready at {base_url}", flush=True)
        _warm(base_url, args.served_name)

        r1 = _decode_pass(server_python, worker_env, base_url, args.served_name,
                          OUT_ROOT / f"{tag}_r1.json")
        r2 = _decode_pass(server_python, worker_env, base_url, args.served_name,
                          OUT_ROOT / f"{tag}_r2.json")

        census = _census_compare(r1["per_request"], r2["per_request"])
        tps1 = _aggregate_tps(r1, WARMUP)
        tps2 = _aggregate_tps(r2, WARMUP)
        result.update({
            "served": True,
            "self_determinism_r1_vs_r2": census,
            "warm_median_tps_r1": tps1["warm_median_tps"],
            "warm_median_tps_r2": tps2["warm_median_tps"],
            "all_full_length_r1": tps1["all_full_length"],
            "all_full_length_r2": tps2["all_full_length"],
        })
        print(f"\n[diag] prefix_cache={args.prefix_cache} "
              f"byte_exact_r1_vs_r2={census['byte_exact']} "
              f"mismatch={census['n_token_mismatch']}/{census['n_compared']} "
              f"prompt_sha_parity={census['prompt_sha_parity']} "
              f"tps_r1={tps1['warm_median_tps']:.2f} tps_r2={tps2['warm_median_tps']:.2f}", flush=True)
    except Exception as exc:  # noqa: BLE001
        result["served"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[diag] FAILED: {result['error']}", flush=True)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), 15)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), 9)
        log.close()

    out_file = OUT_ROOT / f"diag_{tag}.json"
    out_file.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"[diag] wrote {out_file}", flush=True)
    return 0 if result.get("served") else 1


if __name__ == "__main__":
    raise SystemExit(main())
