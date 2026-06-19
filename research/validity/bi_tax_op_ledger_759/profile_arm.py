#!/usr/bin/env python3
"""PR #759 driver: profile a steady-state, pure-decode window of the FIRE served
path for one BI arm.

Protocol (prefix-cache isolates decode from prefill):
  1. wait for /v1/models ready
  2. WARMUP: send the fixed profiling prompt N times (JIT/autotune every Triton +
     cuBLAS + Marlin kernel, warm caches, and populate the prefix cache for the
     prompt) -- profiling must not capture one-time compilation.
  3. POST /start_profile
  4. send ONE generation: same prompt (prefill served from prefix cache => ~free)
     + max_tokens=GEN, temperature=0, ignore_eos -> GEN pure decode steps profiled
  5. POST /stop_profile, wait for the worker to flush *.pt.trace.json* into TRACE_DIR
  6. emit a summary JSON (prompt/completion tokens, wall, decode tps proxy)

Stdlib only (urllib); run on any interpreter.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import urllib.request


def _post(base: str, path: str, payload: dict | None, timeout: float):
    data = json.dumps(payload).encode() if payload is not None else b""
    req = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
        return r.status, (json.loads(body) if body.strip() else {})


def _ready(base: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base + "/v1/models", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def _build_prompt(approx_tokens: int) -> str:
    # Deterministic, representative natural-language prompt. ~0.75 tok/word for
    # Gemma; repeat a paragraph until we reach the target word count. Exact prompt
    # token count is reported from the server usage field.
    para = (
        "You are an expert systems engineer. Explain in careful, concrete detail "
        "how a modern speculative decoding inference server overlaps a small "
        "draft model with a larger verifier, why batch-invariant deterministic "
        "kernels can change the throughput of attention and matrix multiply "
        "operations, and what a profiler trace reveals about where wall-clock "
        "time is actually spent during steady-state autoregressive decoding. "
    )
    words_needed = int(approx_tokens / 0.75)
    words: list[str] = []
    while len(words) < words_needed:
        words.extend(para.split())
    return " ".join(words[:words_needed])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="gemma-4-e4b-it")
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--warmup-tokens", type=int, default=96)
    ap.add_argument("--gen-tokens", type=int, default=256)
    ap.add_argument("--summary", required=True)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    if not _ready(base, 1200):
        print("[driver] server never became ready", file=sys.stderr)
        return 2

    prompt = _build_prompt(args.prompt_tokens)

    def gen(max_tokens: int):
        payload = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": False,
        }
        t0 = time.time()
        status, body = _post(base, "/v1/completions", payload, timeout=1200)
        dt = time.time() - t0
        usage = body.get("usage", {}) if isinstance(body, dict) else {}
        return status, dt, usage

    # 2. warmup (also primes prefix cache for the prompt)
    for i in range(args.warmup):
        st, dt, usage = gen(args.warmup_tokens)
        print(f"[driver] warmup {i+1}/{args.warmup} status={st} dt={dt:.2f}s "
              f"prompt_tok={usage.get('prompt_tokens')} "
              f"completion_tok={usage.get('completion_tokens')}", flush=True)

    # snapshot existing trace files so we can detect the new one
    before = set(glob.glob(os.path.join(args.trace_dir, "*.pt.trace.json*")))

    # 3. start profile
    st, _ = _post(base, "/start_profile", None, timeout=120)
    print(f"[driver] start_profile status={st}", flush=True)

    # 4. profiled generation: prefill = prefix-cache hit, GEN pure decode steps
    st, dt, usage = gen(args.gen_tokens)
    prompt_tok = usage.get("prompt_tokens")
    completion_tok = usage.get("completion_tokens")
    decode_tps_proxy = (completion_tok / dt) if (completion_tok and dt) else None
    print(f"[driver] PROFILED gen status={st} dt={dt:.2f}s prompt_tok={prompt_tok} "
          f"completion_tok={completion_tok} decode_tps_proxy={decode_tps_proxy}",
          flush=True)

    # 5. stop profile + wait for the worker trace to flush
    st, _ = _post(base, "/stop_profile", None, timeout=300)
    print(f"[driver] stop_profile status={st}; waiting for trace flush", flush=True)
    new_files: list[str] = []
    deadline = time.time() + 300
    while time.time() < deadline:
        now = set(glob.glob(os.path.join(args.trace_dir, "*.pt.trace.json*")))
        new_files = sorted(now - before)
        if new_files:
            # let the file finish writing (size stable across 2 polls)
            sizes = {f: os.path.getsize(f) for f in new_files}
            time.sleep(4)
            if all(os.path.getsize(f) == sizes[f] for f in new_files):
                break
        time.sleep(3)

    summary = {
        "port": args.port,
        "prompt_tokens": prompt_tok,
        "warmup_requests": args.warmup,
        "gen_tokens_requested": args.gen_tokens,
        "completion_tokens": completion_tok,
        "profiled_wall_s": dt,
        "decode_tps_proxy": decode_tps_proxy,
        "trace_files": new_files,
        "trace_dir": args.trace_dir,
    }
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    print("[driver] summary:", json.dumps(summary), flush=True)
    if not new_files:
        print("[driver] WARNING: no new trace file detected", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
