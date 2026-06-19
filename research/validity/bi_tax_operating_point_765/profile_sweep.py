#!/usr/bin/env python3
"""PR #765 driver: extends the #759 prefix-cache-isolated profiler to (a) a
GENERATION-LENGTH SWEEP of pure-decode windows and (b) a COLD-PREFILL window, so
the batch-invariance tax can be split into a per-decode-token component and a
one-time prefill component, and tested for operating-point (gen-length) stability.

One server boot per BI arm (the server is booted by the orchestrator and reused
across every window via vLLM's repeated /start_profile + /stop_profile support:
gpu_worker.profile keeps the wrapper alive and tensorboard_trace_handler writes a
fresh unique-timestamped trace file on every stop).

Protocol for one arm (single A10G, MAX_NUM_SEQS=1, serial):
  1. wait for /v1/models ready
  2. WARMUP: send the fixed warm prompt N times (JIT/autotune every Triton +
     cuBLAS + Marlin kernel, warm caches, populate the prefix cache for the warm
     prompt). Profiling must never capture one-time compilation.
  3. THROWAWAY profiled window (GEN=32) -> absorbs CUPTI first-flush overhead so
     it is not charged to the first real sweep point. Discarded.
  4. DECODE SWEEP: for GEN in --gen-sweep, one profiled window:
       /start_profile -> gen(warm prompt, max_tokens=GEN, temp=0, ignore_eos)
       -> /stop_profile. The warm prompt prefill is a prefix-cache hit (~free),
       so the profiled window is GEN *pure decode* steps. Greedy continuations
       are identical across windows but vLLM never cache-skips DECODE (prefix
       cache accelerates prefill input matching only), so each window genuinely
       decodes GEN fresh steps; the wall decode_tps_proxy is the cross-check.
  5. PREFILL window: a FRESH, unique prompt (>=1 block of unique nonce prepended
     so block-0 prefix hash diverges -> guaranteed full cold prefill) with
     max_tokens=1: the profiled window = cold prefill(P tokens) + 1 decode step.
     prefill device time is recovered downstream by subtracting one decode step.

Emits one arm summary JSON: a list of window descriptors, each with kind, the
requested/served token counts, wall, decode_tps_proxy, and its trace file(s).

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


_PARA = (
    "You are an expert systems engineer. Explain in careful, concrete detail "
    "how a modern speculative decoding inference server overlaps a small "
    "draft model with a larger verifier, why batch-invariant deterministic "
    "kernels can change the throughput of attention and matrix multiply "
    "operations, and what a profiler trace reveals about where wall-clock "
    "time is actually spent during steady-state autoregressive decoding. "
)


def _build_prompt(approx_tokens: int, salt: str = "") -> str:
    # Deterministic, representative natural-language prompt (~0.75 tok/word for
    # Gemma). A non-empty salt is prepended (repeated to fill >= ~24 words, i.e.
    # >= 1 KV block) so the block-0 prefix hash diverges from the warm prompt and
    # the prefill is a guaranteed full cache MISS.
    words: list[str] = []
    if salt:
        saltwords = (salt + " ") * 24
        words.extend(saltwords.split())
    words_needed = int(approx_tokens / 0.75)
    while len(words) < words_needed:
        words.extend(_PARA.split())
    return " ".join(words[:words_needed])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="gemma-4-e4b-it")
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--warmup-tokens", type=int, default=96)
    ap.add_argument("--gen-sweep", default="128,256,512,1024")
    ap.add_argument("--throwaway-tokens", type=int, default=32)
    ap.add_argument("--summary", required=True)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    if not _ready(base, 1200):
        print("[driver] server never became ready", file=sys.stderr)
        return 2

    warm_prompt = _build_prompt(args.prompt_tokens)

    def gen(prompt: str, max_tokens: int):
        payload = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": False,
        }
        t0 = time.time()
        status, body = _post(base, "/v1/completions", payload, timeout=2400)
        dt = time.time() - t0
        usage = body.get("usage", {}) if isinstance(body, dict) else {}
        return status, dt, usage

    def snapshot() -> set:
        return set(glob.glob(os.path.join(args.trace_dir, "*.pt.trace.json*")))

    def wait_new(before: set) -> list[str]:
        deadline = time.time() + 300
        new_files: list[str] = []
        while time.time() < deadline:
            new_files = sorted(snapshot() - before)
            if new_files:
                sizes = {f: os.path.getsize(f) for f in new_files}
                time.sleep(4)
                if all(os.path.getsize(f) == sizes[f] for f in new_files):
                    break
            time.sleep(3)
        return new_files

    def profiled_window(kind: str, prompt: str, max_tokens: int) -> dict:
        before = snapshot()
        st, _ = _post(base, "/start_profile", None, timeout=120)
        sg, dt, usage = gen(prompt, max_tokens)
        st2, _ = _post(base, "/stop_profile", None, timeout=300)
        ptok = usage.get("prompt_tokens")
        ctok = usage.get("completion_tokens")
        proxy = (ctok / dt) if (ctok and dt) else None
        files = wait_new(before)
        rec = {
            "kind": kind,
            "gen_tokens_requested": max_tokens,
            "prompt_tokens": ptok,
            "completion_tokens": ctok,
            "wall_s": dt,
            "decode_tps_proxy": proxy,
            "start_status": st, "gen_status": sg, "stop_status": st2,
            "trace_files": files,
        }
        print(f"[driver] WINDOW {kind} gen_req={max_tokens} prompt_tok={ptok} "
              f"completion_tok={ctok} wall={dt:.3f}s proxy={proxy} "
              f"trace_files={len(files)}", flush=True)
        if not files:
            print(f"[driver] WARNING: no trace for window {kind}", file=sys.stderr)
        return rec

    # 2. warmup (JIT + prime prefix cache for warm prompt)
    for i in range(args.warmup):
        st, dt, usage = gen(warm_prompt, args.warmup_tokens)
        print(f"[driver] warmup {i+1}/{args.warmup} status={st} dt={dt:.2f}s "
              f"prompt_tok={usage.get('prompt_tokens')} "
              f"completion_tok={usage.get('completion_tokens')}", flush=True)

    windows: list[dict] = []

    # 3. throwaway profiled window (absorb CUPTI first-flush); discarded
    tw = profiled_window("throwaway", warm_prompt, args.throwaway_tokens)
    tw["discarded"] = True
    windows.append(tw)

    # 4. decode sweep: pure-decode windows (warm prompt -> prefill cache hit)
    gens = [int(x) for x in args.gen_sweep.split(",") if x.strip()]
    for g in gens:
        windows.append(profiled_window(f"decode_gen{g}", warm_prompt, g))

    # 5. prefill window: fresh unique prompt (cold prefill) + 1 decode step
    salt = f"nonce{int(time.time()*1000)%100000000}prefillcold"
    fresh_prompt = _build_prompt(args.prompt_tokens, salt=salt)
    windows.append(profiled_window("prefill", fresh_prompt, 1))

    summary = {
        "port": args.port,
        "prompt_tokens_requested": args.prompt_tokens,
        "warmup_requests": args.warmup,
        "gen_sweep": gens,
        "windows": windows,
        "trace_dir": args.trace_dir,
    }
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    print("[driver] summary written:", args.summary, flush=True)
    missing = [w["kind"] for w in windows if not w["trace_files"]]
    if missing:
        print(f"[driver] WARNING: windows missing traces: {missing}",
              file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
