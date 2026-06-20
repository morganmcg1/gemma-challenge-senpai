#!/usr/bin/env python
"""Opportunistic FlashInfer compat smoke under intermittent co-tenant GPU load.

The assigned A10G is shared with a persistent co-tenant vLLM server that holds
~19.6 GiB most of the time, releasing only briefly (e.g. on its own restart).
This grabber polls free VRAM and, the instant a window opens, builds an offline
LLM with attention_backend=FLASHINFER and runs one greedy generate -- the decisive
test of whether FlashInfer attention DISPATCHES on Gemma4-E4B int4-Marlin.

Exit codes:
  0  PASS  -> FlashInfer attention loaded + decoded (compatible).
  2  FAIL_FLASHINFER -> engine init crashed for a NON-memory reason (the decisive
       incompatibility, e.g. Unsupported max_mma_kv / prefill.cuh dispatch). STOP.
  3  FAIL_GENERATE -> loaded but generate crashed.
  4  NO_FREE_WINDOW -> deadline elapsed without ever catching enough free VRAM.

A memory-OOM at engine init ("Free memory on device ... less than desired") is
NOT decisive -- it just means we lost the race to the co-tenant; the grabber keeps
retrying until the deadline. Modest footprint (util 0.5 / max_model_len 2048,
enforce_eager) so a partial free window suffices and we stay a polite tenant.
LOCAL ONLY. No HF job. CUDA_VISIBLE_DEVICES=0.
"""
from __future__ import annotations

import os
import sys
import time
import traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

MODEL = os.environ.get("MODEL_ID", "google/gemma-4-E4B-it-qat-w4a16-ct")
FREE_MIB_REQUIRED = int(os.environ.get("FREE_MIB_REQUIRED", "12000"))
DEADLINE_S = int(os.environ.get("DEADLINE_S", "1500"))
POLL_S = float(os.environ.get("POLL_S", "4"))
GPU_MEM_UTIL = float(os.environ.get("GPU_MEM_UTIL", "0.5"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "2048"))

_OOM_SIGNS = ("Free memory on device", "less than desired GPU memory", "out of memory")


def free_mib() -> int:
    import torch

    free, _ = torch.cuda.mem_get_info()
    return int(free / 1024**2)


def try_launch_and_generate() -> int:
    from vllm import LLM, SamplingParams
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    try:
        llm = LLM(
            model=MODEL,
            attention_backend=AttentionBackendEnum.FLASHINFER,
            dtype="bfloat16",
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEM_UTIL,
            max_num_seqs=1,
            enforce_eager=True,
            trust_remote_code=True,
        )
    except Exception:
        tb = traceback.format_exc()
        if any(sign in tb for sign in _OOM_SIGNS):
            print("[grab] engine init OOM (lost race to co-tenant) -- will retry.", flush=True)
            return -1  # retryable
        print("\n[grab] RESULT=FAIL_FLASHINFER (non-memory engine-init crash):", flush=True)
        print(tb, flush=True)
        return 2  # decisive incompatibility
    print("[grab] engine init OK; running short greedy generate ...", flush=True)
    try:
        out = llm.generate(["The capital of France is"], SamplingParams(temperature=0.0, max_tokens=16))
        print(f"[grab] generate OK -> {out[0].outputs[0].text!r}", flush=True)
    except Exception:
        print("\n[grab] RESULT=FAIL_GENERATE:", flush=True)
        traceback.print_exc()
        return 3
    print("\n[grab] RESULT=PASS", flush=True)
    print("[grab] FlashInfer attention LOADS and DECODES on Gemma4-E4B int4-Marlin (no-spec).", flush=True)
    return 0


def main() -> int:
    import vllm

    print(f"[grab] vllm={vllm.__version__}", flush=True)
    try:
        import flashinfer

        print(f"[grab] flashinfer={getattr(flashinfer, '__version__', '?')}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[grab] flashinfer import failed: {e}", flush=True)

    print(
        f"[grab] waiting for free>={FREE_MIB_REQUIRED}MiB (util={GPU_MEM_UTIL}, "
        f"max_model_len={MAX_MODEL_LEN}); deadline={DEADLINE_S}s, poll={POLL_S}s",
        flush=True,
    )
    deadline = time.time() + DEADLINE_S
    attempts = 0
    last_log = 0.0
    while time.time() < deadline:
        f = free_mib()
        now = time.time()
        if now - last_log >= 30:
            print(f"[grab] t+{int(now - (deadline - DEADLINE_S))}s free={f}MiB", flush=True)
            last_log = now
        if f >= FREE_MIB_REQUIRED:
            attempts += 1
            print(f"[grab] WINDOW free={f}MiB >= {FREE_MIB_REQUIRED} -> attempt #{attempts}", flush=True)
            rc = try_launch_and_generate()
            if rc != -1:
                return rc
            # OOM race: brief backoff then keep polling.
            time.sleep(POLL_S)
            continue
        time.sleep(POLL_S)
    print(f"\n[grab] RESULT=NO_FREE_WINDOW after {DEADLINE_S}s ({attempts} launch attempts).", flush=True)
    return 4


if __name__ == "__main__":
    sys.exit(main())
