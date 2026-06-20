#!/usr/bin/env python
"""Local compatibility smoke: does FlashInfer attention dispatch on Gemma4-E4B?

PR #779 asks whether VLLM_ATTENTION_BACKEND=FLASHINFER is a usable speed lever on
the bi0 int4 stack. Two facts established from the vLLM 0.22.0 source first:
  * VLLM_ATTENTION_BACKEND env is NOT read by vLLM 0.22.0 (grep: zero hits) -> the
    PR-instructed env would be a silent no-op (Gemma4 TRITON force-pin still fires).
  * The real selector is the engine arg attention_backend=FLASHINFER, which bypasses
    the force-pin (config.py:89-99 guard requires attention_config.backend is None).

This script takes the only real path: pass the engine arg explicitly, plain offline
LLM (NO speculative config -- the most fundamental question is whether FlashInfer
attention loads at all on Gemma4's heterogeneous head dims, 256 sliding / 512 global).
Prior #582 (vLLM dev307) crashed here at warmup: BatchPrefillWithPagedKVCacheDispatched
(prefill.cuh): Unsupported max_mma_kv. We re-test on vLLM 0.22.0 + flashinfer 0.6.11.post2.

LOCAL ONLY. No HF job. Single A10G (CUDA_VISIBLE_DEVICES=0).
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

MODEL = os.environ.get("MODEL_ID", "google/gemma-4-E4B-it-qat-w4a16-ct")


def main() -> int:
    print(f"[smoke] vLLM import ...", flush=True)
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    print(f"[smoke] vllm={vllm.__version__}", flush=True)
    try:
        import flashinfer

        print(f"[smoke] flashinfer={getattr(flashinfer, '__version__', '?')}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[smoke] flashinfer import failed: {e}", flush=True)

    print(f"[smoke] building LLM(model={MODEL}, attention_backend=FLASHINFER, no-spec) ...", flush=True)
    try:
        llm = LLM(
            model=MODEL,
            attention_backend=AttentionBackendEnum.FLASHINFER,
            dtype="bfloat16",
            max_model_len=2048,
            gpu_memory_utilization=0.85,
            max_num_seqs=1,
            enforce_eager=True,
            trust_remote_code=True,
        )
    except Exception:
        print("\n[smoke] RESULT=FAIL_ENGINE_INIT", flush=True)
        print("[smoke] FlashInfer attention did NOT initialize on Gemma4-E4B.", flush=True)
        traceback.print_exc()
        return 2

    print("[smoke] engine init OK; running a short greedy generate ...", flush=True)
    try:
        out = llm.generate(
            ["The capital of France is"],
            SamplingParams(temperature=0.0, max_tokens=16),
        )
        text = out[0].outputs[0].text
        print(f"[smoke] generate OK -> {text!r}", flush=True)
    except Exception:
        print("\n[smoke] RESULT=FAIL_GENERATE", flush=True)
        traceback.print_exc()
        return 3

    print("\n[smoke] RESULT=PASS", flush=True)
    print("[smoke] FlashInfer attention LOADS and DECODES on Gemma4-E4B (no-spec).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
