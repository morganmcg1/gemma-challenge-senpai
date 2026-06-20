#!/usr/bin/env python
"""PR #779 liveness + orientation W&B run (group bi0-flashinfer).

Records the screening plan and the current local-pod state at pickup so the
advisor has a visible signal (W&B run in the named group) while the GPU
contention block is resolved. CPU-only: does NOT load the model.

PLAN (one-change screen): control = bi0 unmodified (int4 W4A16 Marlin target +
gemma4_assistant MTP K=6, VLLM_BATCH_INVARIANT=0, TRITON_ATTN force-2D surgical
patch). variant = same stack with attention swapped to FlashInfer
(attention_backend=FLASHINFER engine arg; the surgattn force-2D patch dropped).
Cheapest decision-relevant step first: does FlashInfer attention even DISPATCH on
Gemma4-E4B's heterogeneous head dims (256 sliding / 512 global) with int4-Marlin?
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.wandb_logging import init_wandb_run, log_event, finish_wandb  # noqa: E402


def gpu_free_total_gib():
    try:
        import torch

        if not torch.cuda.is_available():
            return (None, None)
        free, total = torch.cuda.mem_get_info()
        return (round(free / 1024**3, 2), round(total / 1024**3, 2))
    except Exception:
        return (None, None)


def main() -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    free_gib, total_gib = gpu_free_total_gib()
    gpu_blocked = (free_gib is not None) and (free_gib < 6.0)

    run = init_wandb_run(
        job_type="screening",
        agent="stark",
        name="stark/bi0-flashinfer-liveness",
        group="bi0-flashinfer",
        notes="PR #779 FlashInfer decode-backend screen: liveness + orientation.",
        tags=["pr-779", "flashinfer", "bi0", "screening", "liveness"],
        config={
            "pr": 779,
            "submission_control": "int4_mtp_bi0_surgattn",
            "model_id": "google/gemma-4-E4B-it-qat-w4a16-ct",
            "drafter_model": "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant",
            "num_speculative_tokens": 6,
            "vllm_batch_invariant": 0,
            "backend_control": "TRITON_ATTN_force2d (surgattn)",
            "backend_variant": "FLASHINFER (attention_backend engine arg)",
            "vllm_version": "0.22.0",
            "flashinfer_version": "0.6.11.post2",
            "baseline_official_tps": 218.02,
            "baseline_official_ppl": 2.0058,
            "gpu_free_gib_at_pickup": free_gib,
            "gpu_total_gib": total_gib,
            "gpu_blocked_by_contention": gpu_blocked,
            "prior_memory": (
                "flashinfer-levers-582 (vLLM dev307): FlashInfer attention crashed at "
                "warmup on Gemma4-E4B (BatchPrefillWithPagedKVCacheDispatched prefill.cuh: "
                "Unsupported max_mma_kv). Re-testing on 0.22.0 + flashinfer 0.6.11.post2."
            ),
        },
    )
    if run is None:
        print("[liveness] wandb init returned None (mode/key?) -- no run created.", flush=True)
        return 1

    log_event(
        run,
        "pickup_orientation",
        step=0,
        metrics={
            "gpu/free_gib": free_gib if free_gib is not None else -1.0,
            "gpu/total_gib": total_gib if total_gib is not None else -1.0,
            "gpu/blocked": 1.0 if gpu_blocked else 0.0,
        },
        data={
            "status": "blocked_gpu_contention" if gpu_blocked else "gpu_ok",
            "plan": "control=bi0(TRITON force-2D); variant=FlashInfer; smoke compat first",
        },
    )
    run.summary["status"] = "blocked_gpu_contention" if gpu_blocked else "gpu_ok"
    run.summary["gpu_free_gib_at_pickup"] = free_gib
    print(f"[liveness] W&B run id={run.id} url={run.url}", flush=True)
    print(f"[liveness] gpu_free_gib={free_gib} total={total_gib} blocked={gpu_blocked}", flush=True)
    finish_wandb(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
