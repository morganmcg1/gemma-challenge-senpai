#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #569 worker -- one base_fullhead decode profile pass (graph OR eager).

Run once per CUDA-graph arm (a fresh process each, so the two engines never share
a CUDA context). Mirrors the official decode profiler
(``official/.../gemma_decode_profiler_claudecode/profile_{graph,eager}.py``) LLM
config exactly so the numbers are comparable, but ADDS the three things the
official profilers do not emit and PR #569 needs:

  1. the generated greedy ``token_ids`` (graph<->eager byte-identity check),
  2. the FULL per-kernel self-device-time list (not just top-N) so the parent can
     split head (bf16 dense lm_head GEMV) from body (int4 Marlin GEMVs),
  3. a per-token normalization (clean no-profiler TPS + profiled device busy).

base_fullhead == ``google/gemma-4-E4B-it-qat-w4a16-ct`` (stock int4 body + FULL
262k bf16 tied head; lm_head.weight is BF16 [262144,2560], body weights are int4
compressed-tensors Marlin). Plain M=1 AR decode (NO spec drafter) -- this is the
clean isolation of the MAIN-MODEL-forward CUDA-graph lever; the served 252.69 TPS
anchor (wirbel #553) is the spec frame and is reconciled in the parent.

Env: MODEL_ID, STATE_DIR, ENFORCE_EAGER(0/1), TPS_TOKENS, PROFILE_TOKENS.
Writes ``$STATE_DIR/worker_{graph,eager}.json`` and exits 0.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-4-E4B-it-qat-w4a16-ct")
STATE = os.environ.get("STATE_DIR", "/tmp")
ENFORCE_EAGER = os.environ.get("ENFORCE_EAGER", "0") == "1"
TPS_TOKENS = int(os.environ.get("TPS_TOKENS", "256"))
PROFILE_TOKENS = int(os.environ.get("PROFILE_TOKENS", "256"))
MODE = "eager" if ENFORCE_EAGER else "graph"

# Fixed greedy prompt (verbatim from the official profilers, so graph/eager and
# this card all decode the same byte sequence).
PROMPT = ("Explain, step by step, how a transformer language model generates text "
          "one token at a time, and why decode is memory-bandwidth bound.")


def _self_dev(e) -> float:
    """Self device time (us) for one key_averages entry."""
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(e, attr, None)
        if v is not None:
            return float(v)
    return 0.0


def _is_cuda_kernel(e) -> bool:
    """True iff this entry is an ACTUAL on-device kernel (DeviceType.CUDA), not a
    CPU-side aten/_C operator dispatcher nor a profiler-internal marker.

    CRITICAL for correct attribution: in this torch build's flat ``key_averages``
    the CPU operator carries the SAME ``self_device_time_total`` as the child kernel
    it launched (e.g. ``aten::mm`` 2776us AND its child ``gemv2T_kernel_val`` 2776us;
    ``_C::marlin_gemm`` AND ``marlin::Marlin`` likewise). Summing both double-counts
    real GPU work (and worse in eager, where the op rows carry full device time).
    Spurious rows like ``Activity Buffer Request`` also carry bogus device time.
    Summing self-device over ONLY DeviceType.CUDA events counts each kernel once."""
    return str(getattr(e, "device_type", "")) == "DeviceType.CUDA"


def main() -> int:
    os.makedirs(STATE, exist_ok=True)
    # V1 runs the model in a separate EngineCore process by default -> an in-process
    # torch.profiler sees zero CUDA kernels. Force in-process (must precede vllm import).
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    print(f"[worker:{MODE}] torch {torch.__version__}; dev {torch.cuda.get_device_name(0)}; "
          f"enforce_eager={ENFORCE_EAGER}", flush=True)

    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model=MODEL_ID, quantization="compressed-tensors", dtype="bfloat16",
        max_model_len=4096, gpu_memory_utilization=0.90, max_num_batched_tokens=512,
        max_num_seqs=1, enforce_eager=ENFORCE_EAGER, trust_remote_code=True,
        disable_log_stats=True,
    )
    load_s = time.time() - t0
    print(f"[worker:{MODE}] model ready in {load_s:.1f}s", flush=True)

    sp = lambda n: SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)

    # Warmup so CUDA graphs are captured / autotune settled before timing.
    llm.generate([PROMPT], sp(16))
    torch.cuda.synchronize()

    # 1) Clean per-token wall (NO profiler) -> the faithful TPS for this arm.
    t0 = time.time()
    out = llm.generate([PROMPT], sp(TPS_TOKENS))
    torch.cuda.synchronize()
    wall = time.time() - t0
    token_ids = list(out[0].outputs[0].token_ids)
    n = len(token_ids)
    tps = n / wall if wall else float("nan")
    print(f"[worker:{MODE}] CLEAN TPS: {n} tok / {wall:.3f}s = {tps:.2f} tok/s", flush=True)

    # 2) Profiled pass (CUPTI sees device kernels even under graph replay).
    from torch.profiler import ProfilerActivity, profile
    tp0 = time.time()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        pout = llm.generate([PROMPT], sp(PROFILE_TOKENS))
        torch.cuda.synchronize()
    prof_wall = time.time() - tp0
    pn = len(pout[0].outputs[0].token_ids)

    # Device-kernel rows ONLY (DeviceType.CUDA) -> each real kernel counted once, no
    # operator<->kernel double count, no profiler-artifact device time.
    rows = [(e.key, _self_dev(e), int(getattr(e, "count", 0)))
            for e in prof.key_averages() if _is_cuda_kernel(e)]
    rows = [(k, us, c) for (k, us, c) in rows if us > 0]
    rows.sort(key=lambda r: r[1], reverse=True)
    busy_us = sum(r[1] for r in rows)
    busy_share = 100.0 * (busy_us / 1e6) / prof_wall if prof_wall else float("nan")
    # De-duped CPU-operator rows we EXCLUDED (kept for transparency in the artifact:
    # confirms the double-count we removed, e.g. aten::mm / _C::marlin_gemm).
    cpu_rows = sorted(((e.key, _self_dev(e), int(getattr(e, "count", 0)))
                       for e in prof.key_averages()
                       if (not _is_cuda_kernel(e)) and _self_dev(e) > 0),
                      key=lambda r: r[1], reverse=True)

    print(f"[worker:{MODE}] profiled {pn} tok / {prof_wall*1000:.0f}ms wall; "
          f"GPU-busy(device-kernels) {busy_us/1000:.1f}ms = {busy_share:.1f}% of wall "
          f"-> non-kernel ~{max(0.0,100.0-busy_share):.1f}%", flush=True)
    print(f"[worker:{MODE}] top DEVICE kernels:", flush=True)
    for name, us, cnt in rows[:18]:
        print(f"    {100*us/busy_us:5.1f}% {us/1000:8.2f}ms x{cnt:<6d} {name[:78]}", flush=True)
    print(f"[worker:{MODE}] excluded CPU-op self_device (double-count avoided), top 6:", flush=True)
    for name, us, cnt in cpu_rows[:6]:
        print(f"    -{us/1000:8.2f}ms x{cnt:<6d} {name[:74]}", flush=True)

    result = {
        "mode": MODE, "enforce_eager": ENFORCE_EAGER, "model": MODEL_ID,
        "load_s": load_s,
        "tps": tps, "tps_tokens": n, "tps_wall_s": wall, "token_ids": token_ids,
        "profile_tokens": pn, "profile_wall_s": prof_wall,
        "gpu_busy_us_total": busy_us,
        "gpu_busy_share_of_wall_pct": busy_share,
        "gpu_busy_per_token_us": busy_us / pn if pn else float("nan"),
        # FULL list of DEVICE kernels (name, self_device_us over profiled pass, count).
        "kernel_rows": [{"name": k, "self_us": us, "count": c} for (k, us, c) in rows],
        # Excluded CPU-operator rows (top 12) -- transparency on the de-dup.
        "excluded_cpu_op_rows": [{"name": k, "self_us": us, "count": c}
                                 for (k, us, c) in cpu_rows[:12]],
    }
    out_path = os.path.join(STATE, f"worker_{MODE}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[worker:{MODE}] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
