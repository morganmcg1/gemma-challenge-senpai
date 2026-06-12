#!/usr/bin/env python
"""claudecode — GRAPH-MODE decode profiler (real serving config), via torch.profiler.

Run3 (eager) showed decode compute ~83% weight-GEMM, attention ~4%, sampling ~0.1% — but
eager masks the real overhead gap. This run, with CUDA graphs ON (the real config), answers
the go/no-go for a high-ceiling sub-4-bit kernel:
  - GPU-busy vs wall  -> is there non-kernel (scheduler/python) overhead to claw back?
  - GEMM share of GPU-busy -> the size of the prize a sub-4-bit kernel could capture.

torch.profiler (CUPTI) captures DEVICE kernels even under graph replay (graphs only hide the
CPU-side dispatch). We use self_device_time per entry => clean GPU-busy with no parent/child
double-counting. VLLM_ENABLE_V1_MULTIPROCESSING=0 keeps the engine in-process so the profiler
sees it (proven in run3). Writes JSON + summary to /state and stdout; exits.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-4-E4B-it-qat-w4a16-ct")
STATE = os.environ.get("STATE_DIR", "/state")
TPS_TOKENS = int(os.environ.get("TPS_TOKENS", "256"))
PROFILE_TOKENS = int(os.environ.get("PROFILE_TOKENS", "64"))

CATEGORIES = [
    ("attention", ["attn", "_fwd", "flash", "paged", "unified_attention",
                   "reshape_and_cache", "rotary", "rope"]),
    ("sampling_lmhead", ["log_softmax", "logsoftmax", "argmax", "topk", "top_k",
                         "softmax", "sample", "logits", "cumsum", "sort"]),
    ("matmul_gemm", ["marlin", "gptq", "gemm", "cutlass", "wmma", "gemv", "splitk",
                     "split_k", "ampere", "s16816", "dot", "_C::"]),
    ("norm", ["rms", "layernorm", "layer_norm", "norm_kernel"]),
    ("activation", ["silu", "gelu", "swiglu", "act_and_mul"]),
    ("elementwise_copy", ["elementwise", "copy", "cast", "convert", "memcpy",
                          "fill", "add", "mul", "index", "vectorized", "to"]),
]


def categorize(name: str) -> str:
    n = name.lower()
    if "marlin" in n or "gemv" in n or "gemm" in n:
        return "matmul_gemm"
    for cat, subs in CATEGORIES:
        if any(s in n for s in subs):
            return cat
    return "other"


def self_dev(e) -> float:
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(e, attr, None)
        if v is not None:
            return float(v)
    return 0.0


def main() -> None:
    os.makedirs(STATE, exist_ok=True)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    print(f"[gprof] torch {torch.__version__}; dev {torch.cuda.get_device_name(0)}; graphs ON",
          flush=True)

    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model=MODEL_ID, quantization="compressed-tensors", dtype="bfloat16",
        max_model_len=4096, gpu_memory_utilization=0.90, max_num_batched_tokens=512,
        max_num_seqs=1, enforce_eager=False, trust_remote_code=True, disable_log_stats=True,
    )
    print(f"[gprof] model+graphs ready in {time.time()-t0:.1f}s", flush=True)

    prompt = ("Explain, step by step, how a transformer language model generates text "
              "one token at a time, and why decode is memory-bandwidth bound.")
    sp = lambda n: SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)

    llm.generate([prompt], sp(16))  # warmup
    torch.cuda.synchronize()

    # 1) Clean graph-mode TPS (no profiler).
    t0 = time.time()
    out = llm.generate([prompt], sp(TPS_TOKENS))
    torch.cuda.synchronize()
    wall = time.time() - t0
    n = len(out[0].outputs[0].token_ids)
    tps = n / wall
    print(f"[gprof] GRAPH-MODE TPS: {n} tok / {wall:.3f}s = {tps:.2f} tok/s "
          f"(single-stream, base int4)", flush=True)

    # 2) Profiled generate (graphs on; CUPTI still captures device kernels).
    from torch.profiler import profile, ProfilerActivity
    tp0 = time.time()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        pout = llm.generate([prompt], sp(PROFILE_TOKENS))
        torch.cuda.synchronize()
    prof_wall = time.time() - tp0
    pn = len(pout[0].outputs[0].token_ids)

    rows = [(e.key, self_dev(e), int(getattr(e, "count", 0))) for e in prof.key_averages()]
    rows = [r for r in rows if r[1] > 0]
    rows.sort(key=lambda r: r[1], reverse=True)
    busy_us = sum(r[1] for r in rows)
    cats: dict[str, float] = {}
    for name, us, _ in rows:
        cats[categorize(name)] = cats.get(categorize(name), 0.0) + us

    busy_ms = busy_us / 1000
    busy_share = 100 * (busy_ms / 1000) / prof_wall if prof_wall else 0.0
    print("\n==== GRAPH-MODE PROFILE (self-device time; de-duped) ====", flush=True)
    print(f"  graph-mode TPS (base int4):    {tps:.2f} tok/s", flush=True)
    print(f"  profiled: {pn} tok / {prof_wall*1000:.0f} ms wall", flush=True)
    print(f"  GPU-busy (sum self-device):    {busy_ms:.1f} ms  "
          f"= {busy_share:.1f}% of wall  -> non-kernel overhead ~{max(0,100-busy_share):.1f}%",
          flush=True)
    print(f"  GPU-busy / token:              {busy_ms/pn:.3f} ms", flush=True)
    print("  --- GPU-busy by category (share of kernel time) ---", flush=True)
    for c, us in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"    {c:18s} {us/1000:9.2f} ms  {100*us/busy_us:5.1f}%", flush=True)
    print("  --- top 18 kernels by self-device time ---", flush=True)
    for name, us, cnt in rows[:18]:
        print(f"    {100*us/busy_us:5.1f}% {us/1000:8.2f}ms x{cnt:<6d} [{categorize(name)}] "
              f"{name[:66]}", flush=True)

    result = {
        "model": MODEL_ID, "mode": "graphs_on", "graph_tps": tps,
        "tps_tokens": n, "tps_wall_s": wall, "profile_tokens": pn,
        "profile_wall_s": prof_wall, "gpu_busy_ms": busy_ms,
        "gpu_busy_share_of_wall_pct": busy_share, "gpu_busy_per_token_ms": busy_ms / pn,
        "category_ms": {k: v / 1000 for k, v in cats.items()},
        "category_pct": {k: 100 * v / busy_us for k, v in cats.items()},
        "top_kernels": [{"kernel": nm, "ms": us / 1000, "count": c,
                         "pct": 100 * us / busy_us, "category": categorize(nm)}
                        for nm, us, c in rows[:25]],
    }
    try:
        with open(os.path.join(STATE, "graph_profile.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[gprof] wrote {STATE}/graph_profile.json", flush=True)
    except Exception as e:
        print(f"[gprof] could not write /state ({e})", flush=True)

    print("\n[gprof] done — exiting (harness 'not ready' is expected).", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
