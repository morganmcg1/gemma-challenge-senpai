#!/usr/bin/env python
"""claudecode — decode op-time PROFILER (not a real server).

Goal (step 1 of the kernel/overhead lane): size the ~33% overhead gap @ml-intern
flagged (127 TPS vs ~190 bw ceiling). The board established it is NOT launch overhead
(full batch-1 decode CUDA graph is captured by default) but real compute — head_dim-512
global-attention on Triton + 262k-vocab sampling. This profiles the int4 leader's decode
path and breaks CUDA time down by category so we know which slice is addressable.

This script does NOT start a server. It loads the official QAT W4A16 int4 checkpoint,
runs a single-stream decode under torch.profiler, prints a categorized breakdown +
top kernels to stdout (-> job_logs.txt), and writes JSON to /state. The harness will then
report "server exited before readiness" — expected; the profile is the deliverable.

enforce_eager=True ON PURPOSE: with CUDA graphs the whole decode step collapses into one
opaque graph-launch and per-kernel attribution is lost. Eager mode inflates ABSOLUTE times
(launch overhead returns) but the per-kernel COMPUTE COMPOSITION (attn vs MLP vs sampling)
— which is what we need to pick a kernel target — is faithful.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch


MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-4-E4B-it-qat-w4a16-ct")
STATE = os.environ.get("STATE_DIR", "/state")
GEN_TOKENS = int(os.environ.get("GEN_TOKENS", "256"))

# Category -> substrings matched (lowercase) against kernel names. First match wins,
# in this order, so more specific buckets precede generic GEMM.
CATEGORIES = [
    ("attention", ["attn", "_fwd_kernel", "flash", "paged", "unified_attention",
                   "reshape_and_cache", "rotary", "rope"]),
    ("sampling_lmhead", ["log_softmax", "logsoftmax", "argmax", "topk", "top_k",
                         "softmax", "sample", "logits", "cumsum", "sort"]),
    ("matmul_marlin", ["marlin", "gptq", "gemm", "cutlass", "ampere", "wmma",
                       "matmul", "mm_", "linear", "splitk", "split_k", "gemv"]),
    ("norm", ["rms", "layernorm", "layer_norm", "norm_kernel"]),
    ("activation", ["silu", "gelu", "swiglu", "mul_kernel", "act_"]),
    ("elementwise_copy", ["elementwise", "vectorized_elementwise", "copy", "cast",
                          "convert", "memcpy", "fill", "add_kernel", "index"]),
    ("graph_launch", ["cudagraphlaunch", "graph"]),
]


def categorize(name: str) -> str:
    n = name.lower()
    for cat, subs in CATEGORIES:
        if any(s in n for s in subs):
            return cat
    return "other"


def main() -> None:
    os.makedirs(STATE, exist_ok=True)
    # CRITICAL: vLLM V1 runs the model in a separate EngineCore process by default, so an
    # in-process torch.profiler sees zero CUDA kernels. Force the engine in-process so the
    # profiler captures the decode kernels. Must be set before importing vllm.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    print(f"[profiler] torch {torch.__version__}; cuda {torch.version.cuda}; "
          f"device {torch.cuda.get_device_name(0)}; "
          f"VLLM_ENABLE_V1_MULTIPROCESSING={os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']}", flush=True)

    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model=MODEL_ID,
        quantization="compressed-tensors",
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        max_num_batched_tokens=512,
        max_num_seqs=1,
        enforce_eager=True,          # per-kernel attribution (see module docstring)
        trust_remote_code=True,
        disable_log_stats=True,
    )
    print(f"[profiler] model loaded in {time.time()-t0:.1f}s", flush=True)

    prompt = ("Explain, step by step, how a transformer language model generates "
              "text one token at a time, and why decoding is memory-bandwidth bound.")
    greedy = SamplingParams(temperature=0.0, max_tokens=GEN_TOKENS, ignore_eos=True)

    # Warmup (triggers any lazy compilation / autotune so it doesn't pollute the trace).
    llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True))
    torch.cuda.synchronize()

    from torch.profiler import profile, ProfilerActivity
    t0 = time.time()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        out = llm.generate([prompt], greedy)
    torch.cuda.synchronize()
    wall = time.time() - t0
    n_out = len(out[0].outputs[0].token_ids)
    print(f"[profiler] generated {n_out} tokens in {wall:.2f}s "
          f"(eager TPS {n_out/wall:.1f}; absolute is eager, graphs OFF)", flush=True)

    # Aggregate device (CUDA) time per kernel.
    rows = []
    for e in prof.key_averages():
        dev = getattr(e, "device_time_total", None)
        if dev is None:
            dev = getattr(e, "cuda_time_total", 0.0)
        if dev and dev > 0:
            rows.append((e.key, float(dev), int(getattr(e, "count", 0))))
    rows.sort(key=lambda r: r[1], reverse=True)
    total = sum(r[1] for r in rows) or 1.0

    cats: dict[str, float] = {}
    for name, dev, _ in rows:
        cats[categorize(name)] = cats.get(categorize(name), 0.0) + dev

    print("\n==== CUDA TIME BY CATEGORY (eager; share of total device time) ====", flush=True)
    for cat, dev in sorted(cats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat:18s} {dev/1000:10.2f} ms   {100*dev/total:5.1f}%", flush=True)

    print("\n==== TOP 30 KERNELS BY CUDA TIME ====", flush=True)
    print(f"  {'pct':>5s} {'ms':>10s} {'count':>7s}  kernel", flush=True)
    top = []
    for name, dev, cnt in rows[:30]:
        print(f"  {100*dev/total:5.1f} {dev/1000:10.2f} {cnt:7d}  {name[:90]}", flush=True)
        top.append({"kernel": name, "cuda_ms": dev / 1000, "count": cnt,
                    "pct": 100 * dev / total, "category": categorize(name)})

    breakdown = {
        "model": MODEL_ID, "mode": "enforce_eager (graphs off)",
        "gen_tokens": n_out, "eager_wall_s": wall, "eager_tps": n_out / wall,
        "total_cuda_ms": total / 1000,
        "category_ms": {k: v / 1000 for k, v in cats.items()},
        "category_pct": {k: 100 * v / total for k, v in cats.items()},
        "top_kernels": top,
    }
    try:
        with open(os.path.join(STATE, "profile_breakdown.json"), "w") as f:
            json.dump(breakdown, f, indent=2)
        print(f"\n[profiler] wrote {STATE}/profile_breakdown.json", flush=True)
    except Exception as e:  # /state may be read-only in some setups; stdout is the fallback
        print(f"[profiler] could not write /state ({e}); breakdown is in stdout above", flush=True)

    print("\n[profiler] done — exiting (harness will report 'server exited before "
          "readiness'; that is expected for a profile-only run).", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
