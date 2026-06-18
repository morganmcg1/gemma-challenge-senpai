#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #674 worker -- one int4_g128_lmhead decode arm (one cudagraph-mode boot).

Run once per cudagraph arm (a fresh process each, so the engines never share a
CUDA context). Boots vLLM on the SHIPPED int4_g128_lmhead body at the live serve
config (AR M=1, BI greedy, bf16 KV, max_model_len 4096, gpu_util 0.90,
max_num_batched_tokens 512), forces a given ``cudagraph_mode``, and emits the
three things PR #674 needs that the official profiler does not:

  1. CAPTURE AUDIT -- the resolved ``compilation_config`` (cudagraph_mode,
     cudagraph_capture_sizes, max_capture_size, splitting_ops) + device cc, so the
     parent can answer "is bs=1 captured, in which mode, is the kernel Marlin".
  2. MEDIAN-OF-N CLEAN wall-TPS (no profiler) at TPS_TOKENS, plus the greedy
     ``token_ids`` (cross-arm byte-identity / break_rate guard).
  3. (DO_PROFILE=1 only) the FULL per-DEVICE-kernel self-time list (DeviceType.CUDA
     ONLY -- PR #569's double-count fix) -> GPU-busy share -> decode_overhead_frac.

int4_g128_lmhead == Gemma-4-E4B int4 W4A16 g128 body + UNTIED int4 g128 lm_head
(group_1 targets re:.*lm_head). The int4 head reads ~335 MB/step vs base_fullhead's
1.34 GB bf16 head -> the head GEMV is ~4x cheaper, so the (constant) host bubble is
a LARGER fraction of the step here -- the gap PR #674 audits.

NOTE (authoritative, gpu_model_runner.py:4263-4316): the FULL decode cudagraph wraps
``_model_forward`` (decoder backbone -> hidden_states); ``compute_logits`` (lm_head)
and the sampler run OUTSIDE the graph, eagerly, each step.

Env: MODEL_ID, STATE_DIR, CG_MODE, DO_PROFILE(0/1), TPS_TOKENS, PROFILE_TOKENS,
N_TPS_REPS, CAPTURE_SIZES(json|''), MAX_MODEL_LEN, GPU_MEM_UTIL,
MAX_NUM_BATCHED_TOKENS, BOOT_TAG.
Writes ``$STATE_DIR/worker_<BOOT_TAG>.json`` and exits 0.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

MODEL_ID = os.environ.get("MODEL_ID", "/workspace/gemma_build/int4_g128_lmhead")
STATE = os.environ.get("STATE_DIR", "/tmp")
CG_MODE = os.environ.get("CG_MODE", "FULL_AND_PIECEWISE")   # or 'eager' (enforce_eager)
DO_PROFILE = os.environ.get("DO_PROFILE", "0") == "1"
TPS_TOKENS = int(os.environ.get("TPS_TOKENS", "256"))
PROFILE_TOKENS = int(os.environ.get("PROFILE_TOKENS", "256"))
N_TPS_REPS = int(os.environ.get("N_TPS_REPS", "3"))
CAPTURE_SIZES = os.environ.get("CAPTURE_SIZES", "").strip()
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
GPU_MEM_UTIL = float(os.environ.get("GPU_MEM_UTIL", "0.90"))
MAX_NUM_BATCHED_TOKENS = int(os.environ.get("MAX_NUM_BATCHED_TOKENS", "512"))
BOOT_TAG = os.environ.get("BOOT_TAG", CG_MODE.lower())

# Fixed greedy prompt (verbatim from PR #569 / the official profilers, so every arm
# and card decodes the same byte sequence -> cross-arm byte-identity is meaningful).
PROMPT = ("Explain, step by step, how a transformer language model generates text "
          "one token at a time, and why decode is memory-bandwidth bound.")


def _self_dev(e) -> float:
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(e, attr, None)
        if v is not None:
            return float(v)
    return 0.0


def _is_cuda_kernel(e) -> bool:
    """True iff an ACTUAL on-device kernel (DeviceType.CUDA). In this torch build's
    flat key_averages the CPU op carries the SAME self_device_time as its child
    kernel (aten::mm AND gemv2T_kernel_val; _C::marlin_gemm AND marlin::Marlin);
    summing both double-counts. DeviceType.CUDA-only counts each kernel once
    (PR #569 fix)."""
    return str(getattr(e, "device_type", "")) == "DeviceType.CUDA"


def _introspect_compilation(llm) -> dict:
    """Pull the RESOLVED cudagraph capture config off the live engine. Best-effort
    across the several wrapper layers; the parent also greps the worker log for the
    authoritative 'Capturing CUDA graphs (...)' lines as a backstop."""
    out: dict = {}
    cc = None
    for path in (
        lambda: llm.llm_engine.vllm_config.compilation_config,
        lambda: llm.llm_engine.engine_core.engine_core.vllm_config.compilation_config,
        lambda: llm.llm_engine.model_config and None,
    ):
        try:
            c = path()
            if c is not None:
                cc = c
                break
        except Exception:
            continue
    if cc is not None:
        for fld in ("cudagraph_mode", "cudagraph_capture_sizes",
                    "max_cudagraph_capture_size", "splitting_ops", "mode", "level",
                    "cudagraph_num_of_warmups"):
            try:
                v = getattr(cc, fld, None)
                out[fld] = str(v) if not isinstance(v, (list, int, float, bool, type(None))) else v
            except Exception:
                pass
        try:
            sizes = list(getattr(cc, "cudagraph_capture_sizes", []) or [])
            out["bs1_in_capture_sizes"] = (1 in sizes)
        except Exception:
            out["bs1_in_capture_sizes"] = None
    return out


def _clean_tps(llm, sp, n_tokens: int, reps: int):
    """Median-of-`reps` clean (no-profiler) wall-TPS at `n_tokens` greedy tokens.
    Returns (median_tps, [all_tps], token_ids_of_first_rep, [walls])."""
    tps_list, walls = [], []
    token_ids = None
    for r in range(reps):
        torch.cuda.synchronize()
        t0 = time.time()
        out = llm.generate([PROMPT], sp(n_tokens))
        torch.cuda.synchronize()
        wall = time.time() - t0
        tids = list(out[0].outputs[0].token_ids)
        if token_ids is None:
            token_ids = tids
        tps = len(tids) / wall if wall else float("nan")
        tps_list.append(tps)
        walls.append(wall)
        print(f"[worker:{BOOT_TAG}] clean rep {r}: {len(tids)} tok / {wall:.3f}s = {tps:.2f} tok/s",
              flush=True)
    import statistics
    return statistics.median(tps_list), tps_list, token_ids, walls


def main() -> int:
    os.makedirs(STATE, exist_ok=True)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # in-process: profiler sees kernels
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    # PR #674 locked serve config: VLLM_BATCH_INVARIANT=1 (the greedy contract the
    # 126.94 anchor was measured under). gpu_worker.init_batch_invariance() reads this
    # at worker init -> deterministic softmax/bmm/mean + IEEE fp32, TF32 OFF, no
    # reduced-precision reduction, cublaslt. It does NOT force enforce_eager and does
    # NOT disable cudagraphs, so the capture audit stays valid; it only makes every
    # arm's kernels match the shipped (slightly slower, deterministic) decode path.
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    cc, cm = torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0)
    sm = f"sm_{cc[0]}{cc[1]}"
    print(f"[worker:{BOOT_TAG}] torch {torch.__version__}; dev {cm} ({sm}); "
          f"CG_MODE={CG_MODE} DO_PROFILE={DO_PROFILE}", flush=True)

    from vllm import LLM, SamplingParams

    enforce_eager = (CG_MODE.lower() == "eager")
    kwargs = dict(
        model=MODEL_ID, quantization="compressed-tensors", dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=GPU_MEM_UTIL,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS, max_num_seqs=16, seed=0,
        enforce_eager=enforce_eager, trust_remote_code=True, disable_log_stats=True,
    )
    if not enforce_eager:
        comp: dict = {"cudagraph_mode": CG_MODE}
        if CAPTURE_SIZES:
            comp["cudagraph_capture_sizes"] = json.loads(CAPTURE_SIZES)
        kwargs["compilation_config"] = comp

    t0 = time.time()
    llm = LLM(**kwargs)
    load_s = time.time() - t0
    print(f"[worker:{BOOT_TAG}] model ready in {load_s:.1f}s", flush=True)

    audit = _introspect_compilation(llm)
    print(f"[worker:{BOOT_TAG}] capture audit: {json.dumps(audit, default=str)}", flush=True)

    sp = lambda n: SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)

    # Warmup so graphs are captured / inductor settled before timing.
    llm.generate([PROMPT], sp(16))
    torch.cuda.synchronize()

    # (2) median-of-N clean wall-TPS + greedy token_ids
    med_tps, tps_list, token_ids, walls = _clean_tps(llm, sp, TPS_TOKENS, N_TPS_REPS)
    n = len(token_ids)
    print(f"[worker:{BOOT_TAG}] MEDIAN clean TPS {med_tps:.2f} (reps {['%.2f'%x for x in tps_list]})",
          flush=True)

    result = {
        "boot_tag": BOOT_TAG, "cg_mode": CG_MODE, "enforce_eager": enforce_eager,
        "model": MODEL_ID, "device_name": cm, "sm": sm, "load_s": load_s,
        "capture_audit": audit,
        "tps_median": med_tps, "tps_all": tps_list, "tps_tokens": n, "tps_walls_s": walls,
        "token_ids": token_ids,
        "serve_cfg": {"max_model_len": MAX_MODEL_LEN, "gpu_mem_util": GPU_MEM_UTIL,
                      "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS, "max_num_seqs": 16},
    }

    # (3) profiled pass (only on the arms we breakdown) -- CUPTI sees kernels under graph replay
    if DO_PROFILE:
        from torch.profiler import ProfilerActivity, profile
        tp0 = time.time()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            pout = llm.generate([PROMPT], sp(PROFILE_TOKENS))
            torch.cuda.synchronize()
        prof_wall = time.time() - tp0
        pn = len(pout[0].outputs[0].token_ids)
        rows = [(e.key, _self_dev(e), int(getattr(e, "count", 0)))
                for e in prof.key_averages() if _is_cuda_kernel(e)]
        rows = [(k, us, c) for (k, us, c) in rows if us > 0]
        rows.sort(key=lambda r: r[1], reverse=True)
        busy_us = sum(r[1] for r in rows)
        busy_share = 100.0 * (busy_us / 1e6) / prof_wall if prof_wall else float("nan")
        cpu_rows = sorted(((e.key, _self_dev(e), int(getattr(e, "count", 0)))
                           for e in prof.key_averages()
                           if (not _is_cuda_kernel(e)) and _self_dev(e) > 0),
                          key=lambda r: r[1], reverse=True)
        print(f"[worker:{BOOT_TAG}] profiled {pn} tok / {prof_wall*1000:.0f}ms wall; "
              f"GPU-busy(device) {busy_us/1000:.1f}ms = {busy_share:.1f}% of profiled-wall",
              flush=True)
        for name, us, cnt in rows[:16]:
            print(f"    {100*us/busy_us:5.1f}% {us/1000:8.2f}ms x{cnt:<6d} {name[:74]}", flush=True)
        result.update({
            "profile_tokens": pn, "profile_wall_s": prof_wall,
            "gpu_busy_us_total": busy_us,
            "gpu_busy_share_of_profiled_wall_pct": busy_share,
            "gpu_busy_per_token_us": busy_us / pn if pn else float("nan"),
            "kernel_rows": [{"name": k, "self_us": us, "count": c} for (k, us, c) in rows],
            "excluded_cpu_op_rows": [{"name": k, "self_us": us, "count": c}
                                     for (k, us, c) in cpu_rows[:12]],
        })

    out_path = os.path.join(STATE, f"worker_{BOOT_TAG}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[worker:{BOOT_TAG}] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
