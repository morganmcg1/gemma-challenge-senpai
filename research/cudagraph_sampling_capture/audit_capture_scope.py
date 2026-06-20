#!/usr/bin/env python
"""PR #809 Step-1 CAPTURE-SCOPE AUDIT.

Question (from the PR): is the MAIN/verifier model's greedy sampling step
(argmax / logits->token) INSIDE the CUDA graph, or is it EAGER with a per-token
host<->device sync that serializes the decode loop?

vLLM 0.22.0 (V1) NEVER puts the sampler inside the model-forward CUDA graph: the
forward (-> logits) is captured, then `sample_tokens()` runs the sampler/rejection
sampler OUTSIDE the graph. The latency question is therefore NOT "is sampling in
the graph" but "does sampling cost a SERIAL per-token sync, or is it overlapped?".
vLLM hides that sync with **async scheduling** (sampled tokens kept on GPU as
`prev_sampled_token_ids`; the D2H copy + the data-dependent accepted-token count
are issued on dedicated side streams and overlapped with the next step's compute;
for spec decode the count is corrected GPU-side). The blocking `.synchronize()`
calls only fire for structured-output / penalties / bad_words, which greedy
decode never uses.

This script proves the mechanism empirically for the real spec stack:
  1) Reads `scheduler_config.async_scheduling` from the constructed engine.
  2) Steady-state greedy spec decode: GPU-busy share of wall. A serial per-token
     sampling sync (the PR's premise) would drop this well below ~100%.
  3) Self-device kernel category breakdown (sampling vs gemm vs attention/...).
  4) Chrome-trace stream analysis: are the sampled-token D2H memcpys on a SEPARATE
     CUDA stream (overlap-capable) vs serializing the compute stream, and how many
     host-side cudaStreamSynchronize calls land per decoded token.

PROXY NOTE: uses the cached base int4 (google/gemma-4-E4B-it-qat-w4a16-ct) + the
gemma4_mtp drafter as the sampling-sync proxy for int4head. The only int4head
delta is the int4-quantized lm_head (a smaller per-token GEMV) -- it does NOT
touch the sampler / scheduler / sync path this PR targets, so the async-scheduling
resolution and D2H-overlap behaviour are identical to int4head. (Confirmed by
code: async_scheduling resolution in config/vllm.py depends only on the spec
method, executor, and drafter-batch flag -- never on lm_head dtype.)
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import time
from collections import defaultdict

# --- env MUST be set before torch / vllm import (CUDA init, sampler backend) ---
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # pod quirk: A10G is index 0
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # local JIT is broken
os.environ.setdefault("VLLM_BATCH_INVARIANT", "0")  # match int4head manifest (bi0)
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # in-process: profiler sees kernels
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Apply the int4head submission's sitecustomize patches (attn-group + force2d).
# serve.py puts the submission dir on PYTHONPATH so sitecustomize auto-loads; we
# also do it here so the patches' meta-path finders are installed before vllm import.
_SUB = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..",
                 "submissions", "int4_mtp_bi0_int4head")
)
if _SUB not in sys.path:
    sys.path.insert(0, _SUB)
try:
    import sitecustomize  # noqa: F401  (installs the one-shot import hooks)
    _PATCHED = True
except Exception as e:  # pragma: no cover
    print(f"[audit] WARN could not import sitecustomize: {e}", flush=True)
    _PATCHED = False

import torch  # noqa: E402

MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-4-E4B-it-qat-w4a16-ct")
DRAFTER = os.environ.get("DRAFTER_MODEL",
                         "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant")
NUM_SPEC = int(os.environ.get("NUM_SPECULATIVE_TOKENS", "6"))
STATE = os.environ.get("STATE_DIR", os.path.dirname(os.path.abspath(__file__)))
TPS_TOKENS = int(os.environ.get("TPS_TOKENS", "512"))
PROFILE_TOKENS = int(os.environ.get("PROFILE_TOKENS", "192"))
WARMUP_TOKENS = int(os.environ.get("WARMUP_TOKENS", "48"))

# kernel-name -> category (mirrors official gemma_decode_profiler categories,
# with sampling kept first-class so we can size it directly).
CATS = [
    ("sampling", ("log_softmax", "logsoftmax", "argmax", "topk", "top_k", "softmax",
                  "sample", "logits", "cumsum", "sort", "gather", "scatter",
                  "rejection", "reduce")),
    ("attention", ("attn", "_fwd", "flash", "paged", "unified_attention",
                   "reshape_and_cache", "rotary", "rope")),
    ("matmul_gemm", ("marlin", "gptq", "gemm", "cutlass", "wmma", "gemv", "splitk",
                     "split_k", "ampere", "s16816", "tensorop", "dot")),
    ("norm", ("rms", "layernorm", "layer_norm", "norm_kernel")),
    ("activation", ("silu", "gelu", "swiglu", "act_and_mul")),
    ("elementwise_copy", ("elementwise", "copy", "cast", "convert", "memcpy",
                          "fill", "vectorized")),
]
_GEMM_HINTS = ("marlin", "gemm", "gemv", "cutlass", "s16816", "tensorop", "wmma")


def categorize(name: str) -> str:
    n = name.lower()
    if any(h in n for h in _GEMM_HINTS):
        return "matmul_gemm"
    for cat, subs in CATS:
        if any(s in n for s in subs):
            return cat
    return "other"


def self_dev(e) -> float:
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(e, attr, None)
        if v is not None:
            return float(v)
    return 0.0


def analyze_trace(trace_path: str) -> dict:
    """Per-stream GPU-busy + sampled-token D2H stream/overlap from a chrome trace.

    Decisive signal: if the sampled-token Memcpy-DtoH events sit on a DIFFERENT
    CUDA stream than the dominant compute kernels, the copy is overlap-capable
    (async scheduling) rather than a serial stall on the compute stream.
    """
    opener = gzip.open if trace_path.endswith(".gz") else open
    with opener(trace_path, "rt") as f:
        data = json.load(f)
    ev = data.get("traceEvents", [])

    # GPU device-side ops: cat in {kernel, gpu_memcpy, gpu_memset}. Stream id is
    # usually in args["stream"]; fall back to tid.
    def stream_of(e):
        a = e.get("args") or {}
        return a.get("stream", e.get("tid"))

    kern_by_stream = defaultdict(float)        # stream -> kernel self-time (us)
    dtoh_by_stream = defaultdict(lambda: [0, 0.0])  # stream -> [count, us]
    htod_by_stream = defaultdict(lambda: [0, 0.0])
    span_lo, span_hi = float("inf"), float("-inf")
    for e in ev:
        if e.get("ph") != "X":
            continue
        cat = (e.get("cat") or "").lower()
        nm = (e.get("name") or "")
        dur = float(e.get("dur") or 0.0)
        ts = float(e.get("ts") or 0.0)
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            span_lo = min(span_lo, ts)
            span_hi = max(span_hi, ts + dur)
        if cat == "kernel":
            kern_by_stream[stream_of(e)] += dur
        elif cat == "gpu_memcpy":
            if "dtoh" in nm.lower() or "device->pageable" in nm.lower() \
                    or "device->pinned" in nm.lower() or "device -> host" in nm.lower():
                dtoh_by_stream[stream_of(e)][0] += 1
                dtoh_by_stream[stream_of(e)][1] += dur
            elif "htod" in nm.lower() or "host->device" in nm.lower() \
                    or "host -> device" in nm.lower():
                htod_by_stream[stream_of(e)][0] += 1
                htod_by_stream[stream_of(e)][1] += dur

    # host-side sync runtime calls
    sync_calls = sum(1 for e in ev if e.get("ph") == "X"
                     and "synchronize" in (e.get("name") or "").lower())

    compute_stream = max(kern_by_stream, key=kern_by_stream.get) if kern_by_stream else None
    dtoh_streams = sorted(dtoh_by_stream.keys(), key=lambda s: -dtoh_by_stream[s][1])
    dtoh_total = sum(v[0] for v in dtoh_by_stream.values())
    span_us = (span_hi - span_lo) if span_hi > span_lo else 0.0
    compute_busy_us = kern_by_stream.get(compute_stream, 0.0)

    # is the bulk of sampled-token D2H traffic OFF the compute stream?
    dtoh_off_compute_us = sum(v[1] for s, v in dtoh_by_stream.items() if s != compute_stream)
    dtoh_on_compute_us = dtoh_by_stream.get(compute_stream, [0, 0.0])[1]

    return {
        "trace_file": os.path.basename(trace_path),
        "num_streams_with_kernels": len(kern_by_stream),
        "compute_stream": compute_stream,
        "compute_stream_busy_us": round(compute_busy_us, 1),
        "trace_span_us": round(span_us, 1),
        "compute_stream_busy_share_of_span_pct":
            round(100 * compute_busy_us / span_us, 2) if span_us else None,
        "kernel_us_by_stream": {str(k): round(v, 1) for k, v in
                                sorted(kern_by_stream.items(), key=lambda x: -x[1])},
        "dtoh_memcpy_count_total": dtoh_total,
        "dtoh_memcpy_us_by_stream": {str(s): {"count": v[0], "us": round(v[1], 1)}
                                     for s, v in sorted(dtoh_by_stream.items(),
                                                        key=lambda x: -x[1][1])},
        "dtoh_us_off_compute_stream": round(dtoh_off_compute_us, 1),
        "dtoh_us_on_compute_stream": round(dtoh_on_compute_us, 1),
        "dtoh_overlap_capable":
            (len(dtoh_streams) > 0 and any(s != compute_stream for s in dtoh_streams)),
        "host_synchronize_calls_in_window": sync_calls,
    }


def main() -> None:
    os.makedirs(STATE, exist_ok=True)
    print(f"[audit] torch {torch.__version__}; dev {torch.cuda.get_device_name(0)}; "
          f"patches_applied={_PATCHED}", flush=True)
    print(f"[audit] model={MODEL_ID} drafter={DRAFTER} K={NUM_SPEC} "
          f"(graphs ON, in-process uniproc)", flush=True)

    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model=MODEL_ID, quantization="compressed-tensors", dtype="bfloat16",
        max_model_len=4096, gpu_memory_utilization=0.90, max_num_batched_tokens=512,
        max_num_seqs=1, enforce_eager=False, trust_remote_code=True,
        disable_log_stats=True,
        speculative_config={"model": DRAFTER, "num_speculative_tokens": NUM_SPEC},
    )
    load_s = time.time() - t0
    print(f"[audit] engine ready in {load_s:.1f}s", flush=True)

    # ---- (1) read the decisive scheduler flag straight from the engine ----
    cfg = llm.llm_engine.vllm_config
    sc = cfg.scheduler_config
    spec = cfg.speculative_config
    async_sched = bool(getattr(sc, "async_scheduling", False))
    spec_method = getattr(spec, "method", None) if spec else None
    num_spec_cfg = getattr(spec, "num_speculative_tokens", None) if spec else None
    print(f"[audit] >>> async_scheduling = {async_sched}  "
          f"(spec_method={spec_method}, num_spec={num_spec_cfg}) <<<", flush=True)

    prompt = ("Explain, step by step, how a transformer language model generates "
              "text one token at a time, and why decode is memory-bandwidth bound.")
    sp = lambda n: SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)

    llm.generate([prompt], sp(WARMUP_TOKENS))   # warmup + cudagraph capture
    torch.cuda.synchronize()

    # ---- (2) clean steady-state TPS (no profiler) ----
    t0 = time.time()
    out = llm.generate([prompt], sp(TPS_TOKENS))
    torch.cuda.synchronize()
    wall = time.time() - t0
    n = len(out[0].outputs[0].token_ids)
    tps = n / wall
    print(f"[audit] GREEDY SPEC TPS: {n} tok / {wall:.3f}s = {tps:.2f} tok/s "
          f"(single-stream)", flush=True)

    # ---- (3) profiled window: GPU-busy share + category breakdown ----
    from torch.profiler import profile, ProfilerActivity
    trace_path = os.path.join(STATE, "decode_window.pt.trace.json.gz")
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
    busy_ms = busy_us / 1000.0
    busy_share = 100 * (busy_ms / 1000) / prof_wall if prof_wall else 0.0

    try:
        prof.export_chrome_trace(trace_path)
    except Exception as e:
        print(f"[audit] WARN export_chrome_trace failed: {e}", flush=True)
        trace_path = ""

    print("\n==== STEADY-STATE GREEDY SPEC DECODE (self-device time) ====", flush=True)
    print(f"  spec TPS:                   {tps:.2f} tok/s", flush=True)
    print(f"  profiled:                   {pn} tok / {prof_wall*1000:.0f} ms wall", flush=True)
    print(f"  GPU-busy (sum self-device): {busy_ms:.1f} ms = {busy_share:.1f}% of wall "
          f"-> non-kernel/serial overhead ~{max(0,100-busy_share):.1f}%", flush=True)
    print("  --- GPU-busy by category ---", flush=True)
    for c, us in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"    {c:18s} {us/1000:9.2f} ms  {100*us/busy_us:5.1f}%", flush=True)
    print("  --- top 15 kernels ---", flush=True)
    for name, us, cnt in rows[:15]:
        print(f"    {100*us/busy_us:5.1f}% {us/1000:7.2f}ms x{cnt:<5d} [{categorize(name)}] "
              f"{name[:62]}", flush=True)

    # ---- (4) trace stream / D2H overlap analysis ----
    trace_analysis = {}
    if trace_path and os.path.exists(trace_path):
        try:
            trace_analysis = analyze_trace(trace_path)
            print("\n==== TRACE STREAM ANALYSIS (sampled-token D2H overlap) ====", flush=True)
            print(f"  streams with kernels:        {trace_analysis['num_streams_with_kernels']}", flush=True)
            print(f"  compute-stream busy share:   {trace_analysis['compute_stream_busy_share_of_span_pct']}%", flush=True)
            print(f"  DtoH memcpys (count):        {trace_analysis['dtoh_memcpy_count_total']}", flush=True)
            print(f"  DtoH us OFF compute stream:  {trace_analysis['dtoh_us_off_compute_stream']}", flush=True)
            print(f"  DtoH us ON  compute stream:  {trace_analysis['dtoh_us_on_compute_stream']}", flush=True)
            print(f"  DtoH overlap-capable:        {trace_analysis['dtoh_overlap_capable']}", flush=True)
            print(f"  host synchronize calls/win:  {trace_analysis['host_synchronize_calls_in_window']} "
                  f"(~{trace_analysis['host_synchronize_calls_in_window']/max(1,pn):.3f}/token)", flush=True)
        except Exception as e:
            print(f"[audit] WARN trace analysis failed: {e}", flush=True)
            trace_analysis = {"error": str(e)}

    result = {
        "pr": 809, "step": 1, "kind": "capture_scope_audit",
        "model": MODEL_ID, "drafter": DRAFTER, "num_spec": NUM_SPEC,
        "patches_applied": _PATCHED,
        "proxy_note": "base-int4 + gemma4_mtp; lm_head dtype does not affect the "
                      "sampler/scheduler/sync path (int4head sync behaviour identical)",
        "async_scheduling": async_sched,
        "spec_method": spec_method,
        "num_spec_cfg": num_spec_cfg,
        "engine_load_s": round(load_s, 1),
        "spec_tps": tps, "tps_tokens": n, "tps_wall_s": wall,
        "profile_tokens": pn, "profile_wall_s": prof_wall,
        "gpu_busy_ms": busy_ms,
        "gpu_busy_share_of_wall_pct": busy_share,
        "gpu_busy_per_token_ms": busy_ms / pn if pn else None,
        "category_ms": {k: v / 1000 for k, v in cats.items()},
        "category_pct": {k: 100 * v / busy_us for k, v in cats.items()},
        "sampling_pct_of_gpu_busy": 100 * cats.get("sampling", 0.0) / busy_us if busy_us else 0.0,
        "top_kernels": [{"kernel": nm, "ms": us / 1000, "count": c,
                         "pct": 100 * us / busy_us, "category": categorize(nm)}
                        for nm, us, c in rows[:25]],
        "trace_analysis": trace_analysis,
        "trace_path": trace_path,
    }
    out_json = os.path.join(STATE, "audit_result.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[audit] wrote {out_json}", flush=True)
    print("[audit] done — exiting.", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
