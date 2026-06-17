#!/usr/bin/env python
"""Decompose an AR decode profiler trace into forward vs host-side tail (#604).

Reuses denken #97's GPU-busy/idle interval method + lawine #591 component split.
Reports, for the steady decode window of the int4_g128_lmhead AR path:
  - per-step wall, GPU-busy, GPU-idle (host dispatch/launch/sync)
  - GPU-busy breakdown by component (GEMM body+head / attention / norm / sampling / ...)
  - the host-side logits->token TAIL = sampling-GPU + GPU-idle host window

--inspect first dumps kernel-name histogram + annotation names so the step
marker + classification are grounded in the actual trace. LOCAL CPU-only.
"""
from __future__ import annotations
import argparse, gzip, json
from collections import defaultdict
import numpy as np

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
SYNC_NAMES = {"cudaStreamSynchronize", "cudaEventSynchronize", "cudaDeviceSynchronize",
              "cudaStreamWaitEvent", "cudaEventQuery", "cudaStreamQuery", "cudaMemcpy"}
LAUNCH_NAMES = {"cudaLaunchKernel", "cudaLaunchKernelExC", "cudaGraphLaunch",
                "cudaMemcpyAsync", "cudaMemsetAsync", "cudaLaunchCooperativeKernel"}


def classify(name: str) -> str:
    n = name.lower()
    if "marlin" in n or "gptq" in n or "awq" in n or "machete" in n:
        return "gemm_quant"          # int4 body + int4 lm_head (stark + #593 closed)
    if any(k in n for k in ("cutlass", "ampere_", "s16816gemm", "wmma", "gemm", "gemv", "splitkreduce")):
        return "gemm_other"
    if any(k in n for k in ("attention", "flash", "_attn", "reduce_segments", "merge_attn")):
        return "attention"           # lawine #601
    if "reshape_and_cache" in n or "slot_mapping" in n or "kv_cache" in n:
        return "kv_cache_write"
    if "rms" in n or ("norm" in n and "argmax" not in n):
        return "norm"
    if any(k in n for k in ("topk", "argmax", "sampl", "softmax", "gathertopk", "multinomial", "categorical", "random", "curand", "philox")):
        return "sampling"            # <-- MY TAIL (GPU part)
    if any(k in n for k in ("gelu", "silu", "activation", "mul_", "add_")):
        return "activation"
    if any(k in n for k in ("rope", "rotary", "embedding", "embed")):
        return "embed_rope"
    if any(k in n for k in ("memcpy", "memset", "copy", "elementwise", "index", "cat_", "scatter", "gather", "fill", "stride")):
        return "elementwise_copy"
    return "other_small"


TAIL_GPU = {"sampling"}  # the only post-logits GPU work on the engine critical path


def load(trace_path):
    op = gzip.open if trace_path.endswith(".gz") else open
    with op(trace_path, "rt") as f:
        tr = json.load(f)
    return tr["traceEvents"]


def merge(ivs):
    ivs = sorted(ivs)
    out = []
    for s, e in ivs:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--marker", default="sampling",
                    help="component (from classify) that fires once/decode-step -> step marker")
    ap.add_argument("--clean-mult", type=float, default=1.5)
    ap.add_argument("--out", default="research/ar_logits_tail/trace_decomp.json")
    args = ap.parse_args()

    evs = load(args.trace)
    # GPU device events live on a pid that is NOT the labelled CPU pid.
    cpu_pids = {e["pid"] for e in evs if e.get("ph") == "M" and e.get("name") == "process_labels"
                and "CPU" in str(e.get("args", {}).get("labels", ""))}
    gpu = [e for e in evs if e.get("cat") in GPU_CATS and "ts" in e and "dur" in e and e["dur"] > 0
           and e.get("pid") not in cpu_pids]
    rt = [e for e in evs if e.get("cat") == "cuda_runtime" and e.get("pid") in cpu_pids
          and "ts" in e and "dur" in e]

    if args.inspect:
        hist = defaultdict(lambda: [0, 0.0])
        comp = defaultdict(lambda: [0, 0.0])
        for e in gpu:
            hist[e["name"]][0] += 1
            hist[e["name"]][1] += e["dur"]
            c = classify(e["name"])
            comp[c][0] += 1
            comp[c][1] += e["dur"]
        print(f"GPU events: {len(gpu)}  | runtime events: {len(rt)} | gpu pids: {sorted({e['pid'] for e in gpu})}")
        anns = sorted({e["name"] for e in evs if e.get("cat") == "gpu_user_annotation"})
        print(f"annotations ({len(anns)}):", anns[:25])
        print("\n=== component totals (count, total_ms) ===")
        for c, (n, d) in sorted(comp.items(), key=lambda x: -x[1][1]):
            print(f"  {c:18s} count={n:6d}  total={d/1000:9.3f} ms")
        print("\n=== top-30 kernels by total_ms (count, total_ms, ->component) ===")
        for nm, (n, d) in sorted(hist.items(), key=lambda x: -x[1][1])[:30]:
            print(f"  {d/1000:8.3f} ms  x{n:<5d}  [{classify(nm):16s}] {nm[:80]}")
        return

    # --- step detection via marker-component events on the GPU stream ---
    marks = sorted(e["ts"] for e in gpu if classify(e["name"]) == args.marker)
    if len(marks) < 8:
        # fallback: use the single largest recurring kernel as the per-step marker
        by_name = defaultdict(list)
        for e in gpu:
            by_name[e["name"]].append(e["ts"])
        # pick the name whose count is the modal "once-per-step" (max count with dur>large)
        cand = max(by_name.items(), key=lambda kv: len(kv[1]))
        marks = sorted(cand[1])
        print(f"[marker fallback] using kernel {cand[0][:60]} count={len(marks)}")
    marks = np.array(marks)
    periods = np.diff(marks)
    med = float(np.median(periods))
    clean = periods < med * args.clean_mult
    # longest contiguous clean run = steady decode
    best_s = best_len = cur_s = cur_len = 0
    for i, c in enumerate(clean):
        if c:
            cur_s = cur_s if cur_len else i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_s = cur_len, cur_s
        else:
            cur_len = 0
    win0, win1 = float(marks[best_s]), float(marks[best_s + best_len])
    n_steps = best_len
    wall = win1 - win0

    win_gpu = [(max(e["ts"], win0), min(e["ts"] + e["dur"], win1)) for e in gpu
               if e["ts"] < win1 and e["ts"] + e["dur"] > win0]
    merged = merge(win_gpu)
    busy = sum(e - s for s, e in merged)
    idle = wall - busy
    gaps = [(merged[i][1], merged[i + 1][0]) for i in range(len(merged) - 1)]

    # busy composition by component (restricted to steady window)
    comp = defaultdict(float)
    comp_cnt = defaultdict(int)
    for e in gpu:
        if e["ts"] >= win0 and e["ts"] + e["dur"] <= win1:
            comp[classify(e["name"])] += e["dur"]
            comp_cnt[classify(e["name"])] += 1

    # idle gap classification: launch vs sync vs bubble
    import bisect
    rt_sync = sorted((e["ts"], e["ts"] + e["dur"]) for e in rt if e["name"] in SYNC_NAMES)
    rt_launch = sorted((e["ts"], e["ts"] + e["dur"]) for e in rt if e["name"] in LAUNCH_NAMES)

    def cov(g0, g1, ivs):
        if not ivs:
            return 0.0
        starts = [s for s, _ in ivs]
        i = max(0, bisect.bisect_left(starts, g0) - 1)
        tot = 0.0
        while i < len(ivs) and ivs[i][0] < g1:
            tot += max(0.0, min(g1, ivs[i][1]) - max(g0, ivs[i][0]))
            i += 1
        return tot

    a_launch = b_sync = c_bubble = 0.0
    gap_durs = []
    for g0, g1 in gaps:
        gd = g1 - g0
        gap_durs.append(gd)
        so, lo = cov(g0, g1, rt_sync), cov(g0, g1, rt_launch)
        if so >= lo and so > 0.25 * gd:
            b_sync += gd
        elif lo > 0.25 * gd:
            a_launch += gd
        else:
            c_bubble += gd

    us = 1.0  # trace ts are microseconds
    step_ms = wall / n_steps / 1000.0
    sampling_ms_total = comp.get("sampling", 0.0)
    sampling_ms_step = sampling_ms_total / n_steps / 1000.0
    idle_ms_step = idle / n_steps / 1000.0
    # host-side TAIL upper bound = sampling GPU + ALL host idle (idle is a strict
    # upper bound for "my tail" since some idle is lawine's scheduler/graph bubble).
    tail_ub_ms_step = sampling_ms_step + idle_ms_step

    res = {
        "trace": args.trace,
        "n_steady_steps": int(n_steps),
        "window_wall_ms": wall / 1000.0,
        "step_ms_mean": step_ms,
        "decode_only_tps_from_trace": 1000.0 / step_ms,
        "gpu_busy_ms_step": busy / n_steps / 1000.0,
        "gpu_idle_ms_step": idle_ms_step,
        "gpu_busy_frac": busy / wall,
        "gpu_idle_frac": idle / wall,
        "idle_breakdown_pct_of_step": {
            "a_launch_api": a_launch / wall,
            "b_host_sync": b_sync / wall,
            "c_inter_kernel_bubble": c_bubble / wall,
        },
        "busy_component_ms_step": {k: v / n_steps / 1000.0 for k, v in sorted(comp.items(), key=lambda x: -x[1])},
        "busy_component_pct_of_step": {k: v / wall for k, v in sorted(comp.items(), key=lambda x: -x[1])},
        "busy_component_count_per_step": {k: v / n_steps for k, v in comp_cnt.items()},
        "TAIL": {
            "sampling_gpu_ms_step": sampling_ms_step,
            "sampling_gpu_pct_of_step": sampling_ms_total / wall,
            "host_idle_ms_step": idle_ms_step,
            "host_idle_pct_of_step": idle / wall,
            "tail_upper_bound_ms_step": tail_ub_ms_step,
            "tail_upper_bound_pct_of_step": tail_ub_ms_step / step_ms,
            "note": "tail_upper_bound = sampling-GPU + ALL host-idle. Idle includes lawine's scheduler/graph bubbles, so this OVER-attributes to my tail. detok+output(HTTP/JSON) run in the front-end process, off the EngineCore GPU stream -> not on this critical path.",
        },
        "n_gaps": len(gaps),
        "median_gap_us": float(np.median(gap_durs)) if gap_durs else 0.0,
    }
    from pathlib import Path
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
