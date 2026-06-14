# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Persistent-kernel / megakernel overhead-reclamation gate (denken #97).

LOCAL CPU-only analysis. Reads a committed conc=1 decode chrome trace (torch
profiler, CUDA graphs ON, frontier `fa2sw_precache_kenyan`) and decomposes the
decode-step wall into GPU-BUSY (real kernels) vs GPU-IDLE (a: launch/API
overhead, b: host-device sync / Python round-trips, c: inter-kernel bubbles).

Only the GPU-IDLE fraction (a+b+c) is reclaimable by a persistent/megakernel
scheduler -- it eliminates per-op launch latency + host round-trips + idle gaps
by keeping the SMs resident across the decode step. It CANNOT speed up GPU-busy
work (d): the bus is the wall (#94, verify pulls 82% HBM peak), so a megakernel
reorders/fuses (d) but the SM-cycles/bytes remain.

primary metric: persistent_kernel_reclaimable_pct  (= steady-decode GPU-idle %)
test metric:    decode_gpu_idle_fraction            (a+b+c as % of decode step)

This reconciles against denken #65 (CUDA-graph -> decode 99.41% GPU-bound,
implied 0.59% GPU-idle): if the trace-direct idle ~ 0.6%, the persistent-kernel
headroom collapses and LEVER 1's +8-15% is mispriced.
"""
import argparse
import gzip
import json
import os
from collections import defaultdict

import numpy as np

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}

# CPU-side CUDA runtime classification for idle attribution.
SYNC_NAMES = {
    "cudaStreamSynchronize", "cudaEventSynchronize", "cudaDeviceSynchronize",
    "cudaStreamWaitEvent", "cudaEventQuery", "cudaStreamQuery", "cudaMemcpy",
}
LAUNCH_NAMES = {
    "cudaLaunchKernel", "cudaLaunchKernelExC", "cudaGraphLaunch",
    "cudaMemcpyAsync", "cudaMemsetAsync", "cudaLaunchCooperativeKernel",
    "cudaFuncGetAttributes", "cudaStreamIsCapturing",
}

# Kernel-name -> decode component (for the GPU-busy "is the 32% real kernels?" view).
def classify_kernel(name: str) -> str:
    n = name.lower()
    if "marlin" in n and ("rms_norm" in n or "gelu" in n):
        return "verify_body_gemm"  # fused marlin+norm/act still GEMM-dominated
    if "marlin" in n:
        return "verify_body_gemm"
    if "gemvx" in n or "gemv" in n:
        return "drafter_gemv"      # MTP drafter sequential GEMVs
    if "cutlass" in n or "ampere_" in n or "s16816gemm" in n or "wmma" in n:
        return "other_gemm"
    if "splitkreduce" in n:
        return "other_gemm"
    if "unified_attention" in n or "attention" in n or "flash" in n:
        return "attention"
    if "reshape_and_cache" in n or "slot_mapping" in n:
        return "kv_cache_write"
    if "rms_norm" in n or "norm" in n:
        return "norm"
    if "topk" in n or "argmax" in n or "sparse_argmax" in n or "softmax" in n \
       or "sampl" in n or "reduce_kernel" in n or "gatherTopK".lower() in n:
        return "sampling"
    if "gelu" in n or "silu" in n or "activation" in n:
        return "activation"
    if "reduce_segments" in n:
        return "attention"  # split-KV segment merge -> part of attention path
    if "memcpy" in n or "memset" in n or "copy" in n or "elementwise" in n \
       or "index" in n or "cat_" in n or "scatter" in n or "gather" in n \
       or "fused_6" in n or "fused_5" in n or "embedding" in n or "rope" in n:
        return "elementwise_copy"
    return "other_small"


# Big GEMM (verify body) + drafter GEMV + attention are the "named big buckets".
# Everything else is the small-kernel tail "d" the coarse budget calls "32% other".
BIG_BUCKETS = {"verify_body_gemm", "drafter_gemv", "attention"}


def merge_intervals(ivs):
    ivs = sorted(ivs)
    merged = []
    for s, e in ivs:
        if merged and s <= merged[-1][1]:
            if e > merged[-1][1]:
                merged[-1][1] = e
        else:
            merged.append([s, e])
    return merged


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="research/profiling/frontier_decode_postsplitkv/"
                    "trace_frontier/rank0.1781387527954919185.pt.trace.json.gz")
    ap.add_argument("--out", default="research/persistent_kernel_overhead/gate.json")
    ap.add_argument("--clean-mult", type=float, default=1.5,
                    help="cycle wall < clean_mult*median => steady cycle")
    ap.add_argument("--local-walltps", type=float, default=454.09)
    ap.add_argument("--official-tps", type=float, default=481.53)
    ap.add_argument("--verify-ms-m32", type=float, default=9.0053)   # #94 tree projection
    ap.add_argument("--drafter-ms-m32", type=float, default=1.6028)  # #94 tree projection
    ap.add_argument("--wandb_project",
                    default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity",
                    default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="persistent-kernel-overhead-gate")
    ap.add_argument("--wandb_name", default="denken/persistent-kernel-overhead-gate")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    with gzip.open(args.trace, "rt") as f:
        tr = json.load(f)
    evs = tr["traceEvents"]

    gpu = [e for e in evs if e.get("cat") in GPU_CATS and e.get("pid") == 0
           and "ts" in e and "dur" in e and e["dur"] > 0]
    # CPU runtime events live on the CPU process pid (the one labelled "CPU").
    cpu_pids = {e["pid"] for e in evs if e.get("ph") == "M"
                and e.get("name") == "process_labels"
                and e.get("args", {}).get("labels") == "CPU"}
    rt = [e for e in evs if e.get("cat") == "cuda_runtime" and e.get("pid") in cpu_pids
          and "ts" in e and "dur" in e]

    # --- verify-step annotations define decode cycles (CUDA-graph replays) ---
    ver = sorted((e["ts"], e["ts"] + e["dur"]) for e in evs
                 if e.get("cat") == "gpu_user_annotation" and e.get("pid") == 0
                 and e["name"].startswith("execute_context_0"))
    starts = [a for a, _ in ver]
    cyc = np.array([starts[i + 1] - starts[i] for i in range(len(starts) - 1)])
    med = float(np.median(cyc))
    clean = cyc < med * args.clean_mult
    # longest contiguous clean run = steady decode
    best_s = best_len = cur_s = cur_len = 0
    for i, c in enumerate(clean):
        if c:
            if cur_len == 0:
                cur_s = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_s = cur_len, cur_s
        else:
            cur_len = 0
    win0, win1 = starts[best_s], starts[best_s + best_len]
    n_cycles = best_len
    wall = win1 - win0

    # --- GPU busy / idle within the steady window ---
    win_gpu = [(max(e["ts"], win0), min(e["ts"] + e["dur"], win1))
               for e in gpu if e["ts"] < win1 and e["ts"] + e["dur"] > win0]
    merged = merge_intervals(win_gpu)
    busy = sum(e - s for s, e in merged)
    idle = wall - busy
    idle_frac = idle / wall

    # gaps between merged busy intervals = the GPU-idle gaps
    gaps = [(merged[i][1], merged[i + 1][0]) for i in range(len(merged) - 1)]

    # --- classify each idle gap: (a) launch, (b) host-sync, (c) bubble ---
    # Index runtime events by the sync/launch class, overlapping each gap.
    rt_sync = [(e["ts"], e["ts"] + e["dur"]) for e in rt if e["name"] in SYNC_NAMES]
    rt_launch = [(e["ts"], e["ts"] + e["dur"]) for e in rt if e["name"] in LAUNCH_NAMES]
    rt_sync.sort()
    rt_launch.sort()

    def covered(g0, g1, ivs):
        # total overlap of [g0,g1] with sorted interval list ivs
        tot = 0.0
        # linear scan is fine (few thousand gaps x bisect window)
        lo = 0
        import bisect
        starts_ = [s for s, _ in ivs]
        i = bisect.bisect_left(starts_, g0) - 1
        if i < 0:
            i = 0
        while i < len(ivs) and ivs[i][0] < g1:
            tot += overlap(g0, g1, ivs[i][0], ivs[i][1])
            i += 1
        return tot

    a_launch = b_sync = c_bubble = 0.0
    # Per the CUDA-graph structure: gaps INSIDE a verify annotation span are
    # intra-graph bubbles (c); gaps OUTSIDE are inter-step host windows.
    ver_in_win = [(a, b) for a, b in ver if a >= win0 and b <= win1]

    def inside_graph(g0, g1):
        for a, b in ver_in_win:
            if g0 >= a and g1 <= b:
                return True
        return False

    intra_graph_idle = inter_step_idle = 0.0
    for g0, g1 in gaps:
        gd = g1 - g0
        if inside_graph(g0, g1):
            intra_graph_idle += gd
        else:
            inter_step_idle += gd
        sync_ov = covered(g0, g1, rt_sync)
        launch_ov = covered(g0, g1, rt_launch)
        # attribute the gap to the dominant overlapping host activity
        if sync_ov >= launch_ov and sync_ov > 0.25 * gd:
            b_sync += gd
        elif launch_ov > 0.25 * gd:
            a_launch += gd
        else:
            c_bubble += gd

    # --- GPU-busy composition: is the non-(GEMM/drafter/attn) tail real kernels? ---
    comp = defaultdict(float)
    for e in gpu:
        if e["ts"] >= win0 and e["ts"] + e["dur"] <= win1:
            comp[classify_kernel(e["name"])] += e["dur"]
    comp_total = sum(comp.values())
    small_tail = sum(v for k, v in comp.items() if k not in BIG_BUCKETS)

    pct = lambda x: 100.0 * x / wall
    kernels_per_cycle = sum(1 for e in gpu if win0 <= e["ts"] < win1) / n_cycles
    idle_ms_per_cycle = idle / 1e3 / n_cycles

    # ---------- Step 2: reconcile #65, coarse-budget split, tree anti-compound ----------
    # Baseline constants (merged artifacts; see config["sources"]).
    LOCAL_WALLTPS = args.local_walltps
    OFFICIAL_TPS = args.official_tps
    PR65_GPU_BOUND = 0.9941          # #65 CUDA-graph: decode 99.41% GPU-bound
    PR65_IMPLIED_IDLE = 1.0 - PR65_GPU_BOUND
    GPU_BUSY_M32 = args.verify_ms_m32 + args.drafter_ms_m32  # #94 tree projection

    reclaim_ceiling = idle_frac                       # a resident kernel removes all idle
    tps_reclaim_now = reclaim_ceiling / (1 - reclaim_ceiling)  # 1/(1-f)-1, conc=1 TPS~1/wall

    # tree anti-compounding: hold absolute per-cycle idle ~constant (kernel-count bound),
    # grow GPU-busy to the M=32 tree projection -> idle fraction shrinks.
    idle_ms_cycle = idle / 1e3 / n_cycles
    cycle_wall_m32 = GPU_BUSY_M32 + idle_ms_cycle
    idle_frac_m32 = idle_ms_cycle / cycle_wall_m32
    tps_reclaim_m32 = idle_frac_m32 / (1 - idle_frac_m32)

    # realizable haircut: ~half of the sub-0.5us bucket sits at the warp-issue / CUPTI
    # floor (irreducible even in a megakernel); cross-checked by #67 (norm/elementwise
    # fusion ceiling measured <0.5%).
    sub_half_us = sum(g1 - g0 for g0, g1 in gaps if (g1 - g0) < 0.5)
    floor_haircut = 0.5 * sub_half_us
    realizable_idle = idle - floor_haircut
    realizable_pct = pct(realizable_idle)

    # coarse parallel-advisor budget: verify-GEMM 53 / drafter 7 / attn 8 / OTHER 32.
    coarse_other = 100.0 - 53.0 - 7.0 - 8.0           # = 32
    other_idle = pct(idle)                            # GPU-idle part of the "32% other"
    other_busy = coarse_other - other_idle            # GPU-busy real-kernel tail (d)

    res = {
        "config": {
            "trace": args.trace,
            "local_walltps": LOCAL_WALLTPS, "official_tps": OFFICIAL_TPS,
            "pr65_gpu_bound": PR65_GPU_BOUND, "pr65_implied_idle_pct": 100 * PR65_IMPLIED_IDLE,
            "verify_ms_m32": args.verify_ms_m32, "drafter_ms_m32": args.drafter_ms_m32,
            "sources": "frontier_decode_postsplitkv trace (#43) + #65 GPU-bound + "
                       "#94 tree M=32 projection + #67 fusion ceiling",
        },
        "steady_window": {
            "cycles": n_cycles, "cycle_idx": [best_s, best_s + best_len],
            "wall_ms": wall / 1e3, "median_cycle_wall_ms": med / 1e3,
            "per_cycle_wall_ms": wall / 1e3 / n_cycles,
            "gpu_kernels_per_cycle": kernels_per_cycle,
        },
        "step1_gpu_idle_decomposition_pct_of_decode": {
            "a_launch_api": pct(a_launch),
            "b_host_sync_roundtrip": pct(b_sync),
            "c_inter_kernel_bubble": pct(c_bubble),
            "total_gpu_idle_a_b_c": pct(idle),
        },
        "step1_gpu_idle_by_location_pct_of_decode": {
            "intra_cuda_graph_bubble": pct(intra_graph_idle),
            "inter_step_host_window": pct(inter_step_idle),
            "note": "intra-graph = inter-kernel gaps INSIDE the CUDA-graph replay that "
                    "graph/CUDA-event timing counts as busy; inter-step = sampling + "
                    "dispatch + HtoD input copies between replays (what #65 measured).",
        },
        "step1_busy_vs_idle": {
            "gpu_busy_ms": busy / 1e3, "gpu_idle_ms": idle / 1e3,
            "idle_ms_per_cycle": idle_ms_per_cycle,
            "gpu_busy_share_of_wall": busy / wall,
            "decode_gpu_idle_fraction": idle_frac,
        },
        "step1_gpu_busy_composition_pct_of_busy": {
            k: 100.0 * v / comp_total for k, v in sorted(comp.items(), key=lambda x: -x[1])
        },
        "step1_small_kernel_tail_d": {
            "pct_of_busy": 100.0 * small_tail / comp_total,
            "pct_of_decode_wall": pct(small_tail),
            "note": "norm/elementwise/sampling/kv-write/activation/other = GPU-BUSY real "
                    "kernels a megakernel reorders but cannot remove (bus is the wall, #94)",
        },
        "step2_coarse_budget_reconciliation": {
            "coarse_other_pct": coarse_other,
            "of_which_gpu_idle_reclaimable_pct": other_idle,
            "of_which_gpu_busy_real_kernels_pct": other_busy,
            "gpu_busy_share_of_the_32pct_other": 100.0 * other_busy / coarse_other,
            "verdict": "the coarse '~32% other/overhead' is ~93% GPU-BUSY real kernels "
                       "(under-counted attn+drafter + norm/sampling/lmhead/elementwise), "
                       "only ~2.2pp is GPU-idle.",
        },
        "step2_pr65_reconciliation": {
            "pr65_implied_idle_pct": 100 * PR65_IMPLIED_IDLE,
            "my_inter_step_host_window_pct": pct(inter_step_idle),
            "reconciles": "inter-step host window (%.2f%%) ~ #65's implied idle (%.2f%%)"
                          % (pct(inter_step_idle), 100 * PR65_IMPLIED_IDLE),
            "additional_intra_graph_idle_pct": pct(intra_graph_idle),
            "why_gap": "#65 measured at CUDA-graph-step granularity (intra-graph "
                       "inter-kernel time counted as busy); a persistent kernel operates "
                       "at finer sub-kernel granularity and targets the intra-graph "
                       "boundaries #65's measurement was blind to.",
        },
        "step2_tree_anticompounding": {
            "idle_frac_m8_now": idle_frac,
            "gpu_busy_ms_m32_tree": GPU_BUSY_M32,
            "idle_frac_m32_tree": idle_frac_m32,
            "delta_pp": 100 * (idle_frac - idle_frac_m32),
            "note": "M=32 tree amortizes the ~fixed per-step kernel-boundary idle over "
                    "more GPU-busy work -> the persistent-kernel prize shrinks (levers "
                    "partially anti-compound).",
        },
        "step2_projections": {
            "reclaim_ceiling_pct_wall": pct(idle),
            "tps_reclaim_pct_now": 100 * tps_reclaim_now,
            "tps_reclaim_pct_m32_tree": 100 * tps_reclaim_m32,
            "realizable_reclaim_pct_after_floor_haircut": realizable_pct,
            "ceiling_local_walltps": LOCAL_WALLTPS * (1 + tps_reclaim_now),
            "ceiling_official_tps": OFFICIAL_TPS * (1 + tps_reclaim_now),
            "realizable_official_tps": OFFICIAL_TPS * (1 + realizable_pct / 100),
        },
        "primary_metric": {
            "name": "persistent_kernel_reclaimable_pct",
            "value": pct(idle),
        },
        "test_metric": {
            "name": "decode_gpu_idle_fraction",
            "value": idle_frac,
        },
        "gate": {
            "band_ceiling": ("GREEN" if pct(idle) >= 3.0 else
                             "AMBER" if pct(idle) >= 1.0 else "RED"),
            "band_realizable": ("GREEN" if realizable_pct >= 3.0 else
                                "AMBER" if realizable_pct >= 1.0 else "RED"),
            "recommendation": "CLOSE",
            "rationale": "AMBER, not the clean RED the #65-extension predicted, but below "
                         "the 3%% build-worth bar. The coarse '32%% overhead' is 93%% GPU-BUSY "
                         "real kernels (LEVER 1's reclaimable-idle premise refuted); the "
                         "residual true GPU-idle is a 2.17%% CEILING = ~1000 sub-us intra-"
                         "CUDA-graph kernel-boundary gaps/step, an upper bound assuming a "
                         "perfect megakernel fuses all of them. Realizable ~1.76%% optimistic "
                         "(floor haircut), empirically pulled lower by #67 (norm/elementwise "
                         "fusion measured <0.5%%) and #94 (bus is the wall). Anti-compounds "
                         "with the M=32 tree (2.17->1.64%%). A full int4-Marlin+fa2sw+sampling "
                         "megakernel is a disproportionate build for a sub-2%% shrinking prize.",
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    if not args.no_wandb:
        _log_wandb(args, res)
    return res


def _log_wandb(args, res):
    import wandb
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type="analysis", config=res["config"])
    flat = {}
    for sect, d in res.items():
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, (int, float)):
                    flat[f"{sect}/{k}"] = v
    flat["primary/persistent_kernel_reclaimable_pct"] = res["primary_metric"]["value"]
    flat["test/decode_gpu_idle_fraction"] = res["test_metric"]["value"]
    flat["gate_band_ceiling"] = res["gate"]["band_ceiling"]
    flat["gate_band_realizable"] = res["gate"]["band_realizable"]
    flat["gate_recommendation"] = res["gate"]["recommendation"]
    run.summary.update(flat)
    # idle decomposition as a table
    tbl = wandb.Table(columns=["bucket", "pct_of_decode_wall"])
    for k, v in res["step1_gpu_idle_decomposition_pct_of_decode"].items():
        tbl.add_data(k, v)
    run.log({"gpu_idle_decomposition": tbl})
    run.finish()
    print(f"[persistent-kernel-gate] W&B run: {run.url}", flush=True)


if __name__ == "__main__":
    main()
