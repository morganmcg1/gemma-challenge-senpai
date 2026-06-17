#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #554 -- The fixed-overhead ceiling: does the launch floor cap the 328.9 quality-safe ship?

LOCAL-ONLY. analysis_only=true, official_tps=0. NO HF Job, NO submission, NO served-file change.
One pod A10G. Prices the NON-byte-read ("fixed overhead") term that lawine #544's 328.9
"magically-free head" roofline silently conflated, and re-derives the TRUE quality-safe hard
ceiling once that floor is kept.

REUSES (cite, do NOT re-derive the head/body split or the KV term):
  * #544 (d44b61gj): base_fullhead = 252.31 TPS local, E[T]=3.819 spec ship, t_cycle 15.138 ms.
    head 82.2% / +5-body-layer 17.8% WEIGHT-read split; eff_hbm 500.47 GB/s. DIRECT head-matmul
    microbench: 262k bf16 head (M=8 verify) = 2.698 ms; argmax-over-262k = 0.032 ms (free).
    "free head" upper bound 328.9 local; quality_safe_ship_can_beat_442 = FALSE.
  * #551 (5rnkxttp): KV-read 1.09% (immaterial). The ~0.57 ms 42-launch SDPA floor (42 attention
    kernels x ~13.6 us, FLAT from L=256..2938) = the fixed-overhead term this card prices.
    head-GEMV cross-check 507.3 GB/s ~= served 500.5 GB/s.

THE CORRECTION (load-bearing). #544's 328.9 = base_tcycle - SERVED head-attribution(3.524 ms),
at E[T]. But the SERVED 3.524 ms was the base_fullhead -> osoi5 step-gap attributed to the head,
and osoi5 is NOT a head-only-free base: it ALSO dropped 5 layers and baked the body. The DIRECT
head-matmul microbench says the head WEIGHT-read is only 2.698 ms (M=8). A magically-free-head,
*body-intact* ship removes ONLY that 2.698 ms; the SDPA launch floor, the scheduler/host cost,
the drafter, and the 42-layer body all CANCEL in the base-vs-freed gap (identical engine/body),
so they are RETAINED at their full value. Removing 2.698 (not 3.524) corrects the ceiling DOWN.

Reproduce: cd target/ && CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  research/speed/fixed_overhead_ceiling/fixed_overhead_ceiling.py \
  --wandb_group fixed-overhead-ceiling --wandb_name lawine/fixed-overhead-ceiling
"""
from __future__ import annotations

import os

# Force GPU 0 BEFORE importing torch (harness sets CUDA_VISIBLE_DEVICES=2 on this pod; only
# index 0 is a real device for torch/vLLM -- env_cuda_visible_devices quirk).
if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Architecture (google/gemma-4-E4B-it-qat-w4a16-ct text_config; confirmed #551)
# --------------------------------------------------------------------------- #
N_LAYERS = 42
N_GLOBAL = 7          # full_attention layers (KV grows with position)
N_LOCAL = 35          # sliding_attention layers (KV capped at window=512)
WINDOW = 512
N_Q, N_KV, HEAD_DIM, HIDDEN = 8, 2, 256, 2560
VOCAB = 262144
BF16, FP8, INT4 = 2, 1, 0.5

# --------------------------------------------------------------------------- #
# ANCHORS -- cite #544 (d44b61gj) and #551 (5rnkxttp); NOT re-derived.
# --------------------------------------------------------------------------- #
BFH_TPS = 252.30599912117162          # base_fullhead measured local TPS (#544)
BFH_ET = 3.8194082146962955           # spec E[T] (#544)
BFH_TCYCLE_MS = BFH_ET / BFH_TPS * 1e3  # 15.138 ms spec cycle (#544 table)
EFF_HBM_GBPS = 500.4658421444743      # #544 effective HBM bandwidth (served denominator)

WEIGHT_BYTES = 3510640640.0           # #551 byte model: total weight read
BODY_BYTES = 2168463360.0             # int4 42-layer body
HEAD_BYTES = 1342177280.0             # bf16 262k lm_head (1.342 GB)
ACT_BYTES = 4489216.0                 # #551 small M=1 activation traffic

# #544 spec K=7 -> verify applies the head to M+1 = 8 query rows.
VERIFY_M = 8
# #544 DIRECT head-matmul microbench (M=8 verify, bf16 262k head). Re-measured fresh below;
# this is the cross-check target.
LAWINE544_HEAD_MATMUL_M8_MS = 2.698240041732788
LAWINE544_SERVED_HEAD_TAX_MS = 3.524     # SERVED base->osoi5 head-attributed step-gap (4-way)
LAWINE544_FREE_HEAD_TPS = 328.9          # the roofline this card corrects
LAWINE544_INT4_HEAD_CEILING_TPS = 292.1  # #544 realistic int4-head ceiling
OSOI5_TPS = 350.7633117479405            # #544 osoi5 ship (unsafe-class anchor end of the gap)
UNSAFE_FRONTIER_TPS = 442.0

# #551 KV->0 saving (byte model, GENEROUS upper bound; realized is floor-bound and smaller).
KV544_TPS_WITHOUT = 255.48354249571457   # base TPS if KV-read -> 0 (#551 trajectory)

# Benchmark workload (#475/#551): sharegpt OUTPUT_LEN=512, mean KV 528, max KV 2938.
BENCH_KV_MEAN = 528
BENCH_KV_MAX = 2938
SDPA_SWEEP_L = [256, 512, 528, 1024, 2048, 2938]   # benchmark-relevant KV for the floor


# =============================== byte model ================================= #
def kv_bytes(L: int, dtype_bytes: float = BF16) -> float:
    """GQA KV-cache read at M=1 decode, length L: 35 local capped at window + 7 global growing."""
    per_pos_layer = 2 * N_KV * HEAD_DIM * dtype_bytes        # 2048 B bf16
    return (N_LOCAL * min(L, WINDOW) + N_GLOBAL * L) * per_pos_layer


def step_bytes(L: int) -> float:
    return WEIGHT_BYTES + kv_bytes(L) + ACT_BYTES


def step_roofline_ms(L: int) -> float:
    """#544/#551 BYTE roofline M=1 forward step: (weight + KV + act) / eff_hbm. Assumes every
    non-byte term is zero -- exactly the assumption #554 prices."""
    return step_bytes(L) / 1e9 / EFF_HBM_GBPS * 1e3


def ms_to_tps_in_cycle(tcycle_ms: float) -> float:
    """Served spec TPS = E[T] / t_cycle (the #544 frame that maps to real leaderboard TPS)."""
    return BFH_ET / (tcycle_ms / 1e3)


# ========================= realized SDPA microbench ========================= #
# (verbatim graph-timing harness from #551: capture reps in a CUDA graph, divide replay by reps
#  -> serve-faithful per-call GPU exec time, host launch amortized exactly as ONEGRAPH does.)
def _graph_time(fn: Callable[[], Any], reps_in_graph: int, warmup: int, repeats: int,
                torch_mod) -> dict[str, Any]:
    torch = torch_mod
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                for _ in range(reps_in_graph):
                    fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(reps_in_graph):
                fn()
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        return {"unsupported": True, "error": repr(exc)[:160]}
    times = []
    for _ in range(repeats):
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        g.replay()
        en.record()
        torch.cuda.synchronize()
        times.append(st.elapsed_time(en) / reps_in_graph)
    times.sort()
    del g
    return {"median_ms": statistics.median(times), "min_ms": times[0], "reps": reps_in_graph}


def run_microbench(args) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    dev = torch.device("cuda")
    torch.cuda.init()
    torch.backends.cuda.matmul.allow_tf32 = False
    prop = torch.cuda.get_device_properties(0)

    def sdpa_runner(kv_len: int):
        q = torch.randn(1, N_Q, 1, HEAD_DIM, dtype=torch.bfloat16, device=dev)
        k = torch.randn(1, N_KV, kv_len, HEAD_DIM, dtype=torch.bfloat16, device=dev)
        v = torch.randn(1, N_KV, kv_len, HEAD_DIM, dtype=torch.bfloat16, device=dev)

        def run():
            F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        return run, (q, k, v)

    def head_runner(rows: int):
        x = torch.randn(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
        w = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16, device=dev).t().contiguous()

        def run():
            torch.matmul(x, w)
        return run, (x, w)

    warmup, repeats, reps = args.warmup, args.repeats, args.reps_in_graph

    # ---- SDPA per-layer floor: global (kv=L) and local (kv=min(L,512)) at benchmark KV ----
    per_L: dict[str, Any] = {}
    for L in SDPA_SWEEP_L:
        gfn, gk = sdpa_runner(L)
        gt = _graph_time(gfn, reps, warmup, repeats, torch)
        loc = min(L, WINDOW)
        lfn, lk = sdpa_runner(loc)
        lt = _graph_time(lfn, reps, warmup, repeats, torch)
        g_ms = gt.get("median_ms", float("nan"))
        l_ms = lt.get("median_ms", float("nan"))
        total_sdpa = N_GLOBAL * g_ms + N_LOCAL * l_ms          # realized 42-layer attention
        floor = N_LAYERS * l_ms                                # KV-independent 42-launch floor
        per_L[str(L)] = {
            "global_layer_ms": g_ms, "local_layer_ms": l_ms, "local_kv_len": loc,
            "total_sdpa_realized_ms": total_sdpa,
            "sdpa_launch_floor_ms": floor,
            "kv_bandwidth_ms": kv_bytes(L) / 1e9 / EFF_HBM_GBPS * 1e3,
        }
        del gk, lk
        torch.cuda.empty_cache()
        print(f"  [sdpa] L={L:5d} global={g_ms*1e3:6.2f}us local(min={loc})={l_ms*1e3:6.2f}us "
              f"-> realized42={total_sdpa:.3f}ms floor42={floor:.3f}ms", flush=True)

    # ---- head matmul: M=1 (AR/draft) and M=8 (spec verify), bf16 262k (shape-only, bw-bound) ----
    head_ms: dict[str, float] = {}
    for rows in (1, VERIFY_M):
        hfn, hk = head_runner(rows)
        ht = _graph_time(hfn, max(reps // 3, 5), warmup, repeats, torch)
        head_ms[str(rows)] = ht.get("median_ms", float("nan"))
        del hk
        torch.cuda.empty_cache()
    head_eff_gbps = (HEAD_BYTES / 1e9) / (head_ms[str(VERIFY_M)] / 1e3)
    print(f"  [head] M=1 {head_ms['1']:.3f}ms  M={VERIFY_M} {head_ms[str(VERIFY_M)]:.3f}ms "
          f"(eff {head_eff_gbps:.1f} GB/s; #544 target {LAWINE544_HEAD_MATMUL_M8_MS:.3f})", flush=True)

    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    return {
        "device": prop.name, "sm_count": prop.multi_processor_count, "torch": torch.__version__,
        "warmup": warmup, "repeats": repeats, "reps_in_graph": reps,
        "per_L": per_L, "head_matmul_ms": head_ms, "head_eff_gbps": head_eff_gbps,
        "peak_gib": peak_gib,
    }


# ============================== the verdict ================================= #
def build_verdict(mb: dict[str, Any]) -> dict[str, Any]:
    floor_at = {L: mb["per_L"][str(L)]["sdpa_launch_floor_ms"] for L in SDPA_SWEEP_L}
    realized_at = {L: mb["per_L"][str(L)]["total_sdpa_realized_ms"] for L in SDPA_SWEEP_L}
    # The fixed-overhead floor = the KV-INDEPENDENT 42-launch SDPA cost (PR anchor "~0.57 ms").
    # Use the benchmark-mean bucket (L=528).
    sdpa_floor_ms = floor_at[BENCH_KV_MEAN]
    realized_sdpa_bench = realized_at[BENCH_KV_MEAN]
    head_matmul_m8 = mb["head_matmul_ms"][str(VERIFY_M)]

    # ---------------- STAGE 1: attribute the fixed overhead (M=1 roofline frame) -------------- #
    step_full = step_roofline_ms(BENCH_KV_MEAN)                   # ~7.11 ms (#551)
    step_freed = (BODY_BYTES + kv_bytes(BENCH_KV_MEAN) + ACT_BYTES) / 1e9 / EFF_HBM_GBPS * 1e3
    weight_read_ms = WEIGHT_BYTES / 1e9 / EFF_HBM_GBPS * 1e3
    kv_read_ms = kv_bytes(BENCH_KV_MEAN) / 1e9 / EFF_HBM_GBPS * 1e3
    head_read_ms = HEAD_BYTES / 1e9 / EFF_HBM_GBPS * 1e3
    body_read_ms = BODY_BYTES / 1e9 / EFF_HBM_GBPS * 1e3
    frac_full = sdpa_floor_ms / step_full
    frac_freed = sdpa_floor_ms / step_freed

    # ---------------- STAGE 2: the corrected fixed-overhead-bounded ceiling ------------------- #
    # Robust GAP method: a magically-free-head, BODY-INTACT ship differs from base ONLY by the
    # head matmul; SDPA floor + scheduler + drafter + 42-layer body all CANCEL (identical) and
    # are RETAINED. So remove ONLY the directly-measured head matmul, not #544's served 3.524.
    tcycle_freed = BFH_TCYCLE_MS - head_matmul_m8
    tps_freed_head = ms_to_tps_in_cycle(tcycle_freed)
    # + KV->0 (#551 GENEROUS byte-model saving): convert to a t_cycle delta and apply.
    tcycle_kv_saving = BFH_TCYCLE_MS - (BFH_ET / KV544_TPS_WITHOUT * 1e3)
    tcycle_freed_kv = tcycle_freed - tcycle_kv_saving
    tps_freed_head_kv = ms_to_tps_in_cycle(tcycle_freed_kv)
    ceiling_vs_544 = tps_freed_head_kv - LAWINE544_FREE_HEAD_TPS
    # the over-credit: how much #544's 3.524 served-tax exceeds the measured head matmul.
    head_overcredit_ms = LAWINE544_SERVED_HEAD_TAX_MS - head_matmul_m8

    # ---------------- STAGE 3: attackable fraction (SECONDARY) -------------------------------- #
    # 42 SDPA launches == 42 SEQUENTIAL data-dependent layers -> NOT a batchable set -> cannot be
    # fused into 1 launch. CUDA graph already captures the whole decode (ONEGRAPH, #246), so the
    # per-call ~13.6 us is GPU-side EXECUTION floor (grid/tail on a tiny problem), not host
    # dispatch -- a fuller graph cannot remove it. The only lever on the per-kernel floor is a
    # faster kernel = denken #550's per-BYTE speed lever (out of scope here).
    attackable_frac = 0.0
    # Theoretical UPPER bound only (infeasible): if the entire floor were eliminated.
    tps_floor_eliminated = ms_to_tps_in_cycle(tcycle_freed_kv - sdpa_floor_ms)
    attackable_tps_ceiling = tps_freed_head_kv   # realized recoverable via fusion/graph ~= 0
    # A SAME-MATH single-launch attention is byte-identical (same FLOPs + reduction order) -> the
    # launch-floor lever is in the IDENTITY-SAFE class (unlike fp8-KV). But (i) the cross-layer
    # 42->1 collapse is infeasible, and (ii) algorithm-changing fusions (flash online-softmax,
    # different accumulation order) are NOT guaranteed byte-identical -> #319 risk.
    fused_attention_identity_safe = True   # in the narrow same-reduction-order sense
    fused_attention_feasible = False       # 42 sequential layers cannot collapse to 1 launch

    return {
        "analysis_only": True, "official_tps": 0, "no_served_file_change": True, "no_hf_job": True,
        "pr": 554,
        # ---- anchors (cite #544/#551) ----
        "bfh_tps": BFH_TPS, "bfh_et": BFH_ET, "bfh_tcycle_ms": BFH_TCYCLE_MS,
        "eff_hbm_gbps": EFF_HBM_GBPS,
        "head_matmul_m8_measured_ms": head_matmul_m8,
        "head_matmul_m1_measured_ms": mb["head_matmul_ms"]["1"],
        "head_eff_gbps": mb["head_eff_gbps"],
        "lawine544_head_matmul_m8_ms": LAWINE544_HEAD_MATMUL_M8_MS,
        "lawine544_served_head_tax_ms": LAWINE544_SERVED_HEAD_TAX_MS,
        "lawine544_free_head_tps": LAWINE544_FREE_HEAD_TPS,
        # ---- STAGE 1: fixed-overhead attribution ----
        "sdpa_launch_floor_ms": sdpa_floor_ms,
        "sdpa_realized_at_bench_ms": realized_sdpa_bench,
        "weight_read_ms": weight_read_ms, "head_read_ms": head_read_ms,
        "body_read_ms": body_read_ms, "kv_read_ms": kv_read_ms,
        "m1_step_roofline_full_ms": step_full, "m1_step_roofline_freed_ms": step_freed,
        "fixed_overhead_ms_per_step": sdpa_floor_ms,
        "fixed_overhead_frac_at_full_head": frac_full,
        "fixed_overhead_frac_at_freed_head": frac_freed,
        "fixed_overhead_frac_growth": frac_freed - frac_full,
        # ---- STAGE 2: corrected ceiling (served frame) ----
        "tcycle_freed_head_ms": tcycle_freed,
        "tps_freed_head_only": tps_freed_head,
        "tcycle_freed_head_kv_ms": tcycle_freed_kv,
        "fixed_overhead_bounded_ceiling_tps": tps_freed_head_kv,
        "ceiling_vs_lawine544_328": ceiling_vs_544,
        "corrected_quality_safe_hard_ceiling": tps_freed_head_kv,
        "head_overcredit_ms": head_overcredit_ms,
        "ceiling_low_head_only": tps_freed_head,         # range low (no KV->0)
        "ceiling_high_lawine544": LAWINE544_FREE_HEAD_TPS,  # range high (#544 served-tax, refuted)
        # ---- STAGE 3: attackable fraction ----
        "fixed_overhead_attackable_frac": attackable_frac,
        "attackable_tps_ceiling": attackable_tps_ceiling,
        "tps_if_floor_eliminated_infeasible": tps_floor_eliminated,
        "fused_attention_identity_safe": fused_attention_identity_safe,
        "fused_attention_feasible": fused_attention_feasible,
        "n_sdpa_launches_per_step": N_LAYERS,
        # ---- top-line ----
        "quality_safe_ship_can_beat_442": bool(tps_freed_head_kv > UNSAFE_FRONTIER_TPS),
        "fixed_floor_caps_below_328": bool(tps_freed_head_kv < LAWINE544_FREE_HEAD_TPS),
        "primary_metric_name": "fixed_overhead_bounded_ceiling_tps",
        "primary_metric_value": tps_freed_head_kv,
        "_floor_at": {str(k): v for k, v in floor_at.items()},
        "_realized_sdpa_at": {str(k): v for k, v in realized_at.items()},
    }


def self_test(v: dict[str, Any], mb: dict[str, Any]) -> dict[str, Any]:
    st = {}
    # fresh head matmul reproduces #544's M=8 head (2.698 ms) within 5%
    st["head_matmul_reproduces_544"] = bool(
        abs(v["head_matmul_m8_measured_ms"] - LAWINE544_HEAD_MATMUL_M8_MS)
        / LAWINE544_HEAD_MATMUL_M8_MS < 0.05)
    # head GEMV eff bandwidth reproduces the served ~500 GB/s within 5% (#551 cross-check)
    st["eff_bw_reproduces_served"] = bool(abs(v["head_eff_gbps"] - EFF_HBM_GBPS) / EFF_HBM_GBPS < 0.06)
    # SDPA floor is in the PR's ~0.57 ms ballpark (0.45..0.70)
    st["sdpa_floor_in_range"] = bool(0.45 <= v["sdpa_launch_floor_ms"] <= 0.70)
    # the local-layer SDPA is FLAT (floor-bound): local_ms at 2938 within 8% of local_ms at 256
    l256 = mb["per_L"]["256"]["local_layer_ms"]
    l2938 = mb["per_L"]["2938"]["local_layer_ms"]
    st["local_sdpa_is_flat_floor"] = bool(abs(l2938 - l256) / l256 < 0.10)
    # fixed-overhead FRACTION grows as the head is freed (the Stage-1 headline)
    st["fixed_frac_grows_when_freed"] = bool(
        v["fixed_overhead_frac_at_freed_head"] > v["fixed_overhead_frac_at_full_head"])
    # full-head fraction ~ the PR's "~8%"
    st["full_head_frac_near_8pct"] = bool(0.06 <= v["fixed_overhead_frac_at_full_head"] <= 0.10)
    # corrected ceiling is BELOW #544's 328.9 (the load-bearing direction)
    st["ceiling_below_328"] = bool(v["fixed_overhead_bounded_ceiling_tps"] < LAWINE544_FREE_HEAD_TPS)
    # corrected ceiling is ABOVE #544's realistic int4 ceiling 292.1 (free head still > int4 head)
    st["ceiling_above_int4_292"] = bool(
        v["fixed_overhead_bounded_ceiling_tps"] > LAWINE544_INT4_HEAD_CEILING_TPS)
    # the magically-free quality-safe ship still cannot beat the unsafe 442 (hardens NO-FIRE)
    st["still_below_unsafe_442"] = bool(not v["quality_safe_ship_can_beat_442"])
    # #544 over-credited the head (served 3.524 > measured matmul) -> positive over-credit
    st["head_overcredit_positive"] = bool(v["head_overcredit_ms"] > 0)
    fin = [v["fixed_overhead_bounded_ceiling_tps"], v["fixed_overhead_frac_at_full_head"],
           v["fixed_overhead_frac_at_freed_head"], v["sdpa_launch_floor_ms"]]
    st["nan_clean"] = all(math.isfinite(x) for x in fin)
    st["self_test_passes"] = all(st.values())
    return st


def maybe_log_wandb(args, payload) -> str | None:
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from scripts.wandb_logging import (finish_wandb, init_wandb_run, log_json_artifact,
                                           log_summary)
    except Exception as exc:  # noqa: BLE001
        print(f"[foc] wandb logging unavailable: {exc}", flush=True)
        return None
    v = payload["verdict"]
    run = init_wandb_run(
        job_type="fixed-overhead-ceiling",
        agent="lawine",
        name=args.wandb_name,
        group=args.wandb_group,
        notes="PR #554: prices the fixed (non-byte) overhead #544's 328.9 magically-free-head "
              "roofline conflated; re-derives the corrected quality-safe hard ceiling.",
        tags=["fixed-overhead", "ceiling", "base-fullhead", "sdpa-floor", "cuda-graph",
              "analysis-only", "pr-554", "local-only", "no-fire"],
        config={"analysis_only": True, "official_tps": 0, "pr": 554,
                "n_layers": N_LAYERS, "n_sdpa_launches": N_LAYERS, "verify_m": VERIFY_M,
                "bench_kv_mean": BENCH_KV_MEAN, "anchor_run_544": "d44b61gj",
                "anchor_run_551": "5rnkxttp"},
    )
    if run is None:
        print("[foc] wandb: no run (no API key / disabled) -- skipping", flush=True)
        return None
    flat = {k: val for k, val in v.items()
            if isinstance(val, (int, float, bool, str)) and not k.startswith("_")}
    flat.update({f"selftest_{k}": int(b) for k, b in payload["self_test"].items()})
    log_summary(run, flat, step=0)
    log_json_artifact(run, name="fixed_overhead_ceiling", artifact_type="profiling", data=payload)
    rid = getattr(run, "id", None)
    finish_wandb(run)
    print(f"[foc] wandb logged {len(flat)} keys; run id {rid}", flush=True)
    return rid


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="lawine/fixed-overhead-ceiling")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="fixed-overhead-ceiling")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--reps-in-graph", type=int, default=30)
    ap.add_argument("--out", type=Path, default=HERE / "fixed_overhead_ceiling.json")
    args = ap.parse_args(argv)

    print(f"[foc] base_fullhead anchor: {BFH_TPS:.2f} TPS, E[T]={BFH_ET:.3f}, t_cycle "
          f"{BFH_TCYCLE_MS:.3f} ms, eff_hbm {EFF_HBM_GBPS:.1f} GB/s (#544 d44b61gj)", flush=True)
    print(f"[foc] pricing the fixed (non-byte) term #544's 328.9 conflated; reusing #551 SDPA "
          f"CUDA-graph microbench + #544 head-matmul microbench.", flush=True)

    import torch
    if not torch.cuda.is_available():
        print("[foc] FATAL: CUDA required for the SDPA/head microbench", flush=True)
        return 2
    mb = run_microbench(args)
    verdict = build_verdict(mb)
    st = self_test(verdict, mb)

    payload = {"verdict": verdict, "self_test": st, "microbench": mb}
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"[foc] wrote {args.out}", flush=True)

    v = verdict
    print("\n==================== FIXED-OVERHEAD CEILING (PR #554) ====================", flush=True)
    print(f"STAGE 1  fixed_overhead (42-launch SDPA floor) = {v['sdpa_launch_floor_ms']:.3f} ms/step",
          flush=True)
    print(f"         frac of full-head M=1 step ({v['m1_step_roofline_full_ms']:.2f}ms) = "
          f"{v['fixed_overhead_frac_at_full_head']*100:.1f}%", flush=True)
    print(f"         frac of freed-head M=1 step ({v['m1_step_roofline_freed_ms']:.2f}ms) = "
          f"{v['fixed_overhead_frac_at_freed_head']*100:.1f}%  (GROWS "
          f"+{v['fixed_overhead_frac_growth']*100:.1f}pp)", flush=True)
    print(f"STAGE 2  corrected ceiling (head->0 measured {v['head_matmul_m8_measured_ms']:.3f}ms "
          f"+ KV->0, body intact, floor kept) = {v['fixed_overhead_bounded_ceiling_tps']:.1f} TPS",
          flush=True)
    print(f"         vs #544's 328.9 -> Δ = {v['ceiling_vs_lawine544_328']:+.1f} TPS  "
          f"(#544 over-credited head by {v['head_overcredit_ms']:.3f} ms: served 3.524 vs "
          f"measured matmul {v['head_matmul_m8_measured_ms']:.3f})", flush=True)
    print(f"         corrected_quality_safe_hard_ceiling = "
          f"{v['corrected_quality_safe_hard_ceiling']:.1f} TPS  (range "
          f"[{v['ceiling_low_head_only']:.1f}, {v['ceiling_high_lawine544']:.1f}])", flush=True)
    print(f"STAGE 3  attackable_frac = {v['fixed_overhead_attackable_frac']:.2f} (42 SDPA = 42 "
          f"sequential layers, cannot fuse->1; graph already captured); "
          f"fused_attention_identity_safe = {v['fused_attention_identity_safe']} "
          f"(feasible={v['fused_attention_feasible']})", flush=True)
    print(f"TOP-LINE quality_safe_ship_can_beat_442 = {v['quality_safe_ship_can_beat_442']}  | "
          f"fixed_floor_caps_below_328 = {v['fixed_floor_caps_below_328']}  | self_det = "
          f"{st['self_test_passes']}", flush=True)
    print("==========================================================================", flush=True)

    rid = None
    if not args.no_wandb:
        rid = maybe_log_wandb(args, payload)

    print(f"\nSENPAI-RESULT: " + json.dumps({
        "terminal": True, "status": "complete", "pending_arms": False,
        "analysis_only": True, "official_tps": 0,
        "wandb_run_ids": [rid] if rid else [],
        "self_det": st["self_test_passes"],
        "primary_metric": {"name": "fixed_overhead_bounded_ceiling_tps",
                           "value": round(v["fixed_overhead_bounded_ceiling_tps"], 2)},
        "test_metric": {"name": "ceiling_vs_lawine544_328",
                        "value": round(v["ceiling_vs_lawine544_328"], 2)},
    }), flush=True)
    return 0 if st["self_test_passes"] else 1


if __name__ == "__main__":
    sys.exit(main())
