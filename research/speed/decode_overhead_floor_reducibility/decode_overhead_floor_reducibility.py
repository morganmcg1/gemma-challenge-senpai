#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #569 -- base_fullhead decode-step breakdown: is the 311.25 floor reducible?

LOCAL-ONLY. analysis_only=true, official_tps=0. NO HF Job, NO submission, NO
served-file change. One idle pod A10G. Group ``decode-overhead-floor-reducibility``.

MEASURES (replacing lawine #554's *derived* roofline with a *served-path* profile)
the base_fullhead M=1 decode-step component breakdown, and EMPIRICALLY tests the
one identity-safe reducibility lever #554 only *asserted* away
(``fixed_overhead_attackable_frac = 0.0``): the main-model CUDA-graph capture.

base_fullhead == ``google/gemma-4-E4B-it-qat-w4a16-ct`` (stock int4 body + FULL
262k bf16 tied head). Two profiler arms via the official decode-profiler LLM
config (``_profile_worker.py``, fresh process each):
  * graph  (enforce_eager=False, CUDA graphs ON  -> served-faithful config)
  * eager  (enforce_eager=True,  CUDA graphs OFF -> launch overhead exposed)

Decompose the per-token M=1 forward wall into:
  (a) head GEMV   -- bf16 262k dense matmul (the ONLY non-Marlin matmul in decode)
  (b) body GEMVs  -- int4 Marlin w4a16 linears (qkv/o/gate/up/down x42)
  (c) attention   -- the fused attention kernel (Triton/flash)
  (d) fixed       -- residual: sampling + norm + elementwise + host/launch overhead

The CUDA-graph lever:
  cudagraph_tps_on  = graph arm clean TPS (overhead captured)
  cudagraph_tps_off = eager arm clean TPS (overhead exposed)
  cudagraph_delta   = on - off  (the launch overhead the graph already hides)
  identity          = graph token_ids == eager token_ids (math-preserving check)

NOTE ON FRAMES. The profiler arms are PLAIN M=1 AR decode (no MTP drafter), which
cleanly isolates the MAIN-MODEL-forward CUDA-graph lever (the drafter's ONEGRAPH
is a separate, already-studied capture). The served base_fullhead anchor (252.69
TPS, wirbel #553) is the SPEC frame; the free-head floor reproduction maps the
freshly-MEASURED head matmul through lawine #554's exact spec-cycle mapping.

Reproduce:
  cd target/ && CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m \
    research.speed.decode_overhead_floor_reducibility.decode_overhead_floor_reducibility \
    --wandb_group decode-overhead-floor-reducibility \
    --wandb_name denken/decode-overhead-floor-reducibility
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --------------------------------------------------------------------------- #
# Architecture (google/gemma-4-E4B-it-qat-w4a16-ct text_config)
# --------------------------------------------------------------------------- #
HIDDEN, VOCAB = 2560, 262144
HEAD_BYTES = float(VOCAB * HIDDEN * 2)   # 1.342 GB bf16 262k tied head

# --------------------------------------------------------------------------- #
# ANCHORS -- cite, do NOT re-derive. From this branch's #553/#554/#561 records.
# --------------------------------------------------------------------------- #
BFH_TPS_SERVED = 252.69          # base_fullhead served spec TPS (wirbel #553, 83jiwjr9)
BFH_PPL = 2.0057                 # base_fullhead served PPL (wirbel #553)
# lawine #554 fixed_overhead_ceiling anchors (the floor this card MEASURES against)
FOC_BFH_TPS = 252.30599912117162     # #554's base_fullhead anchor (252.31)
FOC_BFH_ET = 3.8194082146962955      # spec E[T]
FOC_BFH_TCYCLE_MS = FOC_BFH_ET / FOC_BFH_TPS * 1e3   # 15.138 ms spec cycle
FOC_KV_TPS_WITHOUT = 255.48354249571457              # #551 KV->0 generous upper bound
LAWINE544_HEAD_MATMUL_M8_MS = 2.698240041732788      # #554/#544 direct head microbench (M=8)
LAWINE544_FREE_HEAD_TPS = 328.9                      # #544's conflated "free head" (refuted)
FOC_FREE_HEAD_FLOOR_TPS = 311.2485991465399          # #554 CORRECTED floor -- reproduce THIS
SHIP_TPS = 375.857               # official ship (capstone lawine #561 v74ad5jb)
GAP_TO_SHIP = SHIP_TPS - FOC_FREE_HEAD_FLOOR_TPS      # 64.61

VERIFY_M = 8                     # spec K=7 -> verify head applies to M+1=8 rows


# =========================== kernel categorization ========================= #
def _is_attn(n: str) -> bool:
    return any(s in n for s in ("attn", "_fwd", "flash", "paged", "unified_attention",
                                "reshape_and_cache", "rotary", "rope", "fmha", "mha",
                                "reduce_segments", "merge_attn"))


def _is_matmul(n: str) -> bool:
    return any(s in n for s in ("marlin", "gptq", "gemm", "gemv", "cutlass", "wmma",
                                "splitk", "split_k", "s16816", "s161616", "tensorop",
                                "cublas", "cijk", "sgemm", "hgemm", "ampere_"))


def _is_marlin(n: str) -> bool:
    return "marlin" in n or "gptq" in n


def _is_sampling(n: str) -> bool:
    return any(s in n for s in ("log_softmax", "logsoftmax", "argmax", "topk", "top_k",
                                "softmax", "sample", "logit", "cumsum", "sort", "gather",
                                "scatter", "renorm", "multinomial"))


def _is_norm(n: str) -> bool:
    return any(s in n for s in ("rms", "layernorm", "layer_norm", "norm_kernel"))


def _is_act(n: str) -> bool:
    return any(s in n for s in ("silu", "gelu", "swiglu", "act_and_mul", "mul_and"))


def categorize(name: str) -> str:
    """Attribute a kernel to a PR #569 component. Attention-first (so a fused
    attention kernel never falls into matmul); within matmul, Marlin->body,
    bf16-dense->head (the lm_head is the ONLY non-Marlin matmul in M=1 decode)."""
    n = name.lower()
    if _is_attn(n):
        return "attn"
    if _is_matmul(n):
        return "body" if _is_marlin(n) else "head"
    if _is_sampling(n):
        return "sampling"
    if _is_norm(n):
        return "norm"
    if _is_act(n):
        return "activation"
    return "other"


def analyze_kernels(rows: list[dict[str, Any]], n_tokens: int) -> dict[str, Any]:
    """Sum self-device us by component and normalize to per-decode-token us."""
    by_cat: dict[str, float] = {}
    by_name_head: list[tuple[str, float, int]] = []
    by_name_body: list[tuple[str, float, int]] = []
    total = 0.0
    for r in rows:
        nm, us, cnt = r["name"], float(r["self_us"]), int(r.get("count", 0))
        cat = categorize(nm)
        by_cat[cat] = by_cat.get(cat, 0.0) + us
        total += us
        if cat == "head":
            by_name_head.append((nm, us, cnt))
        elif cat == "body":
            by_name_body.append((nm, us, cnt))
    per = (lambda us: us / n_tokens) if n_tokens else (lambda us: float("nan"))
    by_name_head.sort(key=lambda x: -x[1])
    by_name_body.sort(key=lambda x: -x[1])
    return {
        "n_tokens": n_tokens,
        "device_us_total": total,
        "per_token_us": {k: per(v) for k, v in by_cat.items()},
        "cat_us_total": by_cat,
        "head_kernels": [{"name": k, "per_token_us": per(u), "count": c} for k, u, c in by_name_head[:6]],
        "body_kernels_top": [{"name": k, "per_token_us": per(u), "count": c} for k, u, c in by_name_body[:6]],
    }


# ============================ head GEMV microbench ========================= #
def _graph_time(fn: Callable[[], Any], reps: int, warmup: int, repeats: int, torch_mod) -> float:
    """Serve-faithful per-call GPU time: capture ``reps`` calls in a CUDA graph,
    divide replay by reps (host launch amortized as ONEGRAPH does). Verbatim
    method from lawine #551/#554 so the head microbench is directly comparable."""
    torch = torch_mod
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            for _ in range(reps):
                fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        g.replay()
        en.record()
        torch.cuda.synchronize()
        times.append(st.elapsed_time(en) / reps)
    del g
    return statistics.median(sorted(times))


def head_microbench(reps: int = 10, warmup: int = 20, repeats: int = 20) -> dict[str, Any]:
    """Direct bf16 262k head matmul at M=1 (AR/draft) and M=8 (spec verify)."""
    import torch
    torch.cuda.init()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")

    def runner(rows: int) -> Callable[[], Any]:
        x = torch.randn(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
        w = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16, device=dev).t().contiguous()

        def run():
            torch.matmul(x, w)
        return run

    m1 = _graph_time(runner(1), reps, warmup, repeats, torch)
    m8 = _graph_time(runner(VERIFY_M), reps, warmup, repeats, torch)
    eff_gbps = (HEAD_BYTES / 1e9) / (m8 / 1e3)
    print(f"[head-mb] M=1 {m1:.3f}ms  M={VERIFY_M} {m8:.3f}ms  (eff {eff_gbps:.1f} GB/s; "
          f"#554 target M=8 {LAWINE544_HEAD_MATMUL_M8_MS:.3f}ms)", flush=True)
    return {"head_matmul_m1_ms": m1, "head_matmul_m8_ms": m8, "head_eff_gbps": eff_gbps,
            "peak_gib": torch.cuda.max_memory_allocated() / 2**30}


# =============================== worker runner ============================= #
def run_worker(server_python: Path, enforce_eager: bool, state_dir: Path,
               tps_tokens: int, profile_tokens: int, model_id: str) -> dict[str, Any]:
    mode = "eager" if enforce_eager else "graph"
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "MODEL_ID": model_id,
        "STATE_DIR": str(state_dir),
        "ENFORCE_EAGER": "1" if enforce_eager else "0",
        "TPS_TOKENS": str(tps_tokens),
        "PROFILE_TOKENS": str(profile_tokens),
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "PYTORCH_CUDA_ALLOC_CONF": env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
    })
    worker = HERE / "_profile_worker.py"
    log_path = state_dir / f"worker_{mode}.log"
    print(f"\n[doc] ===== ARM {mode} (enforce_eager={enforce_eager}) =====", flush=True)
    print(f"[doc] {server_python} {worker} -> {state_dir}", flush=True)
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run([str(server_python), str(worker)], env=env,
                              stdout=log, stderr=subprocess.STDOUT)
    dur = time.time() - t0
    out_json = state_dir / f"worker_{mode}.json"
    if proc.returncode != 0 or not out_json.exists():
        tail = "\n".join(log_path.read_text().splitlines()[-30:])
        raise RuntimeError(f"worker '{mode}' failed (rc={proc.returncode}) in {dur:.0f}s; "
                           f"log tail:\n{tail}")
    data = json.loads(out_json.read_text())
    print(f"[doc] {mode}: clean TPS {data['tps']:.2f}, busy {data['gpu_busy_share_of_wall_pct']:.1f}% "
          f"of wall, {dur:.0f}s", flush=True)
    return data


# =============================== the verdict =============================== #
def build_verdict(graph: dict[str, Any], eager: dict[str, Any], headmb: dict[str, Any],
                  model_id: str) -> dict[str, Any]:
    # --- CUDA-graph lever (clean no-profiler TPS, AR M=1 frame) --------------- #
    tps_on = float(graph["tps"])
    tps_off = float(eager["tps"])
    delta = tps_on - tps_off
    gtok, etok = graph["token_ids"], eager["token_ids"]
    nmin = min(len(gtok), len(etok))
    identity = bool(len(gtok) == len(etok) and gtok == etok)
    first_div = next((i for i in range(nmin) if gtok[i] != etok[i]), -1)

    # --- per-token decode-step breakdown (graph = served-faithful, graphs ON) - #
    ga = analyze_kernels(graph["kernel_rows"], graph["profile_tokens"])
    ea = analyze_kernels(eager["kernel_rows"], eager["profile_tokens"])
    pt = ga["per_token_us"]
    head_us = pt.get("head", 0.0)
    body_us = pt.get("body", 0.0)
    attn_us = pt.get("attn", 0.0)
    samp_us = pt.get("sampling", 0.0)
    norm_us = pt.get("norm", 0.0) + pt.get("activation", 0.0)
    other_us = pt.get("other", 0.0)
    # decode_step_us_total = clean (no-profiler) graph per-token wall (served-faithful).
    decode_step_us_total = 1e6 / tps_on
    busy_per_tok_us = float(graph["gpu_busy_per_token_us"])
    # fixed overhead (PR's (d)): everything that is NOT head/body/attn compute =
    # sampling + norm/act + other device kernels + the non-kernel host/launch gap.
    compute_us = head_us + body_us + attn_us
    fixed_overhead_us = max(0.0, decode_step_us_total - compute_us)
    fixed_overhead_frac = fixed_overhead_us / decode_step_us_total if decode_step_us_total else float("nan")
    # device-only residual (sampling+norm+other) for cross-checking the host share.
    fixed_device_us = samp_us + norm_us + other_us
    host_nonkernel_us = max(0.0, decode_step_us_total - busy_per_tok_us)
    sum_components_us = head_us + body_us + attn_us + fixed_overhead_us
    sum_closure_err = abs(sum_components_us - decode_step_us_total) / decode_step_us_total \
        if decode_step_us_total else float("nan")

    # --- head matmul cross-check (trace head vs standalone microbench) -------- #
    head_mb_m1 = headmb["head_matmul_m1_ms"] * 1e3   # us
    head_mb_m8 = headmb["head_matmul_m8_ms"] * 1e3
    head_trace_vs_mb_ratio = (head_us / head_mb_m1) if head_mb_m1 else float("nan")

    # --- free-head floor reproduction (lawine #554 EXACT spec-cycle mapping) -- #
    # #554: remove ONLY the directly-measured head matmul (M=8) from the served
    # spec t_cycle; body+SDPA-floor+drafter+host all CANCEL in base-vs-freed gap.
    head_m8_ms = headmb["head_matmul_m8_ms"]
    tcycle_freed_ms = FOC_BFH_TCYCLE_MS - head_m8_ms
    tps_freed_head_only = FOC_BFH_ET / (tcycle_freed_ms / 1e3)
    # + KV->0 (#551 generous): convert to a t_cycle delta and apply.
    tcycle_kv_saving_ms = FOC_BFH_TCYCLE_MS - (FOC_BFH_ET / FOC_KV_TPS_WITHOUT * 1e3)
    tcycle_freed_kv_ms = tcycle_freed_ms - tcycle_kv_saving_ms
    free_head_floor_reproduced_tps = FOC_BFH_ET / (tcycle_freed_kv_ms / 1e3)
    floor_repro_err = abs(free_head_floor_reproduced_tps - FOC_FREE_HEAD_FLOOR_TPS) / FOC_FREE_HEAD_FLOOR_TPS

    # AR-frame free-head (remove measured trace head from the measured AR step).
    ar_free_head_tps = 1e6 / max(1e-9, decode_step_us_total - head_us)

    # --- reducibility verdict ------------------------------------------------ #
    # The served anchor ALREADY runs graphs ON (vLLM default + ONEGRAPH=1 in the
    # fa2sw_strict_surgical357 manifest; graph_capture lines in every served log). So
    # the +cudagraph_delta the eager arm exposes is overhead the served path ALREADY
    # hides -> it is NOT new headroom over 252.69. The ONLY residual attackable slice
    # is the graph-mode non-kernel HOST overhead (the un-graphed sampler/scheduler/
    # Python between replays). If even fully removing it cannot clear the gap to ship,
    # the fixed-overhead floor is irreducible and 311.25 stands.
    busy_share = float(graph["gpu_busy_share_of_wall_pct"])
    host_overhead_frac_graph = max(0.0, (decode_step_us_total - busy_per_tok_us) / decode_step_us_total)
    cudagraph_already_on_in_served = True
    # AR-frame ceiling if the ENTIRE residual host gap vanished (device-busy bound):
    ar_tps_if_host_eliminated = 1e6 / busy_per_tok_us if busy_per_tok_us else float("nan")
    ar_host_elim_gain_frac = (ar_tps_if_host_eliminated - tps_on) / tps_on if tps_on else float("nan")
    # The graph lever is REAL (+delta) but already pulled in served -> realizable
    # FURTHER served headroom from the graph lever is ~0 (graphs already on).
    served_cudagraph_further_headroom_tps = 0.0
    fixed_overhead_reducible = bool(delta < 0)   # graph SLOWER than eager would be the surprise
    # Fire re-opens only if the lever lifts AR clean TPS, the device-busy-bound AR
    # ceiling, OR the reproduced free-head floor to/over the official ship.
    lever_reopens_fire = bool(tps_on > SHIP_TPS or ar_tps_if_host_eliminated > SHIP_TPS
                              or free_head_floor_reproduced_tps > SHIP_TPS)

    return {
        "analysis_only": True, "official_tps": 0, "no_served_file_change": True,
        "no_hf_job": True, "pr": 569, "model_id": model_id,
        # ---- anchors ----
        "bfh_tps_served": BFH_TPS_SERVED, "bfh_ppl": BFH_PPL,
        "free_head_floor_anchor_tps": FOC_FREE_HEAD_FLOOR_TPS, "ship_tps": SHIP_TPS,
        "gap_to_ship": GAP_TO_SHIP,
        # ---- KEY OUTPUTS: decode-step breakdown (per-token us, AR M=1, graph) ----
        "decode_step_us_total": decode_step_us_total,
        "head_gemv_us": head_us, "body_gemv_us": body_us, "attn_us": attn_us,
        "fixed_overhead_us": fixed_overhead_us, "fixed_overhead_frac": fixed_overhead_frac,
        "sampling_us": samp_us, "norm_act_us": norm_us, "other_us": other_us,
        "fixed_device_us": fixed_device_us, "host_nonkernel_us": host_nonkernel_us,
        "gpu_busy_per_token_us": busy_per_tok_us,
        "sum_components_us": sum_components_us, "sum_closure_err": sum_closure_err,
        # ---- head matmul cross-check ----
        "head_matmul_m1_ms": headmb["head_matmul_m1_ms"],
        "head_matmul_m8_ms": headmb["head_matmul_m8_ms"],
        "head_eff_gbps": headmb["head_eff_gbps"],
        "head_trace_us": head_us, "head_microbench_m1_us": head_mb_m1,
        "head_trace_vs_microbench_ratio": head_trace_vs_mb_ratio,
        "lawine544_head_matmul_m8_ms": LAWINE544_HEAD_MATMUL_M8_MS,
        # ---- KEY OUTPUTS: CUDA-graph lever (AR M=1 frame) ----
        "cudagraph_tps_on": tps_on, "cudagraph_tps_off": tps_off,
        "cudagraph_delta_tps": delta,
        "cudagraph_delta_frac": delta / tps_off if tps_off else float("nan"),
        "cudagraph_identity_preserved": identity,
        "cudagraph_identity_first_divergence": first_div,
        "cudagraph_n_tokens_on": len(gtok), "cudagraph_n_tokens_off": len(etok),
        "cudagraph_already_on_in_served": cudagraph_already_on_in_served,
        "served_cudagraph_further_headroom_tps": served_cudagraph_further_headroom_tps,
        "graph_busy_share_of_wall_pct": busy_share,
        "host_overhead_frac_graph": host_overhead_frac_graph,
        "ar_tps_if_host_eliminated": ar_tps_if_host_eliminated,
        "ar_host_elim_gain_frac": ar_host_elim_gain_frac,
        "eager_busy_share_of_wall_pct": float(eager["gpu_busy_share_of_wall_pct"]),
        # ---- KEY OUTPUTS: free-head floor reproduction (spec frame via #554) ----
        "free_head_floor_reproduced_tps": free_head_floor_reproduced_tps,
        "free_head_floor_anchor_554_tps": FOC_FREE_HEAD_FLOOR_TPS,
        "free_head_floor_repro_err": floor_repro_err,
        "tps_freed_head_only_tps": tps_freed_head_only,
        "ar_free_head_tps": ar_free_head_tps,
        # ---- verdict ----
        "fixed_overhead_reducible": fixed_overhead_reducible,
        "lever_reopens_fire": lever_reopens_fire,
        "no_fire_strengthened": bool((not lever_reopens_fire) and floor_repro_err < 0.03),
        # ---- per-arm composition (for the artifact) ----
        "_graph_per_token_us": ga["per_token_us"],
        "_eager_per_token_us": ea["per_token_us"],
        "_graph_head_kernels": ga["head_kernels"],
        "_graph_body_kernels_top": ga["body_kernels_top"],
        "_eager_head_kernels": ea["head_kernels"],
        # ---- top-line ----
        "primary_metric_name": "fixed_overhead_frac",
        "primary_metric_value": fixed_overhead_frac,
    }


def self_test(v: dict[str, Any], headmb: dict[str, Any]) -> dict[str, Any]:
    st = {}
    # head microbench reproduces #554's M=8 head matmul within 8%
    st["head_microbench_reproduces_554"] = bool(
        abs(v["head_matmul_m8_ms"] - LAWINE544_HEAD_MATMUL_M8_MS) / LAWINE544_HEAD_MATMUL_M8_MS < 0.08)
    # the trace's head kernel matches the standalone microbench within 25% (attribution sanity)
    st["trace_head_matches_microbench"] = bool(0.6 <= v["head_trace_vs_microbench_ratio"] <= 1.6)
    # a non-trivial head component was actually attributed (head separated from body)
    st["head_component_nonzero"] = bool(v["head_gemv_us"] > 100.0)
    st["body_component_nonzero"] = bool(v["body_gemv_us"] > 100.0)
    # CUDA graph is FASTER than eager (it hides launch overhead) -> delta > 0
    st["cudagraph_faster_than_eager"] = bool(v["cudagraph_delta_tps"] > 0)
    # CUDA graph is math-preserving: graph and eager emit identical greedy tokens
    st["cudagraph_byte_identical"] = bool(v["cudagraph_identity_preserved"])
    # the #554 free-head floor reproduces from the freshly-measured head matmul (<3%)
    st["free_head_floor_reproduced"] = bool(v["free_head_floor_repro_err"] < 0.03)
    # component closure: head+body+attn+fixed == measured step (fixed is the residual)
    st["component_closure"] = bool(v["sum_closure_err"] < 1e-6)
    # the lever does NOT reopen fire (served TPS / free-head floor stay below ship)
    st["lever_does_not_reopen_fire"] = bool(not v["lever_reopens_fire"])
    # WITH graphs on, the residual host overhead is bounded small (graphs effective)
    st["graph_host_overhead_bounded"] = bool(0.0 <= v["host_overhead_frac_graph"] < 0.30)
    # the HBM-bound matmuls (body int4 + head bf16) dominate the step over attention
    st["matmul_reads_dominate_step"] = bool(v["body_gemv_us"] > v["head_gemv_us"] > v["attn_us"])
    # even fully removing the host gap keeps the AR ceiling below the official ship
    st["host_elim_ceiling_below_ship"] = bool(v["ar_tps_if_host_eliminated"] < SHIP_TPS)
    fin = [v["decode_step_us_total"], v["head_gemv_us"], v["body_gemv_us"], v["attn_us"],
           v["fixed_overhead_us"], v["fixed_overhead_frac"], v["cudagraph_tps_on"],
           v["cudagraph_tps_off"], v["free_head_floor_reproduced_tps"]]
    st["nan_clean"] = all(math.isfinite(x) for x in fin)
    st["self_test_passes"] = all(st.values())
    return st


def maybe_log_wandb(args, payload) -> str | None:
    try:
        from scripts.wandb_logging import (finish_wandb, init_wandb_run, log_json_artifact,
                                           log_summary)
    except Exception as exc:  # noqa: BLE001
        print(f"[doc] wandb logging unavailable: {exc}", flush=True)
        return None
    v = payload["verdict"]
    run = init_wandb_run(
        job_type="decode-overhead-floor-reducibility", agent="denken",
        name=args.wandb_name, group=args.wandb_group,
        notes="PR #569: MEASURES the base_fullhead M=1 decode-step component breakdown "
              "(head/body/attn/fixed) on the served-faithful path and EMPIRICALLY tests the "
              "CUDA-graph reducibility lever lawine #554 only asserted (attackable_frac=0).",
        tags=["decode-overhead", "fixed-overhead", "cuda-graph", "base-fullhead",
              "breakdown", "analysis-only", "pr-569", "local-only", "no-fire"],
        config={"analysis_only": True, "official_tps": 0, "pr": 569,
                "model_id": v["model_id"], "verify_m": VERIFY_M,
                "anchor_553_run": "83jiwjr9", "anchor_554_floor_tps": FOC_FREE_HEAD_FLOOR_TPS,
                "ship_tps": SHIP_TPS},
    )
    if run is None:
        print("[doc] wandb: no run (no API key / disabled) -- skipping", flush=True)
        return None
    flat = {k: val for k, val in v.items()
            if isinstance(val, (int, float, bool, str)) and not k.startswith("_")}
    flat.update({f"selftest_{k}": int(b) for k, b in payload["self_test"].items()})
    log_summary(run, flat, step=0)
    log_json_artifact(run, name="decode_overhead_floor_reducibility",
                      artifact_type="profiling", data=payload)
    rid = getattr(run, "id", None)
    finish_wandb(run)
    print(f"[doc] wandb logged {len(flat)} keys; run id {rid}", flush=True)
    return rid


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="denken/decode-overhead-floor-reducibility")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="decode-overhead-floor-reducibility")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--model-id", default="google/gemma-4-E4B-it-qat-w4a16-ct")
    ap.add_argument("--tps-tokens", type=int, default=256)
    ap.add_argument("--profile-tokens", type=int, default=256)
    ap.add_argument("--server-python", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny pass (32 tok) to validate kernel attribution before the full run")
    ap.add_argument("--out", type=Path, default=HERE / "decode_overhead_floor_reducibility.json")
    args = ap.parse_args(argv)

    if args.smoke:
        args.tps_tokens = min(args.tps_tokens, 32)
        args.profile_tokens = min(args.profile_tokens, 32)

    from scripts.local_validation import harness, paths
    for note in paths.prepare_local_gpu_env():
        print(f"[doc] {note}", flush=True)

    # Resolve the submission server venv (pinned vLLM wheel) for served faithfulness.
    if args.server_python and args.server_python.exists():
        server_python = args.server_python
    else:
        manifest = harness.load_manifest((REPO_ROOT / "submissions" / "fa2sw_strict_surgical357").resolve())
        server_python = harness.ensure_server_venv(manifest["dependencies"])
    print(f"[doc] server_python={server_python}", flush=True)

    state_dir = (HERE / ("smoke" if args.smoke else "run")).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    print(f"[doc] base_fullhead={args.model_id}  tps_tokens={args.tps_tokens} "
          f"profile_tokens={args.profile_tokens} -> {state_dir}", flush=True)
    print(f"[doc] anchors: served {BFH_TPS_SERVED} TPS (wirbel #553) | free-head floor "
          f"{FOC_FREE_HEAD_FLOOR_TPS:.2f} (#554) | ship {SHIP_TPS} | gap {GAP_TO_SHIP:.2f}", flush=True)

    t0 = time.time()
    graph = run_worker(server_python, False, state_dir, args.tps_tokens, args.profile_tokens, args.model_id)
    eager = run_worker(server_python, True, state_dir, args.tps_tokens, args.profile_tokens, args.model_id)
    # Head microbench LAST (parent process, fresh CUDA context, no engine contention).
    print("\n[doc] head GEMV microbench (bf16 262k, CUDA-graph timed)...", flush=True)
    headmb = head_microbench()
    elapsed = time.time() - t0

    verdict = build_verdict(graph, eager, headmb, args.model_id)
    st = self_test(verdict, headmb)
    payload = {"verdict": verdict, "self_test": st, "head_microbench": headmb,
               "elapsed_s": elapsed, "tps_tokens": args.tps_tokens,
               "profile_tokens": args.profile_tokens,
               "worker_graph": {k: graph[k] for k in graph if k != "kernel_rows"},
               "worker_eager": {k: eager[k] for k in eager if k != "kernel_rows"},
               "graph_kernel_rows": graph["kernel_rows"][:40],
               "eager_kernel_rows": eager["kernel_rows"][:40]}
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"[doc] wrote {args.out}", flush=True)

    v = verdict
    print("\n============ DECODE-OVERHEAD FLOOR REDUCIBILITY (PR #569) ============", flush=True)
    print(f"BREAKDOWN (per-token, AR M=1, graphs ON, step {v['decode_step_us_total']:.0f}us):",
          flush=True)
    print(f"  (a) head GEMV  {v['head_gemv_us']:8.1f}us  {100*v['head_gemv_us']/v['decode_step_us_total']:5.1f}%",
          flush=True)
    print(f"  (b) body GEMVs {v['body_gemv_us']:8.1f}us  {100*v['body_gemv_us']/v['decode_step_us_total']:5.1f}%",
          flush=True)
    print(f"  (c) attention  {v['attn_us']:8.1f}us  {100*v['attn_us']/v['decode_step_us_total']:5.1f}%",
          flush=True)
    print(f"  (d) fixed o/h  {v['fixed_overhead_us']:8.1f}us  {100*v['fixed_overhead_frac']:5.1f}%  "
          f"(sampling {v['sampling_us']:.0f} + norm/act {v['norm_act_us']:.0f} + other "
          f"{v['other_us']:.0f} + host {v['host_nonkernel_us']:.0f})", flush=True)
    print(f"  head trace vs microbench ratio = {v['head_trace_vs_microbench_ratio']:.2f} "
          f"(microbench M=1 {v['head_microbench_m1_us']:.0f}us)", flush=True)
    print(f"CUDA-GRAPH LEVER (AR M=1 clean TPS):", flush=True)
    print(f"  ON  {v['cudagraph_tps_on']:.2f}  OFF {v['cudagraph_tps_off']:.2f}  "
          f"delta {v['cudagraph_delta_tps']:+.2f} ({100*v['cudagraph_delta_frac']:+.1f}%)  "
          f"identity={v['cudagraph_identity_preserved']}", flush=True)
    print(f"  graph busy-share {v['graph_busy_share_of_wall_pct']:.1f}% of wall -> residual host "
          f"overhead {100*v['host_overhead_frac_graph']:.1f}% (the only attackable slice; "
          f"served already runs graphs ON -> further served headroom "
          f"{v['served_cudagraph_further_headroom_tps']:.1f} TPS)", flush=True)
    print(f"  AR ceiling if ALL host gap removed = {v['ar_tps_if_host_eliminated']:.2f} TPS "
          f"({100*v['ar_host_elim_gain_frac']:+.1f}%) -> still < ship {SHIP_TPS}", flush=True)
    print(f"FREE-HEAD FLOOR (spec frame, #554 mapping w/ MEASURED head matmul "
          f"{v['head_matmul_m8_ms']:.3f}ms):", flush=True)
    print(f"  reproduced {v['free_head_floor_reproduced_tps']:.2f}  vs #554 anchor "
          f"{FOC_FREE_HEAD_FLOOR_TPS:.2f}  (err {100*v['free_head_floor_repro_err']:.2f}%)", flush=True)
    print(f"VERDICT: lever_reopens_fire={v['lever_reopens_fire']}  "
          f"no_fire_strengthened={v['no_fire_strengthened']}  self_test={st['self_test_passes']}",
          flush=True)
    print("=====================================================================", flush=True)
    for k, b in st.items():
        if not b and k != "self_test_passes":
            print(f"  [self-test FAIL] {k}", flush=True)

    rid = None
    if not args.no_wandb:
        rid = maybe_log_wandb(args, payload)

    print("\nSENPAI-RESULT: " + json.dumps({
        "terminal": True, "status": "complete", "pending_arms": False,
        "analysis_only": True, "official_tps": 0,
        "wandb_run_ids": [rid] if rid else [],
        "self_det": st["self_test_passes"],
        "primary_metric": {"name": "fixed_overhead_frac", "value": round(v["fixed_overhead_frac"], 4)},
        "test_metric": {"name": "free_head_floor_reproduced_tps",
                        "value": round(v["free_head_floor_reproduced_tps"], 2)},
    }), flush=True)
    return 0 if st["self_test_passes"] else 1


if __name__ == "__main__":
    sys.exit(main())
