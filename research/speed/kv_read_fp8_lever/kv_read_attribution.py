#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #551 — KV-cache-read attribution on base_fullhead (the decode-step term #544 left out).

LOCAL-ONLY. analysis_only=true, official_tps=0. NO HF Job, NO submission, NO served-file
change. One pod A10G (sm_86). Stage-1 PRIMARY: attribute the M=1 decode-step KV-cache read as
a function of generated-token position, separating the 35 local sliding-window layers (KV read
BOUNDED at window=512) from the 7 global full-attention layers (KV read GROWS linearly with
position). The verdict gates whether fp8-KV is a fresh quality-safe TPS lever (Stage 2) or a
clean NO-GO that hardens #544's weight-decomposition as KV-robust.

ANCHORS (cite, do not re-derive):
  * #544 (d44b61gj): base_fullhead = 252.31 TPS local, E[T]=3.819 spec ship, full 262k bf16
    head + intact 42-layer int4 body. The base_fullhead->osoi5 GAP decomposes WEIGHT-read only:
    262k-head 82.2% / +5 body layers 17.8%; precision ceiling +38.3 -> 292.1. eff_hbm 500.47
    GB/s. The KV-read term CANCELS in that gap (same attention in both ships) -> un-attributed.
  * #475/#479 benchmark workload: official summary.json:tps runs sharegpt, OUTPUT_LEN=512,
    NUM_PROMPTS=128, MAX_CONCURRENCY=1, ignore_eos -> exactly 512 output tokens/prompt. KV at
    decode step = served_prompt_len + i. Mean KV 527.7, max KV 2938, 97.4% of decode tokens
    have KV<1024. So 16k/64k positions NEVER occur in the real workload; they are reported as
    roofline extrapolation only.

ARCHITECTURE (google/gemma-4-E4B-it-qat-w4a16-ct text config, confirmed from the local snapshot):
  42 layers, layer_types = 5 sliding : 1 full repeated -> 35 sliding(window=512) + 7 full(global);
  num_attention_heads 8, num_key_value_heads 2 (GQA), head_dim 256, hidden 2560, vocab 262144,
  intermediate 10240. Body Linears int4 pack-quantized (group_size=32, symmetric). lm_head BF16
  [262144,2560] (the 1.34 GB term that makes base_fullhead the 'slow' quality-safe ship).

METHOD
  (A) Byte model (exact, CPU): weight bytes (int4 body + bf16 head, position-independent) and
      KV bytes(L) = [35*min(L,512) + 7*L] * 2*n_kv*head_dim*dtype, split local/global. The
      decode-step KV-read FRACTION = kv_bytes(L)/step_bytes(L). Token-weighted over the REAL
      benchmark KV trajectory. kv_attributable_tps = the TPS recoverable if KV-read -> 0.
  (B) SDPA-time microbench (realized, A10G, CUDA-graph): the GQA M=1 decode attention time per
      layer for a global layer (k/v len=L) and a local layer (k/v len=min(L,512)) vs L in
      {256..65536}, plus the bf16 head GEMV time. Confirms the byte FRACTION is realized in
      TIME (the SDPA kernel is less bandwidth-efficient than the weight GEMMs, so the time
      fraction can exceed the byte fraction -- we measure by how much).

Reproduce: cd target/ && CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  research/speed/kv_read_fp8_lever/kv_read_attribution.py \
  --wandb_group kv-read-fp8-lever --wandb_name lawine/kv-read-attribution
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
# Architecture (confirmed: google/gemma-4-E4B-it-qat-w4a16-ct text_config)
# --------------------------------------------------------------------------- #
N_LAYERS = 42
GLOBAL_LAYER_IDX = [5, 11, 17, 23, 29, 35, 41]   # 'full_attention'; rest are 'sliding_attention'
N_GLOBAL = len(GLOBAL_LAYER_IDX)                  # 7
N_LOCAL = N_LAYERS - N_GLOBAL                     # 35
WINDOW = 512                                      # sliding_window
N_Q, N_KV, HEAD_DIM, HIDDEN = 8, 2, 256, 2560
VOCAB = 262144
INTERMEDIATE = 10240
GROUP_SIZE = 32                                   # int4 group quant
BF16 = 2
FP8 = 1
INT4 = 0.5

# --------------------------------------------------------------------------- #
# Anchors (#544 d44b61gj; #475/#479 benchmark workload)
# --------------------------------------------------------------------------- #
BFH_TPS = 252.30599912117162          # base_fullhead measured local TPS (#544)
BFH_ET = 3.8194082146962955           # spec E[T] (#544)
EFF_HBM_GBPS = 500.4658421444743      # #544 effective HBM bandwidth
HEAD_GAP_PCT = 82.22118525431638      # #544: head share of the base_fullhead->osoi5 gap
BODY_GAP_PCT = 17.77881474568362      # #544: +5 body-layers share of the gap
PRECISION_CEILING_TPS = 292.1008105759711   # #544 head+body precision lever ceiling
OSOI5_TPS = 350.7633117479405
UNSAFE_FRONTIER_TPS = 442.0

OUTPUT_LEN = 512
NUM_PROMPTS = 128
GEN_PROMPT_MARKER = (105, 4368, 107)  # <start_of_turn>model\n  (#475)
PPL_TOKENS = REPO_ROOT / ("official/main_bucket/shared_resources/speed_benchmark/"
                          "data/ppl_ground_truth_tokens.jsonl")

MATERIAL_FRAC = 0.10   # "material" threshold: KV-read >= 10% of the decode step
REQUESTED_POS = [256, 1024, 4096, 16384, 65536]   # instruction sweep (incl. extrapolation)
SWEEP_L = [256, 512, 1024, 2048, 2938, 4096, 8192, 16384, 32768, 65536]


# =============================== byte model ================================= #
def linear_int4_bytes(out_f: int, in_f: int) -> float:
    """Packed int4 weight + bf16 group scales (symmetric, no zero-point), group_size=32."""
    return out_f * in_f * INT4 + out_f * (in_f / GROUP_SIZE) * BF16


def body_weight_bytes() -> dict[str, float]:
    q = linear_int4_bytes(N_Q * HEAD_DIM, HIDDEN)
    k = linear_int4_bytes(N_KV * HEAD_DIM, HIDDEN)
    v = linear_int4_bytes(N_KV * HEAD_DIM, HIDDEN)
    o = linear_int4_bytes(HIDDEN, N_Q * HEAD_DIM)
    gate = linear_int4_bytes(INTERMEDIATE, HIDDEN)
    up = linear_int4_bytes(INTERMEDIATE, HIDDEN)
    down = linear_int4_bytes(HIDDEN, INTERMEDIATE)
    norms = 4 * HIDDEN * BF16            # 4 RMSNorms/layer
    per_layer = q + k + v + o + gate + up + down + norms
    return {"per_layer_bytes": per_layer, "total_body_bytes": per_layer * N_LAYERS}


def head_bytes() -> float:
    return VOCAB * HIDDEN * BF16         # bf16 lm_head 262144x2560 = 1.342 GB


def act_bytes_m1() -> float:
    """Small M=1 activation traffic: residual stream + MLP intermediate read/write per layer,
    plus the logits write. Bounded well under 1% of the step -- included for honesty."""
    per_layer = (HIDDEN * 8 + INTERMEDIATE * 2) * BF16
    logits = VOCAB * 4                   # fp32 logits write
    return per_layer * N_LAYERS + logits


def kv_bytes(L: int, dtype_bytes: float = BF16) -> dict[str, float]:
    """KV-cache read at M=1 decode, sequence length L. Per position per layer the kernel reads
    K and V for n_kv heads x head_dim (GQA: KV read once per kv-head, not per query-head)."""
    per_pos_layer = 2 * N_KV * HEAD_DIM * dtype_bytes      # 2048 B bf16 / 1024 B fp8
    local_pos = N_LOCAL * min(L, WINDOW)
    global_pos = N_GLOBAL * L
    return {
        "local_bytes": local_pos * per_pos_layer,
        "global_bytes": global_pos * per_pos_layer,
        "total_bytes": (local_pos + global_pos) * per_pos_layer,
        "per_pos_layer_bytes": per_pos_layer,
    }


def step_bytes(L: int, kv_dtype: float = BF16) -> dict[str, float]:
    body = body_weight_bytes()
    head = head_bytes()
    act = act_bytes_m1()
    kv = kv_bytes(L, kv_dtype)
    weight = body["total_body_bytes"] + head
    total = weight + kv["total_bytes"] + act
    return {
        "weight_bytes": weight, "body_bytes": body["total_body_bytes"], "head_bytes": head,
        "act_bytes": act, "kv_bytes": kv["total_bytes"],
        "kv_local_bytes": kv["local_bytes"], "kv_global_bytes": kv["global_bytes"],
        "step_bytes": total,
        "kv_read_frac": kv["total_bytes"] / total,
        "kv_local_frac": kv["local_bytes"] / total,
        "kv_global_frac": kv["global_bytes"] / total,
    }


# ========================= benchmark KV trajectory ========================== #
def _find_sub(seq, sub):
    n, m = len(seq), len(sub)
    sub = list(sub)
    for i in range(n - m + 1):
        if seq[i:i + m] == sub:
            return i
    return -1


def served_prompt_lengths() -> list[int]:
    """Chat-templated first-human-turn token length per benchmark prompt = served KV at decode
    step 0 (#475 method): PPL ground-truth context cut at the <start_of_turn>model\\n boundary."""
    out = []
    with open(PPL_TOKENS) as fh:
        for line in fh:
            rec = json.loads(line)
            ctx = rec["context_token_ids"]
            pos = _find_sub(ctx, GEN_PROMPT_MARKER)
            if pos < 0:
                raise ValueError(f"gen-prompt marker missing for id={rec.get('id')}")
            out.append(pos + len(GEN_PROMPT_MARKER))
    return sorted(out)


def trajectory_kv_weighted(prompts: list[int], kv_dtype: float = BF16) -> dict[str, Any]:
    """Token-weighted decode-step attribution over the REAL benchmark trajectory: for each of
    the 128 prompts, KV grows from P to P+511 across the 512 output tokens. Aggregate the
    KV-read fraction the way the leaderboard metric does -- per decode token (wall time per
    token) -- via the byte model. kv_attributable_tps = TPS if the KV-read term were 0,
    computed as a token-weighted harmonic over step_bytes(L) (proportional to step time)."""
    n_tok = 0
    sum_frac = 0.0
    inv_with = 0.0     # sum 1/tps  (tps ~ 1/step_bytes)
    inv_without = 0.0
    inv_fp8 = 0.0
    kv_sum = 0
    over_window = 0    # decode tokens whose KV exceeds the sliding window (local layers capped)
    # Calibrate the byte->tps map so step_bytes(meanKV) reproduces the measured 252.31 TPS.
    Ls = [p + i for p in prompts for i in range(OUTPUT_LEN)]
    mean_L = sum(Ls) / len(Ls)
    cal = BFH_TPS * step_bytes(round(mean_L), BF16)["step_bytes"]   # tps = cal/step_bytes
    for p in prompts:
        for i in range(OUTPUT_LEN):
            L = p + i
            sb = step_bytes(L, BF16)
            sb_fp8 = step_bytes(L, FP8)
            sum_frac += sb["kv_read_frac"]
            inv_with += sb["step_bytes"] / cal
            inv_without += (sb["step_bytes"] - sb["kv_bytes"]) / cal
            inv_fp8 += sb_fp8["step_bytes"] / cal
            kv_sum += L
            over_window += int(L > WINDOW)
            n_tok += 1
    tps_with = n_tok / inv_with
    tps_without = n_tok / inv_without
    tps_fp8 = n_tok / inv_fp8
    return {
        "n_decode_tokens": n_tok,
        "mean_kv": kv_sum / n_tok,
        "mean_kv_read_frac": sum_frac / n_tok,
        "frac_decode_tokens_kv_gt_window": over_window / n_tok,
        "tps_with_kv": tps_with,
        "tps_without_kv": tps_without,
        "kv_attributable_tps": tps_without - tps_with,
        "tps_fp8_kv": tps_fp8,
        "fp8kv_attributable_tps": tps_fp8 - tps_with,
        "served_prompt_mean": statistics.mean(prompts),
        "served_prompt_median": statistics.median(prompts),
        "served_prompt_max": max(prompts),
        "kv_max": max(prompts) + OUTPUT_LEN - 1,
        "byte_tps_calibration": cal,
    }


# ========================= realized SDPA microbench ========================= #
def _graph_time(fn: Callable[[], Any], reps_in_graph: int, warmup: int, repeats: int,
                torch_mod) -> dict[str, Any]:
    """Marginal per-call exec time: capture reps_in_graph back-to-back fn() into one CUDA
    graph, divide replay by reps (amortizes the launch floor -- serve-faithful)."""
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
    prop = torch.cuda.get_device_properties(0)

    def sdpa_runner(kv_len: int):
        q = torch.randn(1, N_Q, 1, HEAD_DIM, dtype=torch.bfloat16, device=dev)
        k = torch.randn(1, N_KV, kv_len, HEAD_DIM, dtype=torch.bfloat16, device=dev)
        v = torch.randn(1, N_KV, kv_len, HEAD_DIM, dtype=torch.bfloat16, device=dev)

        def run():
            F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        return run, (q, k, v)

    def head_runner():
        x = torch.randn(1, HIDDEN, dtype=torch.bfloat16, device=dev)
        w = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16, device=dev).t().contiguous()

        def run():
            torch.matmul(x, w)
        return run, (x, w)

    warmup, repeats, reps = args.warmup, args.repeats, args.reps_in_graph
    # Per-L: a GLOBAL layer reads kv_len=L; a LOCAL layer reads kv_len=min(L,512).
    per_L: dict[int, Any] = {}
    for L in SWEEP_L:
        gfn, gkeep = sdpa_runner(L)
        gt = _graph_time(gfn, reps, warmup, repeats, torch)
        loc_len = min(L, WINDOW)
        lfn, lkeep = sdpa_runner(loc_len)
        lt = _graph_time(lfn, reps, warmup, repeats, torch)
        g_ms = gt.get("median_ms", float("nan"))
        l_ms = lt.get("median_ms", float("nan"))
        total_kv_ms = N_GLOBAL * g_ms + N_LOCAL * l_ms
        per_L[L] = {
            "global_layer_ms": g_ms, "local_layer_ms": l_ms, "local_kv_len": loc_len,
            "total_kv_read_ms": total_kv_ms,
            "global_total_ms": N_GLOBAL * g_ms, "local_total_ms": N_LOCAL * l_ms,
        }
        del gkeep, lkeep
        torch.cuda.empty_cache()
        print(f"  [sdpa] L={L:6d}  global={g_ms*1e3:8.2f}us  local(min={loc_len})={l_ms*1e3:8.2f}us "
              f"-> total_kv_read={total_kv_ms:7.3f}ms (7*g + 35*l)", flush=True)

    hfn, hkeep = head_runner()
    head_t = _graph_time(hfn, max(reps // 3, 5), warmup, repeats, torch)
    head_ms = head_t.get("median_ms", float("nan"))
    del hkeep
    torch.cuda.empty_cache()
    head_eff_gbps = (head_bytes() / 1e9) / (head_ms / 1e3) if head_ms == head_ms else float("nan")
    print(f"  [head] bf16 GEMV M=1 262144x2560 = {head_ms:.3f}ms (eff {head_eff_gbps:.1f} GB/s)",
          flush=True)
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    return {
        "device": prop.name, "sm_count": prop.multi_processor_count,
        "torch": torch.__version__, "warmup": warmup, "repeats": repeats, "reps_in_graph": reps,
        "per_L": {str(k): v for k, v in per_L.items()},
        "head_gemv_ms": head_ms, "head_eff_gbps": head_eff_gbps,
        "peak_gib": peak_gib,
    }


# ================================ verdict =================================== #
def build_verdict(traj: dict[str, Any], mb: dict[str, Any] | None) -> dict[str, Any]:
    body = body_weight_bytes()
    head = head_bytes()
    weight = body["total_body_bytes"] + head
    # roofline M=1 forward step time at #544 eff_hbm (denominator for the realized check)
    step_roofline_ms = {L: step_bytes(L)["step_bytes"] / 1e9 / EFF_HBM_GBPS * 1e3 for L in SWEEP_L}

    # byte-fraction at the requested positions + the benchmark anchors
    frac_at = {L: step_bytes(L)["kv_read_frac"] for L in SWEEP_L}
    # threshold-crossing position: smallest integer L where kv_read_frac >= MATERIAL_FRAC
    cross = None
    L = WINDOW
    while L <= 1 << 24:                      # search to 16M positions
        if step_bytes(L)["kv_read_frac"] >= MATERIAL_FRAC:
            cross = L
            break
        L = int(L * 1.5) + 1
    # realized-time KV fraction from the microbench (if present)
    realized = {}
    if mb is not None:
        for L in SWEEP_L:
            kv_ms = mb["per_L"][str(L)]["total_kv_read_ms"]
            denom = step_roofline_ms[L]      # byte-model roofline forward at eff_hbm
            realized[L] = {"kv_read_ms": kv_ms, "step_roofline_ms": denom,
                           "realized_kv_time_frac": kv_ms / denom if denom else None}

    bench_frac = traj["mean_kv_read_frac"]
    bench_max_frac = step_bytes(traj["kv_max"])["kv_read_frac"]
    kv_attr_tps = traj["kv_attributable_tps"]
    fp8_attr_tps = traj["fp8kv_attributable_tps"]

    kv_material = bool(bench_frac >= MATERIAL_FRAC or bench_max_frac >= MATERIAL_FRAC)
    material_uplift = bool(fp8_attr_tps >= 5.0)   # >=5 TPS ~ 1 sigma_hw, the materiality bar
    kv_lever_is_green = bool(kv_material and material_uplift)   # identity/quality gated in Stage 2

    return {
        "analysis_only": True, "official_tps": 0, "no_served_file_change": True, "no_hf_job": True,
        # ---------- architecture ----------
        "n_layers": N_LAYERS, "n_local_sliding": N_LOCAL, "n_global_full": N_GLOBAL,
        "sliding_window": WINDOW, "n_kv_heads": N_KV, "head_dim": HEAD_DIM,
        "kv_bytes_per_pos_per_layer_bf16": kv_bytes(1)["per_pos_layer_bytes"],
        # ---------- weight anchors (#544) ----------
        "weight_bytes_total": weight, "body_bytes_int4": body["total_body_bytes"],
        "head_bytes_bf16": head, "weight_head_share": head / weight,
        "bfh_tps": BFH_TPS, "bfh_et": BFH_ET, "eff_hbm_gbps": EFF_HBM_GBPS,
        "head544_gap_pct": HEAD_GAP_PCT, "body544_gap_pct": BODY_GAP_PCT,
        "precision_ceiling_tps": PRECISION_CEILING_TPS,
        # ---------- benchmark workload ----------
        "benchmark_gen_len": OUTPUT_LEN, "num_prompts": NUM_PROMPTS,
        "benchmark_kv_mean": traj["mean_kv"], "benchmark_kv_max": traj["kv_max"],
        "benchmark_served_prompt_mean": traj["served_prompt_mean"],
        "benchmark_served_prompt_median": traj["served_prompt_median"],
        "frac_decode_tokens_kv_gt_window": traj["frac_decode_tokens_kv_gt_window"],
        # ---------- PRIMARY: KV-read fraction vs position ----------
        "kv_read_frac_at_256": frac_at[256], "kv_read_frac_at_1k": frac_at[1024],
        "kv_read_frac_at_4k": frac_at[4096], "kv_read_frac_at_16k": frac_at[16384],
        "kv_read_frac_at_64k": frac_at[65536],
        "kv_read_frac_benchmark_weighted": bench_frac,
        "kv_read_frac_at_benchmark_max": bench_max_frac,
        "material_threshold_frac": MATERIAL_FRAC,
        "kv_material_crossover_position": cross,
        "kv_material_crossover_gt_benchmark_max": bool(cross is None or cross > traj["kv_max"]),
        # ---------- attributable TPS ----------
        "kv_attributable_tps": kv_attr_tps,
        "fp8kv_attributable_tps": fp8_attr_tps,
        "tps_without_kv": traj["tps_without_kv"], "tps_fp8_kv": traj["tps_fp8_kv"],
        # ---------- realized microbench ----------
        "head_gemv_ms": mb["head_gemv_ms"] if mb else None,
        "head_eff_gbps": mb["head_eff_gbps"] if mb else None,
        "realized_kv_time_frac_benchmark_mean": (
            realized[2938]["realized_kv_time_frac"] if mb else None),  # ~benchmark max bucket
        "realized_kv_read_ms_at_benchmark_max": (
            realized[2938]["kv_read_ms"] if mb else None),
        "peak_gib": mb["peak_gib"] if mb else None,
        # ---------- GO/NO-GO ----------
        "kv_material": kv_material,
        "fp8kv_material_uplift": material_uplift,
        "kv_lever_is_green": kv_lever_is_green,
        "stage2_gated_reached": False,
        "primary_metric_name": "kv_attributable_tps",
        "primary_metric_value": kv_attr_tps,
        "_frac_at_pos": frac_at, "_step_roofline_ms": step_roofline_ms, "_realized": realized,
    }


def self_test(traj, verdict) -> dict[str, Any]:
    st = {}
    # KV byte model reconciles with the verify-roofline GQA anchor (2048 B/pos/layer bf16).
    st["kv_per_pos_layer_is_2048"] = bool(abs(verdict["kv_bytes_per_pos_per_layer_bf16"] - 2048) < 1e-6)
    # local+global layer counts partition the model
    st["layers_partition"] = bool(N_LOCAL + N_GLOBAL == N_LAYERS)
    # head is the dominant single weight term
    st["head_is_dominant_weight"] = bool(verdict["head_bytes_bf16"] > verdict["body_bytes_int4"] * 0.5)
    # KV fraction monotonically increases with position
    fr = verdict["_frac_at_pos"]
    st["kv_frac_monotone"] = bool(fr[256] < fr[1024] < fr[4096] < fr[16384] < fr[65536])
    # benchmark-weighted fraction sits between the L=256 and L=1k point (mean KV ~528)
    st["bench_frac_in_range"] = bool(fr[256] <= verdict["kv_read_frac_benchmark_weighted"] <= fr[2048]
                                     if 2048 in fr else fr[256] <= verdict["kv_read_frac_benchmark_weighted"])
    # tps_without_kv >= measured tps (removing a positive read term cannot slow you down)
    st["kv_removal_nonneg"] = bool(verdict["tps_without_kv"] >= BFH_TPS - 1e-6)
    # benchmark trajectory has the #475 shape (mean KV ~528, max ~2938)
    st["trajectory_anchored"] = bool(520 < traj["mean_kv"] < 535 and 2900 < traj["kv_max"] < 2960)
    st["n_tokens_128x512"] = bool(traj["n_decode_tokens"] == NUM_PROMPTS * OUTPUT_LEN)
    finite = [verdict["kv_read_frac_benchmark_weighted"], verdict["kv_attributable_tps"],
              verdict["fp8kv_attributable_tps"]]
    st["nan_clean"] = all(math.isfinite(x) for x in finite)
    st["self_test_passes"] = all(st.values())
    return st


def maybe_log_wandb(args, payload) -> str | None:
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from scripts.wandb_logging import (finish_wandb, init_wandb_run, log_json_artifact,
                                           log_summary)
    except Exception as exc:  # noqa: BLE001
        print(f"[kv-attr] wandb logging unavailable: {exc}", flush=True)
        return None
    v = payload["verdict"]
    run = init_wandb_run(
        job_type="kv-read-attribution",
        agent="lawine",
        name=args.wandb_name,
        group=args.wandb_group,
        notes="PR #551: M=1 decode KV-cache-read attribution on base_fullhead (35 local "
              "sliding@512 + 7 global). Completes #544's decode-step decomposition; gates fp8-KV.",
        tags=["kv-read", "attribution", "base-fullhead", "fp8-kv", "analysis-only", "pr-551",
              "local-only", "sliding-window"],
        config={"analysis_only": True, "official_tps": 0, "pr": 551,
                "n_layers": N_LAYERS, "n_local": N_LOCAL, "n_global": N_GLOBAL,
                "sliding_window": WINDOW, "n_kv": N_KV, "head_dim": HEAD_DIM,
                "benchmark_gen_len": OUTPUT_LEN, "anchor_run_544": "d44b61gj"},
    )
    if run is None:
        print("[kv-attr] wandb: no run (no API key / disabled) -- skipping", flush=True)
        return None
    flat = {k: val for k, val in v.items()
            if isinstance(val, (int, float, bool, str)) and not k.startswith("_")}
    flat.update({f"selftest_{k}": int(b) for k, b in payload["self_test"].items()})
    log_summary(run, flat, step=0)
    log_json_artifact(run, name="kv_read_attribution", artifact_type="profiling", data=payload)
    rid = getattr(run, "id", None)
    finish_wandb(run)
    print(f"[kv-attr] wandb logged {len(flat)} keys; run id {rid}", flush=True)
    return rid


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="lawine/kv-read-attribution")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="kv-read-fp8-lever")
    ap.add_argument("--no-microbench", action="store_true", help="byte model only (skip GPU)")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--reps-in-graph", type=int, default=30)
    ap.add_argument("--out", type=Path, default=HERE / "kv_read_attribution.json")
    args = ap.parse_args(argv)

    print(f"[kv-attr] arch: {N_LAYERS} layers = {N_LOCAL} local(window={WINDOW}) + "
          f"{N_GLOBAL} global | GQA n_kv={N_KV} head_dim={HEAD_DIM} | head bf16 "
          f"{head_bytes()/2**20:.0f} MiB | body int4 {body_weight_bytes()['total_body_bytes']/2**20:.0f} MiB",
          flush=True)

    prompts = served_prompt_lengths()
    traj = trajectory_kv_weighted(prompts)
    print(f"[kv-attr] benchmark: out_len={OUTPUT_LEN} prompts={len(prompts)} | mean KV "
          f"{traj['mean_kv']:.1f} max KV {traj['kv_max']} | KV-read frac (byte) mean "
          f"{traj['mean_kv_read_frac']*100:.2f}% | kv_attributable_tps {traj['kv_attributable_tps']:+.2f}",
          flush=True)

    mb = None
    if not args.no_microbench:
        try:
            import torch
            if torch.cuda.is_available():
                print("[kv-attr] running realized SDPA microbench (GQA M=1 decode, CUDA-graph)...",
                      flush=True)
                mb = run_microbench(args)
            else:
                print("[kv-attr] CUDA unavailable -- byte model only", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[kv-attr] microbench skipped ({exc!r}) -- byte model only", flush=True)

    verdict = build_verdict(traj, mb)
    st = self_test(traj, verdict)

    payload = {
        "verdict": verdict, "self_test": st, "trajectory": traj,
        "microbench": mb,
        "byte_model_per_L": {str(L): step_bytes(L) for L in SWEEP_L},
        "byte_model_per_L_fp8": {str(L): step_bytes(L, FP8) for L in SWEEP_L},
        "served_prompt_lengths": prompts,
        "analysis_only": True, "official_tps": 0, "no_served_file_change": True,
    }
    args.out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[kv-attr] wrote {args.out}", flush=True)

    rid = None if args.no_wandb else maybe_log_wandb(args, payload)

    v = verdict
    verdict_str = (
        f"KV-read {'MATERIAL' if v['kv_material'] else 'NEGLIGIBLE'} at benchmark lengths: "
        f"byte-weighted {v['kv_read_frac_benchmark_weighted']*100:.2f}% (max-outlier "
        f"{v['kv_read_frac_at_benchmark_max']*100:.2f}%); 35/42 layers sliding-window-bounded@512; "
        f"material crossover at KV~{v['kv_material_crossover_position']} "
        f"(>{v['benchmark_kv_max']:.0f} benchmark max). fp8-KV would buy "
        f"{v['fp8kv_attributable_tps']:+.2f} TPS -> kv_lever_is_green={v['kv_lever_is_green']}. "
        f"{'STOP: clean NO-GO hardens #544 weight-decomp as KV-robust.' if not v['kv_lever_is_green'] else 'PROCEED to Stage 2.'}"
    )
    v["verdict_line"] = verdict_str
    print(f"\n[kv-attr] {verdict_str}", flush=True)
    print(
        f"SENPAI-RESULT analysis_only=true official_tps=0 "
        f"kv_read_frac_at_1k={v['kv_read_frac_at_1k']:.4f} "
        f"kv_read_frac_at_16k={v['kv_read_frac_at_16k']:.4f} "
        f"kv_read_frac_at_64k={v['kv_read_frac_at_64k']:.4f} "
        f"kv_read_frac_benchmark={v['kv_read_frac_benchmark_weighted']:.4f} "
        f"benchmark_gen_len={OUTPUT_LEN} kv_attributable_tps={v['kv_attributable_tps']:.2f} "
        f"fp8kv_attributable_tps={v['fp8kv_attributable_tps']:.2f} "
        f"kv_material={int(v['kv_material'])} kv_lever_is_green={int(v['kv_lever_is_green'])} "
        f"self_det={int(st['self_test_passes'])} peak_gib={v['peak_gib']} "
        f"primary_metric={v['primary_metric_value']:.2f} wandb_run_id={rid}",
        flush=True,
    )
    return 0 if st["self_test_passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
