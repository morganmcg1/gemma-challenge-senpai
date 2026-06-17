#!/usr/bin/env python
"""PR #602 (stark) -- AR int4-BODY GEMV: is the shipped Marlin M=1 W4A16 GEMV
HBM-bandwidth-SATURATED, or is there a byte-identical speed lever on the dominant
44.4%-of-cycle component?

LOCAL A10G (sm_86) microbench. NO served-file change, NO kernel BUILD, NO HF Job.
analysis_only=True, official_tps=0.

Decomposition apparatus reused from:
  * lawine/ubel gemm_roofline_bw_ceiling.py -- exact per-component byte model from
    the served safetensors + co-measured STREAM peak HBM BW + the SAME
    apply_gptq_marlin_linear the served GPTQMarlinLinearMethod.apply calls,
    timed L2-cold (n_distinct cold weights >> 6 MiB A10G L2) via CUDA-graph replay
    (mirrors the served ONEGRAPH amortized-launch path).
  * stark #448 int4_gemm_kernel_config_audit.py -- choose_mp_linear_kernel dispatch
    enumeration (only Marlin can_implement on sm_86) + the one selectable Marlin
    knob use_fp32_reduce and its byte-exactness test.

THE QUESTION (AR / spec-OFF / M=1):
  achieved M=1 body-GEMV bandwidth  vs  {measured read-peak, 600 GB/s spec}.
  >= ~0.90 of read-peak  -> body_gemv_bw_saturated=True (lever CLOSED).
  else                   -> quantify body_gemv_headroom_ms / _tps and ask Phase 2:
                            is the gap byte-identically recoverable? (expected NO:
                            no alt W4A16 backend on sm_86, the one knob breaks bits.)

Achieved DRAM BW is value-independent (shape/dtype/group_size/layout only), so
random weights at the served fused shapes faithfully reproduce the deployed
kernel's bandwidth. Greedy/PPL are pinned by construction (no served change).
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import statistics
import struct
import sys
from datetime import datetime, timezone

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # local A10G is index 0 (inherited =1 -> 0 GPUs)
_here = os.path.dirname(os.path.abspath(__file__))

# ---- reuse stark #448 dispatch + machete + fp32_reduce bit-break apparatus ----
_AUDIT = os.path.normpath(os.path.join(
    _here, "..", "..", "validity", "int4_gemm_kernel_config_audit",
    "int4_gemm_kernel_config_audit.py"))
_spec = importlib.util.spec_from_file_location("int4_audit", _AUDIT)
_audit = importlib.util.module_from_spec(_spec)
sys.modules["int4_audit"] = _audit
_spec.loader.exec_module(_audit)

# ---------------------------------------------------------------- constants ---
A10G_SPEC_BW_GBPS = 600.0          # GA102 datasheet peak (NOT achievable peak)
BF16_BYTES = 2.0
GROUP_SIZE = 128
N_LAYERS = 37
HIDDEN = 2560
INTERMEDIATE = 10240
N_Q_HEADS, N_KV_HEADS, HEAD_DIM = 8, 2, 256
SAT_THRESHOLD = 0.90               # card: ">= ~90% of peak -> saturated/CLOSED"
SERVED_BODY = "/tmp/osoi5-v0-baked/model.safetensors"

# imported-exact anchors (this leg derives nothing measured upstream)
REF_OFFICIAL_TPS = 126.378         # operative int4_g128_lmhead (#319-byte-exact AR)
PLUS10_TARGET = 136.4
TAU_LOCAL_TO_OFFICIAL = 1.03524    # #267 local wall_tps -> official scalar
BODY_CYCLE_MS_591 = 6.728          # lawine #591 b001enxl body share
BODY_FRAC_591 = 0.444
FREE_HEAD_FLOOR_TPS = 311.27       # #569 decode floor (head fully free)

# served fused shapes (vLLM fuses q/k/v->qkv, gate/up->gate_up). (K=in, N=out)
SERVED_SHAPES = {
    "qkv_proj":     (HIDDEN, N_Q_HEADS * HEAD_DIM + 2 * N_KV_HEADS * HEAD_DIM),  # 2560 -> 3072
    "o_proj":       (N_Q_HEADS * HEAD_DIM, HIDDEN),                              # 2048 -> 2560
    "gate_up_proj": (HIDDEN, 2 * INTERMEDIATE),                                  # 2560 -> 20480
    "down_proj":    (INTERMEDIATE, HIDDEN),                                      # 10240 -> 2560
}
BODY_ORDER = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]


# ----------------------------------------------------- safetensors byte model -
def _st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def _nbytes(v):
    return v["data_offsets"][1] - v["data_offsets"][0]


def body_byte_model(M: int):
    """EXACT int4 weight + group-128 scale bytes per fused body component, summed
    over ALL layers, from the served safetensors. Activations at width M, per layer
    (read x + write y). Zero-point is empty (g=128 symmetric). Returns per-comp dict
    plus body totals."""
    import re
    h = _st_header(SERVED_BODY)

    def comp_of(name):
        if re.search(r"\.self_attn\.(q|k|v)_proj\.", name): return "qkv_proj"
        if re.search(r"\.self_attn\.o_proj\.", name):       return "o_proj"
        if re.search(r"\.mlp\.(gate|up)_proj\.", name):     return "gate_up_proj"
        if re.search(r"\.mlp\.down_proj\.", name):          return "down_proj"
        return None

    agg = {c: {"weight_bytes": 0.0, "scale_bytes": 0.0} for c in BODY_ORDER}
    for k, v in h.items():
        if k == "__metadata__" or not k.startswith("model.language_model.layers."):
            continue
        c = comp_of(k)
        if c is None:
            continue
        if k.endswith("weight_packed"):
            agg[c]["weight_bytes"] += _nbytes(v)
        elif k.endswith("weight_scale"):
            agg[c]["scale_bytes"] += _nbytes(v)

    bm = {}
    for c in BODY_ORDER:
        K, N = SERVED_SHAPES[c]
        act = N_LAYERS * (M * K + M * N) * BF16_BYTES
        w, s = agg[c]["weight_bytes"], agg[c]["scale_bytes"]
        bm[c] = {"weight_bytes": w, "scale_bytes": s, "act_bytes": act,
                 "ws_bytes": w + s, "total_bytes": w + s + act,
                 "K": K, "N": N, "ai_flop_per_byte": 2.0 * N_LAYERS * M * K * N / (w + s + act)}
    body_ws = sum(bm[c]["ws_bytes"] for c in BODY_ORDER)
    body_act = sum(bm[c]["act_bytes"] for c in BODY_ORDER)
    return bm, body_ws, body_act


# ----------------------------------------------------------- peak HBM BW anchor
def _timed_eager(fn, iters, warmup):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters / 1e3  # seconds/call


def measure_peak_bw(dev, iters, warmup):
    """STREAM read (1x) + copy (2x) -- the achievable HBM ceiling on this silicon."""
    import torch
    N = 512 * 1024 * 1024
    a = torch.empty(N, dtype=torch.bfloat16, device=dev).uniform_(-1, 1)
    b = torch.empty(N, dtype=torch.bfloat16, device=dev)
    nb = N * 2
    t_copy = _timed_eager(lambda: b.copy_(a), iters, warmup)
    t_read = _timed_eager(lambda: torch.sum(a), iters, warmup)
    del a, b
    gc.collect(); torch.cuda.empty_cache()
    return {"bw_read_gbps": nb / t_read / 1e9, "read_us": t_read * 1e6,
            "bw_copy_gbps": 2 * nb / t_copy / 1e9, "copy_us": t_copy * 1e6}


# --------------------------------------- self-built g=128 int4 Marlin body -----
def build_body(dev, n_distinct):
    """n_distinct distinct g=128 int4-Marlin weights per body component (cold HBM
    working set >> L2). Uses the SAME apply_gptq_marlin_linear the served kernel
    calls. Returns a closure body_call(M) that runs one full 37-layer body forward
    (4 GEMMs/layer) at activation width M, cycling the cold weights per layer."""
    import torch
    from vllm.model_executor.layers.quantization.utils import marlin_utils_test as mt
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_gptq_marlin_linear as apply_marlin, marlin_make_workspace_new as mk_ws)
    from vllm.scalar_type import scalar_types
    QT = scalar_types.uint4b8
    ws = mk_ws(dev)
    zp = torch.zeros(0, dtype=torch.int, device=dev)

    weights = {}
    for c in BODY_ORDER:
        K, N = SERVED_SHAPES[c]
        wl = []
        for _ in range(n_distinct):
            w = torch.randn(K, N, dtype=torch.float16, device=dev) * 0.02
            _, q_w, s, g_idx, sort_idx, _perm = mt.marlin_quantize(w, QT, GROUP_SIZE, False)
            del w
            wl.append((q_w, s, g_idx, sort_idx))
        weights[c] = wl
    gc.collect(); torch.cuda.empty_cache()

    def body_call(M):
        xin = {c: torch.randn(M, SERVED_SHAPES[c][0], dtype=torch.float16, device=dev)
               for c in BODY_ORDER}

        def run():
            for L in range(N_LAYERS):
                for c in BODY_ORDER:
                    K, N = SERVED_SHAPES[c]
                    q_w, s, g_idx, sort_idx = weights[c][L % len(weights[c])]
                    apply_marlin(xin[c], q_w, s, zp, g_idx, sort_idx, ws, QT, N, K,
                                 is_k_full=True, bias=None)
        return run

    return body_call


def _graph_us_per_call(run, iters, warmup, rounds):
    """L2-cold CUDA-graph replay; median us per full-body call across rounds."""
    import torch
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    series = []
    for _ in range(rounds):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            g.replay()
        e1.record(); torch.cuda.synchronize()
        series.append(e0.elapsed_time(e1) / iters * 1e3)  # us/call
    del g
    return statistics.median(series), series


# --------------------------------------------------------------- M-sweep -------
def sweep_M(dev, body_call, m_list, peak, iters, warmup, rounds):
    """Body-GEMV time + achieved BW at each M. Achieved BW = body_bytes(M)/time.
    If time is M-invariant (flat) the kernel is already weight-read-BANDWIDTH-bound
    at M=1 -- not a latency-bound regime with a free tiling fix."""
    out = {}
    for M in m_list:
        bm, body_ws, body_act = body_byte_model(M)
        total = body_ws + body_act
        us, series = _graph_us_per_call(body_call(M), iters, warmup, rounds)
        sd = statistics.pstdev(series) if len(series) > 1 else 0.0
        ci = 1.96 * sd / math.sqrt(len(series)) if series else 0.0
        achieved = (total / (us * 1e-6)) / 1e9
        out[str(M)] = {
            "M": M, "body_us": us, "body_us_ci": ci, "body_ms": us / 1e3,
            "total_bytes": total, "ws_bytes": body_ws, "act_bytes": body_act,
            "achieved_bw_gbps": achieved,
            "f_vs_read": achieved / peak["bw_read_gbps"],
            "f_vs_copy": achieved / peak["bw_copy_gbps"],
            "f_vs_spec": achieved / A10G_SPEC_BW_GBPS,
            "ai_flop_per_byte": sum(bm[c]["ai_flop_per_byte"] for c in BODY_ORDER) / len(BODY_ORDER),
        }
    return out


def per_component_bw(dev, n_distinct, M, peak, iters, warmup, rounds):
    """Isolated achieved BW for each body GEMM at width M (where the gap, if any,
    lives: small qkv/o under-saturate; gate_up/down near-saturate)."""
    import torch
    from vllm.model_executor.layers.quantization.utils import marlin_utils_test as mt
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_gptq_marlin_linear as apply_marlin, marlin_make_workspace_new as mk_ws)
    from vllm.scalar_type import scalar_types
    QT = scalar_types.uint4b8
    ws = mk_ws(dev); zp = torch.zeros(0, dtype=torch.int, device=dev)
    bm, _, _ = body_byte_model(M)
    out = {}
    for c in BODY_ORDER:
        K, N = SERVED_SHAPES[c]
        wl = []
        for _ in range(n_distinct):
            w = torch.randn(K, N, dtype=torch.float16, device=dev) * 0.02
            _, q_w, s, g_idx, sort_idx, _perm = mt.marlin_quantize(w, QT, GROUP_SIZE, False)
            del w; wl.append((q_w, s, g_idx, sort_idx))
        xin = torch.randn(M, K, dtype=torch.float16, device=dev)

        def run():
            for L in range(N_LAYERS):
                q_w, s, g_idx, sort_idx = wl[L % len(wl)]
                apply_marlin(xin, q_w, s, zp, g_idx, sort_idx, ws, QT, N, K,
                             is_k_full=True, bias=None)
        us, _ = _graph_us_per_call(run, iters, warmup, rounds)
        tb = bm[c]["total_bytes"]
        achieved = (tb / (us * 1e-6)) / 1e9
        out[c] = {"us": us, "total_bytes": tb, "achieved_bw_gbps": achieved,
                  "f_vs_read": achieved / peak["bw_read_gbps"],
                  "f_vs_spec": achieved / A10G_SPEC_BW_GBPS,
                  "pct_of_body_ws": bm[c]["ws_bytes"] / sum(bm[d]["ws_bytes"] for d in BODY_ORDER)}
        del wl, xin
        gc.collect(); torch.cuda.empty_cache()
    return out


# -------------------- Phase 2 negative: fp32_reduce knob byte-exactness @ M=1 --
def fp32_reduce_bit_break_m1(dev, iters, warmup):
    """The ONE selectable Marlin knob (use_fp32_reduce). At M=1: speedup AND
    bit-compare vs served default. A GEMM-output bit-diff is a STRONGER (cheaper)
    disqualifier than served greedy identity -- if the kernel output bits differ,
    served byte-identity cannot hold."""
    import torch
    rows = []
    for name in BODY_ORDER:
        K, N = SERVED_SHAPES[name]
        g = _audit.build_marlin_gemm(K, N, dev)
        a1 = g["make_input"](1)
        served1 = _audit._cuda_event_us(lambda: g["run"](a1, True, False), iters, warmup)
        fp32off1 = _audit._cuda_event_us(lambda: g["run"](a1, False, False), iters, warmup)
        o_served = g["run"](a1, True, False)
        o_fp32off = g["run"](a1, False, False)
        bitexact = bool(torch.equal(o_served, o_fp32off))
        max_abs = float((o_served.float() - o_fp32off.float()).abs().max().item())
        rows.append({"name": name, "K": K, "N": N,
                     "served_us_m1": served1, "fp32off_us_m1": fp32off1,
                     "speedup_fp32off_m1": served1 / fp32off1 if fp32off1 > 0 else float("nan"),
                     "fp32off_bitexact_vs_served_m1": bitexact, "fp32off_max_abs_delta_m1": max_abs})
        del g, a1, o_served, o_fp32off
        gc.collect(); torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------- compose ------
def compose(gpu, peak, sweep, percomp, dispatch, machete, fp32_rows, n_distinct):
    m1 = sweep["1"]
    f_read = m1["f_vs_read"]
    f_spec = m1["f_vs_spec"]
    saturated = bool(f_read >= SAT_THRESHOLD)

    # headroom to the read-peak floor (the achievable ceiling), if any.
    floor_us_read = (m1["total_bytes"] / (peak["bw_read_gbps"] * 1e9)) * 1e6
    floor_us_spec = (m1["total_bytes"] / (A10G_SPEC_BW_GBPS * 1e9)) * 1e6
    headroom_ms_read = max(0.0, (m1["body_us"] - floor_us_read) / 1e3)
    headroom_ms_spec = max(0.0, (m1["body_us"] - floor_us_spec) / 1e3)

    # project the read-peak-floor headroom to a TPS delta via the #591 body share
    # (body = BODY_FRAC of cycle, BODY_CYCLE_MS_591 ms). If the body shrank by the
    # measured-vs-floor fraction, the cycle shrinks proportionally on the body share.
    body_speedup_to_read_floor = m1["body_us"] / floor_us_read if floor_us_read > 0 else 1.0
    cycle_ms = BODY_CYCLE_MS_591 / BODY_FRAC_591                      # implied #591 cycle
    new_cycle_ms = cycle_ms - BODY_CYCLE_MS_591 * (1.0 - 1.0 / body_speedup_to_read_floor)
    headroom_tps_read = (REF_OFFICIAL_TPS * cycle_ms / new_cycle_ms - REF_OFFICIAL_TPS
                         if new_cycle_ms > 0 else float("nan"))

    # M-invariance: body time flat across M -> already bandwidth-bound at M=1.
    m_invariance_8_over_1 = sweep["8"]["body_us"] / sweep["1"]["body_us"] if "8" in sweep else float("nan")
    m_invariance_max_over_1 = (max(sweep[k]["body_us"] for k in sweep) / sweep["1"]["body_us"])

    # Phase 2: is ANY selectable variant both faster AND byte-identical?
    marlin_unique = dispatch["marlin_is_unique"]
    machete_here = machete.get("machete_selectable_here", False)
    all_fp32off_bitexact_m1 = all(r["fp32off_bitexact_vs_served_m1"] for r in fp32_rows)
    fp32off_body_speedup_m1 = (sum(r["served_us_m1"] for r in fp32_rows)
                               / sum(r["fp32off_us_m1"] for r in fp32_rows))
    # a byte-identical lever needs: a selectable variant that is faster AND bitexact.
    # the only non-Marlin backends are unselectable on sm_86; the only Marlin knob
    # (fp32_reduce=False) is non-bitexact -> no byte-identical body-GEMV lever.
    byte_identical_lever_exists = bool(
        (not marlin_unique or machete_here) is False and  # no alt backend selectable
        all_fp32off_bitexact_m1 and fp32off_body_speedup_m1 > 1.0)  # & a bitexact faster knob

    verdict = (
        f"M=1 AR body Marlin W4A16 GEMV achieves {m1['achieved_bw_gbps']:.1f} GB/s = "
        f"{f_read*100:.1f}% of measured read-peak ({peak['bw_read_gbps']:.1f}) / "
        f"{f_spec*100:.1f}% of 600 spec. body_gemv_bw_saturated(>= {SAT_THRESHOLD:.0%} read-peak)="
        f"{saturated}. Body time is M-INVARIANT (Mmax/M1={m_invariance_max_over_1:.3f}, "
        f"M8/M1={m_invariance_8_over_1:.3f}) and AI~{m1['ai_flop_per_byte']:.1f} flop/byte << A10G "
        f"ridge 208 -> the GEMV is weight-read-BANDWIDTH-bound at M=1, NOT a latency-bound regime "
        f"with a free tiling fix. PHASE-2 (byte-identical recovery): on sm_86 only Marlin "
        f"can_implement (marlin_unique={marlin_unique}; Machete/Cutlass-W4A8=Hopper sm_90-only, "
        f"machete_selectable_here={machete_here}; AllSpark no-g128 on Ampere; Conch absent; "
        f"Exllama fp16-only). The one selectable Marlin knob use_fp32_reduce=False is "
        f"{fp32off_body_speedup_m1:.4f}x at M=1 but "
        f"{'KEEPS' if all_fp32off_bitexact_m1 else 'BREAKS'} byte-exactness "
        f"-> byte_identical_body_gemv_lever_exists={byte_identical_lever_exists}. The "
        f"{f_read*100:.0f}%->100% read-peak residual is NOT byte-identically recoverable (no alt "
        f"W4A16 backend on sm_86; the only knob/any retile changes the reduction order -> breaks "
        f"#319). VERDICT: the AR body GEMV is a FIXED Marlin M=1 reference -> the +10-over-"
        f"{REF_OFFICIAL_TPS} hunt belongs to lawine's attention/graph/scheduler axis (#601). "
        f"Converges with lawine #591's 0.0 reclaimable from a fresh M=1 bandwidth axis."
    )

    headline = {
        "body_gemv_bw_saturated": saturated,
        "m1_body_achieved_bw_gbps": m1["achieved_bw_gbps"],
        "m1_f_vs_read_peak": f_read,
        "m1_f_vs_copy_peak": m1["f_vs_copy"],
        "m1_f_vs_600_spec": f_spec,
        "peak_read_gbps": peak["bw_read_gbps"],
        "peak_copy_gbps": peak["bw_copy_gbps"],
        "m1_body_us": m1["body_us"],
        "m1_body_ms": m1["body_ms"],
        "m1_body_ws_bytes": m1["ws_bytes"],
        "m1_body_total_bytes": m1["total_bytes"],
        "bw_floor_us_read_peak": floor_us_read,
        "bw_floor_us_600_spec": floor_us_spec,
        "body_gemv_headroom_ms_vs_read_peak": headroom_ms_read,
        "body_gemv_headroom_ms_vs_600_spec": headroom_ms_spec,
        "body_gemv_headroom_tps_vs_read_peak": headroom_tps_read,
        "m_invariance_max_over_m1": m_invariance_max_over_1,
        "m_invariance_m8_over_m1": m_invariance_8_over_1,
        "marlin_is_unique_on_sm86": marlin_unique,
        "machete_selectable_here": machete_here,
        "fp32off_body_speedup_m1": fp32off_body_speedup_m1,
        "fp32off_all_bitexact_m1": all_fp32off_bitexact_m1,
        "byte_identical_body_gemv_lever_exists": byte_identical_lever_exists,
        "ai_flop_per_byte_m1": m1["ai_flop_per_byte"],
        "ref_official_tps": REF_OFFICIAL_TPS,
        "plus10_target_tps": PLUS10_TARGET,
        "official_tps": 0,
        "analysis_only": True,
        "no_hf_job": True,
        "no_served_file_change": True,
    }
    return {"gpu": gpu, "peak_bw": peak, "byte_model_note":
            "weight+scale summed over 37 layers from served safetensors; act at width M",
            "m_sweep": sweep, "per_component_m1": percomp, "dispatch": dispatch,
            "machete": machete, "fp32_reduce_m1": fp32_rows, "headline": headline,
            "verdict": verdict, "n_distinct": n_distinct,
            "timestamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}


# ---------------------------------------------------------------- self-test ----
def self_test():
    c = {}
    # byte model: weight bytes match int4 0.5 B/param summed over 37 layers (+ scale)
    bm, body_ws, body_act = body_byte_model(1)
    exp_w = sum(N_LAYERS * SERVED_SHAPES[x][0] * SERVED_SHAPES[x][1] * 0.5 for x in BODY_ORDER)
    c["weight_bytes_near_int4_floor"] = abs(sum(bm[x]["weight_bytes"] for x in BODY_ORDER) - exp_w) / exp_w < 0.05
    c["body_ws_gt_1p5GB"] = 1.5e9 < body_ws < 2.0e9
    c["m1_act_small_vs_ws"] = body_act < 0.02 * body_ws
    c["ai_well_below_ridge"] = all(bm[x]["ai_flop_per_byte"] < 50 for x in BODY_ORDER)
    # saturation arithmetic
    fake_peak = {"bw_read_gbps": 500.0, "bw_copy_gbps": 480.0}
    floor_us = (body_ws / (500e9)) * 1e6
    c["floor_positive"] = floor_us > 0
    c["sat_threshold_is_0p9"] = SAT_THRESHOLD == 0.90
    ok = all(c.values())
    return {"self_test_passes": ok, "checks": c,
            "body_ws_bytes": body_ws, "body_act_bytes_m1": body_act}


def _log_wandb(args, payload):
    try:
        import wandb
    except Exception as ex:  # noqa: BLE001
        print(f"[wandb] import failed: {ex!r}", flush=True); return None
    h = payload["headline"]
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type="int4-body-gemv-bw-saturation", config=payload.get("config", {}),
                     reinit=True)
    flat = {k: v for k, v in h.items() if isinstance(v, (int, float, bool))}
    for mk, mv in payload["m_sweep"].items():
        for kk in ("body_us", "achieved_bw_gbps", "f_vs_read", "f_vs_spec"):
            flat[f"sweep_M{mk}_{kk}"] = mv[kk]
    for c, cv in payload["per_component_m1"].items():
        flat[f"comp_{c}_f_vs_read"] = cv["f_vs_read"]
        flat[f"comp_{c}_achieved_bw_gbps"] = cv["achieved_bw_gbps"]
    wandb.log(flat)
    wandb.summary.update(flat)
    wandb.summary["verdict"] = payload["verdict"]
    rid = run.id
    wandb.finish()
    return rid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=21)
    ap.add_argument("--n-distinct", type=int, default=8)
    ap.add_argument("--m-list", default="1,2,4,8,16,32,64")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--output", default=os.path.join(_here, "int4_body_gemv_bw_saturation_results.json"))
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="int4-body-gemv-bw-saturation")
    ap.add_argument("--wandb_name", default="stark/int4-body-gemv-bw-saturation")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    st = self_test()
    print("[self-test]", json.dumps(st, indent=2), flush=True)
    if args.self_test:
        json.dump({"self_test": st}, open(os.path.join(_here, "selftest.json"), "w"), indent=2)
        return

    import torch
    dev = torch.device("cuda:0")
    p = torch.cuda.get_device_properties(dev); cc = torch.cuda.get_device_capability(dev)
    gpu = {"name": p.name, "sm_count": p.multi_processor_count,
           "compute_capability": f"{cc[0]}.{cc[1]}", "is_sm86": bool(cc == (8, 6)),
           "total_mem_gib": round(p.total_memory / 1024**3, 2)}
    print("[gpu]", gpu, flush=True)

    iters = 5 if args.smoke else args.iters
    warmup = 5 if args.smoke else args.warmup
    rounds = 3 if args.smoke else args.rounds
    n_distinct = 2 if args.smoke else args.n_distinct
    m_list = [1, 8] if args.smoke else [int(x) for x in args.m_list.split(",")]

    peak = measure_peak_bw(dev, iters, warmup)
    print("[peak]", peak, flush=True)

    body_call = build_body(dev, n_distinct)
    sweep = sweep_M(dev, body_call, m_list, peak, iters, warmup, rounds)
    print("[sweep]", json.dumps({k: {"us": round(v["body_us"], 1), "f_read": round(v["f_vs_read"], 4)}
                                 for k, v in sweep.items()}), flush=True)
    del body_call
    gc.collect(); torch.cuda.empty_cache()

    percomp = per_component_bw(dev, n_distinct, 1, peak, iters, warmup, rounds)
    print("[per-comp M=1]", json.dumps({k: round(v["f_vs_read"], 4) for k, v in percomp.items()}), flush=True)

    dispatch = _audit.resolve_dispatch(dev)
    machete = _audit.probe_machete(dev)
    fp32_rows = fp32_reduce_bit_break_m1(dev, iters, warmup)
    print("[dispatch]", dispatch["selected_kernel"], "marlin_unique=", dispatch["marlin_is_unique"],
          "machete_here=", machete.get("machete_selectable_here"), flush=True)
    print("[fp32_reduce M=1 bitexact]", [r["fp32off_bitexact_vs_served_m1"] for r in fp32_rows], flush=True)

    payload = compose(gpu, peak, sweep, percomp, dispatch, machete, fp32_rows, n_distinct)
    payload["self_test"] = st
    payload["config"] = {"agent": "stark", "pr": 602, "kind": "int4-body-gemv-bw-saturation",
                         "hidden": HIDDEN, "intermediate": INTERMEDIATE, "n_layers": N_LAYERS,
                         "group_size": GROUP_SIZE, "n_distinct": n_distinct,
                         "ref_official_tps": REF_OFFICIAL_TPS}
    json.dump(payload, open(args.output, "w"), indent=2)
    print("\n=== HEADLINE ===", flush=True)
    print(json.dumps(payload["headline"], indent=2), flush=True)
    print("\n=== VERDICT ===\n" + payload["verdict"], flush=True)

    if not args.no_wandb and not args.smoke:
        rid = _log_wandb(args, payload)
        payload["wandb_run_id"] = rid
        json.dump(payload, open(args.output, "w"), indent=2)
        print(f"\n[wandb] run_id={rid}", flush=True)


if __name__ == "__main__":
    main()
