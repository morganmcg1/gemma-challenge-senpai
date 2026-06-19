#!/usr/bin/env python
"""PR #708 (land) -- Mixed-grid Marlin servability: the op-bench leg.

EXTENDS land #707 (g32-tax-empirical, run 8jn9ofx7) from a UNIFORM-group body to
a HETEROGENEOUS (mixed-grid) body, using the EXACT served kernel
(apply_gptq_marlin_linear, the call CompressedTensorsWNA16 -> MarlinLinearKernel
makes). The primary (servability) verdict is decided by the SOURCE READ
(see SOURCE_READ.md); this script measures the op-bench TEST metric:
mixed_grid_opbench_tps vs the 126.27 projection / 126.75 g128 / 121.836 g32.

REAL 42-layer fused served census (corrects #707's 37-layer / qkv-only model):
  * 24 non-KV-shared layers: fused qkv (2560 -> 3072 = q2048+k512+v512)
  *   "  KV-shared (last 18): q-only (2560 -> 2048)              [num_kv_shared=18]
  * 42x  o_proj (2048 -> 2560), gate_up (2560 -> 20480), down (10240 -> 2560)
  * 42x  per_layer_input_gate PLIG (2560 -> 256) + per_layer_projection (256->2560)
  * 1x   per_layer_model_projection (2560 -> 10752)
Config: hidden 2560, intermediate 10240, head_dim 256, q/kv heads 8/2, 42 layers,
18 kv-shared (local /workspace/gemma_build/int4_g128_lmhead/config.json).

ubel #700 recovery subset (48 modules / 1.35% body params): 40 PLIG + 3 q + 3 k
+ 2 v. SOURCE READ finding: PLIG is a standalone ReplicatedLinear -> independently
g32-servable; q/k/v are FUSED into qkv_proj (QKVParallelLinear) -> NOT independently
servable (one group_size per fused layer). So the realizable selective mix is
PLIG-only; the attn members need whole-qkv-block promotion.

Measurements (paired, L2-cold CUDA-graph, seed 707, median-of-rounds 95% CI):
  A. all-g128 vs all-g32 full fused body  -> refined full tax.
  B. per-module isolated g32-vs-g128 delta for every component incl PLIG/q/k/v
     -> reveals small-module launch-bound anomaly + the additive byte-law terms.
  C. realizable selective mix (40 PLIG g32 + rest g128) measured DIRECTLY in one
     fused forward -> tps_mix_servable + linearity residual (measured - additive).
  D. fake-quant ideal (denken #706 projection target): additive 40 PLIG + 3q+3k+2v
     isolated g32 deltas -> tps_fakequant48, compare to 126.27.
  E. whole-qkv-block promotion (servable attn route): 40 PLIG + n_blocks qkv g32.

LOCAL A10G (sm_86) op-bench. analysis_only=True, official_tps=0, no_hf_job=1,
fires=0. NO served-file change, NO HF Job, NO build. Locked int4_g128_lmhead@126.378
untouched. Value-independent achieved-DRAM timing (random weights at served shapes
reproduce deployed kernel timing).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
from datetime import datetime, timezone

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
_here = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ constants --
BF16_BYTES = 2.0
N_LAYERS = 42
N_KV_SHARED = 18                      # last 18 layers reuse KV -> q-only
N_FULL_QKV = N_LAYERS - N_KV_SHARED   # 24 layers with own k/v
HIDDEN = 2560
INTERMEDIATE = 10240
N_Q_HEADS, N_KV_HEADS, HEAD_DIM = 8, 2, 256
PLI = 256                             # hidden_size_per_layer_input
G128, G32 = 128, 32

# imported-exact anchors (in-scope: land #707 8jn9ofx7; #666 cross-read).
REF_OFFICIAL_TPS = 126.378     # locked int4_g128_lmhead (#319-byte-exact AR)
PLUS10_TARGET = 136.378
AR_RUNG_OPBENCH_TPS = 126.75   # land #707/#697 op-bench AR rung (official ~= op-bench)
TPS_G32_FULL_707 = 121.836     # land #707 all-g32 floor
PROJ_706 = 126.27              # denken #706 recovery_tps_at_aime_gate (selective)
SELECTIVE_SUBSET_FRAC = 0.0135 # ubel #700 48-module / 1.35% impact-energy subset
SEED = 707

# (K=in, N=out) at the served fused shapes.
QKV_FULL = (HIDDEN, N_Q_HEADS * HEAD_DIM + 2 * N_KV_HEADS * HEAD_DIM)  # 2560 -> 3072
QKV_QONLY = (HIDDEN, N_Q_HEADS * HEAD_DIM)                            # 2560 -> 2048
O_PROJ = (N_Q_HEADS * HEAD_DIM, HIDDEN)                               # 2048 -> 2560
GATE_UP = (HIDDEN, 2 * INTERMEDIATE)                                  # 2560 -> 20480
DOWN = (INTERMEDIATE, HIDDEN)                                         # 10240 -> 2560
PLIG = (HIDDEN, PLI)                                                  # 2560 -> 256
PLPROJ = (PLI, HIDDEN)                                                # 256 -> 2560
PLMODEL = (HIDDEN, N_LAYERS * PLI)                                    # 2560 -> 10752
# unfused attn shapes (checkpoint granularity, for the fake-quant subset delta).
Q_SHAPE = (HIDDEN, N_Q_HEADS * HEAD_DIM)                              # 2560 -> 2048
K_SHAPE = (HIDDEN, N_KV_HEADS * HEAD_DIM)                             # 2560 -> 512
V_SHAPE = (HIDDEN, N_KV_HEADS * HEAD_DIM)                             # 2560 -> 512

# the real fused body as a flat instance list: (name, K, N, count)
BODY_CENSUS = [
    ("qkv_full", *QKV_FULL, N_FULL_QKV),
    ("qkv_qonly", *QKV_QONLY, N_KV_SHARED),
    ("o_proj", *O_PROJ, N_LAYERS),
    ("gate_up_proj", *GATE_UP, N_LAYERS),
    ("down_proj", *DOWN, N_LAYERS),
    ("per_layer_input_gate", *PLIG, N_LAYERS),
    ("per_layer_projection", *PLPROJ, N_LAYERS),
    ("per_layer_model_projection", *PLMODEL, 1),
]
# isolated per-module shapes we also benchmark (incl unfused attn).
ISO_SHAPES = {
    "qkv_full": QKV_FULL, "qkv_qonly": QKV_QONLY, "o_proj": O_PROJ,
    "gate_up_proj": GATE_UP, "down_proj": DOWN,
    "per_layer_input_gate": PLIG, "per_layer_projection": PLPROJ,
    "per_layer_model_projection": PLMODEL,
    "q_proj": Q_SHAPE, "k_proj": K_SHAPE, "v_proj": V_SHAPE,
}


# ----------------------------------------------------- L2-cold build helpers ---
def _ndistinct_for(K, N, target_mib=24.0, lo=4, hi=64):
    """Enough cold copies so the per-component working set >> A10G 6 MiB L2."""
    copy_mib = K * N * 0.5 / 1024**2
    n = int(math.ceil(target_mib / max(copy_mib, 1e-6)))
    return max(lo, min(hi, n))


def build_marlin(dev, K, N, group_size, seed, n_distinct):
    import torch
    from vllm.model_executor.layers.quantization.utils import marlin_utils_test as mt
    from vllm.scalar_type import scalar_types
    QT = scalar_types.uint4b8
    g = torch.Generator(device=dev).manual_seed(seed)
    wl = []
    for _ in range(n_distinct):
        w = torch.randn(K, N, dtype=torch.float16, device=dev, generator=g) * 0.02
        _, q_w, s, g_idx, sort_idx, _perm = mt.marlin_quantize(w, QT, group_size, False)
        del w
        wl.append((q_w, s, g_idx, sort_idx))
    gc.collect(); torch.cuda.empty_cache()
    return wl


def _apply_fns(dev):
    import torch
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_gptq_marlin_linear as apply_marlin, marlin_make_workspace_new as mk_ws)
    from vllm.scalar_type import scalar_types
    QT = scalar_types.uint4b8
    ws = mk_ws(dev)
    zp = torch.zeros(0, dtype=torch.int, device=dev)
    return apply_marlin, ws, zp, QT


# -------------------------------------------------------------- timing prims ---
def _make_graph(run, warmup):
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
    return g


def _time_graph(g, iters):
    import torch
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        g.replay()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1e3  # us/call


def _stats(series):
    med = statistics.median(series)
    sd = statistics.pstdev(series) if len(series) > 1 else 0.0
    ci = 1.96 * sd / math.sqrt(len(series)) if series else 0.0
    return med, sd, ci


# ------------------------------------------------ full-body runner (mixed) -----
def build_body_weights(dev, M, seed, assign):
    """Build weights for the full fused census. `assign` maps a (name, idx) ->
    group_size; build BOTH g128 and g32 copies for any component that ever needs
    g32, so a single forward can pick per-instance."""
    needs_g32 = set()
    for (name, idx), gs in assign.items():
        if gs == G32:
            needs_g32.add(name)
    weights = {}   # name -> {gs: list-of-cold-weights}
    xins = {}
    import torch
    for (name, K, N, count) in BODY_CENSUS:
        nd = _ndistinct_for(K, N)
        weights[name] = {G128: build_marlin(dev, K, N, G128, seed, nd)}
        if name in needs_g32:
            weights[name][G32] = build_marlin(dev, K, N, G32, seed + 1, nd)
        xins[name] = torch.randn(M, K, dtype=torch.float16, device=dev)
    return weights, xins


def body_runner(weights, xins, dev, assign):
    apply_marlin, ws, zp, QT = _apply_fns(dev)

    def run():
        for (name, K, N, count) in BODY_CENSUS:
            for idx in range(count):
                gs = assign.get((name, idx), G128)
                bank = weights[name][gs]
                q_w, s, g_idx, sort_idx = bank[idx % len(bank)]
                apply_marlin(xins[name], q_w, s, zp, g_idx, sort_idx, ws, QT, N, K,
                             is_k_full=True, bias=None)
    return run


def measure_body(dev, M, seed, assign, iters, warmup, rounds, label):
    import torch
    w, x = build_body_weights(dev, M, seed, assign)
    g = _make_graph(body_runner(w, x, dev, assign), warmup)
    series = [_time_graph(g, iters) for _ in range(rounds)]
    del g, w, x
    gc.collect(); torch.cuda.empty_cache()
    med, sd, ci = _stats(series)
    return {"label": label, "us": med, "us_ci": ci, "us_sd": sd, "series": series}


# ------------------------------------------------ isolated per-module delta ----
def iso_runner(wl, xin, dev, K, N):
    apply_marlin, ws, zp, QT = _apply_fns(dev)
    reps = max(1, len(wl))

    def run():
        for i in range(reps):
            q_w, s, g_idx, sort_idx = wl[i % len(wl)]
            apply_marlin(xin, q_w, s, zp, g_idx, sort_idx, ws, QT, N, K,
                         is_k_full=True, bias=None)
    return run, reps


def measure_iso(dev, M, seed, name, iters, warmup, rounds):
    """Paired g128/g32 for one isolated module shape -> per-CALL delta (us)."""
    import torch
    K, N = ISO_SHAPES[name]
    nd = _ndistinct_for(K, N)
    w128 = build_marlin(dev, K, N, G128, seed, nd)
    w32 = build_marlin(dev, K, N, G32, seed + 1, nd)
    xin = torch.randn(M, K, dtype=torch.float16, device=dev)
    r128, reps = iso_runner(w128, xin, dev, K, N)
    r32, _ = iso_runner(w32, xin, dev, K, N)
    g128 = _make_graph(r128, warmup); g32 = _make_graph(r32, warmup)
    s128, s32, dpair = [], [], []
    for _ in range(rounds):
        t128 = _time_graph(g128, iters); t32 = _time_graph(g32, iters)
        s128.append(t128); s32.append(t32); dpair.append(t32 - t128)
    for _ in range(rounds):  # symmetrize order
        t32 = _time_graph(g32, iters); t128 = _time_graph(g128, iters)
        s128.append(t128); s32.append(t32); dpair.append(t32 - t128)
    m128, _, ci128 = _stats(s128); m32, _, ci32 = _stats(s32); md, _, cid = _stats(dpair)
    del g128, g32, w128, w32, xin
    gc.collect(); torch.cuda.empty_cache()
    # per-CALL values (graph ran `reps` calls of this shape)
    return {"name": name, "K": K, "N": N, "reps": reps,
            "per_call_us_g128": m128 / reps, "per_call_us_g32": m32 / reps,
            "per_call_delta_us": md / reps, "per_call_delta_us_ci": cid / reps,
            "tax_frac": md / m128 if m128 > 0 else float("nan")}


# ----------------------------------------------------------------- compose -----
def tps_from_delta(anchor_tps, delta_us):
    total_ms_g128 = 1000.0 / anchor_tps
    return 1000.0 / (total_ms_g128 + delta_us / 1e3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--anchor-tps", type=float, default=AR_RUNG_OPBENCH_TPS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--output", default=os.path.join(_here, "mixed_grid_results.json"))
    args = ap.parse_args()

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
    M = args.m

    # --- assignments ---
    g128_all = {}
    g32_all = {(n, i): G32 for (n, K, N, c) in BODY_CENSUS for i in range(c)}
    # realizable selective: 40 of 42 PLIG -> g32 (rest g128). (40 vs 42 immaterial; honor subset count)
    plig_count = dict((n, c) for (n, K, N, c) in BODY_CENSUS)["per_layer_input_gate"]
    n_plig_g32 = min(40, plig_count)
    mix_servable = {("per_layer_input_gate", i): G32 for i in range(n_plig_g32)}
    # whole-qkv promotion route (servable attn): 40 PLIG + n_blocks qkv_full g32
    mix_wholeqkv3 = dict(mix_servable); mix_wholeqkv3.update({("qkv_full", i): G32 for i in range(3)})
    mix_wholeqkv8 = dict(mix_servable); mix_wholeqkv8.update({("qkv_full", i): G32 for i in range(8)})

    # --- A: full body g128 / g32 ---
    print("[A] full body g128/g32 ...", flush=True)
    b_g128 = measure_body(dev, M, SEED, g128_all, iters, warmup, rounds, "all_g128")
    b_g32 = measure_body(dev, M, SEED, g32_all, iters, warmup, rounds, "all_g32")
    full_delta = b_g32["us"] - b_g128["us"]
    print(f"    g128={b_g128['us']:.2f}+-{b_g128['us_ci']:.2f}us "
          f"g32={b_g32['us']:.2f}+-{b_g32['us_ci']:.2f}us delta={full_delta:.3f}us", flush=True)

    # --- C: realizable selective mix (measured directly) + whole-qkv ---
    print("[C] mixed-grid realizations ...", flush=True)
    b_serv = measure_body(dev, M, SEED, mix_servable, iters, warmup, rounds, "mix_servable_40plig")
    b_wq3 = measure_body(dev, M, SEED, mix_wholeqkv3, iters, warmup, rounds, "mix_wholeqkv3")
    b_wq8 = measure_body(dev, M, SEED, mix_wholeqkv8, iters, warmup, rounds, "mix_wholeqkv8")
    serv_delta = b_serv["us"] - b_g128["us"]
    wq3_delta = b_wq3["us"] - b_g128["us"]
    wq8_delta = b_wq8["us"] - b_g128["us"]

    # --- B: isolated per-module deltas (incl unfused q/k/v) ---
    print("[B] isolated per-module deltas ...", flush=True)
    iso = {n: measure_iso(dev, M, SEED, n, iters, warmup, rounds) for n in ISO_SHAPES}
    for n, r in iso.items():
        print(f"    {n:28s} delta={r['per_call_delta_us']:.4f}us tax={r['tax_frac']*100:.3f}%", flush=True)

    # --- D: fake-quant ideal additive prediction (denken #706 target) ---
    add = lambda n: iso[n]["per_call_delta_us"]
    fakequant48_delta = 40 * add("per_layer_input_gate") + 3 * add("q_proj") + 3 * add("k_proj") + 2 * add("v_proj")
    # additive prediction for the SERVABLE mix (linearity check vs measured serv_delta)
    serv_additive = n_plig_g32 * add("per_layer_input_gate")
    wholeqkv3_additive = serv_additive + 3 * add("qkv_full")
    wholeqkv8_additive = serv_additive + 8 * add("qkv_full")

    # compose to TPS on the AR rung
    tps = lambda d: tps_from_delta(args.anchor_tps, d)
    out = {
        "gpu": gpu, "timestamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "census": [{"name": n, "K": K, "N": N, "count": c} for (n, K, N, c) in BODY_CENSUS],
        "bodies": {b["label"]: {k: v for k, v in b.items() if k != "series"}
                   for b in [b_g128, b_g32, b_serv, b_wq3, b_wq8]},
        "iso": iso,
        "headline": {
            # ---- PRIMARY (servability) decided by SOURCE READ, logged here as scalar ----
            "mixed_grid_servable": 1,             # per-layer group_size IS supported
            "subset_fully_servable": 0,           # but 8 fused-shard attn modules are NOT
            "fused_shard_blocked_modules": 8,     # 3 q + 3 k + 2 v (qkv_proj fusion)
            "standalone_servable_modules": 40,    # per_layer_input_gate (ReplicatedLinear)
            # ---- TEST metric: realized mixed-grid op-bench TPS ----
            "mixed_grid_opbench_tps": tps(serv_delta),          # realizable ship path (40 PLIG g32)
            "tps_mix_servable_40plig": tps(serv_delta),
            "tps_fakequant48_ideal": tps(fakequant48_delta),    # denken #706 target (NOT servable)
            "tps_mix_wholeqkv3": tps(wq3_delta),
            "tps_mix_wholeqkv8": tps(wq8_delta),
            # ---- anchors ----
            "tps_g128_anchor": args.anchor_tps,
            "tps_g32_full_measured": tps(full_delta),
            "tps_g32_full_707": TPS_G32_FULL_707,
            "proj_706": PROJ_706,
            "ref_official_tps_locked": REF_OFFICIAL_TPS,
            "plus10_target_tps": PLUS10_TARGET,
            # ---- deltas (us/token) ----
            "full_delta_us": full_delta, "full_delta_us_ci": b_g32["us_ci"] + b_g128["us_ci"],
            "serv_delta_us": serv_delta, "serv_additive_us": serv_additive,
            "serv_linearity_resid_us": serv_delta - serv_additive,
            "fakequant48_delta_us": fakequant48_delta,
            "wholeqkv3_delta_us": wq3_delta, "wholeqkv3_additive_us": wholeqkv3_additive,
            "wholeqkv8_delta_us": wq8_delta, "wholeqkv8_additive_us": wholeqkv8_additive,
            # ---- full-body diagnostics ----
            "body_us_g128": b_g128["us"], "body_us_g128_ci": b_g128["us_ci"],
            "body_us_g32": b_g32["us"], "body_us_g32_ci": b_g32["us_ci"],
            "full_tax_frac": full_delta / b_g128["us"] if b_g128["us"] else float("nan"),
            "body_frac_of_anchor": (b_g128["us"] / 1e3) / (1000.0 / args.anchor_tps),
            # ---- subset param accounting ----
            "plig_param_frac_of_subset": None,    # filled below
            # ---- protocol ----
            "protocol_M": M, "protocol_iters": iters, "protocol_rounds_each": rounds,
            "protocol_seed": SEED, "n_layers": N_LAYERS, "n_kv_shared": N_KV_SHARED,
            "selective_subset_frac": SELECTIVE_SUBSET_FRAC,
            # ---- guards ----
            "analysis_only": True, "official_tps": 0, "no_hf_job": 1, "fires": 0,
            "no_served_file_change": True,
        },
        "config": {"agent": "land", "pr": 708, "kind": "mixed-grid-marlin-servability",
                   "hidden": HIDDEN, "intermediate": INTERMEDIATE, "n_layers": N_LAYERS,
                   "n_kv_shared": N_KV_SHARED, "pli": PLI, "m": M, "seed": SEED,
                   "anchor_tps": args.anchor_tps, "smoke": args.smoke},
    }
    # subset param accounting (PLIG vs attn share of the 48-module subset)
    plig_p = 40 * (HIDDEN * PLI)
    attn_p = 3 * (HIDDEN * N_Q_HEADS * HEAD_DIM) + 3 * (HIDDEN * N_KV_HEADS * HEAD_DIM) + 2 * (HIDDEN * N_KV_HEADS * HEAD_DIM)
    out["headline"]["plig_param_frac_of_subset"] = plig_p / (plig_p + attn_p)
    out["headline"]["attn_param_frac_of_subset"] = attn_p / (plig_p + attn_p)

    json.dump(out, open(args.output, "w"), indent=2)
    print("\n=== HEADLINE ===", flush=True)
    print(json.dumps({k: v for k, v in out["headline"].items()
                      if isinstance(v, (int, float, bool, str)) or v is None}, indent=2), flush=True)


if __name__ == "__main__":
    main()
