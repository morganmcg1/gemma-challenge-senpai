#!/usr/bin/env python
"""PR #707 (land) -- the int4 g32-vs-g128 op-bench TPS tax: the EMPIRICAL upper
bound on the int4-body quality-recovery's speed cost.

DECISION-FORCING QUESTION: ubel #700 localized the int4-body AIME deficit to a
48-module / 1.35% impact-energy subset; ubel #702 tries to recover it by upgrading
that subset g128 -> g32 (finer int4 grouping, still strict-#319-admissible, still
deterministic Marlin). What does g32 COST in op-bench TPS vs the locked g128? The
FULL-model g32-vs-g128 tax is the UPPER BOUND on the selective-g32-on-48 tax (the
recovery upgrades only 1.35% of modules).

WHY A MICROBENCH, NOT A FULL-MODEL LOAD (disclosed, load-bearing):
  * The g32 tax lives ENTIRELY in the body matmuls -- g32 reads 4x the per-group
    int4 scales of g128 (scale_bytes/weight_bytes = 4/group_size, a shape-independent
    +9.09% weight+scale traffic g128->g32). The lm_head, embeddings, attention, KV,
    norms, and activations are BYTE-IDENTICAL between a g128-body and a g32-body
    model -- so the body-matmul delta IS the complete per-token tax.
  * A full-model g32 build is disk-infeasible here: the locked g128 build is 9.7 GB,
    the shared overlay has ~11 GB free against a ~8 GB floor (two 9.7 GB builds will
    not coexist), and the bf16 qat_unq source for build_quant.py is not on the pod.
  * The only on-disk g32 (official google/gemma-4-E4B-it-qat-w4a16-ct) carries a
    TIED bf16 lm_head, so an official-g32-vs-locked-g128 full op-bench would be
    confounded by a ~1 GB/token head-traffic difference, NOT the body group size.
  * So the kernel microbench is BOTH the feasible primary AND the cleanest isolation
    of the quantization-group delta (zero head/attention confound). It drives the
    EXACT served kernel: apply_gptq_marlin_linear, the call the deployed
    GPTQMarlinLinearMethod.apply makes. Achieved DRAM time is value-independent
    (shape/dtype/group_size/layout only), so random weights at the served fused
    shapes faithfully reproduce the deployed kernel timing. Greedy/PPL are pinned by
    construction (no served-file change).

PROTOCOL (the #697 op-bench standard -- apples-to-apples discipline):
  * Same harness, same seed, same fused body shapes, PAIRED per-round: every round
    times the g128 body forward AND the g32 body forward back-to-back, so the tax is
    the quantization-group delta and nothing else (cancels thermal/clock drift).
  * L2-cold CUDA-graph replay (n_distinct cold weights per component, working set
    >> A10G 6 MiB L2), median-of-rounds with 95% CI -- mirrors the served amortized
    ONEGRAPH launch path and the #602/#697 op-bench timing primitive.
  * Full 37-layer body forward = 4 fused GEMMs/layer (qkv, o, gate_up, down) at the
    served shapes. M=1 is the AR single-stream decode geometry.

COMPOSITION (transparent): the body-matmul delta is the entire per-token tax;
the per-token DENOMINATOR is anchored on land's established op-bench AR rung
(tps_g128 = 126.75, #697/#642 lineage; official ~= op-bench, #697). The g128 body
fraction we measure cross-checks that anchor.

LOCAL A10G (sm_86) op-bench. analysis_only=True, official_tps=0, no_hf_job=1,
fires=0. NO served-file change, NO HF Job, NO /v1/jobs:run, NO --launch, NO
submission. Locked int4_g128_lmhead@126.378 untouched.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
from datetime import datetime, timezone

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # local A10G is index 0
_here = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ constants --
BF16_BYTES = 2.0
N_LAYERS = 37
HIDDEN = 2560
INTERMEDIATE = 10240
N_Q_HEADS, N_KV_HEADS, HEAD_DIM = 8, 2, 256

# served fused shapes (vLLM fuses q/k/v->qkv, gate/up->gate_up). (K=in, N=out)
SERVED_SHAPES = {
    "qkv_proj":     (HIDDEN, N_Q_HEADS * HEAD_DIM + 2 * N_KV_HEADS * HEAD_DIM),  # 2560 -> 3072
    "o_proj":       (N_Q_HEADS * HEAD_DIM, HIDDEN),                              # 2048 -> 2560
    "gate_up_proj": (HIDDEN, 2 * INTERMEDIATE),                                  # 2560 -> 20480
    "down_proj":    (INTERMEDIATE, HIDDEN),                                      # 10240 -> 2560
}
BODY_ORDER = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]

# imported-exact anchors (in-scope: land #684/#697/#642; #666 cross-read).
# The locked rung + the op-bench AR denominator. This leg measures only the DELTA.
REF_OFFICIAL_TPS = 126.378     # locked int4_g128_lmhead (#319-byte-exact AR), PR #4
PLUS10_TARGET = 136.378        # +10 bar
AR_RUNG_OPBENCH_TPS = 126.75   # land #697/#642 op-bench AR rung (official ~= op-bench)
SELECTIVE_SUBSET_FRAC = 0.0135 # ubel #700: 48-module / 1.35% impact-energy subset

G128, G32 = 128, 32


# ------------------------------------------------------- analytical byte model -
def byte_model(M: int):
    """Weight + int4-scale bytes per fused body component, summed over 37 layers,
    at group_size 128 and 32. Scale is bf16, symmetric (no zero-point). Activations
    at width M per layer (read x + write y). scale_bytes/weight_bytes = 4/group_size
    -> a SHAPE-INDEPENDENT +9.0909% weight+scale traffic from g128 -> g32."""
    out = {}
    for c in BODY_ORDER:
        K, N = SERVED_SHAPES[c]
        wbytes = N_LAYERS * K * N * 0.5                       # int4 packed
        s128 = N_LAYERS * (K // G128) * N * BF16_BYTES        # bf16 group scales
        s32 = N_LAYERS * (K // G32) * N * BF16_BYTES
        act = N_LAYERS * (M * K + M * N) * BF16_BYTES
        out[c] = {"K": K, "N": N, "weight_bytes": wbytes,
                  "scale_bytes_g128": s128, "scale_bytes_g32": s32,
                  "act_bytes": act,
                  "ws_bytes_g128": wbytes + s128, "ws_bytes_g32": wbytes + s32,
                  "total_bytes_g128": wbytes + s128 + act,
                  "total_bytes_g32": wbytes + s32 + act}
    tot = {
        "weight_bytes": sum(out[c]["weight_bytes"] for c in BODY_ORDER),
        "scale_bytes_g128": sum(out[c]["scale_bytes_g128"] for c in BODY_ORDER),
        "scale_bytes_g32": sum(out[c]["scale_bytes_g32"] for c in BODY_ORDER),
        "ws_bytes_g128": sum(out[c]["ws_bytes_g128"] for c in BODY_ORDER),
        "ws_bytes_g32": sum(out[c]["ws_bytes_g32"] for c in BODY_ORDER),
        "total_bytes_g128": sum(out[c]["total_bytes_g128"] for c in BODY_ORDER),
        "total_bytes_g32": sum(out[c]["total_bytes_g32"] for c in BODY_ORDER),
    }
    tot["byte_tax_frac_ws"] = tot["ws_bytes_g32"] / tot["ws_bytes_g128"] - 1.0
    tot["byte_tax_frac_total_m1"] = tot["total_bytes_g32"] / tot["total_bytes_g128"] - 1.0
    return out, tot


# --------------------------------------------- self-built int4 Marlin body -----
def build_weights(dev, n_distinct, group_size, seed):
    """n_distinct distinct int4-Marlin weights per body component at `group_size`
    (cold HBM working set >> L2). SAME apply path the served kernel calls."""
    import torch
    from vllm.model_executor.layers.quantization.utils import marlin_utils_test as mt
    from vllm.scalar_type import scalar_types
    QT = scalar_types.uint4b8
    g = torch.Generator(device=dev).manual_seed(seed)
    weights = {}
    for c in BODY_ORDER:
        K, N = SERVED_SHAPES[c]
        wl = []
        for _ in range(n_distinct):
            w = (torch.randn(K, N, dtype=torch.float16, device=dev, generator=g) * 0.02)
            _, q_w, s, g_idx, sort_idx, _perm = mt.marlin_quantize(w, QT, group_size, False)
            del w
            wl.append((q_w, s, g_idx, sort_idx))
        weights[c] = wl
    gc.collect(); torch.cuda.empty_cache()
    return weights


def body_runner(weights, dev, M):
    """Closure: one full 37-layer body forward (4 GEMMs/layer) at activation width
    M, cycling the cold weights per layer."""
    import torch
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_gptq_marlin_linear as apply_marlin, marlin_make_workspace_new as mk_ws)
    from vllm.scalar_type import scalar_types
    QT = scalar_types.uint4b8
    ws = mk_ws(dev)
    zp = torch.zeros(0, dtype=torch.int, device=dev)
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


# ------------------------------------------------ paired g128 vs g32 measurement
def paired_measure(dev, M, n_distinct, iters, warmup, rounds, seed):
    """Build g128 + g32 body weights; PER ROUND time both back-to-back -> paired
    delta (cancels drift). Returns medians, CIs, and the per-round paired delta."""
    import torch
    w128 = build_weights(dev, n_distinct, G128, seed)
    w32 = build_weights(dev, n_distinct, G32, seed)  # SAME seed -> identical underlying w
    g128 = _make_graph(body_runner(w128, dev, M), warmup)
    g32 = _make_graph(body_runner(w32, dev, M), warmup)

    s128, s32, dpair = [], [], []
    for _ in range(rounds):
        t128 = _time_graph(g128, iters)
        t32 = _time_graph(g32, iters)
        s128.append(t128); s32.append(t32); dpair.append(t32 - t128)
    # second interleave order (g32 first) to symmetrize any ordering bias
    for _ in range(rounds):
        t32 = _time_graph(g32, iters)
        t128 = _time_graph(g128, iters)
        s128.append(t128); s32.append(t32); dpair.append(t32 - t128)

    del g128, g32, w128, w32
    gc.collect(); torch.cuda.empty_cache()

    m128, sd128, ci128 = _stats(s128)
    m32, sd32, ci32 = _stats(s32)
    md, sdd, cid = _stats(dpair)
    return {
        "M": M, "n_distinct": n_distinct, "iters": iters, "rounds_each_order": rounds,
        "body_us_g128": m128, "body_us_g128_ci": ci128, "body_us_g128_sd": sd128,
        "body_us_g32": m32, "body_us_g32_ci": ci32, "body_us_g32_sd": sd32,
        "paired_delta_us": md, "paired_delta_us_ci": cid, "paired_delta_us_sd": sdd,
        "series_g128": s128, "series_g32": s32, "series_delta": dpair,
    }


def per_component(dev, M, n_distinct, iters, warmup, rounds, seed):
    """Per-shape g128 vs g32 tax (instruction #2: the dominant matmul shapes)."""
    import torch
    out = {}
    for c in BODY_ORDER:
        K, N = SERVED_SHAPES[c]
        w128 = build_weights_single(dev, n_distinct, G128, seed, c)
        w32 = build_weights_single(dev, n_distinct, G32, seed, c)
        g128 = _make_graph(comp_runner(w128, dev, M, c), warmup)
        g32 = _make_graph(comp_runner(w32, dev, M, c), warmup)
        s128, s32, dpair = [], [], []
        for _ in range(rounds):
            t128 = _time_graph(g128, iters); t32 = _time_graph(g32, iters)
            s128.append(t128); s32.append(t32); dpair.append(t32 - t128)
        m128, _, ci128 = _stats(s128); m32, _, ci32 = _stats(s32); md, _, cid = _stats(dpair)
        out[c] = {"K": K, "N": N, "us_g128": m128, "us_g128_ci": ci128,
                  "us_g32": m32, "us_g32_ci": ci32, "delta_us": md, "delta_us_ci": cid,
                  "tax_frac": md / m128 if m128 > 0 else float("nan")}
        del g128, g32, w128, w32
        gc.collect(); torch.cuda.empty_cache()
    return out


def build_weights_single(dev, n_distinct, group_size, seed, comp):
    import torch
    from vllm.model_executor.layers.quantization.utils import marlin_utils_test as mt
    from vllm.scalar_type import scalar_types
    QT = scalar_types.uint4b8
    g = torch.Generator(device=dev).manual_seed(seed)
    K, N = SERVED_SHAPES[comp]
    wl = []
    for _ in range(n_distinct):
        w = (torch.randn(K, N, dtype=torch.float16, device=dev, generator=g) * 0.02)
        _, q_w, s, g_idx, sort_idx, _perm = mt.marlin_quantize(w, QT, group_size, False)
        del w
        wl.append((q_w, s, g_idx, sort_idx))
    gc.collect(); torch.cuda.empty_cache()
    return wl


def comp_runner(wl, dev, M, comp):
    import torch
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_gptq_marlin_linear as apply_marlin, marlin_make_workspace_new as mk_ws)
    from vllm.scalar_type import scalar_types
    QT = scalar_types.uint4b8
    ws = mk_ws(dev); zp = torch.zeros(0, dtype=torch.int, device=dev)
    K, N = SERVED_SHAPES[comp]
    xin = torch.randn(M, K, dtype=torch.float16, device=dev)

    def run():
        for L in range(N_LAYERS):
            q_w, s, g_idx, sort_idx = wl[L % len(wl)]
            apply_marlin(xin, q_w, s, zp, g_idx, sort_idx, ws, QT, N, K,
                         is_k_full=True, bias=None)
    return run


# ----------------------------------------------------------------- compose -----
def compose(gpu, bm_comp, bm_tot, paired, percomp, anchor_tps):
    body_us_g128 = paired["body_us_g128"]
    body_us_g32 = paired["body_us_g32"]
    delta_us = paired["paired_delta_us"]            # per-token body tax (us)
    delta_ms = delta_us / 1e3

    # body-relative tax (denominator-free, harness-internal): tax as a fraction of
    # the body-matmul time. This is the "pure" matmul tax, robust to the anchor.
    g32_body_tax_frac = delta_us / body_us_g128 if body_us_g128 > 0 else float("nan")

    # full-model fractional tax: the body-matmul delta adds to the per-token decode
    # time. denominator = anchored op-bench AR rung (tps_g128 ~ 126.75).
    total_ms_g128 = 1000.0 / anchor_tps
    total_ms_g32 = total_ms_g128 + delta_ms
    tps_g32_full = 1000.0 / total_ms_g32
    g32_full_tax_frac = (anchor_tps - tps_g32_full) / anchor_tps   # == delta_ms/total_ms_g32

    # cross-check: what body fraction does the measured body_us imply vs the anchor?
    body_frac_of_anchor = (body_us_g128 / 1e3) / total_ms_g128

    # per-module-fraction coefficient (full = 100% of body upgraded) and the
    # selective-on-48 (1.35%) projection -- the anchor for denken's Pareto.
    g32_per_module_tax = g32_full_tax_frac / 1.0
    selective_g32_tax_proj = g32_per_module_tax * SELECTIVE_SUBSET_FRAC
    selective_tps_cost = selective_g32_tax_proj * REF_OFFICIAL_TPS   # TPS lost on locked rung
    full_tps_cost = g32_full_tax_frac * REF_OFFICIAL_TPS

    # bandwidth-bound ceiling for context: if perfectly weight+scale BW-bound, the
    # body tax would equal the +9.09% byte ratio.
    bw_bound_body_tax = bm_tot["byte_tax_frac_ws"]

    # ---- verdict (decision-relevant quantity = the SELECTIVE tax vs the +10 bar) --
    PLUS10_MARGIN = 10.0
    if selective_tps_cost >= 0.5 * PLUS10_MARGIN:
        # selective alone burns >= half the +10 margin on the rung
        verdict = "G32_TAX_PROHIBITIVE"
    elif selective_tps_cost >= 0.10 * PLUS10_MARGIN or g32_full_tax_frac > 0.10:
        verdict = "G32_TAX_MATERIAL"
    else:
        verdict = "G32_TAX_NEGLIGIBLE"

    headline = {
        # ---- primary + test metrics ----
        "g32_full_tax_frac": g32_full_tax_frac,                 # PRIMARY (upper bound)
        "selective_g32_tax_proj": selective_g32_tax_proj,       # TEST (denken anchor)
        # ---- supporting ----
        "tps_g128": anchor_tps,
        "tps_g32_full": tps_g32_full,
        "g32_per_module_tax": g32_per_module_tax,
        "g32_body_tax_frac": g32_body_tax_frac,
        "body_us_g128": body_us_g128,
        "body_us_g128_ci": paired["body_us_g128_ci"],
        "body_us_g32": body_us_g32,
        "body_us_g32_ci": paired["body_us_g32_ci"],
        "paired_delta_us": delta_us,
        "paired_delta_us_ci": paired["paired_delta_us_ci"],
        "delta_ms_per_token": delta_ms,
        "selective_tps_cost_on_rung": selective_tps_cost,
        "full_tps_cost_on_rung": full_tps_cost,
        "body_frac_of_anchor": body_frac_of_anchor,
        "bw_bound_body_tax_frac": bw_bound_body_tax,
        "byte_tax_frac_ws_g32_vs_g128": bm_tot["byte_tax_frac_ws"],
        # ---- op-bench protocol scalars (W&B-verifiable apples-to-apples) ----
        "protocol_M": paired["M"],
        "protocol_n_distinct": paired["n_distinct"],
        "protocol_iters_per_round": paired["iters"],
        "protocol_rounds_each_order": paired["rounds_each_order"],
        "protocol_seed": SEED,
        "protocol_prompts": "synthetic-marlin-random (value-independent DRAM timing)",
        "protocol_output_len": "n/a (kernel op-bench; per-token body forward)",
        "n_layers": N_LAYERS,
        # ---- anchors ----
        "ref_official_tps_locked": REF_OFFICIAL_TPS,
        "plus10_target_tps": PLUS10_TARGET,
        "ar_rung_opbench_tps": AR_RUNG_OPBENCH_TPS,
        "selective_subset_frac": SELECTIVE_SUBSET_FRAC,
        # ---- guards ----
        "analysis_only": True,
        "official_tps": 0,
        "no_hf_job": 1,
        "fires": 0,
        "no_served_file_change": True,
        "verdict": verdict,
    }

    note = (
        f"g32-vs-g128 op-bench tax (M={paired['M']}): body-matmul paired delta "
        f"{delta_us:.3f} us/token (CI +-{paired['paired_delta_us_ci']:.3f}) on a "
        f"g128 body of {body_us_g128:.2f} us -> g32_body_tax_frac={g32_body_tax_frac*100:.3f}%. "
        f"Composed on the op-bench AR rung ({anchor_tps:.2f} TPS, {total_ms_g128:.4f} ms/tok): "
        f"tps_g32_full={tps_g32_full:.3f} -> g32_full_tax_frac={g32_full_tax_frac*100:.3f}% "
        f"(={full_tps_cost:.3f} TPS on the {REF_OFFICIAL_TPS} rung). Selective-g32-on-48 "
        f"projection = {g32_full_tax_frac*100:.3f}% x {SELECTIVE_SUBSET_FRAC} = "
        f"{selective_g32_tax_proj*100:.4f}% = {selective_tps_cost:.4f} TPS on the rung "
        f"(vs the +10 margin). BW-bound ceiling on the body tax = +{bw_bound_body_tax*100:.3f}% "
        f"(scale_bytes/weight_bytes=4/gs). measured body frac of anchor = "
        f"{body_frac_of_anchor*100:.1f}%. VERDICT={verdict}. analysis_only, official_tps=0, "
        f"no_hf_job=1, fires=0; locked rung + served file untouched; microbench (full g32 "
        f"build disk-infeasible) -- the selective true tax may differ by kernel-occupancy; "
        f"denken's Pareto carries the band."
    )
    return {"gpu": gpu, "byte_model_per_comp": bm_comp, "byte_model_total": bm_tot,
            "paired": {k: v for k, v in paired.items() if not k.startswith("series_")},
            "paired_series": {k: paired[k] for k in ("series_g128", "series_g32", "series_delta")},
            "per_component": percomp, "headline": headline, "note": note,
            "timestamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}


# ---------------------------------------------------------------- self-test ----
def self_test():
    bm_comp, bm_tot = byte_model(1)
    c = {}
    # scale_bytes/weight_bytes = 4/group_size, shape-independent
    for comp in BODY_ORDER:
        r128 = bm_comp[comp]["scale_bytes_g128"] / bm_comp[comp]["weight_bytes"]
        r32 = bm_comp[comp]["scale_bytes_g32"] / bm_comp[comp]["weight_bytes"]
        c[f"{comp}_scale_ratio_g128"] = abs(r128 - 4.0 / G128) < 1e-9
        c[f"{comp}_scale_ratio_g32"] = abs(r32 - 4.0 / G32) < 1e-9
    c["g32_is_4x_scale"] = abs(bm_tot["scale_bytes_g32"] / bm_tot["scale_bytes_g128"] - 4.0) < 1e-9
    c["byte_tax_near_9pct"] = abs(bm_tot["byte_tax_frac_ws"] - 0.0909090909) < 1e-6
    c["weight_bytes_match"] = abs(
        bm_tot["weight_bytes"]
        - sum(N_LAYERS * SERVED_SHAPES[x][0] * SERVED_SHAPES[x][1] * 0.5 for x in BODY_ORDER)
    ) < 1.0
    ok = all(c.values())
    return {"self_test_passes": ok, "checks": c, "byte_model_total": bm_tot}


def _log_wandb(args, payload):
    try:
        import wandb
    except Exception as ex:  # noqa: BLE001
        print(f"[wandb] import failed: {ex!r}", flush=True); return None
    h = payload["headline"]
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type="g32-tax-empirical", config=payload.get("config", {}),
                     reinit=True)
    flat = {k: v for k, v in h.items() if isinstance(v, (int, float, bool))}
    for c, cv in payload["per_component"].items():
        flat[f"comp_{c}_tax_frac"] = cv["tax_frac"]
        flat[f"comp_{c}_delta_us"] = cv["delta_us"]
        flat[f"comp_{c}_us_g128"] = cv["us_g128"]
        flat[f"comp_{c}_us_g32"] = cv["us_g32"]
    wandb.log(flat)
    wandb.summary.update(flat)
    wandb.summary["verdict"] = h["verdict"]
    wandb.summary["note"] = payload["note"]
    rid = run.id
    wandb.finish()
    return rid


SEED = 707


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=15, help="rounds PER interleave order (x2 total)")
    ap.add_argument("--n-distinct", type=int, default=8)
    ap.add_argument("--m", type=int, default=1, help="activation width (AR decode = 1)")
    ap.add_argument("--anchor-tps", type=float, default=AR_RUNG_OPBENCH_TPS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--output", default=os.path.join(_here, "g32_tax_opbench_results.json"))
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="g32-tax-empirical-land")
    ap.add_argument("--wandb_name", default="land/g32-tax-empirical")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    st = self_test()
    print("[self-test]", json.dumps(st["checks"], indent=2), flush=True)
    print("[byte-model]", json.dumps(st["byte_model_total"], indent=2), flush=True)
    if args.self_test:
        json.dump({"self_test": st}, open(os.path.join(_here, "selftest.json"), "w"), indent=2)
        return
    assert st["self_test_passes"], "self-test failed"

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

    bm_comp, bm_tot = byte_model(args.m)
    paired = paired_measure(dev, args.m, n_distinct, iters, warmup, rounds, SEED)
    print(f"[paired M={args.m}] g128={paired['body_us_g128']:.2f}+-{paired['body_us_g128_ci']:.2f}us "
          f"g32={paired['body_us_g32']:.2f}+-{paired['body_us_g32_ci']:.2f}us "
          f"delta={paired['paired_delta_us']:.3f}+-{paired['paired_delta_us_ci']:.3f}us", flush=True)

    percomp = per_component(dev, args.m, n_distinct, iters, warmup, rounds, SEED)
    print("[per-comp tax_frac]", json.dumps({k: round(v["tax_frac"] * 100, 3) for k, v in percomp.items()}), flush=True)

    payload = compose(gpu, bm_comp, bm_tot, paired, percomp, args.anchor_tps)
    payload["self_test"] = st
    payload["config"] = {"agent": "land", "pr": 707, "kind": "g32-tax-empirical",
                         "hidden": HIDDEN, "intermediate": INTERMEDIATE, "n_layers": N_LAYERS,
                         "m": args.m, "n_distinct": n_distinct, "seed": SEED,
                         "anchor_tps": args.anchor_tps, "ref_official_tps": REF_OFFICIAL_TPS,
                         "selective_subset_frac": SELECTIVE_SUBSET_FRAC, "smoke": args.smoke}
    json.dump(payload, open(args.output, "w"), indent=2)
    print("\n=== HEADLINE ===", flush=True)
    print(json.dumps({k: v for k, v in payload["headline"].items()
                      if isinstance(v, (int, float, bool, str))}, indent=2), flush=True)
    print("\n=== NOTE ===\n" + payload["note"], flush=True)

    if not args.no_wandb and not args.smoke:
        rid = _log_wandb(args, payload)
        payload["wandb_run_id"] = rid
        json.dump(payload, open(args.output, "w"), indent=2)
        print(f"\n[wandb] run_id={rid}", flush=True)


if __name__ == "__main__":
    main()
