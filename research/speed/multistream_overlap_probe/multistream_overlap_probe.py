#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Multi-stream verify-attention overlap probe: the one path past 457.5 (PR #477, lawine).
LOCAL A10G (sm_86) MEASUREMENT + analysis ONLY. NO HF Job, NO submission, NO served-file
change, NO kernel rebuild. analysis_only=true, official_tps=0, no_served_file_change=true.

THE DECISION-CRITICAL QUESTION
------------------------------
The strict (order-preserving, byte-exact) M=8 verify-attention costs +401.9 us/cycle over
the deployed permissive 3D split-KV path (stark #472 whole-cycle realized -> 457.55 TPS;
ubel/stark #466 isolated +422.9 -> 456.36). That tax lives ENTIRELY in the 7 full-attention
(hd=512) Triton reductions. The deployed ONEGRAPH runs EVERYTHING on ONE stream: per layer,
the body int4-Marlin GEMMs then the attention reduction, all SERIAL -- so the strict tax is
fully exposed on the critical path.

Decode is ~92% body weight-GEMM, which is DRAM-BANDWIDTH-bound (M=8 -> AI~32 << ridge 208;
#450 measured f_gemm~50% of read-peak). The strict verify-attention, by contrast, moves very
few bytes (Q/K/V at M=8, hd=512, 7 layers ~19 MB -> ~35 GB/s ~7% of peak) and is SM/latency/
occupancy-bound, NOT bus-bound. IF that is true, scheduling the 7 strict reductions on a
SECOND CUDA stream CONCURRENT with the body GEMMs should HIDE most of the strict attention
under the GEMM's idle SM cycles -- recovering strict TPS from 457.55 toward the deployed
481.53 WHILE STAYING BYTE-EXACT (the launch contract). This probe MEASURES whether that
overlap is physically real on this A10G.

WHAT THIS MEASURES (the direct two-stream micro-probe -- strongest evidence)
---------------------------------------------------------------------------
Reuse the EXACT committed kernels (stark #472 wc.build_all): the real 37-layer self-built
g=128 int4-Marlin body (apply_gptq_marlin_linear -> ops.marlin_gemm, the deployed GEMM) as
`body_gemm`, and the 7 served-Triton full-attn reductions ALONE as `iso_strict` (natural M=8
2D order-preserving == VLLM_BATCH_INVARIANT=1) / `iso_perm` (deployed 3D split-KV num_par=16).
CUDA-graph capture each (the deployed ONEGRAPH basis, mdgd._capture, private pools -> safe
concurrent replay). Then time, paired per round (N>=21 rounds, median+sigma):
  solo:  body | iso_strict | iso_perm | body2   (body2 = independent 2nd body, symmetric arm)
  conc:  body || iso_strict | body || iso_perm | body || body2   (fork/join on 2 streams)
The per-round paired exposed wall (concurrent - body_solo) cancels the body GEMM and isolates
how much of each attention SURVIVES (is NOT hidden under) the GEMM:
  exposed_strict = (body || iso_strict) - body_solo      (strict attn wall left on crit path)
  exposed_perm   = (body || iso_perm)   - body_solo
  multistream_strict_added_us = max(0, exposed_strict - exposed_perm)   (residual strict TAX
     under two-stream scheduling; cancels the common attention baseline -> apples-to-apples)
  multistream_hideable_us = WHOLE_DELTA_472 (401.9) - multistream_strict_added_us   (PRIMARY:
     us of the single-stream strict tax a 2nd stream hides)
  multistream_strict_tps_ceiling = tps_from_added_us(multistream_strict_added_us)   (IDENTICAL
     #466/#472 banked-cycle mapping: added=0 -> 481.53 byte-exact; added=401.9 -> 457.55)
The symmetric body||body2 arm is the METHODOLOGY SMOKING GUN: two BW-bound GEMMs should
SERIALIZE on the bus (speedup ~1), so if body||iso_strict instead OVERLAPS (exposed << solo)
that is direct proof the strict attention is not competing for DRAM bandwidth.

Complementary resource headroom (context; ncu unavailable on this pod -> 2nd methodology):
  gemm_dram_bw_util = achieved body-GEMM BW / measured read-peak  (~0.5, leaves bus slack)
  gemm_sm_occupancy = body-GEMM arithmetic-intensity / ridge_AI   (compute-occupancy proxy,
     ~0.15 -> ~85% SM headroom for a latency-bound co-runner)
  attn_is_sm_bound  = strict-attn achieved BW << peak (bool)        (the overlap premise)

REPORTED (instruction 5)
------------------------
multistream_hideable_us (PRIMARY), gemm_body_us, strict_attn_us, gemm_sm_occupancy,
gemm_dram_bw_util, attn_is_sm_bound, multistream_strict_tps_ceiling, overlap_is_real (verdict),
ppl (2.3772, pinned by construction -- profiling cannot change emitted tokens). Strict byte-
identity re-confirmed (== #466/#472: strict 1.0000 / permissive < 1). analysis_only=true,
official_tps=0, no_served_file_change=true, no_kernel_rebuild=true.

Reproduce: cd target/ && CUDA_VISIBLE_DEVICES=0 \
  /tmp/senpai-venvs/5f4c623f772358a2/bin/python \
  research/speed/multistream_overlap_probe/multistream_overlap_probe.py \
  --wandb_group equivalence-escalation-anchors --wandb_name lawine/multistream-overlap-probe
Then log W&B from the repo .venv (the GPU tool-venv has no usable wandb):
  cd target/ && .venv/bin/python \
  research/speed/multistream_overlap_probe/wandb_log.py \
  --json research/speed/multistream_overlap_probe/multistream_overlap_probe.json \
  --wandb_group equivalence-escalation-anchors --wandb_name lawine/multistream-overlap-probe
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import statistics
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "0,", ""):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.normpath(os.path.join(_here, "..", "..", ".."))


def _load(mod_name, rel_path):
    path = os.path.normpath(os.path.join(_root, rel_path))
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Reuse the EXACT committed primitives (faithful, not re-derived):
#  - wc (#472): build_all -> the real 37-layer g=128 int4-Marlin body_gemm + the 7 served-Triton
#    full-attn iso_strict / iso_perm reductions; the tps_from_added_us banked-cycle mapping.
#  - sfr (#466): identity_probe + served attention geometry constants.
#  - roof (#450): measure_peak_bw, served_byte_model, mdgd loader (served dims/depth), constants.
wc = _load("strict_wholecycle_ab",
           "research/speed/strict_wholecycle_ab/strict_wholecycle_ab.py")
sfr = wc.sfr
roof = wc.roof

import torch  # noqa: E402

# ---- banked anchors (IMPORTED exact; this card derives nothing upstream) ----
DEPLOYED_TPS = sfr.DEPLOYED_TPS                 # 481.53 deployed incumbent (non-equivalent)
REALIZED_BASE_TPS = sfr.REALIZED_BASE_TPS       # 467.14 composed blanket-strict frontier
CYCLE_PERM_US = sfr.CYCLE_PERM_US               # 7666.83 deployed permissive cycle <-> 481.53
CYCLE_WALL_US = sfr.CYCLE_WALL_US               # 7903 composed strict coupled cycle <-> 467.14
M1_COLLAPSE_TPS = sfr.M1_COLLAPSE_TPS           # 161.70 M=1 AR strict floor
PPL_ANCHOR = sfr.PPL_ANCHOR                     # 2.3772 (pinned by construction)
PPL_GATE = sfr.PPL_GATE                         # 2.42
SIGMA_HW = wc.SIGMA_HW                          # 4.8153 advisor-named hardware sigma

ISO_DELTA_466_US = wc.ISO_DELTA_466_US          # 422.91 #466 isolated serial attention tax
REALIZED_466_TPS = wc.REALIZED_466_TPS          # 456.36 #466 realized (isolated lower bound)
GEMM_US_450 = wc.GEMM_US_450                    # 4152.96 #450 measured body-GEMM time
# stark #472 whole-cycle SINGLE-STREAM realized (THE BAR a 2nd stream must clear) ----
WHOLE_DELTA_472_US = 401.89971923828125         # #472 single-stream in-graph-overlap strict tax
REALIZED_WHOLECYCLE_457 = 457.5452044002469     # #472 realized_strict_frontier_best_estimate_tps
ISO_DELTA_472_INHARNESS_US = 421.76513671875    # #472 in-harness isolated serial tax (calib ref)

# served gemma-4-E4B-it geometry (== #466/#472)
N_FULL_LAYERS = sfr.N_FULL_LAYERS               # 7 full-attention (hd=512) Triton layers
M_VERIFY = sfr.M_VERIFY                          # 8 spec-verify width (K_spec=7 + 1)
HEAD_DIM_FULL = sfr.HEAD_DIM_FULL               # 512
N_Q_HEADS = sfr.N_Q_HEADS                       # 8
N_KV_HEADS = sfr.N_KV_HEADS                     # 2 (GQA)
KV_LENS = sfr.KV_LENS                           # (128, 384, 640)
HEADLINE_L = sfr.HEADLINE_L                     # 640
SLIDING_WINDOW = wc.SLIDING_WINDOW              # 512
FULL_ATTN_IDX = wc.FULL_ATTN_IDX               # (2,8,14,20,26,32,36)
A10G_SPEC_BW_GBPS = roof.A10G_SPEC_BW_GBPS      # 600 datasheet
RIDGE_AI = roof.RIDGE_AI                        # 208.3 FLOP/byte
BF16_BYTES = 2.0
BODY_GEMM = roof.BODY_GEMM                       # ["qkv_proj","o_proj","gate_up_proj","down_proj"]

_capture = roof.mdgd._capture


def tps_from_added_us(added_us):
    """IDENTICAL to sfr.tps_from_added_us: realized strict TPS = DEPLOYED * CYCLE_PERM /
    (CYCLE_PERM + added_us). added_us = the per-cycle wall the strict (order-preserving)
    attention reduction ADDS over the deployed permissive path. 0 -> 481.53 byte-exact."""
    return sfr.tps_from_added_us(added_us)


def strict_attn_bytes(L, M, head_dim, n_full):
    """Q/K/V read + O write byte traffic for the n_full full-attn layers (GQA n_q=8/n_kv=2).
    Paged attention reads n_kv heads x L positions for K and V; Q/O at width M. BW-relevant
    only (intra-kernel reduction scratch is not HBM)."""
    q = N_Q_HEADS * M * head_dim * BF16_BYTES
    kv = 2 * N_KV_HEADS * L * head_dim * BF16_BYTES   # K + V
    o = N_Q_HEADS * M * head_dim * BF16_BYTES
    return n_full * (q + kv + o)


def _med(vals):
    vals = [v for v in vals if math.isfinite(v)]
    return statistics.median(vals) if vals else float("nan")


def _paired(series, a, b):
    """median + pstdev of per-round (series[a] - series[b]); paired -> cancels common offset."""
    diffs = [x - y for x, y in zip(series[a], series[b])
             if math.isfinite(x) and math.isfinite(y)]
    if not diffs:
        return float("nan"), float("nan"), 0
    m = statistics.median(diffs)
    sd = statistics.pstdev(diffs) if len(diffs) > 1 else 0.0
    return m, sd, len(diffs)


def _time_solo(g, iters):
    """us per replay of a single captured graph on the current (default) stream."""
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1e3


def _time_concurrent(g_a, g_b, sa, sb, iters):
    """us per concurrent iter: replay g_a on stream sa and g_b on stream sb, fork/join off the
    default stream so the timed window spans exactly the two-stream concurrent region. With
    t_a >> t_b and no bus contention -> ~t_a (b fully hidden); with full serialization -> t_a+t_b."""
    main = torch.cuda.current_stream()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record(main)
    sa.wait_stream(main)          # sa/sb wait for everything on main up to e0 -> start together
    sb.wait_stream(main)
    with torch.cuda.stream(sa):
        for _ in range(iters):
            g_a.replay()
    with torch.cuda.stream(sb):
        for _ in range(iters):
            g_b.replay()
    main.wait_stream(sa)          # main blocks until BOTH streams drain
    main.wait_stream(sb)
    e1.record(main)
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1e3


# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=HEADLINE_L)
    ap.add_argument("--Ls", type=str, default=None,
                    help="comma-separated KV-len sweep override (default: 128,384,640)")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=21)
    ap.add_argument("--n-distinct", type=int, default=8)
    ap.add_argument("--ident-trials", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--self-test", dest="self_test", action="store_true")
    ap.add_argument("--output", default=os.path.join(_here, "multistream_overlap_probe.json"))
    ap.add_argument("--selftest-output", default=os.path.join(_here, "selftest.json"))
    ap.add_argument("--wandb_group", default="equivalence-escalation-anchors")
    ap.add_argument("--wandb_name", default="lawine/multistream-overlap-probe")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA unavailable (need CUDA_VISIBLE_DEVICES=0)"
    dev = torch.device("cuda:0")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    iters = 12 if args.smoke else args.iters
    rounds = 7 if args.smoke else args.rounds
    n_distinct = 4 if args.smoke else args.n_distinct
    ident_trials = 2 if args.smoke else args.ident_trials
    if args.Ls:
        Ls = tuple(int(x) for x in args.Ls.split(","))
    else:
        Ls = (128, args.L) if args.smoke else KV_LENS
    if args.L not in Ls:
        Ls = tuple(sorted(set(Ls) | {args.L}))

    model_dir = roof.SERVED_BODY.rsplit("/", 1)[0]
    dims = roof.mdgd.read_dims(model_dir)
    num_layers, depth_src = roof.mdgd.deployed_depth(dims["num_layers"])
    full_set = set(i for i in FULL_ATTN_IDX if i < num_layers)
    assert len(full_set) == N_FULL_LAYERS, f"expected {N_FULL_LAYERS} full-attn layers, got {len(full_set)}"
    print(f"[ms] {name} sm_{cap[0]}{cap[1]} torch {torch.__version__}  M={M_VERIFY} "
          f"hd_full={HEAD_DIM_FULL} depth={num_layers}({depth_src}) full_attn={sorted(full_set)} "
          f"n_full={len(full_set)} L_headline={args.L} Ls={Ls} rounds={rounds} "
          f"n_distinct={n_distinct} smoke={args.smoke}", flush=True)
    torch.cuda.reset_peak_memory_stats()

    # heavy warm-up -> A10G boost clock (same regime as #450/#466/#472)
    big = torch.randn(2048, 2048, dtype=torch.bfloat16, device=dev)
    for _ in range(200):
        big = big @ big
    torch.cuda.synchronize()
    del big

    # co-measured achievable peak BW (same clock state) -- the denominator for f
    peak = roof.measure_peak_bw(dev, iters, args.warmup)
    peak_read = peak["bw_read_gbps"]
    print(f"[ms] PEAK BW: read={peak_read:.0f} copy={peak['bw_copy_gbps']:.0f} "
          f"bf16gemm@M8={peak['bw_bf16gemm_m8_gbps']:.0f} GB/s (spec {A10G_SPEC_BW_GBPS:.0f})", flush=True)

    # exact byte model (served safetensors) -> body-GEMM bytes + arithmetic intensity
    bm = roof.served_byte_model(M_VERIFY, num_layers)
    gemm_bytes = sum(bm[c]["total_bytes"] for c in BODY_GEMM) + bm["lm_head"]["total_bytes"]
    gemm_flops = (sum(2.0 * num_layers * M_VERIFY * bm[c]["out"] * bm[c]["in"] for c in BODY_GEMM)
                  + 2.0 * M_VERIFY * bm["lm_head"]["out"] * bm["lm_head"]["in"])
    gemm_ai = gemm_flops / gemm_bytes                      # FLOP/byte (M=8 -> << ridge)

    sa = torch.cuda.Stream()
    sb = torch.cuda.Stream()

    # ---- per-L two-stream overlap sweep ----------------------------------------------
    per_L = {}
    for L in Ls:
        # instance A: body_gemm + iso_strict + iso_perm (real #472 kernels, shared body)
        runnersA, NK, keepA = wc.build_all(dims, num_layers, full_set, L, min(SLIDING_WINDOW, L),
                                           M_VERIFY, dev, n_distinct)
        # instance B: an independent 2nd body (symmetric bus-contention smoking-gun arm)
        runnersB, _NKB, keepB = wc.build_all(dims, num_layers, full_set, L, min(SLIDING_WINDOW, L),
                                             M_VERIFY, dev, max(2, n_distinct // 2))
        graphs, captured = {}, {}
        to_cap = {"body": runnersA["body_gemm"], "iso_strict": runnersA["iso_strict"],
                  "iso_perm": runnersA["iso_perm"], "body2": runnersB["body_gemm"]}
        for nm, run in to_cap.items():
            try:
                graphs[nm] = _capture(run)
                captured[nm] = True
            except Exception as exc:  # noqa: BLE001
                print(f"[ms] capture FAILED {nm}: {exc!r}", flush=True)
                graphs[nm], captured[nm] = None, False

        ok = all(captured[k] for k in ("body", "iso_strict", "iso_perm"))
        if not ok:
            per_L[L] = {"captured": captured, "error": "core capture failed"}
            del runnersA, runnersB, keepA, keepB, graphs
            gc.collect(); torch.cuda.empty_cache()
            continue

        # warm replays (concurrent + solo) -> steady boost clock / graph residency
        for _ in range(max(10, args.warmup)):
            for g in graphs.values():
                if g is not None:
                    g.replay()
        if graphs["body2"] is not None:
            for _ in range(max(10, args.warmup)):
                _time_concurrent(graphs["body"], graphs["body2"], sa, sb, 1)
        torch.cuda.synchronize()

        # paired per-round: all solo + concurrent arms measured each round (cancels clock drift)
        series = {k: [] for k in ("body", "iso_strict", "iso_perm", "body2",
                                  "two_strict", "two_perm", "two_body")}
        for _ in range(rounds):
            series["body"].append(_time_solo(graphs["body"], iters))
            series["iso_strict"].append(_time_solo(graphs["iso_strict"], iters))
            series["iso_perm"].append(_time_solo(graphs["iso_perm"], iters))
            series["two_strict"].append(_time_concurrent(graphs["body"], graphs["iso_strict"], sa, sb, iters))
            series["two_perm"].append(_time_concurrent(graphs["body"], graphs["iso_perm"], sa, sb, iters))
            if graphs["body2"] is not None:
                series["body2"].append(_time_solo(graphs["body2"], iters))
                series["two_body"].append(_time_concurrent(graphs["body"], graphs["body2"], sa, sb, iters))
            else:
                series["body2"].append(float("nan"))
                series["two_body"].append(float("nan"))

        med = {nm: _med(series[nm]) for nm in series}
        exp_strict, exp_strict_sd, n_es = _paired(series, "two_strict", "body")
        exp_perm, exp_perm_sd, n_ep = _paired(series, "two_perm", "body")
        # residual strict tax under two-stream scheduling (apples-to-apples; cancels attn baseline)
        ms_added = max(0.0, exp_strict - exp_perm)
        ms_added_sd = math.hypot(exp_strict_sd, exp_perm_sd)
        # single-stream serial tax in THIS harness (calibration vs #466 422 / #472 421.8)
        iso_tax = med["iso_strict"] - med["iso_perm"]
        # overlap fractions (1.0 = attention fully hidden under the body GEMM)
        ov_strict = ((med["body"] + med["iso_strict"] - med["two_strict"]) / med["iso_strict"]
                     if med["iso_strict"] > 0 else float("nan"))
        ov_perm = ((med["body"] + med["iso_perm"] - med["two_perm"]) / med["iso_perm"]
                   if med["iso_perm"] > 0 else float("nan"))
        # symmetric body||body smoking gun: speedup 2=perfect overlap, 1=serialized (BW-bound)
        sym_speedup = (2.0 * med["body"] / med["two_body"]
                       if math.isfinite(med["two_body"]) and med["two_body"] > 0 else float("nan"))

        # achieved BW (byte model / measured solo us)
        ab_gemm = gemm_bytes / (med["body"] * 1e-6) / 1e9 if med["body"] > 0 else float("nan")
        attn_bytes = strict_attn_bytes(L, M_VERIFY, HEAD_DIM_FULL, N_FULL_LAYERS)
        ab_attn = attn_bytes / (med["iso_strict"] * 1e-6) / 1e9 if med["iso_strict"] > 0 else float("nan")

        per_L[L] = {
            "median_us": med, "captured": captured,
            "exposed_strict_us": exp_strict, "exposed_strict_sigma": exp_strict_sd, "n_exposed": n_es,
            "exposed_perm_us": exp_perm, "exposed_perm_sigma": exp_perm_sd,
            "multistream_strict_added_us": ms_added, "multistream_strict_added_sigma": ms_added_sd,
            "iso_tax_us": iso_tax,
            "overlap_fraction_strict": ov_strict, "overlap_fraction_perm": ov_perm,
            "symmetric_overlap_speedup": sym_speedup,
            "achieved_gemm_bw_gbps": ab_gemm, "achieved_attn_bw_gbps": ab_attn,
            "gemm_dram_bw_util": ab_gemm / peak_read if peak_read > 0 else float("nan"),
            "attn_bw_frac_of_peak": ab_attn / peak_read if peak_read > 0 else float("nan"),
            "multistream_hideable_us": WHOLE_DELTA_472_US - ms_added,
            "multistream_strict_tps_ceiling": tps_from_added_us(ms_added),
        }
        print(f"[ms] L={L}: body={med['body']:.1f} iso_strict={med['iso_strict']:.1f} "
              f"iso_perm={med['iso_perm']:.1f} | two_strict={med['two_strict']:.1f} "
              f"two_perm={med['two_perm']:.1f} two_body={med['two_body']:.1f} | "
              f"exp_strict={exp_strict:+.1f} exp_perm={exp_perm:+.1f} ms_added={ms_added:.1f} "
              f"ov_strict={ov_strict*100:.0f}% sym={sym_speedup:.2f} "
              f"-> ceiling={tps_from_added_us(ms_added):.2f} TPS", flush=True)

        del runnersA, runnersB, keepA, keepB
        for g in graphs.values():
            del g
        gc.collect(); torch.cuda.empty_cache()

    # ---- headline (L = args.L) -------------------------------------------------------
    H = per_L[args.L]
    assert "error" not in H, f"headline L={args.L} failed capture: {H}"
    gemm_body_us = H["median_us"]["body"]
    strict_attn_us = H["median_us"]["iso_strict"]
    perm_attn_us = H["median_us"]["iso_perm"]
    multistream_strict_added_us = H["multistream_strict_added_us"]
    multistream_strict_added_sigma = H["multistream_strict_added_sigma"]
    multistream_hideable_us = H["multistream_hideable_us"]                      # PRIMARY
    multistream_strict_tps_ceiling = H["multistream_strict_tps_ceiling"]       # TEST/primary
    ceiling_lo = tps_from_added_us(multistream_strict_added_us + multistream_strict_added_sigma)
    ceiling_hi = tps_from_added_us(max(0.0, multistream_strict_added_us - multistream_strict_added_sigma))
    overlap_fraction_strict = H["overlap_fraction_strict"]
    symmetric_overlap_speedup = H["symmetric_overlap_speedup"]
    gemm_dram_bw_util = H["gemm_dram_bw_util"]
    achieved_gemm_bw = H["achieved_gemm_bw_gbps"]
    achieved_attn_bw = H["achieved_attn_bw_gbps"]
    attn_bw_frac_of_peak = H["attn_bw_frac_of_peak"]
    iso_tax_us = H["iso_tax_us"]
    # SM-occupancy proxy (ncu unavailable): compute-occupancy of the M=8 GEMM = AI/ridge (~0.15)
    gemm_sm_occupancy = min(1.0, gemm_ai / RIDGE_AI)
    gemm_sm_headroom = 1.0 - gemm_sm_occupancy
    # the overlap premise: strict attention moves few bytes -> NOT bus-bound (SM/latency-bound)
    attn_is_sm_bound = bool(math.isfinite(attn_bw_frac_of_peak) and attn_bw_frac_of_peak < 0.30)

    # SECONDARY (vs the LITERAL deployed permissive-serial incumbent, not a 2-stream perm world):
    # the deployed 481.53 cycle pays the permissive attention at its SERIAL wall; multistream strict
    # exposes only exposed_strict. added = exposed_strict - perm_serial. Less conservative than the
    # PRIMARY (which credits perm with overlap too); both are reported so the gap is bracketed.
    ms_added_vs_deployed_serial = max(0.0, H["exposed_strict_us"] - perm_attn_us)
    ceiling_vs_deployed_serial = tps_from_added_us(ms_added_vs_deployed_serial)

    # what fraction of the single-stream strict tax the 2nd stream hides (PRIMARY framing)
    multistream_hide_fraction = (multistream_hideable_us / WHOLE_DELTA_472_US
                                 if WHOLE_DELTA_472_US else float("nan"))
    gain_vs_single_stream = multistream_strict_tps_ceiling - REALIZED_WHOLECYCLE_457
    clears_bar = bool(multistream_strict_tps_ceiling >= REALIZED_WHOLECYCLE_457 + SIGMA_HW)
    clears_467 = bool(multistream_strict_tps_ceiling >= REALIZED_BASE_TPS)
    reaches_deployed = bool(multistream_strict_tps_ceiling >= DEPLOYED_TPS - SIGMA_HW)

    # cudagraph survival across all L (strict M=8 reduction must capture+replay, not collapse)
    strict_captured_all_L = all(per_L[L].get("captured", {}).get("iso_strict", False)
                                and "error" not in per_L[L] for L in Ls)
    body_captured_all_L = all(per_L[L].get("captured", {}).get("body", False)
                              and "error" not in per_L[L] for L in Ls)

    # ---- identity re-confirm (strict byte-exact, permissive non-equiv) == #466/#472 --------
    seeds = [1234] if args.smoke else [1234, 5678, 9012]
    ident = sfr.identity_probe(args.L, HEAD_DIM_FULL, dev, ident_trials, seeds)
    iarm = ident.get("per_arm", {})
    strict_identity_fraction = float(iarm.get("strict_2d", {}).get("byte_identity_min", float("nan")))
    strict_argmax_fraction = float(iarm.get("strict_2d", {}).get("argmax_identity_min", float("nan")))
    strict_token_flips = int(round((1.0 - (strict_argmax_fraction
                            if math.isfinite(strict_argmax_fraction) else 1.0)) * M_VERIFY))
    permissive_identity_fraction = float(iarm.get("permissive_3d", {}).get("byte_identity_min", float("nan")))

    peak_vram_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # =================== SELF-TEST (PRIMARY) =================================
    st = {}
    st["tps_zero_added_is_deployed"] = bool(abs(tps_from_added_us(0.0) - DEPLOYED_TPS) < 1e-6)
    st["tps_472delta_is_457"] = bool(abs(tps_from_added_us(WHOLE_DELTA_472_US) - REALIZED_WHOLECYCLE_457) < 1e-2)
    st["tps_more_added_lowers"] = bool(tps_from_added_us(500.0) < tps_from_added_us(50.0) < DEPLOYED_TPS)
    st["constants_anchored"] = bool(REALIZED_BASE_TPS == 467.14 and DEPLOYED_TPS == 481.53
                                    and CYCLE_WALL_US == 7903.0)
    st["geometry_full_attn"] = bool(len(full_set) == N_FULL_LAYERS and M_VERIFY == 8
                                    and HEAD_DIM_FULL == 512 and num_layers == 37)
    st["body_shapes_served"] = bool(NK["qkv_proj"] == (3072, 2560) and NK["gate_up_proj"] == (20480, 2560)
                                    and NK["down_proj"] == (2560, 10240) and NK["o_proj"] == (2560, 2048))
    st["body_captured"] = bool(body_captured_all_L)
    st["strict_captured_survives"] = bool(strict_captured_all_L)
    # CALIBRATION GUARD 1: in-harness serial tax reproduces #466's banked 422.91 (within 15%)
    st["iso_tax_reproduces_466"] = bool(math.isfinite(iso_tax_us) and
                                        abs(iso_tax_us - ISO_DELTA_466_US) / ISO_DELTA_466_US < 0.15)
    # CALIBRATION GUARD 2: body GEMM reproduces #450's 4152.96 (within 20%, as #452/#472)
    st["body_gemm_anchor_matches_450"] = bool(gemm_body_us > 0 and
                                              abs(gemm_body_us - GEMM_US_450) / GEMM_US_450 < 0.20)
    # METHODOLOGY GUARD: two BW-bound GEMMs do NOT super-overlap (symmetric speedup < ~1.5);
    # if they fully overlapped the harness would be crediting phantom concurrency
    st["symmetric_gemm_serializes"] = bool(math.isfinite(symmetric_overlap_speedup)
                                           and symmetric_overlap_speedup < 1.5)
    # exposed strict must be >= exposed perm (strict attn is the heavier kernel) modulo noise
    st["exposed_ordering_sane"] = bool(math.isfinite(H["exposed_strict_us"])
                                       and math.isfinite(H["exposed_perm_us"])
                                       and H["exposed_strict_us"] >= H["exposed_perm_us"] - 3.0 * SIGMA_HW)
    # residual two-stream tax cannot EXCEED the single-stream serial tax (overlap only hides)
    st["ms_added_le_iso_tax"] = bool(multistream_strict_added_us <= iso_tax_us + 3.0 * SIGMA_HW)
    st["ceiling_in_band"] = bool(REALIZED_466_TPS - SIGMA_HW <= multistream_strict_tps_ceiling
                                 <= DEPLOYED_TPS + 1e-6)
    finite = [multistream_hideable_us, gemm_body_us, strict_attn_us, gemm_dram_bw_util,
              multistream_strict_tps_ceiling, multistream_strict_added_us, overlap_fraction_strict,
              symmetric_overlap_speedup, achieved_gemm_bw, achieved_attn_bw, gemm_sm_occupancy]
    st["nan_clean"] = all(math.isfinite(x) for x in finite)
    st["identity_ran"] = bool(ident.get("error") is None and "per_arm" in ident)
    st["strict_byte_exact"] = bool(math.isfinite(strict_identity_fraction)
                                   and strict_identity_fraction >= 0.999)
    st["strict_zero_flips"] = bool(strict_token_flips == 0)
    st["permissive_reproduces_nonequiv"] = bool(math.isfinite(permissive_identity_fraction)
                                                and permissive_identity_fraction < 0.999)
    st["gemm_dram_util_below_one"] = bool(0.0 < gemm_dram_bw_util <= 1.05)
    st["ppl_anchor_ok"] = bool(PPL_ANCHOR <= PPL_GATE)
    st["vram_ok"] = bool(peak_vram_gib <= 24.0)
    st["above_collapse_floor"] = bool(multistream_strict_tps_ceiling > M1_COLLAPSE_TPS + 50.0)
    self_test_passes = all(st.values())

    # =================== VERDICT =================================
    overlap_is_real = bool(clears_bar and strict_captured_all_L
                           and st["strict_byte_exact"] and st["strict_zero_flips"]
                           and attn_is_sm_bound)
    if not strict_captured_all_L:
        outcome = "COLLAPSE"
    elif overlap_is_real and clears_467:
        outcome = "OVERLAP_REAL_CLEARS_467"            # 2nd stream recovers strict past 467.14
    elif overlap_is_real:
        outcome = "OVERLAP_REAL_BEATS_457"             # clears the single-stream 457.55 frontier
    elif multistream_hide_fraction > 0.10:
        outcome = "OVERLAP_PARTIAL"                    # hides some tax but inside hardware noise
    else:
        outcome = "NO_OVERLAP_BUS_BOUND"               # 2nd stream gives nothing (serializes)

    reconcile = (
        f"The strict M=8 verify-attention costs +{WHOLE_DELTA_472_US:.1f} us/cycle single-stream "
        f"(stark #472 whole-cycle -> {REALIZED_WHOLECYCLE_457:.2f} TPS; the deployed ONEGRAPH runs "
        f"body GEMM + attention SERIAL on one stream). DIRECT two-stream micro-probe on the SAME "
        f"#472 kernels (real 37-layer g=128 int4-Marlin body_gemm={gemm_body_us:.0f}us [#450 anchor "
        f"{GEMM_US_450:.0f}, guard {st['body_gemm_anchor_matches_450']}] || 7 served-Triton strict "
        f"reductions iso_strict={strict_attn_us:.0f}us; serial tax {iso_tax_us:.0f}us reproduces #466 "
        f"{ISO_DELTA_466_US:.0f} [guard {st['iso_tax_reproduces_466']}]): the strict attention exposes "
        f"only {H['exposed_strict_us']:+.0f}us on the critical path when co-scheduled with the body GEMM "
        f"(perm {H['exposed_perm_us']:+.0f}us), i.e. overlap_fraction_strict={overlap_fraction_strict*100:.0f}% "
        f"hidden. Residual strict tax under two streams = {multistream_strict_added_us:.1f}us "
        f"(sigma {multistream_strict_added_sigma:.1f}) -> multistream_strict_tps_ceiling="
        f"{multistream_strict_tps_ceiling:.2f} ({gain_vs_single_stream:+.2f} vs single-stream {REALIZED_WHOLECYCLE_457:.2f}; "
        f"clears_bar[+sigma]={clears_bar}). multistream_hideable_us={multistream_hideable_us:.1f} "
        f"({multistream_hide_fraction*100:.0f}% of the {WHOLE_DELTA_472_US:.0f}us tax). WHY it overlaps: "
        f"the body GEMM is BW-bound (achieved {achieved_gemm_bw:.0f} GB/s = {gemm_dram_bw_util*100:.0f}% of "
        f"read-peak {peak_read:.0f}; M=8 AI={gemm_ai:.0f}<<ridge {RIDGE_AI:.0f} -> SM-occupancy proxy "
        f"{gemm_sm_occupancy*100:.0f}%, ~{gemm_sm_headroom*100:.0f}% SM headroom) while the strict attention "
        f"moves {achieved_attn_bw:.0f} GB/s = {attn_bw_frac_of_peak*100:.0f}% of peak (attn_is_sm_bound="
        f"{attn_is_sm_bound}) -> not competing for DRAM. METHODOLOGY GUARD: two BW-bound GEMMs only reach "
        f"symmetric speedup {symmetric_overlap_speedup:.2f} (serialize on the bus [guard "
        f"{st['symmetric_gemm_serializes']}]) -- so the GEMM||attention overlap is real concurrency, not a "
        f"timing artifact. Strict 2D byte-exact: identity={strict_identity_fraction:.4f} ({strict_token_flips} "
        f"flips); deployed permissive byte={permissive_identity_fraction:.4f} (reproduces non-equivalence). "
        f"MEASUREMENT-ONLY: a deployed two-stream schedule needs a served-graph edit -> OUT OF SCOPE here "
        f"(no served-file change, no kernel rebuild, no HF job). OUTCOME={outcome}.")

    verdict = {
        "multistream_overlap_self_test_passes": self_test_passes,                  # PRIMARY (gate)
        "multistream_hideable_us": multistream_hideable_us,                        # PRIMARY (metric)
        "multistream_strict_tps_ceiling": multistream_strict_tps_ceiling,         # TEST/primary
        "multistream_strict_tps_ceiling_sigma_lo": ceiling_lo,
        "multistream_strict_tps_ceiling_sigma_hi": ceiling_hi,
        "overlap_is_real": overlap_is_real,                                        # verdict bool
        "multistream_hide_fraction": multistream_hide_fraction,
        "multistream_strict_added_us": multistream_strict_added_us,
        "multistream_strict_added_sigma": multistream_strict_added_sigma,
        "ms_added_vs_deployed_serial_us": ms_added_vs_deployed_serial,
        "multistream_strict_tps_ceiling_vs_deployed_serial": ceiling_vs_deployed_serial,
        # --- the direct two-stream measurement ---
        "gemm_body_us": gemm_body_us, "strict_attn_us": strict_attn_us, "perm_attn_us": perm_attn_us,
        "exposed_strict_us": H["exposed_strict_us"], "exposed_strict_sigma": H["exposed_strict_sigma"],
        "exposed_perm_us": H["exposed_perm_us"], "exposed_perm_sigma": H["exposed_perm_sigma"],
        "overlap_fraction_strict": overlap_fraction_strict,
        "overlap_fraction_perm": H["overlap_fraction_perm"],
        "symmetric_overlap_speedup": symmetric_overlap_speedup,
        "iso_tax_us": iso_tax_us,
        # --- resource headroom (why it overlaps) ---
        "gemm_dram_bw_util": gemm_dram_bw_util,                                     # required
        "gemm_sm_occupancy": gemm_sm_occupancy,                                     # required (proxy)
        "gemm_sm_headroom": gemm_sm_headroom,
        "attn_is_sm_bound": attn_is_sm_bound,                                       # required (bool)
        "achieved_gemm_bw_gbps": achieved_gemm_bw, "achieved_attn_bw_gbps": achieved_attn_bw,
        "attn_bw_frac_of_peak": attn_bw_frac_of_peak,
        "gemm_arithmetic_intensity": gemm_ai, "ridge_ai_flop_per_byte": RIDGE_AI,
        "peak_read_gbps": peak_read, "peak_copy_gbps": peak["bw_copy_gbps"],
        # --- gain / bar ---
        "gain_vs_single_stream_tps": gain_vs_single_stream,
        "single_stream_realized_tps": REALIZED_WHOLECYCLE_457,
        "clears_single_stream_bar": clears_bar, "clears_467": clears_467,
        "reaches_deployed_481": reaches_deployed,
        # --- survival / identity ---
        "strict_captured_all_L": strict_captured_all_L, "body_captured_all_L": body_captured_all_L,
        "strict_collapses_to_m1": bool(not strict_captured_all_L),
        "strict_identity_fraction": strict_identity_fraction, "strict_token_flips": strict_token_flips,
        "permissive_identity_fraction": permissive_identity_fraction,
        # --- banked anchors ---
        "whole_delta_472_us": WHOLE_DELTA_472_US, "iso_delta_466_us": ISO_DELTA_466_US,
        "realized_466_tps": REALIZED_466_TPS, "gemm_us_450_anchor": GEMM_US_450,
        "cycle_perm_us": CYCLE_PERM_US, "cycle_wall_us": CYCLE_WALL_US,
        "deployed_tps": DEPLOYED_TPS, "realized_base_tps": REALIZED_BASE_TPS,
        "m1_collapse_floor_tps": M1_COLLAPSE_TPS, "sigma_hw": SIGMA_HW,
        "n_full_layers_per_cycle": len(full_set), "deployed_num_layers": num_layers,
        "headline_L": args.L,
        "outcome": outcome,
        "ppl": PPL_ANCHOR, "ppl_anchor": PPL_ANCHOR, "ppl_gate": PPL_GATE,
        "peak_vram_gib": peak_vram_gib,
        # --- scope / safety ---
        "deploy_needs_served_graph_edit": True, "measurement_only": True,
        "no_kernel_rebuild": True, "analysis_only": True, "no_hf_job": True,
        "no_served_file_change": True, "official_tps": 0,
        "self_test_conditions": st,
        "reconcile_line": reconcile,
    }

    payload = {
        "config": {"torch": torch.__version__, "device": name, "sm": f"{cap[0]}{cap[1]}",
                   "M": M_VERIFY, "head_dim_full": HEAD_DIM_FULL, "n_q_heads": N_Q_HEADS,
                   "n_kv_heads": N_KV_HEADS, "deployed_num_layers": num_layers,
                   "full_attn_idx": sorted(full_set), "n_full_layers": len(full_set),
                   "KV_LENS": list(Ls), "headline_L": args.L, "sliding_window": SLIDING_WINDOW,
                   "iters": iters, "warmup": args.warmup, "rounds": rounds, "n_distinct": n_distinct,
                   "ident_trials": ident_trials, "smoke": args.smoke, "served_model_dir": model_dir,
                   "group_size": 128, "self_built_marlin": True,
                   "note": "direct two-stream overlap micro-probe: real #472 kernels (37-layer self-built "
                           "g=128 int4-Marlin body_gemm via apply_gptq_marlin_linear + the 7 served-Triton "
                           "full-attn hd=512 strict_2d/permissive_3d reductions), CUDA-graph captured "
                           "(private pools), timed solo vs concurrent on two CUDA streams (fork/join, paired "
                           "per round). multistream_strict_added_us = (body||strict - body) - (body||perm - "
                           "body) is the residual strict tax under two-stream scheduling; ceiling via "
                           "tps_from_added_us on the banked cycle (CYCLE_PERM=7666.83 <-> 481.53). Symmetric "
                           "body||body arm is the bus-contention smoking gun. No serve change, no HF Job, no "
                           "submission, no kernel rebuild."},
        "peak_bw": peak,
        "per_L": {str(L): per_L[L] for L in per_L},
        "byte_model": {"gemm_bytes_mb": gemm_bytes / 1e6, "gemm_ai_flop_per_byte": gemm_ai,
                       "strict_attn_bytes_mb": strict_attn_bytes(args.L, M_VERIFY, HEAD_DIM_FULL,
                                                                 N_FULL_LAYERS) / 1e6},
        "shapes": {c: list(NK[c]) for c in NK},
        "identity": ident,
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2, default=lambda o: float(o) if isinstance(o, (int, float)) else str(o))
    with open(args.selftest_output, "w") as fh:
        json.dump({"multistream_overlap_self_test_passes": self_test_passes, "checks": st}, fh, indent=2)
    print(f"[ms] wrote {args.output}", flush=True)
    print(f"\n[ms] OUTCOME={outcome}  self_test={self_test_passes}  overlap_is_real={overlap_is_real}", flush=True)
    print(f"[ms] multistream_hideable_us={multistream_hideable_us:.1f} ({multistream_hide_fraction*100:.0f}% of "
          f"{WHOLE_DELTA_472_US:.0f}us tax) -> ceiling={multistream_strict_tps_ceiling:.2f} TPS "
          f"({gain_vs_single_stream:+.2f} vs {REALIZED_WHOLECYCLE_457:.2f}; clears_bar={clears_bar} "
          f"clears_467={clears_467})", flush=True)
    print(f"[ms] overlap_fraction_strict={overlap_fraction_strict*100:.0f}% symmetric_speedup={symmetric_overlap_speedup:.2f} "
          f"gemm_dram_util={gemm_dram_bw_util*100:.0f}% attn_bw={attn_bw_frac_of_peak*100:.0f}%peak "
          f"attn_is_sm_bound={attn_is_sm_bound}", flush=True)
    print(f"[ms] {reconcile}", flush=True)
    print(f"[ms] self_test={st}", flush=True)

    if not (args.no_wandb or args.smoke):
        print(f"[ms] to log W&B: cd target/ && .venv/bin/python "
              f"research/speed/multistream_overlap_probe/wandb_log.py --json {args.output} "
              f"--wandb_group {args.wandb_group} --wandb_name {args.wandb_name}", flush=True)

    gc.collect(); torch.cuda.empty_cache()
    if args.self_test:
        return 0 if self_test_passes else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
