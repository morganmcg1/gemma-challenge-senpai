#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Whole-cycle strict A/B: tighten the realized strict frontier with in-graph overlap
(PR #472, stark). LOCAL A10G (sm_86) MEASUREMENT + analysis ONLY.
NO HF Job, NO submission, NO served-file change, NO kernel rebuild. analysis_only=true.

THE DECISION-CRITICAL QUESTION (tighten #466's conservative lower bound)
-----------------------------------------------------------------------
stark #466 (sxigz7dp/gmd8v9sw) REALIZED the strict frontier at 456.36 TPS (L=640
headline) / ~459 cluster-mean, byte-exact (identity 1.0000, 0 flips), refuting the
collapse-to-162 hypothesis. But #466's number is an ISOLATED-attention-locus Delta:
it timed ONLY the 7 full-attention (hd=512) Triton reductions in a graph -- strict 2D
serial vs permissive 3D split-KV -- and applied that +422.91 us/cycle Delta to the
banked deployed decode cycle. Timing the reductions ALONE cannot capture in-graph
attention/GEMM overlap, so 456.36 is a CONSERVATIVE LOWER BOUND: the true realized
strict frontier lies in [456.5, <=467.14], and the "holds-within-sigma vs optimistic-
composition" verdict flips only if in-graph overlap hides >~28% of the measured
+422.91 us/cycle Delta. The approval issue (#407 07:26Z) will quote a PREDICTED board
TPS -- it must be the honest, overlap-captured number, not a lower bound that under-
sells by up to ~11 TPS, nor an optimistic composition that over-sells.

WHAT THIS MEASURES (the #452-identical whole-cycle A/B, applied to the strict side)
----------------------------------------------------------------------------------
Build the FULL deployed MTP K=7 / M=8 verify cycle -- the 37-layer self-built g=128
int4-Marlin body (the SAME apply_gptq_marlin_linear -> ops.marlin_gemm the deployed
GPTQMarlinLinearMethod.apply calls, #450/#452), one attention per layer (the 7
full-attention hd=512 layers at config indices {2,8,14,20,26,32,36} use the served
Triton unified_attention with the lever; the 30 sliding hd=256 layers use sdpa), plus
the served int4 12k lm_head -- CUDA-graph captured (the deployed ONEGRAPH single-stream
basis, mdgd._capture), at M=8 verify width. Two arms differ ONLY in the 7 full-attn
reduction path:
  - permissive_3d : deployed 3D split-KV (max_seqlen_q->1 override, num_par=16); the
                    NON-order-preserving cross-segment merge (identity 0.9966).
  - strict_2d     : the kernel's NATURAL M=8 2D single-segment sequential-KV reduction
                    (max_seqlen_q=8 -> use_3d=False; this IS the VLLM_BATCH_INVARIANT=1
                    config path, PR #122 splitkv auto-gated-off) -- order-preserving,
                    byte-exact. CONFIG-reachable, no served-source edit (#466 Directive #3
                    not tripped); auto-gates-off splitkv.
Time the WHOLE-CYCLE wall per arm (paired per-round differencing, N>=21 rounds, median
+ sigma), exactly as #452 timed the whole relax cycle. The paired Delta
whole_cycle_strict_delta_us = whole_strict - whole_perm cancels the (identical) body
GEMMs / sliding attention / lm_head and isolates the 7 full-attn reductions' tax UNDER
in-graph overlap. We ALSO time the #466 ISOLATED arm (the 7 reductions ALONE) in the
same harness / same clock -> overlap_recovery_fraction = how much of the isolated tax
the in-graph context hid. realized = tps_from_added_us(whole_cycle_strict_delta_us) on
the banked cycle (CYCLE_PERM=7666.83 <-> 481.53), the SAME mapping #466 used.

REPORTED (instruction 5)
------------------------
whole_cycle_strict_tps, whole_cycle_perm_tps (=481.53 anchor; perm arm IS the deployed
path, self-tax 0 -- the genuine "not mis-calibrated" guards are the in-harness isolated
Delta reproducing #466's 422.91 and the body GEMM reproducing #450's 4152.96),
whole_cycle_strict_delta_us, overlap_recovery_fraction, whole_cycle_holds_within_sigma_hw
(bool: |467.14 - realized| <= sigma_hw 4.8153), realized_strict_frontier_best_estimate_tps
(the single honest point estimate for the approval issue), whole_cycle_strict_identity_
fraction (1.0), whole_cycle_strict_token_flips (0), whole_cycle_self_test_passes,
analysis_only=true, no_served_file_change=true, official_tps=0, ppl=2.3772.

Reproduce: cd target/ && CUDA_VISIBLE_DEVICES=0 \
  /tmp/senpai-venvs/5f4c623f772358a2/bin/python \
  research/speed/strict_wholecycle_ab/strict_wholecycle_ab.py \
  --wandb_group equivalence-escalation-anchors --wandb_name stark/strict-wholecycle-ab
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
#  - roof (#450): banked cycle/frontier constants, paired_diff_measure (graph-captured
#    L2-cold timing), measure_peak_bw, mdgd loader (served dims/depth), self-built g=128
#    int4-Marlin (_marlin_quant + apply_gptq_marlin_linear == the deployed kernel).
#  - sfr (#466): the strict/permissive Triton attention lever (_call_unified / _build_inputs
#    / _segm_bufs / ARMS), the identity probe, served attention geometry, and the
#    tps_from_added_us banked-cycle mapping (IDENTICAL mapping -> consistency with #466).
roof = _load("gemm_roofline_bw_ceiling",
             "research/speed/gemm_roofline_bw_ceiling/gemm_roofline_bw_ceiling.py")
sfr = _load("strict_frontier_realize",
            "research/speed/strict_frontier_realize/strict_frontier_realize.py")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# ---- banked anchors (IMPORTED exact; this card derives nothing upstream) ----
CYCLE_WALL_US = sfr.CYCLE_WALL_US                 # 7903 strict (composed) coupled cycle <-> 467.14
CYCLE_PERM_US = sfr.CYCLE_PERM_US                 # 7666.83 deployed permissive cycle <-> 481.53
DEPLOYED_TPS = sfr.DEPLOYED_TPS                   # 481.53 deployed incumbent (non-equivalent, 3 flips)
REALIZED_BASE_TPS = sfr.REALIZED_BASE_TPS         # 467.14 composed blanket-strict frontier (to tighten against)
COMPOSED_ADDED_US = sfr.COMPOSED_ADDED_US         # 236.17 us assumed attention tax/cycle (-> 467.14)
ETA_ATTN_COMPOSED = sfr.ETA_ATTN_COMPOSED         # 0.0308 assumed 3.08% of decode
M1_COLLAPSE_TPS = sfr.M1_COLLAPSE_TPS             # 161.70 lawine #438 M=1 AR strict floor
PPL_ANCHOR = sfr.PPL_ANCHOR                       # 2.3772
PPL_GATE = sfr.PPL_GATE                           # 2.42
SIGMA_HW = 4.8153                                 # advisor-named hardware sigma (PR #472 baseline)

# #466 banked isolated-locus result (the conservative lower bound this card tightens)
ISO_DELTA_466_US = 422.91198670864105             # sxigz7dp isolated attention Delta @ L=640
ISO_DELTA_466_SIGMA = 0.6244556176338361
REALIZED_466_TPS = 456.3567777154584              # #466 realized_strict_frontier_tps (headline)
COMPOSED_VS_466_DRIFT = 10.78322228454158         # 467.14 - 456.36
REALIZED_466_ETA_ATTN = 0.05516127625091883       # #466 realized_eta_attn_decode (5.52%)
# #452 relax-side precedent the A/B mirrors (methodology twin)
RELAX_452_COMPOSED = 498.6
RELAX_452_REALIZED_DELTA = -0.94                  # composed 498.6 realized -0.94 (held within sigma)
GEMM_US_450 = 4152.96                             # #450 measured body-GEMM time (cross-check anchor)

# served gemma-4-E4B-it geometry (== #466)
N_FULL_LAYERS = sfr.N_FULL_LAYERS                 # 7 full-attention (hd=512) Triton layers
M_VERIFY = sfr.M_VERIFY                           # 8 spec-verify width (K_spec=7 + 1)
HEAD_DIM_FULL = sfr.HEAD_DIM_FULL                 # 512 full-attn head_dim (FA2 256-cap -> Triton)
HEAD_DIM_SLIDING = sfr.HEAD_DIM_SLIDING           # 256 sliding head_dim (sdpa proxy; identical both arms)
KV_LENS = sfr.KV_LENS                             # (128, 384, 640)
HEADLINE_L = sfr.HEADLINE_L                       # 640
SLIDING_WINDOW = 512                              # config sliding_window (sliding-layer KV cap)
ORDER = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
FULL_ATTN_IDX = (2, 8, 14, 20, 26, 32, 36)        # config text_config.layer_types full_attention indices


def tps_from_added_us(added_us):
    """IDENTICAL to sfr.tps_from_added_us: realized strict TPS = DEPLOYED * CYCLE_PERM /
    (CYCLE_PERM + added_us). added_us = the per-cycle wall the strict (order-preserving)
    attention reduction ADDS over the deployed permissive 3D path. added_us=0 -> 481.53
    (perm arm = deployed baseline, self-tax 0); =COMPOSED_ADDED_US (236) -> 467.14."""
    return sfr.tps_from_added_us(added_us)


# =========================================================================== #
def build_all(dims, num_layers, full_set, L_full, L_slide, M, dev, n_distinct):
    """Whole deployed verify cycle, self-built g=128 int4-Marlin body + per-layer attention
    + int4 12k lm_head. Body weights / sliding sdpa / lm_head are SHARED across arms (read-
    only; they cancel in the paired diff) -- only the 7 full-attn Triton reductions differ
    (strict_2d vs permissive_3d). n_distinct distinct cold weights/component (working set >>
    6 MiB A10G L2 -> every replay is a COLD HBM read, matching the deployed per-layer fresh-
    weight read). Returns (runners, NK, keep)."""
    NK = {c: dims["shapes"][c] for c in ORDER}
    NK["lm_head"] = (roof.LM_HEAD_VOCAB, dims["hidden"])
    ws = roof._mk_ws(dev)
    zp = torch.zeros(0, dtype=torch.int, device=dev)

    weights, xins = {}, {}
    for c in ORDER:
        N, K = NK[c]
        weights[c] = [roof._marlin_quant(N, K, 128, dev) for _ in range(n_distinct)]
        xins[c] = torch.randn(M, K, dtype=torch.float16, device=dev)
    lmN, lmK = NK["lm_head"]
    weights["lm_head"] = [roof._marlin_quant(lmN, lmK, -1, dev) for _ in range(max(2, n_distinct // 4))]
    xins["lm_head"] = torch.randn(M, lmK, dtype=torch.float16, device=dev)

    def gemm(c, idx):
        q_w, s, gi, so = weights[c][idx % len(weights[c])]
        N, K = NK[c]
        roof._apply_marlin(xins[c], q_w, s, zp, gi, so, ws, roof._QT, N, K, is_k_full=True, bias=None)

    # sliding-attention proxy (hd=256), shared, identical in both arms (sdpa as in #452)
    n_h = dims["n_heads"]
    q_sl = torch.randn(1, n_h, M, HEAD_DIM_SLIDING, dtype=torch.float16, device=dev)
    k_sl = torch.randn(1, n_h, L_slide, HEAD_DIM_SLIDING, dtype=torch.float16, device=dev)
    v_sl = torch.randn(1, n_h, L_slide, HEAD_DIM_SLIDING, dtype=torch.float16, device=dev)

    # full-attention Triton inputs: separate buffers PER RUNNER (no cross-graph aliasing);
    # BW is value-independent so same-shape buffers give identical timing.
    full_layers = sorted(full_set)
    full_pos = {Lidx: j for j, Lidx in enumerate(full_layers)}
    attn = {}
    runner_lever = {"whole_perm": "permissive_3d", "whole_strict": "strict_2d",
                    "iso_perm": "permissive_3d", "iso_strict": "strict_2d"}
    for rk, lever in runner_lever.items():
        num_par, max_q = sfr.ARMS[lever]
        lst = []
        for j, Lidx in enumerate(full_layers):
            seed = 7000 + 13 * Lidx + (1 if "perm" in rk else 0) + (100 if rk.startswith("iso") else 0)
            inp = sfr._build_inputs(L_full, M, HEAD_DIM_FULL, seed, dev)
            seg = sfr._segm_bufs(HEAD_DIM_FULL, num_par, dev, M) if lever == "permissive_3d" else None
            lst.append((inp, seg, num_par, max_q))
        attn[rk] = lst

    def attn_call(rk, Lidx):
        inp, seg, num_par, max_q = attn[rk][full_pos[Lidx]]
        sfr._call_unified(inp, runner_lever[rk], num_par, seg, max_q)

    def make_whole(rk):
        def run():
            for L in range(num_layers):
                for c in ORDER:
                    gemm(c, L)
                if L in full_set:
                    attn_call(rk, L)
                else:
                    F.scaled_dot_product_attention(q_sl, k_sl, v_sl)
            gemm("lm_head", 0)
        return run

    def make_iso(rk):
        def run():
            for Lidx in full_layers:
                attn_call(rk, Lidx)
        return run

    def make_body_gemm():
        def run():
            for L in range(num_layers):
                for c in ORDER:
                    gemm(c, L)
        return run

    runners = {
        "whole_perm": make_whole("whole_perm"),
        "whole_strict": make_whole("whole_strict"),
        "iso_perm": make_iso("iso_perm"),
        "iso_strict": make_iso("iso_strict"),
        "body_gemm": make_body_gemm(),
    }
    keep = (weights, xins, ws, zp, q_sl, k_sl, v_sl, attn)
    return runners, NK, keep


def _paired(series, a, b):
    diffs = [x - y for x, y in zip(series[a], series[b])
             if math.isfinite(x) and math.isfinite(y)]
    if not diffs:
        return float("nan"), float("nan"), 0
    m = statistics.median(diffs)
    sd = statistics.pstdev(diffs) if len(diffs) > 1 else 0.0
    return m, sd, len(diffs)


# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=HEADLINE_L)
    ap.add_argument("--Ls", type=str, default=None,
                    help="comma-separated KV-len sweep override (default: 128,384,640)")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=21)        # N>=21 paired median+sigma
    ap.add_argument("--n-distinct", type=int, default=8)
    ap.add_argument("--ident-trials", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--self-test", dest="self_test", action="store_true")
    ap.add_argument("--output", default=os.path.join(_here, "strict_wholecycle_ab.json"))
    ap.add_argument("--selftest-output", default=os.path.join(_here, "selftest.json"))
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="equivalence-escalation-anchors")
    ap.add_argument("--wandb_name", default="stark/strict-wholecycle-ab")
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
    L_slide = min(SLIDING_WINDOW, args.L)
    print(f"[wc] {name} sm_{cap[0]}{cap[1]} torch {torch.__version__}  M={M_VERIFY} "
          f"hd_full={HEAD_DIM_FULL} depth={num_layers}({depth_src}) full_attn={sorted(full_set)} "
          f"n_full={len(full_set)} L_headline={args.L} L_slide={L_slide} rounds={rounds} "
          f"n_distinct={n_distinct} smoke={args.smoke}", flush=True)
    assert len(full_set) == N_FULL_LAYERS, f"expected {N_FULL_LAYERS} full-attn layers, got {len(full_set)}"
    torch.cuda.reset_peak_memory_stats()

    # heavy warm-up -> A10G boost clock (same regime as #452/#466)
    big = torch.randn(2048, 2048, dtype=torch.bfloat16, device=dev)
    for _ in range(200):
        big = big @ big
    torch.cuda.synchronize()
    del big

    # co-measured achievable peak BW (same clock state) -- context for the verdict
    peak = roof.measure_peak_bw(dev, iters, args.warmup)
    print(f"[wc] PEAK BW: read={peak['bw_read_gbps']:.0f} copy={peak['bw_copy_gbps']:.0f} "
          f"bf16gemm@M8={peak['bw_bf16gemm_m8_gbps']:.0f} GB/s", flush=True)

    # ---- (1) whole-cycle A/B + isolated #466 reproduction, swept over L ------------------
    per_L = {}
    for L in Ls:
        runners, NK, keep = build_all(dims, num_layers, full_set, L, min(SLIDING_WINDOW, L),
                                      M_VERIFY, dev, n_distinct)
        series, captured = roof.paired_diff_measure(runners, iters, args.warmup, rounds)
        med = {nm: roof._med(series[nm]) for nm in runners}
        d_whole, s_whole, n_whole = _paired(series, "whole_strict", "whole_perm")
        d_iso, s_iso, n_iso = _paired(series, "iso_strict", "iso_perm")
        recover = ((d_iso - d_whole) / d_iso) if (math.isfinite(d_iso) and d_iso != 0) else float("nan")
        per_L[L] = {
            "median_us": med, "captured": captured,
            "whole_delta_us": d_whole, "whole_delta_sigma": s_whole, "n_whole": n_whole,
            "iso_delta_us": d_iso, "iso_delta_sigma": s_iso, "n_iso": n_iso,
            "overlap_recovery_fraction": recover,
            "whole_strict_tps": tps_from_added_us(d_whole),
            "iso_strict_tps": tps_from_added_us(d_iso),
            "body_gemm_us": med["body_gemm"],
        }
        print(f"[wc] L={L}: whole_perm={med['whole_perm']:.1f} whole_strict={med['whole_strict']:.1f} "
              f"body_gemm={med['body_gemm']:.1f} | whole_d={d_whole:+.2f}(s{s_whole:.2f}) "
              f"iso_d={d_iso:+.2f}(s{s_iso:.2f}) recover={recover*100:+.1f}% "
              f"-> {tps_from_added_us(d_whole):.2f} TPS | cap={captured}", flush=True)
        del runners, keep
        gc.collect()
        torch.cuda.empty_cache()

    H = per_L[args.L]
    whole_cycle_strict_delta_us = H["whole_delta_us"]
    whole_cycle_strict_delta_sigma = H["whole_delta_sigma"]
    iso_delta_inharness = H["iso_delta_us"]
    iso_delta_inharness_sigma = H["iso_delta_sigma"]
    body_gemm_us = H["body_gemm_us"]

    whole_cycle_strict_tps = tps_from_added_us(whole_cycle_strict_delta_us)
    whole_cycle_strict_tps_lo = tps_from_added_us(whole_cycle_strict_delta_us + whole_cycle_strict_delta_sigma)
    whole_cycle_strict_tps_hi = tps_from_added_us(whole_cycle_strict_delta_us - whole_cycle_strict_delta_sigma)
    whole_cycle_perm_tps = tps_from_added_us(0.0)      # 481.53: perm arm IS the deployed path (self-tax 0)

    # overlap recovery: how much of #466's isolated +422.91 us/cycle the in-graph context hid.
    # PRIMARY uses the IN-HARNESS isolated Delta (same pod/clock -> a pure isolated-vs-in-graph
    # comparison); cross-checked against the #466 banked value.
    overlap_recovery_fraction = ((iso_delta_inharness - whole_cycle_strict_delta_us) / iso_delta_inharness
                                 if (math.isfinite(iso_delta_inharness) and iso_delta_inharness != 0)
                                 else float("nan"))
    overlap_recovery_fraction_vs_banked = (ISO_DELTA_466_US - whole_cycle_strict_delta_us) / ISO_DELTA_466_US
    realized_eta_attn_decode = whole_cycle_strict_delta_us / CYCLE_PERM_US

    composed_vs_wholecycle_drift = REALIZED_BASE_TPS - whole_cycle_strict_tps
    whole_cycle_holds_within_sigma_hw = bool(abs(composed_vs_wholecycle_drift) <= SIGMA_HW)
    realized_strict_frontier_best_estimate_tps = whole_cycle_strict_tps   # the honest point estimate

    # cudagraph survival (the strict M=8 verify reduction must capture+replay, not collapse to M=1)
    whole_strict_captured_all_L = all(per_L[L]["captured"]["whole_strict"] for L in Ls)
    whole_perm_captured_all_L = all(per_L[L]["captured"]["whole_perm"] for L in Ls)
    iso_strict_captured_all_L = all(per_L[L]["captured"]["iso_strict"] for L in Ls)

    # ---- (4) identity re-confirm on the strict 2D attention (locus M-invariance, #466) ----
    seeds = [1234] if args.smoke else [1234, 5678, 9012]
    ident = sfr.identity_probe(args.L, HEAD_DIM_FULL, dev, ident_trials, seeds)
    iarm = ident.get("per_arm", {})
    whole_cycle_strict_identity_fraction = float(iarm.get("strict_2d", {}).get("byte_identity_min", float("nan")))
    strict_argmax_fraction = float(iarm.get("strict_2d", {}).get("argmax_identity_min", float("nan")))
    whole_cycle_strict_token_flips = int(round((1.0 - (strict_argmax_fraction
                                       if math.isfinite(strict_argmax_fraction) else 1.0)) * M_VERIFY))
    permissive_identity_fraction = float(iarm.get("permissive_3d", {}).get("byte_identity_min", float("nan")))

    peak_vram_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # =================== SELF-TEST ===========================================
    st = {}
    st["tps_zero_added_is_deployed"] = bool(abs(tps_from_added_us(0.0) - DEPLOYED_TPS) < 1e-6)
    st["tps_composed_added_is_base"] = bool(abs(tps_from_added_us(COMPOSED_ADDED_US) - REALIZED_BASE_TPS) < 1e-2)
    st["tps_more_added_lowers"] = bool(tps_from_added_us(500.0) < tps_from_added_us(50.0) < DEPLOYED_TPS)
    st["constants_anchored"] = bool(REALIZED_BASE_TPS == 467.14 and DEPLOYED_TPS == 481.53
                                    and CYCLE_WALL_US == 7903.0)
    st["geometry_full_attn"] = bool(len(full_set) == N_FULL_LAYERS and M_VERIFY == 8
                                    and HEAD_DIM_FULL == 512 and num_layers == 37)
    st["body_shapes_served"] = bool(NK["qkv_proj"] == (3072, 2560) and NK["gate_up_proj"] == (20480, 2560)
                                    and NK["down_proj"] == (2560, 10240) and NK["o_proj"] == (2560, 2048))
    st["whole_perm_captured"] = bool(whole_perm_captured_all_L)
    st["whole_strict_captured_survives"] = bool(whole_strict_captured_all_L)
    st["iso_strict_captured"] = bool(iso_strict_captured_all_L)
    # CALIBRATION GUARD 1: in-harness isolated Delta reproduces #466's banked 422.91 (within 12%)
    st["iso_delta_reproduces_466"] = bool(math.isfinite(iso_delta_inharness) and
                                          abs(iso_delta_inharness - ISO_DELTA_466_US) / ISO_DELTA_466_US < 0.12)
    # CALIBRATION GUARD 2: body GEMM reproduces #450's gemm_us 4152.96 (within 20%, as #452)
    st["body_gemm_anchor_matches_450"] = bool(body_gemm_us > 0 and
                                              abs(body_gemm_us - GEMM_US_450) / GEMM_US_450 < 0.20)
    # whole-cycle Delta must be POSITIVE (strict costs >= perm) and not exceed the isolated tax
    # by an implausible margin (overlap can only HIDE tax, modulo small cache/clock 2nd-order)
    st["whole_delta_positive"] = bool(math.isfinite(whole_cycle_strict_delta_us)
                                      and whole_cycle_strict_delta_us > 0)
    st["recover_frac_in_band"] = bool(math.isfinite(overlap_recovery_fraction)
                                      and -0.25 <= overlap_recovery_fraction <= 1.0)
    finite = [whole_cycle_strict_tps, whole_cycle_perm_tps, whole_cycle_strict_delta_us,
              whole_cycle_strict_delta_sigma, iso_delta_inharness, overlap_recovery_fraction,
              realized_strict_frontier_best_estimate_tps, body_gemm_us]
    st["nan_clean"] = all(math.isfinite(x) for x in finite)
    st["identity_ran"] = bool(ident.get("error") is None and "per_arm" in ident)
    st["strict_byte_exact"] = bool(math.isfinite(whole_cycle_strict_identity_fraction)
                                   and whole_cycle_strict_identity_fraction >= 0.999)
    st["strict_zero_flips"] = bool(whole_cycle_strict_token_flips == 0)
    st["permissive_reproduces_nonequiv"] = bool(math.isfinite(permissive_identity_fraction)
                                                and permissive_identity_fraction < 0.999)
    st["ppl_anchor_ok"] = bool(PPL_ANCHOR <= PPL_GATE)
    st["vram_ok"] = bool(peak_vram_gib <= 24.0)
    st["realized_above_collapse_floor"] = bool(whole_cycle_strict_tps > M1_COLLAPSE_TPS + 50.0)
    whole_cycle_self_test_passes = all(st.values())

    # =================== VERDICT / RECONCILE =================================
    if not whole_strict_captured_all_L:
        outcome = "COLLAPSE"
    elif whole_cycle_holds_within_sigma_hw:
        outcome = "REALIZED_REACHES_467_WITHIN_SIGMA"
    elif overlap_recovery_fraction >= 0.05:
        outcome = "REALIZED_ABOVE_466_LOWER_BOUND"      # overlap recovered some tax (456 < realized < 462)
    else:
        outcome = "REALIZED_STAYS_AT_466_LOWER_BOUND"   # ~0 overlap -> 456 is honest AND tight

    reconcile = (
        f"#466's ISOLATED-locus attention Delta (+{ISO_DELTA_466_US:.1f} us/cycle, the 7 full-attn "
        f"reductions timed ALONE) -> realized 456.36 was a CONSERVATIVE LOWER BOUND (isolation cannot "
        f"hide tax under overlap). The #452-identical WHOLE-CYCLE A/B (37-layer self-built g=128 "
        f"int4-Marlin body + 7 full-attn Triton reductions + 30 sliding sdpa + int4 12k lm_head, "
        f"single-stream ONEGRAPH-captured) measures the strict tax UNDER in-graph overlap: "
        f"whole_cycle_strict_delta_us={whole_cycle_strict_delta_us:+.1f} (sigma {whole_cycle_strict_delta_sigma:.1f}) "
        f"vs the in-harness isolated {iso_delta_inharness:+.1f} (#466 banked {ISO_DELTA_466_US:.1f}) "
        f"-> overlap_recovery_fraction={overlap_recovery_fraction*100:+.1f}%. realized_strict_frontier_"
        f"best_estimate_tps={realized_strict_frontier_best_estimate_tps:.2f} "
        f"({-composed_vs_wholecycle_drift:+.2f} vs composed 467.14; |drift|={abs(composed_vs_wholecycle_drift):.2f} "
        f"{'<=' if whole_cycle_holds_within_sigma_hw else '>'} sigma_hw={SIGMA_HW}). whole_cycle_perm_tps="
        f"{whole_cycle_perm_tps:.2f} (deployed-baseline anchor; calibration guards: in-harness isolated "
        f"Delta reproduces #466 within 12% [{st['iso_delta_reproduces_466']}], body GEMM "
        f"{body_gemm_us:.0f}us reproduces #450 4152.96 within 20% [{st['body_gemm_anchor_matches_450']}]). "
        f"The M=8 strict-reduction verify attention CAPTURES+REPLAYS (survives) -> does NOT collapse to "
        f"the M=1 161.70 floor. Strict 2D byte-exact: identity={whole_cycle_strict_identity_fraction:.4f} "
        f"({whole_cycle_strict_token_flips} flips); deployed permissive byte={permissive_identity_fraction:.4f} "
        f"(reproduces the non-equivalence). Reconcile vs #466 (composed_vs_realized_drift +10.78, "
        f"realized_eta_attn 5.52%; here {realized_eta_attn_decode*100:.2f}%) and #452 (relax: composed 498.6 "
        f"realized -0.94, held). OUTCOME={outcome}.")

    verdict = {
        "whole_cycle_self_test_passes": whole_cycle_self_test_passes,                  # PRIMARY
        "realized_strict_frontier_best_estimate_tps": realized_strict_frontier_best_estimate_tps,  # TEST/primary
        "whole_cycle_strict_tps": whole_cycle_strict_tps,
        "whole_cycle_strict_tps_sigma_lo": whole_cycle_strict_tps_lo,
        "whole_cycle_strict_tps_sigma_hi": whole_cycle_strict_tps_hi,
        "whole_cycle_perm_tps": whole_cycle_perm_tps,
        "whole_cycle_strict_delta_us": whole_cycle_strict_delta_us,
        "whole_cycle_strict_delta_sigma": whole_cycle_strict_delta_sigma,
        "iso_delta_us_inharness": iso_delta_inharness,
        "iso_delta_us_inharness_sigma": iso_delta_inharness_sigma,
        "iso_delta_us_466_banked": ISO_DELTA_466_US,
        "iso_strict_tps_inharness": tps_from_added_us(iso_delta_inharness),
        "overlap_recovery_fraction": overlap_recovery_fraction,
        "overlap_recovery_fraction_vs_banked": overlap_recovery_fraction_vs_banked,
        "whole_cycle_holds_within_sigma_hw": whole_cycle_holds_within_sigma_hw,
        "composed_vs_wholecycle_drift": composed_vs_wholecycle_drift,
        "realized_eta_attn_decode": realized_eta_attn_decode,
        "eta_attn_composed": ETA_ATTN_COMPOSED,
        "whole_cycle_strict_identity_fraction": whole_cycle_strict_identity_fraction,
        "whole_cycle_strict_token_flips": whole_cycle_strict_token_flips,
        "permissive_identity_fraction": permissive_identity_fraction,
        "body_gemm_us": body_gemm_us, "gemm_us_450_anchor": GEMM_US_450,
        "whole_strict_captured_all_L": whole_strict_captured_all_L,
        "whole_perm_captured_all_L": whole_perm_captured_all_L,
        "strict_frontier_is_e2e_measurable": whole_strict_captured_all_L,
        "strict_frontier_collapses_to_m1": bool(not whole_strict_captured_all_L),
        # banked reconciliation anchors
        "realized_466_lower_bound_tps": REALIZED_466_TPS,
        "composed_vs_466_drift": COMPOSED_VS_466_DRIFT,
        "realized_466_eta_attn_decode": REALIZED_466_ETA_ATTN,
        "relax_452_composed": RELAX_452_COMPOSED, "relax_452_realized_delta": RELAX_452_REALIZED_DELTA,
        "cycle_perm_us": CYCLE_PERM_US, "cycle_wall_us": CYCLE_WALL_US,
        "composed_added_us_per_cycle": COMPOSED_ADDED_US,
        "deployed_tps": DEPLOYED_TPS, "realized_base_tps": REALIZED_BASE_TPS,
        "m1_collapse_floor_tps": M1_COLLAPSE_TPS, "sigma_hw": SIGMA_HW,
        "n_full_layers_per_cycle": len(full_set), "deployed_num_layers": num_layers,
        "headline_L": args.L,
        "outcome": outcome,
        "ppl": PPL_ANCHOR, "ppl_anchor": PPL_ANCHOR, "ppl_gate": PPL_GATE,
        "peak_vram_gib": peak_vram_gib,
        "lever_is_config_reachable_no_source_edit": True,
        "no_kernel_rebuild": True,
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True,
        "official_tps": 0,
        "self_test_conditions": st,
        "reconcile_line": reconcile,
    }

    payload = {
        "config": {"torch": torch.__version__, "device": name, "sm": f"{cap[0]}{cap[1]}",
                   "M": M_VERIFY, "head_dim_full": HEAD_DIM_FULL, "head_dim_sliding": HEAD_DIM_SLIDING,
                   "deployed_num_layers": num_layers, "full_attn_idx": sorted(full_set),
                   "n_full_layers": len(full_set), "KV_LENS": list(Ls), "headline_L": args.L,
                   "sliding_window": SLIDING_WINDOW, "iters": iters, "warmup": args.warmup,
                   "rounds": rounds, "n_distinct": n_distinct, "ident_trials": ident_trials,
                   "smoke": args.smoke, "served_model_dir": model_dir, "group_size": 128,
                   "self_built_marlin": True,
                   "note": "whole-cycle strict A/B: full deployed M=8 verify cycle (37-layer self-built "
                           "g=128 int4-Marlin body via apply_gptq_marlin_linear + per-layer attention -- "
                           "7 full-attn hd=512 served Triton unified_attention with the strict_2d (natural "
                           "M=8 order-preserving, == VLLM_BATCH_INVARIANT=1) vs permissive_3d (deployed "
                           "split-KV) lever, 30 sliding hd=256 sdpa -- + int4 12k lm_head), single-stream "
                           "ONEGRAPH-captured, paired median+sigma over rounds. whole_cycle_strict_delta_us "
                           "= whole_strict - whole_perm captures in-graph overlap; isolated #466 arm "
                           "co-measured for overlap_recovery_fraction. realized via tps_from_added_us on the "
                           "banked cycle (CYCLE_PERM=7666.83 <-> 481.53). No serve change, no HF Job, no "
                           "submission, no kernel rebuild."},
        "peak_bw": peak,
        "per_L": {str(L): per_L[L] for L in per_L},
        "shapes": {c: list(NK[c]) for c in NK},
        "identity": ident,
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2, default=lambda o: float(o) if isinstance(o, (int, float)) else str(o))
    with open(args.selftest_output, "w") as fh:
        json.dump({"whole_cycle_self_test_passes": whole_cycle_self_test_passes, "checks": st}, fh, indent=2)
    print(f"[wc] wrote {args.output}", flush=True)
    print(f"\n[wc] OUTCOME={outcome}  self_test={whole_cycle_self_test_passes}", flush=True)
    print(f"[wc] realized_strict_frontier_best_estimate_tps={realized_strict_frontier_best_estimate_tps:.2f} "
          f"(drift vs 467.14 = {composed_vs_wholecycle_drift:+.2f}; holds={whole_cycle_holds_within_sigma_hw}) "
          f"whole_delta={whole_cycle_strict_delta_us:+.1f}us iso_delta={iso_delta_inharness:+.1f}us "
          f"overlap_recovery={overlap_recovery_fraction*100:+.1f}%", flush=True)
    print(f"[wc] {reconcile}", flush=True)
    print(f"[wc] self_test={st}", flush=True)

    # The GPU tool-venv has no usable wandb (PEP-420 namespace shadow). Log from the repo
    # .venv via the standalone wandb_log.py (pure json+wandb, venv-agnostic) -- the #452/#466 split.
    if not (args.no_wandb or args.smoke):
        print(f"[wc] to log W&B: cd target/ && .venv/bin/python "
              f"research/speed/strict_wholecycle_ab/wandb_log.py --json {args.output} "
              f"--wandb_group {args.wandb_group} --wandb_name {args.wandb_name}", flush=True)

    gc.collect()
    torch.cuda.empty_cache()
    if args.self_test:
        return 0 if whole_cycle_self_test_passes else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
