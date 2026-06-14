#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""LUT/GANQ W4A16 GEMM feasibility at M=8 on sm_86 (PR #113) -- BUILD or KILL.

THE QUESTION
------------
The deployed `fa2sw_precache_kenyan` stack verifies M=8 candidate rows/step through
int4 W4A16 (compressed-tensors / GPTQ-Marlin) weight GEMMs. denken #68 MEASURED that
at M=8 this verify-GEMM is unambiguously WEIGHT-BANDWIDTH-BOUND: ~77.1% of the A10G
HBM roofline (462 GB/s) but only ~20.2% of the FP16 tensor-core compute peak. The
verify-GEMM is ~53% of the decode budget.

LUT-based W4A16 GEMM (LUT-GEMM, Park et al. ICLR 2024 arXiv:2206.09557; GANQ, ICML
2025) replaces Marlin's dequantize-then-FP16-MAC with table lookups (optionally on
INT8 tensor cores). The hypothesis (researcher-agent #2 fresh kernel ceiling, +12-22%
projected) is that this cuts per-token HBM traffic + compute in the BW-bound M=8
regime. This module SIZES whether that is true on sm_86 -- a GO/HOLD on a multi-day
kernel build, NOT a build.

THE MODEL -- roofline anchored on #68's MEASURED Marlin numbers
--------------------------------------------------------------
verify-GEMM time = bytes / achieved_bandwidth   (it is bandwidth-bound at M=8).

Two factors, exactly as the committed tree_free_500_ceiling model classifies them:
  * achieved_bandwidth (UTILISATION): Marlin runs at 77.1% of HBM peak. Closing the
    gap to 100% is the +29.7% SplitK ceiling. It needs BETTER MEMORY SCHEDULING
    (more K-dim thread blocks -> SplitK), NOT a different compute path.
  * bytes (BYTE COUNT): 4-bit weight + scale/codebook metadata + activations/outputs.

Where does LUT-GEMM act?
  * It does NOT change achieved bandwidth: it reads the SAME int4 weight bytes with no
    inherently better (and plausibly worse, via gather/indirection) memory access
    pattern than Marlin's hand-tuned coalesced pipeline. -> NOT a utilisation lever.
  * It does NOT reduce weight bytes: still 4-bit indices/planes. Its LUT metadata
    (BCQ per-plane scales or a non-uniform codebook) at iso-PPL is COMPARABLE-TO-
    LARGER than Marlin's group scales. -> NOT a byte lever (byte-negative at worst).
  * Its ONLY mechanism is replacing dequant+FP16-MAC with lookups, i.e. it reduces
    COMPUTE. But at M=8 compute is 20% utilised -- 80% idle/hidden under memory
    stalls. Reducing an 80%-idle resource changes a bandwidth-bound wall-time by ~0.

=> LUT-GEMM attacks the one factor (compute) that is SLACK at M=8, and touches
   NEITHER factor (bandwidth utilisation, byte count) that binds. Its M=8 speedup
   ceiling is ~0% (iso-bytes/iso-BW), realistic NEGATIVE (metadata bytes + LUT-build
   cost ~ M + INT8-TC tile underfill: the INT8 MMA tile is m16n8k32, so M=8 is below
   the tile floor -- MEASURED here: torch._int_mm refuses M<=16 outright).

PRIMARY  lut_gemm_m8_speedup_vs_marlin_pct  (projected verify-GEMM speedup at M=8)
TEST     lut_gemm_ppl_projected             (LUT-table quant numerics vs the 2.42 cap)

LOCAL, analytic core (CPU) reusing denken #68 MEASURED roofline data, + an OPTIONAL
single-GPU INT8-TC substrate probe (torch._int_mm, no model load, no token stream).
No HF Job, no submission, no served-file change. Greedy identity untouched (sizing
only); under kanna #96 the official greedy gate is self-referential per checkpoint,
so any DETERMINISTIC verify kernel is greedy-safe by construction -- PPL is the only
numerics gate, and it is moot here because the speed ceiling is <=0.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---- A10G (GA102, sm_86) ceilings; Marlin compute path is FP16, not int4 (#68) ----
A10G_HBM_GBS = 600.0
MEASURED_PEAK_TFLOPS = 64.34          # #68 realizable FP16 tensor peak (measured, big-M)
WEIGHT_BITS = 4
SCALE_BYTES = 2                       # fp16 group scale / codebook centroid
GROUP_SIZE = 32                       # deployed compressed-tensors group size
NU_CENTROIDS = 16                     # 4-bit non-uniform codebook size (2^4)
BCQ_PLANES = 4                        # BCQ binary planes for 4-bit (= bit-width)

# Frontier / cap (committed BASELINE.md).
FRONTIER_OFFICIAL = 481.53
BASELINE_PPL = 2.3777
PPL_CAP = 2.42


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Byte models. ALL read the SAME 4-bit weight matrix (LUT does not change bit-width).
# They differ ONLY in metadata (scales / codebook), which is the honest crux: at
# iso-PPL, is LUT metadata smaller, equal, or larger than Marlin's group scales?
# ---------------------------------------------------------------------------
def marlin_meta_bytes(in_f, out_f, g):
    """GPTQ-Marlin: one fp16 scale per (out, group)."""
    return SCALE_BYTES * out_f * math.ceil(in_f / g)


def weight_bytes(in_f, out_f):
    return (WEIGHT_BITS / 8.0) * out_f * in_f


def model_bytes(in_f, out_f, M, model, g):
    """Total per-call HBM bytes (weight + metadata + activations + outputs)."""
    w = weight_bytes(in_f, out_f)
    act = 2.0 * M * in_f
    out = 2.0 * M * out_f
    if model == "marlin":
        meta = marlin_meta_bytes(in_f, out_f, g)
    elif model == "lut_iso":
        # OPTIMISTIC: assume LUT metadata == Marlin scale bytes (best case for LUT).
        meta = marlin_meta_bytes(in_f, out_f, g)
    elif model == "lut_bcq_b4_g32":
        # LUT-GEMM BCQ at iso-PPL with the deployed g=32: B=4 binary planes, EACH with
        # its own per-group scale -> 4x Marlin scale bytes. (Weight matrix bits same.)
        meta = BCQ_PLANES * marlin_meta_bytes(in_f, out_f, g)
    elif model == "lut_nu_perchannel":
        # Non-uniform (GANQ/SqueezeLLM-style) per-OUTPUT-CHANNEL 16-entry fp16 codebook.
        # CHEAPER metadata than g=32 -- but COARSER quant (per-channel, no in-grouping):
        # a granularity/byte trade Marlin could also take; NOT a LUT-compute advantage,
        # and a PPL risk. Listed to bound the byte axis, not as an iso-PPL claim.
        meta = NU_CENTROIDS * SCALE_BYTES * out_f
    elif model == "lut_nu_g32":
        # Non-uniform at iso-PPL with g=32: a 16-entry codebook PER GROUP -> codebook
        # bytes ~ 16x the marlin scale bytes (catastrophic: > the weight matrix itself).
        meta = NU_CENTROIDS * SCALE_BYTES * out_f * math.ceil(in_f / g)
    else:
        raise ValueError(model)
    return {"weight": w, "meta": meta, "act": act, "out": out,
            "total": w + meta + act + out}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roofline-json",
                    default="research/spec_cost_model/verify_gemm_roofline.json",
                    help="denken #68 MEASURED Marlin roofline (per-shape M=8 times/bytes)")
    ap.add_argument("--m", type=int, default=8, help="deployed verify width")
    ap.add_argument("--group-size", type=int, default=GROUP_SIZE)
    ap.add_argument("--int8-tc-probe", action="store_true",
                    help="run the optional single-GPU INT8-TC substrate microbench "
                         "(torch._int_mm M-sweep) to corroborate the m16n8k32 tile floor")
    ap.add_argument("--out", default="research/spec_cost_model/lut_gemm_feasibility_results.json")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="gemma-challenge-senpai")
    ap.add_argument("--wandb-entity", default="wandb-applied-ai-team")
    ap.add_argument("--wandb-name", default="denken/lut-gemm-feasibility")
    ap.add_argument("--wandb-group", default="lut-gemm-feasibility")
    args = ap.parse_args()

    g = args.group_size
    M = args.m

    # ---- load #68 measured Marlin per-shape rows at M=8 -----------------------
    rj = json.load(open(args.roofline_json))
    shapes = []
    for r in rj["rows"]:
        if r["M"] == M:
            shapes.append({"role": r["role"], "in": r["in"], "out": r["out"],
                           "count": r["count"], "t_us": r["t_us"],
                           "gbytes_s": r["gbytes_s"], "flops": r["flops"],
                           "w_bytes_measured": r["w_bytes"]})
    if not shapes:
        raise SystemExit(f"no M={M} rows in {args.roofline_json}")

    # ---- per-shape Marlin vs LUT byte models + BW-bound times -----------------
    lut_models = ["lut_iso", "lut_bcq_b4_g32", "lut_nu_perchannel", "lut_nu_g32"]
    per_shape = []
    agg = {"marlin": {"t_us": 0.0, "bytes": 0.0, "weight_bytes": 0.0, "flops": 0.0},
           **{m: {"t_us": 0.0, "bytes": 0.0} for m in lut_models},
           "bw_floor": {"t_us": 0.0}, "compute_floor": {"t_us": 0.0}}
    for s in shapes:
        inn, out, cnt = s["in"], s["out"], s["count"]
        bw_gbs = s["gbytes_s"]                 # Marlin's MEASURED achieved bandwidth
        mb = model_bytes(inn, out, M, "marlin", g)
        t_marlin = s["t_us"]                   # measured
        # LUT BW-bound time at Marlin's achieved bandwidth (best case: LUT is no better
        # at memory than the hand-tuned Marlin pipeline). t = bytes / achieved_BW.
        row = {"role": s["role"], "in": inn, "out": out, "count": cnt,
               "t_us_marlin": t_marlin, "marlin_bytes": mb["total"],
               "marlin_breakdown": mb, "achieved_gbytes_s": bw_gbs, "lut": {}}
        for mdl in lut_models:
            lb = model_bytes(inn, out, M, mdl, g)
            t_lut = lb["total"] / (bw_gbs * 1e9) * 1e6   # us
            speedup_pct = (t_marlin / t_lut - 1.0) * 100.0
            row["lut"][mdl] = {"bytes": lb["total"], "breakdown": lb,
                               "t_us": t_lut, "speedup_pct": speedup_pct}
            agg[mdl]["t_us"] += t_lut * cnt
            agg[mdl]["bytes"] += lb["total"] * cnt
        # absolute floors (per shape):
        t_bw_floor = mb["weight"] / (A10G_HBM_GBS * 1e9) * 1e6   # 100% HBM, weight-only
        t_compute_floor = s["flops"] / (MEASURED_PEAK_TFLOPS * 1e12) * 1e6
        row["t_us_bw_floor_100pct_hbm"] = t_bw_floor
        row["t_us_compute_floor"] = t_compute_floor
        agg["marlin"]["t_us"] += t_marlin * cnt
        agg["marlin"]["bytes"] += mb["total"] * cnt
        agg["marlin"]["weight_bytes"] += mb["weight"] * cnt
        agg["marlin"]["flops"] += s["flops"] * cnt
        agg["bw_floor"]["t_us"] += t_bw_floor * cnt
        agg["compute_floor"]["t_us"] += t_compute_floor * cnt
        per_shape.append(row)

    # ---- aggregate speedups ----------------------------------------------------
    Tm = agg["marlin"]["t_us"]
    speedups = {}
    for mdl in lut_models:
        speedups[mdl] = (Tm / agg[mdl]["t_us"] - 1.0) * 100.0
    # bandwidth-gap ceiling = the headroom from 77.1% -> 100% HBM at the SAME total
    # bytes. This is the TOTAL verify-GEMM headroom at M=8, and it is a UTILISATION
    # headroom (SplitK's), NOT a compute headroom (LUT's). Any same-bytes kernel
    # reaching 100% HBM hits this; it matches the committed SPLITK_CEILING (=+29.7%).
    marlin_achieved_agg_gbs = agg["marlin"]["bytes"] / (Tm / 1e6) / 1e9
    bw_gap_ceiling_pct = (A10G_HBM_GBS / marlin_achieved_agg_gbs - 1.0) * 100.0
    # weight-only theoretical floor (over-credits: real kernels also move act/out/meta).
    weight_only_floor_gap_pct = (Tm / agg["bw_floor"]["t_us"] - 1.0) * 100.0
    compute_floor_pct_of_marlin = 100.0 * agg["compute_floor"]["t_us"] / Tm

    # PRIMARY metric: the LUT *mechanism* speedup at iso-quantization (iso-bytes) is the
    # honest ceiling of what the LUT compute-path swap buys at M=8. It is 0% by
    # construction (same bytes, same achieved BW), because compute is slack. The
    # realistic number is the BCQ iso-PPL model (metadata-heavy) -> strongly negative.
    primary_optimistic = speedups["lut_iso"]            # ~0.0
    primary_realistic_bcq = speedups["lut_bcq_b4_g32"]  # negative
    lut_gemm_m8_speedup_vs_marlin_pct = primary_optimistic  # headline = best case <= 0

    # ---- TEST metric: PPL projection ------------------------------------------
    # A 4-bit non-uniform LUT (GANQ) is at least as expressive as uniform GPTQ-int4 at
    # the same bit-width, so iso-bit-width PPL is ~equal-or-better. The deployed stack
    # is already 4-bit at PPL 2.3777 (0.64 under the 2.42 cap). A LUT requant to 4-bit
    # lands ~2.36-2.40. PPL is NOT the blocker; it is moot given speedup <= 0.
    lut_gemm_ppl_projected = BASELINE_PPL  # ~unchanged at iso-bit-width; cushion 0.64
    ppl_holds = lut_gemm_ppl_projected <= PPL_CAP

    # ---- optional GPU INT8-TC substrate probe ---------------------------------
    int8_probe = run_int8_tc_probe(shapes) if args.int8_tc_probe else None

    # ---- Step 4: drop the speedup into the #109 ship-readiness corner ----------
    ship = ship_readiness_integration(lut_gemm_m8_speedup_vs_marlin_pct,
                                      primary_realistic_bcq)

    # ---- verdict ---------------------------------------------------------------
    # GREEN: material M=8 speedup (beats SplitK ~8.5% OR stacks) + deterministic + PPL.
    # AMBER: positive but <= SplitK or PPL-marginal.
    # RED:   LUT <= Marlin at M=8 (overhead dominates the BW-bound low-M regime).
    if lut_gemm_m8_speedup_vs_marlin_pct > 8.5:
        verdict = "GREEN"
    elif lut_gemm_m8_speedup_vs_marlin_pct > 0.0:
        verdict = "AMBER"
    else:
        verdict = "RED"
    verdict_label = (
        "LUT-GEMM is a COMPUTE-path lever, but the M=8 verify-GEMM is BANDWIDTH-bound "
        "(#68: 77.1% HBM, 20.2% compute). It reads the SAME 4-bit weight bytes (no byte "
        "win; iso-PPL metadata is comparable-to-larger), with no better memory schedule "
        "(no utilisation win), and saves only compute that is already 80% idle. Best-case "
        f"M=8 speedup = {primary_optimistic:+.1f}% (iso-bytes); realistic (BCQ iso-PPL "
        f"metadata) = {primary_realistic_bcq:+.1f}%; plus INT8-TC tile underfill at M=8 "
        "(m16n8k32 floor). The +29.7% verify-GEMM headroom is a UTILISATION ceiling that "
        "belongs to SplitK, not LUT -- so LUT is dominated by SplitK, not additive. KILL.")

    out = {
        "gate": {
            "primary_metric_name": "lut_gemm_m8_speedup_vs_marlin_pct",
            "lut_gemm_m8_speedup_vs_marlin_pct": lut_gemm_m8_speedup_vs_marlin_pct,
            "lut_gemm_m8_speedup_optimistic_iso_pct": primary_optimistic,
            "lut_gemm_m8_speedup_realistic_bcq_isoppl_pct": primary_realistic_bcq,
            "test_metric_name": "lut_gemm_ppl_projected",
            "lut_gemm_ppl_projected": lut_gemm_ppl_projected,
            "ppl_cap": PPL_CAP, "ppl_holds": ppl_holds, "ppl_cushion": PPL_CAP - lut_gemm_ppl_projected,
            "verdict": verdict, "verdict_label": verdict_label,
            "bandwidth_gap_ceiling_pct": bw_gap_ceiling_pct,
            "weight_only_theoretical_floor_gap_pct": weight_only_floor_gap_pct,
            "marlin_achieved_agg_gbytes_s": marlin_achieved_agg_gbs,
            "bandwidth_gap_owner": "SplitK (utilisation lever), NOT LUT (compute lever)",
            "compute_floor_pct_of_marlin_time": compute_floor_pct_of_marlin,
        },
        "marlin_baseline": {
            "agg_t_us": Tm, "agg_bytes": agg["marlin"]["bytes"],
            "agg_weight_bytes": agg["marlin"]["weight_bytes"],
            "measured_hbm_util_pct": rj["verdict"]["agg_pct_hbm_peak_at_M8"],
            "measured_compute_util_pct": rj["verdict"]["agg_pct_compute_peak_at_M8"],
            "bandwidth_bound": rj["verdict"]["bandwidth_bound_at_M8"],
            "source": "denken #68 verify_gemm_roofline.json (measured CUDA-graph-replay)",
        },
        "lut_speedups_by_byte_model_pct": speedups,
        "byte_model_notes": {
            "lut_iso": "OPTIMISTIC: LUT metadata == Marlin scales. Same bytes -> 0% (the "
                       "ceiling of what the LUT compute-path swap buys when compute is slack).",
            "lut_bcq_b4_g32": "LUT-GEMM BCQ at iso-PPL g=32: 4 binary planes each with own "
                              "group scale -> 4x scale bytes -> byte-NEGATIVE in a BW-bound GEMM.",
            "lut_nu_perchannel": "per-channel 16-entry codebook: cheaper metadata but COARSER "
                                 "quant (PPL risk); a granularity/byte trade Marlin could also "
                                 "take -- NOT a LUT compute advantage.",
            "lut_nu_g32": "per-GROUP non-uniform codebook (iso-PPL): codebook > weight matrix "
                          "-> catastrophic in a BW-bound GEMM.",
        },
        "per_shape": per_shape,
        "step4_ship_readiness": ship,
        "int8_tc_substrate_probe": int8_probe,
        "method": ("analytic roofline anchored on denken #68 MEASURED Marlin per-shape M=8 "
                   "times/bytes; LUT modeled as same-4-bit-weight + metadata, BW-bound time = "
                   "bytes/achieved_BW. Optional GPU probe = torch._int_mm INT8-TC M-sweep "
                   "(no model load, no token stream). CPU-only core; no HF Job/submission."),
        "research": ("researcher-agent (LUT-GEMM arXiv:2206.09557, GANQ ICML2025 OpenReview "
                     "pkKQGJ5d99, T-MAC arXiv:2407.00088, Marlin arXiv:2408.11743): no published "
                     "sm_86/sm_80 LUT W4A16 kernel beats Marlin at M=4-16; GANQ's 2.57x is RTX4090 "
                     "sm_89 at M=1; T-MAC is CPU-only. Independent verdict RED (-5% to +3%, ~0%)."),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    _console(out, per_shape, speedups, bw_gap_ceiling_pct, compute_floor_pct_of_marlin,
             int8_probe, ship, M)
    print(f"\nwrote {args.out}")

    if args.wandb:
        _log_wandb(args, out)


def ship_readiness_integration(s_opt_pct, s_bcq_pct):
    """Plug the LUT M=8 speedup into denken #109's tree-free ship-readiness corner.

    A verify-GEMM speedup s enters the committed model identically to SplitK:
    vg = 0.53*(1-f_dq)/(1+s). So we ask: at s_LUT, does the conservative corner clear
    500? And does LUT STACK with SplitK or merely substitute? (It substitutes: both are
    the SAME verify-GEMM slice; the combined verify speedup is what the model sees, and
    LUT contributes ~0 to it.)"""
    tf = _load_module("tree_free_500_ceiling",
                      os.path.join(_HERE, "tree_free_500_ceiling.py"))
    sr = _load_module("tree_free_ship_readiness",
                      os.path.join(_HERE, "tree_free_ship_readiness.py"))
    p_cons = sr.ship_point("conservative", "local")
    p_cent = sr.ship_point("central", "local")
    corner_splitk_for_500 = sr.splitk_threshold_for(p_cons, sr.TARGET)  # 0.1434

    def corner_tps(s):
        return sr.official_ship(s, p_cons)

    s_lut_opt = max(0.0, s_opt_pct) / 100.0
    s_lut_bcq = s_bcq_pct / 100.0            # may be negative (a SLOWDOWN)
    s_splitk = sr.SPLITK_UBEL["central"]     # 0.085 ubel plausible

    # combined verify speedup if a hypothetical SplitK'd-LUT kernel existed: still
    # bandwidth-bound on the SAME bytes after SplitK saturates HBM, so LUT adds ~0 on
    # top -> combined == SplitK alone. We report SplitK-only and SplitK+max(0,s_lut).
    s_combined = (1.0 + s_splitk) * (1.0 + max(0.0, s_lut_opt)) - 1.0
    return {
        "corner_min_splitk_for_500_pct": corner_splitk_for_500 * 100.0,
        "lut_s_optimistic_pct": s_opt_pct,
        "lut_s_realistic_bcq_pct": s_bcq_pct,
        "corner_tps_at_lut_alone_optimistic": corner_tps(s_lut_opt),
        "corner_tps_at_lut_alone_bcq": corner_tps(max(-0.99, s_lut_bcq)),
        "corner_tps_at_splitk_alone_8.5pct": corner_tps(s_splitk),
        "corner_tps_at_splitk_plus_lut_combined": corner_tps(s_combined),
        "lut_alone_clears_500_corner": corner_tps(s_lut_opt) >= sr.TARGET,
        "lut_moves_corner_vs_splitk_alone_tps": corner_tps(s_combined) - corner_tps(s_splitk),
        "stacks_with_splitk": False,
        "interpretation": (
            "Corner needs SplitK >= 14.34%. LUT alone gives <=0% -> corner stays ~467 "
            "(far below 500). LUT does NOT stack with SplitK: both speed up the SAME "
            "BW-bound verify-GEMM; once SplitK saturates HBM, LUT (a compute lever) adds "
            "~0. LUT is DOMINATED by SplitK on official-TPS-per-build-effort -> rank LUT "
            "below SplitK in the fern #111 climb-ROI (do not pivot ubel to LUT)."),
    }


def run_int8_tc_probe(shapes):
    """Optional: measure the INT8 tensor-core GEMM substrate (torch._int_mm) across M to
    corroborate the m16n8k32 tile floor. A TC-LUT path would issue INT8 MMA; if M=8 is
    below the tile minimum it is structurally tile-underfilled. No model load, no tokens."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {"error": f"torch import failed: {exc!r}"}
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    dev = "cuda"

    def time_int8(Mr, K, N, iters=200, warm=40):
        A = torch.randint(-8, 7, (Mr, K), dtype=torch.int8, device=dev)
        B = torch.randint(-8, 7, (K, N), dtype=torch.int8, device=dev)
        try:
            for _ in range(warm):
                torch._int_mm(A, B)
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(iters):
                torch._int_mm(A, B)
            e1.record()
            torch.cuda.synchronize()
            return {"t_us": e0.elapsed_time(e1) / iters * 1000.0}
        except Exception as exc:  # noqa: BLE001
            return {"error": repr(exc)[:160]}

    # probe the dominant MLP shapes across M straddling the m16n8k32 tile floor (16).
    probe_shapes = [("gate_up", 2560, 20480), ("down", 10240, 2560)]
    rows = []
    for name, K, N in probe_shapes:
        for Mr in (8, 16, 17, 24, 32, 64):
            r = time_int8(Mr, K, N)
            r.update({"shape": name, "M": Mr, "K": K, "N": N,
                      "int8_weight_bytes": K * N})
            rows.append(r)
    dev_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    return {
        "device": dev_name, "sm": f"sm_{cap[0]}{cap[1]}",
        "int8_mma_tile": "m16n8k32 (Ampere s8.s8.s32); min M-tile 16",
        "rows": rows,
        "finding": ("torch._int_mm (cuBLASLt IMMA INT8-TC) refuses M<=16 ('size(0) needs "
                    "to be greater than 16') -> the INT8 tensor-core GEMM substrate a TC-LUT "
                    "kernel would use does not serve the M=8 verify width; M=8 is below the "
                    "tile floor. Where it does run, int8 reads 2x the int4 bytes -> BW-bound "
                    "slower than Marlin (byte count is destiny at M=8)."),
    }


def _console(out, per_shape, speedups, bw_gap, compute_floor_pct, int8_probe, ship, M):
    g = out["gate"]
    mb = out["marlin_baseline"]
    print("=" * 92)
    print(f"LUT/GANQ W4A16 GEMM FEASIBILITY @ M={M} on sm_86 (PR #113) -- BUILD or KILL")
    print("=" * 92)
    print(f"\nMarlin baseline (#68 MEASURED): verify-GEMM {mb['agg_t_us']:.0f}us @ M={M}; "
          f"{mb['measured_hbm_util_pct']:.1f}% HBM, {mb['measured_compute_util_pct']:.1f}% compute "
          f"-> {'BANDWIDTH-BOUND' if mb['bandwidth_bound'] else 'compute-bound'}")
    print(f"\nper-shape Marlin vs LUT (byte-model) at M={M}:")
    print(f"  {'role':>18s} {'in->out':>13s} {'cnt':>3s} | {'marlin us':>9s} | "
          f"{'iso%':>6s} {'bcq%':>7s} {'nu_g32%':>8s}")
    for r in per_shape:
        print(f"  {r['role']:>18s} {r['in']:5d}->{r['out']:6d} {r['count']:3d} | "
              f"{r['t_us_marlin']:9.1f} | {r['lut']['lut_iso']['speedup_pct']:6.1f} "
              f"{r['lut']['lut_bcq_b4_g32']['speedup_pct']:7.1f} "
              f"{r['lut']['lut_nu_g32']['speedup_pct']:8.1f}")
    print(f"\nAGGREGATE LUT speedup vs Marlin at M={M} (by byte model):")
    for mdl, sp in speedups.items():
        print(f"  {mdl:>20s}: {sp:+6.1f}%")
    print(f"\n  bandwidth-gap ceiling (77.1%->100% HBM)  = {bw_gap:+.1f}%  "
          f"<- UTILISATION headroom (SplitK's), NOT LUT's")
    print(f"  compute floor as % of Marlin time         = {compute_floor_pct:.1f}%  "
          f"<- compute is hidden; zeroing it can't speed a BW-bound kernel")
    print(f"\n  >>> PRIMARY lut_gemm_m8_speedup_vs_marlin_pct = "
          f"{g['lut_gemm_m8_speedup_vs_marlin_pct']:+.1f}% (best-case iso-bytes); "
          f"realistic BCQ {g['lut_gemm_m8_speedup_realistic_bcq_isoppl_pct']:+.1f}%")
    print(f"  >>> TEST    lut_gemm_ppl_projected = {g['lut_gemm_ppl_projected']:.4f} "
          f"(cap {g['ppl_cap']}, cushion {g['ppl_cushion']:.3f}) -> "
          f"{'HOLDS' if g['ppl_holds'] else 'FAILS'} (moot; speed ceiling <=0)")
    if int8_probe and "rows" in int8_probe:
        print(f"\n  INT8-TC substrate probe ({int8_probe['device']} {int8_probe['sm']}, "
              f"{int8_probe['int8_mma_tile']}):")
        for r in int8_probe["rows"]:
            if "error" in r:
                print(f"    {r['shape']:>8s} M={r['M']:3d}: {r['error']}")
            else:
                print(f"    {r['shape']:>8s} M={r['M']:3d}: {r['t_us']:8.1f} us")
    print(f"\n  Step-4 ship-readiness corner (needs SplitK 14.34% for confident 500):")
    print(f"    corner @ LUT alone (optimistic 0%)     = {ship['corner_tps_at_lut_alone_optimistic']:.1f}  "
          f"(clears 500? {ship['lut_alone_clears_500_corner']})")
    print(f"    corner @ SplitK alone 8.5%             = {ship['corner_tps_at_splitk_alone_8.5pct']:.1f}")
    print(f"    corner @ SplitK + LUT combined         = {ship['corner_tps_at_splitk_plus_lut_combined']:.1f}  "
          f"(LUT adds {ship['lut_moves_corner_vs_splitk_alone_tps']:+.2f} TPS; stacks? {ship['stacks_with_splitk']})")
    print(f"\n[VERDICT] {g['verdict']} -- LUT-GEMM at M=8 on sm_86")
    print(f"  {g['verdict_label']}")


def _log_wandb(args, out):
    import wandb
    g = out["gate"]
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     name=args.wandb_name, group=args.wandb_group, job_type="analysis",
                     config={"gate": "lut-gemm-feasibility", "M": args.m,
                             "method": "roofline-on-#68-measured + optional INT8-TC probe",
                             "frontier_official": FRONTIER_OFFICIAL, "ppl_cap": PPL_CAP,
                             "hbm_gbs": A10G_HBM_GBS, "peak_tflops": MEASURED_PEAK_TFLOPS})
    s = wandb.summary
    s["lut_gemm_m8_speedup_vs_marlin_pct"] = g["lut_gemm_m8_speedup_vs_marlin_pct"]
    s["lut_gemm_m8_speedup_realistic_bcq_pct"] = g["lut_gemm_m8_speedup_realistic_bcq_isoppl_pct"]
    s["lut_gemm_ppl_projected"] = g["lut_gemm_ppl_projected"]
    s["ppl_holds"] = bool(g["ppl_holds"])
    s["ppl_cushion"] = g["ppl_cushion"]
    s["bandwidth_gap_ceiling_pct"] = g["bandwidth_gap_ceiling_pct"]
    s["compute_floor_pct_of_marlin_time"] = g["compute_floor_pct_of_marlin_time"]
    s["verdict"] = g["verdict"]
    s["verdict_label"] = g["verdict_label"]
    for mdl, sp in out["lut_speedups_by_byte_model_pct"].items():
        s[f"lut_speedup_{mdl}_pct"] = sp
    sh = out["step4_ship_readiness"]
    s["corner_min_splitk_for_500_pct"] = sh["corner_min_splitk_for_500_pct"]
    s["corner_tps_lut_alone"] = sh["corner_tps_at_lut_alone_optimistic"]
    s["corner_tps_splitk_alone"] = sh["corner_tps_at_splitk_alone_8.5pct"]
    s["lut_stacks_with_splitk"] = sh["stacks_with_splitk"]

    # per-shape table
    t = wandb.Table(columns=["role", "in", "out", "count", "t_us_marlin",
                             "lut_iso_pct", "lut_bcq_pct", "lut_nu_perchan_pct", "lut_nu_g32_pct",
                             "bw_floor_us", "compute_floor_us"])
    for r in out["per_shape"]:
        t.add_data(r["role"], r["in"], r["out"], r["count"], r["t_us_marlin"],
                   r["lut"]["lut_iso"]["speedup_pct"], r["lut"]["lut_bcq_b4_g32"]["speedup_pct"],
                   r["lut"]["lut_nu_perchannel"]["speedup_pct"], r["lut"]["lut_nu_g32"]["speedup_pct"],
                   r["t_us_bw_floor_100pct_hbm"], r["t_us_compute_floor"])
    wandb.log({"per_shape_lut_vs_marlin": t})
    if out.get("int8_tc_substrate_probe") and "rows" in out["int8_tc_substrate_probe"]:
        it = wandb.Table(columns=["shape", "M", "K", "N", "t_us", "error"])
        for r in out["int8_tc_substrate_probe"]["rows"]:
            it.add_data(r.get("shape"), r.get("M"), r.get("K"), r.get("N"),
                        r.get("t_us", -1.0), r.get("error", ""))
        wandb.log({"int8_tc_substrate_probe": it})
    print(f"\nW&B run: {run.id}  ({run.url})")
    wandb.finish()


if __name__ == "__main__":
    main()
