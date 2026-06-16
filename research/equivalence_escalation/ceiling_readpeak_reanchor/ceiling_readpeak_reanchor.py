#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Independent re-anchor of the 510.87 ceiling's read-peak basis (PR #463, lawine).

LOCAL A10G (sm_86, on-target) MEASUREMENT + analysis only. NO HF Job, NO submission,
NO served-file change, NO deploy. Greedy/PPL pinned BY CONSTRUCTION (a STREAM-read
microbench emits no tokens). BASELINE stays 481.53; PPL anchor stays 2.3772.

THE QUESTION (a skeptic's sharpest attack on the PRIZE axis)
-----------------------------------------------------------
The escalation packet's whole +17..+29 TPS prize and the unified absolute ceiling
  unified_absolute_ceiling_tps = 510.87 +/- 4.82   (land #457, h0uggl9i)
rest on ONE microbenchmarked number: ubel #450's (c5oyb7gv) achieved DRAM read-peak
of 517.58 GB/s on this pod's A10G. "Is 517.58 GB/s robust, or a measurement
artifact?" This card re-measures that ONE leg with a FRESH HAND / FRESH SEEDS /
FRESH PROCESS (N>=5 independent subprocesses), recomposes the ceiling through #457's
EXACT composition (round-tripping the committed JSON; re-deriving ONLY the read-peak
leg), and reports whether 510.87 +/- 4.82 holds.

WHAT IS RE-DERIVED vs ROUND-TRIPPED (single-variable design)
------------------------------------------------------------
RE-DERIVED (the leg under test):
  * the achieved STREAM read-peak (GB/s) -- ubel #450's measure_peak_bw method:
    torch.sum() over a 1 GiB bf16 buffer, iters=50 / warmup=40, after a heavy
    boost-clock warmup. Distinct seeds, distinct fresh processes.
ROUND-TRIPPED from the committed #450 / #457 JSON (NOT re-measured):
  * GEMM_US, GEMM_BYTES (the demand leg: served int4 verify-GEMM time + byte model)
  * ACHIEVED_GEMM_BW (deployed kernel's measured BW, for the headroom fraction)
  * CYCLE_WALL_US 7903, REALIZED_FRONTIER 467.14, LAMBDA1 spec-UB 520.953, sigma_hw
The ONLY thing that moves vs #457 is the read-peak basis -> any ceiling drift is
attributable to the read-peak alone.

THE COMPOSITION (verified to round-trip #450/#457 to machine precision)
----------------------------------------------------------------------
  saved_us(peak) = GEMM_US - GEMM_BYTES / peak           # perfect f->1 re-tiling floor
  new_wall       = CYCLE_WALL_US - saved_us
  ceiling_tps    = min( REALIZED_FRONTIER * CYCLE_WALL_US / new_wall , LAMBDA1 )
  headroom_frac  = 1 - ACHIEVED_GEMM_BW / peak           # ~16% achieved-BW headroom
recompose(517.58) reproduces 510.8724230449973 and saved_us 676.5237733221147;
recompose(600) reproduces the spec-UB 520.953. (asserted in the self-test.)

Reproduce:
  cd target/ && CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
    research/equivalence_escalation/ceiling_readpeak_reanchor/ceiling_readpeak_reanchor.py \
    --seeds 101 202 303 404 505 606 707 \
    --wandb_group equivalence-escalation-anchors \
    --wandb_name lawine/ceiling-readpeak-reanchor
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.normpath(os.path.join(_here, "..", "..", ".."))

# committed artifacts we round-trip (re-derive nothing in them) ---------------
ROOFLINE_JSON = os.path.join(_repo, "research", "speed",
                             "gemm_roofline_bw_ceiling", "roofline_ceiling.json")
UNIFIED_JSON = os.path.join(_repo, "research", "validity",
                            "unified_absolute_ceiling",
                            "unified_absolute_ceiling_results.json")
BODY_GEMM = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
PPL_ANCHOR = 2.3772           # deployed PPL (pinned; a read microbench cannot change it)
PPL_GATE = 2.42


# --------------------------------------------------------------------------- #
# WORKER: one fresh process measures the read/copy/gemm peak BW for one seed.
# (kept import-light: torch only, no vLLM -- the GEMM/byte legs are round-tripped)
# --------------------------------------------------------------------------- #
def _worker(seed: int, iters: int, warmup: int, rounds: int) -> dict:
    import gc
    import torch

    dev = torch.device("cuda:0")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()

    def timed(fn):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            fn()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / iters / 1e3          # seconds / call

    # heavy warmup -> A10G boost clock (ubel #450 basis)
    big = torch.randn(2048, 2048, dtype=torch.bfloat16, device=dev)
    for _ in range(200):
        big = big @ big
    torch.cuda.synchronize()
    del big
    gc.collect()
    torch.cuda.empty_cache()

    # ubel #450 measure_peak_bw shapes: 512Mi bf16 = 1 GiB; read=sum(1x), copy(2x)
    N = 512 * 1024 * 1024
    a = torch.empty(N, dtype=torch.bfloat16, device=dev).uniform_(-1, 1)
    b = torch.empty(N, dtype=torch.bfloat16, device=dev)
    nb = N * 2

    read_bws, copy_bws = [], []
    for _ in range(rounds):
        t_read = timed(lambda: torch.sum(a))
        read_bws.append(nb / t_read / 1e9)
        t_copy = timed(lambda: b.copy_(a))
        copy_bws.append(2 * nb / t_copy / 1e9)

    # saturating bf16 GEMM @ M=8 (kanna #269 anchor; context only)
    hidden, M = 2560, 8
    out = (512 * 2 ** 20) // (hidden * 2)
    w = torch.randn(out, hidden, dtype=torch.bfloat16, device=dev)
    x = torch.randn(M, hidden, dtype=torch.bfloat16, device=dev)
    gb = out * hidden * 2 + (M * hidden + M * out) * 2
    t_gemm = timed(lambda: torch.matmul(x, w.t()))

    peak_vram_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    del a, b, w, x
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "seed": seed,
        "device": name,
        "sm": f"{cap[0]}{cap[1]}",
        "torch": torch.__version__,
        "read_bw_gbps": statistics.median(read_bws),
        "read_bw_gbps_rounds": read_bws,
        "copy_bw_gbps": statistics.median(copy_bws),
        "bf16gemm_m8_gbps": gb / t_gemm / 1e9,
        "peak_vram_gib": peak_vram_gib,
        "rounds": rounds,
        "iters": iters,
        "warmup": warmup,
    }


# --------------------------------------------------------------------------- #
# round-trip the committed ceiling composition (re-derives NOTHING here)
# --------------------------------------------------------------------------- #
def load_committed() -> dict:
    roof = json.load(open(ROOFLINE_JSON))
    v = roof["verdict"]
    comp = roof["components"]
    gemm_us = sum(comp[c]["us"] for c in BODY_GEMM)
    gemm_bytes = sum(comp[c]["total_bytes"] for c in BODY_GEMM)
    # round-trip integrity vs the verdict's own rolled-up numbers
    assert abs(gemm_us - v["gemm_us"]) < 1e-6, (gemm_us, v["gemm_us"])
    assert abs(gemm_bytes / 1e6 - v["gemm_total_bytes_mb"]) < 1e-3
    uni = json.load(open(UNIFIED_JSON))["synthesis"]["constants"]
    return {
        "GEMM_US": gemm_us,
        "GEMM_BYTES": gemm_bytes,
        "ACHIEVED_GEMM_BW": v["achieved_gemm_bw_gbps"],
        "CYCLE_WALL_US": v["cycle_wall_us"],
        "REALIZED_FRONTIER_TPS": v["realized_frontier_tps"],
        "DEPLOYED_TPS": v["frontier_deployed_tps"],
        "LAMBDA1_CEILING_TPS": v["lambda1_ceiling_tps"],
        "PEAK_READ_450": v["peak_read_gbps"],
        "SPEC_BW": v["peak_spec_gbps"],
        "CEIL_READPEAK_450": v["ceiling_tps_read_peak"],
        "SIGMA_HW": uni["sigma_hw"],
        "CEIL_READPEAK_457": uni["ceil_readpeak"],
        "CEIL_SPEC_457": uni["ceil_spec"],
        "DEMAND_RAW_457": uni["demand_raw"],
    }


def recompose(read_peak_gbps: float, K: dict) -> dict:
    """#457's EXACT ceiling composition with ONLY the read-peak leg swapped."""
    saved_us = K["GEMM_US"] - (K["GEMM_BYTES"] / (read_peak_gbps * 1e9)) * 1e6
    new_wall = K["CYCLE_WALL_US"] - saved_us
    speedup = K["CYCLE_WALL_US"] / new_wall if new_wall > 0 else float("inf")
    tps_uncapped = K["REALIZED_FRONTIER_TPS"] * speedup
    tps_capped = min(tps_uncapped, K["LAMBDA1_CEILING_TPS"])
    return {
        "read_peak_gbps": read_peak_gbps,
        "saved_us": saved_us,
        "new_wall_us": new_wall,
        "speedup": speedup,
        "tps_uncapped": tps_uncapped,
        "tps_capped": tps_capped,
        "binds_at_lambda1": bool(tps_uncapped >= K["LAMBDA1_CEILING_TPS"]),
        "headroom_frac": 1.0 - K["ACHIEVED_GEMM_BW"] / read_peak_gbps,
        "headroom_over_deployed": tps_capped - K["DEPLOYED_TPS"],
        "headroom_over_realized": tps_capped - K["REALIZED_FRONTIER_TPS"],
    }


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[101, 202, 303, 404, 505, 606, 707])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=9)       # internal rounds / seed
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--seed", type=int)                    # worker single seed
    ap.add_argument("--output", default=os.path.join(_here, "reanchor_results.json"))
    ap.add_argument("--wandb_project",
                    default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity",
                    default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="equivalence-escalation-anchors")
    ap.add_argument("--wandb_name", default="lawine/ceiling-readpeak-reanchor")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    # ---- worker path: measure one seed in this fresh process, print JSON -----
    if args.worker:
        import torch
        assert torch.cuda.is_available(), "CUDA unavailable (need CUDA_VISIBLE_DEVICES=0)"
        res = _worker(args.seed, args.iters, args.warmup, args.rounds)
        print("WORKER_JSON:" + json.dumps(res), flush=True)
        return 0

    # ---- driver path ---------------------------------------------------------
    K = load_committed()
    print(f"[reanchor] round-trip OK: GEMM_US={K['GEMM_US']:.3f}us "
          f"GEMM_BYTES={K['GEMM_BYTES']/1e6:.2f}MB cycle={K['CYCLE_WALL_US']}us "
          f"realized={K['REALIZED_FRONTIER_TPS']} lambda1={K['LAMBDA1_CEILING_TPS']} "
          f"sigma_hw={K['SIGMA_HW']}", flush=True)
    print(f"[reanchor] anchor under test: ceil_readpeak(#450/#457)="
          f"{K['CEIL_READPEAK_457']:.4f}  on peak_read_450={K['PEAK_READ_450']:.4f} GB/s",
          flush=True)

    # spawn N>=5 FRESH PROCESSES (one per seed), sequentially (single GPU) ------
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    seeds_res = []
    for s in args.seeds:
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--seed", str(s),
               "--iters", str(args.iters), "--warmup", str(args.warmup),
               "--rounds", str(args.rounds)]
        print(f"[reanchor] worker seed={s} (fresh process) ...", flush=True)
        out = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=_here)
        line = next((ln for ln in out.stdout.splitlines()
                     if ln.startswith("WORKER_JSON:")), None)
        if line is None:
            print(f"[reanchor] seed={s} FAILED\nSTDOUT:\n{out.stdout[-2000:]}"
                  f"\nSTDERR:\n{out.stderr[-2000:]}", flush=True)
            continue
        r = json.loads(line[len("WORKER_JSON:"):])
        seeds_res.append(r)
        print(f"[reanchor]   seed={s} read={r['read_bw_gbps']:.2f} "
              f"copy={r['copy_bw_gbps']:.2f} bf16gemm@M8={r['bf16gemm_m8_gbps']:.2f} GB/s",
              flush=True)

    n_ok = len(seeds_res)
    assert n_ok >= 5, f"need >=5 successful seeds, got {n_ok}"

    reads = [r["read_bw_gbps"] for r in seeds_res]
    median_read = statistics.median(reads)
    mean_read = statistics.fmean(reads)
    sigma_read = statistics.pstdev(reads) if n_ok > 1 else 0.0
    sample_sd = statistics.stdev(reads) if n_ok > 1 else 0.0
    ci95 = 1.96 * sample_sd / math.sqrt(n_ok) if n_ok > 1 else 0.0
    min_read, max_read = min(reads), max(reads)

    drift_gbps = median_read - K["PEAK_READ_450"]
    drift_pct = 100.0 * drift_gbps / K["PEAK_READ_450"]
    within_95ci = abs(drift_gbps) <= max(1.96 * sigma_read, 1e-9)
    within_1pct = abs(drift_pct) <= 1.0
    read_peak_reproduces_450 = bool(within_95ci or within_1pct)

    # ---- recompose the ceiling on MY read-peak (and the brackets) -----------
    cz_med = recompose(median_read, K)
    cz_lo = recompose(median_read - sigma_read, K)
    cz_hi = recompose(median_read + sigma_read, K)
    cz_450 = recompose(K["PEAK_READ_450"], K)
    cz_spec = recompose(K["SPEC_BW"], K)

    reanchored_ceiling_tps = cz_med["tps_capped"]
    # symmetric measurement-sigma envelope on the ceiling (monotone in read-peak)
    ceiling_sigma_from_read = (cz_hi["tps_capped"] - cz_lo["tps_capped"]) / 2.0
    ceiling_drift_vs_510 = reanchored_ceiling_tps - K["CEIL_READPEAK_457"]
    ceiling_holds_within_sigma_hw = bool(abs(ceiling_drift_vs_510) <= K["SIGMA_HW"])
    achieved_bw_headroom_frac = cz_med["headroom_frac"]

    # #457 sigma_hw band on the (committed) anchor
    band_lo = K["CEIL_READPEAK_457"] - K["SIGMA_HW"]
    band_hi = K["CEIL_READPEAK_457"] + K["SIGMA_HW"]

    # ---- SELF-TEST (structural validity of the re-anchor; NOT contingent on
    #      whether the ceiling holds -- that is the finding, not a pass gate) --
    rt_450 = recompose(K["PEAK_READ_450"], K)["tps_capped"]
    rt_spec = recompose(K["SPEC_BW"], K)["tps_capped"]
    st = {
        "a_roundtrip_450_reproduces_510": abs(rt_450 - K["CEIL_READPEAK_457"]) <= 1e-6,
        "b_roundtrip_spec_reproduces_520": abs(rt_spec - K["CEIL_SPEC_457"]) <= 1e-3,
        "c_at_least_5_fresh_seeds": n_ok >= 5,
        "d_nan_clean": all(math.isfinite(x) for x in
                           [median_read, sigma_read, reanchored_ceiling_tps,
                            ceiling_sigma_from_read, achieved_bw_headroom_frac,
                            cz_450["tps_capped"], cz_spec["tps_capped"]]),
        "e_read_below_spec": bool(median_read < K["SPEC_BW"]),
        "f_headroom_frac_in_unit": bool(0.0 < achieved_bw_headroom_frac < 1.0),
        "g_brackets_monotone": bool(
            cz_med["tps_uncapped"] <= cz_spec["tps_uncapped"] + 1e-9
            and cz_spec["tps_capped"] <= K["LAMBDA1_CEILING_TPS"] + 1e-6),
        "h_ppl_anchor_within_gate": bool(PPL_ANCHOR <= PPL_GATE),
        "i_demand_exceeds_supply": bool(K["DEMAND_RAW_457"] > cz_spec["tps_capped"]),
    }
    self_test_passes = all(st.values())

    verdict = {
        # ---- the PR-required fields ----
        "reanchored_read_peak_gbps": median_read,
        "read_peak_reproduces_450": read_peak_reproduces_450,
        "reanchored_ceiling_tps": reanchored_ceiling_tps,           # PRIMARY metric
        "ceiling_holds_within_sigma_hw": ceiling_holds_within_sigma_hw,
        "achieved_bw_headroom_frac": achieved_bw_headroom_frac,
        "ceiling_reanchor_self_test_passes": self_test_passes,
        "analysis_only": True,
        "no_served_file_change": True,
        "official_tps": 0,
        "ppl": PPL_ANCHOR,
        # ---- read-peak measurement detail ----
        "n_seeds_ok": n_ok,
        "read_peak_mean_gbps": mean_read,
        "read_peak_sigma_gbps": sigma_read,
        "read_peak_sample_sd_gbps": sample_sd,
        "read_peak_ci95_gbps": ci95,
        "read_peak_min_gbps": min_read,
        "read_peak_max_gbps": max_read,
        "read_peak_450_gbps": K["PEAK_READ_450"],
        "read_peak_drift_gbps": drift_gbps,
        "read_peak_drift_pct": drift_pct,
        "read_peak_within_95ci": bool(within_95ci),
        "read_peak_within_1pct": bool(within_1pct),
        "spec_peak_fraction": median_read / K["SPEC_BW"],
        "copy_peak_median_gbps": statistics.median([r["copy_bw_gbps"] for r in seeds_res]),
        "bf16gemm_m8_median_gbps": statistics.median([r["bf16gemm_m8_gbps"] for r in seeds_res]),
        # ---- ceiling drift / envelope ----
        "ceil_readpeak_anchor_457": K["CEIL_READPEAK_457"],
        "ceiling_drift_vs_510": ceiling_drift_vs_510,
        "ceiling_sigma_from_read_tps": ceiling_sigma_from_read,
        "sigma_hw_tps": K["SIGMA_HW"],
        "anchor_band_lo": band_lo,
        "anchor_band_hi": band_hi,
        "reanchored_in_anchor_band": bool(band_lo <= reanchored_ceiling_tps <= band_hi),
        "saved_us_allowance": cz_med["saved_us"],
        # ---- robustness brackets ----
        "bracket_a_my_median_tps": cz_med["tps_capped"],
        "bracket_b_450_517p58_tps": cz_450["tps_capped"],
        "bracket_c_spec_600_tps_capped": cz_spec["tps_capped"],
        "bracket_c_spec_600_tps_uncapped": cz_spec["tps_uncapped"],
        # ---- prize axis (round-tripped framing) ----
        "deployed_tps": K["DEPLOYED_TPS"],
        "realized_frontier_tps": K["REALIZED_FRONTIER_TPS"],
        "prize_over_deployed_at_realistic_ceiling": reanchored_ceiling_tps - K["DEPLOYED_TPS"],
        "prize_over_strict_at_realistic_ceiling": reanchored_ceiling_tps - K["REALIZED_FRONTIER_TPS"],
        "realistic_basis_is_achieved_read_peak": True,
        "spec_basis_is_over_optimistic_ub": True,
        "ceiling_identity_ppl_is_na_physical_marker": True,
        # ---- housekeeping ----
        "self_test_conditions": st,
        "device": seeds_res[0]["device"],
        "sm": seeds_res[0]["sm"],
        "peak_vram_gib": max(r["peak_vram_gib"] for r in seeds_res),
    }

    verdict["handoff_line"] = (
        f"independent read-peak re-anchor (N={n_ok} fresh processes/seeds): "
        f"median {median_read:.2f} +/- {sigma_read:.2f} GB/s "
        f"(drift {drift_gbps:+.2f} / {drift_pct:+.2f}% vs #450's 517.58; "
        f"reproduces={read_peak_reproduces_450}). Recomposed unified ceiling = "
        f"{reanchored_ceiling_tps:.2f} TPS (drift {ceiling_drift_vs_510:+.2f} vs 510.87; "
        f"{'HOLDS' if ceiling_holds_within_sigma_hw else 'MOVES'} within sigma_hw "
        f"{K['SIGMA_HW']:.2f}). Brackets: realistic(median)={cz_med['tps_capped']:.2f}, "
        f"450-basis={cz_450['tps_capped']:.2f}, spec-600-UB={cz_spec['tps_capped']:.2f} "
        f"(uncapped {cz_spec['tps_uncapped']:.2f}). Achieved-BW headroom "
        f"{achieved_bw_headroom_frac*100:.1f}% (677us allowance); prize +"
        f"{reanchored_ceiling_tps-K['DEPLOYED_TPS']:.1f} over deployed 481.53 is measured "
        f"against the realistic (achieved-read-peak) basis; spec-600 520.95 is an "
        f"over-optimistic physical-limit marker, not an operating point. analysis-only.")

    payload = {
        "config": {
            "pr": 463, "agent": "lawine", "kind": "ceiling-readpeak-reanchor",
            "seeds": args.seeds, "n_seeds_ok": n_ok, "iters": args.iters,
            "warmup": args.warmup, "rounds": args.rounds,
            "device": seeds_res[0]["device"], "sm": seeds_res[0]["sm"],
            "torch": seeds_res[0]["torch"], "analysis_only": True,
            "no_served_file_change": True,
            "method": "ubel#450 measure_peak_bw STREAM read (1 GiB bf16, torch.sum) "
                      "across N>=5 FRESH subprocesses/seeds; #457 ceiling composition "
                      "round-tripped, read-peak leg re-derived. No serve change, no HF "
                      "Job, no submission. Greedy/PPL pinned by construction.",
        },
        "round_tripped_constants": K,
        "per_seed": seeds_res,
        "read_peak_summary": {
            "median_gbps": median_read, "mean_gbps": mean_read,
            "sigma_gbps": sigma_read, "sample_sd_gbps": sample_sd,
            "ci95_gbps": ci95, "min_gbps": min_read, "max_gbps": max_read,
        },
        "brackets": {"a_my_median": cz_med, "b_450_517p58": cz_450, "c_spec_600": cz_spec},
        "ceiling_sigma_envelope": {
            "reanchored_ceiling_tps": reanchored_ceiling_tps,
            "ceiling_sigma_from_read_tps": ceiling_sigma_from_read,
            "ceiling_lo_from_read": cz_lo["tps_capped"],
            "ceiling_hi_from_read": cz_hi["tps_capped"],
            "anchor_band_lo": band_lo, "anchor_band_hi": band_hi,
            "sigma_hw_tps": K["SIGMA_HW"],
        },
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(f"[reanchor] wrote {args.output}", flush=True)

    # ---- print verdict -------------------------------------------------------
    print("\n[reanchor] ===== read-peak (fresh-process seeds) =====", flush=True)
    for r in seeds_res:
        print(f"  seed {r['seed']:>4}: read {r['read_bw_gbps']:7.2f}  "
              f"copy {r['copy_bw_gbps']:7.2f}  bf16gemm@M8 {r['bf16gemm_m8_gbps']:7.2f} GB/s",
              flush=True)
    print(f"  median {median_read:.2f} +/- {sigma_read:.2f} (sample_sd {sample_sd:.2f}, "
          f"95%CI +/-{ci95:.2f})  [{min_read:.2f}, {max_read:.2f}]", flush=True)
    print(f"  drift vs #450 517.58: {drift_gbps:+.2f} GB/s ({drift_pct:+.2f}%)  "
          f"reproduces_450={read_peak_reproduces_450}", flush=True)
    print(f"  spec-peak fraction = {median_read/K['SPEC_BW']*100:.2f}% of 600 GB/s; "
          f"achieved-BW headroom = {achieved_bw_headroom_frac*100:.2f}%", flush=True)
    print("\n[reanchor] ===== recomposed ceiling (read-peak leg re-derived) =====", flush=True)
    print(f"  (a) my median {median_read:.2f} -> {cz_med['tps_capped']:.2f} TPS  "
          f"(saved {cz_med['saved_us']:.1f}us)", flush=True)
    print(f"  (b) #450 517.58           -> {cz_450['tps_capped']:.2f} TPS  (committed 510.87)",
          flush=True)
    print(f"  (c) spec 600 (UB)         -> {cz_spec['tps_capped']:.2f} TPS capped "
          f"(uncapped {cz_spec['tps_uncapped']:.2f})", flush=True)
    print(f"  reanchored_ceiling_tps = {reanchored_ceiling_tps:.4f}  "
          f"(+/- {ceiling_sigma_from_read:.4f} from read-sigma)", flush=True)
    print(f"  drift vs 510.87 = {ceiling_drift_vs_510:+.4f}  "
          f"HOLDS_within_sigma_hw({K['SIGMA_HW']})={ceiling_holds_within_sigma_hw}  "
          f"in_band[{band_lo:.2f},{band_hi:.2f}]={verdict['reanchored_in_anchor_band']}",
          flush=True)
    print(f"\n[reanchor] self_test={self_test_passes}  {st}", flush=True)
    print(f"  {verdict['handoff_line']}", flush=True)

    if not args.no_wandb:
        try:
            _log_wandb(args, payload)
        except Exception as exc:  # noqa: BLE001
            print(f"[reanchor] W&B logging failed (non-fatal): {exc!r}", flush=True)

    return 0 if self_test_passes else 1


def _log_wandb(args, payload):
    import wandb

    v = payload["verdict"]
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type="analysis", config=payload["config"])
    # per-seed read-peak table
    st = wandb.Table(columns=["seed", "read_bw_gbps", "copy_bw_gbps",
                              "bf16gemm_m8_gbps", "peak_vram_gib"])
    for r in payload["per_seed"]:
        st.add_data(r["seed"], r["read_bw_gbps"], r["copy_bw_gbps"],
                    r["bf16gemm_m8_gbps"], r["peak_vram_gib"])
    run.log({"read_peak_per_seed": st})
    # ceiling bracket table
    bt = wandb.Table(columns=["basis", "read_peak_gbps", "saved_us", "tps_uncapped",
                              "tps_capped", "headroom_frac", "role"])
    roles = {"a_my_median": "realistic_reanchored",
             "b_450_517p58": "committed_anchor",
             "c_spec_600": "over_optimistic_ub"}
    for k, c in payload["brackets"].items():
        bt.add_data(k, c["read_peak_gbps"], c["saved_us"], c["tps_uncapped"],
                    c["tps_capped"], c["headroom_frac"], roles[k])
    run.log({"ceiling_brackets": bt})
    run.summary.update({k: val for k, val in v.items()
                        if isinstance(val, (int, float, bool, str))})
    run.finish()
    print(f"[reanchor] W&B run: {run.url}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
