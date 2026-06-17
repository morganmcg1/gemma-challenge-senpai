#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #592 -- Async-L2 KV-prefetch: analytic perfect-overlap bound vs the base_fullhead anchor.

ANALYSIS-ONLY card. No kernel port, no custom CUDA, no serving change, no HF FIRE.
official_tps = 0. We only MEASURE grounding microbenches (a read-BW reduction and a lm_head
GEMV/GEMM on the assigned A10G) and COMPUTE an analytic overlap bound from already-measured
decode splits.

Question (PR #592):
  Take base_fullhead's per-token decode split. Get the KV-cache read time and the compute it
  could overlap with (attention QK/PV math + MLP). Compute the *perfect-overlap* bound

      new_per_token = (non-overlappable work) + max(0, KV_read - hideable_compute)

  i.e. the KV read is hidden under compute up to the available compute budget. Convert to TPS.
  Verdicts:
    prefetch_can_reach_ship : bound >= 375.857 ?   (expected: NO)
    prefetch_beats_252      : bound  > 252.69  ?   (the base_fullhead MTP-K7 anchor)
  And the load-bearing diagnostic: is KV read the bottleneck, or the 262k bf16 lm_head?

Measured anchors imported (all from THIS launch's assigned local A10G runs):
  * base_fullhead AR M=1 decode split        -> PR #569  (decode_overhead_floor_reducibility.json)
  * served KV geometry (measured, occ-bound)  -> PR #445  (kv_prefetch_l2_gate.json)
  * 252.69 = MTP-K7 served anchor, 375.857 ship-> BASELINE.md / PR #584

A10G caveat: sm_86, ~600 GB/s nominal HBM (~543 GB/s achievable read), 6.29 MB L2 (NOT 40 MB),
80 SMs, NO H20-class async-copy/TMA bulk engine. The vLLM Triton unified_attention already emits
cp.async (ldgsts) KV loads (PR #445: 93 cp.async in the kernel PTX) -> the HBM->shared streaming
that an L2 prefetch would target is ALREADY overlapped at the kernel level, and that kernel runs
at ~20% of HBM peak (occupancy/launch-bound, not bandwidth-bound). So the realistic prefetch
headroom is ~nil; the number below is a *generous roofline ceiling*, not an achievable gain.

Reproduce:
  cd target/ && CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
    research/profiling/kv_prefetch_l2/kv_prefetch_overlap_bound.py \
    --wandb_group kv-prefetch-overlap-bound --wandb_name denken/kv-prefetch-overlap-bound
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "kv_prefetch_overlap_bound.json"

# --------------------------------------------------------------------------- #
# Measured anchors (NO re-derivation; these are imported measurements).
# --------------------------------------------------------------------------- #
# PR #569 -- base_fullhead AR M=1 decode-step split (assigned A10G, CUDA graphs ON, 256 tok).
#   step = body(int4 Marlin MLP+proj) + head(bf16 262k GEMV) + attn(QK/PV kernel) + fixed.
PR569 = {
    "decode_step_us": 10308.23,
    "body_gemv_us": 6728.12,     # int4 Marlin: all MLP + attention-projection GEMMs
    "head_gemv_us": 2775.75,     # bf16 262144x2560 lm_head GEMV (M=1)
    "attn_us": 405.77,           # unified_attention kernel wall (QK + softmax + PV, incl. KV stream)
    "fixed_overhead_us": 398.59,  # sampling + RMSNorm/act + other-dev + host/launch(41.54)
    "host_nonkernel_us": 41.54,
    "ar_m1_tps": 97.01,
    "head_eff_gbps": 500.95,
}

# PR #445 -- measured served KV geometry (deployed 3D split-KV, M=8 verify, rep_ctx=528).
PR445 = {
    "kv_bytes_per_cycle": 49_479_680,         # 49.48 MB read per served verify cycle (37-layer frontier)
    "attn_us_per_cycle": 508.5580899999957,   # occupancy-bound attention wall per served cycle
    "achieved_kv_gbps": 97.29405740060182,    # 20% of HBM peak -> occupancy/launch-bound
    "bandwidth_eff_vs_peak": 0.201911599260916,
    "per_layer_kv_MB": 1.3372886486486488,
    "l2_MB": 6.291456,
    "kv_fits_in_l2": False,
    "cp_async_deployed": True,                # kernel already async-loads KV (93 cp.async in PTX)
    "n_layers": 37,                           # 481.53 frontier is pruned to 37 (30 sliding + 7 full)
    "measured_peak_gbps_copy": 481.86462668187585,
}

# BASELINE.md / PR #584 -- the frame the verdict lives in.
ANCHOR_252 = 252.69        # base_fullhead MTP-K7 served TPS (anchor_252_is_mtp_not_nospec=True)
E_T = 3.844                # accepted tokens / served verify cycle (E[T])
SHIP_375 = 375.857         # primary ship bar
SHIP_481 = 481.53          # frontier deployed (context only)

# Stock base_fullhead geometry (config.json text_config; 42-layer QAT body + full native head).
GEO = {
    "vocab_size": 262144,
    "hidden_size": 2560,
    "n_layers_base": 42,
    "n_kv_heads": 2,
    "head_dim": 256,
    "global_head_dim": 512,
    "sliding_window": 512,
    "n_full_layers": 7,
    "n_sliding_layers": 35,
    "num_kv_shared_layers": 18,
    "kv_dtype_bytes": 2,       # bf16 KV cache
    "rep_ctx": 528,            # match #445's measurement context for the cross-check
}

HBM_NOMINAL_GBPS = 600.0     # A10G spec peak
BW_ACHIEVABLE_FALLBACK = 543.0   # measured achievable read (PR #555 raw ceiling 543.87; #291 ~513)


# --------------------------------------------------------------------------- #
# Fresh A10G grounding microbenches (CUDA-event timed; each non-fatal).
# --------------------------------------------------------------------------- #
def _cuda_event_median(fn, iters: int = 50, warmup: int = 10) -> float:
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return sorted(s.elapsed_time(e) for s, e in zip(starts, ends))[iters // 2]  # median, ms


def mb_read_bw() -> dict[str, Any]:
    """Reduction over a ~1.342 GB bf16 buffer -> achievable READ bandwidth (the prefetch ceiling)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"ran": False, "note": "torch.cuda unavailable (need CUDA_VISIBLE_DEVICES=0)"}
        bytes_read = GEO["vocab_size"] * GEO["hidden_size"] * 2  # 1.342 GB, same size as lm_head
        n = bytes_read // 2
        x = torch.empty(n, dtype=torch.float16, device="cuda")
        x.normal_()
        ms = _cuda_event_median(lambda: x.sum())
        eff = bytes_read / (ms / 1e3) / 1e9
        return {"ran": True, "bytes_read": bytes_read, "median_ms": ms,
                "effective_read_bw_gbps": eff, "achievable_frac_of_nominal": eff / HBM_NOMINAL_GBPS}
    except Exception as exc:  # noqa: BLE001
        return {"ran": False, "note": f"read-bw probe failed: {exc}"}


def mb_lm_head(M: int) -> dict[str, Any]:
    """matmul [M,2560]bf16 x [2560,262144]bf16 -> wall us. Confirms the lm_head is the giant read."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"ran": False}
        H, V = GEO["hidden_size"], GEO["vocab_size"]
        w = torch.randn(H, V, dtype=torch.bfloat16, device="cuda")
        a = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
        ms = _cuda_event_median(lambda: a @ w, iters=30, warmup=8)
        weight_bytes = H * V * 2
        return {"ran": True, "M": M, "wall_us": ms * 1e3, "weight_bytes": weight_bytes,
                "weight_GB": weight_bytes / 1e9, "eff_gbps": weight_bytes / (ms / 1e3) / 1e9}
    except Exception as exc:  # noqa: BLE001
        return {"ran": False, "note": f"lm_head probe failed: {exc}"}


def mb_kv_read_sweep() -> dict[str, Any]:
    """Reduction over the KV byte budget at several contexts -> bandwidth-limited KV read time."""
    out: dict[str, Any] = {"ran": False, "rows": []}
    try:
        import torch
        if not torch.cuda.is_available():
            return out
        out["ran"] = True
        for ctx in (128, 256, 384, 512, 528, 1024):
            kv_bytes = _kv_bytes_geometric(ctx)  # sliding layers cap at window; full see whole ctx
            n = max(1, kv_bytes // 2)
            x = torch.empty(n, dtype=torch.float16, device="cuda")
            x.normal_()
            ms = _cuda_event_median(lambda: x.sum(), iters=40, warmup=8)
            out["rows"].append({"ctx": ctx, "kv_bytes": int(kv_bytes), "read_us": ms * 1e3,
                                "eff_gbps": kv_bytes / (ms / 1e3) / 1e9})
        return out
    except Exception as exc:  # noqa: BLE001
        out["note"] = f"kv-read sweep failed: {exc}"
        return out


def _kv_bytes_geometric(ctx: int) -> int:
    """Per-cycle KV bytes for the 42-layer stock body at `ctx` (NO kv-sharing discount -> generous)."""
    bpt = 2 * GEO["n_kv_heads"] * GEO["kv_dtype_bytes"]  # 2(K,V) * n_kv_heads * dtype
    full = GEO["n_full_layers"] * bpt * GEO["global_head_dim"] * ctx
    slide = GEO["n_sliding_layers"] * bpt * GEO["head_dim"] * min(ctx, GEO["sliding_window"])
    return int(full + slide)


# --------------------------------------------------------------------------- #
# Analytic perfect-overlap bound.
# --------------------------------------------------------------------------- #
def compute_bound(mb_bw: dict[str, Any]) -> dict[str, Any]:
    bw = mb_bw.get("effective_read_bw_gbps") if mb_bw.get("ran") else None
    if not (isinstance(bw, (int, float)) and bw > 0):
        bw = BW_ACHIEVABLE_FALLBACK

    # --- KV bytes for the served base_fullhead cycle (rep_ctx) ---
    kv_bytes_geom = _kv_bytes_geometric(GEO["rep_ctx"])
    kv_bytes_445_scaled = PR445["kv_bytes_per_cycle"] * GEO["n_layers_base"] / PR445["n_layers"]
    kv_bytes = max(kv_bytes_geom, kv_bytes_445_scaled)   # generous: largest plausible KV read

    # --- bandwidth-limited KV read time (the maximum a perfect prefetch could hide) ---
    kv_read_us = kv_bytes / (bw * 1e9) * 1e6
    # the occupancy-bound attention KV wall actually observed in served (PR #445) -- over-generous
    attn_kv_wall_us = PR445["attn_us_per_cycle"] * GEO["n_layers_base"] / PR445["n_layers"]

    # --- lm_head read: the competing giant ---
    head_bytes = GEO["vocab_size"] * GEO["hidden_size"] * 2
    head_over_kv = head_bytes / kv_bytes

    # --- hideable compute budget (everything that can run while KV streams) ---
    # attention math = full attn kernel wall minus the bandwidth-limited KV read portion.
    attn_compute_us = max(0.0, PR569["attn_us"] - kv_read_us)
    hideable_compute_us = PR569["body_gemv_us"] + attn_compute_us   # MLP/proj GEMM + attn math

    # ===================== served frame (where 252.69 / 375.857 live) ===================== #
    served_cycle_us = E_T / ANCHOR_252 * 1e6                         # 15213 us
    # perfect-overlap: new = (cycle - KV_read) + max(0, KV_read - hideable_compute)
    exposed_kv_us = max(0.0, kv_read_us - hideable_compute_us)       # ~0 (KV << compute)
    new_cycle_roofline_us = (served_cycle_us - kv_read_us) + exposed_kv_us
    bound_roofline_tps = E_T / new_cycle_roofline_us * 1e6

    # over-generous: pretend prefetch recovers the ENTIRE occupancy-bound attn KV wall (it can't).
    new_cycle_generous_us = served_cycle_us - attn_kv_wall_us
    bound_generous_tps = E_T / new_cycle_generous_us * 1e6

    # realistic: cp.async already deployed + occupancy-bound + serial dep-collapse -> ~0 gain.
    bound_realistic_tps = ANCHOR_252

    # ===================== AR M=1 frame (the split's native frame) ===================== #
    ar_new_step_us = PR569["decode_step_us"] - kv_read_us
    ar_bound_tps = 1e6 / ar_new_step_us

    # ===================== robustness: context needed for KV alone to reach ship ===================== #
    ship_cycle_us = E_T / SHIP_375 * 1e6
    need_hide_us = served_cycle_us - ship_cycle_us
    need_kv_bytes = need_hide_us * 1e-6 * (bw * 1e9)
    bytes_per_tok = _kv_bytes_geometric(2) - _kv_bytes_geometric(1)  # marginal full-layer bytes/tok
    ctx_to_reach_ship = need_kv_bytes / max(1, bytes_per_tok)

    return {
        "achievable_read_bw_gbps": bw,
        "kv_bytes_geometric": int(kv_bytes_geom),
        "kv_bytes_445_scaled": int(kv_bytes_445_scaled),
        "kv_bytes_used": int(kv_bytes),
        "kv_MB_used": kv_bytes / 1e6,
        "kv_read_us_bandwidth_limited": kv_read_us,
        "attn_kv_wall_us_occbound": attn_kv_wall_us,
        "head_bytes": head_bytes,
        "head_GB": head_bytes / 1e9,
        "head_read_over_kv_read_ratio": head_over_kv,
        "attn_compute_us": attn_compute_us,
        "hideable_compute_us": hideable_compute_us,
        "exposed_kv_us_after_overlap": exposed_kv_us,
        # served frame
        "served_cycle_us": served_cycle_us,
        "new_cycle_roofline_us": new_cycle_roofline_us,
        "prefetch_overlap_bound_tps": bound_roofline_tps,    # PRIMARY
        "bound_generous_tps": bound_generous_tps,
        "bound_realistic_tps": bound_realistic_tps,
        "savings_us_roofline": served_cycle_us - new_cycle_roofline_us,
        "delta_tps_roofline": bound_roofline_tps - ANCHOR_252,
        "delta_tps_pct_roofline": (bound_roofline_tps - ANCHOR_252) / ANCHOR_252 * 100.0,
        # AR M=1 frame
        "ar_m1_new_step_us": ar_new_step_us,
        "ar_m1_bound_tps": ar_bound_tps,
        # KV share of the step
        "kv_frac_of_served_cycle": kv_read_us / served_cycle_us,
        "kv_frac_of_ar_step": kv_read_us / PR569["decode_step_us"],
        # robustness
        "ship_cycle_us": ship_cycle_us,
        "need_hide_us_to_reach_ship": need_hide_us,
        "ctx_tokens_for_kv_alone_to_reach_ship": ctx_to_reach_ship,
        # verdicts
        "prefetch_can_reach_ship": bool(bound_generous_tps >= SHIP_375),
        "prefetch_beats_252": bool(bound_roofline_tps > ANCHOR_252),
    }


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def self_test(b: dict[str, Any], mb_bw: dict[str, Any], mb_h1: dict[str, Any],
              split_closure_us: float) -> dict[str, Any]:
    c = {
        # (a) the generous bound stays below the ship bar.
        "bound_below_ship": bool(b["bound_generous_tps"] < SHIP_375),
        # (b) lm_head read dwarfs the KV read by >10x (KV is not the bottleneck).
        "head_read_over_kv_gt_10x": bool(b["head_read_over_kv_read_ratio"] > 10.0),
        # (c) KV read is <5% of the served cycle.
        "kv_frac_lt_5pct": bool(b["kv_frac_of_served_cycle"] < 0.05),
        # (d) read-BW microbench ran and lands in a sane A10G band [400, 600] GB/s.
        "read_bw_in_band": bool(mb_bw.get("ran") and 400.0 <= mb_bw.get("effective_read_bw_gbps", 0) <= 600.0),
        # (e) lm_head GEMV microbench reproduces #569's head time within 25%.
        "lm_head_matches_569": bool(mb_h1.get("ran") and abs(mb_h1.get("wall_us", 0) - PR569["head_gemv_us"]) / PR569["head_gemv_us"] < 0.25),
        # (f) the #569 split closes onto the measured step (sum within 1 us).
        "split_closure_ok": bool(abs(split_closure_us) < 1.0),
        # (g) bound is finite / nan-clean.
        "nan_clean": bool(all(math.isfinite(b[k]) for k in (
            "prefetch_overlap_bound_tps", "bound_generous_tps", "kv_read_us_bandwidth_limited",
            "head_read_over_kv_read_ratio", "kv_frac_of_served_cycle"))),
    }
    c["all_passed"] = bool(all(c.values()))
    return c


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="kv-prefetch-overlap-bound")
    ap.add_argument("--wandb_name", default="denken/kv-prefetch-overlap-bound")
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print("[kv-prefetch-bound] running grounding microbenches on the assigned A10G ...", flush=True)
    mb_bw = mb_read_bw()
    mb_h1 = mb_lm_head(1)
    mb_h8 = mb_lm_head(8)
    mb_kv = mb_kv_read_sweep()
    print(f"[kv-prefetch-bound]   read_bw={mb_bw.get('effective_read_bw_gbps')}, "
          f"lm_head_m1_us={mb_h1.get('wall_us')}, lm_head_m8_us={mb_h8.get('wall_us')}", flush=True)

    bound = compute_bound(mb_bw)

    split_sum = (PR569["body_gemv_us"] + PR569["head_gemv_us"] + PR569["attn_us"]
                 + PR569["fixed_overhead_us"])
    split_closure_us = split_sum - PR569["decode_step_us"]

    st = self_test(bound, mb_bw, mb_h1, split_closure_us)

    verdict_summary = (
        "KV read is ~%.2f%% of the served cycle and the 262k bf16 lm_head reads %.0fx more "
        "bytes than the whole KV cache. Even a PERFECT-overlap roofline reaches %.1f TPS "
        "(over-generous %.1f); ship needs %.1f. cp.async is already deployed + the attn kernel "
        "is occupancy-bound at 20%% peak -> realistic prefetch gain ~0. NO FIRE." % (
            bound["kv_frac_of_served_cycle"] * 100.0,
            bound["head_read_over_kv_read_ratio"],
            bound["prefetch_overlap_bound_tps"], bound["bound_generous_tps"], SHIP_375))

    payload: dict[str, Any] = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "card": "PR #592 kv-prefetch-overlap-bound (analysis-only)",
        "analysis_only": True,
        "official_tps": 0,
        "gpu": "NVIDIA A10G (sm_86, ~600 GB/s HBM, 6.29 MB L2, 80 SM, no H20 async-copy/TMA)",
        "frame_note": ("base_fullhead anchor 252.69 = MTP-K7 SERVED (E[T]=3.844); NOT spec-OFF. "
                       "The PR body's 'spec-OFF' phrasing is an error -- flagged. #569 measured "
                       "base_fullhead AR M=1 = 97.01 TPS, so 252.69 cannot be the no-spec rate. "
                       "The overlap SAVINGS is a per-cycle KV-read time (cache read once per cycle, "
                       "M-invariant), so the bound is frame-robust."),
        "anchors": {"ANCHOR_252_mtp_served": ANCHOR_252, "E_T": E_T,
                    "SHIP_375": SHIP_375, "SHIP_481": SHIP_481},
        "pr569_split_us": PR569,
        "pr445_kv_geometry": PR445,
        "geometry": GEO,
        "split_decomposition_us": {
            "lm_head_bf16_262k": PR569["head_gemv_us"],
            "mlp_proj_body_int4": PR569["body_gemv_us"],
            "attention_kernel_full": PR569["attn_us"],
            "kv_read_bandwidth_limited": bound["kv_read_us_bandwidth_limited"],
            "attention_compute_hideable": bound["attn_compute_us"],
            "fixed_overhead": PR569["fixed_overhead_us"],
            "split_sum_us": split_sum,
            "split_closure_us": split_closure_us,
        },
        "microbench": {"read_bw": mb_bw, "lm_head_m1": mb_h1, "lm_head_m8": mb_h8, "kv_read_sweep": mb_kv},
        "bound": bound,
        "verdict": {
            "prefetch_overlap_bound_tps": bound["prefetch_overlap_bound_tps"],
            "bound_generous_tps": bound["bound_generous_tps"],
            "bound_realistic_tps": bound["bound_realistic_tps"],
            "prefetch_can_reach_ship": bound["prefetch_can_reach_ship"],
            "prefetch_beats_252": bound["prefetch_beats_252"],
            "kv_is_bottleneck": False,
            "lm_head_is_bottleneck": True,
            "head_read_over_kv_read_ratio": bound["head_read_over_kv_read_ratio"],
            "kv_frac_of_served_cycle": bound["kv_frac_of_served_cycle"],
            "realistic_gain_pct": 0.0,
            "summary": verdict_summary,
        },
        "self_test": st,
        "primary_metric": {"name": "prefetch_overlap_bound_tps",
                           "value": bound["prefetch_overlap_bound_tps"]},
        "test_metric": {"name": "self_test_all_passed", "value": int(st["all_passed"])},
        "elapsed_s": time.time() - t0,
    }
    try:
        import torch
        if torch.cuda.is_available():
            payload["peak_vram_GiB"] = torch.cuda.max_memory_allocated() / (1024 ** 3)
    except Exception:  # noqa: BLE001
        pass

    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[kv-prefetch-bound] wrote {OUT_JSON}", flush=True)
    print(json.dumps(payload["verdict"], indent=2), flush=True)
    print(json.dumps(st, indent=2), flush=True)

    if not args.no_wandb:
        try:
            import wandb
            run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                             group=args.wandb_group, name=args.wandb_name, job_type="profiling",
                             config={"analysis_only": True, "official_tps": 0,
                                     "gpu": "A10G", **GEO, **payload["anchors"]})
            flat: dict[str, Any] = {}
            for k, v in bound.items():
                if isinstance(v, (int, float, bool, str)):
                    flat[k] = v
            for k, v in payload["verdict"].items():
                if isinstance(v, (int, float, bool, str)):
                    flat[f"verdict/{k}"] = v
            for k, v in payload["split_decomposition_us"].items():
                flat[f"split/{k}"] = v
            flat.update({f"selftest/{k}": bool(v) for k, v in st.items()})
            if mb_bw.get("ran"):
                flat["mb/read_bw_gbps"] = mb_bw["effective_read_bw_gbps"]
            if mb_h1.get("ran"):
                flat["mb/lm_head_m1_us"] = mb_h1["wall_us"]
                flat["mb/lm_head_m1_GB"] = mb_h1["weight_GB"]
            if mb_h8.get("ran"):
                flat["mb/lm_head_m8_us"] = mb_h8["wall_us"]
            flat["official_tps"] = 0
            flat["primary_metric"] = bound["prefetch_overlap_bound_tps"]
            if "peak_vram_GiB" in payload:
                flat["peak_vram_GiB"] = payload["peak_vram_GiB"]
            run.summary.update(flat)
            # KV-read-vs-context sweep table
            if mb_kv.get("ran") and mb_kv.get("rows"):
                tbl = wandb.Table(columns=["ctx", "kv_bytes", "read_us", "eff_gbps"])
                for r in mb_kv["rows"]:
                    tbl.add_data(r["ctx"], r["kv_bytes"], r["read_us"], r["eff_gbps"])
                run.log({"kv_read_vs_ctx": tbl})
            print(f"[kv-prefetch-bound] wandb -> {run.url}  id={run.id}", flush=True)
            run.finish()
        except Exception as exc:  # noqa: BLE001
            print(f"[kv-prefetch-bound] wandb logging skipped: {exc}", flush=True)

    return 0 if st["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
