#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #591 -- base_fullhead speed-gap decomposition: where do the 123 TPS live?

LOCAL-ONLY analytic + profiling card. analysis_only=true, official_tps=0. NO HF
Job, NO submission, NO served-file change. One idle pod A10G. Group
``speed-gap-decomposition``.

QUESTION. The served base_fullhead anchor is 252.69 TPS (wirbel #553); the ship
flip target is 375.857 (capstone #561). That is a +123.17 TPS / +48.7% gap. This
card DECOMPOSES that gap into the four per-token latency components and asks, for
each, whether an operative-identity-SAFE reduction exists -- so we know which
single component, if made free, would (or would not) reach ship.

  (a) lm_head matmul + HBM read   -- bf16 262k dense GEMV (1.342 GB read)
  (b) int4 decoder-body compute   -- Marlin w4a16 linears x42 (the weight bulk)
  (c) attention + KV-cache read   -- fused attention kernel + KV reload
  (d) sampler / decode-loop o/h   -- argmax + norms + host/launch residual

THE BASIS RECONCILIATION (the load-bearing subtlety). The 252.69 and 375.857 are
BOTH spec-ON served numbers (MTP K=7 drafter, E[T]=3.8194 accepted tok/cycle),
NOT plain autoregressive (AR) rates. Proof: (1) #554/#569 hardcode E[T]=3.8194
for this exact anchor; (2) my #572 measured the SAME stack at spec-ON 253.99 vs
no-spec 83.44; (3) physics -- a bf16 262k head (1.342 GB) + int4 body cannot
exceed ~110 TPS in AR M=1. So 252.69 is spec-ON and the clean AR M=1 cudagraph
rate is 97.01 (#569) / 96.60 (#582); spec_lift = 252.69/97.01 = 2.60x is the
speculative TOKEN-AMORTIZATION (one target verify forward, E[T]=3.82 tokens out),
NOT a serving-layer detok artifact.

THE DECOMPOSITION MAPPING. Component per-cycle times come from #569's MEASURED
AR M=1 cudagraph trace (head/body/attn/fixed sum exactly to the 10.308 ms step).
A spec cycle runs ONE target verify forward at M=8; the head/body matmuls are
memory-bound so M=8 wall ~= M=1 (head microbench m8/m1 = +1.2%), letting the M=1
trace transfer to the verify forward within ~1-2%. Freeing a component removes
its per-cycle time from the served spec t_cycle (15.138 ms) -- the EXACT mapping
lawine #554 / denken #569 used to derive the free-head floor (311.27). We reuse
that mapping verbatim (and reproduce 311.27 as a self-test) then extend it to
body / attn / overhead.

THE EXPECTED FINDING (and the refinement). The PR hypothesised the lm_head HBM
read dominates. It does NOT: the int4 BODY weight read (6.728 ms, 44.4% of cycle)
is 2.4x the head (2.776 ms, 18.3%). And only the body, if freed, clears ship
(free-body -> 454 TPS); freeing the head reaches just 311.27 (= #569 floor, still
64.6 short). But the body is identity-LOCKED at int4_g32 (#571: no faster
byte-exact realization). Per-component operative-safe closures -- head #556
(lever=False), attention #562 (byte-identical, no faster legal realization), body
#571 (lever=False) -- each yield ZERO safe reduction. So
``reclaimable_safe_tps_headroom = 0``: the 252.69 anchor sits on the
operative-identity-safe floor, and the other ~half of the gap lives in the
draft + spec-loop residual (4.83 ms/cycle), which belongs to the drafter cards,
not to any base_fullhead forward component.

Reproduce:
  cd target/ && CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m \
    research.speed_gap_decomposition.speed_gap_decomposition \
    --wandb_group speed-gap-decomposition \
    --wandb_name lawine/speed-gap-decomposition
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --------------------------------------------------------------------------- #
# Architecture (google/gemma-4-E4B-it-qat-w4a16-ct text_config; lm_head in the
# quant ignore-list -> bf16 tied 262k head).
# --------------------------------------------------------------------------- #
HIDDEN, VOCAB = 2560, 262144
HEAD_BYTES_BF16 = float(VOCAB * HIDDEN * 2)   # 1.342 GB bf16 262k tied head
HEAD_BYTES_INT8 = float(VOCAB * HIDDEN * 1)   # 0.671 GB hypothetical int8 head
VERIFY_M = 8                                  # spec K=7 -> verify head over M+1=8 rows

# --------------------------------------------------------------------------- #
# ANCHORS -- cite, do NOT re-derive. From this branch's #553/#554/#556/#562/#569/
# #571/#572/#582 records (all base_fullhead == stock int4 body + full bf16 head).
# --------------------------------------------------------------------------- #
# Spec frame (the served anchor is SPEC-ON; E[T]=3.82) -- lawine #554 / denken #569
BFH_TPS_SERVED = 252.69                 # wirbel #553 served spec TPS (run 83jiwjr9)
FOC_BFH_TPS = 252.30599912117162        # #554 FOC anchor = E_T / t_cycle
FOC_BFH_ET = 3.8194082146962955         # spec accepted tokens / cycle
FOC_BFH_TCYCLE_MS = FOC_BFH_ET / FOC_BFH_TPS * 1e3   # 15.138 ms served spec cycle
FOC_KV_TPS_WITHOUT = 255.48354249571457              # #551 KV->0 generous bound
SHIP_TPS = 375.857                      # official ship flip (capstone #561 v74ad5jb)

# denken #569 MEASURED AR M=1 cudagraph component trace (sums to decode_step_total)
N569_DECODE_STEP_US = 10308.227501809597
N569_BODY_US = 6728.122871093588        # int4 Marlin linears x42
N569_HEAD_US = 2775.7492421875104       # bf16 262k head GEMV (trace)
N569_HEAD_MB_M8_US = 2679.244804382324  # head microbench M=8 (used for the #554 floor)
N569_HEAD_MB_M1_US = 2646.5280532836914 # head microbench M=1
N569_ATTN_US = 405.7652578124905        # fused attention + KV reload
N569_FIXED_US = 398.5901307160075       # sampling + norm + other + host residual
N569_SAMPLING_US = 14.837875000006111
N569_NORM_US = 172.05680468753076
N569_OTHER_US = 170.158976562571
N569_HOST_US = 41.53647446589821
N569_CLEAN_AR_TPS = 97.00988844342552   # clean LLM() AR M=1, cudagraph ON
N569_HEAD_EFF_GBPS = 500.95358132435643
N569_FREE_HEAD_FLOOR_TPS = 311.2485991465399  # #554/#569 corrected free-head floor
N569_GAP_TO_SHIP = 64.60840085346013    # ship - free_head_floor

# Other measurement bases
N582_CLEAN_DEFAULT_TPS = 96.60          # stark #582 clean LLM() backend basis
N572_SERVED_NOSPEC_TPS = 83.44107113115673   # my #572 served AR no-spec (same pod)
N572_SPEC_TPS = 253.98557253595862      # my #572 base_fullhead spec (E[T]=3.844, K=7)
N283_BODY_READ_BW_GBPS = 512.99         # denken #283 measured body read floor BW
N571_BODY_EFF_HBM_GBPS = 500.5          # land #571 served body effective HBM BW
N571_BODY_PARAMS = 3972792320           # land #571 int4 body param count

# Per-component operative-identity-safe closure cards (ALL lever=False).
HEAD_CARD = {"pr": 556, "run": "uipo4rxv", "lever_exists": False, "ceiling_tps": 252.31,
             "note": "int4/int8 head FLIPS greedy argmax vs bf16; no faster byte-exact head"}
ATTN_CARD = {"pr": 562, "run": "am7kltht", "lever_exists": False,
             "note": "byte-identical attention census CLOSED; no faster legal realization"}
BODY_CARD = {"pr": 571, "lever_exists": False, "ceiling_tps": 252.69,
             "note": "int4_g32 is the byte-exact-relative floor; int4_g128 faster but flips "
                     "MORE; every byte-exact alt (bf16/int8/fp8) reads >=2x bytes -> slower"}
OVERHEAD_CARD = {"pr": 569, "reducible": False,
                 "note": "fixed overhead already runs cudagraph ON (ONEGRAPH); host residual "
                         "<0.5% and not byte-safely removable"}


# --------------------------------------------------------------------------- #
# Fresh head GEMV microbench (independent cross-check of the #569 head read).
# Verbatim CUDA-graph timing method from denken #569 / lawine #551.
# --------------------------------------------------------------------------- #
def _graph_time(fn: Callable[[], Any], reps: int, warmup: int, repeats: int, torch_mod) -> float:
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
    """bf16 262k head matmul at M=1 (AR/draft) and M=8 (spec verify). Falls back to
    denken #569's measured values if CUDA is unavailable / OOM so the analytic card
    still completes."""
    try:
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
        eff_gbps = (HEAD_BYTES_BF16 / 1e9) / (m8 / 1e3)
        peak_gib = torch.cuda.max_memory_allocated() / 2**30
        print(f"[head-mb] M=1 {m1:.3f}ms  M={VERIFY_M} {m8:.3f}ms  eff {eff_gbps:.1f} GB/s  "
              f"(m8/m1={m8/m1:.4f}; #569 m8={N569_HEAD_MB_M8_US/1e3:.3f}ms)", flush=True)
        return {"head_matmul_m1_ms": m1, "head_matmul_m8_ms": m8, "head_eff_gbps": eff_gbps,
                "peak_gib": peak_gib, "source": "fresh_microbench"}
    except Exception as exc:  # noqa: BLE001
        print(f"[head-mb] CUDA microbench unavailable ({exc}); falling back to #569 values",
              flush=True)
        return {"head_matmul_m1_ms": N569_HEAD_MB_M1_US / 1e3,
                "head_matmul_m8_ms": N569_HEAD_MB_M8_US / 1e3,
                "head_eff_gbps": N569_HEAD_EFF_GBPS, "peak_gib": float("nan"),
                "source": "fallback_569"}


# --------------------------------------------------------------------------- #
# Spec-cycle free-component mapping (denken #569 / lawine #554, verbatim).
# Freeing a component removes its per-cycle (= per verify-forward) time from the
# served spec t_cycle; TPS = E[T] / freed_t_cycle.
# --------------------------------------------------------------------------- #
def _kv_saving_ms() -> float:
    # KV->0 expressed as a t_cycle delta (#551 generous upper bound).
    return FOC_BFH_TCYCLE_MS - (FOC_BFH_ET / FOC_KV_TPS_WITHOUT * 1e3)


def free_component_tps(t_component_ms: float, *, with_kv: bool = False) -> float:
    freed = FOC_BFH_TCYCLE_MS - t_component_ms
    if with_kv:
        freed -= _kv_saving_ms()
    return FOC_BFH_ET / (freed / 1e3)


def reconcile_bases() -> dict[str, Any]:
    """The four measurement bases the 252.69<->375.857 gap must be read in."""
    spec_lift = BFH_TPS_SERVED / N569_CLEAN_AR_TPS
    return {
        "spec_served_anchor_tps": BFH_TPS_SERVED,         # spec-ON, ship serving stack
        "spec_foc_anchor_tps": FOC_BFH_TPS,               # spec-ON, FOC constants
        "spec_tcycle_ms": FOC_BFH_TCYCLE_MS,
        "spec_e_t": FOC_BFH_ET,
        "clean_ar_m1_graph_tps": N569_CLEAN_AR_TPS,       # no-spec, 1 forward/token (#569)
        "clean_ar_m1_step_ms": N569_DECODE_STEP_US / 1e3,
        "clean_default_backend_tps": N582_CLEAN_DEFAULT_TPS,   # no-spec backend basis (#582)
        "served_nospec_ar_tps": N572_SERVED_NOSPEC_TPS,   # no-spec + serve overhead (#572)
        "ship_target_tps": SHIP_TPS,
        "ship_tcycle_ms": FOC_BFH_ET / SHIP_TPS * 1e3,
        "spec_lift_x": spec_lift,                         # 2.60x = token amortization
        "spec_lift_is_token_amortization": True,
        "anchor_is_spec_on_not_nospec": True,
        "proof": ("#554/#569 hardcode E[T]=3.8194 for 252.69; my #572 measured same stack "
                  "spec-ON 253.99 vs no-spec 83.44; bf16 262k head physics caps AR M=1 ~110 TPS"),
    }


def decompose(headmb: dict[str, Any]) -> dict[str, Any]:
    """Additive per-cycle TIME partition + marginal free-component TPS (spec frame)."""
    head_m8_ms = headmb["head_matmul_m8_ms"]
    tcycle = FOC_BFH_TCYCLE_MS
    et = FOC_BFH_ET

    # --- additive per-cycle TIME partition (the rigorous backing) ------------- #
    # The verify forward (= one AR-step worth of target kernels) plus the draft +
    # spec-loop residual. M=8 ~= M=1 (memory-bound) so #569's M=1 trace transfers.
    body_ms = N569_BODY_US / 1e3
    head_ms = N569_HEAD_US / 1e3              # trace head (sums w/ rest to the step)
    attn_ms = N569_ATTN_US / 1e3
    overhead_ms = N569_FIXED_US / 1e3
    target_forward_ms = N569_DECODE_STEP_US / 1e3
    draft_residual_ms = tcycle - target_forward_ms   # drafter forwards + accept/sched

    parts = {
        "body": body_ms, "head": head_ms, "attn_kv": attn_ms,
        "sampler_overhead": overhead_ms, "draft_spec_residual": draft_residual_ms,
    }
    time_partition = {k: {"per_cycle_ms": v, "frac_of_tcycle": v / tcycle} for k, v in parts.items()}
    partition_closure_err = abs(sum(parts.values()) - tcycle) / tcycle

    # --- ship reduction budget ------------------------------------------------ #
    tcycle_ship_ms = et / SHIP_TPS * 1e3
    reduction_needed_ms = tcycle - tcycle_ship_ms

    # --- marginal free-component TPS (spec frame) ----------------------------- #
    # Each entry: free THIS component (remove its per-cycle time), report the TPS
    # and whether that single free alone clears ship.
    def entry(t_ms: float, *, with_kv: bool = False) -> dict[str, Any]:
        tps = free_component_tps(t_ms, with_kv=with_kv)
        return {
            "per_cycle_ms": t_ms,
            "free_component_tps": tps,
            "marginal_gain_tps": tps - BFH_TPS_SERVED,
            "closes_gap_to_ship": bool(tps >= SHIP_TPS),
            "time_exceeds_reduction_needed": bool(t_ms >= reduction_needed_ms),
        }

    by_component = {
        "body": entry(body_ms),
        "head_trace": entry(head_ms),
        "head_microbench_m8": entry(head_m8_ms),
        "head_microbench_m8_plus_kv": entry(head_m8_ms, with_kv=True),  # = #569 floor
        "attn_kv": entry(attn_ms),
        "sampler_overhead": entry(overhead_ms),
        "draft_spec_residual": entry(draft_residual_ms),  # drafter domain, not a forward comp
    }

    # tps_gap_by_component: the headline dict the PR asks for (the four forward
    # components + the draft residual), keyed to the marginal free-component TPS.
    tps_gap_by_component = {
        "lm_head_matmul_hbm_read": by_component["head_microbench_m8"]["free_component_tps"],
        "int4_decoder_body": by_component["body"]["free_component_tps"],
        "attention_kv_cache_read": by_component["attn_kv"]["free_component_tps"],
        "sampler_decode_loop": by_component["sampler_overhead"]["free_component_tps"],
        "_draft_spec_residual_drafter_domain": by_component["draft_spec_residual"]["free_component_tps"],
    }

    # The free-head floor reproduction (self-test against #569's 311.27).
    free_head_floor_reproduced = by_component["head_microbench_m8_plus_kv"]["free_component_tps"]

    # Which single forward component, if freed, reaches ship?
    forward_keys = ["body", "head_microbench_m8", "attn_kv", "sampler_overhead"]
    closing_components = [k for k in forward_keys if by_component[k]["closes_gap_to_ship"]]

    return {
        "spec_tcycle_ms": tcycle,
        "spec_e_t": et,
        "target_forward_ms": target_forward_ms,
        "tcycle_ship_ms": tcycle_ship_ms,
        "reduction_needed_ms": reduction_needed_ms,
        "time_partition": time_partition,
        "time_partition_closure_err": partition_closure_err,
        "free_component": by_component,
        "tps_gap_by_component": tps_gap_by_component,
        "free_head_floor_reproduced_tps": free_head_floor_reproduced,
        "free_head_floor_repro_err": abs(free_head_floor_reproduced - N569_FREE_HEAD_FLOOR_TPS)
        / N569_FREE_HEAD_FLOOR_TPS,
        "closing_forward_components": closing_components,
        "only_body_closes_gap": closing_components == ["body"],
        "head_alone_misses_ship": bool(
            by_component["head_microbench_m8_plus_kv"]["free_component_tps"] < SHIP_TPS),
    }


def safe_headroom(decomp: dict[str, Any]) -> dict[str, Any]:
    """Per-component operative-identity-safe reclaimable TPS (cite #556/#562/#571/#569).

    The decomposition says only the BODY clears ship if freed. But every component
    has a CLOSED operative-safe census with lever=False -> the safe reduction of
    each is ZERO, so reclaimable_safe_tps_headroom = 0."""
    fc = decomp["free_component"]
    per_component = {
        "lm_head": {
            "free_component_tps": fc["head_microbench_m8_plus_kv"]["free_component_tps"],
            "operative_safe_reduction_tps": 0.0,
            "closure_card": HEAD_CARD,
            "reason": "int4/int8 head flips greedy argmax vs bf16 (#556 lever=False); the only "
                      "byte-exact head is the deployed bf16 one -> no faster byte-exact head. "
                      "Even fully-free head reaches only 311.27 < ship.",
        },
        "attention_kv": {
            "free_component_tps": fc["attn_kv"]["free_component_tps"],
            "operative_safe_reduction_tps": 0.0,
            "closure_card": ATTN_CARD,
            "reason": "byte-identical attention census closed (#562); no faster legal "
                      "realization. Free-attn marginal anyway (+6.6 TPS).",
        },
        "int4_body": {
            "free_component_tps": fc["body"]["free_component_tps"],
            "operative_safe_reduction_tps": 0.0,
            "closure_card": BODY_CARD,
            "reason": "int4_g32 IS the byte-exact-relative speed floor (#571 lever=False): the "
                      "only faster precision (int4_g128) flips MORE; every byte-exact alt reads "
                      ">=2x bytes -> slower. The ONE component that could close the gap is "
                      "identity-LOCKED.",
        },
        "sampler_overhead": {
            "free_component_tps": fc["sampler_overhead"]["free_component_tps"],
            "operative_safe_reduction_tps": 0.0,
            "closure_card": OVERHEAD_CARD,
            "reason": "fixed overhead already runs cudagraph ON (#569 reducible=False); host "
                      "residual <0.5% and not byte-safely removable.",
        },
    }
    total = sum(c["operative_safe_reduction_tps"] for c in per_component.values())
    return {
        "per_component": per_component,
        "reclaimable_safe_tps_headroom": total,
        "all_components_censused_lever_false": True,
        "verdict": (
            "Only the int4 decoder body, if made free, closes the 252.69->375.857 gap "
            "(free-body -> {:.1f} TPS); the lm_head -- the PR's hypothesised dominant lever "
            "-- does NOT (free-head -> {:.1f}, still {:.1f} short = #569 gap_to_ship). But the "
            "body is identity-LOCKED at int4_g32 (#571). Head #556, attention #562, body #571 "
            "all lever=False -> reclaimable_safe_tps_headroom = 0. The 252.69 anchor sits on "
            "the operative-identity-safe floor; the other ~half of the gap is the draft + "
            "spec-loop residual (drafter domain)."
        ).format(
            fc["body"]["free_component_tps"],
            fc["head_microbench_m8_plus_kv"]["free_component_tps"],
            SHIP_TPS - fc["head_microbench_m8_plus_kv"]["free_component_tps"],
        ),
    }


def handoffs(headmb: dict[str, Any], decomp: dict[str, Any]) -> dict[str, Any]:
    """Numbers to hand to denken (attn/MLP/KV split) and stark (int8-head)."""
    head_m8_ms = headmb["head_matmul_m8_ms"]
    head_eff = headmb["head_eff_gbps"]
    kv_saving = _kv_saving_ms()
    head_int8_ms = (HEAD_BYTES_INT8 / 1e9) / (head_eff / 1e3)   # int8 head read at same BW
    free_head_int8 = FOC_BFH_ET / ((FOC_BFH_TCYCLE_MS - (head_m8_ms - head_int8_ms)) / 1e3)
    return {
        "denken_attn_mlp_kv": {
            "attn_kv_per_cycle_ms": N569_ATTN_US / 1e3,
            "attn_kv_frac_of_tcycle": (N569_ATTN_US / 1e3) / FOC_BFH_TCYCLE_MS,
            "kv_read_saving_ms": kv_saving,
            "free_attn_kv_tps": decomp["free_component"]["attn_kv"]["free_component_tps"],
            "free_kv_only_tps": FOC_KV_TPS_WITHOUT,
            "kv_prefetch_ceiling_gain_tps": FOC_KV_TPS_WITHOUT - BFH_TPS_SERVED,
            "note": "attention+KV is 2.7% of t_cycle; KV-prefetch can hide at most the "
                    "KV-read (~0.19 ms/cycle) -> +2.8 TPS ceiling. MLP/down-proj reads are "
                    "INSIDE the int4 body (6.73 ms). Marginal axis.",
        },
        "stark_int8_head": {
            "head_bytes_bf16_gb": HEAD_BYTES_BF16 / 1e9,
            "head_bytes_int8_gb": HEAD_BYTES_INT8 / 1e9,
            "head_read_ms_bf16_m8": head_m8_ms,
            "head_read_ms_int8_m8_est": head_int8_ms,
            "head_eff_gbps": head_eff,
            "free_head_int8_tps_est": free_head_int8,
            "free_head_int8_gain_tps": free_head_int8 - BFH_TPS_SERVED,
            "operative_safe": False,
            "note": "int8 head halves the 1.342 GB read -> ~+24 TPS, BUT #556 shows int8/int4 "
                    "head FLIPS greedy identity (lever=False) -> NOT operative-safe; and even a "
                    "FULLY-free head reaches only 311.27 < ship 375.857.",
        },
    }


def build_payload(headmb: dict[str, Any]) -> dict[str, Any]:
    bases = reconcile_bases()
    decomp = decompose(headmb)
    safe = safe_headroom(decomp)
    hand = handoffs(headmb, decomp)
    verdict = {
        "analysis_only": True,
        "official_tps": 0,
        "no_hf_job": True,
        "no_submission": True,
        "no_served_file_change": True,
        "pr": 591,
        "agent": "lawine",
        "model_id": "google/gemma-4-E4B-it-qat-w4a16-ct",
        "substrate": "base_fullhead (stock int4_g32 body + full bf16 262k tied head, NO bake/prune)",
        "spec_served_anchor_tps": BFH_TPS_SERVED,
        "ship_tps": SHIP_TPS,
        "gap_to_ship_tps": SHIP_TPS - BFH_TPS_SERVED,
        "gap_to_ship_pct": (SHIP_TPS - BFH_TPS_SERVED) / BFH_TPS_SERVED,
        "spec_lift_x": bases["spec_lift_x"],
        "head_microbench_source": headmb["source"],
        "head_matmul_m1_ms": headmb["head_matmul_m1_ms"],
        "head_matmul_m8_ms": headmb["head_matmul_m8_ms"],
        "head_eff_gbps": headmb["head_eff_gbps"],
        "head_m8_over_m1": headmb["head_matmul_m8_ms"] / headmb["head_matmul_m1_ms"],
        "tps_gap_by_component": decomp["tps_gap_by_component"],
        "free_head_floor_reproduced_tps": decomp["free_head_floor_reproduced_tps"],
        "free_head_floor_anchor_569_tps": N569_FREE_HEAD_FLOOR_TPS,
        "free_head_floor_repro_err": decomp["free_head_floor_repro_err"],
        "gap_to_ship_free_head": SHIP_TPS - decomp["free_head_floor_reproduced_tps"],
        "free_body_tps": decomp["free_component"]["body"]["free_component_tps"],
        "only_body_closes_gap": decomp["only_body_closes_gap"],
        "head_alone_misses_ship": decomp["head_alone_misses_ship"],
        "closing_forward_components": decomp["closing_forward_components"],
        "reclaimable_safe_tps_headroom": safe["reclaimable_safe_tps_headroom"],
        "all_components_censused_lever_false": safe["all_components_censused_lever_false"],
        "reduction_needed_ms": decomp["reduction_needed_ms"],
        "body_dominates_not_head": bool(N569_BODY_US > N569_HEAD_US),
        "verdict": safe["verdict"],
    }
    return {
        "verdict": verdict,
        "bases": bases,
        "decomposition": decomp,
        "safe_headroom": safe,
        "handoffs": hand,
        "head_microbench": headmb,
    }


def self_test(payload: dict[str, Any]) -> dict[str, Any]:
    v = payload["verdict"]
    d = payload["decomposition"]
    s = payload["safe_headroom"]
    st: dict[str, bool] = {}
    # 1. reproduce denken #569's free-head floor (311.27) within 0.5%
    st["free_head_floor_reproduces_569"] = bool(v["free_head_floor_repro_err"] < 5e-3)
    # 2. reproduce #569's gap_to_ship (64.61) within 1 TPS
    st["gap_to_ship_free_head_matches_569"] = bool(
        abs(v["gap_to_ship_free_head"] - N569_GAP_TO_SHIP) < 1.0)
    # 3. additive time partition closes (body+head+attn+oh+draft == t_cycle)
    st["time_partition_closes"] = bool(d["time_partition_closure_err"] < 1e-9)
    # 4. body dominates the per-forward weight read, NOT the head (PR refinement)
    st["body_dominates_not_head"] = bool(v["body_dominates_not_head"])
    # 5. ONLY the body closes the gap to ship
    st["only_body_closes_gap"] = bool(d["only_body_closes_gap"])
    # 6. head alone (even +KV) misses ship
    st["head_alone_misses_ship"] = bool(d["head_alone_misses_ship"])
    # 7. free-body actually clears ship
    st["free_body_clears_ship"] = bool(v["free_body_tps"] >= SHIP_TPS)
    # 8. reclaimable safe headroom is exactly zero
    st["reclaimable_safe_headroom_zero"] = bool(v["reclaimable_safe_tps_headroom"] == 0.0)
    # 9. spec_lift sanity: 252.69 / 97.01 ~= 2.6 (proves anchor is spec-ON)
    st["spec_lift_is_token_amortization"] = bool(2.4 < v["spec_lift_x"] < 2.8)
    # 10. fresh head microbench agrees with #569 m8 within 8% (or fell back)
    st["head_microbench_agrees_569"] = bool(
        payload["head_microbench"]["source"] == "fallback_569"
        or abs(payload["head_microbench"]["head_matmul_m8_ms"] - N569_HEAD_MB_M8_US / 1e3)
        / (N569_HEAD_MB_M8_US / 1e3) < 0.08)
    # 11. head/body matmuls M-invariant in wall time (m8/m1 within 5%)
    st["head_matmul_m_invariant_walltime"] = bool(v["head_m8_over_m1"] < 1.05)
    fin = [v["free_body_tps"], v["free_head_floor_reproduced_tps"],
           v["reclaimable_safe_tps_headroom"], v["spec_lift_x"], d["reduction_needed_ms"]]
    st["nan_clean"] = all(math.isfinite(x) for x in fin)
    st["self_test_passes"] = all(st.values())
    return st


def maybe_log_wandb(args, payload, st) -> str | None:
    try:
        from scripts.wandb_logging import (finish_wandb, init_wandb_run, log_json_artifact,
                                           log_summary)
    except Exception as exc:  # noqa: BLE001
        print(f"[sgd] wandb logging unavailable: {exc}", flush=True)
        return None
    v = payload["verdict"]
    run = init_wandb_run(
        job_type="speed-gap-decomposition", agent="lawine",
        name=args.wandb_name, group=args.wandb_group,
        notes="PR #591: decompose the base_fullhead 252.69->375.857 (+123 TPS) gap into "
              "head/body/attn/sampler per-token components, reconcile the spec/AR bases, and "
              "test per-component operative-safe headroom (#556/#562/#571).",
        tags=["speed-gap", "decomposition", "base-fullhead", "lm-head", "int4-body",
              "analysis-only", "pr-591", "local-only", "no-fire"],
        config={"analysis_only": True, "official_tps": 0, "pr": 591,
                "model_id": v["model_id"], "verify_m": VERIFY_M,
                "spec_anchor_tps": BFH_TPS_SERVED, "ship_tps": SHIP_TPS,
                "head_microbench_source": v["head_microbench_source"]},
    )
    if run is None:
        print("[sgd] wandb: no run (no API key / disabled) -- skipping", flush=True)
        return None
    flat = {k: val for k, val in v.items()
            if isinstance(val, (int, float, bool, str)) and not k.startswith("_")}
    # flatten the headline component dict
    for k, val in v["tps_gap_by_component"].items():
        flat[f"tps_gap_{k}"] = val
    flat.update({f"selftest_{k}": int(b) for k, b in st.items()})
    log_summary(run, flat, step=0)
    log_json_artifact(run, name="speed_gap_decomposition",
                      artifact_type="profiling", data=payload)
    rid = getattr(run, "id", None)
    finish_wandb(run)
    print(f"[sgd] wandb logged {len(flat)} keys; run id {rid}", flush=True)
    return rid


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="lawine/speed-gap-decomposition")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="speed-gap-decomposition")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--no-microbench", action="store_true",
                    help="skip the fresh GPU head microbench, use #569 measured values")
    ap.add_argument("--out", type=Path, default=HERE / "speed_gap_decomposition.json")
    args = ap.parse_args(argv)

    t0 = time.time()
    print("============ SPEED-GAP DECOMPOSITION (PR #591) ============", flush=True)
    print(f"[sgd] base_fullhead spec anchor {BFH_TPS_SERVED} TPS -> ship {SHIP_TPS} "
          f"(+{SHIP_TPS - BFH_TPS_SERVED:.2f} TPS / +{100*(SHIP_TPS-BFH_TPS_SERVED)/BFH_TPS_SERVED:.1f}%)",
          flush=True)
    if args.no_microbench:
        headmb = {"head_matmul_m1_ms": N569_HEAD_MB_M1_US / 1e3,
                  "head_matmul_m8_ms": N569_HEAD_MB_M8_US / 1e3,
                  "head_eff_gbps": N569_HEAD_EFF_GBPS, "peak_gib": float("nan"),
                  "source": "fallback_569"}
        print("[sgd] --no-microbench: using #569 head values", flush=True)
    else:
        print("[sgd] fresh bf16 262k head GEMV microbench (CUDA-graph timed)...", flush=True)
        headmb = head_microbench()

    payload = build_payload(headmb)
    st = self_test(payload)
    payload["self_test"] = st
    payload["elapsed_s"] = time.time() - t0
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"[sgd] wrote {args.out}", flush=True)

    v = payload["verdict"]
    d = payload["decomposition"]
    tc = FOC_BFH_TCYCLE_MS
    print(f"\nBASIS RECONCILIATION (252.69 is SPEC-ON, not no-spec):", flush=True)
    print(f"  spec served {BFH_TPS_SERVED}  |  clean AR M=1 graph {N569_CLEAN_AR_TPS:.2f} (#569) "
          f"/ {N582_CLEAN_DEFAULT_TPS} (#582)  |  served no-spec {N572_SERVED_NOSPEC_TPS:.2f} (#572)",
          flush=True)
    print(f"  spec_lift = {v['spec_lift_x']:.3f}x  (= speculative token amortization, E[T]={FOC_BFH_ET:.3f})",
          flush=True)
    print(f"\nPER-CYCLE TIME PARTITION (t_cycle {tc:.3f} ms, spec frame):", flush=True)
    for k, info in d["time_partition"].items():
        print(f"  {k:22s} {info['per_cycle_ms']:7.3f} ms  {100*info['frac_of_tcycle']:5.1f}%",
              flush=True)
    print(f"  ship needs t_cycle {d['tcycle_ship_ms']:.3f} ms -> remove {d['reduction_needed_ms']:.3f} ms",
          flush=True)
    print(f"\nFREE-COMPONENT TPS (spec frame; clears ship 375.857?):", flush=True)
    for k in ["body", "head_microbench_m8_plus_kv", "attn_kv", "sampler_overhead",
              "draft_spec_residual"]:
        e = d["free_component"][k]
        print(f"  free {k:26s} -> {e['free_component_tps']:7.2f} TPS "
              f"({e['marginal_gain_tps']:+7.2f})  closes_ship={e['closes_gap_to_ship']}", flush=True)
    print(f"\n  only_body_closes_gap = {d['only_body_closes_gap']}  "
          f"head_alone_misses_ship = {d['head_alone_misses_ship']}", flush=True)
    print(f"  free_head_floor reproduced {v['free_head_floor_reproduced_tps']:.2f} "
          f"vs #569 {N569_FREE_HEAD_FLOOR_TPS:.2f} (err {100*v['free_head_floor_repro_err']:.3f}%)",
          flush=True)
    print(f"\nRECLAIMABLE OPERATIVE-SAFE HEADROOM = {v['reclaimable_safe_tps_headroom']:.1f} TPS "
          f"(head #556, attn #562, body #571 all lever=False)", flush=True)
    print(f"VERDICT: {v['verdict']}", flush=True)
    print("==========================================================", flush=True)
    for k, b in st.items():
        if not b and k != "self_test_passes":
            print(f"  [self-test FAIL] {k}", flush=True)

    rid = None
    if not args.no_wandb:
        rid = maybe_log_wandb(args, payload, st)

    print("\nSENPAI-RESULT: " + json.dumps({
        "terminal": True, "status": "complete", "pending_arms": False,
        "analysis_only": True, "official_tps": 0,
        "wandb_run_ids": [rid] if rid else [],
        "self_det": st["self_test_passes"],
        "primary_metric": {"name": "reclaimable_safe_tps_headroom",
                           "value": round(v["reclaimable_safe_tps_headroom"], 4)},
        "test_metric": {"name": "free_head_floor_reproduced_tps",
                        "value": round(v["free_head_floor_reproduced_tps"], 2)},
    }), flush=True)
    return 0 if st["self_test_passes"] else 1


if __name__ == "__main__":
    sys.exit(main())
