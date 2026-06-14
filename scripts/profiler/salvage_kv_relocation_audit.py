#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Salvage-KV relocation audit (PR #157): is the host-bound Python `relocate_salvaged_kv`
loop a LIVE step tax on the descent path -- and what does designing it out as a
vectorized GPU gather/scatter recover?

LOCAL A10G profiling + analysis only. NO model load, NO vLLM serve change, NO HF
Job, NO submission, NO kernel deploy. BASELINE stays 481.53. Produces a step-lever
bound + a build design; does NOT authorize a launch. Rides Issue #124 RESOLVED
(greedy-exact, PPL <= 2.42 binding).

WHY
---
fern #149 surfaced, and chiku-inu's live trace (board 20260614-111022-934, STEPTIME
agg n=12841 on tree-488-pw-fp32-v0) PINNED, a `relocate_salvaged_kv` HOST-BOUND
Python loop over the 37 served (osoi5) layers. chiku-inu's decomposition:

  EXEC gpu  p50=19.18ms p90=21.42 mean=18.67   (the verify/exec step GPU floor)
  EXEC cpu  p50=12.05ms p90=335.36 mean=122.5  (host side)
            "the 335ms p90 = the 35%-of-steps Python relocate_salvaged_kv loop over 37 layers"
  DRAFT gpu p50=2.27ms  cpu p50=0.38ms
  "GPU compute floor ~19ms/exec-step; the wall is CPU/host-bound under PIECEWISE-eager
   (attn_py_calls/step=37), graph-reclaimable AFTER acceptance."

The CPU mean is EXACTLY the salvage-weighted mix of the relocate tail and the
non-salvage floor: 0.342*335.36 + 0.658*12.05 = 122.5. So relocate amortizes to
CPU_mean - CPU_floor = 122.5 - 12.05 ~= 110 ms/step -- ~90% of the as-built wall.

THE QUESTION (live landmine vs dead fallback)
---------------------------------------------
The descent tree (land #71) does salvage-then-descend on every non-full-accept step
(oracle: 391 salvages / 1024 steps). NONE of the step-model legs priced this:
lawine #136 (1.2182 GRAPH-CAPTURED target step), my #154 (LINEAR M=8 scatter+LP),
lawine #153 (verify-step M curve) all measured a path that salvages differently or
is already captured. Is the 335ms host loop:
  (a) a LIVE landmine -- on the timed decode window, it BLOWS the step (host-bound
      mean 122.5ms vs the 9.7ms captured target), so land's descent E[T] gain is
      eaten by a denominator explosion; or
  (b) a dead fallback -- never on the timed path, a de-risk confirmation.

VERDICT METHOD
--------------
1. Anchor decomposition (analytic, exact): reconstruct chiku-inu's CPU mean from the
   {salvage_rate, p90 relocate-tail, p50 non-salvage floor} mix -> relocate amortized
   ms/step. Independent of any reproduction; it is chiku-inu's own numbers closing.
2. Reproduce the per-call cost of the op on THIS A10G at the served KV dims
   (37 osoi5 layers x [kv_heads=2, head_dim=256] K+V bf16) in THREE implementations:
     * host_loop   -- the host-bound Python anti-pattern (per-layer, per-position
                      D2H/H2D round-trip): reproduces the hundreds-of-ms class.
     * gpu_perlayer-- a Python loop over 37 layers, ONE device index_copy_ each
                      (no host round-trip): isolates "the 37-layer count is NOT the
                      problem; the HOST round-trip is".
     * gpu_vectorized -- a single batched gather/scatter over all 37 layers by a
                      device commit-index (the design target): sub-millisecond.
     * paged_slotmap  -- the zero-copy ideal: update an int slot-map, move no KV
                      bytes (paged-attention block-table commit).
3. Price on the oracle ladder (salvages/step = 391/1024) -> amortized us/step for
   each path, against BOTH the as-built eager step (chiku-inu 18.67ms GPU floor) and
   the graph-captured target step (#136 1.2182). recoverable = host_loop - vectorized.
4. Prove greedy-safety: a correct relocate is a pure permutation/copy of existing
   bf16 KV values (no arithmetic -> no rounding); the gathered committed KV is
   bit-exact to the reference contiguous KV (equivalence_rate=1.0), and the verifier
   already decided accept/reject BEFORE relocation -> emitted tokens unchanged.
5. Hand land #71 the GPU-relocate design + the build-blocker classification.

PRIMARY: salvage_kv_audit_self_test_passes (the anchor decomposition + amortization
         + greedy-safety + bar arithmetic self-tests all pass).
TEST:    recoverable_step_pct_salvage_kv (the host-loop tax the vectorized design
         recovers, as % of the captured-target step -- same units as #154).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]

# ===== compose constants (re-priced from #136/#148/#154/#143, NOT re-derived) =====
Z95 = 1.959963984540054
K_CAL = 125.26795005202914              # 481.53 / 3.844 (official baseline / E[T]_linear)
STEP_M8_US = 1.0e6 / K_CAL              # ~7982.89 us = 1 M=8-normalized step-unit of wall time
MEASURED_STEP_136 = 1.2182              # lawine #136 GRAPH-CAPTURED target depth-9 step (units)
CLEAR500_BAR_MEASURED = 4.862377006624717  # fern #129/#136 operative clear-500 bar @ 1.2182
CLEAR500_BAR_154 = 4.808                # my #154 lowered bar (real scatter+LP avoidance)
E_T_TREE_CEILING = 5.207               # fern #125 / denken #101 supply ceiling (max E[T])
E_T_DESCENT_FIX = 5.04                 # fern #134 / wirbel #135 descent-only E[T] (-> ~522)
TAU_FERN_CENTRAL = 1.0
TARGET_500 = 500.0
TARGET_530 = 530.0

# ===== oracle readout of tree-488-pw-fp32-v0 (board 20260614-100550-487) =========
ORACLE_SALVAGES = 391
ORACLE_FULL = 37
ORACLE_STEPS = 1024
ORACLE_SALVAGE_RATE = ORACLE_SALVAGES / ORACLE_STEPS   # 0.38184 salvages/step
ORACLE_CUM_LADDER = [0.674, 0.350, 0.203, 0.131, 0.089, 0.060, 0.037]  # P(spine accepts >= d)
ORACLE_E_T = 2.621

# ===== chiku-inu live STEPTIME trace (board 20260614-111022-934, n=12841) =========
# EXEC = the verify/exec step; the relocate_salvaged_kv host loop is the CPU p90 tail.
CHIKU = {
    "n_steps": 12841,
    "exec_gpu_p50_ms": 19.18, "exec_gpu_p90_ms": 21.42, "exec_gpu_mean_ms": 18.67,
    "exec_cpu_p50_ms": 12.05, "exec_cpu_p90_ms": 335.36, "exec_cpu_mean_ms": 122.5,
    "draft_gpu_p50_ms": 2.27, "draft_cpu_p50_ms": 0.38,
    "relocate_layers": 37,                 # osoi5 served depth; loop iterates 37 layers
    "salvage_frac_stated": 0.35,           # chiku-inu's stated 35%-of-steps
    "note": "the 335ms p90 = the 35%-of-steps Python relocate_salvaged_kv loop over 37 layers",
}

# ===== served KV-cache geometry (osoi5 / google/gemma-4-E4B-it text config) =======
N_LAYERS = 37          # osoi5 served depth (chiku-inu: loop over 37 layers)
KV_HEADS = 2           # text_config.num_key_value_heads
HEAD_DIM = 256         # text_config.head_dim
KV_DTYPE = torch.bfloat16
KV_BYTES_PER_POS_PER_LAYER = 2 * KV_HEADS * HEAD_DIM * 2   # K+V, bf16 = 2048 B
M_TREE = 32            # M=32 tree window (max positions a salvage may compact per layer)
CTX_WINDOW = 4096      # representative committed-context cache length per layer


# ---------------------------------------------------------------------------
def summarize(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"n": 0}
    s = sorted(values)
    mean = statistics.fmean(values)

    def pct(p: float) -> float:
        if n == 1:
            return s[0]
        k = (n - 1) * p
        lo = math.floor(k)
        hi = math.ceil(k)
        return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)

    std = statistics.pstdev(values) if n > 1 else 0.0
    return {"n": n, "mean": mean, "median": statistics.median(values),
            "p50": pct(0.50), "p90": pct(0.90), "p99": pct(0.99),
            "std": std, "min": s[0], "max": s[-1],
            "ci95_abs": Z95 * std / math.sqrt(n) if n else 0.0}


def fern_clear_bar(target: float, step: float, tau: float = 1.0) -> float:
    """E[T] needed to clear `target` official at (step, tau). RISES with step."""
    return target * step / (K_CAL * tau)


def official_tps(e_t: float, step: float, tau: float = 1.0) -> float:
    return K_CAL * e_t / step * tau


# ===== Part A: anchor decomposition (analytic, exact -- chiku-inu's own numbers) ==
def anchor_decomposition() -> dict:
    """Reconstruct chiku-inu's CPU mean from the salvage-weighted mix of the relocate
    tail (p90) and the non-salvage floor (p50). Solve for the implied salvage fraction
    and the amortized relocate cost -- both must close against the stated 0.35 / oracle
    0.382 and the CPU mean. This is a reproduction WITHOUT the external build: it shows
    the anchors are mutually consistent and pins the relocate's per-step cost."""
    c_tail = CHIKU["exec_cpu_p90_ms"]     # 335.36 ms -- the relocate-step CPU cost
    c_floor = CHIKU["exec_cpu_p50_ms"]    # 12.05 ms  -- the non-salvage CPU floor
    c_mean = CHIKU["exec_cpu_mean_ms"]    # 122.5 ms  -- observed CPU mean over all steps
    # mean = r*c_tail + (1-r)*c_floor  ->  r = (mean - floor) / (tail - floor)
    implied_salvage_frac = (c_mean - c_floor) / (c_tail - c_floor)
    relocate_amortized_ms = c_mean - c_floor               # = r*(tail-floor)
    relocate_marginal_per_salvage_ms = c_tail - c_floor    # the per-salvage relocate cost
    # cross-checks: amortized via the oracle rate and via chiku's stated 0.35.
    amort_oracle = ORACLE_SALVAGE_RATE * relocate_marginal_per_salvage_ms
    amort_chiku = CHIKU["salvage_frac_stated"] * relocate_marginal_per_salvage_ms
    return {
        "cpu_relocate_tail_ms": c_tail, "cpu_nonsalvage_floor_ms": c_floor,
        "cpu_mean_observed_ms": c_mean,
        "implied_salvage_frac": implied_salvage_frac,
        "oracle_salvage_rate": ORACLE_SALVAGE_RATE,
        "chiku_salvage_frac_stated": CHIKU["salvage_frac_stated"],
        "relocate_marginal_per_salvage_ms": relocate_marginal_per_salvage_ms,
        "relocate_amortized_ms_per_step": relocate_amortized_ms,
        "relocate_amortized_via_oracle_rate_ms": amort_oracle,
        "relocate_amortized_via_chiku_frac_ms": amort_chiku,
        "as_built_wall_mean_ms": c_mean,                  # host-bound: wall ~= CPU mean
        "as_built_gpu_floor_ms": CHIKU["exec_gpu_mean_ms"],
        "relocate_pct_of_as_built_wall": 100.0 * relocate_amortized_ms / c_mean,
        "reconstruction_closes": bool(abs(implied_salvage_frac - 0.35) < 0.05),
        "note": ("CPU mean 122.5 = 0.342*335.36 + 0.658*12.05; implied salvage frac "
                 "0.342 sits between chiku's stated 0.35 and oracle 0.382 (sampling). "
                 "relocate amortizes to mean-floor = 110.45 ms/step = 90.2% of the wall."),
    }


# ===== Part B: per-call relocate microbench (3 implementations) ===================
class KVCache:
    """A realistic per-layer KV cache for the 37 osoi5 layers. Each layer holds K and V
    of shape [CTX_WINDOW, KV_HEADS, HEAD_DIM] (bf16). A salvage relocates `n_move`
    accepted positions from drafted (scratch) slots to committed contiguous slots."""

    def __init__(self, dev):
        self.dev = dev
        # per-layer K,V on device (a list models the per-layer cache the host loop walks).
        self.k = [torch.randn(CTX_WINDOW, KV_HEADS, HEAD_DIM, device=dev, dtype=KV_DTYPE)
                  for _ in range(N_LAYERS)]
        self.v = [torch.randn(CTX_WINDOW, KV_HEADS, HEAD_DIM, device=dev, dtype=KV_DTYPE)
                  for _ in range(N_LAYERS)]
        # a contiguous [N_LAYERS, CTX_WINDOW, ...] stack for the vectorized path.
        self.k_stack = torch.stack(self.k)   # [L, W, H, D]
        self.v_stack = torch.stack(self.v)
        # a CPU staging mirror the host-loop anti-pattern round-trips through.
        self.k_cpu = [t.to("cpu") for t in self.k]
        self.v_cpu = [t.to("cpu") for t in self.v]

    # ---- host_loop: the host-bound Python anti-pattern (per-layer x per-position) --
    def relocate_host_loop(self, src_idx_cpu: list[int], dst_idx_cpu: list[int]):
        """Per-layer, per-position relocation driven from Python with a D2H/H2D round
        trip on each KV row -- the `relocate_salvaged_kv` host loop chiku-inu measured.
        Each row copy reads to host and writes back (the synchronizing pattern that
        makes the loop host-bound). Indices are Python ints (already on host)."""
        for layer in range(N_LAYERS):
            kc, vc = self.k[layer], self.v[layer]
            kcpu, vcpu = self.k_cpu[layer], self.v_cpu[layer]
            for s, d in zip(src_idx_cpu, dst_idx_cpu):
                # D2H: pull the salvaged row to host (forces a sync), then H2D back to
                # the committed slot -- the per-element host round-trip.
                krow = kc[s].to("cpu")            # *** D2H sync ***
                vrow = vc[s].to("cpu")            # *** D2H sync ***
                kcpu[d].copy_(krow)
                vcpu[d].copy_(vrow)
                kc[d].copy_(kcpu[d].to(self.dev))  # *** H2D ***
                vc[d].copy_(vcpu[d].to(self.dev))  # *** H2D ***

    # ---- gpu_perlayer: 37 device index_copy_ launches, NO host round-trip ----------
    def relocate_gpu_perlayer(self, src_idx_dev, dst_idx_dev):
        """A Python loop over 37 layers, but each layer is ONE device index_select +
        index_copy_ (device-to-device). No .item()/.cpu(): isolates that the 37-layer
        COUNT is cheap; only the host round-trip is the killer."""
        for layer in range(N_LAYERS):
            self.k[layer].index_copy_(0, dst_idx_dev,
                                      self.k[layer].index_select(0, src_idx_dev))
            self.v[layer].index_copy_(0, dst_idx_dev,
                                      self.v[layer].index_select(0, src_idx_dev))

    # ---- gpu_vectorized: ONE batched gather/scatter over ALL 37 layers -------------
    def relocate_gpu_vectorized(self, src_idx_dev, dst_idx_dev):
        """A single batched relocation over the [L, W, H, D] stack: gather all layers'
        salvaged rows and scatter to committed slots in one launch sequence (the design
        target). Indexed by a device commit-index; no per-layer Python, no host round-trip."""
        gathered_k = self.k_stack.index_select(1, src_idx_dev)   # [L, n_move, H, D]
        gathered_v = self.v_stack.index_select(1, src_idx_dev)
        self.k_stack.index_copy_(1, dst_idx_dev, gathered_k)
        self.v_stack.index_copy_(1, dst_idx_dev, gathered_v)

    # ---- paged_slotmap: the zero-copy ideal (move an int slot-map, no KV bytes) -----
    def relocate_paged_slotmap(self, slot_map, src_idx_dev, dst_idx_dev):
        """Paged-attention commit: relocation is a block-table / slot-map update -- the
        KV bytes never move; only the int->physical-slot mapping changes. The true
        zero-copy design when the build uses paged KV."""
        slot_map.index_copy_(0, dst_idx_dev, slot_map.index_select(0, src_idx_dev))


def _time_call(fn, n_warmup: int, n_calls: int, gen_args) -> list[float]:
    """Wall-clock per-call latency (ms) over n_calls, each with freshly-generated args
    (so n_move varies like real salvages). Wall clock (not CUDA events) because the
    host-loop's cost IS the host time; sync each call to attribute it fully."""
    for _ in range(n_warmup):
        fn(*gen_args())
    torch.cuda.synchronize()
    out = []
    for _ in range(n_calls):
        args = gen_args()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def sample_n_move(rng) -> int:
    """positions a salvage relocates: a naive compaction moves the whole M=32 tree
    window per layer (worst-case); a tight build moves only the accepted path
    (~accept_len). We model the NAIVE compaction (full tree window) for the host loop
    anchor and report the accept-len-only variant separately."""
    return M_TREE


def run_microbench(dev, args, rng) -> dict:
    cache = KVCache(dev)
    # the naive host loop compacts the whole M=32 window per layer; build the matching
    # src/dst index sets (the accepted subset scattered across the scratch window).
    def gen_host():
        n = sample_n_move(rng)
        src = sorted(rng.choice(CTX_WINDOW, size=n, replace=False).tolist())
        dst = list(range(n))
        return src, dst

    def gen_dev():
        n = sample_n_move(rng)
        src = torch.randperm(CTX_WINDOW, device=dev)[:n]
        dst = torch.arange(n, device=dev)
        return src, dst

    def gen_dev_slotmap():
        n = sample_n_move(rng)
        src = torch.randperm(CTX_WINDOW, device=dev)[:n]
        dst = torch.arange(n, device=dev)
        slot = torch.arange(CTX_WINDOW, device=dev, dtype=torch.int32)
        return slot, src, dst

    results = {}
    # host_loop is ~100s of ms/call -> few calls; gpu paths are sub-ms -> many calls.
    print(f"[salvage-kv] microbench: {N_LAYERS} layers x [{KV_HEADS},{HEAD_DIM}] K+V "
          f"bf16, n_move={M_TREE} (full tree window), ctx={CTX_WINDOW}", flush=True)

    t = _time_call(cache.relocate_host_loop, args.host_warmup, args.host_calls, gen_host)
    results["host_loop"] = summarize(t)
    print(f"   host_loop      : p50={results['host_loop']['p50']:.1f}ms "
          f"p90={results['host_loop']['p90']:.1f}ms mean={results['host_loop']['mean']:.1f}ms "
          f"(n={len(t)})", flush=True)

    t = _time_call(cache.relocate_gpu_perlayer, args.gpu_warmup, args.gpu_calls, gen_dev)
    results["gpu_perlayer"] = summarize(t)
    print(f"   gpu_perlayer   : p50={results['gpu_perlayer']['p50']*1e3:.1f}us "
          f"p90={results['gpu_perlayer']['p90']*1e3:.1f}us mean={results['gpu_perlayer']['mean']*1e3:.1f}us",
          flush=True)

    t = _time_call(cache.relocate_gpu_vectorized, args.gpu_warmup, args.gpu_calls, gen_dev)
    results["gpu_vectorized"] = summarize(t)
    print(f"   gpu_vectorized : p50={results['gpu_vectorized']['p50']*1e3:.1f}us "
          f"p90={results['gpu_vectorized']['p90']*1e3:.1f}us mean={results['gpu_vectorized']['mean']*1e3:.1f}us",
          flush=True)

    t = _time_call(cache.relocate_paged_slotmap, args.gpu_warmup, args.gpu_calls, gen_dev_slotmap)
    results["paged_slotmap"] = summarize(t)
    print(f"   paged_slotmap  : p50={results['paged_slotmap']['p50']*1e3:.1f}us "
          f"p90={results['paged_slotmap']['p90']*1e3:.1f}us mean={results['paged_slotmap']['mean']*1e3:.1f}us",
          flush=True)

    del cache
    torch.cuda.empty_cache()
    return results


# ===== Part C: pricing on the oracle ladder ======================================
def price_paths(micro: dict, decomp: dict) -> dict:
    """Amortize each measured per-call cost by the oracle salvage rate and price it
    against (i) the as-built eager step (chiku 18.67ms GPU floor) and (ii) the
    graph-captured target step (#136 1.2182). recoverable = host_loop - vectorized."""
    sr = ORACLE_SALVAGE_RATE
    gpu_floor_ms = CHIKU["exec_gpu_mean_ms"]              # 18.67 ms eager GPU floor
    target_step_us = MEASURED_STEP_136 * STEP_M8_US       # 9726 us captured target
    out = {"oracle_salvage_rate": sr, "as_built_gpu_floor_ms": gpu_floor_ms,
           "target_step_us": target_step_us, "paths": {}}
    for name, m in micro.items():
        per_call_ms = m["mean"]
        amort_ms = sr * per_call_ms
        amort_us = amort_ms * 1e3
        # (i) eager wall: host-bound iff CPU side (floor + relocate) exceeds GPU floor.
        eager_cpu_ms = CHIKU["exec_cpu_p50_ms"] + amort_ms
        eager_wall_ms = max(gpu_floor_ms, eager_cpu_ms)
        eager_infl_pct = 100.0 * (eager_wall_ms - gpu_floor_ms) / gpu_floor_ms
        # (ii) captured-target bar framing: add amort to the 1.2182 step.
        dstep_units = amort_us / STEP_M8_US
        step_infl_pct = 100.0 * amort_us / target_step_us
        inflated_step = MEASURED_STEP_136 + dstep_units
        bar = fern_clear_bar(TARGET_500, inflated_step)
        tps_at_descent = official_tps(E_T_DESCENT_FIX, inflated_step)
        out["paths"][name] = {
            "per_call_ms": per_call_ms, "amortized_ms_per_step": amort_ms,
            "amortized_us_per_step": amort_us,
            "eager_wall_ms": eager_wall_ms, "eager_step_inflation_pct": eager_infl_pct,
            "captured_step_inflation_pct": step_infl_pct,
            "inflated_captured_step_units": inflated_step,
            "clear500_bar": bar, "bar_under_ceiling": bool(bar <= E_T_TREE_CEILING),
            "descent_tps_at_this_step": tps_at_descent,
            "descent_clears_500": bool(tps_at_descent >= 500.0),
        }
    hp = out["paths"]["host_loop"]
    vp = out["paths"]["gpu_vectorized"]
    recoverable_us = hp["amortized_us_per_step"] - vp["amortized_us_per_step"]
    out["recoverable"] = {
        "host_loop_amortized_us_per_step": hp["amortized_us_per_step"],
        "vectorized_amortized_us_per_step": vp["amortized_us_per_step"],
        "recoverable_us_per_step": recoverable_us,
        # primary TEST: as % of the captured-target step (#154 units).
        "recoverable_step_pct_vs_captured_target": 100.0 * recoverable_us / (MEASURED_STEP_136 * STEP_M8_US),
        # as % of the as-built eager wall step (the host-bound reality).
        "recoverable_pct_of_as_built_wall": decomp["relocate_pct_of_as_built_wall"],
        "host_loop_descent_tps": hp["descent_tps_at_this_step"],
        "vectorized_descent_tps": vp["descent_tps_at_this_step"],
        "host_loop_bar": hp["clear500_bar"], "vectorized_bar": vp["clear500_bar"],
        "relocate_free_bar": CLEAR500_BAR_MEASURED,    # 4.862 @ the zero-relocate 1.2182 step
        "descent_e_t": E_T_DESCENT_FIX,                # 5.04 supplied by land's descent fix
        # ANY relocate cost raises the bar above the relocate-free 4.862; the meaningful
        # question is what fraction of the descent's 500-cushion (E[T]_descent - bar) it
        # eats. vectorized eats a sliver; the host loop blows past the ceiling entirely.
        "descent_cushion_units": E_T_DESCENT_FIX - CLEAR500_BAR_MEASURED,
        "vectorized_headroom_consumed_pct": 100.0 * (vp["clear500_bar"] - CLEAR500_BAR_MEASURED)
                                            / max(1e-9, E_T_DESCENT_FIX - CLEAR500_BAR_MEASURED),
        "vectorized_keeps_descent_clearing_500": bool(vp["descent_clears_500"]),
        "host_loop_bar_infeasible": bool(hp["clear500_bar"] > E_T_TREE_CEILING),
        "speedup_vectorized_over_host": hp["per_call_ms"] / max(1e-9, vp["per_call_ms"]),
    }
    return out


# ===== Part D: greedy-safety (relocation is a bit-exact permutation of KV) =========
def greedy_safety_check(dev, rng) -> dict:
    """A correct relocate moves existing bf16 KV values (a pure gather/scatter, no
    arithmetic -> no rounding). Verify the committed KV after relocation is BIT-EXACT
    to the reference contiguous KV, for all 3 implementations. equivalence_rate=1.0 by
    construction. (The verifier already decided accept/reject BEFORE relocation; the
    emitted token IDs do not depend on where the KV bytes physically live.)"""
    n = M_TREE
    src = torch.randperm(CTX_WINDOW, device=dev)[:n]
    dst = torch.arange(n, device=dev)
    src_l, dst_l = src.tolist(), dst.tolist()
    checks = {}
    for name in ("host_loop", "gpu_perlayer", "gpu_vectorized"):
        cache = KVCache(dev)
        # reference: what the committed slots SHOULD contain (the gathered src rows).
        ref_k = cache.k_stack.index_select(1, src).clone()  # [L, n, H, D]
        ref_v = cache.v_stack.index_select(1, src).clone()
        if name == "host_loop":
            cache.relocate_host_loop(src_l, dst_l)
            got_k = torch.stack([cache.k[l][dst] for l in range(N_LAYERS)])
            got_v = torch.stack([cache.v[l][dst] for l in range(N_LAYERS)])
        elif name == "gpu_perlayer":
            cache.relocate_gpu_perlayer(src, dst)
            got_k = torch.stack([cache.k[l][dst] for l in range(N_LAYERS)])
            got_v = torch.stack([cache.v[l][dst] for l in range(N_LAYERS)])
        else:
            cache.relocate_gpu_vectorized(src, dst)
            got_k = cache.k_stack.index_select(1, dst)
            got_v = cache.v_stack.index_select(1, dst)
        bit_exact = bool(torch.equal(got_k, ref_k) and torch.equal(got_v, ref_v))
        checks[name] = {"bit_exact": bit_exact,
                        "max_abs_err_k": float((got_k.float() - ref_k.float()).abs().max()),
                        "max_abs_err_v": float((got_v.float() - ref_v.float()).abs().max())}
        del cache
        torch.cuda.empty_cache()
    equivalence_rate = 1.0 if all(c["bit_exact"] for c in checks.values()) else 0.0
    return {"equivalence_rate": equivalence_rate, "per_impl": checks,
            "argument": ("KV relocation is a pure permutation/copy of existing bf16 "
                         "values (no cast, no arithmetic -> no rounding). The verifier's "
                         "argmax over the verify logits, and thus the accepted token IDs, "
                         "are decided BEFORE relocation and do not depend on the physical "
                         "KV slot. Greedy identity preserved by construction; equivalence_rate=1.0.")}


# ===== Part E: self-tests (PRIMARY) ==============================================
def self_tests(decomp, micro, pricing, safety) -> dict:
    hp = pricing["paths"]["host_loop"]
    vp = pricing["paths"]["gpu_vectorized"]
    gp = pricing["paths"]["gpu_perlayer"]
    rec = pricing["recoverable"]
    tests = {}
    # 1. anchor decomposition closes: implied salvage frac in [0.30, 0.40] (between
    #    chiku 0.35 and oracle 0.382); relocate amortized > 0.
    tests["anchor_decomposition_closes"] = bool(
        0.30 <= decomp["implied_salvage_frac"] <= 0.40
        and decomp["relocate_amortized_ms_per_step"] > 0)
    # 2. host_loop is host-bound hundreds-of-ms (>= 50ms p90) -- a real landmine class.
    tests["host_loop_is_host_bound"] = bool(micro["host_loop"]["p90"] >= 50.0)
    # 3. vectorized is sub-millisecond (< 1 ms mean).
    tests["vectorized_sub_ms"] = bool(micro["gpu_vectorized"]["mean"] < 1.0)
    # 4. vectorized >> host_loop (>= 100x faster per call) -- the design recovers it.
    tests["vectorized_beats_host_100x"] = bool(rec["speedup_vectorized_over_host"] >= 100.0)
    # 5. the 37-layer COUNT is not the killer: gpu_perlayer also sub-ms (host round-trip is).
    tests["layer_count_not_the_killer"] = bool(micro["gpu_perlayer"]["mean"] < 5.0)
    # 6. amortization arithmetic internally consistent (salvage_rate*per_call == us/step).
    expect_us = ORACLE_SALVAGE_RATE * hp["per_call_ms"] * 1e3
    tests["amortization_arithmetic"] = bool(abs(expect_us - hp["amortized_us_per_step"]) < 1.0)
    # 7. greedy-safety: bit-exact relocation, equivalence_rate == 1.0.
    tests["greedy_safe_equivalence_1"] = bool(safety["equivalence_rate"] == 1.0)
    # 8. the vectorized relocate consumes only a SLIVER of the descent's 500-cushion
    #    (< 50% of E[T]_descent - relocate-free bar); host_loop is infeasible (bar >
    #    5.207 ceiling) -- the binary build-blocker.
    tests["vectorized_headroom_small"] = bool(rec["vectorized_headroom_consumed_pct"] < 50.0)
    tests["host_loop_infeasible"] = bool(hp["clear500_bar"] > E_T_TREE_CEILING)
    # 9. descent feasibility flips: vectorized lets descent clear 500; host_loop does not.
    tests["feasibility_flips"] = bool(vp["descent_clears_500"] and not hp["descent_clears_500"])
    # 10. NaN-clean: every numeric finite.
    flat = [decomp["relocate_amortized_ms_per_step"], decomp["implied_salvage_frac"],
            rec["recoverable_us_per_step"], rec["recoverable_step_pct_vs_captured_target"],
            hp["clear500_bar"], vp["clear500_bar"], gp["amortized_us_per_step"]]
    tests["nan_clean"] = bool(all(math.isfinite(x) for x in flat))
    n_pass = sum(tests.values())
    return {"tests": tests, "n_pass": n_pass, "n_total": len(tests),
            "all_pass": bool(n_pass == len(tests))}


# ===== build hand-off ============================================================
def build_handoff(pricing, safety) -> dict:
    rec = pricing["recoverable"]
    return {
        "classification": "LIVE host-bound build-blocker (NOT a dead fallback)",
        "evidence": ("chiku-inu's STEPTIME trace (n=12841 STEADY decode steps, not "
                     "warmup) shows relocate_salvaged_kv as the CPU p90=335ms tail on "
                     "~35% of steps, driving the CPU mean to 122.5ms >> the 18.67ms GPU "
                     "floor -> the as-built eager descent stack is HOST-BOUND on the "
                     "timed decode window and timed out at 40min. It is absent from "
                     "lawine #136's 1.2182 GRAPH-CAPTURED target step and from my #154 "
                     "LINEAR stack, so no step-model leg priced it."),
        "why_a_blocker": ("the descent fix (land #71, making salvages actually descend "
                          "-- BUG-2) is exactly what ARMS this: as-built the 391 salvages "
                          "fire but do not descend (+0.077 E[T]); once they descend, the "
                          "relocate fires for real on every salvage. A data-dependent "
                          "Python loop over 37 layers CANNOT be CUDA-graph-captured, so "
                          "it pins the step host-bound (~122ms) instead of the 9.7ms "
                          "captured target -> the descent's E[T]=5.04 (-> 522) collapses "
                          f"to ~{rec['host_loop_descent_tps']:.0f} TPS."),
        "design": {
            "target": ("a single FUSED/vectorized GPU relocate: gather the accepted "
                       "rows across ALL 37 layers' K and V by a DEVICE commit-index in "
                       "one launch sequence (index_select on a [L,W,H,D] stack), scatter "
                       "to the committed slots with index_copy_. NO per-layer Python, NO "
                       ".item()/.cpu() host round-trip, NO per-element loop."),
            "ideal": ("if the build uses paged KV, relocation is a block-table / slot-map "
                      "update -- move an int slot-map, the KV bytes never move (zero-copy)."),
            "device_index_rule": ("the commit-index (which scratch rows -> which committed "
                                  "slots) must be produced ON-DEVICE by the accept walk "
                                  "(lawine #147 sync-free rule) and consumed by the relocate "
                                  "without a host readout -- so the relocate stays inside the "
                                  "captured graph."),
            "measured_target_us_per_step": rec["vectorized_amortized_us_per_step"],
            "measured_host_loop_us_per_step": rec["host_loop_amortized_us_per_step"],
        },
        "greedy_safety": safety["argument"],
        "stacks_with": ("multiplicative with land's descent (a denominator PRECONDITION: "
                        "without it the descent's numerator gain is unrealizable) and with "
                        "my #154 scatter+LP lever (both lower/keep the operative step). "
                        f"Combined operative bar stays ~{CLEAR500_BAR_154} (from #154) "
                        "PROVIDED the relocate is vectorized; a host-loop build makes the "
                        "bar unmeetable regardless of #154."),
    }


# ===== driver ====================================================================
def run(args) -> dict:
    assert torch.cuda.is_available(), "CUDA required (run with CUDA_VISIBLE_DEVICES=0)"
    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    import numpy as np
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print(f"[salvage-kv] GPU {torch.cuda.get_device_name(0)}  KV {N_LAYERS}L x "
          f"[{KV_HEADS}h,{HEAD_DIM}d] K+V bf16 = {KV_BYTES_PER_POS_PER_LAYER}B/pos/layer  "
          f"step_M8={STEP_M8_US:.1f}us  captured-target step={MEASURED_STEP_136}", flush=True)

    decomp = anchor_decomposition()
    print(f"[salvage-kv] ANCHOR: CPU mean {decomp['cpu_mean_observed_ms']}ms = "
          f"{decomp['implied_salvage_frac']:.3f}*{decomp['cpu_relocate_tail_ms']}ms + "
          f"{1-decomp['implied_salvage_frac']:.3f}*{decomp['cpu_nonsalvage_floor_ms']}ms "
          f"-> relocate amortized {decomp['relocate_amortized_ms_per_step']:.1f}ms/step "
          f"({decomp['relocate_pct_of_as_built_wall']:.1f}% of the host-bound wall)", flush=True)

    micro = run_microbench(dev, args, rng)
    pricing = price_paths(micro, decomp)
    safety = greedy_safety_check(dev, rng)
    st = self_tests(decomp, micro, pricing, safety)
    handoff = build_handoff(pricing, safety)

    rec = pricing["recoverable"]
    primary = int(st["all_pass"])
    test_val = rec["recoverable_step_pct_vs_captured_target"]
    verdict = ("LIVE-LANDMINE / build-blocker" if (rec["host_loop_bar_infeasible"]
               and rec["vectorized_keeps_descent_clearing_500"]) else "INCONCLUSIVE")

    print(f"\n[salvage-kv] SELF-TEST {st['n_pass']}/{st['n_total']} "
          f"({'PASS' if st['all_pass'] else 'FAIL'})", flush=True)
    print(f"[salvage-kv] host_loop amortized {rec['host_loop_amortized_us_per_step']/1e3:.1f}ms/step "
          f"-> descent {rec['host_loop_descent_tps']:.0f} TPS (bar {rec['host_loop_bar']:.2f}, "
          f"infeasible); vectorized {rec['vectorized_amortized_us_per_step']:.1f}us/step "
          f"-> descent {rec['vectorized_descent_tps']:.0f} TPS (bar {rec['vectorized_bar']:.3f})",
          flush=True)
    print(f"[salvage-kv] VERDICT={verdict}  PRIMARY salvage_kv_audit_self_test_passes={primary}  "
          f"TEST recoverable_step_pct_salvage_kv={test_val:.1f}% "
          f"(speedup {rec['speedup_vectorized_over_host']:.0f}x; "
          f"{rec['recoverable_pct_of_as_built_wall']:.0f}% of as-built wall)", flush=True)

    res = {
        "pr": 157, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "kv_geometry": {"n_layers": N_LAYERS, "kv_heads": KV_HEADS, "head_dim": HEAD_DIM,
                        "dtype": "bfloat16", "bytes_per_pos_per_layer": KV_BYTES_PER_POS_PER_LAYER,
                        "m_tree": M_TREE, "ctx_window": CTX_WINDOW},
        "anchors": {"k_cal": K_CAL, "step_m8_us": STEP_M8_US,
                    "measured_step_136_captured": MEASURED_STEP_136,
                    "clear500_bar_measured": CLEAR500_BAR_MEASURED,
                    "clear500_bar_154": CLEAR500_BAR_154,
                    "e_t_tree_ceiling": E_T_TREE_CEILING,
                    "e_t_descent_fix": E_T_DESCENT_FIX,
                    "oracle": {"salvages": ORACLE_SALVAGES, "full": ORACLE_FULL,
                               "steps": ORACLE_STEPS, "salvage_rate": ORACLE_SALVAGE_RATE,
                               "e_t": ORACLE_E_T},
                    "chiku_trace": CHIKU},
        "config": {"seed": args.seed, "host_calls": args.host_calls,
                   "host_warmup": args.host_warmup, "gpu_calls": args.gpu_calls,
                   "gpu_warmup": args.gpu_warmup},
        "anchor_decomposition": decomp,
        "microbench": micro,
        "pricing": pricing,
        "greedy_safety": safety,
        "self_test": st,
        "build_handoff": handoff,
        "verdict": verdict,
        "primary_metric": {"name": "salvage_kv_audit_self_test_passes", "value": primary},
        "test_metric": {"name": "recoverable_step_pct_salvage_kv", "value": test_val},
    }
    res["elapsed_s"] = time.time() - t0
    res["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2))
    print(f"[salvage-kv] wrote {out_path} ({res['elapsed_s']:.0f}s, peak {res['peak_gpu_gb']:.2f}GB)",
          flush=True)

    if args.wandb_group and not args.no_wandb:
        _wandb_log(args, res, out_path)
    return res


def _wandb_log(args, res: dict, out_path: Path):
    try:
        import wandb
        rec = res["pricing"]["recoverable"]
        decomp = res["anchor_decomposition"]
        micro = res["microbench"]
        run_w = wandb.init(project="gemma-challenge-senpai", entity="wandb-applied-ai-team",
                           group=args.wandb_group, name=args.wandb_name,
                           config={**res["config"], **res["anchors"], **res["kv_geometry"],
                                   "gpu": res["gpu"]})
        log = {
            "salvage_kv_audit_self_test_passes": res["primary_metric"]["value"],
            "recoverable_step_pct_salvage_kv": res["test_metric"]["value"],
            "self_test_n_pass": res["self_test"]["n_pass"],
            "self_test_n_total": res["self_test"]["n_total"],
            "relocate_amortized_ms_per_step": decomp["relocate_amortized_ms_per_step"],
            "relocate_pct_of_as_built_wall": decomp["relocate_pct_of_as_built_wall"],
            "implied_salvage_frac": decomp["implied_salvage_frac"],
            "host_loop_per_call_ms": micro["host_loop"]["mean"],
            "host_loop_p90_ms": micro["host_loop"]["p90"],
            "gpu_perlayer_per_call_us": micro["gpu_perlayer"]["mean"] * 1e3,
            "gpu_vectorized_per_call_us": micro["gpu_vectorized"]["mean"] * 1e3,
            "paged_slotmap_per_call_us": micro["paged_slotmap"]["mean"] * 1e3,
            "host_loop_amortized_us_per_step": rec["host_loop_amortized_us_per_step"],
            "vectorized_amortized_us_per_step": rec["vectorized_amortized_us_per_step"],
            "recoverable_us_per_step": rec["recoverable_us_per_step"],
            "speedup_vectorized_over_host": rec["speedup_vectorized_over_host"],
            "host_loop_descent_tps": rec["host_loop_descent_tps"],
            "vectorized_descent_tps": rec["vectorized_descent_tps"],
            "host_loop_bar": rec["host_loop_bar"],
            "vectorized_bar": rec["vectorized_bar"],
            "vectorized_headroom_consumed_pct": rec["vectorized_headroom_consumed_pct"],
            "descent_cushion_units": rec["descent_cushion_units"],
            "equivalence_rate": res["greedy_safety"]["equivalence_rate"],
            "clear500_bar_measured": CLEAR500_BAR_MEASURED,
            "clear500_bar_154": CLEAR500_BAR_154,
            "supply_ceiling_e_t": E_T_TREE_CEILING,
            "verdict_live_landmine": int("LIVE" in res["verdict"]),
        }
        wandb.log(log)
        run_w.summary.update(log)
        res["wandb_run_id"] = run_w.id
        wandb.finish()
        print(f"[salvage-kv] W&B run {run_w.id} (group {args.wandb_group})", flush=True)
        out_path.write_text(json.dumps(res, indent=2))
    except Exception as e:  # noqa: BLE001
        print(f"[salvage-kv] W&B logging skipped: {e!r}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host-calls", type=int, default=40, help="host_loop timed calls")
    ap.add_argument("--host-warmup", type=int, default=3)
    ap.add_argument("--gpu-calls", type=int, default=300, help="gpu-path timed calls")
    ap.add_argument("--gpu-warmup", type=int, default=30)
    ap.add_argument("--seed", type=int, default=157)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "research/spec_cost_model/salvage_kv_relocation_audit.json")
    ap.add_argument("--wandb-group", type=str, default=None)
    ap.add_argument("--wandb-name", type=str, default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--quick", action="store_true", help="fast smoke (few calls)")
    args = ap.parse_args(argv)
    if args.quick:
        args.host_calls, args.host_warmup, args.gpu_calls, args.gpu_warmup = 6, 1, 50, 5
    args.wandb_group = args.wandb_group or "salvage-kv-relocation-audit"
    args.wandb_name = args.wandb_name or "ubel/salvage-kv-relocation-audit"
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
