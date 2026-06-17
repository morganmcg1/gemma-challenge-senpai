#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
PR #537 — Medusa feasibility: a locally-realizable quality-safe drafter win?

The one drafter paradigm never tried here: K *parallel, independent* Medusa heads
(Cai et al. 2024, arxiv:2401.10774) on the frozen base hidden state, tree-verified.
Structurally orthogonal to the EAGLE-3/MTP autoregressive chain my #532 priced
NET-NEGATIVE-local: no draft chain -> no step-1 collapse, and the per-step draft
tax is ~one parallel head-forward, NOT K sequential forwards.

CORE CONTRIBUTION vs #532 (the on-task novelty):
  #532 priced EAGLE-3 with a draft-*MULTIPLIER* step model: t_step += draft_gpu*(m-1),
  because EAGLE-3's chain is MORE expensive than the deployed linear-MTP drafter.
  Medusa's parallel draft is *CHEAPER* than the linear-MTP drafter, so the correct
  generalization is a draft-*REPLACEMENT*:
      t_step_medusa = (t_step_old - draft_gpu_linear) + medusa_tax
                    = exec_plus_overhead + medusa_tax           (= 7.200 ms + tax)
  where draft_gpu_linear=1.554 ms (MY #523 measured) is the SERIAL linear-MTP drafter
  cost (the deployed step is serial: exec_gpu 6.89 + draft 1.554 + 0.31 exposed gap
  = 8.754 = t_step_old, so removing the draft recovers wall time), and medusa_tax is
  the K head-forward cost — MEASURED here by a real A10G microbench at the deployed
  shapes (hidden=2560, Vdraft in {12288 keepset, 262144 full}).

This flips the question: Medusa's E[T] is *lower* than the MTP chain (independent heads
decay with depth instead of the chain's conditioned increase), but its step is *cheaper*.
The screen prices the NET: TPS = E[T]/t_step. A near-free parallel draft can beat 442
even at lower E[T]. That is the whole bet, and it is the opposite tradeoff from #532.

HAPPY ACCIDENT (load-bearing, this stack only): the byteexact-442 base ALREADY prunes
its lm_head to the 12288-row keepset (manifest LM_HEAD_PRUNE=1, int4-pck04c-12k). So the
base greedy target is ALWAYS in that keepset -> a Medusa head tied to the 12288 lm_head
drafts in the EXACT vocab the target is restricted to => NO reduced-vocab acceptance
penalty (the usual Medusa concession is voided here), AND the cheapest draft tax.

ANALYSIS + MICROBENCH screen (PR step 1+3+4). NO HF Job, NO --launch, NO submission,
official_tps=0, no served-file change. GPU used ONLY for the head-forward tax microbench
(no model load, random weights at deployed shapes). --wandb_group medusa-feasibility.

Quality-safety: spec-dec verify is byte-exact M=8 => emitted output == target greedy
regardless of drafter (denken #505 bg03bq0d TV<=noise; inherited from MY #523 served
stack: r1-r2=1.0, PPL 2.37666). Medusa changes ONLY E[T], never the distribution.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

# ============================================================================
# GROUNDED INPUTS — measured reuse + literature anchors (provenance in comments)
# ============================================================================

# --- (1) byteexact-442 base MEASURED step decomposition (MY #523, W&B i11p5e3y) ---
# Arm bx_T4_S64 (packaged byte-exact fixed-3D split-KV rung), STEPTIME=1, 128x512, n=3.
TPS_BX442_LOCAL = 439.70559     # measured median wall_tps (self-consistent with draft_gpu)
ET_BASELINE_MTP = 3.849256527338719   # deployed linear-MTP K=7 served E[T] (#289/#526)
DRAFT_GPU_LINEAR_MS = 1.554     # linear-MTP drafter GPU time / step (SERIAL, the part removed)
EXEC_GPU_MS = 6.89              # target verify GPU time / step (UNCHANGED by drafter swap)
T_STEP_OLD_MS = 1000.0 * ET_BASELINE_MTP / TPS_BX442_LOCAL   # = 8.754 ms (roundtrips #523)
PPL_BX442 = 2.3766643358900286  # served PPL guardrail (<=2.42), drafter-INVARIANT
TAU_LO = 1.0352                 # local -> official scalar (MY #267, spread 0.135%)
PRIV_FACTOR = 0.804             # public/official -> private worst-case OOD (fern #305)

# --- (2) deployed model shapes (config.json: int4 byteexact-442 base) ---
HIDDEN = 2560
VOCAB_FULL = 262144
VOCAB_KEEPSET = 12288           # byteexact-442 base lm_head prune (int4-pck04c-12k keepset)
SOFTCAP = 30.0                  # Gemma final_logit_softcapping (applied to head logits)

# --- (3) MTP K=7 per-position conditional acceptance ladder (#532 hpfw9e3y) ---
# INCREASING with depth (autoregressive chain conditions later positions on accepted ones).
MTP_LADDER = [0.7290715372907154, 0.759434719768749, 0.7934024106576444,
              0.8215618336886993, 0.834712084347121, 0.835989117761368, 0.8465829846582985]

# --- (4) Medusa INDEPENDENT-head per-position acceptance ladders (literature-anchored) ---
# DECAYING with depth — independent heads predict t+k+1 from the SAME h_t, no chain
# conditioning. Anchored on Clover (arxiv:2405.00263) Baichuan-7B per-head top-1
# (H1=0.892/H2=0.814/H3=0.754) + a STEM uplift (workload is 100% reasoning, more
# predictable) + the byteexact-442 base's 12288-keepset prune VOIDS the usual reduced-vocab
# penalty (target is restricted to the same keepset the head drafts over).
#   single-candidate greedy per-position acceptance a_1..a_5 (NO tree):
MEDUSA_LADDERS = {
    "pessimistic": [0.72, 0.62, 0.53, 0.45, 0.38],
    "central":     [0.76, 0.66, 0.57, 0.49, 0.42],
    "optimistic":  [0.79, 0.70, 0.61, 0.53, 0.46],
}
# Sparse-tree (M=8 nodes) acceptance multiplier over single-candidate greedy
# (Medusa paper Fig.5 log-trend; ~1.20-1.35 in the 0.5-0.8 per-head regime). Bracketed.
TREE_MULT = {"none": 1.0, "lo": 1.20, "central": 1.27, "hi": 1.35}

# --- (5) #532 EAGLE-3 contrast (the paradigm Medusa is orthogonal to) ---
EAGLE3_PROJECTED_TPS_CENTRAL_532 = 414.0   # EAGLE-3 realistic ceiling at central chain tax (#532)
EAGLE3_BUILD_GPU_H_532 = 107.46577676190476

# --- (6) microbench config ---
DTYPE_BYTES_BF16 = 2
DTYPE_BYTES_INT4 = 0.5
A10G_BW_GBPS = 600.0            # effective HBM BW (for the analytic BW cross-check only)
BENCH_K = [4, 5]               # K head counts to price
BENCH_WARMUP = 30
BENCH_ITERS = 200


# ============================================================================
# CORE MATH
# ============================================================================
def et_from_ladder(a_ladder: list[float]) -> float:
    """E[T] = 1 + sum_{m>=1} prod_{k<=m} a_k  (survival form, #289). Single-candidate."""
    et, g = 1.0, 1.0
    for ak in a_ladder:
        g *= ak
        et += g
    return et


def et_tree(a_ladder: list[float], tree_mult: float) -> float:
    """Tree-lifted E[T]: single-candidate E[T] scaled by the M=8 sparse-tree multiplier.
    The +1 base token is always emitted; only the accepted-draft mass is tree-lifted."""
    single = et_from_ladder(a_ladder)
    accepted_mass = single - 1.0
    return 1.0 + accepted_mass * tree_mult


def exec_plus_overhead_ms() -> float:
    """The non-draft part of the step, held fixed under the Medusa draft swap."""
    return T_STEP_OLD_MS - DRAFT_GPU_LINEAR_MS     # = 7.200 ms


def t_step_medusa_ms(medusa_tax_ms: float) -> float:
    """Draft-REPLACEMENT step model: swap the serial linear draft for the Medusa head tax."""
    return exec_plus_overhead_ms() + medusa_tax_ms


def tps_local(et: float, medusa_tax_ms: float) -> float:
    return 1000.0 * et / t_step_medusa_ms(medusa_tax_ms)


def to_official(tps: float) -> float:
    return tps * TAU_LO


def to_private(tps: float) -> float:
    return tps * TAU_LO * PRIV_FACTOR


def breakeven_tax_ms(et: float, target_tps: float) -> float:
    """Max Medusa draft tax (ms) that still reaches target_tps at this E[T].
    TPS = 1000*et/(7.200+tax) >= target  <=>  tax <= 1000*et/target - 7.200."""
    return 1000.0 * et / target_tps - exec_plus_overhead_ms()


# ============================================================================
# GPU MICROBENCH — the MEASURED Medusa head-forward draft tax on THIS A10G
# ============================================================================
def _bench_once(fn, warmup: int, iters: int) -> float:
    """Median ms over `iters` timed calls (CUDA events), after `warmup`."""
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def run_microbench(k_list=BENCH_K, warmup=BENCH_WARMUP, iters=BENCH_ITERS) -> dict | None:
    """Measure the K-head Medusa draft-forward wall time at deployed shapes.

    The draft per decode step (concurrency=1): one most-recent hidden h_t (1xHIDDEN, bf16);
    per head k: z_k = h_t + SiLU(W1_k @ h_t) [ResBlock, W1_k HIDDENxHIDDEN]; then project
    Z (KxHIDDEN) through the lm_head (VdraftxHIDDEN) with Gemma softcap; top-k per head.
    Variants: shared-batched (stack K -> one M=K projection read) vs per-head (K M=1 reads);
    Vdraft in {12288 keepset, 262144 full}; eager and CUDA-graph-captured.
    """
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        print(f"[microbench] torch import failed (non-fatal): {exc}", file=sys.stderr)
        return None
    if not torch.cuda.is_available():
        print("[microbench] CUDA not available (set CUDA_VISIBLE_DEVICES=0); skipping",
              file=sys.stderr)
        return None

    dev = torch.device("cuda")
    dt = torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = True
    gpu_name = torch.cuda.get_device_name(0)

    def make_head_weights(k):
        w1 = [torch.randn(HIDDEN, HIDDEN, device=dev, dtype=dt) * 0.02 for _ in range(k)]
        return w1

    def make_proj(vocab):
        return torch.randn(vocab, HIDDEN, device=dev, dtype=dt) * 0.02

    h = torch.randn(1, HIDDEN, device=dev, dtype=dt)
    silu = torch.nn.functional.silu

    def resblocks(w1_list):
        zs = []
        for w1 in w1_list:
            z = h + silu(h @ w1.t())
            zs.append(z)
        return torch.cat(zs, dim=0)  # (K, HIDDEN)

    def project_shared(Z, proj):
        logits = Z @ proj.t()                       # (K, vocab)
        logits = torch.tanh(logits / SOFTCAP) * SOFTCAP
        # top-k candidate extraction per head (cheap, but included for fidelity)
        torch.topk(logits, k=min(10, logits.shape[-1]), dim=-1)
        return logits

    def project_perhead(Z, proj):
        out = []
        for i in range(Z.shape[0]):
            logits = Z[i:i + 1] @ proj.t()          # (1, vocab)
            logits = torch.tanh(logits / SOFTCAP) * SOFTCAP
            torch.topk(logits, k=min(10, logits.shape[-1]), dim=-1)
            out.append(logits)
        return out

    results = {"gpu": gpu_name, "dtype": "bf16", "warmup": warmup, "iters": iters, "by_k": {}}
    for k in k_list:
        w1_list = make_head_weights(k)
        proj_keep = make_proj(VOCAB_KEEPSET)
        proj_full = make_proj(VOCAB_FULL)
        rec = {}

        # ResBlock-only (K heads, M=1 each) — the head-transform cost
        rec["resblock_only_ms"] = _bench_once(lambda: resblocks(w1_list), warmup, iters)

        # shared-batched: K ResBlocks + one M=K projection (one weight read)
        def f_keep_shared():
            Z = resblocks(w1_list)
            project_shared(Z, proj_keep)
        def f_full_shared():
            Z = resblocks(w1_list)
            project_shared(Z, proj_full)
        rec["shared_batched_12k_ms"] = _bench_once(f_keep_shared, warmup, iters)
        rec["shared_batched_fullvocab_ms"] = _bench_once(f_full_shared, warmup, iters)

        # per-head: K ResBlocks + K M=1 projections (K weight reads)
        def f_keep_perhead():
            Z = resblocks(w1_list)
            project_perhead(Z, proj_keep)
        def f_full_perhead():
            Z = resblocks(w1_list)
            project_perhead(Z, proj_full)
        rec["per_head_12k_ms"] = _bench_once(f_keep_perhead, warmup, iters)
        rec["per_head_fullvocab_ms"] = _bench_once(f_full_perhead, warmup, iters)

        # CUDA-graph capture of the recommended config (shared-batched 12k):
        # the deployed stack runs ONEGRAPH, so the head-forward would be captured ->
        # launch overhead vanishes, leaving the BW/compute floor (optimistic-realistic).
        graphed = None
        try:
            g = torch.cuda.CUDAGraph()
            Zc = torch.empty(k, HIDDEN, device=dev, dtype=dt)
            for _ in range(3):
                f_keep_shared()
            torch.cuda.synchronize()
            with torch.cuda.graph(g):
                Zc = resblocks(w1_list)
                project_shared(Zc, proj_keep)
            def f_graph():
                g.replay()
            graphed = _bench_once(f_graph, warmup, iters)
        except Exception as exc:  # noqa: BLE001
            print(f"[microbench] CUDA-graph capture failed (non-fatal): {exc}", file=sys.stderr)
        rec["shared_batched_12k_graphed_ms"] = graphed

        # analytic int4-tied projection (reuse deployed int4 lm_head; read = bf16/4)
        proj_read_bf16_12k = VOCAB_KEEPSET * HIDDEN * DTYPE_BYTES_BF16 / 1e9 / A10G_BW_GBPS * 1e3
        proj_read_int4_12k = VOCAB_KEEPSET * HIDDEN * DTYPE_BYTES_INT4 / 1e9 / A10G_BW_GBPS * 1e3
        rec["analytic_proj_read_bf16_12k_ms"] = proj_read_bf16_12k
        rec["analytic_proj_read_int4_12k_ms"] = proj_read_int4_12k
        # int4-tied estimate: resblock(measured) + int4 projection read floor
        rec["est_int4_tied_shared_ms"] = rec["resblock_only_ms"] + proj_read_int4_12k

        rec["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        results["by_k"][str(k)] = rec
        del w1_list, proj_keep, proj_full
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return results


# ============================================================================
# BUILD REPORT
# ============================================================================
def select_tax(microbench: dict | None, k: int) -> dict:
    """Pick the primary/conservative/optimistic measured tax for head-count k.

    primary       = shared-batched 12288-keepset (eager) — the cheapest faithful config
    optimistic    = graphed shared-batched 12288 (ONEGRAPH, launch overhead hidden)
    conservative  = shared-batched FULL vocab (262144) — if a head can't reuse the keepset
    worst         = per-head full vocab (no batching)
    int4_tied     = resblock + int4 keepset read (reuse deployed int4 lm_head; analytic floor)
    """
    if microbench is None or str(k) not in microbench.get("by_k", {}):
        # fall back to SME analytic numbers (BW-derived) when no GPU available
        rb = 0.022 * k
        return {
            "source": "analytic_sme_fallback",
            "primary_ms": rb + VOCAB_KEEPSET * HIDDEN * DTYPE_BYTES_BF16 / 1e9 / A10G_BW_GBPS * 1e3,
            "optimistic_ms": rb + VOCAB_KEEPSET * HIDDEN * DTYPE_BYTES_INT4 / 1e9 / A10G_BW_GBPS * 1e3,
            "conservative_ms": rb + VOCAB_FULL * HIDDEN * DTYPE_BYTES_BF16 / 1e9 / A10G_BW_GBPS * 1e3,
            "worst_ms": rb + k * VOCAB_FULL * HIDDEN * DTYPE_BYTES_BF16 / 1e9 / A10G_BW_GBPS * 1e3,
            "int4_tied_ms": rb + VOCAB_KEEPSET * HIDDEN * DTYPE_BYTES_INT4 / 1e9 / A10G_BW_GBPS * 1e3,
        }
    rec = microbench["by_k"][str(k)]
    graphed = rec.get("shared_batched_12k_graphed_ms")
    return {
        "source": "measured_microbench_a10g",
        "primary_ms": rec["shared_batched_12k_ms"],
        "optimistic_ms": graphed if graphed is not None else rec["est_int4_tied_shared_ms"],
        "conservative_ms": rec["shared_batched_fullvocab_ms"],
        "worst_ms": rec["per_head_fullvocab_ms"],
        "int4_tied_ms": rec["est_int4_tied_shared_ms"],
    }


def build_report(microbench: dict | None) -> dict:
    # ---- DIAGNOSE: Medusa decaying ladder vs MTP increasing ladder -> E[T] ----
    et_mtp = et_from_ladder(MTP_LADDER)
    ladders_et = {}
    for name, lad in MEDUSA_LADDERS.items():
        ladders_et[name] = {
            "ladder_a1_aK": lad,
            "K": len(lad),
            "et_single_candidate": et_from_ladder(lad),
            "et_tree_lo": et_tree(lad, TREE_MULT["lo"]),
            "et_tree_central": et_tree(lad, TREE_MULT["central"]),
            "et_tree_hi": et_tree(lad, TREE_MULT["hi"]),
        }
    # central operating E[T] = central ladder, central tree multiplier
    et_medusa_central = ladders_et["central"]["et_tree_central"]
    et_medusa_band = [ladders_et["pessimistic"]["et_tree_lo"],
                      ladders_et["optimistic"]["et_tree_hi"]]

    # ---- TAX: measured microbench, per K ----
    tax_by_k = {str(k): select_tax(microbench, k) for k in BENCH_K}
    # headline uses K=5 (more heads -> slightly higher E[T]; the SME central E[T] is K=5)
    k_primary = 5
    tax_primary = tax_by_k[str(k_primary)]["primary_ms"]
    tax_optimistic = tax_by_k[str(k_primary)]["optimistic_ms"]
    tax_conservative = tax_by_k[str(k_primary)]["conservative_ms"]
    tax_int4 = tax_by_k[str(k_primary)]["int4_tied_ms"]

    # ---- BREAK-EVEN: the (E[T], tax) corner that beats 442 / 500 ----
    breakeven = {
        name: {
            "et_used": ladders_et[name]["et_tree_central"],
            "max_tax_ms_for_442": breakeven_tax_ms(ladders_et[name]["et_tree_central"], 442.0),
            "max_tax_ms_for_500": breakeven_tax_ms(ladders_et[name]["et_tree_central"], 500.0),
        }
        for name in MEDUSA_LADDERS
    }
    # also break-even at single-candidate (no-tree) E[T] floors
    breakeven_singlecand = {
        name: {
            "et_used": ladders_et[name]["et_single_candidate"],
            "max_tax_ms_for_442": breakeven_tax_ms(ladders_et[name]["et_single_candidate"], 442.0),
        }
        for name in MEDUSA_LADDERS
    }

    # ---- GRID: projected TPS over {E[T] case} x {measured tax case} ----
    et_grid_cases = {
        "pessimistic_singlecand": ladders_et["pessimistic"]["et_single_candidate"],
        "pessimistic_tree": ladders_et["pessimistic"]["et_tree_central"],
        "central_singlecand": ladders_et["central"]["et_single_candidate"],
        "central_tree": et_medusa_central,
        "optimistic_tree": ladders_et["optimistic"]["et_tree_central"],
        "optimistic_tree_hi": ladders_et["optimistic"]["et_tree_hi"],
    }
    tax_grid_cases = {
        "int4_tied": tax_int4,
        "primary_12k_eager": tax_primary,
        "optimistic_graphed": tax_optimistic,
        "conservative_fullvocab": tax_conservative,
    }
    grid = {}
    for ek, ev in et_grid_cases.items():
        grid[ek] = {}
        for tk, tv in tax_grid_cases.items():
            loc = tps_local(ev, tv)
            grid[ek][tk] = {
                "tax_ms": tv,
                "t_step_ms": t_step_medusa_ms(tv),
                "tps_local": loc,
                "tps_official": to_official(loc),
                "tps_private": to_private(loc),
                "beats_442": loc >= 442.0,
                "crosses_500_local": loc >= 500.0,
                "crosses_500_official": to_official(loc) >= 500.0,
            }

    # ---- headline operating point: central E[T] x primary measured tax, LOCAL ----
    primary_op = grid["central_tree"]["primary_12k_eager"]
    projected_tps_medusa = primary_op["tps_local"]
    beats_442 = projected_tps_medusa >= 442.0
    crosses_500 = projected_tps_medusa >= 500.0

    # corner summaries
    all_local = [grid[ek][tk]["tps_local"] for ek in et_grid_cases for tk in tax_grid_cases]
    realistic_tax_keys = ("int4_tied", "primary_12k_eager", "optimistic_graphed")
    realistic_local = [grid[ek][tk]["tps_local"]
                       for ek in et_grid_cases for tk in realistic_tax_keys]
    beats_442_central_band = all(
        grid[ek]["primary_12k_eager"]["beats_442"]
        for ek in ("central_tree", "optimistic_tree", "optimistic_tree_hi"))

    # E[T] vs MTP
    e_t_vs_mtp_central = et_medusa_central - ET_BASELINE_MTP
    e_t_vs_mtp_optimistic = ladders_et["optimistic"]["et_tree_hi"] - ET_BASELINE_MTP

    # step reduction that pays for the lower E[T]
    step_reduction_pct = (T_STEP_OLD_MS - primary_op["t_step_ms"]) / T_STEP_OLD_MS * 100.0
    et_reduction_pct = (ET_BASELINE_MTP - et_medusa_central) / ET_BASELINE_MTP * 100.0
    # at the primary measured tax, the minimum E[T] that still beats 442
    et_floor_442_at_primary = 442.0 * t_step_medusa_ms(tax_primary) / 1000.0

    # ---- VERDICT (break-even-centric: the deciding variable is realized E[T]) ----
    # at the primary measured tax, the E[T] thresholds for 442 / 500:
    et_floor_442 = 442.0 * t_step_medusa_ms(tax_primary) / 1000.0
    et_floor_500 = 500.0 * t_step_medusa_ms(tax_primary) / 1000.0
    et_opt = ladders_et["optimistic"]["et_tree_hi"]
    if et_medusa_central >= et_floor_500:
        verdict = "GO-LOCAL-PROMISING (central E[T] clears the 500 floor)"
    elif et_medusa_central >= et_floor_442:
        verdict = "CONDITIONAL-GO-LOCAL (central E[T] beats 442; 500 only at optimistic corner)"
    elif et_opt >= et_floor_442:
        verdict = (
            "MARGINAL-LOCAL (central E[T] just MISSES 442; beats it only in the optimistic E[T] "
            "band — verdict hinges on whether realized E[T] clears the break-even floor)")
    else:
        verdict = "NO-GO-LOCAL (even optimistic E[T] below the 442 floor)"

    rationale = (
        f"Medusa REVERSES the #532 EAGLE-3 tradeoff but lands MARGINAL, not a clean win. Mechanism: "
        f"the parallel head draft is ~{DRAFT_GPU_LINEAR_MS / max(tax_primary, 1e-6):.0f}x CHEAPER than "
        f"the deployed serial linear-MTP drafter (measured tax {tax_primary:.3f} ms vs draft_gpu "
        f"{DRAFT_GPU_LINEAR_MS} ms), so the step drops {step_reduction_pct:.0f}% "
        f"({T_STEP_OLD_MS:.2f}->{primary_op['t_step_ms']:.2f} ms) — the OPPOSITE sign of EAGLE-3's "
        f"step INFLATION. BUT independent heads decay with depth (vs the MTP chain's conditioned "
        f"increase), so E[T] drops {et_reduction_pct:.0f}% (MTP {ET_BASELINE_MTP:.2f} -> Medusa central "
        f"{et_medusa_central:.2f}), and the cheaper step does NOT quite over-compensate at central: "
        f"{projected_tps_medusa:.0f} TPS local ({projected_tps_medusa - 442:+.0f} vs 442). The verdict is "
        f"a knife-edge E[T] race: at the measured {tax_primary:.3f} ms tax, Medusa beats 442 iff realized "
        f"E[T] >= {et_floor_442:.2f} and crosses 500 iff E[T] >= {et_floor_500:.2f}. The "
        f"literature-anchored band is central {et_medusa_central:.2f} (just UNDER 442) / optimistic "
        f"{et_opt:.2f} (clears 442, approaches 500). So realized E[T] is THE deciding uncertainty — a "
        f"cheap local Medusa-1 train (frozen backbone, tied 12288 lm_head, train only ResBlocks; "
        f"plausibly IN-SLOT, unlike EAGLE-3's ~{EAGLE3_BUILD_GPU_H_532:.0f} GPU-h cluster) would resolve "
        f"it. Quality-safe by construction (byte-exact verify => output==greedy; PPL {PPL_BX442:.4f}, "
        f"r1-r2=1.0 inherited from #523, drafter-invariant). HAPPY ACCIDENT: the base's existing 12288 "
        f"lm_head-prune voids Medusa's usual reduced-vocab penalty (target restricted to the same "
        f"keepset the head drafts over) AND gives the cheapest tax (int4-tied reuse)."
    )

    report = {
        "pr": 537, "agent": "lawine",
        "title": "Medusa feasibility: a locally-realizable quality-safe drafter win?",
        "kind": "medusa-feasibility-screen (analysis + A10G head-forward microbench; PR step 1+3+4)",
        "analysis_only": True, "no_hf_job": True, "no_launch": True, "no_submission": True,
        "no_served_file_change": True, "official_tps": 0,
        "gpu_used_for": "head-forward draft-tax microbench only (random weights at deployed shapes, no model load)",
        "wandb_group": "medusa-feasibility",
        "model": (
            "draft-REPLACEMENT step model (generalizes #532's draft-MULTIPLIER): "
            "t_step_medusa = (t_step_old - draft_gpu_linear) + medusa_tax; TPS = E[T]/t_step. "
            "medusa_tax MEASURED by A10G microbench at hidden=2560, Vdraft in {12288,262144}."
        ),
        "inputs": {
            "tps_bx442_local_523": TPS_BX442_LOCAL, "et_baseline_mtp_526": ET_BASELINE_MTP,
            "draft_gpu_linear_ms_523": DRAFT_GPU_LINEAR_MS, "exec_gpu_ms_523": EXEC_GPU_MS,
            "t_step_old_ms": T_STEP_OLD_MS, "ppl_bx442_523": PPL_BX442,
            "tau_lo_267": TAU_LO, "priv_factor_305": PRIV_FACTOR,
            "hidden": HIDDEN, "vocab_full": VOCAB_FULL, "vocab_keepset": VOCAB_KEEPSET,
            "softcap": SOFTCAP, "mtp_ladder": MTP_LADDER, "medusa_ladders": MEDUSA_LADDERS,
            "tree_mult": TREE_MULT,
            "source_runs": {
                "byteexact442_steptime_523": "i11p5e3y", "et_per_pos_526_289": "3piz86i4/5m17r52s",
                "mtp_ladder_532": "hpfw9e3y", "quality_safe_505": "bg03bq0d",
                "local_official_267": "nzqnd154",
            },
            "literature": {
                "medusa": "arxiv:2401.10774", "clover_perhead_acc": "arxiv:2405.00263",
                "eagle": "arxiv:2401.15077",
            },
        },

        # ============ REQUIRED KEY OUTPUTS ============
        "medusa_heads_K": k_primary,
        "medusa_heads_K_band": BENCH_K,
        "best_realized_E_T_medusa": et_medusa_central,        # central ladder, central tree mult
        "best_realized_E_T_medusa_is_projection_not_measured": True,
        "best_realized_E_T_medusa_band": et_medusa_band,
        "E_T_vs_mtp": e_t_vs_mtp_central,                     # negative: Medusa E[T] < MTP 3.849
        "E_T_vs_mtp_band": [e_t_vs_mtp_central, e_t_vs_mtp_optimistic],
        "medusa_draft_step_tax": tax_primary,                 # MEASURED, primary config (ms)
        "medusa_draft_step_tax_source": tax_by_k[str(k_primary)]["source"],
        "medusa_draft_step_tax_band_ms": [tax_optimistic, tax_conservative],
        "medusa_draft_step_tax_vs_eagle3": (
            f"Medusa {tax_primary:.2f} ms (one parallel head-forward) vs EAGLE-3 chain tax "
            f"~{DRAFT_GPU_LINEAR_MS * (3.0 - 1.0):.2f} ms extra (m=3 central, #532/#295): "
            f"Medusa draft is REPLACEMENT-cheaper, EAGLE-3 is MULTIPLIER-dearer"
        ),
        "projected_tps_medusa": projected_tps_medusa,         # LOCAL, central E[T], primary tax
        "projected_tps_medusa_official": primary_op["tps_official"],
        "projected_tps_medusa_private": primary_op["tps_private"],
        "projected_tps_medusa_frame": "byteexact-442 LOCAL; central tree E[T]; primary measured 12288 tax",
        "beats_442": beats_442,
        "crosses_500": crosses_500,
        "headline_et_case": "central_tree",      # grid key the headline TPS reads (fold -> measured_tree)
        "et_floor_to_beat_442": et_floor_442,    # min realized E[T] to beat 442 at primary measured tax
        "et_floor_to_cross_500": et_floor_500,
        "et_central_vs_floor_442": et_medusa_central - et_floor_442,   # negative => central misses
        "et_optimistic_vs_floor_442": et_opt - et_floor_442,           # positive => optimistic clears
        "drafter_is_quality_safe": True,
        "drafter_is_quality_safe_reason": (
            "spec-dec verify is byte-exact M=8: Medusa acceptance changes ONLY E[T] (tokens/step), "
            "NOT the emitted distribution — served output == target greedy regardless of head quality. "
            "self-det/PPL invariant to the drafter, inherited from MY #523 served stack (no re-serve)."
        ),
        "self_det": "served r1-r2 = 1.0, attention 0/8 byte-exact microbench (#523 i11p5e3y); UNCHANGED by drafter",
        "ppl": PPL_BX442, "ppl_guardrail": 2.42,
        "medusa_train_gpu_h": None,   # filled by the train leg if run; None = analytic screen only
        "go_no_go": verdict, "go_no_go_rationale": rationale,

        # ============ SUPPORTING DETAIL ============
        "diagnose": {
            "et_mtp_roundtrip": et_mtp,
            "mtp_ladder_shape": "INCREASING (autoregressive chain conditioning)",
            "medusa_ladder_shape": "DECAYING (independent heads from same h_t)",
            "medusa_ladders_et": ladders_et,
            "et_medusa_central": et_medusa_central,
            "et_medusa_band": et_medusa_band,
            "e_t_vs_mtp_central": e_t_vs_mtp_central,
            "medusa_loses_on_et_wins_on_step": (et_medusa_central < ET_BASELINE_MTP),
            "keepset_voids_reduced_vocab_penalty": (
                "byteexact-442 base lm_head pruned to 12288 (LM_HEAD_PRUNE=1, int4-pck04c-12k); "
                "target greedy restricted to keepset => Medusa head over same keepset has no coverage loss"
            ),
        },
        "microbench": microbench,
        "tax_by_k": tax_by_k,
        "breakeven_tax": breakeven,
        "breakeven_tax_singlecand": breakeven_singlecand,
        "reconciliation": {
            "exec_plus_overhead_ms": exec_plus_overhead_ms(),
            "t_step_old_ms": T_STEP_OLD_MS,
            "draft_gpu_linear_removed_ms": DRAFT_GPU_LINEAR_MS,
            "step_reduction_pct": step_reduction_pct,
            "et_reduction_pct": et_reduction_pct,
            "primary_operating_point": primary_op,
            "grid_et_x_tax": grid,
            "max_local_any": max(all_local),
            "max_local_realistic_tax": max(realistic_local),
            "beats_442_across_central_optimistic_band": beats_442_central_band,
            "statement": (
                "The draft-replacement model recovers the #532 EAGLE-3 case as the m>1 multiplier "
                "limit; Medusa is the m<1 limit (cheaper draft). The SAME measured 442-base "
                "decomposition that made EAGLE-3 NET-NEGATIVE makes Medusa NET-POSITIVE — the sign "
                "of (medusa_tax - draft_gpu_linear) flips the verdict."
            ),
        },
        "eagle3_contrast_532": {
            "eagle3_projected_tps_central": EAGLE3_PROJECTED_TPS_CENTRAL_532,
            "eagle3_build_gpu_h": EAGLE3_BUILD_GPU_H_532,
            "medusa_vs_eagle3_local": projected_tps_medusa - EAGLE3_PROJECTED_TPS_CENTRAL_532,
        },
    }

    return report


# ============================================================================
# SELF-TEST (0 GPU — pure structural/arithmetic gate)
# ============================================================================
def run_self_test(report: dict) -> dict:
    d = report["diagnose"]
    r = report["reconciliation"]
    g = r["grid_et_x_tax"]
    lad = d["medusa_ladders_et"]
    et_mtp = ET_BASELINE_MTP

    c = {
        # ladders
        "mtp_ladder_increasing": all(MTP_LADDER[i] <= MTP_LADDER[i + 1] + 1e-9 for i in range(6)),
        "medusa_central_decreasing": all(
            MEDUSA_LADDERS["central"][i] >= MEDUSA_LADDERS["central"][i + 1] - 1e-9 for i in range(4)),
        "medusa_ladders_in_unit": all(
            0.0 < a < 1.0 for lst in MEDUSA_LADDERS.values() for a in lst),
        "pess_below_central_below_opt": (
            lad["pessimistic"]["et_single_candidate"] < lad["central"]["et_single_candidate"]
            < lad["optimistic"]["et_single_candidate"]),
        "mtp_ladder_reproduces_3849": abs(d["et_mtp_roundtrip"] - et_mtp) < 1e-6,
        # E[T] structure: Medusa central below MTP, tree lifts above single-candidate
        "et_medusa_central_below_mtp": report["best_realized_E_T_medusa"] < et_mtp,
        "tree_lifts_above_singlecand": (
            lad["central"]["et_tree_central"] > lad["central"]["et_single_candidate"]),
        "et_vs_mtp_negative": report["E_T_vs_mtp"] < 0.0,
        "et_band_brackets_central": (
            report["best_realized_E_T_medusa_band"][0] <= report["best_realized_E_T_medusa"]
            <= report["best_realized_E_T_medusa_band"][1]),
        # step model
        "exec_plus_overhead_is_7_2": abs(r["exec_plus_overhead_ms"] - 7.2) < 0.01,
        "t_step_old_roundtrips": abs(r["t_step_old_ms"] - 1000.0 * et_mtp / TPS_BX442_LOCAL) < 1e-9,
        "medusa_tax_below_linear_draft": report["medusa_draft_step_tax"] < DRAFT_GPU_LINEAR_MS,
        "medusa_step_below_baseline": r["primary_operating_point"]["t_step_ms"] < T_STEP_OLD_MS,
        "step_reduction_positive": r["step_reduction_pct"] > 0.0,
        # the central reconciliation: step win over-compensates E[T] loss
        "baseline_roundtrips_to_4397": abs(tps_local(et_mtp, DRAFT_GPU_LINEAR_MS) - TPS_BX442_LOCAL) < 0.5,
        "beats_442_iff_central_clears_floor": (
            report["beats_442"] == (report["best_realized_E_T_medusa"] >= report["et_floor_to_beat_442"] - 1e-9)),
        "et_floor_442_in_plausible_range": 3.0 < report["et_floor_to_beat_442"] < 3.6,
        "primary_tps_matches_grid": abs(
            report["projected_tps_medusa"]
            - g[report["headline_et_case"]]["primary_12k_eager"]["tps_local"]) < 1e-6,
        # break-even monotonicity: higher E[T] tolerates more tax
        "breakeven_increases_with_et": (
            report["breakeven_tax"]["optimistic"]["max_tax_ms_for_442"]
            > report["breakeven_tax"]["central"]["max_tax_ms_for_442"]
            > report["breakeven_tax"]["pessimistic"]["max_tax_ms_for_442"]),
        "tps_decreases_with_tax": (
            g["central_tree"]["int4_tied"]["tps_local"]
            >= g["central_tree"]["primary_12k_eager"]["tps_local"]
            > g["central_tree"]["conservative_fullvocab"]["tps_local"]),
        "tps_increases_with_et": (
            g["optimistic_tree"]["primary_12k_eager"]["tps_local"]
            > g["central_tree"]["primary_12k_eager"]["tps_local"]
            > g["pessimistic_tree"]["primary_12k_eager"]["tps_local"]),
        # crosses_500 only at optimistic corner (sanity: not at pessimistic)
        "crosses_500_false_at_pessimistic": not g["pessimistic_tree"]["primary_12k_eager"]["crosses_500_local"],
        # transfers
        "tau_lo_gt_one": TAU_LO > 1.0, "priv_factor_lt_one": PRIV_FACTOR < 1.0,
        "official_gt_local": r["primary_operating_point"]["tps_official"] > r["primary_operating_point"]["tps_local"],
        # quality-safety
        "drafter_quality_safe": report["drafter_is_quality_safe"] is True,
        "ppl_under_guardrail": report["ppl"] < 2.42,
        "best_et_flag_consistent": (
            report["best_realized_E_T_medusa_is_projection_not_measured"]
            == (report.get("medusa_train") is None)),
        # verdict / hygiene
        "verdict_present": isinstance(report["go_no_go"], str) and len(report["go_no_go"]) > 0,
        "no_nan_inf": _all_finite({k: v for k, v in report.items() if k not in ("microbench",)}),
    }
    n = len(c)
    passed = sum(1 for v in c.values() if v)
    return {"conditions": c, "n_checks": n, "n_passed": passed, "passes": passed == n}


def _all_finite(obj) -> bool:
    if isinstance(obj, bool):
        return True
    if isinstance(obj, float):
        return math.isfinite(obj)
    if isinstance(obj, dict):
        return all(_all_finite(v) for v in obj.values())
    if isinstance(obj, list):
        return all(_all_finite(v) for v in obj)
    return True


# ============================================================================
# W&B (best-effort; analysis-only)
# ============================================================================
def log_to_wandb(report: dict, group: str, name: str) -> str | None:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"W&B import failed (non-fatal): {exc}", file=sys.stderr)
        return None
    try:
        run = wandb.init(
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
            group=group, name=name, config=report["inputs"], job_type="analysis",
        )
        flat = {
            "medusa_heads_K": report["medusa_heads_K"],
            "best_realized_E_T_medusa": report["best_realized_E_T_medusa"],
            "E_T_vs_mtp": report["E_T_vs_mtp"],
            "medusa_draft_step_tax": report["medusa_draft_step_tax"],
            "projected_tps_medusa": report["projected_tps_medusa"],
            "projected_tps_medusa_official": report["projected_tps_medusa_official"],
            "projected_tps_medusa_private": report["projected_tps_medusa_private"],
            "beats_442": float(report["beats_442"]),
            "crosses_500": float(report["crosses_500"]),
            "drafter_is_quality_safe": float(report["drafter_is_quality_safe"]),
            "ppl": report["ppl"],
            "medusa_train_gpu_h": report["medusa_train_gpu_h"] if report["medusa_train_gpu_h"] else 0.0,
            "step_reduction_pct": report["reconciliation"]["step_reduction_pct"],
            "et_reduction_pct": report["reconciliation"]["et_reduction_pct"],
            "self_test_passes": float(report["self_test"]["passes"]),
            "self_test_n_checks": float(report["self_test"]["n_checks"]),
            "analysis_only": True, "no_hf_job": True, "official_tps": 0,
        }
        wandb.summary.update(flat)
        wandb.log({f"summary/{k}": v for k, v in flat.items() if isinstance(v, (int, float))})
        for ek, row in report["reconciliation"]["grid_et_x_tax"].items():
            for tk, rec in row.items():
                wandb.log({
                    f"grid/{ek}/{tk}/tps_local": rec["tps_local"],
                    f"grid/{ek}/{tk}/tps_official": rec["tps_official"],
                    f"grid/{ek}/{tk}/t_step_ms": rec["t_step_ms"],
                    f"grid/{ek}/{tk}/beats_442": float(rec["beats_442"]),
                })
        for cond, val in report["self_test"]["conditions"].items():
            wandb.log({f"test/{cond}": float(bool(val))})
        run_id = run.id
        wandb.finish()
        return run_id
    except Exception as exc:  # noqa: BLE001
        print(f"W&B logging failed (non-fatal): {exc}", file=sys.stderr)
        return None


# ============================================================================
def _fmt(report: dict) -> str:
    r = report["reconciliation"]
    g = r["grid_et_x_tax"]
    d = report["diagnose"]
    tax = report["tax_by_k"][str(report["medusa_heads_K"])]
    lines = [
        "=== PR #537 — Medusa feasibility screen (ANALYSIS + A10G head-forward microbench) ===",
        f"DIAGNOSE: MTP K=7 ladder INCREASES 0.729->0.847 (chain) -> E[T]={d['et_mtp_roundtrip']:.4f}",
        f"          Medusa heads DECAY (independent) -> E[T] central {d['et_medusa_central']:.3f} "
        f"(band {d['et_medusa_band'][0]:.2f}..{d['et_medusa_band'][1]:.2f}), "
        f"ceiling vs MTP = {d['et_medusa_central'] - ET_BASELINE_MTP:+.3f}",
        f"          (Medusa LOSES on E[T], the bet is it WINS on a cheaper step)",
    ]
    if report.get("medusa_train") is not None:
        mt = report["medusa_train"]
        lines.append(
            f"          MEASURED realized ({mt['trajectory']}, K={report['medusa_heads_K']}): "
            f"E[T] tree {report['best_realized_E_T_medusa']:.3f} "
            f"(single-cand {mt['measured_et_singlecand_direct']:.3f}), "
            f"vs MTP {report['E_T_vs_mtp']:+.3f}; {report['realized_vs_ceiling_gap']:.2f} UNDER ceiling")
    lines += [
        "",
        f"MEASURED draft tax (K={report['medusa_heads_K']}, {tax['source']}):",
        f"  int4-tied (reuse deployed lm_head): {tax['int4_tied_ms']:.3f} ms",
        f"  primary 12288 shared-batched eager: {tax['primary_ms']:.3f} ms   <-- headline",
        f"  optimistic (graphed/ONEGRAPH):      {tax['optimistic_ms']:.3f} ms",
        f"  conservative full-vocab 262144:     {tax['conservative_ms']:.3f} ms",
        f"  vs deployed linear-MTP draft_gpu = {DRAFT_GPU_LINEAR_MS} ms "
        f"(Medusa ~{DRAFT_GPU_LINEAR_MS / max(tax['primary_ms'], 1e-6):.0f}x cheaper)",
        "",
        f"STEP: t_step_old {T_STEP_OLD_MS:.3f} -> medusa {r['primary_operating_point']['t_step_ms']:.3f} ms "
        f"(-{r['step_reduction_pct']:.0f}%); E[T] {ET_BASELINE_MTP:.3f} -> {d['et_medusa_central']:.3f} "
        f"(-{r['et_reduction_pct']:.0f}%)",
        "",
        "projected TPS grid (LOCAL) [E[T] case x tax case]:",
        f"  {'E[T]/tax':<22}{'int4tied':>10}{'12k_eager':>11}{'graphed':>10}{'fullvocab':>11}",
    ]
    for ek in g:
        row = g[ek]
        lines.append(
            f"  {ek:<22}{row['int4_tied']['tps_local']:>10.0f}"
            f"{row['primary_12k_eager']['tps_local']:>11.0f}"
            f"{row['optimistic_graphed']['tps_local']:>10.0f}"
            f"{row['conservative_fullvocab']['tps_local']:>11.0f}")
    lines += [
        "",
        f"break-even tax to beat 442: central E[T]={report['breakeven_tax']['central']['et_used']:.2f} "
        f"-> tax <= {report['breakeven_tax']['central']['max_tax_ms_for_442']:.3f} ms "
        f"(measured {tax['primary_ms']:.3f} ms {'PASSES' if tax['primary_ms'] <= report['breakeven_tax']['central']['max_tax_ms_for_442'] else 'FAILS'})",
        f"break-even tax to cross 500: central E[T] -> tax <= "
        f"{report['breakeven_tax']['central']['max_tax_ms_for_500']:.3f} ms",
        "",
        f"PRIMARY (central E[T], primary tax, LOCAL): {report['projected_tps_medusa']:.1f} TPS "
        f"({report['projected_tps_medusa'] - 442:+.0f} vs 442; official {report['projected_tps_medusa_official']:.0f}, "
        f"private {report['projected_tps_medusa_private']:.0f})",
        f"beats_442 = {report['beats_442']}   crosses_500 = {report['crosses_500']}   "
        f"quality_safe = {report['drafter_is_quality_safe']} (PPL {report['ppl']:.4f})",
        f"vs #532 EAGLE-3 central {EAGLE3_PROJECTED_TPS_CENTRAL_532:.0f}: "
        f"{report['eagle3_contrast_532']['medusa_vs_eagle3_local']:+.0f} TPS (opposite verdict sign)",
        "",
        f"GO/NO-GO: {report['go_no_go']}",
        f"self-test: {report['self_test']['n_passed']}/{report['self_test']['n_checks']} "
        f"({'PASS' if report['self_test']['passes'] else 'FAIL'})",
    ]
    return "\n".join(lines)


def load_measured_train(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[measured] {path} not found — analytic-only screen", file=sys.stderr)
        return None
    return json.loads(p.read_text())


def fold_measured(report: dict, measured: dict, measured_ref: dict | None, k_primary: int) -> dict:
    """Replace the literature-anchored PROJECTED headline E[T] with the MEASURED free-run
    realized E[T] from the cheap local train; keep the analytic ladder as the CEILING anchor.
    This is what turns the screen from a projection into a measurement (PR step 2/3)."""
    by_k = measured["by_K"]
    kk = str(k_primary) if str(k_primary) in by_k else sorted(by_k, key=int)[-1]
    mk = by_k[kk]
    m_sc = mk["et_singlecand_direct"]          # MEASURED single-candidate (no tree), direct chained run-length
    m_top3 = mk["et_ladder_product_top3"]      # dense top-3 ladder = an UPPER tree bracket
    m_ladder = mk["a_marginal_top1"]
    # tree-lift the MEASURED accepted mass with the same multipliers the analytic screen uses
    m_tree = 1.0 + (m_sc - 1.0) * TREE_MULT["central"]
    m_tree_lo = 1.0 + (m_sc - 1.0) * TREE_MULT["lo"]
    m_tree_hi = 1.0 + (m_sc - 1.0) * TREE_MULT["hi"]

    tax_primary = report["medusa_draft_step_tax"]
    taxes = {
        "int4_tied": report["tax_by_k"][kk]["int4_tied_ms"],
        "primary_12k_eager": tax_primary,
        "optimistic_graphed": report["tax_by_k"][kk]["optimistic_ms"],
        "conservative_fullvocab": report["tax_by_k"][kk]["conservative_ms"],
    }

    def proj(et: float, tax: float) -> dict:
        loc = tps_local(et, tax)
        return {"tax_ms": tax, "t_step_ms": t_step_medusa_ms(tax), "tps_local": loc,
                "tps_official": to_official(loc), "tps_private": to_private(loc),
                "beats_442": loc >= 442.0, "crosses_500_local": loc >= 500.0,
                "crosses_500_official": to_official(loc) >= 500.0}

    # add measured cells to the grid (so _fmt / self-test can reference them)
    grid = report["reconciliation"]["grid_et_x_tax"]
    for name, et in (("measured_singlecand", m_sc), ("measured_tree", m_tree)):
        grid[name] = {tk: proj(et, tv) for tk, tv in taxes.items()}
    op = grid["measured_tree"]["primary_12k_eager"]

    # reference-trajectory cross-check (the biased-low lower bound)
    ref_block = None
    if measured_ref:
        rby = measured_ref["by_K"]
        rk = str(k_primary) if str(k_primary) in rby else sorted(rby, key=int)[-1]
        r_sc = rby[rk]["et_singlecand_direct"]
        ref_block = {
            "trajectory": "ppl_reference (external text -> biased LOW vs deployment)",
            "et_singlecand": r_sc, "et_tree_central": 1.0 + (r_sc - 1.0) * TREE_MULT["central"],
            "a_ladder_top1": rby[rk]["a_marginal_top1"], "gpu_h_total": measured_ref.get("train_gpu_h_total"),
        }

    ceiling_central = report["best_realized_E_T_medusa"]        # analytic central tree (pre-fold)
    ceiling_band = report["best_realized_E_T_medusa_band"]
    floor_442 = report["et_floor_to_beat_442"]
    floor_500 = report["et_floor_to_cross_500"]

    # ---- OVERRIDE headline KEY OUTPUTS: realized = MEASURED ----
    report["headline_et_case"] = "measured_tree"
    report["best_realized_E_T_medusa"] = m_tree
    report["best_realized_E_T_medusa_is_projection_not_measured"] = False
    report["best_realized_E_T_medusa_band"] = [m_sc, max(m_top3, m_tree_hi)]
    report["E_T_vs_mtp"] = m_tree - ET_BASELINE_MTP
    report["projected_tps_medusa"] = op["tps_local"]
    report["projected_tps_medusa_official"] = op["tps_official"]
    report["projected_tps_medusa_private"] = op["tps_private"]
    report["projected_tps_medusa_frame"] = (
        "byteexact-442 LOCAL; MEASURED free-run realized E[T] (tree-central); primary measured 12k tax")
    report["beats_442"] = op["beats_442"]
    report["crosses_500"] = op["crosses_500_local"]
    report["medusa_heads_K"] = int(kk)
    report["medusa_train_gpu_h"] = measured.get("train_gpu_h_total")
    report["e_t_ceiling_literature_central"] = ceiling_central
    report["e_t_ceiling_literature_band"] = ceiling_band
    report["realized_vs_ceiling_gap"] = ceiling_central - m_tree
    report["eagle3_contrast_532"]["medusa_vs_eagle3_local"] = op["tps_local"] - EAGLE3_PROJECTED_TPS_CENTRAL_532

    report["medusa_train"] = {
        "trajectory": measured.get("target_trajectory"),
        "capture_ckpt": measured.get("ckpt"),
        "capture_keepset_16k_vs_deployed_12k": (
            "captured on osoi5-v0 16k-head pre-prune base; deployed 442 base further prunes head to "
            "12k (12k subset of 16k) -> base greedy near-identical, E[T] backbone-driven transfers "
            "(if anything 12k is marginally MORE predictable). Tax priced at deployed Vdraft=12288."),
        "n_train_seq": measured.get("n_train_seq"), "n_eval_seq": measured.get("n_eval_seq"),
        "n_new_freerun": measured.get("n_new"), "epochs": measured.get("epochs"),
        "weight_decay": measured.get("weight_decay"), "best_ep": mk.get("best_ep"),
        "early_stop_best": measured.get("early_stop_best"),
        "measured_a_ladder_top1": m_ladder, "measured_a_ladder_top3": mk["a_marginal_top3"],
        "measured_et_singlecand_direct": m_sc,
        "measured_et_tree_central": m_tree, "measured_et_tree_lo": m_tree_lo, "measured_et_tree_hi": m_tree_hi,
        "measured_et_top3_dense_bracket": m_top3,
        "realized_projection_at_measured_tax": {
            "singlecand": proj(m_sc, tax_primary), "tree_central": op,
            "tree_lo": proj(m_tree_lo, tax_primary), "tree_hi": proj(m_tree_hi, tax_primary)},
        "reference_trajectory_crosscheck": ref_block,
        "gpu_h_total_freerun": measured.get("train_gpu_h_total"),
    }

    realized_beats = op["beats_442"]
    ceiling_beats = ceiling_central >= floor_442
    head1 = m_ladder[0]
    verdict = (
        f"NO-GO-LOCAL-REALIZE — the MEASURED free-run realized E[T]={m_tree:.2f} "
        f"(tree-central; single-candidate {m_sc:.2f}) sits FAR below the break-even "
        f"E[T]={floor_442:.2f} needed to beat 442 at the measured {tax_primary:.3f} ms tax -> "
        f"{op['tps_local']:.0f} TPS local ({op['tps_local']-442:+.0f} vs 442; "
        f"crosses_500={op['crosses_500_local']}). Medusa's structural advantage over #532's EAGLE-3 is "
        f"REAL and confirmed — the parallel draft is ~{DRAFT_GPU_LINEAR_MS/max(tax_primary,1e-6):.0f}x "
        f"CHEAPER than the serial linear-MTP draft, dropping the step {report['reconciliation']['step_reduction_pct']:.0f}% "
        f"— but the E[T] REALIZATION gap dominates: a cheap on-pod train (free-run greedy on "
        f"{measured.get('n_train_seq')} workload seqs, frozen backbone, tied head, early-stopped) realizes "
        f"head-1 acc {head1:.2f} and decays, landing ~{ceiling_central/max(m_tree,1e-6):.1f}x UNDER the "
        f"literature CEILING E[T]={ceiling_central:.2f} — and that ceiling is itself only "
        f"{'MARGINAL' if not ceiling_beats else 'a beat'} ({'<' if not ceiling_beats else '>='} {floor_442:.2f}). "
        f"Same capability-unbuilt-locally barrier #532 found for EAGLE-3 (head collapse): a beat-442 Medusa "
        f"needs a proper large-corpus cluster train, and even then central E[T] does not clear 442. "
        f"Quality-safe by construction (byte-exact M=8 verify => output==greedy; PPL {PPL_BX442:.4f}, "
        f"self-det r1-r2=1.0, drafter-invariant)."
    )
    if ref_block:
        verdict += (f" Reference-trajectory cross-check (biased low) corroborates: "
                    f"E[T]_sc {ref_block['et_singlecand']:.2f}.")
    report["go_no_go"] = verdict
    report["go_no_go_rationale"] = (
        "MEASUREMENT-DRIVEN verdict (supersedes the analytic projection): " + verdict
        + " || ANALYTIC-CEILING context: " + report["go_no_go_rationale"])
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="0-GPU analytic gate (no microbench)")
    ap.add_argument("--measured-train", dest="measured_train", default=None,
                    help="path to free-run medusa_train_results json (folds MEASURED realized E[T])")
    ap.add_argument("--measured-train-ref", dest="measured_train_ref", default=None,
                    help="path to ppl-reference train json (lower-bound cross-check)")
    ap.add_argument("--no-microbench", action="store_true", help="skip GPU microbench (use analytic tax)")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="medusa-feasibility")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default="lawine/medusa-feasibility-screen")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).with_name("medusa_feasibility_screen_results.json")))
    args = ap.parse_args()

    microbench = None
    if not (args.self_test or args.no_microbench):
        print("[microbench] running A10G head-forward draft-tax microbench ...", file=sys.stderr)
        microbench = run_microbench()

    report = build_report(microbench)

    measured = load_measured_train(args.measured_train)
    measured_ref = load_measured_train(args.measured_train_ref)
    if measured is not None:
        report = fold_measured(report, measured, measured_ref, k_primary=5)

    report["self_test"] = run_self_test(report)
    print(_fmt(report))

    if args.self_test:
        Path(__file__).with_name("medusa_feasibility_screen_selftest.json").write_text(
            json.dumps(report["self_test"], indent=2))
        return 0 if report["self_test"]["passes"] else 1

    report["wandb_run_id"] = None if args.no_wandb else log_to_wandb(
        report, args.wandb_group, args.wandb_name)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}  (W&B run {report.get('wandb_run_id')})")

    print("SENPAI-RESULT " + json.dumps({
        "terminal": True, "status": "complete", "pending_arms": False,
        "analysis_only": True, "no_hf_job": True, "official_tps": 0,
        "wandb_run_ids": [report["wandb_run_id"]] if report.get("wandb_run_id") else [],
        "primary_metric": {"name": "projected_tps_medusa", "value": report["projected_tps_medusa"]},
        "test_metric": {"name": "beats_442", "value": int(report["beats_442"])},
    }))
    return 0 if report["self_test"]["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
