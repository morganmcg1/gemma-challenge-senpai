#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Salvage-walk Python overhead: the LAST un-measured component of the depth-9
verify step denominator (PR #143).

LOCAL only -- pure Python/CUDA timing harness. NO model, NO HF Job, NO submission,
NO kernel build, NO quota.

WHY
---
My #136 (MERGED) firmed the depth-9 verify step at 1.2182 (+0.45% vs the 1.2127
roofline) by proving the eager star-attn launch idle (37 attn_py_calls/step) is
GPU-hidden behind per-layer GEMM -- the step is GPU-bound and the attention CPU
dispatch pipelines ahead. But the ONE component #136 could not measure was the
drafter + salvage-walk Python control flow: openevolve's tree fires 391 salvages +
37 full over 1024 steps, and that tree-expansion / branch-descent / salvage-
selection Python sits in the full-step wall time but NOT in the 37-attn-call
micro-bench. This is the single last unknown in the step denominator, and it is a
build constraint for land #71 (the descending accept-walk kernel).

THE QUESTION
------------
Is the salvage/descent Python GPU-HIDDEN under realistic GEMM overlap (like the
attn idle was) -- in which case land's descent walk pays ~0 step penalty PROVIDED
it stays sync-free -- or does a data-dependent host-sync (.item() / bool(tensor) /
.cpu() / synchronize) SERIALIZE the descent loop, draining the launch queue and
exposing a GPU-idle bubble that inflates the step and raises the operative bar?

METHOD (reuses my #136 isolation-vs-interleaved method verbatim)
----------------------------------------------------------------
Model the control flow as the real sequence of small CUDA launches a descending
accept walk would issue over the static depth-9 / 32-node / max-branch-3 tree at
the oracle's realized rates, in TWO implementation variants:
  * sync_free  -- accept-length + descent path + salvage branch all resolve
                  ON-DEVICE (match-mask -> cumprod -> argmax-first-mismatch ->
                  device gather). NO host-sync. The Python issues launches; the
                  CPU pipelines ahead.
  * sync_bound -- a naive Python while-loop that reads a device value to decide
                  whether to descend (per-node .item()), which branch to salvage
                  (argmax .item()), and the accept length (.item()). One host-sync
                  per descended node + salvage + readout.
Then time each variant TWO ways:
  (a) isolation  -- control flow alone, GPU-starved (the no-overlap UPPER bound).
  (b) interleaved-- with representative per-step GEMM (the step's Marlin gate_up/
                    down + drafter at M=32) issued around the control-flow launches
                    -- the CPU-dispatch-pipelines-ahead test.
idle = event-span - profiler device-busy floor = the exposed GPU-idle the control
flow pays. sync_free interleaved idle -> ~0 (hidden) is GREEN; sync_bound
interleaved idle survives the overlap (the sync drains the filler) -> the
serialized cost land must fuse off the critical path.

Then re-price: step inflation vs the 1.2182 measured anchor and the resulting
clear-500 operative-bar shift (#136 mapping: +dstep -> +dbar; 4.841 roofline /
4.862 measured), checked against the 5.207 supply ceiling. Hand land #71 the sync
constraint.

Primary metric: salvage_walk_step_overhead_pct (per-step % inflation at the
measured anchor, interleaved/overlap regime). Test: salvage_walk_gpu_hidden
(1 iff GPU-hidden under overlap -> bar holds <= 4.86).
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# reuse the exact #136 profiler floor + the banked M=32 tree topology.
from scripts.local_validation.profile_attention import _profiled_device_us  # noqa: E402
from traversal_verify_et import load_m32_topology  # noqa: E402

# ===== #136 / fern compose constants (re-priced, NOT re-derived) ===============
Z95 = 1.959963984540054
K_CAL = 125.26795005202914              # 481.53 / 3.844 (official baseline / E[T]_linear)
STEP_M8_US = 1.0e6 / K_CAL              # ~7982.86 us = 1 M=8-normalized step-unit
STEP_WSTAR_DEPTH9 = 1.2127483746822987  # roofline depth-9 W* step (M=8-norm)
STEP_WSTAR_GEMM = 1.098148338441328     # Marlin staircase M=32 (denken #68)
STEP_WSTAR_DRAFTER_ADD = 0.048          # drafter expansion depth-9 (wirbel)
MEASURED_STEP_136 = 1.2182              # my #136 MEASURED depth-9 anchor (overlap-central)
CLEAR500_BAR_ROOFLINE = 4.840617149792076  # fern #129 operative clear-500 bar @ roofline
E_T_TREE_CEILING = 5.207                # fern #125 / denken #101 supply ceiling (max E[T])
E_T_OPENEVOLVE_ORACLE = 2.621           # openevolve A10G readout (board 100550)
TAU_FERN_CENTRAL = 1.0
TAU_FERN_LOW = 0.9983                   # fern #129 tau band low
TARGET_500 = 500.0
TARGET_530 = 530.0

# ===== oracle readout of tree-488-pw-fp32-v0 (board 20260614-100550-487) =======
ORACLE_CUM_LADDER = [0.674, 0.350, 0.203, 0.131, 0.089, 0.060, 0.037]  # P(spine accepts >= d)
ORACLE_E_T = 2.621
ORACLE_SALVAGES = 391
ORACLE_FULL = 37
ORACLE_STEPS = 1024
ORACLE_DRAFTS = 2417

GEMM_FILLER_N = 2048   # bf16 NxN filler GEMM (matches #136 GEMM_FILLER_N)


# ---------------------------------------------------------------------------
def summarize(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"n": 0}
    mean = statistics.fmean(values)
    median = statistics.median(values)
    std = statistics.pstdev(values) if n > 1 else 0.0
    se = std / math.sqrt(n) if n else 0.0
    return {"n": n, "mean": mean, "median": median, "std": std, "se": se,
            "cv_pct": 100.0 * std / mean if mean else 0.0, "ci95_abs": Z95 * se,
            "min": min(values), "max": max(values)}


def fern_official(e_t: float, step: float, tau: float) -> float:
    """fern #129 compose: official = K_cal * E[T] / step * tau."""
    return K_CAL * e_t / step * tau


def fern_clear_bar(target: float, step: float, tau: float) -> float:
    """E[T] needed to clear `target` official at (step, tau). RISES with step."""
    return target * step / (K_CAL * tau)


# ===== Step 0: faithful model of the descent control flow ======================
def build_tree(parent: list[int]):
    n = len(parent)
    children: list[list[int]] = [[] for _ in range(n)]
    depth = [0] * n
    for i in range(1, n):
        children[parent[i]].append(i)
        depth[i] = depth[parent[i]] + 1
    spine = []  # the rank-1 (first-child) chain from the root = the linear spine
    u = 0
    while True:
        spine.append(u)
        if not children[u]:
            break
        u = children[u][0]
    return children, depth, spine


def build_step_schedule(rng, n_steps: int) -> list[dict]:
    """Replay n_steps of the descent at the oracle's realized rates. Per step:
      descend_len  -- # of spine accept-comparisons (inverse-CDF on the measured
                      cumulative ladder; the # of host-syncs a naive per-node walk
                      pays on the spine).
      salvage      -- a rank>=2 branch rescued a divergence (oracle 391/1024).
      full         -- the walk reached the built depth 9 (oracle 37/1024).
    Reconstructs E[T] = 1 (bonus) + accepted_spine + salvage_rescue ~= 2.621.
    The schedule drives the per-step op/sync COUNT the timing harness replays."""
    ladder = ORACLE_CUM_LADDER
    built_depth = 9
    p_salv = ORACLE_SALVAGES / ORACLE_STEPS
    sched = []
    for _ in range(n_steps):
        # accepted spine tokens k: P(k >= d) = ladder[d-1]; sample via inverse-CDF.
        # E[k] = sum(ladder) = 1.544 -> E[T] = 1 (bonus) + 1.544 + salvage rescue.
        u = rng.random()
        k = 0
        for d in range(1, len(ladder) + 1):
            if u < ladder[d - 1]:
                k = d
            else:
                break
        # full-tree reach == the deep-acceptance tail (k at the ladder max): the
        # oracle's full=37/1024=3.6% ~= P(k>=7)=ladder[6]=3.7% -- so full IS the
        # deepest spine bucket, NOT a separate accepted-count override.
        full = k >= len(ladder)
        salv = (not full) and (rng.random() < p_salv)
        # naive per-node walk op count: traverse to depth (9 if full else k+1 reject
        # check) + 1 extra branch compare on a salvage rescue.
        descend_len = (built_depth if full else k + 1) + (1 if salv else 0)
        sched.append({"accepted_spine": k, "descend_len": descend_len,
                      "salvage": bool(salv), "full": bool(full)})
    return sched


def schedule_stats(sched: list[dict]) -> dict:
    n = len(sched)
    acc = [s["accepted_spine"] for s in sched]
    salv = sum(s["salvage"] for s in sched)
    full = sum(s["full"] for s in sched)
    # E[T] = 1 bonus + accepted spine + salvage rescue. The rescue is SMALL
    # (BUG-2: the walk fires salvage but does not descend) -- the analytic gap
    # 2.621 - (1 + sum(ladder)=2.544) = 0.077 over 391 salvages = 0.197/salvage.
    rescue_per_salvage = (ORACLE_E_T - (1.0 + sum(ORACLE_CUM_LADDER))) * ORACLE_STEPS / ORACLE_SALVAGES
    salvage_rescue = salv * rescue_per_salvage / n
    e_t_recon = 1.0 + statistics.fmean(acc) + salvage_rescue
    return {
        "n_steps": n,
        "mean_accepted_spine": statistics.fmean(acc),
        "mean_descend_len": statistics.fmean(s["descend_len"] for s in sched),
        "salvage_rate": salv / n, "salvage_count": salv,
        "full_rate": full / n, "full_count": full,
        "e_t_reconstructed": e_t_recon,
        "e_t_oracle": ORACLE_E_T,
        "e_t_recon_err_pct": 100.0 * (e_t_recon - ORACLE_E_T) / ORACLE_E_T,
    }


SYNC_POINT_TAXONOMY = [
    {"op": "per-node accept compare (while-descend)",
     "naive": "bool(verify_argmax[u].eq(draft_tok[child]).item())",
     "syncs_per_step": "= descend_len (~2.5 mean)",
     "sync_free_alt": "match_mask = verify_argmax.eq(draft_tok) -> cumprod -> "
                      "argmax-first-mismatch; accept_len stays a DEVICE scalar"},
    {"op": "salvage branch selection",
     "naive": "chosen = int(branch_scores.argmax().item())",
     "syncs_per_step": "= 1 on salvage steps (391/1024 = 0.382)",
     "sync_free_alt": "best = branch_scores.argmax(); gather chosen branch with the "
                      "DEVICE index (no .item())"},
    {"op": "accept-length readout",
     "naive": "n_accept = accept_len.item()",
     "syncs_per_step": "= 1 (every step)",
     "sync_free_alt": "keep accept_len on device; the next step's drafter expand "
                      "indexes the KV/context by the device scalar"},
    {"op": "tree expansion (per-depth bookkeeping)",
     "naive": "static topology -> NO data-dependent branch; top-k/scatter only",
     "syncs_per_step": "= 0 (the rho-optimal tree shape is STATIC; only token "
                       "VALUES are data-dependent)",
     "sync_free_alt": "fixed launch sequence, always pipeline-able"},
]


# ===== control-flow op primitives (small CUDA launches on persistent buffers) ==
class ControlFlowOps:
    """The real sequence of small CUDA launches a descending accept walk issues
    over the 32-node tree. Sizes are tiny (launch-overhead-dominated) ON PURPOSE:
    we are measuring whether the LAUNCHES hide, not GEMM compute (that is the
    filler/denominator)."""

    def __init__(self, n_nodes: int, depth: list[int], spine: list[int],
                 n_expand_depths: int = 9):
        dev = torch.device("cuda")
        self.dev = dev
        self.n_nodes = n_nodes
        self.n_expand = n_expand_depths
        # persistent device buffers (avoid per-step alloc noise).
        self.verify_argmax = torch.randint(0, 256000, (n_nodes,), device=dev, dtype=torch.long)
        self.draft_tok = self.verify_argmax.clone()          # start matching
        self.path_index = torch.tensor(spine + [s for s in range(n_nodes) if s not in spine],
                                       device=dev, dtype=torch.long)[:n_nodes]
        self.branch_scores = torch.randn(8, device=dev)      # rank>=2 candidate scores
        self.scratch = torch.empty(n_nodes, device=dev, dtype=torch.long)
        self.acc_buf = torch.empty(n_nodes, device=dev, dtype=torch.int32)
        # an int "logit" buffer for the per-depth expansion top-k bookkeeping.
        self.draft_logits = torch.randn(n_nodes, 64, device=dev)

    # ---- tree expansion: STATIC topology -> fixed, sync-free launch sequence ----
    def expand(self):
        """Per-depth drafter orchestration bookkeeping (top-k of the drafter logits
        + scatter into the fixed tree buffer). The tree SHAPE is static, so this is
        a fixed launch sequence -- always pipeline-able, no host-sync."""
        for _ in range(self.n_expand):
            # top-k candidate selection over the drafter logits (a small launch).
            vals, idx = self.draft_logits.topk(3, dim=-1)
            # scatter the chosen tokens into the tree token buffer (device gather).
            self.scratch.copy_(self.verify_argmax.index_select(0, self.path_index))

    # ---- descent: SYNC-FREE (accept-length + path resolve on device) -----------
    def descent_sync_free(self, step: dict, terminal_sync: bool = False):
        """vLLM-v1 RejectionSampler pattern (PR #14930, 'zero CPU-GPU sync'):
        accept length is implicit in a device match-mask; the descent path and
        salvage branch resolve via device argmax/gather -- NO per-node .item().
        terminal_sync adds the ONE structurally-unavoidable host transfer
        (output_token_ids.cpu(), vLLM v1 parse_output) that hands accepted tokens
        to the CPU scheduler for KV/stop/streaming -- already present in EVERY
        decode step (so already in the 1.2182 anchor), measured here to show even
        it is hidden behind the GEMM tail."""
        match = self.verify_argmax.eq(self.draft_tok)                 # [N] bool
        path_match = match.index_select(0, self.path_index).to(torch.int32)
        acc = torch.cumprod(path_match, 0)                            # 1 = still accepting
        accept_len = acc.sum()                                        # DEVICE scalar
        accepted = self.draft_tok.index_select(0, self.path_index)    # gather (device)
        if step["salvage"]:
            best = self.branch_scores.argmax()                        # DEVICE scalar
            _ = self.draft_tok.index_select(0, best.unsqueeze(0).clamp_max(self.n_nodes - 1))
        # accept_len + best stay on device; consumed by the next step's expand.
        if terminal_sync:
            _ = accepted.cpu()                                        # *** 1 unavoidable sync ***
        return accept_len, accepted

    # ---- descent: SYNC-BOUND (naive Python while-loop with per-node .item()) ----
    def descent_sync_bound(self, step: dict):
        descend_len = step["descend_len"]
        n_acc = 0
        for j in range(descend_len):
            u = j % self.n_nodes
            hit = self.verify_argmax[u].eq(self.draft_tok[u])         # device bool scalar
            if not bool(hit.item()):                                  # *** HOST-SYNC ***
                break
            n_acc += 1
        if step["salvage"]:
            best = self.branch_scores.argmax()
            _ = int(best.item())                                       # *** HOST-SYNC ***
        # accept-length readout every step.
        _ = self.verify_argmax[0].add(n_acc).item()                   # *** HOST-SYNC ***
        return n_acc


# ===== the per-step control flow, two variants, +/- interleaved filler =========
def make_one_step(ops: ControlFlowOps, sched: list[dict], variant: str,
                  interleaved: bool, filler, gemm_per_op: int):
    """Build a closure that runs ONE scheduled step's control flow. In interleaved
    mode it issues `gemm_per_op` filler GEMMs before each control-flow op-group so
    the GPU has work in flight while the CPU issues the launches (the #136 overlap
    test). Both variants issue the SAME filler + compute launches; sync_bound
    additionally pays the .item() host-syncs."""
    counter = {"i": 0}
    n_sched = len(sched)

    def maybe_filler():
        if interleaved:
            for _ in range(gemm_per_op):
                filler()

    def one_step():
        step = sched[counter["i"] % n_sched]
        counter["i"] += 1
        # tree expansion (static, sync-free in BOTH variants).
        for _ in range(ops.n_expand):
            maybe_filler()
        ops.expand()
        # descent walk.
        maybe_filler()
        if variant == "sync_free":
            ops.descent_sync_free(step, terminal_sync=False)
        elif variant == "sync_free_terminal":
            ops.descent_sync_free(step, terminal_sync=True)
        else:
            ops.descent_sync_bound(step)
    return one_step


def time_regime(one_step, n_passes: int, warmup: int, n_iter: int) -> dict:
    """#136 three-timing: profiler device-busy floor (no gaps) + CUDA-event span
    over N back-to-back steps; idle = span - busy = the exposed GPU-idle the
    control flow pays."""
    device_busy_us = _profiled_device_us(torch, one_step, n_iter, warmup)
    for _ in range(warmup):
        one_step()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(n_passes):
        one_step()
    ev1.record()
    torch.cuda.synchronize()
    span_us = ev0.elapsed_time(ev1) * 1e3 / n_passes
    idle = max(0.0, span_us - device_busy_us)
    return {"device_busy_us": device_busy_us, "span_us": span_us,
            "exposed_idle_us": idle}


def measure_variant(ops, sched, variant, warmup, n_iter, n_passes, filler,
                    gemm_per_op, rounds) -> dict:
    """Measure each regime `rounds` times and take the MEDIAN idle. idle = span -
    device_busy is a small difference of two ~step-sized numbers, so a single
    round sits near the event-timer noise floor; the median over rounds tightens
    it and the spread quantifies the residual noise."""
    iso_step = make_one_step(ops, sched, variant, False, filler, gemm_per_op)
    inter_step = make_one_step(ops, sched, variant, True, filler, gemm_per_op)
    iso_idles, inter_idles, iso_last, inter_last = [], [], None, None
    for _ in range(rounds):
        iso_last = time_regime(iso_step, n_passes, warmup, n_iter)
        inter_last = time_regime(inter_step, max(40, n_passes // 2), warmup,
                                 max(20, n_iter // 2))
        iso_idles.append(iso_last["exposed_idle_us"])
        inter_idles.append(inter_last["exposed_idle_us"])
    idle_iso = statistics.median(iso_idles)
    idle_inter = statistics.median(inter_idles)
    collapse = idle_iso / idle_inter if idle_inter > 1e-9 else float("inf")
    return {"variant": variant, "isolation": iso_last, "interleaved": inter_last,
            "idle_isolation_us": idle_iso, "idle_interleaved_us": idle_inter,
            "idle_isolation_rounds": iso_idles, "idle_interleaved_rounds": inter_idles,
            "idle_isolation_summary": summarize(iso_idles),
            "idle_interleaved_summary": summarize(inter_idles),
            "collapse_factor_iso_over_inter": collapse}


# ===== bar re-pricing (#136 mapping) ===========================================
def price_step(step_overhead_us: float, anchor_step: float) -> dict:
    """Inflate the anchor step by the measured overhead and re-price the clear-500
    operative bar (RISES with step). Anchor = the 1.2182 #136 measured step."""
    dstep_units = step_overhead_us / STEP_M8_US
    step = anchor_step + dstep_units
    bar500 = fern_clear_bar(TARGET_500, step, TAU_FERN_CENTRAL)
    bar500_low = fern_clear_bar(TARGET_500, step, TAU_FERN_LOW)
    bar530 = fern_clear_bar(TARGET_530, step, TAU_FERN_CENTRAL)
    return {
        "overhead_us": step_overhead_us,
        "dstep_units": dstep_units,
        "step_inflation_pct": 100.0 * dstep_units / anchor_step,
        "inflated_step": step,
        "clear500_bar_central": bar500,
        "clear500_bar_taulow": bar500_low,
        "clear530_bar_central": bar530,
        "clear500_bar_shift_vs_measured": bar500 - fern_clear_bar(TARGET_500, anchor_step, TAU_FERN_CENTRAL),
        "clear500_under_ceiling": bar500 <= E_T_TREE_CEILING,
        "clear500_taulow_under_ceiling": bar500_low <= E_T_TREE_CEILING,
        "clear500_holds_4p86": bar500 <= 4.862 + 1e-6,
    }


# ===== driver ==================================================================
def run(args) -> dict:
    assert torch.cuda.is_available(), "CUDA required"
    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    import numpy as np
    rng = np.random.default_rng(args.seed)

    parent = load_m32_topology()
    children, depth, spine = build_tree(parent)
    n_nodes = len(parent)
    sched = build_step_schedule(rng, ORACLE_STEPS)
    sstats = schedule_stats(sched)

    res: dict = {
        "pr": 143, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "l2_bytes": torch.cuda.get_device_properties(0).L2_cache_size,
        "n_nodes": n_nodes, "max_depth": max(depth), "spine_len": len(spine),
        "anchors": {
            "k_cal": K_CAL, "step_m8_us": STEP_M8_US,
            "step_roofline_depth9": STEP_WSTAR_DEPTH9,
            "measured_step_136": MEASURED_STEP_136,
            "clear500_bar_roofline": CLEAR500_BAR_ROOFLINE,
            "clear500_bar_measured_136": fern_clear_bar(TARGET_500, MEASURED_STEP_136, 1.0),
            "e_t_tree_ceiling": E_T_TREE_CEILING,
            "e_t_openevolve_oracle": E_T_OPENEVOLVE_ORACLE,
            "oracle": {"cum_ladder": ORACLE_CUM_LADDER, "e_t": ORACLE_E_T,
                       "salvages": ORACLE_SALVAGES, "full": ORACLE_FULL,
                       "steps": ORACLE_STEPS, "drafts": ORACLE_DRAFTS},
        },
        "config": {"seed": args.seed, "n_iter": args.n_iter, "warmup": args.warmup,
                   "n_passes": args.n_passes, "gemm_filler_n": args.gemm_filler_n,
                   "filler_scale": args.filler_scale},
        "step0_model": {
            "schedule_stats": sstats,
            "sync_point_taxonomy": SYNC_POINT_TAXONOMY,
            "note": ("descent control flow modeled at oracle realized rates over the "
                     "static depth-9/32-node/max-branch-3 topology; expansion is "
                     "sync-free (static tree shape); the data-dependent host-syncs "
                     "live ONLY in the descent + salvage + accept-length readout."),
        },
    }
    print(f"[salvage] GPU {res['gpu']} nodes={n_nodes} depth={max(depth)} "
          f"step_M8={STEP_M8_US:.1f}us anchor={MEASURED_STEP_136}", flush=True)
    print(f"[salvage] Step0 schedule: E[T]_recon={sstats['e_t_reconstructed']:.3f} "
          f"(oracle {ORACLE_E_T}, err {sstats['e_t_recon_err_pct']:+.1f}%)  "
          f"mean_descend={sstats['mean_descend_len']:.2f}  "
          f"salvage={sstats['salvage_rate']*100:.1f}%  full={sstats['full_rate']*100:.1f}%",
          flush=True)

    ops = ControlFlowOps(n_nodes, depth, spine)

    # ---- size the interleaved filler to the step's real GPU work (#136 method) -
    a = torch.randn(args.gemm_filler_n, args.gemm_filler_n, dtype=torch.bfloat16, device=dev)
    b = torch.randn(args.gemm_filler_n, args.gemm_filler_n, dtype=torch.bfloat16, device=dev)
    c = torch.empty(args.gemm_filler_n, args.gemm_filler_n, dtype=torch.bfloat16, device=dev)

    def filler():
        torch.mm(a, b, out=c)

    filler_us_each = _profiled_device_us(torch, filler, args.n_iter, args.warmup)
    # total per-step GPU work to overlap = the step's GEMM + drafter (~1.146 units).
    step_gpu_us = (STEP_WSTAR_GEMM + STEP_WSTAR_DRAFTER_ADD) * STEP_M8_US * args.filler_scale
    # control-flow op-groups per step the filler is distributed across:
    #   n_expand expansion groups + 1 descent group.
    n_op_groups = ops.n_expand + 1
    gemm_per_op = max(1, round(step_gpu_us / (filler_us_each * n_op_groups)))
    res["filler"] = {"gemm_n": args.gemm_filler_n, "filler_us_each": filler_us_each,
                     "step_gpu_us_target": step_gpu_us, "n_op_groups": n_op_groups,
                     "gemm_per_op": gemm_per_op,
                     "realized_filler_us_per_step": filler_us_each * gemm_per_op * n_op_groups}
    print(f"[salvage] filler {args.gemm_filler_n}^3 = {filler_us_each:.0f}us each; "
          f"{gemm_per_op}/op-group x {n_op_groups} groups = "
          f"{res['filler']['realized_filler_us_per_step']:.0f}us/step GPU work", flush=True)

    # ---- measure all variants both regimes ------------------------------------
    measured = {}
    for variant in ("sync_free", "sync_free_terminal", "sync_bound"):
        m = measure_variant(ops, sched, variant, args.warmup, args.n_iter,
                            args.n_passes, filler, gemm_per_op, args.rounds)
        measured[variant] = m
        print(f"   [{variant}] isolation idle={m['idle_isolation_us']:.1f}us/step  "
              f"interleaved idle={m['idle_interleaved_us']:.1f}us/step  "
              f"collapse={m['collapse_factor_iso_over_inter']:.1f}x  "
              f"(inter rounds {[round(x,1) for x in m['idle_interleaved_rounds']]})", flush=True)
    # per-sync cost: the clean signal (terminal/bound deltas over the sync-free floor).
    sf_floor = measured["sync_free"]["idle_interleaved_us"]
    syncs_per_step_bound = sstats["mean_descend_len"] + sstats["salvage_rate"] + 1.0
    per_sync_terminal = measured["sync_free_terminal"]["idle_interleaved_us"] - sf_floor
    per_sync_bound = (measured["sync_bound"]["idle_interleaved_us"] - sf_floor) / max(1e-9, syncs_per_step_bound)
    res["step1_measured"] = measured
    res["step1_per_sync_cost"] = {
        "sync_free_floor_us": sf_floor,
        "terminal_sync_cost_us": per_sync_terminal,
        "syncs_per_step_sync_bound": syncs_per_step_bound,
        "per_sync_cost_sync_bound_us": per_sync_bound,
        "note": ("per-sync cost from the clean variant deltas (terminal=+1 sync, "
                 "sync_bound=+descend+salvage+readout syncs) over the sync-free floor; "
                 "robust vs the span-busy subtraction noise on the absolute floor."),
    }
    print(f"   [per-sync] terminal(+1)={per_sync_terminal:.1f}us  "
          f"sync_bound(~{syncs_per_step_bound:.1f}/step)={per_sync_bound:.1f}us/sync", flush=True)

    # ---- Step 2: step inflation + operative-bar shift -------------------------
    # the CREDIBLE interleaved overhead for each variant (the realistic regime).
    sf_inter = measured["sync_free"]["idle_interleaved_us"]
    sb_inter = measured["sync_bound"]["idle_interleaved_us"]
    sf_iso = measured["sync_free"]["idle_isolation_us"]
    sb_iso = measured["sync_bound"]["idle_isolation_us"]
    sft_inter = measured["sync_free_terminal"]["idle_interleaved_us"]
    sft_iso = measured["sync_free_terminal"]["idle_isolation_us"]
    pricing = {
        "sync_free_interleaved": price_step(sf_inter, MEASURED_STEP_136),
        "sync_free_terminal_interleaved": price_step(sft_inter, MEASURED_STEP_136),
        "sync_bound_interleaved": price_step(sb_inter, MEASURED_STEP_136),
        "sync_free_isolation": price_step(sf_iso, MEASURED_STEP_136),
        "sync_free_terminal_isolation": price_step(sft_iso, MEASURED_STEP_136),
        "sync_bound_isolation": price_step(sb_iso, MEASURED_STEP_136),
    }
    res["step2_bar_shift"] = pricing
    sf = pricing["sync_free_interleaved"]
    sft = pricing["sync_free_terminal_interleaved"]
    sb = pricing["sync_bound_interleaved"]
    print(f"   [2] sync_free interleaved: +{sf['step_inflation_pct']:.2f}% step -> "
          f"bar {sf['clear500_bar_central']:.4f} (hold4.86={int(sf['clear500_holds_4p86'])})  "
          f"| +1 terminal sync: +{sft['step_inflation_pct']:.2f}% -> bar {sft['clear500_bar_central']:.4f}  "
          f"| sync_bound: +{sb['step_inflation_pct']:.2f}% -> bar {sb['clear500_bar_central']:.4f}",
          flush=True)

    # ---- Step 3: gate + hand land the sync constraint -------------------------
    # GPU-hidden iff the sync-free descent launches pipeline behind the GEMM: the
    # interleaved idle COLLAPSES vs isolation AND the residual is a sub-1% fraction
    # of the step (#136 precedent: the 43us hidden attn idle was deemed GPU-bound).
    hidden_thresh_us = args.hidden_threshold_us
    collapse_sf = measured["sync_free"]["collapse_factor_iso_over_inter"]
    sync_free_hidden = (sf_inter <= hidden_thresh_us
                        and collapse_sf >= args.collapse_min
                        and sf["step_inflation_pct"] < 1.0
                        and sf["clear500_under_ceiling"])
    sync_bound_serializes = sb_inter > hidden_thresh_us
    # primary = the realistic build's overhead. land #71 builds the descending walk;
    # the achievable build is sync-free, so the PRIMARY overhead is sync-free
    # interleaved. gpu_hidden = 1 iff that path is hidden (bar holds).
    primary_pct = sf["step_inflation_pct"]
    gpu_hidden = int(sync_free_hidden)

    if sync_free_hidden and sync_bound_serializes:
        verdict = "GREEN"
        verdict_reason = (
            f"sync-free descent is GPU-HIDDEN under interleaved overlap: idle collapses "
            f"{collapse_sf:.0f}x ({measured['sync_free']['idle_isolation_us']:.0f}us isolation "
            f"-> {sf_inter:.0f}us interleaved = +{sf['step_inflation_pct']:.2f}% step, at the "
            f"event-timer floor like #136's 43us hidden attn idle). bar holds "
            f"{sf['clear500_bar_central']:.3f} ~= the 4.862 anchor, far under the 5.207 "
            f"ceiling. land's descending accept-walk pays ~0 step penalty PROVIDED it stays "
            f"sync-free. The naive sync-bound path serializes ({sb_inter:.0f}us/step = "
            f"+{sb['step_inflation_pct']:.2f}% -> bar {sb['clear500_bar_central']:.3f}); that is "
            f"the cost of getting it wrong (per-node .item()).")
    elif sync_free_hidden and not sync_bound_serializes:
        verdict = "GREEN"
        verdict_reason = (
            f"sync-free descent GPU-hidden (idle {sf_inter:.0f}us, collapse {collapse_sf:.0f}x, "
            f"bar {sf['clear500_bar_central']:.3f}); even the sync-bound path stays small "
            f"({sb_inter:.0f}us) -> descent overhead is not load-bearing either way.")
    elif not sync_free_hidden and sf["clear500_under_ceiling"]:
        verdict = "AMBER"
        verdict_reason = (
            f"sync-free descent is only PARTIALLY hidden (idle {sf_inter:.1f}us/step = "
            f"+{sf['step_inflation_pct']:.2f}% -> bar {sf['clear500_bar_central']:.3f}); "
            f"stays under the 5.207 ceiling but no longer at 4.86. Bracket "
            f"[sync_free {sf_inter:.0f}us, sync_bound {sb_inter:.0f}us]; the exact sync "
            f"points to avoid are enumerated in step0.sync_point_taxonomy.")
    else:
        verdict = "RED"
        verdict_reason = (
            f"the descent control flow serializes (sync-free idle {sf_inter:.1f}us, "
            f"sync-bound {sb_inter:.0f}us = +{sb['step_inflation_pct']:.2f}% step -> bar "
            f"{sb['clear500_bar_central']:.3f}); hand land the FUSED-KERNEL requirement: "
            f"the descent + accept-length must resolve on-device in one launch to keep "
            f"it off the critical path.")

    land_constraint = {
        "rule": ("keep the accept-length readout + descent path + salvage branch "
                 "selection SYNC-FREE: no per-branch .item()/bool(tensor)/.cpu(); "
                 "resolve accept_len as a DEVICE scalar (match-mask -> cumprod -> "
                 "argmax-first-mismatch) and gather accepted tokens by the device "
                 "index -- the vLLM-v1 RejectionSampler pattern (PR #14930, 'zero "
                 "CPU-GPU sync'). The next step's drafter expand must consume the "
                 "device accept_len without a host readout."),
        "if_violated": (f"a naive Python descent (.item() per node + salvage + readout, "
                        f"~{sstats['mean_descend_len'] + 1.4:.1f} syncs/step) costs "
                        f"{sb_inter:.0f}us/step = +{sb['step_inflation_pct']:.2f}% step -> "
                        f"clear-500 bar rises {fern_clear_bar(TARGET_500, MEASURED_STEP_136, 1.0):.3f}"
                        f" -> {sb['clear500_bar_central']:.3f}."),
        "unavoidable_terminal_sync": (
            f"exactly ONE host-sync per step is structurally unavoidable: the terminal "
            f"output_token_ids.cpu() that hands accepted tokens to the CPU scheduler "
            f"(KV mgmt / stop check / streaming; vLLM v1 parse_output). It is ALREADY "
            f"in the 1.2182 anchor (every decode step streams). Measured marginal cost "
            f"{sft_inter:.1f}us/step = +{sft['step_inflation_pct']:.2f}% -> bar "
            f"{sft['clear500_bar_central']:.3f}: hidden behind the GEMM tail. land does "
            f"NOT need to fuse this one away."),
        "fused_kernel_required": bool(verdict == "RED"),
        "sync_points_to_avoid": [t["op"] for t in SYNC_POINT_TAXONOMY if "0" not in t["syncs_per_step"][:4]],
        "research_basis": ("vLLM v1 RejectionSampler (PR #14930 zero-sync); SpecInfer "
                           "(arXiv 2305.09781) + Sequoia (arXiv 2402.12374) single-pass "
                           "tree verify + device accept-mask; CUDA launch ~5-6us/launch, "
                           ".item() bubble ~10-50us (TaxBreak arXiv 2603.12465)."),
    }
    res["step3_gate"] = {
        "verdict": verdict, "reason": verdict_reason,
        "sync_free_gpu_hidden": int(sync_free_hidden),
        "sync_bound_serializes": int(sync_bound_serializes),
        "hidden_threshold_us": hidden_thresh_us,
        "land_constraint": land_constraint,
        "rule": ("GREEN = sync-free descent GPU-hidden under overlap (bar holds "
                 "<=4.86), hand land the stay-sync-free rule; AMBER = partially "
                 "hidden / depends on a sync land hasn't committed to, report bracket "
                 "+ sync points; RED = unavoidable host-sync serializes, hand land the "
                 "fused-kernel requirement"),
    }

    # ---- primary / test -------------------------------------------------------
    res["primary_metric"] = {"name": "salvage_walk_step_overhead_pct", "value": primary_pct}
    res["test_metric"] = {"name": "salvage_walk_gpu_hidden", "value": gpu_hidden}
    res["verdict"] = verdict
    res["elapsed_s"] = time.time() - t0
    res["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9

    del a, b, c
    torch.cuda.empty_cache()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2))
    print(f"\n[salvage] VERDICT={verdict}  primary salvage_walk_step_overhead_pct="
          f"{primary_pct:.3f}%  test salvage_walk_gpu_hidden={gpu_hidden}", flush=True)
    print(f"[salvage] {verdict_reason}", flush=True)
    print(f"[salvage] wrote {out_path} ({res['elapsed_s']:.0f}s, peak "
          f"{res['peak_gpu_gb']:.2f}GB)", flush=True)

    # ---- W&B ------------------------------------------------------------------
    if args.wandb_group and not args.no_wandb:
        try:
            import wandb
            run_w = wandb.init(
                project="gemma-challenge-senpai", entity="wandb-applied-ai-team",
                group=args.wandb_group, name=args.wandb_name,
                config={**res["config"], **res["anchors"], "gpu": res["gpu"]})
            log = {
                "salvage_walk_step_overhead_pct": primary_pct,
                "salvage_walk_gpu_hidden": gpu_hidden,
                "sync_free_idle_interleaved_us": sf_inter,
                "sync_free_idle_isolation_us": sf_iso,
                "sync_free_terminal_idle_interleaved_us": sft_inter,
                "sync_free_terminal_idle_isolation_us": sft_iso,
                "sync_bound_idle_interleaved_us": sb_inter,
                "sync_bound_idle_isolation_us": sb_iso,
                "sync_free_step_inflation_pct": sf["step_inflation_pct"],
                "sync_free_terminal_step_inflation_pct": sft["step_inflation_pct"],
                "sync_bound_step_inflation_pct": sb["step_inflation_pct"],
                "sync_free_clear500_bar": sf["clear500_bar_central"],
                "sync_free_terminal_clear500_bar": sft["clear500_bar_central"],
                "sync_bound_clear500_bar": sb["clear500_bar_central"],
                "clear500_bar_measured_anchor": fern_clear_bar(TARGET_500, MEASURED_STEP_136, 1.0),
                "clear500_bar_roofline": CLEAR500_BAR_ROOFLINE,
                "supply_ceiling_e_t": E_T_TREE_CEILING,
                "measured_step_anchor": MEASURED_STEP_136,
                "sync_free_collapse_factor": collapse_sf,
                "sync_bound_collapse_factor": measured["sync_bound"]["collapse_factor_iso_over_inter"],
                "terminal_sync_cost_us": per_sync_terminal,
                "per_sync_cost_sync_bound_us": per_sync_bound,
                "syncs_per_step_sync_bound": syncs_per_step_bound,
                "sync_free_inflated_step": sf["inflated_step"],
                "sync_bound_inflated_step": sb["inflated_step"],
                "e_t_reconstructed": sstats["e_t_reconstructed"],
                "schedule_salvage_rate": sstats["salvage_rate"],
                "schedule_full_rate": sstats["full_rate"],
                "filler_us_per_step": res["filler"]["realized_filler_us_per_step"],
                "verdict_green": int(verdict == "GREEN"),
                "verdict_amber": int(verdict == "AMBER"),
                "verdict_red": int(verdict == "RED"),
            }
            wandb.log(log)
            run_w.summary.update(log)
            res["wandb_run_id"] = run_w.id
            wandb.finish()
            print(f"[salvage] W&B run {run_w.id} (group {args.wandb_group})", flush=True)
            out_path.write_text(json.dumps(res, indent=2))
        except Exception as e:  # noqa: BLE001
            print(f"[salvage] W&B logging skipped: {e!r}", flush=True)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-iter", type=int, default=80, help="profiler self-time iters")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--n-passes", type=int, default=300, help="event-span passes (isolation)")
    ap.add_argument("--rounds", type=int, default=5, help="repeat each regime, take median idle")
    ap.add_argument("--gemm-filler-n", type=int, default=GEMM_FILLER_N)
    ap.add_argument("--filler-scale", type=float, default=1.0,
                    help="scale the per-step filler GPU work (1.0 = real step work)")
    ap.add_argument("--hidden-threshold-us", type=float, default=60.0,
                    help="interleaved idle <= this (us/step) counts as GPU-hidden "
                         "(#136 precedent: 43us hidden attn idle)")
    ap.add_argument("--collapse-min", type=float, default=5.0,
                    help="min isolation/interleaved idle collapse to count as hidden")
    ap.add_argument("--seed", type=int, default=143)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "research/spec_cost_model/salvage_walk_overhead.json")
    ap.add_argument("--wandb-group", type=str, default="salvage-walk-overhead")
    ap.add_argument("--wandb-name", type=str, default="lawine/salvage-walk-overhead")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--quick", action="store_true", help="fast smoke (few passes)")
    args = ap.parse_args(argv)
    if args.quick:
        args.n_iter, args.warmup, args.n_passes, args.rounds = 20, 10, 40, 2
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
