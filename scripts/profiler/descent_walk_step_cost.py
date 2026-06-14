#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Descent-walk step cost: does the salvage-descend accept-prep stay step-neutral
(does land #71's ACTUAL build hold the launch-realized 1.2182)? (PR #173)

LOCAL A10G profiling ONLY -- pure Python/CUDA/Triton timing harness. NO model,
NO HF Job, NO submission, NO served-file change, NO kernel deploy, NO quota.
BASELINE stays 481.53; greedy/PPL untouched. Adds 0 TPS -- it BOUNDS whether the
descent kernel holds the launch-realized step.

WHY
---
My #161 (MERGED) proved the depth-1 SPINE fix (BUG-1) is step-neutral and #168
collapsed the four step anchors to the single launch-realized step 1.2182. But
both priced the step against the CURRENT strictly-linear accept-prep kernel
(`_dixie_fused_accept_prep_kernel`, break-on-mismatch). land #71 replaces it with
a SALVAGE-DESCEND kernel that, on a mismatch, does MORE work: it walks siblings /
descends instead of breaking (wirbel #135 BUG-2 structure; the descent-ordered DFS
over the single corrected `target_logits_indices` map, wirbel #165). ubel #163's
host-residency sweep carried `descent_accept_walk` as "+0 net (GPU-hidden by
design)" -- a design ASSUMPTION (op 5 of the inventory), NOT a measured bound.

THE QUESTION
------------
Does the salvage-descend accept-prep stay step-neutral, or does the extra
per-mismatch sibling-walk / descent add device-busy cost that lifts the step above
1.2182? This is the DESCENT analog of #161's SPINE step-neutrality measurement --
same method (paired device-busy marginal of a kernel-resident worst-case variant),
new target (the descend walk). It confirms (or corrects) ubel #163's "+0 net by
design" claim with a MEASURED number, and pins the step land #71's actual build
holds for fern #167's launch packet.

METHOD (reuses #161's paired-with-filler-GEMM device-busy method VERBATIM)
--------------------------------------------------------------------------
1. SELF-TEST (PRIMARY): re-run #161's measure_overlap_hidden_idle on THIS A10G and
   confirm the depth-9 step recomposes to 1.2182 within tolerance. Gates the rig
   (identical control to #161's 1.21792 reproduction).
2. ACCEPT-PREP MICRO-BENCH: time the REAL linear break kernel (`accept_prep_baseline`,
   replicated verbatim from the served `_dixie_fused_accept_prep_kernel`) against a
   kernel-resident WORST-CASE `accept_prep_descend_walk` variant that does the
   salvage-descend work -- on a mismatch it loads the node's descent-ordered index
   (the corrected `target_logits_indices` deref, #165), walks max-branch siblings,
   selects the best-scoring, and DESCENDS instead of breaking -- on EVERY node of
   the static M=32 / depth-9 / max-branch-3 tree (the adversarial all-mismatch-
   descend case). Each is ONE conc=1 launch; measured paired (per round, back-to-
   back) so the device-busy marginal cancels common-mode event-timer drift.
3. PROPAGATE: descent-kernel step = 1.2182 (launch-realized, #168) + the descend-
   walk marginal (if practically non-zero). official = K_cal*E[T]/step*tau for the
   descent-only (E[T]=5.0564) and both-bugs (5.2070) regimes; confirm both clear 500.

Primary metric: descent_walk_step_self_test_passes (rig reproduces 1.2182 AND the
                officials hold at the descent-kernel step AND NaN-clean).
Test metric:    descent_walk_step_delta_pct (% step inflation from the descend walk).
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

# Reuse the EXACT #136/#143/#161 compose constants + timing primitives (NOT re-derived).
from salvage_walk_overhead import (  # noqa: E402
    K_CAL, STEP_M8_US, STEP_WSTAR_DEPTH9, MEASURED_STEP_136, E_T_TREE_CEILING,
    TAU_FERN_CENTRAL, TAU_FERN_LOW, TARGET_500, TARGET_530, GEMM_FILLER_N,
    Z95, summarize, fern_official, fern_clear_bar,
)
from scripts.local_validation.profile_attention import _profiled_device_us  # noqa: E402
from traversal_verify_et import load_m32_topology  # noqa: E402

# ===== imported anchors (committed; NOT re-derived) ============================
# wirbel #135 bug2_salvage_descent_results.json + #161 both_bugs_step_cost.json
E_T_DESCENT_ONLY = 5.056404568844709     # bug-2 fixed, depth-1=0.679 (descent-only)
E_T_BOTH_BUGS = 5.206954309441963        # both bugs fixed (single corrected map, #165)
DEPTH1_DESCENT = 0.679
DEPTH1_BOTH = 0.728739760479042
# deployed rising-spine conditional accept ladder (both bugs fixed) -- #161 RISING_SPINE
RISING_SPINE = [0.728739760479042, 0.7589764102641635, 0.7924989076194682,
                0.821702519412012, 0.8342716929825772, 0.8352594665096346,
                0.8472621220149911]
ROOFLINE_STEP = STEP_WSTAR_DEPTH9        # 1.2127483746822987 (#136 graphed floor / roofline edge)

# #168 launch-realized officials at step 1.2182 (the numbers the launch must hold).
OFFICIAL_DESCENT_REALIZED = 519.96       # K_cal*5.0564/1.2182
OFFICIAL_BOTH_REALIZED = 535.44          # K_cal*5.2070/1.2182
OFFICIAL_DESCENT_ROOFLINE = 522.29       # K_cal*5.0564/1.2127 (optimistic band edge)
OFFICIAL_BOTH_ROOFLINE = 537.84          # K_cal*5.2070/1.2127
# #161 self-test reproduced the anchor at 1.217916316194892 (the control to re-hit).
PR161_STEP_REPRODUCED = 1.217916316194892

# static descent tree (wirbel #135 / #143 topology): depth-9 spine, max-branch-3, M=32.
MAX_BRANCH = 3

# self-test tolerance on the 1.2182 reproduction (#136 +0.45%, #161 rig +0.43% -> a
# 1.5% band rejects an order-of-magnitude-broken rig but passes the real overlap idle).
SELFTEST_TOL_PCT = 1.5

# PRACTICAL step-neutrality floor (#161): the paired device-busy marginal is the robust
# common-mode-cancelled GPU-work signal. It is so repeatable that a sub-100ns marginal can
# sit outside its own tiny CI yet be PHYSICALLY NIL: 0.10% step ~= 0.5 TPS on the ~535
# official (dofficial/dstep ~= -440 TPS/step-unit). 0.10% of the 1.2182 step = 9.72 us of
# marginal device-busy -- a single conc=1 accept-prep launch (even the heavy descend DFS)
# is microseconds, GPU-hidden behind the ~9150us weight-GEMM step. So the verdict gates on
# PRACTICAL magnitude, NOT statistical within-CI (the CI is a reported diagnostic).
NEUTRAL_STEP_PCT = 0.10


# ===== kernels: the REAL served linear break kernel + the worst-case descend walk =
def _build_kernels():
    import triton
    import triton.language as tl

    # LINEAR BREAK (served `_dixie_fused_accept_prep_kernel`, replicated verbatim from
    # submissions/fa2sw_precache_kenyan/sitecustomize.py:921). Walks the spine left->right;
    # on the first mismatch `rejected` latches True and the `if not rejected` guard stops
    # all further work -- strictly linear, break-on-mismatch.
    @triton.jit(do_not_specialize=["max_spec_len"])
    def accept_prep_baseline(
        output_token_ids_ptr, next_token_ids_ptr, valid_counts_ptr,
        cu_num_draft_tokens_ptr, draft_token_ids_ptr, target_argmax_ptr,
        bonus_token_ids_ptr, max_spec_len,
    ):
        req_idx = tl.program_id(0)
        start_idx = 0
        if req_idx != 0:
            start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
        end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
        num_draft_tokens = end_idx - start_idx

        rejected = False
        valid_count = 0
        next_token_id = tl.load(bonus_token_ids_ptr + req_idx).to(tl.int32)
        row_offset = req_idx * (max_spec_len + 1)
        for pos in range(num_draft_tokens):
            if not rejected:
                draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
                target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos).to(tl.int32)
                rejected = draft_token_id != target_argmax_id
                valid_count = pos + 1
                next_token_id = target_argmax_id
                tl.store(output_token_ids_ptr + row_offset + pos, target_argmax_id)
        if not rejected:
            bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx).to(tl.int32)
            valid_count = num_draft_tokens + 1
            next_token_id = bonus_token_id
            tl.store(output_token_ids_ptr + row_offset + num_draft_tokens, bonus_token_id)
        tl.store(next_token_ids_ptr + req_idx, next_token_id)
        tl.store(valid_counts_ptr + req_idx, valid_count)

    # SALVAGE-DESCEND WORST CASE: land #71's replacement. The strictly-linear break is
    # replaced by a descent-ordered DFS over the static tree. At each visited node it does
    # THREE things the linear break skips:
    #   (1) INDIRECTION -- deref the node's slot through the single corrected
    #       `target_logits_indices` map (#165): +1 indirection load PER NODE vs the linear
    #       kernel's direct start_idx+pos.
    #   (2) ANCESTOR-VALIDITY AND-REDUCTION -- a tree node may commit ONLY if its WHOLE
    #       ancestor chain accepted (SpecInfer/Medusa full-tree verify; traversal_verify_et
    #       walk_leaf_to_root: `full[u] = matches[u] AND full[parent[u]]`). The linear spine
    #       gets this for free (one chain), but a DFS must AND the ancestor statuses. The
    #       worst case re-walks MAX_ANC (= tree max-depth) ancestor statuses PER NODE -- the
    #       heaviest revalidation land #71 could write (a clever O(1) carry-down DFS would be
    #       lighter; this strictly OVER-bounds it). This is the dominant term the first cut
    #       of this kernel under-counted.
    #   (3) SALVAGE -- on a mismatch, load all MAX_BRANCH sibling slots, select the
    #       best-scoring (the rank>=2 rescue), reseed, and DESCEND instead of breaking.
    # Walks `reach` nodes; the adversarial all-mismatch-descend input sets reach=N_NODES and
    # forces a mismatch (=> salvage) at EVERY node -- the most device work the salvage-descend
    # could ever pay in one step. This OVER-models land #71's real kernel (which descends only
    # the realized path + rescues, with cheaper ancestor carry), so if even THIS is GPU-hidden,
    # the real descend walk certainly is. (#163 op-5 "+0 net by design" -> MEASURED.)
    @triton.jit
    def accept_prep_descend_walk(
        output_token_ids_ptr, next_token_ids_ptr, valid_counts_ptr,
        descent_index_ptr, sibling_index_ptr, ancestor_slot_ptr, status_ptr,
        draft_token_ids_ptr, target_argmax_ptr, bonus_token_ids_ptr, reach_ptr,
        N_NODES: tl.constexpr, MAX_BRANCH: tl.constexpr, MAX_ANC: tl.constexpr,
    ):
        req_idx = tl.program_id(0)
        valid_count = 0
        next_token_id = tl.load(bonus_token_ids_ptr + req_idx).to(tl.int32)
        reach = tl.load(reach_ptr + req_idx)
        for node in range(N_NODES):
            if node < reach:
                # (1) corrected target_logits_indices indirection (#165 descent-ordered map):
                # +1 indirection load per node vs the linear kernel's direct base.
                idx = tl.load(descent_index_ptr + node)
                draft_token_id = tl.load(draft_token_ids_ptr + node)
                target_argmax_id = tl.load(target_argmax_ptr + idx).to(tl.int32)
                mism = draft_token_id != target_argmax_id
                # (2) ANCESTOR-VALIDITY AND-REDUCTION: a node commits only if its whole
                # ancestor chain accepted. MAX_ANC parameterizes the cost model: MAX_ANC=1 with
                # ancestor_slot=[parent] is the FAITHFUL O(1) carry (one parent-status AND, exactly
                # walk_leaf_to_root's `full[u]=match[u] AND full[parent]`); MAX_ANC=max_depth with
                # the full chain is the NAIVE ceiling (re-walk every ancestor, the heaviest a DFS
                # could pay). Both load ancestor status from status_ptr (written earlier in the walk).
                anc_valid = 1
                for d in range(MAX_ANC):
                    anc_node = tl.load(ancestor_slot_ptr + node * MAX_ANC + d)
                    anc_st = tl.load(status_ptr + anc_node)
                    anc_valid = anc_valid & anc_st
                # (3) SALVAGE-DESCEND: on a mismatch walk MAX_BRANCH siblings + pick best
                # (the rank>=2 rescue) and descend -- the work the linear break SKIPS.
                if mism:
                    best_id = target_argmax_id
                    for s in range(MAX_BRANCH):
                        sib_slot = tl.load(sibling_index_ptr + node * MAX_BRANCH + s)
                        sib_argmax = tl.load(target_argmax_ptr + sib_slot).to(tl.int32)
                        take = sib_argmax > best_id
                        best_id = tl.where(take, sib_argmax, best_id)
                    target_argmax_id = best_id            # salvaged sibling reseeds, descend continues
                # commit status (drives the next node's ancestor AND); node accepts iff it
                # matched AND its ancestors were valid.
                accepted = tl.where(mism, 0, anc_valid)
                tl.store(status_ptr + idx, accepted)
                next_token_id = target_argmax_id
                valid_count = node + 1
                tl.store(output_token_ids_ptr + node, target_argmax_id)
        tl.store(next_token_ids_ptr + req_idx, next_token_id)
        tl.store(valid_counts_ptr + req_idx, valid_count)

    return accept_prep_baseline, accept_prep_descend_walk


# ===== descent-ordered tree layout (the single corrected target_logits_indices map) =
def build_descent_layout(parent: list[int]):
    """DFS pre-order (descent order) over the static M-node tree -> the descent_index map
    (#165 descent-ordered node layout) + per-node sibling slots for the salvage walk + the
    per-node ancestor-status slots for the ancestor-validity AND-reduction.
    children are in birth order (== drafter rank); the rank-1 child leads the descent."""
    n = len(parent)
    children: list[list[int]] = [[] for _ in range(n)]
    for i in range(1, n):
        children[parent[i]].append(i)
    depth = [0] * n
    for i in range(1, n):
        depth[i] = depth[parent[i]] + 1
    max_anc = max(depth)                  # deepest node's ancestor count (== tree max-depth)
    # DFS pre-order from the root (rank-order children) = the descent the walk follows.
    descent_order: list[int] = []
    stack = [0]
    while stack:
        u = stack.pop()
        descent_order.append(u)
        for c in reversed(children[u]):   # push so rank-1 (lowest id) is visited first
            stack.append(c)
    # descent_index[i] = the flat target_argmax slot for the i-th descent node (== node id;
    # one verify row per node). This is the per-node corrected target_logits_indices deref.
    descent_index = list(descent_order)
    # sibling slots: for descent node i (= node u), its parent's OTHER children (the rank>=2
    # rescue candidates), padded to MAX_BRANCH with u's own slot (harmless self-load) so the
    # worst-case salvage always pays MAX_BRANCH loads + selects.
    sibling_index: list[int] = []
    # ancestor-status slots (status is indexed by tree node id == descent_index, matching the
    # kernel store). TWO models of the ancestor-validity AND-reduction (a node commits only if
    # its whole ancestor chain accepted):
    #   * CARRY (faithful, the GATE): O(1) per node -- AND the PARENT's already-computed status
    #     (exactly traversal_verify_et.walk_leaf_to_root: `full[u]=match[u] AND full[parent]`).
    #     width 1 (the parent id). This is what land #71's competent DFS actually pays.
    #   * FULL (naive ceiling): O(depth) per node -- re-walk the WHOLE ancestor chain (parent up
    #     to root), padded to MAX_ANC with root 0 so every node pays the deepest chain. A strict
    #     upper bound on the dumbest possible ancestor revalidation; reported as a labelled lane.
    ancestor_carry: list[int] = []        # width 1 per node (parent id) -- faithful O(1) carry
    ancestor_full: list[int] = []         # width max_anc per node (full chain) -- naive ceiling
    for u in descent_order:
        p = parent[u] if u != 0 else 0
        sibs = [c for c in children[p] if c != u]
        sibling_index.extend((sibs + [u] * MAX_BRANCH)[:MAX_BRANCH])
        ancestor_carry.append(p)          # O(1) carry: just the parent's status slot
        anc: list[int] = []
        v = u
        while v != 0:
            v = parent[v]
            anc.append(v)                 # parent, grandparent, ..., root (depth[u] entries)
        ancestor_full.extend((anc + [0] * max_anc)[:max_anc])
    return (descent_order, descent_index, sibling_index, ancestor_carry, ancestor_full,
            depth, max_anc)


# ===== input pools =============================================================
def build_linear_pool(dev, spine_len: int, max_spec_len: int):
    """#161's conc=1 linear pool: L+1 variants accepting EXACTLY k of `spine_len` draft
    tokens (first reject forced at pos k). k==0 == the all-mismatch linear (break at pos 0)."""
    pool = []
    for k in range(spine_len + 1):
        draft = torch.arange(1, spine_len + 1, device=dev, dtype=torch.int64)
        argmax = draft.clone()
        if k < spine_len:
            argmax[k] = draft[k] + 10_000          # force first reject at position k
        cu = torch.tensor([spine_len], device=dev, dtype=torch.int32)
        bonus = torch.tensor([7], device=dev, dtype=torch.int64)
        out = torch.empty(max_spec_len + 1, device=dev, dtype=torch.int64)
        nxt = torch.empty(1, device=dev, dtype=torch.int32)
        vcnt = torch.empty(1, device=dev, dtype=torch.int32)
        pool.append({"draft": draft, "argmax": argmax, "cu": cu, "bonus": bonus,
                     "out": out, "nxt": nxt, "vcnt": vcnt})
    return pool


def build_descend_pool(dev, descent_index: list[int], sibling_index: list[int],
                       ancestor_carry: list[int], ancestor_full: list[int],
                       n_nodes: int, spine_len: int):
    """Descend-kernel pool keyed by realized accept-len k (0..spine_len) PLUS a WORST entry
    (index spine_len+1). target_argmax holds one distinct value per node; descent_index /
    sibling_index / ancestor_{carry,full} are the static descent layout. The draft array
    decides match (accept) vs mismatch (salvage) per descent node:
      * k in 0..spine_len -- match the first k descent nodes, mismatch from k on; reach =
        min(n_nodes, k+1): the realized walk descends to the first reject + salvages there.
      * WORST -- mismatch at EVERY node, reach = n_nodes: salvage fires at all M nodes,
        the full-tree adversarial descend (the gate)."""
    di = torch.tensor(descent_index, device=dev, dtype=torch.int64)
    si = torch.tensor(sibling_index, device=dev, dtype=torch.int64)
    anc_carry = torch.tensor(ancestor_carry, device=dev, dtype=torch.int64)
    anc_full = torch.tensor(ancestor_full, device=dev, dtype=torch.int64)
    # distinct positive verify-argmax per node (slot == node id).
    base_argmax = (torch.arange(n_nodes, device=dev, dtype=torch.int64) + 1) * 7
    pool = []

    def make_entry(match_count: int, reach: int):
        argmax = base_argmax.clone()
        draft = argmax.index_select(0, di).clone()     # draft[i] == argmax[descent_index[i]] => match
        for i in range(match_count, n_nodes):          # force mismatch (=> salvage) from match_count on
            draft[i] = draft[i] + 100_000
        reach_t = torch.tensor([reach], device=dev, dtype=torch.int64)
        out = torch.empty(n_nodes, device=dev, dtype=torch.int64)
        nxt = torch.empty(1, device=dev, dtype=torch.int32)
        vcnt = torch.empty(1, device=dev, dtype=torch.int32)
        status = torch.zeros(n_nodes, device=dev, dtype=torch.int32)   # ancestor-AND scratch
        bonus = torch.tensor([7], device=dev, dtype=torch.int64)
        return {"descent_index": di, "sibling_index": si, "ancestor_carry": anc_carry,
                "ancestor_full": anc_full, "status": status, "draft": draft, "argmax": argmax,
                "bonus": bonus, "reach": reach_t, "out": out, "nxt": nxt, "vcnt": vcnt}

    for k in range(spine_len + 1):
        pool.append(make_entry(match_count=k, reach=min(n_nodes, k + 1)))
    pool.append(make_entry(match_count=0, reach=n_nodes))   # WORST: all-mismatch, full descend
    return pool


WORST = None  # set to spine_len+1 at runtime


def sample_accept_schedule(rng, conditional_ladder, depth1_rate, n_steps, spine_len):
    """#161's spine accept-length sampler: depth-1 accepts w.p. depth1_rate, then depth d
    w.p. conditional_ladder[d-1]. Returns accept-lengths 0..spine_len."""
    sched = []
    for _ in range(n_steps):
        k = 0
        for d in range(1, spine_len + 1):
            p = depth1_rate if d == 1 else conditional_ladder[min(d - 1, len(conditional_ladder) - 1)]
            if rng.random() < p:
                k += 1
            else:
                break
        sched.append(k)
    return sched


# ===== launch closures + paired measurement (#161 verbatim structure) ===========
def make_step(launch_fn, pool, schedule, interleaved, filler, gemm_per_op):
    counter = {"i": 0}
    n = len(schedule)

    def one_step():
        inp = pool[schedule[counter["i"] % n]]
        counter["i"] += 1
        if interleaved:
            for _ in range(gemm_per_op):
                filler()
        launch_fn(inp)
    return one_step


def time_regime(one_step, n_passes: int, warmup: int, n_iter: int) -> dict:
    """#136 three-timing: profiler device-busy floor + CUDA-event span over N back-to-back
    steps; idle = span - busy = the exposed GPU-idle the launch pays."""
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
    return {"device_busy_us": device_busy_us, "span_us": span_us,
            "exposed_idle_us": max(0.0, span_us - device_busy_us)}


def measure_paired(specs, filler, gemm_per_op, warmup, n_iter, n_passes, rounds):
    """PAIRED: in EACH round measure every spec back-to-back (same thermal/scheduling block)
    so per-round marginals cancel common-mode event-timer drift. Robust signal = device-busy
    (profiler self-time floor); interleaved idle = the launch bubble (reported as the
    GPU-hidden confirmation)."""
    data = {name: {"busy_iso": [], "idle_iso": [], "busy_inter": [], "idle_inter": []}
            for name, *_ in specs}
    for _ in range(rounds):
        for name, launch_fn, pool, schedule in specs:
            iso = make_step(launch_fn, pool, schedule, False, filler, gemm_per_op)
            inter = make_step(launch_fn, pool, schedule, True, filler, gemm_per_op)
            riso = time_regime(iso, n_passes, warmup, n_iter)
            rinter = time_regime(inter, max(40, n_passes // 2), warmup, max(20, n_iter // 2))
            data[name]["busy_iso"].append(riso["device_busy_us"])
            data[name]["idle_iso"].append(riso["exposed_idle_us"])
            data[name]["busy_inter"].append(rinter["device_busy_us"])
            data[name]["idle_inter"].append(rinter["exposed_idle_us"])
    out = {}
    for name in data:
        dd = data[name]
        out[name] = {
            "device_busy_isolation_us": statistics.median(dd["busy_iso"]),
            "idle_interleaved_us": statistics.median(dd["idle_inter"]),
            "idle_isolation_us": statistics.median(dd["idle_iso"]),
            "device_busy_interleaved_us": statistics.median(dd["busy_inter"]),
            "device_busy_isolation_rounds": dd["busy_iso"],
            "idle_interleaved_rounds": dd["idle_inter"],
            "device_busy_isolation_summary": summarize(dd["busy_iso"]),
            "idle_interleaved_summary": summarize(dd["idle_inter"]),
        }
    return out, data


def paired_marginal(data, a: str, b: str, key: str) -> dict:
    """Per-round paired marginal (a - b) for `key`; median + CI cancel common-mode."""
    per_round = [data[a][key][r] - data[b][key][r] for r in range(len(data[a][key]))]
    s = summarize(per_round)
    return {"per_round": per_round, "median": statistics.median(per_round),
            "mean": s.get("mean", 0.0), "ci95_abs": s.get("ci95_abs", 0.0),
            "within_ci": abs(s.get("mean", 0.0)) <= s.get("ci95_abs", 0.0) + 1e-9,
            "sign_flips": (min(per_round) < 0.0 < max(per_round))}


# ===== self-test: reproduce the #136/#161 1.2182 anchor on THIS A10G =============
def run_self_test(args) -> dict:
    """Re-run #161's measure_overlap_hidden_idle and recompose the depth-9 step.
    Pass iff |step - 1.2182| / 1.2182 <= tol (identical control to #161's 1.21792)."""
    import star_attn_fp32_steptime as sas
    counts = {"sliding": sas.N_SLIDING, "full": sas.N_FULL}
    m = sas.measure_overlap_hidden_idle(
        M=32, ctx=args.ctx, n_passes=args.eager_passes, warmup=args.warmup,
        counts=counts, n_iter=args.n_iter, gemm_n=args.gemm_filler_n)
    idle_us = m["exposed_idle_overlap_us"]
    step_repro = ROOFLINE_STEP + idle_us / STEP_M8_US
    delta_pct = 100.0 * abs(step_repro - MEASURED_STEP_136) / MEASURED_STEP_136
    return {
        "roofline_step": ROOFLINE_STEP, "idle_overlap_us": idle_us,
        "per_call_idle_overlap_us": m["per_call_idle_overlap_us"],
        "step_reproduced": step_repro, "anchor_136": MEASURED_STEP_136,
        "pr161_step_reproduced": PR161_STEP_REPRODUCED,
        "delta_vs_anchor_pct": delta_pct, "tol_pct": SELFTEST_TOL_PCT,
        "gemm_per_call": m["gemm_per_call"], "filler_us_each": m["filler_us_each"],
        "device_busy_us": m["device_busy_us"], "eager_span_us": m["eager_span_us"],
        "reproduces_anchor": bool(delta_pct <= SELFTEST_TOL_PCT),
    }


def run(args) -> dict:
    assert torch.cuda.is_available(), "CUDA required (set CUDA_VISIBLE_DEVICES=0 on this pod)"
    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    rng = __import__("random").Random(args.seed)

    parent = load_m32_topology()
    n_nodes = len(parent)
    (descent_order, descent_index, sibling_index, ancestor_carry, ancestor_full,
     depth, max_anc) = build_descent_layout(parent)
    spine_len = args.spine_len
    global WORST
    WORST = spine_len + 1   # index of the all-mismatch-descend worst entry in the descend pool

    res: dict = {
        "pr": 173, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "l2_bytes": torch.cuda.get_device_properties(0).L2_cache_size,
        "topology": {"n_nodes": n_nodes, "max_branch": MAX_BRANCH, "spine_len": spine_len,
                     "max_anc": max_anc, "depth": depth, "descent_order": descent_order,
                     # per-node scalar-op budget (per node: 4 base = idx+draft+targ load +
                     # compare; + anc*(2 load + 1 AND) ancestor revalidation; + MAX_BRANCH*(2
                     # load + 1 select) salvage; + 1 store). all-mismatch => x N_NODES.
                     # FAITHFUL gate uses the O(1) carry (anc=1); the naive CEILING re-walks
                     # the full chain (anc=max_anc).
                     "worst_scalar_ops_carry": n_nodes * (4 + 1 * 3 + MAX_BRANCH * 3 + 1),
                     "worst_scalar_ops_ceiling": n_nodes * (4 + max_anc * 3 + MAX_BRANCH * 3 + 1)},
        "anchors": {
            "k_cal": K_CAL, "step_m8_us": STEP_M8_US, "roofline_step": ROOFLINE_STEP,
            "measured_step_136": MEASURED_STEP_136, "e_t_descent_only": E_T_DESCENT_ONLY,
            "e_t_both_bugs": E_T_BOTH_BUGS, "depth1_descent": DEPTH1_DESCENT,
            "depth1_both": DEPTH1_BOTH,
            "official_descent_realized": OFFICIAL_DESCENT_REALIZED,
            "official_both_realized": OFFICIAL_BOTH_REALIZED,
            "official_descent_roofline": OFFICIAL_DESCENT_ROOFLINE,
            "official_both_roofline": OFFICIAL_BOTH_ROOFLINE,
            "pr161_step_reproduced": PR161_STEP_REPRODUCED,
        },
        "config": {
            "seed": args.seed, "spine_len": spine_len, "max_spec_len": args.max_spec_len,
            "n_iter": args.n_iter, "warmup": args.warmup, "n_passes": args.n_passes,
            "rounds": args.rounds, "sched_steps": args.sched_steps,
            "gemm_filler_n": args.gemm_filler_n, "hidden_threshold_us": args.hidden_threshold_us,
        },
        # ubel #163 op-5 claim under test: "descent_accept_walk ... design sync-free (GPU-hidden),
        # +0 net by design." This PR MEASURES it with a kernel-resident worst-case variant.
        "claim_under_test": {
            "source": "research/spec_cost_model/host_residency_sweep (ubel #163), op 5",
            "claim": "descent_accept_walk +0 net (GPU-hidden by design)",
            "method": "kernel-resident accept_prep_descend_walk worst-case (all-mismatch-descend) "
                      "paired device-busy marginal vs the linear break kernel (#161 method, new target)",
        },
    }

    # ---- 1. SELF-TEST (PRIMARY leg a): reproduce the 1.2182 anchor -------------
    print("[descent-walk] self-test: reproducing the #136/#161 1.2182 step anchor ...", flush=True)
    res["self_test"] = run_self_test(args)
    st = res["self_test"]
    print(f"[descent-walk] self-test step={st['step_reproduced']:.5f} vs anchor "
          f"{MEASURED_STEP_136} (delta {st['delta_vs_anchor_pct']:.3f}% <= {SELFTEST_TOL_PCT}%) "
          f"-> reproduces_anchor={st['reproduces_anchor']}", flush=True)

    # ---- filler GEMM sized to the step's non-attention GPU work (#161 verbatim) -
    a = torch.randn(args.gemm_filler_n, args.gemm_filler_n, dtype=torch.bfloat16, device=dev)
    b = torch.randn(args.gemm_filler_n, args.gemm_filler_n, dtype=torch.bfloat16, device=dev)
    c = torch.empty(args.gemm_filler_n, args.gemm_filler_n, dtype=torch.bfloat16, device=dev)

    def filler():
        torch.mm(a, b, out=c)

    filler_us = _profiled_device_us(torch, filler, args.n_iter, args.warmup)
    step_gpu_us = 9149.573677586994   # #143 step_gpu_us_target (gemm_mult + drafter); the ~92%-weight-GEMM step
    gemm_per_op = max(1, round(step_gpu_us / filler_us))
    res["filler"] = {"gemm_n": args.gemm_filler_n, "filler_us_each": filler_us,
                     "step_gpu_us_target": step_gpu_us, "gemm_per_op": gemm_per_op,
                     "realized_filler_us_per_step": filler_us * gemm_per_op}

    # ---- 2. ACCEPT-PREP MICRO-BENCH (paired) ----------------------------------
    baseline_k, descend_k = _build_kernels()
    linear_pool = build_linear_pool(dev, spine_len, args.max_spec_len)
    descend_pool = build_descend_pool(dev, descent_index, sibling_index, ancestor_carry,
                                      ancestor_full, n_nodes, spine_len)

    def launch_linear(inp):
        baseline_k[(1,)](inp["out"], inp["nxt"], inp["vcnt"], inp["cu"], inp["draft"],
                         inp["argmax"], inp["bonus"], args.max_spec_len)

    # FAITHFUL descend (the gate): O(1) ancestor carry (MAX_ANC=1, parent-status AND).
    def launch_descend(inp):
        descend_k[(1,)](inp["out"], inp["nxt"], inp["vcnt"], inp["descent_index"],
                        inp["sibling_index"], inp["ancestor_carry"], inp["status"],
                        inp["draft"], inp["argmax"], inp["bonus"],
                        inp["reach"], n_nodes, MAX_BRANCH, 1)

    # NAIVE CEILING: O(depth) ancestor re-walk (MAX_ANC=max_anc, full chain per node).
    def launch_descend_ceiling(inp):
        descend_k[(1,)](inp["out"], inp["nxt"], inp["vcnt"], inp["descent_index"],
                        inp["sibling_index"], inp["ancestor_full"], inp["status"],
                        inp["draft"], inp["argmax"], inp["bonus"],
                        inp["reach"], n_nodes, MAX_BRANCH, max_anc)

    sched_realistic = sample_accept_schedule(rng, RISING_SPINE, DEPTH1_BOTH,
                                             args.sched_steps, spine_len)
    sched_worst_descend = [WORST] * args.sched_steps          # always all-mismatch-descend
    sched_worst_linear = [0] * args.sched_steps               # always reject@0 == linear all-mismatch
    res["schedule_stats"] = {
        "realistic_mean_accept_len": statistics.fmean(sched_realistic),
        "realistic_depth1_rate": DEPTH1_BOTH, "n_steps": len(sched_realistic),
        "worst_reach_nodes": n_nodes, "realistic_mean_reach":
            statistics.fmean(min(n_nodes, k + 1) for k in sched_realistic),
    }

    print("[descent-walk] paired accept-prep micro-bench (linear / descend / descend-worst / "
          "descend-ceiling / linear-worst per round) ...", flush=True)
    specs = [
        ("linear_break", launch_linear, linear_pool, sched_realistic),
        ("descend_walk", launch_descend, descend_pool, sched_realistic),
        ("descend_walk_worst", launch_descend, descend_pool, sched_worst_descend),
        ("descend_walk_ceiling", launch_descend_ceiling, descend_pool, sched_worst_descend),
        ("linear_break_worst", launch_linear, linear_pool, sched_worst_linear),
    ]
    measured, raw = measure_paired(specs, filler, gemm_per_op, args.warmup, args.n_iter,
                                   args.n_passes, args.rounds)
    res["accept_prep_measured"] = measured

    m_lin, m_desc, m_worst, m_ceiling, m_linworst = (
        measured["linear_break"], measured["descend_walk"], measured["descend_walk_worst"],
        measured["descend_walk_ceiling"], measured["linear_break_worst"])

    # ---- 3. PROPAGATE (lead with the paired device-busy marginal) -------------
    busy_lin = m_lin["device_busy_isolation_us"]
    busy_desc = m_desc["device_busy_isolation_us"]
    busy_worst = m_worst["device_busy_isolation_us"]
    busy_ceiling = m_ceiling["device_busy_isolation_us"]
    busy_linworst = m_linworst["device_busy_isolation_us"]

    # GATE marginal: the FAITHFUL worst-case descend kernel (O(1) ancestor carry, all-mismatch-
    # descend, full salvage at every node) vs the served linear break kernel -- the most land
    # #71's actual build could add to the realized step. Cross-checks: worst-vs-worst (same
    # all-mismatch data => pure salvage-DFS extra work, launch common-mode cancelled); CEILING
    # (naive O(depth) ancestor re-walk) -- the strict upper bound on the dumbest ancestor impl.
    marg_gate = paired_marginal(raw, "descend_walk_worst", "linear_break", "busy_iso")
    marg_worstpair = paired_marginal(raw, "descend_walk_worst", "linear_break_worst", "busy_iso")
    marg_realistic = paired_marginal(raw, "descend_walk", "linear_break", "busy_iso")
    marg_ceiling = paired_marginal(raw, "descend_walk_ceiling", "linear_break", "busy_iso")
    idle_marg_gate = paired_marginal(raw, "descend_walk_worst", "linear_break", "idle_inter")

    descend_walk_marginal_us = marg_gate["median"]
    descend_walk_within_ci = marg_gate["within_ci"]
    dstep_marginal = descend_walk_marginal_us / STEP_M8_US

    # descent-kernel operative step vs the launch-realized 1.2182 anchor (#168).
    descent_kernel_step_pinned = MEASURED_STEP_136 + max(0.0, dstep_marginal)
    descent_walk_step_delta_pct = 100.0 * (descent_kernel_step_pinned - MEASURED_STEP_136) / MEASURED_STEP_136

    # naive-ceiling lane (NOT the gate): O(depth) ancestor re-walk at every node.
    ceiling_marginal_us = marg_ceiling["median"]
    dstep_ceiling = ceiling_marginal_us / STEP_M8_US
    descent_kernel_step_ceiling = MEASURED_STEP_136 + max(0.0, dstep_ceiling)
    descent_walk_step_ceiling_delta_pct = 100.0 * (descent_kernel_step_ceiling - MEASURED_STEP_136) / MEASURED_STEP_136
    off_desc_ceiling = fern_official(E_T_DESCENT_ONLY, descent_kernel_step_ceiling, TAU_FERN_CENTRAL)
    off_both_ceiling = fern_official(E_T_BOTH_BUGS, descent_kernel_step_ceiling, TAU_FERN_CENTRAL)

    # whole-descend-kernel device-busy as % of the step (the entire heavy DFS launch).
    descend_busy_pct_of_step = 100.0 * busy_worst / STEP_M8_US / MEASURED_STEP_136

    # propagate official = K_cal*E[T]/step*tau at the descent-kernel step (overlap) AND at the
    # roofline edge (the optimistic band, where #168's 522/538 live) -- honest bracket.
    off_desc_pinned = fern_official(E_T_DESCENT_ONLY, descent_kernel_step_pinned, TAU_FERN_CENTRAL)
    off_both_pinned = fern_official(E_T_BOTH_BUGS, descent_kernel_step_pinned, TAU_FERN_CENTRAL)
    off_desc_roof = fern_official(E_T_DESCENT_ONLY, ROOFLINE_STEP + max(0.0, dstep_marginal), TAU_FERN_CENTRAL)
    off_both_roof = fern_official(E_T_BOTH_BUGS, ROOFLINE_STEP + max(0.0, dstep_marginal), TAU_FERN_CENTRAL)

    # GPU-hidden gate (#161): the whole descend launch device-busy << step AND its launch idle
    # <= thresh in every regime; the worst-case marginal moves the step by < NEUTRAL_STEP_PCT.
    idle_hidden = all(measured[n]["idle_interleaved_us"] <= args.hidden_threshold_us for n in measured)
    busy_hidden = all(measured[n]["device_busy_isolation_us"] <= args.hidden_threshold_us for n in measured)
    step_neutral_practical = descent_walk_step_delta_pct < NEUTRAL_STEP_PCT
    sign_flips = marg_gate["sign_flips"]
    step_neutral = bool(step_neutral_practical and busy_hidden and idle_hidden)

    # officials hold at the descent-kernel step (the #168 launch numbers survive the descend walk).
    descent_clears_500 = off_desc_pinned > 500.0
    both_clears_500 = off_both_pinned > 500.0
    descent_holds_520 = off_desc_pinned >= 515.0     # descent-only ~520 (allow the small overlap band)
    both_holds_535 = off_both_pinned >= 530.0        # both-bugs ~535

    if step_neutral and descent_clears_500 and both_clears_500:
        verdict = "GREEN"
        reason = (f"the salvage-descend accept-prep is STEP-NEUTRAL vs the linear break kernel. "
                  f"MEASURED device-busy: linear break {busy_lin:.3f}us (served) vs FAITHFUL descend-walk "
                  f"WORST-CASE all-mismatch-descend {busy_worst:.3f}us (O(1) ancestor carry + salvage at "
                  f"all {n_nodes} nodes). Paired marginal {descend_walk_marginal_us:+.4f}us "
                  f"(ci95 {marg_gate['ci95_abs']:.4f}) -> +{descent_walk_step_delta_pct:.4f}% step "
                  f"(< {NEUTRAL_STEP_PCT}% practical floor ~= 0.5 TPS ~= 9.7us of marginal device-busy). The "
                  f"whole heavy descend launch is {descend_busy_pct_of_step:.4f}% of the 9150us step (1 "
                  f"sync-free launch, GPU-hidden behind the ~92%-weight-GEMM step). Even the NAIVE O(depth) "
                  f"ancestor-rewalk ceiling adds only {ceiling_marginal_us:+.3f}us = "
                  f"+{descent_walk_step_ceiling_delta_pct:.4f}% (descent {off_desc_ceiling:.1f}/both "
                  f"{off_both_ceiling:.1f}, still clears 500). ubel #163's op-5 '+0 net by design' is "
                  f"CONFIRMED by measurement. The launch quotes ONE step = {descent_kernel_step_pinned:.4f} "
                  f"== 1.2182; descent-only {off_desc_pinned:.2f} / both-bugs {off_both_pinned:.2f} HOLD.")
    elif descent_clears_500 and both_clears_500:
        verdict = "AMBER"
        reason = (f"the FAITHFUL salvage-descend accept-prep adds a PRACTICALLY non-zero but bar-safe "
                  f"per-step cost. Paired worst-case marginal {descend_walk_marginal_us:+.4f}us = "
                  f"+{descent_walk_step_delta_pct:.4f}% step (>= {NEUTRAL_STEP_PCT}% practical floor). "
                  f"The lifted descent-kernel step is {descent_kernel_step_pinned:.4f}; descent-only "
                  f"{off_desc_pinned:.2f} and both-bugs {off_both_pinned:.2f} still clear 500. Quote the "
                  f"LIFTED step, not 1.2182. (Adversarial all-mismatch worst case; the realized descend "
                  f"marginal is {marg_realistic['median']:+.4f}us = step-neutral. The naive O(depth) "
                  f"ancestor-rewalk ceiling is {ceiling_marginal_us:+.3f}us = "
                  f"+{descent_walk_step_ceiling_delta_pct:.4f}%, descent {off_desc_ceiling:.1f}/both "
                  f"{off_both_ceiling:.1f}, still clears 500.)")
    else:
        verdict = "RED"
        reason = (f"the salvage-descend accept-prep BREAKS a bar: worst-case marginal "
                  f"{descend_walk_marginal_us:+.4f}us = +{descent_walk_step_delta_pct:.4f}% step -> "
                  f"lifted step {descent_kernel_step_pinned:.4f}, descent-only {off_desc_pinned:.2f} / "
                  f"both-bugs {off_both_pinned:.2f}. Re-price before land #71 commits the descend kernel.")

    res["propagation"] = {
        "device_busy_linear_break_us": busy_lin,
        "device_busy_descend_walk_us": busy_desc,
        "device_busy_descend_walk_worst_us": busy_worst,
        "device_busy_descend_walk_ceiling_us": busy_ceiling,
        "device_busy_linear_break_worst_us": busy_linworst,
        "descend_busy_pct_of_step": descend_busy_pct_of_step,
        "marginal_descend_walk_gate_paired": marg_gate,
        "marginal_descend_walk_worstpair_paired": marg_worstpair,
        "marginal_descend_walk_realistic_paired": marg_realistic,
        "marginal_descend_walk_ceiling_paired": marg_ceiling,
        "marginal_descend_walk_idle_paired": idle_marg_gate,
        "idle_interleaved_linear_us": m_lin["idle_interleaved_us"],
        "idle_interleaved_descend_us": m_desc["idle_interleaved_us"],
        "idle_interleaved_descend_worst_us": m_worst["idle_interleaved_us"],
        "descend_walk_marginal_us": descend_walk_marginal_us,
        "descend_walk_marginal_worstpair_us": marg_worstpair["median"],
        "descend_walk_marginal_realistic_us": marg_realistic["median"],
        "descend_walk_marginal_ceiling_us": ceiling_marginal_us,
        "descend_walk_within_ci": descend_walk_within_ci,
        "descend_walk_marginal_sign_flips": sign_flips,
        "marginal_dstep_units": dstep_marginal,
        "descent_kernel_step_pinned": descent_kernel_step_pinned,
        "descent_kernel_step_ceiling": descent_kernel_step_ceiling,
        "descent_walk_step_ceiling_delta_pct": descent_walk_step_ceiling_delta_pct,
        "launch_realized_step_anchor": MEASURED_STEP_136,
        "descent_walk_step_delta_pct": descent_walk_step_delta_pct,
        "official": {
            "descent_only_pinned_overlap": off_desc_pinned,
            "both_bugs_pinned_overlap": off_both_pinned,
            "descent_only_pinned_roofline": off_desc_roof,
            "both_bugs_pinned_roofline": off_both_roof,
            "descent_only_ceiling": off_desc_ceiling,
            "both_bugs_ceiling": off_both_ceiling,
            "descent_only_realized_anchor": OFFICIAL_DESCENT_REALIZED,
            "both_bugs_realized_anchor": OFFICIAL_BOTH_REALIZED,
            "descent_only_drop_vs_realized": OFFICIAL_DESCENT_REALIZED - off_desc_pinned,
            "both_bugs_drop_vs_realized": OFFICIAL_BOTH_REALIZED - off_both_pinned,
        },
        "descent_clears_500": descent_clears_500,
        "both_clears_500": both_clears_500,
        "descent_holds_520": descent_holds_520,
        "both_holds_535": both_holds_535,
        "idle_hidden": idle_hidden,
        "busy_hidden": busy_hidden,
        "step_neutral_practical": step_neutral_practical,
        "descend_walk_within_ci": descend_walk_within_ci,
        "step_neutral": step_neutral,
        "neutral_step_pct": NEUTRAL_STEP_PCT,
        "hidden_threshold_us": args.hidden_threshold_us,
    }
    res["verdict"] = verdict
    res["verdict_reason"] = reason

    # ---- 4. SELF-TEST (PRIMARY): rig sound + officials hold + NaN-clean --------
    headline = [st["step_reproduced"], descend_walk_marginal_us, descent_kernel_step_pinned,
                descent_walk_step_delta_pct, off_desc_pinned, off_both_pinned,
                busy_lin, busy_worst, marg_gate["ci95_abs"],
                ceiling_marginal_us, off_desc_ceiling, off_both_ceiling]
    nan_clean = all(math.isfinite(x) for x in headline)
    # (b) the descend-walk marginal is step-neutral (< practical floor) OR sign-flips run-to-run,
    # OR -- if not -- the lifted step is reported honestly AND the officials still hold (c).
    marginal_characterized = bool(step_neutral_practical or sign_flips or (descent_clears_500 and both_clears_500))
    self_test = {
        "a_rig_reproduces_anchor": st["reproduces_anchor"],
        "a_step_reproduced": st["step_reproduced"],
        "a_delta_vs_anchor_pct": st["delta_vs_anchor_pct"],
        "b_marginal_step_neutral": step_neutral_practical,
        "b_marginal_sign_flips": sign_flips,
        "b_marginal_characterized": marginal_characterized,
        "c_descent_only_holds_520": descent_holds_520,
        "c_both_bugs_holds_535": both_holds_535,
        "c_descent_only_official": off_desc_pinned,
        "c_both_bugs_official": off_both_pinned,
        "d_nan_clean": nan_clean,
    }
    descent_walk_step_self_test_passes = bool(
        self_test["a_rig_reproduces_anchor"] and marginal_characterized
        and descent_holds_520 and both_holds_535 and nan_clean)
    self_test["descent_walk_step_self_test_passes"] = descent_walk_step_self_test_passes
    res["self_test_summary"] = self_test

    res["primary_metric"] = {"name": "descent_walk_step_self_test_passes",
                             "value": int(descent_walk_step_self_test_passes)}
    res["test_metric"] = {"name": "descent_walk_step_delta_pct", "value": descent_walk_step_delta_pct}
    res["elapsed_s"] = time.time() - t0
    res["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9

    del a, b, c
    torch.cuda.empty_cache()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2))
    print(f"\n[descent-walk] VERDICT={verdict}", flush=True)
    print(f"[descent-walk] descent_walk_step_self_test_passes={descent_walk_step_self_test_passes}  "
          f"descend_walk_marginal_us={descend_walk_marginal_us:+.4f} (within_ci={descend_walk_within_ci}, "
          f"sign_flips={sign_flips})", flush=True)
    print(f"[descent-walk] descent_kernel_step_pinned={descent_kernel_step_pinned:.4f} "
          f"(delta {descent_walk_step_delta_pct:+.4f}% vs 1.2182)  "
          f"descent-only {off_desc_pinned:.2f} / both-bugs {off_both_pinned:.2f}", flush=True)
    print(f"[descent-walk] {reason}", flush=True)
    print(f"[descent-walk] wrote {out_path} ({res['elapsed_s']:.0f}s, peak {res['peak_gpu_gb']:.3f}GB)", flush=True)

    # ---- W&B ------------------------------------------------------------------
    if args.wandb_group and not args.no_wandb:
        try:
            import wandb
            run_w = wandb.init(project="gemma-challenge-senpai", entity="wandb-applied-ai-team",
                               group=args.wandb_group, name=args.wandb_name,
                               config={**res["config"], **res["anchors"],
                                       "n_nodes": n_nodes, "max_branch": MAX_BRANCH,
                                       "max_anc": max_anc,
                                       "worst_scalar_ops_carry": res["topology"]["worst_scalar_ops_carry"],
                                       "worst_scalar_ops_ceiling": res["topology"]["worst_scalar_ops_ceiling"],
                                       "gpu": res["gpu"]})
            wandb.log({
                "descent_walk_step_self_test_passes": int(descent_walk_step_self_test_passes),
                "self_test_step_reproduced": st["step_reproduced"],
                "self_test_delta_vs_anchor_pct": st["delta_vs_anchor_pct"],
                "descend_walk_marginal_us": descend_walk_marginal_us,
                "descend_walk_marginal_worstpair_us": marg_worstpair["median"],
                "descend_walk_marginal_realistic_us": marg_realistic["median"],
                "descend_walk_marginal_ceiling_us": ceiling_marginal_us,
                "descend_walk_marginal_ci95_us": marg_gate["ci95_abs"],
                "descend_walk_within_ci": int(descend_walk_within_ci),
                "descend_walk_marginal_sign_flips": int(sign_flips),
                "descent_kernel_step_pinned": descent_kernel_step_pinned,
                "descent_kernel_step_ceiling": descent_kernel_step_ceiling,
                "descent_walk_step_delta_pct": descent_walk_step_delta_pct,
                "descent_walk_step_ceiling_delta_pct": descent_walk_step_ceiling_delta_pct,
                "official_descent_only_pinned": off_desc_pinned,
                "official_both_bugs_pinned": off_both_pinned,
                "official_descent_only_roofline": off_desc_roof,
                "official_both_bugs_roofline": off_both_roof,
                "official_descent_only_ceiling": off_desc_ceiling,
                "official_both_bugs_ceiling": off_both_ceiling,
                "device_busy_linear_break_us": busy_lin,
                "device_busy_descend_walk_worst_us": busy_worst,
                "device_busy_descend_walk_ceiling_us": busy_ceiling,
                "descend_busy_pct_of_step": descend_busy_pct_of_step,
                "worst_scalar_ops_carry": res["topology"]["worst_scalar_ops_carry"],
                "worst_scalar_ops_ceiling": res["topology"]["worst_scalar_ops_ceiling"],
                "max_anc": max_anc,
                "idle_interleaved_descend_worst_us": m_worst["idle_interleaved_us"],
                "descent_clears_500": int(descent_clears_500),
                "both_clears_500": int(both_clears_500),
                "step_neutral": int(step_neutral),
                "verdict_green": int(verdict == "GREEN"),
            })
            run_w.summary["verdict"] = verdict
            res["wandb_run_id"] = run_w.id
            run_w.finish()
            out_path.write_text(json.dumps(res, indent=2))
            print(f"[descent-walk] W&B run {res['wandb_run_id']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[descent-walk] W&B logging skipped: {exc!r}", flush=True)

    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spine-len", type=int, default=9, help="depth-9 tree spine length")
    ap.add_argument("--max-spec-len", type=int, default=9)
    ap.add_argument("--sched-steps", type=int, default=512, help="accept-length schedule length")
    ap.add_argument("--n-iter", type=int, default=80, help="profiler device-busy iters")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--n-passes", type=int, default=300, help="event-span passes (isolation)")
    ap.add_argument("--rounds", type=int, default=5, help="repeat each regime, take median")
    ap.add_argument("--eager-passes", type=int, default=200, help="self-test anchor passes")
    ap.add_argument("--ctx", type=int, default=528, help="self-test attention ctx (#136)")
    ap.add_argument("--gemm-filler-n", type=int, default=GEMM_FILLER_N)
    ap.add_argument("--hidden-threshold-us", type=float, default=60.0,
                    help="idle <= this under overlap == GPU-hidden (#136 43us / #143 38us)")
    ap.add_argument("--seed", type=int, default=173)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "research/descent_walk_step_cost/descent_walk_step_cost.json")
    ap.add_argument("--wandb-group", type=str, default="descent-walk-step-neutrality")
    ap.add_argument("--wandb-name", type=str, default="lawine/descent-walk-step-neutrality")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--quick", action="store_true", help="fast smoke (few passes/rounds)")
    args = ap.parse_args(argv)
    if args.quick:
        args.n_passes, args.rounds, args.eager_passes, args.sched_steps = 60, 2, 40, 64
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
