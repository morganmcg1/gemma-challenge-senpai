#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Both-bugs accept-prep step cost: does the bug-1 depth-1 spine fix keep 1.2182? (PR #161)

LOCAL A10G profiling ONLY -- pure Python/CUDA/Triton timing harness. NO model,
NO HF Job, NO submission, NO served-file change, NO kernel deploy, NO quota.
BASELINE stays 481.53; greedy untouched.

WHY
---
My #136 (MERGED) firmed the depth-9 verify step at 1.2182 (overlap-central; +0.45%
vs the 1.2127 roofline). My #143 (salvage-walk) showed the DESCENT-ONLY (bug-2)
accept-prep is GPU-hidden under realistic GEMM overlap (+0.39% sync-free, 19x idle
collapse). The both-bugs official projection 537.8 (= K_cal * 5.207 / 1.2127,
denken #133 `official_at_recovered_depth1_central`) ASSUMES the both-bugs
accept-prep costs the SAME step as descent-only. PR #161 PINS that assumption.

The both-bugs accept-prep ADDS the BUG-1 depth-1 spine fix on top of bug-2. From
denken #133's root-cause (research/validity/fp32_star_verify_crosscheck/
rootcause_results.json): the bug-1 deficit (q1 0.598/0.679 -> 0.7287) is 96%
PLUMBING (b_plumbing_pp_spine_index=12.55pp of 13.07pp; c_intrinsic=0.0). The fix
is "correct the tree depth-1 spine extraction / target_logits_indices so the root
verify-row compares against the drafter's rank-1 (top1) token" -- build-plumbing
UPSTREAM of `_dixie_fused_accept_prep_kernel`, NOT added kernel logic. The served
accept-prep kernel (fa2sw_precache_kenyan/sitecustomize.py:921, byte-identical in
lf29cap444_pupa_check) consumes target_argmax and is INVARIANT to which rank the
depth-1 row points at; fixing bug-1 changes the VALUES (more depth-1 matches ->
higher E[T]), not the op-count.

THE QUESTION
------------
Is the both-bugs accept-prep step-NEUTRAL vs descent-only (so 537.8 LOCKS IN), or
does re-enabling the depth-1 conditional add a measurable per-step tax (so 537.8
re-scores DOWN)?

METHOD (reuses my #136 / #143 interleaved-with-filler-GEMM method VERBATIM;
isolation over-reads ~80x and is reported only as the no-overlap pessimist)
--------------------------------------------------------------------------------
1. SELF-TEST (PRIMARY): re-run the #136 measure_overlap_hidden_idle on THIS A10G
   and confirm the depth-9 step recomposes to 1.2182 within tolerance. Gates the rig.
2. ACCEPT-PREP MICRO-BENCH: time the REAL `_dixie_fused_accept_prep_kernel`
   (replicated verbatim) at conc=1 over a realistic accept-length schedule, in two
   data regimes that differ ONLY in the depth-1 acceptance rate:
     * descent_only -- depth-1 match rate 0.679 (bug-1 contaminated).
     * both_bugs    -- depth-1 match rate 0.7287 (bug-1 corrected, rank-1 spine).
   plus a WORST-CASE kernel variant `_accept_prep_depth1_spine` that adds an
   EXPLICIT depth-1 conditional (an extra target_logits_indices indirection load +
   compare + conditional re-seed at the root) -- the most the bug-1 fix could add at
   the kernel level if it were NOT pure upstream plumbing. Each measured isolation
   (no-overlap upper bound) + interleaved (per-step GEMM overlap -> the credible
   step penalty); idle = event-span - profiler device-busy floor.
3. PROPAGATE: marginal both-bugs step delta = idle_interleaved(both_bugs) -
   idle_interleaved(descent_only); both_bugs_step_pinned = 1.2182 + delta/STEP_M8;
   both_bugs_official_pinned = K_cal * 5.207 / step. Compare to the 537.8 assumption.

Primary metric: step_cost_self_test_passes (bool, the 1.2182 reproduction gate).
Test metric:    both_bugs_step_delta_pct (% step inflation from the depth-1 fix).
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

# Reuse the EXACT #136/#143 compose constants + timing primitives (NOT re-derived).
from salvage_walk_overhead import (  # noqa: E402
    K_CAL, STEP_M8_US, STEP_WSTAR_DEPTH9, MEASURED_STEP_136, E_T_TREE_CEILING,
    TAU_FERN_CENTRAL, TAU_FERN_LOW, TARGET_500, TARGET_530, GEMM_FILLER_N,
    Z95, summarize, fern_official, fern_clear_bar, price_step, time_regime,
)
from scripts.local_validation.profile_attention import _profiled_device_us  # noqa: E402

# ===== both-bugs E[T] ladder + the assumption under test =======================
# research/spec_cost_model/bug2_salvage_descent_results.json (wirbel #135)
E_T_DESCENT_ONLY = 5.056404568844709     # bug2_et_full_alt_d1_0679 (bug-2 fixed, depth-1=0.679)
E_T_BOTH_BUGS = 5.206954309441963        # combined_et_both_fixed (both bugs, depth-1=0.7287)
DEPTH1_DESCENT = 0.679                    # contaminated depth-1 spine accept (bug-1 live)
DEPTH1_BOTH = 0.728739760479042          # deployed_rising_spine[0] (bug-1 fixed: rank-1)
# deployed rising-spine conditional accept ladder (both bugs fixed)
RISING_SPINE = [0.728739760479042, 0.7589764102641635, 0.7924989076194682,
                0.821702519412012, 0.8342716929825772, 0.8352594665096346,
                0.8472621220149911]
# denken #133 official at the recovered depth-1 (= K_cal * 5.207 / 1.2127 roofline)
OFFICIAL_BOTH_ASSUMED_ROOFLINE = 537.8399229182609
ROOFLINE_STEP = STEP_WSTAR_DEPTH9        # 1.2127483746822987

# self-test tolerance on the 1.2182 reproduction (the overlap idle is an event-timer
# floor; #136 itself measured +0.45%, this rig +0.43% -> a 1.5% step band is generous
# but still rejects a broken rig that mis-measures the step by an order of magnitude).
SELFTEST_TOL_PCT = 1.5

# PRACTICAL step-neutrality floor. PR #161 asks "does the both-bugs accept-prep keep
# 1.2182?". The robust signal is the PAIRED device-busy marginal (profiler self-time,
# common-mode-cancelled). That signal is so repeatable that a sub-100ns marginal can be
# "statistically resolvable" (outside its own tiny CI) yet PHYSICALLY NIL: 0.0284us on
# a 9150us step = +0.0003% step = the 537.8 official moves by ~0.001 TPS. So the verdict
# gates on PRACTICAL magnitude (does the marginal move the step by a meaningful amount),
# NOT on statistical within-CI. 0.10% step ~= 0.5 TPS on the 537.8 official (dofficial/
# dstep ~= -440 TPS/step-unit); the 530 bar has ~7.8 TPS of headroom, so <0.10% is nil.
NEUTRAL_STEP_PCT = 0.10


# ===== the REAL served accept-prep kernel (replicated verbatim from
#       submissions/fa2sw_precache_kenyan/sitecustomize.py:921) + a worst-case
#       depth-1-conditional variant ===============================================
def _build_kernels():
    import triton
    import triton.language as tl

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

    # WORST CASE: model the bug-1 fix as an EXPLICIT depth-1 spine conditional that
    # the kernel pays per step (instead of the real upstream plumbing). The fix's
    # root-cause is "compare the root verify-row against the drafter rank-1 token via
    # target_logits_indices" -> at depth-1 (pos 0), deref a spine-root index
    # (+1 indirection load), compare against the rank-1 token, and conditionally
    # re-seed (next_token/valid_count). This OVER-models the real fix (which adds 0
    # kernel ops); if even this is GPU-hidden, the real plumbing fix certainly is.
    @triton.jit(do_not_specialize=["max_spec_len"])
    def accept_prep_depth1_spine(
        output_token_ids_ptr, next_token_ids_ptr, valid_counts_ptr,
        cu_num_draft_tokens_ptr, draft_token_ids_ptr, target_argmax_ptr,
        bonus_token_ids_ptr, spine_root_index_ptr, max_spec_len,
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
                # --- BUG-1 DEPTH-1 CONDITIONAL (the re-enabled root-spine logic) ---
                if pos == 0:
                    root_row = tl.load(spine_root_index_ptr + req_idx)   # +1 indirection load
                    target_argmax_id = tl.load(target_argmax_ptr + root_row).to(tl.int32)
                else:
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

    return accept_prep_baseline, accept_prep_depth1_spine


# ===== conc=1 input pool: one (draft, target_argmax) set per accept-length k ======
def build_input_pool(dev, spine_len: int, max_spec_len: int):
    """For batch=1, build L+1 input variants where the kernel accepts EXACTLY k of
    the `spine_len` draft tokens (k = 0..spine_len): draft[pos]==argmax[pos] for
    pos<k, draft[k]!=argmax[k] (the first reject). k==spine_len -> all accept ->
    bonus appended. The verify row VALUES are identity; only the realized accept
    length varies (the data-dependent device work the kernel actually pays)."""
    pool = []
    for k in range(spine_len + 1):
        draft = torch.arange(1, spine_len + 1, device=dev, dtype=torch.int64)
        argmax = draft.clone()
        if k < spine_len:                       # force first reject at position k
            argmax[k] = draft[k] + 10_000
        cu = torch.tensor([spine_len], device=dev, dtype=torch.int32)
        bonus = torch.tensor([7], device=dev, dtype=torch.int64)
        out = torch.empty(max_spec_len + 1, device=dev, dtype=torch.int64)
        nxt = torch.empty(1, device=dev, dtype=torch.int32)
        vcnt = torch.empty(1, device=dev, dtype=torch.int32)
        spine_root = torch.zeros(1, device=dev, dtype=torch.int32)  # identity root row
        pool.append({"draft": draft, "argmax": argmax, "cu": cu, "bonus": bonus,
                     "out": out, "nxt": nxt, "vcnt": vcnt, "spine_root": spine_root})
    return pool


def sample_accept_schedule(rng, conditional_ladder, depth1_rate, n_steps, spine_len):
    """Sample per-step accept length k from the spine accept distribution: depth-1
    accepts w.p. depth1_rate, then depth d accepts w.p. conditional_ladder[d-1].
    Returns a list of accept-lengths (0..spine_len). The MEAN differs by regime
    (both-bugs accepts a hair deeper) so any accept-length-dependent device cost
    shows up as a regime idle delta."""
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


def make_accept_step(kernel, pool, schedule, max_spec_len, interleaved, filler,
                     gemm_per_op, with_spine_root: bool):
    """One scheduled decode step: (interleaved) issue gemm_per_op filler GEMMs to
    give the GPU the concurrent step-GEMM work, then launch the accept-prep kernel
    on the scheduled accept-length's inputs (grid=(1,), conc=1 decode)."""
    counter = {"i": 0}
    n = len(schedule)

    def one_step():
        k = schedule[counter["i"] % n]
        counter["i"] += 1
        inp = pool[k]
        if interleaved:
            for _ in range(gemm_per_op):
                filler()
        if with_spine_root:
            kernel[(1,)](inp["out"], inp["nxt"], inp["vcnt"], inp["cu"], inp["draft"],
                         inp["argmax"], inp["bonus"], inp["spine_root"], max_spec_len)
        else:
            kernel[(1,)](inp["out"], inp["nxt"], inp["vcnt"], inp["cu"], inp["draft"],
                         inp["argmax"], inp["bonus"], max_spec_len)
    return one_step


def measure_paired(specs, filler, gemm_per_op, max_spec_len, warmup, n_iter,
                   n_passes, rounds):
    """PAIRED measurement: in EACH round, measure every variant back-to-back (same
    thermal/scheduling block), so the per-round marginal (both - descent) cancels
    common-mode event-timer drift. The robust cost signal is device-busy (profiler
    self-time, a clean floor -- NOT a span-busy subtraction); the interleaved idle
    is the launch bubble (regime-invariant: every variant issues ONE accept-prep
    launch) and is reported only as the GPU-hidden confirmation."""
    data = {name: {"busy_iso": [], "idle_iso": [], "busy_inter": [], "idle_inter": []}
            for name, *_ in specs}
    for _ in range(rounds):
        for name, kernel, pool, schedule, wsr in specs:
            iso = make_accept_step(kernel, pool, schedule, max_spec_len, False,
                                   filler, gemm_per_op, wsr)
            inter = make_accept_step(kernel, pool, schedule, max_spec_len, True,
                                     filler, gemm_per_op, wsr)
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
            "within_ci": abs(s.get("mean", 0.0)) <= s.get("ci95_abs", 0.0) + 1e-9}


# ===== self-test: reproduce the #136 1.2182 anchor on THIS A10G ==================
def run_self_test(args) -> dict:
    """Re-run the #136 measure_overlap_hidden_idle and recompose the depth-9 step.
    Pass iff |step - 1.2182| / 1.2182 <= tol."""
    import star_attn_fp32_steptime as sas
    counts = {"sliding": sas.N_SLIDING, "full": sas.N_FULL}
    m = sas.measure_overlap_hidden_idle(
        M=32, ctx=args.ctx, n_passes=args.eager_passes, warmup=args.warmup,
        counts=counts, n_iter=args.n_iter, gemm_n=args.gemm_filler_n)
    idle_us = m["exposed_idle_overlap_us"]
    step_repro = ROOFLINE_STEP + idle_us / STEP_M8_US
    delta_pct = 100.0 * abs(step_repro - MEASURED_STEP_136) / MEASURED_STEP_136
    passes = delta_pct <= SELFTEST_TOL_PCT
    return {
        "roofline_step": ROOFLINE_STEP,
        "idle_overlap_us": idle_us,
        "per_call_idle_overlap_us": m["per_call_idle_overlap_us"],
        "step_reproduced": step_repro,
        "anchor_136": MEASURED_STEP_136,
        "delta_vs_anchor_pct": delta_pct,
        "tol_pct": SELFTEST_TOL_PCT,
        "gemm_per_call": m["gemm_per_call"], "filler_us_each": m["filler_us_each"],
        "device_busy_us": m["device_busy_us"], "eager_span_us": m["eager_span_us"],
        "step_cost_self_test_passes": bool(passes),
    }


def run(args) -> dict:
    assert torch.cuda.is_available(), "CUDA required (set CUDA_VISIBLE_DEVICES=0 on this pod)"
    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    rng = __import__("random").Random(args.seed)

    res: dict = {
        "pr": 161, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "l2_bytes": torch.cuda.get_device_properties(0).L2_cache_size,
        "anchors": {
            "k_cal": K_CAL, "step_m8_us": STEP_M8_US,
            "roofline_step": ROOFLINE_STEP, "measured_step_136": MEASURED_STEP_136,
            "e_t_descent_only": E_T_DESCENT_ONLY, "e_t_both_bugs": E_T_BOTH_BUGS,
            "depth1_descent": DEPTH1_DESCENT, "depth1_both": DEPTH1_BOTH,
            "official_both_assumed_roofline": OFFICIAL_BOTH_ASSUMED_ROOFLINE,
        },
        "config": {
            "seed": args.seed, "spine_len": args.spine_len, "max_spec_len": args.max_spec_len,
            "n_iter": args.n_iter, "warmup": args.warmup, "n_passes": args.n_passes,
            "rounds": args.rounds, "sched_steps": args.sched_steps,
            "gemm_filler_n": args.gemm_filler_n, "hidden_threshold_us": args.hidden_threshold_us,
        },
    }

    # ---- 1. SELF-TEST (PRIMARY): reproduce the 1.2182 anchor ------------------
    print("[both-bugs] self-test: reproducing the #136 1.2182 step anchor ...", flush=True)
    res["self_test"] = run_self_test(args)
    st = res["self_test"]
    print(f"[both-bugs] self-test step={st['step_reproduced']:.4f} vs anchor "
          f"{MEASURED_STEP_136} (delta {st['delta_vs_anchor_pct']:.3f}% <= {SELFTEST_TOL_PCT}%) "
          f"-> passes={st['step_cost_self_test_passes']}", flush=True)

    # ---- op-count model from denken #133 root-cause ---------------------------
    # The served accept-prep kernel is byte-identical between descent-only and
    # both-bugs; bug-1 is an UPSTREAM target_logits_indices fix (rank-2 -> rank-1
    # at depth-1). Per-step accept-prep kernel op-count delta from the fix = 0.
    res["opcount_model"] = {
        "source": "research/validity/fp32_star_verify_crosscheck/rootcause_results.json (denken #133)",
        "bug1_dominant_cause": "plumbing",
        "plumbing_frac_of_deficit": 0.9600264278077516,
        "intrinsic_pp": 0.0,
        "fix": ("correct target_logits_indices so the depth-1 root verify-row compares "
                "against the drafter rank-1 (top1) token -- UPSTREAM of the accept-prep "
                "kernel; the kernel consumes target_argmax unchanged."),
        "kernel_op_count_delta_both_vs_descent": 0,
        "kernel_bytewise_identical_baseline_vs_pupa": True,
        "worst_case_modeled_extra_ops_depth1_spine": {
            "extra_indirection_load": 1, "extra_compare": 0, "extra_store": 0,
            "note": "+1 target_logits_indices deref at pos==0 only; over-models the "
                    "real (0-op) plumbing fix.",
        },
    }

    # ---- filler GEMM sized to the step's non-attention GPU work ---------------
    a = torch.randn(args.gemm_filler_n, args.gemm_filler_n, dtype=torch.bfloat16, device=dev)
    b = torch.randn(args.gemm_filler_n, args.gemm_filler_n, dtype=torch.bfloat16, device=dev)
    c = torch.empty(args.gemm_filler_n, args.gemm_filler_n, dtype=torch.bfloat16, device=dev)

    def filler():
        torch.mm(a, b, out=c)

    filler_us = _profiled_device_us(torch, filler, args.n_iter, args.warmup)
    # the accept-prep kernel is ONE launch per step; overlap it with the step's
    # ~9150us non-attn GEMM work (the conc=1 decode is ~92% weight-GEMM, BASELINE.md).
    step_gpu_us = 9149.573677586994  # #143 step_gpu_us_target (gemm_mult + drafter)
    gemm_per_op = max(1, round(step_gpu_us / filler_us))
    res["filler"] = {"gemm_n": args.gemm_filler_n, "filler_us_each": filler_us,
                     "step_gpu_us_target": step_gpu_us, "gemm_per_op": gemm_per_op,
                     "realized_filler_us_per_step": filler_us * gemm_per_op}

    # ---- 2. ACCEPT-PREP MICRO-BENCH (paired) ----------------------------------
    baseline_k, depth1_k = _build_kernels()
    pool = build_input_pool(dev, args.spine_len, args.max_spec_len)

    sched_descent = sample_accept_schedule(rng, RISING_SPINE, DEPTH1_DESCENT,
                                           args.sched_steps, args.spine_len)
    sched_both = sample_accept_schedule(rng, RISING_SPINE, DEPTH1_BOTH,
                                        args.sched_steps, args.spine_len)
    res["schedule_stats"] = {
        "descent_only": {"mean_accept_len": statistics.fmean(sched_descent),
                         "depth1_rate": DEPTH1_DESCENT, "n_steps": len(sched_descent)},
        "both_bugs": {"mean_accept_len": statistics.fmean(sched_both),
                      "depth1_rate": DEPTH1_BOTH, "n_steps": len(sched_both)},
    }

    print("[both-bugs] paired accept-prep micro-bench (descent / both / worst per round) ...", flush=True)
    specs = [
        ("descent_only", baseline_k, pool, sched_descent, False),
        ("both_bugs", baseline_k, pool, sched_both, False),
        ("both_bugs_worst_case", depth1_k, pool, sched_both, True),
    ]
    measured, raw = measure_paired(specs, filler, gemm_per_op, args.max_spec_len,
                                   args.warmup, args.n_iter, args.n_passes, args.rounds)
    res["accept_prep_measured"] = measured

    m_descent, m_both, m_worst = (measured["descent_only"], measured["both_bugs"],
                                  measured["both_bugs_worst_case"])

    # ---- 3. PROPAGATE (lead with device-busy: the robust GPU-work signal) ------
    # device-busy (profiler self-time) is the actual GPU work the accept-prep kernel
    # adds to the step; idle is the launch bubble (regime-invariant: 1 launch each).
    busy_descent = m_descent["device_busy_isolation_us"]
    busy_both = m_both["device_busy_isolation_us"]
    busy_worst = m_worst["device_busy_isolation_us"]
    # PAIRED marginals cancel common-mode event-timer drift.
    busy_marg = paired_marginal(raw, "both_bugs", "descent_only", "busy_iso")
    busy_marg_worst = paired_marginal(raw, "both_bugs_worst_case", "descent_only", "busy_iso")
    idle_marg = paired_marginal(raw, "both_bugs", "descent_only", "idle_inter")
    idle_marg_worst = paired_marginal(raw, "both_bugs_worst_case", "descent_only", "idle_inter")

    # the bug-1 marginal GPU work (paired device-busy median) is the step cost.
    marginal_busy_us = busy_marg["median"]
    marginal_busy_worst_us = busy_marg_worst["median"]
    dstep_marginal = marginal_busy_us / STEP_M8_US
    dstep_worst = marginal_busy_worst_us / STEP_M8_US

    # both-bugs operative step vs the descent-only 1.2182 anchor (descent accept-prep
    # GPU-hidden per #143 -> descent operative step ~= the 1.2182 anchor).
    both_bugs_step_pinned = MEASURED_STEP_136 + max(0.0, dstep_marginal)
    both_bugs_step_worst = MEASURED_STEP_136 + max(0.0, dstep_worst)
    both_bugs_step_delta_pct = 100.0 * (both_bugs_step_pinned - MEASURED_STEP_136) / MEASURED_STEP_136
    both_bugs_step_delta_worst_pct = 100.0 * (both_bugs_step_worst - MEASURED_STEP_136) / MEASURED_STEP_136

    # official at the pinned step (measured-overlap) AND at the roofline anchor (the
    # 537.8 was computed at roofline 1.2127; report both for an honest bracket).
    off_pinned_overlap = fern_official(E_T_BOTH_BUGS, both_bugs_step_pinned, TAU_FERN_CENTRAL)
    off_pinned_roofline = fern_official(E_T_BOTH_BUGS, ROOFLINE_STEP + max(0.0, dstep_marginal), TAU_FERN_CENTRAL)
    off_pinned_worst_roofline = fern_official(E_T_BOTH_BUGS, ROOFLINE_STEP + max(0.0, dstep_worst), TAU_FERN_CENTRAL)
    off_assumed_overlap = fern_official(E_T_BOTH_BUGS, MEASURED_STEP_136, TAU_FERN_CENTRAL)

    # whole-accept-prep hidden fraction: the ENTIRE kernel device-busy as % of step.
    accept_busy_pct_of_step = 100.0 * busy_both / STEP_M8_US / MEASURED_STEP_136
    # GPU-hidden gate (paired, device-busy-led):
    #  - the whole accept-prep kernel device-busy << step AND its launch idle <= thresh
    #    in every regime (sync-free, overlapped behind the ~92%-weight-GEMM step);
    #  - the bug-1 marginal moves the step by a PRACTICALLY negligible amount
    #    (< NEUTRAL_STEP_PCT). device-busy is so repeatable that a sub-100ns marginal
    #    can sit outside its own tiny CI yet be physically nil -- so gate on practical
    #    magnitude, not statistical within-CI (the CI is reported as a diagnostic).
    idle_hidden = all(measured[n]["idle_interleaved_us"] <= args.hidden_threshold_us
                      for n in measured)
    busy_hidden = busy_both <= args.hidden_threshold_us and busy_worst <= args.hidden_threshold_us
    marginal_within_ci = busy_marg["within_ci"]          # reported diagnostic, not a gate
    real_neutral = both_bugs_step_delta_pct < NEUTRAL_STEP_PCT
    worst_neutral = both_bugs_step_delta_worst_pct < NEUTRAL_STEP_PCT
    step_neutral = bool(real_neutral and busy_hidden and idle_hidden)

    if step_neutral and worst_neutral:
        verdict = "GREEN"
        reason = (f"both-bugs accept-prep is STEP-NEUTRAL vs descent-only. bug-1 is upstream "
                  f"target_logits_indices plumbing (0 added kernel ops, denken #133); the served "
                  f"accept-prep kernel is byte-identical and consumes target_argmax unchanged. "
                  f"MEASURED device-busy {busy_descent:.3f}us (descent) vs {busy_both:.3f}us "
                  f"(both-bugs): paired marginal {marginal_busy_us:+.4f}us "
                  f"(ci95 {busy_marg['ci95_abs']:.4f}) -> +{both_bugs_step_delta_pct:.4f}% step "
                  f"(< {NEUTRAL_STEP_PCT}% practical floor ~= 0.5 TPS). The whole accept-prep "
                  f"kernel is {accept_busy_pct_of_step:.4f}% of the 9150us step (1 sync-free launch, "
                  f"GPU-hidden behind the ~92%-weight-GEMM step). Even the WORST-CASE explicit "
                  f"depth-1 conditional adds only {marginal_busy_worst_us:+.4f}us device-busy "
                  f"(+{both_bugs_step_delta_worst_pct:.4f}% step). The 537.8 both-bugs official "
                  f"LOCKS IN (both_bugs_official_pinned {off_pinned_roofline:.2f} @roofline / "
                  f"{off_pinned_overlap:.2f} @overlap).")
    elif step_neutral:
        verdict = "AMBER"
        reason = (f"both-bugs accept-prep real fix is step-neutral (paired marginal device-busy "
                  f"{marginal_busy_us:+.4f}us = +{both_bugs_step_delta_pct:.4f}% step < "
                  f"{NEUTRAL_STEP_PCT}%) but the WORST-CASE explicit depth-1 conditional adds "
                  f"+{both_bugs_step_delta_worst_pct:.4f}% step (>= {NEUTRAL_STEP_PCT}%) -- flag "
                  f"the kernel-op variant; the 537.8 holds for the real (0-op plumbing) fix.")
    else:
        verdict = "RED"
        reason = (f"both-bugs accept-prep adds measurable per-step GPU work: paired marginal "
                  f"device-busy {marginal_busy_us:+.4f}us = +{both_bugs_step_delta_pct:.4f}% step "
                  f"(>= {NEUTRAL_STEP_PCT}% practical floor) -> both_bugs_official_pinned "
                  f"{off_pinned_roofline:.2f} (< 537.8). Re-score before the build commits.")

    res["propagation"] = {
        "device_busy_descent_us": busy_descent,
        "device_busy_both_bugs_us": busy_both,
        "device_busy_worst_case_us": busy_worst,
        "accept_busy_pct_of_step": accept_busy_pct_of_step,
        "marginal_bug1_busy_paired": busy_marg,
        "marginal_bug1_busy_worst_paired": busy_marg_worst,
        "marginal_bug1_idle_paired": idle_marg,
        "marginal_bug1_idle_worst_paired": idle_marg_worst,
        "idle_interleaved_descent_us": m_descent["idle_interleaved_us"],
        "idle_interleaved_both_us": m_both["idle_interleaved_us"],
        "idle_interleaved_worst_us": m_worst["idle_interleaved_us"],
        "marginal_busy_us": marginal_busy_us,
        "marginal_busy_worst_us": marginal_busy_worst_us,
        "marginal_dstep_units": dstep_marginal,
        "both_bugs_step_pinned": both_bugs_step_pinned,
        "both_bugs_step_pinned_worst": both_bugs_step_worst,
        "descent_only_step_anchor": MEASURED_STEP_136,
        "both_bugs_step_delta_pct": both_bugs_step_delta_pct,
        "both_bugs_step_delta_worst_pct": both_bugs_step_delta_worst_pct,
        "official": {
            "both_bugs_official_assumed_roofline": OFFICIAL_BOTH_ASSUMED_ROOFLINE,
            "both_bugs_official_pinned_roofline": off_pinned_roofline,
            "both_bugs_official_pinned_worst_roofline": off_pinned_worst_roofline,
            "both_bugs_official_assumed_overlap": off_assumed_overlap,
            "both_bugs_official_pinned_overlap": off_pinned_overlap,
            "official_drop_vs_537p8_roofline": OFFICIAL_BOTH_ASSUMED_ROOFLINE - off_pinned_roofline,
        },
        "idle_hidden": idle_hidden,
        "busy_hidden": busy_hidden,
        "marginal_within_ci": marginal_within_ci,
        "real_neutral": real_neutral,
        "worst_neutral": worst_neutral,
        "step_neutral": step_neutral,
        "neutral_step_pct": NEUTRAL_STEP_PCT,
        "hidden_threshold_us": args.hidden_threshold_us,
    }
    res["verdict"] = verdict
    res["verdict_reason"] = reason

    res["primary_metric"] = {"name": "step_cost_self_test_passes",
                             "value": int(st["step_cost_self_test_passes"])}
    res["test_metric"] = {"name": "both_bugs_step_delta_pct", "value": both_bugs_step_delta_pct}
    res["elapsed_s"] = time.time() - t0
    res["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9

    del a, b, c
    torch.cuda.empty_cache()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2))
    print(f"\n[both-bugs] VERDICT={verdict}", flush=True)
    print(f"[both-bugs] step_cost_self_test_passes={st['step_cost_self_test_passes']}  "
          f"both_bugs_step_delta_pct={both_bugs_step_delta_pct:+.4f}%  "
          f"both_bugs_step_pinned={both_bugs_step_pinned:.4f} (vs descent 1.2182)", flush=True)
    print(f"[both-bugs] both_bugs_official_pinned={off_pinned_roofline:.2f} @roofline "
          f"(assumption 537.84)  / {off_pinned_overlap:.2f} @overlap", flush=True)
    print(f"[both-bugs] {reason}", flush=True)
    print(f"[both-bugs] wrote {out_path} ({res['elapsed_s']:.0f}s, peak {res['peak_gpu_gb']:.3f}GB)", flush=True)

    # ---- W&B ------------------------------------------------------------------
    if args.wandb_group and not args.no_wandb:
        try:
            import wandb
            run_w = wandb.init(project="gemma-challenge-senpai", entity="wandb-applied-ai-team",
                               group=args.wandb_group, name=args.wandb_name,
                               config={**res["config"], **res["anchors"], "gpu": res["gpu"]})
            wandb.log({
                "step_cost_self_test_passes": int(st["step_cost_self_test_passes"]),
                "self_test_step_reproduced": st["step_reproduced"],
                "self_test_delta_vs_anchor_pct": st["delta_vs_anchor_pct"],
                "both_bugs_step_delta_pct": both_bugs_step_delta_pct,
                "both_bugs_step_delta_worst_pct": both_bugs_step_delta_worst_pct,
                "both_bugs_step_pinned": both_bugs_step_pinned,
                "both_bugs_official_pinned_roofline": off_pinned_roofline,
                "both_bugs_official_pinned_overlap": off_pinned_overlap,
                "descent_only_accept_busy_us": busy_descent,
                "both_bugs_accept_busy_us": busy_both,
                "both_bugs_worst_busy_us": busy_worst,
                "accept_busy_pct_of_step": accept_busy_pct_of_step,
                "descent_only_accept_idle_us": m_descent["idle_interleaved_us"],
                "both_bugs_accept_idle_us": m_both["idle_interleaved_us"],
                "both_bugs_worst_idle_us": m_worst["idle_interleaved_us"],
                "marginal_bug1_busy_us": marginal_busy_us,
                "marginal_bug1_busy_worst_us": marginal_busy_worst_us,
                "marginal_bug1_busy_ci95_us": busy_marg["ci95_abs"],
                "step_neutral": int(step_neutral),
                "verdict_green": int(verdict == "GREEN"),
            })
            run_w.summary["verdict"] = verdict
            res["wandb_run_id"] = run_w.id
            run_w.finish()
            out_path.write_text(json.dumps(res, indent=2))
            print(f"[both-bugs] W&B run {res['wandb_run_id']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[both-bugs] W&B logging skipped: {exc!r}", flush=True)

    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spine-len", type=int, default=9, help="depth-9 tree spine length")
    ap.add_argument("--max-spec-len", type=int, default=9)
    ap.add_argument("--sched-steps", type=int, default=512, help="accept-length schedule length")
    ap.add_argument("--n-iter", type=int, default=80, help="profiler device-busy iters")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--n-passes", type=int, default=300, help="event-span passes (isolation)")
    ap.add_argument("--rounds", type=int, default=5, help="repeat each regime, take median idle")
    ap.add_argument("--eager-passes", type=int, default=200, help="self-test anchor passes")
    ap.add_argument("--ctx", type=int, default=528, help="self-test attention ctx (#136)")
    ap.add_argument("--gemm-filler-n", type=int, default=GEMM_FILLER_N)
    ap.add_argument("--hidden-threshold-us", type=float, default=60.0,
                    help="idle <= this under overlap == GPU-hidden (#136 43us / #143 38us)")
    ap.add_argument("--seed", type=int, default=161)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "research/both_bugs_step_cost/both_bugs_step_cost.json")
    ap.add_argument("--wandb-group", type=str, default="both-bugs-step-cost")
    ap.add_argument("--wandb-name", type=str, default="lawine/both-bugs-step-cost")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--quick", action="store_true", help="fast smoke (few passes/rounds)")
    args = ap.parse_args(argv)
    if args.quick:
        args.n_passes, args.rounds, args.eager_passes, args.sched_steps = 60, 2, 40, 64
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
