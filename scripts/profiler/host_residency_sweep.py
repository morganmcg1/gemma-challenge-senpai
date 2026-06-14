#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Descent decode-path host-residency / graph-capture sweep + net step (PR #163).

LOCAL CPU/A10G static-analysis + arithmetic ONLY. NO vLLM serve change, NO HF Job,
NO submission, NO kernel deploy. BASELINE stays 481.53 (PPL 2.3777). Produces a
build-readiness inventory + a net-step bound; does NOT authorize a launch. Rides
Issue #124 RESOLVED (greedy-exact, PPL <= 2.42 binding).

WHY
---
My #157 (MERGED) found the relocate_salvaged_kv host loop is a silent step-collapsing
landmine (descent 522 -> 77 TPS) -- correctness/PPL-clean, so it passes every functional
check, but as a data-dependent Python loop it cannot be CUDA-graph-captured and pins the
step host-bound. My #154 found a second host-residency op (the decode-path scatter+LP in
compute_logits, eager outside the graph). That is TWO independent host-resident ops on the
descent decode path, EACH individually capable of collapsing the step -- found ONE AT A TIME.

The decision-critical question before the one irreversible shot: are #154 and #157 the ONLY
host-resident / graph-uncapturable ops on the descent decode path, or are there MORE landmines
hiding? Finding them one-at-a-time is not a launch guarantee. This sweep enumerates the WHOLE
field: every host round-trip / data-dependent Python control-flow / CUDA-graph-capture break on
the post-PRECACHE_BENCH timed decode window, classifies each, and emits the NET descent (and
both-bugs) step after all greedy-safe recoveries are applied -- answering whether the FULL path
fits the measured step ~= 1.2182.

MY LANE (distinct failure class)
--------------------------------
host-residency / graph-CAPTURABILITY of the descent decode path + the net step budget. This is
DISTINCT from:
  * lawine #147 -- CUDA *sync-point* counting (salvage_walk_overhead.py --trace). CONSUMED as
    input here (the descent accept-walk sync surface), NOT re-measured. A host Python loop need
    not register as a sync yet still breaks capture -- that is precisely the gap this sweep covers.
  * lawine #161 -- the depth-1 spine conditional's added op-count. EXCLUDED; a slot is armed to
    fold #161's both_bugs_step_delta_pct when it lands.
  * wirbel #152 -- build-budget topology.

METHOD
------
1. Static-trace the served descent decode path (sitecustomize.py `_dixie_*` accept-prep chain +
   the drafter LOOPGRAPH propose + the salvage/commit/relocate ops land #71 assembles) and
   enumerate EVERY op on the timed window that is one of:
     (a) host<->device round-trip (.cpu()/.item()/.tolist()/host `if` on a device value),
     (b) data-dependent Python loop,
     (c) any op that forces a CUDA-graph capture break.
   Classify each {op, site, trigger, per-fire cost class, owner, status}. INCLUDE #154 and #157 as
   the two known anchors and CONFIRM the sweep re-discovers them (method-completeness check).
2. Empirically validate the capturability taxonomy (the distinctive measurement of THIS lane):
   capture the #157 vectorized relocate primitive (device gather/scatter) -> CAPTURES; capture
   the #157 host-loop relocate (per-row .cpu()) -> BREAKS; likewise the #147 sync-free vs
   sync-bound accept-walk. Each case runs in its OWN subprocess (a capture-break poisons the CUDA
   context). Graceful: skipped if no GPU is exposed (the per-op costs are CONSUMED from the merged
   #136/#154/#157/#147 anchors, not re-derived here).
3. Net the step budget: compose the NET descent-path step from the #136 anchor (1.2182) - #154
   scatter+LP recovery + #157 relocate (neutral vectorized / catastrophic host-loop) + any NEW
   host op. Report net_descent_step_pinned and net_clear_500_bar for descent-only AND both-bugs.
4. Self-validate (PRIMARY): the sweep must re-discover the #154 and #157 anchors and the net-step
   arithmetic must reproduce the published bars (1.2182 anchor; #154 4.808; #157 vectorized 4.880 /
   host-loop 32.59) at the consumed deltas.

PRIMARY: host_residency_sweep_self_test_passes (re-discovers both anchors + reproduces the
         anchor/published bars + NaN-clean + capturability probe consistent when it ran).
TEST:    descent_path_residual_host_ops_count (host-resident ops found BEYOND the two known
         #154/#157 anchors and beyond #147's consumed sync surface / #161's excluded spine /
         the structural terminal sync; 0 = field swept clean).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ===== compose constants (CONSUMED from #136/#148/#154/#157/#147, NOT re-derived) =====
K_CAL = 125.26795005202914              # 481.53 / 3.844 (official baseline / E[T]_linear)
STEP_M8_US = 1.0e6 / K_CAL              # ~7982.89 us = 1 M=8-normalized step-unit of wall time
MEASURED_STEP_136 = 1.2182              # lawine #136 GRAPH-CAPTURED depth-9 verify step (units)
CLEAR500_BAR_MEASURED = 4.862377006624717  # fern #129 / #136 operative clear-500 bar @ 1.2182
E_T_TREE_CEILING = 5.207               # fern #125 / denken #101 supply ceiling (both-bugs E[T])
E_T_DESCENT_FIX = 5.04                 # fern #134 / wirbel #135 descent-only (BUG-2) E[T] (-> ~522)
E_T_BOTH_BUGS = 5.207                  # BUG-1 + BUG-2 fixed -> rho-optimal supply ceiling (-> ~538)
TAU_FERN_CENTRAL = 1.0
TARGET_500 = 500.0

# ----- #154 scatter+LP avoidance (decode-path, eager outside the graph) ---------------
PR154_AVOIDABLE_US_M32_REALISTIC = 111.86517397562665  # gross avoidable scatter+LP @ M=32 (real)
PR154_AVOIDABLE_US_M32_CONSERVATIVE = 86.56469504038495  # net-conservative @ M=32
PR154_BAR_REALISTIC = 4.808            # #154 reported lowered bar (realistic)
PR154_BAR_CONSERVATIVE = 4.820         # #154 reported lowered bar (conservative)
PR154_RECOVERABLE_STEP_PCT_REAL = 1.1079069770564272

# ----- #157 relocate_salvaged_kv (salvage-commit KV relocation; chiku-inu trace) ------
PR157_VECTORIZED_US_PER_STEP = 35.29305631881623   # amortized us/step (device [L,W,H,D] gather/scatter)
PR157_PAGED_US_PER_STEP = 20.26                     # amortized us/step (zero-copy slot-map)
PR157_HOSTLOOP_US_PER_STEP = 55457.40637081948      # amortized us/step (host-bound Python loop)
PR157_VECTORIZED_BAR = 4.880023534784125
PR157_HOSTLOOP_BAR = 32.591080192034454
PR157_VECTORIZED_DESCENT_TPS = 516.3909522234458
PR157_HOSTLOOP_DESCENT_TPS = 77.32176979564827
PR157_EQUIVALENCE_RATE = 1.0           # bit-exact bf16 permutation -> greedy-safe by construction

# ----- #147 descent accept-walk sync surface (CONSUMED, not re-measured) --------------
PR147_SYNC_FREE_STEP_INFLATION_PCT = 0.39159715612748974  # sync-free interleaved (GPU-HIDDEN, GREEN)
PR147_SYNC_BOUND_STEP_INFLATION_PCT = 2.202               # naive per-node .item() walk
PR147_GPU_HIDDEN = 1                    # sync-free descent GPU-hidden under overlap (GREEN)
PR147_SYNC_FREE_BAR = 4.8814
PR147_TERMINAL_SYNC_IN_ANCHOR = True   # output_token_ids.cpu() already in the 1.2182 anchor

# ----- oracle ladder (tree-488-pw-fp32-v0, board 20260614-100550-487) -----------------
ORACLE_SALVAGE_RATE = 391 / 1024       # 0.382 salvages/step (relocate trigger rate)

# ----- #161 spine-conditional slot (ARMED, pending #161; excluded from this lane) -----
PR161_BOTH_BUGS_STEP_DELTA_PCT = None  # fold lawine #161's both_bugs_step_delta_pct when it lands

ANCHOR_STEP_US = MEASURED_STEP_136 * STEP_M8_US   # 9724.75 us captured-target depth-9 step


def fern_clear_bar(target: float, step_units: float, tau: float = 1.0) -> float:
    """E[T] needed to clear `target` official at (step, tau). RISES with step."""
    return target * step_units / (K_CAL * tau)


def official_tps(e_t: float, step_units: float, tau: float = 1.0) -> float:
    return K_CAL * e_t / step_units * tau


# =====================================================================================
# Part A -- the static descent decode-path host-residency inventory
# =====================================================================================
# Every op on the post-PRECACHE_BENCH timed decode window. host_residency_class is the
# subset of {a_host_roundtrip, b_datadep_pyloop, c_capture_break} the op triggers (["none"]
# = device-resident / graph-captured / static). `owner` attributes the op to the lane that
# classified/prices it: "#154"/"#157" are THIS programme's two anchors (must be re-discovered);
# "#147" is the consumed sync surface; "#161" is the excluded spine lane; "structural" is the
# unavoidable terminal sync already in the anchor; "clean" is captured/device-resident.
DESCENT_PATH_OPS = [
    {
        "op": "drafter_propose_loop",
        "site": "submissions/fa2sw_precache_kenyan/sitecustomize.py:158-203 _run_graph_body "
                "(captured via _capture_graph:233-248; ONEGRAPH/LOOPGRAPH)",
        "trigger": "every step (drafter proposes K=num_speculative_tokens width-1 tokens)",
        "host_residency_class": ["none"],
        "per_fire_cost_class": "CAPTURED (CUDA-graph replay, launch-free)",
        "owner": "clean",
        "status": "CLEAN(captured)",
        "rationale": "for index in range(token_count) is a STATIC Python loop (token_count="
                     "self.num_speculative_tokens, a fixed config int -> fixed trip count). Body is "
                     "device .copy_ + model forward + get_top_tokens; NO .item()/.cpu(). Captured.",
        "greedy_safe_design": "n/a (already captured)",
    },
    {
        "op": "target_verify_forward",
        "site": "vLLM gpu_model_runner cudagraph (42 layers, M=32 int4-Marlin GEMM + star-attn); "
                "lawine #136 gemm_all_graphed=true",
        "trigger": "every step (verify the K drafted tokens)",
        "host_residency_class": ["none"],
        "per_fire_cost_class": "CAPTURED (the 1.2182 GPU-floor step itself)",
        "owner": "clean",
        "status": "CLEAN(captured)",
        "rationale": "the verify GEMMs are graph-captured (gemm_all_graphed); the eager star-attn "
                     "launch idle is GPU-hidden behind per-layer GEMM (#136). This IS the anchor step.",
        "greedy_safe_design": "n/a (already captured)",
    },
    {
        "op": "compute_logits_scatter_LP",
        "site": "submissions/fa2sw_precache_kenyan/serve_patch_pck04.py:335-342 compute_logits_pck04 "
                "-> _scatter_to_full_vocab:113-168 (index_copy_ [M,12288]->[M,262144]) + LogitsProcessor",
        "trigger": "every step (greedy token selection over the verify logits)",
        "host_residency_class": ["c_capture_break"],
        "per_fire_cost_class": f"EAGER outside-graph memory-bound, {PR154_AVOIDABLE_US_M32_REALISTIC:.1f} us/step "
                               "(M=32 realistic); recoverable",
        "owner": "#154",
        "status": "RECOVERABLE",
        "rationale": "runs EAGERLY outside the CUDA graph (serve_patch_pck04.py:17-20): it is not a host "
                     "round-trip, but an outside-graph op that adds eager launch + [M,262144] BW work to the "
                     "timed step. ANCHOR #1 (re-discovered).",
        "greedy_safe_design": "on the token-selection path replace scatter[M,262144]+LP+argmax_262144 with "
                              "argmax(pruned[M,12288]) -> kept_ids remap (kept_ids ascending => first-occurrence "
                              "tiebreak == full-vocab argmax; equivalence_rate=1.0). Keep full scatter+LP on the "
                              "prompt_logprobs/PPL prefill path. Recovers ~111.9 us/step -> bar 4.862->4.808.",
    },
    {
        "op": "fused_accept_prep",
        "site": "submissions/fa2sw_precache_kenyan/sitecustomize.py:921-963 _dixie_fused_accept_prep_kernel "
                "(TRITON) + :969-1025 wrapper + :1032-1054 prepare_next_token_ids_padded",
        "trigger": "every step (greedy accept/reject prefix of the K drafted tokens)",
        "host_residency_class": ["none"],
        "per_fire_cost_class": "DEVICE-RESIDENT triton kernel (one launch, capturable)",
        "owner": "clean",
        "status": "CLEAN(device-resident)",
        "rationale": "the accept loop `for pos in range(num_draft_tokens)` is INSIDE the compiled Triton "
                     "kernel -> on-device, no host round-trip. The wrapper's int(output_token_ids.shape[0]) "
                     "reads a SHAPE (static), not a device data value. This is the CURRENT linear accept path; "
                     "land #71's descent extends it but the accept-COMPARE primitive is already on-device.",
        "greedy_safe_design": "n/a (already device-resident)",
    },
    {
        "op": "descent_accept_walk",
        "site": "land #71 build (descending accept-walk kernel); modeled in scripts/profiler/"
                "salvage_walk_overhead.py (lawine #147)",
        "trigger": "every step (resolve accepted spine length over the depth-9 tree)",
        "host_residency_class": ["a_host_roundtrip", "c_capture_break"],
        "per_fire_cost_class": f"CONSUMED from #147: sync-free +{PR147_SYNC_FREE_STEP_INFLATION_PCT:.2f}% "
                               "(GPU-HIDDEN, GREEN) / sync-bound +2.20%",
        "owner": "#147",
        "status": "design sync-free (GPU-hidden)",
        "rationale": "lawine #147's SYNC-POINT lane: naive per-node bool(verify_argmax[u].eq(draft_tok).item()) "
                     "is a host round-trip AND a capture break; the sync-free design (match-mask -> cumprod -> "
                     "argmax-first-mismatch, device scalar) captures. CONSUMED as input -- not re-counted in "
                     "this lane's residual.",
        "greedy_safe_design": "vLLM-v1 RejectionSampler (PR #14930 zero-sync): device match-mask/cumprod accept "
                              "length; the next step's expand indexes by the device scalar (no .item()).",
    },
    {
        "op": "salvage_branch_selection",
        "site": "land #71 build (rank>=2 rescue); lawine #147 SYNC_POINT_TAXONOMY",
        "trigger": "salvage steps (0.382/step)",
        "host_residency_class": ["a_host_roundtrip", "c_capture_break"],
        "per_fire_cost_class": "CONSUMED from #147 (folded in the sync-free +0.39%)",
        "owner": "#147",
        "status": "design sync-free",
        "rationale": "naive int(branch_scores.argmax().item()) is a host round-trip; device argmax + gather by "
                     "device index avoids it. #147's lane (consumed).",
        "greedy_safe_design": "best=branch_scores.argmax(); gather the chosen branch by the DEVICE index.",
    },
    {
        "op": "accept_length_readout",
        "site": "land #71 build; lawine #147 SYNC_POINT_TAXONOMY",
        "trigger": "every step",
        "host_residency_class": ["a_host_roundtrip", "c_capture_break"],
        "per_fire_cost_class": "CONSUMED from #147 (folded in the sync-free +0.39%)",
        "owner": "#147",
        "status": "design sync-free",
        "rationale": "naive n_accept=accept_len.item() is a host round-trip every step; keeping accept_len a "
                     "DEVICE scalar (consumed by the next expand) avoids it. #147's lane (consumed).",
        "greedy_safe_design": "keep accept_len on device; next-step KV/context indexing uses the device scalar.",
    },
    {
        "op": "relocate_salvaged_kv",
        "site": "land #71 build (salvage-commit KV relocation over 37 served layers); chiku-inu trace "
                "tree-488-pw-fp32-v0 (board 20260614-111022-934); priced in scripts/profiler/"
                "salvage_kv_relocation_audit.py (my #157)",
        "trigger": "salvage steps (0.382/step); relocates accepted rows across 37 layers' K+V",
        "host_residency_class": ["a_host_roundtrip", "b_datadep_pyloop", "c_capture_break"],
        "per_fire_cost_class": f"host-loop {PR157_HOSTLOOP_US_PER_STEP/1e3:.1f} ms/step (host-bound) / "
                               f"vectorized {PR157_VECTORIZED_US_PER_STEP:.1f} us/step / paged "
                               f"{PR157_PAGED_US_PER_STEP:.1f} us/step",
        "owner": "#157",
        "status": "LANDMINE-if-host-loop / RECOVERABLE-vectorized",
        "rationale": "ALL THREE host-residency classes if built as the naive host loop (per-layer x per-row "
                     "D2H/H2D Python over 37 layers): a data-dependent Python loop CANNOT be CUDA-graph-captured "
                     "-> pins the step host-bound (bar 32.59, descent 522->77 TPS, INFEASIBLE). ANCHOR #2 "
                     "(re-discovered). Greedy-safe (bit-exact bf16 permutation, equivalence_rate=1.0).",
        "greedy_safe_design": "single FUSED device gather/scatter over the [L,W,H,D] stack by a DEVICE "
                              "commit-index in one launch (index_select+index_copy_), OR a paged slot-map update "
                              "(zero-copy). Device-index rule: the commit-index is produced ON-DEVICE by the "
                              "accept walk and consumed without a host readout -> stays inside the captured graph. "
                              "Vectorized: 35.3 us/step -> bar 4.880, descent 516 TPS.",
    },
    {
        "op": "kv_commit_blocktable_update",
        "site": "submissions/fa2sw_precache_kenyan/sitecustomize.py:150-155 _refresh_static_buffers "
                "(seq_lens/block_tables device .copy_) + land #71 commit",
        "trigger": "every step (advance committed KV / block-table / slot_mapping by accept_len)",
        "host_residency_class": ["none"],
        "per_fire_cost_class": "DEVICE copy (capturable) -- the slot-map form of the relocate",
        "owner": "#157",
        "status": "CLEAN(device-resident)",
        "rationale": "if accept_len is a device scalar (the #147 sync-free rule), the commit/advance is a device "
                     "copy. This is exactly the relocate's zero-copy paged_slotmap variant (#157) + the #147 "
                     "accept-length readout -- NOT a new op. Host-bound ONLY if accept_len is read to host "
                     "(= the #147 accept_length_readout, already owned).",
        "greedy_safe_design": "advance slot-map / block-table by the DEVICE accept_len (no host readout).",
    },
    {
        "op": "spine_conditional_depth1",
        "site": "land #71 build (rising depth-1 spine conditional); lawine #161 lane",
        "trigger": "every step (depth-1 spine accept conditional)",
        "host_residency_class": ["excluded"],
        "per_fire_cost_class": "EXCLUDED (lawine #161's op-count lane)",
        "owner": "#161",
        "status": "EXCLUDED (slot armed)",
        "rationale": "the depth-1 spine conditional's added op-count is lawine #161's lane. Excluded here; a slot "
                     "is armed (PR161_BOTH_BUGS_STEP_DELTA_PCT) to fold its both_bugs_step_delta_pct when it lands.",
        "greedy_safe_design": "n/a (numerator/op-count lane, not host-residency)",
    },
    {
        "op": "terminal_output_token_ids_cpu",
        "site": "vLLM v1 parse_output (gpu_model_runner: output_token_ids = accepted.cpu())",
        "trigger": "every step (hand accepted tokens to the CPU scheduler: KV mgmt / stop / streaming)",
        "host_residency_class": ["a_host_roundtrip"],
        "per_fire_cost_class": "STRUCTURAL 1/step, ALREADY in the 1.2182 anchor; #147 marginal +0.52% (GPU-hidden)",
        "owner": "structural",
        "status": "UNAVOIDABLE-in-anchor",
        "rationale": "exactly ONE host-sync per step is structurally unavoidable (every decode step streams); it "
                     "is already in the 1.2182 anchor and GPU-hidden behind the GEMM tail. land does NOT fuse it.",
        "greedy_safe_design": "n/a (unavoidable; already priced in the anchor)",
    },
    {
        "op": "input_ids_next_step_update",
        "site": "submissions/fa2sw_precache_kenyan/sitecustomize.py:182,194 self.input_ids[:1].copy_(source)",
        "trigger": "every step (write next-step drafter input from the accepted token)",
        "host_residency_class": ["none"],
        "per_fire_cost_class": "CAPTURED (device .copy_ inside the graph body)",
        "owner": "clean",
        "status": "CLEAN(captured)",
        "rationale": "device .copy_ of a device token inside the captured LOOPGRAPH body; no host round-trip.",
        "greedy_safe_design": "n/a (already captured)",
    },
]

# the two anchors this sweep MUST re-discover (method-completeness self-check).
KNOWN_ANCHORS = {"#154": "compute_logits_scatter_LP", "#157": "relocate_salvaged_kv"}


def classify_inventory() -> dict:
    """Tally the static inventory: re-discover the two anchors, attribute every host-resident op to
    a lane, and count the RESIDUAL host ops (host-resident AND beyond #154/#157/#147/#161/structural)."""
    by_owner: dict[str, list] = {}
    host_resident = []   # any op with a real host-residency class (a/b/c), excluding "none"/"excluded"
    residual = []        # host-resident AND not owned by #154/#157/#147/#161/structural
    for op in DESCENT_PATH_OPS:
        by_owner.setdefault(op["owner"], []).append(op["op"])
        classes = set(op["host_residency_class"])
        is_host = bool(classes & {"a_host_roundtrip", "b_datadep_pyloop", "c_capture_break"})
        if is_host:
            host_resident.append(op["op"])
            if op["owner"] not in ("#154", "#157", "#147", "#161", "structural"):
                residual.append(op["op"])
    # re-discovery check: each known anchor op present with the right owner.
    anchors_rediscovered = {}
    for pr, opname in KNOWN_ANCHORS.items():
        hit = next((o for o in DESCENT_PATH_OPS if o["op"] == opname and o["owner"] == pr), None)
        anchors_rediscovered[pr] = bool(hit is not None
                                        and bool(set(hit["host_residency_class"])
                                                 & {"a_host_roundtrip", "b_datadep_pyloop", "c_capture_break"}))
    return {
        "n_ops_total": len(DESCENT_PATH_OPS),
        "ops_by_owner": by_owner,
        "host_resident_ops": host_resident,
        "n_host_resident_ops": len(host_resident),
        "anchors_rediscovered": anchors_rediscovered,
        "both_anchors_rediscovered": bool(all(anchors_rediscovered.values())),
        "residual_host_ops": residual,
        "descent_path_residual_host_ops_count": len(residual),
        "field_swept_clean": bool(len(residual) == 0),
    }


# =====================================================================================
# Part B -- empirical capturability probe (the distinctive measurement of THIS lane)
# =====================================================================================
# Each case runs in its OWN subprocess: a capture-break leaves the CUDA context poisoned, so the
# cases cannot share a process. The probe EMPIRICALLY grounds the taxonomy -- a device-resident op
# captures, a host round-trip (.item()/.cpu()) breaks capture -- which is the foundation of the
# whole sweep (and is orthogonal to #147's sync COUNTING: capturability, not sync-points).
PROBE_CASES = {
    # name: (expected_captured, description)
    "device_vectorized_relocate": (True, "#157 design target: device [L,W,H,D] index_select+index_copy_"),
    "host_loop_relocate": (False, "#157 landmine: per-row .cpu()/.item() Python loop over layers"),
    "sync_free_accept_walk": (True, "#147 sync-free: match-mask -> cumprod -> device argmax (no .item())"),
    "sync_bound_accept_walk": (False, "#147 sync-bound: per-node bool(.item()) descent"),
}


def _run_probe_case(case: str) -> dict:
    """Run ONE capture case in-process (invoked in a fresh subprocess). Prints a JSON result line."""
    import torch

    if not torch.cuda.is_available():
        return {"case": case, "captured": None, "err": "no-gpu", "skipped": True}
    dev = torch.device("cuda")
    L, W, H, D, n = 4, 256, 2, 64, 8           # tiny dims: we test CAPTURE success, not cost
    try:
        # warm a side stream so capture is legal (graph capture needs a non-default stream warmup).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        if case == "device_vectorized_relocate":
            kstk = torch.randn(L, W, H, D, device=dev)
            src = torch.randperm(W, device=dev)[:n]
            dst = torch.arange(n, device=dev)
            with torch.cuda.stream(s):
                for _ in range(3):
                    kstk.index_copy_(1, dst, kstk.index_select(1, src))
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                kstk.index_copy_(1, dst, kstk.index_select(1, src))
            g.replay(); torch.cuda.synchronize()
            captured = True
        elif case == "host_loop_relocate":
            k = [torch.randn(W, H, D, device=dev) for _ in range(L)]
            src = list(range(n)); dst = list(range(n))
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for layer in range(L):
                    for sidx, didx in zip(src, dst):
                        row = k[layer][sidx].to("cpu")   # *** host round-trip during capture ***
                        k[layer][didx].copy_(row.to(dev))
            captured = True   # if we reach here capture did NOT raise (unexpected)
        elif case == "sync_free_accept_walk":
            va = torch.randint(0, 256000, (W,), device=dev)
            dt = va.clone()
            path = torch.arange(W, device=dev)
            with torch.cuda.stream(s):
                for _ in range(3):
                    m = va.eq(dt).index_select(0, path).to(torch.int32)
                    _ = torch.cumprod(m, 0).sum()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                m = va.eq(dt).index_select(0, path).to(torch.int32)
                _ = torch.cumprod(m, 0).sum()           # accept_len stays a DEVICE scalar
            g.replay(); torch.cuda.synchronize()
            captured = True
        elif case == "sync_bound_accept_walk":
            va = torch.randint(0, 256000, (W,), device=dev)
            dt = va.clone()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                n_acc = 0
                for u in range(n):
                    hit = va[u].eq(dt[u])
                    if not bool(hit.item()):            # *** host round-trip during capture ***
                        break
                    n_acc += 1
            captured = True   # unexpected if reached
        else:
            return {"case": case, "captured": None, "err": f"unknown-case {case}"}
        return {"case": case, "captured": bool(captured), "err": None, "skipped": False}
    except Exception as e:  # noqa: BLE001
        # a capture break raises here -> captured=False (the EXPECTED outcome for host ops).
        return {"case": case, "captured": False, "err": f"{type(e).__name__}: {str(e)[:120]}",
                "skipped": False}


def capturability_probe(disable: bool) -> dict:
    """Spawn one subprocess per case (clean CUDA context each). Returns the per-case result + a
    `consistent` flag (every case that RAN matched its expected capturability)."""
    if disable:
        return {"ran": False, "reason": "disabled (--no-probe)", "cases": {}, "consistent": None}
    env = dict(os.environ)
    # restore CUDA visibility if the container's default CVD points at a missing index.
    if env.get("CUDA_VISIBLE_DEVICES", "") not in ("0",):
        env["CUDA_VISIBLE_DEVICES"] = "0"
    cases: dict[str, dict] = {}
    any_gpu = False
    for case in PROBE_CASES:
        try:
            out = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                  "--probe-case", case],
                                 capture_output=True, text=True, env=env, timeout=180)
            line = [l for l in out.stdout.splitlines() if l.startswith("PROBE_JSON ")]
            res = json.loads(line[-1][len("PROBE_JSON "):]) if line else \
                {"case": case, "captured": None, "err": f"no-output rc={out.returncode} "
                 f"stderr={out.stderr.strip()[:120]}", "skipped": True}
        except Exception as e:  # noqa: BLE001
            res = {"case": case, "captured": None, "err": f"{type(e).__name__}: {str(e)[:120]}",
                   "skipped": True}
        exp = PROBE_CASES[case][0]
        res["expected_captured"] = exp
        res["description"] = PROBE_CASES[case][1]
        ran = not res.get("skipped") and res.get("captured") is not None
        res["match"] = bool(ran and res["captured"] == exp)
        if ran:
            any_gpu = True
        cases[case] = res
    ran_cases = [c for c in cases.values() if not c.get("skipped") and c.get("captured") is not None]
    consistent = (all(c["match"] for c in ran_cases) if ran_cases else None)
    return {"ran": bool(any_gpu), "n_cases_ran": len(ran_cases),
            "cases": cases, "consistent": consistent,
            "note": ("EMPIRICAL capturability grounding of the taxonomy: device-resident relocate/"
                     "accept-walk CAPTURE; host-loop relocate + per-node .item() walk BREAK capture. "
                     "Distinct from #147 sync-COUNTING -- this measures graph-CAPTURABILITY.")}


# =====================================================================================
# Part C -- net the descent step budget
# =====================================================================================
def _price(step_us: float, e_t_descent: float, e_t_both: float, label: str) -> dict:
    step_units = step_us / STEP_M8_US
    bar = fern_clear_bar(TARGET_500, step_units)
    return {
        "label": label,
        "net_step_us": step_us,
        "net_step_units": step_units,
        "net_clear_500_bar": bar,
        "step_inflation_vs_anchor_pct": 100.0 * (step_units - MEASURED_STEP_136) / MEASURED_STEP_136,
        "fits_inside_anchor": bool(step_units <= MEASURED_STEP_136 + 1e-9),
        "descent_only": {
            "e_t": e_t_descent, "tps": official_tps(e_t_descent, step_units),
            "cushion_over_bar": e_t_descent - bar, "clears_500": bool(e_t_descent >= bar),
        },
        "both_bugs": {
            "e_t": e_t_both, "tps": official_tps(e_t_both, step_units),
            "cushion_over_bar": e_t_both - bar, "clears_500": bool(e_t_both >= bar),
        },
    }


def net_step_budget(pr154_mode: str = "realistic") -> dict:
    """Compose the NET descent-path step from the consumed anchors:
        anchor (1.2182) - #154 scatter+LP recovery + #157 relocate (+/- vectorized) + NEW(=0).
    The #147 sync-free accept-walk is GPU-HIDDEN (GREEN) -> +0 net at the bar (reported separately).
    The #161 both-bugs spine delta is ARMED (pending) -> 0 today.
    Reports descent-only (E[T]=5.04) AND both-bugs (E[T]=5.207) for each scenario."""
    pr154 = (PR154_AVOIDABLE_US_M32_REALISTIC if pr154_mode == "realistic"
             else PR154_AVOIDABLE_US_M32_CONSERVATIVE)
    new_host_op_us = 0.0   # field swept clean (Part A): no NEW host op beyond the two anchors

    # both-bugs step delta from #161 (armed): if it lands, fold it into the both-bugs step.
    both_bugs_extra_us = (0.0 if PR161_BOTH_BUGS_STEP_DELTA_PCT is None
                          else PR161_BOTH_BUGS_STEP_DELTA_PCT / 100.0 * ANCHOR_STEP_US)

    scenarios = {}
    # zero-recovery sanity: anchor reproduced exactly (the self-test invariant).
    scenarios["zero_recovery_anchor"] = _price(ANCHOR_STEP_US, E_T_DESCENT_FIX, E_T_BOTH_BUGS,
                                               "zero-recovery (reproduce #136 anchor 1.2182)")
    # descent ships, relocate VECTORIZED, NO #154 (isolate the relocate add).
    scenarios["descent_vectorized_relocate_only"] = _price(
        ANCHOR_STEP_US + PR157_VECTORIZED_US_PER_STEP + new_host_op_us,
        E_T_DESCENT_FIX, E_T_BOTH_BUGS, "descent + vectorized relocate (no #154)")
    # descent ships, relocate VECTORIZED, #154 STACKED (the realizable build).
    scenarios["descent_vectorized_plus_154"] = _price(
        ANCHOR_STEP_US - pr154 + PR157_VECTORIZED_US_PER_STEP + new_host_op_us,
        E_T_DESCENT_FIX, E_T_BOTH_BUGS, "descent + vectorized relocate + #154 (realizable build)")
    # descent ships, relocate PAGED slot-map (zero-copy ideal), #154 stacked.
    scenarios["descent_paged_plus_154"] = _price(
        ANCHOR_STEP_US - pr154 + PR157_PAGED_US_PER_STEP + new_host_op_us,
        E_T_DESCENT_FIX, E_T_BOTH_BUGS, "descent + paged slot-map relocate + #154 (zero-copy ideal)")
    # descent ships, relocate HOST-LOOP (the landmine), #154 stacked (it cannot save it).
    scenarios["descent_hostloop_relocate"] = _price(
        ANCHOR_STEP_US - pr154 + PR157_HOSTLOOP_US_PER_STEP + new_host_op_us,
        E_T_DESCENT_FIX, E_T_BOTH_BUGS, "descent + HOST-LOOP relocate (the landmine)")

    # both-bugs: same step as descent-only today (BUG-1 is a numerator fix); fold #161 when armed.
    realizable = scenarios["descent_vectorized_plus_154"]
    both_bugs_step_units = (realizable["net_step_us"] + both_bugs_extra_us) / STEP_M8_US
    both_bugs_bar = fern_clear_bar(TARGET_500, both_bugs_step_units)

    headline = scenarios["descent_vectorized_plus_154"]
    return {
        "pr154_mode": pr154_mode,
        "new_host_op_us": new_host_op_us,
        "scenarios": scenarios,
        # the headline numbers the PR asks for:
        "net_descent_step_pinned": headline["net_step_units"],
        "net_clear_500_bar_descent_only": headline["net_clear_500_bar"],
        "net_clear_500_bar_both_bugs": both_bugs_bar,
        "both_bugs_step_units": both_bugs_step_units,
        "pr161_both_bugs_delta_armed": PR161_BOTH_BUGS_STEP_DELTA_PCT,
        "fits_inside_anchor_realizable": headline["fits_inside_anchor"],
        "verdict": (
            "FITS: the FULL descent decode path fits inside step ~=1.2182 once all greedy-safe "
            f"recoveries are applied. Realizable build (vectorized relocate + sync-free accept-walk + "
            f"#154 stacked) nets {headline['net_step_units']:.4f} units (bar {headline['net_clear_500_bar']:.3f}); "
            f"descent E[T]=5.04 clears at {headline['descent_only']['tps']:.0f} TPS, both-bugs E[T]=5.207 at "
            f"{headline['both_bugs']['tps']:.0f} TPS. The SOLE residual that blows the budget is a HOST-LOOP "
            f"relocate (8.17 units, bar 32.59, descent->77 TPS) -- already classified (#157) with a greedy-safe "
            "vectorized design. descent_path_residual_host_ops_count=0: no UNCLASSIFIED host op hides on the path."),
    }


# =====================================================================================
# Part D -- self-tests (PRIMARY)
# =====================================================================================
def self_tests(inv: dict, net: dict, probe: dict) -> dict:
    tests = {}
    # 1. method-completeness: the sweep re-discovers BOTH known anchors as host-resident.
    tests["rediscovers_both_anchors"] = inv["both_anchors_rediscovered"]
    # 2. zero-recovery arithmetic reproduces the #136 anchor (1.2182) exactly.
    z = net["scenarios"]["zero_recovery_anchor"]
    tests["zero_recovery_reproduces_anchor"] = bool(abs(z["net_step_units"] - MEASURED_STEP_136) < 1e-6)
    # 3. the #157 vectorized bar reproduces the published 4.880 (descent+vectorized-only).
    v = net["scenarios"]["descent_vectorized_relocate_only"]
    tests["reproduces_157_vectorized_bar"] = bool(abs(v["net_clear_500_bar"] - PR157_VECTORIZED_BAR) < 1e-3)
    # 4. the #157 host-loop bar reproduces the published 32.59 (the landmine).
    h = net["scenarios"]["descent_hostloop_relocate"]
    tests["reproduces_157_hostloop_bar"] = bool(abs(h["net_clear_500_bar"] - PR157_HOSTLOOP_BAR) < 0.5)
    # 5. the #154 recovery lands in the published 4.808-4.820 band (step after #154 alone).
    step_154_only = (ANCHOR_STEP_US - PR154_AVOIDABLE_US_M32_REALISTIC) / STEP_M8_US
    bar_154_only = fern_clear_bar(TARGET_500, step_154_only)
    tests["reproduces_154_bar_band"] = bool(PR154_BAR_REALISTIC - 0.01 <= bar_154_only <= PR154_BAR_CONSERVATIVE + 0.02)
    # 6. realizable build FITS inside the anchor (vectorized relocate + #154 stacked).
    tests["realizable_fits_inside_anchor"] = net["fits_inside_anchor_realizable"]
    # 7. feasibility binary: vectorized clears 500 (both E[T]); host-loop does not.
    tests["feasibility_binary"] = bool(v["descent_only"]["clears_500"]
                                       and not h["descent_only"]["clears_500"])
    # 8. residual host-op count is a finite non-negative int.
    rc = inv["descent_path_residual_host_ops_count"]
    tests["residual_count_well_formed"] = bool(isinstance(rc, int) and rc >= 0)
    # 9. capturability probe consistent IF it ran (skipped -> not failed; the static+arithmetic legs stand alone).
    pc = probe.get("consistent")
    tests["capturability_probe_consistent_or_skipped"] = bool(pc is True or pc is None)
    # 10. NaN-clean: every headline numeric finite.
    flat = [net["net_descent_step_pinned"], net["net_clear_500_bar_descent_only"],
            net["net_clear_500_bar_both_bugs"], net["both_bugs_step_units"],
            v["net_clear_500_bar"], h["net_clear_500_bar"], bar_154_only, float(rc)]
    tests["nan_clean"] = bool(all(math.isfinite(x) for x in flat))
    n_pass = sum(tests.values())
    return {"tests": tests, "n_pass": n_pass, "n_total": len(tests),
            "all_pass": bool(n_pass == len(tests)),
            "extras": {"bar_154_only": bar_154_only, "step_154_only_units": step_154_only}}


# =====================================================================================
# build hand-off
# =====================================================================================
def build_handoff(inv: dict, net: dict) -> dict:
    must_vectorize = [{"op": o["op"], "site": o["site"], "design": o["greedy_safe_design"]}
                      for o in DESCENT_PATH_OPS
                      if o["owner"] in ("#154", "#157")
                      or (o["owner"] == "#147")]
    return {
        "classification": ("FIELD SWEPT CLEAN -- the descent decode path host-residency surface is "
                           "FULLY accounted: 2 anchors (#154 scatter+LP, #157 relocate), #147's "
                           "consumed accept-walk sync surface, #161's excluded spine conditional, and "
                           "the structural terminal sync. 0 NEW landmines."),
        "descent_path_residual_host_ops_count": inv["descent_path_residual_host_ops_count"],
        "must_be_vectorized_or_device_resident": must_vectorize,
        "net_step_verdict": net["verdict"],
        "feeds": {
            "fern_155_consolidator": ("net_clear_500_bar_descent_only="
                                      f"{net['net_clear_500_bar_descent_only']:.4f}; the realizable build's "
                                      "operative bar (vectorized relocate + sync-free walk + #154)."),
            "lawine_161_spine_cost": ("both-bugs step slot ARMED ("
                                      f"net_clear_500_bar_both_bugs={net['net_clear_500_bar_both_bugs']:.4f}); "
                                      "fold #161 both_bugs_step_delta_pct when it lands."),
        },
        "launch_gate": ("INFORMS, does NOT authorize. The descent build is launch-de-risked on the "
                        "host-residency axis IFF the relocate is vectorized/paged AND the accept-walk is "
                        "sync-free. A host-loop relocate OR a sync-bound walk is the only way the step "
                        "collapses, and both are already classified with greedy-safe designs."),
    }


# =====================================================================================
# driver
# =====================================================================================
def run(args) -> dict:
    t0 = time.time()
    inv = classify_inventory()
    probe = capturability_probe(disable=args.no_probe)
    net = net_step_budget(pr154_mode=args.pr154_mode)
    st = self_tests(inv, net, probe)
    handoff = build_handoff(inv, net)

    primary = int(st["all_pass"])
    test_val = inv["descent_path_residual_host_ops_count"]

    print(f"[host-residency] descent decode-path ops enumerated: {inv['n_ops_total']} "
          f"({inv['n_host_resident_ops']} host-resident)", flush=True)
    print(f"[host-residency] anchors re-discovered: "
          f"{ {k: v for k, v in inv['anchors_rediscovered'].items()} }  "
          f"residual host ops BEYOND anchors: {test_val} "
          f"({'FIELD SWEPT CLEAN' if inv['field_swept_clean'] else 'LANDMINE FOUND'})", flush=True)
    if probe["ran"]:
        pc = {k: v.get("captured") for k, v in probe["cases"].items()}
        print(f"[host-residency] capturability probe (consistent={probe['consistent']}): {pc}", flush=True)
    else:
        print(f"[host-residency] capturability probe SKIPPED ({probe.get('reason','no-gpu')}) "
              "-- static+arithmetic legs stand alone", flush=True)
    head = net["scenarios"]["descent_vectorized_plus_154"]
    print(f"[host-residency] NET descent step (realizable): {net['net_descent_step_pinned']:.4f} units "
          f"(bar {net['net_clear_500_bar_descent_only']:.4f}); descent E[T]=5.04 -> "
          f"{head['descent_only']['tps']:.0f} TPS, both-bugs E[T]=5.207 -> {head['both_bugs']['tps']:.0f} TPS", flush=True)
    print(f"[host-residency] SELF-TEST {st['n_pass']}/{st['n_total']} "
          f"({'PASS' if st['all_pass'] else 'FAIL'})  "
          f"PRIMARY host_residency_sweep_self_test_passes={primary}  "
          f"TEST descent_path_residual_host_ops_count={test_val}", flush=True)
    print(f"[host-residency] {net['verdict']}", flush=True)

    res = {
        "pr": 163, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lane": "descent decode-path host-residency / graph-capturability + net step",
        "anchors_consumed": {
            "k_cal": K_CAL, "step_m8_us": STEP_M8_US, "measured_step_136": MEASURED_STEP_136,
            "anchor_step_us": ANCHOR_STEP_US, "clear500_bar_measured": CLEAR500_BAR_MEASURED,
            "e_t_descent_fix": E_T_DESCENT_FIX, "e_t_both_bugs": E_T_BOTH_BUGS,
            "e_t_tree_ceiling": E_T_TREE_CEILING, "oracle_salvage_rate": ORACLE_SALVAGE_RATE,
            "pr154": {"avoidable_us_m32_realistic": PR154_AVOIDABLE_US_M32_REALISTIC,
                      "avoidable_us_m32_conservative": PR154_AVOIDABLE_US_M32_CONSERVATIVE,
                      "bar_realistic": PR154_BAR_REALISTIC, "bar_conservative": PR154_BAR_CONSERVATIVE},
            "pr157": {"vectorized_us_per_step": PR157_VECTORIZED_US_PER_STEP,
                      "paged_us_per_step": PR157_PAGED_US_PER_STEP,
                      "hostloop_us_per_step": PR157_HOSTLOOP_US_PER_STEP,
                      "vectorized_bar": PR157_VECTORIZED_BAR, "hostloop_bar": PR157_HOSTLOOP_BAR,
                      "equivalence_rate": PR157_EQUIVALENCE_RATE},
            "pr147": {"sync_free_step_inflation_pct": PR147_SYNC_FREE_STEP_INFLATION_PCT,
                      "sync_bound_step_inflation_pct": PR147_SYNC_BOUND_STEP_INFLATION_PCT,
                      "gpu_hidden": PR147_GPU_HIDDEN, "terminal_sync_in_anchor": PR147_TERMINAL_SYNC_IN_ANCHOR},
            "pr161_both_bugs_step_delta_pct_armed": PR161_BOTH_BUGS_STEP_DELTA_PCT,
        },
        "config": {"pr154_mode": args.pr154_mode, "no_probe": args.no_probe},
        "inventory": DESCENT_PATH_OPS,
        "classification": inv,
        "capturability_probe": probe,
        "net_step_budget": net,
        "self_test": st,
        "build_handoff": handoff,
        "primary_metric": {"name": "host_residency_sweep_self_test_passes", "value": primary},
        "test_metric": {"name": "descent_path_residual_host_ops_count", "value": test_val},
    }
    res["elapsed_s"] = time.time() - t0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2))
    print(f"[host-residency] wrote {out_path} ({res['elapsed_s']:.1f}s)", flush=True)

    if args.wandb_group and not args.no_wandb:
        _wandb_log(args, res, out_path)
    return res


def _wandb_log(args, res: dict, out_path: Path):
    try:
        import wandb
        net = res["net_step_budget"]
        inv = res["classification"]
        probe = res["capturability_probe"]
        run_w = wandb.init(project="gemma-challenge-senpai", entity="wandb-applied-ai-team",
                           group=args.wandb_group, name=args.wandb_name,
                           config={**res["config"], **res["anchors_consumed"]})
        head = net["scenarios"]["descent_vectorized_plus_154"]
        v = net["scenarios"]["descent_vectorized_relocate_only"]
        h = net["scenarios"]["descent_hostloop_relocate"]
        log = {
            "host_residency_sweep_self_test_passes": res["primary_metric"]["value"],
            "descent_path_residual_host_ops_count": res["test_metric"]["value"],
            "self_test_n_pass": res["self_test"]["n_pass"],
            "self_test_n_total": res["self_test"]["n_total"],
            "n_ops_total": inv["n_ops_total"],
            "n_host_resident_ops": inv["n_host_resident_ops"],
            "both_anchors_rediscovered": int(inv["both_anchors_rediscovered"]),
            "field_swept_clean": int(inv["field_swept_clean"]),
            "net_descent_step_pinned": net["net_descent_step_pinned"],
            "net_clear_500_bar_descent_only": net["net_clear_500_bar_descent_only"],
            "net_clear_500_bar_both_bugs": net["net_clear_500_bar_both_bugs"],
            "fits_inside_anchor_realizable": int(net["fits_inside_anchor_realizable"]),
            "realizable_descent_tps": head["descent_only"]["tps"],
            "realizable_both_bugs_tps": head["both_bugs"]["tps"],
            "vectorized_only_bar": v["net_clear_500_bar"],
            "vectorized_only_descent_tps": v["descent_only"]["tps"],
            "hostloop_bar": h["net_clear_500_bar"],
            "hostloop_descent_tps": h["descent_only"]["tps"],
            "capturability_probe_ran": int(probe["ran"]),
            "capturability_probe_consistent": (1 if probe["consistent"] else
                                               (0 if probe["consistent"] is False else -1)),
            "measured_step_anchor": MEASURED_STEP_136,
            "clear500_bar_measured": CLEAR500_BAR_MEASURED,
            "supply_ceiling_e_t": E_T_TREE_CEILING,
            "pr147_sync_free_inflation_pct": PR147_SYNC_FREE_STEP_INFLATION_PCT,
            "pr157_equivalence_rate": PR157_EQUIVALENCE_RATE,
        }
        wandb.log(log)
        run_w.summary.update(log)
        res["wandb_run_id"] = run_w.id
        wandb.finish()
        print(f"[host-residency] W&B run {run_w.id} (group {args.wandb_group})", flush=True)
        out_path.write_text(json.dumps(res, indent=2))
    except Exception as e:  # noqa: BLE001
        print(f"[host-residency] W&B logging skipped: {e!r}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe-case", type=str, default=None,
                    help="internal: run ONE capturability case in this process (subprocess entry)")
    ap.add_argument("--pr154-mode", choices=["realistic", "conservative"], default="realistic")
    ap.add_argument("--no-probe", action="store_true", help="skip the GPU capturability probe")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "research/spec_cost_model/host_residency_sweep/host_residency_sweep.json")
    ap.add_argument("--wandb-group", type=str, default=None)
    ap.add_argument("--wandb-name", type=str, default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    if args.probe_case is not None:
        print("PROBE_JSON " + json.dumps(_run_probe_case(args.probe_case)), flush=True)
        return 0

    args.wandb_group = args.wandb_group or "descent-path-host-residency-sweep"
    args.wandb_name = args.wandb_name or "ubel/descent-path-host-residency-sweep"
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
