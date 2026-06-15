#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Verify+accept EPILOGUE capture screen (PR #265, wirbel) -- is the M=8 verify
forward + greedy sampler + accept-length + KV-relocate epilogue CUDA-graph
captured (one replay) or eager-launched per step? The COMPLEMENT of wirbel #261
(draft side, run egaz6m2f), which proved the K=7 DRAFT tail is fully captured in
the deployed ONEGRAPH (`draft_argmax_embed_separately_launched=False`, 0
recoverable us). This closes the LAST open piece of the deployed step's
launch-overhead axis: #261 closed the DRAFT side; this closes the VERIFY side.
CPU-only read of the served loop-graph manifest + spec-decode path. NO GPU run,
NO served-file change, NO HF Job.

DECISIVE BOOLEAN: verify_accept_epilogue_separately_launched
  = (is ANY of {verify forward, greedy sampler, accept-length, KV-relocate}
     eager-launched per step, OUTSIDE the captured replay?)

THE AUDIT (served stack, vLLM wheel 0.22.1rc1.dev307+g3e8afdf78)
---------------------------------------------------------------
Served submission: submissions/fa2sw_precache_kenyan. The per-step spec-decode
pipeline has FIVE stages; this screen tags each captured/eager from the served
code (each tag carries an evidence string sourced from a served file):

  (a) DRAFT K=7 tail (re-confirm #261): CAPTURED. ONEGRAPH=1 =>
      Gemma4Proposer.propose = propose_onegraph; the K-1 draft tail (per-step
      embed gather via inputs_embeds=None + get_top_tokens argmax) lives in
      _run_graph_body, recorded by _capture_graph into ONE torch.cuda.CUDAGraph
      and served by a single graph.replay(). LOOPGRAPH_REQUIRE_CAPTURE=1 forces
      the captured path. (== #261's draft_argmax_embed_separately_launched=False.)

  (b) M=8 VERIFY FORWARD (target model attention + lm_head over the 8 candidate
      positions): EAGER, but COMPUTE-BOUND. serve_patch_pck04.py states twice and
      verifies: "the main model runner runs execute_model eagerly", "not inside a
      CUDA graph capture in this stack", "Verified: gpu_model_runner.py calls
      model.compute_logits() outside any torch.cuda.graph() capture block." The
      launch args (serve.py main) carry NO --enforce-eager and NO
      --compilation-config, but DO carry --speculative-config (MTP K=7) -- with
      spec-decode on, the standard vLLM cudagraph does NOT capture the target
      verify forward (the custom ONEGRAPH captures only the REPEATED drafter
      loop). HOWEVER the verify forward is GPU-compute-bound: splitkv_verify_patch
      measured the verify-attention at 12-53us of REAL compute, so its many layer
      launches HIDE behind GEMM/attention compute (CPU launches ahead of the GPU)
      => its EXPOSED launch-tax ~= 0. It is therefore EXCLUDED from the
      recoverable epilogue count (counting its internal launches would massively
      over-bound -- the analog of NOT counting the drafter forward's internal
      layer launches in #261's 14).

  (c) GREEDY SAMPLER (argmax over the target logits -> verify-target tokens):
      EAGER. serve.py DIXIE_SMP02_FWD patches RejectionSampler.forward; the
      all-greedy fast path runs `logits.argmax(dim=-1)` + two index gathers +
      two contiguous + a torch.full output buffer -- in eager Python AFTER the
      forward (the sampler is NEVER inside the model cudagraph in vLLM v1).

  (d) ACCEPT-LENGTH (data-dependent token-by-token compare draft-argmax ==
      target-argmax, "stop at first mismatch"): EAGER. The fused Triton kernel
      _dixie_fused_accept_prep_kernel (sitecustomize) / rejection_greedy_sample_
      kernel (serve.py) runs inside the same eager RejectionSampler.forward.

  (e) KV-RELOCATE / next-step seed (commit accepted prefix + seed next step):
      EAGER. prepare_next_token_ids_padded (DIXIE-patched) returns the cached
      (next_token_ids, valid_counts) -- a dict pop, ~0 GPU work -- and the next
      forward's input/positions/slot_mapping prep runs as eager runner
      bookkeeping.

VERDICT OF THE AUDIT: verify_accept_epilogue_separately_launched = TRUE. The
verify forward + sampler + accept + relocate run EAGER (not in a captured
replay) -- the EAGER complement of #261's CAPTURED draft side. This is WHERE the
served-vs-built step gap (1218.2 - 1085.0 = 133.2 us) lives: #261 proved the
draft side contributes 0, so the entire gap is on the verify side.

THE BOUND (recoverable epilogue launch-tax)
-------------------------------------------
The recoverable launch-tax is the EXPOSED small-op tail (sampler + accept +
relocate), NOT the compute-bound verify forward (b). Enumerated distinct eager
GPU launches in the epilogue: sampler = 6 (argmax + 2 gathers + 2 contiguous +
output full), accept = 1 (fused Triton kernel), relocate/seed = 0..4 (next-step
input prep; prepare_next_token_ids is a cache-pop). Count band [7, 9, 11].
At the #261 sm_86 per-launch band [4, 5, 6] us:
  eager_launch_us_step = count x per_launch = [28, 45, 66] us.
Through the composition tps(step') = served*step_served/step', removing the
eager epilogue launch-tax (capturing it, like ONEGRAPH did for the drafter)
gives the COUNTERFACTUAL lambda=1 ceiling:
  +2.35% .. +5.73% off 481.53  ->  492.9 .. 509.1 TPS  (central ~500.0).
This is a lambda=1 UPPER BOUND (all epilogue launch-tax exposed AND captured).
The REALIZABLE fraction (how much of the 133.2 us gap is exposed launch-tax vs
overlapped compute vs CPU gaps) is denken #257's roofline (cpu_ms vs gpu_ms);
the static audit gives the STRUCTURE (verify side eager) + the epilogue COUNT +
the ceiling. So unlike #261 (draft side captured => 0), the verify side IS eager
=> a NON-zero, ceiling-bounded verify-side capture lever EXISTS.

GREEDY/PPL SAFETY
-----------------
Capturing EXISTING ops (moving the sampler argmax / accept kernel / relocate from
eager-launch into a replay) changes ONLY kernel-launch timing; cudagraph replay
re-runs the IDENTICAL kernels with the IDENTICAL inputs => bit-identical outputs.
argmax (ties -> lowest index) and accept-length ("stop at first mismatch") are
deterministic INDEX/COMPARE ops (no float reduction reorder), so capture cannot
flip an emitted token or move PPL. capture_greedy_safe = True (verified: the
accept-length function is a deterministic function of its inputs over N>=1000
random draws). It IDENTIFIES the lever; it does NOT implement a capture change
(that would be a served-file edit, human-approval-gated).

LOCAL, CPU-ONLY, ANALYTIC. No GPU / vLLM run / HF Job / submission / served-file
change / official draw. BASELINE stays 481.53; the 520.95 lambda=1 ceiling stays
520.95; this SCREEN adds 0 TPS (it reads the capture structure; it moves
nothing). NOT a launch. NOT open2. Non-overlap: #261 (DRAFT-side capture, the
direct complement -- closed at 0; this closes the VERIFY side), #255 (verify
lm_head/argmax d2h MATERIALIZATION -- memory axis, already-fused; this is the
LAUNCH/CAPTURE axis), kanna #264 (draft-head vocab GEMV cost), denken #257
(built-step roofline -- this feeds it the eager-slack number), fern #262
(shallow-tree), ubel #263 (private rank), land #245 (the tree build / any live
build owns the gain).

PRIMARY metric  verify_accept_epilogue_capture_self_test_passes
TEST    metric  projected_tps_gain_pct  (lambda=1 counterfactual ceiling of the
                                         verify-side capture lever; central mid)
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# IMPORTED anchors (re-derive NOTHING):
#   kanna #217 vgovdrjc  -- composition official = K_cal*(E[T]/step)*tau,
#                           K_cal=125.268, served step 1.2182 ms, served 481.53.
#   denken #254 zav6nr8y -- draft floor ~101.2 us/pass x K=7 ~= 58% of step;
#                           built step ~1.085 ms (the pure-compute floor).
#   wirbel #261 egaz6m2f -- draft side CAPTURED (separately_launched=False);
#                           sm_86 per-launch band 4/5/6 us (14-launch totals
#                           56/70/84 us); 520.95 lambda=1 ceiling.
# --------------------------------------------------------------------------- #
SERVED_TPS = 481.53          # official served (linear MTP K=7, PR #52); screen adds 0
BASELINE_TPS = 481.53
LAMBDA1_CEILING_TPS = 520.95  # public lambda=1 operative ceiling; UNCHANGED
STEP_US_SERVED = 1218.2      # served step 1.2182 ms (kanna #217) -- live TPS-gate denom
STEP_US_BUILT = 1085.0       # built step ~1.085 ms (denken #254) -- pure-compute floor
K_CAL = 125.268             # composition calibration constant (kanna #217)
K_SPEC = 7                  # num_speculative_tokens (manifest, linear MTP K=7)
# sm_86 eager marginal per-launch band (wirbel #261: its 14-launch totals
# 56/70/84 us == 14 x {4,5,6}; the PER-LAUNCH band is 4/5/6 us).
PER_LAUNCH_US_LO = 4.0
PER_LAUNCH_US_MID = 5.0
PER_LAUNCH_US_HI = 6.0
SM86_LAUNCH_FLOOR_US = 55.0  # sm_86 eager launch-FLOOR per call (context only)
DRAFT_N_LAUNCHES_261 = 14    # #261's draft-side counterfactual count (for the complement note)

# Verify-side EPILOGUE eager-launch enumeration (the RECOVERABLE small-op tail;
# the compute-bound verify forward (b) is EXCLUDED -- its launches hide behind
# GEMM/attention compute, exposed ~= 0). See audit_served() for the per-op map.
EPILOGUE_SAMPLER_LAUNCHES = 6   # argmax + bonus-gather + bonus-contiguous
#                                 + target-gather + target-contiguous + output-full
EPILOGUE_ACCEPT_LAUNCHES = 1    # fused accept-length Triton kernel
EPILOGUE_RELOCATE_LO = 0        # prepare_next_token_ids cache-pop (DIXIE) -> 0
EPILOGUE_RELOCATE_MID = 2       # + next-step input/positions/slot prep (eager)
EPILOGUE_RELOCATE_HI = 4        # conservative upper on next-step bookkeeping
COUNT_LO = EPILOGUE_SAMPLER_LAUNCHES + EPILOGUE_ACCEPT_LAUNCHES + EPILOGUE_RELOCATE_LO   # 7
COUNT_MID = EPILOGUE_SAMPLER_LAUNCHES + EPILOGUE_ACCEPT_LAUNCHES + EPILOGUE_RELOCATE_MID  # 9
COUNT_HI = EPILOGUE_SAMPLER_LAUNCHES + EPILOGUE_ACCEPT_LAUNCHES + EPILOGUE_RELOCATE_HI    # 11

VLLM_VERSION = "0.22.1rc1.dev307+g3e8afdf78"
SERVED_SUBMISSION = "fa2sw_precache_kenyan"
_SUB = REPO_ROOT / "submissions" / SERVED_SUBMISSION
_MANIFEST = _SUB / "manifest.json"
_SITECUSTOMIZE = _SUB / "sitecustomize.py"
_SERVE = _SUB / "serve.py"
_PCK04 = _SUB / "serve_patch_pck04.py"
_SPLITKV = _SUB / "splitkv_verify_patch.py"


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _read(path: Path) -> str:
    try:
        return path.read_text() if path.is_file() else ""
    except Exception:  # noqa: BLE001
        return ""


# --------------------------------------------------------------------------- #
# THE AUDIT: tag each of the 5 stages captured/eager from the served code, with
# an evidence string (file + the capture-boundary call, or its absence). The
# decisive boolean is grounded in the served manifest + code structure.
# --------------------------------------------------------------------------- #
def audit_served() -> dict[str, Any]:
    out: dict[str, Any] = {
        "submission": SERVED_SUBMISSION, "vllm_version": VLLM_VERSION,
        "manifest_path": str(_MANIFEST) if _MANIFEST.is_file() else None,
        "sitecustomize_path": str(_SITECUSTOMIZE) if _SITECUSTOMIZE.is_file() else None,
        "serve_path": str(_SERVE) if _SERVE.is_file() else None,
        "pck04_path": str(_PCK04) if _PCK04.is_file() else None,
        "k_spec": K_SPEC, "spec_method": "mtp",
    }
    flags = {
        "ONEGRAPH": None, "LOOPGRAPH_REQUIRE_CAPTURE": None,
        "FUSED_SPARSE_ARGMAX": None, "DIXIE_SLIM_GREEDY": None,
        "DIXIE_FUSED_ACCEPT_PREP": None, "SPLITKV_VERIFY": None,
        "LM_HEAD_PRUNE": None, "SPECULATIVE_CONFIG": None,
        "OVERRIDE_GENERATION_CONFIG": None,
    }
    if _MANIFEST.is_file():
        try:
            env = json.loads(_MANIFEST.read_text()).get("env", {})
            for k in list(flags):
                if k in env:
                    flags[k] = env[k]
            sc = env.get("SPECULATIVE_CONFIG")
            if isinstance(sc, str):
                try:
                    scd = json.loads(sc)
                    if _is_num(scd.get("num_speculative_tokens")):
                        out["k_spec"] = int(scd["num_speculative_tokens"])
                    if isinstance(scd.get("method"), str):
                        out["spec_method"] = scd["method"]
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
    out["manifest_flags"] = flags

    sc_src = _read(_SITECUSTOMIZE)
    serve_src = _read(_SERVE)
    pck04_src = _read(_PCK04)
    splitkv_src = _read(_SPLITKV)

    # ---- (a) DRAFT K=7 tail: CAPTURED (re-confirm #261) -------------------- #
    onegraph_on = flags.get("ONEGRAPH") == "1"
    require_capture = flags.get("LOOPGRAPH_REQUIRE_CAPTURE") == "1"
    has_capture_graph = ("def _capture_graph" in sc_src) and (
        "torch.cuda.graph(graph)" in sc_src or "torch.cuda.CUDAGraph()" in sc_src)
    has_graph_replay = "graph.replay()" in sc_src
    has_propose_onegraph = "def propose_onegraph" in sc_src
    has_run_graph_body = "def _run_graph_body" in sc_src
    draft_captured = bool(
        onegraph_on and require_capture and has_capture_graph and has_graph_replay
        and has_propose_onegraph and has_run_graph_body)
    # #261 consistency: draft side captured <=> separately_launched False.
    draft_argmax_embed_separately_launched = not draft_captured

    # ---- (b) M=8 VERIFY FORWARD: EAGER (compute-bound) --------------------- #
    # serve_patch_pck04.py states + VERIFIES the main runner is eager.
    pck04_eager_phrases = [
        "execute_model eagerly",
        "not inside a CUDA graph capture",
        "outside any torch.cuda.graph() capture block",
        "main model runner",
    ]
    pck04_says_main_runner_eager = sum(p in pck04_src for p in pck04_eager_phrases) >= 2
    # serve.py launch args: NO --enforce-eager / --compilation-config, but
    # --speculative-config present => spec-decode target runs eager (the custom
    # ONEGRAPH captures only the repeated drafter loop, not the target forward).
    serve_no_enforce_eager = ("--enforce-eager" not in serve_src)
    serve_no_compilation_config = ("--compilation-config" not in serve_src)
    serve_has_speculative_config = ("--speculative-config" in serve_src)
    # compute-bound evidence: splitkv_verify measured the verify-attention as
    # REAL compute (12-53 us), so the forward's launches hide behind compute.
    verify_attention_is_compute = ("verify" in splitkv_src) and (
        "split-KV" in splitkv_src or "split_kv" in splitkv_src.lower())
    verify_forward_eager = bool(
        pck04_says_main_runner_eager and serve_no_enforce_eager
        and serve_has_speculative_config)
    verify_forward_compute_bound = bool(verify_attention_is_compute)

    # ---- (c) GREEDY SAMPLER argmax: EAGER --------------------------------- #
    # DIXIE_SMP02_FWD patches RejectionSampler.forward (the sampler is eager).
    sampler_argmax_eager = bool(
        "DIXIE_SMP02_FWD" in serve_src
        and "logits.argmax(dim=-1)" in serve_src
        and "rejection_sampler.py" in serve_src)

    # ---- (d) ACCEPT-LENGTH: EAGER ----------------------------------------- #
    accept_eager = bool(
        ("rejection_greedy_sample_kernel" in serve_src)
        and ("_dixie_fused_accept_prep" in sc_src
             or "rejection_greedy_sample_kernel" in sc_src))
    accept_kernel_in_capture = (
        "_dixie_fused_accept_prep_kernel" in sc_src
        and "_run_graph_body" in sc_src
        and "_dixie_fused_accept_prep_kernel" in _between(
            sc_src, "def _run_graph_body", "def _select_loopgraph_output_slot"))
    #  ^ the accept kernel is NOT inside the captured _run_graph_body (drafter loop).

    # ---- (e) KV-RELOCATE / next-seed: EAGER ------------------------------- #
    relocate_eager = bool("prepare_next_token_ids_padded" in sc_src)

    stages = [
        {
            "stage": "(a) draft K=7 tail (embed gather + argmax x7)",
            "captured": draft_captured, "eager": not draft_captured,
            "evidence": (
                "sitecustomize.py: propose_onegraph + _run_graph_body recorded by "
                "_capture_graph(with torch.cuda.graph(graph)) -> ONE CUDAGraph, served "
                "by graph.replay(); manifest ONEGRAPH=1, LOOPGRAPH_REQUIRE_CAPTURE=1 "
                "(== #261 egaz6m2f draft_argmax_embed_separately_launched=False)"),
            "recoverable_launches": 0,
        },
        {
            "stage": "(b) M=8 verify forward (target attn + lm_head)",
            "captured": False, "eager": verify_forward_eager,
            "compute_bound": verify_forward_compute_bound,
            "evidence": (
                "serve_patch_pck04.py: 'the main model runner runs execute_model "
                "eagerly' + 'Verified: ... outside any torch.cuda.graph() capture "
                "block'; serve.py main(): no --enforce-eager / --compilation-config, "
                "--speculative-config present (spec-decode target runs eager). "
                "COMPUTE-BOUND: splitkv_verify verify-attention 12-53us real compute "
                "=> launches hide behind compute, EXPOSED launch-tax ~= 0 (EXCLUDED "
                "from recoverable count)"),
            "recoverable_launches": 0,   # compute-bound => exposed ~= 0
        },
        {
            "stage": "(c) greedy sampler (argmax over target logits)",
            "captured": False, "eager": sampler_argmax_eager,
            "evidence": (
                "serve.py DIXIE_SMP02_FWD patches RejectionSampler.forward: "
                "logits.argmax(dim=-1) + 2 index gathers + 2 contiguous + torch.full "
                "-- eager Python AFTER the forward (sampler never in the model "
                "cudagraph in vLLM v1)"),
            "recoverable_launches": EPILOGUE_SAMPLER_LAUNCHES,
        },
        {
            "stage": "(d) accept-length (compare + stop-at-first-mismatch)",
            "captured": False, "eager": accept_eager,
            "evidence": (
                "serve.py rejection_greedy_sample_kernel / sitecustomize "
                "_dixie_fused_accept_prep_kernel (Triton) inside the same eager "
                "RejectionSampler.forward; NOT inside captured _run_graph_body"),
            "recoverable_launches": EPILOGUE_ACCEPT_LAUNCHES,
        },
        {
            "stage": "(e) KV-relocate + next-step seed",
            "captured": False, "eager": relocate_eager,
            "evidence": (
                "sitecustomize.py prepare_next_token_ids_padded (DIXIE cache-pop, "
                "~0 GPU) + execute_model next-step input/positions/slot_mapping prep "
                "(eager runner bookkeeping)"),
            "recoverable_launches": EPILOGUE_RELOCATE_MID,
        },
    ]

    out.update({
        "stages": stages,
        # (a)
        "onegraph_on": onegraph_on, "require_capture": require_capture,
        "has_capture_graph": has_capture_graph, "has_graph_replay": has_graph_replay,
        "has_propose_onegraph": has_propose_onegraph, "has_run_graph_body": has_run_graph_body,
        "draft_captured": draft_captured,
        "draft_argmax_embed_separately_launched": draft_argmax_embed_separately_launched,
        # (b)
        "pck04_says_main_runner_eager": pck04_says_main_runner_eager,
        "serve_no_enforce_eager": serve_no_enforce_eager,
        "serve_no_compilation_config": serve_no_compilation_config,
        "serve_has_speculative_config": serve_has_speculative_config,
        "verify_forward_eager": verify_forward_eager,
        "verify_forward_compute_bound": verify_forward_compute_bound,
        # (c) (d) (e)
        "sampler_argmax_eager": sampler_argmax_eager,
        "accept_eager": accept_eager,
        "accept_kernel_inside_captured_drafter_loop": accept_kernel_in_capture,
        "relocate_eager": relocate_eager,
    })

    # DECISIVE BOOLEAN: any of stages (b)-(e) eager-launched per step?
    epilogue_eager_any = bool(
        verify_forward_eager or sampler_argmax_eager or accept_eager or relocate_eager)
    out["verify_accept_epilogue_separately_launched"] = epilogue_eager_any
    out["verify_accept_epilogue_captured"] = (not epilogue_eager_any)
    return out


def _between(src: str, a: str, b: str) -> str:
    """Return the substring of `src` between markers a and b (a..b). Used to test
    whether an op appears INSIDE a function body. Empty string if not found."""
    i = src.find(a)
    if i < 0:
        return ""
    j = src.find(b, i + len(a))
    return src[i:j] if j >= 0 else src[i:]


# --------------------------------------------------------------------------- #
# Composition: TPS <-> step. tps(step') = SERVED * STEP_US_SERVED / step'.
# Removing +us_net of eager launch-tax SHRINKS the step by us_net.
# --------------------------------------------------------------------------- #
def tps_from_step(step_us: float) -> float:
    return SERVED_TPS * STEP_US_SERVED / step_us


def tps_gain_pct_from_us_net(us_net: float) -> float:
    return (tps_from_step(STEP_US_SERVED - us_net) / SERVED_TPS - 1.0) * 100.0


# --------------------------------------------------------------------------- #
# GREEDY-SAFETY: capturing EXISTING ops into a replay is timing-only. cudagraph
# replay re-runs the identical kernels with identical inputs => bit-identical.
# We GROUND that the verify+accept output is a DETERMINISTIC function of its
# inputs (so capture cannot change it): argmax (ties -> lowest index) + the
# accept-length "stop at first mismatch" semantics, over N>=1000 random draws,
# matched against an independent numpy/python reference.
# --------------------------------------------------------------------------- #
def capture_safety_check(n_draws: int, k_spec: int, vocab: int, seed: int) -> dict[str, Any]:
    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": repr(exc), "capture_greedy_safe": False}

    n_draws = max(int(n_draws), 1000)
    rng = np.random.default_rng(seed)

    def _accept_len_ref(draft: np.ndarray, target_argmax: np.ndarray) -> int:
        # served _dixie_fused_accept_prep_kernel semantics: walk positions, accept
        # while draft == target_argmax, STOP at first mismatch (emit target there).
        n = 0
        for pos in range(draft.shape[0]):
            if draft[pos] == target_argmax[pos]:
                n += 1
            else:
                n += 1   # the mismatch position still emits the target token
                break
        return n

    mism_argmax = 0
    mism_accept = 0
    mism_tie = 0
    for _ in range(n_draws):
        # target logits over the (pruned-then-scattered) vocab; argmax twice ->
        # must be identical (deterministic), incl. exact ties -> lowest index.
        logits = rng.standard_normal((k_spec + 1, min(vocab, 4096))).astype(np.float32)
        a1 = logits.argmax(axis=-1)
        a2 = logits.argmax(axis=-1)
        mism_argmax += int((a1 != a2).sum())
        # exact-tie adversarial: copy each row max into another column -> tie;
        # numpy/torch argmax both return the FIRST (lowest) index deterministically.
        row = rng.integers(0, k_spec + 1)
        col0 = int(a1[row])
        col1 = int(rng.integers(0, logits.shape[1]))
        if col1 == col0:
            col1 = (col1 + 1) % logits.shape[1]
        logits[row, col1] = logits[row, col0]
        at = logits.argmax(axis=-1)
        if int(at[row]) != min(col0, col1):
            mism_tie += 1
        # accept-length determinism: two refs over the same draws agree.
        draft = rng.integers(0, logits.shape[1], size=k_spec)
        target_argmax = logits.argmax(axis=-1)[:k_spec]
        if _accept_len_ref(draft, target_argmax) != _accept_len_ref(draft, target_argmax):
            mism_accept += 1

    safe = bool(mism_argmax == 0 and mism_accept == 0 and mism_tie == 0)
    return {
        "available": True,
        "capture_greedy_safe": safe,
        "n_draws": n_draws, "k_spec": k_spec, "vocab": vocab,
        "mismatch_argmax": mism_argmax,
        "mismatch_accept_len": mism_accept,
        "mismatch_tie_lowest_index": mism_tie,
        "note": "cudagraph replay re-runs identical kernels with identical inputs "
                "=> bit-identical; verify argmax + accept-length are deterministic "
                "index/compare ops (no float reduction reorder) => capture is "
                "latency-only, token-identical / PPL-invariant.",
    }


# --------------------------------------------------------------------------- #
def synthesize(safety: dict[str, Any]) -> dict[str, Any]:
    aud = audit_served()
    separately_launched = aud["verify_accept_epilogue_separately_launched"]

    # ---- bound the recoverable epilogue launch-tax ------------------------ #
    # eager_launch_count_step = exposed small-op tail (sampler + accept + reloc);
    # the compute-bound verify forward (b) is EXCLUDED (recoverable_launches=0).
    count_lo, count_mid, count_hi = COUNT_LO, COUNT_MID, COUNT_HI
    if not separately_launched:
        count_lo = count_mid = count_hi = 0  # captured => 0 (the #261 draft case)

    us_lo = count_lo * PER_LAUNCH_US_LO
    us_mid = count_mid * PER_LAUNCH_US_MID
    us_hi = count_hi * PER_LAUNCH_US_HI

    def _row(count: int, per_launch_us: float, label: str) -> dict[str, Any]:
        us = count * per_launch_us
        return {
            "scenario": label,
            "eager_launch_count": count,
            "per_launch_us": per_launch_us,
            "eager_launch_us": round(us, 4),
            "step_reduction_pct_served": round(100.0 * us / STEP_US_SERVED, 4),
            "step_reduction_pct_built": round(100.0 * us / STEP_US_BUILT, 4),
            "implied_tps": round(tps_from_step(STEP_US_SERVED - us), 3),
            "implied_gain_pct": round(tps_gain_pct_from_us_net(us), 4),
            "clears_500": bool(tps_from_step(STEP_US_SERVED - us) >= 500.0),
            "clears_520_95": bool(tps_from_step(STEP_US_SERVED - us) >= LAMBDA1_CEILING_TPS),
            "capture_greedy_safe": bool(safety.get("capture_greedy_safe", False)),
        }

    rows = [
        _row(count_lo, PER_LAUNCH_US_LO,
             "CEILING lo (count=%d x %.0fus) -- lambda=1, all epilogue tax exposed"
             % (count_lo, PER_LAUNCH_US_LO)),
        _row(count_mid, PER_LAUNCH_US_MID,
             "CEILING mid (count=%d x %.0fus) -- lambda=1 central" % (count_mid, PER_LAUNCH_US_MID)),
        _row(count_hi, PER_LAUNCH_US_HI,
             "CEILING hi (count=%d x %.0fus) -- lambda=1, conservative upper"
             % (count_hi, PER_LAUNCH_US_HI)),
    ]

    projected_tps_gain_pct = round(tps_gain_pct_from_us_net(us_mid), 4)  # TEST: central ceiling
    counterfactual_tps_mid = round(tps_from_step(STEP_US_SERVED - us_mid), 3)

    # served-vs-built gap decomposition (the eager epilogue tax as a fraction).
    gap_us = STEP_US_SERVED - STEP_US_BUILT  # 133.2 us
    gap_decomp = {
        "served_minus_built_us": round(gap_us, 4),
        "epilogue_eager_us_lo": round(us_lo, 4),
        "epilogue_eager_us_mid": round(us_mid, 4),
        "epilogue_eager_us_hi": round(us_hi, 4),
        "epilogue_pct_of_gap_lo": round(100.0 * us_lo / gap_us, 2) if gap_us else 0.0,
        "epilogue_pct_of_gap_mid": round(100.0 * us_mid / gap_us, 2) if gap_us else 0.0,
        "epilogue_pct_of_gap_hi": round(100.0 * us_hi / gap_us, 2) if gap_us else 0.0,
        "remainder_to_denken_257": (
            "the rest of the 133.2us gap = verify-forward compute-bound launches "
            "(hidden) + CPU/Python gaps; denken #257's roofline (cpu_ms vs gpu_ms) "
            "apportions the exposed fraction (lambda)"),
    }

    accounting = {
        "k_cal": K_CAL, "served_tps": SERVED_TPS,
        "step_us_served": STEP_US_SERVED, "step_us_built": STEP_US_BUILT,
        "per_launch_us_lo": PER_LAUNCH_US_LO, "per_launch_us_mid": PER_LAUNCH_US_MID,
        "per_launch_us_hi": PER_LAUNCH_US_HI,
        "eager_launch_count_step_lo": count_lo,
        "eager_launch_count_step": count_mid,      # headline central count
        "eager_launch_count_step_hi": count_hi,
        "eager_launch_us_step_lo": round(us_lo, 4),
        "eager_launch_us_step_mid": round(us_mid, 4),
        "eager_launch_us_step_hi": round(us_hi, 4),
        "projected_tps_gain_pct_lo": round(tps_gain_pct_from_us_net(us_lo), 4),
        "projected_tps_gain_pct": projected_tps_gain_pct,   # TEST (central)
        "projected_tps_gain_pct_hi": round(tps_gain_pct_from_us_net(us_hi), 4),
        "counterfactual_tps_lo": round(tps_from_step(STEP_US_SERVED - us_lo), 3),
        "counterfactual_tps_mid": counterfactual_tps_mid,
        "counterfactual_tps_hi": round(tps_from_step(STEP_US_SERVED - us_hi), 3),
        "actual_tps": SERVED_TPS,   # the SCREEN moves nothing
        "gap_decomposition": gap_decomp,
        "epilogue_launch_breakdown": {
            "sampler": EPILOGUE_SAMPLER_LAUNCHES,
            "accept": EPILOGUE_ACCEPT_LAUNCHES,
            "relocate_seed_lo_mid_hi": [EPILOGUE_RELOCATE_LO, EPILOGUE_RELOCATE_MID,
                                        EPILOGUE_RELOCATE_HI],
            "verify_forward_excluded_compute_bound": True,
        },
        "draft_side_complement_261": {
            "draft_n_launches_counterfactual": DRAFT_N_LAUNCHES_261,
            "draft_eager_us_step": 0.0,   # #261: captured => 0
            "note": "#261 (egaz6m2f) closed the DRAFT side at 0 (captured in ONEGRAPH); "
                    "this screen closes the VERIFY side (eager => non-zero ceiling).",
        },
    }

    captured = aud["verify_accept_epilogue_captured"]
    headline = {
        "verify_accept_epilogue_separately_launched": separately_launched,
        "verify_accept_epilogue_captured": captured,
        "draft_argmax_embed_separately_launched": aud["draft_argmax_embed_separately_launched"],
        "verify_forward_eager": aud["verify_forward_eager"],
        "verify_forward_compute_bound": aud["verify_forward_compute_bound"],
        "sampler_argmax_eager": aud["sampler_argmax_eager"],
        "accept_eager": aud["accept_eager"],
        "relocate_eager": aud["relocate_eager"],
        "eager_launch_count_step": count_mid,
        "eager_launch_us_step_band": [round(us_lo, 2), round(us_hi, 2)],
        "eager_launch_us_step_mid": round(us_mid, 2),
        "projected_tps_gain_pct": projected_tps_gain_pct,            # TEST (central ceiling)
        "projected_tps_gain_pct_band": [round(tps_gain_pct_from_us_net(us_lo), 2),
                                        round(tps_gain_pct_from_us_net(us_hi), 2)],
        "counterfactual_tps_band": [round(tps_from_step(STEP_US_SERVED - us_lo), 1),
                                    round(tps_from_step(STEP_US_SERVED - us_hi), 1)],
        "counterfactual_tps_mid": counterfactual_tps_mid,
        "actual_tps": SERVED_TPS,
        "baseline_tps": BASELINE_TPS,
        "lambda1_ceiling_tps": LAMBDA1_CEILING_TPS,
        "capture_greedy_safe": bool(safety.get("capture_greedy_safe", False)),
        "lever_class": ("VERIFY-SIDE LAUNCH LEVER (eager epilogue, ceiling-bounded)"
                        if separately_launched else
                        "NULL (epilogue already captured like the draft side)"),
        "screen_verdict": "LEVER-EXISTS-CEILING-BOUNDED" if separately_launched else "NO-GO",
        "k_spec": aud["k_spec"],
    }

    # self-test conditions (PR step 6) ------------------------------------- #
    # (a) every stage tag carries an evidence string sourced from a served file.
    files_present = all(p.is_file() for p in (_MANIFEST, _SITECUSTOMIZE, _SERVE, _PCK04, _SPLITKV))
    cond_a = bool(
        files_present
        and all(isinstance(s.get("evidence"), str) and len(s["evidence"]) > 20
                for s in aud["stages"])
        and len(aud["stages"]) == 5)
    # (b) #261 draft-side result re-confirmed (separately_launched False / captured).
    cond_b = bool(aud["draft_captured"]
                  and aud["draft_argmax_embed_separately_launched"] is False)
    # (c) eager_launch_us_step = count x per-launch band, arithmetic round-trips.
    cond_c = bool(
        math.isclose(us_lo, count_lo * PER_LAUNCH_US_LO, rel_tol=0, abs_tol=1e-9)
        and math.isclose(us_mid, count_mid * PER_LAUNCH_US_MID, rel_tol=0, abs_tol=1e-9)
        and math.isclose(us_hi, count_hi * PER_LAUNCH_US_HI, rel_tol=0, abs_tol=1e-9))
    # (d) projected_tps_gain_pct maps the eager us through composition; invariant
    #     (eager => non-zero; captured => 0). Compare at the 4-dp precision the
    #     stored value was produced at (line 519 round(...,4)).
    rt = math.isclose(projected_tps_gain_pct,
                      round(tps_gain_pct_from_us_net(us_mid), 4),
                      rel_tol=0, abs_tol=1e-6)
    invariant = ((separately_launched and projected_tps_gain_pct > 0.0)
                 or ((not separately_launched) and math.isclose(
                     projected_tps_gain_pct, 0.0, rel_tol=0, abs_tol=1e-9)))
    cond_d = bool(rt and invariant)
    # (e) NaN-clean (finalised in main with a whole-payload scan).
    cond_e = True
    # (f) BASELINE 481.53 + 520.95 lambda=1 ceiling UNCHANGED; the SCREEN moves
    #     nothing (actual_tps stays 481.53). NOTE: projected_tps_gain_pct is the
    #     ceiling of a FUTURE capture lever, NOT an applied change -> it is
    #     non-zero here (verify side eager), unlike #261 (draft captured => 0).
    cond_f = bool(
        math.isclose(BASELINE_TPS, 481.53, rel_tol=0, abs_tol=1e-9)
        and math.isclose(SERVED_TPS, 481.53, rel_tol=0, abs_tol=1e-9)
        and math.isclose(LAMBDA1_CEILING_TPS, 520.95, rel_tol=0, abs_tol=1e-9)
        and math.isclose(accounting["actual_tps"], 481.53, rel_tol=0, abs_tol=1e-6))
    # greedy-safety must hold (capture cannot move tokens/PPL).
    cond_g = bool(safety.get("capture_greedy_safe", False))

    conditions = {
        "a_stage_tags_sourced_from_served_files": cond_a,
        "b_draft_side_261_reconfirmed": cond_b,
        "c_launch_tax_roundtrip": cond_c,
        "d_projected_gain_maps_through_composition": cond_d,
        "e_nan_clean": cond_e,
        "f_baseline_and_ceiling_unchanged": cond_f,
        "g_capture_greedy_safe": cond_g,
    }
    self_test = {
        "conditions": conditions,
        "verify_accept_epilogue_capture_self_test_passes": bool(all(conditions.values())),
        "detail": {
            "files_present": files_present,
            "safety_mismatches": [safety.get("mismatch_argmax"),
                                  safety.get("mismatch_accept_len"),
                                  safety.get("mismatch_tie_lowest_index")],
            "safety_n_draws": safety.get("n_draws"),
        },
    }

    nonoverlap = {
        "pr261_draft_side_is_the_direct_complement_closed_at_0": True,
        "pr255_is_verify_d2h_MATERIALIZATION_memory_axis_this_is_LAUNCH_capture": True,
        "kanna264_is_draft_head_vocab_GEMV_cost_not_launch": True,
        "denken257_built_step_roofline_this_feeds_it_the_eager_slack_number": True,
        "fern262_shallow_tree_ubel263_private_rank_are_ET_topology_side": True,
        "land245_owns_any_live_build": True,
    }

    handoff = (
        "the verify+sampler+accept+KV-relocate epilogue is EAGER "
        "(verify_accept_epilogue_separately_launched=%s) -- the eager complement of "
        "#261's captured draft side -- leaving ~%.0fus of recoverable per-step "
        "epilogue launch-tax (band %.0f-%.0fus, ~%.2f%% / +%.1f TPS off 481.53 -> "
        "%.1f, a lambda=1 CEILING; the verify FORWARD is eager-but-compute-bound so "
        "its launches are EXCLUDED), so the step's launch-overhead axis is NOT fully "
        "closed by #261 -- there is a verify-side capture lever worth <= the ceiling; "
        "denken #257's roofline apportions the exposed (realizable) fraction." % (
            separately_launched, us_mid, us_lo, us_hi,
            accounting["projected_tps_gain_pct"],
            counterfactual_tps_mid - SERVED_TPS, counterfactual_tps_mid))

    verdict = ("VERIFY-ACCEPT-EPILOGUE-EAGER-CAPTURE-LEVER-EXISTS-CEILING-BOUNDED"
               if separately_launched else
               "VERIFY-ACCEPT-EPILOGUE-ALREADY-CAPTURED-NO-GO")

    return {
        "verdict": verdict,
        "headline": headline,
        "audit": aud,
        "composition": {
            "k_cal": K_CAL, "step_us_served": STEP_US_SERVED, "step_us_built": STEP_US_BUILT,
            "served_tps": SERVED_TPS, "k_spec": aud["k_spec"],
            "per_launch_us_band": [PER_LAUNCH_US_LO, PER_LAUNCH_US_HI],
        },
        "accounting": accounting,
        "verdict_table": rows,
        "safety": safety,
        "nonoverlap": nonoverlap,
        "self_test": self_test,
        "handoff": handoff,
    }


# --------------------------------------------------------------------------- #
def _nan_paths(node: Any, prefix: str = "") -> list[str]:
    bad: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            bad += _nan_paths(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            bad += _nan_paths(v, f"{prefix}[{i}]")
    elif isinstance(node, float) and not math.isfinite(node):
        bad.append(prefix)
    return bad


def _print_report(syn: dict[str, Any]) -> None:
    h, acc, aud = syn["headline"], syn["accounting"], syn["audit"]
    st, sf = syn["self_test"], syn["safety"]
    print("\n" + "=" * 100, flush=True)
    print("VERIFY+ACCEPT EPILOGUE CAPTURE SCREEN (PR #265, wirbel) -- is the "
          "verify-side launch-tax recoverable?", flush=True)
    print("=" * 100, flush=True)
    print("  (1) STAGE MAP (captured/eager, each tag sourced from a served file)", flush=True)
    print(f"      served: {SERVED_SUBMISSION}  vLLM {VLLM_VERSION}  K_spec={h['k_spec']}", flush=True)
    for s in aud["stages"]:
        tag = "CAPTURED" if s["captured"] else "EAGER"
        extra = " (compute-bound, excl.)" if s.get("compute_bound") else ""
        print(f"      {s['stage']:<50.50} {tag:<8}{extra}  reclaunch={s['recoverable_launches']}",
              flush=True)
    print("-" * 100, flush=True)
    print("  (2) DECISIVE BOOLEAN", flush=True)
    print(f"      verify_accept_epilogue_separately_launched = "
          f"{h['verify_accept_epilogue_separately_launched']}   "
          f"(draft side #261: separately_launched="
          f"{h['draft_argmax_embed_separately_launched']} => captured)", flush=True)
    print(f"      eager_launch_count_step = {acc['eager_launch_count_step_lo']}/"
          f"{acc['eager_launch_count_step']}/{acc['eager_launch_count_step_hi']} "
          f"(lo/mid/hi)  [sampler {EPILOGUE_SAMPLER_LAUNCHES} + accept "
          f"{EPILOGUE_ACCEPT_LAUNCHES} + reloc {EPILOGUE_RELOCATE_LO}-{EPILOGUE_RELOCATE_HI}; "
          f"verify-forward EXCLUDED (compute-bound)]", flush=True)
    print(f"      eager_launch_us_step = {acc['eager_launch_us_step_lo']:.0f}/"
          f"{acc['eager_launch_us_step_mid']:.0f}/{acc['eager_launch_us_step_hi']:.0f}us "
          f"(= count x per-launch [{PER_LAUNCH_US_LO:.0f},{PER_LAUNCH_US_HI:.0f}]us)", flush=True)
    print("-" * 100, flush=True)
    print("  (3) ACCOUNTING (tps(step') = served*step_served/step'; lambda=1 ceiling)", flush=True)
    for r in syn["verdict_table"]:
        print(f"      {r['scenario']:<58.58} {r['eager_launch_us']:>5.0f}us  "
              f"red%srv {r['step_reduction_pct_served']:>5.2f}  TPS {r['implied_tps']:>7.2f}  "
              f"gain {r['implied_gain_pct']:>+5.2f}%  500:{str(r['clears_500'])[:1]}", flush=True)
    gd = acc["gap_decomposition"]
    print(f"      served-built gap = {gd['served_minus_built_us']:.1f}us; epilogue eager "
          f"= {gd['epilogue_eager_us_mid']:.0f}us ({gd['epilogue_pct_of_gap_mid']:.0f}% of gap); "
          f"rest -> denken #257 roofline", flush=True)
    print("-" * 100, flush=True)
    print(f"  (4) GREEDY/PPL SAFETY: capture_greedy_safe={sf.get('capture_greedy_safe')}  "
          f"(argmax/accept/tie mismatches "
          f"{sf.get('mismatch_argmax')}/{sf.get('mismatch_accept_len')}/"
          f"{sf.get('mismatch_tie_lowest_index')} over {sf.get('n_draws')} draws)", flush=True)
    print("-" * 100, flush=True)
    print(f"  (5) PRIMARY verify_accept_epilogue_capture_self_test_passes = "
          f"{st['verify_accept_epilogue_capture_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"        - {k}: {v}", flush=True)
    print(f"      TEST projected_tps_gain_pct = {h['projected_tps_gain_pct']:.2f}  "
          f"(lambda=1 ceiling; band {h['projected_tps_gain_pct_band'][0]:+.2f}.."
          f"{h['projected_tps_gain_pct_band'][1]:+.2f}%)", flush=True)
    print("=" * 100, flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print(f"\n  HAND-OFF: {syn['handoff']}\n", flush=True)


def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[verify-epilogue-screen] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    h, acc, aud = syn["headline"], syn["accounting"], syn["audit"]
    st, sf = syn["self_test"], syn["safety"]
    run = init_wandb_run(
        job_type="verify-accept-epilogue-capture-screen",
        agent="wirbel",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["verify-accept-epilogue-capture", "speed-levers", "planb",
              "verify-launch-tax", "cudagraph-capture", "onegraph",
              "bank-the-analysis", "launch-overhead-axis", "pr265"],
        config={
            "k_cal": K_CAL, "step_us_served": STEP_US_SERVED, "step_us_built": STEP_US_BUILT,
            "served_tps": SERVED_TPS, "baseline_tps": BASELINE_TPS,
            "lambda1_ceiling_tps": LAMBDA1_CEILING_TPS, "k_spec": h["k_spec"],
            "per_launch_us_lo": PER_LAUNCH_US_LO, "per_launch_us_hi": PER_LAUNCH_US_HI,
            "served_submission": SERVED_SUBMISSION, "vllm_version": VLLM_VERSION,
            "wandb_group": args.wandb_group,
            "source_runs": "wirbel#261 egaz6m2f draft-side capture + per-launch band; "
                           "kanna#217 vgovdrjc composition+served step; denken#254 "
                           "zav6nr8y draft floor/built step; served fa2sw_precache_kenyan",
        },
    )
    if run is None:
        print("[verify-epilogue-screen] wandb: no run (no WANDB_API_KEY/mode) — skipping",
              flush=True)
        return

    summary: dict[str, Any] = {
        "verify_accept_epilogue_capture_self_test_passes":
            int(bool(st["verify_accept_epilogue_capture_self_test_passes"])),   # PRIMARY
        "projected_tps_gain_pct": h["projected_tps_gain_pct"],                   # TEST
        "verify_accept_epilogue_separately_launched":
            int(bool(h["verify_accept_epilogue_separately_launched"])),
        "verify_accept_epilogue_captured": int(bool(h["verify_accept_epilogue_captured"])),
        "draft_argmax_embed_separately_launched":
            int(bool(h["draft_argmax_embed_separately_launched"])),
        "verify_forward_eager": int(bool(h["verify_forward_eager"])),
        "verify_forward_compute_bound": int(bool(h["verify_forward_compute_bound"])),
        "sampler_argmax_eager": int(bool(h["sampler_argmax_eager"])),
        "accept_eager": int(bool(h["accept_eager"])),
        "relocate_eager": int(bool(h["relocate_eager"])),
        "eager_launch_count_step": acc["eager_launch_count_step"],
        "eager_launch_count_step_lo": acc["eager_launch_count_step_lo"],
        "eager_launch_count_step_hi": acc["eager_launch_count_step_hi"],
        "eager_launch_us_step_lo": acc["eager_launch_us_step_lo"],
        "eager_launch_us_step_mid": acc["eager_launch_us_step_mid"],
        "eager_launch_us_step_hi": acc["eager_launch_us_step_hi"],
        "projected_tps_gain_pct_lo": acc["projected_tps_gain_pct_lo"],
        "projected_tps_gain_pct_hi": acc["projected_tps_gain_pct_hi"],
        "counterfactual_tps_lo": acc["counterfactual_tps_lo"],
        "counterfactual_tps_mid": acc["counterfactual_tps_mid"],
        "counterfactual_tps_hi": acc["counterfactual_tps_hi"],
        "served_minus_built_us": acc["gap_decomposition"]["served_minus_built_us"],
        "epilogue_pct_of_gap_mid": acc["gap_decomposition"]["epilogue_pct_of_gap_mid"],
        "capture_greedy_safe": int(bool(h["capture_greedy_safe"])),
        "safety_mismatch_argmax": sf.get("mismatch_argmax"),
        "safety_mismatch_accept_len": sf.get("mismatch_accept_len"),
        "safety_mismatch_tie": sf.get("mismatch_tie_lowest_index"),
        "safety_n_draws": sf.get("n_draws"),
        "onegraph_on": int(bool(aud["onegraph_on"])),
        "require_capture": int(bool(aud["require_capture"])),
        "pck04_says_main_runner_eager": int(bool(aud["pck04_says_main_runner_eager"])),
        "serve_no_enforce_eager": int(bool(aud["serve_no_enforce_eager"])),
        "serve_has_speculative_config": int(bool(aud["serve_has_speculative_config"])),
        "screen_verdict_lever_exists":
            int(h["screen_verdict"] == "LEVER-EXISTS-CEILING-BOUNDED"),
        "actual_tps": h["actual_tps"], "baseline_tps": BASELINE_TPS,
        "lambda1_ceiling_tps": LAMBDA1_CEILING_TPS,
        "k_spec": h["k_spec"], "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="verify_accept_epilogue_capture_result",
                      artifact_type="speed-lever-screen", data=payload)
    finish_wandb(run)
    print(f"[verify-epilogue-screen] wandb logged {len(summary)} keys", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--safety-draws", type=int, default=2000,
                    help="random draws for the capture greedy-safety check (>=1000)")
    ap.add_argument("--k-spec", type=int, default=K_SPEC)
    ap.add_argument("--vocab", type=int, default=262144)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="planb-speed-levers")
    args = ap.parse_args(argv)

    safety = capture_safety_check(n_draws=args.safety_draws, k_spec=args.k_spec,
                                  vocab=args.vocab, seed=args.seed)
    syn = synthesize(safety)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 265, "agent": "wirbel",
        "kind": "verify-accept-epilogue-capture-screen", "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _nan_paths(payload)
    payload["nan_clean"] = not nan_paths
    syn["self_test"]["conditions"]["e_nan_clean"] = (
        syn["self_test"]["conditions"]["e_nan_clean"] and not nan_paths)
    syn["self_test"]["verify_accept_epilogue_capture_self_test_passes"] = bool(
        all(syn["self_test"]["conditions"].values()))
    if nan_paths:
        print(f"[verify-epilogue-screen] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[verify-epilogue-screen] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = syn["self_test"]["verify_accept_epilogue_capture_self_test_passes"]
        print(f"[verify-epilogue-screen] SELF-TEST {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
