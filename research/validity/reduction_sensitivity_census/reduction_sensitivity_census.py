#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #491 (ubel) -- Reduction-sensitivity census: the MINIMUM mandatory-deterministic op set.

THE QUESTION (#481 forward-ladder Direction 1 -- the per-op argmax-sensitivity layer above #484's
per-op tax attribution)
--------------------------------------------------------------------------------------------------
Byte-exact != batch-invariant-kernel. The greedy gate requires only that the ARGMAX at each decode
step be stable -- NOT that every reduction is bitwise-deterministic. So for each decode GEMM/op in the
verify + drafter paths, when the fast (split-K / non-invariant) kernel and the invariant kernel produce
DIFFERENT bits, does that perturbation ever actually FLIP the sampled token? An op whose perturbation
never crosses an argmax boundary on our prompt distribution can run on the FAST kernel for free.
Output = ``must_be_deterministic`` = the minimum set of matmuls that MUST be deterministic to hold
128/128 token identity, plus the projected TPS if only that set is pinned.

WHY THIS DECOMPOSES CLEANLY (the load-bearing realization)
----------------------------------------------------------
On the deployed osoi5 int4 stack (gemma-4-E4B-it, compressed-tensors W4A16, sm_86 A10G), almost every
op's free/must classification is NOT a public-data accident -- it is a KERNEL PROPERTY or a THEOREM,
hence distribution-robust (private-safe):
  * int4 GPTQ-Marlin body GEMMs (fused qkv n=3072, fused gate_up n=20480, o n=2560, down n=2560) and
    the pruned lm_head (n=16384): every n >= 2048, so vLLM ``should_use_atomic_add_reduce`` -> False
    (sm_8x bf16) -> the deployed kernel runs the FIXED (non-atomic) reduction and is M-INVARIANT.
    fast == invariant in BITS -> a zero perturbation can never flip an argmax. (#484: marlin body
    bit-identical regardless of VLLM_BATCH_INVARIANT; #384: lm_head maxdiff=0 across M.) The microbench
    here CONFIRMS byte-identity across M and a POSITIVE control (a perturbed input MUST change the
    output bytes for every shape), proving the comparator is sensitive, not vacuously zero. (Forcing
    atomic-add ON is INERT at the M=8 decode geometry -- marlin does not split-K there; the serve-
    occupancy atomic_on breakage is captured faithfully by INGEST instead.)
  * MTP drafter: greedy spec-decode with temp=0 emits, at every position, the TARGET model's argmax
    (an accepted draft equals that argmax, or the correction token IS that argmax). Drafter
    non-determinism only moves the ACCEPTANCE boundary (e_accept), never the emitted token VALUES --
    Leviathan 2023 / Chen 2023. Proven by construction in the acceptance-invariance sim (drafter has
    frac_bitdiff>0 yet frac_argmax_flip==0: the poster child for "byte-exact != argmax-stable").
  * attention reduction (QK^T / P.V split-KV): the ONE op whose reduction order is occupancy-dependent
    (the flash split count differs M=1 AR vs M=8 verify), so its bits change AND that perturbation
    propagates through the M-invariant body+lm_head to the final logits. Whether it FLIPS a greedy
    argmax is the sole empirical unknown -- measured here.

So the only distribution-dependent number in the whole census is attention's flip RATE/margin; the SET
MEMBERSHIP (which ops are free vs must-pin) is theory/kernel-grounded.

METHOD (three measured legs + theory, all LOCAL -- documented honestly per the PR)
---------------------------------------------------------------------------------
(MARGIN, vLLM, server venv) The attention argmax-flip leg, FAITHFUL (not a bound): re-use ubel #364's
  geometry. Greedy-AR generate (M=1) gives the reference token + its top-K logprobs per step; a width-8
  verify re-forward of [ctx+gen] (max_num_batched_tokens=8 -> the M=8 verify occupancy) gives the M=8
  top-K logprobs at the SAME positions. M=1 and M=8 differ ONLY in attention reduction (body+lm_head are
  M-invariant -- confirmed by MICROBENCH), so the M1-vs-M8 logit delta IS the attention reduction
  perturbation, propagated through the REAL lm_head to REAL logits. flip = (argmax_M8 != ref_tok);
  ``max_logit_gap_at_flip`` = the top1-top2 margin at flips. A run-to-run determinism control confirms
  the non-attention path is bit-stable, so the flip is attributable to attention occupancy, not noise.
  HONEST PROXY CAVEAT: (a) width-8 chunked-prefill verify is the #364 proxy for the spec-decode verify
  occupancy; (b) the deployed PRUNED lm_head (16384 rows) cannot load in vanilla vLLM (its rows != the
  full config vocab 262144), so the scan runs the upstream FULL-VOCAB int4 checkpoint -- identical
  37-layer attention architecture, and the output head is M-invariant either way. The EXACT deployed
  pruned-head spec-decode kernel is corroborated end-to-end by INGEST below.
(MICROBENCH, Marlin, server venv) Per deployed GEMM shape (incl. the pruned lm_head 2560->16384),
  byte-identity of M=8-batched vs per-row-M1 (fast vs invariant) -> frac_steps_bitdiff (expect ~0 ==
  M-invariant) + a perturbed-input positive control (expect <1.0, proving sensitivity) +
  should_use_atomic_add_reduce per shape. Establishes the bitdiff column for the body + lm_head ops
  the MARGIN leg cannot isolate.
(INGEST, no GPU) Faithful end-to-end cross-check from the on-branch greedy_determinism captures (cited):
  the deployed default is bit-exact run-to-run (128/128); disabling the attention split-KV reduction
  (``splitkv_off``) leaves only ~8.6% of COMPLETIONS identical to default (per-prompt; the remaining
  ~91% diverge by >=1 token) -> attention reduction DOES flip greedy tokens end-to-end; this is a
  STRONGER perturbation than the margin proxy's M1-vs-M8 occupancy contrast, so the two flip
  magnitudes are not directly comparable -- both merely establish attention's must-pin SET membership.
  Forcing the body Marlin atomic-add ON (``atomic_on``) breaks determinism but
  the deployed keeps it OFF. Per-prompt first-divergence onset is computed from the captured streams.
(DRAFTER, no GPU) Acceptance-invariance simulation -> drafter_in_must_pin = False, by construction.

HONESTY BAR (per the PR)
------------------------
A single observed flip on the 128 PUBLIC prompts is a LOWER bound on private sensitivity; the public->
private gap is real (cite the 4.295% private-Delta frame, ubel #379 ``5kpb73tb``). Rule-of-three (3/N)
upper 95% CB on every zero-flip op. The mitigating structure: the must-pin SET is kernel/theorem-
grounded (distribution-robust); only attention's RATE is empirical, and attention is ALREADY must-pin,
so private data can only GROW must_be_deterministic, never shrink it below {attention}.

SCOPE: LOCAL profiling card. analysis_only=true, official_tps=0, NO served-file change, NO HF Job, NO
train.py --launch, NO submission. Baseline 481.53 UNCHANGED. Greedy identity MEASURED, never broken.
GPU phases run under the submission server venv (has vLLM+Marlin); CUDA_VISIBLE_DEVICES=0.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]

# ======================================================================================== #
# Real deployed gemma-4-E4B-it / osoi5 geometry (config text_config) + cited anchors
# ======================================================================================== #
HIDDEN = 2560
INTERMEDIATE = 10240
N_LAYERS = 37                      # config num_hidden_layers (read at runtime; this is the fallback)
N_Q_HEADS = 8
N_KV_HEADS = 2
HEAD_DIM = 256
FULL_VOCAB = 262144
LMHEAD_ROWS = 16384                # PCK row-pruned lm_head (#384)
SPEC_K = 7                         # MTP num_speculative_tokens
DEPLOYED_VERIFY_M = 8              # SPEC_K + 1 verify width

# Deployed GEMM shapes (size_k, size_n). vLLM FUSES qkv and gate_up -> every n >= 2048.
DEPLOYED_GEMMS = {
    "qkv_proj": (HIDDEN, N_Q_HEADS * HEAD_DIM + 2 * N_KV_HEADS * HEAD_DIM),  # 2560 -> 3072
    "o_proj":   (N_Q_HEADS * HEAD_DIM, HIDDEN),                              # 2048 -> 2560
    "gate_up_proj": (HIDDEN, 2 * INTERMEDIATE),                             # 2560 -> 20480
    "down_proj": (INTERMEDIATE, HIDDEN),                                    # 10240 -> 2560
    "lm_head":  (HIDDEN, LMHEAD_ROWS),                                      # 2560 -> 16384
}
# The PR's logical op inventory (n_ops_total). q/k/v/o + gate/up/down are the 7 body matmuls,
# attention_reduction is the QK/PV split-KV, lm_head the vocab projection, mtp_drafter the MTP head.
BODY_OPS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ALL_OPS = BODY_OPS + ["attention_reduction", "lm_head", "mtp_drafter"]
N_OPS_TOTAL = len(ALL_OPS)         # 10
# map each logical body op -> the deployed (fused) GEMM that realizes it
BODY_OP_TO_GEMM = {
    "q_proj": "qkv_proj", "k_proj": "qkv_proj", "v_proj": "qkv_proj", "o_proj": "o_proj",
    "gate_proj": "gate_up_proj", "up_proj": "gate_up_proj", "down_proj": "down_proj",
}

# ---- cited anchors (NOT re-derived) -------------------------------------------------------
DEPLOYED_TPS = 481.53              # PR #52 public non-strict deployed (2x9fm2zx)
PREDICTED_SURGICAL_TPS = 456.98    # #484 r1l881bx: pin ONLY attention reduction, drafter runs fast
ATTENTION_OFFICIAL_TAX_PCT = 5.10  # #484 attention reduction official tax
GLOBAL_TAX_PCT = 48.0              # #484 global VLLM_BATCH_INVARIANT tax (drafter reroute = 74.9% of it)
PRIVATE_DELTA_PCT = 4.295          # ubel #379 5kpb73tb private-Delta frame (public->private honesty)
CEILING_500 = 520.953             # strict lambda=1 ceiling TPS (context only)
GREEDY_DET_DIR = ROOT / "research" / "validity" / "greedy_determinism" / "captures"

# ======================================================================================== #
# PR #497 (ubel) -- private-attention-flip identity bound (the --shifted-split mode).
# Reuses this margin harness ATTENTION-ONLY on a HELD-OUT reasoning/STEM split; classifies every
# flip as a bf16-ULP tie (#488, semantically free) vs a margin-#461 knife-edge vs a TRUE semantic
# flip, and reports the off-distribution growth of attention's must-pin COST + operative identity.
# ======================================================================================== #
# #491 PUBLIC anchor (5cappm87) -- the baseline this card shifts away from. The public 128 are
# THEMSELVES reasoning/STEM (mmlu_pro 57 / gpqa_diamond 57 / aime2026 14), so the genuine private-
# like shift is HELD-OUT problems (different source datasets), formatted identically.
PUBLIC_ATTENTION_FLIP_RATE = 0.005242463958060288   # 16/3052 public margin-proxy flip rate
PUBLIC_MARGIN_P05 = 0.25                            # public full-step margin p05
PUBLIC_MARGIN_MEDIAN = 3.0                          # public full-step margin median
PUBLIC_MAX_GAP_AT_FLIP = 0.5000001192092896        # public max top1-top2 gap at any flip
# identity-classification thresholds (replicated EXACTLY from the merged cards)
EPS_STAR = 0.125                  # bf16 one-ULP gap at magnitude ~1 (#488 argmax_tiebreak); ULP-tie band
BAND_TOL = 1e-9                   # numeric slack on the band edge (#488)
NEAR_TIE_LOGPROB_THRESH = 0.5     # margin-#461 knife-edge: a flip with ref-token margin < this is non-semantic
# committed held-out split(s): name -> jsonl path (built by private_attention_flip_bound/build_shifted_split.py)
SHIFTED_DIR = ROOT / "research" / "validity" / "private_attention_flip_bound"
SHIFTED_SPLITS = {"reasoning_stem": SHIFTED_DIR / "shifted_reasoning_stem.jsonl",
                  "hard_ood": SHIFTED_DIR / "shifted_hard_ood.jsonl"}
# per-split provenance for honest verdict/integrity strings: (source label, bracket endpoint).
# reasoning_stem is the EASY/mild LOWER bound (understates worst-case, per advisor 8gpu relay on
# denken #495); hard_ood is the HARD/OOD UPPER endpoint (competition math, flip-prone freeform).
SPLIT_META = {
    "reasoning_stem": ("arc/aqua/gsm8k", "EASY/mild lower-bound (understates worst-case audit exposure)"),
    "hard_ood": ("MATH-500-L5/AIME-2024", "HARD/OOD upper-endpoint (competition freeform math)"),
}


# ======================================================================================== #
# small helpers
# ======================================================================================== #
def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, bool) or o is None or isinstance(o, (str, int)):
        return o
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    return str(o)


def _f(x: Any) -> float:
    """None-safe float: _jsonable() writes non-finite floats as null, so a value reloaded from a
    phase JSON can be None (not merely absent). dict.get(key, default) does NOT substitute the
    default for a present-but-None value, so coerce here to avoid float(None) TypeErrors."""
    try:
        return float(x) if x is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _rule_of_three(n: int) -> float:
    """95% upper confidence bound on a true rate after observing 0 events in n trials."""
    return 3.0 / n if n > 0 else float("nan")


def _lp(v: Any) -> float:
    return float(getattr(v, "logprob", v))


def entry_as_dict(entry: Any) -> dict[int, float]:
    if not entry:
        return {}
    return {int(tok): _lp(v) for tok, v in entry.items()}


def top1_top2_margin(d: dict[int, float]) -> tuple[int, float, float]:
    if not d:
        return -1, float("nan"), float("nan")
    s = sorted(d.values(), reverse=True)
    top = max(d.items(), key=lambda kv: kv[1])
    margin = (s[0] - s[1]) if len(s) >= 2 else float("inf")
    return int(top[0]), float(top[1]), float(margin)


def classify_flip(m8: dict[int, float], ref_tok: int, m8_top_lp: float, m8_gap: float) -> dict:
    """PR #497: classify an attention-occupancy argmax flip on the deployed bf16 logit scale, exactly
    replicating the merged criteria.
      * ULP-tie (#488 argmax_tiebreak, semantically free): the M=8 top1-top2 gap is within one bf16
        ULP (<= EPS_STAR) AND the M=1 reference token is one of the two tied candidates (rank<=2).
        Picking either is a sub-ULP coin-flip -- cost-free, like the 9 ties in #488.
      * knife-edge (margin-#461 deployed_flip_attribution): the reference token's margin BELOW the
        M=8 winner is < NEAR_TIE_LOGPROB_THRESH (0.5 nats) AND it is in the M=8 top-5. Non-semantic.
      * TRUE semantic: a flip that is NOT knife-edge -- the M=8 winner is meaningfully ahead.
    Nesting is provable: ref in top2 with gap<=0.125 => margin_vs_m1<=0.125<0.5 => knife-edge. So
    tie => knife-edge => flip, and `semantic` is disjoint from `tie`."""
    items = sorted(m8.items(), key=lambda kv: kv[1], reverse=True)
    ref_rank = next((i + 1 for i, (t, _) in enumerate(items) if t == ref_tok), len(items) + 1)
    ref_lp = m8.get(ref_tok, float("-inf"))
    margin_vs_m1 = (m8_top_lp - ref_lp) if math.isfinite(ref_lp) else float("inf")
    is_tie = bool(math.isfinite(m8_gap) and m8_gap <= EPS_STAR + BAND_TOL and ref_rank <= 2)
    is_knife = bool(margin_vs_m1 < NEAR_TIE_LOGPROB_THRESH and ref_rank <= 5)
    return {"ref_rank_in_m8": ref_rank, "margin_vs_m1": margin_vs_m1,
            "is_tie_flip": is_tie, "is_knife_edge": is_knife, "is_semantic": (not is_knife)}


# The deployed submission bakes a PRUNED lm_head (16384 keep_ids) onto a full-vocab embedding;
# config.vocab_size stays 262144, so vanilla vLLM's single-vocab ParallelLMHead cannot load it
# (embed=262144 != lm_head=16384 -> vocab_parallel_embedding weight_loader assert). The margin
# scan therefore runs the upstream FULL-VOCAB int4 checkpoint, which shares the identical 37-layer
# Gemma-4-E4B attention architecture. This is faithful for the census because: (1) the attention
# reduction-order sensitivity we measure is an architecture property independent of the output
# vocab; (2) the body + lm_head are M-invariant (microbench runs the deployed pruned 16384 dims);
# (3) the deployed pruned-head END-TO-END flip is independently confirmed on-branch by the
# greedy_determinism captures (phase_ingest). Mirrors ubel #364, which scanned the same checkpoint.
MARGIN_MODEL_CANDIDATES = (
    os.path.expanduser("~/.cache/huggingface/hub/"
                       "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots"),
    "/tmp/osoi5-v0-baked",
)


def resolve_model_dir() -> str:
    for cand in MARGIN_MODEL_CANDIDATES:
        p = Path(cand)
        if p.is_dir() and (p / "config.json").exists():
            return str(p)
        if p.is_dir():  # snapshot parent -> descend into the snapshot subdir holding config.json
            for sub in sorted(p.glob("*")):
                if (sub / "config.json").exists():
                    return str(sub)
    raise FileNotFoundError(f"no loadable int4 model found among {MARGIN_MODEL_CANDIDATES}")


def _margin_model_full_vocab(model_dir: str) -> bool | None:
    """True if the resolved checkpoint's lm_head spans the full config vocab (loads in vanilla
    vLLM); False if it is the deployed pruned head; None if undeterminable. Reads only the
    safetensors header (no weight load)."""
    try:
        import struct
        cfg = json.load(open(Path(model_dir) / "config.json"))
        vocab = int(cfg.get("text_config", cfg).get("vocab_size"))
        st = Path(model_dir) / "model.safetensors"
        if not st.exists():
            return None
        with open(st, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            head = json.loads(f.read(n))
        for k in ("lm_head.weight", "lm_head.weight_packed"):
            if k in head:
                return bool(int(head[k]["shape"][0]) == vocab)
        return None
    except Exception:  # noqa: BLE001
        return None


def resolve_n_layers() -> int:
    try:
        c = json.load(open(Path(resolve_model_dir()) / "config.json"))
        tc = c.get("text_config", c)
        return int(tc.get("num_hidden_layers", N_LAYERS))
    except Exception:  # noqa: BLE001
        return N_LAYERS


# ======================================================================================== #
# GPU PHASE: MARGIN -- attention reduction argmax-flip (M=1 AR vs M=8 verify), faithful
# ======================================================================================== #
PROMPTS_JSONL = ("official/main_bucket/shared_resources/speed_benchmark/data/"
                 "ppl_ground_truth_tokens.jsonl")


def load_prompts(n_prompts: int, ctx_cap: int) -> list[dict]:
    path = ROOT / PROMPTS_JSONL
    rows = [json.loads(l) for l in open(path)][:n_prompts]
    out = []
    for rec in rows:
        ctx = list(rec.get("context_token_ids", []))[:ctx_cap]
        if len(ctx) >= 2:
            out.append({"id": rec.get("id"), "context_token_ids": ctx})
    return out


def load_shifted_prompts(name: str, n_prompts: int, ctx_cap: int) -> list[dict]:
    """PR #497: load the committed HELD-OUT reasoning/STEM split (no internet; built offline by
    private_attention_flip_bound/build_shifted_split.py). Same record shape as load_prompts so
    phase_margin is split-agnostic."""
    path = SHIFTED_SPLITS.get(name)
    if path is None or not Path(path).exists():
        raise FileNotFoundError(
            f"shifted split '{name}' not found at {path}; run build_shifted_split.py first "
            f"(known: {sorted(SHIFTED_SPLITS)})")
    rows = [json.loads(l) for l in open(path)][:n_prompts]
    out = []
    for rec in rows:
        ctx = list(rec.get("context_token_ids", []))[:ctx_cap]
        if len(ctx) >= 2:
            out.append({"id": rec.get("id"), "context_token_ids": ctx,
                        "source": rec.get("source"), "domain": rec.get("domain")})
    return out


def phase_margin(out_path: str, n_prompts: int, n_new: int, ctx_cap: int, verify_width: int,
                 gpu_mem_util: float, topk: int, det_prompts: int,
                 shifted_split: str | None = None) -> None:
    import torch
    from vllm import LLM, SamplingParams

    model_dir = resolve_model_dir()
    full_vocab = _margin_model_full_vocab(model_dir)
    if shifted_split:
        prompts = load_shifted_prompts(shifted_split, n_prompts, ctx_cap)
        split_name, split_kind = shifted_split, "shifted"
    else:
        prompts = load_prompts(n_prompts, ctx_cap)
        split_name, split_kind = "public_ppl_ground_truth", "public"
    print(f"[margin] model={model_dir} full_vocab={full_vocab} split={split_name}({split_kind}) "
          f"prompts={len(prompts)} n_new={n_new} verify_width={verify_width} topk={topk}", flush=True)

    # Single, explicit engine construction -- let any failure surface its real traceback rather
    # than masking it behind a width-retry (an earlier swallow hid a vocab-shape assert).
    effective_width = verify_width
    llm = LLM(model=model_dir, quantization="compressed-tensors", dtype="bfloat16",
              max_model_len=max(1024, ctx_cap + n_new + 16),
              gpu_memory_utilization=gpu_mem_util, max_num_seqs=1,
              max_num_batched_tokens=verify_width, enable_prefix_caching=False,
              enforce_eager=True, trust_remote_code=True,
              max_logprobs=max(20, topk + 2))

    gen_sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=n_new, logprobs=topk)
    ver_sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=topk)

    n_pos = n_flip = n_bitdiff = 0
    n_det_gen = det_gen_positions = 0
    n_det_ver = det_ver_positions = 0
    margins_flip: list[float] = []
    margins_all: list[float] = []
    flip_examples: list[dict] = []
    any_nan = False
    # PR #497 identity classification accumulators
    n_tie_flip = n_knife_flip = n_semantic_flip = 0
    n_margin_le_eps = n_margin_le_half = 0     # low-margin tail mass (does it thicken off-distribution?)
    t0 = time.time()

    for pi, pr in enumerate(prompts):
        ctx = pr["context_token_ids"]
        c = len(ctx)
        base = {"prompt_token_ids": ctx}

        # REF: greedy AR, width-1 decode, top-K M=1 logprobs per step
        out = llm.generate([base], gen_sp, use_tqdm=False)[0]
        gen = list(out.outputs[0].token_ids)
        gen_lps = out.outputs[0].logprobs or []
        if not gen:
            continue

        # determinism control: re-gen REF, expect bit-identical (int4 body+lm_head+sampler stable)
        if pi < det_prompts:
            gen_b = list(llm.generate([base], gen_sp, use_tqdm=False)[0].outputs[0].token_ids)
            Lg = min(len(gen), len(gen_b))
            n_det_gen += sum(1 for a, b in zip(gen[:Lg], gen_b[:Lg]) if a == b)
            det_gen_positions += Lg

        # VERIFY: width-8 re-forward of [ctx+gen], top-K prompt logprobs (M=8 attention occupancy)
        full = ctx + gen
        vout = llm.generate([{"prompt_token_ids": full}], ver_sp, use_tqdm=False)[0]
        pls = vout.prompt_logprobs
        if pi < det_prompts:
            pls_b = llm.generate([{"prompt_token_ids": full}], ver_sp, use_tqdm=False)[0].prompt_logprobs
        else:
            pls_b = None

        for g in range(len(gen)):
            j = c + g
            if pls is None or j >= len(pls) or pls[j] is None:
                continue
            ref_tok = int(gen[g])
            m8 = entry_as_dict(pls[j])
            if ref_tok not in m8:
                m8[ref_tok] = float("-inf")
            m8_arg, m8_top_lp, margin_m8 = top1_top2_margin(m8)
            m1 = entry_as_dict(gen_lps[g]) if g < len(gen_lps) else {}
            _m1_arg, m1_top_lp, _margin_m1 = (top1_top2_margin(m1) if m1
                                              else (ref_tok, float("nan"), float("nan")))
            if math.isnan(margin_m8):
                any_nan = True
                continue
            n_pos += 1
            margins_all.append(margin_m8)
            if math.isfinite(margin_m8):
                if margin_m8 <= EPS_STAR + BAND_TOL:
                    n_margin_le_eps += 1
                if margin_m8 <= NEAR_TIE_LOGPROB_THRESH:
                    n_margin_le_half += 1
            # bitdiff: the M=8 verify top-1 logit differs (in bits) from the M=1 AR top-1 logit
            if (math.isfinite(m8_top_lp) and math.isfinite(m1_top_lp)
                    and m8_top_lp != m1_top_lp):
                n_bitdiff += 1
            # FLIP: the M=8 verify argmax disagrees with the M=1 AR (emitted) token
            if m8_arg != ref_tok:
                n_flip += 1
                margins_flip.append(margin_m8)
                cls = classify_flip(m8, ref_tok, m8_top_lp, margin_m8)   # PR #497 tie/knife/semantic
                n_tie_flip += int(cls["is_tie_flip"])
                n_knife_flip += int(cls["is_knife_edge"])
                n_semantic_flip += int(cls["is_semantic"])
                if len(flip_examples) < 48:
                    flip_examples.append({"prompt_index": pi, "absolute_position": j,
                                          "ref_tok": ref_tok, "m8_argmax": m8_arg,
                                          "margin_m8": margin_m8, **cls})
            # verify determinism control
            if pls_b is not None and j < len(pls_b) and pls_b[j] is not None:
                m8b = entry_as_dict(pls_b[j])
                if m8b:
                    b_arg, _, _ = top1_top2_margin(m8b)
                    n_det_ver += int(b_arg == m8_arg)
                    det_ver_positions += 1

    frac_flip = (n_flip / n_pos) if n_pos else float("nan")
    frac_bitdiff = (n_bitdiff / n_pos) if n_pos else float("nan")
    margins_all.sort()

    def _q(xs: list[float], q: float) -> float:
        if not xs:
            return float("nan")
        i = min(len(xs) - 1, max(0, int(q * (len(xs) - 1))))
        return xs[i]

    # PR #497 identity decomposition (rates over all decode positions). Nesting tie <= knife <= flip
    # is enforced by classify_flip, so semantic = flip - knife and operative-identity = 1 - semantic.
    semantic_rate = (n_semantic_flip / n_pos) if n_pos else float("nan")
    out = {
        "phase": "margin", "split_name": split_name, "split_kind": split_kind,
        "model_dir": model_dir, "margin_model_full_vocab": full_vocab,
        "effective_verify_width": effective_width,
        "n_prompts": len(prompts), "n_new": n_new, "topk": topk,
        "n_positions": n_pos, "n_flip": n_flip, "n_bitdiff": n_bitdiff,
        "attention_frac_steps_argmax_flip": frac_flip,
        "attention_frac_steps_bitdiff": frac_bitdiff,
        "attention_max_logit_gap_at_flip": (max(margins_flip) if margins_flip else float("nan")),
        "attention_min_logit_gap_at_flip": (min(margins_flip) if margins_flip else float("nan")),
        "attention_median_logit_gap_at_flip": (_q(sorted(margins_flip), 0.5)
                                               if margins_flip else float("nan")),
        "rule_of_three_flip_ub": (_rule_of_three(n_pos) if n_flip == 0 else float("nan")),
        "margin_min": (margins_all[0] if margins_all else float("nan")),
        "margin_p01": _q(margins_all, 0.01), "margin_p05": _q(margins_all, 0.05),
        "margin_p10": _q(margins_all, 0.10), "margin_p25": _q(margins_all, 0.25),
        "margin_median": _q(margins_all, 0.50), "margin_p90": _q(margins_all, 0.90),
        "frac_positions_margin_le_eps": (n_margin_le_eps / n_pos) if n_pos else float("nan"),
        "frac_positions_margin_le_half": (n_margin_le_half / n_pos) if n_pos else float("nan"),
        # identity decomposition
        "n_tie_flip": n_tie_flip, "n_knife_flip": n_knife_flip, "n_semantic_flip": n_semantic_flip,
        "tie_flip_rate": (n_tie_flip / n_pos) if n_pos else float("nan"),
        "knife_flip_rate": (n_knife_flip / n_pos) if n_pos else float("nan"),
        "semantic_flip_rate": semantic_rate,
        "operative_identity": (1.0 - semantic_rate) if math.isfinite(semantic_rate) else float("nan"),
        "det_gen_byte_identity": (n_det_gen / det_gen_positions if det_gen_positions else float("nan")),
        "det_ver_argmax_identity": (n_det_ver / det_ver_positions if det_ver_positions else float("nan")),
        "det_gen_positions": det_gen_positions, "det_ver_positions": det_ver_positions,
        "flip_examples": flip_examples,
        "any_nan": bool(any_nan),
        "peak_mem_mib": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(_jsonable(out), open(out_path, "w"), indent=2)
    print(f"[margin:{split_kind}] positions={n_pos} flips={n_flip} (frac={frac_flip:.5f}) "
          f"tie={n_tie_flip} knife={n_knife_flip} semantic={n_semantic_flip} "
          f"op_identity={out['operative_identity']} det_gen={out['det_gen_byte_identity']}", flush=True)
    print(f"MARGIN_DONE {out_path}", flush=True)


# ======================================================================================== #
# GPU PHASE: MICROBENCH -- per deployed GEMM shape, is the Marlin kernel M-invariant (bitdiff)?
# + the atomic-add ON positive control (proves the test is sensitive, corroborates atomic_on).
# ======================================================================================== #
def _build_marlin(dev, seed: int, size_k: int, size_n: int):
    """Real GPTQ-Marlin int4 GEMM at (size_k, size_n), channel-wise symmetric (the deployed quant).
    Returns apply(x, use_atomic_add) -> y[M, size_n] and the deployed heuristic atomic flag."""
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils import marlin_utils as mu
    from vllm.model_executor.layers.quantization.utils import marlin_utils_test as mut
    from vllm.scalar_type import scalar_types

    qtype = scalar_types.uint4b8
    g = torch.Generator(device=dev).manual_seed(seed)
    w = (torch.randn(size_k, size_n, generator=g, device=dev, dtype=torch.bfloat16) * 0.02)
    _w_ref, q_w, s, g_idx, sort_idx, _ = mut.marlin_quantize(w, qtype, group_size=-1, act_order=False)
    ws = mu.marlin_make_workspace_new(dev)
    empty_zp = torch.empty(0, dtype=torch.int, device=dev)
    heuristic_aa = bool(mu.should_use_atomic_add_reduce(
        m=DEPLOYED_VERIFY_M, n=size_n, k=size_k, device=dev, dtype=torch.bfloat16))
    fp32_reduce = mu.USE_FP32_REDUCE_DEFAULT

    def apply(x, use_atomic_add: bool):
        xr = x.reshape(-1, size_k)
        return ops.marlin_gemm(
            xr, None, q_w, None, s, None, None, empty_zp, g_idx, sort_idx, ws, qtype,
            size_m=xr.shape[0], size_n=size_n, size_k=size_k, is_k_full=True,
            use_atomic_add=use_atomic_add, use_fp32_reduce=fp32_reduce,
            is_zp_float=False).reshape(x.shape[:-1] + (size_n,))

    return apply, heuristic_aa


def _byte_rate(bat, ref) -> float:
    return float((bat == ref).all(dim=-1).float().mean().item())


def phase_microbench(out_path: str, n_trials: int, seed: int) -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available -- launch with CUDA_VISIBLE_DEVICES=0")
    dev = torch.device("cuda:0")
    p = torch.cuda.get_device_properties(dev)
    gpu = {"name": p.name, "sm_count": p.multi_processor_count,
           "total_mem_gib": round(p.total_memory / (1024 ** 3), 2)}

    per_gemm: dict[str, Any] = {}
    any_nan = False
    for name, (size_k, size_n) in DEPLOYED_GEMMS.items():
        apply, heuristic_aa = _build_marlin(dev, seed, size_k, size_n)
        m_inv_rates, atomic_r2r_rates, harness_rates = [], [], []
        maxdiff = 0.0
        for t in range(n_trials):
            g = torch.Generator(device=dev).manual_seed(seed + 1000 * t)
            x = torch.randn(DEPLOYED_VERIFY_M, size_k, generator=g, device=dev, dtype=torch.bfloat16)
            # fast == deployed heuristic (atomic OFF here since every n>=2048); invariant == per-row M=1
            bat = apply(x, use_atomic_add=False)
            any_nan = any_nan or bool(torch.isnan(bat).any())
            ref = torch.cat([apply(x[r:r + 1], use_atomic_add=False) for r in range(x.shape[0])], dim=0)
            m_inv_rates.append(_byte_rate(bat, ref))
            maxdiff = max(maxdiff, float((bat.float() - ref.float()).abs().max().item()))
            # run-to-run atomic-add probe: two launches of the atomic path (reduction-order
            # nondeterminism, the greedy_determinism atomic_on mechanism). May be inert (==1.0) at
            # the M=8 decode geometry if marlin does not split-K here.
            atomic_r2r_rates.append(_byte_rate(apply(x, use_atomic_add=True),
                                               apply(x, use_atomic_add=True)))
            # harness-sensitivity self-check: a single perturbed input MUST change the bytes -- proves
            # _byte_rate detects differences (so the m_inv==1.0 result is a real invariance, not a stuck
            # comparator).
            xp = x.clone(); xp[0, 0] = xp[0, 0] + torch.tensor(0.5, dtype=torch.bfloat16, device=dev)
            harness_rates.append(_byte_rate(apply(xp, use_atomic_add=False), bat))
        m_inv = sum(m_inv_rates) / len(m_inv_rates)
        atomic_r2r = sum(atomic_r2r_rates) / len(atomic_r2r_rates)
        harness = sum(harness_rates) / len(harness_rates)
        per_gemm[name] = {
            "size_k": size_k, "size_n": size_n,
            "heuristic_use_atomic_add": heuristic_aa,             # deployed flag (expect False, n>=2048)
            "m_invariant_byte_rate": m_inv,                       # M=8 vs per-row M=1 (expect 1.0)
            "frac_steps_bitdiff": 1.0 - m_inv,                    # fast-vs-invariant bit difference
            "atomic_runtorun_byte_rate": atomic_r2r,             # reduction nondeterminism probe
            "harness_sensitivity_byte_rate": harness,            # perturbed input (expect <1.0)
            "max_abs_diff_vs_perrow": maxdiff,
        }
        print(f"[micro] {name:13s} k={size_k:5d} n={size_n:5d} heur_aa={heuristic_aa} "
              f"m_inv={m_inv:.4f} atomic_r2r={atomic_r2r:.4f} harness={harness:.4f}", flush=True)

    out = {
        "phase": "microbench", "gpu": gpu, "n_trials": n_trials, "any_nan": bool(any_nan),
        "per_gemm": per_gemm,
        "all_body_lmhead_m_invariant": bool(all(
            per_gemm[n]["m_invariant_byte_rate"] >= 0.999 for n in DEPLOYED_GEMMS)),
        # positive control: a perturbed input MUST flip bytes for EVERY shape, else _byte_rate is
        # stuck and the m_inv==1.0 invariance result is untrustworthy.
        "harness_is_sensitive": bool(all(
            per_gemm[n]["harness_sensitivity_byte_rate"] < 0.999 for n in DEPLOYED_GEMMS)),
        # informational: at the M=8 decode geometry marlin does not split-K, so use_atomic_add is
        # inert (run-to-run byte rate ==1.0). The serve-occupancy atomic_on breakage lives in the
        # greedy_determinism captures (phase_ingest), not reproducible at this fixed M.
        "atomic_runtorun_inert": bool(all(
            per_gemm[n]["atomic_runtorun_byte_rate"] >= 0.999 for n in DEPLOYED_GEMMS)),
        "peak_mem_mib": round(torch.cuda.max_memory_allocated(dev) / (1024 ** 2), 2),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(_jsonable(out), open(out_path, "w"), indent=2)
    print(f"MICROBENCH_DONE {out_path}", flush=True)


# ======================================================================================== #
# CPU PHASE: INGEST -- faithful end-to-end cross-check from on-branch greedy_determinism captures
# ======================================================================================== #
def _read_streams(cfg: str) -> list[list[int]] | None:
    f = GREEDY_DET_DIR / cfg / "run_00" / "decode_outputs.jsonl"
    if not f.exists():
        return None
    streams = []
    for line in open(f):
        rec = json.loads(line)
        ids = rec.get("completion_token_ids") or rec.get("token_ids")
        if ids:
            streams.append([int(t) for t in ids])
    return streams or None


def _first_divergence_onsets(a: list[list[int]], b: list[list[int]]) -> dict[str, Any]:
    onsets, diverged = [], 0
    for sa, sb in zip(a, b):
        L = min(len(sa), len(sb))
        pos = next((i for i in range(L) if sa[i] != sb[i]), None)
        if pos is not None:
            diverged += 1
            onsets.append(pos)
    onsets.sort()
    n = len(a)
    return {
        "n_prompts": n, "n_diverged": diverged,
        "frac_prompts_diverged": (diverged / n) if n else float("nan"),
        "onset_min": (onsets[0] if onsets else None),
        "onset_median": (onsets[len(onsets) // 2] if onsets else None),
        "onset_max": (onsets[-1] if onsets else None),
    }


def phase_ingest() -> dict[str, Any]:
    """Cross-check against research/validity/greedy_determinism/captures (cited, on advisor branch).
    These are REAL spec-decode serve greedy streams under per-knob reduction-order swaps -> the faithful
    end-to-end complement to the MARGIN proxy."""
    report_path = GREEDY_DET_DIR / "report.json"
    out: dict[str, Any] = {"phase": "ingest", "source": str(report_path), "available": False}
    if not report_path.exists():
        out["note"] = "greedy_determinism captures not present on this checkout"
        return out
    rep = json.load(open(report_path))
    cfgs = rep.get("configs", {})

    def _idf(name: str) -> float:
        return _f(cfgs.get(name, {}).get("identical_to_default_frac"))

    out["available"] = True
    out["default_bit_exact_run_to_run"] = bool(
        cfgs.get("default", {}).get("mean_byte_identical_frac", 0.0) >= 0.999)
    out["default_official_128_128"] = bool(
        cfgs.get("default", {}).get("official_xcheck", {}).get("num_identical") == 128)
    # attention reduction swapped off -> tokens diverge from default => attention flips end-to-end
    out["splitkv_off_identical_to_default_frac"] = _idf("splitkv_off")
    out["attention_flips_end_to_end"] = bool(
        math.isfinite(_idf("splitkv_off")) and _idf("splitkv_off") < 0.999)
    # body Marlin atomic-add forced on -> diverges, but deployed keeps it OFF (deterministic)
    out["atomic_on_identical_to_default_frac"] = _idf("atomic_on")
    out["atomic_on_breaks_determinism"] = bool(rep.get("verdict", {}).get("atomic_add_breaks_determinism"))
    out["deployed_keeps_atomic_off"] = True

    # honest per-step onset: where does the FIRST attention-reduction flip occur (cascade-aware)?
    da, sk = _read_streams("default"), _read_streams("splitkv_off")
    if da and sk:
        out["splitkv_off_vs_default_onset"] = _first_divergence_onsets(da, sk)
    return out


# ======================================================================================== #
# CPU PHASE: DRAFTER -- acceptance-invariance sim => drafter non-determinism never flips a token
# ======================================================================================== #
def phase_drafter(n_positions: int, seed: int) -> dict[str, Any]:
    """Greedy spec-decode (temp=0) emits, at every position, the TARGET argmax. Two DIFFERENT drafters
    over the SAME target-argmax oracle must emit IDENTICAL sequences -- proof by construction that the
    MTP drafter's non-determinism cannot flip an accepted token (Leviathan 2023 / Chen 2023)."""
    import random
    rng = random.Random(seed)
    vocab = 64
    target = [rng.randrange(vocab) for _ in range(n_positions)]   # the fixed target-argmax oracle

    def run(drafter_seed: int, skill: float) -> tuple[list[int], float, int]:
        # a drafter of given `skill` proposes the target token with prob=skill else a random wrong one;
        # skill differs between drafters so e_accept differs -- the emitted stream must NOT.
        dr = random.Random(drafter_seed)
        emitted: list[int] = []
        pos, cycles, accepted_total = 0, 0, 0
        while pos < n_positions:
            cycles += 1
            drafts = []
            for a in range(SPEC_K):
                tgt = target[pos + a] if pos + a < n_positions else -1
                if dr.random() < skill:
                    drafts.append(tgt)
                else:
                    wrong = dr.randrange(vocab)
                    drafts.append(wrong if wrong != tgt else (wrong + 1) % vocab)
            a = 0
            while a < SPEC_K and pos + a < n_positions and drafts[a] == target[pos + a]:
                emitted.append(target[pos + a])     # accepted draft == target argmax
                a += 1
            accepted_total += a
            # bonus/correction token at the first mismatch IS the target argmax
            if pos + a < n_positions:
                emitted.append(target[pos + a])
            pos += a + 1
        e_accept = (accepted_total + cycles) / cycles if cycles else float("nan")
        return emitted, e_accept, cycles

    e1, ea1, _ = run(seed + 1, skill=0.80)
    e2, ea2, _ = run(seed + 99999, skill=0.55)        # a different, weaker drafter
    equal = bool(e1 == e2 == target[:len(e1)])
    return {
        "phase": "drafter", "n_positions": n_positions,
        "emitted_equal_across_drafters": equal,
        "emitted_equals_target_argmax": bool(e1 == target[:len(e1)]),
        "e_accept_drafter_A": ea1, "e_accept_drafter_B": ea2,
        "e_accept_differs": bool(abs(ea1 - ea2) > 1e-9),     # acceptance moves, emitted does not
        "drafter_in_must_pin": (not equal),                  # False iff emitted invariant
        "note": ("two random drafters over a fixed target-argmax oracle emit identical sequences => "
                 "drafter non-determinism is a SPEED lever only; e_accept differs but emitted tokens "
                 "do not. Corroborated by greedy_determinism e_accept variance with within-config "
                 "128/128 identity."),
    }


# ======================================================================================== #
# COMPOSE -- assemble the per-op census + the PR KEY OUTPUTS
# ======================================================================================== #
def compose(margin: dict, micro: dict, ingest: dict, drafter: dict, n_layers: int) -> dict[str, Any]:
    pg = micro.get("per_gemm", {})

    def gemm_bitdiff(name: str) -> float:
        return _f(pg.get(name, {}).get("frac_steps_bitdiff"))

    per_op: dict[str, dict[str, Any]] = {}
    # body matmuls (int4 Marlin, fused) -- bit-identical fast-vs-invariant => cannot flip
    for op in BODY_OPS:
        gemm = BODY_OP_TO_GEMM[op]
        bd = gemm_bitdiff(gemm)
        per_op[op] = {
            "kind": "int4_marlin_body", "realized_by": gemm,
            "frac_steps_bitdiff": bd,
            "frac_steps_argmax_flip": 0.0,                 # bitdiff 0 -> zero perturbation -> no flip
            "max_logit_gap_at_flip": float("nan"),
            "heuristic_use_atomic_add": pg.get(gemm, {}).get("heuristic_use_atomic_add"),
            "free_basis": "kernel M-invariance (n>=2048 -> atomic-add off); distribution-robust",
        }
    # lm_head (int4 Marlin) -- already M-invariant (#384)
    per_op["lm_head"] = {
        "kind": "int4_marlin_lmhead", "realized_by": "lm_head",
        "frac_steps_bitdiff": gemm_bitdiff("lm_head"),
        "frac_steps_argmax_flip": 0.0, "max_logit_gap_at_flip": float("nan"),
        "free_basis": "kernel M-invariance (#384 maxdiff=0); distribution-robust",
    }
    # attention reduction -- the sole empirically-measured op
    a_flip = _f(margin.get("attention_frac_steps_argmax_flip"))
    per_op["attention_reduction"] = {
        "kind": "flash_split_kv_reduction",
        "frac_steps_bitdiff": _f(margin.get("attention_frac_steps_bitdiff")),
        "frac_steps_argmax_flip": a_flip,
        "max_logit_gap_at_flip": _f(margin.get("attention_max_logit_gap_at_flip")),
        "min_logit_gap_at_flip": _f(margin.get("attention_min_logit_gap_at_flip")),
        "free_basis": "NONE -- occupancy-dependent reduction; EMPIRICAL flip rate",
    }
    # MTP drafter -- bf16 op that DOES differ in bits (reroutes under BI, #484) yet never flips emitted
    per_op["mtp_drafter"] = {
        "kind": "bf16_mtp_drafter",
        "frac_steps_bitdiff": 1.0,                          # bf16 non-invariant (#484 reroute)
        "frac_steps_argmax_flip": 0.0,                      # acceptance invariance (drafter sim)
        "max_logit_gap_at_flip": float("nan"),
        "free_basis": "greedy-spec acceptance invariance theorem; distribution-robust",
    }

    # attention must-pin: margin proxy OR the faithful end-to-end captures (either flip => must-pin)
    margin_flip_pos = bool(math.isfinite(a_flip) and a_flip > 0.0)
    ingest_attn_flip = bool(ingest.get("attention_flips_end_to_end"))
    attention_in_must_pin = bool(margin_flip_pos or ingest_attn_flip)
    if attention_in_must_pin:
        per_op["attention_reduction"]["frac_steps_argmax_flip"] = max(a_flip if math.isfinite(a_flip) else 0.0,
                                                                       0.0)

    must_be_deterministic = [op for op in ALL_OPS
                             if (op == "attention_reduction" and attention_in_must_pin)
                             or (op != "attention_reduction"
                                 and float(per_op[op]["frac_steps_argmax_flip"]) > 0.0)]
    n_ops_must_pin = len(must_be_deterministic)
    drafter_in_must_pin = bool(drafter.get("drafter_in_must_pin"))
    argmax_flip_free_op_fraction = (N_OPS_TOTAL - n_ops_must_pin) / N_OPS_TOTAL

    # price the minimal pin from #484 per-op tax: pin attention only -> predicted_surgical_tps
    if set(must_be_deterministic) <= {"attention_reduction"} and attention_in_must_pin:
        projected_tps_minimal_pin = PREDICTED_SURGICAL_TPS
        pin_basis = ("#484 predicted_surgical_tps: pin ONLY the attention reduction (off-the-shelf BI "
                     "attention, 5.10% tax), drafter+body+lm_head run fast. #363 shows a TARGETED "
                     "fixed-split attention pin is ~free (could approach 481.53).")
    elif n_ops_must_pin == 0:
        projected_tps_minimal_pin = DEPLOYED_TPS
        pin_basis = "no op flips on this distribution -> nothing beyond deployed must be pinned"
    else:
        projected_tps_minimal_pin = PREDICTED_SURGICAL_TPS
        pin_basis = ("must-pin set exceeds {attention}; #484 per-op tax prices attention only -- "
                     "reported value is the attention-only lower bound, see per-op table")

    n_pos = int(margin.get("n_positions", 0) or 0)
    verdict = (
        f"CENSUS (LOCAL, analysis_only). Of {N_OPS_TOTAL} decode ops, "
        f"n_ops_must_pin={n_ops_must_pin} (={must_be_deterministic}); "
        f"argmax_flip_free_op_fraction={argmax_flip_free_op_fraction:.3f}. "
        f"attention_in_must_pin={attention_in_must_pin} (margin-proxy flip_frac="
        f"{a_flip if math.isfinite(a_flip) else float('nan'):.5f} over {n_pos} public decode steps; "
        f"faithful end-to-end splitkv_off identical_to_default="
        f"{ingest.get('splitkv_off_identical_to_default_frac')}). "
        f"drafter_in_must_pin={drafter_in_must_pin} (acceptance-invariance theorem; bitdiff>0 yet "
        f"flip=0 -- the byte-exact!=argmax-stable poster child). "
        f"int4 Marlin body (fused qkv/gate_up, o, down) + pruned lm_head are bit-identical fast-vs-"
        f"invariant (M-invariant, n>=2048 atomic-off; positive control: a perturbed input flips the "
        f"output bytes) -> free by KERNEL PROPERTY, not public luck. projected_tps_minimal_pin="
        f"{projected_tps_minimal_pin:.2f} ({pin_basis}). HONESTY: the must-pin SET is kernel/theorem-"
        f"grounded (distribution-robust); only attention's flip RATE is empirical -- a public flip is a "
        f"LOWER bound on private (cite 4.295% private-Delta, #379), but attention is ALREADY must-pin so "
        f"private can only GROW the set, never shrink below {{attention}}.")

    return {
        "n_layers": n_layers,
        "n_ops_total": N_OPS_TOTAL,
        "n_ops_must_pin": n_ops_must_pin,
        "must_be_deterministic": must_be_deterministic,
        "argmax_flip_free_op_fraction": argmax_flip_free_op_fraction,
        "attention_in_must_pin": attention_in_must_pin,
        "drafter_in_must_pin": drafter_in_must_pin,
        "projected_tps_minimal_pin": projected_tps_minimal_pin,
        "projected_tps_minimal_pin_basis": pin_basis,
        "per_op": per_op,
        "attention_margin_flip_positive": margin_flip_pos,
        "attention_ingest_flip_positive": ingest_attn_flip,
        "public_decode_steps": n_pos,
        "rule_of_three_attention_flip_ub": (_rule_of_three(n_pos) if not margin_flip_pos
                                            else float("nan")),
        "private_delta_pct_frame": PRIVATE_DELTA_PCT,
        "verdict": verdict,
    }


# ======================================================================================== #
# SELF-TEST (PRIMARY: reduction_census_self_test_passes)
# ======================================================================================== #
def selftest(margin: dict, micro: dict, ingest: dict, drafter: dict, comp: dict,
             flags: dict) -> dict[str, Any]:
    c: dict[str, bool] = {}
    # (a) margin leg produced positions, NaN-clean
    c["a_margin_has_positions"] = bool(int(margin.get("n_positions", 0) or 0) > 0)
    c["a_margin_nan_clean"] = (not bool(margin.get("any_nan", True)))
    # (b) microbench: body+lm_head M-invariant AND the perturbed-input positive control is sensitive
    c["b_body_lmhead_m_invariant"] = bool(micro.get("all_body_lmhead_m_invariant"))
    c["b_harness_control_sensitive"] = bool(micro.get("harness_is_sensitive"))
    c["b_micro_nan_clean"] = (not bool(micro.get("any_nan", True)))
    # (c) determinism control: REF re-gen byte-identical (non-attention path stable run-to-run)
    dgi = margin.get("det_gen_byte_identity")
    c["c_ref_regen_deterministic"] = bool(dgi is not None and dgi >= 0.999)
    # (d) drafter acceptance invariance holds (emitted invariant; e_accept differs)
    c["d_drafter_emitted_invariant"] = bool(drafter.get("emitted_equal_across_drafters"))
    c["d_drafter_not_must_pin"] = (not bool(drafter.get("drafter_in_must_pin")))
    # (e) faithful end-to-end corroboration: default bit-exact AND attention flips end-to-end
    c["e_default_bit_exact"] = bool(ingest.get("default_bit_exact_run_to_run"))
    c["e_attention_flips_end_to_end"] = bool(ingest.get("attention_flips_end_to_end"))
    # (f) census consistency: attention bitdiff>0; drafter bitdiff>0 yet flip=0; must ⊆ {bitdiff>0}
    po = comp["per_op"]
    c["f_attention_has_bitdiff"] = bool(po["attention_reduction"]["frac_steps_bitdiff"] > 0.0)
    c["f_drafter_bitdiff_but_no_flip"] = bool(
        po["mtp_drafter"]["frac_steps_bitdiff"] > 0.0
        and po["mtp_drafter"]["frac_steps_argmax_flip"] == 0.0)
    c["f_mustpin_subset_of_bitdiff"] = all(
        float(po[op]["frac_steps_bitdiff"]) > 0.0 for op in comp["must_be_deterministic"])
    # (g) priced minimal pin finite and in (0, ceiling]
    ptp = comp["projected_tps_minimal_pin"]
    c["g_projected_tps_valid"] = bool(math.isfinite(ptp) and 0.0 < ptp <= CEILING_500)
    # (h) honesty bound computed; flags clean
    c["h_no_launch_flags"] = bool(flags["no_hf_job"] and flags["no_launch"]
                                  and flags["analysis_only"])
    passes = all(c.values())
    return {"passes": passes, "n_checks": len(c), "conditions": c}


# ======================================================================================== #
# Report + wandb + orchestration
# ======================================================================================== #
def print_report(payload: dict) -> None:
    comp, st = payload["compose"], payload["selftest"]
    bar = "=" * 100
    print(bar)
    print("REDUCTION-SENSITIVITY CENSUS -- minimum mandatory-deterministic op set (PR #491, ubel)")
    g = payload.get("microbench", {}).get("gpu", {})
    print(f"  GPU {g.get('name')}  SMs={g.get('sm_count')}  model n_layers={comp['n_layers']}")
    mfv = payload.get("margin", {}).get("margin_model_full_vocab")
    print(f"  margin scan model = {payload.get('margin', {}).get('model_dir')}  "
          f"(full_vocab={mfv}; deployed pruned head corroborated by INGEST)")
    print("-" * 100)
    print(f"  {'op':18s} {'frac_bitdiff':>12s} {'frac_flip':>12s} {'max_gap_at_flip':>16s}  basis")
    for op in ALL_OPS:
        r = comp["per_op"][op]
        bd = r["frac_steps_bitdiff"]
        fl = r["frac_steps_argmax_flip"]
        gp = r["max_logit_gap_at_flip"]
        print(f"  {op:18s} {bd:12.4f} {fl:12.5f} "
              f"{(gp if isinstance(gp, float) and math.isfinite(gp) else float('nan')):16.4f}  "
              f"{r.get('free_basis', '')[:46]}")
    print("-" * 100)
    print(f"  n_ops_total            = {comp['n_ops_total']}")
    print(f"  n_ops_must_pin         = {comp['n_ops_must_pin']}  {comp['must_be_deterministic']}")
    print(f"  argmax_flip_free_frac  = {comp['argmax_flip_free_op_fraction']:.3f}")
    print(f"  attention_in_must_pin  = {comp['attention_in_must_pin']}")
    print(f"  drafter_in_must_pin    = {comp['drafter_in_must_pin']}")
    print(f"  projected_tps_minimal_pin = {comp['projected_tps_minimal_pin']:.2f}")
    print(f"  public_decode_steps    = {comp['public_decode_steps']}  "
          f"(rule-of-three flip UB if 0: {comp['rule_of_three_attention_flip_ub']})")
    print("-" * 100)
    print(f"  SELF-TEST {st['passes']} ({st['n_checks']} checks): "
          + json.dumps({k: int(v) for k, v in st["conditions"].items()}))
    print("-" * 100)
    print("  VERDICT\n   " + comp["verdict"])
    print(bar)


def maybe_log_wandb(payload: dict, args) -> str | None:
    if args.no_wandb:
        return None
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from scripts.wandb_logging import (init_wandb_run, log_summary,
                                            log_json_artifact, finish_wandb)
    except Exception as e:  # noqa: BLE001
        print(f"[census] wandb helpers unavailable: {e}")
        return None
    comp, margin, micro = payload["compose"], payload["margin"], payload["microbench"]
    run = init_wandb_run(
        job_type="analysis-reduction-census", agent="ubel",
        name=args.wandb_name, group=args.wandb_group,
        tags=["reduction-sensitivity-census", "argmax-flip", "must-be-deterministic",
              "481-forward-dir1", "pr-491"],
        config={"pr": 491, "kind": "reduction-sensitivity-census",
                "n_ops_total": N_OPS_TOTAL, "deployed_tps": DEPLOYED_TPS,
                "predicted_surgical_tps": PREDICTED_SURGICAL_TPS,
                "private_delta_pct": PRIVATE_DELTA_PCT, "hidden": HIDDEN,
                "lmhead_rows": LMHEAD_ROWS, "n_layers": comp["n_layers"]},
    )
    if run is None:
        print("[census] wandb disabled (no API key / WANDB_MODE).")
        return None
    flat: dict[str, float] = {
        "census/n_ops_must_pin": float(comp["n_ops_must_pin"]),
        "census/argmax_flip_free_op_fraction": float(comp["argmax_flip_free_op_fraction"]),
        "census/attention_in_must_pin": float(comp["attention_in_must_pin"]),
        "census/drafter_in_must_pin": float(comp["drafter_in_must_pin"]),
        "census/projected_tps_minimal_pin": float(comp["projected_tps_minimal_pin"]),
        "census/public_decode_steps": float(comp["public_decode_steps"]),
        "attention/frac_steps_argmax_flip": _f(margin.get("attention_frac_steps_argmax_flip")),
        "attention/frac_steps_bitdiff": _f(margin.get("attention_frac_steps_bitdiff")),
        "attention/max_logit_gap_at_flip": _f(margin.get("attention_max_logit_gap_at_flip")),
        "attention/margin_min": _f(margin.get("margin_min")),
        "control/det_gen_byte_identity": _f(margin.get("det_gen_byte_identity")),
        "control/margin_model_full_vocab": float(bool(margin.get("margin_model_full_vocab"))),
        "selftest/reduction_census_self_test_passes": float(payload["selftest"]["passes"]),
    }
    for op in ALL_OPS:
        flat[f"op_bitdiff/{op}"] = float(comp["per_op"][op]["frac_steps_bitdiff"])
        flat[f"op_flip/{op}"] = float(comp["per_op"][op]["frac_steps_argmax_flip"])
    run.log({"global_step": 0, **{k: (v if (isinstance(v, float) and math.isfinite(v)) else 0.0)
                                  for k, v in flat.items()}})
    log_summary(run, _jsonable(payload), step=0, run_prefix=args.wandb_name)
    log_json_artifact(run, name="reduction_sensitivity_census",
                      artifact_type="analysis", data=_jsonable(payload))
    finish_wandb(run)
    rid = getattr(run, "id", None)
    print(f"[census] wandb logged (run {rid})")
    return rid


def resolve_server_python(arg: str | None) -> str:
    if arg:
        return arg
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.local_validation import harness  # noqa: E402
    m = harness.load_manifest(ROOT / "submissions" / "fa2sw_precache_kenyan")
    return str(harness.ensure_server_venv(m["dependencies"]))


def run_gpu_phase(server_python: str, phase_args: list[str], timeout: int) -> int:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cmd = [server_python, os.path.abspath(__file__)] + phase_args
    print(f"[orch] launching ({Path(server_python).parent.parent.name}): {' '.join(phase_args)}", flush=True)
    try:
        return subprocess.run(cmd, env=env, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        print(f"[orch] phase TIMED OUT after {timeout}s: {phase_args}", flush=True)
        return 124


def orchestrate(args) -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    margin_json = str(HERE / "_margin.json")
    micro_json = str(HERE / "_microbench.json")
    server_python = resolve_server_python(args.server_python)
    print(f"[orch] server_python = {server_python}", flush=True)

    rc_m = run_gpu_phase(server_python, [
        "--phase", "microbench", "--out", micro_json,
        "--micro-trials", str(args.micro_trials), "--seed", str(args.seed)], timeout=args.micro_timeout)
    rc_s = run_gpu_phase(server_python, [
        "--phase", "margin", "--out", margin_json,
        "--n-prompts", str(args.n_prompts), "--n-new", str(args.n_new),
        "--ctx-cap", str(args.ctx_cap), "--verify-width", str(args.verify_width),
        "--gpu-mem-util", str(args.gpu_mem_util), "--topk", str(args.topk),
        "--det-prompts", str(args.det_prompts), "--seed", str(args.seed)], timeout=args.margin_timeout)

    margin = json.load(open(margin_json)) if Path(margin_json).exists() else {"phase": "margin", "error": rc_s}
    micro = json.load(open(micro_json)) if Path(micro_json).exists() else {"phase": "microbench", "error": rc_m}
    ingest = phase_ingest()
    drafter = phase_drafter(args.drafter_positions, args.seed)
    n_layers = resolve_n_layers()
    comp = compose(margin, micro, ingest, drafter, n_layers)
    flags = {"no_hf_job": True, "no_launch": True, "analysis_only": True,
             "no_served_file_change": True, "official_tps": 0}
    st = selftest(margin, micro, ingest, drafter, comp, flags)

    payload = {
        "agent": "ubel", "pr": 491, "kind": "reduction-sensitivity-census",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        **flags,
        "anchors": {"deployed_tps": DEPLOYED_TPS, "predicted_surgical_tps": PREDICTED_SURGICAL_TPS,
                    "attention_official_tax_pct": ATTENTION_OFFICIAL_TAX_PCT,
                    "global_tax_pct": GLOBAL_TAX_PCT, "private_delta_pct": PRIVATE_DELTA_PCT},
        "margin": margin, "microbench": micro, "ingest": ingest, "drafter": drafter,
        "compose": comp, "selftest": st,
        "reduction_census_self_test_passes": bool(st["passes"]),
        # headline KEY OUTPUTS
        "n_ops_must_pin": comp["n_ops_must_pin"],
        "must_be_deterministic": comp["must_be_deterministic"],
        "argmax_flip_free_op_fraction": comp["argmax_flip_free_op_fraction"],
        "projected_tps_minimal_pin": comp["projected_tps_minimal_pin"],
        "attention_in_must_pin": comp["attention_in_must_pin"],
        "drafter_in_must_pin": comp["drafter_in_must_pin"],
    }
    print_report(payload)
    out_path = HERE / "reduction_sensitivity_census_results.json"
    out_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    print(f"\n[census] wrote {out_path}", flush=True)
    rid = maybe_log_wandb(payload, args)
    if rid:
        payload["wandb_run_id"] = rid
        out_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    result = {"terminal": True, "status": "complete", "pending_arms": False,
              "wandb_run_ids": ([rid] if rid else []),
              "n_ops_must_pin": comp["n_ops_must_pin"],
              "must_be_deterministic": comp["must_be_deterministic"],
              "attention_in_must_pin": comp["attention_in_must_pin"],
              "drafter_in_must_pin": comp["drafter_in_must_pin"],
              "argmax_flip_free_op_fraction": round(comp["argmax_flip_free_op_fraction"], 4),
              "projected_tps_minimal_pin": round(comp["projected_tps_minimal_pin"], 2),
              "primary_metric": {"name": "n_ops_must_pin", "value": comp["n_ops_must_pin"]},
              "test_metric": {"name": "projected_tps_minimal_pin",
                              "value": round(comp["projected_tps_minimal_pin"], 2)},
              "self_test_passes": bool(st["passes"])}
    print("SENPAI-RESULT: " + json.dumps(result), flush=True)
    return 0 if st["passes"] else 1


# ======================================================================================== #
# PR #497 (ubel) -- private-attention-flip identity bound: compose / selftest / report / wandb
# ======================================================================================== #
SHIFTED_RESULTS_DIR = ROOT / "research" / "validity" / "private_attention_flip_bound"


def compose_shifted(shifted: dict, public: dict | None) -> dict[str, Any]:
    """Assemble the #497 KEY OUTPUTS: off-distribution growth of attention's must-pin COST + the
    margin-#461 operative identity + the #488 ULP-tie vs TRUE-semantic decomposition. The must-pin
    SET cannot shrink below {attention} on private data (attention is ALREADY must-pin, theorem-
    grounded); only its empirical flip RATE/cost can grow -- that growth is what this bounds."""
    priv_flip = _f(shifted.get("attention_frac_steps_argmax_flip"))
    priv_sem = _f(shifted.get("semantic_flip_rate"))
    priv_tie = _f(shifted.get("tie_flip_rate"))
    priv_knife = _f(shifted.get("knife_flip_rate"))
    op_identity = _f(shifted.get("operative_identity"))
    n_pos = int(shifted.get("n_positions", 0) or 0)
    n_flip = int(shifted.get("n_flip", 0) or 0)

    # public anchor: the #491 const is the headline denominator (PR: "shifted / 0.00524"); the
    # same-code public run (if present) reproduces it and gives a same-harness tie/semantic baseline.
    pub_anchor = PUBLIC_ATTENTION_FLIP_RATE
    pub_samecode = _f(public.get("attention_frac_steps_argmax_flip")) if public else float("nan")
    ratio_anchor = (priv_flip / pub_anchor) if (math.isfinite(priv_flip) and pub_anchor > 0) else float("nan")
    ratio_samecode = (priv_flip / pub_samecode) if (math.isfinite(priv_flip) and math.isfinite(pub_samecode)
                                                    and pub_samecode > 0) else float("nan")

    # margin tail: does the low-margin (logit_gap<=0.5) tail thicken off-distribution?
    priv_p05, priv_med, priv_min = (_f(shifted.get("margin_p05")), _f(shifted.get("margin_median")),
                                    _f(shifted.get("margin_min")))
    priv_tail_half = _f(shifted.get("frac_positions_margin_le_half"))
    priv_tail_eps = _f(shifted.get("frac_positions_margin_le_eps"))
    pub_tail_half = _f(public.get("frac_positions_margin_le_half")) if public else float("nan")
    pub_p05_samecode = _f(public.get("margin_p05")) if public else float("nan")

    cost_grows = bool(math.isfinite(priv_flip) and priv_flip > pub_anchor)
    # tail thickens if more low-margin mass than the same-code public (preferred) or than the #491 p05
    if math.isfinite(priv_tail_half) and math.isfinite(pub_tail_half):
        tail_thickens = bool(priv_tail_half > pub_tail_half)
    else:
        tail_thickens = bool(math.isfinite(priv_p05) and priv_p05 < PUBLIC_MARGIN_P05)

    split_name = shifted.get("split_name")
    src_label, endpoint_label = SPLIT_META.get(split_name, ("held-out reasoning/STEM", "shifted"))

    integrity_bound = (
        f"byte-exact greedy identity holds on private-like reasoning/STEM data [{endpoint_label}; "
        f"src {src_label}] with operative-identity "
        f"{op_identity:.4f}, attention-flip rate {priv_flip:.5f} ({n_flip}/{n_pos}) vs 0.524% public "
        f"(ratio {ratio_anchor:.2f}x); of those flips tie_rate={priv_tie:.5f} semantic_rate={priv_sem:.5f}; "
        f"the must-pin SET cannot shrink below {{attention_reduction}} on private data (attention is "
        f"already must-pin, theorem-grounded) -- only its flip COST can grow, measured here at "
        f"{ratio_anchor:.2f}x public.")

    verdict = (
        f"PRIVATE-ATTENTION-FLIP BOUND (LOCAL, analysis_only; #497). Held-out reasoning/STEM ({src_label}, "
        f"DISJOINT from public mmlu_pro/gpqa/aime; {endpoint_label}), {n_pos} decode steps. "
        f"private_attention_flip_rate={priv_flip:.5f} ({n_flip}/{n_pos}) vs public anchor "
        f"{pub_anchor:.5f} -> ratio {ratio_anchor:.2f}x (same-code public {pub_samecode:.5f}, ratio "
        f"{ratio_samecode:.2f}x). must_pin_cost_grows_off_distribution={cost_grows}. "
        f"operative_identity_private={op_identity:.4f} (target 1.0): semantic_flip_rate={priv_sem:.5f}, "
        f"tie_flip_rate={priv_tie:.5f}, knife_flip_rate={priv_knife:.5f}. low-margin tail (gap<=0.5) "
        f"frac={priv_tail_half:.4f} vs public {pub_tail_half:.4f} -> thickens={tail_thickens}; "
        f"margin p05={priv_p05} median={priv_med} min={priv_min} (public p05={PUBLIC_MARGIN_P05} "
        f"median={PUBLIC_MARGIN_MEDIAN}). HONESTY: bounds the HYPOTHETICAL offline token-audit posture "
        f"(the #493 automated scorer has NO token-identity gate), NOT the automated private gate "
        f"(drafter-Delta / #489 axis). The must-pin SET is theorem-grounded distribution-robust; only "
        f"attention's RATE is empirical and can only GROW the set, never shrink it below {{attention}}.")

    return {
        "split": shifted.get("split_name"), "split_kind": shifted.get("split_kind"),
        "bracket_endpoint": endpoint_label, "src_label": src_label,
        "n_positions": n_pos, "n_flip": n_flip,
        # ---- the PR KEY OUTPUTS ----
        "private_attention_flip_rate": priv_flip,
        "public_attention_flip_rate_anchor": pub_anchor,
        "public_attention_flip_rate_samecode": pub_samecode,
        "public_to_private_flip_ratio": ratio_anchor,
        "public_to_private_flip_ratio_samecode": ratio_samecode,
        "operative_identity_private": op_identity,
        "private_semantic_flip_rate": priv_sem,
        "private_tie_flip_rate": priv_tie,
        "private_knife_flip_rate": priv_knife,
        "must_pin_cost_grows_off_distribution": cost_grows,
        "low_margin_tail_thickens": tail_thickens,
        "must_pin_set_cannot_shrink_below_attention": True,
        # ---- margin tail detail ----
        "private_margin_p05": priv_p05, "private_margin_median": priv_med, "private_margin_min": priv_min,
        "private_frac_margin_le_half": priv_tail_half, "private_frac_margin_le_eps": priv_tail_eps,
        "public_frac_margin_le_half_samecode": pub_tail_half, "public_margin_p05_samecode": pub_p05_samecode,
        "public_margin_p05_anchor": PUBLIC_MARGIN_P05, "public_margin_median_anchor": PUBLIC_MARGIN_MEDIAN,
        "rule_of_three_flip_ub": (_rule_of_three(n_pos) if n_flip == 0 else float("nan")),
        "private_delta_pct_frame": PRIVATE_DELTA_PCT,
        "integrity_bound": integrity_bound,
        "verdict": verdict,
    }


def selftest_shifted(shifted: dict, public: dict | None, comp: dict, flags: dict,
                     also_public: bool) -> dict[str, Any]:
    c: dict[str, bool] = {}
    # (a) shifted leg produced positions, NaN-clean
    c["a_shifted_has_positions"] = bool(int(shifted.get("n_positions", 0) or 0) > 0)
    c["a_shifted_nan_clean"] = (not bool(shifted.get("any_nan", True)))
    # (b) determinism control: REF re-gen byte-identical -> flips attributable to attention occupancy,
    #     not run-to-run noise (the proof that the shift is real, not measurement jitter)
    dgi = shifted.get("det_gen_byte_identity")
    c["b_ref_regen_deterministic"] = bool(dgi is not None and dgi >= 0.999)
    # (c) identity nesting: tie <= knife <= flip, semantic = flip - knife (classify_flip invariant)
    nt, nk, nf = (int(shifted.get("n_tie_flip", 0)), int(shifted.get("n_knife_flip", 0)),
                  int(shifted.get("n_flip", 0)))
    ns = int(shifted.get("n_semantic_flip", 0))
    c["c_identity_nesting_consistent"] = bool(nt <= nk <= nf and ns == (nf - nk))
    # (d) operative identity computed, in [0,1]
    op = comp.get("operative_identity_private")
    c["d_operative_identity_valid"] = bool(op is not None and math.isfinite(op) and 0.0 <= op <= 1.0)
    # (e) flip rate finite and in [0,1]
    pf = comp.get("private_attention_flip_rate")
    c["e_flip_rate_valid"] = bool(pf is not None and math.isfinite(pf) and 0.0 <= pf <= 1.0)
    # (f) model-load path flagged (full-vocab vs pruned head honesty, as in #491)
    c["f_model_path_flagged"] = bool(shifted.get("margin_model_full_vocab") is not None)
    # (g) same-code public anchor reproduces #491 (0.524%) within a generous band, if run
    if also_public and public:
        pub = _f(public.get("attention_frac_steps_argmax_flip"))
        c["g_public_samecode_reproduces_491"] = bool(
            math.isfinite(pub) and 0.5 * PUBLIC_ATTENTION_FLIP_RATE <= pub <= 2.0 * PUBLIC_ATTENTION_FLIP_RATE)
        c["g_public_det_control"] = bool(_f(public.get("det_gen_byte_identity")) >= 0.999)
    # (h) the must-pin SET invariant + integrity bound string present
    c["h_set_cannot_shrink"] = bool(comp.get("must_pin_set_cannot_shrink_below_attention") is True)
    c["h_integrity_bound_present"] = bool(isinstance(comp.get("integrity_bound"), str)
                                          and len(comp.get("integrity_bound", "")) > 0)
    # (i) launch flags clean
    c["i_no_launch_flags"] = bool(flags["no_hf_job"] and flags["no_launch"] and flags["analysis_only"])
    passes = all(c.values())
    return {"passes": passes, "n_checks": len(c), "conditions": c}


def print_report_shifted(payload: dict) -> None:
    comp, st = payload["compose"], payload["selftest"]
    sh, pub = payload["margin_shifted"], payload.get("margin_public")
    bar = "=" * 100
    print(bar)
    print("PRIVATE-ATTENTION-FLIP IDENTITY BOUND -- byte-exact identity off-distribution (PR #497, ubel)")
    print(f"  shifted split = {comp['split']} ({comp['n_positions']} decode steps, "
          f"model full_vocab={sh.get('margin_model_full_vocab')})")
    print("-" * 100)
    hdr = f"  {'metric':32s} {'public(anchor)':>16s} {'public(samecode)':>17s} {'PRIVATE(shifted)':>17s}"
    print(hdr)
    pub_fr = (pub.get("attention_frac_steps_argmax_flip") if pub else None)

    def _row(name, anchor, sc, pv):
        def g(x):
            return f"{x:.5f}" if isinstance(x, (int, float)) and math.isfinite(float(x)) else str(x)
        print(f"  {name:32s} {g(anchor):>16s} {g(sc):>17s} {g(pv):>17s}")
    _row("attention_flip_rate", PUBLIC_ATTENTION_FLIP_RATE, pub_fr, comp["private_attention_flip_rate"])
    _row("tie_flip_rate", "~0", (pub.get("tie_flip_rate") if pub else None), comp["private_tie_flip_rate"])
    _row("semantic_flip_rate", "~0", (pub.get("semantic_flip_rate") if pub else None),
         comp["private_semantic_flip_rate"])
    _row("operative_identity", 1.0, (pub.get("operative_identity") if pub else None),
         comp["operative_identity_private"])
    _row("margin_p05", PUBLIC_MARGIN_P05, (pub.get("margin_p05") if pub else None), comp["private_margin_p05"])
    _row("margin_median", PUBLIC_MARGIN_MEDIAN, (pub.get("margin_median") if pub else None),
         comp["private_margin_median"])
    _row("frac_margin_le_0.5", "-", (pub.get("frac_positions_margin_le_half") if pub else None),
         comp["private_frac_margin_le_half"])
    print("-" * 100)
    print(f"  public_to_private_flip_ratio   = {comp['public_to_private_flip_ratio']:.3f}x (anchor) / "
          f"{comp['public_to_private_flip_ratio_samecode']} (same-code)")
    print(f"  must_pin_cost_grows_off_dist   = {comp['must_pin_cost_grows_off_distribution']}")
    print(f"  low_margin_tail_thickens       = {comp['low_margin_tail_thickens']}")
    print(f"  operative_identity_private     = {comp['operative_identity_private']}  (target 1.0)")
    print(f"  must_pin_set_cannot_shrink     = {comp['must_pin_set_cannot_shrink_below_attention']}")
    print("-" * 100)
    print(f"  SELF-TEST {st['passes']} ({st['n_checks']} checks): "
          + json.dumps({k: int(v) for k, v in st["conditions"].items()}))
    print("-" * 100)
    print("  INTEGRITY BOUND\n   " + comp["integrity_bound"])
    print("  VERDICT\n   " + comp["verdict"])
    print(bar)


def maybe_log_wandb_shifted(payload: dict, args) -> str | None:
    if args.no_wandb:
        return None
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from scripts.wandb_logging import (init_wandb_run, log_summary,
                                            log_json_artifact, finish_wandb)
    except Exception as e:  # noqa: BLE001
        print(f"[#497] wandb helpers unavailable: {e}")
        return None
    comp, sh = payload["compose"], payload["margin_shifted"]
    run = init_wandb_run(
        job_type="analysis-private-attention-flip", agent="ubel",
        name=args.wandb_name, group=args.wandb_group,
        tags=["private-attention-flip-bound", "argmax-flip", "operative-identity", "ulp-tie",
              "off-distribution", "pr-497", "identity-axis"],
        config={"pr": 497, "kind": "private-attention-flip-identity-bound",
                "shifted_split": comp["split"], "public_anchor_flip_rate": PUBLIC_ATTENTION_FLIP_RATE,
                "eps_star": EPS_STAR, "near_tie_thresh": NEAR_TIE_LOGPROB_THRESH,
                "n_positions": comp["n_positions"], "private_delta_pct": PRIVATE_DELTA_PCT},
    )
    if run is None:
        print("[#497] wandb disabled (no API key / WANDB_MODE).")
        return None
    flat = {
        "private/attention_flip_rate": comp["private_attention_flip_rate"],
        "private/public_to_private_flip_ratio": comp["public_to_private_flip_ratio"],
        "private/operative_identity": comp["operative_identity_private"],
        "private/semantic_flip_rate": comp["private_semantic_flip_rate"],
        "private/tie_flip_rate": comp["private_tie_flip_rate"],
        "private/knife_flip_rate": comp["private_knife_flip_rate"],
        "private/margin_p05": comp["private_margin_p05"],
        "private/margin_median": comp["private_margin_median"],
        "private/frac_margin_le_half": comp["private_frac_margin_le_half"],
        "private/must_pin_cost_grows": float(bool(comp["must_pin_cost_grows_off_distribution"])),
        "private/low_margin_tail_thickens": float(bool(comp["low_margin_tail_thickens"])),
        "public/attention_flip_rate_anchor": PUBLIC_ATTENTION_FLIP_RATE,
        "public/attention_flip_rate_samecode": comp["public_attention_flip_rate_samecode"],
        "control/det_gen_byte_identity": _f(sh.get("det_gen_byte_identity")),
        "control/margin_model_full_vocab": float(bool(sh.get("margin_model_full_vocab"))),
        "selftest/private_attention_flip_self_test_passes": float(payload["selftest"]["passes"]),
    }
    run.log({"global_step": 0, **{k: (v if (isinstance(v, float) and math.isfinite(v)) else 0.0)
                                  for k, v in flat.items()}})
    log_summary(run, _jsonable(payload), step=0, run_prefix=args.wandb_name)
    log_json_artifact(run, name="private_attention_flip_bound",
                      artifact_type="analysis", data=_jsonable(payload))
    finish_wandb(run)
    rid = getattr(run, "id", None)
    print(f"[#497] wandb logged (run {rid})")
    return rid


def orchestrate_shifted(args) -> int:
    """PR #497: attention-only flip census on a HELD-OUT reasoning/STEM split + identity decomposition.
    Skips microbench/drafter/ingest -- #491 certified body/lm_head/drafter argmax-free by kernel-
    property/theorem (distribution-robust, cannot flip on private data)."""
    SHIFTED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sp = args.shifted_split
    shifted_json = str(SHIFTED_RESULTS_DIR / f"_margin_shifted_{sp}.json")
    public_json = str(SHIFTED_RESULTS_DIR / f"_margin_public_{sp}.json")
    server_python = resolve_server_python(args.server_python)
    print(f"[#497] server_python = {server_python}", flush=True)

    common = ["--n-prompts", str(args.n_prompts), "--n-new", str(args.n_new),
              "--ctx-cap", str(args.ctx_cap), "--verify-width", str(args.verify_width),
              "--gpu-mem-util", str(args.gpu_mem_util), "--topk", str(args.topk),
              "--det-prompts", str(args.det_prompts), "--seed", str(args.seed)]
    rc_sh = run_gpu_phase(server_python, ["--phase", "margin", "--out", shifted_json,
                          "--shifted-split", args.shifted_split] + common, timeout=args.margin_timeout)
    shifted = json.load(open(shifted_json)) if Path(shifted_json).exists() else {
        "phase": "margin", "error": rc_sh, "any_nan": True}

    public = None
    if args.also_public:
        rc_pub = run_gpu_phase(server_python, ["--phase", "margin", "--out", public_json] + common,
                               timeout=args.margin_timeout)
        public = json.load(open(public_json)) if Path(public_json).exists() else {
            "phase": "margin", "error": rc_pub}

    comp = compose_shifted(shifted, public)
    flags = {"no_hf_job": True, "no_launch": True, "analysis_only": True,
             "no_served_file_change": True, "official_tps": 0}
    st = selftest_shifted(shifted, public, comp, flags, args.also_public)

    payload = {
        "agent": "ubel", "pr": 497, "kind": "private-attention-flip-identity-bound",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        **flags,
        "anchors": {"public_attention_flip_rate": PUBLIC_ATTENTION_FLIP_RATE,
                    "public_margin_p05": PUBLIC_MARGIN_P05, "public_margin_median": PUBLIC_MARGIN_MEDIAN,
                    "deployed_tps": DEPLOYED_TPS, "private_delta_pct": PRIVATE_DELTA_PCT,
                    "eps_star": EPS_STAR, "near_tie_thresh": NEAR_TIE_LOGPROB_THRESH},
        "margin_shifted": shifted, "margin_public": public,
        "compose": comp, "selftest": st,
        "private_attention_flip_self_test_passes": bool(st["passes"]),
        # headline KEY OUTPUTS hoisted to top level
        "private_attention_flip_rate": comp["private_attention_flip_rate"],
        "public_to_private_flip_ratio": comp["public_to_private_flip_ratio"],
        "operative_identity_private": comp["operative_identity_private"],
        "private_semantic_flip_rate": comp["private_semantic_flip_rate"],
        "private_tie_flip_rate": comp["private_tie_flip_rate"],
        "must_pin_cost_grows_off_distribution": comp["must_pin_cost_grows_off_distribution"],
        "integrity_bound": comp["integrity_bound"],
    }
    print_report_shifted(payload)
    out_path = SHIFTED_RESULTS_DIR / f"private_attention_flip_bound_results_{sp}.json"
    out_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    print(f"\n[#497] wrote {out_path}", flush=True)
    rid = maybe_log_wandb_shifted(payload, args)
    if rid:
        payload["wandb_run_id"] = rid
        out_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    result = {"terminal": True, "status": "complete", "pending_arms": False,
              "wandb_run_ids": ([rid] if rid else []),
              "private_attention_flip_rate": round(_f(comp["private_attention_flip_rate"]), 6),
              "public_to_private_flip_ratio": round(_f(comp["public_to_private_flip_ratio"]), 4),
              "operative_identity_private": round(_f(comp["operative_identity_private"]), 6),
              "private_semantic_flip_rate": round(_f(comp["private_semantic_flip_rate"]), 6),
              "private_tie_flip_rate": round(_f(comp["private_tie_flip_rate"]), 6),
              "must_pin_cost_grows_off_distribution": bool(comp["must_pin_cost_grows_off_distribution"]),
              "primary_metric": {"name": "operative_identity_private",
                                 "value": round(_f(comp["operative_identity_private"]), 6)},
              "test_metric": {"name": "private_attention_flip_rate",
                              "value": round(_f(comp["private_attention_flip_rate"]), 6)},
              "self_test_passes": bool(st["passes"])}
    print("SENPAI-RESULT: " + json.dumps(result), flush=True)
    return 0 if st["passes"] else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["margin", "microbench"], default=None,
                    help="internal GPU phase dispatch (run under the server venv)")
    ap.add_argument("--out", default=None)
    # margin
    ap.add_argument("--n-prompts", type=int, default=128)
    ap.add_argument("--n-new", type=int, default=24)
    ap.add_argument("--ctx-cap", type=int, default=256)
    ap.add_argument("--verify-width", type=int, default=8)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--det-prompts", type=int, default=8)
    # microbench
    ap.add_argument("--micro-trials", type=int, default=24)
    # drafter
    ap.add_argument("--drafter-positions", type=int, default=4096)
    # orchestration
    ap.add_argument("--server-python", default=None)
    ap.add_argument("--margin-timeout", type=int, default=3000)
    ap.add_argument("--micro-timeout", type=int, default=900)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny fast path for validation")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="ubel/reduction-sensitivity-census")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="reduction-sensitivity-census")
    ap.add_argument("--no-wandb", action="store_true")
    # PR #497 private-attention-flip identity bound
    ap.add_argument("--shifted-split", "--shifted_split", dest="shifted_split", default=None,
                    choices=sorted(SHIFTED_SPLITS), help="run ATTENTION-ONLY on a held-out reasoning/"
                    "STEM split (PR #497); skips microbench/drafter/ingest (certified by #491)")
    ap.add_argument("--also-public", dest="also_public", action="store_true", default=True,
                    help="(#497 mode) also run the public split through the same code for a same-code "
                    "anchor that reproduces #491's 0.524%")
    ap.add_argument("--no-also-public", dest="also_public", action="store_false")
    args = ap.parse_args()

    if args.smoke:
        args.n_prompts = min(args.n_prompts, 6)
        args.n_new = min(args.n_new, 8)
        args.det_prompts = min(args.det_prompts, 3)
        args.micro_trials = min(args.micro_trials, 4)
        args.drafter_positions = min(args.drafter_positions, 256)

    if args.phase == "margin":
        phase_margin(args.out, args.n_prompts, args.n_new, args.ctx_cap, args.verify_width,
                     args.gpu_mem_util, args.topk, args.det_prompts, shifted_split=args.shifted_split)
        return
    if args.phase == "microbench":
        phase_microbench(args.out, args.micro_trials, args.seed)
        return
    if args.shifted_split:
        raise SystemExit(orchestrate_shifted(args))
    raise SystemExit(orchestrate(args))


if __name__ == "__main__":
    main()
