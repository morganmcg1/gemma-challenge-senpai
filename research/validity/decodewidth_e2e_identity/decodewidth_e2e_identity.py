#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #381 (stark) -- Decode-width e2e token-identity: is the int4-Marlin residual GONE at the
literal 8-row spec-verify width (M=8 query rows against a REAL paged KV cache), so the pinned
identity reaches 1.0?

WHY THIS IS THE DECISIVE FOLLOW-UP TO #376
------------------------------------------
My #376 (`ipe3ofie`, merged) measured the M=8-verify-vs-M=1-AR e2e token identity on the deployed
int4 stack and returned RED: pinning the attention split (VLLM_BATCH_INVARIANT -> num_splits=1) left
identity at 0.992555 (== heuristic 0.992708). BUT #376's "M=8" was *prefill-replication* geometry --
8 identical co-batched prefill replicas, so the body GEMM M-dim was 8*seq_len = 2048. My own Marlin
size_m sweep in #376 showed the int4-Marlin body GEMM is M-VARIANT at size_m>=64 (first-divergent
64; bit-exact at 1 and 8). So #376's RED is an artifact of the prefill-replication width pushing
Marlin to size_m=2048 -- NOT necessarily the deployed decode-verify geometry.

The DEPLOYED EAGLE-3 verify step runs **8 query rows (K_spec=7 + 1 bonus) against the cached KV** --
a *decode* width, body GEMM size_m = 8, where my #376 sweep says Marlin is BIT-EXACT. At that literal
width, with attention (num_splits=1) + the tied bf16 lm_head + the aten matmuls all batch-invariant
in the pinned arm, every M-variant op is either bit-exact (Marlin@8) or pinned -> pinned identity
should reach 1.0. This card MEASURES that directly instead of arguing it by composition.

THE FAITHFUL DECODE-WIDTH GEOMETRY (the load-bearing realization)
----------------------------------------------------------------
The clean way to get M=8 query rows against a REAL paged KV cache through the served stack, with NO
spec-trajectory branching (the #114/#122 56% artifact), and using only vLLM's high-level API:

  Step A (M=1 AR reference):  generate(prefix[0:C], max_tokens=8, temperature=0) -- prefills the
      C-token context (writes prefix-cache blocks 0..C/16-1) AND greedily extends it by 8 tokens.
      Each of those 8 tokens is the argmax of a M=1 DECODE step (single sequence), so the
      continuation IS the per-position M=1 AR greedy prediction. No teacher-forcing needed: the
      greedy continuation is, by definition, the M=1 path.

  Step B (M=8 verify chunk):  full = prefix + continuation (len C+8). generate(full,
      prompt_logprobs=1, max_tokens=1, temperature=0, skip_reading_prefix_cache=False).
      With enable_prefix_caching=True the [0:C] prefix (C a multiple of block_size=16) is a CACHE
      HIT, so vLLM computes ONLY the 8 uncached suffix tokens -- as ONE forward of 8 query rows
      attending to the cached KV. Body GEMM size_m = 8 (decode width); attention = TRITON_ATTN
      varlen paged-KV (the #375 served reality). prompt_logprobs at the suffix positions are the
      M=8-width argmax predictions.

  CRITICAL vLLM 0.22.0 gotcha (resolved by reading the installed source): SamplingParams auto-sets
  `skip_reading_prefix_cache = (prompt_logprobs is not None)` -> requesting prompt_logprobs SILENTLY
  DISABLES prefix-cache reuse, recomputing the FULL C+8 prompt as one prefill (size_m = C+8). That
  would reproduce #376's prefill-geometry RED. We MUST pass skip_reading_prefix_cache=False to force
  the real M=8 chunk. The harness asserts the chunk isolated (non-null prompt_logprobs confined to
  the suffix) as a primary geometry control.

THE COMPARISON (no branching):
  identity at position C+1+j  <=>  argmax(prompt_logprobs[C+1+j])  ==  continuation[1+j]
  i.e. M=8-chunk argmax == M=1-greedy token, query row C+j (reading the same token, same context) at
  width 8 vs width 1. Both walk the SAME token path (the greedy continuation), so a flip is purely
  the kernel M-variance, never trajectory divergence. Usable positions: C+1..C+7 (the chunk's query
  rows C..C+6 land in prompt_logprobs; the last row C+7 predicts the stripped first-decode token).

TWO ARMS, identical except the pin (same as #376):
  heuristic -- stock vLLM (VLLM_BATCH_INVARIANT=0). The verify-width attention split heuristic differs
               between M=8 and M=1 -> the divergence to eliminate.
  pinned    -- VLLM_BATCH_INVARIANT=1. attention num_splits=1 (single-segment, M-independent) + aten
               matmul / lm_head / norm batch-invariant. Marlin is NOT reached by the override but is
               bit-exact at size_m=8 -> pinned identity should be 1.0 at decode width.

DECIDER: pinned_reaches_identity_1p0 -- does pinned decode-width identity hit 1.0 (byte-tie band)?
  GREEN -> the eta-axis identity factor is env-reachable on the served DECODE path (only #375's
           mha_varlen attention rebuild remains, NOT the Marlin body-GEMM rebuild); fern #357 banks
           identity=1.0 as attention-rebuild-gated, and we do NOT flag the human on a Marlin rebuild.
  RED   -> pinned < 1.0 with the residual localised to Marlin -> the fixed-split-k Marlin rebuild is
           binding even on the served decode path -> #376's prefill RED carries to the deployed
           geometry -> second kernel-rebuild line item is real.

SCOPE: LOCAL A10G probe under #319 (judicious-use). 0 HF Job / 0 submission / 0 served-file change /
0 official TPS draw / no train.py --launch. The served int4 path is READ, never modified. Each arm
runs as an isolated subprocess so the pinned arm's process-wide batch-invariant override never leaks
into the heuristic arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------------------
# Imported fleet anchors (DO NOT re-derive -- import, do not re-measure)
# --------------------------------------------------------------------------------------
INT4_IDENTITY_232 = 0.9927083333333333   # #232 nxwv6pam clean deployed M1-vs-M8 identity (prefill-repl)
PREFILL_REPL_PINNED_376 = 0.9925551470588235  # #376 ipe3ofie pinned identity at prefill-repl width (RED)
PREFILL_REPL_HEUR_376 = 0.9927083333333333    # #376 heuristic identity at prefill-repl width
MARLIN_FIRST_DIVERGENT_SIZE_M_376 = 64    # #376 Marlin size_m sweep: first divergent size_m
MARLIN_BITEXACT_AT_8_376 = True           # #376 Marlin bit-exact at the decode width (size_m=8)
DEPLOYED_FLIP_362 = 0.0052                # #362 5k3px8p1 deployed M8-verify-vs-M1-AR flip rate
SERVED_ATTN_375 = "TRITON_ATTN varlen paged-KV decode; M-invariant served config = VLLM_BATCH_INVARIANT=1 (num_splits=1)"

OFFICIAL_BASELINE = 481.53                # #52 official TPS (this leg adds 0)
K_SPEC = 7                               # num_speculative_tokens (manifest)
M_VERIFY = K_SPEC + 1                    # = 8, the deployed decode-verify query width
IDENTITY_EPS = 1e-12                     # pinned_identity >= 1 - eps treated as "== 1.0" (GREEN)
# A token flip whose top-1-vs-M1 logprob margin is below this is a NUMERICAL near-tie: the M=8 and
# M=1 distributions agree on the ordering except at a knife-edge where two tokens are within e^0.5
# ~= 1.65x probability, so a sub-ULP kernel perturbation flips the argmax. Above it, a flip would be
# a genuine systematic op-level divergence. Used to CHARACTERISE a residual, not to set the verdict.
NEAR_TIE_LOGPROB_THRESH = 0.5
BLOCK_SIZE = 16                          # vLLM prefix-cache block granularity (config/cache.py:45)
# Gemma-4 is HYBRID-attention (5 sliding_attention : 1 full_attention over 42 layers, window=512).
# The hybrid KV-cache prefix hit is the INTERSECTION across the two cache groups and EMPIRICALLY
# commits in 32-token (=2*BLOCK_SIZE) units: a block-aligned prefix that is an ODD multiple of 16
# (e.g. C=240) caps the hit one block short (nct=224) -> the verify chunk recomputes 16+8=24 rows
# (size_m=24, a 2-Marlin-block tile), NOT the literal decode width. Aligning the prefix DOWN to a
# multiple of 32 makes the whole prefix a cache hit -> exactly n_verify rows computed -> size_m=8
# (1 Marlin block), the PR-required geometry. Measured via the C-sweep: n_computed==8 iff C%32==0.
HYBRID_PREFIX_COMMIT = 32                # Gemma-4 hybrid prefix-cache commit granularity (measured)

DEFAULT_PROXY = "google/gemma-4-E4B-it-qat-w4a16-ct"
MODEL_CANDIDATES = [
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots"
    ),
    "/tmp/osoi5-v0-baked",
]
PROMPTS_JSONL = "official/main_bucket/shared_resources/speed_benchmark/data/ppl_ground_truth_tokens.jsonl"

OUT_DIR = Path("research/validity/decodewidth_e2e_identity")
ARMS = ("heuristic", "pinned")


# --------------------------------------------------------------------------------------
# Small helpers (resolve_model_dir / read_text_dims reused from #376/#232)
# --------------------------------------------------------------------------------------
def resolve_model_dir() -> str:
    for cand in MODEL_CANDIDATES:
        p = Path(cand)
        if p.is_dir() and (p / "config.json").exists():
            return str(p)
        if p.is_dir():
            for sub in sorted(p.glob("*")):
                if (sub / "config.json").exists():
                    return str(sub)
    raise FileNotFoundError(f"no int4 model found among {MODEL_CANDIDATES}")


def read_text_dims(model_dir: str) -> dict:
    cfg = json.load(open(Path(model_dir) / "config.json"))
    tc = cfg.get("text_config", cfg)
    h = tc["hidden_size"]
    n_heads = tc["num_attention_heads"]
    n_kv = tc["num_key_value_heads"]
    hd = tc["head_dim"]
    inter = tc["intermediate_size"]
    return {
        "hidden": h, "n_heads": n_heads, "n_kv": n_kv, "head_dim": hd,
        "intermediate": inter, "num_layers": tc.get("num_hidden_layers"),
        "shapes": {
            "qkv_proj": ((n_heads + 2 * n_kv) * hd, h),
            "o_proj": (h, n_heads * hd),
            "gate_up_proj": (2 * inter, h),
            "down_proj": (h, inter),
        },
    }


def block_align(n: int) -> int:
    # Align DOWN to the hybrid prefix-commit granularity (32) so the whole prefix is a cache hit and
    # the verify chunk computes EXACTLY n_verify rows (size_m=8). A bare BLOCK_SIZE (16) alignment is
    # insufficient: odd multiples of 16 cap the hybrid hit one block short -> size_m=24 (see csweep).
    return (n // HYBRID_PREFIX_COMMIT) * HYBRID_PREFIX_COMMIT


# ======================================================================================
# PHASE: one arm. The pin (VLLM_BATCH_INVARIANT) is set in the subprocess ENV by the
# orchestrator for the ``pinned`` arm; this phase just reads it.
# ======================================================================================
def _argmax_from_logprob_entry(entry) -> int:
    return int(max(entry.items(), key=lambda kv: getattr(kv[1], "logprob", kv[1]))[0])


def _sorted_logprobs(entry) -> list[tuple[int, float]]:
    """(token_id, logprob) pairs sorted by logprob descending (rank 0 first)."""
    return sorted(((int(t), float(getattr(lp, "logprob", lp))) for t, lp in entry.items()),
                  key=lambda kv: kv[1], reverse=True)


def phase_arm(out_path: str, arm: str, n_prompts: int, ctx_len: int, n_verify: int,
              gpu_mem_util: float, max_batched_tokens: int, verbose_k: int) -> None:
    import torch
    from vllm import LLM, SamplingParams

    batch_invariant_env = os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1"
    model_dir = resolve_model_dir()
    dims = read_text_dims(model_dir)
    C = block_align(ctx_len)  # prefix length, a multiple of 32 so the WHOLE prefix is a cache hit
                              # -> verify chunk computes exactly n_verify rows (size_m=8, 1 Marlin block)
    print(f"[arm:{arm}] model={model_dir} layers={dims['num_layers']} hidden={dims['hidden']} "
          f"C(prefix)={C} n_verify={n_verify} VLLM_BATCH_INVARIANT={batch_invariant_env}", flush=True)

    t0 = time.time()
    # enable_prefix_caching=True so step B's [0:C] prefix is a cache hit and ONLY the 8-token suffix
    # is computed (the M=8 chunk). enforce_eager=True so no CUDA-graph bucket padding changes the
    # chunk width. max_num_seqs small (single-sequence verify occupancy, the regime where the split
    # heuristic becomes M-dependent).
    llm = LLM(
        model=model_dir,
        quantization="compressed-tensors",
        dtype="bfloat16",
        max_model_len=max(512, C + 64),
        gpu_memory_utilization=gpu_mem_util,
        max_num_seqs=16,
        max_num_batched_tokens=max_batched_tokens,
        enable_prefix_caching=True,
        enforce_eager=True,
        trust_remote_code=True,
    )
    print(f"[arm:{arm}] vLLM load done in {time.time()-t0:.0f}s", flush=True)

    try:
        import vllm.v1.attention.ops.triton_unified_attention as _ua
        attn_is_batch_invariant = bool(getattr(_ua, "is_batch_invariant", False))
    except Exception:
        attn_is_batch_invariant = False

    # Step A: greedy continuation (M=1 AR). Step B: M=8 verify chunk via prompt_logprobs with the
    # prefix-cache override that vLLM 0.22.0 otherwise auto-disables.
    # detokenize=False is LOAD-BEARING: with the prefix cache-hit, vLLM allocates the prompt_logprobs
    # tensor for the WHOLE prompt (torch.empty) but only fills the computed suffix; the cached-prefix
    # rows stay UNINITIALIZED garbage. Detokenizing those garbage token_ids raises OverflowError
    # ("out of range integral type conversion"). detokenize=False routes the LogprobsProcessor's
    # tokenizer to None (output_processor.py: `if not sampling_params.detokenize: tokenizer = None`),
    # so no detokenization is attempted and we read raw token_ids only. The engine tokenizer stays
    # alive (needed for the multimodal gemma-4 input preprocessor), so we do NOT use skip_tokenizer_init.
    sp_gen = SamplingParams(temperature=0.0, max_tokens=n_verify, detokenize=False)
    sp_warm = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False)
    # prompt_logprobs=5 (not 1): we read the argmax AND the top-2 gap + the M=1 token's rank/logprob
    # to CHARACTERISE any residual flip as a knife-edge near-tie vs a systematic op divergence.
    sp_chunk = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=5,
                              skip_reading_prefix_cache=False, detokenize=False)

    rows = [json.loads(l) for l in open(PROMPTS_JSONL)][:n_prompts]
    per_prompt = []
    n_match = n_total = 0
    n_det_m1 = n_det_m8 = n_within = 0
    n_chunk_isolated = 0          # geometry control: num_cached_tokens==C (exactly n_verify rows computed)
    chunk_width_obs = []          # observed readable chunk width per prompt (expect n_verify-1)
    n_computed_rows_total = 0     # sum of computed query rows per chunk (expect n_verify each)
    # near-tie characterisation of the residual (M=8 chunk top-1 vs M=1 token at flips):
    all_div_gaps = []             # top1-vs-top2 logprob gap at every DIVERGENT position
    all_div_margins = []          # logprob(M8 argmax) - logprob(M1 token) at flips (None if M1 outside top-5)
    all_min_gap_per_prompt = []   # min top1-top2 gap over a prompt's suffix (typical-gap context)

    for ri, rec in enumerate(rows):
        src = list(rec.get("context_token_ids", [])) + list(rec.get("target_token_ids", []))
        if len(src) < C + 1:
            continue
        prefix = src[:C]

        # ---- WARM the prefix cache first (served-faithful + control-clean) ----
        # In real spec-decode serving the context is ALREADY in the paged KV cache when the verify
        # step runs; the M=8 verify AND the M=1 reference both read that warm cache. We replicate that
        # by committing the prefix once here, so Step A (M=1 AR ref), its det control, and the M=8
        # chunk all operate on the SAME cached prefix KV. Without this, run-1 prefills the prefix as a
        # size_m=C GEMM while run-2 reads it warm (last token recomputed at size_m=1); those two paths
        # produce a hair-different last-token KV (the #376 Marlin M-variance) that propagates through
        # the AR loop and flips a downstream near-tie -> a spurious det_m1<1.0 (diagnosed: two WARM AR
        # runs are always bit-identical; only cold-vs-warm diverges, at a downstream index not 0).
        # Warming makes det_m1 measure genuine M=1 AR reproducibility and matches the served geometry.
        llm.generate([{"prompt_token_ids": prefix}], sp_warm, use_tqdm=False)

        # ---- Step A (M=1 AR greedy continuation) + M1-vs-M1 determinism control (both warm) ----
        outA = llm.generate([{"prompt_token_ids": prefix}], sp_gen, use_tqdm=False)[0]
        cont = list(outA.outputs[0].token_ids)[:n_verify]
        outA2 = llm.generate([{"prompt_token_ids": prefix}], sp_gen, use_tqdm=False)[0]
        cont2 = list(outA2.outputs[0].token_ids)[:n_verify]
        det_m1 = int(cont == cont2)
        if len(cont) < n_verify:
            continue
        full = prefix + cont  # length C + n_verify; positions 0..C+n_verify-1

        # ---- Step B (M=8 verify chunk) + M8-vs-M8 determinism control ----
        # GEOMETRY: with the C-token prefix a cache HIT, vLLM computes ONLY the n_verify suffix rows
        # (positions C..C+n_verify-1) -> body GEMM size_m = n_verify (the decode-verify width). The
        # AUTHORITATIVE isolation signal is RequestOutput.num_cached_tokens == C (==> exactly n_verify
        # rows computed). The prompt_logprobs list still spans the whole prompt, but its entries at
        # positions <=C are uninitialized torch.empty GARBAGE (never computed); we read ONLY the real
        # suffix positions C+1..C+n_verify-1 (rows C..C+n_verify-2; the last row predicts the stripped
        # first-decode token, which is not a prompt-logprob position).
        def chunk_argmax(full_ids):
            out = llm.generate([{"prompt_token_ids": full_ids}], sp_chunk, use_tqdm=False)[0]
            nct = out.num_cached_tokens or 0
            pls = out.prompt_logprobs or []
            am = {}   # absolute position -> argmax token (REAL computed suffix only)
            ent = {}  # absolute position -> [(token_id, logprob), ...] sorted rank 0 first
            for i in range(C + 1, len(full_ids)):
                entry = pls[i] if i < len(pls) else None
                if entry is not None:
                    am[i] = _argmax_from_logprob_entry(entry)
                    ent[i] = _sorted_logprobs(entry)
            return am, nct, ent

        m8, nct8, ent8 = chunk_argmax(full)
        m8b, nct8b, _ = chunk_argmax(full)

        # geometry: cache-hit isolates the chunk to exactly n_verify computed rows (size_m=n_verify).
        # The ROBUST size_m=8 invariant is n_computed_rows == n_verify in BOTH chunk calls (equivalent
        # to nct==C, but stated as the body-GEMM width the PR decides on). A hybrid-cache short-commit
        # (size_m=24) fails this and is excluded from "faithful".
        suffix_pos = sorted(m8)
        n_computed_rows = len(full) - nct8
        n_computed_rows_b = len(full) - nct8b
        chunk_isolated = (n_computed_rows == n_verify and n_computed_rows_b == n_verify)
        n_computed_rows_total += n_computed_rows
        chunk_width_obs.append(len(suffix_pos))
        n_chunk_isolated += int(chunk_isolated)

        # det M8: same readable positions, same argmax
        det_m8 = int(all(m8.get(p) == m8b.get(p) for p in suffix_pos) and bool(suffix_pos))

        # ---- within-batch copy0 vs copy1 control (two identical chunks co-batched) ----
        outW = llm.generate([{"prompt_token_ids": full}, {"prompt_token_ids": full}],
                            sp_chunk, use_tqdm=False)
        def _am(out):
            d = {}
            pls = out.prompt_logprobs or []
            for i in range(C + 1, len(full)):
                entry = pls[i] if i < len(pls) else None
                if entry is not None:
                    d[i] = _argmax_from_logprob_entry(entry)
            return d
        w0, w1 = _am(outW[0]), _am(outW[1])
        within = int(bool(w0) and all(w0.get(p) == w1.get(p) for p in suffix_pos))

        # ---- the signal: M=8 chunk argmax vs M=1 greedy token, position by position ----
        # Also capture, at each flip, HOW CLOSE the call was: the M=8 distribution's top1-top2 gap and
        # the margin by which its argmax beat the M=1 token. A residual built only of sub-0.5-nat
        # margins is a knife-edge near-tie (a kernel-perturbation coin-flip), not a systematic op bug.
        match = total = 0
        prompt_min_gap = float("inf")
        prompt_div_gaps = []
        for p in suffix_pos:
            m1_tok = full[p]  # the M=1 greedy continuation token at position p
            total += 1
            sl = ent8.get(p, [])
            gap = (sl[0][1] - sl[1][1]) if len(sl) >= 2 else float("inf")  # top1-top2 logprob gap
            prompt_min_gap = min(prompt_min_gap, gap)
            if m8.get(p) == m1_tok:
                match += 1
            else:
                prompt_div_gaps.append(gap)
                all_div_gaps.append(gap)
                lp_map = dict(sl)
                top1_lp = sl[0][1] if sl else float("nan")
                all_div_margins.append((top1_lp - lp_map[m1_tok]) if m1_tok in lp_map else None)
        if math.isfinite(prompt_min_gap):
            all_min_gap_per_prompt.append(prompt_min_gap)

        n_match += match
        n_total += total
        n_det_m1 += det_m1 * max(1, total)
        n_det_m8 += det_m8 * max(1, total)
        n_within += within * max(1, total)

        sha = hashlib.sha256(bytes(str([m8.get(p) for p in suffix_pos]), "utf8")).hexdigest()[:16]
        per_prompt.append({
            "id": rec.get("id"), "C": C, "chunk_width": len(suffix_pos),
            "chunk_isolated": chunk_isolated, "num_cached_tokens": nct8,
            "n_computed_rows": n_computed_rows,
            "argmax_match_M8_vs_M1": match, "positions": total, "sha": sha,
            "det_match_M1_vs_M1": det_m1, "det_match_M8_vs_M8": det_m8,
            "within_copy0_vs_copy1": within,
            "min_top2_gap": (prompt_min_gap if math.isfinite(prompt_min_gap) else None),
            "divergent_top2_gaps": [round(g, 5) for g in prompt_div_gaps if math.isfinite(g)],
        })
        if ri < verbose_k or ri == len(rows) - 1:
            print(f"[arm:{arm}] prompt {ri} id={rec.get('id')} chunk_w={len(suffix_pos)} "
                  f"isolated={chunk_isolated} match={match}/{total} det_m1={det_m1} det_m8={det_m8} "
                  f"within={within} suffix_pos={suffix_pos[:3]}..{suffix_pos[-1:] if suffix_pos else []}",
                  flush=True)

    n_seq = len(per_prompt)
    identity = (n_match / n_total) if n_total else float("nan")
    det_m1_frac = (n_det_m1 / n_total) if n_total else float("nan")
    det_m8_frac = (n_det_m8 / n_total) if n_total else float("nan")
    within_frac = (n_within / n_total) if n_total else float("nan")
    chunk_isolated_frac = (n_chunk_isolated / n_seq) if n_seq else float("nan")
    median_chunk_width = (statistics.median(chunk_width_obs) if chunk_width_obs else float("nan"))

    # ---- near-tie characterisation of the residual ----
    margins_present = [m for m in all_div_margins if m is not None]
    all_margins_in_top5 = (len(margins_present) == len(all_div_margins))
    # knife-edge IFF there ARE flips, every flipped M1 token sat in the M=8 top-5, and the WORST flip
    # margin is still below the near-tie threshold (every flip is a sub-0.5-nat coin-flip). None if no
    # flips (identity 1.0 -> nothing to characterise).
    if not all_div_margins:
        residual_is_knife_edge = None
    else:
        residual_is_knife_edge = bool(all_margins_in_top5 and margins_present
                                      and max(margins_present) < NEAR_TIE_LOGPROB_THRESH)
    near_tie = {
        "divergent_count": len(all_div_gaps),
        "near_tie_logprob_thresh": NEAR_TIE_LOGPROB_THRESH,
        "residual_is_knife_edge_near_tie": residual_is_knife_edge,
        "n_divergent_m1_in_top5": len(margins_present),
        "all_divergent_m1_in_top5": all_margins_in_top5,
        "gap_top2_median_divergent": (statistics.median(all_div_gaps) if all_div_gaps else None),
        "gap_top2_max_divergent": (max(all_div_gaps) if all_div_gaps else None),
        "margin_vs_m1_median_divergent": (statistics.median(margins_present) if margins_present else None),
        "margin_vs_m1_max_divergent": (max(margins_present) if margins_present else None),
        "min_gap_all_positions_median": (statistics.median(all_min_gap_per_prompt)
                                         if all_min_gap_per_prompt else None),
        "divergent_gaps": [round(g, 5) for g in all_div_gaps if math.isfinite(g)],
        "divergent_margins": [round(m, 5) if m is not None else None for m in all_div_margins],
    }

    # ---- pin-engaged positive control: aten torch.mm row-0 bit-exactness M=1 vs M=8 ----
    aten_ctrl = aten_mm_invariance_control(torch, dims["hidden"], M_VERIFY)

    # ---- Marlin size_m diag: re-confirm the int4 body GEMM is bit-exact at the DECODE width (8) ----
    try:
        marlin_diag = marlin_sizem_diag(llm, dims, torch, M_VERIFY)
    except Exception as exc:
        marlin_diag = {"status": f"failed: {exc!r}", "per_size": {},
                       "first_divergent_size_m": None,
                       "bitexact_at_decode_width": None}
        print(f"[arm:{arm}] marlin diag unavailable -> {marlin_diag['status']}", flush=True)

    nan_clean = all(math.isfinite(x) for x in (identity, det_m1_frac, det_m8_frac, within_frac))
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    out = {
        "phase": "arm", "arm": arm, "model_dir": model_dir,
        "vllm_batch_invariant_env": batch_invariant_env,
        "attn_is_batch_invariant": attn_is_batch_invariant,
        "n_prompts": n_seq, "ctx_len_requested": ctx_len, "C": C, "n_verify": n_verify,
        "total_positions": n_total, "matching_positions": n_match,
        "decodewidth_e2e_token_identity_rate": identity,
        "decodewidth_e2e_divergence_rate": (1.0 - identity) if math.isfinite(identity) else float("nan"),
        "determinism_M1_vs_M1": det_m1_frac,
        "determinism_M8_vs_M8": det_m8_frac,
        "within_batch_copy0_vs_copy1": within_frac,
        "chunk_isolated_fraction": chunk_isolated_frac,
        "median_chunk_width": median_chunk_width,
        "n_computed_rows_total": n_computed_rows_total,
        "expected_computed_rows_per_chunk": n_verify,
        "near_tie": near_tie,
        "aten_mm_control": aten_ctrl,
        "marlin_sizem_diag": marlin_diag,
        "nan_clean": bool(nan_clean), "peak_gpu_gb": peak_gb,
        "per_prompt": per_prompt,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[arm:{arm}] decodewidth identity={identity:.6f} (divergence={1.0-identity:.6f}) "
          f"chunk_isolated={chunk_isolated_frac:.4f} median_chunk_w={median_chunk_width} peak={peak_gb:.1f}GB",
          flush=True)
    print(f"[arm:{arm}] controls: det_m1={det_m1_frac:.6f} det_m8={det_m8_frac:.6f} "
          f"within={within_frac:.6f} | pin_engaged(aten_mm bitexact)={aten_ctrl.get('bitexact_M1_vs_M8')} "
          f"attn_batch_invariant={attn_is_batch_invariant}", flush=True)
    print(f"[arm:{arm}] marlin bitexact@decode(size_m=8)={marlin_diag.get('bitexact_at_decode_width')} "
          f"first_divergent_size_m={marlin_diag.get('first_divergent_size_m')}", flush=True)
    print(f"[arm:{arm}] near-tie: {near_tie['divergent_count']} flips, knife_edge={near_tie['residual_is_knife_edge_near_tie']} "
          f"margin_max={near_tie['margin_vs_m1_max_divergent']} gap_max={near_tie['gap_top2_max_divergent']} "
          f"(thresh={NEAR_TIE_LOGPROB_THRESH}, m1_in_top5={near_tie['n_divergent_m1_in_top5']}/{near_tie['divergent_count']})", flush=True)
    print(f"ARM_DONE {out_path}", flush=True)


def aten_mm_invariance_control(torch, hidden: int, batch_m: int) -> dict:
    """torch.mm row-0 bit-exactness at M=1 vs M=batch_m -- proves the batch-invariant override (pin)
    is live in this process. Pure aten op (NOT Marlin)."""
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    n = hidden
    w = torch.randn(n, n, dtype=torch.bfloat16, device=dev)
    x = torch.randn(max(batch_m, 16), n, dtype=torch.bfloat16, device=dev)
    y1 = torch.mm(x[:1].contiguous(), w)
    ym = torch.mm(x[:batch_m].contiguous(), w)
    torch.cuda.synchronize()
    bitexact = bool(torch.equal(ym[:1].float(), y1.float()))
    return {
        "bitexact_M1_vs_M8": bitexact,
        "max_abs_diff_M1_vs_M8": float((ym[:1].float() - y1.float()).abs().max()),
        "batch_m": batch_m,
    }


def marlin_sizem_diag(llm, dims: dict, torch, decode_width: int) -> dict:
    """Row-0 bit-exactness of the int4-Marlin body GEMMs across size_m, re-confirming in-process that
    the body GEMM is bit-exact at the DECODE-VERIFY width (size_m=decode_width=8). Reuses #376/#232's
    model navigation. VLLM_BATCH_INVARIANT does NOT patch this custom CUDA op, so the result is the
    same in both arms (reported in both as a cross-check)."""
    import torch.nn as nn

    dev = torch.device("cuda:0")
    shapes = dims["shapes"]

    def get_model():
        paths = [
            lambda: llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model,
            lambda: llm.llm_engine.model_executor.driver_worker.worker.model_runner.model,
            lambda: llm.llm_engine.model_executor.driver_worker.model_runner.model,
        ]
        for p in paths:
            try:
                m = p()
                if m is not None:
                    return m
            except Exception:
                continue
        raise RuntimeError("could not locate model_runner.model")

    def find_layers(root):
        chains = [("model", "layers"), ("model", "language_model", "layers"),
                  ("language_model", "model", "layers"), ("language_model", "layers"),
                  ("model", "model", "layers"), ("layers",)]
        for chain in chains:
            obj = root
            ok = True
            for attr in chain:
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    ok = False
                    break
            if ok and isinstance(obj, (nn.ModuleList, list)) and len(obj) > 0:
                return obj
        for _, mod in root.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) > 0:
                el = mod[0]
                if hasattr(el, "self_attn") and hasattr(el.self_attn, "qkv_proj"):
                    return mod
        raise RuntimeError("could not locate decoder ModuleList")

    def module_out_in(mod):
        out = getattr(mod, "output_size_per_partition", None)
        inp = getattr(mod, "input_size_per_partition", None)
        if out is None or inp is None:
            w = getattr(mod, "weight", None)
            if w is not None and w.dim() == 2:
                out, inp = int(w.shape[0]), int(w.shape[1])
        return (int(out), int(inp)) if out and inp else None

    layers = find_layers(get_model())
    targets = None
    for layer in layers:
        try:
            cand = {
                "qkv_proj": layer.self_attn.qkv_proj,
                "o_proj": layer.self_attn.o_proj,
                "gate_up_proj": layer.mlp.gate_up_proj,
                "down_proj": layer.mlp.down_proj,
            }
        except AttributeError:
            continue
        if all(hasattr(m, "quant_method") and module_out_in(m) == shapes[name]
               for name, m in cand.items()):
            targets = cand
            break
    if targets is None:
        raise RuntimeError("no layer matched canonical body shapes")

    sizes = sorted({1, decode_width, 16, 64})
    torch.manual_seed(0)
    per_size = {}
    first_divergent = None
    for sm in sizes:
        all_bitexact = True
        max_diff = 0.0
        for name, (out, inp) in shapes.items():
            x = torch.randn(max(sm, 1), inp, dtype=torch.bfloat16, device=dev)
            apply_fn = lambda t, _m=targets[name]: _m.quant_method.apply(_m, t, bias=None)
            y1 = apply_fn(x[:1].contiguous())[0].detach().float()
            ym = apply_fn(x[:sm].contiguous())[0].detach().float()
            torch.cuda.synchronize()
            be = bool(torch.equal(ym, y1))
            md = float((ym - y1).abs().max())
            all_bitexact = all_bitexact and be
            max_diff = max(max_diff, md)
        per_size[str(sm)] = {"bitexact_row0_vs_M1": all_bitexact, "max_abs_diff": max_diff}
        if not all_bitexact and first_divergent is None and sm > 1:
            first_divergent = sm

    return {
        "status": "ran",
        "sizes_tested": sizes,
        "per_size": per_size,
        "first_divergent_size_m": first_divergent,
        "bitexact_at_decode_width": per_size.get(str(decode_width), {}).get("bitexact_row0_vs_M1"),
        "decode_width": decode_width,
    }


# ======================================================================================
# Orchestrator: two isolated subprocess arms, compose, self-test, wandb
# ======================================================================================
def run_phase_subprocess(args_list: list[str], extra_env: dict | None = None) -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")  # Gemma4 -> vLLM overrides to TRITON_ATTN
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if extra_env:
        env.update(extra_env)
    cmd = [sys.executable, os.path.abspath(__file__)] + args_list
    print(f"[orch] launching: {' '.join(args_list)} "
          f"(VLLM_BATCH_INVARIANT={env.get('VLLM_BATCH_INVARIANT', '0')})", flush=True)
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        raise RuntimeError(f"phase subprocess failed (rc={rc}): {args_list}")


def _run_arm(a: argparse.Namespace, arm: str) -> dict:
    out_json = str(OUT_DIR / f"arm_{arm}_result.json")
    extra_env = {"VLLM_BATCH_INVARIANT": "1"} if arm == "pinned" else {"VLLM_BATCH_INVARIANT": "0"}
    run_phase_subprocess([
        "--phase", "arm", "--arm", arm, "--out", out_json,
        "--n-prompts", str(a.n_prompts), "--ctx-len", str(a.ctx_len),
        "--n-verify", str(a.n_verify), "--gpu-mem-util", str(a.gpu_mem_util),
        "--max-batched-tokens", str(a.max_batched_tokens), "--verbose-k", str(a.verbose_k),
    ], extra_env=extra_env)
    return json.load(open(out_json))


def _locus_from_diag(pinned: dict) -> str:
    """If the pinned arm still flips at decode width, name the residual locus from the diagnostics.
    With attention (num_splits=1) + aten matmuls + lm_head all batch-invariant AND Marlin bit-exact at
    size_m=8, a residual flip at the decode width is a NEW finding (not Marlin -- Marlin is bit-exact
    here, unlike #376's prefill width)."""
    md = pinned.get("marlin_sizem_diag", {})
    marlin_be8 = md.get("bitexact_at_decode_width")
    pin_engaged = pinned.get("aten_mm_control", {}).get("bitexact_M1_vs_M8")
    pin_attn = pinned.get("attn_is_batch_invariant")
    nt = pinned.get("near_tie", {})
    knife = nt.get("residual_is_knife_edge_near_tie")
    nt_clause = ""
    if knife is True:
        nt_clause = (f" The residual is a KNIFE-EDGE NEAR-TIE, not a systematic op divergence: all "
                     f"{nt.get('divergent_count')} flip(s) had the M=1 token inside the M=8 top-5 with a "
                     f"worst top-1 margin {nt.get('margin_vs_m1_max_divergent')} nats < "
                     f"{nt.get('near_tie_logprob_thresh')} (median {nt.get('margin_vs_m1_median_divergent')}); "
                     f"a sub-ULP attention-split perturbation coin-flips the argmax. Matches #375 "
                     f"(attention-rebuild-gated), NOT a Marlin rebuild.")
    elif knife is False:
        nt_clause = (f" The residual is NOT purely a near-tie: worst flip margin "
                     f"{nt.get('margin_vs_m1_max_divergent')} nats >= {nt.get('near_tie_logprob_thresh')} "
                     f"(or M=1 token outside top-5 in {nt.get('divergent_count')-nt.get('n_divergent_m1_in_top5',0)} "
                     f"flip(s)) -> a systematic component is present.")
    if marlin_be8 is True:
        return ("NON-Marlin residual at decode width (Marlin IS bit-exact at size_m=8, so unlike "
                f"#376's prefill-replication RED this is NOT the body GEMM). pin_engaged(aten)="
                f"{pin_engaged}, attn_batch_invariant={pin_attn}. Candidate loci: the served "
                "TRITON_ATTN varlen paged-KV split combine not fully pinned by VLLM_BATCH_INVARIANT, "
                "or the tied bf16 lm_head reduction. Run the lm_head ablation (#376 follow-up #3)."
                + nt_clause)
    if marlin_be8 is False:
        return ("int4 Marlin body GEMM is M-variant even at size_m=8 in this build (contradicts #376 "
                f"sweep) -> Marlin rebuild binding at decode width. pin_engaged(aten)={pin_engaged}")
    return f"undetermined (marlin diag status={md.get('status')}, pin_engaged(aten)={pin_engaged})"


def compose_and_report(arms: dict, a: argparse.Namespace) -> dict:
    heuristic = arms["heuristic"]
    pinned = arms["pinned"]

    heuristic_identity = heuristic["decodewidth_e2e_token_identity_rate"]
    pinned_identity = pinned["decodewidth_e2e_token_identity_rate"]
    heuristic_flip = heuristic["decodewidth_e2e_divergence_rate"]
    pinned_flip = pinned["decodewidth_e2e_divergence_rate"]

    pinned_reaches_identity_1p0 = bool(
        math.isfinite(pinned_identity) and pinned_identity >= 1.0 - IDENTITY_EPS)
    heuristic_below_one = bool(
        math.isfinite(heuristic_identity) and heuristic_identity < 1.0 - IDENTITY_EPS)
    pinned_restores_identity_vs_heuristic = bool(pinned_reaches_identity_1p0 and heuristic_below_one)

    marlin_bitexact_at_decode_width = bool(
        pinned.get("marlin_sizem_diag", {}).get("bitexact_at_decode_width") is True)

    # geometry faithfulness: chunk isolated to the suffix (real M=8 chunk, not full prefill) in BOTH
    # arms, attention is the served TRITON_ATTN varlen paged-KV path (#375), pin live in pinned.
    geom_isolated = bool(heuristic.get("chunk_isolated_fraction", 0) >= 0.99
                         and pinned.get("chunk_isolated_fraction", 0) >= 0.99)
    median_w_ok = bool(abs((pinned.get("median_chunk_width") or 0) - (a.n_verify - 1)) <= 1)
    decodewidth_geometry_is_served_faithful = bool(geom_isolated and median_w_ok)
    geometry_justification = (
        f"M={a.n_verify} query rows computed as ONE chunk against the cached paged KV, proven by "
        f"RequestOutput.num_cached_tokens==C in {min(heuristic.get('chunk_isolated_fraction',0), pinned.get('chunk_isolated_fraction',0)):.3f} "
        f"of prompts (==> exactly n_verify rows go through the body GEMM -> size_m={a.n_verify}, the "
        f"Marlin bit-exact regime), median readable width {pinned.get('median_chunk_width')}=K_spec, "
        f"served TRITON_ATTN varlen paged-KV decode (#375). The cached-prefix prompt_logprobs rows are "
        f"uninitialized torch.empty garbage (never computed) and are NOT read. FAITHFUL PROXY: the "
        f"chunk is a prefill-flagged {a.n_verify}-row suffix, not the literal EAGLE-3 spec-verify "
        f"decode step; in vLLM v1 TRITON_ATTN both are varlen-query-vs-paged-KV through the same "
        f"unified kernel, and the pinned arm forces num_splits=1 so its decider is robust to the "
        f"prefill/decode routing nuance.")

    residual_divergence_locus = ("none / identity reached" if pinned_reaches_identity_1p0
                                 else _locus_from_diag(pinned))

    # near-tie characterisation of the (pinned) residual: is every remaining flip a knife-edge
    # numerical coin-flip, or is there a systematic op-level component?
    pinned_near_tie = pinned.get("near_tie", {})
    residual_is_knife_edge_near_tie = pinned_near_tie.get("residual_is_knife_edge_near_tie")

    if pinned_reaches_identity_1p0 and decodewidth_geometry_is_served_faithful:
        verdict = "GREEN_identity_env_reachable_at_decode_width"
    elif not pinned_reaches_identity_1p0 and marlin_bitexact_at_decode_width:
        verdict = "RED_other_residual"
    elif not pinned_reaches_identity_1p0:
        verdict = "RED_marlin_rebuild_binding_even_at_decode_width"
    else:
        verdict = "AMBER_pinned_1p0_but_geometry_unfaithful"

    # ---- self-test (PRIMARY): harness sanity / calibration, not the science verdict ----
    def ctrls_ok(d: dict) -> bool:
        return (d["determinism_M1_vs_M1"] == 1.0 and d["determinism_M8_vs_M8"] == 1.0
                and d["within_batch_copy0_vs_copy1"] == 1.0)

    def arith_ok(d: dict) -> bool:
        ident = d["decodewidth_e2e_token_identity_rate"]
        return (math.isfinite(ident) and 0.0 <= ident <= 1.0
                and abs(d["decodewidth_e2e_divergence_rate"] - (1.0 - ident)) < 1e-9
                and bool(d["nan_clean"]))

    heuristic_controls_ok = ctrls_ok(heuristic)
    pinned_controls_ok = ctrls_ok(pinned)
    pin_engaged = bool(pinned["aten_mm_control"].get("bitexact_M1_vs_M8"))
    pin_attn_flag = bool(pinned.get("attn_is_batch_invariant"))
    heuristic_arith_ok = arith_ok(heuristic)
    pinned_arith_ok = arith_ok(pinned)
    geometry_ok = decodewidth_geometry_is_served_faithful

    self_test = {
        "heuristic_controls_eq_1": heuristic_controls_ok,
        "pinned_controls_eq_1": pinned_controls_ok,
        "pin_engaged_aten_mm_bitexact": pin_engaged,
        "pin_attn_is_batch_invariant_flag": pin_attn_flag,
        "heuristic_arith_consistent": heuristic_arith_ok,
        "pinned_arith_consistent": pinned_arith_ok,
        "decodewidth_geometry_isolated": geometry_ok,
        "marlin_bitexact_at_decode_width": marlin_bitexact_at_decode_width,
    }
    decodewidth_identity_self_test_passes = bool(
        heuristic_controls_ok and pinned_controls_ok and pin_engaged and pin_attn_flag
        and heuristic_arith_ok and pinned_arith_ok and geometry_ok)

    report = {
        "pr": 381,
        "leg": "decode-width e2e token-identity: M=8-verify-vs-M=1-AR at the literal 8-row decode "
               "geometry (local A10G)",
        "imported_anchors": {
            "int4_identity_232": INT4_IDENTITY_232,
            "prefill_repl_pinned_376": PREFILL_REPL_PINNED_376,
            "prefill_repl_heuristic_376": PREFILL_REPL_HEUR_376,
            "marlin_first_divergent_size_m_376": MARLIN_FIRST_DIVERGENT_SIZE_M_376,
            "marlin_bitexact_at_8_376": MARLIN_BITEXACT_AT_8_376,
            "deployed_flip_362": DEPLOYED_FLIP_362,
            "served_attn_375": SERVED_ATTN_375,
            "official_baseline": OFFICIAL_BASELINE, "M_verify": M_VERIFY,
        },
        # ---- REQUIRED deliverable fields ----
        "decodewidth_e2e_token_identity_rate": {
            "heuristic": heuristic_identity, "pinned": pinned_identity},
        "pinned_reaches_identity_1p0": pinned_reaches_identity_1p0,
        "marlin_bitexact_at_decode_width": marlin_bitexact_at_decode_width,
        "residual_divergence_locus": residual_divergence_locus,
        "residual_is_knife_edge_near_tie": residual_is_knife_edge_near_tie,
        "residual_near_tie_summary": {
            "divergent_count": pinned_near_tie.get("divergent_count"),
            "n_divergent_m1_in_top5": pinned_near_tie.get("n_divergent_m1_in_top5"),
            "margin_vs_m1_max_divergent": pinned_near_tie.get("margin_vs_m1_max_divergent"),
            "margin_vs_m1_median_divergent": pinned_near_tie.get("margin_vs_m1_median_divergent"),
            "gap_top2_max_divergent": pinned_near_tie.get("gap_top2_max_divergent"),
            "near_tie_logprob_thresh": pinned_near_tie.get("near_tie_logprob_thresh"),
        },
        "decodewidth_geometry_is_served_faithful": decodewidth_geometry_is_served_faithful,
        "decodewidth_geometry_justification": geometry_justification,
        "verdict": verdict,
        "decodewidth_identity_self_test_passes": decodewidth_identity_self_test_passes,  # PRIMARY
        # ---- supporting ----
        "pinned_restores_identity_vs_heuristic": pinned_restores_identity_vs_heuristic,
        "residual_flip_rate_pinned": pinned_flip,
        "heuristic_flip_rate": heuristic_flip,
        "pin_engaged_aten_mm_bitexact": pin_engaged,
        "pin_attn_is_batch_invariant": pin_attn_flag,
        # cross to fleet anchors: did moving prefill-repl -> decode width restore identity?
        "pinned_decodewidth_minus_prefillrepl_376": (
            pinned_identity - PREFILL_REPL_PINNED_376 if math.isfinite(pinned_identity) else float("nan")),
        # per-arm detail
        "arms": {
            arm: {
                "decodewidth_e2e_token_identity_rate": d["decodewidth_e2e_token_identity_rate"],
                "decodewidth_e2e_divergence_rate": d["decodewidth_e2e_divergence_rate"],
                "determinism_M1_vs_M1": d["determinism_M1_vs_M1"],
                "determinism_M8_vs_M8": d["determinism_M8_vs_M8"],
                "within_batch_copy0_vs_copy1": d["within_batch_copy0_vs_copy1"],
                "chunk_isolated_fraction": d["chunk_isolated_fraction"],
                "median_chunk_width": d["median_chunk_width"],
                "n_computed_rows_total": d["n_computed_rows_total"],
                "expected_computed_rows_per_chunk": d["expected_computed_rows_per_chunk"],
                "vllm_batch_invariant_env": d["vllm_batch_invariant_env"],
                "attn_is_batch_invariant": d["attn_is_batch_invariant"],
                "near_tie": d.get("near_tie", {}),
                "aten_mm_control": d["aten_mm_control"],
                "marlin_sizem_diag": d["marlin_sizem_diag"],
                "total_positions": d["total_positions"], "n_prompts": d["n_prompts"],
                "peak_gpu_gb": d["peak_gpu_gb"],
            } for arm, d in arms.items()
        },
        "self_test": self_test,
        "C": heuristic["C"], "n_verify": heuristic["n_verify"],
        "n_prompts": heuristic["n_prompts"], "model_dir": heuristic["model_dir"],
    }
    return report


def orchestrate(a: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arms = {arm: _run_arm(a, arm) for arm in ARMS}
    report = compose_and_report(arms, a)
    _finish(report, a)


def reanalyze(a: argparse.Namespace) -> None:
    arms = {}
    for arm in ARMS:
        p = OUT_DIR / f"arm_{arm}_result.json"
        if not p.exists():
            raise FileNotFoundError(f"--reanalyze needs {p} (run the GPU arms first)")
        arms[arm] = json.load(open(p))
    report = compose_and_report(arms, a)
    _finish(report, a)


def _finish(report: dict, a: argparse.Namespace) -> None:
    report_path = OUT_DIR / "decodewidth_e2e_report.json"
    json.dump(report, open(report_path, "w"), indent=2)
    _print_console(report)
    if not a.no_wandb:
        log_wandb(report, a)


def _print_console(report: dict) -> None:
    ident = report["decodewidth_e2e_token_identity_rate"]
    print("\n========== DECODE-WIDTH E2E TOKEN-IDENTITY (PR #381) ==========", flush=True)
    print(f" VERDICT                                   : {report['verdict']}", flush=True)
    print(f" decodewidth identity heuristic / pinned   : {ident['heuristic']:.6f} / {ident['pinned']:.6f}", flush=True)
    print(f" pinned_reaches_identity_1p0 (DECIDER)     : {report['pinned_reaches_identity_1p0']}", flush=True)
    print(f" marlin_bitexact_at_decode_width (size_m=8): {report['marlin_bitexact_at_decode_width']}", flush=True)
    print(f" decodewidth_geometry_is_served_faithful   : {report['decodewidth_geometry_is_served_faithful']}", flush=True)
    print(f" residual_divergence_locus                 : {report['residual_divergence_locus']}", flush=True)
    nt = report["residual_near_tie_summary"]
    print(f" residual_is_knife_edge_near_tie           : {report['residual_is_knife_edge_near_tie']} "
          f"({nt['divergent_count']} flip(s), worst margin {nt['margin_vs_m1_max_divergent']} nats "
          f"vs thresh {nt['near_tie_logprob_thresh']})", flush=True)
    print(f" pinned vs #376 prefill-repl (0.992555)    : {report['pinned_decodewidth_minus_prefillrepl_376']:+.6f}", flush=True)
    for arm in ARMS:
        d = report["arms"][arm]
        md = d["marlin_sizem_diag"]
        print(f"   [{arm}] det_m1/det_m8/within = {d['determinism_M1_vs_M1']:.4f}/"
              f"{d['determinism_M8_vs_M8']:.4f}/{d['within_batch_copy0_vs_copy1']:.4f}  "
              f"chunk_isolated={d['chunk_isolated_fraction']:.4f} median_w={d['median_chunk_width']}  "
              f"marlin_be@8={md.get('bitexact_at_decode_width')}  "
              f"aten_pin={d['aten_mm_control'].get('bitexact_M1_vs_M8')}", flush=True)
    print(f" SELF-TEST PASSES (PRIMARY)                : {report['decodewidth_identity_self_test_passes']}", flush=True)
    print(f"   {report['self_test']}", flush=True)
    print(f" report -> {OUT_DIR / 'decodewidth_e2e_report.json'}", flush=True)
    print("===============================================================\n", flush=True)


def log_wandb(report: dict, a: argparse.Namespace) -> None:
    sys.path.insert(0, os.getcwd())
    try:
        from scripts.wandb_logging import init_wandb_run, finish_wandb
    except Exception as exc:
        print(f"[wandb] helper import failed: {exc!r}; skipping", flush=True)
        return
    run = init_wandb_run(
        job_type="local_profiling",
        agent="stark",
        name=a.wandb_name,
        group=a.wandb_group,
        notes="PR#381 decode-width e2e token-identity: M=8-verify-vs-M=1-AR at the literal 8-row "
              "decode geometry (M=8 chunk vs cached paged KV); does pinned identity reach 1.0?",
        config={
            "pr": 381, "M_verify": M_VERIFY, "n_prompts": report["n_prompts"],
            "C": report["C"], "n_verify": report["n_verify"], "model_dir": report["model_dir"],
            "prefill_repl_pinned_376": PREFILL_REPL_PINNED_376,
            "marlin_first_divergent_size_m_376": MARLIN_FIRST_DIVERGENT_SIZE_M_376,
            "deployed_flip_362": DEPLOYED_FLIP_362, "official_baseline": OFFICIAL_BASELINE,
        },
    )
    if run is None:
        print("[wandb] disabled (no API key/mode); results saved to JSON only", flush=True)
        return

    ident = report["decodewidth_e2e_token_identity_rate"]
    summary = {
        "decodewidth_identity_self_test_passes": report["decodewidth_identity_self_test_passes"],
        "decodewidth_identity_heuristic": ident["heuristic"],
        "decodewidth_identity_pinned": ident["pinned"],
        "pinned_reaches_identity_1p0": report["pinned_reaches_identity_1p0"],
        "marlin_bitexact_at_decode_width": report["marlin_bitexact_at_decode_width"],
        "decodewidth_geometry_is_served_faithful": report["decodewidth_geometry_is_served_faithful"],
        "pinned_restores_identity_vs_heuristic": report["pinned_restores_identity_vs_heuristic"],
        "residual_flip_rate_pinned": report["residual_flip_rate_pinned"],
        "heuristic_flip_rate": report["heuristic_flip_rate"],
        "pinned_decodewidth_minus_prefillrepl_376": report["pinned_decodewidth_minus_prefillrepl_376"],
        "verdict": report["verdict"],
        "verdict_green": report["verdict"].startswith("GREEN"),
        "verdict_red": report["verdict"].startswith("RED"),
        "pin_engaged_aten_mm_bitexact": report["pin_engaged_aten_mm_bitexact"],
        "pin_attn_is_batch_invariant": report["pin_attn_is_batch_invariant"],
        "prefill_repl_pinned_376": PREFILL_REPL_PINNED_376,
        "residual_is_knife_edge_near_tie": report["residual_is_knife_edge_near_tie"],
        "residual_margin_vs_m1_max_divergent": report["residual_near_tie_summary"]["margin_vs_m1_max_divergent"],
        "residual_margin_vs_m1_median_divergent": report["residual_near_tie_summary"]["margin_vs_m1_median_divergent"],
        "residual_divergent_count_pinned": report["residual_near_tie_summary"]["divergent_count"],
    }
    for arm in ARMS:
        d = report["arms"][arm]
        md = d["marlin_sizem_diag"]
        nt = d.get("near_tie", {})
        summary[f"{arm}/identity"] = d["decodewidth_e2e_token_identity_rate"]
        summary[f"{arm}/divergence"] = d["decodewidth_e2e_divergence_rate"]
        summary[f"{arm}/det_m1"] = d["determinism_M1_vs_M1"]
        summary[f"{arm}/det_m8"] = d["determinism_M8_vs_M8"]
        summary[f"{arm}/within"] = d["within_batch_copy0_vs_copy1"]
        summary[f"{arm}/chunk_isolated"] = d["chunk_isolated_fraction"]
        summary[f"{arm}/median_chunk_width"] = d["median_chunk_width"]
        summary[f"{arm}/aten_mm_bitexact"] = bool(d["aten_mm_control"].get("bitexact_M1_vs_M8"))
        summary[f"{arm}/marlin_bitexact_at_8"] = bool(md.get("bitexact_at_decode_width"))
        summary[f"{arm}/divergent_count"] = nt.get("divergent_count")
        summary[f"{arm}/residual_knife_edge"] = nt.get("residual_is_knife_edge_near_tie")
        summary[f"{arm}/margin_vs_m1_max_divergent"] = nt.get("margin_vs_m1_max_divergent")
    for k, v in summary.items():
        run.summary[k] = v
    for k, v in report["self_test"].items():
        run.summary[f"selftest/{k}"] = v
    run.summary["residual_divergence_locus"] = report["residual_divergence_locus"]
    run.summary["decodewidth_geometry_justification"] = report["decodewidth_geometry_justification"]
    finish_wandb(run)
    print(f"[wandb] logged run {run.id}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["arm"], default=None,
                    help="internal: run one GPU arm (subprocess). Omit for the orchestrator.")
    ap.add_argument("--arm", choices=list(ARMS), default=None, help="internal: which arm")
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-test", dest="self_test", action="store_true",
                    help="run both arms + the PRIMARY self-test (default orchestrator path)")
    ap.add_argument("--reanalyze", action="store_true",
                    help="0-GPU: recompose the report + self-test from saved arm_*.json")
    ap.add_argument("--smoke", action="store_true", help="tiny run (few prompts) to validate the path")
    # card reproduce-skeleton flags (accepted for parity; mapped/recorded)
    ap.add_argument("--gpu", action="store_true", help="(compat) GPU is always used for the arms")
    ap.add_argument("--decode-width", dest="decode_width", action="store_true",
                    help="(compat) this harness is the decode-width mode by construction")
    ap.add_argument("--real-lmhead", dest="real_lmhead", action="store_true",
                    help="(compat) vLLM path always uses the real tied bf16 lm_head")
    ap.add_argument("--real-int4-body", dest="real_int4_body", action="store_true",
                    help="(compat) vLLM path always uses the real int4-Marlin body")
    ap.add_argument("--proxy", default=DEFAULT_PROXY, help="(compat) real-weight proxy id; resolved from cache")
    ap.add_argument("--eval-prompts", dest="n_prompts", type=int, default=128,
                    help="number of official eval prompts (alias of the card's --eval-prompts)")
    ap.add_argument("--n-prompts", dest="n_prompts", type=int, default=128)
    ap.add_argument("--ctx-len", dest="ctx_len", type=int, default=224,
                    help="context/prefix length (aligned down to a multiple of 32 = the Gemma-4 hybrid "
                         "prefix-commit granularity, so the verify chunk is the literal size_m=8). "
                         "224 is the largest faithful prefix <=240.")
    ap.add_argument("--n-verify", dest="n_verify", type=int, default=M_VERIFY,
                    help="verify-chunk width = K_spec+1 = 8")
    ap.add_argument("--gpu-mem-util", type=float, default=0.55)
    ap.add_argument("--max-batched-tokens", type=int, default=8192)
    ap.add_argument("--verbose-k", dest="verbose_k", type=int, default=3)
    ap.add_argument("--wandb_group", dest="wandb_group", default="strict-bi-verify-gemm")
    ap.add_argument("--wandb_name", dest="wandb_name", default="stark/decodewidth-e2e-identity")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()

    if a.smoke and a.phase is None:
        a.n_prompts = min(a.n_prompts, 4)

    if a.phase == "arm":
        phase_arm(a.out, a.arm, a.n_prompts, a.ctx_len, a.n_verify,
                  a.gpu_mem_util, a.max_batched_tokens, a.verbose_k)
    elif a.reanalyze:
        reanalyze(a)
    else:
        orchestrate(a)


if __name__ == "__main__":
    main()
