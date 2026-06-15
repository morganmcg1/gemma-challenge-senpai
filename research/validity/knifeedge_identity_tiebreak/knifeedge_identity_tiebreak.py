#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #397 (stark) -- Knife-edge identity tie-break: can a POSITION-SELECTIVE deterministic
tie-break recover e2e greedy identity 1.0 at far less than the ~11-TPS blanket FA_SLIDING=0 cost?

THE QUESTION
------------
My #381 (`9edps20u`) localized the sole e2e decode-width greedy-identity residual to **1 flip in
~891 tokens** at the literal M=8 decode geometry -- a **0.125-nat knife-edge near-tie** in the served
attention (the pinned arm: VLLM_BATCH_INVARIANT=1, TRITON_ATTN, num_splits=1), NOT Marlin (int4 body
byte-exact at size_m=8). The blanket fix that restores byte-identity (FA_SLIDING=0, reverting the
sliding-window target layers from FA2 back to TRITON) costs ~11 TPS == eta_attn (the whole attention
strict tax; kanna #38: FA_SLIDING=0 restores byte-identity 0/32 at an unmeasured TPS cost).

Hypothesis: a tie-break applied ONLY where the top-2 verify-logit margin is inside the knife-edge band
can recover identity 1.0 at cost ~ f * eta_attn << eta_attn, because the flip lives at a handful of
near-tie positions and a matched-precision attention reduction there is nearly free.

THE KEY RISK -- ubel #364's PRECISION WALL (the closest prior, MERGED on this branch)
------------------------------------------------------------------------------------
ubel #364 (`margin_localized_identity`, `i08xlqbg`) tested exactly a logit-margin-gated selective
identity repair over 512-token decode trajectories and got RED: the margin gate has **recall 1.0**
(catches every flip, AUC 0.972, all flips <=1.125 nat) but **catastrophic precision** -- flips are a
0.39% needle in a **~17% natural low-margin haystack**, so any threshold catching every flip flags
17-45% of positions. Priced at a **full M=1 forward per flagged position** (lm_head-only does NOT
restore identity; the divergence is bf16-attention-injected upstream and propagates -- wirbel #362),
selective eta = frac_flagged * full_forward = 17-44.6% >> 9.841% blanket -> `selective_beats_blanket
= False`, 4.5x more expensive.

WHY #397 IS A GENUINE, NON-DUPLICATIVE TEST OF #364 (two differences that can move the verdict)
----------------------------------------------------------------------------------------------
1. GEOMETRY: #364 averaged over 512-token trajectories (many high-entropy positions). #397 measures
   the **literal M=8 verify step** (one 8-row chunk against a cached 224-token paged KV -- the exact
   deployed EAGLE-3 verify geometry, #381). The near-tie base rate at the verify step can differ from
   the trajectory background.
2. COST UNIT: #364 priced each flagged position at a FULL M=1 forward (re-decode). #397 prices the fix
   as the BLANKET MECHANISM applied selectively: a near-tie verify STEP upgrades its attention
   reduction to the matched-precision (FA_SLIDING=0 / num_splits=1) path. That is `eta_attn` for that
   one step, not a full extra forward -- so `selective_fix_tps_cost ~ f_step_band * eta_attn`.

This card MEASURES the decode-width band fractions and reconciles them against #364's 17% wall, then
decides whether the selective fix is materially cheaper than the blanket at the decode width.

SCOPE: LOCAL A10G probe. analysis_only / no_hf_job / no_served_file_change / official_tps=0. A tie-break
in the *served* decode path would be the flagged step and is OUT OF SCOPE -- this card measures
VIABILITY and COST only, by post-hoc analysis of the M=8 verify logits + the M=1 AR reference that the
#381 served-faithful harness already produces. No served file is touched; the int4 path is READ only.

THE RECOVERY MODEL (why it is grounded, not circular)
-----------------------------------------------------
At a verify position p, the M=1 AR reference token `full[p]` IS, by construction + the MEASURED
det_m1=1.0 determinism control, the output of the matched (num_splits=1, single-segment M=1) attention
reduction at p. The matched-precision reduction engaged selectively at p therefore yields `full[p]`.
So the selective fix recovers identity to 1.0 **iff every flip's top-2 gap is inside the engaged band**
(max_flip_gap <= eps). That is a MEASURED quantity (the flip gaps), not an assumption.

  identity_selective(eps) = 1 - (# flips with gap > eps) / total_positions
  f_step_band(eps)        = (# verify steps with min top-2 gap <= eps) / (# verify steps)
  selective_fix_tps_cost  = f_step_band(eps*) * eta_attn      [eps* = smallest band covering all flips]

TWO ARMS (reuse #381, isolated subprocesses; the pin is set per-arm in ENV):
  heuristic -- stock vLLM (VLLM_BATCH_INVARIANT=0): the stock verify-width attention split heuristic.
  pinned    -- VLLM_BATCH_INVARIANT=1 (num_splits=1 + aten batch-invariant): the strict-deployed
               candidate where #381 localized the lone residual flip. PRIMARY arm for the band readout.
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
# #381 (this student) decode-width pinned residual at the literal size_m=8 verify geometry:
PINNED_IDENTITY_381 = 0.9988751406074241      # 888/889 -- 1 flip
PINNED_FLIP_COUNT_381 = 1
PINNED_FLIP_MARGIN_381 = 0.125                 # nats; the lone flip's top-1-vs-M1 margin AND top-2 gap
HEURISTIC_IDENTITY_381 = 0.9966254218222722    # 886/889 -- 3 flips (all knife-edge, margins 0.125)

# Blanket attention strict tax (IMPORTED -- not re-measured; would need a TPS run = out of scope):
FA_SLIDING0_TPS_COST = 11.0                    # eta_attn: FA_SLIDING=0 reverts FA2 sliding layers to
                                               # TRITON, restoring byte-identity (kanna #38: 0/32) at
                                               # ~11 TPS. The blanket this card tries to beat.

# ubel #364 margin_localized_identity (`i08xlqbg`, MERGED) -- the closest prior, RED:
MARGIN364_FLIP_RATE = 0.0039                   # 32/8192 over 512-tok trajectories
MARGIN364_LOWMARGIN_BACKGROUND = 0.1702        # frac positions < 1.125 nat (the precision-wall haystack)
MARGIN364_BLANKET_GEMM = 0.09841               # batch-invariant GEMM blanket (wirbel #360), fraction
MARGIN364_SELECTIVE_ETA_PROVABLE = 0.4460      # frac flagged at the provable tau (full-forward cost)
MARGIN364_SELECTIVE_BEATS_BLANKET = False      # the RED headline

# Strict-base context from the #397 PR baseline:
OFFICIAL_BASELINE = 481.53                      # #52 deployed frontier TPS (this leg adds 0)
CORRECTED_STRICT_BASE_390 = 471.42             # #390 5y64zbjz realized strict base
GAP_TO_500 = 28.58                             # 500 - corrected strict base
BAND_CEILING = 509.78                          # if eta_attn -> 0 cheaply, base rises toward this

K_SPEC = 7
M_VERIFY = K_SPEC + 1                           # = 8, the deployed decode-verify query width
IDENTITY_EPS = 1e-12
NEAR_TIE_LOGPROB_THRESH = 0.5                   # #381 knife-edge characterisation threshold
# Knife-edge band thresholds for the position-selective tie-break (the PR-required bands).
# Membership uses gap <= thr (+ a tiny tol): the bf16 logprob grid quantises gaps to multiples of
# 0.125, so the lone 0.125-nat flip must land in the TIGHTEST band -- a strict < 0.125 would wrongly
# exclude it. logprob top-2 gap == logit top-2 margin exactly (softmax is a per-row shift).
BAND_THRESHOLDS = (0.125, 0.25, 0.5)
BAND_TOL = 1e-9

BLOCK_SIZE = 16
HYBRID_PREFIX_COMMIT = 32                       # Gemma-4 hybrid prefix-cache commit granularity (#381)

DEFAULT_PROXY = "google/gemma-4-E4B-it-qat-w4a16-ct"
MODEL_CANDIDATES = [
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots"
    ),
    "/tmp/osoi5-v0-baked",
]
PROMPTS_JSONL = "official/main_bucket/shared_resources/speed_benchmark/data/ppl_ground_truth_tokens.jsonl"

OUT_DIR = Path("research/validity/knifeedge_identity_tiebreak")
ARMS = ("heuristic", "pinned")
PRIMARY_ARM = "pinned"   # the strict-deployed candidate; band readout + verdict come from here


# --------------------------------------------------------------------------------------
# Small helpers (resolve_model_dir / read_text_dims / block_align reused from #381)
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
    return (n // HYBRID_PREFIX_COMMIT) * HYBRID_PREFIX_COMMIT


def _argmax_from_logprob_entry(entry) -> int:
    return int(max(entry.items(), key=lambda kv: getattr(kv[1], "logprob", kv[1]))[0])


def _sorted_logprobs(entry) -> list[tuple[int, float]]:
    """(token_id, logprob) pairs sorted by logprob descending (rank 0 first)."""
    return sorted(((int(t), float(getattr(lp, "logprob", lp))) for t, lp in entry.items()),
                  key=lambda kv: kv[1], reverse=True)


# ======================================================================================
# PHASE: one arm. Extends #381's phase_arm to collect EVERY suffix position's top-2 gap +
# token-ids (for the band histogram and the deterministic tie-break test), not just flips.
# ======================================================================================
def phase_arm(out_path: str, arm: str, n_prompts: int, ctx_len: int, n_verify: int,
              gpu_mem_util: float, max_batched_tokens: int, verbose_k: int) -> None:
    import torch
    from vllm import LLM, SamplingParams

    batch_invariant_env = os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1"
    model_dir = resolve_model_dir()
    dims = read_text_dims(model_dir)
    C = block_align(ctx_len)
    print(f"[arm:{arm}] model={model_dir} layers={dims['num_layers']} C(prefix)={C} "
          f"n_verify={n_verify} VLLM_BATCH_INVARIANT={batch_invariant_env}", flush=True)

    t0 = time.time()
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

    sp_gen = SamplingParams(temperature=0.0, max_tokens=n_verify, detokenize=False)
    sp_warm = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False)
    sp_chunk = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=5,
                              skip_reading_prefix_cache=False, detokenize=False)

    rows = [json.loads(l) for l in open(PROMPTS_JSONL)][:n_prompts]
    per_prompt = []
    n_match = n_total = 0
    n_det_m1 = n_det_m8 = n_within = 0
    n_chunk_isolated = 0
    chunk_width_obs = []
    n_computed_rows_total = 0

    # --- #397 per-position collection (the new instrumentation) -------------------------
    position_gaps: list[float] = []        # top-2 gap at EVERY readable suffix position
    position_is_flip: list[int] = []       # parallel: 1 if M=8 argmax != M=1 token
    chunk_min_gaps: list[float] = []        # per verify-step (chunk) min top-2 gap (for f_step_band)
    flip_details: list[dict] = []           # full detail per divergent position

    for ri, rec in enumerate(rows):
        src = list(rec.get("context_token_ids", [])) + list(rec.get("target_token_ids", []))
        if len(src) < C + 1:
            continue
        prefix = src[:C]

        # warm the prefix cache (served-faithful + control-clean, #381)
        llm.generate([{"prompt_token_ids": prefix}], sp_warm, use_tqdm=False)

        # Step A: M=1 AR greedy continuation + M1-vs-M1 determinism control (both warm)
        outA = llm.generate([{"prompt_token_ids": prefix}], sp_gen, use_tqdm=False)[0]
        cont = list(outA.outputs[0].token_ids)[:n_verify]
        outA2 = llm.generate([{"prompt_token_ids": prefix}], sp_gen, use_tqdm=False)[0]
        cont2 = list(outA2.outputs[0].token_ids)[:n_verify]
        det_m1 = int(cont == cont2)
        if len(cont) < n_verify:
            continue
        full = prefix + cont

        # Step B: M=8 verify chunk + M8-vs-M8 determinism control
        def chunk_argmax(full_ids):
            out = llm.generate([{"prompt_token_ids": full_ids}], sp_chunk, use_tqdm=False)[0]
            nct = out.num_cached_tokens or 0
            pls = out.prompt_logprobs or []
            am = {}
            ent = {}
            for i in range(C + 1, len(full_ids)):
                entry = pls[i] if i < len(pls) else None
                if entry is not None:
                    am[i] = _argmax_from_logprob_entry(entry)
                    ent[i] = _sorted_logprobs(entry)
            return am, nct, ent

        m8, nct8, ent8 = chunk_argmax(full)
        m8b, nct8b, _ = chunk_argmax(full)

        suffix_pos = sorted(m8)
        n_computed_rows = len(full) - nct8
        n_computed_rows_b = len(full) - nct8b
        chunk_isolated = (n_computed_rows == n_verify and n_computed_rows_b == n_verify)
        n_computed_rows_total += n_computed_rows
        chunk_width_obs.append(len(suffix_pos))
        n_chunk_isolated += int(chunk_isolated)

        det_m8 = int(all(m8.get(p) == m8b.get(p) for p in suffix_pos) and bool(suffix_pos))

        # within-batch copy0 vs copy1 control
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

        # the signal: M=8 chunk argmax vs M=1 greedy token, per position, WITH the full gap record
        match = total = 0
        prompt_min_gap = float("inf")
        for p in suffix_pos:
            m1_tok = full[p]
            total += 1
            sl = ent8.get(p, [])
            gap = (sl[0][1] - sl[1][1]) if len(sl) >= 2 else float("inf")
            is_flip = int(m8.get(p) != m1_tok)
            if math.isfinite(gap):
                position_gaps.append(gap)
                position_is_flip.append(is_flip)
                prompt_min_gap = min(prompt_min_gap, gap)
            if not is_flip:
                match += 1
            else:
                lp_map = dict(sl)
                top1_id = sl[0][0] if sl else None
                top2_id = sl[1][0] if len(sl) >= 2 else None
                top1_lp = sl[0][1] if sl else float("nan")
                m1_in_top5 = m1_tok in lp_map
                m1_margin = (top1_lp - lp_map[m1_tok]) if m1_in_top5 else None
                # deterministic stable lowest-index tie-break over the M=8 top-2: does it pick M=1?
                band_pair_ids = [t for t in (top1_id, top2_id) if t is not None]
                id_tiebreak_pick = min(band_pair_ids) if band_pair_ids else None
                id_tiebreak_recovers = bool(id_tiebreak_pick == m1_tok)
                flip_details.append({
                    "prompt_idx": ri, "pos": p, "gap": round(gap, 6),
                    "m8_top1_id": top1_id, "m8_top2_id": top2_id, "m1_tok_id": m1_tok,
                    "m1_in_top5": m1_in_top5,
                    "m1_margin": (round(m1_margin, 6) if m1_margin is not None else None),
                    "id_tiebreak_pick": id_tiebreak_pick,
                    "id_tiebreak_recovers": id_tiebreak_recovers,
                })
        if math.isfinite(prompt_min_gap):
            chunk_min_gaps.append(prompt_min_gap)

        n_match += match
        n_total += total
        n_det_m1 += det_m1 * max(1, total)
        n_det_m8 += det_m8 * max(1, total)
        n_within += within * max(1, total)

        sha = hashlib.sha256(bytes(str([m8.get(p) for p in suffix_pos]), "utf8")).hexdigest()[:16]
        per_prompt.append({
            "id": rec.get("id"), "C": C, "chunk_width": len(suffix_pos),
            "chunk_isolated": chunk_isolated, "num_cached_tokens": nct8,
            "argmax_match_M8_vs_M1": match, "positions": total, "sha": sha,
            "det_match_M1_vs_M1": det_m1, "det_match_M8_vs_M8": det_m8,
            "within_copy0_vs_copy1": within,
            "min_top2_gap": (prompt_min_gap if math.isfinite(prompt_min_gap) else None),
        })
        if ri < verbose_k or ri == len(rows) - 1:
            print(f"[arm:{arm}] prompt {ri} chunk_w={len(suffix_pos)} isolated={chunk_isolated} "
                  f"match={match}/{total} det_m1={det_m1} det_m8={det_m8} within={within} "
                  f"min_gap={prompt_min_gap if math.isfinite(prompt_min_gap) else None}", flush=True)

    n_seq = len(per_prompt)
    identity = (n_match / n_total) if n_total else float("nan")
    det_m1_frac = (n_det_m1 / n_total) if n_total else float("nan")
    det_m8_frac = (n_det_m8 / n_total) if n_total else float("nan")
    within_frac = (n_within / n_total) if n_total else float("nan")
    chunk_isolated_frac = (n_chunk_isolated / n_seq) if n_seq else float("nan")
    median_chunk_width = (statistics.median(chunk_width_obs) if chunk_width_obs else float("nan"))

    # ---- band characterisation (position-level + step-level) ----
    bands = compute_bands(position_gaps, position_is_flip, chunk_min_gaps)

    # ---- pin-engaged positive control: aten torch.mm row-0 bit-exactness M=1 vs M=8 ----
    aten_ctrl = aten_mm_invariance_control(torch, dims["hidden"], M_VERIFY)

    # ---- Marlin size_m diag: re-confirm int4 body GEMM bit-exact at the decode width (8) ----
    try:
        marlin_diag = marlin_sizem_diag(llm, dims, torch, M_VERIFY)
    except Exception as exc:
        marlin_diag = {"status": f"failed: {exc!r}", "per_size": {},
                       "first_divergent_size_m": None, "bitexact_at_decode_width": None}
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
        "bands": bands,
        "flip_details": flip_details,
        "n_chunks": len(chunk_min_gaps),
        "aten_mm_control": aten_ctrl,
        "marlin_sizem_diag": marlin_diag,
        "nan_clean": bool(nan_clean), "peak_gpu_gb": peak_gb,
        "per_prompt": per_prompt,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[arm:{arm}] identity={identity:.6f} flips={bands['flip_count']} "
          f"band(<=.125/.25/.5) pos={bands['position_count_0p125']}/{bands['position_count_0p25']}/"
          f"{bands['position_count_0p5']} of {bands['total_positions']} "
          f"step={bands['step_count_0p125']}/{bands['step_count_0p25']}/{bands['step_count_0p5']} "
          f"of {bands['n_chunks']} peak={peak_gb:.1f}GB", flush=True)
    print(f"ARM_DONE {out_path}", flush=True)


def compute_bands(position_gaps: list[float], position_is_flip: list[int],
                  chunk_min_gaps: list[float]) -> dict:
    """Count near-tie positions and verify-steps in each knife-edge band, plus per-flip gaps.

    position-level count drives the #364 precision reconciliation; step-level count drives the
    eta_attn selective cost (a verify step engages the matched-precision attention iff ANY of its
    rows is a near-tie)."""
    total = len(position_gaps)
    n_chunks = len(chunk_min_gaps)
    flip_gaps = [g for g, f in zip(position_gaps, position_is_flip) if f]
    out: dict = {
        "total_positions": total,
        "n_chunks": n_chunks,
        "flip_count": len(flip_gaps),
        "flip_gaps": [round(g, 6) for g in flip_gaps],
        "max_flip_gap": (max(flip_gaps) if flip_gaps else None),
        "median_position_gap": (round(statistics.median(position_gaps), 6) if position_gaps else None),
        "median_chunk_min_gap": (round(statistics.median(chunk_min_gaps), 6) if chunk_min_gaps else None),
    }
    for thr in BAND_THRESHOLDS:
        key = _band_key(thr)
        pos_n = sum(1 for g in position_gaps if g <= thr + BAND_TOL)
        step_n = sum(1 for g in chunk_min_gaps if g <= thr + BAND_TOL)
        flip_in = sum(1 for g in flip_gaps if g <= thr + BAND_TOL)
        out[f"position_count_{key}"] = pos_n
        out[f"position_frac_{key}"] = (pos_n / total) if total else float("nan")
        out[f"step_count_{key}"] = step_n
        out[f"step_frac_{key}"] = (step_n / n_chunks) if n_chunks else float("nan")
        out[f"flips_in_{key}"] = flip_in
    return out


def _band_key(thr: float) -> str:
    return f"{thr:g}".replace(".", "p")


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
    """Row-0 bit-exactness of the int4-Marlin body GEMMs across size_m, re-confirming the body GEMM
    is bit-exact at the decode width (size_m=8). Reused from #381/#376/#232."""
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
        "status": "ran", "sizes_tested": sizes, "per_size": per_size,
        "first_divergent_size_m": first_divergent,
        "bitexact_at_decode_width": per_size.get(str(decode_width), {}).get("bitexact_row0_vs_M1"),
        "decode_width": decode_width,
    }


# ======================================================================================
# Orchestrator: isolated subprocess arms, compose, self-test, wandb
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


def selective_identity_at(bands: dict, eps: float) -> float:
    """identity_selective(eps) = 1 - (# flips with gap > eps) / total_positions.
    The selective matched-precision attention engaged at gap <= eps recovers the M=1 token at every
    engaged near-tie (grounded in det_m1=1.0); flips outside the band are NOT repaired."""
    total = bands["total_positions"]
    if not total:
        return float("nan")
    flips_outside = sum(1 for g in bands["flip_gaps"] if g > eps + BAND_TOL)
    return 1.0 - flips_outside / total


def compose_and_report(arms: dict, a: argparse.Namespace) -> dict:
    primary = arms[PRIMARY_ARM]
    pb = primary["bands"]

    # --- (1) knife-edge band characterisation (primary = pinned arm) ---
    knifeedge_position_count_0p125 = pb["position_count_0p125"]
    knifeedge_position_count_0p25 = pb["position_count_0p25"]
    knifeedge_position_count_0p5 = pb["position_count_0p5"]
    flip_count_at_decode = pb["flip_count"]
    max_flip_gap = pb["max_flip_gap"]
    total_positions = pb["total_positions"]
    # the #381 single flip should live in the TIGHTEST band:
    flip_in_tightest_band = bool(
        flip_count_at_decode > 0 and max_flip_gap is not None
        and max_flip_gap <= BAND_THRESHOLDS[0] + BAND_TOL)

    # eps* = smallest band threshold that covers EVERY flip (so the selective fix can reach 1.0)
    eps_star = None
    for thr in BAND_THRESHOLDS:
        if max_flip_gap is not None and max_flip_gap <= thr + BAND_TOL:
            eps_star = thr
            break
    # if a flip sits above the widest band (NOT a knife-edge), eps_star stays None -> not recoverable
    # by any in-band tie-break (would corroborate a systematic, not near-tie, residual)

    # --- (2) tie-break recovery ---
    selective_identity_by_eps = {
        _band_key(thr): selective_identity_at(pb, thr) for thr in BAND_THRESHOLDS}
    tiebreak_recovers_identity_1p0 = bool(
        eps_star is not None
        and selective_identity_at(pb, eps_star) >= 1.0 - IDENTITY_EPS)
    # secondary: the cheap stable lowest-index tie-break over the M=8 top-2 (zero recompute)
    id_tiebreak_recovers_all_flips = bool(
        flip_count_at_decode > 0
        and all(f["id_tiebreak_recovers"] for f in primary["flip_details"]))

    # --- (3) cost: selective (eta_attn unit) vs blanket FA_SLIDING=0 ---
    fa_sliding0_tps_cost = FA_SLIDING0_TPS_COST
    if eps_star is not None:
        f_step_band = pb[f"step_frac_{_band_key(eps_star)}"]
        f_position_band = pb[f"position_frac_{_band_key(eps_star)}"]
    else:
        f_step_band = float("nan")
        f_position_band = float("nan")
    selective_fix_tps_cost = (f_step_band * fa_sliding0_tps_cost
                              if math.isfinite(f_step_band) else float("nan"))
    selective_cheaper_than_blanket = bool(
        math.isfinite(selective_fix_tps_cost)
        and selective_fix_tps_cost < fa_sliding0_tps_cost)
    cheapness_ratio = (fa_sliding0_tps_cost / selective_fix_tps_cost
                       if (math.isfinite(selective_fix_tps_cost) and selective_fix_tps_cost > 0)
                       else float("inf"))

    # #364 reconciliation: the pessimistic full-M=1-forward cost unit (their cost model) at the
    # decode width, in the same fraction units as #364's 9.841% blanket and 17% background.
    pessimistic_full_forward_eta = f_position_band  # frac flagged * 1 full forward
    # does the decode-width low-margin background corroborate #364's 17% trajectory background?
    decodewidth_lowmargin_background_0p5 = pb["position_frac_0p5"]
    corroborates_364_precision_wall = bool(
        math.isfinite(pessimistic_full_forward_eta)
        and pessimistic_full_forward_eta >= MARGIN364_BLANKET_GEMM)  # >= blanket -> #364 RED holds

    targeted_identity_recovery_viable = bool(
        tiebreak_recovers_identity_1p0 and selective_cheaper_than_blanket)

    # --- verdict ---
    if not (flip_count_at_decode > 0):
        verdict = "NO_RESIDUAL_identity_already_1p0"
    elif tiebreak_recovers_identity_1p0 and selective_cheaper_than_blanket and cheapness_ratio >= 2.0:
        verdict = "GREEN_selective_materially_cheaper_than_blanket"
    elif tiebreak_recovers_identity_1p0 and selective_cheaper_than_blanket:
        verdict = "AMBER_selective_cheaper_but_precision_limited"
    elif tiebreak_recovers_identity_1p0:
        verdict = "RED_selective_recovers_but_not_cheaper_than_blanket"
    else:
        verdict = "RED_residual_not_in_band_systematic"

    # --- self-test (PRIMARY): >= 20 harness sanity / arithmetic / band-logic checks ---
    self_test, n_checks = build_self_test(arms, pb, eps_star, selective_identity_by_eps,
                                          selective_fix_tps_cost, fa_sliding0_tps_cost,
                                          tiebreak_recovers_identity_1p0,
                                          selective_cheaper_than_blanket,
                                          flip_in_tightest_band)
    knifeedge_tiebreak_self_test_passes = bool(all(self_test.values()) and n_checks >= 20)

    report = {
        "pr": 397,
        "leg": "knife-edge identity tie-break: position-selective recovery vs the ~11-TPS "
               "FA_SLIDING=0 blanket, at the literal M=8 decode geometry (local A10G)",
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
        "primary_arm": PRIMARY_ARM,
        "imported_anchors": {
            "pinned_identity_381": PINNED_IDENTITY_381,
            "pinned_flip_count_381": PINNED_FLIP_COUNT_381,
            "pinned_flip_margin_381": PINNED_FLIP_MARGIN_381,
            "heuristic_identity_381": HEURISTIC_IDENTITY_381,
            "fa_sliding0_tps_cost": FA_SLIDING0_TPS_COST,
            "margin364_flip_rate": MARGIN364_FLIP_RATE,
            "margin364_lowmargin_background": MARGIN364_LOWMARGIN_BACKGROUND,
            "margin364_blanket_gemm": MARGIN364_BLANKET_GEMM,
            "margin364_selective_eta_provable": MARGIN364_SELECTIVE_ETA_PROVABLE,
            "margin364_selective_beats_blanket": MARGIN364_SELECTIVE_BEATS_BLANKET,
            "official_baseline": OFFICIAL_BASELINE,
            "corrected_strict_base_390": CORRECTED_STRICT_BASE_390,
            "gap_to_500": GAP_TO_500, "band_ceiling": BAND_CEILING,
            "M_verify": M_VERIFY,
        },
        # ---- REQUIRED deliverable fields ----
        "knifeedge_position_count_0p125": knifeedge_position_count_0p125,
        "knifeedge_position_count_0p25": knifeedge_position_count_0p25,
        "knifeedge_position_count_0p5": knifeedge_position_count_0p5,
        "flip_count_at_decode": flip_count_at_decode,
        "flip_in_tightest_band": flip_in_tightest_band,
        "max_flip_gap": max_flip_gap,
        "eps_star": eps_star,
        "tiebreak_recovers_identity_1p0": tiebreak_recovers_identity_1p0,
        "id_tiebreak_recovers_all_flips": id_tiebreak_recovers_all_flips,
        "selective_identity_by_eps": selective_identity_by_eps,
        "selective_fix_tps_cost": selective_fix_tps_cost,
        "fa_sliding0_tps_cost": fa_sliding0_tps_cost,
        "selective_cheaper_than_blanket": selective_cheaper_than_blanket,
        "cheapness_ratio_blanket_over_selective": cheapness_ratio,
        "targeted_identity_recovery_viable": targeted_identity_recovery_viable,
        "knifeedge_tiebreak_self_test_passes": knifeedge_tiebreak_self_test_passes,  # PRIMARY
        # ---- cost-model detail ----
        "f_step_band_at_eps_star": f_step_band,
        "f_position_band_at_eps_star": f_position_band,
        "pessimistic_full_forward_eta": pessimistic_full_forward_eta,
        "decodewidth_lowmargin_background_0p5": decodewidth_lowmargin_background_0p5,
        "corroborates_364_precision_wall": corroborates_364_precision_wall,
        "total_positions": total_positions,
        "n_chunks_primary": pb["n_chunks"],
        "verdict": verdict,
        # ---- per-arm detail ----
        "arms": {
            arm: {
                "decodewidth_e2e_token_identity_rate": d["decodewidth_e2e_token_identity_rate"],
                "decodewidth_e2e_divergence_rate": d["decodewidth_e2e_divergence_rate"],
                "determinism_M1_vs_M1": d["determinism_M1_vs_M1"],
                "determinism_M8_vs_M8": d["determinism_M8_vs_M8"],
                "within_batch_copy0_vs_copy1": d["within_batch_copy0_vs_copy1"],
                "chunk_isolated_fraction": d["chunk_isolated_fraction"],
                "median_chunk_width": d["median_chunk_width"],
                "vllm_batch_invariant_env": d["vllm_batch_invariant_env"],
                "attn_is_batch_invariant": d["attn_is_batch_invariant"],
                "aten_mm_bitexact": bool(d["aten_mm_control"].get("bitexact_M1_vs_M8")),
                "marlin_bitexact_at_decode_width": bool(
                    d["marlin_sizem_diag"].get("bitexact_at_decode_width")),
                "bands": d["bands"],
                "flip_details": d["flip_details"],
                "total_positions": d["total_positions"], "n_prompts": d["n_prompts"],
                "peak_gpu_gb": d["peak_gpu_gb"],
            } for arm, d in arms.items()
        },
        "self_test": self_test,
        "self_test_n_checks": n_checks,
        "C": primary["C"], "n_verify": primary["n_verify"],
        "n_prompts": primary["n_prompts"], "model_dir": primary["model_dir"],
    }
    return report


def build_self_test(arms, pb, eps_star, sel_by_eps, selective_cost, blanket_cost,
                    recovers, cheaper, flip_in_tightest):
    """>= 20 boolean checks: per-arm determinism/arith controls + band-logic + cost-model arithmetic."""
    checks: dict = {}

    def ctrls_ok(d):
        return (d["determinism_M1_vs_M1"] == 1.0 and d["determinism_M8_vs_M8"] == 1.0
                and d["within_batch_copy0_vs_copy1"] == 1.0)

    def arith_ok(d):
        ident = d["decodewidth_e2e_token_identity_rate"]
        return (math.isfinite(ident) and 0.0 <= ident <= 1.0
                and abs(d["decodewidth_e2e_divergence_rate"] - (1.0 - ident)) < 1e-9
                and bool(d["nan_clean"]))

    for arm, d in arms.items():
        checks[f"{arm}_determinism_m1_eq_1"] = bool(d["determinism_M1_vs_M1"] == 1.0)
        checks[f"{arm}_determinism_m8_eq_1"] = bool(d["determinism_M8_vs_M8"] == 1.0)
        checks[f"{arm}_within_eq_1"] = bool(d["within_batch_copy0_vs_copy1"] == 1.0)
        checks[f"{arm}_arith_consistent"] = bool(arith_ok(d))
        checks[f"{arm}_geometry_isolated"] = bool(d["chunk_isolated_fraction"] >= 0.99)
        checks[f"{arm}_marlin_bitexact_at_8"] = bool(
            d["marlin_sizem_diag"].get("bitexact_at_decode_width") is True)
        b = d["bands"]
        checks[f"{arm}_band_monotonic"] = bool(
            b["position_count_0p125"] <= b["position_count_0p25"] <= b["position_count_0p5"])
        checks[f"{arm}_band_le_total"] = bool(b["position_count_0p5"] <= b["total_positions"])
        checks[f"{arm}_flipcount_matches_divergence"] = bool(
            b["flip_count"] == round(d["decodewidth_e2e_divergence_rate"] * d["total_positions"]))
        checks[f"{arm}_all_flips_in_widest_band"] = bool(
            b["flip_count"] == 0 or b["flips_in_0p5"] == b["flip_count"])

    # pin must be engaged in the pinned arm (positive control)
    checks["pinned_pin_engaged_aten"] = bool(arms["pinned"]["aten_mm_control"].get("bitexact_M1_vs_M8"))
    checks["pinned_attn_batch_invariant"] = bool(arms["pinned"].get("attn_is_batch_invariant"))
    # heuristic arm must NOT be batch-invariant (control separation)
    checks["heuristic_not_batch_invariant"] = bool(
        not arms["heuristic"].get("attn_is_batch_invariant"))

    # band-logic / recovery checks (primary arm)
    checks["primary_has_residual"] = bool(pb["flip_count"] > 0)
    checks["flip_lives_in_tightest_band"] = bool(flip_in_tightest)
    checks["selective_identity_monotonic_in_eps"] = bool(
        sel_by_eps[_band_key(BAND_THRESHOLDS[0])] <= sel_by_eps[_band_key(BAND_THRESHOLDS[1])]
        <= sel_by_eps[_band_key(BAND_THRESHOLDS[2])])
    checks["selective_identity_widest_band_eq_1"] = bool(
        sel_by_eps[_band_key(BAND_THRESHOLDS[-1])] >= 1.0 - IDENTITY_EPS)
    checks["eps_star_resolved"] = bool(eps_star is not None)
    checks["recovery_consistent_with_eps_star"] = bool(
        (eps_star is not None) == recovers)

    # cost-model arithmetic
    checks["blanket_cost_is_eta_attn"] = bool(abs(blanket_cost - FA_SLIDING0_TPS_COST) < 1e-9)
    checks["selective_cost_finite_nonneg"] = bool(
        math.isfinite(selective_cost) and selective_cost >= 0.0)
    checks["cheaper_flag_consistent"] = bool(
        cheaper == (math.isfinite(selective_cost) and selective_cost < blanket_cost))

    n_checks = len(checks)
    return checks, n_checks


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
    report_path = OUT_DIR / "knifeedge_identity_tiebreak_results.json"
    json.dump(report, open(report_path, "w"), indent=2)
    _print_console(report)
    if not a.no_wandb:
        log_wandb(report, a)


def _print_console(report: dict) -> None:
    print("\n========== KNIFE-EDGE IDENTITY TIE-BREAK (PR #397) ==========", flush=True)
    print(f" VERDICT                                  : {report['verdict']}", flush=True)
    print(f" flip_count_at_decode (primary={report['primary_arm']}) : {report['flip_count_at_decode']} "
          f"(max gap {report['max_flip_gap']}, tightest_band={report['flip_in_tightest_band']})", flush=True)
    print(f" band counts pos <=.125/.25/.5            : "
          f"{report['knifeedge_position_count_0p125']}/{report['knifeedge_position_count_0p25']}/"
          f"{report['knifeedge_position_count_0p5']} of {report['total_positions']}", flush=True)
    print(f" eps_star (covers all flips)              : {report['eps_star']}", flush=True)
    print(f" tiebreak_recovers_identity_1p0           : {report['tiebreak_recovers_identity_1p0']} "
          f"(id-tiebreak recovers all flips: {report['id_tiebreak_recovers_all_flips']})", flush=True)
    print(f" selective vs blanket TPS cost            : {report['selective_fix_tps_cost']:.4f} vs "
          f"{report['fa_sliding0_tps_cost']:.2f}  (cheaper={report['selective_cheaper_than_blanket']}, "
          f"{report['cheapness_ratio_blanket_over_selective']:.2f}x)", flush=True)
    print(f" f_step / f_position at eps_star          : {report['f_step_band_at_eps_star']:.4f} / "
          f"{report['f_position_band_at_eps_star']:.4f}", flush=True)
    print(f" pessimistic full-fwd eta (vs #364 9.841%): {report['pessimistic_full_forward_eta']:.4f} "
          f"(corroborates_364_wall={report['corroborates_364_precision_wall']})", flush=True)
    print(f" targeted_identity_recovery_viable        : {report['targeted_identity_recovery_viable']}", flush=True)
    print(f" SELF-TEST PASSES (PRIMARY)               : {report['knifeedge_tiebreak_self_test_passes']} "
          f"({sum(report['self_test'].values())}/{report['self_test_n_checks']} checks)", flush=True)
    fails = [k for k, v in report["self_test"].items() if not v]
    if fails:
        print(f"   self-test FAILS: {fails}", flush=True)
    print(f" report -> {OUT_DIR / 'knifeedge_identity_tiebreak_results.json'}", flush=True)
    print("=============================================================\n", flush=True)


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
        notes="PR#397 knife-edge identity tie-break: can a position-selective deterministic tie-break "
              "recover e2e greedy identity 1.0 cheaper than the ~11-TPS FA_SLIDING=0 blanket?",
        config={
            "pr": 397, "M_verify": M_VERIFY, "n_prompts": report["n_prompts"],
            "C": report["C"], "n_verify": report["n_verify"], "model_dir": report["model_dir"],
            "primary_arm": report["primary_arm"],
            "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
            **{f"anchor/{k}": v for k, v in report["imported_anchors"].items()},
        },
    )
    if run is None:
        print("[wandb] disabled (no API key/mode); results saved to JSON only", flush=True)
        return

    summary = {
        # PRIMARY + required deliverables
        "knifeedge_tiebreak_self_test_passes": report["knifeedge_tiebreak_self_test_passes"],
        "knifeedge_position_count_0p125": report["knifeedge_position_count_0p125"],
        "knifeedge_position_count_0p25": report["knifeedge_position_count_0p25"],
        "knifeedge_position_count_0p5": report["knifeedge_position_count_0p5"],
        "flip_count_at_decode": report["flip_count_at_decode"],
        "flip_in_tightest_band": report["flip_in_tightest_band"],
        "tiebreak_recovers_identity_1p0": report["tiebreak_recovers_identity_1p0"],
        "id_tiebreak_recovers_all_flips": report["id_tiebreak_recovers_all_flips"],
        "selective_fix_tps_cost": report["selective_fix_tps_cost"],
        "fa_sliding0_tps_cost": report["fa_sliding0_tps_cost"],
        "selective_cheaper_than_blanket": report["selective_cheaper_than_blanket"],
        "cheapness_ratio_blanket_over_selective": report["cheapness_ratio_blanket_over_selective"],
        "targeted_identity_recovery_viable": report["targeted_identity_recovery_viable"],
        # cost-model detail
        "eps_star": report["eps_star"],
        "max_flip_gap": report["max_flip_gap"],
        "f_step_band_at_eps_star": report["f_step_band_at_eps_star"],
        "f_position_band_at_eps_star": report["f_position_band_at_eps_star"],
        "pessimistic_full_forward_eta": report["pessimistic_full_forward_eta"],
        "decodewidth_lowmargin_background_0p5": report["decodewidth_lowmargin_background_0p5"],
        "corroborates_364_precision_wall": report["corroborates_364_precision_wall"],
        "total_positions": report["total_positions"],
        "n_chunks_primary": report["n_chunks_primary"],
        "verdict": report["verdict"],
        "verdict_green": report["verdict"].startswith("GREEN"),
        "verdict_amber": report["verdict"].startswith("AMBER"),
        "verdict_red": report["verdict"].startswith("RED"),
        # scope flags
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
        "self_test_n_checks": report["self_test_n_checks"],
    }
    for k, v in report["selective_identity_by_eps"].items():
        summary[f"selective_identity_eps_{k}"] = v
    for arm in ARMS:
        d = report["arms"][arm]
        b = d["bands"]
        summary[f"{arm}/identity"] = d["decodewidth_e2e_token_identity_rate"]
        summary[f"{arm}/flip_count"] = b["flip_count"]
        summary[f"{arm}/position_count_0p125"] = b["position_count_0p125"]
        summary[f"{arm}/position_count_0p25"] = b["position_count_0p25"]
        summary[f"{arm}/position_count_0p5"] = b["position_count_0p5"]
        summary[f"{arm}/step_count_0p125"] = b["step_count_0p125"]
        summary[f"{arm}/position_frac_0p5"] = b["position_frac_0p5"]
        summary[f"{arm}/max_flip_gap"] = b["max_flip_gap"]
        summary[f"{arm}/marlin_bitexact_at_8"] = bool(
            d["marlin_sizem_diag"].get("bitexact_at_decode_width"))
        summary[f"{arm}/aten_mm_bitexact"] = bool(d["aten_mm_control"].get("bitexact_M1_vs_M8"))
    for k, v in summary.items():
        run.summary[k] = v
    for k, v in report["self_test"].items():
        run.summary[f"selftest/{k}"] = v
    run.summary["verdict_text"] = report["verdict"]
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
    ap.add_argument("--eval-prompts", dest="n_prompts", type=int, default=128)
    ap.add_argument("--n-prompts", dest="n_prompts", type=int, default=128)
    ap.add_argument("--ctx-len", dest="ctx_len", type=int, default=224)
    ap.add_argument("--n-verify", dest="n_verify", type=int, default=M_VERIFY)
    ap.add_argument("--gpu-mem-util", type=float, default=0.55)
    ap.add_argument("--max-batched-tokens", type=int, default=8192)
    ap.add_argument("--verbose-k", dest="verbose_k", type=int, default=3)
    ap.add_argument("--wandb_group", dest="wandb_group", default="knifeedge-identity-tiebreak")
    ap.add_argument("--wandb_name", dest="wandb_name", default="stark/knifeedge-identity-tiebreak")
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
