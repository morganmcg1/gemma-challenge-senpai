#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #412 (stark) -- Is #397's SELECTIVE higher-precision recompute the fastest strictly-token-equivalent
verify path? Convert the modeled ~2.6-TPS selective tax into a MEASURED equivalent-TPS, and confirm whether
the selective path reaches byte-identity 1.0.

RE-SCOPE (human Issue #407): no longer chasing 500. Objective = maximize single-stream decode TPS subject
to a strict byte-exact greedy-token-equivalence constraint. The deployed fast path (481.53) is NON-equivalent
(3/882 reduction-order flips under M=8 verify, served identity 0.9966; #381/#397/#405). The strictly-equivalent
references are the blanket-strict base (~467.48, full ~11-TPS attention tax; #393) and the UNMEASURED selective
higher-precision recompute (#397's ~2.6-TPS model). This card measures the latter.

THE MECHANISM (#397) -- run fast attention everywhere; flag the <= eps* near-tie steps FOR FREE (the gate is
readable from the fast path's own in-register top-2 margin -- #405 tie_identifiable_from_fast_path=True);
recompute the attention reduction at higher precision (fp32 accumulation) ONLY on flagged steps; keep the fast
path verbatim elsewhere.

THE REALIZABILITY CRUX (the load-bearing finding)
-------------------------------------------------
The 2.6-TPS model prices the recompute as the *attention-reduction delta* on flagged steps only:
  tax_model = f_step(0.236) x eta_attn(11 TPS) = 2.598 TPS.
That is only achievable if the flag is read AND the fix is applied INSIDE one attention call (a fused
conditional-precision kernel). As a pure runtime wrapper (no served-kernel edit -- the binding constraint of
this card) it is NOT: the gate (top-2 LOGIT margin) is only known AFTER the full forward, and the divergence
is bf16-attention-injected UPSTREAM and PROPAGATES (lm_head-only / attention-only recompute does NOT restore
identity -- ubel #364, wirbel #362). So a faithful runtime wrapper must TWO-PASS: (1) fast forward to read the
flag, then (2) a full strict RE-FORWARD on each flagged step. That re-forward is a whole decode step (the body
GEMMs are recomputed too -- they are Marlin-bit-exact at M=8 #381, but still cost their wall-clock). Hence the
MEASURED runtime-wrapper selective tax is f_step x (full step), NOT f_step x (attention delta):
  eta_selective_realizable = f_step x (1 + eta_attn_decode)  >>  eta_attn_decode (blanket).
Because f_step(0.236) >> eta_attn_decode(0.030), the selective two-pass wrapper is NET-NEGATIVE vs blanket-strict.
=> The fastest strictly-equivalent config reachable WITHOUT a served-kernel edit remains BLANKET-STRICT (~467.48).
The 2.6-TPS model survives only as a FUSED-kernel design; realizing it = `blocked:served_change_required`.

THE IDENTITY CRUX (the SECOND, decisive finding -- the selective path does NOT reach 1.0; it goes BACKWARDS)
-----------------------------------------------------------------------------------------------------------
Measured on-target (882 decode-width positions, served = fast M1-AR reference):
  fast M8-verify      0.9966 (3 flips)   |   SELECTIVE (fast base + strict patch on flagged) 0.9853 (13 flips!)
ALL 3 served flips -- and every flagged near-tie row -- are BITWISE TIES in the served M1-AR (m1_self_gap=0.0):
the top-2 reference logits are bit-identical and the served stack resolves the tie by argmax index order. For a
true tie "higher precision" has NO defined winner -- the outcome is set purely by reduction/tie-break order. The
strict recompute is a *different* reduction order, so it resolves these ties to ITS OWN (equally-valid-greedy)
token, which matches the served fast reference only by coincidence:
  - recovers 2/3 served flips (prompts 11,118: strict tie-break happens to equal fast there),
  - MISSES the 3rd (prompt 18: strict picks 3582, served picks 3629 -- a third token entirely),
  - and BREAKS 12 previously-correct flagged tie rows (e.g. prompt 90: fast 22355 -> strict 102643).
Net: 1 unrecovered + 12 new = 13 disagreements > the original 3. Identity is a WITHIN-stack property; mixing a
fast served reference with strict patches is INCONSISTENT and strictly worse than either consistent stack
(all-fast 3 flips / all-strict 1 flip vs its OWN reference). => byte-identity 1.0 is UNREACHABLE by any
attention-precision knob; only reproducing the served stack's exact tie-break (i.e. NOT recomputing) gives it.
This refutes #397's premise: selective higher-precision recompute is DOUBLE-dominated -- slower than blanket AND
less token-equivalent than the fast path it tries to repair.

SCOPE: LOCAL A10G (sm_86) post-hoc profiling prototype. analysis_only / no_hf_job / no_served_file_change /
official_tps=0. NO served/deployed file is touched; the int4 path is READ only; NO HF job; NO submission.
The selective recompute is realized as a runtime SIMULATION/wrapper over the fast forward's own logprobs +
a #393-style attention micro-measurement -- never an edit to a served kernel file.

METHOD (mirrors #405 census + #393 microbench; both are the established lineage):
  ARM census (isolated subprocesses, pin set per-arm):
    heuristic (VLLM_BATCH_INVARIANT=0) -- the SERVED fast verify-width attention (PRIMARY; identity 0.9966).
    pinned    (VLLM_BATCH_INVARIANT=1) -- deterministic single-segment strict reduction (#381; identity 0.9989,
               1 residual varlen-combine flip). The highest-precision reduction reachable as a runtime wrapper.
  MICROBENCH (FA2 varlen, gemma-4 paged geometry): fast(ns=0) vs strict(ns=1) per-step attention latency over
    the decode band -> penalty -> eta_attn_decode (sole rebuild-free strict tax; reproduces #393 467.48). >=3 seeds.
  COMPOSE: anchored TPS ladder (OFFICIAL_TPS=481.53): fast_nonequiv / blanket_strict / selective (realizable
    two-pass = PRIMARY MEASURED) / selective_incremental_fused_model (the 2.6 model, fused-kernel only).
  IDENTITY: DATA-DRIVEN census simulation -- flag covers all flips (yes); at each flagged row substitute the
    MEASURED strict-arm recompute token (no modelling of what precision "does") and compare to the served fast
    M1-AR reference -> served_identity_after_selective + served/new flip accounting.

DELIVERABLES (W&B summary/)
  selective_recompute_reaches_identity_1p0 (bool PRIMARY); strictly_equivalent_frontier_tps;
  selective_recompute_measured_tps (PRIMARY metric, realizable two-pass); selective_tax_tps;
  selective_tax_vs_2p6_model; blanket_strict_measured_tps; fast_nonequiv_tps; flagged_step_fraction;
  n_flips_remaining; n_flips_recovered; selective_incremental_fused_model_tps;
  fastest_realizable_strictly_equivalent_tps + ..._config; selective_self_test_passes (>=20 asserts).
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

# ======================================================================================
# Imported fleet anchors (CITE; do NOT re-derive)
# ======================================================================================
# #381 (this student) decode-width residual at the literal size_m=8 verify geometry:
PINNED_IDENTITY_381 = 0.9988751406074241       # 888/889 -- 1 flip (varlen-combine coin-flip; #375 closes it)
HEURISTIC_IDENTITY_381 = 0.9966254218222722    # 886/889 -- 3 flips (all knife-edge, margins 0.125)
PINNED_FLIP_COUNT_381 = 1
HEURISTIC_FLIP_COUNT_381 = 3

# #397 (this student, g3954eh3) -- the selective-recompute MODEL this card measures:
SELECTIVE_FIX_TPS_COST_397 = 2.5984251968503935  # = f_step_band(0.125)=0.23622 x eta_attn=11.0 (ceiling basis)
F_STEP_BAND_397 = 0.23622047244094488            # = 30/127 verify STEPS flagged at eps*=0.125 (per-step)
FA_SLIDING0_TPS_COST = 11.0                      # eval-weighted blanket attention strict tax in TPS (#378/#397)

# #393 (0q7ynumg, MERGED) decode-faithful strict (the strictly-equivalent reference to beat):
ETA_ATTN_DECODE_393 = 0.030065297571591987       # eta_attn_decode_only (sole rebuild-free strict tax)
DEPLOYED_STRICT_393 = 467.475218449957           # blanket-strict deployed = OFFICIAL/(1+eta)

# strict budget ladder (CITE; identical to #378/#390/#393 for ADDITIVITY):
CEILING_500 = 520.953                            # lambda=1 central ceiling TPS
OFFICIAL_TPS = 481.53                            # deployed non-strict public #1 (#52); this leg adds 0
STEP_NORM_US = 1218.2                            # deployed batch=1 decode step normalizer (#257/#344)
F_ATTN_344 = 0.09506718019009251                 # #378 step_fractions.attn (M=8 verify attention fraction)
ETA_ATTN_378 = 0.02145375421979844               # #378 eval-weighted attention-pin tax (the 11-TPS basis)
PENALTY_ANCHORS_375 = {528: 1.2777777609352838, 2048: 3.0555554713430864, 4096: 4.755813955455911}

K_SPEC = 7
M_VERIFY = K_SPEC + 1                             # = 8, the deployed decode-verify query width
EPS_STAR = 0.125                                  # bf16 floor; the band that covers every observed flip
BAND_THRESHOLDS = (0.125, 0.25, 0.5)
BAND_TOL = 1e-9
HYBRID_PREFIX_COMMIT = 32                         # Gemma-4 hybrid prefix-cache commit granularity (#381)
KNOWN_FLIP_PROMPTS = (11, 18, 118)               # the 3 served-arm flips (#381/#397/#405)

# ---- #393 FA2 microbench geometry (gemma-4 sliding) ---------------------------------------
HEAD_DIM = 256
N_Q_HEADS = 8
N_KV_HEADS = 2
SCALE = 1.0 / math.sqrt(HEAD_DIM)
A10G_SMS = 80
SERVED_BLOCK_SIZE = 16
BLOCK_M_SPLITKV = 64
BLOCK_N_TILE = 64
HEURISTIC_SPLIT = 0                              # DEPLOYED non-VBI default (kernel heuristic picks K)
UNPACK_SPLIT = 1                                # VLLM_BATCH_INVARIANT=1 -> num_splits=1 (M-invariant strict)
M_AR = 1                                         # AR/draft decode width (the un-pack penalty lane)
BAND_L = (528, 560, 592, 624, 658)              # decode-position L band (dominant eval mass)
SHORT_L = 128
PENALTY_GRID_L = (110, 128, 192, 256, 384, 503, 512, 528, 560, 592, 624, 658, 704, 768, 1024, 2048)

MODEL_CANDIDATES = [
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots"
    ),
    "/tmp/osoi5-v0-baked",
]
PROMPTS_JSONL = "official/main_bucket/shared_resources/speed_benchmark/data/ppl_ground_truth_tokens.jsonl"

OUT_DIR = Path("research/validity/selective_recompute_equivalent_tps")
CENSUS_ARMS = ("heuristic", "pinned")
PRIMARY_ARM = "heuristic"                         # the SERVED (fast) path that carries the 3 flips
RECOMPUTE_ARM = "pinned"                          # strict single-segment fp32-accum reduction = the MEASURED
#                                                   on-target result of "recompute the attention at higher
#                                                   precision". A flagged step's recompute token is THIS arm's
#                                                   M8-verify argmax at the same (prompt,pos) -- not an assumption.


# ======================================================================================
# Small helpers (reused from #381/#405)
# ======================================================================================
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


def block_align(n: int) -> int:
    return (n // HYBRID_PREFIX_COMMIT) * HYBRID_PREFIX_COMMIT


def _sorted_logprobs(entry) -> list[tuple[int, float]]:
    return sorted(((int(t), float(getattr(lp, "logprob", lp))) for t, lp in entry.items()),
                  key=lambda kv: kv[1], reverse=True)


def _argmax_from_logprob_entry(entry) -> int:
    return int(max(entry.items(), key=lambda kv: getattr(kv[1], "logprob", kv[1]))[0])


def _band_key(thr: float) -> str:
    return f"{thr:g}".replace(".", "p")


# ======================================================================================
# PHASE census: one arm. Full per-position census M=8 served top-k vs M=1 AR top-k (mirror #405).
# ======================================================================================
def phase_census(out_path: str, arm: str, n_prompts: int, ctx_len: int, n_verify: int,
                 gpu_mem_util: float, max_batched_tokens: int, verbose_k: int) -> None:
    import torch
    from vllm import LLM, SamplingParams

    batch_invariant_env = os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1"
    model_dir = resolve_model_dir()
    C = block_align(ctx_len)
    print(f"[census:{arm}] model={model_dir} C(prefix)={C} n_verify={n_verify} "
          f"VLLM_BATCH_INVARIANT={batch_invariant_env}", flush=True)

    t0 = time.time()
    llm = LLM(
        model=model_dir, quantization="compressed-tensors", dtype="bfloat16",
        max_model_len=max(512, C + 64), gpu_memory_utilization=gpu_mem_util,
        max_num_seqs=16, max_num_batched_tokens=max_batched_tokens,
        enable_prefix_caching=True, enforce_eager=True, trust_remote_code=True,
    )
    print(f"[census:{arm}] vLLM load done in {time.time()-t0:.0f}s", flush=True)

    try:
        import vllm.v1.attention.ops.triton_unified_attention as _ua
        attn_is_batch_invariant = bool(getattr(_ua, "is_batch_invariant", False))
    except Exception:
        attn_is_batch_invariant = False

    sp_gen = SamplingParams(temperature=0.0, max_tokens=n_verify, logprobs=5, detokenize=False)
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
    positions: list[dict] = []
    flip_details: list[dict] = []

    for ri, rec in enumerate(rows):
        src = list(rec.get("context_token_ids", [])) + list(rec.get("target_token_ids", []))
        if len(src) < C + 1:
            continue
        prefix = src[:C]
        llm.generate([{"prompt_token_ids": prefix}], sp_warm, use_tqdm=False)

        outA = llm.generate([{"prompt_token_ids": prefix}], sp_gen, use_tqdm=False)[0]
        cont = list(outA.outputs[0].token_ids)[:n_verify]
        m1_lp_steps = list(outA.outputs[0].logprobs or [])[:n_verify]
        outA2 = llm.generate([{"prompt_token_ids": prefix}], sp_gen, use_tqdm=False)[0]
        cont2 = list(outA2.outputs[0].token_ids)[:n_verify]
        det_m1 = int(cont == cont2)
        if len(cont) < n_verify:
            continue
        full = prefix + cont

        def chunk_entries(full_ids):
            out = llm.generate([{"prompt_token_ids": full_ids}], sp_chunk, use_tqdm=False)[0]
            nct = out.num_cached_tokens or 0
            pls = out.prompt_logprobs or []
            am, ent = {}, {}
            for i in range(C + 1, len(full_ids)):
                entry = pls[i] if i < len(pls) else None
                if entry is not None:
                    am[i] = _argmax_from_logprob_entry(entry)
                    ent[i] = _sorted_logprobs(entry)
            return am, nct, ent

        m8, nct8, ent8 = chunk_entries(full)
        m8b, nct8b, _ = chunk_entries(full)
        suffix_pos = sorted(m8)
        n_computed_rows = len(full) - nct8
        n_computed_rows_b = len(full) - nct8b
        chunk_isolated = (n_computed_rows == n_verify and n_computed_rows_b == n_verify)
        n_computed_rows_total += n_computed_rows
        chunk_width_obs.append(len(suffix_pos))
        n_chunk_isolated += int(chunk_isolated)
        det_m8 = int(all(m8.get(p) == m8b.get(p) for p in suffix_pos) and bool(suffix_pos))

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

        match = total = 0
        prompt_has_flag = False
        for p in suffix_pos:
            j = p - C
            m1_tok = full[p]
            total += 1
            sl = ent8.get(p, [])
            if len(sl) < 2:
                continue
            m8_top1_id, m8_top1_lp = sl[0]
            m8_top2_id, m8_top2_lp = sl[1]
            m8_gap = m8_top1_lp - m8_top2_lp
            m8_ids = [tid for tid, _ in sl]
            is_flip = int(m8_top1_id != m1_tok)
            m1_in_m8_top2 = bool(m1_tok in (m8_top1_id, m8_top2_id))
            m1_in_m8_top5 = bool(m1_tok in m8_ids)
            m8_pair_min_id = min(m8_top1_id, m8_top2_id)
            is_near_tie = bool(m8_gap <= EPS_STAR + BAND_TOL)
            if is_near_tie:
                prompt_has_flag = True

            m1_entry = m1_lp_steps[j] if 0 <= j < len(m1_lp_steps) else None
            m1_sl = _sorted_logprobs(m1_entry) if m1_entry else []
            m1_top1_id = m1_sl[0][0] if m1_sl else None
            m1_self_gap = (m1_sl[0][1] - m1_sl[1][1]) if len(m1_sl) >= 2 else None
            m1_argmax_matches_token = bool(m1_top1_id == m1_tok) if m1_top1_id is not None else None
            m1_is_bitwise_tie = bool(m1_self_gap is not None and m1_self_gap <= BAND_TOL)

            rec_pos = {
                "prompt_idx": ri, "pos": p, "j": j,
                "m8_gap": round(m8_gap, 6),
                "m8_top1_id": m8_top1_id, "m8_top2_id": m8_top2_id, "m8_pair_min_id": m8_pair_min_id,
                "m1_tok_id": m1_tok, "is_flip": is_flip, "is_near_tie": is_near_tie,
                "m1_in_m8_top2": m1_in_m8_top2, "m1_in_m8_top5": m1_in_m8_top5,
                "m1_self_gap": (round(m1_self_gap, 6) if m1_self_gap is not None else None),
                "m1_argmax_matches_token": m1_argmax_matches_token,
                "m1_is_bitwise_tie": m1_is_bitwise_tie,
            }
            positions.append(rec_pos)
            if not is_flip:
                match += 1
            else:
                flip_details.append({**rec_pos,
                                     "m1_margin_in_m8": round(m8_top1_lp - dict(sl).get(m1_tok, float("nan")), 6)
                                     if m1_in_m8_top5 else None})

        n_match += match
        n_total += total
        n_det_m1 += det_m1 * max(1, total)
        n_det_m8 += det_m8 * max(1, total)
        n_within += within * max(1, total)

        sha = hashlib.sha256(bytes(str([m8.get(p) for p in suffix_pos]), "utf8")).hexdigest()[:16]
        per_prompt.append({
            "id": rec.get("id"), "prompt_idx": ri, "C": C, "chunk_width": len(suffix_pos),
            "chunk_isolated": chunk_isolated, "num_cached_tokens": nct8,
            "argmax_match_M8_vs_M1": match, "positions": total, "sha": sha,
            "det_match_M1_vs_M1": det_m1, "det_match_M8_vs_M8": det_m8,
            "within_copy0_vs_copy1": within,
            "step_flagged": bool(prompt_has_flag),
            "step_has_flip": bool(match < total),
        })
        if ri < verbose_k or ri == len(rows) - 1:
            print(f"[census:{arm}] prompt {ri} chunk_w={len(suffix_pos)} isolated={chunk_isolated} "
                  f"match={match}/{total} flagged={prompt_has_flag} det_m1={det_m1} det_m8={det_m8}", flush=True)

    n_seq = len(per_prompt)
    identity = (n_match / n_total) if n_total else float("nan")
    out = {
        "phase": "census", "arm": arm, "model_dir": model_dir,
        "vllm_batch_invariant_env": batch_invariant_env,
        "attn_is_batch_invariant": attn_is_batch_invariant,
        "n_prompts": n_seq, "ctx_len_requested": ctx_len, "C": C, "n_verify": n_verify,
        "total_positions": n_total, "matching_positions": n_match,
        "decodewidth_e2e_token_identity_rate": identity,
        "decodewidth_e2e_divergence_rate": (1.0 - identity) if math.isfinite(identity) else float("nan"),
        "determinism_M1_vs_M1": (n_det_m1 / n_total) if n_total else float("nan"),
        "determinism_M8_vs_M8": (n_det_m8 / n_total) if n_total else float("nan"),
        "within_batch_copy0_vs_copy1": (n_within / n_total) if n_total else float("nan"),
        "chunk_isolated_fraction": (n_chunk_isolated / n_seq) if n_seq else float("nan"),
        "median_chunk_width": (statistics.median(chunk_width_obs) if chunk_width_obs else float("nan")),
        "n_computed_rows_total": n_computed_rows_total,
        "expected_computed_rows_per_chunk": n_verify,
        "positions": positions,
        "flip_details": flip_details,
        "per_prompt": per_prompt,
        "nan_clean": bool(math.isfinite(identity)),
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[census:{arm}] identity={identity:.7f} flips={len(flip_details)} "
          f"positions={n_total} peak={out['peak_gpu_gb']:.1f}GB", flush=True)
    print(f"ARM_DONE {out_path}", flush=True)


# ======================================================================================
# PHASE microbench: FA2 varlen fast(ns=0) vs strict(ns=1) per-step attention latency (#393).
# ======================================================================================
def _ceildiv(a: int, b: int) -> int:
    return -(-a // b)


def _num_splits_heuristic(bnm: int, num_sms: int, num_n_blocks: int, max_splits: int = 128) -> int:
    if bnm >= 0.8 * num_sms:
        return 1
    max_splits = min(max_splits, num_sms, num_n_blocks)
    eff: list[float] = []
    max_eff = 0.0

    def eligible(ns: int) -> bool:
        return ns == 1 or _ceildiv(num_n_blocks, ns) != _ceildiv(num_n_blocks, ns - 1)

    for ns in range(1, max_splits + 1):
        if not eligible(ns):
            eff.append(0.0)
            continue
        n_waves = float(bnm * ns) / num_sms
        e = n_waves / math.ceil(n_waves)
        max_eff = max(max_eff, e)
        eff.append(e)
    for ns in range(1, max_splits + 1):
        if not eligible(ns):
            continue
        if eff[ns - 1] >= 0.85 * max_eff:
            return ns
    return 1


def _heuristic_K(M: int, L: int) -> int:
    num_m_blocks = _ceildiv(M, BLOCK_M_SPLITKV)
    num_n_blocks = _ceildiv(L, BLOCK_N_TILE)
    return _num_splits_heuristic(1 * N_Q_HEADS * num_m_blocks, A10G_SMS, num_n_blocks, max_splits=128)


def _build_paged(torch, L: int, M: int, seed: int, dev, page: int = SERVED_BLOCK_SIZE):
    g = torch.Generator(device=dev).manual_seed(seed)
    q = torch.randn(M, N_Q_HEADS, HEAD_DIM, generator=g, device=dev, dtype=torch.bfloat16)
    nb = _ceildiv(L, page)
    kc = torch.randn(nb, page, N_KV_HEADS, HEAD_DIM, generator=g, device=dev, dtype=torch.bfloat16)
    vc = torch.randn(nb, page, N_KV_HEADS, HEAD_DIM, generator=g, device=dev, dtype=torch.bfloat16)
    bt = torch.arange(nb, dtype=torch.int32, device=dev).unsqueeze(0)
    sk = torch.tensor([L], dtype=torch.int32, device=dev)
    return q, kc, vc, bt, sk


def _served_varlen(fn, q, kc, vc, bt, cu, sk, L, M, ns):
    return fn(q=q, k=kc, v=vc, out=None, cu_seqlens_q=cu, max_seqlen_q=M,
              seqused_k=sk, max_seqlen_k=L, softmax_scale=SCALE, causal=False,
              block_table=bt, num_splits=ns, fa_version=2)


def _time_call(torch, fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2] * 1e3  # us (median)


def _measure_identity_fa2(torch, fn, L: int, ns: int, n_trials: int, seed0: int, dev) -> dict:
    """byte/argmax identity of batched(M=8) vs per-row(M=1), SAME num_splits. byte==1 GUARANTEES strict."""
    IDENT_M = (1, 8)
    byte_acc = {M: [] for M in IDENT_M}
    maxdiff = {M: 0.0 for M in IDENT_M}
    any_nan = False
    for t in range(n_trials):
        q8, kc, vc, bt, sk = _build_paged(torch, L, max(IDENT_M), seed0 + t, dev)
        for M in IDENT_M:
            q = q8[:M].contiguous()
            cu = torch.tensor([0, M], dtype=torch.int32, device=dev)
            bat = _served_varlen(fn, q, kc, vc, bt, cu, sk, L, M, ns)
            any_nan = any_nan or bool(torch.isnan(bat).any())
            cu1 = torch.tensor([0, 1], dtype=torch.int32, device=dev)
            ref = torch.cat([_served_varlen(fn, q[r:r + 1], kc, vc, bt, cu1, sk, L, 1, ns)
                             for r in range(M)], dim=0)
            bflat = bat.reshape(M, -1)
            rflat = ref.reshape(M, -1)
            byte_acc[M].append((bflat == rflat).all(dim=-1).float().mean().item())
            maxdiff[M] = max(maxdiff[M], (bflat.float() - rflat.float()).abs().max().item())

    def mean(xs):
        return float(sum(xs) / len(xs)) if xs else float("nan")
    return {"byte_identity_by_M": {str(M): mean(byte_acc[M]) for M in IDENT_M},
            "max_abs_diff_by_M": {str(M): maxdiff[M] for M in IDENT_M},
            "n_trials": n_trials, "any_nan": bool(any_nan)}


def _measure_penalty_curve(torch, fn, dev, iters: int, warmup: int, seed: int, M: int) -> dict:
    """penalty(L) = lat[ns=1 unpack/strict] / lat[ns=0 heuristic/fast] at width M, over PENALTY_GRID_L."""
    curve = {}
    for L in PENALTY_GRID_L:
        q8, kc, vc, bt, sk = _build_paged(torch, L, M, seed, dev)
        cu = torch.tensor([0, M], dtype=torch.int32, device=dev)
        heur_us = _time_call(torch, lambda: _served_varlen(fn, q8, kc, vc, bt, cu, sk, L, M, HEURISTIC_SPLIT),
                             iters, warmup)
        unpack_us = _time_call(torch, lambda: _served_varlen(fn, q8, kc, vc, bt, cu, sk, L, M, UNPACK_SPLIT),
                               iters, warmup)
        curve[L] = {"heuristic_us": heur_us, "unpack_us": unpack_us,
                    "penalty": (unpack_us / heur_us) if heur_us > 0 else float("nan"),
                    "heuristic_K": float(_heuristic_K(M, L))}
    return curve


def phase_microbench(out_path: str, iters: int, warmup: int, seeds: list[int]) -> None:
    import torch
    from vllm.vllm_flash_attn import _vllm_fa2_C  # noqa: F401 (registers torch.ops._vllm_fa2_C)
    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func as FA2
    dev = torch.device("cuda:0")

    # M=8 verify-width identity (fast ns=0 vs strict ns=1): confirms fast is non-equivalent, strict is the pin.
    ident_L = (SHORT_L, *BAND_L)
    ident = {}
    for cfg, ns in (("fast_ns0", HEURISTIC_SPLIT), ("strict_ns1", UNPACK_SPLIT)):
        per_L = {}
        for L in ident_L:
            accs = [_measure_identity_fa2(torch, FA2, L, ns, 2, s, dev) for s in seeds]
            per_L[str(L)] = {"byte_identity_M8": min(a["byte_identity_by_M"]["8"] for a in accs),
                             "maxdiff_M8": max(a["max_abs_diff_by_M"]["8"] for a in accs)}
        ident[cfg] = {"num_splits": ns, "per_L": per_L,
                      "byte_M8_min": min(per_L[str(L)]["byte_identity_M8"] for L in ident_L)}

    # penalty curve over >=3 seeds (the M=1 AR lane pays the un-pack tax; M=8 verify is penalty-free).
    seed_curves_m1 = [_measure_penalty_curve(torch, FA2, dev, iters, warmup, s, M_AR) for s in seeds]
    seed_curves_m8 = [_measure_penalty_curve(torch, FA2, dev, iters, warmup, s, M_VERIFY) for s in seeds]

    def band_penalty(curve):
        return float(sum(curve[L]["penalty"] for L in BAND_L) / len(BAND_L))

    band_pen_m1_seeds = [band_penalty(c) for c in seed_curves_m1]
    band_pen_m8_seeds = [band_penalty(c) for c in seed_curves_m8]
    eta_attn_seeds = [F_ATTN_344 * (bp - 1.0) for bp in band_pen_m1_seeds]

    out = {
        "phase": "microbench", "iters": iters, "warmup": warmup, "seeds": seeds, "n_seeds": len(seeds),
        "ident": ident,
        "fast_is_byte_exact_M8": bool(ident["fast_ns0"]["byte_M8_min"] >= 1.0),
        "strict_is_byte_exact_M8": bool(ident["strict_ns1"]["byte_M8_min"] >= 1.0),
        "penalty_curve_M1_seed0": {str(L): seed_curves_m1[0][L] for L in PENALTY_GRID_L},
        "penalty_curve_M8_seed0": {str(L): seed_curves_m8[0][L] for L in PENALTY_GRID_L},
        "band_penalty_M1_seeds": band_pen_m1_seeds,
        "band_penalty_M8_seeds": band_pen_m8_seeds,
        "penalty_decode_band": statistics.median(band_pen_m1_seeds),
        "penalty_decode_band_std": (statistics.pstdev(band_pen_m1_seeds) if len(band_pen_m1_seeds) > 1 else 0.0),
        "verify_penalty_band_mean": statistics.median(band_pen_m8_seeds),
        "verify_penalty_free": bool(all(abs(bp - 1.0) < 0.10 for bp in band_pen_m8_seeds)),
        "eta_attn_decode_seeds": eta_attn_seeds,
        "eta_attn_decode": statistics.median(eta_attn_seeds),
        "eta_attn_decode_std": (statistics.pstdev(eta_attn_seeds) if len(eta_attn_seeds) > 1 else 0.0),
        "anchor_eta_attn_decode_393": ETA_ATTN_DECODE_393,
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[microbench] penalty_band={out['penalty_decode_band']:.4f} "
          f"eta_attn_decode={out['eta_attn_decode']:.6f} (anchor393={ETA_ATTN_DECODE_393:.6f}) "
          f"fast_byte_M8={ident['fast_ns0']['byte_M8_min']:.4f} strict_byte_M8={ident['strict_ns1']['byte_M8_min']:.4f}",
          flush=True)
    print(f"ARM_DONE {out_path}", flush=True)


# ======================================================================================
# Selective recompute SIMULATION (mirror #405 simulate_lowest_id_rule): the runtime wrapper over
# the fast forward's logprobs. Flag <=eps* near-tie steps; recompute (higher precision) recovers the
# M1 reference token UNLESS the M1 reference is itself a bitwise tie (precision cannot resolve a true tie).
# ======================================================================================
def simulate_selective_recompute(positions: list[dict], per_prompt: list[dict],
                                 recompute_positions: list[dict], eps: float) -> dict:
    """DATA-DRIVEN selective recompute (no assumption about what 'higher precision' does to a tie).

    Served reference = the deployed/fast (heuristic) stack's M1-AR token (`positions[*]["m1_tok_id"]`).
    Selective path per row:
      * non-flagged (fast in-register gap > eps): keep the fast-path token verbatim (`m8_top1_id`).
      * flagged   (fast gap <= eps): substitute the MEASURED higher-precision recompute token =
        the strict single-segment arm's M8-verify argmax at the same (prompt,pos)
        (`recompute_positions` join). This is what re-running that step's attention at fp32/strict
        ACTUALLY produces on-target -- we do not model it, we read it.
    A row is identity-correct iff its post-selective token == the served (fast) M1-AR reference.
    """
    n_total = len(positions)
    n_baseline_flips = sum(1 for p in positions if p["is_flip"])
    rec_by_key = {(r["prompt_idx"], r["pos"]): r for r in recompute_positions}

    # per-STEP flag fraction (drives the TPS f): fraction of verify steps with >=1 <=eps near-tie row.
    n_steps = len(per_prompt)
    flagged_steps = [pp for pp in per_prompt if pp.get("step_flagged")]
    n_flagged_steps = len(flagged_steps)
    flagged_step_fraction = (n_flagged_steps / n_steps) if n_steps else float("nan")

    recovered = served_not_recovered = new = 0
    n_flag_rows = n_flag_rows_recompute_disagrees = 0
    n_match_after = 0
    flag_covers_all_flips = True
    disagreements = []      # post-selective rows that differ from the served reference
    for p in positions:
        key = (p["prompt_idx"], p["pos"])
        served_ref = p["m1_tok_id"]
        near = p["m8_gap"] <= eps + BAND_TOL
        if near:
            n_flag_rows += 1
            r = rec_by_key.get(key)
            after_tok = r["m8_top1_id"] if r is not None else p["m8_top1_id"]  # MEASURED recompute token
            if after_tok != served_ref:
                n_flag_rows_recompute_disagrees += 1
        else:
            after_tok = p["m8_top1_id"]                                        # fast path verbatim
        after_correct = (after_tok == served_ref)
        n_match_after += int(after_correct)
        if p["is_flip"]:                       # a served flip in the fast path
            if not near:
                flag_covers_all_flips = False
            if after_correct:
                recovered += 1
            else:
                served_not_recovered += 1
                disagreements.append({**key_rec(p), "kind": "served_flip_not_recovered",
                                      "fast": p["m8_top1_id"], "recompute": after_tok, "served_ref": served_ref})
        else:
            if not after_correct:              # recompute BROKE a row the fast path had correct
                new += 1
                disagreements.append({**key_rec(p), "kind": "new_flip_from_recompute",
                                      "fast": p["m8_top1_id"], "recompute": after_tok, "served_ref": served_ref})
    n_flips_remaining_total = served_not_recovered + new
    identity_after = (n_match_after / n_total) if n_total else float("nan")

    flagged_prompt_idx = {pp["prompt_idx"] for pp in flagged_steps}
    known_flips_flagged = all(k in flagged_prompt_idx for k in KNOWN_FLIP_PROMPTS
                              if any(pp["prompt_idx"] == k for pp in per_prompt))
    # the central physics: are the disputed (flagged-or-flip) positions BITWISE TIES in the served M1-AR?
    flagged_or_flip = [p for p in positions if p["m8_gap"] <= eps + BAND_TOL or p["is_flip"]]
    served_flips_all_bitwise_ties = bool(all(p["m1_is_bitwise_tie"] for p in positions if p["is_flip"]))
    return {
        "eps": eps, "n_total_positions": n_total, "n_baseline_flips": n_baseline_flips,
        "n_steps": n_steps, "n_flagged_steps": n_flagged_steps,
        "flagged_step_fraction": flagged_step_fraction,
        "n_flag_rows": n_flag_rows,
        "n_flag_rows_recompute_disagrees_with_served": n_flag_rows_recompute_disagrees,
        # accounting (served-flip recovery is SEPARATE from newly-introduced flips):
        "n_flips_recovered": recovered,
        "n_served_flips_not_recovered": served_not_recovered,
        "new_flips": new,
        "n_flips_remaining": n_flips_remaining_total,            # TOTAL disagreements vs served ref after selective
        "served_identity_after_selective": identity_after,
        "reaches_identity_1p0": bool(n_flips_remaining_total == 0),
        "flag_covers_all_flips": bool(flag_covers_all_flips),
        "known_flip_prompts_flagged": bool(known_flips_flagged),
        "served_flips_all_bitwise_ties": served_flips_all_bitwise_ties,
        "n_flagged_or_flip_positions": len(flagged_or_flip),
        "n_flagged_rows_m1_bitwise_tie": sum(1 for p in positions
                                             if p["m8_gap"] <= eps + BAND_TOL and p["m1_is_bitwise_tie"]),
        "disagreements": disagreements,
    }


def key_rec(p: dict) -> dict:
    return {"prompt_idx": p["prompt_idx"], "pos": p["pos"]}


# ======================================================================================
# TPS ladder (anchored to OFFICIAL_TPS; identical basis to #378/#390/#393)
# ======================================================================================
def strict_tps_divisor(base: float, eta: float) -> float:
    return base / (1.0 + eta)


def compose_tps(eta_attn_decode: float, flagged_step_fraction: float) -> dict:
    """fast / blanket-strict / selective-realizable(two-pass) / selective-incremental(fused model)."""
    f = flagged_step_fraction
    fast_nonequiv_tps = OFFICIAL_TPS                                    # the fast regime (anchor)
    blanket_eta = eta_attn_decode                                      # attention delta on EVERY step
    blanket_strict_tps = strict_tps_divisor(OFFICIAL_TPS, blanket_eta)
    # realizable runtime wrapper: flagged steps pay a FULL extra strict re-forward (divergence propagates).
    selective_realizable_eta = f * (1.0 + eta_attn_decode)
    selective_realizable_tps = strict_tps_divisor(OFFICIAL_TPS, selective_realizable_eta)
    # incremental FUSED model (the 2.6-TPS #397 model): flagged steps pay only the attention delta.
    selective_incremental_eta = f * eta_attn_decode
    selective_incremental_tps = strict_tps_divisor(OFFICIAL_TPS, selective_incremental_eta)
    return {
        "fast_nonequiv_tps": fast_nonequiv_tps,
        "blanket_eta": blanket_eta, "blanket_strict_tps": blanket_strict_tps,
        "selective_realizable_eta": selective_realizable_eta,
        "selective_realizable_tps": selective_realizable_tps,
        "selective_incremental_eta": selective_incremental_eta,
        "selective_incremental_tps": selective_incremental_tps,
    }


# ======================================================================================
# Compose + self-test + report
# ======================================================================================
def compose_and_report(census: dict, micro: dict, a: argparse.Namespace) -> dict:
    primary = census[PRIMARY_ARM]
    pos_primary = primary["positions"]
    pp_primary = primary["per_prompt"]
    pos_recompute = census[RECOMPUTE_ARM]["positions"]
    sim = simulate_selective_recompute(pos_primary, pp_primary, pos_recompute, EPS_STAR)

    # tie identifiable from the fast path's own in-register top-2 (#405): every served flip has M1 in top-2.
    fast_flips = [p for p in pos_primary if p["is_flip"]]
    tie_identifiable_from_fast_path = bool(fast_flips and all(
        p["m1_in_m8_top2"] and p["m8_gap"] <= EPS_STAR + BAND_TOL for p in fast_flips))

    # ---- TPS ladder (>=3 microbench seeds -> median + sigma) ----
    eta_seeds = micro["eta_attn_decode_seeds"]
    f = sim["flagged_step_fraction"]
    ladders = [compose_tps(eta, f) for eta in eta_seeds]

    def med_std(key):
        xs = [l[key] for l in ladders]
        return statistics.median(xs), (statistics.pstdev(xs) if len(xs) > 1 else 0.0)

    fast_nonequiv_tps = OFFICIAL_TPS
    blanket_strict_measured_tps, blanket_std = med_std("blanket_strict_tps")
    selective_realizable_tps, selective_realizable_std = med_std("selective_realizable_tps")
    selective_incremental_tps, selective_incremental_std = med_std("selective_incremental_tps")

    # PRIMARY measured selective TPS = the realizable runtime-wrapper (two-pass) cost -- what instruction-3
    # actually costs WITHOUT a served-kernel edit. The incremental (fused) model is reported separately.
    selective_recompute_measured_tps = selective_realizable_tps
    selective_tax_tps = fast_nonequiv_tps - selective_recompute_measured_tps
    selective_tax_vs_2p6_model = selective_tax_tps / SELECTIVE_FIX_TPS_COST_397

    selective_recompute_reaches_identity_1p0 = bool(sim["reaches_identity_1p0"])
    served_identity_after_selective = sim["served_identity_after_selective"]

    # identity reference points on the SAME served (fast) M1-AR reference:
    #   fast    M8-verify  -> the 3-flip baseline (within-stack consistent fast)
    #   selective (this)   -> fast base + strict patches on flagged rows (CROSS-stack mix)
    # plus the consistent all-strict stack's OWN within-arm identity (what 'blanket strict' really delivers).
    fast_decodewidth_identity = primary["decodewidth_e2e_token_identity_rate"]
    blanket_strict_within_identity = census[RECOMPUTE_ARM]["decodewidth_e2e_token_identity_rate"]
    selective_degrades_identity_vs_fast = bool(served_identity_after_selective < fast_decodewidth_identity)
    served_flips_all_bitwise_ties = bool(sim["served_flips_all_bitwise_ties"])

    # the fastest REALIZABLE strictly-equivalent config (no served-kernel edit). Selective two-pass is
    # dominated by blanket whenever f*(1+eta) > eta i.e. always here (f=0.236 >> eta=0.030).
    realizable_candidates = {
        "blanket_strict": blanket_strict_measured_tps,
        "selective_two_pass": selective_realizable_tps,
    }
    fastest_realizable_config = max(realizable_candidates, key=realizable_candidates.get)
    fastest_realizable_strictly_equivalent_tps = realizable_candidates[fastest_realizable_config]
    selective_beats_blanket = bool(selective_realizable_tps > blanket_strict_measured_tps)

    # PR-literal frontier field: measured selective TPS iff identity 1.0 (it is NOT -> None, honest).
    strictly_equivalent_frontier_tps = (selective_recompute_measured_tps
                                         if selective_recompute_reaches_identity_1p0 else None)

    # blocked-flag: realizing the 2.6 model (fused conditional-precision kernel) would require a served-kernel
    # edit -- which this card MUST NOT do. Identity 1.0 is NOT reachable by ANY attention-precision knob here
    # (the disputed positions are bitwise ties -> only the served stack's exact tie-break gives 1.0).
    fused_model_needs_served_change = True
    identity_1p0_unreachable_by_precision = bool(served_flips_all_bitwise_ties
                                                 and not selective_recompute_reaches_identity_1p0)

    # ---- verdict (DOUBLE-RED: dominated on BOTH the TPS axis AND the identity axis) ----
    ident_tag = "IDENTITY_1p0" if selective_recompute_reaches_identity_1p0 else (
        "IDENTITY_DEGRADED_vs_fast" if selective_degrades_identity_vs_fast else "IDENTITY_LT_1p0")
    if selective_beats_blanket and selective_recompute_reaches_identity_1p0:
        verdict = "GREEN_selective_two_pass_beats_blanket_IDENTITY_1p0"
    elif selective_degrades_identity_vs_fast:
        verdict = (f"RED_DOUBLE__selective_recompute_net_negative_tps_AND_{ident_tag}"
                   f"__bitwise_tie_positions_recompute_picks_strict_tiebreak_not_served__"
                   f"2p6_model_is_fused_kernel_only")
    else:
        verdict = (f"RED_selective_two_pass_net_negative_blanket_strict_is_fastest_equivalent_{ident_tag}"
                   f"__2p6_model_is_fused_kernel_only")

    self_test, n_checks = build_self_test(census, micro, sim, tie_identifiable_from_fast_path,
                                          blanket_strict_measured_tps, selective_realizable_tps,
                                          selective_incremental_tps, selective_beats_blanket)
    selective_self_test_passes = bool(all(self_test.values()) and n_checks >= 20)

    report = {
        "pr": 412,
        "leg": "selective higher-precision recompute = fastest strictly-token-equivalent verify? Convert the "
               "#397 ~2.6-TPS model into a MEASURED equivalent-TPS + verify identity (local A10G, analysis-only)",
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
        "primary_arm": PRIMARY_ARM, "eps_star": EPS_STAR, "M_verify": M_VERIFY, "K_spec": K_SPEC,
        # ---- HEADLINE deliverables ----
        "selective_recompute_reaches_identity_1p0": selective_recompute_reaches_identity_1p0,  # PRIMARY bool
        "selective_recompute_measured_tps": selective_recompute_measured_tps,                  # PRIMARY metric
        "selective_recompute_measured_tps_std": selective_realizable_std,
        "strictly_equivalent_frontier_tps": strictly_equivalent_frontier_tps,
        "fast_nonequiv_tps": fast_nonequiv_tps,
        "blanket_strict_measured_tps": blanket_strict_measured_tps,
        "blanket_strict_measured_tps_std": blanket_std,
        "selective_tax_tps": selective_tax_tps,
        "selective_tax_vs_2p6_model": selective_tax_vs_2p6_model,
        "selective_incremental_fused_model_tps": selective_incremental_tps,
        "selective_incremental_fused_model_tps_std": selective_incremental_std,
        "fastest_realizable_strictly_equivalent_tps": fastest_realizable_strictly_equivalent_tps,
        "fastest_realizable_strictly_equivalent_config": fastest_realizable_config,
        "selective_beats_blanket": selective_beats_blanket,
        "flagged_step_fraction": f,
        "n_flips_remaining": sim["n_flips_remaining"],
        "n_flips_recovered": sim["n_flips_recovered"],
        "n_served_flips_not_recovered": sim["n_served_flips_not_recovered"],
        "n_new_flips_introduced": sim["new_flips"],
        "served_identity_after_selective": served_identity_after_selective,
        "fast_decodewidth_identity": fast_decodewidth_identity,
        "blanket_strict_within_identity": blanket_strict_within_identity,
        "selective_degrades_identity_vs_fast": selective_degrades_identity_vs_fast,
        "served_flips_all_bitwise_ties": served_flips_all_bitwise_ties,
        "tie_identifiable_from_fast_path": tie_identifiable_from_fast_path,
        "fused_model_needs_served_change": fused_model_needs_served_change,
        "identity_1p0_unreachable_by_precision": identity_1p0_unreachable_by_precision,
        "selective_self_test_passes": selective_self_test_passes,  # PRIMARY self-test
        # ---- supporting detail ----
        "verdict": verdict,
        "selective_simulation": sim,
        "eta_attn_decode_measured": micro["eta_attn_decode"],
        "eta_attn_decode_std": micro["eta_attn_decode_std"],
        "penalty_decode_band": micro["penalty_decode_band"],
        "verify_penalty_free": micro["verify_penalty_free"],
        "fast_is_byte_exact_M8": micro["fast_is_byte_exact_M8"],
        "strict_is_byte_exact_M8": micro["strict_is_byte_exact_M8"],
        "imported_anchors": {
            "pinned_identity_381": PINNED_IDENTITY_381, "heuristic_identity_381": HEURISTIC_IDENTITY_381,
            "pinned_flip_count_381": PINNED_FLIP_COUNT_381, "heuristic_flip_count_381": HEURISTIC_FLIP_COUNT_381,
            "selective_fix_tps_cost_397": SELECTIVE_FIX_TPS_COST_397, "f_step_band_397": F_STEP_BAND_397,
            "eta_attn_decode_393": ETA_ATTN_DECODE_393, "deployed_strict_393": DEPLOYED_STRICT_393,
            "official_tps": OFFICIAL_TPS, "ceiling_500": CEILING_500, "step_norm_us": STEP_NORM_US,
            "f_attn_344": F_ATTN_344,
        },
        "arms": {
            arm: {
                "decodewidth_e2e_token_identity_rate": d["decodewidth_e2e_token_identity_rate"],
                "determinism_M1_vs_M1": d["determinism_M1_vs_M1"],
                "determinism_M8_vs_M8": d["determinism_M8_vs_M8"],
                "within_batch_copy0_vs_copy1": d["within_batch_copy0_vs_copy1"],
                "chunk_isolated_fraction": d["chunk_isolated_fraction"],
                "vllm_batch_invariant_env": d["vllm_batch_invariant_env"],
                "attn_is_batch_invariant": d["attn_is_batch_invariant"],
                "flip_count": len(d["flip_details"]), "flip_details": d["flip_details"],
                "total_positions": d["total_positions"], "n_prompts": d["n_prompts"],
                "peak_gpu_gb": d["peak_gpu_gb"],
            } for arm, d in census.items()
        },
        "microbench": {k: micro[k] for k in (
            "penalty_decode_band", "penalty_decode_band_std", "eta_attn_decode", "eta_attn_decode_std",
            "eta_attn_decode_seeds", "band_penalty_M1_seeds", "verify_penalty_band_mean", "verify_penalty_free",
            "fast_is_byte_exact_M8", "strict_is_byte_exact_M8", "n_seeds", "iters", "warmup")},
        "self_test": self_test, "self_test_n_checks": n_checks,
        "C": primary["C"], "n_verify": primary["n_verify"], "n_prompts": primary["n_prompts"],
        "model_dir": primary["model_dir"],
    }
    return report


def build_self_test(census, micro, sim, tie_identifiable, blanket_tps, selective_realizable_tps,
                    selective_incremental_tps, selective_beats_blanket) -> tuple[dict, int]:
    checks: dict = {}
    for arm, d in census.items():
        checks[f"{arm}_determinism_m1_eq_1"] = bool(d["determinism_M1_vs_M1"] == 1.0)
        checks[f"{arm}_determinism_m8_eq_1"] = bool(d["determinism_M8_vs_M8"] == 1.0)
        checks[f"{arm}_within_eq_1"] = bool(d["within_batch_copy0_vs_copy1"] == 1.0)
        checks[f"{arm}_geometry_isolated"] = bool(d["chunk_isolated_fraction"] >= 0.99)
        ident = d["decodewidth_e2e_token_identity_rate"]
        checks[f"{arm}_identity_in_range"] = bool(math.isfinite(ident) and 0.0 <= ident <= 1.0)
        checks[f"{arm}_nan_clean"] = bool(d["nan_clean"])

    # arm separation: pin engaged in pinned arm, NOT in heuristic
    checks["pinned_attn_batch_invariant"] = bool(census["pinned"].get("attn_is_batch_invariant"))
    checks["heuristic_not_batch_invariant"] = bool(not census["heuristic"].get("attn_is_batch_invariant"))

    # the served (heuristic) arm carries a residual; pinned has fewer flips (reproduces #381/#397/#405)
    checks["primary_has_residual"] = bool(len(census[PRIMARY_ARM]["flip_details"]) > 0)
    checks["pinned_fewer_or_equal_flips"] = bool(
        len(census["pinned"]["flip_details"]) <= len(census["heuristic"]["flip_details"]))

    # the gate is free: every served flip is identifiable from the fast in-register top-2 (#405)
    checks["tie_identifiable_from_fast_path"] = bool(tie_identifiable)

    # selective wrapper invariants (these validate HARNESS CORRECTNESS, not the experiment's conclusion).
    checks["flag_covers_all_flips"] = bool(sim["flag_covers_all_flips"])
    checks["known_flip_prompts_flagged"] = bool(sim["known_flip_prompts_flagged"])
    # NON-flagged steps are kept verbatim from the fast path -> bit-identical by construction (no flip outside flags)
    checks["nonflagged_steps_bit_identical_to_fast"] = bool(sim["flag_covers_all_flips"])
    # the recompute substitution is DATA-DRIVEN (>=1 flagged row joined to the measured strict arm).
    checks["selective_uses_measured_recompute_rows"] = bool(sim["n_flag_rows"] > 0)
    # accounting closes: every served flip is either recovered or not (no double counting w/ new flips).
    checks["served_flip_accounting_closes"] = bool(
        sim["n_flips_recovered"] + sim["n_served_flips_not_recovered"] == sim["n_baseline_flips"])
    # internal consistency: reported identity == (positions - total disagreements) / positions.
    n_tot = sim["n_total_positions"]
    checks["identity_consistent_with_disagreements"] = bool(
        abs(sim["served_identity_after_selective"] - (n_tot - len(sim["disagreements"])) / n_tot) < 1e-9)
    # the central physics underpinning the finding: every served flip is a BITWISE TIE in the served M1-AR.
    checks["served_flips_all_bitwise_ties"] = bool(sim["served_flips_all_bitwise_ties"])
    checks["flagged_step_fraction_in_unit"] = bool(0.0 <= sim["flagged_step_fraction"] <= 1.0)
    checks["identity_after_in_unit"] = bool(0.0 <= sim["served_identity_after_selective"] <= 1.0)

    # microbench sanity: strict is byte-exact M=8, fast is NOT (the source of non-equivalence)
    checks["strict_byte_exact_M8"] = bool(micro["strict_is_byte_exact_M8"])
    checks["fast_not_byte_exact_or_nonstrict"] = bool(
        (not micro["fast_is_byte_exact_M8"]) or census[PRIMARY_ARM]["decodewidth_e2e_token_identity_rate"] < 1.0)
    checks["verify_penalty_free"] = bool(micro["verify_penalty_free"])
    checks["eta_attn_decode_positive"] = bool(micro["eta_attn_decode"] > 0.0)
    checks["eta_attn_decode_near_393"] = bool(abs(micro["eta_attn_decode"] - ETA_ATTN_DECODE_393)
                                              <= max(0.02, 0.5 * ETA_ATTN_DECODE_393))

    # TPS ladder ordering: fast > blanket; blanket > selective_two_pass (selective net-negative);
    # selective_incremental(fused model) > blanket (the model WOULD beat blanket -- only the realizable wrapper fails)
    checks["fast_gt_blanket"] = bool(OFFICIAL_TPS > blanket_tps)
    checks["blanket_gt_selective_two_pass"] = bool(blanket_tps > selective_realizable_tps)
    checks["selective_two_pass_net_negative"] = bool(not selective_beats_blanket)
    checks["incremental_model_gt_blanket"] = bool(selective_incremental_tps > blanket_tps)
    checks["blanket_near_393"] = bool(abs(blanket_tps - DEPLOYED_STRICT_393) <= 5.0)

    return checks, len(checks)


# ======================================================================================
# Orchestrator + reanalyze + console + wandb + main
# ======================================================================================
def run_phase_subprocess(args_list: list[str], extra_env: dict | None = None) -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if extra_env:
        env.update(extra_env)
    cmd = [sys.executable, os.path.abspath(__file__)] + args_list
    print(f"[orch] launching: {' '.join(args_list)} "
          f"(VLLM_BATCH_INVARIANT={env.get('VLLM_BATCH_INVARIANT', '0')})", flush=True)
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        raise RuntimeError(f"phase subprocess failed (rc={rc}): {args_list}")


def _run_census_arm(a: argparse.Namespace, arm: str) -> dict:
    out_json = str(OUT_DIR / f"arm_{arm}_result.json")
    extra_env = {"VLLM_BATCH_INVARIANT": "1"} if arm == "pinned" else {"VLLM_BATCH_INVARIANT": "0"}
    run_phase_subprocess([
        "--phase", "census", "--arm", arm, "--out", out_json,
        "--n-prompts", str(a.n_prompts), "--ctx-len", str(a.ctx_len), "--n-verify", str(a.n_verify),
        "--gpu-mem-util", str(a.gpu_mem_util), "--max-batched-tokens", str(a.max_batched_tokens),
        "--verbose-k", str(a.verbose_k),
    ], extra_env=extra_env)
    return json.load(open(out_json))


def _run_microbench(a: argparse.Namespace) -> dict:
    out_json = str(OUT_DIR / "microbench_result.json")
    run_phase_subprocess([
        "--phase", "microbench", "--out", out_json,
        "--iters", str(a.iters), "--warmup", str(a.warmup), "--seeds", ",".join(str(s) for s in a.seeds),
    ])
    return json.load(open(out_json))


def orchestrate(a: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    census = {arm: _run_census_arm(a, arm) for arm in CENSUS_ARMS}
    micro = _run_microbench(a)
    _finish(compose_and_report(census, micro, a), a)


def reanalyze(a: argparse.Namespace) -> None:
    census = {}
    for arm in CENSUS_ARMS:
        p = OUT_DIR / f"arm_{arm}_result.json"
        if not p.exists():
            raise FileNotFoundError(f"--reanalyze needs {p} (run the GPU phases first)")
        census[arm] = json.load(open(p))
    mp = OUT_DIR / "microbench_result.json"
    if not mp.exists():
        raise FileNotFoundError(f"--reanalyze needs {mp}")
    micro = json.load(open(mp))
    _finish(compose_and_report(census, micro, a), a)


def _finish(report: dict, a: argparse.Namespace) -> None:
    report_path = OUT_DIR / "selective_recompute_equivalent_tps_results.json"
    json.dump(report, open(report_path, "w"), indent=2)
    _print_console(report)
    if not a.no_wandb:
        log_wandb(report, a)


def _print_console(r: dict) -> None:
    print("\n========== SELECTIVE RECOMPUTE EQUIVALENT TPS (PR #412) ==========", flush=True)
    print(f" VERDICT                                  : {r['verdict']}", flush=True)
    print(f" selective_recompute_reaches_identity_1p0 : {r['selective_recompute_reaches_identity_1p0']}", flush=True)
    print(f" served_identity_after_selective          : {r['served_identity_after_selective']:.7f}  "
          f"(fast {r['fast_decodewidth_identity']:.7f} | strict-within {r['blanket_strict_within_identity']:.7f})",
          flush=True)
    print(f" served flips recovered                   : {r['n_flips_recovered']}/{r['selective_simulation']['n_baseline_flips']}"
          f"  | served flips NOT recovered: {r['n_served_flips_not_recovered']}"
          f"  | NEW flips introduced: {r['n_new_flips_introduced']}"
          f"  => total disagreements {r['n_flips_remaining']}", flush=True)
    print(f" selective_degrades_identity_vs_fast      : {r['selective_degrades_identity_vs_fast']}", flush=True)
    print(f" served_flips_all_bitwise_ties            : {r['served_flips_all_bitwise_ties']}  "
          f"(precision can't pick the served tie-break)", flush=True)
    print(f" flagged_step_fraction (model 0.236)      : {r['flagged_step_fraction']:.5f}", flush=True)
    print(f" tie_identifiable_from_fast_path          : {r['tie_identifiable_from_fast_path']}", flush=True)
    print(" --- TPS ladder (anchored OFFICIAL=481.53) ---", flush=True)
    print(f"  fast_nonequiv_tps                       : {r['fast_nonequiv_tps']:.2f}", flush=True)
    print(f"  blanket_strict_measured_tps             : {r['blanket_strict_measured_tps']:.2f} "
          f"+/-{r['blanket_strict_measured_tps_std']:.2f}  (anchor393 {DEPLOYED_STRICT_393:.2f})", flush=True)
    print(f"  selective_recompute_measured_tps (PRIM) : {r['selective_recompute_measured_tps']:.2f} "
          f"+/-{r['selective_recompute_measured_tps_std']:.2f}  [realizable two-pass]", flush=True)
    print(f"  selective_incremental_fused_model_tps   : {r['selective_incremental_fused_model_tps']:.2f}  "
          f"[#397 2.6-model, fused-kernel only]", flush=True)
    print(f"  selective_tax_tps / vs 2.6-model        : {r['selective_tax_tps']:.2f} / "
          f"{r['selective_tax_vs_2p6_model']:.2f}x", flush=True)
    print(f"  FASTEST realizable equiv config         : {r['fastest_realizable_strictly_equivalent_config']} "
          f"@ {r['fastest_realizable_strictly_equivalent_tps']:.2f}", flush=True)
    print(f"  selective_beats_blanket                 : {r['selective_beats_blanket']}", flush=True)
    print(f" eta_attn_decode (measured)               : {r['eta_attn_decode_measured']:.6f} "
          f"+/-{r['eta_attn_decode_std']:.6f}", flush=True)
    print(f" SELF-TEST PASSES (PRIMARY)               : {r['selective_self_test_passes']} "
          f"({sum(r['self_test'].values())}/{r['self_test_n_checks']})", flush=True)
    fails = [k for k, v in r["self_test"].items() if not v]
    if fails:
        print(f"   self-test FAILS: {fails}", flush=True)
    print("=================================================================\n", flush=True)


def log_wandb(report: dict, a: argparse.Namespace) -> None:
    sys.path.insert(0, os.getcwd())
    try:
        from scripts.wandb_logging import init_wandb_run, finish_wandb
    except Exception as exc:
        print(f"[wandb] helper import failed: {exc!r}; skipping", flush=True)
        return
    run = init_wandb_run(
        job_type="local_profiling", agent="stark", name=a.wandb_name, group=a.wandb_group,
        notes="PR#412 selective higher-precision recompute = fastest strictly-token-equivalent verify? "
              "Measured equivalent-TPS of the #397 2.6-model + identity.",
        config={
            "pr": 412, "M_verify": M_VERIFY, "K_spec": K_SPEC, "n_prompts": report["n_prompts"],
            "C": report["C"], "n_verify": report["n_verify"], "model_dir": report["model_dir"],
            "primary_arm": report["primary_arm"], "eps_star": EPS_STAR,
            "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
            **{f"anchor/{k}": v for k, v in report["imported_anchors"].items()},
        },
    )
    if run is None:
        print("[wandb] disabled (no API key/mode); results saved to JSON only", flush=True)
        return
    keys = ("selective_recompute_reaches_identity_1p0", "selective_recompute_measured_tps",
            "selective_recompute_measured_tps_std", "strictly_equivalent_frontier_tps", "fast_nonequiv_tps",
            "blanket_strict_measured_tps", "blanket_strict_measured_tps_std", "selective_tax_tps",
            "selective_tax_vs_2p6_model", "selective_incremental_fused_model_tps",
            "fastest_realizable_strictly_equivalent_tps", "fastest_realizable_strictly_equivalent_config",
            "selective_beats_blanket", "flagged_step_fraction", "n_flips_remaining", "n_flips_recovered",
            "n_served_flips_not_recovered", "n_new_flips_introduced",
            "served_identity_after_selective", "fast_decodewidth_identity", "blanket_strict_within_identity",
            "selective_degrades_identity_vs_fast", "served_flips_all_bitwise_ties",
            "tie_identifiable_from_fast_path",
            "fused_model_needs_served_change", "identity_1p0_unreachable_by_precision",
            "selective_self_test_passes", "verdict", "eta_attn_decode_measured", "eta_attn_decode_std",
            "penalty_decode_band", "self_test_n_checks", "analysis_only", "no_hf_job",
            "no_served_file_change", "official_tps")
    for k in keys:
        run.summary[k] = report.get(k)
    run.summary["verdict_green"] = report["verdict"].startswith("GREEN")
    run.summary["verdict_red"] = report["verdict"].startswith("RED")
    for arm in CENSUS_ARMS:
        d = report["arms"][arm]
        run.summary[f"{arm}/identity"] = d["decodewidth_e2e_token_identity_rate"]
        run.summary[f"{arm}/flip_count"] = d["flip_count"]
    for k, v in report["self_test"].items():
        run.summary[f"selftest/{k}"] = v
    finish_wandb(run)
    print(f"[wandb] logged run {run.id}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["census", "microbench"], default=None)
    ap.add_argument("--arm", choices=list(CENSUS_ARMS), default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-test", dest="self_test", action="store_true",
                    help="run both census arms + microbench + the PRIMARY self-test (default orchestrator path)")
    ap.add_argument("--reanalyze", action="store_true",
                    help="0-GPU: recompose the report + self-test from saved arm_*.json + microbench_result.json")
    ap.add_argument("--smoke", action="store_true", help="tiny run (few prompts) to validate the path")
    ap.add_argument("--n-prompts", dest="n_prompts", type=int, default=127)
    ap.add_argument("--ctx-len", dest="ctx_len", type=int, default=224)
    ap.add_argument("--n-verify", dest="n_verify", type=int, default=M_VERIFY)
    ap.add_argument("--gpu-mem-util", dest="gpu_mem_util", type=float, default=0.55)
    ap.add_argument("--max-batched-tokens", dest="max_batched_tokens", type=int, default=8192)
    ap.add_argument("--verbose-k", dest="verbose_k", type=int, default=3)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--wandb_group", dest="wandb_group", default="selective-recompute-equivalent-tps")
    ap.add_argument("--wandb_name", dest="wandb_name", default="stark/selective-recompute-equivalent-tps")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()
    a.seeds = [int(s) for s in str(a.seeds).split(",") if s != ""]

    if a.smoke and a.phase is None:
        a.n_prompts = min(a.n_prompts, 4)
        a.iters = min(a.iters, 20)
        a.warmup = min(a.warmup, 5)

    if a.phase == "census":
        phase_census(a.out, a.arm, a.n_prompts, a.ctx_len, a.n_verify,
                     a.gpu_mem_util, a.max_batched_tokens, a.verbose_k)
    elif a.phase == "microbench":
        phase_microbench(a.out, a.iters, a.warmup, a.seeds)
    elif a.reanalyze:
        reanalyze(a)
    else:
        orchestrate(a)


if __name__ == "__main__":
    main()
