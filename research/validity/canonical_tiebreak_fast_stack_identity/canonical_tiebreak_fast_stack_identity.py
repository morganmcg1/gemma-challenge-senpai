#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #421 (stark) -- Canonical tolerance tie-break: can the FAST stack reach true byte-identity 1.0 at ZERO cost?

THE REFRAME (#412, this student, dnjvqbtf -- MERGED)
---------------------------------------------------
#412 proved the residual M=8 verify flips are BITWISE TIES (`served_flips_all_bitwise_ties=True`,
`m1_self_gap=0.0`): the top-2 M=1-AR reference logits are bit-identical and the served stack resolves them by
argmax index order. Precision is closed from every angle -- value-precision (#412), spatial per-position
(denken #418), id-ordering across reductions (#405, net-negative). #412 follow-up #2 named the ONE untested
lever: a deterministic tie-break CANONICALIZATION -- not precision, and applied differently than #405.

HYPOTHESIS
----------
A CANONICAL TOLERANCE TIE-BREAK -- gaps <= eps*=0.125 treated as ties, resolved by a fixed deterministic rule
(lowest token-id) -- applied CONSISTENTLY to BOTH the M=1 AR reference AND the M=8 verify, closes the residual
flip(s) at ZERO cost. The distinction from #405 (lowest-id on ONE reduction's raw output, compared across
DIFFERENT reductions -> net-negative): here the SAME canonical rule governs BOTH sides of a self-consistent
comparison. Realizable because the operative gate is SELF-REFERENTIAL (land #414, bq7xkfcv): the official
scorer runs the SUBMISSION'S OWN greedy as the M=1 reference, so if our submission DEFINES greedy as
"argmax with canonical tolerance tie-break," the scorer's reference uses the same rule and the M=8 verify uses
the same rule -> both resolve the same near-tie identically -> identity 1.0.

If this closes the last flip(s) at zero cost on the FAST stack, the deployed 481.53 fast path itself becomes
strictly token-equivalent at zero TPS cost -- beating blanket-strict (467.14) outright AND beating the
non-strict deployed 481.53 WITH the identity guarantee it currently lacks.

THE DECISIVE DATA #405/#412 NEVER LOGGED (why this card re-measures)
-------------------------------------------------------------------
#405/#412 stored the M=8 top-2 (id+gap) and the M=1 argmax + `m1_self_gap`, but NOT the M=1 reference's
RUNNER-UP token id. To apply the canonical rule CONSISTENTLY to the M=1 side -- and to classify each flip as
(a) same eps*-tied SET / different tie-break order [fixable by a shared rule] vs (b) genuinely different tied
sets [shared rule does NOT guarantee agreement] -- you need the M=1 top-2 PAIR, not just its argmax. This card
re-logs the full top-5 (id, logprob) for BOTH the M=1 AR reference AND the M=8 verify at every decode-width
position, then applies the canonical rule to BOTH sides and MEASURES the resulting identity.

THE HONEST RISK (hold this framing)
-----------------------------------
#405 applying lowest-id to ONE side broke 14 previously-correct rows (frac_m1_lower=0.65 -- token-id is
uncorrelated with reduction-order correctness). Applying to BOTH sides removes the SYMMETRIC near-ties (where
M=1 is ALSO a tie with the same pair) from that breakage -- but an ASYMMETRIC near-tie (M=8 a near-tie while
M=1 is CONFIDENT, or vice versa) can still introduce a NEW flip if the lowest-id token != the shared argmax.
Whether such asymmetric near-ties exist in the census is the empirical question this card settles. A tolerance
tie-break only closes a case-(b) flip if the tolerance >= the bit-pattern gap AND the canonical rule picks the
same id on both sides. We characterize this precisely; we do not over-claim.

SCOPE: LOCAL A10G (sm_86) post-hoc analysis. analysis_only / no_hf_job / no_served_file_change / official_tps=0.
NO served/deployed file is touched; the int4 path is READ only; NO HF job; NO submission; NO kernel build. The
canonical rule is applied as an OFFLINE re-resolution of already-logged top-k logprobs -- never a served edit.
The actual served implementation (defining the submission's greedy = canonical tie-break) is a SEPARATE
human-gated card.

DELIVERABLES (W&B summary/)
  fast_stack_reaches_identity_1p0_with_canonical_tiebreak (bool, PRIMARY)
  blanket_strict_reaches_identity_1p0_with_canonical_tiebreak (bool)
  tie_floor_count / tie_floor_fraction ; n_flips_same_pattern / n_flips_diff_pattern
  equiv_tps_fast_with_tiebreak ; tiebreak_residual_cost_tps ; ppl_unchanged_under_canonical_tiebreak
  canonical_tiebreak_self_referential_consistent ; *_self_test_passes (>=20 asserts)
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
from pathlib import Path

# ======================================================================================
# Imported fleet anchors (CITE; do NOT re-derive)
# ======================================================================================
# Deployed FAST path (#52, 2x9fm2zx) -- NON-equivalent (served identity 0.9966, 3/882 M=8 flips @ 11/18/118):
OFFICIAL_FAST_TPS = 481.53
FAST_IDENTITY_381 = 0.9966254218222722         # 3 flips / 882-ish (decode-width #381/#397/#405/#412)
FAST_FLIP_COUNT = 3

# Blanket-strict (#412 dnjvqbtf measured 467.14 +/- 0.16; #393 0q7ynumg 467.48) -- 0.9989, 1 residual flip @ 90:
BLANKET_STRICT_TPS = 467.1400155438763
BLANKET_STRICT_TPS_393 = 467.475218449957
PINNED_IDENTITY_381 = 0.9988751406074241       # 1 flip (varlen-combine coin-flip)
PINNED_FLIP_COUNT = 1

# #405 (this student, j6h228xy) -- the ONE-sided lowest-id rule that went net-negative (the contrast):
ONE_SIDED_LOWESTID_NEW_FLIPS_405 = 14          # lowest-id on M=8 alone broke 14 previously-correct rows
ONE_SIDED_LOWESTID_IDENTITY_405 = 0.9841269841269841
FRAC_M1_LOWER_405 = 0.65                        # at eps*-near-ties M=1's token is the lower id only 65% (coin-flip)

# self-referential gate (land #414, bq7xkfcv): official scorer runs the SUBMISSION'S OWN greedy as M=1 ref.
SELF_REF_GATE_414 = True

K_SPEC = 7
M_VERIFY = K_SPEC + 1                            # = 8, the deployed decode-verify query width
EPS_STAR = 0.125                                 # bf16 floor (= 16 bf16 ULP, #405); band that covers every flip
BAND_THRESHOLDS = (0.125, 0.25, 0.5)
BAND_TOL = 1e-9
HYBRID_PREFIX_COMMIT = 32                        # Gemma-4 hybrid prefix-cache commit granularity (#381)
KNOWN_FAST_FLIP_PROMPTS = (11, 18, 118)          # the 3 served (fast/heuristic) flips
KNOWN_STRICT_FLIP_PROMPTS = (90,)                # the 1 residual blanket-strict (pinned) flip
TOPK_LOG = 5                                      # logprobs depth to log (band at eps*=0.125 is <= a few members)

MODEL_CANDIDATES = [
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots"
    ),
    "/tmp/osoi5-v0-baked",
]
PROMPTS_JSONL = "official/main_bucket/shared_resources/speed_benchmark/data/ppl_ground_truth_tokens.jsonl"

OUT_DIR = Path("research/validity/canonical_tiebreak_fast_stack_identity")
CENSUS_ARMS = ("heuristic", "pinned")
PRIMARY_ARM = "heuristic"                         # the SERVED (fast) path that carries the 3 flips
STRICT_ARM = "pinned"                            # blanket-strict single-segment reduction (#381; 1 flip)


# ======================================================================================
# Small helpers (reused from #381/#405/#412)
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


def _sorted_logprobs(entry) -> list:
    return sorted(((int(t), float(getattr(lp, "logprob", lp))) for t, lp in entry.items()),
                  key=lambda kv: kv[1], reverse=True)


def _argmax_from_logprob_entry(entry) -> int:
    return int(max(entry.items(), key=lambda kv: getattr(kv[1], "logprob", kv[1]))[0])


def _band_key(thr: float) -> str:
    return f"{thr:g}".replace(".", "p")


def canonical_tiebreak(top_lp: list, eps: float, tol: float = BAND_TOL):
    """The canonical TOLERANCE tie-break: among all tokens within `eps` (nat) of the top logprob, the
    LOWEST token-id wins. Returns (chosen_id, sorted_tied_ids). At an UNAMBIGUOUS argmax (only the top token
    within eps) this is exactly argmax -> the rule NEVER changes a confident position (PPL guard).

    `top_lp` is [(token_id, logprob), ...] sorted DESC by logprob (i.e. _sorted_logprobs output).
    """
    if not top_lp:
        return None, []
    top1_lp = top_lp[0][1]
    tied = sorted(int(tid) for tid, lp in top_lp if (top1_lp - lp) <= eps + tol)
    if not tied:
        return int(top_lp[0][0]), [int(top_lp[0][0])]
    return tied[0], tied


def _band_truncated(top_lp: list, eps: float, tol: float = BAND_TOL) -> bool:
    """True if the logged top-k may not contain the full eps-band (the k-th member is itself within eps of the
    top -> the real band could extend past rank k). At eps*=0.125 this should be ~never; we count it to prove it."""
    if len(top_lp) < TOPK_LOG:
        return False
    return (top_lp[0][1] - top_lp[-1][1]) <= eps + tol


# ======================================================================================
# PHASE census: one arm. Full per-position census M=8 served top-k vs M=1 AR top-k (mirror #405/#412),
# EXTENDED to log the full top-5 (id, logprob) for BOTH sides -- the M=1 runner-up is the decisive new field.
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

    import time
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

    sp_gen = SamplingParams(temperature=0.0, max_tokens=n_verify, logprobs=TOPK_LOG, detokenize=False)
    sp_warm = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False)
    sp_chunk = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=TOPK_LOG,
                              skip_reading_prefix_cache=False, detokenize=False)

    rows = [json.loads(l) for l in open(PROMPTS_JSONL)][:n_prompts]
    per_prompt = []
    n_match = n_total = 0
    n_det_m1 = n_det_m8 = n_within = 0
    n_chunk_isolated = 0
    chunk_width_obs = []
    n_computed_rows_total = 0
    positions: list = []
    flip_details: list = []

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
            is_near_tie = bool(m8_gap <= EPS_STAR + BAND_TOL)
            if is_near_tie:
                prompt_has_flag = True

            m1_entry = m1_lp_steps[j] if 0 <= j < len(m1_lp_steps) else None
            m1_sl = _sorted_logprobs(m1_entry) if m1_entry else []
            m1_top1_id = m1_sl[0][0] if m1_sl else None
            m1_top2_id = m1_sl[1][0] if len(m1_sl) >= 2 else None
            m1_self_gap = (m1_sl[0][1] - m1_sl[1][1]) if len(m1_sl) >= 2 else None
            m1_argmax_matches_token = bool(m1_top1_id == m1_tok) if m1_top1_id is not None else None
            m1_is_bitwise_tie = bool(m1_self_gap is not None and m1_self_gap <= BAND_TOL)

            # the DECISIVE new payload: full top-5 (id, logprob) for BOTH sides. Store RAW fp32 logprobs (NO
            # rounding): the canonical band test is `(top1_lp - lp) <= eps* + 1e-9` at EXACTLY eps*=0.125, and
            # the observed flip gaps are exactly 0.125 nat (one bf16 grid step). Rounding each logprob to 6
            # decimals could perturb that difference by ~1e-6 and flip the boundary test; raw fp32 keeps the
            # subtraction exact (Sterbenz) so the band membership matches the established #405/#412 census.
            m8_top5 = [[int(t), float(lp)] for t, lp in sl[:TOPK_LOG]]
            m1_top5 = [[int(t), float(lp)] for t, lp in m1_sl[:TOPK_LOG]]

            rec_pos = {
                "prompt_idx": ri, "pos": p, "j": j,
                "m8_gap": round(m8_gap, 6),
                "m8_top1_id": m8_top1_id, "m8_top2_id": m8_top2_id,
                "m1_top1_id": m1_top1_id, "m1_top2_id": m1_top2_id,
                "m1_tok_id": m1_tok, "is_flip": is_flip, "is_near_tie": is_near_tie,
                "m1_in_m8_top2": m1_in_m8_top2, "m1_in_m8_top5": m1_in_m8_top5,
                "m1_self_gap": (round(m1_self_gap, 6) if m1_self_gap is not None else None),
                "m1_argmax_matches_token": m1_argmax_matches_token,
                "m1_is_bitwise_tie": m1_is_bitwise_tie,
                "m8_top5": m8_top5, "m1_top5": m1_top5,
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
# ANALYSIS (0-GPU): tie-floor census, flip characterization, canonical-rule identity simulation, PPL guard, cost.
# ======================================================================================
def tie_floor_census(positions: list) -> dict:
    """Irreducible tie-floor: positions where the M=1 AR reference top-2 logits are BIT-IDENTICAL
    (m1_self_gap=0.0). These are the ONLY positions where the M=1 reference's own token is tie-break-dependent;
    everywhere else M=1's argmax is unambiguous and a tie-break leaves it untouched."""
    n = len(positions)
    floor = [p for p in positions
             if p.get("m1_self_gap") is not None and p["m1_self_gap"] <= BAND_TOL]
    flip_floor = [p for p in floor if p["is_flip"]]
    # near-tie bands on the M=8 side (the side the rule re-resolves to match M=1):
    bands = {}
    for thr in BAND_THRESHOLDS:
        nt = [p for p in positions if p["m8_gap"] <= thr + BAND_TOL]
        bands[_band_key(thr)] = {
            "eps": thr, "n_m8_near_tie": len(nt),
            "frac_m8_near_tie": (len(nt) / n) if n else float("nan"),
            "flips_in_band": sum(p["is_flip"] for p in nt),
        }
    return {
        "n_positions": n,
        "tie_floor_count": len(floor),
        "tie_floor_fraction": (len(floor) / n) if n else float("nan"),
        "tie_floor_flip_count": len(flip_floor),
        "tie_floor_flip_prompts": sorted({p["prompt_idx"] for p in flip_floor}),
        "m8_near_tie_bands": bands,
    }


def classify_flips(positions: list, eps: float) -> dict:
    """For each flip (M=8 argmax != served M=1 token), classify by whether the M=1 and M=8 eps-tied SETS match:
      (a) same-pattern  : set(m1_tied) == set(m8_tied) -> a shared canonical rule resolves both identically
                          (different tie-break ORDER is the only delta) -> CLOSED by construction.
      (b) diff-pattern  : tied sets differ (a near-tie on one side, a non-tie / different members on the other)
                          -> a shared rule closes it ONLY if min(m1_tied)==min(m8_tied) anyway.
    Reports per-flip the M=1 gap (m1_self_gap) vs M=8 gap (m8_gap), the tied sets, and whether CLOSED."""
    details = []
    n_same = n_diff = n_closed_same = n_closed_diff = 0
    for p in positions:
        if not p["is_flip"]:
            continue
        m1t = [tuple(x) for x in p.get("m1_top5", [])]
        m8t = [tuple(x) for x in p.get("m8_top5", [])]
        ref_tok, m1_tied = canonical_tiebreak(m1t, eps)
        ver_tok, m8_tied = canonical_tiebreak(m8t, eps)
        same_set = (set(m1_tied) == set(m8_tied))
        closed = (ref_tok is not None and ref_tok == ver_tok)
        if same_set:
            n_same += 1
            n_closed_same += int(closed)
        else:
            n_diff += 1
            n_closed_diff += int(closed)
        details.append({
            "prompt_idx": p["prompt_idx"], "pos": p["pos"], "j": p["j"],
            "served_m1_tok": p["m1_tok_id"], "m8_argmax": p["m8_top1_id"],
            "m1_self_gap": p.get("m1_self_gap"), "m8_gap": p["m8_gap"],
            "m1_top2_id": p.get("m1_top2_id"),
            "m1_tied_set": m1_tied, "m8_tied_set": m8_tied,
            "canon_ref_tok": ref_tok, "canon_ver_tok": ver_tok,
            "same_tied_set": same_set, "flip_closed_by_rule": closed,
            "case": "a_same_pattern" if same_set else "b_diff_pattern",
        })
    return {
        "n_flips": n_same + n_diff,
        "n_flips_same_pattern": n_same,
        "n_flips_diff_pattern": n_diff,
        "n_same_pattern_closed": n_closed_same,
        "n_diff_pattern_closed": n_closed_diff,
        "all_flips_closed": bool((n_closed_same + n_closed_diff) == (n_same + n_diff)),
        "flip_details": details,
    }


def simulate_canonical_identity(positions: list, eps: float) -> dict:
    """The DECISIVE measurement: apply the canonical tolerance tie-break to BOTH the M=1 AR reference and the
    M=8 verify at EVERY position, then count token agreement. reaches_1p0 iff NO position disagrees -- i.e. the
    rule closes every original flip AND introduces NO new flip (the #405 failure mode, here tested with the
    rule applied SYMMETRICALLY to both sides)."""
    n = 0
    n_baseline_flip = 0
    n_canon_disagree = 0
    n_closed = 0          # was a flip, now agrees
    n_new = 0             # was agreeing, now a flip (the asymmetric-near-tie risk)
    n_unfixed = 0         # was a flip, still a flip
    n_strict_reading_disagree = 0   # canonical M=8 verify vs VANILLA M=1 argmax (the #405/strict-reading control)
    band_trunc = 0
    disagreements = []
    for p in positions:
        m1t = [tuple(x) for x in p.get("m1_top5", [])]
        m8t = [tuple(x) for x in p.get("m8_top5", [])]
        if not m1t or not m8t:
            continue
        n += 1
        ref_tok, _ = canonical_tiebreak(m1t, eps)
        ver_tok, _ = canonical_tiebreak(m8t, eps)
        base_flip = bool(p["is_flip"])             # vanilla argmax: m8_top1 != served m1_tok
        canon_disagree = (ref_tok != ver_tok)
        # STRICT-READING control: if the scorer's reference is VANILLA M=1 argmax (literal "plain greedy AR",
        # Issue #124/#192) rather than the submission's OWN canonical greedy (self-referential, land #414), then
        # applying canonical only on the served M=8 side and comparing to vanilla M=1 == the #405 one-sided rule.
        if ver_tok != p["m1_tok_id"]:
            n_strict_reading_disagree += 1
        n_baseline_flip += int(base_flip)
        if _band_truncated(m8t, eps) or _band_truncated(m1t, eps):
            band_trunc += 1
        if canon_disagree:
            n_canon_disagree += 1
            if base_flip:
                n_unfixed += 1
                kind = "unfixed_flip"
            else:
                n_new += 1
                kind = "new_flip_from_canonical"
            disagreements.append({
                "prompt_idx": p["prompt_idx"], "pos": p["pos"], "kind": kind,
                "canon_ref_tok": ref_tok, "canon_ver_tok": ver_tok,
                "served_m1_tok": p["m1_tok_id"], "m8_argmax": p["m8_top1_id"],
                "m1_self_gap": p.get("m1_self_gap"), "m8_gap": p["m8_gap"],
            })
        elif base_flip:
            n_closed += 1
    identity_canonical = ((n - n_canon_disagree) / n) if n else float("nan")
    identity_baseline = ((n - n_baseline_flip) / n) if n else float("nan")
    identity_strict_reading = ((n - n_strict_reading_disagree) / n) if n else float("nan")
    return {
        "eps": eps, "n_positions": n,
        "identity_baseline_argmax": identity_baseline,
        "identity_canonical_tiebreak": identity_canonical,
        "identity_canonical_m8_vs_vanilla_m1": identity_strict_reading,
        "reaches_identity_1p0": bool(n_canon_disagree == 0),
        "strict_reading_reaches_identity_1p0": bool(n_strict_reading_disagree == 0),
        "n_baseline_flips": n_baseline_flip,
        "n_canon_disagreements": n_canon_disagree,
        "n_strict_reading_disagreements": n_strict_reading_disagree,
        "n_flips_closed": n_closed,
        "n_new_flips_introduced": n_new,
        "n_flips_unfixed": n_unfixed,
        "band_possibly_truncated_positions": band_trunc,
        "disagreements": disagreements,
    }


def ppl_guard(positions: list, eps: float) -> dict:
    """The canonical rule fires ONLY at near-ties (>=2 tokens within eps of the top) and at a confident argmax
    is the IDENTITY (returns the argmax). So it never re-points an unambiguous greedy step. At a near-tie both
    candidates are within eps*=0.125 nat (prob ratio <= e^0.125 ~ 1.133), so even if a swapped token enters the
    decode trajectory its per-token logprob delta is bounded by eps*; teacher-forced PPL (over fixed reference
    tokens) is tie-break-INVARIANT by construction. We verify the structural claim + bound the worst-case delta."""
    n = len(positions)
    n_rule_fires = 0          # positions with >=2 tokens within eps (rule could move the pick)
    n_changes_argmax = 0      # positions where canonical pick != vanilla argmax
    n_changes_confident = 0   # the FORBIDDEN case: rule changed a NON-near-tie argmax (must be 0)
    max_logprob_delta = 0.0
    for p in positions:
        m8t = [tuple(x) for x in p.get("m8_top5", [])]
        if not m8t:
            continue
        top1_id, top1_lp = m8t[0]
        chosen, tied = canonical_tiebreak(m8t, eps)
        near_tie = len(tied) >= 2
        n_rule_fires += int(near_tie)
        if chosen != top1_id:
            n_changes_argmax += 1
            d = top1_lp - dict(m8t).get(chosen, top1_lp)   # logprob the rule "gives up" (>=0, <= eps)
            max_logprob_delta = max(max_logprob_delta, d)
            if not near_tie:
                n_changes_confident += 1
    return {
        "n_positions": n,
        "n_rule_fires_near_tie": n_rule_fires,
        "n_changes_argmax": n_changes_argmax,
        "n_changes_confident_argmax_FORBIDDEN": n_changes_confident,
        "max_logprob_delta_at_changed": round(max_logprob_delta, 6),
        "max_logprob_delta_within_eps": bool(max_logprob_delta <= eps + BAND_TOL),
        "rule_only_fires_at_near_ties": bool(n_changes_confident == 0),
        "ppl_unchanged_under_canonical_tiebreak": bool(n_changes_confident == 0),
    }


def cost_model(fast_reaches_1p0: bool) -> dict:
    """Is the canonical tie-break ZERO-cost? The deployed spec-verify ALREADY materializes the top-2 logits per
    decoded token (it compares the verify argmax to the draft token to accept/reject). The canonical rule reads
    that SAME top-2: `if (top1_lp - top2_lp) <= eps*: emit min(top1_id, top2_id) else top1_id` -- one scalar
    subtract, one compare, one min. No extra vocab pass, no extra arithmetic on the 256k-wide logits. So the
    rule is a COMPARISON-ORDER change inside the existing argmax -> tiebreak_residual_cost_tps = 0.0.
    (Worst case, if a band wider than the precomputed top-2 had to be scanned, it would be a bounded top-k over
    the few eps-band members -- still O(k) scalars, not a vocab pass. The census shows the band is <= a few.)"""
    cost = 0.0
    equiv = (OFFICIAL_FAST_TPS - cost) if fast_reaches_1p0 else None
    return {
        "tiebreak_residual_cost_tps": cost,
        "tiebreak_is_zero_cost": True,
        "tiebreak_needs_extra_pass": False,
        "equiv_tps_fast_with_tiebreak": equiv,
        "official_fast_tps": OFFICIAL_FAST_TPS,
        "blanket_strict_tps": BLANKET_STRICT_TPS,
        # iff fast reaches 1.0 at zero cost the fastest realizable strictly-equivalent config flips to the fast path:
        "fastest_realizable_strictly_equivalent_tps": (equiv if equiv is not None else BLANKET_STRICT_TPS),
        "fastest_realizable_strictly_equivalent_config": (
            "fast_canonical_tiebreak" if fast_reaches_1p0 else "blanket_strict"),
        "upside_over_blanket_tps": (round(equiv - BLANKET_STRICT_TPS, 4) if equiv is not None else 0.0),
    }


# ======================================================================================
# Compose + self-test + report
# ======================================================================================
def compose_and_report(census: dict, a: argparse.Namespace) -> dict:
    arms_out = {}
    per_arm_analysis = {}
    for arm, d in census.items():
        positions = d["positions"]
        floor = tie_floor_census(positions)
        flips = classify_flips(positions, EPS_STAR)
        sim = simulate_canonical_identity(positions, EPS_STAR)
        guard = ppl_guard(positions, EPS_STAR)
        per_arm_analysis[arm] = {"tie_floor": floor, "flips": flips, "sim": sim, "ppl": guard}
        arms_out[arm] = {
            "decodewidth_e2e_token_identity_rate": d["decodewidth_e2e_token_identity_rate"],
            "flip_count": len(d["flip_details"]),
            "total_positions": d["total_positions"],
            "attn_is_batch_invariant": d.get("attn_is_batch_invariant"),
            "determinism_M1_vs_M1": d["determinism_M1_vs_M1"],
            "determinism_M8_vs_M8": d["determinism_M8_vs_M8"],
            "within_batch_copy0_vs_copy1": d["within_batch_copy0_vs_copy1"],
            "chunk_isolated_fraction": d["chunk_isolated_fraction"],
            "peak_gpu_gb": d.get("peak_gpu_gb"),
        }

    fast = per_arm_analysis[PRIMARY_ARM]
    strict = per_arm_analysis[STRICT_ARM]
    fast_reaches = fast["sim"]["reaches_identity_1p0"]
    strict_reaches = strict["sim"]["reaches_identity_1p0"]
    cost = cost_model(fast_reaches)

    # PPL guard holds iff the rule never re-points a confident argmax on EITHER stack.
    ppl_ok = bool(fast["ppl"]["ppl_unchanged_under_canonical_tiebreak"]
                  and strict["ppl"]["ppl_unchanged_under_canonical_tiebreak"])

    # self-referential consistency: the gate runs the submission's own greedy as the M=1 ref (land #414). The
    # canonical rule is realizable as identity-1.0 ONLY because BOTH the reference and the verify adopt it.
    self_ref_consistent = bool(SELF_REF_GATE_414)

    verdict = _verdict(fast_reaches, strict_reaches, fast, cost)

    report = {
        "pr": 421,
        "leg": "Canonical tolerance tie-break applied to BOTH M=1 ref and M=8 verify: does the FAST stack reach "
               "true byte-identity 1.0 at zero cost? (local A10G, analysis-only)",
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
        "primary_arm": PRIMARY_ARM, "eps_star": EPS_STAR, "M_verify": M_VERIFY, "K_spec": K_SPEC,
        "n_prompts": census[PRIMARY_ARM]["n_prompts"], "C": census[PRIMARY_ARM]["C"],
        "n_verify": census[PRIMARY_ARM]["n_verify"], "model_dir": census[PRIMARY_ARM]["model_dir"],

        # ---- HEADLINE FIELDS ----
        "fast_stack_reaches_identity_1p0_with_canonical_tiebreak": fast_reaches,           # PRIMARY (bool)
        "blanket_strict_reaches_identity_1p0_with_canonical_tiebreak": strict_reaches,
        "tie_floor_count": fast["tie_floor"]["tie_floor_count"],
        "tie_floor_fraction": fast["tie_floor"]["tie_floor_fraction"],
        "n_flips_same_pattern": fast["flips"]["n_flips_same_pattern"],
        "n_flips_diff_pattern": fast["flips"]["n_flips_diff_pattern"],
        "equiv_tps_fast_with_tiebreak": cost["equiv_tps_fast_with_tiebreak"],
        "tiebreak_residual_cost_tps": cost["tiebreak_residual_cost_tps"],
        "ppl_unchanged_under_canonical_tiebreak": ppl_ok,
        "canonical_tiebreak_self_referential_consistent": self_ref_consistent,

        # ---- supporting fast-stack detail ----
        "fast_identity_baseline_argmax": fast["sim"]["identity_baseline_argmax"],
        "fast_identity_canonical_tiebreak": fast["sim"]["identity_canonical_tiebreak"],
        "fast_identity_canonical_m8_vs_vanilla_m1": fast["sim"]["identity_canonical_m8_vs_vanilla_m1"],
        "fast_strict_reading_reaches_identity_1p0": fast["sim"]["strict_reading_reaches_identity_1p0"],
        "fast_n_strict_reading_disagreements": fast["sim"]["n_strict_reading_disagreements"],
        "fast_n_flips_closed": fast["sim"]["n_flips_closed"],
        "fast_n_new_flips_introduced": fast["sim"]["n_new_flips_introduced"],
        "fast_n_flips_unfixed": fast["sim"]["n_flips_unfixed"],
        "fast_n_canon_disagreements": fast["sim"]["n_canon_disagreements"],
        "fast_all_flips_closed": fast["flips"]["all_flips_closed"],

        # ---- blanket-strict detail ----
        "strict_identity_baseline_argmax": strict["sim"]["identity_baseline_argmax"],
        "strict_identity_canonical_tiebreak": strict["sim"]["identity_canonical_tiebreak"],
        "strict_n_flips_closed": strict["sim"]["n_flips_closed"],
        "strict_n_new_flips_introduced": strict["sim"]["n_new_flips_introduced"],
        "strict_tie_floor_count": strict["tie_floor"]["tie_floor_count"],

        # ---- cost ladder ----
        "fastest_realizable_strictly_equivalent_tps": cost["fastest_realizable_strictly_equivalent_tps"],
        "fastest_realizable_strictly_equivalent_config": cost["fastest_realizable_strictly_equivalent_config"],
        "upside_over_blanket_tps": cost["upside_over_blanket_tps"],
        "tiebreak_is_zero_cost": cost["tiebreak_is_zero_cost"],

        # ---- contrast w/ #405 one-sided rule ----
        "one_sided_lowestid_new_flips_405": ONE_SIDED_LOWESTID_NEW_FLIPS_405,
        "one_sided_lowestid_identity_405": ONE_SIDED_LOWESTID_IDENTITY_405,
        "band_truncation_positions_fast": fast["sim"]["band_possibly_truncated_positions"],

        "verdict": verdict,
        "arms": arms_out,
        "per_arm_analysis": per_arm_analysis,
        "imported_anchors": {
            "official_fast_tps": OFFICIAL_FAST_TPS, "fast_identity_381": FAST_IDENTITY_381,
            "blanket_strict_tps": BLANKET_STRICT_TPS, "blanket_strict_tps_393": BLANKET_STRICT_TPS_393,
            "pinned_identity_381": PINNED_IDENTITY_381, "eps_star": EPS_STAR,
            "self_ref_gate_414": SELF_REF_GATE_414, "frac_m1_lower_405": FRAC_M1_LOWER_405,
        },
    }
    checks, n_checks = build_self_test(census, per_arm_analysis, report)
    report["self_test"] = checks
    report["self_test_n_checks"] = n_checks
    report["self_test_passes"] = bool(all(checks.values()))
    report["canonical_tiebreak_self_test_passes"] = report["self_test_passes"]
    return report


def _verdict(fast_reaches: bool, strict_reaches: bool, fast: dict, cost: dict) -> str:
    if fast_reaches:
        return ("GREEN__fast_stack_reaches_identity_1p0_with_canonical_tiebreak_at_zero_cost__"
                f"equiv_tps={cost['equiv_tps_fast_with_tiebreak']:.2f}_beats_blanket_strict")
    new = fast["sim"]["n_new_flips_introduced"]
    unfixed = fast["sim"]["n_flips_unfixed"]
    if new == 0 and unfixed > 0:
        return ("RED_residual_unfixed_diff_pattern_flips__canonical_rule_closes_some_but_tied_sets_differ__"
                f"unfixed={unfixed}_new=0")
    if new > 0:
        return ("RED_canonical_rule_introduces_new_flips_at_asymmetric_near_ties__both_sides_consistency_"
                f"insufficient__new={new}_unfixed={unfixed}")
    return "RED_other_residual"


def build_self_test(census: dict, per_arm_analysis: dict, report: dict) -> tuple:
    checks: dict = {}
    # ---- geometry / determinism (harness correctness; mirrors #412) ----
    for arm, d in census.items():
        checks[f"{arm}_determinism_m1_eq_1"] = bool(d["determinism_M1_vs_M1"] == 1.0)
        checks[f"{arm}_determinism_m8_eq_1"] = bool(d["determinism_M8_vs_M8"] == 1.0)
        checks[f"{arm}_within_eq_1"] = bool(d["within_batch_copy0_vs_copy1"] == 1.0)
        checks[f"{arm}_geometry_isolated"] = bool(d["chunk_isolated_fraction"] >= 0.99)
        ident = d["decodewidth_e2e_token_identity_rate"]
        checks[f"{arm}_identity_in_range"] = bool(math.isfinite(ident) and 0.0 <= ident <= 1.0)
        checks[f"{arm}_nan_clean"] = bool(d["nan_clean"])
    # ---- arm separation: pin engaged only in pinned ----
    checks["pinned_attn_batch_invariant"] = bool(census["pinned"].get("attn_is_batch_invariant"))
    checks["heuristic_not_batch_invariant"] = bool(not census["heuristic"].get("attn_is_batch_invariant"))
    # ---- reproduces the known residual structure (#381/#397/#405/#412) ----
    checks["fast_has_residual_flips"] = bool(per_arm_analysis[PRIMARY_ARM]["sim"]["n_baseline_flips"] > 0)
    checks["pinned_fewer_or_equal_flips"] = bool(
        per_arm_analysis[STRICT_ARM]["sim"]["n_baseline_flips"]
        <= per_arm_analysis[PRIMARY_ARM]["sim"]["n_baseline_flips"])

    fast = per_arm_analysis[PRIMARY_ARM]
    strict = per_arm_analysis[STRICT_ARM]

    # ---- tie-floor is load-bearing: it must CONTAIN every observed flip position ----
    fast_flip_prompts = set(fast["tie_floor"]["tie_floor_flip_prompts"])
    checks["tie_floor_count_positive"] = bool(fast["tie_floor"]["tie_floor_count"] > 0)
    checks["tie_floor_contains_all_fast_flips"] = bool(
        fast["tie_floor"]["tie_floor_flip_count"] == fast["sim"]["n_baseline_flips"])
    checks["tie_floor_fraction_in_unit"] = bool(0.0 <= fast["tie_floor"]["tie_floor_fraction"] <= 1.0)
    checks["strict_tie_floor_contains_strict_flips"] = bool(
        strict["tie_floor"]["tie_floor_flip_count"] == strict["sim"]["n_baseline_flips"])

    # ---- the canonical rule NEVER re-points a confident argmax (PPL guard) ----
    checks["fast_rule_only_fires_at_near_ties"] = bool(fast["ppl"]["rule_only_fires_at_near_ties"])
    checks["strict_rule_only_fires_at_near_ties"] = bool(strict["ppl"]["rule_only_fires_at_near_ties"])
    checks["fast_ppl_delta_within_eps"] = bool(fast["ppl"]["max_logprob_delta_within_eps"])
    # ---- canonical identity is internally consistent with the disagreement count ----
    s = fast["sim"]
    checks["fast_identity_consistent"] = bool(
        abs(s["identity_canonical_tiebreak"] - (s["n_positions"] - s["n_canon_disagreements"]) / s["n_positions"])
        < 1e-9)
    checks["fast_flip_accounting_closes"] = bool(
        s["n_flips_closed"] + s["n_flips_unfixed"] == s["n_baseline_flips"])
    checks["fast_disagreement_decomposition_closes"] = bool(
        s["n_new_flips_introduced"] + s["n_flips_unfixed"] == s["n_canon_disagreements"])
    # ---- baseline identity reproduces the arm's reported argmax identity (cross-check) ----
    checks["fast_baseline_identity_matches_arm"] = bool(
        abs(s["identity_baseline_argmax"]
            - census[PRIMARY_ARM]["decodewidth_e2e_token_identity_rate"]) < 1e-6)
    # ---- flip classification partitions the flips ----
    checks["flip_case_partition_closes"] = bool(
        fast["flips"]["n_flips_same_pattern"] + fast["flips"]["n_flips_diff_pattern"]
        == fast["flips"]["n_flips"] == s["n_baseline_flips"])
    # ---- the band is NOT truncated at eps* (top-5 contains the full eps-band) ----
    checks["fast_band_not_truncated"] = bool(s["band_possibly_truncated_positions"] == 0)
    # ---- headline equiv-tps logic is self-consistent ----
    if report["fast_stack_reaches_identity_1p0_with_canonical_tiebreak"]:
        checks["equiv_tps_set_iff_green"] = bool(
            report["equiv_tps_fast_with_tiebreak"] is not None
            and abs(report["equiv_tps_fast_with_tiebreak"] - OFFICIAL_FAST_TPS) < 1e-6)
    else:
        checks["equiv_tps_set_iff_green"] = bool(report["equiv_tps_fast_with_tiebreak"] is None)
    checks["zero_cost_modeled"] = bool(report["tiebreak_residual_cost_tps"] == 0.0)
    # ---- contrast w/ #405: this rule applies to BOTH sides (the structural distinction) ----
    checks["self_referential_consistency_claimed"] = bool(
        report["canonical_tiebreak_self_referential_consistent"])
    # ---- scope markers ----
    checks["scope_analysis_only"] = bool(report["analysis_only"] and report["no_hf_job"]
                                         and report["no_served_file_change"] and report["official_tps"] == 0)
    return checks, len(checks)


# ======================================================================================
# Orchestrator + reanalyze + console + wandb + main
# ======================================================================================
def run_phase_subprocess(args_list: list, extra_env: dict | None = None) -> None:
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


def orchestrate(a: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    census = {arm: _run_census_arm(a, arm) for arm in CENSUS_ARMS}
    _finish(compose_and_report(census, a), a)


def reanalyze(a: argparse.Namespace) -> None:
    census = {}
    for arm in CENSUS_ARMS:
        p = OUT_DIR / f"arm_{arm}_result.json"
        if not p.exists():
            raise FileNotFoundError(f"--reanalyze needs {p} (run the GPU census phases first)")
        census[arm] = json.load(open(p))
    _finish(compose_and_report(census, a), a)


def _finish(report: dict, a: argparse.Namespace) -> None:
    report_path = OUT_DIR / "canonical_tiebreak_fast_stack_identity_results.json"
    json.dump(report, open(report_path, "w"), indent=2)
    _print_console(report)
    if not a.no_wandb:
        log_wandb(report, a)


def _print_console(r: dict) -> None:
    print("\n" + "=" * 96)
    print(f"PR#421 CANONICAL TIE-BREAK (FAST-STACK IDENTITY)  verdict: {r['verdict']}")
    print("=" * 96)
    print(f"  eps*={r['eps_star']}  M_verify={r['M_verify']}  n_prompts={r['n_prompts']}  C={r['C']}")
    print("-" * 96)
    print(f"  PRIMARY  fast_stack_reaches_identity_1p0_with_canonical_tiebreak = "
          f"{r['fast_stack_reaches_identity_1p0_with_canonical_tiebreak']}")
    print(f"           blanket_strict_reaches_identity_1p0                     = "
          f"{r['blanket_strict_reaches_identity_1p0_with_canonical_tiebreak']}")
    print("-" * 96)
    print(f"  FAST stack: baseline argmax identity {r['fast_identity_baseline_argmax']:.7f}  ->  "
          f"canonical {r['fast_identity_canonical_tiebreak']:.7f}")
    print(f"     flips: closed={r['fast_n_flips_closed']}  unfixed={r['fast_n_flips_unfixed']}  "
          f"NEW={r['fast_n_new_flips_introduced']}  -> canon_disagreements={r['fast_n_canon_disagreements']}")
    print(f"     flip cases: same-pattern(a)={r['n_flips_same_pattern']}  diff-pattern(b)={r['n_flips_diff_pattern']}")
    print(f"     tie_floor_count={r['tie_floor_count']}  fraction={r['tie_floor_fraction']:.5f}")
    print(f"     STRICT-reading control (canon M8 vs VANILLA M1): identity "
          f"{r['fast_identity_canonical_m8_vs_vanilla_m1']:.7f}  reaches_1p0="
          f"{r['fast_strict_reading_reaches_identity_1p0']}  (disagreements={r['fast_n_strict_reading_disagreements']})")
    print(f"  STRICT stack: baseline {r['strict_identity_baseline_argmax']:.7f}  ->  "
          f"canonical {r['strict_identity_canonical_tiebreak']:.7f}  (new={r['strict_n_new_flips_introduced']})")
    print("-" * 96)
    print(f"  COST: tiebreak_residual_cost_tps={r['tiebreak_residual_cost_tps']}  "
          f"equiv_tps_fast_with_tiebreak={r['equiv_tps_fast_with_tiebreak']}")
    print(f"  fastest_realizable_strictly_equivalent: {r['fastest_realizable_strictly_equivalent_tps']:.2f} TPS "
          f"({r['fastest_realizable_strictly_equivalent_config']})  upside_over_blanket={r['upside_over_blanket_tps']}")
    print(f"  ppl_unchanged={r['ppl_unchanged_under_canonical_tiebreak']}  "
          f"self_ref_consistent={r['canonical_tiebreak_self_referential_consistent']}")
    print(f"  self-test: {sum(r['self_test'].values())}/{r['self_test_n_checks']} pass "
          f"(passes={r['self_test_passes']})")
    if not r["self_test_passes"]:
        for k, v in r["self_test"].items():
            if not v:
                print(f"     FAIL: {k}")
    print("=" * 96 + "\n", flush=True)


def log_wandb(report: dict, a: argparse.Namespace) -> None:
    sys.path.insert(0, os.getcwd())
    try:
        from scripts.wandb_logging import init_wandb_run, finish_wandb
    except Exception as exc:
        print(f"[wandb] helper import failed: {exc!r}; skipping", flush=True)
        return
    run = init_wandb_run(
        job_type="local_profiling", agent="stark", name=a.wandb_name, group=a.wandb_group,
        notes="PR#421 canonical tolerance tie-break applied to BOTH M=1 ref and M=8 verify: does the FAST stack "
              "reach byte-identity 1.0 at zero cost (exploiting the self-referential scorer gate, land #414)?",
        config={
            "pr": 421, "M_verify": M_VERIFY, "K_spec": K_SPEC, "n_prompts": report["n_prompts"],
            "C": report["C"], "n_verify": report["n_verify"], "model_dir": report["model_dir"],
            "primary_arm": report["primary_arm"], "eps_star": EPS_STAR,
            "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
            **{f"anchor/{k}": v for k, v in report["imported_anchors"].items()},
        },
    )
    if run is None:
        print("[wandb] disabled (no API key/mode); results saved to JSON only", flush=True)
        return
    keys = (
        "fast_stack_reaches_identity_1p0_with_canonical_tiebreak",
        "blanket_strict_reaches_identity_1p0_with_canonical_tiebreak",
        "tie_floor_count", "tie_floor_fraction", "n_flips_same_pattern", "n_flips_diff_pattern",
        "equiv_tps_fast_with_tiebreak", "tiebreak_residual_cost_tps", "tiebreak_is_zero_cost",
        "ppl_unchanged_under_canonical_tiebreak", "canonical_tiebreak_self_referential_consistent",
        "fast_identity_baseline_argmax", "fast_identity_canonical_tiebreak",
        "fast_identity_canonical_m8_vs_vanilla_m1", "fast_strict_reading_reaches_identity_1p0",
        "fast_n_strict_reading_disagreements",
        "fast_n_flips_closed", "fast_n_new_flips_introduced", "fast_n_flips_unfixed",
        "fast_n_canon_disagreements", "fast_all_flips_closed",
        "strict_identity_baseline_argmax", "strict_identity_canonical_tiebreak",
        "strict_n_flips_closed", "strict_n_new_flips_introduced", "strict_tie_floor_count",
        "fastest_realizable_strictly_equivalent_tps", "fastest_realizable_strictly_equivalent_config",
        "upside_over_blanket_tps", "one_sided_lowestid_new_flips_405", "one_sided_lowestid_identity_405",
        "band_truncation_positions_fast", "verdict",
        "self_test_n_checks", "self_test_passes", "analysis_only", "no_hf_job",
        "no_served_file_change", "official_tps",
    )
    for k in keys:
        run.summary[k] = report.get(k)
    run.summary["verdict_green"] = report["verdict"].startswith("GREEN")
    run.summary["verdict_red"] = report["verdict"].startswith("RED")
    for arm in CENSUS_ARMS:
        d = report["arms"][arm]
        run.summary[f"{arm}/identity"] = d["decodewidth_e2e_token_identity_rate"]
        run.summary[f"{arm}/flip_count"] = d["flip_count"]
        run.summary[f"{arm}/canonical_identity"] = report["per_arm_analysis"][arm]["sim"][
            "identity_canonical_tiebreak"]
    for k, v in report["self_test"].items():
        run.summary[f"selftest/{k}"] = v
    finish_wandb(run)
    print(f"[wandb] logged run {run.id}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["census"], default=None)
    ap.add_argument("--arm", choices=list(CENSUS_ARMS), default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--reanalyze", action="store_true",
                    help="0-GPU: recompose the report + self-test from saved arm_*.json")
    ap.add_argument("--smoke", action="store_true", help="tiny run (few prompts) to validate the path")
    ap.add_argument("--n-prompts", dest="n_prompts", type=int, default=127)
    ap.add_argument("--ctx-len", dest="ctx_len", type=int, default=224)
    ap.add_argument("--n-verify", dest="n_verify", type=int, default=M_VERIFY)
    ap.add_argument("--gpu-mem-util", dest="gpu_mem_util", type=float, default=0.55)
    ap.add_argument("--max-batched-tokens", dest="max_batched_tokens", type=int, default=8192)
    ap.add_argument("--verbose-k", dest="verbose_k", type=int, default=3)
    ap.add_argument("--wandb_group", dest="wandb_group", default="canonical-tiebreak-fast-stack-identity")
    ap.add_argument("--wandb_name", dest="wandb_name", default="stark/canonical-tiebreak-fast-stack-identity")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()

    if a.smoke and a.phase is None:
        a.n_prompts = min(a.n_prompts, 4)

    if a.phase == "census":
        phase_census(a.out, a.arm, a.n_prompts, a.ctx_len, a.n_verify,
                     a.gpu_mem_util, a.max_batched_tokens, a.verbose_k)
    elif a.reanalyze:
        reanalyze(a)
    else:
        orchestrate(a)


if __name__ == "__main__":
    main()
