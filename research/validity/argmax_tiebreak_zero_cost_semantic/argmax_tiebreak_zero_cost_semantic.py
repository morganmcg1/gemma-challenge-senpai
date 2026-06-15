#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #405 (stark) -- Is the lowest-index argmax tie-break a ZERO-cost deterministic semantic?

THE QUESTION
------------
My #397 (`g3954eh3`) priced the *realizable* decode-width identity fix as a position-selective
higher-precision attention reduction at **2.6 TPS** (0.236 near-tie steps x 11-TPS eta_attn). It also
observed that the sole decode-width identity residual is a **pure argmax coin-flip**: all 4/4 observed
flips (1 pinned + 3 heuristic) have the M=1 reference token as the M=8 **top-2**, exactly 0.125 nat (the
bf16 floor) below the M=8 top-1, and the M=1 token is the **lower token-id** of the tied pair.

This card decides whether there is a **ZERO-cost** alternative hiding in that mechanism. If the
divergence is purely that the served argmax breaks a 0.125-nat tie in a reduction-order-dependent way,
then replacing the argmax tie-break with a **deterministic lowest-index rule** (resolve any <= eps
near-tie to the lower token id) could recover identity 1.0 *for free* -- no higher-precision attention,
no extra reduction, just a different (arguably more correct) tie-break over the values the argmax
already holds in-register. That removes the 2.6-TPS cost from the rebuild ledger entirely.

THE CRUX (and the killer risk)
------------------------------
A *global* lowest-id tie-break is free ONLY IF it is also SAFE: it must fix every flip AND introduce
**zero new flips** at currently-correct near-tie positions. A correct near-tie where the served top-1
( == M=1 token) is the HIGHER id would be BROKEN by a lowest-id override (it would pick the lower-id
runner-up). So the load-bearing measurement is NOT "do the 4 flips have M1 = lower id" (selecting on
flips) -- it is "across ALL near-tie positions, is the M=1 reference ALWAYS the lower id of the served
top-2?". #397 explicitly flagged this as the open n=4 risk. This card answers it by a full per-position
census, not just the flips.

SCOPE: LOCAL A10G post-hoc analysis. analysis_only / no_hf_job / no_served_file_change / official_tps=0.
We characterise the EXISTING served argmax/tie-break semantics and the cost of a lowest-id rule
analytically + by micro-measurement. No served file is touched; the int4 path is READ only.

DELIVERABLES (W&B summary/)
  served_argmax_tiebreak_rule (str); m1_reference_uses_lowest_index (bool);
  tie_identifiable_from_fast_path (bool); lowest_index_tiebreak_is_free (bool);
  tiebreak_fix_tps_cost (0.0 or 2.598); free_tiebreak_recovers_identity_1p0 (bool);
  free_tiebreak_new_flips (int, must be 0 to be free); decode_width_identity_is_free (bool);
  reconciles_397_selective_cost (bool); PRIMARY argmax_tiebreak_zero_cost_self_test_passes (>=20 asserts).

TWO ARMS (reuse #381/#397; isolated subprocesses, pin set per-arm in ENV):
  heuristic -- stock vLLM (VLLM_BATCH_INVARIANT=0): the FAST, *served* verify-width attention. PRIMARY
               arm -- a free tie-break would be applied HERE (the deployed frontier does not pay the pin).
  pinned    -- VLLM_BATCH_INVARIANT=1 (num_splits=1 + aten batch-invariant): the strict candidate where
               #381 localised the lone residual flip. Control + #381/#397 reproduction.
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
# #381 (this student) decode-width residual at the literal size_m=8 verify geometry:
PINNED_IDENTITY_381 = 0.9988751406074241       # 888/889 -- 1 flip
HEURISTIC_IDENTITY_381 = 0.9966254218222722    # 886/889 -- 3 flips (all knife-edge, margins 0.125)
PINNED_FLIP_COUNT_381 = 1
HEURISTIC_FLIP_COUNT_381 = 3

# #397 (this student, `g3954eh3`) -- the selective-recompute cost this card tries to *zero out*:
SELECTIVE_FIX_TPS_COST_397 = 2.5984251968503935  # = f_step_band(0.125)=0.23622 x eta_attn=11.0
F_STEP_BAND_397 = 0.23622047244094488
FA_SLIDING0_TPS_COST = 11.0                      # eta_attn: blanket attention strict tax (#38/#397)
ALL4_FLIPS_M1_LOWER_ID_397 = True                # #397: 4/4 flips had M1 = lower-id of the M8 top-2

# Strict-base context from the #397/#405 PR baseline:
OFFICIAL_BASELINE = 481.53                        # #52 deployed frontier TPS (this leg adds 0)
CORRECTED_STRICT_BASE_390 = 471.42               # #390 5y64zbjz realized strict base
GAP_TO_500 = 28.58
BAND_CEILING = 509.78

K_SPEC = 7
M_VERIFY = K_SPEC + 1                             # = 8, the deployed decode-verify query width
IDENTITY_EPS = 1e-12
EPS_STAR = 0.125                                  # bf16 floor; the band that covers every observed flip
BAND_THRESHOLDS = (0.125, 0.25, 0.5)
BAND_TOL = 1e-9

HYBRID_PREFIX_COMMIT = 32                         # Gemma-4 hybrid prefix-cache commit granularity (#381)

MODEL_CANDIDATES = [
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots"
    ),
    "/tmp/osoi5-v0-baked",
]
PROMPTS_JSONL = "official/main_bucket/shared_resources/speed_benchmark/data/ppl_ground_truth_tokens.jsonl"

OUT_DIR = Path("research/validity/argmax_tiebreak_zero_cost_semantic")
ARMS = ("heuristic", "pinned")
PRIMARY_ARM = "heuristic"   # the SERVED (fast) path -- where a free tie-break would actually be applied


# --------------------------------------------------------------------------------------
# Small helpers (reused from #381/#397)
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


def block_align(n: int) -> int:
    return (n // HYBRID_PREFIX_COMMIT) * HYBRID_PREFIX_COMMIT


def _sorted_logprobs(entry) -> list[tuple[int, float]]:
    """(token_id, logprob) pairs sorted by logprob descending (rank 0 first)."""
    return sorted(((int(t), float(getattr(lp, "logprob", lp))) for t, lp in entry.items()),
                  key=lambda kv: kv[1], reverse=True)


def _argmax_from_logprob_entry(entry) -> int:
    return int(max(entry.items(), key=lambda kv: getattr(kv[1], "logprob", kv[1]))[0])


# ======================================================================================
# torch.argmax tie-break PROBE -- what does the served argmax actually do on a TRUE tie?
# ======================================================================================
def argmax_tiebreak_probe(torch) -> dict:
    """Micro-characterise the tie-break convention of the served greedy argmax (torch.argmax over
    logits). vLLM greedy sampling reduces to argmax; on a BITWISE tie the index convention decides.
    We test CPU + CUDA, scalar + vectorised, and repeat for stability. We ALSO note that the observed
    0.125-nat flips are NOT bitwise ties (one bf16 ULP apart), so argmax there is strict-greater-wins;
    the 'tie-break' the card proposes is an IMPOSED <=eps override, not torch.argmax's native behaviour.
    """
    dev = torch.device("cuda:0")
    out: dict = {}

    def first_or_last(vec_vals: list[float]) -> str:
        t_cpu = torch.tensor(vec_vals, dtype=torch.float32)
        t_cuda = t_cpu.to(dev)
        i_cpu = int(torch.argmax(t_cpu))
        i_cuda = int(torch.argmax(t_cuda))
        return i_cpu, i_cuda

    # a single tie between idx 1 and idx 3 (both == max). lowest-index => 1, highest-index => 3.
    vals = [0.0, 5.0, 1.0, 5.0, 2.0]
    i_cpu, i_cuda = first_or_last(vals)
    out["tie_idx_cpu"] = i_cpu
    out["tie_idx_cuda"] = i_cuda
    out["cpu_returns_lowest_index"] = bool(i_cpu == 1)
    out["cuda_returns_lowest_index"] = bool(i_cuda == 1)

    # stability of CUDA argmax across repeats (is the convention deterministic run-to-run?)
    reps = []
    big = torch.zeros(4096, 2048, dtype=torch.float32, device=dev)
    big[:, 700] = 5.0
    big[:, 1900] = 5.0   # tie between col 700 and 1900 for every row; lowest-index => 700
    for _ in range(8):
        reps.append(int(torch.argmax(big, dim=-1)[0]))
    out["cuda_vectorised_tie_idx"] = reps[0]
    out["cuda_vectorised_lowest_index"] = bool(reps[0] == 700)
    out["cuda_argmax_stable_across_repeats"] = bool(len(set(reps)) == 1)

    # bf16 one-ULP gap: are two adjacent bf16 logits distinguishable by argmax? (yes -> strict wins)
    a = torch.tensor([1.0], dtype=torch.bfloat16, device=dev)
    one_ulp = torch.nextafter(a, a + 1)
    out["bf16_one_ulp_gap_nats"] = float((one_ulp - a).float().item())
    pair = torch.cat([a, one_ulp]).to(dev)              # [1.0, 1.0+ULP]
    out["bf16_strict_picks_larger"] = bool(int(torch.argmax(pair)) == 1)

    if out["cuda_returns_lowest_index"] and out["cuda_vectorised_lowest_index"]:
        rule = "lowest_index_first (torch.argmax returns the FIRST/lowest index on a bitwise tie)"
    elif (not out["cuda_returns_lowest_index"]) and (out["tie_idx_cuda"] == 3):
        rule = "highest_index_last (torch.argmax returns the LAST/highest index on a bitwise tie)"
    else:
        rule = "reduction_order_dependent (no stable index convention on a bitwise tie)"
    if not out["cuda_argmax_stable_across_repeats"]:
        rule = "reduction_order_dependent_nondeterministic"
    out["served_argmax_tiebreak_rule"] = rule
    return out


# ======================================================================================
# PHASE: one arm. Full per-position census: M=8 served top-k AND M=1 AR top-k, aligned.
# ======================================================================================
def phase_arm(out_path: str, arm: str, n_prompts: int, ctx_len: int, n_verify: int,
              gpu_mem_util: float, max_batched_tokens: int, verbose_k: int) -> None:
    import torch
    from vllm import LLM, SamplingParams

    batch_invariant_env = os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1"
    model_dir = resolve_model_dir()
    C = block_align(ctx_len)
    print(f"[arm:{arm}] model={model_dir} C(prefix)={C} n_verify={n_verify} "
          f"VLLM_BATCH_INVARIANT={batch_invariant_env}", flush=True)

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

    argmax_probe = argmax_tiebreak_probe(torch)

    # M=1 AR continuation WITH per-step top-5 logprobs (the M=1 reference census).
    sp_gen = SamplingParams(temperature=0.0, max_tokens=n_verify, logprobs=5, detokenize=False)
    sp_warm = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False)
    # M=8 served verify chunk WITH per-position top-5 prompt_logprobs (the served census).
    sp_chunk = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=5,
                              skip_reading_prefix_cache=False, detokenize=False)

    rows = [json.loads(l) for l in open(PROMPTS_JSONL)][:n_prompts]
    per_prompt = []
    n_match = n_total = 0
    n_det_m1 = n_det_m8 = n_within = 0
    n_chunk_isolated = 0
    chunk_width_obs = []
    n_computed_rows_total = 0

    positions: list[dict] = []   # FULL per-position census (every readable suffix position)
    flip_details: list[dict] = []

    for ri, rec in enumerate(rows):
        src = list(rec.get("context_token_ids", [])) + list(rec.get("target_token_ids", []))
        if len(src) < C + 1:
            continue
        prefix = src[:C]

        llm.generate([{"prompt_token_ids": prefix}], sp_warm, use_tqdm=False)

        # Step A: M=1 AR greedy continuation + per-step top-5 (M1 census) + det control
        outA = llm.generate([{"prompt_token_ids": prefix}], sp_gen, use_tqdm=False)[0]
        cont = list(outA.outputs[0].token_ids)[:n_verify]
        m1_lp_steps = list(outA.outputs[0].logprobs or [])[:n_verify]   # one dict per generated token
        outA2 = llm.generate([{"prompt_token_ids": prefix}], sp_gen, use_tqdm=False)[0]
        cont2 = list(outA2.outputs[0].token_ids)[:n_verify]
        det_m1 = int(cont == cont2)
        if len(cont) < n_verify:
            continue
        full = prefix + cont

        # Step B: M=8 served verify chunk + det control
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

        # per-position census: align M=8 served top-k with M=1 AR top-k (both predict cont[j], j=i-C)
        match = total = 0
        prompt_min_gap = float("inf")
        for p in suffix_pos:
            j = p - C                       # suffix index; M1 census lives at gen-step j
            m1_tok = full[p]                # == cont[j], the M=1 AR argmax (M=1 reference)
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
            m8_pair_max_id = max(m8_top1_id, m8_top2_id)
            m1_is_lower_of_m8_pair = bool(m1_in_m8_top2 and m1_tok == m8_pair_min_id)
            m8_top1_is_lower_of_pair = bool(m8_top1_id == m8_pair_min_id)

            # --- M=1 AR census at the same position (from the generation logprobs) ---
            m1_entry = m1_lp_steps[j] if 0 <= j < len(m1_lp_steps) else None
            m1_sl = _sorted_logprobs(m1_entry) if m1_entry else []
            m1_top1_id = m1_sl[0][0] if m1_sl else None
            m1_top2_id = m1_sl[1][0] if len(m1_sl) >= 2 else None
            m1_self_gap = (m1_sl[0][1] - m1_sl[1][1]) if len(m1_sl) >= 2 else None
            m1_argmax_matches_token = bool(m1_top1_id == m1_tok) if m1_top1_id is not None else None
            # is the M1 token the lower id of ITS OWN top-2? (does M1 itself resolve toward lower id?)
            m1_self_pair_min = (min(m1_top1_id, m1_top2_id)
                                if (m1_top1_id is not None and m1_top2_id is not None) else None)
            m1_picks_lower_of_own_pair = (bool(m1_top1_id == m1_self_pair_min)
                                          if m1_self_pair_min is not None else None)

            if math.isfinite(m8_gap):
                prompt_min_gap = min(prompt_min_gap, m8_gap)

            rec_pos = {
                "prompt_idx": ri, "pos": p, "j": j,
                "m8_gap": round(m8_gap, 6),
                "m8_top1_id": m8_top1_id, "m8_top2_id": m8_top2_id,
                "m8_pair_min_id": m8_pair_min_id, "m8_pair_max_id": m8_pair_max_id,
                "m8_top1_is_lower_of_pair": m8_top1_is_lower_of_pair,
                "m1_tok_id": m1_tok, "is_flip": is_flip,
                "m1_in_m8_top2": m1_in_m8_top2, "m1_in_m8_top5": m1_in_m8_top5,
                "m1_is_lower_of_m8_pair": m1_is_lower_of_m8_pair,
                "m1_self_gap": (round(m1_self_gap, 6) if m1_self_gap is not None else None),
                "m1_argmax_matches_token": m1_argmax_matches_token,
                "m1_picks_lower_of_own_pair": m1_picks_lower_of_own_pair,
            }
            positions.append(rec_pos)
            if not is_flip:
                match += 1
            else:
                flip_details.append({**rec_pos,
                                     "m1_margin_in_m8": round(m8_top1_lp - dict(sl).get(m1_tok, float("nan")), 6)
                                     if m1_in_m8_top5 else None})
        if math.isfinite(prompt_min_gap):
            pass

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
                  f"match={match}/{total} det_m1={det_m1} det_m8={det_m8} within={within}", flush=True)

    n_seq = len(per_prompt)
    identity = (n_match / n_total) if n_total else float("nan")
    out = {
        "phase": "arm", "arm": arm, "model_dir": model_dir,
        "vllm_batch_invariant_env": batch_invariant_env,
        "attn_is_batch_invariant": attn_is_batch_invariant,
        "argmax_tiebreak_probe": argmax_probe,
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
        "nan_clean": bool(math.isfinite(identity)),
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
        "per_prompt": per_prompt,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[arm:{arm}] identity={identity:.7f} flips={len(flip_details)} "
          f"positions={n_total} peak={out['peak_gpu_gb']:.1f}GB", flush=True)
    print(f"ARM_DONE {out_path}", flush=True)


# ======================================================================================
# ANALYSIS (0-GPU): the census-driven near-tie / lowest-id rule logic
# ======================================================================================
def _band_key(thr: float) -> str:
    return f"{thr:g}".replace(".", "p")


def census_band_counts(positions: list[dict]) -> dict:
    """Position-level near-tie counts + flip gaps, mirrors #397 bands for continuity."""
    gaps = [p["m8_gap"] for p in positions]
    flip_gaps = [p["m8_gap"] for p in positions if p["is_flip"]]
    total = len(gaps)
    out = {
        "total_positions": total,
        "flip_count": len(flip_gaps),
        "flip_gaps": [round(g, 6) for g in flip_gaps],
        "max_flip_gap": (max(flip_gaps) if flip_gaps else None),
        "median_position_gap": (round(statistics.median(gaps), 6) if gaps else None),
    }
    for thr in BAND_THRESHOLDS:
        key = _band_key(thr)
        out[f"position_count_{key}"] = sum(1 for g in gaps if g <= thr + BAND_TOL)
        out[f"position_frac_{key}"] = (out[f"position_count_{key}"] / total) if total else float("nan")
        out[f"flips_in_{key}"] = sum(1 for g in flip_gaps if g <= thr + BAND_TOL)
    return out


def simulate_lowest_id_rule(positions: list[dict], eps: float) -> dict:
    """Apply the GLOBAL deterministic lowest-id tie-break to the served (M=8) argmax at every position:

        rule_pick(p) = min(top1_id, top2_id)   if  m8_gap(p) <= eps     (near-tie -> lower id wins)
                     = top1_id                  otherwise               (clear win -> unchanged)

    Then compare to the M=1 reference token. Decompose into fixed flips, NEW flips (currently-correct
    positions the rule breaks), and unfixed flips. A rule is SAFE+COMPLETE iff new_flips==0 AND it
    fixes every flip (rule_identity == 1.0)."""
    n_total = len(positions)
    n_baseline_flips = sum(1 for p in positions if p["is_flip"])
    fixed = new = unfixed = 0
    n_rule_correct = 0
    n_near_tie = n_near_tie_correct = 0
    n_near_tie_m1_not_in_top2 = 0
    for p in positions:
        near = p["m8_gap"] <= eps + BAND_TOL
        rule_pick = p["m8_pair_min_id"] if near else p["m8_top1_id"]
        baseline_correct = (p["m8_top1_id"] == p["m1_tok_id"])
        rule_correct = (rule_pick == p["m1_tok_id"])
        n_rule_correct += int(rule_correct)
        if near:
            n_near_tie += 1
            if baseline_correct:
                n_near_tie_correct += 1
            if not p["m1_in_m8_top2"]:
                n_near_tie_m1_not_in_top2 += 1
        if (not baseline_correct) and rule_correct:
            fixed += 1
        elif baseline_correct and (not rule_correct):
            new += 1
        elif (not baseline_correct) and (not rule_correct):
            unfixed += 1
    rule_identity = (n_rule_correct / n_total) if n_total else float("nan")
    return {
        "eps": eps,
        "n_total": n_total,
        "n_baseline_flips": n_baseline_flips,
        "fixed_flips": fixed,
        "new_flips": new,
        "unfixed_flips": unfixed,
        "rule_identity": rule_identity,
        "rule_flip_count": n_total - n_rule_correct,
        "n_near_tie": n_near_tie,
        "n_near_tie_correct": n_near_tie_correct,        # the denominator of NEW-flip risk
        "n_near_tie_m1_not_in_top2": n_near_tie_m1_not_in_top2,
        "recovers_identity_1p0": bool((n_total - n_rule_correct) == 0),
    }


def m1_lowest_index_census(positions: list[dict], eps: float) -> dict:
    """Across ALL near-tie positions (m8_gap <= eps) where the M=1 token is in the served top-2, is the
    M=1 reference ALWAYS the lower id of the served top-2? This is the structural test (NOT selecting on
    flips) behind `m1_reference_uses_lowest_index`. Also reports whether M1 resolves toward the lower id
    of its OWN top-2 (evidence the M1 argmax itself carries a lowest-index tie-break)."""
    near = [p for p in positions if p["m8_gap"] <= eps + BAND_TOL]
    near_in_top2 = [p for p in near if p["m1_in_m8_top2"]]
    n = len(near_in_top2)
    n_m1_lower = sum(1 for p in near_in_top2 if p["m1_is_lower_of_m8_pair"])
    n_m1_higher = n - n_m1_lower
    # M1-own-pair lowest-id evidence (where M1 census present)
    own = [p for p in near if p["m1_picks_lower_of_own_pair"] is not None]
    n_own_lower = sum(1 for p in own if p["m1_picks_lower_of_own_pair"])
    # m1 self gap == 0 means a BITWISE tie in the clean M1 path (argmax tie-break actually fires)
    m1_true_tie = [p for p in near if p["m1_self_gap"] is not None and p["m1_self_gap"] <= BAND_TOL]
    return {
        "eps": eps,
        "n_near_tie_m1_in_top2": n,
        "n_m1_is_lower": n_m1_lower,
        "n_m1_is_higher": n_m1_higher,
        "frac_m1_lower": (n_m1_lower / n) if n else float("nan"),
        "m1_always_lower_id": bool(n > 0 and n_m1_higher == 0),
        "n_m1_own_pair_resolved": len(own),
        "n_m1_own_pair_lower": n_own_lower,
        "frac_m1_own_pair_lower": (n_own_lower / len(own)) if own else float("nan"),
        "n_m1_self_bitwise_tie": len(m1_true_tie),
    }


def tie_identifiable(fast_positions: list[dict], flip_coords: set, eps: float) -> dict:
    """Does the FAST (served) attention place the M=1 token in its top-2 at the flip positions, so the
    near-tie is identifiable from the fast path alone (no higher-precision recompute)? Checks (a) every
    fast-arm flip, and (b) the union of all 4 flip coords across both arms looked up in the fast census.
    """
    fast_flips = [p for p in fast_positions if p["is_flip"]]
    fast_flip_ok = all((p["m1_in_m8_top2"] and p["m8_gap"] <= eps + BAND_TOL) for p in fast_flips) \
        if fast_flips else None
    by_coord = {(p["prompt_idx"], p["pos"]): p for p in fast_positions}
    union_hits, union_total = 0, 0
    union_detail = []
    for (pi, ps) in sorted(flip_coords):
        union_total += 1
        p = by_coord.get((pi, ps))
        ok = bool(p is not None and p["m1_in_m8_top2"] and p["m8_gap"] <= eps + BAND_TOL)
        union_hits += int(ok)
        union_detail.append({"prompt_idx": pi, "pos": ps, "in_fast_census": p is not None,
                             "m1_in_fast_top2": (p["m1_in_m8_top2"] if p else None),
                             "fast_gap": (p["m8_gap"] if p else None), "identifiable": ok})
    return {
        "eps": eps,
        "n_fast_flips": len(fast_flips),
        "all_fast_flips_identifiable": fast_flip_ok,
        "union_flip_coords": union_total,
        "union_identifiable_hits": union_hits,
        "all_union_flips_identifiable_from_fast": bool(union_total > 0 and union_hits == union_total),
        "union_detail": union_detail,
    }


# ======================================================================================
# Compose + self-test + report
# ======================================================================================
def compose_and_report(arms: dict, a: argparse.Namespace) -> dict:
    primary = arms[PRIMARY_ARM]                  # heuristic / fast / served
    pos_primary = primary["positions"]
    bands_primary = census_band_counts(pos_primary)

    # union of flip coords across BOTH arms (the "4 flip positions" the card references)
    flip_coords = set()
    for arm, d in arms.items():
        for p in d["positions"]:
            if p["is_flip"]:
                flip_coords.add((p["prompt_idx"], p["pos"]))

    # ---- Deliverable 1: served argmax tie-break rule + m1_reference_uses_lowest_index ----
    served_argmax_tiebreak_rule = primary["argmax_tiebreak_probe"]["served_argmax_tiebreak_rule"]
    m1_census = {arm: {ek: m1_lowest_index_census(d["positions"], thr)
                       for thr, ek in ((t, _band_key(t)) for t in BAND_THRESHOLDS)}
                 for arm, d in arms.items()}
    # the load-bearing structural fact, at eps_star, on the PRIMARY (served) arm:
    m1_primary_epsstar = m1_census[PRIMARY_ARM][_band_key(EPS_STAR)]
    m1_reference_uses_lowest_index = bool(m1_primary_epsstar["m1_always_lower_id"])

    # ---- Deliverable 2: tie identifiable from fast path? + is the rule free? ----
    # Load-bearing question: can the SERVED (fast) path resolve ITS OWN divergences from its in-register
    # top-2 (no higher-precision recompute)? The cross-arm union (incl. the pinned-only flip coord) is
    # reported as supporting detail -- a pinned-arm flip coord may be a clear win on the fast path.
    tie_id = tie_identifiable(arms[PRIMARY_ARM]["positions"], flip_coords, EPS_STAR)
    tie_identifiable_from_fast_path = bool(tie_id["all_fast_flips_identifiable"] is True)

    # ---- Deliverable 3: global lowest-id rule simulation on the served (primary) arm ----
    rule_sims = {arm: {ek: simulate_lowest_id_rule(d["positions"], thr)
                       for thr, ek in ((t, _band_key(t)) for t in BAND_THRESHOLDS)}
                 for arm, d in arms.items()}
    sim_primary = rule_sims[PRIMARY_ARM][_band_key(EPS_STAR)]
    free_tiebreak_new_flips = int(sim_primary["new_flips"])
    free_tiebreak_recovers_identity_1p0 = bool(sim_primary["recovers_identity_1p0"])

    # the mechanism is FREE iff the tie is readable from the in-register fast-path top-2 (no recompute).
    # it is a VALID free fix iff it ALSO recovers identity 1.0 with zero new flips.
    lowest_index_tiebreak_is_free = bool(
        tie_identifiable_from_fast_path                      # margin readable for free
        and free_tiebreak_recovers_identity_1p0              # rule reaches 1.0
        and free_tiebreak_new_flips == 0)                    # ... without breaking correct positions

    # ---- Deliverable 4: the decision ----
    decode_width_identity_is_free = bool(lowest_index_tiebreak_is_free)
    tiebreak_fix_tps_cost = 0.0 if decode_width_identity_is_free else SELECTIVE_FIX_TPS_COST_397

    # ---- reconciliation with #397 ----
    reproduces_flip_structure = bool(
        census_band_counts(arms["pinned"]["positions"])["flip_count"] == PINNED_FLIP_COUNT_381
        and census_band_counts(arms["heuristic"]["positions"])["flip_count"] == HEURISTIC_FLIP_COUNT_381
        and bands_primary["max_flip_gap"] is not None
        and bands_primary["max_flip_gap"] <= EPS_STAR + BAND_TOL)
    # cost logic is consistent: the 2.6 stands as cheapest IFF the free rule fails; it is superseded
    # (replaced by 0.0) IFF the free rule holds. Either way our accounting is consistent with #397.
    cost_logic_consistent = bool(
        (decode_width_identity_is_free and tiebreak_fix_tps_cost == 0.0)
        or ((not decode_width_identity_is_free) and tiebreak_fix_tps_cost == SELECTIVE_FIX_TPS_COST_397))
    reconciles_397_selective_cost = bool(reproduces_flip_structure and cost_logic_consistent)

    # ---- verdict ----
    if not (bands_primary["flip_count"] > 0):
        verdict = "NO_RESIDUAL_identity_already_1p0"
    elif decode_width_identity_is_free:
        verdict = "GREEN_decode_width_identity_is_FREE_zero_cost_lowest_id_rule"
    elif tie_identifiable_from_fast_path and free_tiebreak_new_flips > 0:
        verdict = "RED_free_rule_unsafe_global_lowest_id_introduces_new_flips_2p6_stands"
    elif not tie_identifiable_from_fast_path:
        verdict = "RED_tie_not_identifiable_from_fast_path_needs_2p6_precision"
    else:
        verdict = "RED_free_rule_incomplete_2p6_stands"

    self_test, n_checks = build_self_test(arms, bands_primary, m1_census, rule_sims, tie_id,
                                          served_argmax_tiebreak_rule, sim_primary,
                                          decode_width_identity_is_free, tiebreak_fix_tps_cost,
                                          reproduces_flip_structure)
    argmax_tiebreak_zero_cost_self_test_passes = bool(all(self_test.values()) and n_checks >= 20)

    report = {
        "pr": 405,
        "leg": "argmax tie-break zero-cost semantic: is a deterministic lowest-id rule a FREE "
               "decode-width identity fix, or genuinely 2.6-TPS-priced (#397)? (local A10G, analysis-only)",
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
        "primary_arm": PRIMARY_ARM, "eps_star": EPS_STAR,
        "imported_anchors": {
            "pinned_identity_381": PINNED_IDENTITY_381,
            "heuristic_identity_381": HEURISTIC_IDENTITY_381,
            "pinned_flip_count_381": PINNED_FLIP_COUNT_381,
            "heuristic_flip_count_381": HEURISTIC_FLIP_COUNT_381,
            "selective_fix_tps_cost_397": SELECTIVE_FIX_TPS_COST_397,
            "f_step_band_397": F_STEP_BAND_397,
            "fa_sliding0_tps_cost": FA_SLIDING0_TPS_COST,
            "all4_flips_m1_lower_id_397": ALL4_FLIPS_M1_LOWER_ID_397,
            "official_baseline": OFFICIAL_BASELINE,
            "corrected_strict_base_390": CORRECTED_STRICT_BASE_390,
            "gap_to_500": GAP_TO_500, "band_ceiling": BAND_CEILING, "M_verify": M_VERIFY,
        },
        # ---- REQUIRED deliverable fields ----
        "served_argmax_tiebreak_rule": served_argmax_tiebreak_rule,
        "m1_reference_uses_lowest_index": m1_reference_uses_lowest_index,
        "tie_identifiable_from_fast_path": tie_identifiable_from_fast_path,
        "lowest_index_tiebreak_is_free": lowest_index_tiebreak_is_free,
        "tiebreak_fix_tps_cost": tiebreak_fix_tps_cost,
        "free_tiebreak_recovers_identity_1p0": free_tiebreak_recovers_identity_1p0,
        "free_tiebreak_new_flips": free_tiebreak_new_flips,
        "decode_width_identity_is_free": decode_width_identity_is_free,
        "reconciles_397_selective_cost": reconciles_397_selective_cost,
        "argmax_tiebreak_zero_cost_self_test_passes": argmax_tiebreak_zero_cost_self_test_passes,  # PRIMARY
        # ---- supporting detail ----
        "verdict": verdict,
        "reproduces_flip_structure": reproduces_flip_structure,
        "cost_logic_consistent": cost_logic_consistent,
        "primary_bands": bands_primary,
        "m1_lowest_index_census": m1_census,
        "rule_simulation": rule_sims,
        "tie_identifiable_detail": tie_id,
        "flip_coords_union": sorted(list(flip_coords)),
        "n_flip_coords_union": len(flip_coords),
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
                "argmax_tiebreak_probe": d["argmax_tiebreak_probe"],
                "bands": census_band_counts(d["positions"]),
                "flip_details": d["flip_details"],
                "total_positions": d["total_positions"], "n_prompts": d["n_prompts"],
                "peak_gpu_gb": d["peak_gpu_gb"],
            } for arm, d in arms.items()
        },
        "self_test": self_test, "self_test_n_checks": n_checks,
        "C": primary["C"], "n_verify": primary["n_verify"],
        "n_prompts": primary["n_prompts"], "model_dir": primary["model_dir"],
    }
    return report


def build_self_test(arms, bands_primary, m1_census, rule_sims, tie_id, argmax_rule, sim_primary,
                    is_free, tps_cost, reproduces_flip_structure):
    """>=20 boolean checks: per-arm determinism/geometry controls + census arithmetic + rule logic."""
    checks: dict = {}
    for arm, d in arms.items():
        checks[f"{arm}_determinism_m1_eq_1"] = bool(d["determinism_M1_vs_M1"] == 1.0)
        checks[f"{arm}_determinism_m8_eq_1"] = bool(d["determinism_M8_vs_M8"] == 1.0)
        checks[f"{arm}_within_eq_1"] = bool(d["within_batch_copy0_vs_copy1"] == 1.0)
        checks[f"{arm}_geometry_isolated"] = bool(d["chunk_isolated_fraction"] >= 0.99)
        ident = d["decodewidth_e2e_token_identity_rate"]
        checks[f"{arm}_arith_consistent"] = bool(
            math.isfinite(ident) and 0.0 <= ident <= 1.0
            and abs(d["decodewidth_e2e_divergence_rate"] - (1.0 - ident)) < 1e-9)
        b = census_band_counts(d["positions"])
        checks[f"{arm}_band_monotonic"] = bool(
            b["position_count_0p125"] <= b["position_count_0p25"] <= b["position_count_0p5"])
        # every flip lives in the tightest 0.125 band (knife-edge, reproduces #381/#397)
        checks[f"{arm}_all_flips_in_tightest_band"] = bool(
            b["flip_count"] == 0 or b["flips_in_0p125"] == b["flip_count"])
        # M1 census present + M1 argmax matches the generated token (alignment sanity)
        m1_ok = all(p["m1_argmax_matches_token"] in (True, None) for p in d["positions"])
        checks[f"{arm}_m1_census_argmax_aligned"] = bool(m1_ok)
        # rule simulation arithmetic closes: baseline_flips == fixed + unfixed at eps_star
        s = rule_sims[arm][_band_key(EPS_STAR)]
        checks[f"{arm}_rule_arith_closes"] = bool(
            s["n_baseline_flips"] == s["fixed_flips"] + s["unfixed_flips"]
            and s["rule_flip_count"] == s["unfixed_flips"] + s["new_flips"])

    # pin engaged in pinned arm; heuristic arm NOT batch-invariant (control separation)
    checks["pinned_attn_batch_invariant"] = bool(arms["pinned"].get("attn_is_batch_invariant"))
    checks["heuristic_not_batch_invariant"] = bool(not arms["heuristic"].get("attn_is_batch_invariant"))

    # reproduce #381/#397 flip counts (cross-card consistency)
    checks["reproduces_flip_structure_381_397"] = bool(reproduces_flip_structure)

    # argmax probe produced a definite rule string
    checks["argmax_rule_resolved"] = bool(isinstance(argmax_rule, str) and len(argmax_rule) > 0)
    checks["argmax_probe_stable"] = bool(
        arms[PRIMARY_ARM]["argmax_tiebreak_probe"].get("cuda_argmax_stable_across_repeats"))
    checks["bf16_strict_picks_larger"] = bool(
        arms[PRIMARY_ARM]["argmax_tiebreak_probe"].get("bf16_strict_picks_larger"))

    # primary-arm residual exists (else the question is moot)
    checks["primary_has_residual"] = bool(bands_primary["flip_count"] > 0)

    # rule monotonicity: a WIDER band can only flag >= near-ties (more new-flip opportunity)
    rp = rule_sims[PRIMARY_ARM]
    checks["near_tie_count_monotonic_in_eps"] = bool(
        rp[_band_key(0.125)]["n_near_tie"] <= rp[_band_key(0.25)]["n_near_tie"]
        <= rp[_band_key(0.5)]["n_near_tie"])

    # m1-lowest-index census denominator is non-empty at eps_star (the test is actually exercised)
    mc = m1_census[PRIMARY_ARM][_band_key(EPS_STAR)]
    checks["m1_census_nonempty_at_eps_star"] = bool(mc["n_near_tie_m1_in_top2"] > 0)
    # the new-flip count equals the number of correct near-ties whose served top-1 is the HIGHER id
    # (i.e. exactly the positions a lowest-id override breaks) -- internal consistency of the mechanism
    pos = arms[PRIMARY_ARM]["positions"]
    manual_new = sum(1 for p in pos
                     if p["m8_gap"] <= EPS_STAR + BAND_TOL
                     and (p["m8_top1_id"] == p["m1_tok_id"])           # currently correct
                     and (not p["m8_top1_is_lower_of_pair"]))          # top1 is the HIGHER id
    checks["new_flip_count_matches_higher_id_correct_near_ties"] = bool(
        manual_new == sim_primary["new_flips"])

    # decision/cost coherence
    checks["cost_zero_iff_free"] = bool((tps_cost == 0.0) == is_free)
    checks["tps_cost_is_0_or_397"] = bool(tps_cost in (0.0, SELECTIVE_FIX_TPS_COST_397))

    return checks, len(checks)


# ======================================================================================
# Orchestrator
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


def orchestrate(a: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arms = {arm: _run_arm(a, arm) for arm in ARMS}
    _finish(compose_and_report(arms, a), a)


def reanalyze(a: argparse.Namespace) -> None:
    arms = {}
    for arm in ARMS:
        p = OUT_DIR / f"arm_{arm}_result.json"
        if not p.exists():
            raise FileNotFoundError(f"--reanalyze needs {p} (run the GPU arms first)")
        arms[arm] = json.load(open(p))
    _finish(compose_and_report(arms, a), a)


def _finish(report: dict, a: argparse.Namespace) -> None:
    report_path = OUT_DIR / "argmax_tiebreak_zero_cost_semantic_results.json"
    json.dump(report, open(report_path, "w"), indent=2)
    _print_console(report)
    if not a.no_wandb:
        log_wandb(report, a)


def _print_console(report: dict) -> None:
    print("\n========== ARGMAX TIE-BREAK ZERO-COST SEMANTIC (PR #405) ==========", flush=True)
    print(f" VERDICT                                  : {report['verdict']}", flush=True)
    print(f" served_argmax_tiebreak_rule              : {report['served_argmax_tiebreak_rule']}", flush=True)
    print(f" m1_reference_uses_lowest_index           : {report['m1_reference_uses_lowest_index']}", flush=True)
    print(f" tie_identifiable_from_fast_path          : {report['tie_identifiable_from_fast_path']}", flush=True)
    print(f" lowest_index_tiebreak_is_free            : {report['lowest_index_tiebreak_is_free']}", flush=True)
    print(f" free_tiebreak_recovers_identity_1p0      : {report['free_tiebreak_recovers_identity_1p0']}", flush=True)
    print(f" free_tiebreak_new_flips (MUST be 0)      : {report['free_tiebreak_new_flips']}", flush=True)
    print(f" tiebreak_fix_tps_cost (0.0 or 2.598)     : {report['tiebreak_fix_tps_cost']}", flush=True)
    print(f" DECODE_WIDTH_IDENTITY_IS_FREE            : {report['decode_width_identity_is_free']}", flush=True)
    print(f" reconciles_397_selective_cost            : {report['reconciles_397_selective_cost']}", flush=True)
    primary = report["arms"][report["primary_arm"]]
    print(f" primary(served={report['primary_arm']}) identity   : "
          f"{primary['decodewidth_e2e_token_identity_rate']:.7f} "
          f"(flips={report['primary_bands']['flip_count']})", flush=True)
    sp = report["rule_simulation"][report["primary_arm"]]["0p125"]
    print(f" rule@0.125: fixed={sp['fixed_flips']} new={sp['new_flips']} unfixed={sp['unfixed_flips']} "
          f"rule_identity={sp['rule_identity']:.7f} "
          f"(near_tie_correct denom={sp['n_near_tie_correct']})", flush=True)
    print(f" SELF-TEST PASSES (PRIMARY)               : {report['argmax_tiebreak_zero_cost_self_test_passes']} "
          f"({sum(report['self_test'].values())}/{report['self_test_n_checks']} checks)", flush=True)
    fails = [k for k, v in report["self_test"].items() if not v]
    if fails:
        print(f"   self-test FAILS: {fails}", flush=True)
    print("===================================================================\n", flush=True)


def log_wandb(report: dict, a: argparse.Namespace) -> None:
    sys.path.insert(0, os.getcwd())
    try:
        from scripts.wandb_logging import init_wandb_run, finish_wandb
    except Exception as exc:
        print(f"[wandb] helper import failed: {exc!r}; skipping", flush=True)
        return
    run = init_wandb_run(
        job_type="local_profiling", agent="stark",
        name=a.wandb_name, group=a.wandb_group,
        notes="PR#405 argmax tie-break zero-cost semantic: is a deterministic lowest-id argmax rule a "
              "FREE decode-width identity fix, or genuinely 2.6-TPS-priced (#397)?",
        config={
            "pr": 405, "M_verify": M_VERIFY, "n_prompts": report["n_prompts"],
            "C": report["C"], "n_verify": report["n_verify"], "model_dir": report["model_dir"],
            "primary_arm": report["primary_arm"], "eps_star": EPS_STAR,
            "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
            **{f"anchor/{k}": v for k, v in report["imported_anchors"].items()},
        },
    )
    if run is None:
        print("[wandb] disabled (no API key/mode); results saved to JSON only", flush=True)
        return
    summary = {
        "argmax_tiebreak_zero_cost_self_test_passes": report["argmax_tiebreak_zero_cost_self_test_passes"],
        "served_argmax_tiebreak_rule": report["served_argmax_tiebreak_rule"],
        "m1_reference_uses_lowest_index": report["m1_reference_uses_lowest_index"],
        "tie_identifiable_from_fast_path": report["tie_identifiable_from_fast_path"],
        "lowest_index_tiebreak_is_free": report["lowest_index_tiebreak_is_free"],
        "tiebreak_fix_tps_cost": report["tiebreak_fix_tps_cost"],
        "free_tiebreak_recovers_identity_1p0": report["free_tiebreak_recovers_identity_1p0"],
        "free_tiebreak_new_flips": report["free_tiebreak_new_flips"],
        "decode_width_identity_is_free": report["decode_width_identity_is_free"],
        "reconciles_397_selective_cost": report["reconciles_397_selective_cost"],
        "verdict": report["verdict"],
        "verdict_green": report["verdict"].startswith("GREEN"),
        "verdict_red": report["verdict"].startswith("RED"),
        "reproduces_flip_structure": report["reproduces_flip_structure"],
        "self_test_n_checks": report["self_test_n_checks"],
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
    }
    pb = report["primary_bands"]
    for k in ("flip_count", "max_flip_gap", "position_count_0p125", "position_count_0p25",
              "position_count_0p5", "total_positions"):
        summary[f"primary/{k}"] = pb[k]
    for arm in ARMS:
        d = report["arms"][arm]
        summary[f"{arm}/identity"] = d["decodewidth_e2e_token_identity_rate"]
        summary[f"{arm}/flip_count"] = d["bands"]["flip_count"]
        s = report["rule_simulation"][arm]["0p125"]
        summary[f"{arm}/rule_new_flips_0p125"] = s["new_flips"]
        summary[f"{arm}/rule_fixed_flips_0p125"] = s["fixed_flips"]
        summary[f"{arm}/rule_identity_0p125"] = s["rule_identity"]
        summary[f"{arm}/near_tie_correct_0p125"] = s["n_near_tie_correct"]
        mc = report["m1_lowest_index_census"][arm]["0p125"]
        summary[f"{arm}/m1_always_lower_id_0p125"] = mc["m1_always_lower_id"]
        summary[f"{arm}/frac_m1_lower_0p125"] = mc["frac_m1_lower"]
        summary[f"{arm}/n_m1_self_bitwise_tie_0p125"] = mc["n_m1_self_bitwise_tie"]
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
    ap.add_argument("--phase", choices=["arm"], default=None)
    ap.add_argument("--arm", choices=list(ARMS), default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-test", dest="self_test", action="store_true",
                    help="run both arms + the PRIMARY self-test (default orchestrator path)")
    ap.add_argument("--reanalyze", action="store_true",
                    help="0-GPU: recompose the report + self-test from saved arm_*.json")
    ap.add_argument("--smoke", action="store_true", help="tiny run (few prompts) to validate the path")
    ap.add_argument("--n-prompts", dest="n_prompts", type=int, default=128)
    ap.add_argument("--ctx-len", dest="ctx_len", type=int, default=224)
    ap.add_argument("--n-verify", dest="n_verify", type=int, default=M_VERIFY)
    ap.add_argument("--gpu-mem-util", type=float, default=0.55)
    ap.add_argument("--max-batched-tokens", type=int, default=8192)
    ap.add_argument("--verbose-k", dest="verbose_k", type=int, default=3)
    ap.add_argument("--wandb_group", dest="wandb_group", default="argmax-tiebreak-zero-cost-semantic")
    ap.add_argument("--wandb_name", dest="wandb_name", default="stark/argmax-tiebreak-zero-cost-semantic")
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
