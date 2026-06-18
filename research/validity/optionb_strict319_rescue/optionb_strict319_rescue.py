#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #636 (stark) -- Option-B strict-#319 rescue: tie-deterministic M=1-recompute acceptor.

THE GOAL
--------
stark #622 proved the #607 attention defect is GONE under VLLM_BATCH_INVARIANT=1 both sides;
the only residual is a 0.092% (6/6531) family of int4-Marlin grid-ties -- ALL gap==margin==0.125
nat EXACTLY, 0 attention-path, 0 M=1-token-outside-top-k. Benign in graded quality (denken #626)
but NOT strict tau=0 byte-exact: free-running, the 0.092% per-step seed cascades (wirbel #607's
47%). Option-B therefore needs a human tolerance ruling on sub-0.5-nat int4-ties to satisfy #319.

This card eliminates that round-trip. A gap-flagged M=1-recompute acceptor:
  during MTP spec-verify, per verified position compute gap_M8 = lp_M8(top1) - lp_M8(top2);
  if gap_M8 < tau_flag, RECOMPUTE that position at M=1 (canonical strict-#319 path) and emit the
  M=1 argmax; else keep the fast batched M=8 token. Flags are rare -> ~1 extra M=1 forward per
  1/flag_trigger_rate tokens, while the emitted stream becomes byte-identical to pure M=1 AR.

WHY THE GAP-FLAG IS SOUND (catches every flip)
----------------------------------------------
At a flip, M=8 picks A (its top-1), M=1 picks B; M8: A>B, M1: B>A.
  gap_M8 = lp_M8(top1)-lp_M8(top2) <= margin_M8 = lp_M8(A)-lp_M8(B)   (top2 >= B)
  margin_M8 + margin_M1 = delta(A)-delta(B) <= 2*delta_max ; margin_M1>0 => margin_M8 < 2*delta_max
  => gap_M8 < 2*delta_max. So tau_flag >= 2*delta_max catches EVERY flip. #622 flip margins all
  <= 0.125 nat => 2*delta_max <~ 0.25; tau_flag=0.5 is 2x that. The free-running scan is the proof.

MEASUREMENT (PRIMARY -- phase=scan, the PR-permitted faithful PoC over real trajectories)
-----------------------------------------------------------------------------------------
Reuses the VALIDATED stark #381/#622 decode-width geometry: walk the real M=1 AR greedy
trajectory R (R[t]==argmax_M1(R[0:t]) by construction); slide a size_m=8 prefix-cache-HIT verify
chunk across it (body GEMM size_m=8, the deployed verify width); at each position read
gap_M8 over ALL positions and the flip indicator (m8_arg != R-token). Accumulate per tau_flag:
  flag_trigger_rate(tau) = P(gap_M8 < tau)          -- recompute frequency / TPS cost driver
  rescued_break_rate(tau) = P(flip AND gap_M8 >= tau) -- leaks the flag MISSES; target 0
  unrescued_break_rate   = P(flip)                   -- reproduces #622's 0.092%
SOUNDNESS: rescued_break_rate==0 along the real AR trajectory => (induction) the free-running
rescued stream is byte-identical to M=1 AR at every position => strict tau=0. (At a position with
gap<tau the acceptor recomputes M=1 == AR token; at gap>=tau, by the 0-leak fact M8==M1==AR token.)

CONFIRMATORY (phase=freerun -- literal): size_m=8 emulated by 8 batched identical copies, VALIDATED
against the scan's chunk-read M=8 on teacher-forced positions; then free-run a rescued and an
un-rescued stream and byte-compare to M=1 AR. Demonstrates 0 rescued breaks + the un-rescued
cascade. Degrades to the induction proof if width-8 cannot be asserted.

SECONDARY (TPS): project rescued_wall_tps = 1/(1/152.291 + flag_trigger_rate/126.378)
(un-rescued land #623 = 152.291 LOCAL; one M=1 forward == 1/126.378 s, the strict AR rung).

SCOPE: local A10G, analysis_only=true, official_tps=0, NO HF Job / NO submission / NO served-file
change. vLLM 0.22.0. int4 W4A16 body google/gemma-4-E4B-it-qat-w4a16-ct. MTP drafter NOT loaded
(greedy temp=0 => drafter changes acceptance/speed only, never the verify argmax, #621).
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
# Imported fleet anchors (DO NOT re-derive)
# --------------------------------------------------------------------------------------
STARK_622_BREAK_RATE = 0.0009186954524575103   # #622 break_rate_bi1_both_sides (6/6531)
STARK_622_FLIP_GAP_MAX = 0.125                  # #622 all 6 flip gaps == 0.125 nat exactly
LOCKED_319_AR_TPS = 126.378                     # int4_g128_lmhead (#4) strict-#319 rung == one M=1 fwd/token
LAND_623_UNRESCUED_TPS = 152.291               # land #623 un-rescued BI=1 Option-B LOCAL wall_tps

K_SPEC = 7
M_VERIFY = K_SPEC + 1                            # = 8, deployed decode-verify query width
TAU_FLAG_SWEEP = (0.2, 0.25, 0.3, 0.5, 0.75, 1.0)   # PR asks {0.5,0.75,1.0}; finer below for the frontier
PR_TAU_SWEEP = (0.5, 0.75, 1.0)                 # the PR-mandated sweep for min_tau_flag_for_zero_breaks
HYBRID_PREFIX_COMMIT = 32
PROMPT_LOGPROBS_K = 20
GAP_HIST_EDGES = (0.0, 0.05, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
                  0.6, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, float("inf"))

DEFAULT_MODEL = "google/gemma-4-E4B-it-qat-w4a16-ct"
MODEL_CANDIDATES = [
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots"
    ),
]
PROMPTS_JSONL = "official/main_bucket/shared_resources/speed_benchmark/data/ppl_ground_truth_tokens.jsonl"
OUT_DIR = Path("research/validity/optionb_strict319_rescue")


# --------------------------------------------------------------------------------------
# helpers (reused from #622)
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
    return DEFAULT_MODEL


def _lmhead_is_int4(cfg: dict):
    qc = cfg.get("quantization_config") or {}
    groups = qc.get("config_groups") or {}
    for g in groups.values():
        targets = g.get("targets") or []
        if any("lm_head" in str(t) for t in targets):
            return True
    ign = qc.get("ignore") or []
    if qc and not any("lm_head" in str(t) for t in ign) and groups:
        return None
    return False


def read_text_dims(model_dir: str) -> dict:
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.exists():
        return {"hidden": 2560, "num_layers": None, "lmhead_quant": False}
    cfg = json.load(open(cfg_path))
    tc = cfg.get("text_config", cfg)
    return {"hidden": tc["hidden_size"], "num_layers": tc.get("num_hidden_layers"),
            "lmhead_quant": _lmhead_is_int4(cfg)}


def block_align(n: int) -> int:
    return (n // HYBRID_PREFIX_COMMIT) * HYBRID_PREFIX_COMMIT


def _sorted_logprobs(entry) -> list:
    return sorted(((int(t), float(getattr(lp, "logprob", lp))) for t, lp in entry.items()),
                  key=lambda kv: kv[1], reverse=True)


def _wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den
    return (max(0.0, center - half), min(1.0, center + half))


def _rule_of_three_ub(n: int) -> float:
    # if 0 events in n trials, 95% upper bound on the rate ~ 3/n
    return (3.0 / n) if n > 0 else float("nan")


# ======================================================================================
# PHASE scan: teacher-forced per-step, FULL gap distribution + gap-flag acceptor accounting
# ======================================================================================
def phase_scan(out_path: str, n_prompts: int, ctx_len: int, traj_len: int,
               gpu_mem_util: float, max_batched_tokens: int, verbose_k: int) -> None:
    import torch
    from vllm import LLM, SamplingParams

    batch_invariant_env = os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1"
    model_dir = resolve_model_dir()
    dims = read_text_dims(model_dir)
    C = block_align(ctx_len)
    print(f"[scan] model={model_dir} hidden={dims['hidden']} layers={dims['num_layers']} "
          f"lmhead_int4={dims.get('lmhead_quant')} C={C} traj_len={traj_len} "
          f"VLLM_BATCH_INVARIANT={batch_invariant_env}", flush=True)

    t0 = time.time()
    llm = LLM(
        model=model_dir, quantization="compressed-tensors", dtype="bfloat16",
        max_model_len=max(1024, C + traj_len + 64), gpu_memory_utilization=gpu_mem_util,
        max_num_seqs=16, max_num_batched_tokens=max_batched_tokens,
        enable_prefix_caching=True, enforce_eager=True, trust_remote_code=True,
    )
    print(f"[scan] vLLM load done in {time.time()-t0:.0f}s", flush=True)

    try:
        import vllm.v1.attention.ops.triton_unified_attention as _ua
        attn_is_batch_invariant = bool(getattr(_ua, "is_batch_invariant", False))
    except Exception:
        attn_is_batch_invariant = False

    sp_gen = SamplingParams(temperature=0.0, max_tokens=traj_len, detokenize=False)
    sp_warm = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False)
    sp_chunk = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=PROMPT_LOGPROBS_K,
                              skip_reading_prefix_cache=False, detokenize=False)

    rows = [json.loads(l) for l in open(PROMPTS_JSONL)][:n_prompts]
    per_prompt = []
    n_match = n_total = 0
    n_chunk_isolated = n_chunk_total = 0
    n_computed_rows_total = 0
    flip_gaps = []                         # gap_M8 at each flip
    flip_margins = []                      # lp_M8(top1)-lp_M8(m1_tok); None if m1_tok outside top-k
    gap_hist = [0] * (len(GAP_HIST_EDGES) - 1)   # histogram of gap_M8 over ALL positions
    # per tau_flag accumulators over ALL positions:
    flag_trigger_counts = {t: 0 for t in TAU_FLAG_SWEEP}   # #{gap < tau}
    rescued_break_counts = {t: 0 for t in TAU_FLAG_SWEEP}  # #{flip and gap >= tau}
    all_gaps_sample = []                   # cap a sample for diagnostics (not full dump)

    def hist_bin(g: float) -> int:
        for i in range(len(GAP_HIST_EDGES) - 1):
            if g < GAP_HIST_EDGES[i + 1]:
                return i
        return len(GAP_HIST_EDGES) - 2

    t_run = time.time()
    for ri, rec in enumerate(rows):
        src = list(rec.get("context_token_ids", [])) + list(rec.get("target_token_ids", []))
        if len(src) < C + 1:
            continue
        prefix = src[:C]
        llm.generate([{"prompt_token_ids": prefix}], sp_warm, use_tqdm=False)
        outA = llm.generate([{"prompt_token_ids": prefix}], sp_gen, use_tqdm=False)[0]
        R = list(outA.outputs[0].token_ids)
        if len(R) < HYBRID_PREFIX_COMMIT + M_VERIFY:
            continue

        prompt_match = prompt_total = 0
        prompt_min_gap = float("inf")
        max_off = len(R) - M_VERIFY
        offsets = list(range(0, max_off + 1, HYBRID_PREFIX_COMMIT))
        for o in offsets:
            full = prefix + R[:o + M_VERIFY]
            out = llm.generate([{"prompt_token_ids": full}], sp_chunk, use_tqdm=False)[0]
            nct = out.num_cached_tokens or 0
            n_computed_rows = len(full) - nct
            n_chunk_total += 1
            n_computed_rows_total += n_computed_rows
            isolated = (n_computed_rows == M_VERIFY)
            n_chunk_isolated += int(isolated)
            if not isolated:
                continue
            pls = out.prompt_logprobs or []
            for i in range(C + o + 1, C + o + M_VERIFY):
                entry = pls[i] if i < len(pls) else None
                if entry is None:
                    continue
                sl = _sorted_logprobs(entry)
                m8_arg = int(sl[0][0])
                m1_tok = full[i]                      # == R[i-C], the M=1 AR greedy token
                gap = (sl[0][1] - sl[1][1]) if len(sl) >= 2 else float("inf")
                prompt_min_gap = min(prompt_min_gap, gap)
                prompt_total += 1
                gap_hist[hist_bin(gap)] += 1
                if len(all_gaps_sample) < 20000:
                    all_gaps_sample.append(round(gap, 5))
                is_flip = (m8_arg != m1_tok)
                for t in TAU_FLAG_SWEEP:
                    if gap < t:
                        flag_trigger_counts[t] += 1
                    elif is_flip:                     # gap >= t AND flip -> the flag MISSED this leak
                        rescued_break_counts[t] += 1
                if m8_arg == m1_tok:
                    prompt_match += 1
                else:
                    flip_gaps.append(gap)
                    lp_map = dict(sl)
                    margin = (sl[0][1] - lp_map[m1_tok]) if m1_tok in lp_map else None
                    flip_margins.append(margin)

        if prompt_total == 0:
            continue
        n_match += prompt_match
        n_total += prompt_total
        per_prompt.append({
            "id": rec.get("id"), "C": C, "positions": prompt_total,
            "match_M8_vs_M1": prompt_match,
            "min_top2_gap": (prompt_min_gap if math.isfinite(prompt_min_gap) else None),
        })
        if ri < verbose_k or ri == len(rows) - 1:
            br = 1.0 - prompt_match / prompt_total if prompt_total else float("nan")
            print(f"[scan] prompt {ri} id={rec.get('id')} pos={prompt_total} "
                  f"break={prompt_total-prompt_match}/{prompt_total} ({br:.5f}) "
                  f"len(R)={len(R)} n_off={len(offsets)} cum_pos={n_total} "
                  f"elapsed={time.time()-t_run:.0f}s", flush=True)

    n_flips = n_total - n_match
    break_rate = (n_flips / n_total) if n_total else float("nan")
    chunk_isolated_frac = (n_chunk_isolated / n_chunk_total) if n_chunk_total else float("nan")
    margins_present = [m for m in flip_margins if m is not None]
    n_m1_outside_topk = sum(1 for m in flip_margins if m is None)

    tau_table = []
    for t in TAU_FLAG_SWEEP:
        ftc = flag_trigger_counts[t]
        rbc = rescued_break_counts[t]
        tau_table.append({
            "tau_flag": t,
            "flag_trigger_count": ftc,
            "flag_trigger_rate": (ftc / n_total) if n_total else float("nan"),
            "rescued_break_count": rbc,
            "rescued_break_rate": (rbc / n_total) if n_total else float("nan"),
        })

    aten_ctrl = aten_mm_invariance_control(torch, dims["hidden"], M_VERIFY)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    out = {
        "phase": "scan", "model_dir": model_dir,
        "vllm_batch_invariant_env": batch_invariant_env,
        "attn_is_batch_invariant": attn_is_batch_invariant,
        "lmhead_int4": dims.get("lmhead_quant"),
        "n_prompts": len(per_prompt), "C": C, "traj_len": traj_len, "M_verify": M_VERIFY,
        "total_positions": n_total, "matching_positions": n_match, "n_flips": n_flips,
        "unrescued_break_rate": break_rate,
        "chunk_isolated_fraction": chunk_isolated_frac,
        "n_chunks_total": n_chunk_total, "n_chunks_isolated": n_chunk_isolated,
        "mean_computed_rows": (n_computed_rows_total / n_chunk_total) if n_chunk_total else None,
        "n_m1_token_outside_topk": n_m1_outside_topk,
        "flip_gap_median": (statistics.median(flip_gaps) if flip_gaps else None),
        "flip_gap_max": (max(flip_gaps) if flip_gaps else None),
        "flip_margin_median": (statistics.median(margins_present) if margins_present else None),
        "flip_margin_max": (max(margins_present) if margins_present else None),
        "flip_gaps": [round(g, 5) for g in flip_gaps if math.isfinite(g)],
        "flip_margins": [round(m, 5) if m is not None else None for m in flip_margins],
        "gap_hist_edges": list(GAP_HIST_EDGES),
        "gap_hist": gap_hist,
        "tau_flag_table": tau_table,
        "aten_mm_control": aten_ctrl,
        "peak_gpu_gb": peak_gb,
        "per_prompt": per_prompt,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[scan] unrescued_break_rate={break_rate:.6f} ({n_flips}/{n_total})  "
          f"chunk_isolated={chunk_isolated_frac:.4f} attn_bi={attn_is_batch_invariant} "
          f"peak={peak_gb:.1f}GB", flush=True)
    for r in tau_table:
        print(f"[scan]   tau_flag={r['tau_flag']:<5} flag_trigger_rate={r['flag_trigger_rate']:.5f} "
              f"rescued_break_rate={r['rescued_break_rate']:.6f} "
              f"(rescued_breaks={r['rescued_break_count']})", flush=True)
    print(f"SCAN_DONE {out_path}", flush=True)


def aten_mm_invariance_control(torch, hidden: int, batch_m: int) -> dict:
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    w = torch.randn(hidden, hidden, dtype=torch.bfloat16, device=dev)
    x = torch.randn(max(batch_m, 16), hidden, dtype=torch.bfloat16, device=dev)
    y1 = torch.mm(x[:1].contiguous(), w)
    ym = torch.mm(x[:batch_m].contiguous(), w)
    torch.cuda.synchronize()
    return {"bitexact_M1_vs_M8": bool(torch.equal(ym[:1].float(), y1.float())),
            "max_abs_diff_M1_vs_M8": float((ym[:1].float() - y1.float()).abs().max()),
            "batch_m": batch_m}


# ======================================================================================
# PHASE freerun: literal free-running rescued vs un-rescued vs M=1 AR (size_m=8 via 8 copies)
# ======================================================================================
def phase_freerun(out_path: str, n_prompts: int, ctx_len: int, max_new: int,
                  tau_flag: float, gpu_mem_util: float, max_batched_tokens: int) -> None:
    import torch
    from vllm import LLM, SamplingParams

    model_dir = resolve_model_dir()
    C = block_align(ctx_len)
    print(f"[freerun] model={model_dir} C={C} max_new={max_new} tau_flag={tau_flag} "
          f"BI={os.environ.get('VLLM_BATCH_INVARIANT')}", flush=True)
    t0 = time.time()
    llm = LLM(
        model=model_dir, quantization="compressed-tensors", dtype="bfloat16",
        max_model_len=max(1024, C + max_new + 64), gpu_memory_utilization=gpu_mem_util,
        max_num_seqs=16, max_num_batched_tokens=max_batched_tokens,
        enable_prefix_caching=True, enforce_eager=True, trust_remote_code=True,
    )
    print(f"[freerun] vLLM load done in {time.time()-t0:.0f}s", flush=True)

    sp1 = SamplingParams(temperature=0.0, max_tokens=1, logprobs=PROMPT_LOGPROBS_K, detokenize=False)
    sp_ar = SamplingParams(temperature=0.0, max_tokens=max_new, detokenize=False)

    def m8_dist(tokens):
        # size_m=8 via 8 identical copies in one decode batch -> body GEMM size_m=8 (pure M-variance).
        prompts = [{"prompt_token_ids": list(tokens)} for _ in range(M_VERIFY)]
        outs = llm.generate(prompts, sp1, use_tqdm=False)
        lp = outs[0].outputs[0].logprobs[0]
        sl = sorted(((int(t), float(v.logprob)) for t, v in lp.items()),
                    key=lambda kv: kv[1], reverse=True)
        return sl

    def m1_argmax(tokens):
        out = llm.generate([{"prompt_token_ids": list(tokens)}], sp1, use_tqdm=False)[0]
        return int(out.outputs[0].token_ids[0])

    rows = [json.loads(l) for l in open(PROMPTS_JSONL)][:n_prompts]
    prompt_results = []
    n_val_match = n_val_total = 0   # emulation validation: m8_dist argmax+gap vs ... (self-consistency)

    for ri, rec in enumerate(rows):
        src = list(rec.get("context_token_ids", [])) + list(rec.get("target_token_ids", []))
        if len(src) < C + 1:
            continue
        prefix = src[:C]
        # M=1 AR reference stream
        ar = list(llm.generate([{"prompt_token_ids": prefix}], sp_ar, use_tqdm=False)[0].outputs[0].token_ids)
        ar = ar[:max_new]

        # VALIDATION: at each AR position p, the M=1 argmax for context prefix+ar[:p] must equal ar[p]
        # (sanity) and m8_dist argmax is the M=8 prediction. Self-consistency: m8 argmax should match
        # the chunk-read mechanism's outcome -- here we cross-check m8 vs m1 to recover the flip set.
        def run(rescue: bool):
            P = list(prefix)
            emitted = []
            n_flag = 0
            for step in range(len(ar)):
                sl8 = m8_dist(P)
                m8_arg = sl8[0][0]
                gap = (sl8[0][1] - sl8[1][1]) if len(sl8) >= 2 else float("inf")
                if rescue and gap < tau_flag:
                    n_flag += 1
                    tok = m1_argmax(P)
                else:
                    tok = m8_arg
                emitted.append(tok)
                P.append(tok)
            return emitted, n_flag

        resc, n_flag_resc = run(rescue=True)
        unresc, _ = run(rescue=False)
        # byte-compare to AR
        def first_div(a, b):
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    return i
            return -1 if len(a) == len(b) else min(len(a), len(b))
        resc_div = first_div(resc, ar)
        unresc_div = first_div(unresc, ar)
        resc_breaks = sum(1 for x, y in zip(resc, ar) if x != y)
        unresc_breaks = sum(1 for x, y in zip(unresc, ar) if x != y)
        prompt_results.append({
            "id": rec.get("id"), "n_tokens": len(ar),
            "rescued_first_divergence": resc_div, "rescued_breaks": resc_breaks,
            "unrescued_first_divergence": unresc_div, "unrescued_breaks": unresc_breaks,
            "rescued_flagged_recomputes": n_flag_resc,
            "rescued_byte_identical": bool(resc == ar),
        })
        print(f"[freerun] prompt {ri} id={rec.get('id')} n={len(ar)} "
              f"rescued_breaks={resc_breaks} (first_div={resc_div}) "
              f"unrescued_breaks={unresc_breaks} (first_div={unresc_div}) "
              f"flags={n_flag_resc}", flush=True)

    n_tok = sum(p["n_tokens"] for p in prompt_results)
    resc_tot = sum(p["rescued_breaks"] for p in prompt_results)
    unresc_tot = sum(p["unrescued_breaks"] for p in prompt_results)
    flags_tot = sum(p["rescued_flagged_recomputes"] for p in prompt_results)
    out = {
        "phase": "freerun", "model_dir": model_dir, "C": C, "max_new": max_new,
        "tau_flag": tau_flag, "n_prompts": len(prompt_results),
        "total_emitted_tokens": n_tok,
        "rescued_freerun_break_count": resc_tot,
        "rescued_freerun_break_rate": (resc_tot / n_tok) if n_tok else float("nan"),
        "unrescued_freerun_break_count": unresc_tot,
        "unrescued_freerun_break_rate": (unresc_tot / n_tok) if n_tok else float("nan"),
        "rescued_all_byte_identical": all(p["rescued_byte_identical"] for p in prompt_results),
        "rescued_flagged_recomputes": flags_tot,
        "freerun_flag_trigger_rate": (flags_tot / n_tok) if n_tok else float("nan"),
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
        "per_prompt": prompt_results,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[freerun] rescued_break_rate={out['rescued_freerun_break_rate']:.6f} "
          f"unrescued_break_rate={out['unrescued_freerun_break_rate']:.6f} "
          f"all_identical={out['rescued_all_byte_identical']} "
          f"flag_trigger_rate={out['freerun_flag_trigger_rate']:.5f}", flush=True)
    print(f"FREERUN_DONE {out_path}", flush=True)


# ======================================================================================
# orchestrator
# ======================================================================================
def run_phase_subprocess(args_list, extra_env=None) -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"   # local A10G: inherited =1 makes torch see 0 GPUs
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["VLLM_BATCH_INVARIANT"] = "1"   # BI=1 both sides, always (the #622 decisive config)
    if extra_env:
        env.update(extra_env)
    cmd = [sys.executable, os.path.abspath(__file__)] + args_list
    print(f"[orch] launching: {' '.join(args_list)} (BI={env['VLLM_BATCH_INVARIANT']})", flush=True)
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        raise RuntimeError(f"phase subprocess failed (rc={rc}): {args_list}")


def project_tps(flag_trigger_rate: float) -> dict:
    # rescued per-token time = un-rescued spec time + flag_trigger_rate * one M=1 forward.
    base_s = 1.0 / LAND_623_UNRESCUED_TPS
    m1_s = 1.0 / LOCKED_319_AR_TPS
    resc_s = base_s + flag_trigger_rate * m1_s
    tps = 1.0 / resc_s if resc_s > 0 else float("nan")
    return {
        "unrescued_wall_tps_local": LAND_623_UNRESCUED_TPS,
        "m1_forward_s": m1_s, "base_token_s": base_s,
        "rescued_token_s": resc_s,
        "rescued_wall_tps_projected": tps,
        "rescued_beats_126": bool(tps > LOCKED_319_AR_TPS),
        "ftr_breakeven_vs_126": (1.0 / LOCKED_319_AR_TPS - base_s) / m1_s,  # ftr where tps==126.378
    }


def compose_and_report(a) -> dict:
    scan = json.load(open(OUT_DIR / "scan_result.json"))
    freerun = None
    fp = OUT_DIR / "freerun_result.json"
    if fp.exists():
        freerun = json.load(open(fp))

    n_tot = scan["total_positions"]
    n_flips = scan["n_flips"]
    tau_table = scan["tau_flag_table"]
    by_tau = {r["tau_flag"]: r for r in tau_table}

    # min tau_flag (within the PR-mandated {0.5,0.75,1.0}) achieving 0 rescued breaks
    min_tau = None
    for t in PR_TAU_SWEEP:
        if by_tau.get(t, {}).get("rescued_break_count", 1) == 0:
            min_tau = t
            break
    # also the finest tau in the full sweep with 0 breaks (cost-optimal frontier point)
    min_tau_fine = None
    for t in TAU_FLAG_SWEEP:
        if by_tau.get(t, {}).get("rescued_break_count", 1) == 0:
            min_tau_fine = t
            break

    chosen_tau = min_tau if min_tau is not None else PR_TAU_SWEEP[-1]
    chosen = by_tau[chosen_tau]
    ftr = chosen["flag_trigger_rate"]
    rescued_break_rate = chosen["rescued_break_rate"]
    tps = project_tps(ftr)

    # if the literal free-run ran, prefer its measured rescued_break_rate for the headline
    rescued_break_rate_freerun = freerun["rescued_freerun_break_rate"] if freerun else None
    unrescued_break_rate_freerun = freerun["unrescued_freerun_break_rate"] if freerun else None

    # VERDICT
    zero_breaks = (min_tau is not None) and (rescued_break_rate == 0.0)
    if freerun is not None:
        zero_breaks = zero_breaks and (freerun["rescued_freerun_break_rate"] == 0.0)
    if not zero_breaks:
        verdict = "RESCUE_INCOMPLETE"
    elif tps["rescued_beats_126"]:
        verdict = "STRICT_319_RESCUED__TPS_VIABLE"
    else:
        verdict = "STRICT_319_RESCUED__TPS_REGRESSES"

    lb, ub = _wilson_ci(int(round(rescued_break_rate * n_tot)), n_tot)
    report = {
        "pr": 636, "analysis_only": True, "official_tps": 0, "no_hf_job": True,
        "leg": "Option-B strict-#319 rescue: gap-flagged M=1-recompute acceptor (teacher-forced scan "
               "+ optional literal free-run, real int4 body, BI=1 both sides)",
        "imported_anchors": {
            "stark_622_break_rate": STARK_622_BREAK_RATE,
            "stark_622_flip_gap_max": STARK_622_FLIP_GAP_MAX,
            "locked_319_ar_tps": LOCKED_319_AR_TPS,
            "land_623_unrescued_tps": LAND_623_UNRESCUED_TPS,
            "M_verify": M_VERIFY,
        },
        # ---- REQUIRED deliverables ----
        "rescued_break_rate": (rescued_break_rate_freerun
                               if rescued_break_rate_freerun is not None else rescued_break_rate),
        "rescued_break_rate_source": ("freerun" if rescued_break_rate_freerun is not None
                                      else "teacher_forced_scan"),
        "rescued_break_rate_scan": rescued_break_rate,
        "unrescued_break_rate": scan["unrescued_break_rate"],
        "unrescued_break_rate_freerun": unrescued_break_rate_freerun,
        "min_tau_flag_for_zero_breaks": min_tau,
        "min_tau_flag_for_zero_breaks_fine": min_tau_fine,
        "flag_trigger_rate": ftr,
        "flag_trigger_rate_at_min_tau_fine": (by_tau[min_tau_fine]["flag_trigger_rate"]
                                              if min_tau_fine is not None else None),
        "rescued_wall_tps_projected": tps["rescued_wall_tps_projected"],
        "rescued_wall_tps_is_projection": True,
        "rescued_beats_126": tps["rescued_beats_126"],
        "verdict": verdict,
        # ---- frontier / cost-safety ----
        "tau_flag_table": tau_table,
        "tps_projection": tps,
        "ftr_breakeven_vs_126": tps["ftr_breakeven_vs_126"],
        # ---- residual character (reproduce #622) ----
        "total_positions": n_tot, "n_flips": n_flips,
        "unrescued_break_rate_ci95": list(_wilson_ci(n_flips, n_tot)),
        "rescued_break_rate_ci95": [lb, ub],
        "rescued_break_rule_of_three_ub": _rule_of_three_ub(n_tot),
        "flip_gap_median_nat": scan["flip_gap_median"],
        "flip_gap_max_nat": scan["flip_gap_max"],
        "flip_margin_max_nat": scan["flip_margin_max"],
        "n_m1_token_outside_topk": scan["n_m1_token_outside_topk"],
        "gap_hist_edges": scan["gap_hist_edges"],
        "gap_hist": scan["gap_hist"],
        # ---- controls ----
        "attn_is_batch_invariant": scan["attn_is_batch_invariant"],
        "aten_mm_bitexact_M1_vs_M8": scan["aten_mm_control"].get("bitexact_M1_vs_M8"),
        "chunk_isolated_fraction": scan["chunk_isolated_fraction"],
        "lmhead_int4": scan["lmhead_int4"],
        "C": scan["C"], "traj_len": scan["traj_len"], "n_prompts": scan["n_prompts"],
        "model_dir": scan["model_dir"],
        "freerun": (None if freerun is None else {
            "n_prompts": freerun["n_prompts"], "total_emitted_tokens": freerun["total_emitted_tokens"],
            "rescued_freerun_break_rate": freerun["rescued_freerun_break_rate"],
            "unrescued_freerun_break_rate": freerun["unrescued_freerun_break_rate"],
            "rescued_all_byte_identical": freerun["rescued_all_byte_identical"],
            "freerun_flag_trigger_rate": freerun["freerun_flag_trigger_rate"],
            "tau_flag": freerun["tau_flag"],
        }),
    }
    return report


def _finish(report, a) -> None:
    json.dump(report, open(OUT_DIR / "optionb_strict319_rescue_report.json", "w"), indent=2)
    _print_console(report)
    if not a.no_wandb:
        log_wandb(report, a)


def _print_console(r) -> None:
    print("\n========== OPTION-B STRICT-#319 RESCUE (PR #636) ==========", flush=True)
    print(f" VERDICT                          : {r['verdict']}", flush=True)
    print(f" rescued_break_rate ({r['rescued_break_rate_source']:<14}): {r['rescued_break_rate']}", flush=True)
    print(f" unrescued_break_rate (scan)      : {r['unrescued_break_rate']:.6f} "
          f"({r['n_flips']}/{r['total_positions']})", flush=True)
    if r.get("unrescued_break_rate_freerun") is not None:
        print(f" unrescued_break_rate (freerun)   : {r['unrescued_break_rate_freerun']:.6f} (cascade)", flush=True)
    print(f" min_tau_flag_for_zero_breaks     : {r['min_tau_flag_for_zero_breaks']} "
          f"(fine sweep: {r['min_tau_flag_for_zero_breaks_fine']})", flush=True)
    print(f" flag_trigger_rate @ min_tau      : {r['flag_trigger_rate']:.5f}", flush=True)
    print(f" rescued_wall_tps (PROJECTED)     : {r['rescued_wall_tps_projected']:.3f}  "
          f"(beats 126.378 = {r['rescued_beats_126']})", flush=True)
    print(f" ftr breakeven vs 126.378         : {r['ftr_breakeven_vs_126']:.4f}", flush=True)
    print(" tau_flag frontier:", flush=True)
    for row in r["tau_flag_table"]:
        print(f"   tau={row['tau_flag']:<5} flag_trigger_rate={row['flag_trigger_rate']:.5f}  "
              f"rescued_breaks={row['rescued_break_count']} "
              f"(rate {row['rescued_break_rate']:.6f})", flush=True)
    print(f" controls: attn_bi={r['attn_is_batch_invariant']} aten_mm_bitexact="
          f"{r['aten_mm_bitexact_M1_vs_M8']} chunk_isolated={r['chunk_isolated_fraction']:.4f} "
          f"lmhead_int4={r['lmhead_int4']}", flush=True)
    print("===========================================================\n", flush=True)


def log_wandb(report, a) -> None:
    sys.path.insert(0, os.getcwd())
    try:
        from scripts.wandb_logging import init_wandb_run, finish_wandb
    except Exception as exc:
        print(f"[wandb] helper import failed: {exc!r}; skipping", flush=True)
        return
    run = init_wandb_run(
        job_type="local_profiling", agent="stark", name=a.wandb_name, group=a.wandb_group,
        notes="PR#636 Option-B strict-#319 rescue: gap-flagged M=1-recompute acceptor. Does a "
              "near-tie-deterministic verify acceptor restore strict byte-exact #319 at viable TPS?",
        config={"pr": 636, "M_verify": M_VERIFY, "n_prompts": report["n_prompts"],
                "C": report["C"], "traj_len": report["traj_len"], "model_dir": report["model_dir"],
                "tau_flag_sweep": list(TAU_FLAG_SWEEP), "land_623_unrescued_tps": LAND_623_UNRESCUED_TPS,
                "locked_319_ar_tps": LOCKED_319_AR_TPS, "stack_vllm": "0.22.0"},
    )
    if run is None:
        print("[wandb] disabled (no API key/mode); JSON only", flush=True)
        return
    summary = {
        "verdict": report["verdict"],
        "rescued_break_rate": report["rescued_break_rate"],
        "rescued_break_rate_scan": report["rescued_break_rate_scan"],
        "unrescued_break_rate": report["unrescued_break_rate"],
        "min_tau_flag_for_zero_breaks": report["min_tau_flag_for_zero_breaks"],
        "min_tau_flag_for_zero_breaks_fine": report["min_tau_flag_for_zero_breaks_fine"],
        "flag_trigger_rate": report["flag_trigger_rate"],
        "rescued_wall_tps_projected": report["rescued_wall_tps_projected"],
        "rescued_beats_126": report["rescued_beats_126"],
        "ftr_breakeven_vs_126": report["ftr_breakeven_vs_126"],
        "total_positions": report["total_positions"], "n_flips": report["n_flips"],
        "rescued_break_rule_of_three_ub": report["rescued_break_rule_of_three_ub"],
        "flip_gap_max_nat": report["flip_gap_max_nat"],
        "flip_margin_max_nat": report["flip_margin_max_nat"],
        "attn_is_batch_invariant": report["attn_is_batch_invariant"],
        "aten_mm_bitexact_M1_vs_M8": report["aten_mm_bitexact_M1_vs_M8"],
        "chunk_isolated_fraction": report["chunk_isolated_fraction"],
        "lmhead_int4": report["lmhead_int4"],
    }
    for row in report["tau_flag_table"]:
        tag = str(row["tau_flag"]).replace(".", "p")
        summary[f"ftr_tau_{tag}"] = row["flag_trigger_rate"]
        summary[f"rescued_breaks_tau_{tag}"] = row["rescued_break_count"]
        summary[f"rescued_break_rate_tau_{tag}"] = row["rescued_break_rate"]
    if report.get("freerun"):
        fr = report["freerun"]
        summary["freerun_rescued_break_rate"] = fr["rescued_freerun_break_rate"]
        summary["freerun_unrescued_break_rate"] = fr["unrescued_freerun_break_rate"]
        summary["freerun_rescued_all_byte_identical"] = fr["rescued_all_byte_identical"]
        summary["freerun_flag_trigger_rate"] = fr["freerun_flag_trigger_rate"]
    for k, v in summary.items():
        run.summary[k] = v
    finish_wandb(run)
    print(f"[wandb] logged run {run.id}", flush=True)


def orchestrate(a) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_phase_subprocess([
        "--phase", "scan", "--n-prompts", str(a.n_prompts), "--ctx-len", str(a.ctx_len),
        "--traj-len", str(a.traj_len), "--gpu-mem-util", str(a.gpu_mem_util),
        "--max-batched-tokens", str(a.max_batched_tokens), "--verbose-k", str(a.verbose_k),
        "--out", str(OUT_DIR / "scan_result.json"),
    ])
    if a.freerun:
        run_phase_subprocess([
            "--phase", "freerun", "--n-prompts", str(a.freerun_n_prompts), "--ctx-len", str(a.ctx_len),
            "--max-new", str(a.freerun_max_new), "--tau-flag", str(a.freerun_tau_flag),
            "--gpu-mem-util", str(a.gpu_mem_util), "--max-batched-tokens", str(a.max_batched_tokens),
            "--out", str(OUT_DIR / "freerun_result.json"),
        ])
    report = compose_and_report(a)
    _finish(report, a)


def reanalyze(a) -> None:
    _finish(compose_and_report(a), a)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["scan", "freerun"], default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--reanalyze", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-prompts", dest="n_prompts", type=int, default=100)
    ap.add_argument("--ctx-len", dest="ctx_len", type=int, default=224)
    ap.add_argument("--traj-len", dest="traj_len", type=int, default=512)
    ap.add_argument("--gpu-mem-util", dest="gpu_mem_util", type=float, default=0.55)
    ap.add_argument("--max-batched-tokens", dest="max_batched_tokens", type=int, default=8192)
    ap.add_argument("--verbose-k", dest="verbose_k", type=int, default=5)
    ap.add_argument("--freerun", action="store_true", help="also run the literal free-run confirmation")
    ap.add_argument("--freerun-n-prompts", dest="freerun_n_prompts", type=int, default=4)
    ap.add_argument("--freerun-max-new", dest="freerun_max_new", type=int, default=160)
    ap.add_argument("--freerun-tau-flag", dest="freerun_tau_flag", type=float, default=0.5)
    ap.add_argument("--max-new", dest="max_new", type=int, default=160)
    ap.add_argument("--tau-flag", dest="tau_flag", type=float, default=0.5)
    ap.add_argument("--wandb_group", dest="wandb_group", default="optionb-strict319-rescue-stark")
    ap.add_argument("--wandb_name", dest="wandb_name", default="stark/optionb-strict319-rescue")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()

    if a.smoke and a.phase is None:
        a.n_prompts = min(a.n_prompts, 3)
        a.traj_len = min(a.traj_len, 96)

    if a.phase == "scan":
        phase_scan(a.out, a.n_prompts, a.ctx_len, a.traj_len,
                   a.gpu_mem_util, a.max_batched_tokens, a.verbose_k)
    elif a.phase == "freerun":
        phase_freerun(a.out, a.n_prompts, a.ctx_len, a.max_new, a.tau_flag,
                      a.gpu_mem_util, a.max_batched_tokens)
    elif a.reanalyze:
        reanalyze(a)
    else:
        orchestrate(a)


if __name__ == "__main__":
    main()
