#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #364 -- Margin-localized identity repair: a sub-9.841% eta to restore strict?

#319 strict locks byte-exact greedy-token-identity. The M=8 batched verify breaks identity
by a DETERMINISTIC 0.46% (ubel #361, bxmrvzwf: 19/4096 flip, both determinism controls = 1.0).
wirbel #360 (6s9vgnw9) prices the BLANKET cost of restoring identity (a batch-invariant
bf16 lm_head+attn kernel on EVERY position) at eta = 9.841% of TPS, which exceeds the entire
lambda=1 headroom -> blanket caps strict at 520.953*(1-0.09841) = 469.68 < 500.

THIS CARD attacks the surviving lever (b): a CHEAPER-than-9.841% identity mechanism =
**margin-localized selective repair**. The premise (from #361): flips are LOW-MARGIN events
(first divergence flipped on a top1-top2 gap of just 0.4375 nats). A position whose M=8
logit margin is large CANNOT flip: the M=8-vs-M=1 numerical perturbation is bounded, so only
positions with margin below that perturbation are at risk. Mechanism: detect the small
fraction of low-margin positions, recompute ONLY those in bit-exact M=1 (AR) geometry
(identity=1.0, lawine #196), leave the rest on the fast M=8 path.

PROVABLE robustness bound (derived in-card): let delta_spread = max over shared top-K tokens of
(lp_M8(t) - lp_M1(t)) - min(...) at a position. If A is the M=1 argmax (=ref) and B!=A wins in
M=8, then margin_M8(B over 2nd) <= lp_M8(B)-lp_M8(A) = (lp_M1(B)-lp_M1(A)) + (dB-dA) < 0 +
delta_spread. So EVERY flip has margin_M8 < delta_spread_max. Choosing tau >= delta_spread_max
(measured worst-case) GUARANTEES every flip is flagged -> selective repair -> identity = 1.0.
The cost is then frac_positions_below_tau * recompute_cost.

WHAT THIS RUNS (LOCAL pod-GPU profiling ONLY. NO train.py --launch, NO HF Job, NO submission,
NO served-file change):

  Phase margin_scan : on the int4 DEPLOYED substrate, the IDENTICAL #361 width-8 verify
    geometry (enforce_eager, prefix-caching off). For every generated position record the
    M=8 verify top1-top2 margin, the M=1 AR top1-top2 margin, whether it flips (M=8 argmax vs
    M=1 AR ref token), and the centered M8-vs-M1 logprob perturbation spread over shared
    top-K. Re-run the verify forward (det_ver) + re-gen ref on a subset (det_gen) for the
    >=2-seed determinism control.

  Phase recompute_timing : the per-position M=1 recompute cost. Times (a) the FULL bf16
    lm_head GEMV [1,2560]x[2560,262144] (the strict recompute must use the full vocab; a
    pruned lmhead12k cannot guarantee the argmax), (b) the int4-body single-token GEMM stack
    (a compute-bound decode-step proxy), (c) a single-query SDPA, giving f_lmhead =
    lmhead_us / (body+lmhead)_us, the lm_head's compute fraction of a decode step.

  Orchestrator (CPU) : margin-flip concentration + flip_margin_auc, tau_robust = top-T spread,
    frac_below_tau, the tau-sweep identity<->eta frontier, selective_eta under two cost models
    (conservative = full M=1 decode recompute; optimistic = lm_head-only recompute, the PR's
    named cost), the decision bools, and the self-test.

PRIMARY metric : token_identity_rate (raw M=8 verify-geometry, corroborates #361 ~0.99536).
TEST   metric  : margin_identity_self_test_passes (bool).
Headline       : selective_eta_at_identity_1, selective_beats_blanket, selective_clears_500_budget.

All TPS/us fields are LOCAL RELATIVE (~7x off official, #245); ratios are portable. BASELINE
481.53 UNCHANGED -- this is a measurement, 0 official TPS.

Reproduce:
    cd target/ && python research/validity/margin_localized_identity/margin_localized_identity.py \
        --gpu --wandb_group margin-localized-identity --wandb_name ubel/margin-localized-identity-eta
"""
from __future__ import annotations

import argparse
import hashlib
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
REPO_ROOT = HERE.parents[2]  # .../target
OUT_DIR = HERE

# --------------------------------------------------------------------------------------
# Imported fleet anchors (import, do NOT re-derive).
# --------------------------------------------------------------------------------------
OFFICIAL_BASELINE = 481.53            # #52 official frontier TPS (this card adds 0)
TARGET_TPS = 500.0
LAMBDA1_CEIL = 520.953               # #326/#327 lambda=1 spec-model ceiling
K_SPEC = 7
M_VERIFY = K_SPEC + 1                 # = 8, batched-verify width
BLANKET_ETA = 0.09841249119201488    # wirbel #360 (6s9vgnw9) correct-locus blanket eta
BLANKET_OVERHEAD_US = 119.88609677011253   # wirbel step_overhead_us at the 9.841% floor
STEP_US_WIRBEL = 1218.2              # wirbel #360 step basis (deployed-scale)
BUDGET_ETA = 1.0 - TARGET_TPS / LAMBDA1_CEIL   # = 0.040211...  (>500 kernel budget)
# wirbel #362 (5k3px8p1): a lm_head-ONLY M=1 re-verify does NOT restore identity (the M=8
# divergence is injected by bf16 attention upstream and propagates through all layers into the
# final hidden state -> token_identity_rate_lmhead_only=0.9948<1.0). So the correct selective
# recompute is a FULL M=1 forward per flagged position. wirbel #362 also priced the counterfactual:
# a cheap+correct detector flagging only the ~0.49% truly-divergent steps -> eta probe_gated=0.97%.
# The margin is exactly that candidate FREE detector (computed from the M=8 logits we already have).
PROBE_GATED_ETA_362 = 0.0097         # wirbel #362 probe-gated floor (perfect cheap detector)
LMHEAD_ONLY_IDENTITY_362 = 0.9948    # wirbel #362: lm_head-only recompute leaves ~0.52% wrong
RAW_IDENTITY_361 = 0.995361328125    # ubel #361 (bxmrvzwf) M=8 verify identity
RAW_DIVERGENCE_361 = 0.004638671875
DEPLOYED_M8_DIVERGENCE_232 = 0.007291666666666696
RECONCILE_TOL = 0.01
# A flip can only promote a token already near the M1 top; deep-tail (rank>>1) perturbations
# cannot change the argmax, so the flip-relevant perturbation is the spread of (lp_M8 - lp_M1)
# over the top-T M1 candidates, NOT over the whole top-K (which is tail-contaminated). T=2 is the
# tightest provable bound (most argmax flips are 2-way swaps: argmax <-> runner-up).
SPREAD_TOPTS = (2, 4, 8)

MODEL_CANDIDATES = [
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots"
    ),
    "/tmp/osoi5-v0-baked",
]
PROMPTS_JSONL = ("official/main_bucket/shared_resources/speed_benchmark/data/"
                 "ppl_ground_truth_tokens.jsonl")


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


def load_prompts(n_prompts: int, ctx_cap: int) -> list[dict]:
    rows = [json.loads(l) for l in open(PROMPTS_JSONL)][:n_prompts]
    out = []
    for rec in rows:
        ctx = list(rec.get("context_token_ids", []))[:ctx_cap]
        if len(ctx) >= 2:
            out.append({"id": rec.get("id"), "context_token_ids": ctx})
    return out


def _lp(v: Any) -> float:
    return float(getattr(v, "logprob", v))


def entry_as_dict(entry: Any) -> dict[int, float]:
    """A vLLM logprobs entry (dict token_id->Logprob) -> {int token_id: float logprob}."""
    out: dict[int, float] = {}
    if not entry:
        return out
    for tok, v in entry.items():
        out[int(tok)] = _lp(v)
    return out


def top1_top2_margin(d: dict[int, float]) -> tuple[int, float, float]:
    """argmax token, top1 logprob, (top1 - top2) margin (nats). Needs >=2 entries."""
    if not d:
        return -1, float("nan"), float("nan")
    s = sorted(d.values(), reverse=True)
    top = max(d.items(), key=lambda kv: kv[1])
    margin = (s[0] - s[1]) if len(s) >= 2 else float("inf")
    return int(top[0]), float(top[1]), float(margin)


# ======================================================================================
# GPU PHASE 1: margin scan -- REF (greedy AR, logprobs=K) + M=8 verify (prompt_logprobs=K)
# ======================================================================================
def phase_margin_scan(out_path: str, n_prompts: int, n_new: int, ctx_cap: int,
                      verify_width: int, gpu_mem_util: float, topk: int,
                      det_prompts: int) -> None:
    import torch
    from vllm import LLM, SamplingParams

    model_dir = resolve_model_dir()
    prompts = load_prompts(n_prompts, ctx_cap)
    print(f"[scan] model={model_dir} prompts={len(prompts)} n_new={n_new} "
          f"verify_width={verify_width} topk={topk}", flush=True)

    # IDENTICAL #361 verify geometry: one engine at the verify width; decode steps are width-1
    # (canonical AR), re-fed prompts processed in width-M chunked forwards (literal spec-verify
    # geometry). enforce_eager so no CUDA-graph padding perturbs M; prefix caching off so
    # re-feeds do a real forward. max_logprobs lifted so top-K prompt/gen logprobs are allowed.
    effective_width = verify_width
    llm = None
    for w in (verify_width, 16, 32, 64):
        try:
            llm = LLM(
                model=model_dir,
                quantization="compressed-tensors",
                dtype="bfloat16",
                max_model_len=max(1024, ctx_cap + n_new + 16),
                gpu_memory_utilization=gpu_mem_util,
                max_num_seqs=1,
                max_num_batched_tokens=w,
                enable_prefix_caching=False,
                enforce_eager=True,
                trust_remote_code=True,
                max_logprobs=max(20, topk + 2),
            )
            effective_width = w
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[scan] engine init at max_num_batched_tokens={w} failed: {exc!r}", flush=True)
            llm = None
    if llm is None:
        raise RuntimeError("could not construct the int4 verify engine at any width")
    if effective_width != verify_width:
        print(f"[scan] NOTE effective verify width = {effective_width} "
              f"(requested {verify_width})", flush=True)

    gen_sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=n_new, logprobs=topk)
    ver_sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=topk)

    positions: list[dict] = []     # one record per generated position
    per_prompt: list[dict] = []
    n_match = n_total = 0
    n_det_gen = n_det_ver = 0
    gen_tokens_total = 0
    det_gen_positions = 0
    first_div = None
    spread_max = 0.0               # full-K spread (tail-contaminated; reference only)
    spread_max_where: dict | None = None
    spread_t_max = {t: 0.0 for t in SPREAD_TOPTS}   # flip-relevant top-T spreads
    max_flip_target_rank = -1     # max M1-rank of any flip target (bound safety vs top-T)
    m1_argmax_eq_ref = 0            # self-consistency: M1 top1 == greedily-emitted token

    t0 = time.time()
    ref_decode_time = 0.0
    for pi, pr in enumerate(prompts):
        ctx = pr["context_token_ids"]
        c = len(ctx)
        base = {"prompt_token_ids": ctx}

        # --- REF: plain greedy AR, width-1 decode, capture top-K M=1 logprobs per step ---
        _tg = time.time()
        out = llm.generate([base], gen_sp, use_tqdm=False)[0]
        ref_decode_time += time.time() - _tg
        gen = list(out.outputs[0].token_ids)
        gen_lps = out.outputs[0].logprobs or []   # len == len(gen); dict per step (M=1 geometry)
        gen_tokens_total += len(gen)

        # determinism control (subset): re-gen ref, expect identical (int4 bit-exact)
        if pi < det_prompts:
            gen_b = list(llm.generate([base], gen_sp, use_tqdm=False)[0].outputs[0].token_ids)
            Lg = min(len(gen), len(gen_b))
            n_det_gen += sum(1 for a, b in zip(gen[:Lg], gen_b[:Lg]) if a == b)
            det_gen_positions += Lg

        # --- SPEC-equiv: M=8 verify-geometry re-forward of [ctx+gen], top-K prompt logprobs ---
        full = ctx + gen
        vout = llm.generate([{"prompt_token_ids": full}], ver_sp, use_tqdm=False)[0]
        pls = vout.prompt_logprobs
        vout_b = llm.generate([{"prompt_token_ids": full}], ver_sp, use_tqdm=False)[0]
        pls_b = vout_b.prompt_logprobs

        match = det_ver = 0
        verify_stream: list[int] = []
        for g in range(len(gen)):
            j = c + g
            if j >= len(pls) or pls[j] is None:
                continue
            ref_tok = int(gen[g])
            m8 = entry_as_dict(pls[j])
            if ref_tok not in m8:   # ensure ref present so margins/flip are well-defined
                m8[ref_tok] = float("-inf")
            m8_arg, m8_top_lp, margin_m8 = top1_top2_margin(m8)
            # M=1 geometry top-K (from the AR generate step)
            m1 = entry_as_dict(gen_lps[g]) if g < len(gen_lps) else {}
            m1_arg, _m1_top, margin_m1 = top1_top2_margin(m1) if m1 else (ref_tok, float("nan"),
                                                                          float("nan"))
            if m1_arg == ref_tok:
                m1_argmax_eq_ref += 1
            # centered M8-vs-M1 perturbation spread. The flip-relevant quantity is the spread of
            # delta_t = lp_M8(t) - lp_M1(t) over the TOP-T M1 candidates (the only tokens that can
            # overtake the argmax); the full-K spread is tracked for reference only.
            shared = [t for t in m8 if t in m1 and math.isfinite(m8[t]) and math.isfinite(m1[t])]
            n_shared = len(shared)
            spread = float("nan")
            spread_t = {t: float("nan") for t in SPREAD_TOPTS}
            if n_shared >= 2:
                deltas_all = [m8[t] - m1[t] for t in shared]
                spread = max(deltas_all) - min(deltas_all)
                if spread > spread_max:
                    spread_max = spread
                    spread_max_where = {"prompt_index": pi, "absolute_position": j,
                                        "n_shared": n_shared, "spread": spread}
                m1_ranked = sorted(shared, key=lambda t: m1[t], reverse=True)
                for T in SPREAD_TOPTS:
                    topT = m1_ranked[:T]
                    if len(topT) >= 2:
                        d = [m8[t] - m1[t] for t in topT]
                        spread_t[T] = max(d) - min(d)
                        if spread_t[T] > spread_t_max[T]:
                            spread_t_max[T] = spread_t[T]

            verify_stream.append(m8_arg)
            flip = int(m8_arg != ref_tok)
            ok = 1 - flip
            match += ok; n_total += 1; n_match += ok
            if flip:
                # M1-rank of the flip target (m8 argmax) -- bound holds iff target within top-T
                m1_sorted = sorted(m1.items(), key=lambda kv: kv[1], reverse=True)
                tgt_rank = next((r for r, (t, _) in enumerate(m1_sorted) if t == m8_arg), 999)
                max_flip_target_rank = max(max_flip_target_rank, tgt_rank)

            if pls_b[j] is not None:
                am_b, *_ = top1_top2_margin(entry_as_dict(pls_b[j]))
                det_ver += int(am_b == m8_arg)

            positions.append({
                "prompt_index": pi, "g": g, "abs_pos": j,
                "ref_tok": ref_tok, "m8_argmax": m8_arg,
                "margin_m8": float(margin_m8) if math.isfinite(margin_m8) else None,
                "margin_m1": float(margin_m1) if math.isfinite(margin_m1) else None,
                "flip": flip,
                "spread_fullk": float(spread) if math.isfinite(spread) else None,
                "spread_t2": float(spread_t[2]) if math.isfinite(spread_t[2]) else None,
                "spread_t4": float(spread_t[4]) if math.isfinite(spread_t[4]) else None,
                "spread_t8": float(spread_t[8]) if math.isfinite(spread_t[8]) else None,
                "n_shared": n_shared,
            })
            if flip and first_div is None:
                first_div = {
                    "prompt_index": pi, "prompt_id": pr["id"],
                    "generated_offset": g, "absolute_position": j,
                    "ref_token_id": ref_tok, "spec_verify_token_id": m8_arg,
                    "margin_m8": float(margin_m8), "margin_m1": float(margin_m1),
                    "spread_t2": float(spread_t[2]) if math.isfinite(spread_t[2]) else None,
                    "spread_t4": float(spread_t[4]) if math.isfinite(spread_t[4]) else None,
                    "flip_target_m1_rank": tgt_rank,
                }

        n_det_ver += det_ver
        ref_sha = hashlib.sha256(bytes(str(gen), "utf8")).hexdigest()[:16]
        ver_sha = hashlib.sha256(bytes(str(verify_stream), "utf8")).hexdigest()[:16]
        per_prompt.append({
            "prompt_index": pi, "id": pr["id"], "context_len": c,
            "n_generated": len(gen), "compared_positions": len(verify_stream),
            "verify_match": match, "n_flip": len(verify_stream) - match,
            "sha_equal": ref_sha == ver_sha, "det_ver_match": det_ver,
        })
        print(f"[scan] prompt {pi} id={pr['id']} gen={len(gen)} "
              f"flip={len(verify_stream)-match}/{len(verify_stream)} sha_eq={ref_sha==ver_sha}",
              flush=True)

    identity = (n_match / n_total) if n_total else float("nan")
    det_gen_frac = (n_det_gen / det_gen_positions) if det_gen_positions else 1.0
    det_ver_frac = (n_det_ver / n_total) if n_total else float("nan")
    strict_pass_frac = (sum(1 for p in per_prompt if p["sha_equal"]) / len(per_prompt)
                        if per_prompt else float("nan"))
    ref_tps_local = (gen_tokens_total / ref_decode_time) if ref_decode_time > 0 else float("nan")
    ref_step_us = (1e6 / ref_tps_local) if ref_tps_local and math.isfinite(ref_tps_local) else float("nan")
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    out = {
        "phase": "margin_scan",
        "model_dir": model_dir,
        "n_prompts": len(per_prompt), "n_new": n_new, "topk": topk,
        "verify_width_requested": verify_width, "verify_width_effective": effective_width,
        "M_verify": M_VERIFY,
        "total_compared_positions": n_total, "matching_positions": n_match,
        "token_identity_rate": identity,
        "verify_divergence": (1.0 - identity) if math.isfinite(identity) else float("nan"),
        "n_flip": n_total - n_match,
        "per_sequence_strict_pass_fraction": strict_pass_frac,
        "determinism_ref_gen": det_gen_frac, "determinism_ref_gen_positions": det_gen_positions,
        "determinism_verify_geometry": det_ver_frac,
        "first_divergence": first_div,
        "spread_max_fullk": spread_max, "spread_max_where": spread_max_where,
        "spread_t2_max": spread_t_max[2], "spread_t4_max": spread_t_max[4],
        "spread_t8_max": spread_t_max[8],
        "max_flip_target_m1_rank": max_flip_target_rank,
        "m1_argmax_eq_ref_fraction": (m1_argmax_eq_ref / n_total) if n_total else float("nan"),
        "ref_tps_local_relative": ref_tps_local,
        "ref_decode_step_us_local_relative": ref_step_us,
        "ref_gen_tokens_total": gen_tokens_total, "ref_decode_time_s": ref_decode_time,
        "elapsed_s": time.time() - t0, "peak_gpu_gb": peak_gb,
        "nan_clean": bool(math.isfinite(identity)),
        "per_prompt": per_prompt,
        "positions": positions,
        "local_relative_tps": True,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[scan] identity={identity:.6f} div={1.0-identity:.6f} n_flip={n_total-n_match} "
          f"spread_t2={spread_t_max[2]:.4f} spread_t4={spread_t_max[4]:.4f} "
          f"spread_fullk={spread_max:.4f} flip_tgt_maxrank={max_flip_target_rank} "
          f"det_gen={det_gen_frac:.4f} det_ver={det_ver_frac:.4f} peak={peak_gb:.1f}GB", flush=True)
    print(f"MARGINSCAN_DONE {out_path}", flush=True)


# ======================================================================================
# GPU PHASE 2: per-position M=1 recompute cost (full bf16 lm_head GEMV + body + 1q SDPA)
# ======================================================================================
def phase_recompute_timing(out_path: str, iters: int) -> None:
    import torch

    dev = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False  # bf16 path; no tf32 surprises

    # Real Gemma-4-E4B shapes. lm_head (full vocab) dominates the strict recompute: a pruned
    # lmhead12k cannot certify the argmax, so the strict recompute reads all 262144 rows.
    VOCAB, HID, INTER, QKV, Q = 262144, 2560, 10240, 3072, 2048
    GU = 2 * INTER
    def z(*s): return torch.zeros(*s, dtype=torch.bfloat16, device=dev)
    W_lmhead = z(VOCAB, HID)
    W_qkv, W_o = z(QKV, HID), z(HID, Q)
    W_gu, W_dn = z(GU, HID), z(HID, INTER)
    x = z(1, HID)

    def lmhead_only():
        return torch.nn.functional.linear(x, W_lmhead)          # [1,2560]x[2560,262144]

    def body_one_token():
        # int4 body is bit-exact across M (#326) -- here only a bf16 GEMM-cost PROXY for the
        # decode-step compute denominator (one attn proj set + one MLP), x35 layers folded via
        # the per-layer cost * depth in the orchestrator if needed; we report a single-layer
        # stack and the full lm_head, and let the orchestrator form ratios.
        q = torch.nn.functional.linear(x, W_qkv)
        _ = torch.nn.functional.linear(q[:, :Q], W_o)
        gu = torch.nn.functional.linear(x, W_gu)
        g, u = gu[:, :INTER], gu[:, INTER:]
        _ = torch.nn.functional.linear(torch.nn.functional.silu(g) * u, W_dn)

    def attn_1q(L=512):
        # single-query SDPA against L cached keys (the per-position attn recompute)
        qh = z(8, 1, 256); kh = z(8, L, 256); vh = z(8, L, 256)
        _ = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh)

    def timed(fn, n):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.time() - t) / n * 1e6

    lmhead_us = timed(lmhead_only, iters)
    body_layer_us = timed(body_one_token, iters)
    attn_us = timed(lambda: attn_1q(512), iters)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    # f_lmhead = lm_head's share of a STRICT decode step (full-vocab bf16 lm_head + int4 body x
    # depth). Pure bandwidth model from real shapes (less noisy than wall-time, scale-portable):
    # decode is memory-bound so time ~ weight bytes read. int4 body = 0.5 byte/param, bf16
    # lm_head = 2 byte/param. n_layers read from config.
    n_layers = 42  # gemma-4-E4B text_config.num_hidden_layers (fallback if config unreadable)
    try:
        cfg = json.load(open(Path(resolve_model_dir()) / "config.json"))
        n_layers = int((cfg.get("text_config") or cfg).get("num_hidden_layers", n_layers))
    except Exception:  # noqa: BLE001
        pass
    lmhead_bytes = VOCAB * HID * 2.0
    body_params_layer = (QKV * HID) + (HID * Q) + (GU * HID) + (HID * INTER)
    body_bytes = body_params_layer * 0.5 * n_layers     # int4 body, all layers
    f_lmhead_bandwidth = lmhead_bytes / (lmhead_bytes + body_bytes)

    out = {
        "phase": "recompute_timing",
        "lmhead_full_vocab_us": lmhead_us,
        "body_single_layer_us": body_layer_us,
        "attn_single_query_us": attn_us,
        "n_layers": n_layers,
        "lmhead_bytes": lmhead_bytes, "body_bytes_int4_all_layers": body_bytes,
        "f_lmhead_bandwidth": f_lmhead_bandwidth,
        "vocab": VOCAB, "hidden": HID, "iters": iters,
        "note": ("lm_head full-vocab GEMV is the strict per-position recompute (pruned lmhead12k "
                 "cannot certify argmax). f_lmhead_bandwidth = lm_head's share of a strict decode "
                 "step (full bf16 lm_head + int4 body x n_layers), a scale-portable bandwidth "
                 "ratio. int4 body is bit-exact across M (#326)."),
        "peak_gpu_gb": peak_gb,
        "local_relative_tps": True,
        "nan_clean": bool(math.isfinite(lmhead_us) and math.isfinite(body_layer_us)),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[timing] lmhead={lmhead_us:.1f}us body1layer={body_layer_us:.1f}us "
          f"attn1q={attn_us:.1f}us (LOCAL RELATIVE) peak={peak_gb:.1f}GB", flush=True)
    print(f"RECOMPUTETIMING_DONE {out_path}", flush=True)


# ======================================================================================
# Orchestrator
# ======================================================================================
def run_phase(args_list: list[str], timeout: int | None = None) -> int:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cmd = [sys.executable, os.path.abspath(__file__)] + args_list
    print(f"[orch] launching: {' '.join(args_list)} (timeout={timeout})", flush=True)
    try:
        return subprocess.run(cmd, env=env, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        print(f"[orch] phase TIMED OUT after {timeout}s: {args_list}", flush=True)
        return 124


def _frac_below(margins: list[float], tau: float) -> float:
    if not margins:
        return float("nan")
    return sum(1 for m in margins if m < tau) / len(margins)


def _identity_after_repair(positions: list[dict], tau: float, n_total: int) -> float:
    # repair flags positions with margin_m8 < tau and takes the M=1 (ref) token there.
    # identity = 1 - (flips that are NOT flagged) / n_total
    unrepaired_flips = sum(1 for p in positions
                           if p["flip"] and (p["margin_m8"] is None or p["margin_m8"] >= tau))
    return 1.0 - unrepaired_flips / n_total if n_total else float("nan")


def _flip_recall(positions: list[dict], tau: float, n_flip: int) -> float:
    # RECALL = fraction of TRUE flips (divergent positions) the margin gate flags (margin_m8<tau).
    # The crux per advisor: the margin is a logit-level signal for a HIDDEN-state-driven divergence,
    # so recall<1.0 would leave flips un-repaired -> identity<1.0 even at low eta. recall=1.0 at
    # tau_robust is the provable-bound guarantee, confirmed empirically here.
    if not n_flip:
        return float("nan")
    flagged = sum(1 for p in positions
                  if p["flip"] and p["margin_m8"] is not None and p["margin_m8"] < tau)
    return flagged / n_flip


def _flip_margin_auc(margins_flip: list[float], margins_noflip: list[float]) -> float:
    # P(margin_flip < margin_noflip) + 0.5*P(tie): rank-based (Mann-Whitney U). A low-margin
    # flag is a perfect predictor when this == 1.0 (every flip has a smaller margin than every
    # non-flip). O(n log n) via bisection over the sorted non-flip margins.
    import bisect
    if not margins_flip or not margins_noflip:
        return float("nan")
    nf = sorted(margins_noflip)
    wins = ties = 0.0
    for m in margins_flip:
        lo = bisect.bisect_left(nf, m)   # # non-flip strictly < m
        hi = bisect.bisect_right(nf, m)  # # non-flip <= m
        wins += (len(nf) - hi)           # non-flip > m  => flip margin smaller => correct order
        ties += (hi - lo)
    return (wins + 0.5 * ties) / (len(margins_flip) * len(nf))


def orchestrate(a: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scan_json = str(OUT_DIR / "_marginscan.json")
    timing_json = str(OUT_DIR / "_timing.json")

    rc = run_phase([
        "--phase", "margin_scan", "--out", scan_json,
        "--n-prompts", str(a.n_prompts), "--n-new", str(a.n_new),
        "--ctx-cap", str(a.ctx_cap), "--verify-width", str(a.verify_width),
        "--gpu-mem-util", str(a.gpu_mem_util), "--topk", str(a.topk),
        "--det-prompts", str(a.det_prompts),
    ], timeout=a.scan_timeout)
    if rc != 0:
        raise RuntimeError(f"margin_scan phase failed (rc={rc}) -- core measurement missing")
    scan = json.load(open(scan_json))

    timing: dict[str, Any] = {"phase": "recompute_timing", "status": "not_run"}
    rc = run_phase(["--phase", "recompute_timing", "--out", timing_json,
                    "--dh-iters", str(a.dh_iters)], timeout=a.timing_timeout)
    if rc == 0 and Path(timing_json).exists():
        timing = json.load(open(timing_json))
    else:
        timing["status"] = f"failed_rc_{rc}"

    compose(a, scan, timing)


def compose(a: argparse.Namespace, scan: dict, timing: dict) -> None:
    positions = scan["positions"]
    n_total = scan["total_compared_positions"]
    identity = scan["token_identity_rate"]
    divergence = scan["verify_divergence"]
    spread_fullk = scan["spread_max_fullk"]      # tail-contaminated full top-K spread (reference)
    spread_t2 = scan["spread_t2_max"]            # flip-relevant top-2 spread (tightest provable)
    spread_t4 = scan["spread_t4_max"]            # flip-relevant top-4 M1-candidate spread
    spread_t8 = scan["spread_t8_max"]            # flip-relevant top-8 M1-candidate spread
    flip_tgt_maxrank = scan["max_flip_target_m1_rank"]   # 0-indexed; -1 if no flips observed

    margins_all = [p["margin_m8"] for p in positions if p["margin_m8"] is not None]
    flips = [p for p in positions if p["flip"]]
    margins_flip = [p["margin_m8"] for p in flips if p["margin_m8"] is not None]
    margins_noflip = [p["margin_m8"] for p in positions if not p["flip"] and p["margin_m8"] is not None]
    n_flip = len(flips)

    # ---- margin-flip concentration ----
    max_flip_margin = max(margins_flip) if margins_flip else float("nan")
    flip_margin_auc = _flip_margin_auc(margins_flip, margins_noflip)
    # empirical thresholds: smallest tau catching 100% / 99% of flips
    tau100 = (max_flip_margin + 1e-6) if margins_flip else 0.0
    if margins_flip:
        srt = sorted(margins_flip)
        idx99 = max(0, math.ceil(0.99 * len(srt)) - 1)
        tau99 = srt[idx99] + 1e-9
    else:
        tau99 = 0.0
    # provable robustness threshold: every flip has margin_m8 < the spread of (lp_M8 - lp_M1) over
    # a candidate set containing BOTH the M1 argmax (rank 0) and the flip target (M1-rank tgt).
    # The top-T M1 set covers a flip iff tgt_rank < T, so pick the TIGHTEST top-T spread that
    # provably covers EVERY observed flip target; fall back to the full-K spread if a target sits
    # beyond top-8. With no flips, identity is already 1.0 and we report the tight top-4 bound.
    if flip_tgt_maxrank < 0:
        tau_robust, spread_basis = spread_t2, "t2_no_flips"
    elif flip_tgt_maxrank < 2:
        tau_robust, spread_basis = spread_t2, "t2"
    elif flip_tgt_maxrank < 4:
        tau_robust, spread_basis = spread_t4, "t4"
    elif flip_tgt_maxrank < 8:
        tau_robust, spread_basis = spread_t8, "t8"
    else:
        tau_robust, spread_basis = spread_fullk, "fullk"
    bound_holds = all((m < tau_robust + 1e-9) for m in margins_flip) if margins_flip else True

    # ---- tau-sweep frontier ----
    sweep_taus = sorted(set(
        [0.0, tau99, tau100, tau_robust]
        + [round(0.05 * k, 4) for k in range(0, 41)]   # 0.00 .. 2.00 nats
    ))
    sweep = []
    for tau in sweep_taus:
        frac = _frac_below(margins_all, tau)
        ident = _identity_after_repair(positions, tau, n_total)
        recall = _flip_recall(positions, tau, n_flip)
        sweep.append({"tau": tau, "frac_below": frac, "identity_after_repair": ident,
                      "flip_recall": recall, "selective_eta_conservative": frac})

    frac_at_robust = _frac_below(margins_all, tau_robust)
    frac_at_100 = _frac_below(margins_all, tau100)
    frac_at_99 = _frac_below(margins_all, tau99)
    ident_at_robust = _identity_after_repair(positions, tau_robust, n_total)
    ident_at_100 = _identity_after_repair(positions, tau100, n_total)
    # RECALL of the margin gate (advisor-required): does margin<tau flag ~100% of true flips?
    recall_at_robust = _flip_recall(positions, tau_robust, n_flip)
    recall_at_100 = _flip_recall(positions, tau100, n_flip)
    recall_at_99 = _flip_recall(positions, tau99, n_flip)

    # ---- recompute cost models ----
    lmhead_us = timing.get("lmhead_full_vocab_us")
    body_layer_us = timing.get("body_single_layer_us")
    attn_us = timing.get("attn_single_query_us")
    # f_lmhead = lm_head's share of a STRICT decode step. Decode is memory-bound, so use the
    # scale-portable BANDWIDTH ratio (full bf16 lm_head bytes vs int4 body bytes x n_layers)
    # computed in the timing phase; fall back to the noisier single-layer wall-time ratio.
    f_lmhead = timing.get("f_lmhead_bandwidth")
    if not (isinstance(f_lmhead, (int, float)) and math.isfinite(f_lmhead)):
        f_lmhead = (lmhead_us / (lmhead_us + body_layer_us)
                    if (lmhead_us and body_layer_us) else float("nan"))
    # CORRECT cost (wirbel #362): a flagged position needs a FULL M=1 forward (lm_head-only does
    # NOT restore identity -- divergence is hidden-state-driven). A full M=1 forward ~= one decode
    # step, and the margin probe is FREE (read off the M=8 logits we already have), so the headline
    # selective_eta = frac_below_tau * 1.0. The lm_head-only variant (frac * f_lmhead) is reported
    # ONLY as the NOT-identity-valid floor it is (wirbel #362: lm_head-only identity = 0.9948<1.0).
    def eta_full_m1(frac): return frac                    # full M=1 recompute = one decode step
    def eta_lmhead_only(frac): return frac * f_lmhead if math.isfinite(f_lmhead) else float("nan")

    selective_eta_at_identity_1 = eta_full_m1(frac_at_robust)        # HEADLINE (provable, valid)
    selective_eta_empirical = eta_full_m1(frac_at_100)              # in-sample tau100 (optimistic tau)
    selective_eta_lmhead_only_NOT_valid = eta_lmhead_only(frac_at_robust)  # refuted by #362
    selective_beats_blanket = bool(selective_eta_at_identity_1 < BLANKET_ETA)
    selective_clears_500_budget = bool(selective_eta_at_identity_1 < BUDGET_ETA)
    selective_beats_probe_gated = bool(selective_eta_at_identity_1 <= PROBE_GATED_ETA_362)
    # empirical-tau gates (in-sample tau100; not out-of-sample guaranteed but the tighter optimum)
    beats_blanket_emp = bool(selective_eta_empirical < BLANKET_ETA)
    clears_budget_emp = bool(selective_eta_empirical < BUDGET_ETA)

    flips_margin_concentrated = bool(
        math.isfinite(flip_margin_auc) and flip_margin_auc >= 0.95 and frac_at_robust < 0.5)
    # the GUARANTEE holds iff the margin gate at tau_robust catches every flip (recall=1.0 ->
    # identity_after_repair=1.0). With no flips in-sample we cannot exercise/measure selectivity.
    identity_restored = (n_flip > 0 and recall_at_robust is not None
                         and recall_at_robust >= 1.0 - 1e-9
                         and abs(ident_at_robust - 1.0) < 1e-9)

    # ---- decision verdict ----
    if n_flip == 0:
        verdict_band = "INCONCLUSIVE_no_flips_in_sample"
    elif not identity_restored or selective_eta_at_identity_1 >= BLANKET_ETA:
        verdict_band = "RED_selectivity_no_help"
    elif selective_eta_at_identity_1 < BUDGET_ETA:
        verdict_band = "GREEN_live_cheaper_identity_clears_500"
    else:
        verdict_band = "AMBER_beats_blanket_below_budget_gap"

    # ---- self-test ----
    checks = {
        "scan_positions_present": n_total > 0 and len(positions) == n_total,
        "nan_clean": bool(scan["nan_clean"]) and math.isfinite(identity) and math.isfinite(divergence),
        "identity_in_range": (0.0 <= identity <= 1.0),
        "divergence_eq_1_minus_identity": abs(divergence - (1.0 - identity)) < 1e-9,
        "first_div_iff_div_gt_0": ((divergence > 0) == (scan["first_divergence"] is not None)),
        "m1_argmax_eq_ref": scan["m1_argmax_eq_ref_fraction"] > 0.999,
        "robust_bound_holds_all_flips_below_tau_robust": bool(bound_holds),
        "identity_1_at_tau_robust": abs(ident_at_robust - 1.0) < 1e-9,
        "identity_1_at_tau100": abs(ident_at_100 - 1.0) < 1e-9,
        "flip_recall_1_at_tau_robust": (n_flip == 0 or abs(recall_at_robust - 1.0) < 1e-9),
        "recall_monotone_in_tau": all(
            (sweep[i]["flip_recall"] is None or sweep[i + 1]["flip_recall"] is None
             or sweep[i]["flip_recall"] <= sweep[i + 1]["flip_recall"] + 1e-12)
            for i in range(len(sweep) - 1)),
        "frac_monotone_in_tau": all(sweep[i]["frac_below"] <= sweep[i + 1]["frac_below"] + 1e-12
                                    for i in range(len(sweep) - 1)),
        "identity_monotone_in_tau": all(
            sweep[i]["identity_after_repair"] <= sweep[i + 1]["identity_after_repair"] + 1e-12
            for i in range(len(sweep) - 1)),
        "determinism_ref_gen_eq_1": scan["determinism_ref_gen"] == 1.0,
        "determinism_verify_eq_1": scan["determinism_verify_geometry"] == 1.0,
        "corroborates_361_identity": abs(identity - RAW_IDENTITY_361) <= 0.01,
        "timing_present": bool(lmhead_us and body_layer_us),
        "no_hf_job_no_launch": True,
        "tps_local_relative_tagged": bool(scan.get("local_relative_tps")),
    }
    self_test_passes = bool(all(checks.values()))

    no_block = {"no_hf_job": True, "no_launch": True, "no_submission": True,
                "no_served_file_change": True, "gpu_used": True, "local_pod_gpu_only": True}

    report = {
        "card": "margin_localized_identity", "pr": 364, "issue": 319, "author": "ubel",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        **no_block,
        # PRIMARY + TEST
        "token_identity_rate": identity,
        "margin_identity_self_test_passes": self_test_passes,
        # margin-flip concentration (step 1)
        "verify_divergence": divergence, "n_flip": n_flip,
        "first_divergence": scan["first_divergence"],
        "max_flip_margin_m8": max_flip_margin,
        "flip_margin_auc": flip_margin_auc,
        "spread_max_perturbation": tau_robust,   # flip-relevant top-T spread (= tau_robust)
        "tau_robust_spread_basis": spread_basis,
        "spread_t2_max": spread_t2,
        "spread_t4_max": spread_t4, "spread_t8_max": spread_t8,
        "spread_max_fullk": spread_fullk, "spread_max_where": scan["spread_max_where"],
        "max_flip_target_m1_rank": flip_tgt_maxrank,
        "tau_robust": tau_robust, "tau100_empirical": tau100, "tau99_empirical": tau99,
        "robust_bound_holds": bool(bound_holds),
        "flips_margin_concentrated": flips_margin_concentrated,
        "identity_restored_at_tau_robust": identity_restored,
        # selective eta (step 2/3)
        "frac_positions_below_tau_robust": frac_at_robust,
        "frac_positions_below_tau100": frac_at_100,
        "frac_positions_below_tau99": frac_at_99,
        "identity_after_repair_at_tau_robust": ident_at_robust,
        "identity_after_repair_at_tau100": ident_at_100,
        # RECALL of the margin gate (advisor-required: logit signal vs hidden-driven divergence)
        "flip_recall_at_tau_robust": recall_at_robust,
        "flip_recall_at_tau100": recall_at_100,
        "flip_recall_at_tau99": recall_at_99,
        "lmhead_full_vocab_recompute_us": lmhead_us,
        "body_single_layer_us": body_layer_us, "attn_single_query_us": attn_us,
        "f_lmhead_compute_fraction": f_lmhead,
        # selective eta -- HEADLINE is the full-M=1 cost (lm_head-only is NOT identity-valid, #362)
        "selective_eta_at_identity_1": selective_eta_at_identity_1,
        "selective_eta_empirical_tau100": selective_eta_empirical,
        "selective_eta_lmhead_only_NOT_identity_valid": selective_eta_lmhead_only_NOT_valid,
        "recompute_is_full_m1_forward": True,
        "lmhead_only_identity_362": LMHEAD_ONLY_IDENTITY_362,
        # decision
        "blanket_eta": BLANKET_ETA, "budget_eta_500": BUDGET_ETA,
        "probe_gated_eta_362": PROBE_GATED_ETA_362,
        "selective_beats_blanket": selective_beats_blanket,
        "selective_clears_500_budget": selective_clears_500_budget,
        "selective_beats_probe_gated_362": selective_beats_probe_gated,
        "selective_beats_blanket_empirical_tau": beats_blanket_emp,
        "selective_clears_500_budget_empirical_tau": clears_budget_emp,
        "verdict_band": verdict_band,
        # controls / reconcile
        "determinism_ref_gen": scan["determinism_ref_gen"],
        "determinism_verify_geometry": scan["determinism_verify_geometry"],
        "per_sequence_strict_pass_fraction": scan["per_sequence_strict_pass_fraction"],
        "raw_identity_361": RAW_IDENTITY_361,
        "deployed_m8_divergence_232": DEPLOYED_M8_DIVERGENCE_232,
        # secondary / bookkeeping
        "ref_tps_local_relative": scan["ref_tps_local_relative"],
        "ref_decode_step_us_local_relative": scan["ref_decode_step_us_local_relative"],
        "model_dir": scan["model_dir"],
        "substrate": "int4 w4a16 deployed path (bit-exact strict-ladder reference, #196/#232)",
        "verify_width_effective": scan["verify_width_effective"],
        "M_verify": M_VERIFY, "k_spec": K_SPEC,
        "n_prompts": scan["n_prompts"], "n_new": scan["n_new"], "topk": scan["topk"],
        "total_compared_positions": n_total,
        "official_baseline_unchanged": OFFICIAL_BASELINE,
        "lambda1_ceil": LAMBDA1_CEIL, "step_us_wirbel": STEP_US_WIRBEL,
        "peak_gpu_gb": max(scan.get("peak_gpu_gb", 0.0), timing.get("peak_gpu_gb", 0.0)),
        "self_test": checks,
        "tau_sweep": sweep,
        "local_relative_tps": True,
    }

    report_path = OUT_DIR / "_results.json"
    json.dump(report, open(report_path, "w"), indent=2, default=str)

    bar = "=" * 86
    print("\n" + bar, flush=True)
    print(" MARGIN-LOCALIZED IDENTITY REPAIR (PR #364, #319) -- selective sub-blanket eta?", flush=True)
    print(bar, flush=True)
    print(f" token_identity_rate (PRIMARY, raw M=8)   : {identity:.6f}  (361 anchor {RAW_IDENTITY_361:.6f})", flush=True)
    print(f" n_flip / positions                       : {n_flip} / {n_total}", flush=True)
    print(f" flip_margin_auc (separability)           : {flip_margin_auc:.4f}", flush=True)
    print(f" max flip margin_m8                        : {max_flip_margin:.4f} nats", flush=True)
    print(f" tau_robust (top-{spread_basis} spread)        : {tau_robust:.4f} nats  bound_holds={bound_holds}", flush=True)
    print(f"   spreads t2/t4/t8/fullk + flip_tgt_rank  : {spread_t2:.3f} / {spread_t4:.3f} / "
          f"{spread_t8:.3f} / {spread_fullk:.3f} nats  rank={flip_tgt_maxrank}", flush=True)
    print(f" frac positions < tau_robust              : {frac_at_robust:.4%}", flush=True)
    print(f" frac positions < tau100 (all flips)      : {frac_at_100:.4%}", flush=True)
    print(f" flip RECALL @ tau_robust / tau100        : {recall_at_robust} / {recall_at_100}", flush=True)
    print(f" identity after repair @ tau_robust       : {ident_at_robust:.6f}", flush=True)
    print(f" f_lmhead (lm_head share, bandwidth)       : {f_lmhead:.4f} "
          f"(lmhead_us={lmhead_us} body1_us={body_layer_us})", flush=True)
    print(f" --- selective eta @ identity=1.0 (full-M=1 recompute; lm_head-only invalid #362) ---", flush=True)
    print(f"  HEADLINE  selective_eta (tau_robust)     : {selective_eta_at_identity_1:.4%}", flush=True)
    print(f"  empirical (tau100, in-sample)            : {selective_eta_empirical:.4%}", flush=True)
    print(f"  lm_head-only floor (NOT identity-valid)  : {selective_eta_lmhead_only_NOT_valid:.4%}", flush=True)
    print(f"  blanket eta (wirbel #360)                : {BLANKET_ETA:.4%}", flush=True)
    print(f"  >500 budget eta                          : {BUDGET_ETA:.4%}", flush=True)
    print(f"  probe-gated floor (wirbel #362)          : {PROBE_GATED_ETA_362:.4%}", flush=True)
    print(f" beats_blanket / clears_500 / beats_probe  : {selective_beats_blanket} / "
          f"{selective_clears_500_budget} / {selective_beats_probe_gated}", flush=True)
    print(f" VERDICT BAND                              : {verdict_band}", flush=True)
    print(f" SELF-TEST PASSES (TEST metric)            : {self_test_passes}", flush=True)
    print(f" report -> {report_path}", flush=True)
    print(bar + "\n", flush=True)

    run_ids = []
    if not a.no_wandb:
        run_ids = log_wandb(report, a)
    report["wandb_run_ids"] = run_ids
    json.dump(report, open(report_path, "w"), indent=2, default=str)

    marker = {
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": run_ids,
        "primary_metric": {"name": "token_identity_rate", "value": identity},
        "test_metric": {"name": "margin_identity_self_test_passes", "value": int(self_test_passes)},
    }
    print("SENPAI-RESULT: " + json.dumps(marker), flush=True)


def log_wandb(report: dict, a: argparse.Namespace) -> list[str]:
    try:
        import wandb as _wb  # noqa: F401
        if not hasattr(_wb, "init"):
            raise ImportError("resolved a stub wandb with no .init (shadowed by a wandb/ dir)")
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] real wandb import failed (analysis unaffected): {exc}", flush=True)
        return []
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    try:
        from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                            log_json_artifact, log_summary)
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] scripts.wandb_logging import failed (analysis unaffected): {exc}", flush=True)
        return []
    os.environ.setdefault("WANDB_DIR", "/tmp/wandb_ubel_margin")
    Path(os.environ["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)
    run = init_wandb_run(
        job_type="local_profiling", agent="ubel",
        name=a.wandb_name, group=a.wandb_group,
        notes="PR#364 margin-localized identity repair: per-position M=8 verify margin vs flip, "
              "tau_robust=top-T flip-relevant spread, selective_eta vs the 9.841% blanket and "
              "4.02% >500 budget. "
              "LOCAL pod-GPU, 0 official TPS, no HF job/launch/submission/served-file change.",
        tags=["margin-localized", "greedy-identity", "selective-repair", "strict-gate",
              "local-relative", "issue-319", "pr-364"],
        config={"pr": 364, "issue": 319, "wandb_group": a.wandb_group,
                "M_verify": M_VERIFY, "k_spec": K_SPEC, "topk": report["topk"],
                "substrate": report["substrate"], "n_prompts": report["n_prompts"],
                "n_new": report["n_new"], "verify_width_effective": report["verify_width_effective"],
                "blanket_eta": BLANKET_ETA, "budget_eta_500": BUDGET_ETA,
                "official_baseline": OFFICIAL_BASELINE},
    )
    if run is None:
        print("[wandb] disabled (no API key/mode); JSON-only", flush=True)
        return []
    summ = {
        "token_identity_rate": report["token_identity_rate"],
        "margin_identity_self_test_passes": int(report["margin_identity_self_test_passes"]),
        "verify_divergence": report["verify_divergence"], "n_flip": report["n_flip"],
        "flip_margin_auc": report["flip_margin_auc"],
        "max_flip_margin_m8": report["max_flip_margin_m8"],
        "spread_max_perturbation": report["spread_max_perturbation"],
        "spread_t2_max": report["spread_t2_max"],
        "spread_t4_max": report["spread_t4_max"], "spread_t8_max": report["spread_t8_max"],
        "spread_max_fullk": report["spread_max_fullk"],
        "max_flip_target_m1_rank": report["max_flip_target_m1_rank"],
        "tau_robust": report["tau_robust"], "tau100_empirical": report["tau100_empirical"],
        "frac_positions_below_tau_robust": report["frac_positions_below_tau_robust"],
        "frac_positions_below_tau100": report["frac_positions_below_tau100"],
        "identity_after_repair_at_tau_robust": report["identity_after_repair_at_tau_robust"],
        "flip_recall_at_tau_robust": report["flip_recall_at_tau_robust"],
        "flip_recall_at_tau100": report["flip_recall_at_tau100"],
        "lmhead_full_vocab_recompute_us": report["lmhead_full_vocab_recompute_us"],
        "f_lmhead_compute_fraction": report["f_lmhead_compute_fraction"],
        "selective_eta_at_identity_1": report["selective_eta_at_identity_1"],
        "selective_eta_empirical_tau100": report["selective_eta_empirical_tau100"],
        "selective_eta_lmhead_only_NOT_identity_valid": report["selective_eta_lmhead_only_NOT_identity_valid"],
        "blanket_eta": BLANKET_ETA, "budget_eta_500": BUDGET_ETA,
        "probe_gated_eta_362": PROBE_GATED_ETA_362,
        "selective_beats_blanket": int(report["selective_beats_blanket"]),
        "selective_clears_500_budget": int(report["selective_clears_500_budget"]),
        "selective_beats_probe_gated_362": int(report["selective_beats_probe_gated_362"]),
        "selective_beats_blanket_empirical_tau": int(report["selective_beats_blanket_empirical_tau"]),
        "selective_clears_500_budget_empirical_tau": int(report["selective_clears_500_budget_empirical_tau"]),
        "determinism_ref_gen": report["determinism_ref_gen"],
        "determinism_verify_geometry": report["determinism_verify_geometry"],
        "per_sequence_strict_pass_fraction": report["per_sequence_strict_pass_fraction"],
        "total_compared_positions": report["total_compared_positions"],
        "tps_added_by_this_card": 0, "peak_gpu_gb": report["peak_gpu_gb"],
    }
    summ = {k: v for k, v in summ.items() if v is not None and not (isinstance(v, float) and math.isnan(v))}
    log_summary(run, summ, step=0)
    run.summary["verdict_band"] = report["verdict_band"]
    for k, v in report["self_test"].items():
        run.summary[f"selftest/{k}"] = int(bool(v))
    # tau-sweep frontier as a wandb Table
    try:
        import wandb
        tbl = wandb.Table(columns=["tau", "frac_below", "identity_after_repair",
                                   "flip_recall", "selective_eta_conservative"])
        for row in report["tau_sweep"]:
            tbl.add_data(row["tau"], row["frac_below"], row["identity_after_repair"],
                         row.get("flip_recall"), row.get("selective_eta_conservative"))
        run.log({"tau_sweep_frontier": tbl, "global_step": 0})
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] tau_sweep table skipped: {exc}", flush=True)
    log_json_artifact(run, name="margin_localized_identity_result", artifact_type="analysis",
                      data=report)
    rid = getattr(run, "id", "") or ""
    finish_wandb(run)
    print(f"[wandb] logged run {rid}", flush=True)
    return [rid] if rid else []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["margin_scan", "recompute_timing"], default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpu", action="store_true", help="orchestrate the full screen on GPU")
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate the path")
    ap.add_argument("--n-prompts", type=int, default=16)
    ap.add_argument("--n-new", type=int, default=512)
    ap.add_argument("--ctx-cap", type=int, default=256)
    ap.add_argument("--verify-width", type=int, default=M_VERIFY)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--det-prompts", type=int, default=2)
    ap.add_argument("--gpu-mem-util", type=float, default=0.55)
    ap.add_argument("--dh-iters", type=int, default=200)
    ap.add_argument("--scan-timeout", type=int, default=4200)
    ap.add_argument("--timing-timeout", type=int, default=600)
    ap.add_argument("--wandb_group", dest="wandb_group", default="margin-localized-identity")
    ap.add_argument("--wandb_name", dest="wandb_name", default="ubel/margin-localized-identity-eta")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--relog-wandb", action="store_true",
                    help="log the existing _results.json to wandb without re-running GPU phases")
    a = ap.parse_args()

    if a.relog_wandb:
        report = json.load(open(OUT_DIR / "_results.json"))
        run_ids = log_wandb(report, a)
        report["wandb_run_ids"] = run_ids
        json.dump(report, open(OUT_DIR / "_results.json", "w"), indent=2, default=str)
        marker = {"terminal": True, "status": "complete", "pending_arms": False,
                  "wandb_run_ids": run_ids,
                  "primary_metric": {"name": "token_identity_rate", "value": report["token_identity_rate"]},
                  "test_metric": {"name": "margin_identity_self_test_passes",
                                  "value": int(report["margin_identity_self_test_passes"])}}
        print("SENPAI-RESULT: " + json.dumps(marker), flush=True)
        return

    if a.smoke:
        a.n_prompts = min(a.n_prompts, 3)
        a.n_new = min(a.n_new, 48)
        a.det_prompts = min(a.det_prompts, 1)
        a.dh_iters = min(a.dh_iters, 30)

    if a.phase == "margin_scan":
        phase_margin_scan(a.out, a.n_prompts, a.n_new, a.ctx_cap, a.verify_width,
                          a.gpu_mem_util, a.topk, a.det_prompts)
    elif a.phase == "recompute_timing":
        phase_recompute_timing(a.out, a.dh_iters)
    else:
        orchestrate(a)


if __name__ == "__main__":
    main()
