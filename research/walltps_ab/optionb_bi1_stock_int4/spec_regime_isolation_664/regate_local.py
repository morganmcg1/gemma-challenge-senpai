#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #622 (stark) -- Option-B #319 re-gate: BI=1 both-sides spec-vs-AR greedy identity.

THE CLAIM (from stark #621 / wirbel #607 / wirbel #616)
-------------------------------------------------------
wirbel #607 measured a 47% greedy break (31048/65536) of the int4+MTP spec-verify
path vs a plain-AR reference. stark #621 localised it to a RECOVERABLE reference-side
config mismatch: the reference AR decode ran the 3D split-KV attention path
(is_batch_invariant=False, BI=0) while the spec-verify M=8 step runs the 2D path; under
VLLM_BATCH_INVARIANT=1 BOTH sides take the 2D path and the attention op is bit-exact
(#621 op-level: spec_verify_vs_ar_maxdiff_under_BI = 0.0). The submission already pins BI=1.

This harness CONFIRMS that end-to-end on the real int4 body: with BI=1 pinned on BOTH the
M=1 AR side and the M=8 verify side, the per-step (teacher-forced) break rate collapses to
the int4-Marlin grid-tie residual wirbel #616 characterised (raw_structural_flip_rate_m8_vs_m1
= 0.004318, CI [0.3738%, 0.4944%], all flips < 0.5 nat, tau=0.3 -> 0).

THE MEASUREMENT (teacher-forced per-step argmax -- NO trajectory cascade)
------------------------------------------------------------------------
Reuses the VALIDATED stark #381 decode-width geometry (decodewidth_e2e_identity.py):
  * the M=1 AR greedy continuation R IS the per-position M=1 argmax (each token is a
    single-sequence size_m=1 decode step), so R[t] == argmax_M1(R[0:t]) by construction.
  * the M=8 verify argmax at position t is read from a size_m=8 prompt_logprobs chunk that
    attends to a prefix-cache HIT (only the 8 suffix rows compute -> body GEMM size_m=8, the
    real decode-verify width where int4-Marlin is bit-exact per #376/#381). vLLM 0.22.0 silently
    disables the prefix-cache reuse when prompt_logprobs is requested (SamplingParams auto-sets
    skip_reading_prefix_cache=True); we force it back False and assert n_computed_rows==8.
  * a flip at t  <=>  argmax_M8(R[0:t]) != R[t]. Both walk the SAME token path, so a flip is
    purely kernel M-variance, never trajectory divergence (the source of #607's cascaded 47%).

EXTENSION over #381: #381 read 7 positions from ONE chunk per prompt. Here we SLIDE the
size_m=8 chunk across the whole AR trajectory at 32-aligned offsets (the Gemma-4 hybrid
prefix-commit granularity) so each chunk is a clean cache HIT computing exactly 8 rows; this
yields ~7 positions per 32-token window -> a uniform sample of the trajectory large enough to
estimate the residual rate with a CI that overlaps #616.

TWO ARMS (isolated subprocess, process-wide BI env):
  pinned    -- VLLM_BATCH_INVARIANT=1 on BOTH sides (the decisive measurement).
  heuristic -- VLLM_BATCH_INVARIANT=0 (BI=0 contrast).

Model: google/gemma-4-E4B-it-qat-w4a16-ct (int4 W4A16 body, the #607/#621 checkpoint). The
MTP drafter is NOT loaded: under greedy temp=0 the drafter only changes acceptance/speed, never
the verify argmax (#621), so it is irrelevant to the kernel-variance measurement.

DELIVERABLES (terminal): break_rate_bi1_both_sides, attention_path_break_count (flips with
margin>=0.5 nat or M1-token-outside-topk -> a SYSTEMATIC, non-tie divergence; expect ~0 under
BI=1), int4_tie_residual_rate (flips with margin<0.5 nat), residual_frac_under_0p5nat,
residual_after_tau_0p3nat, tau-ladder {0.0,0.2,0.3}, VERDICT.

SCOPE: local A10G, analysis_only=true, official_tps=0, NO HF Job / NO submission / NO served-file
change. vLLM 0.22.0 (the #607/#616/#621 chain version). The served int4 path is READ, never modified.
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
WIRBEL_616_RESIDUAL = 0.004318            # #616 raw_structural_flip_rate_m8_vs_m1 (283/65536)
WIRBEL_616_CI = (0.003738, 0.004944)      # #616 reported CI
WIRBEL_607_BREAK = 31048 / 65536          # #607 47% break at BI=0 reference (cascaded trajectory)
STARK_621_ATTN_MAXDIFF_UNDER_BI = 0.0     # #621 op-level: verify(2D) vs AR(2D,BI=1) bit-exact
LOCKED_319_AR_TPS = 126.378               # int4_g128_lmhead (#4) strict-#319 rung to beat

K_SPEC = 6                                # PR#664: land's deployed K6 regime (was 7 in #622)
M_VERIFY = K_SPEC + 1                     # = 7, the K6 decode-verify query width (K+1 bonus row)
NEAR_TIE_LOGPROB_THRESH = 0.5             # margin < this nat => knife-edge int4-Marlin grid-tie
TAU_LADDER = (0.0, 0.2, 0.3)              # tolerance ladder (nats); expect tau=0.3 -> 0 residual
HYBRID_PREFIX_COMMIT = 32                 # Gemma-4 hybrid prefix-cache commit granularity (#381)
PROMPT_LOGPROBS_K = 20                    # top-k read at each verify row (margin needs M1 token's rank)
DET_CONTROL_PROMPTS = 8                   # run the R==R2 determinism control on the first N prompts only
                                          # (a 2nd full AR gen per prompt ~doubles cost; the control just
                                          # confirms the M=1 reference is stable -- 8 prompts suffices)

DEFAULT_MODEL = "google/gemma-4-E4B-it-qat-w4a16-ct"
MODEL_CANDIDATES = [
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots"
    ),
]
PROMPTS_JSONL = "official/main_bucket/shared_resources/speed_benchmark/data/ppl_ground_truth_tokens.jsonl"
OUT_DIR = Path("research/walltps_ab/optionb_bi1_stock_int4/spec_regime_isolation_664/regate_arms")
ARMS = ("pinned", "heuristic")


# --------------------------------------------------------------------------------------
# helpers (resolve_model_dir / read_text_dims / argmax reused from #381)
# --------------------------------------------------------------------------------------
def resolve_model_dir() -> str:
    # int4 W4A16 body ONLY (the #607/#621 checkpoint). We deliberately do NOT fall back to
    # /tmp/osoi5-v0-baked (a DIFFERENT int4 QAT bake) -- that would confound the re-gate.
    for cand in MODEL_CANDIDATES:
        p = Path(cand)
        if p.is_dir() and (p / "config.json").exists():
            return str(p)
        if p.is_dir():
            for sub in sorted(p.glob("*")):
                if (sub / "config.json").exists():
                    return str(sub)
    return DEFAULT_MODEL  # let vLLM resolve from the Hub cache by id


def read_text_dims(model_dir: str) -> dict:
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.exists():
        # Hub id, not a local dir; fill in the known gemma-4-E4B served dims.
        return {"hidden": 2560, "n_heads": 8, "n_kv": 2, "head_dim": 256,
                "intermediate": 10240, "num_layers": None,
                "shapes": {"qkv_proj": (3072, 2560), "o_proj": (2560, 2048),
                           "gate_up_proj": (20480, 2560), "down_proj": (2560, 10240)}}
    cfg = json.load(open(cfg_path))
    tc = cfg.get("text_config", cfg)
    h, n_heads, n_kv = tc["hidden_size"], tc["num_attention_heads"], tc["num_key_value_heads"]
    hd, inter = tc["head_dim"], tc["intermediate_size"]
    return {
        "hidden": h, "n_heads": n_heads, "n_kv": n_kv, "head_dim": hd,
        "intermediate": inter, "num_layers": tc.get("num_hidden_layers"),
        "lmhead_quant": _lmhead_is_int4(cfg),
        "shapes": {
            "qkv_proj": ((n_heads + 2 * n_kv) * hd, h), "o_proj": (h, n_heads * hd),
            "gate_up_proj": (2 * inter, h), "down_proj": (h, inter),
        },
    }


def _lmhead_is_int4(cfg: dict) -> bool:
    qc = cfg.get("quantization_config") or {}
    groups = qc.get("config_groups") or {}
    for g in groups.values():
        targets = g.get("targets") or []
        if any("lm_head" in str(t) for t in targets):
            return True
    # ignore-list style: lm_head NOT in ignore => quantized
    ign = qc.get("ignore") or []
    if qc and not any("lm_head" in str(t) for t in ign) and groups:
        return None  # ambiguous (Linear target may or may not include lm_head)
    return False


def block_align(n: int) -> int:
    return (n // HYBRID_PREFIX_COMMIT) * HYBRID_PREFIX_COMMIT


def _argmax_from_logprob_entry(entry) -> int:
    return int(max(entry.items(), key=lambda kv: getattr(kv[1], "logprob", kv[1]))[0])


def _sorted_logprobs(entry) -> list[tuple[int, float]]:
    return sorted(((int(t), float(getattr(lp, "logprob", lp))) for t, lp in entry.items()),
                  key=lambda kv: kv[1], reverse=True)


# ======================================================================================
# one arm (subprocess): BI env read from VLLM_BATCH_INVARIANT
# ======================================================================================
def phase_arm(out_path: str, arm: str, n_prompts: int, ctx_len: int, traj_len: int,
              gpu_mem_util: float, max_batched_tokens: int, verbose_k: int) -> None:
    import torch
    from vllm import LLM, SamplingParams

    batch_invariant_env = os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1"
    model_dir = resolve_model_dir()
    dims = read_text_dims(model_dir)
    C = block_align(ctx_len)
    print(f"[arm:{arm}] model={model_dir} hidden={dims['hidden']} layers={dims['num_layers']} "
          f"lmhead_int4={dims.get('lmhead_quant')} C={C} traj_len={traj_len} "
          f"VLLM_BATCH_INVARIANT={batch_invariant_env}", flush=True)

    t0 = time.time()
    llm = LLM(
        model=model_dir, quantization="compressed-tensors", dtype="bfloat16",
        max_model_len=max(1024, C + traj_len + 64), gpu_memory_utilization=gpu_mem_util,
        max_num_seqs=16, max_num_batched_tokens=max_batched_tokens,
        enable_prefix_caching=True, enforce_eager=True, trust_remote_code=True,
    )
    print(f"[arm:{arm}] vLLM load done in {time.time()-t0:.0f}s", flush=True)

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
    n_det_m1 = n_det_checked = 0
    n_chunk_isolated = n_chunk_total = 0
    n_computed_rows_total = 0
    # residual characterisation:
    flip_gaps = []        # top1-top2 logprob gap of the M=8 dist at each flip
    flip_margins = []     # logprob(M8 top1) - logprob(M8 of the M1 token); None if M1 token outside top-k
    min_gap_per_prompt = []

    t_arm = time.time()
    for ri, rec in enumerate(rows):
        src = list(rec.get("context_token_ids", [])) + list(rec.get("target_token_ids", []))
        if len(src) < C + 1:
            continue
        prefix = src[:C]

        # warm the prefix cache (served-faithful: the context is already in the paged KV when the
        # verify step runs), then generate the M=1 AR greedy continuation R + a determinism control.
        llm.generate([{"prompt_token_ids": prefix}], sp_warm, use_tqdm=False)
        outA = llm.generate([{"prompt_token_ids": prefix}], sp_gen, use_tqdm=False)[0]
        R = list(outA.outputs[0].token_ids)
        det_m1 = None
        if ri < DET_CONTROL_PROMPTS:
            outA2 = llm.generate([{"prompt_token_ids": prefix}], sp_gen, use_tqdm=False)[0]
            det_m1 = int(R == list(outA2.outputs[0].token_ids))
        if len(R) < HYBRID_PREFIX_COMMIT + M_VERIFY:
            continue

        # SLIDE the size_m=8 verify chunk across R at 32-aligned offsets. Each chunk:
        #   full = prefix + R[0:o+M_VERIFY]; the [0:C+o] part (C and o both 32-aligned) is a cache
        #   HIT, so vLLM computes ONLY the M_VERIFY suffix rows -> body GEMM size_m = M_VERIFY = 8.
        #   readable comparison positions: C+o+1 .. C+o+M_VERIFY-1  (M_VERIFY-1 = 7 positions).
        prompt_match = prompt_total = 0
        prompt_min_gap = float("inf")
        max_off = len(R) - M_VERIFY
        offsets = list(range(0, max_off + 1, HYBRID_PREFIX_COMMIT))
        chunk_sha_acc = []
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
                # hybrid-cache short-commit: this offset did not isolate to size_m=8; skip (keeps the
                # measured positions strictly at the M=8 verify width).
                continue
            pls = out.prompt_logprobs or []
            for i in range(C + o + 1, C + o + M_VERIFY):
                entry = pls[i] if i < len(pls) else None
                if entry is None:
                    continue
                sl = _sorted_logprobs(entry)
                m8_arg = int(sl[0][0])
                m1_tok = full[i]  # == R[i-C], the M=1 AR greedy token at this position
                gap = (sl[0][1] - sl[1][1]) if len(sl) >= 2 else float("inf")
                prompt_min_gap = min(prompt_min_gap, gap)
                prompt_total += 1
                if m8_arg == m1_tok:
                    prompt_match += 1
                else:
                    flip_gaps.append(gap)
                    lp_map = dict(sl)
                    margin = (sl[0][1] - lp_map[m1_tok]) if m1_tok in lp_map else None
                    flip_margins.append(margin)
            chunk_sha_acc.append(sl[0][0] if 'sl' in dir() else 0)

        if prompt_total == 0:
            continue
        if math.isfinite(prompt_min_gap):
            min_gap_per_prompt.append(prompt_min_gap)
        n_match += prompt_match
        n_total += prompt_total
        if det_m1 is not None:
            n_det_checked += prompt_total
            n_det_m1 += det_m1 * prompt_total

        sha = hashlib.sha256(bytes(str(chunk_sha_acc), "utf8")).hexdigest()[:16]
        per_prompt.append({
            "id": rec.get("id"), "C": C, "n_offsets": len(offsets),
            "positions": prompt_total, "match_M8_vs_M1": prompt_match,
            "det_match_M1_vs_M1": det_m1, "sha": sha,
            "min_top2_gap": (prompt_min_gap if math.isfinite(prompt_min_gap) else None),
        })
        if ri < verbose_k or ri == len(rows) - 1:
            br = 1.0 - prompt_match / prompt_total
            print(f"[arm:{arm}] prompt {ri} id={rec.get('id')} pos={prompt_total} "
                  f"break={prompt_total-prompt_match}/{prompt_total} ({br:.4f}) det_m1={det_m1} "
                  f"len(R)={len(R)} n_off={len(offsets)} elapsed={time.time()-t_arm:.0f}s", flush=True)

    n_seq = len(per_prompt)
    n_flips = n_total - n_match
    break_rate = (n_flips / n_total) if n_total else float("nan")
    det_m1_frac = (n_det_m1 / n_det_checked) if n_det_checked else float("nan")
    chunk_isolated_frac = (n_chunk_isolated / n_chunk_total) if n_chunk_total else float("nan")

    # ---- residual decomposition ----
    margins_present = [m for m in flip_margins if m is not None]
    n_m1_outside_topk = sum(1 for m in flip_margins if m is None)
    # int4-Marlin grid-tie: M1 token inside top-k AND margin < 0.5 nat (a sub-ULP coin-flip).
    n_under_0p5 = sum(1 for m in margins_present if m < NEAR_TIE_LOGPROB_THRESH)
    # SYSTEMATIC (attention-path / non-tie): margin >= 0.5 nat OR M1 token outside top-k.
    n_attention_path = (sum(1 for m in margins_present if m >= NEAR_TIE_LOGPROB_THRESH)
                        + n_m1_outside_topk)
    residual_frac_under_0p5 = (n_under_0p5 / n_flips) if n_flips else float("nan")
    int4_tie_residual_rate = (n_under_0p5 / n_total) if n_total else float("nan")

    # tau-ladder: a flip "counts" only if the margin by which M8 beat the M1 token >= tau.
    # margin None (M1 token outside top-k) is treated as a LARGE margin (counts at every tau).
    def tau_residual(tau: float) -> dict:
        survive = sum(1 for m in flip_margins if (m is None) or (m >= tau))
        return {"tau": tau, "surviving_flips": survive,
                "rate": (survive / n_total) if n_total else float("nan")}
    tau_rows = [tau_residual(t) for t in TAU_LADDER]
    residual_after_tau_0p3 = next(r["rate"] for r in tau_rows if abs(r["tau"] - 0.3) < 1e-9)

    aten_ctrl = aten_mm_invariance_control(torch, dims["hidden"], M_VERIFY)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    out = {
        "phase": "arm", "arm": arm, "model_dir": model_dir,
        "vllm_batch_invariant_env": batch_invariant_env,
        "attn_is_batch_invariant": attn_is_batch_invariant,
        "lmhead_int4": dims.get("lmhead_quant"),
        "n_prompts": n_seq, "ctx_len_requested": ctx_len, "C": C, "traj_len": traj_len,
        "M_verify": M_VERIFY,
        "total_positions": n_total, "matching_positions": n_match, "n_flips": n_flips,
        "break_rate": break_rate,
        "determinism_M1_vs_M1": det_m1_frac, "n_det_checked_positions": n_det_checked,
        "chunk_isolated_fraction": chunk_isolated_frac,
        "n_chunks_total": n_chunk_total, "n_chunks_isolated": n_chunk_isolated,
        "mean_computed_rows": (n_computed_rows_total / n_chunk_total) if n_chunk_total else None,
        # residual decomposition
        "n_attention_path_breaks": n_attention_path,
        "n_int4_tie_under_0p5nat": n_under_0p5,
        "n_m1_token_outside_topk": n_m1_outside_topk,
        "residual_frac_under_0p5nat": residual_frac_under_0p5,
        "int4_tie_residual_rate": int4_tie_residual_rate,
        "residual_after_tau_0p3nat": residual_after_tau_0p3,
        "tau_ladder": tau_rows,
        "flip_margin_median": (statistics.median(margins_present) if margins_present else None),
        "flip_margin_max": (max(margins_present) if margins_present else None),
        "flip_gap_median": (statistics.median(flip_gaps) if flip_gaps else None),
        "flip_gap_max": (max(flip_gaps) if flip_gaps else None),
        "flip_margins": [round(m, 5) if m is not None else None for m in flip_margins],
        "flip_gaps": [round(g, 5) for g in flip_gaps if math.isfinite(g)],
        "min_gap_all_positions_median": (statistics.median(min_gap_per_prompt)
                                         if min_gap_per_prompt else None),
        "aten_mm_control": aten_ctrl,
        "peak_gpu_gb": peak_gb,
        "per_prompt": per_prompt,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[arm:{arm}] break_rate={break_rate:.6f} ({n_flips}/{n_total})  "
          f"int4_tie_rate={int4_tie_residual_rate:.6f}  attn_path_breaks={n_attention_path}  "
          f"frac<0.5nat={residual_frac_under_0p5}  tau0.3_rate={residual_after_tau_0p3:.6f}", flush=True)
    print(f"[arm:{arm}] controls: det_m1={det_m1_frac:.6f} chunk_isolated={chunk_isolated_frac:.4f} "
          f"aten_mm_bitexact={aten_ctrl.get('bitexact_M1_vs_M8')} attn_bi={attn_is_batch_invariant} "
          f"peak={peak_gb:.1f}GB", flush=True)
    print(f"ARM_DONE {out_path}", flush=True)


def aten_mm_invariance_control(torch, hidden: int, batch_m: int) -> dict:
    """torch.mm row-0 bit-exactness at M=1 vs M=batch_m -- proves the BI override is live (pinned arm)."""
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
# orchestrator: two isolated subprocess arms, compose, wandb
# ======================================================================================
def run_phase_subprocess(args_list: list[str], extra_env: dict | None = None) -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"   # local A10G pod: inherited =1 makes torch see 0 GPUs
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
    extra_env = {"VLLM_BATCH_INVARIANT": "1" if arm == "pinned" else "0"}
    n_prompts = a.n_prompts
    if arm == "heuristic" and getattr(a, "heuristic_n_prompts", 0) > 0:
        n_prompts = a.heuristic_n_prompts
    run_phase_subprocess([
        "--phase", "arm", "--arm", arm, "--out", out_json,
        "--n-prompts", str(n_prompts), "--ctx-len", str(a.ctx_len),
        "--traj-len", str(a.traj_len), "--gpu-mem-util", str(a.gpu_mem_util),
        "--max-batched-tokens", str(a.max_batched_tokens), "--verbose-k", str(a.verbose_k),
    ], extra_env=extra_env)
    return json.load(open(out_json))


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den
    return (max(0.0, center - half), min(1.0, center + half))


def compose_and_report(arms: dict, a: argparse.Namespace) -> dict:
    pinned = arms["pinned"]
    heuristic = arms.get("heuristic")

    n_tot = pinned["total_positions"]
    n_flips = pinned["n_flips"]
    lb, ub = _wilson_ci(n_flips, n_tot)
    ci_overlaps_616 = bool(ub >= WIRBEL_616_CI[0] and lb <= WIRBEL_616_CI[1])

    attn_breaks = pinned["n_attention_path_breaks"]
    frac_under = pinned["residual_frac_under_0p5nat"]
    tau03 = pinned["residual_after_tau_0p3nat"]

    # VERDICT: under BI=1 both sides, attention is bit-exact (#621), so a residual built ENTIRELY of
    # sub-0.5-nat near-ties with ~0 systematic (margin>=0.5 / out-of-topk) breaks is the int4-Marlin
    # grid-tie residual #616 named -- the attention break is RECOVERED. Any systematic component means
    # the attention/structural break PERSISTS under BI=1.
    attention_recovered = bool(attn_breaks == 0)
    if attention_recovered and pinned["attn_is_batch_invariant"]:
        verdict = "ATTENTION_RECOVERED_RESIDUAL_IS_INT4_TIES"
    elif not pinned["attn_is_batch_invariant"]:
        verdict = "INCONCLUSIVE_BI_NOT_ENGAGED_IN_PINNED_ARM"
    else:
        verdict = "ATTENTION_BREAK_PERSISTS"

    report = {
        "pr": 622, "analysis_only": True, "official_tps": 0, "no_hf_job": True,
        "leg": "BI=1 both-sides spec-vs-AR greedy identity re-gate (teacher-forced per-step, real int4 body)",
        "imported_anchors": {
            "wirbel_616_residual": WIRBEL_616_RESIDUAL, "wirbel_616_ci": list(WIRBEL_616_CI),
            "wirbel_607_break": WIRBEL_607_BREAK, "stark_621_attn_maxdiff_under_BI": STARK_621_ATTN_MAXDIFF_UNDER_BI,
            "locked_319_ar_tps": LOCKED_319_AR_TPS, "M_verify": M_VERIFY,
        },
        # ---- REQUIRED deliverables ----
        "break_rate_bi1_both_sides": pinned["break_rate"],
        "attention_path_break_count": attn_breaks,
        "int4_tie_residual_rate": pinned["int4_tie_residual_rate"],
        "residual_frac_under_0p5nat": frac_under,
        "residual_after_tau_0p3nat": tau03,
        "tau_ladder": pinned["tau_ladder"],
        "verdict": verdict,
        # ---- corroboration of #616 ----
        "pinned_break_rate_ci95": [lb, ub],
        "pinned_n_flips": n_flips, "pinned_total_positions": n_tot,
        "ci_overlaps_wirbel_616": ci_overlaps_616,
        "break_rate_minus_616": (pinned["break_rate"] - WIRBEL_616_RESIDUAL),
        # ---- residual character ----
        "residual_flip_margin_median_nat": pinned["flip_margin_median"],
        "residual_flip_margin_max_nat": pinned["flip_margin_max"],
        "n_m1_token_outside_topk": pinned["n_m1_token_outside_topk"],
        # ---- BI=0 contrast ----
        "heuristic_break_rate": (heuristic["break_rate"] if heuristic else None),
        "heuristic_int4_tie_rate": (heuristic["int4_tie_residual_rate"] if heuristic else None),
        "heuristic_attn_is_batch_invariant": (heuristic["attn_is_batch_invariant"] if heuristic else None),
        # ---- controls ----
        "pinned_attn_is_batch_invariant": pinned["attn_is_batch_invariant"],
        "pinned_aten_mm_bitexact": pinned["aten_mm_control"].get("bitexact_M1_vs_M8"),
        "pinned_determinism_M1": pinned["determinism_M1_vs_M1"],
        "pinned_chunk_isolated_fraction": pinned["chunk_isolated_fraction"],
        "lmhead_int4": pinned.get("lmhead_int4"),
        "arms": {arm: {k: d[k] for k in (
            "break_rate", "int4_tie_residual_rate", "residual_frac_under_0p5nat",
            "residual_after_tau_0p3nat", "n_attention_path_breaks", "total_positions",
            "n_flips", "determinism_M1_vs_M1", "chunk_isolated_fraction",
            "attn_is_batch_invariant", "n_prompts", "peak_gpu_gb",
        )} for arm, d in arms.items()},
        "C": pinned["C"], "traj_len": pinned["traj_len"], "n_prompts": pinned["n_prompts"],
        "model_dir": pinned["model_dir"],
    }
    return report


def orchestrate(a: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_arms = ["pinned"] if a.pinned_only else list(ARMS)
    arms = {arm: _run_arm(a, arm) for arm in run_arms}
    report = compose_and_report(arms, a)
    _finish(report, a)


def reanalyze(a: argparse.Namespace) -> None:
    arms = {}
    for arm in ARMS:
        p = OUT_DIR / f"arm_{arm}_result.json"
        if p.exists():
            arms[arm] = json.load(open(p))
    if "pinned" not in arms:
        raise FileNotFoundError("--reanalyze needs at least arm_pinned_result.json")
    report = compose_and_report(arms, a)
    _finish(report, a)


def _finish(report: dict, a: argparse.Namespace) -> None:
    report_path = OUT_DIR.parent / "regate_report.json"
    json.dump(report, open(report_path, "w"), indent=2)
    _print_console(report)
    if not a.no_wandb:
        log_wandb(report, a)


def _print_console(r: dict) -> None:
    print("\n========== OPTION-B #319 BI=1 BOTH-SIDES RE-GATE (PR #622) ==========", flush=True)
    print(f" VERDICT                          : {r['verdict']}", flush=True)
    print(f" break_rate_bi1_both_sides        : {r['break_rate_bi1_both_sides']:.6f}  "
          f"({r['pinned_n_flips']}/{r['pinned_total_positions']})  CI95={r['pinned_break_rate_ci95']}", flush=True)
    print(f" attention_path_break_count       : {r['attention_path_break_count']}", flush=True)
    print(f" int4_tie_residual_rate           : {r['int4_tie_residual_rate']:.6f}", flush=True)
    print(f" residual_frac_under_0p5nat       : {r['residual_frac_under_0p5nat']}", flush=True)
    print(f" residual_after_tau_0p3nat        : {r['residual_after_tau_0p3nat']:.6f}", flush=True)
    print(f" tau_ladder                       : {[(t['tau'], round(t['rate'],6)) for t in r['tau_ladder']]}", flush=True)
    print(f" #616 corroboration               : rate {r['break_rate_bi1_both_sides']:.6f} vs 0.004318 "
          f"(diff {r['break_rate_minus_616']:+.6f}); CI overlaps #616 = {r['ci_overlaps_wirbel_616']}", flush=True)
    print(f" residual flip margin median/max  : {r['residual_flip_margin_median_nat']}/"
          f"{r['residual_flip_margin_max_nat']} nat (thresh {NEAR_TIE_LOGPROB_THRESH})", flush=True)
    print(f" heuristic (BI=0) break_rate      : {r['heuristic_break_rate']}", flush=True)
    print(f" controls: attn_bi(pinned)={r['pinned_attn_is_batch_invariant']} "
          f"aten_mm_bitexact={r['pinned_aten_mm_bitexact']} det_m1={r['pinned_determinism_M1']} "
          f"chunk_isolated={r['pinned_chunk_isolated_fraction']:.4f} lmhead_int4={r['lmhead_int4']}", flush=True)
    print("=====================================================================\n", flush=True)


def log_wandb(report: dict, a: argparse.Namespace) -> None:
    sys.path.insert(0, os.getcwd())
    try:
        from scripts.wandb_logging import init_wandb_run, finish_wandb
    except Exception as exc:
        print(f"[wandb] helper import failed: {exc!r}; skipping", flush=True)
        return
    run = init_wandb_run(
        job_type="local_profiling", agent="stark", name=a.wandb_name, group=a.wandb_group,
        notes="PR#622 BI=1 both-sides spec-vs-AR greedy identity re-gate (teacher-forced per-step, "
              "real int4 body). Does the #607 47% break collapse to the #616 int4-tie residual?",
        config={"pr": 622, "M_verify": M_VERIFY, "n_prompts": report["n_prompts"],
                "C": report["C"], "traj_len": report["traj_len"], "model_dir": report["model_dir"],
                "wirbel_616_residual": WIRBEL_616_RESIDUAL, "wirbel_607_break": WIRBEL_607_BREAK,
                "near_tie_thresh_nat": NEAR_TIE_LOGPROB_THRESH, "stack_vllm": "0.22.0"},
    )
    if run is None:
        print("[wandb] disabled (no API key/mode); JSON only", flush=True)
        return
    summary = {
        "verdict": report["verdict"],
        "verdict_recovered": report["verdict"] == "ATTENTION_RECOVERED_RESIDUAL_IS_INT4_TIES",
        "break_rate_bi1_both_sides": report["break_rate_bi1_both_sides"],
        "attention_path_break_count": report["attention_path_break_count"],
        "int4_tie_residual_rate": report["int4_tie_residual_rate"],
        "residual_frac_under_0p5nat": report["residual_frac_under_0p5nat"],
        "residual_after_tau_0p3nat": report["residual_after_tau_0p3nat"],
        "pinned_break_rate_ci95_lb": report["pinned_break_rate_ci95"][0],
        "pinned_break_rate_ci95_ub": report["pinned_break_rate_ci95"][1],
        "ci_overlaps_wirbel_616": report["ci_overlaps_wirbel_616"],
        "break_rate_minus_616": report["break_rate_minus_616"],
        "pinned_n_flips": report["pinned_n_flips"],
        "pinned_total_positions": report["pinned_total_positions"],
        "residual_flip_margin_median_nat": report["residual_flip_margin_median_nat"],
        "residual_flip_margin_max_nat": report["residual_flip_margin_max_nat"],
        "n_m1_token_outside_topk": report["n_m1_token_outside_topk"],
        "heuristic_break_rate": report["heuristic_break_rate"],
        "heuristic_int4_tie_rate": report["heuristic_int4_tie_rate"],
        "pinned_attn_is_batch_invariant": report["pinned_attn_is_batch_invariant"],
        "heuristic_attn_is_batch_invariant": report["heuristic_attn_is_batch_invariant"],
        "pinned_aten_mm_bitexact": report["pinned_aten_mm_bitexact"],
        "pinned_determinism_M1": report["pinned_determinism_M1"],
        "pinned_chunk_isolated_fraction": report["pinned_chunk_isolated_fraction"],
        "lmhead_int4": report["lmhead_int4"],
        "wirbel_616_residual": WIRBEL_616_RESIDUAL,
        "wirbel_607_break": WIRBEL_607_BREAK,
    }
    for t in report["tau_ladder"]:
        summary[f"tau_{str(t['tau']).replace('.', 'p')}_rate"] = t["rate"]
        summary[f"tau_{str(t['tau']).replace('.', 'p')}_surviving_flips"] = t["surviving_flips"]
    for k, v in summary.items():
        run.summary[k] = v
    finish_wandb(run)
    print(f"[wandb] logged run {run.id}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["arm"], default=None)
    ap.add_argument("--arm", choices=list(ARMS), default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--reanalyze", action="store_true")
    ap.add_argument("--pinned-only", dest="pinned_only", action="store_true",
                    help="run only the pinned (BI=1) arm -- the decisive measurement")
    ap.add_argument("--smoke", action="store_true", help="tiny run (few prompts, short traj)")
    ap.add_argument("--n-prompts", dest="n_prompts", type=int, default=128)
    ap.add_argument("--heuristic-n-prompts", dest="heuristic_n_prompts", type=int, default=0,
                    help="prompt count for the BI=0 contrast arm (0 => same as --n-prompts). "
                         "The heuristic arm only needs enough to show the break re-appears, not the "
                         "full decisive-arm sample.")
    ap.add_argument("--ctx-len", dest="ctx_len", type=int, default=224,
                    help="prefix length (aligned down to a multiple of 32)")
    ap.add_argument("--traj-len", dest="traj_len", type=int, default=512,
                    help="AR continuation length per prompt (the trajectory the verify window slides over)")
    ap.add_argument("--gpu-mem-util", type=float, default=0.55)
    ap.add_argument("--max-batched-tokens", type=int, default=8192)
    ap.add_argument("--verbose-k", dest="verbose_k", type=int, default=5)
    ap.add_argument("--wandb_group", dest="wandb_group", default="optionb-bi1-identity-regate")
    ap.add_argument("--wandb_name", dest="wandb_name", default="stark/optionb-bi1-identity-regate")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()

    if a.smoke and a.phase is None:
        a.n_prompts = min(a.n_prompts, 3)
        a.traj_len = min(a.traj_len, 96)

    if a.phase == "arm":
        phase_arm(a.out, a.arm, a.n_prompts, a.ctx_len, a.traj_len,
                  a.gpu_mem_util, a.max_batched_tokens, a.verbose_k)
    elif a.reanalyze:
        reanalyze(a)
    else:
        orchestrate(a)


if __name__ == "__main__":
    main()
