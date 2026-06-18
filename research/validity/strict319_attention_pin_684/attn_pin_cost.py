#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""PR #684 (land) -- served TPS cost of pinning the Triton split-KV attention
reduction, and whether the pin makes the M=K+1 spec-dec verify byte-identical to
the M=1 AR (so strict-#319 holds with NO recompute-rescue).

MECHANISM (vLLM 0.22.0, grounded in source -- triton_unified_attention.py:918-932):
  The Gemma4 model is forced onto the TRITON_ATTN backend (heterogeneous head dims).
  Its decode attention picks a 2D vs 3D softmax-reduction path:
      use_3d = not ( ... or max_seqlen_q > 1 or num_seqs > seq_threshold_3D
                         or is_batch_invariant )
  * M=1 AR decode (max_seqlen_q=1, 1 seq, below threshold) -> 3D SEGMENTED path
    (NUM_PAR_SOFTMAX_SEGMENTS=16 + reduce_segments float32 reduction).
  * M=6 verify (max_seqlen_q=6 > 1)                         -> forced 2D path.
  The 2D vs 3D float reductions differ -> that is #680's 90% bit-diff / argmax flips.
  Pinning the attention == forcing the M=1 AR onto the SAME 2D path as the verify:
      (b) VLLM_BATCH_INVARIANT=1  -> is_batch_invariant -> 2D  (+ aten-op swaps).
      (c) seq_threshold_3D = 0     -> num_seqs(1) > 0     -> 2D  (pure attn pin; no swaps).
  My #680 already proved the int4 g=128 Marlin GEMM is byte-identical across M, so if
  the pinned (2D) M=6 verify is byte-identical to the pinned (2D) M=1 AR, the verify is
  lossless by construction and strict-#319 needs zero recompute-rescue.

This script measures ONE --config in a fresh process (BI / monkeypatch are process-global
and snapshotted at import, so they must not be mixed):
  1. M=1 decode step latency (slope method -> cancels prefill/fixed overhead).
  2. M=6 verify per-token forward latency (secondary, faithful cross-check).
  3. #680 Leg-B identity: M=6 verify argmax vs M=1 AR argmax (flip + bit-diff rates).
  4. Ground-truth 2D/3D path log (wraps unified_attention) to PROVE the pin took.

analysis_only, LOCAL, full-vocab QAT ckpt (the deployed pruned-16k-head can't load in
vanilla vLLM; identical fidelity caveat to #680/#491 -- the attention M-dependence is a
kernel-occupancy property independent of lm_head vocab size). No HF Job, no served-file
change. Run with the vLLM 0.22.0 venv: /tmp/senpai-venvs/20f658587e8a6643/bin/python
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
CENSUS_DIR = ROOT / "research" / "validity" / "reduction_sensitivity_census"
sys.path.insert(0, str(CENSUS_DIR))

# Native sampler: curand dev headers absent in this venv; greedy argmax unaffected.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

# Direct attention-op GPU timer (CUDA events around the wrapped unified_attention).
# This isolates the 2D-vs-3D reduction cost WITHOUT the lm_head GEMM that dilutes the
# full-step slope on the full-vocab head -> the delta is head-INDEPENDENT absolute ms.
_ATTN_TIMER: dict[str, Any] = {"enabled": False, "events": [], "n": 0}


def _setup_config(config: str) -> dict[str, Any]:
    """Set process-global pin BEFORE importing vllm. Returns provenance dict.

    Force the IN-PROCESS engine (VLLM_ENABLE_V1_MULTIPROCESSING=0 -> InprocClient) so
    the fixed2d monkeypatch + the 2D/3D path probe run in the SAME process that does the
    forward (the default background EngineCore subprocess would not see them). Identical
    setting across all 3 configs -> relative timing stays apples-to-apples."""
    prov: dict[str, Any] = {"config": config}
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if config == "bi1":
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    else:
        os.environ.setdefault("VLLM_BATCH_INVARIANT", "0")
    prov["env_VLLM_BATCH_INVARIANT"] = os.environ.get("VLLM_BATCH_INVARIANT")
    prov["env_VLLM_ENABLE_V1_MULTIPROCESSING"] = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
    return prov


def _apply_fixed2d_patch(prov: dict[str, Any]) -> None:
    """config==fixed2d: force seq_threshold_3D=0 so M=1 decode takes the 2D path,
    matching the M>1 verify -- a pure attention pin (no aten-op swap). Python-level
    monkeypatch of a module constant; no CUDA rebuild => config-reachable."""
    import vllm.v1.attention.backends.triton_attn as ta

    prov["MIN_LAUNCH_GRID_SIZE_2D_before"] = ta.MIN_LAUNCH_GRID_SIZE_2D
    ta.MIN_LAUNCH_GRID_SIZE_2D = 0  # 0 // num_heads_kv == 0 -> threshold 0 -> 1 > 0 -> 2D
    prov["MIN_LAUNCH_GRID_SIZE_2D_after"] = ta.MIN_LAUNCH_GRID_SIZE_2D


def _install_path_probe(path_log: list[dict]) -> None:
    """Wrap the unified_attention entry triton_attn calls, to record the GROUND-TRUTH
    2D/3D decision per launch (proves the pin actually changed the M=1 path).

    Cap PER max_seqlen_q BUCKET (m1 vs m>1), not globally: with max_num_batched_tokens=M
    the context prefill is CHUNKED into M-token chunks (each max_seqlen_q=M>1), which
    would otherwise exhaust a single global budget before any real M=1 decode launch is
    seen -> the M=1 bucket would come back empty."""
    import vllm.v1.attention.backends.triton_attn as ta_backend
    import vllm.v1.attention.ops.triton_unified_attention as tua

    orig = ta_backend.unified_attention
    cap_per_bucket = 48
    counts = {"m1": 0, "mgt1": 0}

    def _wrapped(*args, **kwargs):
        msq = kwargs.get("max_seqlen_q")
        if msq is not None:
            m = int(msq)
            bucket = "m1" if m == 1 else "mgt1"
            if counts[bucket] < cap_per_bucket:
                st3 = kwargs.get("seq_threshold_3D")
                nseg = kwargs.get("num_par_softmax_segments")
                cusq = kwargs.get("cu_seqlens_q")
                som = kwargs.get("softmax_segm_output")
                nseq = (cusq.shape[0] - 1) if cusq is not None else None
                is_bi = bool(tua.is_batch_invariant)
                use_3d = not (
                    st3 is None or nseg is None or som is None
                    or m > 1
                    or (nseq is not None and nseq > st3)
                    or is_bi
                )
                path_log.append({
                    "max_seqlen_q": m,
                    "num_seqs": int(nseq) if nseq is not None else None,
                    "seq_threshold_3D": int(st3) if st3 is not None else None,
                    "is_batch_invariant": is_bi,
                    "use_3d": bool(use_3d),
                })
                counts[bucket] += 1
            # Direct GPU timing of the M=1 decode attention op (segmented kernel +
            # reduce_segments for 3D; single kernel for 2D -- both inside orig()).
            if _ATTN_TIMER["enabled"] and m == 1:
                import torch
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                r = orig(*args, **kwargs)
                e.record()
                _ATTN_TIMER["events"].append((s, e))
                _ATTN_TIMER["n"] += 1
                return r
        return orig(*args, **kwargs)

    ta_backend.unified_attention = _wrapped


def _measure_attn_step_ms(llm, sp_cls, ctx_ids, *, n_tokens, warmup, reps):
    """Per-decode-step attention GPU-time (summed over layers) via CUDA events on the
    wrapped unified_attention. The CONFIG-to-CONFIG delta of this == the pure 2D-vs-3D
    attention-reduction cost in ABSOLUTE ms (lm_head/body cancel; head-independent).
    Returns (median_ms_per_step, samples, layers_per_step)."""
    import torch
    base = {"prompt_token_ids": ctx_ids}
    llm.generate([base], sp_cls(temperature=0.0, max_tokens=warmup, ignore_eos=True),
                 use_tqdm=False)  # warmup (compile/JIT)
    torch.cuda.synchronize()
    samples: list[float] = []
    layers_seen: list[float] = []
    for _ in range(reps):
        _ATTN_TIMER["events"].clear(); _ATTN_TIMER["n"] = 0
        _ATTN_TIMER["enabled"] = True
        llm.generate([base], sp_cls(temperature=0.0, max_tokens=n_tokens, ignore_eos=True),
                     use_tqdm=False)
        _ATTN_TIMER["enabled"] = False
        torch.cuda.synchronize()
        total_ms = sum(s.elapsed_time(e) for s, e in _ATTN_TIMER["events"])
        n_launch = _ATTN_TIMER["n"]
        if n_tokens:
            samples.append(total_ms / n_tokens)
            layers_seen.append(n_launch / n_tokens)
    _ATTN_TIMER["events"].clear()
    samples.sort()
    med = samples[len(samples) // 2] if samples else float("nan")
    layers = layers_seen[0] if layers_seen else float("nan")
    return med, samples, layers


def _measure_decode_step_ms(llm, sp_cls, ctx_ids, *, warmup, n_long, n_short, reps):
    """M=1 greedy decode step latency via two-length slope (cancels prefill+overhead).
    Returns (median_step_ms, raw_samples)."""
    base = {"prompt_token_ids": ctx_ids}
    sp_long = sp_cls(temperature=0.0, top_p=1.0, max_tokens=n_long, ignore_eos=True)
    sp_short = sp_cls(temperature=0.0, top_p=1.0, max_tokens=n_short, ignore_eos=True)
    sp_warm = sp_cls(temperature=0.0, top_p=1.0, max_tokens=warmup, ignore_eos=True)

    import torch
    llm.generate([base], sp_warm, use_tqdm=False)  # warmup (compile/JIT all shapes)
    torch.cuda.synchronize()

    samples = []
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.perf_counter()
        llm.generate([base], sp_short, use_tqdm=False)
        torch.cuda.synchronize(); t_short = time.perf_counter() - t

        torch.cuda.synchronize(); t = time.perf_counter()
        llm.generate([base], sp_long, use_tqdm=False)
        torch.cuda.synchronize(); t_long = time.perf_counter() - t

        step_ms = 1000.0 * (t_long - t_short) / (n_long - n_short)
        samples.append(step_ms)
    samples.sort()
    return samples[len(samples) // 2], samples


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, choices=["baseline", "bi1", "fixed2d"])
    ap.add_argument("--verify-width", type=int, default=6, help="M=K+1 (PR K=5 -> 6)")
    ap.add_argument("--n-prompts", type=int, default=48)
    ap.add_argument("--n-new", type=int, default=32)
    ap.add_argument("--ctx-cap", type=int, default=512)
    ap.add_argument("--det-prompts", type=int, default=12)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--near-tie", type=float, default=0.5)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    # TPS slope-timing knobs
    ap.add_argument("--tps-warmup", type=int, default=32)
    ap.add_argument("--tps-long", type=int, default=128)
    ap.add_argument("--tps-short", type=int, default=16)
    ap.add_argument("--tps-reps", type=int, default=5)
    ap.add_argument("--tps-ctx-prompts", type=int, default=2)
    # direct attention-op timing knobs (the head-independent pin-cost measurement)
    ap.add_argument("--attn-tokens", type=int, default=64)
    ap.add_argument("--attn-warmup", type=int, default=16)
    ap.add_argument("--attn-reps", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    prov = _setup_config(args.config)

    import torch
    from vllm import LLM, SamplingParams
    from reduction_sensitivity_census import (  # noqa: E402
        load_prompts, resolve_model_dir, _margin_model_full_vocab,
        entry_as_dict, top1_top2_margin,
    )

    if args.config == "fixed2d":
        _apply_fixed2d_patch(prov)
    path_log: list[dict] = []
    _install_path_probe(path_log)

    model_dir = resolve_model_dir()
    full_vocab = _margin_model_full_vocab(model_dir)
    prompts = load_prompts(args.n_prompts, args.ctx_cap)
    M = args.verify_width
    print(f"[684:{args.config}] model={model_dir} full_vocab={full_vocab} "
          f"prompts={len(prompts)} M={M} prov={prov}", flush=True)

    llm = LLM(model=model_dir, quantization="compressed-tensors", dtype="bfloat16",
              max_model_len=max(1024, args.ctx_cap + args.n_new + 16),
              gpu_memory_utilization=args.gpu_mem_util, max_num_seqs=1,
              max_num_batched_tokens=M, enable_prefix_caching=False,
              enforce_eager=True, trust_remote_code=True,
              max_logprobs=max(20, args.topk + 2))

    # ---- 1. M=1 decode step latency (slope) over a few prompt contexts ----
    decode_step_samples: list[float] = []
    for pi in range(min(args.tps_ctx_prompts, len(prompts))):
        med, samples = _measure_decode_step_ms(
            llm, SamplingParams, prompts[pi]["context_token_ids"],
            warmup=args.tps_warmup, n_long=args.tps_long,
            n_short=args.tps_short, reps=args.tps_reps)
        decode_step_samples.append(med)
        print(f"[684:{args.config}] decode-step ctx#{pi} median={med:.4f} ms "
              f"samples={[round(s,4) for s in samples]}", flush=True)
    decode_step_samples.sort()
    decode_step_ms = decode_step_samples[len(decode_step_samples) // 2]
    decode_tps_local = 1000.0 / decode_step_ms

    # ---- 1b. DIRECT attention-op step time (head-independent pin cost) ----
    attn_step_samples: list[float] = []
    attn_layers_per_step = float("nan")
    for pi in range(min(args.tps_ctx_prompts, len(prompts))):
        med_a, samp_a, layers_a = _measure_attn_step_ms(
            llm, SamplingParams, prompts[pi]["context_token_ids"],
            n_tokens=args.attn_tokens, warmup=args.attn_warmup, reps=args.attn_reps)
        attn_step_samples.append(med_a)
        attn_layers_per_step = layers_a
        print(f"[684:{args.config}] attn-step ctx#{pi} median={med_a:.5f} ms/step "
              f"layers/step={layers_a:.1f} samples={[round(s,5) for s in samp_a]}", flush=True)
    attn_step_samples.sort()
    attn_step_ms = attn_step_samples[len(attn_step_samples) // 2] if attn_step_samples else float("nan")

    # ---- 2 + 3. M=6 verify timing + #680 Leg-B identity ----
    gen_sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.n_new, logprobs=args.topk)
    ver_sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=args.topk)

    n_pos = n_flip = n_bitdiff = 0
    nt_pos = nt_flip = 0
    n_det_gen = det_gen_positions = 0
    n_prompts_with_flip = 0
    flip_examples: list[dict] = []
    verify_tok_time = 0.0
    verify_tok_count = 0
    any_nan = False
    t0 = time.time()

    for pi, pr in enumerate(prompts):
        ctx = pr["context_token_ids"]
        c = len(ctx)
        base = {"prompt_token_ids": ctx}

        out = llm.generate([base], gen_sp, use_tqdm=False)[0]
        gen = list(out.outputs[0].token_ids)
        gen_lps = out.outputs[0].logprobs or []
        if not gen:
            continue

        if pi < args.det_prompts:
            gen_b = list(llm.generate([base], gen_sp, use_tqdm=False)[0].outputs[0].token_ids)
            Lg = min(len(gen), len(gen_b))
            n_det_gen += sum(1 for a, b in zip(gen[:Lg], gen_b[:Lg]) if a == b)
            det_gen_positions += Lg

        full = ctx + gen
        torch.cuda.synchronize(); tv = time.perf_counter()
        vout = llm.generate([{"prompt_token_ids": full}], ver_sp, use_tqdm=False)[0]
        torch.cuda.synchronize(); dt = time.perf_counter() - tv
        verify_tok_time += dt
        verify_tok_count += len(full)

        pls = vout.prompt_logprobs
        prompt_flipped = False
        for g in range(len(gen)):
            j = c + g
            if pls is None or j >= len(pls) or pls[j] is None:
                continue
            ref_tok = int(gen[g])
            mv = entry_as_dict(pls[j])
            if ref_tok not in mv:
                mv[ref_tok] = float("-inf")
            v_arg, v_top_lp, _v_margin = top1_top2_margin(mv)
            m1 = entry_as_dict(gen_lps[g]) if g < len(gen_lps) else {}
            _m1_arg, m1_top_lp, m1_margin = (top1_top2_margin(m1) if m1
                                             else (ref_tok, float("nan"), float("nan")))
            n_pos += 1
            is_near_tie = bool(math.isfinite(m1_margin) and m1_margin <= args.near_tie)
            if is_near_tie:
                nt_pos += 1
            if (math.isfinite(v_top_lp) and math.isfinite(m1_top_lp) and v_top_lp != m1_top_lp):
                n_bitdiff += 1
            if v_arg != ref_tok:
                n_flip += 1
                prompt_flipped = True
                if is_near_tie:
                    nt_flip += 1
                if len(flip_examples) < 40:
                    flip_examples.append({"prompt_index": pi, "absolute_position": j,
                                          "ref_tok": ref_tok, "verify_argmax": v_arg,
                                          "m1_margin": m1_margin, "near_tie": is_near_tie})
            any_nan = any_nan or bool(not math.isfinite(v_top_lp))
        if prompt_flipped:
            n_prompts_with_flip += 1

    frac_flip = (n_flip / n_pos) if n_pos else float("nan")
    frac_bitdiff = (n_bitdiff / n_pos) if n_pos else float("nan")
    nt_flip_rate = (nt_flip / nt_pos) if nt_pos else float("nan")
    seq_break_rate = (n_prompts_with_flip / len(prompts)) if prompts else float("nan")
    ar_vs_ar = (n_det_gen / det_gen_positions) if det_gen_positions else float("nan")
    verify_tok_ms = 1000.0 * verify_tok_time / verify_tok_count if verify_tok_count else float("nan")

    # path-decision summary: what did M=1 decode vs M>1 verify actually do?
    m1_paths = [p for p in path_log if p["max_seqlen_q"] == 1]
    mv_paths = [p for p in path_log if p["max_seqlen_q"] > 1]
    m1_use_3d = sorted({p["use_3d"] for p in m1_paths})
    mv_use_3d = sorted({p["use_3d"] for p in mv_paths})

    is_lossless = bool(n_pos > 0 and n_flip == 0)
    is_bitexact = bool(n_pos > 0 and n_bitdiff == 0)

    result = {
        "phase": "attn_pin_cost", "config": args.config, "provenance": prov,
        "model_dir": model_dir, "margin_model_full_vocab": full_vocab,
        "verify_width": M, "n_prompts": len(prompts), "n_new": args.n_new, "topk": args.topk,
        # ---- TPS / step latency ----
        "decode_step_ms": decode_step_ms, "decode_step_samples_med_per_ctx": decode_step_samples,
        "decode_tps_local": decode_tps_local,
        "verify_per_token_ms": verify_tok_ms,
        # ---- DIRECT attention-op step time (head-independent pin-cost lever) ----
        "attn_step_ms": attn_step_ms, "attn_step_samples_med_per_ctx": attn_step_samples,
        "attn_layers_per_step": attn_layers_per_step,
        # ---- identity (Leg-B) ----
        "n_positions": n_pos, "n_flip": n_flip, "n_bitdiff": n_bitdiff,
        "fullforward_frac_steps_argmax_break": frac_flip,
        "fullforward_frac_steps_bitdiff": frac_bitdiff,
        "fullforward_seq_break_rate": seq_break_rate,
        "n_prompts_with_flip": n_prompts_with_flip,
        "near_tie_n_positions": nt_pos, "near_tie_n_flip": nt_flip, "near_tie_break_rate": nt_flip_rate,
        "ar_vs_ar_token_identity": ar_vs_ar, "ar_vs_ar_positions": det_gen_positions,
        "is_lossless_argmax": is_lossless, "is_bitexact_logprob": is_bitexact,
        "flip_examples": flip_examples, "any_nan": bool(any_nan),
        # ---- ground-truth path decision ----
        "path_m1_use_3d_values": m1_use_3d, "path_verify_use_3d_values": mv_use_3d,
        "path_log_head": path_log[:12],
        "peak_mem_mib": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = args.out or (HERE / "runs" / f"{args.config}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out_path, "w"), indent=2, default=str)
    print("\n" + "=" * 70, flush=True)
    print(f"[684:{args.config}] decode_step={decode_step_ms:.4f}ms "
          f"(local TPS {decode_tps_local:.2f}) verify/tok={verify_tok_ms:.4f}ms", flush=True)
    print(f"[684:{args.config}] attn_step={attn_step_ms:.5f}ms/step "
          f"(layers/step={attn_layers_per_step:.1f})  <-- head-independent pin lever", flush=True)
    print(f"[684:{args.config}] M=1 use_3d={m1_use_3d}  M>1 verify use_3d={mv_use_3d}", flush=True)
    print(f"[684:{args.config}] argmax_break={frac_flip:.6f} ({n_flip}/{n_pos})  "
          f"bitdiff={frac_bitdiff:.4f}  LOSSLESS={is_lossless} BITEXACT={is_bitexact}", flush=True)
    print(f"[684:{args.config}] AR-vs-AR={ar_vs_ar:.6f}  -> {out_path}", flush=True)
    print("=" * 70, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
