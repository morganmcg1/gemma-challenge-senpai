#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""PR #680 (land) Leg B -- full-forward width isolation: does the spec-dec verify
break SURVIVE a byte-identical GEMM? (i.e. is the source attention, not Marlin?)

Leg A (gemm_width_microbench.py) proved the deployed g=128 int4 Marlin GEMMs (qkv,
o, gate_up, down, lm_head) are BYTE-IDENTICAL across batch width M (M=1 AR vs M=6
verify; maxdiff=0) under EVERY reduction-order knob (split-K/atomic on/off, fp32/fp16
reduce). So the GEMM cannot flip any argmax. If the full-forward width-(K+1) verify
STILL breaks greedy identity vs the width-1 AR -- which kanna #673 reports at
0.33-0.38 seq -- the break MUST come from the one remaining M-dependent reduction:
the flash split-KV ATTENTION reduction (occupancy differs M=1 AR vs M=6 verify).

This is the end-to-end confirmation of the refutation. Method = ubel #491's validated
phase_margin geometry, re-pointed to M=VERIFY_WIDTH (PR #680 K=5 -> M=6):
  * REF: greedy AR generate (M=1 decode) -> reference tokens + top-K M=1 logprobs.
  * VERIFY: width-M re-forward of [ctx+gen] (max_num_batched_tokens=M -> the M=K+1
    chunked-prefill occupancy) -> top-K M-occupancy logprobs at the SAME positions.
  * break = (argmax_verify != ref_tok). Per-position flip rate + per-prompt seq break.
  * AR-vs-AR determinism control (instr 4): re-gen REF -> must be byte-identical.
  * near-tie subset = positions with M=1 top1-top2 margin <= NEAR_TIE (the "#673
    break-set" analog, reconstructed from the IN-SCOPE public STEM prompts, NOT from
    #673's out-of-scope branch).

FIDELITY (honest, per the PR): the loadable full-vocab int4 ckpt is the upstream QAT
checkpoint (the deployed int4_g128_lmhead has a PRUNED 16384-row lm_head that vanilla
vLLM cannot load -- vocab-shape assert; identical to the #491 caveat). The
load-bearing claim (GEMM byte-invariance across M) was measured at the EXACT g=128
deployed shapes in Leg A; attention's M-dependence is a kernel-occupancy property
independent of weight quant granularity, so the full-vocab ckpt is faithful for the
attention-attribution question. analysis_only, LOCAL, no served-file change, no HF Job.
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
# reuse ubel #491's validated helpers (on-branch, in scope)
CENSUS_DIR = ROOT / "research" / "validity" / "reduction_sensitivity_census"
sys.path.insert(0, str(CENSUS_DIR))

# The flashinfer sampler JIT-builds a curand kernel at startup; the CUDA dev headers
# (curand.h) are absent in this venv, so route sampling to the native path (greedy
# argmax is unaffected). Mirrors the run_identity/selfdet harness env. Must be set
# before vLLM imports anything.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify-width", type=int, default=6, help="M=K+1 verify occupancy (PR #680 K=5 -> 6)")
    ap.add_argument("--n-prompts", type=int, default=64)
    ap.add_argument("--n-new", type=int, default=32)
    ap.add_argument("--ctx-cap", type=int, default=512)
    ap.add_argument("--det-prompts", type=int, default=12, help="# prompts with AR-vs-AR control")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--near-tie", type=float, default=0.5, help="M1 margin <= this == near-tie subset")
    ap.add_argument("--out", type=Path, default=HERE / "runs" / "verify_break_fullforward.json")
    args = ap.parse_args()

    import torch
    from vllm import LLM, SamplingParams
    # validated helpers from the census module
    from reduction_sensitivity_census import (  # noqa: E402
        load_prompts, resolve_model_dir, _margin_model_full_vocab,
        entry_as_dict, top1_top2_margin,
    )

    model_dir = resolve_model_dir()
    full_vocab = _margin_model_full_vocab(model_dir)
    prompts = load_prompts(args.n_prompts, args.ctx_cap)
    M = args.verify_width
    print(f"[legB] model={model_dir} full_vocab={full_vocab} prompts={len(prompts)} "
          f"verify_width={M} n_new={args.n_new} topk={args.topk}", flush=True)

    llm = LLM(model=model_dir, quantization="compressed-tensors", dtype="bfloat16",
              max_model_len=max(1024, args.ctx_cap + args.n_new + 16),
              gpu_memory_utilization=args.gpu_mem_util, max_num_seqs=1,
              max_num_batched_tokens=M, enable_prefix_caching=False,
              enforce_eager=True, trust_remote_code=True,
              max_logprobs=max(20, args.topk + 2))

    gen_sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.n_new, logprobs=args.topk)
    ver_sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=args.topk)

    n_pos = n_flip = n_bitdiff = 0
    nt_pos = nt_flip = 0                       # near-tie subset
    n_det_gen = det_gen_positions = 0
    n_prompts_with_flip = 0
    margins_flip: list[float] = []
    flip_examples: list[dict] = []
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

        # AR-vs-AR determinism control (instr 4)
        if pi < args.det_prompts:
            gen_b = list(llm.generate([base], gen_sp, use_tqdm=False)[0].outputs[0].token_ids)
            Lg = min(len(gen), len(gen_b))
            n_det_gen += sum(1 for a, b in zip(gen[:Lg], gen_b[:Lg]) if a == b)
            det_gen_positions += Lg

        # width-M verify re-forward of [ctx+gen]
        full = ctx + gen
        vout = llm.generate([{"prompt_token_ids": full}], ver_sp, use_tqdm=False)[0]
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
            if math.isnan(m1_margin) and not m1:
                pass
            n_pos += 1
            is_near_tie = bool(math.isfinite(m1_margin) and m1_margin <= args.near_tie)
            if is_near_tie:
                nt_pos += 1
            if (math.isfinite(v_top_lp) and math.isfinite(m1_top_lp) and v_top_lp != m1_top_lp):
                n_bitdiff += 1
            if v_arg != ref_tok:
                n_flip += 1
                prompt_flipped = True
                margins_flip.append(m1_margin if math.isfinite(m1_margin) else float("nan"))
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
    mf = [m for m in margins_flip if math.isfinite(m)]

    out = {
        "phase": "verify_break_fullforward", "model_dir": model_dir,
        "margin_model_full_vocab": full_vocab, "verify_width": M,
        "n_prompts": len(prompts), "n_new": args.n_new, "topk": args.topk,
        "near_tie_threshold": args.near_tie,
        "n_positions": n_pos, "n_flip": n_flip, "n_bitdiff": n_bitdiff,
        # full-forward break (the PR instr-1 break_rate; GEMM is byte-identical so this is attention)
        "fullforward_frac_steps_argmax_break": frac_flip,
        "fullforward_frac_steps_bitdiff": frac_bitdiff,
        "fullforward_seq_break_rate": seq_break_rate,
        "n_prompts_with_flip": n_prompts_with_flip,
        # near-tie subset ("#673 break-set" analog reconstructed from public STEM prompts)
        "near_tie_n_positions": nt_pos, "near_tie_n_flip": nt_flip,
        "near_tie_break_rate": nt_flip_rate,
        # instr-4 linchpin
        "ar_vs_ar_token_identity": ar_vs_ar, "ar_vs_ar_positions": det_gen_positions,
        "max_m1_margin_at_break": (max(mf) if mf else float("nan")),
        "min_m1_margin_at_break": (min(mf) if mf else float("nan")),
        "flip_examples": flip_examples, "any_nan": bool(any_nan),
        "peak_mem_mib": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
        "elapsed_s": round(time.time() - t0, 1),
        "interpretation": (
            "Leg A: g=128 Marlin GEMMs byte-identical M=1 vs M=6 (maxdiff=0, all knobs). "
            "So a nonzero fullforward break here is NOT the GEMM -- it is the M-dependent "
            "flash split-KV attention reduction. AR-vs-AR identity confirms the M=1 reference "
            "is deterministic (the break is a width effect, not run-to-run noise)."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=str)
    print("\n" + "=" * 70, flush=True)
    print(f"[LEG B] verify_width={M}", flush=True)
    print(f"  fullforward break (per-position): {frac_flip:.5f}  ({n_flip}/{n_pos})", flush=True)
    print(f"  fullforward seq break-rate:       {seq_break_rate:.4f}  ({n_prompts_with_flip}/{len(prompts)})", flush=True)
    print(f"  near-tie subset break-rate:       {nt_flip_rate:.5f}  ({nt_flip}/{nt_pos})", flush=True)
    print(f"  AR-vs-AR token identity:          {ar_vs_ar:.6f}  ({n_det_gen}/{det_gen_positions})", flush=True)
    print(f"  result -> {args.out}", flush=True)
    print("=" * 70, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
