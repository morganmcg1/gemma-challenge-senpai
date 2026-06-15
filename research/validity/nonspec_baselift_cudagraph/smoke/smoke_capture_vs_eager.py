#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""SMOKE (PR #371): does vLLM offline cleanly isolate CUDA-graph CAPTURE from
inductor FUSION on the int4 M=1 decode, and does capture preserve byte-exact
greedy token identity vs eager?

Three configs on the SAME engine/kernels (only the lever toggles):
  E (eager)            : enforce_eager=True            -> mode=NONE, cudagraph=NONE
  C (capture, NO fuse) : mode=0 (no inductor) + cudagraph_mode=FULL
  F (capture + fuse)   : mode=3 (VLLM_COMPILE inductor) + cudagraph_mode=FULL

Decisive smoke check: token_ids(C) == token_ids(E) for every prompt (capture is
numerically identity-safe). F is the kanna #359 bundle (expected to be the
identity-breaker if any). Tiny + fast; the full deliverable scales this up.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

# Determinism preamble (must precede CUDA init). int4 body is Marlin (custom CUDA,
# unaffected) but the bf16 tied lm_head/attn use cuBLAS -> pin the workspace so the
# algo/workspace cannot be re-selected between eager and capture (byte-exact gate).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# PyTorch-native sampler: this container ships cuRAND headers only in the pip
# nvidia-cu13 package, so vLLM's default FlashInfer sampler JIT-build dies on
# `curand.h: No such file or directory`. The sampler does NOT touch logits, so
# greedy argmax identity is unchanged (matches scripts.local_validation.paths).
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def build_prompts(n: int) -> list[str]:
    base = [
        "Explain in one paragraph why the sky appears blue during the day.",
        "Write a short function in Python that returns the nth Fibonacci number.",
        "List three differences between TCP and UDP networking protocols.",
        "Summarize the plot of Romeo and Juliet in three sentences.",
        "What is the capital of France, and name two famous landmarks there?",
        "Describe how a hash map achieves average O(1) lookup time.",
        "Give a recipe outline for a simple vegetable soup.",
        "Explain the difference between supervised and unsupervised learning.",
    ]
    out = []
    i = 0
    while len(out) < n:
        out.append(base[i % len(base)])
        i += 1
    return out[:n]


def run_config(name: str, model: str, *, enforce_eager: bool, comp_cfg,
               prompts: list[str], max_tokens: int, max_model_len: int,
               gpu_mem: float):
    from vllm import LLM, SamplingParams
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    except Exception:  # noqa: BLE001
        pass

    kwargs = dict(
        model=model,
        tokenizer=model,
        dtype="bfloat16",
        max_model_len=max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=gpu_mem,
        enforce_eager=enforce_eager,
        disable_log_stats=True,
        enable_prefix_caching=False,
        trust_remote_code=True,
    )
    if comp_cfg is not None:
        kwargs["compilation_config"] = comp_cfg

    t_load = time.time()
    llm = LLM(**kwargs)
    load_s = time.time() - t_load

    # Inspect the RESOLVED compilation config (what vLLM actually did).
    resolved = {}
    try:
        cc = llm.llm_engine.vllm_config.compilation_config
        resolved = {
            "mode": getattr(getattr(cc, "mode", None), "value", getattr(cc, "mode", None)),
            "cudagraph_mode": str(getattr(cc, "cudagraph_mode", None)),
            "backend": getattr(cc, "backend", None),
        }
    except Exception as exc:  # noqa: BLE001
        resolved = {"error": repr(exc)}

    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens,
                        ignore_eos=True, seed=None)

    # Warmup (capture / compile happens here for C/F).
    _ = llm.generate(prompts[:1], sp, use_tqdm=False)

    t0 = time.time()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    elapsed = time.time() - t0

    token_ids = {}
    total_out = 0
    for o in outs:
        ids = list(o.outputs[0].token_ids)
        token_ids[o.prompt] = ids
        total_out += len(ids)
    tps = total_out / elapsed if elapsed > 0 else 0.0

    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass

    print(f"[{name}] load={load_s:.1f}s gen={elapsed:.2f}s out_tok={total_out} "
          f"wall_tps~={tps:.2f} resolved={resolved}", flush=True)
    return {"name": name, "resolved": resolved, "wall_tps": tps,
            "load_s": load_s, "gen_s": elapsed, "total_out": total_out,
            "token_ids": token_ids}


def identity_rate(ref: dict, cand: dict) -> dict:
    rt = ref["token_ids"]
    ct = cand["token_ids"]
    n = len(rt)
    matches = 0
    first_div = []
    for p, ids in rt.items():
        cids = ct.get(p, [])
        if cids == ids:
            matches += 1
        else:
            d = next((i for i, (a, b) in enumerate(zip(ids, cids)) if a != b),
                     min(len(ids), len(cids)))
            first_div.append(d)
    return {"rate": matches / n if n else 0.0, "matches": matches, "n": n,
            "first_div_positions": first_div}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/senpai-run/home/student-ubel/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0")
    ap.add_argument("--n-prompts", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.88)
    ap.add_argument("--configs", default="E,C,F")
    ap.add_argument("--out", default=str(Path(__file__).parent / "smoke_results.json"))
    args = ap.parse_args()

    prompts = build_prompts(args.n_prompts)
    want = [c.strip() for c in args.configs.split(",") if c.strip()]
    results = {}

    if "E" in want:
        results["E"] = run_config(
            "E-eager", args.model, enforce_eager=True, comp_cfg=None,
            prompts=prompts, max_tokens=args.max_tokens,
            max_model_len=args.max_model_len, gpu_mem=args.gpu_mem)
    if "C" in want:
        results["C"] = run_config(
            "C-capture-nofuse", args.model, enforce_eager=False,
            comp_cfg={"mode": 0, "cudagraph_mode": "FULL"},
            prompts=prompts, max_tokens=args.max_tokens,
            max_model_len=args.max_model_len, gpu_mem=args.gpu_mem)
    if "F" in want:
        results["F"] = run_config(
            "F-capture-fuse", args.model, enforce_eager=False,
            comp_cfg={"mode": 3, "cudagraph_mode": "FULL"},
            prompts=prompts, max_tokens=args.max_tokens,
            max_model_len=args.max_model_len, gpu_mem=args.gpu_mem)

    summary = {}
    if "E" in results and "C" in results:
        summary["C_vs_E_identity"] = identity_rate(results["E"], results["C"])
    if "E" in results and "F" in results:
        summary["F_vs_E_identity"] = identity_rate(results["E"], results["F"])

    print("\n=== SMOKE SUMMARY ===", flush=True)
    for k, v in summary.items():
        print(f"  {k}: rate={v['rate']:.4f} ({v['matches']}/{v['n']}) "
              f"first_div={v['first_div_positions']}", flush=True)
    for k in ("E", "C", "F"):
        if k in results:
            r = results[k]
            print(f"  {r['name']}: wall_tps~={r['wall_tps']:.2f} resolved={r['resolved']}",
                  flush=True)

    # strip token_ids from the saved blob (keep it small) but keep lengths.
    save = {"summary": summary, "configs": {
        k: {kk: vv for kk, vv in r.items() if kk != "token_ids"}
        for k, r in results.items()}}
    Path(args.out).write_text(json.dumps(save, indent=2))
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
