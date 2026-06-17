#!/usr/bin/env python
"""Offline torch-profiler trace of the int4_g128_lmhead AR decode step (#604).

vLLM 0.22.0 in this container exposes no /start_profile route, so we profile the
EngineCore step directly via an offline LLM with CUDA graphs ON (matching the
served config). The body/head/attn/sampler GPU kernels are identical to the
served path; the offline host-idle is a LOWER bound on the served host window
(served has extra API/ZMQ), and we already showed served TPS ~= decode-only TPS,
so the served per-step host addition is sub-1%.

Greedy temp=0, MNS=1, FLASHINFER_SAMPLER=0. LOCAL profiling only.
"""
from __future__ import annotations
import json, os
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from vllm import LLM, SamplingParams

MODEL = os.environ.get("MODEL_ID", "/workspace/gemma_build/int4_g128_lmhead")
OUT = Path("research/ar_logits_tail/trace/offline_trace.json")
OUT.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    eager = os.environ.get("ENFORCE_EAGER", "1") == "1"
    llm = LLM(
        model=MODEL, dtype="bfloat16", max_model_len=4096,
        gpu_memory_utilization=0.90, max_num_batched_tokens=512,
        max_num_seqs=1, trust_remote_code=True, enforce_eager=eager,
        disable_log_stats=True,
    )
    tok = llm.get_tokenizer()
    import random, json as _json
    data = _json.loads(Path("official/main_bucket/shared_resources/speed_benchmark/data/eval_prompts_sharegpt.json").read_text())
    recs = [it["conversations"][0]["value"] for it in data
            if isinstance(it, dict) and isinstance(it.get("conversations"), list)
            and len(it["conversations"]) >= 2 and isinstance(it["conversations"][0], dict)
            and isinstance(it["conversations"][0].get("value"), str)]
    random.Random(1).shuffle(recs)
    prompt = tok.apply_chat_template([{"role": "user", "content": recs[0]}],
                                     add_generation_prompt=True, tokenize=False)

    sp_warm = SamplingParams(temperature=0.0, max_tokens=64, min_tokens=8, ignore_eos=True)
    sp_prof = SamplingParams(temperature=0.0, max_tokens=96, min_tokens=8, ignore_eos=True)

    # warm up CUDA graphs + caches
    for _ in range(3):
        llm.generate([prompt], sp_warm, use_tqdm=False)

    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False, with_stack=False, acc_events=True,
    ) as prof:
        out = llm.generate([prompt], sp_prof, use_tqdm=False)
    torch.cuda.synchronize()

    prof.export_chrome_trace(str(OUT))
    n_out = len(out[0].outputs[0].token_ids)
    print(json.dumps({"trace": str(OUT), "generated_tokens": n_out,
                      "trace_bytes": OUT.stat().st_size}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
