#!/usr/bin/env python3
"""Base greedy decode with per-step argmax token-id capture (PR #528, GPU step).

Loads the FULL bf16 base model google/gemma-4-E4B-it (the 0.668/0.444 anchor arm,
NOT the int4/osoi5/pruned ship), applies the gemma chat template exactly as the
vLLM OpenAI server does, and greedily decodes each prompt. The generated token-id
sequence IS the base-argmax "needed token" stream (greedy emit == argmax at every
step). This is the only GPU cost in the coverage-gap analysis.

Reads prompts.jsonl (build_prompts.py) and writes decode.jsonl with, per sample:
  task, id, gold_answer, gold_kind, prompt_len, completion_token_ids,
  generated_text, finish_reason, num_completion_tokens.

Settings mirror ubel #511's base server for comparability: dtype bf16,
max_model_len 4096, greedy (temperature 0), max-tokens MMLU 2048 / GPQA 3072 /
AIME 3072. Per-request max_tokens is clamped to fit the 4096 context (same
truncation behavior the served base arm had).

Run (vLLM env):
  CUDA_VISIBLE_DEVICES=0 .venvs/vllm022/bin/python \
      research/validity/keepset_coverage_gap/gen_base_greedy.py \
      --prompts research/validity/keepset_coverage_gap/prompts.jsonl \
      --out research/validity/keepset_coverage_gap/decode.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sysconfig
import time

# vLLM's memory-profiling pass JIT-compiles a flashinfer top-k/top-p sampling
# kernel even under greedy decoding, and that build needs curand.h -- which the
# system CUDA toolkit at /usr/local/cuda lacks (it ships only in the pip
# nvidia-cu13 wheels). For greedy/argmax the flashinfer sampler is unnecessary,
# so force the native sampler: numerically identical (argmax of the same logits),
# only stochastic top-k/top-p would differ. Also expose the wheel's CUDA headers
# via CPATH so any residual flashinfer JIT can still find curand.h.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
_cu13_incs = [
    p
    for p in (
        os.path.join(sysconfig.get_paths().get(k, ""), "nvidia", "cu13", "include")
        for k in ("purelib", "platlib")
    )
    if p and os.path.isdir(p)
]
if _cu13_incs:
    _cpath = os.environ.get("CPATH", "")
    os.environ["CPATH"] = os.pathsep.join(_cu13_incs + ([_cpath] if _cpath else []))

MODEL_ID = "google/gemma-4-E4B-it"
MAX_MODEL_LEN = 4096
TASK_MAX_TOKENS = {"mmlu_pro": 2048, "gpqa_diamond": 3072, "aime2024": 3072}
DEFAULT_MAX_TOKENS = 2048


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    ap.add_argument("--gpu-mem-util", type=float, default=0.88)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="cap total prompts (smoke)")
    ap.add_argument("--chunk", type=int, default=64,
                    help="generate+flush this many prompts per checkpoint (resumable)")
    ap.add_argument("--resume", action="store_true",
                    help="skip (task,id) pairs already present in --out and append")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    records = []
    with open(args.prompts) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if args.limit > 0:
        records = records[: args.limit]
    print(f"[gen] loaded {len(records)} prompts from {args.prompts}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)

    # Apply the chat template exactly as the vLLM OpenAI server does:
    # messages=[{user}], add_generation_prompt=True. tokenize=True returns the
    # full prompt id stream incl. the template's special tokens (no extra BOS).
    prompt_ids_list = []
    samp_params = []
    for r in records:
        msgs = [{"role": "user", "content": r["prompt_text"]}]
        # gemma-4's apply_chat_template returns a BatchEncoding even with
        # tokenize=True; return_dict + ["input_ids"] gives the plain int id list.
        enc = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=True
        )
        pids = list(enc["input_ids"])
        cap = TASK_MAX_TOKENS.get(r["task"], DEFAULT_MAX_TOKENS)
        room = args.max_model_len - len(pids) - 8
        max_toks = max(16, min(cap, room))
        prompt_ids_list.append(pids)
        samp_params.append(
            SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_toks)
        )

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=True,
        enforce_eager=True,
        disable_log_stats=True,
    )

    # --- resume: skip (task,id) already decoded so an interrupted run is recoverable ---
    done = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    done.add((d["task"], str(d["id"])))
                except (json.JSONDecodeError, KeyError):
                    continue  # tolerate a partial final line from a hard kill
        print(f"[gen] resume: {len(done)} decodes already present in {args.out}", flush=True)

    idx = [i for i, r in enumerate(records) if (r["task"], str(r["id"])) not in done]
    print(f"[gen] {len(idx)} prompts pending (of {len(records)})", flush=True)

    # Chunked generate+append: each chunk is flushed+fsync'd so a kill loses at
    # most one chunk, not the whole run (vLLM returns outputs in input order).
    mode = "a" if (args.resume and done) else "w"
    n_tok = 0
    t0 = time.time()
    with open(args.out, mode) as f:
        for c0 in range(0, len(idx), args.chunk):
            cidx = idx[c0 : c0 + args.chunk]
            prompts = [TokensPrompt(prompt_token_ids=prompt_ids_list[i]) for i in cidx]
            params = [samp_params[i] for i in cidx]
            outs = llm.generate(prompts, params)
            for i, o in zip(cidx, outs):
                r = records[i]
                co = o.outputs[0]
                ctoks = list(co.token_ids)
                n_tok += len(ctoks)
                f.write(
                    json.dumps(
                        {
                            "task": r["task"],
                            "id": r["id"],
                            "gold_answer": r["gold_answer"],
                            "gold_kind": r["gold_kind"],
                            "prompt_len": len(prompt_ids_list[i]),
                            "num_completion_tokens": len(ctoks),
                            "finish_reason": co.finish_reason,
                            "completion_token_ids": ctoks,
                            "generated_text": co.text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            f.flush()
            os.fsync(f.fileno())
            print(f"[gen] checkpoint {min(c0 + args.chunk, len(idx))}/{len(idx)} "
                  f"({n_tok} new tokens, {time.time() - t0:.0f}s)", flush=True)
    dt = time.time() - t0

    from collections import Counter

    by_task = Counter(records[i]["task"] for i in idx)
    print(
        f"[gen] wrote {len(idx)} decodes ({n_tok} completion tokens) -> {args.out}",
        flush=True,
    )
    print(f"[gen] by task: {dict(by_task)} | wall={dt:.1f}s "
          f"({n_tok / max(dt, 1e-9):.0f} tok/s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
