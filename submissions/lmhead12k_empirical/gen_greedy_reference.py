#!/usr/bin/env python
"""Generate the *plain greedy autoregressive* reference for the gate check.

The challenge's hard validity rule (program.md, greedy-identity verifier README):

  > the served endpoint's greedy decode must be token-identical to plain greedy
  > autoregressive decode of the same submitted checkpoint.

The *candidate* is the served ``decode_outputs.jsonl`` (produced through serve.py
+ the OpenAI API by the harness ``decode_outputs.py``). This script produces the
*reference*: the SAME submitted checkpoint decoded with vLLM's offline engine in
the plainest greedy config -- ``enforce_eager=True`` (no cudagraph), greedy
(``temperature=0``), ``ignore_eos=True``, ``max_tokens=512`` -- over the SAME 128
ShareGPT prompts selected and chat-template-encoded by the harness helpers, so the
record ``id`` keys line up one-for-one for the verifier.

If serving introduced no token-changing optimization (we use none: standard vLLM
int4 Marlin + the exact lmhead12k full-vocab argmax scatter + greedy-preserving
continuous batching / chunked prefill), the official verifier returns
GREEDY_IDENTICAL. This is the authoritative gate -- NOT a diff against the bf16
original model (int4 quantization legitimately diverges from bf16, which the gate
permits; its reference is the submitted checkpoint's own plain greedy decode).

GPU-only. Run inside the A10G window with
``CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "official/main_bucket/shared_resources/speed_benchmark/scripts"
DATASET = ROOT / "official/main_bucket/shared_resources/speed_benchmark/data/eval_prompts_sharegpt.json"
DEFAULT_MODEL = "/workspace/gemma_build/lmhead12k_empirical"


def sha256_tokens(tokens: list[int]) -> str:
    return hashlib.sha256(",".join(str(t) for t in tokens).encode("ascii")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dataset-path", default=str(DATASET))
    ap.add_argument("--output-file", required=True)
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tokenizer", default="google/gemma-4-E4B-it")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()

    # The custom model class locates kept_ids.json via MODEL_ID (serve.py sets it
    # in the engine subprocess); mirror that for the offline engine.
    os.environ.setdefault("MODEL_ID", args.model)
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    # Reuse the harness's exact prompt selection + chat-template encoding so the
    # reference record ids/prompts line up with the served candidate.
    sys.path.insert(0, str(HARNESS))
    from decode_outputs import encode_prompt, read_sharegpt_prompts  # type: ignore
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    records = read_sharegpt_prompts(
        Path(args.dataset_path), num_prompts=args.num_prompts, seed=args.seed
    )
    if len(records) != args.num_prompts:
        raise SystemExit(f"expected {args.num_prompts} prompts, found {len(records)}")

    prompt_token_ids = [encode_prompt(tokenizer, r["prompt_text"]) for r in records]

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )
    sp = SamplingParams(
        temperature=0.0, max_tokens=args.output_len, ignore_eos=True
    )
    outputs = llm.generate(
        [{"prompt_token_ids": ids} for ids in prompt_token_ids], sp
    )

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec, ids, out in zip(records, prompt_token_ids, outputs):
            comp = list(out.outputs[0].token_ids)
            row = {
                "id": rec["id"],
                "dataset_index": rec["dataset_index"],
                "prompt_token_ids": ids,
                "completion_token_ids": comp,
                "completion_token_sha256": sha256_tokens(comp),
                "num_completion_tokens": len(comp),
            }
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    print(json.dumps({
        "output_file": str(out_path),
        "num_records": len(records),
        "output_len": args.output_len,
        "engine": "vllm-offline enforce_eager greedy (plain autoregressive)",
        "model": args.model,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
