#!/usr/bin/env python
"""Co-scheduled (concurrent) greedy decode driver for the determinism probe.

Companion to the harness `decode_outputs.py`, which issues prompts strictly
sequentially (max_concurrency 1 / M=1). This driver fires all prompts
*concurrently* so vLLM co-schedules them in the same decode steps, letting us
test whether batched co-scheduling perturbs the Marlin split-K / Triton-attn
reduction order (and thus argmax at near-tie logits) relative to the M=1 path.

Records are written in the same shape decode_outputs.py emits, so the official
`check_greedy_identity.py` can compare a concurrent run against another
concurrent run (run-to-run) or against an M=1 run (composition effect).

Diagnostic only: changes nothing in the submission, launches no HF job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def sha256_tokens(tokens: list[int]) -> str:
    return hashlib.sha256(",".join(str(t) for t in tokens).encode("ascii")).hexdigest()


def read_sharegpt_prompts(path: Path, num_prompts: int, seed: int) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    records: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        conv = item.get("conversations")
        if not isinstance(conv, list) or len(conv) < 2 or not isinstance(conv[0], dict):
            continue
        prompt = conv[0].get("value")
        if not isinstance(prompt, str) or not prompt:
            continue
        records.append({"id": str(item.get("id", index)), "dataset_index": index, "prompt_text": prompt})
    random.Random(seed).shuffle(records)
    return records[:num_prompts]


def post(url: str, payload: dict, timeout_s: int) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read().decode())


def extract_completion(resp: dict, prompt_token_ids: list[int]) -> list[int]:
    ch = resp["choices"][0]
    tids = ch.get("token_ids") or []
    if len(tids) >= len(prompt_token_ids) and tids[: len(prompt_token_ids)] == prompt_token_ids:
        return tids[len(prompt_token_ids) :]
    return tids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="gemma-4-e4b-it")
    ap.add_argument("--tokenizer", default="google/gemma-4-E4B-it-qat-w4a16-ct")
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--output-file", required=True)
    ap.add_argument("--request-timeout-s", type=int, default=300)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    records = read_sharegpt_prompts(Path(args.dataset_path), args.num_prompts, args.seed)

    def encode(text: str) -> list[int]:
        enc = tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True
        )
        if hasattr(enc, "input_ids"):
            enc = enc.input_ids
        if hasattr(enc, "tolist"):
            enc = enc.tolist()
        if isinstance(enc, dict):
            enc = enc.get("input_ids", enc)
        if len(enc) == 1 and isinstance(enc[0], (list, tuple)):
            enc = enc[0]
        return [int(t) for t in enc]

    prompts = [encode(r["prompt_text"]) for r in records]

    def one(i: int) -> dict:
        ptoks = prompts[i]
        payload = {
            "model": args.model,
            "prompt": ptoks,
            "max_tokens": args.output_len,
            "temperature": 0.0,
            "stream": False,
            "add_special_tokens": False,
            "ignore_eos": True,
            "return_token_ids": True,
        }
        resp = post(f"{args.base_url}/v1/completions", payload, args.request_timeout_s)
        comp = extract_completion(resp, ptoks)
        return {
            "id": records[i]["id"],
            "index": i,
            "dataset_index": records[i]["dataset_index"],
            "prompt_token_ids": ptoks,
            "completion_token_ids": comp,
            "completion_token_sha256": sha256_tokens(comp),
            "num_completion_tokens": len(comp),
        }

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        rows = list(ex.map(one, range(len(records))))
    dt = time.time() - t0

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: r["id"])
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    total_out = sum(r["num_completion_tokens"] for r in rows)
    print(
        f"concurrent decode: {len(rows)} prompts, concurrency={args.concurrency}, "
        f"{total_out} completion tokens in {dt:.1f}s -> {out_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
