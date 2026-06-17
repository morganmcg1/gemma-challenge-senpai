#!/usr/bin/env python3
"""Sampled GSM8K eval with an EOS-guard arm for the intact-body head-width sweep (PR #547).

GSM8K is the gate's long-ish generative cell. Unlike MMLU-Pro/GPQA (greedy, single
MC letter) it is SAMPLED per the model's own generation_config (the "lewtun
directive": temperature=1.0, top_p=0.95, top_k=64 -- the values in
gemma-4-E4B-it-qat-w4a16-ct/generation_config.json). It is driven directly over the
OpenAI-compatible endpoint (not inspect) so we control the exact request body,
in particular vLLM's `min_tokens` EOS-guard.

Two arms (run the script twice, --min-tokens 0 vs 8):
  * as-served  (--min-tokens 0): default decoding. A first-token-EOS empty
    completion scores wrong and depresses the cell -- this is the artefact
    wirbel #541 flagged.
  * guarded    (--min-tokens 8): vLLM is told to emit >=8 tokens before EOS is
    allowed, recovering the recoverable empties.

Per-item seeds are derived from --seed so the two arms differ ONLY in the guard
(same sampling stream otherwise), making the as-served vs guarded delta clean.

Scoring: gold = number after the last '####' in the GSM8K answer; pred = number
after the LAST '####' the model emits, else the last number in the text. Both
normalised (strip $ , % and trailing punctuation), compared as floats.

empty_rate = fraction of completions whose stripped text is "" (or 0 completion
tokens). analysis_only -- official_tps=0, local serve only.
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from datasets import load_dataset

INSTRUCTION = (
    "Solve the following grade-school math problem. Reason step by step, then on "
    "the final line write the answer in the form '#### <number>'."
)

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _norm_num(s: str) -> float | None:
    s = s.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def gold_answer(ans_field: str) -> float | None:
    # GSM8K gold is the value after the final '#### '
    tail = ans_field.split("####")[-1]
    m = _NUM_RE.search(tail)
    return _norm_num(m.group(0)) if m else None


def pred_answer(text: str) -> float | None:
    if "####" in text:
        tail = text.split("####")[-1]
        m = _NUM_RE.search(tail)
        if m:
            return _norm_num(m.group(0))
    # fallback: last number anywhere in the completion
    nums = _NUM_RE.findall(text)
    return _norm_num(nums[-1]) if nums else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="label, e.g. head12k")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--min-tokens", type=int, default=0, help="EOS guard; 0=as-served, 8=guarded")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--conc", type=int, default=32)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="gemma-4-e4b-it")
    # lewtun directive / model generation_config defaults:
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=64)
    args = ap.parse_args()

    ds = load_dataset("openai/gsm8k", "main", split="test")
    n = min(args.n, len(ds))
    rows = list(ds.select(range(n)))

    results: list[dict] = [None] * n  # type: ignore
    lock = threading.Lock()
    done = {"k": 0}

    def work(i: int) -> None:
        q = rows[i]["question"]
        gold = gold_answer(rows[i]["answer"])
        body = {
            "model": args.model,
            "messages": [
                {"role": "user", "content": f"{INSTRUCTION}\n\nQuestion: {q}"}
            ],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed": args.seed + i,
            "extra_body": {"top_k": args.top_k},
        }
        if args.min_tokens > 0:
            body["extra_body"]["min_tokens"] = args.min_tokens
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{args.base_url}/chat/completions", json=body, timeout=600)
            r.raise_for_status()
            d = r.json()
            text = d["choices"][0]["message"]["content"] or ""
            ctoks = int(d.get("usage", {}).get("completion_tokens") or 0)
            err = None
        except Exception as e:  # noqa: BLE001
            text, ctoks, err = "", 0, str(e)[:200]
        dt = time.perf_counter() - t0
        pred = pred_answer(text)
        empty = (text.strip() == "") or (ctoks == 0)
        correct = (gold is not None and pred is not None and abs(gold - pred) < 1e-6)
        results[i] = {
            "i": i, "gold": gold, "pred": pred, "correct": bool(correct),
            "empty": bool(empty), "completion_tokens": ctoks, "err": err,
            "latency_s": dt, "text_tail": text[-160:],
        }
        with lock:
            done["k"] += 1
            if done["k"] % 50 == 0 or done["k"] == n:
                nc = sum(1 for x in results if x and x["correct"])
                ne = sum(1 for x in results if x and x["empty"])
                print(f"[gsm8k:{args.arm} mt={args.min_tokens}] {done['k']}/{n} "
                      f"acc={nc/done['k']:.3f} empty={ne/done['k']:.3f}", flush=True)

    with ThreadPoolExecutor(max_workers=args.conc) as ex:
        list(ex.map(work, range(n)))

    n_correct = sum(1 for x in results if x["correct"])
    n_empty = sum(1 for x in results if x["empty"])
    n_err = sum(1 for x in results if x["err"])
    out = {
        "task": "gsm8k", "arm": args.arm, "model": args.model,
        "sampled": True, "temperature": args.temperature, "top_p": args.top_p,
        "top_k": args.top_k, "min_tokens": args.min_tokens,
        "seed": args.seed, "n": n, "max_tokens": args.max_tokens, "conc": args.conc,
        "n_correct": n_correct, "n_empty": n_empty, "n_err": n_err,
        "accuracy": n_correct / n if n else 0.0,
        "empty_rate": n_empty / n if n else 0.0,
        "base_url": args.base_url, "official_tps": 0, "analysis_only": True,
        "per_sample": results,
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[gsm8k:{args.arm} mt={args.min_tokens}] DONE acc={out['accuracy']:.4f} "
          f"empty={out['empty_rate']:.4f} n_err={n_err} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
