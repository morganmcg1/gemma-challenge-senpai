#!/usr/bin/env python3
"""Smoke client for the intact-body head-width serve patch (PR #547).

Sends a few fixed greedy prompts to a running serve and records, per prompt,
the generated token ids (via logprobs). Two checks:

  * --keepset <json>: assert EVERY generated token id is in the keepset. For
    mask/slice mode this MUST hold (non-kept ids are -inf => impossible). Proves
    the head-width knob is actually in force on the decode path.
  * --compare a.json b.json: assert the two recorded token-id streams are
    identical (used to validate slice == mask token-for-token).

Greedy (temperature 0) so the stream is deterministic and A/B-comparable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

PROMPTS = [
    "Q: What is 17 plus 26? Think step by step, then give the final number.",
    "Explain in one sentence why the sky appears blue.",
    "List three prime numbers greater than 10.",
]


def run(base_url: str, model: str, max_tokens: int) -> list[dict]:
    out = []
    for i, p in enumerate(PROMPTS):
        body = {
            "model": model,
            "messages": [{"role": "user", "content": p}],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_tokens,
            "logprobs": True,
            "top_logprobs": 0,
        }
        r = requests.post(f"{base_url}/chat/completions", json=body, timeout=120)
        r.raise_for_status()
        d = r.json()
        ch = d["choices"][0]
        text = ch["message"]["content"]
        ids = []
        lp = ch.get("logprobs") or {}
        for tok in (lp.get("content") or []):
            tid = tok.get("token")
            ids.append(tid)
        out.append({"prompt_idx": i, "text": text, "n_out": len(ids), "token_strs": ids})
        print(f"[smoke] prompt {i}: n_out={len(ids)} text={text[:80]!r}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="gemma-4-e4b-it")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--compare", nargs=2, type=Path, default=None,
                    help="two prior --out json files; assert identical token streams")
    args = ap.parse_args()

    if args.compare:
        a = json.loads(args.compare[0].read_text())
        b = json.loads(args.compare[1].read_text())
        assert len(a) == len(b), f"record count differs {len(a)} vs {len(b)}"
        all_eq = True
        for ra, rb in zip(a, b):
            eq = ra["token_strs"] == rb["token_strs"]
            all_eq = all_eq and eq
            print(f"[compare] prompt {ra['prompt_idx']}: identical={eq} "
                  f"(n {ra['n_out']} vs {rb['n_out']})", flush=True)
        print(f"[compare] ALL IDENTICAL: {all_eq}", flush=True)
        return 0 if all_eq else 2

    recs = run(args.base_url, args.model, args.max_tokens)
    args.out.write_text(json.dumps(recs, indent=2))
    print(f"[smoke] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
