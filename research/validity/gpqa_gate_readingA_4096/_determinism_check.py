#!/usr/bin/env python3
"""Pre-sweep gate-validity check at model_len 8192 (PR #643).

Two things, cheaply, before trusting the model_len 6144->8192 bump:
  1) COLLISION CLEARED: the 1 long GPQA-D item (~2429 input tokens) must now return
     200 with mt=4096 (2429+4096=6525 < 8192), not the 400 it hit at model_len 6144.
  2) DETERMINISM TRANSFERS: greedy (temp=0) at conc=1 must be byte-identical run-to-run.
     #631 established this at model_len 6144 (1/198 fragile). At conc=1+BI=1 model_len
     only changes KV capacity, not per-sequence compute, so it should transfer verbatim.

Sends each selected prompt TWICE (serially, so conc=1) and compares completions byte
for byte. Uses the same decode params run_eval.py sends for the greedy arm
(temp=0, top_p=1.0, seed=0, max_tokens=4096, min_tokens=8).
"""
from __future__ import annotations
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))
from transformers import AutoTokenizer  # noqa: E402
from inspect_evals.gpqa.gpqa import get_gpqa_diamond_dataset  # noqa: E402

BASE = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "gemma-4-e4b-it"
INSTR = ("Answer the following multiple choice question. The entire content of your "
         "response should be of the following format: 'ANSWER: $LETTER' (without quotes) "
         "where LETTER is one of ABCD. Think step by step before answering.\n\n")
LETTERS = "ABCD"


def render(s) -> str:
    return (INSTR + str(s.input) + "\n\n"
            + "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(s.choices)) + "\n")


def post(body: str) -> tuple[int, str]:
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": body}],
        "max_tokens": 4096, "temperature": 0.0, "top_p": 1.0,
        "seed": 0, "min_tokens": 8,
    }).encode()
    req = urllib.request.Request(BASE, data=payload,
                                headers={"Content-Type": "application/json",
                                         "Authorization": "Bearer EMPTY"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.loads(r.read())
        return 200, d["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:200]


def main() -> int:
    tok = AutoTokenizer.from_pretrained("/workspace/gemma_build/int4_g128_lmhead")
    samples = list(get_gpqa_diamond_dataset(shuffle_choices=False))
    toks = [(len(tok(render(s), add_special_tokens=False)["input_ids"]), i)
            for i, s in enumerate(samples)]
    toks.sort(reverse=True)
    longest_i = toks[0][1]
    # the longest item (collision item) + 4 short ones by dataset order
    pick = [longest_i] + [i for i in range(len(samples)) if i != longest_i][:4]
    print(f"longest item idx={longest_i} ~{toks[0][0]} input tokens (collision item)", flush=True)

    all_ident = True
    any_400 = False
    for n, i in enumerate(pick):
        body = render(samples[i])
        c1, t1 = post(body)
        c2, t2 = post(body)
        if c1 != 200 or c2 != 200:
            any_400 = True
            print(f"[{n}] idx={i} HTTP {c1}/{c2}  <-- NON-200 "
                  f"{t1[:120] if c1!=200 else ''}", flush=True)
            continue
        ident = (t1 == t2)
        all_ident = all_ident and ident
        tag = "BYTE-IDENTICAL" if ident else "*** DIFFERS ***"
        print(f"[{n}] idx={i} HTTP 200/200 len={len(t1)}/{len(t2)} {tag}", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    print(f"collision_cleared (long item 200): {not any_400}", flush=True)
    print(f"greedy_byte_identical (all picks): {all_ident}", flush=True)
    ok = (not any_400) and all_ident
    print(f"VERDICT: {'PASS - gate point valid at model_len 8192' if ok else 'INVESTIGATE'}",
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
