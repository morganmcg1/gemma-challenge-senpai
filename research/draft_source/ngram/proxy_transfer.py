#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Transfer-risk tokenization for PR #250 (student ubel).

The PRIMARY n-gram diagnostic (profile.py) runs on the OFFICIAL public benchmark
set (mmlu_pro / gpqa_diamond / aime2026 reasoning) with EXACT cached greedy
reference token ids. PR step 5 asks whether that match-rate transfers to the
PRIVATE prompt distribution. We approximate the private distribution with the
native HARD proxies (PR #164): code / longctx / math / casual / multilingual.

We only have the proxy GENERATIONS as text (generated_texts in the cached
private_gap_probe bench files), not as decoded token ids, and the proxy PROMPT
construction does not faithfully reconstruct from the raw conversation. On the
public set, copy-FROM-PROMPT contributes only +0.097 E[T] while self-repetition
in the generation contributes the bulk (E[T] 1.273 of 1.370). So we measure the
reliably-recoverable component -- GENERATION-INTERNAL n-gram E[T] -- on each
proxy category, re-tokenizing generated_texts with the gemma tokenizer, and
compare apples-to-apples against the public gen-only number.

This script loads the gemma tokenizer straight from the cached tokenizer.json
via the `tokenizers` library (no torch; transformers' lazy AutoTokenizer import
is flaky under the CUDA-void student container) and caches token ids to
proxy_refs.json so profile.py stays dependency-free.

Usage:
  python research/draft_source/ngram/proxy_transfer.py
"""
import glob
import json
import os

CATS = ["code", "longctx", "math", "casual", "multilingual"]
BENCH = "research/validity/private_gap_probe/native_{c}/bench_private_rerun.jsonl"
OUT = "research/draft_source/ngram/proxy_refs.json"
MODEL = "google/gemma-4-E4B-it"


def load_tokenizer():
    from tokenizers import Tokenizer
    pat = os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/*/tokenizer.json")
    hits = glob.glob(pat)
    if not hits:
        raise FileNotFoundError(f"no cached tokenizer.json at {pat}")
    return Tokenizer.from_file(hits[0])


def main():
    tok = load_tokenizer()

    out = {"model": MODEL, "note": "gen-only re-tokenized completions per proxy category",
           "categories": {}}
    for c in CATS:
        path = BENCH.format(c=c)
        if not os.path.exists(path):
            print(f"skip {c}: no bench file")
            continue
        with open(path) as fh:
            b = json.loads(fh.readline())
        gens = b.get("generated_texts") or []
        comps = []
        for g in gens:
            if not g:
                continue
            ids = tok.encode(g, add_special_tokens=False).ids
            comps.append(ids)
        out["categories"][c] = {
            "n_records": len(comps),
            "mean_len": (sum(len(x) for x in comps) // max(1, len(comps))),
            "completions": comps,
        }
        print(f"{c:>12}: {len(comps)} gens, mean_len "
              f"{out['categories'][c]['mean_len']}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"wrote {OUT} ({os.path.getsize(OUT)//1024} KiB)")


if __name__ == "__main__":
    main()
