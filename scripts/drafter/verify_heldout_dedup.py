#!/usr/bin/env python3
"""Re-confirm the held-out shard has ZERO overlap with the 128 public bench
prompts, using the exact dedup keys build_corpus.py applied (512-token-id prefix
sha1 AND normalized-text exact match). Run after any rebase that touches the
shard so the offline-gate proof stays valid. Exits non-zero on any overlap.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys

from transformers import AutoTokenizer

# Mirror build_corpus.py exactly (kept inline so this runs in the eval venv,
# which lacks `datasets`; build_corpus imports datasets at module load).
TOKENIZER = "google/gemma-4-E4B-it"
EVAL_PROMPTS = "official/main_bucket/shared_resources/speed_benchmark/data/eval_prompts_sharegpt.json"
HELDOUT = "research/wide_drafter/corpus/heldout.jsonl"


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def prefix_hash(tok, text: str, n: int = 512) -> str:
    ids = tok(text, add_special_tokens=False).input_ids[:n]
    return hashlib.sha1(",".join(map(str, ids)).encode()).hexdigest()


def main():
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    eval_data = json.load(open(EVAL_PROMPTS))
    eval_hashes, eval_norms = set(), set()
    for r in eval_data:
        human = next((c["value"] for c in r["conversations"] if c.get("from") == "human"), "")
        eval_hashes.add(prefix_hash(tok, human))
        eval_norms.add(norm_text(human)[:400])

    rows = [json.loads(l) for l in open(HELDOUT)]
    hash_hits = norm_hits = 0
    for r in rows:
        if prefix_hash(tok, r["prompt"]) in eval_hashes:
            hash_hits += 1
        if norm_text(r["prompt"])[:400] in eval_norms:
            norm_hits += 1

    overlap = hash_hits + norm_hits
    print(json.dumps({
        "eval_prompts_file": EVAL_PROMPTS,
        "n_eval_prompts": len(eval_data),
        "n_eval_prefix_hashes": len(eval_hashes),
        "heldout_file": HELDOUT,
        "n_heldout": len(rows),
        "heldout_prefix_hash_overlap": hash_hits,
        "heldout_norm_text_overlap": norm_hits,
        "zero_overlap": overlap == 0,
    }, indent=2))
    if overlap:
        print(f"FAIL: {overlap} held-out prompt(s) overlap the public bench", file=sys.stderr)
        sys.exit(1)
    print("DEDUP_OK")


if __name__ == "__main__":
    main()
