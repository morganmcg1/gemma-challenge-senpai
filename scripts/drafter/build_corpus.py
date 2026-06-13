#!/usr/bin/env python3
"""Build a wide, distribution-matched prompt corpus for KL-distilling the
Gemma-4 MTP drafter, per `kl_distill_reference_itaca/corpus_spec.md`.

Design goals (the hypothesis under test): width + private-distribution match,
NOT public-bench overfit. We:
  * sample from ShareGPT (chat) + MMLU-Pro + a science-MCQ GPQA proxy +
    MATH/AIME + misc instruct, matching the four eval distributions plus chat,
  * dedup every prompt against the 128 public bench prompts by 512-token-id
    prefix hash AND normalized-text exact match (never train on the bench),
  * reserve a stratified held-out shard (never trained on) for the offline
    acceptance + stability gate.

We store (prompt, reference) pairs only. Trace capture (target hidden states +
shared-KV + top-2048 target dist) happens on-the-fly inside the training / eval
harness, because the Gemma4 MTP assistant is an EAGLE-style head that conditions
on the target's hidden states + shared KV — those are prefix-dependent and far
too large to pre-serialize (unlike the standalone-draft assumption in the
reference `train_kl_drafter.py`).

GPQA note: Idavidrein/gpqa is gated and only ships a `test` split (eval-overlap
risk), so the 10% GPQA slice is proxied by MMLU-Pro restricted to graduate
science categories (physics/chemistry/biology/health) — same hard-science MCQ
distribution, no eval-contamination risk. Documented in composition.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict

from datasets import load_dataset
from transformers import AutoTokenizer

TOKENIZER = "google/gemma-4-E4B-it"
EVAL_PROMPTS = "official/main_bucket/shared_resources/speed_benchmark/data/eval_prompts_sharegpt.json"

MCQ_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your "
    "response should be of the following format: 'ANSWER: $LETTER' (without "
    "quotes) where LETTER is one of {letters}. Think step by step before "
    "answering.\n\n{question}\n\n{options}"
)
SCIENCE_CATS = {"physics", "chemistry", "biology", "health"}


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def prefix_hash(tok, text: str, n: int = 512) -> str:
    ids = tok(text, add_special_tokens=False).input_ids[:n]
    return hashlib.sha1(",".join(map(str, ids)).encode()).hexdigest()


def fmt_mcq(question: str, options: list[str]) -> str:
    letters = ",".join(chr(ord("A") + i) for i in range(len(options)))
    opt_lines = "\n".join(f"{chr(ord('A') + i)}. {o}" for i, o in enumerate(options))
    return MCQ_TEMPLATE.format(letters=letters, question=question.strip(), options=opt_lines)


def stream_take(label, repo, cfg, split, n, extract, dist, source):
    """Yield up to n (prompt, reference, dist, source) dicts from a streamed source."""
    out = []
    try:
        ds = load_dataset(repo, cfg, split=split, streaming=True)
    except Exception as e:
        print(f"[WARN] {label}: load failed {type(e).__name__}: {str(e)[:120]}")
        return out
    seen = 0
    for row in ds:
        seen += 1
        if seen > n * 25 and len(out) >= n:
            break
        try:
            pr = extract(row)
        except Exception:
            continue
        if pr is None:
            continue
        prompt, ref = pr
        if not prompt or len(prompt) < 16:
            continue
        out.append({"prompt": prompt, "reference": ref or "", "dist": dist, "source": source})
        if len(out) >= n:
            break
    print(f"[ok] {label}: collected {len(out)} (scanned {seen})")
    return out


def ex_sharegpt(row):
    convs = row.get("conversations") or []
    human = next((c["value"] for c in convs if c.get("from") in ("human", "user")), None)
    gpt = next((c["value"] for c in convs if c.get("from") in ("gpt", "assistant")), None)
    if not human:
        return None
    return human, gpt


def ex_openthoughts(row):
    convs = row.get("conversations") or []
    human = next((c["value"] for c in convs if c.get("from") in ("human", "user")), None)
    gpt = next((c["value"] for c in convs if c.get("from") in ("gpt", "assistant")), None)
    if not human:
        return None
    return human, gpt


def ex_mmlu(row, science_only=False):
    cat = (row.get("category") or "").lower()
    if science_only and cat not in SCIENCE_CATS:
        return None
    if (not science_only) and cat in SCIENCE_CATS:
        # leave science rows for the gpqa-proxy slice to avoid double counting
        return None
    q = row.get("question")
    opts = row.get("options") or []
    if not q or not opts:
        return None
    prompt = fmt_mcq(q, opts)
    cot = row.get("cot_content") or ""
    ans = row.get("answer") or ""
    ref = cot.strip() if cot.strip() else f"Let me think step by step.\nANSWER: {ans}"
    return prompt, ref


def ex_math(row):
    prob = row.get("problem")
    if not prob:
        return None
    prompt = prob.strip() + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    return prompt, (row.get("solution") or "")


def ex_aime(row):
    prob = row.get("Problem")
    if not prob:
        return None
    prompt = prob.strip() + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    return prompt, (row.get("Solution") or "")


def ex_openorca(row):
    q = row.get("question")
    if not q:
        return None
    return q.strip(), (row.get("response") or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/wide_drafter/corpus")
    ap.add_argument("--n-sharegpt", type=int, default=3000)
    ap.add_argument("--n-mmlu", type=int, default=1200)
    ap.add_argument("--n-gpqa-proxy", type=int, default=600)
    ap.add_argument("--n-math", type=int, default=450)
    ap.add_argument("--n-aime", type=int, default=150)
    ap.add_argument("--n-misc", type=int, default=600)
    ap.add_argument("--heldout-frac", type=float, default=0.15)
    ap.add_argument("--min-prompt-tokens", type=int, default=12)
    ap.add_argument("--max-prompt-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20260613)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    # --- eval dedup sets ---
    eval_data = json.load(open(EVAL_PROMPTS))
    eval_hashes, eval_norms = set(), set()
    for r in eval_data:
        human = next((c["value"] for c in r["conversations"] if c.get("from") == "human"), "")
        eval_hashes.add(prefix_hash(tok, human))
        eval_norms.add(norm_text(human)[:400])
    print(f"[dedup] {len(eval_hashes)} eval prefixes to exclude")

    # --- collect sources ---
    rows = []
    rows += stream_take("sharegpt", "Aeala/ShareGPT_Vicuna_unfiltered", None, "train",
                        args.n_sharegpt, ex_sharegpt, "chat", "sharegpt")
    rows += stream_take("mmlu_pro", "TIGER-Lab/MMLU-Pro", None, "test",
                        args.n_mmlu, lambda r: ex_mmlu(r, science_only=False), "reasoning_mcq", "mmlu_pro")
    rows += stream_take("gpqa_proxy", "TIGER-Lab/MMLU-Pro", None, "test",
                        args.n_gpqa_proxy, lambda r: ex_mmlu(r, science_only=True), "reasoning_mcq", "mmlu_pro_science")
    rows += stream_take("math", "nlile/hendrycks-MATH-benchmark", None, "train",
                        args.n_math, ex_math, "math", "hendrycks_math")
    rows += stream_take("aime", "Maxwell-Jia/AIME_2024", None, "train",
                        args.n_aime, ex_aime, "math", "aime_2024")
    rows += stream_take("misc", "Open-Orca/OpenOrca", None, "train",
                        args.n_misc, ex_openorca, "chat", "openorca")

    # --- dedup (eval + within corpus) + length filter ---
    kept, seen_hashes = [], set()
    drops = Counter()
    for r in rows:
        h = prefix_hash(tok, r["prompt"])
        nrm = norm_text(r["prompt"])[:400]
        ntok = len(tok(r["prompt"], add_special_tokens=False).input_ids)
        if h in eval_hashes or nrm in eval_norms:
            drops["eval_overlap"] += 1
            continue
        if h in seen_hashes:
            drops["intra_dup"] += 1
            continue
        if ntok < args.min_prompt_tokens or ntok > args.max_prompt_tokens:
            drops["length"] += 1
            continue
        seen_hashes.add(h)
        r["prefix_hash"] = h
        r["n_prompt_tokens"] = ntok
        kept.append(r)
    print(f"[dedup] kept {len(kept)} / {len(rows)}; drops={dict(drops)}")

    # --- stratified held-out split by dist ---
    by_dist = defaultdict(list)
    for r in kept:
        by_dist[r["dist"]].append(r)
    train, heldout = [], []
    for dist, items in by_dist.items():
        random.shuffle(items)
        k = max(1, int(len(items) * args.heldout_frac))
        heldout += items[:k]
        train += items[k:]
    random.shuffle(train)
    random.shuffle(heldout)

    with open(os.path.join(args.out, "train.jsonl"), "w") as fh:
        for r in train:
            fh.write(json.dumps(r) + "\n")
    with open(os.path.join(args.out, "heldout.jsonl"), "w") as fh:
        for r in heldout:
            fh.write(json.dumps(r) + "\n")

    comp = {
        "tokenizer": TOKENIZER,
        "seed": args.seed,
        "totals": {"train": len(train), "heldout": len(heldout), "kept": len(kept), "raw": len(rows)},
        "dedup_drops": dict(drops),
        "eval_prefixes_excluded": len(eval_hashes),
        "by_source_train": dict(Counter(r["source"] for r in train)),
        "by_source_heldout": dict(Counter(r["source"] for r in heldout)),
        "by_dist_train": dict(Counter(r["dist"] for r in train)),
        "by_dist_heldout": dict(Counter(r["dist"] for r in heldout)),
        "notes": {
            "gpqa": "GPQA gated/test-only -> proxied by MMLU-Pro science cats "
                    "(physics/chemistry/biology/health).",
            "math_source": "nlile/hendrycks-MATH-benchmark (hendrycks/competition_math removed).",
            "dedup": "512-token-id prefix sha1 + normalized-text exact match vs the 128 public prompts.",
        },
    }
    with open(os.path.join(args.out, "composition.json"), "w") as fh:
        json.dump(comp, fh, indent=2)
    print(json.dumps(comp, indent=2))


if __name__ == "__main__":
    main()
