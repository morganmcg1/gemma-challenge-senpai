#!/usr/bin/env python
"""Build a chat-heavy *private-proxy* prompt set that length-matches the public 128.

The official public eval set (``official/.../speed_benchmark/data/eval_prompts_sharegpt.json``)
is ~100% reasoning (MMLU-Pro / GPQA / AIME-style instructions). The challenge's
private re-run set is believed *wide / chat-heavy*. The point of the private-gap
probe (PR #44) is to predict the public->private TPS gap that comes from
**distribution shift** -- chiefly the speculative drafter accepting fewer tokens
on conversational text than on its reasoning-heavy training distribution -- and
NOT from a trivial change in prompt length (which would change prefill cost).

So this builder samples real ShareGPT conversations, then selects a 128-prompt
subset whose *chat-templated token-length distribution* matches the public set
as closely as possible (greedy nearest-length matching against the public
per-prompt lengths), hard-deduped against the public prompts. The output is a
ShareGPT-schema JSON consumed byte-identically by both
``decode_outputs.read_sharegpt_prompts`` and ``sglang.bench_serving`` (so the
probe and the official harness load it the same way the public set is loaded).

Run with a python that has ``transformers`` (the fa2sw server venv works):

    /tmp/senpai-venvs/<hash>/bin/python scripts/validity/build_private_proxy.py \
        --sharegpt /tmp/sharegpt_src/ShareGPT_V3_unfiltered_cleaned_split.json \
        --out data/private_proxy_sharegpt.json

The 128-prompt JSON + a ``.meta.json`` provenance sidecar are committed; the
~673 MB raw ShareGPT download is not (kept under /tmp).
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import random
import re
import statistics
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
PUBLIC_DEFAULT = REPO / "official/main_bucket/shared_resources/speed_benchmark/data/eval_prompts_sharegpt.json"
SHAREGPT_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"

# Substrings that mark a ShareGPT turn as actually being a reasoning/eval-style
# prompt (we want the chat contrast, so these are dropped from the proxy).
REASONING_MARKERS = (
    "the last line of your response should be",
    "answer the following multiple choice question",
    "$letter",
    "answer: $",
)

_WS = re.compile(r"\s+")


def norm(text: str) -> str:
    return _WS.sub(" ", text.strip().lower())


def text_hash(text: str) -> str:
    return hashlib.sha256(norm(text).encode("utf-8")).hexdigest()


def load_tokenizer(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def templ_len(tok, text: str) -> int:
    """Chat-templated token length, mirroring decode_outputs.encode_prompt
    (apply_chat_template, add_generation_prompt=True) with robust extraction
    across transformers return shapes (list / BatchEncoding / dict / nested)."""
    enc = tok.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True
    )
    if hasattr(enc, "input_ids"):
        enc = enc.input_ids
    if isinstance(enc, dict):
        enc = enc.get("input_ids", enc)
    if hasattr(enc, "tolist"):
        enc = enc.tolist()
    if isinstance(enc, list) and enc and isinstance(enc[0], list):
        enc = enc[0]
    return len(enc)


def first_human_prompt(item: dict[str, Any]) -> str | None:
    """First human/user turn value, requiring at least one following turn
    (a real exchange, matching read_sharegpt_prompts' len(conversations) >= 2)."""
    conv = item.get("conversations")
    if not isinstance(conv, list) or len(conv) < 2:
        return None
    first = conv[0]
    if not isinstance(first, dict):
        return None
    if str(first.get("from", "")).lower() not in {"human", "user"}:
        return None
    value = first.get("value")
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def load_public(path: Path, tok, num: int, seed: int) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    recs: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        prompt = first_human_prompt(item)
        if prompt is None:
            continue
        recs.append({"id": str(item.get("id", index)), "prompt_text": prompt})
    rng = random.Random(seed)
    rng.shuffle(recs)
    recs = recs[:num]
    for r in recs:
        r["tok_len"] = templ_len(tok, r["prompt_text"])
    return recs


def collect_candidates(
    path: Path, public_hashes: set[str], public_prefixes: set[str],
    *, min_char: int, max_char: int, seed: int, cap: int,
) -> list[str]:
    """Stream ShareGPT, keep first-human chat prompts that (a) are not the public
    reasoning template, (b) hard-dedup vs public and within the proxy, and (c) sit
    in a char-length band wide enough to cover the public token-length range while
    bounding tokenization cost. Returns a shuffled, capped candidate text list."""
    data = json.loads(path.read_text())
    seen: set[str] = set()
    out: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        prompt = first_human_prompt(item)
        if prompt is None:
            continue
        clen = len(prompt)
        if clen < min_char or clen > max_char:
            continue
        low = prompt.lower()
        if any(m in low for m in REASONING_MARKERS):
            continue
        h = text_hash(prompt)
        if h in public_hashes or h in seen:
            continue
        if norm(prompt)[:80] in public_prefixes:  # near-dup prefix guard
            continue
        seen.add(h)
        out.append(prompt)
    rng = random.Random(seed)
    rng.shuffle(out)
    return out[:cap]


def nearest_match(
    targets: list[int], cand_lens: list[int], cand_texts: list[str]
) -> list[tuple[int, int, str]]:
    """Greedy nearest-length matching, each candidate used once. Longest targets
    first so the thin long tail is matched before the dense middle exhausts it."""
    order = sorted(cand_lens)
    paired = sorted(zip(cand_lens, range(len(cand_lens))))
    sorted_lens = [p[0] for p in paired]
    sorted_idx = [p[1] for p in paired]
    used = [False] * len(sorted_lens)
    matched: list[tuple[int, int, str]] = []
    for t in sorted(targets, reverse=True):
        pos = bisect.bisect_left(sorted_lens, t)
        best = None
        # expand outward from the insertion point to the nearest unused candidate
        lo, hi = pos - 1, pos
        while lo >= 0 or hi < len(sorted_lens):
            cand = None
            if hi < len(sorted_lens) and (
                lo < 0 or abs(sorted_lens[hi] - t) <= abs(sorted_lens[lo] - t)
            ):
                cand = hi
                hi += 1
            else:
                cand = lo
                lo -= 1
            if not used[cand]:
                best = cand
                break
        if best is None:
            raise RuntimeError("ran out of candidates during length matching")
        used[best] = True
        idx = sorted_idx[best]
        matched.append((t, cand_lens[idx], cand_texts[idx]))
    return matched


def dist(vals: list[int]) -> dict[str, float]:
    s = sorted(vals)
    n = len(s)
    q = lambda f: s[min(n - 1, int(n * f))]
    return {
        "n": n, "min": s[0], "p10": q(.1), "p25": q(.25), "p50": s[n // 2],
        "p75": q(.75), "p90": q(.9), "max": s[-1], "mean": round(statistics.mean(s), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sharegpt", default="/tmp/sharegpt_src/" + SHAREGPT_FILE,
                    help="path to ShareGPT_V3_unfiltered_cleaned_split.json")
    ap.add_argument("--public", default=str(PUBLIC_DEFAULT))
    ap.add_argument("--tokenizer", default="/tmp/osoi5-v0-baked")
    ap.add_argument("--out", default=str(REPO / "data/private_proxy_sharegpt.json"))
    ap.add_argument("--num", type=int, default=128)
    ap.add_argument("--seed", type=int, default=44)  # PR #44
    ap.add_argument("--candidate-cap", type=int, default=8000,
                    help="max ShareGPT candidates to tokenize for matching")
    args = ap.parse_args()

    public_path = Path(args.public)
    sharegpt_path = Path(args.sharegpt)
    if not sharegpt_path.exists():
        from huggingface_hub import hf_hub_download
        print(f"[build] downloading {SHAREGPT_FILE} ...", flush=True)
        sharegpt_path = Path(hf_hub_download(
            SHAREGPT_REPO, SHAREGPT_FILE, repo_type="dataset",
            local_dir="/tmp/sharegpt_src"))

    print("[build] loading tokenizer", flush=True)
    tok = load_tokenizer(args.tokenizer)

    print("[build] loading public set (official seed=1)", flush=True)
    public = load_public(public_path, tok, args.num, seed=1)
    public_lens = [r["tok_len"] for r in public]
    public_hashes = {text_hash(r["prompt_text"]) for r in public}
    public_prefixes = {norm(r["prompt_text"])[:80] for r in public}
    pmin, pmax = min(public_lens), max(public_lens)
    print(f"[build] public lens: {dist(public_lens)}", flush=True)

    # char band: wide enough to cover the token range, generous on the long side
    # so a ~2400-token outlier still has chat candidates to match.
    print("[build] collecting ShareGPT chat candidates", flush=True)
    cands = collect_candidates(
        sharegpt_path, public_hashes, public_prefixes,
        min_char=80, max_char=40000, seed=args.seed, cap=args.candidate_cap)
    print(f"[build] {len(cands)} candidates after filter/dedup; tokenizing", flush=True)

    cand_lens: list[int] = []
    cand_texts: list[str] = []
    for i, text in enumerate(cands):
        tl = templ_len(tok, text)
        if tl < max(8, pmin // 3) or tl > pmax * 3:
            continue
        cand_lens.append(tl)
        cand_texts.append(text)
        if (i + 1) % 2000 == 0:
            print(f"[build]   tokenized {i+1}/{len(cands)}", flush=True)
    print(f"[build] {len(cand_lens)} candidates in token band; matching", flush=True)

    matched = nearest_match(public_lens, cand_lens, cand_texts)
    proxy_lens = [m[1] for m in matched]
    residuals = [abs(m[0] - m[1]) for m in matched]

    records = []
    for i, (_t, _l, text) in enumerate(matched):
        records.append({
            "id": f"proxy-{i:04d}",
            "conversations": [
                {"from": "human", "value": text},
                {"from": "gpt", "value": "Sure -- here is a helpful response."},
            ],
        })
    # deterministic, schema-identical to the public file
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2))

    meta = {
        "purpose": "chat-heavy private-proxy prompt set, length-matched to the "
                   "public 128, for the public->private TPS gap probe (PR #44)",
        "source": {"repo": SHAREGPT_REPO, "file": SHAREGPT_FILE,
                   "path": str(sharegpt_path)},
        "seed": args.seed, "num": args.num, "tokenizer": args.tokenizer,
        "candidate_cap": args.candidate_cap,
        "candidates_in_band": len(cand_lens),
        "public_token_len_dist": dist(public_lens),
        "proxy_token_len_dist": dist(proxy_lens),
        "length_match_residual_tokens": {
            "max": max(residuals), "mean": round(statistics.mean(residuals), 2),
            "p90": sorted(residuals)[int(len(residuals) * .9)],
        },
        "dedup": "hard text-hash + 80-char prefix guard vs public; unique within proxy",
        "reasoning_markers_excluded": list(REASONING_MARKERS),
    }
    Path(str(out_path) + ".meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[build] wrote {out_path} ({len(records)} records)", flush=True)
    print(f"[build] public dist : {dist(public_lens)}", flush=True)
    print(f"[build] proxy  dist : {dist(proxy_lens)}", flush=True)
    print(f"[build] residual tok: {meta['length_match_residual_tokens']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
