#!/usr/bin/env python
"""Build a token-frequency histogram over the verifier's greedy OUTPUT tokens.

Hypothesis under test (PR #48): adding a static additive frequency bias to the
top-K most common continuation tokens, applied to the speculative DRAFTER's
logits, shifts the drafter toward the corpus mode and raises acceptance rate.

The most favourable corpus for that claim is the distribution the drafter is
actually trying to predict: the verifier's own greedy continuation tokens (not
the prompt tokens). We therefore histogram ``completion_token_ids`` from the
committed served greedy reference for the target submission.

Output: research/drafter_freq_bias/freq_top_tokens.json
  {
    "source": <path>, "num_records": N, "total_output_tokens": T,
    "vocab_size": V,
    "top": [[token_id, count], ...]   # descending, up to --top-k rows
    "top_k_token_ids": [token_id, ...]  # convenience, length == requested K
    "decoded_preview": [[token_id, count, repr], ...]  # first 40, for the report
  }
LOCAL analysis only; no GPU, no HF Job.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_REF = (
    REPO
    / "research/greedy_reference"
    / "workspace__senpai__target__submissions__fa2sw_precache_kenyan__google__gemma-4-E4B-it"
    / "decode_outputs.jsonl"
)
DEFAULT_OUT = REPO / "research/drafter_freq_bias/freq_top_tokens.json"
DEFAULT_TOKENIZER = "/tmp/osoi5-v0-baked"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", default=str(DEFAULT_REF), help="served greedy decode_outputs.jsonl")
    ap.add_argument("--field", default="completion_token_ids")
    ap.add_argument("--top-k", type=int, default=1000, help="how many ranked rows to persist")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--include-special",
        action="store_true",
        help="keep special/control tokens (<eos>, <end_of_turn>, ...) in the boost set. "
        "Default EXCLUDES them: boosting the drafter toward <eos>/<turn> makes it propose "
        "premature stops the verifier rejects, which would unfairly doom the hypothesis.",
    )
    args = ap.parse_args()

    ref = Path(args.reference)
    counter: collections.Counter[int] = collections.Counter()
    n_rec = 0
    for line in ref.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        toks = rec.get(args.field)
        if not isinstance(toks, list):
            continue
        counter.update(int(t) for t in toks)
        n_rec += 1
    total = sum(counter.values())

    # Special / control ids to drop unless --include-special. Prefer the tokenizer's
    # own registry; fall back to the known gemma-4 control ids (<pad>0 <eos>1 <bos>2
    # <start_of_turn>105 <end_of_turn>106) if transformers is unavailable.
    tok = None
    try:
        from transformers import AutoTokenizer  # type: ignore

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        vocab_size = int(getattr(tok, "vocab_size", 0)) or (max(counter) + 1)
        special_ids = set(int(i) for i in (getattr(tok, "all_special_ids", []) or []))
    except Exception as exc:  # noqa: BLE001
        vocab_size = max(counter) + 1
        special_ids = set()
        print(f"[freq] tokenizer unavailable ({exc!r}); using fallback special ids")
    special_ids |= {0, 1, 2, 105, 106}

    excluded = []
    if not args.include_special:
        for sid in sorted(special_ids):
            if sid in counter:
                excluded.append([int(sid), int(counter.pop(sid))])

    top = counter.most_common(args.top_k)
    decoded_preview = None
    if tok is not None:
        decoded_preview = [[tid, c, tok.decode([tid])] for tid, c in top[:40]]
        excluded = [[sid, c, tok.decode([sid])] for sid, c in excluded]

    out = {
        "source": str(ref),
        "field": args.field,
        "num_records": n_rec,
        "total_output_tokens": total,
        "unique_tokens": len(counter),
        "vocab_size": vocab_size,
        "include_special": args.include_special,
        "excluded_special": excluded,
        "top": [[int(t), int(c)] for t, c in top],
        "top_k_token_ids": [int(t) for t, _ in top[: args.top_k]],
        "decoded_preview": decoded_preview,
        "cumulative_mass_at": {
            str(k): round(sum(c for _, c in top[:k]) / total, 4)
            for k in (100, 500, 1000)
            if k <= len(top)
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print(f"[freq] records={n_rec} total_output_tokens={total} unique={len(counter)} vocab={vocab_size}")
    print(f"[freq] cumulative output-mass covered by top-K: {out['cumulative_mass_at']}")
    if decoded_preview:
        print("[freq] top-20 output tokens (id, count, repr):")
        for tid, c, rep in decoded_preview[:20]:
            print(f"        {tid:>7d}  {c:>6d}  {rep!r}")
    print(f"[freq] wrote -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
