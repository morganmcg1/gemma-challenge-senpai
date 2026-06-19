#!/usr/bin/env python
"""Anatomy of an ar-vs-fast greedy divergence for PR #720.

The dual-substrate profiler scores **full-stream** sha256 identity, but every leg
is decoded with ``ignore_eos=True`` to a fixed ``output_len`` cap (the benchmark's
forced-length TPS protocol). So a stream is: [meaningful answer] <end_of_turn>
[unconstrained forced-length garbage tail]. Two faithful engines can produce the
SAME answer yet different garbage tails -> full-stream sha differs (0/N) while the
#319-relevant answer is identical.

This tool aligns each prompt's reference vs current token streams and reports, per
prompt and in aggregate:
  - first_div   : index of first differing token (None if identical)
  - eot_ref     : index of the first <end_of_turn>/<eos> in the REFERENCE stream
                  (the natural answer boundary; tokens after it are forced garbage)
  - pre_eot     : was the first divergence BEFORE the natural answer boundary?
                  (True => a REAL answer-level break; False => garbage-tail only)
  - prefix_len  : number of leading tokens that agree

Verdict on the COMPARISON (not the config):
  ANSWER_IDENTICAL_TAIL_DIVERGES : 0 prompts diverge before their EOT -> the
       full-stream break is a forced-length tail artifact; answers are identical.
  ANSWER_LEVEL_BREAK             : >=1 prompt diverges before EOT -> a real break
       in the meaningful answer region (count + indices reported).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(jsonl: Path) -> dict[int, list[int]]:
    out = {}
    for line in jsonl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out[int(r["dataset_index"])] = r["completion_token_ids"]
    return out


def first_div(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def first_eot(ids: list[int], eot_ids: set[int]) -> int | None:
    for i, t in enumerate(ids):
        if t in eot_ids:
            return i
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", type=Path, required=True, help="reference jsonl (ar_ref)")
    ap.add_argument("--cur", type=Path, required=True, help="current jsonl (dev307_fast / eager_floor)")
    ap.add_argument("--tokenizer", default="google/gemma-4-E4B-it")
    ap.add_argument("--eot-ids", default="", help="comma ints to add to EOT set (besides tokenizer specials)")
    ap.add_argument("--show", type=int, default=3, help="decode this many pre-EOT break examples to text")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    eot_ids: set[int] = set()
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        if tok.eos_token_id is not None:
            eot_ids.add(int(tok.eos_token_id))
        for name in ("<end_of_turn>", "<eos>"):
            tid = tok.convert_tokens_to_ids(name)
            if isinstance(tid, int) and tid >= 0:
                eot_ids.add(tid)
    except Exception as e:
        print(f"[warn] tokenizer load failed ({e}); EOT set from --eot-ids only")
    if args.eot_ids:
        eot_ids |= {int(x) for x in args.eot_ids.split(",") if x.strip()}

    ref = load(args.ref)
    cur = load(args.cur)
    keys = sorted(set(ref) & set(cur))

    rows = []
    pre_eot_breaks = []
    for k in keys:
        a, b = ref[k], cur[k]
        fd = first_div(a, b)
        eot = first_eot(a, eot_ids)
        pre = (fd is not None) and (eot is not None) and (fd <= eot)
        # if no EOT found in ref, treat whole stream as "answer" (conservative -> pre=real)
        if fd is not None and eot is None:
            pre = True
        rows.append({"dataset_index": k, "first_div": fd, "eot_ref": eot,
                     "prefix_len": (fd if fd is not None else len(a)), "pre_eot": pre})
        if pre:
            pre_eot_breaks.append(k)

    n = len(keys)
    n_identical = sum(1 for r in rows if r["first_div"] is None)
    n_pre = len(pre_eot_breaks)
    n_tail_only = n - n_identical - n_pre
    eots = [r["eot_ref"] for r in rows if r["eot_ref"] is not None]
    prefixes = [r["prefix_len"] for r in rows]
    verdict = ("ANSWER_LEVEL_BREAK" if n_pre > 0 else
               ("ALL_IDENTICAL" if n_identical == n else "ANSWER_IDENTICAL_TAIL_DIVERGES"))

    report = {
        "ref": str(args.ref), "cur": str(args.cur),
        "eot_ids": sorted(eot_ids), "n_prompts": n,
        "n_identical_full": n_identical,
        "n_pre_eot_break": n_pre,
        "n_tail_only_break": n_tail_only,
        "pre_eot_break_indices": pre_eot_breaks[:20],
        "median_eot_ref": (sorted(eots)[len(eots)//2] if eots else None),
        "min_prefix_len": min(prefixes) if prefixes else None,
        "median_prefix_len": (sorted(prefixes)[len(prefixes)//2] if prefixes else None),
        "max_prefix_len": max(prefixes) if prefixes else None,
        "verdict": verdict,
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    # decode a few examples (prefer pre-EOT breaks; else earliest tail breaks)
    if tok is not None and args.show > 0:
        show_keys = pre_eot_breaks[:args.show] if pre_eot_breaks else \
            [r["dataset_index"] for r in sorted(rows, key=lambda r: (r["first_div"] is None, r["first_div"] or 1e9))[:args.show]]
        for k in show_keys:
            a, b = ref[k], cur[k]
            fd = first_div(a, b)
            eot = first_eot(a, eot_ids)
            lo = max(0, (fd or 0) - 12)
            hi = (fd or 0) + 12
            print(f"\n----- dataset_index={k} first_div={fd} eot_ref={eot} (pre_eot={fd is not None and eot is not None and fd<=eot}) -----")
            print(f"  REF[{lo}:{hi}] = {a[lo:hi]}")
            print(f"  CUR[{lo}:{hi}] = {b[lo:hi]}")
            print(f"  REF text around break: {tok.decode(a[lo:hi])!r}")
            print(f"  CUR text around break: {tok.decode(b[lo:hi])!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
