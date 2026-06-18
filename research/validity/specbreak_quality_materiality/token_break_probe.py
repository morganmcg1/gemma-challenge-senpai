#!/usr/bin/env python
"""Spec-vs-AR greedy token-break-rate probe (PR #682, wirbel).

Confirms kanna #673's int4-Marlin verify-width break is ACTIVE on THIS config
(int4_g128_lmhead body, /tmp/qat-assistant K=6 MTP drafter, dev307, BI=1) before
the quality delta is interpreted. The card requires a NON-ZERO token break-rate;
if the break is absent here the quality census is inconclusive.

Mechanism metric (directly comparable to the strict-#319 census #607 / kanna #673
/ #616): greedy-decode the SAME 128 sharegpt prompts x 512 tokens (paths defaults,
ignore_eos) on each arm, capturing completion_token_ids, then diff position-by-
position. AR (k=0) is the M=1 reference; SPEC (k=6) is the candidate. A break is a
position where spec_tok != ar_tok over the COMMON prefix length.

  capture : call harness.capture_decode against a running endpoint -> arm jsonl.
  diff    : load ar.jsonl + spec.jsonl, compute per-token break-rate + per-seq
            divergence + Wilson CIs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))
from scripts.local_validation import harness, paths  # noqa: E402

DEV307_VENV = Path("/tmp/senpai-venvs/a341b8bdf5ec1fe0/bin/python")
HERE = Path(__file__).resolve().parent


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def _load(jsonl: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        rows[int(o["index"])] = o
    return rows


def do_capture(args) -> int:
    out_file = Path(args.out)
    summary_file = out_file.with_suffix(".summary.json")
    summ = harness.capture_decode(
        DEV307_VENV, base_url=args.base_url, model=args.model,
        out_file=out_file, summary_file=summary_file,
        num_prompts=args.num_prompts, output_len=args.output_len,
    )
    print(f"[capture] arm={args.arm} wrote {out_file} "
          f"n_completion_tokens={summ.get('num_completion_tokens')}", flush=True)
    return 0


def do_diff(args) -> int:
    ar = _load(Path(args.ar))
    spec = _load(Path(args.spec))
    common = sorted(set(ar) & set(spec))
    if not common:
        raise SystemExit("no common prompt indices between ar and spec captures")

    tot_pos = 0          # compared positions (common prefix length, summed)
    break_pos = 0        # positions where spec_tok != ar_tok
    n_seq = 0
    n_seq_divergent = 0  # sequences with >=1 break
    first_break_positions: list[int] = []
    len_mismatch = 0
    per_seq: list[dict[str, Any]] = []
    for idx in common:
        a = ar[idx].get("completion_token_ids") or []
        s = spec[idx].get("completion_token_ids") or []
        L = min(len(a), len(s))
        if len(a) != len(s):
            len_mismatch += 1
        n_seq += 1
        nb = 0
        first = None
        for i in range(L):
            if a[i] != s[i]:
                nb += 1
                if first is None:
                    first = i
        tot_pos += L
        break_pos += nb
        if nb > 0:
            n_seq_divergent += 1
            first_break_positions.append(first)
        per_seq.append({"index": idx, "len_ar": len(a), "len_spec": len(s),
                        "compared": L, "breaks": nb, "first_break": first})

    tok_rate = break_pos / tot_pos if tot_pos else float("nan")
    tok_lo, tok_hi = wilson(break_pos, tot_pos)
    seq_rate = n_seq_divergent / n_seq if n_seq else float("nan")
    seq_lo, seq_hi = wilson(n_seq_divergent, n_seq)
    fbp_sorted = sorted(first_break_positions)
    median_first = (fbp_sorted[len(fbp_sorted) // 2] if fbp_sorted else None)

    out = {
        "analysis_only": 1, "official_tps": 0, "fires": 0,
        "prompt_set": "sharegpt_128_x512_ignore_eos (strict-#319 census set)",
        "n_seq": n_seq, "n_seq_divergent": n_seq_divergent,
        "specbreak_seq_divergence_rate": seq_rate,
        "specbreak_seq_divergence_ci95": [seq_lo, seq_hi],
        "total_positions_compared": tot_pos, "break_positions": break_pos,
        "specbreak_token_break_rate": tok_rate,
        "specbreak_token_break_rate_ci95": [tok_lo, tok_hi],
        "len_mismatch_seqs": len_mismatch,
        "median_first_break_position": median_first,
        "break_present": bool(break_pos > 0),
        "per_seq": per_seq,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("=" * 64)
    print(f"TOKEN-BREAK PROBE  (spec k=6 vs ar k=0, int4_g128_lmhead, dev307, BI=1)")
    print(f"  token break-rate = {tok_rate*100:.4f}%  "
          f"({break_pos}/{tot_pos})  CI95 [{tok_lo*100:.4f},{tok_hi*100:.4f}]%")
    print(f"  seq divergence   = {seq_rate*100:.2f}%  "
          f"({n_seq_divergent}/{n_seq})  CI95 [{seq_lo*100:.2f},{seq_hi*100:.2f}]%")
    print(f"  median first-break position = {median_first}  (of {args.output_len} tok)")
    print(f"  break_present = {out['break_present']}")
    print(f"  wrote {args.out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture")
    c.add_argument("--arm", choices=["ar", "spec"], required=True)
    c.add_argument("--base-url", required=True)
    c.add_argument("--model", default="gemma-4-e4b-it")
    c.add_argument("--out", required=True)
    c.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    c.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    c.set_defaults(func=do_capture)

    d = sub.add_parser("diff")
    d.add_argument("--ar", required=True)
    d.add_argument("--spec", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    d.set_defaults(func=do_diff)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
