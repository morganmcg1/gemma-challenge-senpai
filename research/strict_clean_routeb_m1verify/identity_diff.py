#!/usr/bin/env python
"""PR #746: byte-exact greedy-identity diff of each captured arm vs the served
M=1 AR reference, using the official greedy comparator (scripts.local_validation
.greedy_gate.compare -> the shared check_greedy_identity rule).

Two questions this answers from already-saved decode_outputs.jsonl (no GPU):
  1. How NON-byte-exact is the batched M=K+1 verify fire (route-a / #730)? ->
     num_divergent/128 + divergence onset. This is the G1 private-repro DQ source
     route-b exists to remove.
  2. Is the M=1 AR greedy output kernel-stable (batchinv-ON arref vs fast-kernel
     arref_fastkern identical 128/128)? If yes, route-b's fast-kernel M=1 verify
     accepts exactly the AR tokens -> route-b is byte-exact by shape identity.

Usage:
  .venv/bin/python -m research.strict_clean_routeb_m1verify.identity_diff \
      --root research/strict_clean_routeb_m1verify --out <root>/identity.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.local_validation import greedy_gate  # noqa: E402

OUTPUT_LEN = 512


def _diff(reference: Path, candidate: Path) -> dict:
    if not reference.exists() or not candidate.exists():
        return {"error": f"missing ref={reference.exists()} cand={candidate.exists()}"}
    report = greedy_gate.compare(reference, candidate)
    summ = greedy_gate.onset_summary(report)
    n_id, n_div = summ["num_identical"], summ["num_divergent"]
    total = n_id + n_div
    return {
        "num_identical": n_id, "num_divergent": n_div, "total": total,
        "byte_exact_128": bool(n_div == 0 and total > 0),
        "identity_rate": (n_id / total) if total else None,
        "onset_line": greedy_gate.onset_line(summ, OUTPUT_LEN),
        "onset_min": summ.get("onset_min"), "onset_median": summ.get("onset_median"),
        "onset_max": summ.get("onset_max"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="research/strict_clean_routeb_m1verify")
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    root = Path(args.root)
    ref = root / "arref" / "decode_outputs.jsonl"  # served M=1 AR greedy reference

    results: dict = {"reference": str(ref)}
    # 1. kernel-stability of M=1 greedy: fast-kernel arref vs batchinv arref.
    fk = root / "arref_fastkern" / "decode_outputs.jsonl"
    if fk.exists():
        results["arref_fastkern_vs_arref"] = _diff(ref, fk)
    # 2. each batched arm vs the AR reference (the route-a non-exactness).
    for k in args.ks:
        cand = root / f"batched_k{k}" / "decode_outputs.jsonl"
        results[f"batched_k{k}_vs_arref"] = _diff(ref, cand)

    text = json.dumps(results, indent=2)
    if args.out:
        Path(args.out).write_text(text)
        print(f"[identity] wrote {args.out}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
