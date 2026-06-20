#!/usr/bin/env python
"""Greedy token-identity comparison for the surgattn 2D-vs-3D check (PR #785).

Reads the per-prompt ``decode_outputs.jsonl`` from the control arm (surgattn ON,
force-2D) and the variant arm (surgattn OFF, kernel-gate picks 3D split-KV on the
M=1 forwards) and reports whether the emitted greedy token streams are identical,
plus an exact divergence rate at the record and token level.

Records are matched by ``id``. Each record carries ``completion_token_ids`` and
``completion_token_sha256``; both arms decode at temperature=0, ignore_eos=True,
max_tokens=512, so a faithful greedy-identity check is a direct per-id token-list
comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rows[str(r["id"])] = r
    return rows


def first_divergence(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out", default=None, help="optional JSON summary path")
    args = ap.parse_args()

    ctrl = load(Path(args.control))
    var = load(Path(args.variant))

    common = sorted(set(ctrl) & set(var))
    only_ctrl = sorted(set(ctrl) - set(var))
    only_var = sorted(set(var) - set(ctrl))

    n = len(common)
    identical_records = 0
    diverging = []  # (id, first_div_pos, len_ctrl, len_var, n_tok_diff)
    total_tokens = 0
    total_tok_diff = 0

    for rid in common:
        a = list(ctrl[rid]["completion_token_ids"])
        b = list(var[rid]["completion_token_ids"])
        m = min(len(a), len(b))
        total_tokens += m
        ndiff = sum(1 for i in range(m) if a[i] != b[i]) + abs(len(a) - len(b))
        total_tok_diff += ndiff
        if a == b:
            identical_records += 1
        else:
            diverging.append((rid, first_divergence(a, b), len(a), len(b), ndiff))

    rec_div_rate = (len(diverging) / n) if n else 0.0
    tok_div_rate = (total_tok_diff / total_tokens) if total_tokens else 0.0

    summary = {
        "control_file": args.control,
        "variant_file": args.variant,
        "n_records_control": len(ctrl),
        "n_records_variant": len(var),
        "n_common": n,
        "only_in_control": only_ctrl,
        "only_in_variant": only_var,
        "identical_records": identical_records,
        "diverging_records": len(diverging),
        "record_divergence_rate": rec_div_rate,
        "total_tokens_compared": total_tokens,
        "total_token_diffs": total_tok_diff,
        "token_divergence_rate": tok_div_rate,
        "byte_exact_greedy_identity": (len(diverging) == 0 and not only_ctrl and not only_var),
        "diverging_detail": [
            {"id": rid, "first_divergence_pos": pos, "len_control": la,
             "len_variant": lb, "n_token_diffs": nd}
            for (rid, pos, la, lb, nd) in diverging[:50]
        ],
    }

    print(json.dumps(summary, indent=2))
    print(
        f"\nGREEDY-IDENTITY: {'PASS (byte-exact)' if summary['byte_exact_greedy_identity'] else 'DIVERGENT'} "
        f"| records {identical_records}/{n} identical "
        f"| record_div_rate={rec_div_rate:.6f} token_div_rate={tok_div_rate:.8f}",
        flush=True,
    )
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
