#!/usr/bin/env python
"""Strict-#319 served greedy-token-identity check.

Compares completion_token_ids from two decode_outputs.jsonl captures (the same
128-prompt protocol the official harness decode_capture uses: /v1/completions,
integer-token prompt, temperature=0, max_tokens=512, ignore_eos,
return_token_ids). Candidate = speculative-ON serve; reference = spec-OFF M=1 AR
serve on the SAME engine/BI. A prompt is byte-exact iff the candidate completion
token id list equals the reference list element-for-element.

Reports N_exact/total and per-prompt first-divergence index, so a BI=0 break is
characterised (where the first greedy argmax flip cascades).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        key = str(r.get("dataset_index", r.get("id")))
        rows[key] = r
    return rows


def first_div(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return -1  # identical


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="spec-ON decode_outputs.jsonl")
    ap.add_argument("--reference", required=True, help="spec-OFF decode_outputs.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    cand = load(Path(args.candidate))
    ref = load(Path(args.reference))
    keys = sorted(set(cand) & set(ref), key=lambda k: int(k))
    missing = sorted(set(cand) ^ set(ref))

    n_total = len(keys)
    n_exact = 0
    n_len_mismatch = 0
    divs: list[dict] = []
    for k in keys:
        a = cand[k]["completion_token_ids"]
        b = ref[k]["completion_token_ids"]
        fd = first_div(a, b)
        if fd == -1:
            n_exact += 1
        else:
            if len(a) != len(b):
                n_len_mismatch += 1
            divs.append(
                {
                    "key": k,
                    "id": cand[k].get("id"),
                    "first_div_index": fd,
                    "len_cand": len(a),
                    "len_ref": len(b),
                    "cand_tok": a[fd] if fd < len(a) else None,
                    "ref_tok": b[fd] if fd < len(b) else None,
                }
            )

    fd_indices = [d["first_div_index"] for d in divs]
    summary = {
        "label": args.label,
        "n_total": n_total,
        "n_exact": n_exact,
        "identity": f"{n_exact}/{n_total}",
        "n_diverged": len(divs),
        "n_len_mismatch": n_len_mismatch,
        "keys_only_in_one": missing,
        "first_div_min": min(fd_indices) if fd_indices else None,
        "first_div_max": max(fd_indices) if fd_indices else None,
        "first_div_mean": (sum(fd_indices) / len(fd_indices)) if fd_indices else None,
        "diverged_examples": divs[:20],
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(
        f"[{args.label}] identity={summary['identity']}  diverged={len(divs)}  "
        f"len_mismatch={n_len_mismatch}  first_div(min/mean/max)="
        f"{summary['first_div_min']}/{summary['first_div_mean']}/{summary['first_div_max']}",
        flush=True,
    )
    if missing:
        print(f"[WARN] keys present in only one capture: {missing}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
