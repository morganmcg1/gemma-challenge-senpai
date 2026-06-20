#!/usr/bin/env python3
"""Greedy-divergence between two decode_outputs.jsonl captures (same prompts/seed).

Both files are produced by the official decode_outputs.py at temperature=0 with
ignore_eos=true, so each prompt yields a fixed-length (output_len) greedy token
sequence. We align by prompt_token_sha256 (identical prompts) and report, per the
PR #788 contract, greedy divergence as EVIDENCE (not a pass/fail gate):

  * prompts compared, prompts byte-identical, prompts diverged
  * per-prompt greedy-prefix-match length (tokens before first mismatch); since
    greedy decode is autoregressive, the first mismatch is the only causally
    meaningful divergence point -- everything after is a different trajectory.
  * mean prefix-match fraction = mean(prefix_len / seq_len)
  * first-divergence position stats over diverged prompts (min/median/mean)

Usage: compare_divergence.py <ref.jsonl> <cand.jsonl> [out.json]
  ref  = bi0 control (bf16 lm_head)
  cand = candidate (int8/int4 lm_head)
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    by_prompt: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        by_prompt[row["prompt_token_sha256"]] = row
    return by_prompt


def prefix_match_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def main() -> int:
    ref_path, cand_path = Path(sys.argv[1]), Path(sys.argv[2])
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    ref, cand = load(ref_path), load(cand_path)

    common = sorted(set(ref) & set(cand))
    only_ref, only_cand = len(set(ref) - set(cand)), len(set(cand) - set(ref))

    n = 0
    n_identical = 0
    prefix_fracs: list[float] = []
    first_div: list[int] = []
    for ph in common:
        a = ref[ph]["completion_token_ids"]
        b = cand[ph]["completion_token_ids"]
        seqlen = min(len(a), len(b))
        if seqlen == 0:
            continue
        n += 1
        pm = prefix_match_len(a, b)
        prefix_fracs.append(pm / seqlen)
        if pm == seqlen and len(a) == len(b):
            n_identical += 1
        else:
            first_div.append(pm)

    summary = {
        "ref_file": str(ref_path),
        "cand_file": str(cand_path),
        "prompts_compared": n,
        "prompts_only_in_ref": only_ref,
        "prompts_only_in_cand": only_cand,
        "prompts_identical": n_identical,
        "prompts_diverged": n - n_identical,
        "frac_identical": round(n_identical / n, 4) if n else None,
        "frac_diverged": round((n - n_identical) / n, 4) if n else None,
        "mean_prefix_match_frac": round(statistics.fmean(prefix_fracs), 4) if prefix_fracs else None,
        "min_prefix_match_frac": round(min(prefix_fracs), 4) if prefix_fracs else None,
        "first_divergence_pos": {
            "min": min(first_div) if first_div else None,
            "median": statistics.median(first_div) if first_div else None,
            "mean": round(statistics.fmean(first_div), 2) if first_div else None,
        },
    }
    print(json.dumps(summary, indent=2))
    if out_path:
        out_path.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
