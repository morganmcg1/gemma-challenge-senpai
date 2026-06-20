#!/usr/bin/env python3
"""PR #797 identity-class diff: STACK (surgattn-3D) greedy token_ids vs CONTROL (force-2D).

Complements research/_int8head_smoke/compare_divergence.py (prompt-level + first-divergence
locus) with the two token-level numbers the #784 gate wants reported alongside it:

  * frac_tokens_diff   -- aligned token positions that differ / all aligned positions.
                          Counts everything after the first divergence too (post-divergence
                          the two greedy trajectories are genuinely different sequences),
                          so this is an UPPER bound on "how much output text changed".
  * first_token_flips  -- prompts whose position-0 (first generated) token differs. This is
                          the purest identity signal, unconfounded by autoregressive drift:
                          a single kernel-path ULP flip on the very first decode step.

Both files come from the official decode_outputs.py at temperature=0, ignore_eos=true,
fixed output_len, aligned by prompt_token_sha256.

Usage: token_identity_diff.py <control.jsonl> <stack.jsonl> [out.json]
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


def first_div(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def main() -> int:
    ctrl_path, stack_path = Path(sys.argv[1]), Path(sys.argv[2])
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    ctrl, stack = load(ctrl_path), load(stack_path)
    common = sorted(set(ctrl) & set(stack))

    n_prompts = 0
    n_prompts_div = 0
    n_first_token_flip = 0
    total_positions = 0
    total_mismatch = 0
    first_div_pos: list[int] = []
    for ph in common:
        a = ctrl[ph]["completion_token_ids"]
        b = stack[ph]["completion_token_ids"]
        seqlen = min(len(a), len(b))
        if seqlen == 0:
            continue
        n_prompts += 1
        total_positions += seqlen
        mism = sum(1 for i in range(seqlen) if a[i] != b[i])
        total_mismatch += mism
        if a[0] != b[0]:
            n_first_token_flip += 1
        fd = first_div(a, b)
        if fd != seqlen or len(a) != len(b):
            n_prompts_div += 1
            first_div_pos.append(fd)

    summary = {
        "control_file": str(ctrl_path),
        "stack_file": str(stack_path),
        "prompts_compared": n_prompts,
        "prompts_diverged": n_prompts_div,
        "frac_prompts_diff": round(n_prompts_div / n_prompts, 4) if n_prompts else None,
        "first_token_flips": n_first_token_flip,
        "frac_first_token_flip": round(n_first_token_flip / n_prompts, 4) if n_prompts else None,
        "total_aligned_positions": total_positions,
        "total_token_mismatches": total_mismatch,
        "frac_tokens_diff": round(total_mismatch / total_positions, 6) if total_positions else None,
        "first_divergence_pos": {
            "min": min(first_div_pos) if first_div_pos else None,
            "median": statistics.median(first_div_pos) if first_div_pos else None,
            "mean": round(statistics.fmean(first_div_pos), 2) if first_div_pos else None,
            "max": max(first_div_pos) if first_div_pos else None,
        },
    }
    print(json.dumps(summary, indent=2))
    if out_path:
        out_path.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
