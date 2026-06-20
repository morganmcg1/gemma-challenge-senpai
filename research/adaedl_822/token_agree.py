#!/usr/bin/env python3
"""Per-prompt greedy token-agreement between two decode_outputs.jsonl captures.

For the AdaEDL early-stop quality check (#822). Greedy speculative decode emits
the target's argmax regardless of how many draft tokens were proposed, so an
early-stopped run should match the full-K (inf) run up to the FP noise floor of
the serving stack (memory pr801: 512-tok A10G greedy is FP-noise-dominated, so
even same-config AR-vs-AR is far below 128/128 identical). We therefore report
agreement RELATIVE to a noise floor (two same-config reps), not absolute 128/128.

Matches rows by (id, dataset_index). Reports, per pair:
  prompts_compared, exact_match_prompts (completion_token_sha256 equal),
  mean first-divergence index, mean per-prompt token-agreement fraction
  (matched prefix is not required; we count position-wise equal tokens over the
  min length), and total positionwise agreement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    rows = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (str(r.get("id")), r.get("dataset_index"))
            rows[key] = r
    return rows


def first_divergence(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return -1  # identical


def positionwise_agree(a: list[int], b: list[int]) -> tuple[int, int]:
    n = min(len(a), len(b))
    eq = sum(1 for i in range(n) if a[i] == b[i])
    return eq, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file_a")
    ap.add_argument("file_b")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    A = load(Path(args.file_a))
    B = load(Path(args.file_b))
    keys = sorted(set(A) & set(B))
    if not keys:
        print("NO COMMON PROMPTS")
        return 1

    exact = 0
    div_idxs = []
    tot_eq = 0
    tot_n = 0
    per_prompt_frac = []
    for k in keys:
        a = A[k]["completion_token_ids"]
        b = B[k]["completion_token_ids"]
        if A[k].get("completion_token_sha256") == B[k].get("completion_token_sha256"):
            exact += 1
        di = first_divergence(a, b)
        div_idxs.append(len(a) if di == -1 else di)
        eq, n = positionwise_agree(a, b)
        tot_eq += eq
        tot_n += n
        per_prompt_frac.append(eq / n if n else 1.0)

    lbl = f"[{args.label}] " if args.label else ""
    print(f"{lbl}{Path(args.file_a).parent.name}/{Path(args.file_a).name}  vs  "
          f"{Path(args.file_b).parent.name}/{Path(args.file_b).name}")
    print(f"  prompts_compared      = {len(keys)}")
    print(f"  exact_match_prompts   = {exact}/{len(keys)}  ({100*exact/len(keys):.1f}%)")
    print(f"  mean_first_divergence = {sum(div_idxs)/len(div_idxs):.1f} tokens "
          f"(higher = later divergence)")
    print(f"  positionwise_agree    = {tot_eq}/{tot_n}  ({100*tot_eq/tot_n:.2f}%)")
    print(f"  mean_per_prompt_frac  = {100*sum(per_prompt_frac)/len(per_prompt_frac):.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
