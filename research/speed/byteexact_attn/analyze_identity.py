"""PR #496 post-hoc identity analysis: reconcile the surprising serve-level
identity numbers (all ~0.5-0.68) against #488's clean 0.9763 surgical-vs-full_flag.

Distinguishes:
  (1) within-arm warm determinism (r1 vs r2 same arm) -> is the stack run-to-run
      deterministic this session? (#488 saw 1.000)
  (2) cross-config warm (last-round vs last-round) for every pair, esp.
      byteexact_vs_surgical (candidate vs the accepted 357.6 rung; NOT in harness).
  (3) first-divergence structure: common-prefix length before the first flip.
      Long common prefix + single flip that cascades = bf16 ULP-tie (#461 class);
      immediate/early divergence across most seqs = a real per-step difference.

LOCAL CPU-only over the saved decode jsonls. No model, no GPU, no serve.
"""
from __future__ import annotations
import json
import statistics
from pathlib import Path

RUN = Path(__file__).resolve().parent / "serve_run"
ARMS = ["deployed", "surgical", "byteexact", "full_flag", "byteexact_ref", "full_flag_ref"]


def load_round(arm: str, rnd: int) -> dict[str, list[int]]:
    f = RUN / arm / f"decode_round{rnd:02d}.jsonl"
    seqs: dict[str, list[int]] = {}
    if not f.exists():
        return seqs
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        key = str(o.get("id"))
        toks = o.get("completion_token_ids")
        if isinstance(toks, list):
            seqs[key] = [int(t) for t in toks]
    return seqs


def n_rounds(arm: str) -> int:
    return len(list((RUN / arm).glob("decode_round*.jsonl")))


def compare(sa: dict[str, list[int]], sb: dict[str, list[int]]) -> dict:
    common = sorted(set(sa) & set(sb))
    total = matched = nflip = 0
    prefix_lens = []   # common-prefix length for flipped seqs
    first_div_positions = []
    for k in common:
        ta, tb = sa[k], sb[k]
        n = min(len(ta), len(tb))
        # common prefix
        p = 0
        while p < n and ta[p] == tb[p]:
            p += 1
        seq_flips = sum(1 for i in range(n) if ta[i] != tb[i])
        total += n
        matched += n - seq_flips
        if seq_flips or len(ta) != len(tb):
            nflip += 1
            prefix_lens.append(p)
            first_div_positions.append(p)
    return {
        "n_prompts": len(common),
        "rate": (matched / total) if total else None,
        "n_flipped_seqs": nflip,
        "median_common_prefix_of_flipped": (statistics.median(prefix_lens) if prefix_lens else None),
        "min_common_prefix_of_flipped": (min(prefix_lens) if prefix_lens else None),
        "max_common_prefix_of_flipped": (max(prefix_lens) if prefix_lens else None),
        "first_div_positions_sorted": sorted(first_div_positions),
    }


def main():
    print("=" * 78)
    print("(1) WITHIN-ARM warm determinism  (r1 vs r2 same arm; #488 saw 1.000)")
    print("=" * 78)
    for arm in ARMS:
        nr = n_rounds(arm)
        if nr >= 3:
            r1, r2 = load_round(arm, 1), load_round(arm, 2)
            c = compare(r1, r2)
            print(f"  {arm:16s} r1-vs-r2: rate={c['rate']:.6f} flipped={c['n_flipped_seqs']}/{c['n_prompts']}")
        elif nr == 2:
            r0, r1 = load_round(arm, 0), load_round(arm, 1)
            c = compare(r0, r1)
            print(f"  {arm:16s} r0-vs-r1: rate={c['rate']:.6f} flipped={c['n_flipped_seqs']}/{c['n_prompts']}  (ref arm, only 2 rounds)")

    print()
    print("=" * 78)
    print("(2) CROSS-CONFIG warm (last warm round of each arm)")
    print("=" * 78)
    last = {}
    for arm in ARMS:
        nr = n_rounds(arm)
        last[arm] = load_round(arm, nr - 1)
    pairs = [
        ("byteexact", "surgical", "CANDIDATE vs accepted 357.6 rung (both byte-exact attn + fast matmul, cudagraph M=8)"),
        ("byteexact", "full_flag", "candidate vs batch-inv ground truth (served-vs-served)"),
        ("surgical", "full_flag", "#488 reported 0.9763 here (both 2D attn, matmul-axis only)"),
        ("deployed", "surgical", "#488 reported 0.4564 (attention axis: 3D adaptive vs 2D)"),
        ("byteexact", "deployed", "candidate vs fast non-exact 3D adaptive"),
        ("deployed", "full_flag", "non-exact control"),
    ]
    for a, b, note in pairs:
        c = compare(last[a], last[b])
        print(f"  {a:11s} vs {b:13s} rate={c['rate']:.6f} flipped={c['n_flipped_seqs']}/{c['n_prompts']} "
              f"prefix(min/med/max)={c['min_common_prefix_of_flipped']}/{c['median_common_prefix_of_flipped']}/{c['max_common_prefix_of_flipped']}")
        print(f"        ^ {note}")
        print(f"        first_div_positions={c['first_div_positions_sorted']}")

    print()
    print("=" * 78)
    print("(3) CROSS-ROUND robustness: byteexact(r2) vs surgical(r1) and (r2) — pairing sanity")
    print("=" * 78)
    for ra in (0, 1, 2):
        for rb in (0, 1, 2):
            c = compare(load_round("byteexact", ra), load_round("surgical", rb))
            print(f"  byteexact(r{ra}) vs surgical(r{rb}): rate={c['rate']:.6f} flipped={c['n_flipped_seqs']}/{c['n_prompts']}")


if __name__ == "__main__":
    main()
