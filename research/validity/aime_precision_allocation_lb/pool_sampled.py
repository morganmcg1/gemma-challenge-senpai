#!/usr/bin/env python
"""PR #659 Phase-2 pooling: combine per-seed sampled AIME jsonls into one Wilson CI.

The greedy int8-on-locus crossing (0.45, +3/60 over int4) is noise-dominated: its n=60
Wilson CI overlaps both int4 0.400 and the 0.420 bar. Phase 2 adds statistical power by
sampling each item under K independent seeds (temp 1.0 / top_p 0.95 / top_k 64, per
generation_config.json), pooling K*60 (item,seed) outcomes for a ~sqrt(K)-tighter CI.

Usage:
  pool_sampled.py mix_int8_L14-27_s12345 mix_int8_L14-27_s23456 ...   # body-names
Reports: pooled acc + Wilson 95%/90%, per-seed breakdown, and per-item cross-seed
pass-rate (how many of K seeds solved each item) = the consistency picture greedy can't show.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
BAR = 0.4200
INT4 = 0.4000
BF16 = 0.4667


def wilson(k: int, n: int, z: float) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main(names: list[str]) -> int:
    per_item: dict[str, list[int]] = defaultdict(list)
    tot_k = tot_n = 0
    print(f"{'seed cell':<32} {'n':>3} {'correct':>7} {'acc':>6}  Wilson95")
    for name in names:
        path = RES / f"{name}_aime.jsonl"
        if not path.exists():
            print(f"{name:<32}  MISSING {path}")
            continue
        recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        recs = [r for r in recs if not r.get("error")]
        k = sum(1 for r in recs if r.get("correct"))
        n = len(recs)
        for r in recs:
            per_item[str(r["id"])].append(1 if r.get("correct") else 0)
        lo, hi = wilson(k, n, 1.96)
        acc = k / n if n else 0.0
        print(f"{name:<32} {n:>3} {k:>7} {acc:>6.3f}  [{lo:.3f},{hi:.3f}]")
        tot_k += k
        tot_n += n
    if tot_n == 0:
        print("no data")
        return 1
    pooled = tot_k / tot_n
    lo95, hi95 = wilson(tot_k, tot_n, 1.96)
    lo90, hi90 = wilson(tot_k, tot_n, 1.645)
    print("-" * 70)
    print(f"POOLED  {tot_k}/{tot_n}  acc={pooled:.4f}")
    print(f"  Wilson95 [{lo95:.4f}, {hi95:.4f}]   Wilson90 [{lo90:.4f}, {hi90:.4f}]")
    print(f"  bar=0.420  int4=0.400  bf16=0.4667")
    print(f"  clears_bar(point)={pooled >= BAR}  CI95_lo_clears_bar={lo95 >= BAR}  "
          f"CI90_lo_clears_bar={lo90 >= BAR}")
    print(f"  CI95_lo_above_int4={lo95 >= INT4}  (separates recovery from pure-int4 noise)")
    # per-item cross-seed pass rate (consistency)
    n_items = len(per_item)
    full = sum(1 for v in per_item.values() if v and all(v))
    none = sum(1 for v in per_item.values() if v and not any(v))
    split = n_items - full - none
    print(f"  per-item (n={n_items}): all-seeds-solve={full}  none-solve={none}  split={split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
