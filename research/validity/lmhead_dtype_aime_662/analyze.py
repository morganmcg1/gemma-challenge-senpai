#!/usr/bin/env python
"""Stats for the lm_head-dtype AIME panel (PR #662).

Loads per-arm aime_eval JSONs (per_problem.maj_correct keyed by problem id) and
computes, with NO external deps:
  * per cell: maj@1 acc, Wilson 95% CI, %-of-bf16-base (0.4667), clears-0.420
  * HEADLINE Delta_head = bf16head - shipped (paired, same 60 problems):
      McNemar exact two-sided p + Newcombe (1998) method-10 95% CI for the
      paired difference of proportions
  * calibration residual = official_g32 (cited #653) - bf16head
  * int8head as the intermediate (paired vs shipped too)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

BF16_BASE = 0.4667   # ubel #628 bf16 base AIME anchor
BAR = 0.420          # 90% pass bar
OFFICIAL_G32 = 0.4167  # cited #653 75mzy4ur (Google calib + tied bf16 head)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact (binomial) McNemar p over discordant pairs b, c."""
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    m = min(b, c)
    tail = sum(comb(n, k) for k in range(0, m + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def newcombe_method10(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    """Newcombe 1998 method 10: 95% CI for paired proportion difference.

    Table (rows=arm1 correct/wrong, cols=arm2 correct/wrong):
        a = both correct
        b = arm1 correct, arm2 wrong
        c = arm1 wrong, arm2 correct
        d = both wrong
    Returns (delta, lower, upper) for delta = p1 - p2 = (b - c)/n,
    p1 = (a+b)/n (arm1), p2 = (a+c)/n (arm2).
    """
    n = a + b + c + d
    p1 = (a + b) / n
    p2 = (a + c) / n
    delta = p1 - p2
    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    # correlation phi estimate (Newcombe); guard zero marginals
    A = (a + b) * (c + d) * (a + c) * (b + d)
    if A <= 0:
        phi = 0.0
    else:
        phi = (a * d - b * c) / math.sqrt(A)
        phi = max(-1.0, min(1.0, phi))
    lo_in = (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2
    hi_in = (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2
    lower = delta - math.sqrt(max(0.0, lo_in))
    upper = delta + math.sqrt(max(0.0, hi_in))
    return delta, max(-1.0, lower), min(1.0, upper)


def load_arm(path: Path) -> dict[str, bool]:
    d = json.loads(Path(path).read_text())
    return {r["id"]: bool(r["maj_correct"]) for r in d["per_problem"]}


def cell_report(name: str, correct: dict[str, bool]) -> dict:
    n = len(correct)
    k = sum(correct.values())
    acc = k / n if n else 0.0
    lo, hi = wilson(k, n)
    return {
        "arm": name, "n": n, "n_correct": k, "maj1_acc": acc,
        "wilson95": [lo, hi], "pct_of_bf16_base": acc / BF16_BASE,
        "clears_0p420": acc >= BAR,
    }


def paired(name1: str, c1: dict[str, bool], name2: str, c2: dict[str, bool]) -> dict:
    ids = sorted(set(c1) & set(c2))
    a = sum(1 for i in ids if c1[i] and c2[i])
    b = sum(1 for i in ids if c1[i] and not c2[i])
    c = sum(1 for i in ids if not c1[i] and c2[i])
    d = sum(1 for i in ids if not c1[i] and not c2[i])
    delta, lo, hi = newcombe_method10(a, b, c, d)
    p = mcnemar_exact(b, c)
    return {
        "arm1": name1, "arm2": name2, "n_paired": len(ids),
        "table": {"both_correct": a, f"{name1}_only": b, f"{name2}_only": c, "both_wrong": d},
        "delta_acc": delta, "newcombe95": [lo, hi],
        "mcnemar_b": b, "mcnemar_c": c, "mcnemar_exact_p": p,
        "significant_0p05": p < 0.05,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shipped", type=Path, required=True)
    ap.add_argument("--bf16head", type=Path, required=True)
    ap.add_argument("--int8head", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    arms = {
        "shipped_g128": load_arm(args.shipped),
        "our_g128_bf16head": load_arm(args.bf16head),
    }
    if args.int8head and args.int8head.exists():
        arms["our_g128_int8head"] = load_arm(args.int8head)

    cells = {name: cell_report(name, c) for name, c in arms.items()}

    out = {
        "pr": 662, "analysis_only": True, "official_tps": 0,
        "anchors": {"bf16_base": BF16_BASE, "bar_0p90": BAR, "official_g32": OFFICIAL_G32},
        "cells": cells,
        "headline_delta_head_bf16head_minus_shipped":
            paired("our_g128_bf16head", arms["our_g128_bf16head"],
                   "shipped_g128", arms["shipped_g128"]),
        "calibration_residual_officialg32_minus_bf16head":
            OFFICIAL_G32 - cells["our_g128_bf16head"]["maj1_acc"],
    }
    if "our_g128_int8head" in arms:
        out["int8head_minus_shipped"] = paired(
            "our_g128_int8head", arms["our_g128_int8head"],
            "shipped_g128", arms["shipped_g128"])
        out["bf16head_minus_int8head"] = paired(
            "our_g128_bf16head", arms["our_g128_bf16head"],
            "our_g128_int8head", arms["our_g128_int8head"])

    args.out.write_text(json.dumps(out, indent=2))
    # console summary
    print("=== per-cell ===")
    for name, c in cells.items():
        print(f"  {name:22s} maj@1={c['maj1_acc']:.4f} ({c['n_correct']}/{c['n']}) "
              f"W95=[{c['wilson95'][0]:.3f},{c['wilson95'][1]:.3f}] "
              f"%bf16={c['pct_of_bf16_base']*100:.1f}% clears0.420={c['clears_0p420']}")
    h = out["headline_delta_head_bf16head_minus_shipped"]
    print(f"\n=== HEADLINE Delta_head (bf16head - shipped) ===")
    print(f"  delta={h['delta_acc']:+.4f} Newcombe95=[{h['newcombe95'][0]:+.4f},{h['newcombe95'][1]:+.4f}]")
    print(f"  McNemar b(bf16-only)={h['mcnemar_b']} c(shipped-only)={h['mcnemar_c']} "
          f"exact_p={h['mcnemar_exact_p']:.4f} sig={h['significant_0p05']}")
    print(f"\n=== calibration residual (official_g32 {OFFICIAL_G32} - bf16head) = "
          f"{out['calibration_residual_officialg32_minus_bf16head']:+.4f} ===")
    print(f"\n[analyze] wrote {args.out}")


if __name__ == "__main__":
    main()
