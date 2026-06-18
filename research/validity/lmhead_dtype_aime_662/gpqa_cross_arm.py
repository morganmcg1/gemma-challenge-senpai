#!/usr/bin/env python
"""Cross-arm GPQA-D guard stats for PR #662.

The GPQA-D guard's signal is CROSS-ARM FLATNESS (does lm_head dtype trade AIME
for GPQA?), not the absolute (mt2048 truncates long-CoT GPQA-D, depressing the
absolute for ALL arms equally). Reads gpqa_<arm>.json per_sample (id, correct),
reports per-arm acc + Wilson95, and pairwise exact McNemar + Newcombe95 paired
difference for every arm pair present. Graceful if bf16head not yet finished.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
ARMS = ["shipped_g128", "our_g128_int8head", "our_g128_bf16head"]


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
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    m = min(b, c)
    tail = sum(comb(n, k) for k in range(0, m + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def newcombe10(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    delta = p1 - p2
    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    A = (a + b) * (c + d) * (a + c) * (b + d)
    phi = 0.0 if A <= 0 else max(-1.0, min(1.0, (a * d - b * c) / math.sqrt(A)))
    lo_in = (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2
    hi_in = (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2
    return delta, max(-1.0, delta - math.sqrt(max(0.0, lo_in))), min(1.0, delta + math.sqrt(max(0.0, hi_in)))


def load(arm: str) -> dict[str, bool] | None:
    p = RES / f"gpqa_{arm}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {r["id"]: bool(r["correct"]) for r in d["per_sample"]}


def main() -> None:
    arms = {a: load(a) for a in ARMS}
    present = {a: c for a, c in arms.items() if c is not None}
    out = {"per_arm": {}, "pairwise": {}}
    print("=== GPQA-D per-arm (mt2048 truncating; signal is CROSS-ARM FLATNESS) ===")
    for a, c in present.items():
        n, k = len(c), sum(c.values())
        lo, hi = wilson(k, n)
        out["per_arm"][a] = {"n": n, "n_correct": k, "acc": k / n, "wilson95": [lo, hi]}
        print(f"  {a:22s} acc={k/n:.4f} ({k}/{n}) W95=[{lo:.3f},{hi:.3f}]")
    names = list(present)
    print("\n=== pairwise (flatness check: delta ~ 0, McNemar NS) ===")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            c1, c2 = present[n1], present[n2]
            ids = sorted(set(c1) & set(c2))
            a = sum(1 for x in ids if c1[x] and c2[x])
            b = sum(1 for x in ids if c1[x] and not c2[x])
            cc = sum(1 for x in ids if not c1[x] and c2[x])
            d = sum(1 for x in ids if not c1[x] and not c2[x])
            delta, lo, hi = newcombe10(a, b, cc, d)
            p = mcnemar_exact(b, cc)
            out["pairwise"][f"{n1}__vs__{n2}"] = {
                "n_paired": len(ids), "delta_acc": delta, "newcombe95": [lo, hi],
                "mcnemar_b": b, "mcnemar_c": cc, "mcnemar_exact_p": p,
                "significant_0p05": p < 0.05,
            }
            print(f"  {n1} vs {n2}: delta={delta:+.4f} N95=[{lo:+.3f},{hi:+.3f}] "
                  f"McNemar b={b} c={cc} p={p:.4f} sig={p < 0.05}")
    (RES / "gpqa_cross_arm.json").write_text(json.dumps(out, indent=2))
    print(f"\n[gpqa] wrote {RES / 'gpqa_cross_arm.json'} ({len(present)} arms)")


if __name__ == "__main__":
    main()
