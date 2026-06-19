#!/usr/bin/env python
"""PR #659 Phase-2 DECISIVE instrument: paired McNemar, int8-on-locus vs int4-N=0.

A standalone int8 sampled Wilson CI vs the greedy-defined 0.420 bar is biased to FAIL
by construction (sampled<=greedy; bar set on greedy 0.9*bf16). The noise-robust recovery
test is the PAIRED comparison: int8-on-locus sampled vs int4-N=0 sampled on the SAME
seeds. Pairing each (item,seed) cancels the common sampling penalty and directly answers
"does the int8 upgrade on L14-27 lift AIME?". ubel #650 precedent.

For each seed s, pairs item-id outcomes between:
  int8:  mix_int8_L14-27_s{seed}_aime.jsonl
  int4:  int4_N0_s{seed}_aime.jsonl
Pools the 2x2 discordant counts across all seeds and reports:
  - per-seed int8 acc / int4 acc / delta
  - pooled int8 acc / int4 acc / paired diff + Wald paired 95% CI
  - exact McNemar (binomial sign test on discordant pairs), two-sided p
  - normal-approx McNemar chi^2 (continuity-corrected) for reference

Usage:
  mcnemar_paired.py 12345 23456 34567 45678 56789
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
INT8_TMPL = "mix_int8_L14-27_s{seed}_aime.jsonl"
INT4_TMPL = "int4_N0_s{seed}_aime.jsonl"


def load_correct(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error"):
            continue
        out[str(r["id"])] = 1 if r.get("correct") else 0
    return out


def exact_mcnemar_two_sided(b: int, c: int) -> float:
    """Exact binomial (sign) test on discordant pairs; p = 0.5 under H0."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def main(seeds: list[str]) -> int:
    a = b = c = d = 0  # both-correct / int8-only / int4-only / both-wrong
    print(f"{'seed':>7} {'paired':>6} {'int8':>6} {'int4':>6} {'delta':>7}  (int8-only/int4-only)")
    per_seed_delta = []
    for s in seeds:
        i8 = load_correct(RES / INT8_TMPL.format(seed=s))
        i4 = load_correct(RES / INT4_TMPL.format(seed=s))
        common = sorted(set(i8) & set(i4))
        if not common:
            print(f"{s:>7}  no overlap (int8={len(i8)} int4={len(i4)})")
            continue
        sa = sb = sc = sd = 0
        for iid in common:
            x, y = i8[iid], i4[iid]
            if x and y:
                sa += 1
            elif x and not y:
                sb += 1
            elif (not x) and y:
                sc += 1
            else:
                sd += 1
        np_ = len(common)
        i8_acc = (sa + sb) / np_
        i4_acc = (sa + sc) / np_
        print(f"{s:>7} {np_:>6} {i8_acc:>6.3f} {i4_acc:>6.3f} {i8_acc - i4_acc:>+7.3f}"
              f"  ({sb}/{sc})")
        per_seed_delta.append(i8_acc - i4_acc)
        a += sa; b += sb; c += sc; d += sd
    n_pairs = a + b + c + d
    if n_pairs == 0:
        print("no paired data")
        return 1
    p8 = (a + b) / n_pairs
    p4 = (a + c) / n_pairs
    diff = p8 - p4  # == (b - c)/n_pairs
    # Wald paired difference-of-proportions CI
    var = ((b + c) - (b - c) ** 2 / n_pairs) / (n_pairs ** 2)
    se = math.sqrt(max(var, 0.0))
    lo95, hi95 = diff - 1.96 * se, diff + 1.96 * se
    lo90, hi90 = diff - 1.645 * se, diff + 1.645 * se
    p_exact = exact_mcnemar_two_sided(b, c)
    chi2_cc = ((abs(b - c) - 1) ** 2) / (b + c) if (b + c) > 0 else 0.0
    print("-" * 72)
    print(f"POOLED pairs={n_pairs}  2x2: both_ok={a} int8_only={b} int4_only={c} both_wrong={d}")
    print(f"  int8-on-locus acc = {p8:.4f}   int4-N=0 acc = {p4:.4f}")
    print(f"  paired diff (int8 - int4) = {diff:+.4f}")
    print(f"    Wald95 [{lo95:+.4f}, {hi95:+.4f}]   Wald90 [{lo90:+.4f}, {hi90:+.4f}]")
    print(f"  discordant: int8_only(b)={b}  int4_only(c)={c}")
    print(f"  exact McNemar (binomial, two-sided) p = {p_exact:.4f}")
    print(f"  McNemar chi^2 (cont-corrected) = {chi2_cc:.3f}  (crit_0.05=3.841)")
    if per_seed_delta:
        md = sum(per_seed_delta) / len(per_seed_delta)
        pos = sum(1 for x in per_seed_delta if x > 0)
        print(f"  per-seed delta: mean={md:+.4f}  seeds_int8>int4={pos}/{len(per_seed_delta)}")
    print(f"  RECOVERY_LIFT_SIGNIFICANT_0.05 = {p_exact < 0.05 and diff > 0}")
    print(f"  diff_CI95_excludes_zero = {lo95 > 0 or hi95 < 0}")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:] or ["12345", "23456", "34567", "45678", "56789"]
    raise SystemExit(main(args))
