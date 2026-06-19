#!/usr/bin/env python3
"""PR #696 -- cross-boot pooling sensitivity for the GPQA-D 30-seed pool.

The seed pool mixes two server boots: seeds 0-9 are the banked #589/#692 draws
(carried in as bf_gpqa_sampled_mt8_s{0..9}.json), seeds 10-29 are freshly drawn
on THIS A10G boot. The run_gpqa_seeds.py --repro-check re-ran sampling_seed 0 on
this boot and got 100/198 vs the banked 102/198 (repro_check.json reproduced=false).

That Delta=2/198 is EXPECTED, not a bug: under vLLM continuous-batched SAMPLED decode
(concurrency 16) the per-request logits are not bit-identical across boots (kernel
autotune + non-associative reduction order), so a fixed --sampling-seed does NOT
byte-reproduce token-for-token across boots. Per-item byte reproduction is the wrong
validity check for a SEED-DISTRIBUTION estimate; the right one is whether the banked
draws and the fresh draws come from the SAME accuracy distribution. This script runs
that check (Welch two-sample on the per-seed accuracies) and shows whether the gate
verdict is robust to dropping the banked seeds entirely (this-boot-only).

LOCAL, analysis_only, NO FIRE. Reads only local seed files.
"""
from __future__ import annotations

import glob
import json
import math
import statistics as st
from pathlib import Path

HERE = Path("/workspace/senpai/target/research/validity/int4_quality_31basis_confirm")
RES = HERE / "results_gpqa"
GATE = 0.471
BASE = 0.5236
Z = 1.959963984540054
BOOT_SPLIT = 10  # seed < 10 = banked (#589/#692 boot); seed >= 10 = this-session fresh

TCRIT = {
    2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306,
    10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131,
    17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093, 21: 2.086, 22: 2.080, 23: 2.074,
    24: 2.069, 25: 2.064, 26: 2.060, 27: 2.056, 28: 2.052, 29: 2.048, 30: 2.045,
}


def wilson(k, n, z=Z):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (c - m) / den, (c + m) / den


def summarize(rows, label):
    accs = [d["accuracy"] for d in rows]
    k = sum(d["n_correct"] for d in rows)
    n = sum(d["n_scored"] for d in rows)
    K = len(rows)
    mean = st.mean(accs)
    sd = st.stdev(accs) if K > 1 else 0.0
    pooled_lo, pooled_hi = wilson(k, n)
    if K > 1:
        t = TCRIT.get(K, 1.96)
        se = sd / math.sqrt(K)
        seed_lo, seed_hi = mean - t * se, mean + t * se
    else:
        seed_lo = seed_hi = float("nan")
    return {
        "label": label, "K_seeds": K, "mean_acc": mean, "std_acc": sd,
        "pct_of_base": 100 * mean / BASE,
        "pooled_wilson_lo": pooled_lo, "pooled_wilson_hi": pooled_hi,
        "pooled_wilson_clears_0471": bool(pooled_lo >= GATE),
        "seed_mean_lo": seed_lo, "seed_mean_hi": seed_hi,
        "seed_mean_clears_0471": bool(seed_lo >= GATE),
        "min_seed_acc": min(accs), "max_seed_acc": max(accs),
        "_accs": accs,
    }


def welch(a, b):
    na, nb = len(a), len(b)
    ma, mb = st.mean(a), st.mean(b)
    va, vb = st.variance(a), st.variance(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return float("nan"), float("nan")
    tval = (ma - mb) / se
    # Welch-Satterthwaite df
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return tval, df


def main():
    rows = []
    for f in sorted(glob.glob(str(RES / "bf_gpqa_sampled_mt8_s*.json")),
                    key=lambda p: int(Path(p).stem.split("_s")[-1])):
        d = json.load(open(f))
        d["_seed"] = d["sampling_seed"]
        rows.append(d)
    banked = [d for d in rows if d["_seed"] < BOOT_SPLIT]
    fresh = [d for d in rows if d["_seed"] >= BOOT_SPLIT]

    all_s = summarize(rows, "all_30")
    bk_s = summarize(banked, "banked_0to9") if banked else None
    fr_s = summarize(fresh, "thisboot_10plus") if fresh else None

    out = {"pr": 696, "analysis_only": True, "boot_split_seed": BOOT_SPLIT,
           "all": all_s, "banked": bk_s, "thisboot": fr_s}

    if bk_s and fr_s:
        tval, df = welch(bk_s["_accs"], fr_s["_accs"])
        out["welch_banked_vs_thisboot"] = {
            "t": tval, "df": df,
            "mean_diff": bk_s["mean_acc"] - fr_s["mean_acc"],
            # |t| < ~2.0 over these df => not significant at 0.05 => pooling defensible
            "abs_t_lt_2": bool(abs(tval) < 2.0),
        }

    rc = HERE / "repro_check.json"
    if rc.exists():
        out["repro_check"] = json.load(open(rc))

    for s in (all_s, bk_s, fr_s):
        if s:
            s.pop("_accs", None)

    (HERE / "sensitivity_boot.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
