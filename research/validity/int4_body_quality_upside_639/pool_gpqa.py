#!/usr/bin/env python3
"""PR #639 -- pool a GPQA-Diamond sampled arm's 10 seeds into n=1980.

Mirrors the ubel #628 / fern #629 denominator pooling (same Wilson CI, same
gb6144 ctx-overflow de-confounding via n_error) so the arm reads apples-to-apples
with bf16 base 0.5404 and Option-B int4+spec 0.4652. Also pools finish_length_rate
(the PR's crater guard: denominator is 0.000).

Usage: pool_gpqa.py <ARM> [seed ...]   (default seeds = the matched 10)
"""
import json
import math
import sys
from pathlib import Path

DIR = Path("research/validity/int4_body_quality_upside_639")
DEFAULT_SEEDS = ["12345", "23456", "34567", "45678", "56789",
                 "67890", "78901", "89012", "90123", "13579"]
BAR_SAMPLED = 0.4864   # PR #639 recalibrated 0.9x bar (0.9 * 0.5404)
OPTIONB = 0.4652       # fern #629 int4+spec numerator
BF16_BASE = 0.5404     # ubel #628 bf16 denominator


def wilson(p, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main():
    arm = sys.argv[1]
    seeds = sys.argv[2:] or DEFAULT_SEEDS
    res = DIR / "results" / arm
    tot_c = tot_n = tot_err = 0
    tot_len = tot_samp = 0
    rows = []
    for s in seeds:
        p = res / f"gpqa_gb6144_s{s}.json"
        if not p.exists():
            print(f"  seed={s} MISSING {p}")
            continue
        d = json.loads(p.read_text())
        c, n = d["n_correct"], d["n_scored"]
        nsamp = d.get("n_samples", n)
        nlen = d.get("n_length", 0)
        tot_c += c
        tot_n += n
        tot_samp += nsamp
        tot_len += nlen
        tot_err += d.get("n_error", 0)
        rows.append({
            "seed": s, "accuracy": d["accuracy"], "n_correct": c, "n_scored": n,
            "n_error": d.get("n_error", 0), "finish_length_rate": d.get("finish_length_rate"),
            "length_stop_rate": d.get("length_stop_rate"), "empty_rate": d.get("empty_rate"),
            "completion_tokens_mean": d.get("completion_tokens_mean"),
            "completion_tokens_p95": d.get("completion_tokens_p95"),
            "completion_tokens_max": d.get("completion_tokens_max"),
        })
        print(f"  seed={s} acc={d['accuracy']:.4f} ({c}/{n}) err={d.get('n_error',0)} "
              f"fl_rate={d.get('finish_length_rate')} len_stop={d.get('length_stop_rate')} "
              f"ctok_mean={d.get('completion_tokens_mean')}")

    if tot_n == 0:
        print("NO SEEDS POOLED"); return
    pooled = tot_c / tot_n
    se = math.sqrt(pooled * (1 - pooled) / tot_n)
    lo_w, hi_w = wilson(pooled, tot_n)
    fl_rate = tot_len / tot_samp if tot_samp else float("nan")

    n_deconf = tot_n - tot_err
    acc_deconf = (tot_c / n_deconf) if n_deconf else float("nan")
    lo_wd, hi_wd = wilson(acc_deconf, n_deconf)

    out = {
        "tag": f"{arm}_gpqa_gb6144", "arm": arm, "seeds": seeds,
        "pooled_accuracy": pooled, "n_correct": tot_c, "n_scored": tot_n, "stderr": se,
        "ci95_wilson": [lo_w, hi_w],
        "pooled_finish_length_rate": fl_rate, "n_length": tot_len, "n_samples": tot_samp,
        "n_request_error": tot_err,
        "accuracy_excl_request_error": acc_deconf, "n_scored_excl_request_error": n_deconf,
        "ci95_wilson_excl_request_error": [lo_wd, hi_wd],
        "anchors": {"bf16_base": BF16_BASE, "optionb_int4_spec": OPTIONB,
                    "recalibrated_0p9_bar": BAR_SAMPLED},
        "vs_optionb": pooled - OPTIONB, "vs_bf16_base": pooled - BF16_BASE,
        "pct_of_bf16_base": pooled / BF16_BASE,
        "clears_recalibrated_bar": bool(pooled >= BAR_SAMPLED),
        "per_seed": rows,
    }
    (res / "pooled.json").write_text(json.dumps(out, indent=2))
    print(f"\n  POOLED {arm} acc={pooled:.4f} ({tot_c}/{tot_n}) stderr={se:.4f}")
    print(f"  95% CI wilson [{lo_w:.4f}, {hi_w:.4f}]")
    print(f"  finish_length_rate(pooled)={fl_rate:.4f}  ({tot_len}/{tot_samp})")
    print(f"  de-confounded (excl {tot_err} ctx-overflow): acc={acc_deconf:.4f} "
          f"({tot_c}/{n_deconf}) wilson[{lo_wd:.4f},{hi_wd:.4f}]")
    print(f"  vs Option-B {OPTIONB}: {pooled-OPTIONB:+.4f} | vs bf16 base {BF16_BASE}: {pooled-BF16_BASE:+.4f} "
          f"| {100*pooled/BF16_BASE:.1f}% of base")
    print(f"  recalibrated 0.9x bar {BAR_SAMPLED}: {'CLEARS' if pooled>=BAR_SAMPLED else 'MISSES'}")


if __name__ == "__main__":
    main()
