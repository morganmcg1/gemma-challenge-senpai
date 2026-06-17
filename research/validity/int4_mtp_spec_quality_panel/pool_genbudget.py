#!/usr/bin/env python3
"""PR #612 — pool GPQA-Diamond seeds per generation-budget arm and report the
verdict vs the 0.471 bar, plus the truncation diagnostics #605 lacked.

Usage: pool_genbudget.py <tag> <seed> [seed ...]
  e.g. pool_genbudget.py gb4096 12345 23456 34567
Writes <tag>_pooled.json and prints a one-line verdict.
"""
import json
import math
import sys
from pathlib import Path

R = Path("research/validity/int4_mtp_spec_quality_panel/results")
BAR = 0.471


def wilson(p, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main():
    tag = sys.argv[1]
    seeds = sys.argv[2:]
    tot_c = tot_n = 0
    tot_trunc = tot_samp = 0
    tot_err = 0
    rows = []
    ctok_all = []
    ctok_samples = []  # per-item completion_tokens pooled across seeds (for retro-trunc curve)
    for s in seeds:
        p = R / f"spec_gpqa_{tag}_s{s}.json"
        if not p.exists():
            print(f"  seed={s} MISSING {p}")
            continue
        d = json.loads(p.read_text())
        c, n = d["n_correct"], d["n_scored"]
        tot_c += c
        tot_n += n
        nlt = d.get("n_length_truncated", 0)
        nsamp = d.get("n_samples", n)
        tot_trunc += nlt
        tot_samp += nsamp
        tot_err += d.get("n_error", 0)
        rows.append({
            "seed": s, "accuracy": d["accuracy"], "n_correct": c, "n_scored": n,
            "n_error": d.get("n_error", 0), "empty_rate": d.get("empty_rate"),
            "length_stop_rate": d.get("length_stop_rate"),
            "n_length_truncated": nlt, "n_stop_max_tokens": d.get("n_stop_max_tokens"),
            "n_stop_model_length": d.get("n_stop_model_length"),
            "stop_reason_counts": d.get("stop_reason_counts"),
            "completion_tokens_mean": d.get("completion_tokens_mean"),
            "completion_tokens_p50": d.get("completion_tokens_p50"),
            "completion_tokens_p95": d.get("completion_tokens_p95"),
            "completion_tokens_max": d.get("completion_tokens_max"),
            "max_tokens": d.get("max_tokens"),
        })
        cm = d.get("completion_tokens_mean")
        if cm is not None:
            ctok_all.append((cm, nsamp))
        for ps in d.get("per_sample", []):
            ct = ps.get("completion_tokens")
            if isinstance(ct, (int, float)) and ct > 0:
                ctok_samples.append(ct)
        print(f"  seed={s} acc={d['accuracy']:.4f} ({c}/{n}) err={d.get('n_error',0)} "
              f"empty_rate={d.get('empty_rate')} len_stop_rate={d.get('length_stop_rate')} "
              f"len_trunc={nlt} ctok(mean/p95/max)="
              f"{d.get('completion_tokens_mean')}/{d.get('completion_tokens_p95')}/{d.get('completion_tokens_max')}")

    pooled = tot_c / tot_n if tot_n else float("nan")
    se = math.sqrt(pooled * (1 - pooled) / tot_n) if tot_n else float("nan")
    lo_n, hi_n = pooled - 1.96 * se, pooled + 1.96 * se
    lo_w, hi_w = wilson(pooled, tot_n)
    trunc_rate = tot_trunc / tot_samp if tot_samp else float("nan")
    ctok_w = (sum(m * w for m, w in ctok_all) / sum(w for _, w in ctok_all)) if ctok_all else None

    # De-confounded accuracy: exclude request-error items. At max_tokens=6144 with
    # max_model_len=8192, the one GPQA item whose prompt is >=2049 tokens is rejected
    # up-front (2049+6144 > 8192 ctx -> vLLM 400), force-scored wrong in every seed.
    # That is a max_tokens/context-fit config artifact, not a quality signal, and it
    # does not occur at max_tokens=3072 (2049+3072 < 8192). Errored items are all
    # scored incorrect (empty output), so n_correct is unchanged; we only shrink the
    # denominator. This is the fair budget-controlled number for the marginal verdict.
    n_deconf = tot_n - tot_err
    acc_deconf = (tot_c / n_deconf) if n_deconf else float("nan")
    se_deconf = math.sqrt(acc_deconf * (1 - acc_deconf) / n_deconf) if n_deconf else float("nan")
    lo_wd, hi_wd = wilson(acc_deconf, n_deconf)

    # Retro-truncation curve: from the de-biased run's NATURAL completion lengths,
    # what fraction of items WOULD truncate at a tighter cap C (= frac with ctok > C).
    # This recovers the length-stop rate #605 would have suffered at its 3072 cap,
    # apples-to-apples on (near-)identical generations (#605 logged no ctok diagnostics).
    nct = len(ctok_samples)
    retro = {}
    for cap in (1024, 2048, 3072, 4096, 5120, 6144):
        over = sum(1 for x in ctok_samples if x > cap)
        retro[str(cap)] = {"n_over": over, "frac_over": (over / nct) if nct else float("nan")}

    out = {
        "tag": tag, "seeds": seeds, "pooled_accuracy": pooled,
        "n_correct": tot_c, "n_scored": tot_n, "stderr": se,
        "ci95_normal": [lo_n, hi_n], "ci95_wilson": [lo_w, hi_w],
        "bar": BAR, "pass": bool(pooled >= BAR),
        "margin": pooled - BAR, "sigma_vs_bar": (pooled - BAR) / se if se else None,
        "pooled_length_stop_rate": trunc_rate, "n_length_truncated": tot_trunc,
        "n_request_error": tot_err,
        "accuracy_excl_request_error": acc_deconf, "n_scored_excl_request_error": n_deconf,
        "stderr_excl_request_error": se_deconf,
        "ci95_wilson_excl_request_error": [lo_wd, hi_wd],
        "pass_excl_request_error": bool(acc_deconf >= BAR),
        "n_samples": tot_samp, "completion_tokens_mean_weighted": ctok_w,
        "retro_truncation_at_cap": retro, "n_ctok_samples": nct,
        "per_seed": rows,
    }
    (Path("research/validity/int4_mtp_spec_quality_panel") / f"{tag}_pooled.json").write_text(
        json.dumps(out, indent=2))
    print(f"\n  POOLED {tag} acc={pooled:.4f} ({tot_c}/{tot_n}) stderr={se:.4f}")
    print(f"  95% CI normal [{lo_n:.4f}, {hi_n:.4f}] | wilson [{lo_w:.4f}, {hi_w:.4f}]  bar={BAR}")
    print(f"  length_stop_rate(pooled)={trunc_rate:.4f}  ({tot_trunc}/{tot_samp})  ctok_mean~{ctok_w}")
    print(f"  de-confounded (excl {tot_err} request-err/overflow): acc={acc_deconf:.4f} "
          f"({tot_c}/{n_deconf}) wilson[{lo_wd:.4f},{hi_wd:.4f}] "
          f"-> {'PASS' if acc_deconf >= BAR else 'FAIL'}")
    print(f"  retro-trunc (frac of NATURAL ctok > cap, from de-biased run):")
    for cap in ("2048", "3072", "4096"):
        r = retro.get(cap, {})
        print(f"    @cap {cap}: {r.get('frac_over', float('nan')):.4f}  ({r.get('n_over')}/{nct})")
    sig = (pooled - BAR) / se if se else float("nan")
    print(f"  -> {'PASS' if pooled >= BAR else 'FAIL'}  (margin {pooled-BAR:+.4f} = {sig:+.2f} sigma)")


if __name__ == "__main__":
    main()
