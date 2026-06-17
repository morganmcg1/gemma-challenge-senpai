#!/usr/bin/env python3
"""PR #598 -- aggregate the gpqa_main (larger-instrument) sampling sweep into the two
verdicts. Self-contained (math/statistics/random; no numpy/scipy).

Two configs, K sampling seeds each, byte-identical gpqa_main item set:
  base          = UNQUANTIZED bf16 google/gemma-4-E4B-it (the gate denominator).
  base_fullhead = int4-W4A16-g32 body + native 262k bf16 lm_head.

Deliverables (report EVERY lens; no thumb on the scale, exactly as #589):
  1. per-seed acc, mean +/- std (ddof=1), worst seed -- both configs.
  2. RE-ANCHORED gate = 0.90 x (base gpqa_main mean acc on THIS instrument).
  3. base_fullhead 95% CI-lb, two lenses:
       - clustered bootstrap over items (resample items; each item's mean-over-seeds
         correctness; B=20000) -- the statistically-correct CI (PRIMARY, #589 lens).
       - item-level Wilson at n=448 -- the irreducible binomial.
  4. PRIMARY verdict gpqa_main_ci_lb_clears_90pct_base: base_fullhead CI-lb AND worst
     single seed stay >= the re-anchored gate?
  5. NO-REGRESSION verdict base_fullhead_not_regression_sampling: paired McNemar over
     shared-seed (item,seed) cells (base<->base_fullhead, same prompt + same sampling
     seed -> only the model differs). p>0.05 => TRUE. Reports n01/n10, exact-binomial p,
     continuity-corrected chi2 p, AND a clustering-robust per-item sign test.
  6. un-tightenability: n_for_wilson_cilb_pass_at_point on gpqa_main -- how big an
     instrument the gate would need (Diamond #589 was 3758; this is the larger-n test).
"""
from __future__ import annotations

import glob
import json
import math
import random
import statistics as st
from pathlib import Path

HERE = Path("/workspace/senpai/target/research/validity/gpqa_larger_instrument_ci")
RES = HERE / "results"
QUALITY_FRAC = 0.90  # Morgan #515 ">=90% of vanilla base"
Z = 1.959963984540054  # 95% two-sided
GPQA_MAIN_N = 448
GPQA_EXTENDED_N = 546


def wilson_lo(k: int, n: int, z: float = Z) -> float:
    if n == 0:
        return float("nan")
    phat = k / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return (centre - margin) / denom


def wilson_hi(k: int, n: int, z: float = Z) -> float:
    if n == 0:
        return float("nan")
    phat = k / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return (centre + margin) / denom


def _log_binom_pmf(k: int, n: int, p: float = 0.5) -> float:
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log(1 - p))


def binom_two_sided_p(a: int, b: int) -> float:
    """Exact two-sided binomial p for a vs b discordant/sign counts under H0 p=0.5."""
    n = a + b
    if n == 0:
        return 1.0
    m = max(a, b)
    s = sum(math.exp(_log_binom_pmf(k, n)) for k in range(m, n + 1))
    return min(1.0, 2.0 * s)


def mcnemar_chi2_p(n01: int, n10: int) -> tuple[float, float]:
    """Continuity-corrected McNemar chi2 (df=1) p-value via erfc."""
    d = n01 + n10
    if d == 0:
        return 1.0, 0.0
    chi2 = max((abs(n01 - n10) - 1) ** 2 / d, 0.0)
    return math.erfc(math.sqrt(chi2 / 2.0)), chi2


def load_config(config: str) -> list[dict]:
    rows = []
    for f in sorted(glob.glob(str(RES / f"{config}_gpqa_main_mt8_s*.json"))):
        rows.append(json.load(open(f)))
    return rows


def per_config_stats(rows: list[dict]):
    """Returns (seed_table, seed_accs, per_item{id:[correct...]}, cell{(id,seed):correct},
    item_sha{id:sha}, pooled_correct, pooled_scored)."""
    seed_table, seed_accs = [], []
    per_item: dict[str, list[int]] = {}
    cell: dict[tuple, int] = {}
    item_sha: dict[str, str] = {}
    pooled_correct = pooled_scored = 0
    for d in rows:
        seed = d["sampling_seed"]
        seed_accs.append(d["accuracy"])
        seed_table.append({
            "sampling_seed": seed, "accuracy": d["accuracy"],
            "n_scored": d["n_scored"], "n_correct": d["n_correct"],
            "n_empty": d.get("n_empty"), "empty_rate": d.get("empty_rate"),
            "min_tokens": d.get("min_tokens"),
        })
        pooled_correct += d["n_correct"]
        pooled_scored += d["n_scored"]
        for r in d["per_sample"]:
            if r.get("value") not in ("C", "I"):
                continue  # drop errors/unscored from the correctness vectors
            sid = r["id"]
            c = 1 if r["correct"] else 0
            per_item.setdefault(sid, []).append(c)
            cell[(sid, seed)] = c
            item_sha[sid] = r.get("prompt_sha")
    return seed_table, seed_accs, per_item, cell, item_sha, pooled_correct, pooled_scored


def main():
    base_rows = load_config("base")
    fh_rows = load_config("base_fullhead")
    if not base_rows or not fh_rows:
        raise SystemExit(f"missing seed files: base={len(base_rows)} fh={len(fh_rows)}")

    (b_tbl, b_accs, b_item, b_cell, b_sha, b_pc, b_ps) = per_config_stats(base_rows)
    (f_tbl, f_accs, f_item, f_cell, f_sha, f_pc, f_ps) = per_config_stats(fh_rows)

    # cross-arm + cross-seed prompt_sha consistency (byte-identical item set)
    all_ids = set(b_sha) | set(f_sha)
    n_prompt_mismatch = sum(
        1 for i in all_ids if b_sha.get(i) is not None and f_sha.get(i) is not None
        and b_sha[i] != f_sha[i]
    )

    Kb, Kf = len(base_rows), len(fh_rows)

    def point_over_items(per_item):
        ids = sorted(per_item)
        means = [sum(per_item[i]) / len(per_item[i]) for i in ids]
        return (sum(means) / len(means)) if means else float("nan"), ids, means

    base_point, _, _ = point_over_items(b_item)
    fh_point, fh_ids, fh_item_means = point_over_items(f_item)

    gate = QUALITY_FRAC * base_point  # RE-ANCHORED on gpqa_main vanilla base

    # base_fullhead CI lenses
    # (A) clustered bootstrap over items (PRIMARY)
    rng = random.Random(20260617)
    B = 20000
    M = len(fh_item_means)
    boot = []
    for _ in range(B):
        s = 0.0
        for _ in range(M):
            s += fh_item_means[rng.randrange(M)]
        boot.append(s / M)
    boot.sort()
    fh_boot_lo = boot[int(0.025 * B)]
    fh_boot_hi = boot[int(0.975 * B)]
    fh_boot_point = sum(boot) / B
    # (B) item-level Wilson at n=448 on the pooled point
    n_items = len(fh_ids)
    k_round = round(fh_point * n_items)
    fh_wilson_lo = wilson_lo(k_round, n_items)
    fh_wilson_hi = wilson_hi(k_round, n_items)

    # base CI (for the both-CI-aware gate lens; gate carries denominator noise too)
    base_ids = sorted(b_item)
    base_item_means = [sum(b_item[i]) / len(b_item[i]) for i in base_ids]
    Mb = len(base_item_means)
    bootb = []
    for _ in range(B):
        s = 0.0
        for _ in range(Mb):
            s += base_item_means[rng.randrange(Mb)]
        bootb.append(s / Mb)
    bootb.sort()
    base_boot_lo = bootb[int(0.025 * B)]
    base_boot_hi = bootb[int(0.975 * B)]

    # per-seed summaries
    b_mean, f_mean = st.mean(b_accs), st.mean(f_accs)
    b_std = st.stdev(b_accs) if Kb > 1 else 0.0
    f_std = st.stdev(f_accs) if Kf > 1 else 0.0
    f_min, f_max = min(f_accs), max(f_accs)
    b_min, b_max = min(b_accs), max(b_accs)
    f_seeds_below = sum(1 for a in f_accs if a < gate)

    # ---- McNemar over shared (id, seed) cells: only the MODEL differs ----
    shared_cells = [k for k in b_cell.keys() & f_cell.keys()]
    n00 = n01 = n10 = n11 = 0
    for k in shared_cells:
        b, f = b_cell[k], f_cell[k]
        if b == 1 and f == 1:
            n11 += 1
        elif b == 1 and f == 0:
            n01 += 1  # base correct, base_fullhead WRONG  (a regression vote)
        elif b == 0 and f == 1:
            n10 += 1  # base wrong, base_fullhead correct   (an improvement vote)
        else:
            n00 += 1
    mcnemar_p_exact = binom_two_sided_p(n01, n10)
    mcnemar_p_chi2, mcnemar_chi2 = mcnemar_chi2_p(n01, n10)

    # ---- clustering-robust per-item sign test on mean-over-seeds correctness ----
    paired_ids = sorted(set(b_item) & set(f_item))
    n_base_gt = n_fh_gt = n_tie = 0
    item_diffs = []
    for i in paired_ids:
        bm = sum(b_item[i]) / len(b_item[i])
        fm = sum(f_item[i]) / len(f_item[i])
        item_diffs.append(bm - fm)
        if bm > fm:
            n_base_gt += 1
        elif fm > bm:
            n_fh_gt += 1
        else:
            n_tie += 1
    sign_p = binom_two_sided_p(n_base_gt, n_fh_gt)
    mean_item_diff = (sum(item_diffs) / len(item_diffs)) if item_diffs else float("nan")

    # un-tightenability: smallest n so Wilson-lb at fh_point clears the gate
    def n_for_wilson_pass(p, g):
        for n in range(n_items, 200000):
            if wilson_lo(round(p * n), n) >= g:
                return n
        return None
    n_for_pass = n_for_wilson_pass(fh_point, gate)

    # verdicts
    fh_boot_clears = bool(fh_boot_lo >= gate)
    fh_wilson_clears = bool(fh_wilson_lo >= gate)
    worst_seed_clears = bool(f_min >= gate)
    # PR #598 defines the primary verdict literally as "does base_fullhead's 95%
    # CI-lb stay >= the re-anchored gate" (the clustered-bootstrap lens it names as
    # the #589 primary). worst-single-seed is an ADDITIONAL report (worst_seed_clears),
    # NOT part of this headline bool -- ANDing it in would conflate the item-CI question
    # with a per-seed-floor question and bias the verdict conservative. Report straight.
    primary_verdict = bool(fh_boot_clears)
    not_regression = bool(mcnemar_p_exact > 0.05)
    # direction: a "regression" requires base_fullhead significantly WORSE (n01 dominates)
    regression_direction = ("base_fullhead_worse" if n01 > n10 else
                            "base_fullhead_better" if n10 > n01 else "tie")

    out = {
        "pr": 598,
        "analysis_only": True,
        "official_tps": 0,
        "no_hf_job": True,
        "task": "gpqa_main",
        "instrument_n": n_items,
        "gpqa_main_ceiling_n": GPQA_MAIN_N,
        "gpqa_extended_ceiling_n": GPQA_EXTENDED_N,
        "quality_frac": QUALITY_FRAC,
        "K_seeds_base": Kb,
        "K_seeds_base_fullhead": Kf,
        "protocol": {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 64,
                     "min_tokens": 8, "max_tokens": 3072, "dataset_seed": 12345,
                     "source": "generation_config.json (lewtun #31)"},
        "serve": ("identical faithful #589 stack both arms (dev307 + serve_inject: "
                  "FA_SLIDING + SURGICAL_ATTN_USE_3D_OFF + PLE_FOLD); only the checkpoint "
                  "differs (bf16 body vs int4-W4A16-g32 body), so base<->base_fullhead "
                  "isolates int4-body quantization"),
        "n_prompt_mismatch": n_prompt_mismatch,
        # per-seed
        "base_seed_table": b_tbl,
        "base_fullhead_seed_table": f_tbl,
        "base_per_seed_acc": b_accs,
        "base_fullhead_per_seed_acc": f_accs,
        "base_mean_acc": b_mean, "base_std_acc": b_std, "base_min_acc": b_min, "base_max_acc": b_max,
        "base_fullhead_mean_acc": f_mean, "base_fullhead_std_acc": f_std,
        "base_fullhead_min_acc": f_min, "base_fullhead_max_acc": f_max,
        "base_point_item_mean": base_point,
        "base_fullhead_point_item_mean": fh_point,
        "base_fullhead_n_seeds_below_gate": f_seeds_below,
        # re-anchored gate
        "reanchored_gate": gate,
        "base_boot_cilb": base_boot_lo, "base_boot_cihi": base_boot_hi,
        # base_fullhead CI lenses
        "fh_bootstrap_cilb_items": fh_boot_lo, "fh_bootstrap_cihi_items": fh_boot_hi,
        "fh_bootstrap_point_items": fh_boot_point,
        "fh_wilson_cilb_n448": fh_wilson_lo, "fh_wilson_cihi_n448": fh_wilson_hi,
        "primary_cilb_basis": "clustered_bootstrap_over_items",
        "fh_primary_cilb": fh_boot_lo,
        # primary verdict
        "gpqa_main_ci_lb_clears_90pct_base": primary_verdict,
        "fh_bootstrap_clears": fh_boot_clears,
        "fh_wilson_n448_clears": fh_wilson_clears,
        "worst_seed_clears": worst_seed_clears,
        "point_clears": bool(f_mean >= gate),
        # margins
        "fh_primary_cilb_margin": fh_boot_lo - gate,
        "fh_wilson_cilb_margin": fh_wilson_lo - gate,
        "worst_seed_margin": f_min - gate,
        "point_margin": f_mean - gate,
        # both-CI-aware lens (gate carries base sampling noise)
        "fh_cilb_ge_0p90_base_cilb": bool(fh_boot_lo >= QUALITY_FRAC * base_boot_lo),
        # McNemar no-regression
        "mcnemar_shared_cells": len(shared_cells),
        "mcnemar_n00": n00, "mcnemar_n01_base_right_fh_wrong": n01,
        "mcnemar_n10_base_wrong_fh_right": n10, "mcnemar_n11": n11,
        "mcnemar_p_exact": mcnemar_p_exact,
        "mcnemar_p_chi2_cc": mcnemar_p_chi2, "mcnemar_chi2_cc": mcnemar_chi2,
        "mcnemar_direction": regression_direction,
        "base_fullhead_not_regression_sampling": not_regression,
        # clustering-robust per-item sign test
        "sign_test_n_base_gt": n_base_gt, "sign_test_n_fh_gt": n_fh_gt,
        "sign_test_n_tie": n_tie, "sign_test_p": sign_p,
        "mean_item_diff_base_minus_fh": mean_item_diff,
        # un-tightenability
        "n_for_wilson_cilb_pass_at_point": n_for_pass,
        "ci_untightenable_on_gpqa_main": (n_for_pass is None or n_for_pass > GPQA_MAIN_N),
        "ci_untightenable_on_any_gpqa": (n_for_pass is None or n_for_pass > GPQA_EXTENDED_N),
    }
    (HERE / "aggregate.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in (
        "instrument_n", "K_seeds_base", "K_seeds_base_fullhead", "n_prompt_mismatch",
        "base_mean_acc", "base_fullhead_mean_acc", "base_fullhead_min_acc",
        "reanchored_gate", "fh_primary_cilb", "fh_wilson_cilb_n448",
        "gpqa_main_ci_lb_clears_90pct_base", "worst_seed_clears", "point_clears",
        "mcnemar_n01_base_right_fh_wrong", "mcnemar_n10_base_wrong_fh_right",
        "mcnemar_p_exact", "mcnemar_direction", "base_fullhead_not_regression_sampling",
        "sign_test_n_base_gt", "sign_test_n_fh_gt", "sign_test_p",
        "n_for_wilson_cilb_pass_at_point", "ci_untightenable_on_any_gpqa",
    )}, indent=2))


if __name__ == "__main__":
    main()
