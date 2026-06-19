#!/usr/bin/env python3
"""PR #696 -- aggregate the int4-body GPQA-Diamond #31-SAMPLED seed sweep (target 30 seeds)
into the gate-clear verdict, hardening #692's 10-seed point (0.4869, pooled Wilson lo 0.4649).

Reports EVERY lens (no thumb on the scale), exactly the #589 machinery extended to K=30:
  - per-seed accuracy, mean +/- std (ddof=1), worst single seed, n_seeds_below_gate
  - pooled Wilson 95% lo over all K*198 draws  -> the PR's literal primary metric
      (gpqa_d_sampled_wilson_lo_30seed). ANTI-CONSERVATIVE: treats every (item,seed) as an
      independent Bernoulli trial, so its half-width shrinks ~1/sqrt(K) even though the SAME
      198 items repeat every seed. This is the number the PR asks me to "halve."
  - seed-mean t-CI 95% lo over the K seed accuracies -> CONDITIONS ON the fixed Diamond-198
      instrument; the only stochastic element is decode RNG. Tightenable by seeds. This is the
      operationally-relevant CI for a FIXED-benchmark gate ("score >=0.471 ON GPQA-Diamond").
  - clustered bootstrap 95% lo over the 198 items (resample items; each item's mean-over-seeds
      correctness) -> treats Diamond-198 as a SAMPLE of the GPQA population. UN-tightenable by
      seeds (item-population variance dominates; n=198 ceiling). The statistically-correct CI
      if you want to GENERALIZE beyond Diamond.
  - item-level Wilson 95% lo at n=198 on the point -> the irreducible binomial ceiling.

pct_of_base uses the gate denominator base_sampled_3seed = 0.5236 (qi24h8zx). Gate = 0.471.
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
import statistics as st
from pathlib import Path

HERE = Path("/workspace/senpai/target/research/validity/int4_quality_31basis_confirm")
RES = HERE / "results_gpqa"
GATE = 0.471
BASE_SAMPLED_3SEED = 0.5236
Z = 1.959963984540054

# two-sided 95% t critical values (df = K-1); extended past #589's K<=15 to K=40.
TCRIT = {
    2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306,
    10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131,
    17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093, 21: 2.086, 22: 2.080, 23: 2.074,
    24: 2.069, 25: 2.064, 26: 2.060, 27: 2.056, 28: 2.052, 29: 2.048, 30: 2.045,
    31: 2.042, 32: 2.040, 33: 2.037, 34: 2.035, 35: 2.032, 36: 2.030, 37: 2.028,
    38: 2.026, 39: 2.024, 40: 2.023,
}


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


def main():
    rows = []
    for f in sorted(glob.glob(str(RES / "bf_gpqa_sampled_mt8_s*.json")),
                    key=lambda p: int(Path(p).stem.split("_s")[-1])):
        rows.append(json.load(open(f)))
    if not rows:
        raise SystemExit("no seed result files found")

    per_item = {}   # id -> list[int]
    item_sha = {}
    seed_accs, seed_table = [], []
    total_correct = total_scored = n_prompt_mismatch = 0
    for d in rows:
        seed = d["sampling_seed"]
        acc = d["accuracy"]
        seed_accs.append(acc)
        seed_table.append({"sampling_seed": seed, "accuracy": acc, "n_scored": d["n_scored"],
                           "n_correct": d["n_correct"], "n_empty": d.get("n_empty"),
                           "below_gate": bool(acc < GATE)})
        total_correct += d["n_correct"]
        total_scored += d["n_scored"]
        for r in d["per_sample"]:
            sid = r["id"]; sha = r.get("prompt_sha")
            if sid in item_sha and item_sha[sid] != sha:
                n_prompt_mismatch += 1
            item_sha[sid] = sha
            if r.get("value") in ("C", "I"):
                per_item.setdefault(sid, []).append(1 if r["correct"] else 0)

    K = len(rows)
    n_items = len(per_item)
    mean_a = st.mean(seed_accs)
    std_a = st.stdev(seed_accs) if K > 1 else 0.0
    min_a, max_a = min(seed_accs), max(seed_accs)
    n_below = sum(1 for a in seed_accs if a < GATE)

    item_ids = sorted(per_item)
    item_means = [sum(per_item[i]) / len(per_item[i]) for i in item_ids]
    point_item_mean = sum(item_means) / len(item_means)

    # (A) pooled Wilson over K*198 draws  -- PR primary metric
    pooled_lo = wilson_lo(total_correct, total_scored)
    pooled_hi = wilson_hi(total_correct, total_scored)

    # (B) seed-mean t-CI (conditions on the fixed Diamond instrument)
    if K > 1:
        t = TCRIT.get(K, 1.96)
        se = std_a / math.sqrt(K)
        seed_lo, seed_hi = mean_a - t * se, mean_a + t * se
    else:
        seed_lo = seed_hi = float("nan")

    # (C) clustered bootstrap over items (generalize beyond Diamond)
    rng = random.Random(20260619)
    B = 20000
    M = len(item_means)
    boot = []
    for _ in range(B):
        s = 0.0
        for _ in range(M):
            s += item_means[rng.randrange(M)]
        boot.append(s / M)
    boot.sort()
    boot_lo, boot_hi = boot[int(0.025 * B)], boot[int(0.975 * B)]

    # (D) item-level Wilson at n=198 on the point
    k198 = round(point_item_mean * n_items)
    w198_lo, w198_hi = wilson_lo(k198, n_items), wilson_hi(k198, n_items)

    def n_for_pooled_pass(p, gate=GATE):
        for n in range(n_items, 200000, 99):
            if wilson_lo(round(p * n), n) >= gate:
                return n
        return None

    pct = 100 * point_item_mean / BASE_SAMPLED_3SEED
    pooled_lo_pct = 100 * pooled_lo / BASE_SAMPLED_3SEED
    seed_lo_pct = 100 * seed_lo / BASE_SAMPLED_3SEED

    out = {
        "pr": 696, "analysis_only": True, "official_tps": 0, "no_hf_job": True, "fires": 0,
        "gate_abs": GATE, "base_sampled_3seed": BASE_SAMPLED_3SEED,
        "task": "gpqa_diamond", "arm": "int4-body-isolated (g32 QAT body + bf16 262k head)",
        "n_items": n_items, "K_seeds": K, "n_prompt_mismatch": n_prompt_mismatch,
        "protocol": {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 64,
                     "min_tokens": 8, "max_tokens": 3072, "source": "generation_config.json (#31)"},
        "seed_table": seed_table, "per_seed_acc": seed_accs,
        "mean_acc": mean_a, "std_acc": std_a, "min_seed_acc": min_a, "max_seed_acc": max_a,
        "n_seeds_below_gate": n_below, "frac_seeds_below_gate": n_below / K,
        "point_item_mean": point_item_mean, "pct_of_base": pct,
        # --- CI lenses (lo / hi / clears / margin) ---
        "pooled_wilson_lo": pooled_lo, "pooled_wilson_hi": pooled_hi,
        "pooled_k": total_correct, "pooled_n": total_scored,
        "pooled_wilson_lo_pct_of_base": pooled_lo_pct,
        "seed_mean_lo": seed_lo, "seed_mean_hi": seed_hi, "seed_mean_lo_pct_of_base": seed_lo_pct,
        "bootstrap_items_lo": boot_lo, "bootstrap_items_hi": boot_hi,
        "wilson_n198_lo": w198_lo, "wilson_n198_hi": w198_hi,
        # PR primary metric
        "gpqa_d_sampled_wilson_lo_30seed": pooled_lo,
        # clears bools
        "pooled_wilson_clears_0471": bool(pooled_lo >= GATE),
        "seed_mean_clears_0471": bool(seed_lo >= GATE),
        "bootstrap_items_clears_0471": bool(boot_lo >= GATE),
        "wilson_n198_clears_0471": bool(w198_lo >= GATE),
        "worst_seed_clears_0471": bool(min_a >= GATE),
        "point_clears_0471": bool(point_item_mean >= GATE),
        # margins
        "pooled_wilson_margin": pooled_lo - GATE,
        "seed_mean_margin": seed_lo - GATE,
        "bootstrap_items_margin": boot_lo - GATE,
        "point_margin": point_item_mean - GATE,
        "n_for_pooled_wilson_pass_at_point": n_for_pooled_pass(point_item_mean),
        "diamond_ceiling_n": n_items,
        "ci_untightenable_on_diamond_population": bool(boot_lo < GATE),
    }
    (HERE / "gpqa_30seed.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in (
        "K_seeds", "n_items", "n_prompt_mismatch", "mean_acc", "std_acc", "min_seed_acc",
        "max_seed_acc", "n_seeds_below_gate", "point_item_mean", "pct_of_base",
        "pooled_wilson_lo", "gpqa_d_sampled_wilson_lo_30seed", "pooled_wilson_clears_0471",
        "seed_mean_lo", "seed_mean_clears_0471", "bootstrap_items_lo",
        "bootstrap_items_clears_0471", "worst_seed_clears_0471",
    )}, indent=2))


if __name__ == "__main__":
    main()
