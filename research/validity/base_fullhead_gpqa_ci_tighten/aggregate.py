#!/usr/bin/env python3
"""PR #589 — aggregate the base_fullhead GPQA-Diamond sampling seed sweep into the
CI-tighten verdict `gpqa_ci_lb_clears_0471`.

Reports EVERY lens, no thumb on the scale:
  - per-seed accuracy, mean +/- std (sample, ddof=1), worst single seed
  - item-level Wilson 95% CI-lb at n=198 (the irreducible binomial; #564/#574 lens)
  - clustered bootstrap 95% CI-lb over the 198 items (resample items; each item's
    mean-over-seeds correctness) -- the statistically-correct CI on the population
    sampling-protocol accuracy
  - pooled Wilson 95% CI-lb over all K*198 draws (anti-conservative: ignores item
    repetition; a "tightest defensible single number" reference)
  - seed-mean t-CI 95% lb over the K seed accuracies (decode/seed-noise-only lens)

Verdict = does the CI-lb (primary = clustered bootstrap over items) AND the worst
single seed stay >= 0.471?  Also reports the pooled-Wilson and seed-mean verdicts.
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics as st
from pathlib import Path

HERE = Path("/workspace/senpai/target/research/validity/base_fullhead_gpqa_ci_tighten")
RES = HERE / "results"
GATE = 0.471
Z = 1.959963984540054  # 95% two-sided


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


def load_seeds():
    rows = []
    for f in sorted(glob.glob(str(RES / "bf_gpqa_sampled_mt8_s*.json"))):
        d = json.load(open(f))
        rows.append(d)
    return rows


def main():
    rows = load_seeds()
    if not rows:
        raise SystemExit("no seed result files found")

    # per-item correctness across seeds (assert byte-identical item set via prompt_sha)
    item_sha = {}
    per_item = {}  # id -> list[int correct]
    seed_accs = []
    seed_table = []
    total_correct = 0
    total_scored = 0
    n_prompt_mismatch = 0
    for d in rows:
        seed = d["sampling_seed"]
        acc = d["accuracy"]
        seed_accs.append(acc)
        seed_table.append({
            "sampling_seed": seed, "accuracy": acc,
            "n_scored": d["n_scored"], "n_correct": d["n_correct"],
            "n_empty": d.get("n_empty"), "empty_rate": d.get("empty_rate"),
            "min_tokens": d.get("min_tokens"),
            "below_gate": bool(acc < GATE),
        })
        total_correct += d["n_correct"]
        total_scored += d["n_scored"]
        for r in d["per_sample"]:
            sid = r["id"]
            sha = r.get("prompt_sha")
            if sid in item_sha and item_sha[sid] != sha:
                n_prompt_mismatch += 1
            item_sha[sid] = sha
            # only count scored (non-error) items in the per-item correctness vector
            if r.get("value") in ("C", "I"):
                per_item.setdefault(sid, []).append(1 if r["correct"] else 0)

    K = len(rows)
    n_items = len(per_item)
    mean_a = st.mean(seed_accs)
    std_a = st.stdev(seed_accs) if K > 1 else 0.0
    min_a = min(seed_accs)
    max_a = max(seed_accs)
    n_seeds_below = sum(1 for a in seed_accs if a < GATE)

    # item mean-over-seeds correctness
    item_ids = sorted(per_item)
    item_means = [sum(per_item[i]) / len(per_item[i]) for i in item_ids]
    point_item_mean = sum(item_means) / len(item_means)  # == mean_a if every seed scores all items

    # (A) item-level Wilson at n=198 on the pooled point (round mean to nearest k)
    n198 = n_items
    k198 = round(point_item_mean * n198)
    wilson_n198_lo = wilson_lo(k198, n198)
    wilson_n198_hi = wilson_hi(k198, n198)

    # (B) pooled Wilson over all K*198 draws (anti-conservative reference)
    wilson_pooled_lo = wilson_lo(total_correct, total_scored)
    wilson_pooled_hi = wilson_hi(total_correct, total_scored)

    # (C) clustered bootstrap over items (primary CI-lb on accuracy)
    import random
    rng = random.Random(20260617)
    B = 20000
    M = len(item_means)
    boot = []
    for _ in range(B):
        s = 0.0
        for _ in range(M):
            s += item_means[rng.randrange(M)]
        boot.append(s / M)
    boot.sort()
    boot_lo = boot[int(0.025 * B)]
    boot_hi = boot[int(0.975 * B)]
    boot_point = sum(boot) / B

    # (D) seed-mean t-CI (decode/seed-noise only)
    if K > 1:
        tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
                 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201,
                 13: 2.179, 14: 2.160, 15: 2.145}.get(K, 1.96)
        se = std_a / math.sqrt(K)
        seed_mean_lo = mean_a - tcrit * se
        seed_mean_hi = mean_a + tcrit * se
    else:
        seed_mean_lo = seed_mean_hi = float("nan")

    # n needed for a Wilson lb to clear the gate at this point estimate
    def n_for_wilson_pass(p, gate=GATE):
        for n in range(n198, 5000):
            if wilson_lo(round(p * n), n) >= gate:
                return n
        return None
    n_for_pass = n_for_wilson_pass(point_item_mean)

    primary_cilb = boot_lo
    verdict = bool(primary_cilb >= GATE and min_a >= GATE)

    out = {
        "pr": 589,
        "analysis_only": True,
        "official_tps": 0,
        "no_hf_job": True,
        "gate_abs": GATE,
        "task": "gpqa_diamond",
        "n_items": n_items,
        "K_seeds": K,
        "protocol": {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 64,
                     "min_tokens": 8, "source": "generation_config.json (lewtun #31)"},
        "serve": "base_fullhead surgical+fold (FA_SLIDING+SURGICAL_ATTN_USE_3D_OFF+PLE_FOLD), dev307",
        "n_prompt_mismatch": n_prompt_mismatch,
        "seed_table": seed_table,
        "per_seed_acc": seed_accs,
        "mean_acc": mean_a,
        "std_acc": std_a,
        "min_seed_acc": min_a,
        "max_seed_acc": max_a,
        "n_seeds_below_gate": n_seeds_below,
        "frac_seeds_below_gate": n_seeds_below / K,
        "point_item_mean": point_item_mean,
        # CI lenses
        "wilson_cilb_n198": wilson_n198_lo,
        "wilson_cihi_n198": wilson_n198_hi,
        "wilson_cilb_pooled": wilson_pooled_lo,
        "wilson_cihi_pooled": wilson_pooled_hi,
        "pooled_k": total_correct, "pooled_n": total_scored,
        "bootstrap_cilb_items": boot_lo,
        "bootstrap_cihi_items": boot_hi,
        "bootstrap_point_items": boot_point,
        "seed_mean_cilb": seed_mean_lo,
        "seed_mean_cihi": seed_mean_hi,
        # verdicts
        "primary_cilb_basis": "clustered_bootstrap_over_items",
        "primary_cilb": primary_cilb,
        "gpqa_ci_lb_clears_0471": verdict,
        "worst_seed_clears_0471": bool(min_a >= GATE),
        "wilson_n198_clears_0471": bool(wilson_n198_lo >= GATE),
        "wilson_pooled_clears_0471": bool(wilson_pooled_lo >= GATE),
        "seed_mean_clears_0471": bool(seed_mean_lo >= GATE),
        "point_clears_0471": bool(mean_a >= GATE),
        # how far under
        "primary_cilb_margin": primary_cilb - GATE,
        "worst_seed_margin": min_a - GATE,
        "wilson_n198_margin": wilson_n198_lo - GATE,
        "wilson_pooled_margin": wilson_pooled_lo - GATE,
        "seed_mean_margin": seed_mean_lo - GATE,
        "point_margin": mean_a - GATE,
        "n_for_wilson_cilb_pass_at_point": n_for_pass,
        "gpqa_diamond_ceiling_n": n198,
        "ci_untightenable_on_diamond": (n_for_pass is None or n_for_pass > n198),
    }
    (HERE / "aggregate.json").write_text(json.dumps(out, indent=2))
    # console summary
    print(json.dumps({k: out[k] for k in (
        "K_seeds", "n_items", "mean_acc", "std_acc", "min_seed_acc", "max_seed_acc",
        "n_seeds_below_gate", "wilson_cilb_n198", "bootstrap_cilb_items",
        "wilson_cilb_pooled", "seed_mean_cilb", "primary_cilb",
        "gpqa_ci_lb_clears_0471", "worst_seed_clears_0471", "point_clears_0471",
        "n_for_wilson_cilb_pass_at_point", "n_prompt_mismatch",
    )}, indent=2))


if __name__ == "__main__":
    main()
