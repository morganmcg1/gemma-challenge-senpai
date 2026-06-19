#!/usr/bin/env python3
"""PR #696 -- aggregate the int4-body AIME #31-SAMPLED seed sweep (the TEST leg) into the
basis-correction verdict: does AIME RECOVER toward the 0.420 gate on the sampled basis, or
stay below (the genuinely-harder leg)?

Same machinery as aggregate_gpqa.py, on the int4 g32 body-isolation arm (base_fullhead),
so the joint {GPQA,AIME} verdict is config- AND engine-consistent. Reports EVERY lens:
  - per-seed maj@1 accuracy, mean +/- std (ddof=1), worst seed, n_seeds_below_gate
  - pooled Wilson 95% lo over K*60 (item,seed) draws (ANTI-CONSERVATIVE; the PR's headline
      analogue to the gpqa pooled metric -- shrinks ~1/sqrt(K) though the SAME 60 problems
      repeat every seed)
  - seed-mean t-CI 95% lo over the K seed accuracies (conditions on the fixed AIME-60
      instrument; the only stochastic element is decode RNG)
  - clustered bootstrap 95% lo over the 60 problems (resample problems; each problem's
      mean-over-seeds correctness -> generalize beyond AIME-60; UN-tightenable by seeds)
  - item-level Wilson 95% lo at n=60 on the point -- the irreducible binomial ceiling

GATE = 0.420 (= 0.90 x 0.4667 vanilla-base GREEDY, the banked #515 AIME bar). The greedy
anchor on THIS arm is measured alongside so the greedy->sampled shift is a within-config
contrast. base denominators reported: 0.4667 (greedy bar source) and 0.4833 (bf16 control,
#679, 60/60 bit-exact). NOTE: base is GREEDY; a fully #31-consistent pct would need a
vanilla-base SAMPLED denominator (not measured here -- flagged in the verdict).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
import statistics as st
from pathlib import Path

HERE = Path("/workspace/senpai/target/research/validity/int4_quality_31basis_confirm")
RES = HERE / "results_aime"

# PR #696 fix: numerator protocol MUST match the denominator protocol or pct_of_base is
# apples-to-oranges. Two internally-consistent regimes (the AIME-protocol question the PR
# flags for the advisor):
#   think   -- the cited 0.4667/0.3500 wall regime. base GREEDY 0.4667 (#515 bar source),
#              bf16 control 0.4833 (ubel #679, banked). gate 0.420 = 0.90 x 0.4667.
#   nothink -- the #580 floor regime (yokbmy9i): harness-grounded bf16 base 0.10 (no-thinking
#              greedy maj@1 min8), gate 0.090 = 0.90 x 0.10. int4-body greedy 0.1167 (#580).
REGIME = {
    "think":   {"base_greedy": 0.4667, "base_bf16_control": 0.4833, "gate": 0.420},
    "nothink": {"base_greedy": 0.10,   "base_bf16_control": 0.10,   "gate": 0.090},
}
Z = 1.959963984540054

TCRIT = {
    2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306,
    10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131,
    17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093, 21: 2.086, 22: 2.080, 23: 2.074,
    24: 2.069, 25: 2.064, 26: 2.060, 27: 2.056, 28: 2.052, 29: 2.048, 30: 2.045,
}


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    phat = k / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return (centre - margin) / denom, (centre + margin) / denom


def load(path: Path) -> dict:
    return json.load(open(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=("think", "nothink"), default="think")
    args = ap.parse_args()
    R = REGIME[args.protocol]
    GATE = R["gate"]
    BASE_GREEDY = R["base_greedy"]
    BASE_BF16_CONTROL = R["base_bf16_control"]
    tag = args.protocol

    greedy = None
    gp = RES / f"int4body_aime_{tag}_greedy.json"
    if gp.exists():
        greedy = load(gp)

    rows = []
    for f in sorted(glob.glob(str(RES / f"int4body_aime_{tag}_sampled_s*.json")),
                    key=lambda p: int(Path(p).stem.split("_s")[-1])):
        rows.append((int(Path(f).stem.split("_s")[-1]), load(f)))
    if not rows:
        raise SystemExit(f"no AIME {tag} sampled seed files found")

    per_item = {}     # id -> list[int] maj_correct over seeds
    seed_accs, seed_table = [], []
    total_correct = total_scored = 0
    n_extract_fail = 0
    for seed, d in rows:
        acc = d["maj_k_accuracy"]
        seed_accs.append(acc)
        seed_table.append({"seed": seed, "maj_k_accuracy": acc, "n_correct_maj": d["n_correct_maj"],
                           "n_problems": d["n_problems"], "mean_pass_rate": d.get("mean_pass_rate"),
                           "extract_fail_rate": d.get("extract_fail_rate"),
                           "below_gate": bool(acc < GATE)})
        total_correct += d["n_correct_maj"]
        total_scored += d["n_problems"]
        n_extract_fail += sum(1 for r in d["per_problem"] for a in r["answers"] if a is None)
        for r in d["per_problem"]:
            per_item.setdefault(r["id"], []).append(1 if r["maj_correct"] else 0)

    K = len(rows)
    n_items = len(per_item)
    mean_a = st.mean(seed_accs)
    std_a = st.stdev(seed_accs) if K > 1 else 0.0
    min_a, max_a = min(seed_accs), max(seed_accs)
    n_below = sum(1 for a in seed_accs if a < GATE)

    item_ids = sorted(per_item)
    item_means = [sum(per_item[i]) / len(per_item[i]) for i in item_ids]
    point_item_mean = sum(item_means) / len(item_means)

    pooled_lo, pooled_hi = wilson(total_correct, total_scored)

    if K > 1:
        t = TCRIT.get(K, 1.96)
        se = std_a / math.sqrt(K)
        seed_lo, seed_hi = mean_a - t * se, mean_a + t * se
    else:
        seed_lo = seed_hi = float("nan")

    rng = random.Random(20260619)
    B = 20000
    M = len(item_means)
    boot = []
    for _ in range(B):
        s = sum(item_means[rng.randrange(M)] for _ in range(M))
        boot.append(s / M)
    boot.sort()
    boot_lo, boot_hi = boot[int(0.025 * B)], boot[int(0.975 * B)]

    k_n = round(point_item_mean * n_items)
    wN_lo, wN_hi = wilson(k_n, n_items)

    pct_greedy = 100 * point_item_mean / BASE_GREEDY
    pct_bf16 = 100 * point_item_mean / BASE_BF16_CONTROL
    pooled_lo_pct = 100 * pooled_lo / BASE_GREEDY

    out = {
        "pr": 696, "analysis_only": True, "official_tps": 0, "no_hf_job": True, "fires": 0,
        "task": "aime_2024_2025", "arm": "int4-body-isolated (g32 QAT body + bf16 262k head)",
        "protocol_regime": tag,
        "gate_abs": GATE, "base_greedy": BASE_GREEDY, "base_bf16_control": BASE_BF16_CONTROL,
        "n_items": n_items, "K_seeds": K, "n_extract_fail": n_extract_fail,
        "protocol": {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 64,
                     "min_tokens": 8, "max_tokens": 6144, "k": 1,
                     "thinking": (tag == "think"), "no_thinking": (tag == "nothink"),
                     "source": "generation_config.json (#31)"},
        "greedy_anchor_same_arm": (greedy["maj_k_accuracy"] if greedy else None),
        "greedy_anchor_correct": (greedy["n_correct_maj"] if greedy else None),
        "greedy_anchor_n": (greedy["n_problems"] if greedy else None),
        "seed_table": seed_table, "per_seed_acc": seed_accs,
        "mean_acc": mean_a, "std_acc": std_a, "min_seed_acc": min_a, "max_seed_acc": max_a,
        "n_seeds_below_gate": n_below, "frac_seeds_below_gate": n_below / K,
        "point_item_mean": point_item_mean,
        "pct_of_base_greedy": pct_greedy, "pct_of_base_bf16": pct_bf16,
        "pooled_wilson_lo": pooled_lo, "pooled_wilson_hi": pooled_hi,
        "pooled_k": total_correct, "pooled_n": total_scored,
        "pooled_wilson_lo_pct_of_base_greedy": pooled_lo_pct,
        "seed_mean_lo": seed_lo, "seed_mean_hi": seed_hi,
        "bootstrap_items_lo": boot_lo, "bootstrap_items_hi": boot_hi,
        "wilson_nitems_lo": wN_lo, "wilson_nitems_hi": wN_hi,
        # TEST metric (PR): does AIME recover on #31?
        "aime_sampled_pct_of_base": pct_greedy,
        "aime_sampled_point": point_item_mean,
        # clears bools (vs 0.420 absolute gate)
        "point_clears_0420": bool(point_item_mean >= GATE),
        "pooled_wilson_clears_0420": bool(pooled_lo >= GATE),
        "seed_mean_clears_0420": bool(seed_lo >= GATE),
        "bootstrap_items_clears_0420": bool(boot_lo >= GATE),
        "worst_seed_clears_0420": bool(min_a >= GATE),
        # 90%-of-base bools (the #515 framing)
        "point_clears_90pct_greedy": bool(pct_greedy >= 90.0),
        "pooled_lo_clears_90pct_greedy": bool(pooled_lo_pct >= 90.0),
        # margins
        "point_margin": point_item_mean - GATE,
        "pooled_wilson_margin": pooled_lo - GATE,
        "seed_mean_margin": seed_lo - GATE,
    }
    (HERE / f"aime_sampled_{tag}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in (
        "protocol_regime",
        "K_seeds", "n_items", "n_extract_fail", "greedy_anchor_same_arm",
        "mean_acc", "std_acc", "min_seed_acc", "max_seed_acc", "n_seeds_below_gate",
        "point_item_mean", "pct_of_base_greedy", "pct_of_base_bf16",
        "pooled_wilson_lo", "pooled_wilson_clears_0420", "seed_mean_lo", "seed_mean_clears_0420",
        "bootstrap_items_lo", "point_clears_0420", "worst_seed_clears_0420",
    )}, indent=2))


if __name__ == "__main__":
    main()
