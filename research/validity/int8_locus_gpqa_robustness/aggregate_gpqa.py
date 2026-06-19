#!/usr/bin/env python3
"""PR #717 -- aggregate the INT8-LOCUS GPQA-Diamond #31-SAMPLED 30-seed sweep into the
gate-robustness verdict: does GPQA-D clear 0.471 COMFORTABLY, or reproduce the int4-body
MARGINAL-TIE pathology (#696: point clears but a CI lens straddles + the point drifts down)?

Reports EVERY lens (no thumb on the scale), the EXACT #696 machinery, PLUS the programmatic
@10/@21/@30 drift trajectory the PR asks for:
  - per-seed accuracy, mean +/- std (ddof=1), worst single seed, n_seeds_below_gate
  - pooled Wilson 95% lo over all K*198 draws  -> the PR's literal primary metric
      (gpqa_d_int8locus_sampled_wilson_lo_30seed). ANTI-CONSERVATIVE.
  - seed-mean t-CI 95% lo over the K seed accuracies -> CONDITIONS ON the fixed Diamond-198
      instrument; the operationally-relevant fixed-benchmark gate CI.
  - clustered bootstrap 95% lo over the 198 items -> generalize beyond Diamond; UN-tightenable.
  - item-level Wilson 95% lo at n=198 on the point -> irreducible binomial ceiling.
  - TRAJECTORY @K in {10,21,30}: pooled point, pooled-Wilson lo, seed-mean lo computed on the
      FIRST K seeds (sorted by sampling_seed). Mirrors #696's int4-body drift checkpoints
      (0.4894@10 -> 0.4822@21 -> 0.4783@30) so the two arms are directly comparable.

COMFORTABLE  := point_30 >= gate AND pooled_lo_30 >= gate AND seed_mean_lo_30 >= gate
                AND point stable/rising (point_30 >= point_10).
MARGINAL_TIE := point_30 >= gate AND (NOT both CI lenses clear OR point drifts down).
REAL_BELOW   := point_30 < gate (the point itself fails -- worse than a tie).

pct_of_base uses the gate denominator base_sampled_3seed = 0.5236 (qi24h8zx), same as #696,
so int8-locus pct is directly comparable to the int4-body number. Gate = 0.471.
"""
from __future__ import annotations

import glob
import json
import math
import random
import statistics as st
from pathlib import Path

HERE = Path("/workspace/senpai/target/research/validity/int8_locus_gpqa_robustness")
RES = HERE / "results_gpqa"
GATE = 0.471
BASE_SAMPLED_3SEED = 0.5236
# #696 int4-body reference (W&B g5lma5qf, gpqa_30seed.json) -- the adjudication anchor.
# All numbers verbatim from PR #717 body / #696 banked verdict.
INT4_BODY = {
    "point_30seed": 0.4783,
    "pooled_wilson_lo_30seed": 0.4656,     # FAILS 0.471
    "seed_mean_lo_30seed": 0.46995,        # FAILS 0.471
    "greedy_point": 0.4697,                # FAILS 0.471
    "point_at_10": 0.4894, "point_at_21": 0.4822, "point_at_30": 0.4783,  # drifts DOWN
    "pooled_clears": False, "seed_mean_clears": False,
    "verdict_leg": "MARGINAL_TIE (point clears, both CI lenses straddle below, point drifts down)",
}
INT4_BODY_POINT_30SEED = INT4_BODY["point_30seed"]
Z = 1.959963984540054
TRAJ_CHECKPOINTS = (10, 21, 30)

# two-sided 95% t critical values (df = K-1).
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


def seed_mean_ci(accs):
    """(mean, lo, hi, half_width) two-sided 95% t-CI over the seed accuracies."""
    K = len(accs)
    mean_a = st.mean(accs)
    if K < 2:
        return mean_a, float("nan"), float("nan"), float("nan")
    std_a = st.stdev(accs)
    t = TCRIT.get(K, 1.96)
    hw = t * std_a / math.sqrt(K)
    return mean_a, mean_a - hw, mean_a + hw, hw


def trajectory_point(rows_sorted, K):
    """@K checkpoint on the FIRST K seeds (sorted by sampling_seed): pooled point /
    pooled-Wilson lo / seed-mean lo. Pooled point == seed-mean (balanced 198xK)."""
    sub = rows_sorted[:K]
    tc = sum(d["n_correct"] for d in sub)
    ts = sum(d["n_scored"] for d in sub)
    accs = [d["accuracy"] for d in sub]
    pooled_point = tc / ts if ts else float("nan")
    p_lo = wilson_lo(tc, ts)
    sm_mean, sm_lo, sm_hi, sm_hw = seed_mean_ci(accs)
    return {
        "K": K, "pooled_point": pooled_point, "seed_mean": sm_mean,
        "pooled_wilson_lo": p_lo, "seed_mean_lo": sm_lo,
        "seed_mean_half_width": sm_hw,
        "pooled_clears_0471": bool(p_lo >= GATE),
        "seed_mean_clears_0471": bool(sm_lo >= GATE),
        "point_clears_0471": bool(pooled_point >= GATE),
        "pooled_n": ts, "pooled_k": tc,
    }


def _read_greedy():
    """Instruction #5: the greedy (temperature=0) GPQA-D point on the SAME int8-locus server
    is the precision anchor -- reported alongside the #31-sampled basis and the
    basis-degeneration contingency reference. Returns (point, n_correct, n_scored, empty_rate)
    or None if the greedy anchor has not been written yet."""
    gf = RES / "gpqa_int8locus_greedy.json"
    if not gf.exists():
        return None
    d = json.load(open(gf))
    return {
        "greedy_point": d["accuracy"],
        "greedy_n_correct": d["n_correct"],
        "greedy_n_scored": d["n_scored"],
        "greedy_n_empty": d.get("n_empty"),
        "greedy_empty_rate": d.get("empty_rate"),
        "greedy_wilson_lo": wilson_lo(d["n_correct"], d["n_scored"]),
        "greedy_clears_0471": bool(d["accuracy"] >= GATE),
    }


def main():
    rows = []
    for f in sorted(glob.glob(str(RES / "gpqa_int8locus_sampled_s*.json")),
                    key=lambda p: int(Path(p).stem.split("_s")[-1])):
        rows.append(json.load(open(f)))
    if not rows:
        raise SystemExit("no seed result files found")
    rows_sorted = sorted(rows, key=lambda d: d["sampling_seed"])
    greedy = _read_greedy()

    per_item = {}   # id -> list[int]
    item_sha = {}
    seed_accs, seed_table = [], []
    total_correct = total_scored = n_prompt_mismatch = 0
    for d in rows_sorted:
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

    K = len(rows_sorted)
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
    _, seed_lo, seed_hi, seed_hw = seed_mean_ci(seed_accs)

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

    # --- TRAJECTORY @10/@21/@30 (drift signature) ---
    traj = [trajectory_point(rows_sorted, kk) for kk in TRAJ_CHECKPOINTS if kk <= K]
    p10 = next((t["pooled_point"] for t in traj if t["K"] == 10), None)
    p21 = next((t["pooled_point"] for t in traj if t["K"] == 21), None)
    p30 = next((t["pooled_point"] for t in traj if t["K"] == 30), None)
    drift_10_30 = (p30 - p10) if (p10 is not None and p30 is not None) else None
    # stable/rising := the 30-seed point did NOT fall below the 10-seed point (no downward drift)
    point_stable_or_rising = bool(drift_10_30 is not None and drift_10_30 >= 0.0)

    # --- VERDICT booleans (single source of truth; log_wandb assembles the string) ---
    point_clears = bool(point_item_mean >= GATE)
    pooled_clears = bool(pooled_lo >= GATE)
    seed_mean_clears = bool(seed_lo >= GATE)
    both_ci_clear = bool(pooled_clears and seed_mean_clears)
    comfortable = bool(point_clears and both_ci_clear and point_stable_or_rising)
    marginal_tie = bool(point_clears and not comfortable)   # point clears but CI straddles / drifts
    real_below = bool(not point_clears)
    if comfortable:
        verdict = "INT8_GPQA_COMFORTABLE"
    elif marginal_tie:
        verdict = "INT8_GPQA_MARGINAL_TIE"
    else:
        verdict = "INT8_GPQA_REAL_BELOW"
    # which lens straddles below the gate (for the report)
    straddling_lenses = [name for name, lo in (
        ("pooled_wilson", pooled_lo), ("seed_mean", seed_lo),
        ("bootstrap_items", boot_lo), ("wilson_n198", w198_lo)) if lo < GATE]

    pct = 100 * point_item_mean / BASE_SAMPLED_3SEED
    pooled_lo_pct = 100 * pooled_lo / BASE_SAMPLED_3SEED
    seed_lo_pct = 100 * seed_lo / BASE_SAMPLED_3SEED

    out = {
        "pr": 717, "analysis_only": True, "official_tps": 0, "no_hf_job": True, "fires": 0,
        "gate_abs": GATE, "base_sampled_3seed": BASE_SAMPLED_3SEED,
        "task": "gpqa_diamond",
        "arm": "int8-locus (int4-g128 body skeleton + int8 L14-27 + int4-g128 lm_head; "
               "in-memory RTN fake-quant on bf16 qat-unquantized base)",
        "fern_recipe_ref": "nmjvtfov (#659)",
        "int8_group_size": int(__import__("os").environ.get("FAKEQUANT_INT8_GROUP", "128")),
        "n_items": n_items, "K_seeds": K, "n_prompt_mismatch": n_prompt_mismatch,
        "protocol": {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 64,
                     "min_tokens": 8, "max_tokens": 3072, "dataset_seed": 12345,
                     "source": "generation_config.json (#31), #696 instrument verbatim"},
        "seed_table": seed_table, "per_seed_acc": seed_accs,
        "mean_acc": mean_a, "std_acc": std_a, "min_seed_acc": min_a, "max_seed_acc": max_a,
        "n_seeds_below_gate": n_below, "frac_seeds_below_gate": n_below / K,
        "point_item_mean": point_item_mean, "pct_of_base": pct,
        # --- greedy precision anchor (instruction #5; basis-degeneration contingency) ---
        "greedy_point": (greedy or {}).get("greedy_point"),
        "greedy_wilson_lo": (greedy or {}).get("greedy_wilson_lo"),
        "greedy_clears_0471": (greedy or {}).get("greedy_clears_0471"),
        "greedy_n_correct": (greedy or {}).get("greedy_n_correct"),
        "greedy_n_scored": (greedy or {}).get("greedy_n_scored"),
        "greedy_empty_rate": (greedy or {}).get("greedy_empty_rate"),
        "sampled_minus_greedy_delta": (
            point_item_mean - greedy["greedy_point"] if greedy else None),
        # --- CI lenses (lo / hi / clears / margin) ---
        "pooled_wilson_lo": pooled_lo, "pooled_wilson_hi": pooled_hi,
        "pooled_k": total_correct, "pooled_n": total_scored,
        "pooled_wilson_lo_pct_of_base": pooled_lo_pct,
        "seed_mean_lo": seed_lo, "seed_mean_hi": seed_hi, "seed_mean_half_width": seed_hw,
        "seed_mean_lo_pct_of_base": seed_lo_pct,
        "bootstrap_items_lo": boot_lo, "bootstrap_items_hi": boot_hi,
        "wilson_n198_lo": w198_lo, "wilson_n198_hi": w198_hi,
        # PR primary metric
        "gpqa_d_int8locus_sampled_wilson_lo_30seed": pooled_lo,
        # clears bools
        "pooled_wilson_clears_0471": pooled_clears,
        "seed_mean_clears_0471": seed_mean_clears,
        "bootstrap_items_clears_0471": bool(boot_lo >= GATE),
        "wilson_n198_clears_0471": bool(w198_lo >= GATE),
        "worst_seed_clears_0471": bool(min_a >= GATE),
        "point_clears_0471": point_clears,
        "both_ci_lenses_clear": both_ci_clear,
        "straddling_lenses": straddling_lenses,
        # margins
        "pooled_wilson_margin": pooled_lo - GATE,
        "seed_mean_margin": seed_lo - GATE,
        "bootstrap_items_margin": boot_lo - GATE,
        "point_margin": point_item_mean - GATE,
        "n_for_pooled_wilson_pass_at_point": n_for_pooled_pass(point_item_mean),
        "diamond_ceiling_n": n_items,
        "ci_untightenable_on_diamond_population": bool(boot_lo < GATE),
        # --- trajectory / drift ---
        "trajectory": traj,
        "point_at_10": p10, "point_at_21": p21, "point_at_30": p30,
        "point_drift_10_to_30": drift_10_30,
        "point_stable_or_rising": point_stable_or_rising,
        # --- cross-arm delta vs #696 int4-body (g5lma5qf) ---
        "int4_body_reference": INT4_BODY,
        "int4_body_point_30seed": INT4_BODY_POINT_30SEED,
        "int8_vs_int4_body_gpqa_delta": point_item_mean - INT4_BODY_POINT_30SEED,
        "int8_vs_int4_body_pooled_lo_delta": pooled_lo - INT4_BODY["pooled_wilson_lo_30seed"],
        "int8_vs_int4_body_seed_mean_lo_delta": seed_lo - INT4_BODY["seed_mean_lo_30seed"],
        "moves_up_and_out_of_int4_tie": bool(comfortable),
        # --- verdict ---
        "verdict": verdict,
        "comfortable": comfortable, "marginal_tie": marginal_tie, "real_below": real_below,
        "int8_gpqa_comfortable": int(comfortable),
    }
    (HERE / "gpqa_int8locus_30seed.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in (
        "verdict", "K_seeds", "n_items", "n_prompt_mismatch", "mean_acc", "std_acc",
        "min_seed_acc", "max_seed_acc", "n_seeds_below_gate", "point_item_mean", "pct_of_base",
        "greedy_point", "greedy_clears_0471", "sampled_minus_greedy_delta",
        "pooled_wilson_lo", "gpqa_d_int8locus_sampled_wilson_lo_30seed",
        "pooled_wilson_clears_0471", "seed_mean_lo", "seed_mean_clears_0471",
        "bootstrap_items_lo", "bootstrap_items_clears_0471", "worst_seed_clears_0471",
        "point_at_10", "point_at_21", "point_at_30", "point_drift_10_to_30",
        "point_stable_or_rising", "straddling_lenses", "int8_vs_int4_body_gpqa_delta",
        "comfortable", "marginal_tie",
    )}, indent=2))


if __name__ == "__main__":
    main()
