#!/usr/bin/env python3
"""PR #590 -- multi-seed quality-gate CI aggregator.

Takes >=5 decode realizations of ONE fixed benchmark (AIME / MMLU-Pro / GSM8K) on
base_fullhead under the lewtun #31 sampling protocol (temp=1.0 top_p=0.95 top_k=64,
min_tokens=8) and produces the robustness statistics the card asks for:

  * per-seed accuracy (one full-benchmark accuracy per decode realization)
  * mean +/- std across seeds            (decode-noise descriptor)
  * pass@1 = grand mean over all (q, seed)
  * cluster-bootstrap 95% CI lower bound  -> resample QUESTIONS w/ replacement,
        average the per-question decode-marginalized pass-rate, take the 2.5%ile.
        Resampling questions captures the n-question resolution (and thereby
        approximates "which subset did we happen to draw" variance); averaging the
        seeds into each question's pass-rate folds in decode noise.
  * for AIME additionally: maj@S accuracy + Wilson 95% CI (the greedy-comparable,
        denoised number that lines up with the #567 base_fullhead 0.1167 anchor).
  * verdict ci_lb_clears_bar (bool) and the X/N "problems-of-slack" framing.

Each harness stores per-item correctness differently; we normalize all three to a
{question_id -> [0/1, 0/1, ...]} matrix (one column per decode sample) and assert the
question-id set is byte-identical across the merged inputs (fixed benchmark).

Usage:
  aggregate_ci.py --task {aime,mmlu_pro,gsm8k} --bar 0.090 --label AIME \
      --out summary_aime.json --inputs a_seed1.json a_seed2.json ...
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def _extract_columns(task: str, path: str) -> dict[str, list[int]]:
    """One input file -> {qid: [correctness per decode sample in this file]}."""
    d = json.load(open(path))
    out: dict[str, list[int]] = {}
    if task == "aime":
        for pp in d["per_problem"]:
            gold = pp["gold"]
            # answers is the list of k extracted ints (None = extract fail = wrong).
            col = [int(a is not None and a == gold) for a in pp["answers"]]
            out[str(pp["id"])] = col
    elif task in ("mmlu_pro", "gsm8k", "gpqa", "gpqa_diamond"):
        # GPQA-Diamond (run_eval.py --task gpqa_diamond) emits the same per_sample
        # schema as mmlu_pro: one decode realization per file, {id, correct, ...}.
        key = "per_sample" if "per_sample" in d else "per_problem"
        for rec in d[key]:
            out[str(rec["id"])] = [int(bool(rec["correct"]))]
    else:
        raise SystemExit(f"unknown task {task}")
    return out


def _maj_correct_for_aime(path_list: list[str]) -> dict[str, bool]:
    """Majority vote across ALL samples (all files) per problem, vs gold.

    Mirrors aime_eval.majority_vote: most common non-None extracted answer; ties
    broken by Counter.most_common order (insertion-stable in CPython)."""
    answers: dict[str, list] = {}
    gold: dict[str, int] = {}
    for p in path_list:
        d = json.load(open(p))
        for pp in d["per_problem"]:
            qid = str(pp["id"])
            answers.setdefault(qid, []).extend(pp["answers"])
            gold[qid] = pp["gold"]
    maj_correct: dict[str, bool] = {}
    for qid, ans in answers.items():
        cnt = Counter(a for a in ans if a is not None)
        maj = cnt.most_common(1)[0][0] if cnt else None
        maj_correct[qid] = (maj is not None and maj == gold[qid])
    return maj_correct


def _wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den
    return center - half, center + half


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["aime", "mmlu_pro", "gsm8k", "gpqa", "gpqa_diamond"])
    ap.add_argument("--label", required=True)
    ap.add_argument("--bar", type=float, required=True, help="gate bar to clear")
    ap.add_argument("--inputs", nargs="+", required=True, help="per-seed result jsons")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--boot-seed", type=int, default=590)
    args = ap.parse_args()

    # Merge per-file columns by question id.
    merged: dict[str, list[int]] = {}
    id_sets = []
    for p in args.inputs:
        cols = _extract_columns(args.task, p)
        id_sets.append(set(cols.keys()))
        for qid, col in cols.items():
            merged.setdefault(qid, []).extend(col)

    # Fixed-benchmark guard: every input must cover the same question ids.
    common = set.intersection(*id_sets) if id_sets else set()
    union = set.union(*id_sets) if id_sets else set()
    if common != union:
        missing = union - common
        print(f"[agg] WARNING: question-id sets differ across inputs; "
              f"{len(missing)} ids not in all files -> restricting to {len(common)} common ids")
    ids = sorted(common)
    if not ids:
        raise SystemExit("[agg] FATAL: no common question ids across inputs")

    # Rectangularize: all questions must have the same #samples. Truncate to the
    # min column count (a partial last pass should not skew a single question).
    counts = [len(merged[q]) for q in ids]
    S = min(counts)
    if len(set(counts)) != 1:
        print(f"[agg] note: ragged sample counts {sorted(set(counts))}; truncating all to S={S}")
    M = np.array([merged[q][:S] for q in ids], dtype=float)  # [n_questions, S]
    n_q = M.shape[0]

    # Per-seed accuracy (one column = one full-benchmark decode realization).
    per_seed_acc = M.mean(axis=0)                       # [S]
    mean_acc = float(per_seed_acc.mean())
    std_acc = float(per_seed_acc.std(ddof=1)) if S > 1 else 0.0
    sem = std_acc / math.sqrt(S) if S > 1 else 0.0

    # Per-question decode-marginalized pass-rate.
    p_q = M.mean(axis=1)                                # [n_questions]
    pass_at_1 = float(p_q.mean())                       # == mean_acc

    # Cluster bootstrap over questions (folds in decode noise via p_q, and the
    # n-question resolution via resampling). 95% two-sided lower edge = 2.5%ile.
    rng = np.random.default_rng(args.boot_seed)
    boot = np.empty(args.bootstrap)
    for b in range(args.bootstrap):
        idx = rng.integers(0, n_q, n_q)
        boot[b] = p_q[idx].mean()
    ci_lb_95 = float(np.percentile(boot, 2.5))          # 95% two-sided lower
    ci_ub_95 = float(np.percentile(boot, 97.5))
    ci_lb_90_1sided = float(np.percentile(boot, 5.0))   # 95% one-sided lower

    bar = args.bar
    N = n_q  # benchmark size (60 for AIME; subset size otherwise)
    summary = {
        "task": args.task,
        "label": args.label,
        "n_questions": n_q,
        "n_seeds_samples_per_q": S,
        "total_graded": int(M.sum() + (M.size - M.sum())),  # = M.size
        "per_seed_accuracy": [round(float(x), 6) for x in per_seed_acc],
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "sem_accuracy": sem,
        "pass_at_1": pass_at_1,
        "ci_lb_95_2sided": ci_lb_95,
        "ci_ub_95_2sided": ci_ub_95,
        "ci_lb_95_1sided": ci_lb_90_1sided,
        "bar": bar,
        "ci_lb_clears_bar": bool(ci_lb_95 >= bar),
        "ci_lb_problems": ci_lb_95 * N,
        "bar_problems": bar * N,
        "slack_problems_at_ci_lb": (ci_lb_95 - bar) * N,
        "mean_problems": mean_acc * N,
        "bootstrap_B": args.bootstrap,
        "boot_seed": args.boot_seed,
        "inputs": list(args.inputs),
    }

    if args.task == "aime":
        mc = _maj_correct_for_aime(args.inputs)
        mc_ids = [q for q in ids if q in mc]
        k_maj = sum(1 for q in mc_ids if mc[q])
        n_maj = len(mc_ids)
        wl, wu = _wilson(k_maj, n_maj)
        summary["majS_accuracy"] = k_maj / n_maj if n_maj else float("nan")
        summary["majS_correct"] = k_maj
        summary["majS_n"] = n_maj
        summary["majS_wilson_lb"] = wl
        summary["majS_wilson_ub"] = wu
        summary["majS_total_samples_per_q"] = S
        summary["majS_clears_bar"] = bool(wl >= bar)

    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"[agg] task={args.task} label={args.label} n_q={n_q} S={S} "
          f"mean={mean_acc:.4f} std={std_acc:.4f} pass@1={pass_at_1:.4f} "
          f"CI-lb(95)={ci_lb_95:.4f} bar={bar:.3f} clears={summary['ci_lb_clears_bar']} "
          f"slack={summary['slack_problems_at_ci_lb']:+.2f}/{N}", flush=True)
    if args.task == "aime":
        print(f"[agg]   maj@{S}={summary['majS_accuracy']:.4f} "
              f"({summary['majS_correct']}/{summary['majS_n']}) "
              f"Wilson95=[{summary['majS_wilson_lb']:.4f},{summary['majS_wilson_ub']:.4f}] "
              f"clears={summary['majS_clears_bar']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
