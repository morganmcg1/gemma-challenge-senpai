#!/usr/bin/env python3
"""PR #615 -- summarize one stack's 3-eval panel.

Per task (gpqa/mmlu/gsm8k): pool seeds into a {qid: [correct per seed]} matrix,
report per-seed accuracy, pooled mean, std, and a cluster-bootstrap 95% CI lower
bound (resample questions, fold decode noise via per-question pass-rate) -- same
estimator as quality_gates_ci/aggregate_ci.py. Also attach the finish_reason=length
(truncation) rate: GPQA/MMLU from the inspect .eval logs, GSM8K from its JSON.

Usage (under the inspect client venv):
  summarize_stack.py <stack_label> [runs_dir] [logs_dir]
"""
from __future__ import annotations
import glob
import json
import math
import os
import sys
from collections import Counter

import numpy as np

TRUNC = {"max_tokens", "length", "model_length"}
BARS = {"gpqa": 0.471, "mmlu": 0.605, "gsm8k": 0.807}


def boot_ci(p_q: np.ndarray, B: int = 10000, seed: int = 615):
    rng = np.random.default_rng(seed)
    n = len(p_q)
    boot = np.empty(B)
    for b in range(B):
        boot[b] = p_q[rng.integers(0, n, n)].mean()
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def matrix_from_runeval(paths):
    merged = {}
    for p in paths:
        d = json.load(open(p))
        for rec in d["per_sample"]:
            merged.setdefault(str(rec["id"]), []).append(int(bool(rec["correct"])))
    return merged


def matrix_from_gsm8k(paths):
    merged = {}
    for p in paths:
        d = json.load(open(p))
        for rec in d["per_problem"]:
            merged.setdefault(str(rec["id"]), []).append(int(bool(rec["correct"])))
    return merged


def finish_rate_eval_logs(log_dirs):
    from inspect_ai.log import read_eval_log
    n = ntr = 0
    outtok = []
    for ld in log_dirs:
        cands = sorted(glob.glob(os.path.join(ld, "*.eval")))
        if not cands:
            continue
        best, bn = cands[0], -1
        for c in cands:
            try:
                k = len(read_eval_log(c).samples or [])
            except Exception:
                k = -1
            if k > bn:
                best, bn = c, k
        log = read_eval_log(best)
        for s in log.samples or []:
            n += 1
            sr = s.output.choices[0].stop_reason if (s.output and s.output.choices) else None
            if sr in TRUNC:
                ntr += 1
            try:
                if s.output and s.output.usage:
                    outtok.append(int(s.output.usage.output_tokens))
            except Exception:
                pass
    return {"finish_length_rate": (ntr / n) if n else None, "n": n, "n_trunc": ntr,
            "out_tokens_mean": (sum(outtok) / len(outtok)) if outtok else None,
            "out_tokens_max": max(outtok) if outtok else None}


def summarize(task, merged, finish):
    ids = sorted(merged)
    counts = [len(merged[q]) for q in ids]
    S = min(counts) if counts else 0
    M = np.array([merged[q][:S] for q in ids], dtype=float)
    per_seed = M.mean(axis=0)
    p_q = M.mean(axis=1)
    mean = float(per_seed.mean())
    std = float(per_seed.std(ddof=1)) if S > 1 else 0.0
    lb, ub = boot_ci(p_q) if len(p_q) else (float("nan"), float("nan"))
    bar = BARS[task]
    return {
        "task": task, "n_questions": len(ids), "n_seeds": S,
        "per_seed_accuracy": [round(float(x), 6) for x in per_seed],
        "mean_accuracy": mean, "std_accuracy": std,
        "ci_lb_95": lb, "ci_ub_95": ub, "bar": bar,
        "ci_lb_clears_bar": bool(lb >= bar), "mean_clears_bar": bool(mean >= bar),
        **{f"finish_{k}": v for k, v in finish.items()},
    }


def main():
    stack = sys.argv[1]
    runs = sys.argv[2] if len(sys.argv) > 2 else "research/validity/eval_stack_accuracy_validity/runs"
    logs = sys.argv[3] if len(sys.argv) > 3 else "research/validity/eval_stack_accuracy_validity/logs"
    out = {"stack": stack}

    gpqa = sorted(glob.glob(f"{runs}/gpqa_{stack}_s*.json"))
    if gpqa:
        ld = [f"{logs}/gpqa_{stack}_s{p.split('_s')[-1].split('.')[0]}" for p in gpqa]
        out["gpqa"] = summarize("gpqa", matrix_from_runeval(gpqa), finish_rate_eval_logs(ld))
        out["gpqa"]["inputs"] = gpqa

    mmlu = sorted(glob.glob(f"{runs}/mmlu_{stack}_s*.json"))
    if mmlu:
        ld = [f"{logs}/mmlu_{stack}_s{p.split('_s')[-1].split('.')[0]}" for p in mmlu]
        out["mmlu"] = summarize("mmlu", matrix_from_runeval(mmlu), finish_rate_eval_logs(ld))
        out["mmlu"]["inputs"] = mmlu

    gsm = sorted(glob.glob(f"{runs}/gsm8k_{stack}_sampled_s*.json"))
    if gsm:
        trs = [json.load(open(p)).get("truncation_rate") for p in gsm]
        trs = [t for t in trs if t is not None]
        fin = {"finish_length_rate": (sum(trs) / len(trs)) if trs else None}
        out["gsm8k"] = summarize("gsm8k", matrix_from_gsm8k(gsm), fin)
        out["gsm8k"]["inputs"] = gsm

    path = f"{runs}/_summary_{stack}.json"
    json.dump(out, open(path, "w"), indent=2)
    for t in ("gpqa", "mmlu", "gsm8k"):
        if t in out:
            s = out[t]
            print(f"[{stack}] {t}: mean={s['mean_accuracy']:.4f} std={s['std_accuracy']:.4f} "
                  f"CI95=[{s['ci_lb_95']:.4f},{s['ci_ub_95']:.4f}] bar={s['bar']} "
                  f"clears(mean)={s['mean_clears_bar']} clears(CIlb)={s['ci_lb_clears_bar']} "
                  f"finish_len_rate={s.get('finish_finish_length_rate')} n_q={s['n_questions']} S={s['n_seeds']}")
    print(f"[{stack}] wrote {path}")


if __name__ == "__main__":
    main()
