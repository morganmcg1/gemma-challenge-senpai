#!/usr/bin/env python3
"""PR #661 -- reconstruct the run_eval.py summary JSON from an inspect .eval log.

run_eval.py only writes its summary JSON AFTER inspect_eval() returns. If the
86-min watchdog SIGINTs the bf16 leg mid-run, inspect finalizes the .eval log
(scored samples preserved) but the summary JSON is never written. This recovers
the SAME-schema JSON from whatever samples the .eval captured, so the paired gate
read still works on the n both arms completed. prompt_sha is recomputed from each
sample's (input, choices, target) with run_eval's exact hasher, so it remains a
byte-identical integrity key against the int4 arm. No model is invoked.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

# run_eval.py lives in research/validity/downstream_quality_eval/.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "downstream_quality_eval"))
from run_eval import _sample_prompt_sha  # noqa: E402

import inspect_ai.log as L  # noqa: E402
from inspect_ai.scorer import CORRECT  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="mmlu_pro")
    ap.add_argument("--arm", default="bf16_mmlu")
    ap.add_argument("--model", default="gemma-4-e4b-it")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--n-requested", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--min-tokens", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--sampling-seed", type=int, default=0)
    args = ap.parse_args()

    log = L.read_eval_log(args.eval_log, header_only=False)
    per_sample = []
    n_correct = 0
    n_scored = 0
    n_error = 0
    for s in log.samples or []:
        sid = str(s.id)
        err = None
        if s.error is not None:
            err = getattr(s.error, "message", None) or str(s.error)
            n_error += 1
        score = (s.scores or {}).get("choice")
        val = getattr(score, "value", None) if score is not None else None
        answer = getattr(score, "answer", None) if score is not None else None
        correct = val == CORRECT
        if score is not None and val in (CORRECT, "I"):
            n_scored += 1
            if correct:
                n_correct += 1
        comp = ""
        stop_reason = None
        completion_tokens = None
        try:
            out_obj = getattr(s, "output", None)
            if out_obj is not None:
                comp = out_obj.completion or ""
                choices = getattr(out_obj, "choices", None)
                if choices:
                    stop_reason = getattr(choices[0], "stop_reason", None)
                usage = getattr(out_obj, "usage", None)
                if usage is not None:
                    completion_tokens = getattr(usage, "output_tokens", None)
        except Exception:
            comp = comp or ""
        is_empty = bool(err is None and not comp.strip())
        is_length_trunc = bool(
            err is None and stop_reason in ("max_tokens", "model_length"))
        output_tokens = completion_tokens
        is_truncated = bool(stop_reason in ("max_tokens", "model_length"))
        tgt = s.target if isinstance(s.target, str) else json.dumps(s.target)
        per_sample.append({
            "id": sid, "target": tgt, "answer": answer, "value": val,
            "correct": bool(correct), "error": err, "empty": is_empty,
            "completion_chars": len(comp), "stop_reason": stop_reason,
            "completion_tokens": completion_tokens,
            "length_truncated": is_length_trunc, "output_tokens": output_tokens,
            "truncated": is_truncated, "prompt_sha": _sample_prompt_sha(s),
        })

    accuracy = (n_correct / n_scored) if n_scored else float("nan")
    n_empty = sum(1 for r in per_sample if r["empty"])
    empty_rate = (n_empty / len(per_sample)) if per_sample else float("nan")
    n_len_trunc = sum(1 for r in per_sample if r["length_truncated"])
    n_stop_max_tokens = sum(1 for r in per_sample if r["stop_reason"] == "max_tokens")
    n_stop_model_length = sum(1 for r in per_sample if r["stop_reason"] == "model_length")
    length_stop_rate = (n_len_trunc / len(per_sample)) if per_sample else float("nan")
    _ctoks = sorted(r["completion_tokens"] for r in per_sample
                    if isinstance(r["completion_tokens"], (int, float)))
    if _ctoks:
        _n = len(_ctoks)
        ctok_mean = sum(_ctoks) / _n
        ctok_p50 = _ctoks[_n // 2]
        ctok_p95 = _ctoks[min(int(0.95 * _n), _n - 1)]
        ctok_max = _ctoks[-1]
    else:
        ctok_mean = ctok_p50 = ctok_p95 = ctok_max = None
    stop_reason_counts = dict(Counter(r["stop_reason"] for r in per_sample))
    n_length = sum(1 for r in per_sample if r["truncated"])
    finish_length_rate = (n_length / len(per_sample)) if per_sample else float("nan")

    def _len_rate_at(cap: int):
        have = [r for r in per_sample if r["output_tokens"] is not None]
        if not have:
            return None, None
        n = sum(1 for r in have if r["output_tokens"] > cap)
        return n, n / len(have)

    n_length_at_2048, finish_length_rate_at_2048 = _len_rate_at(2048)

    out = {
        "task": args.task, "arm": args.arm, "model": args.model,
        "seed": args.seed, "n_requested": args.n_requested, "limit": None,
        "n_dataset": len(per_sample), "n_samples": len(per_sample),
        "n_scored": n_scored, "n_correct": n_correct, "n_error": n_error,
        "n_empty": n_empty, "empty_rate": empty_rate,
        "n_length_truncated": n_len_trunc, "n_stop_max_tokens": n_stop_max_tokens,
        "n_stop_model_length": n_stop_model_length, "length_stop_rate": length_stop_rate,
        "stop_reason_counts": stop_reason_counts,
        "completion_tokens_mean": ctok_mean, "completion_tokens_p50": ctok_p50,
        "completion_tokens_p95": ctok_p95, "completion_tokens_max": ctok_max,
        "n_length": n_length, "finish_length_rate": finish_length_rate,
        "n_length_at_2048": n_length_at_2048,
        "finish_length_rate_at_2048": finish_length_rate_at_2048,
        "accuracy": accuracy, "max_tokens": args.max_tokens,
        "min_tokens": args.min_tokens or None, "temperature": args.temperature,
        "top_p": args.top_p, "top_k": None, "sampling_seed": args.sampling_seed,
        "decode": ("greedy" if args.temperature == 0.0 else "sampling"),
        "base_url": None, "eval_log": getattr(log, "location", args.eval_log),
        "recovered_from_eval_log": True, "eval_log_status": log.status,
        "per_sample": sorted(per_sample, key=lambda r: r["id"]),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[recover] status={log.status} n_samples={len(per_sample)} "
          f"n_scored={n_scored} n_correct={n_correct} acc={accuracy:.4f} -> {args.out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
