#!/usr/bin/env python3
"""PR #615 -- read inspect .eval log(s) and report stop_reason histogram +
finish_reason=length (truncation) rate + completion-token length stats.

Truncation set mirrors debias_mmlu.py: {max_tokens, length, model_length}.
Usage: finish_stats.py <log_dir_or_eval> [<log_dir_or_eval> ...]
Run under the inspect client venv.
"""
from __future__ import annotations
import glob
import json
import os
import sys
from collections import Counter

from inspect_ai.log import read_eval_log

TRUNC = {"max_tokens", "length", "model_length"}


def latest_eval(path: str) -> str:
    if path.endswith(".eval"):
        return path
    cands = sorted(glob.glob(os.path.join(path, "*.eval")))
    if not cands:
        raise SystemExit(f"no .eval in {path}")
    best, best_n = cands[0], -1
    for c in cands:
        try:
            n = len(read_eval_log(c).samples or [])
        except Exception:
            n = -1
        if n > best_n:
            best, best_n = c, n
    return best


def stats_for(path: str) -> dict:
    ev = latest_eval(path)
    log = read_eval_log(ev)
    stops: Counter = Counter()
    out_tokens = []
    n = 0
    n_trunc = 0
    for s in log.samples or []:
        n += 1
        sr = None
        if s.output and s.output.choices:
            sr = s.output.choices[0].stop_reason
        stops[str(sr)] += 1
        if sr in TRUNC:
            n_trunc += 1
        try:
            ct = s.output.usage.output_tokens if (s.output and s.output.usage) else None
        except Exception:
            ct = None
        if ct is not None:
            out_tokens.append(int(ct))
    out_tokens.sort()
    def pct(p):
        if not out_tokens:
            return None
        i = min(len(out_tokens) - 1, int(round(p / 100.0 * (len(out_tokens) - 1))))
        return out_tokens[i]
    return {
        "eval": ev,
        "n_samples": n,
        "n_trunc": n_trunc,
        "finish_length_rate": (n_trunc / n) if n else None,
        "stop_reasons": dict(stops),
        "out_tokens_mean": (sum(out_tokens) / len(out_tokens)) if out_tokens else None,
        "out_tokens_p50": pct(50),
        "out_tokens_p95": pct(95),
        "out_tokens_max": out_tokens[-1] if out_tokens else None,
    }


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    results = [stats_for(p) for p in sys.argv[1:]]
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
