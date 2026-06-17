#!/usr/bin/env python3
"""PR #615 -- separate the truncation confound from genuine reasoning quality.

For each inspect .eval log, join per-sample stop_reason with the choice score and
report:
  * raw accuracy      = correct / scored            (truncation-confounded)
  * terminated subset = stop_reason == 'stop'       (clean: model emitted EOS)
      n, accuracy among terminated
  * truncated subset  = stop_reason in TRUNC        (hit the cap)
      n, accuracy among truncated
This decides whether dev307's low GPQA is a generation pathology (it doesn't
terminate, so the answer is cut off) or a real quality collapse (it terminates
and is still wrong).

Usage: conditional_acc.py <log_dir_or_eval> [more...]   (inspect client venv)
"""
from __future__ import annotations
import glob
import json
import os
import sys

from inspect_ai.log import read_eval_log

TRUNC = {"max_tokens", "length", "model_length"}
CORRECT = "C"


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
    n = n_correct = 0
    term_n = term_correct = 0
    trunc_n = trunc_correct = 0
    for s in log.samples or []:
        sr = None
        if s.output and s.output.choices:
            sr = s.output.choices[0].stop_reason
        sc = (s.scores or {}).get("choice")
        val = getattr(sc, "value", None) if sc is not None else None
        correct = int(val == CORRECT)
        n += 1
        n_correct += correct
        if sr in TRUNC:
            trunc_n += 1
            trunc_correct += correct
        elif sr == "stop":
            term_n += 1
            term_correct += correct
    return {
        "eval": ev.split("/")[-1],
        "n": n,
        "raw_acc": (n_correct / n) if n else None,
        "term_n": term_n,
        "term_acc": (term_correct / term_n) if term_n else None,
        "trunc_n": trunc_n,
        "trunc_acc": (trunc_correct / trunc_n) if trunc_n else None,
        "trunc_rate": (trunc_n / n) if n else None,
    }


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    rows = [stats_for(p) for p in sys.argv[1:]]
    print(json.dumps(rows, indent=2))
    # compact pooled summary by label prefix (dir basename up to _s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
