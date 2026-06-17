#!/usr/bin/env python3
"""Extract accuracy + Wilson 95% CI + empty/extract-fail rate from every arm's
result JSON (this PR's drop2/drop3 carves + the #538 42L/37L endpoints), so the
depth-gate Pareto and EOS-guard caveat can be read off one table.

empty/extract-fail := a scored sample whose extracted `answer` is None/blank.
That is exactly the first-token-EOS empty (wirbel #541) signature: the request
returned no usable letter, so the choice scorer marks it 'I'. n_error is the
separate hard-failure count (e.g. context overflow retried to exhaustion)."""
import json
import math
import os
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
ENDPOINTS = os.path.normpath(os.path.join(HERE, "..", "body_decomp_served_2x2"))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, center - half, center + half)


def load(path):
    with open(path) as f:
        d = json.load(f)
    samples = d.get("per_sample", [])
    n_answer_none = 0
    for s in samples:
        ans = s.get("answer")
        if ans is None or (isinstance(ans, str) and ans.strip() == ""):
            n_answer_none += 1
    n_scored = d.get("n_scored", 0)
    n_correct = d.get("n_correct", 0)
    p, lo, hi = wilson(n_correct, n_scored)
    return {
        "file": os.path.basename(path),
        "arm": d.get("arm"),
        "task": d.get("task"),
        "min_tokens": d.get("min_tokens", 0),
        "n_dataset": d.get("n_dataset"),
        "n_samples": d.get("n_samples"),
        "n_scored": n_scored,
        "n_correct": n_correct,
        "n_error": d.get("n_error"),
        "n_answer_none": n_answer_none,
        "accuracy": d.get("accuracy"),
        "wilson_lo": lo,
        "wilson_hi": hi,
        "empty_rate": (n_answer_none / d["n_samples"]) if d.get("n_samples") else float("nan"),
    }


def main():
    files = []
    # This PR's arms (drop2 [37,38], drop3 [36,37,38]); skip the superseded 3637.
    for pat in ["bf16_drop2_*.json", "bf16_drop3_*.json"]:
        files += glob.glob(os.path.join(HERE, pat))
    files = [f for f in files if "_3637_" not in f and "_meta" not in f]
    # #538 endpoints.
    for pat in ["bf16_42L_*.json", "bf16_37L_*.json"]:
        files += glob.glob(os.path.join(ENDPOINTS, pat))

    rows = []
    for f in sorted(files):
        try:
            rows.append(load(f))
        except Exception as e:
            print(f"ERR {f}: {e}")

    hdr = (f"{'arm':22s} {'task':12s} {'mt':>3s} {'nset':>4s} {'nscor':>5s} "
           f"{'ncor':>4s} {'nerr':>4s} {'empty':>5s} {'acc':>7s} {'wilson95':>17s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        acc = r["accuracy"]
        accs = f"{acc:.4f}" if acc == acc else "  nan"
        wil = f"[{r['wilson_lo']:.3f},{r['wilson_hi']:.3f}]" if acc == acc else "[ nan, nan]"
        print(f"{str(r['arm']):22s} {str(r['task']):12s} {r['min_tokens']:>3} "
              f"{str(r['n_dataset']):>4s} {r['n_scored']:>5} {r['n_correct']:>4} "
              f"{r['n_error']:>4} {r['n_answer_none']:>5} {accs:>7s} {wil:>17s}")

    out = os.path.join(HERE, "_metrics_table.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
