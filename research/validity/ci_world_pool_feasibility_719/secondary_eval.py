"""PR #719 -- SECONDARY-source base-greedy comparability driver (CI-world pool feasibility).

Past-AIME is the only same-format primary candidate, and its full historical supply
is structurally < 1040 (see pastaime_eval / corpus count). So whether the CI world is
constructible turns on whether a SECONDARY source can add base-rate-comparable,
gate-faithful, distinct problems. The two candidates the card names:

  * AMC (AI-MO/aimo-validation-amc, 83 integer-normalized problems) -- the SWING
    factor: it IS integer-gradeable (answers normalize to ints 0-999) and the full
    AMC corpus is large, so its admissibility is decided purely by DIFFICULTY
    comparability. If base greedy >> the canonical AIME anchor, AMC is "too easy"
    and falls outside the comparability band -> inadmissible.
  * MATH level-5 (nlile/hendrycks-MATH-benchmark, level==5) -- ~51% of answers are
    NON-integer (fractions / complex / tuples / expressions), so the gate's
    boxed-integer-0-999 grader is NOT faithful for half the set; we still measure
    base greedy on the gradeable integer subset for a complete per-source row, but
    the source is excluded on grader-faithfulness regardless of rate.

Reuses aime_eval verbatim (build_messages / extract_answer / eval_endpoint), so the
per-source base rate is apples-to-apples with the canonical anchor. Greedy = k=1, T=0.
Drives an already-running bf16 base --base-url. Analysis-only, no HF job.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "downstream_quality_aime"))
import aime_eval  # noqa: E402


def _clean_int_0_999(v: Any) -> int | None:
    """Gate-faithful gold: the answer must be a bare integer in 0..999.

    AMC answers come as floats ('142.0'); strip a trailing '.0' then require the
    whole token to be an integer. Anything else (fraction, expression, >999) is
    NOT gate-gradeable and is dropped from the gradeable subset.
    """
    s = str(v).strip().replace(",", "")
    s = re.sub(r"\.0+$", "", s)  # 142.0 -> 142
    if not re.fullmatch(r"-?\d+", s):
        return None
    n = int(s)
    return n if 0 <= n <= 999 else None


def load_amc() -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("AI-MO/aimo-validation-amc", split="train")
    out = []
    for i, r in enumerate(ds):
        a = _clean_int_0_999(r["answer"])
        if a is None:
            continue
        out.append({"id": f"amc-{i:03d}", "year": "amc", "problem": str(r["problem"]), "answer": a})
    return out


def load_math_l5() -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("nlile/hendrycks-MATH-benchmark", split="test")
    out = []
    for i, r in enumerate(ds):
        if str(r["level"]) != "5":
            continue
        a = _clean_int_0_999(r["answer"])
        if a is None:
            continue  # gradeable integer subset only
        out.append({"id": f"math5-{i:03d}", "year": "math-l5", "problem": str(r["problem"]), "answer": a})
    return out


def wilson(acc: float, n: int) -> tuple[float, float]:
    if not n:
        return 0.0, 0.0
    z = 1.96
    denom = 1 + z * z / n
    center = (acc + z * z / (2 * n)) / denom
    half = (z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n))) / denom
    return center - half, center + half


SOURCES = {"amc": load_amc, "math_l5": load_math_l5}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="gemma-4-e4b-it")
    ap.add_argument("--sources", default="amc,math_l5")
    ap.add_argument("--limit", type=int, default=0, help="cap problems per source (0=all gradeable)")
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--min-tokens", type=int, default=8)
    ap.add_argument("--client-concurrency", type=int, default=16)
    ap.add_argument("--request-timeout-s", type=int, default=1200)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    rows = []
    for name in [s.strip() for s in args.sources.split(",") if s.strip()]:
        probs = SOURCES[name]()
        if args.limit:
            probs = probs[: args.limit]
        print(f"[secondary] {name}: {len(probs)} gradeable-integer problems", flush=True)
        res = aime_eval.eval_endpoint(
            args.base_url, args.model, probs, k=1, temperature=0.0, top_p=1.0, top_k=-1,
            max_tokens=args.max_tokens, seed=1234, enable_thinking=False,
            request_timeout_s=args.request_timeout_s, save_text=False,
            min_tokens=args.min_tokens, client_concurrency=args.client_concurrency,
        )
        acc, n = res["maj_k_accuracy"], res["n_problems"]
        lo, hi = wilson(acc, n)
        rows.append({
            "source": name, "n_graded": n, "acc": acc, "n_correct": res["n_correct_maj"],
            "wilson_lo": lo, "wilson_hi": hi, "extract_fail_rate": res["extract_fail_rate"],
            "wall_s": res["wall_s"],
            "per_problem": [{k: pp[k] for k in ("id", "gold", "maj_answer", "maj_correct", "finish_reasons")}
                            for pp in res["per_problem"]],
        })
        print(f"[secondary] {name}: acc={acc:.4f} ({res['n_correct_maj']}/{n}) "
              f"wilson=[{lo:.4f},{hi:.4f}] extract_fail={res['extract_fail_rate']:.3f} "
              f"wall={res['wall_s']:.0f}s", flush=True)

    out = {
        "created_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "sampling": {"regime": "greedy", "k": 1, "temperature": 0.0,
                     "max_tokens": args.max_tokens, "min_tokens": args.min_tokens},
        "sources": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[secondary] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
