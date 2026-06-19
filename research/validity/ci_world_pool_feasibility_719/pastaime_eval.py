"""PR #719 -- past-AIME base-greedy comparability driver (CI-world pool feasibility).

Prices the OBJECTIVE half of admissibility for the hypothetical >=1040-problem
CI-certification pool: base-bf16 greedy accuracy on past-AIME year-bands, measured
with the IDENTICAL harness used for the canonical reference set so the per-band
base rate is apples-to-apples comparable to the gate's ~0.4667 anchor.

Reuses `research/downstream_quality_aime/aime_eval.py` verbatim for prompting
(`build_messages`), extraction (`extract_answer`, boxed-integer + 0-999 fallback),
and scoring/dispatch (`eval_endpoint`). The ONLY new thing here is the problem
source: the public `di-zhang-fdu/AIME_1983_2024` corpus (933 distinct problems,
all integer-answer 0-999), seeded-sampled into year-bands. Greedy = k=1, T=0.

NO model is served here; this drives an already-running bf16 base `--base-url`
(stand it up with the sibling serve recipe). Analysis-only, no HF job.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

# Import the canonical AIME harness (same prompt/extractor/scorer as the anchor).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "downstream_quality_aime"))
import aime_eval  # noqa: E402


def load_corpus() -> list[dict[str, Any]]:
    """Load di-zhang-fdu/AIME_1983_2024 -> list of {id, year, problem, answer}."""
    from datasets import load_dataset

    ds = load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")
    out: list[dict[str, Any]] = []
    for r in ds:
        ans = aime_eval._to_int(r["Answer"])
        if ans is None or not (0 <= ans <= 999):
            continue  # gate-faithful: AIME integer answers only
        out.append(
            {
                "id": str(r["ID"]),
                "year": int(r["Year"]),
                "problem": str(r["Question"]),
                "answer": ans,
            }
        )
    return out


def band_sample(
    corpus: list[dict[str, Any]],
    lo: int,
    hi: int,
    n: int,
    seed: int,
    exclude_years: set[int],
) -> list[dict[str, Any]]:
    """Seeded uniform sample of <=n problems with lo<=year<=hi (excl. exclude_years)."""
    pool = [p for p in corpus if lo <= p["year"] <= hi and p["year"] not in exclude_years]
    rng = random.Random(seed * 100003 + lo)
    rng.shuffle(pool)
    return sorted(pool[:n], key=lambda p: p["id"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True, help="already-running bf16 base endpoint")
    ap.add_argument("--model", default="gemma-4-e4b-it")
    ap.add_argument(
        "--bands",
        default="1983-1994,1995-2004,2005-2014,2015-2023",
        help="comma list of lo-hi year bands",
    )
    ap.add_argument("--per-band", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--exclude-years", default="2024", help="comma years to dedup vs canonical ref")
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--min-tokens", type=int, default=8)
    ap.add_argument("--request-timeout-s", type=int, default=1200)
    ap.add_argument("--client-concurrency", type=int, default=16)
    ap.add_argument("--determinism-check", type=int, default=0,
                    help="if >0, re-run the first N problems of band 1 a 2nd time to verify greedy determinism")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    exclude = {int(y) for y in args.exclude_years.split(",") if y.strip()}
    corpus = load_corpus()
    print(f"[pastaime] corpus loaded: {len(corpus)} integer-answer problems", flush=True)

    bands: list[tuple[int, int]] = []
    for b in args.bands.split(","):
        lo, hi = b.split("-")
        bands.append((int(lo), int(hi)))

    band_results: list[dict[str, Any]] = []
    for lo, hi in bands:
        probs = band_sample(corpus, lo, hi, args.per_band, args.seed, exclude)
        print(f"[pastaime] band {lo}-{hi}: sampled {len(probs)} problems", flush=True)
        res = aime_eval.eval_endpoint(
            args.base_url,
            args.model,
            probs,
            k=1,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=args.max_tokens,
            seed=args.seed,
            enable_thinking=False,
            request_timeout_s=args.request_timeout_s,
            save_text=False,
            min_tokens=args.min_tokens,
            client_concurrency=args.client_concurrency,
        )
        acc = res["maj_k_accuracy"]
        n = res["n_problems"]
        # Wilson 95% CI
        import math
        z = 1.96
        if n:
            denom = 1 + z * z / n
            center = (acc + z * z / (2 * n)) / denom
            half = (z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n))) / denom
            lo_ci, hi_ci = center - half, center + half
        else:
            lo_ci = hi_ci = 0.0
        band_results.append(
            {
                "band": f"{lo}-{hi}",
                "lo": lo,
                "hi": hi,
                "n": n,
                "acc": acc,
                "n_correct": res["n_correct_maj"],
                "wilson_lo": lo_ci,
                "wilson_hi": hi_ci,
                "extract_fail_rate": res["extract_fail_rate"],
                "wall_s": res["wall_s"],
                "per_problem": [
                    {k: pp[k] for k in ("id", "year", "gold", "maj_answer", "maj_correct", "finish_reasons")}
                    for pp in res["per_problem"]
                ],
            }
        )
        print(
            f"[pastaime] band {lo}-{hi}: acc={acc:.4f} ({res['n_correct_maj']}/{n}) "
            f"wilson=[{lo_ci:.4f},{hi_ci:.4f}] extract_fail={res['extract_fail_rate']:.3f} "
            f"wall={res['wall_s']:.0f}s",
            flush=True,
        )

    determinism: dict[str, Any] | None = None
    if args.determinism_check > 0:
        lo, hi = bands[0]
        probs = band_sample(corpus, lo, hi, args.per_band, args.seed, exclude)[: args.determinism_check]
        r1 = aime_eval.eval_endpoint(
            args.base_url, args.model, probs, k=1, temperature=0.0, top_p=1.0, top_k=-1,
            max_tokens=args.max_tokens, seed=args.seed, enable_thinking=False,
            request_timeout_s=args.request_timeout_s, save_text=False, min_tokens=args.min_tokens,
            client_concurrency=1,
        )
        r2 = aime_eval.eval_endpoint(
            args.base_url, args.model, probs, k=1, temperature=0.0, top_p=1.0, top_k=-1,
            max_tokens=args.max_tokens, seed=args.seed, enable_thinking=False,
            request_timeout_s=args.request_timeout_s, save_text=False, min_tokens=args.min_tokens,
            client_concurrency=1,
        )
        agree = sum(
            1 for a, b in zip(r1["per_problem"], r2["per_problem"]) if a["maj_answer"] == b["maj_answer"]
        )
        determinism = {
            "n": len(probs),
            "answer_agree": agree,
            "agree_frac": agree / len(probs) if probs else 0.0,
        }
        print(f"[pastaime] determinism: {agree}/{len(probs)} identical answers across repeats", flush=True)

    out = {
        "source": "di-zhang-fdu/AIME_1983_2024",
        "corpus_distinct": len(corpus),
        "exclude_years": sorted(exclude),
        "per_band": args.per_band,
        "seed": args.seed,
        "sampling": {"regime": "greedy", "k": 1, "temperature": 0.0, "max_tokens": args.max_tokens,
                     "min_tokens": args.min_tokens, "enable_thinking": False},
        "created_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "bands": band_results,
        "determinism": determinism,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[pastaime] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
