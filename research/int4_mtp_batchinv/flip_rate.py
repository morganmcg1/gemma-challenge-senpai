#!/usr/bin/env python3
"""Turn the greedy-identity verifier's --json report into a per-token flip rate.

The verifier reports a strict bit-exact verdict (GREEDY_IDENTICAL/DIVERGENT) and,
per prompt, the index of the FIRST divergence. In free-running greedy decode a
single argmax flip cascades (every downstream token then differs), so
`total_divergent_tokens / total_tokens` massively over-counts the underlying
event rate. The physically meaningful, cross-arm-comparable quantity is the
per-token probability `p` that the spec-decode batched-verify argmax differs from
the M=1 AR argmax at a position whose prefix is still identical.

Under an i.i.d.-per-token model the first-divergence index is geometric(p), with
identical prompts right-censored at their compared length. The censored-geometric
MLE is:

    p_hat = (# first-flip events) / (# trials up to and including each first flip,
             plus the full compared length of every identical prompt)

This script consumes the verifier ComparisonReport JSON (stdin or a path) and
prints both a human summary and a one-line JSON for logging/aggregation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report", nargs="?", default="-",
                    help="verifier --json report path ('-' = stdin)")
    ap.add_argument("--arm", default="", help="arm label for the JSON line")
    args = ap.parse_args()

    raw = sys.stdin.read() if args.report == "-" else open(args.report).read()
    rep = json.loads(raw)

    per = rep.get("per_prompt", [])
    verdict = rep.get("verdict")
    n_prompts = rep.get("num_prompts_compared", len(per))
    n_identical = rep.get("num_identical", 0)
    n_divergent = rep.get("num_divergent", 0)
    tot_tokens = rep.get("total_tokens_compared", 0)
    tot_div_tokens = rep.get("total_divergent_tokens", 0)

    # Censored-geometric MLE over per-prompt first divergence.
    flip_events = 0          # prompts with an in-range argmax flip
    trials = 0               # geometric trials (denominator of the MLE)
    length_only = 0          # divergences that are pure length/EOS, not a flip
    first_idxs: list[int] = []
    for pc in per:
        num_compared = pc.get("num_compared", 0)
        fdi = pc.get("first_divergence_index")
        identical = pc.get("identical", False)
        if (not identical) and fdi is not None and fdi < num_compared:
            # An argmax flip at position fdi: fdi matched trials + 1 flip.
            flip_events += 1
            trials += fdi + 1
            first_idxs.append(fdi)
        else:
            # Identical (censored at num_compared) or divergence only by length.
            trials += num_compared
            if not identical:
                length_only += 1

    p_hat = (flip_events / trials) if trials else float("nan")
    # Poisson 95% CI on the flip count -> CI on p_hat.
    if flip_events > 0 and trials:
        lo = flip_events - 1.96 * math.sqrt(flip_events)
        hi = flip_events + 1.96 * math.sqrt(flip_events)
        p_lo, p_hi = max(lo, 0.0) / trials, hi / trials
    else:
        # 0 events: one-sided 95% upper bound ~3.0 events (rule of three).
        p_lo, p_hi = 0.0, (3.0 / trials if trials else float("nan"))

    mean_fdi = (sum(first_idxs) / len(first_idxs)) if first_idxs else None
    first_idxs.sort()
    median_fdi = first_idxs[len(first_idxs) // 2] if first_idxs else None
    raw_cascade = (tot_div_tokens / tot_tokens) if tot_tokens else float("nan")

    out = {
        "arm": args.arm,
        "verdict": verdict,
        "prompts": n_prompts,
        "identical": n_identical,
        "divergent": n_divergent,
        "length_only_divergences": length_only,
        "flip_events": flip_events,
        "geom_trials": trials,
        "flip_rate_per_token": p_hat,
        "flip_rate_ci95": [p_lo, p_hi],
        "mean_first_divergence_index": mean_fdi,
        "median_first_divergence_index": median_fdi,
        "raw_cascade_divergent_fraction": raw_cascade,
        "total_tokens_compared": tot_tokens,
    }

    pct = (p_hat * 100) if p_hat == p_hat else float("nan")  # NaN-safe
    print(f"ARM {args.arm or '?'}: {verdict}  "
          f"identical={n_identical}/{n_prompts}  "
          f"flip_rate={pct:.4f}%/tok "
          f"(events={flip_events}, trials={trials}, "
          f"95%CI=[{p_lo*100:.4f},{p_hi*100:.4f}]%)  "
          f"mean_first_div={mean_fdi}  "
          f"cascade_frac={raw_cascade:.3f}")
    print("FLIPRATE_JSON " + json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
