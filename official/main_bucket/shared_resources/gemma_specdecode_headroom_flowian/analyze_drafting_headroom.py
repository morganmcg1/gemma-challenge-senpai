#!/usr/bin/env python3
"""
analyze_drafting_headroom.py — estimate speculative-drafting headroom OFFLINE,
before spending an a10g-small benchmark run.

It replays drafting policies against a harness `decode_outputs.jsonl` (the per-prompt
token-ID capture the speed benchmark already writes). Because the capture holds the exact
greedy token IDs, we can measure precisely how many *target forward passes* a given
drafter would need — i.e. tokens/forward, which is the single quantity that sets decode
TPS once the per-forward cost is fixed.

Policies:
  * AR        : autoregressive baseline (1 token / forward).
  * PLD(n,D)  : prompt-lookup / suffix drafting. At each step, match the last n tokens
                to the most recent earlier occurrence in (prompt + text-so-far), propose
                up to D following tokens, accept the greedy-correct prefix. Model-free.
  * HYBRID    : per step advance = max(PLD run, an MTP/EAGLE chain capped at K).
                MTP per-step advance isn't in the capture, so we model it with a fixed
                cap K (its long-run ceiling); use --mtp-tokens-per-forward for a realistic
                blended estimate. The HYBRID number is an UPPER BOUND on what stacking PLD
                onto an MTP chain can buy, and isolates the runs PLD captures that an
                MTP-K chain structurally cannot (length > K).

Usage:
  python analyze_drafting_headroom.py decode_outputs.jsonl --n 3 --depth 24 --mtp-cap 8
"""
from __future__ import annotations
import argparse, json, statistics
from collections import defaultdict


def pld_replay(seq, gen_start, n, depth):
    """Causal prompt-lookup replay. Returns (forwards, covered, run_lengths)."""
    end = len(seq)
    idx = {}              # ngram tuple -> most recent end-index e (predicts seq[e])
    added = n - 1
    forwards = covered = 0
    run_lengths = []
    pos = gen_start
    while pos < end:
        for e in range(max(n, added + 1), pos):   # index all causal ngrams < pos
            idx[tuple(seq[e - n:e])] = e
        added = max(added, pos - 1)
        forwards += 1
        adv = 1
        if pos >= n:
            e = idx.get(tuple(seq[pos - n:pos]), 0)
            if e and e < pos:
                a = 0
                while a < depth and e + a < pos and pos + a < end and seq[e + a] == seq[pos + a]:
                    a += 1
                if a > 0:
                    run_lengths.append(a)
                    covered += a
                    adv = a + 1 if pos + a < end else a
        pos += adv
    return forwards, covered, run_lengths


def pld_beyond_cap(run_lengths, gen_tokens, mtp_cap):
    """Honest 'what PLD adds on top of an MTP-K chain' proxy.

    An MTP chain of cap K (=mtp_cap) can advance at most mtp_cap tokens in one forward.
    PLD runs LONGER than mtp_cap are tokens an MTP-K chain structurally cannot get in that
    forward -- the clearest place a PLD/suffix hybrid helps. This is a screening proxy, not
    a TPS prediction; for the net effect, measure on hardware (and mind the async->sync tax,
    which on this stack eats the gain -- see README)."""
    over = [x - mtp_cap for x in run_lengths if x > mtp_cap]
    return {
        "mtp_cap": mtp_cap,
        "n_runs_over_cap": len(over),
        "extra_tokens_over_cap": sum(over),
        "frac_gen_tokens_over_cap": round(sum(over) / gen_tokens, 4),
        "note": "tokens only a suffix/PLD draft (not an MTP-K chain) can grab in one forward; "
                "screening proxy only -- confirm on hardware, async->sync tax may erase it",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", help="decode_outputs.jsonl from a benchmark run")
    ap.add_argument("--n", type=int, default=3, help="prompt-lookup match window")
    ap.add_argument("--depth", type=int, default=24, help="max draft length")
    ap.add_argument("--mtp-cap", type=int, default=8, help="MTP/EAGLE chain cap (K+1)")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.capture)]
    gen = sum(len(r["completion_token_ids"]) for r in recs)
    f_pld = cov = 0
    runs = []
    for r in recs:
        seq = r["prompt_token_ids"] + r["completion_token_ids"]
        gs = len(r["prompt_token_ids"])
        a, b, rl = pld_replay(seq, gs, args.n, args.depth)
        f_pld += a; cov += b; runs += rl
    rl = sorted(runs)
    out = {
        "records": len(recs), "gen_tokens": gen,
        "AR_tokens_per_forward": 1.0,
        "PLD": {
            "n": args.n, "depth": args.depth,
            "tokens_per_forward": round(gen / f_pld, 3),
            "frac_tokens_from_lookup": round(cov / gen, 3),
            "n_runs": len(rl),
            "run_len_mean": round(statistics.mean(rl), 2) if rl else 0,
            "run_len_p50": rl[len(rl) // 2] if rl else 0,
            "run_len_p95": rl[int(len(rl) * 0.95)] if rl else 0,
            "run_len_max": max(rl) if rl else 0,
        },
        "pld_beyond_mtp_cap": pld_beyond_cap(rl, gen, args.mtp_cap),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
