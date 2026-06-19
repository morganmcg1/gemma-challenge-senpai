#!/usr/bin/env python
"""Engine-faithfulness gate for PR #720 (advisor 09:19Z directive).

kanna #699 proved the substitute ``.venvs/vllm022`` corrupts greedy decode even
at cc=1/BI=1 (base AIME -> 0.1333, int4 -> repetition-to-cap), and warned that on
a corrupting engine the served-fast and own-AR paths can derail *differently*,
faking a #319 self-consistency break that is really an engine artifact. So before
trusting any ``RECOVERY_319_*`` break, prove neither leg is in the corruption
regime.

The corruption signature is concrete and measurable on the completion token ids
the profiler already captures: **repetition-to-cap** (every sequence runs to
``max_tokens`` emitting a short repeating cycle), which shows up as a collapsed
distinct-token ratio and a long maximal repeat run. A faithful engine produces
coherent prose: high token diversity, short repeat runs, and a spread of
completion lengths (many stop on EOS before the cap).

This is the per-leg, automatic half of the gate (every leg in an out-dir). The
quantitative arm (greedy-AIME int4 anchor -> ~0.350 coherent on this same engine)
is a separate run; the two together discharge the advisor's reconciliation gate.

Verdict per leg: FAITHFUL / SUSPECT_CORRUPT. The run is FAITHFUL only if every
leg is FAITHFUL; otherwise the break verdict is not trustworthy on this engine.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _longest_periodic_run(ids: list[int], max_period: int = 8) -> int:
    """Longest tail run explained by a repeating cycle of period <= max_period.

    Repetition-to-cap emits e.g. 'A B A B A B ...' (period 2) or a single token
    (period 1) for hundreds of tokens. We scan small periods and return the
    longest contiguous run anywhere where ids[i] == ids[i-p]. A coherent
    completion has only short such runs (normal English bigram echoes); a
    collapsed one has a run that reaches near the whole length.
    """
    n = len(ids)
    best = 0
    for p in range(1, max_period + 1):
        run = 0
        for i in range(p, n):
            if ids[i] == ids[i - p]:
                run += 1
                if run > best:
                    best = run
            else:
                run = 0
    return best


def _distinct_ratio(ids: list[int]) -> float:
    return (len(set(ids)) / len(ids)) if ids else 1.0


def analyze_leg(jsonl: Path, output_len: int) -> dict:
    recs = [json.loads(l) for l in jsonl.open()]
    lengths, dratios, repeats = [], [], []
    n_degenerate = 0
    n_at_cap = 0
    for r in recs:
        ids = r.get("completion_token_ids") or []
        ntok = r.get("num_completion_tokens") or len(ids)
        lengths.append(ntok)
        dr = _distinct_ratio(ids)
        rr = _longest_periodic_run(ids)
        dratios.append(dr)
        repeats.append(rr)
        if ntok >= output_len:
            n_at_cap += 1
        # A single sequence is degenerate if it runs long AND is dominated by a
        # short repeating cycle (low diversity OR a repeat run covering most of it).
        if ntok >= 64 and (dr < 0.15 or rr > max(50, 0.5 * ntok)):
            n_degenerate += 1
    n = len(recs)
    degen_frac = n_degenerate / n if n else 0.0
    cap_frac = n_at_cap / n if n else 0.0
    mean_dr = statistics.mean(dratios) if dratios else 1.0
    # SUSPECT if a meaningful fraction of completions look collapsed, or if every
    # completion is pinned at the cap with low mean diversity (classic repeat-to-cap).
    suspect = (degen_frac > 0.10) or (cap_frac > 0.95 and mean_dr < 0.25)
    return {
        "leg": jsonl.stem,
        "n": n,
        "mean_completion_len": round(statistics.mean(lengths), 1) if lengths else 0,
        "median_completion_len": int(statistics.median(lengths)) if lengths else 0,
        "frac_at_cap": round(cap_frac, 4),
        "mean_distinct_ratio": round(mean_dr, 4),
        "min_distinct_ratio": round(min(dratios), 4) if dratios else 1.0,
        "max_repeat_run": max(repeats) if repeats else 0,
        "degenerate_frac": round(degen_frac, 4),
        "verdict": "SUSPECT_CORRUPT" if suspect else "FAITHFUL",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True, help="dir with <label>.<leg>.jsonl files")
    ap.add_argument("--output-len", type=int, default=512, help="decode cap (to flag at-cap sequences)")
    ap.add_argument("--out", type=Path, default=None, help="optional JSON path for the gate report")
    args = ap.parse_args()

    jsonls = sorted(p for p in args.out_dir.glob("*.jsonl"))
    legs = [analyze_leg(p, args.output_len) for p in jsonls]
    all_faithful = all(l["verdict"] == "FAITHFUL" for l in legs) if legs else False
    report = {
        "out_dir": str(args.out_dir),
        "output_len": args.output_len,
        "engine_faithful": all_faithful,
        "legs": legs,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
