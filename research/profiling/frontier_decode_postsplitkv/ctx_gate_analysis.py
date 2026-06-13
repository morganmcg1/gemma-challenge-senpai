"""denken #51 context-gate cross-check on the #53 SERVED re-profile.

Question from the advisor (PR #53): the merged denken #51 cost-model curve says
the #43 split-KV redirect is NET-NEGATIVE at the deployed operating point
(linear MTP K=7 -> M=8 verify) on a short KV axis, only turning net-positive at
long context.  How much of the SERVED decode sits in that short-ctx penalty band,
and what would a `seqlen_kv >= threshold` context-gate recover?

This script does NOT re-run anything.  It takes:
  * denken's M=8 cost-model verify Delta%(ctx) curve  (results_pr51_*; reproduced
    here via compare_splitkv_curves.py at --operating-M 8), and
  * THIS run's real per-cycle seqlen_kv distribution (decode_frontier.jsonl prompt
    lengths swept by E_accept over output_len),
and computes the cost-model-PREDICTED aggregate served verify Delta%, to compare
against the MEASURED served aggregate (-17.5%, #30 re-profile + #43 A/B).

A large gap => the cost-model M=8 penalty does not transfer to the served stack,
i.e. the context-gate would gate off a measured win rather than recover a penalty.

LOCAL read-only analysis. No serving/PPL/greedy surface touched.
"""
from __future__ import annotations

import json
from pathlib import Path

RUN = Path("research/profiling/frontier_decode_postsplitkv")

# ---- denken #51 M=8 cost-model verify Delta%(ctx), split-KV ON vs OFF ----
# Reproduced from results_pr51_{baseline,splitkv}{,_longctx}.json via
# scripts/profiler/compare_splitkv_curves.py --operating-M 8 (see PR #53 comment).
# (ctx, baseline_ms, splitkv_ms, delta_pct).  M=8 is the deployed linear-stack M.
DENKEN_M8 = [
    (256, 11.693, 13.511, +15.5),
    (512, 12.434, 13.433, +8.0),
    (1024, 12.827, 13.482, +5.1),
]

# Measured SERVED aggregate (this re-profile + #43 served A/B, p50 over all steps).
MEASURED_VERIFY_DELTA_PCT = -17.5  # 7.906 -> 6.519 ms


def interp(curve, x, key_i, val_i):
    """Piecewise-linear interp on curve[key_i]->curve[val_i]; flat outside range."""
    xs = [c[key_i] for c in curve]
    if x <= xs[0]:
        return curve[0][val_i]
    if x >= xs[-1]:
        return curve[-1][val_i]
    for a, b in zip(curve, curve[1:]):
        if a[key_i] <= x <= b[key_i]:
            t = (x - a[key_i]) / (b[key_i] - a[key_i])
            return a[val_i] + t * (b[val_i] - a[val_i])
    return curve[-1][val_i]


def main() -> int:
    recs = [json.loads(l) for l in (RUN / "decode_frontier.jsonl").open()]
    a = json.loads((RUN / "frontier_decode_profile.json").read_text())["analysis"]
    e_accept = a["e_accept"]
    output_len = 512

    # Build the per-cycle seqlen_kv list across all prompts.  Each decode cycle
    # advances the KV axis by ~E_accept accepted tokens; the verify at cycle c
    # attends to seqlen_kv ~= prompt_len + E_accept*c.  Weight each cycle by the
    # cost-model baseline verify cost b(ctx) (cost-weighted aggregate Delta%).
    n_cycles = int(round(output_len / e_accept))
    ctx_samples = []
    for r in recs:
        p = r["num_prompt_tokens"]
        for c in range(n_cycles):
            ctx_samples.append(p + e_accept * c)

    N = len(ctx_samples)
    # Regime fractions (cycle-weighted).
    f_lt256 = sum(x < 256 for x in ctx_samples) / N
    f_lt512 = sum(x < 512 for x in ctx_samples) / N
    f_ge1024 = sum(x >= 1024 for x in ctx_samples) / N

    # Cost-model-predicted served aggregate verify Delta% (cost-weighted).
    num = den = 0.0
    for x in ctx_samples:
        b = interp(DENKEN_M8, x, 0, 1)       # baseline ms at ctx x
        d = interp(DENKEN_M8, x, 0, 3)       # delta% at ctx x
        num += b * d
        den += b
    pred_delta = num / den

    # If a context-gate fired split-KV ONLY where the cost model says it helps
    # (delta<0): M=8 curve is >0 at ALL measured ctx (256..1024), so within the
    # served range the gate would NEVER fire -> equivalent to split-KV OFF at M=8.
    gate_fires_frac = sum(interp(DENKEN_M8, x, 0, 3) < 0 for x in ctx_samples) / N

    print("# denken #51 context-gate cross-check (M=8 deployed linear stack)\n")
    print(f"E_accept = {e_accept:.3f} tok/cycle  ->  ~{n_cycles} verify cycles/prompt")
    print(f"per-cycle seqlen_kv samples: {N:,} (128 prompts x {n_cycles} cycles)\n")

    print("## SERVED per-cycle seqlen_kv regime (cycle-weighted)")
    print(f"  ctx < 256  : {100*f_lt256:5.1f}%  (denken M=8: +15.5% SLOWER)")
    print(f"  ctx < 512  : {100*f_lt512:5.1f}%  (denken M=8: +8.0..15.5% SLOWER)")
    print(f"  ctx >= 1024: {100*f_ge1024:5.1f}%  (denken M=8: +5.1% SLOWER)")
    print()
    print("## Cost-model-PREDICTED served aggregate verify Delta% (M=8 curve x this run's ctx dist)")
    print(f"  predicted = {pred_delta:+.1f}%  (split-KV SLOWER, cost-weighted)")
    print(f"  MEASURED  = {MEASURED_VERIFY_DELTA_PCT:+.1f}%  (#30 re-profile + #43 served A/B)")
    print(f"  GAP       = {MEASURED_VERIFY_DELTA_PCT - pred_delta:+.1f} pp "
          f"(cost-model penalty does NOT transfer to served stack)")
    print()
    print("## Context-gate (fire split-KV only where cost model says delta<0)")
    print(f"  fraction of served cycles where M=8 cost model says split-KV helps: "
          f"{100*gate_fires_frac:.1f}%")
    print("  => within the served ctx range the M=8 curve is >0 everywhere, so a")
    print("     cost-model-honest seqlen_kv gate fires ~never at M=8 == split-KV OFF.")
    print("  => but the SERVED stack measures -17.5% (a WIN) over this same dist,")
    print("     so gating OFF would FORFEIT the measured win, not recover a penalty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
