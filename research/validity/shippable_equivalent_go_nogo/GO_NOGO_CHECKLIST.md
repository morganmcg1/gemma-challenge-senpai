# SHIP THE FASTEST STRICTLY-EQUIVALENT CONFIG -- HUMAN GO/NO-GO CHECKLIST

_PR #419 (lawine) static-analysis handoff -- generated 20260615T223511Z. This card SHIPS NOTHING; it is
the one page a human reads to authorize {served-file change + leaderboard submission}._

## The decision (executable predicate)

```
SHIP iff (measured_fastest_equivalent_tps > 481.53)   # kanna #416, beats the deployed non-strict #1
         AND byte_identity_verified                    # #319 e2e gate returns identity 1.0
         AND (ppl <= 2.42)                             # quality guardrail (unchanged 2.3772)
         AND (completed == 128)                        # full public run
```

- Ship-breakeven (NO-GO boundary): **481.53 TPS** (the deployed non-strict #1, PR #52 `2x9fm2zx`).
- Modeled combined TPS bracket: **[492.08, 494.08]** -> modeled margin **+10.55 .. +12.55** (central +11.55).
- Robustness: even a ZERO-speedup selective-recompute still ships (**483.08 TPS**, +1.55) -- the predicate is robust, not knife-edge.

## Every line that MUST be GREEN before approval

| # | Gate | Threshold | Source |
|---|------|-----------|--------|
| 1 | Measured combined TPS beats breakeven | `> 481.53` | kanna #416 `fastest_equivalent_tps` |
| 2 | Served greedy identity | `== 1.0` (flips 3/882 -> 0) | #319 e2e self-referential gate |
| 3 | PPL | `<= 2.42` (expect 2.3772) | PPL stage |
| 4 | Completed | `== 128` | benchmark |
| 5 | Whole-stack reversible | re-submit prior package; cb3 by bucket flip; selrec by `SELECTIVE_RECOMPUTE_VERIFY=0` | #417 + this card |
| 6 | Flag-revert confirmed | OFF == today's served, ON == strict, residual 0.0 TPS | this card Part 3 |

## Pre-submission identity-verify CI (the byte-identity EVIDENCE)

- Budget: **41.8 GPU-min** (tier3 e2e shared 35.8 + tier2 decode-width shared 4.0 + 2x tier-1 micro (cb3 new-ref + selrec byte-exact) = 41.8 (vs naive unshared 81.6).)
- TIER1 per-GEMM/-config byte-identity micro (#390); TIER2 decode-width e2e (#381); TIER3 e2e self-referential gate (#319 gen_greedy_reference --mode served + greedy_gate.compare + greedy_identity_interlock).
- This is measured LOCALLY on the A10G BEFORE any served-file change -- it proves byte-identity, it is not a TPS claim.

## Feature-flag de-risking the one binding in-place line

- `SELECTIVE_RECOMPUTE_VERIFY` -- resolved ONCE at serve startup (read env -> bind verify-reduction function pointer); NO per-step hot-path branch.
  - **ON (=1, shipped default):** selective-recompute reduction (fast attention everywhere + eps near-tie gate + higher-prec reduction on the ~23.6% flagged near-tie steps) -> byte-identical to BLANKET-STRICT (the strict reference) -> served flips 0, identity 1.0.
  - **OFF (=0, rollback):** today's-served reduction (the deployed fast verify/attention path, unchanged) -> byte-identical to TODAY'S SERVED verify (the deployed fast path) -> the deployed 481.53 behavior.
  - Rollback-while-keeping-cb3: flip SELECTIVE_RECOMPUTE_VERIFY=0 in the manifest env -> selective-recompute bypassed, cb3's additive checkpoint+kernel UNTOUCHED. A flag flip, NOT a code re-edit (this is #417's binding line de-risked).

## Safe operation order

1. **Measure locally on the A10G** -- run the 41.8 GPU-min pre-submission verify CI; confirm ALL six GREEN lines above (especially identity `== 1.0` and the measured TPS `> 481.53`).
2. **Human approves in GitHub** -- the gated approval issue (PR + branch + exact command + the GREEN CI evidence).
3. **Flip served file + submit** -- deploy the equivalent submission (flag ON = strict) and submit to the leaderboard. THIS step is the human-gated action; everything above is pre-flight.

_Rollback at any point = re-submit the prior package, or flip `SELECTIVE_RECOMPUTE_VERIFY=0` to keep cb3 while dropping selective-recompute._
