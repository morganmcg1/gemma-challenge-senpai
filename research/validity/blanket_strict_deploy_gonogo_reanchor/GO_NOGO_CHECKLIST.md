# BLANKET-STRICT DEPLOY GO/NO-GO -- HUMAN CHECKLIST (re-anchored off the refuted selrec leg)

_PR #425 (lawine) static-analysis handoff -- generated 20260615T232316Z. This card SHIPS NOTHING; it is
the decision surface a human reads to authorize {served-file change + leaderboard submission} once
BOTH pending conjuncts go GREEN. The flag a GO entails selects BLANKET-STRICT, NOT selrec._

## The decision (executable predicate -- a conjunction of TWO pending inputs)

```
GO iff (measured_margin_tps > 0)    # kanna #416: measured(blanket-strict+cb3) > 481.53   [PENDING]
       AND (identity_value == 1.0)   # stark #421: canonical tie-break closes prompt-90      [PENDING]
       AND (ppl <= 2.42)             # cb3 PPL-safe, 2.3772 unchanged
       AND (completed == 128)        # full public run
```

- Realizable strict stack: **blanket-strict 467.14 + cb3 +15.6 = 482.74 TPS** (knife-edge **+1.21** over the deployed non-strict 481.53, WITH a byte-identity guarantee the deployed config lacks).
- Ship-breakeven (NO-GO boundary): **481.53 TPS** (deployed non-strict #1, PR #52 `2x9fm2zx`).
- **CURRENT VERDICT: HOLD-for-conjunction** -- conjunct_margin_green=False (kanna #416 un-measured), conjunct_identity_green=False (blanket-strict 0.9989, 1 flip @ prompt 90).
- selrec is EXCLUDED: stark #412 refuted it (384.11 realizable / 0.9853 identity). Do NOT price or flag the deploy on selrec.

## Every line that MUST be GREEN before approval

| # | Gate | Threshold | Source | Now |
|---|------|-----------|--------|-----|
| 1 | Measured stack beats breakeven | `margin > 0` (stack `> 481.53`) | kanna #416 | PENDING (modeled +1.21) |
| 2 | Served greedy identity | `== 1.0` (1 flip @ prompt 90 -> 0) | stark #421 canonical tie-break | 0.9989 |
| 3 | PPL | `<= 2.42` (expect 2.3772) | cb3 PPL-safe | OK by construction |
| 4 | Completed | `== 128` | benchmark | OK by construction |
| 5 | Whole-stack reversible | flag `STRICT_VERIFY_REDUCTION=0` (verify) / cb3 bucket flip | #417 + this card | OK |

## The binding contingency (what kills this deploy)

**Ranked most -> least likely to fail: ['margin', 'identity'].** Binding = **MARGIN**.

> If kanna #416 measures the cb3-over-blanket-strict additivity haircut above ~7.8% (well inside ubel #410's <=14.9% bound), the strict stack lands at or below the deployed 481.53, the +1.21 evaporates, and there is then NO TPS reason to ship the strict config at all -- a byte-identity guarantee with zero speed upside is a NO-GO. The identity flip is the lesser risk: it is one canonical tie-break (stark #421) at a single true bitwise tie, a discrete fix #412 shows precision cannot do but a tie-break can.

- Margin knife-edge: the realized cb3 lift must clear **14.39 TPS** (haircut < **7.8%**); #410 admits up to **14.9%** -> the failure region is ~48% of the admissible band. Worst-case stack 480.42 (margin -1.11).
- Identity: #412 `identity_1p0_unreachable_by_precision=True`: no attention-precision lever closes the prompt-90 flip (it is a bitwise tie). #405 (merged) showed a GLOBAL lowest-id rule is RED (14 new flips, M=1 ref not uniformly lowest-id); #421 must canonicalize ONLY true ties.

## The exact deploy config a GO entails (BLANKET-STRICT, not selrec)

PIN blanket-strict verify reduction (flag STRICT_VERIFY_REDUCTION=1) + ADD cb3 supply (k*=229). Selects BLANKET-STRICT, NOT selrec. Whole stack additive + reversible (flag flip / bucket flip); human-gated.

- **Attention/verify pin:** PIN the served verify/attention reduction to BLANKET-STRICT (high-precision reduction EVERYWHERE -- the strict reference path), NOT the deployed non-strict fast reduction and NOT the refuted selrec eps-near-tie kernel. No new kernel is built: blanket-strict reuses the high-precision reduction the verify already supports, applied unconditionally.
  - flag `STRICT_VERIFY_REDUCTION` ON -> 1 (ship): blanket-strict reduction -> byte-identical to the strict reference (after #421 tie-break, served flips -> 0, identity 1.0)
  - flag `STRICT_VERIFY_REDUCTION` OFF -> 0 (rollback): today's-served non-strict reduction -> the deployed 481.53 behavior
  - selects_selrec = False
- **cb3 supply:** ADD the cb3 body-read-shrink supply lever (k*=229, +15.60 M=8, PPL-safe, equivalence-neutral) as 6 ADDITIVE files / 0 in-place edits, per the #417 ledger. Orthogonal subsystem to the attention pin -> stacks additively in ONE combined submission + ONE checkpoint.

## Safe operation order

1. **Wait for BOTH conjuncts** -- kanna #416 (measured margin > 0) AND stark #421 (identity 1.0).
2. **Measure locally on the A10G** -- run the #319 3-tier identity-verify CI; confirm identity `== 1.0` and the measured stack `> 481.53`.
3. **Human approves in GitHub** -- the gated approval issue (PR + branch + exact command + GREEN CI).
4. **Flip served file + submit** -- pin blanket-strict (flag ON) + cb3 and submit. Human-gated.

_Until BOTH conjuncts are GREEN, the honest verdict is **HOLD-for-conjunction**. Rollback at any point = re-submit the prior package, or flip `STRICT_VERIFY_REDUCTION=0` to drop the strict pin while keeping cb3._
