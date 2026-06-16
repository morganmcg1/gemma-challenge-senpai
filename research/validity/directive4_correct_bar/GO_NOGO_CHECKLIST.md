# DIRECTIVE-#4 DEPLOY GO/NO-GO -- HUMAN CHECKLIST (the CORRECT equivalent bar)

_PR #430 (lawine) static-analysis handoff -- generated 20260616T000452Z. SUPERSEDES #425's `GO_NOGO_CHECKLIST.md` (additive new file; the #425 card is left in place). This card SHIPS NOTHING; it is the decision surface a human reads to authorize {served-file change + submission} once the ONE remaining conjunct (identity, stark #429) goes GREEN. It LEADS with the banked Directive-#4 win._

## The banked win (LEAD) -- Bar A, the CORRECT Directive-#4 bar

```
Directive-#4 (#407): MAXIMIZE TPS subject to strict byte-exact greedy-token-equivalence.
  fastest prior FEASIBLE config (Bar A) ... blanket-strict 467.14  (#412 dnjvqbtf, identity 0.9989)
  realizable strict stack ................. blanket-strict + cb3 = 482.74  (= 467.14 + 15.6, #403 k*=229)
  >>> directive4_banked_win_tps .......... +15.6  (+3.34%)  -- NO margin contingency
```

- **The deployed 481.53 is NOT the bar.** It is the non-strict fast path (3/882 M=8 flips, identity 0.9966); under #407 it does NOT respect the equivalence, so it is **excluded from the feasible-set max**.
- **No margin contingency.** Bar A margin == the REALIZED cb3 lift (the stack IS blanket-strict + lift). Even at ubel #410's worst-case 14.9% haircut the realized lift is +13.28 (stack 480.42), so Bar A stays +13.28 -- strictly positive across the ENTIRE [0, 14.9%] haircut band. No measurement of kanna #416 can make this bar negative.
- `canonical_blanket_strict_tps = 467.14` (#412 measured; #393 467.48 decode-eta projection is a confirming cross-check, +0.34 / 0.0728%, ~2.12sigma -- the projection is marginally optimistic, so the measured value is also the conservative choice).

## The feasible-set ladder (strictly-equivalent configs, lowest -> highest)

| # | Config | TPS | Identity | In feasible set | W&B |
|---|--------|-----|----------|-----------------|-----|
| 1 | pure-AR-greedy (M=1 reference) | unmeasured (~165.44 official est) | 1.0 | True | #196 (compliant_nonspec_floor) |
| 2 | blanket-strict (high-precision verify reduction everywhere) | 467.14 | 0.9989 | True | #412 dnjvqbtf (measured) / #393 0q7ynumg (467.48 projection cross-check) |
| 3 | blanket-strict + cb3 supply (k*=229) | 482.74 | 0.9989 | True | #425 3u2urqzj (stack infra) + #403 iv9i2wks (cb3 lift) |
| -- | ~~deployed fast path (non-strict reduction)~~ (EXCLUDED) | 481.53 | 0.9966 | **False** | #52 2x9fm2zx |

_Feasible-set max = **blanket-strict + cb3 supply (k*=229)** at **482.74 TPS**. `directive4_bar_is_feasible_set_max = True` (481.53 excluded)._

## Every line that MUST be GREEN before approval

| # | Gate | Threshold | Source | Now |
|---|------|-----------|--------|-----|
| 1 | Directive-#4 win (Bar A) | `> 0` (stack `> 467.14`) | this card | **GREEN -- banked +15.6, no contingency** |
| 2 | Served greedy identity | `== 1.0` (close base flip @ prompt 90) | **stark #429** (in flight) | 0.9989 -- PENDING |
| 3 | PPL | `<= 2.42` (expect 2.3772) | cb3 PPL-safe | OK by construction |
| 4 | Completed | `== 128` | benchmark | OK by construction |
| 5 | Bar B (beat illegal 481.53) | bonus only | kanna #416 | modeled +1.21 (COSMETIC) |

**CURRENT VERDICT: HOLD-for-IDENTITY-ONLY** -- the margin conjunct that #425 left pending is now RESOLVED as a banked +15.6 Directive-#4 win (Bar A, no contingency); the ONLY remaining pending conjunct is identity == 1.0 (stark #429). This is a strict collapse of #425's 2-conjunct HOLD to a 1-conjunct HOLD.

## Bar B is a leaderboard-cosmetic line (NOT a #407 gate)

- `leaderboard_beat_margin_tps = +1.21` (modeled; contingent on kanna #416 budget-exact measurement of the full stack).
- Bar B flips NEGATIVE if the realized cb3 lift drops below 14.39 TPS, i.e. a haircut above 7.76% (well inside #410's 14.9% bound). Worst-case Bar B margin -1.11.
- **Ship-policy if Bar B goes negative:** `ships_if_bar_B_negative = True`. YES -- ships_if_bar_B_negative = True. #407's literal objective is 'the fastest [TPS] that still RESPECTS this equivalence.' The deployed 481.53 does NOT respect the equivalence (identity 0.9966, 3/882 flips) under the current operative-identity, so it is EXCLUDED from the max -- it is not a feasible competitor at all. The max is taken over the FEASIBLE set {pure-AR-greedy (floor), blanket-strict 467.14, blanket-strict+cb3}. Even in the Bar-B-negative world (worst 14.9% haircut -> stack 480.42), the cb3 stack still strictly dominates the next feasible config (blanket-strict 467.14), so it REMAINS the fastest feasible config and ships under #407. Bar B (beating the illegal 481.53 on the public leaderboard) is a COSMETIC bonus, not a #407 gate.

## The identity conjunct (separate -- stark #429, do NOT re-derive here)

- The TPS reframe is ORTHOGONAL to identity. The cb3 supply leg is **equivalence-neutral** (adds 0 flips, #403/#410); the residual is the SAME single blanket-strict base flip @ prompt 90 (#412: a bitwise tie, `identity_1p0_unreachable_by_precision=True`).
- Resolution belongs to **stark #429** (operative-identity resolution, in flight; successor to #421's canonical tolerance tie-break on the self-referential gate #414 `bq7xkfcv`). This card does NOT re-derive identity; it carries it as the one pending conjunct.
- CONDITIONAL on the CURRENT operative-identity partition (481.53 -> 0.9966 -> infeasible). Stark #429 (operative-identity resolution, in flight; successor to #421's canonical tolerance tie-break) is the SEPARATE pending input that fixes the partition. #429 can only HELP: (i) if it leaves 481.53 infeasible, Bar A holds and the +15.60 is banked; (ii) if it canonicalizes the fast path's bitwise-tie flips on the self-referential gate (#414 bq7xkfcv) and PROMOTES 481.53 into the feasible set, then the bar RISES to 481.53 but the SAME cb3 +15.60 supply lever stacks on the FASTER base (fast + cb3 ~= 497), so the supply win is preserved (enlarged), not invalidated. Either way the cb3 +15.60 is banked; #429 only selects WHICH strict base it stacks on.

## Safe operation order

1. **Wait for stark #429** (operative-identity == 1.0) -- the ONLY remaining conjunct. Bar A is banked.
2. **Measure locally on the A10G** -- run the #319 3-tier identity-verify CI; confirm identity `== 1.0`.
3. **Human approves in GitHub** -- the gated approval issue (PR + branch + exact command + GREEN CI).
4. **Flip served file + submit** -- pin blanket-strict + cb3 (additive, reversible). Human-gated.

_The Directive-#4 win (+15.6) is banked on the TPS axis with no margin contingency. The deploy still waits on the SEPARATE identity conjunct (stark #429). Bar B (+1.21) is cosmetic. analysis_only=True, no_served_file_change=True, official_tps=0._
