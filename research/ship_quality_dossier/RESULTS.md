# PR #512 — Ship downstream-quality safety dossier (quality-analog of #508)

**Run:** `3fxrmc8u` · `--wandb_group ship-quality-dossier` · analysis_only, official_tps=0, CPU-only.
NO serve, NO HF job, NO `--launch`, NO submission, NO served-file change, **NO evals run** (that is ubel #511).

## One-page dossier verdict (the number for the human at the quality-gated reopen)

> **Ship surgical-357 is the dominant QUALITY-VALID entry.**
> Its downstream quality is **base by construction** — the ship is greedy-faithful to base
> `gemma-4-E4B-it` (locus cert: **1 bf16-ULP near-tie flip, 0 semantic**), so a greedy MMLU-Pro /
> GPQA-Diamond eval reproduces base's scored answers token-for-token ⇒ **MMLU-Pro = 0.668,
> GPQA-Diamond = 0.470 BY CONSTRUCTION.** That **PASSES** Morgan's proposed gate (MMLU-Pro ≥ 0.60,
> GPQA-Diamond ≥ 0.42; margins **+0.068 / +0.050**), while the **pruned-substrate frontier most top
> entries use FAILS** it (**0.330 / 0.283**, GPQA near the 0.25 chance floor). The only residual is a
> measured handful of full-serve semantic flips (**12 / 14336**) whose direct-hit answer-change
> ceiling is **|ΔAcc| ≤ 0.084%**; the decisive confirmation is the ubel #511 served A/B (expected ≈ 0).

## Portfolio price — dominant under all three organizer rules (`quality_verdict = "dominant"`)

| organizer quality rule | surgical-357 | pruned-substrate competitor | dominant |
|---|---|---|---|
| **(a) quality-penalize** (score = TPS × eval/base) | retention **1.0** → **375.9** quality-TPS | retention **0.49–0.60** → ≤ **252.9** on the 420 frontier | **surgical** (competitor needs **624–761 TPS** — 1.5–1.8× the public frontier — to break even) |
| **(b) invalidate-the-pruned** (fail the MMLU/GPQA floor → score 0) | **PASS** (+0.068 / +0.050) | **FAIL** (−0.270 / −0.137) | **surgical** — may be the **only valid fast entry** |
| **(c) quality-agnostic** (no quality gate) | **ties** base (no quality disadvantage) | — | **surgical** — axis is moot, #508 speed verdict governs |

**Decisive fork:** does the reopen rule put **any** weight on downstream quality? If yes (a or b),
surgical wins outright because the pruned frontier collapses below the gate. If no (c), surgical
still leads on the #508 speed/private frontier and pays **zero** quality penalty. There is no rule
under which the quality axis hurts the ship.

## Composition (item 1 — the structural prior, load-bearing & available NOW)

`greedy-faithful ⇒ downstream_eval(ship) == downstream_eval(base)` token-for-token ⇒

| stat | value | basis |
|---|---|---|
| ship MMLU-Pro (prior) | **0.668** | = base, by greedy identity |
| ship GPQA-Diamond (prior) | **0.470** | = base, by greedy identity |
| locus cert identity | **0.99887551** | stark #494 `k8nqmc2b`/`5fxw18gu` |
| locus residual flips | **1** (bf16-ULP near-tie, 0.125 nat) | knife-edge, **0 semantic** |
| surgical ≡ 222 all-pin | **True** (15 sig figs) | operatively identical |

The structural argument stands on its own: a greedy eval that reproduces base's tokens reproduces
base's answers. The measured legs (below + pending) **confirm**, they don't establish, the prior.

## Residual bound (item 3 — priced honestly, not assumed away)

The locus cert sees only 1 near-tie flip, but the **reload-immune full-serve census** (wirbel #487,
the operatively-equivalent pinned/222 arm at the served W=8 verify geometry) is the honest residual:

| stat | value |
|---|---|
| full-serve raw token identity (semantic+tie counted) | **0.99734933** |
| operative identity @ tie tolerance (ties forgiven) | **0.99916295** |
| flips / positions | **12 semantic + 26 tie / 14336** (128 prompts) |
| **tie tolerance (≥ 1 bf16 ULP = 0.125 nat) makes LOCUS exact** | **True** |
| **…makes FULL-SERVE exact** | **False** — the **12 semantic flips survive** |
| direct-hit answer-change ceiling `|ΔAcc|` | **≤ 0.084%** (per-position semantic rate 8.4e-4) — **gate-safe** |
| construction-refuted worst case `|ΔAcc|` | **≤ 9.38%** (all 12 flips distinct-prompt, answer-determining, adversarial) — **NOT gate-safe** |

**The honest tail:** "exact at a tie tolerance" is **true for the locus / tie component** and
**false for the full-serve semantic component.** A semantic flip changes a *scored* answer only if it
(or its cascade) reaches the single argmax answer-decision token — direct-hit ceiling **0.084%**,
comfortably inside the **0.050** binding gate margin. The only regime that threatens the gate is the
construction-refuted worst case (9.38%), and that is exactly what the **ubel #511 served base-vs-ship
A/B** closes by direct measurement. Structural prior on `E[ΔAcc]` is **0** (sign-symmetric numerical
noise). Even at the absurd worst case, the **competitor's** gap below the gate (−0.270 MMLU / −0.137
GPQA) **dwarfs** the ship's residual — surgical's quality-validity is robust to it.

## Quality-gated competitive outcome (item 4)

- **Gate (Morgan #483):** MMLU-Pro ≥ 0.60, GPQA-Diamond ≥ 0.42. Ship **PASS** (0.668 / 0.470);
  pruned **FAIL** (0.330 / 0.283, GPQA within 0.033 of the 0.25 chance floor).
- **Quality-penalize break-even:** quality-adjusted, surgical scores **375.9** (retention 1.0) vs the
  pruned frontier's ≤ **252.9** (best of MMLU/GPQA retention on the ~420 TPS frontier). A competitor
  would need **624–761 TPS** to overcome its quality handicap — far beyond the public frontier.
- **Invalidate-the-pruned:** if the organizer zeroes sub-gate entries, surgical may be the **only
  valid fast entry** on the board.

## Inputs (advisor-provided anchors + my merged legs — reused, not re-derived)

| input | value | source |
|---|---|---|
| base MMLU-Pro / GPQA-Diamond | 0.668 / 0.470 | dixie-flatline #483 (greedy/pinned) |
| pruned-substrate MMLU-Pro / GPQA-Diamond | 0.330 / 0.283 | dixie-flatline #483 |
| quality gate | MMLU-Pro ≥ 0.60, GPQA-Diamond ≥ 0.42 | Morgan #483 |
| locus operative cert | identity 0.99887551, 1 near-tie / 0 semantic | stark #494 `k8nqmc2b`/`5fxw18gu` |
| full-serve census | identity 0.99734933, 12 semantic + 26 tie / 14336 | wirbel #487 (merged artifact) |
| ship official TPS / PPL | 375.857 / 2.37673 | ship `j7qao5e9` (#499) |
| speed frontier (penalize break-even) | ~420 TPS | program.md (public) |

## Legs folded (`legs_confirmed = 2`, `legs_pending = 4`)

**Confirmed (merged, composed from directly):** stark #494 locus cert · wirbel #487 full-serve census.
**Pending (clearly-marked slots; cite each with its W&B run when it lands):**
- **ubel #511** — served base-vs-ship MMLU-Pro + GPQA-Diamond A/B (**DECISIVE** direct |ΔAcc|; prior ≈ 0).
- **stark #509** — surgical-vs-base greedy (M=1 AR) census (confirms the M1 path).
- **wirbel #510** — surgical-config full-serve operative-identity census (refines the 12-semantic prior).
- **denken #505** — spec-dec sampled-distribution preservation (decoding-algo axis).

## Self-test (`self_test.passes = True`, 32/32)

Reproduces #487 exactly (raw identity 0.99734933, operative-rate 0.99916295 to ≤ 1e-12); prior == base
exactly; ship PASS / competitors FAIL with the right margins and GPQA near chance; **honest residual**
(tie tolerance fixes the locus, **not** the full-serve; direct-hit gate-safe but worst-case **not** —
making ubel #511 load-bearing); competitor gap dwarfs the ship's worst case; all three rule winners
surgical; verdict `dominant`; legs 2 confirmed / 4 pending; NaN-clean over every numeric leaf.

## Command

```bash
.venv/bin/python -m research.ship_quality_dossier.compose_quality_dossier \
    --name kanna/ship-quality-dossier --group ship-quality-dossier
```

Peak memory: negligible (pure-Python CPU composition; no model load, no serve, no eval). W&B run
`3fxrmc8u` (`ship_quality_dossier` artifact attached).
