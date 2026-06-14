# Tree-vs-tree-free crossover + build-milestone ladder (PR #106) — the capstone synthesis

**Verdict: AMBER — conditional / the tree is UPSIDE, not the critical path (pending denken #105).**
Crossing my tree model against the tree-free lever stack, the load-bearing
number is the **crossover E[T]ₓ(C)** — the realized tree accept_length at which
`tree_official(E[T]) = tree_free_ceiling C`. It is **exactly my #102 break-even
generalized from target=500 to target=C** (REUSED verbatim), so at **C=500 the
crossover is 4.624** — the synthesis is consistent with #102 by construction.

The decision surface is clean and entirely determined by where denken #105's
tree-free ceiling C lands:

| denken #105 ceiling C (central) | crossover E[T]ₓ (tree-alone) | verdict | meaning |
|---|---|---|---|
| **C < 500** | < 4.624 | 🟢 **GREEN** | tree-free can't hit 500 → tree on the critical path |
| **500 ≤ C < 540.7** | 4.624 – 5.000 | 🟡 **AMBER** | tree-free clears 500 → tree is **upside**, worth it only if recovery clears E[T]ₓ |
| **C ≥ 540.7** | ≥ 5.000 | 🔴 **RED** | tree barely beats tree-free even near its ceiling → pivot |
| **C ≥ 563.1** | > 5.207 | 🔴 **deep-RED** | tree can **never** beat tree-free, even at its analytical ceiling |

**My own independent bracket (pending #105) puts C in the AMBER region:** the
tree-FREE stack (SplitK #84 + LK #95, +double-quant #104) reads
**cons 481 / central 511 / opt 529** off my #100 linear-chain levers — it
**straddles 500**, centering in AMBER. So the most-likely call is **AMBER: 500 is
reachable tree-free, and the build-blocked tree is upside that only pays off if
recovery reaches E[T] ≥ ~4.73** (alone) — **~65 % of the way up denken #101's
recoverable band [3.844, 5.207]**. At the conservative corner (tree-free 481 < 500)
it flips GREEN and the tree re-becomes the only path to 500.

Primary `tree_vs_treefree_crossover_ET = 4.73` at the likely ceiling (≡ 4.624 at
the C=500 anchor). Test `build_milestone_ladder` (below). W&B `1qkiheqb`.

---

## 1. The question (PR #106)

My #100 (forward composition) and #102 (inverse break-even) fully characterized
the tree **in isolation**: it clears 500 iff realized E[T] ≥ 4.624, and it is a
**regression below 4.45** (the accept_length that ties linear's 481.53). denken
#101 diagnosed the as-built 2.097 as a **fixable build defect**, recoverable band
**[3.844, 5.207]**. But the tree no longer stands alone — denken #105 is pricing
the **tree-FREE** ceiling (SplitK #84 + LK #95 + double-quant #104, no tree). The
fleet now needs the **crossover**:

> At what realized tree-E[T] does the tree path **overtake** the best tree-free
> lever stack — and is continuing the (build-blocked, AMBER) tree build worth it
> given how much TPS the tree-free path already delivers?

Below the crossover the tree-free path wins and the build is not worth
continuing; above it the tree is the win. This is the synthesis of #102 (what the
tree *needs*) and denken #105 (what we get *without* it). **CPU-only modelling
gate** (a projection computes nothing served → greedy identity untouched by
construction).

## 2. The model — faithful to #100 and #102 by construction

The #100 forward model is `official_TPS = K_cal · (E[T] / step_time) · τ`, **linear
in the accept_length numerator E[T]**; the tree's `step_time` is an **M=32 topology
fact** independent of how many drafted tokens are accepted. So

```
tree_official(E[T]) = compose(levers, p).official · (E[T] / 5.207)        # Step 1
```

is an exact linear rescaling of #100's value at E[T]=5.207. The **crossover** with
a tree-free ceiling C is the E[T] that makes `tree_official(E[T]) = C`:

```
E[T]ₓ(C) = 5.207 · C / compose(levers, p).official
         = C · step_time / (K_cal · lk_factor · τ)                        # Step 2
```

which is **identical to #102's break-even inversion with target=C** — the script
calls `tree_et_breakeven.breakeven_raw_et(levers, p, target=C)` verbatim. **At
C=500 the crossover collapses to #102's 4.624** (rescale error ~1e-13). The
synthesis cannot drift from #100/#102; it *is* #100/#102 evaluated at a
parameterized target.

`K_cal = 481.53 / 3.844 = 125.268`. Two tree paths are crossed against C:

- **PRIMARY — tree ALONE** (matches the milestone-ladder gates 4.45 / 4.624; the
  honest "is the bare tree worth the *entire* tree-free stack" bar).
- **COMPANION — tree + splitk + lk** (the build-realistic *marginal* decision: if
  the fleet keeps the tree it also builds the compounding levers, so the tree gets
  their help → a lower, more tree-favorable crossover).

## 3. Step 1 — the build-milestone ladder `official(E[T])` (the TEST metric)

official-TPS as a function of realized tree E[T] (tree ALONE), at the
conservative / central / optimistic corners, with the load-bearing ship-gates
marked:

| gate | E[T] | official (cons / central / opt) | meaning |
|---|---|---|---|
| **as-built** | 2.097 | 209 / **227** / 234 | byteshark accept-collapse — a **deep regression** |
| **linear floor** | 3.844 | 382 / **416** / 429 | denken #101 structural floor — **still a TPS loss** vs 481.53 |
| **beat-linear** (abort line) | 4.45 cen | **= 481.53** (ties linear) | first ship-gate: below it the tree is a **net regression** |
| **clear-500** (#102) | 4.62 cen | **= 500** | target ship-gate: bare tree clears 500 |
| **analytical ceiling** | 5.207 | 518 / **563** / 581 | de-risked ceiling (fern #92) |

(beat-linear / clear-500 are *threshold* gates — the E[T] at which official ties
481.53 / 500 — so their E[T] varies by corner: beat-linear 4.841/**4.453**/4.316,
clear-500 5.026/**4.624**/4.481; official is fixed by definition.)

**Machine-readable ship-gate ladder for the build team:**

1. **beat-linear: realized E[T] ≥ ~4.45** — or the tree is a *net regression* vs
   the deployed linear 481.53. (This is the abort line; the tree is only worth
   *shipping* above it.)
2. **clear-500: realized E[T] ≥ ~4.62** — the bare tree hits the 500 target.
3. **stretch: realized E[T] → 5.207** — gives ~563 central (the de-risked ceiling).

The ladder is steep and **linear at ~108 official-TPS per unit E[T]** (central):
every +0.1 accept_length the build claws back ≈ +10.8 official TPS. From the
as-built 2.097, the build must recover **+2.36 accept_length just to beat linear**
and **+2.53 to clear 500**.

## 4. Step 2 — the crossover E[T]ₓ(C) (the PRIMARY metric)

Crossover vs the parameterized tree-free ceiling C (central corner; companion is
tree+splitk+lk):

| C (tree-free) | E[T]ₓ alone | E[T]ₓ +splitk+lk | tree-free ≥ 500? | reachable ≤ 5.207? |
|---|---|---|---|---|
| 480 | 4.439 | 4.246 | no | yes |
| 490 | 4.531 | 4.334 | no | yes |
| **500** | **4.624** | 4.422 | yes | yes |
| 510 | 4.716 | 4.511 | yes | yes |
| 520 | 4.808 | 4.599 | yes | yes |
| 530 | 4.901 | 4.688 | yes | yes |
| 540 | 4.993 | 4.776 | yes | yes |
| 550 | 5.086 | 4.865 | yes | yes |
| 563 | 5.206 | 4.980 | yes | ~at ceiling |
| 565 | 5.225 | 4.997 | yes | **no** |

**Three load-bearing threshold C values (central, tree-alone):**

- **C = 500** — below it the tree-free path can't hit 500, so the **tree is on the
  critical path** (GREEN region).
- **C = 540.7** — at it the crossover hits **5.0**; above it the tree only beats
  tree-free within the top ~0.2 of the recoverable band (RED region).
- **C = 563.1** — at it the crossover hits the **analytical ceiling 5.207**; above
  it the tree **can never** beat tree-free, even fully recovered (deep-RED).

**Placing denken #101's recoverable band [3.844, 5.207] against the crossover:**
a *plausibly-recovered* tree clears the crossover iff E[T]ₓ(C) ≤ 5.207, i.e. **C ≤
563.1** — but the band **floor 3.844 never clears any plausible C** (it gives only
416 official, below even linear). So the tree's official-TPS recoverable band is
**[416, 563] central**: it overtakes a tree-free ceiling C only if recovery
reaches the *upper* part of denken #101's band.

**Robustness — the corner-matched crossover is stable ~4.7–4.8.** Treating C as an
exogenous scalar (the PR's framing) makes E[T]ₓ swing 4.44 → 5.23 across the C
sweep. But if denken #105's ceiling is itself banded and we compare **corner-for-
corner** (the same optimism that lifts the tree-free ceiling *also* lifts the tree
— net_tree, splitk and lk bands move together), the crossover collapses to a tight
**4.834 / 4.727 / 4.737** (cons/central/opt). The decision is therefore far less
fragile than the raw sweep suggests: **whatever corner reality picks, the tree
must recover to ≈4.7–4.8 to overtake the tree-free path.**

## 5. Step 3 — the gate (continue-vs-pivot)

The verdict is a clean function of denken #105's ceiling C (central headline):

- 🟢 **GREEN / tree clearly worth it** — **C < 500.** Tree-free cannot hit 500
  alone → the tree is the *only* path to 500, and its crossover (< 4.624) is below
  the 500-break-even it must clear anyway → keep the build the #1 priority.
- 🟡 **AMBER / conditional** — **500 ≤ C < 540.7.** Tree-free clears 500 (denken
  #105 GREEN); the tree is **upside**, overtaking only if the build recovers past
  E[T]ₓ ∈ [4.62, 5.0] → name the recovery threshold; continue only if recovery
  clears it.
- 🔴 **RED / deprioritize** — **C ≥ 540.7.** Crossover ≥ 5.0: the tree barely beats
  tree-free even near its own ceiling → pivot to the tree-free levers + escalate
  for a fresh accept-length lever class. Deep-RED above C = 563.1.

**Most-likely call: AMBER**, on my own independent tree-free bracket
(cons 481 / **central 511** / opt 529, pending denken #105). At C ≈ 511 the tree is
upside that pays off only if **recovery reaches E[T] ≥ ~4.73** (tree-alone) /
**~4.52** (with splitk+lk also built) — **~65 % up denken #101's recoverable band.**
Two honest caveats on the boundaries:

- The bracket **straddles 500**: at the conservative corner (tree-free 481) the
  verdict flips **GREEN** — the tree re-becomes the only path to 500. So the
  GREEN/AMBER call hinges on confidence in the tree-free *central* estimate, which
  is exactly what denken #105 will pin.
- Even inside GREEN near C = 500, "critical path" does **not** mean "easy": the
  tree still needs ~4.6 recovery to clear 500. Being *necessary* (tree-free < 500)
  is not the same as being *cheap*.

## 6. What this hands the fleet

1. **The milestone ladder** (§3) — intermediate ship-gates for the build team:
   *beat-linear at E[T] ≥ 4.45, clear-500 at 4.62, stretch to 5.207*, at ~10.8
   official TPS per +0.1 accept_length.
2. **The crossover curve + three thresholds** (§4) — the exact C values that flip
   the verdict (500 / 540.7 / 563.1), ready to consume denken #105's number.
3. **The continue-vs-pivot gate** (§5) — AMBER in the likely region: **500 is
   reachable tree-free, so the build-blocked tree is upside worth continuing only
   if recovery clears E[T] ≈ 4.7**. The synthesis of #100 + #102 + #101 + #105.

## 7. Limitations & faithfulness

- **Parameterized on denken #105.** The tree-free ceiling C is an input; the
  headline verdict consumes my own #100-derived bracket as the *likely* value and
  is flagged pending #105. Re-run with `--treefree-ceiling <C>` when #105 lands to
  fix the headline. My bracket omits a proper double-quant #104 interaction (I
  apply it as a flat +0.4–1.1 %); denken #105 prices it natively.
- **Companion crossover is slightly conservative.** The tree+splitk+lk path is
  compared against a C that *includes* double-quant, while the tree-path side
  omits it — under-crediting the tree by ~0.75 % (≈0.03 in E[T]). Adding
  double-quant to the tree side (a flat M≤32 GEMM saving, ~cancels) would lower
  the companion crossover marginally further.
- **Faithful to #100/#102 by construction.** The crossover reuses
  `breakeven_raw_et` verbatim; the C=500 cross-check reproduces 4.624 to ~1e-13.
  Every tree step_time is #100's `compose` value; no independent re-derivation.
- **No served change.** CPU-only analytic; no GPU, no vLLM, no HF Job, no
  submission. Greedy identity untouched by construction.
