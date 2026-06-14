# PR #123 — Re-price the tree-free 500-path after #117: is the tree now mandatory?

**Verdict: 🔴 RED — the tree (land #71) goes from insurance to REQUIRED-for-500.**
Substituting denken **#117**'s PHYSICAL SplitK ceiling (net **1.56%** central, gross
band-high **7.81%**) for #105's assumed ≥4.44% / #109's assumed ubel ~8.5%, the composed
tree-free path **caps at 491.8 official TPS** — **8.2 below 500**. It clears 500 *only* at
the optimistic **7.81%** band-high (the 88%-GDDR6 corner #117 itself prices as
optimistic-not-expected). The cheap stack is **spent**: SplitK is at its physical wall,
palette/LK are tiny. **Closing 500 now requires the tree's E[T] numerator (≥ 4.624), so
the build-blocked tree at tok/step=2.10 (#101 fixable defect) becomes the critical path.**

- **Primary** `tree_free_500_ceiling_at_splitk_wall` = **491.8** central (band
  **[489.5 conservative corner, 527.3 optimistic band-high]**; 495.9 at the 3.20% gross
  wall). `clears_500_central = False`.
- **Test** `tree_required_et_to_clear_500` = **4.624** (bare tree; **4.555** with the
  surviving cheap levers) — ~**57%** up #101's recoverable band [3.844, 5.207].
- W&B `0yv2nw9s` (group `tree-free-500-reprice`). Repro:
  `python scripts/profiler/tree_free_500_reprice.py [--wandb]`.
- LOCAL, CPU-only, analytic. REUSES denken #105 compose + lawine #99 multiplier-CI + fern
  #102 break-even (over fern #100), swapping **exactly two inputs**: SplitK → #117 ceiling,
  τ → lawine #116 derived band. No GPU/vLLM/HF Job/submission/kernel build. Greedy identity
  untouched (SplitK 0-flip, palette bit-exact, LK prediction-only).

## What changed vs #105 / #109 (exactly two inputs; everything else reused verbatim)

| input | #105 / #109 assumed | #123 (this re-price) | source |
|---|---|---|---|
| **SplitK `s`** | ≥4.44% (#105) / ubel ~8.5% (#109) | **net 1.56% central**, band [1.6% net, **7.81% gross**] | denken **#117** physical ceiling |
| **τ** | asserted [0.96, 1.00] | **derived [0.9983, 1.00]**, central 1.0000 | lawine **#116** roofline |

The τ swap *lifts* the conservative floor (τ is no longer the drag #109 feared), so the
re-price **isolates the SplitK ceiling as the sole mover**. The model is otherwise
identical: `official = K_cal·(E[T]/step)·τ·(mult/mult_central)`, `vg = 0.53·(1−f_palette)/(1+s)`.

**Self-check (load-bearing):** with `s=3.20%` gross + corner levers, the model reproduces
#117's published cross-check to the decimal — **474.6 / 489.4 / 494.3** at τ=0.96/0.99/1.00.
The re-price is faithful to #117 by construction.

## Step 1 — tree-free TPS at the #117 SplitK wall (the PRIMARY)

| scenario | SplitK `s` | tree-free official TPS | clears 500? |
|---|---|---|---|
| conservative corner | 1.60% net | **489.5** | ❌ |
| **CENTRAL** | **1.56% net** | **491.8** | ❌ **← PRIMARY** |
| (central @ gross wall) | 3.20% gross | 495.9 | ❌ |
| optimistic band-high | 7.81% gross | **527.3** | ✅ (only here) |

**The straddle curve** (central levers, τ=1.0) shows exactly where 500 is crossed — and
it is crossed **above** the #117 central/measured wall, only inside the optimistic 88%-GDDR6
region:

| SplitK `s` | central-levers TPS | clears 500? | #117 status of this `s` |
|---|---|---|---|
| 1.56% (net measured) | 491.8 | ❌ | **PRIMARY ceiling** |
| 3.20% (gross measured) | 495.9 | ❌ | gross at measured 79.2% wall |
| 6.19% (net 88%-GDDR6) | 503.3 | ✅ | optimistic |
| 7.81% (gross 88%-GDDR6) | 507.2 | ✅ | **band-high (optimistic-not-expected)** |

Tree-free needs SplitK **≥ 4.84%** (central levers) / **5.84%** (conservative corner) to
clear 500 — matching lawine #116's own corner thresholds (5.49–5.84%). **#117 delivers
1.56% net central → MISS by a wide margin; 7.81% optimistic band-high → clears.** Classic
straddle, and the **central wall misses**.

## Step 2 — composed lever table (cumulative ΔTPS, central, #117 wall, τ=1.0)

| lever | cumulative TPS | ΔTPS | note |
|---|---|---|---|
| frontier (481.53, #52) | 481.5 | — | τ=1.0 (lawine #116 central), E[T]=3.844 linear |
| + SplitK #117 net 1.56% | 485.5 | **+3.95** | BW-util lever `vg/(1+s)`; **physical** ceiling (was assumed 8.5%) |
| + palette #110 (0.3%) | 486.9 | **+1.45** | lossless byte lever (bit-exact); double-quant #104 **DEAD/excluded** |
| + LK-loss #95 (1.010) | 491.8 | **+4.87** | E[T] numerator (+1.0% central, linear in TPS); prediction-only |
| **composed central total** | **491.8** | | **still BELOW 500** (LK-high upside 499.6, still < 500) |

The entire cheap stack adds **+10.3 TPS** over the frontier and lands at **491.8** — even
granting LK its high (1.024) end the upside total is **499.6**, *still* short of 500. There
is no cheap-lever combination that clears 500 at the #117 central wall. **double-quant #104
is excluded (DEAD/KILL)**; it is not a lever here.

## Step 3 — the tree re-price (the TEST) + the re-priced fern #106 crossover

Because tree-free misses 500, the binding question becomes **how much E[T] the tree must
realize**:

| quantity | E[T] | reference |
|---|---|---|
| **bare tree → clear 500** (TEST) | **4.624** | fern #102 break-even, the absolute gate |
| tree + splitk(#117) + lk → clear 500 | 4.555 | cheap levers shave the bar slightly |
| bare tree beat-linear floor | 4.453 | worth-building-at-all |
| as-built tok/step | **2.097** | land #71 build-blocked (#101 fixable defect) |
| #101 recoverable band | [3.844, 5.207] | tree E[T] envelope |

The clear-500 E[T] **4.624** sits **~57% up** the recoverable band — comfortably inside
[3.844, 5.207], so it is **physically recoverable IF the #101 build defect is fixed**, but
it is **far above the as-built 2.097**. The tree build is now the critical path to 500.

**Re-priced fern #106 crossover.** #106 asked the E[T] at which the tree *overtakes*
tree-free. With tree-free dropping from #106's C=518.1 to this re-price's **491.8**, that
overtake-floor crossover **drops 4.791 → 4.548** (4.480 with the full cheap stack). **But
that is no longer the binding gate:** since tree-free now *misses* 500 outright, the tree
must clear the **absolute 500** (E[T] ≥ 4.624), which is *higher* than the overtake-floor.
**The tree's status flips from bounded-UPSIDE (#106 AMBER, optional) to REQUIRED-for-500.**

## Step 4 — verdict + fleet hand-off

**🔴 RED.** Trigger: `tree-free central (491.8) < 500 at the #117 central wall`. The cheap
path caps **8.2 below 500** and clears it only at the optimistic 7.81% band-high (527.3) —
which #117 prices as *optimistic-not-expected* (the 88%-GDDR6 corner, above gate_up's
measured 79.2% wall). This is the RED arm of the PR's own gate: *"tree-free caps below 500
at the central wall → the tree is required for 500."*

This is the downstream consequence of #117's negative: SplitK was the last cheap lever that
could have carried 500 alone, and #117 proved it is at its physical wall (3.20% gross /
1.56% net, not the assumed 8.5%). With SplitK spent, palette (+1.45) and LK (+4.87) are too
small to bridge an 8.2-TPS gap.

### Hand-off

- **→ land #71 (the tree BUILD):** you are now **on the critical path**, not insurance. The
  tree must recover realized E[T] **≥ 4.624** (bare) / **4.555** (with splitk+lk) to clear
  500 — ~57% up #101's recoverable band, well above the as-built **2.097**. **Fixing the
  #101 tok/step=2.10 build defect is now the single highest-leverage 500-path action in the
  fleet.** Everything cheap has been tried and priced; the numerator lever is what's left.
- **→ denken #109 (the ship gate):** the AMBER hold was correct and is now *mechanistically*
  closed: tree-free 500 was never a tuning gap SplitK could close — it's a physical wall.
  The conservative corner now reads **τ/tree-gated AND below 500 on the cheap path**.
- **→ ubel #108 (SplitK BUILD):** consistent with #117 — **stop tuning SplitK past ~7.8%.**
  Even the full 7.81% optimistic band-high only just clears 500 (507.2 central-levers) and
  is not the expected wall. SplitK is not a 500-path on its own; it is a ~+4 TPS contributor
  to a stack that still needs the tree.
- **→ lawine #116 (τ):** the derived τ floor 0.9983 is doing its job — it *removed* τ as the
  conservative drag, which is what let us isolate SplitK as the sole cause of the miss. τ is
  no longer the gate; the SplitK ceiling is.

## Public-evidence cross-check (digest, 2026-06-14)

Public #1 **frantic-penguin skv64 489.63** (valid). The SplitK/argmax-block class —
byteshark **484.62**, need-for-speed **488.07** — realizes only **+0.6–1.7%** over the
481.53 frontier and **none clears 500**. This is field-side corroboration that SplitK alone
is at its physical wall (consistent with #117's 3.20% ceiling and this re-price's 485.5
post-SplitK central). The competitive field has not crossed 500 on the cheap path because,
per this re-price, **there is no cheap path across 500.**

## Validity caveat (SEPARATE GATE)

This is a **TPS-ceiling result only**. Even a tree-free *or* tree 500 still needs the
greedy-identity **VALIDITY ruling** (kanna #114 RED + the human contract), which sits on top
of ALL TPS math here. SplitK (0-flip), palette (bit-exact), and LK (prediction-only) are
greedy-lossless by construction, and the tree is greedy-exact, so this re-price does not move
the validity question — but it does not resolve it either.

## Bottom line

#117 falsified the precondition that made tree-free 500 GREEN. Re-priced against the
physical SplitK ceiling, the **cheap path tops out at 491.8 — 8.2 short of 500** — and clears
500 only at the optimistic 88%-GDDR6 band-high. The tree flips from #106's bounded-upside
insurance to **required-for-500**: it must deliver E[T] ≥ **4.624**, and the #101 build defect
(stuck at 2.10) is now the critical path. This is a valuable negative — it **redirects the
fleet's 500 effort from cheap-lever tuning (spent) to the tree build (the only remaining
numerator lever).**
