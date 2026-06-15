# EAGLE-3 eager-path break-even (PR #314, wirbel)

**Can the EAGLE-3 build's higher E[T] hit 500 TPS on the simple EAGER proposer,
letting the launch skip the T5/T6/T7 loopgraph rewrite?**

CPU-analytic, **0 GPU, 0 TPS, NO served-file change, NO HF Job, NO build.**
BASELINE 481.53 unchanged. W&B `fwqbz7zf`. Self-test **16/16**, NaN-clean.

## The question this closes

wirbel #312 (`9b1arani`, MERGED) priced the EAGLE-3-on-EAGER fallback at **402.1
TPS** (band [302.3, 471.0], −16.5% vs 481.53) but **explicitly at iso-MTP-acceptance**
(E[T]=3.844), holding "any EAGLE E[T] gain as a SEPARATE numerator axis, not credited
here." The open question: the eager floor is **not fixed — it rises with E[T]**. If
the build's higher E[T] lifts the eager path to ≥481.53 (no-regret) or ≥500 (target)
**without** the rewrite, the launch ships on the low-risk eager proposer and skips a
2-moderate + 1-correctness-rederivation surface that reopens all four gates
(greedy-identity / PPL≤2.42 / boot-500 / TPS).

## Cost model (stated explicitly)

A K=7-chain spec-decode step = `verify_fixed + drafter_loop`:

| component | value (µs) | scales with E[T]? |
|---|---|---|
| `verify_fixed` = decode − drafter_graph = 11600 − 566.49 | 11033.51 | **no** (drafter-independent) |
| `drafter_loop` graph (loopgraph) | 566.49 | no (fixed K=7) |
| `drafter_loop` eager (MTP) | 2859.34 | no (fixed K=7) |
| eager penalty (graph→eager launch tax) | 2292.85 | no |

**E[T] is the acceptance OUTPUT, not a step-cost input.** At fixed K=7 + fixed
drafter, the step is **constant in E[T]**, so

```
TPS(E[T]; step) = K_cal · E[T] · (decode_step / step)      [LINEAR, monotone]
```

calibrated (kanna #217) so `TPS(3.844; 11600) = K_cal·3.844 = 125.268·3.844 = 481.53`.
Reproduces #312 exactly: `481.53 · 11600/13892.85 = 402.059 → 402.1` (resid 0).

## The bind: the break-even E[T] is drafter-cost-dependent

The E[T] gain to 6.11 is **delivered by the heavier EAGLE-3 fusion/own-KV drafter**
(#293/#295 — the light MTP drafter caps at ~3.844). A heavier drafter inflates the
step. So the eager break-even is a **band, not a point**:

| regime | eager step (µs) | E[T]@481 | E[T]@500 | TPS@6.11 |
|---|---|---|---|---|
| MTP-light (#312 headline) | 13892.85 | 4.60 | **4.78** | 639 ⚠ |
| EAGLE-heavy 3× (iso-decode) | 18478.56 | 6.12 | 6.36 | 480.6 |
| launch-count upper | 11859.57 | 3.93 | 4.08 | 749 ⚠ |

⚠ **Light-curve absurdity:** extrapolating the constant-light-step curve to E[T]=6.11
gives **639 TPS > the loopgraph path's 500** — i.e. eager beating graph for the same
drafter, which is physically impossible. The MTP-light break-even (4.78) is therefore a
**mirage**: the light step belongs to the light drafter, which cannot reach 4.78
(caps ~3.844). Only the heavy EAGLE-3 drafter reaches high E[T], and it pays the heavy
step → consistent break-even-500 is **E[T] ≥ 6.36**, *above* the central build target
6.11 and at/above the upper bracket 6.8588.

## The rewrite's actual value at E[T]=6.11

At E[T]=6.11 the rewritten loopgraph (frontier-step) path nets ~500 by #295
construction. Backing out the implied heavy loopgraph step (17761 µs) and adding the
eager penalty (same heavy drafter, no capture):

| eager penalty | eager step (µs) | eager TPS@6.11 | **rewrite GAP** |
|---|---|---|---|
| 1× (optimistic) | 20053 | 442.8 | **+57.2 TPS** |
| 3× (#312 draft-ratio) | 24639 | 360.4 | **+139.6 TPS** |

Even break-even (481.53) is unreachable on the eager path under the 3× penalty
(needs E[T]=8.16 > 8.0 ceiling); 500 needs E[T]=8.48.

## Verdict

**`rewrite_avoidable_at_build_target = False` — the launch CANNOT skip the rewrite.**

Crediting the build's E[T] gain does *not* lift the eager path to target. At the
central build E[T]=6.11 the eager path nets **360–481 TPS** (below 500 under every
convention; below baseline 481.53 under the #295-reconciled gap), because the heavier
drafter that produces 6.11 also inflates the eager step. The rewrite is worth
**+57…+140 TPS** at the build target — decisively above the TPS noise floor.

- `eager_path_et_for_500` = **4.78** (headline MTP-light step — optimistic/mirage; the
  consistent EAGLE-heavy value is 6.36+, unreachable at 3× penalty).
- `rewrite_avoidable_at_build_target` = **False**.

## Caveats (carried)

- Eager cost is #312's **banked ESTIMATE**; the heavier own-KV EAGLE-3 drafter makes
  402.1 an **upper-bound** floor — the true eager floor is **lower** (worse).
- 6.11 is wirbel #295's **step-profile target** (conditional on the fusion-step
  profile), not a trained drafter; whether a1→0.9213 (#304) is trainable is a separate
  lane.
- The MTP-light eager curve is valid only near E[T]~3.844; its 4.78 break-even is a
  mirage (extrapolation gives eager>graph).
- Prices the eager-vs-rewrite **deployment decision** as a function of E[T]; builds
  **neither** path. NOT a launch. NOT a build.

## Reproduce

```bash
python3 research/validity/eagle3_eager_breakeven/eagle3_eager_breakeven.py \
  --self-test --wandb_name "wirbel/eagle3-eager-breakeven" \
  --wandb_group "eagle3-eager-breakeven"
```

Peak mem 12.1 MiB. Anchors (cite, not re-derived): #312 `9b1arani`, #295 `c334qaqu`,
#304 `dtf1ouml`, #217 central convention (K_cal=125.268, τ=1.218, 481.53).
