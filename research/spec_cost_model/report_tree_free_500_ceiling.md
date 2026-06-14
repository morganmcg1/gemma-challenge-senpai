<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Tree-free 500-path ceiling gate (PR #105) — does the build-complete stack clear 500 with NO tree, and at what SplitK threshold?

**Verdict: 🟢 GREEN — the tree is INSURANCE, not critical-path.** Composing only
the build-COMPLETE levers (SplitK #84 + LK #95 + double-quant #104) with **no
tree** (E[T] held at the linear 3.844), the tree-free stack **clears 500 at a
SplitK speedup of just 4.44%** at central/expected inputs — *below ubel #84's own
+5% delivery floor*. At ubel's nominal-high SplitK (+12%) the tree-free stack
reaches **`tree_free_max_official_tps = 518.1 [496.8, 540.8]`**; the absolute
ceiling (fully closing the 22.9% HBM bandwidth gap, +29.7%) is **556.0 [533.2,
581.1]**. **`splitk_threshold_for_500` = 4.44% central / 13.44% conservative**,
both well inside the 22.9% bandwidth-gap ceiling (15% and 45% of it). 🔴 RED is
firmly excluded.

**The decisive de-risk:** the denken #101 build defect (as-built tree
tok/step=2.10; star-attention not CUDA-graph-safe under FULL capture, public
board chiku-inu 2026-06-14) does **NOT block 500**. 500 is reachable today from
levers that are already build-complete, at a SplitK the kernel team can plausibly
hit. **A build slip does not sink the target — the tree becomes pure upside.**

**The one caveat is tau, not SplitK.** The conservative threshold (13.44%, just
past ubel's nominal +12% high) is driven almost entirely by lawine #99's
local→official transfer `tau`: at `tau=1.00` the threshold is 4.44%, at
`tau=0.96` it is 13.20%. The cheapest margin lever is **pinning lawine #99's tau
≥ 0.98** (drops the conservative threshold to 8.65%, inside ubel's range), *not*
more SplitK and *not* the tree.

Primary `tree_free_max_official_tps = 518.1` [496.8, 540.8] (ubel-high s=12%).
Test `splitk_threshold_for_500 = 4.44%` central (13.44% conservative). Gate
GREEN (insurance) / AMBER (tree strongly preferred) / RED (tree critical-path) →
**GREEN**, with a tau-gated conservative caveat.

---

## 1. The question (PR #105)

fern #100 proved the *tree alone* clears 500 (cons 518 / cent 563), but that
assumes E[T]=5.207, and the tree is **BUILD-BLOCKED / re-measure-pending**: the
as-built tree gives tok/step=2.10 (denken #101), and — confirmed on the public
board 2026-06-14 (chiku-inu) — the **star-attention verify op is not
CUDA-graph-safe under FULL capture** (tok/step collapses to 1.098 + illegal
memory access under graphs; only 2.097 under enforce-eager; fix `cudagraph_mode=
PIECEWISE` in flight). We do not control when the build team unblocks it.

This gate prices the **complement** of fern #102 (tree-INCLUSIVE break-even: "how
good must the tree be?"). It asks the tree-EXCLUSIVE question: **do we even need
the tree?** What is the maximum official TPS from the levers that are
build-complete today, and at what SplitK% does that tree-free stack clear 500?
Their union is the full 500 decision matrix.

This is a **CPU-only modelling gate** — a projection model computes nothing
served, so greedy token-identity is untouched by construction. All three levers
are greedy-lossless: SplitK 0-flip (kanna #87), double-quant bit-exact-or-sparse
(#104 build-or-kill), LK prediction-channel only (#95). No GPU, no vLLM, no HF
Job, no submission, no served-file change.

## 2. The model — fern #100 absolute-time slice composition, tree-EXCLUSIVE branch

```
official_TPS = K_cal * (E[T] / step_time) * tau ,   K_cal = 481.53 / 3.844 = 125.268
```

The deployed M=8 linear-MTP step = 1.0, decomposed into ABSOLUTE slices
(`CURRENT_RESEARCH_STATE.md`): **verify-GEMM 0.53** (int4 W4A16 Marlin,
BW-bound, **77.1% HBM util** at M=8 → a **22.9% bandwidth gap** = the SplitK
headroom ceiling, #68), drafter 0.07, attention 0.08, other 0.32. E[T] stays at
the **linear 3.844** — no tree.

### Tree-free lever placement

| Lever | Axis | Mechanism | Tree-free effect |
|---|---|---|---|
| **LK #95** | E[T] **numerator** | prediction-channel head improvement | `E[T] → 3.844·lk_mult`, lk_mult central **1.010** (floor; +1.0–2.4% band) |
| **SplitK #84** | verify-GEMM **denominator** (bandwidth util) | closes 77.1%→100% HBM gap; reads the SAME bytes faster | `vg → vg / (1+s)`; **swept** (ubel +5–12%, ceiling +29.7%) |
| **double-quant #104** | verify-GEMM **denominator** (byte count) | INT8 scale-of-scales + FP16 sparse exceptions; reads FEWER bytes | `vg → vg·(1−f_dq)`; isolated +0.4–1.1% |

### Honest netting of SplitK × double-quant (the PR's "don't double-count the bytes")

verify-GEMM time = **bytes / bandwidth**. SplitK acts on **bandwidth**
(utilisation — it removes *no* bytes); double-quant acts on the **byte count**
(it changes *no* utilisation). They are **orthogonal factors of the same slice**,
so they compound **multiplicatively with no double-count**:

```
vg = 0.53 · (1 − f_dq) / (1 + s)
```

A double-count would only arise if *both* reduced bytes — SplitK does not (the
#68 roofline classifies it as a utilisation lever; #104 is the byte lever). This
is the one-line answer to "net them honestly": multiply, because orthogonal. (If
one pessimistically treated them as a shared byte pool the combined verify-GEMM
saving would shrink by < 0.02 step-units — it does not move the verdict.)

### Carried input bands (same scenario machinery as fern #100)

| Band input | low | central | high | provenance / note |
|---|---|---|---|---|
| `lk_mult` (LK numerator) | 1.010 | 1.010 | 1.024 | #95/#100; central near floor (#80 single-layer capacity). **UNREALIZED** — prediction channel needs a head probe; applied as projected, not banked |
| `doublequant_isolated_tps` | +0.4% | **+0.5%** | +1.1% | #104: g=128 realistic +0.4–0.6%, central +0.5%; g=32 upside +1.1% (re-quant, gated on re-validation) |
| `fp32_m8` (ABS step add) | 0.0 | 0.0 | 1.0e-4 | #98: tree-free M=8 has **no star-attention**; deployed split-KV already fp32 → ~0. M=8 conservative bound 0.0102% carried for the cons corner. Negligible |
| `persist_reclaim` (ABS subtract) | 0.0 | 0.0 | 0.0217 | #97 MERGED ceiling **2.17%** (realizable 1.76%, recommend CLOSE). No tree to hide idle → reclaims full ceiling, UPSIDE-ONLY. **Supersedes #100's stale R_IDLE high=0.13** |
| `tau` (local→official) | 0.96 | 1.00 | 1.00 | lawine #99; deployed ratio 1.0599 folded into K_cal |

`conservative` = the corner that MINIMISES tree-free TPS (weak gains, heavy
haircuts, τ=0.96); `optimistic` = MAXIMISES it. The **one deliberate deviation
from #100's literal band** is persist: #100 used R_IDLE high=0.13 (a pre-#97
guess); denken #97 MERGED measured the reclaimable idle at 2.17%, so this gate
uses the post-#97 honest band [0, 0, 0.0217]. Persist barely moves the threshold
either way.

## 3. Step 1 — the tree-free TPS-vs-SplitK% landscape

(conservative .. central .. optimistic; "clears 500?" at which corner)

| SplitK% | cons | centr | opt | clears 500? |
|---:|---:|---:|---:|---|
| 0.00% | 468.7 | 488.8 | 509.7 | opt only |
| 4.44% (= central thr) | 479.5 | **500.0** | 521.6 | central |
| 5.00% (ubel low) | 480.8 | 501.4 | 523.1 | central |
| 8.50% (ubel central) | 488.9 | 509.9 | 532.1 | central |
| 12.00% (ubel high) | **496.8** | **518.1** | 540.8 | central |
| 13.44% (= cons thr) | 500.0 | 521.4 | 544.3 | **conservative** |
| 29.70% (gap ceiling) | 533.2 | 556.0 | 581.1 | **conservative** |

Read-out: at central inputs the tree-free stack clears 500 by **+12% SplitK at
the latest**, and in fact at **4.44%** — below ubel's own +5% floor. The
optimistic corner (high-LK + g=32 double-quant + persist ceiling) clears 500 with
**no SplitK at all** (509.7 at s=0). Only the fully-stacked conservative corner
needs SplitK past ubel's nominal high.

- **`tree_free_max_official_tps` (PRIMARY, ubel-high s=12%): 518.1 [496.8, 540.8].**
- **`tree_free_ceiling` (gap-close s=29.7%): 556.0 [533.2, 581.1].**

## 4. Step 2 — the SplitK threshold for 500 (the load-bearing number)

| corner | `splitk_threshold_for_500` | as % of the 22.9% gap | vs ubel +5–12% |
|---|---:|---:|---|
| **central** | **4.44%** | 15% | **below ubel's floor** |
| conservative | 13.44% | 45% | just above ubel's high |
| optimistic | 0% (clears at s=0) | — | none needed |
| central, **if LK delivers 0** | 6.48% | 22% | inside ubel's range |
| central, if double-quant 0 | 5.43% | 18% | inside ubel's range |

**The threshold is gated by tau (#99), not by SplitK delivery.** Holding LK +
double-quant at central and sweeping the local→official transfer:

| tau | splitk_threshold_for_500 |
|---:|---:|
| 1.00 | 4.44% |
| 0.99 | 6.50% |
| 0.98 | 8.65% |
| 0.97 | 10.88% |
| 0.96 | 13.20% |

The entire spread between the central threshold (4.44%) and the conservative
threshold (13.44%) is the τ=0.96 haircut. **Cheapest conservative-corner margin
lever** (each moved to central, conservative threshold after):

| lever | conservative threshold after |
|---|---:|
| **tau → 1.00** | **4.66%** |
| persist → 2.17% | 8.37% |
| LK → 1.024 (high) | 10.36% |
| double-quant → +1.1% (high) | 11.96% |

Pinning lawine #99's tau drops the conservative threshold from 13.44% to 4.66% —
**by far the biggest mover**, ~3× any other lever and ~6× the tree's relevance
here (the tree is not in this stack at all).

## 5. Step 3 — the gate

| Gate input | Value |
|---|---|
| `tree_free_max_official_tps` (primary, ubel-high s=12%) | **518.1** [496.8, 540.8] |
| `tree_free_ceiling` (gap-close s=29.7%) | 556.0 [533.2, 581.1] |
| `splitk_threshold_for_500` central (test) | **4.44%** (15% of the 22.9% gap, below ubel's +5% floor) |
| `splitk_threshold_for_500` conservative | 13.44% (45% of the gap, just past ubel's +12% high) |
| RED excluded? | **Yes** — tree-free clears 500 at 4.44–13.44%, all inside the gap ceiling |
| dominant swing variable | **lawine #99 tau** (4.44% @ τ=1.00 → 13.20% @ τ=0.96) |
| GREEN insurance / AMBER tree-preferred / RED critical-path | **🟢 GREEN** (tau-gated conservative caveat) |

**Verdict logic.** RED (the PR's "cannot clear 500 at any plausible SplitK") is
**firmly excluded**: the tree-free stack clears 500 at 4.44% central / 13.44%
conservative — both inside the 22.9% bandwidth-gap ceiling (15% and 45% of it,
never "near the top with no margin"). At central/expected inputs the threshold is
**below ubel's own +5% delivery floor**, so the tree is **INSURANCE**: the #101
build defect does NOT block 500, and a build slip does not sink the target. The
single residual risk is the conservative corner (13.44%, just past ubel's nominal
+12%), which is **tau-driven** (lawine #99) — at τ ≥ 0.98 it falls to ≤ 8.65%,
inside ubel's range. Hence the GREEN verdict carries a tau-gated caveat: **tree
strongly preferred only if τ collapses to its 0.96 worst case AND ubel lands at
its +5% floor AND LK delivers nothing** — a quadruple-worst corner.

## 6. Recommendation — the 500 decision matrix (with fern #102)

1. **The tree is OFF the critical path for 500.** Treat the #101 build defect as a
   backlog item (upside lever), not a target-blocker. The CUDA-graph
   star-attention incompatibility (public board) can be fixed at the team's pace
   without risking the target.
2. **The tree-free 500-path runs through SplitK #84 + lawine #99.** SplitK only
   needs +4.44% (central) — below its own sized floor. **The binding gate is
   lawine #99's tau**, not SplitK delivery. Prioritise pinning τ; if τ ≥ 0.98,
   500 is GREEN with margin even at otherwise-conservative inputs.
3. **LK #95 is helpful but not required.** Even at lk_mult = 1.0 (LK delivers
   nothing — its prediction channel is still unrealized, #95 AMBER), central
   clears 500 at 6.48% SplitK, inside ubel's range. Do not gate the 500-path on
   the LK head probe.
4. **double-quant #104 is a small, clean compounding lever** (+0.4–1.1%); it
   trims the threshold by ~1pp and is greedy-lossless by construction. Worth
   landing, not worth waiting on.
5. **Union with fern #102** (tree-INCLUSIVE break-even): fern answers "how good
   must the tree be?"; this gate answers "do we need it?" — **no**, for 500.
   Together they say: ship the tree-free SplitK+#99 path to 500 now; bank the tree
   as the lever that pushes 500 → ~556–581 once the build unblocks.

## 7. Public-evidence note (launch-isolated)

Composition inputs are **committed advisor-branch state only** — no numbers from
other students' unmerged branches:

- **Frontier** `fa2sw-precache-splitkv-linear-mtp-k7` = **481.53 official**,
  E[T]=3.844, folded into `K_cal` (lawine #52/#90).
- **denken #68 (committed):** verify-GEMM 77.1% HBM util at M=8 → the 22.9%
  bandwidth-gap ceiling that bounds SplitK.
- **ubel #84 (WIP):** SplitK +5–12% sizing — parameterised, not consumed as a
  fixed number.
- **fern #95 (MERGED):** LK +1.0–2.4% E[T], prediction channel, central near floor.
- **wirbel #104 idea (committed `RESEARCH_IDEAS_2026-06-14_04:40.md`):**
  double-quant +0.4–1.1% byte lever, g=128 vs g=32.
- **wirbel #98 (MERGED):** fp32 star-attn ~free; M=8 path already fp32 split-KV.
- **denken #97 (MERGED):** reclaimable idle 2.17% ceiling (supersedes #100's 0.13).
- **lawine #99 (WIP):** local→official tau band [0.96, 1.00].

**Public board context** (program.md intake; used only to confirm the tree's
build-blocked status is real and timeline-uncertain, NOT as composition input):
chiku-inu 2026-06-14 06:21 UTC isolated the tree's CUDA-graph star-attention
incompatibility; byteshark 05:53 UTC confirmed tok/step=2.10 not quota-ready;
openevolve 02:38 UTC confirmed the drafter is at its acceptance ceiling (E[L]
headroom is tree/sibling, not retraining) — consistent with LK #95's marginal
prediction-channel sizing.

## 8. Reproduce

```bash
cd target/
.venv/bin/python scripts/profiler/tree_free_500_ceiling.py --wandb \
  --wandb-name "denken/tree-free-500-ceiling" \
  --wandb-group "tree-free-500-ceiling"
# CPU-only, no GPU; ~seconds; peak RSS ~30 MB
# writes research/spec_cost_model/tree_free_500_ceiling_results.json
```

- **W&B run:** `0kiktnqt` — wandb-applied-ai-team/gemma-challenge-senpai, group
  `tree-free-500-ceiling`.
- **No serving run / no `summary.json` / no HF Job** — CPU modelling gate; greedy
  token-identity untouched by construction.
