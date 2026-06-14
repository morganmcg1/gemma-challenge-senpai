# Tree E[T] break-even / margin-of-safety gate (PR #102) — the inverse of #100

**Verdict: AMBER — must-recover-most-of-the-way.** Inverting my #100 model
(`official_TPS = K_cal·E[T]/step_time·τ`, solved for E[T] at official=500), the
**tree must realize accept_length E[T]\* ≥ 4.624 central [4.481 opt, 5.026 cons]
to clear 500 alone** (`breakeven_ET_tree_alone`). That sits squarely in the
AMBER band (4.0–5.0). The load-bearing finding: **byteshark's as-built 2.097 is
nowhere near** — it must recover **+2.527 accept_length** (central; +2.929
conservative) — and **no lever stack pulls the break-even below 4.0.** The full
stack (tree+lk+splitk+persist) only reaches **4.339 central / 4.897 conservative**.
So the entire 500-path now hinges on denken #101's recoverable band reaching
**≥ ~4.34–4.62**; if recovery stalls at the linear-chain 3.844, **the tree does
not clear 500 even with every lever on.**

This reframes #100's GREEN. #100 said "tree clears 500 with margin" — but that
margin lives in TPS-at-fixed-E[T]=5.207. **In E[T] terms the margin is thin:**
the tree-alone break-even (4.624) is only **0.583 below the 5.207 ceiling
(central) and 0.181 below it conservatively.** The 500-path's true sensitivity is
to accept-length recovery, and the build is currently failing that badly.

Primary `breakeven_ET_tree_alone = 4.624` [4.481, 5.026]. Test
`ET_recovery_needed_from_2p10 = 2.527` tree-alone central (full-stack ladder
2.242–2.929). Gate GREEN (≤4.0) / AMBER (4.0–5.0) / RED (near 5.207, no lever
lowers it) → **AMBER**, leaning fragile at the conservative corner (5.026 ≈ the
analytical ceiling → zero margin).

---

## 1. The question (PR #102)

#100 proved the tree clears 500 **conditional on E[T]≈5.207**. byteshark's first
real tree build (`tree-v2-merge-eager-v1`, CURRENT_RESEARCH_STATE.md L44)
delivers **tok/step = 2.097** — accept-length collapsed, below even the linear
chain's 3.844. denken #101 is diagnosing WHERE the 2.10-vs-5.207 gap comes from
and what is **recoverable**. This gate answers the complement — what is
**required**:

> Given official_TPS = 500, what is the MINIMUM realized tree accept_length
> E[T]\* — alone and per compounding lever stack — and where do byteshark's
> 2.097 and denken #101's recoverable band fall relative to that break-even?

The intersection of "what denken says we can recover" and "what I say we need"
is the real go/no-go for the #1 lever. This is a **CPU-only modelling gate** (a
projection computes nothing served → greedy identity untouched by construction).

## 2. The inversion — why it is a clean linear rescaling (faithful to #100)

```
official_TPS = K_cal · (E[T] / step_time) · τ          # the #100 model
```

is **linear in the accept_length numerator E[T]**, and the tree's `step_time` is
a **TOPOLOGY (M=32) fact** — verify-GEMM width (FLAT M≤32), attention
amortization (1.06×), drafter, host/overhead, the fp32-star-attn haircut (#98),
and any denominator levers (SplitK #84, persistent-kernel #97). It is **not** a
function of how many drafted tokens get *accepted*; accept_length is a
numerator-only quality property. So hold `step_time` at its topology value and
solve official = 500:

```
E[T]*_raw = 500 · step_time / (K_cal · lk_factor · τ)
          = E_T_TREE · 500 / official_TPS_stack(@E[T]=5.207)      ← exact rescaling
```

The second form is exact because #100 already evaluated `official_TPS` at
E[T]=5.207 for every stack, so the break-even raw tree accept_length is just
`5.207 · 500 / (that stack's official@5.207)`. `lk_factor` (the tree LK
numerator boost) folds in automatically — dividing by the LK stack's official
returns the **raw** (pre-LK) accept_length. The script (`tree_et_breakeven.py`)
imports `compose`/`point` **verbatim** from #100's `lever_composition.py`; the
**max |direct − rescale| break-even error is 1.8e-15** (machine zero) → the
inversion is the same model, run backwards.

**What this buys over #100:** it SEPARATES the two quantities #100's `net_tree`
band bundled together — the M=32 **denominator widening** (a step_time fact) vs
the accept-length **numerator gain** (the free variable here). That is the
physically honest decomposition. In #100, the 558–581 envelope at fixed E[T]=5.207
is *all denominator uncertainty*; here it maps to the step_time band that sets
the break-even band.

## 3. Step 1 — the break-even E[T]\* ladder

Break-even **raw tree accept_length** E[T]\* to clear 500 official (cons ‖ central
‖ opt), and the recovery the build must claw back from the as-built 2.097:

| stack | cons | **central** | opt | recov@central | recov@cons |
|---|---:|---:|---:|---:|---:|
| **tree** (primary) | 5.026 | **4.624** | 4.481 | **2.527** | 2.929 |
| tree+splitk #84 | 4.922 | 4.458 | 4.254 | 2.361 | 2.825 |
| tree+lk #95 | 5.001 | 4.587 | 4.376 | 2.490 | 2.904 |
| tree+lk+splitk | 4.897 | 4.422 | 4.155 | 2.325 | 2.800 |
| *tree+persist #97* | *5.026* | *4.540* | *3.962* | *2.443* | *2.929* |
| *tree+splitk+persist* | *4.922* | *4.374* | *3.736* | *2.277* | *2.825* |
| *tree+lk+persist* | *5.001* | *4.504* | *3.869* | *2.407* | *2.904* |
| ***full stack*** *(tree+lk+splitk+persist)* | *4.897* | ***4.339*** | *3.648* | *2.242* | *2.800* |

(Italic = persist-inclusive upside. At the conservative corner persist reclaims
**nothing** — r_idle=0 per denken #97's 2.17%-idle finding — so its conservative
rows equal the no-persist rows.)

**Read-out.** The compounding levers move the break-even only a little: SplitK
shaves the verify slice (−0.17 central), LK boosts the numerator (−0.04
central), persist (central, optimistic only) helps most because it attacks the
fat "other" slice — but **the full stack still requires E[T] ≥ 4.339 central /
4.897 conservative.** SplitK is the single most useful lever for the break-even
(it acts on the 0.53 GEMM slice, the largest denominator term), but even
SplitK+LK+persist cannot pull the central break-even under 4.0. **The break-even
is governed by the M=32 step being ~1.16× heavier than linear; no lever
materially changes that.**

## 4. Step 2 — placing the empirical + reference points on the E[T] axis

Official-TPS at the **tree-alone central step** (1.158) for each anchor:

| anchor | accept_length | official @ tree step | clears 500? |
|---|---:|---:|---|
| **byteshark as-built** | **2.097** | **226.8** | no — a *regression* vs linear 481.53 |
| linear-chain | 3.844 | 415.7 | no — *still below linear's own 481.53* |
| **beat-linear-OFFICIAL floor** | **4.453** | 481.5 | ties linear (the true "worth-building" floor) |
| analytical ceiling | 5.207 | 563.1 | YES (this is #100's GREEN point) |

**Two load-bearing nuances:**

1. **byteshark's 2.097 makes the tree a regression.** At 2.097 the tree yields
   ~227 official — **less than half** the linear frontier (481.53) — because the
   tree pays a ~1.16× heavier step for *fewer* accepted tokens than linear. The
   as-built tree is strictly worse than shipping nothing.

2. **The "3.844 floor" undersells the bar.** The PR marks linear's 3.844
   accept_length as "the floor the tree must beat to be worth building" — but
   beating linear's *accept_length* is **not enough**. Because the tree step is
   heavier, the tree must reach **E[T] ≥ 4.453** just to *tie* linear's 481.53
   official. The honest "worth building" floor is **4.45, not 3.844.** The window
   where the tree beats linear but still misses 500 is narrow (4.45 → 4.62), so
   in practice the binary is almost "does the tree beat linear at all" — if it
   clears ~4.5 it is already within a whisker of 500.

3. **denken #101's recoverable band is parameterized** (not landed; I am
   launch-isolated to fern's PRs). The margin-of-safety curve below sweeps it.

## 5. Step 2 (cont.) — margin of safety vs the recoverable E[T] (the go/no-go)

`margin_of_safety(E_rec) = recoverable_E[T] − breakeven_E[T]*`. Central corner,
sweeping the recoverable accept_length denken #101 will report:

| recoverable E[T] | any stack clears 500? | cheapest clearing stack | tree-alone margin |
|---:|---|---|---:|
| 2.097 (as-built) | **no** | — | −2.527 |
| 3.000 | no | — | −1.624 |
| 3.844 (linear) | **no** | — | −0.780 |
| 4.000 | **no** | — | −0.624 |
| 4.340 | yes | full stack (all 4 levers) | −0.284 |
| 4.450 | yes | tree+lk+splitk | −0.174 |
| 4.620 | yes | tree+splitk | −0.004 |
| 5.000 | yes | **tree alone** | +0.376 |
| 5.207 (full) | yes | **tree alone** | +0.583 |

**The critical recoverable threshold is ~4.34–4.62 (central):**

- **recoverable < 4.34** → **no path to 500 with any lever stack** (central) →
  escalate: the build needs a new accept-length lever class, or must hit
  near-analytical E[T] exactly.
- **recoverable ∈ [4.34, 4.62)** → clears **only with compounding levers**;
  4.34 needs *all four*, 4.45 needs tree+lk+splitk, 4.62 needs just SplitK.
- **recoverable ≥ 4.62** → **tree alone clears** (central); ≥ 5.0 gives comfortable margin.

At the **conservative** corner everything shifts up ~0.4: tree-alone break-even
is **5.026 ≈ the 5.207 ceiling**, so the conservative path demands *near-perfect*
recovery AND favorable bands — **margin of safety just +0.181** even at full
recovery. This is the RED-adjacent edge of the AMBER verdict.

## 6. Step 3 — the gate

| Gate input | Value |
|---|---|
| `breakeven_ET_tree_alone` (**primary**) | **4.624** central [4.481 opt, 5.026 cons] |
| `ET_recovery_needed_from_2p10` tree-alone (**test**) | **2.527** central (band 2.49–2.93) |
| lowest break-even via full stacking (central) | 4.339 (still > 4.0) |
| margin of safety @ full recovery to 5.207 (tree-alone) | +0.583 central / **+0.181 conservative** |
| critical recoverable E[T] (central): tree-alone / any-stack | ≥ 4.62 / ≥ 4.34 |
| GREEN ≤4.0 / AMBER 4.0–5.0 / RED near 5.207 no-lever | **AMBER** (conservative corner RED-adjacent) |

**Verdict: AMBER.** The tree must recover **most of the way to 5.207** — to ≥4.62
to clear 500 standalone (central), or ≥4.34 with the full lever stack. Recovering
only to the linear-chain 3.844 is **insufficient** (3.844 < 4.34). Compounding
levers help but do not rescue a badly-under-recovered build — they shave ≤0.28
off the central break-even. The conservative corner (5.026) is essentially the
analytical ceiling, so under unfavorable bands there is **no margin at all**.

## 7. Recommendation — aim the build at E[T] ≥ 4.62, and treat 4.45 as the abort line

1. **The minimum accept-length target for land #71 / byteshark / chiku-inu's
   build is E[T] ≥ 4.62** (clears 500 tree-alone, central) — or ≥ 4.34 if SplitK
   #84 + LK #95 (+ persist) are also landed. Aim the build at the analytical
   5.207; **4.62 is the floor, not the goal**, because the conservative corner
   needs ~5.0.
2. **E[T] ≈ 4.45 is the abort line.** Below ~4.45 the tree does not even beat the
   linear frontier — it is strictly worse than shipping the current 481.53. If
   denken #101's recoverable band lands **below ~4.34, escalate**: no lever stack
   reaches 500 and the 500-path needs a new accept-length lever class.
3. **SplitK #84 is the most valuable break-even lever** (largest denominator
   slice); LK #95 is nearly free but barely moves the break-even (−0.04); persist
   #97 helps only off the conservative corner (and #97 already closed it at 2.17%
   idle). Sequence SplitK first if the build lands just short of tree-alone
   break-even.
4. **Consume denken #101 the moment it lands.** Plug its corrected recoverable
   E[T] into `margin_of_safety(E_rec)`: ≥4.62 = GREEN go (tree-alone), [4.34,4.62)
   = needs the lever stack, <4.34 = RED escalate.

## 8. Reconciliation with #100 (the forward twin)

#100 (forward: E[T]→TPS) and #102 (inverse: TPS→E[T]\*) are the **same model**.
#100's GREEN was correct *given* E[T]=5.207; #102 prices the **margin of that
assumption in accept-length units** and finds it thin (tree-alone break-even
4.62, only 0.58/0.18 below the ceiling central/conservative). The as-built 2.097
falsifies the assumption hard. **The verdict moves GREEN→AMBER not because the
model changed, but because the build's realized E[T] (now an empirical 2.097, not
the de-risked 5.207) is the binding variable** — exactly the gap denken #101 is
sizing.

## 9. Public-evidence note (launch-isolated)

All inputs are committed advisor-branch state or the PR #102 body — no inspection
of other students' unmerged branches. denken #101's recoverable band is
**parameterized** (not read) and consumed only as a free variable.

- **Frontier** 481.53 official / E[T]_linear 3.844 / E[T]_tree analytical 5.207 —
  CURRENT_RESEARCH_STATE.md L6–10; folded into `K_cal=125.268` (= #100).
- **byteshark as-built** `tree-v2-merge-eager-v1` tok/step 2.097, accept-hist
  `[0,5761,5061,1765,854,355,214,126,200]` (mean 2.102, matching the headline
  tok/step; CRS also quotes full≈1.1%) — CURRENT_RESEARCH_STATE.md L44. Only the
  mean 2.10 anchor is load-bearing here.
- **Lever bands** (net_tree/splitk_s/lk_mult_tree/fp32_haircut/r_idle/τ) carried
  verbatim from #100's `lever_composition.py` (MERGED).
- **Pending gates as bands:** persist idle (#97, 2.17%), fp32-star-attn (#98), τ (#99).

## 10. Reproduce

```bash
cd target/
python scripts/profiler/tree_et_breakeven.py --wandb \
  --wandb-name "fern/tree-et-breakeven" --wandb-group "tree-et-breakeven"
# CPU-only, no GPU; ~0.09s; peak RSS ~24 MB
# writes research/spec_cost_model/tree_et_breakeven_results.json
```

- **W&B run:** `l12ikxea` — wandb-applied-ai-team/gemma-challenge-senpai, group
  `tree-et-breakeven`.
- **Peak memory:** ~24.2 MB RSS (CPU; zero GPU).
- **No serving run / no `summary.json` / no HF Job** — CPU modelling gate;
  greedy token-identity untouched by construction.
