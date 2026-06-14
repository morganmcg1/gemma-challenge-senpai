# Lever-composition economics gate (PR #100) — composed official-TPS landscape + minimal lever ordering to clear 500

**Verdict: GREEN — tree-sufficient.** Composing the four in-flight 500-path
levers in ABSOLUTE-time slice space (not relative multipliers) over the committed
decode budget, the **tree (land #71) alone clears 500 official-TPS with margin
even at the conservative corner**: `tree_alone_official = 563.1 [518.0, 581.0]`
(central [conservative, optimistic]). `min_levers_to_clear_500_conservative =
['tree']` (n=1, 518.0). The full stack reaches `composed_official_tps = 600.0
[531.6, 713.7]`. **Every other lever is INSURANCE / upside, not a requirement.**

The anti-compounding map is the actionable secondary finding: the three
non-tree levers compound multiplicatively *with each other* on the tree base, but
**each is DILUTED by the tree** — persistent-kernel #97 worst (dilution 0.60),
LK #95 next (0.80), SplitK #84 mildest (0.86, pure geometric). So once the tree
lands, the marginal value of the remaining levers shrinks; sequence the tree
FIRST and treat the rest as margin.

Primary `composed_official_tps = 600.0` (full stack, band [531.6, 713.7]). Test
`min_levers_to_clear_500 = 1` (`['tree']`, conservative). Gate GREEN (tree clears
500 at the conservative corner) / AMBER (needs 1 lever or full stack) / RED (stack
straddles 500) → **GREEN**.

---

## 1. The question (PR #100)

Every in-flight lever on the 500-path is now sized or being sized, but each is
priced *in isolation*. PR #100 asks: fold them into ONE composed official-TPS
projection so the team sequences the 500-path optimally. Three steps:

1. **Classify** each lever by WHERE it acts on `official-TPS = f(E[T],
   wall_time_per_step)` — E[T]-numerator vs wall-time-denominator.
2. **Compose** official-TPS for the lever subsets/orderings that matter, carrying
   input bands for the still-pending gates.
3. **Gate** — report `composed_official_tps` per subset, `min_levers_to_clear_500`,
   and the anti-compounding map (which pairs fight). GREEN/tree-sufficient |
   AMBER/one-lever-needed | RED/stack-insufficient.

This is a **CPU-only modelling gate**: a projection model computes nothing
served, so the greedy token-identity contract is untouched by construction. No
GPU, no vLLM, no HF Job, no submission, no served-file change.

## 2. The model — absolute-time slice composition (why it's correct)

```
official_TPS = K_cal * (E[T] / step_time) * tau
```

The deployed M=8 linear-MTP decode step is normalised to **1.0** and decomposed
into ABSOLUTE time slices from the committed budget
(`research/CURRENT_RESEARCH_STATE.md`): **verify-GEMM 0.53** (int4 W4A16 Marlin,
weight-BW-bound, FLAT for M≤32, hard tile cliff at M=33), **drafter 0.07**,
**attention 0.08** (conc=1 BW floor), **other/overhead 0.32** (host-device
scheduling, Python round-trips — largely un-mined). `K_cal = 481.53/3.844 =
125.268` is fixed so the deployed frontier reproduces exactly at
(E[T]=3.844, step=1.0, τ=1). At τ=1 with only the tree on, the model gives
`481.53·(1+net_tree)` — i.e. the committed denken #85 / fern #92 projection.

Working in absolute-time space is what makes the composition correct:

- **E[T]-NUMERATOR levers** (tree, LK) multiply E[T].
- **WALL-TIME-DENOMINATOR levers** (SplitK, persistent-kernel) subtract an
  **absolute** saving from the slice they act on. Two denominator levers on
  *different* slices add their absolute savings and the final step is
  **order-independent** — the apparent "ordering matters" is purely an artefact of
  attributing RELATIVE gains sequentially against a shrinking base.

### Step 1 — lever classification

| Lever | Axis | Committed effect | Note |
|---|---|---|---|
| **tree #71** | **E[T] numerator** (+M-widen denom) | E[T] 3.844→5.207, net **+18.2%** (band 558–581 official) | verify-GEMM stays 0.53 ABS (FLAT M≤32, #85); attention amortises 1.06× (#85). Carries fp32-star-attn haircut (#98) |
| **SplitK #84** | **wall-time denom** | +5–12% verify-GEMM speedup on the 0.53 ABS slice | GEMM flat ⇒ same ABS saving in linear AND tree |
| **persist #97** | **wall-time denom** | reclaims GPU-idle slice of the 0.32 "other" bucket (+8–15% IFF idle) | **ANTI-compounds with tree**: longer M=32 compute hides idle + amortises fixed overhead |
| **LK #95** | **E[T] numerator** | +1.0–2.4% E[T] (prediction channel; re-rank CLOSED) | partial overlap with tree rank-2 harvest (#88) ⇒ marginal value shrinks on tree |

### Inputs carried as BANDS (pending gates #97/#98/#99)

Sized from the PR body + committed advisor-branch state (NOT by inspecting other
students' unmerged branches):

| Band input | low | central | high | provenance |
|---|---|---|---|---|
| `net_tree` (tree NET local gain) | 0.1588 (558) | 0.1796 (568) | 0.2065 (581) | fern #92 envelope |
| `splitk_s` (GEMM speedup) | 0.05 | 0.085 | 0.12 | SplitK #84 +5–12% |
| `lk_mult` (LK on linear) | 1.010 | 1.010 | 1.024 | fern #95; central near #80 floor |
| `lk_mult_tree` (LK on tree) | 1.005 | 1.008 | 1.024 | partial-overlap haircut |
| `fp32_haircut` (ABS step add) | 0.0 | 0.01 | 0.04 | wirbel #98; likely ~free at conc=1 BW floor |
| `r_idle` (reclaimable idle) | 0.0 | 0.03 | 0.13 | denken #97; #65/#94 GPU-bound ⇒ LOW prior |
| `a_tree_hide` (tree hides idle) | 0.0 | 0.30 | 0.50 | denken #97 |
| `tau` (local→official) | 0.96 | 1.00 | 1.00 | lawine #99; 1.0599 folded into K_cal |

`conservative` = the corner that MINIMISES composed TPS (weak gains, heavy
haircuts, τ=0.96); `optimistic` = the corner that MAXIMISES it.

## 3. Step 2 — the composed official-TPS landscape

(conservative .. central .. optimistic; "clears 500?" at which corner)

| levers | cons | centr | opt | clears 500? |
|---|---:|---:|---:|---|
| (frontier) | 462.3 | 481.5 | 481.5 | no |
| lk | 466.9 | 486.3 | 493.1 | no |
| persist | 462.3 | 496.4 | 553.5 | no |
| splitk | 474.2 | 502.4 | 510.5 | yes (central) |
| **tree** | **518.0** | **563.1** | **581.0** | **YES (conservative)** |
| lk+tree | 520.5 | 567.6 | 594.9 | YES (conservative) |
| persist+tree | 518.0 | 573.5 | 657.1 | YES (conservative) |
| lk+persist+tree | 520.5 | 578.1 | 672.9 | YES (conservative) |
| splitk+tree | 529.0 | 584.0 | 612.0 | YES (conservative) |
| lk+splitk+tree | 531.6 | 588.7 | 626.6 | YES (conservative) |
| persist+splitk+tree | 529.0 | 595.2 | 697.0 | YES (conservative) |
| **lk+persist+splitk+tree** | **531.6** | **600.0** | **713.7** | **YES (conservative)** |

Read-out: **no single non-tree lever clears 500 at its own conservative corner**
(splitk gets there only centrally, 502.4). The tree is the only standalone lever
that clears 500 even conservatively (518.0), and it clears centrally with a large
margin (563.1). Note **`persist+tree` conservative = `tree` conservative = 518.0**:
at the conservative corner `r_idle = 0`, so persistent-kernel reclaims *nothing*
— honest behaviour given the #65 (decode 99.41% GPU-bound) / #94 (A10G bus
serialises) prior that the idle slice may be ≈0.

## 4. Step 3 — the gate

### Minimal lever set to clear 500

| Gate | Set | n | TPS |
|---|---|---:|---:|
| `min_levers_to_clear_500_conservative` | **`['tree']`** | **1** | **518.0** |
| `min_levers_to_clear_500_central` | `['tree']` | 1 | 563.1 |

The tree clears 500 alone at both corners. The GREEN margin is robust: to drive
tree-alone conservative below 500 you would need the fp32 haircut to *double*
beyond its modelled worst case (≈0.084 vs high 0.04) OR the tree net to fall below
its committed 558-official floor — both outside the de-risked bands. (Algebraically
the tree-alone step collapses to `step = step_base + fp32_haircut` — idle and
attention cancel because idle is only "saved" when persist reclaims it — so the
conservative number is driven purely by net_tree-low, fp32-haircut-high, and
τ=0.96.)

### Anti-compounding map (pairwise marginal, on the TREE base)

| pair | joint gain | additive null | kind |
|---|---:|---:|---|
| splitk × persist | +5.71% | +5.56% | multiplicative (compound) |
| splitk × lk | +4.55% | +4.52% | multiplicative (compound) |
| persist × lk | +2.66% | +2.65% | multiplicative (compound) |

The three non-tree levers act on *different* slices (GEMM / idle / E[T]) so they
**compound multiplicatively with each other** — none fights another. The fight is
**tree vs each lever**:

### Tree × lever dilution (standalone gain vs marginal-on-tree)

| lever | standalone | on-tree | dilution | mechanism |
|---|---:|---:|---:|---|
| splitk | +4.33% | +3.72% | **0.86** | pure geometric (bigger step base shrinks the relative GEMM saving) |
| persist | +3.09% | +1.85% | **0.60** | geometric × `a_tree_hide` (tree hides 30% of reclaimable idle) — the genuine #97-vs-tree anti-compound |
| lk | +1.00% | +0.80% | **0.80** | partial-overlap haircut (tree already harvests promoted rank-2, #88) |

This is the headline economic finding: **every lever is worth less once the tree
lands.** SplitK's dilution is benign geometry (an attribution artefact, not a real
fight). Persistent-kernel #97 is the one that *mechanistically* fights the tree —
diluted to 0.60 because the tree's longer M=32 compute hides part of the
reclaimable idle. LK #95 dilutes to 0.80 from the rank-2 partial overlap.

## 5. Gate and recommendation

| Gate input | Value |
|---|---|
| `composed_official_tps` full stack (primary) | **600.0** [531.6, 713.7] |
| `tree_alone_official` | **563.1** [518.0, 581.0] |
| `min_levers_to_clear_500` conservative (test) | **1 — `['tree']`** (518.0) |
| anti-compound map | non-tree pairs compound; tree dilutes persist 0.60 / lk 0.80 / splitk 0.86 |
| GREEN tree-clears-500-conservative / AMBER 1-lever / RED straddles | **GREEN** |

**Recommendation — sequence the 500-path tree-first:**

1. **Land the tree (land #71) FIRST.** It is the only lever that clears 500
   standalone at the conservative corner (518.0) and it does the heavy lifting
   (central 563.1). Everything downstream is margin.
2. **The other three levers are INSURANCE, not requirements.** Prioritise them by
   *marginal-on-tree* value, not standalone value: **SplitK #84 first** (least
   diluted, +3.72% on-tree, and it is real GEMM-BW saving that survives the
   conservative corner), then **LK #95** (+0.80% on-tree, already merged — free),
   then **persistent-kernel #97 last** (most diluted at +1.85% on-tree and ≈0 at
   the conservative corner — its value is conditional on the idle slice being
   non-empty, which the #65/#94 GPU-bound prior says is unlikely).
3. **The fp32-star-attn requirement (wirbel #98) is already priced in** via the
   haircut band and does not threaten the GREEN verdict.

The composed stack gives a comfortable buffer above 500 (central 600.0) so the
team can absorb a partial miss on any one pending gate and still clear the target.

## 6. Public-evidence note (launch-isolated)

All inputs are committed advisor-branch state or the PR #100 body — no inspection
of other students' unmerged branches.

- **Frontier** `fa2sw-precache-splitkv-linear-mtp-k7` = **481.53 official /
  454.338 local** (ratio 1.0599), E[T]=3.844 — leaderboard + lawine #52/#90.
  Folded into `K_cal`.
- **denken #83/#85 (MERGED):** DP-optimal M=32 tree, net +18.2% decode; verify-GEMM
  FLAT M≤32; attention 1.06× under split-KV; tree non-GEMM overhead
  (`report_tree_nongemm_overhead.md`).
- **fern #92 (MERGED):** realised tree E[T]=5.207, 558–581 official envelope —
  the `net_tree` band.
- **fern #95 (MERGED):** LK +1.0–2.4% E[T], prediction channel only, re-rank
  CLOSED (`drafter_accept_objective_gate.py`) — the `lk_mult` band.
- **fern #80 (MERGED):** native-acceptance machinery; LK central near its floor.
- **fern #88 (MERGED):** tree root-to-leaf already harvests rank-2 sibling mass —
  the LK partial-overlap haircut on the tree.
- **denken #65 / #94 (committed):** decode 99.41% GPU-bound; A10G bus serialises —
  the LOW prior on the persistent-kernel idle slice.
- **Pending gates carried as BANDS, not facts:** persistent-kernel idle (#97),
  fp32-star-attn haircut (#98), local→official τ (#99).

## 7. Reproduce

```bash
cd target/
python scripts/profiler/lever_composition.py --wandb \
  --wandb-name "fern/lever-composition-economics" \
  --wandb-group "lever-composition-economics"
# CPU-only, no GPU; ~seconds; peak RSS ~27 MB
# writes research/spec_cost_model/lever_composition_results.json
```

- **W&B run:** `ncseu3ar` — wandb-applied-ai-team/gemma-challenge-senpai, group
  `lever-composition-economics`.
- **Peak memory:** ~27 MB RSS (CPU; zero GPU).
- **No serving run / no `summary.json` / no HF Job** — CPU modelling gate;
  greedy token-identity untouched by construction.
