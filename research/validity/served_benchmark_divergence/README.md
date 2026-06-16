<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->
# Served benchmark divergence — audit + pin the head-prune collapse (PR #529, denken)

**One-line verdict (filled by the run):** see `## Results` below and the `verdict` field in
`served_benchmark_divergence.json`.

This card reconciles the two results that looked contradictory:

- **ubel #511 / fern #514** — the **live board ship** (`fa2sw_strict_surgical357`, official
  **375.857 TPS**) *collapses* on the organizer's downstream evals: MMLU-Pro 0.668→0.274,
  GPQA-Diamond 0.444→0.232, AIME 0.400→0.033.
- **denken #520** — the *same* ship certified **sampled-quality-neutral** (TV=KL=0) under the
  official sampling config.

They don't conflict — together they localize the collapse to the **12k head-prune** on the broad
benchmark distribution, invisible to #520's geometry. This card *measures* that, end to end.

## The audit headline (the team needs this)

`served_ship_has_head_prune = True`, served keepset **K = 12288** unique vocab ids over the full
**262144** vocab — **the live 375.857 board submission inherits the benchmark collapse.**

Read directly from the live submission (`submissions/fa2sw_strict_surgical357/`):

| Evidence | File | What it proves |
|---|---|---|
| `LM_HEAD_PRUNE=1`, `LM_HEAD_PRUNE_REQUIRE=1`, `LM_HEAD_KEEPSET_BUCKET=…/int4-pck04c-12k` | `manifest.json` | the served job **requires** the 12k prune to load |
| `_prune_lm_head_rows` = pure `torch.index_select` row-slice of the int4 packed `lm_head` (no re-quant) | `serve.py` | the **kept** rows' logits are **bit-identical** to base |
| `_scatter_to_full_vocab` scatters the K kept logits to full vocab with **−inf** at every non-kept id | `serve_patch_pck04.py` | non-keepset tokens get probability **exactly 0** |
| served keepset `pck04_keepset.json`: `pruned_vocab_K=12288`, `full_vocab=262144` | `…/int4-pck04c-12k` bucket (committed here as `pck04_keepset_12k.json`) | K matches what **ubel #511 loaded** |

So **the served ship's next-token distribution == the true base distribution masked to the 12288
keepset** (kept logits unchanged; everything else −inf). The only difference between base and ship
is the keepset mask = exactly the head-prune.

### Two prune layers (advisor fern #531) — and where each is attributed

The live ship reaches its 12288 keepset through **two** stacked prunes, which this card labels
explicitly so the collapse is pinned to the right layer:

| layer | where | substrate | keepset |
|---|---|---|---|
| **bake** 262144 → 16384 | baked into `WEIGHTS_BUCKET=…/osoi5-v0-baked` (`lm_head=[16384,320]`) | 16k baked head | `osoi5-v0-baked/pck04_keepset.json` (committed here as `pck04_keepset_16k_baked.json`) |
| **serve** 16384 → 12288 | in-job at `serve.py:_lmhead_prune_phase` (`LM_HEAD_PRUNE=1`, line 671), pure `index_select` from the 16k head | 12k served head | `…/int4-pck04c-12k` (committed here as `pck04_keepset_12k.json`) |

`serve.py:613-618` **enforces** `12288 ⊂ 16384 ⊂ 262144`, so the two prunes form a clean nested
chain. We measure all three arms from **one** native-262k forward and split the collapse:

```
base│ship  (262144 → 12288)  = combined collapse  (the headline)
base│bake  (262144 → 16384)  = BAKE-layer divergence   (only fern #535's native-262k re-bake fixes)
bake│ship  (16384 → 12288)   = SERVE-layer divergence  (a 12k→16k serve-keepset widen would recover)
```

The base/no-prune reference is the **native-262144 int4 head**
(`gemma-4-E4B-it-qat-w4a16-ct`, #520's quality-neutral int4 base proxy — the same native head fern
#535 stands up on base-int4). `killed_at_serve_rate` (a wanted token the 16k bake keeps but the
12k serve prune drops) is the **actionable** number: it is exactly the quality a 12k→16k serve
widen would recover, bridging kanna #528's static coverage gap and ubel #527's keepset-width Pareto.

## Method — an emulation that *isolates* the head-prune exactly

Because the prune is a verified pure row-slice + −inf scatter, we serve **one** int4 full-head
substrate (`gemma-4-E4B-it-qat-w4a16-ct`, full 262144 `lm_head` — #520's quality-neutral int4
base proxy) and derive **both arms from the same forward pass** at every decode position:

```
p_base = gen_config( top-K raw logprobs )                    # full head
p_ship = gen_config( top-K raw logprobs masked to keepset )  # 12k head (−inf off-keepset)
```

`gen_config` is the **official sampling transform** (`generation_config.json`: T=1.0, top_k=64,
top_p=0.95) — the exact `gen_config_dist` from the merged #520 harness. Because both arms share
the identical forward, the **only** thing that differs is the keepset mask. This is strictly
cleaner than serving two separately-quantized checkpoints (which would confound the prune with
quant noise): here the measured divergence is **100% head-prune**.

**Exactness (not an approximation).** `p_ship` from the captured top-K equals the *true* served
ship distribution iff the uncaptured keepset tail can't enter the top_p nucleus. We bound that
tail per position (`n_uncaptured_keepset × smallest captured prob`) and flag a position
**provably exact** when the bound is below the `(1−top_p)` slack of the captured keepset mass
(or ≥64 keepset tokens are already captured). At `topk=256`, `ship_nucleus_safe_rate = 0.99976`
(41952/41962 positions provably exact) and `n_ship_unresolved = 0` (no position has zero keepset
mass in the captured top-K). The 10 residual positions (0.024%) are bounded-not-certified — their
captured keepset tail *might* nudge the nucleus by an amount below the 256th captured prob — but
they are never disjoint-support, so their TV is mid-range either way and **cannot** move the
pooled `max TV = 1.0` or the `wanted_token_killed_rate` (both are decided by argmax keepset
membership, which is exact regardless of the tail). The harness self-reports these so the rigor
is auditable, not assumed.

**Distribution & trajectory.** Prompts are a broad **benchmark proxy** — MMLU (18 cached STEM
subjects) + Hendrycks MATH + GSM8K — rendered through the gemma chat template, eliciting **long
CoT** (`n_new=320`, *not* the n_new=24 PPL window #520 lived on). The decode trajectory is the
**base** model's own sampled CoT under the official config (fixed seed → reproducible); at each
position along it we score `TV(p_base, p_ship)` and `KL(p_base‖p_ship)`. Per-step divergence along
the base trajectory is a **lower bound** on the served quality gap — the real ship additionally
*drifts* once it is forced off a killed token.

> Proxy note: the organizer scores MMLU-Pro / GPQA-Diamond / AIME, which are not pullable offline
> on the pod. MMLU-STEM / MATH / GSM8K are the same *kind* of distribution (reasoning MCQ +
> competition math + word-problem CoT); the mechanism (out-of-keepset answer tokens → −inf) is
> distribution-independent, and the per-task split shows where it bites hardest.

## The blind-spot contrast (the proof)

Three geometries, all from the same capture:

| geometry | arms | expected | what it means |
|---|---|---|---|
| `selftest` | full-head vs full-head | TV = KL = **0** | control: the transform pipeline is a pure, deterministic function |
| `prune²` | 12k-head vs 12k-head | TV = KL = **0** | **#520's geometry** — both arms behind the 12k head, so the prune is **common-mode and cancels exactly**. #520's base substrate *was* the pruned osoi5, so its A/B lived in this column → structurally blind. |
| `base│ship` | full-head vs 12k-head | **blow-up** | the column #520 never measured |

`TV(prune²)=0 ≪ TV(base│ship)~1` **is** the proof that the collapse is the head-prune and that any
prune-vs-prune (or ship-vs-ship teacher-forced) measurement cannot see it.

## Files

| file | what |
|---|---|
| `prepare_prompts.py` | builds the benchmark-proxy prompt set (MMLU-STEM + MATH + GSM8K) via the gemma chat template → `prompts.jsonl` (pre-tokenized `context_token_ids`). Run under `uv run --with datasets --with transformers`. |
| `served_benchmark_divergence.py` | the harness: GPU capture (base CoT + per-position raw top-K) → derive base│ship (and base│bake + bake│ship when the 16k keepset is present) → per-position TV/KL, `wanted_token_killed_rate`, per-layer kill split, answer-vs-filler split, blind-spot contrast, self-test, W&B. |
| `pck04_keepset_12k.json` | the **served** 12288-id keepset (committed for reproducibility; K=12288, full_vocab=262144). |
| `pck04_keepset_16k_baked.json` | the **bake** 16384-id keepset (`osoi5-v0-baked` source keepset; K=16384, full_vocab=262144). Enables the bake-vs-serve layer split; `12288 ⊂ 16384` verified. |
| `prompts.jsonl` | the built prompt set (135 prompts: 45 each MMLU/MATH/GSM8K). |
| `served_benchmark_divergence.json` | the full analysis payload (compose + raw gpu block). |

## Reproduce

```bash
cd target

# 0) (optional) rebuild the benchmark-proxy prompts
uv run --with datasets --with transformers python \
  research/validity/served_benchmark_divergence/prepare_prompts.py \
  --out research/validity/served_benchmark_divergence/prompts.jsonl --n-per-domain 45

# 1) GPU-free self-test (control + served-keepset check)
.venv/bin/python research/validity/served_benchmark_divergence/served_benchmark_divergence.py --self-test

# 2) full divergence run (logs W&B). The committed 16k bake keepset is auto-detected, so this is
#    the 3-arm layered run (base│ship + base│bake + bake│ship). Add --no-bake for base│ship only.
.venv/bin/python research/validity/served_benchmark_divergence/served_benchmark_divergence.py \
  --prompts research/validity/served_benchmark_divergence/prompts.jsonl \
  --n-new 320 --topk 256 \
  --wandb_name denken/served-benchmark-divergence --wandb_group served-benchmark-divergence
```

**Scope:** LOCAL profiling card. `analysis_only=true`, `official_tps=0`, no served-file change,
no HF Job, no `train.py --launch`, no submission. The served config and the baseline are
UNCHANGED; the served distribution is **measured**, never altered. GPU phase runs under the
submission server venv (vLLM + Marlin), `CUDA_VISIBLE_DEVICES=0`.

## Results

**Verdict — DIVERGES; cause pinned to the head-prune; dominated by the bake (262k→16k) layer.**
W&B [`86ro6fjv`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/86ro6fjv) ·
`served_benchmark_divergence.json`. 135 benchmark-proxy prompts (MMLU-STEM + MATH + GSM8K),
**41 962** decode positions along base's own long CoT under the official sampling config
(T=1.0, top_k=64, top_p=0.95; seed 1234). Self-test **7/7** (full run) + **15/15** (GPU-free);
NaN-clean; `ship_nucleus_safe_rate=0.99976`, `n_ship_unresolved=0`; peak **20.5 GiB**; ~36 min.

### 1. Audit headline (verified against the live submission, not assumed)
`served_ship_has_head_prune = True`; served keepset **K = 12288** over full **262144** vocab.
**The live 375.857 TPS board submission inherits the collapse** — `manifest.json` sets
`LM_HEAD_PRUNE=1` + `LM_HEAD_PRUNE_REQUIRE=1`, so the job refuses to start without the prune.
Mechanism (read from `serve.py:569-665` / `serve_patch_pck04.py:113-165`): the prune is a pure
`torch.index_select` row-slice of the int4-packed `lm_head` (kept rows **bit-identical** to base)
then a scatter into a `[M, 262144]` buffer pre-filled with **−inf** (non-kept prob exactly 0).
Both committed keepsets verified **exact** against the live checkpoints: 12k == served
`int4-pck04c-12k` (symdiff 0), 16k == `osoi5-v0-baked` (symdiff 0); `12k ⊂ 16k ⊂ 262144`.

### 2. Per-position base│ship divergence (the benchmark distribution)
| stat | pooled | gsm8k | math | mmlu |
|---|---|---|---|---|
| n positions | 41962 | 13267 | 14400 | 14295 |
| mean TV | **0.0661** | 0.0935 | 0.0234 | 0.0838 |
| p99 TV | 1.000 | 1.000 | 1.000 | 1.000 |
| max TV | **1.000** | 1.000 | 1.000 | 1.000 |

max KL **100** (finite cap; disjoint-support positions are genuinely +∞). TV histogram is
**bimodal**: 37179 positions (88.6%) untouched at TV=0, but **2658 (6.33%) blow up** (TV>0.5) and
2120 (5.1%) sit in TV∈[0.9,1] — the "mostly fine until it catastrophically kills the wanted
token" signature a hard −inf mask produces. GSM8K/MMLU bite hardest; MATH least (its answers are
mostly digits — see §3).

### 3. The collapse mechanism + localization
- `wanted_token_killed_rate = 0.0652` — at 6.5% of positions the served ship assigns the
  **base-argmax token −inf**; it cannot emit the token base wanted.
- **Answer-bearing vs filler:** answer-bearing (content-word) mean TV **0.1036** / killed
  **10.3%** vs filler mean TV **0.0053** / killed **0.38%** → a **~20× (mean TV) / ~27× (kill
  rate)** concentration on answer tokens. Sub-split: alpha content words mean TV **0.125** / killed
  **12.5%**; **digit tokens ≈ untouched (TV 0.0003, 0% killed)** — the keepset keeps numerals but
  drops content words. The hole is a *vocabulary* hole on reasoning/answer words, not numerical.
- Raw high-TV reads (mechanism color): base wants ` isomorphic`(p=1.0)→ship forced ` necessarily`;
  ` isomorphism`→` group`; ` Klein`→` cyclic`; ` subgroup`→` direct`; ` bone`→` specific`;
  ` articul`→` connects`. The ship emits a fluent, *wrong* neighbor because the right token is −inf.

### 4. Blind-spot contrast (the proof)
| geometry | max TV | meaning |
|---|---|---|
| full-head vs full-head (control) | **0.0** | the transform pipeline is a pure deterministic function |
| **prune² (12k vs 12k) — #520's geometry** | **0.0** | prune is common-mode, cancels exactly → **structurally blind** |
| base (full) vs ship (12k) | **1.0** | the column #520 never measured |

`TV(prune²)=0 ≪ TV(base│ship)=1.0` **is** the proof: any prune-vs-prune (or ship-vs-ship
teacher-forced) measurement — #520, the operative-1.0 census — *cannot* see this collapse, because
both arms carry the 12k head and the mask cancels.

### 5. Layer attribution (advisor fern #531) — which prune layer owns it
Both substrates verified exact against the live checkpoints; the kill decomposition is **exactly
additive** (2315 + 419 = 2734 total kills).
| layer | substrate cut | killed rate | share of kills | mean TV | what recovers it |
|---|---|---|---|---|---|
| **bake** | 262144 → 16384 (`osoi5-v0-baked`) | **0.0552** (2315) | **84.7%** | 0.0564 | only fern #535's native-262k re-bake |
| **serve** | 16384 → 12288 (`int4-pck04c-12k`) | **0.0100** (419) | **15.3%** | 0.0142 | a 12k→16k serve-keepset widen |

**The bake layer owns the collapse** (≈85% of kills). The serve-time 16k→12k prune adds only 15% —
`serve_widen_recoverable_kill_rate = 0.0100` is the actionable quick-win (the slice ubel #527's
keepset-width Pareto / kanna #528's static gap would target), but widening the *served* keepset
alone leaves the 85% bake-layer hole untouched. **The 262k→16k bake is the dominant cause; a serve
widen is a partial fix, not the fix.**

### One-line verdict
The served ship's token distribution **diverges sharply** from base on the benchmark distribution
(max TV 1.0, mean 0.066; 6.5% of positions −inf the base-wanted token, ~27× concentrated on
answer-bearing words while digits are untouched); the cause is **the `lm_head` prune** (the prune²
blind-spot at TV=0 proves #520's neutrality structurally couldn't see it); and it is **dominated by
the 262k→16k bake layer (≈85% of kills)**, with the serve-time 16k→12k prune contributing the
recoverable 15%. The live 375.857 TPS board submission inherits all of it.
