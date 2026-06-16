STUDENT kanna:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"analysis_only":true,"no_hf_job":true,"official_tps":0,"wandb_run_ids":["3piz86i4"],"primary_metric":{"name":"drafter_acceptance_headroom_self_test_passes","value":1},"test_metric":{"name":"drafter_improvement_tps_upside_splitkv442","value":122.43}}

## Results

**Verdict: `go_no_go_drafter_slot = NO-GO-DEFER`.** A drafter-improvement slot is **NOT worth the *immediate* next slot vs the #1 realization-gap ("S-sweep", supply-side)** on a TPS-per-GPU-hour basis — but **direction #3 is NOT tapped**: it holds the single largest *realistic* TPS upside (**+104 / +122 TPS** on surgical-357 / split-KV, crosses 500), and it should be the **#2-priority slot, sequenced *after* #1**, not abandoned.

0-GPU pure-analytic card. **No HF Job, no submission, no served-file change, `official_tps=0`** (challenge PAUSED). Reuses **MEASURED served acceptance** — no fresh serve needed (the shipped drafter's E[T] is already measured directly, see source). W&B `3piz86i4` (group `drafter-acceptance-headroom`). Self-test **24/24**. New file: `research/validity/drafter_acceptance_headroom/`.

### KEY OUTPUTS (required)

| output | value |
|---|---|
| `measured_E_T` (served accepted-tokens/step) | **3.849** (band 3.844–3.849) |
| `measured_E_T_source` | **#289 `accept_calibration`** (W&B `5m17r52s`/`fi34s269`), vLLM `spec_decode_num_accepted_tokens_per_pos` counters on the **shipped** kenyan-duma linear-MTP **K=7** drafter (`DRAFTER_SHA256 ed159e..dd18e`), 128×512 public sharegpt — **reused measured served draw** |
| `tps_per_accepted_token_surgical357` | **97.6 TPS / +1.0 E[T]**  (→ **9.76 TPS per +0.1**) |
| `tps_per_accepted_token_splitkv442` | **114.9 TPS / +1.0 E[T]**  (→ **11.49 TPS per +0.1**) |
| `acceptance_ceiling_E_T` (realistic) | **4.91** (band 4.91–5.30) — vs theoretical-max **8.0** |
| `drafter_improvement_tps_upside` | surgical-357 **+104 TPS** (→480); split-KV **+122 TPS** (→565, crosses 500) |
| `go_no_go_drafter_slot` | **NO-GO-DEFER** (defer behind #1; #3 not tapped, = #2 slot) |

### 1 — Measured served E[T] (and a premise correction)

The shipped **surgical-357** (`submissions/fa2sw_strict_surgical357`) runs `SPECULATIVE_CONFIG {"method":"mtp","num_speculative_tokens":7}` — a **linear-MTP K=7** drafter, `DRAFTER_SHA256 ed159e..dd18e`. That is the **exact** drafter #289 already measured end-to-end: **E[T] = 3.849** (prometheus, num_drafts=17082; server-log cross-check 3.844). Acceptance is a **shared drafter property** — #522 confirms surgical-357 / split-KV / frontier all carry the same SHA + K=7 and are byte-exact greedy-identical, so **E[T] is identical across all three rungs**; they differ only in `t_step`. So I **reused** this measured draw rather than standing up a redundant serve.

⚠️ **Premise correction (decisive):** the PR proposes deriving current E[T] from "measured mean `R_ea`≈0.8877". But `R_ea` from #522 (`w71zjxot`) is a **private/public E[T] *transfer ratio*** (`R_ea = ea_pri/ea_pub`, with `ea_pub`≈4.06 and `ea_pri`≈3.57) — **not** a per-position acceptance. Misreading 0.8877 as a uniform per-position accept would give E[T]=**5.47**, inconsistent with the directly-measured 3.849 (self-test `rea_misread_gives_inconsistent_et`). The honest served E[T] is **3.849 public** (#289) / **≈3.57 private** (#522 `ea_pri`), measured, not inferred. PF≈1.0 (#504 `0urxqwob`) so `TPS = E[T]/t_step` holds cleanly.

### 2 — `dTPS/dE[T]` on each rung (= TPS/E[T] = 1/t_step, t_step invariant to same-K lift)

| rung | TPS | t_step | dTPS/dE[T] | **per +0.1 E[T]** |
|---|---|---|---|---|
| surgical-357 (official `j7qao5e9`) | 375.857 | 10.241 ms | 97.6 / +1.0 | **+9.76 TPS** |
| split-KV 442 (local byte-exact, #519 `kwhylaeg`) | 442.35 | 8.702 ms | 114.9 / +1.0 | **+11.49 TPS** |

Because `dTPS/dE[T] = TPS/E[T]` **rises as t_step falls**, each accepted token is worth **18% more TPS on the faster (split-KV) rung** — a load-bearing sequencing fact (below). *Sensitivity:* anchoring on #522's public draw E[T]=4.06 instead of 3.849 lowers these by ~5% (9.3 / 10.9 per +0.1) — doesn't move the verdict.

### 3 — Acceptance ceiling: realistic vs theoretical-max

- **Theoretical max (K=7):** E[T]=**8.0** (every draft + bonus accepted) — not realistic.
- **Linear-MTP structural cap ≈ 3.845** (denken #119 via #289): the deployed drafter sits **AT its cap** (`deployed_at_linear_cap=True`). A same-topology retrain buys **~0** — confirmed independently by **ubel #399** (`ec7i3z5t`): every no-retrain / no-served-kernel lever (temperature, affine calibration) is a rank-order **no-op**; the d-cov "**must be supplied by a drafter retrain or a tree verify**". So "improve the drafter" = a **topology change (EAGLE-3 class)**, not more epochs.
- **Realistic ceiling ≈ 4.91** — the falsifiable #289 EAGLE-3 target: lift the **deep** positions `a_2..a_7 → 0.91` (flat) while holding `a_1=0.729` (deep-lift is feasible; first-token-alone is ceiling-bound). Optimistic (EAGLE-3 also lifts `a_1→0.80`) → **5.30**. So **ΔE[T] realistic = +1.07** (band +1.07…+1.45).

### 4 — `drafter_improvement_tps_upside` (realistic, not theoretical-max)

`TPS_new = TPS_old · E[T]_new/E[T]_old` (t_step fixed):

| rung | realistic (E[T]→4.91) | band (→5.30) |
|---|---|---|
| surgical-357 | **+104 TPS → 480** | +104…+142 → 480…518 |
| split-KV 442 | **+122 TPS → 565** | +122…+167 → 565…610 |

This is **large** and **crosses 500 on split-KV** — direction #3 is genuinely **not tapped**.

### 5 — GO/NO-GO vs the #1 S-sweep (TPS-per-GPU-hour)

| lever | upside | cost | risk | realized? |
|---|---|---|---|---|
| **#1 realization-gap (supply / "S-sweep")** | 442→457 = **+15** remaining (375.857→457 = +81 to certify) | **~6–8 GPU-h** (reduction-profiling ~4 + cuBLASLt-det 2–4) | **~0** (byte-exact, 0 PPL / 0 greedy-identity) | **442 of ~457 already realized locally** |
| **#3 drafter (demand / EAGLE-3 retrain)** | **+104 / +122** (realistic) | **~15–40+ GPU-h** (train + stack-integration + greedy-identity/PPL re-validation) | delivery-**uncertain** (ceiling is a target); spec-dec keeps PPL safe by construction | unrealized |

**Why NO-GO for the immediate slot, and why #3 is still #2:**
1. **#1 wins TPS-per-GPU-hour now** — cheap, near-certain, byte-exact, and *already 442/457 realized locally* (just needs official cert + the last ~15 TPS).
2. **The levers are multiplicative** (`TPS = E[T]/t_step`: #1 attacks `t_step`, #3 attacks `E[T]`). Banking #1 *first* **raises #3's payoff by 18%** (dTPS/dE[T] is higher on the faster rung). Spending #3 first would *under-price its own output*.
3. **#3 is the larger absolute prize** (+104/+122, the only lane that clears 500 with margin) but it's a **cluster-training slot** with **back-loaded, uncertain delivery**. Right call: **bank #1 (supply) → then spend the EAGLE-3 slot (demand) on the 442+ rung.**

### Command

```
cd target && .venv/bin/python research/validity/drafter_acceptance_headroom/price_drafter_acceptance.py --self-test
cd target && .venv/bin/python research/validity/drafter_acceptance_headroom/price_drafter_acceptance.py \
  --wandb_group drafter-acceptance-headroom --wandb_name kanna/drafter-acceptance-headroom
```

Peak memory: **~0.1 GB** (CPU-only float analytic; no torch/GPU/serve). W&B run **`3piz86i4`**.

### What happened

The hypothesis (#481 dir #3) is that a better drafter = free quality-safe speed. **Priced, it's a real but deferred lever.** The deployed linear-MTP K=7 sits at its **structural acceptance cap (3.845)**, so the upside requires an **EAGLE-3-class retrain** (ubel #399 already proved no cheap lever exists). That retrain's *realistic* ceiling (E[T]≈4.91, the #289 deep-lift target) is worth **+104/+122 TPS** — the only lane that clears 500 with margin — so #3 is **not tapped**. But for the **next** slot, **#1 dominates on TPS-per-GPU-hour** (≈6–8 GPU-h, byte-exact, ~0 risk, already 442/457 realized) **and**, because the two levers multiply, banking #1 first makes every future accepted token **18% more valuable**. The clean answer to the PR's question is therefore **"NO for the immediate slot — #1 dominates — but keep #3 alive as the #2 slot, sequenced after #1."**

### Suggested follow-ups

- **Pin the EAGLE-3 prize before committing a training slot.** The +1.07 E[T] / +0.1286 top-1→top-4 coverage (ubel #399) is a *target*, not a delivered number. A **local GPU profiling card** that loads the deployed MTP drafter and reads per-position **top-8/top-16** coverage on the official 128 would convert the locked ceiling into a *bankable* `realized ΔE[T]` and de-risk the training request — the natural next step before any EAGLE-3 cluster run.
- **When #1 lands, re-price #3 on the realized rung.** dTPS/dE[T] = TPS/E[T] is rung-dependent; recompute the drafter upside on the *certified* post-#1 t_step so the training-request validity argument quotes the right (higher) payoff.
- **Tree-verify as a cheaper demand path.** ubel #399's locked +0.1286 coverage is also harvestable by a depth-1 top-K **tree** (no drafter retrain) — but it pays a verify-M / CUDA-graph-rebuild step-time tax. Composing that tax against the coverage gain is a separate go/no-go that may beat a full EAGLE-3 retrain on GPU-hours.

### Public evidence used

Checked the shared digest (`?as=senpai`). The public a10g frontier is now **~508.6 TPS** (`ff-splitkv-frantic-fawindow-clean-v0-w256`), with **505.9** (`fawindow-w256-knightgemma`) and a cluster at **~489** (`osoi5-…-skv64`, `hayai-ctk48`) just under 500 — **all split-KV MTP K=7 stacks**, and the >500 crossings are driven by **window-attention / split-KV (supply-side, step-time)** levers, **not** by a better-acceptance drafter. This is direct public corroboration of the verdict: the live lever clearing 500 today is **supply-side (the #1 lane)**, while the **drafter-acceptance (demand-side) headroom remains unrealized across the entire public frontier** — consistent with **NO-GO-now / #3-is-the-deferred-#2-slot**. **Extends** (does not reproduce) ubel #399's internal NULL-lever result by pricing the retrain it proved is required; grounded in #289 (`fi34s269`), #522 (`w71zjxot`), #504 (`0urxqwob`).
