# PR #659 Results draft — HELD pending ubel #702 (advisor 01:48Z steer; do NOT submit until #702 terminates)

STUDENT fern:
SENPAI-RESULT: {"terminal":false,"status":"in_progress","pending_arms":true,"analysis_only":true,"official_tps":0,"wandb_run_ids":["nmjvtfov","o0l2921f","pplshmpa","a87uiair","goi70q76","c7otd7l7","mvifnz3a","ky8io79b","ioo2fz2n","nb9upoaa","hsqjcve5","u936qrqz","n1m9d40r","oksetppj","g47nbala"],"primary_metric":{"name":"aime_paired_mcnemar_diff_int8locus_vs_int4","value":0.0633},"test_metric":{"name":"aime_int8onlocus_greedy","value":0.450}}

> HOLD STATE (updated 2026-06-19 04:0xZ): **all PR-required cells now COMPLETE** — decisive 5-seed McNemar locked, N=all uniform-int8 anchor done (`oksetppj` 0.4000), and the GPQA n=198 guard done (`g47nbala` 0.5455, clears). finish_chain2 + disk watchdog exited clean (floor never hit), GPU idle, no data loss. **The ONLY remaining gate is the final LB framing/verdict, held until ubel #702 (proxy→measured on the 48-module activation-salience set) terminates, per advisor #666 relay.** Nothing left to run on my side.

## Results — AIME precision-allocation lower bound (LOCAL, analysis_only, official_tps=0, A10G GPU0, NO HF job)

**Verdict: `AIME_RECOVERABLE_CHEAP` → SURFACE-to-human** (int4-QAT-mandate ruling is the human's; this is a measurement/feasibility probe, not a fire).

### Headline
A **targeted int8 upgrade on the middle third of decoder layers (L14–27, 14 layers, ~1/3 of the body)** recovers int4-body AIME to greedy **0.450** (clears the 0.420 bar), and the recovery LIFT is **statistically real but modest** under the noise-robust paired instrument: paired McNemar int8-on-locus vs int4-N=0 (same 5 sampled seeds, 300 pairs) **diff +0.0633, exact p=0.0248, Wald95 [+0.011, +0.116] (excludes 0), 4/5 seeds int8>int4**. The TPS price is modest: **−9.6% local M=1 decode** (matched sampled pools). bf16-on-the-same-layers recovers no better (0.433) at **−62%**, so int8 strictly dominates the recovery.

> **Cross-validations (2026-06-19):** (1) my int4-N=0 5-seed sampled pool = **0.3467** exactly matches **lawine #693 `GAP_CONFIRMED_REAL`** int4-body 0.3467 → independent harness, same gap. (2) my coarse locus L14–27 **contains L23**, which **ubel #700 `ACTIVATION_LOCALIZED`** ranks as the single largest-impact module (`L23.per_layer_input_gate` = 35% impact-energy) → my layer-block read and #700's 48-module salience map agree at the top, but #700 is far finer (48 modules / 1.35% body-params vs my 118 modules / 14 layers). **LB framing HELD until ubel #702** decides whether to anchor on the salience ranking (REVIVES_CONFIRMED) or treat my mid-third int8 as "the allocation the proxy missed" (PROXY_REFUTED).

### Ranked collapse-locus (Phase A: bf16-on-thirds, int4 elsewhere, greedy AIME)
| third | layers | AIME greedy | tok/s | read |
|---|---|---|---|---|
| first | L0–13 | 0.354 (17/48, soft-capped) | 34.98 | below int4 → NOT the locus |
| **middle** | **L14–27** | **0.4333 (26/60)** | 33.97 | **lone third clearing 0.420 → the locus** |
| last | L28–41 | 0.333 (20/60) | 33.52 | below int4 → NOT the locus |
- int4-everywhere baseline = 0.400; bf16 ceiling = 0.4667. **Localization is WEAK** — the thirds differ by only +2–4/60, all Wilson CIs overlap (n=60 noise ≈ ±8 items). The mid-third edges it but barely above noise. So the locus is "mid-third, weakly."

### N-ladder (int4 everywhere, int8/bf16 on the named layers) — greedy AIME n=60 + local TPS proxy
| cell | upgraded layers | AIME greedy | local TPS (tok/s) | ΔTPS vs int4 |
|---|---|---|---|---|
| N=0 pure int4 | 0 | 0.400 (banked dh0tbwpp, conf. ubel#650 t4limybq) | 88.89 (sampled pool) | baseline |
| **int8-on-locus** | **14 (L14–27, int8)** | **0.450 (27/60)** | 80.35 (sampled pool) | **−9.6%** |
| bf16-on-locus | 14 (L14–27, bf16) | 0.4333 (26/60) | 33.97 | −62% |
| N=all uniform int8 | 42 (int8) | 0.4000 (24/60, oksetppj) / 0.4167 (#646 jz3ojbio) | 67.2 (greedy) | −31% (greedy base 97.73) |
- **All greedy points sit within ±~0.10 (±8/60) of each other** — the greedy ladder is too noisy to read a clean monotone slope or a precise 0.420 crossing (exactly the #646 CI caveat). Note the non-monotone N=all this-harness (0.4000) < int8-on-locus (0.450): a 3-item gap, pure n=60 noise; uniform-int8 *everywhere* equals int4-greedy 0.400 and sits below targeted int8-on-locus, reinforcing that the *allocation* (where, not how-much) is what matters. **The only statistically resolved signal is the paired McNemar below.**

### DECISIVE instrument — paired McNemar (int8-on-locus vs int4-N=0, SAME 5 sampled seeds)
Cancels the sampling penalty (sampled≤greedy) that biases a standalone sampled-vs-0.420 read to fail by construction (ubel #650 precedent). 5 seeds × 60 items = 300 paired outcomes.

| seed | paired | int8 | int4 | delta | (int8-only/int4-only) |
|---|---|---|---|---|---|
| 12345 | 60 | 0.450 | 0.317 | **+0.133** | 11/3 |
| 23456 | 60 | 0.417 | 0.350 | **+0.067** | 7/3 |
| 34567 | 60 | 0.433 | 0.317 | **+0.117** | 11/4 |
| 45678 | 60 | 0.367 | 0.317 | **+0.050** | 7/4 |
| 56789 | 60 | 0.383 | 0.433 | **−0.050** | 6/9 |

- POOLED (300 pairs): int8-on-locus **0.4100** (123/300) vs int4-N=0 **0.3467** (104/300), **paired diff +0.0633**, Wald95 [+0.0111, +0.1155] (excludes 0), **exact McNemar p=0.0248**, chi²cc 4.985, **per-seed 4/5 int8>int4**. RECOVERY_LIFT_SIGNIFICANT_0.05 = True. 2×2: both_ok=81, int8_only=42, int4_only=23, both_wrong=154.
- **Honest p-trajectory:** p sharpened 1→0.057, 2→0.023, 3→0.0034, 4→0.0026 (4-seed read), then **softened to 5→0.0248** when s56789's int4 drew a lucky-high 0.433 (`hsqjcve5`) — the lone negative pair-set. The 5-seed pooled read (+0.0633, p=0.0248) is the honest verdict: significant, CI excludes zero, but the lift is modest, not the sharper 4-seed picture.
- int4-N=0 pool **0.3467** = exact match to lawine #693 `GAP_CONFIRMED_REAL` int4-body 0.3467 (independent cross-validation of the gap).

### GPQA-D guard (greedy n=198, int8-on-locus) — **PASS, no regression**
`mix_int8_L14-27` greedy GPQA-D n=198 = **0.5455 (108/198)**, Wilson95 [0.476, 0.613], 20.9 s/item, extract_fail 0, trunc 0%, peak ~10.5 GiB. W&B **`g47nbala`** (DONE 2026-06-19 04:02Z). **Clears the bar on every basis:** 90%-of-bf16-greedy bar **0.4409** (=0.9×0.4899) at **111% of the bf16 greedy endpoint** (CI 97.1%–125.2%); also above the int4-AR endpoint 0.4798, the bf16 greedy endpoint 0.4899, and the 0.4864 sampled-bf16 bar; even the CI lower bound 0.476 clears the greedy bar. The AIME-recovering allocation does **not** trade away GPQA. Prior: #646 showed GPQA precision-insensitive (int4 0.5000 / int8 0.5657 / bf16 0.4899, all clear), and int8-on-locus is bracketed by int4 and uniform-int8 — both clear — so a pass was near-certain; **verified here.**

### Lower bound
- **`min_N_to_clear_0.420` = ≤14 layers (the mid-third L14–27, int8).** This is the *sufficient* N established; a finer within-third minimum is **not resolvable at n=60 greedy noise**, so 14 layers is the defensible upper bound on the minimum. TPS cost of that bound: **−9.6% local M=1 decode**.

### Commands
```
# locus + ladder + paired seeds, all under group aime-precision-allocation-lb-fern:
cd target && VLLM_BATCH_INVARIANT=1 /usr/bin/python3 \
  research/validity/aime_precision_allocation_lb/eval_mixed.py \
  --body-name <cell> --body-path /workspace/gemma_build/<build> \
  --evals aime|gpqa --mode full --decode greedy|sampled --seed <s> \
  --upgrade-layers <14-27|all|none> --upgrade-precision <int8|bf16|int4> \
  --soft-cap-min 82 --wandb-group aime-precision-allocation-lb-fern
# paired McNemar:
/usr/bin/python3 research/validity/aime_precision_allocation_lb/mcnemar_paired.py 12345 23456 34567 45678 56789
```
- Peak mem: ~19.3/23 GiB (GPU0, conc1 M=1). Disk: jsonl-only writes, KB-scale.

### What happened
- int4's AIME deficit is **weakly localized** to the mid-third (L14–27); upgrading just that third to int8 recovers AIME above 0.420 (greedy 0.450) and the lift is **paired-significant** (McNemar p=0.0248, the sampling-penalty-cancelled instrument — the single greedy +items crossing alone would be noise-dominated).
- **int8 dominates bf16 for recovery:** same layers, equal AIME recovery (0.45 vs 0.433), but int8 costs −9.6% vs bf16 −62% (the bf16 weights lose the fast quant GEMM).
- The recovery is **cheap** (1/3 of layers at int8, −9.6% local decode), NOT near-bf16 — so the quality-recovery axis is **open**, contingent on the mandate ruling.

### Why this is SURFACE, not fire
A mixed-precision (int4+int8) body may not satisfy the challenge's **int4-QAT body mandate**. Whether a 14-layer-int8 body is mandate-compatible is the **human's** call. I am reporting a feasible, priced recovery — not shipping it.

### Suggested follow-ups
1. If the human rules mixed-precision mandate-OK: a finer within-mid-third N-sweep at **higher n** (pooled multi-seed sampled, McNemar per step) to push the lower bound below 14 layers — n=60 greedy can't resolve it.
2. Price int8-on-locus at the **official batched serve point** (this −9.6% is a local M=1 proxy; the official 126.378 is batched — the read-bound fraction differs, see denken #283).
3. QAT a native int8-on-mid-third + int4-elsewhere body (vs post-hoc precision override) to test whether trained mixed-precision recovers more.
