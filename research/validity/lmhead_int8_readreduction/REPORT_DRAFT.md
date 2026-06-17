STUDENT stark:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"analysis_only":true,"official_tps":0,"wandb_run_ids":["h5tc43wr"],"primary_metric":{"name":"safe_head_tps_spec_served","value":276.52},"test_metric":{"name":"int8_head_teacher_forced_ppl","value":2.0063}}

## Results — Cheaper identity-safe lm_head: int8-per-channel READ-cut under the OPERATIVE int4-referenced identity

**Headline:** an int8-per-channel full-262k lm_head is **operative-identity-safe** against the served int4 `base_fullhead` (**0 confident argmax flips** at the #429 operative band across 75,344 decoded positions; teacher-forced PPL neutral) and projects to **276.52 TPS** in the MTP-K7 spec-served frame (**+23.83** over the 252.69 anchor). It is a **quality-safe speed lever, NOT a ship-reacher** — even a fully-free head ceilings at 311.27 (lawine #591), 99.34 short of the 375.857 prune-ship. **NO FIRE** (this card is `analysis_only=true`, `official_tps=0`; an int8 head would still need its own approval issue + a downstream MMLU/GPQA/AIME/GSM8K re-cert, since int8 logits can shift answers even when the greedy argmax is operatively preserved).

`operative_safe_head_read_reduction_exists = True` · `int8_head_identity_safe = True` · `safe_head_tps = 276.52` · `int8_head_is_ship_reacher = False`

### Measurement (local A10G, profiling only — NO served-file change)
Captured the lm_head-input hidden state `h` at every greedy-decoded position from the **served INT4 QAT body** (`run_compressed=False` → dense bf16 forward; `min_tokens=8` EOS-guard, #541) over the official-128 (operative/flip-rich) + multilingual/code held-out corpora, reusing land #556's validated `decode_capture`/`census_precisions` apparatus. On the SAME `h` (fp32 accum + softcap 30): `bf16-head argmax(h)` **is** the served `base_fullhead` emission by construction (the operative #407 int4-referenced reference); compared `int8-head argmax(h)` against it. A flip is `bf16_argmax != int8_argmax`; each flip is classified CONFIDENT (bf16 top1–top2 margin > 0.125 operative band) vs near-tie (≤ band, PPL-neutral coin-flip). `fp8_e4m3` is censused alongside as the 1-byte cross-check.

| metric | int8 head (full 262k) | fp8_e4m3 head (1-byte cross-check) |
|---|---|---|
| **operative confident flips @ band 0.125 (official+heldout)** | **0** | **4** (official) — NOT operative-safe |
| strict any-flip rate — official (int4-body h) | 0.00122 (30/24576) | 0.00342 (84/24576) |
| strict any-flip rate — heldout (int4-body h) | 1.97e-4 (10/50768) | 6.70e-4 (34/50768) |
| max flip margin (official, softcap logit units) | 0.0855 (≪ 0.125 band) | 0.168 (> band → 4 confident) |
| ref reproduction rate (official / heldout) | 0.99422 / 0.99858 | — |

- **PPL (teacher-forced, 128 seqs, int4-body h):** bf16 head **2.0063** / int8 head **2.0063** (Δ **+3.6e-5**) — within the 2.42 gate; the bf16-head pass reproduces the 2.0057 `base_fullhead` anchor (Δ 0.0006), validating the apparatus.
- **Free-running greedy identity (24 prompts × 128 tok):** **0.875** (21/24 identical), 3 divergent, median first-divergence index **53** (late, near-tie cascade — not early corruption).
- **M=1 head-read microbench:** bf16 full-262k **2.79 ms** / half-vocab GEMV **1.40 ms** / ratio **0.503** (~0.5, read-bound byte-linear). Head bytes: bf16 1.25 GiB → int8 0.625 GiB.

### TPS frame & ceiling (advisor points 1 + 3)
- **Anchor frame corrected.** 252.69 is the **MTP-K7 spec-ON SERVED** rate (wirbel #553; triply re-confirmed lawine #591 / denken #592 / fern #587) — **not** spec-OFF. The PR-body "Recipe: spec-OFF" label was wrong; spec-OFF AR M=1 = 97.0 (#569) is a different (lower) frame. `safe_head_tps` 276.52 is the land-#556 spec-served projection (`int8_tps_if_safe`), whose bf16 baseline (252.31) coincides with the 252.69 anchor to within σ_hw, so **+23.83 is a like-for-like spec-served gain** and agrees with lawine's relayed `free_head_int8 ≈ 276.8`.
- **Hard ceiling 311.27.** lawine #591 (`b001enxl`) decomposed the served cycle: the lm_head is only **18.3%** (2.776 ms); the int4 **BODY dominates at 44.4%** (6.728 ms). So even a **fully-free** head reaches only **311.27 TPS** (= #569 decode floor). int8 (276.52) sits **34.75 below** that ceiling and **99.34 short of the 375.857 prune-ship** → `int8_head_is_ship_reacher = False`. This is a quality-safe rung on the speed frontier with a hard head-only ceiling at ~311.

### #556 reconciliation (advisor point 2 — auditable)
**Pinned (read from `int4_head_strict_identity.py` + its results JSON):** land #556 (run `uipo4rxv`) **did** compute an int8-head argmax flip rate, but (a) on `h` captured from the **BF16 body** (`int4_head_strict_identity.py:603-604`, `_BF16_MODEL_DIR`, dtype bf16), and (b) under a **strict any-flip** lens — it set `strict_safe_head_precision="bf16"`, `strict_safe_head_lever_exists=false` (any flip rejected → int8 declared strict-UNSAFE). It did **not** capture from the served int4 body, nor apply the operative confident-flip lens. So my argmax-match-on-the-served-int4-body, under the operative lens, is the genuinely novel number.

| | #556 (`uipo4rxv`) | this card (#593) |
|---|---|---|
| reference body for `h` | **bf16** | **int4 QAT (served)** |
| lens | strict any-flip | **operative confident-flip @ 0.125** |
| int8 official flip rate | 0.00127 (82/64533) | 0.00122 (30/24576) |
| int8 heldout flip rate | 2.96e-5 (3/101222) | 1.97e-4 (10/50768) |
| int8 verdict | **strict-UNSAFE** | **operative-SAFE (0 confident)** |
| `int8_tps_if_safe` | 276.52 | 276.52 (same model) |

The **official** strict rates agree across bodies (0.00127 vs 0.00122 ≈ 0.12%), cross-validating the apparatus; the heldout rates are both O(1e-4) (rare-event counting noise; mine marginally higher, consistent with the int4 body's hidden states being slightly noisier than bf16). The operative lens — every int8 flip is a sub-0.125 near-tie → 0 confident — is what reopens int8 as a *safe* lever where #556's strict verdict closed it. Same 276.52 projection.

### top-K / hierarchical-argmax gather (instruction 2, analysis)
No obviously operative-safe top-K gather beats int8's 0.5 read-fraction: the only cheap static variant (12k freq shortlist) **is** the shipped prune (375.857) and collapses AIME/quality (no candidate-set guarantee → confident flips); an exact hierarchical cluster-routing argmax that still reads all rows yields NO HBM saving (a direction-blind Cauchy–Schwarz norm bound rarely prunes enough rows to beat 0.5); a low-rank-sketch-then-verify *could* go sub-0.5 read but its operative-safety hinges on an **unproven** shortlist-contains-argmax guarantee (a follow-up recall/miss-margin card). int8 is the concrete realized read-cut: reads all 262k rows at 1 byte = 0.5 fraction, no shortlist guarantee needed.

### What happened — honest analysis
The int8-per-channel head **preserves the greedy argmax at operative grade**: of 40 total int8 flips over 75,344 positions (30 official + 10 heldout), **every one is a sub-0.125 near-tie** (max margin 0.0855), so the forbidden-event count (#429 confident argmax change) is **0**, and teacher-forced PPL moves by Δ+3.6e-5. Importantly the operative lens is **not** a rubber stamp: `fp8_e4m3` (same 1-byte budget) shows **4 confident flips** (max margin 0.168 > band) and is therefore NOT operative-safe — int8-per-row is specifically the safe 1-byte precision. The **strict** byte-precision story is unchanged from #556 (~0.12% official flips) — int8 is **not** bit-identical, and free-running greedy sequences diverge in 3/24 prompts (median first divergence ~token 53), but every divergence is a late near-tie cascade, not confident corruption (PPL-neutral). Free-running bit-identity is the strict lens the served int4 body *itself* fails vs bf16 (10.9%/pos, #585); the live #319/#407 contract is the operative confident-flip lens, which int8 passes. The microbench confirms the head GEMV is HBM-read-bound and byte-linear (half/full ratio 0.503 ≈ 0.5), so a *fused* int8 GEMV (Marlin-int8) realizes the ~2× head-read cut underlying the 276.52 projection. Net: a real, quality-safe spec-served lever (+23.83), but bounded by the 311.27 free-head ceiling — **not** a ship-path.

### Reproduce
```
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python3 research/validity/lmhead_int8_readreduction/lmhead_int8_readreduction.py --gpu
```
- **Peak GPU:** 20.5 GiB (int4 body → dense bf16; head census after model free). **Elapsed:** ~1573 s (~26 min).
- **W&B:** run [`h5tc43wr`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/h5tc43wr), group `lmhead-int8-readreduction`. `analysis_only=true`, `official_tps=0` — **no HF Job, no submission, no served-file change**, so there is no `summary.json`/`run_prefix` (local diagnostic only).

### Public evidence used
- lawine #591 (`b001enxl`) served-cycle decomposition → lm_head = 18.3% / int4 body = 44.4% → 311.27 free-head ceiling, `free_head_int8 ≈ 276.8` (advisor relay 2026-06-17).
- land #556 (`uipo4rxv`) head census → int8 strict-UNSAFE on bf16-body, `int8_tps_if_safe=276.52`, `strict_safe_head_precision="bf16"` (reconciled above; numbers read from the #556 results JSON).
- wirbel #553 (`83jiwjr9`) 252.69 spec-served anchor / PPL 2.0057; #569 decode floor 311.27 & AR M=1 = 97.0; wirbel #585 (`2u44yaa1`) operative #407 identity + int4-body 10.9% bf16-strict flip.

### Suggested follow-ups
1. **Fused int8 lm_head GEMV (Marlin-int8) on the served stack** to *realize* the 276.52 — eager has no fused int8 GEMV (the microbench's `int8_dequant_matmul` = 29.1 ms materializes a bf16 weight and is an upper bound, not the lever). Would require its own approval issue + downstream MMLU/GPQA/AIME/GSM8K re-cert.
2. **Low-rank-sketch-then-verify recall card** — measure whether a V×r sketch shortlist (r≈256, K≈64) provably contains the argmax (or misses only near-ties) to push read-fraction sub-0.5; the only un-closed path to beat int8's 0.5.
3. Because the head is only 18.3% of the cycle, the **higher-EV** frontier work is the **int4 body** (44.4%), not the head — flag for portfolio prioritization.
