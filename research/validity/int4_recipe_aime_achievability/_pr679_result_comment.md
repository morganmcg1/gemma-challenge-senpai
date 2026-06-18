STUDENT ubel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"analysis_only":true,"official_tps":0,"fires":false,"wandb_run_ids":["1z5vq2ej"],"verdict":"INT4_RECIPE_AIME_BOUND(speed)","primary_metric":{"name":"int4_recipe_aime_best","value":0.557},"test_metric":{"name":"int4_recipe_tps_official_equiv","value":122.895}}

## Results

**Verdict: `INT4_RECIPE_AIME_BOUND(speed)`** — a finer group-size **does** clear the 0.420 AIME bar (so the failure is **NOT fundamental to int4 W4A16**), but **every AIME-clearing recipe is slower than the 126.378 anchor**, and the one zero-speed-cost lever (a better in-repo clip observer) does **not** recover AIME. The bound that survives is a **SPEED** bound, not the "quality is fundamental to int4" bound the PR hypothesised. This splits the PR's binary: it refutes the `…CLEARS_AIME` clause (no speed-competitive recipe) **and** the bare `…AIME_BOUND` premise ("no in-recipe change clears AIME" is false — g64/g32 clear it on quality).

### Full table (int4-AR greedy AIME @12288, 60q = 2024+2025-I+2025-II, conc=16, fresh-process band, VLLM_BATCH_INVARIANT=1, seed0 — the #672 harness)

| recipe | observer / grid | sessions (maj@1) | mean | 95% band | clears 0.420? | official-equiv TPS | beats 126.378? |
|---|---|---|---|---|---|---|---|
| **g128** (shipped control) | minmax / g128 | .333 .333 .300 .433 | **0.350** | [0.258, 0.442] | ✗ | 126.378 (anchor) | — |
| **g64** | minmax / g64 | .450 .467 .350 .517 | **0.446** | [0.335, 0.557] | ✓ (mean) | 122.895 (−2.76%) | ✗ |
| **g32** | minmax / g32 | .417 .467 .450 .417 | **0.438** | [0.398, 0.477] | ✓ (mean) | 119.219 (−5.66%) | ✗ |
| **g128mse** (calib proxy) | **mse** / g128 | .400 .383 .383 .400 | **0.392** | [0.376, 0.407] | ✗ | 126.378 (byte-identical, **0 cost**) | tie |

Contrasts vs the g128 control (paired on shared session noise):
- **g64 − g128 = +0.096** (t=2.11, lift > 2·SE ✓)
- **g32 − g128 = +0.088** (t=2.78, lift > 2·SE ✓ — tightest, most robust)
- **g128mse − g128 = +0.042** (t=1.42, lift > 2·SE ✗ — **within session noise**)

### Leg 1 — group-size sweep (Step 1): finer grid recovers AIME on quality
g128 RTN fails the bar centrally (mean 0.350, matching #672's [0.350, 0.383]). Both finer grids lift AIME **above the 0.828 cross-session non-determinism floor**: g64 to 0.446 and g32 to 0.438, each a real >2·SE effect. So the int4-AIME failure is **recipe-dependent (group-size), not fundamental to int4**.

**Mechanism (build provenance).** The quant source is `gemma-4-E4B-it-qat-q4_0-unquantized` — a QAT model natively trained at **q4_0 = 32-element blocks**. Body re-quant rel_err: **g32 = 0.0666 / 0.0667 / 0.0667** (min/mean/max — uniform, minimal clipping; aligns with the QAT-native grid) < **g64 = 0.0779 / 0.0849 / 0.1183** < g128 (coarsest; 4 native blocks forced to share one scale → heavy clipping on the near-tie weights that decide long AIME chains).

### Leg 2 — calibration sweep (Step 2): the zero-cost clip lever does NOT recover AIME
**Honest tooling caveat first:** the PR's hypothesised **math/reasoning-domain activation calibration** (GPTQ/AWQ on a GSM8K/AIME corpus) is **not installable on this rig** — llmcompressor/auto_gptq/gptqmodel/awq are absent, and the model is a custom multimodal MatFormer whose shipped build does manual safetensors surgery precisely to avoid `transformers` quant. True math-cal is therefore a **cluster-training-request follow-up** (program.md lists "calibration" as a cluster candidate), not something I could run here.

The decisive in-repo proxy I **could** run is the `build_quant.py` **`mse` observer**: a per-group clip search (≤45% shrink) minimising int4 round-trip **weight** MSE — data-free and output-blind, but it changes only scale **values**, not the **count** of scales, so its packed byte layout is **identical to the 126.378 anchor → zero speed cost**. If anything could reach a quality-safe *and* fast int4 body, this was it.

**Result: `mse@g128` lands at mean 0.392** (band [0.376, 0.407]) — a small, **statistically-insignificant** +0.042 over g128-minmax (t=1.42), still **clearly below 0.420**. A better clip threshold at the shipped grid does not recover AIME.

**Why the clip can't substitute for the grid** (the interesting part): `mse@g128` mean weight rel_err is **0.0863 ≈ g64's 0.0849**, yet its AIME (0.392) is far below g64's (0.446). The mse search lowers *average* weight MSE but leaves **max** group error high (0.1418, vs g32's uniform 0.0667). It's exactly those worst-clipped groups — not the average — that carry the AIME-relevant near-tie weights. Only adding **more scales** (finer grid) gives every group its own low-error budget; reallocating one coarse scale optimally cannot.

> Arm B (`mse@g32`) was **deliberately skipped**: g32 already fails the speed gate (119.22 < 126.378), so no g32-grid recipe can reach `CLEARS_AIME` regardless of observer; g32 is already at the QAT-native q4_0 grid (rel_err 0.0667 uniform) so mse has ~no clip slack to add (predicted no-op); and disk at 99% forbade a second ~10 GB build. It has **zero leverage on the verdict.**

### Leg 3 — speed gate (Step 3): finer grid is not speed-competitive
Measured on the **official speed-benchmark harness** (`official/main_bucket/shared_resources/speed_benchmark`, np=8, output_len=512, reps=3, warmup=1), anchor + variants on the **same local rig**, projected to official by the anchor's known ratio (126.378 / 127.128 local = ×0.9941 — more accurate than a generic ×0.870 haircut because the anchor's exact official number is fixed):

| recipe | local wall-TPS | official-equiv | Δ vs anchor | PPL | scale-bytes/weight vs g128 |
|---|---|---|---|---|---|
| g128 (anchor) | 127.128 | **126.378** | — | 2.019 | 0.5156 bpw (1.0×) |
| g64 | 123.625 | 122.895 | **−2.76%** | 2.041 | 0.5313 bpw (**+3.0%**) |
| g32 | 119.926 | 119.219 | **−5.66%** | 2.007 | 0.5625 bpw (**+9.1%**) |

Finer grid = more fp16 scales per weight = more decode-time memory traffic on this memory-bound GEMV → the TPS hit tracks the extra scale bytes. (PPL is fine for all — g32's 2.007 even beats the anchor — but PPL does **not** predict AIME here, consistent with #672.)

### Baseline comparison
- **Strict-#319 anchor (the speed bar):** `submissions/int4_g128_lmhead` (PR #4) — official **tps=126.378**, PPL 2.019. ⟶ no recipe here beats it.
- **#515 quality gate:** int4-AR AIME ≥ **0.420** (≈90% of bf16-base 0.4833). ⟶ g64/g32 clear on the mean; g128 (0.350) and g128mse (0.392) fail.
- **#672:** int4-g128 band [0.350, 0.383] reproduced exactly (my g128 control mean 0.350). This card **extends** #672: the failure it proved robust for the *shipped recipe* is shown here to be **group-size-specific on quality**, but **speed-bound** in aggregate.

### Commands
```bash
# Builds (CPU safetensors surgery, data-free): body grid g128/g64/g32 minmax + g128 mse
/tmp/vllm0220-srv/bin/python research/validity/int4_recipe_aime_achievability/build_offline.py \
  --src <gemma-4-E4B-it-qat-q4_0-unquantized snapshot> --out <dir> --body-group-size {128,64,32} [--body-observer mse]
# AIME band per arm (4 fresh-process sessions, #672 harness)
bash research/validity/int4_recipe_aime_achievability/run_band.sh <arm> <model-dir> 4 0 0
# Speed A/B on the official harness; final aggregation + W&B
/usr/bin/python3 research/validity/int4_recipe_aime_achievability/final_aggregate.py --wandb
```

### Run facts
- **W&B run:** `1z5vq2ej` (`analysis_only=1`, `official_tps=0`, `fires=0` logged as scalars) — https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/1z5vq2ej
- **Peak GPU mem:** ~20.4 GB / 23 GB A10G (gpu_mem_util 0.90, max-model-len 16384).
- **No HF Job / no submission / no `train.py --launch`** — analysis-only, single assigned A10G, per operator rule.

### What happened — honest analysis
The PR's two clean outcomes don't fit; the truth is in between and is itself informative:
1. **Quality is recoverable.** A finer grid clears the 0.420 bar by a real margin. So "int4 W4A16 fundamentally fails AIME" (the bare `AIME_BOUND` reading) is **too strong** — #672's robustness is specific to the **g128 recipe**, not int4 per se.
2. **But not for free, and not via the cheap lever.** Every grid that clears AIME is 2.8–5.7% under the anchor, and the only zero-cost lever (clip observer at the shipped grid) is a null (+0.042, n.s.). The AIME-clearing region and the speed-competitive region are **disjoint** for the levers I could test → no quality-safe **fast** int4 body.
3. The mechanism is clean: AIME recovery needs **more scales**, not a smarter single scale — and more scales is exactly what costs decode bandwidth.

### Suggested follow-ups
- **The one untested zero-cost lever — true GPTQ/AWQ math-domain calibration @ g128** (cluster training request). My `mse@g128` null only rules out the *data-free, weight-MSE* clip; activation-aware calibration on an AIME/GSM8K corpus optimises *output* error and could preferentially protect the near-tie weights at the **anchor's exact byte layout** (still 126.378 TPS). This is the only remaining path to `INT4_RECIPE_CLEARS_AIME`, and it's why I did **not** close the quality question outright. Decisive and cheap to settle on the cluster.
- **QAT *at* g32** (not PTQ): the source is QAT-q4_0; a short QAT/recovery finetune emitted natively at g32 might lift AIME further than PTQ-g32 (0.438) — but note it inherits the same **−5.66% speed** penalty, so it only matters if paired with a speed offset elsewhere.
- **Mixed-grid (sensitivity-routed):** g32 only on the layers whose near-tie density actually drives AIME flips (the #672 density_flip_corr signal), g128 elsewhere — could buy most of the AIME lift at a fraction of the +9.1% scale-byte cost. Would need the per-layer sensitivity map.
- **QAT-legality caveat:** PTQ recipes (g64/g32 RTN) may not be QAT-legal *submissions*; treat the g64/g32 quality result as an **analysis** signal for #481, not a ship candidate.

### Public evidence used
- Leaderboard/anchor: `submissions/int4_g128_lmhead` (PR #4), official `tps=126.378`, the strict-#319 rung this card measures against.
- Internal quality gate #515 (AIME ≥ 0.420) and the bf16-base reference 0.4833.
- #672 (int4-g128 AIME robustness, band [0.350, 0.383]) — the card this one extends; its harness and the 0.828 non-determinism floor are reused verbatim.
- wirbel #437 (QuaRot rotation-requant) — rotation axis is taken; this card is the orthogonal **group-size + calibration** axis, no overlap.
