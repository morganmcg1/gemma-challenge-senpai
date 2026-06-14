STUDENT wirbel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["z6wi4z4v","6wr8r2y0"],"primary_metric":{"name":"drafter_rank2_coverage","value":0.4165},"test_metric":{"name":"reprice_M32_proj_tps","value":521.64}}

## Results — ρ measured, the last borrowed input to the tree gain is now pinned

**Headline:** ρ₂ = **0.4165** (local, full greedy path) vs byteshark's external **0.4130** — agree to **0.85%**, inside lawine #72's ±4.4% floor. Re-pricing #76's tree with the measured ρ lands the M=32 gain at **+21.8% central** (521.6 TPS), **above** the +18.7% borrowed-0.565 central and toward the +25% ceiling. **M=32 dominates M=16, full max-branch-4 is justified, fail-fast NOT triggered. Tree beats deployed linear — land #71 should build.**

### 1. Measured rank-of-true-token distribution (run `z6wi4z4v`)
Read-only top-4 probe on a **scratch copy** of `fa2sw_precache_kenyan` (served files untouched, greedy identity preserved). At each draft position, following the verifier's true greedy continuation, recorded the rank at which the true greedy token appears in the drafter's top-4.

- **16,524 records, align_bad = 0** (100% byte-identity: unbiased `logits.topk` rank-1 == deployed fused argmax on every record) · **12,869 divergence events**
- **top-1 acceptance q0 = 0.7335** vs #76's 0.7287 → **|diff| = 0.0048**, max per-depth |diff| = 0.0066 (tight cross-check; validates drafter path + alignment + verify pairing)

**ρ vector (conditional, pooled over depths):**

| quantity | value | meaning |
|---|---|---|
| ρ₂ = P(rank-2 \| miss₁) | **0.4165** | rank-2 rescue given rank-1 missed |
| ρ₃ = P(rank-3 \| miss₁,₂) | **0.2655** | rank-3 rescue given ranks 1,2 missed |
| ρ₄ = P(rank-4 \| miss₁,₂,₃) | **0.1908** | rank-4 rescue given ranks 1,2,3 missed |
| cumulative cov₂ | 0.4165 | caught at rank ≤2 |
| cumulative cov₃ | 0.5715 | caught at rank ≤3 |
| **cumulative cov₄** | **0.6532** | caught at rank ≤4 (vs borrowed **0.565**) |
| true beyond top-4 | 0.3468 | 34.7% hard miss (unrescuable by any width-4 tree) |

**Per-depth ρ₂ (k=1..7): [0.397, 0.431, 0.413, 0.428, 0.435, 0.445, 0.410]** — flat across depth (range 0.40–0.45), so depth-pooling is valid and the tree's rank-2 branch pays uniformly at every spine position. This per-depth breakdown is the **additional value my probe adds over byteshark** (who measured only at the first divergence); divergence denominators per depth: [12121, 9279, 7392, 6106, 5134, 4302, 3655].

### 2. byteshark cross-validation — PASS on all three quantities
(byteshark's `splitkv-k7-rank2-branch-v0`, job `6a2dfd7d234ca64b601225aa`, 32768 proposer steps, posted by advisor in this thread)

| quantity | local (mine) | byteshark | rel. diff | within ±4.4%? |
|---|---|---|---|---|
| ρ₂ | 0.4165 | 0.4130 | **0.85%** | ✅ |
| cov₂₋₄ (ranks 2–4 aggregate) | 0.6532 | 0.6609 | 1.16% | ✅ |
| mean emit / E[T] | 3.844 | 3.921 | 1.96% (probe overhead) | ✅ |

**Config-robustness note (re: byteshark's Block64 flag):** my deployed stack runs `FUSED_SPARSE_ARGMAX_BLOCK=16`; byteshark's external official stack runs `BLOCK=64`. ρ₂ agreeing to 0.85% **across** that block-size difference confirms ρ₂ is a property of the drafter/target token distributions, **not** of the centroid sparse-argmax reduction — my probe reads the full `_select_and_score` logits + `topk`, bypassing the sparse fast path entirely (forced eager via `LOOPGRAPH_WARMUP_CALLS=1e9`), so the reduction block size cannot enter the rank extraction. (FYI denken #77 — the centroid sampler is rank-order-stable here.)

### 3. Re-price with measured ρ (run `6wr8r2y0`, #76 machinery + #68 real GEMM curve)
Anchor reproduced: F_linear(8) under the depth model = **3.84445** == measured E[T] (|err| = 3.2e-4). Re-priced #74 topologies two ways:

**Exact chain-rule per-rank split** (rho_cond = [0.4165, 0.2655, 0.1908]):

| M | F_tree | cost_mult (#68) | E[T]/lin8 | **gain** | proj TPS |
|---|---|---|---|---|---|
| 16 | 4.512 | 1.0339 | 1.174 | **+13.5%** | 486.3 |
| **32** | **5.157** | **1.0981** | **1.341** | **+22.2%** | **523.3** |

**Scalar cov_W = 0.6532 geometric** (matches above to <0.5pp): M=32 **+21.8% (521.6 TPS)**, M=16 +13.1% (484.5 TPS).

**ρ sweep, M=32 gain:** ρ=0 → **−1.1%** · ρ=0.35 → +11.2% · ρ=0.565 (borrowed) → +18.7% · **ρ=0.6532 (measured) → +21.8%** · ρ=0.75 → +25.2%.
**Full gain band** (ρ × decay × g × GEMM-share × extrap): **[−1.1%, +25.2%]**.

- **Where it lands:** measured cov₄ (0.653) > borrowed (0.565) ⇒ gain comes in **above** the +18.7% central → **+21.8%**. That's **+1.7pp** above #74's modeled +20.1% and **+3.1pp** above the borrowed-0.565 estimate.
- **M=32 dominates M=16: True.** **Tree beats deployed linear: True.** **fail_fast: NOT triggered.**
- **ρ=0 floor is −1.1%** (≈ break-even): even with zero rank-2+ rescue the M=32 deep spine nearly holds serve — **ρ is upside, not the foundation**. The tree's value does not collapse if ρ is overstated.

### 4. Width / branch-factor verdict — full max-branch-4, M=32
Decision frame from PR body: ρ₂ ≳ 0.20 → width-2 pays; marginal ρ₃, ρ₄ each ≳ 0.10 → width-3/4 pay.

- **ρ₂ = 0.42 ≫ 0.20** → width-2 pays decisively.
- **ρ₃ = 0.27 ≫ 0.10 and ρ₄ = 0.19 ≫ 0.10** → width-3 **and** width-4 **both** clear the bar → **full max-branch-4 is justified; do NOT collapse to width-2/M=16.**
- Absolute marginal acceptance by rank at depth-1: rank-1 = 0.729, rank-2 = **0.113**, rank-3 = **0.042**, rank-4 = **0.022** — each successive branch still contributes real mass that clears its #68 GEMM verify-slot cost.
- **Build target for land #71: M=32** (32-node, depth-9 spine; parent array in `treeshape_measured_results.json` → `handoff_land71`). Expected **+21.8% TPS (521.6), E[T] = 5.14**. M=16 is the secondary/fallback target (+13.1%, 484.5 TPS).

### Command
```bash
# Measurement (run z6wi4z4v):
python scripts/profiler/rank_coverage.py \
  --num-prompts 128 --output-len 512 --seed 1 \
  --wandb-group rank-coverage --wandb-name wirbel/rank-coverage
#   -> serves a SCRATCH COPY of submissions/fa2sw_precache_kenyan (served files byte-identical),
#      env RANKPROBE_ENABLE=1 RANKPROBE_W=4 LOOPGRAPH_WARMUP_CALLS=1e9 VLLM_USE_FLASHINFER_SAMPLER=0,
#      decode_outputs.py over the 128 public sharegpt prompts, conc=1.

# Re-price (run 6wr8r2y0):
python scripts/profiler/treeshape_measured_accept.py \
  --rank-coverage-json research/rank_coverage/rank_coverage_results.json \
  --wandb-group rank-coverage --wandb-name wirbel/tree-reprice-measured
```

### Peak memory
Deployed serving config unchanged (read-only probe streams records to disk, negligible host overhead): model load **8.84 GiB**, KV cache reserved **9.47 GiB** at `GPU_MEMORY_UTILIZATION=0.90` (GPU KV cache size 377,201 tokens), CUDA-graph pool 0.04 GiB. Single assigned GPU (CUDA_VISIBLE_DEVICES=6→0). No HF job / no submission launched.

### What happened
The hypothesis holds and the tree gain **firms up**. The last borrowed input (EAGLE-3's ρ=0.565) is now measured at cov₄ = 0.653 on our own stack, **higher** than borrowed, so the tree-verify gain moves from the modeled +18.7%–20.1% band up to **+21.8% central**, with M=32 dominant and all four branches justified. The result is corroborated by byteshark's independent external probe (ρ₂ 0.85% apart) and is internally consistent (q0 matches #76 to 0.005, ρ₂ flat across depth, F_linear8 reproduces E[T] to 3e-4). The ρ=0 floor (−1.1%) shows the deep spine — not ρ — is the load-bearing structure, so the conclusion is robust to ρ mis-estimation. Two minor benign server-log artifacts (a vLLM usage-telemetry `cpuinfo` thread JSONDecodeError, and `[kduma-precache] dataset unavailable` ungating) do not touch acceptance, greedy identity, or rank extraction (align_bad=0 confirms).

### Suggested follow-ups
- **land #71:** build the M=32 tree (parent array provided) targeting +21.8% / 521.6 TPS; treat ρ=0 (−1.1%) as the conservative floor for the go/no-go.
- **Beyond-top-4 (34.7%):** this is the irreducible per-step miss mass a width-4 tree cannot rescue; widening to W=5+ would need ρ₅ measured — cheap to add to this same probe (bump `RANKPROBE_W`) if land #71 wants to test whether a 5th branch clears its GEMM cost.
- **Drafter quality (fern #34):** ρ₂ is flat at ~0.42 across depth; a better-trained drafter that lifts rank-1 q0 would compress the divergence set but the rank-2+ tail (this ρ) is where tree value lives — worth tracking ρ as a drafter-training diagnostic alongside top-1.
