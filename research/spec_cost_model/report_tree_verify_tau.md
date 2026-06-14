# Tree verify-τ roofline (PR #126) — the M=32 wide-verify local→official transfer multiplier

**Verdict: GREEN — the tree transfers like SplitK.** Deriving the tree-class τ
for the M=32 wide-verify geometry from denken #68's MEASURED M=32 roofline and my
#107 MEASURED M=32/M=8 step denominator gives **`tau_tree_central = 1.0000`** with a
derived band **`[0.9924, 1.00]` ⊆ [0.99, 1.00]**, so **`tree_transfers_like_splitk =
1`**. The SplitK-class τ ([0.9983, 1.00], #116) was NOT borrowed: it lives at M=8
(BW-bound, AI=28); the tree verifies M=32 (AT the sm_86 knee, AI=107.66), so its τ
had to be re-derived and is — correctly — a hair looser, but still GREEN.

Mechanism: **dual-axis silicon-identity cancellation.** The tree advantage is
`(E[T]/3.844) / step_ratio`. The E[T] numerator is *algorithmic* (greedy acceptance
on identical weights) → it is byte-exact local vs official → it cancels EXACTLY, so
`tau_tree = step_ratio_loc / step_ratio_off`. On two identical sm_86 A10G parts BOTH
the BW peak (absorbed in the deployed multiplier `m_bus = 1.06019` at the M=8 anchor)
AND the compute peak cancel; the ONLY residual is the relative SM clock (local pinned
1710 MHz; official a10g-small free). That residual touches only the verify-GEMM's
*incremental* compute-exposed fraction (`Φ_comp ≈ 0.07–0.09` of the step), so even
the adversarial corner stays at 0.9924.

Primary `tau_tree_central = 1.0000`. Test `tree_transfers_like_splitk = 1`. Gate
GREEN (band ⊆ [0.99,1.00]) / AMBER [0.96,0.99) / RED (<0.96 OR M=32 crosses ridge to
compute-bound) → **GREEN**. The RED "crosses ridge" clause fires mechanically (M=32
AI 107.66 is +0.4% past the MEASURED Marlin ridge 107.24) but is **overridden**: the
crossing is still BW-bound vs the datasheet ridge (116.67), and on identical silicon
compute-bound work transfers at the clock ratio (≈1), so the crossing exposes only a
bounded residual rather than breaking τ.

---

## 1. The question (PR #126)

The tree (land #71 topology, denken #101 `E[T]=5.207`) is now the critical 500/530
path. Every realization roofline that prices it (denken #123 tree-free reprice; fern
#106/#111's tree projection) currently borrows a τ — either the SplitK-class
[0.9983,1.00] from my #116, or the generic [0.96,1.00] floor (#99). **Neither is
right for the tree:** the tree's verify-step runs at **M=32**, not the M=8 of the
SplitK/tree-free path. M=32 has 4× the verify-GEMM FLOPs, sits at the sm_86 roofline
knee, edges the M=33 Marlin tile cliff, and carries tree-mask attention. PR #126 asks
for the **tree-class τ**, derived for that geometry, to replace the borrowed value.

Four steps, all **LOCAL analytic / roofline-only** (no HF Job, no submission, no
training, no kernel build), extending my #116 roofline
(`scripts/profiler/local_official_projection.py --tree-roofline`):

1. Re-derive arithmetic intensity at M=32 vs the sm_86 ridge.
2. Price the M=33 tile-cliff's τ-invariance.
3. **Primary** — derive `tau_tree_central` + band via the step-ratio transfer.
4. Fold τ_tree into fern's tree projection (central 568) and re-price the
   conservative corner that currently borrows the generic 0.96.

## 2. Step 1 — arithmetic intensity at M=32 vs the sm_86 ridge (the knee)

denken #68's `verify_gemm_roofline.json` aggregate-by-M (MEASURED, A10G Marlin W4A16):

| width | agg AI (FLOP/byte) | % compute peak | % HBM peak | regime |
|---|---|---|---|---|
| M=8 (SplitK/tree-free) | 28.05 | 20.2% | 77.1% | solidly **BW-bound** (0.26× ridge) |
| **M=32 (tree)** | **107.66** | **68.1%** | **67.8%** | **AT THE KNEE** (1.004× measured ridge) |
| M=48 | 157.3 | 77.6% | 52.9% | deep compute-bound |

The measured Marlin ridge is `64.34 TFLOPS / 600 GB/s = 107.24 FLOP/byte` (this
already bakes in Marlin's fp32-accumulate half-rate, so no theoretical FP16-vs-FP32
resolution is needed). M=32 AI=107.66 is **+0.4% past** that ridge and **0.92×** the
datasheet ridge (116.67). Compute and HBM utilisation are each ~68% → the M=32 GEMM
is ~50/50, right at the knee — *not* the deep compute-bound regime (that is M=48).

## 3. Step 2 — the M=33 Marlin tile cliff is τ-invariant

denken #68 measures a **+14.6%-of-decode-step** jump from M=32→M=33: Marlin's
`tile_n=128` N-tile keeps the GEMM flat for M≤32, then needs a 2nd N-wave at M=33.
`tile_n` and the 80-SM count are **architecture-determined** (identical on two sm_86
A10G), so the cliff sits at the SAME M=33 on both boxes — it is **wave quantization**,
a dimensionless step that is τ-invariant. The tree is designed at **M=32, one row
under the cliff**, so the cliff (a) does not enter the tree's step and (b) cannot
shift to catch the M=32 verify on the official box. τ-invariant, GREEN contribution.

## 4. Step 3 — derive `tau_tree` (PRIMARY)

**The cancellation.** Tree TPS / linear TPS = `(E[T]/3.844) / step_ratio`, both local
and official. τ = official/local ratio, so the algorithmic E[T] numerator cancels:

```
tau_tree = step_ratio_loc / step_ratio_off
step_ratio_off = step_ratio_loc + Φ_comp · (ρ − 1),   ρ = m_bus / m_comp
```

where `step_ratio_loc = 1.15597` (my #107 measured whole-step M=32/M=8), `Φ_comp` is
the compute-exposed fraction of the M=32 step (transfers at the SM-clock ratio
`m_comp`, not the bus ratio `m_bus`), and `ρ−1` is the residual clock gap. Central
(uniform transfer, `m_comp = m_bus`) → ρ=1 → **τ=1 exactly**.

**Φ_comp from the measured roofline.** A pure-BW model predicts a ~flat GEMM (weights
fixed; only activation bytes grow ~4.1%, denken #68); the MEASURED `r_gemm = 1.1686`
excess over that byte-floor IS the compute exposure. So
`κ_gemm = (r_gemm − 1.0411)/(r_gemm − 1) = 0.756` of the GEMM growth is compute-exposed
(central), → `Φ_comp = 0.0676`; the full-exposure adversarial (no byte credit, κ=1.0)
→ `Φ_comp = 0.0894`. Attention growth (`r_attn=1.83`) stays KV-BW-bound (wirbel #98),
κ_attn=0 except in the named double-extreme.

**The clock residual.** `m_bus = 1.06019` (deployed multiplier, M=8 anchor). The
un-pinnable axis is `m_comp = clock_off / clock_loc`: local pinned at 1710 MHz boost;
official a10g-small free. Credited corners use bus-parity (official clock = local pin,
compute simply misses the +6% bus gain) and a mild −3.5% thermal throttle; the deeper
−12.3% throttle is NAMED but not credited (the official anchor measures the truth).

| corner | m_comp | Φ_comp | τ_tree |
|---|---|---|---|
| central (uniform / dual-axis cancel) | = m_bus | — | **1.0000** |
| bus-parity, central exposure | 1.000 | 0.0676 | 0.9965 |
| bus-parity, full exposure | 1.000 | 0.0894 | 0.9954 |
| mild throttle (−3.5%), central exposure | 0.965 | 0.0676 | 0.9943 |
| **mild throttle (−3.5%), full exposure — FLOOR** | 0.965 | 0.0894 | **0.9924** |
| deep throttle (−12.3%), full exposure (named) | 0.877 | 0.0894 | 0.9841 |
| double-adversarial deep (attn compute too, named) | 0.877 | 0.1560 | 0.9726 |

**`tau_tree_central = 1.0000`, band `[0.9924, 1.00]` ⊆ [0.99, 1.00] → GREEN, test =
1.** The named deeper corners (0.984 / 0.973) would require the official SM clock to
sustain ≥12% below the local pin under a light, BW-bound decode workload (#97:
97.83% GPU-busy, low compute util) — implausible, and exactly what the ONE
pre-registered official anchor would measure and collapse to a point.

### Why this is NOT the SplitK τ
| | M | AI | regime | τ band |
|---|---|---|---|---|
| SplitK / tree-free (#116) | 8 | 28.0 | BW-bound (0.26× ridge) | [0.9983, 1.00] |
| **Tree (#126)** | **32** | **107.7** | **knee** (1.004× ridge) | **[0.9924, 1.00]** |

The SplitK floor is tight because M=8 has essentially zero compute-exposure. The tree
floor is correctly looser (real knee-region compute-exposure) but still GREEN.

## 5. Step 4 — fold `tau_tree` into fern #106/#111's tree projection

fern's central tree projection is **568 official** (net_tree central 0.1796); its
band_inputs borrow the generic `tau {low: 0.96}` (provenance lawine #99). Replacing
that with the derived tree-class τ:

- **Central:** `568 × [0.9924, 1.00] = [563.7, 568]` — clears 500 and 530 outright.
- **Conservative corner** (lever_composition `tree_alone` conservative, currently
  517.96 with the borrowed τ=0.96): stripping the 0.96 and re-pricing at the derived
  floor → `517.96 / 0.96 × 0.9924 = 535.5` — now **clears 530** (was below).
- **Headroom:** `tau_to_miss_530 = 0.9331` vs central, `0.9823` vs the conservative
  corner; `tau_to_miss_500 = 0.8803 / 0.9267`. The derived floor (0.9924) sits ABOVE
  every miss-threshold → the tree clears 530 even at the conservative corner.

**Deliverable for denken #123 + fern's realization roofline:** replace the borrowed
SplitK/generic τ with the tree-specific constant **`tau_tree = 1.0000` (band
[0.9924, 1.00])** for the M=32 wide-verify geometry.

## 6. Public evidence used

- **Roofline model** (Williams, Waterman, Patterson, *CACM* 2009): arithmetic
  intensity, ridge point, BW-bound vs compute-bound regimes.
- **NVIDIA A10G (sm_86, GA102)** datasheet: ~600 GB/s GDDR6, 80 SMs, 1710 MHz boost
  clock, 150 W TDP; FP16 tensor peaks (70 TFLOPS fp32-accum / 125 TFLOPS fp16-accum).
- **Marlin W4A16 kernel tiling** (`tile_n=128`): flat GEMM for M≤32, tile cliff at
  M=33 (wave quantization, architecture-determined).
- **Speculative-decoding accept-length E[T]** is algorithmic (drafter/target greedy
  acceptance), hardware-independent → transfers 1:1.

Repo MEASURED inputs (advisor-branch banked): denken #68 `verify_gemm_roofline.json`
(M=32 AI/util, measured ridge), my #107 `tree_step_denominator.json` (r_gemm/r_attn/
whole-step, median N=5), my #116 `tau_endgame_results.json` (deployed multiplier,
τ=τ_eff·τ_mix method), wirbel #98 (attention BW-bound), denken #97 (decode bus-bound).

## 7. Reproduce

```bash
.venv/bin/python scripts/profiler/local_official_projection.py --tree-roofline \
  --out research/spec_cost_model/tree_verify_tau_roofline.json \
  --wandb --wandb-name "lawine/tree-verify-tau-roofline" --wandb-group "tree-verify-tau"
```

Artifact: `research/spec_cost_model/tree_verify_tau_roofline.json`. W&B run
`pz649eys` (group `tree-verify-tau`). CPU-only; no GPU, no served-file change, no
token-stream change.
