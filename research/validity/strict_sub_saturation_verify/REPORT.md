# Strict sub-saturation verify: does a sub-80-SM M escape the 473.5 tax?

**PR #358 · stark · branch `stark/strict-sub-saturation-verify` (from `approval-gated-8gpu-20260613`)**

CPU-analytic + on-target-GPU mechanism probe. **0 official TPS. Baseline 481.53 UNCHANGED.**
NOT an HF Job / launch / submission / served-file change / model swap / modality change.
Greedy identity is **measured**, never broken.

---

## Hypothesis

The human reversed #124 (issue #319, 2026-06-15 10:56Z): **strict byte-exact greedy-token
identity is the live contract.** denken #332's strict ceiling **473.5** assumes the batched
multi-token verify forward is occupancy-**SATURATED** (the c=64 tree-attention spawns 96 CTAs
> the A10G's 80 SMs), so the determinism tax φ=0.075 is irreducible. **But below the 80-SM
wall, a smaller-M / narrower-tree verify has SM headroom** — could it pay a lower tax (room
for a deterministic single-pass reduction without forgoing parallelism), so that the strict
ceiling `520.953·(1−tax(M))·(E[T]_M/E[T]_8)` beats 473.5 — or does the E[T] loss from a
smaller M outpace the φ gain?

## Method (two halves, per advisor amendment 2026-06-15 11:40Z citing human #319 11:27Z)

1. **CPU-analytic** (`strict_sub_saturation_verify.py`): model the verify CTA count vs M,
   find `M_sat`, model φ(M) under sub-saturation, compute `ceiling(M)` for M∈{2,4,8,16,32}
   over banked anchors. PRIMARY `strict_sub_saturation_self_test_passes`; TEST
   `max_strict_ceiling_over_M` + `sub_saturation_escapes_473`.
2. **GPU mechanism probe** (`gpu_mechanism_probe.py`): on the pod GPU, MEASURE whether a
   deterministic single-pass small-M verify reduction (M∈{2,4,8}) preserves byte-exact greedy
   identity vs plain AR (GPU-1), and the realized per-M latency (GPU-2). New MEASURED metric
   `greedy_identity_rate_by_M`.

### ★ HONEST hardware note (contra the advisor's assumption)

The advisor expected a *"96 GB different-SM pod GPU"*. The **actual pod GPU is an NVIDIA
A10G — 80 SMs, 23.7 GiB — the SAME GA102/80-SM architecture as the deployment target.** So the
80-SM occupancy wall is **ON-target** here: mechanism, determinism, and the 80-SM-relative
latency shape are all measured on the real target SM count. What this probe does **not**
reproduce is the full 42-layer served forward + lm_head + vLLM/FlashInfer kernel + official
benchmark harness → the *exact* official strict TPS still needs the served a10g path
(Tier-2, approval-gated #319). **This probe measures the MECHANISM; the a10g confirms the number.**

---

## Results — CPU analytic  (run `2i45d673`, self-test 36/36 PASS, NaN-clean, peak 12 MiB)

CTA-vs-M occupancy (real config: head_dim 256, 8 q-heads, 2 kv-heads → BLOCK_Q=4):

| M | q-blocks | N_nonreduction CTAs | N_full_3d CTAs (×16 split-KV) | regime |
|---|---|---|---|---|
| 2 | 1 | 2 | 32 | sub-saturation |
| 4 | 2 | 4 | 64 | sub-saturation |
| **8 (deployed)** | 3 | 6 | **96 > 80 SMs** | **SATURATED** |
| 16 | 5 | 10 | 160 | saturated |
| 32 | 9 | 18 | 288 | saturated |

`M_sat = 6.0`. Ceiling(M) under two φ models (faithful = N_nonreduction/80, grows with M;
steelman = idle-SM headroom):

| M | E[T]_M (×E[T]_8) | φ_faithful → ceil | φ_steelman → ceil | beats 473.5? |
|---|---|---|---|---|
| 2 | 1.729 (×0.449) | 0.025 → **211.5** | 0.400 → 220.1 | no |
| 4 | 2.722 (×0.707) | 0.050 → **333.8** | 0.800 → 361.0 | no |
| 8 | 3.851 (×1.000) | 0.075 → 473.5 | 0.075 → 473.5 | — (anchor) |
| 16 | 4.718 (×1.225) | 0.125 → 583.3 | 0.125 → 583.3 | yes (super-sat) |
| 32 | 5.007 (×1.300) | 0.225 → 625.6 | 0.225 → 625.6 | yes (super-sat) |

- **`max_strict_ceiling_over_M = 625.6`** at **M*=32** — but that is E[T]-driven **SUPER**-saturation
  (the OPPOSITE of the hypothesis) and rests on the optimistic linear-E[T] (M-independent
  verify step) convention, so it is an UPPER bound.
- **max SUB-saturation ceiling = 333.8 (faithful) / 361.0 (steelman) ≪ 473.5.**
- **`sub_saturation_escapes_473 = False`** under BOTH φ models.
- On the faithful launch geometry `recovery_phi(M)=N_nonreduction(M)/80` **grows** with M
  (M=2→0.025, M=4→0.050, M=8→0.075): a smaller M shrinks the deterministic-compatible grid and
  *lowers* recovery — sub-saturation is **doubly penalised** (lower φ AND lower E[T]).

## Results — GPU mechanism probe  (run `ecfuv5ud`, self-test 11/11 PASS, NaN-clean, peak 19.7 GiB)

Real gemma-4-E4B-it text-decoder attention dims, bf16, on the **A10G (80 SMs)**. Three real
SDPA paths: `ar_ref` = MATH per-row (plain AR), `det_batched` = MATH one M-query forward
(deterministic single-pass), `flash_batched` = FLASH one M-query forward (deployed split-KV).

**GPU-1 — `greedy_identity_rate_by_M`** (per-layer attention-output row byte-identity vs plain AR; a
byte-identical row can never flip a downstream token → a *sufficient* condition for greedy identity):

| M | **det (single-pass) vs AR** | flash (split-KV) vs AR | det reproducibility (rerun) | flash surrogate-argmax flip |
|---|---|---|---|---|
| 2 | **0.844** | **0.000** | 1.000 | ~0.6 % |
| 4 | **0.850** | **0.000** | 1.000 | ~0.7 % |
| 8 | **0.846** | **0.000** | 1.000 | ~0.6 % |

(L=2048 full-attention context; at L=512 det rises to ~0.91, flash still 0.000.)

- The deployed **split-KV (flash) verify is NEVER byte-identical to AR** (row-identity 0.000 at
  every M) → the deployed fast path **genuinely violates strict #319**. The φ tax is REAL, not a
  modelling artifact.
- A **deterministic single-pass reduction is bit-REPRODUCIBLE** run-to-run (1.000) and **recovers
  most** of the identity the split-KV path destroys (0.000 → ~0.85). Determinism is achievable.
- **BUT even the deterministic batched verify is NOT fully AR-byte-exact** (~0.85, not 1.000):
  batching M query rows tiles the QK/PV GEMMs differently than per-row AR, so strict byte-identity
  needs **more than a deterministic reduction** (a batch-invariant GEMM too). **The lever is HARDER
  than the hypothesis assumed, not easier.**
- Controlled manual proof (we own every bit): the 16-way split-KV online-softmax combine differs
  from a single-pass reduction by ~5e-4–1e-3 (bf16) → the reduction-order non-associativity that
  *is* the φ tax is measurably nonzero.
- Cross-check: the per-layer flash surrogate-argmax flip (~0.5–0.7 %) compounded over 42 layers is
  the same ballpark as the banked ~0.73 % end-to-end M=8 token divergence (denken/#52).

**GPU-2 — realized per-M latency** (det MATH single-pass vs split-KV FLASH; batched throughput proxy):

| M | det single-pass | split-KV flash | det/split |
|---|---|---|---|
| 2 | 17.28 ms | 2.065 ms | 8.37× |
| 4 | 17.50 ms | 2.067 ms | 8.46× |
| 8 | 17.71 ms | 2.039 ms | 8.68× |

- **Split-KV verify latency is ~FLAT across M∈{2,4,8}** (M8/M2 = **0.987**). If the small-M verify
  were occupancy/compute-bound, latency would scale with M. Flatness ⇒ the fixed KV read
  (**bandwidth**) dominates, and the sub-saturation idle SMs are **NOT** the binding resource → there
  is **no free headroom to "spend" on a deterministic reduction.** This is the direct on-target
  measurement that confirms the CPU model's central BW-bound claim (AI 7.88 ≪ ridge 208).
- The det/split absolute ratio (~8×) is an **UPPER bound** on the determinism cost — SDPA-MATH is an
  *unfused, score-materialising* reference, not an optimised deterministic kernel (the ratio inflates
  with L/batch for that reason). The confound-free, decision-relevant signal is the **flatness** above.

---

## Integrated verdict — REFUTED (analytic + measured agree)

**No sub-saturation (M, tree-shape) escapes the 473.5 strict determinism tax.**

- **Analytic:** the only sub-saturation lever is smaller M / narrower tree; on the faithful
  geometry that *lowers* recovery_phi AND E[T], collapsing the ceiling to 333.8 / 211.5 ≪ 473.5
  (steelman 361.0 still ≪ 473.5). The ceiling exceeds 473.5 only at LARGER M (super-saturation,
  E[T]-driven, optimistic convention) — the opposite of the hypothesis.
- **Measured (on-target A10G):** the deployed split-KV verify is never AR-byte-exact (tax real);
  a deterministic reduction is reproducible but recovers only to ~0.85 byte-identity (batch-invariance
  gap → strict identity is *harder*, not cheaper, at sub-saturation); and the verify is **bandwidth-
  bound with latency flat across M** → sub-saturation gives no usable headroom.
- A wider tree only ADDS CTAs (deeper saturation); it cannot move a verify below the 80-SM wall.

The strict >500 lane is **not** reachable by narrowing under the occupancy wall. It needs a TRUE
deterministic-reduction kernel that is *also* batch-invariant (denken's UNBUILT, human-gated artifact)
or a different contract (#124). Exact official strict TPS still needs the served 42-layer a10g path
(Tier-2, approval-gated #319).

## Caveats

- `ceiling(M)` uses the linear-E[T] convention (M-independent verify step) — optimistic at large M;
  the M=16/32 ceilings are an UPPER bound and do not change the verdict.
- GPU-1 identity is a **per-layer attention-output** byte-identity proxy (a sufficient condition for
  token identity), not the end-to-end 42-layer token rate (Tier-2). GPU-2 det/split absolute latency
  is an UPPER bound (unfused MATH reference); the robust signal is the M-flatness (BW-bound).
- Probe uses a shared-context KV block (no intra-M causal mask) to isolate the reduction cleanly; the
  reduction-order non-associativity (the φ tax) is independent of masking.
- ORTHOGONAL to denken (body-bits / saturated φ-floor), kanna (non-batched M=1), wirbel (composition),
  lawine (frontier baseline), fern (recovery→ceiling integrator). This owns the verify occupancy /
  M-shape axis.

## Reproduce

```bash
# CPU analytic (run 2i45d673)
cd target/ && python research/validity/strict_sub_saturation_verify/strict_sub_saturation_verify.py \
  --self-test --wandb_group strict-sub-saturation-verify --wandb_name stark/strict-sub-saturation-verify

# GPU mechanism probe (run ecfuv5ud) — single A10G; CUDA_VISIBLE_DEVICES=0 (pod default points at a
# non-existent 2nd GPU on this single-A10G pod)
cd target/ && CUDA_VISIBLE_DEVICES=0 uv run --no-sync python \
  research/validity/strict_sub_saturation_verify/gpu_mechanism_probe.py \
  --wandb_group strict-sub-saturation-verify --wandb_name stark/strict-sub-saturation-gpu-mechanism
```

W&B (`wandb-applied-ai-team/gemma-challenge-senpai`): CPU `2i45d673`, GPU `ecfuv5ud`.
