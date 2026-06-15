# PR #306 — Does the EAGLE-3 build's VRAM headroom survive ONEGRAPH capture?

**PRIMARY `eagle3_capture_peak_self_test_passes` = True** (all 7 conditions a–g)
**TEST `eagle3_build_peak_fits_24gb` = True** · `fits_23_usable` = True · `fits_device_visible(22.058)` = True
**`eagle3_build_peak_gb` = 20.158 GiB** · **`capture_transient_gib` = 0.041** · `total_transient_gib` = 0.058
**`dominant_transient_term` = capture_time_scratch** · W&B `y1lji0c6` (group `eagle3-capture-peak`) · A10G, measured

> **Verdict:** the 3.90 GiB resident headroom **SURVIVES** runtime. The EAGLE-3 build's peak = #299 resident **20.10 GiB** + measured step transient **0.058 GiB** = **20.158 GiB** — **3.84 GiB** under 24-hard, **2.84 GiB** under 23-usable, under the 22.058 GiB device-visible cap. The capture-time + tree-verify transient eats only **1.5%** of the headroom. A build that "fits at rest" also **fits at runtime**; VRAM is **not** the binding constraint, and the build does **not** OOM during ONEGRAPH capture. The genuine #101-class risk is the capture-**size** dispatch boundary (not VRAM), which the deployed M=8 spine clears.

## 1. What was measured (random-init, A10G, 0 TPS, no served-file change)

Local profiling of the runtime PEAK (`max_memory_allocated`/`reserved`, reset between phases) of the faithful EAGLE-3 fusion step (repo's `Eagle3DraftHead`: 2560-dim, 1 Llama layer + fc[7680→2560] over {2,21,39}, reuses target embed+lm_head per #299 S0). Device total measured **22.058 GiB == #299's banked cap**. Proxy resident 2.91 GiB (used only for the per-phase peak≥resident check); the build peak **stacks the measured transient on #299's imported 20.10 GiB resident**, exactly as #299 stacked its delta on the #284 anchor.

| phase | what | measured transient | note |
|---|---|---|---|
| **(i)** ONEGRAPH capture of the **fused draft→verify** step (K=7 draft + M=8 verify, 15 capture tokens) | private capture pool (cuBLAS workspace + held activations) | **+0.0410 GiB** (`reserved` 2.962 − 2.922) | captured ✓; replay reuses pool (no new alloc) |
| **(ii)** M-wide tree-verify 262144-vocab logit GEMM | logit buffer `V×M×2B` | M8 **+0.0048** / M16 +0.0080 / M32 +0.0159 | monotone in M ✓; buffer exact = `262144·M·2B` |
| **(iii)** {2,21,39} 3-layer hidden-state retention held live | `L_FUSE·H·ctx·2B` + draft scratch | **+0.0121 GiB** (held 0.0076) | resident analog of #299's 0.0073 GiB |

## 2. Transient decomposition → dominant term (the runtime analog of #299's `extra_kv`)

`peak − resident = capture_scratch + tree_verify_logit_buffer + workspace/remat` (sums by construction; self-test c):

| transient term | GiB | share |
|---|---|---|
| **capture-time scratch** (DOMINANT) | **0.0410** | 71% |
| workspace / remat (hidden retention + verify scratch) | 0.0121 | 21% |
| tree-verify logit buffer (M=8) | 0.0048 | 8% |
| **total transient** | **0.0579** | — |

The dominant runtime transient is the **CUDA-graph capture pool** (~42 MiB: default cuBLAS ~4 MiB/handle held for the graph's life + activations the capture cannot free), not the logit buffer. The 262144-vocab logit buffer is **MB-scale even at M=32** (16 MiB) — bf16-native (PyTorch does **not** upcast argmax/softmax to fp32 unless `dtype=` is passed; issue #123911), so no doubled fp32 temporary.

**Build peak = 20.100 (resident, #299) + 0.058 (transient) = 20.158 GiB.**

| ceiling | build peak | headroom | fits |
|---|---|---|---|
| 24.0 GiB hard | 20.158 | **3.84** | ✅ |
| 23.0 GiB usable | 20.158 | **2.84** | ✅ |
| 22.058 GiB device-visible | 20.158 | 1.90 | ✅ |

## 3. The #101 precedent: carried honestly (it is **not** a VRAM OOM)

lawine's "size-29 CUDA-graph-capture crash" (#245 cycle-1, `EXPERIMENTS_LOG.md:318`) is framed in this PR as a memory spike, but the repo's own diagnosis — corroborated by vLLM source (#29091 / PR#23679) — is a capture-**size-list dispatch failure**: `max_cudagraph_capture_size=16`, and a request at batch=29 finds **no captured graph** for that size → `IndexError` lookup crash, **not** an allocation failure. For EAGLE with K=7, captured sizes must be **multiples of (1+K)=8** within the size-16 ceiling, i.e. **{8, 16}**.

| width | verify tokens | VRAM peak | captured under size-16? | #101 regime |
|---|---|---|---|---|
| **deployed M=8** | 8 | 20.158 (fits) | ✅ (8 ✓, 8\|8) | **clear** |
| M=16 | 16 | +4 MiB | ✅ boundary (16 ✓, 16\|8) | at boundary |
| M=32 | 32 | +12 MiB | ❌ (32 > 16) | **crash regime (dispatch, not VRAM)** |

So the EAGLE-3 capture sits **far below any VRAM-OOM regime** on the memory axis this leg prices, **and** the deployed M=8 spine clears the #101 capture-size boundary. The crash regime is only re-entered by **widening the tree past M=16** — a topology choice that is orthogonal to VRAM bytes (the VRAM cost of M=32 is +12 MiB, trivially affordable; the blocker there would be the dispatch list, fixed by adding the size to `cudagraph_capture_sizes`, not by freeing memory).

## 4. Mitigations — not required for the memory axis

The peak fits 23-usable with **2.84 GiB** margin, so no memory mitigation is needed. For completeness: chunked/streamed logits or an M-cap recover only MB-scale; the cheapest hygiene item is capturing only the deployed spine (K+M=15 tokens < the size-16 list) which both minimizes the pool **and** avoids the #101 dispatch crash. **Launch-hygiene flag (not a blocker):** a serving process that inherits a training-tuned `CUBLAS_WORKSPACE_CONFIG=:4096:8` would inflate the capture pool from ~4 to ~24 MiB/handle — still sub-GiB, far inside the headroom, but worth pinning unset at launch.

## Self-test (PRIMARY, a–g, NaN-clean)
a peak≥resident every phase ✓ · b all peaks finite/positive ✓ · c decomposition sums to peak−resident ✓ · d logit-buffer M-scaling monotone ✓ · e imported #299 constants exact (≤1e-6) ✓ · f GPU smoke NaN-clean ✓ · g honest caveats carried ✓ → **PRIMARY PASS**.

## Greedy/PPL-safety certificate
`analysis_only = True`. No served-file change, no emitted-token change, no HF Job, no submission, NOT a launch, NOT a build (random-init weights/activations have the same tensor **shapes/bytes** as a trained build, so the VRAM footprint transfers; the numeric values do not, and are irrelevant to byte accounting). BASELINE **481.53 TPS unchanged**; this leg adds **0 TPS**.

## Hand-off
EAGLE-3 build **survives runtime**: resident 20.10 GiB (#299) + step transient 0.058 GiB = **20.158 GiB peak**, 3.84 GiB under 24-hard / 2.84 under 23-usable. The runtime transient is **capture-pool-dominated** (~42 MiB) and tiny (1.5% of the 3.90 GiB headroom); the 262144-vocab logit buffer is MB-scale even at M=32. **VRAM is not the binding constraint at runtime**, closing the runtime half of #299's resident axis (e). The only #101-class risk is the capture-**size** dispatch boundary (size-16, multiples of 8), which deployed M=8 clears — widening the tree past M=16 re-enters lawine's crash regime, a topology decision independent of VRAM. The human GO/NO-GO can treat the VRAM≤24 clause as **satisfied at runtime**, not just at rest.
