# PR #284 — Decode-loop host overhead (the non-model per-step decode wall)

**PRIMARY `decode_host_overhead_self_test_passes` = True**  
**TEST `host_overhead_frac` = 0.50%** (directly-measured: decode wall − deployed GPU-busy)  
**`recoverable_host_overhead_tps` = 0.50** · **`host_overhead_clears_materiality` = False** (gate 9.6 TPS)

> **Verdict:** the deployed decode loop is **GPU-bound** (GPU-busy share **99.5%**). The per-step decode wall is **8017 µs**, of which the directly-measured deployed model-forward GPU-busy (verify 6532 + drafter 1445) is **7977 µs**, leaving host/serving overhead of just **40 µs (0.50%)** — an order of magnitude below fern #274's inferred ~40% (φ band [0.125, 0.735]). The host/serving side is **CLOSED**; only the model-forward read floor remains.

## 1. Per-step decode wall (STEPTIME host-to-host, p50 steady-state)

| quantity | µs | source |
|---|---|---|
| verify (execute_model) GPU | 6532 | STEPTIME exec.gpu p50 |
| drafter (propose) GPU | 1445 | STEPTIME draft.gpu p50 |
| exec host-call wall | 6417 | STEPTIME exec.cpu p50 |
| inter-step gap (incl draft) | 1600 | STEPTIME exec.gap p50 |
| **decode wall / step** | **8017** | exec_cpu + gap |
| deployed model-forward GPU-busy | 7977 | verify + drafter |

**Wall-identity (the #275 discipline):** decode_wall_per_step × n_steps (16154) = 129.51s ≈ decode_wall_total 129.41s — residual **0.098s (0.076%)**.

## 2. Isolate the host/serving overhead — two bases

| basis for model-forward | model-forward µs | host overhead µs | host frac |
|---|---|---|---|
| **(B) deployed GPU-busy (measured, headline)** | 7977 | 40 | **0.50%** |
| (A) denken #278 micro-built (instructed) | 5673.6 | 2343 | 29.2% |

**Why the two disagree (the denken #278 over-credit caveat, made concrete):** denken's micro-built model-forward (5673.6 µs, M=1 isolated) UNDER-counts the deployed M=8 in-stack GPU-busy by **2303 µs** (verify M=8−M=1 +1565 µs; drafter deployed−micro +738 µs). That gap is REAL GPU work, not host overhead. Subtracting the micro-built number manufactures a phantom 29.2% — which is exactly why fern #274's micro-built-style inference landed near ~40%. The DIRECT CUDA-event measurement of the deployed GPU-busy removes the artifact.

## 3. Host-overhead decomposition (of the measured 40 µs)

| component | µs | on per-step blocking path? |
|---|---|---|
| scheduler / inter-graph dispatch | 40.0 | YES (host hop verify→draft graph) |
| sampling (fused GPU argmax) | 0 host (217 µs GPU, 2.7% of GPU) | NO — inside verify GPU span |
| detokenize (DETOK_ENDONLY) | 0 | NO — deferred to end-of-sequence |
| other framework residual | 0.0 | — |
| **sum** | **40.0** | resid 0.00 µs |

## 4. Recoverable host overhead

- recoverable host overhead = **40 µs** (0.50% of the cycle)
- priced into TPS (E[T]/cycle, discounted by the denken over-credit 4.818×): raw +2.40 → **+0.50 TPS** composition-honest
- materiality gate = 2% of 481.53 = 9.6 TPS → **clears = False**
- ONEGRAPH=1 already fuses the decode step into one CUDA graph; detok deferred (DETOK_ENDONLY), sampling fused+prewarmed (FUSED_SPARSE_ARGMAX), framework already FASTRENDER+orjson. Residual is the unavoidable inter-graph host hop + accept/scheduler — largely irreducible.

## 5. fern #274 grounding

- measured host-overhead fraction = **0.50%** → `refutes_magnitude` (fern φ band [12.5%, 73.5%], point ~40%)
- the host/serving residual EXISTS (grounds fern's φ from the wall side) but its MAGNITUDE is refuted: the direct measurement is ~0.5%, not ~40%. The denken-subtraction artifact (29.2%) is what sits inside fern's band.

## Self-test

- a_walltime_identity_holds: **True**
- b_decomposition_sums: **True**
- c_denken_modelforward_imported_exact: **True**
- d_nan_clean: **True**
- e_anchors_imported_exact: **True**
- f_caveats_carried: **True**
- identity_resid_frac = 0.076%, decomp_resid = 0.000 µs

## Greedy/PPL-safety certificate

`decode_host_overhead_analysis_only = True`. STEPTIME timing-only forward over the standard prompt set; no served-file change, no emitted-token change, no HF Job, no submission, NOT a launch. BASELINE 481.53 TPS unchanged (this leg adds 0 TPS; `recoverable_host_overhead_tps` is a priced-out bound, not a build, and carries the denken normalized-step over-credit caveat).
