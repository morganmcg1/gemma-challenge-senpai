STUDENT wirbel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"analysis_only":true,"official_tps":0,"wandb_run_ids":["qgyqilcm"],"primary_metric":{"name":"specdec_asymptotic_ceiling_tps","value":11816.9},"test_metric":{"name":"verify_cost_k7_measured_ms","value":13.137},"specdec_regime":"acceptance_limited","specdec_ceiling_exceeds_ship":true,"serve_equiv_check":true}

## Results — spec-dec verify-cost curve `C(M)` + asymptotic ceiling on base_fullhead

**Verdict: `specdec_regime = acceptance_limited` (the expected falsifiable outcome).** The verify cost does **not** bound spec-dec below the ship. The drafter-independent A→∞ ceiling is ~11.8k TPS — an order of magnitude above the 500 gate — so whether spec-dec clears 375.857 is determined **entirely by achievable mean-acceptance** (lawine #572 MTP-K7 + fern #573 ngram + `A_ship`), not by the per-verify-step cost. My cost-structure leg pins the binding constraint to **acceptance** and grounds fern's model denominator with a measured number.

This is the 3rd spec-dec leg (group `base-fullhead-specdec-ceiling`). **LOCAL A10G, analysis-only, NO HF fire, no served-file change.**

### 1. The verify-cost curve `C(M)` (PRIMARY, directly measured)

Per-step GPU-event latency (CUDA-event around `execute_model`, bucketed by M = `total_num_scheduled_tokens`, prefill excluded, 5%-trimmed warm-median, MAX_NUM_SEQS=1). M = K+1 verified positions; ngram drafter (drafter-independent VERIFY cost, draft ~0).

| M | K | `C(M)` ms (±ci95) | n_steps | head-GEMM ms | `TPS_max=1000·M/C(M)` |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 11.4960 ± 0.0094 | 8000 | 2.789 | 87.0 |
| 2 | 1 | 12.9621 ± 0.0124 | 4436 | 2.788 | 154.3 |
| 3 | 2 | 13.0065 ± 0.0134 | 3855 | 2.795 | 230.7 |
| 5 | 4 | 13.1367 ± 0.0136 | 3599 | 2.809 | **380.6** ← clears ship |
| 8 | 7 | 13.1369 ± 0.0148 | 3399 | 2.840 | 609.0 |
| 9 | 8 | 13.3716 ± 0.0161 | 3390 | 2.850 | 673.1 |
| 17 | 16 | 14.2623 ± 0.0142 | 3278 | 3.068 | 1192.0 |

- **The head GEMM is ~flat in M** (2.789 ms @M=1 → 3.068 ms @M=17): the full 262k bf16 head reads its 1.342 GB **once per step** and computes logits at M positions for ~free. Measured head read ≈ 2.79 ms ⇒ effective ~480 GB/s (A10G HBM floor at 600 GB/s = 2.24 ms). **This is exactly the weight-read amortization the hypothesis predicted** — and it is why the ceiling is high.
- **`TPS_max(M)` (all-accepted) crosses the ship (375.857) at M=5** and keeps climbing. This is the robust, directly-measured statement: even modest draft-length with perfect acceptance clears the ship.

### 2. Fit + asymptotic ceiling (PRIMARY)

Weighted LS on the consistent spec-dec verify path (M ≥ 2; the M=1 no-spec point is excluded because it does not pay the ~1.4 ms spec-machinery fixed overhead):

```
C(M) = 12.7044 + 0.08462·M    (ms)    R² = 0.9272
c_compute = 0.08462 ms / position
specdec_asymptotic_ceiling_tps = 1 / c_compute = 11,816.9 TPS
```

- `specdec_ceiling_exceeds_ship = True`; `specdec_ceiling_gap_to_ship = −11,441` (ceiling is **11,441 TPS above** the ship).
- All-points fit (incl. no-spec M=1) gives ceiling 6,972.8 TPS (R²=0.675, biased by the no-spec-overhead step); reported for completeness only.
- **Caveat (honest):** `1/c_compute` extrapolates the **locally-measured marginal cost** in the M≤17 range, where the A10G is weight-read-bound and each extra verified position costs only ~0.085 ms. The curve would eventually steepen once genuinely compute-bound, but that is far beyond any realistic M — so the verdict (ceiling ≫ ship) is robust regardless of the exact headline value. The directly-measured `TPS_max(M)` crossover at M=5 is the un-extrapolated version of the same conclusion.

### 3. Serve-equivalence — **the 252.69 anchor is NOT `1/C(1)`** (flag for advisor)

The PR's serve-equivalence identity is `1/C(1) == 252.69`. **On a10g that is physically impossible**, and I have direct same-substrate evidence for the correct interpretation:

- **`1/C(1)` (true no-spec, full 262k head) = 86.99 TPS**, reproducing the K=0 served warm-median 87.18 TPS (Δ 0.2%, `serve_equiv_nospec_consistent=True`). A no-spec step that reads the full head + int4 body is BW-bound at C(1)=11.50 ms; the head GEMM alone is 2.79 ms, so it **cannot** be 3.96 ms (=1/252.69).
- **base_fullhead + the ship's MTP-K7 drafter = 251.22 TPS**, reproducing the #553 anchor 252.69 (Δ 0.6%, `serve_equiv_mtp_substrate_ok=True`).

So **252.69 is the MTP-K7 *served* number (A_mtp / C(8)), not `1/C(1)`.** `serve_equiv_check = True` under this corrected split. Cross-checked a third way with a no-probe 3-way diagnostic on the identical substrate (`diag_anchor.py`): nospec **87.65** / MTP-K7 **251.28** / ngram-K7 **102.99** TPS. The "no-spec" label on the #553 anchor is a misnomer — it is the MTP serve.

- **Implied MTP mean-acceptance** `A_mtp = 251.22 · C(8)_mtp / 1000 = 2.95 tok/step` (using the graph-captured MTP step cost below; 3.30 if using the eager verify cost). This is consistent with lawine #572's MTP-K7 design point, derived here purely from cost + served-TPS.

### 4. Grounding fern #573's model denominator (PRIMARY cross-check)

The measured K=7 (M=8) verify cost, banked for fern to consume (the "no asserted-but-unchecked input" standard):

- **`verify_cost_k7_measured_ms` = 13.137 ms** — ngram drafter-free, **eager** verify (loopgraph disengaged). Use this as a generic/eager verify denominator. This is an **upper bound** on the verify cost.
- **`verify_cost_k7_mtp_step_ms` = 11.741 ms** — the ship's actual MTP-K7 step (verify + draft head), **CUDA-graph-captured** via ONEGRAPH. The MTP step is *cheaper* than the eager ngram verify because the loopgraph fuses the K=7 loop and removes per-kernel launch overhead. **For modeling the ship (MTP + loopgraph), use ~11.74 ms; 251.22 = 2.95 / 11.741 × 1000.**

### Comparison to anchors

| quantity | value | anchor |
|---|---:|---|
| no-spec `1/C(1)` | 86.99 TPS | (true base_fullhead no-spec) |
| base_fullhead + MTP-K7 served | 251.22 TPS | **252.69** (#553) ✓ |
| `TPS_max(M=5)` all-accepted | 380.6 TPS | ship **375.857** (clears) |
| asymptotic ceiling `1/c_compute` | 11,816.9 TPS | ≫ gate 500, baseline 481.53 |
| regime | **acceptance_limited** | — |

`quality_gate_passes_by_construction = True` (base_fullhead full bf16 head). **identity_note:** this card measures COST, not tokens — greedy-identity is fern/lawine's empirical job (#566 standard), not asserted here.

### Run details

- **Command:**
  ```
  CUDA_VISIBLE_DEVICES=6 python research/specdec_verify_cost/cost_curve_driver.py \
    --wandb_name wirbel/specdec-verify-cost-asymptote \
    --wandb_group base-fullhead-specdec-ceiling
  ```
  (serves base_fullhead once per K∈{0,1,2,4,7,8,16} with an ngram drafter + a default-off per-step probe via the research-dir `sitecustomize` chain — no shipped submission file changed — plus one MTP-K7 substrate-anchor pass.)
- **base_fullhead substrate:** stock `gemma-4-E4B-it-qat-w4a16-ct` snapshot (own cache, NO baked bucket), `LM_HEAD_PRUNE=0`, `LM_HEAD_PRUNE_REQUIRE=0`, `PCK04_KEEPSET=""`, `PLE_FOLD_EMBED_SCALE=1`, MAX_NUM_SEQS=1.
- **Peak VRAM:** 19.39 GB. **Elapsed:** ~1608 s (8 server boots × 32×256 conc=1 decode passes).
- **W&B:** run `qgyqilcm` (group `base-fullhead-specdec-ceiling`), full report artifact `specdec-verify-cost-report` attached. NaN-clean.
- **No HF Job / `--launch` / submission / served-file change.** `analysis_only=true`, `official_tps=0`.

### What happened — honest analysis

The hypothesis is confirmed in its expected direction: **base_fullhead spec-dec is acceptance-limited, not verify-cost-limited.** The mechanism is clean in the data — at served verify granularities (M ≤ 17) the A10G is weight-read-bound (the int4 body + the 262k head read dominate, both ~flat in M), so the marginal cost of an extra verified position is ~0.085 ms. That makes `M/C(M)` rise almost linearly with M and clear the ship by M=5. The verify pass therefore imposes no real ceiling on spec-dec below ~11.8k TPS; the entire open axis lives in **how many of those M positions a drafter can actually get accepted.**

The one substantive surprise is the **anchor reinterpretation**: the #553 "base_fullhead no-spec 252.69" is really base_fullhead **+ MTP-K7** (251.22 reproduced three ways). `1/C(1)` is 87, not 252.69. I flag this because the PR's serve-equivalence identity rests on it; the corrected reading (nospec-consistency + MTP-substrate-anchor) is what `serve_equiv_check=True` certifies. A secondary surprise: the MTP step (11.74 ms, graph-captured) is *faster* than the drafter-free eager ngram verify (13.14 ms) — eager launch overhead, not compute, accounts for the gap. fern should pick the denominator that matches her path.

The three legs now close the axis: **cost structure (this card) + achieved acceptance (lawine #572) + acceptance→TPS model / `A_ship` (fern #573)**. Cost is not the constraint; acceptance is.

### Suggested follow-ups

- **fern #573 cross-check:** reconcile her modeled per-step verify cost against `verify_cost_k7_mtp_step_ms=11.741` (ship path) / `verify_cost_k7_measured_ms=13.137` (eager). If her denominator differs, that delta directly moves `A_ship`.
- **`A_ship` from this cost curve:** with C(8)_mtp=11.741 ms, clearing the 375.857 ship needs `A ≥ 375.857 · 11.741/1000 = 4.41 tok/step` at K=7 (vs measured MTP A≈2.95). Clearing the 500 gate needs `A ≥ 5.87`. That is the acceptance bar the drafter axis must hit — a concrete target for lawine/fern to size against.
- **Advisor decision needed:** confirm the 252.69-is-MTP reinterpretation so the shared `base-fullhead-specdec-ceiling` model uses `1/C(1)=87` for the no-spec leg and `A/C(8)` for the spec legs consistently across all three cards.
- Not implemented (out of scope): an MTP K-sweep to get the *graph-captured* marginal cost directly (this card used ngram for clean drafter-free M-buckets; the MTP slope would be the exact ship-path c_compute, but the verdict is already unambiguous).
