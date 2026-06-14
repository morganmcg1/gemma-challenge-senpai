# Submission MUST-RETAIN manifest — flag-by-flag packaging de-risk (PR #186)

**Lane:** operational/packaging de-risk for the projected both-bugs/descent tree build (land #71).
**Method:** LOCAL CPU-only **consolidation + reproduction-gate**. No GPU / vLLM / HF Job / submission /
served-file change / kernel deploy. Adds **0 TPS**. BASELINE stays **481.53**. Greedy/PPL untouched.
**This document INFORMS the `Approval request: HF job` packaging check and land #71's tree manifest. It
does NOT authorize a launch.**

## What this is

Across #148 / #154 / #157 / #163 / #169 (mine) + kanna #138 (merged) the *cost-of-omission* of each
load-bearing serving flag was measured, but those costs live as point-measurements scattered across 5+
merged PRs. This leg consolidates them into **one verified manifest**: for every flag the projected stack
rides on, `{present_value, must_retain, measured_cost_if_dropped, source_leg, greedy/PPL-safe}` — and a
**self-test that re-loads each banked source JSON at runtime and reproduces its imported cost**, proving the
consolidation is faithful (no drift from the source-of-truth artifacts).

It is the build-side twin of fern #185 (numerical GO/NO-GO): fern asks *"do the measured numbers clear the
bar?"*; this asks *"does the as-submitted BUILD faithfully carry the flags those numbers assume?"*. Both
must pass before the one human-approved shot.

## Headline numbers

| metric | value |
|---|---|
| **`manifest_self_test_passes`** (PRIMARY) | **1** (25/25 checks reproduce banked sources) |
| **`binding_packaging_cost_pct`** (TEST) | **85.17 %** of realizable descent TPS |
| `n_flags_enumerated` | 22 |
| `n_must_retain` | 19 |
| `n_double_load_bearing` | 5 |
| `n_banked_cost_rows` (priced) | 8 |
| metrics NaN-clean | true |
| projected official (pinned, descent / both-bugs) | 519.95 / 535.44 TPS |

`binding_packaging_cost_pct` = `(522.38 − 77.45) / 522.38 × 100` from #163's own apples-to-apples scenario
pair (realizable descent vs the SAME build with the relocate reverted to the host-loop). **A single dropped
flag — the relocate vectorization — costs 85 % of the projected throughput.**

## MUST-RETAIN manifest (ordered by descending cost-of-omission)

Row 1 is the binding packaging risk. ✓ = greedy-identical + PPL-safe (speed-only). **dLB** =
double-load-bearing (omission breaks *validity*, not just speed).

| # | flag | present_value | cost if dropped | Δ-sort (TPS) | source | greedy/PPL | dLB |
|---|---|---|---|---:|---|:---:|:---:|
| 1 | **`relocate_salvaged_kv` == vectorized/device (NOT host-loop)** | land #71: fused `[L,W,H,D]` index_select+index_copy_ by **device** commit-index (or paged slot-map) | host-loop reverts descent **516→77 TPS** (1571× per-call; +570 % step; bar 4.88→32.6) | 444.92 | #157 / #163 | ✓ (eq=1.0) | — |
| 2 | **`PRECACHE_BENCH=1`** (+`PRECACHE_REQUIRE=1` fail-closed) | 1 | PRECACHE=0 → **3.526 %** single-shot divergence (#169); also holds the +6.019 % local→official multiplier (#148 Leg B) | 18.33 | #169 / #148 | ✓ | — |
| 3 | **`num_speculative_tokens` == 7** (MTP draft length) | 7 | K=8/9 → **−13/−16 TPS** (#138); #90 K8 −14.1, K9 −13.6; K6 −3.3, K5 −15.9 (inverted-U) | 14.50 | #90 / #138 | ✓ | — |
| 4 | **descent accept-walk == sync-free device** (no `.item()`) | land #71: device-scalar accept length (vLLM-v1 RejectionSampler, zero-sync) | sync-bound → +2.20 % step vs +0.39 % sync-free (#147) **AND breaks CUDA-graph capture** (#163) | 9.41 | #147 / #163 | ✓ | — |
| 5 | **decode-path argmax-only logits** (scatter+LP avoidance) | land #71: `argmax(pruned[M,12288])→kept_ids`; **full scatter+LP kept on the prefill PPL path** | revert full scatter`[M,262144]`+LP on decode → **−1.11 % step** (~−3.6..−5.6 TPS); bar 4.808→4.862 | 5.76 | #154 / #163 | ✓ | **dLB** |
| 6 | **`CENTROID_TOP_K` == 64** | 64 | topk128 → **−3.9 TPS**, no accept gain (#138); 64 is the optimum | 3.90 | #138 | ✓ | — |
| 7 | **`ONEGRAPH=1` + `LOOPGRAPH_REQUIRE_CAPTURE=1`** | 1 / 1 | drafter propose loop → eager; K=7 width-1 iters become per-launch-bound (capture-class) | capture-class | #154 / #163 | ✓ | — |
| 8 | **`DIXIE_FUSED_ACCEPT_PREP=1` + `DIXIE_SLIM_GREEDY=1`** | 1 / 1 | accept-prep leaves the device-resident Triton kernel for the host path (#163 = CLEAN/device) | capture-class | #163 | ✓ | — |

Rows 7–8 are capturability-class (not separately TPS-priced in the banked import set), so `Δ-sort = 0`;
they sort last but remain MUST-RETAIN because losing capture re-prices rows 1/3/4.

## Full served-surface enumeration (classification only — costs owned by their own merge legs, or free)

**Speed flags (must_retain, cost owned by their own non-imported leg):** `LM_HEAD_PRUNE=1`, `FA_SLIDING=1`,
`SPLITKV_VERIFY=1`, `PLE_FOLD_EMBED_SCALE=1` (+PLE fastpaths), `FEOPT_ORJSON`/`FASTRENDER`/`DETOK_ENDONLY`,
`LD_PRELOAD=tcmalloc`/`PYTORCH_CUDA_ALLOC_CONF`/`PERFORMANCE_MODE`, `DRAFTER_BUCKET=ft-v1-epoch_001`
(+sha256 guard, acceptance/E[T]).

**Validity-critical (double-load-bearing — omission breaks PPL/greedy, not just speed):**
`OVERRIDE_GENERATION_CONFIG temperature=0.0`, `MAX_NUM_SEQS=1`/`MAX_MODEL_LEN=4096`/`DTYPE=bfloat16`
(scoring contract), `WEIGHTS_BUCKET/LOCAL_MODEL_DIR → int4-pck04 baked dir` + `PCK04_KEEPSET`.

**Free / cosmetic:** `FUSED_SPARSE_ARGMAX_BLOCK` (16 or 64 — K-neutral, greedy-identical 128/128, 0 standalone
TPS per #138), `UVICORN_LOG_LEVEL`/`DISABLE_LOG_STATS`/`PATCH_BENCH_JINJA2`, diagnostic probes
(`STEPTIME`/`FA_SLIDING_DIAG`/`PROFILER_CONFIG`, must stay off).

**TRAP (must remain UNSET):** `LSK_SKIP_LAYERS` — if accidentally set it drops decoder layers and breaks
output. The manifest asserts it ABSENT.

## Submission checklist — for the `Approval request: HF job` issue + land #71 build review

Before the irreversible shot, verify flag-by-flag that the as-submitted both-bugs/descent build realizes the
projected stack:

1. **[BINDING] relocate is vectorized/device, NOT a host Python loop.** Confirm the salvaged-KV relocation in
   the land #71 build is a single fused device index_select/index_copy_ (or paged slot-map) keyed on a
   **device** commit-index — no per-layer `.item()`/host loop over the 37 layers. *If this reverts, descent
   collapses 516→77 TPS (−85 %) with no submit-time warning.* (#157 / #163)
2. **`PRECACHE_BENCH=1`** present in the launch env (and `PRECACHE_REQUIRE=1` fail-closed). (#169 / #148)
3. **`num_speculative_tokens == 7`** in `SPECULATIVE_CONFIG` (not 6/8/9). (#90 / #138)
4. **Accept-walk is sync-free device** (RejectionSampler zero-sync; no `.item()` in the per-node accept path)
   — this is what keeps rows 1/3 inside the captured graph. (#147 / #163)
5. **Decode emits argmax-only over the pruned head; the FULL scatter+LP stays on the prefill/prompt_logprobs
   path** (double-load-bearing — dropping it on prefill breaks PPL). (#154)
6. **`CENTROID_TOP_K == 64`** (not 128). (#138)
7. **`ONEGRAPH=1`, `LOOPGRAPH_REQUIRE_CAPTURE=1`, `DIXIE_FUSED_ACCEPT_PREP=1`, `DIXIE_SLIM_GREEDY=1`** all set
   (capture + device-resident accept). (#154 / #163)
8. **Validity contract intact:** `temperature=0.0`, `MAX_NUM_SEQS=1`, `MAX_MODEL_LEN=4096`,
   `DTYPE=bfloat16`, weights → validated int4-pck04 baked dir + `PCK04_KEEPSET`.
9. **`LSK_SKIP_LAYERS` is UNSET.**

Pairs with **fern #185** (numerical GO/NO-GO calculator) and **denken**'s tree-submission validity preflight
(boot/PPL/128). Together: *do the numbers clear?* (fern) + *does the build carry the flags the numbers
assume?* (this) + *does it boot valid?* (denken). **The launch GO remains gated on land #71's measured
self-KV λ (denken #178) — this manifest is orthogonal to that λ gate and authorizes nothing.**

## Self-test (PRIMARY) — how fidelity is proven

`self_test()` re-loads each banked JSON at runtime and asserts the recomputed/stored value matches source
(25/25 pass):

- **#148:** `K_cal == 481.53/3.844 == 125.26795`; `multiplier == 481.53/454.1937 == 1.06019`; source
  self-test passed.
- **#169:** PRECACHE=0 divergence == 3.526 %; `bus_ratio_tree_invariant == 1`; `official_shift_tps == 0`.
- **#157:** host/vec per-call == 145.24 ms / 0.0924 ms → **1571×**; host descent 77.3 / vec 516.4;
  `recoverable_step_pct == 569.9 %`; `equivalence_rate == 1.0`.
- **#154:** `recoverable_step_pct` 0.857 % (cons) / 1.108 % (real); source self-test passed.
- **#163:** realizable descent 522.38 / host-loop 77.45; `residual_host_ops == 0` of 12; self-test passed.
- **#90/#138:** cited −13/−16 brackets #90 measured (−14.1/−13.6); K7 is the inverted-U argmax; #138
  K7-block64 reproduces #90 K7 within 0.1 %.
- **manifest internal:** ordered by descending cost; row 1 == the relocate host-loop.

## Artifacts

- `tree_submission_manifest.json` — full manifest + banked imports + self-test checks + test metric.
- `build_tree_submission_manifest.py` — the consolidation + reproduction-gate (CPU-only; re-runnable).
- W&B run `u9kje7sn` (group `tree-submission-must-retain-manifest`).

## Scope / isolation

CONSOLIDATION of my own merged legs (#148/#154/#157/#163/#169) + kanna #138 + the served
`fa2sw_precache_kenyan` config. No re-derivation, no new measurement. All imported artifacts are merged into
`approval-gated-8gpu-20260613` (in `research/`). No other live student branches inspected. **NOT a launch.
NOT open2.**
