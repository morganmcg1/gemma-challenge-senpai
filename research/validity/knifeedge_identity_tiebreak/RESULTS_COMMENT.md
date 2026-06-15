STUDENT stark:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["g3954eh3"],"primary_metric":{"name":"selective_fix_tps_cost","value":2.5984251968503935},"test_metric":{"name":"knifeedge_tiebreak_self_test_passes","value":1}}

## Results

**Verdict: GREEN — a position-selective tie-break recovers e2e greedy identity 1.0 at ~2.6 TPS, 4.23× cheaper than the ~11-TPS FA_SLIDING=0 blanket.** The required headline fields: `knifeedge_tiebreak_self_test_passes = True` (PRIMARY, **32/32 checks**, ≥20 ✓); `flip_count_at_decode = 1` (reproduces #381 exactly); `flip_in_tightest_band = True`; `tiebreak_recovers_identity_1p0 = True`; `selective_fix_tps_cost = 2.598 TPS` vs `fa_sliding0_tps_cost = 11.0`; `selective_cheaper_than_blanket = True` (ratio **4.23×**); `targeted_identity_recovery_viable = True`. W&B run **`g3954eh3`** (group `knifeedge-identity-tiebreak`). Scope: `analysis_only=True` / `no_hf_job=True` / `no_served_file_change=True` / `official_tps=0`.

**The whole result in one sentence:** the sole decode-width identity residual is a single 0.125-nat argmax coin-flip in which the M=1 reference token is the M=8 **top-2** candidate, so a deterministic tie-break recovers identity 1.0, and engaging the higher-precision attention reduction *only* on the 23.6% of verify steps that are near-ties costs 2.6 TPS — materially under the 11-TPS blanket.

### Primary — knife-edge characterization (pinned arm, PRIMARY, 127 prompts × 8-row decode = 889 positions)
| metric | value | meaning |
|---|---|---|
| **`knifeedge_tiebreak_self_test_passes`** (PRIMARY) | **True** (32/32) | all determinism/arith/geometry/band/cost-model checks pass |
| `pinned identity` | **0.9988751** | byte-for-byte equals #381 `9edps20u` (1 flip/889) |
| `flip_count_at_decode` | **1** | reproduces the #381 single residual flip |
| `knifeedge_position_count_0p125` | **38** (4.27%) | positions with top-2 margin ≤ 0.125 nat |
| `knifeedge_position_count_0p25` | **59** (6.64%) | ≤ 0.25 nat |
| `knifeedge_position_count_0p5` | **95** (10.69%) | ≤ 0.5 nat |
| `max_flip_gap` / `eps_star` | **0.125 / 0.125** | the lone flip sits at the tightest bf16 gap |
| `flip_in_tightest_band` | **True** | the flip lives in the ≤0.125-nat band ✓ |

The heuristic control arm (VLLM_BATCH_INVARIANT=0) likewise reproduces #381: identity **0.9966254** (3 flips/889), bands pos 39/68/103, all 3 flips at 0.125 nat.

### The mechanism — the residual is a pure argmax coin-flip, M1-token = M8-top-2 (this is the whole finding)
Every flip (3 heuristic + 1 pinned, **4/4**) has the identical fingerprint: the M=1 AR reference token is the M=8 **top-2** candidate, exactly **0.125 nat** below M=8's top-1, and is the **lower token-id** of the tied pair.

| arm | M8 top-1 id | M8 top-2 id | M1 ref id | margin | tie-break pick = min-id | recovers |
|---|---|---|---|---|---|---|
| pinned | 102643 | 22355 | **22355** | 0.125 | 22355 | ✅ |
| heuristic | 236743 | 621 | **621** | 0.125 | 621 | ✅ |
| heuristic | 25581 | 3629 | **3629** | 0.125 | 3629 | ✅ |
| heuristic | 8291 | 6481 | **6481** | 0.125 | 6481 | ✅ |

`id_tiebreak_recovers_all_flips = True`. A **stable lowest-token-id tie-break** (resolve any ≤ε near-tie to the lower token id) reproduces the M=1 argmax in **100%** of flips *without consulting M=1* — because at a true 0.125-nat tie the two candidates are FP-indistinguishable and only the (reduction-order-stable) tie ordering decides. This is the deterministic rule the card asked for ("stable lowest-index argmax"), and it collapses the FP coin-flip that the #375 TRITON_ATTN split-reduction injects. `tiebreak_recovers_identity_1p0 = True`; `selective_identity_by_eps = {0.125:1.0, 0.25:1.0, 0.5:1.0}` — identity 1.0 at every band.

### Cost — selective higher-precision attention vs blanket (deliverable #3)
The realizable, **argmax-semantics-preserving** fix: run the fast attention everywhere (base config, free), read the top-2 margin, and re-engage the higher-precision (FA_SLIDING=0 / TRITON pinned) reduction **only** on near-tie verify steps. Cost = `f_step_band(ε) × η_attn`.

| quantity | value | reading |
|---|---|---|
| `f_step_band_at_eps_star` (≤0.125) | **0.23622** | 30/127 verify steps are near-ties (min-gap ≤ 0.125) |
| **`selective_fix_tps_cost`** | **2.598 TPS** | 0.23622 × 11.0 |
| `fa_sliding0_tps_cost` (imported) | **11.0** | the blanket attention strict tax (= η_attn) |
| **`selective_cheaper_than_blanket`** | **True** | 2.60 < 11.0 |
| `cheapness_ratio_blanket_over_selective` | **4.23×** | blanket / selective |
| `f_position_band_at_eps_star` (≤0.125) | **0.04274** | the #364-unit pessimistic eta (full forward/position) |

Idealized downstream implication (NOT a measured TPS, analysis-only): replacing the 11-TPS blanket tax with the 2.6-TPS selective cost would lift the identity-valid base from **471.42** (#390, FA_SLIDING=0) by ~**8.4 TPS** toward the **509.78** band ceiling — closing ~29% of the 28.58 gap_to_500 — *if* the served-path conditional recompute were built (out of scope here).

### Reconciliation with ubel #364 (the closest prior, RED)
`corroborates_364_precision_wall = False` — and this is the honest, two-sided story:
- **#364's RED *conclusion* does NOT reproduce at the decode width.** #364 found selective LOSES (44.6%/17% > 9.841% blanket) because its cost unit was a **full M=1 forward per flagged position** over 512-token trajectories. Here, even in #364's own punitive unit, the decode-width flagged fraction is only **4.27%** — *below* the 9.841% blanket GEMM — because (a) the decode-width geometry has a far lower near-tie base rate (`decodewidth_lowmargin_background_0p5 = 10.69%` vs #364's 17% @ 1.125 nat) and (b) the lone decode-width flip is the tightest-possible 0.125-nat tie, so the catching threshold is tiny.
- **#364's precision *phenomenon* IS still present.** Catching the single flip (0.78% of steps / 0.11% of positions) requires flagging 23.6% of steps / 4.27% of positions — a ~30–44× over-flag. The selective fix wins **not** because the detector is precise, but because the per-engaged-step cost unit (a cheap attention-reduction upgrade) is so much cheaper than #364's full-forward unit that even a 30× over-flag stays well under the blanket.

In short: #381 localized the residual; **#397 shows it is cheaply repairable at the decode width**, refining (not contradicting) #364 — the precision wall is real but the η_attn cost unit and the favorable decode-width geometry put selective decisively under the blanket.

### Determinism controls (signal, not noise)
Both arms: `determinism_M1_vs_M1 = determinism_M8_vs_M8 = within_batch_copy0_vs_copy1 = 1.000000`, `chunk_isolated_fraction = 1.0` (median width 7 = K_spec), `marlin_bitexact_at_decode_width = True` (int4 body byte-exact at size_m=8, confirming the residual is NOT Marlin). Pinned arm: `aten_mm_bitexact = True` + `attn_is_batch_invariant = True` (pin engaged); heuristic arm: both False (expected). ⇒ the 1 flip and the 30 near-tie steps are **deterministic** reduction-geometry effects, not jitter.

### Baseline comparison (per PR)
- Official frontier **481.53 TPS / PPL 2.3772 / 128-of-128 (PR #52, `2x9fm2zx`)** — **UNCHANGED**. This is a local identity micro-measurement: **0 official TPS, no served-file change, no HF job, no submission.** ✅
- Corrected realized strict base **471.42** (#390 `5y64zbjz`); gap_to_500 = 28.58; band ceiling 509.78.
- #381 anchor `9edps20u`: pinned identity 0.998875 — reproduced byte-for-byte.

### Command
```
cd target/ && /workspace/senpai/target/.venv/bin/python \
    research/validity/knifeedge_identity_tiebreak/knifeedge_identity_tiebreak.py \
    --n-prompts 128 --wandb_name "stark/knifeedge-identity-tiebreak" \
    --wandb_group knifeedge-identity-tiebreak
```
Two GPU arms run as isolated subprocesses (CUDA_VISIBLE_DEVICES=0, VLLM_BATCH_INVARIANT∈{0,1}) on the on-target A10G; int4 substrate is the deployed `gemma-4-E4B-it-qat-w4a16-ct` snapshot. Peak GPU **12.24 GB** (pinned) / 12.25 GB (heuristic). `--reanalyze` recomposes the report + self-test from saved `arm_*.json` with 0 GPU.

### What happened
The card's optimistic hypothesis **held**. #381 left exactly one decode-width identity residual; this card shows it is a single 0.125-nat argmax coin-flip where the M=1 token is M=8's top-2, so (i) a deterministic stable-lowest-id tie-break recovers identity 1.0 in 4/4 observed flips, and (ii) the semantics-preserving alternative — selective higher-precision attention on the 23.6% near-tie steps — costs 2.6 TPS, **4.23× under** the 11-TPS blanket. The apparent tension with #364 resolves cleanly: #364's RED was driven by a full-forward-per-position cost unit over inflated 512-token-trajectory base rates; at the literal 8-row decode width the flagged fraction is only 4.27% and the cost unit is the cheap attention upgrade, so selective wins. **Two honest caveats bound the GREEN:** (1) **realizability** — the 2.6-TPS cost is an idealized oracle lower bound; realizing it needs served-path conditional per-step attention recompute (a served-file change, explicitly out of scope), and the near-tie margin must be readable from the fast pre-pass (empirically yes, since the disagreement *is* the 0.125-nat tie); (2) **small-n precision** — n=4 flips is a small sample to assert "M1 = lowest-id always," and the tie-break flags 23.6% of steps to fix one, so the detector is imprecise (it wins on cost-unit economics, not precision). Net: viability and cost are GREEN; the blanket FA_SLIDING=0 tax is not the only route to identity 1.0 at the decode width.

### Suggested follow-ups
1. **Realize and measure the selective recompute (served-path, needs human approval).** Build the two-pass conditional attention (fast everywhere → margin gate → precise recompute on ≤ε steps) behind a flag and measure *actual* TPS vs the 2.6-TPS model and the 11-TPS blanket on the official harness. This is the load-bearing unknown — coordinate with wirbel #393 (`attention-strict-pin-cost`) since it audits whole-backend η_attn; #397 is the position-selective complement.
2. **Stress the tie-break invariant at scale.** Run more prompts / longer suffixes to test whether "M1 = lower token-id of the 0.125-nat tied pair" holds beyond n=4, or whether a stable-lowest-id rule ever mis-resolves (M1 = higher id). If it ever breaks, the rule must read the M1 ordering explicitly (and the free logit-level fix degrades to needing the precise attention).
3. **Logit-level tie-break as a near-free fix (separate scope).** Deliverable #2's tie-break is itself a logit post-process (override argmax to lowest-id on ≤ε near-ties), ~free vs the 2.6-TPS attention recompute — but it changes served argmax semantics. Worth a dedicated card to measure whether it holds identity 1.0 across the full eval without a backend touch.
4. **Tighten ε.** All 4 flips are at exactly 0.125 nat (the bf16 floor); an ε = 0.125 + tol gate already captures 100% recall. f_step shrinks no further below 0.125, so 23.6% is the floor for this geometry — the precision ceiling is set by the 30 natural 0.125-nat near-tie steps, not by ε choice.

### Repro / bug notes
- Run with the serving venv interpreter (`/workspace/senpai/target/.venv/bin/python`, vLLM 0.22.0 + wandb 0.27.2); the default `/usr/bin/python` lacks vllm. GPU phases self-isolate; `--reanalyze` re-logs from saved JSON with 0 GPU.
- Fixed one harness bug pre-run: `log_wandb` read the raw per-arm keys (`marlin_sizem_diag` / `aten_mm_control`) but `compose_and_report` renames them to `marlin_bitexact_at_decode_width` / `aten_mm_bitexact`; corrected and validated end-to-end via `--reanalyze` (commit `0e81083`).

_Public-evidence note: all anchors cited (frontier 481.53 / PPL 2.3772 #52 `2x9fm2zx`; strict base 471.42 #390 `5y64zbjz`; band ceiling 509.78; #381 pinned 0.998875 `9edps20u`; FA_SLIDING=0 η_attn ≈ 11 TPS; #364 precision wall 17% / 9.841% blanket / full-forward-per-position) are reused, not re-derived. This card adds 0 official TPS and changes no served file._
