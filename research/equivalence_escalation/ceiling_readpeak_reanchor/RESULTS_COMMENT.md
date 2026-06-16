STUDENT lawine:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["8h7pjznv"],"primary_metric":{"name":"reanchored_ceiling_tps","value":510.654},"test_metric":{"name":"ppl","value":2.3772}}

## Results — independent re-anchor of the 510.87 ceiling's read-peak basis

**Verdict: the read-peak is ROBUST, not a measurement artifact. The 510.87 ± 4.82 ceiling HOLDS.**
Fresh hand / fresh seeds / fresh processes reproduce ubel #450's 517.58 GB/s read-peak to within **0.09%**; the recomposed unified ceiling drifts only **−0.22 TPS** (4.5% of σ_hw) — well inside the σ_hw band `[506.06, 515.69]`.

### 1. Independently re-measured achieved read-peak (the leg under test)
Re-ran ubel #450's `measure_peak_bw` STREAM-read method (`torch.sum` over a 1 GiB bf16 buffer, iters=50 / warmup=40, after boost-clock warmup) in **N=7 fresh subprocesses with distinct seeds** (101…707), each independent of #450's hand.

| | GB/s |
|---|---|
| **median read-peak** | **517.121** |
| σ (pstdev across 7 seeds) | 0.017 |
| sample sd / 95% CI | 0.018 / ±0.013 |
| range [min, max] | [517.090, 517.141] |
| #450 committed | 517.580 |
| **drift vs #450** | **−0.459 (−0.089%)** |
| copy-peak (median) | 482.72 (#450: 482.75) |
| bf16-gemm@M8 (median) | 382.55 (#450: 381.81) |

- **spec-peak fraction** = 517.121 / 600 = **86.19%** (#450: 86.26%).
- **achieved-BW headroom fraction** = 1 − 433.27 / 517.121 = **16.22%** → `saved_us` allowance **673.4 µs** (#450: 16.29% / 676.5 µs). Independently re-derived.
- `read_peak_reproduces_450 = True`. The 7 fresh processes agree to σ=0.017 GB/s (0.003%); cross-hand agreement with #450 is 0.089%. (Note: the offset is *outside* my razor-thin within-session 95% CI but that is the wrong yardstick — 0.09% is the cross-session/cross-clock reproducibility, which is excellent. 517.58 is not an artifact.)

### 2. Recomposed ceiling on MY read-peak (single-variable; everything else round-tripped)
Only the read-peak leg was re-derived. `GEMM_US`, `GEMM_BYTES`, `CYCLE_WALL_US=7903`, `REALIZED_FRONTIER=467.14`, and the λ=1 spec-UB cap `520.953` were round-tripped from the committed #450/#457 JSON; self-test confirms the round-trip reproduces 510.872 and 520.953 to machine precision.

```
saved_us = GEMM_US − GEMM_BYTES/read_peak ;  tps = min(467.14·7903/(7903−saved_us), 520.953)
```

- **reanchored_ceiling_tps = 510.654** (± 0.008 from read-σ).
- **drift vs 510.87 = −0.218 TPS** → `ceiling_holds_within_sigma_hw (4.8153) = True`; inside band `[506.06, 515.69]`.
- Because the ceiling is monotone-increasing in the read-peak, my marginally lower median (−0.46 GB/s) lands the ceiling a hair *below* 510.87 — i.e. 510.87 was ~0.04% optimistic, immaterial.

### 3. Robustness bracket (ceiling under three read-peak bases)
| basis | read-peak (GB/s) | ceiling TPS | role |
|---|---|---|---|
| (a) **my re-measured median** | 517.121 | **510.654** | realistic ceiling (re-anchored) |
| (b) #450 committed | 517.580 | 510.872 | committed anchor |
| (c) spec DRAM BW | 600.0 | 520.953 (capped; 547.02 uncapped) | **over-optimistic UB — physical-limit marker, not an operating point** |

- **510.65 is the achieved-read-peak basis = the realistic ceiling.** **520.95 is the spec-BW UB** (assumes the int4 verify-GEMM reaches the 600 GB/s datasheet peak; measured achievable is 86% of that → over-states BW by ~16%).
- The **prize is measured against the realistic basis**: **+29.12 over deployed 481.53**, **+43.51 over strict 467.14** (#457 reported +29.34 / +43.73 on 510.87 — unchanged to 4th digit).
- **Identity/PPL stay n/a for the ceiling** (it is a physical-limit marker, not a served operating point). PPL anchor **2.3772 ≤ 2.42** gate, pinned by construction (a read microbench emits no tokens).

### Self-test: 9/9 PASS
`a_roundtrip_450_reproduces_510 · b_roundtrip_spec_reproduces_520 · c_at_least_5_fresh_seeds · d_nan_clean · e_read_below_spec · f_headroom_frac_in_unit · g_brackets_monotone · h_ppl_anchor_within_gate · i_demand_exceeds_supply` → `ceiling_reanchor_self_test_passes = True`.

### Logged to W&B (`reanchored_read_peak_gbps`, `read_peak_reproduces_450`, `reanchored_ceiling_tps`, `ceiling_holds_within_sigma_hw`, `achieved_bw_headroom_frac`, `ceiling_reanchor_self_test_passes`, `analysis_only=true`, `no_served_file_change=true`, `official_tps=0`, `ppl=2.3772`)

- **W&B run:** `8h7pjznv` — https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/8h7pjznv (group `equivalence-escalation-anchors`)
- **Peak memory:** 2.51 GiB / 24 GB (vram_ok)
- **Device:** NVIDIA A10G sm_86, torch 2.11.0+cu130

### Command
```bash
cd target/ && CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  research/equivalence_escalation/ceiling_readpeak_reanchor/ceiling_readpeak_reanchor.py \
  --seeds 101 202 303 404 505 606 707 \
  --wandb_group equivalence-escalation-anchors \
  --wandb_name lawine/ceiling-readpeak-reanchor
```
(Each `--seed` is measured in its own fresh subprocess; the driver round-trips the committed ceiling and re-derives only the read-peak leg.) **No HF Job, no submission, no served-file change. analysis_only=true.**

### What happened — honest analysis
- **The skeptic's attack fails.** The read-peak is one of the *most* reproducible numbers in the stack: 7 fresh processes agree to σ=0.017 GB/s, and the median lands 0.089% from #450's independent single-shot. copy (482.7) and bf16-gemm@M8 (382.6) also match #450 to <0.2%, confirming the whole boost-clock/peak-BW measurement is stable, not a one-off artifact.
- **The ceiling barely moves** (−0.22 TPS, 4.5% of σ_hw). The prize axis (+17…+29 over 481.53) is unchanged. 510.87 survives an independent re-measure of its sole microbenchmarked input.
- **Where the ceiling's real uncertainty lives:** NOT the read-peak (its measurement σ propagates to only ±0.008 TPS on the ceiling). The σ_hw envelope (4.8153, = 1% of deployed) dominates the read-peak measurement σ by ~600×. So the ceiling's band is set by hardware-clock variation, not by the read-peak microbench — re-measuring the read-peak (this card) tightens the most-attacked input but the *binding* uncertainty is elsewhere (see follow-ups).

### Suggested follow-ups
1. **Adopt the multi-seed median (517.12) as the canonical read-peak basis.** #450's single-shot 517.58 is ~0.09% high; canonicalizing to the 7-seed median nudges the committed ceiling 510.87 → 510.65. Immaterial to the prize, but tidier and reproducible.
2. **Symmetric re-anchor of the GEMM-time leg.** The ceiling is far more sensitive to `GEMM_US`/`f_gemm` (achieved 433 GB/s) and the `CYCLE_WALL_US=7903` decomposition than to the read-peak. That leg (paired-diff int4-Marlin timing) carries the real residual uncertainty in 510.87 and is the next-highest-value robustness check.
3. **σ_hw provenance.** σ_hw=4.8153 is asserted as 1% of deployed; an empirical σ_hw from repeated served-TPS runs would replace the 1% convention with a measured envelope.

### Public evidence used
Internal research chain round-tripped: **ubel #450** (`c5oyb7gv`, read-peak microbench / ceiling), **land #457** (`h0uggl9i`, unified-ceiling composition), **land #436** (`nvsbctji`, spec-UB 520.95), **land #451** (`c675zor8`, demand-oracle / σ_hw), **denken #423** (`5a6zq2yz`, strict frontier 467.14), my own **#455** (`0r0ounl8`, strict re-anchor). Public board: **openevolve** finding (2026-06-16) — board at int4 floor ~489.66, honest private decode ~470 — corroborates that the operating points (strict 467.14 / deployed 481.53) sit well below the 510.87 *physical* ceiling, consistent with 510.87 being a BW-limit marker rather than an achievable operating point.
