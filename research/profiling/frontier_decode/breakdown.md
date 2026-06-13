# Frontier decode-step profile — component breakdown

_Submission_: `/workspace/senpai/target/submissions/fa2sw_precache_kenyan`  
_Workload_: conc=1, 128 prompts, output_len 512, CUDA graphs ON  
_Captured_: 2026-06-13T16:56:52Z  
_Local A10G exploratory probe — NOT the official a10g-small TPS._

**E_accept (mean acceptance length)** = **3.817** tokens/cycle (source: `server_log_counters_exact`; TPS×cycle cross-check 3.68)

## Steady-state spec-decode cycle (p50 per-step; means reject prefill + inter-request outliers)

| quantity | ms | note |
|---|---|---|
| drafter forward (GPU) | 1.446 | STEPTIME `kind=draft` |
| verify forward (GPU) | 7.906 | STEPTIME `kind=exec` |
| GPU-busy / cycle | 9.352 | drafter + verify |
| host overhead / cycle | 0.064 | cycle wall − GPU-busy |
| **cycle wall** | **9.416** | verify call + inter-step gap |
| GPU-busy share of wall | 99.3% | **decode is GPU-bound** |
| _(host gap p50 vs polluted mean)_ | 1.598 vs 1.728 | 52 request-switch gaps >3× median |

## Decode GPU-busy composition (share of GPU-busy/cycle)

| component | % of GPU-busy | measured/inferred |
|---|---|---|
| verify body int4-Marlin GEMM | 53.2% | trace − drafter − lmhead |
| verify attention (fa2sw) | 19.6% | trace (direct) |
| drafter forward | 15.5% | STEPTIME (direct) |
| verify norm/elementwise | 6.7% | trace (direct) |
| sampling | 2.6% | trace (direct) |
| verify lmhead12k GEMM | 1.0% | isolation 16k↔12k |

## TPS reconstruction (local A10G probe — not the official a10g-small TPS)

- TPS_reconstructed = E_accept / cycle_wall = **405.4 tok/s** (drafter-inclusive lower bound 351.4)
- measured steady decode TPS (whole-run engine meter) = 391.3 tok/s  ← the honest local number
- warm single-burst probe (≈ #22's ~867) = 861.6 tok/s (overstates steady; reported for #22 continuity)
- ratio recon/steady = 1.036

## lm_head share vs PR #8 (262k base = 26.4% of decode GPU)

- lmhead12k GEMM now **1.0%** of GPU-busy (12288 rows). Isolation est 0.0900 ms, bandwidth-model est 0.0263 ms.
- per-row scaling implies a *full* 262k head would be ~20.5% — i.e. lmhead12k cut the head ~21×, consistent with the drop from 26.4%.

## Next lever

**verify_body_int4_gemm** is the largest addressable component at **53.2%** of decode GPU-busy.
