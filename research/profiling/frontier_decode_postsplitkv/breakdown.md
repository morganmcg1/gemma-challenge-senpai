# Frontier decode-step profile — component breakdown

_Submission_: `/workspace/senpai/target/submissions/fa2sw_precache_kenyan`  
_Workload_: conc=1, 128 prompts, output_len 512, CUDA graphs ON  
_Captured_: 2026-06-13T21:46:41Z  
_Local A10G exploratory probe — NOT the official a10g-small TPS._

**E_accept (mean acceptance length)** = **3.847** tokens/cycle (source: `server_log_counters_exact`; TPS×cycle cross-check 3.65)

## Steady-state spec-decode cycle (p50 per-step; means reject prefill + inter-request outliers)

| quantity | ms | note |
|---|---|---|
| drafter forward (GPU) | 1.445 | STEPTIME `kind=draft` |
| verify forward (GPU) | 6.519 | STEPTIME `kind=exec` |
| GPU-busy / cycle | 7.964 | drafter + verify |
| host overhead / cycle | 0.047 | cycle wall − GPU-busy |
| **cycle wall** | **8.011** | verify call + inter-step gap |
| GPU-busy share of wall | 99.4% | **decode is GPU-bound** |
| _(host gap p50 vs polluted mean)_ | 1.606 vs 1.712 | 46 request-switch gaps >3× median |

## Decode GPU-busy composition (share of GPU-busy/cycle)

| component | % of GPU-busy | measured/inferred |
|---|---|---|
| verify body int4-Marlin GEMM | 60.6% | trace − drafter − lmhead |
| drafter forward | 18.1% | STEPTIME (direct) |
| verify attention (fa2sw) | 7.6% | trace (direct) |
| verify norm/elementwise | 7.5% | trace (direct) |
| sampling | 2.9% | trace (direct) |
| verify lmhead12k GEMM | 0.3% | isolation 16k↔12k |

## TPS reconstruction (local A10G probe — not the official a10g-small TPS)

- TPS_reconstructed = E_accept / cycle_wall = **480.2 tok/s** (drafter-inclusive lower bound 406.8)
- measured steady decode TPS (whole-run engine meter) = 455.3 tok/s  ← the honest local number
- warm single-burst probe (≈ #22's ~867) = 886.2 tok/s (overstates steady; reported for #22 continuity)
- ratio recon/steady = 1.055

## lm_head share vs PR #8 (262k base = 26.4% of decode GPU)

- lmhead12k GEMM now **0.3%** of GPU-busy (12288 rows). Isolation est 0.0000 ms, bandwidth-model est 0.0263 ms.
- per-row scaling implies a *full* 262k head would be ~7.0% — i.e. lmhead12k cut the head ~21×, consistent with the drop from 26.4%.

## Next lever

**verify_body_int4_gemm** is the largest addressable component at **60.6%** of decode GPU-busy.
