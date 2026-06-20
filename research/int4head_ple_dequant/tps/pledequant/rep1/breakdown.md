# Frontier decode-step profile — component breakdown

_Submission_: `/workspace/senpai/target/submissions/int4_mtp_bi0_int4head_pledequant`  
_Workload_: conc=1, 128 prompts, output_len 512, CUDA graphs ON  
_Captured_: 2026-06-20T17:20:34Z  
_Local A10G exploratory probe — NOT the official a10g-small TPS._

**E_accept (mean acceptance length)** = **3.383** tokens/cycle (source: `server_log_counters_exact`; TPS×cycle cross-check 0.00)

## Steady-state spec-decode cycle (p50 per-step; means reject prefill + inter-request outliers)

| quantity | ms | note |
|---|---|---|
| drafter forward (GPU) | 0.000 | STEPTIME `kind=draft` |
| verify forward (GPU) | 0.000 | STEPTIME `kind=exec` |
| GPU-busy / cycle | 0.000 | drafter + verify |
| host overhead / cycle | 0.000 | cycle wall − GPU-busy |
| **cycle wall** | **0.000** | verify call + inter-step gap |
| GPU-busy share of wall | nan% | **decode is HOST-bound** |
| _(host gap p50 vs polluted mean)_ | 0.000 vs 0.000 | 0 request-switch gaps >3× median |

## Decode GPU-busy composition (share of GPU-busy/cycle)

| component | % of GPU-busy | measured/inferred |
|---|---|---|
| drafter forward | 0.0% | STEPTIME (direct) |
| verify body int4-Marlin GEMM | 0.0% | trace − drafter − lmhead |
| verify lmhead12k GEMM | 0.0% | isolation 16k↔12k |
| verify attention (fa2sw) | 0.0% | trace (direct) |
| verify norm/elementwise | 0.0% | trace (direct) |
| sampling | 0.0% | trace (direct) |

## TPS reconstruction (local A10G probe — not the official a10g-small TPS)

- TPS_reconstructed = E_accept / cycle_wall = **nan tok/s** (drafter-inclusive lower bound 0.0)
- measured steady decode TPS (whole-run engine meter) = 265.9 tok/s  ← the honest local number
- warm single-burst probe (≈ #22's ~867) = 576.8 tok/s (overstates steady; reported for #22 continuity)
- ratio recon/steady = 0.000

## lm_head share vs PR #8 (262k base = 26.4% of decode GPU)

- lmhead12k GEMM now **0.0%** of GPU-busy (12288 rows). Isolation est 0.0000 ms, bandwidth-model est 0.0263 ms.

## Next lever

**drafter_forward** is the largest addressable component at **0.0%** of decode GPU-busy.
