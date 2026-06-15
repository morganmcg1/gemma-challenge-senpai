# PR #275 — Prefill / TPS-denominator slack probe

**PRIMARY `prefill_denominator_self_test_passes` = True**  
**TEST `prefill_wall_share_pct` = 2.849%** (of e2e, at `precache_on_512`)  
**`prefill_lever_material` = True** (optimistic upper 2.849% vs 2.0% gate) · supported-edge material = False

> **Verdict:** DENOMINATOR ESSENTIALLY CLOSED. Prefill is 2.85% of wall at the deployed (precache-on, 512-token) operating point. The recoverable band is [0.00%, 2.85%]: the upper edge clears the 2% gate ONLY under the physically-unreachable assumption of eliminating 100% of residual prefill, while the supported lower edge is ~0 because every standard prefill lever (precache warmup, vLLM prefix caching, chunked + cudagraphed prefill) is ALREADY deployed (precache alone already banks 1.65 pct-points). Decode (~97%) is the sole remaining TPS front.

## Prefill wall share at the deployed (precache-on) operating point

| basis | prefill share |
|---|---|
| of e2e | 2.849% |
| of inference (prefill+decode) | 2.865% |
| of client wall | 2.843% |

## What the deployed precache/prefix-cache already banks

- prefill share **precache OFF** = 4.498%
- prefill share **precache ON** = 2.849%
- **recovered by precache = 1.65 pct-points** (prefill_sum 6.11s -> 3.816s)

## Prefill-phase decomposition (precache_off basis; valid partition)

| sub-component | share | seconds |
|---|---|---|
| target_prefill | 88.1% | 6.113s |
| drafter_prefill | 0.0% | 0.000s |
| tokenize | 1.0% | 0.069s |
| scheduler_plumbing | 10.9% | 0.759s |
| **sum** | 100.0% | partition_valid=True |

- **MTP drafter marginal prefill ≈ 0** (negligible=True): an independent spec-off run prefills the target alone in 7.430s ≳ the spec-on combined prefill 6.113s, so the raw subtraction is -1.317s (≤0) — the recurrent MTP drafter reuses the target's prompt hidden states, so it adds no measurable prompt prefill.

## Decode-side consistency (validates the phase split)

- E_accept measured = 4.069642193883867 (physical band (2, 5.207))
- decode throughput (Σout/Σdecode, official meter) = 506.5 tok/s vs official 481.53 TPS (+5.2%, reproduces=True)
- e2e throughput = 489.3 tok/s; per-token decode = 1.974 ms
- deployed verify step p50 = 6.53 ms (NB banked 1.2182 ms is the hypothetical depth-9 *tree* roofline step, not the deployed linear-MTP step — see verify_step_note)

## Self-test

- a_walltime_identity_holds: **True**
- b_decode_reproduces_served_step: **True**
- c_prefill_partition_sums_to_one: **True**
- d_nan_clean_all_finite: **True**
- e_recoverable_band_reported: **True**
- identity residual (inference − prefill − decode) = 0.0000s (ε=0.5s)

## Greedy/PPL-safety certificate

`prefill_probe_analysis_only = True`. Timing-only forward over the standard prompt set; no served-file change, no emitted-token change, no HF Job, no submission. BASELINE 481.53 TPS and the λ=1 ceiling 520.953 unchanged (this leg adds 0 TPS).
