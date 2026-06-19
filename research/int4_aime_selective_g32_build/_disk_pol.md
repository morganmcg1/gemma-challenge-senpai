STUDENT ubel: disk heads-up (honoring my 07:24 commitment to ping below ~8G) + proof-of-life. **No action needed — bounded and self-resolving.**

**Disk: 1.1G free (below the ~8G flag).** This is the *expected bounded floor*, not a leak: the 17G `fq_selective` bf16 checkpoint is on disk only for the duration of the final eval arm. `run_sweep.sh` `rm -rf`s it the instant seed4 completes (KEEP_BUILD=0), which frees ~17G back to ~18G. Remaining writes during seed4 are <1MB (result JSON + serve log), so **no ENOSPC risk** to the running eval.

**Selective (decision) arm live and healthy.** Run https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/nqk9izab (group `int4-aime-selective-g32-build-ubel`). Seeds 0-3 done; **seed 4/5 running, 26/60 returned**, eval process alive (~9 min in), ETA **~09:09Z**. `analysis_only=1, official_tps=0, no_hf_job=1, fires=0` — no HF Job / submission.

**Interim 4-seed read (NOT final — seed4 pending):**

| arm | seeds | pooled | n_corr | per-seed |
|---|---|---|---|---|
| full_g128 | 5 | 0.3033 | 91/300 | 0.250 / 0.283 / 0.333 / 0.367 / 0.283 |
| full_g32 | 5 | 0.3867 | 116/300 | 0.400 / 0.300 / 0.417 / 0.417 / 0.400 |
| selective | 4 | 0.3458 | 83/240 | 0.367 / 0.350 / 0.283 / 0.383 |

Interim `recovery_fraction` = (0.3458−0.3033)/(0.3867−0.3033) ≈ **0.51**, int4-scale projection ≈ 0.393 < 0.420 → tracking **PARTIAL** (below the rf≥0.802 clear threshold). Seed4 + propagated-CI aggregation will finalize it; full 3-arm verdict comment to follow.
