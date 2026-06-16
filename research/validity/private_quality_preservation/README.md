# Private-distribution quality preservation (PR #513)

**Is the spec-alive surgical-357 PRIVATE leaderboard acceptance shift a pure
SPEED risk, or does it carry downstream-quality exposure?**

The QUALITY twin of denken #489 / kanna #504, #508, which priced the **SPEED**
side of the private acceptance shift. This card prices the **QUALITY** side and
extends denken #505 (`bg03bq0d`, public-distribution preservation) from the
public anchor down through the private-breach acceptance regime.

**Verdict: PURE SPEED RISK / ZERO QUALITY EXPOSURE.** W&B run
[`krma4lm7`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/krma4lm7).

## Mechanism (verified from the pinned kernel source)

Deployed serve = surgical-357 (PR #499), spec-alive MTP K=7, pinned vLLM
`0.22.1rc1.dev307+g3e8afdf78` (`/tmp/server-venv`). Under temperature sampling
(`do_sample:true, T=1.0, top_k=64, top_p=0.95`) the rejection sampler runs the
stock random/recovered kernels (the dixie patch only short-circuits the
`all_greedy` temp=0 path — `rejection_sampler.py:126-188`, skipped here).

With a deterministic greedy MTP draft `x_d` and `draft_probs=None`
(`NO_DRAFT_PROBS`, `rejection_sampler.py:575`), per draft position:

- **accept** `x_d` with probability `min(1, p(x_d)) = p(x_d)` (`L913-926`, `draft_prob=1`),
- **on reject** resample from the recovered distribution = `p` masked to exclude
  the draft token, i.e. `p|{y != x_d}` renormalized (`L1006-1011`), via the stock
  exponential-race Gumbel-max (`L758,765`).

This is **exactly distribution-preserving: output ~ p for ANY draft token**. The
realized acceptance at a position is exactly `p(x_d)`. So "draft quality /
acceptance" is just *which token is drafted* — a worse (private-OOD) drafter
proposes lower-`p(x_d)` tokens (lower acceptance, more resamples) but the output
distribution is **invariant**. Therefore the private acceptance breach moves
`E[T]`/TPS only, with zero effect on the output distribution.

## Method

`specdec_acceptance_invariance.py` drives the **exact deployed** `rejection_sample()`
kernel (no serve, no submission, `analysis_only=true`, `official_tps=0`).

- **Leg 1 — single-position acceptance sweep.** For FIXED target dists, draft
  every support token => realized acceptance spans the public anchor (~0.387)
  down through the private breach band (≤0.10, to 0.003). The iid noise floor is
  held FIXED (same `p`, same `N=M` emit/trial), so TV staying pinned at the floor
  across the whole acceptance range **is** acceptance-rate-invariance.
  Corroborated on the real #497 reasoning answer dists (128 cases, vocab 16384).
- **Leg 2 — K=7 multi-position chaining.** Depth-isolated probe (always-accept
  prefix => full statistics at every spine depth 0..6) + a natural greedy chain.
  Confirms preservation does NOT accumulate error across the 7 MTP positions.

### Statistical methodology (the fix over the smoke harness)

"At the floor" is defined against a **redraw band**, not a single iid draw. For
each case we redraw the iid floor `R=160` times → `(mu, sd)` and a high quantile
`hi`. Under the exact-preservation null, TV_deployed at every acceptance level is
just another draw from this band, so:

- `z = (TV_deployed − mu)/sd` is O(1) under the null; a **real** shift `delta`
  gives `z ~ sqrt(N)` (hundreds at N≥50k) — the band cleanly separates a genuine
  distribution shift from finite-sample noise.
- **Acceptance-invariance** is the pooled, floor-normalized correlation
  `corr(realized_accept, z) ~ 0` (a per-case TV-vs-accept slope is ill-conditioned
  when a case's acceptance range is narrow; the pooled correlation is robust).
- **No systematic bias**: mean signed excess `(TV_deployed − mu) ~ 0`.
- **M-independent corroboration**: a per-point G-test (likelihood-ratio GOF,
  Wilson–Hilferty p-value) of the deployed histogram vs `N·p`. Under exact
  preservation p-values are ~Uniform(0,1); we report the global Bonferroni min-p
  and the fraction with `p>0.05`.

> The earlier smoke verdict flagged "quality exposure" purely as an artifact of a
> single-draw floor: `max_d TV_deployed − (one floor draw)` is upward-biased and
> crossed an absolute 0.01 tol on noise alone. The mean signed excess over 128
> real cases was already ~0.0002 (53% positive) — no real deviation. The band
> fixes this.

## Key results (full run: synthetic M=300k, real M=50k, R=160, B=300k)

| metric | value |
| --- | --- |
| verdict | **PURE SPEED RISK / ZERO QUALITY EXPOSURE** |
| `quality_acceptance_invariant` | **true** |
| `private_quality_exposure` | **0.0** |
| `max_tv_across_acceptance_sweep` | 0.0114 (raw, at the finite-M floor) |
| mean iid noise floor | 0.00295 |
| `max_z_over_floor` (590 pts) | 4.04 (< K_SIGMA=6); **0** band exceedances |
| mean signed excess over mu | 9.7e-05 (≈0, no bias) |
| `corr(accept, z)` | −0.061 (\|r\|<0.10 ⇒ acceptance-invariant) |
| G-test global Bonferroni p | 1.0 (not significant) |
| G-test fraction p>0.05 | 0.941 (uniform-p signature ⇒ output==p) |
| `max_tv_over_k7_positions` (2..7) | 0.0069; depth slope −1.7e-05 |
| `k7_no_accumulation` | **true** |
| self-tests | 6/6 |
| peak GPU mem | 13.11 GB |

The **private-bracket ladder** is the direct demonstration: as acceptance descends
0.388 → 0.226 → 0.101 → 0.050 → 0.020 → 0.010 → 0.003 (public anchor into the deep
private breach band), TV stays flat at 0.0011–0.0019, every point within band,
every G-test `p>0.05`.

## Run

```bash
cd research/validity/private_quality_preservation
CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  /tmp/server-venv/bin/python specdec_acceptance_invariance.py --self-test
CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  /tmp/server-venv/bin/python specdec_acceptance_invariance.py \
    --M 300000 --M-real 50000 --B 300000 --R 160 \
    --real-logits ../specdec_quality_preservation/real_reasoning_p.pt --out results.json
CUDA_VISIBLE_DEVICES="" /tmp/server-venv/bin/python log_wandb.py --results results.json
```

Reuses the real #497 reasoning answer distributions from denken #505
(`../specdec_quality_preservation/real_reasoning_p.pt`).
