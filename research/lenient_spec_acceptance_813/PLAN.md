# PR #813 Step-2 — Synthetic-acceptance ceiling oracle (advisor probe)

Status: ACTIVE (local A10G, EXPLORATORY, **NO HF job**). Picked up advisor
follow-up 2026-06-20T18:14Z.

## What the advisor asked (the last probe on the acceptance axis)
Step-1 was a viability null (no servable lenient knob; vLLM 0.22 hard-wires
exact-argmax-match for greedy spec-accept). The advisor wants ONE cheap oracle
before closing the axis: use vLLM's `rejection_sample_method=synthetic` (which
*imposes* a chosen accept rate in the greedy branch under temp=0) as a **timing
model** for "what if E_accept were higher". Emitted tokens are garbage (NOT
shippable, NOT a quality run) but the decode loop runs identical
drafter+verify+lm_head kernels, so TPS-vs-accept-rate is a faithful speed ceiling.

- Serve int4head with `rejection_sample_method=synthetic`, sweep the imposed
  rate over **{0.56, 0.70, 0.85, 1.00}** (flat per-position list `[r]*K`, K=6).
  0.56 ≈ current E_accept 3.38 / K=6 — the sanity anchor (should reproduce ~256).
- conc=1, output_len=512, temp=0 (the benchmark decode condition). W&B group
  `bi0-int4head-accept-oracle`. Report TPS + mean-accept-length per rate + the
  rate=1.00 ceiling.

### Decision rule (advisor)
- ceiling@rate=1.00 **> ~282 TPS (>~10% over 256.74)** → real headroom →
  GREENLIGHT a separate custom greedy-kernel top-k-match PR (k env var, shipped
  as a `*_patch.py`), with its own quality panel + greedy-waiver under #784.
- ceiling **< ~270 TPS (<~5%)** → acceptance subspace exhausted → CLOSE it;
  point next lane off-axis (47.6% body verify-GEMM).

## Synthetic-rate semantics (vLLM 0.22 `config/speculative.py:197-254`)
`synthetic_acceptance_rates` = per-position UNCONDITIONAL rates, length K=6,
each in [0,1], monotonically non-increasing. Position i = P(first i+1 draft
tokens all accepted). Flat list `[r]*6` ⇒ N_accepted ∈ {0,6} with mean = 6·r;
mean accept length = 6r+1. At temp=0 the synthetic greedy branch
(`rejection_sampler.py:737-742`) accepts the DRAFT token when
`uniform < precomputed_rate`, else emits target_argmax. TPS ≈ (1+mean_N)/T_step,
so mean_N (=6r) is the driver; the oracle measures how much of that analytic
gain actually realizes vs fixed per-token overhead.

## Key facts found on pickup
- **256.74 is a RECONSTRUCTION, not a served number** (W&B `9tcygwjf` =
  ubel/bi0-lmhead-bytes): base_model_id = `w4a16-ct` (int4 body, **bf16 tied
  head**); measured `tps_bf16_control=219.34`; int4-head TPS projected by
  swapping bf16 head-GEMV 2.777 ms/tok → int4 0.7496 ms/tok ⇒ 256.74. The int4
  head fires per ACCEPTED token, so a cheaper head AMPLIFIES the acceptance gain
  → serving the REAL int4head (not bf16-control) is required for a faithful
  ceiling, and is the first true int4head serve (also validates/corrects 256.74).
- **CUDA_VISIBLE_DEVICES=1 inherited** (only GPU index 0 exists) ⇒ torch sees 0
  GPUs. MUST force `CUDA_VISIBLE_DEVICES=0` on every serve/benchmark command.
- **Disk:** `/` 14 GB free (99% used). Cached: `w4a16-ct` 11 GB (NOT needed to
  serve self-contained int4head), drafter 152 MB. int4head model =
  **10.58 GB** (HF access confirmed, repo
  `gemma-challenge/gemma-4-e4b-it-int4-mtp-bi0-int4head` rev ad42984).
  Plan: download int4head → free redundant `w4a16-ct` cache (re-downloadable) →
  serve with ~14 GB headroom.

## Harness
Reuse the official serve path (`submissions/int4_mtp_bi0_int4head/serve.py`,
which applies the force-2D attn + attn-group patches via PYTHONPATH) + the
official `decode_outputs.py` 128×512 temp=0 workload, parsing steady decode TPS
+ vLLM SpecDecoding accept metrics. serve.py gets an env-gated synthetic
passthrough (`REJECTION_SAMPLE_METHOD`/`SYNTHETIC_ACCEPTANCE_RATES`), defaults
OFF so shipped behavior is byte-identical.
