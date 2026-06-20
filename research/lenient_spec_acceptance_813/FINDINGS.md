# PR #813 — Lenient/typical spec acceptance: Step-1 viability gate

**Verdict: VIABILITY NULL — the verifier-acceptance *criterion* axis is closed on
the pinned stack (vLLM 0.22.0 + `gemma4_mtp` + greedy benchmark).** No
config-exposed lenient/typical acceptance mode serves for this stack. Per the
PR's Step-1 STOP condition, no sweep is run.

This is a source-level audit of the installed engine
(`target/.venv/lib/python3.11/site-packages/vllm`, version 0.22.0, confirmed via
`vllm-0.22.0.dist-info/METADATA`). EXPLORATORY only — no HF job, no local run.

## The hypothesis this kills

#784 made byte-identity non-sacred (gate = quality within 5% of base), which in
principle legalizes accepting a draft token that is *not* the verifier's greedy
argmax — e.g. accept if the draft falls in the verifier's top-k or above a
probability/entropy threshold. Higher acceptance ⇒ longer mean accepted length ⇒
the dominant body verify-GEMM (47.6%) amortized over more output tokens ⇒ higher
TPS. The question for Step 1 is purely: **does the pinned engine expose such a
relaxed acceptance criterion that actually serves for `gemma4_mtp`?**

## Decisive findings (4 facts, all from installed source)

### 1. The benchmark decodes greedy (temperature = 0.0)
Both the speed/decode harness and the PPL harness send `temperature: 0.0` with
no `top_k`/`top_p`:
- `official/main_bucket/shared_resources/speed_benchmark/scripts/decode_outputs.py:222`
  (and the recorded request at `:297`): `"temperature": 0.0`.
- `.../scripts/ppl_endpoint.py:134`: `"temperature": 0.0`.

⇒ `SamplingMetadata.all_greedy == True` at serve time.

### 2. The greedy acceptance kernel hard-wires EXACT-ARGMAX-MATCH
`vllm/v1/sample/rejection_sampler.py`:
- `rejection_sample(...)` short-circuits for greedy: after running the greedy
  kernel, `if sampling_metadata.all_greedy: return output_token_ids` (line
  465–466). The random/"standard" distribution-matching kernel is **never
  reached** at temp=0.
- The greedy kernel's real (non-synthetic) accept branch (lines 743–745):
  ```python
  token_id = target_argmax_id
  rejected = draft_token_id != target_argmax_id
  ```
  A draft token is accepted **iff it equals the verifier's greedy argmax**. There
  is no top-k window, no probability/entropy threshold, no `posterior_alpha`.
  This is exactly the strict criterion the PR hoped to relax, and it is not
  parameterized.

The submission's own `serve.py:8–10` already documents this:
> "At temperature=0 vLLM's rejection sampler short-circuits to target-argmax…"

### 3. No lenient/typical acceptance sampler exists anywhere in the install
`grep -rE "TypicalAcceptance|posterior_threshold|posterior_alpha|acceptance_method|typical_accept"`
over the entire `vllm/` package → **0 matches**. The v0-engine
`TypicalAcceptanceSampler` (Medusa-style typical acceptance: accept if draft prob
≥ `min(posterior_threshold, posterior_alpha·sqrt(entropy))`) was **dropped in the
v1 engine**. There is no class, config field, or CLI flag to enable it.

### 4. The only `rejection_sample_method` options are `{"standard","synthetic"}` — neither is a quality-preserving leniency knob
`vllm/config/speculative.py:69`: `RejectionSampleMethod = Literal["standard","synthetic"]`
(default `"standard"`, `speculative.py:191`).
- **`"standard"`** = probabilistic rejection sampling. Its greedy path is the
  exact-match kernel above; its random path (`rejection_sampler.py:810`,
  `accepted = draft_prob > 0 and target_prob/draft_prob >= uniform_prob`) is the
  Leviathan/Chen rule that provably reproduces the **target's own**
  distribution. It is distribution-preserving, not a tunable leniency threshold —
  it cannot raise E_accept beyond what the drafter's proposal quality earns, and
  it does not run at temp=0 anyway.
- **`"synthetic"`** is NOT a servable quality mode. Its greedy branch
  (`rejection_sampler.py:737–742`) accepts the **draft** token whenever
  `uniform_prob < precomputed_rate` **regardless of whether it matches the target
  argmax**, else emits target_argmax. It exists to *impose* a chosen acceptance
  length (`synthetic_acceptance_length` / `synthetic_acceptance_rates`,
  `speculative.py:197–254`) decoupled from a real drafter — a profiling/simulation
  aid. By construction it emits wrong tokens at the configured rate, so it would
  blow PPL and the generated-text quality panel. It is not a quality-bounded
  acceptance criterion.

## Conclusion

There is no servable, config-exposed lenient/typical verifier-acceptance
criterion for `gemma4_mtp` on vLLM 0.22.0 at the benchmark's greedy decode
setting. The leniency the PR proposes would require **patching the rejection
sampler** (re-introducing a typical-acceptance kernel or a top-k-match greedy
branch) — a custom vLLM patch, not a config knob, and out of scope for a Step-1
cheap viability gate. The STOP condition is met.

## Acceptance-frontier map — now complete
| Axis | Lever | PR | Verdict |
|---|---|---|---|
| Depth | K (num_speculative_tokens) | #774 | K=6 is the knee |
| Pool | CENTROID_TOP_K candidate width | #792 | net-zero TPS + breaks 128/128 |
| Topology | tree / EAGLE | #799 | vLLM 0.22 MTP is single-linear-chain |
| **Threshold** | **verifier accept criterion** | **#813** | **CLOSED — no servable lenient knob (this audit)** |

## What it would take (NOT in scope; for follow-up framing only)
A real lenient-accept experiment on this stack requires a custom greedy-kernel
patch: replace `rejected = draft_token_id != target_argmax_id` with a top-k-match
test (`draft ∈ topk(target_logits, k)`) gated behind a new env var, shipped as a
submission patch like the existing `vllm_attn_group_patch.py`. That is a
runtime-kernel change with direct PPL/quality risk and would need its own
research pass + greedy-identity-waiver framing under #784.

---

# PR #813 — Step-2: synthetic-acceptance ceiling oracle (advisor follow-up)

**Verdict: GREENLIGHT.** The acceptance axis has REAL headroom on int4head. At
the imposed-accept ceiling (r=1.00, every draft token accepted) local conc=1
decode reaches **471.1 steady TPS / 506.8 wall TPS** — far above the advisor's
+10% greenlight band (>282 absolute). A separate custom greedy-kernel
top-k-match accept-branch PR is warranted (its own quality panel +
greedy-identity-waiver under #784).

## What this probe is
Step-1 closed the *config* axis (no servable lenient knob). Before closing the
acceptance axis *entirely*, the advisor asked for one cheap timing oracle: use
vLLM's `rejection_sample_method=synthetic` to **impose** a chosen accept rate in
the greedy branch and measure the resulting TPS. Emitted tokens are GARBAGE
(synthetic accepts the draft whenever `uniform < rate` regardless of argmax
match — see Step-1 finding #4) so this is **NOT a quality run and NOT
shippable**. But the decode loop runs the identical drafter→verify→lm_head
kernels, so TPS-vs-accept-rate is a faithful **speed ceiling** for "what if
E_accept were higher". It answers go/no-go on whether chasing acceptance is even
worth a kernel patch — it does NOT claim any token is correct.

## Setup
- Submission `int4_mtp_bi0_int4head` (int4 W4A16 g32 Marlin body + untied int4
  g32 lm_head), served via its own `serve.py` with the env-gated synthetic
  passthrough (`REJECTION_SAMPLE_METHOD=synthetic`,
  `SYNTHETIC_ACCEPTANCE_RATES=[r]*6`). **The passthrough defaults OFF — the
  shipped leaderboard serving path is byte-identical** (no synthetic keys in
  `spec_config` unless these env vars are set).
- Workload: official `decode_outputs.py`, 128 prompts × 512 output tokens,
  conc=1, temp=0, `ignore_eos=True` (65 536 completion tokens/rate) — the exact
  benchmark decode condition.
- Sweep `r ∈ {0.56, 0.70, 0.85, 1.00}`, flat per-position list, K=6. r=0.56 ≈
  current served E_accept (3.36 mean accepted draft tokens / K=6) = the sanity
  anchor. W&B group `bi0-int4head-accept-oracle`.

## Results (the ceiling curve)
| rate r | mean acc. len (1+6r) | steady TPS | wall TPS | measured E_accept | draft_acc | W&B |
|---:|---:|---:|---:|---:|---:|---|
| 0.56 (anchor) | 4.36 | 309.96 | 323.29 | 4.355 | 0.559 | myk4s0ft |
| 0.70 | 5.20 | 362.37 | 381.83 | 5.202 | 0.700 | lip0l3qq |
| 0.85 | 6.10 | 421.29 | 445.27 | 6.111 | 0.852 | jpuid3t5 |
| **1.00 (ceiling)** | **7.00** | **471.09** | **506.84** | **7.000** | **1.000** | j60r68os |

Ceiling run (table + PNG): `zc76n7xz`. PNG:
`research/lenient_spec_acceptance_813/runs/sweep/ceiling_curve.png`.

- **Mechanism is faithful**: imposed mean acceptance length is reproduced almost
  exactly (4.355/5.202/6.111/7.000 measured vs 4.36/5.20/6.10/7.00 imposed),
  confirming the synthetic branch drives the decode loop as intended.
- **Ceiling gain = +52.0% steady / +56.8% wall** over the r=0.56 anchor. TPS is
  near-linear in mean-acceptance-length (per-token overhead does NOT swamp the
  analytic gain — the lm_head being int4 amortizes well per accepted token).

## Correction to the 256.74 anchor (report this)
The PR/PLAN carried **256.74 TPS** as the int4head AR-equiv anchor. That number
was **a RECONSTRUCTION, never a served measurement**: W&B `9tcygwjf`
(ubel/bi0-lmhead-bytes) served the **bf16-tied-head** control (`w4a16-ct`,
measured 219.34 TPS) and *projected* int4-head TPS by swapping the bf16 head-GEMV
(2.777 ms/tok) for an int4 estimate (0.7496 ms/tok) ⇒ 256.74. **This probe is the
first true served int4head**, and at matched acceptance (r=0.56) it reads
**309.96 steady / 323.29 wall — ~21% ABOVE the 256.74 reconstruction.** The
reconstruction under-counted the realized int4-head benefit. The greenlight
verdict is robust under either framing:
- absolute: ceiling 471 ≫ 282 greenlight band;
- relative to the (correct) served anchor 309.96: +52.0%.

## Decision (advisor rule)
ceiling@r=1.00 = 471.09 steady TPS **> 282** ⇒ **GREENLIGHT** a separate custom
greedy-kernel top-k-match accept-branch PR. Scope for that follow-up (NOT
implemented here): replace `rejected = draft_token_id != target_argmax_id` with a
top-k-match test (`draft ∈ topk(target_logits, k)`) behind a new `k` env var,
shipped as a `*_patch.py` like `vllm_attn_group_patch.py`, with its own PPL +
generated-text quality panel and a greedy-identity-waiver under #784. The realized
gain there will be a FRACTION of this 52% ceiling (only the extra draft tokens a
finite k actually wins), so the follow-up must measure E_accept(k) AND the
quality cost — this oracle only proves the speed headroom exists to chase.

Per the #784 cap, this is the **last** probe on the acceptance axis: one sweep,
plotted, summarized. No HF job, no further iterations on #813.
