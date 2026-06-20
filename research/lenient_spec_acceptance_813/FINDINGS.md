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
