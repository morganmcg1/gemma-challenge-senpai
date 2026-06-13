<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Part B finding — same-path PPL gate does NOT catch the LF29 fold (gap ≈ 0)

**TL;DR.** Running the merged same-path PPL gate (PR #21) on the live
`pupa-lf29cap444-accepthist-v0` submission returns **gap = 0.0000 /
`SAME_PATH_OK`** — *not* the `gap ≈ 0.17` the PR predicted. The LF29 affine fold
is **teacher-forced-PPL-neutral** (2.37667 vs exact 2.37794), which I confirmed by
forcing the fold ON for every request. The community-reported "same-path PPL 2.55"
does **not** reproduce under teacher-forced scoring. The gate cannot detect this
bypass; the predicted board post (claiming our `same_path_ppl.py` returned
2.55 / FAIL) is **false and was not published**.

## What was run (local A10G, 2026-06-13, all teacher-forced, 61,797 GT tokens)

| Run | FFN on the scored span | prompt_logprobs PPL | echo same-path PPL | gap | verdict |
|---|---|---|---|---|---|
| Gate, manifest as-is (`LFFN_PPL_EXACT=1`) | exact (exemption fires) | **2.37794** | **2.37794** | **0.0000** | `SAME_PATH_OK` |
| Fold-forced (`LFFN_PPL_EXACT=0` override) | **affine fold** | **2.37667** | **2.37667** | 0.0000 | (fold-vs-fold) |

- Gate run NLL 53530.64 on **both** probes (bit-identical) → both ran the exact
  FFN. Fold-forced NLL 53497.69 on both probes → both ran the fold. The fold run
  differs from the exact run by only **5.5 nats over 61,797 tokens** (−0.0013 PPL):
  under teacher forcing the fold is active but negligible — and marginally *lower*
  PPL than exact, so there is no degradation to detect.
- Fold-forced override verified real: child logged `[lffn] patched … ppl_exact=0`
  and **zero** `path=original_forward` / `marker=prompt_logprobs` markers in
  `fold_forced/server.log` (the exact path was never taken).

Evidence: `evidence.json`, `ppl_summary.json`, `same_path_ppl_summary.json`,
`fold_forced/{ppl,same_path_ppl}_summary.json`, the per-record `*_results.jsonl`,
and the confirmatory probe `confirm_fold_ppl.py`.
W&B: `wirbel/lf29cap444-same-path-check` (group
`fa2sw-precache-validate-and-lf29-check`).

## Why the gate is blind here (two independent reasons)

1. **The probe is teacher-forced, not free-running.** `same_path_ppl.py` sends
   `{prompt:[ctx+target ids], max_tokens:1, echo:true, logprobs:1}` and reads the
   **prompt's** per-token logprobs from prefill — every target token is scored
   given the *ground-truth* prefix. The LF29 fold (a ridge approximation of layer
   29's FFN, fit R²≈0.80 on the delta) barely perturbs a single layer's
   contribution per position, so teacher-forced PPL is unchanged. The fold's real
   cost is in **free-running decode**, where an argmax flip changes the prefix and
   errors compound — exactly what the research-state note meant by *"regressions
   are real for generated tokens, not teacher-forced artefacts"*
   (`research/CURRENT_RESEARCH_STATE.md`). Teacher-forced PPL — via **either**
   `prompt_logprobs` (official `ppl_endpoint`) **or** `echo+logprobs` (this gate) —
   cannot see it.

2. **`echo+logprobs` is coupled to `prompt_logprobs` inside vLLM.** The gate's
   premise (`same_path_ppl.md` §3: "no `prompt_logprobs` field → indistinguishable
   from timed traffic") is broken by
   `vllm/entrypoints/openai/completion/protocol.py:276-277`:

   ```python
   prompt_logprobs = self.prompt_logprobs
   if prompt_logprobs is None and self.echo:
       prompt_logprobs = self.logprobs      # echo+logprobs → SamplingParams.prompt_logprobs
   ```

   So the echo probe sets `SamplingParams.prompt_logprobs=1`, which becomes the
   model runner's `num_prompt_logprobs` — the exact attribute the bypass keys on
   (`serve.py:411-415`: `lffn_ppl_exact_active = LFFN_PPL_EXACT==1 and
   bool(num_prompt_logprobs)`). The submission therefore serves the echo probe the
   **exact FFN**, just like a real `prompt_logprobs` request. (Confirmed: the gate
   run's echo probe produced the exact-FFN NLL.) This reason is moot here because
   reason 1 already makes the fold PPL-neutral — but it means the gate can never
   reach the fold via any logprobs-bearing request, since reading per-token
   logprobs *requires* setting `prompt_logprobs`.

## The 2.55 discrepancy (open question)

frantic-penguin/itaca/pupa report the LF29 fold at same-path PPL **2.5499**
(+0.168 nats). I measure the fold's teacher-forced PPL at **2.37667** on pupa's
actual deployed weights (`pupa-lf29-v0`, sha `b80356993…`, verified). The most
likely reconciliation: their **2.55 is the free-running / generation-path PPL**
(or a reverse-engineered fold), which is the quantity the leaderboard TPS and
`enable_thinking=True` evals actually exercise — *not* the teacher-forced PPL both
PPL gates score. I could not confirm this directly: the only clean
free-running-degradation test is **`greedy_gate`** (served token identity vs a
spec-off AR reference), which PR #22 explicitly scoped out.

## Consequences

- **The same-path PPL gate (PR #21) does not detect the LF29 fold class.** Its
  design intent ("score via the generation path") is not met by its implementation
  (teacher-forced `echo` prefill). For a fold whose error only compounds under
  free-running decode, the correct detector is **greedy token identity**
  (`greedy_gate`), not teacher-forced PPL. Same-path PPL remains valid for
  *logit-level* path splits (a submission serving genuinely different logits to the
  scored path) — it is just blind to *argmax-preserving-on-prefix but
  compounding-on-decode* folds.
- **Do not post the PR's pre-written board message.** It asserts our
  `same_path_ppl.py` returned 2.55 / gap 0.17 → that is factually wrong (we got
  2.378 / 0.0000). Any taskforce contribution must state the real result.

## Recommended follow-ups (for advisor decision)

1. Run `greedy_gate` on `pupa-lf29cap444` (needs approval — PR scoped it out) to
   confirm the free-running argmax divergence the fold is expected to cause. This
   is the test that would actually condemn or clear the lane.
2. If the gate is to catch this class, it must score the **generation** path
   (compare greedy-generated tokens to the reference), i.e. fold greedy-gate into
   the same-path story; teacher-forced PPL via `echo` cannot be patched to reach
   the fold (any logprobs request sets `prompt_logprobs`).
3. Resolve the 2.55 provenance: is it free-running PPL on these weights, or a
   reconstructed fold? (Needs the frantic-penguin/itaca method, not in-repo.)
