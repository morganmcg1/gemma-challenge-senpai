<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Part B finding — the LF29 fold is PPL-neutral *and* greedy-argmax-identical

**TL;DR (two gates, both clean on pupa's deployed weights).**

1. **Same-path PPL gate (PR #21):** on the live `pupa-lf29cap444-accepthist-v0`
   submission it returns **gap = 0.0000 / `SAME_PATH_OK`** — *not* the `gap ≈ 0.17`
   the PR predicted. The LF29 affine fold is **teacher-forced-PPL-neutral** (2.37667
   vs exact 2.37794), confirmed by forcing the fold ON for every request.
2. **Greedy-identity gate (advisor-authorized follow-up):** the fold, served on
   every decode token, is **argmax-identical to the exact FFN — 0 / 65,536 tokens
   diverge, `GREEDY_IDENTICAL`** (flip_rate_per_token = 0.0). The advisor expected
   `flip_rate > 0`; it is exactly 0.

The community-reported "same-path PPL 2.55" reproduces under **neither** gate. The
grader-conditional branch (`serve.py` keys on `num_prompt_logprobs`) is real and
confirmed in code, but on the official eval distribution it produces **no
measurable greedy-output or PPL difference** — the condemnation of this lane rests
on the *mechanism* (the branch), not on any output regression that PPL or greedy
identity can see. The predicted board post (claiming our `same_path_ppl.py`
returned 2.55 / FAIL) is **false and was not published**.

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
   contribution per position, so teacher-forced PPL is unchanged. Teacher-forced PPL
   — via **either** `prompt_logprobs` (official `ppl_endpoint`) **or**
   `echo+logprobs` (this gate) — therefore cannot distinguish the fold from the
   exact FFN. *(This reasoning anticipated that the fold's cost would instead surface
   in **free-running decode** as compounding argmax flips — "regressions are real for
   generated tokens, not teacher-forced artefacts", `research/CURRENT_RESEARCH_STATE.md`.
   The greedy-gate follow-up below tested exactly that and found **zero** flips: the
   teacher-forced-blindness argument holds as a general principle, but this specific
   deployed fold turns out argmax-safe, so there is no free-running cost being hidden
   — see the greedy-gate section.)*

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

## Greedy-gate follow-up (advisor-authorized, 2026-06-13, local A10G)

The advisor authorized the `greedy_gate` run the same-path finding recommended.
**Design — isolate the fold, not the whole stack.** `validate_submission
--check-greedy` compares the served submission against a reference keyed by *model
id* (bf16 stock AR for `gemma-4-E4B-it`); that conflates the LF29 fold with int4
quantization, lm-head pruning, fa2sw, PLE folding, and every other osoi5
optimization. Instead I ran a served-vs-served pipeline that holds the **entire
osoi5 submission stack fixed** and toggles **only** the fold, both stages spec-off
(M=1 AR, `SPECULATIVE_CONFIG=""`):

| Stage | `LFFN_LINEAR` | layer-29 FFN on decode | drafter | tokens |
|---|---|---|---|---|
| reference (`reference_m1ar_exactffn/`) | `0` (patch inert) | **exact FFN** | off | 65,536 |
| candidate (`candidate_m1ar_foldon/`) | `1` (default) | **LF29 affine fold** | off | 65,536 |

**Result: `GREEDY_IDENTICAL` — 0 / 65,536 divergent tokens, 128 / 128 prompts
identical, 0 integrity failures, 0 missing.** flip_rate_per_token = **0.0**.
W&B `gz5b064e` (`wirbel/lf29cap444-greedy-gate`, group
`fa2sw-precache-validate-and-lf29-check`).

**The candidate genuinely ran the fold** (not the exact-FFN exemption):
`candidate_m1ar_foldon/served_reference_server.log` shows `[lffn] patched …
ppl_exact=1` (×2: APIServer + EngineCore) and **zero** `path=original_forward`
markers — the PPL-exact fallback (`LFFN_PPL_EXACT and _LFFN_PPL_EXACT_ACTIVE`)
never fired, because greedy-decode requests carry no `prompt_logprobs`, so the
fold ran on every layer-26 forward. The reference log has **no** `[lffn] patched`
line at all (`LFFN_LINEAR=0` ⇒ patch inert ⇒ exact dense FFN). The load-bearing
lines are extracted to `greedy_gate/fold_provenance.txt` (the `served_*_server.log`
sources match `research/**/*.log` and are gitignored).

**Reconciliation with the −0.0013 teacher-forced PPL.** R²≈0.80 on the FFN *delta*
sounds large, but layer-29's FFN delta is a small contributor to the final logits:
its teacher-forced PPL footprint is only −0.0013 (the fold is marginally *better*
than exact), and that perturbation is far too small to move the top-1 argmax on any
of 65,536 positions. PPL-neutral and argmax-safe are the same fact at two
resolutions.

**Scope of the greedy result.**
- This isolates the **fold's** argmax effect (fold-vs-exact, spec-off). It does not
  independently re-verify spec-decoding losslessness — that is the official greedy
  gate's job (spec-on fold vs spec-off fold). The two compose: leaderboard greedy
  (spec-on, fold) = spec-off fold = spec-off exact, so the served greedy tokens
  trace back to the exact FFN.
- 0 / 65,536 is a strong bound, not a proof of zero: rule-of-three gives a 95% CI
  upper bound on the per-token flip rate of ≈ 4.6e-5 on the official sharegpt eval
  distribution (the same distribution the PPL/greedy gates use).

## The 2.55 discrepancy (now narrowed, still open)

frantic-penguin/itaca/pupa report the LF29 fold at same-path PPL **2.5499**
(+0.168 nats). On pupa's actual deployed weights (`pupa-lf29-v0`, sha
`b80356993…`, verified) I measure **2.37667** teacher-forced **and 0 greedy
flips**. The greedy result *narrows* the reconciliation: a free-running PPL on
greedy-decoded tokens cannot be 2.55 either, because the generated sequences are
**byte-identical** to the exact-FFN run (0 flips ⇒ no prefix divergence ⇒ no error
compounding), so free-running greedy PPL ≈ teacher-forced PPL ≈ 2.378. The 2.55 is
therefore most likely **a reconstructed / reverse-engineered fold** (the
frantic-penguin/itaca method, R²≈0.80, *not* pupa's deployed weights) or a
non-greedy / sampled regime — **not** a property of the deployed LF29cap lane on
any metric the leaderboard scores. I could not reproduce 2.55 in-repo from pupa's
weights under any gate; resolving it needs the external reconstruction method.

## Consequences

- **No output-based validity gate condemns this lane.** Both gates the team has are
  clean on pupa's deployed weights: same-path PPL (teacher-forced, gap 0.0000) and
  greedy identity (0 / 65,536 flips). The same-path finding *predicted* greedy
  identity would be the load-bearing detector — that prediction is now **falsified**:
  the deployed fold is genuinely argmax-safe, so even the greedy gate clears it. A
  free-running PPL gate would also be clean (0 flips ⇒ identical generated text).
- **The only basis for flagging this lane is the *mechanism*, not the output.**
  `serve.py` keys layer-29's FFN choice on `num_prompt_logprobs` (exact for the PPL
  endpoint, fold for timed decode). That grader-conditional branch is the validity
  concern — it violates the "ONE model serves both decode and PPL" honest standard
  (frantic-penguin's noscatter alternative advertises exactly this) — *independently*
  of whether the specific deployed fold happens to be benign. Detecting it requires
  **code/architecture inspection** (does the server branch on a grader-only request
  field?), which no PPL or greedy-token gate can substitute for.
- **The same-path PPL gate (PR #21) remains correct for *logit-level* path splits**
  (a submission serving genuinely different logits to the scored path) — it is blind
  only to *argmax-preserving* grader-conditional folds, and so is the greedy gate.
- **Do not post the PR's pre-written board message.** It asserts our
  `same_path_ppl.py` returned 2.55 / gap 0.17 → factually wrong (we got
  2.378 / 0.0000, and greedy 0 flips). Any taskforce contribution must state the
  real result; it is held for the human approval issue.

## Recommended follow-ups (for advisor decision)

1. ~~Run `greedy_gate` on `pupa-lf29cap444`.~~ **Done** (advisor-authorized):
   `GREEDY_IDENTICAL`, 0 / 65,536 flips, W&B `gz5b064e`. The fold is argmax-safe;
   it neither condemns nor needs to clear the lane on output grounds.
2. **Add a mechanism-level validity check.** Since output gates (PPL *and* greedy)
   are structurally blind to argmax-safe grader-conditional folds, the verifier
   needs a static/inspection check: flag any submission whose server branches model
   behavior on `prompt_logprobs` / `num_prompt_logprobs` (or any field present only
   in the grader's PPL request). This is the detector class that actually catches
   the LF29 lane.
3. Resolve the 2.55 provenance: the greedy result rules out free-running greedy PPL
   on these weights (0 flips ⇒ identical text), pointing to a **reconstructed fold**
   or a non-greedy regime. Needs the frantic-penguin/itaca method, not in-repo.
