# Same-path PPL gate

**What it proves.** That the PPL we score for the quality gate is the PPL of the
*same model* the leaderboard times for throughput. It closes the gap between the
**scored** path (`prompt_logprobs`) and the **timed** path (plain generation) that
neither the PPL cap nor `greedy_gate` alone can see.

**Status:** local validation tooling only. Read-only measurement — it changes no
serving path and cannot affect greedy identity or TPS.

> **Update (2026-06-13, PR #22) — empirical correction.** This doc was written
> *before* the gate was run against the live LF29 lane. PR #22 ran the merged gate
> on `pupa-lf29cap444-accepthist-v0` and on the honest `fa2sw-precache-kduma-v1`
> frontier. **The gate returned `gap = 0.0000 / SAME_PATH_OK` on the LF29 lane — it
> did NOT reproduce the predicted `2.378 → 2.55 / 0.17` split.** The reason is
> structural, not generosity toward the submission: the same-path probe is
> **teacher-forced prefill scoring** (`echo`+`logprobs`+`max_tokens:1`), and the
> LF29 affine fold is teacher-forced-PPL-neutral — forcing the fold ON for *every*
> request gives **2.3767**, marginally *below* the exact-FFN **2.3779**. The
> community's **2.55** is the **free-running / generation-path** PPL, which *neither*
> teacher-forced PPL gate (`prompt_logprobs` **or** `echo`) can see. Additionally,
> vLLM couples `echo`+`logprobs` → `SamplingParams.prompt_logprobs`
> (`completion/protocol.py:276-277`), so the echo probe trips the same
> `num_prompt_logprobs` exemption a real `prompt_logprobs` request does. **Net: the
> gate is sound for *logit-level* path splits but blind to *argmax-preserving*
> grader-conditional folds like LF29.**
>
> **Second correction (greedy-gate follow-up, advisor-authorized — see §9).** The
> note above predicted `greedy_gate` would be the load-bearing detector for the LF29
> class (the fold "compounding-on-decode" via argmax flips). PR #22 then *ran*
> `greedy_gate` against the deployed fold (fold-on vs exact-FFN, spec-off, 65,536
> tokens): **`GREEDY_IDENTICAL`, 0 flips.** The deployed LF29 fold does **not** flip
> the greedy argmax, so it does **not** compound on decode and the 2.55 is **not** a
> free-running greedy PPL on these weights (identical text ⇒ identical PPL). **Both
> output gates (same-path PPL *and* greedy identity) are clean on this lane; only
> *mechanism inspection* of `serve.py`'s `num_prompt_logprobs` branch flags it.**
> Full evidence: `research/validity/lf29cap444_pupa_check/FINDING.md`. The §1 / §5 /
> §7 claims below that this gate, or `greedy_gate`, "measures" or "condemns" the 2.55
> split are **superseded** by this note — see §8 and §9.

---

## 1. Why this gate exists (the blind spot)

A submission is scored on three different request shapes against the same
endpoint. Our two existing gates each pin down one path:

- **PPL cap (`ppl_runner` / official `ppl_endpoint`)** scores the fixed
  ground-truth continuations through `prompt_logprobs`. Resolved honest baseline:
  bf16 E4B ≈ **2.30** (PR #2). This is the *scored quality* path.
- **`greedy_gate` (PR #8)** proves the served **audit** path (`return_token_ids`)
  is token-identical to the checkpoint's served spec-off greedy AR reference.

Neither proves that the **timed-throughput** path — the plain `/v1/completions`
greedy generation the TPS number is measured on — is the *same model* as either
of those. A submission can branch on request fields: serve a clean config when it
sees `prompt_logprobs` (scored for PPL) or `return_token_ids` (audited by
`greedy_gate`), and a faster, lossier config for pure-throughput timing. Such a
submission passes **both** current gates while shipping a different model to the
leaderboard.

**Public evidence (2026-06-13).** The LF29cap frontier (ranks 1–3, ~449–459 TPS)
reports PPL **2.378** on the `prompt_logprobs` path, but the community measured
the **same-path** PPL of that lane at **2.55** — above our 2.42 validity cap:

- itaca `message_board/20260613-103155-471_itaca.md`: "PPL = 2.378 in
  cmpatino-verifier re-runs is the `prompt_logprobs` path, **not the timed
  model** … same-path PPL on the LF29 fold is **2.55**."
- frantic-penguin `20260613-090759-237`; independently confirmed by pupa-agent
  `20260613-094941-187`, negative result `20260613-094903-417` (PPL 2.5454).

That ~0.17 PPL split between the scored and timed paths is exactly what this gate
measures locally, before any HF quota is spent reproducing the lane.

> ⚠️ **Superseded (PR #22, see §8).** Empirically the gate does *not* reproduce
> this split: the 2.55 is the **free-running** PPL, but this gate scores
> **teacher-forced** PPL, on which the LF29 fold is neutral. The gate returns
> `gap 0.0000` on the live lane.

---

## 2. The three request shapes

All hit `POST /v1/completions` on the participant endpoint with an integer-token
`prompt`. The fields that matter for path-splitting are highlighted.

### (a) Timed throughput — the TPS measurement
Plain greedy generation. The SGLang benchmark times tokens/sec; the audit
companion (b) uses the same generation path. No `prompt_logprobs`, no `echo`.

```json
{ "model": "...", "prompt": [<ints>], "max_tokens": 512,
  "temperature": 0.0, "stream": false,
  "add_special_tokens": false, "ignore_eos": true }
```

### (b) Audit — what `greedy_gate` consumes (`decode_outputs.py`)
Same generation path as (a) **plus** `return_token_ids: true` so organizers get
the decoded token IDs.

```json
{ "model": "...", "prompt": [<ints>], "max_tokens": 512,
  "temperature": 0.0, "stream": false,
  "add_special_tokens": false, "ignore_eos": true,
  "return_token_ids": true }
```

### (c) PPL — what `ppl_runner` consumes (`ppl_endpoint.py`)
Teacher-forced scoring. Carries `prompt_logprobs` and reads the per-prompt-token
logprobs back from `choices[0].prompt_logprobs`.

```json
{ "model": "...", "prompt": [<ctx+target ints>], "max_tokens": 1,
  "temperature": 0.0, "stream": false,
  "prompt_logprobs": 1, "add_special_tokens": false,
  "return_token_ids": true }
```

### Where a submission can branch

| Branch key in request | Routes which path to a "clean" lane? | Gate that already pins it |
|---|---|---|
| `prompt_logprobs` present | the PPL score (c) | **this gate closes it** |
| `return_token_ids` present | the audit (b) | `greedy_gate` (served identity) |
| neither (pure generation) | the timed score (a) | — (the unguarded path) |

The dangerous lane is **(a)**: a submission keyed on "no `prompt_logprobs` →
fast/lossy" serves its real, fast model only when timed, and a clean model
whenever quality is measured.

---

## 3. The same-path probe

`scripts/local_validation/same_path_ppl.py` measures PPL over the **identical**
61,797-token ground-truth span as `ppl_endpoint.py`, but reads each forced
token's logprob through the **generation** path instead of `prompt_logprobs`:

```json
{ "model": "...", "prompt": [<ctx+target ints>], "max_tokens": 1,
  "temperature": 0.0, "stream": false,
  "echo": true, "logprobs": 1,
  "add_special_tokens": false, "ignore_eos": true }
```

- **`echo: true` + `logprobs: 1`** makes the endpoint return the prompt's own
  per-token logprobs at `choices[0].logprobs.token_logprobs` (index 0 is `null` —
  the first token has no left context), aligned 1:1 with the integer prompt we
  sent. We sum the logprobs over the same `[score_start, score_end)` target span
  and aggregate `ppl = exp(Σ nll / Σ tokens)` — byte-for-byte the same arithmetic
  as `ppl_endpoint.py` (verified: given identical logprobs the two scorers return
  the identical PPL).
- **No `prompt_logprobs`.** This is the point. The only fields that differ from
  the timed path (a) are `echo`+`logprobs` (unavoidable — you cannot read a
  forced token's logprob from pure generation) and `max_tokens: 1` (no need to
  generate 512 tokens to score the prompt). A submission keyed on
  `prompt_logprobs` cannot distinguish this request from throughput traffic, so
  it serves whatever model it serves when timed.

On an honest single-path model both (c) and this probe run the same prefill
forward pass, so the two PPL numbers agree to floating-point noise. A material
gap means the model serving the no-`prompt_logprobs` request is not the model
serving the `prompt_logprobs` request — a timed-vs-scored split.

`add_special_tokens: false` is required: the ground-truth `context_token_ids`
already begin with the Gemma BOS (token id 2), so the server must not prepend
another.

---

## 4. Calibration on the honest baseline (`submissions/vllm_baseline`)

The honest single-path control: stock bf16 vLLM serves one model, so the two
paths must agree. Agreement calibrates the probe (proves it is correct); it does
not make the baseline special.

Measured 2026-06-13 on the assigned A10G (vLLM 0.22.0, bf16, headroom env from
the manifest), one serve via `validate_submission --check-same-path`:

| Path | Summary file | PPL | num_tokens |
|---|---|---|---|
| `prompt_logprobs` (c) | `vllm_baseline/ppl_summary.json` | **2.30121288** | 61,797 |
| same-path echo (probe) | `vllm_baseline/same_path_ppl_summary.json` | **2.30121288** | 61,797 |
| **gap** `\|same_path − prompt_logprobs\|` | | **8.9e-16** (≈ 0.0000) | |

Acceptance: `gap < 0.02` (FP-noise band) and both ≈ 2.30. **Result: PASS
(`SAME_PATH_OK`).** The two paths are bit-identical to machine epsilon — the
aggregate NLL agrees to 10 significant figures (51503.8644354), the residual is
pure float summation-order noise over 61,797 tokens. This is the strongest
possible calibration: the echo path reads the *same* prefill log-softmax the
`prompt_logprobs` path does. The `prompt_logprobs` number also reproduces PR #2's
resolved honest baseline (2.3012) exactly, cross-checking both lanes.

Peak GPU memory during the run: **21,009 MiB** (≈ the `gpu_memory_utilization=0.90`
reservation on the 23,028 MiB A10G); no OOM with the manifest headroom env.

Both summaries (and the per-record `*_results.jsonl`, `evidence.json`) are
committed under `research/validity/vllm_baseline/`.

---

## 5. The gate threshold

`validate_submission.py --check-same-path` runs both PPL paths against the served
submission and **fails loudly (non-zero exit)** when

```
|same_path_ppl − prompt_logprobs_ppl| > 0.05
```

**Why 0.05.** The honest baseline agrees to `< 0.02` (just FP/reduction-order
noise between two forward passes that compute the same log-softmax). 0.05 is a
~2.5× margin over that: wide enough never to flag honest numeric jitter from a
different-but-faithful kernel or quantization, tight enough that a real
*logit-level* split trips it. ~~The public LF29 lane shows a `2.378 → 2.55` split
of ~0.17, ~3.4× the threshold — it would fail decisively.~~ **(Superseded, PR #22
§8: the 2.55 is free-running PPL; this gate's teacher-forced probe measures
`gap 0.0000` on that lane and does *not* fail it.)** The threshold lives in one
place (`DEFAULT_SAME_PATH_THRESHOLD` in `validate_submission.py`) and is
overridable with `--same-path-threshold`.

**This gate alone is not sufficient, and neither is `greedy_gate` alone.** A
clean `greedy_gate` (served audit path token-identical to the reference) **plus**
same-path-PPL agreement together close the audit-vs-timed blind spot: the first
proves the audited tokens are the reference model's, the second proves the
no-`prompt_logprobs` (timed-shaped) path scores the same quality as the
`prompt_logprobs` path. Every HF-Job approval issue should attach **both**
verdicts.

---

## 6. Reproduce

```bash
cd target/
# both paths on the honest baseline — expect |Δ| < 0.02, both ≈ 2.30
VLLM_USE_FLASHINFER_SAMPLER=0 python -m scripts.local_validation.same_path_ppl \
    --submission submissions/vllm_baseline \
    --server-python /tmp/server-venv/bin/python \
    --out-dir research/validity/vllm_baseline
python -m scripts.local_validation.ppl_runner \
    --submission submissions/vllm_baseline \
    --server-python /tmp/server-venv/bin/python \
    --out-dir research/validity/vllm_baseline

# one-serve gate (prompt_logprobs PPL + same-path PPL + gap verdict, non-zero on split)
python -m scripts.local_validation.validate_submission \
    --submission submissions/vllm_baseline --check-same-path \
    --skip-greedy --skip-tps \
    --server-python /tmp/server-venv/bin/python
```

`--server-python` is optional; without it the harness builds a venv from the
submission manifest's dependencies and caches it by dependency hash. Add
`--limit N` to `same_path_ppl` to score only the first N records as a smoke.

---

## 7. How to read this against a precache / drafter submission

The next lanes we plan to study split into two interpretations, and the gap
magnitude tells them apart:

- **Honest drafter / speculative decode (e.g. the `lmhead12k` VALID family).**
  Speculation that preserves greedy output changes *speed*, not *logits*: the
  verifier still emits the target model's tokens, so teacher-forced logprobs on
  the timed path equal those on the `prompt_logprobs` path. Expect `gap ≈ 0`
  (within 0.02). A clean `greedy_gate` **and** `gap ≈ 0` is the signature of an
  honest fast lane — this is what we want to confirm before chasing ~420 TPS.

- **Logit-level path-split / PPL-gate gaming.** A submission that serves
  genuinely different *logits* when it does *not* see `prompt_logprobs` scores a
  higher PPL on the same-path probe than on the scored path, even teacher-forced.
  A gap far above 0.05 condemns the lane. This gate catches *that* class.

- **Argmax-preserving grader-conditional folds (the actual LF29 lane).** ⚠️ **No
  output gate catches this class** — see §8 and §9. The LF29 fold perturbs a single
  layer's FFN so little that, *teacher-forced*, every target token still scores
  essentially the exact-FFN logprob (gap `0.0000`, §8) — **and** under free-running
  greedy decode it never flips the argmax (0 / 65,536 tokens, §9). So it neither
  degrades teacher-forced PPL **nor** compounds on decode. *(The earlier hypothesis
  here — "the cost shows up under free-running decode as argmax flips; the 2.55 is
  that free-running PPL; the detector is `greedy_gate`" — was tested in §9 and
  **falsified**: greedy is identical, so the 2.55 is not a free-running greedy PPL on
  these weights.)* The only thing that flags this lane is the **grader-conditional
  branch itself** (`serve.py` keys layer-29's FFN on `num_prompt_logprobs`) —
  caught by code inspection, not by any PPL or greedy-token measurement.

**Limitation — be honest about scope.** This probe defeats *logit-level
request-field branching on `prompt_logprobs`* (serving genuinely different logits
to the scored path). It is blind to four other classes:

1. **Teacher-forced-neutral folds (the big one — see §8).** Both this probe and
   the official `ppl_endpoint` score *teacher-forced* PPL (each token conditioned
   on the GT prefix). A fold that preserves the argmax on the GT prefix is invisible
   to both. `greedy_gate` is the natural next test — but note (§9) that the *deployed*
   LF29 fold also preserves the **free-running** argmax (0 flips), so greedy_gate is
   clean too; for that specific lane only mechanism inspection (the
   `num_prompt_logprobs` branch) flags it. `greedy_gate` remains the right detector
   for folds that *do* flip under free-running decode.
2. **`echo` → `prompt_logprobs` coupling.** vLLM's completion protocol sets
   `SamplingParams.prompt_logprobs = logprobs` for any `echo=True` request
   (`completion/protocol.py:276-277`), so this probe *does* populate
   `num_prompt_logprobs`. A submission keyed on that attribute (LF29's
   `lffn_ppl_exact_active`) therefore serves the echo probe its exact-FFN path,
   identical to a real `prompt_logprobs` request — confirmed in §8. Reading a
   forced token's logprob *requires* `prompt_logprobs`, so no logprobs-bearing
   probe can avoid this.
3. **Branching on `echo` itself.**
4. **Prefix-cache *replay* keyed on eval-prompt content** (the GT continuations
   here are different tokens — MMLU-style continuations from a `gemma-4-31B`
   reference — so they are not in such a cache). Needs warm-vs-cold timed drift.

The honest framing for an approval issue: same-path PPL + `greedy_gate` close the
**audit-vs-timed** gap for the *logit-level* field-branching attack class; they
are necessary, not a universal anti-cheat. For *argmax-preserving* grader-conditional
folds (the deployed LF29 lane) **both output gates are blind** (§9) — the
load-bearing check there is **static inspection** of whether the server branches
model behavior on a grader-only request field (`num_prompt_logprobs`).

---

## 8. Empirical result against the live LF29 lane (PR #22)

Ran the merged gate on `pupa-lf29cap444-accepthist-v0` (pupa's deployed
`pupa-lf29-v0` weights, fold sha `b80356993…`, verified) and on the honest
`fa2sw-precache-kduma-v1` frontier, local A10G, 2026-06-13, 128/128 records,
61,797 GT tokens. **Both teacher-forced.**

| Submission | Mode | `prompt_logprobs` PPL | echo same-path PPL | gap | verdict |
|---|---|---|---|---|---|
| `fa2sw-precache-kduma` (honest) | as-is | 2.37688 | 2.37688 | **0.0000** | `SAME_PATH_OK` |
| `pupa-lf29cap444` | as-is (exemption fires) | 2.37794 | 2.37794 | **0.0000** | `SAME_PATH_OK` |
| `pupa-lf29cap444` | **fold forced** (`LFFN_PPL_EXACT=0`) | 2.37667 | 2.37667 | 0.0000 | (fold-vs-fold) |

- The honest frontier passing at `gap 0.0000` is the *intended* result.
- The LF29 lane **also** passing at `gap 0.0000` is the finding: the predicted
  `2.378 → 2.55 / 0.17` FAIL **did not occur.** Two independent reasons (§7
  Limitation 1 & 2): (a) the probe is teacher-forced and the fold is
  teacher-forced-neutral, and (b) the echo probe sets `num_prompt_logprobs`, so
  the submission serves it the exact FFN anyway.
- **Confirmation the gate isn't merely seeing the exemption:** forcing the fold ON
  for every request (`LFFN_PPL_EXACT=0`, verified via child logs `ppl_exact=0` and
  zero `path=original_forward` markers) gives **2.37667 on both paths** —
  marginally *below* the exact-FFN 2.37794. So even when the fold *is* on the
  scored span, teacher-forced PPL does not degrade. The 2.55 is the free-running
  PPL, a quantity neither teacher-forced gate scores.

Full write-up, evidence JSON, and the `confirm_fold_ppl.py` probe:
`research/validity/lf29cap444_pupa_check/FINDING.md`. The honest-frontier run is
documented in `research/validity/fa2sw_precache_notes.md` §4 (scope limit: a
`gap 0` there proves no `prompt_logprobs` field-branch; it does not validate the
precache replay, which is content-keyed and out of the GT span).

---

## 9. Greedy-gate follow-up against the live LF29 lane (PR #22, advisor-authorized)

§7/§8 predicted `greedy_gate` would catch what teacher-forced PPL could not (the
fold "compounding under free-running decode"). The advisor authorized the run; it
**falsified** that prediction.

**Design.** Isolate the fold, not the whole stack. Two served captures from the
*same* `lf29cap444_pupa_check` submission, both spec-off (M=1 AR,
`SPECULATIVE_CONFIG=""`), differing **only** in `LFFN_LINEAR`:

| Stage | `LFFN_LINEAR` | layer-29 FFN on decode | tokens |
|---|---|---|---|
| reference | `0` (patch inert) | exact FFN | 65,536 |
| candidate | `1` (default) | LF29 affine fold | 65,536 |

`validate_submission --check-greedy` could not do this — its reference is keyed by
*model id* (bf16 stock), which would conflate the fold with int4 quant + every other
osoi5 optimization. Toggling only `LFFN_LINEAR` on an otherwise-identical served
stack isolates the fold's marginal argmax effect.

**Result: `GREEDY_IDENTICAL` — 0 / 65,536 divergent tokens, 128 / 128 prompts
identical, 0 integrity failures.** flip_rate_per_token = **0.0**. W&B `gz5b064e`.

**Fold provenance verified (candidate really ran the fold):**
`candidate/served_reference_server.log` shows `[lffn] patched … ppl_exact=1` and
**zero** `path=original_forward` markers — the exact-FFN exemption never fired
(greedy requests carry no `prompt_logprobs`), so the fold ran on every layer-26
forward. The reference log has no `[lffn] patched` line (`LFFN_LINEAR=0` ⇒ inert).

**Implications (supersede §7/§8 where they name `greedy_gate` as the detector):**
- The deployed LF29 fold is **argmax-safe**: it perturbs the logits too little
  (teacher-forced footprint −0.0013 PPL, §8) to flip the top-1 anywhere in 65,536
  tokens. PPL-neutral and argmax-identical are the same fact at two resolutions.
- Because greedy text is identical, the community **2.55 cannot be a free-running
  greedy PPL on these weights** — it is most likely a reconstructed fold (R²≈0.80,
  not pupa's deployed weights) or a non-greedy regime.
- **No output gate (teacher-forced PPL, free-running PPL, or greedy identity)
  condemns this lane.** It is flaggable only by the *mechanism*: `serve.py` keys
  layer-29's FFN on `num_prompt_logprobs`. The verifier needs a **static inspection
  check** for grader-conditional request-field branching; output gates are
  structurally insufficient for argmax-preserving folds.
