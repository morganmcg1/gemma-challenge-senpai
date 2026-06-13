# Same-path PPL gate

**What it proves.** That the PPL we score for the quality gate is the PPL of the
*same model* the leaderboard times for throughput. It closes the gap between the
**scored** path (`prompt_logprobs`) and the **timed** path (plain generation) that
neither the PPL cap nor `greedy_gate` alone can see.

**Status:** local validation tooling only. Read-only measurement — it changes no
serving path and cannot affect greedy identity or TPS.

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
different-but-faithful kernel or quantization, tight enough that a real split
trips it. The public LF29 lane shows a `2.378 → 2.55` split of **~0.17**, ~3.4×
the threshold — it would fail decisively. The threshold lives in one place
(`DEFAULT_SAME_PATH_THRESHOLD` in `validate_submission.py`) and is overridable
with `--same-path-threshold`.

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

- **Path-split / PPL-gate gaming (e.g. the LF29 `2.378 → 2.55` lane).** A
  submission that serves a lossier config when it does *not* see `prompt_logprobs`
  scores a higher PPL on the same-path probe than on the scored path. A gap of
  ~0.17 (LF29) is far above 0.05 and condemns the lane: the leaderboard TPS was
  produced by a model whose real quality is **2.55 > 2.42 cap**, i.e. invalid.

**Limitation — be honest about scope.** This probe specifically defeats
*request-field branching on `prompt_logprobs`*. It does **not** by itself catch a
submission that branches on `echo`, nor a prefix-cache *replay* attack keyed on
the public eval-prompt **content** (the GT continuations here are different
tokens — MMLU-style continuations from a `gemma-4-31B` reference — so they are
not in such a cache). Those need their own checks (e.g. drift between warm and
cold timed runs, or scoring novel continuations). The honest framing for an
approval issue: same-path PPL + `greedy_gate` close the **audit-vs-timed** gap for
the field-branching attack class; they are necessary, not a universal anti-cheat.
