# Greedy-Identity Verifier

A small, dependency-free tool that checks the Gemma challenge's **hard validity
rule** by comparing two harness `decode_outputs.jsonl` files: a candidate
submission against an exact-greedy reference.

## What it checks

The challenge rule, quoted verbatim:

> the served endpoint's greedy decode must be token-identical to plain greedy
> autoregressive decode of the same submitted checkpoint... Any optimization
> that changes the generated token IDs, even if TPS improves or PPL remains
> similar, is not valid for leaderboard scoring.

This tool answers one question: **are the candidate's generated token IDs
byte-for-byte identical to the reference's, prompt by prompt?**

## Why PPL can't catch this

Perplexity (PPL) is **teacher-forced**: it scores the model's log-probabilities
against the *ground-truth* continuation, feeding the correct next token at every
step regardless of what the model would actually have emitted. That makes PPL
**immune to which tokens the model actually generates** — a candidate can drift
to different token IDs (via a lossy kernel, quantization, speculative decoding,
or any other "optimization") while its PPL stays effectively unchanged.

Token divergence is exactly what the validity rule forbids and exactly what PPL
cannot see. This tool fills that gap by comparing the *emitted token IDs*
directly.

## The sha256 recipe

Each record may carry a `completion_token_sha256` over its `completion_token_ids`.
The harness (and this tool) compute it as:

```python
sha256(",".join(str(t) for t in token_ids).encode("ascii")).hexdigest()
```

The tool runs a **per-record integrity check**: if a record's stored
`completion_token_sha256` does not match the sha of its own
`completion_token_ids`, the data is considered tampered/untrustworthy and the
verdict is `INCOMPARABLE` (exit 2) rather than a comparison result. Records may
omit the field entirely; it is only checked when present.

## Input format

Each line of a `decode_outputs.jsonl` file is one JSON record:

| field                     | type        | notes                                    |
| ------------------------- | ----------- | ---------------------------------------- |
| `id`                      | str/int     | record key (falls back to `prompt_sha256`) |
| `completion_token_ids`    | list[int]   | required; the emitted token IDs          |
| `completion_token_sha256` | str         | optional; integrity-checked when present |
| `prompt_sha256`           | str         | optional                                 |

## Install-free usage

Standard library only — **no install, no dependencies**. Just Python 3.

```
python3 check_greedy_identity.py --reference REF.jsonl --candidate CAND.jsonl [--json] [--max-examples N]
```

Flags:

- `--reference PATH` — exact-greedy reference `decode_outputs.jsonl` (required).
- `--candidate PATH` — candidate `decode_outputs.jsonl` under test (required).
- `--json` — emit the full report as JSON instead of human-readable text.
- `--max-examples N` — max divergent prompts to list in human output; a
  negative value shows all (default: 5).

## Exit codes

| code | verdict             | meaning                                                       |
| ---- | ------------------- | ------------------------------------------------------------- |
| 0    | `GREEDY_IDENTICAL`  | valid — candidate token IDs match the reference exactly       |
| 1    | `DIVERGENT`         | invalid — same prompts, but >=1 prompt's token IDs differ     |
| 2    | `INCOMPARABLE`      | prompt sets differ, a stored-sha integrity check failed, or an error (missing/malformed/empty input, bad args) |

## Worked example

The committed `fixtures/` directory contains a 3-prompt reference and two
candidates. The commands and output below are real.

### Valid candidate → exit 0

```
$ python3 check_greedy_identity.py \
    --reference fixtures/reference.jsonl \
    --candidate fixtures/candidate_valid.jsonl
VERDICT: GREEDY_IDENTICAL (valid)
  prompts compared:       3
  identical:              3
  divergent:              0
  total tokens compared:  18
  total divergent tokens: 0
$ echo $?
0
```

### Divergent candidate → exit 1

`candidate_divergent.jsonl` flips two tokens in `prompt-1` (with a recomputed,
consistent sha so the integrity check passes and the result is a genuine
divergence, not `INCOMPARABLE`):

```
$ python3 check_greedy_identity.py \
    --reference fixtures/reference.jsonl \
    --candidate fixtures/candidate_divergent.jsonl
VERDICT: DIVERGENT (invalid)
  prompts compared:       3
  identical:              2
  divergent:              1
  total tokens compared:  18
  total divergent tokens: 2
  divergent prompts (showing 1 of 1):
    - prompt-1: first divergence at index 2
$ echo $?
1
```

### JSON output

```
$ python3 check_greedy_identity.py \
    --reference fixtures/reference.jsonl \
    --candidate fixtures/candidate_valid.jsonl --json
{
  "verdict": "GREEDY_IDENTICAL",
  "num_prompts_compared": 3,
  "num_identical": 3,
  "num_divergent": 0,
  "total_tokens_compared": 18,
  "total_divergent_tokens": 0,
  "missing_in_candidate": [],
  "missing_in_reference": [],
  "integrity_failures": [],
  "per_prompt": [
    {
      "key": "prompt-0",
      "identical": true,
      "ref_len": 6,
      "cand_len": 6,
      "length_mismatch": false,
      "first_divergence_index": null,
      "num_divergent_tokens": 0,
      "num_compared": 6,
      "stored_sha_consistent": true
    },
    ...
  ]
}
```

## Tests

```
python3 -m unittest discover -s tests -v
```

## Notes & assumptions

- **Record matching:** prompts are matched by the `id` field (falling back to `prompt_sha256`
  when `id` is absent). This assumes the candidate and reference share the harness's stable
  `id`↔prompt alignment (true for any two runs of the same eval set). If the two files cover
  different prompt sets, the verdict is `INCOMPARABLE` and the missing keys are reported.
- **Integrity gate:** when a record carries `completion_token_sha256`, it is checked against
  `sha256_tokens(completion_token_ids)`; any mismatch forces `INCOMPARABLE` (the data is
  untrustworthy). Hand-edited fixtures must recompute this field — see `fixtures/`.
- **Scope:** this tool checks only greedy token-identity. TPS/perplexity significance is a
  separate concern (see `shared_resources/gemma_specdecode_headroom_flowian/`).
