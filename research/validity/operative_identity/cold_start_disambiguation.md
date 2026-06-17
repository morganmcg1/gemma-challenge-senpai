<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Cold-start C1 disambiguation: is the operative-#319 bar warm-scored? (PR #599, wirbel)

**Status: analysis-only diagnostic card. `analysis_only=true`, `official_tps=0`. No HF Job,
no `train.py --launch`, no `/v1/jobs:run`, no submission, no served-file change.** The
`enable_prefix_caching=False` toggle used in §2 is a **diagnostic instrument** to localize the
cold-start transient C1 — it is **not** a proposed change to any served/submission file.

This card executes the two suggested follow-ups left open by PR #588's
[`contract.md`](contract.md) §5 (#1 disambiguate/eliminate C1, #2 confirm leaderboard
warmup) and closes the one live unknown in the operative-#319 identity contract: whether the
**first-pass cold-start transient C1** can trip the official scored greedy-identity pass.

## TL;DR

| question | verdict |
| -------- | ------- |
| **Q1 — `official_harness_warms_before_greedy_identity`** | **TRUE** (decisive, from the code) |
| **`operative_319_bar_is_warm_scored`** | **TRUE** — the literal-warm bar is the operative contract, **zero margin consumed** |
| Q2 — `prefix_caching_off_collapses_C1` | **TRUE** — prefix-OFF makes even cold pass `a` byte-identical to warm (`a==b==c`, 128/128, 0/65536) |
| Q2 — `warmup_pass_collapses_C1` | **TRUE** (#588 `b_vs_c`; re-confirmed §2 prefix-OFF `b_vs_c`) |
| Q2 — `served_stack_byte_deterministic_from_pass1` (best mitigation) | **TRUE** — prefix-OFF collapses C1 from pass 1; the deployed prefix-ON config needs one warmup pass |
| fern #597 MTP draft cold-start | **SEPARATE risk** — not inherited automatically; must be re-measured on int4_g128+MTP |

This card's re-measure run: W&B [`7w8mrmgy`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/7w8mrmgy) (group `operative-identity-coldstart`); artifact [`coldstart_poff.json`](coldstart_poff.json).

---

## 0. Background — the contract and the one live unknown

PR #588 ([`contract.md`](contract.md), run
[`n32yblfs`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/n32yblfs))
pinned the canonical operative-#319 bar: **literal byte-identity at warm steady state** of
the served int4 stack's free-running greedy decode (MAX_NUM_SEQS=1, spec-OFF, temp=0) to the
same int4 checkpoint's plain greedy AR decode, scored by the official
`check_greedy_identity.py` (verdict `GREEDY_IDENTICAL`, zero tolerance). `base_fullhead`
**passes it byte-for-byte**: warm/warm `b_vs_c` = `GREEDY_IDENTICAL`, **128/128** prompts,
**0/65536** divergent tokens.

The one live unknown #588 surfaced: a one-time **first-pass cold-start transient C1**. Pass
`a` (first inference after server load) dissents from **both** warm passes at the *identical*
67 prompts / 21815 tokens, every divergence a **PPL-neutral near-tie ≤ 4 ULP** (0 of 65536
exceed it; clean {0,2,4}-ULP ladder). Mechanism candidates: lazy Triton-JIT / FlashInfer-
autotune kernel-config settling on first inference, **and** prefix-cache cold-vs-warm
chunked-prefill numerics.

The canonical bar is zero-tolerance, but the **official harness scores SOME pass**. If it
scored a **cold** pass, a C1 reduction-order near-tie could trip the verifier to FAIL even
though it is semantically identical. This card answers: **is the scored pass warm or cold?**

---

## 1. Q1 — Does the official harness warm the endpoint before the scored greedy pass?

### Verdict: `official_harness_warms_before_greedy_identity` = **TRUE** (decisive)

The scored greedy-identity **candidate** is the harness-generated
`/state/decode_outputs.jsonl`, which the leaderboard verifier
(`check_greedy_identity.py`/`greedy_identity.py`) compares against the exact-greedy
reference. That candidate is produced by `run_decode_capture()`, which the in-job
orchestrator runs **strictly after** a full warmup + speed benchmark over the *same* prompt
suite. The control flow is unambiguous.

**Scoring path** (read in full; line numbers are exact):

1. **`run_hf_bucket_benchmark.py`** launches an HF Job whose command runs
   **`hf_bucket_single_job.py`** with `--enable-decode-capture`, writing the scored candidate
   to `/state/decode_outputs.jsonl`
   (`run_hf_bucket_benchmark.py:267-308`; `--enable-decode-capture` +
   `--decode-output-file /state/decode_outputs.jsonl` at `:290-296`). The leaderboard verifier
   `check_greedy_identity.py` scores **exactly this candidate** — it compares the candidate
   `decode_outputs.jsonl` against the exact-greedy reference via `greedy_identity.compare_files`
   (`check_greedy_identity.py:5-6,48-53,113`).

2. **`hf_bucket_single_job.py:main()`** executes a fixed sequence:
   - `:455-465` — start the participant server (`subprocess.Popen(serve_cmd, …)`).
   - `:466` — `wait_for_models(base_url, …)` polls **only** `GET /v1/models` until HTTP 200
     (`:182-197`). **No inference, no decode** — pure readiness probe. The server is *cold*
     at this instant.
   - `:479` — **`run_benchmark(...)`** runs **first**. It invokes `sglang.bench_serving`
     (`:200-247`) with **`--warmup-requests 4`** (`WARMUP_REQUESTS = 4`, `:39`; passed at
     `:234-235`) **plus the full benchmark** of `--num-prompts 128` × `--sharegpt-output-len
     512` at `--max-concurrency 1`, `--seed 1`, `ignore_eos=True`
     (`NUM_PROMPTS=128`/`OUTPUT_LEN=512`/`MAX_CONCURRENCY=1`/`SEED=1` at `:35-40`; flags at
     `:222-245`) over the **same** `eval_prompts_sharegpt.json` suite.
   - `:504-514` — **`run_decode_capture(...)`** runs **after** the benchmark returns
     (`:487-488` early-returns on benchmark failure, so decode-capture is reached only once
     the benchmark has completed). This is the call that produces the **scored** candidate
     `/state/decode_outputs.jsonl` (`run_decode_capture` body `:250-289`), at the identical
     128×512 / seed=1 geometry.
   - `:527-546` — PPL runs last.

3. By the time the scored decode-capture runs, the server has already served **4 explicit
   warmup requests + 128 full 512-token benchmark decodes** over the **identical** prompts.
   This consumes **both** legs of C1: the JIT/autotune-settling leg (every decode kernel
   shape used by the capture has already JIT-compiled and autotuned during the benchmark) and
   the prefix-cache leg (every one of the 128 prompts has already been prefilled once, so the
   capture reads warm cache entries). **The scored pass is thoroughly warm.**

**Consequence.** The official scored greedy-identity pass corresponds to #588's **warm**
passes (`b`/`c`), **not** the cold pass `a`. `base_fullhead`'s warm self-determinism is
`GREEDY_IDENTICAL` (`b_vs_c` 128/128, 0/65536 divergent) → the scored candidate is
byte-clean. **C1 is excluded from scoring.**

This is corroborated empirically by the **shipped** `int4_g128_lmhead @ 126.38` (HF Job
`6a2d5a96234ca64b60121aa5`, W&B
[`905tbujn`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/905tbujn)),
which **already passed** the official greedy-identity check — i.e. the harness empirically
scored the plain-AR int4 body as identical, exactly as the warm-scoring control flow predicts.

---

## 2. Q2 — Can the served stack be made byte-deterministic from pass 1?

**Design.** Re-run the **exact** #588 R=3 free-running census (official `decode_outputs.py`,
128 prompts × 512 tokens, seed=1, served M=1 spec-OFF) with the single diagnostic toggle
**`enable_prefix_caching=False`**, isolating C1 leg (1) (prefix-cache cold-vs-warm numerics).
Combined with #588 (the stock **prefix-ON** baseline) this is a clean **2-config factorial**
on the prefix-caching dimension. Driver: [`measure_coldstart.py`](measure_coldstart.py)
(reuses the #588 module's decode + official-verifier + near-tie helpers verbatim);
prefix-OFF artifact `coldstart_poff.json`.

Because `base_fullhead` is spec-OFF at M=1, each independent pass **is** a plain greedy AR
decode, so pass-pairwise scoring certifies the served stack against its own int4 AR
reference. Per config we score three pairs with the **official** `check_greedy_identity.py`:
- **`a_vs_{b,c}`** — cold pass 1 vs the two warm passes (does the mitigation collapse the
  cold transient *without* a warmup?).
- **`b_vs_c`** — warm/warm steady-state self-determinism (and, under prefix-OFF, the
  **combined** prefix-off+warmup arm: `b`,`c` are both prefix-off AND warmed by `a`).

The CUDA-graph capture is **not** a C1 candidate: it completes at server startup
(`cudagraph_num_of_warmups=1`, captured before pass `a`; #588 server log). C1's JIT leg is
JIT-*during-inference* (`_compute_slot_mapping_kernel`, #588 log `jit_monitor` at 13:54:36,
during pass `a`) + FlashInfer-autotune (`enable_flashinfer_autotune=True`). Disabling prefix
caching therefore isolates leg (1); any residual `a` dissent under prefix-OFF is leg (2).

### 2.1 The 2-config factorial (literal official verifier, zero tolerance)

| config (M=1, spec-OFF, temp=0) | `a_vs_{b,c}` (cold vs warm) | `b_vs_c` (warm/warm) |
| ------------------------------ | --------------------------- | -------------------- |
| **stock, prefix ON** [#588 `n32yblfs`] | **DIVERGENT** — 67/128 prompts, 21815/65536 tok, all ≤4 ULP near-ties | **`GREEDY_IDENTICAL`** — 128/128, 0/65536 |
| **prefix OFF** [this run `7w8mrmgy`, `coldstart_poff.json`] | **`GREEDY_IDENTICAL`** — 128/128, **0/65536** (both `a_vs_b` and `a_vs_c`) | **`GREEDY_IDENTICAL`** — 128/128, 0/65536 |

**Readout.** The single prefix-caching toggle flips the cold-vs-warm cell from DIVERGENT (67 prompts) to **`GREEDY_IDENTICAL` (0/65536)**. Because the toggle is the only difference between the two rows, the prefix-cache cold-vs-warm chunked-prefill numerics leg **fully accounts for C1**. The JIT/autotune leg (2) is shown **inert for greedy identity**: pass `a` under prefix-OFF still triggers first-inference Triton-JIT + FlashInfer-autotune settling (those are independent of prefix caching), yet `a==b==c` byte-for-byte — so that settling does not perturb the selected token. C1 had **one** leg, and it is the prefix cache.

### 2.2 Q2 verdicts

Prefix-OFF re-measure complete (W&B `7w8mrmgy`, `coldstart_poff.json`): all three passes
`a==b==c` are `GREEDY_IDENTICAL` (128/128, **0/65536** divergent on every pair). Peak GPU
19.11 GB; server startup 90 s; ~13.2 min/pass × 3.

- `prefix_caching_off_collapses_C1` = **TRUE** — prefix-OFF `a_vs_b` and `a_vs_c` are both
  `GREEDY_IDENTICAL` (0/65536). Removing the prefix-cache leg makes even the **literal first
  (cold) pass** byte-identical to the warm passes. The cold pass does **not** survive under
  prefix-OFF → the residual JIT/autotune leg is inert (it changes nothing in the selected
  token), so the prefix-cache leg was the **sole** source of C1.
- `warmup_pass_collapses_C1` = **TRUE** — established by #588's `b_vs_c` = `GREEDY_IDENTICAL`
  (pass `a` acts as a warmup; the two post-warmup passes are byte-perfect) on the **deployed**
  prefix-ON config, and re-confirmed by this run's prefix-OFF `b_vs_c` (128/128, 0/65536).
- `served_stack_byte_deterministic_from_pass1` (best mitigation) = **TRUE** — under prefix-OFF
  the served stack is byte-identical from pass 1 (no warmup needed); on the deployed prefix-ON
  config it is byte-identical from pass 2 (one warmup pass — exactly what the official harness
  performs, §1). Either mitigation yields a from-pass-1-deterministic scored candidate.

---

## 3. Synthesis — `operative_319_bar_is_warm_scored`

### Verdict: **TRUE** — the literal-warm operative-#319 bar is the operative contract, with **ZERO margin consumed**.

Q1 is decisive on its own: the official harness scores a **warm** server (the scored
decode-capture runs after a 4-request warmup + a full 128×512 benchmark over the same
prompts), so the first-pass cold-start transient C1 is **excluded** from the scored pass. The
scored candidate therefore corresponds to #588's warm passes, where `base_fullhead` is
**literally byte-identical** to its own int4 AR reference (`b_vs_c` `GREEDY_IDENTICAL`,
128/128, 0/65536). The near-tie envelope (`eps_star = 4 ULP`) that #588 measured to absorb C1
is **not consumed** under real scoring — it is pure, unused robustness margin.

Q2 provides an orthogonal confirmation: even a freshly-loaded **cold** server is cheaply made
byte-deterministic. The prefix-OFF re-measure (`7w8mrmgy`) collapses C1 entirely (`a==b==c`
`GREEDY_IDENTICAL`, 0/65536), localizing C1 to a **single leg** — the prefix-cache
cold-vs-warm chunked-prefill numerics — and showing the JIT/autotune leg is inert for greedy
identity. C1 is therefore eliminable two independent ways: a warmup pass
(`warmup_pass_collapses_C1 = TRUE`, the exact mechanism the harness uses) **or** disabling the
prefix-cache leg (`prefix_caching_off_collapses_C1 = TRUE`). It is a characterized, fully
eliminable warmup artifact, not a steady-state contract risk.

**Bottom line:** the operative-#319 bar a real int4 submission is scored on is the
**literal-warm** bar, and `base_fullhead` (and the shipped int4 body family) passes it
byte-for-byte with the entire 4-ULP near-tie envelope held in reserve.

---

## 4. Application to the fire candidate (fern #597, int4_g128+MTP)

This card does **not** measure #597 — it states the inheritance logic for the integrator.

- **What IS inherited.** #597's int4 **body** is the same int4 family as `base_fullhead` /
  the shipped `int4_g128_lmhead`. Its plain-AR greedy identity is body-set/head-invariant
  (stark #536) and its cold-start C1 is the same warmup artifact, which the official harness
  warms away (§1) exactly as for the shipped config. The body's warm-scored byte-identity is
  inherited.

- **What is NOT inherited — a SEPARATE risk.** #597 adds an **MTP draft model** and a
  **speculative-decode verify path** that `base_fullhead` (spec-OFF) does not have. Two
  distinct concerns, neither covered by this card's base-stack measurement:
  1. **Spec-dec verify identity (the dominant concern).** Whether the MTP-augmented served
     stack emits byte-identical tokens to plain greedy AR is a property of the **verify/accept
     path**, *not* of cold-start. Spec-dec is identity-preserving only if the target model's
     verify rejects every off-distribution draft and the accept/relocate logic is exact at
     temp=0. This must be re-measured on the int4_g128+MTP config (a free-running greedy
     census of the MTP-served stack vs its plain-AR reference) before a fire.
  2. **MTP draft-model cold-start.** The draft model has its **own** first-inference
     JIT/autotune settling. The harness warmup (4 requests + 128-prompt benchmark) **also**
     exercises the spec-dec path, so the draft's cold-start is warmed away **provided the
     benchmark runs spec-dec** (it does — same served stack). So the draft cold-start is
     **not** an additional scored-pass risk *beyond* concern (1), but it should be confirmed
     that #597's served decode is byte-deterministic from its warm steady state, just as #588
     did for the base.

  Net: #597's fire decision rides on a **separate spec-dec verify-identity census** on
  int4_g128+MTP, not on this card's base-stack cold-start result. Cold-start C1 is **not** an
  obstacle for #597 (the harness warms it); the open question for #597 is verify-path
  byte-identity, which is orthogonal.

---

## 5. Scope guard

This card reads the official scoring path and runs one local diagnostic census. It changes
**no** served/submission file, launches **no** job, submits **nothing** (`official_tps=0`).
The `enable_prefix_caching=False` toggle exists only inside the local diagnostic driver to
localize C1; it is **not** proposed as a serve change. The one new substantive finding is
that the operative-#319 bar is **warm-scored**, so the literal-warm byte-identity bar #588
pinned is the real leaderboard contract with the 4-ULP cold-start envelope held entirely in
reserve.
