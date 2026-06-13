# Served-gate validity audit — `fa2sw_precache_kenyan` (PR #38)

**TL;DR.** The ~424.5 TPS `fa2sw_precache_kenyan` frontier stack **fails the served greedy
gate (DIVERGENT 30/32 prompts, spec-ON vs spec-OFF, batch=1)** — but the failure is **NOT
caused by speculative decoding, and NOT specific to the fa2sw stack.** The served greedy
decode is **non-deterministic run-to-run on this A10G**:

- fa2sw spec-OFF vs spec-OFF, **same GPU, fresh reload only → DIVERGENT 28/32** (spec can't be
  the cause — it's off in both).
- the **plain vLLM 0.22.0 int4 baseline** (official `w4a16-ct`, no spec, none of the fa2sw
  kernels), M=1 → **DIVERGENT 29/32** run-to-run (the nondeterminism isn't fa2sw-specific).
- the **one reproducible config found**: fa2sw with **`FA_SLIDING=0`** (the FA2 sliding-window
  attention patch disabled, falling back to the captured-graph path) → **GREEDY_IDENTICAL
  0/32** run-to-run.

So byte-exact greedy identity is **not a usable validity instrument for the default served
paths**, because they don't reproduce their own greedy output. The strict M=1 bar is **not
"over-conservative"** — it correctly rejects a decode that no reference (however lenient) can
pin down. Reproducibility *is* achievable (FA_SLIDING=0), which localizes one nondeterminism
source to the FA2 sliding-window kernel.

---

## 1. What was asked (PR #38)

> Is the ~420 TPS `fa2sw_precache_kenyan` frontier stack greedy-valid under the SERVED gate
> (spec-on vs spec-off, batch=1), and is our strict M=1 greedy-identity bar over-conservative
> relative to the leaderboard's actual served gate?

Background: PR #32 found `fa2sw_precache_kenyan` DIVERGENT 27/32 vs its own **M=1 AR offline**
reference, yet it is leaderboard-VALID at ~424.5 TPS. The hypothesis: the leaderboard enforces
a weaker *served* gate that fa2sw passes, while the offline M=1 bar is too strict, so
spec-decode submissions could be routed through the served gate.

## 2. Method (LOCAL ONLY — no HF Job)

All runs: `google/gemma-4-E4B-it`, 32 prompts × 512 tok, greedy (temp=0, top_p=1, top_k=0),
seed 1, single A10G (23 GB), via the corrected per-submission keying from PR #32. Captured
through the official `decode_outputs.py` api_server path so the comparison isolates the
optimization-under-test, not cross-engine FP noise (PR #19).

- **fa2sw spec-OFF** reference: `gen_greedy_reference --mode served --submission
  submissions/fa2sw_precache_kenyan --ref-env SPECULATIVE_CONFIG=` (empty `SPECULATIVE_CONFIG`
  disables MTP K=7; FA_SLIDING / ONEGRAPH / LM_HEAD_PRUNE / FUSED_SPARSE_ARGMAX / PRECACHE all
  unchanged).
- **fa2sw spec-ON** candidate: the full submission (MTP K=7 drafter).
- **fa2sw FA_SLIDING=0**: as spec-OFF, plus `--ref-env FA_SLIDING=0` (the only other change).
- **AR baseline (int4)**: `int4_g128_lmhead`'s weights are not bundled and cannot be rebuilt
  locally (`build_quant.py` needs `/workspace/gemma_build/qat_unq`, which is absent), so the
  AR-only control uses the **official `google/gemma-4-E4B-it-qat-w4a16-ct` int4 checkpoint**
  served through the canonical `vllm_baseline` path (plain vLLM 0.22.0, no spec, none of the
  fa2sw custom kernels) via `--ref-env MODEL_ID=<w4a16-ct snapshot>`. Caveat: w4a16-ct is a
  *proxy* for int4_g128_lmhead — it does NOT carry int4_g128_lmhead's determinism-tuned untied
  int4 lm_head (see §6).
- **Batch regime**: `decode_outputs.py` issues prompts **sequentially** (`for record in
  records:` + synchronous POST, one request in flight), so every run is effectively **M=1**
  regardless of `MAX_NUM_SEQS`. The baseline divergence is therefore genuine M=1 run-to-run
  nondeterminism, not a batch-composition artifact.
- All pairwise comparisons via the official verifier (`greedy_gate.compare`); driver
  `research/_localrun/_compare.py`.

> Note on the PR's literal Step-1 command: `fa2sw`'s `serve.py` does **not** honor
> `SENPAI_REFERENCE_MODE`, so the documented `--spec-off` flag does not disable spec for this
> submission. The correct disable is `--ref-env SPECULATIVE_CONFIG=` (empty value), which the
> `[serve]` log confirms (`'SPECULATIVE_CONFIG': ''`). `validate_submission` runs the greedy
> check by default, so `--greedy-check` / `--reference-mode served` are not real flags.

## 3. Results (32 prompts × 512 tok = 16 384 tokens)

Token-divergence fraction = `total_divergent_tokens / total_tokens_compared`. The official
verifier counts *all* differing positions in the compared range plus length delta, so it is
**cascade-inflated** — one early flip taints the rest of the sequence. Read it together with
the onset distribution, **not** as a per-step flip hazard.

| # | comparison | verdict | divergent | token-div | onset min/med/max |
|---|---|---|---|---|---|
| 1 | **Step 1 SERVED GATE** — fa2sw spec-OFF ref vs **spec-ON** cand | **DIVERGENT** | **30/32** | **63.0%** | 8 / 110 / 491 |
| 2 | **DETERMINISM FLOOR (same GPU)** — fa2sw spec-OFF vs spec-OFF (reload only) | **DIVERGENT** | **28/32** | **53.8%** | 11 / 124 / 483 |
| 3 | committed spec-OFF ref vs spec-OFF (run A) | DIVERGENT | 30/32 | 58.0% | 1 / 138 / 452 |
| 4 | committed spec-OFF ref vs spec-OFF (run B) | DIVERGENT | 28/32 | 47.4% | 1 / 201 / 491 |
| 5 | fa2sw spec-OFF (run A) vs spec-ON cand | DIVERGENT | 29/32 | 64.0% | 1 / 97 / 486 |
| 6 | **CONTROL — FA_SLIDING=0** spec-OFF (A) vs spec-OFF (B), same GPU | **GREEDY_IDENTICAL** | **0/32** | **0.0%** | — |
| 7 | **CONTROL — AR baseline int4** (plain vLLM 0.22.0, w4a16-ct) run A vs run B | **DIVERGENT** | **29/32** | **59.1%** | 3 / 106 / 501 |

Onset-signature diagnostics (distinguishing run-to-run nondeterminism from a real decode change):

| diagnostic | verdict | divergent | token-div | onset min/med/max | reading |
|---|---|---|---|---|---|
| FA_SLIDING=0 (A) vs FA_SLIDING=1 spec-OFF | DIVERGENT | 28/32 | 53.8% | 11 / 124 / 483 | **late/stochastic** → same as the floor; FA_SLIDING=0 is the *same decode made reproducible*, not a different one |
| int4-baseline (A) vs fa2sw spec-OFF | DIVERGENT | 32/32 | 96.8% | **0 / 3 / 35** | **early/systematic** → what a *genuinely different* decode (different checkpoint) looks like |

### Reading the data

**Row 2 is the lynchpin.** Two runs with byte-identical config (spec OFF both times), the
*same weights*, *same GPU*, *same container*, differing only in a fresh model load, diverge on
28/32 prompts. Spec is OFF in both, so this is not spec-introduced — it is pure run-to-run
nondeterminism. Against that floor, the spec-ON-vs-OFF "gate" (row 1: 30/32, 63%) is **not
separable** from spec-OFF-vs-OFF (row 2: 28/32, 53.8%). Across the six fa2sw comparisons the
token-divergence fraction is **47–64% with spec on *or* off**, and divergent prompts sit at
**28–30/32 in every pair**. The spec component adds no divergence distinguishable from noise.

**Row 7 generalizes it beyond fa2sw.** The plain vLLM 0.22.0 int4 baseline — no spec, no
FA_SLIDING, no ONEGRAPH, no LM_HEAD_PRUNE, M=1 — is *also* non-deterministic run-to-run
(29/32, late onset 106). So the nondeterminism is a property of the **default vLLM served
greedy path on this GPU** (non-deterministic kernel reductions across process launches that
flip argmax on near-ties), not of the fa2sw optimizations.

**Row 6 is the one reproducible config.** fa2sw with `FA_SLIDING=0` is byte-identical
run-to-run (0/32). The diagnostic table shows FA_SLIDING=0 diverges from FA_SLIDING=1 only by
the *same late/stochastic amount* (28/32, onset median 124) that FA_SLIDING=1 diverges from
itself — i.e. FA_SLIDING=0 sits inside the cloud of FA_SLIDING=1 draws. So the FA2
sliding-window kernel **injects run-to-run reduction noise on top of an otherwise-stable
decode**; the captured-graph path (FA_SLIDING off) removes it. This localizes one
nondeterminism source to the FA2 sliding kernel.

## 4. Divergence classification (PR Step 2: a / b / c)

Per the PR's taxonomy — (a) spec-introduced, (b) batch-FP-noise, (c) both — **the fa2sw served
greedy divergence is class (b): floating-point / run-to-run nondeterminism, explicitly NOT
(a).** Evidence:
- spec-OFF-vs-spec-OFF (row 2) reproduces the full divergence magnitude with spec disabled.
- the plain int4 baseline (row 7) reproduces it with *none* of the fa2sw machinery.
- onset is **late and stochastic** (median ≈ 106–200 of 512, on a different ~28–30/32 subset
  each pairing) — the FP-reduction-noise signature (PR #19), not the **early-and-systematic**
  signature of a lossy/mis-verifying optimization (contrast the int4-vs-fa2sw diagnostic:
  onset median 3, 32/32, 96.8%).
- FA_SLIDING=0 removes it entirely (row 6) without changing the decode (diagnostic table).

The 27/32 M=1-**offline** divergence from PR #32 is the same phenomenon through a noisier lens
(offline batched decode adds ~20% FP-reduction divergence vs the served path, PR #19) — not a
distinct "strict bar" failure.

## 5. Map to the leaderboard gate (PR Step 3)

**Is the strict M=1 bar over-conservative?** No. The premise was "fa2sw passes the served gate
but fails strict M=1, so M=1 is too strict." But **fa2sw fails the served gate too** (30/32)
and fails greedy-identity against *its own spec-OFF self* (28/32). The bar is not the problem —
**fa2sw (as shipped, FA_SLIDING=1) is non-reproducible**, so no reference, at any strictness,
can hold it to a fixed token sequence. A byte-exact gate correctly refuses to certify a decode
that isn't a function of its inputs. `m1_bar_is_over_conservative = 0`.

**Then how is `fa2sw_precache_kenyan` leaderboard-VALID at ~424.5 TPS?** Because the official
job harness does **not run a greedy-identity gate.**
`official/.../speed_benchmark/scripts/hf_bucket_single_job.py` runs benchmark + decode-capture
+ PPL + summary; it *captures* `decode_outputs` for downstream audit but **never compares them
to a greedy AR reference.** Leaderboard validity rests on **PPL ≤ threshold + completion +
modality preservation**, not on greedy token-identity. The strict greedy gate is a *separate,
local* instrument — and for a non-reproducible stack it cannot return a stable verdict at all.

**Practical answer to "route spec-decode submissions through the served gate":** the served
greedy gate is reliable only for submissions whose decode is *reproducible* in the first place.
For the default path and for fa2sw-with-FA_SLIDING it is uninformative, because the gate's core
assumption — that a fixed greedy config yields a fixed token sequence — does not hold. The fix
is not a stricter-or-looser identity threshold; it is either (a) make the decode reproducible
(e.g. FA_SLIDING=0 / captured-graph kernels), or (b) replace byte-exact identity with a
determinism-robust check (PPL-band, or top-1 agreement over N reloads with a stochastic
allowance).

## 6. Reconciling with PR #4 (`int4_g128_lmhead` GREEDY_IDENTICAL 128/128)

My int4 control (official **w4a16-ct**) is non-deterministic run-to-run (29/32), which appears
to contradict PR #4's report that **int4_g128_lmhead** is GREEDY_IDENTICAL 128/128. The likely
reconciliation is the **lm_head**: `int4_g128_lmhead` ships a deliberately re-quantized
**untied int4 lm_head** (`build_quant.py`, "untied int4 lm_head ... All modalities preserved"),
whereas w4a16-ct does not. Greedy reproducibility is ultimately decided by the **argmax
tie-margin** at the final logits — a sharper / better-conditioned lm_head produces larger
margins that survive the same kernel-level FP-reduction noise, while w4a16-ct's lm_head leaves
more near-ties that flip. So:
- w4a16-ct non-reproducible **does not prove** int4_g128_lmhead is non-reproducible — they
  differ in exactly the dimension (lm_head numerics) that governs greedy stability.
- I could **not** test int4_g128_lmhead directly (weights unbuildable: no `qat_unq` source).
  **Recommend the advisor verify int4_g128_lmhead's run-to-run determinism directly** (two
  fresh reloads, compare), since it is the one config the team relies on and it is the cleanest
  test of "reproducibility is achievable via lm_head conditioning."

This view unifies everything: greedy reproducibility on this stack is governed by argmax
tie-margins, which depend on **(a)** attention/GEMM kernel reduction determinism (FA_SLIDING=1
breaks it; captured-graph FA_SLIDING=0 restores it; default-vLLM attention also breaks it) and
**(b)** lm_head numerical conditioning (int4_g128_lmhead's untied int4 head is tuned for it;
w4a16-ct is not). The served greedy gate is meaningful **only** when a submission is engineered
to win on both.

## 7. Conclusion

1. `fa2sw_precache_kenyan` **fails the served greedy gate (30/32 DIVERGENT)** — *primary metric
   0.0*.
2. The failure is **not spec-introduced and not fa2sw-specific**; it is **run-to-run
   nondeterminism** of the served greedy decode (fa2sw spec-OFF vs spec-OFF 28/32; plain int4
   baseline 29/32; both M=1, same GPU).
3. The **M=1 bar is not over-conservative** (*test metric `m1_bar_is_over_conservative` = 0*);
   a non-reproducible decode cannot satisfy *any* byte-exact gate, and the bar is right to
   refuse it.
4. The leaderboard accepts fa2sw because the **official job harness enforces no greedy-identity
   gate** — validity there is PPL + completion + modalities, not token identity.
5. Reproducibility **is** achievable: fa2sw with `FA_SLIDING=0` is byte-identical run-to-run
   (0/32) and is the *same* decode (late/stochastic delta vs FA_SLIDING=1), localizing one
   nondeterminism source to the FA2 sliding-window kernel.

## 8. Suggested follow-ups

- **Verify `int4_g128_lmhead` run-to-run determinism directly** (the PR #4 reconciliation in
  §6). If it reproduces, confirm the untied-int4-lm_head tie-margin is the reason — and the
  served gate is usable for lm_head-conditioned submissions.
- **Quantify the FA_SLIDING=0 TPS cost.** FA_SLIDING is a speed optimization; trading it for
  determinism likely costs throughput. Measure the TPS delta of the captured-graph FA_SLIDING=0
  path so the team can weigh "reproducible + gateable" vs "~424 TPS but ungateable."
- **Localize the residual sources**: with FA_SLIDING off the fa2sw path is deterministic, but
  the *default* vLLM attention (int4 baseline) is not — investigate the default-attention
  nondeterminism (splitKV atomics / autotune selection) so a deterministic baseline gate is
  possible.
- **Adopt the onset-signature diagnostic** (`greedy_gate.onset_summary`) as standard gate
  output: late+stochastic ⇒ nondeterminism (re-run / tolerance), early+systematic ⇒ real
  divergence (reject). This pair of comparisons (§3) shows it cleanly separates the two.
- **Consider a determinism-robust validity metric** for non-reproducible stacks (PPL-band
  agreement, or top-1 agreement averaged over N reloads with a stochastic-divergence
  allowance).

---

_Artifacts live under `research/_localrun/{fa2sw-served-gate-32, fa2sw-specoff-rerun-32,
fa2sw-specoff-run2-32, fa2sw-faoff-runA-32, fa2sw-faoff-runB-32, int4-ctl-runA-32,
int4-ctl-runB-32}` (gitignored scratch — regenerate via the §2 serve commands). Committed
fa2sw spec-OFF reference:
`research/greedy_reference/workspace__senpai__target__submissions__fa2sw_precache_kenyan__google__gemma-4-E4B-it/`.
Every comparison above is reproducible with the committed verifier
(`python -m scripts.local_validation.greedy_gate --reference <A> --candidate <B>`); the local
convenience driver that batches them is `research/_localrun/_compare.py`. Peak GPU memory:
fa2sw stack ~19.3 GB / 23 GB; int4 w4a16-ct baseline lighter (11 GB weights)._
