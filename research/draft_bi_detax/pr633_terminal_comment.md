STUDENT wirbel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"analysis_only":true,"official_tps":0,"wandb_run_ids":["33ulzfg8"],"verdict":"DRAFT_AFFECTS_OUTPUT","draft_output_invariant":false,"bi_draft_selectable":false,"primary_metric":{"name":"draftswap_divergent_frac","value":0.11703},"test_metric":{"name":"floor_divergent_tokens","value":0}}

## Results

**Verdict: `DRAFT_AFFECTS_OUTPUT` — `draft_output_invariant=false`, `bi_draft_selectable=false`. The detax is dead for byte-exact #319, on two independent grounds. Important negative; stopped at the leg-1 gate as carded (no leg-3).**

The hypothesis ("in greedy spec-dec the emitted token stream is determined *entirely* by the target verify forward; the draft head's determinism is irrelevant to *which* tokens come out") is **empirically false for this int4 stack.** Swapping *only* the draft head — target weights, BI=1, K/M, prompts, seed all held fixed — flips **11.7% of emitted tokens** against a **perfectly clean run-to-run floor (0/32768)**. The draft provably changes the bytes.

This refutes the load-bearing assumption "…nor on `t_{i+1…K}` *once the target is batch-invariant*." The target is **not** batch-invariant here: `VLLM_BATCH_INVARIANT=1` covers the bf16 SM80 mm/bmm/softmax path but **not the int4 Marlin W4A16 verify GEMM** (consistent with #607/#616). So the target's argmax *does* depend on which draft tokens share its verify batch.

### Leg 1 — theory-confirm (served, int4 target, BI=1, greedy, K=6, 64×512, prompt_sha parity)

Four decode streams captured with `gen_greedy_reference --mode served`, then re-compared byte-exact with `scripts/local_validation/greedy_gate.py`:

| Comparison | What varies | Verdict | Divergent tokens | Divergent prompts | onset (min/med/max) |
|---|---|---|---|---|---|
| **Floor**: `arm_x_qat` vs `arm_x_qat_rep` | same drafter, fresh boot | **GREEDY_IDENTICAL** | **0 / 32768 (0.0%)** | 0 / 64 | — |
| **Draft-swap**: `arm_x_qat` vs `arm_y_alt` | **drafter only** (qat → published `gemma-4-E4B-it-assistant`) | **DIVERGENT** | **3835 / 32768 (11.7%)** | 32 / 64 | 173 / **421** / 509 |
| M-confound: `ref_m1ar` vs `arm_x_qat` | M=1 AR vs M=7 spec (qat) | DIVERGENT | 18440 / 32768 (56.3%) | 55 / 64 | 3 / 99 / 477 |
| M-confound: `ref_m1ar` vs `arm_y_alt` | M=1 AR vs M=7 spec (alt) | DIVERGENT | 18614 / 32768 (56.8%) | 57 / 64 | 3 / 101 / 504 |

- **Floor is clean (0/32768)** → the spec-ON BI=1 path is run-to-run byte-deterministic, so the draft-swap divergence is **100% attributable to the draft head** (not boot noise).
- **Draft-swap diverges (11.7%, 32/64 prompts).** Acceptance differs across drafters (qat **3.589** / qat-rep **3.563** / alt **3.185**), confirming the two heads genuinely propose differently. **A different draft → a different emitted byte stream. That is the refutation.**
- Onset is **LATE** (median 421/512) and only half the prompts diverge — signature of a *rare* draft-composition-sensitive tie-flip that seeds a free-running cascade, not pervasive per-position corruption.

### Mechanism — why the draft leaks into the output

In a verify forward the target processes `prompt + K` draft tokens as one M=`K+1`-row batch. With a *truly* batch-invariant kernel, logit_i would depend only on context `[0…i-1]`. **Marlin int4 is not batch-invariant**: its tile/reduction order depends on the M-row batch composition, so the low bits of logit_i depend on the draft tokens at positions `i+1…K`. At int4-grid ties the argmax flips → a different accepted/emitted token → cascade. Same root cause as #607 (`SPEC_STRUCTURALLY_BREAKS_319`) and #616 (per-step flip 0.43%, int4-grid ties). **BI=1 does not, and here cannot, close this** — it never touches the Marlin path.

### Divergence character — quality-neutral tie-flips (tolerance-contract caveat)

The flips are coherent near-synonym swaps, not corruption (decoded first-divergence tokens, qat vs alt):

```
@tok214  ctx="…unless they explicitly state"   ' the'        vs  ' they'
@tok369  ctx="…By standardizing the response"  ' options'    vs  ' format'
@tok241  ctx="…(Incorrect. The"                ' standard'   vs  ' test'
@tok358  ctx="…dependent on the structural"    ' features'   vs  ' dynamics'
@tok272  ctx="…due to scattering with the"     ' remaining'  vs  ' gas'
```

This matches #616 (100% of flips < 0.5 nats, int4-grid ties, τ=0.3 nat → 100% rescued). So under a **tolerance** #319 contract these are interchangeable and the detax *might* be a viable speed lever — **but that is moot here**: it still fails strict byte-exact #319 (leg-1) and is not implementable without engine surgery (leg-2). I did **not** run a logprob-gap confirmation since the leg-1 gate already stops the experiment.

### Leg 2 — is BI draft-selectable? **No. `bi_draft_selectable=false`.**

Read `vllm/model_executor/layers/batch_invariant.py`: `enable_batch_invariant_mode()` installs a **permanent process-global** `torch.library.Library("aten","IMPL")` (mm/addmm/matmul/linear/softmax/_softmax/_log_softmax/mean/bmm), monkeypatches `torch.bmm`, and sets `torch.backends.cuda.matmul.*` flags. There is **no disable function, no call-time mode check in the wrappers, no scoping/teardown**; `init_batch_invariance()` decides once at init from `envs.VLLM_BATCH_INVARIANT`. The draft head also runs under a **piecewise cudagraph captured while BI is active**, so the BI kernels are baked in at capture time. A harness-only wrapper around `propose()` **cannot** make the draft BI=0 while the target stays BI=1.

**Exact engine change that would be required (out of the PR's harness-only scope — not implemented):** either
- **(A)** convert the permanent `Library` registration into a scoped `TorchDispatchMode` (upstream thinking-machines pattern) and enter it *only* around the target verify forward, leaving the draft outside; **or**
- **(B)** add a thread-local/contextvar checked inside every wrapper that redispatches to the original aten kernel when disabled, restore `torch.bmm` + the matmul backend flags in-scope, **and** capture the draft piecewise-cudagraph with BI disabled (capture-time, not replay-time).

Both touch `submissions/**`/engine internals, which the PR explicitly forbids ("Do NOT modify any submission / served file"). Even if implemented, leg-1 already shows it would **break** byte-exact #319 on the int4 Marlin path, so it is not a fix for the strict contract.

### ⚠️ Flag for the advisor — the leg-1(a) control as carded is incorrect

The card describes leg-1a as: *"BI=1 spec output vs AR M=1 reference … must be 0 flips — land #623's already-passing identity, re-assert as control."* Both halves don't hold:
- **It is 56% DIVERGENT, not 0** (the M-confound rows above): M=8 spec verify vs M=1 AR is exactly the M-dependent Marlin break #607 measured (`passes_strict_319=False`), with EARLY onset (median 99).
- **#623 never passed byte-exact greedy** — it passed a **PPL ≤2.42** gate (2.0055) and a *local wall-TPS* screen (152.29). PPL-pass ≠ byte-identity; #607/#616 are the byte-exact census and they say this path breaks strict #319.

So the correct run-to-run floor for leg-1 is **same-drafter fresh-boot** (the `arm_x_qat` vs `arm_x_qat_rep` row, 0/32768), which I used instead. The M=1-AR stream is a *confound* (changes M **and** the draft), not a clean control — including it as "must be 0" would have masked the result.

### Public evidence

Greedy spec-dec output-equivalence (Leviathan et al., arXiv:2211.17192; Chen et al., arXiv:2302.01318) is a statement about **exact arithmetic**: the verified token equals the target's own argmax *given identical logits*. It says nothing about finite-precision kernels whose logits shift with batch composition. The hypothesis inherits that exact-arithmetic guarantee but the int4 Marlin verify violates its premise — which is precisely what the 11.7% draft-attributable divergence measures.

### Config / commands

- **Submission:** `submissions/int4_mtp_batchinv` (int4 W4A16 `google/gemma-4-E4B-it-qat-w4a16-ct` target + gemma4_assistant MTP drafter, K=6/M=7), `VLLM_BATCH_INVARIANT=1`, vLLM 0.22.0, dev307, single A10G.
- **Capture (per arm):** `python -m scripts.local_validation.gen_greedy_reference --mode served --submission submissions/int4_mtp_batchinv --num-prompts 64 --output-len 512 --seed 1 --out research/draft_bi_detax/<arm> --port 8001` (drafter swapped via `DRAFTER_MODEL`; `ref_m1ar` adds `--spec-off`).
- **Compare + log:** `.venvs/vllm022/bin/python research/draft_bi_detax/log_results.py` → `research/draft_bi_detax/summary.json` + W&B.
- **Peak memory:** model 9.85 GiB + KV 8.51 GiB ≈ 18.4 GiB working set (`GPU_MEMORY_UTILIZATION=0.90` of 22.5 GiB A10G), no OOM.
- **Local only — no HF Job, no submission.** `analysis_only=true`, `official_tps=0`, as carded. No served/submission file modified (research harness only).

### W&B

- **`33ulzfg8`** — `wirbel/draft-bi-detax`, group `optionb-draft-bi-detax`, job_type `analysis`. ([link](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/33ulzfg8))

### What happened — honest analysis

- **The hypothesis is correct in exact arithmetic and false for the deployed int4 stack.** The detax assumes the target verify is batch-invariant; under int4 it is not (Marlin is uncovered by BI=1), so the draft's proposals reach the emitted bytes through the verify batch composition. Measured: **11.7% of tokens flip on a draft-swap, against a 0/32768 floor.**
- **Two independent kills.** (1) Draft *does* affect output (leg-1, decisive). (2) BI is process-global, not draft-selectable (leg-2). Either alone sinks "BI=0 draft + BI=1 target, byte-exact." Leg-3 (the A/B TPS measurement) was correctly gated out — there is no byte-exact arm to measure.
- **Not corruption — quality-neutral ties.** The flips are int4-grid near-synonyms (#616), so this path is *not* a quality regression. If Option-B ever moves to a **tolerance** #319 contract (≥0.3 nats), the draft-BI detax becomes a *candidate* speed lever again — but it would still need the engine surgery in leg-2, and it would never satisfy *strict* byte-exactness.

### Suggested follow-ups

1. **Re-target the detax at the real blocker.** The detax cannot help under strict #319 (the draft leaks through Marlin). It only becomes interesting under a **tolerance** #319 contract — which is the same contract-policy decision #616 already surfaced (τ=0.3 nats → 100% rescued). Decide that policy first; the kernel work is downstream of it.
2. **If a tolerance contract is adopted**, the minimal engine change is leg-2 option (A): a scoped `TorchDispatchMode` wrapping only the target verify, draft captured BI-off. Then leg-3 (paired A/B, BI=0-draft vs BI=1-both anchor 152.29) measures how much of the +101 TPS the detax recovers — under tolerance-identity, not byte-identity.
3. **Pin the deployed decode-branch break** (optional, diagnostic): these captures use the served decode path end-to-end, so unlike #616's prefill-branch lower bound this is the *deployed* number. The 11.7% draft-swap divergence is a clean, deployment-faithful measurement of the Marlin batch-composition sensitivity that a future batch-invariant int4 GEMM would have to close.
