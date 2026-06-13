# Batch-invariant vLLM × int4-Marlin-spec — de-risk + decisive result (PR #19, Steps 1–4)

**Author:** kanna · **Date:** 2026-06-13 · **Mode:** Step 1 CPU/source audit + Steps 2–4 LOCAL A10G, no HF Job.
**Branch:** `kanna/batch-invariant-vllm-spec` · **Builds on:** PR #5 (`kanna/int4-mtp-drafter`).

> ## ⛔ TERMINAL RESULT (Step 3, measured) — **BLOCKED / definitive negative**
> `VLLM_BATCH_INVARIANT=1` is **compatible** with the full vLLM-0.22.0 Gemma-4 + int4-Marlin + MTP-spec
> stack and was **confirmed active**, but it does **NOT** make int4+MTP spec decode greedy-identical:
> **int4 ON = 0.376%/tok flip (5/32), statistically unchanged from int4 OFF = 0.332%/tok (6/32).**
> The **bf16 positive control discriminates the cause**: bf16 ON (no Marlin, full aten coverage) is
> **still DIVERGENT at 0.111%/tok (16/32)** — so two *independent, un-covered* sources defeat greedy
> identity: **(a)** the int4 **Marlin** weight-GEMM (a `_C` op the aten override can't reach) and
> **(b)** a **non-aten spec-verify component** (the measured vLLM-#27433 "no spec-decode integration").
> Neither the int4 target nor a bf16 target is rescued. The Step-1 audit below (verdict "GO to test,
> no version bump") was correct as a *pre-test* call — there was zero downside to testing — and the
> decisive Step-3 measurement is in **"Step 3 — EMPIRICAL RESULT"** near the end of this doc.
> Next lane: **verify-rollback** (arxiv 2601.17768), not system-wide invariant kernels.

## TL;DR verdict (Step 1, pre-GPU) — **GO (safe to test), NO vLLM version bump**

The PR brief assumed we would have to *pin a newer vLLM* that ships batch-invariant
kernels, and worried that the engine bump could break Gemma-4 / Marlin / spec-decode.
**That assumption is wrong in our favour: vLLM `0.22.0` — the exact version PR #5 already
validated end-to-end (Gemma-4 multimodal + int4 W4A16 Marlin + `gemma4_assistant` MTP spec)
— already ships the full batch-invariant kernel set.** We enable it with a single env var
`VLLM_BATCH_INVARIANT=1`; the engine wires it automatically in the GPU worker. So the entire
"newer engine regression" risk surface evaporates — there is no version change at all.

> **⚠️ CRITICAL UPDATE (literature pass, appended 2026-06-13).** vLLM's *own docs* state the
> batch-invariant feature **"does not currently integrate with speculative decoding"**, and
> vLLM issue **#27433** files spec-decode support under *"Nice to have … this might be hard."*
> The documented, supported use cases are **cross-request determinism + RL training stability**,
> **not** the spec-decode M=K+1 verify forward we are targeting. This does **not** contradict our
> kernel-level result — we empirically confirmed (`confirm_invariant.py`, A10G) that the
> `aten::mm` override IS batch-invariant (M=1 vs M=7 rows **bit-identical**, max|diff|=0; OFF
> control max|diff|=1.0) — but it means the *full* spec verify path may carry a batch-variant
> component **outside** the covered aten ops (attention metadata, the rejection sampler, or a
> non-aten verify kernel). **Net effect on the verdict: still GO to test (zero downside to the
> int4 floor), but shift the prior toward "ON may remain DIVERGENT."** Step 3 is the arbiter; if
> ON is still divergent, that *corroborates the vLLM disclaimer with a measured flip rate* — a
> definitive program-steering negative. See the "Literature corroboration" section at the end.

| dimension | finding | confidence |
|---|---|---|
| **Q3 version + mechanism** | `0.22.0` already ships `batch_invariant`; enable via env `VLLM_BATCH_INVARIANT=1`; worker auto-calls `init_batch_invariance()` | **HIGH (installed source)** |
| **Q1 Marlin GEMM — force a non-Marlin fallback?** | **NO.** Marlin = `ops.marlin_gemm` (a `_C` custom op). Overrides only touch `aten::{mm,addmm,matmul,linear,bmm}`. Marlin path untouched → **int4 bandwidth win preserved.** | **HIGH (source)** |
| **Q1 Marlin GEMM — made batch-invariant?** | **NO** (not an aten op). But the int4 body GEMMs are *plausibly already M-invariant* (fixed per-row K-reduction, no default split-K atomics). | MED — empirical (Step 3) |
| **Q2 verify-forward coverage** | RMSNorm, softmax/`_log_softmax`/`mean`, attention (`num_splits=1`), **and the bf16 `lm_head`/argmax matmul** are all made batch-invariant. The verify forward's argmax-determining GEMM IS covered. | **HIGH (source)** |
| **Q4a Gemma-4 multimodal** | unchanged from 0.22.0 → already proven in PR #5; all modalities stay loaded | **HIGH** |
| **Q4b int4 W4A16 Marlin sm_86** | unchanged from 0.22.0 → proven in PR #5 | **HIGH** |
| **Q4c spec decode w/ MTP draft** | unchanged from 0.22.0 → proven in PR #5 | **HIGH** |
| **Q4d {8,4} attn-group fix (#43543) native?** | **NO** — not in 0.22.0 (that is why PR #5 monkeypatches it). **KEEP the monkeypatch.** (PR brief's guess that it'd be obsolete is wrong — no version bump.) | **HIGH** |
| **Q4e selected attn backend supports invariance?** | TRITON_ATTN (our stack's backend) `supports_batch_invariance()==True`; FlashAttn/Flex too → selector will not reject | **HIGH (source)** |
| **Q5 TPS cost** | invariant Triton persistent-matmul + `num_splits=1` attn + TF32 off → slower than tuned cuBLAS/FA. Magnitude empirical (Step 4). On int4 the dominant weight-GEMM stays Marlin, so the hit is bounded to the bf16 ops + attention. | pending empirical + lit |

**Decision:** PROCEED to Step 2/3. The make-or-break "does enabling it kill Marlin" fear is
resolved NO from source — enabling batch-invariance leaves the int4 Marlin path byte-for-byte
unchanged, so there is **zero downside to the int4 throughput floor** from merely testing it.
Whether the *covered* ops (esp. the bf16 lm_head + rmsnorm + attention) plus Marlin's intrinsic
M-invariance are **sufficient** to push the int4 M=K+1 verify forward to bit-identity with the
M=1 AR forward is the one genuinely-empirical question, and Step 3 answers it directly. The bf16
target arm is the clean positive control (PR #5: bf16 flips 0.72%/tok and is *all-aten*, so it is
fully covered → should go GREEDY_IDENTICAL with the flag on).

---

## Our exact stack (recap)

- vLLM **0.22.0** (V1 engine), transformers 5.9.0, Python 3.12, single A10G (sm_86, Ampere, 23 GB).
- Target: `google/gemma-4-E4B-it-qat-w4a16-ct` — compressed-tensors W4A16 `pack-quantized`, Marlin.
  - `quantization_config.ignore` includes **`lm_head`** (+ all vision/audio tower projections);
    `tie_word_embeddings: True` → **the lm_head is the tied bf16 embedding matrix, NOT int4.**
- Drafter: `google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant` (`gemma4_assistant` → `Gemma4MTPModel`),
  bf16, K=6, vLLM rejection-sampler spec path. (Draft GEMMs are bf16 → aten → covered.)
- Attn backend: **TRITON_ATTN** (draft + target global layers share it; the `{8,4}` monkeypatch
  splits them into their own attention groups by `num_heads_q`).

## The measured problem (PR #5)

At temp=0 the spec-decode **M=K+1 batched verify forward** flips argmax vs the **M=1 AR forward**
at **0.33%/tok (int4) / 0.37% (fp8) / 0.72% (bf16)** — precision-INDEPENDENT, so it is a
batch-shape numerical-nondeterminism effect (not a quantization near-tie). Over 512 tokens this
compounds to a DIVERGENT strict-bit-exact greedy verdict, killing the submission. K0-vs-K0
control is identical → it is 100% the spec verify path.

---

## Q1 (MAKE-OR-BREAK) — Does batch-invariance touch the int4 Marlin W4A16 GEMM?

**Answer: it does NOT force a fallback (int4 win safe), and it does NOT make Marlin itself
invariant (Marlin is not an aten op).**

`enable_batch_invariant_mode()`
(`vllm/model_executor/layers/batch_invariant.py:905`) installs its overrides on the **`aten`**
library only:

```python
_batch_invariant_LIB = torch.library.Library("aten", "IMPL")
if current_platform.is_device_capability_family(80):   # A10G sm_86 → True (see Q4e)
    _batch_invariant_LIB.impl("aten::mm",     mm_batch_invariant,     "CUDA")
    _batch_invariant_LIB.impl("aten::addmm",  addmm_batch_invariant,  "CUDA")
    _batch_invariant_LIB.impl("aten::matmul", matmul_batch_invariant, "CUDA")
    _batch_invariant_LIB.impl("aten::linear", linear_batch_invariant, "CUDA")
# + aten::{_log_softmax,softmax,_softmax,mean.dim,bmm}; TF32 off; reduced-precision reduction off
```

The int4 Marlin GEMM is a **custom CUDA op**, not an aten op:
`apply_gptq_marlin_linear` → `output = ops.marlin_gemm(...)` with
`from vllm import _custom_ops as ops` (`vllm/model_executor/layers/quantization/utils/marlin_utils.py:9,510,~552`).
`ops.marlin_gemm` lives in the `_C` (`torch.ops._C`) namespace, **never `aten`**. The
`Library("aten", "IMPL")` overrides therefore cannot and do not intercept it.

Consequences:
- **(c) "forces a non-Marlin path" → NO.** Enabling the flag leaves Marlin byte-identical. The
  int4 4×-weight-bandwidth win is fully preserved. This was the single biggest fear in the PR
  brief; it is resolved from source.
- **(a) "provides a batch-invariant Marlin variant" → NO.** No invariant Marlin kernel is
  installed; the int4 body GEMMs run exactly as in PR #5.
- The residual question is whether the **un-overridden Marlin body GEMM** is *intrinsically*
  M-invariant. Marlin computes each output row with a fixed K-tiling reduction that does not
  depend on the batch row-count M, and does not use split-K atomic accumulation for our shapes by
  default (`MARLIN_USE_ATOMIC_ADD` is a separate, default-off knob already mapped as a dead-end
  parity/noise lever). So Marlin is *plausibly already M-invariant* — consistent with int4 flipping
  **less** (0.33%) than all-aten bf16 (0.72%): if the Marlin GEMM were the dominant batch-variant
  source, int4 would flip *more* than bf16, not less. The dominant batch-variance is in the shared
  aten ops (which the flag fixes), not in Marlin. **Net: covering the aten ops should remove most
  or all of the int4 flip; Step 3 measures the residual.**

## Q2 — Does coverage include the spec-decode VERIFY forward's GEMM?

**Yes for the decisive ops.** The verify forward is a batched pass over M=K+1 tokens; the
argmax that the strict gate compares is taken on the **lm_head logits**. Because `lm_head` is
**bf16 / tied** (in `ignore`, `tie_word_embeddings:True`), the lm_head matmul is `aten::{linear,
matmul,mm}` → **made batch-invariant** → for identical hidden states the M=1 and M=K+1 logits
become bit-identical, so the argmax matches. Feeding those logits:
- RMSNorm → `rms_norm_batch_invariant` path; attention → backend switches to `num_splits=1`
  (no split-KV reduction variance: `flash_attn.py:1194/1219`, `triton_attn` invariant config);
  softmax/`_log_softmax`/`mean` → invariant kernels; residual adds via aten → covered.
- The **only** un-covered link in the int4 body is the Marlin weight-GEMM (Q1) — plausibly already
  M-invariant. So Q2 coverage of the verify path is **as complete as it can be without an int4
  Marlin invariant kernel**, and the argmax-determining final GEMM is fully covered.

## Q3 — Version + exact enable mechanism

- **Version:** `vllm==0.22.0` (our existing pin) **already contains** the module at
  `vllm/model_executor/layers/batch_invariant.py` and the symbols
  `enable_batch_invariant_mode / init_batch_invariance / VLLM_BATCH_INVARIANT`. **No bump.**
- **Enable:** env var **`VLLM_BATCH_INVARIANT=1`** (`envs.py:86` default `False`,
  `envs.py:596` reads `bool(int(os.getenv("VLLM_BATCH_INVARIANT","0")))`).
- **Auto-wiring:** the GPU worker calls it during init —
  `vllm/v1/worker/gpu_worker.py:1128-1130`:
  `from vllm.model_executor.layers.batch_invariant import init_batch_invariance; init_batch_invariance()`.
  `init_batch_invariance()` (`batch_invariant.py:982`) runs only `if envs.VLLM_BATCH_INVARIANT:`,
  then `override_envs_for_invariance()` (NCCL/cuBLAS determinism, `VLLM_USE_AOT_COMPILE=0`) +
  `enable_batch_invariant_mode()` + sets `torch.backends.cuda.matmul.fp32_precision="ieee"` (TF32 off).
  → **Setting the env var is sufficient; no code change needed to activate it.**
- **Runtime confirmation (Step 2):** assert the env is honored — log that
  `vllm.envs.VLLM_BATCH_INVARIANT` is True in the worker, that the attn selector ran with
  `use_batch_invariant=True` (`selector.py:95`), and (belt-and-suspenders) that
  `torch.ops.aten.mm`/`linear` dispatch to our impls. Do NOT assume — the PR demands we *confirm
  mode is active*.

## Q4 — Compatibility with the rest of the stack (all unchanged: same 0.22.0)

- **(a) Gemma-4 multimodal + vision/audio towers:** identical engine to PR #5 → already loads all
  modalities. The towers' projections are bf16 (`ignore` list) and are not exercised in text decode.
- **(b) int4 W4A16 Marlin on sm_86:** identical engine → proven in PR #5 (~127 TPS floor / PPL ~2.02).
- **(c) spec decode w/ `gemma4_assistant` MTP draft:** identical `speculative_config` schema → proven
  in PR #5. Draft is bf16 → its GEMMs are covered too.
- **(d) `{8,4}` attn-group fix (vLLM PR #43543 / `dede691c9536`):** **NOT native in 0.22.0** — PR #5
  added it precisely because 0.22.0 predates it. Since we do not bump the engine, **KEEP the
  `sitecustomize.py` + `vllm_attn_group_patch.py` monkeypatch unchanged.** (The PR brief's
  instruction to "drop the monkeypatch if native" does not apply — there is no newer pin.)
- **(e) attn-backend invariance support:** the invariance selector raises if the chosen backend
  lacks support (`selector.py:153`). Our **TRITON_ATTN** returns `supports_batch_invariance()→True`
  (`triton_attn.py:305-307`); FlashAttn (`flash_attn.py:109-111`) and Flex also True. → no rejection.

## Q5 — Throughput cost of batch-invariance

Invariant mode trades speed for determinism: Triton persistent matmul replaces tuned cuBLASLt for
bf16 GEMMs; attention loses split-KV parallelism (`num_splits=1`); TF32 and reduced-precision
reductions are disabled. On our int4 stack the **dominant** decode cost is the int4 Marlin
weight-GEMM (~92% of step, BW-bound) which **stays on Marlin (unchanged)** — so the invariant
penalty applies only to the comparatively small bf16 ops (lm_head, norms, residual) + attention.
Magnitude is empirical (Step 4); external literature corroboration (Thinking Machines blog +
vLLM benchmarks) pending the research-agent pass — will append. Calibration from the PR: even
greedy-valid, the QAT-MTP drafter caps ~2.2 tok/step, so the realistic landing is ~250–270 TPS
*before* invariant overhead, possibly lower after — still expected to clear the ~127 int4 floor,
but the deliverable is (greedy yes/no) + (the measured TPS cost), not a frontier number.

---

## Evidence index (installed `0.22.0` source, this box)

- `vllm/model_executor/layers/batch_invariant.py` — module; `enable_batch_invariant_mode():905`,
  SM80 aten overrides `:915-921`, softmax/mean/bmm `:934-947`, TF32 off `:988-991`,
  `init_batch_invariance():982`, gated on `envs.VLLM_BATCH_INVARIANT`.
- `vllm/v1/worker/gpu_worker.py:1128-1130` — worker calls `init_batch_invariance()`.
- `vllm/envs.py:86,596` — `VLLM_BATCH_INVARIANT` default 0.
- `vllm/platforms/interface.py:355-367` — `is_device_capability_family(80)`: `cap//10==capability//10`;
  A10G `86//10==8==80//10` → True (SM80 aten-override branch fires).
- `vllm/model_executor/layers/quantization/utils/marlin_utils.py:9,510` — `from vllm import
  _custom_ops as ops`; `apply_gptq_marlin_linear` → `ops.marlin_gemm(...)` (`_C`, not aten).
- `vllm/v1/attention/backends/triton_attn.py:305-307` / `flash_attn.py:109-111` —
  `supports_batch_invariance()→True`; `selector.py:95,153` — invariance-aware selection / reject.
- `vllm/v1/attention/backends/flash_attn.py:1194,1219` — `num_splits=1 if VLLM_BATCH_INVARIANT`.
- Target `config.json`: `quant_method=compressed-tensors`, `format=pack-quantized`,
  `ignore=[... ,'lm_head']`, `tie_word_embeddings=True`, `architectures=['Gemma4ForConditionalGeneration']`.

## Step-2/3 test plan (consequences of this audit)

1. New submission `submissions/int4_mtp_batchinv/` = copy of `int4_mtp_drafter`, with
   `manifest.json` env adding `VLLM_BATCH_INVARIANT: "1"` (and keeping vLLM **0.22.0**,
   the `{8,4}` monkeypatch, K=6, all the existing serve flags). **No dependency change.**
2. Smoke (GPU, local): load int4 target + MTP draft with the flag on; **confirm** invariant mode
   active (env honored + attn selector `use_batch_invariant=True`); confirm all modalities load.
3. Decisive greedy (Step 3): reuse PR #5 `run_arm.sh` harness; for each arm run M=1 AR ref vs
   M=K+1 spec candidate through the official `check_greedy_identity.py` + `flip_rate.py`:
   - **int4 OFF** (control; expect ~0.33%/tok, DIVERGENT) vs **int4 ON** (target: flip→0, IDENTICAL).
   - **bf16 ON** as the positive control (all-aten → expect IDENTICAL) if int4 ON is ambiguous.
   - If int4 ON GREEDY_IDENTICAL on 32 prompts → confirm on 128.

---

## Literature corroboration (research-agent pass, 2026-06-13)

External web/literature pass to ground the source audit. Headline: the underlying
batch-invariant kernel mechanism is real and our Marlin reasoning is corroborated, **but
vLLM does not officially support batch-invariance for speculative decoding** — so the
empirical Step 3 outcome is genuinely open.

1. **Mechanism (Thinking Machines, "Defeating Nondeterminism in LLM Inference," Horace He et
   al., ~Sep 2025, thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference).** Three
   batch-variance sources in stock kernels: (a) **split-K** GEMM — #splits scales with M, so the
   float reduction *order* changes with batch size (the primary M=1 vs M=K+1 flip source); (b)
   **tensor-core instruction switching** at small M (can't fill 2D tiles → narrower warp op); (c)
   **attention split scheduling** (FlashInfer picks split-count to saturate cores → M-dependent).
   The fix enforces "the reduction order for each element must be fixed regardless of the
   batch-size of the kernel" — *exactly* the M=1 vs M=K+1 equivalence we need, and *exactly* what
   `confirm_invariant.py` measured (bit-identical rows with the flag ON). Our root-cause story is
   corroborated.

2. **⚠️ vLLM does NOT integrate batch-invariance with speculative decoding (the key caveat).**
   vLLM docs (docs.vllm.ai/en/latest/features/batch_invariance/) state verbatim: *"The
   implementation does not currently integrate with speculative decoding–based LLM inference."*
   vLLM issue **#27433** lists spec-decode support under *"Nice to have"* with *"this might be
   hard."* Supported use cases: cross-request determinism, RL training stability. **Implication:**
   the flag is *not designed/validated* to bit-match the spec verify forward; our covered aten
   ops (lm_head/RMSNorm/softmax/attention `num_splits=1`) may be *necessary but not sufficient* if
   the verify path has an uncovered batch-variant op (attention metadata build, the rejection
   sampler's logits compare, or a fused/non-aten verify step). This is precisely the
   "ON → still DIVERGENT" branch the PR anticipates, and it now has *official documentation
   weight* behind it. Step 3 decides.

3. **Throughput cost (Step 4 calibration).** TM blog: unoptimized batch-invariant ≈ **2.1× slower**
   (26s→55s/batch); with fixed-split-size attention ≈ **1.6×** (42s). Cost concentrates in (a)
   attention losing split-KV parallelism and (b) matmuls swapping cuBLAS→Triton persistent kernel.
   vLLM #27433 logs *partial* recovery via BMM opt (+18.1% TPS), fused RMSNorm (+2.1% E2E), Cutlass
   FP8 (+28.9% E2E) — but no full parity. On our int4 stack the dominant weight-GEMM stays on
   **Marlin (untouched)**, so the penalty is bounded to the bf16 ops + attention (still real —
   measure in Step 4).

4. **Marlin / int4 untouched (corroborated).** No public statement directly on
   `VLLM_BATCH_INVARIANT` × Marlin, but the architecture confirms our audit: batch-invariance
   overrides **aten-dispatch** ops; Marlin kernels are `torch.ops._C.*` custom extension ops that
   **bypass aten entirely**, so an aten-level override *cannot* intercept them. → enabling the flag
   leaves Marlin byte-identical (int4 bandwidth win safe), and the int4 body weight-GEMM is *not*
   made invariant by the flag (consistent with int4 flipping less than all-aten bf16 in PR #5).
   Requires compute capability ≥ 8.0 (A10G sm_86 ✓).

5. **Alternative if ON is insufficient (follow-up).** arxiv **2601.17768** ("LLM-42 / Enabling
   Determinism in LLM Inference with Verified Speculation") proposes a **verify-rollback** loop:
   decode on the fast (non-deterministic) path, re-verify under fixed-shape reduction schedules,
   commit consistent tokens, roll back violators. Designed *specifically* for the spec-decode
   determinism setting; overhead scales only with traffic needing determinism. If batch-invariance
   proves insufficient for our verify forward, this is the natural next lane (and reframes the
   whole drafter ladder around a rollback gate rather than system-wide invariant kernels).

---

## Step 3 — EMPIRICAL RESULT (measured on A10G, 2026-06-13)

**Headline: batch-invariance is COMPATIBLE but INSUFFICIENT — int4+MTP spec decode stays
greedy-DIVERGENT with `VLLM_BATCH_INVARIANT=1`.** The "ON → still DIVERGENT" branch fired,
corroborating the vLLM #27433 disclaimer with a measured flip rate.

### Mode-active confirmation (Step 2 requirement — confirmed, not assumed)
- Kernel probe `confirm_invariant.py` on A10G: **ON** → `INVARIANT_ACTIVE_AND_FUNCTIONAL`, M=1 and
  M=7 rows both bit-identical to the M=8 batched GEMM (max|diff|=0); **OFF** control →
  `OFF_CONTROL_NONZERO`, M=1 vs M=8 max|diff|=1.0. The aten GEMM override works on this box.
- Server-side: the int4 ON EngineCore worker log shows the override **installed in the worker
  process** — `Overriding ... operator: aten::mm ... new kernel: registered at
  batch_invariant.py:913`. So invariance was genuinely active during the decisive decode, not just
  in the standalone probe.
- Stack confirmed under ON: `MarlinLinearKernel for CompressedTensorsWNA16` (int4 Marlin preserved),
  `AttentionBackendEnum.TRITON_ATTN` (invariance-supported backend), `Resolved architecture:
  Gemma4ForConditionalGeneration` + `profiled with 1 video items` (all modalities loaded), spec
  init OK with the {8,4} monkeypatch.

### Greedy verdict — int4 target, MTP K=6, 32 prompts × 512 tok, seed 1, ignore_eos, eager, strict official verifier

| arm | INV | verdict | identical/32 | flip_rate /tok | 95% CI |
|---|---|---|---|---|---|
| int4 ON  | 1 | **DIVERGENT** | 5/32 | **0.376%** | [0.234, 0.518]% |
| int4 OFF | 0 | **DIVERGENT** (same-session control) | 6/32 | **0.332%** | [0.205, 0.460]% |

The int4 ON flip rate (0.376%/tok) is **statistically indistinguishable from the same-session int4
OFF control (0.332%/tok, CIs fully overlap)** — and from PR #5's int4 OFF (0.33%/tok). Enabling
batch-invariance produced **no measurable reduction** in the spec-decode greedy flip (ON is, if
anything, fractionally *higher*). The covered aten ops (lm_head / RMSNorm / softmax / attention `num_splits=1`) are
**necessary but not sufficient**: the residual batch-variance that flips the M=K+1 verify argmax vs
the M=1 AR argmax lives in an **un-covered** path.

### Two candidate root causes — the bf16 positive control discriminates them
- **(a) int4 Marlin weight-GEMM.** Marlin is a `torch.ops._C` custom op that bypasses the aten
  override (Q1), so the int4 body weight-GEMM is *not* made invariant. If its M=K+1-vs-M=1 reduction
  order differs, it alone would flip argmax even with every aten op covered.
- **(b) spec verify machinery.** Attention-metadata build, the rejection sampler's logits compare,
  or a fused/non-aten verify step — none aten — would defeat invariance regardless of weight
  precision (this is precisely what vLLM #27433 means by "does not integrate with spec decode").

**Discriminator:** bf16 target is **all-aten, no Marlin**. bf16 ON → GREEDY_IDENTICAL ⇒ cause (a)
Marlin; bf16 ON → still DIVERGENT ⇒ cause (b) the spec path itself.

### bf16 ON positive control — RESULT (measured on A10G, 2026-06-13)

bf16 target (`google/gemma-4-E4B-it`, unquantized, **no Marlin** — server log confirms
`quantization=None`, pure aten path), MTP K=6, INV=1, same 32 prompts × 512 tok / seed 1 /
ignore_eos / eager / strict official verifier.

| arm | INV | target GEMM | verdict | identical/32 | flip /tok | 95% CI |
|---|---|---|---|---|---|---|
| int4 ON  | 1 | Marlin `_C` (un-covered) | DIVERGENT | 5/32  | **0.376%** | [0.234, 0.518]% |
| int4 OFF | 0 | Marlin `_C` (un-covered) | DIVERGENT | 6/32  | **0.332%** | [0.205, 0.460]% |
| **bf16 ON** | 1 | aten linear (**covered**) | **DIVERGENT** | **16/32** | **0.111%** | [0.057, 0.166]% |
| bf16 OFF (PR #5 ref) | 0 | aten linear | DIVERGENT | — | 0.72% | — |

**The control is DIVERGENT, not identical — so the answer is NOT a clean "cause (a) only."
The data instead pins BOTH causes, additively:**

1. **Cause (a) — int4 Marlin GEMM IS batch-variant (confirmed; refutes the Step-1 prior).**
   The only stack delta between **int4 ON** and **bf16 ON** (both INV=1) is the target weight GEMM:
   int4 Marlin (`_C`, un-covered) vs bf16 linear (aten, covered). int4 ON flips at 0.376% but bf16 ON
   at only 0.111% — a **~0.265%/tok excess that is attributable to the un-covered Marlin GEMM** being
   M-variant (M=K+1 verify vs M=1 AR reduction order differs). This **refutes** the Step-1/Q1
   hypothesis that Marlin was "plausibly already M-invariant": if it were, int4 ON would have dropped
   to the bf16 floor. It did not. Equivalently, **int4 ON ≈ int4 OFF (0.376 vs 0.332, CIs overlap)**:
   covering the aten ops does *nothing* for int4 because int4's dominant flip source (Marlin) is the
   one op the aten override cannot reach.
2. **Cause (b) — the spec verify path has an un-covered batch-variant component (confirmed).**
   bf16 ON has **zero Marlin and every aten op covered**, yet still flips at **0.111%/tok**
   (16/32 divergent). That irreducible residual cannot be a weight-GEMM effect — it must live in a
   **non-aten part of the spec verify forward** (attention-metadata build, the rejection sampler's
   logits compare, or a fused verify step). This is the measured, quantified version of the vLLM
   **#27433** disclaimer ("does not currently integrate with speculative decoding"). Cross-PR
   corroboration: bf16 OFF was 0.72%/tok (PR #5); ON cuts it **6.5×** to 0.111% — aten coverage
   removes the *bulk* of bf16's batch-variance but **cannot close the spec-path residual**.

**Consistency check (first-order, CIs wide):** int4 ON ≈ Marlin contribution + spec-path residual
≈ 0.265% + 0.111% = 0.376% — the decomposition adds up to the observed int4 ON flip rate. The two
sources are independent and additive, and *both* are outside the aten override's reach.

**Net for the program:** `VLLM_BATCH_INVARIANT` rescues greedy-valid spec decode at **neither**
precision. The int4 target fails on cause (a)+(b); even a hypothetical **bf16-target** drafter
ladder fails on cause (b) alone (0.111%/tok, still DIVERGENT). The "pin batch-invariant vLLM" lane
is a **definitive negative for greedy-valid spec decode at any precision** in vLLM 0.22.0.

### TPS cost (eager, single-stream decode-capture wall-clock, INV=1; diagnostic, not leaderboard)
- int4 AR (M=1, spec off): 16384 tok / 799.2 s = **20.5 tok/s** (~6× below the ~127 int4 floor).
- int4 spec (K=6): 16384 tok / 403.8 s = **40.6 tok/s** (spec ~2.0× over AR, ~2.2 tok/step MTP).
- Even with spec, invariant **eager** throughput (~40 tok/s) is **~3× below** the greedy-valid int4
  floor. Per the PR ("a TPS number for a greedy-INVALID stack is worthless") the full cudagraph
  Step-4 characterization is **moot** — greedy gating already failed. The eager numbers are reported
  only to show the cost direction; cudagraph-compat of invariant kernels was not tested.

### Verdict (terminal): **BLOCKED / definitive negative**
Batch-invariant kernels exist in our exact validated engine (vLLM 0.22.0), are compatible with the
full Gemma-4 + int4-Marlin + spec stack, and were confirmed active — yet they do **not** make
int4+MTP spec decode greedy-identical (0.376%/tok, no change vs OFF 0.332%). The bf16 positive
control resolves *why*, pinning **two independent un-covered sources**: (a) the int4 **Marlin**
weight-GEMM is batch-variant and bypasses the aten override (int4 ON ≫ bf16 ON at equal INV=1);
and (b) the **spec verify path** carries a non-aten batch-variant component (bf16 ON, zero Marlin +
full aten coverage, **still** DIVERGENT at 0.111%/tok — the measured vLLM-#27433 disclaimer). The
"pin batch-invariant vLLM" rescue therefore **does not unlock the drafter ladder at the int4 target
*or* at a bf16 target** — cause (b) blocks bf16 even after cause (a) is removed. Next lane for
greedy-valid spec decode: the **verify-rollback** approach (arxiv 2601.17768), which gates the spec
path with a fixed-shape re-verification rather than relying on system-wide invariant kernels;
batch-invariance alone is insufficient for the spec verify forward at any precision in 0.22.0.
