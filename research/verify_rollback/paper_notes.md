# Verify-rollback gate — paper notes, hook-point analysis, and the cost theorem

PR #24. Builds on PR #5 (spec is greedy-INVALID in vLLM 0.22.0) and PR #19
(invariant-kernel lane is a definitive negative; flip = 0.376%/tok int4 ON,
decomposed into Marlin `_C` 0.265% + spec-path residual 0.111%).

**Scope: LOCAL ONLY. No HF Job.** Mechanism validation.

---

## 0. TL;DR (read this first)

Two distinct things are called "verify-rollback" in PR #24, and they are not the
same mechanism:

1. **What arxiv 2601.17768 actually does** (the *real* paper, "LLM-42"): re-verify
   accepted tokens under a **fixed-shape batched** forward (8 requests × 32 tokens
   → constant 256-token pass). This makes output **reproducible across runs given a
   fixed serving schedule** — *batch-self-consistency* — **NOT** identity to a
   sequential M=1 AR forward. The paper never claims M=1-greedy-identity. **It
   solves a strictly weaker property than this challenge's gate requires.**

2. **What the PR-body protocol specifies** (the advisor's construction): re-verify
   each accepted token under a **fixed-shape M=1 sequential AR** forward and roll
   back on disagreement. This *does* restore M=1-greedy-identity (flip → 0 by
   construction) — but it is **structurally net-negative TPS**: detecting a flip
   requires computing the M=1 reference for **every** token, and computing the M=1
   reference for every token *is* AR decode. So this variant = AR decode + the
   (now-discarded) speculative work. It cannot beat the AR baseline.

**Either way the hypothesis "restore greedy-valid spec decode AND maintain
net-positive TPS over int4 AR" fails:** variant (1) does not restore the *required*
identity; variant (2) restores it but is slower than AR. The drafter ladder stays
blocked through the shared rejection-sampler verify path, exactly as PR #19's
follow-up #1 anticipated — but the precise reason is sharper than "try it and see":
it is a **counting argument**, below.

---

## 1. The real paper: arxiv 2601.17768

**"Enabling Determinism in LLM Inference with Verified Speculation"** — Gond,
Kamath, Ramjee, Panwar (Microsoft Research / IISc / UW), submitted 2026-01-25,
revised 2026-01-30. Code: `github.com/microsoft/llm-42`.

### What it actually does
- Groups in-flight requests into **windows of 8**, pads each request to **32
  tokens**, and runs a **constant-shape 256-token forward** to re-verify
  speculated tokens.
- The fixed shape forces a **consistent parallel-reduction order**, so the same
  input produces the **same output across runs / restarts / batch-composition
  changes**.
- **No margin/confidence gate. No per-token M=1 check. No "2.2% overhead" figure.**
- Reported recompute overhead: **0.32%–10.97%** (workload-dependent; low on
  synthetic, high on ArXiv text). End-to-end TPS: ~6% slower than nondeterministic
  at 100% deterministic traffic, within 2% at 10%, and **33% faster than
  SGLang-Deterministic** (the batch-invariant-kernel baseline — i.e. the
  Thinking-Machines lane PR #19 closed for us).

### Why this does not satisfy our gate
The challenge contract (`program.md`): *"Greedy decode must remain token-identical
to plain greedy autoregressive decode."* The reference is a **sequential M=1 AR**
forward. LLM-42's verifier produces a token that is **reproducible given its own
fixed 8×32 schedule**, but that token is **not** the token an M=1 AR forward
produces on the same prefix — the 256-wide batched reduction is a *different*
computation than a width-1 reduction (this is the very batch-variance PR #19
measured). LLM-42 gives "determinism" = *self-consistency across runs*; we need
"determinism" = *identity to M=1 AR*. These differ by exactly the 0.376%/tok we
already measured. **The paper's mechanism, applied verbatim, would still be
greedy-DIVERGENT against our reference.**

This is not my inference — it is the paper's stated design (confirmed via the
alphaxiv overview of 2601.17768):
- **Observation O3 (verbatim intent):** *"Determinism requires only
  position-consistent reductions. It is sufficient for a particular token position
  to use the same reduction schedule **across runs of a given request**."* The
  paper **explicitly relaxes** the constraint from batch-invariant (He et al.) to
  *position-consistent-across-runs*. That is run-to-run reproducibility, not
  M=1-identity.
- **The verifier reference is a wide fixed-shape window** (32–64 tokens, grouped
  across 8 requests), chosen because the *only* divergence is FP rounding (same
  model for decode and verify, no approximate drafter). A 32–64-wide reduction is
  **not** a width-1 reduction; its argmax can differ from M=1 AR by the same FP
  drift we measure. LLM-42 makes the *fast path* agree with *this wide verifier*,
  not with M=1.
- **Hidden assumption that fails on our stack — Observation O2:** *"Most GPU
  kernels use uniform, shape-consistent reductions."* LLM-42 assumes the verifier's
  fixed shape ⇒ a fixed reduction schedule. PR #19 measured the int4 **Marlin
  `_C`** W4A16 GEMM to be **batch-variant** (the 0.265%/tok Marlin excess). So even
  LLM-42's *own weaker* guarantee (run-to-run reproducibility) is not free on our
  int4 target without verifying that Marlin is position-consistent at the chosen
  window shape — and it would still miss M=1-identity regardless.

This is the single most important correction to the PR premise: **2601.17768 does
not provide a cheap route to M=1-greedy-identity, because it never targets
M=1-greedy-identity.**

---

## 2. The advisor's per-token protocol and the cost theorem

The PR-body Step-2 protocol (per-token M=1 re-verify) is a *different* mechanism
from the paper. Analyze it on its own terms.

### 2.1 It does restore greedy identity (flip → 0), trivially
The committed output of per-token verify-rollback is, position by position, the
**M=1 AR argmax** (either the spec token already equalled it → commit, or it didn't
→ roll back to it). So the verify-rollback output stream **is** the M=1 AR greedy
stream, bit-for-bit. Verified against the M=1 AR reference → `GREEDY_IDENTICAL`,
flip_rate = 0. There is nothing to "discover" here; it is definitional.

### 2.2 The cost theorem (why net-positive TPS is impossible)
> **Claim.** In a stack where the batched M=K+1 verify forward is not bit-identical
> to M=1 AR (our stack: Marlin int4 `_C` GEMM + spec-path residual, PR #19), any
> verify-rollback that **guarantees** M=1-greedy-identity must run **one M=1 AR
> forward per committed token**, and therefore costs **≥ plain M=1 AR decode**,
> plus the speculative work it now discards. Net TPS over AR is strictly negative.

**Proof.** To *guarantee* a committed token equals the M=1 AR argmax, you must know
the M=1 AR argmax at that position. The only way to know it (absent a batch-invariant
kernel that makes M=K+1 ≡ M=1 — ruled out by PR #19) is to **compute** it: an M=1
forward whose context is the prefix + the committed tokens so far. For position `k`
that forward extends the context by one token and emits one argmax — which is
**exactly one step of autoregressive greedy decode**. Re-verifying the `j` tokens a
spec step accepts therefore runs `j` sequential M=1 forwards — **identical to the
`j` forwards AR would run to emit those `j` tokens**. So per output token:

```
t_VR  =  t_M1_reverify  +  t_spec_propose_and_verify
      =  t_AR           +  (drafter + M=K+1 verify, amortized)        [t_spec > 0]
      >  t_AR
```

Hence `TPS_VR = 1 / (1/TPS_AR + 1/TPS_spec)  <  TPS_AR`, always. ∎

### 2.3 The flaw in the PR's overhead estimate
The PR estimates overhead as "≈ 2.2% of steps × one extra M=1 forward ≈ 0.15 ms".
That assumes the M=1 forward is needed **only on the 2.2% of steps that roll back**.
But **you cannot know which 2.2% roll back without computing the M=1 reference for
all of them.** Flip detection is not free; it *is* the M=1 forward. The "extra M=1
only on rollback" model under-counts the work by ~45× (every token needs it, not
2.2% of steps). The 2.2% figure is the **rollback rate**, not the **re-verify
rate** — the re-verify rate is 100% of tokens.

### 2.4 Could you batch the re-verify to recover speed?
Yes — run all K re-verify positions in one M=K forward. But that M=K forward is
**not** bit-identical to M=1 (same batch-variance), so it reintroduces the
0.376%/tok flips and the gate fails again. This is precisely the dilemma:
**per-token M=1 re-verify → identity ✓, speed ✗; batched M=K re-verify → speed ✓,
identity ✗.** No third option exists in a non-batch-invariant stack (confirmed by
literature review: Thinking-Machines batch-invariant kernels cost 34%+ and don't
cover Marlin; LLM-42 sidesteps by redefining determinism; "Batch Speculative
Decoding Done Right" arxiv 2510.22876 reaches ~95% match, not bit-identity; no
method cheaply detects M=1-vs-M=K+1 flips without running one of the two forwards).

---

## 3. vLLM 0.22.0 hook-point analysis

**Architecture correction:** the PR body references `vllm/v1/spec_decode/
spec_decode_worker.py` (a v0 path). vLLM 0.22.0 is **v1**: there is no
`spec_decode_worker.py`. The greedy verify + accept happens in:

- `vllm/v1/sample/rejection_sampler.py` — `RejectionSampler.forward` →
  `rejection_sample` → **`rejection_greedy_sample_kernel`** (Triton). In the greedy
  branch it sets `target_argmax = target_logits.argmax(-1)` (line ~452) and, per
  draft position, commits `token_id = target_argmax_id` and stops at the first
  `draft_token_id != target_argmax_id` (line ~744). **Every committed token is the
  argmax of the M=K+1 batched verify forward** — this is the divergence source.
- `vllm/v1/worker/gpu_model_runner.py` — `_sample` (line ~3481) calls
  `self.rejection_sampler(spec_decode_metadata, draft_probs, logits, ...)` (line
  ~3504). The proposer (`self.drafter`, Gemma4MTPModel) produces draft tokens
  upstream (`_calc_spec_decode_metadata`, line ~2713).

**Where honest verify-rollback would have to live:** wrap `_sample` /
`rejection_sampler.forward` so that, after the M=K+1 commit, it runs a per-token
M=1 AR forward (re-entering the model with M=1 attention metadata for each accepted
position) and overrides the commit on disagreement. This is a **deep**
`gpu_model_runner` patch (new attention-metadata build + extra forwards inside
`execute_model`), not a one-line monkeypatch. Per §2.2 it is also unnecessary to
*prove* the result: the per-token M=1 re-verify forward is, by construction,
bit-identical to a step of the spec-OFF M=1 AR path we already run and time. So we
realize verify-rollback faithfully by composition:

- **Re-verify / committed output = the spec-OFF M=1 AR arm** (this *is* what VR
  commits, and its wall-clock *is* the dominant VR cost).
- **Discarded speculative work = the spec-ON arm** (its per-token time is the
  amortized propose + M=K+1 verify cost VR adds on top).
- **VR output ≡ M=1 AR reference** → flip = 0 (official verifier).
- **`TPS_VR = 1/(1/TPS_AR + 1/TPS_spec)`** from the two measured arms.

A light instrumentation patch on `rejection_greedy_sample_kernel`'s host wrapper
records, per spec step, the committed `target_argmax` tokens, the draft tokens, and
the accept count — enough to *observe* rollback decisions on real model outputs
(`verify_rollback_patch.py`).

### cudagraph note
Honest VR is cudagraph-compatible: it always runs (a) the drafter, (b) the M=K+1
verify, (c) K× M=1 re-verify — all **fixed shapes**; commit-vs-rollback is a CPU-side
selection, not a shape change. So cudagraph does *not* rescue it: VR-cudagraph TPS
= `1/(1/TPS_AR_cudagraph + 1/TPS_spec_cudagraph)` < `TPS_AR_cudagraph` = 126.378
(PR #4). The variable-control-flow worry in the PR only arises if you try to *skip*
the re-verify on clean steps — which §2.3 shows you cannot do without breaking the
guarantee.

---

## 4. Predictions (to be confirmed empirically)

| metric | prediction | basis |
|---|---|---|
| flip_rate, int4_VR (per-token M=1 re-verify) | **0.0** (`GREEDY_IDENTICAL`) | §2.1, definitional |
| rollback_rate / spec step (K=6) | **≈ 2.2%** = 1−(1−0.00376)^6 | PR #19 p=0.376%/tok |
| TPS_VR eager | **≈ 13.6** = 1/(1/20.5 + 1/40.6) | §2.2 + PR #19 eager |
| TPS_VR cudagraph | **< 126.378** (≈ 70–80) | §2.2 + PR #4 AR floor |
| verdict | greedy-valid ✓ **but TPS net-negative vs AR** ✗ | cost theorem |

## 5. Plan

1. `verify_rollback_patch.py` — rejection-sampler instrumentation (per-step spec
   `target_argmax` + draft + accept count) **and** the verify-rollback decision/
   reconstruction logic (apply M=1 rollback, count rollbacks, confirm output ≡ M=1
   AR reference).
2. `run_vr_arm.py` — driver: run spec-OFF (M=1 AR = VR committed output / re-verify
   cost) and spec-ON (discarded spec work) arms on int4; official verifier →
   flip_rate; rollback_rate (derived from p + directly observed); measure TPS_AR /
   TPS_spec (eager + a cudagraph TPS pass); derive TPS_VR; log W&B group
   `verify-rollback-gate`.
3. Report flip_rate, rollback_rate, TPS, and the §1 paper-premise correction.

## 6. Public evidence used
- arxiv **2601.17768** ("LLM-42") — the cited paper; its mechanism gives
  batch-self-consistency, not M=1-greedy-identity (premise correction, §1).
- Thinking-Machines "Defeating Nondeterminism in LLM Inference" + SGLang-Deterministic
  (batch-invariant kernels, 34%+ overhead, no Marlin coverage) — the lane PR #19 closed.
- arxiv **2510.22876** "Batch Speculative Decoding Done Right" (~95% match, not bit-identity).
- vLLM issue **#27433** (batch-invariance does not integrate with spec decode) — PR #19.
- My PR #5 (spec greedy-INVALID at every precision) and PR #19 (invariant-kernel
  lane negative; 0.376%/tok int4, Marlin+spec-residual decomposition).
