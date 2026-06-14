# Tree-path PPL-margin bound: does M=32 batched-verify stay under PPL ≤ 2.42? (PR #166)

**Verdict: SAFE (GREEN).** The 0.0433 PPL margin survives M=32 int4-Marlin
batch-variance. Two independent arguments clear it; only an unphysical
adversarial-on-every-token model breaches, and that model is falsified by the 0
measured argmax flips at M=32.

- **PRIMARY** `ppl_margin_bound_self_test_passes = True`
- **TEST** `tree_path_ppl_worst_case = 2.4134` (binding every-token extreme) ≤ 2.42
- W&B run `z4l8ljd7` (group `tree-path-ppl-margin-bound`). Evidence:
  `research/validity/tree_path_ppl_margin/runs/20260614T142818Z/ppl_margin_bound_result.json`.
  Pure-analytic, CPU-only (peak 12 MiB, no GPU / vLLM / model load). BASELINE
  untouched: 481.53 TPS. Adds 0 TPS — a validity leg, the aggregate-PPL complement
  to #158's per-token leg.

---

## Why this bound

The launch's validity rests on three scorer gates: TPS, **PPL ≤ 2.42**, and 128/128
completions (`MAX_CONCURRENCY=1`). The measured frontier PPL is **2.37667** — a
margin of only **0.0433**, measured on the **LINEAR (M=1)** stack. The **TREE path**
(land #71: M=32 batched verify, K=7 + 3D split-KV) runs int4-Marlin GEMMs at batch
width 32, which carries documented **batch-variance** — the same FP-reduction-order
source as the **0.6169** spec-vs-AR completion divergence #158 characterised (118/128
prompts, Issue #124, ruled NOT a contract violation).

denken #150 (tree-submission preflight) and denken #158 (per-token greedy-exactness)
both **assume** the tree-path aggregate PPL stays ≤ 2.42 but neither **bounds** it.
#158 proved per-token **argmax** fidelity (the kernel commits the in-step target
argmax, rate 1.0); it did **not** bound how much M=32 batch-variance can inflate the
**aggregate softmax mass** that PPL integrates over. Greedy-exactness constrains the
top-1; PPL = exp(mean NLL) integrates the probability the model assigns the *target*
token, which is **not** the argmax wherever PPL > 1. This is the last unbounded
validity dimension before launch.

This is a **synthesis**, not a new measurement: it imports committed outputs and
propagates them analytically.

---

## The two anchors (one variance source, two committed measurements)

int4-Marlin batch-variance is **one** source; two committed advisor-branch artifacts
measure its two relevant axes:

| Axis | Anchor (committed) | Value |
|---|---|---|
| **Magnitude** | `verify_argmax_margin/20260614T041541Z/summary.json` (kanna #87) | real Marlin kernel, **M=32 vs M=8: max\|Δlogit\| = 0.25, 0 argmax flips** over 65,536 positions; M=16 **bit-identical** (Δ=0) |
| **Frequency** | `descent_greedy_exact_harness/runs/20260614T135333Z/...json` (denken #158) | M=1-AR-vs-batched completion token divergence **40426/65536 = 0.6169** (118/128 prompts) |

Central PPL anchor: #158's committed linear-stack cross-check **2.376664808823738**,
cross-checked against the int4 split-KV verify *prefill* PPL (2.3766775) and the
noise-floor PPL (2.3766828) — spread **1.8e-5**, all = 2.37667, `2.42 − 2.37667 =
0.0433`. Scored span **N = 61,797** tokens (canonical `ppl_summary.json`).

> This is the *same* variance source #158 names — not a new one. #158 measures its
> **frequency** (compounded completion divergence); verify_argmax_margin measures its
> **magnitude** (the per-logit Δ on the real kernel). The bound consumes both.

---

## 1. Structural finding (the model-free headline)

**The scorer's PPL never traverses the M=32 verify GEMM.** PPL is **teacher-forced
prefill** via `prompt_logprobs` (`max_tokens:1`) — see `same_path_ppl.md` §2(c)/§3
and the three request shapes there. Speculative tree decode (the M=32 verify batch)
runs **only in the decode phase**; a `prompt_logprobs` request scores the prompt in a
single prefill forward pass and never enters the speculative path. The prefill that
computes the scored logprobs is therefore **M-invariant**:

> scored tree-path PPL ≡ scored linear-path PPL = **2.37667**, and the M=32
> batch-variance contributes **zero** to the gated quantity. The 0.0433 margin is
> **not consumed by M=32.**

This is the strongest statement because it is independent of any perturbation model.
It is the PPL-side analogue of the audit-vs-timed reasoning in `same_path_ppl.md`:
the *speculation* changes speed, not the *teacher-forced logits*.

**Scope (honest).** This covers the **M=32 verify-batch** dimension #158
characterised. If land #71's tree submission *also* changes the **prefill chunk
geometry** (a different batch-variance leg), that prefill change *would* touch the
scored PPL and must be audited separately — the same caveat #158 raised for upstream
logits. The bound below conservatively transplants the *decode* M=32 jitter onto
prefill anyway, so it covers that case too.

---

## 2. Conservative transplant bound (the corroborating leg)

Grant the worst-case counterfactual that the *decode* M=32 logit jitter **did** land
on every *prefill*-scored token. Propagate logit perturbation → NLL perturbation →
aggregate PPL. With `NLL_i = −log softmax(z_i)[t_i]` and per-coordinate perturbation
‖δ‖∞ ≤ ε:

- gradient `g_i = −(1−p_t)·e_t + p_{k≠t}`, `‖g_i‖₁ = 2(1−p_t) ≤ 2`;
- Hessian `H_i = diag(p_i) − p_ip_iᵀ ⪰ 0` (softmax/logsumexp Jacobian, PSD),
  `tr(H_i) = 1 − ‖p_i‖² ≤ 1`.

**(1) Expected** — batch-variance is mean-zero FP-reduction noise (`E[δ]=0`). The
first-order term vanishes in expectation; the PSD Hessian gives a one-signed
second-order bias: `E[ΔNLL_i] = ½ tr(H_iΣ_δ) ≤ ½σ²(1−‖p_i‖²) ≤ ½σ²`. **Symmetric
logit noise can only *increase* PPL, by O(σ²).** (Boyd & Vandenberghe §3.1; Gao &
Pavel, arXiv:1704.00805, softmax/logsumexp properties.)

**(2) Worst-case** — bias + a 6σ aggregate fluctuation (`Var(ΔNLL_i) ≤ 2σ²`,
aggregate-mean std `≤ √(f·2σ²/N)`).

**(3) Adversarial Lipschitz** (unphysical) — every perturbed token pushed
worst-direction by the full ε: `mean ΔNLL ≤ f·2ε·mean(1−p_t)`, `mean(1−p_t) ≤ 1−1/PPL`
(Jensen). Requires the rounding noise to conspire against the target token on every
token — falsified by the **0 measured argmax flips** and the symmetric ±1-bf16-ULP
final-cast regime (`verify_argmax_margin.md` "Numerics regime").

σ is taken conservatively from the measured **max** (`σ² = ε²/3`, treating ε=0.25 as a
per-logit std — an over-count, since M=16 is bit-identical and the median Δlogit ≈ 0).
f is swept over three regimes.

### Results (ε = 0.25, N = 61,797, cap = 2.42)

| frequency model | f | expected PPL | **worst-case PPL** | margin |
|---|---|---|---|---|
| per-step flip rate (physical for PPL) | 0.0050 | 2.37679 | **2.37761** | +0.0424 |
| #158 completion divergence | 0.6169 | 2.39199 | **2.40126** | +0.0187 |
| every token (binding extreme) | 1.0000 | 2.40155 | **2.41341** | +0.0066 |
| adversarial Lipschitz (unphysical) | 1.0000 | — | 3.1750 | **−0.755 BREACH** |

- The **per-step flip rate ≈ 0.5%** is the physically-correct perturbed fraction for
  *teacher-forced* PPL (no compounding): from `P(identical 512-token completion) =
  10/128 = (1−q)^512 ⇒ q ≈ 0.005`. The 0.6169 completion-divergence rate is the
  *compounded* footprint of that 0.5% per-step rate, so using it is conservative.
- **Even the every-token extreme** (every scored token gets the measured M=32 jitter
  as a mean-zero perturbation) gives worst-case **2.4134 ≤ 2.42**.
- Only the **adversarial** model breaches — and it is unphysical (0 flips; mean-zero
  symmetric rounding).

### ε sensitivity — how much headroom?

**Break-even ε** (where the worst-case first touches 2.42): **0.351** at the #158
frequency, **0.275** at the every-token extreme. The measured M=32 perturbation is
**0.25**. So the per-logit M=32 jitter would have to grow ~40% (#158 f) or ~10%
(every-token f) — from 2 bf16-ULP to ~3 ULP — before the *conservative transplant*
worst-case breaches. The structural finding (§1) makes both moot.

---

## 3. Self-test (PRIMARY)

`ppl_margin_bound_self_test_passes = True` requires all three:

| condition | check | result |
|---|---|---|
| (a) central reproduces | PPL at ε→0 equals measured 2.37667 (abs err = 0.0 ≤ 1e-9) | ✅ |
| (b) conservative ordering | central ≤ expected ≤ worst-case ≤ adversarial | ✅ |
| (c) `ppl_margin_under_2p42` | **binding (every-token) worst-case** 2.4134 ≤ 2.42 | ✅ |

The pass is gated on the **most conservative** (every-token) worst case, not the
convenient one. The adversarial breach is reported as a labelled non-gating
diagnostic (`adversarial_breaches_but_unphysical = True`), not hidden.

---

## 4. Scope and limitations (honest)

- **Bounds the aggregate-PPL leg only.** It complements #158's per-token argmax leg;
  together they cover "is the tree path's *output quality* (argmax + softmax mass)
  within contract." It does not measure TPS.
- **The σ = ε/√3 model treats the measured *max* as a per-logit std** — conservative.
  The real per-logit jitter is far smaller (M=16 bit-identical; median Δ ≈ 0). A
  tighter σ would widen every margin.
- **The transplant is a counterfactual.** The structural finding (§1) says the M=32
  jitter does not actually reach the scored prefill; the transplant bound holds even
  if it did.
- **Prefill-chunk batch-variance is out of scope** (see §1 scope note) — a distinct
  leg if land #71 changes prefill geometry.
- Anchors are committed advisor-branch artifacts; no external-PR borrow, no official
  draws, no served-file change.

**Citations.** Batch-variance / FP reduction-order non-determinism: He et al.,
"Defeating Nondeterminism in LLM Inference" (Thinking Machines Lab, 2025) and
arXiv:2511.17826 ("Deterministic Inference across Tensor Parallel Sizes"); Marlin
kernel context: Frantar et al., arXiv:2408.11743. Softmax/logsumexp convexity: Boyd &
Vandenberghe §3.1; Gao & Pavel, arXiv:1704.00805.

---

## 5. Hand-off (launch evidence-line)

**SAFE: the 0.0433 PPL margin survives M=32 batch-variance.** (1) The scored PPL is
teacher-forced prefill (`prompt_logprobs`); M=32 is decode-only, so the scored PPL is
M-invariant = 2.37667, margin untouched. (2) Even conservatively transplanting the
decode M=32 jitter (ε=0.25) onto every prefill token, worst-case PPL = 2.4013 (#158
f=0.6169) / 2.4134 (every-token) — both ≤ 2.42. The M=32 per-logit perturbation would
have to grow from 0.25 to ~0.35 (#158 f) / ~0.28 (every-token f) to breach; only an
unphysical adversarial-on-every-token model does, ruled out by 0 measured argmax flips
and the symmetric ±1-bf16-ULP regime. **The PPL gate clears at the projected 535–538
TPS.** Feeds fern #155's consolidator and the eventual `Approval request: HF job`
validity stamp.

**Non-collision:** complements #158 (per-token exactness) with the aggregate-PPL
bound, closing the dimension #150 assumed. Distinct from kanna #159 σ_hw (TPS
variance), stark #164 native private-drop (TPS), fern #162 frontier (uses PPL as
input, doesn't bound it), land #71 (builds the kernel).

---

## 6. Reproduce

```bash
cd target/
# pure-analytic, CPU-only (no GPU / vLLM / model load); ~12 MiB peak
python research/validity/tree_path_ppl_margin/ppl_margin_bound.py --self-test \
    --wandb-name "denken/tree-path-ppl-margin-bound" \
    --wandb-group "tree-path-ppl-margin-bound"
```

Exit 0 ⇔ `ppl_margin_bound_self_test_passes` and NaN-clean. All anchors are read from
committed paths (`--divergence-anchor`, `--margin-anchor`, `--ppl-anchor`
overridable).
