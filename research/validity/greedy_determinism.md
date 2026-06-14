# Greedy-identity on the deployed frontier: bit-exact or distributional? (PR #73)

**Question.** `program.md` (lines 27–28, 324) requires "Greedy decode must remain
token-identical to plain greedy autoregressive decode." PR #66's relaxed-accept
audit raised the contract-foundation question: is that a **bit-exact** property the
deployed spec-decode frontier can actually satisfy run-to-run, or only a
**distributional** one the PPL gate enforces? With DECISION 2 closing the
relaxed-accept lane (greedy-identity stays non-negotiable, advisor #46 2026-06-14),
the load-bearing question is whether our shipped frontier *honors* the contract it
is held to. This is a **local, no-GPU-launch, contract-safe** characterization: no
served-file change, no relaxed-accept, no HF Job.

## TL;DR verdict

**Greedy-identity on the deployed frontier is BIT-EXACT — and the deployed stack
*satisfies* it.** Measured directly (the deployed `fa2sw_precache_kenyan` stack,
served greedy spec-ON M=1, fresh reload each time, N=7 reloads), the frontier
reproduces its 128-prompt × 512-token output **byte-identical run-to-run** (mean
byte-identical prompt fraction **1.0**, run×run matrix all-ones, **1** distinct
output across all 7 reloads, official `greedy_gate.compare` **GREEDY_IDENTICAL,
0/65,536 divergent tokens**, intrinsic flip hazard **0.0/token**), at ~489 local
TPS — i.e. *above* the ~286-TPS bound the PR posed. PPL is run-to-run invariant to
12 digits (2.376976138392039) and E_accept is identical (3.854).

This **contradicts the premise** that the deployed stack is non-reproducible:
BASELINE.md line 49's "non-reproducible" numbers are *proxies* (spec-OFF control,
plain int4, a `MAX_NUM_BATCHED_TOKENS` parity sweep), **not** the deployed M=1
spec-ON greedy stack with its determinism engineering. Measured directly, the real
stack honors the bit-exact contract — across **all** reloads including the first.

**Source isolation (one factor toggled at a time, `extra_env` only).** The
determinism is not luck: it is the deployed config's deliberate choice to disable
the associativity-breaking reductions while freezing the one live reduction inside a
captured graph. Two of the PR's three named FP-noise sources are demonstrably *off
in the deployed default*, and forcing the third one **on** breaks bit-exactness — a
clean positive control:
- **FA2 sliding-window is INERT on this build.** With `FA_SLIDING=1` (the deployed
  default) the Attention wrapper installs but flips **0** target layers (0
  `-> FLASH_ATTN` lines across all 7 reloads), and `FA_SLIDING=0` produces
  **byte-identical** tokens to default (identical-to-default fraction **1.0**). The
  toggle changes nothing; it cannot be a nondeterminism source here.
- **int4 Marlin atomic-add is OFF in the deployed default, and turning it ON breaks
  determinism.** The deployed config does **not** set `VLLM_MARLIN_USE_ATOMIC_ADD`.
  Forcing `VLLM_MARLIN_USE_ATOMIC_ADD=1` (single changed env var; FA-flips and
  split-KV redirects match default) is **not a no-op**: it shifts the verify-GEMM
  numerics (tokens only **0.1116** identical to default) **and** it fails the clean
  bit-exactness the default shows — official xcheck **DIVERGENT**, **2** distinct
  outputs over 7 reloads, mean byte-identical **0.8214**. So atomic-add is a **live,
  latent** nondeterminism source on this A10G that the deployed config correctly
  avoids by leaving the flag unset. (This **refutes** the earlier "hardware-gated
  OFF, cannot be a source" reading of `marlin_utils.py` — the measured token change
  proves the path engages when the flag is set.)
- **#43 split-KV is active but run-to-run stable.** It genuinely engages in default
  (≥5 verify-redirects/reload); disabling it (`SPLITKV_VERIFY=0`) changes tokens
  (0.0859 identical to default) yet default stays self-identical (1.0) across
  reloads ⇒ its 3D reduction order is replayed identically inside the captured graph.

**FA_SLIDING=0 TPS cost = ~0% (−0.02%) — but as a *no-op artifact*, not a real
kernel-swap cost** (FA_SLIDING flips 0 layers, so =0 vs =1 is the same kernels, same
tokens, same throughput). The BASELINE.md "FA_SLIDING=0 restores byte-identity at an
unmeasured TPS cost" framing is moot on the deployed stack: byte-identity already
holds at `FA_SLIDING=1`, and there is no swap to pay for.

**Contract + DECISION 2:** greedy-identity as written (bit-exact) is *satisfiable
and satisfied* by the deployed frontier, so it is not merely a distributional
property the PPL gate happens to enforce — the stack delivers literal bit-exactness.
The deployed default's **intrinsic run-to-run token churn is zero**, so any
relaxed-accept rule (DECISION 2, closed) would inject *strictly more* divergence
than the stack produces on its own. Reported as evidence; the call is Morgan's.

## The stack under test

`submissions/fa2sw_precache_kenyan` (PR #52, the official #2 frontier: 481.53
a10g-small TPS / PPL 2.3772 / 128/128 VALID; private re-run VERIFIED VALID Δ4.3%).
Linear MTP K=7 + #43 split-KV. Served **unchanged** (git-clean vs HEAD verified),
greedy (`temperature=0, top_p=1, top_k=0`), spec-ON, M=1 single-stream — exactly
the official benchmark protocol.

### Why this is the *right* stack to measure (and why prior evidence didn't measure it)

BASELINE.md line 49 records the deployed stack as "non-reproducible run-to-run,"
but the cited numbers are **proxies, not the deployed stack**:
- spec-OFF control diverges 28/32 — spec decode is OFF, not the served path.
- plain int4 baseline diverges 29/32 — no fa2sw kernels, no determinism engineering.
- lawine #56's ~10/128 self-identical — a `MAX_NUM_BATCHED_TOKENS` parity sweep
  (since closed as a parity keeper), not the deployed M=1 greedy config.

None of these is the deployed **spec-ON** stack with its full determinism
engineering. PR #73 measures that stack directly for the first time.

## Mechanism: token identity reduces to argmax-margin stability

In the deployed greedy fast path (`serve.py` DIXIE_SLIM_GREEDY, lines ~390–456):
the sampler computes `logits.argmax(dim=-1)` and **spec decode emits the target
model's argmax regardless of drafter proposals** (`rejection_greedy_sample_kernel`
emits `target_argmax`). The drafter changes *how many* tokens are accepted per step
(speed), never *which* tokens (identity). So run-to-run token identity reduces
entirely to: **is the target logits' argmax stable across reloads?**

`torch.argmax` is deterministic on ties (lowest index). Run-to-run flips happen only
when floating-point reduction noise (attention/GEMM order) perturbs logits enough to
change *which* token is the max — i.e. only on **near-ties** whose top-1/top-2 margin
is smaller than the FP noise. The deployed argmax is bit-stable because the config
**suppresses the associativity-breaking reductions and freezes the rest**:
- **The two associativity-breaking sources are off in the deployed default.** FA2
  sliding-window flips 0 layers on this build (the `Attention.__init__` eligibility
  test never matches), so it injects no reduction-order noise. int4 Marlin
  **atomic-add** is OFF because the deployed config does not set
  `VLLM_MARLIN_USE_ATOMIC_ADD`; the positive control below shows that when it *is*
  set, the non-associative atomic reduction flips argmaxes and breaks identity — so
  leaving it unset is load-bearing, not incidental.
- **What remains is frozen.** `LOOPGRAPH`/`ONEGRAPH` captured-graph decode fixes the
  kernel launch + reduction order ⇒ bit-identical forward run-to-run; `#43 split-KV`
  (the one live reduction-order change) runs *inside* that captured graph, so its 3D
  order is replayed identically every reload (split-KV toggled off changes tokens but
  default stays self-identical).
- **The margin is wide.** `LM_HEAD_PRUNE` → 12k-vocab lm_head leaves far fewer
  competing logits ⇒ larger top-1 margins (cf. PR #4 `int4_g128_lmhead`), so even
  residual bf16 rounding in split-KV does not reach a near-tie.

Net: argmax is bit-stable because the deployed config disables atomic-add and FA2
sliding is naturally inert on this build, while the remaining math is replayed inside
a captured graph. The positive control (forcing atomic-add ON) confirms the
mechanism: re-introduce one non-associative reduction and bit-exactness fails.

## Method

`scripts/validity/greedy_determinism.py` serves the deployed stack UNCHANGED, N
times with fresh reloads, capturing greedy spec-ON decode token IDs via the official
`decode_outputs.py` (128 public prompts × 512 tok, `ignore_eos`, M=1). Source
isolation toggles ONE factor per config via `extra_env` only (no served-file change):

| config           | toggle                          | tests |
|------------------|---------------------------------|-------|
| `default`        | deployed (FA=1, splitkv=1, atomic off) | baseline run-to-run identity (N=7) |
| `fa_sliding_off` | `FA_SLIDING=0`                  | captured-graph path; **+ FA_SLIDING=0 TPS cost** (load-bearing) |
| `splitkv_off`    | `SPLITKV_VERIFY=0`              | does #43 3D split-KV reduction order break identity? |
| `atomic_on`      | `VLLM_MARLIN_USE_ATOMIC_ADD=1`  | does non-associative int4 Marlin atomic-add break identity? + its TPS |

`scripts/validity/analyze_determinism.py` (CPU-only) builds the run×run byte-identity
matrix, mean per-token agreement, first-divergence onset, censored-geometric flip
hazard per token, **cluster signatures** (distinct-output count + largest mutually
byte-identical cluster — distinguishes a first-reload outlier from stochastic
scatter), official `greedy_gate.compare` cross-check, and folds in sglang TPS +
teacher-forced PPL + E_accept per reload.

**Faithfulness caveats.**
- `VLLM_USE_FLASHINFER_SAMPLER=0` (PyTorch-native sampler) is forced by this
  container (cuRAND JIT unavailable). The submission does not pin the sampler. For
  greedy decode this is determinism-neutral: both samplers do argmax over the *same*
  logits; the run-to-run noise lives in logit computation, which is sampler-
  independent. The official a10g runner may auto-select flashinfer, but the argmax-
  margin mechanism is identical.
- Local sglang `output_throughput` reads ~489 TPS here vs 428 "local steady" /
  481.53 official — absolute TPS is harness-dependent. The **FA_SLIDING=0 TPS cost**
  is reported as a *ratio* (default vs fa_sliding_off under the identical bench
  protocol), which is harness-independent.

## Results

All four source-isolation configs, deployed stack served unchanged, fresh reload
per run. Token identity by SHA256 of completion token IDs; official cross-check via
`greedy_gate.compare`. (W&B run `45y7ui1o`, group `greedy-determinism`.)

| config | N | mean byte-id (self) | distinct outputs | largest id cluster | identical-to-default | official xcheck | flip hazard/tok | TPS median | PPL spread | E_accept |
|--------|---|---------------------|------------------|--------------------|----------------------|-----------------|-----------------|------------|------------|----------|
| `default`        | 7 | **1.0** | **1** | **7/7** | — (reference) | **GREEDY_IDENTICAL** (0/65536) | **0.0** | 489.22 | 0.0 (2.376976138392039) | 3.854 |
| `fa_sliding_off` | 3 | 1.0 | 1 | 3/3 | **1.0** | GREEDY_IDENTICAL (0/65536) | 0.0 | 489.34 | 0.0 | 3.854 |
| `splitkv_off`    | 3 | 1.0 | 1 | 3/3 | **0.0859** | GREEDY_IDENTICAL (0/65536) | 0.0 | — | — | 3.858 |
| `atomic_on`      | 7 | **0.8214** | **2** | **6/7** | **0.1116** | **DIVERGENT** (80/128 prompts, 23387/65536 tok = 35.7%) | 3.9e-4 | 491.05 | — | 3.85→3.822 |

Reading the table:
- **`default` is bit-exact run-to-run.** All 21 pairwise comparisons across 7 fresh
  reloads are byte-identical (matrix all-ones); a single distinct output;
  `GREEDY_IDENTICAL` with 0 divergent tokens of 65,536; flip hazard 0.0/token. PPL
  identical to 12 digits and E_accept identical across every reload. This is the
  load-bearing result: **the deployed stack literally honors the bit-exact contract.**
- **`fa_sliding_off` reproduces default byte-for-byte** (identical-to-default 1.0)
  and FA flips 0 layers in default ⇒ FA2 sliding-window is inert; its −0.02% TPS
  delta is a no-op artifact (same kernels), **not** a kernel-swap cost.
- **`splitkv_off` is active but stable.** It changes tokens vs default (0.0859
  identical — split-KV genuinely engages, 5 redirects/reload) yet stays self-identical
  across reloads ⇒ the #43 3D reduction order is run-to-run stable inside the captured
  graph.
- **`atomic_on` is the positive control — it breaks bit-exactness.** Forcing the
  non-associative int4 Marlin atomic-add reduction shifts ~88% of tokens vs default
  (0.1116 identical) and is **not** self-reproducible: **2** distinct outputs over 7
  reloads. The structure is a **first-reload outlier** — run_00 differs while
  run_01…run_06 are mutually byte-identical (largest cluster 6/7). That is **not**
  per-token stochastic FP noise (which would scatter into many distinct signatures);
  it is consistent with a **one-time autotune/warm-up** of the newly-enabled
  atomic-add kernel path that then settles. Crucially, the deployed default shows
  **no** such first-reload effect — all 7 reloads including the first are identical —
  so keeping atomic-add OFF is load-bearing.

## Contract implication

**Greedy-identity as written in `program.md` (bit-exact, token-identical) is
satisfiable AND satisfied by the deployed frontier.** It is not merely a
distributional property the teacher-forced PPL gate happens to enforce: the served
M=1 spec-ON greedy stack reproduces its full 128×512 output byte-for-byte across 7
fresh reloads, with the official `greedy_gate.compare` returning `GREEDY_IDENTICAL`
and zero divergent tokens. **Verdict code 0 (bit-exact).** This resolves the #66
contract-foundation question: the contract rests on a real, measured bit-exact
property of the shipped stack, not on a gate's tolerance.

Two reconciliations:
- **BASELINE.md line 49 ("non-reproducible") is not wrong about its configs — it is
  wrong about the deployed stack.** Its numbers are proxies (spec-OFF control, plain
  int4, a batched-tokens parity sweep). None is the deployed spec-ON M=1 greedy stack
  with captured-graph decode + atomic-add-off + inert FA2. Measured directly, that
  stack is bit-exact. The "FA_SLIDING=0 restores byte-identity at a TPS cost" framing
  is moot: byte-identity already holds at FA_SLIDING=1, and FA flips 0 layers so there
  is no swap to pay for (−0.02% is a no-op artifact).
- **Run-to-run token churn does not move the gates.** PPL is teacher-forced on a
  fixed GT corpus (#21), so it is invariant to free-running token identity by
  construction — observed spread 0.0 (12 digits) across reloads. The private re-run
  TPS gate (Δ ≤ 5%) is unthreatened: TPS is stable across reloads (489.0–489.9 in
  default, ~0.2% spread). So even were the stack *not* bit-exact, the gates would not
  detect it — which is exactly why the direct token-level measurement was needed, and
  why kanna #38 flagged the official gate as token-identity-blind for default paths.

**Caveat on portability of the literal bytes.** The *bit-exactness* is measured with
the container's PyTorch-native sampler (`VLLM_USE_FLASHINFER_SAMPLER=0`). The
mechanism (argmax-margin stability over sampler-independent logits) is sampler-
agnostic, so the property should hold on the official runner; but the specific token
*bytes* could differ if the official runner selects flashinfer and a different kernel
order. The contract claim is about **run-to-run identity of a given stack**, which is
what we measured, and which holds.

## DECISION 2 relevance (evidence, not recommendation)

DECISION 2 (advisor #46, 2026-06-14) closed the relaxed-accept lane: greedy-identity
stays non-negotiable. This PR supplies the *evidence* that makes that ruling
self-consistent, without making the call (Morgan's):

- **The deployed default's intrinsic run-to-run divergence is zero** (0/65,536
  tokens, flip hazard 0.0/token). The stack is not "already a little nondeterministic,
  so a relaxed-accept rule is free." It is bit-exact.
- Therefore **any relaxed-accept rule would inject strictly more divergence than the
  stack produces on its own** (which is none). The #66 "spend the PPL headroom"
  framing has no slack to spend at the token level: the gate's PPL invariance is not
  evidence of token-level slack — it is orthogonal to token identity (teacher-forced).
- **The positive control quantifies what one extra non-associative reduction costs:**
  forcing atomic-add ON flips ~36% of tokens on a divergent pair and produces a
  first-reload outlier. A relaxed-accept rule is a deliberate, larger version of the
  same thing — trading bit-exactness for speed. The data says the deployed stack does
  not need that trade to hit ~489 local TPS, and program.md forbids it. Reported as
  evidence; the call is Morgan's.
