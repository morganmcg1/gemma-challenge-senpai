# Forward-Direction Research Ideas — answer to Issue #481

- **2026-06-16 11:33Z**
- Source: forward-direction literature survey (researcher-agent), cross-checked against our own measured anchors.
- Question (#481, human): what's next after the strict 222 ship — vLLM vs SGLang, kernels vs spec-dec, what large areas are we missing?

---

## THE CRUX ANSWER (decisive): there is no free fast byte-exact GEMM

**Does a fast byte-exact (deterministic-reduction) GEMM kernel exist that is NOT ~48% slower than nondeterministic split-K? Per the literature: No — not at BF16, not yet.** The determinism tax is a **structural IEEE-754 floor**, not a vLLM implementation deficiency. It is independently reproduced across three published systems:

- **vLLM `batch_invariant`** (Triton persistent kernel): up to **~50% slower** than fused nondeterministic CUDA.
- **TBIK** (Tree-Based Invariant Kernels, arXiv 2511.17826): MatMul at **63% of cuBLAS** throughput at large batch; the full deterministic-IO overhead band is **22–63% vs vanilla BF16**, and the authors explicitly note batch-1/small-M is at the **high end** (fixed block-size waste).
- **LayerCast** (NeurIPS 2025 oral, arXiv 2411.02076): buys determinism by computing in FP32 with BF16 storage — but FP32 is 2× the per-FLOP cost, and decode is already **memory-bandwidth-bound**, so it trades bottlenecks with no batch-1 win.

**Mechanism:** FP non-associativity means any *correct parallel* reduction that must produce a fixed bitwise result has to either (a) **serialize the reduction order** (lose the parallelism that makes split-K fast), or (b) **enforce a canonical tree order** via padding/masking/fixed blocks (pay extra work). There is no known BF16 technique that escapes both.

**Our own data corroborates the floor exactly.** land measured the global-flag tax end-to-end at **51.39%** (run [`f7zwyoc8`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/f7zwyoc8): deployed-no-flag arm 460.12 TPS vs strict-flag arm 223.65 TPS, single-stream). Our 51.39% sits right at the **top** of the literature's 22–63% range — consistent with batch-1 single-stream being the worst case for invariant kernels. So the 222→481 gap is **not** a tuning miss; it is the determinism tax (the bulk of the gap) stacked on the ~3.7% drafter-acceptance private risk (denken [#486](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/shcdordv)).

**vLLM vs SGLang:** the tax is **engine-agnostic** (it's IEEE-754, not a vLLM bug) — switching engines does NOT dodge it. SGLang's only material relevance here is that it ships **EAGLE-3** spec-dec (see Direction 5), which is a drafter-quality lever, not a determinism lever.

**Implication:** the strict frontier above 222 is NOT reachable by "find a better invariant matmul." It requires one of: **(i) run fewer matmuls deterministically** (only the ones that can actually flip an output token), or **(ii) make the matvec case structurally cheaper** (M=1 decode is a matvec, not a matmul), or **(iii) make the drafter more accurate on hard prompts** (closes the private-set gate risk, orthogonal to TPS). Ladder below.

---

## RANKED DIRECTIONS

> **Epistemic note:** the *mechanisms* below are well-grounded (IEEE-754 non-associativity; matvec ⊂ matmul; EAGLE multi-layer fusion). The specific recent arXiv IDs are **survey leads to verify before committing GPU time** — several are very recent with no public code found. Treat Directions 2 & 4 (pure diagnostics/benchmarks) as the de-risking front door.

### 1 — Selective-determinism scheduling (LLM-42 style) — highest upside, least de-risked
Route a matmul through the **fast** split-K kernel whenever the input is in a "consistent state" (the fast kernel's rounding is provably order-independent for *this* input), fall back to the slow invariant kernel only when a cheap consistency oracle fails. **Decouples determinism from kernel choice** — the only published idea that does. If even ~50% of decode matmuls pass the check, effective tax drops ~48%→~24% → strict TPS ~222→~300+. **Unknown:** the check pass-rate on Gemma-3/A10G BF16 — that is the whole ballgame. Lead: arXiv 2601.17768 (no public code). Prereq: Direction 2.

### 2 — Layer/op reduction-sensitivity profiling — cheap diagnostic, do FIRST
For each GEMM in decode (Q/K/V/O projections, gate/up/down MLP), run fast-split-K vs invariant over ~1000 decode steps; record (a) how often outputs differ and (b) when they differ, whether the perturbation is large enough to flip the **argmax** (the only thing the byte-exact gate sees). Output = **the minimum set of matmuls that MUST be deterministic** to keep 128/128 token identity. If even a handful of low-variance projection layers (post-RMSNorm) are order-insensitive, they run fast for free. **This is pure measurement, ~4 GPU-h, and it gates Directions 1 & 3.** This is the single highest-value next experiment. *(Note: this overlaps ubel #484's per-op tax attribution and lawine #488's surgical-attention realization — the natural home is to extend that line from "where is the tax" to "which ops can legally shed it.")*

### 3 — Custom Triton **matvec** kernel for the M=1 Gemma-3 shapes — most actionable kernel change
At batch-1 decode, every GEMM is M=1 → a **matvec**, a strictly simpler determinism problem than GEMM (no N-parallelism to synchronize, only a K-reduction). A fixed canonical binary-tree reduction over K — with no padding waste if K is tile-divisible — should land **~15–25% overhead vs split-K** instead of ~48%, pushing strict ~222→~260–280. Survey's preliminary tile-divisibility check on Gemma-3 weight dims looked clean (divisible by 128), **but verify the exact E4B shapes before staking the kernel on it.** Leads: TBIK arXiv 2511.17826, Flash-Decoding arXiv 2311.01581; start from vLLM's Triton matmul dispatch, use an explicit-tree `combine_fn` instead of nondeterministic `tl.sum`.

### 4 — cuBLASLt deterministic-algorithm mode — cheapest possible win, benchmark-only
vLLM currently routes *all* deterministic matmuls through Triton `batch_invariant`. NVIDIA's cuBLASLt has a deterministic-algorithm restriction that yields hand-tuned SASS kernels — which **may be faster than Triton** for our exact M=1, K∈{...} shapes. This is a **2–4 GPU-h benchmark** (cuBLASLt-deterministic vs Triton-invariant at the Gemma-3 decode shapes); if it's 10%+ faster, wiring it into the dispatch path for the mandatory-deterministic ops (from Direction 2) is a targeted change. Free-10%-or-closes-the-path. Do concurrently with Direction 2.

### 5 — EAGLE-3 drafter upgrade — directly de-risks the LIVE ship gate
**This connects straight to the denken #486 ship risk.** The drafter-keeping strict config (~222–234) is gate-risky because the drafter's acceptance rate sags on the shifted PRIVATE prompt distribution (the 3.661% acceptance bucket → 24–37% one-shot breach). EAGLE-3 fuses features from **multiple decoder layers** (e.g. {−1,−4,−8}) instead of just the top layer → higher, more robust E_accept on hard/OOD prompts. A higher-acceptance drafter **shrinks the private-set Δ below the 5% gate** — i.e. it could make a *fast, drafter-keeping* config private-safe, which is exactly the coexistence question denken #489 is analyzing right now. Low cost (a few GPU-h to train + measure E_accept on a held-out split), bounded downside (revert to current drafter). Ships in SGLang. Lead: EAGLE-3, NeurIPS 2025.

### Ruled out / low-priority
- **6 — MPK megakernel fusion** (arXiv 2512.22219): 1.7× number is **multi-GPU TP≥2**; single-GPU batch-1 gain is second-order (only saves intermediate-DRAM round-trips). Validate single-GPU before any effort.
- **7 — DASH deterministic-attention scheduling** (ICLR 2026): training/backward-pass technique; batch-1 forward decode has almost no operator-level parallelism to overlap. Methodological reference only.
- **8 — INT8/FP8 exact accumulation: CLOSED.** Integer accumulation IS order-independent (would kill the tax), but INT8/INT4 quant does **not** preserve byte-exact BF16 token identity, and **FP8 tensor cores do not exist on A10G (sm_86 Ampere)**. Dead on this hardware target.

---

## RECOMMENDED EXECUTION ORDER
1. **(this week, ~4 GPU-h)** Direction 2 — reduction-sensitivity profiling → the minimum mandatory-deterministic op set. Extends ubel #484 / lawine #488.
2. **(concurrent, 2–4 GPU-h)** Direction 4 — cuBLASLt-deterministic vs Triton-invariant benchmark at the M=1 shapes.
3. **(1–2 wk, conditional)** Direction 3 — custom Triton matvec, if (1) shows most ops must stay deterministic AND (2) shows cuBLASLt isn't materially better.
4. **(parallel, 2–4 GPU-h)** Direction 5 — EAGLE-3 drafter, measure E_accept on a hard held-out split. **De-risks the live ship gate independent of all kernel work.**
5. **(research horizon)** Direction 1 — LLM-42 selective scheduling, once the boundary op set from (1) is known.

## CONFIDENCE
- **48% determinism tax is real and structural** — strong (3 independent systems + our own measured 51.39%). No published BF16 approach closes it without serializing the reduction or changing the arithmetic type.
- **Direction 2 (profiling) is the highest-value, lowest-risk next move** — it converts "the tax is ~48%" into "exactly which ops are paying it and which can legally stop," and gates the two real kernel paths.
- **Direction 5 (EAGLE-3) is the one with a direct line to the live decision** — it attacks the private-breach gap that is currently forcing the floor-lock-vs-222 choice.
