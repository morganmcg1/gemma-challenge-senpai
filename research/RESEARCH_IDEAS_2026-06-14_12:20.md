# Research Ideas — 2026-06-14 12:20

Generated after: full TPS-accounting audit, non-GEMM overhead profiling review,
scatter-elimination equivalence proof (lmhead12k_scatter_equiv.py), and literature
survey covering Sequoia/OPT-Tree, Lookahead/Jacobi, LayerSkip, REST/RASD,
DynaSpec/NanoSpec/MicroSpec, BASS, SwiftSpec, FastMTP.

Constraint: do NOT re-propose tree-shape re-optimization, descent kernel itself,
GEMM-bandwidth/SplitK (#117/#130/#108), sub-4-bit weight body (#132), QuantSpec
drafter-KV (#121), fp8 KV-cache (#141), Marlin coarser-group (#140),
lm_head column-prune (#144), or scale-palette/LUT (#110).
Off-limits: drafter acceptance-rate, drafter-architecture work, AdaEDL,
private-gap probes, KV-cache prefix prewarm.

---

## RANK 1 — Scatter-free verify argmax [PCK-04 path]

### What it is
Replace the current `scatter(M, 262144) + argmax(262144)` verify path with
`kept_ids[argmax(partial[M, 12k])]`. Eliminates the full-vocab scatter buffer
and shrinks argmax from 262144 → ~12k elements per row.

### Mechanism and evidence
Proven token-identical by `scripts/profiler/lmhead12k_scatter_equiv.py`:
equivalence_rate = 1.0 on real Gemma-4 weights AND on adversarial tie-injection
sweep (M = 1,7,17,25,45). Proof relies on: (1) kept_ids strictly ascending
(verified), (2) bf16 argmax first-occurrence tie-break on A10G, (3) -inf fill
in unused slots of the scatter buffer is never selected.

Memory savings at M=32: eliminates one M × 262144 × 2B = 16.8 MB buffer fill
per step. Argmax compute: 21x fewer reads (262144 → 12k). Both savings are
inside ONEGRAPH, so they reduce per-step wall time without adding launch
overhead.

### Literature support
NanoSpec (2025) and MicroSpec (2025) both use GPU-resident "active vocabulary"
of 40x smaller size (< 3k tokens) for exactly this pattern. FR-Spec shows >75%
vocab can be removed with high acceptance maintained.

### Implementation
In the verify path (post lm_head projection), after the `[M, 12k]` partial
logits tensor is computed:
- Skip the scatter-to-full-vocab step entirely.
- Run `token_ids = kept_ids[partial.argmax(dim=-1)]`.
- `kept_ids` is a static int32 tensor of size ~12k, already resident in GPU
  memory as part of PCK-04 setup.

No architecture change. No PPL impact (greedy-equivalent by proof). Step_time
reduction is direct.

### Suggested experiment
Minimal diff: comment out the scatter kernel call and buffer allocation in the
verify forward pass; replace the final argmax with the two-liner above.
Smoke test: run equivalence check from lmhead12k_scatter_equiv.py against live
model to confirm token-by-token identity. Then run local validation scorer.
Primary observable: step_time reduction (profiler) and output_tps gain (scorer).

### Taste scores
- Mechanistic grounding: 4 (equivalence proven in codebase profiler, zero
  speculation)
- Research-state value: 4 (directly reduces step_time, the only remaining
  free lever outside E[T])
- Execution value: 4 (minimal diff, cheap validation, no correctness risk)

---

## RANK 2 — Static tree mask precomputation

### What it is
Precompute the fixed [M, M] ancestor/causal tree mask once at ONEGRAPH capture
time and store as a constant GPU buffer, eliminating the per-step dynamic
`tree_mask_construct` call.

### Mechanism and evidence
Tree topology is fully fixed at deployment: depth-9, max-branch-3, M=32.
The mask does not vary across requests. `scripts/profiler/tree_nongemm_overhead.py`
profiles `tree_mask_construct` as a per-step non-GEMM op. Under ONEGRAPH,
op launches are pre-queued, but the compute itself still runs. A precomputed
constant buffer replaces runtime computation with a single cudaMemcpy at
capture time.

Cost: ~0 per step (constant load). Zero correctness risk — the mask is
deterministic given the fixed topology.

### Implementation
At ONEGRAPH capture: compute tree mask once, store as
`self.static_tree_mask = tree_mask_construct(...).contiguous()` in the
model/engine state. In the per-step verify path, replace the construct call
with a reference to `self.static_tree_mask`. If CUDA graph replay reuses the
same tensor address, no copy is needed at all.

### Suggested experiment
Single-function replacement. Profile step_time before/after with
tree_nongemm_overhead.py. Confirm mask byte-equality under a sweep of M values.

### Taste scores
- Mechanistic grounding: 3 (profiler shows the op exists; absolute saving
  depends on GPU scheduling which has not been directly measured for this op
  alone)
- Research-state value: 3 (separates static-mask saving from other per-step
  costs; result directly interpretable)
- Execution value: 4 (near-zero implementation effort, cheap verification)

---

## RANK 3 — N-gram lookahead cache for tree draft augmentation

### What it is
A training-free cache of the last W n-grams from the generation stream, used
to inject high-confidence additional branches into the draft tree. Completely
separate from drafter acceptance-rate (adds proposals; does not change drafter).

### Mechanism and evidence
Lookahead Decoding (ICML 2024, Fu et al.): caches Jacobi-trajectory n-grams;
1.5–2.3x speedup on single GPU, training-free. REST (NAACL 2024): longest-prefix
match from a datastore. RASD: retrieval-augmented speculative decoding,
datastore-free variant.

In this setting: the benchmark uses 128 fixed prompts with fixed output_len=512
and a fixed seed. The n-gram cache warms up within the first few prompts. By
prompt 10–20, the cache has seen enough output subsequences to provide useful
proposals for later prompts on the same benchmark. This is particularly
advantageous under the fixed-benchmark eval where the same prompt set is always
used.

E[T] impact: n-gram hits that enter the tree as additional branches and get
accepted by the verifier count as accepted tokens. They do not bypass the
verifier — they are verified exactly as usual.

### Implementation
Maintain a Python dict mapping (t_{-n}, ..., t_{-1}) → t_0 for n=3,4. After
each accepted step, insert all subgrams of the accepted suffix. Before each
draft tree proposal, check if the current top-1 prefix matches any cached
n-gram; if so, insert it as a branch (with a cap to avoid exceeding M=32).

### Caveat
Does not help if the benchmark prompts are fully independent (no repeated
subsequences). Needs empirical measurement on the actual 128-prompt set. If
the prompts share no output subsequences, the cache hit rate will be near zero
and the overhead should be negligible (dict lookup only).

### Suggested experiment
Implement n-gram cache as a wrapper around the draft step. Run local scorer.
Log cache hit rate alongside TPS. If hit rate < 1%, abandon. If hit rate > 5%,
measure E[T] uplift.

### Taste scores
- Mechanistic grounding: 3 (well-grounded in literature for single-GPU serial
  decode; applicability to fixed-benchmark eval is speculative but testable)
- Research-state value: 4 (cache hit rate is a direct discriminating observable;
  result is interpretable regardless of direction)
- Execution value: 3 (training-free, pure Python, cheap to prototype; risk is
  near-zero overhead if cache misses)

---

## RANK 4 — Fused [lm_head_partial → argmax_12k → token_select] kernel

### What it is
After scatter elimination (Rank 1), the verify path becomes:
lm_head GEMM (M×2560 → M×12k) → argmax(M×12k) → gather(kept_ids).
Fuse the argmax and gather into the lm_head GEMM epilogue as a single
Triton/CUDA kernel.

### Mechanism and evidence
MicroSpec (2025): "highly parallel, lock-free CUDA kernels" for in-context
vocab argmax. NanoSpec (2025): "asynchronous gathering and GPU-resident state
management." Both show that fusing the argmax + gather into the projection
epilogue reduces intermediate buffer stores and launches even under graph capture.

At M=32, K=12k, the argmax is a small reduction (32 rows × 12k cols). Fusing
it eliminates one round-trip to HBM for the [M, 12k] tensor. Under ONEGRAPH
the launch is pre-queued but the HBM traffic is real.

### Implementation
Triton kernel: take the matmul output tile as input, compute row-wise argmax
in shared memory, emit a single int32 per row (the argmax index into kept_ids).
Then a scalar gather (trivially fused). The GEMM itself remains Marlin W4A16;
only the epilogue changes.

This is higher implementation effort than Rank 1–3 and depends on Rank 1 being
in place first.

### Suggested experiment
Rank 1 first. Then write a Triton kernel for the fused argmax+gather epilogue,
benchmark it vs the unfused path at M=32 using the profiler. Only proceed to
full integration if the isolated kernel benchmark shows measurable latency
reduction.

### Taste scores
- Mechanistic grounding: 3 (mechanism is sound; saving depends on HBM bandwidth
  vs shared-memory compute trade-off at M=32, K=12k which has not been directly
  measured here)
- Research-state value: 3 (isolated kernel benchmark is cheap and discriminating)
- Execution value: 2 (higher implementation effort; depends on Rank 1; should
  be staged behind the Rank 1 diff)

---

## RANK 5 — BUG-1 fix: depth-1 spine accept rate via target_logits_indices plumbing

### What it is
BUG-1: depth-1 spine accept rate = 0.679 vs target 0.7287 (~96% of gap fixable
via `target_logits_indices` plumbing per EXPERIMENTS_LOG). This is a verify-path
logits routing fix, NOT a drafter-architecture change.

### Mechanism and evidence
The depth-1 position in the tree should receive logits from the target model
indexed by `target_logits_indices`. A plumbing bug causes it to receive the
wrong slice, depressing acceptance at the first spine position. Fix is in the
verify path index arithmetic.

E[T] uplift estimate: depth-1 is the root accept; every path through the tree
passes through it. If the depth-1 accept rate rises from 0.679 to 0.7287
(+7.3%), E[T] rises proportionally for all depth-≥1 tokens. With E[T]=5.04
post-descent-fix, a 7% uplift at depth-1 translates to roughly +0.2–0.35 E[T]
depending on the tree shape.

### Implementation
Identify where `target_logits_indices` is constructed for the depth-1 spine
position in the verify forward pass. Check whether the index into the target
model output logits is off by one or using the wrong token position. Compare
against the descent-walk reference implementation.

### Suggested experiment
Log depth-1 accept rate before and after the fix in a short local run. Confirm
it rises toward 0.7287. Then run the full scorer.

### Taste scores
- Mechanistic grounding: 4 (root cause identified in EXPERIMENTS_LOG, ~96%
  fixable fraction quantified, mechanism is a concrete indexing error)
- Research-state value: 4 (depth-1 accept rate is a precise observable; result
  cleanly confirms or refutes the hypothesis)
- Execution value: 3 (implementation is a targeted plumbing fix; low risk if
  isolated and tested against greedy-equivalence)

---

## RANK 6 — Partial verify early termination

### What it is
In the tree descent walk, once all live branches at depth d are chain-rejected,
skip the argmax/sampling scan for deeper subtrees of those branches. Reduces
per-step argmax invocations on low-acceptance paths.

### Mechanism and evidence
At depth-9/M=32, the tree has many branches. On a step where accept rates are
low (e.g. the top-2 branch at depth-3 is rejected), all descendants of that
branch are guaranteed rejected regardless of their token values. The current
implementation may still scan their logits. Early termination avoids that work.

Saving scales with the fraction of branches that terminate early, which is
(1 - α)^d for a uniform accept rate α at depth d. At α=0.7, depth-9:
probability of at least one early termination = high; expected computation
saved = moderate.

### Implementation
In the descent walk loop, maintain a bitmask of active branches. After each
depth's accept/reject pass, mask out rejected subtrees. Skip argmax for
masked rows. Pure Python/PyTorch logic change, no kernel changes.

### Suggested experiment
Instrument the descent walk to log the fraction of rows skipped per step.
If average skip fraction > 10%, implement the early exit and measure step_time.

### Taste scores
- Mechanistic grounding: 2 (mechanism is valid; quantitative saving depends on
  branch structure and actual accept rates not directly measured here)
- Research-state value: 3 (skip-fraction logging is a cheap diagnostic that
  directly determines whether to proceed)
- Execution value: 3 (pure logic change, no kernel work, easy to instrument)

---

## RANK 7 — K=8–9 static depth increase (post-descent-fix)

### What it is
After descent-fix (#71) is live, increase the static draft depth from K=7 to
K=8 or K=9. Sequoia (NeurIPS 2024) shows E[T] scales logarithmically with
tree size; each additional depth level adds diminishing but positive E[T].

### Mechanism and evidence
Sequoia (NeurIPS 2024, Chen et al.): dynamic-programming optimal tree
construction; E[T] grows logarithmically with number of draft tokens. OPT-Tree:
outperforms Sequoia under various tree sizes.

Current tree: depth-9, M=32. Adding K=8→K=9 adds one depth level. The E[T]
gain at depth-9 is small (diminishing returns) but the step_time increase must
be profiled. Net gain = ΔE[T] / E[T] must exceed Δstep_time / step_time.

This is a static topology change — unlike AdaEDL (off-limits), the tree shape
is fixed at compile time.

### Caveat
kanna #138 is already running a K-sweep. Do not assign this independently until
#138 results are in. If #138 shows K=8 or K=9 is net-positive, that result
confirms this hypothesis.

### Suggested experiment
Wait for #138 results. If K=8 shows net TPS gain, propose a confirmation run
with the descent-fix in place (since #138 may run before #71 is merged).

### Taste scores
- Mechanistic grounding: 3 (logarithmic E[T] scaling is well-supported in
  Sequoia; the net trade-off depends on step_time profiling)
- Research-state value: 2 (partially covered by in-flight #138; independent
  value is low until those results are in)
- Execution value: 2 (low implementation cost but gated on #138 results and
  #71 merge; not independently actionable right now)

---

## Ruled-out directions (do not re-propose)

- **LayerSkip self-speculative**: requires special training-time layer dropout
  recipe; not applicable to existing gemma-4-e4b-it weights without retraining.
- **Pure Jacobi iteration without n-gram cache**: "can barely see wall-clock
  speedup in real-world LLM applications" (Lookahead Decoding paper).
- **TTFT/prefill optimization**: at MAX_CONCURRENCY=1, output_len=512, prefill
  is < 1% of total wall time; not worth optimizing for the official scorer.
- **DynaSpec dynamic vocab routing**: meta-classifier routing overhead requires
  parallel GPU streams; designed for drafter LM head bottleneck, not the
  verify-side argmax which is already post-pruning to 12k.
- **SwiftSpec dual-GPU draft+target parallelism**: requires two physical GPU
  sets; a10g-small is single-node, single-GPU target.
- **BASS batched attention**: designed for batch > 1; MAX_CONCURRENCY=1 means
  batch is always 1 at the request level.

---

## TPS composition reminder (for experiment planning)

```
official_TPS = K_cal * (E[T] / step_time) * tau
K_cal = 125.268, tau = 1.06019, step_time = 1.2182 ms (FIRM, depth-9)

Current (BUG-2 broken):  E[T] = 2.621  → TPS ≈ 286
Post-descent-fix (#71):  E[T] = 5.041  → TPS ≈ 550 (oracle measured ~522)
Clear-500 bar:           E[T] ≥ 4.862
```

Step_time reductions (Rank 1, 2, 4) multiply the entire TPS budget.
E[T] increases (Rank 3, 5, 6, 7) add to the numerator only.
The two levers are orthogonal and compound.
