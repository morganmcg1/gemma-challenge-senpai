# Research Ideas — 2026-06-13 Round 3

Researcher-agent synthesis. All 7 hypotheses below are net-new relative to in-flight
experiments as of 2026-06-13. Priority ordering: drafter-quality levers first (the
deciding bottleneck for clearing 500 TPS), then architectural alternatives, then
bandwidth/kernel micro-gains.

---

## H1 — Draft-OPD: On-Policy Distillation via Error-Position Replay

**Mechanism.** After standard EAGLE-3 SFT training plateaus, continue training the
drafter via on-policy rollouts: run the draft model autoregressively, let the target
verify, record the positions where tokens were rejected, then replay drafting from
those exact positions with a KL / acceptance-aware loss. This closes the
train-vs-eval distributional gap (drafter trained on target trajectories, evaluated
on draft-induced states) which is the root cause of SFT plateau.

**Expected TPS delta / PPL risk.** +23% acceptance rate over EAGLE-3 SFT baseline
(per Draft-OPD paper). At current p≈0.78 and K*≈10, p=0.96 translates to
~+70–80 TPS on top of the EAGLE-3 baseline. PPL risk: zero — lossless by design
(speculative decode theorem guarantees target distribution). Greedy risk: none beyond
the existing rollback gate.

**First implementation step.** Install the SafeAILab EAGLE-3 repo (already used for
drafter training). After initial SFT checkpoint is produced: wrap the EAGLE-3 draft
step in a rejection-sampling loop against the target logits; log rejected positions
to a replay buffer; on each OPD minibatch, sample from the replay buffer and compute
an acceptance-weighted cross-entropy loss (accepted tokens: standard CE; rejected
tokens: KL toward target). Train for 500–1000 steps with lr 1e-5.

**Reference.** Draft-OPD — arxiv 2605.29343 (May 2026, Shanghai AI Lab + SJTU).
The paper provides acceptance-aware distillation formulation and shows 23% over
EAGLE-3 and 13% over DFlash at matched FLOPs.
https://arxiv.org/abs/2605.29343

---

## H2 — DFlash: Block Diffusion Drafter (Single-Pass Parallel Draft)

**Mechanism.** Replace the autoregressive EAGLE-3 draft head with a block diffusion
model that generates all K draft tokens in a SINGLE forward pass. The diffusion
drafter is conditioned on the target model's hidden-state context (same as EAGLE)
but uses a masked-diffusion objective over a fixed block of K tokens, iteratively
denoising them in parallel. Because draft generation is O(1) passes regardless of K,
the tree-cost model changes fundamentally: wider trees become cheaper.

**Expected TPS delta / PPL risk.** Paper reports 6x lossless acceleration and 2.5x
over EAGLE-3. On the A10G single-stream benchmark, eliminating the K autoregressive
draft steps removes a significant fraction of drafter latency, potentially pushing
K* from ~10 toward 20+ with the same wall-clock budget. Conservative estimate:
+40–60 TPS beyond current EAGLE-3 baseline. Lossless by speculative decode theorem.

**First implementation step.** Clone the DFlash GitHub repo. Identify the drafter
architecture (block diffusion head over target hidden states). Adapt the drafter
input to accept Gemma-4-E4B-it hidden states at the same layer hook as current
EAGLE-3 drafter. Train using DFlash's provided training script on the same
distillation dataset. Profile draft latency per K tokens on A10G to verify
single-pass cost advantage over AR EAGLE-3 draft steps.

**Reference.** DFlash — arxiv 2602.06036 (Feb 2026).
https://arxiv.org/abs/2602.06036

---

## H3 — TALON: Training-Free Adaptive Draft Tree Topology

**Mechanism.** Instead of a fixed-width, fixed-depth tree, TALON builds the draft
tree iteratively at inference time until a fixed token budget is consumed. It uses a
hybrid expansion strategy: expand "deep and narrow" (follow the single most likely
path) for high-certainty contexts where the drafter is confident, and "shallow and
wide" (branch at the top token level) for uncertain contexts. The budget is set to
match the current fixed tree's cost, so no extra compute is spent — topology is
reallocated, not added.

**Expected TPS delta / PPL risk.** Paper reports 1.1–1.3x over fixed-tree baselines
with identical compute budgets by reducing wasted tree nodes. At current K*≈10,
recovering even 15% of wasted tree nodes translates to ~+15–20 TPS. Zero PPL risk
(lossless). Training-free: no new model weights needed, plugs into existing
EAGLE-3 + tree-verify infrastructure.

**First implementation step.** Read TALON paper appendix for the budget-allocation
algorithm. Implement as a wrapper around the existing tree-construction step in
vLLM's speculative decode path: replace the fixed `[4,4,4]` tree spec with a
dynamic expander that queries the draft model's top-k logits, computes entropy, and
decides expand-deep vs. expand-wide per node until the token budget (set = current
tree size) is exhausted. No retraining required.

**Reference.** TALON — ACL ARR 2026 (training-free adaptive draft trees).
Search: "TALON adaptive draft tree speculative decoding ACL 2026"

---

## H4 — Hydra Sequential Draft Heads (vs. Medusa-style Independent Heads)

**Mechanism.** Medusa and similar multi-head drafters predict each future token
independently from the base model's hidden state. Hydra replaces this with
sequentially-dependent draft heads: head i+1 conditions on the output of head i,
approximating autoregressive generation while amortizing the KV cache over a single
target forward pass. This improves acceptance rate over independent heads because
the draft distribution is closer to the true AR distribution.

**Expected TPS delta / PPL risk.** Hydra++ achieves 1.31x over Medusa and 2.70x
over AR baseline on standard benchmarks. In our setting, the gain over a naive
multi-head drafter (if used) would be ~20–30% acceptance lift. The architecture
adds a small sequential dependency overhead in the draft head but eliminates the
need for a separate drafter forward pass per step. Risk: the Hydra architecture was
not benchmarked on A10G or against EAGLE-3 — acceptance parity with EAGLE-3 needs
to be confirmed before committing to full training.

**First implementation step.** Run a feasibility probe: implement 4 sequential Hydra
heads on top of Gemma-4-E4B-it's layer-30 hidden state (same layer used by current
EAGLE drafter). Train for 1000 steps on the distillation dataset. Measure acceptance
rate on 500 greedy-reference samples. If acceptance rate >= 0.70 (below current
EAGLE-3 target), close this branch. If >= 0.75, proceed to full training.

**Reference.** Hydra — arxiv 2402.05109 (Feb 2024).
https://arxiv.org/abs/2402.05109

---

## H5 — SWIFT / CLaSp Self-Speculation via Adaptive Layer Skipping

**Mechanism.** Instead of a separate drafter model, SWIFT uses the target model
itself as its own drafter by skipping a subset of transformer layers during draft
generation. A lightweight gating mechanism learns which layers to skip per token,
so easy tokens take a shallow path (fast, low-quality draft) and hard tokens take
the full path. CLaSp extends this with a dynamic-programming optimizer that selects
the skip schedule per verification stage. No auxiliary model weights needed.

**Expected TPS delta / PPL risk.** SWIFT reports 1.3–1.5x on standard benchmarks
without an auxiliary model. On A10G, layer skipping reduces the memory-bandwidth
cost of the draft step (fewer weight matrices loaded), which is the dominant cost at
M=1. Estimated: +20–35 TPS if acceptance rate of skipped-layer drafts >= 0.65.
Risk: acceptance rate may be lower than EAGLE-3 on Gemma-4-E4B-it because Gemma-4
uses a heterogeneous architecture (alternating local/global attention layers) —
skipping a global-attention layer may cause a large quality drop.

**First implementation step.** Install SWIFT (arxiv 2410.06916, ICLR 2025).
Identify Gemma-4's heterogeneous layer schedule. Restrict layer-skip candidates to
local-attention (non-global) layers only. Run SWIFT's adaptive gate calibration on
500 samples. Measure draft acceptance rate. If acceptance >= 0.65, benchmark TPS.

**Reference.** SWIFT — arxiv 2410.06916 (ICLR 2025).
CLaSp — arxiv 2505.24196.
https://arxiv.org/abs/2410.06916
https://arxiv.org/abs/2505.24196

---

## H6 — Online Drafter Adaptation at Inference Time (OnlineSPEC / DVI)

**Mechanism.** After offline distillation, the drafter's acceptance rate degrades on
out-of-distribution prompts. OnlineSPEC addresses this with an online learning loop:
at inference time, use the verifier's accept/reject signal as a supervision signal
to update the drafter's parameters with a small gradient step per batch (dynamic
regret minimization). DVI (Draft, Verify & Improve) frames this as an online
KL→RL schedule: early updates use KL toward target logits, later updates switch to
an RL reward based on acceptance count. Both approaches are model-agnostic and can
wrap any existing drafter.

**Expected TPS delta / PPL risk.** OnlineSPEC reports 24% speedup over seven
benchmarks. DVI achieves 2.16x AR acceleration with 100x less data than EAGLE-2.
In our setting, the gain is over the EAGLE-3 offline baseline — realistic estimate
+15–25 TPS if the adaptation budget (gradient steps) fits within the per-request
latency budget. Key risk: gradient steps at inference time add latency on each
request; the break-even point requires careful profiling.

**First implementation step.** Implement DVI's online update loop (simpler than
OnlineSPEC's full regret minimizer): after each spec-decode step, compute the
acceptance-weighted KL loss on the just-verified draft tokens (no replay buffer
needed, uses the live verification signal). Apply one Adam step with lr=1e-6 on the
drafter head only. Benchmark: does per-request drafter update overhead < 2ms?
If yes, proceed to full evaluation. Profile first on a single benchmark prompt.

**Reference.** OnlineSPEC — LLA 2026 (online learning for draft models).
DVI — ICLR 2026 submission (arxiv: search "Draft Verify Improve online KL RL").
https://arxiv.org/abs/2510.24021 (SpecKD, related acceptance-aware distillation)

---

## H7 — LiquidGEMM W4A8 Kernel: Recovering Non-Bandwidth Overhead

**Mechanism.** Current int4 Marlin (W4A16) uses Tensor Cores for the weight-fetch
but must dequantize int4 weights to fp16 before multiply-accumulate, creating a
throughput mismatch between Tensor Core load speed and CUDA Core dequant speed.
LiquidGEMM (LiquidQuant) solves this with a hardware-efficient overflow-safe
dequantization primitive that uses exactly two arithmetic instructions, enabling
W4A8 to run at the full Tensor Core rate. On A10G, where ~81% of bandwidth is
already utilized, the remaining ~19% overhead is dominated by kernel launch,
dequant stalls, and CUDA graph gaps — W4A8 directly targets the dequant stall.

**Expected TPS delta / PPL risk.** Paper reports significant GEMM throughput
improvement over W4A16 Marlin. On A10G, 19% non-bandwidth headroom caps the
ceiling; if dequant stalls are 5–10% of that, W4A8 could recover +5–15 TPS.
PPL risk: W4A8 adds int8 activation quantization error on top of W4 weight error —
must measure PPL delta; if PPL > 2.10 (vs. current 2.019) the kernel is a dead end.
Greedy risk: activation quantization may change logits non-trivially, requiring
a new flip-rate measurement.

**First implementation step.** Check if LiquidGEMM has vLLM integration or
standalone CUTLASS kernels compatible with sm_86. If standalone: benchmark a single
Gemma-4-E4B-it linear layer (e.g., q_proj at layer 20) with LiquidGEMM vs. Marlin
W4A16 on A10G. Measure GEMM throughput in GB/s and latency in microseconds.
Compute implied TPS ceiling. If throughput > Marlin by > 10%, file a vLLM
integration PR.

**Reference.** LiquidGEMM — arxiv 2509.01229 (Sep 2025).
https://arxiv.org/abs/2509.01229

---

## Summary Table

| Rank | Hypothesis | Mechanism | TPS Delta Est. | PPL Risk | Reference |
|------|-----------|-----------|----------------|----------|-----------|
| 1 | Draft-OPD | On-policy error-replay post EAGLE-3 | +70–80 TPS | None (lossless) | 2605.29343 |
| 2 | DFlash | Block diffusion single-pass drafter | +40–60 TPS | None (lossless) | 2602.06036 |
| 3 | TALON | Training-free adaptive tree topology | +15–20 TPS | None (lossless) | ACL ARR 2026 |
| 4 | Online adaptation (DVI) | Per-request drafter RL update | +15–25 TPS | Low | ICLR 2026 |
| 5 | Hydra seq. heads | Sequential draft head chain | +20–30 TPS | Low | 2402.05109 |
| 6 | SWIFT/CLaSp | Layer-skip self-speculation | +20–35 TPS | Medium (arch risk) | 2410.06916 |
| 7 | LiquidGEMM W4A8 | Dequant-stall removal via two-op kernel | +5–15 TPS | Medium (PPL risk) | 2509.01229 |
