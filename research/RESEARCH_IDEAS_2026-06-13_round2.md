# Research Ideas — Round 2 (2026-06-13)

Generated from: public challenge bucket docs + external literature only.
Scope: push above ~420 TPS (current top VALID entry) and/or close the 4–9% public→private TPS
reproduction gap. Confirmed dead-end paths excluded.

---

## Dead-End Map (DO NOT RE-PROPOSE)

Consolidated from public bucket docs (int4_ceiling_notes, gemma_specdecode_headroom_flowian):

- All sub-4-bit weight paths: AWQ/GPTQ/AQLM/QuIP#/NVFP4 — blocked vLLM 0.22 + Ampere sm_86
- fp8 KV cache: A10G rejects both e5m2 and e4m3; Gemma 4 attn asserts fp8/e4m3/nvfp4
- Attention backend swap: TRITON_ATTN force-pinned (mixed head dims 256/512 forbid FA/FlashInfer)
- GEMM kernel swap: already MarlinLinearKernel for CompressedTensorsWNA16 — no swap gain
- MTP depth K>7: spec8 = -5.82 TPS (statistically real regression)
- Centroid top_k widening: no acceptance gain, TPS regressed to ~265
- Pure n-gram/PLD drafter: 1.29–1.41 tok/fwd vs 3.55 (MTP) — below break-even
- MTP+PLD hybrid via host-side lookup: net negative (async worth ~+50 TPS killed by sync scheduling)
- Body channel-wise quantization: no TPS gain, costs PPL
- Runtime knob sweep (max_num_seqs, MARLIN_USE_ATOMIC_ADD, MNBT, etc.): community-swept
- Community DFlash drafter: ~2–5% acceptance (near-random — conditioning/training bug)

---

## Ranked Research Ideas

### Rank 1 — Distribution-Robust KL-Distilled Drafter (≥9k Corpus)

**What it is.** Retrain the existing MTP drafter with a wide, distribution-matched corpus (≥9k
prompts spanning ShareGPT, MMLU-Pro, GPQA, MATH/AIME, misc instruct) using KL-distillation on
top-2048 logits per propose-call trace instead of argmax-CE on the 128 public bench prompts.

**Why it might help.** The 4–9% public→private TPS gap is not engine noise (within-bucket
run-to-run Δ is ~0.2%). Analysis in tps_repro_gap_itaca/README.md confirms the gap is a
prompt-distribution shift: the drafter's accepted-tokens/step is prompt-content-sensitive, so
a drafter fit to 128 public prompts degrades on the private held-out set. A drafter trained on
a superset of both distributions should generalize to both at parity. On the current frontier,
this does not raise TPS on the public bench — it stabilizes TPS on the private set, which is
the mechanism that converts invalid verdicts to valid ones at the same raw TPS level.

**Expected TPS effect.** No gain on public bench. The private-set Δ should shrink from 4–9%
toward the 0.7% across-bucket baseline, converting borderline-invalid runs to valid. Net effect
on the leaderboard: a robust 415–420 TPS valid entry instead of a probabilistic invalid one.

**PPL / greedy-identity risk.** Low. The drafter change only affects draft acceptance statistics.
The verifier target model does the final greedy selection; PPL and greedy identity are functions
of the target, not the drafter. PPL invariant, greedy identity invariant.

**Feasibility on vLLM 0.22 + A10G sm_86.** High. No new architecture; existing MTP drafter
training pipeline. kl_distill_reference_itaca/corpus_spec.md describes the exact recipe
(≥9k prompts, per-prompt trace capture: prefix token IDs + top-2048 softmax-applied logits +
drafter argmax, ~1.08M records). The paxenos-gemma-2 `kltrace-v0` format is already correct;
only the corpus needs widening. Training is tiny-model (hidden=256, 4 layers) — fits on any GPU.

**Concrete first local validation step.** Build the corpus dedup check: hash first 512 tokens of
every public bench prompt, confirm zero overlap with any of the ≥9k training prompts using the
bigram-prefix filter described in corpus_spec.md. Then run `analyze_drafting_headroom.py` on a
held-out shard (not the 128 bench prompts) to confirm tok/fwd >= 3.5 before submitting.

**Key recipe detail.** Hybrid loss `L = α·CE(target_argmax) + (1-α)·KL(top-k softmax)` with
α∈[0.3,0.5]. Use top-2048 (covers >99.9% of mass on PCK04-pruned ~16k vocab). Stratified
10% held-out shard. Do not seed from eval_prompts_sharegpt.json. Trace schema:
`train_kl_drafter.py:TRACE_SCHEMA` (kltrace-v0 format).

**Stop condition.** If held-out shard tok/fwd is <3.0 after ≥4 epochs over the full ≥9k corpus,
the corpus mix is miscalibrated — re-examine the distribution weights.

**Taste scores.** Mechanistic grounding: 4. Research-state value: 4. Execution value: 4.
Mode: diagnostic (separates prompt-distribution cause of invalid verdicts from engine noise).

---

### Rank 2 — PARD-2 Parallel Drafting with CAT Optimization

**What it is.** Replace the sequential MTP drafter with a PARD-2-style parallel drafter
(arXiv 2605.08632) that (a) predicts all K draft tokens simultaneously from a shared hidden
state (no autoregressive chain bottleneck), and (b) uses Confidence-Adaptive Token (CAT)
optimization to upweight training on tokens likely to be accepted. Replaces the per-position
serial structure of MTP decoding.

**Why it might help.** The current MTP drafter has measured per-position acceptance decay:
0.69/0.53/0.43/0.34/0.27/0.22/0.17 at positions 1–7. K saturates at ~6 because deep-chain
positions have acceptance near random. PARD-2's parallel mode eliminates the autoregressive
chain bottleneck: all K tokens are drafted from a single forward pass of the drafter. CAT
additionally focuses training signal on positions where acceptance probability is non-trivial,
so the learned distribution is more calibrated at each depth. Measured result (PARD-2 paper):
6.94× speedup on LLaMA 3.1-8B, 1.9× over EAGLE-3, 1.3× over PARD-1. If parallel drafting
lifts Gemma 4 E4B accepted-tokens/step from ~3.55 (MTP K=6 effective) toward 5–6, and if the
per-step overhead of the drafter stays comparable to MTP, the TPS ceiling rises substantially.
Projected range: 450–520 TPS (speculative; depends on Gemma 4 architecture adaptability).

**PPL / greedy-identity risk.** Medium. Parallel drafting uses the same target-model verification
loop as MTP — the target model still does final greedy selection. PPL is invariant by
construction. Greedy identity risk: only if the spec-decode acceptance logic in vLLM is modified
incorrectly. The safe path is to use vLLM's existing `parallel_drafting: true` flag
(already wired for P-EAGLE), verify with check_greedy_identity.py on 128 bench prompts before
any submission.

**Feasibility on vLLM 0.22 + A10G sm_86.** Medium-high. vLLM nightly already has
`parallel_drafting: true` for EAGLE (P-EAGLE path). The PARD-2 architecture (parallel forward
from shared mask token, COD training) needs a new drafter checkpoint; the inference path
(parallel → verify) is already available. Main risk: PARD-2's dual-mode (target-dependent +
target-independent) — the target-independent variant is the one to use (avoids coupling to
target features; simpler serve path). Training: TRL framework + PARD-2's COD schedule
(r=0.7, r_min=0.2, γ=max(r^(k-1), r_min)), 4 epochs, drafter hidden=256 (match existing).

**Concrete first local validation step.** Overfit a PARD-2 drafter on the 128 bench prompts
(intentionally — this is a feasibility check, not the final training run). Verify that parallel
acceptance at K=6 exceeds the current MTP acceptance at K=6 (current: Σ_k α_k ≈ 3.55/step).
If overfit-bench acceptance is <3.8, the architecture is not adapting to Gemma 4 — investigate
conditioning. Use analyze_drafting_headroom.py to replay the policy against decode_outputs.jsonl.

**Key recipe detail.** PARD-2 code at AMD-AGI/PARD. COD r=0.7. Shared mask token ID must be
consistent between training and serve config. Use `--speculative-config
'{"model":"<pard2-draft>","num_speculative_tokens":8,"parallel_drafting":true}'` in vLLM serve.

**Stop condition.** If parallel acceptance (tok/fwd) is <=3.0 on an overfit run, the Gemma 4
architecture is incompatible with the current drafter design — re-examine shared hidden state
dimensionality or switch to EAGLE-3 as the base.

**Taste scores.** Mechanistic grounding: 3. Research-state value: 4. Execution value: 3.
Mode: tier shift (new drafter architecture with demonstrated 1.9× over EAGLE-3 externally).

---

### Rank 3 — EAGLE-3 Drafter Architecture

**What it is.** Train a new drafter using the EAGLE-3 architecture (arXiv 2503.01840): direct
token prediction (not feature prediction), multi-layer feature fusion via "training-time test"
(uses features from multiple target layers rather than the last hidden state alone).

**Why it might help.** EAGLE-2/DFlash-style drafters predict the next token from the target's
last hidden state. EAGLE-3 instead fuses features from multiple layers and trains with a
"training-time test" that dynamically selects which layers are most informative per position.
Measured: 6.5× speedup over AR baseline, ~1.4× over EAGLE-2 equivalents. Unlike EAGLE-2,
EAGLE-3's acceptance scales with training data size — it is not architecturally capped at the
feature-prediction ceiling. If EAGLE-3 lifts tok/fwd from ~3.55 toward 5, the TPS ceiling
rises to the 480–550 range (rough projection; A10G int4 bandwidth limits apply independently).

**PPL / greedy-identity risk.** Low-medium. Same verification loop as MTP; PPL invariant.
Greedy identity risk is the same as any spec-decode drafter: correctly implemented, zero risk;
incorrectly wired acceptance logic, full divergence. Run check_greedy_identity.py.

**Feasibility on vLLM 0.22 + A10G sm_86.** Medium. EAGLE-3 code is at SafeAILab/EAGLE.
vLLM supports EAGLE-style inference via the `eagle` speculative method. The key implementation
question: does Gemma 4 E4B expose its intermediate layer features in the vLLM forward pass for
multi-layer fusion? This needs to be verified before training. If the feature extraction hook
is absent, EAGLE-3 degrades to EAGLE-2 (single last-hidden-state). Check `model_runner.py`
in the vLLM nightly for Gemma 4 feature export.

**Concrete first local validation step.** Clone SafeAILab/EAGLE, run the data collection step
on 512 prompts from the four target distributions (ShareGPT, MMLU-Pro, GPQA, MATH) using
the Gemma 4 E4B int4 target. Confirm multi-layer feature tensors are capturable. If yes,
train for 1 epoch and check acceptance > 0.70 at position 1 (current MTP: 0.69). If multi-layer
features are not available, fall back to EAGLE-2 (last hidden state only).

**Key recipe detail.** EAGLE-3 "training-time test" uses K=1 verification forward passes during
training to identify informative layers. This roughly triples training time per step vs EAGLE-2.
Budget accordingly on the training GPU. Inference serve: `--speculative-model <eagle3-draft>
--num-speculative-tokens 6` (start at K=6 matching current MTP optimum; adjust if acceptance
plateau shifts).

**Stop condition.** If multi-layer features are not exportable from vLLM's Gemma 4 forward pass
and EAGLE-3 degrades to EAGLE-2, this becomes a Rank 5 idea (EAGLE-2 has already been explored
in the community). In that case, pivot to PARD-2 (Rank 2).

**Taste scores.** Mechanistic grounding: 3. Research-state value: 3. Execution value: 3.
Mode: tier shift (new drafter architecture; 1.4× over EAGLE-2 with strong external evidence).

---

### Rank 4 — P-EAGLE Parallel EAGLE (Already in vLLM, Zero Training Required)

**What it is.** Enable P-EAGLE (arXiv 2602.01469) using the existing drafter checkpoint and
vLLM's already-wired `parallel_drafting: true` flag. P-EAGLE replaces the autoregressive draft
chain with simultaneous prediction from a learnable shared hidden state with per-position
attention masks. The key advantage over Rank 2/3: no new drafter training is needed for the
initial test — just a serve-config change.

**Why it might help.** P-EAGLE measures 1.10–1.36× over autoregressive EAGLE-3 in the paper.
On the current frontier, the MTP drafter's autoregressive chain is the acceptance bottleneck
(positions 3–7 have 43% → 17% acceptance). If parallel drafting breaks the chain bottleneck and
each position is predicted more accurately, the effective tok/fwd should rise. The mechanism is
different from MTP: P-EAGLE uses a shared hidden state with custom attention masks, so the
drafter's forward cost is slightly different — important to benchmark on A10G.

**PPL / greedy-identity risk.** Low. Same target-model verification loop; PPL invariant.
Run check_greedy_identity.py before any submission.

**Feasibility on vLLM 0.22 + A10G sm_86.** High for the initial no-training test. vLLM nightly
already implements `parallel_drafting: true` for the EAGLE inference path. First test: serve
the existing drafter with parallel_drafting enabled; measure tok/fwd on 128 bench prompts. Cost:
one local test run. If tok/fwd improves with the existing checkpoint, train a dedicated P-EAGLE
checkpoint for full gains. If the existing checkpoint does not gain with parallel mode (because
it was trained for sequential drafting), training is required (Rank 2 path).

**Concrete first local validation step.** Modify serve.py to add
`"parallel_drafting": true` to the speculative-config. Run the harness on 128 bench prompts.
Check tok/fwd using analyze_drafting_headroom.py replayed against decode_outputs.jsonl.
If tok/fwd >= 3.8 (vs current 3.55), proceed to a full HF Job. Cost: one local startup check.

**Key recipe detail.** P-EAGLE code: arXiv 2602.01469; vLLM serve flag already present.
The paper reports that P-EAGLE benefits most when trained with its parallel objective (shared
hidden state + attention mask pre-computation). Using a sequentially trained checkpoint in
parallel-decode mode is a zero-cost probe but unlikely to hit the full 1.10–1.36× gain.

**Stop condition.** If tok/fwd with parallel_drafting=true on the existing checkpoint is <=3.55
(no improvement), the sequential checkpoint is not compatible with parallel mode — pivot to
training a P-EAGLE checkpoint (Rank 2 path). Cost of this diagnostic: negligible.

**Taste scores.** Mechanistic grounding: 3. Research-state value: 3. Execution value: 4.
Mode: diagnostic (zero-cost probe of a mechanism already wired into the stack).

---

### Rank 5 — GPU-Side In-Graph Suffix/Megakernel Hybrid

**What it is.** Implement suffix matching (SuffixDecoding / SAM-Decoding) entirely on-device
within the captured CUDA graph, so the suffix lookup and the MTP proposal happen in the same
graph step without any host round-trip. The key differentiation from the ruled-out host-side
PLD hybrid: the sync scheduling tax (~+50 TPS async worth) is avoided because the suffix match
never leaves the GPU.

**Why it might help.** Analysis from gemma_specdecode_headroom_flowian/README.md shows that
3.6–3.9% of all generated tokens on the public bench sit in verbatim runs longer than MTP's
K=8 reach (426 such runs at n=2 suffix, max run=24 tokens). On those runs, a suffix proposal
of depth > K would accept all tokens without any drafter forward passes. If this 3.6–3.9% of
tokens can be served at near-zero drafter cost (suffix match is O(1) per step via suffix
automaton — SAM-Decoding arXiv 2411.10666), the effective tok/fwd rises on the subset of
prompts with long repetitive structure. SAM-Decoding measured 3.28–11.13% additional speedup
over EAGLE-2 when integrated in parallel. On the current ~420 TPS baseline, a 3–5% gain
would be 12–21 TPS — bringing the frontier toward 435–441 TPS.

**PPL / greedy-identity risk.** Medium-high. The suffix proposal must exactly match what
autoregressive greedy decode would produce — no approximation. For verbatim repetitions from
the prompt or previous output, this is guaranteed if the suffix lookup is over token IDs (not
text). Greedy identity risk is real if the suffix index contains any approximation, sampling,
or off-by-one in the boundary handling. check_greedy_identity.py is mandatory before any
submission.

**Feasibility on vLLM 0.22 + A10G sm_86.** Low-medium. In-graph suffix match on the device
requires a custom CUDA kernel or a Triton kernel that can be captured inside torch.compile /
CUDA graph. SuffixDecoding (arXiv 2411.04975) and SAM-Decoding (arXiv 2411.10666) are
implemented as CPU-side lookups in their reference code; the GPU-side port is the custom
engineering work. This is explicitly identified in the bucket docs as "hayai/chiku's
chain-collapse direction" — meaning it is a known-open but heavy-engineering lane. Expected
effort: 2–4 days of CUDA/Triton work before the first measurable result.

**Concrete first local validation step.** Implement a CPU-side suffix automaton (SAM-Decoding
reference code) on top of the existing decode_outputs.jsonl captures. Replay it offline with
analyze_drafting_headroom.py (--n 2 --depth 24 --mtp-cap 8) to confirm the theoretical +3.6–
3.9% token budget exists on the actual benchmark prompts. If the budget is confirmed, prototype
a Triton kernel for O(1) suffix lookup against a per-sequence KV-cache of token IDs, stub it
into the vLLM forward loop, and benchmark tok/fwd before committing to a full CUDA graph
capture implementation.

**Key recipe detail.** SAM-Decoding paper (arXiv 2411.10666): O(1) per-step suffix match via
suffix automaton; reference code available; 18%+ faster than retrieval-based competitors;
3.28–11.13% additional speedup over EAGLE-2 in hybrid mode. SuffixDecoding paper
(arXiv 2411.04975): suffix tree approach; adaptive speculation depth; better on agentic/
repetitive workloads. SAM-Decoding is preferred for the in-graph path because the O(1)
per-step guarantee is essential for CUDA graph compatibility.

**Stop condition.** If the offline token-budget analysis shows <2% of tokens in verbatim runs
>K=8 on the actual 128 bench prompts (below what was measured in flowian's headroom analysis),
the mechanism does not apply and this idea should be deprioritized. If the Triton kernel
prototype cannot be captured in a CUDA graph without breaking the onegraph optimization, the
implementation is infeasible without a major refactor.

**Taste scores.** Mechanistic grounding: 3. Research-state value: 3. Execution value: 2.
Mode: tier shift (custom kernel work; high ceiling if the engineering is solved; high effort).

---

## Experiment Decision Tree

```
Start: current top VALID ~420 TPS (osoi5 substrate, PPL ~2.377)

Primary bottleneck: private-set TPS reproducibility (4–9% Δ) AND raw TPS ceiling

Branch A — Fix reproducibility first (Rank 1)
  [Rank 1] KL-distilled ≥9k-corpus drafter
    → success (private Δ <2%): proceed to Rank 2/3 for raw TPS lift
    → failure (private Δ still >4%): investigate corpus dedup and distribution mix;
       check that training corpus is not dominated by ShareGPT (50% cap)

Branch B — Lift raw TPS ceiling (Ranks 2-5, can parallel-run with Branch A)
  [Rank 4] P-EAGLE zero-cost probe (parallel_drafting=true, existing drafter)
    → tok/fwd >= 3.8: proceed to a full HF Job; expected 10–20 TPS gain
    → tok/fwd <= 3.55 (no gain with sequential checkpoint):
      → [Rank 2] PARD-2: train parallel drafter, COD schedule, K=8
          → overfit acceptance >3.8: train on full corpus
          → overfit acceptance <=3.0: architecture incompatibility, pivot to Rank 3
      → [Rank 3] EAGLE-3: verify multi-layer feature export from vLLM
          → features available: train EAGLE-3 drafter
          → features unavailable: EAGLE-3 degrades to EAGLE-2 (deprioritize)

Branch C — Long-tail token budget (Rank 5, parallel or after Branches A/B)
  [Rank 5] GPU-side SAM-Decoding
    → offline token-budget confirms >3.6%: prototype Triton kernel
    → offline token-budget <2%: deprioritize this direction
```

---

## Research State Update

**Current best explanation.** Two independent bottlenecks:

1. Reproducibility: the drafter is overfit to 128 public bench prompts; prompt-distribution
   shift causes 4–9% TPS loss on the verifier's private set. This is the proximate cause of
   all invalid verdicts at the frontier.

2. Raw TPS ceiling: the sequential MTP drafter chain saturates at K=6 effective (acceptance
   decay 0.69→0.17 per position). Parallel drafting (PARD-2, P-EAGLE) or a better-trained
   sequential drafter (EAGLE-3) could lift tok/fwd from 3.55 toward 5–6, raising the TPS
   ceiling from ~420 toward 480–550.

**Ruled-out paths.** All sub-4-bit quantization, fp8 KV, attention backend swap, GEMM swap,
MTP K>7, host-side PLD hybrid, pure n-gram, body channel-wise quant, DFlash drafter (2–5%
acceptance), all runtime knobs.

**Open uncertainties.**
1. Can PARD-2 / P-EAGLE achieve parallel acceptance > sequential MTP on Gemma 4 E4B's
   specific architecture? (Needs a no-training probe with P-EAGLE first.)
2. Is EAGLE-3's multi-layer feature extraction compatible with vLLM's Gemma 4 forward pass?
3. Does the KL-distilled wide-corpus drafter close the private-set gap while maintaining
   public-bench tok/fwd >=3.5?

**Next discriminating experiment.** Rank 4 (P-EAGLE zero-cost probe): modify serve.py to add
`parallel_drafting: true`, run harness on 128 bench prompts, check tok/fwd. Cost: one local
run. Result either opens the parallel-drafting lane (proceed to Rank 2 training) or confirms
the existing sequential checkpoint is incompatible (proceed to training a new architecture).

**Stop condition for this round.** If all five ranked ideas are exhausted without a valid
leaderboard entry above 425 TPS, return to first principles: profile the per-step breakdown
on the A10G (weight GEMM vs sampling vs attention vs norm overhead) and look for any
non-quantization overhead that has grown with the current stack (e.g., centroid mask
overhead, PCK04 vocab-prune sampling cost).
