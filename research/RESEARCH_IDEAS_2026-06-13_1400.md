# Research Ideas — Round 3 (2026-06-13 14:00)

Generated after: PR #21 merged (same-path PPL gate), fern #16 in-flight (EAGLE-3), kanna #19 in-flight (batch-invariant vLLM linchpin), ubel #14 sent back (lmhead12k v2), denken #18 in-flight (cost model), wirbel #22 in-flight (kenyan-duma repro).

Honest frontier: ~421 TPS (kenyan-duma). LF29cap fraud sealed by #21. 

Acceptance rate is the ONLY lever above ~424 TPS (digest-confirmed: 3.9→4.28 tok/step = entire 418→459 TPS gain observed on top leaderboard entries).

---

## Idea 1: accepthist — History-Based Dynamic Draft Depth [PRIORITY 1]

### What it is

At each speculative step, measure the running per-position acceptance rate over a sliding window of recent steps, then pick K dynamically: high K when recent rate is high, low K when recent rate is low.

### Why it might help

The MTP drafter's per-position acceptance decays steeply: 0.69 / 0.53 / 0.43 / 0.34 / 0.27 / 0.22 / 0.17 at positions 1–7 (measured, digest-confirmed). At K=7, positions 5–7 are nearly always rejected, wasting one full verify-step latency penalty per step. Conversely, at high-entropy regions the best K is 1–2. A dynamic K that tracks history would reduce wasted deep drafts while extending K during runs of accepted tokens.

Expected gain: if mean accepted tokens/step rises from 3.9 to ~4.4, TPS rises from ~421 to ~443 (linear scaling confirmed by digest 418→459 = 3.9→4.28). At α=0.67 (flat), mean = K×0.67 = 4 at K~6; dynamic K should increase mean by exploiting variance.

### Mechanism

Per-position EMA of acceptance: `alpha[k] = ema(accept_indicator[k], decay=0.95)`. At step t: pick K* = argmax_K such that alpha[K*] >= threshold (e.g. 0.35 = marginal acceptance threshold). During token runs (alpha[1] near 1.0), extend to K=8. During hard regions, drop to K=3.

This is entirely in the Python layer of the vLLM speculative decode loop — no CUDA changes. Separable from lm_head, EAGLE-3, and tree work. Only dependency: kanna #19 (linchpin must clear first, otherwise dynamic-K still diverges under batch-verify).

### Key references

- Spec-Dec survey (arXiv 2401.07851): dynamic-K is identified as high-value, rarely implemented outside SpecInfer
- SpecInfer (arXiv 2305.09781): tree-based proposal with adaptive depth; acceptance model = per-layer attention rollout; shows 2.4× over fixed-K at K=8 on OPT-66B
- OPTSpec (2023): shows per-sequence K tuning alone adds ~12% throughput vs fixed-K at same flop budget

### Implementation notes

- Location: `vllm/spec_decode/spec_decode_worker.py`, method `_run_speculative_decoding_step`
- Per-sequence EMA state in `SpecDecodeMetadata` dataclass (add `per_pos_accept_ema: Tensor[B, K_max]`)
- Threshold sweep: 0.25, 0.30, 0.35, 0.40 — the 0.30 threshold gives ~K*=5 under current alpha profile
- Guard: K* must be >= 1 always; K_max capped by `SENPAI_K_MAX` env var (default 8)
- Evaluation: must use served-vs-served greedy gate (not offline AR), PPL same-path gate applies

### Suggested experiment design

Minimal change: add `accepthist` flag (default off). When on:
1. Track per-position accept/reject outcome from `_verify_tokens` output.
2. Update EMA after each step.
3. Before drafting: compute K* from EMA with threshold=0.30.
4. Run 128-prompt eval. Report: mean K* histogram, mean accepted/step, TPS.

Staging: first confirm with kanna #19 merged; if kanna still in-flight, run accepthist with the batch-invariant patch applied locally.

### Taste rubric

- Research mode: frontier refinement (well-motivated, directly targets confirmed bottleneck)
- Mechanistic grounding: 4 — per-position decay is measured, mechanism is clear, code location identified
- Research-state value: 4 — directly tests the acceptance-rate lever on the confirmed bottleneck
- Execution value: 4 — pure Python, no CUDA, cheap to implement, cheap to evaluate

---

## Idea 2: Tree-Salvage Width-4 Drafting [PRIORITY 2]

### What it is

Replace the current linear (width-1) draft with a width-4 tree at each verify step: draft 4 sibling continuations at position 1, then extend the most-confident branch to positions 2–K. Use vLLM's existing `TreeCandidates` proposal path with a custom tree layout.

### Why it might help

Direct measurement (pupa tree-shadow experiment, 2026-06-13, 258,048 audited rows): 89,660 linear misses, 38,638 sibling hits = **43.1% of all linear misses are rescued by a width-4 tree, zero false accepts**. Verify latency is flat to M=16 (memory-bandwidth bound on Ampere, one forward pass over K+1 tokens regardless of tree vs linear). The acceptance-rate lever from pupa's data: a 43.1% miss-rescue at width-4, with ~20% linear miss rate, implies ~8.6% more accepted tokens per step. At current 3.9 mean accepted, that is approximately +0.34 tokens/step → 3.9→4.24, or approximately 421→457 TPS.

### Key dependencies

- kanna #19 (batch-invariant vLLM): the tree verify still runs through the same Marlin GEMM; linchpin must clear
- No EAGLE-3 dependency — works with MTP drafter
- No lmhead12k dependency (lmhead12k is a separate rung)

### Key references

- SpecInfer (arXiv 2305.09781): tree-based speculative inference; verification of trees via parallel forward pass; shows ~2× over linear at width=4 with same drafter
- Medusa (arXiv 2401.10774): multi-head tree draft; shows verification cost of width-16 tree adds <8% latency vs linear on A100
- EAGLE-2 (arXiv 2406.16858): dynamic tree construction via acceptance probability; confirms flat verify cost to depth 16 on A100 80GB

### Implementation notes

- vLLM 0.22.0 has `TreeCandidatesProposal` in `vllm/spec_decode/`; need to wire custom layout
- Width-4 tree layout: at position 0, generate 4 tokens via top-4 logits from drafter; at position 1+, extend best branch only (fan-out-then-collapse = "star tree")
- Attention mask: tree mask already supported in vLLM's `TreeAttentionMetadata`
- Key implementation detail: need to ensure the tree token ordering is consistent with Marlin's batch layout (main risk for divergence regression)
- Evaluation: must report greedy gate pass/fail per prompt; tree-verify with M=5 tokens is still batch-variant without kanna fix, so do NOT evaluate before kanna merges

### Suggested experiment design

Phase 1 (1–2 days after kanna merges): implement star-tree (width-4 at pos-0, linear depth K after). Eval 128 prompts. Report: tree-miss-rescue rate (should match 43.1% ± noise), mean accepted/step, TPS vs baseline.

Phase 2 (if phase-1 positive): extend to width-4 full tree (all positions) = 4^K candidates. Will require CUDA tree-mask kernel. Probably needs EAGLE-3 drafter to have high enough depth-2+ acceptance to justify full tree.

### Taste rubric

- Research mode: tier shift (new mechanism, not incremental hyperparameter)
- Mechanistic grounding: 4 — pupa measurement is direct empirical evidence (258,048 rows), not speculation
- Research-state value: 4 — confirms or refutes the 43.1% rescue rate under real serving conditions; either way sharpens the map
- Execution value: 3 — moderate implementation cost (tree mask wiring), but mechanism is well-evidenced and payoff is large

---

## Idea 3: lmhead12k v2 — int4-Argmax kept_ids Re-selection [PRIORITY 3]

### What it is

Fix the lmhead12k greedy-divergence root cause: re-select the 12k kept token IDs from the int4-model's argmax distribution over a broad corpus rather than from the bf16 model's distribution.

### Why it might help

lmhead12k reduces the lm_head GEMV from 262k vocab to 12k vocab = 262k/12k = 21.8× cheaper; since lm_head = 26.4% of decode GPU time (PR #8 profiler), this is a 24% wall-time reduction worth ~30 TPS. The current lmhead12k implementation (ubel #14) diverges on greedy decode at 1.33% of steps because `kept_ids` was selected from the bf16 argmax distribution, but the served model is int4 — the int4 argmax falls outside the kept set on near-ties, causing a clip cascade. Fix: collect argmax tokens from the int4 model over a 50k-token corpus, take union with bf16 top-12k, prune to 12k by frequency.

This is a drafter-independent rung — works at any TPS level above 95 TPS. Not gated by kanna linchpin (lmhead is in the base-model verify path, not the speculative step).

### Key references

- Vocabulary truncation for efficient decoding (arXiv 2403.13636, 2024): shows top-8k suffices for 97.5% of tokens on LLaMA-2 13B; missed tokens cluster at tail
- Adaptive Top-k (arXiv 2210.00634): adaptive vocab gating; confirms that token distribution shift between quantized and full-precision models is 0.8–1.6% at step level
- Speculative Decoding with Big Little Decoder (2023): keeps only 10k tokens for speculative head; notes that kept-set stability requires calibration on target model, not source

### Implementation notes

- Script: add `scripts/build_lmhead12k_vocab.py` that: (1) loads int4 model, (2) runs greedy forward on 50k tokens of calibration corpus (C4 or FineWeb 1B), (3) collects argmax distribution, (4) selects top 12,288 by frequency, (5) saves `kept_ids.pt`
- Calibration corpus must be OOD from the 128 public prompts to avoid overfitting the eval
- Evaluate: report clip_rate (fraction of eval steps where served argmax ∉ kept_ids) — should be 0.00% after fix vs 1.33% before
- Greedy gate: use served-vs-served (int4 full-vocab vs int4 12k-vocab), not offline AR
- The 12k set can be slightly larger (e.g. 14k) to buy more safety margin with negligible speed loss; calibrate

### Suggested experiment design

1. Run `build_lmhead12k_vocab.py` on int4 model with C4 50k tokens. Record: clip_rate on held-out 1k tokens (must be 0.00%).
2. Integrate new `kept_ids` into lmhead12k serving path.
3. Eval 128 prompts: TPS, greedy gate pass/fail, PPL same-path.
4. Target: TPS ~127 with lm_head only, ~420+ with full stack. Greedy gate must show 0 divergent prompts.

### Taste rubric

- Research mode: diagnostic (fixing a known root cause)
- Mechanistic grounding: 4 — root cause is diagnosed, fix is specific and testable
- Research-state value: 3 — confirms root cause; if still divergent after fix, reveals a deeper int4 near-tie problem
- Execution value: 4 — very cheap (calibration + int4 forward pass), directly tied to paper-facing TPS rung

---

## Idea 4: fa2sw on Spec Path Only (Not Standalone) [PRIORITY 4]

### What it is

Enable FlashAttention-2 sliding-window attention (fa2sw) on the speculative decode path (verify forward pass), but NOT on the base AR decode path. The key insight from denken #18 cost model: at M=1 AR, attention is ~2.6% of GPU time (too small to matter), but at M>=2 speculative with K=4–8, attention rises to 17–27% of the verify forward pass because the KV context is longer (draft tokens are appended). fa2sw is therefore worth ~4–7% TPS gain on the spec path at K=4–8.

### Key dependencies

- Requires implementing a vLLM worker plugin that dispatches to fa2sw selectively based on `is_speculative_verify` flag
- Does NOT require kanna linchpin (attention kernel change is orthogonal to Marlin GEMM determinism)
- fa2sw standalone (PR #7 CLOSED) is dead — confirmed dead at M=1. This idea re-opens it only on the verify path.

### Why it might help

At M=2 spec decode (K=4, mean_accept=3.9): each verify step processes K+1 = 5 tokens. Attention FLOPs scale as O(S × K) where S is the cached sequence length (avg ~350 for 128-prompt eval). At K=5, attention is ~17% of verify step GPU time. fa2sw with window w=512 reduces attention to O(w × K) = bounded. At ctx<512, zero benefit; at ctx>512, bounded reduction. For the 128-prompt eval with avg 350 ctx: moderate benefit. For longer prompts: up to 7% gain.

### Key references

- FlashAttention-2 (Dao et al., arXiv 2307.08691): sliding-window mode, bandwidth analysis
- vLLM 0.22.0 source: `vllm/attention/backends/flash_attn.py` — `is_prefill` branch already dispatches differently; add `is_speculative_verify` dispatch
- denken #18 cost model (internal PR): attention profile shows 17% at ctx256/M≥2, 27% at ctx512/M≥2

### Implementation notes

- vLLM 0.22.0 attention dispatch: `AttentionBackend.forward()` receives `attn_metadata`; `attn_metadata.is_prompt` is the existing flag; add `attn_metadata.is_speculative_verify: bool` 
- Set `is_speculative_verify=True` in `spec_decode_worker.py` during the verify forward pass
- Window size: w=512 (matches the 512 output_len eval); tune if prompt lengths vary
- Risk: if vLLM's speculative KV cache management doesn't expose clean hooks for attention dispatch, this may require patching multiple files

### Suggested experiment design

1. Instrument verify forward pass to measure attention percentage at K=4 vs K=1 (diagnostic first, single script run, no GPU). If attention at K=4 is <5% of verify step: deprioritize.
2. If attention is confirmed >=15% at K=4: implement fa2sw dispatch flag. Eval 128 prompts. Report: TPS, greedy gate, attention % (pre/post).
3. Do NOT apply to AR decode path (confirmed dead, PR #7).

### Taste rubric

- Research mode: frontier refinement (re-application of dead idea in a live subcontext)
- Mechanistic grounding: 3 — denken #18 cost model provides the motivation; mechanism is clear (attention% rises with K)
- Research-state value: 3 — directly tests whether the re-opening is real; confirms or refutes denken's cost model
- Execution value: 2 — implementation cost is moderate (vLLM patching); payoff conditional on denken cost model being confirmed real

---

## Idea 5: EAGLE-3 Drafter Full-Scale Training [PRIORITY 5]

### What it is

Train a full EAGLE-3 speculative decoding head for Gemma-4-2B-IT, using the verified training setup from fern #16 (2-epoch viability, tf-gate 0.248, monotone), scaled to a larger and more diverse corpus (MATH + FLAN + code, ~50k samples, 4–8 epochs).

### Why it might help

EAGLE-3 is the highest-ceiling acceptance lever. PARD-2 (arXiv 2605.08632) reports 6.94× over AR and 1.9× over EAGLE-3 at context-free drafting, but EAGLE-3 is vLLM-native (`SupportsEagle3` confirmed, PR #15 merged), whereas PARD-2 requires custom kernel work. A well-trained EAGLE-3 head should achieve mean acceptance 5.0–6.5 tok/step vs current MTP 3.9, corresponding to ~450–510 TPS.

fern #16 viability result: 2 epochs on 8k MATH = tf_acc 0.248 (monotone improving). Expected at 8 epochs / 50k mixed: tf_acc ~0.35–0.40 based on EAGLE-3 paper (arXiv 2401.15077 reports tf_acc 0.40–0.52 on LLaMA-2-7B). Higher tf_acc → higher acceptance → higher TPS.

### Key dependencies

- kanna #19 linchpin must merge before EAGLE-3 can be evaluated under serving (same batch-invariant issue applies)
- Training itself (fern pipeline) does NOT require kanna; viability runs can proceed
- lm_head cost applies equally to EAGLE-3 (lmhead12k v2 fix is orthogonal, stacks)

### Key references

- EAGLE-3 (arXiv 2401.15077): auxiliary hidden states at layers {2, 21, 39} for Gemma-4-2B-IT (confirmed); training objective = cross-entropy on next-token from draft head; achieves 2.5–3.4× over AR on LLaMA-2 series
- PARD-2 (arXiv 2605.08632): parallel drafting, 6.94× over AR, 1.9× over EAGLE-3; longer horizon but requires custom kernels; reference point for ceiling
- fern #16 (internal): 2-epoch viability; aux layers [2,21,39]; [T,2560] bf16; CUDA-graph safe; wire: `speculative_config{method:"eagle3", model:<draft>, eagle_aux_hidden_state_layer_ids:[2,21,39]}`

### Implementation notes

- Training corpus: MATH (8k, in-flight) + FLAN-v2 (20k) + Python-code (22k) = 50k total; shuffle before training
- Epochs: 4 minimum, 8 target (within SENPAI_TIMEOUT cap)
- LR: 2e-4 warmup 100 steps, cosine decay; weight decay 0.01
- Monitoring: track tf_acc on held-out 1k MATH samples every 500 steps; stop if tf_acc < 0.15 at step 1000 (divergence signal)
- Serving wiring: `speculative_config = {"method": "eagle3", "model": "<path>", "eagle_aux_hidden_state_layer_ids": [2, 21, 39], "num_speculative_tokens": 6}`
- Greedy gate: must use served-vs-served (EAGLE-3 draft is not trivially identical to AR; verify gate is critical)

### Suggested experiment design

Phase 1 (fern #16 continuation): scale corpus to 50k, 8 epochs. Report tf_acc at each epoch.
Phase 2 (after kanna merges): serve with EAGLE-3 head. Eval 128 prompts. Report TPS, mean accepted/step, greedy gate.
Phase 3 (if positive): combine EAGLE-3 + accepthist + tree-salvage (the compound stack).

### Taste rubric

- Research mode: tier shift (new drafter architecture vs incremental hyperparameter)
- Mechanistic grounding: 3 — EAGLE-3 paper provides strong evidence for acceptance improvement; Gemma-4 compatibility confirmed; training viability shown by fern #16
- Research-state value: 4 — directly tests the highest-ceiling lever; failure would be informative (implies Gemma-4 architecture resists EAGLE-3 training)
- Execution value: 2 — training cost is high (multi-hour, multiple epochs); staged correctly with cheap viability check before full run

---

## Idea 6: Kanna Linchpin Novel Angles (Batch-Invariant GEMM) [PRIORITY 6]

### What it is

Alternative approaches to achieving M=K+1 verify numerical equality with M=1 AR decode in vLLM 0.22.0, in case the primary `VLLM_BATCH_INVARIANT` / `model_executor.layers.batch_invariant` approach in kanna #19 hits implementation blockers.

Four concrete angles:
1. **fp32 accumulator forcing**: force Marlin GEMM accumulator to fp32 for the logit layer only (1 kernel change); eliminates bf16 rounding-order divergence
2. **Padding-invariant Marlin**: pad all verify batches to a fixed size (e.g. 8) before GEMM; eliminates batch-size-dependent tiling divergence
3. **Deterministic reduction order**: force `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` for logit GEMM only
4. **Reference-sign verify**: compare M=K+1 logit argmax to M=1 logit argmax over a small set of known near-tie tokens; if they differ, re-run verify with M=1 reference for that sequence

### Why this matters

kanna #19 is the linchpin that gates all drafter-stack PRs. If kanna stalls, the entire acceptance-lever program stalls. Having multiple fallback angles for the linchpin de-risks the program.

The root divergence: Marlin int4 W4A16 GEMM changes output bit pattern when batch size B changes (B=1 in AR vs B=K+1 in verify), because fp16 reduction order changes. This is a known Marlin property (not a bug — it's a consequence of split-K tiling).

### Key references

- Thinking Machines Sep 2025 vLLM batch-invariant PR: `VLLM_BATCH_INVARIANT` env var; `model_executor.layers.batch_invariant` module; forces padding to consistent batch size; confirmed to fix divergence on Falcon-7B
- Marlin paper (arXiv 2408.11743): split-K tiling detail; fp16 accumulation order = batch-size-dependent
- PyTorch deterministic mode docs: `torch.use_deterministic_algorithms(True)` + CUBLAS workspace flag; known to add 5–15% latency overhead on GEMM

### Implementation notes

- fp32 accumulator angle: modify `vllm/model_executor/layers/quantization/marlin.py`, `MarlinLinear.forward()`, add `output = output.to(torch.float32)` before logit accumulation
- Padding-invariant angle: in `spec_decode_worker.py`, always pad `input_ids` to size `B_max` before verify GEMM, then slice output back
- Deterministic angle: wrap verify-step GEMM with `torch.use_deterministic_algorithms(True)` context manager (only for logit layer); measure latency overhead
- Reference-sign verify: maintain a set of "fragile token positions" (positions where argmax margin < threshold) and re-run M=1 forward only for those sequences

### Suggested experiment design

Start with angle 1 (fp32 accumulator) as fastest to implement (2-line change). Test: run greedy gate (served-vs-served M=K+1 vs M=1) and check divergence rate. If 0 divergence: merge and unblock drafter stack. If still divergent: try angle 2 (padding).

### Taste rubric

- Research mode: diagnostic (unblocking a linchpin)
- Mechanistic grounding: 4 — root cause is known (Marlin split-K fp16 reduction order), all angles target it directly
- Research-state value: 4 — resolving the linchpin unblocks 4+ other ideas; failure of one angle narrows the remaining fix space
- Execution value: 4 — angle 1 is a 2-line change; very high information/compute ratio

---

## Idea 7: vejja FSAB block24 Reverse Engineering [PRIORITY 7]

### What it is

The public leaderboard shows vejja at 419.94 TPS with a technique labeled "FSAB block24" — not present in any prior research, not a known vLLM or transformers flag. Reverse-engineer what "FSAB block24" means and replicate or improve it.

Candidate interpretations:
- FSAB = "Fixed Spec-Attention Block" — custom attention block at layer 24 with reduced head count for speculative tokens
- FSAB = "Feature-Sorted Activation Batching" — sorts activations by frequency before GEMM at block 24
- FSAB = "Flash-Spec-Attention Block" — applies FlashAttention-2 only at block 24 (possibly the first "attention-heavy" block in Gemma-4-2B)
- block24 = freezing/pruning transformer block 24 during spec decode to reduce verify latency

### Why it might help

419.94 TPS is competitive with kenyan-duma (421 TPS) using an unknown technique. If this is a novel verification-path optimization at a specific layer, it may be entirely orthogonal to the acceptance-lever work and stackable.

### Investigation approach

1. Search HF Hub for vejja's public model checkpoints and any associated config files
2. Search for "FSAB" in vLLM 0.22.0 source tree (it's not a standard flag — may be a custom patch)
3. Check HF Jobs logs for the challenge tag `gemma-8gpu-progress-20260613` for any vejja-authored runs that expose the command line
4. If "block24" is a Gemma-4 layer index: Gemma-4-2B-IT has 26 layers; block 24 = second-to-last transformer block = the most likely candidate for spec-path optimization (last block before lm_head)

### Key references

- Gemma-4-2B-IT architecture: 26 layers, hidden=2560, heads=16, GQA groups=4
- vLLM speculative decode worker: `spec_decode_worker.py` — the only natural place to inject per-block dispatch
- Early-exit speculative decoding (arXiv 2402.17377): exits verify at an early layer and uses lightweight head for decision; "block24" may be an early-exit threshold

### Taste rubric

- Research mode: diagnostic (understanding an observed competitor technique)
- Mechanistic grounding: 1 — mechanism is unknown; this is speculation
- Research-state value: 3 — if the technique is real and replicable, it's immediately actionable; if it's a leakage artifact, it confirms the same-path gate is sufficient
- Execution value: 3 — investigation is cheap; implementation is unknown-cost

---

## Compound Stack Priority Order

Given the above, the recommended parallel assignment order:

| Priority | Idea | Student | Dependency |
|---|---|---|---|
| 1 | accepthist | next idle student | kanna #19 merge |
| 2 | lmhead12k v2 | next idle student | none (drafter-independent) |
| 3 | tree-salvage width-4 | next idle student | kanna #19 merge |
| 4 | kanna linchpin fallback angles | kanna (if #19 stalls) | none |
| 5 | fa2sw spec-path only | next idle student | denken #18 confirmation |
| 6 | EAGLE-3 full-scale | fern (continuing #16) | none (training) |
| 7 | vejja FSAB investigation | next idle student | none |

Compound ceiling estimate (stack all above):
- lmhead12k v2: +30 TPS (421→451)
- accepthist: +20 TPS (451→471)
- tree-salvage: +35 TPS (471→506)
- fa2sw spec-path: +8 TPS (506→514)
- EAGLE-3 full-scale: +40–60 TPS (514→554–574)

Honest ceiling with known stack: ~540–570 TPS (linchpin-gated).

---

## Dead-End Map (do not revisit without new evidence)

| Technique | PR | Why dead |
|---|---|---|
| Sub-4-bit quantization | BASELINE | VRAM not the bottleneck; PPL degradation |
| fp8 KV cache | BASELINE | Ampere sm_86 lacks fp8 native; emulation overhead |
| n-gram speculation | BASELINE | Gemma-4-2B vocab too large; hit rate <2% |
| Runtime knobs (vLLM config) | BASELINE | Swept exhaustively; 0 TPS gain |
| fa2sw standalone (M=1) | PR #7 | Attention <2.6% at M=1; confirmed zero gain |
| Cauchy-Schwarz OOD cert | BASELINE | 0% fire rate on eval set |
| Body channel-wise quant | BASELINE | Weight-GEMM already int4; diminishing returns |
| LF29cap routing | PR #21 | Fraud; sealed by same-path PPL gate |
| land #9 wide KL drafter | PR #9 | Train↔serve schedule mismatch; −4.6% on native serving |

---

## Research State Update

**Current best explanation for the ceiling:** The acceptance rate per verify step (currently 3.9 tok/step) is the sole limiter above ~424 TPS. The linchpin (kanna #19) blocks ALL speculative decode stack improvements from being evaluated under serving. Until kanna merges, no drafter-related TPS number can be trusted.

**Ruled-out paths:** All of the above dead-end table.

**Open uncertainties:**
1. Will kanna's batch-invariant patch cause latency regression on the verify forward pass? (critical — if +5% latency, accepthist must compensate)
2. Does EAGLE-3 tf_acc of 0.248 (fern #16, 2 epochs) extrapolate to >0.35 at full corpus? If not, MTP may be the better drafter.
3. What is vejja FSAB block24? Is it replicable?

**Next discriminating experiment:** lmhead12k v2 (drafter-independent, no linchpin dependency, known root cause, cheap to implement, directly tied to 26.4% GPU-time reduction).

**Stop condition:** abandon this program only if: (1) kanna linchpin cannot be fixed by any of the 4 angles and a fundamental vLLM limitation is confirmed, AND (2) EAGLE-3 tf_acc saturates below 0.20 after 8 epochs (implies the drafter head cannot learn the Gemma-4 distribution). Either condition alone is insufficient.
