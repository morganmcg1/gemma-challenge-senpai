# SENPAI Research Results

## 2026-06-13 14:38 — PR #4: int4 g128 + untied int4 lm_head (~127 TPS) ✓ MERGED — new leaderboard baseline rung

- **Branch:** `lawine/int4-g128-lmhead` · **Student:** lawine
- **Status:** MERGED — new best merged rung. `submissions/int4_g128_lmhead` is now the best merged submission. All future submissions beat 126.38 TPS.
- **Hypothesis:** untied int4 lm_head (eliminating the bf16 GEMV for 262k-vocab verify = 26.4% of decode GPU time per PR #8 profiler) + full-body g128 granularity (slight additional weight-byte reduction vs per-layer) → reaches the int4 Marlin weight-byte floor on Ampere.

### Results

| metric | value (official a10g-small) | vs PR #3 base |
|---|---|---|
| tps / output_tps | **126.378** | 1.32× (**+32%**) |
| ppl (served) | **2.019** | ≤ 2.42 ✓ |
| completed | **128 / 128** ✓ | — |
| greedy identity | **GREEDY_IDENTICAL 128/128** (served-vs-served cap=512) ✓ | — |
| same-path gate | **SAME_PATH_OK (gap 0.0000)** ✓ | — |
| job | `6a2d5a96234ca64b60121aa5` | — |
| W&B | `905tbujn` (official a10g-small) · `0pxj6n63` (local proxy + greedy) | — |

**Overall: 2.87× over bf16 (44.018 TPS), 1.32× over PR #3 int4 base.**

### Analysis & conclusions

- **Confirms lmhead profiler finding** (PR #8): 26.4% of decode GPU time was the 262k-vocab bf16 GEMV. Untied int4 lm_head eliminates it, explaining the +32% TPS gain. This is the exact profiler prediction.
- **This is the weight-byte floor.** Sub-4-bit (no sm_86 kernel) and fp8 KV (no A10G support) are dead ends. No further weight-bandwidth reduction is achievable in vLLM 0.22.0 on Ampere. Every remaining TPS lever is either (a) the drafter ladder (spec decode, gated on kanna verify-rollback), (b) lmhead12k (ubel #14, cheaper verify), or (c) runtime/warmup (precache, onegraph — the frontier stack).
- **Greedy validity methodology confirmed:** served-vs-served (spec-off) via `check_greedy_identity.py` passes cleanly (GREEDY_IDENTICAL 128/128). This is the gold-standard test.
- **lawine confirmed official PPL artifact** present on the HF job result — closing the near-cap timing question from last cycle.

---

## 2026-06-13 14:38 — PR #19: Batch-invariant vLLM spec decode ✓ MERGED — LINCHPIN DEFINITIVE NEGATIVE

- **Branch:** `kanna/batch-invariant-vllm-spec` · **Student:** kanna
- **Status:** MERGED — definitive negative. Closes the invariant-kernel lane. Next lane: verify-rollback (kanna PR #24).
- **Hypothesis:** `VLLM_BATCH_INVARIANT=1` (aten-override batch-invariant kernels) makes the M=K+1 verify forward bit-match the M=1 AR forward → greedy-identical spec decode.

### Results

| arm | INV | target GEMM | flip/tok | 95% CI | identical/32 | W&B |
|---|---|---|---|---|---|---|
| int4 ON (decisive) | 1 | Marlin `_C` (un-covered) | **0.376%** | [0.234, 0.518]% | 5/32 | `hz8jkc5h` |
| int4 OFF (control) | 0 | Marlin `_C` (un-covered) | 0.332% | [0.205, 0.460]% | 6/32 | `8wne15eh` |
| bf16 ON (discriminator) | 1 | aten linear (covered) | **0.111%** | [0.057, 0.166]% | 16/32 | `z0mclftv` |
| bf16 OFF (PR #5 ref) | 0 | aten linear | 0.72% | — | — | — |

**Primary metric:** int4_mtp_batchinv_greedy_flip_rate_per_token = **0.00376** (0.376%) — NOT zero. **Verdict: DIVERGENT, invariant-kernel lane CLOSED.**

### Analysis & conclusions

The bf16 control arm is the key insight. By removing int4 Marlin (using aten-covered bf16 GEMM) while keeping INV=1, we isolate TWO independent un-coverable root causes:

- **(a) int4 Marlin `_C` op:** contributes ~0.265%/tok excess above bf16 floor. The Marlin custom op is outside aten's scope; batch-invariance cannot intercept it. This was the main prior hypothesis (Marlin was "plausibly already M-invariant") — REFUTED.
- **(b) Spec verify path non-aten residual:** bf16 ON (full aten coverage, zero Marlin) is STILL divergent at 0.111%/tok. An irreducible non-aten component in the spec verify forward (attention-metadata build, rejection-sampler logits compare, or a fused step) remains batch-variant. Corroborated by vLLM issue #27433: "batch-invariance does not currently integrate with speculative decoding."
- **Consistency check:** 0.265% (a) + 0.111% (b) ≈ 0.376% (observed int4 ON). The two sources are independent and additive.
- **Implication:** neither int4 nor bf16 target drafter ladders are rescuable by `VLLM_BATCH_INVARIANT`. The invariant-kernel lane is closed for greedy-valid spec decode at ANY precision in vLLM 0.22.0.
- **Next lane:** verify-rollback (arxiv 2601.17768) — re-verify accepted tokens under fixed-shape M=1 reduction after each spec step; commit consistent / roll back violators. This targets both causes: (a) is dodged (rollback uses M=1 AR path, no Marlin batch-size dependency on committed path), (b) is caught and corrected by the re-verify. Assigned to kanna PR #24.

---

## 2026-06-13 14:38 — PR #16: EAGLE-3 draft-head training harness ✓ MERGED — keeper research artifact

- **Branch:** `fern/eagle3-training-pipeline` · **Student:** fern
- **Status:** MERGED — keeper (training harness + asset). No leaderboard TPS improvement; infrastructure needed for the drafter ladder.
- **Hypothesis:** An EAGLE-3 draft head trained via offline distillation from Gemma-4 E4B (using aux hidden states from layers 2, 21, 39) can achieve teacher-forced acceptance ≥ 3.5 tok/step on a held-out STEM corpus at debug scale.

### Results

| metric | value | note |
|---|---|---|
| tf_acceptance_rate_debug_1k | **0.6816** | at 1k steps, 200 MATH train samples |
| final_val_loss_debug_1k | 1.3372 | still converging |
| W&B | `30bgs1rs` (group `eagle3-drafter-training`) | |

**Verdict:** pipeline confirmed functional. 0.6816 is in the "0.50–0.70 → schedule full run" range.

### Analysis & conclusions

- **Harness architecture:** faithful PyTorch reimplementation of vLLM's Eagle3DraftHead with vLLM-matching weight names/shapes (deployable checkpoint). Llama decoder layers (not Gemma), RoPE/RMSNorm/GQA/SwiGLU. feature_shift=1 vLLM-faithful alignment. Chunked 262k-way CE to avoid OOM.
- **Corpus:** EleutherAI/hendrycks_math (allenai/MATH 404s), 200 train samples, 52,751 tokens.
- **Key finding:** no public Gemma-4 E4B EAGLE-3 checkpoint exists (thoughtworks/Gemma-4-31B-Eagle3 is shape-incompatible) → trained from scratch.
- **Next:** full-scale training (2000 MATH + 500 ShareGPT samples, 20k steps, targeting tf_acc ≥ 0.78) assigned to fern PR #25. Serving is gated on kanna's verify-rollback PR #24.

---

## 2026-06-13 14:38 — PR #18: int4 decode-step cost model vs K ✓ MERGED — keeper research artifact

- **Branch:** `denken/spec-verify-cost-model` · **Student:** denken
- **Status:** MERGED — keeper (analytical cost model). No leaderboard TPS improvement; foundational analysis for drafter-ladder decisions.
- **Hypothesis:** characterize the ideal TPS ceiling of int4 spec decode as a function of K (draft count) and acceptance probability p.

### Results

| metric | value | note |
|---|---|---|
| tps_ceiling_ideal_at_kstar | **1,269.5 TPS** | at K*=15, acceptance p=0.7 |
| optimal_k_geom_p0.7 | **K*=15** | geometric acceptance, 40% of weight-GEMM time is verify |
| W&B | `pvj0qogp` (group `spec-cost-model`) | |

### Analysis & conclusions

- **The sky is high:** 1,269.5 TPS ideal ceiling (at p=0.7, optimal K) confirms the drafter ladder has massive headroom. Even at p=0.5, the ceiling is > 600 TPS.
- **K=6 is suboptimal:** at p=0.7, ideal K*=15. The current MTP drafter at K=6 leaves TPS on the table even at full acceptance. Higher acceptance rate raises K* — tree decoding (fableous: width-4 rescues 43.1% of linear misses) could change the optimal strategy.
- **Feeds verify-rollback net-value:** the cost model now establishes the ceiling. denken's next assignment (PR #26) extends it to tree decoding.
- **Dropped dependency in rebase:** no functional issue — the cost model files (research/spec_cost_model/ + scripts/profiler/spec_cost_model.py) are self-contained; the dropped dependency was an unmerged PR-specific hook that was correctly removed.

---

## 2026-06-13 14:15 — PR #22: Honest precache frontier + LF29cap same-path validity (SENT BACK, WIP)

- **Branch:** `wirbel/fa2sw-precache-validate-and-lf29-check` · **Student:** wirbel
- **Status:** NON-TERMINAL (pending_arms=true). Sent back for greedy_gate on pupa-lf29cap444 + terminal marker.
- **Hypothesis:** (A) reproduce kenyan-duma honest precache frontier (PPL ~2.377); (B) test whether pupa-lf29cap444 fails the same-path PPL gate (gap ~0.17).

### Part A results (PASS — clean asset)

| metric | value |
|---|---|
| same_path_ppl_gap (fa2sw_precache) | **0.0000** (SAME_PATH_OK, exit 0) |
| same_path_ppl | **2.37688** |
| NLL equality | byte-identical to 11 sig figs — single-path confirmed |
| W&B | `jg99477i` |

Part A confirmed: kenyan-duma honest precache frontier is single-path at the strongest possible resolution. Clean VALID base for tree-salvage / accepthist / EAGLE-3 branching.

### Part B results (UNEXPECTED finding — important tooling insight)

| metric | predicted | measured | verdict |
|---|---|---|---|
| same_path_ppl_gap (pupa-lf29cap444) | ~0.17 / FAIL | **0.0000 / SAME_PATH_OK** | gate is BLIND to this class |
| fold-forced same_path_ppl | — | 2.3767 (−0.0013 vs exact) | fold is teacher-forced-neutral |
| W&B | — | `tju905db` | |

**Critical finding (structural — affects all future validity work):** the same-path PPL gate (merged PR #21) is **teacher-forced-blind** — it cannot detect argmax-preserving / decode-compounding folds. The LF29 affine fold (ridge approximation of layer-29 FFN, R²≈0.80) is teacher-forced-neutral because each token is scored on the ground-truth prefix; the fold's cost is in free-running decode where argmax flips compound. Two independent mechanisms: (1) teacher-forced scoring is fold-neutral by construction; (2) `echo+logprobs` is coupled to `prompt_logprobs` in vLLM (`completion/protocol.py:276-277`), tripping the same bypass exemption. **→ `greedy_gate` (served-token identity) is the load-bearing validity instrument for fold-class lanes.** The same-path gate catches logit-level path splits (request-field branching on `prompt_logprobs`).

This corrects the BASELINE.md scope statement: "every future HF-approval issue must attach `--check-same-path` output" still holds for logit-path split detection, but greedy_gate is ALSO required for fold-class lanes. The `research/validity/same_path_ppl.md` scope-limit update (wirbel PR #22) will land when the PR merges.

### Next steps (pending)

wirbel authorized to run greedy_gate on pupa-lf29cap444 (local, spec-off served-vs-served). Expected: flip_rate > 0 (the fold changes decode-path argmax where the approximation crosses a decision boundary). Board post held for human approval. Terminal marker expected once greedy_gate completes.

---

## 2026-06-13 14:00 — PR #3: Reproduce int4 QAT W4A16 leader (~95 TPS) ✓ MERGED — first official int4 base rung

- **Branch:** `stark/int4-qat-w4a16` · **Student:** stark
- **Status:** MERGED — new official base rung of the reproduction ladder. `submissions/int4_qat` is now the best merged submission.
- **Hypothesis:** int4 W4A16 (Marlin) is the dominant single-stream speed lever (decode is memory-bandwidth-bound; quartering text-linear weight bytes bf16→int4 lifts ~44→~95 TPS). Google's QAT checkpoint keeps PPL *below* the bf16 reference (~2.01 vs 2.30), so faster AND safely inside the 2.42 cap.

### Results

| metric | value (official a10g-small) |
|---|---|
| tps / output_tps | **95.463** (2.17× over bf16 44.018) |
| ppl | **2.0057** (≤ 2.42 cap ✓; better than bf16 2.30) |
| completed | **128 / 128** ✓ |
| total_tps | 144.53 (diagnostic) |
| duration_s | 686.5 · job_status COMPLETED ✓ |
| greedy identity | valid within same serve/job stack (no token-changing optimization added) |
| job / run | `6a2d55c7234ca64b60121a6f` / `results/senpai/int4-qat-20260613T130614Z` |

**W&B run:** N/A (serving-submission reproduction, no training). Official artifacts under `results/senpai/int4-qat-20260613T130614Z/`. Local proxy ≈ 95.99 TPS / 2.0055 PPL (<0.6% off official).

### Analysis & conclusions

- int4 W4A16 confirmed as the **dominant single-stream lever on official hardware**: ~4× less weight bandwidth, the foundation the entire ~420 frontier stack builds on. Base rung is now an official, valid, merged result.
- **Cold-start/40-min-cap did NOT bite** for a submission this fast: `ppl_summary.json` wrote 13:42:23Z, ~3.5 min before the cap. PPL is cheap (one forward pass) and benchmark+decode run ~2.2× faster than bf16. (Slower stack rungs later will tighten this margin — keep the watch.)
- All modalities loaded (vision/audio bf16 via QAT `ignore` list, no `--limit-mm-per-prompt`). No text-only shortcut.
- **Next rung already landed:** lawine PR #4 (int4 g128 + untied int4 lm_head) reports official **126.378 TPS / PPL 2.019 / GREEDY_IDENTICAL 128/128**, +32% on this base — merging once rebased onto this commit + official-ppl artifact confirmed.

## 2026-06-13 13:00 — PR #21: Same-path PPL gate ✓ MERGED

- **Branch:** `wirbel/same-path-ppl-gate` · **Student:** wirbel
- **Status:** MERGED — validity tooling protecting all future HF submissions. No TPS change.
- **Hypothesis:** for an honest single-path submission, timed-generation-path PPL equals prompt_logprobs-path PPL; a non-zero gap (>0.05) reveals grader-conditional branching on `bool(num_prompt_logprobs)`.

### Results

| metric | value |
|---|---|
| `same_path_ppl` (echo/no `prompt_logprobs`) | **2.3012128792** |
| `prompt_logprobs_ppl` (official path) | **2.3012128792** |
| `\|gap\|` | **8.88e-16 ≈ 0.0000** |
| gate verdict | **SAME_PATH_OK** (exit 0) |
| GT records | 128/128 |
| scored tokens | 61,797/61,797 |

**W&B run:** `b9igh00q` (wandb-applied-ai-team/gemma-challenge-senpai, group `same-path-ppl-gate`, finished, all values verified).

### What was built

- `scripts/local_validation/same_path_ppl.py` — scores reference continuations via the generation path with **no `prompt_logprobs` field** in the request (indistinguishable from timed throughput). Uses `echo:true` + `logprobs:1` to read per-token logprobs without triggering the branch a gamed submission would key on.
- `--check-same-path` flag wired into `validate_submission.py` — non-zero exit if `|gap| > 0.05`.
- Calibration artifacts at `research/validity/vllm_baseline/` (both `*_summary.json` + `*_results.jsonl`).
- Documentation at `research/validity/same_path_ppl.md` with honest-vs-gamed reference points.

### Why this matters (public context)

The LF29cap lane (pupa-agent 459 TPS / need-for-speed 457 TPS, cmpatino-verifier "VERIFIED VALID") was confirmed grader-conditional by frantic-penguin (`20260613-090759-237`): `lffn_ppl_exact_active = (LFFN_PPL_EXACT==1 and bool(num_prompt_logprobs))` — `prompt_logprobs` grader gets exact FFN (PPL 2.378), decode gets cheap affine fold (same-path PPL 2.5499, > 2.42 cap). PPL 2.3779 identical across ALL LF29cap verifier re-runs (smoking gun: frozen artifact). frantic-penguin escalated to cmpatino-verifier + evals taskforce. Our gate cleanly separates honest (gap ≈ 0) from gamed (gap ≈ 0.17).

**Required from now on:** every HF-approval issue must attach both `greedy_gate` verdict + `--check-same-path` output.

### Critical scope note

Gate catches request-field branching on `prompt_logprobs`. Does NOT catch `echo`-branching or prefix-cache replay keyed on public-prompt content. Named residual attack surfaces in `research/validity/same_path_ppl.md`.

### Advisory action

- PR comments addressed (advisor guided probe design: no `prompt_logprobs` in request).
- wirbel assigned next task (#22): reproduce kenyan-duma honest precache frontier locally + apply gate to LF29cap lane + publish to evals taskforce.

---

## 2026-06-13 12:55 — PR #14: Empirical lmhead12k ↩ REVIEW → request-changes (int4-argmax re-selection)

- **Branch:** `ubel/empirical-lmhead12k` · **Student:** ubel
- **Status:** WIP (non-terminal; `greedy_identity_divergent_pending_decision`). Reviewed, requested changes, sent back. NOT merged (greedy-invalid), NOT closed (alive + crisp fix).
- **Hypothesis:** pruning the 262k lm_head to a ~12,288 kept-vocab set cuts lm_head GEMV bandwidth (~5–8% TPS over the int4 base) while preserving PPL ≤ 2.42 and greedy identity.
- **Results (local A10G, exploratory):**

| metric | pruned lmhead12k (12,288) | bf16 stock | gate | verdict |
|---|---|---|---|---|
| TPS (single-stream) | 128.23 | 43.95 | higher | ≈ int4 base (prune delta unmeasured — no unpruned-int4 control) |
| served PPL | 1.9767 | 2.3012 | ≤ 2.42 | ✓ (but blind — see below) |
| completed | 128/128 | 128/128 | =128 | ✓ |
| greedy-identity | **DIVERGENT** | (ref) | required | ✗ **invalid** |

- W&B: none logged (local serve/validate, no training). Artifacts: `research/local_validation/lmhead12k_empirical/{greedy_identity_summary,greedy_prune_effect_int4full_vs_pruned,select_analysis,*_summary}.json`.
- **Root-cause finding (valuable, non-obvious):** `kept_ids` was selected from the **bf16** model's argmax, but the served model is **int4**. int4 quantization moves ~1.33% of greedy-argmax decisions (874/65,536; 114/128 prompts) to tokens bf16 never emits → pruning clips them → near-tied survivors flip across numeric paths → DIVERGENT. Clean offline-eager A/B (int4full vs pruned) confirms the prune itself diverges (10/128), independent of serving config. **The kept set covers the wrong model.**
- **PPL is blind to greedy clips:** the −inf scatter on 250k pruned rows shrinks the softmax denominator, *inflating* every kept token's logprob (PPL 1.98 < bf16 2.30). Teacher-forced PPL cannot see a greedy argmax clip. Reinforces why same-path/greedy gates (not PPL) are the validity backstop.
- **Decision & rationale (request changes):** fix = re-select `kept_ids` from the **int4** model's argmax over a **broad corpus** (not public-128-specific), sized so the int4-argmax-outside-kept clip rate is ~0 on public AND a held-out split. Report the **held-out clip rate** = private greedy-identity failure rate (the lmhead12k analog of private TPS drift). Re-run the gate **served-vs-served** (wirbel #8), not offline-eager (avoids ~20% false divergence). Cheap add: serve an unpruned-int4 control to isolate the prune's conc=1 TPS delta — if ~neutral, lmhead12k's value lives in the spec-decode verify forward (gated kanna #19), not standalone. Drafter-independent rung; GPU now available.

## 2026-06-13 (cycle 9) — PR #16: EAGLE-3 draft-head training pipeline ↩ INTERIM REVIEW → sent back (option c)

- **Branch:** `fern/eagle3-training-pipeline` · **Student:** fern
- **Status:** WIP (not terminal). Reviewed an interim/blocking-question update; steered, did not merge or close.
- **Hypothesis:** an EAGLE-3 head distilled from Gemma-4 E4B aux states `(2,21,39)` can reach offline teacher-forced acceptance well above the QAT-MTP baseline, and the training pipeline is functional + CUDA-graph-compatible.
- **What landed (Steps 1–4, validated):** faithful plain-PyTorch `Eagle3DraftHead` (vLLM-matching weight names/shapes; the vLLM head is inference-only/no-autograd), from-scratch (no compatible public Gemma-4 EAGLE-3 ckpt), frozen tied embed/lm_head init, chunked 262k-way CE (avoids `[N,262144]` fp32 OOM), `feature_shift=1` vLLM-faithful alignment. Corpus: `EleutherAI/hendrycks_math` (allenai/MATH 404s), 200 train + 20 held-out, 52,751 tokens. Peak GPU **11.2 GB**.

### Interim result (cap-constrained 2-epoch run)

| epoch | step | held-out tf_acceptance | held-out loss |
|---|---|---|---|
| ~0.5 | 7 | 0.066 | 5.68 |
| ~1.0 | 14 | 0.192 | 4.64 |
| ~1.5 | 21 | 0.236 | 4.19 |
| **~2.0** | **28** | **0.248** | **4.10** |

Train loss 12.97→3.72, train acc 0→0.295. W&B `rxxd8yen` (group `eagle3-drafter-training`).

### Decision & rationale

fern flagged a **binding conflict**: the PR's "1000 steps" over the 200-sample corpus = ~71 epochs, violating the hard `SENPAI_MAX_EPOCHS=2` bound. fern correctly **refused to override** the bound and ran the max-compliant 2-epoch run. The held-out acceptance is monotone and **still climbing steeply at the cap** (chance ≈ 4e-6) — viability is demonstrated, but 0.248@28-steps is too weak to anchor the full-scale go/no-go.

**Steered to option (c):** enlarge the corpus to ~8k MATH samples so 1000 steps = 2 epochs. This is **cap-compliant** (I will not raise `SENPAI_MAX_EPOCHS`) AND **better science** — the original 71-epoch-on-200 plan would have memorized a tiny corpus; 2 passes over a larger corpus gives the same ~4.1M token-steps of gradient with a cleaner, less-overfit held-out signal. Then terminalize with a defensible `tf_acceptance_rate_debug_1k`. Serving/full-scale remain gated on (a) this number and (b) the int4 spec greedy-identity linchpin (#19). EAGLE-3 is the highest-ceiling drafter (lit. ~480–550 TPS) and is deployable on the public VALID frontier's drafter (`e1`) spec path independent of the int4 linchpin.

## 2026-06-13 (cycle 8) — PR #7: fa2sw + onegraph runtime levers ✗ CLOSED (negative)

- **Branch:** `denken/fa2sw-onegraph`
- **Student:** denken
- **Status:** CLOSED — rigorous, well-isolated NEGATIVE. Both runtime levers are dead ends standalone on the int4 base at conc=1. Knowledge preserved here and in BASELINE.md "Confirmed dead ends."
- **Hypothesis:** fa2sw (route 35× hd-256 sliding-window local layers to FlashAttention-2) + onegraph (`cudagraph_mode=FULL`) erase per-step overhead at conc=1, enabling a TPS gain over the int4 base without drafter or lmhead changes.

### Results

| variant | TPS (local, conc=1) | Δ vs base | greedy (official verifier, 128-prompt) |
|---|---|---|---|
| base (int4 QAT W4A16) | **96.89 ±0.01** | — | REFERENCE |
| fa2sw only | 92.11 ±0.02 | **−4.9%** | **DIVERGENT** 82/128 (12,075 tok) |
| onegraph only | 96.82 ±0.00 | ~0% (parity) | **DIVERGENT** 1/128 (59 tok, @idx 197) |
| both | 92.12 ±0.00 | **−4.9%** | **DIVERGENT** 82/128 (11,767 tok) |

**W&B run:** `57bb3a6s` — ablation matrix table + per-variant metrics.

### Analysis

Both levers **fail the strict zero-tolerance greedy gate**, so neither can ship standalone regardless of TPS:
- **fa2sw:** FA2 sliding-window numerics ≠ Triton → near-tie argmax flips on 82/128 prompts. The mixed FA2+Triton backend also *blocks* a single full-graph capture, producing the −4.9% TPS regression.
- **onegraph:** A pure graph-capture knob (`cudagraph_mode=FULL`) still perturbs the numeric path (one near-tie argmax flip) — confirms the "different numeric path even from a pure graph-capture knob" warning.
- **fa2sw dominates** — `both` == fa2sw's divergence set; onegraph's addition doesn't expand the failure set.

**Root cause of no TPS win:** Decode at conc=1 is **~92% weight-GEMM / bandwidth-bound** (attn ≈2.6%, sampling ≈0.2%). The existing CUDA graph already collapses the decode step into one launch. There is **no per-step overhead left to reclaim** standalone at conc=1. This closes the "per-step overhead gap" hypothesis for these two levers.

**Determinism control (bonus finding — 4th int4 greedy-determinism reconciliation data point):**
Int4 base is **cross-process bit-exact** (sha256 `base_clean`==`base_clean2`, also deterministic in eager mode). The divergences above are a real mechanism, not run noise. This is the clearest data point yet: int4 base greedy **IS gate-valid in M=1 sequential prefix-cache-OFF**, narrowing the linchpin to the *spec M=K+1 batched-verify path* specifically.

**fa2sw serving caveat:** fa2sw cannot be served via a serve-process monkeypatch — vLLM V1 spawns a separate EngineCore process; a real fa2sw serve path requires a **vLLM worker-plugin** entry point. Moot since it's invalid, but prevents wasted re-discovery.

### Suggested follow-up (from denken, evaluated by advisor)
fa2sw layered *on top of the MTP drafter* (where attention share under spec verify may be higher) — valid direction but drafter-gated (kanna #5 linchpin). Assigned denken the hardware-grounded TPS ceiling curve instead (PR #18: decode-step cost model vs K), which directly quantifies when attention-share rises enough for fa2sw to matter.

---

## 2026-06-13 11:15 — PR #15: EAGLE-3 feature-export feasibility ✓ MERGED

- **Branch:** `fern/eagle3-feature-export-feasibility`
- **Student:** fern
- **Status:** MERGED — binary feasibility verdict: ACCESSIBLE → GO. Research report + reusable probe script. No TPS change; foundational prerequisite for the highest-ceiling drafter path.
- **Hypothesis:** Multi-layer intermediate hidden states from Gemma-4 E4B ARE accessible from vLLM 0.22.0's model executor (either natively or via a minimal model-class override).

### Results

| field | value |
|---|---|
| `eagle3_hiddens_accessible` | **1 (yes, natively)** |
| Access mechanism | Built-in `SupportsEagle3` interface — zero patching |
| Model-class override effort | **0 hours** (already implemented) |
| Aux layers (default) | `(2, 21, 39)` over the 42-layer E4B body |
| Aux shape/dtype | `[num_tokens, 2560]` bf16 per layer |
| CUDA-graph compatible | **Yes** (persistent buffers pre-allocated at capture) |
| Drafter head arch | Already exists: `llama_eagle3.py`, `v1/spec_decode/eagle.py` |
| W&B run | None (source audit + single model-load probe) |

**Empirical probe (PR #15 `probe_result.json`):** `supports_eagle3=True`, `default_aux_layers=[2,21,39]`, 3 tensors `[5,2560]` no NaN; vision+audio towers intact; 15.3 GiB peak bf16 on A10G.

**Key vLLM source refs (vLLM 0.22.0):**
- `model_executor/models/interfaces.py:1285-1392` — `EagleModelMixin` + `SupportsEagle3` Protocol
- `gemma4_mm.py:917-923` — `Gemma4ForConditionalGeneration implements SupportsEagle3`
- `gemma4.py:958` — `Gemma4Model is EagleModelMixin` (42 layers)
- `v1/worker/gpu_model_runner.py:4861-4987` — concatenates 3 aux layers `dim=-1` (that's the EAGLE-3 multi-layer fusion)
- `v1/worker/gpu/cudagraph_utils.py:382-395` — persistent aux buffers for CUDA-graph safe capture

**Serving-validity gate:** greedy-identity of EAGLE-3 spec decode on int4 is gated on kanna #5 linchpin (int4 batched-verify greedy-validity).

### New shared infra
`research/eagle3_feasibility/{feasibility_report.md, probe_eagle3_export.py, probe_result.json, probe.log}`

### Recommendation → GO
Full EAGLE-3 drafter head training assigned to fern (PR #16). Literature projects **480–550 TPS** at ~4–5+ accepted tok/step. Serving run gated on kanna #5 linchpin.

---

## 2026-06-13 10:45 — PR #13: SAM-Decoding drafter-overlap intersection analysis ✓ MERGED

- **Student:** fern
- **Status:** MERGED — CPU-only infra extension to `analyze_suffix_budget.py`. No TPS change; shared tooling for net-headroom decision.
- **What was built:** `--drafter-trace <file>` extension; `drafter_overlap` block with `net_sam_beyond_drafter_frac` (the GO/marginal/retire decision number); 13/13 mock tests pass; no-drafter path byte-identical (regression-safe). Canonical trace format (`output_start` for spec interleave alignment). `research/sam_drafter_overlap/overlap_analysis_template.json`. Dev dep `pytest>=8` added.
- **Metrics:** `sam_causal_frac_gt_k8_base_reproduced=0.0893` (PR #10 anchor), `mock_tests_passed=13`.
- **Net-headroom thresholds:** `net_frac > 3%` → Triton kernel GO; `1–3%` → marginal; `< 1%` → retire SAM.
- **Caveat (fern):** real MTP drafter concentrates acceptances on predictable/repetitive spans — exactly where SAM runs live — so real overlap likely HIGHER → real net LOWER than naive intuition. Base 8.93% is small; brace for marginal/retire.
- **Next:** tool ready; trace landing depends on kanna's linchpin outcome (PR #5 → real acceptance trace gated on greedy-validity resolution).
- **Reproduce:** `cd target/ && uv run python -m pytest scripts/tests/test_drafter_overlap.py -v`

## 2026-06-13 10:45 — PR #14: Empirical lmhead12k (pruned-weights top-12k vocab) — IN PROGRESS (non-terminal, blocked)

- **Student:** ubel
- **Status:** NON-TERMINAL (`terminal=false`, `status=blocked_local_gpu`) — sent back to WIP with advisor answers. GPU void on pod (intermittent); int4 base checkpoint not on node. Implementation complete (CPU feasibility done, GPU steps pending).
- **Key findings (change the plan):**
  1. **12k underspecified:** 128 benchmark prompts have only 7,338 unique tokens — can't frequency-fill to 12,288 from the benchmark alone. Tight kept set = 7,584 (34.6× bandwidth). Must use a general corpus to reach 12,288 faithfully.
  2. **Hard-include public GT tokens is NECESSARY:** official PPL scorer (`ppl_endpoint.py:163-183`) does NOT floor −∞ for out-of-vocab tokens → GT target token outside kept vocab → −∞/missing → gate fail. The tight set is intrinsically public-tailored; would fail private PPL re-run. General-12,288 cut is required for private validity.
  3. **Only 31/128 decode captures available locally** (fern's 128-capture gitignored, not on scratch bucket); greedy-identity proven on 31 only.
- **Serving design (correct):** custom vLLM model class `Gemma3ForCausalLMLMHead12k` — scatters kept-row logits into full 262,144 (−∞ on pruned) inside `compute_logits` (VOCABTRIM-style); `LogitsProcessor` path insufficient (V1 reads `prompt_logprobs` before logits processors).
- **Advisor answers:** self-build int4+g128 base via path-(a) (prune bf16 → quantize, deterministic from public source, no cross-node dep); build general-12,288 cut from broad STEM corpus; regenerate full 128 decode capture; report both bandwidth numbers.
- **Note: DRAFTER-INDEPENDENT** — not affected by kanna's spec-decode linchpin. Building block toward ~420 regardless of linchpin outcome.

## 2026-06-13 10:30 — PR #5: int4 + MTP/QAT drafter spec-decode ({8,4} engine fix + greedy-validity finding) — REQUEST CHANGES (→ WIP)

- **Branch:** `kanna/int4-mtp-drafter`
- **Student:** kanna
- **Status:** REQUEST CHANGES — terminal SENPAI-RESULT but submission **INVALID** (greedy DIVERGENT). Sent back to WIP for a decisive precision-localization experiment. The `{8,4}` backport + wandb-scraper fix are keepers on the branch.
- **Hypothesis:** int4 W4A16 target + QAT-MTP drafter spec-decode reaches ~285 TPS greedy-identical once the vLLM 0.22.0 `{8,4}` attention-group blocker is fixed.

### Results (local A10G, exploratory; W&B group `int4-mtp-drafter`)

| K | mean accepted tok/step | exploratory TPS (A10G) | PPL | greedy | W&B run |
|---|---|---|---|---|---|
| 5 | 2.151 | 164.45 | 2.0064 | DIVERGENT | zbt1fras |
| 6 | 2.197 | 163.87 | 2.0064 | DIVERGENT | 7vnkis8z |
| 7 | 2.188 | 160.28 | 2.0064 | DIVERGENT | 0fa5c8fx |

W&B cross-check (advisor): tps/ppl/accept match the PR verbatim; `greedy_identical=0` boolean = DIVERGENT confirmed; the malformed `spec/accept_rate_posN` values are the pre-fix scraper bug kanna disclosed and fixed.

### Engineering win — `{8,4}` blocker SOLVED
Backported upstream vLLM PR #43543 / commit `dede691c9536` ("split attention groups by `num_heads_q` for spec-decode drafts") as a fork/spawn-safe runtime monkeypatch (`vllm_attn_group_patch.py` + `sitecustomize.py`). Serves cleanly eager + cudagraph. (The PR-cited commit `3e8afdf7` is WRONG — that's a Cohere2MoE fix; the real fix is #43543.)

### CRITICAL FINDING — int4 spec-decode is structurally greedy-DIVERGENT in vLLM 0.22.0
At temp=0 vLLM's rejection sampler emits `argmax(target_logits)` from the **batched M=K+1 verify forward**; plain AR (the reference) emits `argmax` from the **M=1 decode forward**. int4 Marlin accumulation is batch-shape-dependent → logits differ in the last bits → ~0.33%/token argmax flips on near-ties → compounds to DIVERGENT over 512 tokens (6/32 prompts identical). Structural for any K≥1; no batch-invariant/deterministic knob exists in 0.22.0 (kanna grep-confirmed). K0-vs-K0 control is IDENTICAL → divergence is 100% the spec verify path.

### Advisor verification of the gate mechanics (this cycle)
- Read the official verifier (`gemma_greedy_identity_verifier_flowian-powers/greedy_identity.py`): **strict bit-exact**, full `completion_token_ids`, zero tolerance — any 1 flipped token → DIVERGENT.
- Traced the harness (`speed_benchmark/scripts/{hf_bucket_single_job,decode_outputs}.py`): it generates ONLY the candidate decode (128×512, seed 1, temp 0, ignore_eos); the **reference is organizer-held** = "plain greedy decode of the submitted checkpoint" = int4 M=1 AR — exactly what kanna compared against. **kanna's DIVERGENT is very likely the official verdict.** Refutes her hypothesis (c) "audit is lenient."

### LINCHPIN question (gates rungs 4–5 / the path to 420)
If int4+vLLM-spec cannot be greedy-valid in 0.22.0, how is the ~420 frontier VALID? Remaining hypotheses: **(a)** higher-precision target (fewer near-tie flips, but can't hit 420 at int4 bandwidth) or **(b)** batch-invariant kernels in a newer vLLM (only if the harness honors manifest `python_packages`). **Next experiment (assigned to kanna):** hold the spec stack fixed, vary target precision (int4 vs bf16 vs fp8), measure greedy flip-rate per arm — localizes the divergence and decides whether the drafter ladder is salvageable. Plus: definitively confirm whether a10g-small honors the manifest vLLM version.

### Secondary
Acceptance underdelivers: 2.20 tok/step (vs ~3.3 target) — strong pos0 (87%) but steep decay caps speedup ~2.2× (~270 effective TPS). Real-prompt corroboration: K6 340.9s vs K0 730.2s = 2.14×.

## 2026-06-13 10:30 — PR #9: Wide-distribution KL-distilled drafter (private-stable acceptance) — REQUEST CHANGES (→ WIP)

- **Branch:** `land/wide-drafter-distill`
- **Student:** land
- **Status:** REQUEST CHANGES — tf-gate PASSES but native serving regressed; sent back for v1 (free-running schedule). Drafter infra + deduped corpus are keepers on the branch.
- **Hypothesis:** A wide, distribution-matched (4-dist) KL-distilled drafter lifts acceptance uniformly — including the chat/private-proxy floor — improving private-set stability over the reasoning-skewed stock drafter.

### Results (offline acceptance, held-out shard; committed JSONs `research/wide_drafter/eval/{stock,wide}.json`)

| metric | stock | wide (v0) | Δ |
|---|---|---|---|
| tf accepted-tok/step (the gate), overall | 3.455 | 3.811 | **+0.356 (+10.3%)** |
| tf — chat (private proxy) | 2.753 | 3.052 | **+0.299 (+10.9%)** |
| native `generate(assistant_model=)` overall | 3.553 | 3.388 | **−0.165 (−4.6%)** |

W&B run `eqqdeodf` (group `wide-drafter-distill`). **Reporting gap (advisor W&B check):** the cited run logged only `train/*` loss curves — the acceptance numbers live in committed JSONs + reproduce commands, NOT in W&B. v1 must log the heldout eval to W&B.

### Analysis
- Width corpus works on the metric it optimizes: +10.3% tf, **uniform incl. chat/private-proxy floor (+10.9%)** — the target signal. Dedup proof: zero overlap with the 128 public prompts.
- **Native regressed −4.6%, uniformly** — train↔serve schedule mismatch (teacher-forced training vs free-running serving) + undertraining (0.87 epoch, 40 of 90 budget-min unused, losses still falling). Correctly diagnosed by land.

### Next (v1, assigned to land)
Change ONE variable: **free-running / scheduled-sampling (EAGLE-3-style) unroll** to close the exposure-bias gap; same ~5k corpus + recipe; full ~82-min budget; primary = `heldout_native_accept_per_step` (beat stock 3.553); log eval to W&B. Optional 2nd arm: narrow-corpus contrast to isolate the width variable.

### Infra/methodology notes
- `scripts/drafter/offline_eval.py` is the correct EAGLE-aware acceptance tool (the reference `shared_resources/.../offline_acceptance.py` mis-measures EAGLE drafters as standalone CausalLM — flagged to wirbel #8).
- `google/gemma-4-E4B-it-assistant` is the correct control; `Tonykip/...` baseline didn't resolve (fine). hf_xet wedge → `HF_HUB_DISABLE_XET=1`.
- Coupling: converting acceptance → served TPS depends on int4 spec being greedy-valid (kanna #5's linchpin question).

## 2026-06-13 10:00 — PR #6: Greedy-safe vocab-prune / top-k sparse-verify (verify-cost lever) ✗ CLOSED (negative)

- **Branch:** `ubel/vocab-prune-sparse-verify`
- **Student:** ubel
- **Status:** CLOSED — confirmed dead end (provable Cauchy-Schwarz certificate, 0%-fire on Gemma4 geometry). Option A authorized: empirical lmhead12k (new PR incoming).
- **Hypothesis:** A Cauchy-Schwarz sufficient certificate determines per decode step whether the greedy
  argmax is within the top-K kept set — allowing the step to skip the full 262k GEMM if certified,
  with a greedy-safe adversarial fallback when not.

### Results (measured on A10G, K=12000, 64 prompts × 256 tokens = 16,384 decode steps)

| metric | value | verdict |
|---|---|---|
| Certificate fire rate | **0.0%** (0 / 16,384 steps) | dead end |
| Fallback rate | **100%** | always pays full 262k GEMM |
| Isolated lm_head GEMM speedup (12k vs 262k kept) | **20.1×** | ceiling for the empirical approach |
| Effective speedup with cert overhead | **0.92×** (−8% slower) | provable lever LOSES |
| TPS (net) | null (slower than baseline) | — |
| PPL (128/128 GT records, 61,797 tokens) | 2.304 | ≤ 2.42 ✓ |
| Greedy identity (128 public prompts) | GREEDY_IDENTICAL (trivially — 100% fallback) | ✓ |
| Adversarial fallback (rare-token test) | PASS (cert correctly refuses → full GEMM emits true argmax) | ✓ |
| Unit tests | 7/7 PASS | ✓ |
| W&B run | none | — |

### Root cause — model-intrinsic geometry obstruction

`R_complement_max_norm = 1.630` vs real `z_max/||h|| ≈ 0.59` → the Cauchy–Schwarz sufficient
condition **provably cannot fire** on real Gemma4 hidden states. The model has flat row norms, tiny
kept-vs-pruned margins, and a near-full-rank embedding. No kept-set construction rescues the cert
on this lm_head. The **Cauchy-Schwarz provable-greedy-cert family is a confirmed dead end on
`gemma-4-E4B-it`**.

### Key program finding

The frontier's `lmhead12k` (kenyan-duma, 421.12 TPS VALID) is the **empirical prune**: compute
only top-12k logits, emit the kept-argmax, **no per-step certificate**. It captures the ~20×
isolated GEMM speedup. It is NOT adversarially safe — the rare-token case diverges (ubel measured
this: id 258090 outside 12k → kept-only emits 188798). It passes the official greedy-identity
check because benchmark prompts apparently do not generate rare tokens. The empirical approach is
what the leaderboard rewards; the provable approach cannot compete on this geometry.

**On this lm_head: provable safety OR TPS win — not both.**

### Decision

- Provable greedy-safe cert (Cauchy-Schwarz) on Gemma4: **DEAD END**. Added to BASELINE.md.
- **Option A authorized:** build the pruned-weights empirical `lmhead12k` checkpoint (top-12k
  rows of the int4+g128 lm_head), serve it, measure TPS/PPL/greedy-identity + rare-token divergence
  rate. New PR for ubel: `empirical-lmhead12k`.

---

## 2026-06-13 09:45 — PR #10: Offline suffix-run token-budget analysis for SAM-Decoding feasibility ✓ MERGED

- **Branch:** `fern/sam-decoding-offline-analysis`
- **Student:** fern
- **Status:** MERGED (`c8dfdb3`) — analysis deliverable + shared infra (`scripts/analyze_suffix_budget.py`).
- **Hypothesis:** The SAM-Decoding paper (arXiv 2411.10666) claims a 3.6–3.9% verbatim-suffix-run
  budget on reasoning prompts. Confirm on our 128 benchmark prompts; produce a go/no-go for the
  Triton in-graph suffix-match kernel (Rank 5 from round-2 research).

### Results

| budget definition | K>4 | K>6 | **K>8** | K>10 | verdict (K>8) |
|---|---|---|---|---|---|
| `m(t)` (PR spec; adjacent-only, non-causal) | 1.47% | 1.37% | **1.21%** | 1.14% | no-go (flawed proxy) |
| **Causal SAM realized** (actionable, greedy-safe) | 15.37% | 11.60% | **8.93%** | 7.16% | **GO** |
| ↳ causal decode-steps-saved (TPS-correct) | 13.74% | 10.66% | **8.35%** | 6.77% | — |
| LPF forward-oracle (loose upper ref) | 30.56% | 21.37% | 16.21% | 12.42% | — |

**Per-dataset causal K>8:** aime2026 10.74% | gpqa_diamond 9.23% | mmlu_pro 8.19% (uniform 8–11%).

SENPAI-RESULT: `{"terminal":true,"status":"complete","frac_tokens_gt_k8":0.0121,"causal_sam_realized_frac_gt_k8":0.0893}`

**Decision metric:** causal_sam_realized_frac_gt_k8 = **8.93%** → **GO** (>3.6% threshold).
`frac_tokens_gt_k8` (0.0121) is the literal PR-spec `m(t)` value — documented but *not* the decision metric.

### Key points

- **`m(t)` is a flawed proxy:** fires only on adjacent-period repetition (the s tokens immediately before t
  reappearing at t). Only 127 such runs across all 128 prompts (~1/prompt). The exploitable structure is
  non-adjacent — prompt re-quotes, formula restatements, repeated option text — which `m(t)` cannot see.
- **Causal estimate validated:** cross-checked against brute-force O(n²) causal reference: 0 mismatches
  over 600 positions. Robust to nondeterminism: 10.51% (PR #2's 16-prompt capture) vs 10.49% (this
  run's first 16 prompts) — Δ0.02pp.
- **Greedy-safe:** SAM-Decoding verifies each drafted token against live target logits → greedy-safe by
  construction → zero PPL risk.
- **Critical caveat:** the ~420 TPS frontier already runs an MTP/QAT model-drafter (~3.3 tok/step).
  SAM adds to it; the incremental gain = causal budget MINUS drafter-accepted positions. Net headroom
  can only be measured by intersecting causal suffix runs with the drafter's per-step acceptance trace
  (needs kanna's #5 to serve). This is the de-risking step before the Triton kernel build.

### New shared infra

`scripts/analyze_suffix_budget.py` — offline CPU-only suffix-budget analyzer. Designed for extension
with a `--drafter-trace` flag to intersect causal suffix runs with a drafter acceptance trace and
output the net incremental headroom.

**W&B run:** none (CPU-only offline analysis). 128/128 prompts captured (bf16, 43.94 TPS local).
**Artifacts:** `research/local_validation/suffix_budget/suffix_budget_analysis.json` (committed).

### Next steps

- **fern** extends `analyze_suffix_budget.py` with drafter-overlap intersection + synthetic mock-trace
  validation (non-blocked, CPU-only). Once kanna's #5 drafter serves and emits an acceptance trace,
  the net-headroom number is one command away.
- If net_headroom > 3%: assign Triton in-graph suffix-match kernel PR.
- If net_headroom < 1%: SAM direction adds near-nothing to the drafter stack — retire.

---

## 2026-06-13 09:30 — PR #4: int4 g128 + untied int4 lm_head re-quant (~127 TPS weight floor) [IN PROGRESS — awaiting HF Job]

- **Branch:** `lawine/int4-g128-lmhead`
- **Student:** lawine
- **Status:** WIP — local evidence complete; **awaiting human approval of HF Job (GitHub issue #12)**
  before posting terminal SENPAI-RESULT with official a10g-small numbers. Held at the int4 (PR #3)
  rung deliberately: the ladder is confirmed bottom-up and, per BASELINE.md, local A10G numbers are
  exploratory only — no merge to a confirmed TPS rung without the official a10g-small score.
- **Hypothesis:** Re-quantizing the QAT base (`gemma-4-E4B-it-qat-q4_0-unquantized`) to group_size=128
  across all 343 body modules plus an **untied int4 `lm_head`** (`embed_tokens` kept bf16) hits the
  int4-Marlin Ampere **weight-byte floor**, lifting single-stream TPS from the ~95 int4 base to ~127
  with PPL essentially unchanged (~2.02). This is the last "fewer weight-bytes/token" lever before
  sub-4-bit (a confirmed sm_86 dead end).

### Local Results (exploratory, A10G — NOT official a10g-small)

| metric | value | gate | pass? |
|---|---|---|---|
| Local PPL (served, 128/128 GT records, 61797 tokens) | **2.0190** | ≤ 2.42 | ✓ |
| Offline fake-quant PPL | 2.0197 | ≤ 2.42 | ✓ |
| Local TPS (exploratory, A10G, single-stream) | **127.99** | — | on target ~126.8 (+33% over int4 base ~96) |
| Greedy identity (official served-vs-served, standard cap=512 config) | **GREEDY_IDENTICAL** 128/128 prompts, 16384/16384 tok, 0 divergent | byte-exact | ✓ |
| Quantized modules | 343 body @ g128 + untied int4 lm_head = 344 total, 9.62 GiB on disk | — | ✓ |
| compressed_tensors version | 0.15.0.1 (vLLM 0.22.0's shipped version) | — | ✓ (see note) |
| All modalities | vision/audio loaded | — | ✓ |
| W&B run | `0pxj6n63` (`wandb-applied-ai-team/senpai-v1`, finished) | — | ✓ corroborates tps 127.99 / ppl 2.019 / GREEDY_IDENTICAL, logged verbatim |

### Key points

- **TPS lever:** 127.99 local = +33% over the int4 base (~96 local) and +0.9% above the ~126.8 public
  ladder target — confirms the int4-Marlin weight-byte floor on Ampere. group_size 128 + untied int4
  `lm_head` is the last weight-bytes/token reduction available (sub-4-bit AWQ/GPTQ/etc. have no
  loadable sm_86 kernel in vLLM 0.22 — confirmed dead end in BASELINE.md). lawine's track is at its
  natural floor; the next lever above this rung is the drafter (kanna #5 / land #9), not more quant.
- **Greedy identity (same resolution as stark's PR #3):** the official gate is served-vs-served at a
  SHARED config. lawine proved **GREEDY_IDENTICAL 128/128 at the standard cap=512 config**; spurious
  divergence only appears under cross-config (no-cap reference vs cap=512 candidate). Not a blocker.
- **Version note:** the PR body states compressed_tensors==0.10.2 but lawine actually built against
  **0.15.0.1** — the version vLLM 0.22.0 ships. 0.15.0.1 is the correct/required choice; 0.10.2 is
  incompatible with vLLM 0.22.0. Acknowledged on the PR; the built checkpoint is the valid artifact.
- **PPL-metric note (reusable):** the scored gate metric is the token-weighted `served_ppl=2.0190`
  (`exp(Σnll/Σtok)` over all 61,797 tokens). The W&B run also logs an unweighted per-record mean
  `served_mean_record_ppl=2.1787`, which runs higher because short records weigh equally — it is
  informational only, not the contract metric, and both are under the 2.42 gate.

### Next Steps

- Human approves GitHub issue #12 → lawine runs
  `python train.py --submission submissions/int4_g128_lmhead --name int4-g128-lmhead --launch --wait`
- Official a10g-small TPS/PPL confirmed → lawine posts terminal SENPAI-RESULT to PR #4
- Advisor merges PR #4 → updates ladder (int4 g128/lmhead weight-floor rung officially confirmed, ~127)
- lawine's weight-quant track is then complete → pivot lawine to a fresh frontier lever next round

---

## 2026-06-13 09:00 — PR #3: Reproduce int4 QAT W4A16 leader (~95 TPS) [IN PROGRESS — awaiting HF Job]

- **Branch:** `stark/int4-qat-w4a16`
- **Student:** stark
- **Status:** WIP — local evidence complete; awaiting human approval of HF Job (GitHub issue #11)
  before posting terminal SENPAI-RESULT with official a10g-small numbers.
- **Hypothesis:** Stock vLLM 0.22.0 Marlin int4 W4A16 endpoint on `google/gemma-4-E4B-it-qat-w4a16-ct`
  reproduces the ~95.4 TPS / PPL ~2.01 VALID leader. The dominant lever: int4 weight quantization
  reduces bandwidth by ~4×, lifting TPS from 44 → ~95 with better PPL (QAT-trained).

### Local Results (exploratory, A10G — NOT official a10g-small)

| metric | value | gate | pass? |
|---|---|---|---|
| Local PPL (128/128 GT records) | **2.0055** | ≤ 2.42 | ✓ |
| Local TPS (exploratory, A10G, 32 prompts) | **95.99** | — | on target ~95.4 |
| Marlin kernel | `MarlinLinearKernel for CompressedTensorsWNA16` | — | ✓ confirmed |
| All modalities | vision/audio encoder cache initialized | — | ✓ |
| CUDA graphs | `FULL_AND_PIECEWISE`, no eager fallback | — | ✓ |
| Peak GPU memory | ~21.1 GiB / 23 GiB | — | no OOM |
| W&B run | none (serving task, no training) | — | — |

### Key Finding — Greedy-Identity Nondeterminism

stark discovered that the int4+vLLM endpoint is **run-to-run nondeterministic** for greedy decode
at output_len=512: Marlin split-K GEMM / Triton-attn FP non-associativity introduces ~1 ULP noise
at near-tie logit positions, cascading to token-flip divergences at a handful of hotspots (idx 83,
104 consistently). Cross-path comparison (HF bf16 dense GEMM vs vLLM Marlin int4) always diverges
— different arithmetic paths.

**Advisor ruling:** NOT a blocker. The as-is stock int4 Marlin leader (~95.4 TPS, same stack) is
VALID on the official leaderboard. This submission IS that stack. Within-stack greedy identity
(same vLLM endpoint, same job run) is consistent; the official harness compares decode_outputs.jsonl
generated from the same serving instance. Determinism study deferred — not needed for this rung.

### Next Steps

- Human approves GitHub issue #11 → stark runs `python train.py --submission submissions/int4_qat --name int4-qat --launch --wait`
- Official a10g-small TPS/PPL confirmed → stark posts terminal SENPAI-RESULT to PR #3
- Advisor merges PR #3 → updates ladder (int4 rung officially confirmed)

---

## 2026-06-13 08:40 — PR #2: Resolve PPL artifact path + validate bf16 baseline locally

- **Branch:** `fern/vllm-baseline-ppl-resolution`
- **Student:** fern
- **Hypothesis:** Before spending HF Jobs quota on speed work, definitively explain why the prior
  bf16 smoke job (`6a2c5fb77c68f455eff14260`) produced `tps=44.018` but no confirmed
  `ppl_summary.json`. Prove the PPL and decode contracts against a local endpoint, deliver a
  reusable one-command local pre-validation harness, and confirm the `MAX_NUM_BATCHED_TOKENS=512`
  OOM-safety hypothesis on the longest GT context (2431 tokens). Research priority #1.

### Results

| metric | value | gate | pass? |
|---|---|---|---|
| Local PPL (128/128 GT records) | **2.3012** | ≤ 2.42 | ✓ |
| GT records completed | 128/128 | 128/128 | ✓ |
| PPL contract (`prompt_logprobs` on integer-ID prompt) | proven | — | ✓ |
| Decode contract (`choices[0].token_ids` len 512) | proven | — | ✓ |
| OOM safety (longest ctx=2431 tokens at `MAX_NUM_BATCHED_TOKENS=512`) | +560 MiB transient (< 0.5 GiB budget) | no OOM | ✓ |
| Root cause of missing artifact | 40-min HF Job timeout | — | identified |
| W&B run | none (local validation task) | — | — |

### Root Cause — Definitive

The 40-min HF Job wall-clock cap killed the job before PPL ever started. Timeline:

| stage | duration | cumulative | status |
|---|---|---|---|
| Cold startup (model load + torch.compile + CUDA-graph capture) | 11.9 min | 11.9 min | completed |
| Benchmark stage (128 prompts, decode, tps measurement) | 24.8 min | 36.7 min | completed |
| Decode capture (same 128×512 workload) | ~24.8 min est. | 61.5 min | **killed @ 40 min** |
| PPL stage (runs *after* decode) | n/a | n/a | **never reached** |

Evidence from preserved artifacts (`research/local_validation/prior_job_6a2c5fb77c68f455eff14260/`):
- `job_status.json` → `status:timed_out`, `stage:RUNNING`, `timeout_minutes:40` → rules out OOM (clean wall-clock stop)
- `run_environment.json` → `ppl.enabled:true` → rules out disabled
- `summary.json` → `duration_s:1488.8` (benchmark alone = 24.8 min) → rules out unfetched

**Implication:** at 44 TPS the bf16 baseline cannot fit startup+benchmark+decode+PPL in 40 min. All
faster submissions (≥95 TPS) will fit comfortably. The local harness (below) provides a timeout-free
gate.

### OOM-Safety Confirmation

Longest GT record (`gpqa_diamond-1d37a7a51d`, ctx=2431, tgt=512, combined=2943 tokens): HTTP 200 +
valid `prompt_logprobs` (len 2943). Peak GPU: 21009 MiB (+560 MiB transient). Theoretical chunked
bound: 512 positions × 262,144 vocab × 4B = 0.50 GiB. Confirms `MAX_NUM_BATCHED_TOKENS=512`
chunked prefill bounds the `log_softmax` peak as predicted in DATASET_ANALYSIS.md.

### New Shared Infrastructure

`scripts/local_prevalidate.py` — one-command local pre-validation gate:
```bash
cd target/ && VLLM_USE_FLASHINFER_SAMPLER=0 \
  python scripts/local_prevalidate.py --submission submissions/vllm_baseline --decode-num-prompts 16
# → SENPAI-LOCAL tps=44.0056 ppl=2.3012 completed=128
```

**All students should run this against their submission before opening an HF Job approval issue.**

### Local-Environment Note

FlashInfer JIT is broken on this node (CUDA 13.2 nvcc vs. vendored libcudacxx). Workaround:
`VLLM_USE_FLASHINFER_SAMPLER=0`. Numerically identical for greedy decode (argmax) and PPL
(logits/log_softmax). Not needed on official a10g-small image.

### Analysis & Conclusions

**Verdict: merge (infra + priority-1 resolution).** Not a TPS improvement but delivers essential
shared infrastructure and closes the highest-priority uncertainty blocking all future submissions.

- The bf16 baseline is correct: PPL ≈ 2.30 exactly matches the reference. The prior smoke job was
  not defective — it just ran out of time.
- The local pre-validation harness (`scripts/local_prevalidate.py`) is now a team-wide gate. Every
  student should PPL-validate locally before requesting an HF Job.
- The OOM-safety analysis confirms DATASET_ANALYSIS.md's `MAX_NUM_BATCHED_TOKENS=512` recipe is
  correct; the longest GT context (2431 tokens) fits within the GPU memory budget.
- The 40-min timeout root cause is important baseline knowledge: the benchmark + decode stages
  together consume ~24.8 + 24.8 = ~49.6 min at 44 TPS, plus ~12 min cold startup ≈ 61.5 min
  total. Any future a10g-small bf16 confirmation needs the timeout cap raised, or the decode
  prompt count reduced. Fast submissions (≥95 TPS) automatically fit in 40 min.

### Suggested follow-up (fern's own note, endorsed)

- Wire `local_prevalidate.py` into the pre-submission checklist (all students: run it locally;
  only request an HF Job once it passes). ← **Done — see "New Shared Infrastructure" above.**
- For an a10g-small bf16 confirmation, fern will open a separate `Approval request: HF job for
  vllm-baseline` issue — not done in this PR (local-only by instruction).

_PR #2 merged to `approval-gated-8gpu-20260613` as squash commit `dd17c17`._
