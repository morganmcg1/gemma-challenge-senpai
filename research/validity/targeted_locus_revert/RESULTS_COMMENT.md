STUDENT lawine:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["m6orjqkk"],"primary_metric":{"name":"output_tps","value":211.42},"test_metric":{"name":"divergence_count","value":9729}}

## Results — targeted-locus-revert (surgical force-2D attention, BI=0)

**Verdict: REFUTED in the SERVED path. The surgical force-2D attention patch (the #761 divergence locus) restores byte-exact run-to-run determinism in EAGER (1.0000) but NOT in the production CUDA-graph path (spec-on 0.1875). The decisive control: full BI=1 batchinv IS served-deterministic (1.0000) on the *same* CUDA-graph stack — so byte-exact served identity is real and achievable, but the #761 attention locus is necessary-but-insufficient for it. The surgical revert recovers the speed (+37%, 211 vs 154 TPS) but forfeits the byte-exact property the PR wanted, so it is NOT a strict-quality-safe replacement for `int4_mtp_batchinv`.**

This closes the exact open caveat from my #761 result (caveat #1: *"in-process enforce_eager proxy — NOT the live CUDA-graph served path"*) with a negative answer for the served path.

### Headline metrics

| metric | value | meaning |
|---|---|---|
| **`output_tps` (primary)** | **211.42** | surgical served spec-on median wall TPS — clears the PR's >157 target |
| **`divergence_count` (test)** | **9729** | official `greedy_gate.compare` token divergences vs own M=1 AR ref (target **0** → FAILS) |
| served gate verdict | **INCONCLUSIVE** | spec-ON not self-deterministic run-to-run → gate precondition (PR #38 served wobble) violated |
| surgical-vs-batchinv speedup | **+37.28%** | 211.42 vs 154.01 TPS — the speed side of the hypothesis holds |

### Four-arm design (one changed factor per arm, same deployed int4 Marlin stack)

Self-determinism = run_00 vs run_01, **fresh server reloads** (32 prompts × 512 tok, MAX_NUM_SEQS=1). This is the **precondition** the official self-referential greedy gate requires — a stack that cannot reproduce its OWN decode run-to-run cannot be byte-exact vs any reference.

| arm | env | exec path | self-determ (run0 vs run1) | divergent prompts | wall TPS |
|---|---|---|---|---|---|
| **eager surgattn** | BI=0 + force-2D | `--enforce-eager` | **1.0000** ✅ | 0 / 32 | 47.84 |
| **served surgattn** (spec-on, production) | BI=0 + force-2D | CUDA graphs + inductor | **0.1875** ❌ | 26 / 32 | **211.42** |
| **served surgattn** (M=1 AR ref) | BI=0 + force-2D | CUDA graphs + inductor | **0.5000** ❌ | 16 / 32 | 83.63 |
| **served batchinv** (spec-on) | **BI=1** | CUDA graphs + inductor | **1.0000** ✅ | 0 / 32 | 154.01 |

E[T] (drafter acceptance) is identical across spec-on arms (surgical 3.288, batchinv 3.292), so the 211-vs-154 TPS gap is **pure kernel-path cost** (the BI-tax on matmul/norm reductions), not drafter behavior.

### The decisive logic (why REFUTED, and why it is honest)

1. **EAGER surgattn = byte-exact (1.0000).** Reproduces #761's finding — forcing the 2D single-pass attention path makes the verify-attention split-KV reduction M-invariant, and in eager that is sufficient for run-to-run byte-exactness.
2. **SERVED surgattn = NOT byte-exact (0.1875 spec-on, 0.5000 M=1 AR).** The patch fails in the production CUDA-graph path. 26/32 prompts diverge run-to-run even for plain M=1 AR decode.
3. **SERVED batchinv (BI=1) = byte-exact (1.0000) on the SAME CUDA-graph path.** This is the load-bearing control. It proves two things at once:
   - Byte-exact served identity **is** achievable on this stack (the PR's batchinv baseline holds — confirmed at 154 TPS, matching the stated ~157).
   - The served run-to-run nondeterminism is **NOT** the CUDA-graph/inductor compilation layer (batchinv shares that layer and reproduces). It is the **BI=0 fast-path reductions (matmul / norm)** that global batch-invariance freezes but the attention-only force-2D patch leaves untouched.
4. **Therefore: the #761 attention locus is necessary-but-insufficient for SERVED byte-exactness.** #761 measured **M-invariance** (M=K verify vs M=1 decode, same process); the served gate needs **run-to-run determinism** (fresh reloads) — these are orthogonal. force-2D fixes the former; only global BI=1 fixes the latter in the served path.

The official gate (`greedy_identity_interlock.py --self-referential`) correctly returns **INCONCLUSIVE** (not RED): with the precondition violated, the 9729-token divergence conflates run-to-run noise with the spec-vs-AR signal. The honest read is "uncertifiable," and the root cause is the precondition failure, not a clean identity break.

### Mapping to the PR's expected outcomes
This is the PR's **"Worst case: the op is too intertwined with BI=0 fast paths to patch surgically → report findings, close this direction"** — with a sharper mechanism than "intertwined": the divergence-locus op (attention reduction) is cleanly patchable for M-invariance, but **served byte-exactness depends on a broader set of BI=0 reductions** than the single #761 locus. PR step 5 (MMLU-Pro n=50 quality confirmation) is **not triggered** — it is gated on "if identity passes," and served identity does not pass.

### Reproduce
```bash
# Captures: 2 fresh reloads/arm, 32 prompts × 512 tok, MAX_NUM_SEQS=1, A10G, no HF job
SP=.venv/bin/python
# served surgical (BI=0 + force-2D), spec-on + own M=1 AR ref (SENPAI_REFERENCE_MODE=1)
$SP scripts/validity/greedy_determinism.py --submission submissions/int4_mtp_bi0_surgattn \
  --config default --runs 2 --num-prompts 32 --output-len 512 \
  --out-root research/validity/targeted_locus_revert/served_interlock
# served batchinv control (BI=1)
$SP scripts/validity/greedy_determinism.py --submission submissions/int4_mtp_batchinv \
  --config default --runs 2 --num-prompts 32 --output-len 512 \
  --out-root research/validity/targeted_locus_revert/batchinv_served
# eager surgical screen (--enforce-eager) — the #761 byte-exact path
ENFORCE_EAGER=1 $SP scripts/validity/greedy_determinism.py --submission submissions/int4_mtp_bi0_surgattn \
  --config default --runs 2 --num-prompts 32 --output-len 512 \
  --out-root research/validity/targeted_locus_revert/eager_screen
# cross-arm self-determinism verdict
$SP research/validity/targeted_locus_revert/analyze_served_determinism.py
# official self-referential greedy gate on served captures (the divergence_count)
$SP scripts/validity/greedy_identity_interlock.py --self-referential --skip-capture \
  --spec-root research/validity/targeted_locus_revert/served_interlock/default \
  --ar-root   research/validity/targeted_locus_revert/served_interlock/default__specoff \
  --config default --output-len 512 \
  --report research/validity/targeted_locus_revert/served_interlock_gate.json
# W&B (repo .venv has wandb; no GPU)
$SP research/validity/targeted_locus_revert/wandb_log.py
```

### Run facts
- **W&B:** run_id `m6orjqkk`, group `targeted-locus-revert`, name `lawine/targeted-locus-revert`, project `wandb-applied-ai-team/gemma-challenge-senpai` (verified via `api.run()`).
- **Stack:** deployed `google/gemma-4-E4B-it-qat-w4a16-ct` int4 W4A16 Marlin + MTP drafter, vLLM 0.22.0, single A10G, `MAX_NUM_SEQS=1`, `VLLM_USE_FLASHINFER_SAMPLER=0`.
- **Mem:** model weights 9.86 GiB + KV cache 8.45 GiB, `--gpu-memory-utilization=0.90` (~21.6 GiB allocated of 24 GiB A10G). CUDA graphs ON in all three served arms (graph capture logged); OFF in the eager screen.
- **Elapsed (decode, run_00):** served surgical 79.4 s / served batchinv 106.4 s / served M=1 AR 196.3 s / eager surgical 345.1 s. 16384 completion tokens per run.
- `analysis_only_after_capture=true, official_tps=0, no_hf_job=1, fires=0`. **No HF Job, no submission, no `--launch`.**

### Public evidence / scope framing
- The headline competition number is the **non-strict** leaderboard frontier (~500+ TPS, identity-blind PPL≤2.42 scorer). This experiment is on the **separate internal strict-byte-exact lane** — a quality-safety property the organizer scorer does not require. Non-byte-exactness is not a DQ.
- The mechanism under test (`VLLM_BATCH_INVARIANT`) is vLLM's public batch-invariant-kernels path (the "defeating nondeterminism in LLM inference" line of work); all measurements here are on our own deployed stack, not borrowed numbers.

### What happened — honest analysis
The hypothesis ("force BI=1 for ONLY the #761 locus op → byte-exact served identity at >157 TPS") **fails on the identity half**. The speed half is confirmed (211 TPS, +37% over batchinv), and the eager byte-exactness is confirmed (1.0000, reproducing #761). But the served CUDA-graph path needs more than the attention reduction frozen — the surgical patch leaves BI=0 matmul/norm reductions in the fast path, and those break run-to-run determinism under CUDA-graph capture. The clean batchinv control (1.0000 served) rules out the compilation layer as the cause and confirms the strict lane is reachable only by the full BI=1 mode at its 154-TPS cost. So there is no surgical shortcut to a faster strict submission via attention alone.

### Suggested follow-ups
1. **Identify the residual served-determinism ops by subtraction:** capture run-to-run determinism with BI=0 + force-2D + each *additional* BI=1 op family toggled on one at a time (matmul, then norm) to find the minimal BI=1 subset that recovers served 1.0000. If matmul-only (without the full norm/softmax tax) suffices, that would be a genuinely faster strict candidate — the surgical idea, just at the correct (larger) locus.
2. **Price the minimal-subset TPS:** hand the minimal BI=1 op subset to fern #750's per-kernel-cost axis to see whether it lands above batchinv's 154 TPS — the only way this direction yields a faster strict submission.
3. **CUDA-graph capture-order hypothesis:** test whether the served BI=0 nondeterminism is inductor combo-kernel autotune (`benchmark_combo_kernel`) vs atomic-add reduction order by capturing with `VLLM_DISABLE_COMPILE_CACHE` / fixed autotune — narrows whether the fix is "disable autotune" (cheap) or "fixed-order kernels" (the BI tax).
