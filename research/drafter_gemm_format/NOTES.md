# PR #786 — MTP drafter GEMM format: is the q4_0 drafter BW-optimal on bi0?

Baseline: bi0 = `int4_mtp_bi0_surgattn`, official TPS 218.02, PPL 2.0058, 128/128,
W&B `s63tb03x`. Drafter = `google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant`.

## Step 1 — drafter serving format (code-inspection, no GPU)

**Finding: the drafter is served dense bf16, by design and by checkpoint.**

1. `serve.py` builds `--speculative-config` as
   `{"model": drafter_model, "num_speculative_tokens": 6}` with **no `quantization`
   key** (serve.py:122-129). vLLM therefore infers the drafter format from the
   checkpoint config.
2. Drafter `config.json` (`Gemma4AssistantForCausalLM`, model_type
   `gemma4_assistant`) has **`dtype: bfloat16`** and **no `quantization_config`**.
   → vLLM loads it as dense bf16 (2 bytes/param). The "q4_0" in the repo name
   refers to the QAT checkpoint it was *matched to*; the assistant weights
   themselves are stored **unquantized** ("…-unquantized-assistant").
3. serve.py docstring (lines 16-19) already documents the deliberate decision:
   "The draft head is left in its native bf16/centroid path (never
   force-quantized): the assistant's masked-embedding centroid logits have no
   packed-weight branch, so quantizing it would force the ~11x-slower dense path."

### Drafter weight footprint (why bf16 is plausibly already fine)

From `config.json.text_config`:
- `num_hidden_layers: 4`, `hidden_size: 256`, `intermediate_size: 2048`,
  `num_attention_heads: 4`, `head_dim: 256`, `num_kv_shared_layers: 4`
  (Q-only, shares the target KV cache → no own K/V projection cost).
- Per-layer Linear params ≈ q_proj(256×1024) + o_proj(1024×256) + mlp
  gate/up/down(3×256×2048) ≈ 2.1M. × 4 layers ≈ **8.4M params ≈ 16.8 MB bf16**.
- Output: centroid head (`num_centroids: 2048`, `centroid_intermediate_top_k: 32`,
  `vocab_size: 262144`, `tie_word_embeddings: true`). NOT a dense 256→262144
  matmul — this is the "masked-embedding centroid" path. The tied embedding
  (262144×256 ≈ 67M params, 134 MB bf16) is read by **gather** (active rows
  only) on input, not a per-pass GEMM.

So the per-draft-pass *dense GEMM* weight read is only ~16.8 MB bf16. The target
int4 W4A16 verifier read per step is multi-GB (profiler: ~92% GEMM). Quantizing
the 16.8 MB drafter body to int4 (→ ~4.2 MB) saves a few MB per pass — a small
fraction of total step BW — and at M=1 (decode) the tiny drafter GEMMs are
latency-bound, not weight-BW-bound, where int4 Marlin dequant overhead can
exceed the BW saving. **Prior: likely null/negative. Measuring to confirm.**

## Public evidence used (challenge board digest, 2026-06-20)
- 133 agents; frontier ~505 TPS (sparkgemma-s46b `w192-ctk48-noprecache`,
  vidraft/osoi5 family), PPL ~2.39. These stacks run the **same** Gemma4 MTP
  assistant drafter and describe it as "output-neutral (MTP drafter proposes;
  int4 target greedy-verifies token-identically)" — independent confirmation
  that drafter precision cannot break greedy identity at temp=0.
- The drafter speed lever the frontier actually tunes is **CENTROID_TOP_K**
  (CTK44/48) — the drafter's centroid-head top-k — NOT drafter weight
  quantization. No top leaderboard method quantizes the drafter weights. In a
  hyper-competitive field this is indirect evidence that drafter weight-quant is
  not a known win. This PR tests it directly on bi0 (extending that evidence).
- bi0 (218 TPS / PPL 2.0058) is stark's quality-safe rung, not the public
  frontier; the gate here is vs the bi0 control, local-only.

## Step 2 — drafter vs verifier pass cost (GPU) — MEASURED
Tooling: reuse the shipped `steptime_patch.py` (wraps `Gemma4Proposer.propose`
= drafter, and `GPUModelRunner.execute_model` = verify) + `serve_profile.parse_steptime`.
Control run = bf16 drafter, 16 prompts x 256 tok. W&B `h1nsfad1`.

| metric | control (bf16) |
|---|---|
| verify_gpu_ms (execute_model, M=7) p50 | **11.752** |
| drafter_gpu_ms (propose = K=6 × M=1) p50 | **2.434** |
| drafter / verify ratio | **0.207** |
| drafter frac of GPU-busy | **17.2 %** |
| draft acceptance rate | 0.3856 (3054/7920) |
| E_accept (mean accept length) | 3.314 |
| steady decode TPS (local A10G) | 210.1 |

**Why the BW premise is refuted.** The 2.434 ms drafter cost is *latency-bound*,
not weight-BW-bound: it is 6 **sequential** M=1 micro-launches over a tiny 4-layer
body (~16.8 MB bf16). At A10G ~600 GB/s realizable BW the weight read is ~0.028 ms
*per pass* (~0.17 ms for all 6) — under ~7 % of the 2.434 ms; the rest is kernel
launch + tiny-GEMM latency at M=1. Quantizing the body to int4 could save at most
~12.6 MB/pass of that ~0.17 ms while *adding* Marlin dequant overhead → ≤ ~1 % of
the decode cycle even in the best case, and plausibly net-negative. The drafter is
already cheap; the verifier (int4 W4A16 Marlin, ~92 % GEMM) is the real budget.

## Step 3 — drafter quantization acceptance test (GPU) — MEASURED: SILENT NO-OP
`SpeculativeConfig.quantization` field EXISTS in vllm 0.22.0 and is plumbed into
the draft `ModelConfig(quantization=...)` (config/speculative.py:680). But for the
`gemma4_mtp` drafter it is a **silent no-op** — vLLM accepts the key (no error,
server starts), yet attaches **no** quant kernel to the drafter, which stays bf16.

| label | spec_quant | server | drafter_gpu_ms | accept rate | greedy | decode TPS | W&B |
|---|---|---|---|---|---|---|---|
| control | bf16 | ✅ | 2.434 | 0.385606 (3054/7920) | — (ref) | 210.1 | h1nsfad1 |
| gptq_marlin | gptq_marlin | ✅ | 2.409 | 0.385606 (3054/7920) | IDENTICAL | 210.6 | hfm6qflx |
| awq_marlin | awq_marlin | ✅ | 2.406 | 0.385606 (3054/7920) | IDENTICAL | 210.6 | ify0thuk |
| fp8 | fp8 | ✅ | 2.408 | 0.385606 (3054/7920) | IDENTICAL | 210.6 | bmfwaw2j |

Proof it is a no-op, not a real quantization:
1. **Accept rate byte-identical** (3054/7920) across all 4 runs → the drafter
   proposed byte-identical tokens → its weights were byte-identical (bf16) in
   every run. fp8/int4 would perturb logits → change proposals → change accept.
2. **drafter_gpu_ms unchanged** (2.41 ± 0.01) — no speedup, no dequant slowdown.
3. **Server log kernel census**: exactly **one** quant kernel instantiated in
   every run — `MarlinLinearKernel for CompressedTensorsWNA16` — and that is the
   int4 **target**. No `Fp8`/`AWQ`/second-`Marlin` kernel ever appears for the
   drafter. (The requested key shows only in `non-default args`; the engine's
   `SpeculativeConfig(...)` repr never carries it.)

Mechanism: the `…q4_0-unquantized-assistant` checkpoint is dense bf16 with **no**
`quantization_config` and no packed weights, and its centroid masked-embedding head
has no packed-weight Linear branch (serve.py docstring) — so gptq/awq_marlin find
nothing to attach to, and online fp8 is not wired through the MTP proposer's
load path. The marlin lever the PR asked about does not even produce an *error* to
document; it simply does not take.

## Verdict — NULL (valid finding)
- Drafter serves **bf16 by design + by checkpoint** (Step 1).
- It is only **17.2 %** of GPU-busy and **latency-bound at M=1** (Step 2) — the BW
  premise (bf16 drafter is a disproportionate share of the 92 % GEMM budget) is
  refuted; the drafter dense-GEMM weight read is ~0.17 ms/step.
- No `--speculative-config quantization=` value (gptq_marlin/awq_marlin/fp8) changes
  the drafter on vLLM 0.22.0 — all are silent no-ops (Step 3). Gate: no TPS gain
  (210.5 ≈ control 210.1, within noise), so **not fire-worthy**; greedy stays
  IDENTICAL and PPL is drafter-invariant regardless.
- Independent public-frontier evidence agrees: top stacks tune CENTROID_TOP_K, not
  drafter weight-quant (no leaderboard method quantizes the drafter).

## Reproduction (bi0 kept byte-pristine)
The bi0 submission was reverted to pristine for this null. To re-run:
```
cp research/drafter_gemm_format/steptime_patch.py submissions/int4_mtp_bi0_surgattn/
git apply research/drafter_gemm_format/bi0_diagnostic_hooks.patch
CUDA_VISIBLE_DEVICES=0 <vllm022-python> research/drafter_gemm_format/measure_drafter.py \
  --label control                                   # bf16 control
CUDA_VISIBLE_DEVICES=0 <vllm022-python> research/drafter_gemm_format/measure_drafter.py \
  --label fp8 --spec-quant fp8 \
  --reference research/drafter_gemm_format/runs/control/decode_outputs.jsonl
uv run python research/drafter_gemm_format/log_results_wandb.py <labels...>   # W&B (project venv)
```
The patch re-adds the env-gated `SPECULATIVE_QUANTIZATION` knob (serve.py) and the
`STEPTIME=1` probe import (sitecustomize.py); both are inert unless their env var
is set. LOCAL A10G only — no HF Job.
