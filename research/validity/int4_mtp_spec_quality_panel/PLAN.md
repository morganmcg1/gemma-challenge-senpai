# PR #605 — int4+MTP spec-config 4-eval quality panel (option-B decision evidence)

ANALYSIS-ONLY. Local A10G. NO HF Job, NO served-file change, NO submission, NO fire.
`analysis_only=True`, `official_tps=0`. W&B group `int4-mtp-spec-quality-panel`.

## Question
Does the `int4_g128_lmhead` body + MTP-K7 spec config (fern #597, freerun_seq_exact=0.3125
NOT #319-identical, ~427.7 official-proxy TPS) clear **>=90% of vanilla base on all four
downstream evals**? This is the missing evidence for the human's #481 A/B decision.

- YES -> OPTION-B-FIREABLE (spec keeps quality despite non-identity).
- NO  -> OPTION-B-DEAD (A/B collapses to A, strict-#319 AR-frame only).

## Gate bars (Morgan #579, 90%-of-vanilla-base)
| eval | bar | (implied base) |
|------|-----|----------------|
| MMLU-Pro     | >= 0.605 | 0.672 |
| GPQA-Diamond | >= 0.471 | 0.523 |
| AIME (24+25) | >= 0.090 | 0.10  |
| GSM8K        | >= 0.807 | 0.897 |

## Phase 0 — disk preflight (DONE)
- Disk at pickup: ~235 GB free / 78% used (> 170 GB threshold -> rebuild cleared).
- Drafter `/tmp/qat-assistant` (MTP-K7 head, 183 MB): **PERSISTS** — reuse.
- Body `/workspace/gemma_build/int4_g128_lmhead`: **GONE** (#603 disk purge). Rebuilding from
  `google/gemma-4-E4B-it-qat-q4_0-unquantized` (15.9 GB source, NOT 163 GB) via
  `submissions/int4_g128_lmhead/build_quant.py` (g128 body + untied int4 g128 lm_head).

## Spec serve config (fern #597, `runs/int4g128_k7_bi1_n16`)
- submission: `submissions/int4_mtp_batchinv` (serve.py + attn-group patch + sitecustomize)
- MODEL_ID=`/workspace/gemma_build/int4_g128_lmhead`, DRAFTER_MODEL=`/tmp/qat-assistant`
- NUM_SPECULATIVE_TOKENS=7, VLLM_BATCH_INVARIANT=1, vllm==0.22.0
- **guards:** MAX_MODEL_LEN=**6144** (land #598, NOT 4096), min_tokens=8 (wirbel #541),
  sampling per generation_config.json (T=1.0/top_p=0.95/top_k=64, lewtun #31).
- MAX_NUM_SEQS bumped from 1 -> 16 for eval concurrency (6144 admits ~18-way; quality is
  per-sequence so batching does not bias accuracy under sampling).

## Eval harness
- MMLU-Pro / GPQA-Diamond: `research/validity/downstream_quality_eval/run_eval.py`
- AIME: `research/downstream_quality_aime/aime_eval.py`
- GSM8K: `research/downstream_quality_gsm8k/gsm8k_eval.py`
- client venv `/tmp/eval-serve-venv` (inspect_ai/inspect_evals + stdlib HTTP).

## Coordination
SPEC-config panel = the option-B gate. Complements kanna #579 (AR-config panel — dead pod,
no output). Do NOT touch the AR submission, any served file, body GEMM (#602), attention/graph
(lawine #601), host-tail (fern #604).
