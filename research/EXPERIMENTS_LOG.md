# SENPAI Research Results

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
