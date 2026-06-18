STUDENT kanna:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["cr3c4y3q"],"primary_metric":{"name":"optionb_gpqa_d_sampled_mean","value":0.4586},"test_metric":{"name":"optionb_gpqa_d_sampled_mean","value":0.4586},"verdict":"READING_A_GPQA_FAILS","optionb_gpqa_sampled_mean":0.4586,"optionb_gpqa_sampled_ci":[0.4451,0.4721],"optionb_gpqa_greedy_mean":0.5034,"optionb_gpqa_greedy_ci":[0.4309,0.5758],"pct_of_base_sampled":0.8486,"pct_of_base_sampled_ci":[0.8236,0.8736],"pct_of_ar_body_sampled":0.9190,"finish_length_at_4096":0.0253,"implied_finish_length_at_3072":0.1178,"cap_released_healthy":true,"n_seeds_sampled":10,"n_seeds_greedy":3,"bar":0.4864}

## Results — Option-B GPQA-D Reading-A verdict at the dev307/conc=1 gate point

**VERDICT: `READING_A_GPQA_FAILS` (robust at n=10).** Option-B sampled GPQA-D = **0.4586** (95% t-CI **[0.4451, 0.4721]**, n=10). The entire CI sits below the **0.4864** bar (CI-upper 0.4721 is 0.0143 under). **84.9% of base** sampled (0.5404), CI [82.4%, 87.4%] — wholly below the 90% bar. The mt=4096 cap fix did **not** rescue it: finish-length dropped 13%→2.5% but accuracy stayed put, so the low score is **genuine**, not truncation-depressed.

### Sampled GPQA-D — the Reading-A axis (per lewtun #31, downstream evals use sampled params)
| metric | value | reference / bar |
|---|---|---|
| **mean accuracy (n=10)** | **0.4586** | base sampled **0.5404** (ubel #628 `ilg4z6e9`) |
| 95% t-CI | **[0.4451, 0.4721]** | bar **0.4864** (≥90% of base) |
| between-seed SD | 0.0189 | — |
| **% of base sampled** | **84.9%** [82.4%, 87.4%] | bar **90%** → **FAILS** (CI all below) |
| % of AR-body sampled | 91.9% [89.2%, 94.6%] | AR-body **0.4990** (ubel #638) |
| per-seed acc | 0.4899, 0.4444, 0.4747, 0.4495, 0.4747, 0.4747, 0.4293, 0.4545, 0.4495, 0.4444 | all clean (`n_error=0`, `n_empty=0`, scored 198/198) |

Seeds = [12345, 13579, 23456, 34567, 45678, 56789, 67890, 78901, 89012, 90123] → **n=10 × 198 = 1980 evals**, CI-comparable to ubel #638 (AR-body 0.4990, n=1980) and lawine #639 (official g32 0.5056, n=1980).

### Greedy GPQA-D — health read (capped n=3, the strict gate axis)
| metric | value | reference |
|---|---|---|
| mean accuracy (n=3) | **0.5034** [0.4309, 0.5758] | base greedy **0.4899** (ubel #628 `g3cig1xo`) |
| % of base greedy | 102.7% | clears the bar |
| per-seed acc | 0.5202, 0.5202, 0.4697 | clean |

Greedy clears comfortably; the binding Reading-A read is the **sampled** axis, which fails.

### Cap-artifact confirmed → gate point is genuinely healthy (`cap_released_healthy=true`)
The #631 ~13% finish-length at mt=3072 was a **cap artifact**, not a crater:
- finish_length@**4096** = **2.5%** (sampled mean; greedy 2.5%)
- back-derived implied finish_length@**3072** = **11.8%** greedy / 10.8% sampled ≈ #631's 13.1% anchor
- ctok p95 ≈ 3.4k–3.8k, ctok max = 4096 (only ~2–8 items/seed hit the 4096 cap)

So at mt=4096 the cap releases (13%→2.5%) and the accuracy **does not move up** — the truncation was never the cause of the low score. **Decisive #481 number:** Option-B sampled does **not** rise toward the AR-body's 0.499 at mt=4096; it lands at 0.4586 (91.9% of AR-body, CI-upper 0.4721 still < 0.499). The spec stack carries a real ~8% relative GPQA-D quality gap vs the AR body — it is not a measurement artifact.

### Comparison vs the PR baseline
| source | Option-B GPQA-D sampled | % of base | verdict |
|---|---|---|---|
| fern #629 (single-seed) | 0.4652 | 86.1% | FAILS |
| **this PR (n=10, mt=4096)** | **0.4586** | **84.9%** [82.4%, 87.4%] | **FAILS (robust)** |

The multi-seed read lands **slightly below** fern's single seed and the CI confirms the failure is robust — not a single-seed fluke, and not lifted by the 4096 cap fix.

### Two confounds cleared before the read (both flagged + blessed earlier in-thread)
1. **model_len 6144→8192:** mt=4096 + idx-127's 2429 input tokens = 6525 > 6144 → HTTP-400 → `score_on_error` miss (a ~0.5% headwind landing right at the bar). Bumped to 8192 (KV-capacity only at conc=1+BI=1; the 197 fitting items are numerically unchanged). Collision cleared (idx-127 now 200; `n_error=0` on all 13 runs).
2. **Determinism:** the gate point is **answer-level** near-deterministic (#631 union 1/198), not byte-deterministic — intrinsic dev307 + int4-Marlin-GEMM + greedy-A10G. Each `--seed` is a deterministic choice-shuffle run once; the multi-seed t-CI conservatively absorbs within-seed decode variance into the between-seed SD. Documented caveat, not the operative axis for a sampled read.

### Config / exact commands
**Server (Option-B):** int4_g128_lmhead body + Gemma4-MTP K=7 drafter, BI=1, dev307, conc=1, model_len 8192:
```
.venv/bin/python research/validity/int4_mtp_spec_quality_panel/serve_spec.py \
  --port 8000 --drafter google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant \
  --k 7 --max-model-len 8192 --max-num-seqs 16 --batch-invariant 1 \
  --max-num-batched-tokens 2048 --engine dev307
```
(body served as vLLM `--model /workspace/gemma_build/int4_g128_lmhead`, `--speculative-config {K=7 drafter}`, `--gpu-memory-utilization 0.90`.)

**Gate (per seed, resumable one-cell-per-invocation):**
```
research/validity/gpqa_gate_readingA_4096/_launch_seed.sh <seed> <greedy|sampled>
# → run_gate.py: gpqa_diamond, conc=1, max_tokens=4096, min_tokens_guard=8,
#   sampled = generation_config.json (temp 1.0 / top_p 0.95 / top_k 64)
```

### Run metadata
- **W&B:** `cr3c4y3q` (group `gpqa-gate-readingA-4096-kanna`), state **finished** (clean; the keepalive thread ended the false-`crashed` flapping).
- **Peak GPU:** **19.62 GB** (A10G 24 GB; no OOM across the ~12h panel).
- **Completed:** 198/198 scored on every one of the 13 runs (10 sampled + 3 greedy), `n_error=0`, `n_empty=0`.
- **Guardrails held:** `analysis_only=true`, `official_tps=0`, LOCAL A10G, **no HF Job / no submission**, live submission untouched. (No HF `summary.json` — this is a local quality gate, not a benchmark run.)

### Bug fixes / hardening shipped in this PR (flagging for review)
- **`run_gate.py` W&B keepalive thread** — 60s daemon `heartbeat/*` logger so the resume-by-id run doesn't heartbeat-timeout to false `crashed` while `subprocess.run()` blocks ~45 min in the eval. Confirmed effective: `cr3c4y3q` stayed `running` from seed 89012 onward and closed `finished`.
- **`_launch_seed.sh` double-launch guard** — refuses if any `run_gate.py`/`run_eval.py` is alive (not just the `_gate.pid`-recorded one), enforcing the conc=1 single-client invariant regardless of how a cell was started. Added after a stray direct `run_eval.py` co-ran with the driver for ~27 min on seed 12345 sampled; that contaminated seed was **discarded** and re-run clean.

## What happened — honest analysis
The Reading-A GPQA leg for Option-B **fails robustly**. Three things make this a clean call rather than a marginal one:
1. **It's not a truncation artifact.** The cap-artifact hypothesis was correct about finish-length (13%→2.5% at mt=4096) but the accuracy did not rise with it — Option-B genuinely scores ~0.459 sampled. The gate point is healthy; the model is the limit.
2. **It's not a single-seed fluke.** n=10 at 1980 evals puts the whole CI below the bar (CI-upper 0.4721 < 0.4864), and below fern #629's single-seed 0.4652. The between-seed SD (0.019) is small.
3. **It's a real spec-vs-body gap.** 91.9% of the AR body's 0.4990 (CI-upper 0.4721 < 0.499) — the K=7 spec stack costs a genuine ~8% relative on GPQA-D that the cap fix does not recover. Greedy clears (0.5034) because greedy GPQA-D is easier here, but the binding sampled axis does not.

## Suggested follow-ups
- **Denominator picture for #481:** with Option-B sampled = 0.4586 (91.9% of AR-body) confirmed at mt=4096, the remaining Reading-B question is *where* the ~8% goes — drafter accept-rate vs verifier disagreement on long-chain GPQA items. A per-item agreement diff (Option-B vs AR-body, same seeds) would localize it.
- **Is the gap K-dependent?** A short K (e.g. K=3) sampled GPQA-D mini-panel (n=3–5) at this same gate point would test whether shrinking the draft window narrows the quality gap toward the AR body — i.e. whether the failure is a speculation-depth artifact or intrinsic to the int4 body under sampling.
- **Reading-A panel close:** this is the GPQA leg only (I own GPQA; denken #637 AIME, fern MMLU-Pro+GSM8K). With GPQA failing, the 4-bar panel cannot pass on GPQA regardless of the other three — worth surfacing in the Reading-A rollup.
- **Cherry-pick the keepalive + launch-guard hardening** to the advisor branch if other students run long resume-by-id panels — both are run-agnostic.
