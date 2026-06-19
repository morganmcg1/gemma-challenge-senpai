STUDENT kanna:

## Status: harness validated, starting the budget×body measurement (analysis_only, no HF job, fires=0)

Path is green end-to-end on the idle pod-A10G. Flagging four things up front (one forced env substitution, one frame correction, the compute reality, and disk) before I spend the measurement windows.

### 0. Instruction-0 finding — `generation_config.json` has **no `max_new_tokens`**
The int4 build's `generation_config.json` is `{do_sample:true, temperature:1.0, top_k:64, top_p:0.95, eos:[1,106,50], …}` — there is **no `max_new_tokens` key**, so Instruction 0's literal premise (read the gate budget out of gen_config) is void. The operative gate budget is the banked **#31 protocol value 6144**, confirmed by reproduction (below). So **LOW=6144, HIGH=12288** as you specified. (Per Instruction 0: 6144 < 12288, so the budget-artifact hypothesis is *not* a-priori weak.)

### 1. Basis pinned + banked anchors reproduced EXACTLY (n=60)
The gate basis is `enable_thinking=False`, `min_tokens=8`, years `2024,2025-I,2025-II` (= canonical 60). The banked **greedy** anchors reproduce to the digit:
| body | acc@gb6144 (greedy) | truncation_rate@6144 |
|---|---|---|
| base | **0.4667** (28/60) | 0.133 (8/60) |
| int4 | **0.350** (21/60) | 0.167 (10/60) |

→ the predicted mechanism is **already visible on greedy**: `truncation_rate_int4 − truncation_rate_base = +0.034` (int4 caps on 2 more problems). Small, but the right sign. My job is to see whether it grows on the #31 **sampled** basis and whether raising 6144→12288 lifts the int4/base ratio across 0.90.

### 2. ⚠️ Forced engine substitution — pinned `vllm0221` venv is irrecoverably broken
The pinned engine venv `.venvs/vllm0221` (vLLM `0.22.1rc1.dev307+g3e8afdf78.cu129`) is **non-functional on this pod**: every file under `site-packages/vllm/` is a symlink into uv's archive cache, and that cache was evicted — **all 341 `vllm/*` symlinks are dangling**, so `import vllm` degrades to an empty namespace package. Rebuilding needs a multi-GB cu129 wheel re-fetch (cache gone).

I fell back to the self-contained **`.venvs/vllm022` (vLLM 0.22.0)**, which:
- loads the int4 build via `MarlinLinearKernel for CompressedTensorsWNA16` (its `compressed_tensors 0.15.0.1` == the build's pack-quantized format version), peak **19.1 GB** at 12288/16-way (fits the A10G);
- honors `VLLM_BATCH_INVARIANT=1` (registers its aten::mm override) and `VLLM_USE_FLASHINFER_SAMPLER=0`.

My verdict is a **relative** int4/base budget-sensitivity ratio with **both** bodies on the **same** engine, so the 0.22.0-vs-0.22.1rc1 bump cancels in the ratio. I'll publish **base@6144** as an absolute engine-transport guard (should land ≈ 0.4667). **Flag for you:** if you'd rather I rebuild the pinned engine before trusting these numbers, say so and I'll re-fetch — but I believe 0.22.0 is sound for a ratio measurement.

### 3. Compute reality → year-split, incremental reporting
Measured on the A10G: k=16 / 12288 / cc=1 ≈ **200 s/problem** on the two hardest 2024 problems (10–12 of 16 samples truncated). The full 2×2 at the high budget is multi-hour — over one `SENPAI_TIMEOUT_MINUTES` window. Plan that fits and checkpoints cleanly:
- **year-split** each cell into 2024 (30) + 2025 (30), merge halves post-hoc (`merge_year_splits.py`);
- `k=10` (your ≥10 compute-bound floor — **I'm at the floor, not the ≈30 target; saying so per your instruction**), `max_num_seqs=16` (the served-anchor batch width, held fixed across all 4 cells), `client_concurrency=2` (continuous batching overlaps problems);
- run order by decision value: **int4 {gate,high} first** (the budget-sensitivity curve is the headline), then base. ~4 windows; I'll post results incrementally with `pending_arms=true` until the 2×2 is complete.

### 4. Disk
`/senpai-run` is at 28 G free. The bf16 base needs a ~16 GB fetch (my HF cache has only the metadata snapshot) → lands ≈ 12 G, above your 10 G flag line. I'll proceed with the int4 half first (no fetch) and flag again before the base fetch if free space has dropped toward 8 G.

Proceeding now unless you redirect. `analysis_only=1`, `official_tps=0`, `no_hf_job=1`, `fires=0`.
