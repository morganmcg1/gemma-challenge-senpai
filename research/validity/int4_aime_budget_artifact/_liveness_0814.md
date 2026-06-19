STUDENT kanna:

## Liveness — Option-A landed; greedy non-reconciliation root-caused to a batch-width engine artifact; testing the fix

Answering your 07:33Z proof-of-life. **Alive, pod healthy, GPU local + currently mine, disk 18G free.** `analysis_only=1`, `official_tps=0`, `no_hf_job=1`, `fires=0`. The Option-A steer landed — I'm on the **greedy basis**, not Option B.

**Why no result run yet:** the greedy 2×2 does **not** reconcile to the banked anchors on the substitute engine (`.venvs/vllm022`) at the client-concurrency I was using. Both bodies collapse:

| body | engine/config | acc@6144 greedy | banked anchor |
|---|---|---|---|
| base bf16 | `.venvs/vllm022`, eager, **cc=16** | ~0.18–0.22 | **0.4667** |
| int4 | `.venvs/vllm022`, eager, **cc=16** | 0.067 | **0.350** |
| int4 | `.venvs/vllm022`, compile-ON, cc=16 | ~0.05 | 0.350 |

**Root cause (isolated this session):** the banked **int4** greedy anchor (0.350) was run at **`client_concurrency=1`** (serialized; banked base used cc=16) on the now-**gone** `/tmp/vllm0220-srv` engine. My greedy attempts ran **cc=16**. On `.venvs/vllm022` the **batched (cc>1) greedy decode path is corrupt for BOTH bodies** — the `/tmp/vllm0220-srv` engine that tolerated cc=16 batching is irrecoverable (same dead-venv class I flagged at 01:26Z). This is *engine-wide at cc16*, not int4-specific (base collapses too), which is exactly the signature of a broken batched/batch-invariance path rather than a quant issue.

**The faithful fix = cc=1 (batch width 1), the banked int4 config.** At cc=1 there is no batch to corrupt. Smoke (int4 cc=1, 12 problems @6144) **in flight right now** on the int4 server (`pid 772859`, eager, BI=1, mml=13312). If it reconciles (coherent decode, no `qlql…` repetition salad), I run the full greedy 2×2 at cc=1 — int4 {6144,12288} + base {6144,12288} — and deliver the BUDGET_ARTIFACT vs REAL_PRECISION_LOSS verdict. **I'll drop the W&B run id + the 6144 reconciliation the moment the smoke lands.**

No HF job, no fire, served file untouched. Holding the line.
