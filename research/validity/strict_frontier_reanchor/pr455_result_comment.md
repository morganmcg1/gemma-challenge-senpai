STUDENT lawine:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["0r0ounl8"],"primary_metric":{"name":"reanchored_strict_frontier_tps","value":466.0177160736458},"test_metric":{"name":"ppl","value":2.3772}}

## Results — Independent re-anchor: strict frontier 467.14 + deployed 481.53 non-equivalence

**LOCAL A10G (sm_86), analysis-only. No HF job, no submission, no served-file change.** Fresh hand (lawine), fresh seeds (5,6,7,8,9 — distinct from #412's 0,1,2), fresh process / CUDA-graph capture. **Both load-bearing anchors reproduce; self-test 33/33.**

### Step 1 — strict frontier reproduces 467.14 (within σ_hw)

| metric | value |
|---|---|
| `reanchored_strict_frontier_tps` | **466.02 ± 0.22** (5 seeds: 466.17 / 465.93 / 466.17 / 465.58 / 466.02) |
| `eta_attn_decode` (median, N=5) | 0.033287 (σ 0.00049) |
| drift vs anchor 467.14 | **−1.12 TPS** ( \|drift\| ≪ σ_hw 4.8 → **anchor HOLDS** ) |

**What 466.02 actually is (settled from public W&B, not assumed).** denken #423 (`5a6zq2yz`) is a 1-second *composition* run that consumes `base_467_measured = 467.14` from stark #412 (MERGED to this branch @8cff7c6). #412's number is itself COMPOSED: `strict_frontier = OFFICIAL_TPS / (1 + eta_attn_decode)`, where `eta` is a FA2 varlen-attention microbench delta (fast `num_splits=0` heuristic vs strict `num_splits=1` / M-invariant) over the gemma-4 decode-position band. It is an **isolation of the single component that differs** between the strict and the deployed-fast path — the attention reduction-**order** tax — holding cudagraph/ONEGRAPH/precache/lm_head-prune constant. It is **official-basis** (anchored on the 481.53 official deployed), hence directly comparable to 481.53.

It is **NOT a naive end-to-end serve, and cannot be**: serving with `VLLM_BATCH_INVARIANT=1` *also* disables cudagraph/ONEGRAPH, which would confound the small attention tax with the large graph-disable penalty. The only end-to-end-MEASURABLE strictly-equivalent config is the M=1 AR reference (no speculation), which my #438 measured at 156.20 local / **161.70 official** — ~305 TPS below deployed. So the honest reproduction of denken #423 = re-run #412's census + microbench myself with fresh seeds and re-compose. That is exactly what this card does (the "N≥5 runs" = 5 independent microbench seed estimates; census identity is deterministic, 1/arm).

### Step 2 — deployed 481.53's non-equivalence (the equivalence tax it spends)

Census over 126 prompts / **882 served verify positions**; both arms byte-deterministic (M8-vs-M8 = 1.0) and prompt-isolated (chunk-isolated = 1.0 → order-independent):

| arm | config | identity | flips | flip prompts |
|---|---|---|---|---|
| **deployed (heuristic)** | `VLLM_BATCH_INVARIANT=0`, split-KV `num_splits=0` | **0.996599** (879/882) | **3** | **{11, 18, 118}** |
| strict (pinned) | `VLLM_BATCH_INVARIANT=1`, `num_splits=1` (M-invariant) | 0.998866 (881/882) | 1 | {90} |

- `deployed_identity_fraction = 0.99660`, `deployed_token_flips = 3` — **reproduces the 0.9966 / 3 anchor exactly.**
- **Flips are STABLE, not jitter.** Within-run M8-vs-M8 byte-determinism = 1.0; chunks processed in isolation (so the flip *set* is prompt-order-independent); and my fresh run hit the **identical** {11, 18, 118} that the #381/#405/#412 lineage reported. Mechanism: int4 Gemma-4-E4B is near-tie dense, so at these positions the split-KV reduction order breaks the bf16 argmax tie differently from the M=1 AR path — a **deterministic** flip that recurs every run, not FP noise.
- The strict (pinned) arm closes 2 of 3 (0.99887, 1 residual flip @ prompt 90, a varlen-combine tie). `strict_is_byte_exact_M8 = True`, `fast_is_byte_exact_M8 = False`.

### Step 3 — equivalence tax vs the σ_hw envelope

- **`equivalence_tax_tps` = 481.53 − 466.02 = 15.51 TPS = 3.23× σ_hw (4.8)** → **EXCEEDS** the hardware-noise envelope. The tax is a **real, banked cost**, not noise. (drift vs the 14.39 anchor: +1.12, purely because my frontier landed 1.12 below 467.14.)
- The reanchored frontier itself sits **−1.12 TPS** from 467.14 — **within σ_hw** → the 467.14 floor holds.

### Step 4 — PPL + flags

- `ppl = 2.3772` ≤ 2.42 gate ✓ (local #438 corroboration 2.3767)
- `analysis_only=true`, `no_hf_job=true`, `no_served_file_change=true`, `official_tps=0`
- **`strict_frontier_reanchor_self_test_passes = True` (33/33)**

### Bottom line for the relax-strict-equivalence decision

Both load-bearing anchors are independently confirmed on this pod today. The strict frontier sits at **466.02 ± 0.22** (467.14 ± σ_hw — holds), and the deployed 481.53 buys its **+15.5 TPS** over the strict frontier *only* by spending **3 deterministic reduction-order flips** (identity 0.9966, not 1.0). The equivalence tax is **3.2× the hardware-noise band**, so the "relax-strict-equivalence" trade is a genuine, well-measured cost — not a measurement artifact. The foundation under the escalation is solid.

### Command

```bash
.venv/bin/python research/validity/strict_frontier_reanchor/strict_frontier_reanchor.py --measure \
  --wandb_group equivalence-escalation-anchors --wandb_name lawine/strict-frontier-reanchor
# n_prompts=126(rows), ctx_len=224, n_verify=8 (M=8), gpu_mem_util=0.55, seeds=5,6,7,8,9, iters=50, warmup=10
# drives stark #412 selective_recompute_equivalent_tps.py (in-boundary @8cff7c6) as census×2 + microbench, then composes
```

- **Peak GPU:** 12.25 GB (census arms; microbench 0.006 GB)
- **W&B run:** `0r0ounl8` (group `equivalence-escalation-anchors`)
- **Artifacts:** `research/validity/strict_frontier_reanchor/{strict_frontier_reanchor_results.json, arm_heuristic_result.json, arm_pinned_result.json, microbench_result.json, full_measure.log}`

### What happened

Clean confirmation. The fresh-seed microbench reproduced `eta` tightly (composed-frontier σ = 0.22 TPS, ~0.05% CV), the census reproduced the deployed identity bit-for-bit (same 3 prompts), and the tax cleared σ_hw by 3.2×. The one nuance worth surfacing: **467.14 is a composed attention-tax isolation, not a wall-clock serve.** I re-derived it the same way #412/#423 did (the only faithful reproduction of *that* config) and flagged the framing so the number isn't over-read as an end-to-end TPS.

### Suggested follow-ups

- If the human **relaxes** strict equivalence: the deployed 481.53 is admissible as-is (3 bitwise-tie flips, PPL-neutral). If they **hold the line**: the strict frontier is 466–467, and the next legal lever is the +cb3 / pinned-K recapture stack — separate cards.
- The 1 residual strict flip @ prompt 90 is a varlen-combine tie; if a fully-1.0 *literal* strict identity is ever required, that single position is the last obstacle (operative identity is already 1.0 under the verify arbiter).

### Public evidence used

Internal reproduction card: re-derives the in-repo strict-frontier anchor (denken #423 `5a6zq2yz`, stark #412 method merged @8cff7c6) and the deployed identity (PR #52 `2x9fm2zx`). The config audited (FA2-sliding + split-KV-verify) is the deployed `fa2sw_precache_kenyan` stack, the same family on the public leaderboard (`osoi5-…-fa2sw-precache-…`). This card **reproduces** (does not extend); no submission, no board write.
