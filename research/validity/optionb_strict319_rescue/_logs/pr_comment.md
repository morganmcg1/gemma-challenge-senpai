STUDENT stark:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["ukiyyuca"],"primary_metric":{"name":"rescued_break_rate","value":0.0},"test_metric":{"name":"rescued_wall_tps_projected","value":139.2035572197178},"verdict":"STRICT_319_RESCUED__TPS_VIABLE","rescued_break_rate":0.0,"unrescued_break_rate":0.0011400071250445315,"min_tau_flag_for_zero_breaks":0.5,"flag_trigger_rate":0.07801923762023513,"rescued_wall_tps_projected":139.2035572197178,"rescued_wall_tps_is_projection":true,"rescued_beats_126":true,"analysis_only":true,"official_tps":0}

## Results — Option-B strict-#319 rescue (gap-flagged M=1-recompute acceptor)

**VERDICT: `STRICT_319_RESCUED__TPS_VIABLE`.** The near-tie-deterministic verify acceptor restores **strict byte-exact #319 (τ=0)** for Option-B at a **projected 139.20 local wall_tps (+10.1% over the 126.378 locked rung)** — so the residual int4-grid-tie family is rescued with no human tolerance-ruling round-trip required.

W&B run: [`ukiyyuca`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/ukiyyuca) · group `optionb-strict319-rescue-stark` · `analysis_only=true`, `official_tps=0`, **NO HF Job / NO submission**.

### PRIMARY — the identity test (decisive, scan-authoritative)

Teacher-forced per-step along the **real M=1 greedy-AR trajectory**, M=8 verify distribution read via the validated #381/#622 **chunk-read** geometry (`chunk_isolated_fraction=1.0` → every scored position is genuinely size_m=8). **14035 positions across 127 prompts** (> the 6531-token #622 scale → power to catch a single break).

| deliverable | value |
|---|---|
| **`rescued_break_rate`** (vs M=1 AR, τ_flag=0.5) | **0.0** (0 / 14035) |
| rescued_break_rate 95% UB (rule-of-three) | 2.14e-4 |
| **`unrescued_break_rate`** (contrast, reproduces #622) | **0.001140** (16 / 14035), CI95 [7.0e-4, 1.85e-3] |
| **`min_tau_flag_for_zero_breaks`** (PR sweep {0.5,0.75,1.0}) | **0.5** |
| min_tau_flag (fine sweep, cost-optimal) | 0.3 |
| **`flag_trigger_rate`** @ τ=0.5 | **0.07802** (7.80%) |

**Soundness:** `rescued_break_rate = 0` along the real AR trajectory ⇒ at every position the rescued acceptor emits the AR token (flagged → M=1 recompute = AR token by construction; un-flagged → no flip remains) ⇒ by teacher-forcing induction the free-running rescued stream is **byte-identical to pure M=1 AR**, i.e. strict τ=0. This is the PR-permitted faithful Python reconstruction over real generated trajectories.

### τ_flag frontier

| τ_flag | flag_trigger_rate | rescued_breaks | projected wall_tps |
|---|---|---|---|
| 0.2 | 0.03570 | **1** (leak) | — |
| 0.25 | 0.03684 | **1** (leak) | — |
| **0.3** | 0.05529 | **0** | 142.78 |
| **0.5** | 0.07802 | **0** | **139.20** |
| 0.75 | 0.11820 | 0 | 133.30 |
| 1.0 | 0.15383 | 0 | 128.48 |

All 16 flips are sub-0.5-nat int4 grid-ties: **flip_gap median 0.125, max 0.25 nat**; `flip_margin_max = 0.25`; **`n_m1_token_outside_topk = 0`** (every divergence's M=1 token is inside the M=8 top-k — bounded, as #622 predicted). The single flip at gap=0.25 is the binding constraint (caught at τ≥0.3). τ_flag=0.5 sits at 2× the observed max gap → comfortable margin. Even the most conservative zero-break τ=1.0 has ftr=0.154, **below the 0.170 break-even ftr**, so every zero-break τ beats 126.378.

### SECONDARY — TPS cost (projection)

`rescued_wall_tps = 1 / (1/152.291 + flag_trigger_rate / 126.378)` — un-rescued spec time (land #623 = 152.291) plus one M=1 forward (from the 126.378 AR rung) per flagged position.

- **`rescued_wall_tps_projected` = 139.20** at τ=0.5 (ftr 7.80% → ~1 extra M=1 forward per **12.8** verified tokens). **`rescued_beats_126 = true`.**
- This is a **projection, not a measurement** (flagged as such). It is **conservative**: it charges each recompute a full cold M=1-AR-token; the in-engine recompute reuses the verify KV cache and is cheaper, so true rescued TPS ≥ 139.20.
- Net trade: un-rescued Option-B was +20.5% (152.291) **but needed a human tolerance ruling**; the rescue gives **+10.1% (139.20) AND strict τ=0, ruling-free**. ~8.6% of the gain is spent to buy strictness.

### Controls (all green)

`attn_is_batch_invariant = true` · `aten_mm_bitexact_M1_vs_M8 = true` (lm_head GEMM M=1 vs M=8 bit-exact) · `chunk_isolated_fraction = 1.0` · BI=1 both sides · int4 W4A16 body (`google/gemma-4-E4B-it-qat-w4a16-ct`), lm_head not int4 · vLLM 0.22.0, TRITON_ATTN (forced for Gemma4 heterogeneous head dims) · peak GPU **12.2 GB**.

### CONFIRMATORY literal free-run — INCONCLUSIVE (non-authoritative, disregarded)

I also ran a literal free-running byte-compare (2 prompts × 96 tokens, τ=0.5). Its size_m=8 emulation uses 8 identical batched copies, and the harness's built-in faithfulness self-check **FAILED**: 8-copy vs chunk-read agreement **0.667** (28/42), `emul_faithful = false`. Root cause: prefix-caching collapses the 8 identical copies, so the body GEMM is **not** genuinely size_m=8 — the 8-copy argmax/gap diverge from the deployed-faithful chunk-read M=8. Its break counts (`rescued_freerun_break_rate=0.396`) are therefore **emulation artifacts, not real verify divergences**, and the scan-authoritative verdict logic correctly **disregards** them (`freerun_confirms_strict_identity=false`; the design forbids an unfaithful emulation from flipping the verdict to INCOMPLETE). Directional sanity only: one of the two prompts was literally **byte-identical** under rescue (0 breaks) vs 82 un-rescued cascade breaks — but I am **not** leaning on these numbers. The authoritative proof is the scan's induction above.

### Comparison vs baselines (from PR body)

| stack | wall_tps | strict-#319 (τ=0) | note |
|---|---|---|---|
| Locked rung `int4_g128_lmhead` (#4) | 126.378 (official) | ✅ byte-exact | PPL 2.019, 128/128 |
| Un-rescued BI=1 Option-B (land #623) | 152.291 (local) | ❌ needs tolerance ruling | 6/6531 int4-tie residual |
| **Rescued Option-B (this PR, τ=0.5)** | **139.20 (local, projected)** | **✅ byte-exact (0/14035)** | ruling-free |

### Command / environment

```bash
cd target/
# PRIMARY scan (authoritative) — orchestrator launches the --phase scan subprocess with
# env CUDA_VISIBLE_DEVICES=0 VLLM_BATCH_INVARIANT=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
uv run python research/validity/optionb_strict319_rescue/optionb_strict319_rescue.py \
  --n-prompts 128 --ctx-len 224 --traj-len 512 \
  --wandb_group optionb-strict319-rescue-stark --wandb_name stark/optionb-strict319-rescue
#   -> internally: --phase scan --gpu-mem-util 0.55 --max-batched-tokens 8192 --verbose-k 5

# CONFIRMATORY literal free-run (inconclusive — emulation unfaithful)
... --phase freerun --n-prompts 2 --ctx-len 224 --max-new 96 --tau-flag 0.5 \
    --gpu-mem-util 0.55 --max-batched-tokens 8192
```

Single A10G (CUDA_VISIBLE_DEVICES=0). Scan wall ~62 min (128 prompts ≈ 29 s/prompt). Artifacts: `research/validity/optionb_strict319_rescue/{scan_result.json,freerun_result.json,optionb_strict319_rescue_report.json}`.

### What happened

The hypothesis held. Every observed verify divergence is a bounded sub-0.5-nat int4-Marlin grid-tie (max 0.25 nat, all M=1 tokens inside the M=8 top-k), so a **single cheap top-1/top-2 gap scalar catches 100% of them** before they can seed a free-running cascade. Flagging `gap < 0.5` and recomputing those ~7.8% of positions at M=1 yields `rescued_break_rate = 0` over 14035 positions (rule-of-three 95% UB 2.1e-4) → strict τ=0 by induction. Because the break-even flag rate is 17.0% and the measured rate is 7.8%, the recompute cost leaves the stack comfortably above the 126.378 locked rung (projected 139.20). The literal free-run leg was inconclusive purely due to an unfaithful 8-copy size_m=8 emulation (prefix-cache collapse), which the harness self-detected and correctly excluded; it does not affect the scan-authoritative result.

### Suggested follow-ups

1. **In-engine acceptor + measured TPS.** Implement the gap-flag-and-recompute inside the MTP spec-verify path and measure `rescued_wall_tps` directly (replacing the conservative projection). Expect ≥139.20 since the in-engine recompute reuses verify KV.
2. **Faithful literal free-run.** Replace the freerun's 8-copy `m8_dist` with the chunk-read M=8 (or disable prefix caching for the copies) so the literal byte-compare is width-8-faithful and can serve as a clean second proof alongside the scan induction.
3. **τ_flag = 0.5 is the safe ship point** (2× observed max gap, 0 leaks, +10.1%); τ=0.3 trims cost to 142.78 with 0 leaks but only 0.05-nat margin over the binding flip — keep 0.5 for safety headroom.

### Public evidence used

Internal Option-B chain on `approval-gated-8gpu-20260613`: builds directly on stark #622 (`15g9q3wc`, the 6/6531 int4-tie residual being rescued), land #623 (152.291 un-rescued local anchor), denken #626 (residual-immaterial in graded quality), and locked rung #4 (126.378 strict-#319). No external leaderboard method reproduced; this is a strict-identity/TPS analysis, not a submission.
