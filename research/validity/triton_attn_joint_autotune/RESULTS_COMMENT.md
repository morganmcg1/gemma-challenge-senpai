STUDENT wirbel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["e5n9a2dc","tfejmv3o"],"analysis_only":true,"no_hf_job":true,"no_served_file_change":true,"official_tps":0,"existing_autotune_config_count":0,"autotune_realized_tps_delta":15.86,"autotune_frontier_tps":483.0,"joint_beats_numstages_only":true,"autotune_beats_deployed_481":true,"realized_crossing_481_proven":false,"ppl":2.3772,"self_test_passes":true,"primary_metric":{"name":"autotune_frontier_tps","value":483.0},"test_metric":{"name":"self_test_passes","value":1}}

## Results

**One-line verdict:** the joint Triton autotune for the served (M=8, head 256/512, GQA 8/2) sm_86 verify shape finds a **byte-exact** config — **`BLOCK_M 16→4` + `num_stages 3→2`** (`{block_m:4, tile:32, warps:4, stages:2}`) — that models **+15.86 TPS over 467.14 → 483.00** (Amdahl @ realistic ctx=512); the joint optimum **differs from** my #428 num_stages-only sweep (1.528× vs 1.408× isolated) and the Amdahl frontier **crosses the deployed 481.53 — but that crossing is a MODELED upper bound, not a proven wall result** (3.03× isolated-vs-in-graph over-count; realized band [0, +15.86]; `realized_crossing_481_proven=False`). This is the **direct analog of cb3's modeled +15.60** (realized-NULL on the served kernel, #437): a modeled lift that needs the served-kernel wall test before it counts.

This is **analysis-only / local profiling** — no served-file change, no HF job, no submission. `official_tps=0`.

### (1) Inventory — `existing_autotune_config_count = 0`
The served `kernel_unified_attention` (vLLM TRITON_ATTN, `triton_unified_attention.py`) is a **bare `@triton.jit`** (`jit_type=JITFunction`), **not** `@triton.autotune` — a single BLOCK_M *heuristic* config (`16 if nqpkv≤16`), Triton-default `num_warps=4`/`num_stages=3`. So there is **no existing joint sweep** to be dense; the lever is fully open. I swept **81 full-grid / 27 byte-exact-candidate configs per shape × 8 shapes** (BLOCK_M∈{4,8,16}, BLOCK_N/tile∈{16,32,64}, warps∈{2,4,8}, stages∈{2,3,4}).

*Lynchpin (deployed default confirmed):* on all 8 shapes the **real wrapper wall-clock == forced-(warps=4, num_stages=3)** to <0.1µs (`wrapper_matches_deployed=True`), so the bare-jit default genuinely IS `num_stages=3` and `s2` is real recoverable slack.

### (2) Joint autotune — `joint_beats_numstages_only = True`
Per-call kernel µs (CUDA-graph replay, ≥5 reps, mean; byte-exact gated by `torch.equal` maxdiff==0.0):

| shape (M=8, ctx=512) | deployed (bm16/s3) | best byte-exact | speedup |
|---|---|---|---|
| sliding head-256 (×35 layers) | 52.85 µs | **bm4**/s2 = 32.60 µs | **1.62×** |
| global head-512 (×7 layers) | 65.24 µs | bm8/s2 = 52.39 µs | 1.25× |

A single **deployable** config must be byte-exact in *both* head dims (one launch-config serves all 42 layers). Layer-weighted (35:7) total:

- **default** bm16/t32/w4/s3 = **2306.3 µs**
- **best joint byte-exact** `bm4/t32/w4/s2` = **1509.8 µs → 1.528×** (byte-exact, maxdiff **0.0**)
- best **num_stages-only** (#428 axis) bm16/t32/w4/s2 = 1638.4 µs → 1.408×

→ **the joint optimum beats the num_stages-only sweep**: BLOCK_M 16→4 recovers an extra **128.6 µs/step weighted** that the single-axis sweep could not see. **Why:** the deployed `BLOCK_M=16 → BLOCK_Q=4` launches only **~6 thread-blocks** (`q.shape[0]//4 + 1 = 3` × 2 kv-heads) for the M=8 single-stream 2D-path verify (`max_seqlen_q>1 → 2D`, not 3D split-KV) → severe SM under-occupancy on A10G; `BLOCK_M=4 → BLOCK_Q=1` **triples blocks to ~18**. The default heuristic was tuned for head_dim=64/128 / batched decode, not this shape.

### (3) Wall translation — `autotune_realized_tps_delta = +15.86`, `autotune_frontier_tps = 483.00`
Amdahl on 467.14 with f=T_ATTN_FRAC=0.09507 (#408), S=1.528× → **+15.86 TPS → 483.00** (conservative η-low f=0.03058 → +4.99). The config + sign are **stable across the ctx ladder** (the bm4 choice never flips):

| ctx | S_joint | S_ns | ΔTPS_joint | frontier | beats481 |
|---|---|---|---|---|---|
| 128 (PR card) | 1.535 | 1.416 | +16.01 | 483.15 | True |
| 256 | 1.534 | 1.412 | +15.98 | 483.12 | True |
| **512 (realistic anchor)** | **1.528** | **1.408** | **+15.86** | **483.00** | **True** |
| 1024 | 1.471 | 1.361 | +14.66 | 481.80 | True |

(Anchored at ctx=512 ≈ the committed serving roofline ctx=528, `built_step_roofline`; the PR-specified ctx=128 is ~4× too short but, unlike a pure occupancy artifact, the bm4 *config* and the speedup persist at realistic ctx — the BLOCK_M lever is genuine, not a short-context mirage.)

### ⚠️ Realization honesty — the +15.86 / 483.00 is an UPPER BOUND
The isolated per-call sum (35×h256 + 7×h512) at ctx=512 = **2306.5 µs**, but the in-graph attention budget implied by f×(real 8017 µs decode step, #284) is only **762.2 µs** → **3.03× over-count**. Isolated lone-kernel timing does **not** capture in-graph behavior, so the Amdahl frontier is a ceiling, not a realized number:

- **`realized_crossing_481_proven = False`** — joint realized band ≈ **[0, +15.86] TPS**.
- My banked **#428 priced the num_stages sub-lever's *realized* contribution at [0, +0.94] TPS** (a ~14× haircut vs its own isolated Amdahl) against the same 8017 µs step.
- **BUT** the bm4 occupancy fix has a **stronger a-priori realization case than num_stages**: it is **launch-intrinsic** (6→18 thread-blocks is a structural grid property that survives serial in-graph execution), and attention is **serialized on the critical path** (#284: decode 99.5% GPU-bound, verify 81.5% of the step). A pipeline-depth (num_stages) tweak depends on latency-hiding that may not survive in-graph; an occupancy fix should realize a larger fraction. **This is exactly the open question — and it cannot be answered without the served-kernel wall A/B.** Direct parallel to **#437** ("does cb3's modeled +15.60 survive on the served kernel?").

### (4) Equivalence / PPL / self-test
- **Byte-exact by construction:** the deployable bm4 config has `torch.equal` **maxdiff = 0.0** vs deployed in both head dims → greedy-identity and **PPL = 2.3772** preserved (tiling is a performance knob, not a correctness one). `ppl_ok=True`, PPL gate ≤2.42 satisfied.
- **Self-test (PRIMARY) = PASS:** 11/11 CPU analytic conditions (Amdahl identity/monotonicity, S_req=1.458× crossing identity, byte-exact partition logic, grid counts, inventory-not-autotuner) + 8/8 empirical (clone-fidelity, byte-exact partition non-empty, tile-is-nonexact knob, lynchpin wrapper-match, nan-clean, speedup-sane, inventory-count-zero, deployable-cfg-byte-exact).

### Reproduce / env
```bash
cd target/ && CUDA_VISIBLE_DEVICES=0 .venvs/vllm022/bin/python \
  research/validity/triton_attn_joint_autotune/triton_attn_joint_autotune.py \
  --wandb_group triton-joint-autotune --wandb_name wirbel/triton-attn-joint-autotune
# 0-GPU self-test gate: append --self-test --no-wandb
```
- **Device:** NVIDIA A10G sm_86, torch 2.11.0+cu130, triton 3.6.0. **Peak VRAM: 0.318 GiB.**
- **W&B:** `e5n9a2dc` (headline) — reproduced by `tfejmv3o` (+15.91 / 483.05, within timing noise). Group `triton-joint-autotune`.

### What happened — honest analysis
The joint sweep **did** find a real lever the single-axis sweep missed: **BLOCK_M is the load-bearing axis, not num_stages.** The deployed BLOCK_M=16 heuristic is mis-tuned for the M=8 single-stream verify shape (6-block SM under-occupancy), and dropping to BLOCK_M=4 (byte-exact) gives 1.53× isolated vs the 1.41× ceiling of num_stages alone — so `joint_beats_numstages_only` is unambiguously True, and the **modeled** Amdahl frontier (483.00) clears 481.53. **However, the honest verdict is that this is a *modeled* result, not a realized one.** The isolated kernel over-counts the in-graph attention budget by ~3×, my own #428 measured a 14× realization haircut on the sibling num_stages lever, and — exactly like cb3's modeled +15.60 (realized-NULL per #437) — a modeled +15.86 means nothing until it is measured on the served decode kernel. I am reporting `autotune_beats_deployed_481=True` because that is the PR-specified Amdahl computation, and `realized_crossing_481_proven=False` because the wall test has not been run.

### Suggested follow-ups
1. **The #437-analog served-kernel wall A/B for bm4 (the decisive test).** Serve int4 `google/gemma-4-E4B-it-qat-w4a16-ct` locally, inject bm4 via an env-gated launch override (BLOCK_M 16→4 + BLOCK_Q + grid recompute — I already have the validated byte-exact launch logic in `launch_tuned`, maxdiff 0.0), and run `scripts/profiler/paired_tps_ab.py` toggle→measure→revert (≥5 reps) to get the **realized ratio** that converts [0, +15.86] into a measured number. I am **flagging, not running**, this: it is a live-serve kernel-config injection, which the PR boundary reserves as human-gated ("deploy candidate I bring to the human — flag it, do not land it"). **Want me to run it?**
2. **bm4 as a human-gated deploy candidate.** If the wall A/B realizes, the served change is a one-line BLOCK_M heuristic override at `triton_unified_attention.py:~839` — byte-exact, greedy-safe, PPL-neutral by construction.
3. **Compose with denken's K-opt acceptance lever** (orthogonal axis): bm4's per-call attention saving is independent of the K-opt path and would stack on whatever realized fraction the wall A/B confirms.
