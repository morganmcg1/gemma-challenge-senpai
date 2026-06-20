STUDENT ubel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["uaq6btet"],"primary_metric":{"name":"decode_tps","value":218.49},"test_metric":{"name":"ppl","value":2.005655}}

## Results — TERMINAL NEGATIVE (programme-level): the W4A16 kernel-swap lever does not exist on A10G sm_86

**Verdict:** This is the PR's *"equally valuable negative outcome."* On A10G (sm_86) in the pinned vLLM 0.22.0 stack there is **no numerics-equivalent faster W4A16 GEMM kernel or config** to swap to — Marlin is the sole servable kernel for this checkpoint, its one behavioral toggle is hardware-gated off, and it exposes no tile/thread/split knobs. So the highest-leverage *kernel-schedule* lever in the 218→300 programme is closed; the remaining levers are **drafter-acceptance** (fern #774 K-sweep, ngram) and **fewer-weight-bytes**.

### 1. Control reproduction (bi0 as shipped) — matches the anchor
| metric | my control | anchor (PR) | ok |
|---|---|---|---|
| decode TPS (aggregate) | **218.49** (65,536 tok / 299.95 s, sharegpt speed set, out=512, seed 1) | 218.02 | ✓ |
| PPL | **2.005655** | 2.0058 | ✓ |
| completed | **128 / 128** | 128/128 | ✓ |
| official gate (PPL ≤ 2.42 ∧ completion ∧ modalities) | **PASS** | — | ✓ |
| modalities (text/image/audio/video) | all loaded | all | ✓ |

Greedy verdict vs the spec-off (M=1) reference was DIVERGENT (108/128) — this is the **documented int4-Marlin split-K batch-non-invariance** between M=7 verify and M=1 decode (BASELINE #114/#126/#225/#270), explicitly **NOT an official scorer gate** (#38). The binding quality gate (PPL ≤ 2.42, 128/128) PASSES.

### 2. Kernel identification + the key bandwidth diagnostic
**Dispatch:** `INFO compressed_tensors_wNa16.py:112] Using MarlinLinearKernel for CompressedTensorsWNA16` (compressed-tensors, symmetric int4, group_size=32, uint4b8) — confirmed in both the control and probe server logs.

**Profile (conc=1 / M=1, official `gemma_decode_profiler`, torch.profiler):** weight-GEMM = **91.4 %** of GPU-busy time (re-confirms BASELINE's ~92 % on THIS stack). Per-token device times: int4 Marlin body **6.28 ms**, lm_head GEMV **2.776 ms** (graph- and eager-mode cross-validated within 0.2 %).

**Achieved % of A10G peak HBM (600 GB/s) — the key diagnostic:**
| GEMM | bytes/token | per-token | achieved BW | % of 600 GB/s |
|---|---|---|---|---|
| int4 W4A16 Marlin (body, 2.188 GB) | 2.18825 GB | 6.28 ms | 348 GB/s | **58.0 %** |
| bf16 lm_head GEMV (1.342 GB) | 1.34218 GB | 2.776 ms | 483.5 GB/s | **80.6 %** |
| combined weight GEMM | 3.530 GB | 9.06 ms | 389.6 GB/s | **~65 %** |

**Reading the 58 %:** at M=1 the int4 Marlin is *not* HBM-saturated; the gap to the ~80 % one-wave wall is 4-bit dequant (unpack + group-scale apply) plus small-M tile underutilization (`thread_m_blocks=1`). **But that headroom is not exploitable**, for two independent reasons:
- **(a) No alternate kernel exists to capture it** (see §3) — the lever is absent from the stack.
- **(b) Deployed serving does not run at M=1.** With the MTP drafter (`NUM_SPECULATIVE_TOKENS=6`) the target verify GEMM runs at **M=7** (1 bonus + 6 draft rows), reusing the same weight bytes across 7 rows; Marlin then approaches the one-wave HBM wall (BASELINE measured 79.4 % at M=8). The proof it's already harvested: the M=1 single-stream profiled step is 13.29 ms (≈ 75 tok/s), yet deployed spec-on serving hits **218.49 tok/s (≈ 2.9×)** — speculation is already amortizing the weight read. The lm_head (80.6 % at M=1) is *already* at the wall and does not benefit from M>1 (it is run once per accepted token).

### 3. sm_86 W4A16 kernel/knob inventory — every lever is a dead end (code-traced + empirically probed)
**Dispatch table** (`choose_mp_linear_kernel` over `_POSSIBLE_KERNELS[CUDA]`):
| prio | kernel | min cap | sm_86 outcome |
|---|---|---|---|
| 1 | CutlassW4A8 | 90 | skipped (cap) |
| 2 | Machete | 90 | skipped (cap) |
| 3 | AllSpark | 80 | reject — W8 (uint8b128) only, not uint4b8 |
| 4 | **Marlin** | 75 | **SELECTED** |
| 5 | Conch | 80 | `conch-triton-kernels` not installed |
| 6 | Exllama | 60 | rejects bf16 (act_type must be float16) |

**Empirical lever probes (this run, logged to W&B):**
- `VLLM_MARLIN_USE_ATOMIC_ADD=1` → still logs *"Using MarlinLinearKernel"* (no-op). Hardware-gated: `should_use_atomic_add_reduce` returns False when `device_capability[0] < 9 ∧ dtype == bf16` (Ampere lacks native bf16 atomicAdd). **Confirmed.**
- `--quantization machete` → vLLM **hard-rejects at config validation**: `ModelConfig ValidationError: Quantization method specified in the model config (compressed-tensors) does not match the quantization method specified in the 'quantization' argument (machete)`; server fails to boot. This is the documented machete refusal — and it generalizes: *any* alt method name (machete/marlin/gptq_marlin/awq_marlin) mismatches the checkpoint's `compressed-tensors` and is refused the same way; only `--quantization compressed-tensors` (the no-op match) is accepted. **The CLI cannot force a different W4A16 kernel.** Confirmed.
- `use_fp32_reduce`: hardcoded `USE_FP32_REDUCE_DEFAULT=True`, not env-exposed; flipping changes reduction precision (numerics) → fails the greedy-identity gate by construction.
- tile/thread/split: compile-time constants (`GPTQ_MARLIN_TILE=16`, `MIN_THREAD_N=64`, `MIN_THREAD_K=128`, `MAX_PARALLEL=16`); `ops.marlin_gemm()` takes no tile/warp/split args. Zero user-facing config.
- `VLLM_DISABLED_KERNELS=MarlinLinearKernel` → falls to Conch (absent) / Exllama (rejects bf16) → no servable kernel (won't boot).

### Gates — no fire-worthy variant (none could even be constructed)
A variant is fire-worthy only if **local TPS > control ∧ > 218.02 ∧ greedy token-identical ∧ PPL ≤ 2.42 ∧ 128/128**. There is **no alternate numerics-equivalent kernel/config to test**, so `fire_worthy_variant_found = 0`, `best_variant_tps_delta_vs_control = 0.0`. **LOCAL ONLY — no HF Job, no submission.**

### #777 cleanup (folded in)
`submissions/int4_mtp_bi0_fp8kv/` is **already absent from both HEAD and the base branch** (`approval-gated-8gpu-20260613`) — it was removed in the prior #777 cycle. A stray *untracked* local copy was deleted this session. Tree is clean; no non-bootable submission dir remains.

### Reproduction
```bash
# Control (local-only validation harness; stands up serve.py with the shipped bi0 manifest env:
#   NUM_SPECULATIVE_TOKENS=6, VLLM_BATCH_INVARIANT=0, force-2D attn; out=512, 128 prompts, sharegpt speed set)
python scripts/local_validation/validate_submission.py --submission submissions/int4_mtp_bi0_surgattn \
  --output-dir research/_localrun/control-bi0
# conc=1 decode-step profile (official gemma_decode_profiler, spec-off graph+eager) -> graph_profile.json / profile_breakdown.json
# empirical lever probes (atomic_add=1; quantization=machete):
bash research/_localrun/_lever_probe.sh
# W&B logging (SYSTEM python3 — submission venv's wandb is a namespace shim without .init):
/usr/bin/python3 research/bi0_marlin_gemm/_log_wandb.py
```

### Peak memory
Model weights 9.86 GiB + KV cache 8.45 GiB (336,922 tokens) at `--gpu-memory-utilization 0.90` ≈ 21.3 GiB of the 23.68 GiB A10G cap. CUDA-graph pool 0.04 GiB.

### W&B
Run **`uaq6btet`** (group `bi0-marlin-gemm`, job_type `kernel-diagnostic`): https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/uaq6btet
Artifacts: `bandwidth_diagnostic`, `kernel_inventory_sm86`, `graph_profile`, `eager_profile_breakdown`, `control_evidence`.

### What happened — honest analysis
The hypothesis's *positive* branch is **falsified on this hardware**: a faster numerics-equivalent W4A16 kernel for the same int4 weights cannot exist in vLLM 0.22.0 on sm_86, because Marlin is the only servable kernel and it has no greedy-safe tunable. The hypothesis's *negative* branch is **confirmed and is the result**: the deployed weight-GEMM is already at the practical bandwidth ceiling **where it actually runs (M=7 verify)**, and speculation is the mechanism already harvesting the M=1 dequant/tile slack (218.5 vs ~75 tok/s single-stream). The 58 % M=1 number is real but a red herring for *this* lever — it is not idle bandwidth a kernel swap could claim. **The 218→300 programme should refocus on drafter-acceptance and fewer-weight-bytes**, exactly as the PR anticipated.

### Suggested follow-ups (not implemented — flagging only)
- **Fewer-weight-bytes** is the one untouched bandwidth lever the kernel can't supply: the bf16 lm_head GEMV (1.342 GB/token, 80.6 % HBM, ~31 % of the M=1 weight read) is the single largest non-int4 read. An int4/int8 lm_head (or a greedy-safe vocab-prune) would cut that read — but prior work flagged lm_head GEMV cross-session argmax instability (memory: bf16-head x-session nondeterminism) and vocab-prune certificates firing 0 % (memory: gemma-lmhead-geometry), so any such arm must carry its own greedy-identity proof.
- **Drafter-acceptance** (fern #774 K-sweep / ngram): higher acceptance raises effective M and pushes the verify GEMM further into the one-wave-saturated regime — directly compounding with this finding.
- A **future sm_90 (H100) port** would unlock Machete/CutlassW4A8 and bf16 atomicAdd, but that is out of scope for the A10G leaderboard target.
