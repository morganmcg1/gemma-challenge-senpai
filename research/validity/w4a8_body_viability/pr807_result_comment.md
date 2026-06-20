STUDENT wirbel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["t2msgab0"],"primary_metric":{"name":"local_decode_tps_single_stream_w4a8","value":422.52},"test_metric":{"name":"ppl_w4a8","value":2.0105}}

## Results — W4A8 body activation quant: SERVES + quality-safe, but −16.68% decode TPS (clean NEGATIVE speed lever)

**Verdict:** the W4A8 body path is **servable on A10G sm_86** and **PPL-safe**, but it is a **−16.68% / −84.6 TPS decode regression** versus the identical-checkpoint W4A16 control. The fire-gate **FAILS on speed**. Step 3 (4-axis quality panel) was **not triggered** — the PR gates it behind "PPL **and** speed clear", and speed did not clear.

The PR's core premise — *"the body is near HBM-bandwidth-bound, so halving bf16→int8 activation reads → a large decode-TPS gain"* — is **false for low-batch decode**. At M=1–8 the body GEMM is **weight-read-bound**, not activation-read-bound (see physics below).

### Gate scorecard
| Gate | Result | Pass? |
|---|---|---|
| Serves on A10G sm_86 (Step 1) | W4A8 Marlin path dispatches via `VLLM_MARLIN_INPUT_DTYPE=int8`; sane greedy text; greedy output **differs** from W4A16 → int8 activations genuinely live (not a silent W4A16 fallback) | ✅ |
| PPL ≤ 2.42 | W4A8 PPL **2.0105** (+0.0048 vs control 2.0057) | ✅ |
| 128/128 prompts | 128/128 both arms (`num_records=128`) | ✅ |
| local TPS > base | W4A8 **422.52** < W4A16 control **507.12** → **−16.68%** | ❌ |

### Step 2 A/B (same `w4a16-ct` checkpoint + MTP K=6; the ONLY variable is `VLLM_MARLIN_INPUT_DTYPE`)
| Arm | decode TPS — local single-stream probe (mean ± std, n=5) | PPL | completed |
|---|---|---|---|
| **W4A16** control (bf16 acts; env **unset**) | **507.12 ± 0.39** | 2.0057 | 128/128 |
| **W4A8** treatment (int8 acts; env=`int8`) | **422.52 ± 0.63** | 2.0105 | 128/128 |
| **Δ (W4A8 − W4A16)** | **−84.60 TPS = −16.68%** | +0.0048 | — |

Per-arm spread is <0.7 TPS, so the −84.6 TPS gap is **>100σ** — the sign and magnitude are unambiguous (the 5-rep spread was added precisely because physics predicted a near-zero effect that a single point probe could not resolve).

> ⚠️ **Absolute-number caveat (read before comparing to 256.74).** The 507/422 figures are **local single-stream `probe_tps` numbers** — steady-state decode, spec-decode-amplified, measured on this AWS A10G — and are **NOT** official a10g-small TPS, nor directly comparable to the PR's `256.74` int4head reference (a different, official-style measurement). They are reported only as a **controlled within-experiment A/B**; the robust, transferable result is the **Δ = −16.68%**, measured identically for both arms on the same probe.

### Design note — why the `w4a16-ct` base, not the int4head base
`VLLM_MARLIN_INPUT_DTYPE` is a **global** runtime flag: it flips **every** gptq-Marlin int4 layer to W4A8, with no per-layer control. On the int4head base the lm_head is *also* int4-Marlin, so the toggle would flip **body + head together**, confounding the body-only measurement the PR asked for ("body→W4A8, **keep** int4 head"). The `google/gemma-4-E4B-it-qat-w4a16-ct` checkpoint has a **bf16 tied** lm_head (not Marlin), so the toggle touches **only the body** — exactly the body-isolated lever. The body-W4A8 lever is orthogonal to the head quant (separate GEMMs), so the measured Δ transfers; and since Δ is negative, composing with int4head cannot rescue it. (No requantization was needed: the same official int4 weights serve as W4A8 just by setting the env var — `marlin_moe.py:127` int8-quantizes activations per-token at runtime.)

### Why it loses — physics
At single-stream decode (M=1, or M≈7 on the K=6 MTP verify pass) the body MoE-expert GEMM is **weight-read-bound**:
- int4 expert weights (**identical** W4A16→W4A8) dominate HBM read traffic;
- bf16→int8 activations save **<0.6%** of the bytes moved (the activation tensor is M×K vs the K×N weight; at M≈1 the ratio ≈ 4M/N ≪ 1);
- W4A8 **adds** a per-token `marlin_quant_input` activation-quant kernel on **every** body layer, and decode is memory-bound so int8 tensor cores (2× bf16 peak) give **no** compute payoff.

Net: pure added overhead, no compensating benefit → −16.68%. The activation-precision lever only helps where the path is activation-read-heavy or compute-bound (large-batch / prefill), which is **not** the challenge's single-stream decode regime. (A second, compounding mechanism is plausible — W4A8 perturbs the target verify logits, which can lower MTP acceptance and thus tokens/step — but the net is unambiguously negative regardless; see follow-ups for the cheap AR-only decomposition.)

### Positive corollaries (worth banking even though the lever is dead)
1. **A servable W4A8 Marlin kernel DOES exist on Ampere sm_86.** vLLM 0.22.0 ships it behind `VLLM_MARLIN_INPUT_DTYPE=int8` (`envs.py:161`, choices `int8|fp8`), with **no requantization** — the official `…-qat-w4a16-ct` int4 weights serve as W4A8 via runtime per-token int8 activation quant. This rules a W4A8 path **IN** on sm_86, unlike the Cutlass/Machete sm_90 walls (#781/#779). Not useful for decode TPS, but it is the right tool if a future lever becomes compute-bound (e.g. heavy prefill / high concurrency).
2. **int8 MoE-activation quant is quality-safe for Gemma-4-E4B** (PPL 2.0105, +0.0048, within the 2.42 cap, 128/128). The feared MoE-outlier collapse did **not** materialize on the 128 validity prompts — so this is a clean **speed/physics** negative, not a quality negative.

### Commands
```bash
# Step 1 — viability kill-gate (plain AR, both arms): PASS
CUDA_VISIBLE_DEVICES=0 .venv/bin/python research/validity/w4a8_body_viability/run_w4a8_ab.py --step 1

# Step 2 — PPL + speed A/B (MTP K=6, 5 TPS reps/arm + official 128-prompt PPL)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python research/validity/w4a8_body_viability/run_w4a8_ab.py --step 2 --tps-reps 5
```
LOCAL A10G only — **no HF job launched** (operator rule). `nvidia-smi` checked clean before launch (0 MiB, no procs).

### Peak memory
~**19.7 GiB** steady-state both arms (W4A16 19651 MiB, W4A8 19679 MiB; bounded by `GPU_MEMORY_UTILIZATION=0.90`). int8 activations do not change the steady-state footprint (weights are identical int4; activation tensors are transient). Model weights load = 9.7 GiB; KV cache ≈ 8.6 GiB.

### W&B
Run `t2msgab0` (group `w4a8-body-viability`, project `gemma-challenge-senpai`): per-arm TPS reps/spread, PPL, GPU mem, and the full delta block + `step2_result.json` artifact.
https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/t2msgab0

### Public evidence used
Checked the challenge digest (`/v1/digest?as=senpai`): the current leaderboard frontier (ranks 1–5, ~505–514 TPS) is the **W160/W192 + CENTROID_TOP_K + noprecache + split-KV + MTP-K7** weight-quant/sliding-window family (e.g. `mikasa-inbound` hayai-repro, `sparkgemma-s46b` w192-ctk48). **No public result uses W4A8 / int8 activations** — this lane is novel, and this negative records that the activation-precision axis is inert-to-harmful for single-stream decode on sm_86. No inbox `@senai` warning touches W4A8 (the 2 mentions are an unrelated human PPL-validation question and a W192 repro claim).

### Suggested follow-ups
- **None for W4A8-as-decode-lever** — refuted by physics + measurement; do not re-propose body or head activation-precision (int8/fp8) quant for low-batch decode TPS on sm_86.
- *(cheap, optional decomposition)* Re-run the A/B with `NUM_SPECULATIVE_TOKENS=0` (plain AR) to split the −16.68% into (a) raw per-layer int8-quant kernel overhead vs (b) MTP-acceptance drop from perturbed verify logits. ~15 min local; **does not change the verdict** (both components are losses) — only file if the mechanism split is wanted for the record.
- *(orthogonal)* The decode wall remains the body MLP/MoE **weight read** (74–79% HBM BW, stark #798). The lever that actually moves it is reducing **weight** bytes (lower-bit weights / better packing / sparsity) or raising the int4 Marlin kernel's BW efficiency — not activation precision.

### Bug-fix note (separate from the experiment)
`scripts/local_validation/harness.py`: `LocalServer.extra_env` now accepts a `None` value meaning "**unset** this manifest-baked env key" (plain `dict.update` cannot express deletion). This is load-bearing for any same-checkpoint A/B that must **remove** a baked env var — here the W4A16 control needs `VLLM_MARLIN_INPUT_DTYPE` *absent* while the submission bakes it as `int8`. Backward-compatible (str values unchanged). Please review/merge alongside.
