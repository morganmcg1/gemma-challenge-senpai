STUDENT kanna:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["hz8jkc5h","8wne15eh","z0mclftv"],"primary_metric":{"name":"int4_mtp_batchinv_greedy_flip_rate_per_token","value":0.0037578288100208767},"test_metric":{"name":"tps_batchinv_on_local","value":40.6}}

## Results — BLOCKED / definitive negative

**`VLLM_BATCH_INVARIANT=1` does NOT rescue greedy-valid spec decode — at the int4 target or at a bf16 target.** Batch-invariant kernels are compatible with our exact validated stack and were confirmed active, but int4+MTP spec decode stays greedy-**DIVERGENT** (0.376%/tok ON, statistically unchanged from 0.332%/tok OFF). The bf16 positive control localizes *why*, and the answer is two independent un-coverable sources — so the drafter-ladder unlock this PR was chasing is closed for the invariant-kernel lane.

### Headline numbers (decisive Step 3)

| arm | INV | target GEMM | verdict | identical/32 | flip /tok | 95% CI | W&B |
|---|---|---|---|---|---|---|---|
| **int4 ON** (decisive) | 1 | Marlin `_C` (un-covered) | **DIVERGENT** | 5/32 | **0.376%** | [0.234, 0.518]% | `hz8jkc5h` |
| int4 OFF (same-session control) | 0 | Marlin `_C` (un-covered) | DIVERGENT | 6/32 | 0.332% | [0.205, 0.460]% | `8wne15eh` |
| **bf16 ON** (discriminator) | 1 | aten linear (**covered**) | **DIVERGENT** | 16/32 | **0.111%** | [0.057, 0.166]% | `z0mclftv` |
| bf16 OFF (PR #5 ref) | 0 | aten linear | DIVERGENT | — | 0.72% | — | — |

Official strict bit-exact verifier (`check_greedy_identity.py --json`); per-token flip = censored-geometric MLE; 32 prompts × 512 tok, seed 1, `ignore_eos`, eager. Within each arm the only delta is `speculative_config` (M=1 AR ref vs M=K+1 spec cand, **both under the same INV flag**); across arms the only delta is INV (plus target model for bf16).

### Decision-rule outcome (from the PR brief)
- **Step-3 rule: ON → still DIVERGENT.** Fired. → not Step 4 (frontier TPS); terminal negative.
- The brief's discriminator was binary ("bf16 ON identical ⇒ Marlin; bf16 ON divergent ⇒ spec path"). The measured bf16 ON is **DIVERGENT but 6.5× reduced** (0.72%→0.111%), which is *richer*: it pins **both** causes additively.

### Two-part root cause (the bf16 control is the lever)

1. **Cause (a): the int4 Marlin weight-GEMM is batch-variant, and the aten override can't reach it.** The only stack difference between **int4 ON** and **bf16 ON** (both INV=1) is the target weight GEMM — int4 Marlin (a `torch.ops._C` custom op) vs bf16 linear (aten, covered). int4 ON flips at 0.376% but bf16 ON at 0.111%: the **~0.265%/tok excess is the un-covered Marlin GEMM**. Equivalently, **int4 ON ≈ int4 OFF** (CIs overlap) — covering the aten ops does *nothing* for int4 because int4's dominant flip source is the one op batch-invariance structurally cannot intercept. This **refutes** my Step-1 prior (that Marlin was "plausibly already M-invariant"); if it were, int4 ON would have dropped to the bf16 floor. It didn't.
2. **Cause (b): the spec verify path has a non-aten batch-variant component.** bf16 ON has **zero Marlin and full aten coverage**, yet is **still DIVERGENT at 0.111%/tok**. That irreducible residual is not a weight-GEMM effect — it lives in a non-aten part of the spec verify forward (attention-metadata build / rejection-sampler logits compare / a fused verify step). This is the **measured, quantified** version of vLLM's own disclaimer (issue **#27433**: batch-invariance *"does not currently integrate with speculative decoding"*).

**First-order consistency check** (CIs wide): Marlin contribution + spec residual ≈ 0.265% + 0.111% = 0.376% ≈ observed int4 ON. The two sources are independent, additive, and both outside the aten override's reach.

**Implication:** neither the int4 target (a+b) **nor** a hypothetical bf16-target drafter ladder (b alone) is rescuable by `VLLM_BATCH_INVARIANT`. The "pin a batch-invariant vLLM" lane is a definitive negative for greedy-valid spec decode at **any** precision in 0.22.0.

### Step 2 — invariance confirmed ACTIVE (not assumed)
- Kernel probe (`confirm_invariant.py`, A10G): **ON** → `INVARIANT_ACTIVE_AND_FUNCTIONAL`, M=1 & M=7 rows bit-identical to the M=8 batched GEMM (max|diff|=0); **OFF** control → max|diff|=1.0. The `aten::mm` override works on this box.
- Server-side (decisive run): EngineCore worker log shows the override installed in-process (`new kernel: registered at batch_invariant.py:913`), `MarlinLinearKernel for CompressedTensorsWNA16` (int4 Marlin preserved), `AttentionBackendEnum.TRITON_ATTN` (invariance-supported), `Resolved architecture: Gemma4ForConditionalGeneration` + `Gemma4MTPModel` + `profiled with 1 video items` (**all modalities loaded**), spec init OK with the {8,4} monkeypatch.

### Step 1 — compatibility audit corrected the brief's two key assumptions
Full write-up: `research/batch_invariant/version_compat.md`.
- **No vLLM version bump.** The brief assumed we'd pin a *newer* engine (and worried it would break Gemma-4/Marlin/spec). Wrong in our favour: **vLLM 0.22.0 — the exact version PR #5 validated end-to-end — already ships the batch-invariant kernel set** (`model_executor/layers/batch_invariant.py`; worker auto-calls `init_batch_invariance()` when `VLLM_BATCH_INVARIANT=1`). So the entire "engine-bump regression" risk surface never existed.
- **{8,4} attn-group fix is NOT native** in 0.22.0 (no bump), so the PR #5 monkeypatch is **kept**, not dropped. The brief's "drop if native" instruction doesn't apply.
- **Marlin is safe.** Batch-invariance overrides only the `aten` library; Marlin is a `_C` op → enabling the flag leaves the int4 weight-GEMM byte-identical → **zero downside to the int4 bandwidth floor from merely testing.** (This is also *why* it can't fix int4 — same fact, both directions.)

### Step 4 — TPS cost (eager diagnostic only; full characterization moot)
Per the PR's own rule ("a TPS number for a greedy-INVALID stack is worthless"), I did **not** pursue the cudagraph frontier characterization. Eager single-stream wall-clock, INV=1, shown only for cost direction:
- int4 AR (M=1, spec off, INV=1): 16384 tok / 799.2 s = **20.5 tok/s**
- int4 spec (K=6, INV=1): 16384 tok / 403.8 s = **40.6 tok/s** (~2.0× over AR; ~2.2 tok/step MTP)
- bf16 spec (K=6, INV=1): 16384 tok / 447.7 s = 36.6 tok/s

Even with spec, invariant **eager** throughput (~40 tok/s) is ~3× below the greedy-valid int4 floor (~127 TPS, cudagraph). The eager-vs-cudagraph gap dominates here; the point is moot regardless because the greedy gate already failed.

### PPL / modalities
- **PPL: not separately measured** — moot because the greedy gate failed first (a leaderboard PPL for a greedy-INVALID stack is meaningless; the live leaderboard validity gate is greedy-identity + PPL, and we fail the former). For reference, PR #5's int4+MTP PPL was 2.006 (also greedy-invalid). No PPL claim is made here.
- **All modalities loaded** (vision/audio towers present; encoder cache `profiled with 1 video items`) in both int4 and bf16 arms. No modality disabled; model unchanged; greedy-decode identity is exactly what we *tested* (and report failing).

### Exact commands
```bash
# decisive int4 ON arm (and the two controls via the driver)
cd research/int4_mtp_batchinv
ARM=int4_on  INV=1 K=6 NPROMPTS=32 TARGET_MODEL_ID=google/gemma-4-E4B-it-qat-w4a16-ct \
  OUTDIR="$PWD/arms" bash run_arm.sh
bash run_controls.sh   # int4_off (INV=0) + bf16_on (INV=1, target=google/gemma-4-E4B-it)
# verify per arm: official check_greedy_identity.py --json  ->  flip_rate.py
# W&B: python log_arm_wandb.py --outdir arms --arm int4_off:0 --arm int4_on:1 --arm bf16_on:1
```
Submission shell built at `submissions/int4_mtp_batchinv/` (vLLM 0.22.0 + `VLLM_BATCH_INVARIANT=1`, K=6, `max-num-seqs=1`, `gpu_memory_utilization=0.90`, `MAX_NUM_BATCHED_TOKENS=512`, {8,4} patch). **Not submitted** — no HF Job, per the LOCAL-ONLY guardrail and because the greedy gate fails (no approval issue opened).

### Peak memory
int4 arm: model weights **9.85 GiB** + KV cache **8.55 GiB** (340,746 tok), ~20.7 GiB reserved at `gpu_memory_utilization=0.90` on the 23 GB A10G. Comfortable headroom; no OOM.

### Public evidence used
- Live leaderboard frontier (digest, 2026-06-13): top valid methods ~459 TPS / PPL ~2.38, all `verification: valid` — confirms greedy-identity + PPL is the gate this experiment must pass, which it does not.
- vLLM docs + issue **#27433** ("does not currently integrate with speculative decoding") — now corroborated by a measured 0.111%/tok bf16 residual.
- Builds on my PR #5 (the program's #1 linchpin: vLLM-0.22.0 rejection-sampler spec is greedy-INVALID at every precision; the M=K+1 verify-GEMM shape is the cause).

### What happened — honest analysis
The hypothesis (batch-invariant kernels make the M=K+1 verify forward bit-match the M=1 AR forward → GREEDY_IDENTICAL) **failed**, and the failure is now *explained*, not just observed. Batch-invariance's coverage is real (the kernel probe is bit-exact) but it is `aten`-scoped, and the two things that actually flip our spec argmax both sit outside `aten`: the int4 Marlin weight-GEMM (a `_C` op) and a non-aten component of the spec verify path. The bf16 control cleanly separates them — removing Marlin (bf16) drops the flip 3.4× vs int4 but cannot reach zero, exposing the irreducible spec-path residual. This is a strong, program-steering negative: it rules out the entire "system-wide invariant kernels" lane for greedy-valid spec decode (int4 *and* bf16), with a documented mechanism (#27433) and a measured residual behind it.

### Suggested follow-ups (not implemented — flagging only)
1. **Verify-rollback gate (arxiv 2601.17768, "Enabling Determinism in LLM Inference with Verified Speculation").** Decode on the fast (non-deterministic) spec path, re-verify accepted tokens under a fixed-shape reduction schedule, commit consistent tokens and roll back violators. This targets cause (b) directly (the spec-verify residual) and is precision-agnostic, so it would also dodge cause (a). This is the natural next lane now that invariant kernels are ruled out.
2. **Locate cause (b) precisely.** Instrument the bf16 ON verify forward to find the specific non-aten batch-variant op (attention-metadata build vs rejection-sampler logits compare vs a fused verify step). If it's a single isolable kernel, a targeted fixed-shape patch *might* close the 0.111% residual without the full rollback machinery — but this is speculative and lower-priority than (1).
3. **Reframe the drafter ladder.** Since greedy-valid spec is not achievable via invariant kernels at any precision in 0.22.0, fern's EAGLE-3 (PR #16) and any future PARD inherit the same blocker through the shared rejection-sampler verify path; they should be evaluated under the verify-rollback gate rather than assuming an invariant-kernel rescue.
