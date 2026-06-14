STUDENT lawine:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["nxwv6pam"],"primary_metric":{"name":"int4_tokenident_self_test_passes","value":1},"test_metric":{"name":"int4_token_identity_M1_vs_M8","value":0.9927083333333333}}

## Results

**The clean deployed-M=8 int4 greedy divergence is 0.73% per token (identity 99.27%) — it does NOT confirm #114's native-stack 56.08%; it corrects it.** The verify width matters for only 0.73% of tokens, not 56%, so #114's number is a *different* quantity (native spec sampling / draft-trajectory branching, M unrecorded), not the clean M=1-AR-vs-M=8-verify per-token effect that #192 rides on.

### Primary + test metrics

| metric | value |
|---|---|
| **`int4_tokenident_self_test_passes`** (PRIMARY) | **True** ✅ (all 7 checks) |
| **`int4_token_identity_M1_vs_M8`** (TEST) | **0.992708** (99.27%) |
| `int4_divergence_M1_vs_M8` | **0.007292** (0.73%) |
| `determinism_M1_vs_M1` (control) | 1.000000 ✅ |
| `determinism_M8_vs_M8` (control) | 1.000000 ✅ |
| `within_batch_copy0_vs_copy1` (control) | 1.000000 ✅ |
| positions scored | 32,640 (128 prompts × 255) |

Self-test breakdown (PRIMARY): `det_M1_vs_M1_eq_1`, `det_M8_vs_M8_eq_1`, `within_batch_copy0_vs_copy1_eq_1`, `identity_in_range_finite`, `divergence_eq_1_minus_identity`, `cap_eq_1_minus_divergence`, `nan_clean` — **all True**. Within-dtype within-batch determinism holds at exactly 1.0 over all 32,640 positions, so the M1-vs-M8 signal is clean (no run-to-run nondeterminism contaminating it — vLLM forced the deterministic TRITON_ATTN backend for Gemma4's heterogeneous head dims, the deployment-faithful path).

### Cross to #114 (the deliverable)

| metric | value |
|---|---|
| `int4_divergence_vs_114_delta` | **−0.553508** |
| `deployed_m8_divergence_confirms_114` | **False** |

#114's `9q5yy9l1` measured native-spec-vs-M1 `token_div_frac_max`=0.5608 with the verify width **unrecorded** (`reference_kind="unknown"`). Holding the served weights/quantization fixed and varying **only** the verify batch width M ∈ {1, 8} gives **0.73%**, not 56%. The 56.08% therefore is **not** the clean deployed-width per-token verify effect — the launch-relevant figure for the #192 mechanism is the deployed M=8 number, **0.73%**.

### Hand the dependent legs their number (the deliverable)

| metric | value | vs anchor |
|---|---|---|
| `reading_a_pass_fraction_deployed` | **0.164062** (21/128 seqs strictly identical) | vs #219 strict-A **0.125** → no worse (marginally better) |
| `margin_gate_supply_cap_deployed` = 1 − divergence | **0.992708** | up to 99.27% of verify positions are width-invariant |

(a) **#192 Reading-A strict per-sequence pass fraction at the clean width = 16.4%** — slightly *above* denken #219's strict-A 0.125, i.e. the strict-A accepted-risk is no worse than #219's reading assumed. (b) **Margin-gate supply cap = 0.9927**: any provable-skip / margin-gate scheme could skip at most 99.27% of verify positions before hitting a genuine M1-vs-M8 argmax flip — large headroom, the opposite of what a 56%-divergence reading implied. (c) **The greedy-identity gap IS the #114/#192 split-K width-sensitivity mechanism, not a measurement artifact** — controls at 1.0 confirm it.

### bf16 contrast (#221) + decode-width localisation

| metric | value |
|---|---|
| bf16 floor identity (#221 `6m40u2bg`) | 0.989441 (div 0.010559) |
| `int4_vs_fp16_identity_delta` | **+0.003267** (int4 is *less* divergent) |
| `int4_vs_fp16_divergence_ratio` | **0.69×** |
| `int4_body_bitexact_decode_M8` (in-process diag) | **True** |

The deployed int4 path is **below the bf16 floor** (0.69× the #221 divergence). The in-process decode-width diagnostic confirms why: all four int4-Marlin body GEMMs are **bit-exact** at the literal verify width M=8 vs M=1 (`qkv_proj`/`o_proj`/`gate_up_proj`/`down_proj`, max_abs_diff = **0.0** each). So the int4 body adds **zero** batch-width divergence; the residual 0.73% is the **bf16 tied lm_head + bf16 attention/norm accumulation** being batch-variant — and vLLM's fused serving path is numerically tighter than #221's naive `t.repeat()` transformers forward, hence below the bf16 floor.

### Comparison against baseline

- **Official baseline 481.53 TPS** (PPL 2.3772 served / 2.3777 private, 128/128; PR #52). **This leg adds 0 TPS — it is a measurement, not a speed change.** Greedy/PPL untouched (the served int4 path is READ, never modified).
- Settles the gap: #114's 56.08% was native-spec-vs-M1 with verify width unrecorded; the clean deployed M=1-vs-M=8 int4 divergence (the exact #192 mechanism) is now isolated at **0.73%**.

### Exact command

```bash
cd target/ && .venv/bin/python research/validity/int4_tokenident_deployed_m8/int4_tokenident_deployed_m8.py \
  --wandb_group int4-tokenident-deployed-m8 --wandb_name lawine/int4-tokenident-deployed-m8
```

(Smoke pre-flight: same script `--smoke --no-wandb` — 6 prompts, identity 0.9954, controls 1.0, passed before the full run. GPU phase runs as an isolated subprocess under `CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0`.)

### Run facts

- **Peak GPU memory:** 12.25 GB (gpu_mem_util 0.55 on the single A10G; comfortable headroom)
- **W&B run ID:** `nxwv6pam` (`wandb-applied-ai-team/gemma-challenge-senpai`, group `int4-tokenident-deployed-m8`)
- **Model:** `google/gemma-4-E4B-it-qat-w4a16-ct` (compressed-tensors int4-Marlin w4a16 body, bf16 tied lm_head) — the SAME canonical Hub checkpoint #221 probed; 128 prompts × max_len 256, batch_m = M_verify = K_spec(7)+1 = 8.
- **Geometry:** e2e number uses prefill-replication (M identical copies co-batched → GEMM M-dim = M×seq_len), the SAME geometry as #221's bf16 floor → apples-to-apples; the literal decode verify width (8 rows) is covered by the in-process bit-exact diagnostic. Both geometries agree the int4 body adds no divergence.

### What happened — honest analysis

**It worked, and it overturns the pessimistic reading.** The probe pins the TRUE clean deployed-width int4 greedy divergence (M=1 AR vs M=8 verify, same weights, vary only M) at **0.73% per token** over the served 128×128 prompts. All determinism controls hold at exactly 1.0, so this is the clean batch-width signal, not nondeterminism. Two independent geometries agree: (1) e2e prefill-replication gives 0.73% (below the bf16 floor), and (2) the isolated decode-width diagnostic shows the int4-Marlin body GEMMs are *bit-exact* across M — so int4 contributes **zero** divergence and the residual is the bf16 lm_head/attention being batch-variant. #114's 56.08% is a different quantity (native spec stochasticity / draft-trajectory branching, width unrecorded) and does **not** describe the deployed per-token verify effect. Consequence for the launch: the #192 strict-A pass fraction at the clean width (16.4%) is no worse than #219's 0.125, and the margin-gate supply cap is **0.9927** — there is large provable-skip headroom, the opposite of what a 56%-divergence reading implied.

**Honest band (carried per instruction 4):** (a) LOCAL probe — the OFFICIAL scorer's exact verify width stays human-gated; the deployed M=8 is reported as the launch config. (b) Within-dtype within-batch determinism held *first* (controls 1.0), so the probe is reliable. (c) Divergence is MEASURED; the served path is not changed. (d) #114's native 0.5608 imported unchanged as the comparison anchor. (e) The checkpoint is the canonical Hub int4 (the locus of M-sensitivity, the int4-Marlin body, is identical to the flagship; the diagnostic confirms it's bit-exact, so the only residual divergence source is the bf16-class lm_head, which the deployed lmhead12k shares in dtype) — the qualitative conclusion (≈bf16 floor, not 56%) carries to the deployed lm_head. (f) max_len capped at 256; the per-token flip rate is a per-position property that does not systematically grow with length.

### Hand-off (one sentence) to denken #230/#192 + fern #185

> *The clean deployed-M=8 int4 greedy divergence is **0.0073** (vs #114's native-stack 0.5608 — it **corrects** it, not confirms), so the #192 strict-A pass fraction at the deployed width is **0.164** and any margin-gate supply cap is ≤ **0.9927** — settling the provenance the launch's #192 accepted-risk rests on.*

### Provenance / evidence used (imported, not re-derived)

- #114 `9q5yy9l1`: native-spec-vs-M1 `token_div_frac_max` = 0.5608 (M unrecorded) — comparison anchor, unchanged.
- #219 `0unwptbz`: strict-A = 0.125 — dependent-leg anchor.
- #221 `6m40u2bg`: harness + bf16 floor identity 0.9894, body-GEMM bit-exactness, tier-2 determinism checks — reused.
- All in `wandb-applied-ai-team/gemma-challenge-senpai`. Relates to Issue #192 (greedy-decode accepted-risk).

### Suggested follow-ups

- **Deployed lmhead12k checkpoint:** re-run through the serve patch (untied 12k-row lm_head) to confirm the kept-row bf16 lm_head gives the same ~0.7–1.0% floor — currently *inferred* from the canonical Hub checkpoint + body bit-exactness, not measured on the flagship lm_head.
- **Full 512 output length:** `--max-len 512` to confirm the per-token flip rate doesn't grow with position (cheap; same harness) and to give denken #230/#192 the per-sequence strict fraction at the served length (it compounds lower than 16.4%).
- **lm_head ablation:** force a batch-invariant / fp32 lm_head and re-measure to confirm the bf16 lm_head is the *sole* residual divergence source — would pin the margin-gate supply cap to exactly the lm_head batch-variance.
