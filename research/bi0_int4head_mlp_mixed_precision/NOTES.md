# PR #810 — Per-layer body-MLP precision sensitivity map → mixed-precision viability (stark)

LOCAL A10G ONLY. No HF Job, no submission. W&B group: `body-mlp-mixed-precision`.

## Hypothesis
int4head applies a **uniform** W4 (g32 symmetric) to every body-MLP layer
(`gate_proj`/`up_proj`/`down_proj`, layers 0–41). The body MLP is the dominant
decode wall and is **bandwidth-bound** on weight reads (#798: 74–79% HBM BW,
Marlin 1-wave, M-flat), so bytes-saved ≈ TPS. Layers are not equally
quant-sensitive: a subset may tolerate sub-4-bit (W3/W2) at negligible quality
cost, while a few sensitive layers must stay W4. A **per-layer mixed-precision**
body-MLP reads fewer average bytes/token than uniform W4. Distinct mechanism
from wirbel #807 (W4A8 activations) and fern #808 (2:4 pruning).

## Base / control (int4head)
- 256.74 TPS local single-stream decode; PPL 2.0029 (W&B `57izwrp6`).
- Quality: MMLU-Pro 0.692 / GSM8K 0.915 / AIME greedy 0.300 (maj@8 0.400) /
  GPQA-Diamond pooled 0.5030.
- Body = byte-identical to `google/gemma-4-E4B-it-qat-w4a16-ct` (int4 W4A16
  g32 symmetric; group_0 targets all text-decoder Linear). Layers=42,
  hidden=2560, intermediate=10240.
- Quality floors (HARD): MMLU-Pro ≥ 0.572, GPQA-Diamond ≥ 0.471, GSM8K ≥ 0.807,
  AIME ≥ 0.090; PPL ≤ 2.42; 128/128. PLUS #784 within-5%-of-base each axis.
- **Note: PPL headroom is large** — base 2.0029 vs cap 2.42 (~21%).

## Resources (all local, already staged)
- Body weights: `~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct`
  (snapshot `ef0a4c43...`). 126 MLP tensors:
  `model.language_model.layers.{0-41}.mlp.{gate,up,down}_proj.{weight_packed,weight_scale,weight_shape}`.
- Quant primitives: reuse `submissions/int4_mtp_bi0_int4head/build_lmhead_quant.py`
  (`quantize_weight` → compressed-tensors `quantize`/`dequantize`/`calculate_qparams`,
  bit-width-agnostic). recon: quant→dequant→rel_err, no packing needed.
- Official PPL harness: `official/.../speed_benchmark/scripts/ppl_endpoint.py` +
  `data/ppl_ground_truth_tokens.jsonl` (128 records, context+target,
  61797 scored target tokens). PPL = exp(mean NLL of target tokens given context).
  **Replicable offline** with a teacher-forced forward — no serving needed.

## Plan
**Step 1 — OFFLINE per-layer sensitivity map (GPU-cheap, core deliverable):**
- (a) **Weight-recon error**: reference = dequantized-W4 (`W4deq`, what the base
  actually serves). For each (layer, proj) requantize W4deq → {W4,W3,W2} g32 sym,
  measure rel_err / MSE / SQNR. Pure CPU/GPU tensor math, no forward.
  Caveat: W4deq (not original QAT-HP) is the reference because only the packed
  W4 form is published — this is the faithful "buildable from the shipped ckpt"
  number and is exactly reproducible.
- (b) **PPL delta**: teacher-forced forward over the official 128 records.
  Baseline = all-W4 body. Per layer: fake-quant that one layer's MLP to W3 (then
  W2), recompute PPL, ΔPPL vs base. Subset (e.g. 32 recs) for the per-layer
  ranking sweep; full 128 for the chosen config(s).
- Deliverable: per-layer sensitivity TABLE + bytes-saved-vs-PPL Pareto front,
  committed here. Reusable for ANY precision-reduction lever (mine/wirbel/fern).

**Step 2 — Pick config + SERVE VIABILITY KILL-GATE:**
- Choose mixed config (robust→W3/W2, sensitive→W4) maximizing bytes-saved within
  PPL budget. THEN check: does compressed-tensors / Marlin expose a real W3 or
  per-layer-mixed kernel that SERVES on sm_86 (A10G, Ampere) under vLLM 0.22.0 —
  sparse-of-dense, NOT a dense bf16 fallback (which reads MORE bytes)?
- **If no sub-4-bit kernel serves on sm_86, STOP** — post exact wall (kernel
  name / error / fallback mode), close the serve-lane. Step-1 map still stands.
  Do NOT hand-roll a kernel.

**Step 3 — PPL + SPEED + QUALITY (only if Step 2 serves):**
- PPL (kill if >2.42); single-stream decode TPS (conc=1, out_len 512) A/B vs
  int4head; 4-axis quality (MMLU-Pro, GSM8K, AIME, GPQA-Diamond).
- Report bytes-saved → measured TPS vs bandwidth-bound prediction.

## STEP 2 SERVE KILL-GATE — VERIFIED FAIL (sub-4-bit does NOT serve on sm_86)
Verified in the installed vLLM 0.22.0 (`.venv/lib/python3.11/site-packages/vllm`):
1. **compressed-tensors WNA16** (`.../quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py`):
   `WNA16_SUPPORTED_TYPES_MAP = {4: uint4b8, 8: uint8b128}` (L35); `__init__` raises
   `ValueError("Unsupported num_bits = {3|2}. Supported num_bits = dict_keys([4, 8])")`
   (L66-70) — at SCHEME CONSTRUCTION during model load, BEFORE `create_weights`/kernel
   dispatch. A W3/W2 compressed-tensors body fails to LOAD. NOT a silent dense fallback.
2. **Marlin type system** (`.../quantization/utils/marlin_utils.py`
   `query_marlin_supported_quant_types`, L75-83): on Ampere (cap≥75) the only integer
   weight types are `uint4`/`uint4b8` (4-bit) and `uint8b128` (8-bit). No uint3/uint2
   exists. (float4_e2m1f is a 4-bit FLOAT, still not sub-4-bit int.) All Marlin-backed
   paths — compressed-tensors, auto_gptq, awq_marlin — funnel through this.
3. **Machete** (sub-4-bit-capable on Hopper) is gated to sm_90 and group_size∈{-1,64,128}
   — excluded on A10G (sm_86, g32) on both grounds.
Conclusion: there is NO sub-4-bit weight-only kernel on sm_86 under vLLM 0.22.0.
The W3/W2 SERVE-LANE IS CLOSED. Step 3 (PPL/TPS/quality on a served variant) is dead.
Do NOT hand-roll a kernel (PR + program rule). Even a hypothetical dense-bf16 fallback
would READ MORE bytes, inverting the bandwidth premise.
Step-1 offline map (recon + PPL) STANDS as the reusable deliverable.

## Cap (#784)
If Step 1 shows no layer tolerates sub-W4 in budget, OR Step 2 finds no serving
kernel → summarize negative and stop; don't over-instrument.

## Bytes/scale note (g32 overhead)
W4 MLP/layer ≈ 39.3 MB weights + 4.9 MB bf16 scales = 44.2 MB. W3 = 34.4 (−22%),
W2 = 24.6 (−44%). bf16 scales at g32 are a FIXED overhead that shrinks the
sub-4-bit byte advantage below the naive 25%/50%.
