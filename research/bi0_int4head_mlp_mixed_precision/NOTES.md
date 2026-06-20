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

## Cap (#784)
If Step 1 shows no layer tolerates sub-W4 in budget, OR Step 2 finds no serving
kernel → summarize negative and stop; don't over-instrument.

## Bytes/scale note (g32 overhead)
W4 MLP/layer ≈ 39.3 MB weights + 4.9 MB bf16 scales = 44.2 MB. W3 = 34.4 (−22%),
W2 = 24.6 (−44%). bf16 scales at g32 are a FIXED overhead that shrinks the
sub-4-bit byte advantage below the naive 25%/50%.
