STUDENT ubel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["7ntx4nrn"],"primary_metric":{"name":"decode_tps_local_median","value":265.61},"test_metric":{"name":"ppl_128rec","value":2.0031}}

## Results — int4head + PLE-input-gate de-quant: **CONFIRMED (+5.14% local decode TPS, quality holds), fire-ready (published + repointed, NOT fired)**

The lever lands. De-quantizing just the 42 `per_layer_input_gate` projections (N=256) from int4-Marlin back to bf16-cuBLAS buys **+5.14%** decode TPS over the int4head control in a rigorous same-session interleaved A/B — matching stark's #798 projection (+5.3%) almost exactly — with PPL flat, GSM8K up, all modalities/128 completions, and the official gate green. As a bonus, the bf16 path also **removes a cross-session greedy nondeterminism source** that the int4 gate introduced.

This was a **LOCAL-only** PR (build + serve + measure). **No HF Job launched, no submission made.**

---

### 1. Build delta (Step 1)
`build_lmhead_quant.py --num-bits 4 --head-group-size 32 --dequant-ple` (one new flag on the int4head recipe):
- **int4 g32 untied lm_head** (rel_err 0.0674, 3.56× smaller), **int4 W4A16 body** byte-identical to `google/gemma-4-E4B-it-qat-w4a16-ct` (2636 tensors copied), bf16 embeds, MTP K=6 drafter, force-2D attn — all unchanged from int4head.
- **PLE delta:** 42× `per_layer_input_gate` → bf16, and `re:.*per_layer_input_gate` added to `quantization_config.ignore` (confirmed: ignore now ends with that regex; `config_groups` = group_0 `['Linear']` 4b/g32, group_1 `['re:.*lm_head']` 4b/g32).
- **bf16 source = DEQUANT of the existing int4 gate** (packed+scales → bf16), **not** `google/gemma-4-E4B-it`. Rationale (measured in `inspect_ple.log`): the google bf16 gate is **16–27% rel-divergent** (cos≈0.99) from the QAT int4 parent — i.e. it is **not** the higher-precision superset of these QAT-trained weights, while dequant reconstructs the **exact served values**. Dequant gives a clean kernel-isolated A/B (only the GEMM kernel changes, weight values identical) + guaranteed quality-neutrality. Per-store rel-err of the bf16 round = 0.00137 mean.

### 2. Serve dispatch proof (Step 2) — `verify_dispatch.log`, **PASS**
Two independent proofs agree (vLLM `should_ignore_layer` static predicate + live module walk of the constructed model):

| module | quant_method | scheme | kernel |
|---|---|---|---|
| `per_layer_input_gate` (L0, L20) | **UnquantizedLinearMethod** | None | **bf16 / cuBLAS** ✅ TARGET |
| `per_layer_projection` (sibling, N=2560) | CompressedTensorsLinearMethod | CompressedTensorsWNA16 | int4 / Marlin (stays int4) |
| `self_attn.o_proj`, `mlp.*` (body) | CompressedTensorsLinearMethod | CompressedTensorsWNA16 | int4 / Marlin |
| `lm_head` | CompressedTensorsLinearMethod | CompressedTensorsWNA16 | int4 / Marlin |

The de-quant took effect: PLE-input-gate serves bf16/cuBLAS; body + head + the PLE sibling stay int4/Marlin.

### 3. Decode TPS A/B (Step 3) — 128×512, warm median, 3 reps, **interleaved**, conc=1
| arm | rep1 | rep2 | rep3 | median | mean | CV | E_accept |
|---|---|---|---|---|---|---|---|
| **+PLE-dequant** | 265.94 | 265.30 | 265.61 | **265.61** | 265.62 | **0.10%** | 3.383 |
| int4head control | 252.52 | 254.02 | 252.62 | 252.62 | 253.05 | 0.27% | 3.379 |

- **Δ = +12.99 tok/s = +5.14%** (in-harness A/B) → **clears the ≥ +4% confirm gate**; matches the +5.3% projection.
- **E_accept identical** (3.383 vs 3.379): the drafter acceptance rate is untouched, so the gain is pure verify-side kernel time removed — exactly the serial conc=1 ~1:1 kernel-time→wall-clock conversion the PR predicted. The saving was **not** hidden.
- Honesty note on the anchor: the PR quotes the control at 256.74 (a prior-session number). The fresh control re-measured at **252.62** (~1.6% cross-session TPS drift). The **interleaved same-session A/B (+5.14%) is the trustworthy lever measurement**; comparing pledequant-now to the stale 256.74 anchor would give +3.45% but conflates the kernel lever with session drift — which is precisely why an in-harness control was re-measured.

### 4. Quality gate (Step 4) — all hold or improve
| metric | +PLE-dequant | int4head control | floor / cap | verdict |
|---|---|---|---|---|
| **PPL** (128-rec) | **2.0031** | 2.0029 | ≤ ~2.42 cap | ✅ flat (+0.0002) |
| **GSM8K greedy** (N=200, 8-shot) | **0.925** (185/200) | 0.9150 | ≥ 0.807 | ✅ **beats control** |
| GSM8K sampled (N=200) | 0.890 (178/200) | — | — | ✅ |
| completions | **128/128** | 128/128 | 128 | ✅ |
| all-4 modalities | **loaded** | loaded | required | ✅ |
| `validate_submission --official-gate` | **PASS** | PASS | — | ✅ |

**Within-job greedy-identity vs int4head control:** verdict DIVERGENT, **109/128 divergent, first_token_flips = 0**, onset median tok 132 (late). This is **not** a regression — it is the documented int4 cross-session greedy nondeterminism floor (#784: greedy-identity is internal-only), and the cross-session test below proves it. Numerics legitimately change (int4-Marlin → bf16-cuBLAS for the gate), but the real quality axes (PPL, GSM8K, official gate, modalities) all hold/improve, consistent with bf16 ⊇ the int4 values.

### 5. Cross-session determinism (bonus finding) — `xsession.log`
Each build vs **itself** across two independent serve sessions, **speculation OFF**:
- **+PLE-dequant: 0/128 divergent** → bit-stable across sessions.
- **int4head control: 98/128 divergent** → NOT stable across sessions.

The two builds differ **only** in the 42 PLE gates (int4-Marlin vs bf16-cuBLAS), so the int4head cross-session nondeterminism is attributable to the int4 Marlin GEMV at the starved N=256 tile; the bf16 cuBLAS path **removes** it. So most of the 109/128 cross-checkpoint A/B divergence above is the control's own session noise (98/128), not the PLE change. **The lever isn't only faster — it's more deterministic.** (Caveat: one session-pair per build; reported as observed, not over-claimed.)

### 6. Fire-readiness (Step 5) — published + repointed, **NOT fired**
- Build published to **private** Hub repo `gemma-challenge/gemma-4-e4b-it-int4-mtp-bi0-int4head-pledequant` @ rev `f5a0dfd1caa52b429b6a0e973b53d2aac8e14a22` (model.safetensors 10.589 GB + config/tokenizer/chat-template; `HfApi` file-completeness confirmed).
- `submissions/int4_mtp_bi0_int4head_pledequant/` written; **`manifest.json` MODEL_ID repointed to the Hub repo** (was the local `/workspace/...` path). serve scaffolding (`serve.py`, `sitecustomize.py`, both attn patches) is **byte-identical** to the proven `int4_mtp_bi0_int4head` control → clean +1-delta submission.
- Remote-load evidence: Hub `config.json` (the dispatch-determining file) is **byte-identical** to the locally dispatch-verified build; safetensors integrity guaranteed by HF content-addressed LFS + size match; identical scaffolding already serves the int4head control from an analogous Hub repo.
- **Deferred:** the full 10 GB Hub-repo load + greedy smoke is blocked by tight HF-cache disk on this pod (6.7 GB free < 10.6 GB) and is **not** required by this no-fire PR — it should be the first step of the separate HF-approval fire issue.

### Exact commands
```bash
# build (one new flag)
python submissions/int4_mtp_bi0_int4head/build_lmhead_quant.py --num-bits 4 --head-group-size 32 --dequant-ple
# dispatch proof
BUILD=/workspace/gemma_build/bi0_int4head_pledequant python research/int4head_ple_dequant/verify_dispatch.py
# TPS A/B (3 reps interleaved, 128x512), quality (validate --official-gate + GSM8K), x-session determinism
python research/int4head_ple_dequant/tps_ab_interleaved.py
python research/int4head_ple_dequant/quality.py
python research/int4head_ple_dequant/xsession_test.py
# publish private Hub repo
python research/int4head_ple_dequant/publish_hub.py
```

### Peak memory
Model weights load **10.2 GiB**, KV cache **8.1 GiB** (326,736 tokens, 79.8× concurrency at 4096), served at `gpu_memory_utilization=0.90` on a 22.5 GiB A10G — comfortably inside the 24 GB official `a10g-small` envelope.

### W&B
Run `ubel/pledequant` (id **`7ntx4nrn`**), group **`bi0-int4head-ple-dequant`**, project `wandb-applied-ai-team/gemma-challenge-senpai`. Logs TPS A/B + per-rep table, PPL/GSM8K/greedy/gate, x-session determinism, publish record, and PR baselines.
https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/7ntx4nrn

### Public evidence used
stark's #798 re-profile (W&B `tgdo0imp`): int4-Marlin 20.8 µs vs bf16-cuBLAS 5.8 µs at N=256 (the only 1 of 11 body GEMM shapes where int4 loses); int4head published Hub repo + #802 publish pipeline.

---

### What happened
The projected kernel lever materialized served, essentially on the nose: removing the int4 PLE-input-gate Marlin GEMV (starved at N=256) and replacing it with bf16 cuBLAS recovered **+5.14%** decode TPS at conc=1, with **identical E_accept** confirming it's pure verify-side kernel time (not a speculation artifact). Quality is neutral-to-better: PPL flat (2.0031 vs 2.0029), GSM8K greedy **up** to 0.925 (> control 0.9150 > floor 0.807), 128/128, all modalities, official gate PASS. The dequant-source decision (dequant the QAT int4 gate, **not** the google bf16 base) was the right call — the google bf16 weights are 16–27% rel-divergent from the QAT parent and would have injected untrained weights. The standout secondary result: the int4 N=256 Marlin gate was a **cross-session greedy nondeterminism source** (98/128 self-divergence across sessions); bf16-cuBLAS makes the build bit-stable (0/128). Net: a clean, orthogonal, quality-neutral, determinism-improving +5% that stacks on int4head.

### Suggested follow-ups
1. **Fire it.** Meets Step 5's confirm bar (≥ +4% TPS, quality holds). Open the HF-approval issue for `int4_mtp_bi0_int4head_pledequant`; first action there = the full Hub-repo load + greedy smoke deferred above (needs disk headroom).
2. **Sweep the N=256 GEMM family.** The cross-session determinism win suggests other narrow-N int4 projections (any N≤256 Marlin GEMV) may be both pessimized *and* nondeterministic — worth a per-shape int4-vs-bf16 µbench to find more dequant candidates beyond the PLE gate.
3. **Re-confirm the x-session determinism finding** with ≥3 session-pairs per build before treating "bf16 dequant removes nondeterminism" as load-bearing for any greedy-identity claim.
