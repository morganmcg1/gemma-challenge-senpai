STUDENT wirbel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"analysis_only":true,"official_tps":0,"wandb_run_ids":["vxh2u99u","aq85tput"],"primary_metric":{"name":"local_warm_median_tps_stock_byteexact","value":127.787},"test_metric":{"name":"official_int4_g128_lmhead_tps_ref","value":126.378},"engine_or_lever":"vllm-0.22.0 M=1 attention-backend env knob (FLASHINFER/FLASH_ATTN both force-resolve to TRITON_ATTN) + CUDA-graph-on roofline","local_tps":{"stock_byteexact":127.787,"flashinfer":127.727,"flash_attn":127.704},"tps_delta_vs_126":{"best_byteexact_knob_vs_stock_local":-0.061,"local_stock_vs_official_ref":1.409,"note":"126.378 is OFFICIAL HF-Job sglang, NOT locally reproducible; on-harness knob delta is ~0"},"byte_exact_319":{"bool":true,"n_prompts":64,"n_mismatches":0,"basis":"each knob cold-r1 vs stock cold-r1","warm_pass_self_det_caveat":"34/64 under default prefix caching — orthogonal SURFACE"},"ppl":{"value":2.019,"basis":"byte-exact tokens => identical logits => locked PPL; not re-measured"},"verdict":"ZOOMOUT_NO_CHEAP_LEVER","surface":"prefix_cache_warm_pass_self_determinism_34of64"}

## Results

**Verdict: `ZOOMOUT_NO_CHEAP_LEVER`** — the unswept vLLM 0.22.0 M=1 backend-knob axis is structurally inert at the strict `int4_g128_lmhead` rung, and the roofline proves no byte-exact knob can supply +10 TPS. **Plus an orthogonal determinism SURFACE** (not a speed lever, not a fire): stock M=1 greedy is **not warm-pass self-deterministic** (34/64), driver = `enable_prefix_caching`; official cold-pass scoring is unaffected.

All LOCAL on the assigned A10G. `analysis_only=true`, `official_tps=0`. **NO HF Job, NO submission.**

### 1. The attention-backend env knob is structurally inert — all configs collapse to `TRITON_ATTN`

vLLM 0.22.0 **hard-forces** `TRITON_ATTN` for Gemma4's heterogeneous head_dim (256 local / 512 global), overriding `VLLM_ATTENTION_BACKEND` entirely:

```
config.py:100 Gemma4 model has heterogeneous head dimensions (head_dim=256, global_head_dim=512).
              Forcing TRITON_ATTN backend to prevent mixed-backend numerical divergence.
cuda.py:318   Using AttentionBackendEnum.TRITON_ATTN backend.
```

This held for **all three** env settings → the knob changes nothing. Single-stream (conc=1, seed=1, 64×512, `ignore_eos`, warm-median; same recipe as the #533 base-int4 floor):

| config | env | resolved backend | local warm-median TPS | Δ vs stock | byte-exact vs stock cold-r1 (n=64) | peak VRAM |
|---|---|---|---|---|---|---|
| **stock** (ref) | — | TRITON_ATTN | **127.787** | — | — | 19.86 GB |
| flashinfer | `VLLM_ATTENTION_BACKEND=FLASHINFER` | TRITON_ATTN (forced) | 127.727 | **−0.061** | **True (0/64)** | 19.86 GB |
| flash_attn | `VLLM_ATTENTION_BACKEND=FLASH_ATTN` | TRITON_ATTN (forced) | 127.704 | **−0.084** | **True (0/64)** | 19.86 GB |

No knob is faster (Δ within ±0.08 = noise) and all are byte-exact (identical code path). This empirically closes the last item on the PR's candidate menu — corroborating the alt-engine closures (#498/#558/#502/#507) and the roofline's `attention_pinned_to_only_byteexact_path`.

### 2. Roofline: no byte-exact knob *can* supply +10 (self-tested, reproducible, CPU-only)

bytes/token measured from the w4a16-ct safetensors header. `int4_g128_lmhead = 2.4165 GB/token` (body int4 1.9864 + scale_g128 0.0621 + head_int4 0.3355 + head_scale 0.0105 + KV 0.022). Two-term fit on the two **official** anchors (base-int4 99 @ 3.5989 GB ; int4-head 126.378 @ 2.4165 GB), `t_token = bytes/BW_marginal + c_fixed`:

- `BW_marginal = 540.3 GB/s = 90.1% of the 600 GB/s A10G peak` — the marginal byte-stream is already near peak.
- `c_fixed = 3.441 ms/token = 43.5%` of the 7.913 ms token time at 126.378.

Headroom to reach 136.378 (+10), three mutually-exclusive levers:

| lever | what it needs | byte-exact knob can deliver? |
|---|---|---|
| **A** raise effective BW | **+7.91%** eff BW at fixed bytes | **No** — Marlin is the *only* w4a16 kernel on sm_86 (#550); split-K refuted −5.82 (#433) |
| **B** cut fixed overhead | **−0.58 ms** (−16.9%) of `c_fixed` | **No** — that ≈ the *entire* 0.573 ms pure launch floor (#554), and CUDA-graph already captures it on stock |
| **C** read fewer bytes | **−0.1772 GB** (−7.3%) | **Not byte-exact** — sub-int4/sparse relaxes #319 (option C, already dead at #613) |

`byteexact_knob_can_supply_plus10 = false`. The strict-#319 AR +10 lever is empty for a structural, hardware reason — confirming the on-branch "AR-frame = A10G HBM ceiling" line.

### 3. SURFACE (not a fire, not a speed lever): M=1 greedy is not warm-pass self-deterministic — driver is prefix caching

The required self-determinism census surfaced a real anomaly. Same server, same 64 prompts, greedy (T=0), `prompt_sha` parity holding, **two back-to-back passes diverge in 34/64 completions** (some at decode position 0). I isolated the mechanism with a standalone A/B (identical stock flags, single toggle; does **not** touch `submissions/int4_g128_lmhead/serve.py`):

| pair | byte-exact | mismatch | reading |
|---|---|---|---|
| stock r1-vs-r2 (cache ON, serve.py) | False | **34/64** | warm-pass self-determinism fails |
| standalone-ON r1-vs-r2 (control) | False | **34/64** | reproduces stock *exactly* (identical 34 indices + divergence positions) |
| **standalone-OFF r1-vs-r2** | **True** | **0/64** | **`--no-enable-prefix-caching` restores warm-pass identity** |
| standalone-ON-r1 vs stock-r1 | True | 0/64 | harness ≡ serve.py (validates the A/B) |
| OFF-r1 vs stock cold-r1 | False | 53/64 | caching-OFF is a *different* (but stable) token stream |
| flashinfer/flash_attn-r1 vs stock-r1 | True | 0/64 | env-knob arms are byte-identical (same backend) |

**Mechanism:** `enable_prefix_caching=True` (vLLM V1 default). Pass-2 reuses pass-1's block-cached prefix KV under a different chunk-boundary alignment than pass-1's cold full chunked prefill → int4-Marlin grid-tie flips at greedy argmax (same int4-tie family as **#616**'s 0.43% / **#607** / **#621**, here on the *prefix-cache-state* axis rather than batch-size).

**Why this does NOT break the live submission or #319:**
- The official sglang TPS bench and the **#319 cross-start identity gate run COLD** (fresh server / one shot per prompt). Every cold first-pass is byte-identical (stock-r1 = flashinfer-r1 = flash_attn-r1 = standalone-ON-r1) — this *corroborates* lawine **#606**'s cross-*start* 128/128. The 34/64 only appears on **cross-*pass* warm-cache reuse**, an axis #606's cold-start method never exercised.
- TPS is unaffected (all completions full-length 512; flips are <0.5 nat ties).

**The speed-neutral determinism option (a SURFACE for you, not something I'd ship):** `--no-enable-prefix-caching` gives warm-pass identity (0/64 self) at **−0.256 TPS** (127.53 vs 127.79, noise) — prefix caching is ~0 benefit at unique-prompt M=1. **Caveat:** it changes the token stream vs the current cold reference (53/64), so adopting it would require **re-locking** the #319 reference; it is *not* a drop-in preserver of the existing locked tokens.

### Commands

```bash
cd target/
# 3-arm backend-knob speed + byte-exact census (run vxh2u99u)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python research/zoomout_ar_speed_screen/serve_census.py \
  --wandb_name wirbel/zoomout-ar-speed-census --wandb_group zoomout-ar-speed-screen
# roofline (CPU-only, self-tested)
.venv/bin/python research/zoomout_ar_speed_screen/roofline.py
# prefix-cache mechanism A/B (standalone server; does not touch serve.py)
.venv/bin/python research/zoomout_ar_speed_screen/prefix_cache_diag.py --prefix-cache off --port 8011
.venv/bin/python research/zoomout_ar_speed_screen/prefix_cache_diag.py --prefix-cache on  --port 8012
.venv/bin/python research/zoomout_ar_speed_screen/compare_all.py            # determinism matrix
.venv/bin/python research/zoomout_ar_speed_screen/finalize_determinism.py   # run aq85tput
```

### Peak memory
19.86 GB (all arms, gpu-mem-util 0.90 on the 23 GB A10G).

### W&B
- `vxh2u99u` — 3-arm speed + byte-exact census (verdict `ZOOMOUT_NO_CHEAP_LEVER`)
- `aq85tput` — prefix-cache determinism diagnostic (the SURFACE), group `zoomout-ar-speed-screen`

### What happened — honest analysis
The PR's hypothesis (an un-run byte-exact backend knob worth ≥+10 TPS) is **falsified on two independent legs**: (i) empirically, the only env-injectable knob (`VLLM_ATTENTION_BACKEND`) is *inert* because vLLM force-pins `TRITON_ATTN` for the het head_dim, so all configs are the same kernel at the same TPS; (ii) analytically, the roofline shows the residual is split between a 90.1%-of-peak marginal byte-stream (no faster byte-exact GEMM exists on sm_86) and a fixed overhead whose removable part (≈the 0.573 ms launch floor) is already CUDA-graph-minimized — the +10 ask exceeds what either byte-exact term can give. CUDA-graph-vs-eager and Marlin-variant were priced by the roofline rather than re-served because eager only *adds* launch overhead and Marlin is the sole kernel. The genuinely new datum is the determinism SURFACE: the strict rung is byte-exact across cold starts (as #606 found) but not across warm-cache passes, and I pinned that to prefix caching with a clean within-harness A/B. It doesn't move the speed verdict and doesn't threaten the live cold-scored submission, but it does mean "byte-exact vs the AR reference" is only well-defined for cold passes — worth your awareness for any future warm-reuse config or a stricter #319 framing.

### Suggested follow-ups
- **None for speed** — the strict-#319 AR axis is closed by hardware (band-aid knobs exhausted); the only >126.378 paths remain Option-B (int4+spec, #319-broken) and Option-C (sub-int4, dead). I'd not spend more student-GPU on AR knobs.
- **(SURFACE, your call)** If a stricter/warm #319 framing is ever wanted, `--no-enable-prefix-caching` buys warm-pass determinism at ~0 TPS but needs a re-lock of the reference tokens (53/64 differ from today's cold reference). Cheap to formalize if useful; I did not change the submission.
- **(determinism map, optional)** The same prefix-cache flip likely affects *within-run* cross-prompt determinism when prompts share a system/chat prefix — untested here; a one-server, shared-prefix probe would close it.

### Public evidence used
On-branch alt-engine closures (#498/#558/#502/#507) and the head-BW wall #550 / launch-floor #554 (cited in my intake + roofline); the int4-tie determinism family #616/#607/#621 and the cross-start identity baseline lawine #606 (which this refines on the warm-pass axis). No HF Jobs, no submission, no AWS-only numbers reported as challenge results.
