# fa2sw_onegraph

int4 QAT Gemma endpoint (`google/gemma-4-E4B-it-qat-w4a16-ct`, Marlin W4A16)
with two **env-gated, target-side runtime levers** that are **OFF by default**.

| env | lever | what it does |
|-----|-------|--------------|
| `FA2SW=1`   | fa2sw   | Neutralise Gemma4Config's heterogeneous-head-dim TRITON force-pin so per-head selection routes the 35 sliding hd=256 local layers to `FLASH_ATTN` (FA2, honouring `sliding_window=512`) while the 7 global hd=512 layers stay on `TRITON_ATTN` (FA caps head_size at 256). Also drops `FLASHINFER` from the sm_86 priority so the hd=512 layers don't pick a kernel that can't dispatch head_dim=512. |
| `ONEGRAPH=1`| onegraph| `--compilation-config {"cudagraph_mode":"FULL"}` — capture the whole decode step as one full CUDA graph instead of the default `FULL_AND_PIECEWISE`. |

## Why default OFF (negative result, PR #7)

Controlled conc=1 ablation on the AWS A10G (`research/fa2sw_onegraph/`),
clean M=1 AR regime (sequential, no prefix cache — the only int4-deterministic
regime; prefix-cache + batching flips int4 near-ties):

| variant | backend map | TPS (256 tok) | Δ vs base | greedy vs base |
|---------|-------------|---------------|-----------|----------------|
| base     | 35×hd256 TRITON / 7×hd512 TRITON      | 96.89 | —      | reference |
| fa2sw    | 35×hd256 **FLASH_ATTN** / 7×hd512 TRITON | 92.11 | **−4.9%** | DIVERGENT (invalid) |
| onegraph | 35×hd256 TRITON / 7×hd512 TRITON      | 96.82 | +0.0% (parity) | DIVERGENT (invalid) |
| both     | 35×hd256 **FLASH_ATTN** / 7×hd512 TRITON | 92.12 | −4.9% | DIVERGENT (invalid) |

- **No TPS win.** Decode is ~92% weight-GEMM / bandwidth-bound at conc=1;
  attention is ~2.6% of GPU-busy time and CUDA graphs already collapse the
  decode step into one launch. fa2sw's mixed FA2+Triton backend also blocks a
  single full-graph capture, netting a regression. (Confirms the prior
  `gemma4_mtp_frontier_map_kitan` / `int4_ceiling_notes` / decode-profiler
  audits that attention + launch-overhead are conc=1 dead ends.)
- **Both levers break greedy identity.** int4 Marlin argmax near-ties flip
  under any numeric-path change; base int4 is cross-process bit-exact, so the
  flips are real, not run noise. A DIVERGENT endpoint is leaderboard-invalid
  regardless of speed.
- **fa2sw cannot be served through the API server anyway.** The fa2sw
  monkeypatch runs in the `serve.py` (APIServer) process, but vLLM V1 runs the
  model in a **separate EngineCore process**; the `FLASHINFER`-removal does not
  reach it, so the hd=512 global layers pick `FLASHINFER` and crash at CUDA-graph
  warmup (`Unsupported max_mma_kv: 0`). A real fa2sw serve path would need a
  vLLM general-plugin entry point so the patch runs inside the worker.

With both levers off, `serve.py` serves the verified int4 base:
loads, `/v1/models` ready, **all modalities answer through the live server**
(text→`Paris`, image→`Blue`, audio→`No.`; `ALL_OK=True`), 128/128 ShareGPT
greedy capture at output_len 512, served PPL **2.005** (≤ 2.42 cap). The
`vllm[audio]==0.22.0` manifest dependency is what makes audio *input* decode
work (it pulls `soundfile`/`av`/`torchaudio`); the model checkpoint itself
carries the full multimodal stack (`audio_config` + `vision_config`), so no
modality is removed or shortcut to win speed.

## Reproduce the levers (offline only)

```bash
# offline ablation matrix (base / fa2sw / onegraph / both) + greedy verdicts
research/fa2sw_onegraph/gate_official.sh
```
