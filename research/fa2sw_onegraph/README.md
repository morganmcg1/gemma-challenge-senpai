# fa2sw + onegraph ablation (PR #7, denken) — local AWS A10G, LOCAL ONLY

Negative result. Both target-side runtime levers are **greedy-DIVERGENT** (leaderboard-invalid)
on the int4 base at conc=1, and neither gives a valid TPS win. Submission ships both **OFF**
(`submissions/fa2sw_onegraph`, serves the verified int4 base). See the submission README for the
shipping decision; this dir is the evidence.

## Authoritative result: `runs_official/`
Offline ablation matrix in the only int4-deterministic regime — **M=1 AR** (sequential, one request
at a time, prefix-cache OFF), greedy-checked with the **official** verifier
(`gemma_greedy_identity_verifier_flowian-powers/check_greedy_identity.py`). Reproduce with
`gate_official.sh`; full console in `runs_official/gate.log`.

| variant | attn backends (offline) | TPS (256 tok, conc=1) | greedy vs base |
|---|---|---|---|
| base     | 35×hd256 TRITON / 7×hd512 TRITON      | 96.89 ±0.01 | reference |
| fa2sw    | 35×hd256 **FLASH_ATTN** / 7×hd512 TRITON | 92.11 ±0.02 (**−4.9%**) | **DIVERGENT** 82/128 prompts (12075 tok) |
| onegraph | 35×hd256 TRITON / 7×hd512 TRITON      | 96.82 ±0.00 (parity) | **DIVERGENT** 1/128 (59 tok, near-tie flip @idx197) |
| both     | 35×hd256 **FLASH_ATTN** / 7×hd512 TRITON | 92.12 ±0.00 (−4.9%) | **DIVERGENT** 82/128 (11767 tok) |

256-token gate is sufficient for a DIVERGENT verdict (all first-divergences land ≤ idx208); a 512-token
gate would only add more divergence, never remove it.

## Determinism (so the divergences are real, not run noise)
- `runs/base_clean/decode_outputs.jsonl` == `runs/base_clean2/decode_outputs.jsonl` (identical sha256) →
  int4 base is **cross-process bit-exact**.
- `runs_official/base_eager.log` in-process determinism check: 0 divergent prompts/tokens.
- fa2sw and `both` diverge on the **identical 82 prompts at identical first-divergence indices** →
  reproducible mechanism, not noise.

## Served validation (real serve.py API path, levers OFF): `serve_runs/`
- `serve_runs/base/` — loads, `/v1/models` ready, 128/128 ShareGPT greedy @ output_len 512,
  served PPL **2.005** (≤ 2.42).
- `serve_runs/base_modality_recheck/recheck.log` — all modalities through the live server:
  text→`Paris`, image→`Blue`, audio→`No.`, **ALL_OK=True** (audio needs the `vllm[audio]` manifest dep,
  which the local `.venv` lacked at the first smoke).
- `serve_runs/both/` — fa2sw+onegraph **cannot start through the API server**: vLLM spawns a separate
  EngineCore process the in-process monkeypatch can't reach, so hd=512 layers pick FLASHINFER and the
  engine crashes at CUDA-graph warmup. A real fa2sw serve path needs a vLLM worker-plugin entry point.

## Mechanism (why no win)
Decode at conc=1 is ~92% weight-GEMM / bandwidth-bound; attention ≈2.6%, sampling ≈0.2%, and CUDA graphs
already collapse the decode step into one launch. fa2sw's mixed FA2+Triton backend blocks a single full
graph (net −4.9%); onegraph (`cudagraph_mode=FULL` vs `FULL_AND_PIECEWISE`) can't speed up an
already-graphed step. Both only perturb int4 near-tie argmaxes → invalid.

## Scripts
- `ablate.py` — one variant per process (levers are process-global): backend map, greedy capture,
  TPS, PPL, optional determinism self-check.
- `gate_official.sh` — the authoritative base/fa2sw/onegraph/both gate + official verifier verdicts.
- `serve_and_validate.sh` — full server validation for one lever config.
- `modality_recheck.sh` — focused served-base modality smoke (closes the audio gate).
- `modality_smoke.py` — text/image/audio readiness probe against a running endpoint.
