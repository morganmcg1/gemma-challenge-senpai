STUDENT fern:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"analysis_only":true,"official_tps":0,"wandb_run_ids":["m7y0qg5h"],"primary_metric":{"name":"serving_kernel_viable_w4_2of4_sm86","value":0},"test_metric":{"name":"ppl_measured","value":0},"verdict":"VIABILITY_KILL__VLLM_0220_REMOVED_COMPRESSED_TENSORS_SPARSITY","kill_gate":"step1_viability","ppl_gate_reached":false,"loader_error":"DeprecationWarning: Sparsity support has been removed from compressed-tensors"}

## Results

**Verdict: KILLED at Step 1 (viability), before any GPU spend.** On our pinned serving stack, there is **no sparse-aware kernel path for W4 + 2:4 on sm_86** — and worse, the serving loader **hard-fails at load** the moment it sees any `sparsity_config`. This is the cleanest form of the kill condition the PR asked for ("no working sparse kernel … load error … STOP"). I did **not** proceed to Step 2/3, did **not** install `llm-compressor`, and did **not** hand-roll a kernel (per the PR's explicit constraint).

### The dispositive wall — the serving stack rejects sparsity at load

I drove the **real** vLLM load-time config parser (`CompressedTensorsConfig.from_config`, the exact function vLLM calls during engine init) on a faithful reproduction of what `llm-compressor` writes for a 2:4 + int4-W4A16 body-MLP export. It raises:

```
DeprecationWarning: Sparsity support has been removed from compressed-tensors.
Please use a model without sparsity configuration.
```

Source: `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py`
`from_config()` (L228) → `_parse_sparsity_config()` (L263) raises on **any non-empty `sparsity_config` with targets**, regardless of `format` / `sparsity_structure` / group size. The only `sparsity_config` it tolerates is an empty one — i.e. a dense model, which defeats the entire hypothesis.

### Three independent confirmations on the installed binary (no GPU used)

Probe: `research/body_mlp_2of4_sparsity/viability_probe.py` (reproducible; config-parse only, never loads a model). Output:

```
==== PINNED STACK ====
  vllm==0.22.0  transformers==5.9.0  compressed-tensors==0.15.0.1  torch==2.11.0
  device capability: sm_86

==== INSTALLED compressed_tensors SCHEMES ====
  ['compressed_tensors_w4a16_nvfp4.py', 'w4a4_mxfp4', 'w4a4_nvfp4', 'w4a8_fp8',
   'w4a8_int', 'w8a16_fp8', 'w8a8_fp8', 'w8a8_int8', 'w8a8_mxfp8', 'wNa16.py']
  -> 2:4 sparse scheme file present? False        # no compressed_tensors_w4a16_24.py

==== COMPILED SPARSE/MARLIN-24 KERNELS in torch.ops._C ====
  marlin/sparse/24 ops: ['awq_marlin_repack', 'gptq_marlin_repack', 'marlin_gemm']
  -> gptq_marlin_24_gemm compiled? False          # dense Marlin only; no 2:4 kernel

==== DRIVE THE REAL vLLM LOAD-TIME PARSER on a 2:4 + W4A16 config ====
  RAISED DeprecationWarning: Sparsity support has been removed from compressed-tensors.
  VERDICT: serving stack rejects W4 + 2:4 at load -> lane DEAD at viability.
```

1. **No 2:4 scheme class** — `CompressedTensorsW4A16Sparse24` (the W4+2:4 routing class) is gone from `model_executor/`; no `Sparse24` / `sparse_w4a16` reference anywhere.
2. **No compiled kernel** — `gptq_marlin_24_gemm` (Sparse-Marlin) is **not** in `torch.ops._C`. Only dense `marlin_gemm` + repack ops are compiled. There is nothing for a sparse checkpoint to route to even if the scheme class existed.
3. **Loader actively rejects it** — the explicit `raise` above. Not a dense fallback that we'd have to detect — a hard, named load failure.

(The only "sparse" hits in the tree are flash-attention *block*-sparsity and inert CUTLASS `OpClassSparseTensorOp` C++ template headers under `third_party/deep_gemm/` for sm_80/100/120. Neither is a W4A16+2:4 Ampere weight-GEMM with a Python binding; using them is exactly the hand-rolled-kernel path the PR forbids.)

### Version-timeline note (so this isn't second-guessed against public docs)

Public vLLM `0.22.0` (the OSS release tag) predates the upstream Sparse24 removal, so a naive version lookup would suggest the kernel "should" be present. **It is not in our installed binary.** This pod's pinned `vllm==0.22.0` / `compressed-tensors==0.15.0.1` already has sparsity stripped — verified by direct source inspection **and** by driving the live loader. The installed binary is authoritative here; public release-date reasoning does not apply.

### Corroborating obstacles (each independently fatal even if serving worked)

These are *downstream* of the serving wall, so they don't change the verdict — but they confirm the lane is dead from multiple directions (literature pass + W&B `m7y0qg5h`):

- **Artifact production is also gone.** `llm-compressor` deprecated its `marlin24` compressor (tracking issue vllm-project/llm-compressor#2267, 2026-01-20) and upstream vLLM removed Sparse24 in vllm-project/vllm#36799 (merged 2026-03-23). Current docs state 2-of-4 sparsity "is no longer supported … due to lack of hardware support and user interest." The official `quantization_2of4_sparse_w4a16` example 404s. So even producing the checkpoint would require pinning a pre-deprecation toolchain — and it still wouldn't load on our server.
- **Group-size mismatch.** Sparse-Marlin (IST-DASLab, arXiv 2408.11743) is restricted to `group_size=128`. Our int4 lm_head is `g32` and the body inherits the QAT-w4a16-ct grouping; a `g32`/`g64` int4 body would not route to the sparse kernel regardless.
- **Quality band risk.** Literature on stacked 4-bit + 50% 2:4 at sub-7B scale (e.g. arXiv 2511.08360) puts combined degradation in the ~0.15–0.40+ PPL range *without* finetuning, and larger per-stage at ~4B; GPQA-Diamond is the most fragile axis. The +0.4 PPL band (≤2.42 vs our 2.0029 floor) would likely be consumed on PPL alone. This matches the PR's own "HIGH-quality-risk swing" framing.

### Reproduce / commands used

```bash
cd target/
# 1. confirm stack + scan for any 2:4 kernel + drive the real loader (NO GPU):
/tmp/senpai-venvs/<server-hash>/bin/python research/body_mlp_2of4_sparsity/viability_probe.py
# (server-hash = the venv backing the running int4head server, e.g. 20f658587e8a6643)
```

- **Peak memory:** ~0 incremental. The probe never loads a model — it only parses a config dict and initializes a CUDA context for the `sm_86` capability query. The running int4head control server (`/workspace/gemma_build/bi0_int4head_g32`, port 8020, ~20.3 GB) was left untouched.
- **W&B run:** `m7y0qg5h` (analysis-only viability record — stack versions, kernel-presence booleans, loader error, verdict). https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/m7y0qg5h
- **No HF Job launched** (local A10G only, per PR + operator rules).

### Public evidence used

- Motivation is the advisor-supplied profiler reference already in the PR body — stark #798 decode breakdown (W&B `dpc36210`): body verify-GEMM 5.92 ms / 12.42 ms-tok (47.6%), MLP ~68% of body @ 74–79% HBM BW — i.e. the wall this PR aimed to attack with fewer weight reads.
- Control baseline = int4head (256.74 TPS local, PPL 2.0029) per the PR baseline block.
- The kill itself is grounded in the pinned-stack source + upstream OSS removal (vLLM#36799, llm-compressor#2267); it did **not** require the public leaderboard/board.

### What happened — honest analysis

The hypothesis was mechanically sound (2:4 on Ampere sparse tensor cores genuinely reads ~half the weight elements, and it is orthogonal to the W4A8 activation-path lane). It dies on a pure **tooling/stack** fact, not on physics: **our pinned vLLM has had compressed-tensors sparsity support removed**, so a W4+2:4 checkpoint cannot be served at all — there is no kernel to route to and the loader refuses the config. This is the fast, clean negative the PR explicitly wanted, and it cost ~0 GPU (config-parse + a literature pass). The corroborating obstacles (artifact toolchain removed, g128 constraint, quality risk) mean that even resurrecting the serving path would be a large out-of-scope stack downgrade with a likely PPL-band failure waiting at the end — not worth it.

### Suggested follow-ups

- **Close this lane.** Structured 2:4 sparsity (W4+2:4 *or* W8+2:4) is dead on the current pinned serving stack — the deconfound W8+2:4 arm is moot for the same loader reason, so I did not run it. Any future structured-sparsity idea is blocked until/unless the serving vLLM is intentionally moved to a build that re-includes a 2:4 kernel (a deliberate infra decision, not an experiment delta).
- **Redirect the "fewer weight reads on the MLP wall" intent** to mechanisms our stack *does* support: (a) wirbel's W4A8 activation-path lane (#807) is the live orthogonal attack on the same wall; (b) the PLE-dequant +5.3% lever (#798) is already identified; (c) sub-int4 weight-only (e.g. NVFP4/MXFP4 schemes — note `compressed_tensors_w4a4_nvfp4.py` / `w4a4_mxfp4.py` *are* present in our binary) could cut MLP weight bytes further, though those are A16→A4 activation paths and carry their own quality risk and sm_86 support questions worth a separate viability probe.
- If structured sparsity is ever revisited, it should come in as an **infra/stack PR** (pin a vLLM build with the Sparse-Marlin kernel + verify int4-body group_size=128) *before* any quality experiment is assigned — the experiment is pointless until the serving path exists.
