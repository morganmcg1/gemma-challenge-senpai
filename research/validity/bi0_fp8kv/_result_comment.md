STUDENT ubel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["s63tb03x"],"primary_metric":{"name":"output_tps","value":218.02},"test_metric":{"name":"ppl","value":2.0058}}

## Results

**Verdict: NEGATIVE — fp8 KV cache does not boot on A10G (sm_86) for this stack. The hypothesis is untestable on the target hardware; it is not a quality-collapse, it is a serve-time failure at three independent layers.** The reported `output_tps`/`ppl` above are the **bf16 fallback** the submission ships with (`KV_CACHE_DTYPE=auto` == byte-for-byte `int4_mtp_bi0_surgattn`); the fp8 **delta is zero because no fp8 config runs.** This re-confirms the prior fp8-KV dead-end (`wirbel #141`; BASELINE.md *"fp8 KV cache — rejected by A10G + Gemma4 attn"*, *"no sm_86 kernel"*) and pins the precise mechanism.

### What happened — three-layer block (each patch past one layer hits the next)

The PR premise ("FP8 KV is natively supported in vLLM 0.22.0") does not hold for the **int4 compressed-tensors target + force-2D `TRITON_ATTN` + torch.compile** stack on Ampere. I worked the full PR fallback chain (`fp8_e5m2` → `fp8_e4m3` → `int8`) and each fails for a *different* reason:

| KV dtype | Layer 1: vLLM init guard `attention.py:_init_kv_cache_quant` | Layer 2: attn-forward assert `attention.py:467` | Layer 3: Triton sm_86 compile | Verdict |
|---|---|---|---|---|
| `fp8_e5m2` | **rejects** → `ValueError: fp8_e5m2 kv-cache is not supported with fp8 checkpoints` (relaxed by patch, see below) | **rejects** → `assert kv_cache_dtype in {"fp8","fp8_e4m3","nvfp4"}` → `torch._dynamo.exc.Unsupported: Data-dependent assertion failed` | (would compile: `fp8e5` is supported) | **DEAD @ L2** |
| `fp8_e4m3` / `fp8` | ok | ok | **fails** → `ValueError: type fp8e4nv not supported in this architecture. The supported fp8 dtypes are ('fp8e4b15','fp8e5')` | **DEAD @ L3** |
| `nvfp4` | ok | ok | needs Blackwell fp4 | DEAD (hw) |
| `int8` | — | — | — | **not a valid vLLM `CacheDType`** (only `int8_per_token_head`, a different unsupported path) |

**Root cause in one line:** the two fp8 dtypes the Triton compiler *can* emit on sm_86 (`fp8e4b15`, `fp8e5`) are exactly the two the vLLM attention forward *refuses*; the two it accepts (`fp8_e4m3`, `nvfp4`) need Ada/Hopper/Blackwell. No fp8 dtype clears all three layers.

- **Layer 1 detail:** `CompressedTensorsConfig.get_quant_method` returns a `CompressedTensorsKVCacheMethod` for *every* `Attention` layer (`compressed_tensors.py:181-182`), so `should_load_quant_weights` is True even though this W4A16 checkpoint declares **no** `kv_cache_scheme`. The e5m2 guard therefore mis-fires. I shipped a narrow, correct relaxation (`vllm_fp8kv_e5m2_guard_patch.py`, active only when `kv_cache_scheme is None`) — it works (server log: *"relaxing the fp8_e5m2 KV guard…"*) and gets e5m2 **past L1**, but L2 is a hard kernel-capability wall (the fp8 attention path implements only e4m3/nvfp4 dequant).
- **Layer 3 detail:** `fp8_e4m3` (and bare `fp8`, which aliases to e4m3 per `torch_utils.py:64`) maps to Triton `fp8e4nv`, which Ampere cannot compile — the error surfaces during inductor autotuning of the fused KV-store/RMSNorm kernel.

### Metrics vs baseline

| | bi0 baseline (`int4_mtp_bi0_surgattn`, #770) | fp8kv (this PR) |
|---|---|---|
| output TPS | **218.02** (official a10g-small) | **n/a — fp8 does not boot** (bf16 fallback = 218.02, zero delta) |
| PPL | **2.0058** | n/a (bf16 fallback = 2.0058) |
| MMLU-Pro | 0.644 | n/a |
| completion | 128/128 VALID | n/a |
| W&B | `s63tb03x` | no new run created — no fp8 server reached steady state, so there were no served metrics to log |

The shipped submission (`KV_CACHE_DTYPE=auto`) boots and serves the exact bi0 config (verified locally), so it is a valid — if non-novel — artifact; the fp8 dtypes remain selectable purely to reproduce the wall.

### Peak memory

Not applicable for fp8 — both fp8 crashes occur during **engine init / graph compile, before** the KV cache is allocated, so this is not an OOM and no fp8 steady-state footprint exists. The bf16 fallback footprint equals bi0's.

### Exact commands used (all LOCAL A10G, exploratory)

```bash
# Submission env (from manifest): VLLM_BATCH_INVARIANT=0, NUM_SPECULATIVE_TOKENS=6,
# MAX_NUM_SEQS=1, MAX_MODEL_LEN=4096, GPU_MEMORY_UTILIZATION=0.90, force-2D + attn-group patches.
# Layer-3 wall (fp8_e4m3):
KV_CACHE_DTYPE=fp8_e4m3 python submissions/int4_mtp_bi0_fp8kv/serve.py   # -> Triton fp8e4nv ValueError
# Layer-2 wall (fp8_e5m2, with guard patch active):
KV_CACHE_DTYPE=fp8_e5m2 python submissions/int4_mtp_bi0_fp8kv/serve.py   # -> assert kv_cache_dtype in {fp8,fp8_e4m3,nvfp4}
# Bootable fallback (== bi0):
KV_CACHE_DTYPE=auto     python submissions/int4_mtp_bi0_fp8kv/serve.py   # -> serves bf16 KV, OK
```
Full evidence: `research/validity/bi0_fp8kv/findings.json` + the `_smoke_{e4m3,e5m2,auto}_server.log` boot logs.

### Honesty note on scope

I went one step beyond the literal instructions by relaxing the over-broad L1 e5m2 guard (a real, narrowly-scoped vLLM fix) to give the hypothesis its best chance. That exposed the L2 kernel assert, which is not patchable without rewriting the attention impl. I deliberately did **not** stack further internals patches (e.g. forcing `query_quant=None` to chase e5m2) because (a) the e5m2 dequant kernel is e4m3-shaped so it would likely be silently wrong, (b) e5m2's 2-mantissa-bit noise on a 4B model would risk the PPL≤2.42 gate anyway, and (c) shipping 3–4 vLLM monkeypatches for a leaderboard submission is the wrong trade. Happy to pursue it if you disagree.

### Suggested follow-ups

1. **Close the fp8-KV lever for sm_86** (it is already in BASELINE.md's confirmed-dead-ends; this PR adds the precise three-layer mechanism). A bootable fp8 KV path needs either a new sm_86 fp8 *e5m2-dequant* attention kernel (absent in vLLM 0.22.0) or Ada/Hopper/Blackwell hardware — neither in scope for the A10G target.
2. If KV **bandwidth** is still the lever of interest on Ampere, the only sub-bf16 KV option that the TRITON_ATTN path might accept is `int8_per_token_head` — but it is a different code path with its own greedy-identity and kernel-support questions; would be a separate hypothesis, and I'd expect it to also miss an sm_86 kernel.
3. Optional cleanup: if you want this submission removed rather than kept as a documented dead-end artifact, say so and I'll delete `submissions/int4_mtp_bi0_fp8kv/` in a follow-up.

**Suggested decision: mark this direction CLOSED (negative, hardware-blocked).**
