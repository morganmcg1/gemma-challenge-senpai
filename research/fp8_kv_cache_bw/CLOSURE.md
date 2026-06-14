# fp8 KV-cache BW lever — CLOSURE (PR #141, wirbel)

**Verdict: RED (servability). The fp8 KV-cache lane is closed on the official
`a10g-small` (sm_86) hardware for the deployed int4 `fa2sw_precache_kenyan`
stack.** No token is ever served, so PPL (Step 1) and wall_tps/BW (Step 2) are
not measurable. This banks the KV-read BW lane — the one un-attacked memory
stream after weights are floored at int4.

W&B: `zif6pueq` (group `fp8-kv-cache-bw`, job_type `servability`) —
https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/zif6pueq

## Hypothesis (PR #141)

KV cache is a currently-bf16 BW stream read every decode step. Switching it to
fp8 (`--kv-cache-dtype fp8`) would halve KV-read bytes and cut attention's BW
slice, IF PPL stays ≤ 2.42. Step 0 first checks servability on Ampere.

## What was tested (LOCAL, own A10G seat, deployed submission unchanged)

A minimal env-gated `KV_CACHE_DTYPE` → `--kv-cache-dtype` passthrough was added to
`serve.py` only to drive the two arms, then **reverted** (the deployed submission
is byte-identical to before this PR). Both arms served via
`scripts/local_validation/ppl_runner.py --submission fa2sw_precache_kenyan` with
the full deployed env (spec K=7 MTP, FA_SLIDING, SPLITKV_VERIFY, ONEGRAPH,
lmhead-prune, precache). vLLM `0.22.1rc1.dev307+g3e8afdf78`, model
`/tmp/osoi5-12k-baked` (compressed-tensors int4 W4A16, Marlin).

### Arm A — `--kv-cache-dtype fp8` (e4m3) → HARD FAIL at engine init

```
torch._inductor.exc.InductorError: RuntimeError: Failed to run autotuning code block:
ValueError("type fp8e4nv not supported in this architecture.
            The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')")
```

Crashed inside `vllm/v1/engine/core.py:_initialize_kv_caches` while
torch.compile/inductor autotuned the fused RMSnorm+KV-write kernel whose output
is `*fp8e4nv` at `cc: 86`. **e4m3 (`fp8e4nv`) requires sm_89+ (Ada/Hopper); the
A10G is sm_86 (Ampere).** No software workaround in the harness-fixed
`vllm/vllm-openai` image (FlashInfer is not in the v1 default stack). Confirmed by
literature pass: vLLM Issue #7714 (`fp8e4nv … not supported on CUDA arch < 89`),
PR #14221 (FA V1 + fp8 KV `NotImplementedError`).

### Arm B — `--kv-cache-dtype fp8_e5m2` (e5m2, the only fp8 sm_86 lists) → HARD FAIL at engine init

```
ValueError: fp8_e5m2 kv-cache is not supported with fp8 checkpoints.
```

e5m2 got **further** than e4m3: vLLM accepted the dtype
(`Using fp8_e5m2 data type to store kv cache …`), split-KV verify armed, then the
engine-core init guard rejected it. This is vLLM **Issue #39137**: the guard
fires for *any* `compressed-tensors`-quantized checkpoint (our int4 W4A16), not
only true fp8 checkpoints — a known over-broad misfire.

## Why this closes the lane (not just "two flags failed")

1. **e4m3 is hardware-impossible on a10g-small.** sm_86 has no `fp8e4nv`. This is
   the official scoring hardware, so it is dispositive.
2. **e5m2 is software-blocked** by the quant-compat guard for the int4
   compressed-tensors checkpoint we serve.
3. **Even if Arm B's guard were bypassed, the payoff is negligible and the risk
   is high:** (a) the FA_SLIDING layers force `FlashAttentionBackend`, which in
   vLLM v1 raises `NotImplementedError: FlashAttention V1 with FP8 KV cache not
   yet supported` on Ampere; (b) at single-stream / 512-ctx on A10G GDDR6,
   kernel-launch + dequant overhead swamps the KV-byte saving (vLLM's own fp8-KV
   study finds break-even only at multi-thousand-token contexts, on H100/FA3);
   (c) e5m2 has 2 mantissa bits — coarse — against a PPL gate with only ~1.8%
   headroom (2.3772 vs 2.42). Bypassing a vLLM safety guard to chase a negligible,
   PPL-risky gain is not worth it and was not done.

## Public evidence used

- Leaderboard (digest `as=senpai`, 2026-06-14): the entire spec frontier
  (frantic-penguin 489.63, need-for-speed 488.07, … senpai 481.53) is **bf16-KV +
  split-KV + fa2sw**. No entry ships fp8 KV — consistent with it being
  non-dispatchable on the a10g-small target, not merely unexplored.
- vLLM Issues #7714, #39137; PR #14221 (literature pass).

## Commands

```bash
# Arm A (e4m3): hard fail at engine init
KV_CACHE_DTYPE=fp8      python3 -m scripts.local_validation.ppl_runner \
  --submission submissions/fa2sw_precache_kenyan --out-dir research/_localrun/fp8-kv-arm --port 8000
# Arm B (e5m2): hard fail at engine init
KV_CACHE_DTYPE=fp8_e5m2 python3 -m scripts.local_validation.ppl_runner \
  --submission submissions/fa2sw_precache_kenyan --out-dir research/_localrun/fp8e5m2-kv-arm --port 8000
```

Peak memory: n/a — both crashed during engine-core KV-cache init, before the
model served or the KV cache was allocated.

## Suggested follow-ups (not implemented — out of PR scope)

- **Do NOT re-propose `kv_cache_dtype=fp8` on a10g-small.** It is hardware-closed.
- The KV-read BW stream is **not** attackable via fp8 on this hardware. If KV-BW
  is revisited, the only honest paths are (a) different official hardware (sm_89+
  Ada/Hopper) — not available; or (b) a calibrated fp8 KV checkpoint + a backend
  that consumes fp8 KV on the served layers — a large, low-expected-value change
  given the negligible 512-ctx BW payoff.
- A vLLM-side fix for Issue #39137 (narrow the guard to true fp8 checkpoints)
  would let Arm B *dispatch*, but Arm B is still expected to fail the FA_SLIDING
  `NotImplementedError` and/or the PPL gate; low priority.
