# fa2sw_strict_surgical357 — staged strict draw-ready submission (penalize lane)

**Lever:** `SURGICAL_ATTN_USE_3D_OFF=1` forces the 7 full-attention reductions onto vLLM's
byte-exact **2D order-preserving sequential-KV** path (sets
`triton_unified_attention.is_batch_invariant=True` via `surgical_attn_patch.py`; this also
short-circuits the `splitkv-verify` 3D redirect, so the M=8 spec-verify takes the same 2D
path). It does **NOT** set `VLLM_BATCH_INVARIANT=1`, so `init_batch_invariance()` never
installs the ~48% matmul tax — MLP/QKV/lm_head keep the fast Marlin path. **Spec-alive**:
`SPECULATIVE_CONFIG` (MTP drafter, K_spec 7) is kept, which is what makes this 357 rather
than the floor-lock 161.

**Provenance:** lawine PR #488 (`ko01dcyy`) measured the lever at **357.64 local TPS**
(+135.7 over the 222 global-flag floor, 28× σ_hw), **PPL 2.3767**, **128/128**. The −135 TPS
the global flag costs is the identity-unnecessary matmul tax (#488: 9/128 residual flips vs
the 222 config, all bf16-ULP near-ties per merged #461, 0 semantic).

**Byte-faithfulness:** byte-identical to the deployed `fa2sw_precache_kenyan` serve stack
except (a) this README, (b) the new default-off `surgical_attn_patch.py`, (c) one gated
import in `sitecustomize.py`, and (d) manifest `name` / `description` + the
`SURGICAL_ATTN_USE_3D_OFF=1` env. With the env unset the served compute path is
byte-identical to the parent.

**Local operative-1.0 certification (PR #494, stark — served run `k8nqmc2b`, margin census `5fxw18gu`):** CERTIFIED operative-1.0.
- `surgical357_operative_identity`: **operative-1.0 ✓** — (1) served-vs-served matched-config
  self-determinism r1-vs-r2 = **1.000** (65536/65536 tokens, 0/128 sequences flipped, warm); (2)
  #461-style logit-margin census (127 prompts × C=224 × M=8 verify): surgical (`attn_only`) identity
  **0.998875** (1 residual flip), and that flip is a bf16-ULP knife-edge near-tie — **margin 0.125 nat
  = exactly one bf16 logit step**, < the 0.5 near-tie threshold ⇒ **0 semantic flips**. The surgical
  `attn_only` divergence (0.00112486) **== the 222 `all_pin` divergence** (identical to 15 sig figs) ⇒
  the lever is operatively **identical** to the shipped 222, dropping only the matmul tax (lm_head /
  GEMM-Marlin / RMSNorm each contribute 0 flips; `flip_attr_self_test_passes=True`).
- `surgical357_pod_tps_sanity`: **357.43 TPS** (warm median of 357.46/357.40, **+135.43 over the 222
  floor**), **PPL 2.37698** (≤ 2.42 gate), **128/128** completion — matches lawine #488's 357.64 /
  2.3767. Lever fired in the packaged stack on a **stock serve-venv wheel** (`[surgical-attn] armed` +
  `forced is_batch_invariant=True`, `splitkv_redirects=0`, ONEGRAPH captured, `init_batch_invariance`
  never ran ⇒ matmul tax off), proving the gate is the packaged `surgical_attn_patch.py`, not a venv edit.

**Risk note (denken #489):** spec-alive ⇒ ~24% private-TPS-drift breach ⇒ **NOT private-safe**.
This is staged as the **penalize-lane** option (where it dominates the 222 ship by +135 at the
same byte-equivalence standard); the floor-lock `fa2sw_strict_m1ar_int4` (161.70) remains the
only private-safe ship. **The DRAW decision is the human's #474 call** — this dir only makes
357.6 fire-ready the instant penalize-lane is ruled. `analysis_only=true`, `official_tps=0`,
no HF job.
