# fa2sw_strict_byteexact_splitkv399 — staged strict draw-ready submission (penalize lane)

**Lever:** `BYTEEXACT_FIXED_TPS=4` + `BYTEEXACT_NUM_SEGMENTS=64` pin vLLM's 3D split-KV
(FlashDecoding) `tiles_per_segment` to a **fixed constant** (via `byteexact_splitkv_patch.py`).
The deployed fast path computes `tiles_per_segment = cdiv(seq_len, num_segments * TILE_SIZE)`
— **adaptive** in `seq_len`, so segment cuts move as the context grows and the M=8 spec-verify
can reduce keys in a different order than the M=1 AR decode of the same token (lawine #496
microbench: 6/8 byte-flips on straddle positions). Pinning `tiles_per_segment = 4` makes every
segment cover a **fixed absolute key span** `[s·64, (s+1)·64)` regardless of `seq_len`
(T=4 tiles × TILE_SIZE=16 = 64 keys/segment; 64 segments cover 4096 = `max_model_len`). The
reduction order at any absolute key position is then **M-invariant** ⇒ the M=8 verify is
byte-identical to M=1 AR (Thinking-Machines "fix the split **SIZE**, not the split **COUNT**").
It composes with `splitkv-verify` (which routes the M=8 verify onto this same 3D path) and does
**NOT** set `VLLM_BATCH_INVARIANT=1`, so `init_batch_invariance()` never installs the ~48% matmul
tax — MLP/QKV/lm_head keep the fast Marlin path. **Spec-alive**: `SPECULATIVE_CONFIG` (MTP
drafter, K_spec 7) is kept.

**Why it beats surgical-357:** the surgical 2D path is byte-exact but gives up all split-KV
parallelism (357.6 TPS). The fixed-order 3D split-KV keeps the parallelism *and* the byte-exact
identity, recovering ~92% of the deployed-minus-surgical gap.

**Provenance:** lawine PR #496 (`42qroec1`) measured the lever at **399.75 local TPS**
(+42.1 over the surgical-357 byte-exact rung, ≈8.7× σ_hw 4.864), **PPL 2.3767**, **128/128**,
with **0/8 kernel flips** where the adaptive path flips 6/8 (microbench M-invariance proof at
straddle positions 251–258 / 507–514).

**Byte-faithfulness:** byte-identical to the deployed `fa2sw_precache_kenyan` serve stack
except (a) this README, (b) the new default-off `byteexact_splitkv_patch.py`, (c) one gated
import in `sitecustomize.py`, and (d) manifest `name` / `description` + the
`BYTEEXACT_FIXED_TPS=4` / `BYTEEXACT_NUM_SEGMENTS=64` env. With `BYTEEXACT_FIXED_TPS` unset/0
the patch module is not even imported and the served compute path is byte-identical to the
parent. Armed on a **stock serve-venv wheel** by re-jitting `kernel_unified_attention` +
`reduce_segments` (no installed-wheel file edit) — the same `sitecustomize.py` → monkeypatch
mechanism stark used for surgical-357.

**Local operative-1.0 certification (PR #500):** see the PR results comment for the served-run
self-determinism (r1-vs-r2) + margin census and the fire-time recert dry-run (TPS within σ_hw of
399.75, PPL ≤ 2.42, 128/128, `[byteexact] fixed split-KV armed` + ONEGRAPH log signatures).
`analysis_only=true`, `official_tps=0`, no HF job.

**Risk note (denken #489):** spec-alive ⇒ ~24% private-TPS-drift breach ⇒ **NOT private-safe**.
This is staged as the **penalize-lane** option (where it dominates both the 222 ship and the
surgical-357 rung at the same byte-equivalence standard); the floor-lock `fa2sw_strict_m1ar_int4`
(161.70) remains the only private-safe ship. **The DRAW decision is the human's call** — this dir
only makes 399.75 fire-ready the instant penalize-lane is ruled.
