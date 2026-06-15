STUDENT land:
SENPAI-RESULT: {"terminal":false,"status":"in_progress","pending_arms":true,"wandb_run_ids":[],"primary_metric":{"name":"treeverify_faithful_tree_anchor_masked","value":0.8318},"test_metric":{"name":"treeverify_faithful_tree_vs_linear_delta_masked","value":-0.0021}}

## Results — Cycle 3: faithful-tree confirmation (GO)

I ported the cycle-2 real-KV KV-location fix into the **full M=16 tree** verify path and re-measured the anchor under the representative live-build conditions. **The confirmation passes — and lands stronger than the ~0.77 prediction.** All LOCAL (1×A10G, `CUDA_VISIBLE_DEVICES=0`); no HF Job, no submission change, no edit to `fa2sw_precache_kenyan`. Committed path stayed linear K=7 (probes are observational: real KV slots snapshot/restored, all writes at offsets ≥ root_position). **This is NOT a launch.**

### TL;DR — the tree is faithful; there is no tree penalty
Under matched correct masks, the **faithful M=16 tree verify reproduces the deployed linear-greedy argmax at 0.832 — within −0.002 of the M=8 linear-real ceiling (0.834)**. The "−0.06 tree penalty" from cycle 2 was **entirely plumbing + a missing tree-causal mask**, both of which the LIVE build has correct. Decisive STOP condition (tree verify itself diverges) is **NOT met**.

### The full decomposition (1100 steps, 7,644 rows each, `forward_err=0`)
| anchor (M=16 tree spine vs deployed verifier) | row-match | confident-wrong (gap≥1.0) |
|---|---|---|
| scratch M=16 tree, unmasked (cycle-2 plumbing) | **0.5393** | — |
| ↳ + real-KV KV-location fix → faithful, unmasked | **0.7442** | 0.441 |
| ↳ + tree-causal qq_bias → faithful, **MASKED (live-build verify)** | **0.8318** | 0.380 |
| M=8 linear-real (isolation ceiling, same machinery) | **0.8339** | 0.387 |

- KV-location fix recovers **+0.205** (0.539→0.744); the tree-causal mask recovers a further **+0.088** (0.744→0.832).
- **Masked tree-vs-linear delta = −0.0021 ≈ 0.** (Cycle-2 reproduced exactly: scratch-tree 0.5393 vs your 0.539; linear-real 0.8339 vs your 0.834.)
- The masked tree's confident-wrong fraction (**0.380**) matches the linear ceiling (**0.387**) — the residual ~0.166 miss has the *same* near-tie/confident structure as the linear control, i.e. it's the **same isolation-ceiling effect** (int4-Marlin batch variance + hand-built metadata), **not** a tree-specific divergence.

### Per-depth anchor breakdown (the requested format)
Masked faithful tree vs linear-real, per spine depth:

| depth | unmasked tree | **masked tree** | linear-real | masked − linear |
|---|---|---|---|---|
| 0 | 0.6163 | 0.6163 | 0.6163 | 0.0000 |
| 1 | 0.8260 | 0.8260 | 0.8260 | 0.0000 |
| 2 | 0.7747 | 0.8361 | 0.8379 | −0.0018 |
| 3 | 0.7390 | 0.8681 | 0.8700 | −0.0019 |
| 4 | 0.7143 | 0.8828 | 0.8874 | −0.0046 |
| 5 | 0.7335 | 0.8956 | 0.8974 | −0.0018 |
| 6 | 0.8059 | 0.8974 | 0.9020 | −0.0046 |

The shape is the smoking gun. The **unmasked** tree is *identical* to linear at depth 0–1 (no branch nodes precede those spine positions), then loses ground exactly as interleaved branch nodes accumulate — peaking at **depth 4 (−0.173)**, where spine index 9 wrongly attends 5 branch nodes (2,4,5,7,8) under pure causal. The **tree-causal mask masks out precisely those non-ancestor branches**, and the deficit collapses to −0.002…−0.005 at *every* depth. That is branch contamination removed, not a fundamental divergence repaired.

### Honest read on the ~0.77 prediction
Your ~0.77 = linear-real 0.834 − the cycle-2 **unmasked** delta 0.06. Under matched (unmasked) conditions the faithful tree is **0.744** — i.e. the unmasked delta *widened* to −0.090, the pre-capture warning you flagged. **But that unmasked delta is an artifact:** the M=8 linear is already correctly causal (degenerate chain ⇒ lower-triangular mask is a no-op), so it has no branches to contaminate it, while the unmasked M=16 tree is penalized for branch KV it shouldn't see. The apples-to-apples comparison — **each leg under its own correct mask** — is masked-tree 0.832 vs linear 0.834 (**−0.002**). The live build uses the tree-causal mask, so **the masked number is the representative one**, and it says the tree carries zero excess penalty.

### Fidelity-gate reconciliation (≥0.95 at M=8)
The masked tree sits at the **isolation ceiling (0.832 ≈ 0.834), not ≥0.95** — as you noted, the ~1.0 absolute is unreachable by *any* separate scratch probe (the linear control itself caps at 0.834 under int4-Marlin batch variance + hand-built metadata). The ≥0.95 absolute gate belongs to the **live integrated path** (native verify metadata, no scratch reconstruction), which the capture step produces. The transferable go/no-go — tree-vs-linear delta under correct masks = **−0.002** — says the tree is as faithful as linear, so when the live path runs linear at its true ~1.0, the tree should follow.

### Command (both runs identical except the masked one adds the 3 `TREE_QQ_BIAS_*` envs)
```
CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 SPLITKV_VERIFY=1 \
TREE_EMIT_PROBE=1 TREE_EMIT_PROBE_M=16 TREE_SALVAGE_PROBE=1 \
TREE_VERIFY_PROBE=m16 TREE_VERIFY_REAL_KV=1 TREE_VERIFY_REAL_KV_TREE=1 \
[masked only:] TREE_QQ_BIAS_PROBE=1 TREE_QQ_BIAS_M=16 TREE_QQ_BIAS_PARENT=m16 \
TREE_{SALVAGE,VERIFY}_PROBE_VERDICT=<out>/{salvage,verify}_verdict.json \
/tmp/server-venv/bin/python scripts/local_prevalidate.py \
  --submission submissions/fa2sw_treeverify_kenyan --no-ppl \
  --decode-num-prompts 8 --decode-output-len 512 --output-dir <out>
```
- qq_bias dispatch confirmed: `[tree-qq-bias] DISPATCHED [16x16] fp32 parent=m16` ×1683 (1092 verify steps masked).
- Backend: `AttentionBackendEnum.TRITON_ATTN` (matches lawine #246's lock-in).
- **Peak VRAM:** KV pool 9.46 GiB / 376,880 tokens + ~8.47 GiB int4 weights; probe reuses the existing pool (a few top-of-pool scratch blocks, snapshot/restored) + one extra M=16 forward (negligible) — well under 24 GB.
- **Token contract:** 8/8 completed, 4096/4096 tokens. PPL unmeasured (`--no-ppl`); probes are observational so PPL holds at 2.3767. Two tracebacks per log are benign vLLM infra noise (the `_report_usage_worker` cpuinfo-JSON thread at startup; a post-decode shutdown-race `EngineDeadError`) — neither touches inference (`forward_err=0`, full token count produced).
- Artifacts: `research/local_validation/treeverify_realkvtree_{unmasked,masked}_20260615/` (verify_verdict.json + server.log + node-argmax dumps).

### What happened
The faithful-tree forward reads the real prefix block-table (committed KV incl. the partial seam block, no redirect+copy), allocates scratch slots **only** for the M new node rows that overrun the deployed M=8 verify's allocation, carries depth-RoPE + node-index KV slots, and (masked variant) receives the [16×16] tree-causal qq_bias from the existing env-gated splitkv wrapper. It works first try, `forward_err=0` over 1100 steps. The two-leg design isolates every confound: KV-location (+0.205), tree-causal mask (+0.088), residual tree-vs-linear (−0.002).

### Recommendation: GO — capture the M=16 verify at size-16 (long-pole-1)
Surfacing this for your read at the declared pivot point before I commit multi-cycle capture effort — you flagged path-A as knife-edge after #257, so I'd rather you greenlight the long-pole (or redirect to the 1-leaf fallback) than have me silently dive in. Confirmation is positive and stronger than the bar. Reading in the new constraints you relayed:
- **#257 step band [1.12, 1.43] ms / E[T] floor moved up** — noted; the build wins on accepted-tokens, and I'll measure the live step against [1.12, 1.43] (not the retired 1.085).
- **Build shallow (fern #262 spine + ρ₂ leaf-salvage, E[T]≈4.30)** — the per-depth data supports this: faithful fidelity is healthy through depth 6 (≥0.88 masked), so a shallow spine + leaf salvage is well within the faithful regime.
- **TRITON_ATTN + CUDAGraph ON** — my probe already runs TRITON_ATTN; capture will keep CUDAGraph ON (the validated 481.53/PPL-2.3772 frontier is the capture-ON config).

### Suggested follow-ups
1. **Capture path (long-pole-1):** widen the verify CUDA-graph capture past the size-29 crash to M=16 at size-16 (lawine owns the isolated size-29 repro — treat as input, don't fork). Success = a captured M=16 verify step + first MEASURED `treeverify_served_gain_MEASURED_realized > 0`.
2. **Fallback in pocket:** vidraft-darwin's 1-depth leaf-sibling tree (E[L] 3.136→3.653, ~+50 TPS, no continuation/megakernel) if the full-tree capture stalls — the faithful-fidelity data says even a 1-leaf tree clears the mask cleanly.
3. **Branch-interior fidelity:** this cycle measured spine nodes; the node-argmax dumps let me extend the masked faithful-tree anchor to branch-interior (rank-2 salvage) edges on request, to confirm ρ₂ leaf-salvage commits are faithful before they enter E[T].
