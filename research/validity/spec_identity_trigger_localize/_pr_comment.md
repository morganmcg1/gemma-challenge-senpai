STUDENT stark:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["4jv01n87"],"primary_metric":{"name":"official_tps","value":0},"test_metric":{"name":"spec_verify_vs_ar_maxdiff_under_BI","value":0.0}}

## Results — Localize the real spec-#319 trigger (non-Marlin op probe, vLLM 0.22.0, single A10G sm_86, `analysis_only`, NO build / NO checkpoint / NO HF Job)

**Verdict: `SPEC_TRIGGER_RECOVERABLE__attention_3D_splitKV__VLLM_BATCH_INVARIANT`**

> Under the submission's served config (`VLLM_BATCH_INVARIANT=1`, `MAX_NUM_SEQS=1`, TRITON_ATTN, int4-Marlin), **every op in the per-token forward is byte-identical at the spec-verify width (M=8) vs the plain-AR width (M=1)** — int4-Marlin (body **and lm_head**), TRITON_ATTN attention (all 8 verify rows, global + sliding window), `rms_norm`/`fused_add_rms_norm`/`rotary_embedding`, and the greedy argmax. The **only** batch-variant op found anywhere is the attention **3D split-KV** path, which the M=1 AR decode takes **only when BI is OFF**, and which `VLLM_BATCH_INVARIANT=1` (already set) closes. So there is **no intrinsic spec-verify trigger** to recover with a new kernel; the recoverable knob is already pinned. A residual #607-style break at BI=1 therefore points at the **reference/comparison config**, not the served numeric path. **Surface — do NOT auto-fire (knob already on); recommend pinning the #319 reference within-stack.**

This **extends and strengthens** #617 (`fa1f9vm1`): #617 proved BODY Marlin is bit-exact at the verify width but left the **lm_head Marlin untested**. The #607 build (`int4_g128_lmhead/config.json`) quantizes the head to int4 g128 (`group_1 targets=['re:.*lm_head']`, `tie_word_embeddings=False`, K=2560→**N=262144**) — a custom `_C` op outside BI and the prime remaining suspect. It is **also bit-exact at M=7/8** (first divergence M=32). Marlin is fully exonerated at the deployed width, head included.

### Probe 1 — int4-Marlin row-0 M-sweep (#617 harness + the lm_head gap)
Faithful int4 g128-symmetric synthetic weights (vLLM's `marlin_quantize`), served kernel defaults (`use_fp32_reduce=True`, atomic-add hard-False on sm_86). Compare output **row 0** across M to the M=1 result.

| shape | K | N | m7 bit-exact | m8 bit-exact | first divergence |
|---|---|---|---|---|---|
| qkv | 2560 | 3072 | ✅ | ✅ | none (≤128) |
| o_proj | 2048 | 2560 | ✅ | ✅ | none (≤128) |
| gate_up | 2560 | 20480 | ✅ | ✅ | M=32 (1 bf16-ULP) |
| down | 10240 | 2560 | ✅ | ✅ | M=128 (1 bf16-ULP) |
| **lm_head** | **2560** | **262144** | ✅ | ✅ | **M=32 (1 bf16-ULP)** |

Body rows reproduce #617 exactly. **The lm_head, despite N=262144, is M-invariant at the verify width** and only diverges at M≥32 by a single bf16-ULP — not a verify-width trigger.

### Probe 2 — TRITON_ATTN `unified_attention`, verify(2D) vs AR(2D/3D), all rows (PR instruction 1)
Source (`triton_unified_attention.py:923-932`): the verify forward (`max_seqlen_q=8>1`) is **always 2D**; the M=1 AR decode (`num_seqs=1 ≤ seq_threshold_3D=64`) takes the **3D split-KV** path **unless `is_batch_invariant`**. So BI=1 forces both to 2D. Single-sequence synthetic paged KV (prefix 2048); compare **every** verify row j against its own M=1 AR forward at the matching absolute position; both global and sliding-512 windows.

| comparison | window | max row-0..7 maxdiff |
|---|---|---|
| verify(2D) vs per-row AR(2D, **BI=1**) | global | **0.0 (all 8 rows bit-exact)** |
| verify(2D) vs per-row AR(2D, **BI=1**) | sliding-512 | **0.0 (all 8 rows bit-exact)** |
| verify(2D) vs AR row-0(**3D, BI=0**) | global | 3.05e-5 (1 bf16-ULP) |
| verify(2D) vs AR row-0(**3D, BI=0**) | sliding-512 | 6.10e-5 (1 bf16-ULP) |

**The BI attention guard is taken and is load-bearing.** With BI on (the submission), verify == AR byte-exact for the whole block. With BI off, the 3D split-KV AR decode diverges from the 2D verify by ~1 ULP — exactly the kind of pervasive low-amplitude perturbation that flips near-tie greedy argmaxes across many "stable" positions.

### Probe 3 — greedy sampler / argmax + `final_logit_softcapping` (PR instruction 2b)
`argmax(softcap(logits))` is per-row; row-0 argmax within an M=8 batch == argmax at M=1 (247681 == 247681). Row-independent. The only batch-variance entry point upstream of argmax is the lm_head GEMM (Probe 1 — clean at M=8).

### Probe 4 — custom `_C` norm/rope ops **outside BI** (completes the forward)
`rms_norm`, `fused_add_rms_norm`, `rotary_embedding` are `torch.ops._C` ops BI does not patch. All **bit-exact M=8 vs M=1 row 0** (per-row reduction / per-position rotation; `fused_add_rms_norm` is even source-annotated "batch invariant"). No hidden M-dependence survives BI here.

### MTP drafter forward (PR instruction 2a) — logically cannot be a greedy-identity trigger
Under greedy (temp=0), identity is decided by the **target** argmax accepting/rejecting drafted tokens. A different draft only changes the **acceptance rate (speed)**, never the emitted token: a rejected draft falls back to the target's own argmax = the AR token. So the drafter is out of the identity path by construction (not swept).

### Why this verdict (the rigorous chain)
1. Every op in the position-0 forward is byte-identical at M=8 vs M=1 under BI=1 (Probes 1-4).
2. Induction over positions: greedy ⇒ an accepted draft `d_i` equals the AR argmax `a_i` ⇒ identical conditioning ⇒ identical next-position logits ⇒ identical argmax. So the **spec-verify output is byte-identical to plain-AR within-stack**.
3. The only batch-variance anywhere (attention 2D-verify vs 3D-AR, ~1 ULP) is fully gated by `VLLM_BATCH_INVARIANT` and is **off in the submission**.

Therefore a 47% #607 break measured at BI=1 cannot originate in the served per-position numerics. The parsimonious reading consistent with all measured data: the #319 greedy-identity **reference was not pinned to the served stack** — most likely BI not set identically on both sides (reference AR on the 3D path → the ~1-ULP perturbation above), or a batch width above the served `MAX_NUM_SEQS=1` pushing the effective Marlin M past the M=32 divergence threshold. Both are **config/serving mismatches (recoverable)**, not a kernel defect. (I cannot inspect #607's gate methodology directly, so I surface this as the inference my data supports, with the alternative — a structural spec-decode control-flow issue — noted but made unlikely by the op-level + inductive evidence.)

### What this means for action
- **Do NOT build a batch-invariant Marlin or attention kernel.** There is no verify-width M-dependence in the served path to fix (#319-class risk for zero identity gain).
- **Recommended knob (already partially in place): pin the #319 greedy-identity reference within-stack** — same `VLLM_BATCH_INVARIANT=1` AND same `MAX_NUM_SEQS=1` on both the spec-on and reference sides. The attention guard must be effective on the reference too (it forces the 2D path). Surface only.

### Reproduction
```bash
cd target/
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  research/validity/spec_identity_trigger_localize/spec_trigger_localize.py --probe all
# subcommands: --probe {lmhead,attention,sampler,normrope,all}
```
- **W&B run:** `4jv01n87` (group `spec-identity-trigger-localize`, `analysis_only=true`, `official_tps=0`); per-shape M-sweep table logged as `marlin_m_sweep_with_lmhead`.
- **Peak GPU memory:** ~10.0 GiB allocated / 12.6 GiB reserved incl. CUDA context (dominated by the synthetic N=262144 lm_head Marlin build; **no checkpoint loaded, no kernel build** — disk-neutral).
- vLLM 0.22.0 (`.venv`), single A10G sm_86.

### Public evidence used
- **wirbel #607** (`yuvztndu`): the 31048/65536 (47%) spec-#319 break on the clean 0.22.0 gate — the motivating measurement this card localizes.
- **stark #617** (`fa1f9vm1`): BODY Marlin bit-exact at verify width; this card closes the lm_head gap #617 left and confirms attention/norms/sampler.

### Suggested follow-ups
- **Re-run the #607 gate with the reference pinned within-stack** (`VLLM_BATCH_INVARIANT=1`, `MAX_NUM_SEQS=1` on both sides) and confirm the break collapses toward the 0/N floor. If it does, the int4+MTP spec stack is greedy-identity-safe as served.
- If a residual break survives a within-stack BI=1 reference, the next suspect is **structural spec-decode bookkeeping** (bonus-token / accepted-length / position threading), not op numerics — probe with a token-level spec-on-vs-spec-off trace at `MAX_NUM_SEQS=1`.
- Optional belt-and-suspenders: repeat Probe 1 on the real baked `int4_g128_lmhead` weights (expected identical — M-invariance is schedule-, not data-dependent).
