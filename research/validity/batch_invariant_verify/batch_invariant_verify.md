# PR #122 — Batch-invariant verify kernel: can spec==own-AR be restored, and at what TPS cost?

**Verdict: 🔴 RED.** The off-the-shelf batch-invariant override does **not** reduce the
spec-vs-own-AR self-divergence (56.08% → **58.57%**, statistically unchanged) and costs
**51.78% wall_tps** (454.338 → 219.08). An honest greedy-identity GREEN via batch-invariant
verify is **structurally unreachable** on the deployed `fa2sw_precache_kenyan` stack, because
its speed-critical GEMMs are **int4 Marlin** — a custom CUDA op outside the aten dispatcher
that `VLLM_BATCH_INVARIANT` patches, with no batch-invariant path and no split-K knob in the
pinned wheel.

This is the load-bearing follow-up to [#114](../self_referential_gate/self_referential_greedy_gate.md):
#114 proved the deployed spec stack diverges 56.08% from its own M=1 AR (deterministically).
#122 asked whether making the verify reduction M-independent restores 0 divergence. Answer: the
named mechanism (Thinking-Machines / `VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT` class) cannot engage
the divergence source on this stack.

---

## Result (128×512, 2 reloads each arm, identical BASE_ENV)

| metric | #114 deployed | #122 batch-invariant | gate |
|---|---|---|---|
| `batch_invariant_self_divergence_tokens` | 36751 (56.08%) | **38387 (58.57%)** | target 0 → **FAIL** |
| divergent prompts | 112/128 | 113/128 | |
| onset (min/median/max) | —/121/— | 0/100/496 | late/stochastic (FP near-tie) |
| `batch_invariant_wall_tps` | 454.338 (PR #90 ref) | **219.08** | |
| `batch_invariant_tps_cost_pct` | 0 | **51.78%** | <2% GREEN → **FAIL** |
| spec-ON self-determinism | GREEDY_IDENTICAL | GREEDY_IDENTICAL (1.0) | reload-robust |
| spec-OFF AR self-determinism | GREEDY_IDENTICAL | GREEDY_IDENTICAL (1.0) | reload-robust |

Both reloads of each arm are **bit-identical** (spec-ON run_00==run_01 at 38387 tokens; spec-OFF
run_00==run_01) → this is real structural batch-variance, **not** the #38 served-gate wobble.

**Corroboration:** the *same* prompts diverge in both runs — 102 prompts diverge in BOTH
deployed and batch-invariant (Jaccard **0.829**, −10/+11 churn). The override merely reshuffled a
handful of near-tie flips; it did not touch the divergence source.

W&B run: `n5bypf5h` (`kanna/batch-invariant-verify`, group `batch-invariant-verify`).

---

## Mechanism — why the override is a no-op for validity here

`VLLM_BATCH_INVARIANT=1` was **confirmed active** in the run (server logs):
- `init_batch_invariance()` installs persistent Triton matmul over **aten** `mm/addmm/matmul/linear`
  + `softmax/bmm/mean` (`batch_invariant.py:910-931`, SM80 Ampere path; A10G is SM8.6).
- Attention forced to **TRITON_ATTN** for ALL layers ("Gemma4 heterogeneous head dims … Forcing
  TRITON_ATTN"), and `is_batch_invariant` forces `use_3d=False → num_segments=1` (single-segment,
  M-independent) for both M=1 decode and M=8 verify (`triton_unified_attention.py:923-957`).
- `fa_sliding_patch` FLASH_ATTN redirect **never fired** (0 layers) → no FA2 M-dependence.
- `splitkv_verify_patch` redirect **gated OFF** (0 fires) by its `_batch_invariant()` check
  (`splitkv_verify_patch.py:98-99`).
- `FUSED_SPARSE_ARGMAX` is **M-invariant by construction** — `token_idx = program_id(0)` processes
  each row independently and reduces the LM-head over the full hidden dim in one `tl.sum`
  (`sitecustomize.py:645,680`).

So attention, aten matmuls, the SPLITKV redirect, and the fused argmax are **all batch-invariant**.
By elimination, the **only** remaining M-dependent reduction in the verify forward pass is the
int4 weight GEMM:

- Server log: `Using MarlinLinearKernel for CompressedTensorsWNA16`. The QKV / MLP / o_proj
  GEMMs are **int4 Marlin** (`ops.marlin_gemm`), a custom CUDA op called directly — **not** an
  aten op, so `matmul_persistent` never sees it.
- `marlin_utils.py` has **zero** `VLLM_BATCH_INVARIANT` awareness (grep: only fp8/awq/humming do).
- `ops.marlin_gemm(..., size_m=reshaped_x.shape[0], ...)` takes `size_m` (the M/token count) but
  exposes **no `num_splits`/`max_par` knob** — split-K geometry is chosen internally by the
  compiled kernel as a function of `size_m`. M=1 decode and M=8 verify therefore reduce the K axis
  in different float order → low-bit logit deltas → near-tie argmax flips cascade (~0.8%/tok,
  consistent with #114 / #5).
- A10G is SM8.6 + bf16, so `should_use_atomic_add_reduce` returns False (`marlin_utils.py:490`) —
  the reduction is deterministic (hence reload-robust) but still **M-dependent**.
- **No** batch-invariant-aware mixed-precision kernel exists anywhere in the pinned wheel
  (`quantization/kernels/` grep: NONE), and **no** env routes compressed-tensors off Marlin.

**Conclusion:** the override patches the parts of the stack that were *already* cheap to make
invariant (attention, aten matmuls) and leaves the dominant int4 Marlin GEMM — the actual
divergence source — untouched. Net effect: 51.78% wall_tps lost (mostly from forcing TRITON
single-segment 2D attention, giving up the 4.14×-faster 3D split-KV decode path) for **zero**
divergence reduction.

---

## Where the TPS cost comes from

454.338 → 219.08 (−51.78%). The dominant cost is attention: `VLLM_BATCH_INVARIANT` forces
`use_3d=False` (num_segments=1) for the **M=1 decode** path too, surrendering the deployed 3D
split-KV decode (`splitkv_verify_patch.py` docstring: 3D is 4.14× the 2D path at M=1). Plus
matmul_persistent on the residual aten matmuls and TF32-off. None of this buys any validity,
because the int4 GEMM stays M-variant.

---

## Gate mapping (PR #122)

- 🟢 GREEN (div→0, cost <2%): **not met.**
- 🟡 AMBER (div→0, cost ≥2%): **not met** — divergence did not reach 0 (or move at all).
- 🔴 **RED** (batch-invariance cannot reach 0): **met, and stronger than imagined.** The PR's RED
  branch assumed "residual near-tie flips survive a *working* batch-invariant kernel." Reality:
  the off-the-shelf override **never engages** the divergence source (int4 Marlin), so divergence
  is unchanged. Reaching 0 would require a *new* fixed-split-K int4 Marlin/Machete CUDA kernel
  (not in the pinned wheel) or dequantizing to bf16 aten GEMMs (catastrophic TPS, defeats the
  submission) — both far beyond a local override probe, and the attention+aten changes alone
  already cost 51.78%.

**Implication for the spec-decode 500 programme:** an honest greedy-identity green-light with the
drafter ON is **validity-blocked** on this stack. The decision now rests on the open human
contract ruling — does program.md 27–28 bind, given the official `speed_benchmark` harness runs
no token-identity check (#114)? A cheap GREEN would have made that question moot; this RED makes
the escalation load-bearing.

## PPL

Not separately re-measured. PPL is teacher-forced (greedy argmax flips never enter it — #114), and
the override perturbs only low bits, so PPL ≈ 2.378 first-order (deployed 2.3777, cap 2.42). PPL is
not the binding constraint and cannot change a verdict that is RED on divergence **and** TPS.

## Reproduce

```bash
cd target/
python scripts/validity/greedy_identity_interlock.py --self-referential \
  --config batch_invariant --runs 2 --num-prompts 128 --output-len 512 \
  --submission submissions/fa2sw_precache_kenyan \
  --out-root research/validity/batch_invariant_verify/<ts> \
  --tps-ref 454.338 --wandb-group batch-invariant-verify \
  --wandb-name kanna/batch-invariant-verify
```
