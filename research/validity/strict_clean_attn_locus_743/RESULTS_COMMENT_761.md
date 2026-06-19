STUDENT lawine:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["2z8o5fnv"],"primary_metric":{"name":"top_op_divergence_share","value":1.0},"test_metric":{"name":"literal_strict_achievable_targeted","value":1}}

## Results — served-divergence-locus (first-divergence census)

**Verdict: the residual served-spec divergence is single-source. 100% of first-divergences localize to the verify-attention split-KV reduction (`attn_out`). It is NOT the matmul family BI=1 already patches, and it is closable with a current toggle.**

| metric | value | meaning |
|---|---|---|
| **`top_op_divergence_share` (primary)** | **1.000** | 55/55 divergent positions first-diverge at one op family (attention reduction) |
| **`literal_strict_achievable_targeted` (test)** | **1** | forcing batch-invariance on that op (BI=1, a current toggle) drives in-process per-op divergence to literal 0 |
| top op family | `attn` (`attn_out`) | verify-attention split-KV reduction |
| Marlin GEMMs incl. full-vocab lm_head | M-invariant, BOTH arms | controlled microbench, 0 divergent → refutes the matmul-locus guess |

### Two-arm design (isolates the locus)
- **bi0 arm — divergence-OPEN** (`VLLM_BATCH_INVARIANT=0`): M=1 decode takes the 3D split-KV attention path (`num_splits>1`); M=K+1 verify takes the 2D path (`num_splits=1`). The reduction-order split is left open — this is where the locus is measured + attributed.
- **bi1 arm — the targeted FIX** (`VLLM_BATCH_INVARIANT=1`): `is_batch_invariant=True` freezes `use_3d=False` for BOTH M → `num_splits=1` everywhere → the 3D-decode-vs-2D-verify split collapses to one kernel path.

### bi0 (divergence-open) — locus + op-family attribution
- **2304 positions** (48 prompts × 48 new tokens), strict reference = spec-off in-process M=1 greedy-AR decode of the deployed `int4_g128_lmhead` ckpt (the #755 strict reference); divergence = `M=K+1 verify activation != M=1 decode activation` in ULP (`torch.equal`).
- **55 positions** show intermediate-activation divergence; **first-divergence-by-op = `{attn_out: 55}` → 100% attention**. Ranked breakdown: `[("attn_out", 55)]` (no matmul / norm / softmax / KV-gather / sampling op is ever the first diverging op).
- **10 e2e argmax flips** (rate 0.43%), **all 10 upstream of the head, 0 head-only** — consistent with the flip being seeded in the attention reduction and carried forward, not minted at the lm_head.
- **Controlled Phase-2 microbench:** the int4 Marlin GEMMs — qkv, o_proj, mlp gate/up, mlp down, **and the full-vocab (262144) int4 Marlin lm_head** — are all M-invariant (`n_divergent=0`, `max_abs=0.0`) at the census positions. The atomic-add toggle on the head is inert. **→ the matmul family is NOT the locus.** This refutes my earlier #755 "un-BI Marlin matmul is the residual flip source" hypothesis.

### bi1 (the fix) — does the targeted toggle close it?
- **0 / 2304 positions divergent, 0 e2e argmax flips → byte-exact in-process.** Forcing batch-invariance on the single top op family closes the per-op divergence to literal 0 in this harness. `literal_strict_achievable_targeted = 1` (realizable with a current toggle; no vLLM patch required for the per-op fix).

### Honesty / scope caveats (important)
1. **In-process, `enforce_eager` proxy** — not the live CUDA-graph served path. The eager harness reproduces the reduction-order split exactly, but cannot reproduce CUDA-graph decode geometry.
2. **Reconciliation with #755 (the key nuance):** #755's served arm clamped `num_splits=1` and still saw 104/128 divergent. Clamping `num_splits` alone (an `nseg`-clamp) does **not** route M=1 decode off the 3D kernel. Proper BI=1 instead flips `is_batch_invariant=True` → `use_3d=False` for both M, which is why this arm is byte-exact while #755's nominal-num_splits arm was not. The served residual surviving nominal-BI in #755 most plausibly reflects **CUDA-graph decode geometry / spec-decode acceptance**, not a second per-op locus — the eager proxy shows no second op family contributes.
3. **The TPS cost** of forcing BI=1 on the attention reduction is the orthogonal per-kernel-cost axis (fern #750) — out of scope for this locus census.
4. **Publishable-rung framing unchanged:** the shipped rung is self-consistent-gate + PPL-clean (G1-immune), **not** literal byte-exact, and non-byte-exactness is **not** a DQ (organizer scorer is identity-blind). This census answers the narrow question "*where* would literal-strict break, and is that locus targetable" — answer: one op family (attention reduction), targetable with a current toggle.

### Hypothesis outcome
**CONFIRMED, at the ceiling.** The PR predicted "the divergent positions are dominated by one op family (most likely the verify-attention reduction… not the matmul family BI=1 already patches), such that the single top op accounts for ≥ 50%." Measured share = **100% > 50%**, and the dominant family is exactly the predicted attention reduction. The "multi-source / smeared → irreducibly non-strict" alternative is refuted: the fast rung's literal-strict gap is single-source and (per-op) closable.

### Reproduce
```bash
# bi0 (divergence-open locus arm) and bi1 (BI=1 fix arm), served venv, A10G, no HF job
SP=/tmp/senpai-venvs/20f658587e8a6643/bin/python
DIR=research/validity/strict_clean_attn_locus_743/runs/locus_census
CUDA_VISIBLE_DEVICES=0 "$SP" research/validity/strict_clean_attn_locus_743/served_locus_census.py \
  --n-prompts 48 --n-new 48 --ctx-cap 128 --topk 8 --bi 0 --out "$DIR/bi0_report.json"
CUDA_VISIBLE_DEVICES=0 "$SP" research/validity/strict_clean_attn_locus_743/served_locus_census.py \
  --n-prompts 48 --n-new 48 --ctx-cap 128 --topk 8 --bi 1 --out "$DIR/bi1_report.json"
# cross-arm verdict + wandb (repo .venv has wandb; no GPU)
.venv/bin/python research/validity/strict_clean_attn_locus_743/bi_arm_orchestrator.py \
  --bi0 "$DIR/bi0_report.json" --bi1 "$DIR/bi1_report.json"
```

### Run facts
- **W&B:** run_id `2z8o5fnv`, group `fire_bi_tax_750`, name `lawine/served-divergence-locus` (project `senpai`).
- **Peak mem:** 18.49 GB (bi0) / 18.51 GB (bi1), single A10G. **Elapsed:** 414.5 s (bi0) + 435.8 s (bi1).
- `analysis_only=true, official_tps=0, no_hf_job=1, fires=0`. LOCAL ONLY — no HF Job, no served-file change, no submission, no `--launch`.
- Config: K=4 (verify width 5), TRITON_ATTN, `enforce_eager`, 42 layers, deployed `int4_g128_lmhead` (full-vocab int4 Marlin lm_head).

### Suggested follow-ups
1. **Served CUDA-graph confirmation (the open item):** run the same first-divergence trace against the live served path (CUDA graphs on, not `enforce_eager`) to test whether the #755 served residual is fully explained by CUDA-graph decode geometry + spec-decode acceptance once the attention reduction is pinned — the one thing the eager proxy cannot settle.
2. **Hand to fern #750:** price the TPS cost of `VLLM_BATCH_INVARIANT=1` on the attention reduction so the literal-strict rung can be cost-compared against the shipped G1-immune rung.
3. **Layer-resolved attention attribution:** all 55 first-diverge at layer 0 `attn_out` in this run; a per-layer histogram across more prompts would confirm whether the sliding-vs-full attention split (35 sliding / 7 full layers) concentrates the reduction-order sensitivity.
