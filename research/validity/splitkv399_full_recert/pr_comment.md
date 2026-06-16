STUDENT stark:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["kwhylaeg"],"primary_metric":{"name":"splitkv399_full_warm_median_tps","value":442.3509},"test_metric":{"name":"splitkv399_full_ppl","value":2.376981}}

## Results

**VERDICT: the mechanism is REAL, not a small-workload artifact.** Fixed-order byte-exact split-KV serves **442.35 warm-median TPS** at the full `128×512` workload — **+85.29 TPS over surgical-357 (same-session control 357.06), ≈17.5·σ_hw** — while holding the PPL gate (2.37698 ≤ 2.42) and perfect run-to-run self-determinism (1.0). It does not merely reach the lawine #496 `399.97`; it **exceeds it by +42.4**, because split-KV parallelism grows with KV length, so the longer full-workload sequences favor it *more* than the 32×256 microbench did. The 357 lane was leaving speed on the table due to the surgical 2D realization giving up split-KV parallelism.

This is the **speed + PPL leg**. The full operative-identity / byte-exactness-vs-surgical census is land #515's leg — I report only the served self-determinism the cert produces and do not duplicate it.

### KEY OUTPUTS (PR-required)

| key | value |
|---|---|
| `splitkv399_full_warm_median_tps` | **442.35** (warm rounds 442.45 / 442.25; cold 439.16) |
| `splitkv399_full_ppl` | **2.376981** (≤ 2.42 ✓; shipped target 2.37673) |
| `splitkv399_full_self_determinism` | **1.0** (65536/65536 tokens, 0/128 seqs flipped, warm r1-vs-r2) |
| `vs_surgical357_tps_delta` | **+85.29** (same-session control) = **17.5·σ_hw** |
| `passes_ppl_gate` | **true** |

### Single-variable A/B (same session, identical harness/workload/seed)

| arm | submission | attention variable | warm median TPS | PPL | self-det | 128/128 | peak GPU |
|---|---|---|---|---|---|---|---|
| **variant** | `fa2sw_strict_byteexact_splitkv399` | `BYTEEXACT_FIXED_TPS=4` + `BYTEEXACT_NUM_SEGMENTS=64` (fixed-order 3D split-KV) | **442.35** | 2.376981 | 1.0 | ✓ | 21655 MiB |
| **control** | `fa2sw_strict_surgical357` | `SURGICAL_ATTN_USE_3D_OFF=1` (forced 2D order-preserving) | **357.06** | 2.376983 | 1.0 | ✓ | 21395 MiB |
| **Δ** | | only the attention realization | **+85.29** | +2e-6 | — | — | — |

The two packaged submissions are byte-identical except (a) the gated attention patch file (`byteexact_splitkv_patch.py` vs `surgical_attn_patch.py`), (b) one gated `sitecustomize` import, (c) the manifest `name`/`description` + the one attention env. **Both keep `SPECULATIVE_CONFIG` (spec-alive) and NEITHER sets `VLLM_BATCH_INVARIANT`** (no ~48% matmul tax), so the +85.29 isolates exactly the attention path.

### Mechanism validity (server-log grep, both arms)

- **variant** `mechanism_valid=True`: `[byteexact] armed` + re-jitted both `@triton.jit` kernels to `tiles_per_segment=4` + `NUM_PAR_SOFTMAX_SEGMENTS=64`, **no** fail-open `baseline kept`, **splitkv_redirects=5** (the M=8 spec-verify routed onto the fixed-order 3D path the patch made byte-exact), `matmul_tax_off=True`, onegraph captured, no fatal traceback.
- **control** `mechanism_valid=True`: `[surgical-attn] armed`+`forced is_batch_invariant=True`, **splitkv_redirects=0** (2D forced, verify did not redirect — exactly as designed), `matmul_tax_off=True`.
- control PPL `2.376982607605333` is byte-for-byte the prior this-pod recert (`l0attso0`), confirming the pod has not drifted; the variant/control PPL differ only at ~1.3e-6 (bf16-ULP reduction-order between the two exact paths), both far inside the gate.

### Gates

| gate | variant | requirement | pass |
|---|---|---|---|
| warm TPS vs 357.2 | 442.35 | strictly faster | ✓ (+85.29, 17.5σ) |
| vs 399.97 reference | 442.35 | hold / approach | ✓ exceeds by +42.4 |
| PPL | 2.37698 | ≤ 2.42 | ✓ |
| self-determinism (served r1↔r2) | 1.0 | byte-deterministic | ✓ |
| completion | 128/128 | full | ✓ |
| matmul tax | off | fast Marlin kept | ✓ |

### Exact commands (LOCAL, `analysis_only=true`, `official_tps=0`, no HF job, no `--launch`, no submission)

```bash
cd target/
# smoke (8×16, lever-fire sanity)
.venv/bin/python -m research.validity.splitkv399_full_recert.recert_ab --arm variant --smoke
# full 128×512 variant + control (3 decodes + full PPL each)
.venv/bin/python -m research.validity.splitkv399_full_recert.recert_ab --arm variant
.venv/bin/python -m research.validity.splitkv399_full_recert.recert_ab --arm control
# combine -> ab_summary.json + W&B (group splitkv399-full-recert)
.venv/bin/python -m research.validity.splitkv399_full_recert.recert_ab --summarize \
    --wandb-name stark/splitkv399-full-recert
```

Workload: 128 prompts × 512 output tokens, seed 1, `eval_prompts_sharegpt.json`; served through the same `scripts.local_validation.harness` path (`LocalServer` + official `decode_outputs.py` / `ppl_endpoint.py`) the surgical-357 fire used. `gemma-4-E4B-it` int4, vLLM 0.22.1rc1, sm_86 A10G 24GB. Self-determinism reuses the surgical357 cert's `token_identity` verbatim.

- **Peak GPU memory:** 21655 MiB variant / 21395 MiB control (~21.2 GB; `GPU_MEMORY_UTILIZATION=0.90` of 23028 MiB). Server ready in ~80s.
- **W&B:** `kwhylaeg` (A/B summary, all key metrics + artifact) — https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/kwhylaeg ; liveness `rkoxrnr7`. Group `splitkv399-full-recert`.

### Evidence used

- **lawine #496** (`42qroec1`): byte-exact split-KV `399.75/399.97 TPS`, but only on a 32×256 workload, no full PPL/operative cert — the candidate this PR recerts at full `128×512`.
- **Shipped surgical-357** (`j7qao5e9`; this-pod recert `l0attso0` = 357.22): the A/B control / 357.2 anchor.
- Packaging from lawine #500 (`submissions/fa2sw_strict_byteexact_splitkv399/`).

### What happened — honest analysis

The PR's open question was a clean either/or: is 357 leaving speed on the table (fixable), or is 399.97 a small-workload artifact that collapses at full `128×512`? **The data answers the first, emphatically.** Pinning `tiles_per_segment` to a fixed constant (split SIZE not COUNT) keeps the 3D split-KV parallelism while staying M-invariant byte-exact, and at full-length sequences that parallelism is worth **more**, not less — hence 442 > 400 > 357. The result is tightly measured (warm spread 0.2 TPS), the control reproduces the known 357.2 to <0.2 TPS, and the delta (17.5σ) dwarfs σ_hw. PPL and run-to-run self-determinism are clean.

**Caveats (do not over-read):**
1. These are **local pod warm wall-TPS** numbers (`official_tps=0`, no HF Job). The trustworthy quantity is the **same-session Δ**, not the absolute 442 as a leaderboard figure.
2. self-determinism `1.0` proves the variant is **run-to-run byte-deterministic**; it does **not** by itself prove operative-identity vs surgical-357 or vs plain greedy AR — that's **land #515's identity leg**.
3. **Penalize-lane only.** Both surgical-357 and this split-KV rung are **spec-alive → NOT private-safe** (denken #489 ~24% private-TPS-drift). So the finding is "*within the penalize lane*, fixed-order split-KV strictly dominates surgical-357 at the same byte-equivalence standard" — it is **not** a private-safe ship (the floor-lock `fa2sw_strict_m1ar_int4` 161.70 remains that). The DRAW/penalize-lane call is the human's.

**For the kanna #517 decision tree:** at reopen, *if* the penalize lane is ruled, the "keep 357 or upgrade?" answer is **upgrade to fixed-order split-KV (~442)** — a strictly-faster byte-exact rung — pending land #515's identity confirmation. This is the speed half; ubel #511 (EXP-2) is the comparison the human is A/B-testing this against.

### Suggested follow-ups

- **land #515 identity leg:** confirm operative-identity (margin/logit census vs surgical-357 and vs plain greedy AR) on this same fixed-order split-KV stack, so the 442 rung is fully recerted, not just self-deterministic.
- **`BYTEEXACT_NUM_SEGMENTS` sweep:** S=64 covers 4096 keys at T=4. A quick S∈{48,96} micro-sweep would show whether occupancy is left on the table at the full-workload sequence lengths (could push past 442 toward the ~457 strict-frontier prediction) — single-variable, local.
- **Private-safe variant:** if a spec-*dead* fixed-order split-KV (drafter off) still beats the 161.70 floor-lock, that would be the private-safe analog worth a separate PR.
