STUDENT land:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["2rmeroz8"],"primary_metric":{"name":"min_strict_bi_tps","value":156.95},"test_metric":{"name":"full_bi_necessary","value":1}}

## Results

**VERDICT: `FULL_BI_NECESSARY` (`full_bi_necessary=1`).** No realizable partial-BI config below the full fire stack achieves literal **128/128** served-greedy identity. The deployable realizable rungs cap at **24/128**; the single realizable rung that climbs higher (**84/128**) does so only by disabling CUDA graphs (a non-deployable diagnostic) and **still** falls short. So the cheapest strict (128/128) rung's anchored TPS is the full-BI fire anchor, **`min_strict_bi_tps = 156.95`**. The hypothesis stands; the ~157 fire is hardened as the honest strict floor.

This is an **analysis/identity card** — `analysis_only=1, official_tps=0, no_hf_job=1, fires=0`. Locked `int4_g128_lmhead`@126.378 untouched. Reused my own merged **#748** harness (`research/validity/strict_clean_served_byteexact_748/`, isolation-clean) + the merged **#750** anchoring method; one new local flashinfer boot-probe (no full arm). No served-file change, no `--launch`.

### 1. The realizable coverage ladder is COARSER than the hypothetical op-family ladder — two source-level findings

The card's suggested rungs ("matmul-only / matmul+attention / matmul+attention+reduction / full") are **mostly not realizable with current toggles**, for two concrete reasons I verified against the actual vLLM 0.22.0 source:

- **(a) `VLLM_BATCH_INVARIANT` is MONOLITHIC.** `batch_invariant.py:enable_batch_invariant_mode()` registers **all** op families in one call gated by a single bool — `aten::mm, addmm, matmul, linear, bmm, _log_softmax, softmax, mean.dim`. There is **no per-family env toggle**. "matmul-only" / "matmul+reduction" as separate rungs require *patching* that function → **out of scope** (honesty carry-forward).
- **(b) Gemma-4 ARCHITECTURALLY FORCES `TRITON_ATTN`, and TRITON+BI does NOT pin the attention split.** Boot-log proof (`runs/boot_FLASHINFER.log`, `config.py:100`): *"Gemma4 model has heterogeneous head dimensions (head_dim=256, global_head_dim=512). Forcing TRITON_ATTN backend to prevent mixed-backend numerical divergence."* Requesting `VLLM_ATTENTION_BACKEND=FLASHINFER` (or any FA backend) is **silently overridden** to TRITON_ATTN. And under TRITON_ATTN the parallel softmax-segment reduction is hard-coded `NUM_PAR_SOFTMAX_SEGMENTS=16` **unconditionally** — `VLLM_BATCH_INVARIANT` never appears in `triton_attn.py`, so **BI does not pin the attention split for this model**. The FA/flashinfer BI paths that *do* set `num_splits=1`/`disable_split_kv` (`flash_attn.py`, `flashinfer.py:559`) are unreachable for Gemma-4. flashinfer 0.6.11 *is* installed and boots — but is overridden to TRITON, so a flashinfer arm would just reproduce R1 (21/128); not run as redundant.

So the only **env-realizable** coverage axes (no vLLM patch) are: **BI on/off** (GEMM+reduction together; near-moot for the int4 path — see §3) and **CUDA-graph capture symmetry** (`--enforce-eager`, orthogonal to BI). That gives the (BI × capture) grid my #748 already measured.

### 2. The identity-vs-coverage Pareto (per rung: identity k/128 + anchored TPS)

Identity reference per rung = the **same-config spec-OFF served AR** rollout (only spec on/off differs within a pair); the **determinism floor = 128/128** (a BI=1 AR config run twice) certifies the stack is bit-reproducible within-config, so every <128/128 is a real reduction-order divergence, not noise. AR-only BI x-check = 18/128.

| rung | BI op-families | attn-split pinned | capture-sym | deployable | **identity** | **anchored TPS (fire)** | source |
|---|---|---|---|---|---|---|---|
| **R0** zero-BI (deployed) | — | no | no (graphs) | ✅ | **24/128** | 213.1 | land #748 (mine) |
| **R1** GEMM+reduction BI [route-b cheap] | mm/addmm/matmul/linear/bmm/softmax/mean | no | no (graphs) | ✅ | **21/128** | 198.4 | land #748 (mine) |
| **R2** + capture-sym [EAGER PROBE] | same | no | yes (eager) | ❌ diag | **84/128** | 57.4† | land #748 (mine) |
| G1 attention-split only (num_splits=1) | — | yes | no | ✅ | 24/128 | 236.0 | lawine #755 (PR curve pt) |
| G2 **FULL BI** (fire stack) | + attn-split + capture-aligned verify | yes | yes | ✅ | **128/128** | **156.95** | fern #750 (PR target) |

†R2's low anchored TPS is the **no-CUDA-graph penalty** (eager is ~4× slower), **not** a coverage cost — enforce_eager is a mechanism probe, never a deployable rung. It sits off the deployable frontier.

**Shape:** every *partial* coverage lands ≤ 84/128 — route-b BI (21), num_splits=1 alone (24, G1), even capture-symmetry-via-eager (84). Only the **full combination** (G2) reaches 128/128. The deployable partials (R0/R1/G1) are *fast* (~200–236) but nowhere near strict; the one realizable rung that climbs (R2, 84/128) is non-deployable and *still* 44 tokens short. **There is no realizable point that is simultaneously deployable, 128/128, and faster than 157** — the strict-identity frontier collapses to the single G2 point.

### 3. Why BI coverage is *not* the identity lever here (int4 + forced-TRITON)

Under the int4 QAT proxy with forced TRITON_ATTN, `VLLM_BATCH_INVARIANT=1` changes almost nothing greedy-relevant: the int4 matmuls are **Marlin** custom ops (already M-invariant — land #680: byte-identical across M, **not** routed through `aten::mm`), so BI's aten-GEMM overrides catch only a few bf16 ops; the attention segment reduction is unpinned (§1b); and softmax/mean don't move a greedy argmax. Hence **R1 (BI=1, 21/128) ≈ R0 (BI=0, 24/128)** — BI is ~null on this path. The only realizable identity gain (R1→R2, 21→84) comes from the **orthogonal** capture-symmetry axis (`enforce_eager`), consistent with #748's mechanism decomposition (CUDA-graph capture asymmetry = 81% of the per-token flip hazard; secondary = the M=K verify split). Both fixes together (= the full stack) are needed for 128/128 — neither alone, and neither realizable as a cheaper *deployable* rung.

### 4. Anchoring (fern #750 method, applied honestly)

`anchored = local_tps × (ANCHOR_OFFICIAL / local_int4_qat_nospec)`, `ANCHOR_OFFICIAL=95.463`. My **`bi0_spec0` arm IS that denominator** (int4-QAT, BI=0, spec-off, my api_server meter) = **95.2821**, matching fern's local anchor 95.19 to **0.097%** → my meter is anchorable (`R_int4=1.0019`). The bare api_server proxy sits at the int4_qat baseline (~95 local), **not** the fire stack (~229) — absolute TPS does not transfer, but the **relative BI/coverage cost does** (#748). So the table's `anchored TPS (fire)` is the official-anchored local **prediction** = `local_tps × (229.847 / 95.2821)`, placing each rung on fern's realized 229.85→156.95 axis via the transferable relative cost (literal int4_qat anchoring instead leaves R0/R1 at ~88/82 — i.e. confirms the proxy ≈ the int4_qat baseline, not the fire stack).

### Command

```bash
cd target/
# realizable (BI × capture) grid + determinism floor: REUSED from merged #748 (no re-run)
#   research/validity/strict_clean_served_byteexact_748/runs/{bi0_spec0,bi0_spec1,bi1_spec0,
#   bi1_spec1,bi1_spec0_eager,bi1_spec1_eager,bi1_spec0_rep}
# new this card:
python3 research/validity/partial_bi_identity_pareto_760/boot_smoke.py FLASHINFER   # forced-TRITON probe
python3 research/validity/partial_bi_identity_pareto_760/pareto.py                  # assemble Pareto + verdict
python3 research/validity/partial_bi_identity_pareto_760/wandb_log.py               # log group fire_bi_tax_750
```

- **Peak GPU mem:** ~19.5 GiB (reused #748 `bi1_spec` 19543 MiB; flashinfer boot-probe same model footprint) on the single A10G.
- **W&B run:** [`2rmeroz8`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/2rmeroz8) (group `fire_bi_tax_750`).
- **Reused W&B:** #748 [`fikec7di`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/fikec7di).

### What happened — honest analysis

The hypothesis (`full_bi_necessary=1`) **holds**. Walking the realizable BI-coverage ladder, every partial config — BI-alone (21/128), num_splits=1-alone (24/128, given), and even capture-symmetry-via-eager (84/128) — falls short of 128/128; only the full fire combination reaches it. The deeper reason this card surfaces: on the int4 + **architecturally-forced-TRITON** served path, the *intended* BI op-family levers are near-moot (Marlin GEMM already M-invariant; TRITON attention split unpinned by BI), so the realizable identity variation is driven by the **orthogonal** CUDA-graph-capture axis, which is non-deployable when used as a lever (eager) — and the deployable rungs never exceed 24/128. The strict-identity frontier is therefore a **single point** (full BI, 128/128, anchored 156.95) with no cheaper realizable deployable neighbour. **No refutation found; the ~157 fire is the honest strict floor.**

### Suggested follow-ups

- **The only path to a cheaper strict rung is a vLLM-source change** (out of scope here), e.g. capturing the M=K verify shape under BI (close the dominant 81% capture-asymmetry hazard *without* eager) or forcing TRITON's 2D single-pass attention for all shapes (kill the 16-segment reduction). Either would test whether a *deployable* 128/128 below the full tax exists — worth an advisor-scoped patch card if the ~157 floor is binding.
- **Don't-care near-tie band** (cf. land #654/#748): 104/107 of the route-b residual onsets are pure bf16-ULP ties and PPL is BI-neutral (#748: Δ0.001%), so a tie-tolerant identity gate would pass the benign ~97% — but the strict #319 contract still needs the verify fix for the marginal confident tail.

### Public evidence used

Challenge leaderboard frontier is now ~**504 TPS** (top rows; the ~489 `osoi5…lmhead12k-fa2sw-precache` lane has been passed) — a different lane from this DQ-risk-free byte-exact route over the locked **126.378** anchor. This card is the **identity-coverage half** of the #750 3-way decomposition: it produces the clean identity-vs-coverage Pareto and confirms (against lawine #755's num_splits=1=18.75% and fern #750's full-BI=128/128@156.95 curve points, both given in the card body) that full batch-invariance is **necessary** for 128/128 served-greedy identity on the realizable proxy.
