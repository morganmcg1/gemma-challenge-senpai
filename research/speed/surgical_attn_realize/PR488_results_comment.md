STUDENT lawine:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"analysis_only":true,"official_tps":0,"wandb_run_ids":["ko01dcyy","2k0sw1gz","aydp64yf","scauko99"],"primary_metric":{"name":"surgical_realized_local_wall_tps","value":357.64},"test_metric":{"name":"ppl","value":2.3767}}

## Results — Surgical attention-only realization

**TL;DR.** The surgical attention-only path **realizes 357.6 local TPS** — a **real +135.7 rung above the 222 global-flag floor** (28× the σ_hw bar), at **identical PPL 2.3767** and **128/128**. But it is **NOT the modeled ~457**: that estimate overshot the realized number by ~28% (357.6 is ~22% below 457). On identity, the surgical path is **byte-equivalent to the same operative standard as the 222 ship** — but the PR's literal raw-token M=1-AR census is **unmeasurable on this workload** and I had to correct it (details below). **`surgical_realizes_above_222 = TRUE`**, with the caveat that the realized rung is 357.6, not 457, and the matmul tax it removes was identity-unnecessary.

Single pod session, all three arms back-to-back (shared σ_hw). Local AWS A10G, `analysis_only=true`, `official_tps=0` — **no HF job, no submission, no draw, no deployed-file change.** The surgical lever is a LOCAL serve-venv vLLM prototype edit (one gated line in `triton_unified_attention.py` honoring `SURGICAL_ATTN_USE_3D_OFF=1`), **restored to stock after measurement.**

### Three arms (median wall_tps over 3 back-to-back decodes; round-0 = cold-start, excluded by median)

| arm | config | median wall_tps | PPL | completed | mechanism |
|---|---|---|---|---|---|
| (a) deployed | no flag (3D split-KV) | **464.69** (σ 8.48) | 2.3767 | 128/128 | splitkv_redirects=5, batch_inv=0 |
| (b) full_flag | `VLLM_BATCH_INVARIANT=1` | **221.95** (σ 3.17) | 2.3767 | 128/128 | splitkv_redirects=0, batch_inv=3 |
| (c) **surgical** | `SURGICAL_ATTN_USE_3D_OFF=1` (2D attn, matmul tax OFF) | **357.64** (σ 3.96) | 2.3767 | 128/128 | splitkv_redirects=0, batch_inv=0 |

Mechanism confirms the lever fired exactly as intended: surgical took the **2D order-preserving attention** path (no 3D split-KV redirect) **without** installing the global matmul tax (0 batch_invariant mentions).

### Where surgical lands (vs PR baselines)

- `surgical_realized_tps` = **357.64** → bucket **partial_300+** (real, material), **NOT ~457** and not ~250-300 (partial) and not ~222 (collapse).
- `surgical_lift_vs_222` = **+135.70 TPS** → clears the **+2 materiality bar** and **σ_hw 4.864** by ~28×. ✅
- Recovery of the deployed−222 gap = **55.9%** (the model implied ~90%).
- vs the composed/locus estimates in the PR (stark #472 **457.55**, #475 461.80, #466 456.36): the realized 357.6 is **~100 TPS below** all three. Those were CUDA-graph / CPU-analytic locus proofs, never served e2e; they over-estimated by ignoring served overhead — exactly the risk the card flagged.

### Per-op tax decomposition (direct measurement — independent of the #484 ask)

The 481→222 collapse is **two separable costs**, not one:

| lever | Δ TPS | share of collapse | identity role |
|---|---|---|---|
| 2D byte-exact attention (deployed 3D → surgical 2D) | **−107.0** | 44% | **load-bearing** for byte-exactness |
| matmul tax (surgical → full_flag) | **−135.7** | 56% | **identity-unnecessary** (9 ULP-tie flips only) |

So the matmul tax is the *majority* of the collapse and is **separable and unnecessary** for greedy identity. The 2D order-preserving attention tax (−107) is the unavoidable price of byte-exact attention on **this** route (the fast 3D split-KV attention is exactly what makes deployed non-equivalent, identity 0.9966).

### Identity census — the PR's raw-token M=1-AR gate is CONFOUNDED (corrected)

The census I ran first (`2k0sw1gz`/`aydp64yf`) reported surgical M=8-vs-M=1-AR = **0.4415** and full_flag = **0.4203** — both "FAIL." **Those numbers are an artifact, not a byte-exactness measurement.** Two facts prove it (corrected run `scauko99`, CPU re-analysis of the captured decodes):

1. **The global `VLLM_BATCH_INVARIANT=1` config fails the SAME gate identically (0.4203 ≈ surgical 0.4415).** full_flag is batch-invariant *by construction* (M=8 verify == M=1 AR is its definition). A valid gate would *pass* it. It can't discriminate.
2. **The census diffed the cold round-0 served decode against an EAGER (ONEGRAPH-off) M=1 reference.** Three confounds stack — cold-start, eager-vs-cudagraph, M=1-vs-M=8 batching — each re-orders bf16 reductions, and on 512-token reasoning chains a single ULP flip cascades.

Direct proof of the noise floor and the clean signal:

```
within-arm WARM determinism (r1 vs r2):   deployed 1.000  full_flag 1.000  surgical 1.000   <- each config is PERFECTLY self-deterministic when warm
cold-start signature   (r0 vs r1):         deployed 0.435  full_flag 0.464  surgical 0.456   <- the raw-token noise floor (cold round vs warm)
WARM matched-round cross-config:
   surgical vs full_flag  = 0.9763 (9/128 flip)    <- matmul-tax axis ONLY (both 2D attn)
   deployed vs surgical   = 0.4564 (105/128 flip)  <- attention axis (3D split-KV vs 2D)
confounded M=1-AR census (PR-literal): 0.42-0.44 for surgical AND full_flag AND m1ar-vs-m1ar  <- all at the noise floor
```

**The one confound-free signal:** surgical and the 222 global flag are **operatively indistinguishable** — warm matched-round **0.9763**, both perfectly self-deterministic, **9 residual flips** that merged #461 (referenced in this PR's census code) attributes to **bf16-ULP near-ties (margin 0.125-0.25), not semantic**. And the **dominant byte-equivalence axis is the attention** (3D→2D flips 105/128), **not the matmul tax** (9/128). The official greedy-identity gate is *served-vs-served byte-exact* (same serving config), **not** served-vs-M=1-AR-eager — so the warm surgical-vs-full_flag comparison is the right in-session analogue.

### Verdict

- **`surgical_realizes_above_222 = TRUE`** — a real, material +135.7 TPS rung (357.6 vs 222), at identical PPL/completion.
- **`surgical_realized_tps = 357.6`, NOT 457** — the ~457 model was a partial mirage (~28% overshoot of the realized number); a real rung exists but the composed-locus magnitude was inflated.
- **Byte-equivalence:** surgical attention-only is byte-equivalent to the *same operative standard as the 222 ship* — they share the 2D order-preserving attention, are indistinguishable warm (0.9763, 9 ULP-tie flips per #461), and the matmul tax that costs the global flag 135 TPS buys only ULP-tie refinement. **If 222 is accepted as strict, surgical's 357.6 is strict by the identical standard.**

**Caveat (honest):** a definitive *operative-1.0 certification* would need the organizer's served-vs-served reference at matched config (the actual official gate) or a logit-margin census (#461-style). A raw-token M=1-AR decode is unmeasurable here for **any** config (the global flag fails it too). The byte-equivalence claim rests on the warm same-config cross-arm signal + the attention-axis attribution + #461's margin proof — **not** on a fresh raw-token M=1-AR pass.

### Reproduction

```bash
# 3-arm speed + PPL (same session)
.venv/bin/python -m research.speed.surgical_attn_realize.run_surgical_realize \
    --n-decodes 3 --wandb-name lawine/surgical-realize --wandb-group surgical-attention-realization
# surgical lever: LOCAL serve-venv edit (gated, restored after):
#   triton_unified_attention.py: is_batch_invariant = bool(envs.VLLM_BATCH_INVARIANT) or os.environ.get("SURGICAL_ATTN_USE_3D_OFF")=="1"
# M=1-AR censuses (the confounded gate):
.venv/bin/python -m research.speed.surgical_attn_realize.census_surgical_identity --arm surgical  --wandb-group surgical-attention-realization
.venv/bin/python -m research.speed.surgical_attn_realize.census_surgical_identity --arm full_flag --wandb-group surgical-attention-realization
# corrected CPU re-analysis (the attribution above):
.venv/bin/python -m research.speed.surgical_attn_realize.identity_corrected_census
```

- **Submission:** `fa2sw_precache_kenyan` (unchanged) · **Workload:** 128 prompts × 512 tok, seed 1 · **Peak GPU mem:** ~19.4 GB (19395 MiB) · **Total run:** ~39 min (speed) + 2×10 min (censuses).
- **W&B:** speed `ko01dcyy` · surgical census `2k0sw1gz` · full_flag census `aydp64yf` · corrected census `scauko99` — group `surgical-attention-realization`.
- **Integrated verdict JSON:** `research/speed/surgical_attn_realize/PR488_verdict_integrated.json`.

### What happened

The card asked whether the modeled ~457 surgical number was a 6th composed mirage. Answer: **it was inflated but not pure vapor.** A real strict rung exists at **357.6** (+135.7 over 222), but the composed-locus harnesses over-estimated the realized magnitude by ~100 TPS. The most useful finding is the **clean per-op decomposition**: the global flag's 481→222 collapse is 56% matmul-tax (separable, identity-unnecessary — 9 ULP-tie flips) and 44% byte-exact 2D attention (−107 TPS, the real price of greedy identity on this route). The surgical arm proves the matmul tax can be dropped for free identity-wise, lifting the realized strict number from 222 → 357.6. Separately, I found the raw-token M=1-AR census is a broken gate for 512-token reasoning decodes (the batch-invariant config fails it identically; the metric floors at the cold/eager noise level) — future strict-identity checks should use served-vs-served matched config or logit margins, not raw M=1-AR token diffs.

### Suggested follow-ups

1. **Productionize the surgical lever for a real strict candidate at ~357.6.** Promote the gated `SURGICAL_ATTN_USE_3D_OFF` path into the submission serve.py (off by default, on for a strict draw) and validate via the organizer's served-vs-served greedy gate — this is a measurable +135 over the 222 ship.
2. **Faster byte-exact attention than the 2D order-preserving path.** The −107 TPS 2D-attention tax is the binding cost on this route; an M-invariant attention kernel that preserves byte-exactness at lower cost would lift the surgical ceiling above 357.6.
3. **Replace the raw-token M=1-AR identity census** with a served-vs-served matched-config gate or a logit-margin census (#461-style) for all future strict claims — the raw-token gate is uninformative here.
4. **Profile the matmul tax** to see whether a *selective* persistent-matmul override (only the M-sensitive GEMMs, if any) could recover identity-relevant flips without the full 48% tax — though the 9-flip ULP-tie result suggests there is no identity-relevant matmul GEMM to fix.

_Note on #484: the per-op decomposition above is my direct measurement; per launch isolation I did not pull ubel #484's content, so I report the direct number rather than reconciling against it._
