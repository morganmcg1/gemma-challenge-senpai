STUDENT lawine:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"analysis_only":true,"official_tps":0,"wandb_run_ids":["cn4cqim9","i11p5e3y","b0244ccu","jo0a050y"],"primary_metric":{"name":"gap_largest_lever_tps_byteexact","value":87.74},"test_metric":{"name":"byteexact_fixed3d_local_wall_tps_128x512","value":439.71},"key_outputs":{"realization_gap_tps":99.9,"gap_largest_lever_name":"attention_splitkv_geometry_2D_inorder_to_fixed3D","gap_largest_lever_tps":87.74,"gap_largest_lever_is_byteexact":true,"gap_is_kernel_overhead":true,"457p5_is_128x512_measured":false}}

## Results

**Headline: the ~100 TPS gap is DOMINANTLY (≈88%) kernel/realization overhead — closable and quality-safe — NOT an unavoidable byte-exact cost. The single largest realizable lever is the attention split-KV geometry (2D in-order → fixed-3D split-KV): +87.74 byte-exact local TPS. The 457.5 "frontier" was a single-shape microbench projection, never a served 128×512 number.** #481's realization-overhead hypothesis is **confirmed**.

All LOCAL on-pod, `analysis_only=true`, `official_tps=0`, no HF Job/submission/served-file change. Robust metric = `wall_tps` (median, official-spec; my #72). σ_hw 4.864, A10G sm_86 24 GB, int4 vLLM 0.22.1rc1.

### KEY OUTPUTS
| output | value |
|---|---|
| `realization_gap_tps` (457.5 − 357.6) | **99.9** |
| `gap_largest_lever_name` | **attention split-KV geometry (2D in-order → fixed-3D)** |
| `gap_largest_lever_tps` | **+87.74** (byte-exact, same-session) |
| `gap_largest_lever_is_byteexact` | **true** (0/8 microbench, all geometry configs; PPL 2.3767; 128/128) |
| `gap_is_kernel_overhead` | **true** (83.1% of the surgical→457.5 span is byte-exact-recoverable) |
| `457p5_is_128x512_measured` | **false** (microbench projection @ KV-len 640, never served) |

### 1. 457.5 provenance (task step 1) → NOT a 128×512 served number
`457p5_is_128x512_measured = FALSE`. It is a **single-shape microbench projection**:
- `research/speed/strict_frontier_realize/strict_frontier_realize.json`: KV-len **640** (headline_L), M=8 verify, hd512, 7 full layers; per-cycle attention added-µs applied to the deployed 481.53 decode cycle → `realized_strict_frontier_tps=456.36`. No serve, single locus.
- `research/speed/surgical_attn_realize/PR488_verdict_integrated.json`: `modeled_surgical_estimate=457.0` (projection) vs `surgical_attn_only_tps=357.64` (**realized serve**); verdict `is_457_a_mirage: "partially — the rung is REAL but the ~457 magnitude was a ~22% overshoot … Realized surgical = 357.6, not 457."`

When actually served at full 128×512, the byte-exact lane realizes **351.97 (surgical) / 439.71 (byteexact)** local; the fast NON-byte-exact stack tops **453.93** same-session. **457.5 exceeds even the real fast stack** → a projection, not a frontier. So ~3.6 of the headline "100" is illusory.

### 2. Component-attributed ledger (same-session, batch A, n=3 back-to-back @ 128×512)
| arm | attention scheme | byte-exact attn? | local wall_tps | PPL | verify exec.gpu ms |
|---|---|---|---|---|---|
| **surgical-357** | 2D in-order | **yes** (0/8) | **351.97** | 2.37668 | 8.064 |
| **byteexact-399** | fixed-3D split-KV | **yes** (0/8) | **439.71** | 2.37666 | 6.890 |
| deployed (ship-fast) | adaptive-3D split-KV | **no** (18 straddle flips) | 453.93 | 2.37668 | 6.804 |
| 457.5 "frontier" | — (microbench projection) | — | *never served* | — | — |

Decomposition of the 357.6→457.5 gap (same-session basis):
```
surgical 2D byte-exact floor          351.97
 + split-KV geometry  (BYTE-EXACT)    +87.74  ->  byteexact 439.71   [LARGEST LEVER]
 + fixed->adaptive    (NOT byte-exact)+14.22  ->  deployed  453.93   [fundamental byte-exact tax]
 + projection overshoot (never served)+3.57   ->  457.5 (illusory)
```
**88% (+87.74/99.9) of the gap is the attention split-KV geometry that surgical-357 surrendered by taking the 2D in-order path** — recovered *inside the byte-exact lane* by the fixed-3D split-KV scheme (M-invariant ⇒ byte-exact, yet keeps the parallel split-KV reduction). Only **~14 TPS is fundamental** (the adaptive geometry's per-shape M-dependence — exactly what makes deployed non-equivalent — which you cannot keep while byte-exact).

Steptime corroboration: verify-step GPU time (where attention lives) drops **8.06 → 6.89 → 6.80 ms** (surgical → byteexact → deployed); the TPS delta lands in `exec.gpu`, confirming the lever is the attention path (host-gap and drafter-GPU also drop because surgical's 2D in-order penalizes every attention call).

### 3. #481 realization-overhead hypotheses, with numbers
- **Attention split-KV geometry** (hold coverage = 4096 keys, vary parallel segments @ L=512; **all 0/8 byte-exact**, adaptive contrast = 18 straddle flips):

  | segments @ L=512 | config | local wall_tps |
  |---|---|---|
  | 2 | T16/S16 | 383.35 |
  | 4 | T8/S32 | 415.01 |
  | **8** | **T4/S64 (packaged)** | **439.71** ← optimum |
  | 16 | T2/S128 | 389.60 |

  Segment count moves served TPS by ~57 (non-monotonic, peak at 8 segs). **The packaged byteexact config is already at the geometry optimum** — no extra free TPS from re-tuning, and every config stays byte-exact.
- **CUDA-graph capture** (drafter ONEGRAPH=1 vs eager K-iters ONEGRAPH=0): **+3.77 TPS** (439.71 vs 435.94). Small — graph overhead is **not** the gap.
- **FlashInfer sampler**: **unmeasurable locally** — server exits code 1 on the documented cuRAND-JIT toolchain crash (byteexact armed 4/64 first, so not a byteexact issue). Local-only; works on HF a10g. Not dominant regardless (sampler is a sliver of the decode step vs attention+matmul).
- **Reduction-order**: the recoverable part is recovered *without* breaking byte-exact identity — the fixed-3D split-KV is M-invariant (0/8). The non-recoverable ~14 TPS is precisely the adaptive (M-dependent) reduction order.

### Verdict
The ~100 TPS realization gap is **kernel-overhead (closable, quality-safe), not fundamental.** ~88% is the attention split-KV geometry — already realized by the packaged `byteexact-399` rung at **439.71 local @ 128×512** (well above its 399.97 32×256 headline), 0/8 byte-exact attention, PPL 2.37666, 128/128, served warm r1-r2 = 1.0. Only ~14 TPS is the genuine byte-exact tax (fixed-vs-adaptive). 457.5 itself was a microbench projection.

**Largest realizable lever, draw-ready check:** speed +87.74 (≈18σ_hw) ✓ · PPL 2.37666 ≤ 2.42 ✓ · 128/128 ✓ · byte-exact attention 0/8 ✓ · served r1-r2 = 1.0 ✓ — **but** end-to-end token census = **5 ULP-tie flips (0 semantic) vs surgical's 1** (my #500/#461 locus). Those 5 flips come from the **shared un-taxed Marlin matmul, NOT the attention geometry**; matching surgical's 1-flip census needs the matmul tax (~135 TPS, #488), defeating the purpose. So the byteexact rung is the correct *fast byte-exact-attention* frontier point; whether it clears the team's strict bar reduces to the same open question my #500 flagged — are 5 ULP-tie (0-semantic) census flips acceptable.

### Comparison vs PR baselines
- surgical-357: anchor 357.6 → measured **351.97** same-session (−5.6, ~1.2σ_hw, fire-time noise). PPL 2.37668 (anchor 2.37673). ✓
- byteexact-399: anchor 399.97 @ 32×256 → measured **439.71 @ full 128×512** (+39.7; full-workload reads *higher* than the small-shape headline). PPL 2.37666 (anchor 2.37666). ✓ (consumes/agrees with stark #519's full-workload recert direction; I did not re-run their net-TPS A/B.)
- deployed non-strict ref 481.53 official → 453.93 local same-session (precache-skipped local; 465 local / 481.53 official in other sessions, my #267).

### Repro
```bash
cd target && export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
# A ledger (deployed+surgical+byteexact, 3 decodes + PPL)
$PY -m research.speed.byteexact_realization_gap.run_realization_gap \
    --arms deployed,surgical,bx_T4_S64 --n-decodes 3 --tag ledger \
    --wandb-name lawine/realization-gap-ledger --wandb-group byteexact-realization-gap
# B geometry sweep (2/4/16-seg byte-exact configs, +PPL)
$PY -m research.speed.byteexact_realization_gap.run_realization_gap \
    --arms bx_T16_S16,bx_T8_S32,bx_T2_S128 --n-decodes 2 --tag geometry --wandb-group byteexact-realization-gap
# M microbench 0/8 byte-exactness per geometry (no serve)
$PY -m research.speed.byteexact_realization_gap.run_microbench_sweep
# C levers (FlashInfer sampler + eager-drafter cudagraph)
$PY -m research.speed.byteexact_realization_gap.run_realization_gap \
    --arms bx_fisampler,bx_eager_drafter --n-decodes 2 --no-ppl --tag levers --wandb-group byteexact-realization-gap
# verdict consolidation (CPU-only)
$PY -m research.speed.byteexact_realization_gap.summarize_verdict --wandb-group byteexact-realization-gap
```
- **Peak GPU mem:** 19535 MiB (~19.1 GiB). **NaN-clean** (bx_fisampler NaN is the expected local sampler crash, handled).
- **W&B:** verdict `cn4cqim9` · ledger `i11p5e3y` · geometry `b0244ccu` · levers `jo0a050y` (group `byteexact-realization-gap`).
- **Pristine restore:** all experimental levers (surgical / byteexact / split-KV / steptime) are env-gated **in-process runtime monkeypatches** (sitecustomize meta-path finders) — **0 experimental on-disk markers in both venvs**, serve-venv `triton_unified_attention.py` byte-identical to stock. The PLE/sampler/orjson markers in the serve venv are the submission's normal idempotent serve-time patches (present before this run, re-applied every serve), not from this experiment.

### What happened
The gap is real but mostly *self-inflicted by surgical-357's attention scheme*, not by byte-exactness. Surgical reached byte-exact identity by forcing the 2D order-preserving sequential-KV path, which throws away split-KV parallelism (+the matmul-tax-free Marlin). The fixed-3D split-KV (byteexact) gets the *same* byte-exact attention (M-invariant fixed order ⇒ 0/8) **and** keeps the parallel reduction — recovering +87.74 of the ~100. The residual ~14 is the only part that needs the M-dependent adaptive geometry (non-exact). 457.5 was never reachable as a served number: it was a per-cycle-Δµs projection at a single KV length.

### Suggested follow-ups
1. **Re-anchor the team's "strict frontier" to a served number.** Replace 457.5 (projection) with the measured byteexact **439.71 local @ 128×512** (→ ~455 official via τ_lo 1.0352) as the byte-exact-attention frontier; the projection over-promised by ~18 TPS.
2. **Close the residual ~14 TPS** only if a *fixed-order* approximation of the adaptive per-shape geometry can stay M-invariant (0/8) — otherwise it is fundamental and should be retired as a lever.
3. **The real remaining decision is the matmul census, not attention.** The byteexact rung's 5 vs 1 census flips are 100% the un-taxed Marlin matmul. A batch-invariant *Marlin verify GEMM* (lane-a, my #196) is the only thing that tightens the census without the full ~135 TPS tax — that's where the next byte-exactness work should go, not attention.
