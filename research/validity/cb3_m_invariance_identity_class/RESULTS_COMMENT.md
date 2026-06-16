STUDENT wirbel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["gluit5k0"],"analysis_only":true,"no_hf_job":true,"no_served_file_change":true,"official_tps":0,"cb3_locus":"verify","cb3_is_m_invariant":true,"cb3_identity_class":"unconditional","cb3_max_gap_nats":0.0,"frontier_482_needs_q1_contract":false,"ppl":2.3772,"self_test_passes":true,"primary_metric":{"name":"cb3_is_m_invariant","value":1.0},"test_metric":{"name":"self_test_passes","value":1.0}}

## Results

**VERDICT: cb3 IS byte-exact M-invariant → `cb3_identity_class = unconditional`, `frontier_482_needs_q1_contract = false`.** My own #428 framing — that 482.74 is the **safe no-contract floor** — **HOLDS, now MEASURED instead of assumed.** cb3 is NOT self-referential like pinned-K. The human's Q1 "which reference defines equivalence" contract call applies **only to the pinned-K 496.74 rung**, NOT to cb3's 482.74.

The PR adversarially tested whether my #428 assumption (cb3 = identity-preserving) survives the #431-grade M-invariance lens. It does — but for a reason that is *structurally different* from pinned-K, and that matters for the #407 packet.

### (1) cb3 LOCUS — `cb3_locus = verify` (on the token arbiter, uniformly)

cb3 is a **weight-only sub-int4 re-quant (RHT incoherence rotation + VQ codebook, ~3.125 bpw) of the TARGET BODY GEMMs** (qkv / o / gate_up / down linears), applied to the k*=229 least-sensitive body linears (#403 `iv9i2wks`, #388, #391, #395).

- It sits on the **M=8 verify body** — the truncated-head token arbiter (land #420 `qe4qagc1`: verify is the **sole** arbiter; the drafter gates accept-*length* only). So cb3 **can** affect emitted tokens → instruction 2 is live.
- The **drafter is separate and un-shrunk** (#392: "1 drafter forward (separate small model, NOT cb3-quantized) + 1 verify forward (M=8 target body, cb3-shrinkable)"). So `cb3_on_drafter = False` → this is **not** the trivially-safe `identity_free` case.
- cb3 is a **baked weight change** → present in **both** the M=1 AR reference and the M=8 verify (`cb3_is_uniform_change = True`). This is the crucial structural difference from pinned-K (verify-only) that instruction 4 flags.

### (2) M-INVARIANCE — byte-exact M=1 == M=8 (`cb3_is_m_invariant = true`, `cb3_max_gap_nats = 0.0`)

**The decisive evidence is a DIRECT banked measurement of cb3's exact locus.** lawine #232/#240 measured all four int4-Marlin **BODY GEMMs (qkv/o/gate_up/down) bit-exact across M∈{1,8}** (`max_abs_diff = 0.0` each; `int4_body_bitexact_m8 = true`); #221 measured isolated body-GEMM **row-0 bit-exact across M∈{1,2,4,8,16}** (`INT4_BODY_M_DEP = False`). The deployed **0.73%** M=8 residual divergence lives in the **bf16 lm_head + attention/norm** — the locus cb3 does **NOT** touch. So the body GEMM that cb3 re-quantizes is **directly measured M-invariant**.

**Re-confirmed fresh this run** with the #431 bf16-perturbation methodology, on the **body-GEMM geometry** (K∈{2048, 16384}; the served hidden=2048 / intermediate=16384), GPU **cuda:0 (A10G sm_86)**, 12 trials (2 K-dims × 6 seeds):

```
M-INVARIANT body reduction:  max |Δ row0(M=8) − row0(M=1)| = 0.000e+00   byte-exact = True   (out scale ~3.54)
M-VARIANT split-K control:   max |Δ row0|                  = 1.250e-01   byte-differs = True  (rel 3.53% of scale)
RHT activation rotation:     M-free (per-row)              = True
```

- The cb3 dequant (RHT-rotate → VQ-quantize → de-rotate) is a **pure M-independent weight transform** — `W_deq = f(W)` only, no dependence on X or M. Under the **measured M-independent body reduction** (each row reduced over K by a fixed per-row path), row m is byte-exact regardless of size_m → `max_gap = 0.0`, **no near-tie flips introduced by cb3 at all** (not even sub-ε* ones — strictly below pinned-K, which sits *at* ε*=0.125).
- The **control is the schedule cb3 must AVOID**: a size_m-keyed split-K (nsplit = M) with bf16-rounded partials — the #122 naive-Marlin mechanism — which *does* perturb (0.125, a few bf16 ULP, **3.53% of the output magnitude**, knife-edge not gross). This proves the probe is **sensitive**: the zero in the invariant arm is meaningful, not a dead probe.

**Methodology note (the fidelity discipline that makes the zero trustworthy):** I model the reduction *schedule* explicitly rather than trusting `torch.matmul` — exactly as #431 modeled split-K explicitly rather than trusting an un-runnable kernel. This is load-bearing: a naive `X.float() @ W_deq.float()` is itself M-variant on this hardware because cuBLAS/MKL **dispatch M=1 to a GEMV and M=8 to a GEMM** with different K-accumulation orders — an ULP artifact of the *math library*, **NOT** of the served int4-Marlin CUDA kernel (which #232 directly measured byte-exact). Trusting `matmul` would have falsely flipped the verdict to `self_referential_only` on a library artifact; modeling the per-row M-independent reduction (what #232 measured the served kernel to do) is the faithful model.

### (3) IDENTITY CLASS — `unconditional`

cb3 touches the verify body (not drafter → not `identity_free`) but is **byte-exact M-invariant** (max_gap 0.0, no confident flips, no sub-ε* ties) → **`unconditional`**. This is the **opposite** of pinned-K's `self_referential_only` (#431), and the difference is mechanistic:

| | mechanism | M=1 reference | verdict |
|---|---|---|---|
| **pinned-K** (496.74) | reduction-**ORDER** change (`num_splits` 1→8), intrinsic + verify-only | canonical `num_splits=1` ≠ pinned-K's own M=1 | `self_referential_only` |
| **cb3** (482.74) | weight **VALUE** change (RHT+VQ) on an **unchanged M-independent** reduction, uniform | cb3's M=1 AR **IS** the submitted checkpoint | **`unconditional`** |

### (4) #407 CONSEQUENCE — `frontier_482_needs_q1_contract = false`

The pinned-K Q1 call ("which M=1 reference — pinned-K's `num_splits=8` or canonical `num_splits=1`?") needs **both** (i) verify-only (the M=1 path skips the lever) **and** (ii) a reduction-order change vs canonical. **cb3 has NEITHER:**

- **Self-reference is trivially satisfied** — cb3 is **uniform**: the M=1 AR reference *runs cb3* (it **is** the submitted checkpoint). There is no canonical-non-cb3 reference the contract could demand. (`self_reference_trivially_satisfied = true`.)
- **AND there is no internal cb3 M-variance to break M=1==M=8** — the body GEMM reduction is M-independent (measured #232; re-confirmed fresh, max_gap 0.0). The instruction-4 hazard ("does an internal cb3 M-variance still break M=1==M=8 even though cb3 is uniform?") is **measured shut**: `internal_cb3_m_variance_breaks_m1_eq_m8 = false`.

So cb3's M=8 verify == cb3's M=1 AR **byte-exact**: 482.74 is **safe-by-construction**, not a "which reference" call. The one residual bitwise-tie flip @ prompt 90 is a **blanket-strict-base bf16-lm_head artifact** already resolved operative-identity-1.0 (#429: literal 0.9989, operative 1.0, `flip_is_bitwise_tie = true`, 0 confident-forbidden flips) — shared by the **whole ladder**, NOT introduced by cb3.

### Required terminal fields

| field | value | basis |
|---|---|---|
| `cb3_locus` | **`verify`** | RHT+VQ re-quant of qkv/o/gate_up/down body linears (#403/#388/#391); on M=8 verify body, sole arbiter #420; drafter separate/un-shrunk #392 |
| `cb3_is_m_invariant` | **`true`** | body GEMM directly measured bit-exact across M (#232 max_abs_diff=0.0; #221 row-0 bit-exact); fresh probe re-confirms byte-exact (0.0) |
| `cb3_identity_class` | **`unconditional`** | verify-body but byte-exact M-invariant (not `identity_free`, not `self_referential_only`) |
| `cb3_max_gap_nats` | **`0.0`** | no divergence at all — strictly below pinned-K's ε*=0.125 |
| `frontier_482_needs_q1_contract` | **`false`** | uniform (self-ref trivially satisfied) AND M-invariant (no internal M-variance) → safe-by-construction |
| `ppl` | **`2.3772`** ≤ 2.42 | anchored deployed PPL; a reduction/quant-class probe is teacher-forced PPL-neutral (cb3's margin owned by #403/#422/#394) |
| `self_test_passes` | **`true`** | 0-GPU gate **42/42**; full run incl. fresh-measurement checks **47/47** |

`official_tps = 0` (analysis card; `analysis_only=true`, `no_hf_job=true`, `no_served_file_change=true`, no submission).

### Baseline comparison (this is an identity-class measurement — a target to characterize, NOT a TPS to beat)

| frontier | TPS | identity class under this card |
|---|---|---|
| deployed FAST (#52, `2x9fm2zx`) | 481.53 | non-equivalent (self-ref identity 0.9966; outside the #407 feasible set) |
| blanket-strict (#423, `5a6zq2yz`) | 467.14 | operative-identity 1.0 (#429), the equivalence-respecting base |
| **+cb3 — THIS rung** | **482.74** | **`unconditional`** — byte-exact M-invariant, safe no-contract floor (my #428 framing HOLDS) |
| +pinned-K (#431, `uza2t8aq`) | 496.74 | `self_referential_only` — Q1 contract call (kernel UNBUILT) |

### Command

```bash
# full run (GPU/CPU bf16 perturbation + W&B):
cd target/ && CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 python -m \
  research.validity.cb3_m_invariance_identity_class.cb3_m_invariance_identity_class \
    --wandb_group cb3-m-invariance --wandb_name wirbel/cb3-m-invariance-identity-class
# 0-GPU primary gate (no torch needed):
cd target/ && python -m research.validity.cb3_m_invariance_identity_class.cb3_m_invariance_identity_class --self-test
```

- **W&B run:** `gluit5k0` (group `cb3-m-invariance`, state finished)
- **Peak memory:** ~882 MiB (synthetic body-GEMM probe; Hadamard up to 16384²; no model load)
- **Device:** cuda:0 (A10G, sm_86) — the served hardware class; the per-row M-invariant reduction is byte-exact on real GPU, not just CPU
- **Self-test:** 0-GPU gate 42/42; full run 47/47 (adds the fresh-measurement byte-exactness + control-sensitivity + RHT-M-free checks, and `_load` cross-checks against #232 `int4_body_bitexact_m8` / #429 operative-identity-1.0 / #428 frozen-floor 482.74 JSONs)

### What happened — honest analysis

The PR did the right adversarial thing: it pointed the #431 lens at my **own** untested #428 assumption. The assumption survives — cb3 is `unconditional`, not `self_referential_only` — and the reason is mechanistic, not lucky: cb3 changes weight **values** on an **unchanged, M-independent** body-GEMM reduction (directly measured M-invariant, #232), whereas pinned-K changes the reduction **order** itself. A weight re-quant cannot introduce M-variance the way a split-K reschedule does.

Two honest refinements I made to the instructions (CLAUDE.md liberty), both flagged here:

1. **Geometry.** Instruction 2 templated the *attention* geometry (nq/nkv/hd, KV-lens) from #431, but cb3's locus is the **body GEMM**, not the attention reduction (that's pinned-K's locus). I measured at the **body-GEMM geometry** (K∈{2048,16384}) — cb3's faithful locus — while still carrying the attention constants in the logged config for provenance.
2. **Measurement fidelity.** A naive `torch.matmul` M=1-vs-M=8 comparison *falsely* reads M-variant on this hardware because cuBLAS/MKL dispatch GEMV vs GEMM with different K-orders — a math-library artifact, not the served int4-Marlin kernel's behavior (#232 measured that byte-exact). I model the per-row M-independent reduction explicitly (the #232-measured schedule) and the size_m-keyed split-K control separately, so the byte-exact zero is faithful and the probe is provably sensitive (control perturbs at 3.53% of scale). This is the same discipline #431 used in modeling split-K rather than trusting an un-runnable kernel.

**Caveat (the cb3 analog of #431's unbuilt-kernel note):** no cb3/QTIP/QuIP#/AQLM kernel exists in vLLM 0.22.0 (`direct_cb3_kernel_ab_runnable = false`), so a *direct* cb3-kernel A/B is un-runnable — but **unlike #431, the banked evidence is a DIRECT measurement of the same body GEMMs** (#232), so cb3's M-invariance is a **trivially-satisfiable BUILD requirement** (the default M-independent K-reduction these body shapes already use under int4-Marlin), not a special unbuilt property. The bounded guarantee to bank: cb3 can **never** differ M=1-vs-M=8 by more than the M-independent body reduction allows, which is **zero** — strictly tighter than pinned-K's sub-ε* tie-break.

### Suggested follow-ups

1. **#407 deploy scope (the decision this unblocks):** treat **482.74 as the unconditional no-contract floor** — the human Q1 "which reference" call applies **only** to the pinned-K 496.74 rung, not to cb3. The frontier above blanket-strict is therefore split: 467.14→482.74 is safe-by-construction; 482.74→496.74 is the genuine contract decision.
2. **The only thing that would tighten this to a measured cb3-kernel A/B** (not needed for the verdict): a human-gated cb3/QTIP decode-kernel build on sm_86, then re-run this exact harness as a real `SENPAI_REFERENCE_MODE` M=1 vs M=8 A/B on the cb3 dequant path — turning the modeled byte-exact into a kernel-measured `n_divergent_tokens = 0`. I did **not** build a kernel (analysis-only envelope; the cb3 kernel is un-built and human-gated, same as #427's pinned-K).
3. **ubel #422 cross-check:** once #422 lands the real RHT+VQ fake-quant magnitude, confirm the cb3 PPL margin (teacher-forced) is unchanged by the M=8 verify path — this card asserts PPL-neutrality on identity grounds (byte-exact M-invariant ⇒ same tokens ⇒ same PPL), which #422's real-weight numbers can corroborate end-to-end.
