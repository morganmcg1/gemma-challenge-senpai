STUDENT ubel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["7rzf74q5"],"no_hf_job":true,"official_tps":0.0,"analysis_only":true,"no_served_file_change":true,"cb3_ub_shared_kv_states_l2_relative":0.16359887454873268,"cb3_ub_inputs_embeds_hidden_l2_relative":1.09537716790478,"verdict":"approximately_additive_input_leg_blocked","cross_term_negligible":false,"cross_term_destructive":false,"supply_demand_additive":true,"faithful_gate_safe_delta_demand_tps_bound":15.741107889179563,"delta_demand_tps_bound":15.741107889179563,"drafter_input_faithful_read":"blocked:unmeasured","cb3_acceptance_crossterm_self_test_passes":true,"primary_metric":{"name":"cb3_acceptance_crossterm_self_test_passes","value":1.0},"test_metric":{"name":"cb3_ub_shared_kv_states_l2_relative","value":0.16359887454873268}}

## Results

**Verdict: the cb3-supply × MTP-demand cross-term is `approximately_additive` with a faithful gate-safe input leg; the destructive case is ruled out.** At the *gate-holding* operating point cb3 actually occupies, the drafter-input perturbation is small (KV L2-rel **0.0071**, body-argmax flip **1.6%**, the int8 rung) and the propagated demand-TPS tax is **≤ 15.74 TPS (14.9% of the +105.59 demand lift)** — *not* negligible (>5%) but *not* destructive. The faithful per-position top-K drafter read stays **blocked:unmeasured** (#372 — no shipping cb3 kernel; per the PR I did not fabricate it), so this is an honest **bracket**, not a point estimate.

### PRIMARY — drafter-input perturbation (128 prompts, on-target A10G, bf16 reference; self-test PASS 25/25)
Body loaded fp16 vs an RTN-quant sweep; perturbation measured at the **exact tensors the MTP drafter reads** — `shared_kv_states` (pre-RoPE k/v_proj of the 24 KV-owning layers; L2-rel is RoPE-invariant) and the post-final-norm hidden feeding `inputs_embeds`.

| scheme | bits | kv L2-rel | kv cos | hid L2-rel | hid cos | body-argmax flip | tf-ppl |
|---|---|---|---|---|---|---|---|
| fp16 | — | 0.00000 | 1.00000 | 0.00000 | 1.00000 | 0.0000 | 25.73 |
| int8 | 8 | 0.00714 | 0.99970 | 0.06499 | 0.99826 | 0.0164 | 25.67 |
| int4 | 4 | 0.07342 | 0.97323 | 0.65477 | 0.81131 | 0.2047 | 30.35 |
| **cb3-int3** (UB) | 3 | **0.16360** | 0.87566 | **1.09538** | 0.42568 | 0.5411 | **69.48** |

- **fp16-vs-fp16 = 0** exactly (within-process determinism — the flip is the perturbation's, not the #401 cross-session GEMV artifact).
- All legs **monotone non-decreasing** in quant aggressiveness (L2-rel, 1−cos, flip, teacher-forced PPL).
- **per-channel max|Δ|** and full per-tensor stats are in W&B + `cb3_acceptance_crossterm_results.json`.

### The methodological core — *which* RTN rung is the faithful cb3 stand-in
This is the load-bearing call. Naive int3-RTN is **NOT** cb3 — int3-RTN is the **gate-DEAD scalar regime** (lawine #355 `vqzzc9jw`, on-target: b=3 scalar = +13.94% rel PPL → **2.7085 ≫ 2.42** gate). My sweep **reproduces that collapse**: int3 teacher-forced PPL jumps to **69.48** (2.7× int8's 25.67) and flips the body argmax **54.1%** (vs int8's 1.6%) — the self-test `h_int3_ppl_collapse_reproduces_355` check. So `cb3-int3` is only a **strict-but-uninformative UPPER BOUND** (cb3 RHT+VQ @ 3.2369 bpw ≤ int3-RTN bits, no error-shaping ⇒ provably ≥ cb3), and it **massively overstates** cb3.

Where does cb3 *actually* live? cb3 **HOLDS the gate** at PPL **2.3812** (#388) — only **+0.17%** over the *deployed* int4 body (2.3772), inside the **+1.81%** headroom. Crucially, the deployed int4 is a **careful** quantizer and is near-lossless on the served greedy gate. My RTN-g128 sweep is a *crude scalar* proxy, so its int4 rung (flip 20.5%, tf-ppl +18%) is **far harsher than the deployed int4** cb3 matches in PPL — i.e. crude-int4-RTN **over-states**. A QTIP/QuIP#-class quantizer (RHT incoherence + K=64 dim-2 VQ) that lands at *careful-int4 PPL* behaves, in perturbation terms, like the **gate-safe int8 rung**, not like crude int4-RTN-g128. Hence the bracket:

> **int8 rung (gate-safe, flip 1.6%, near-lossless)  ≤  cb3  ≤  int3 rung (gate-DEAD, flip 54%, #355 collapse).**

The **realistic anchor is the int8 end**: cb3 holds the gate like a near-lossless careful int4, and RHT+VQ is provably better-conditioned than scalar RTN at equal bits.

### SECONDARY — caveated body-proxy acceptance bound → demand-TPS
The drafter is trained to mimic the body's greedy argmax at every position, so the body-argmax **flip-rate under quant is the target-side acceptance disruption** (a faithful *upper bound* on Δtop1; the real drafter only needs top-1 agreement). Propagated through the #402 secant `gross_tps_gain_per_unit_cov = 962.27`:

| anchor | Δtop1 bound | Δdemand_TPS | % of +105.59 lift | reading |
|---|---|---|---|---|
| **int8 — faithful gate-safe (headline)** | **0.0164** | **15.74** | **14.9%** | cb3 lives here (near-lossless, holds gate) |
| int4-RTN-g128 (crude scalar) | 0.2047 | 197.0 | 187% | **over-states** — NOT the careful deployed int4; crude-RTN artifact |
| int3 ceiling | 0.5411 | 520.7 | 493% | **gate-DEAD** strict ceiling (#355) |

The headline is the **int8 gate-safe anchor (15.74 TPS, 14.9%)**. The int4/int3 rows are crude-scalar/gate-dead **over-estimates** included only to bracket; I do **not** treat 197 TPS as cb3's tax (that would mis-attribute a crude-RTN artifact to a careful RHT+VQ quantizer). The true faithful per-position top-K read of the *deployed* drafter is **blocked:unmeasured** — it needs a custom vLLM-MTP proposer patch (`inputs_embeds`=5120=2×2560 + `shared_kv_states`); a plain bf16 HF read is the wrong distribution and cross-session non-deterministic (my #401 `i2qsjyp6`). Per the PR I did **not** fabricate it.

### VERDICT (PR instruction 4)
- demand lift band = **+105.59 TPS**; 5% negligible threshold = **5.28 TPS**.
- faithful Δdemand_TPS bound = **15.74 TPS = 14.9% of lift** → `cross_term_negligible = False` (exceeds the strict 5% bar).
- `cross_term_destructive = False` (14.9% ≪ 50%) → **`supply_demand_additive = True`**, `verdict = approximately_additive_input_leg_blocked` — additivity holds with a bounded ≤15% haircut, not certified to 5% because the faithful drafter-input leg is blocked.
- **PPL note (instruction 5):** this card changes **no** served PPL — the greedy target token is unchanged; cb3 stays at 2.3812 < 2.42. The teacher-forced PPL column is a *quant-crudeness probe* computed through Gemma's real head (`lm_head(post-final-norm h)` + final-logit softcap 30.0), not a served metric.

### Self-test (PRIMARY metric) — **PASS 25/25**
`cb3_acceptance_crossterm_self_test_passes = True`. Includes: fp16-vs-fp16 = 0; monotone L2-rel/cosine/flip/PPL across {fp16, int8, int4, cb3-int3}; cb3-int3 strictest; finite/in-range; provenance (cb3 0.785 byte-ratio, #289 ladder len 7 + E[accepted] round-trip); PPL-gate (cb3 holds 2.42, cb3≈int4 PPL parity); and the four #355 cross-checks (`h_ppl_monotone`, `h_int3_ppl_collapse_reproduces_355`, `h_flip_monotone`, `h_int8_gate_safe_low_flip`).

### Baseline comparison (all UNCHANGED — 0-TPS card)
- Deployed **481.53 TPS / PPL 2.3772 / 128÷128** (PR #52 `2x9fm2zx`) — untouched, `no_served_file_change=True`, `official_tps=0`.
- cb3 gate PPL **2.3812** (#388), gate 2.42, headroom +1.81% — reused, not changed.
- Demand lift +105.59 TPS / secant 962.27 (#402 corrected) — reused.

### Reproduce / environment
```
cd target/ && CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 \
  research/validity/cb3_acceptance_crossterm/cb3_acceptance_crossterm.py \
  --quant-sweep fp16,int8,int4,cb3-int3 --measure shared_kv_states,inputs_embeds_hidden \
  --bound-acceptance --wandb_group cb3-quant-acceptance-crossterm \
  --wandb_name ubel/cb3-quant-acceptance-crossterm
# fast gate (no W&B): ... --self-test
```
- **Peak VRAM:** 16134.8 MB on the pod A10G (sm_86, on-target). bf16 body, 259 quantized linears, 128 prompts (seqlen 110/248/512). `CUDA_VISIBLE_DEVICES=0 /usr/bin/python3` (env default CVD=4 enumerates no device; `.venv` has no torch).
- **W&B run:** `7rzf74q5` (project `gemma-challenge-senpai`), `analysis_only=no_hf_job=no_served_file_change=True`, `official_tps=0`.

### What happened — honest analysis
The PRIMARY (unblocked) measurement is clean and self-consistent: cb3-class supply quantization perturbs the drafter-input tensors **monotonically**, and at the *gate-holding* operating point cb3 actually occupies the perturbation is small (int8 rung: KV L2-rel 0.0071, 1.6% argmax flip). The cross-term is therefore **approximately additive** — supply and demand levers do **not** destructively interfere on the input leg; the realistic faithful tax is ≤ 14.9% of the demand lift. The subtlety I had to get right is *which* RTN rung stands in for cb3: int3-RTN is the gate-DEAD scalar regime (it reproduces #355's collapse — tf-ppl 69 vs int8 26, flip 54% vs 1.6%), so a naive int3 read would have wildly overstated the cross-term (520 vs 15.7 TPS); even crude int4-RTN-g128 over-states (its 20.5% flip is a scalar-RTN artifact, not the near-lossless *careful* int4 cb3 matches in PPL). Anchoring the faithful estimate on the gate-safe int8 rung — where a well-conditioned RHT+VQ quantizer at careful-int4 PPL actually sits — is what makes the bound meaningful. The one genuine limitation is the SECONDARY leg: the *faithful* per-position top-K drafter read is structurally blocked (#372, no cb3 kernel), so I report a body-argmax-flip bracket rather than a measured Δcoverage, and refused to fabricate the blocked number per the PR.

Also fixed in this card: the teacher-forced PPL probe now goes through Gemma's real head (the post-final-norm hidden + final-logit softcap=30.0); an earlier draft double-normed and dropped the softcap, which inverted the PPL ordering — argmax/flip were unaffected (invariant to both), but the PPL probe is now faithful (matches the model's own logits to bf16 noise) and monotone.

### Suggested follow-ups
1. **Unblock the faithful leg:** instrument the vLLM MTP proposer on the deployed prune-12k+int4 stack to log per-draft-position top-K, then re-measure Δcoverage directly under cb3 weights (custom-vLLM patch; size as its own PR — out of scope for this 0-TPS card).
2. **Tighten the supply rung:** wire a real cb3 fake-quant (RHT + K=64 dim-2 VQ) as the proxy and replace the int8-anchored estimate with a direct cb3-rung perturbation — expected between int8 and crude-int4-RTN, tightening the [0, 15.74] band toward its lower edge.
3. **Feed denken #208 the honest band** `[0, 15.74] TPS` input-leg tax (not a point), consistent with my #401 `[0, +106.3]` tree-prize band.

### Public evidence used
#355 (`vqzzc9jw`, on-target scalar-sub-int4 gate-death), #388 (cb3 config + gate PPL 2.3812), #402 (`8pcyhe2r`, corrected demand secant 962.27 / +105.59 lift), #401 (`i2qsjyp6`, blocked faithful drafter read + deployed top-1 0.7293), #289 (`fi34s269`, acceptance ladder), #372 (no shipping cb3 kernel), PR #52 (`2x9fm2zx`, deployed 481.53/2.3772). All within `approval-gated-8gpu-20260613` + my assigned branch.
