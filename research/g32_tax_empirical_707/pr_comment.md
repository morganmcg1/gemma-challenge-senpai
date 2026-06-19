STUDENT land:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"analysis_only":true,"official_tps":0,"fires":false,"no_hf_job":1,"wandb_run_ids":["8jn9ofx7"],"primary_metric":{"name":"g32_full_tax_frac","value":0.03877},"test_metric":{"name":"selective_g32_tax_proj","value":0.000523},"verdict":"G32_TAX_NEGLIGIBLE"}

## Results — g32 op-bench tax: empirical upper bound on the int4-body recovery speed cost

**Verdict: `G32_TAX_NEGLIGIBLE`** — the selective-g32-on-48 recovery (ubel #700's 1.35% subset) is **unconditionally speed-free** (projected **0.066 TPS** on the locked rung, vs the +10 margin). Honest rider: this is driven by the **1.35% subset dilution (74×)**, NOT by g32 being cheap per se — **full** g32 costs ~**4.90 TPS** (material at full scale). The linear coefficient lets denken price any subset size.

### Headline numbers (W&B [`8jn9ofx7`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/8jn9ofx7), group `g32-tax-empirical-land`)

| metric | value | meaning |
|---|---|---|
| **`g32_full_tax_frac`** (primary) | **3.877 %** (= 4.900 TPS on 126.378) | UPPER BOUND: full-model g32 vs g128 |
| **`selective_g32_tax_proj`** (test) | **0.0523 %** (= 0.066 TPS on rung) | selective-on-48 (1.35%) — denken's Pareto anchor |
| `g32_per_module_tax` (coefficient) | 3.877 % per 100% upgraded | the linear slope; multiply by any upgraded-param-fraction |
| `tps_g128` (anchor) | 126.75 | op-bench AR rung (#697/#642); cross-checked below |
| `tps_g32_full` (composed) | 121.836 | = 1000/(1000/126.75 + Δ_body) |
| `g32_body_tax_frac` | 7.820 % | body-matmul-relative tax (denominator-free) |
| `paired_delta_us` | **318.23 ± 0.19 us/token** | g32−g128 body forward, M=1 (CI ±0.06%) |
| `body_us_g128` / `body_us_g32` | 4069.6 / 4387.8 us | full 37-layer body forward, M=1 |
| BW-bound ceiling | +9.091 % | scale_bytes/weight_bytes = 4/group_size |
| `body_frac_of_anchor` | 51.6 % | measured body share (cross-checks the 126.75 anchor vs lawine #591's ~44%) |

**Per-component g32 tax (M=1):** qkv_proj 5.05 %, o_proj 5.46 %, gate_up_proj 8.21 %, down_proj 8.20 %. The large bandwidth-bound MLP matmuls sit near the +9.09 % BW ceiling; the smaller attention projections are launch/latency-bound at M=1 so the extra scale reads cost less. The body aggregate (7.82 %) is weighted toward the MLP matmuls that dominate the body bytes.

### Method — why a microbench, not a full-model load (load-bearing, disclosed)

The g32 tax lives **entirely** in the body matmuls: g32 reads **4× the per-group int4 scales** of g128 (`scale_bytes/weight_bytes = 4/group_size`, a shape-independent **+9.09 %** weight+scale traffic). lm_head, embeddings, attention, KV, norms and activations are **byte-identical** between a g128-body and a g32-body model — so the body-matmul delta **is** the complete per-token tax.

A full-model g32 build is **disk-infeasible here**: the locked g128 build is 9.7 GB, the shared overlay has ~10 GB free against the ~8 GB floor (two 9.7 GB builds can't coexist), and the bf16 `qat_unq` source for `build_quant.py` is not on the pod. The only on-disk g32 (official `google/gemma-4-E4B-it-qat-w4a16-ct`, `submissions/int4_qat`) carries a **tied bf16 lm_head**, so an official-g32-vs-locked-g128 full op-bench would be confounded by ~1 GB/token of head traffic, not the body group size.

So the **kernel microbench is both the feasible primary AND the cleanest isolation** of the quantization-group delta (zero head/attention confound). It drives the **exact served kernel** — `apply_gptq_marlin_linear`, the call deployed `GPTQMarlinLinearMethod.apply` makes — reusing stark #602's `int4_body_gemv_bw_saturation` apparatus. Achieved DRAM time is value-independent (shape/dtype/group_size/layout only), so random weights at the served fused shapes faithfully reproduce the deployed kernel timing; greedy/PPL are pinned by construction (no served-file change).

**Apples-to-apples discipline (#697 standard):** same harness, same seed (707), same fused body shapes, **paired per round** — every round times the g128 AND g32 body forward back-to-back (15 rounds × 2 interleave orders), L2-cold CUDA-graph replay (n_distinct=8 cold weights, working set ≫ A10G 6 MiB L2), median-of-rounds with 95% CI. M=1 is the AR single-stream geometry **and the conservative worst case** — at higher M (spec verify, M=K) the scale reads amortize over more activation columns, so the realized tax is *smaller*.

**Composition (transparent):** the body delta adds directly to the per-token decode time; the denominator is anchored on land's established op-bench AR rung (`tps_g128 = 126.75`, #697/#642; official ≈ op-bench, #697). The measured body fraction (51.6 %) cross-checks that anchor (≈ lawine #591's ~44 % body share).

### Selective-subset projection (hand-off to denken, group `recovery-speed-pareto-denken`)

`g32_per_module_tax = g32_full_tax_frac / 1.0 = 3.877 %` per 100 % of body params upgraded. For ubel #700's **48-module / 1.35 %** subset:

```
selective_g32_tax_proj ≈ g32_per_module_tax × 0.0135 = 0.0523 % = 0.066 TPS on the 126.378 rung
```

The linear projection is **mechanistically sound** for this lever: g32 scale traffic ∝ parameter count, so upgrading 1.35 % of body params adds ~1.35 % of the body's g32-vs-g128 scale delta. **Caveat (flagged for denken's band):** the *true* selective tax can differ from the linear projection via kernel-occupancy effects (the per-component table shows the per-shape tax ranges 5.05–8.21 %, so a subset concentrated in attention projections would tax *less* than one concentrated in MLP). denken's Pareto carries that band; the coefficient lets him price any subset size, e.g. ~20 % of modules → ~0.78 % (~1 TPS), ~50 % → ~2.5 TPS, full → ~4.9 TPS.

### Comparison vs baseline

- **Locked anchor** `int4_g128_lmhead` (PR #4): official 126.378, +10 bar 136.378, AR-rung op-bench 126.75. **Untouched** (no served-file change, no submission).
- g32-on-the-recovery-subset would cost **0.066 TPS** — i.e. it leaves **+10 headroom essentially intact** (it consumes 0.66 % of the 10-TPS margin).

### Exact commands

```bash
# measurement (vLLM venv has the Marlin kernel; CUDA_VISIBLE_DEVICES=0 = local A10G)
cd /workspace/senpai/target
CUDA_VISIBLE_DEVICES=0 /tmp/senpai-venvs/5f4c623f772358a2/bin/python \
  -m research.g32_tax_empirical_707.g32_tax_opbench \
  --iters 50 --warmup 40 --rounds 15 --n-distinct 8 --m 1 \
  --wandb_group g32-tax-empirical-land --wandb_name land/g32-tax-empirical

# W&B logged via a standalone script (the vLLM venv lacks wandb, and the repo-root
# wandb/ data dir shadows the package under `-m`; running the logger as a FILE with
# the .venv python sidesteps both):
/workspace/senpai/target/.venv/bin/python \
  research/g32_tax_empirical_707/log_wandb.py \
  --results research/g32_tax_empirical_707/g32_tax_opbench_results.json
```

### Resources

- **W&B run:** `8jn9ofx7` (group `g32-tax-empirical-land`, state finished).
- **Peak VRAM:** 1.49 GB allocated / 1.82 GB reserved (no model load — Marlin body weights only).
- **Disk:** added **0** bulky build (GPU random weights only, freed); overlay ~9.9 GB free, above the ~8 GB floor — no ping needed. Artifacts are KB-scale (`research/g32_tax_empirical_707/`: harness + results JSON + log).
- **Guards (W&B summary scalars):** `analysis_only=True`, `official_tps=0`, `no_hf_job=1`, `fires=0`, `no_served_file_change=True`. NOT a fire, NOT an approval trigger; the served file and the locked rung are untouched.

### What happened

The body M=1 Marlin W4A16 GEMV is **largely scale-bandwidth-bound**: the measured body tax (7.82 %) sits just below the analytical +9.09 % weight+scale BW ceiling, confirming the extra scales are mostly *read*, not hidden behind compute (the per-shape spread shows the big MLP matmuls hit the ceiling while the smaller attention projections, more launch-bound at M=1, tax less). Diluted across the per-token decode (body ≈ 52 %), full-model g32 costs **3.88 % / 4.90 TPS** — **material at full scale** (≈ half the +10 margin). But the recovery upgrades only **1.35 %** of body modules, diluting the tax **74×** to **0.066 TPS**, which is decisively speed-free. The decision-forcing answer is: **the selective-g32-on-48 recovery removes the speed objection from the int4-body quality-recovery program** — the fire stays separately quality-blocked (int4-body AIME 0.3467 < 0.420).

### Suggested follow-ups

1. **denken's Pareto:** anchor the speed axis on `g32_per_module_tax = 3.877 %/100%` (this measurement) rather than a first-principles guess; the selective-on-48 point is 0.066 TPS.
2. **Exact-subset tax (tightens the band):** if ubel #700's 48-module list is available cross-read, re-run the per-component microbench restricted to those exact (out,in) shapes for the *literal* selective tax instead of the linear projection — removes the kernel-occupancy band (cheap, same harness).
3. **Spec-stack realization:** if the recovery is ever evaluated on a spec-decode serve rather than the AR rung, the M=1 number here is a *conservative upper bound* — a quick M=K (verify-geometry) microbench would quantify how much smaller the realized tax is.

### Public evidence used

- Leaderboard / challenge state (digest, `as=senpai`): the int4 QAT W4A16 lane (`int4-qat`, `submissions/int4_qat/manifest.json`: `google/gemma-4-E4B-it-qat-w4a16-ct`, **group_size 32**, Marlin int4, "~95 TPS / PPL ~2.01 leader") confirms **g32 is the official/deployed Marlin grouping** — so this g32-vs-g128 comparison is between two real, supported Marlin configs, not a synthetic one. Our locked `int4_g128_lmhead` is the g128 re-quant of that same lane.
- Cross-reads (#666-authorized): land #684/#697/#642 (op-bench AR rung 126.75, official ≈ op-bench), ubel #700 (48-module / 1.35 % impact-energy subset), stark #602 (`int4_body_gemv_bw_saturation` apparatus reused). No out-of-scope branch read.
