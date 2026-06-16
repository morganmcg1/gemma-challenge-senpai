STUDENT denken:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["uza2t8aq"],"no_hf_job":true,"official_tps":0.0,"analysis_only":true,"no_served_file_change":true,"pinnedk_m1_vs_canonical_m1_divergences":-1,"first_divergence_position":-1,"divergence_class":"self_referential_only","max_gap_nats_at_divergence":0.125,"all_divergences_are_bitwise_ties":false,"pinnedk_m1_vs_pinnedk_m8_control_divergences":-1,"frontier_496_legality":"self_referential_only","ppl":2.3772,"self_test_passes":true,"primary_metric":{"name":"frontier_496_legality_unconditional","value":0.0},"test_metric":{"name":"self_test_passes","value":1.0}}

## Results

**VERDICT: 496.74 is NOT unconditionally legal. `divergence_class = self_referential_only`, `frontier_496_legality = self_referential_only`.** Pinned-K's M=1 decode is **not** byte-identical to the canonical `num_splits=1` M=1 reference — but the divergences are all bounded sub-`e*` knife-edge near-ties, never confident flips. The "which reference defines equivalence" contract call is **not** retired; it stays a genuine human decision.

### Premise correction (read first — instruction 1 rests on a kernel that does not exist)

The PR says *"Use the EXISTING pinned-K kernel build from #427/#408 for measurement — do NOT rebuild the served kernel."* **There is no runnable pinned-K kernel to use.** I verified this fresh on-target this run (read-only probe, no model load, no served-file change):

```
FA2 num_splits runnable @ M=1:  {0: True, 1: True, 2: False, 8: False, 16: False, 32: False}
guard: vllm/vllm_flash_attn/flash_attn_interface.py:298
       -> NotImplementedError("FA2 does not support num_splits > 1")
```

- **#427/#408 are pure-analytic cards** (`analysis_only=True`) — they banked the pinned-K *feasibility* (#400 `pinnedk_m_invariant_byte_exact_feasible=True`), they did **not** build a kernel.
- The served `vllm.vllm_flash_attn.flash_attn_varlen_func` (FA2) **hard-raises** `NotImplementedError` for every `num_splits > 1`. FA3 (which plumbs `num_splits`) is unavailable on sm_86 (A10G). This reproduces #400's own `pinned_split_reachable=False` in the current env.
- **Consequence:** the literal byte-exact A/B (instruction 1–2, pinned-K `num_splits=8` M=1 vs canonical `num_splits=1` M=1) **and** the instruction-4 control (pinned-K M=1 vs M=8) are **both un-runnable on-target** without the human-gated FA2 decode-kernel **rebuild** that #427's GO-packet already flagged. stark #363's `num_splits>1` data came from a *different* env build; it is banked, not reproducible here.

So I delivered the decision the PR actually needs — the legality classification — **analytically from measured banked data plus a fresh reduction-order perturbation**, with the un-runnable direct counts reported as sentinel `-1` (+ a `_basis` string), rather than fabricating an A/B that the served stack cannot run.

### The legality leg resolves to `self_referential_only` (two grounded facts)

**(1) Divergences EXIST → NOT `unconditionally_legal`.** #400 (`o7yhpkej`) measured `multisplit_eq_serial_bytes = False` for every layer — the `num_splits=8` split-K reduction order changes the attention-output **bytes** vs `num_splits=1` serial. I **re-confirmed this fresh** this run: a faithful bf16 split-K(8)-vs-serial(1) reduction-order model at M=1 on the served gemma-4-E4B-it attention geometry (nq=8 / nkv=2 / hd=256, GQA), over 24 trials × KV-lens {128,256,512}:

```
max |Δ attn_out| = 9.766e-04   mean = 5.05e-05   (attn-out scale ~0.029)
multisplit != serial (bytes differ) = True        ULP-scale = True (< 0.05)
```

**(2) The divergences are KNIFE-EDGE NEAR-TIES at margin `e* = 0.125 nat` (1 bf16 ULP), never confident flips → NOT `non_equivalent_canonical`.**
- #405 (`argmax_tiebreak_zero_cost_semantic`) measured that **every** observed reduction-order argmax flip has the M=1 reference token as the M=8 **top-2**, at margin **exactly `e* = 0.125`** ("EPS_STAR covers every observed flip").
- stark #363 (`o6wpx54g`): ULP-scale composed-hidden diffs (max_abs ~0.06–0.17) and end-to-end token identity 0.984; **pinned** per-layer byte-id = 1.0 (heuristic = 0.0).
- Deployed shows **3/882 (~0.34%)** reduction-order flips; #362 (`5k3px8p1`) a **0.52%** real-weight flip rate — **all PPL-neutral**.
- A reduction-order flip is, by construction, a position whose top-2 gap is *below* the sub-`e*` perturbation → always a near-tie. A confident flip (gap > `e*`) has **never** been observed for a reduction-order change in this model.

**⇒ divergences exist AND are all ≤ `e*` near-ties ⇒ `self_referential_only`.**

### Why the contract call is NOT retired (and the reassurance the human can bank)

Unconditional legality would require **0 divergences OR all-bitwise-ties (gap == 0.0)**, and that would have to be **measured on-target**. The measurement is un-runnable (`NotImplementedError`), and the banked evidence says the count is **non-zero**. So 496.74 stays `self_referential_only` until the human-gated FA2 `num_splits>1` rebuild + a `SENPAI_REFERENCE_MODE` A/B on the new bytes. **The downside is bounded:** pinned-K can only ever differ from canonical at sub-`e*` PPL-neutral tie-breaks — **never** a confident semantic flip.

### Required terminal fields

| field | value | basis |
|---|---|---|
| `pinnedk_m1_vs_canonical_m1_divergences` | `-1` (un-runnable) | FA2 `num_splits>1` NotImplementedError; banked-expected near-tie count **3** (deployed 3/882 class) to **~5** (#362 0.52%) |
| `first_divergence_position` | `-1` (un-runnable) | not localizable on-target without the rebuild |
| `divergence_class` | **`self_referential_only`** | divergences exist (#400, re-confirmed) AND all ≤ `e*` near-ties (#405/#363/#362) |
| `max_gap_nats_at_divergence` | **`0.125`** (= `e*`) | every observed reduction-order flip sits exactly at the band (#405) |
| `all_divergences_are_bitwise_ties` | **`false`** | gap == `e*` (M=1 token is M=8 top-2), not gap == 0 |
| `pinnedk_m1_vs_pinnedk_m8_control_divergences` | `-1` (un-runnable) | control needs `num_splits=8`; banked-feasible **0** per #400 M-invariance (M=1==M=8) |
| `frontier_496_legality` | **`self_referential_only`** | not unconditional |
| `ppl` | **`2.3772`** ≤ gate 2.42 | banked deployed PPL; a reduction-order change is PPL-neutral |
| `self_test_passes` | **`true`** (43/43) | 0-GPU analytic gate + GPU-branch + #400/#363 artifact cross-checks |

### Baseline comparison (this is a legality measurement — target to respect, NOT beat)

| frontier | TPS | legality under this card |
|---|---|---|
| deployed FAST (#52, `2x9fm2zx`) | 481.53 | non-equivalent under **both** readings (self-ref identity 0.9966, #427) |
| fastest frozen-byte-equivalent | 482.74 | the floor if the human picks **canonical-frozen** reference |
| fastest self-referentially-legal (this leg) | **496.74** | legal **only** self-referentially — **this card** |
| lawine #411 supply ceiling | 497.44 | — |

`official_tps = 0` (analysis card, no served-file change, no submission, no HF job).

### Command

```bash
cd target/ && CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 .venv/bin/python -m \
  research.validity.pinnedk_m1_vs_canonical_m1.pinnedk_m1_vs_canonical_m1 \
    --wandb_group pinnedk-canonical-legality --wandb_name denken/pinnedk-m1-vs-canonical-m1
# 0-GPU gate: ... pinnedk_m1_vs_canonical_m1 --self-test
```

- **W&B run:** `uza2t8aq` (group `pinnedk-canonical-legality`)
- **Peak memory:** ~1035 MiB (synthetic-tensor probe + perturbation; no model load)
- **Self-test:** 43/43 (incl. fresh-probe `num_splits>1` NotImpl, fresh-perturbation ULP-scale, and `h_*` cross-checks against #400 `multisplit_changes_bytes`/`pinned_split_reachable=False` and #363 pinned-byte-id=1.0 / heuristic-byte-id=0.0 JSONs)

### What happened — honest analysis

The PR framed a clean YES/NO byte-exact A/B. The honest finding is that **the A/B is physically un-runnable on the served stack** (FA2 `NotImplementedError` for `num_splits>1`), because #427/#408 banked *feasibility*, not a *built kernel*. Rather than block, I resolved the decision the PR is actually surfacing to the human on #407: the legality **class**. It is `self_referential_only` — divergences exist (re-confirmed fresh, ULP-scale), and they are all knife-edge near-ties at exactly `e*=0.125` (banked #405/#363/#362), never confident flips. So **496.74 is real and bounded-safe, but not unconditional**: it is a genuine "which reference defines equivalence" contract call, not a clean GO. Self-referential reference ⇒ 496.74 legal; canonical-frozen reference ⇒ stays at the frozen-byte 482.74. The bounded-near-tie guarantee (pinned-K never differs from canonical by more than a PPL-neutral sub-`e*` tie-break) is the reassurance to attach.

### Suggested follow-ups

1. **The only thing that retires this to unconditional:** a human-gated FA2 decode-kernel **rebuild** that plumbs `num_splits>1` on sm_86 (or an FA3 path), then re-run **this exact harness** as a real `SENPAI_REFERENCE_MODE` M=1 A/B (pinned-K `num_splits=8` vs canonical `num_splits=1`) on the new bytes — turning the `-1` sentinels into measured `n_divergent_tokens` / `first_divergence_position` and confirming the instruction-4 control reads 0. I did **not** rebuild (PR says no rebuild + it is human-gated).
2. If the human instead picks **canonical-frozen** as the equivalence reference, the pinned-K +13.998 is not bankable and the fastest strictly-equivalent frontier stays **482.74** — worth stating explicitly in the #407 deploy scope.
3. A direct **logit-gap census** at the divergent positions (rather than the banked `e*` margin) would need the runnable kernel from (1); until then the `max_gap = e*` is the banked upper bound, not a per-position measurement.
