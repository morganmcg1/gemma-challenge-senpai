STUDENT stark:
SENPAI-RESULT: {"analysis_only":true,"official_tps":0,"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["j3gjxxts"],"primary_metric":{"name":"aime2024_greedy_maj@1_transplant","value":0.033333},"test_metric":{"name":"aime2024_greedy_maj@1_transplant","value":0.033333},"collapse_locus":"BODY","cheap_fast_ship_exists":false,"aime_greedy_base":0.266667,"aime_greedy_osoi5_16k":0.033333,"aime_greedy_transplant":0.033333,"aime_pct_of_base":0.125,"recovered_fraction":0.0,"head_input_dim_compatible":true,"tied_head_is_base_head_byte_verified":true,"apples_to_apples":true,"ppl":2.404,"ppl_gate_pass":true,"transplant_tps_local_lowerbound":95.16,"peak_mem_gib":8.67}

## Results — lm_head transplant: the AIME collapse is the BAKED BODY, not the head

**One-line verdict:** Bolting the **full native base 262k BF16 head** onto the **osoi5-baked body** does **NOT** recover AIME (transplant **0.0333** vs osoi5-16k **0.0333** vs base **0.267**; recovered fraction **0.0%**). **`collapse_locus = BODY`**, **`cheap_fast_ship_exists = false`** — a re-bake-free head-only fix does **not** exist; only base-int4 (fern #535) or a full re-bake can restore reasoning.

### The decisive cell (AIME-2024 greedy maj@1, n=30, seed=1234 — same protocol as fern #531)

| substrate | AIME-greedy maj@1 | pct-of-base | source |
|---|---|---|---|
| base-body + base-262k-head | **0.2667** (8/30) | 100% | fern #531 `vcacv804` (banked) |
| osoi5-body + osoi5-16k-head | **0.0333** (1/30) | 12.5% | fern #531 `vcacv804` (banked) |
| **osoi5-body + base-262k-head (THIS PROBE)** | **0.0333** (1/30) | **12.5%** | **#536 `j3gjxxts`** |

- `delta_head_swap_vs_osoi5_16k` = **0.000** (2·se band ±0.093 → within noise): swapping the pruned 16k-int4 head for the full 262k BF16 head changed accuracy by **nothing**.
- `delta_transplant_to_base` = **+0.233** (band ±0.174 → outside noise): the transplant is **decisively still collapsed** vs base.
- `recovered_fraction` = **0.0** of the 0.233 full collapse.
- Extract-fail rate: base n/a · osoi5-16k **0.367** · transplant **0.267** — the full head made answers *modestly more parseable* (fewer un-extractable), but reasoning stayed broken. Per-problem, the transplant's maj answer **differs from osoi5-16k on 10/30** problems (so the base head is genuinely active and changing the logits — it just changed *which wrong answer*, not correctness).

### KEY OUTPUTS (required by the PR)
- **`head_input_dim_compatible` = TRUE.** osoi5 body emits a standard 2560-dim final hidden (`embed_tokens [262144,2560]`, hidden_size 2560; osoi5 `lm_head.weight_packed [16384,320]` = int4 in=2560). No `<2560` bottleneck inserted at bake → dimensionally transplantable.
  - **Load-bearing PR-premise correction:** the base `…qat-w4a16-ct` head is **NOT** int4-packed `[262144,320]`. It is **unquantized BF16 `[262144,2560]`** and **tied** (`tie_word_embeddings: True`, `lm_head` in the quant `ignore` list). So the transplanted head is the *strongest possible* head (full vocab + full precision) — the cleanest probe: if even this head fails on the osoi5 body, the BODY is definitively implicated.
- **`aime_greedy_transplant` = 0.0333**, `aime_pct_of_base` = **0.125**.
- **`collapse_locus` = BODY.**
- **`cheap_fast_ship_exists` = FALSE.**
- **Secondary (caveat loudly — local AWS A10G, exploratory, NOT an official number):**
  - `ppl` = **2.4040** (token-weighted, full 128-record official PPL set, 61,797 tokens) → gate ≤ 2.42 **PASS** (margin 0.016, `completed=128/128`). Teacher-forced, hardware-independent. **Note:** slightly *worse* than the surgical-357 ship's **2.3767** (same body, pruned-int4-16k head) — i.e. the *full-precision* base head gives marginally higher PPL on this body than the head it was baked with. That is a soft corroboration of **body↔head co-adaptation**: the bake drifted the body's hidden representation to its own baked head, so even the strictly-richer base head is slightly mismatched. PPL (local next-token fluency) is preserved while AIME (multi-step reasoning) collapses → the bake spared calibration but destroyed reasoning, and reasoning lives in the body.
  - `transplant_tps` = **95.16 tok/s** — **deep lower bound, NOT a ship number.** Single-stream local decode proxy (16×512) with the **full-vocab argmax tax** (FUSED_SPARSE_ARGMAX + 16k-prune OFF) **and spec OFF** (the mtp drafter + DIXIE/LOOPGRAPH fused-accept paths are 16k-vocab-keyed, so `SENPAI_REFERENCE_MODE=1`). ~4× under the surgical-357 ship ref (375.857 official) because the two big speed levers — the fast pruned-head argmax and spec — are both structurally incompatible with a raw 262k head. A head-optimized ship would need a *new* fast 262k-argmax kernel + a 262k-compatible drafter; neither exists.
  - `peak_mem` = **8.67 GiB** model weights + **9.65 GiB** KV-cache reserved (384,625 tokens, 93.9× concurrency at 4096 ctx) on the 22.5 GiB A10G at `GPU_MEMORY_UTILIZATION=0.90`. From server log.

### Transplant validity (why the 0.0333 is trustworthy, not a silent 16k fallback)
The transplant serves the osoi5 body but ties the output head to osoi5's **own** `embed_tokens`. I verified bit-exactly (sha256 over raw tensor bytes, `verify_tie_identity.py` → `tie_identity_verified.json`):
- (A) base `lm_head.weight` **==** base `embed_tokens.weight` (base internal tie) — sha `4b61614…704315`
- (B) osoi5 `embed_tokens.weight` **==** base `embed_tokens.weight` (bake left embed intact) — same sha
- ⇒ the tied head is **byte-exactly the base 262k head**. The **only** moved variable vs the osoi5-16k row is the head.
- Runtime proof it served 262k (not the stale 16k packed head): `[pck04] PCK04_KEEPSET not set — scatter INACTIVE`; generations are fluent full-vocab LaTeX/math (`\boxed{}`, `\begin{enumerate}`) impossible under a 16k-pruned vocab; 10/30 maj answers differ from osoi5-16k; extract-fail moved 0.367→0.267.

### What happened (honest analysis)
The model produces fluent, well-formatted solution *scaffolding* ("Step 1: convert the logs… Step 2: solve the ratios…") but never executes the math — it loops/repeats and emits garbage finals (e.g. gold 33 → 3, gold 23 → 2; `finish_reasons` = 16/30 hit the 3072-token cap). That is the signature of **degraded reasoning in the transformer body**, with the output head working correctly. The QAT bake — not the head prune — is what destroyed AIME reasoning. This is consistent with fern #531's `delta_prune(12k→16k)=0.000` and now isolates the previously-confounded `delta_bake` to the **body**, not the head.

### Decision impact
- **Do NOT spend a cluster slot on a head-only re-bake** — it cannot recover AIME (proven here at the strongest-possible head).
- The osoi5 body is **not** salvageable for quality by swapping the head. A quality-safe FAST ship requires fixing the **body**: base-int4 + fast kernels (fern #535's lane) or a full re-bake. This PR de-risks that contingency: the cheap path is closed.

### Reproduction (LOCAL, analysis-only — NO HF Job, NO --launch, NO submission)
```bash
# 1) compatibility + byte-identity of the tied head vs base head
python research/validity/lmhead_transplant_osoi5body/verify_tie_identity.py   # -> tie_identity_verified.json

# 2) build the transplant model dir (osoi5 body, tie head -> base 262k embed)
python research/validity/lmhead_transplant_osoi5body/build_transplant_dir.py   # -> /tmp/osoi5-transplant-tie

# 3) AIME-2024 greedy maj@1 n=30 seed=1234 on the transplant
research/validity/lmhead_transplant_osoi5body/run_transplant_aime.sh \
  research/validity/lmhead_transplant_osoi5body/full transplant-n30 --limit 30

# 4) head-vs-body verdict vs banked fern #531 rows + W&B log
uv run --no-sync python research/validity/lmhead_transplant_osoi5body/transplant_decompose.py \
  --transplant research/validity/lmhead_transplant_osoi5body/full/aime_transplant-n30.json \
  --osoi5-16k research/downstream_quality_aime/osoi5_16k_greedy_aime.json \
  --decompose research/downstream_quality_aime/aime_substrate_decompose.json \
  --out research/validity/lmhead_transplant_osoi5body/transplant_decompose.json \
  --wandb --wandb-name stark/lm-head-transplant-osoi5body --wandb-group head-transplant-osoi5body

# 5) caveated secondaries (PPL gate + local TPS proxy + peak mem)
uv run --no-sync python scripts/local_prevalidate.py \
  --submission submissions/transplant_osoi5_basehead \
  --venv-python /tmp/senpai-venvs/5f4c623f772358a2/bin/python --ppl-records 0
```

- **W&B run:** `j3gjxxts` (group `head-transplant-osoi5body`, `analysis_only=true`, `official_tps=0`)
- **Public evidence used:** fern #531 banked substrate decompose (`research/downstream_quality_aime/aime_substrate_decompose.json`, run `vcacv804`) for the base/osoi5-16k rows; ubel #511 (`pfu3vy7c`) MMLU-Pro/GPQA collapse context. Cross-link (not duplicated): fern #535 base-int4 lane.

### Suggested follow-ups
1. **Separate prune-vs-quant *within* the head is now moot for the verdict** (full BF16 262k head already fails) — no need.
2. **Confirm BODY-drift generalises** beyond AIME: run the cheaper MMLU-Pro (n≥200) / GSM8K (n≥100) greedy legs on the *same* transplant dir (`run_transplant_aime.sh` pattern, swap the eval harness) to show the body collapse is general reasoning, not AIME-specific. Optional — does not change the verdict.
3. **All quality-recovery effort should move to the body** (fern #535 base-int4 + fast kernels, or a re-bake with a reasoning-preserving recipe). Head work is a dead end for quality.
