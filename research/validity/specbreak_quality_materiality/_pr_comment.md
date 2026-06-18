STUDENT wirbel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"analysis_only":true,"verdict":"SPEC_BREAK_QUALITY_BORDERLINE","wandb_run_ids":["ezvgx3et","kwu5bv5k","958vaw51","e4lmv768","btka69v0"],"primary_metric":{"name":"specbreak_worst_retention","value":0.9570},"test_metric":{"name":"official_tps","value":0}}

## Results — Spec-break quality-materiality census (body-matched int4 AR vs SPEC)

**Headline: the strict-#319 break is MASSIVE at the token level (51.70% break-rate, 82.8% of sequences diverge) yet costs ~0 measurable benchmark points.** Across all four #515 benchmarks the body-matched spec-vs-AR difference is statistically indistinguishable from noise (`statistical_verdict = NO_SIGNIFICANT_DEGRADATION`; every McNemar p ≥ 0.62, every retention CI spans 1.0). The break is a *symmetric near-tie reshuffle*: it churns individual answers heavily but nets ~zero directional quality effect.

`analysis_only=1`, `official_tps=0`, `fires=0`. No HF Job, no `/v1/jobs:run`, no served-file change. Local single-A10G only.

### Isolation (validity)
Both arms serve the **same int4 body** (`int4_g128_lmhead`, the strict-#319 ship anchor / PR #4) on the **same engine** (vLLM dev307, `VLLM_BATCH_INVARIANT=1`, native sampler), via `submissions/int4_mtp_batchinv/serve.py`. The **only** removed variable is speculation:
- **AR arm** — `NUM_SPECULATIVE_TOKENS=0` → pure M=1 autoregressive greedy (this *is* the exact-greedy reference strict #319 is defined against).
- **SPEC arm** — `NUM_SPECULATIVE_TOKENS=6`, the wirbel #671 ~170-band QAT drafter (`/tmp/qat-assistant`).

So spec−AR isolates the int4-Marlin verify-width break (kanna #673) from the int4 body's own pre-existing quality gap. Cross-check: the SPEC arm reproduces the published DEV307 spec panel within noise (mmlu 0.690 vs 0.664, gsm8k 0.930 vs 0.928, gpqa 0.4495 vs 0.4764; aime 0.333 vs the *refuted* 0.400 extended-thinking number).

### 1. The break IS active (mechanism gate) — token-break probe
Canonical strict-#319 config (128 sharegpt × 512 greedy, `ignore_eos`, `decode_outputs.py`, seed 1), spec k=6 vs ar k=0:

| metric | value |
|---|---|
| **token break-rate** | **51.70%** (33883/65536), CI95 [51.32, 52.08]% |
| sequence divergence | 82.81% (106/128), CI95 [75.35, 88.37]% |
| median first-break position | 140 / 512 |
| `break_present` | **True** |

This decisively fails strict byte-exact #319 and is consistent with the #607 census (~47% divergent). Mechanism: the intrinsic per-step flip is tiny (#616: ~0.43%, 100% rescuable under τ=0.3nat), but greedy autoregression *cascades* a single early flip (median onset pos 140) so ~half of all downstream positions differ. **The break is large; the question is what it costs — answered below.**

### 2. The break costs ~0 quality (the card's finding) — body-matched panel
All legs greedy/temp=0, `min_tokens=8` EOS-guard (wirbel #541), gb6144 budget, conc=16:

| leg | SPEC | AR | retention (spec/ar) | retention CI95 | McNemar p | noise-consistent |
|---|---|---|---|---|---|---|
| gsm8k (n=300) | 0.9300 | 0.9367 | 0.9929 | [0.951, 1.036] | 0.625 | ✅ |
| mmlu_pro (n=300) | 0.6900 | 0.7000 | 0.9857 | [0.887, 1.096] | 0.701 | ✅ |
| **gpqa_diamond (n=198)** | 0.4495 | 0.4697 | **0.9570** | [0.773, 1.185] | 0.627 | ✅ |
| aime (n=60) | 0.3333 | 0.3333 | 1.0000 | [0.603, 1.659] | 1.000 | ✅ |

- **`specbreak_worst_retention = 0.9570`** (gpqa_diamond) → mechanically **`SPEC_BREAK_QUALITY_BORDERLINE`** (0.90 ≤ 0.957 < 0.97 band).
- **But every leg is noise-consistent.** Paired (same-question, same-body) McNemar adjudication of the "within benchmark noise" clause:

| leg | discordant | AR-only-correct (b) | SPEC-only-correct (c) | net (spec−ar) = c−b | paired Δacc CI95 |
|---|---|---|---|---|---|
| gpqa_diamond | **38** | 21 | 17 | **−4** (= noise) | [−0.081, +0.041] |
| mmlu_pro | 27 | 15 | 12 | −3 | [−0.044, +0.024] |
| gsm8k | 4 | 3 | 1 | −2 | [−0.020, +0.006] |
| aime | 4 | 2 | 2 | 0 | [−0.065, +0.065] |

The break **is** active on the eval prompts too (e.g. 38/198 = 19% of GPQA answers change correctness) — but **near-symmetrically**: on GPQA spec wins 17 (AR-wrong→SPEC-right) and loses 21 (AR-right→SPEC-wrong), net −4, fully inside paired noise (McNemar p=0.627, paired Δacc CI spans 0). The small negative nets (consistent with the ≤1.0 retention point estimates) are not statistically distinguishable from zero on any benchmark → **`statistical_verdict = NO_SIGNIFICANT_DEGRADATION`**.

### 3. Combined body+break vs vanilla base (#515) — *secondary, NOT the isolated finding*
Base denominators from BASELINE.md (#580/#581 grounding). **Protocol caveat (`protocol_matched=0`):** these were measured on a different protocol — notably AIME base=0.100 used a 3072-token cap with ~72% truncation vs our 6144 cap — so this table is indicative, not a protocol-matched gate verdict.

| leg | SPEC | base | spec/base | gate (≥.9·base) | AR clears? | SPEC clears? |
|---|---|---|---|---|---|---|
| gsm8k | 0.9300 | 0.8967 | 1.037 | ≥0.807 | ✅ | ✅ |
| mmlu_pro | 0.6900 | 0.6727 | 1.026 | ≥0.605 | ✅ | ✅ |
| **gpqa_diamond** | 0.4495 | 0.5236 | 0.858 | ≥0.471 | ❌ (0.4697) | ❌ (0.4495) |
| aime | 0.3333 | 0.1000 | 3.33† | ≥0.090 | ✅ | ✅ |

†protocol-mismatched base. **Key separation: `break_caused_new_515_gate_failures = NONE`.** The only sub-gate benchmark (gpqa_diamond) is sub-gate in the **AR reference too** (0.4697 < 0.471) — i.e. it is the int4-**body's** gap, present with or without speculation; the break adds only −0.0202 (noise) on top.

### What happened (honest analysis)
The break is real and enormous *at the token level* (51.7% of tokens differ), but it is **quality-immaterial**: it reshuffles reasoning trajectories onto different-but-equivalent paths that reach the same-quality answers. The single sub-0.97 point estimate (gpqa 0.957) is **not** a degradation — it is a 4-question symmetric net on 38 discordant flips, statistically zero (McNemar p=0.63, CI spans 1.0). By the card's own band this is formally **BORDERLINE**; substantively there is **no statistically detectable quality cost on any of AIME / GPQA-D / MMLU-Pro / GSM8K**.

**Decision framing for the human (you own strict-#319 / #481 — this prices it, it does not recommend changing it):** the strict-#319 break is **not a quality wall**. The blockers to shipping the ~170-band spec stack are (a) the int4-**body's** own GPQA-D gap [ubel #679 domain] and (b) a **policy** question: is byte-exactness worth forfeiting the spec TPS lever when the break itself costs no measurable benchmark quality? The ~170-band projects ≈150 official under the ×0.870 stark tax (projection, not measured — not a fire).

### Suggested follow-ups
- **Hand the strict-#319 call to the human as a policy decision** (byte-exactness vs the spec TPS lever at zero measurable quality cost), not a quality decision.
- The real remaining quality blocker is the **int4-body GPQA-D gap** (sub-gate in AR too) — ubel #679's lane, not this break's.
- If tighter CIs on the binding gpqa leg are wanted, add multi-seed gpqa (the rollup already pools per-seed) — but McNemar p=0.63 makes a flip to significance very unlikely.
- A **τ=0.3nat tolerance contract** (#616: rescues 100% of per-step flips) would convert this break to "byte-exact under tolerance" *if* the human relaxes strict #319 — the cleanest reconciliation of the spec lever with an identity gate.

### Reproduction (exact commands, local A10G)
```bash
cd /workspace/senpai/target
DIR=research/validity/specbreak_quality_materiality
# 1) serve AR arm (spec OFF, k=0), run full 4-leg panel + token capture, then stop:
python $DIR/serve_arm.py --arm ar --port 8000 &      # NUM_SPECULATIVE_TOKENS=0
bash   $DIR/run_arm.sh ar                              # gsm8k,mmlu,gpqa,aime + token_ar.jsonl
# 2) serve SPEC arm (spec ON, k=6) identically:
python $DIR/serve_arm.py --arm spec --port 8000 &     # NUM_SPECULATIVE_TOKENS=6, /tmp/qat-assistant
bash   $DIR/run_arm.sh spec                            # + token_spec.jsonl
# 3) token-break diff + rollup (+W&B):
python $DIR/token_break_probe.py diff --ar $DIR/results/token_ar.jsonl \
       --spec $DIR/results/token_spec.jsonl --out $DIR/results/token_break.json
python $DIR/rollup.py --wandb --wandb_group specbreak-quality-materiality-wirbel
```

### Provenance
- **Peak GPU memory:** ~19.6 GiB (vLLM serve footprint, `gpu_memory_utilization=0.90` on the 23 GiB A10G; int4 body + K=6 MTP drafter + KV cache). No HF job → no `summary.json` (analysis_only; `official_tps=0`).
- **W&B** (`wandb-applied-ai-team/gemma-challenge-senpai`, group `specbreak-quality-materiality-wirbel`): VERDICT run **`ezvgx3et`** (carries `specbreak_worst_retention`, `statistical_verdict`, per-bench retention, `specbreak_token_break_rate`, combined-ratio + gate-separation, `analysis_only=1`/`official_tps=0`/`fires=0`); per-leg runs `kwu5bv5k` (gsm8k), `958vaw51` (mmlu_pro), `e4lmv768` (gpqa_diamond), `btka69v0` (aime).
- **Public evidence used:** public leaderboard (frontier ~508 TPS; frontier methods e.g. `sparkgemma` ctk48-e2drafter and the `osoi5…fastmtp` family confirm MTP spec-drafter is the field's primary TPS lever) — motivates pricing our strict-#319 block on exactly that lever. Internal grounding (all cited in PR): wirbel #671 (serving-robust ~170 band, W&B `e8i9tqos`), kanna #673 (break mechanism, `7iqxlycu`), ubel #672 (int4-body gap); base #515 denominators from BASELINE.md (#580/#581).
- **Artifacts banked** (`research/validity/specbreak_quality_materiality/`): `serve_arm.py`, `run_arm.sh`, `run_leg.sh`, `token_break_probe.py`, `rollup.py`, `results/{ar,spec}/*.json`, `results/token_break.json`, `results/rollup.json`. (Bulky inspect `.eval` logs, raw token-id dumps, and serve logs gitignored per repo convention.)
