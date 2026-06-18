STUDENT land:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"verdict":"REGIME_IS_ACCEPTANCE","analysis_only":true,"official_tps":0,"fires":false,"wandb_run_ids":["4fbu9b3o"],"primary_metric":{"name":"fastest_identity_safe_unrescued_k6_walltps","value":170.21},"test_metric":{"name":"rescued_k6_official_equiv","value":134.87}}

## Results — VERDICT: `REGIME_IS_ACCEPTANCE`

The 155↔170 un-rescued K6 gap is **the drafter (acceptance), not a per-step latency config knob.** `#666` cleared the cross-student read, so this completes ALL of #664 (steps 1–5), with stark's `#642` spec-path config + K6 espec **read directly from his branch** and **independently re-verified against source** (not just trusting the prior synthesis).

`analysis_only=true`, **no HF Job, no submission, official_tps stays 0**, locked 126.378 untouched. W&B: **`4fbu9b3o`** (https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/4fbu9b3o), group `spec-regime-isolation-land`.

### Step 1 — Espec discriminator (latency vs acceptance) → ACCEPTANCE
Both land-170 and stark-155 are `NUM_SPECULATIVE_TOKENS=6` on the **same** `int4_mtp_batchinv` stack (w4a16-ct body, BI=1, vllm 0.22.0, 128×512 seed1), measured by the **same** `scripts/profiler/paired_tps_ab.py` runner logging the **same** `e_accept_exact` key.

| regime | drafter | espec | wall_tps | implied step-rate |
|---|---|---|---|---|
| **land** (fast) | `/tmp/qat-assistant` (local QAT-matched) | **3.6574** | 170.21 | 46.54 steps/s |
| **stark** (slow) | `...qat-q4_0-unquantized-assistant` (stock Hub) | **3.3332** | 155.57 | 46.67 steps/s |
| ratio | — | **1.097** | **1.094** | **0.997** |

The espec gap (+9.7%) tracks the wall_tps gap (+9.4%); the per-spec-step rate is **identical** (−0.28%). A latency knob would move the step-rate — it doesn't. **The locus is acceptance.**

### Step 2 — Config attribution → the single differing field is `DRAFTER_MODEL`
Diffing the full spec-path config, the **only** differing field is the drafter. stark's bc run used `override_env: {}` → the submission **default** drafter; land overrode `DRAFTER_MODEL=/tmp/qat-assistant`. Verified the default in-repo: `submissions/int4_mtp_batchinv/manifest.json:17` + `serve.py:82` both pin `google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant`.

**Decisive for the #1 question:** the **official submission ships the stock-Hub drafter**, so the official harness realizes **stark's ~155 / rescued-~135 regime**. land's 170 needs the **local-only** `/tmp/qat-assistant` (not on the Hub, not in the submission). It is NOT a per-step latency knob, so the Step-2 "pin to stark's value" latency A/B is moot.

### Step 3 — Ratio-stability → **PROPAGATES** (+9.40%)
The un-rescued delta does **not** cancel in the rescued official-equiv, because the AR-rung denominator (126.75 local → 126.378 official) is **drafter-independent**:

| regime | rescued K6 | official-equiv | vs locked 126.378 | clears +10? |
|---|---|---|---|---|
| stark / stock drafter (**ships**) | 135.27 (MEASURED, stark #663 acceptor) | **134.87** | +6.7% | **No** |
| land / local drafter | 147.99 (PROJECTED via stark's measured tax 0.870) | **147.55** | +16.8% | **Yes** |

So the drafter is the lever that flips +6.7% → clears the +10 bar — but only if `/tmp/qat-assistant` is published and shipped.

### Step 4 — Identity + PPL re-confirm (audit gap closed)
Full #319 BI=1 both-sides teacher-forced re-gate, 64 prompts × 5,982 positions:
- `break_rate` @ τ=0.3 = **0.0** (τ-ladder: τ0.0→6 flips=0.001003, τ0.2→0, τ0.3→0). The 6 residual flips at τ0 are **int4 exact-ties** (margin 0.125 nat ≪ 0.5), `attention_path_break_count=0`, `attn_is_batch_invariant=true`, `aten_mm_bitexact=true`. Verdict `ATTENTION_RECOVERED_RESIDUAL_IS_INT4_TIES` — consistent with the irreducible int4-tie residual.
- **PPL = 2.0055 ≤ 2.42** ✓ (target-only, drafter/K-independent).
- **Audit gap from #660 closed:** explicit machine-checkable scalars now logged in-run — `summary/gate/break_rate_tau0p3=0`, `summary/gate/spec_fire_rate=0.0727`, `summary/gate/break_rate_bi1_both_sides=0.001003`, `summary/gate/attention_path_break_count=0`, `summary/gate/ppl_spec=2.0055`.

### Own-config knob sweep (the actionable half) → no material identity-safe latency lever
| knob | setting | cand wall_tps | Δ vs 169.95 | identity-safe | disposition |
|---|---|---|---|---|---|
| `sampler1` | `VLLM_USE_FLASHINFER_SAMPLER=1` | — | — | n/a | **BOOT_FAIL** (ninja JIT exit 1) **+ greedy-moot** (sampling-only path) |
| `eager1` | `ENFORCE_EAGER=1` (cudagraph OFF) | 45.10 | **−73.5%** | **No** (byte 18/128) | **disqualified** — cudagraph-OFF exposes inductor fusion reorder → identity break |
| `flashattn` | `VLLM_ATTENTION_BACKEND=FLASH_ATTN` | 170.19 | +0.142% | Yes (byte 128/128) | **immaterial** — see note |

**flashattn note:** Gemma4 **overrides** the requested `FLASH_ATTN` back to `TRITON_ATTN` (both base and candidate server logs show `Using AttentionBackendEnum.TRITON_ATTN`), so the knob is effectively a **no-op on the identical backend** — the +0.142% (≈0.24 TPS) is pure run-to-run noise (CV 0.009%), not a real lever. I added a **materiality gate** to the synthesis (`MATERIAL_SPEEDUP_PCT=1.0`): a statistically-real but sub-1% move is within hardware noise and cannot "make the slow regime run fast", so `shippable_identity_safe_speedup_exists=False`. The +0.142% is recorded transparently as `is_material=False`.

### Step 5 — Deliverable framing: SURFACE, not fire
There is **no shippable, identity-safe latency knob** that raises the official-harness regime. The only path from stark's ~135 to land's ~147 official-equiv is **publishing `/tmp/qat-assistant` to the Hub + repointing `manifest.DRAFTER_MODEL`** — a drafter swap that is greedy-identity-safe **by construction** (`serve.py:14` enforces greedy-identity for any drafter; the re-gate's `break_rate=0` doesn't even load the drafter) and PPL-neutral. **This is a SURFACE candidate for #481, NOT a fire** (official_tps stays 0; the "no blind fire on unresolved official speed" guardrail holds — the point was to *reduce* official-speed uncertainty before quota). **Does it warrant an official re-measure?** Yes — *conditionally on first publishing the local drafter*; re-measuring the current stock-drafter submission would only re-confirm the ~135 regime.

### Baseline comparison (from PR body)
| quantity | PR baseline | this card | agreement |
|---|---|---|---|
| land un-rescued K6 (fast) | 170.16 | 170.21 | ✓ |
| stark un-rescued K6 (slow) | 155.58 | 155.5693 (read from source) | ✓ |
| stark AR-rung local | 126.75 | 126.7518 (read from source) | ✓ |
| stark captured rescued K6 | 135.27 | 135.2721 (read from source) | ✓ |
| rescued K6 band | [135.82, 146.82] | 134.87 (stock) … 147.55 (local) | ✓ consistent |

K-definition cross-check (own stack): K6=170.21 (e_acc 3.657), K7=152.31 (e_acc 3.825); **land's own K7 lands −2.10% from stark's K6 155.58** — i.e. one extra spec token on land's better drafter ≈ stark's K6 on the stock drafter, an independent corroboration that the gap is acceptance/draft-budget, not latency.

### Independent verification (cross-student, #666-authorized)
Confirmed `#666` is a real human grant (`morganmcg1`, `author_association: OWNER`, body `human: … Authorized!`, 2026-06-18T14:54:09Z) before any cross-read. Re-derived all four stark constants directly from `stark/optionb-rescue-deproject` source (`research/validity/optionb_rescue_deproject/bc/paired_ab.json` + `rate_sweep_captured/`) — every value matches the synthesis to full float precision. Read-only; stark's branch untouched.

### Public evidence used
Locked OFFICIAL anchor `submissions/int4_g128_lmhead` @ 126.378 TPS / PPL 2.019 (strict-#319 GREEDY_IDENTICAL). Drafter attribution grounded in the in-repo `submissions/int4_mtp_batchinv/{manifest.json,serve.py}` default. Cross-student stark `#642`/`#663` reads authorized via human issue `#666`.

### Commands
```bash
# Identity re-gate (#319 teacher-forced, BI=1 both-sides):
VLLM_BATCH_INVARIANT=1 .venv/bin/python \
  research/walltps_ab/optionb_bi1_stock_int4/spec_regime_isolation_664/regate_local.py

# Own-config knob sweep (each knob an independent paired_tps_ab.py call):
#   sampler1 : cand_env VLLM_USE_FLASHINFER_SAMPLER=1   -> BOOT_FAIL
#   eager1   : cand_env ENFORCE_EAGER=1                 -> -73.5% + id-break
#   flashattn: cand_env VLLM_ATTENTION_BACKEND=FLASH_ATTN (Gemma4 -> TRITON_ATTN)
.venv/bin/python scripts/profiler/paired_tps_ab.py \
  --baseline int4_mtp_batchinv --candidate int4_mtp_batchinv \
  --baseline-env VLLM_BATCH_INVARIANT=1 --baseline-env DRAFTER_MODEL=/tmp/qat-assistant \
  --baseline-env NUM_SPECULATIVE_TOKENS=6 \
  --candidate-env VLLM_ATTENTION_BACKEND=FLASH_ATTN [+ same base env] \
  --n 3 --num-prompts 128 --output-len 512 --seed 1

# Synthesis (folds knobs + cross-student reads, logs audit scalars to W&B):
.venv/bin/python research/walltps_ab/optionb_bi1_stock_int4/spec_regime_isolation_664/synthesize_664.py
```

### Peak memory
- Identity re-gate (teacher-forced, no served KV): **12.24 GB**.
- Served knob/benchmark arms: **~19.5 GB** resident (vLLM `gpu_memory_utilization=0.90` of the 23 GB A10G; util ~97%).

### What happened
The #1 pre-quota unknown is resolved: the official harness realizes **stark's ~155 / rescued-~135 regime** because the shipped submission uses the stock-Hub drafter. land's 170 regime is **real and identity-safe**, but it is **acceptance-driven** (a better local drafter, espec 3.657 vs 3.333), not a latency config the official harness can be pinned to. The own-config knob sweep found **no material identity-safe latency lever** (sampler non-bootable+greedy-moot; eager −73%+id-break; attention-backend overridden by Gemma4 → no-op). The 9.4% delta **propagates** to the rescued official-equiv (134.87 stock vs 147.55 local), so the drafter is the one lever that would clear the +10 bar — via a clean, identity-safe Hub-publish + manifest repoint, surfaced to #481.

### Suggested follow-ups
1. **(SURFACE → #481)** Publish `/tmp/qat-assistant` to a private Hub repo, repoint `submissions/int4_mtp_batchinv` (or the int4_g128 rescue submission) `DRAFTER_MODEL`, smoke-test load + greedy identity, then a single OFFICIAL re-measure of the **rescued** path to convert the projected 147.55 official-equiv into a measured number. This is the only action that moves the official regime.
2. Characterize `/tmp/qat-assistant` vs the stock q4_0-unquantized assistant (size, quant, training) to confirm the +0.32 espec is robust across the eval distribution, not a 128-prompt artifact, before spending the publish + quota.
3. Optional: a hosted-drafter local re-measure (stock Hub drafter, BI=1) to close the last proxy — confirm land's harness reproduces stark's 155.57 exactly when pointed at the same Hub drafter, fully nailing the attribution end-to-end.
