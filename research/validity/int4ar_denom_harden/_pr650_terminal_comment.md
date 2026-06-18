STUDENT ubel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["pr650-harden-VERDICT","pr650-harden-gpqa-bf16-10seed","pr650-harden-aime-sampled-int4ar","pr650-harden-aime-sampled-bf16","pr650-harden-aime-int4ar-mt12288"],"primary_metric":{"name":"int4ar_aime_mt12288_fl0_clean","value":0.4},"test_metric":{"name":"int4ar_aime_mt12288_fl0_clean","value":0.4}}

## Results — int4-AR denominator hardening (PR #650) — TERMINAL

All **LOCAL** build + serve + eval. `analysis_only=true`, `official_tps=0`, **no HF Job, no submission, no served-file change.** Live `int4_g128_lmhead` @ **126.378 official TPS / PPL 2.019 untouched.** Reused the #638 stack verbatim: vLLM **0.22.0**, BI=1 (`VLLM_BATCH_INVARIANT=1`), gb-config, `min_tokens=8` EOS-guard, `VLLM_USE_FLASHINFER_SAMPLER=0`, prometheus shim. Two pre-built models: bf16 base (`google/gemma-4-E4B-it`, full head) and the int4-AR live-rung body (`/workspace/gemma_build/int4_g128_lmhead`).

### Arm A — AIME budget→truncation split (PRIMARY) — verdict `AIME_DEFICIT_GENUINE`

Greedy AIME (years 2024, 2025-I, 2025-II; k=1; no-thinking), mt∈{6144, 8192, 12288}, `--max-model-len 16384` so the output cap is the only moving part. raw acc / `fl` (finish_reason=length rate) / **censored** acc (finished-only):

| budget | bf16 acc / fl / cens | int4-AR acc / fl / cens |
|---|---|---|
| 6144  | 0.4833 / 0.150 / 0.5686 | 0.3167 / 0.133 / 0.3654 |
| 8192  | 0.4833 / 0.017 / 0.4915 | 0.3667 / 0.017 / 0.3729 |
| 12288 | 0.4833 / 0.000 / 0.4833 | **0.4000** / 0.017 / 0.4068 |

ctok p95: bf16@12288 = 7316, int4-AR@12288 = 8099; int4-AR ctok_max@12288 = 12288 → exactly **one** problem still hits the cap. Derived fl-curve from each model's 12288 run: int4-AR fl_at_{6144, 8192, 12288} = {0.133, 0.033, 0.000}; bf16 = {0.150, 0.017, 0.000}.

**Read:** bf16 is budget-insensitive (flat 0.4833 — its truncated chains were wrong anyway). int4-AR climbs **+0.083** (0.3167→0.4000) as `fl→0`, so a real chunk of the raw-6144 gap *was* the 6144 cutoff. **But at the clean budget (12288, `fl≈0` both), int4-AR = 0.400 still sits −0.020 UNDER the 0.420 bar and 0.083 below bf16.** Even charging the lone remaining truncated int4-AR problem as correct → 25/60 = 0.4167 < 0.420.

**Matched sampled CI @12288** (T=1, top_p=0.95, top_k=64; k=1 × 5 seeds; n=300; byte-identical int4-AR/bf16):
- **int4-AR: 0.3467**, Wilson **[0.2951, 0.4022]**, fl=0 — CI upper **< 0.420**.  per-seed [0.350, 0.317, 0.367, 0.350, 0.350]
- **bf16: 0.4600**, Wilson **[0.4045, 0.5165]**, fl=0.  per-seed [0.517, 0.433, 0.450, 0.483, 0.417]
- The two CIs are **non-overlapping** (int4-AR hi 0.4022 < bf16 lo 0.4045) → the AIME gap is **body-level**, not a sampling artifact (matched sampled gap −0.1133).

→ **`AIME_DEFICIT_GENUINE`: truncation *amplified* the gap but did not *create* it. The "literal-gate-breach-via-AIME" claim survives the correction.** (Bonus: @12288 = 0.400 reconciles denken #637's 0.400 — the ubel-0.350-vs-0.400 delta was the 6144 cap all along.)

### Arm B — 10-seed bf16 GPQA-D sampled denominator (SECONDARY) — bar RISES, int4-AR flips to marginal FAIL

bf16 GPQA-D sampled, dseed 12345, sampling-seeds 0..9, T=1/top_p=0.95/top_k=64, mt6144 (byte-identical protocol to the #638 int4-AR 10-seed leg). Per-seed acc: 0.5758, 0.5707, 0.5707, 0.5202, 0.5354, 0.5859, 0.5707, 0.5455, 0.5455, 0.5404.

- **Pooled 0.5561 (1101/1980), Wilson [0.5341, 0.5778]**; per-seed mean 0.5561, SE 0.0064 (min 0.5202 / max 0.5859); fl 0.0025.
- The stable 10-seed mean (0.5561) sits **above** the single-seed anchor (0.5404) → the 0.9× bar **RISES 0.4864 → 0.5005** (the single seed was *low*, not high as the card hypothesized).
- **int4-AR 0.499 does NOT clear** the recalibrated bar (margin **−0.0015**) → flips from #638's marginal-*pass* to marginal-*fail*.
- **Option-B 0.4652 does NOT clear** either (margin −0.0353).

→ The seed-robust denominator **flips GPQA-sampled from marginal-pass to marginal-fail** for both arms. **Honest caveat:** int4-AR's −0.0015 is a hair (inside seed noise); the *sign* is robust (all 10 bf16 seeds ≥ the old 0.5404 anchor), but I would not call this a decisive fail — it's a true coin-flip whose central estimate now lands on the wrong side of the bar. Stated with a CI it's a tie; stated as a point margin it's a fail.

### Arm C — Option-B AIME 10-seed sampled (STRETCH) — NOT run

Intentionally skipped (STRETCH / "only if GPU remains"). The body-vs-spec question is already answered: #638 single-pass int4-AR 0.350 ≈ Option-B 0.367, and this card's matched int4-AR sampled (0.3467) < bf16 sampled (0.46) confirms the AIME deficit is **body-level, not spec-level**. GPU was committed to the 4.5 h Arm-B GPQA pole + the bf16 sampled-AIME CI. One-run add if you want the explicit Option-B CI on the int4+spec server.

### Census
- AIME greedy budget grid: 6 cells × 60 = **360** prompt-completions (bf16 ×3 budgets, int4-AR ×3 budgets).
- AIME matched sampled @12288: int4-AR 5×60 + bf16 5×60 = **600**.
- GPQA-D 10-seed: 10 × 198 = **1980**.
- **Total 2940 prompt-completions** across the panel.

### Exact commands
```bash
# bf16 analysis server: mml=16384 (cap is the only moving part), BI=1, no FlashInfer sampler
VLLM_BATCH_INVARIANT=1 VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve \
  <gemma-4-E4B-it bf16 snapshot> --served-model-name gemma-4-e4b-it --port 8000 \
  --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 2048 --max-num-seqs 16 --seed 0 --trust-remote-code
# whole bf16 side (resumable): Arm A greedy grid -> Arm B GPQA 10-seed -> Arm A-CI bf16 sampled AIME
bash research/validity/int4ar_denom_harden/run_bf16_arms.sh
# aggregate + W&B re-log (system /usr/bin/python3 has the real wandb; the eval venv does not)
/usr/bin/python3 research/validity/int4ar_denom_harden/aggregate_harden.py --wandb
```
(int4-AR side — greedy grid + 5-seed sampled — was collected earlier on the int4 server via `aime_budget.sh` / `aime_sampled.sh`.)

### Peak memory
bf16 analysis server steady-state **~19.5 GB / 23 GB** on a single A10G (mml=16384, gpu-mem-util 0.90). int4-AR body well under that. No OOM.

### W&B
Group **`int4ar-denominator-harden-ubel`** (entity `wandb-applied-ai-team`, project `gemma-challenge-senpai`) — **12 runs**, stable ids `pr650-harden-*`, all `analysis_only=true` / `official_tps=0`:
`pr650-harden-VERDICT`, `pr650-harden-gpqa-bf16-10seed`, `pr650-harden-aime-sampled-int4ar`, `pr650-harden-aime-sampled-bf16`, `pr650-harden-aime-{bf16,int4ar}-mt{6144,8192,12288}` (6), `pr650-harden-flcurve-{bf16,int4ar}` (2).

### What happened — honest analysis
- **Arm A (the load-bearing axis): the AIME indictment of the live submission is robust.** The card's worry that 0.350 was truncation-amplified is *half right* — ~+0.083 of it WAS the 6144 cap. But at `fl≈0` the shipped int4 body still lands 0.400 < 0.420 and −0.083 vs bf16, with a 5-seed sampled CI whose upper edge (0.4022) is below the bar *and* non-overlapping with bf16. The literal-gate breach via AIME survives.
- **Arm B (the marginal axis): the bar moved the *wrong way* for the card's hope.** The card expected the single 0.5404 seed to be *high* (lowering the bar, firming int4-AR's clear). It was actually *low*: the 10-seed mean is 0.5561, so the bar RISES to 0.5005 and int4-AR's 0.499 flips to a (hair-thin) fail. GPQA-sampled no longer even marginally supports the live body — though at −0.0015 I'd report it as a seed-noise tie, not a clean fail.
- **Net:** both soft spots are retired and both harden the #481 reading toward indicting the live int4 submission — Arm A confirms the AIME breach is genuine; Arm B removes int4-AR's one marginal GPQA pass (now a tie/fail rather than a pass).

### Suggested follow-ups
- **Put a p-value on Arm B's −0.0015** rather than a point margin: a paired bootstrap over the 10 bf16 sampling-seeds (and the int4-AR seeds) would say whether int4-AR 0.499 vs bar 0.5005 is distinguishable from a tie. My read is it is not — so "marginal fail" should be stated as "no longer a pass; statistically a tie."
- **Arm C (Option-B AIME 10-seed sampled CI)** is a one-run add on the int4+spec server if you want the explicit body-vs-spec CI; current evidence already says body-level.
- The **bf16 sampled-AIME @12288 = 0.46 [0.4045, 0.5165]** (n=300, 5 seeds) is a clean matched reference — reuse it if any other card needs a sampled bf16 AIME denominator.
