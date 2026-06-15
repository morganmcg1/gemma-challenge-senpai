STUDENT ubel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["bxmrvzwf"],"primary_metric":{"name":"token_identity_rate","value":0.995361328125},"test_metric":{"name":"greedy_identity_screen_self_test_passes","value":1}}

## Results

**Verdict: the M=8 batched verify DOES break strict byte-exact greedy-token-identity on this hardware — `token_identity_rate = 0.995361` (19/4096 positions flip, 0.46% divergence), and the divergence is DETERMINISTIC (not run-to-run noise).** Combined with the known EAGLE-3 SPEED cap (473.5 < 500), EAGLE-3 spec is **doubly-dead under #319 strict: speed-capped AND identity-failing.** This is the measured empirical grounding for why a reduction-invariant verify kernel (wirbel #360's lane) is required.

### Primary
| metric | value | meaning |
|---|---|---|
| **`token_identity_rate`** (PRIMARY) | **0.995361** | 4077/4096 positions match plain greedy AR |
| `verify_divergence` | 0.004639 | **19/4096** positions flip (8 prompts × 512 new tokens) |
| `eagle3_verify_breaks_strict_token_identity` | **True** | a single flipped ID fails strict |
| `per_sequence_strict_pass_fraction` | **0.375** | only **3/8** sequences are fully byte-exact; **5/8 fail strict** |
| **`greedy_identity_screen_self_test_passes`** (TEST) | **True** | all 10 self-test checks pass |

**First divergence** (the mechanistic fingerprint): prompt 0 (`mmlu_pro-000c2031fb`), generated offset 75 / abs pos 331 — AR picks token **616** (logprob −1.090), but the M=8 verify ranks token **2086** higher (logprob −0.652): a local logit gap of just **0.4375 nats**. Small-margin argmax flips are exactly where the M=8 chunked-forward reduction order tips the winner away from the M=1 AR decode.

### Determinism controls (the crux — this is signal, not noise)
- `determinism_ref_gen = 1.000000` — greedy-AR regeneration is bit-exact (int4 within-session).
- `determinism_verify_geometry = 1.000000` — the M=8 verify re-forward reproduces bit-exactly.
- ⇒ the 0.46% divergence is the **deterministic, repeatable** consequence of the M=8 batched-verify numerics differing from M=1 AR-decode numerics — a property of the reduction geometry, not jitter.

### Corroborates the deployed-M8 strict-failure signature (#232)
| anchor | divergence | geometry |
|---|---|---|
| this card | **0.46%** | faithful single-sequence M=8 chunked-prefill verify vs M=1 AR decode |
| lawine #232 | 0.73% | deployed M=8 batch-replica verify |
| delta | −0.0027 | within 1% reconcile tol ⇒ **corroborates** same mechanism |
| lawine #196 (strict floor) | 0.00% (identity=1.0) | non-spec int4 M=1 AR — the reference that *does* hold strict |

My 0.46% (literal causal verify geometry) sits between the #196 strict floor (1.0) and the #232 deployed signature (0.73%) — same mechanism, reproduced on real silicon.

### SPEC arm: native-acceptance blocked by head WEIGHTS, not integration (fallback step 6 — and a bonus)
The native EAGLE-3 K=7 acceptance stream could **not** be run: the trained head (`gua9x68j`/`56ksyxgw`) is unretrievable — W&B `logged_artifacts==[]` for both runs, `.pt` absent on disk, publish was human-owned and never done; only the #333 **synthetic-zero** candidate exists (a zero head drafts degenerately, α≈0). Per the card's step-6 fallback I measured the **structural quantity the eagle3 verify is made of** instead (the M=8 verify-geometry identity above), which directly answers the binding question.

**Bonus, stronger than the fallback anticipated:** the vLLM eagle3 greedy spec engine **CONSTRUCTS and RUNS end-to-end** on this int4 target with the #333 candidate head (`eagle3_engine_constructed=True`, `eagle3_generate_ran=True`, `blocker=null`; aux layers (2,21,39) loaded, distinct embed/lm_head detected). **There is NO eagle3 local-integration blocker** — the #338 **C2 integration unknown is CLOSED** (integration works). The only missing piece for a native-acceptance measurement is the trained **head weights**, not the engine. This sharpens C2 to: *served greedy-identity*, not *integration*.

### Secondary (LOCAL RELATIVE — ~7× off official per #245; NOT the official metric, 0 official TPS)
- REF greedy-AR decode: **20.0 tok/s** (50.0 ms/step) local-relative.
- Draft-head forward in isolation (real #333 candidate shapes/dtype, GPU): **3334.7 µs/token**, K=7 chain **23341 µs** local-relative (shape/dtype-bound; synthetic-zero values faithful for timing).
- PPL spot-check: **1.2968** (sanity only, well under the ~2.42 cap — NOT the gate).
- α (accepted-tokens-per-verify): **N/A** (native trained head unavailable).
- Peak GPU: **11.91 GB**.

### Baseline comparison (per PR)
- Official frontier **481.53 TPS / PPL 2.3772 (PR #52)** — **UNCHANGED**; this is a local relative screen, **0 official TPS**. ✅
- Strict ladder context (#319): 165.44 (#196 non-spec int4, identity=1.0) → 357.32 (#326 off-shelf BI spec) → ≤481.53 (#354 reduction-invariant kernel). EAGLE-3 spec SPEED-capped 473.5<500; **its IDENTITY status — measured here — is FAIL (0.46% divergence).**

### Command
```
cd target/ && python research/validity/eagle3_greedy_identity_screen/eagle3_greedy_identity_screen.py \
    --gpu --wandb_group eagle3-greedy-identity-screen --wandb_name ubel/eagle3-greedy-identity-screen
```
GPU work runs as isolated subprocesses (CUDA_VISIBLE_DEVICES=0); the int4 substrate is the deployed `gemma-4-E4B-it-qat-w4a16-ct` snapshot (bit-exact strict-ladder reference). **W&B run `bxmrvzwf`.**

### What happened
The screen cleanly settles the binding strict question: **an M>1 batched verify (EAGLE-3 K=7 ⇒ M=8) does not preserve strict greedy-token-identity on this hardware, by a deterministic 0.46%.** Because both determinism controls are exactly 1.0, this is not measurement variance — it is the genuine numerical divergence between the width-8 chunked re-forward and the width-1 AR decode, the same mechanism behind the deployed-M8 0.73% (#232), reproduced in the literal causal verify geometry. The native-acceptance arm was blocked only by missing head weights (not integration — the eagle3 engine constructs and runs), so the structural M=8 measurement is the faithful answer the card's fallback was designed for. Net: EAGLE-3 is doubly-dead under strict, and the reduction-invariant verify kernel is empirically warranted.

### Suggested follow-ups
1. **Retrieve/publish the native trained head** (`gua9x68j`/`56ksyxgw` `.pt`) — the only missing piece for a direct native-acceptance K=7 stream; engine integration is already proven here.
2. **Validate the fix:** run this same M=8 verify-geometry identity through wirbel #360's reduction-invariant verify kernel — expect it to restore `token_identity_rate=1.0`; this card is its baseline-breakage reference.
3. **Tighten the rate CI:** extend N / prompt count and characterize *where* small-margin flips concentrate (low- vs high-entropy positions) to bound worst-case strict exposure.
4. **dtype vs geometry split:** repeat on the served bf16-lm_head path (cross-session nondeterministic, per memory) to separate reduction-geometry breakage from dtype breakage.

### Repro note (bugs fixed in-card + env)
- Fixed an orchestrator/argparse flag mismatch (`--iters` → `--dh-iters`) that had silently dropped the draft-head latency phase; now populated.
- Fixed a W&B import-order bug: a prior-run `target/wandb/` output dir (namespace pkg, no `.init`) shadowed the installed package when REPO_ROOT was inserted at `sys.path[0]`. Now wandb is imported **before** REPO_ROOT is appended, and run output is redirected to `/tmp` so the shadow is never recreated. Added `--relog-wandb` to log existing `_results.json` without re-running GPU.
- The vllm serving venv lacked `wandb` (already a project dep, `wandb>=0.19.0`); installed it via `uv pip install`. If reproducing, run with an interpreter that has both `vllm` and `wandb`, or use `--relog-wandb` from a wandb-capable interpreter afterward.

_Public-evidence note: all anchors cited (frontier 481.53/PPL 2.3772 #52; strict floor identity=1.0 #196; deployed-M8 0.73% #232; #322/#333/#338/#350 reads) are reused, not re-derived. This card adds 0 official TPS and changes no served file._
