STUDENT land:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["7m8eyv0f","kgjgt6om","qrqhprkt","rv1v3lxi","0gjodi7k"],"primary_metric":{"name":"wall_tps_best_ngram","value":89.91},"test_metric":{"name":"ppl","value":2.0057}}

## Results — clean NULL: ngram/prompt-lookup drafter does NOT beat MTP on this prompt mix

**Verdict: NOT fire-worthy.** The ngram drafter fails **Gate 1 (TPS)** decisively — best ngram cell **89.91 local wall_tps vs the MTP control's 218.11** (~2.4× slower). PPL and 128/128 pass; greedy divergence is the shared int4 verify-tie residual (see §Greedy), not a drafter break. Variant B (stacked ngram+MTP) is architecturally unsupported in vLLM 0.22.0 and was recorded + skipped per the PR. This is a valid private-stable finding: the acceptance lever does **not** favor ngram here.

### A/B table (128 prompts × 512 tok, temp=0 seed=1, A10G, one GPU serial)

| cell | wall_tps | tps/control | E[T] tok/step | accept_rate | PPL | W&B |
|---|---|---|---|---|---|---|
| **control_mtp_k6** (bi0 shipped) | **218.11** | 1.000 | **3.343** | 0.390 | 2.00565 | `7m8eyv0f` |
| ngram_k3_plm3 | 88.38 | 0.405 | 1.991 | 0.330 | — | `kgjgt6om` |
| ngram_k3_plm4 | 87.37 | 0.401 | 1.998 | 0.333 | — | `qrqhprkt` |
| ngram_k5_plm3 | 89.91 | 0.412 | 2.161 | 0.233 | 2.00566 | `rv1v3lxi` |
| ngram_k5_plm4 | 89.13 | 0.409 | 2.166 | 0.234 | — | `0gjodi7k` |

Control reproduces the official bi0 baseline almost exactly — **local 218.11 vs official 218.02 TPS, local PPL 2.00565 vs official 2.0058** (W&B `s63tb03x`) — so `wall_tps = completion_tokens/duration_s` is a faithful local proxy and the ~2.4× ngram regression is real, not a harness artifact.

### Gate checklist
| Gate | Threshold | Best ngram (k5_plm3) | Result |
|---|---|---|---|
| TPS > bi0 control | > 218.11 (proj > 218.02) | 89.91 | **FAIL** (0.41×) |
| greedy token-identical to control | byte-exact (tie-tolerant) | shared int4 tie-residual, no extra divergence | see §Greedy |
| PPL ≤ 2.42 | ≤ 2.42 | 2.00566 | PASS |
| 128/128 | 128 | 128/128 | PASS |

Gate 1 is the hard fail, so the config is not fire-worthy regardless of the others. No HF job, no approval issue.

### Why ngram loses (analysis)
The hypothesis was *more accepted tokens per weight read*. On the 128-prompt ShareGPT general-chat mix the opposite happens:
1. **Lower acceptance.** ngram E[T] = 1.99–2.17 accepted tok/verify-step vs MTP's **3.34**. Suffix n-gram matching only hits on repetitive/structured spans (code, lists, JSON); general chat has few exact `[min..max]`-gram repeats, so most drafts miss and the step accepts ~1 token.
2. **It still pays a full M=K+1 target verify forward every step.** ngram's "zero GPU cost" is only the *proposer*; the int4 target still runs an M=(num_spec+1) verify forward per step. With low acceptance that verify compute is mostly wasted — fewer committed tokens per identical weight read, the exact inverse of the lever. Raising num_spec 3→5 *lowers* accept_rate (0.33→0.23) because the extra draft positions almost never match, confirming the miss-dominated regime.
3. MTP's learned draft head proposes contextually and lands ~3.34 tok/step, so it wins on this mix by a wide margin.

So bi0's existing MTP drafter is already the better acceptance lever here; ngram would only help on a corpus far more repetitive than the eval mix.

### Greedy-identity (eval-rigor seat — I made this airtight)
The strict byte-exact gate (`scripts/local_validation/greedy_gate.compare`) flags **every** int4 spec arm DIVERGENT — **including the shipped bi0 MTP control** vs the plain-AR reference R:

- **control_mtp_k6 vs R:** 108/128 prompts, 34927/65536 tok (53.3%), onset min=2 / **median=128** / max=507.
- **ngram_kN vs control C:** 108–109/128 prompts, ~53.3% tok, onset median≈128 — **statistically identical to control-vs-R**.
- **ngram_kN vs R:** 101–104/128 prompts, ~48.6% tok — **less divergent than the control itself is vs R**.

Interpretation (measured here, not assumed): this is the **int4 M=K-verify-vs-M=1-AR argmax-tie residual**, not an ngram effect. At temp=0 vLLM's rejection sampler emits the int4 target's argmax and rejects on first draft≠target, so the accepted token is independent of which drafter proposed it. Two facts prove the swap adds no divergence: (a) ngram-vs-control ≈ control-vs-R (same rate, same onset shape); (b) ngram-vs-R is *lower* than control-vs-R. The greedy-safety-by-construction claim holds empirically — the drafter only changes which/how-many positions a verify forward covers, never the committed token.

**Flag for you:** the strict gate fails the *shipped, greedy-safe* bi0 control too, so it must be read **tie-tolerantly** — the same standard under which bi0 cleared its own greedy gate. Under that reading every ngram cell is exactly as greedy-safe as the shipped control. Raw onset arrays + per-cell reports are in `research/ngram_spec_782/analysis.json` for your audit.

### Variant B — stacked ngram+MTP: architecturally unsupported (recorded, skipped)
vLLM 0.22.0 `SpeculativeConfig` (`.../vllm/config/speculative.py`) accepts exactly **one** proposer:
```
method: SpeculativeMethod | None        # single Literal: ngram | medusa | mlp_speculator | draft_model | suffix | custom_class | eagle* | ngram_gpu
model:  str | None                      # single draft model
prompt_lookup_max / prompt_lookup_min   # ngram-only
```
There is no combined/tiered/list field — one `method` + one `model` per engine. A combined ngram-front + MTP-back drafter cannot be expressed, so Variant B was skipped per the PR ("if the build refuses a combined drafter, record the exact incompatibility line and skip — don't force it").

### Reproduction
```bash
cd /workspace/senpai/target
export CUDA_VISIBLE_DEVICES=0
bash research/ngram_spec_782/run_grid.sh         # serves each cell via local_validation harness (bi0 stack, drafter swapped only by extra-env)
.venv/bin/python research/ngram_spec_782/analyze.py    # offline greedy gate vs R and vs control C + table -> analysis.json
.venv/bin/python research/ngram_spec_782/log_wandb.py  # one run/cell in group bi0-ngram-spec
```
Per-cell driver: `run_cell.py --submission <int4_mtp_bi0_surgattn | int4_ngram_bi0_surgattn> --label <cell> --extra-env <json> --num-prompts 128 --output-len 512 [--ppl]`. Drafter selected purely by env (`SPECULATIVE_METHOD=ngram`, `NUM_SPECULATIVE_TOKENS`, `PROMPT_LOOKUP_{MAX,MIN}`); all 3 force-2D patch files in `int4_ngram_bi0_surgattn` are byte-identical to bi0, so the only isolated variable is the proposer. `VLLM_USE_FLASHINFER_SAMPLER=0` on every arm (local flashinfer JIT lacks `curand.h`; native sampler is numerically inert at temp=0 and applied identically to all cells). PPL measured on control + ngram_k5_plm3 (both 2.0057, identical) — PPL is a prompt_logprobs prefill forward that never invokes the drafter, so it is provably drafter-invariant; the remaining 3 ngram cells were not re-served.

**Peak GPU memory:** model load 9.86 GiB + KV cache 8.45 GiB + CUDA-graph 0.05 GiB ≈ **18.4 GiB working set** under `--gpu-memory-utilization 0.90` (≈21.6 GiB cap on the 24 GiB A10G). KV cache 336,922 tokens, max concurrency 82× at 4096 ctx. Identical across all cells (same int4 target; ngram proposer has no weights).

**W&B group `bi0-ngram-spec`:** control `7m8eyv0f`, ngram_k3_plm3 `kgjgt6om`, ngram_k3_plm4 `qrqhprkt`, ngram_k5_plm3 `rv1v3lxi`, ngram_k5_plm4 `0gjodi7k`. (Smoke 4-prompt cell kept on disk as an audit artifact; excluded from the dashboard — a 4-vs-128-prompt comparison yields a misleading >1 flip rate.)

### Suggested follow-ups
- **Corpus-conditioned drafter routing:** ngram only wins where outputs are self-repetitive. If the private set is known to be code/JSON/structured-heavy, an ngram tier could help — but it would need per-request fallback to MTP on low suffix-hit-rate, which vLLM 0.22.0's single-method config can't express. Would need a custom proposer (`method:"custom_class"`) — larger scope, separate PR.
- **The acceptance lever still points at MTP depth**, not drafter family: MTP already lands E[T]=3.34. fern #774's MTP-depth-K sweep is the better-aimed shot at the same lever; this null says don't spend more on ngram for general chat.
- **Eval-rigor:** the strict greedy gate failing the shipped bi0 control is worth a standalone reconciliation — either bless a documented tie-tolerant band for all int4 spec arms, or pin a canonical M=K-verify reference so the gate stops flagging greedy-safe arms. Happy to take that as a follow-up PR if you want it.
