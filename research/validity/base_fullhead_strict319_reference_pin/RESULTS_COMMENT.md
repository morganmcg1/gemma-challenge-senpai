STUDENT wirbel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["2u44yaa1"],"primary_metric":{"name":"base_fullhead_strict319_flip_rate","value":0.06756591796875},"test_metric":{"name":"base_fullhead_free_running_seq_exact_rate","value":0.0}}

## Results — does base_fullhead itself pass strict-#319, bf16 or int4-QAT?

**VERDICT: `bf16_referenced_int4_unsatisfiable`.** The canonical greedy reference is **bf16**, and the quality-safe anchor **base_fullhead** (stock int4_g32 QAT body + full native 262,144-row BF16 lm_head) is **NOT byte-exact** against it: teacher-forced per-position flip rate **6.757 %** (4428/65536), free-running sequence-exact **0/128**. base_fullhead is the *best-case* int4 config (best body + full BF16 head + operative-tuned kernels), yet it still flips. Since the int4-body argmax perturbation is irreducible, **NO int4 config — including the anchor the entire NO-FIRE verdict rests on — can pass a literal-bf16 strict-#319.** Therefore the live "strict #319" contract is operationally the **operative / int4-referenced identity** (#407 "operative-1.0" lane), *not* literal bf16. Under that operative contract base_fullhead remains by construction the valid quality-safe identity anchor.

### KEY OUTPUTS
| output | value |
|---|---|
| `canonical_reference_is_bf16` | **true** |
| `canonical_reference_matches_int4qat` | **false** |
| `base_fullhead_strict319_byte_exact` | **false** |
| `base_fullhead_strict319_flip_rate` | **0.06757** (4428/65536, teacher-forced) |
| `base_fullhead_free_running_seq_exact_rate` | **0.0** (0/128) |
| `strict319_reference_verdict` | **bf16_referenced_int4_unsatisfiable** |
| `base_fullhead_is_quality_safe_319_anchor` | **true** (operative/#407 contract) |
| `self_det` | **true** (steady-state, 8/8; see §3) |
| `analysis_only` | **true** |
| `official_tps` | **0** |

NaN-clean. **No HF Job, no `--launch`, no submission, no served-file change** — all local on the idle pod A10G.

### 1. Reference provenance → **bf16** (public evidence used)
The canonical reference `research/greedy_reference/google__gemma-4-E4B-it/decode_outputs.jsonl` (`reference_kind="served_spec_off"`) is unambiguously the **bf16** base, from three independent in-repo records:
- **`meta.json`** (served): `model_id="google/gemma-4-E4B-it"`, `served_via="plain-baseline:…/vllm_baseline (MODEL_ID=google/gemma-4-E4B-it)"`.
- **`served_reference_server.log`** (decisive): vLLM EngineCore loaded `model='google/gemma-4-E4B-it', dtype=torch.bfloat16, quantization=None, quantization_config=None` — plain bf16, **not** int4 compressed-tensors.
- **`meta.offline.json`** (offline sibling): `"dtype": "bfloat16", "quantization": null`.
- **served-vs-offline corroboration:** the two bf16 captures agree 102/128 seq-exact (0.797; residual is the served-vs-offline numeric-path gap, *not* quantization — both are bf16).

So `canonical_reference_is_bf16=true`. base_fullhead's free-running output is byte-exact on **0/128** prompts → `canonical_reference_matches_int4qat=false` (the reference is not the int4-QAT base).

### 2. base_fullhead strict-#319 — dual measurement (vs the bf16 canonical reference, 65,536 positions)
Served base_fullhead = stock `gemma-4-E4B-it-qat-w4a16-ct` snapshot (NO baked bucket) + full 262k BF16 head, spec-OFF, greedy, `MAX_NUM_SEQS=1`, `min_tokens=8`, `VLLM_USE_FLASHINFER_SAMPLER=0`. Server log confirms the substrate: `clearing SPECULATIVE_CONFIG (M=1 AR greedy reference, drafter OFF)`, `speculative_config=None`, `verified full lm_head: 262144 rows`, `quantization=compressed-tensors` (int4 body).

- **(A) Teacher-forced per-position argmax** (clean, #571-comparable, no cascade — each position conditioned on the fixed bf16 reference context): **flip_rate = 0.06757 (4428/65536)**. `byte_exact=false`.
- **(B) Free-running greedy** (literal #319 verifier protocol): **seq-exact 0/128**, median first-divergence index **7.5** (base_fullhead diverges right at the `min_tokens=8` boundary), then cascades to 0.945 divergent-token fraction. The very first generated token already flips on the GPQA prompts (reference 5471 → candidate 8291, *consistently* across prompts → a genuine int4-body flip, not noise).

### 3. self_det → **true** (steady-state), with a documented warmup caveat
A dedicated determinism diagnostic (`diag_determinism.py`, same recipe) on this exact substrate:
- **Free-running warm-vs-warm: 8/8 byte-identical** → base_fullhead is deterministic at steady state.
- **Teacher-forced self-consistency: 1.0** (0 disagreements / 12,288 positions across two full re-runs) → the headline 6.757 % is a perfectly reproducible per-position number.

The census's first-pass `self_det` leg read **1/8** only because it compared the **first 8 Phase-A captures (taken COLD, during ONEGRAPH + `LOOPGRAPH_WARMUP_CALLS=20` graph warmup)** against a warmed re-run: a single warmup-transient ULP flip cascades the 512-token stream. The server log confirms the mechanism (`enable_prefix_caching=True, enable_chunked_prefill=True`, FULL+PIECEWISE CUDA-graph capture). This is **not** decode non-determinism — it is a graph-warmup transient, and teacher-forced (which is what the flip rate is measured on, fully warmed) is bit-stable. **Independent reproducibility:** the full 128×512 census was run twice end-to-end and the teacher-forced flip rate reproduced *bit-exactly* (0.06756591796875 → 0.06756591796875).

### The int4-body flip ladder (all teacher-forced, vs the bf16 reference)
| config | head | body | flip vs bf16 |
|---|---|---|---|
| **base_fullhead (this card)** | full 262k BF16 | int4_g32 (surgical operative path) | **6.76 %** |
| land #571 stock int4_g32 body | — | int4_g32 (naive serve) | 10.90 % |
| land #571 int4_g128 body | — | int4_g128 | 13.74 % |
| wirbel #578 int4_g128_lmhead | 12k pruned int4 | int4_g128 | 16.2 % |

The ordering is physically coherent: better body + full BF16 head + argmax-preserving operative kernels (PLE_FOLD, FA_SLIDING, SPLITKV_VERIFY, SURGICAL_ATTN_USE_3D_OFF) ⇒ **fewest** flips. base_fullhead is **lower** than #571's naive int4_g32 (Δ = −0.0415): its 262k head is the full native BF16 head, so *all* divergence is int4-body-driven, and the operative path tracks bf16 more closely at near-ties. Crucially the flip is still **> 0** — the irreducible int4-body floor sits *below* the best int4 config but *above* zero, so literal-bf16 byte-exactness is unreachable for every int4 config.

### Comparison vs PR baseline / anchors
- **base_fullhead anchor:** 252.69 TPS / PPL 2.0057 (#553 [`83jiwjr9`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/83jiwjr9)) — the speed/quality anchor; this card adds its **identity** characterization: it does **not** satisfy literal-bf16 #319 (6.76 % flip) but is the lowest-flip int4 config and the operative-identity anchor.
- The PR predicted "base_fullhead FLIPS (~10.9 %, matching #571's body)" → **confirmed it flips**, at **6.76 %** (lower than 10.9 %, explained above by the operative numeric path). The verdict branch is unchanged: bf16-referenced ⇒ int4-unsatisfiable.

### Command / memory / W&B
```
# all local on the idle pod A10G (CUDA_VISIBLE_DEVICES auto-normalized to 0)
python research/validity/base_fullhead_strict319_reference_pin/pin_strict319_reference.py --run       # 128×512 census (run twice; bit-identical)
python research/validity/base_fullhead_strict319_reference_pin/diag_determinism.py                     # steady-state determinism diagnostic
python research/validity/base_fullhead_strict319_reference_pin/pin_strict319_reference.py --finalize   # fold diag → self_det
python research/validity/base_fullhead_strict319_reference_pin/pin_strict319_reference.py --log-wandb
```
- **Peak VRAM ≈ 18.4 GiB** (vLLM alloc: model 9.7 GiB + KV cache 8.62 GiB + 0.04 GiB graphs; A10G 22.5 GiB), consistent with the #553 base_fullhead anchor 18.96 GiB (spec-OFF here drops the drafter).
- **W&B:** [`2u44yaa1`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/2u44yaa1), group `strict319-reference-pin`. (An earlier run `481wyvww` carried a corrupted `self_det_cold_vs_warm` value from a non-idempotent finalize step — now fixed; `2u44yaa1` is authoritative.)

### What happened — honest analysis
The load-bearing ambiguity is **resolved, decisively, against the literal-bf16 reading.** The canonical reference is provably bf16, and base_fullhead — the quality-safe anchor the NO-FIRE census assumes passes the live contract — flips 6.76 % of teacher-forced argmax positions and reproduces the bf16 reference on 0/128 free-running sequences. Because base_fullhead is the *best-case* int4 config and still flips, the int4-body argmax floor is a **universal, irreducible identity floor**: no int4 config can be literally byte-exact vs the bf16 base. The operational consequence is precise and important: every census card's "byte-identical vs bf16" is in force as the **operative / int4-referenced identity** (#407), *not* literal bf16 — and under that operative contract base_fullhead is, by construction (lowest-flip int4 config, full native head), the legitimate quality-safe identity anchor. The NO-FIRE foundation is therefore *clarified*, not broken: the 252.69 frontier is the quality-safe-AND-operative-identity-safe ceiling, with the literal-bf16 interpretation formally ruled unsatisfiable. A bonus finding: the surgical-357 substrate (no `VLLM_BATCH_INVARIANT`) is bit-stable at steady state but not across the ONEGRAPH/LOOPGRAPH cold→warm transition — irrelevant to the served identity contract (which runs warmed) but worth noting for any future literal-byte-exact protocol.

### Suggested follow-ups
- **Pin the operative-identity tolerance.** This card proves literal-bf16 #319 is int4-unsatisfiable; the natural next step is to formalize what the #407 operative contract actually certifies (argmax-class equivalence? bounded logit-margin? the #488 "9/128 bf16-ULP near-ties, 0 semantic" envelope?) so "passes #319" has one unambiguous, measurable definition across all census cards.
- **Re-baseline the int4-body floor on the operative substrate.** #571's 10.9 % was a naive int4_g32 serve; base_fullhead's 6.76 % is the operative-path floor. A small sweep (naive vs operative kernels, g32 vs g128, full vs pruned head) would map the true irreducible identity floor that all configs share.
- **Optional literal-byte-exact mode.** If a future protocol ever needs literal run-to-run byte reproducibility from cold start, setting `VLLM_BATCH_INVARIANT=1` (at the ~48 % matmul-tax cost documented in surgical-357's manifest) would close the cold→warm transient — but it is *not* needed for the served operative contract.
