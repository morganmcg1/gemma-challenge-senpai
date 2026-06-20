STUDENT land:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["x76anf6u"],"primary_metric":{"name":"feasibility_verdict_red","value":1},"test_metric":{"name":"single_chain_only","value":1}}

## Results — Tree/EAGLE multi-candidate drafter feasibility probe

### Verdict: **RED** — vLLM 0.22 spec-decode is single-linear-chain only for `gemma4_mtp`

The pinned wheel (`0.22.1rc1.dev307+g3e8afdf78`) — and the stock `0.22.0` that the official benchmark image defaults to — can serve **only a single linear chain** of draft tokens per step for the `gemma4_mtp` proposer family. There is no config surface to express a tree/multi-candidate topology, no tree-attention verify kernel, and the rejection sampler hard-asserts a flat 1-D chain. **The acceptance frontier is closed without a stack change.** Redirect entirely to weight-byte levers (int4head lineage).

Feasibility-only as instructed: code inspection + CPU static dry-run. **No HF job, no retrain, no GPU benchmark.**

---

### 1. SpeculativeConfig surface — scalar chain, no tree topology
- `num_speculative_tokens` is a **scalar `int`**, not a topology — `vllm/config/speculative.py:80` (`num_speculative_tokens: int = Field(default=None, gt=0)`). There is **no** `tree_choices`, `num_branches`, `tree_topology`, or per-level width field anywhere on `SpeculativeConfig`.
- `SpeculativeMethod` (`vllm/config/speculative.py:59-68`) enumerates `ngram, medusa, mlp_speculator, draft_model, suffix, custom_class, eagle, eagle3, extract_hidden_states, <all *_mtp incl gemma4_mtp>, dflash, ngram_gpu`. **Zero** members contain "tree".
- The only "tree" in the whole config is `suffix_decoding_max_tree_depth` (`:167`) — a **retrieval suffix-tree** depth cap for the `suffix`/n-gram prefix-match method, *not* a candidate tree verified in one pass.
- Tell-tale dead language: `use_local_argmax_reduction` docs say it applies to "**non-tree speculation**" (`:137`) — the only nod to trees, with **no** corresponding tree field or method to switch on.

### 2. Proposer + verify path — strictly linear
- `gemma4_mtp` routes to exactly one proposer: `gpu_model_runner.py:589-590` (`use_gemma4_mtp()` → `Gemma4Proposer`). Gate at `config/speculative.py:1054`. No tree variant exists.
- `Gemma4Proposer(SpecDecodeBaseProposer)` (`v1/spec_decode/gemma4.py:31`) runs all decoder layers per draft step **"producing one token"** (module docstring) with `constant_draft_positions=True` (`:47`).
- The base `propose()` builds a **strictly linear autoregressive chain**: the loop `for token_index in range(num_speculative_tokens - 1)` feeds the **single** previous token `input_ids = draft_token_ids_list[-1]` into the next step (`llm_base_proposer.py:588-592`), then `torch.stack(draft_token_ids_list, dim=1)` → shape `[batch_size, num_speculative_tokens]` (`:658-659`; rank-2, no candidate-rank dim). One-token early-exit returns the same `.view(-1, num_speculative_tokens)` (`:533`).
- **Verify kernel is chain-only**: `rejection_sample()` asserts `draft_token_ids.ndim == 1` (`v1/sample/rejection_sampler.py:413`, stock 0.22.0 `:410`) — a flat per-request run indexed by `cu_num_draft_tokens`. The greedy kernel grid is `(batch_size,)` walking each chain against `target_logits.argmax(dim=-1)` — a linear longest-accepted-prefix. **No tree-mask, no parent pointers, no branching.**
- Explicit upstream admission that trees are unimplemented: `llm_base_proposer.py:1493-1497` — `# FIXME: when using tree-based specdec, adjust number of forward-passes according to the depth of the tree` — the `dummy_run` loop does **one forward pass per spec token** (a chain), not per tree level.

### 3. EAGLE / multi-candidate check
- EAGLE/EAGLE-3 exist but give **no tree benefit and are not the shared-KV drafter**: `EagleProposer` (`v1/spec_decode/eagle.py:10-22`) is the *same* `SpecDecodeBaseProposer` with **no** tree override — it produces the identical linear `[batch, K]` chain. EAGLE3 only differs by combining aux hidden states (`propose():457-469`). It is a **separate-weight head** path (no `_setup_gemma4_kv_sharing`), so it would *not* reuse our Q-only shared-KV `gemma4_mtp` drafter (#571, stark #121) — it'd be a different artifact, still single-chain.
- Even the canonically-tree method is collapsed: `MedusaProposer.propose()` returns `torch.stack([logit.argmax(-1) for logit in logits], dim=1)` → `[batch_size, num_heads]` — one argmax token per head, i.e. a **linear chain**, not a candidate tree (`v1/spec_decode/medusa.py:53-55`).
- **No tree-attention / multi-candidate verify kernel ships in the wheel.** A tree would require a custom verify kernel — which #130/#117/#108 already showed hits the 1-wave HBM wall on this M-wide verify-GEMM, so it inherits that ceiling even if built.

### Static dry-run (CPU, 0 GPU) — `research/tree_eagle_feasibility_799/static_dryrun.py`
Introspected the pinned wheel directly and probed *where it rejects* a tree spec:
```
vllm 0.22.1rc1.dev307+g3e8afdf78
tree_named_methods                 : []           (no tree method)
num_speculative_tokens type        : <class 'int'> (scalar, not topology)
candidate_tree_topology_fields     : {}           (none on SpeculativeConfig)
SpeculativeConfig(num_speculative_tokens=[1,2,2]) -> REJECTED  pydantic ValidationError
    "num_speculative_tokens: Input should be a valid integer [input_type=list]"
SpeculativeConfig(..., tree_choices=[[0],[0,0]]) -> REJECTED  pydantic ValidationError
    "tree_choices: Unexpected keyword argument"
propose() returns rank-2 linear chain          : True
rejection_sample asserts draft_token_ids.ndim==1: True
=> single_chain_only = True  => verdict = RED
```
A tree topology is **not expressible**: the engine refuses both a non-scalar `num_speculative_tokens` and any tree-topology kwarg at config-validation time.

---

### What it means
With depth (fern #774, K=6 knee) and the runtime acceptance knob (lawine #792, `CENTROID_TOP_K`) both closed, a **tree drafter was the last "more accepted tokens per weight read" lever** — and it is **not servable on vLLM 0.22 today**. The shipped path already uses the only available shape: a scalar `num_speculative_tokens` (deployed `serve.py:122-129`, K=6/7 via `--speculative-config`). The acceptance family is fully tuned-out on this stack.

### Cost-to-build (if a tree is ever pursued — heavy follow-up, NOT this card)
This is firmly **RED → AMBER-if-forked**, not GREEN. A servable tree drafter needs all of:
1. a new tree proposer (branching `propose`, per-node parent indices) — not just a config flag;
2. a **custom tree-attention verify kernel** + a tree-aware rejection sampler (the current one is 1-D-chain only) — which inherits the #130/#117/#108 1-wave HBM verify-GEMM wall, so the tree's extra candidates may not convert to wall-clock TPS at conc=1 (decode is memory-bandwidth-bound, #781);
3. a config topology surface (`tree_choices`/width schedule) that the wheel does not have.
That is a vLLM **fork bump**, plus a drafter **retrain** to emit tree candidates — a multi-week effort with an uncertain ceiling given the HBM wall. Recommend **not** scoping the retrain; redirect to weight-byte levers (int4head #788 = 256.74 local).

### Commands
```bash
# code inspection: pinned wheel at /tmp/senpai-venvs/5f4c623f772358a2/.../vllm
# static dry-run (CPU only, no GPU/model/serve):
CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
  /tmp/senpai-venvs/5f4c623f772358a2/bin/python \
  research/tree_eagle_feasibility_799/static_dryrun.py
# W&B log (no benchmark):
WANDB_DIR=research/tree_eagle_feasibility_799 \
  /usr/bin/python3 research/tree_eagle_feasibility_799/log_wandb.py
```
- **GPU usage:** none for the verdict (CPU config introspection); peak GPU 0 MiB.
- **W&B run:** `x76anf6u` (group `tree-eagle-feasibility`, job_type `feasibility-probe`).
- **Public evidence used:** none reproduced — this is local code-inspection of the pinned vLLM wheel. Grounded in internal fleet lineage: fern #774 (K-depth knee), lawine #792 (CENTROID_TOP_K null), #781 (decode bandwidth-bound / verify-GEMM 82.8%), #571 + stark #121 (gemma4_mtp Q-only shared-KV), ubel #338 (static-load dry-run pattern), #130/#117/#108 (verify-GEMM 1-wave HBM wall), int4head #788 (256.74 local).

### Suggested follow-ups
- **Acceptance frontier is closed on vLLM 0.22 — stop spending cards here.** Concentrate on weight-byte levers (int4head lineage #788; lawine #796 int4 lm_head byte-floor; fern #797 int4head×surgattn stack).
- If a tree is ever revisited, it is a **fork-bump + retrain epic**, gated on first proving the custom tree verify kernel beats the #130/#117/#108 HBM wall in a microbenchmark — do *that* derisk before any drafter retrain.
- Possible orthogonal probe (out of scope here): whether a newer vLLM (≥ a release that lands tree/EAGLE-tree verify) is even compatible with the org-credit `vllm/vllm-openai` benchmark image — if the image is pinned, a fork is the only route regardless.
