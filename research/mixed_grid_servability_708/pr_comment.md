STUDENT land:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["8yf0622s"],"primary_metric":{"name":"mixed_grid_servable","value":1},"test_metric":{"name":"mixed_grid_opbench_tps","value":126.68}}

## Results

**Verdict: `MIXED_GRID_SERVABLE` (per-layer group_size IS supported) — but the ubel #700 subset is only PARTIALLY servable: 40/48 standalone modules ship cleanly, the 8 fused-shard attention modules (q/k/v) do NOT.** The selective-g32 recovery is **cheap in speed** (≤0.17 TPS, far from the full-g32 4.90 TPS), so the recovery is *not* the expensive full-g32 route. The catch is in *which modules ship*, not in the speed.

W&B run: **`8yf0622s`** (group `mixed-grid-marlin-servability-land`), verdict `MIXED_GRID_SERVABLE_STANDALONE_FUSED_BLOCKED`.
`analysis_only=1, official_tps=0, no_hf_job=1, fires=0` — no HF Job, no submission, no served-file change; locked `int4_g128_lmhead`@126.378 untouched.

### 1. Source read (PRIMARY metric `mixed_grid_servable=1`) — decisive

The served path is **compressed-tensors W4A16 → Marlin** (not `GPTQMarlin*`, which doesn't exist in this build). Read in `vllm 0.22.1rc1.dev307+g3e8afdf78`:

- **group_size is read PER-LAYER, not model-level.** `compressed_tensors_wNa16.py:55` each scheme carries its own `self.group_size`; `:104` it flows into a per-layer `MPLinearLayerConfig`; `:127` `scales_and_zp_size = input_size // group_size` is per-layer; `:212` a per-layer kernel is built. The compressed-tensors **format** expresses this via `config_groups` (N groups, each with its own `weights.group_size`); the locked build already uses two (`build_quant.py:161-173`: `group_0` body, `group_1` head, different group_size). Dispatch is per-layer via `find_matched_target` (`compressed_tensors.py:885`). **⇒ a per-module heterogeneous group_size IS expressible and IS dispatched per-layer.**
- **BUT fusion blocks the subset's 8 attention modules.** vLLM fuses q/k/v → `qkv_proj` (`gemma3n.py:312` `QKVParallelLinear`, packed mapping `:1094`), gate/up → `gate_up_proj`. A fused layer gets exactly **one** scheme. `_match_fused_layer` (`utils.py:239`) returns the **first shard's** target if *all* shards match *something* — it does **not** require them to match the **same** scheme, and `should_ignore_layer` (`:88`) only guards the ignore boundary. So q=g32 / k,v=g128 in one qkv block → either silent mis-assignment or (more concretely) a **weight-load shape mismatch** (the g32 layer expects `input//32` scale groups; the on-disk k/v g128 scales have `input//128`). The single fused qkv scale tensor cannot be serialized with mixed per-shard group sizes.
- `per_layer_input_gate` is a **standalone `ReplicatedLinear`** (`gemma3n.py:485`, not in the packed mapping) → independently g32-servable.

**⇒ The 40 `per_layer_input_gate` (54% of the subset's params) ship as isolated g32. The 8 attention modules (46% of subset params) do NOT — they need WHOLE-qkv-block promotion (the whole layer's q+k+v → g32), which drags in k/v that ubel #700 did not target. So the SERVED attn recovery ≠ the fake-quant 48-isolated-module recovery.** Full evidence + line refs in the W&B artifact `mixed_grid_source_read_708` / `research/mixed_grid_servability_708/SOURCE_READ.md`.

### 2. Op-bench (TEST metric `mixed_grid_opbench_tps`) — tax realized, no occupancy penalty

Extended my #707 apparatus (exact served `apply_gptq_marlin_linear`, paired, L2-cold CUDA-graph, M=1, seed 707, 50 iters/round, median of 15 rounds for bodies / 30 paired rounds per-module, 95% CI) to the **real 42-layer fused census** (corrects #707's 37-layer/qkv-only model: now incl. KV-sharing q-only layers + `per_layer_input_gate`/`per_layer_projection`/`per_layer_model_projection`; `body_frac_of_anchor` 51.6% → 70.4%).

| config | op-bench TPS | tax vs g128 | note |
|---|---|---|---|
| all-g128 anchor (#707) | **126.75** | 0 | AR rung |
| all-g32 full (measured) | **121.27** | −5.48 | refines #707's 121.836; full tax 6.43% (Δ356.74±0.48 µs) |
| denken #706 projection | 126.27 | −0.48 | linear byte-law target |
| **mix servable (40 PLIG g32)** | **126.68** | **−0.07** | **the ship path — speed-free** |
| fake-quant-48 ideal (NOT servable) | 126.58 | −0.17 | 40 PLIG + 3q+3k+2v isolated; **realizes/beats** the 126.27 projection |
| whole-qkv route (3 blocks) | 126.71 | −0.04 | servable attn route |
| whole-qkv route (8 blocks) | 126.64 | −0.11 | servable attn route |

All selective configs land in a **0.04–0.17 TPS band** (at the body-to-body measurement-noise floor, ~0.1 TPS) — the linear projection is realized, and the in-body tax is if anything **smaller** than the isolated-microbench additive prediction (serv Δ 4.24 µs measured vs 8.10 µs additive). **No per-module kernel-occupancy / lost-fusion penalty** — each Marlin GEMM is its own launch, so mixing g32/g128 across modules is additive. Per-module taxes confirm: small modules have *lower* relative g32 tax (`per_layer_input_gate` 1.09%, k/v ~1.9%) than the BW-bound big ones (`gate_up` 8.40%, `down` 7.94%) — the opposite of a launch-overhead anomaly, so the byte-law projection is conservative for the subset.

### 3. Greedy-identity / determinism spot-check (servable → ran)

`MIXED_GRID_DETERMINISTIC` — the exact served kernel is byte-deterministic on repeated calls at **both** g32 and g128 for every body shape, **and** an interleaved g32/g128 sequence is byte-deterministic across repeated runs (default `VLLM_MARLIN_USE_ATOMIC_ADD=0`). Mixed-grid kernel selection introduces **no** nondeterminism — the strict-#319 reproducibility precondition holds. (This is the determinism precondition, not the 128/128 launch gate, which is out of scope.)

### Command

```bash
# op-bench (dev307 vLLM venv) -> JSON
CUDA_VISIBLE_DEVICES=0 /tmp/senpai-venvs/5f4c623f772358a2/bin/python \
  research/mixed_grid_servability_708/mixed_grid_opbench.py
# determinism spot-check
CUDA_VISIBLE_DEVICES=0 /tmp/senpai-venvs/5f4c623f772358a2/bin/python \
  research/mixed_grid_servability_708/greedy_determinism_check.py
# W&B log (base python; dev307 ships a broken wandb stub)
python3 research/mixed_grid_servability_708/log_wandb.py
```

**Peak memory:** HBM-only op-bench (no checkpoint build, disk-safe by construction); transient weight builds ≈0.7 GB resident on the 22 GB A10G, no OOM. Disk untouched (~9.8 GB free maintained).

### What happened

The three cards resting on the "speed-free selective recovery" premise (ubel #700 localization, ubel #702 quality, denken #706 Pareto) assumed a heterogeneous mixed-grid Marlin model serves. **It does — for standalone modules — and the speed tax is realized at/below the projection (≤0.17 TPS).** So the recovery is *not* expensive in speed; full-g32's 4.90 TPS is not the floor. **The real finding is a granularity mismatch the projection hid:** ubel #700's subset targets 8 *individual* q/k/v modules, but vLLM serves them *fused*. They can't ship as isolated g32; the servable route promotes whole qkv blocks (cheap in speed, but quantizes finer than the targeted 8 — it pulls in untargeted k/v). So **ubel #702's fake-quant quality number, measured on the isolated-48 config, is not the quality of any servable model for the attention part.** The clean servable recovery is PLIG-dominant (40 modules); the attention recovery requires re-pricing quality on the whole-qkv-promoted config.

### Suggested follow-ups

- **Re-anchor the recovery on what actually ships.** Two clean servable variants: (a) **PLIG-only** (40 standalone modules, exactly speed-free, 126.68); (b) **PLIG + whole-qkv-block promotion** for the attention layers (126.64–126.71). Ask ubel #702 to evaluate AIME quality on **these two served configs**, not the isolated-48 fake-quant — the attention part's quality must be measured on the whole-qkv config it will actually serve.
- **Cheap servability win:** the per-layer dispatch means a `group_2` config_group (targets = the 40 `per_layer_input_gate`, group_size 32) is a trivial `build_quant.py` delta — no kernel change, no #655-class work — if (a) clears the AIME gate.
- The full-g32 floor refines to **121.27** (was 121.836) on the real 42-layer census; denken #706's Pareto can adopt the realized 0.04–0.17 TPS band (replacing the linear projection) and the corrected `body_frac` 70.4%.
