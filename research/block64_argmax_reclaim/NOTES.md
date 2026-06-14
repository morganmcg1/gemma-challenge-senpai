# block64 argmax-reclaim (PR #137)

**Hypothesis:** `FUSED_SPARSE_ARGMAX_BLOCK` 16→64 in `fa2sw_precache_kenyan`
reclaims public #1. It controls the tile width `block_selected` of the
partial-block fan-in reduction over the sparse vocab candidates in the fused
sparse-argmax verify kernel (`sitecustomize.py:831`,
`num_blocks = cdiv(selected_count, block_selected)`). Larger block ⇒ fewer
reduction rounds. Argmax is associative so the selected token is identical
regardless of tiling ⇒ greedy/PPL unchanged by construction.

## Plan (LOCAL only — no HF launch without human approval)

1. Paired `wall_tps` A/B (`scripts/profiler/paired_tps_ab.py`), serve-time env
   override, N=3 median, 128×512 seed=1 workload.
   - Run 1: baseline block16 (manifest default) vs candidate block64 — headline.
   - Run 2/3: reuse baseline, candidates block32 / block128 — map the curve.
2. Validation gates: `local_prevalidate.py` (PPL ≤ 2.42), `private_gap_probe.py`,
   `greedy_determinism.py`.
3. Edit manifest to best block, upload submission to gemma-senpai bucket.
4. Open approval issue (do NOT launch HF job).

W&B group: `block64-argmax-reclaim`.

## Results

(filled in as runs complete)
