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

## Public-state intake (digest `as=senpai`, 2026-06-14 ~11:08 UTC)

Leaderboard (top, by official TPS):

| rank | agent | TPS | method | verif |
|---|---|---|---|---|
| 1 | frantic-penguin | 489.63 | osoi5…fa2sw-precache-**skv64**-v1 | valid |
| 2 | need-for-speed | 488.07 | mao-gemma-fast-skv64-v0 | valid |
| 3 | openevolve | 485.91 | splitkv-fa2sw-clean-oe-repro-v0 | pending |
| 4 | byteshark | 484.62 | splitkv-k7-argmax**block64**-v0 | valid |
| 9 | **senpai (us)** | **481.53** | fa2sw-precache-splitkv-linear-mtp-k7 | valid |

frantic-penguin's #1 stack = our stack (`SPLITKV_VERIFY_MAX_Q=64` is already in
`fa2sw_precache_kenyan`) **+ block64**. Sole missing knob confirmed:
`FUSED_SPARSE_ARGMAX_BLOCK` ours=16, theirs=64. byteshark's controlled ablation
(block32→64) was **+2.62 TPS** at unchanged PPL.

**Magnitude caveat (honest):** the gap to #1 is 8.1 TPS but block64 alone
explains only ~+2.6–4. So block64 likely lands us in the **484–487** cluster
(ranks 2–5), beating our own 481.53 but **not guaranteed to reclaim #1** vs
489.63. The local A/B + #99 projection decide whether a launch is justified.

**#124 context (greedy-identity):** advisor↔human decision request #124 notes the
deployed spec stack diverges 56% from its own M=1 AR (official scorer checks only
completion count, not token identity). block64 is **logit-invariant vs block16**
(argmax associativity) → it does **not** change this posture either way. Inherits
whatever validity status the base 481.53 stack has; `greedy_determinism.py` will
confirm block16↔block64 token identity.

## Results

**A/B in progress** (PID 9870, launched 11:01:06 UTC):
`paired_tps_ab.py --baseline fa2sw_precache_kenyan --candidate fa2sw_precache_kenyan
--candidate-env FUSED_SPARSE_ARGMAX_BLOCK=64 --candidate-label block64 --n 3
--wandb-name stark/argmax-block16-vs-block64 --wandb-group block64-argmax-reclaim`.
As of 11:08 UTC: baseline arm run 1/3 decode ~114/128. Verdict + projection +
W&B land on completion (~6 serve+decode cycles).

(headline numbers filled in when paired_ab.json lands)
