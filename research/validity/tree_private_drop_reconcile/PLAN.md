<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# PR #156 — Pin the tree's TRUE private drop (reconcile 4.3% / 11.3% / 19.6%)

LOCAL single-A10G profiling/analysis only. No training, no HF Job, no submission,
no served-file change. Greedy/PPL untouched, BASELINE unchanged (481.53).
`--wandb_group tree-private-drop-reconcile`.

## Reconciliation thesis (to verify with a fresh sglang-path ladder measurement)

The three "private drop" numbers are NOT the same measurement. They differ on two
independent axes — **harness path** and **proxy-vs-real difficulty**:

| # | value | what it actually is | protocol | distribution |
|---|-------|---------------------|----------|--------------|
| GT-flagship | **4.3%** | organizer's real public→private **TPS** drop (481.53→460.85, VERIFIED 2026-06-13 23:04Z) | sglang `vllm-chat` **scored** bench (`hf_bucket_single_job.run_benchmark`), `ignore_eos`, out 512 | **real** private set |
| sglang-probe (kanna #44) | **11.3%** | precache-neutral **TPS** distribution gap (public_cold 418.4 → private_rerun 371.0); E_accept 4.06→3.565 | **same** sglang `vllm-chat` scored protocol | hard chat **proxy** |
| official-decode (stark #151) | **19.6%** | per-position **E[T]** drop (public 3.844 → proxy 3.090) | `decode_outputs.py` **audit** path: client-side chat template, `/v1/completions`, own prompt subset | hard chat **proxy** |

- **19.6% → 11.3%** is the **harness-path** gap: `decode_outputs.py` is the unscored
  greedy-identity *audit* pass, not the leaderboard's scored protocol. It systematically
  under-reads E[T] (3.844 vs 4.06 public; 3.090 vs 3.565 proxy) — the gap widens on the
  proxy → inflated 19.6% vs the faithful ~11–12%. Prime suspects (PR step 1): client- vs
  server-side chat templating and prompt-subset selection (`ignore_eos`/`max_tokens` are
  matched in both).
- **11.3% → 4.3%** is **proxy difficulty**: `data/private_proxy_sharegpt.json` is a
  deliberately-hard chat tail (~2.6× the real private drop). The organizer's real private
  set is much closer to public. BASELINE: "the ground-truth 4.3% is now the number to
  calibrate every private-gap probe against."

**=> Pinned organizer-matching protocol = sglang `vllm-chat` scored bench; calibration
anchor = 4.3% (organizer's real LINEAR drop). 19.6% is a harness artifact to discard.**

## Plan

1. [liveness] CPU self-test of the banked tree DP + W&B run in this group. (this commit)
2. [GPU] Re-measure the per-position acceptance ladder under the **sglang scored**
   protocol (public_cold vs private_rerun) on the deployed linear `fa2sw_precache_kenyan`
   stack; parse vLLM's per-position lines from the server logs. Confirms ~11–12% (not
   19.6%) under the organizer-matching protocol AND gives the ladder **shape** the tree
   amplifies (the new data #156 adds over #151).
3. [propagate] Feed the pinned ladder into `tree_private_acceptance_gap.py`, calibrated
   to 4.3%. Report `tree_private_tps_proj_pinned` (descent-only + both-bugs),
   `descent_only_clears_500_pinned` / `both_bugs_clears_500_pinned`, with band.
4. [self-validate] `harness_pin_reproduces_flagship_4p3` (PRIMARY), `tree_private_drop_pct_pinned` (TEST).
5. Reconciliation JSON under this dir + PR report.
