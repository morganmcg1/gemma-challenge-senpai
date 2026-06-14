# Tree-submission local preflight — runbook

Validates a fully-assembled submission against the scorer's **three hard validity
gates** and emits one **READY / NOT-READY** verdict *before* a human is asked to
authorize the one official shot. Local A10G + CPU only — **no HF Job, no
leaderboard spend, no launch authorization.** Validates VALIDITY (will it score),
not TPS.

| Gate | Question | How (mirrors `hf_bucket_single_job.py`) |
|------|----------|------------------------------------------|
| **A** boot/serve | Engine inits + serves? (#141-class crash) | `LocalServer` brings up `serve.py`; tiny smoke decode + finite-logprob probe |
| **B** PPL ≤ 2.42 | Greedy-exactness held? | official `ppl_endpoint.py` on `ppl_ground_truth_tokens.jsonl` (token-weighted `exp(ΣNLL/Σtok)`) |
| **C** 128/128 | No hang/OOM, full completion? | official `decode_outputs.py`, 128 prompts × output_len 512, `ignore_eos` |

`READY` iff A ∧ B ∧ C (fail-closed: an unrun gate blocks).

## One-line: drop-in preflight for the tree stack (the instant it lands)

```bash
cd target && .venv/bin/python research/tree_submission_preflight/preflight.py --submission submissions/<tree-dir> --server-python /tmp/server-venv/bin/python --wandb-name "denken/tree-preflight" --wandb-group tree-submission-preflight
```

## One-line: harness self-validation (PRIMARY — known-good READY + injected faults NOT-READY)

```bash
cd target && .venv/bin/python research/tree_submission_preflight/preflight.py --self-test --submission submissions/fa2sw_precache_kenyan --server-python /tmp/server-venv/bin/python --wandb-name "denken/preflight-selftest" --wandb-group tree-submission-preflight
```

## Notes

- `--server-python` must point at a venv with the submission's pinned vLLM wheel
  (`/tmp/server-venv` here). The orchestrator python only needs stdlib + the repo
  `scripts.local_validation` package (+ optional `wandb`).
- Exit code: `0` = READY / self-test passed; `1` = NOT-READY / self-test failed.
- `--num-prompts N` (< 128) runs a documented Gate-C subset (necessary, not
  sufficient — re-run full 128 before a launch).
- Results JSON + per-gate artifacts (`ppl_summary.json`, `decode_summary.json`,
  `server.log`) land under `--out-dir` (default `runs/<name>-<stamp>/`).
- W&B (group `tree-submission-preflight`): PRIMARY `harness_self_test_passes`;
  TEST `live_preflight_ready` (logged only on a real drop-in run).
- A `READY` verdict is the **validity leg** of an `Approval request: HF job`
  evidence-line; it does **not** authorize the spend.
