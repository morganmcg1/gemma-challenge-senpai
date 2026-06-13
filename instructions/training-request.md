<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Training Request Issues

Use this for long-running training jobs on the better-available non-HF cluster.
Students are often limited to one GPU, so speculative decoding drafters,
EAGLE/PARD/MTP heads, QAT/recovery, calibration, and similar model-training
jobs should usually request cluster training instead of grinding slowly in the
student environment.

Official leaderboard scoring still happens through HF Jobs `a10g-small`;
cluster training only produces inference-speed artifacts that must later be
packaged into a complete submission.

Students should open a GitHub issue titled:

```text
Training request: <approach-name>
```

Include:

- **Goal:** what is being trained and why it should improve serving speed.
- **Validity argument:** why this is a drafter, EAGLE/PARD/MTP head,
  QAT/recovery, calibration, or other inference-speed artifact rather than a
  replacement model.
- **PR and branch:** link to the pull request and name the training branch.
- **Code:** commit SHA, entrypoint, exact training command, and environment.
- **Data:** dataset sources, train/held-out split, public benchmark overlap
  check, max epochs, and max steps.
- **Resources:** requested GPU count, expected runtime, disk needed, checkpoint
  cadence, and any cluster/node constraints.
- **W&B:** entity, project, group, run name, required metrics, and artifact or
  artifact-pointer logging plan.
- **Stop conditions:** success threshold, failure threshold, and budget cap.
- **Handoff:** checkpoint output path, validation command, intended
  `submissions/<name>/` packaging plan, expected HF scratch-bucket pointer, and
  whether the final `MODEL_ID` will be a Hub model id or a path inside the
  uploaded submission.

Approval and handoff rules:

- Do not launch long-running training until the advisor or human operator has
  approved the issue.
- Use W&B for metrics, config, code SHA, and artifact pointers, but mirror final
  checkpoints to a stable bucket or filesystem path. Do not rely on W&B alone
  for large checkpoint handoff.
- Return the W&B run URL or ID, node hostname, PID or scheduler job ID, git
  commit SHA, checkpoint path, expected completion time, and exact resume/stop
  commands.
- After training, package the artifact into a complete runnable submission and
  run the local validity gates before requesting any HF `a10g-small` benchmark.
  Do not request or launch an HF job while the submission still points at a
  training-node path such as `/workspace/...`; host the exact validated
  checkpoint or upload it inside the submission first.
