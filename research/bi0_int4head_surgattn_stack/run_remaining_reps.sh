#!/usr/bin/env bash
# PR #797 stack measurement: remaining reps after stack_rep0 (the cold smoke rep).
# Sequential (single GPU, max_num_seqs=1). Each rep = one fresh-server
# run_prevalidate.sh pass (PPL 128 validity gate + 128-prompt decode), exactly the
# #788 harness that produced int4head's 256.74. rep0 per arm is the cold rep (dropped);
# rep1/rep2 are the warm reps we median. CONTROL uses port 8021, STACK uses port 8022.
set -uo pipefail
ROOT=/workspace/senpai/target
LOGDIR="$ROOT/research/bi0_int4head_surgattn_stack"
cd "$ROOT"

run() {  # arm_submission port out_subdir logname
  echo "=== [$(date -u +%H:%M:%SZ)] START $4 ==="
  bash research/_int8head_smoke/run_prevalidate.sh "$1" "$2" "$3" \
    > "$LOGDIR/$4.out" 2>&1
  echo "=== [$(date -u +%H:%M:%SZ)] END $4 rc=$? ==="
}

# CONTROL arm (force-2D ON) — rep0 cold + 2 warm reps
run int4_mtp_bi0_int4head           8021 pr797_control_rep0 control_rep0
run int4_mtp_bi0_int4head           8021 pr797_control_rep1 control_rep1
run int4_mtp_bi0_int4head           8021 pr797_control_rep2 control_rep2

# STACK arm (force-2D OFF, 3D split-KV) — rep0 already done as smoke; 2 warm reps
run int4_mtp_bi0_int4head_surgattn3d 8022 pr797_stack_rep1   stack_rep1
run int4_mtp_bi0_int4head_surgattn3d 8022 pr797_stack_rep2   stack_rep2

echo "=== [$(date -u +%H:%M:%SZ)] ALL REMAINING REPS DONE ==="
