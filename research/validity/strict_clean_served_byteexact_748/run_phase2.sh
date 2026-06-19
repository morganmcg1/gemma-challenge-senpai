#!/usr/bin/env bash
# PR #748 phase 2: the analyses that need the GPU after the eager mechanism arms free it.
# File-based wait (bi1_spec1_eager/arm_summary.json is written only at eager-spec completion) ->
# no pgrep self-match. Then three GPU-serialized steps:
#   1. tau=0.3 gap_probe + PPL(BI=1)  -- advisor-mandated benign-tie reframe (self-consistency)
#   2. PPL(BI=0)                       -- BI-neutrality of PPL (quality unaffected by reduction order)
#   3. bi1_spec0_rep determinism floor -- proves 21/128 is a real spec-vs-AR reduction divergence,
#                                         not run-to-run nondeterminism of the served greedy stack
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/tmp/senpai-venvs/20f658587e8a6643/bin/python
RUNS="$HERE/runs"

echo "[phase2] waiting for eager arms (bi1_spec1_eager/arm_summary.json) $(date -u +%H:%M:%SZ)"
until [ -f "$RUNS/bi1_spec1_eager/arm_summary.json" ]; do sleep 30; done
echo "[phase2] eager arms done; settling GPU $(date -u +%H:%M:%SZ)"
sleep 25  # let the eager server release GPU memory before we grab 0.9 util

echo "[phase2] STEP1 selfconsist_ppl --bi 1 (gap_probe + PPL) $(date -u +%H:%M:%SZ)"
CUDA_VISIBLE_DEVICES=0 "$PY" "$HERE/selfconsist_ppl.py" --bi 1 \
  > "$RUNS/selfconsist_ppl_bi1.out" 2>&1
echo "[phase2] STEP1 exit=$? $(date -u +%H:%M:%SZ)"

echo "[phase2] STEP2 selfconsist_ppl --bi 0 (PPL only) $(date -u +%H:%M:%SZ)"
CUDA_VISIBLE_DEVICES=0 "$PY" "$HERE/selfconsist_ppl.py" --bi 0 --do-gapprobe 0 \
  > "$RUNS/selfconsist_ppl_bi0.out" 2>&1
echo "[phase2] STEP2 exit=$? $(date -u +%H:%M:%SZ)"

echo "[phase2] STEP3 determinism floor bi1_spec0_rep $(date -u +%H:%M:%SZ)"
"$PY" "$HERE/run_arm.py" --bi 1 --spec 0 --tag bi1_spec0_rep --port 8030 \
  --n-prompts 128 --output-len 512 --startup-timeout 600 \
  > "$RUNS/bi1_spec0_rep.out" 2>&1
echo "[phase2] STEP3 exit=$? $(date -u +%H:%M:%SZ)"

echo "[phase2] ALL DONE $(date -u +%H:%M:%SZ)"
