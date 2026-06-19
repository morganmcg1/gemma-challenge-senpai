#!/usr/bin/env bash
# PR #696 -- VENV-HOMOGENIZE the GPQA-D 30-seed pool. Discovery (sensitivity_boot.json):
# the banked seeds 0-9 were COPIED from #692's now-deleted /tmp/land-inspect eval venv
# (all mtimes 00:24:47), while seeds 10-29 were freshly run this session on the surviving
# /tmp/eval-serve-venv. The cross-boot repro of seed 0 got 100/198 vs the banked 102/198
# (reproduced=false by exact match), and the banked batch mean sits ~1pp above the new-venv
# batch (Welch t=1.21, underpowered) -> a possible inspect_evals-version / extractor artifact
# that FLIPS the GPQA seed-mean lens (mixed clears 0.471 @0.472; new-venv-only straddles @0.465).
#
# Fix: re-run seeds 0-9 on the SAME /tmp/eval-serve-venv the new seeds (and the AIME leg, and
# lawine #693) use, so the full 30-seed pool is engine-homogeneous and the joint {GPQA,AIME}
# verdict is engine-consistent. Banked land-inspect files are PRESERVED for the paired
# venv-effect comparison. Runs AFTER the AIME chain (aime2) frees the single A10G.
# LOCAL, analysis_only, NO HF JOB, NO FIRE.
set -u
cd /workspace/senpai/target/research/validity/int4_quality_31basis_confirm

AIME2_PID="$(cat _chain_aime2.pid 2>/dev/null || echo '')"
echo "[renew09] $(date +%H:%M:%S) waiting for aime2 pid=${AIME2_PID} to finish (AIME has GPU priority)"
if [ -n "${AIME2_PID}" ]; then
  while kill -0 "${AIME2_PID}" 2>/dev/null; do sleep 30; done
fi
echo "[renew09] $(date +%H:%M:%S) aime2 exited; waiting for GPU memory to free (<4000 MiB)"
used=99999
for _ in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ "${used:-99999}" -lt 4000 ]; then break; fi
  sleep 10
done
echo "[renew09] $(date +%H:%M:%S) GPU used=${used}MiB; backing up banked land-inspect seeds 0-9"
mkdir -p results_gpqa/_banked_landinspect
for s in 0 1 2 3 4 5 6 7 8 9; do
  f="results_gpqa/bf_gpqa_sampled_mt8_s${s}.json"
  if [ -f "$f" ]; then cp -p "$f" "results_gpqa/_banked_landinspect/" && rm -f "$f"; fi
done
echo "[renew09] $(date +%H:%M:%S) launching eval-serve re-run of seeds 0-9 (fresh, homogeneous pool)"
exec /tmp/senpai-venvs/5f4c623f772358a2/bin/python -u run_gpqa_seeds.py \
  --seeds 0,1,2,3,4,5,6,7,8,9
