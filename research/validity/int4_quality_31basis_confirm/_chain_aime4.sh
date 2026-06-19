#!/usr/bin/env bash
# PR #696 -- AIME think-leg TIGHTENING (seeds 3..7 -> 8 total).
#
# Why: the 3-seed think #31 pooled point is 0.3722 (67/180, 79.75% of base 0.4667) -- below the
# 0.420 gate on the POINT, but seed-2 drew high (0.4333) so the 3-seed pooled Wilson HI ~0.445
# straddles ABOVE the gate. That is not decision-grade for an AIME-REAL call. lawine #693 (relayed
# by advisor) had int4-body AIME #31 = 0.3467 with Wilson-hi 0.4022 < 0.420; my point is consistent
# within seed noise but my own CI is too wide at K=3. Adding seeds 3,4,5,6,7 (-> N=480) brings the
# pooled Wilson HI robustly below 0.420 IF the rate holds ~0.37, making the AIME leg's own CI as
# decision-grade as the GPQA-D 30-seed leg. This is the "enough seeds for a Wilson LB" instruction,
# not scope creep (same single experiment: int4-body AIME think #31 sampled).
# LOCAL, analysis_only, NO HF JOB, NO submission, NO FIRE.
set -u
cd /workspace/senpai/target/research/validity/int4_quality_31basis_confirm
PY=/tmp/senpai-venvs/5f4c623f772358a2/bin/python

PREV_PID="$(cat _chain_aime3.pid 2>/dev/null || echo '')"
echo "[aime4] $(date +%H:%M:%S) waiting for aime3 chain pid=${PREV_PID} (nothink secondary) to exit"
if [ -n "${PREV_PID}" ]; then
  while kill -0 "${PREV_PID}" 2>/dev/null; do sleep 20; done
fi
echo "[aime4] $(date +%H:%M:%S) aime3 exited; waiting for GPU memory to free (<4000 MiB)"
used=99999
for _ in $(seq 1 120); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ "${used:-99999}" -lt 4000 ]; then break; fi
  sleep 10
done
echo "[aime4] $(date +%H:%M:%S) GPU used=${used}MiB; THINK #31-sampled tighten seeds 3,4,5,6,7"
"$PY" -u run_aime.py --thinking --sampled-seeds 3,4,5,6,7
echo "[aime4] $(date +%H:%M:%S) AIME THINK TIGHTEN DONE"
