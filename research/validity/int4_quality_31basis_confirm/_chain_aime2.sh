#!/usr/bin/env bash
# PR #696 -- AIME leg, REDESIGNED for protocol-consistency (replaces the killed _chain_aime.sh).
#
# Why redesigned: aggregate_aime.py's gate (0.420 = 0.90 x 0.4667) is a THINKING-enabled base
# denominator, but the old run_aime.py measured NO-THINKING (~0.10 floor). Numerator/denominator
# protocol mismatch would spuriously show AIME failing at ~29%. Now protocol is explicit and the
# numerator is matched to its base:
#   * think regime   -- the cited 0.4667/0.3500 wall lives here. greedy decomposes the wall on the
#                       body-isolation arm (g32 body + bf16 head) vs the cited g128-both-int4 0.3500;
#                       #31-sampled tests basis recovery. gate 0.420.
#   * nothink regime -- #580 floor cross-check (base 0.10, int4-body greedy 0.1167 already banked,
#                       so we only add #31-sampled here). gate 0.090.
#
# LOCAL, analysis_only, NO HF JOB, NO FIRE. Single A10G. Idempotent (run_aime.py skips existing).
set -u
cd /workspace/senpai/target/research/validity/int4_quality_31basis_confirm
PY=/tmp/senpai-venvs/5f4c623f772358a2/bin/python

CHAIND_PID="$(cat _chainD.pid 2>/dev/null || echo '')"
echo "[aime2] $(date +%H:%M:%S) waiting for chainD pid=${CHAIND_PID} (GPQA seeds 23-29) to exit"
if [ -n "${CHAIND_PID}" ]; then
  while kill -0 "${CHAIND_PID}" 2>/dev/null; do sleep 30; done
fi
echo "[aime2] $(date +%H:%M:%S) chainD exited; waiting for GPU memory to free (<4000 MiB)"
used=99999
for _ in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ "${used:-99999}" -lt 4000 ]; then break; fi
  sleep 10
done
echo "[aime2] $(date +%H:%M:%S) GPU used=${used}MiB; thinking smoke (2 problems) to gate the sweep"
if ! "$PY" -u run_aime.py --thinking --smoke; then
  echo "[aime2] $(date +%H:%M:%S) THINKING SMOKE FAILED -- aborting"
  exit 1
fi
echo "[aime2] $(date +%H:%M:%S) smoke OK; THINK leg: greedy (wall decomposition) + #31-sampled 0,1,2"
"$PY" -u run_aime.py --thinking --greedy --sampled-seeds 0,1,2
echo "[aime2] $(date +%H:%M:%S) THINK leg done; NOTHINK cross-check: #31-sampled 0,1,2 (base 0.10)"
"$PY" -u run_aime.py --sampled-seeds 0,1,2
echo "[aime2] $(date +%H:%M:%S) AIME ALL DONE"
