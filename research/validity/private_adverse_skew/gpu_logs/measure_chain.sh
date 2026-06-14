#!/usr/bin/env bash
# Sequential GPU measurement chain for PR #176 new axes (single A10G).
# Waits for the already-running multilingual probe, then runs math + longctx.
set -u
cd /workspace/senpai/target
VENV=/tmp/senpai-venvs/5f4c623f772358a2/bin/python
PROBE=scripts/validity/private_gap_probe.py
GLOG=research/validity/private_adverse_skew/gpu_logs
ML_PID="$1"

echo "[chain] waiting for multilingual report (pid $ML_PID) ..."
until [ -f research/validity/private_gap_probe/native_multilingual/report.json ] || ! kill -0 "$ML_PID" 2>/dev/null; do sleep 20; done
if [ ! -f research/validity/private_gap_probe/native_multilingual/report.json ]; then
  echo "[chain] ABORT: multilingual produced no report (process died)."; exit 1
fi
echo "[chain] multilingual done; launching math ..."
$VENV $PROBE --submission submissions/fa2sw_precache_kenyan \
  --private data/private_proxy_native_math.json \
  --out-dir research/validity/private_gap_probe/native_math --no-decompose \
  --wandb-group descent-private-adverse-skew --wandb-name stark/private-gap-native-math \
  > $GLOG/full_math.log 2>&1
echo "[chain] math exit=$?; launching longctx ..."
$VENV $PROBE --submission submissions/fa2sw_precache_kenyan \
  --private data/private_proxy_native_longctx.json \
  --out-dir research/validity/private_gap_probe/native_longctx --no-decompose \
  --wandb-group descent-private-adverse-skew --wandb-name stark/private-gap-native-longctx \
  > $GLOG/full_longctx.log 2>&1
echo "[chain] longctx exit=$?"
echo "[chain] ALL_MEASUREMENTS_DONE"
