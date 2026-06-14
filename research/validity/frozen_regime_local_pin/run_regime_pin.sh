#!/usr/bin/env bash
# lawine #209 frozen-vs-fresh LOCAL regime-pin driver.
# LOCAL single-A10G serve profiling only: N>=8 fresh LocalServer reloads of the
# DEPLOYED fa2sw_precache_kenyan stack (served files UNCHANGED), decode-only.
# No HF Job / submission / official draw / launch. BASELINE stays 481.53; adds 0 TPS.
# Run under the repo .venv (has wandb). Forwards all flags to run_regime_pin.py.
#
#   bash research/validity/frozen_regime_local_pin/run_regime_pin.sh \
#     --reloads 8 --wandb_group frozen-regime-local-pin \
#     --wandb_name lawine/frozen-regime-local-pin
set -uo pipefail
cd /workspace/senpai/target
export CUDA_VISIBLE_DEVICES=0   # local single in-container GPU (see memory env note)
PY=.venv/bin/python
echo "[driver] $(date -u +%FT%TZ) START frozen-regime-local-pin"
$PY research/validity/frozen_regime_local_pin/run_regime_pin.py "$@"
rc=$?
echo "[driver] $(date -u +%FT%TZ) DONE rc=$rc"
exit $rc
