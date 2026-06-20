#!/usr/bin/env bash
# PR #805 Step 5 — prove the published private Hub repo is remote-loadable + dispatches
# correctly (PLE=bf16/cuBLAS, body+head+sibling=int4/Marlin) via the SAME verify_dispatch.py
# used on the local build. This closes the only unproven fire-readiness link: remote load.
set -uo pipefail
cd /workspace/senpai/target
P=/tmp/senpai-venvs/20f658587e8a6643/bin/python
REPO="gemma-challenge/gemma-4-e4b-it-int4-mtp-bi0-int4head-pledequant"
LOG=research/int4head_ple_dequant/hub_load_smoke.log
: > "$LOG"

echo "[smoke] snapshot_download $REPO ..." | tee -a "$LOG"
SNAP=$("$P" - <<PY 2>>"$LOG"
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="$REPO", repo_type="model")
print(p)
PY
)
rc=$?
echo "[smoke] snapshot_download rc=$rc path=$SNAP" | tee -a "$LOG"
if [ $rc -ne 0 ] || [ -z "$SNAP" ]; then echo "[smoke] DOWNLOAD FAILED" | tee -a "$LOG"; exit 2; fi

# config.json byte-identical to the locally-validated build? (dispatch is decided by config.ignore)
echo "[smoke] config.json local-vs-hub diff:" | tee -a "$LOG"
if diff -q /workspace/gemma_build/bi0_int4head_pledequant/config.json "$SNAP/config.json" >>"$LOG" 2>&1; then
  echo "[smoke] config.json IDENTICAL (hub == local build)" | tee -a "$LOG"
else
  echo "[smoke] config.json DIFFERS — investigate" | tee -a "$LOG"
fi

echo "[smoke] running verify_dispatch.py against HUB snapshot ..." | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 BUILD="$SNAP" \
  "$P" research/int4head_ple_dequant/verify_dispatch.py >>"$LOG" 2>&1
vrc=$?
echo "[smoke] verify_dispatch rc=$vrc" | tee -a "$LOG"
echo "[smoke] DONE" | tee -a "$LOG"
exit $vrc
