#!/usr/bin/env bash
set -uo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=0
echo "[dl] start $(date '+%H:%M:%S')"
echo "[dl] === base ==="
hf download google/gemma-4-E4B-it-qat-w4a16-ct --quiet 2>&1 | tail -5
echo "[dl] base rc=$? $(date '+%H:%M:%S')"
echo "[dl] === drafter ==="
hf download google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant --quiet 2>&1 | tail -5
echo "[dl] drafter rc=$? $(date '+%H:%M:%S')"
echo "[dl] DONE $(date '+%H:%M:%S')"
