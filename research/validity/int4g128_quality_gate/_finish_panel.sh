#!/usr/bin/env bash
# PR #579: finish the int4_g128_lmhead 4-eval panel after the in-flight GPQA
# run_mc.sh driver completes. Sequential by design: the borderline GPQA gate
# (>=0.471, greedy 0.4646) is kept isolated from any concurrent batch so its
# 3-seed mean is not perturbed by vLLM int4-Marlin batch-noise. The GPU stays
# continuously busy (no idle gap), just one eval at a time.
#   1) wait for the running GPQA driver (PID arg) to exit
#   2) MMLU-Pro greedy + sampled s0/s1/s2   (run_mc.sh mmlu_pro 2048, #563 harness)
#   3) AIME n=60 greedy maj@1               (#567 protocol, min_tokens=8)
# Server must already be serving int4_g128_lmhead on :PORT. Writes _panel_done.
set -uo pipefail

HERE="/workspace/senpai/target/research/validity/int4g128_quality_gate"
GPQA_PID="${1:-}"
PORT="${2:-8000}"
EVAL_PY="/workspace/senpai/target/.venv/bin/python"
AIME="/workspace/senpai/target/research/downstream_quality_aime/aime_eval.py"
DONE="$HERE/_panel_done"
rm -f "$DONE"

log(){ echo "[finish] $(date -u +%H:%M:%SZ) $*"; }

# 1) wait for the GPQA driver to finish all 3 seeds + greedy anchor
if [[ -n "$GPQA_PID" ]]; then
  log "waiting for GPQA driver pid=$GPQA_PID"
  while kill -0 "$GPQA_PID" 2>/dev/null; do sleep 15; done
  log "GPQA driver exited"
fi

# 2) MMLU-Pro greedy + 3 sampled seeds (2048 tok, matches #563 run_arm.sh)
log "MMLU-Pro start"
bash "$HERE/run_mc.sh" mmlu_pro 2048 "0 1 2" "$PORT" >"$HERE/_mmlu.log" 2>&1
rc=$?; log "MMLU-Pro done rc=$rc"

# 3) AIME n=60 greedy maj@1 (#567 greedy protocol, no-thinking, 3072 cap, mt8)
log "AIME n=60 start"
"$EVAL_PY" "$AIME" \
  --base-url "http://127.0.0.1:${PORT}" \
  --model gemma-4-e4b-it \
  --years 2024,2025 --k 1 \
  --temperature 0.0 --top-p 1.0 --top-k -1 \
  --max-tokens 3072 --min-tokens 8 --seed 1234 --no-thinking \
  --label int4g128_greedy_n60 \
  --out "$HERE/aime_int4g128_min8_n60.json" >"$HERE/_aime.log" 2>&1
rc=$?; log "AIME done rc=$rc"

touch "$DONE"
log "PANEL COMPLETE"
