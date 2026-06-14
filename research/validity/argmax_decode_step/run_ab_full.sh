#!/usr/bin/env bash
# PR #180 realized-step measurement: full paired wall_tps A/B on the canonical
# bench (128 prompts x 512 tokens), N=3 per arm, control vs ARGMAX_ONLY_DECODE=1.
# Same throwaway submission (fa2sw_argmax_decode) for both arms; candidate sets
# ARGMAX_ONLY_DECODE=1. Writes per-run decode jsonls (for the 128/128 token
# identity proof) + paired_ab.json (wall_tps ratio = step ratio under E[T] parity).
set -euo pipefail
cd "$(dirname "$0")/../../.."   # -> target/

OUT_DIR="research/validity/argmax_decode_step/ab_full"
PY=.venv/bin/python

"$PY" scripts/profiler/paired_tps_ab.py \
  --baseline fa2sw_argmax_decode \
  --candidate fa2sw_argmax_decode \
  --candidate-env ARGMAX_ONLY_DECODE=1 \
  --baseline-label argmax_off \
  --candidate-label argmax_on \
  --n 3 \
  --num-prompts 128 \
  --output-len 512 \
  --seed 1 \
  --out-dir "$OUT_DIR" \
  --wandb-group argmax-decode-step-realization \
  --wandb-name lawine/argmax-decode-step-realization

echo ""
echo "=== token identity diff (per run, control vs patched) ==="
for r in 00 01 02; do
  C="${OUT_DIR}/argmax_off/decode/run${r}.jsonl"
  P="${OUT_DIR}/argmax_on/decode/run${r}.jsonl"
  if [[ -f "$C" && -f "$P" ]]; then
    echo "--- run ${r} ---"
    "$PY" research/validity/argmax_decode_step/token_identity_diff.py \
      "$C" "$P" --json-out "${OUT_DIR}/token_identity_run${r}.json" --show 4 \
      | tail -2 || true
  fi
done
