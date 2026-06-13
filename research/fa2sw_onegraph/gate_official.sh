#!/usr/bin/env bash
# Decision gate: offline greedy-identity of each lever vs base on the OFFICIAL
# ShareGPT decode-audit prompts (sourced from the served base capture), in the
# clean deterministic regime (sequential, prefix-cache off). Also reports
# single-stream TPS per variant. Compares with the official verifier.
set -u
ROOT=/workspace/senpai/target
RES="$ROOT/research/fa2sw_onegraph"
PY="$ROOT/.venv/bin/python"
PROMPTS="$RES/serve_runs/base/decode_outputs.jsonl"   # official audit prompts
VERIFY="$ROOT/official/main_bucket/shared_resources/gemma_greedy_identity_verifier_flowian-powers/check_greedy_identity.py"
OUT="$RES/runs_official"
NPROMPTS=128
GENTOK=256
LOG="$OUT/gate.log"

mkdir -p "$OUT"
: > "$LOG"
echo "=== gate_official start $(date -u +%H:%M:%S)  prompts=$NPROMPTS gentok=$GENTOK ===" | tee -a "$LOG"

for V in base fa2sw onegraph both; do
  echo "--- variant=$V $(date -u +%H:%M:%S) ---" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=0 "$PY" "$RES/ablate.py" \
    --variant "$V" --outdir "$OUT/$V" \
    --decode-prompts "$PROMPTS" --n-identity "$NPROMPTS" --gen-tokens "$GENTOK" \
    --sequential --no-prefix-cache --skip-ppl \
    --tps-tokens 256 --tps-repeats 3 >> "$LOG" 2>&1
  echo "[$V] ablate rc=$?" | tee -a "$LOG"
done

echo "=== VERDICTS (base = reference) ===" | tee -a "$LOG"
for V in fa2sw onegraph both; do
  echo "--- $V vs base ---" | tee -a "$LOG"
  "$PY" "$VERIFY" --reference "$OUT/base/decode_outputs.jsonl" \
                  --candidate "$OUT/$V/decode_outputs.jsonl" >> "$LOG" 2>&1
  echo "[$V] verifier exit=$?" | tee -a "$LOG"
done
echo "=== gate_official DONE $(date -u +%H:%M:%S) ===" | tee -a "$LOG"
