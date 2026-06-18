#!/usr/bin/env bash
# PR #629 — full Option-B quality panel on vLLM 0.22.0 (engine=manifest).
# Orchestrates the whole A/B variant arm against the already-running 0.22.0 spec server:
#   1) greedy panel (GSM8K -> MMLU -> AIME) @ gb6144   [results-greedy-0p22/]
#   2) GPQA-Diamond 10 seeds (sampled) @ gb6144         [results/spec_gpqa_0p22gb6144_s*.json]
#   3) pool GPQA 10 seeds -> 0p22gb6144_pooled.json
#   4) rollup -> panel_0p22.json + verdict
# Sequential (one leg at a time) so each gets the full 16-seq KV cache, matching #624.
set -uo pipefail
cd /workspace/senpai/target
DIR=research/validity/int4_mtp_spec_quality_panel
TOP=$DIR/_panel_0p22_all.status
SEEDS=(12345 23456 34567 45678 56789 67890 78901 89012 90123 13579)
: > "$TOP"
echo "ALL-START $(date -u +%FT%TZ)" | tee -a "$TOP"

echo "[stage 1/4] greedy panel $(date -u +%H:%M:%S)" | tee -a "$TOP"
bash "$DIR/run_greedy_panel_0p22.sh"
echo "  greedy panel rc=$? $(date -u +%H:%M:%S)" | tee -a "$TOP"

echo "[stage 2/4] GPQA 10 seeds gb6144 $(date -u +%H:%M:%S)" | tee -a "$TOP"
bash "$DIR/run_gpqa_genbudget.sh" 6144 0p22gb6144 "${SEEDS[@]}"
echo "  gpqa 10-seed rc=$? $(date -u +%H:%M:%S)" | tee -a "$TOP"

echo "[stage 3/4] pool GPQA $(date -u +%H:%M:%S)" | tee -a "$TOP"
/tmp/eval-serve-venv/bin/python "$DIR/pool_genbudget.py" 0p22gb6144 "${SEEDS[@]}" 2>&1 | tee -a "$TOP"
echo "  pool rc=$? $(date -u +%H:%M:%S)" | tee -a "$TOP"

echo "[stage 4/4] rollup + verdict $(date -u +%H:%M:%S)" | tee -a "$TOP"
/tmp/eval-serve-venv/bin/python "$DIR/rollup_0p22.py" 2>&1 | tee -a "$TOP"
echo "  rollup rc=$? $(date -u +%H:%M:%S)" | tee -a "$TOP"

echo "ALL-DONE $(date -u +%FT%TZ)" | tee -a "$TOP"
