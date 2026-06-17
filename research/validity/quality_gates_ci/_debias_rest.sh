#!/usr/bin/env bash
# PR #590 -- finish the MMLU-Pro truncation de-bias (server already up at max-model-len 6144).
# 1) splice seeds 3,4,5 (re-run their truncated ids at max_tokens=4096, merge into N=2000).
# 2) seed 2 had no 2048 base run (ENOSPC casualty) -> run it FRESH full-N=2000 at max_tokens=4096.
#    This is the 5th de-biased seed AND a from-scratch cross-check of the splice (its acc should
#    land near the spliced seeds' ~0.665).
set -u
cd /workspace/senpai/target
PYI=/tmp/eval-serve-venv/bin/python
HERE=research/validity/quality_gates_ci
log(){ echo "[debias_rest $(date -u +%H:%M:%SZ)] $*"; }

log "splice seeds 3 4 5 ..."
$PYI "$HERE/debias_mmlu.py" --seeds 3 4 5 > "$HERE/_debias_345.out" 2>&1
log "splice 3 4 5 rc=$? (tail):"; tail -6 "$HERE/_debias_345.out"

log "fresh full-N seed 2 @ max_tokens=4096 ..."
$PYI research/validity/downstream_quality_eval/run_eval.py --task mmlu_pro --arm base_fullhead \
  --n 2000 --seed 12345 --sampling-seed 2 --temperature 1.0 --top-p 0.95 --top-k 64 \
  --min-tokens 8 --max-tokens 4096 --max-connections 32 --base-url http://127.0.0.1:8000/v1 \
  --out "$HERE/runs/mmlu_debias_n2000_s2.json" --log-dir "$HERE/logs/mmlu_debias_s2" \
  > "$HERE/_debias_s2fresh.out" 2>&1
log "seed2 fresh rc=$? (tail):"; tail -3 "$HERE/_debias_s2fresh.out"

echo "DEBIAS_REST_DONE $(date -u +%FT%TZ)" > "$HERE/_debias_rest.status"
log "ALL DONE -> runs/mmlu_debias_n2000_s{1,2,3,4,5}.json"
