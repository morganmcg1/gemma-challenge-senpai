#!/usr/bin/env bash
# PR #653 (lawine) -- AIME on g32: does reverting group size recover the binding
# Reading-A fail? ONE harness, three arms, self-consistent.
#
# Reuses the ubel #638 AIME instrument (research/downstream_quality_aime/aime_eval.py)
# against whichever int4 checkpoint is currently served on :8000 by one of:
#   shipped_g128 : optionb_denom_0p22_gb6144/serve_int4ar_0p22.sh   (int4_g128_lmhead, minmax)
#   ours_g32     : int4_body_quality_upside_639/serve_ours_g32.sh    (int4_g32_lmhead, untied int4 head)
#   official_g32 : int4_body_quality_upside_639/serve_official_g32.sh (google w4a16-ct, tied bf16 head)
# ALL three serve through submissions/bf16_base_aime/serve.py (plain vLLM, NO drafter
# = clean M=1 AR) at MAX_NUM_SEQS=1, BI=1, gb6144 -- byte-for-byte the ubel #628/#638
# denominator config, so only the served checkpoint changes across arms.
#
# Protocol (matches PR #653 + ubel #638 AIME leg byte-for-byte):
#   years 2024,2025-I,2025-II (n=60), k=1, greedy (temp 0, top_p 1.0, top_k -1),
#   max_tokens 6144, min_tokens 8, --no-thinking, seed 1234, client_concurrency 1
#   (serial -> batch-1 decode, deterministic greedy, identical to ubel).
set -euo pipefail
ARM="${1:?usage: run_arm.sh <arm-label>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
# aime_eval.py is referenced repo-root-relative below; the orchestrator invokes this
# script with CWD=the 653 dir, so cd to repo root (exactly how arm1 was launched).
ROOT="$(cd "$HERE/../../.." && pwd)"
CLIENT=/tmp/eval-serve-venv/bin/python
BASE=http://127.0.0.1:8000
MODEL=gemma-4-e4b-it
MT=6144
OUT="$HERE/results/${ARM}_aime_gb6144.json"

# Guard: server must be up and serving the expected model before we spend the slot.
if ! curl -s --max-time 5 "$BASE/v1/models" 2>/dev/null | grep -q "$MODEL"; then
  echo "[run_arm] FATAL: no server answering $MODEL at $BASE"; exit 2
fi

echo "[run_arm] arm=$ARM start=$(date -u +%FT%TZ) out=$OUT cwd=$ROOT"
cd "$ROOT"
"$CLIENT" research/downstream_quality_aime/aime_eval.py \
  --base-url "$BASE" --model "$MODEL" \
  --years 2024,2025-I,2025-II --k 1 \
  --temperature 0.0 --top-p 1.0 --top-k -1 --max-tokens "$MT" --min-tokens 8 \
  --no-thinking --seed 1234 --save-text --client-concurrency 1 \
  --label "$ARM" --out "$OUT" \
  > "$HERE/_${ARM}_aime.out" 2>&1
echo "[run_arm] arm=$ARM done=$(date -u +%FT%TZ) rc=$?"
tail -3 "$HERE/_${ARM}_aime.out"
