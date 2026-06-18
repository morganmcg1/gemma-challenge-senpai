#!/usr/bin/env bash
# PR #650 Arm A -- ONE AIME greedy budget point against the already-running server.
# Reuses research/downstream_quality_aime/aime_eval.py VERBATIM with the #638 panel's
# exact AIME invocation (years 2024,2025-I,2025-II; k=1; T=0; top_p=1; top_k=-1;
# min_tokens=8; no-thinking; seed 1234) EXCEPT:
#   * --max-tokens=$MT          (the swept output budget: 6144 / 8192 / 12288)
#   * --client-concurrency=16   (identical scores on the VLLM_BATCH_INVARIANT=1 stack;
#                                #638's bf16 AIME already ran conc16 -> 0.4667)
#   * --save-text               (diagnostics; the #650 completion_tokens capture rides
#                                along so ctok p50/p95 + threshold-fl are available)
# Greedy + batch-invariant => max-model-len is immaterial to the per-request output, so
# every leg is served at one fixed mml=16384 (validated by reproducing #638's 6144 0.350).
set -uo pipefail
cd /workspace/senpai/target
HERE=research/validity/int4ar_denom_harden
CLIENT=/tmp/eval-serve-venv/bin/python
TAG="${TAG:?set TAG=bf16|int4ar}"
MT="${MT:?set MT=6144|8192|12288}"
CONC="${CONC:-16}"
RES="$HERE/results"; mkdir -p "$RES"
OUT="$RES/${TAG}_aime_greedy_mt${MT}.json"
LOG="$HERE/_aime_${TAG}_mt${MT}.out"
echo "AIME-$TAG-mt$MT START $(date -u +%FT%TZ)"
$CLIENT research/downstream_quality_aime/aime_eval.py \
  --base-url http://127.0.0.1:8000 --model gemma-4-e4b-it \
  --years 2024,2025-I,2025-II --k 1 \
  --temperature 0.0 --top-p 1.0 --top-k -1 --max-tokens "$MT" --min-tokens 8 \
  --no-thinking --seed 1234 --save-text --client-concurrency "$CONC" \
  --label "${TAG}_aime_greedy_mt${MT}" --out "$OUT" \
  > "$LOG" 2>&1
rc=$?
echo "AIME-$TAG-mt$MT DONE rc=$rc $(date -u +%FT%TZ)"
tail -2 "$LOG"
exit $rc
