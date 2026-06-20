#!/usr/bin/env bash
# PR #814 Step-3 FOCUSED g128 quality panel (per human #784: cap analysis per
# lane, measure enough to decide, then move on). GPQA-Diamond is already done
# (5 seeds, pooled 0.4808 >= floor 0.4712). This completes the gate with ONE
# clean config per remaining axis, using the IDENTICAL #795 protocol/flags:
#   * MMLU-Pro n=250 @ max_tokens=4096 (clean-gate config; #795 int4head 0.6920)
#   * GSM8K n=300 sampled (PRIMARY; #795/#788 int4head 0.915)
#   * AIME greedy maj@1 n=30 (the gate; floor 0.090, #795 int4head 0.300)
# Skips the redundant like-for-like configs (MMLU@2048, GSM8K greedy diag, AIME
# sampled maj@8) to keep the lane fast. Re-add them only if a floor is marginal.
# LOCAL ONLY -- launches a local vLLM server on :8021, no HF Job.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
EVAL_PY=/tmp/eval-serve-venv/bin/python
RUN_EVAL="$HERE/../../validity/downstream_quality_eval/run_eval.py"
GSM8K_EVAL="$HERE/../../downstream_quality_gsm8k/gsm8k_eval.py"
AIME_EVAL="$HERE/../../downstream_quality_aime/aime_eval.py"
RES="$HERE/results"
STATUS="$HERE/_focused.status"
BASE_URL="http://127.0.0.1:8021/v1"
GSM_BASE_URL="http://127.0.0.1:8021"
ARM=g128_body
mkdir -p "$RES"

echo "FOCUSED START $(date -u +%FT%TZ)" | tee "$STATUS"

# ---- launch g128 server on :8021 ----
bash "$HERE/serve_g128.sh" >>"$HERE/_focused.out" 2>&1
SRV_PID="$(cat "$HERE/_server.pid" 2>/dev/null || echo '')"
echo "server pid=$SRV_PID" | tee -a "$STATUS"

ready=0
for _ in $(seq 1 90); do
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:8021/v1/models" 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then ready=1; break; fi
  # bail early if the server process died
  if [ -n "$SRV_PID" ] && ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "FOCUSED ABORT: server pid $SRV_PID died (see server_g128.log)" | tee -a "$STATUS"; exit 1
  fi
  sleep 5
done
if [ "$ready" -ne 1 ]; then
  echo "FOCUSED ABORT: server not ready (last code=$code) $(date -u +%FT%TZ)" | tee -a "$STATUS"
  [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null
  exit 1
fi
echo "FOCUSED: server ready $(date -u +%FT%TZ)" | tee -a "$STATUS"

# ---- MMLU-Pro n=250 @4096 (clean gate) ----
mout="$RES/${ARM}_mmlu_n250_mt4096_s12345.json"
echo "===== mmlu_pro n=250 max_tokens=4096 seed=12345 $(date -u +%H:%M:%SZ) =====" | tee -a "$STATUS"
"$EVAL_PY" "$RUN_EVAL" \
  --task mmlu_pro --arm "$ARM" \
  --out "$mout" \
  --base-url "$BASE_URL" --model gemma-4-e4b-it \
  --n 250 --temperature 1.0 --top-p 0.95 --top-k 64 \
  --max-tokens 4096 --min-tokens 8 --sampling-seed 0 --seed 12345 \
  --max-connections 16 \
  >>"$HERE/_mmlu_mt4096.out" 2>&1
rc=$?
if [ -f "$mout" ]; then
  acc=$("$EVAL_PY" -c "import json;d=json.load(open('$mout'));print(f\"acc={d['accuracy']:.4f} ({d['n_correct']}/{d['n_scored']}) err={d['n_error']} trunc={d['n_length_truncated']} ctok_mean={d['completion_tokens_mean']:.0f}\")" 2>/dev/null)
  echo "  mmlu n=250 mt=4096 rc=$rc $acc" | tee -a "$STATUS"
else
  echo "  mmlu n=250 mt=4096 rc=$rc NO_OUTPUT (see _mmlu_mt4096.out)" | tee -a "$STATUS"
fi

# ---- GSM8K n=300 sampled (PRIMARY) ----
echo "===== gsm8k n=300 sampled $(date -u +%H:%M:%SZ) =====" | tee -a "$STATUS"
"$EVAL_PY" "$GSM8K_EVAL" \
  --base-url "$GSM_BASE_URL" --model gemma-4-e4b-it \
  --label "$ARM" --regimes sampled --n 300 \
  --top-p 0.95 --top-k 64 \
  --max-tokens 512 --min-tokens 8 --sampling-seed 0 --seed 1234 \
  --concurrency 16 --max-num-seqs 16 \
  --out-dir "$RES" \
  >>"$HERE/_gsm8k.out" 2>&1
rc=$?
gout="$RES/${ARM}_sampled_s0.json"
if [ -f "$gout" ]; then
  v=$("$EVAL_PY" -c "import json;d=json.load(open('$gout'));print(f\"acc={d['accuracy']:.4f} ({d['n_correct']}/{d['n_scored']}) trunc={d.get('n_length_truncated','?')}\")" 2>/dev/null)
  echo "  gsm8k sampled rc=$rc $v" | tee -a "$STATUS"
else
  echo "  gsm8k sampled rc=$rc NO_OUTPUT (see _gsm8k.out)" | tee -a "$STATUS"
fi

# ---- AIME greedy maj@1 n=30 (the gate) ----
echo "===== aime greedy maj@1 n=30 $(date -u +%H:%M:%SZ) =====" | tee -a "$STATUS"
"$EVAL_PY" "$AIME_EVAL" --base-url "$GSM_BASE_URL" --model gemma-4-e4b-it \
  --label ${ARM}_aime_greedy --years 2024 --k 1 \
  --temperature 0.0 --top-p 1.0 --top-k -1 \
  --max-tokens 3072 --min-tokens 8 --seed 1234 --no-thinking \
  --client-concurrency 1 --save-text \
  --out "$RES/${ARM}_aime_greedy_n30.json" \
  >>"$HERE/_aime_greedy.out" 2>&1
rc=$?
g="$RES/${ARM}_aime_greedy_n30.json"
if [ -f "$g" ]; then
  v=$("$EVAL_PY" -c "import json;d=json.load(open('$g'));tr=sum(1 for p in d['per_problem'] for f in p['finish_reasons'] if f=='length');print(f\"maj@1={d['maj_k_accuracy']:.4f} ({d['n_correct_maj']}/{d['n_problems']}) extract_fail={d['extract_fail_rate']:.3f} length_trunc={tr}/{d['total_samples']}\")" 2>/dev/null)
  echo "  aime greedy rc=$rc $v" | tee -a "$STATUS"
else
  echo "  aime greedy rc=$rc NO_OUTPUT (see _aime_greedy.out)" | tee -a "$STATUS"
fi

echo "FOCUSED PANEL DONE $(date -u +%FT%TZ)" | tee -a "$STATUS"

# ---- tear down server ----
if [ -n "$SRV_PID" ]; then
  kill "$SRV_PID" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
fi
echo "FOCUSED ALL DONE $(date -u +%FT%TZ)" | tee -a "$STATUS"
