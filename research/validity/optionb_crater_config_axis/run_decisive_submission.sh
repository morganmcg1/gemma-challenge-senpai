#!/usr/bin/env bash
# PR #627 -- DECISIVE arm: GPQA-Diamond GREEDY at the EXACT submission serving
# config. Assumes serve_submission_config.sh has the submission's serve.py up on
# :8000 (MAX_MODEL_LEN=4096, MAX_NUM_BATCHED_TOKENS=512, default max_num_seqs).
# Single-stream (conc=1) -- the benchmark serves the submission single-stream.
# Identical client decode params to the H_conc1 arm so the ONLY delta vs the
# healthy 6144 single-stream arm is the submission serve bundle.
#
# Usage: run_decisive_submission.sh [limit]   (limit 0 = full n=198; default 100)
set -u
cd /workspace/senpai/target
LIM="${1:-100}"
EVPY=/senpai-run/home/student-lawine/eval-client-venv/bin/python
RUN_EVAL=research/validity/downstream_quality_eval/run_eval.py
URL=http://127.0.0.1:8000/v1
OUT=research/validity/optionb_crater_config_axis/runs
LOGD=research/validity/optionb_crater_config_axis/logs
SUMMARY="$OUT/_serverH_summary.tsv"
mkdir -p "$OUT" "$LOGD"

# Guard: confirm the running server is actually the 4096 submission config.
mml=$(curl -s --max-time 5 "$URL/models" | /usr/bin/python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0].get('max_model_len'))" 2>/dev/null)
if [[ "$mml" != "4096" ]]; then
  echo "[decisive] ABORT: server max_model_len=$mml (expected 4096 submission config). Re-serve first." ; exit 1
fi
echo "[decisive] server max_model_len=$mml OK (submission config)"

lab="sub_conc1"; o="$OUT/gpqa_${lab}.json"; ld="$LOGD/gpqa_${lab}"
limflag=""; [[ "$LIM" != "0" ]] && limflag="--limit $LIM"
echo "[decisive:$lab] START conc=1 mt=4096 limit=${LIM} $(date -u +%FT%TZ)"; ts=$(date +%s)
if "$EVPY" "$RUN_EVAL" --task gpqa_diamond --arm "$lab" $limflag \
    --seed 12345 --temperature 0.0 --top-p 1.0 --top-k 0 \
    --min-tokens 0 --max-tokens 4096 \
    --max-connections 1 --base-url "$URL" --out "$o" --log-dir "$ld" \
    >"$LOGD/gpqa_${lab}.log" 2>&1; then
  wall=$(( $(date +%s)-ts ))
  /usr/bin/python3 - "$o" "$lab" 4096 0 1 "$wall" "$SUMMARY" <<'PY'
import json,sys
o,lab,MT,MIN,C,wall,summ=sys.argv[1:8]
d=json.load(open(o))
acc=d.get("accuracy"); lsr=d.get("length_stop_rate")
row=[lab,MT,MIN,C,str(d.get("n_scored")),f"{acc:.4f}" if acc==acc else "nan",
     f"{lsr:.4f}" if lsr is not None else "nan",
     str(d.get("completion_tokens_mean")),str(d.get("completion_tokens_p95")),
     json.dumps(d.get("stop_reason_counts"),separators=(",",":")),wall]
open(summ,"a").write("\t".join(row)+"\n")
print(f"[decisive:{lab}] DONE acc={acc:.4f} len_stop_rate={lsr:.4f} "
      f"ctok_mean={d.get('completion_tokens_mean'):.0f} stops={d.get('stop_reason_counts')} wall={wall}s")
PY
else
  echo "[decisive:$lab] FAIL wall=$(( $(date +%s)-ts ))s"; tail -8 "$LOGD/gpqa_${lab}.log"
fi
