#!/usr/bin/env bash
# PR #627 -- DECISIVE arm (runnable cap). GPQA-Diamond GREEDY at the EXACT
# submission serving config (serve.py: MAX_MODEL_LEN=4096, MAX_NUM_BATCHED_TOKENS=512,
# default max_num_seqs), single-stream conc=1, n=198.
#
# Why mt=3072 and not 4096: the team's mandated GPQA cap is max_tokens>=4096, but
# the submission ships MAX_MODEL_LEN=4096, so prompt(~860 tok)+4096 > 4096 -> every
# request is HTTP-400 (recorded separately as the structural sub_conc1_mt4096 arm,
# 100% null). 3072 is the HIGHEST cap that fits mml=4096 and is NOT truncation-
# dominated (serverH H_mt3072 finish-length 12% vs H_mt2048 40%). So this is the
# best-available accuracy verdict the submission config can actually produce.
#
# Identical client decode params to the H_conc1_mt3072 healthy anchor (0.5000), so
# the ONLY delta vs that anchor is the submission serve bundle (mml 6144->4096 +
# batch512 chunked prefill). Clean one-bundle interpolation.
set -u
cd /workspace/senpai/target
EVPY=/senpai-run/home/student-lawine/eval-client-venv/bin/python
RUN_EVAL=research/validity/downstream_quality_eval/run_eval.py
URL=http://127.0.0.1:8000/v1
HERE=research/validity/optionb_crater_config_axis
OUT="$HERE/runs"; LOGD="$HERE/logs"; SUMMARY="$OUT/_serverH_summary.tsv"
mkdir -p "$OUT" "$LOGD"

# Guard: confirm the running server is the 4096 submission config.
mml=$(curl -s --max-time 5 "$URL/models" | /usr/bin/python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0].get('max_model_len'))" 2>/dev/null)
if [[ "$mml" != "4096" ]]; then
  echo "[decisive] ABORT: server max_model_len=$mml (expected 4096 submission config). Re-serve first." ; exit 1
fi
echo "[decisive] server max_model_len=$mml OK (submission config)"

lab="sub_conc1_mt3072"; o="$OUT/gpqa_${lab}.json"; ld="$LOGD/gpqa_${lab}"
echo "[decisive:$lab] START conc=1 mt=3072 n=198 $(date -u +%FT%TZ)"; ts=$(date +%s)
if "$EVPY" "$RUN_EVAL" --task gpqa_diamond --arm "$lab" \
    --seed 12345 --temperature 0.0 --top-p 1.0 --top-k 0 \
    --min-tokens 0 --max-tokens 3072 \
    --max-connections 1 --base-url "$URL" --out "$o" --log-dir "$ld" \
    >"$LOGD/gpqa_${lab}.log" 2>&1; then
  wall=$(( $(date +%s)-ts ))
  /usr/bin/python3 - "$o" "$lab" 3072 0 1 "$wall" "$SUMMARY" <<'PY'
import json,sys
o,lab,MT,MIN,C,wall,summ=sys.argv[1:8]
d=json.load(open(o))
acc=d.get("accuracy"); lsr=d.get("length_stop_rate")
ctm=d.get("completion_tokens_mean"); ctp=d.get("completion_tokens_p95")
def f(x,fmt="{:.4f}"):
    try: return fmt.format(x)
    except Exception: return "nan"
row=[lab,MT,MIN,C,str(d.get("n_scored")),f(acc),f(lsr),
     f(ctm,"{:.4f}") if ctm is not None else "None",
     str(ctp),json.dumps(d.get("stop_reason_counts"),separators=(",",":")),wall]
open(summ,"a").write("\t".join(row)+"\n")
print(f"[decisive:{lab}] DONE acc={f(acc)} len_stop_rate={f(lsr)} "
      f"ctok_mean={f(ctm,'{:.0f}') if ctm is not None else 'None'} "
      f"stops={d.get('stop_reason_counts')} wall={wall}s")
PY
else
  echo "[decisive:$lab] FAIL wall=$(( $(date +%s)-ts ))s"; tail -12 "$LOGD/gpqa_${lab}.log"
fi
