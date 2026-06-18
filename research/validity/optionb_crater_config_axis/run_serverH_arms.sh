#!/usr/bin/env bash
# PR #627 -- one-axis config sweep against the ALREADY-RUNNING vLLM 0.22.0 server
# on :8000 (serve config H: int4_g128_lmhead, max_model_len=6144, max_num_seqs=32,
# flashinfer_sampler=OFF -- the #615 healthy serve config). All arms GREEDY (temp=0).
# Only CLIENT decode params vary across arms (max_tokens, min_tokens, concurrency),
# so the serve config is held byte-fixed and the running server is reused for all.
#
# Arm spec: label:max_tokens:min_tokens:max_connections:limit   (limit 0 = full n=198)
set -u
cd /workspace/senpai/target
EVPY=/senpai-run/home/student-lawine/eval-client-venv/bin/python
RUN_EVAL=research/validity/downstream_quality_eval/run_eval.py
URL=http://127.0.0.1:8000/v1
OUT=research/validity/optionb_crater_config_axis/runs
LOGD=research/validity/optionb_crater_config_axis/logs
mkdir -p "$OUT" "$LOGD"
SUMMARY="$OUT/_serverH_summary.tsv"
[[ -f "$SUMMARY" ]] || echo -e "label\tmax_tokens\tmin_tokens\tconc\tn_scored\tacc\tlen_stop_rate\tctok_mean\tctok_p95\tstop_reasons\twall_s" > "$SUMMARY"

# Decisive arms front-loaded: healthy anchor (reproduce #615 0.4646), then conc=1.
ARMS=(
  "H_anchor615:4096:0:32:0"     # reproduce #615 greedy healthy anchor (min0,conc32)
  "H_conc1:4096:0:1:100"        # DECISIVE concurrency flip 32->1 (submission single-stream)
  "H_conc16:4096:0:16:0"        # kanna's conc=16
  "H_mt3072:3072:0:32:0"        # max_tokens axis: kanna's 3072 cap
  "H_mt2048:2048:0:32:0"        # max_tokens axis: 2048
  "H_min8:4096:8:32:0"          # min_tokens axis: 0->8 (#541 EOS-guard)
  "H_conc1_mt3072:3072:0:1:100" # kanna reconstruction: conc=1 AND her 3072 cap
)

for spec in "${ARMS[@]}"; do
  IFS=':' read -r lab MT MIN C LIM <<< "$spec"
  o="$OUT/gpqa_${lab}.json"; ld="$LOGD/gpqa_${lab}"
  limflag=""; [[ "$LIM" != "0" ]] && limflag="--limit $LIM"
  need=190; [[ "$LIM" != "0" ]] && need=$((LIM-5))
  if [[ -f "$o" ]] && /usr/bin/python3 -c "import json,sys;d=json.load(open('$o'));sys.exit(0 if d.get('n_scored',0)>=$need else 1)" 2>/dev/null; then
    echo "[arm:$lab] SKIP (complete)"; continue; fi
  echo "[arm:$lab] START MT=$MT MIN=$MIN conc=$C limit=${LIM} $(date -u +%FT%TZ)"; ts=$(date +%s)
  if "$EVPY" "$RUN_EVAL" --task gpqa_diamond --arm "$lab" $limflag \
      --seed 12345 --temperature 0.0 --top-p 1.0 --top-k 0 \
      --min-tokens "$MIN" --max-tokens "$MT" \
      --max-connections "$C" --base-url "$URL" --out "$o" --log-dir "$ld" \
      >"$LOGD/gpqa_${lab}.log" 2>&1; then
    wall=$(( $(date +%s)-ts ))
    /usr/bin/python3 - "$o" "$lab" "$MT" "$MIN" "$C" "$wall" "$SUMMARY" <<'PY'
import json,sys
o,lab,MT,MIN,C,wall,summ=sys.argv[1:8]
d=json.load(open(o))
acc=d.get("accuracy"); lsr=d.get("length_stop_rate")
row=[lab,MT,MIN,C,str(d.get("n_scored")),f"{acc:.4f}" if acc==acc else "nan",
     f"{lsr:.4f}" if lsr is not None else "nan",
     str(d.get("completion_tokens_mean")),str(d.get("completion_tokens_p95")),
     json.dumps(d.get("stop_reason_counts"),separators=(",",":")),wall]
open(summ,"a").write("\t".join(row)+"\n")
print(f"[arm:{lab}] DONE acc={acc:.4f} len_stop_rate={lsr:.4f} "
      f"ctok_mean={d.get('completion_tokens_mean'):.0f} stops={d.get('stop_reason_counts')} wall={wall}s")
PY
  else
    echo "[arm:$lab] FAIL wall=$(( $(date +%s)-ts ))s"; tail -8 "$LOGD/gpqa_${lab}.log"
  fi
done
echo "[serverH] ALL DONE $(date -u +%FT%TZ)"
