#!/usr/bin/env bash
# PR #703 -- aggregate the int4_g128_lmhead MMLU-Pro CI summaries once the
# run_mmlu_int4.sh sweep + debias has produced all 5 debiased seeds. Mirrors the
# #590 aggregate_debias.sh recipe (bar=0.605, same aggregate_ci.py) but writes into
# the int4 panel dir with the filename assemble_panel.py expects (mmlu_debiased.json).
# Also aggregates the raw truncated-2048 int4 arm for the debias-delta diagnostic.
set -u
cd /workspace/senpai/target
HERE=research/int4body_gate_panel
OUT=$HERE/runs
SUMM=$HERE/summaries
AGG=research/validity/quality_gates_ci/aggregate_ci.py
PYI=/tmp/eval-serve-venv/bin/python
N=2000
mkdir -p "$SUMM"

miss=0
for s in 1 2 3 4 5; do
  f="$OUT/mmlu_debias_n${N}_s${s}.json"
  if [[ -f "$f" ]] && $PYI -c "import json,sys;d=json.load(open('$f'));sys.exit(0 if len(d.get('per_sample',[]))>=$N else 1)" 2>/dev/null; then
    echo "debias seed $s OK acc=$($PYI -c "import json;print(round(json.load(open('$f'))['accuracy'],4))")"
  else echo "debias seed $s MISSING/INCOMPLETE"; miss=1; fi
done
[[ $miss -ne 0 ]] && { echo "FATAL: not all 5 int4 debiased seeds ready"; exit 1; }

echo "=== aggregate int4 DE-BIASED MMLU-Pro (5 seeds, bar=0.605) ==="
$PYI "$AGG" --task mmlu_pro --label MMLU_Pro_int4_debiased --bar 0.605 \
  --out "$SUMM/mmlu_debiased.json" \
  --inputs "$OUT"/mmlu_debias_n${N}_s{1,2,3,4,5}.json

echo "=== aggregate int4 RAW truncated-2048 MMLU-Pro (5 seeds, bar=0.605) ==="
$PYI "$AGG" --task mmlu_pro --label MMLU_Pro_int4_truncated2048 --bar 0.605 \
  --out "$SUMM/mmlu_truncated_int4.json" \
  --inputs "$OUT"/mmlu_base_fullhead_n${N}_s{1,2,3,4,5}.json

echo "=== int4 debias provenance ==="
$PYI - <<'PY'
import json
trunc=rec=still=basen=0; spliced=[]
for s in (1,2,3,4,5):
    d=json.load(open(f"research/int4body_gate_panel/runs/mmlu_debias_n2000_s{s}.json"))
    m=d.get("debias")
    if not m: continue
    spliced.append(s); trunc+=m["n_trunc"]; rec+=m["n_recovered"]; still+=m["still_trunc_at_redo"]; basen+=len(d["per_sample"])
mt=json.load(open("research/int4body_gate_panel/summaries/mmlu_truncated_int4.json"))
md=json.load(open("research/int4body_gate_panel/summaries/mmlu_debiased.json"))
prov={"trunc_total":trunc,"recovered_total":rec,"still_trunc_4096":still,
      "trunc_rate_2048":round(trunc/basen,4) if basen else None,
      "recovered_frac":round(rec/trunc,4) if trunc else None,"spliced_seeds":spliced,
      "raw2048_mean":mt["mean_accuracy"],"raw2048_ci_lb":mt["ci_lb_95_2sided"],
      "debiased_mean":md["mean_accuracy"],"debiased_ci_lb":md["ci_lb_95_2sided"],
      "debias_delta":round(md["mean_accuracy"]-mt["mean_accuracy"],4)}
json.dump(prov, open("research/int4body_gate_panel/summaries/mmlu_debias_provenance.json","w"), indent=2)
print(json.dumps(prov, indent=2))
PY
echo "DONE -> $SUMM/mmlu_debiased.json (+ mmlu_truncated_int4.json, mmlu_debias_provenance.json)"
