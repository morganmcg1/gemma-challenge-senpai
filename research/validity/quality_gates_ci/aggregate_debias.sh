#!/usr/bin/env bash
# PR #590 -- aggregate the DE-BIASED MMLU-Pro CI (and the truncated-2048 baseline for
# comparison) once all 5 de-biased seeds are on disk. Does NOT log W&B (review first).
set -u
cd /workspace/senpai/target
HERE=research/validity/quality_gates_ci
OUT=$HERE/runs
PYI=/tmp/eval-serve-venv/bin/python
mkdir -p "$HERE/summaries"

# verify all 5 de-biased seeds present + complete
miss=0
for s in 1 2 3 4 5; do
  f="$OUT/mmlu_debias_n2000_s${s}.json"
  if [[ -f "$f" ]] && $PYI -c "import json,sys;d=json.load(open('$f'));sys.exit(0 if len(d.get('per_sample',[]))>=2000 else 1)" 2>/dev/null; then
    echo "debias seed $s OK acc=$($PYI -c "import json;print(round(json.load(open('$f'))['accuracy'],4))")"
  else echo "debias seed $s MISSING/INCOMPLETE"; miss=1; fi
done
[[ $miss -ne 0 ]] && { echo "FATAL: not all 5 de-biased seeds ready"; exit 1; }

echo "=== aggregate DE-BIASED MMLU-Pro (5 seeds, bar=0.605) ==="
$PYI "$HERE/aggregate_ci.py" --task mmlu_pro --label MMLU_Pro_debiased --bar 0.605 \
  --out "$HERE/summaries/mmlu_debias.json" \
  --inputs "$OUT"/mmlu_debias_n2000_s{1,2,3,4,5}.json

echo "=== aggregate TRUNCATED-2048 baseline (4 base seeds 1,3,4,5, bar=0.605) ==="
$PYI "$HERE/aggregate_ci.py" --task mmlu_pro --label MMLU_Pro_truncated2048 --bar 0.605 \
  --out "$HERE/summaries/mmlu_truncated.json" \
  --inputs "$OUT"/mmlu_base_fullhead_n2000_s{1,3,4,5}.json

echo "=== de-bias provenance (across spliced seeds 1,3,4,5) + extras json ==="
$PYI - <<'PY'
import json, glob
trunc_tot=rec_tot=still_tot=base_n=0
spliced=[]
for s in (1,3,4,5):
    d=json.load(open(f"research/validity/quality_gates_ci/runs/mmlu_debias_n2000_s{s}.json"))
    m=d.get("debias")
    if not m: continue
    spliced.append(s); trunc_tot+=m["n_trunc"]; rec_tot+=m["n_recovered"]; still_tot+=m["still_trunc_at_redo"]
    base_n+=len(d["per_sample"])
mt=json.load(open("research/validity/quality_gates_ci/summaries/mmlu_truncated.json"))
md=json.load(open("research/validity/quality_gates_ci/summaries/mmlu_debias.json"))
extra={
  "mmlu_truncated2048_mean": mt["mean_accuracy"],
  "mmlu_truncated2048_ci_lb": mt["ci_lb_95_2sided"],
  "mmlu_truncated2048_clears": mt["ci_lb_clears_bar"],
  "mmlu_truncated2048_seeds": mt["n_seeds_samples_per_q"],
  "mmlu_debias_redo_max_tokens": 4096,
  "mmlu_truncation_rate_2048": round(trunc_tot/base_n,4) if base_n else None,
  "mmlu_n_truncated_total_spliced": trunc_tot,
  "mmlu_n_recovered_total": rec_tot,
  "mmlu_n_still_trunc_at_4096": still_tot,
  "mmlu_recovered_frac_of_truncated": round(rec_tot/trunc_tot,4) if trunc_tot else None,
  "mmlu_debias_spliced_seeds": spliced,
  "mmlu_debias_note": "seed2 run fresh full-N at 4096 (no 2048 base; ENOSPC); seeds 1,3,4,5 spliced",
}
json.dump(extra, open("research/validity/quality_gates_ci/summaries/_debias_extra.json","w"), indent=2)
print(json.dumps(extra, indent=2))
print(f"\nMMLU truncated-2048: mean={mt['mean_accuracy']:.4f} CI-lb={mt['ci_lb_95_2sided']:.4f} clears={mt['ci_lb_clears_bar']}")
print(f"MMLU de-biased-4096: mean={md['mean_accuracy']:.4f} CI-lb={md['ci_lb_95_2sided']:.4f} clears={md['ci_lb_clears_bar']} "
      f"slack={md['slack_problems_at_ci_lb']:+.1f}/{md['n_questions']}")
PY
echo "DONE -> summaries/mmlu_debias.json, mmlu_truncated.json, _debias_extra.json"
