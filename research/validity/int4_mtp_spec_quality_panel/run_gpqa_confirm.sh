#!/usr/bin/env bash
# PR #605 — GPQA-Diamond is the verdict pivot (seed 12345 -> 0.4495 vs bar 0.471,
# only -0.6 sigma below at n=198). Two additional independent sampling seeds to
# pool (n~594 -> stderr ~0.020) so the FIREABLE/DEAD call is robust, not a coin flip.
# Same guards/sampling as the panel (T=1.0 top_p=0.95 top_k=64, min_tokens=8,
# max-model-len 6144 server). Report ALL seeds + pooled, no cherry-pick.
set -uo pipefail
cd /workspace/senpai/target
PY=/tmp/eval-serve-venv/bin/python
DIR=research/validity/int4_mtp_spec_quality_panel
RES=$DIR/results
BASE=http://127.0.0.1:8000
MODEL=gemma-4-e4b-it

for SEED in 23456 34567; do
  echo "===== GPQA-Diamond confirm seed=$SEED $(date -u +%H:%M:%S) ====="
  $PY research/validity/downstream_quality_eval/run_eval.py \
    --task gpqa_diamond --arm spec --out "$RES/spec_gpqa_s${SEED}.json" \
    --seed "$SEED" --max-tokens 3072 --max-connections 16 \
    --temperature 1.0 --top-p 0.95 --top-k 64 --min-tokens 8 \
    --base-url "$BASE/v1" --model "$MODEL" \
    > "$DIR/_gpqa_s${SEED}.out" 2>&1
  echo "  gpqa seed=$SEED rc=$? $(date -u +%H:%M:%S)"
done

echo "===== GPQA POOL $(date -u +%H:%M:%S) ====="
$PY - <<'PYEOF'
import json
from pathlib import Path
R = Path("research/validity/int4_mtp_spec_quality_panel/results")
files = [("12345", R/"spec_gpqa.json"),
         ("23456", R/"spec_gpqa_s23456.json"),
         ("34567", R/"spec_gpqa_s34567.json")]
tot_c = tot_n = 0
rows = []
for seed, p in files:
    if not p.exists():
        print(f"  seed={seed} MISSING {p}"); continue
    d = json.loads(p.read_text())
    c = d.get("n_correct"); n = d.get("n_scored"); a = d["accuracy"]
    tot_c += c; tot_n += n
    rows.append((seed, a, c, n))
    print(f"  seed={seed} acc={a:.4f} ({c}/{n}) err={d.get('n_error',0)} empty={d.get('empty_rate')}")
pooled = tot_c / tot_n if tot_n else 0.0
import math
se = math.sqrt(pooled*(1-pooled)/tot_n) if tot_n else 0.0
bar = 0.471
print(f"\n  POOLED gpqa acc={pooled:.4f} ({tot_c}/{tot_n})  stderr={se:.4f}")
print(f"  95% CI ~ [{pooled-1.96*se:.4f}, {pooled+1.96*se:.4f}]  bar={bar}")
print(f"  -> {'PASS' if pooled>=bar else 'FAIL'}  (margin {pooled-bar:+.4f} = {(pooled-bar)/se if se else 0:+.2f} sigma)")
Path("research/validity/int4_mtp_spec_quality_panel/gpqa_pooled.json").write_text(
    json.dumps({"pooled_accuracy": pooled, "n_correct": tot_c, "n_scored": tot_n,
                "stderr": se, "bar": bar, "pass": pooled>=bar,
                "seeds": [{"seed": s, "accuracy": a, "n_correct": c, "n_scored": n} for s,a,c,n in rows]}, indent=2))
PYEOF
echo "===== GPQA CONFIRM DONE $(date -u +%H:%M:%S) ====="
