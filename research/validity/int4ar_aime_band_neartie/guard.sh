#!/usr/bin/env bash
# Identity guard: prove logprobs-on greedy == official aime_eval.py greedy on the
# SAME fresh server (8 problems, full 12288 budget). If per-problem answers match,
# the logprobs capture does not move the argmax on this stack -> band numbers from
# the logprobs sessions are equivalent to the official harness.
set -uo pipefail
cd /workspace/senpai/target
PY=/tmp/vllm0220-srv/bin/python
DIR=research/validity/int4ar_aime_band_neartie

echo "[guard] official aime_eval.py (no logprobs) limit=8 ..."
"$PY" research/downstream_quality_aime/aime_eval.py --base-url http://127.0.0.1:8000 \
  --model gemma-4-e4b-it --years 2024 --k 1 --temperature 0 --max-tokens 12288 \
  --min-tokens 8 --no-thinking --client-concurrency 8 --seed 0 --limit 8 \
  --save-text --out "$DIR/_guard_official.json" || exit 11

echo "[guard] band_neartie session (logprobs top2) limit=8 ..."
"$PY" "$DIR/band_neartie.py" session --arm int4 --session-idx 99 \
  --base-url http://127.0.0.1:8000 --years 2024 --limit 8 --client-concurrency 8 \
  --max-tokens 12288 --out "$DIR/_guard_logprobs.json" || exit 12

echo "[guard] comparing per-problem answers ..."
"$PY" - "$DIR/_guard_official.json" "$DIR/_guard_logprobs.json" <<'PYEOF'
import json, sys
off = json.load(open(sys.argv[1]))
lp = json.load(open(sys.argv[2]))
off_ans = {r["id"]: r["maj_answer"] for r in off["per_problem"]}
lp_ans = {pid: rec["answer"] for pid, rec in lp["per_problem"].items()}
ids = sorted(set(off_ans) & set(lp_ans))
mism = [i for i in ids if off_ans[i] != lp_ans[i]]
print(f"[guard] n={len(ids)} matched; mismatches={len(mism)}")
for i in ids:
    flag = "OK" if off_ans[i] == lp_ans[i] else "MISMATCH"
    print(f"  {i}: official={off_ans[i]} logprobs={lp_ans[i]} {flag}")
print("[guard] IDENTITY_PASS" if not mism else "[guard] IDENTITY_FAIL")
sys.exit(0 if not mism else 13)
PYEOF
