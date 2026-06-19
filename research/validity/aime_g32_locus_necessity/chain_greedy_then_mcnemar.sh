#!/bin/bash
# Bounded local chain: wait for the in-flight greedy campaign (cells_greedy.txt) to
# exit, then launch the 5-seed paired McNemar campaign (cells_mcnemar.txt) so the
# single A10G never sits idle overnight. Pair-interleaved cells yield COMPLETE
# seed-pairs incrementally, so analysis at 3 seeds is valid if 5 is overkill.
set -u
cd /workspace/senpai/target/research/validity/aime_g32_locus_necessity
GREEDY_PID="${1:?need greedy campaign pid}"

echo "[chain] $(date -u +%FT%TZ) waiting for greedy campaign pid=$GREEDY_PID"
until ! kill -0 "$GREEDY_PID" 2>/dev/null; do sleep 60; done
echo "[chain] $(date -u +%FT%TZ) greedy campaign exited; launching McNemar"

# Guard: only proceed if greedy L14-27 actually completed 60/60 (don't burn GPU on a
# stuck greedy). If short, leave a note and stop — a human/wakeup will inspect.
n=$(grep -c . results/fqg32_L14-27_aime.jsonl 2>/dev/null || echo 0)
if [ "$n" -lt 60 ]; then
  echo "[chain] ABORT: greedy L14-27 only $n/60 — not chaining McNemar; inspect."
  exit 2
fi
echo "[chain] greedy L14-27 complete ($n/60); starting McNemar campaign $(date -u +%FT%TZ)"
bash run_campaign.sh cells_mcnemar.txt >> results/_campaign_mcnemar.log 2>&1
echo "[chain] $(date -u +%FT%TZ) McNemar campaign returned rc=$?"
