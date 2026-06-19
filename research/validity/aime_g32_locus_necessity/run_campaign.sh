#!/bin/bash
# PR #713 g32-locus AIME campaign — sequential cells, idempotent per-item resume.
# Substrate: bf16 fakequant + --enforce-eager (CUDA graphs REJECTED: broke token-id
# parity vs enforce-eager AND only ~10% faster — bottleneck is bf16 bandwidth, not
# kernel launch). ~13.8 tok/s, ~4 min/item. Driver = system python3 (has wandb +
# harness); it spawns the venv python to serve. Each invocation is soft-capped at
# 78 min so it exits cleanly (partial, resumable) before the 90-min hard bound.
#
# Usage: run_campaign.sh <cellspec_file>
#   cellspec lines:  cell_name|g32_layers|decode|seed|extra_flags
#   ('' layers = N=0 all-g128 anchor; decode greedy|sampled)
set -u
cd /workspace/senpai/target/research/validity/aime_g32_locus_necessity
PYDRV=/usr/bin/python3
RES=results
N_TARGET=60
SOFT_CAP=78
MAX_ATTEMPTS=6
# Serve+drive concurrency. conc>1 is a FAITHFUL throughput multiplier on this bf16+BI=1
# substrate: conc8 greedy token ids are byte-identical to conc1 (sha256 parity gate,
# fqg32_paritychk8 vs fqg32_N0, all 4 incl the 6144-tok straggler — 2026-06-19). Pinned
# identically across every cell so paired/apples-to-apples comparisons are unaffected.
CONC="${CONC:-8}"
SPEC="${1:?need cellspec file}"

count() { [ -f "$1" ] && grep -c . "$1" 2>/dev/null || echo 0; }

echo "[campaign] START $(date -u +%FT%TZ) spec=$SPEC"
while IFS='|' read -r cell layers decode seed extra; do
  [ -z "${cell// }" ] && continue
  case "$cell" in \#*) continue;; esac
  jsonl="$RES/${cell}_aime.jsonl"
  for attempt in $(seq 1 $MAX_ATTEMPTS); do
    n=$(count "$jsonl")
    if [ "$n" -ge "$N_TARGET" ]; then
      echo "[campaign] $cell COMPLETE ($n/$N_TARGET) $(date -u +%H:%M:%S)"; break
    fi
    echo "[campaign] $cell attempt $attempt: $n/$N_TARGET done, launching $(date -u +%H:%M:%S)"
    $PYDRV eval_g32.py --cell-name "$cell" --g32-layers "$layers" \
        --decode "$decode" --seed "$seed" --evals aime \
        --soft-cap-min "$SOFT_CAP" --max-num-seqs "$CONC" \
        --wandb-group aime-g32-locus-necessity-fern \
        $extra >> "$RES/_drv_${cell}.log" 2>&1
    rc=$?
    echo "[campaign] $cell attempt $attempt exited rc=$rc, now $(count "$jsonl")/$N_TARGET $(date -u +%H:%M:%S)"
  done
  n=$(count "$jsonl")
  [ "$n" -ge "$N_TARGET" ] || echo "[campaign] WARN $cell stuck at $n/$N_TARGET after $MAX_ATTEMPTS attempts"
done < "$SPEC"
echo "[campaign] DONE $(date -u +%FT%TZ)"
