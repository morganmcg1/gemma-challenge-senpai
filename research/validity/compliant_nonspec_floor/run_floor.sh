#!/usr/bin/env bash
# lawine #196 compliant-nonspec-floor driver: wall_tps N=3 -> PPL -> consolidate.
# LOCAL single-A10G serve profiling only. No HF Job / submission. Reuses the
# verified spec baseline (PR #82 self-null, restart-invariant per #72) so only
# the non-spec candidate arm re-serves. Run under repo .venv (has wandb).
set -uo pipefail
cd /workspace/senpai/target
D=research/validity/compliant_nonspec_floor
PY=.venv/bin/python

echo "[driver] $(date -u +%FT%TZ) START"

# Clean stale wall_tps artifacts from the interrupted run (run02 was 75/128).
rm -rf "$D/walltps/nonspec" "$D/walltps/paired_ab.json" "$D/walltps/records.jsonl"

echo "[driver] $(date -u +%FT%TZ) === STEP A: wall_tps N=3 nonspec (reuse spec baseline) ==="
$PY scripts/profiler/paired_tps_ab.py \
  --baseline fa2sw_precache_kenyan --candidate fa2sw_nonspec_int4 \
  --candidate-label nonspec --n 3 \
  --reuse-baseline-from research/walltps_ab/selfnull/paired_ab.json \
  --out-dir "$D/walltps" \
  --wandb-name lawine/nonspec-floor-walltps --wandb-group compliant-nonspec-floor
echo "[driver] $(date -u +%FT%TZ) wall_tps exit=$?"

echo "[driver] $(date -u +%FT%TZ) === STEP B: PPL validity pass (nonspec serve) ==="
$PY "$D/ppl_check.py"
echo "[driver] $(date -u +%FT%TZ) ppl exit=$?"

echo "[driver] $(date -u +%FT%TZ) === STEP C: consolidate floor_report + wandb ==="
$PY "$D/analyze_floor.py"
echo "[driver] $(date -u +%FT%TZ) analyze exit=$?"

echo "[driver] $(date -u +%FT%TZ) DONE"
