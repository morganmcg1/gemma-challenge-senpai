#!/usr/bin/env bash
# PR #748 (land): run the 4 served byte-exact transfer arms SEQUENTIALLY (single A10G).
# Each arm boots its own raw api_server, generates 128 greedy completions, tears down.
# Resumable: skips an arm whose arm_summary.json already exists. Pass FORCE=1 to redo all.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=/workspace/senpai/target
PY=/tmp/senpai-venvs/20f658587e8a6643/bin/python
N=${N:-128}
OLEN=${OLEN:-512}
PORT=${PORT:-8020}
ST=${ST:-600}
FORCE=${FORCE:-0}

cd "$ROOT"

run_arm () {
  local bi=$1 spec=$2 tag=$3
  local summ="$HERE/runs/$tag/arm_summary.json"
  if [[ "$FORCE" != "1" && -f "$summ" ]]; then
    echo "[run_all] SKIP $tag (exists)"; return 0
  fi
  echo "[run_all] ===== ARM $tag (bi=$bi spec=$spec) $(date -u +%H:%M:%SZ) ====="
  "$PY" "$HERE/run_arm.py" --bi "$bi" --spec "$spec" --n-prompts "$N" --output-len "$OLEN" \
    --port "$PORT" --tag "$tag" --startup-timeout "$ST" \
    > "$HERE/runs/${tag}.out" 2>&1
  local rc=$?
  echo "[run_all] ARM $tag exit=$rc $(date -u +%H:%M:%SZ)"
  return $rc
}

# order: AR references first (cheap, no spec), then spec arms.
run_arm 1 0 bi1_spec0   # BI=1 AR reference (PRIMARY ref)
run_arm 0 0 bi0_spec0   # BI=0 AR reference (control ref)
run_arm 1 1 bi1_spec1   # BI=1 batched-verify spec (PRIMARY)
run_arm 0 1 bi0_spec1   # BI=0 spec (deployed-order control + tax)

echo "[run_all] all arms done $(date -u +%H:%M:%SZ)"
