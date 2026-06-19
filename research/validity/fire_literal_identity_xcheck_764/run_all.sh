#!/usr/bin/env bash
# PR #764 (land): independent cross-validation of the fire's literal served-greedy identity.
# Runs 3 served arms SEQUENTIALLY (single A10G), each booting the fire submission's OWN serve.py,
# generating greedy completions over the 128-prompt set, tearing down. Resumable: skips an arm whose
# arm_summary.json already exists. FORCE=1 redoes all. Each arm is well under SENPAI_TIMEOUT_MINUTES.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=/workspace/senpai/target
PY=/tmp/senpai-venvs/20f658587e8a6643/bin/python
N=${N:-128}
OLEN=${OLEN:-512}
NREP=${NREP:-32}
PORT=${PORT:-8033}
ST=${ST:-900}
FORCE=${FORCE:-0}

cd "$ROOT"

run_arm () {
  local tag=$1 refmode=$2 n=$3
  local summ="$HERE/runs/$tag/arm_summary.json"
  if [[ "$FORCE" != "1" && -f "$summ" ]]; then
    echo "[run_all] SKIP $tag (exists)"; return 0
  fi
  echo "[run_all] ===== ARM $tag (reference_mode=$refmode n=$n) $(date -u +%H:%M:%SZ) ====="
  "$PY" "$HERE/run_xcheck.py" --tag "$tag" --reference-mode "$refmode" --n-prompts "$n" \
    --output-len "$OLEN" --port "$PORT" --startup-timeout "$ST" \
    > "$HERE/runs/${tag}.out" 2>&1
  local rc=$?
  echo "[run_all] ARM $tag exit=$rc $(date -u +%H:%M:%SZ)"
  return $rc
}

# 1) my independent spec-OFF M=1 AR reference (drafter OFF) -- the gate anchor for the identity
run_arm spec_off_ref   1 "$N"    || { echo "[run_all] spec_off_ref FAILED"; exit 1; }
# 2) the fire candidate: full BI=1, MTP drafter spec ON
run_arm spec_on        0 "$N"    || { echo "[run_all] spec_on FAILED"; exit 1; }
# 3) AR-vs-AR determinism control: a 2nd fresh M=1 AR run (shares the first NREP prompts with #1)
run_arm spec_off_repB  1 "$NREP" || { echo "[run_all] spec_off_repB FAILED"; exit 1; }

echo "[run_all] all arms done $(date -u +%H:%M:%SZ)"
echo "[run_all] analyzing..."
"$PY" "$HERE/analyze_xcheck.py"
