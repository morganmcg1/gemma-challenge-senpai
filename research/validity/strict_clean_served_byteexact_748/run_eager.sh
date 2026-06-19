#!/usr/bin/env bash
# PR #748: decisive mechanism test -- replicate #743's enforce_eager condition but with the
# REAL online batched spec path. If eager BI=1 spec == eager BI=1 AR (128/128), the primary
# arm's divergence is the CUDA-graph capture (batched cheap path is FIXABLE). If still <128/128,
# the M=K batched-verify reduction itself diverges under BI=1 (offline prompt_logprobs proxy was
# unfaithful; dedicated kernel / K-sequential required).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/tmp/senpai-venvs/20f658587e8a6643/bin/python
run () { local bi=$1 spec=$2 tag=$3
  echo "[eager] ==== $tag (bi=$bi spec=$spec enforce_eager=1) $(date -u +%H:%M:%SZ) ===="
  "$PY" "$HERE/run_arm.py" --bi "$bi" --spec "$spec" --enforce-eager 1 \
    --n-prompts 128 --output-len 512 --port 8021 --tag "$tag" --startup-timeout 600 \
    > "$HERE/runs/${tag}.out" 2>&1
  echo "[eager] $tag exit=$? $(date -u +%H:%M:%SZ)"
}
run 1 0 bi1_spec0_eager
run 1 1 bi1_spec1_eager
echo "[eager] done $(date -u +%H:%M:%SZ)"
