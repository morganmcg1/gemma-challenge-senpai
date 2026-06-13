#!/usr/bin/env bash
# Compare candidate decode_outputs.jsonl against an exact-greedy reference.
# Usage: greedy_check.sh <reference.jsonl> <candidate.jsonl>
# Exit 0 = GREEDY_IDENTICAL, 1 = DIVERGENT, 2 = INCOMPARABLE/error.
set -euo pipefail

REF="${1:?reference.jsonl}"
CAND="${2:?candidate.jsonl}"

ROOT="/workspace/senpai/target"
VDIR="$ROOT/official/main_bucket/shared_resources/gemma_greedy_identity_verifier_flowian-powers"
PY="$ROOT/.venvs/vllm022/bin/python"

cd "$VDIR"
"$PY" check_greedy_identity.py --reference "$REF" --candidate "$CAND"
