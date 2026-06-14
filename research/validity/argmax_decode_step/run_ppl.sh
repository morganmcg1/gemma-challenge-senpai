#!/usr/bin/env bash
# PR #180 output-neutrality via PPL. Scores the fixed ppl_ground_truth_tokens.jsonl
# (128 records) on the PATCHED serve (ARGMAX_ONLY_DECODE=1 -> ppl_argmax, the
# load-bearing PR metric) and the CONTROL serve (parity proof Delta_ppl ~ 0).
# The M==8 gate is provably PPL-safe: min PPL prefill chunk is 11 tokens (no
# record's full length == 8 mod 512), so the skip never touches the prompt_logprobs
# path. This run confirms it empirically.
set -euo pipefail
cd "$(dirname "$0")/../../.."   # -> target/

PY=.venv/bin/python
OUT=research/validity/argmax_decode_step
SUB=submissions/fa2sw_argmax_decode

echo "=== PPL [patched] ARGMAX_ONLY_DECODE=1 ==="
ARGMAX_ONLY_DECODE=1 "$PY" -m scripts.local_validation.ppl_runner \
  --submission "$SUB" --out-dir "$OUT/ppl_patched"

echo ""
echo "=== PPL [control] (unpatched) ==="
"$PY" -m scripts.local_validation.ppl_runner \
  --submission "$SUB" --out-dir "$OUT/ppl_control"

echo ""
echo "=== PPL summaries ==="
for arm in patched control; do
  f="$OUT/ppl_${arm}/ppl_summary.json"
  [[ -f "$f" ]] && "$PY" -c "import json;d=json.load(open('$f'));print('$arm: ppl=%.5f num_tokens=%s model=%s'%(d['ppl'],d.get('num_tokens'),d.get('model')))"
done
