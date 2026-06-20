#!/usr/bin/env bash
# Core local validation for PR #777 (fp8 KV cache on bi0).
# Phases (one GPU, serial server boots):
#   1. greedy reference for int4_mtp_bi0_fp8kv (spec-off, M=1 AR int4 + fp8 KV)
#   2. validate_submission int4_mtp_bi0_fp8kv  -> greedy gate + PPL(128) + TPS + modalities
#   3. validate_submission int4_mtp_bi0_surgattn (control) -> PPL(128) + TPS (gate=NO_REFERENCE ok)
# All numbers are LOCAL A10G (exploratory); only the RELATIVE fp8kv-vs-bi0 delta transfers.
set -uo pipefail

ROOT=/workspace/senpai/target
VENV=/tmp/senpai-venvs/20f658587e8a6643/bin/python
OUT=$ROOT/research/validity/bi0_fp8kv
NP=32           # greedy/decode prompt count (PPL always runs full 128 GT records)
OL=512          # output_len (official protocol)
cd "$ROOT" || exit 1
mkdir -p "$OUT"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "[$(ts)] CORE VALIDATION START (np=$NP output_len=$OL)"

echo "[$(ts)] === PHASE 1: greedy reference (fp8kv e4m3, spec-off) ==="
"$VENV" -m scripts.local_validation.gen_greedy_reference \
  --mode served --submission submissions/int4_mtp_bi0_fp8kv --spec-off \
  --num-prompts "$NP" --output-len "$OL" --port 8001 \
  --server-python "$VENV" > "$OUT/phase1_reference.log" 2>&1
echo "[$(ts)] phase1 rc=$?"

echo "[$(ts)] === PHASE 2: validate fp8kv e4m3 (greedy gate + PPL128 + TPS + modalities) ==="
"$VENV" -m scripts.local_validation.validate_submission \
  --submission submissions/int4_mtp_bi0_fp8kv \
  --server-python "$VENV" \
  --num-prompts "$NP" --output-len "$OL" --port 8000 \
  --out-dir "$OUT/validate_fp8kv_e4m3" \
  --wandb-name "ubel/bi0-fp8kv-e4m3" --wandb-group "bi0-fp8-kv" \
  > "$OUT/phase2_validate_fp8kv.log" 2>&1
echo "[$(ts)] phase2 rc=$?"

echo "[$(ts)] === PHASE 3: validate bi0 surgattn baseline (control: PPL128 + TPS) ==="
"$VENV" -m scripts.local_validation.validate_submission \
  --submission submissions/int4_mtp_bi0_surgattn \
  --server-python "$VENV" \
  --num-prompts "$NP" --output-len "$OL" --port 8000 \
  --skip-modalities \
  --out-dir "$OUT/validate_bi0_control" \
  --wandb-name "ubel/bi0-baseline-control" --wandb-group "bi0-fp8-kv" \
  > "$OUT/phase3_validate_bi0.log" 2>&1
echo "[$(ts)] phase3 rc=$?"

echo "[$(ts)] CORE VALIDATION DONE"
echo "--- fp8kv evidence ---"; cat "$OUT/validate_fp8kv_e4m3/evidence.json" 2>/dev/null | "$VENV" -c "import json,sys; d=json.load(sys.stdin); print(json.dumps({k:d.get(k) for k in ['ppl','tps_single_stream_a10g','greedy_verdict','official_gate','completed','all_modalities_loaded']}, indent=2))" 2>/dev/null
echo "--- bi0 control evidence ---"; cat "$OUT/validate_bi0_control/evidence.json" 2>/dev/null | "$VENV" -c "import json,sys; d=json.load(sys.stdin); print(json.dumps({k:d.get(k) for k in ['ppl','tps_single_stream_a10g','greedy_verdict','completed']}, indent=2))" 2>/dev/null
