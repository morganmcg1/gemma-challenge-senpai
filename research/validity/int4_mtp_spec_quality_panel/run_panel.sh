#!/usr/bin/env bash
# PR #605 — 4-eval spec-config quality panel, run sequentially against the running
# int4_g128_lmhead+MTP-K7 spec server (http://127.0.0.1:8000). Sampling per
# generation_config.json (T=1.0 top_p=0.95 top_k=64) + min_tokens=8 on ALL four.
set -uo pipefail
cd /workspace/senpai/target
PY=/tmp/eval-serve-venv/bin/python
DIR=research/validity/int4_mtp_spec_quality_panel
RES=$DIR/results
mkdir -p "$RES"
BASE=http://127.0.0.1:8000
MODEL=gemma-4-e4b-it

echo "===== [1/4] GSM8K (sampled, n=500) $(date -u +%H:%M:%S) ====="
$PY research/downstream_quality_gsm8k/gsm8k_eval.py \
  --base-url "$BASE" --model "$MODEL" --label spec --regimes sampled \
  --n 500 --n-shot 8 --seed 1234 --top-p 0.95 --top-k 64 \
  --max-tokens 512 --min-tokens 8 --concurrency 16 --out-dir "$RES" \
  > "$DIR/_gsm8k.out" 2>&1
echo "  gsm8k rc=$? $(date -u +%H:%M:%S)"

echo "===== [2/4] MMLU-Pro (sampling, n=500) $(date -u +%H:%M:%S) ====="
$PY research/validity/downstream_quality_eval/run_eval.py \
  --task mmlu_pro --arm spec --out "$RES/spec_mmlu_pro.json" \
  --n 500 --seed 12345 --max-tokens 2048 --max-connections 16 \
  --temperature 1.0 --top-p 0.95 --top-k 64 --min-tokens 8 \
  --base-url "$BASE/v1" --model "$MODEL" \
  > "$DIR/_mmlu.out" 2>&1
echo "  mmlu rc=$? $(date -u +%H:%M:%S)"

echo "===== [3/4] GPQA-Diamond (sampling, n=198) $(date -u +%H:%M:%S) ====="
$PY research/validity/downstream_quality_eval/run_eval.py \
  --task gpqa_diamond --arm spec --out "$RES/spec_gpqa.json" \
  --seed 12345 --max-tokens 3072 --max-connections 16 \
  --temperature 1.0 --top-p 0.95 --top-k 64 --min-tokens 8 \
  --base-url "$BASE/v1" --model "$MODEL" \
  > "$DIR/_gpqa.out" 2>&1
echo "  gpqa rc=$? $(date -u +%H:%M:%S)"

echo "===== [4/4] AIME (maj@8, 2024+2025=60) $(date -u +%H:%M:%S) ====="
$PY research/downstream_quality_aime/aime_eval.py \
  --base-url "$BASE" --model "$MODEL" --years 2024,2025 --k 8 \
  --temperature 1.0 --top-p 0.95 --top-k 64 --max-tokens 3072 --min-tokens 8 \
  --no-thinking --seed 1234 --save-text \
  --label spec_aime --out "$RES/spec_aime.json" \
  > "$DIR/_aime.out" 2>&1
echo "  aime rc=$? $(date -u +%H:%M:%S)"

echo "===== AGGREGATE $(date -u +%H:%M:%S) ====="
$PY - <<'PYEOF'
import json
from pathlib import Path
R = Path("research/validity/int4_mtp_spec_quality_panel/results")
out = {}
# GSM8K
p = R / "spec_sampled.json"
if p.exists():
    d = json.loads(p.read_text())
    out["gsm8k"] = {"accuracy": d["accuracy"], "n": d.get("n_problems"),
                    "n_correct": d.get("n_correct"), "regime": "sampled"}
# MMLU-Pro
p = R / "spec_mmlu_pro.json"
if p.exists():
    d = json.loads(p.read_text())
    out["mmlu_pro"] = {"accuracy": d["accuracy"], "n": d.get("n_scored"),
                       "n_correct": d.get("n_correct"), "empty_rate": d.get("empty_rate"),
                       "decode": d.get("decode")}
# GPQA
p = R / "spec_gpqa.json"
if p.exists():
    d = json.loads(p.read_text())
    out["gpqa_diamond"] = {"accuracy": d["accuracy"], "n": d.get("n_scored"),
                           "n_correct": d.get("n_correct"), "empty_rate": d.get("empty_rate"),
                           "decode": d.get("decode")}
# AIME
p = R / "spec_aime.json"
if p.exists():
    d = json.loads(p.read_text())
    out["aime"] = {"accuracy": d["maj_k_accuracy"], "n": d.get("n_problems"),
                   "n_correct": d.get("n_correct_maj"), "k": d.get("maj_k")}
(Path("research/validity/int4_mtp_spec_quality_panel") / "panel_results.json").write_text(json.dumps(out, indent=2))
bars = {"mmlu_pro":0.605,"gpqa_diamond":0.471,"aime":0.090,"gsm8k":0.807}
print(json.dumps(out, indent=2))
print("\n=== VERDICT vs >=90% bars ===")
allp = True
for k in ["mmlu_pro","gpqa_diamond","aime","gsm8k"]:
    if k in out:
        a = out[k]["accuracy"]; b = bars[k]; ok = a >= b
        allp = allp and ok
        print(f"  {k:14s} acc={a:.4f} bar={b} -> {'PASS' if ok else 'FAIL'}")
    else:
        allp = False
        print(f"  {k:14s} MISSING")
print(f"\n  OVERALL: {'OPTION-B-FIREABLE' if allp else 'OPTION-B-DEAD'}")
PYEOF
echo "===== PANEL DONE $(date -u +%H:%M:%S) ====="
