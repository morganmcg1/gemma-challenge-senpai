#!/usr/bin/env bash
# PR #590 -- master driver: run all three multi-seed quality-gate sweeps against the
# live base_fullhead server (must already be up on :8000). Order = AIME (watch item,
# ~31min concurrent) -> GSM8K (~20min) -> MMLU-Pro N=2000 (~3h, the swing gate, last).
# Dataset-level resumable: skip a dataset whose 5 per-seed outputs all exist.
# Each inner python eval process stays < 90 min; this bash orchestrator is not a run.
set -u
cd /workspace/senpai/target
HERE=research/validity/quality_gates_ci
OUT=$HERE/runs
STATUS=$HERE/_all_status.txt
export N=2000            # MMLU-Pro subset size (sized so finite-sample CI-lb clears the thin 0.026 margin)

mkdir -p "$OUT"
echo "MASTER START $(date -u +%FT%TZ)" > "$STATUS"

have_all() { for f in "$@"; do [[ -f "$f" ]] || return 1; done; return 0; }

# ---- AIME (5 seeds x k=5, concurrent) ----
AIME_OUT=("$OUT"/aime_base_fullhead_s{1234,2345,3456,4567,5678}.json)
if have_all "${AIME_OUT[@]}"; then
  echo "AIME SKIP (5 outputs present) $(date -u +%H:%M:%SZ)" | tee -a "$STATUS"
else
  echo "AIME RUN $(date -u +%H:%M:%SZ)" | tee -a "$STATUS"
  bash "$HERE/run_aime_multiseed.sh" >"$HERE/_aime_master.out" 2>&1
  echo "AIME rc=$? $(date -u +%H:%M:%SZ)" | tee -a "$STATUS"
fi

# ---- GSM8K (5 sampling-seeds, sequential, short) ----
GSM_OUT=("$OUT"/base_fullhead_sampled_s{1,2,3,4,5}.json)
if have_all "${GSM_OUT[@]}"; then
  echo "GSM8K SKIP (5 outputs present) $(date -u +%H:%M:%SZ)" | tee -a "$STATUS"
else
  echo "GSM8K RUN $(date -u +%H:%M:%SZ)" | tee -a "$STATUS"
  bash "$HERE/run_gsm8k_multiseed.sh" >"$HERE/_gsm8k_master.out" 2>&1
  echo "GSM8K rc=$? $(date -u +%H:%M:%SZ)" | tee -a "$STATUS"
fi

# ---- MMLU-Pro (5 sampling-seeds, sequential, N=2000; per-seed resumable inside) ----
MMLU_OUT=("$OUT"/mmlu_base_fullhead_n${N}_s{1,2,3,4,5}.json)
if have_all "${MMLU_OUT[@]}"; then
  echo "MMLU SKIP (5 outputs present) $(date -u +%H:%M:%SZ)" | tee -a "$STATUS"
else
  echo "MMLU RUN N=$N $(date -u +%H:%M:%SZ)" | tee -a "$STATUS"
  bash "$HERE/run_mmlu_multiseed.sh" >"$HERE/_mmlu_master.out" 2>&1
  echo "MMLU rc=$? $(date -u +%H:%M:%SZ)" | tee -a "$STATUS"
fi

echo "MASTER DONE $(date -u +%FT%TZ)" | tee -a "$STATUS"
