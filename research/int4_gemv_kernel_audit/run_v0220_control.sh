#!/usr/bin/env bash
# Deterministic-venv CONTROL for the int4 GEMV kernel audit (PR #675).
#
# The dev307 sweep showed that the *provably identical* Marlin kernel (atomicadd1,
# base2, base3 -- same gptq_marlin_gemm custom op as base1) diverges ~90% vs base1.
# That is dev307 autotune run-to-run non-determinism (#601), NOT a kernel-induced
# identity break -- so break_rate is uninterpretable as a #319 gate *in dev307*.
#
# This control re-runs the SHIPPED config (serve.py: NO VLLM_BATCH_INVARIANT) on the
# SHIP vLLM version 0.22.0 (venv 20f658587e8a6643), twice, fresh servers. If the two
# 0.22.0 captures are byte-identical (break_rate=0) then the SHIP environment IS
# run-to-run deterministic and #319 holds there -- proving the dev307 0.906 is a
# local-audit artifact, not a ship-gate risk. One server at a time on GPU 0.
set -u
SVENV=/tmp/senpai-venvs/20f658587e8a6643/bin/python   # vLLM 0.22.0 = SHIP version
ROOT=/workspace/senpai/target
ARMDIR="$ROOT/research/int4_gemv_kernel_audit/arms_v0220"
mkdir -p "$ARMDIR"
cd "$ROOT" || exit 2

run_arm () {
  local name="$1"; shift
  local out="$ARMDIR/$name"
  if [ -f "$out/arm_result.json" ]; then echo "[skip] $name (already done)"; return 0; fi
  echo "[v0220] $name START $(date -u +%FT%TZ)  extra=[$*]"
  "$SVENV" -m research.ar_identity_safe_tps.run_arm \
    --arm-name "$name" --out-dir "$out" --port 8000 "$@" \
    >>"$ARMDIR/$name.console.log" 2>&1
  echo "[v0220] $name DONE rc=$? $(date -u +%FT%TZ)"
}

echo "[v0220] BEGIN $(date -u +%FT%TZ)  vllm=$("$SVENV" -c 'import vllm;print(vllm.__version__)')"
run_arm v0220_a   # shipped serve config, no BI, fresh server #1
run_arm v0220_b   # shipped serve config, no BI, fresh server #2
echo "[v0220] PHASE COMPLETE $(date -u +%FT%TZ)"
