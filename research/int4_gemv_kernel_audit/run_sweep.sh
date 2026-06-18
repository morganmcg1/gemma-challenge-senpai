#!/usr/bin/env bash
# Sequential int4 GEMV kernel-selection sweep over the SHIPPED int4_g128_lmhead body.
# Every arm goes through the canonical denken #674 AR M=1 harness (run_arm.py): same
# body, same official 128x512 greedy decode, same wall_tps protocol; arms differ ONLY
# by the per-arm --extra-env (the kernel-selection knob). One server at a time on GPU 0.
#
# ANALYSIS-ONLY: no HF job, no submission, no served-file change.
#   base*      : default auto-selection  -> MarlinLinearKernel (ship config, no BI)
#   humming*   : force HummingLinearKernel  (disable Marlin)
#   triton*    : force TritonW4A16LinearKernel (disable Marlin+Humming)
#   atomicadd1 : VLLM_MARLIN_USE_ATOMIC_ADD=1 on Marlin (knob; inert on sm8x+bf16)
#   baseBI1    : VLLM_BATCH_INVARIANT=1 on Marlin (PR-stated-config control)
set -u
SVENV=/tmp/senpai-venvs/5f4c623f772358a2/bin/python
ROOT=/workspace/senpai/target
ARMDIR="$ROOT/research/int4_gemv_kernel_audit/arms"
mkdir -p "$ARMDIR"
cd "$ROOT" || exit 2

run_arm () {
  local name="$1"; shift
  local out="$ARMDIR/$name"
  if [ -f "$out/arm_result.json" ]; then echo "[skip] $name (already done)"; return 0; fi
  echo "[arm] $name START $(date -u +%FT%TZ)  extra=[$*]"
  "$SVENV" -m research.ar_identity_safe_tps.run_arm \
    --arm-name "$name" --out-dir "$out" --port 8000 "$@" \
    >>"$ARMDIR/$name.console.log" 2>&1
  echo "[arm] $name DONE rc=$? $(date -u +%FT%TZ)"
}

echo "[sweep] BEGIN $(date -u +%FT%TZ)"
# --- Deliverable 1 confirm + Deliverable 4 anchor (rep1 = byte-identity reference) ---
run_arm base1
# --- Deliverable 2: byte-identical kernel sweep (one fresh server each) ---
run_arm humming1 --extra-env VLLM_DISABLED_KERNELS=MarlinLinearKernel
run_arm triton1  --extra-env "VLLM_DISABLED_KERNELS=MarlinLinearKernel,HummingLinearKernel"
# --- Deliverable 3: kernel-config knob on the active (Marlin) kernel ---
run_arm atomicadd1 --extra-env VLLM_MARLIN_USE_ATOMIC_ADD=1
# --- PR-stated-config control (BI=1 is TPS/identity-neutral at fixed M=1) ---
run_arm baseBI1 --extra-env VLLM_BATCH_INVARIANT=1
# --- Deliverable 4: finish the median-of-3 anchor ---
run_arm base2
run_arm base3
echo "[sweep] PHASE1 COMPLETE $(date -u +%FT%TZ)"
