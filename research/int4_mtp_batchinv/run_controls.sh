#!/usr/bin/env bash
set -uo pipefail
cd /workspace/senpai/target/research/int4_mtp_batchinv
OUT=/workspace/senpai/target/research/int4_mtp_batchinv/arms
echo "##### DRIVER START $(date +%H:%M:%S) #####"
echo "--- ARM int4_off (INV=0 control, int4 target, 32p) ---"
ARM=int4_off INV=0 K=6 NPROMPTS=32 OUTDIR="$OUT" bash run_arm.sh > "$OUT/int4_off.run.log" 2>&1
echo "int4_off rc=$? $(date +%H:%M:%S)"
echo "--- ARM bf16_on (INV=1 positive control, bf16 target, 32p) ---"
ARM=bf16_on INV=1 K=6 NPROMPTS=32 TARGET_MODEL_ID=google/gemma-4-E4B-it OUTDIR="$OUT" bash run_arm.sh > "$OUT/bf16_on.run.log" 2>&1
echo "bf16_on rc=$? $(date +%H:%M:%S)"
echo "##### DRIVER DONE $(date +%H:%M:%S) #####"
echo
echo "########## VERDICTS ##########"
cat "$OUT/int4_off_fliprate.txt" 2>/dev/null || echo "(int4_off no fliprate)"
cat "$OUT/bf16_on_fliprate.txt" 2>/dev/null || echo "(bf16_on no fliprate)"
echo
echo "########## bf16_on: confirm NO Marlin (pure aten) + batch-invariant active ##########"
grep -iE "Marlin|UnquantizedLinear|LinearMethod|quantization=|batch_invariant.py:913|TRITON_ATTN|video items|Resolved architecture" "$OUT/bf16_on_ref_server.log" 2>/dev/null | head -15
echo "########## int4_off: confirm Marlin + invariance OFF ##########"
grep -iE "Marlin|batch_invariant.py:913|TRITON_ATTN" "$OUT/int4_off_ref_server.log" 2>/dev/null | head -6
