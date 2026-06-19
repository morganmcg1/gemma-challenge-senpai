#!/usr/bin/env python
"""PR #729 -- fp8/int8 KV-cache decode-lever FEASIBILITY VERDICT.

The PR asked: measure served TPS at output_len {512,2048,8192} with fp8 then int8
KV on the locked int4_g128_lmhead body; check self-consistency + PPL<=2.42; and
decide whether KV-dtype reduction is a self-consistent, PPL-safe long-output speed
lever, and at what output length it beats the fp16(=bf16/auto) KV baseline.

OUTCOME: on the locked submission stack (compressed-tensors int4 checkpoint +
A10G/Ampere sm_86 + vLLM 0.22.0 + forced TRITON_ATTN, mandated by the model's
heterogeneous head dims) there is NO runnable KV-cache-dtype reduction variant:

  * kv_cache_dtype="fp8" (e4m3): the KV write/quant kernel must store *fp8e4nv;
    Triton on sm_86 cannot emit it. Confirmed in BOTH compiled and enforce_eager
    modes (eager fails inside reshape_and_cache_kernel_flash), so it is a hardware
    dtype limit, not a torch.compile autotune artifact.
        ValueError("type fp8e4nv not supported in this architecture.
                    The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')")
  * kv_cache_dtype="fp8_e5m2" (=fp8e5, which sm_86 *does* support): vLLM 0.22.0
    HARD-BLOCKS it at layer init for quantized checkpoints --
    attention.py:_init_kv_cache_quant -> should_load_quant_weights(quant_method)
    is True for the compressed-tensors checkpoint ->
        ValueError("fp8_e5m2 kv-cache is not supported with fp8 checkpoints.")
  * int8 KV: not an accepted kv_cache_dtype value in vLLM 0.22.0 at all
    (only auto / fp8 / fp8_e5m2).

So only kv_cache_dtype="auto" (= bf16 KV) loads and runs. We still report a
fully-grounded verdict on three legs:
  1. FEASIBILITY census (the three failures above, with reproducible signatures).
  2. EVEN-IF-RUNNABLE roofline (roofline_int4head.json): at the only scored length
     (output_len=512) KV is 1.8% of step bytes -> +0.9% OPTIMISTIC pure-HBM upper
     bound; the KV-read fraction reaches 10% only at ~16384 tokens, far beyond the
     deployed max_model_len=4096. So even if it ran, fp8-KV is immaterial where the
     leaderboard scores.
  3. BASELINE validation: the auto(bf16-KV) arm reproduces the anchor (PPL 2.0189
     vs anchor 2.019; TPS 122.0@512 vs official 126.4) and the comparison harness
     self-test (aggregate.py auto-vs-auto) yields the trivial identity verdict
     (0 flips, GREEDY_IDENTICAL 32/32), proving the would-be fp8-vs-bf16 comparator
     is correctly wired -- it simply has no second runnable arm.

analysis_only=1, official_tps=0. No served-file change, no HF Job.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path("research/speed/fp8_kv_decode_speed")

# captured failure logs (advisor can grep these)
LOG_E4M3_COMPILED = HERE / "run_all.log"
LOG_E4M3_EAGER = HERE / "_probe_fp8_eager.log"
LOG_E5M2 = HERE / "arm_fp8_e5m2.log"

SIG_E4M3 = "type fp8e4nv not supported in this architecture"
SIG_E5M2 = "fp8_e5m2 kv-cache is not supported with fp8 checkpoints"


def log_has(path: Path, needle: str) -> bool:
    try:
        return needle in path.read_text(errors="ignore")
    except FileNotFoundError:
        return False


def first_match_line(path: Path, needle: str) -> str | None:
    try:
        for ln in path.read_text(errors="ignore").splitlines():
            if needle in ln:
                return ln.strip()
    except FileNotFoundError:
        return None
    return None


def main():
    auto = json.loads((HERE / "arm_auto.json").read_text())
    roof = json.loads((HERE / "roofline_int4head.json").read_text())
    selftest = json.loads((HERE / "_selftest_aggregate_auto_vs_auto.json").read_text())

    # --- leg 1: feasibility census ---
    e4m3_compiled = log_has(LOG_E4M3_COMPILED, SIG_E4M3)
    e4m3_eager = log_has(LOG_E4M3_EAGER, SIG_E4M3)
    e5m2_blocked = log_has(LOG_E5M2, SIG_E5M2)
    feasibility = {
        "fp8_e4m3": {
            "runnable": False,
            "reason": "Triton on sm_86 (A10G/Ampere) cannot emit fp8e4nv (e4m3); "
                      "the KV write/quant kernel store dtype is unsupported.",
            "layer": "hardware/triton_codegen",
            "compiled_fails": e4m3_compiled,
            "eager_fails": e4m3_eager,  # proves it is HW, not torch.compile
            "supported_fp8_dtypes_on_sm86": ["fp8e4b15", "fp8e5"],
            "error_signature": SIG_E4M3,
            "eager_error_line": first_match_line(LOG_E4M3_EAGER, SIG_E4M3),
        },
        "fp8_e5m2": {
            "runnable": False,
            "reason": "vLLM 0.22.0 hard-blocks fp8_e5m2 KV for quantized checkpoints "
                      "at attention.py:_init_kv_cache_quant (should_load_quant_weights "
                      "True for the compressed-tensors int4 checkpoint). Note: e5m2 "
                      "(=fp8e5) IS in sm_86's supported list -- this is a SOFTWARE "
                      "guard, not a hardware limit.",
            "layer": "vllm_init_guard",
            "init_fails": e5m2_blocked,
            "error_signature": SIG_E5M2,
            "error_line": first_match_line(LOG_E5M2, SIG_E5M2),
        },
        "int8": {
            "runnable": False,
            "reason": "int8 is not an accepted kv_cache_dtype value in vLLM 0.22.0 "
                      "(only auto / fp8 / fp8_e5m2).",
            "layer": "vllm_unsupported_enum",
        },
        "auto_bf16": {"runnable": True, "reason": "default bf16 KV; measured baseline."},
        "any_kv_dtype_reduction_runnable_on_stack": False,
    }

    # --- leg 2: even-if-runnable roofline (already corrected for int4 head) ---
    roofline_leg = {
        "basis": roof["basis"],
        "decode_weight_read_gib": roof["weight_read_gib"],
        "scored_output_len": roof["scored_output_len"],
        "deployed_max_model_len": roof["deployed_max_model_len"],
        "kv_read_frac_at_512": roof["kv_read_frac_at_512"],
        "fp8_uplift_pct_at_512_optimistic": roof["fp8_uplift_pct_at_512_optimistic"],
        "fp8_uplift_pct_at_2048_optimistic": roof["fp8_uplift_pct_at_2048_optimistic"],
        "fp8_uplift_pct_at_8192_optimistic": roof["fp8_uplift_pct_at_8192_optimistic"],
        "kv_material_crossover_10pct_position": roof["kv_material_crossover_10pct_position"],
        "crossover_beyond_deployed_cap": roof["crossover_beyond_deployed_cap"],
        "note": "OPTIMISTIC pure-HBM upper bound; realized lever would be smaller "
                "(fixed/compute overhead floor + fp8 dequant cost).",
    }
    material_even_optimistic_at_512 = roof["fp8_uplift_pct_at_512_optimistic"] > 2.0

    # --- leg 3: baseline validation + comparator self-test ---
    baseline_leg = {
        "auto_bf16_kv": {
            "tps": {L: auto["tps"][L]["output_tps"] for L in auto["tps"]},
            "ppl": auto["ppl"]["ppl"],
            "ppl_records": auto["ppl"]["num_records"],
            "ppl_safe": auto["ppl"]["ppl"] <= 2.42,
            "peak_gib": auto["peak_gib"],
            "load_s": auto["load_s"],
            "n_greedy_freerun": len(auto["greedy_freerun"]),
            "n_matched_state_records": len(auto["matched_state"]["per_record"]),
        },
        "anchor": {"submission": "int4_g128_lmhead", "official_tps": 126.378,
                   "ppl": 2.019, "wandb": "905tbujn", "scored_output_len": 512},
        "ppl_reproduces_anchor": abs(auto["ppl"]["ppl"] - 2.019) < 0.01,
        "tps_512_near_official": abs(auto["tps"]["512"]["output_tps"] - 126.378) < 12,
        "sliding_window_pin": (  # TPS roughly flat 512->2048 == sliding-window-bound
            auto["tps"]["2048"]["output_tps"] >= 0.95 * auto["tps"]["512"]["output_tps"]),
        "comparator_self_test_auto_vs_auto": {
            "greedy_gate": selftest["freerun_served_gate"]["verdict"],
            "flips": selftest["matched_state_self_consistency"]["flips"],
            "delta_pct_at_512": selftest["scored_point_512"]["delta_pct"],
            "self_test_passes": selftest["self_test"]["self_test_passes"],
            "note": "degenerate identity check -- proves byte_compare/matched_state "
                    "are correctly wired; NOT an fp8 result.",
        },
    }

    verdict_str = "INFEASIBLE_AND_IMMATERIAL"
    verdict = {
        "pr": 729, "lever": "fp8_int8_kv_cache_decode",
        "analysis_only": True, "official_tps": 0,
        "no_served_file_change": True, "no_hf_job": True,
        "stack": {"checkpoint": "int4_g128_lmhead (compressed-tensors w4a16)",
                  "gpu": "A10G (sm_86, Ampere)", "vllm": "0.22.0",
                  "attn_backend": "TRITON_ATTN (forced by heterogeneous head dims)"},
        "feasibility": feasibility,
        "roofline_even_if_runnable": roofline_leg,
        "baseline_validation": baseline_leg,
        "kv_lever_is_green_for_official": False,
        "verdict": verdict_str,
        "verdict_plain": (
            "fp8-KV and int8-KV are NOT a usable decode-speed lever for this "
            "submission: no KV-dtype reduction variant is runnable on the locked "
            "stack (fp8/e4m3 blocked by sm_86 Triton; fp8_e5m2 blocked by a vLLM "
            "0.22.0 guard for quantized checkpoints; int8 not supported). Even if "
            "one ran, at the only scored output length (512) KV is 1.8% of step "
            "bytes -> at most +0.9% on an optimistic HBM roofline; the KV-bound "
            "regime (>=10% of step bytes) starts near 16384 tokens, ~4x beyond the "
            "deployed max_model_len of 4096. The lever neither runs nor would pay "
            "where the leaderboard scores."),
        "crossover_length_material_fp8_win": (
            roof["kv_material_crossover_10pct_position"]),
        "crossover_reachable_in_deployed_cap": not roof["crossover_beyond_deployed_cap"],
        "material_even_optimistic_at_512": material_even_optimistic_at_512,
    }

    # self-tests on the verdict artifact itself
    st = {
        "no_kv_dtype_reduction_runnable": (
            feasibility["any_kv_dtype_reduction_runnable_on_stack"] is False),
        "e4m3_fails_compiled_and_eager": e4m3_compiled and e4m3_eager,
        "e5m2_init_blocked": e5m2_blocked,
        "auto_ppl_safe": baseline_leg["auto_bf16_kv"]["ppl_safe"],
        "auto_ppl_reproduces_anchor": baseline_leg["ppl_reproduces_anchor"],
        "comparator_self_test_passes": (
            baseline_leg["comparator_self_test_auto_vs_auto"]["self_test_passes"]),
        "comparator_identity_zero_flips": (
            baseline_leg["comparator_self_test_auto_vs_auto"]["flips"] == 0),
        "scored_point_immaterial_even_optimistic": not material_even_optimistic_at_512,
        "crossover_beyond_deployed_cap": roof["crossover_beyond_deployed_cap"],
    }
    st["self_test_passes"] = all(st.values())
    verdict["self_test"] = st

    (HERE / "feasibility_verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps({
        "verdict": verdict_str,
        "any_kv_dtype_runnable": feasibility["any_kv_dtype_reduction_runnable_on_stack"],
        "e4m3": "HW sm_86 (compiled+eager)", "e5m2": "vLLM guard", "int8": "unsupported enum",
        "fp8_uplift_pct_at_512_optimistic": roof["fp8_uplift_pct_at_512_optimistic"],
        "crossover_10pct": roof["kv_material_crossover_10pct_position"],
        "auto_ppl": auto["ppl"]["ppl"], "auto_tps_512": auto["tps"]["512"]["output_tps"],
        "self_test_passes": st["self_test_passes"],
    }, indent=2))


if __name__ == "__main__":
    main()
