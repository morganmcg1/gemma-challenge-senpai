"""PR #544 -- base_fullhead TPS-ceiling: gap decomposition + optimized-ceiling model.

LOCAL analysis-only. Consumes the two MEASURED artifacts produced on this pod:
  run/serve_results.json   in-serve A/B (base_fullhead 252.31 vs osoi5_ship 350.76),
                           robust wall_tps + STEPTIME per-step split + E[T] + PPL.
  head_results.json        microbench: real bf16 262k head matmul+argmax vs 12k/16k,
                           eff HBM back-solved, int4/fp8 head projections.

Decomposes the single-stream step-time gap into the four PR-named components and
models the realistic optimized ceiling, holding the +5-layer depth cost irreducible
(kanna #539: layer-drop is the body-damage cause -> base_fullhead keeps the layers).

  base_fullhead = stock int4 google/gemma-4-E4B-it-qat-w4a16-ct (42L, full 262k tied
                  bf16 lm_head) + fast stack (fern #535 whh42dgd recipe).
  osoi5_ship    = 37L baked body + 12k int4 pruned head (unsafe ship).

NO HF job, NO submission, NO served-file change. analysis_only=true, official_tps=0.

    .venv/bin/python -m research.validity.base_fullhead_tps_ceiling.decompose
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
SERVE = HERE / "run" / "serve_results.json"
HEAD = HERE / "head_results.json"

# PR anchors (LOCAL scale; never mix with official).
BFH_TPS_ANCHOR = 253.78        # fern #535 whh42dgd base_fullhead local wall_tps
OSOI5_TPS_ANCHOR = 353.73      # unsafe osoi5 ship local wall_tps
UNSAFE_FRONTIER_LOCAL = 442.0  # byte-equivalence-legal unsafe-class frontier (#523 lineage)
TAU_LO = 1.03524               # local->official transfer (PR #267); official ~= tau_lo*local


def tps(et: float, t_cycle_ms: float) -> float:
    return et / (t_cycle_ms / 1e3)


def main() -> int:
    serve = json.loads(SERVE.read_text())
    head = json.loads(HEAD.read_text())
    bfh = serve["arms"]["base_fullhead"]
    osoi5 = serve["arms"]["osoi5_ship"]

    # ---- measured in-serve step-time (STEPTIME, decode phase) ----
    b_exec_gpu = bfh["steptime_exec"]["gpu_mean"]
    o_exec_gpu = osoi5["steptime_exec"]["gpu_mean"]
    b_tcyc, o_tcyc = bfh["t_cycle_ms"], osoi5["t_cycle_ms"]
    b_tps, o_tps = bfh["median_wall_tps"], osoi5["median_wall_tps"]
    b_et, o_et = bfh["e_t_steptime"], osoi5["e_t_steptime"]

    exec_gpu_gap = b_exec_gpu - o_exec_gpu       # body(+5L) + head(262k bf16 - 12k int4)
    tcyc_gap = b_tcyc - o_tcyc
    host_gap_delta = bfh["steptime_exec"]["gap_mean"] - osoi5["steptime_exec"]["gap_mean"]
    draft_gpu_delta = bfh["steptime_draft"]["gpu_mean"] - osoi5["steptime_draft"]["gpu_mean"]
    et_delta = b_et - o_et

    gap_tps_measured = o_tps - b_tps
    gap_tps_anchor = OSOI5_TPS_ANCHOR - BFH_TPS_ANCHOR

    # ---- microbench head costs (ms, rows=8 = M=8 spec verify) ----
    h262k_bf16 = head["head_bf16_ms"]["262144/8"]
    h12k_int4 = head["head_int4_ms"]["12288/8"]
    h262k_int4 = head["head_int4_ms"]["262144/8"]
    mm262_bf16 = head["matmul_bf16_ms"]["262144/8"]
    argmax262 = head["argmax_ms"]["262144/8"]
    eff_hbm = head["eff_hbm_gbps"]
    head_intrinsic_delta = h262k_bf16 - h12k_int4  # direct GPU lower bound on the head tax

    # ---- body / head split of the exec-gpu gap ----
    # osoi5's 37L exec.gpu caps the marginal per-layer cost (overhead>=0):
    #   37*L + head_12k_int4 + overhead = o_exec_gpu  ->  L <= (o_exec_gpu - h12k)/37
    L_cap = (o_exec_gpu - h12k_int4) / 37.0
    body5_cap = 5.0 * L_cap                       # UPPER bound on the +5-layer cost
    # weight-bandwidth per-layer floor (~46 MB int4 weights / eff HBM):
    L_floor_ms = (0.046e9 / (eff_hbm * 1e9)) * 1e3
    body5_floor = 5.0 * L_floor_ms
    body5_central = round((body5_cap + body5_floor) / 2.0, 3)   # ~0.76 ms
    head_serve_central = exec_gpu_gap - body5_central
    head_serve_lo = exec_gpu_gap - body5_cap      # in-serve head LOWER bound (=osoi5 cap)
    head_serve_hi = exec_gpu_gap - body5_floor

    head_pct_central = 100.0 * head_serve_central / exec_gpu_gap
    head_pct_lo = 100.0 * head_serve_lo / exec_gpu_gap
    body_pct_central = 100.0 * body5_central / exec_gpu_gap

    # ---- TPS-additive cross-check (hold E[T]=bfh; remove head then body -> recover osoi5) ----
    tc_minus_head = b_tcyc - head_serve_central
    tc_minus_both = tc_minus_head - body5_central
    tps_after_head = tps(b_et, tc_minus_head)
    tps_after_both = tps(b_et, tc_minus_both)
    head_tps_if_free = tps_after_head - b_tps          # head tax in TPS (if head were free)
    body5_tps_irreducible = tps_after_both - tps_after_head
    crosscheck_recovers_osoi5 = abs(tps_after_both - o_tps)  # should be ~0

    # ---- realistic recoverable: quantize the FULL head (identity-preserving in principle) ----
    # The cost is the dense weight READ (argmax reduction is only ~0.03 ms), so the only
    # identity-preserving lever is lower-precision head weights, NOT a faster argmax.
    save_int4 = h262k_bf16 - h262k_int4                # int4 full head
    save_fp8 = mm262_bf16 * 0.5                         # fp8 = half the bf16 read
    ceil_int4 = tps(b_et, b_tcyc - save_int4)
    ceil_fp8 = tps(b_et, b_tcyc - save_fp8)
    recover_int4 = ceil_int4 - b_tps
    recover_fp8 = ceil_fp8 - b_tps
    # anchor-relative ceiling (PR asks 253.78 + recoverable)
    ceil_int4_anchor = BFH_TPS_ANCHOR + recover_int4
    ceil_fp8_anchor = BFH_TPS_ANCHOR + recover_fp8

    optimized_ceiling = ceil_int4_anchor               # headline (aggressive int4 head)
    irreducible_deficit_vs_unsafe = OSOI5_TPS_ANCHOR - optimized_ceiling

    quality_safe_can_beat_442 = bool(optimized_ceiling >= UNSAFE_FRONTIER_LOCAL)
    # even a FREE head (upper-bound fantasy) check:
    free_head_tps = tps_after_head
    free_head_beats_442 = bool(free_head_tps >= UNSAFE_FRONTIER_LOCAL)
    argmax_tax_dominant = bool(head_pct_lo >= 50.0)    # head >= body, robustly (lower bound 75%)

    out: dict[str, Any] = {
        "schema": "base_fullhead_tps_ceiling_decomposition_v1",
        "analysis_only": True,
        "official_tps": 0,
        "pr": 544,
        "anchors": {
            "bfh_tps_anchor": BFH_TPS_ANCHOR, "bfh_tps_measured": round(b_tps, 3),
            "bfh_ppl_measured": round(bfh["ppl"], 4),
            "osoi5_tps_anchor": OSOI5_TPS_ANCHOR, "osoi5_tps_measured": round(o_tps, 3),
            "osoi5_ppl_measured": round(osoi5["ppl"], 4),
            "unsafe_frontier_local": UNSAFE_FRONTIER_LOCAL, "tau_lo": TAU_LO,
        },
        "gap": {
            "gap_tps_anchor": round(gap_tps_anchor, 2),
            "gap_tps_measured": round(gap_tps_measured, 2),
            "exec_gpu_gap_ms": round(exec_gpu_gap, 3),
            "t_cycle_gap_ms": round(tcyc_gap, 3),
            "host_gap_delta_ms": round(host_gap_delta, 3),
            "draft_gpu_delta_ms": round(draft_gpu_delta, 3),
            "et_delta": round(et_delta, 4),
        },
        "microbench_head": {
            "h262k_bf16_ms_m8": round(h262k_bf16, 4),
            "h12k_int4_ms_m8": round(h12k_int4, 4),
            "h262k_int4_ms_m8": round(h262k_int4, 4),
            "matmul262k_bf16_ms_m8": round(mm262_bf16, 4),
            "argmax262k_ms_m8": round(argmax262, 4),
            "eff_hbm_gbps": round(eff_hbm, 1),
            "head_intrinsic_delta_ms": round(head_intrinsic_delta, 3),
            "head_intrinsic_pct_of_exec_gap": round(100 * head_intrinsic_delta / exec_gpu_gap, 1),
        },
        "decomposition_4way": {
            # C1 body +5 layers (IRREDUCIBLE)
            "c1_body_5layer_ms": round(body5_central, 3),
            "c1_body_5layer_ms_range": [round(body5_floor, 3), round(body5_cap, 3)],
            "c1_body_5layer_pct": round(body_pct_central, 1),
            "c1_body_5layer_tps_irreducible": round(body5_tps_irreducible, 1),
            "c1_per_layer_ms_central": round(body5_central / 5, 4),
            # C2 262k-head verify tax (DOMINANT, partly recoverable)
            "c2_head_verify_tax_ms": round(head_serve_central, 3),
            "c2_head_verify_tax_ms_range": [round(head_serve_lo, 3), round(head_serve_hi, 3)],
            "c2_head_verify_tax_pct": round(head_pct_central, 1),
            "c2_head_verify_tax_pct_range": [round(head_pct_lo, 1),
                                             round(100 * head_serve_hi / exec_gpu_gap, 1)],
            "c2_head_tax_tps_if_free": round(head_tps_if_free, 1),
            "c2_cost_is_logits_matmul_not_argmax": True,
            # C3 MTP drafter E[T] (NON-DRIVER; tiny mitigant)
            "c3_et_delta": round(et_delta, 4),
            "c3_et_is_gap_driver": False,
            "c3_et_tps_mitigant": round(et_delta / (o_tcyc / 1e3), 2),
            # C4 residual (host gap + drafter gpu)
            "c4_residual_ms": round(host_gap_delta + draft_gpu_delta, 3),
        },
        "crosscheck": {
            "method": "hold E[T]=base; remove head then +5L -> must land on osoi5",
            "base_tps": round(b_tps, 1),
            "tps_after_remove_head": round(tps_after_head, 1),
            "tps_after_remove_head_and_5L": round(tps_after_both, 1),
            "osoi5_target_tps": round(o_tps, 1),
            "abs_recovery_error_tps": round(crosscheck_recovers_osoi5, 2),
        },
        "ceiling_model": {
            "lever": "quantize the FULL 262k head (int4/fp8) -- keeps all tokens; "
                     "argmax reduction is already cheap (~0.03 ms), the cost is the dense "
                     "weight read, so only lower-precision head weights help identity-safely",
            "int4_head_saves_ms": round(save_int4, 3),
            "fp8_head_saves_ms": round(save_fp8, 3),
            "recoverable_tps_int4": round(recover_int4, 1),
            "recoverable_tps_fp8": round(recover_fp8, 1),
            "base_fullhead_optimized_tps_ceiling": round(optimized_ceiling, 1),
            "base_fullhead_optimized_tps_ceiling_fp8_conservative": round(ceil_fp8_anchor, 1),
            "base_fullhead_optimized_official_proj": round(optimized_ceiling * TAU_LO, 1),
            "base_fullhead_irreducible_floor": round(optimized_ceiling, 1),
            "base_fullhead_irreducible_5layer_tps": round(body5_tps_irreducible, 1),
            "irreducible_deficit_vs_unsafe_ship_tps": round(irreducible_deficit_vs_unsafe, 1),
            "int4_head_byte_exact_risk": "int4 head perturbs logits -> may flip greedy "
                                         "near-ties (byte-exactness NOT guaranteed); fp8 "
                                         "lower-risk; this ceiling is an UPPER bound",
        },
        "topline": {
            "quality_safe_ship_can_beat_442": quality_safe_can_beat_442,
            "even_free_head_beats_442": free_head_beats_442,
            "free_head_tps_upper_bound": round(free_head_tps, 1),
            "argmax_tax_is_dominant_gap_driver": argmax_tax_dominant,
            "head_tax_pct_central": round(head_pct_central, 1),
        },
    }

    out_path = HERE / "decomposition.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    print(f"\n[decompose] wrote {out_path}", flush=True)

    # ---- wandb (group base-fullhead-tps-ceiling); never let logging discard the result ----
    try:
        from scripts.wandb_logging import init_wandb_run, finish_wandb

        flat = {
            "analysis_only": True, "official_tps": 0,
            "gap_tps_anchor": gap_tps_anchor, "gap_tps_measured": gap_tps_measured,
            "exec_gpu_gap_ms": exec_gpu_gap,
            "bfh_tps_measured": b_tps, "osoi5_tps_measured": o_tps,
            "bfh_et": b_et, "osoi5_et": o_et, "et_delta": et_delta,
            "head_intrinsic_delta_ms": head_intrinsic_delta,
            "c1_body_5layer_ms": body5_central, "c1_body_5layer_pct": body_pct_central,
            "c1_body_5layer_tps_irreducible": body5_tps_irreducible,
            "c2_head_verify_tax_ms": head_serve_central, "c2_head_verify_tax_pct": head_pct_central,
            "c2_head_tax_tps_if_free": head_tps_if_free,
            "recoverable_tps_int4": recover_int4, "recoverable_tps_fp8": recover_fp8,
            "base_fullhead_optimized_tps_ceiling": optimized_ceiling,
            "base_fullhead_optimized_official_proj": optimized_ceiling * TAU_LO,
            "irreducible_deficit_vs_unsafe_ship_tps": irreducible_deficit_vs_unsafe,
            "quality_safe_ship_can_beat_442": int(quality_safe_can_beat_442),
            "argmax_tax_is_dominant_gap_driver": int(argmax_tax_dominant),
            "eff_hbm_gbps": eff_hbm,
        }
        run = init_wandb_run(
            job_type="analysis", agent="lawine",
            name="lawine/base-fullhead-tps-ceiling",
            group="base-fullhead-tps-ceiling",
            notes="PR #544 base_fullhead vs osoi5 step-time gap decomposition + optimized ceiling",
            tags=["pr544", "tps-ceiling", "analysis-only", "step-decomposition"],
            config={"pr": 544, **{f"cfg_{k}": v for k, v in serve.items()
                                  if k in ("k_spec", "num_prompts", "output_len", "seed")}},
        )
        if run is not None:
            run.log({**flat, "global_step": 0})
            run.summary.update(flat)
            print(f"[decompose] wandb run = {run.id} ({run.url})", flush=True)
            finish_wandb(run)
        else:
            print("[decompose] wandb disabled (no API key/mode) -- JSON artifact still written", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[decompose] WARN wandb logging failed (non-fatal): {exc!r}", flush=True)

    # ---- structured terminal marker ----
    marker = {
        "pr": 544, "analysis_only": True, "official_tps": 0,
        "gap_tps": round(gap_tps_anchor, 1),
        "head_verify_tax_pct": round(head_pct_central, 1),
        "body_5layer_pct": round(body_pct_central, 1),
        "et_delta": round(et_delta, 4),
        "head_intrinsic_delta_ms": round(head_intrinsic_delta, 3),
        "base_fullhead_optimized_tps_ceiling": round(optimized_ceiling, 1),
        "base_fullhead_irreducible_floor": round(optimized_ceiling, 1),
        "irreducible_5layer_tps": round(body5_tps_irreducible, 1),
        "quality_safe_ship_can_beat_442": quality_safe_can_beat_442,
        "argmax_tax_is_dominant_gap_driver": argmax_tax_dominant,
    }
    print("SENPAI-CEILING " + json.dumps(marker), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
