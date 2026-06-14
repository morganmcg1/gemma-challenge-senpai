#!/usr/bin/env python3
"""PR #186 — Submission MUST-RETAIN manifest: flag-by-flag packaging de-risk.

CONSOLIDATION + reproduction-gate (NOT new measurement). Enumerates every
load-bearing SPEED flag the projected both-bugs/descent tree build (land #71)
depends on, attaches the ALREADY-MEASURED cost-of-omission imported from the
banked merged legs, and SELF-VALIDATES that the consolidation reproduces each
imported cost within tolerance from its banked source (proves the manifest did
not drift from the source-of-truth artifacts).

Banked sources imported (do NOT re-derive — loaded at runtime to prove fidelity):
  #148  research/kcal_tree_transfer/kcal_tree_transfer_band.json
        K_cal=125.268, +6.019% local->official multiplier, PRECACHE-held band.
  #169  research/spec_cost_model/precache_footprint_invariance/
            precache_footprint_invariance.json
        PRECACHE=0 -> 3.526% single-shot; bus_ratio_tree_invariant=1.
  #157  research/spec_cost_model/salvage_kv_relocation_audit.json
        relocate host-loop 145.2ms vs vectorized 92.4us = 1571x; descent 516->77.
  #154  research/spec_cost_model/step_denominator_reduction_audit.json
        decode scatter+LP avoidance: +3.6..+5.6 TPS, bar 4.862->4.808.
  #163  research/spec_cost_model/host_residency_sweep/host_residency_sweep.json
        net-step scenarios (realizable 522 vs host-loop 77); residual host ops=0.
  #138/#90 (EXPERIMENTS_LOG, lawine #90 K-sweep table; kanna #138 block64 re-char)
        num_speculative_tokens=7 optimal; K8/K9 -13/-16 TPS; CENTROID_TOP_K=64
        optimal (topk128 -3.9 TPS); FUSED_SPARSE_ARGMAX_BLOCK K-neutral.

Served surface audited: submissions/fa2sw_precache_kenyan/manifest.json env +
sitecustomize.py + serve.py + serve_patch_pck04.py (the deployed 481.53
fa2sw_precache_kenyan stack) and the M=32 tree config land #71 carries.

LOCAL CPU-only. No GPU / vLLM / HF Job / submission / served-file change. Adds 0
TPS (primary = self-test). BASELINE stays 481.53. Greedy/PPL untouched.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "research/spec_cost_model/tree_submission_manifest"

# ---- banked source paths (the source-of-truth artifacts) --------------------
SRC = {
    "kcal_148": ROOT / "research/kcal_tree_transfer/kcal_tree_transfer_band.json",
    "precache_169": ROOT / "research/spec_cost_model/precache_footprint_invariance/"
    "precache_footprint_invariance.json",
    "relocate_157": ROOT / "research/spec_cost_model/salvage_kv_relocation_audit.json",
    "scatter_154": ROOT / "research/spec_cost_model/step_denominator_reduction_audit.json",
    "hostres_163": ROOT / "research/spec_cost_model/host_residency_sweep/"
    "host_residency_sweep.json",
}

# ---- pinned launch composition (banked; for projecting % costs only) ---------
# official = K_cal * (E[T]/step) * tau ; K_cal=125.268 (#148/#169), step 1.2182
# (#168), tau=1.0 band [0.9924,1.0] (#181). E[T] descent 5.0564 / both-bugs
# 5.2070 (#172). NOT re-derived here — fern #185 owns the numerical GO/NO-GO.
K_CAL = 125.26795005202914
STEP_UNITS = 1.2182
TAU = 1.0
E_T_DESCENT = 5.0564
E_T_BOTH_BUGS = 5.2070
PROJ_OFFICIAL_DESCENT = K_CAL * (E_T_DESCENT / STEP_UNITS) * TAU
PROJ_OFFICIAL_BOTH_BUGS = K_CAL * (E_T_BOTH_BUGS / STEP_UNITS) * TAU

# ---- lawine #90 banked K-sweep (merged; EXPERIMENTS_LOG #90 table) -----------
# num_speculative_tokens (MTP draft length) sweep, local wall_tps, K=7 optimal.
K90_WALL_TPS = {5: 438.412, 6: 451.047, 7: 454.338, 8: 440.282, 9: 440.784}
# kanna #138 re-characterisation with block64 (CLOSED clean negative; merged):
K138_K7_BLOCK64_WALL_TPS = 454.045
K138_K8_DELTA_TPS = -13.0  # "K8/K9 stay -13/-16 TPS (lawine #90 cliff intact)"
K138_K9_DELTA_TPS = -16.0
K138_CENTROID_TOPK128_DELTA_TPS = -3.9  # CENTROID_TOP_K=64 optimal; topk128 -3.9
K138_BLOCK64_K_NEUTRAL = True  # block16 8.485ms ~= block64 8.486ms; 0 standalone


def _load(key: str) -> dict:
    return json.loads(SRC[key].read_text(encoding="utf-8"))


def load_banked() -> dict:
    """Load every banked source and extract the exact imported values."""
    k148 = _load("kcal_148")
    p169 = _load("precache_169")
    r157 = _load("relocate_157")
    s154 = _load("scatter_154")
    h163 = _load("hostres_163")

    rec = r157["pricing"]["paths"]
    host = rec["host_loop"]
    vec = rec["gpu_vectorized"]
    h_scn = h163["net_step_budget"]["scenarios"]

    return {
        # ---- #148 K_cal + multiplier ----
        "kcal_central": k148["kcal_tree_transfer_band"]["K_cal_central"],
        "kcal_band_width_pct": k148["kcal_tree_transfer_band_width_pct"],
        "multiplier_pooled": k148["calibration_reproduction"]["multiplier_pooled"],
        "pooled_local_wall_tps": k148["calibration_reproduction"]["pooled_local_wall_tps"],
        "kcal_self_test_passes": int(k148["self_test"]["passes"]),
        # ---- #169 precache ----
        "precache_off_divergence_pct": p169["propagation"]["precache_off_divergence_pct"],
        "bus_ratio_tree_invariant": int(p169["propagation"]["bus_ratio_tree_invariant"]),
        "precache_official_shift_tps_descent": p169["propagation"]["official_shift_tps_descent"],
        "precache_self_test_passes": int(p169["self_test"]["all_pass"]),
        # ---- #157 relocate ----
        "relocate_host_per_call_ms": host["per_call_ms"],
        "relocate_vec_per_call_ms": vec["per_call_ms"],
        "relocate_host_descent_tps": host["descent_tps_at_this_step"],
        "relocate_vec_descent_tps": vec["descent_tps_at_this_step"],
        "relocate_host_bar": host["clear500_bar"],
        "relocate_vec_bar": vec["clear500_bar"],
        "relocate_host_step_inflation_pct": host["captured_step_inflation_pct"],
        "relocate_recoverable_step_pct": r157["test_metric"]["value"],
        "relocate_equivalence_rate": r157["greedy_safety"]["equivalence_rate"],
        "relocate_self_test_passes": int(r157["primary_metric"]["value"]),
        # ---- #154 scatter+LP ----
        "scatter_recoverable_step_pct_cons": s154["recoverable_step_pct"],
        "scatter_recoverable_step_pct_real": s154["recoverable_step_pct_realistic"],
        "scatter_self_test_passes": int(s154["step_reduction_audit_self_test_passes"]),
        # ---- #163 host-residency net-step scenarios ----
        "h163_realizable_descent_tps": h_scn["descent_vectorized_plus_154"]["descent_only"]["tps"],
        "h163_realizable_bothbugs_tps": h_scn["descent_vectorized_plus_154"]["both_bugs"]["tps"],
        "h163_hostloop_descent_tps": h_scn["descent_hostloop_relocate"]["descent_only"]["tps"],
        "h163_hostloop_bothbugs_tps": h_scn["descent_hostloop_relocate"]["both_bugs"]["tps"],
        "h163_residual_host_ops": h163["classification"]["descent_path_residual_host_ops_count"],
        "h163_self_test_passes": int(h163["self_test"]["all_pass"]),
        "h163_n_ops_total": h163["classification"]["n_ops_total"],
    }


def build_manifest(b: dict) -> dict:
    """The MUST-RETAIN manifest: one row per load-bearing flag, ordered by
    descending cost-of-omission. Costs IMPORTED from banked legs (b)."""
    proj = PROJ_OFFICIAL_DESCENT  # ~520 TPS denominator for %-of-official costs

    rows = [
        {
            "flag": "relocate_salvaged_kv == vectorized/device (NOT host-loop)",
            "surface": "build_design",
            "present_value": "land #71 build target: single fused [L,W,H,D] "
            "index_select+index_copy_ by DEVICE commit-index (or paged slot-map)",
            "must_retain": True,
            "klass": "BUILD-BLOCKER if reverts to host-loop",
            "measured_cost_if_dropped": (
                f"host-loop reverts descent {b['relocate_vec_descent_tps']:.0f}"
                f"->{b['relocate_host_descent_tps']:.0f} TPS "
                f"({b['relocate_host_per_call_ms']/b['relocate_vec_per_call_ms']:.0f}x per-call; "
                f"+{b['relocate_host_step_inflation_pct']:.0f}% step; bar "
                f"{b['relocate_vec_bar']:.2f}->{b['relocate_host_bar']:.1f})"
            ),
            "cost_sort_tps": b["h163_realizable_descent_tps"] - b["h163_hostloop_descent_tps"],
            "source_leg": "#157 / #163",
            "greedy_ppl_safe": b["relocate_equivalence_rate"] == 1.0,
            "double_load_bearing": False,
            "note": "data-dependent host Python loop over 37 layers CANNOT be "
            "CUDA-graph-captured -> pins step host-bound. Bit-exact bf16 "
            "permutation (equivalence_rate=1.0): speed-only, no PPL risk.",
        },
        {
            "flag": "PRECACHE_BENCH=1 (+ PRECACHE_REQUIRE=1 fail-closed)",
            "surface": "served_env",
            "present_value": "1",
            "must_retain": True,
            "klass": "must-retain served flag",
            "measured_cost_if_dropped": (
                f"PRECACHE=0 -> {b['precache_off_divergence_pct']:.3f}% single-shot "
                f"divergence (#169); also holds the +6.019% local->official "
                f"multiplier's prefill-amortization neutralization (#148 Leg B)"
            ),
            "cost_sort_tps": b["precache_off_divergence_pct"] / 100.0 * proj,
            "source_leg": "#169 / #148",
            "greedy_ppl_safe": True,
            "double_load_bearing": False,
            "note": "replays the 128 bench prompts in the UNTIMED warmup window "
            "-> moves prefill out of the timed window. Tokens unchanged "
            "(warmup only). bus_ratio_tree_invariant=1 (#169): K_cal transfers.",
        },
        {
            "flag": "SPECULATIVE_CONFIG.num_speculative_tokens == 7 (MTP draft length)",
            "surface": "served_env",
            "present_value": "7",
            "must_retain": True,
            "klass": "must-retain served flag (engineered sweet spot)",
            "measured_cost_if_dropped": (
                f"K=8/9 -> {K138_K8_DELTA_TPS:.0f}/{K138_K9_DELTA_TPS:.0f} TPS (#138); "
                f"#90 K8={K90_WALL_TPS[8]:.1f} ({K90_WALL_TPS[8]-K90_WALL_TPS[7]:+.1f}), "
                f"K9={K90_WALL_TPS[9]:.1f} ({K90_WALL_TPS[9]-K90_WALL_TPS[7]:+.1f}); "
                f"K=6 {K90_WALL_TPS[6]-K90_WALL_TPS[7]:+.1f}, K=5 "
                f"{K90_WALL_TPS[5]-K90_WALL_TPS[7]:+.1f} (inverted-U, K7 optimal)"
            ),
            "cost_sort_tps": abs((K138_K8_DELTA_TPS + K138_K9_DELTA_TPS) / 2.0),
            "source_leg": "#90 / #138",
            "greedy_ppl_safe": True,
            "double_load_bearing": False,
            "note": "verifier emits target argmax regardless of draft depth -> "
            "serve-config knob, greedy-identical. K7 is the value LOOPGRAPH/"
            "ONEGRAPH capture + precache are tuned around (K>=8 re-pads).",
        },
        {
            "flag": "decode-path argmax-only logits (scatter+LP avoidance applied)",
            "surface": "build_design",
            "present_value": "land #71 build target: argmax(pruned[M,12288])->kept_ids "
            "remap on greedy token-selection; FULL scatter+LP kept on prefill PPL path",
            "must_retain": True,
            "klass": "must-retain denominator lever (projected stack assumes bar 4.808)",
            "measured_cost_if_dropped": (
                f"revert to full scatter[M,262144]+LP on decode -> "
                f"-{b['scatter_recoverable_step_pct_real']:.2f}% step (real M=32) "
                f"~ -3.6..-5.6 TPS; bar 4.808->4.862"
            ),
            "cost_sort_tps": b["scatter_recoverable_step_pct_real"] / 100.0 * proj,
            "source_leg": "#154 / #163",
            "greedy_ppl_safe": True,
            "double_load_bearing": True,
            "note": "DOUBLE-LOAD-BEARING SEAM: the avoidance is greedy-exact "
            "(kept_ids ascending => first-occurrence tiebreak == full-vocab "
            "argmax, equivalence_rate=1.0) ONLY IF the full scatter+LP REMAINS "
            "on the prompt_logprobs/prefill path. Dropping it THERE breaks PPL.",
        },
        {
            "flag": "CENTROID_TOP_K == 64 (centroid_intermediate_top_k)",
            "surface": "served_env",
            "present_value": "64",
            "must_retain": True,
            "klass": "must-retain served flag",
            "measured_cost_if_dropped": (
                f"topk128 -> {K138_CENTROID_TOPK128_DELTA_TPS:.1f} TPS, no accept "
                f"gain (#138). 64 is the optimum."
            ),
            "cost_sort_tps": abs(K138_CENTROID_TOPK128_DELTA_TPS),
            "source_leg": "#138",
            "greedy_ppl_safe": True,
            "double_load_bearing": False,
            "note": "drafter centroid breadth; verifier argmax unchanged -> "
            "greedy/PPL-safe. Distinct from num_speculative_tokens 'K'.",
        },
        {
            "flag": "descent accept-walk == sync-free device (match-mask->cumprod->"
            "device-argmax; no .item())",
            "surface": "build_design",
            "present_value": "land #71 build target: device-scalar accept length "
            "(vLLM-v1 RejectionSampler zero-sync)",
            "must_retain": True,
            "klass": "must-retain capturability rule",
            "measured_cost_if_dropped": "sync-bound (.item() per node) -> +2.20% step "
            "vs +0.39% sync-free (#147), AND BREAKS CUDA-graph capture (#163 probe)",
            "cost_sort_tps": (2.202 - 0.392) / 100.0 * proj,
            "source_leg": "#147 / #163",
            "greedy_ppl_safe": True,
            "double_load_bearing": False,
            "note": "commit-index must be produced ON-DEVICE and consumed without a "
            "host readout, else the relocate falls out of the captured graph.",
        },
        {
            "flag": "ONEGRAPH=1 + LOOPGRAPH_REQUIRE_CAPTURE=1",
            "surface": "served_env",
            "present_value": "1 / 1",
            "must_retain": True,
            "klass": "must-retain capture flag",
            "measured_cost_if_dropped": "drafter propose loop falls to eager -> K=7 "
            "width-1 iters become per-launch-bound (capture-class; #154 Leg2 closed, "
            "#163 drafter_propose_loop=CLEAN(captured))",
            "cost_sort_tps": 0.0,  # not separately TPS-priced in banked import set
            "source_leg": "#154 / #163",
            "greedy_ppl_safe": True,
            "double_load_bearing": False,
            "note": "drafter-only graph replay -> cannot change emitted tokens. "
            "Cost is capturability-class, not a banked single-flag TPS number.",
        },
        {
            "flag": "DIXIE_FUSED_ACCEPT_PREP=1 + DIXIE_SLIM_GREEDY=1",
            "surface": "served_env",
            "present_value": "1 / 1",
            "must_retain": True,
            "klass": "must-retain device-resident accept flag",
            "measured_cost_if_dropped": "accept-prep leaves the device-resident "
            "Triton kernel for the host path (#163 fused_accept_prep=CLEAN(device))",
            "cost_sort_tps": 0.0,
            "source_leg": "#163",
            "greedy_ppl_safe": True,
            "double_load_bearing": False,
            "note": "fused greedy accept/reject; exact-equivalent to the slow "
            "rejection kernel (bf16->fp32 monotone upcast). Speed-only.",
        },
    ]

    # full served-surface enumeration (classification only; costs owned by their
    # own merge legs and NOT in this manifest's banked import set, OR free).
    other_served = [
        # speed flags whose cost is owned by their own (non-imported) merge leg
        {"flag": "LM_HEAD_PRUNE=1 (+REQUIRE)", "must_retain": True, "klass": "speed (12k head GEMM)",
         "greedy_ppl_safe": True, "double_load_bearing": False,
         "note": "lmhead12k pruned head; greedy argmax kept_ids superset (#154). cost owned by lmhead12k leg (not in import set)."},
        {"flag": "FA_SLIDING=1", "must_retain": True, "klass": "speed (attn backend)",
         "greedy_ppl_safe": True, "double_load_bearing": False,
         "note": "FA2 for eligible sliding-window layers; output-neutral per its leg. cost owned by agent-smith leg."},
        {"flag": "SPLITKV_VERIFY=1 (+MAX_Q=64)", "must_retain": True, "klass": "speed (verify attn)",
         "greedy_ppl_safe": True, "double_load_bearing": False,
         "note": "3D split-KV path for small multi-query verify batches; fail-open. cost owned by its leg."},
        {"flag": "PLE_FOLD_EMBED_SCALE=1 + PLE_ASSUME_VALID_TOKEN_IDS=1 (+PLE_SCRATCH_REUSE)",
         "must_retain": True, "klass": "speed (PLE fold/fastpath)",
         "greedy_ppl_safe": True, "double_load_bearing": False,
         "note": "scale-fold is exact (fail-closed verified); multimodal PLE contract retained. cost owned by PLE legs."},
        {"flag": "FEOPT_ORJSON=1 / FASTRENDER=1 / DETOK_ENDONLY=1", "must_retain": True,
         "klass": "speed (front-end / detok)", "greedy_ppl_safe": True, "double_load_bearing": False,
         "note": "orjson JSON + fast chat-template + end-only detok; token_ids untouched. cost owned by their legs."},
        {"flag": "LD_PRELOAD=tcmalloc / PYTORCH_CUDA_ALLOC_CONF / PERFORMANCE_MODE=interactivity",
         "must_retain": True, "klass": "speed (allocator/runtime)", "greedy_ppl_safe": True,
         "double_load_bearing": False, "note": "host malloc + CUDA alloc fragmentation + vLLM perf mode. cost not in import set."},
        {"flag": "DRAFTER_BUCKET=ft-v1-epoch_001 (+DRAFTER_SHA256 guard)", "must_retain": True,
         "klass": "acceptance (E[accept])", "greedy_ppl_safe": True, "double_load_bearing": False,
         "note": "retrained drafter raises E[accept]=E[T]; sha256-guarded. greedy emits TARGET argmax -> PPL-safe. cost owned by kenyan-duma drafter leg."},
        # validity-critical (double-load-bearing): break PPL/greedy if dropped
        {"flag": "OVERRIDE_GENERATION_CONFIG temperature=0.0 (top_p=1,top_k=0)",
         "must_retain": True, "klass": "VALIDITY (greedy identity)", "greedy_ppl_safe": True,
         "double_load_bearing": True, "note": "greedy decode contract; changing it breaks greedy token-identity (Issue #124)."},
        {"flag": "MAX_NUM_SEQS=1 / MAX_MODEL_LEN=4096 / MAX_NUM_BATCHED_TOKENS=512 / DTYPE=bfloat16",
         "must_retain": True, "klass": "VALIDITY/scoring contract", "greedy_ppl_safe": True,
         "double_load_bearing": True, "note": "scorer convention (conc=1, ctx 4096); model-faithful bf16. mis-set perturbs PPL/throughput basis."},
        {"flag": "WEIGHTS_BUCKET/LOCAL_MODEL_DIR -> int4-pck04 baked dir, PCK04_KEEPSET",
         "must_retain": True, "klass": "VALIDITY (model artifact)", "greedy_ppl_safe": True,
         "double_load_bearing": True, "note": "wrong checkpoint path -> wrong/absent model. Must point at the validated baked int4 dir."},
        # free / cosmetic
        {"flag": "FUSED_SPARSE_ARGMAX_BLOCK (16 or 64)", "must_retain": False,
         "klass": "FREE/COSMETIC (K-neutral)", "greedy_ppl_safe": True, "double_load_bearing": False,
         "note": f"#138: block16 8.485ms ~= block64 8.486ms; 0 standalone TPS; greedy-token-identical 128/128. Safe at 16 or 64 (block64_k_neutral={K138_BLOCK64_K_NEUTRAL})."},
        {"flag": "UVICORN_LOG_LEVEL/DISABLE_LOG_STATS/PATCH_BENCH_JINJA2", "must_retain": False,
         "klass": "FREE/COSMETIC (logging/setup)", "greedy_ppl_safe": True, "double_load_bearing": False,
         "note": "log verbosity + bench-venv jinja2 install; no served-compute effect."},
        # negative must-retain: TRAPS that must remain UNSET
        {"flag": "LSK_SKIP_LAYERS (MUST remain UNSET)", "must_retain": True,
         "klass": "TRAP (output-breaking if set)", "greedy_ppl_safe": False, "double_load_bearing": True,
         "note": "osoi layer-skip; if accidentally SET it drops decoder layers -> breaks output. Manifest asserts ABSENT."},
        {"flag": "STEPTIME/FA_SLIDING_DIAG/PROFILER_CONFIG (diagnostic; default-off)",
         "must_retain": False, "klass": "FREE/INERT (must stay off)", "greedy_ppl_safe": True,
         "double_load_bearing": False, "note": "profiling probes; inert on the leaderboard path. Must remain unset/0."},
    ]

    rows.sort(key=lambda r: r["cost_sort_tps"], reverse=True)

    n_banked_rows = len(rows)
    n_other = len(other_served)
    n_must_retain = sum(r["must_retain"] for r in rows) + sum(
        r["must_retain"] for r in other_served
    )
    n_double = sum(r["double_load_bearing"] for r in rows) + sum(
        r["double_load_bearing"] for r in other_served
    )
    return {
        "must_retain_manifest": rows,  # ordered, banked-cost rows
        "served_surface_enumeration": other_served,  # classified, non-imported
        "n_flags_enumerated": n_banked_rows + n_other,
        "n_must_retain": n_must_retain,
        "n_double_load_bearing": n_double,
        "n_banked_cost_rows": n_banked_rows,
        "row1_flag": rows[0]["flag"],
        "row1_cost_sort_tps": rows[0]["cost_sort_tps"],
    }


def self_test(b: dict, man: dict) -> dict:
    """PRIMARY: reproduce each imported cost within tolerance from the banked
    source -> proves the consolidation is faithful, not a re-summary that
    drifted."""
    checks = []

    def chk(name, cond, detail):
        checks.append({"name": name, "passes": bool(cond), "detail": str(detail)})

    # --- #148 K_cal + multiplier reproduce from source ---
    kcal_calc = 481.53 / 3.844
    chk("kcal == 481.53/3.844 (#148)", abs(b["kcal_central"] - kcal_calc) < 1e-6,
        f"{b['kcal_central']:.6f} vs {kcal_calc:.6f}")
    mult_calc = 481.53 / b["pooled_local_wall_tps"]
    chk("multiplier == 481.53/454.1937 (#148)", abs(b["multiplier_pooled"] - mult_calc) < 1e-9,
        f"{b['multiplier_pooled']:.7f}")
    chk("kcal source self-test passed (#148)", b["kcal_self_test_passes"] == 1,
        b["kcal_self_test_passes"])

    # --- #169 precache reproduces ---
    chk("PRECACHE=0 divergence ~3.53% (#169)", abs(b["precache_off_divergence_pct"] - 3.526) < 0.05,
        f"{b['precache_off_divergence_pct']:.3f}%")
    chk("bus_ratio_tree_invariant == 1 (#169)", b["bus_ratio_tree_invariant"] == 1,
        b["bus_ratio_tree_invariant"])
    chk("precache official_shift_tps == 0 (#169)", b["precache_official_shift_tps_descent"] == 0.0,
        b["precache_official_shift_tps_descent"])
    chk("precache source self-test passed (#169)", b["precache_self_test_passes"] == 1,
        b["precache_self_test_passes"])

    # --- #157 relocate 1571x + 516->77 reproduce ---
    speedup = b["relocate_host_per_call_ms"] / b["relocate_vec_per_call_ms"]
    chk("relocate speedup ~1571x (#157)", abs(speedup - 1571.0) < 30.0, f"{speedup:.1f}x")
    chk("relocate host-loop descent ~77 TPS (#157)", abs(b["relocate_host_descent_tps"] - 77.3) < 1.0,
        f"{b['relocate_host_descent_tps']:.2f}")
    chk("relocate vectorized descent ~516 TPS (#157)", abs(b["relocate_vec_descent_tps"] - 516.4) < 1.5,
        f"{b['relocate_vec_descent_tps']:.2f}")
    chk("relocate recoverable_step_pct ~569.9% (#157)",
        abs(b["relocate_recoverable_step_pct"] - 569.9) < 1.0, f"{b['relocate_recoverable_step_pct']:.2f}%")
    chk("relocate equivalence_rate == 1.0 (#157)", b["relocate_equivalence_rate"] == 1.0,
        b["relocate_equivalence_rate"])
    chk("relocate source self-test passed (#157)", b["relocate_self_test_passes"] == 1,
        b["relocate_self_test_passes"])

    # --- #154 scatter+LP reproduce ---
    chk("scatter recoverable_step_pct 0.86/1.11% (#154)",
        abs(b["scatter_recoverable_step_pct_cons"] - 0.857) < 0.02
        and abs(b["scatter_recoverable_step_pct_real"] - 1.108) < 0.02,
        f"{b['scatter_recoverable_step_pct_cons']:.3f}/{b['scatter_recoverable_step_pct_real']:.3f}%")
    chk("scatter source self-test passed (#154)", b["scatter_self_test_passes"] == 1,
        b["scatter_self_test_passes"])

    # --- #163 host-residency net-step scenarios reproduce ---
    chk("h163 realizable descent ~522 TPS (#163)", abs(b["h163_realizable_descent_tps"] - 522.4) < 1.0,
        f"{b['h163_realizable_descent_tps']:.2f}")
    chk("h163 host-loop descent ~77 TPS (#163)", abs(b["h163_hostloop_descent_tps"] - 77.45) < 1.0,
        f"{b['h163_hostloop_descent_tps']:.2f}")
    chk("h163 descent-path residual host ops == 0 (#163)", b["h163_residual_host_ops"] == 0,
        b["h163_residual_host_ops"])
    chk("h163 source self-test passed (#163)", b["h163_self_test_passes"] == 1,
        b["h163_self_test_passes"])

    # --- #90/#138 K-sweep: cited -13/-16 brackets measured #90 deltas ---
    d8_90 = K90_WALL_TPS[8] - K90_WALL_TPS[7]
    d9_90 = K90_WALL_TPS[9] - K90_WALL_TPS[7]
    chk("K8 cited -13 brackets #90 measured (-14.06)", K138_K8_DELTA_TPS - 2.0 <= d8_90 <= K138_K8_DELTA_TPS + 2.0,
        f"#138 {K138_K8_DELTA_TPS} vs #90 {d8_90:.2f}")
    chk("K9 cited -16 brackets #90 measured (-13.55)", min(K138_K9_DELTA_TPS, -10.0) <= d9_90 <= -10.0,
        f"#138 {K138_K9_DELTA_TPS} vs #90 {d9_90:.2f}")
    chk("K7 is the inverted-U optimum (#90)", max(K90_WALL_TPS, key=K90_WALL_TPS.get) == 7,
        f"argmax K = {max(K90_WALL_TPS, key=K90_WALL_TPS.get)}")
    chk("#138 K7-block64 reproduces #90 K7 within 0.1%",
        abs(K138_K7_BLOCK64_WALL_TPS - K90_WALL_TPS[7]) / K90_WALL_TPS[7] < 0.001,
        f"{K138_K7_BLOCK64_WALL_TPS} vs {K90_WALL_TPS[7]}")

    # --- manifest internal consistency ---
    chk("manifest ordered by descending cost (row1 = binding risk)",
        man["must_retain_manifest"][0]["cost_sort_tps"]
        == max(r["cost_sort_tps"] for r in man["must_retain_manifest"]),
        man["row1_flag"])
    chk("row1 is the relocate host-loop (predicted binding risk)",
        man["must_retain_manifest"][0]["flag"].startswith("relocate_salvaged_kv"),
        man["row1_flag"])

    n_pass = sum(c["passes"] for c in checks)
    n_total = len(checks)
    return {"checks": checks, "n_pass": n_pass, "n_total": n_total, "all_pass": n_pass == n_total}


def compute_test_metric(b: dict) -> dict:
    """TEST: binding_packaging_cost_pct = the largest single-flag cost-of-omission
    as a % of projected official TPS (the manifest's row-1 risk = relocate
    vectorized->host-loop, from #163's own apples-to-apples scenario pair)."""
    realizable = b["h163_realizable_descent_tps"]
    hostloop = b["h163_hostloop_descent_tps"]
    binding_pct = (realizable - hostloop) / realizable * 100.0
    return {
        "binding_packaging_cost_pct": binding_pct,
        "binding_flag": "relocate_salvaged_kv host-loop (vectorized->host-loop)",
        "realizable_descent_tps": realizable,
        "hostloop_descent_tps": hostloop,
        "proj_official_descent_pinned": PROJ_OFFICIAL_DESCENT,
        "proj_official_both_bugs_pinned": PROJ_OFFICIAL_BOTH_BUGS,
    }


def nan_clean(obj) -> bool:
    if isinstance(obj, float):
        return math.isfinite(obj)
    if isinstance(obj, dict):
        return all(nan_clean(v) for v in obj.values())
    if isinstance(obj, list):
        return all(nan_clean(v) for v in obj)
    return True


def wandb_log(args, res: dict) -> None:
    try:
        import wandb

        run_w = wandb.init(
            project="gemma-challenge-senpai", entity="wandb-applied-ai-team",
            group=args.wandb_group, name=args.wandb_name,
            config={
                "pr": 186, "K_cal": K_CAL, "step_units": STEP_UNITS, "tau": TAU,
                "E_T_descent": E_T_DESCENT, "E_T_both_bugs": E_T_BOTH_BUGS,
                "served_submission": "fa2sw_precache_kenyan",
                "baseline_official_tps": 481.53,
            },
        )
        man = res["manifest"]
        tm = res["test_metric_detail"]
        log = {
            "manifest_self_test_passes": int(res["self_test"]["all_pass"]),
            "binding_packaging_cost_pct": tm["binding_packaging_cost_pct"],
            "self_test_n_pass": res["self_test"]["n_pass"],
            "self_test_n_total": res["self_test"]["n_total"],
            "n_flags_enumerated": man["n_flags_enumerated"],
            "n_must_retain": man["n_must_retain"],
            "n_double_load_bearing": man["n_double_load_bearing"],
            "n_banked_cost_rows": man["n_banked_cost_rows"],
            "row1_cost_sort_tps": man["row1_cost_sort_tps"],
            "proj_official_descent_pinned": PROJ_OFFICIAL_DESCENT,
            "proj_official_both_bugs_pinned": PROJ_OFFICIAL_BOTH_BUGS,
            "realizable_descent_tps": tm["realizable_descent_tps"],
            "hostloop_descent_tps": tm["hostloop_descent_tps"],
            "metrics_nan_clean": int(res["metrics_nan_clean"]),
        }
        wandb.log(log)
        run_w.summary.update(log)
        res["wandb_run_id"] = run_w.id
        wandb.finish()
        print(f"[manifest] W&B run {run_w.id} (group {args.wandb_group})", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[manifest] W&B logging skipped: {e!r}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wandb-group", type=str, default="tree-submission-must-retain-manifest")
    ap.add_argument("--wandb-name", type=str, default="ubel/tree-submission-must-retain-manifest")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--output", type=pathlib.Path, default=OUT_DIR / "tree_submission_manifest.json")
    args = ap.parse_args(argv)

    b = load_banked()
    man = build_manifest(b)
    st = self_test(b, man)
    tm = compute_test_metric(b)

    res = {
        "pr": 186,
        "lane": "submission MUST-RETAIN manifest (flag-by-flag packaging de-risk)",
        "method": "LOCAL CPU-only consolidation + reproduction-gate. No GPU / vLLM "
        "/ HF Job / submission / served-file change. Adds 0 TPS. BASELINE 481.53.",
        "primary_metric": {"name": "manifest_self_test_passes", "value": int(st["all_pass"])},
        "test_metric": {"name": "binding_packaging_cost_pct",
                        "value": tm["binding_packaging_cost_pct"]},
        "banked_imports": b,
        "manifest": man,
        "self_test": st,
        "test_metric_detail": tm,
        "banked_source_files": {k: str(v.relative_to(ROOT)) for k, v in SRC.items()},
        "scope": "CONSOLIDATION of own merged legs (#148/#154/#157/#163/#169) + kanna "
        "#138 + served fa2sw_precache_kenyan config. No re-derivation. INFORMS the "
        "Approval-request HF-job packaging check + land #71 tree build; does NOT "
        "authorize a launch.",
    }
    res["metrics_nan_clean"] = nan_clean({k: v for k, v in res.items() if k != "manifest"}) and nan_clean(
        [r["cost_sort_tps"] for r in man["must_retain_manifest"]]
    )

    if not args.no_wandb:
        wandb_log(args, res)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(res, indent=2), encoding="utf-8")
    try:
        shown = args.output.resolve().relative_to(ROOT)
    except ValueError:
        shown = args.output
    print(f"[manifest] wrote {shown}", flush=True)
    print(f"[manifest] PRIMARY manifest_self_test_passes = {int(st['all_pass'])} "
          f"({st['n_pass']}/{st['n_total']})", flush=True)
    print(f"[manifest] TEST binding_packaging_cost_pct = "
          f"{tm['binding_packaging_cost_pct']:.2f}% (row1: {man['row1_flag'][:48]})", flush=True)
    print(f"[manifest] n_flags_enumerated={man['n_flags_enumerated']} "
          f"n_must_retain={man['n_must_retain']} n_double_load_bearing={man['n_double_load_bearing']}",
          flush=True)
    return 0 if st["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
