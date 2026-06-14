# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Draft-verify overlap gate (PR #94) -- LOCAL analysis only, no HF Job, no GPU.

Can the ~15.5% drafter forward be hidden behind verify on a *secondary CUDA
stream* (Saguaro-style parallel spec-decode, arXiv:2603.03251) at conc=1 on a
bandwidth-bound A10G?  Three steps:

  Step 1  timing gate     r = drafter_step_time / verify_step_time   (M=8, M=32)
                          + naive compute-limited overlap ceiling.
  Step 2  HBM-bandwidth contention -- does (drafter_bytes + verify_bytes)/HBM
                          fit inside verify_step_time?  -> bandwidth-limited
                          overlap ceiling vs the naive ceiling.  THE crux.
  Step 3  gate            GREEN >=5% / AMBER 1-5% / RED <1% or r>0.85,
                          plus the serial accept-boundary realizability haircut.

Everything is composed from this author's already-MERGED measured artifacts:
  - frontier_decode  : measured drafter/verify STEPTIME + decode composition (#69/#30)
  - #68 verify_gemm_roofline.json   : verify-GEMM bytes / achieved HBM bandwidth
  - #75 drafter_forward_roofline.json : drafter GEMM-chain bytes
  - #85 tree_nongemm_overhead       : M=8->M=32 attention amortization + tree overhead
  - accept_calibration              : measured accept-length distribution (E[T]=3.844)

Step 2's analytic bytes-roofline test ("does (drafter_bytes + verify_bytes)/HBM
still fit within verify_step_time") PASSES on time-average -- but that is
NECESSARY-NOT-SUFFICIENT: verify averages ~48% HBM yet its 66%-of-wall GEMM core
runs at 77% HBM, and a naive secondary stream cannot pin the drafter into the
brief bus-idle windows.  The crux is whether two memory-bound streams overlap or
contend on the one A10G bus, which is *measured* by the companion probe
scripts/profiler/dual_stream_hbm_contention.py (GPU) -> dual_stream_contention.json.
This script reads that measured drafter_overlap_efficiency and scales the naive
ceiling by it; it remains pure-CPU (no GPU, no HF Job) itself.
"""
import argparse
import json
import os

HBM_GBS = 600.0  # A10G HBM roofline (datasheet, used throughout #68/#75/#85)

# ---- MEASURED inputs (frontier_decode profile, conc=1, fa2sw_precache_kenyan) ----
# research/profiling/frontier_decode/frontier_decode_profile.json  (#69/#30)
DRAFTER_MS_M8 = 1.446      # STEPTIME kind=draft p50 (the "15.5% block")
VERIFY_MS_M8 = 7.906       # STEPTIME kind=exec  p50
GPU_BUSY_MS_M8 = 9.352     # drafter + verify
FRAC_DRAFTER = 0.15461933
FRAC_VERIFY_GEMM = 0.53158297
FRAC_VERIFY_ATTN = 0.19629872
FRAC_VERIFY_NORM = 0.06659784
FRAC_VERIFY_LMHEAD = 0.00962361
# sampling = remainder of gpu-busy
FRAC_SAMPLING = 1.0 - (FRAC_DRAFTER + FRAC_VERIFY_GEMM + FRAC_VERIFY_ATTN
                       + FRAC_VERIFY_NORM + FRAC_VERIFY_LMHEAD)

# bases for TPS projection (deployed frontier; projections, not fresh submissions)
LOCAL_WALLTPS = 454.09
OFFICIAL_TPS = 481.53

# ---- non-GEMM verify byte estimate (small; bandwidth-idle slack region) ----
# KV cache read/step ~22.7 MB (#85), pruned lm_head int4 ~15.7 MB (1.0% decode,
# bandwidth-model 0.0263ms@600 => ~16MB), activations/norms ~few MB.
VERIFY_NONGEMM_BYTES = (22.7 + 15.7 + 6.0) * 1e6  # ~0.044 GB

# drafter non-GEMM bytes: embed gather 8192x256x2=4.2MB (#77) + small KV/act
DRAFTER_NONGEMM_BYTES = (4.2 + 6.0) * 1e6  # ~0.010 GB


def load(p):
    with open(p) as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-json",
                    default="research/spec_cost_model/verify_gemm_roofline.json")
    ap.add_argument("--drafter-json",
                    default="research/spec_cost_model/drafter_forward_roofline.json")
    ap.add_argument("--accept-json",
                    default="research/accept_calibration/accept_calibration_results.json")
    ap.add_argument("--contention-json",
                    default="research/draft_verify_overlap/dual_stream_contention.json",
                    help="measured A10G dual-stream HBM contention probe (Step-2 crux)")
    ap.add_argument("--output",
                    default="research/draft_verify_overlap/overlap_gate.json")
    ap.add_argument("--green-pct", type=float, default=5.0)
    ap.add_argument("--amber-pct", type=float, default=1.0)
    ap.add_argument("--r-close", type=float, default=0.85)
    ap.add_argument("--wandb_project",
                    default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity",
                    default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="draft-verify-overlap-gate")
    ap.add_argument("--wandb_name", default="denken/draft-verify-overlap-gate")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--log-only", default=None,
                    help="replay an existing JSON payload to W&B (no recompute)")
    args = ap.parse_args()

    if args.log_only:
        _log_wandb(args, load(args.log_only))
        return

    vj = load(args.verify_json)
    dj = load(args.drafter_json)
    aj = load(args.accept_json)
    # measured A10G dual-stream contention (Step-2 crux).  If absent, fall back to
    # the analytic byte-fit test but flag the ceiling as UNMEASURED/optimistic.
    cj = load(args.contention_json) if os.path.exists(args.contention_json) else None

    # ================= verify-side bytes / bandwidth (#68) =================
    vagg = vj["aggregate_by_M"]
    v_gemm_us_m8 = vagg["8"]["total_gemm_us"]
    v_gemm_gbs_m8 = vagg["8"]["agg_gbytes_s"]
    v_gemm_pct_hbm_m8 = vagg["8"]["agg_pct_hbm_peak"]
    v_gemm_bytes = v_gemm_gbs_m8 * 1e9 * v_gemm_us_m8 * 1e-6      # bytes/step
    v_gemm_us_m32 = vagg["32"]["total_gemm_us"]
    v_gemm_scale_8_32 = v_gemm_us_m32 / v_gemm_us_m8             # 1.184x (#68)

    verify_bytes = v_gemm_bytes + VERIFY_NONGEMM_BYTES
    verify_avg_gbs = verify_bytes / (VERIFY_MS_M8 * 1e-3)
    verify_avg_pct_hbm = 100.0 * verify_avg_gbs / 1e9 / HBM_GBS

    # verify time split: GEMM core (bandwidth-saturated) vs non-GEMM (bw-idle)
    v_gemm_ms = FRAC_VERIFY_GEMM * GPU_BUSY_MS_M8
    v_nongemm_ms = VERIFY_MS_M8 - v_gemm_ms

    # ================= drafter-side bytes (#75) =================
    d_chain_bytes = dj["chain"]["per_pass_total_bytes"] * 7      # 7 passes
    d_chain_pct_hbm = dj["chain"]["chain_pct_hbm_peak_graph"]    # 47.2%
    drafter_bytes = d_chain_bytes + DRAFTER_NONGEMM_BYTES
    drafter_avg_gbs = drafter_bytes / (DRAFTER_MS_M8 * 1e-3)
    drafter_avg_pct_hbm = 100.0 * drafter_avg_gbs / 1e9 / HBM_GBS

    # ================= STEP 1 -- timing gate =================
    r_m8 = DRAFTER_MS_M8 / VERIFY_MS_M8

    # M=32 projection.  drafter grows by the M-row tree-candidate sampler
    # (#85 centroid_sampler_Mrows 85.18->241.97us => +0.157ms); verify grows by
    # GEMM (x1.184, #68) + attention amortized x1.06 (#85) + argmax(+0.037ms #85).
    drafter_ms_m32 = DRAFTER_MS_M8 + (241.97 - 85.18) * 1e-3
    verify_ms_m32 = (v_gemm_ms * v_gemm_scale_8_32          # GEMM widens
                     + FRAC_VERIFY_ATTN * GPU_BUSY_MS_M8 * 1.06   # attn amortizes
                     + FRAC_VERIFY_NORM * GPU_BUSY_MS_M8           # ~flat
                     + FRAC_SAMPLING * GPU_BUSY_MS_M8 + 0.037      # +argmax M-rows
                     + FRAC_VERIFY_LMHEAD * GPU_BUSY_MS_M8         # ~flat (bw-bound)
                     + 0.035)                                      # verify-side tree glue
    r_m32 = drafter_ms_m32 / verify_ms_m32
    gpu_busy_m32 = drafter_ms_m32 + verify_ms_m32

    def naive_ceiling(drafter, verify):
        total = drafter + verify
        hide = min(drafter, verify)                  # PR formula
        wall_pct = 100.0 * hide / total
        cycle_overlapped = max(drafter, verify)      # steady-state when hidden
        tps_pct = 100.0 * (total / cycle_overlapped - 1.0)
        return wall_pct, tps_pct, cycle_overlapped

    naive_wall_m8, naive_tps_m8, cyc_ov_m8 = naive_ceiling(DRAFTER_MS_M8, VERIFY_MS_M8)
    naive_wall_m32, naive_tps_m32, cyc_ov_m32 = naive_ceiling(drafter_ms_m32, verify_ms_m32)

    # ================= STEP 2 -- HBM-bandwidth contention =================
    combined_bytes = drafter_bytes + verify_bytes
    bw_floor_ms = combined_bytes / (HBM_GBS * 1e9) * 1e3        # ms at peak HBM
    fits_in_verify = bw_floor_ms <= VERIFY_MS_M8
    bw_margin_ms = VERIFY_MS_M8 - bw_floor_ms

    # slack capacity in verify's bandwidth-idle non-GEMM window
    nongemm_used_gbs = VERIFY_NONGEMM_BYTES / (v_nongemm_ms * 1e-3)
    nongemm_slack_gbs = HBM_GBS - nongemm_used_gbs / 1e9
    nongemm_slack_bytes = nongemm_slack_gbs * 1e9 * (v_nongemm_ms * 1e-3)
    # slack during the GEMM core (shared bus headroom under peak).
    # NOTE: vagg agg_gbytes_s is already GB/s; drafter_avg_gbs is bytes/s.
    gemm_slack_gbs = HBM_GBS - v_gemm_gbs_m8
    gemm_slack_bytes = gemm_slack_gbs * 1e9 * (v_gemm_ms * 1e-3)
    drafter_fits_nongemm_x = nongemm_slack_bytes / drafter_bytes
    drafter_fits_gemm_x = gemm_slack_bytes / drafter_bytes
    # does drafter co-running with the GEMM stay under peak bus?
    gemm_plus_drafter_gbs = v_gemm_gbs_m8 + drafter_avg_gbs / 1e9
    gemm_coexists = gemm_plus_drafter_gbs <= HBM_GBS

    # --- byte-fit test (NECESSARY-NOT-SUFFICIENT) ---
    # The combined bytes fit inside the verify wall when amortized over the whole
    # step (verify averages ~48% HBM, #68), and the featherweight drafter fits the
    # time-averaged non-GEMM slack many times over.  This *byte-fit* PASS is the
    # PR's analytic Step-2 test -- but it is a time-AVERAGING artifact: a naive
    # secondary stream cannot pin the drafter into verify's brief non-GEMM windows.
    # The bus is saturated during the 66%-of-wall GEMM core (77% HBM, #68), so two
    # memory-bound streams contend.  Byte-fit is necessary, not sufficient.
    byte_fit_ok = fits_in_verify and drafter_fits_nongemm_x >= 1.0

    # --- MEASURED bandwidth-limited ceiling (Step-2 crux) ---
    # The dual-stream probe runs a verify-sized GEMM (2.25 GB) and a drafter-sized
    # GEMM (0.16 GB) on two concurrent CUDA streams on THIS A10G and measures how
    # much of the drafter actually hides.  drafter_overlap_efficiency in [0,1]:
    #   1.0 -> drafter fully hidden (bus had real slack)
    #   0.0 -> fully serialized (HBM bus contention)
    # The bandwidth-limited overlap ceiling is the naive ceiling scaled by the
    # measured fraction of the drafter that the bus actually lets hide.
    if cj is not None:
        overlap_eff = cj["drafter_overlap_efficiency"]
        sym_speedup = cj["symmetric_overlap_speedup"]
        contention_factor = cj["bus_contention_factor"]
        contention_source = "measured_a10g_dual_stream_probe"
    else:
        # no probe -> fall back to byte-fit (optimistic; flagged UNMEASURED)
        overlap_eff = 1.0 if byte_fit_ok else 0.0
        sym_speedup = float("nan")
        contention_factor = float("nan")
        contention_source = "UNMEASURED_byte_fit_fallback"

    bw_ceiling_wall_m8 = overlap_eff * naive_wall_m8
    # overlapped steady-state cycle after only `overlap_eff` of the drafter hides
    drafter_hidden_ms = overlap_eff * DRAFTER_MS_M8
    bw_overlapped_cycle = GPU_BUSY_MS_M8 - drafter_hidden_ms
    bw_ceiling_tps_m8 = 100.0 * (GPU_BUSY_MS_M8 / bw_overlapped_cycle - 1.0)

    # ================= STEP 3a -- serial accept-boundary realizability =================
    C = aj["server_log_metrics"]["cumulative_acceptance_C"]      # P(>= k+1 draft accepted)
    K = aj["server_log_metrics"]["num_speculative_tokens"]
    E_T = aj["server_log_metrics"]["mean_tokens_per_step_E_T"]
    # pmf of #accepted draft tokens j = 0..K
    pmf = [1.0 - C[0]] + [C[i - 1] - C[i] for i in range(1, K)] + [C[K - 1]]
    pmf_sorted = sorted(enumerate(pmf), key=lambda kv: -kv[1])
    # single-path speculation: hit = max single accept-boundary probability
    hit1 = pmf_sorted[0][1]
    hit2 = hit1 + pmf_sorted[1][1]                # 2-path continuation tree
    hit3 = hit2 + pmf_sorted[2][1]                # 3-path

    # realized = speculation hits the accept boundary (hit) AND, on a hit, only the
    # bus-permitted fraction (overlap_eff) of the drafter actually hides.
    def realized_tps(hit):
        save = hit * overlap_eff * DRAFTER_MS_M8
        return 100.0 * (GPU_BUSY_MS_M8 / (GPU_BUSY_MS_M8 - save) - 1.0), save

    realized1_tps, save1 = realized_tps(hit1)
    realized2_tps, save2 = realized_tps(hit2)
    realized3_tps, save3 = realized_tps(hit3)

    # ================= verdict =================
    primary = bw_ceiling_wall_m8       # bandwidth_limited_overlap_ceiling_pct (wall)
    if r_m8 > args.r_close or primary < args.amber_pct:
        band = "RED"
    elif primary >= args.green_pct:
        band = "GREEN"
    else:
        band = "AMBER"
    # realizability band on the single-path realized TPS
    if realized1_tps >= args.green_pct:
        realize_band = "GREEN"
    elif realized1_tps >= args.amber_pct:
        realize_band = "AMBER"
    else:
        realize_band = "RED"

    config = {
        "HBM_GBS": HBM_GBS, "K": K, "E_T": E_T,
        "drafter_ms_m8": DRAFTER_MS_M8, "verify_ms_m8": VERIFY_MS_M8,
        "gpu_busy_ms_m8": GPU_BUSY_MS_M8,
        "local_walltps": LOCAL_WALLTPS, "official_tps": OFFICIAL_TPS,
        "sources": "frontier_decode(#69/#30) + #68 verify-gemm + #75 drafter + #85 tree + accept_calibration",
    }
    verdict = {
        # ---- PRIMARY + TEST metrics ----
        "bandwidth_limited_overlap_ceiling_pct": round(primary, 3),
        "drafter_verify_step_time_ratio": round(r_m8, 4),
        "drafter_verify_step_time_ratio_m32": round(r_m32, 4),
        "band": band,
        "realizability_band_single_path": realize_band,
        # ---- Step 1 ----
        "r_m8": round(r_m8, 4), "r_m32": round(r_m32, 4),
        "r_close_threshold": args.r_close,
        "naive_overlap_ceiling_wall_pct_m8": round(naive_wall_m8, 3),
        "naive_overlap_ceiling_tps_pct_m8": round(naive_tps_m8, 3),
        "naive_overlap_ceiling_wall_pct_m32": round(naive_wall_m32, 3),
        "naive_overlap_ceiling_tps_pct_m32": round(naive_tps_m32, 3),
        "drafter_ms_m32": round(drafter_ms_m32, 4),
        "verify_ms_m32": round(verify_ms_m32, 4),
        # ---- Step 2 (bandwidth) ----
        "verify_bytes_gb": round(verify_bytes / 1e9, 4),
        "verify_gemm_bytes_gb": round(v_gemm_bytes / 1e9, 4),
        "verify_avg_hbm_gbs": round(verify_avg_gbs / 1e9, 1),
        "verify_avg_hbm_pct": round(verify_avg_pct_hbm, 2),
        "verify_gemm_hbm_pct": round(v_gemm_pct_hbm_m8, 2),
        "drafter_bytes_gb": round(drafter_bytes / 1e9, 4),
        "drafter_avg_hbm_gbs": round(drafter_avg_gbs / 1e9, 1),
        "drafter_avg_hbm_pct": round(drafter_avg_pct_hbm, 2),
        "combined_bytes_gb": round(combined_bytes / 1e9, 4),
        "bandwidth_floor_ms": round(bw_floor_ms, 4),
        "bandwidth_floor_fits_in_verify": bool(fits_in_verify),
        "bandwidth_margin_ms": round(bw_margin_ms, 4),
        "drafter_fits_in_nongemm_slack_x": round(drafter_fits_nongemm_x, 2),
        "drafter_fits_in_gemm_slack_x": round(drafter_fits_gemm_x, 2),
        "gemm_plus_drafter_gbs": round(gemm_plus_drafter_gbs, 1),
        "gemm_coexists_under_peak": bool(gemm_coexists),
        "byte_fit_passes_necessary_not_sufficient": bool(byte_fit_ok),
        # ---- Step 2 MEASURED contention (A10G dual-stream probe) ----
        "contention_source": contention_source,
        "measured_drafter_overlap_efficiency": round(overlap_eff, 4),
        "measured_symmetric_overlap_speedup": round(sym_speedup, 4),
        "measured_bus_contention_factor": round(contention_factor, 4),
        "bandwidth_limited_overlap_ceiling_tps_pct": round(bw_ceiling_tps_m8, 3),
        # ---- Step 3a (serial dependency) ----
        "accept_boundary_pmf": [round(p, 4) for p in pmf],
        "spec_hit_rate_single_path": round(hit1, 4),
        "spec_hit_rate_2path_tree": round(hit2, 4),
        "spec_hit_rate_3path_tree": round(hit3, 4),
        "realized_tps_pct_single_path": round(realized1_tps, 3),
        "realized_tps_pct_2path_tree": round(realized2_tps, 3),
        "realized_tps_pct_3path_tree": round(realized3_tps, 3),
        # ---- TPS projections off the ceiling ----
        "ceiling_local_walltps": round(LOCAL_WALLTPS * (1 + bw_ceiling_tps_m8 / 100), 2),
        "ceiling_official_tps": round(OFFICIAL_TPS * (1 + bw_ceiling_tps_m8 / 100), 2),
        "realized_2path_official_tps": round(OFFICIAL_TPS * (1 + realized2_tps / 100), 2),
    }

    payload = {"config": config, "verdict": verdict,
               "verify_msweep": {m: vagg[m]["total_gemm_us"] for m in vagg},
               "frac": {"drafter": FRAC_DRAFTER, "verify_gemm": FRAC_VERIFY_GEMM,
                        "verify_attn": FRAC_VERIFY_ATTN, "verify_norm": FRAC_VERIFY_NORM,
                        "sampling": FRAC_SAMPLING, "verify_lmhead": FRAC_VERIFY_LMHEAD}}

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(json.dumps(verdict, indent=2))
    print(f"\n[overlap-gate] wrote {args.output}")

    if not args.no_wandb:
        try:
            _log_wandb(args, payload)
        except Exception as e:  # noqa: BLE001
            print(f"[overlap-gate] W&B logging skipped: {e}")


def _log_wandb(args, payload):
    import wandb
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type="analysis", config=payload["config"])
    v = payload["verdict"]
    # accept-boundary pmf as a table
    tbl = wandb.Table(columns=["accepted_draft_tokens", "probability"])
    for j, p in enumerate(v["accept_boundary_pmf"]):
        tbl.add_data(j, p)
    run.log({"accept_boundary_pmf": tbl})
    run.summary.update({k: val for k, val in v.items()
                        if not isinstance(val, list)})
    run.finish()
    print(f"[overlap-gate] W&B run: {run.url}", flush=True)


if __name__ == "__main__":
    main()
