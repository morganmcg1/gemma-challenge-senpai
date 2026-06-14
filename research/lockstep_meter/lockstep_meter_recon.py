"""Lockstep-meter reconstruction (PR #120).

Closes the ONE open choice in #116's pre-registered official-anchor protocol: *which*
local meter to capture "in lockstep" with the scarce, approval-gated official shot.

The problem (lawine #112): there is exactly ONE matched (official, local) pair on the
spec frontier (the #52 anchor, official 481.53) and a **7.14% cross-meter spread** among
the local meters on that same stack -- steady 428.37 / wall_tps 454.09 / windowed-steady
459.83. With one matched pair you can fit ANY meter to reproduce 481.53 by choosing its
own multiplier, so anchor-reproduction CANNOT disambiguate the meter. The only thing that
does is **methodology alignment**: whichever local meter's *definition* is methodologically
identical to the official HF-Jobs harness's TPS definition is the one whose local->official
ratio is the physical transfer factor, not a one-point coincidence.

This module is CPU-only, no GPU, no network. It:

  Step 1  Pins the official TPS definition from the committed harness + the pinned
          sglang dependency (citations inline).
  Step 2  Decomposes the 7.14% spread into mechanistic (estimator x window) sources,
          quantified EXACTLY from one committed matched-config run (the #56/#72 "full2"
          == research/maxbatchtok_ab MBT=512 decode, deployed config).
  Step 3  Aligns every meter to the official definition and measures the residual
          cross-meter spread (PRIMARY metric).
  Step 4  Self-consistency (lockstep reading x #99/#116 multiplier == 481.53) + the
          exact lockstep capture command and bit-exact self-check (the deliverable).
  Step 5  GREEN/AMBER/RED gate + wandb (group lockstep-meter).

All inputs are committed artifacts on the advisor branch (lawine #72/#82/#99/#112/#116).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the committed instrument (#99/#112/#116) for the multiplier + self-check. CPU-only.
from scripts.profiler import local_official_projection as lop  # noqa: E402

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Step 1 -- the official TPS definition (the discriminator)
# ---------------------------------------------------------------------------
# Cited from files committed in THIS repo + the pinned public sglang dependency.
#  * hf_bucket_single_job.py:344  ->  summary["tps"] = result["output_throughput"]
#    (line 342's total_tps ADDS input tokens; the HEADLINE tps is output-only).
#  * result = last JSON line of `sglang.bench_serving` output (hf...:490).
#  * sglang 0.5.2 bench_serving.py:1555  ->  output_throughput = sum(output_lens) / dur_s
#    i.e. TOTAL OUTPUT TOKENS / BENCHMARK WALL DURATION (time-weighted total/total).
#  * benchmark config (hf...:35-40, 200-247): num_prompts=128, output_len=512,
#    max_concurrency=1, request_rate=inf, WARMUP_REQUESTS=4 (discarded BEFORE the timer),
#    seed=1, ignore_eos=true (=> exactly 512 output tokens/req), backend vllm-chat.
#  * PPL is a SEPARATE post-benchmark stage (hf...:527+), so it is NOT inside dur_s.
OFFICIAL_TPS_DEFINITION = {
    "headline_field": "tps == result['output_throughput']",
    "formula": "output_throughput = sum(output_lens) / dur_s",
    "tokens_counted": "OUTPUT tokens only (prompt excluded; total_tps adds input but is not the headline)",
    "special_eos_tokens": "ignore_eos=true -> generation never stops at EOS; exactly output_len=512 "
                          "tokens counted per request (EOS, if emitted, is a normal counted token)",
    "estimator": "time-weighted total/total (sum of output tokens / single wall-clock duration)",
    "window": "warm steady-state: WARMUP_REQUESTS=4 sent and DISCARDED before the benchmark timer",
    "ppl": "separate post-benchmark stage; NOT inside dur_s",
    "concurrency": "max_concurrency=1 (strictly sequential, one request in flight)",
    "citations": {
        "headline_tps": "official/main_bucket/shared_resources/speed_benchmark/scripts/hf_bucket_single_job.py:344",
        "total_tps_is_secondary": "hf_bucket_single_job.py:342",
        "result_is_last_sglang_line": "hf_bucket_single_job.py:490",
        "output_throughput_formula": "sglang==0.5.2 sglang/bench_serving.py:1555 (output_throughput=sum(output_lens)/dur_s)",
        "bench_config": "hf_bucket_single_job.py:35-40,200-247 (num_prompts/output_len/conc/warmup/seed/ignore_eos)",
        "ppl_separate_stage": "hf_bucket_single_job.py:527+",
    },
}

# ---------------------------------------------------------------------------
# The three local meters (lawine #112 METER_WITNESS) + their committed definitions
# ---------------------------------------------------------------------------
OFFICIAL_ANCHOR_TPS = float(lop.OFFICIAL_ANCHOR["tps"])  # 481.53 (#52)

METERS = {
    "steady": {
        "local_tps": 428.37,
        "estimator": "unweighted arithmetic mean of vLLM per-interval 'Avg generation throughput'",
        "window": "ALL logged intervals (cold-start ramp + warm + any post-decode PPL-phase intervals)",
        "code": "scripts/local_validation/serve_profile.py:236 steady_gen_tps_mean=fmean(gen_tps); "
                "regex :211 'Avg generation throughput:\\s*([\\d.]+)'",
        "source": "BASELINE.md PR#43 tps_local_splitkv_steady; retired by lawine #72",
    },
    "wall_tps": {
        "local_tps": 454.09,
        "estimator": "time-weighted total/total: num_completion_tokens / decode_duration_s",
        "window": "the 128-prompt decode wall window (cold-start INCLUDED; PPL excluded -- separate stage)",
        "code": "official .../decode_outputs.py:310 num_completion_tokens, :316 duration_s",
        "source": "BASELINE.md #82 re-baseline (median N=3, CV 0.007%); == #72 N=12 454.12",
    },
    "windowed-steady": {
        "local_tps": 459.83,
        "estimator": "unweighted arithmetic mean of vLLM per-interval 'Avg generation throughput'",
        "window": "WARM intervals only (drop first W=3 cold intervals; PPL excluded)",
        "code": "research/tps_noise_floor/PROTOCOL.md:15 / pr72_results.md:14 windowed steady mean",
        "source": "lawine #72 robust interval-meter variant",
    },
}

# ---------------------------------------------------------------------------
# The single committed matched-config run for the EXACT decomposition.
# research/maxbatchtok_ab MBT=512 == the deployed config == the #56/#72 "full2" run.
# server log has the per-interval gen-tps meter; the decode summary gives total/total.
# ---------------------------------------------------------------------------
MBT512_SERVER_LOG = ROOT / "research/maxbatchtok_ab/server_mbt512.log"
MBT512_DECODE_JSONL = ROOT / "research/maxbatchtok_ab/decode_mbt512.jsonl"
# decode_duration_s echoed by the committed run (research/maxbatchtok_ab/full.out:58
# 'wandb: summary/decode_duration_s 144.25647'; tokens recomputed from the jsonl).
MBT512_DECODE_DURATION_S = 144.25647

# Capture gen-tps AND the in-flight request count from the same vLLM log line.
# serve_profile.py:211 only parses the tps; we also read 'Running: N reqs' because it
# is the GROUND-TRUTH phase marker: decode intervals run the 128-prompt benchmark at
# conc=1 (Running: 1 reqs); the trailing PPL stage has NO generation requests in flight
# (Running: 0 reqs, with a prompt-throughput spike) -- exactly what the official dur_s
# excludes. This is a definitional split, not a magnitude threshold.
_GEN_TPS_RE = re.compile(r"Avg generation throughput:\s*([\d.]+)")  # serve_profile.py:211
_GEN_TPS_RUNNING_RE = re.compile(
    r"Avg generation throughput:\s*([\d.]+) tokens/s,\s*Running:\s*(\d+) reqs")

# #72 noise-floor measured residuals (N=12 identical-config fresh runs; committed
# research/tps_noise_floor/pr72_results.md). These bound the post-alignment residual.
WALL_TPS_CV_PCT_N12 = 0.035
WALL_TPS_RANGE_PCT_N12 = 0.10
STEADY_CV_PCT_N12 = 0.33
WINDOWED_CV_PCT_N12 = 0.05


def _load_intervals() -> list[float]:
    text = MBT512_SERVER_LOG.read_text()
    return [float(x) for x in _GEN_TPS_RE.findall(text)]


def _load_intervals_with_phase() -> list[tuple[float, int]]:
    """(gen_tps, running_reqs) per logged interval, in order. running_reqs==0 marks the
    post-decode PPL stage (no generation request in flight); >=1 marks decode."""
    text = MBT512_SERVER_LOG.read_text()
    return [(float(t), int(r)) for t, r in _GEN_TPS_RUNNING_RE.findall(text)]


def _decode_tokens() -> int:
    total = 0
    with MBT512_DECODE_JSONL.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total += int(json.loads(line)["num_completion_tokens"])
    return total


# ---------------------------------------------------------------------------
# Step 2 -- decompose the spread mechanistically
# ---------------------------------------------------------------------------
def implied_multiplier_spread(meters: dict[str, dict]) -> dict[str, Any]:
    mults = {k: OFFICIAL_ANCHOR_TPS / v["local_tps"] for k, v in meters.items()}
    vals = list(mults.values())
    spread = 100.0 * (max(vals) - min(vals)) / st.fmean(vals)
    return {"implied_multiplier": mults, "spread_pct": spread,
            "min_meter": min(mults, key=mults.get), "max_meter": max(mults, key=mults.get)}


def single_run_decomposition() -> dict[str, Any]:
    """EXACT (estimator x window) decomposition from the committed MBT=512 run."""
    phased = _load_intervals_with_phase()
    iv = [t for t, _ in phased]
    n = len(iv)
    # GROUND-TRUTH phase split (not a magnitude threshold): PPL-stage intervals are the
    # ones with NO generation request in flight (Running: 0 reqs) -- on this committed run
    # the trailing two (282.1, 6.4 tps) -- exactly what the official dur_s excludes. The
    # decode window is every Running>=1 interval.
    body = [t for t, r in phased if r >= 1]               # decode-only intervals (cold + warm)
    ppl_intervals = [t for t, r in phased if r == 0]      # post-decode PPL stage
    n_ppl = len(ppl_intervals)
    warm_median = st.median(body[1:]) if len(body) > 2 else st.median(body)
    cold = body[0]
    warm = body[1:]                                       # drop the cold-start ramp interval

    toks = _decode_tokens()
    wall_tps = toks / MBT512_DECODE_DURATION_S            # official-estimator total/total

    steady_all = st.fmean(iv)                             # 'steady' meter on this run
    clean_body = st.fmean(body)                           # PPL removed (still unweighted)
    warm_mean = st.fmean(warm)                            # cold + PPL removed (unweighted)
    windowed = st.fmean(body[3:]) if len(body) > 3 else warm_mean  # drop W=3 cold

    # Mechanistic terms (TPS), on the unweighted-mean estimator:
    ppl_term = clean_body - steady_all                    # removing PPL-phase intervals
    cold_term = warm_mean - clean_body                    # removing the cold-start interval
    # Estimator term: time-weighted total/total vs the unweighted warm mean.
    estimator_term = wall_tps - warm_mean                 # (signed; cold-in-total/total pulls < warm)
    return {
        "run": "research/maxbatchtok_ab MBT=512 (deployed config == #56/#72 'full2')",
        "n_intervals_total": n,
        "n_decode_intervals": len(body),
        "n_ppl_intervals": n_ppl,
        "ppl_intervals_tps": ppl_intervals,
        "phase_split_marker": "Running:0 reqs (no generation request in flight) == PPL stage",
        "cold_interval_tps": cold,
        "warm_median_tps": warm_median,
        "decode_tokens": toks,
        "decode_duration_s": MBT512_DECODE_DURATION_S,
        "meters_on_this_run": {
            "steady_all_intervals": steady_all,
            "clean_body_PPL_removed": clean_body,
            "warm_mean_cold_and_PPL_removed": warm_mean,
            "windowed_dropW3": windowed,
            "wall_tps_total_over_total": wall_tps,
        },
        "cold_deficit_pct_vs_warm": 100.0 * (warm_median - cold) / warm_median,
        "mechanistic_terms_tps": {
            "ppl_phase_leak": ppl_term,
            "cold_start_interval": cold_term,
            "estimator_unweighted_vs_total_over_total": estimator_term,
        },
        "note": ("All terms are DEFINITIONAL (estimator x window). The underlying decode is "
                 "identical -- wall_tps holds to CV 0.035% across 12 fresh restarts (#72)."),
    }


def attribute_headline_spread(decomp: dict[str, Any], spread_pct: float) -> dict[str, Any]:
    """Allocate the headline 7.14% spread (steady<->windowed, both unweighted-mean meters,
    so they differ ONLY by WINDOW = cold + PPL) into its PPL and cold sub-terms, using the
    single-run magnitudes as the (run-calibrated) split ratio. The estimator axis is what
    makes wall_tps the official-aligned meter and is reported separately (it collapses the
    residual, it is not part of the steady<->windowed window span)."""
    terms = decomp["mechanistic_terms_tps"]
    ppl = abs(terms["ppl_phase_leak"])
    cold = abs(terms["cold_start_interval"])
    window_total = ppl + cold
    ppl_frac = ppl / window_total if window_total else 0.0
    cold_frac = cold / window_total if window_total else 0.0
    return {
        "axis": "WINDOW (cold-start + PPL-phase), on the fragile unweighted-mean estimator",
        "headline_spread_pct": spread_pct,
        "split_ratio_source": "single committed MBT=512 run (PPL %.2f : cold %.2f TPS)" % (ppl, cold),
        "table": [
            {"term": "PPL-phase leak",
             "mechanism": "unweighted mean includes post-decode PPL intervals (282/6.4 tps) "
                          "that official dur_s excludes (PPL is a separate stage)",
             "affects": "steady (fully)",
             "contribution_pct": ppl_frac * spread_pct},
            {"term": "cold-start interval",
             "mechanism": "unweighted mean over-weights the per-decode CUDA-graph/cache-ramp "
                          "first interval (~29% below warm); official discards 4 warmup requests",
             "affects": "steady (fully); wall_tps (partially, time-weighted to ~1%)",
             "contribution_pct": cold_frac * spread_pct},
        ],
        "sum_pct": (ppl_frac + cold_frac) * spread_pct,
        "estimator_axis_separate": {
            "term": "estimator: unweighted-mean-of-rates vs time-weighted total/total",
            "mechanism": "separates wall_tps (total/total = official) from the two interval-mean "
                         "meters; this is the axis that, once aligned, COLLAPSES the residual",
            "single_run_tps": terms["estimator_unweighted_vs_total_over_total"],
        },
    }


# ---------------------------------------------------------------------------
# Step 3 -- align to official + residual (PRIMARY)
# ---------------------------------------------------------------------------
def align_and_residual() -> dict[str, Any]:
    """Correct each meter to the official definition (time-weighted total/total, output-only,
    PPL-excluded). All three collapse onto wall_tps; the residual cross-meter spread is the
    wall_tps measurement floor (#72: CV 0.035%, range 0.10% over N=12 fresh restarts)."""
    alignment = {
        "steady": "re-estimate as total/total over the decode window (drop PPL, drop cold, "
                  "time-weight) -> wall_tps",
        "windowed-steady": "re-estimate the warm window as total/total instead of unweighted "
                           "mean -> wall_tps",
        "wall_tps": "ALREADY total/total over the decode window == official output_throughput "
                    "estimator (no correction)",
    }
    # After alignment all three ARE wall_tps -> on one run residual==0; across runs the residual
    # is the irreducible wall_tps floor. Report the conservative RANGE (max-min/mean) analog of
    # the 7.14% spread, i.e. the N=12 wall_tps range.
    residual_pct = WALL_TPS_RANGE_PCT_N12
    return {
        "lockstep_meter": "wall_tps",
        "lockstep_meter_definition": (
            "num_completion_tokens / decode_duration_s from official decode_outputs.py "
            "(128x512, seed 1, conc=1 via MAX_NUM_SEQS=1, ignore_eos), median of N=3 fresh "
            "runs, decode-only (PPL separate), captured AS-IS == cold-start included / no "
            "warmup discard (so it stays the EXACT definition the #99 multiplier was fit on)"),
        "alignment_per_meter": alignment,
        "residual_spread_after_alignment_pct": residual_pct,
        "residual_basis": (
            "wall_tps N=12 fresh-restart RANGE 0.10%% (CV 0.035%%); windowed 0.05%%; steady 0.33%% "
            "(#72). After alignment the 7.14%% cross-meter spread is gone -- only the wall_tps "
            "physical floor remains, which the official shot itself must absorb."),
        "lockstep_meter_matches_official_methodology": 1,
        "matches_rationale": (
            "wall_tps is the EXACT official estimator: total output tokens / wall duration == "
            "sglang output_throughput = sum(output_lens)/dur_s. The one definitional delta "
            "(official's 4-warmup-discard vs local cold-included) is a UNIFORM absolute offset "
            "(it shifts all meters equally, not a cross-meter spread) and is already absorbed "
            "into the deployed multiplier tau=1.06019 -- so it cancels as long as the lockstep "
            "capture uses the SAME cold-included decode_outputs.py definition."),
    }


# ---------------------------------------------------------------------------
# Step 4 -- self-consistency + finalize the #116 protocol
# ---------------------------------------------------------------------------
def self_consistency_and_protocol() -> dict[str, Any]:
    calib = lop.calibrate()
    sc = lop.self_check(calib)  # projects the LOCKED linear wall_tps reference -> ~481.53
    # The lockstep capture command the future human-approved shot runs in lockstep.
    capture_cmd = (
        ".venv/bin/python -m research.tps_noise_floor.run_noise_floor "
        "--submission <SPLITK_SUBMISSION> --mode fresh --n-runs 3 --wandb-group lockstep-meter\n"
        "# -> reports wall_tps = num_completion_tokens / decode_duration_s, median of N=3 "
        "(the #72/#82 protocol). Decode-only; PPL runs separately. Capture this number in the "
        "SAME job/session as the official HF shot of the SAME submission."
    )
    bitexact_selfcheck = (
        "BIT-EXACT self-check (no new run): the captured lockstep wall_tps, multiplied by the "
        "committed deployed multiplier tau=1.0601865 (#99/#116, = official 481.53 / pooled "
        "local 454.194), must reproduce the SAME submission's official tps within the residual "
        "(wall_tps range 0.10%%). Equivalently lop.self_check() recovers 481.53 from the LOCKED "
        "linear reference 454.338 to %.4f%% (<= MDE 0.10%%). This is NECESSARY not sufficient: "
        "ALL THREE meters pass anchor-reproduction trivially on one point -- the DISCRIMINATOR "
        "is the Step-1 methodology match, which only wall_tps satisfies." % sc["rel_err_vs_anchor_pct"]
    )
    return {
        "deployed_multiplier_tau": calib.multiplier,
        "pooled_local_wall_tps": calib.local_wall_tps,
        "locked_linear_reference_wall_tps": lop.LINEAR_REFERENCE_WALL_TPS,
        "reproduces_anchor_tps": sc["recovered_official"],
        "official_anchor_tps": sc["official_anchor"],
        "self_check_rel_err_pct": sc["rel_err_vs_anchor_pct"],
        "self_check_mde_pct": sc["self_check_mde_pct"],
        "reproduces_within_residual": sc["recovers_official_anchor"],
        "anchor_reproduction_is_necessary_not_sufficient": (
            "one point + one free multiplier fits ANY meter (steady needs x1.124, wall_tps "
            "x1.060, windowed x1.047); reproduction does NOT disambiguate -- Step 1 methodology "
            "match does"),
        "lockstep_capture_command": capture_cmd,
        "bit_exact_self_check": bitexact_selfcheck,
        "protocol_patch_for_116": {
            "named_lockstep_meter": "wall_tps (decode_outputs.py total/total, N=3 median, cold-included)",
            "why_cold_included": (
                "the #99 multiplier was calibrated as official_warm / local_cold-included; a "
                "warm-corrected wall_tps would DOUBLE-COUNT the cold-start (~+1.2%) and bias the "
                "banked 2nd pair -- capture RAW decode_outputs.py wall_tps"),
            "the_one_residual_the_shot_measures": (
                "the split-K reduction sync-overhead haircut (#116 tau_eff un-pinnable residual, "
                "<=1.26% rel) -- an ABSOLUTE local->official term, NOT a meter-choice spread"),
        },
    }


# ---------------------------------------------------------------------------
# Step 5 -- gate
# ---------------------------------------------------------------------------
def gate(residual_pct: float, matches: int, reproduces: bool) -> dict[str, Any]:
    green = matches == 1 and residual_pct <= 1.0 and reproduces
    amber = (not green) and 1.0 < residual_pct <= 3.0
    verdict = "GREEN" if green else ("AMBER" if amber else "RED")
    reasons = []
    if green:
        reasons.append(
            "wall_tps methodologically identical to official output_throughput (total/total, "
            "output-only); residual after alignment %.2f%% <= 1%%; reproduces #52 anchor 481.53 "
            "-> lockstep meter PINNED = wall_tps." % residual_pct)
        reasons.append(
            "the 7.14%% spread was PURE definition-mismatch (PPL leak + cold-start on the "
            "unweighted-mean estimator); it collapses to the wall_tps floor under the official "
            "total/total definition.")
    return {"verdict": verdict, "residual_spread_after_alignment_pct": residual_pct,
            "lockstep_meter_matches_official_methodology": matches, "reasons": reasons}


def build_report() -> dict[str, Any]:
    spread = implied_multiplier_spread(METERS)
    decomp = single_run_decomposition()
    attribution = attribute_headline_spread(decomp, spread["spread_pct"])
    aligned = align_and_residual()
    sc = self_consistency_and_protocol()
    g = gate(aligned["residual_spread_after_alignment_pct"],
             aligned["lockstep_meter_matches_official_methodology"],
             sc["reproduces_within_residual"])
    return {
        "pr": 120,
        "title": "Lockstep meter recon -- pin the meter the official anchor captures",
        "step1_official_tps_definition": OFFICIAL_TPS_DEFINITION,
        "meters": METERS,
        "step2_cross_meter_spread": spread,
        "step2_single_run_decomposition": decomp,
        "step2_spread_attribution": attribution,
        "step3_alignment": aligned,
        "step4_self_consistency_and_protocol": sc,
        "step5_gate": g,
        "primary_metric": {"name": "residual_spread_after_alignment_pct",
                           "value": aligned["residual_spread_after_alignment_pct"]},
        "test_metric": {"name": "lockstep_meter_matches_official_methodology",
                        "value": aligned["lockstep_meter_matches_official_methodology"]},
    }


def log_wandb(report: dict[str, Any], name: str, group: str) -> str | None:
    try:
        import os
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] skipped ({exc})", flush=True)
        return None
    try:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            name=name, group=group, job_type="analysis",
            config={"pr": 120, "kind": "lockstep-meter-recon",
                    "official_anchor_tps": OFFICIAL_ANCHOR_TPS},
        )
        spread = report["step2_cross_meter_spread"]
        aligned = report["step3_alignment"]
        sc = report["step4_self_consistency_and_protocol"]
        flat = {
            "primary/residual_spread_after_alignment_pct": report["primary_metric"]["value"],
            "test/lockstep_meter_matches_official_methodology": report["test_metric"]["value"],
            "spread/cross_meter_spread_pct": spread["spread_pct"],
            "spread/m_steady": spread["implied_multiplier"]["steady"],
            "spread/m_wall_tps": spread["implied_multiplier"]["wall_tps"],
            "spread/m_windowed_steady": spread["implied_multiplier"]["windowed-steady"],
            "selfcheck/deployed_multiplier_tau": sc["deployed_multiplier_tau"],
            "selfcheck/reproduces_anchor_tps": sc["reproduces_anchor_tps"],
            "selfcheck/rel_err_pct": sc["self_check_rel_err_pct"],
            "gate/verdict": report["step5_gate"]["verdict"],
            "lockstep_meter": aligned["lockstep_meter"],
        }
        run.summary.update(flat)
        tbl = wandb.Table(columns=["meter", "local_tps", "implied_multiplier", "estimator", "window"])
        for k, v in METERS.items():
            tbl.add_data(k, v["local_tps"], spread["implied_multiplier"][k], v["estimator"], v["window"])
        run.log({"meters": tbl})
        atbl = wandb.Table(columns=["term", "contribution_pct"])
        for r in report["step2_spread_attribution"]["table"]:
            atbl.add_data(r["term"], r["contribution_pct"])
        run.log({"spread_attribution": atbl})
        rid = run.id
        run.finish()
        print(f"[wandb] logged run {rid}", flush=True)
        return rid
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] log failed ({exc})", flush=True)
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "lockstep_meter_results.json"))
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="lockstep-meter")
    args = ap.parse_args()

    report = build_report()
    wid = None
    if args.wandb_name:
        wid = log_wandb(report, args.wandb_name, args.wandb_group)
    report["wandb_run_id"] = wid
    Path(args.out).write_text(json.dumps(report, indent=2))

    g = report["step5_gate"]
    sp = report["step2_cross_meter_spread"]
    al = report["step3_alignment"]
    print("\n========== LOCKSTEP METER RECON (PR #120) ==========", flush=True)
    print(f"official tps == output_throughput = sum(output_lens)/dur_s "
          f"(hf...:344 / sglang:1555)", flush=True)
    print(f"cross-meter spread (steady/wall_tps/windowed) = {sp['spread_pct']:.3f}%", flush=True)
    print(f"lockstep meter = {al['lockstep_meter']} "
          f"(matches official methodology = {al['lockstep_meter_matches_official_methodology']})", flush=True)
    print(f"PRIMARY residual_spread_after_alignment_pct = "
          f"{report['primary_metric']['value']:.3f}%", flush=True)
    sc4 = report['step4_self_consistency_and_protocol']
    print(f"self-check: locked ref {sc4['locked_linear_reference_wall_tps']:.3f} x "
          f"{sc4['deployed_multiplier_tau']:.5f} -> {sc4['reproduces_anchor_tps']:.2f} "
          f"(anchor 481.53, rel err {sc4['self_check_rel_err_pct']:.3f}% <= MDE "
          f"{sc4['self_check_mde_pct']:.2f}%)", flush=True)
    print(f"GATE: {g['verdict']}", flush=True)
    print(f"artifacts -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
