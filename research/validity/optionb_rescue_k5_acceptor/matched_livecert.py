#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #669 (stark) -- MATCHED-BASIS live-trajectory identity cert for the K=5
recompute-acceptor, under the STRICT-GLOBAL #319 lens (byte-exact greedy-token
identity on the SERVED OUTPUT incl. prefill; no subsystem carve-out).

The advisor ruled: strict-global governs, and the way to PASS it is to PROVE the
prefill flips are an artifact on a MATCHED reference -- not to redefine the gate.
This driver runs the single decisive arm with the proof layer:

  Phase `identity` (one invocation; matched container/build/seed/GPU):
    ar_a    SENPAI_REFERENCE_MODE=1 -> matched M=1-AR reference R_A (the cert ref)
    ar_b    SENPAI_REFERENCE_MODE=1 -> M=1-AR control R_B (the AR-vs-AR floor)
    spec    SENPAI_LIVECERT_REF_JSONL=R_A -> K=5 spec serve; the patch records the
            per-position verify-gap + de-teacher-force flip along the LIVE stream.
  Offline cascade (keyed on prompt_token_sha256 -> completion_token_ids):
    cascade(R_A, R_B)   = the within-stack nondeterminism FLOOR (pos0 + seq fork)
    cascade(R_A, spec)  = the spec divergence, GROUND TRUTH (walk-artifact-free)
  The prefill flips "collapse" iff cascade(R_A,spec).pos0 <= the AR-vs-AR floor;
  genuine divergences "survive" iff the live draft break (gap>=tau) > 0.

  Phase `speed` (one invocation; takes --inject-rate = the measured pre-fork flag
  rate from Phase identity):
    a real correcting rescue stays ON R_A, so it fires at the PRE-FORK rate; the
    as-built tau gate does NOT correct -> the stream diverges into lower-margin
    territory and over-fires at the GLOBAL rate. We MEASURE both:
      r=0                    -> tps0 (un-rescued K=5 ceiling)
      r=inject_rate          -> correcting-rescue speed (fires at the pre-fork rate)
      SENPAI_ACCEPTOR_TAU=t  -> the literal live tau gate (dummy fire, over-fires)
    official_equiv = (eff_tps/ar_rung_local) * 126.378; +10 bar = 136.378.

The MEASUREMENT is the validated PR #72/#82 protocol REUSED verbatim (run_arm ->
timed_decode -> median wall_tps over N fresh serves). analysis_only, official_tps=0,
no_hf_job=1, NO HF Job / submission / served-file change.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402
from scripts.profiler.paired_tps_ab import ArmSpec  # noqa: E402
from research.validity.optionb_rescue_k5_acceptor.deproject_runner import (  # noqa: E402
    LOCKED_319_AR_TPS,
    STARK_636_UNRESCUED_CEILING,
    build_args,
    fit_additive_cost,
    parse_env,
    read_acceptor_stats,
    read_livecert_summary,
    _run_one,
)

# Rule-of-three style slack: a spec pos0 count is "within the floor" iff it does not
# exceed (floor + 3) -- i.e. the excess over the AR-vs-AR floor is consistent with a
# zero-excess Poisson process at ~95% (the floor itself is a noisy 1-sample estimate).
POS0_FLOOR_SLACK = 3


# ---------------------------------------------------------------------------
# Offline served-stream cascade (keyed on prompt_token_sha256)
# ---------------------------------------------------------------------------
def load_streams(jsonl_path: Path) -> dict[str, list[int]]:
    """sha256(prompt) -> completion_token_ids from a served decode jsonl. Keyed on the
    stored ``prompt_token_sha256`` -- the same key the harness greedy-identity check and
    the live recorder's ``_hash_tokens`` use -- so the three streams align by prompt."""
    out: dict[str, list[int]] = {}
    if not jsonl_path.exists():
        return out
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            key = r.get("prompt_token_sha256")
            comp = r.get("completion_token_ids") or []
            if not key or not comp:
                continue
            out[key] = [int(t) for t in comp]
    return out


def cascade(ref: dict[str, list[int]], cand: dict[str, list[int]]) -> dict[str, Any]:
    """Compare two served streams prompt-by-prompt (matched on sha256 key). pos0 =
    prefill-bonus divergence (the advisor's "prefill flips"); seq = whether the stream
    EVER forks; first_divergence = where. The per-token break rate is reported but the
    *decisive* metrics are pos0 and seq (a single fork cascades every later token, so a
    raw per-token rate conflates one fork with its tail)."""
    keys = [k for k in ref if k in cand]
    n = len(keys)
    pos0 = seq = tok_break = tok_pos = 0
    first_div: list[int] = []
    examples: list[dict[str, Any]] = []
    for k in keys:
        a, b = ref[k], cand[k]
        m = min(len(a), len(b))
        if m == 0:
            continue
        if a[0] != b[0]:
            pos0 += 1
        fd = None
        for i in range(m):
            tok_pos += 1
            if a[i] != b[i]:
                tok_break += 1
                if fd is None:
                    fd = i
        if fd is not None:
            seq += 1
            first_div.append(fd)
            if len(examples) < 25:
                examples.append({"key": k, "first_divergence": fd,
                                 "ref_tok": a[fd], "cand_tok": b[fd],
                                 "is_pos0": fd == 0})
    return {
        "n_matched_prompts": n,
        "pos0_disagree": pos0,
        "pos0_disagree_rate": (pos0 / n) if n else None,
        "seq_diverge": seq,
        "seq_diverge_rate": (seq / n) if n else None,
        "per_token_break": tok_break,
        "per_token_positions": tok_pos,
        "per_token_break_rate": (tok_break / tok_pos) if tok_pos else None,
        "first_divergence_median": (statistics.median(first_div) if first_div else None),
        "first_divergence_min": (min(first_div) if first_div else None),
        "examples": examples,
    }


# ---------------------------------------------------------------------------
# Phase identity: ar_a, ar_b, spec + matched-basis cascade + strict verdict
# ---------------------------------------------------------------------------
def run_identity(a, extra: dict[str, str], out_dir: Path, server_python: Path,
                 rargs) -> dict[str, Any]:
    sub = a.submission
    per_arm: dict[str, dict[str, Any]] = {}
    records_path = out_dir / "records.jsonl"
    with open(records_path, "w") as fh:
        # ---- ar_a: matched M=1-AR reference R_A (the cert ref) ----
        ar_a = ArmSpec("ar_a", sub, {"SENPAI_REFERENCE_MODE": "1", **extra})
        per_arm["ar_a"] = _run_one(ar_a, 1, rargs, server_python, out_dir, fh)
        ref_a = out_dir / "ar_a" / "decode" / "run00.jsonl"
        # ---- ar_b: M=1-AR control R_B (the AR-vs-AR nondeterminism floor) ----
        ar_b = ArmSpec("ar_b", sub, {"SENPAI_REFERENCE_MODE": "1", **extra})
        per_arm["ar_b"] = _run_one(ar_b, 1, rargs, server_python, out_dir, fh)
        ref_b = out_dir / "ar_b" / "decode" / "run00.jsonl"
        if not ref_a.exists() or not ref_b.exists():
            raise SystemExit(f"[identity] AR decode jsonl missing: {ref_a} / {ref_b}")
        print(f"[identity] R_A={ref_a} (ar wall_tps={per_arm['ar_a']['wall_tps_median']}) "
              f"R_B={ref_b} (ar wall_tps={per_arm['ar_b']['wall_tps_median']})", flush=True)

        # ---- spec: live cert arm reading R_A ----
        stat_dir = (out_dir / "livecert_stat").resolve()
        lc_env = {
            **extra,
            "SENPAI_LIVECERT_REF_JSONL": str(ref_a),
            "SENPAI_LIVECERT_STAT_DIR": str(stat_dir),
            "SENPAI_LIVECERT_N_PROMPTS": str(a.num_prompts),
            "SENPAI_RECOMPUTE_LOG_EVERY": "512",
        }
        spec = ArmSpec("spec", sub, lc_env)
        per_arm["spec"] = _run_one(spec, 1, rargs, server_python, out_dir, fh)
        spec_jsonl = out_dir / "spec" / "decode" / "run00.jsonl"
        cert = read_livecert_summary(stat_dir)
        if not cert:
            raise SystemExit(f"[identity] no cert summary written to {stat_dir}")

    # ---- offline cascade (ground truth, walk-artifact-free) ----
    s_a = load_streams(ref_a)
    s_b = load_streams(ref_b)
    s_spec = load_streams(spec_jsonl)
    casc_floor = cascade(s_a, s_b)     # AR-vs-AR within-stack nondeterminism floor
    casc_spec = cascade(s_a, s_spec)   # spec divergence (ground truth)
    print(f"[identity] CASCADE floor(R_A vs R_B): pos0={casc_floor['pos0_disagree']} "
          f"seq={casc_floor['seq_diverge']}/{casc_floor['n_matched_prompts']} "
          f"per_tok_break_rate={casc_floor['per_token_break_rate']}", flush=True)
    print(f"[identity] CASCADE spec(R_A vs spec): pos0={casc_spec['pos0_disagree']} "
          f"seq={casc_spec['seq_diverge']}/{casc_spec['n_matched_prompts']} "
          f"per_tok_break_rate={casc_spec['per_token_break_rate']} "
          f"first_div_med={casc_spec['first_divergence_median']}", flush=True)

    verdict = strict_identity_verdict(casc_floor, casc_spec, cert, a.tau)
    inject_rate = verdict.get("prefork_flag_rate_per_emit_at_tau")
    print(f"[identity] STRICT verdict={verdict['verdict']} "
          f"prefill_collapsed={verdict['prefill_collapsed']} "
          f"draft_clean={verdict['draft_clean']} "
          f"inject_rate(per_emit@tau)={inject_rate}", flush=True)

    result = {
        "pr": 669, "leg": "matched_identity", "phase": "identity",
        "analysis_only": True, "official_tps": 0, "no_hf_job": True,
        "submission": sub, "extra_env": extra, "tau": a.tau,
        "workload": {"num_prompts": a.num_prompts, "output_len": a.output_len,
                     "seed": a.seed},
        "ref_a_jsonl": str(ref_a), "ref_b_jsonl": str(ref_b),
        "spec_jsonl": str(spec_jsonl),
        "cascade_floor_arar": casc_floor,
        "cascade_spec": casc_spec,
        "cert": cert,
        "verdict": verdict,
        "arms": per_arm,
        "anchors": {"locked_ar_tps": LOCKED_319_AR_TPS,
                    "ar_rung_local": a.ar_rung_local,
                    "plus10_bar_official": LOCKED_319_AR_TPS + 10.0},
    }
    (out_dir / "matched_identity.json").write_text(
        json.dumps(result, indent=2, default=str))
    return result


def strict_identity_verdict(casc_floor: dict, casc_spec: dict, cert: dict,
                            tau: float) -> dict[str, Any]:
    """STRICT-GLOBAL #319 matched-basis verdict (incl. prefill, no carve-out).

    The served strict break decomposes into:
      * PREFILL component -- pos0 (output-pos-0) flips, structurally un-flaggable (no
        draft row). These are an ARTIFACT iff cascade(R_A,spec).pos0 does not exceed the
        AR-vs-AR floor cascade(R_A,R_B).pos0 (within rule-of-three slack): then they are
        the SAME within-stack run-to-run nondeterminism #319 tolerates, not spec-induced.
      * DRAFT component -- a spec draft-row flip whose verify gap >= tau (the acceptor
        could NOT recompute it). A real correcting rescue cannot catch these -> if > 0,
        a genuine served divergence SURVIVES even with rescue.

    identity_holds(matched, strict) = prefill_collapsed AND draft_clean.
    Also surfaces the measured PRE-FORK flag rate at tau (per emit) -> the rate a
    correcting rescue fires at -> the speed phase's inject-rate."""
    per_tau = (cert.get("per_tau") or {})
    tkey = f"{tau:g}"
    st = per_tau.get(tkey, {})
    pos0_floor = casc_floor.get("pos0_disagree", 0) or 0
    pos0_spec = casc_spec.get("pos0_disagree", 0) or 0
    seq_floor = casc_floor.get("seq_diverge", 0) or 0
    seq_spec = casc_spec.get("seq_diverge", 0) or 0
    draft_break = st.get("prefork_break_count_draftonly")
    if draft_break is None:
        draft_break = cert.get("n_draft_flips")  # conservative fallback

    prefill_collapsed = pos0_spec <= (pos0_floor + POS0_FLOOR_SLACK)
    seq_within_floor = seq_spec <= (seq_floor + POS0_FLOOR_SLACK)
    draft_clean = (draft_break == 0)
    identity_holds = bool(prefill_collapsed and draft_clean)

    if identity_holds:
        v = "LIVE_FLAG_REDUCIBLE_HOLDS"   # strict, matched -> pending speed phase
    else:
        v = "LIVE_FLAG_IRREDUCIBLE"

    return {
        "verdict": v,
        "lens": "strict_global_matched",
        "identity_holds_matched": identity_holds,
        "prefill_collapsed": bool(prefill_collapsed),
        "seq_within_floor": bool(seq_within_floor),
        "draft_clean": bool(draft_clean),
        "tau": tau,
        "pos0_floor_arar": pos0_floor,
        "pos0_spec": pos0_spec,
        "pos0_excess_over_floor": pos0_spec - pos0_floor,
        "seq_floor_arar": seq_floor,
        "seq_spec": seq_spec,
        "seq_excess_over_floor": seq_spec - seq_floor,
        "draft_break_count_at_tau": draft_break,
        "n_prefill_flips_cert": cert.get("n_prefill_flips"),
        "n_draft_flips_cert": cert.get("n_draft_flips"),
        "prefork_break_count_strict_at_tau": st.get("prefork_break_count"),
        "prefork_break_count_draftonly_at_tau": st.get("prefork_break_count_draftonly"),
        "prefork_flag_rate_per_emit_at_tau": st.get("prefork_flag_rate_per_emit"),
        "prefork_flag_rate_per_draft_at_tau": st.get("prefork_flag_rate_per_draft"),
        "global_flag_rate_per_draft_at_tau": st.get("global_flag_rate_per_draft"),
        "rule_of_three_ub_over_draft": cert.get("rule_of_three_ub_over_draft"),
        "pos0_floor_slack": POS0_FLOOR_SLACK,
    }


# ---------------------------------------------------------------------------
# Phase speed: rate sweep (correcting-rescue) + dummy tau gate (over-fire)
# ---------------------------------------------------------------------------
def run_speed(a, extra: dict[str, str], out_dir: Path, server_python: Path,
              rargs) -> dict[str, Any]:
    sub = a.submission
    inject = a.inject_rate
    if inject is None or inject <= 0:
        raise SystemExit("[speed] --inject-rate (the measured pre-fork per-emit flag "
                         "rate from the identity phase) is required and must be > 0")
    per_arm: dict[str, dict[str, Any]] = {}
    rate_to_tps: dict[float, float] = {}
    records_path = out_dir / "records.jsonl"
    with open(records_path, "w") as fh:
        # ---- rate sweep: 0 (tps0), inject (correcting rescue), 2*inject (slope) ----
        rates = [0.0, round(inject, 6), round(2 * inject, 6)]
        for r in rates:
            lbl = f"r{r:g}".replace(".", "p").replace("-", "m")
            env = {**extra}
            if r > 0:
                env["SENPAI_RECOMPUTE_RATE"] = repr(r)
                env["SENPAI_RECOMPUTE_STAT_DIR"] = str((out_dir / f"stat_{lbl}").resolve())
            arm = ArmSpec(lbl, sub, env)
            per_arm[arm.label] = _run_one(arm, a.n, rargs, server_python, out_dir, fh)
            rate_to_tps[r] = per_arm[arm.label]["wall_tps_median"]
            print(f"[speed] rate r={r}: wall_tps={per_arm[arm.label]['wall_tps_median']}",
                  flush=True)

        # ---- the literal live tau gate (dummy fire -> over-fires at global rate) ----
        tau_lbl = f"tau{a.tau:g}".replace(".", "p")
        tau_env = {**extra, "SENPAI_ACCEPTOR_TAU": repr(a.tau),
                   "SENPAI_RECOMPUTE_STAT_DIR": str((out_dir / f"stat_{tau_lbl}").resolve())}
        tau_arm = ArmSpec(tau_lbl, sub, tau_env)
        per_arm[tau_arm.label] = _run_one(tau_arm, a.n_tau, rargs, server_python, out_dir, fh)
        tau_acc = read_acceptor_stats(Path(tau_env["SENPAI_RECOMPUTE_STAT_DIR"]))
        per_arm[tau_arm.label]["acceptor_stats"] = tau_acc
        print(f"[speed] tau={a.tau} (dummy gate): wall_tps="
              f"{per_arm[tau_arm.label]['wall_tps_median']} "
              f"realized_fire_rate={tau_acc.get('realized_fire_rate')} "
              f"realized_flag_rate={tau_acc.get('realized_flag_rate')}", flush=True)

    fit = fit_additive_cost(rate_to_tps)
    tps0 = rate_to_tps.get(0.0)
    eff_rate = round(inject, 6)
    eff_tps = rate_to_tps.get(eff_rate)
    ratio_ar = (eff_tps / a.ar_rung_local) if eff_tps else None
    official_equiv = (ratio_ar * LOCKED_319_AR_TPS) if ratio_ar else None
    plus10 = LOCKED_319_AR_TPS + 10.0
    speed = {
        "inject_rate_per_emit": inject,
        "rate_to_wall_tps": {str(k): v for k, v in sorted(rate_to_tps.items())},
        "additive_cost_fit": fit,
        "tps0_unrescued_ceiling_local": tps0,
        "correcting_rescue_tps_local": eff_tps,
        "dummy_tau_tps_local": per_arm[tau_lbl]["wall_tps_median"],
        "dummy_tau_realized_fire_rate": tau_acc.get("realized_fire_rate"),
        "dummy_tau_realized_flag_rate": tau_acc.get("realized_flag_rate"),
        "ar_rung_local": a.ar_rung_local,
        "ar_rung_official": LOCKED_319_AR_TPS,
        "acceptor_over_ar_ratio": ratio_ar,
        "official_equiv_tps": official_equiv,
        "clears_rung_by_tps": (official_equiv - LOCKED_319_AR_TPS) if official_equiv else None,
        "clears_plus10_bar": (official_equiv >= plus10) if official_equiv else None,
        "correcting_over_unrescued": (eff_tps / tps0) if (eff_tps and tps0) else None,
        "correcting_over_dummy_tau": (
            (eff_tps / per_arm[tau_lbl]["wall_tps_median"])
            if (eff_tps and per_arm[tau_lbl]["wall_tps_median"]) else None),
    }
    print(f"[speed] SPEED: tps0={tps0} correcting_rescue@{eff_rate}={eff_tps} "
          f"dummy_tau={speed['dummy_tau_tps_local']} official_equiv={official_equiv} "
          f"clears_+10={speed['clears_plus10_bar']}", flush=True)

    result = {
        "pr": 669, "leg": "matched_speed", "phase": "speed",
        "analysis_only": True, "official_tps": 0, "no_hf_job": True,
        "submission": sub, "extra_env": extra, "tau": a.tau, "n": a.n,
        "workload": {"num_prompts": a.num_prompts, "output_len": a.output_len,
                     "seed": a.seed},
        "speed": speed,
        "arms": per_arm,
        "anchors": {"locked_ar_tps": LOCKED_319_AR_TPS,
                    "ar_rung_local": a.ar_rung_local,
                    "unrescued_636_ceiling": STARK_636_UNRESCUED_CEILING,
                    "plus10_bar_official": plus10},
    }
    (out_dir / "matched_speed.json").write_text(json.dumps(result, indent=2, default=str))
    return result


# ---------------------------------------------------------------------------
# wandb
# ---------------------------------------------------------------------------
def _log_wandb(a, result: dict[str, Any]) -> None:
    if a.no_wandb:
        return
    try:
        from scripts.wandb_logging import init_wandb_run, finish_wandb
    except Exception as exc:
        print(f"[wandb] import failed: {exc!r}; JSON only", flush=True)
        return
    phase = result["phase"]
    extra_env = result["extra_env"]
    run = init_wandb_run(
        job_type="local_profiling", agent="stark",
        name=a.wandb_name or f"stark/k5-matched-{phase}", group=a.wandb_group,
        notes="PR#669 K=5 MATCHED-BASIS strict-global live identity cert "
              "(ar_a/ar_b/spec + offline cascade) / correcting-rescue speed. analysis_only.",
        config={"pr": 669, "harness_lineage_pr": 642, "mode": f"matched_{phase}",
                "submission": a.submission, "extra_env": extra_env, "n": a.n,
                "tau": a.tau, "num_prompts": a.num_prompts, "output_len": a.output_len,
                "num_speculative_tokens": extra_env.get("NUM_SPECULATIVE_TOKENS"),
                "analysis_only": True, "official_tps": 0},
    )
    if run is None:
        print("[wandb] disabled; JSON only", flush=True)
        return
    summary: dict[str, Any] = {"analysis_only": 1, "official_tps": 0, "no_hf_job": 1}
    if phase == "identity":
        cf = result["cascade_floor_arar"]
        cs = result["cascade_spec"]
        cert = result.get("cert") or {}
        v = result["verdict"]
        for tag, c in (("floor_arar", cf), ("spec", cs)):
            for key in ("n_matched_prompts", "pos0_disagree", "pos0_disagree_rate",
                        "seq_diverge", "seq_diverge_rate", "per_token_break_rate",
                        "first_divergence_median"):
                val = c.get(key)
                if isinstance(val, (int, float)):
                    summary[f"cascade/{tag}/{key}"] = val
        for key in ("identity_holds_matched", "prefill_collapsed", "seq_within_floor",
                    "draft_clean"):
            summary[f"verdict/{key}"] = int(bool(v.get(key)))
        for key in ("pos0_floor_arar", "pos0_spec", "pos0_excess_over_floor",
                    "seq_floor_arar", "seq_spec", "seq_excess_over_floor",
                    "draft_break_count_at_tau", "prefork_break_count_strict_at_tau",
                    "prefork_break_count_draftonly_at_tau",
                    "prefork_flag_rate_per_emit_at_tau",
                    "prefork_flag_rate_per_draft_at_tau",
                    "global_flag_rate_per_draft_at_tau", "rule_of_three_ub_over_draft"):
            val = v.get(key)
            if isinstance(val, (int, float)):
                summary[f"verdict/{key}"] = val
        summary["cert/n_prefill_flips"] = cert.get("n_prefill_flips")
        summary["cert/n_draft_flips"] = cert.get("n_draft_flips")
        summary["cert/prefork_draft_positions"] = cert.get("prefork_draft_positions")
        summary["cert/n_matched"] = cert.get("n_matched")
        summary["cert/n_reqs_seen"] = cert.get("n_reqs_seen")
        summary["verdict"] = v.get("verdict")
    else:  # speed
        sp = result["speed"]
        for key in ("inject_rate_per_emit", "tps0_unrescued_ceiling_local",
                    "correcting_rescue_tps_local", "dummy_tau_tps_local",
                    "dummy_tau_realized_fire_rate", "dummy_tau_realized_flag_rate",
                    "acceptor_over_ar_ratio", "official_equiv_tps", "clears_rung_by_tps",
                    "correcting_over_unrescued", "correcting_over_dummy_tau",
                    "ar_rung_local"):
            val = sp.get(key)
            if isinstance(val, (int, float)):
                summary[f"speed/{key}"] = val
        if sp.get("clears_plus10_bar") is not None:
            summary["speed/clears_plus10_bar"] = int(bool(sp["clears_plus10_bar"]))
        fit = sp.get("additive_cost_fit")
        if fit:
            summary["speed/fit_C_sec_per_recompute"] = fit.get("C_sec_per_recompute")
            summary["speed/fit_tps0"] = fit.get("tps0_fit")
            summary["speed/fit_r2"] = fit.get("r2")
    for lbl, info in result["arms"].items():
        if isinstance(info.get("wall_tps_median"), (int, float)):
            summary[f"arm/{lbl}/wall_tps"] = info["wall_tps_median"]
        if isinstance(info.get("e_accept_exact_mean"), (int, float)):
            summary[f"arm/{lbl}/e_accept"] = info["e_accept_exact_mean"]
    for k, val in summary.items():
        if val is not None:
            run.summary[k] = val
    finish_wandb(run)
    print(f"[wandb] logged matched-{phase} run {run.id}", flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["identity", "speed"], required=True)
    ap.add_argument("--submission", default="int4_mtp_batchinv")
    ap.add_argument("--tau", type=float, default=0.27,
                    help="the gap-flag threshold (advisor's single arm: 0.27)")
    ap.add_argument("--inject-rate", type=float, default=None,
                    help="speed phase: the measured pre-fork per-emit flag rate at --tau "
                         "from the identity phase (the rate a correcting rescue fires at)")
    ap.add_argument("--extra-env", action="append", default=[],
                    help="KEY=VALUE serve-time env (repeatable). Defaults set K=5 + cudagraph.")
    ap.add_argument("--n", type=int, default=2, help="fresh serves per speed rate arm")
    ap.add_argument("--n-tau", type=int, default=1, help="fresh serves for the dummy tau arm")
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--seed", type=int, default=paths.SEED)
    ap.add_argument("--clock-interval-ms", type=int, default=250)
    ap.add_argument("--settle-s", type=float, default=2.5)
    ap.add_argument("--ar-rung-local", type=float, default=126.75,
                    help="local AR-rung anchor for the official-equiv ratio (#319 a10g-small)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="optionb-livecert-k5-stark")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args(argv)

    for note in paths.prepare_local_gpu_env():
        print(f"[matched] {note}", flush=True)

    # Default serve env: K=5 spec + recompute cudagraph (matches #663/full_k5 harness).
    extra = {"NUM_SPECULATIVE_TOKENS": "5", "SENPAI_RECOMPUTE_CUDAGRAPH": "1"}
    extra.update(parse_env(a.extra_env))

    out_dir = a.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_dir = (ROOT / "submissions" / a.submission).resolve()
    if not sub_dir.exists():
        raise SystemExit(f"submission not found: {sub_dir}")
    manifest = harness.load_manifest(sub_dir)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    rargs = build_args(a)

    print(f"[matched] phase={a.phase} submission={a.submission} extra_env={extra} "
          f"tau={a.tau} workload={a.num_prompts}x{a.output_len} -> {out_dir}", flush=True)
    t0 = time.time()
    if a.phase == "identity":
        result = run_identity(a, extra, out_dir, server_python, rargs)
    else:
        result = run_speed(a, extra, out_dir, server_python, rargs)
    result["elapsed_s"] = time.time() - t0
    print(f"[matched] ===== {a.phase} done in {result['elapsed_s']/60:.1f} min =====",
          flush=True)
    _log_wandb(a, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
