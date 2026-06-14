#!/usr/bin/env python
"""Self-referential greedy-identity interlock (PR #114 — corrects PR #96).

WHAT #96 GOT WRONG.  PR #96's interlock compared a COMPOSED frontier submission
(#71 tree M-widen × #84 SplitK W4A16) against the *deployed baseline* submission
and RED-flagged ANY byte divergence between them. That check is over-strict and
does NOT match the official gate. program.md (lines 27–28) requires greedy decode
to be token-identical to *plain greedy AR decode for the submitted checkpoint* —
the gate is SELF-REFERENTIAL: it compares the submission's speculative output to
the SAME submission's OWN drafter-off M=1 autoregressive decode, NOT to a canonical
fp32/bf16 reference and NOT to a different deployed build.

THE ANCHOR.  PR #52 shipped an int4-Marlin submission that PASSED the official
128/128 greedy gate. That is impossible against a canonical fp32/bf16 reference
(int4 quant-noise dwarfs the near-tie ULP gaps that batch-geometry flips ride on);
it is only possible because the official reference is the submission's *own int4
AR* decode. So two submissions with different kernels/quant can BOTH pass the gate
while emitting different tokens from each other — exactly the case #96 would have
RED-flagged. Comparing a new frontier to the deployed baseline is the wrong test.

THE CORRECT INTERLOCK.  For any new frontier the only question the gate actually
asks is: does the submission's full speculative stack emit byte-identical greedy
tokens to the SAME submission's own drafter-off M=1 AR decode?  If yes, it passes
the official gate BY CONSTRUCTION — regardless of how far its tokens sit from the
deployed baseline. (Acceptance-rule proof: the fused accept kernel always emits
``target_argmax_id`` and rejects on the first draft≠target mismatch, so a perfectly
batch-invariant verifier reproduces the M=1 argmax trajectory exactly; any residual
divergence is the M=K+1 batched-verify GEMM reducing in a different float order than
M=1 sequential decode and flipping a near-tie argmax — see PR #5 / #73.)

THREE CHECKS (self-referential), in precondition order:
  1. spec-ON self-determinism   (run-to-run byte-identical)        -> precondition
       a non-deterministic candidate (PR #38 served wobble) makes the gate verdict
       unstable; pin a reproducible --config (e.g. fa_sliding_off) and re-run.
  2. spec-OFF self-determinism  (the own-AR reference is itself stable) -> precondition
  3. self-consistency (THE GATE): every spec-ON run is GREEDY_IDENTICAL to the
       spec-OFF own-AR reference, via the OFFICIAL verifier (greedy_gate.compare).

GREEN iff (3) is GREEDY_IDENTICAL with 0 divergent AND (1)/(2) are self-deterministic
  -> the stack reproduces its own greedy AR; it is greedy-safe BY CONSTRUCTION.
RED if (3) is DIVERGENT -> the stack does NOT reproduce its own greedy AR (the
  "greedy-safe by construction" claim FAILS for this verifier); onset diagnostics
  (first divergent token, late/stochastic vs early/systematic) localize the cause.
INCONCLUSIVE if captures are missing, a self-determinism precondition fails, or the
  official verifier reads INCOMPARABLE (prompt-set / integrity mismatch).

spec-OFF is captured by ``greedy_determinism.py --spec-off``, which injects
SENPAI_REFERENCE_MODE=1 so serve.py clears SPECULATIVE_CONFIG and vLLM runs M=1 AR
on the submission's OWN engine / kernels / quant — the only removed variable is
speculation. spec-ON and spec-OFF therefore differ ONLY in the drafter, which is
exactly the variable the official gate isolates.

It reuses the proven #73 primitives verbatim:
  * greedy_determinism.py            — served reload + capture_decode (subprocess)
  * analyze_determinism.load_runs    — per-reload token loader (keyed by prompt idx)
  * analyze_determinism.pair_stats   — run-to-run byte-identity (self-determinism)
  * greedy_gate.compare              — the OFFICIAL token-identity verdict (the gate)

Single command (capture both arms of ONE submission + run the gate):

  python scripts/validity/greedy_identity_interlock.py --self-referential \
      --submission submissions/fa2sw_precache_kenyan \
      --runs 3 --config default

Diff pre-captured roots only (no GPU; offline re-analysis / synthetic self-test):

  python scripts/validity/greedy_identity_interlock.py --self-referential --skip-capture \
      --spec-root  <OUT>/default \
      --ar-root    <OUT>/default__specoff --config default

LOCAL ONLY. The capture step serves on the single assigned GPU; analysis is CPU.
No HF Job, no submission, greedy-identity contract untouched.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.local_validation import greedy_gate  # noqa: E402
from scripts.validity.analyze_determinism import load_runs, pair_stats  # noqa: E402

DET_SCRIPT = REPO / "scripts/validity/greedy_determinism.py"
IDENTITY_MIN = 0.999  # mean byte-identical frac >= this == "self-deterministic" (no slack)
SPECOFF_SUFFIX = "__specoff"  # greedy_determinism.py writes spec-off to <config>__specoff/
DEFAULT_TPS_REF = 454.338  # PR #90 locked linear-chain wall_tps (deployed spec-ON, same box)
TPS_GREEN_MAX_PCT = 2.0    # PR #122 gate: GREEN iff 0-divergent AND cost < this


def _wall_tps_for_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Median wall_tps over a capture arm (PR #122 test metric).

    wall_tps = num_completion_tokens / duration_s from each run's
    decode_summary.json -- the official ``output_throughput`` definition
    (research/tps_noise_floor/PROTOCOL.md; hf_bucket_single_job.py). Defensive:
    synthetic self-test trees carry no decode_summary.json, so missing/zero -> None.
    """
    vals: list[float] = []
    for r in runs:
        summ = Path(r["dir"]) / "decode_summary.json"
        if not summ.exists():
            continue
        try:
            s = json.loads(summ.read_text())
            ct, dur = s.get("num_completion_tokens"), s.get("duration_s")
            if ct and dur:
                vals.append(ct / dur)
        except (ValueError, OSError):
            continue
    if not vals:
        return {"median_wall_tps": None, "per_run_wall_tps": [], "n": 0}
    return {"median_wall_tps": statistics.median(vals),
            "per_run_wall_tps": sorted(vals), "n": len(vals)}


def _capture_pair(submission: Path, out_root: Path, *, config: str, runs: int,
                  num_prompts: int, output_len: int) -> None:
    """Serve ONE submission and capture both arms: spec-ON then spec-OFF (M=1 AR).

    Writes out_root/<config>/run_XX (spec-ON) and out_root/<config>__specoff/run_XX
    (spec-OFF) — the layout load_runs() reads. The two arms differ ONLY in the
    --spec-off flag (SENPAI_REFERENCE_MODE=1 -> drafter cleared), so any token
    divergence between them is attributable solely to speculation.
    """
    base = [
        sys.executable, str(DET_SCRIPT),
        "--submission", str(submission),
        "--config", config,
        "--runs", str(runs),
        "--num-prompts", str(num_prompts),
        "--output-len", str(output_len),
        "--out-root", str(out_root),
    ]
    print(f"[interlock] capture spec-ON : {' '.join(base)}", flush=True)
    subprocess.run(base, check=True)
    print(f"[interlock] capture spec-OFF: {' '.join(base)} --spec-off", flush=True)
    subprocess.run(base + ["--spec-off"], check=True)


def _self_determinism(runs: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """All run-to-run pairs within one stack must be byte-identical (sha-level)."""
    if len(runs) < 2:
        return {"label": label, "num_runs": len(runs), "checkable": False,
                "min_byte_identical_frac": None, "deterministic": None,
                "num_divergent_pairs": 0, "onsets": []}
    fracs, onsets, ndiv = [], [], 0
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            s = pair_stats(runs[i]["rows"], runs[j]["rows"])
            fracs.append(s["byte_identical_frac"])
            onsets.extend(s["onsets"])
            ndiv += s["num_divergent"]
    mn = min(fracs) if fracs else None
    return {"label": label, "num_runs": len(runs), "checkable": True,
            "min_byte_identical_frac": mn,
            "deterministic": (mn is not None and mn >= IDENTITY_MIN),
            "num_divergent_pairs": ndiv, "onsets": sorted(onsets)}


def _self_consistency(ar_runs: list[dict[str, Any]], spec_runs: list[dict[str, Any]],
                      output_len: int) -> dict[str, Any]:
    """THE GATE: every spec-ON run must be GREEDY_IDENTICAL to the spec-OFF own-AR
    reference, judged by the OFFICIAL verifier (greedy_gate.compare).

    The spec-OFF reference is run_00 of the drafter-off capture (the submission's
    own M=1 AR greedy). Each spec-ON reload is scored against it exactly as the
    official gate scores a candidate against its served spec-off reference.
    """
    ref_dir = Path(ar_runs[0]["dir"])
    ref_path = ref_dir / "decode_outputs.jsonl"
    per_run, all_onsets, n_div_runs, n_incomparable = [], [], 0, 0
    for sr in spec_runs:
        cand_path = Path(sr["dir"]) / "decode_outputs.jsonl"
        try:
            report = greedy_gate.compare(ref_path, cand_path)
        except (ValueError, FileNotFoundError, OSError) as exc:
            per_run.append({"spec_run_idx": sr["run_idx"], "verdict": "ERROR", "error": str(exc)})
            n_incomparable += 1
            continue
        onset = greedy_gate.onset_summary(report)
        per_run.append({
            "spec_run_idx": sr["run_idx"],
            "verdict": report.verdict,
            "num_identical": report.num_identical,
            "num_divergent": report.num_divergent,
            "total_tokens_compared": report.total_tokens_compared,
            "total_divergent_tokens": report.total_divergent_tokens,
            "token_div_frac": (report.total_divergent_tokens / report.total_tokens_compared
                               if report.total_tokens_compared else None),
            "onset_min": onset.get("onset_min"),
            "onset_median": onset.get("onset_median"),
            "onset_max": onset.get("onset_max"),
        })
        all_onsets.extend(onset.get("onsets", []))
        if report.verdict == "GREEDY_IDENTICAL":
            pass
        elif report.verdict == "INCOMPARABLE":
            n_incomparable += 1
        else:  # DIVERGENT
            n_div_runs += 1
    all_onsets.sort()
    judged = [r for r in per_run if r["verdict"] in ("GREEDY_IDENTICAL", "DIVERGENT")]
    all_identical = (len(judged) == len(per_run) and len(per_run) > 0
                     and all(r["verdict"] == "GREEDY_IDENTICAL" for r in per_run))
    signature = None
    if all_onsets:
        med = statistics.median(all_onsets)
        signature = ("late/stochastic (FP-reduction near-tie flips)" if med > 0.1 * output_len
                     else "early/systematic (genuine decode/path change)")
    return {
        "reference_dir": str(ref_dir),
        "reference_kind": greedy_gate.reference_kind(ref_path),
        "num_spec_runs": len(spec_runs),
        "per_run": per_run,
        "all_greedy_identical": all_identical,
        "num_divergent_runs": n_div_runs,
        "num_incomparable_runs": n_incomparable,
        "onset_min": all_onsets[0] if all_onsets else None,
        "onset_median": int(statistics.median(all_onsets)) if all_onsets else None,
        "onset_max": all_onsets[-1] if all_onsets else None,
        "onset_signature": signature,
    }


def interlock(spec_runs: list[dict[str, Any]], ar_runs: list[dict[str, Any]],
              config: str, output_len: int,
              tps_ref: float = DEFAULT_TPS_REF) -> dict[str, Any]:
    if not spec_runs or not ar_runs:
        return {"verdict": "INCONCLUSIVE",
                "reason": f"missing captures (spec-ON runs={len(spec_runs)}, "
                          f"spec-OFF runs={len(ar_runs)}) under config '{config}'",
                "config": config}

    spec_self = _self_determinism(spec_runs, "spec_on")
    ar_self = _self_determinism(ar_runs, "spec_off_ar")
    consistency = _self_consistency(ar_runs, spec_runs, output_len)

    # precondition 1: spec-ON must be self-deterministic (PR #38 wobble swamps the gate)
    if spec_self["checkable"] and not spec_self["deterministic"]:
        verdict = "INCONCLUSIVE"
        reason = (f"spec-ON NOT self-deterministic run-to-run "
                  f"(min byte-identical {spec_self['min_byte_identical_frac']:.4f}); served "
                  f"nondeterminism (PR #38) makes the gate verdict unstable — pin a reproducible "
                  f"--config (e.g. fa_sliding_off) and re-run")
    # precondition 2: the spec-OFF own-AR reference must itself be stable
    elif ar_self["checkable"] and not ar_self["deterministic"]:
        verdict = "INCONCLUSIVE"
        reason = (f"spec-OFF own-AR reference NOT self-deterministic "
                  f"(min byte-identical {ar_self['min_byte_identical_frac']:.4f}); the reference "
                  f"wobbles run-to-run so the self-consistency comparison is meaningless — pin a "
                  f"reproducible --config and re-run")
    elif consistency["num_incomparable_runs"] > 0:
        verdict = "INCONCLUSIVE"
        reason = (f"official verifier read INCOMPARABLE on "
                  f"{consistency['num_incomparable_runs']}/{consistency['num_spec_runs']} spec-ON "
                  f"runs (prompt-set / integrity mismatch between spec-ON and spec-OFF captures)")
    elif consistency["all_greedy_identical"]:
        verdict = "GREEN"
        reason = ("spec-ON emits byte-identical greedy tokens to its OWN drafter-off M=1 AR "
                  f"(official GREEDY_IDENTICAL on all {consistency['num_spec_runs']} reloads, 0 "
                  "divergent) and both arms are self-deterministic — the stack passes the "
                  "self-referential greedy gate BY CONSTRUCTION")
    else:
        verdict = "RED"
        bits = [f"{consistency['num_divergent_runs']}/{consistency['num_spec_runs']} spec-ON reloads "
                f"DIVERGENT from the submission's own M=1 AR (official verifier)"]
        if consistency["onset_min"] is not None:
            bits.append(f"first divergence at token {consistency['onset_min']} "
                        f"(median {consistency['onset_median']}; {consistency['onset_signature']})")
        bits.append("the speculative stack does NOT reproduce its own greedy AR — "
                    "'greedy-safe by construction' FAILS for this verifier")
        reason = "; ".join(bits)

    # --- PR #122 deliverables: divergence-token count + wall_tps cost ----------
    # batch_invariant_self_divergence_tokens: the per-comparison divergent-token
    # count (target 0, from #114's 36751). Runs are self-deterministic so all
    # per_run counts coincide; take the max (0 iff every spec-ON reload is
    # byte-identical to the own-AR reference).
    div_tokens = max((r.get("total_divergent_tokens") or 0
                      for r in consistency["per_run"]), default=0)
    bi_tps = _wall_tps_for_runs(spec_runs)
    bi_wall_tps = bi_tps["median_wall_tps"]
    tps_cost_pct = (100.0 * (tps_ref - bi_wall_tps) / tps_ref
                    if bi_wall_tps else None)
    # Tri-color batch-invariance gate (divergence AND TPS), distinct from the
    # divergence-only `verdict` above: GREEN iff 0-divergent and cost < 2%.
    if verdict == "GREEN":
        if tps_cost_pct is None:
            bi_gate = "GREEN_TPS_UNKNOWN"
        elif tps_cost_pct < TPS_GREEN_MAX_PCT:
            bi_gate = "GREEN"
        else:
            bi_gate = "AMBER"
    elif verdict == "RED":
        bi_gate = "RED"
    else:
        bi_gate = "INCONCLUSIVE"

    return {
        "verdict": verdict,
        "reason": reason,
        "config": config,
        "mode": "self_referential",
        "identity_min_threshold": IDENTITY_MIN,
        "spec_on_self_determinism": spec_self,
        "spec_off_ar_self_determinism": ar_self,
        "self_consistency_gate": consistency,
        "self_referential_gate_confirmed": ("yes" if verdict == "GREEN"
                                            else "no" if verdict == "RED" else "inconclusive"),
        "primary_metric": {"name": "self_referential_divergent_runs",
                           "value": consistency["num_divergent_runs"]},
        # PR #122 named metrics:
        "batch_invariant_self_divergence_tokens": div_tokens,
        "batch_invariant_wall_tps": bi_wall_tps,
        "batch_invariant_tps_cost_pct": tps_cost_pct,
        "tps_ref": tps_ref,
        "spec_on_wall_tps": bi_tps,
        "batch_invariance_gate": bi_gate,
    }


_VERDICT_CODE = {"GREEN": 1, "INCONCLUSIVE": 0, "RED": -1}


def _wandb_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Flatten the interlock verdict into numeric metrics for W&B."""
    sc = report.get("self_consistency_gate") or {}
    ss = report.get("spec_on_self_determinism") or {}
    asd = report.get("spec_off_ar_self_determinism") or {}
    div_fracs = [r["token_div_frac"] for r in sc.get("per_run", [])
                 if r.get("token_div_frac") is not None]
    return {
        "verdict": report.get("verdict"),
        "verdict_code": _VERDICT_CODE.get(report.get("verdict"), -9),
        "self_referential_gate_confirmed": report.get("self_referential_gate_confirmed"),
        "self_referential_divergent_runs": (report.get("primary_metric") or {}).get("value"),
        "num_spec_runs": sc.get("num_spec_runs"),
        "all_greedy_identical": int(bool(sc.get("all_greedy_identical"))),
        "num_divergent_runs": sc.get("num_divergent_runs"),
        "num_incomparable_runs": sc.get("num_incomparable_runs"),
        "token_div_frac_max": max(div_fracs) if div_fracs else 0.0,
        "onset_min": sc.get("onset_min"),
        "onset_median": sc.get("onset_median"),
        "onset_max": sc.get("onset_max"),
        "spec_on_self_deterministic": int(bool(ss.get("deterministic"))),
        "spec_off_self_deterministic": int(bool(asd.get("deterministic"))),
        "spec_on_min_byte_identical": ss.get("min_byte_identical_frac"),
        "spec_off_min_byte_identical": asd.get("min_byte_identical_frac"),
        "splitk_ppl_projected": 2.378,  # Step 4: first-order, moot (greedy-identity is binding)
        # PR #122 named metrics:
        "batch_invariant_self_divergence_tokens": report.get("batch_invariant_self_divergence_tokens"),
        "batch_invariant_wall_tps": report.get("batch_invariant_wall_tps"),
        "batch_invariant_tps_cost_pct": report.get("batch_invariant_tps_cost_pct"),
        "tps_ref": report.get("tps_ref"),
        "batch_invariance_gate": report.get("batch_invariance_gate"),
    }


def log_to_wandb(report: dict[str, Any], *, wandb_group: str, wandb_name: str,
                 report_path: Path | None) -> None:
    summary = _wandb_summary(report)
    try:
        from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                           log_file_artifact, log_summary)
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] unavailable: {exc}", flush=True)
        return
    run = init_wandb_run(
        job_type="self-referential-greedy-gate", agent="kanna", name=wandb_name,
        group=wandb_group or None,
        tags=["self-referential-greedy-gate", *([wandb_group] if wandb_group else [])],
        config={"mode": report.get("mode"), "config": report.get("config"),
                "identity_min_threshold": report.get("identity_min_threshold"),
                "wandb_group": wandb_group},
    )
    if run is None:
        print("[wandb] run not created (no creds/disabled); interlock_report.json is the record",
              flush=True)
        return
    log_summary(run, summary, step=0)
    if report_path is not None:
        log_file_artifact(run, path=report_path, name="self_referential_interlock_report",
                          artifact_type="greedy-identity-interlock-report")
    finish_wandb(run)
    print(f"[wandb] logged run {wandb_name} (group={wandb_group})", flush=True)


def _runs_from(out_root: Path, config: str, explicit: str | None) -> list[dict[str, Any]]:
    """load_runs() for either an explicit <config>-dir or out_root/<config>."""
    if explicit:
        p = Path(explicit)
        return load_runs(p.parent, p.name)
    return load_runs(out_root, config)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-referential", action="store_true",
                    help="run the PR #114 self-referential gate (spec-ON vs the SAME "
                         "submission's spec-OFF M=1 AR). This is the only supported mode.")
    ap.add_argument("--submission", default=None,
                    help="ONE submission dir; both arms (spec-ON + spec-OFF) are captured from it")
    ap.add_argument("--config", default="default",
                    help="served config to pin for BOTH arms (default/fa_sliding_off/"
                         "splitkv_off/atomic_on); pin a reproducible one if spec-ON wobbles")
    ap.add_argument("--runs", type=int, default=3, help="fresh reloads per arm")
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--out-root", default=None,
                    help="parent dir; capture writes <config>/ and <config>__specoff/ under it")
    ap.add_argument("--spec-root", default=None,
                    help="explicit spec-ON capture dir (contains run_XX); overrides out_root/<config>")
    ap.add_argument("--ar-root", default=None,
                    help="explicit spec-OFF capture dir (contains run_XX); overrides out_root/<config>__specoff")
    ap.add_argument("--skip-capture", action="store_true",
                    help="diff existing --spec-root/--ar-root only (no GPU)")
    ap.add_argument("--report", default=None)
    ap.add_argument("--tps-ref", "--tps_ref", dest="tps_ref", type=float, default=DEFAULT_TPS_REF,
                    help="deployed spec-ON wall_tps reference for batch_invariant_tps_cost_pct "
                         f"(default {DEFAULT_TPS_REF}, PR #90 same-box anchor)")
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="4 prompts x 32 tok, 1 run/arm — plumbing check (spec-on vs own spec-off)")
    args = ap.parse_args()

    if not args.self_referential:
        ap.error("this interlock now runs ONLY the PR #114 self-referential gate; pass "
                 "--self-referential. (The PR #96 baseline-vs-composed check was over-strict — "
                 "see the module docstring.)")

    if args.smoke:
        args.num_prompts, args.output_len, args.runs = 4, 32, 1

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(args.out_root) if args.out_root else (
        REPO / "research/validity/self_referential_gate" / f"interlock-{ts}")

    if not args.skip_capture:
        if not args.submission:
            ap.error("--submission is required unless --skip-capture")
        _capture_pair(Path(args.submission), out_root, config=args.config, runs=args.runs,
                      num_prompts=args.num_prompts, output_len=args.output_len)

    spec_runs = _runs_from(out_root, args.config, args.spec_root)
    ar_runs = _runs_from(out_root, args.config + SPECOFF_SUFFIX, args.ar_root)

    report = interlock(spec_runs, ar_runs, args.config, args.output_len, tps_ref=args.tps_ref)
    report_path = Path(args.report) if args.report else out_root / "interlock_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 78, flush=True)
    print("SELF-REFERENTIAL GREEDY-IDENTITY INTERLOCK (PR #114 — corrects #96)", flush=True)
    print("=" * 78, flush=True)
    ss, asd, sc = (report.get("spec_on_self_determinism"), report.get("spec_off_ar_self_determinism"),
                   report.get("self_consistency_gate"))
    if ss is not None:
        print(f"  spec-ON  self-determ : runs={ss['num_runs']} "
              f"min_byte_identical={ss['min_byte_identical_frac']} det={ss['deterministic']}", flush=True)
        print(f"  spec-OFF self-determ : runs={asd['num_runs']} "
              f"min_byte_identical={asd['min_byte_identical_frac']} det={asd['deterministic']}", flush=True)
        print(f"  GATE (spec-ON vs own M=1 AR, official verifier):", flush=True)
        print(f"      reference         : {sc['reference_dir']} (kind={sc['reference_kind']})", flush=True)
        print(f"      all GREEDY_IDENTICAL={sc['all_greedy_identical']}  "
              f"divergent_runs={sc['num_divergent_runs']}  incomparable={sc['num_incomparable_runs']}", flush=True)
        for r in sc["per_run"]:
            extra = (f" tok_div_frac={r.get('token_div_frac')} onset_min={r.get('onset_min')}"
                     if r["verdict"] == "DIVERGENT" else "")
            print(f"        run {r['spec_run_idx']}: {r['verdict']}{extra}", flush=True)
        if sc.get("onset_signature"):
            print(f"      onset             : min={sc['onset_min']} median={sc['onset_median']} "
                  f"max={sc['onset_max']} [{sc['onset_signature']}]", flush=True)
    print("-" * 78, flush=True)
    print(f"VERDICT: {report['verdict']}  (self_referential_gate_confirmed="
          f"{report.get('self_referential_gate_confirmed')})", flush=True)
    print(f"  {report['reason']}", flush=True)
    if "batch_invariant_self_divergence_tokens" in report:
        cost = report.get("batch_invariant_tps_cost_pct")
        print("-" * 78, flush=True)
        print(f"PR #122 metrics:", flush=True)
        print(f"  batch_invariant_self_divergence_tokens = "
              f"{report['batch_invariant_self_divergence_tokens']}  (target 0, #114=36751)", flush=True)
        print(f"  batch_invariant_wall_tps               = {report.get('batch_invariant_wall_tps')}  "
              f"(ref {report.get('tps_ref')})", flush=True)
        print(f"  batch_invariant_tps_cost_pct           = "
              f"{f'{cost:.3f}%' if cost is not None else None}", flush=True)
        print(f"  batch_invariance_gate                  = {report.get('batch_invariance_gate')}", flush=True)
    print(f"[interlock] wrote {report_path}", flush=True)

    if args.wandb_group:
        log_to_wandb(report, wandb_group=args.wandb_group,
                     wandb_name=args.wandb_name or "kanna/self-referential-greedy-gate",
                     report_path=report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
