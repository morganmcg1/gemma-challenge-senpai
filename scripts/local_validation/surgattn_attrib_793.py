"""Surgattn drafter-vs-main attribution A/B/C orchestrator (PR #793).

Drives the ``int4_mtp_bi0_surgattn_attrib`` submission across its three
``SURGATTN_ARM`` settings on one shared int4 + MTP-drafter serve per arm, and
attributes the surgattn 3D-decode speedup between the *byte-identical* drafter
share and the *identity-breaking* main-model share.

Arms (see ``submissions/int4_mtp_bi0_surgattn_attrib/_surgattn_arm.py``)
----------------------------------------------------------------------
* ``control_2d``      force-2D on every M=1 forward == shipped bi0. Greedy-identity
                      + TPS reference; its 128-prompt decode is the identity oracle.
* ``drafter_only_3d`` force-2D on the main-model forwards (byte-identical emitted
                      tokens), let the kernel gate pick 3D on the drafter proposer
                      forwards only. Expected 128/128 greedy-identical to control.
* ``all_3d``          surgattn OFF (3D wherever the gate fires). Identity-breaking
                      anchor; ``all_3d - control`` should ~reproduce wirbel's +6.69%.

Measurement per arm (one serve, ordered so the engine meter stays clean):
  1. one tiny completion to fire every forward type, then assert the per-forward
     2D/3D dispatch from the server log (the smoke discriminator check);
  2. official 128x512 decode capture (greedy-identity + completion + the clean
     whole-run engine-meter TPS + E_accept parsed from the log right after);
  3. N>=5 warm single-stream TPS probe reps (rep0 discarded) -> median + CV;
  4. 128-record served PPL;
  5. Prometheus E_accept cross-check before teardown.

Attribution (``--mode analyze``): greedy-identity of each arm vs control_2d's
decode, the drafter/main/total steady-TPS split, a Welch z of each arm's probe
reps vs control, and the drafter_only_3d fire-worthiness gates (byte-identical
AND TPS > control by >2 sigma AND PPL <= 2.42). One W&B run per arm under group
``bi0-surgattn-attrib``.

LOCAL ONLY. This never launches an HF job or a competition submission; it only
serves on the assigned local GPU. A fire-worthy drafter_only_3d result is a
terminal finding the advisor turns into an HF-approval issue by hand.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import urllib.error
from pathlib import Path
from typing import Any

from . import greedy_gate, harness, paths, serve_profile

ARMS = ("control_2d", "drafter_only_3d", "all_3d")
CONTROL = "control_2d"

SUBMISSION = paths.ROOT / "submissions" / "int4_mtp_bi0_surgattn_attrib"
OUT_ROOT = paths.ROOT / "research" / "surgattn_attrib_793"
# Pre-built vLLM 0.22.0 serve venv (matches manifest dep hash); never rebuilt here.
SERVER_PY = Path("/tmp/senpai-venvs/20f658587e8a6643/bin/python")

WANDB_GROUP = "bi0-surgattn-attrib"
DISPATCH_PREFIX = "[surgattn-attrib"
PPL_CAP = 2.42  # official leaderboard PPL cap.
# wirbel's reported full surgattn-3D quality-regate speedup, for the attribution
# cross-check (all_3d - control should land near this).
WIRBEL_TOTAL_PCT = 6.69
# bi0 control's official a10g-small TPS, the projection base named in PR #793.
BI0_OFFICIAL_TPS = 218.02


def _arm_env(arm: str) -> dict[str, str]:
    """Submission env overrides for one arm. CUDA pinned to the in-container GPU
    (inherited index is stale, fleet-leak history #780); native sampler avoids the
    cuRAND JIT failure — neither touches greedy/PPL numerics."""
    return {
        "SURGATTN_ARM": arm,
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "CUDA_VISIBLE_DEVICES": "0",
    }


def _dispatch_lines(log_text: str) -> list[str]:
    return [ln.strip() for ln in log_text.splitlines() if DISPATCH_PREFIX in ln]


def _check_dispatch(arm: str, lines: list[str]) -> dict[str, Any]:
    """Verify the server log proves each forward type took its intended kernel path.

    The load-bearing assertion for ``drafter_only_3d`` is that the drafter line says
    the segm buffers are PRESENT (so the launch gate *can* pick 3D); if they are
    ABSENT the arm collapses onto control_2d and there is no speedup to recover.
    """
    text = "\n".join(lines)
    main_forced = "forcing 2D" in text and "main-model forward" in text
    main_allowed = "allowing gate choice on main-model forward" in text
    drafter_forced = "forcing 2D" in text and "drafter-proposer forward" in text
    drafter_allowed = "allowing gate choice on drafter-proposer forward" in text
    drafter_segm_present = "drafter-proposer forward (segm buffers PRESENT" in text
    proposer_marker = any("proposer scope marker installed" in ln for ln in lines)

    if arm == "control_2d":
        ok = main_forced and drafter_forced
        expected = "force-2D on BOTH main-model and drafter-proposer forwards"
    elif arm == "drafter_only_3d":
        # main forced to 2D; drafter allowed to pick 3D AND segm present so it can.
        ok = main_forced and drafter_allowed and drafter_segm_present
        expected = ("force-2D on main-model; allow gate on drafter-proposer with "
                    "segm buffers PRESENT (3D engageable)")
    else:  # all_3d
        ok = main_allowed and drafter_allowed
        expected = "allow gate choice on BOTH forward types (surgattn OFF)"
    return {
        "ok": ok,
        "expected": expected,
        "main_forced_2d": main_forced,
        "main_allowed_gate": main_allowed,
        "drafter_forced_2d": drafter_forced,
        "drafter_allowed_gate": drafter_allowed,
        "drafter_segm_present": drafter_segm_present,
        "proposer_scope_marker_installed": proposer_marker,
    }


def _cv(values: list[float]) -> float | None:
    vals = [v for v in values if v == v]
    if len(vals) < 2:
        return None
    m = statistics.fmean(vals)
    if m == 0:
        return None
    return statistics.pstdev(vals) / m


def run_arm(
    arm: str,
    *,
    num_prompts: int,
    output_len: int,
    tps_reps: int,
    decode_tokens: int,
    smoke: bool,
) -> dict[str, Any]:
    """Serve one arm and capture its measurements; write ``<arm>/result.json``."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}")
    out_dir = OUT_ROOT / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "server.log"
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    result: dict[str, Any] = {
        "arm": arm,
        "smoke": smoke,
        "num_prompts": num_prompts,
        "output_len": output_len,
        "tps_reps_requested": tps_reps,
        "decode_tokens": decode_tokens,
        "submission": str(SUBMISSION),
        "server_python": str(SERVER_PY),
        "created_at": stamp,
        "out_dir": str(out_dir),
        "server_log": str(log_path),
        "failures": [],
    }
    print(f"\n===== ARM {arm} (smoke={smoke}) =====", flush=True)
    with harness.LocalServer(
        SUBMISSION,
        server_python=SERVER_PY,
        port=8000,
        log_path=log_path,
        extra_env=_arm_env(arm),
        startup_timeout_s=1800,
    ) as srv:
        result["model_id"] = srv.model_id
        result["served_model_name"] = srv.served_model_name

        # (1) Fire every forward type, then prove the per-forward dispatch.
        try:
            harness._completion(srv.base_url, srv.served_model_name,
                                "Explain step by step how attention works.", 24)
        except (urllib.error.URLError, OSError) as exc:
            result["failures"].append(f"dispatch-fire completion error: {exc}")
        time.sleep(1.0)
        dl = _dispatch_lines(log_path.read_text())
        result["dispatch_lines"] = dl
        result["dispatch_check"] = _check_dispatch(arm, dl)
        print(f"[{arm}] dispatch_ok={result['dispatch_check']['ok']} :: "
              f"{result['dispatch_check']['expected']}", flush=True)
        for ln in dl:
            print(f"    {ln}", flush=True)

        if smoke:
            (out_dir / "smoke.json").write_text(json.dumps(result, indent=2, sort_keys=True))
            return result

        if not result["dispatch_check"]["ok"]:
            # Don't burn the heavy 128x512 decode on a misrouted arm.
            result["failures"].append(
                "dispatch check failed pre-decode; skipping heavy measurement")
            (out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
            print(f"[{arm}] ABORT: dispatch check failed; see {log_path}", flush=True)
            return result

        # (2) Official 128x512 decode capture (identity + completion source).
        try:
            t0 = time.time()
            decode_summary = harness.capture_decode(
                SERVER_PY,
                base_url=srv.base_url,
                model=srv.served_model_name,
                out_file=out_dir / "decode_outputs.jsonl",
                summary_file=out_dir / "decode_summary.json",
                num_prompts=num_prompts,
                output_len=output_len,
            )
            result["decode_wall_s"] = time.time() - t0
            result["completed"] = decode_summary["num_records"]
            result["decode_summary"] = decode_summary
        except Exception as exc:  # noqa: BLE001 - record and continue
            result["failures"].append(f"decode stage error: {exc}")
            print(f"[{arm}] ERROR decode: {exc}", flush=True)

        # Clean whole-run engine meter + E_accept: parse the log NOW, before the
        # TPS probe bursts add idle-diluted "Avg generation throughput" lines.
        result["spec_log_after_decode"] = serve_profile.parse_spec_log(log_path.read_text())

        # (3) Warm single-stream TPS probe reps (rep0 discarded as arm warmup).
        reps: list[float] = []
        for i in range(tps_reps):
            try:
                pr = harness.probe_tps(srv.base_url, srv.served_model_name,
                                       decode_tokens=decode_tokens)
                reps.append(float(pr["decode_tps_single_stream"]))
            except (urllib.error.URLError, OSError, RuntimeError) as exc:
                reps.append(float("nan"))
                result["failures"].append(f"tps rep {i} error: {exc}")
        result["tps_probe_reps_all"] = reps
        kept = [v for v in reps[1:] if v == v]
        result["tps_probe_reps_kept"] = kept
        if kept:
            result["tps_probe_median"] = statistics.median(kept)
            result["tps_probe_mean"] = statistics.fmean(kept)
            result["tps_probe_stdev"] = statistics.stdev(kept) if len(kept) > 1 else 0.0
            result["tps_probe_cv"] = _cv(kept)
            result["tps_probe_n_kept"] = len(kept)
            print(f"[{arm}] probe TPS median={result['tps_probe_median']:.2f} "
                  f"mean={result['tps_probe_mean']:.2f} cv={result['tps_probe_cv']}", flush=True)

        # (4) Served PPL (128 records, output-length independent).
        try:
            ppl_summary = harness.run_ppl(
                SERVER_PY,
                base_url=srv.base_url,
                model=srv.served_model_name,
                out_file=out_dir / "ppl_results.jsonl",
                summary_file=out_dir / "ppl_summary.json",
            )
            result["ppl"] = ppl_summary["ppl"]
            result["ppl_summary"] = ppl_summary
            print(f"[{arm}] PPL={ppl_summary['ppl']:.4f}", flush=True)
        except Exception as exc:  # noqa: BLE001
            result["failures"].append(f"ppl stage error: {exc}")
            print(f"[{arm}] ERROR ppl: {exc}", flush=True)

        # (5) Prometheus E_accept cross-check before teardown.
        try:
            result["spec_metrics"] = serve_profile.parse_spec_metrics(
                serve_profile._get_text(f"{srv.base_url}/metrics"))
        except (urllib.error.URLError, OSError) as exc:
            result["spec_metrics"] = {"error": str(exc)}

    # Final full-log parse (engine meter now includes probe bursts; kept separately).
    log_text = log_path.read_text()
    result["spec_log_full"] = serve_profile.parse_spec_log(log_text)
    result["dispatch_lines"] = _dispatch_lines(log_text)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"[{arm}] artifacts -> {out_dir}", flush=True)
    return result


def _steady_tps(res: dict[str, Any]) -> float | None:
    """Clean whole-run engine-meter decode TPS (parsed right after the 128 decode)."""
    sl = res.get("spec_log_after_decode") or {}
    v = sl.get("steady_gen_tps_mean")
    return float(v) if isinstance(v, (int, float)) else None


def _welch_z(a: dict[str, Any], b: dict[str, Any]) -> float | None:
    """Welch z of arm ``a``'s probe reps vs arm ``b``'s (mean diff / pooled se)."""
    ma, mb = a.get("tps_probe_mean"), b.get("tps_probe_mean")
    sa, sb = a.get("tps_probe_stdev"), b.get("tps_probe_stdev")
    na, nb = a.get("tps_probe_n_kept"), b.get("tps_probe_n_kept")
    if None in (ma, mb, sa, sb, na, nb) or na < 1 or nb < 1:
        return None
    se = math.sqrt((sa * sa) / na + (sb * sb) / nb)
    if se == 0:
        return math.inf if ma != mb else 0.0
    return (ma - mb) / se


def _load_decode(path: Path) -> dict[str, list[int]]:
    """``id`` -> completion token ids for one decode_outputs.jsonl."""
    out: dict[str, list[int]] = {}
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            out[rec["id"]] = rec["completion_token_ids"]
    return out


def _category(prompt_id: str) -> str:
    """Prompt-set tag from the decode id (``mmlu_pro-…`` -> ``mmlu_pro``)."""
    return prompt_id.rsplit("-", 1)[0]


def _divergence_by_category(ref_decode: Path, cand_decode: Path) -> dict[str, Any]:
    """Per-prompt-set divergence of a candidate decode vs the control reference.

    The advisor's #792 finding is that the verify-GEMM ULP near-tie flips that a
    perturbed drafter proposal leaks into emitted tokens land ENTIRELY on the
    GPQA/MMLU-Pro reasoning prompts (0 on AIME). The official decode set is a
    reasoning-heavy mix, so this slices the byte-identity result by prompt set to
    show *where* any divergence lives — the load-bearing cross-check for whether
    a 'byte-identical by construction' claim actually holds on reasoning prompts.
    """
    ref, cand = _load_decode(ref_decode), _load_decode(cand_decode)
    by_cat: dict[str, dict[str, Any]] = {}
    for pid, ref_ids in ref.items():
        cat = _category(pid)
        slot = by_cat.setdefault(cat, {"total": 0, "divergent": 0, "divergent_ids": [], "onsets": []})
        slot["total"] += 1
        cand_ids = cand.get(pid)
        if cand_ids is None or cand_ids == ref_ids:
            continue
        slot["divergent"] += 1
        slot["divergent_ids"].append(pid)
        onset = next((i for i, (a, b) in enumerate(zip(ref_ids, cand_ids)) if a != b),
                     min(len(ref_ids), len(cand_ids)))
        slot["onsets"].append(onset)
    return by_cat


def _identity(ref_decode: Path, cand_decode: Path) -> dict[str, Any]:
    rep = greedy_gate.compare(ref_decode, cand_decode)
    onset = greedy_gate.onset_summary(rep)
    return {
        "verdict": rep.verdict,
        "num_prompts_compared": rep.num_prompts_compared,
        "num_identical": rep.num_identical,
        "num_divergent": rep.num_divergent,
        "total_tokens_compared": rep.total_tokens_compared,
        "total_divergent_tokens": rep.total_divergent_tokens,
        "onset": onset,
        "by_category": _divergence_by_category(ref_decode, cand_decode),
    }


def analyze(*, wandb_log: bool) -> dict[str, Any]:
    """Read all per-arm result.json, compute identity + attribution + gates."""
    results: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        p = OUT_ROOT / arm / "result.json"
        if not p.exists():
            raise FileNotFoundError(f"missing {p}; run --mode run --arms {arm} first")
        results[arm] = json.loads(p.read_text())

    ref_decode = OUT_ROOT / CONTROL / "decode_outputs.jsonl"
    if not ref_decode.exists():
        raise FileNotFoundError(f"control decode reference missing: {ref_decode}")

    identity: dict[str, Any] = {}
    for arm in ARMS:
        cand = OUT_ROOT / arm / "decode_outputs.jsonl"
        identity[arm] = _identity(ref_decode, cand) if cand.exists() else {"verdict": "NO_DECODE"}

    tps = {arm: _steady_tps(results[arm]) for arm in ARMS}
    ctl, dft, alld = tps[CONTROL], tps["drafter_only_3d"], tps["all_3d"]

    def pct(x: float | None) -> float | None:
        if x is None or ctl in (None, 0):
            return None
        return 100.0 * (x - ctl) / ctl

    attribution = {
        "steady_gen_tps": tps,
        "drafter_share_tps_delta": (dft - ctl) if (dft is not None and ctl is not None) else None,
        "main_share_tps_delta": (alld - dft) if (alld is not None and dft is not None) else None,
        "total_tps_delta": (alld - ctl) if (alld is not None and ctl is not None) else None,
        "drafter_share_pct": pct(dft),
        "main_share_pct": (pct(alld) - pct(dft)) if (pct(alld) is not None and pct(dft) is not None) else None,
        "total_pct": pct(alld),
        "wirbel_total_pct_reference": WIRBEL_TOTAL_PCT,
        "bi0_official_tps_base": BI0_OFFICIAL_TPS,
        "drafter_only_3d_projected_official_tps": (
            BI0_OFFICIAL_TPS * (1.0 + pct(dft) / 100.0) if pct(dft) is not None else None),
    }

    z_drafter = _welch_z(results["drafter_only_3d"], results[CONTROL])
    z_all = _welch_z(results["all_3d"], results[CONTROL])
    significance = {
        "probe_median_tps": {arm: results[arm].get("tps_probe_median") for arm in ARMS},
        "probe_mean_tps": {arm: results[arm].get("tps_probe_mean") for arm in ARMS},
        "probe_cv": {arm: results[arm].get("tps_probe_cv") for arm in ARMS},
        "welch_z_drafter_vs_control": z_drafter,
        "welch_z_all3d_vs_control": z_all,
    }

    dft_id = identity["drafter_only_3d"]
    dft_ppl = results["drafter_only_3d"].get("ppl")
    gate = {
        "byte_identical_to_control": dft_id.get("verdict") == "GREEDY_IDENTICAL",
        "byte_identical_frac": (
            dft_id.get("num_identical", 0) / dft_id["num_prompts_compared"]
            if dft_id.get("num_prompts_compared") else None),
        "tps_gt_control": (dft is not None and ctl is not None and dft > ctl),
        "tps_gt_control_2sigma": (z_drafter is not None and z_drafter > 2.0),
        "ppl_within_cap": (isinstance(dft_ppl, (int, float)) and dft_ppl <= PPL_CAP),
        "ppl": dft_ppl,
    }
    gate["fire_worthy"] = bool(
        gate["byte_identical_to_control"]
        and gate["tps_gt_control"]
        and gate["tps_gt_control_2sigma"]
        and gate["ppl_within_cap"]
    )
    # A clean null (byte-identical but ~0% faster) is an equally-complete finding.
    gate["clean_null"] = bool(
        gate["byte_identical_to_control"]
        and not gate["tps_gt_control_2sigma"]
    )

    # Cross-arm control: drafter_only_3d vs all_3d. These two arms differ ONLY in
    # the main-model attention path (2D vs gate-picked 3D); the drafter is on 3D in
    # both. If their decodes are byte-identical it proves two things at once —
    # (a) the stack is run-to-run deterministic across independent serves (so any
    # drafter_only_3d-vs-control divergence is induced, not sampler noise, the
    # equivalent of #792's same-config-twice control), and (b) the main-model 3D
    # path is a token-level no-op (the entire identity break + speedup is drafter
    # induced, since 3D only fires on M=1 forwards and the verify forward is M=K).
    cross = OUT_ROOT / "all_3d" / "decode_outputs.jsonl"
    dft_decode = OUT_ROOT / "drafter_only_3d" / "decode_outputs.jsonl"
    cross_arm = {"verdict": "NO_DECODE"}
    if cross.exists() and dft_decode.exists():
        rep_x = greedy_gate.compare(dft_decode, cross)
        cross_arm = {
            "comparison": "drafter_only_3d_vs_all_3d",
            "verdict": rep_x.verdict,
            "num_identical": rep_x.num_identical,
            "num_prompts_compared": rep_x.num_prompts_compared,
            "total_divergent_tokens": rep_x.total_divergent_tokens,
            "byte_identical": rep_x.verdict == "GREEDY_IDENTICAL",
        }

    report = {
        "arms": ARMS,
        "out_root": str(OUT_ROOT),
        "identity_vs_control": identity,
        "cross_arm_drafter_only_vs_all_3d": cross_arm,
        "attribution": attribution,
        "significance": significance,
        "drafter_only_3d_gate": gate,
        "per_arm": {arm: {
            "ppl": results[arm].get("ppl"),
            "completed": results[arm].get("completed"),
            "steady_gen_tps": tps[arm],
            "e_accept": (results[arm].get("spec_log_after_decode") or {}).get("e_accept_exact"),
            "probe_median_tps": results[arm].get("tps_probe_median"),
            "dispatch_ok": (results[arm].get("dispatch_check") or {}).get("ok"),
        } for arm in ARMS},
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "attribution_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    _print_report(report)
    if wandb_log:
        report["wandb_run_ids"] = _log_wandb(results, report)
        (OUT_ROOT / "attribution_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def _print_report(r: dict[str, Any]) -> None:
    print("\n" + "=" * 30 + " SURGATTN ATTRIBUTION (PR #793) " + "=" * 30, flush=True)
    for arm in ARMS:
        pa = r["per_arm"][arm]
        idv = r["identity_vs_control"][arm]
        print(f"  {arm:16s} tps={pa['steady_gen_tps']} ppl={pa['ppl']} "
              f"E_accept={pa['e_accept']} completed={pa['completed']} "
              f"id_vs_control={idv.get('verdict')} "
              f"({idv.get('num_identical')}/{idv.get('num_prompts_compared')})", flush=True)
    a = r["attribution"]
    print(f"\n  drafter share (byte-identical) : {a['drafter_share_pct']}%  "
          f"(+{a['drafter_share_tps_delta']} tps)", flush=True)
    print(f"  main-model share (id-breaking) : {a['main_share_pct']}%  "
          f"(+{a['main_share_tps_delta']} tps)", flush=True)
    print(f"  total (all_3d - control)       : {a['total_pct']}%  "
          f"(wirbel ref +{a['wirbel_total_pct_reference']}%)", flush=True)
    print(f"  drafter_only_3d projected official TPS: "
          f"{a['drafter_only_3d_projected_official_tps']} (base {a['bi0_official_tps_base']})", flush=True)
    # Where does any drafter_only_3d divergence live? (advisor #792: reasoning, not AIME)
    bycat = r["identity_vs_control"]["drafter_only_3d"].get("by_category") or {}
    print("\n  drafter_only_3d divergence by prompt set (advisor #792 cross-check):", flush=True)
    for cat in sorted(bycat):
        slot = bycat[cat]
        print(f"    {cat:14s} {slot['divergent']}/{slot['total']} divergent  "
              f"onsets={sorted(slot['onsets'])}", flush=True)
    x = r.get("cross_arm_drafter_only_vs_all_3d") or {}
    print(f"  cross-arm drafter_only_3d vs all_3d: {x.get('verdict')} "
          f"({x.get('num_identical')}/{x.get('num_prompts_compared')}) "
          f"-> determinism + main-model-3D no-op control", flush=True)
    g = r["drafter_only_3d_gate"]
    print(f"\n  GATE drafter_only_3d: fire_worthy={g['fire_worthy']} clean_null={g['clean_null']}", flush=True)
    print(f"    byte_identical={g['byte_identical_to_control']} "
          f"tps>control={g['tps_gt_control']} >2sigma={g['tps_gt_control_2sigma']} "
          f"ppl<=cap={g['ppl_within_cap']} (z={r['significance']['welch_z_drafter_vs_control']})", flush=True)
    print("=" * 92 + "\n", flush=True)


def _log_wandb(results: dict[str, dict[str, Any]], report: dict[str, Any]) -> dict[str, str]:
    try:
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_summary
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] unavailable: {exc}", flush=True)
        return {}
    run_ids: dict[str, str] = {}
    for arm in ARMS:
        res = results[arm]
        idv = report["identity_vs_control"][arm]
        run = init_wandb_run(
            job_type="surgattn-attrib",
            agent="land",
            name=f"land/surgattn-attrib-{arm}",
            group=WANDB_GROUP,
            tags=["pr793", "surgattn-attrib", arm],
            config={
                "arm": arm,
                "submission": res.get("submission"),
                "num_prompts": res.get("num_prompts"),
                "output_len": res.get("output_len"),
                "model_id": res.get("model_id"),
            },
        )
        if run is None:
            # init returns None when no real wandb is importable — most commonly
            # the local ``target/wandb/`` run-data dir shadows the package, or the
            # serve venv has no wandb. Warn loudly: a silent skip looks like a
            # successful no-op and loses the rich per-arm record this card needs.
            print(f"[wandb] WARNING: no run created for {arm!r} (wandb unavailable / "
                  "shadowed by local wandb/ dir); run analyze under a python with "
                  "wandb installed from outside target/", file=sys.stderr, flush=True)
            continue
        summary = {
            "arm": arm,
            "ppl": res.get("ppl"),
            "completed": res.get("completed"),
            "steady_gen_tps": report["per_arm"][arm]["steady_gen_tps"],
            "e_accept_exact": report["per_arm"][arm]["e_accept"],
            "tps_probe_median": res.get("tps_probe_median"),
            "tps_probe_mean": res.get("tps_probe_mean"),
            "tps_probe_cv": res.get("tps_probe_cv"),
            "dispatch_ok": (res.get("dispatch_check") or {}).get("ok"),
            "id_vs_control_verdict": idv.get("verdict"),
            "id_vs_control_num_identical": idv.get("num_identical"),
            "id_vs_control_num_divergent": idv.get("num_divergent"),
            "id_vs_control_total_divergent_tokens": idv.get("total_divergent_tokens"),
            "cross_arm_vs_all3d_byte_identical": (
                report.get("cross_arm_drafter_only_vs_all_3d") or {}).get("byte_identical"),
            "drafter_share_pct": report["attribution"]["drafter_share_pct"],
            "main_share_pct": report["attribution"]["main_share_pct"],
            "total_pct": report["attribution"]["total_pct"],
            "drafter_only_3d_fire_worthy": report["drafter_only_3d_gate"]["fire_worthy"],
            "drafter_only_3d_clean_null": report["drafter_only_3d_gate"]["clean_null"],
            "welch_z_drafter_vs_control": report["significance"]["welch_z_drafter_vs_control"],
        }
        for cat, slot in (idv.get("by_category") or {}).items():
            summary[f"div_{cat}"] = slot["divergent"]
            summary[f"total_{cat}"] = slot["total"]
        log_summary(run, summary, step=0)
        rid = getattr(run, "id", None)
        if rid:
            run_ids[arm] = rid
        finish_wandb(run)
        print(f"[wandb] logged {arm} -> {rid}", flush=True)
    return run_ids


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("smoke", "run", "analyze"), required=True)
    ap.add_argument("--arms", default=",".join(ARMS),
                    help="comma list of arms for smoke/run (default: all three)")
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--tps-reps", type=int, default=6, help="probe reps; rep0 discarded")
    ap.add_argument("--decode-tokens", type=int, default=256)
    ap.add_argument("--smoke-num-prompts", type=int, default=1)
    ap.add_argument("--smoke-output-len", type=int, default=24)
    ap.add_argument("--no-wandb", action="store_true", help="skip W&B logging in analyze")
    args = ap.parse_args(argv)

    for note in paths.prepare_local_gpu_env():
        print(f"[surgattn-attrib] {note}", flush=True)

    if args.mode == "analyze":
        analyze(wandb_log=not args.no_wandb)
        return 0

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    smoke = args.mode == "smoke"
    for arm in arms:
        run_arm(
            arm,
            num_prompts=args.smoke_num_prompts if smoke else args.num_prompts,
            output_len=args.smoke_output_len if smoke else args.output_len,
            tps_reps=args.tps_reps,
            decode_tokens=args.decode_tokens,
            smoke=smoke,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
