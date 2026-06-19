#!/usr/bin/env python
"""PR #744 lawine — Recover the publishable-drafter penalty via a higher-acceptance
stock drafter K-sweep.

#734 (W&B ka31xa6v) put the STOCK-HUB **publishable** drafter
(``google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant``, an MTP assistant) K=5
ceiling at [201.31 floor / 231.80 anchored] official-equiv TPS — 12.7% BELOW the
non-publishable ``/tmp/qat-assistant`` drafter (#728 K=5 = [.. / 265.45]),
because the stock publishable drafter accepts less. This sweep re-optimises the
publishable drafter's K against its (lower) per-position acceptance curve to
recover the penalty on a path with lower private-reproduction (G1) risk.

It is a thin #744 driver on top of the #728 harness (``run_sweep.py``): same
DEPLOYED serve on faithful vLLM 0.22.0, BI=1, conc=1, free-run greedy decode of
the official 128x512 set, strict + tau=0.3-rescued self-consistency vs the
config's OWN served-AR reference. It ADDS, per spec K:

  * acceptance    : vLLM ``SpecDecoding metrics`` parsed from the server log —
                    overall draft-acceptance rate (exact: sum_accepted/sum_drafted),
                    mean acceptance length E[T]=1+K*rate, and the per-position
                    acceptance curve (the thing K is optimised against).
  * official-equiv BRACKET [floor, anchored] inline (the #728 augment_report model):
        floor    = wall_tps * TAU_LO(1.0352)            (conservative; carries BI=1 tax)
        anchored = wall_tps * (126.378 / ar_base_local)  (base-anchored; ship-no-BI)
  * tau=0.3 onset HEADROOM: 0.3 - max(surviving onset gap), i.e. how close the
    worst rescued (non-confident) onset flip sits to the confident-miss threshold.
    #734 K=5 reported only 0.05-nat headroom — report it per variant so we see if
    any config erodes the self-consistency margin.

The AR M=1 reference (drafter OFF, SENPAI_REFERENCE_MODE=1) is DRAFTER-INDEPENDENT
(reference mode forces num_speculative_tokens=0 -> the drafter is never loaded), so
by default we REUSE the completed #728 ``runs/sweep`` reference (ar_base_local=106.02,
ppl=2.0189, 128/128) rather than re-pay the ~30-min boot+decode+teacher-force+PPL.
``--fresh-ref`` re-measures it.

LOCAL ONLY: analysis_only=1, official_tps=0, single A10G, NO HF Job / no --launch /
no submission change. Group: publishable-drafter-accept-recover.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Reuse the #728 harness wholesale (serve, decode, gate, onset-rescue, env).
import run_sweep as rs  # noqa: E402
from scripts.local_validation import greedy_gate, harness, paths  # noqa: E402

PUB_DRAFTER = "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant"
ANCHOR_TPS = rs.ANCHOR_TPS          # 126.378 locked int4 official non-spec anchor (the bar)
TAU_LO = rs.TAU_LO                  # 1.0352 banked local->official flat scalar (floor)
TAU_GATE = rs.TAU_GATE              # 0.3 nat self-consistency confident-miss threshold
PPL_GATE = rs.PPL_GATE              # 2.42
# #734 publishable K=5 baseline (the number to recover toward / past):
BASELINE_PUB_K5_BRACKET = [201.31, 231.80]
# #728 non-publishable K=5 anchored ceiling (the 12.7%-better target):
NONPUB_K5_ANCHORED = 265.45


# --------------------------------------------------------------------------- acceptance metrics
_SPEC_RE = re.compile(
    r"Mean acceptance length:\s*([\d.]+).*?"
    r"Accepted:\s*(\d+)\s*tokens.*?"
    r"Drafted:\s*(\d+)\s*tokens.*?"
    r"Per-position acceptance rate:\s*([0-9.,\s]+?),\s*Avg Draft acceptance rate:\s*([\d.]+)%"
)


def parse_spec_metrics(log_path: Path, k: int) -> dict[str, Any]:
    """Aggregate vLLM ``SpecDecoding metrics`` windows from the server log.

    Each logged line is a per-interval window (NOT cumulative), so the exact
    run-aggregate draft-acceptance rate is sum(Accepted)/sum(Drafted). The
    speedup-relevant mean acceptance length is E[T] = 1 + K*rate (1 bonus token +
    K*rate accepted draft tokens per target-verify step). The per-position curve
    is averaged element-wise across windows (each window reports K rates)."""
    text = Path(log_path).read_text(errors="replace") if Path(log_path).exists() else ""
    sum_acc = sum_draft = 0
    logged_mal: list[float] = []
    perpos_acc: list[list[float]] = []
    windows = 0
    for m in _SPEC_RE.finditer(text):
        mal = float(m.group(1)); acc = int(m.group(2)); draft = int(m.group(3))
        perpos = [float(x) for x in m.group(4).replace(" ", "").split(",") if x != ""]
        rate_pct = float(m.group(5))
        sum_acc += acc; sum_draft += draft
        logged_mal.append(mal)
        if perpos:
            perpos_acc.append(perpos)
        windows += 1
        _ = rate_pct
    overall_rate = (sum_acc / sum_draft) if sum_draft else None
    et = (1.0 + k * overall_rate) if overall_rate is not None else None
    # element-wise mean of per-position curves (guard ragged windows -> use modal len k)
    perpos_mean: list[float] | None = None
    if perpos_acc:
        width = k
        cols: list[list[float]] = [[] for _ in range(width)]
        for row in perpos_acc:
            for i in range(min(width, len(row))):
                cols[i].append(row[i])
        perpos_mean = [round(sum(c) / len(c), 4) if c else None for c in cols]  # type: ignore[misc]
    return {
        "windows": windows,
        "sum_accepted": sum_acc,
        "sum_drafted": sum_draft,
        "avg_draft_acceptance_rate": round(overall_rate, 5) if overall_rate is not None else None,
        "mean_acceptance_length_ET": round(et, 4) if et is not None else None,
        "mean_acceptance_length_logged": (round(sum(logged_mal) / len(logged_mal), 4)
                                          if logged_mal else None),
        "per_position_acceptance": perpos_mean,
    }


# --------------------------------------------------------------------------- tau=0.3 headroom
def tau03_headroom(rescue: dict[str, Any]) -> dict[str, Any]:
    """Margin to the confident-miss threshold. A SURVIVING (rescued) flip is a
    divergent onset that is NOT a confident genuine flip at tau=0.3: gap<=0.3 and
    inside the base top-k. Headroom = 0.3 - max(surviving gap); small headroom =>
    a near-confident flip => fragile self-consistency."""
    per = rescue.get("per_prompt", [])
    surviving = [p for p in per
                 if p.get("divergent") and isinstance(p.get("gap"), (int, float))
                 and not p.get("outside_topk") and p["gap"] <= TAU_GATE]
    worst = max((p["gap"] for p in surviving), default=0.0)
    cgf = rescue.get("confident_genuine_flips_at_gate")
    return {
        "confident_genuine_flips_at_0.3": cgf,
        "self_consistent_tau03": (cgf == 0),
        "num_surviving_flips": len(surviving),
        "worst_surviving_onset_gap": round(worst, 4),
        "headroom_nat": round(TAU_GATE - worst, 4),
        "num_onset_outside_topk": rescue.get("num_onset_outside_topk"),
    }


# --------------------------------------------------------------------------- reference (reuse or fresh)
def load_reused_reference(ref_dir: Path, run_dir: Path) -> dict[str, Any]:
    """Load the drafter-independent AR reference (decode jsonl + base_dist + info)
    from a completed #728 sweep dir and copy the decode jsonl into our run dir so
    greedy_gate.compare can read it. base_dist stays in memory only."""
    ref_jsonl = ref_dir / "ref.jsonl"
    bd_file = ref_dir / "ref.base_dist.json"
    info_file = ref_dir / "ref.info.json"
    for f in (ref_jsonl, bd_file, info_file):
        if not f.exists():
            raise FileNotFoundError(f"reusable reference missing {f}; pass --fresh-ref to re-measure")
    info = json.loads(info_file.read_text())
    # sanity: the reused ref MUST be a drafter-OFF reference-mode capture on BI=1
    env = info.get("extra_env", {})
    if env.get("SENPAI_REFERENCE_MODE") != "1" or str(env.get("NUM_SPECULATIVE_TOKENS")) not in ("0",):
        raise ValueError(f"reused ref {info_file} is not a drafter-OFF reference (env={env})")
    if env.get("VLLM_BATCH_INVARIANT") != "1":
        raise ValueError(f"reused ref {info_file} not BI=1 (env={env})")
    base_dist = {int(k): v for k, v in json.loads(bd_file.read_text()).items()}
    (run_dir / "ref.jsonl").write_text(ref_jsonl.read_text())
    (run_dir / "ref.info.json").write_text(json.dumps(info, indent=2))
    return {"info": info, "base_dist": base_dist, "wall_tps": info.get("wall_tps"),
            "ppl": info.get("ppl"), "num_records": info.get("num_records"),
            "reused_from": str(ref_dir)}


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ks", default="4,5,6,7", help="comma list of K (num_speculative_tokens)")
    ap.add_argument("--drafter", default=PUB_DRAFTER, help="publishable Hub drafter id")
    ap.add_argument("--model-id", default=rs.MODEL_DIR)
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--ppl-ks", default="", help="comma Ks to also PPL (default: the fastest complete K)")
    ap.add_argument("--ref-dir", type=Path, default=HERE / "runs" / "sweep",
                    help="completed #728 sweep dir to reuse the drafter-independent AR ref from")
    ap.add_argument("--fresh-ref", action="store_true", help="re-measure the AR reference instead of reusing")
    ap.add_argument("--cand-port", type=int, default=8022)
    ap.add_argument("--ref-port", type=int, default=8021)
    ap.add_argument("--out-dir", type=Path, default=HERE / "runs")
    ap.add_argument("--label", default="pubdrafter")
    ap.add_argument("--smoke", action="store_true", help="4 prompts, K=6 only, no PPL — wiring check")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-name", default="lawine/publishable-drafter-accept-recover")
    ap.add_argument("--wandb-group", default="publishable-drafter-accept-recover")
    args = ap.parse_args()

    ks = [6] if args.smoke else [int(x) for x in args.ks.split(",") if x.strip()]
    num_prompts = 4 if args.smoke else args.num_prompts
    do_ppl = not args.smoke
    ppl_ks = {int(x) for x in args.ppl_ks.split(",") if x.strip()}
    run_dir = args.out_dir / (f"{args.label}_smoke" if args.smoke else args.label)
    run_dir.mkdir(parents=True, exist_ok=True)

    for note in paths.prepare_local_gpu_env():
        print(f"[pub] {note}", flush=True)

    manifest = harness.load_manifest(rs.SUBMISSION)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    vllm_ver = harness._dist_version(server_python, "vllm")
    tf_ver = harness._dist_version(server_python, "transformers")
    print(f"[pub] server_python={server_python} vllm={vllm_ver} transformers={tf_ver}", flush=True)
    if vllm_ver != "0.22.0":
        print(f"[pub] WARNING: expected faithful vllm 0.22.0, got {vllm_ver}", flush=True)

    base = rs.base_env(args.model_id, args.drafter, batch_invariant=1)

    # ---- AR M=1 reference (drafter-independent) ----
    if args.fresh_ref or args.smoke:
        ref_env = {**base, "SENPAI_REFERENCE_MODE": "1", "NUM_SPECULATIVE_TOKENS": "0"}
        print("[pub] === AR M=1 REFERENCE (drafter OFF, BI=1): decode + PPL + base dist ===", flush=True)
        ref_info = rs.serve_capture(
            rs.SUBMISSION, server_python, label="ref", run_dir=run_dir, extra_env=ref_env,
            port=args.ref_port, num_prompts=num_prompts, output_len=args.output_len,
            do_ppl=do_ppl, do_logprobs=True, ref_recs=None, startup_timeout_s=1800,
        )
        base_dist = ref_info.get("_base_dist") or {int(k): v for k, v in json.loads(
            (run_dir / "ref.base_dist.json").read_text()).items()}
        ar_local = ref_info.get("wall_tps")
        ar_ppl = ref_info.get("ppl")
        ar_records = ref_info.get("num_records")
        ref_provenance = "fresh"
    else:
        print(f"[pub] === REUSING drafter-independent AR reference from {args.ref_dir} ===", flush=True)
        reused = load_reused_reference(args.ref_dir, run_dir)
        base_dist = reused["base_dist"]
        ar_local = reused["wall_tps"]; ar_ppl = reused["ppl"]; ar_records = reused["num_records"]
        ref_provenance = f"reused:{reused['reused_from']}"
    ref_file = run_dir / "ref.jsonl"
    ref_rows = rs.load_decode_jsonl(ref_file)
    ratio_anchored = ANCHOR_TPS / ar_local if ar_local else None
    print(f"[pub] AR ref: wall_tps={ar_local} ppl={ar_ppl} records={ar_records}/{num_prompts} "
          f"ratio_anchored={ratio_anchored:.5f} ({ref_provenance})"
          if ratio_anchored else f"[pub] AR ref incomplete ({ref_provenance})", flush=True)

    # default PPL target: cheapest informative — do it on every K in smoke=off unless overridden
    if not ppl_ks and do_ppl:
        ppl_ks = set(ks)  # PPL is drafter-independent in greedy spec (== AR base), but measure to prove it

    # ---- spec candidates ----
    results: list[dict[str, Any]] = []
    for K in ks:
        cand_env = {**base, "NUM_SPECULATIVE_TOKENS": str(K)}
        label = f"spec_k{K}"
        print(f"[pub] === SPEC K={K} (publishable drafter {args.drafter}, BI=1) ===", flush=True)
        cand_info = rs.serve_capture(
            rs.SUBMISSION, server_python, label=label, run_dir=run_dir, extra_env=cand_env,
            port=args.cand_port, num_prompts=num_prompts, output_len=args.output_len,
            do_ppl=(do_ppl and K in ppl_ks), do_logprobs=False, ref_recs=None,
            startup_timeout_s=1800,
        )
        cand_file = run_dir / f"{label}.jsonl"
        cand_rows = rs.load_decode_jsonl(cand_file)

        report = greedy_gate.compare(ref_file, cand_file)
        onset = greedy_gate.onset_summary(report)
        n_cmp = report.num_prompts_compared or 1
        seq_exact = report.num_identical / n_cmp
        tok_total = report.total_tokens_compared or 1
        tok_id = 1.0 - report.total_divergent_tokens / tok_total

        rescue = rs.classify_onsets(ref_rows, cand_rows, base_dist)
        headroom = tau03_headroom(rescue)
        accept = parse_spec_metrics(Path(cand_info.get("server_log") or (run_dir / f"{label}.server.log")), K)

        wt = cand_info.get("wall_tps")
        floor = wt * TAU_LO if isinstance(wt, (int, float)) else None
        anchored = wt * ratio_anchored if isinstance(wt, (int, float)) and ratio_anchored else None
        records = cand_info.get("num_records")
        comp_tokens = cand_info.get("num_completion_tokens")
        complete = (records == num_prompts) and (comp_tokens == num_prompts * args.output_len)

        res = {
            "k": K,
            "wall_tps_local": wt,
            "official_equiv_floor": round(floor, 2) if floor else None,
            "official_equiv_anchored": round(anchored, 2) if anchored else None,
            "official_equiv_bracket": [round(floor, 2), round(anchored, 2)] if floor and anchored else None,
            "beats_anchor_126": (anchored > ANCHOR_TPS) if anchored else None,
            "beats_floor_126": (floor > ANCHOR_TPS) if floor else None,
            "acceptance": accept,
            "records": records, "completion_tokens": comp_tokens, "complete_128_128": complete,
            "ppl": cand_info.get("ppl"),
            "ppl_ok": (cand_info.get("ppl") <= PPL_GATE) if isinstance(cand_info.get("ppl"), (int, float)) else None,
            "strict_verdict": report.verdict,
            "strict_seq_exact": seq_exact,
            "strict_token_identity": tok_id,
            "strict_num_divergent": report.num_divergent,
            "onset": {k: onset.get(k) for k in ("onset_min", "onset_median", "onset_max", "num_divergent")},
            "tau03": headroom,
            "self_consistent_tau03": headroom["self_consistent_tau03"],
            "rescue": {k: v for k, v in rescue.items() if k != "per_prompt"},
            "peak_vram_gb": cand_info.get("peak_vram_gb"),
            "serve_ready_s": cand_info.get("serve_ready_s"),
            "server_log": cand_info.get("server_log"),
        }
        (run_dir / f"{label}.rescue.json").write_text(json.dumps(rescue, indent=2, default=str))
        results.append(res)

        et = accept.get("mean_acceptance_length_ET")
        print(f"[pub] K={K}: wall_tps={wt:.2f} bracket=[{res['official_equiv_floor']} / "
              f"{res['official_equiv_anchored']}] | accept_rate={accept.get('avg_draft_acceptance_rate')} "
              f"E[T]={et} | strict={report.verdict} seq_exact={seq_exact:.4f} | "
              f"cgf@0.3={headroom['confident_genuine_flips_at_0.3']} headroom={headroom['headroom_nat']} "
              f"self_consistent={res['self_consistent_tau03']} | ppl={res['ppl']} 128/128={complete}"
              if isinstance(wt, (int, float)) else f"[pub] K={K}: incomplete", flush=True)

    # ---- pick fastest self-consistent + best recovery vs #734 ----
    sc = [r for r in results if r.get("self_consistent_tau03") and r.get("complete_128_128")]
    fastest_sc = max(sc, key=lambda r: r["wall_tps_local"]) if sc else None
    fastest_any = max((r for r in results if isinstance(r.get("wall_tps_local"), (int, float))),
                      key=lambda r: r["wall_tps_local"], default=None)

    report_obj = {
        "pr": 744,
        "analysis_only": True,
        "official_tps": 0,
        "smoke": args.smoke,
        "config": {
            "model_dir": args.model_id, "drafter": args.drafter, "drafter_publishable": True, "ks": ks,
            "vllm_version": vllm_ver, "transformers_version": tf_ver,
            "batch_invariant": 1, "max_num_seqs": 1, "max_num_batched_tokens": 512,
            "num_prompts": num_prompts, "output_len": args.output_len, "seed": paths.SEED,
            "tau_gate_nats": TAU_GATE, "tau_lo_floor": TAU_LO, "anchor_tps": ANCHOR_TPS,
            "ppl_gate": PPL_GATE, "ref_provenance": ref_provenance,
            "method": "publishable stock-Hub MTP drafter K-sweep; DEPLOYED serve free-run greedy vs "
                      "config's OWN served-AR (drafter OFF); wall_tps=tokens/decode_s; official-equiv "
                      "bracket [wall*1.0352 floor, wall*(126.378/ar_local) anchored]; acceptance from "
                      "vLLM SpecDecoding metrics; strict greedy_gate + tau=0.3 onset-gap headroom.",
        },
        "transfer_model": {"ar_base_local_bi1": ar_local, "anchor_official": ANCHOR_TPS,
                           "ratio_anchored": ratio_anchored, "tau_lo_floor": TAU_LO},
        "baselines": {"pub_k5_734_bracket": BASELINE_PUB_K5_BRACKET,
                      "nonpub_k5_728_anchored": NONPUB_K5_ANCHORED, "anchor_bar": ANCHOR_TPS},
        "ar_reference": {"wall_tps_local": ar_local, "ppl": ar_ppl, "records": ar_records,
                         "provenance": ref_provenance},
        "results": results,
        "fastest_self_consistent": fastest_sc,
        "fastest_any": fastest_any,
    }
    out = run_dir / ("report.smoke.json" if args.smoke else "report.json")
    out.write_text(json.dumps(report_obj, indent=2, default=str))

    print("\n" + "=" * 80, flush=True)
    print(f"[PR744] publishable-drafter accept-recover ({'SMOKE' if args.smoke else 'FULL'}) "
          f"vllm={vllm_ver} BI=1 conc=1 {num_prompts}x{args.output_len} drafter={args.drafter}", flush=True)
    print(f"  AR ref wall_tps={ar_local} ppl={ar_ppl} ratio_anchored={ratio_anchored}", flush=True)
    print(f"  #734 pub K=5 bracket={BASELINE_PUB_K5_BRACKET} | #728 nonpub K=5 anchored={NONPUB_K5_ANCHORED} "
          f"| bar={ANCHOR_TPS}", flush=True)
    for r in results:
        a = r["acceptance"]
        print(f"  K={r['k']:>1} | wall={r['wall_tps_local']} bracket=[{r['official_equiv_floor']} / "
              f"{r['official_equiv_anchored']}] | rate={a.get('avg_draft_acceptance_rate')} "
              f"E[T]={a.get('mean_acceptance_length_ET')} | self_consistent={r['self_consistent_tau03']} "
              f"headroom={r['tau03']['headroom_nat']} | ppl={r['ppl']} 128/128={r['complete_128_128']}", flush=True)
    if fastest_sc:
        rec_vs_734 = (fastest_sc["official_equiv_anchored"] - BASELINE_PUB_K5_BRACKET[1]) if fastest_sc.get("official_equiv_anchored") else None
        print(f"  >>> FASTEST SELF-CONSISTENT: K={fastest_sc['k']} bracket=[{fastest_sc['official_equiv_floor']} / "
              f"{fastest_sc['official_equiv_anchored']}] anchored Δvs#734_pubK5={rec_vs_734:+.2f}"
              if rec_vs_734 is not None else f"  >>> FASTEST SELF-CONSISTENT: K={fastest_sc['k']}", flush=True)
    else:
        print("  >>> NO self-consistent config at tau=0.3", flush=True)
    print(f"  report -> {out}", flush=True)
    print("=" * 80, flush=True)

    if not args.no_wandb and not args.smoke:
        try:
            _log_wandb(report_obj, name=args.wandb_name, group=args.wandb_group)
        except Exception as exc:  # noqa: BLE001
            print(f"[pub] WARNING: wandb logging failed ({type(exc).__name__}: {exc}); "
                  f"report preserved at {out}.", flush=True)
    return 0


def _log_wandb(report: dict[str, Any], *, name: str, group: str) -> None:
    try:
        from scripts import wandb_logging as wl
    except ImportError:
        print("[pub] wandb_logging unavailable — skipping", flush=True)
        return
    run = wl.init_wandb_run(
        job_type="publishable-drafter-accept-recover", agent="lawine", name=name, group=group,
        notes="PR744 publishable MTP drafter K-sweep: acceptance-raise to recover the 12.7% penalty",
        tags=["pr744", "specdec", "publishable-drafter", "acceptance", "k-sweep", "self-consistency"],
        config=report["config"],
    )
    if run is None:
        print("[pub] wandb not configured — skipping", flush=True)
        return
    for r in report["results"]:
        a = r.get("acceptance", {})
        m = {
            f"k{r['k']}/wall_tps_local": r.get("wall_tps_local"),
            f"k{r['k']}/official_equiv_floor": r.get("official_equiv_floor"),
            f"k{r['k']}/official_equiv_anchored": r.get("official_equiv_anchored"),
            f"k{r['k']}/avg_draft_acceptance_rate": a.get("avg_draft_acceptance_rate"),
            f"k{r['k']}/mean_acceptance_length_ET": a.get("mean_acceptance_length_ET"),
            f"k{r['k']}/strict_seq_exact": r.get("strict_seq_exact"),
            f"k{r['k']}/confident_genuine_flips_0.3": r["tau03"].get("confident_genuine_flips_at_0.3"),
            f"k{r['k']}/headroom_nat": r["tau03"].get("headroom_nat"),
            f"k{r['k']}/self_consistent": 1 if r.get("self_consistent_tau03") else 0,
            f"k{r['k']}/ppl": r.get("ppl") if isinstance(r.get("ppl"), (int, float)) else None,
            f"k{r['k']}/complete_128": 1 if r.get("complete_128_128") else 0,
        }
        run.summary.update({k: v for k, v in m.items() if v is not None})
    fsc = report.get("fastest_self_consistent")
    run.summary["fastest_self_consistent_k"] = fsc["k"] if fsc else None
    run.summary["fastest_self_consistent_official_equiv_anchored"] = (
        fsc.get("official_equiv_anchored") if fsc else None)
    run.summary["ar_base_local_bi1"] = report["transfer_model"]["ar_base_local_bi1"]
    run.summary["ratio_anchored"] = report["transfer_model"]["ratio_anchored"]
    run.summary["analysis_only"] = True
    run.summary["official_tps"] = 0
    wl.log_json_artifact(run, name="pr744_pubdrafter_report", artifact_type="accept-recover", data=report)
    wl.finish_wandb(run)
    print("[pub] wandb logged", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
