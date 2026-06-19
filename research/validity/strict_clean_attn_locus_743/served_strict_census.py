#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""PR #755 lawine -- SERVED strict identity census under the num_splits=1 force
(instruction 2) + an enforce_eager LOCALIZATION arm.

The #755 num_splits PROBE (served_numsplits_probe.py, run 2026-06-19) REFUTED the
hypothesis premise on the served path: under ``VLLM_BATCH_INVARIANT=1`` the live
EngineCore worker reports ``is_batch_invariant=True`` and EVERY attention forward --
the M=1 decode, the M=K+1 verify, AND prefill -- runs ``use_3d=0, num_segments=1``
(the 2D one-shot reduction). num_splits is ALREADY 1 for both decode and verify, so
the verify/decode reduction order is NOT split apart. wirbel #747's "BI=1 verify is
already single-pass" therefore DOES transfer to the served 512-token run; the
hypothesis that the served verify-attention sits on adaptive ``num_splits>1`` is
false.

This driver runs ONE arm per invocation:

  arm=force_ns1
      Serve the publishable-K4-BI1 config with ``SENPAI_FORCE_NUMSPLITS1=1`` active
      (served_numsplits_force pins the kernel global True) PLUS the read-only probe
      (so a single server.log proves the force fired AND every bucket is nseg=1),
      then re-run the strict self-consistency census vs the config's OWN served-AR.
      Because the force pins a global that is already True, this is EXPECTED to be
      byte-identical to #752 (strict 24/128). MEASURING it turns the deliverable
      ``served_numsplits1_strict_identity`` from "inferred no-op" into "measured
      no-op", and its anchored/wall TPS give the (≈zero) byte-exactness tax.

  arm=eager
      Toggle ``ENFORCE_EAGER=1`` (BI=1 held constant) on BOTH the AR reference and
      the K=4 spec candidate, to LOCALIZE the residual source of #752's 24/128.
      land #743 / wirbel #747's byte-exact results were measured offline under
      ``enforce_eager=True`` (no CUDA graphs). If eager collapses the served census
      to (near) all-identical, then CUDA-graph batch-size-keyed capture (the bs=1
      decode graph vs the bs=K+1 verify graph picking different kernel tilings /
      reduction trees) is the residual divergence source -- and ``enforce_eager`` is
      the served lever to literal-strict at a TPS cost. If eager STAYS ~24/128, the
      divergence is a deeper M-dependence in the served kernels (not the split, not
      the matmul family, not CUDA graphs).

Reuses the #728/#744 harness wholesale: rs.serve_capture / rs.base_env /
rs.classify_onsets / rs.load_decode_jsonl, greedy_gate.compare, and
run_pubdrafter_sweep.parse_spec_metrics / tau03_headroom / load_reused_reference.

LOCAL A10G only. analysis_only=1, official_tps=0. NO HF Job / no --launch.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
SPEC_DIR = ROOT / "research" / "spec_achievable_ceiling"
for p in (str(SPEC_DIR), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_sweep as rs  # noqa: E402
import run_pubdrafter_sweep as rp  # noqa: E402
from run_pubdrafter_sweep import PUB_DRAFTER  # noqa: E402
from scripts.local_validation import greedy_gate, harness, paths  # noqa: E402

ANCHOR_TPS = rs.ANCHOR_TPS    # 126.378
TAU_LO = rs.TAU_LO            # 1.0352
PPL_GATE = rs.PPL_GATE        # 2.42
# #752 baseline (the publishable-K4-BI1 rung this PR is trying to make literal-strict)
BASE_752 = {
    "wall_tps": 198.00157247638245,
    "anchored": 236.02,
    "floor": 204.97,
    "strict_seq_exact": 0.1875,   # 24/128
    "strict_num_divergent": 104,
    "ppl": 2.0189025477916,
    "ar_local_bi1": 106.02275748221821,
    "et": 3.0332,
}


def census(ref_file: Path, cand_file: Path, ref_rows, cand_rows, base_dist) -> dict[str, Any]:
    report = greedy_gate.compare(ref_file, cand_file)
    onset = greedy_gate.onset_summary(report)
    n_cmp = report.num_prompts_compared or 1
    seq_exact = report.num_identical / n_cmp
    tok_total = report.total_tokens_compared or 1
    tok_id = 1.0 - report.total_divergent_tokens / tok_total
    rescue = rs.classify_onsets(ref_rows, cand_rows, base_dist) if base_dist is not None else {}
    headroom = rp.tau03_headroom(rescue) if rescue else {}
    return {
        "verdict": report.verdict,
        "num_prompts_compared": report.num_prompts_compared,
        "num_identical": report.num_identical,
        "num_divergent": report.num_divergent,
        "seq_exact": seq_exact,
        "token_identity": tok_id,
        "total_tokens_compared": report.total_tokens_compared,
        "total_divergent_tokens": report.total_divergent_tokens,
        "onset": {k: onset.get(k) for k in ("onset_min", "onset_median", "onset_max", "num_divergent")},
        "tau03": headroom,
        "self_consistent_tau03": headroom.get("self_consistent_tau03"),
        "rescue": {k: v for k, v in rescue.items() if k != "per_prompt"} if rescue else {},
    }


def resolve_reference(arm: str, *, server_python, run_dir, base, ref_dir,
                      num_prompts, output_len, do_ppl, ref_port) -> dict[str, Any]:
    """force_ns1 reuses the drafter-independent BI=1 AR ref (the force is a no-op on
    AR too, so it is byte-identical); eager re-measures a FRESH enforce_eager AR ref
    (the config's OWN AR must share the eager serving path for self-consistency)."""
    if arm == "force_ns1":
        print(f"[census] === REUSING drafter-independent BI=1 AR reference from {ref_dir} ===", flush=True)
        reused = rp.load_reused_reference(ref_dir, run_dir)
        return {"base_dist": reused["base_dist"], "ar_local": reused["wall_tps"],
                "ar_ppl": reused["ppl"], "ar_records": reused["num_records"],
                "provenance": f"reused:{reused['reused_from']}",
                "ref_file": run_dir / "ref.jsonl"}
    # eager: fresh enforce_eager AR reference (drafter OFF), with base dist for the rescue
    ref_env = {**base, "ENFORCE_EAGER": "1",
               "SENPAI_REFERENCE_MODE": "1", "NUM_SPECULATIVE_TOKENS": "0"}
    print("[census] === FRESH enforce_eager AR M=1 REFERENCE (drafter OFF, BI=1, eager) ===", flush=True)
    ref_info = rs.serve_capture(
        rs.SUBMISSION, server_python, label="ref", run_dir=run_dir, extra_env=ref_env,
        port=ref_port, num_prompts=num_prompts, output_len=output_len,
        do_ppl=do_ppl, do_logprobs=True, ref_recs=None, startup_timeout_s=1800,
    )
    base_dist = ref_info.get("_base_dist") or {
        int(k): v for k, v in json.loads((run_dir / "ref.base_dist.json").read_text()).items()}
    return {"base_dist": base_dist, "ar_local": ref_info.get("wall_tps"),
            "ar_ppl": ref_info.get("ppl"), "ar_records": ref_info.get("num_records"),
            "provenance": "fresh:enforce_eager", "ref_file": run_dir / "ref.jsonl"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["force_ns1", "eager"], required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--drafter", default=PUB_DRAFTER)
    ap.add_argument("--model-id", default=rs.MODEL_DIR)
    ap.add_argument("--ref-dir", type=Path, default=SPEC_DIR / "runs" / "sweep",
                    help="completed sweep dir to reuse the BI=1 AR ref from (force_ns1 arm)")
    ap.add_argument("--out-dir", type=Path, default=HERE / "runs")
    ap.add_argument("--cand-port", type=int, default=8042)
    ap.add_argument("--ref-port", type=int, default=8041)
    ap.add_argument("--no-ppl", action="store_true")
    args = ap.parse_args()

    run_dir = args.out_dir / f"strict_census_{args.arm}"
    run_dir.mkdir(parents=True, exist_ok=True)
    do_ppl = not args.no_ppl
    t0 = time.time()

    for note in paths.prepare_local_gpu_env():
        print(f"[census] {note}", flush=True)

    manifest = harness.load_manifest(rs.SUBMISSION)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    vllm_ver = harness._dist_version(server_python, "vllm")
    print(f"[census] arm={args.arm} server_python={server_python} vllm={vllm_ver}", flush=True)

    base = rs.base_env(args.model_id, args.drafter, batch_invariant=1)

    ref = resolve_reference(
        args.arm, server_python=server_python, run_dir=run_dir, base=base,
        ref_dir=args.ref_dir, num_prompts=args.num_prompts, output_len=args.output_len,
        do_ppl=do_ppl, ref_port=args.ref_port)
    ar_local = ref["ar_local"]
    ratio_anchored = (ANCHOR_TPS / ar_local) if ar_local else None
    ref_file = ref["ref_file"]
    ref_rows = rs.load_decode_jsonl(ref_file)
    print(f"[census] AR ref: wall_tps={ar_local} ppl={ref['ar_ppl']} records={ref['ar_records']} "
          f"ratio_anchored={ratio_anchored} ({ref['provenance']})", flush=True)

    # ---- spec candidate under the arm's serving lever ----
    if args.arm == "force_ns1":
        cand_env = {**base, "NUM_SPECULATIVE_TOKENS": str(args.k),
                    "SENPAI_PR755_DIR": str(HERE),
                    "SENPAI_FORCE_NUMSPLITS1": "1",
                    # read-only probe in the SAME run: server.log proves force fired AND nseg=1
                    "SENPAI_NUMSPLITS_PROBE": str(run_dir / f"spec_k{args.k}.numsplits.json")}
    else:  # eager
        cand_env = {**base, "NUM_SPECULATIVE_TOKENS": str(args.k), "ENFORCE_EAGER": "1"}

    label = f"spec_k{args.k}"
    print(f"[census] === SPEC K={args.k} arm={args.arm} (BI=1, drafter={args.drafter}) ===", flush=True)
    cand_info = rs.serve_capture(
        rs.SUBMISSION, server_python, label=label, run_dir=run_dir, extra_env=cand_env,
        port=args.cand_port, num_prompts=args.num_prompts, output_len=args.output_len,
        do_ppl=do_ppl, do_logprobs=False, ref_recs=None, startup_timeout_s=1800,
    )
    cand_file = run_dir / f"{label}.jsonl"
    cand_rows = rs.load_decode_jsonl(cand_file)

    cen = census(ref_file, cand_file, ref_rows, cand_rows, ref["base_dist"])
    accept = rp.parse_spec_metrics(
        Path(cand_info.get("server_log") or (run_dir / f"{label}.server.log")), args.k)

    wt = cand_info.get("wall_tps")
    floor = wt * TAU_LO if isinstance(wt, (int, float)) else None
    anchored = wt * ratio_anchored if isinstance(wt, (int, float)) and ratio_anchored else None
    records = cand_info.get("num_records")
    comp_tokens = cand_info.get("num_completion_tokens")
    complete = (records == args.num_prompts) and (comp_tokens == args.num_prompts * args.output_len)
    ppl = cand_info.get("ppl")

    report = {
        "pr": 755, "arm": args.arm, "analysis_only": True, "official_tps": 0,
        "vllm_version": vllm_ver, "k": args.k,
        "config": {
            "model_dir": args.model_id, "drafter": args.drafter, "batch_invariant": 1,
            "enforce_eager": args.arm == "eager", "force_numsplits1": args.arm == "force_ns1",
            "max_num_seqs": 1, "max_num_batched_tokens": 512,
            "num_prompts": args.num_prompts, "output_len": args.output_len, "seed": paths.SEED,
            "anchor_tps": ANCHOR_TPS, "tau_lo_floor": TAU_LO, "ppl_gate": PPL_GATE,
            "ref_provenance": ref["provenance"],
        },
        "baseline_752": BASE_752,
        "transfer_model": {"ar_local_bi1": ar_local, "ratio_anchored": ratio_anchored},
        "ar_reference": {"wall_tps_local": ar_local, "ppl": ref["ar_ppl"], "records": ref["ar_records"]},
        "result": {
            "k": args.k,
            "wall_tps_local": wt,
            "official_equiv_floor": round(floor, 2) if floor else None,
            "official_equiv_anchored": round(anchored, 2) if anchored else None,
            "official_equiv_bracket": [round(floor, 2), round(anchored, 2)] if floor and anchored else None,
            "beats_anchor_126": (anchored > ANCHOR_TPS) if anchored else None,
            "strict_verdict": cen["verdict"],
            "strict_seq_exact": cen["seq_exact"],
            "strict_token_identity": cen["token_identity"],
            "strict_num_divergent": cen["num_divergent"],
            "strict_num_identical": cen["num_identical"],
            "strict_num_prompts_compared": cen["num_prompts_compared"],
            "onset": cen["onset"],
            "tau03": cen["tau03"],
            "self_consistent_tau03": cen["self_consistent_tau03"],
            "rescue": cen["rescue"],
            "acceptance": accept,
            "ppl": ppl,
            "ppl_ok": (ppl <= PPL_GATE) if isinstance(ppl, (int, float)) else None,
            "records": records, "completion_tokens": comp_tokens, "complete_128_128": complete,
            "peak_vram_gb": cand_info.get("peak_vram_gb"),
            "serve_ready_s": cand_info.get("serve_ready_s"),
        },
        "tax_vs_752": {
            "d_anchored": round(anchored - BASE_752["anchored"], 2) if anchored else None,
            "d_wall": round(wt - BASE_752["wall_tps"], 2) if isinstance(wt, (int, float)) else None,
            "pct_anchored": (round(100 * (anchored - BASE_752["anchored"]) / BASE_752["anchored"], 2)
                             if anchored else None),
            "d_strict_seq_exact": round(cen["seq_exact"] - BASE_752["strict_seq_exact"], 4),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    out = run_dir / "report.json"
    out.write_text(json.dumps(report, indent=2, default=str))

    r = report["result"]
    print("\n" + "=" * 80, flush=True)
    print(f"[PR755 STRICT CENSUS arm={args.arm}] vllm={vllm_ver} BI=1 K={args.k} "
          f"{args.num_prompts}x{args.output_len}", flush=True)
    print(f"  AR ref wall={ar_local} ppl={ref['ar_ppl']} ratio_anchored={ratio_anchored} ({ref['provenance']})",
          flush=True)
    print(f"  spec wall={wt} bracket=[{r['official_equiv_floor']} / {r['official_equiv_anchored']}] "
          f"E[T]={accept.get('mean_acceptance_length_ET')} rate={accept.get('avg_draft_acceptance_rate')}",
          flush=True)
    print(f"  STRICT: verdict={r['strict_verdict']} seq_exact={r['strict_seq_exact']:.4f} "
          f"({r['strict_num_identical']}/{r['strict_num_prompts_compared']}) "
          f"token_id={r['strict_token_identity']:.4f} num_div={r['strict_num_divergent']}", flush=True)
    print(f"  tau=0.3 self_consistent={r['self_consistent_tau03']} "
          f"headroom={r['tau03'].get('headroom_nat')} | ppl={r['ppl']} ppl_ok={r['ppl_ok']} "
          f"128/128={r['complete_128_128']}", flush=True)
    print(f"  vs #752: d_anchored={report['tax_vs_752']['d_anchored']} "
          f"({report['tax_vs_752']['pct_anchored']}%) d_seq_exact={report['tax_vs_752']['d_strict_seq_exact']}",
          flush=True)
    print(f"  report -> {out}", flush=True)
    print("=" * 80, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
