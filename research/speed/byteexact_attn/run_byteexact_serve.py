"""PR #496 -- Faster byte-exact attention: does fixed-order split-KV lift the
357.6 byte-exact ceiling while preserving operative-1.0 greedy identity?

LOCAL MEASUREMENT ONLY. analysis_only=true, official_tps=0. No HF job, no
submission, no draw, no deployed-file change. Two LOCAL serve-venv vLLM prototype
edits (both gated, default-off, restored after):
  * triton_unified_attention.py: FIXED_TILES_PER_SEGMENT (env BYTEEXACT_FIXED_TPS)
    pins tiles_per_segment so split-KV segment boundaries are a fixed function of
    ABSOLUTE key position -> M-invariant -> byte-exact (Thinking-Machines "fix the
    split SIZE not count"). Also the #488 SURGICAL_ATTN_USE_3D_OFF 2D lever.
  * triton_attn.py: BYTEEXACT_NUM_SEGMENTS raises the parallel-segment count so a
    small fixed chunk still covers long context AND keeps occupancy.

The #488 result: deployed 3D split-KV = ~464 TPS but NOT byte-exact (M=8 verify !=
M=1 AR, identity 0.9966); the surgical byte-exact 2D path = 357.6 TPS. #488 framed
the -107 gap as "the price of byte-exact attention on this route ... unavoidable
here." This PR tests that: the microbench (microbench_attn_tax.py) shows the -107
is ~100% split-KV PARALLELISM loss, and a FIXED-order split-KV recovers 95-100% of
it while staying byte-exact (0/8 flips where adaptive flips 6/8). This harness
confirms it END-TO-END on the served stack.

Arms (single pod, back-to-back, shared sigma_hw):
  (a) deployed   : no flag                                    -- fast 3D adaptive, ~464, NON-exact
  (b) surgical   : SURGICAL_ATTN_USE_3D_OFF=1                 -- byte-exact 2D, the 357.6 bar
  (c) byteexact  : BYTEEXACT_FIXED_TPS=T BYTEEXACT_NUM_SEGMENTS=S  -- byte-exact fixed split-KV <-- TEST
Reference arms (SENPAI_REFERENCE_MODE=1 -> spec OFF, M=1 AR, SAME cudagraph serve):
  (d) byteexact_ref : byteexact flags + ref-mode             -- candidate's own M=1 AR
  (e) deployed_ref  : ref-mode                               -- deployed's own M=1 AR (control)

Operative-identity gate (served-vs-served, matched config, WARM round -- NOT the
#488-broken raw-M=1-AR-eager census): byteexact(M=8) vs byteexact_ref(M=1 AR) must
be token-identical (1.0). deployed vs deployed_ref is the discrimination control
(expect ~0.9966, the known deployed flip).

Run under repo .venv (wandb); serve/decode use the submission serve venv::
    .venv/bin/python -m research.speed.byteexact_attn.run_byteexact_serve \
        --n-decodes 3 --fixed-tps 4 --num-segments 64 \
        --wandb-name lawine/byteexact-serve --wandb-group faster-byteexact-attention
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402
from research.tps_noise_floor.run_noise_floor import (  # noqa: E402
    preflight_gpu,
    _gpu_mem_used_mib,
)

OUT_ROOT = ROOT / "research" / "speed" / "byteexact_attn"

# Banked anchors (#488 / equivalence-frontier reframe). The 357.6 surgical rung is
# the byte-exact ceiling this PR tries to beat; 464.69 deployed is the fast (non-
# exact) speed ceiling the candidate tries to approach.
SIGMA_HW = 4.864
MATERIALITY_TPS = 2.0
DEPLOYED_REF_TPS = 464.69     # #488 in-session deployed 3D split-KV (identity 0.9966)
SURGICAL_CEILING_TPS = 357.64  # #488 byte-exact 2D rung -- the bar to beat
PPL_GATE = 2.42


def build_arms(fixed_tps: int, num_segments: int) -> list[dict[str, Any]]:
    be_env = {
        "BYTEEXACT_FIXED_TPS": str(int(fixed_tps)),
        "BYTEEXACT_NUM_SEGMENTS": str(int(num_segments)),
    }
    ref_env = {paths.REFERENCE_MODE_ENV: "1"}
    return [
        {"name": "deployed", "extra_env": {}, "kind": "speed",
         "label": "(a) deployed 3D split-KV adaptive (no flag, ~464, NON-exact)"},
        {"name": "surgical", "extra_env": {"SURGICAL_ATTN_USE_3D_OFF": "1"}, "kind": "speed",
         "label": "(b) surgical byte-exact 2D (the 357.6 ceiling)"},
        {"name": "byteexact", "extra_env": dict(be_env), "kind": "speed",
         "label": f"(c) byteexact fixed split-KV T={fixed_tps} seg={num_segments}  <-- TEST"},
        {"name": "byteexact_ref", "extra_env": {**be_env, **ref_env}, "kind": "ref",
         "label": "(d) byteexact M=1 AR reference (ref-mode, fixed split-KV)"},
        {"name": "deployed_ref", "extra_env": dict(ref_env), "kind": "ref",
         "label": "(e) deployed M=1 AR reference (ref-mode, adaptive) -- control"},
        {"name": "surgical_ref", "extra_env": {"SURGICAL_ATTN_USE_3D_OFF": "1", **ref_env}, "kind": "ref",
         "label": "(f) surgical 2D M=1 AR reference (ref-mode, byte-exact 2D) -- ORACLE"},
        {"name": "full_flag", "extra_env": {"VLLM_BATCH_INVARIANT": "1"}, "kind": "speed",
         "label": "(g) full_flag batch-invariant (M=8==M=1 BY CONSTRUCTION, ~222)"},
        {"name": "full_flag_ref", "extra_env": {"VLLM_BATCH_INVARIANT": "1", **ref_env}, "kind": "ref",
         "label": "(h) full_flag M=1 AR reference -- DECISIVE gate-validity control"},
    ]


# ---------------------------------------------------------------------------
def grep_log(log_path: Path) -> dict[str, Any]:
    out = {
        "splitkv_armed": False, "splitkv_redirects": 0,
        "graph_capture_lines": 0, "onegraph_captured": False,
        "fatal_traceback": False, "n_tracebacks": 0, "benign_usage_tracebacks": 0,
        "batch_invariant_mentions": 0, "byteexact_armed": False,
        "byteexact_log": None, "ref_mode_cleared_spec": False,
    }
    try:
        text = Path(log_path).read_text(errors="replace")
    except OSError:
        return out
    out["splitkv_armed"] = ("[splitkv-verify] wrapped" in text) or ("[splitkv-verify] armed" in text)
    out["splitkv_redirects"] = text.count("-> 3D split-KV")
    out["graph_capture_lines"] = text.count("Capturing CUDA graph") + text.count("Capturing cudagraph")
    out["onegraph_captured"] = "[onegraph] captured" in text
    n_tb = text.count("Traceback (most recent call last)")
    n_usage = text.count("_report_usage_worker")
    out["n_tracebacks"] = n_tb
    out["benign_usage_tracebacks"] = n_usage
    out["fatal_traceback"] = ("CUDA error" in text) or (n_tb > n_usage)
    low = text.lower()
    out["batch_invariant_mentions"] = low.count("batch_invariant") + low.count("batch-invariant")
    out["byteexact_armed"] = "[byteexact] fixed split-KV armed" in text
    for line in text.splitlines():
        if "[byteexact] fixed split-KV armed" in line:
            out["byteexact_log"] = line.strip()
            break
    out["ref_mode_cleared_spec"] = "SENPAI_REFERENCE_MODE active: clearing SPECULATIVE_CONFIG" in text
    return out


def run_arm(arm, submission_dir, server_python, out_dir, *,
            n_decodes, num_prompts, output_len, seed, do_ppl, records_fh):
    name = arm["name"]
    arm_dir = out_dir / name
    arm_dir.mkdir(parents=True, exist_ok=True)
    server_log = arm_dir / "server.log"
    print(f"\n[byteexact] ===== ARM {name} :: {arm['label']} =====", flush=True)
    print(f"[byteexact] extra_env={arm['extra_env']}", flush=True)

    preflight_gpu()
    decodes: list[dict[str, Any]] = []
    peak_mem_mib = 0
    server_ready_s = None
    ppl_summary: dict[str, Any] | None = None
    decode_outs: list[Path] = []

    t_load0 = time.time()
    with harness.LocalServer(
        submission_dir, server_python=server_python,
        log_path=server_log, extra_env=arm["extra_env"],
    ) as server:
        server_ready_s = time.time() - t_load0
        print(f"[byteexact] {name}: server ready in {server_ready_s:.0f}s", flush=True)
        m = _gpu_mem_used_mib()
        if m:
            peak_mem_mib = max(peak_mem_mib, m)

        for i in range(n_decodes):
            decode_out = arm_dir / f"decode_round{i:02d}.jsonl"
            decode_summary = arm_dir / f"decode_round{i:02d}.summary.json"
            decode_outs.append(decode_out)
            t0 = time.time()
            summary = harness.capture_decode(
                server_python, base_url=server.base_url,
                model=server.served_model_name, out_file=decode_out,
                summary_file=decode_summary, num_prompts=num_prompts,
                output_len=output_len, seed=seed,
            )
            wall_around = time.time() - t0
            n_tok = int(summary.get("num_completion_tokens", 0))
            dur = float(summary.get("duration_s", wall_around))
            wall_tps = n_tok / dur if dur > 0 else float("nan")
            n_completed = int(summary.get("num_records", 0))
            rec = {
                "arm": name, "round": i, "wall_tps": wall_tps,
                "num_completion_tokens": n_tok, "decode_duration_s": dur,
                "wall_around_decode_s": wall_around, "num_completed_prompts": n_completed,
                "expected_tokens": num_prompts * output_len,
            }
            decodes.append(rec)
            print(f"[byteexact] {name} round {i}: wall_tps={wall_tps:.2f} "
                  f"tok={n_tok}/{num_prompts * output_len} dur={dur:.1f}s completed={n_completed}",
                  flush=True)
            mm = _gpu_mem_used_mib()
            if mm:
                peak_mem_mib = max(peak_mem_mib, mm)

        if do_ppl:
            try:
                ppl_summary = harness.run_ppl(
                    server_python, base_url=server.base_url,
                    model=server.served_model_name,
                    out_file=arm_dir / "ppl.jsonl", summary_file=arm_dir / "ppl.summary.json",
                )
                print(f"[byteexact] {name}: PPL={ppl_summary.get('ppl')} "
                      f"records={ppl_summary.get('num_records')}", flush=True)
            except Exception as exc:
                print(f"[byteexact] {name}: WARN PPL failed: {exc}", flush=True)

    mech = grep_log(server_log)
    wall_tps_vals = [d["wall_tps"] for d in decodes if d["wall_tps"] == d["wall_tps"]]
    # cold round-0 excluded from the median when >=3 rounds (warm = round>=1)
    warm_vals = [d["wall_tps"] for d in decodes[1:] if d["wall_tps"] == d["wall_tps"]] or wall_tps_vals
    median_tps = statistics.median(warm_vals) if warm_vals else float("nan")
    # warmest decode for the identity comparison (avoids #488 cold-start confound)
    warm_decode_out = str(decode_outs[-1]) if decode_outs else None
    arm_rec = {
        "arm": name, "label": arm["label"], "kind": arm["kind"],
        "extra_env": arm["extra_env"], "median_wall_tps": median_tps,
        "wall_tps_values": wall_tps_vals, "warm_wall_tps_values": warm_vals,
        "wall_tps_n": len(warm_vals),
        "wall_tps_std": statistics.stdev(warm_vals) if len(warm_vals) > 1 else 0.0,
        "server_ready_s": server_ready_s, "peak_gpu_mem_mib": peak_mem_mib,
        "ppl": (ppl_summary or {}).get("ppl"),
        "ppl_num_records": (ppl_summary or {}).get("num_records"),
        "num_completed_prompts": decodes[0]["num_completed_prompts"] if decodes else None,
        "completion_full": bool(decodes and decodes[0]["num_completion_tokens"] == num_prompts * output_len),
        "mechanism": mech, "warm_decode_out": warm_decode_out,
        "decodes": decodes,
    }
    records_fh.write(json.dumps(arm_rec) + "\n")
    records_fh.flush()
    mech = arm_rec.get("mechanism") or {}
    print(f"[byteexact] ARM {name} SUMMARY: median_wall_tps={median_tps:.2f} "
          f"(n={arm_rec['wall_tps_n']}, std={arm_rec['wall_tps_std']:.2f}) PPL={arm_rec['ppl']} "
          f"completed={arm_rec['num_completed_prompts']} full={arm_rec['completion_full']} "
          f"peak={peak_mem_mib}MiB | splitkv_redirects={mech.get('splitkv_redirects')} "
          f"byteexact_armed={mech.get('byteexact_armed')} onegraph={mech.get('onegraph_captured')} "
          f"fatal_tb={mech.get('fatal_traceback')}", flush=True)
    if mech.get("byteexact_log"):
        print(f"[byteexact]   mech: {mech['byteexact_log']}", flush=True)
    return arm_rec


# ---------------------------------------------------------------------------
def _load_token_seqs(path: Path | None) -> dict[str, list[int]] | None:
    if not path or not Path(path).exists():
        return None
    seqs: dict[str, list[int]] = {}
    try:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = str(obj.get("id", obj.get("dataset_index", obj.get("index", len(seqs)))))
            toks = obj.get("completion_token_ids")
            if isinstance(toks, list):
                seqs[key] = [int(t) for t in toks]
    except Exception as exc:
        print(f"[byteexact] token-seq load failed for {path}: {exc}", flush=True)
        return None
    return seqs or None


def cross_arm_token_diff(a: Path | None, b: Path | None, label: str) -> dict[str, Any]:
    sa, sb = _load_token_seqs(a), _load_token_seqs(b)
    if not sa or not sb:
        return {"label": label, "available": False}
    common = sorted(set(sa) & set(sb))
    total = matched = n_flipped_seqs = 0
    first_div = []
    for k in common:
        ta, tb = sa[k], sb[k]
        n = min(len(ta), len(tb))
        seq_flips = sum(1 for i in range(n) if ta[i] != tb[i])
        total += n
        matched += n - seq_flips
        if seq_flips or len(ta) != len(tb):
            n_flipped_seqs += 1
            for i in range(n):
                if ta[i] != tb[i]:
                    first_div.append({"prompt": k, "pos": i, "a": ta[i], "b": tb[i]})
                    break
    return {
        "label": label, "available": True, "n_prompts_compared": len(common),
        "n_tokens_compared": total, "n_tokens_matched": matched,
        "token_identity_rate": (matched / total) if total else None,
        "n_sequences_with_any_flip": n_flipped_seqs, "first_divergences": first_div[:10],
    }


# ---------------------------------------------------------------------------
def build_verdict(arm_recs, diffs, fixed_tps, num_segments) -> dict[str, Any]:
    dep = arm_recs.get("deployed", {})
    surg = arm_recs.get("surgical", {})
    be = arm_recs.get("byteexact", {})
    be_tps = be.get("median_wall_tps")
    surg_tps = surg.get("median_wall_tps")
    dep_tps = dep.get("median_wall_tps")

    # The operative-1.0 gate: candidate served (M=8) vs its OWN M=1 AR reference.
    be_ident = diffs.get("byteexact_vs_byteexact_ref", {})
    dep_ident = diffs.get("deployed_vs_deployed_ref", {})
    be_identity_rate = be_ident.get("token_identity_rate") if be_ident.get("available") else None
    be_operative_1 = (be_identity_rate == 1.0) if be_identity_rate is not None else None

    # DECISIVE gate-validity control: full_flag is byte-exact M=8==M=1 BY
    # CONSTRUCTION. If its own M=1-AR gate is not ~1.0, the M=1-AR gate cannot
    # discriminate (the #488-broken regime) and be_identity_rate is uninformative.
    ff_ctrl = diffs.get("full_flag_vs_full_flag_ref", {})
    ff_ctrl_rate = ff_ctrl.get("token_identity_rate") if ff_ctrl.get("available") else None
    m1ar_gate_valid = (ff_ctrl_rate is not None and ff_ctrl_rate >= 0.9999)
    # served-vs-served matched-config against the M=8==M=1 ground truth.
    be_vs_ff = diffs.get("byteexact_vs_full_flag", {})
    be_vs_ff_rate = be_vs_ff.get("token_identity_rate") if be_vs_ff.get("available") else None
    surg_vs_ff = diffs.get("surgical_vs_full_flag", {})
    surg_vs_ff_rate = surg_vs_ff.get("token_identity_rate") if surg_vs_ff.get("available") else None

    lift_vs_surgical = None
    if isinstance(be_tps, (int, float)) and isinstance(surg_tps, (int, float)):
        lift_vs_surgical = be_tps - surg_tps
    # how much of the deployed-minus-surgical (-107 byte-exact) gap the candidate recovers
    recovery = None
    if all(isinstance(x, (int, float)) for x in (be_tps, surg_tps, dep_tps)) and (dep_tps - surg_tps) > 0:
        recovery = (be_tps - surg_tps) / (dep_tps - surg_tps)

    lift_clears = bool(isinstance(lift_vs_surgical, (int, float)) and lift_vs_surgical > max(MATERIALITY_TPS, SIGMA_HW))
    ceiling_lifted = bool(lift_clears and be_operative_1 is True)

    return {
        "candidate_scheme": f"fixed-order split-KV (tiles_per_segment={fixed_tps}, num_par_softmax_segments={num_segments}, CHUNK={fixed_tps*16} keys)",
        "candidate_realized_tps": be_tps,
        "candidate_realized_tps_basis": "full-serve median warm wall_tps (num_completion_tokens / decode duration_s)",
        "surgical_ceiling_tps": surg_tps,
        "deployed_ref_tps": dep_tps,
        "candidate_lift_vs_surgical_357": lift_vs_surgical,
        "materiality_bar_tps": MATERIALITY_TPS, "sigma_hw": SIGMA_HW,
        "lift_clears_materiality_and_sigma": lift_clears,
        "candidate_recovery_fraction_of_deployed_minus_surgical_gap": recovery,
        "candidate_operative_identity_rate": be_identity_rate,
        "candidate_operative_1_0": be_operative_1,
        "gate_validity_control_rate": ff_ctrl_rate,
        "m1ar_gate_valid": m1ar_gate_valid,
        "candidate_vs_groundtruth_m8_rate": be_vs_ff_rate,
        "surgical_vs_groundtruth_m8_rate": surg_vs_ff_rate,
        "deployed_control_identity_rate": (dep_ident.get("token_identity_rate") if dep_ident.get("available") else None),
        "gate_discriminates": bool(
            be_operative_1 is True
            and dep_ident.get("available")
            and isinstance(dep_ident.get("token_identity_rate"), (int, float))
            and dep_ident["token_identity_rate"] < 1.0
        ),
        "candidate_ppl": be.get("ppl"),
        "candidate_ppl_passes_gate": (isinstance(be.get("ppl"), (int, float)) and be["ppl"] <= PPL_GATE),
        "candidate_completion_full": be.get("completion_full"),
        "candidate_byteexact_armed": (be.get("mechanism") or {}).get("byteexact_armed"),
        "candidate_graph_capture_ok": (
            (be.get("mechanism") or {}).get("onegraph_captured", False)
            and not (be.get("mechanism") or {}).get("fatal_traceback", False)
        ),
        "fast_strict_ceiling_lifted": ceiling_lifted,
        "analysis_only": True, "official_tps": 0, "no_served_file_change": True,
    }


# ---------------------------------------------------------------------------
def log_wandb(args, arm_recs, verdict, diffs, microbench, fixed_tps, num_segments):
    if args.no_wandb:
        return None
    try:
        from scripts import wandb_logging
    except Exception as exc:
        print(f"[byteexact] wandb_logging import failed ({exc}); skipping", flush=True)
        return None
    try:
        run = wandb_logging.init_wandb_run(
            job_type="faster-byteexact-attention", agent="lawine",
            name=args.wandb_name or "lawine/byteexact-serve", group=args.wandb_group,
            tags=["faster-byteexact-attention", "pr496", "analysis-only"],
            config={
                "n_decodes": args.n_decodes, "num_prompts": args.num_prompts,
                "output_len": args.output_len, "seed": args.seed,
                "fixed_tps": fixed_tps, "num_segments": num_segments,
                "chunk_keys": fixed_tps * 16, "coverage_keys": fixed_tps * num_segments * 16,
                "sigma_hw": SIGMA_HW, "deployed_ref_tps": DEPLOYED_REF_TPS,
                "surgical_ceiling_tps": SURGICAL_CEILING_TPS,
                "analysis_only": True, "official_tps": 0,
            },
        )
    except Exception as exc:
        print(f"[byteexact] wandb init failed ({exc}); skipping", flush=True)
        return None
    if run is None:
        print("[byteexact] wandb disabled (no API key); skipping", flush=True)
        return None
    run_id = getattr(run, "id", None)
    try:
        for i, name in enumerate(["deployed", "surgical", "byteexact", "full_flag", "byteexact_ref", "full_flag_ref", "deployed_ref"]):
            rec = arm_recs.get(name)
            if not rec:
                continue
            metrics = {
                f"arm/{name}/median_wall_tps": rec.get("median_wall_tps"),
                f"arm/{name}/wall_tps_std": rec.get("wall_tps_std"),
                f"arm/{name}/ppl": rec.get("ppl"),
                f"arm/{name}/peak_gpu_mem_mib": rec.get("peak_gpu_mem_mib"),
                f"arm/{name}/splitkv_redirects": (rec.get("mechanism") or {}).get("splitkv_redirects"),
            }
            metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            wandb_logging.log_event(run, f"arm_{name}", step=i, metrics=metrics)
        for k, d in diffs.items():
            if d.get("available") and isinstance(d.get("token_identity_rate"), (int, float)):
                wandb_logging.log_summary(run, {f"identity/{k}": d["token_identity_rate"]})
        flat = {f"verdict/{k}": v for k, v in verdict.items() if isinstance(v, (int, float, bool))}
        wandb_logging.log_summary(run, flat, step=5)
        wandb_logging.log_json_artifact(
            run, name="byteexact_attn_serve", artifact_type="faster-byteexact-attention",
            data={"arms": arm_recs, "verdict": verdict, "identity_diffs": diffs,
                  "microbench_attn_tax": microbench},
        )
    except Exception as exc:
        print(f"[byteexact] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:
            pass
    return run_id


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", default="fa2sw_precache_kenyan")
    ap.add_argument("--arms", default="deployed,surgical,byteexact,byteexact_ref,deployed_ref")
    ap.add_argument("--n-decodes", type=int, default=3)
    ap.add_argument("--ref-decodes", type=int, default=2, help="decodes for reference arms (warm identity)")
    ap.add_argument("--fixed-tps", type=int, default=4, help="pinned tiles_per_segment (CHUNK=16*T keys)")
    ap.add_argument("--num-segments", type=int, default=64, help="parallel softmax segments (coverage=16*T*S)")
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--seed", type=int, default=paths.SEED)
    ap.add_argument("--no-ppl", dest="do_ppl", action="store_false", default=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--microbench-json", type=Path, default=OUT_ROOT / "microbench_results.json")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="faster-byteexact-attention")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    if args.smoke:
        args.num_prompts = min(args.num_prompts, 8)
        args.output_len = min(args.output_len, 16)
        args.n_decodes = 1
        args.ref_decodes = 1
        args.do_ppl = False
        args.no_wandb = True

    for note in paths.prepare_local_gpu_env():
        print(f"[byteexact] {note}", flush=True)

    submission_dir = (ROOT / "submissions" / args.submission).resolve()
    if not submission_dir.exists():
        raise SystemExit(f"submission not found: {submission_dir}")
    manifest = harness.load_manifest(submission_dir)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    coverage = args.fixed_tps * args.num_segments * 16
    print(f"[byteexact] submission={submission_dir.name} server_python={server_python}", flush=True)
    print(f"[byteexact] candidate: fixed_tps={args.fixed_tps} num_segments={args.num_segments} "
          f"CHUNK={args.fixed_tps*16} keys coverage={coverage} keys (max_model_len=4096)", flush=True)
    if coverage < 4096:
        print(f"[byteexact] WARNING: coverage {coverage} < max_model_len 4096 -> long seqs may flip!", flush=True)

    all_arms = build_arms(args.fixed_tps, args.num_segments)
    want = [a.strip() for a in args.arms.split(",") if a.strip()]
    arms = [a for a in all_arms if a["name"] in want]
    if not arms:
        raise SystemExit(f"no arms selected from {args.arms!r}")

    out_dir = (args.out_dir or (OUT_ROOT / ("smoke" if args.smoke else "serve_run"))).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "arm_records.jsonl"
    print(f"[byteexact] arms={[a['name'] for a in arms]} -> {out_dir}", flush=True)

    t0 = time.time()
    arm_recs: dict[str, dict[str, Any]] = {}
    with open(records_path, "w") as records_fh:
        for arm in arms:
            nd = args.ref_decodes if arm["kind"] == "ref" else args.n_decodes
            rec = run_arm(
                arm, submission_dir, server_python, out_dir,
                n_decodes=nd, num_prompts=args.num_prompts,
                output_len=args.output_len, seed=args.seed,
                do_ppl=(args.do_ppl and arm["kind"] == "speed"), records_fh=records_fh,
            )
            arm_recs[arm["name"]] = rec
    elapsed = time.time() - t0

    # operative-identity diffs (WARM matched-round, served-vs-served)
    p = {n: (Path(r["warm_decode_out"]) if r.get("warm_decode_out") else None)
         for n, r in arm_recs.items()}
    diffs: dict[str, Any] = {}
    if "byteexact" in arm_recs and "byteexact_ref" in arm_recs:
        diffs["byteexact_vs_byteexact_ref"] = cross_arm_token_diff(
            p.get("byteexact"), p.get("byteexact_ref"),
            "byteexact M=8 vs its M=1 AR reference (OPERATIVE GATE; expect 1.0)")
    if "deployed" in arm_recs and "deployed_ref" in arm_recs:
        diffs["deployed_vs_deployed_ref"] = cross_arm_token_diff(
            p.get("deployed"), p.get("deployed_ref"),
            "deployed M=8 vs its M=1 AR reference (CONTROL; expect ~0.9966)")
    if "byteexact" in arm_recs and "deployed" in arm_recs:
        diffs["byteexact_vs_deployed"] = cross_arm_token_diff(
            p.get("byteexact"), p.get("deployed"),
            "byteexact vs deployed served (different configs; expect flips)")
    if "byteexact_ref" in arm_recs and "deployed_ref" in arm_recs:
        diffs["byteexact_ref_vs_deployed_ref"] = cross_arm_token_diff(
            p.get("byteexact_ref"), p.get("deployed_ref"),
            "the two M=1 AR references (do fixed & adaptive M=1 agree?)")
    if "surgical" in arm_recs and "surgical_ref" in arm_recs:
        diffs["surgical_vs_surgical_ref"] = cross_arm_token_diff(
            p.get("surgical"), p.get("surgical_ref"),
            "surgical 2D M=8 vs its M=1 AR reference (ORACLE: byte-exact attn + adaptive matmul; isolates matmul non-inv)")
    # DECISIVE gate-validity control: full_flag is M=8==M=1 BY CONSTRUCTION. If its
    # own M=8-vs-M=1-AR gate is NOT ~1.0, the M=1-AR gate is broken (#488) and the
    # candidate's byteexact_vs_byteexact_ref number is uninformative.
    if "full_flag" in arm_recs and "full_flag_ref" in arm_recs:
        diffs["full_flag_vs_full_flag_ref"] = cross_arm_token_diff(
            p.get("full_flag"), p.get("full_flag_ref"),
            "full_flag M=8 vs its M=1 AR reference (DECISIVE CONTROL: byte-exact BY CONSTRUCTION; gate valid iff ~1.0)")
    # served-vs-served matched-config (the #488-clean gate): candidate & surgical
    # vs the batch-invariant ground truth at M=8 (both cudagraph, warm).
    if "byteexact" in arm_recs and "full_flag" in arm_recs:
        diffs["byteexact_vs_full_flag"] = cross_arm_token_diff(
            p.get("byteexact"), p.get("full_flag"),
            "byteexact M=8 vs full_flag M=8 ground truth (served-vs-served matched-config; flips => margin-census ULP-ties?)")
    if "surgical" in arm_recs and "full_flag" in arm_recs:
        diffs["surgical_vs_full_flag"] = cross_arm_token_diff(
            p.get("surgical"), p.get("full_flag"),
            "surgical M=8 vs full_flag M=8 ground truth (reproduces #488 0.9763 accepted-byte-exact standard)")
    if "deployed" in arm_recs and "full_flag" in arm_recs:
        diffs["deployed_vs_full_flag"] = cross_arm_token_diff(
            p.get("deployed"), p.get("full_flag"),
            "deployed M=8 vs full_flag M=8 ground truth (NON-exact control; expect many flips)")

    verdict = build_verdict(arm_recs, diffs, args.fixed_tps, args.num_segments)

    microbench = None
    try:
        if args.microbench_json.exists():
            microbench = json.loads(args.microbench_json.read_text())
    except Exception as exc:
        print(f"[byteexact] microbench json load failed: {exc}", flush=True)

    try:
        from scripts import wandb_logging
        git = wandb_logging.git_info()
    except Exception:
        git = {}

    result = {
        "pr": 496, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "submission": args.submission,
        "candidate": {"fixed_tps": args.fixed_tps, "num_segments": args.num_segments,
                      "chunk_keys": args.fixed_tps * 16, "coverage_keys": coverage},
        "workload": {"num_prompts": args.num_prompts, "output_len": args.output_len,
                     "seed": args.seed, "n_decodes": args.n_decodes},
        "elapsed_s": elapsed, "git": git, "arms": arm_recs,
        "identity_diffs": diffs, "verdict": verdict,
    }
    run_id = None
    if not args.smoke:
        run_id = log_wandb(args, arm_recs, verdict, diffs, microbench, args.fixed_tps, args.num_segments)
    result["wandb_run_id"] = run_id
    result_path = out_dir / "byteexact_serve_result.json"
    result_path.write_text(json.dumps(result, indent=2))

    print(f"\n[byteexact] ============ VERDICT ({elapsed/60:.1f} min) ============", flush=True)
    print(f"  deployed_ref_tps (fast, NON-exact) = {verdict.get('deployed_ref_tps')}", flush=True)
    print(f"  surgical_ceiling_tps (2D byte-exact)= {verdict.get('surgical_ceiling_tps')}", flush=True)
    print(f"  candidate_realized_tps             = {verdict.get('candidate_realized_tps')}", flush=True)
    print(f"  candidate_lift_vs_surgical_357     = {verdict.get('candidate_lift_vs_surgical_357')} "
          f"(materiality>{MATERIALITY_TPS}, sigma_hw {SIGMA_HW})", flush=True)
    print(f"  candidate_recovery_frac            = {verdict.get('candidate_recovery_fraction_of_deployed_minus_surgical_gap')}", flush=True)
    print(f"  candidate_operative_identity_rate  = {verdict.get('candidate_operative_identity_rate')} "
          f"(operative_1.0={verdict.get('candidate_operative_1_0')})", flush=True)
    print(f"  >>> GATE-VALIDITY CONTROL full_flag(M8)-vs-M1AR = {verdict.get('gate_validity_control_rate')} "
          f"(m1ar_gate_valid={verdict.get('m1ar_gate_valid')}) <<<", flush=True)
    print(f"  candidate_vs_groundtruth_M8 (served-vs-served) = {verdict.get('candidate_vs_groundtruth_m8_rate')} "
          f"| surgical_vs_groundtruth_M8 = {verdict.get('surgical_vs_groundtruth_m8_rate')}", flush=True)
    print(f"  deployed_control_identity_rate     = {verdict.get('deployed_control_identity_rate')} "
          f"(gate_discriminates={verdict.get('gate_discriminates')})", flush=True)
    print(f"  candidate_ppl                      = {verdict.get('candidate_ppl')} "
          f"(gate<={PPL_GATE}: {verdict.get('candidate_ppl_passes_gate')})", flush=True)
    print(f"  candidate_byteexact_armed          = {verdict.get('candidate_byteexact_armed')}", flush=True)
    print(f"  >>> fast_strict_ceiling_lifted = {verdict.get('fast_strict_ceiling_lifted')} <<<", flush=True)
    for k, d in diffs.items():
        if d.get("available"):
            print(f"  identity {k}: rate={d.get('token_identity_rate')} "
                  f"flips_seqs={d.get('n_sequences_with_any_flip')}/{d.get('n_prompts_compared')}", flush=True)
    print(f"[byteexact] artifacts -> {result_path}  wandb_run_id={run_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
