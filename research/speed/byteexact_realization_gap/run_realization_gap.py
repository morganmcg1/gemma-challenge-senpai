"""PR #523 -- Decompose the 357->457 byte-exact realization gap (~100 TPS).

LOCAL MEASUREMENT ONLY. ``analysis_only=true``, ``official_tps=0``. No HF job, no
submission, no served-file change. Serves the PACKAGED strict submissions back-to-
back on the same pod and measures the robust official-spec ``wall_tps`` (=
num_completion_tokens / decode duration_s) at the full 128x512 workload, with the
env-gated ``steptime_patch`` (STEPTIME=1) giving a per-decode-step coarse split:

  exec.gap   host/python gap between steps (CPU, the cudagraph-replay can't hide)
  exec.gpu   verify-step GPU time (M=8 spec-verify: attention + body GEMM + lm_head)
  draft.gpu  drafter (MTP propose) GPU time

Because the three byte-exact rungs (surgical 2D / byteexact fixed-3D / deployed
adaptive-3D) are byte-identical in EVERYTHING except the attention path (same int4
Marlin body, same drafter, same lm_head, same sampler, same ONEGRAPH), any served
TPS delta between them is attributable to the attention path by construction; the
steptime split CONFIRMS it lands in exec.gpu (verify) with host/draft ~constant.

Arms (select with --arms; each serves a packaged submission + optional env override):

  deployed     fa2sw_precache_kenyan          adaptive 3D split-KV (NOT byte-exact)
  surgical     fa2sw_strict_surgical357       2D order-preserving (byte-exact, no split-KV)
  bx_T4_S64    fa2sw_..._byteexact_splitkv399 fixed 3D split-KV T=4/S=64 (packaged, byte-exact)
  bx_T16_S16   byteexact399 + T=16/S=16       coarse fixed-3D (2 segs @ L=512)  -- geometry sweep
  bx_T8_S32    byteexact399 + T=8/S=32         (4 segs @ L=512)                  -- geometry sweep
  bx_T2_S128   byteexact399 + T=2/S=128        (16 segs @ L=512)                 -- geometry sweep
  bx_fisampler byteexact399 + FlashInfer sampler ON (#481 sampler lever; may crash on this pod)

The geometry sweep holds coverage = S*T*TILE_SIZE = 4096 keys (= max_model_len) CONSTANT
while varying the parallel-segment granularity (act_num_segments @ seq L = ceil(L/(T*16))),
so every config stays byte-exact (fixed T => M-invariant) -- verified separately by the
0/8 microbench (verify_packaged_patch.py). This is the #481 "attention split-KV geometry"
lever realized end-to-end.

Run under the repo .venv (has wandb); serve/decode subprocs use the submission serve venv::

    .venv/bin/python -m research.speed.byteexact_realization_gap.run_realization_gap \
        --arms deployed,surgical,bx_T4_S64 --n-decodes 2 \
        --wandb-name lawine/realization-gap-ledger --wandb-group byteexact-realization-gap
"""
from __future__ import annotations

import argparse
import json
import re
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

OUT_ROOT = ROOT / "research" / "speed" / "byteexact_realization_gap"

# Banked anchors (PR #523 body). Local-scale references; official scale differs by
# tau_lo ~ 1.0352 (lawine #267) -- never mix local and official numbers.
SIGMA_HW = 4.864              # between-session local wall_tps sigma (fresh server per arm)
PPL_GATE = 2.42
DEPLOYED_OFFICIAL = 481.53    # PR #52 deployed FAST (non-equivalent); local serves ~450
SURGICAL_LOCAL = 357.6        # PR #488 byte-exact 2D rung (local)
BYTEEXACT_LOCAL = 399.75      # PR #496 byte-exact fixed-3D rung (local, 32x256 headline -- recert @128x512 here)
STRICT_FRONTIER_PRED = 457.5  # #466/#474 MICROBENCH PROJECTION (never served; provenance = task step 1)

# Geometry-sweep configs hold coverage = T*S*16 = 4096 keys constant; segment
# granularity @ L=512 = ceil(512/(T*16)).
ARMS: dict[str, dict[str, Any]] = {
    "deployed": {
        "submission": "fa2sw_precache_kenyan",
        "extra_env": {},
        "label": "deployed adaptive 3D split-KV (NOT byte-exact, ~450 local ref)",
        "byte_exact": False,
    },
    "surgical": {
        "submission": "fa2sw_strict_surgical357",
        "extra_env": {},
        "label": "surgical 2D order-preserving (byte-exact, no split-KV) -- the 357 floor",
        "byte_exact": True,
    },
    "bx_T4_S64": {
        "submission": "fa2sw_strict_byteexact_splitkv399",
        "extra_env": {},
        "label": "byteexact fixed-3D T=4/S=64 (packaged, byte-exact) -- 8 segs @ L=512",
        "byte_exact": True,
    },
    "bx_T16_S16": {
        "submission": "fa2sw_strict_byteexact_splitkv399",
        "extra_env": {"BYTEEXACT_FIXED_TPS": "16", "BYTEEXACT_NUM_SEGMENTS": "16"},
        "label": "byteexact fixed-3D T=16/S=16 (byte-exact) -- 2 segs @ L=512 [geometry sweep]",
        "byte_exact": True,
    },
    "bx_T8_S32": {
        "submission": "fa2sw_strict_byteexact_splitkv399",
        "extra_env": {"BYTEEXACT_FIXED_TPS": "8", "BYTEEXACT_NUM_SEGMENTS": "32"},
        "label": "byteexact fixed-3D T=8/S=32 (byte-exact) -- 4 segs @ L=512 [geometry sweep]",
        "byte_exact": True,
    },
    "bx_T2_S128": {
        "submission": "fa2sw_strict_byteexact_splitkv399",
        "extra_env": {"BYTEEXACT_FIXED_TPS": "2", "BYTEEXACT_NUM_SEGMENTS": "128"},
        "label": "byteexact fixed-3D T=2/S=128 (byte-exact) -- 16 segs @ L=512 [geometry sweep]",
        "byte_exact": True,
    },
    "bx_fisampler": {
        "submission": "fa2sw_strict_byteexact_splitkv399",
        "extra_env": {"VLLM_USE_FLASHINFER_SAMPLER": "1"},
        "label": "byteexact fixed-3D + FlashInfer sampler ON (#481 sampler lever)",
        "byte_exact": True,
    },
    "bx_eager_drafter": {
        "submission": "fa2sw_strict_byteexact_splitkv399",
        "extra_env": {"ONEGRAPH": "0"},
        "label": "byteexact fixed-3D + ONEGRAPH=0 (drafter eager K-iters, token-equiv) -- cudagraph lever",
        "byte_exact": True,
    },
}


# ---------------------------------------------------------------------------
# Server-log mechanism + steptime evidence
# ---------------------------------------------------------------------------
_STEPTIME_AGG = re.compile(
    r"\[steptime\] agg n=(\d+) kind=(\w+) "
    r"gap p50=([\d.]+) p90=([\d.]+) mean=([\d.]+) \| "
    r"cpu p50=([\d.]+) p90=([\d.]+) mean=([\d.]+) \| "
    r"gpu p50=([\d.]+) p90=([\d.]+) mean=([\d.]+) \| "
    r"dcpu p50=([\d.]+) p90=([\d.]+) mean=([\d.]+) \| "
    r"dgpu p50=([\d.]+) p90=([\d.]+) mean=([\d.]+)"
)


def parse_steptime(text: str) -> dict[str, Any]:
    """Parse the LAST cumulative steptime agg line per kind (exec / draft).

    Returns per-kind p50/mean ms for gap (host between steps), cpu (call wall),
    gpu (in-call GPU), dgpu (drafter GPU when inside the call)."""
    out: dict[str, Any] = {}
    for m in _STEPTIME_AGG.finditer(text):
        kind = m.group(2)
        out[kind] = {
            "n": int(m.group(1)),
            "gap_p50": float(m.group(3)), "gap_mean": float(m.group(5)),
            "cpu_p50": float(m.group(6)), "cpu_mean": float(m.group(8)),
            "gpu_p50": float(m.group(9)), "gpu_mean": float(m.group(11)),
            "dgpu_p50": float(m.group(15)), "dgpu_mean": float(m.group(17)),
        }
    return out


def grep_log(log_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "splitkv_armed": False, "splitkv_redirects": 0,
        "byteexact_armed": False, "byteexact_fixed_tps": None, "byteexact_num_segments": None,
        "surgical_armed": False,
        "onegraph_captured": False, "graph_capture_lines": 0,
        "fatal_traceback": False, "n_tracebacks": 0, "benign_usage_tracebacks": 0,
        "flashinfer_sampler": None, "steptime": {},
    }
    try:
        text = Path(log_path).read_text(errors="replace")
    except OSError:
        return out
    out["splitkv_armed"] = ("[splitkv-verify] wrapped" in text) or ("[splitkv-verify] armed" in text)
    out["splitkv_redirects"] = text.count("-> 3D split-KV")
    out["byteexact_armed"] = "[byteexact] fixed split-KV armed" in text
    bx = re.search(r"tiles_per_segment=(\d+) num_par_softmax_segments=(\d+)", text)
    if bx:
        out["byteexact_fixed_tps"] = int(bx.group(1))
        out["byteexact_num_segments"] = int(bx.group(2))
    out["surgical_armed"] = "[surgical-attn] armed" in text or "forced is_batch_invariant=True" in text
    out["graph_capture_lines"] = text.count("Capturing CUDA graph") + text.count("Capturing cudagraph")
    out["onegraph_captured"] = "[onegraph] captured" in text
    n_tb = text.count("Traceback (most recent call last)")
    n_usage = text.count("_report_usage_worker")
    out["n_tracebacks"] = n_tb
    out["benign_usage_tracebacks"] = n_usage
    out["fatal_traceback"] = ("CUDA error" in text) or (n_tb > n_usage)
    if "flashinfer" in text.lower():
        out["flashinfer_sampler"] = "flashinfer-mentioned"
    out["steptime"] = parse_steptime(text)
    return out


# ---------------------------------------------------------------------------
# One arm: fresh server, N back-to-back decodes (median wall_tps), one PPL pass
# ---------------------------------------------------------------------------
def run_arm(
    arm_name: str, arm: dict[str, Any], server_python: Path, out_dir: Path,
    *, n_decodes: int, num_prompts: int, output_len: int, seed: int,
    do_ppl: bool, records_fh,
) -> dict[str, Any]:
    submission_dir = (ROOT / "submissions" / arm["submission"]).resolve()
    arm_dir = out_dir / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    server_log = arm_dir / "server.log"
    extra_env = dict(arm["extra_env"])
    extra_env["STEPTIME"] = "1"  # coarse per-step split; events at python boundary (no compute change)
    print(f"\n[gap] ===== ARM {arm_name} :: {arm['label']} =====", flush=True)
    print(f"[gap] submission={arm['submission']} extra_env={extra_env}", flush=True)

    preflight_gpu()
    decodes: list[dict[str, Any]] = []
    peak_mem_mib = 0
    server_ready_s = None
    ppl_summary: dict[str, Any] | None = None
    first_decode_out: Path | None = None
    server_error: str | None = None

    t_load0 = time.time()
    try:
        with harness.LocalServer(
            submission_dir, server_python=server_python,
            log_path=server_log, extra_env=extra_env,
        ) as server:
            server_ready_s = time.time() - t_load0
            print(f"[gap] {arm_name}: server ready in {server_ready_s:.0f}s", flush=True)
            m = _gpu_mem_used_mib()
            if m:
                peak_mem_mib = max(peak_mem_mib, m)
            for i in range(n_decodes):
                decode_out = arm_dir / f"decode_round{i:02d}.jsonl"
                decode_summary = arm_dir / f"decode_round{i:02d}.summary.json"
                if i == 0:
                    first_decode_out = decode_out
                summary = harness.capture_decode(
                    server_python, base_url=server.base_url,
                    model=server.served_model_name, out_file=decode_out,
                    summary_file=decode_summary, num_prompts=num_prompts,
                    output_len=output_len, seed=seed,
                )
                n_tok = int(summary.get("num_completion_tokens", 0))
                dur = float(summary.get("duration_s", 0.0))
                wall_tps = n_tok / dur if dur > 0 else float("nan")
                n_completed = int(summary.get("num_records", 0))
                rec = {
                    "round": i, "wall_tps": wall_tps, "num_completion_tokens": n_tok,
                    "decode_duration_s": dur, "num_completed_prompts": n_completed,
                    "expected_tokens": num_prompts * output_len,
                }
                decodes.append(rec)
                print(f"[gap] {arm_name} round {i}: wall_tps={wall_tps:.2f} "
                      f"tok={n_tok}/{num_prompts * output_len} dur={dur:.1f}s "
                      f"completed={n_completed}", flush=True)
                mm = _gpu_mem_used_mib()
                if mm:
                    peak_mem_mib = max(peak_mem_mib, mm)
            if do_ppl:
                try:
                    ppl_summary = harness.run_ppl(
                        server_python, base_url=server.base_url,
                        model=server.served_model_name,
                        out_file=arm_dir / "ppl.jsonl",
                        summary_file=arm_dir / "ppl.summary.json",
                    )
                    print(f"[gap] {arm_name}: PPL={ppl_summary.get('ppl')} "
                          f"records={ppl_summary.get('num_records')}", flush=True)
                except Exception as exc:
                    print(f"[gap] {arm_name}: WARN PPL failed: {exc}", flush=True)
    except Exception as exc:  # server failed to start (e.g. flashinfer sampler JIT crash)
        server_error = repr(exc)
        print(f"[gap] {arm_name}: SERVER ERROR: {server_error}", flush=True)

    mech = grep_log(server_log)
    wall_tps_vals = [d["wall_tps"] for d in decodes if d["wall_tps"] == d["wall_tps"]]
    median_tps = statistics.median(wall_tps_vals) if wall_tps_vals else float("nan")
    arm_rec = {
        "arm": arm_name, "submission": arm["submission"], "label": arm["label"],
        "extra_env": arm["extra_env"], "byte_exact_claim": arm["byte_exact"],
        "median_wall_tps": median_tps, "wall_tps_values": wall_tps_vals,
        "wall_tps_n": len(wall_tps_vals),
        "wall_tps_std": statistics.stdev(wall_tps_vals) if len(wall_tps_vals) > 1 else 0.0,
        "server_ready_s": server_ready_s, "peak_gpu_mem_mib": peak_mem_mib,
        "ppl": (ppl_summary or {}).get("ppl"), "ppl_num_records": (ppl_summary or {}).get("num_records"),
        "num_completed_prompts": decodes[0]["num_completed_prompts"] if decodes else None,
        "completion_full": bool(decodes and decodes[0]["num_completion_tokens"] == num_prompts * output_len),
        "server_error": server_error, "mechanism": mech,
        "first_decode_out": str(first_decode_out) if first_decode_out else None,
        "decodes": decodes,
    }
    records_fh.write(json.dumps(arm_rec) + "\n")
    records_fh.flush()
    st = (mech.get("steptime") or {}).get("exec", {})
    dr = (mech.get("steptime") or {}).get("draft", {})
    print(f"[gap] ARM {arm_name} SUMMARY: median_wall_tps={median_tps:.2f} "
          f"(n={len(wall_tps_vals)}) PPL={arm_rec['ppl']} "
          f"onegraph={mech.get('onegraph_captured')} redirects={mech.get('splitkv_redirects')} "
          f"bx_armed={mech.get('byteexact_armed')}({mech.get('byteexact_fixed_tps')}/{mech.get('byteexact_num_segments')}) "
          f"| steptime exec gpu={st.get('gpu_mean')} gap={st.get('gap_mean')} "
          f"draft gpu={dr.get('gpu_mean')}", flush=True)
    return arm_rec


# ---------------------------------------------------------------------------
# Cross-arm served token diff (corroborates byte-exactness on the served path)
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
    except Exception as exc:  # noqa: BLE001
        print(f"[gap] token-seq load failed for {path}: {exc}", flush=True)
        return None
    return seqs or None


def cross_arm_token_diff(a: Path | None, b: Path | None, label: str) -> dict[str, Any]:
    sa, sb = _load_token_seqs(a), _load_token_seqs(b)
    if not sa or not sb:
        return {"label": label, "available": False}
    common = sorted(set(sa) & set(sb))
    total = matched = n_flipped = 0
    for k in common:
        ta, tb = sa[k], sb[k]
        n = min(len(ta), len(tb))
        seq_flips = sum(1 for i in range(n) if ta[i] != tb[i])
        total += n
        matched += n - seq_flips
        if seq_flips or len(ta) != len(tb):
            n_flipped += 1
    return {
        "label": label, "available": True, "n_prompts_compared": len(common),
        "n_tokens_compared": total, "n_tokens_matched": matched,
        "token_identity_rate": (matched / total) if total else None,
        "n_sequences_with_any_flip": n_flipped,
    }


# ---------------------------------------------------------------------------
# wandb
# ---------------------------------------------------------------------------
def log_wandb(args, arm_recs: dict[str, dict[str, Any]], ledger: dict[str, Any]) -> str | None:
    if args.no_wandb:
        return None
    try:
        from scripts import wandb_logging
    except Exception as exc:  # noqa: BLE001
        print(f"[gap] wandb_logging import failed ({exc}); skipping", flush=True)
        return None
    try:
        run = wandb_logging.init_wandb_run(
            job_type="byteexact-realization-gap", agent="lawine",
            name=args.wandb_name or "lawine/realization-gap",
            group=args.wandb_group,
            tags=["byteexact-realization-gap", "pr523", "analysis-only"],
            config={
                "n_decodes": args.n_decodes, "num_prompts": args.num_prompts,
                "output_len": args.output_len, "seed": args.seed, "sigma_hw": SIGMA_HW,
                "arms": list(arm_recs), "analysis_only": True, "official_tps": 0,
                "strict_frontier_pred_457p5": STRICT_FRONTIER_PRED,
                "457p5_is_128x512_measured": False,
            },
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[gap] wandb init failed ({exc}); skipping", flush=True)
        return None
    if run is None:
        print("[gap] wandb disabled (no API key); skipping", flush=True)
        return None
    run_id = getattr(run, "id", None)
    try:
        for i, (name, rec) in enumerate(arm_recs.items()):
            mech = rec.get("mechanism") or {}
            st = (mech.get("steptime") or {}).get("exec", {})
            dr = (mech.get("steptime") or {}).get("draft", {})
            metrics = {
                f"arm/{name}/median_wall_tps": rec.get("median_wall_tps"),
                f"arm/{name}/wall_tps_std": rec.get("wall_tps_std"),
                f"arm/{name}/ppl": rec.get("ppl"),
                f"arm/{name}/peak_gpu_mem_mib": rec.get("peak_gpu_mem_mib"),
                f"arm/{name}/splitkv_redirects": mech.get("splitkv_redirects"),
                f"arm/{name}/server_ready_s": rec.get("server_ready_s"),
                f"arm/{name}/steptime_exec_gpu_ms": st.get("gpu_mean"),
                f"arm/{name}/steptime_exec_gap_ms": st.get("gap_mean"),
                f"arm/{name}/steptime_exec_cpu_ms": st.get("cpu_mean"),
                f"arm/{name}/steptime_draft_gpu_ms": dr.get("gpu_mean"),
            }
            metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            wandb_logging.log_event(run, f"arm_{name}", step=i, metrics=metrics)
        flat = {f"ledger/{k}": v for k, v in ledger.items() if isinstance(v, (int, float, bool))}
        wandb_logging.log_summary(run, flat, step=len(arm_recs))
        wandb_logging.log_json_artifact(
            run, name="realization_gap", artifact_type="byteexact-realization-gap",
            data={"arms": arm_recs, "ledger": ledger},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[gap] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:
            pass
    return run_id


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default="deployed,surgical,bx_T4_S64",
                    help="comma list subset of: " + ",".join(ARMS))
    ap.add_argument("--n-decodes", type=int, default=2)
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--seed", type=int, default=paths.SEED)
    ap.add_argument("--no-ppl", dest="do_ppl", action="store_false", default=True)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny serve+decode sanity (8 prompts x 16 tok, 1 decode, no ppl)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--tag", default="run", help="subdir under the harness root for this batch")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="byteexact-realization-gap")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    if args.smoke:
        args.num_prompts = min(args.num_prompts, 8)
        args.output_len = min(args.output_len, 16)
        args.n_decodes = 1
        args.do_ppl = False
        args.no_wandb = True

    for note in paths.prepare_local_gpu_env():
        print(f"[gap] {note}", flush=True)

    want = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in want if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {list(ARMS)}")

    # All arms share the byteexact submission's dep set (vLLM wheel etc); precache_kenyan
    # and surgical357 share the same wheel, so one server venv covers them all.
    base_manifest = harness.load_manifest(
        (ROOT / "submissions" / "fa2sw_strict_byteexact_splitkv399").resolve())
    server_python = harness.ensure_server_venv(base_manifest["dependencies"])
    print(f"[gap] server_python={server_python}", flush=True)

    out_dir = (args.out_dir or (OUT_ROOT / ("smoke" if args.smoke else args.tag))).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "arm_records.jsonl"
    print(f"[gap] arms={want} n_decodes={args.n_decodes} "
          f"workload={args.num_prompts}x{args.output_len} seed={args.seed} -> {out_dir}", flush=True)

    t0 = time.time()
    arm_recs: dict[str, dict[str, Any]] = {}
    with open(records_path, "w") as records_fh:
        for name in want:
            rec = run_arm(
                name, ARMS[name], server_python, out_dir,
                n_decodes=args.n_decodes, num_prompts=args.num_prompts,
                output_len=args.output_len, seed=args.seed,
                do_ppl=args.do_ppl, records_fh=records_fh,
            )
            arm_recs[name] = rec
    elapsed = time.time() - t0

    # cross-arm served byte-identity (corroboration): each byte-exact arm vs surgical
    diffs: dict[str, Any] = {}
    p = {n: Path(r["first_decode_out"]) if r.get("first_decode_out") else None
         for n, r in arm_recs.items()}
    if "surgical" in arm_recs:
        for name in arm_recs:
            if name != "surgical" and arm_recs[name].get("byte_exact_claim"):
                diffs[f"{name}_vs_surgical"] = cross_arm_token_diff(
                    p.get(name), p.get("surgical"), f"{name}_vs_surgical (expect identical)")

    ledger = build_ledger(arm_recs)
    try:
        from scripts import wandb_logging
        git = wandb_logging.git_info()
    except Exception:
        git = {}

    result = {
        "pr": 523, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "workload": {"num_prompts": args.num_prompts, "output_len": args.output_len,
                     "seed": args.seed, "n_decodes": args.n_decodes},
        "elapsed_s": elapsed, "git": git, "arms": arm_recs,
        "cross_arm_token_diffs": diffs, "ledger": ledger,
        "strict_frontier_457p5_provenance": {
            "457p5_is_128x512_measured": False,
            "source": "stark #466 strict_frontier_realize.json microbench projection: attention-kernel "
                      "added-us applied to the deployed 481.53 decode cycle at headline KV-len 640 "
                      "(realized_strict_frontier_tps=456.36). #488 surgical variant predicted 456.98.",
            "realized_when_served": "global VLLM_BATCH_INVARIANT=1 -> 234.47 official / 221.16 local "
                                    "(#487); surgical attn-only realization -> 357.6 local (#488). The "
                                    "457.5 over-promised the realized surgical serve by ~100 TPS.",
        },
    }
    run_id = log_wandb(args, arm_recs, ledger) if not args.smoke else None
    result["wandb_run_id"] = run_id
    result_path = out_dir / "realization_gap_result.json"
    result_path.write_text(json.dumps(result, indent=2, default=float))
    _print_final(ledger, diffs, elapsed, result_path, run_id)
    return 0


def build_ledger(arm_recs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Component-attributed served ledger across whichever arms ran."""
    def tps(n):
        r = arm_recs.get(n)
        return r.get("median_wall_tps") if r else None

    def steptime(n, kind, field):
        r = arm_recs.get(n)
        if not r:
            return None
        return ((r.get("mechanism") or {}).get("steptime") or {}).get(kind, {}).get(field)

    led: dict[str, Any] = {
        "deployed_local_tps": tps("deployed"),
        "surgical_local_tps": tps("surgical"),
        "byteexact_T4S64_local_tps": tps("bx_T4_S64"),
        "strict_frontier_457p5_pred": STRICT_FRONTIER_PRED,
        "457p5_is_128x512_measured": False,
    }
    # step deltas (local)
    if tps("surgical") and tps("bx_T4_S64"):
        led["delta_357_to_399_splitkv_recovery"] = tps("bx_T4_S64") - tps("surgical")
    if tps("bx_T4_S64") and tps("deployed"):
        led["delta_399_to_deployed_fixed_vs_adaptive_tax"] = tps("deployed") - tps("bx_T4_S64")
    if tps("surgical") and tps("deployed"):
        led["delta_surgical_to_deployed_total_attn_tax"] = tps("deployed") - tps("surgical")
    # steptime coarse split per arm (verify-GPU is where attention lives)
    led["steptime_exec_gpu_ms"] = {n: steptime(n, "exec", "gpu_mean") for n in arm_recs}
    led["steptime_exec_gap_ms"] = {n: steptime(n, "exec", "gap_mean") for n in arm_recs}
    led["steptime_draft_gpu_ms"] = {n: steptime(n, "draft", "gpu_mean") for n in arm_recs}
    # geometry sweep best
    sweep = {n: tps(n) for n in ("bx_T16_S16", "bx_T8_S32", "bx_T4_S64", "bx_T2_S128")
             if tps(n) is not None}
    if sweep:
        best = max(sweep, key=sweep.get)
        led["geometry_sweep_tps"] = sweep
        led["geometry_sweep_best_arm"] = best
        led["geometry_sweep_best_tps"] = sweep[best]
        if tps("bx_T4_S64"):
            led["geometry_sweep_best_over_packaged"] = sweep[best] - tps("bx_T4_S64")
    # cudagraph lever: drafter graph-replay benefit (ONEGRAPH=1 captured vs =0 eager K-iter)
    if tps("bx_T4_S64") and tps("bx_eager_drafter"):
        led["cudagraph_drafter_benefit_tps"] = tps("bx_T4_S64") - tps("bx_eager_drafter")
    # sampler lever: FlashInfer sampler on vs PyTorch-native (byteexact baseline)
    if tps("bx_T4_S64") and tps("bx_fisampler"):
        led["sampler_flashinfer_delta_tps"] = tps("bx_fisampler") - tps("bx_T4_S64")
    return led


def _print_final(ledger, diffs, elapsed, result_path, run_id):
    print(f"\n[gap] ================= LEDGER ({elapsed/60:.1f} min) =================", flush=True)
    for k, v in ledger.items():
        if not isinstance(v, dict):
            print(f"  {k:48s} = {v}", flush=True)
    for k, v in ledger.items():
        if isinstance(v, dict):
            print(f"  {k}:", flush=True)
            for kk, vv in v.items():
                print(f"      {kk:16s} = {vv}", flush=True)
    for k, d in diffs.items():
        if d.get("available"):
            print(f"  token-diff {k}: identity_rate={d.get('token_identity_rate')} "
                  f"flips_seqs={d.get('n_sequences_with_any_flip')}/{d.get('n_prompts_compared')}", flush=True)
    print(f"[gap] artifacts -> {result_path}  wandb_run_id={run_id}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
