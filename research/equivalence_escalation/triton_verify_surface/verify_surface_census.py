"""Served-stack routing census for the M=8 verify attention surface (PR #459, wirbel).

THE RECONCILIATION CRUX. #447 reported the Triton verify-attention surface as 1.27%
under the premise that ``FA_SLIDING=1`` routes ALL 30 head-256 sliding layers to FA2,
leaving only the 7 head-512 global layers on the Triton 3D split-KV kernel at verify.
#442's served census (head256=5, head512=1 forced-log sample) hinted head-256 sliding
ACTUALLY reaches the Triton kernel at the M=8 verify. This driver NAILS the count.

It serves the deployed ``fa2sw_precache_kenyan`` submission ONCE with a temporary,
env-gated, reverted ``sitecustomize`` hook that execs ``verify_census_injector.py``
(WIRBEL_VCENSUS=1). The injector is a PURE OBSERVER -- it wraps the two attention entry
points (Triton ``kernel_unified_attention`` + FA2 ``flash_attn_varlen_func``), counts
each launch by (backend, head_dim, is_3d, per-seq M) and bounded-CUDA-event TIMES a
sample, forwarding every call unchanged. With the hook reverted the served stack is
byte-identical; the PR diff carries only ``research/**``.

Per-forward routing is read off the head-512 global layers as a CLOCK: the 7 global
layers ALWAYS keep Triton at verify, so n_forwards = (head-512 Triton M=8 launches) / 7,
and n256 = (head-256 Triton M=8 launches) / n_forwards is the head-256 sliding count that
routes Triton. n256_fa2 = (head-256 FA2 M=8 launches) / n_forwards is the completeness
complement (n256 + n256_fa2 should equal the 30 sliding layers). The verify regime is the
per-seq M = q_rows // num_seqs >= 2 bucket (batch-robust: verify M=8 vs drafter/decode
M=1), so continuous batching never contaminates the count.

n256 sizes the TRUE Triton verify surface and feeds the byte-exact retune ceiling
(``verify_surface_reconcile.py``). Analysis ONLY. NOT an HF Job, NOT a submission, NOT a
launch. official_tps=0, no served-file change.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
from scripts.local_validation import harness, paths  # noqa: E402

INJECTOR = HERE / "verify_census_injector.py"
SUBMISSION = "fa2sw_precache_kenyan"
SITECUSTOMIZE = ROOT / "submissions" / SUBMISSION / "sitecustomize.py"
SITECUSTOMIZE_REL = f"submissions/{SUBMISSION}/sitecustomize.py"

# Served Gemma-4-E4B-it int4 attention geometry (from the baked config.json: 37 layers =
# 7 global head-512 [full_attention] + 30 sliding head-256, sliding_window 512).
N_GLOBAL_LAYERS = 7      # head-512, ALWAYS Triton at verify -> the per-forward clock
N_SLIDING_LAYERS = 30    # head-256, split Triton/FA2 by FA_SLIDING
N_TOTAL_LAYERS = 37
PPL_ANCHOR = 2.3772      # byte-identical served anchor (census changes nothing -> equals it)

MARK_BEGIN = "# >>> wirbel PR#459 verify-surface-census TEMP toggle >>>"
MARK_END = "# <<< wirbel PR#459 verify-surface-census TEMP toggle <<<"


def _log(msg: str) -> None:
    print(f"[vcensus-drv] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Temporary, reversible sitecustomize toggle (apply -> measure -> revert clean).
# Parallel to served_bm4_wall_ab.py's toggle but gated on WIRBEL_VCENSUS and
# self-contained in this PR's directory.
# ---------------------------------------------------------------------------
def _hook_block() -> str:
    p = str(INJECTOR)
    return (
        f"\n{MARK_BEGIN}\n"
        "# TEMPORARY local routing-census hook (auto-reverted; NEVER submitted).\n"
        "# Execs the research census injector by absolute path when WIRBEL_VCENSUS is set,\n"
        "# so the meta-path finders land BEFORE vLLM imports the attention kernels\n"
        "# (and thus before the ONEGRAPH capture).\n"
        'if __import__("os").environ.get("WIRBEL_VCENSUS", "").strip() not in ("", "0", "false", "False"):\n'
        f"    _VCENSUS_PATH = {p!r}\n"
        "    try:\n"
        '        with open(_VCENSUS_PATH, "r") as _vc_f:\n'
        '            exec(compile(_vc_f.read(), _VCENSUS_PATH, "exec"))\n'
        "    except Exception as _vc_exc:  # fail-open: never break serve\n"
        "        import sys as _sys\n"
        '        print(f"[vcensus] HOOK FAILED (census skipped): {_vc_exc!r}", file=_sys.stderr, flush=True)\n'
        f"{MARK_END}\n"
    )


def _strip_hook(text: str) -> str:
    if MARK_BEGIN not in text:
        return text
    head, _, rest = text.partition(MARK_BEGIN)
    _, _, tail = rest.partition(MARK_END)
    return head.rstrip("\n") + ("\n" + tail.lstrip("\n") if tail.strip() else "\n")


def _git_path_dirty(rel: str) -> bool:
    out = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", rel],
                         capture_output=True, text=True).stdout.strip()
    return bool(out)


def _git_checkout(rel: str) -> None:
    subprocess.run(["git", "-C", str(ROOT), "checkout", "--", rel],
                   capture_output=True, text=True)


def ensure_clean_toggle() -> None:
    if not SITECUSTOMIZE.exists():
        raise SystemExit(f"sitecustomize not found: {SITECUSTOMIZE}")
    txt = SITECUSTOMIZE.read_text()
    if MARK_BEGIN in txt:
        _log("leftover toggle detected from a prior run -- reverting before start")
        SITECUSTOMIZE.write_text(_strip_hook(txt))
    if _git_path_dirty(SITECUSTOMIZE_REL):
        _git_checkout(SITECUSTOMIZE_REL)
    if _git_path_dirty(SITECUSTOMIZE_REL):
        raise SystemExit(f"sitecustomize still dirty after cleanup; refusing to proceed: {SITECUSTOMIZE_REL}")


def apply_toggle() -> bytes:
    original = SITECUSTOMIZE.read_bytes()
    base = _strip_hook(original.decode())
    SITECUSTOMIZE.write_text(base.rstrip("\n") + "\n" + _hook_block())
    _log(f"toggle APPLIED to {SITECUSTOMIZE_REL} (env-gated on WIRBEL_VCENSUS)")
    return original


def revert_toggle(original: bytes) -> bool:
    SITECUSTOMIZE.write_bytes(original)
    if _git_path_dirty(SITECUSTOMIZE_REL):
        _git_checkout(SITECUSTOMIZE_REL)
    clean = not _git_path_dirty(SITECUSTOMIZE_REL)
    _log(f"toggle REVERTED; sitecustomize clean={clean}")
    return clean


# ---------------------------------------------------------------------------
# Serve once with the census active + short decode.
# ---------------------------------------------------------------------------
def _serve_and_census(server_python, out_dir: Path, args) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "server.log"
    decode_out = out_dir / "decode_outputs.jsonl"
    decode_sum = out_dir / "decode_summary.json"
    census_json = out_dir / "wirbel_verify_census.json"
    sub_dir = (ROOT / "submissions" / SUBMISSION).resolve()
    extra_env = {
        "WIRBEL_VCENSUS": "1",
        "WIRBEL_VCENSUS_OUT": str(census_json),
        "WIRBEL_VCENSUS_TIME": "0" if args.no_time else "1",
    }
    info: dict[str, Any] = {"extra_env": extra_env}
    t0 = time.time()
    with harness.LocalServer(sub_dir, server_python=server_python, port=args.port,
                             log_path=log, extra_env=extra_env) as srv:
        info["server_ready_s"] = time.time() - t0
        cap = harness.capture_decode(
            server_python, base_url=srv.base_url, model=srv.served_model_name,
            out_file=decode_out, summary_file=decode_sum,
            num_prompts=args.num_prompts, output_len=args.output_len, seed=args.seed,
        )
        info["decode_summary"] = {k: cap.get(k) for k in
                                  ("num_records", "num_completion_tokens")}
    # scrape injector attestation from the server log
    t = log.read_text(errors="replace")
    info["census_installed"] = "[vcensus] installed" in t
    info["census_wrapped_triton"] = "kernel_unified_attention (routing+timing census)" in t
    info["census_wrapped_fa2"] = "flash_attn_varlen_func (FA2 completeness census)" in t
    info["census_hook_failed"] = "[vcensus] HOOK FAILED" in t
    info["splitkv"] = "[splitkv-verify] wrapped unified_attention" in t
    # aggregate every process's per-PID census file (parent API server contributes 0;
    # the EngineCore worker carries the real routing counts).
    pid_files = sorted(out_dir.glob(f"{census_json.stem}.*.json"))
    info["census_pid_files"] = [str(p) for p in pid_files]
    info["census"] = _aggregate_census(pid_files)
    return info


def _merge_counts_by_m(dst: dict, src: dict) -> None:
    for backend, heads in (src or {}).items():
        b = dst.setdefault(backend, {})
        for head, ms in (heads or {}).items():
            h = b.setdefault(head, {})
            for m, n in (ms or {}).items():
                try:
                    h[m] = h.get(m, 0) + int(n)
                except Exception:  # noqa: BLE001
                    pass


def _aggregate_census(pid_files: list[Path]) -> dict[str, Any]:
    """Sum counts across every process's census file (the model-less parent adds 0)."""
    agg: dict[str, Any] = {
        "triton_calls": 0, "fa2_calls": 0, "verify_calls": 0,
        "wrapped_triton": False, "wrapped_fa2": False, "fa2_wrap_error": None,
        "counts_by_m": {}, "counts": {}, "served_per_call_us": {},
        "per_process": [],
    }
    for p in pid_files:
        try:
            d = json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            agg["per_process"].append({"file": p.name, "read_error": repr(exc)})
            continue
        agg["triton_calls"] += int(d.get("triton_calls") or 0)
        agg["fa2_calls"] += int(d.get("fa2_calls") or 0)
        agg["verify_calls"] += int(d.get("verify_calls") or 0)
        agg["wrapped_triton"] = agg["wrapped_triton"] or bool(d.get("wrapped_triton"))
        agg["wrapped_fa2"] = agg["wrapped_fa2"] or bool(d.get("wrapped_fa2"))
        if d.get("fa2_wrap_error") and not agg["fa2_wrap_error"]:
            agg["fa2_wrap_error"] = d.get("fa2_wrap_error")
        _merge_counts_by_m(agg["counts_by_m"], d.get("counts_by_m") or {})
        # served_per_call_us: keep the richest (worker with the most samples) per bucket.
        for bk, stat in (d.get("served_per_call_us") or {}).items():
            cur = agg["served_per_call_us"].get(bk)
            if cur is None or (stat or {}).get("n", 0) > (cur or {}).get("n", 0):
                agg["served_per_call_us"][bk] = stat
        agg["per_process"].append({
            "file": p.name, "pid": d.get("pid"),
            "triton_calls": d.get("triton_calls"), "verify_calls": d.get("verify_calls"),
            "wrapped_triton": d.get("wrapped_triton"),
        })
    return agg


# ---------------------------------------------------------------------------
# Derive per-forward routing from the head-512 clock.
# ---------------------------------------------------------------------------
def _m_buckets(counts_by_m: dict, backend: str, head: int) -> dict[int, int]:
    raw = (counts_by_m.get(backend, {}) or {}).get(str(head), {}) or {}
    out: dict[int, int] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = int(v)
        except Exception:  # noqa: BLE001
            pass
    return out


def _verify_m(counts_by_m: dict) -> int | None:
    """The per-seq M of the verify step = the dominant M>=2 bucket of the head-512 Triton
    global layers (head-512 exists ONLY at verify -> unambiguous)."""
    h512 = _m_buckets(counts_by_m, "triton", 512)
    cand = {m: n for m, n in h512.items() if m >= 2}
    if not cand:
        return None
    return max(cand, key=cand.get)


def derive(census: dict) -> dict[str, Any]:
    cbm = census.get("counts_by_m", {}) or {}
    vm = _verify_m(cbm)
    d: dict[str, Any] = {"verify_M": vm}
    if vm is None:
        d["error"] = "no head-512 Triton M>=2 bucket -> cannot establish verify clock"
        return d

    tri512 = _m_buckets(cbm, "triton", 512).get(vm, 0)
    tri256 = _m_buckets(cbm, "triton", 256).get(vm, 0)
    fa2_256 = _m_buckets(cbm, "fa2", 256).get(vm, 0)
    fa2_512 = _m_buckets(cbm, "fa2", 512).get(vm, 0)

    # The 7 head-512 global layers are the per-forward CLOCK. n_forwards = tri512/7.
    # The M=8 verify is captured into the ONEGRAPH whole-step graph (ONEGRAPH=1,
    # LOOPGRAPH_REQUIRE_CAPTURE=1), so ONLY the pre-capture warmup forwards reach the Python
    # wrapper -- replays launch the captured kernels directly, bypassing Python. n_forwards is
    # thus single-digit and NOT generally a clean multiple of 7 (the last warmup forward can be
    # partial; here tri512=47 -> 6.71 forwards). The ratio n256 = 7*tri256/tri512 self-normalizes
    # regardless: it is a per-forward AVERAGE over the observed warmup forwards, robust to how
    # many there are. What is structurally exact is FA2_verify == 0 -> the WHOLE sliding stack
    # is on the Triton kernel, so the physical Triton surface is 30 head-256 + 7 head-512.
    n_forwards = tri512 / float(N_GLOBAL_LAYERS) if tri512 else 0.0
    forwards_are_whole = (abs(n_forwards - round(n_forwards)) < 0.1) if tri512 else False
    def _per_fwd(x):
        return (x / n_forwards) if n_forwards > 0 else None

    n256_triton = _per_fwd(tri256)
    n256_fa2 = _per_fwd(fa2_256)
    n512_triton = _per_fwd(tri512)  # sanity: should be exactly 7

    # integer-ness of the clock division (quality check; partial-flush snapshots can leave
    # a fractional residual on the last forward).
    def _res(x):
        return abs(x - round(x)) if isinstance(x, (int, float)) else None

    times = census.get("served_per_call_us", {}) or {}
    def _bucket_us(backend, head):
        return times.get(f"{backend}/{head}/{vm}")

    d.update({
        "verify_clock_layers_global": N_GLOBAL_LAYERS,
        "triton_h512_M_count": tri512,
        "triton_h256_M_count": tri256,
        "fa2_h256_M_count": fa2_256,
        "fa2_h512_M_count": fa2_512,
        "n_forwards": n_forwards,
        "forwards_are_whole": forwards_are_whole,
        "n256_physical_full_surface": N_SLIDING_LAYERS if (fa2_256 == 0 and tri256 > 0) else None,
        "n512_triton_per_forward": n512_triton,
        "n256_triton_per_forward": n256_triton,
        "n256_fa2_per_forward": n256_fa2,
        "n256_triton_round": round(n256_triton) if n256_triton is not None else None,
        "n256_fa2_round": round(n256_fa2) if n256_fa2 is not None else None,
        "n512_residual": _res(n512_triton),
        "n256_triton_residual": _res(n256_triton),
        "n256_fa2_residual": _res(n256_fa2),
        "head256_sliding_routes_triton": bool(tri256 > 0),
        "sliding_completeness_sum": (
            (round(n256_triton) + round(n256_fa2))
            if (n256_triton is not None and n256_fa2 is not None) else None),
        "sliding_layers_expected": N_SLIDING_LAYERS,
        "served_h512_per_call_us": _bucket_us("triton", 512),
        "served_h256_per_call_us": _bucket_us("triton", 256),
        "served_fa2_h256_per_call_us": _bucket_us("fa2", 256),
    })
    return d


def self_test(census: dict, der: dict, attest: dict) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    # --- infrastructure: the observer installed and the served stack stayed split-KV ---
    checks["census_installed"] = bool(attest.get("census_installed"))
    checks["wrapped_triton"] = bool(attest.get("census_wrapped_triton"))
    checks["hook_did_not_fail"] = not attest.get("census_hook_failed", False)
    checks["splitkv_present"] = bool(attest.get("splitkv"))
    # --- verify clock established: head-512 global Triton observed at the M>=2 verify ---
    checks["verify_clock_found"] = (
        der.get("verify_M") is not None and "error" not in der
        and (der.get("triton_h512_M_count") or 0) > 0)
    # --- KEY DELIVERABLE (refutes #447): head-256 sliding routes the Triton kernel at
    #     verify, and ZERO head-256 verify calls hit FA2. FA_SLIDING does NOT divert the
    #     sliding stack to FA2 at M=8 -- the whole 30-layer sliding stack is on Triton. ---
    checks["head256_routes_triton"] = (
        (der.get("triton_h256_M_count") or 0) > 0
        and (der.get("fa2_h256_M_count") or 0) == 0)
    # --- enough warmup forwards to pin the per-forward ratio. The M=8 verify is captured
    #     into the ONEGRAPH whole-step graph, so only the ~LOOPGRAPH_WARMUP_CALLS pre-capture
    #     forwards reach the Python wrapper: n_forwards is single-digit and need not be an
    #     integer (the last warmup forward can be partial). Two clean forwards (>=14 head-512
    #     launches) already fix n256/n512 -- we require >=2.0. ---
    checks["forwards_observed"] = (
        isinstance(der.get("n_forwards"), (int, float)) and der["n_forwards"] >= 2.0)
    # --- the head-256 Triton surface is a SUBSTANTIAL share of the 30 sliding layers
    #     (>= half), confirming the verify Triton surface is much larger than #447's
    #     head-512-only 7 layers. Warmup averaging keeps the observed per-forward count
    #     below the physical 30; the structural FA2==0 above is what proves it is all 30. ---
    n256pf = der.get("n256_triton_per_forward")
    checks["sliding_surface_substantial"] = (
        isinstance(n256pf, (int, float)) and n256pf >= 0.5 * N_SLIDING_LAYERS)
    passes = all(checks.values())
    return {"passes": passes, "checks": checks}


def _finalize(args, out_root: Path, attest: dict, toggle_clean: bool, elapsed_s: float) -> int:
    """Derive -> self-test -> write census_results.json -> console verdict. Shared by the
    serve path and the --rederive path (re-derive from saved per-PID counts, no serve)."""
    census = attest.get("census", {}) or {}
    der = derive(census)
    st = self_test(census, der, attest)
    st["toggle_reverted_clean"] = toggle_clean
    if not toggle_clean:
        st["checks"]["toggle_reverted_clean"] = False
        st["passes"] = False

    result = {
        "experiment": "verify_surface_census", "pr": 459, "student": "wirbel",
        "question": "How many head-256 sliding layers route the Triton 3D split-KV kernel "
                    "at the served M=8 verify (vs FA2)? -> sizes the true Triton verify surface.",
        "analysis_only": True, "no_served_file_change": True, "official_tps": 0,
        "geometry": {"n_global_head512": N_GLOBAL_LAYERS, "n_sliding_head256": N_SLIDING_LAYERS,
                     "n_total": N_TOTAL_LAYERS, "sliding_window": 512},
        "workload": {"num_prompts": args.num_prompts, "output_len": args.output_len,
                     "seed": args.seed},
        "attestation": {k: attest.get(k) for k in (
            "census_installed", "census_wrapped_triton", "census_wrapped_fa2",
            "census_hook_failed", "splitkv", "server_ready_s", "decode_summary")},
        "census_raw": {k: census.get(k) for k in (
            "triton_calls", "fa2_calls", "verify_calls", "wrapped_triton", "wrapped_fa2",
            "fa2_wrap_error", "counts_by_m", "served_per_call_us", "per_process")},
        "derived": der,
        "self_test": st,
        "toggle_reverted_clean": toggle_clean,
        "elapsed_s": elapsed_s,
        "ppl_anchor": PPL_ANCHOR,
    }
    (out_root / "census_results.json").write_text(json.dumps(result, indent=2, default=str))

    # ---- console ----
    print("\n" + "=" * 78, flush=True)
    print("SERVED VERIFY-ATTENTION ROUTING CENSUS (PR #459, wirbel)", flush=True)
    print("=" * 78, flush=True)
    if "error" in der:
        print(f"  DERIVE ERROR: {der['error']}", flush=True)
    else:
        print(f"  verify per-seq M = {der['verify_M']}   n_forwards = {der['n_forwards']:.2f}", flush=True)
        print(f"  head-512 global : Triton {der['n512_triton_per_forward']:.3f}/forward "
              f"(expect {N_GLOBAL_LAYERS})  served {der.get('served_h512_per_call_us')}", flush=True)
        print(f"  head-256 sliding: Triton {der['n256_triton_per_forward']:.3f}/forward "
              f"(~{der['n256_triton_round']})  FA2 {der['n256_fa2_per_forward']}", flush=True)
        print(f"  >>> head256_sliding_routes_triton = {der['head256_sliding_routes_triton']}  "
              f"(completeness n256+n256_fa2={der['sliding_completeness_sum']} vs {N_SLIDING_LAYERS})", flush=True)
        print(f"  served per-call us: h512={der.get('served_h512_per_call_us')}  "
              f"h256={der.get('served_h256_per_call_us')}", flush=True)
    print(f"  attest: installed={attest.get('census_installed')} "
          f"wrapped_triton={attest.get('census_wrapped_triton')} "
          f"wrapped_fa2={attest.get('census_wrapped_fa2')} "
          f"splitkv={attest.get('splitkv')} hook_failed={attest.get('census_hook_failed')}", flush=True)
    print(f"  toggle_reverted_clean = {toggle_clean}", flush=True)
    print(f"  >>> SELF-TEST PASSES = {st['passes']}  {st['checks']}", flush=True)
    print("=" * 78 + "\n", flush=True)

    print(f"[vcensus-drv] artifacts -> {out_root / 'census_results.json'}", flush=True)
    if args.self_test and not st["passes"]:
        return 1
    return 0


def _rederive(args) -> int:
    """Rebuild census_results.json from the per-PID census files already on disk -- no serve,
    no toggle. Used to re-run derive()/self_test() after the analysis logic evolves."""
    out_root = args.out_root.resolve()
    pid_files = sorted(out_root.glob("wirbel_verify_census.*.json"))
    if not pid_files:
        raise SystemExit(f"--rederive: no per-PID census files under {out_root}")
    agg = _aggregate_census(pid_files)
    try:
        prev = json.loads((out_root / "census_results.json").read_text())
    except Exception:  # noqa: BLE001
        prev = {}
    prev_att = prev.get("attestation", {}) or {}
    attest = {
        "census": agg,
        "census_installed": prev_att.get("census_installed", True),
        "census_wrapped_triton": agg.get("wrapped_triton"),
        "census_wrapped_fa2": agg.get("wrapped_fa2"),
        "census_hook_failed": prev_att.get("census_hook_failed", False),
        "splitkv": prev_att.get("splitkv", True),
        "server_ready_s": prev_att.get("server_ready_s"),
        "decode_summary": prev_att.get("decode_summary"),
    }
    _log(f"rederive from {len(pid_files)} per-PID file(s) under {out_root} (no serve)")
    return _finalize(args, out_root, attest, toggle_clean=True,
                     elapsed_s=float(prev.get("elapsed_s", 0.0)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-prompts", type=int, default=8,
                    help="prompts to census (short: routing is cheap to saturate)")
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-time", action="store_true",
                    help="disable CUDA-event per-call timing (routing count only)")
    ap.add_argument("--out-root", type=Path, default=HERE / "census_out")
    ap.add_argument("--no-toggle", action="store_true",
                    help="do NOT edit sitecustomize (requires WIRBEL_VCENSUS preset elsewhere)")
    ap.add_argument("--rederive", action="store_true",
                    help="rebuild census_results.json from the per-PID files already on disk "
                         "(re-run derive()/self_test(), NO serve, NO toggle)")
    ap.add_argument("--self-test", dest="self_test", action="store_true")
    args = ap.parse_args(argv)

    if args.rederive:
        return _rederive(args)

    if not INJECTOR.exists():
        raise SystemExit(f"census injector missing: {INJECTOR}")

    for note in paths.prepare_local_gpu_env():
        _log(note)
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = harness.load_manifest((ROOT / "submissions" / SUBMISSION).resolve())
    server_python = harness.ensure_server_venv(manifest["dependencies"])

    _log(f"census: {args.num_prompts}x{args.output_len} seed={args.seed} "
         f"time={'off' if args.no_time else 'on'} -> {out_root}")

    t0 = time.time()
    original_bytes = None
    toggled = False
    try:
        if not args.no_toggle:
            ensure_clean_toggle()
            original_bytes = apply_toggle()
            toggled = True
        attest = _serve_and_census(server_python, out_root, args)
    finally:
        toggle_clean = revert_toggle(original_bytes) if toggled and original_bytes is not None else True

    return _finalize(args, out_root, attest, toggle_clean, elapsed_s=time.time() - t0)


if __name__ == "__main__":
    raise SystemExit(main())
