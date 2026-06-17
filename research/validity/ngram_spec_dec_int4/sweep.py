"""PR #609 — draft-free ngram/prompt-lookup option-B lane on int4_g128_lmhead.

Serves the SHIPPED int4_g128_lmhead flags (byte-identical scratch wrapper) on the
rebuilt checkpoint and, per config, drives the official single-stream
decode_outputs workload to capture:

  * steady_gen_tps_mean  — vLLM's own "Avg generation throughput" (the honest
    local M=1 single-stream proxy; decode_outputs is strictly sequential).
  * E_accept / draft_acceptance_rate — vLLM's own server-log SpecDecoding
    counters (1 + K*accepted/drafted), with Prometheus /metrics as cross-check.
  * per-prompt completion_token_sha256 — greedy (temp=0, ignore_eos) token IDs
    for the #319 identity census.
  * warm probe_tps + peak GPU mem — secondary.

AR floor = SPECULATIVE_CONFIG="" (the shipped serve path). Official-TPS projection
is anchored on the AR floor: tau = 126.378 / steady_gen_tps_AR, applied to every
config. LOCAL A10G analysis-only — official_tps stays 0; no HF Job, no submission.

Usage:
  python sweep.py screen           # AR + 9 ngram configs, light workload
  python sweep.py census <plmax> <k>   # full 128x512 official decode, one config
  python sweep.py ar_census        # full 128x512 AR reference (+ ref-vs-ref floor)
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))
from scripts.local_validation import harness, paths, serve_profile  # noqa: E402

SCRATCH = ROOT / "research" / "validity" / "ngram_spec_dec_int4" / "serve_scratch"
CKPT = "/workspace/gemma_build/int4_g128_lmhead"
VENV = Path("/tmp/senpai-venvs/20f658587e8a6643/bin/python")
OUT = ROOT / "research" / "validity" / "ngram_spec_dec_int4" / "_sweep"

AR_OFFICIAL_TPS = 126.378  # plain-AR int4_g128_lmhead official baseline (W&B 905tbujn)
MTP_PROXY_TPS = 427.7      # int4_g128 + MTP-K7 candidate (W&B p7jo2ap4)

# Sweep axes from PR #609. prompt_lookup_min fixed at 2 (smoke-proven spelling;
# isolates the two swept axes — max and K). min<=max always holds (max>=2).
PLMAX = [2, 3, 4]
KSPEC = [3, 5, 7]
PLMIN = 2


def ngram_spec(plmax: int, k: int) -> dict[str, Any]:
    return {
        "method": "ngram",
        "num_speculative_tokens": k,
        "prompt_lookup_max": plmax,
        "prompt_lookup_min": min(PLMIN, plmax),
    }


def gpu_mem_used_mib() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def read_completion_shas(jsonl_path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: r["index"])
    return [
        {
            "index": r["index"], "id": r["id"], "dataset_index": r["dataset_index"],
            "completion_token_sha256": r["completion_token_sha256"],
            "num_completion_tokens": r["num_completion_tokens"],
        }
        for r in rows
    ]


def run_one(
    label: str, spec: dict[str, Any] | None, *, num_prompts: int, output_len: int,
    out_dir: Path, seed: int = paths.SEED,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"server_{label}.log"
    spec_json = json.dumps(spec) if spec else ""
    extra_env = {
        "MODEL_ID": CKPT,
        "SPECULATIVE_CONFIG": spec_json,
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "DISABLE_LOG_STATS": "0",
    }
    print(f"\n===== {label}: spec={spec_json or 'AR(none)'} "
          f"workload={num_prompts}x{output_len} seed={seed} =====", flush=True)
    rec: dict[str, Any] = {
        "label": label, "spec_config": spec, "num_prompts": num_prompts,
        "output_len": output_len, "seed": seed,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    t_boot = time.time()
    with harness.LocalServer(
        SCRATCH, server_python=VENV, port=8000, log_path=log_path,
        extra_env=extra_env, startup_timeout_s=1800,
    ) as srv:
        rec["boot_s"] = time.time() - t_boot
        decode_jsonl = out_dir / f"decode_{label}.jsonl"
        decode_summary = out_dir / f"decode_{label}.summary.json"
        t0 = time.time()
        summary = harness.capture_decode(
            VENV, base_url=srv.base_url, model=srv.served_model_name,
            out_file=decode_jsonl, summary_file=decode_summary,
            num_prompts=num_prompts, output_len=output_len, seed=seed, timeout_s=3600,
        )
        rec["decode_wall_s"] = time.time() - t0
        rec["decode_summary"] = summary
        rec["gpu_mem_used_mib"] = gpu_mem_used_mib()
        try:
            with urllib.request.urlopen(f"{srv.base_url}/metrics", timeout=30) as r:
                metrics_text = r.read().decode("utf-8", "replace")
            (out_dir / f"metrics_{label}.txt").write_text(metrics_text)
            rec["prom_spec"] = serve_profile.parse_spec_metrics(metrics_text)
        except Exception as exc:  # noqa: BLE001
            rec["prom_spec"] = {"error": str(exc)}
        try:
            rec["tps_probe"] = harness.probe_tps(srv.base_url, srv.served_model_name)
        except Exception as exc:  # noqa: BLE001
            rec["tps_probe"] = {"error": str(exc)}
    log_text = log_path.read_text()
    rec["spec_log"] = serve_profile.parse_spec_log(log_text)
    rec["completion_shas"] = read_completion_shas(decode_jsonl)
    rec["metrics"] = derive_metrics(rec)
    # Persist immediately so a later crash/timeout keeps finished configs.
    (out_dir / f"result_{label}.json").write_text(json.dumps(rec, indent=2))
    m = rec["metrics"]
    print(f"[{label}] proxy_tps={m['proxy_tps']:.2f} (src={m['proxy_src']}; "
          f"agg={m['decode_agg_tps']:.2f} steady={m['steady_gen_tps']} "
          f"probe={m['probe_tps']:.1f}) E_accept={m['e_accept']} "
          f"accept_rate={m['draft_acceptance_rate']} (src={m['accept_src']}) "
          f"boot={rec['boot_s']:.0f}s decode={rec['decode_wall_s']:.0f}s "
          f"mem={rec.get('gpu_mem_used_mib')}MiB", flush=True)
    return rec


def derive_metrics(rec: dict[str, Any]) -> dict[str, Any]:
    """Collapse the per-config sources into one comparable metric block.

    Speed proxy priority: vLLM's whole-run engine meter (steady_gen_tps, cleanest;
    excludes first-request warmup) when it logged >=3 intervals, else the
    self-contained decode aggregate (completion_tokens / decode_wall_s — always
    available; decode-dominated at 16x512). probe_tps (warm single burst) is a
    secondary upper bound only. E_accept/acceptance: Prometheus /metrics counters
    (reliable on this stack, even on tiny runs) first, server-log SpecDecoding
    counters as cross-check.
    """
    ds = rec.get("decode_summary") or {}
    sl = rec.get("spec_log") or {}
    pr = rec.get("prom_spec") or {}
    tp = rec.get("tps_probe") or {}
    ct = ds.get("num_completion_tokens") or 0
    dur = ds.get("duration_s") or rec.get("decode_wall_s") or 0.0
    decode_agg = (ct / dur) if dur else float("nan")
    steady = sl.get("steady_gen_tps_mean")
    steady_n = sl.get("steady_gen_tps_n") or 0
    probe = tp.get("decode_tps_single_stream") if isinstance(tp, dict) else None
    if steady and steady_n >= 3:
        proxy, src = steady, "steady_gen_tps"
    else:
        proxy, src = decode_agg, "decode_aggregate"
    # acceptance (None for AR): Prometheus first, then server log.
    e_acc = pr.get("e_accept_mean_acceptance_length")
    acc_rate = pr.get("draft_acceptance_rate")
    acc_src = "prometheus"
    if e_acc is None and sl.get("e_accept_exact"):
        e_acc, acc_rate, acc_src = sl.get("e_accept_exact"), sl.get("draft_acceptance_rate"), "server_log"
    return {
        "proxy_tps": proxy, "proxy_src": src,
        "decode_agg_tps": decode_agg, "steady_gen_tps": steady,
        "steady_gen_tps_n": steady_n, "probe_tps": probe or float("nan"),
        "e_accept": e_acc, "draft_acceptance_rate": acc_rate, "accept_src": acc_src,
        "num_drafts": pr.get("num_drafts"), "num_accepted_tokens": pr.get("num_accepted_tokens"),
        "num_draft_tokens": pr.get("num_draft_tokens"),
    }


def cmd_screen() -> int:
    out_dir = OUT / "screen"
    np_s, ol_s = 16, 512
    results: list[dict[str, Any]] = []
    # AR floor first (anchors tau + identity reference).
    results.append(run_one("ar", None, num_prompts=np_s, output_len=ol_s, out_dir=out_dir))
    for plmax in PLMAX:
        for k in KSPEC:
            lbl = f"ng_max{plmax}_k{k}"
            results.append(run_one(lbl, ngram_spec(plmax, k),
                                   num_prompts=np_s, output_len=ol_s, out_dir=out_dir))
    (out_dir / "screen_all.json").write_text(json.dumps(results, indent=2))
    # Projection anchored on AR: proj_official = AR_OFFICIAL_TPS * (S_cfg / S_AR),
    # using the SAME proxy for AR and every config (so the relative Pareto is
    # proxy-choice-invariant). tau = AR_OFFICIAL_TPS / S_AR falls out.
    ar = next(r for r in results if r["label"] == "ar")
    s_ar = ar["metrics"]["proxy_tps"]
    proxy_src = ar["metrics"]["proxy_src"]
    tau = AR_OFFICIAL_TPS / s_ar if s_ar else None
    print(f"\n========== SCREEN PARETO (tau={tau:.4f} from S_AR={s_ar:.2f}, "
          f"proxy={proxy_src}) ==========", flush=True)
    print(f"{'config':18s} {'S_local':>8s} {'proj_TPS':>9s} {'vsAR%':>7s} "
          f"{'E_acc':>6s} {'acc_rt':>7s}", flush=True)
    pareto = []
    for r in results:
        m = r["metrics"]
        s = m["proxy_tps"]
        proj = tau * s if (tau and s) else None
        vs_ar = (100.0 * (s - s_ar) / s_ar) if (s and s_ar) else None
        pareto.append({"label": r["label"], "spec_config": r["spec_config"],
                       "proxy_tps": s, "proxy_src": m["proxy_src"],
                       "decode_agg_tps": m["decode_agg_tps"], "steady_gen_tps": m["steady_gen_tps"],
                       "probe_tps": m["probe_tps"], "proj_official_tps": proj,
                       "vs_ar_pct": vs_ar, "e_accept": m["e_accept"],
                       "draft_acceptance_rate": m["draft_acceptance_rate"],
                       "gpu_mem_used_mib": r.get("gpu_mem_used_mib")})
        print(f"{r['label']:18s} {s or 0:8.2f} {proj or 0:9.2f} "
              f"{vs_ar if vs_ar is not None else 0:7.1f} "
              f"{m['e_accept'] or 0:6.3f} {m['draft_acceptance_rate'] or 0:7.3f}", flush=True)
    summary = {"tau": tau, "s_ar_local": s_ar, "proxy_src": proxy_src,
               "ar_official_tps": AR_OFFICIAL_TPS, "mtp_proxy_tps": MTP_PROXY_TPS,
               "plmin": PLMIN, "pareto": pareto}
    (out_dir / "pareto.json").write_text(json.dumps(summary, indent=2))
    print(f"\nartifacts -> {out_dir}", flush=True)
    return 0


def cmd_census(plmax: int, k: int) -> int:
    out_dir = OUT / "census"
    lbl = f"ng_max{plmax}_k{k}"
    run_one(lbl, ngram_spec(plmax, k), num_prompts=paths.NUM_PROMPTS,
            output_len=paths.OUTPUT_LEN, out_dir=out_dir, seed=paths.SEED)
    return 0


def cmd_ar_census() -> int:
    out_dir = OUT / "census"
    # Two AR runs at the SAME seed to expose any cross-start nondeterminism floor
    # (dev307 control): ar_ref vs ar_ref2 sha mismatch count is the noise floor the
    # ngram-vs-AR comparison must clear.
    run_one("ar_ref", None, num_prompts=paths.NUM_PROMPTS, output_len=paths.OUTPUT_LEN,
            out_dir=out_dir, seed=paths.SEED)
    run_one("ar_ref2", None, num_prompts=paths.NUM_PROMPTS, output_len=paths.OUTPUT_LEN,
            out_dir=out_dir, seed=paths.SEED)
    return 0


def main() -> int:
    for note in paths.prepare_local_gpu_env():
        print(f"[env] {note}", flush=True)
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    mode = sys.argv[1]
    if mode == "screen":
        return cmd_screen()
    if mode == "census":
        return cmd_census(int(sys.argv[2]), int(sys.argv[3]))
    if mode == "ar_census":
        return cmd_ar_census()
    print(f"unknown mode: {mode}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
