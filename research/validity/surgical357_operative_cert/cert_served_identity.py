"""PR #494 — Local operative-1.0 cert (Census 1): served packaged surgical-357 stack.

LOCAL ONLY. ``analysis_only=true``, ``official_tps=0``. No HF job, no submission,
no change to the deployed served file. Serves the PACKAGED submission
``fa2sw_strict_surgical357`` exactly as an official job would (its own manifest +
serve.py + sitecustomize), so the surgical lever fires through the gated
``sitecustomize -> surgical_attn_patch.py`` meta-path finder.

Why this proves the PACKAGE (not a venv edit): the serve-venv wheel's
``triton_unified_attention.py`` is STOCK on this pod (line 34
``is_batch_invariant = envs.VLLM_BATCH_INVARIANT``; no ``SURGICAL_*`` gate). The
manifest sets ``SURGICAL_ATTN_USE_3D_OFF=1`` and does NOT set
``VLLM_BATCH_INVARIANT``, so the ONLY thing that can flip the attention module's
``is_batch_invariant`` to True is my packaged ``surgical_attn_patch.py``. The
server log must show it arm + force the flag; the 222 matmul tax must NOT be
installed (TPS ~357, not ~222).

Census 1 (served-vs-served matched-config self-determinism):
    3 back-to-back warm decodes of the same packaged config + seed. round1-vs-round2
    token identity must be 1.000 (the lever's 2D order-preserving sequential-KV path
    is byte-deterministic run-to-run -- that is the whole point of the lever).

Speed sanity (PR ask: 1-2 warm decodes, not a fresh full benchmark):
    median warm wall_tps (target ~357), PPL (target 2.3767), completion (128/128).

The sibling logit-margin census (#461-style, 0 semantic flips vs the 222 config)
is the ``deployed_flip_attribution.py`` orchestration -- its ``attn_only`` arm pins
the SAME attention ``is_batch_invariant=True`` this package installs.

Run under the repo .venv (has wandb); the serve/decode subprocs use the cached
submission serve venv::

    .venv/bin/python -m research.validity.surgical357_operative_cert.cert_served_identity \
        --n-decodes 3 --wandb-name stark/surgical357-served-cert
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

OUT_ROOT = ROOT / "research" / "validity" / "surgical357_operative_cert"
SUBMISSION = "fa2sw_strict_surgical357"

# Provenance anchors (lawine #488 ko01dcyy, the measured rung this packages).
SURGICAL_TARGET_TPS = 357.64
STRICT_FLOOR_222 = 222.0
PPL_TARGET = 2.3767
PPL_GATE = 2.42
SIGMA_HW = 4.864


def grep_log(log_path: Path) -> dict[str, Any]:
    """Pull lever-mechanism signals out of the packaged stack's server log.

    The lever is proven fired by TWO surgical-attn lines (armed = finder
    registered at sitecustomize time; forced = flag flipped when vLLM imported
    the ops module) AND by ``splitkv_redirects == 0`` (the 2D order-preserving
    path was taken, so the spec-verify 3D split-KV redirect short-circuited).
    ``onegraph_captured`` confirms forcing use_3d=False did not break CUDA-graph
    capture. The matmul tax being OFF is proven by TPS (~357 not ~222), not the
    log, but we still confirm ``VLLM_BATCH_INVARIANT`` was never set.
    """
    out = {
        "surgical_armed": False,
        "surgical_forced_true": False,
        "splitkv_armed": False,
        "splitkv_redirects": 0,
        "onegraph_captured": False,
        "fatal_traceback": False,
        "n_tracebacks": 0,
        "benign_usage_tracebacks": 0,
        "batch_invariant_mentions": 0,
        "init_batch_invariance_ran": False,
    }
    try:
        text = Path(log_path).read_text(errors="replace")
    except OSError:
        return out
    out["surgical_armed"] = "[surgical-attn] armed" in text
    out["surgical_forced_true"] = ("[surgical-attn] forced" in text) and (
        "is_batch_invariant=True" in text
    )
    out["splitkv_armed"] = ("[splitkv-verify] wrapped" in text) or (
        "[splitkv-verify] armed" in text
    )
    out["splitkv_redirects"] = text.count("-> 3D split-KV")
    out["onegraph_captured"] = "[onegraph] captured" in text
    # The matmul tax is installed only by init_batch_invariance(); it must NEVER
    # run here (we never set VLLM_BATCH_INVARIANT). If it ran, the tax is on.
    out["init_batch_invariance_ran"] = (
        "init_batch_invariance" in text and "Activating batch invariant" in text
    )
    n_tb = text.count("Traceback (most recent call last)")
    n_usage = text.count("_report_usage_worker")
    out["n_tracebacks"] = n_tb
    out["benign_usage_tracebacks"] = n_usage
    out["fatal_traceback"] = ("CUDA error" in text) or (n_tb > n_usage)
    low = text.lower()
    out["batch_invariant_mentions"] = low.count("batch_invariant") + low.count(
        "batch-invariant"
    )
    return out


def _load_token_seqs(path: Path | None) -> dict[str, list[int]] | None:
    """Map prompt-id -> generated token-id list from a decode_outputs.jsonl."""
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
        print(f"[cert] token-seq load failed for {path}: {exc}", flush=True)
        return None
    return seqs or None


def token_identity(a: Path | None, b: Path | None, label: str) -> dict[str, Any]:
    """Per-token identity between two decode_outputs.jsonl files (same workload)."""
    sa, sb = _load_token_seqs(a), _load_token_seqs(b)
    if not sa or not sb:
        return {"label": label, "available": False}
    common = sorted(set(sa) & set(sb))
    total = 0
    matched = 0
    n_flipped_seqs = 0
    first_div: list[dict[str, Any]] = []
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
        "label": label,
        "available": True,
        "n_prompts_compared": len(common),
        "n_tokens_compared": total,
        "n_tokens_matched": matched,
        "token_identity_rate": (matched / total) if total else None,
        "n_sequences_with_any_flip": n_flipped_seqs,
        "first_divergences": first_div[:10],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-decodes", type=int, default=3,
                    help="back-to-back decodes (round0 cold + warm rounds)")
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--seed", type=int, default=paths.SEED)
    ap.add_argument("--no-ppl", dest="do_ppl", action="store_false", default=True)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny serve+decode sanity (8 prompts x 16 tok, 2 decodes, no ppl)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", default="stark/surgical357-served-cert")
    ap.add_argument("--wandb-group", default="surgical357-package")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    if args.smoke:
        args.num_prompts = min(args.num_prompts, 8)
        args.output_len = min(args.output_len, 16)
        args.n_decodes = max(2, min(args.n_decodes, 2))
        args.do_ppl = False
        args.no_wandb = True

    for note in paths.prepare_local_gpu_env():
        print(f"[cert] {note}", flush=True)

    submission_dir = (ROOT / "submissions" / SUBMISSION).resolve()
    if not submission_dir.exists():
        raise SystemExit(f"submission not found: {submission_dir}")
    manifest = harness.load_manifest(submission_dir)
    # Hard preconditions on the PACKAGED manifest: lever ON, matmul-tax flag OFF.
    env_block = manifest.get("env") or {}
    assert env_block.get("SURGICAL_ATTN_USE_3D_OFF") == "1", \
        "packaged manifest must set SURGICAL_ATTN_USE_3D_OFF=1"
    assert "VLLM_BATCH_INVARIANT" not in env_block, \
        "packaged manifest must NOT set VLLM_BATCH_INVARIANT (that is the -135 tax)"
    assert "SPECULATIVE_CONFIG" in env_block, \
        "packaged manifest must keep SPECULATIVE_CONFIG (spec-alive -> 357 not 161)"

    server_python = harness.ensure_server_venv(manifest["dependencies"])
    print(f"[cert] submission={submission_dir.name} server_python={server_python}", flush=True)

    out_dir = (args.out_dir or (OUT_ROOT / ("smoke" if args.smoke else "served_run"))).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    server_log = out_dir / "server.log"
    print(f"[cert] workload={args.num_prompts}x{args.output_len} seed={args.seed} "
          f"n_decodes={args.n_decodes} -> {out_dir}", flush=True)

    decodes: list[dict[str, Any]] = []
    decode_files: list[Path] = []
    ppl_summary: dict[str, Any] | None = None
    server_ready_s = None

    t0 = time.time()
    # NOTE: no extra_env -- the lever is baked into the packaged manifest.
    with harness.LocalServer(
        submission_dir, server_python=server_python, log_path=server_log,
    ) as server:
        server_ready_s = time.time() - t0
        print(f"[cert] server ready in {server_ready_s:.0f}s", flush=True)
        for i in range(args.n_decodes):
            decode_out = out_dir / f"decode_round{i:02d}.jsonl"
            decode_summary = out_dir / f"decode_round{i:02d}.summary.json"
            td = time.time()
            summary = harness.capture_decode(
                server_python,
                base_url=server.base_url,
                model=server.served_model_name,
                out_file=decode_out,
                summary_file=decode_summary,
                num_prompts=args.num_prompts,
                output_len=args.output_len,
                seed=args.seed,
            )
            wall_around = time.time() - td
            n_tok = int(summary.get("num_completion_tokens", 0))
            dur = float(summary.get("duration_s", wall_around))
            wall_tps = n_tok / dur if dur > 0 else float("nan")
            n_completed = int(summary.get("num_records", 0))
            rec = {
                "round": i,
                "warm": i > 0,
                "wall_tps": wall_tps,
                "num_completion_tokens": n_tok,
                "decode_duration_s": dur,
                "wall_around_decode_s": wall_around,
                "num_completed_prompts": n_completed,
                "expected_tokens": args.num_prompts * args.output_len,
            }
            decodes.append(rec)
            decode_files.append(decode_out)
            print(f"[cert] round {i} ({'warm' if i > 0 else 'cold'}): "
                  f"wall_tps={wall_tps:.2f} tok={n_tok}/{args.num_prompts * args.output_len} "
                  f"dur={dur:.1f}s completed={n_completed}", flush=True)

        if args.do_ppl:
            try:
                ppl_summary = harness.run_ppl(
                    server_python,
                    base_url=server.base_url,
                    model=server.served_model_name,
                    out_file=out_dir / "ppl.jsonl",
                    summary_file=out_dir / "ppl.summary.json",
                )
                print(f"[cert] PPL={ppl_summary.get('ppl')} "
                      f"records={ppl_summary.get('num_records')}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[cert] WARN PPL failed: {exc}", flush=True)

    mech = grep_log(server_log)

    # Self-determinism: compare the two WARM rounds (round1 vs round2). Falls back
    # to whatever warm rounds exist.
    warm_files = [decode_files[i] for i in range(len(decodes)) if decodes[i]["warm"]]
    if len(warm_files) >= 2:
        self_det = token_identity(warm_files[0], warm_files[1],
                                  "warm round1 vs round2 (self-determinism)")
    else:
        self_det = {"label": "self-determinism", "available": False,
                    "note": "need >=2 warm decodes"}

    warm_tps = [d["wall_tps"] for d in decodes if d["warm"] and d["wall_tps"] == d["wall_tps"]]
    median_warm_tps = statistics.median(warm_tps) if warm_tps else float("nan")
    ppl_val = (ppl_summary or {}).get("ppl")
    full_completion = bool(decodes and all(
        d["num_completion_tokens"] == args.num_prompts * args.output_len for d in decodes
    ))

    sanity = {
        "median_warm_wall_tps": median_warm_tps,
        "warm_wall_tps_values": warm_tps,
        "surgical_target_tps": SURGICAL_TARGET_TPS,
        "strict_floor_222": STRICT_FLOOR_222,
        "lift_vs_222": (median_warm_tps - STRICT_FLOOR_222) if warm_tps else None,
        "tps_above_222_floor": bool(warm_tps and median_warm_tps > STRICT_FLOOR_222 + SIGMA_HW),
        "matmul_tax_off": bool(warm_tps and median_warm_tps > STRICT_FLOOR_222 + SIGMA_HW)
                          and not mech["init_batch_invariance_ran"],
        "ppl": ppl_val,
        "ppl_target": PPL_TARGET,
        "ppl_passes_gate": isinstance(ppl_val, (int, float)) and ppl_val <= PPL_GATE,
        "completion_128_128": full_completion,
        "num_completed_prompts": decodes[0]["num_completed_prompts"] if decodes else None,
    }

    lever_fired = bool(
        mech["surgical_armed"] and mech["surgical_forced_true"]
        and mech["splitkv_redirects"] == 0 and not mech["fatal_traceback"]
    )
    self_det_rate = self_det.get("token_identity_rate") if self_det.get("available") else None
    self_det_perfect = bool(self_det.get("available") and self_det_rate == 1.0)

    result = {
        "pr": 494,
        "census": "served-vs-served matched-config self-determinism + speed sanity",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "submission": SUBMISSION,
        "submission_dir": str(submission_dir),
        "serve_venv_wheel_stock": True,  # verified out-of-band: triton line 34 stock
        "workload": {"num_prompts": args.num_prompts, "output_len": args.output_len,
                     "seed": args.seed, "n_decodes": args.n_decodes},
        "server_ready_s": server_ready_s,
        "decodes": decodes,
        "mechanism": mech,
        "lever_fired_in_packaged_stack": lever_fired,
        "self_determinism": self_det,
        "self_determinism_perfect_r1_r2": self_det_perfect,
        "speed_sanity": sanity,
        "analysis_only": True,
        "official_tps": 0,
        "no_served_file_change": True,
    }
    result_path = out_dir / "served_cert_result.json"
    result_path.write_text(json.dumps(result, indent=2))

    print("\n[cert] ================= SERVED CERT (Census 1) =================", flush=True)
    print(f"  lever_fired_in_packaged_stack = {lever_fired} "
          f"(armed={mech['surgical_armed']} forced={mech['surgical_forced_true']} "
          f"splitkv_redirects={mech['splitkv_redirects']} onegraph={mech['onegraph_captured']})", flush=True)
    print(f"  self-determinism r1-vs-r2     = {self_det_rate} "
          f"(perfect={self_det_perfect}, flips_seqs={self_det.get('n_sequences_with_any_flip')})", flush=True)
    print(f"  median_warm_wall_tps          = {median_warm_tps:.2f} "
          f"(target ~{SURGICAL_TARGET_TPS}, floor 222, matmul_tax_off={sanity['matmul_tax_off']})", flush=True)
    print(f"  PPL                           = {ppl_val} (target {PPL_TARGET}, gate<={PPL_GATE}: {sanity['ppl_passes_gate']})", flush=True)
    print(f"  completion_128_128            = {full_completion}", flush=True)
    print(f"[cert] artifacts -> {result_path}", flush=True)

    run_id = None
    if not args.no_wandb:
        run_id = _log_wandb(args, result)
        result["wandb_run_id"] = run_id
        result_path.write_text(json.dumps(result, indent=2))
    return 0


def _log_wandb(args, result: dict[str, Any]) -> str | None:
    try:
        from scripts import wandb_logging
    except Exception as exc:  # noqa: BLE001
        print(f"[cert] wandb import failed ({exc}); skip", flush=True)
        return None
    try:
        run = wandb_logging.init_wandb_run(
            job_type="surgical357-operative-cert",
            agent="stark",
            name=args.wandb_name,
            group=args.wandb_group,
            tags=["surgical357-package", "pr494", "operative-cert", "analysis-only"],
            config={
                "submission": SUBMISSION,
                "workload_prompts": args.num_prompts,
                "workload_output_len": args.output_len,
                "seed": args.seed,
                "analysis_only": True,
                "official_tps": 0,
            },
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[cert] wandb init failed ({exc}); skip", flush=True)
        return None
    if run is None:
        print("[cert] wandb disabled (no API key); skip", flush=True)
        return None
    run_id = getattr(run, "id", None)
    try:
        san = result["speed_sanity"]
        mech = result["mechanism"]
        flat = {
            "cert/lever_fired": int(result["lever_fired_in_packaged_stack"]),
            "cert/self_determinism_rate": result["self_determinism"].get("token_identity_rate") or 0.0,
            "cert/self_determinism_perfect": int(result["self_determinism_perfect_r1_r2"]),
            "cert/median_warm_wall_tps": san["median_warm_wall_tps"],
            "cert/lift_vs_222": san.get("lift_vs_222") or 0.0,
            "cert/ppl": san.get("ppl") or 0.0,
            "cert/ppl_passes_gate": int(san["ppl_passes_gate"]),
            "cert/completion_128_128": int(san["completion_128_128"]),
            "cert/splitkv_redirects": mech["splitkv_redirects"],
            "cert/matmul_tax_off": int(san["matmul_tax_off"]),
        }
        flat = {k: v for k, v in flat.items() if isinstance(v, (int, float))}
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run, name="surgical357_served_cert",
            artifact_type="operative-cert", data=result,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[cert] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:
            pass
    return run_id


if __name__ == "__main__":
    raise SystemExit(main())
