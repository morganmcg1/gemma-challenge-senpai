#!/usr/bin/env python
"""PR #595 -- base_fullhead vs int4_g128_lmhead: same-recipe local TPS head-to-head.

ANALYSIS-ONLY local profiling card. NO FIRE, NO served-file change, NO benchmark
launch. The deliverable is an apples-to-apples 2x2 of:

    {base_fullhead, int4_g128_lmhead} x {spec-OFF AR M=1, spec-ON MTP-K7}

measured LOCALLY on this pod under the IDENTICAL serving recipe -- the ship's
``fa2sw_strict_surgical357`` substrate (onegraph / fa-sliding / surgical-attn /
fused-argmax / MTP K=7 drafter) with substrate-swap overrides that disable the
osoi5 bake + 12k lm_head prune, so the ONLY difference between the two configs is
the served checkpoint:

  - ``base_fullhead``     : stock int4_g32 QAT body + full native 262k **bf16** lm_head
                           (lawine #572: spec-ON 253.99, spec-OFF 83.44 on this pod).
  - ``int4_g128_lmhead`` : int4_g128 body + untied int4-g128 lm_head (the missing leg;
                           official 126.378 on its OWN plain serve.py, no spec).

This probe serves ONE config (``--config``) through the surgical357 stack and
measures BOTH spec frames in one process, writing a per-config JSON. The combine /
head-read decomposition / W&B logging happen in ``assemble_headtohead.py`` (run
under ``.venv`` so wandb is importable -- the serve venv has none).

TPS metric = official wall_tps = num_completion_tokens / decode_duration_s
(conc=1, 128x512, seed 1), warm-median of N passes (#72). Local A10G number;
official projection = TAU_LO * local (#267). spec-OFF uses
``SENPAI_REFERENCE_MODE=1`` -> drafter cleared, plain M=1 AR greedy.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path
from statistics import median, pstdev

ROOT = Path("/workspace/senpai/target")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.local_validation import harness, paths, serve_profile  # noqa: E402

# Same surgical357 substrate that produced the base_fullhead #572 anchors.
SUB = ROOT / "submissions" / "fa2sw_strict_surgical357"
OUT = ROOT / "research" / "submission_headtohead"

# ---- the two checkpoints (the ONLY thing that differs between configs) ----
# lawine's OWN stock int4_g32 snapshot (clean launch isolation; identical bytes to
# the google hub model, NO osoi5 bake). bf16 native 262k lm_head (quant ignore list).
BASE_INT4 = (
    "/senpai-run/home/student-lawine/.cache/huggingface/hub/"
    "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/"
    "ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0"
)
# int4_g128 body + untied int4-g128 lm_head (the human's chosen best-quality fast
# submission; bundled checkpoint, already on local disk).
INT4_G128_LMHEAD = str(ROOT / "submissions" / "int4_g128_lmhead" / "model")

CONFIGS = {
    "base_fullhead": {
        "model_dir": BASE_INT4,
        "desc": "stock int4_g32 body + native 262k bf16 lm_head (NO bake, NO prune)",
    },
    "int4_g128_lmhead": {
        "model_dir": INT4_G128_LMHEAD,
        "desc": "int4_g128 body + untied int4-g128 lm_head (embed_tokens bf16)",
    },
}

TAU_LO = 1.03524          # local->official transfer factor (#267)
SIGMA_HW = 4.864          # cross-config/session hardware TPS band (card constant)


def overrides(config: str, spec_on: bool) -> dict[str, str]:
    """Substrate-swap the surgical357 submission onto a target checkpoint.

    Byte-for-byte the #572 base_fullhead recipe, only LOCAL_MODEL_DIR /
    PLE_FOLD_TARGET_MODEL change per config. spec_on=False adds
    SENPAI_REFERENCE_MODE=1 -> drafter OFF, plain M=1 AR greedy.
    """
    model_dir = CONFIGS[config]["model_dir"]
    env = {
        # honest single-stream; precache OFF (bench-specific warm, inert here).
        "PRECACHE_BENCH": "0",
        "PRECACHE_REQUIRE": "0",
        "PRECACHE_DATASET": "/tmp/senpai_aime_no_precache.json",
        "MAX_NUM_SEQS": "1",
        "MAX_NUM_BATCHED_TOKENS": "512",
        # the swapped checkpoint, no bake, no prune (PLE fold is in-memory, scale=1
        # -> identity; no model copy -> safe on a disk-constrained node).
        "LOCAL_MODEL_DIR": model_dir,
        "PLE_FOLD_TARGET_MODEL": model_dir,
        "PLE_FOLD_EMBED_SCALE": "1",
        "LM_HEAD_PRUNE": "0",
        "LM_HEAD_PRUNE_REQUIRE": "0",
        "PCK04_KEEPSET": "",
        # enable stats so vLLM prints SpecDecoding metrics (acceptance_length) and
        # exposes /metrics (manifest ships =1, which suppresses them).
        "DISABLE_LOG_STATS": "0",
    }
    if not spec_on:
        env["SENPAI_REFERENCE_MODE"] = "1"
    return env


def _completion(base_url: str, model: str, prompt: str, max_tokens: int, timeout: int = 300) -> dict:
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "stream": False, "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _get_text(url: str, timeout: float = 30.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def decode_wall_tps(summary: dict) -> float:
    dur = summary.get("duration_s") or 0.0
    toks = summary.get("num_completion_tokens") or 0
    return toks / dur if dur > 0 else float("nan")


def serve_arm(server_python: Path, *, config: str, spec_on: bool, num_prompts: int,
              output_len: int, n_decodes: int, port: int, tag: str) -> dict:
    ov = overrides(config, spec_on)
    log = OUT / f"server_{tag}.log"
    res: dict = {"tag": tag, "config": config, "spec_on": spec_on,
                 "serve_overrides": ov, "serve_ok": False, "error": None}
    peak = {"mib": 0}
    stop = threading.Event()

    def sample_gpu() -> None:
        while not stop.is_set():
            try:
                o = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                peak["mib"] = max(peak["mib"], int(o.stdout.strip().splitlines()[0]))
            except Exception:
                pass
            time.sleep(2)

    gpu_thread = threading.Thread(target=sample_gpu, daemon=True)
    gpu_thread.start()
    try:
        with harness.LocalServer(SUB, server_python=server_python, port=port,
                                 startup_timeout_s=1800, log_path=log,
                                 extra_env=ov) as srv:
            res["serve_ok"] = True
            model = srv.served_model_name
            base_url = srv.base_url
            res["served_model_name"] = model
            # self-test (proves the checkpoint loads + decodes coherently through the
            # patched stack) + warmup (trigger cudagraph capture before timed passes).
            st = _completion(base_url, model, "The capital of France is", 6)
            res["self_test_text"] = (st.get("choices") or [{}])[0].get("text", "")
            _completion(base_url, model, "Explain how a transformer decodes one token at a time.", 16)

            tps_runs, decode_files = [], []
            for i in range(1, n_decodes + 1):
                df = OUT / f"decode_{tag}_r{i}.jsonl"
                sf = OUT / f"decode_{tag}_r{i}.summary.json"
                s = harness.capture_decode(server_python, base_url=base_url, model=model,
                                           out_file=df, summary_file=sf,
                                           num_prompts=num_prompts, output_len=output_len,
                                           timeout_s=3600)
                tps = decode_wall_tps(s)
                tps_runs.append(tps)
                decode_files.append(str(df))
                print(f"[{tag}] decode r{i}: wall_tps={tps:.3f} "
                      f"toks={s.get('num_completion_tokens')} dur={s.get('duration_s'):.2f}s", flush=True)
            res["tps_runs"] = tps_runs
            res["warm_median_tps"] = median(tps_runs) if tps_runs else float("nan")
            res["tps_run_std"] = pstdev(tps_runs) if len(tps_runs) > 1 else 0.0
            res["decode_files"] = decode_files
            # Prometheus /metrics acceptance (spec-on only) before teardown.
            if spec_on:
                try:
                    res["spec_metrics"] = serve_profile.parse_spec_metrics(_get_text(f"{base_url}/metrics"))
                except Exception as exc:  # noqa: BLE001
                    res["spec_metrics"] = {"error": str(exc)}
    except Exception as e:  # noqa: BLE001
        res["error"] = "".join(traceback.format_exception(type(e), e, e.__traceback__))[-6000:]
        print(f"[{tag}] EXCEPTION serve_ok={res['serve_ok']}\n{res['error']}", flush=True)
    finally:
        stop.set()
        gpu_thread.join(timeout=5)
    res["peak_gpu_mib"] = peak["mib"]
    if spec_on and log.exists():
        res["spec_log"] = serve_profile.parse_spec_log(log.read_text())
    res["server_log"] = str(log)
    return res


def acceptance_of(arm: dict) -> tuple[float | None, str]:
    sl = arm.get("spec_log") or {}
    sm = arm.get("spec_metrics") or {}
    acc_log = sl.get("e_accept_exact")
    acc_log_iv = sl.get("e_accept_interval_mean")
    acc_prom = sm.get("e_accept_mean_acceptance_length")
    val = acc_log or acc_log_iv or acc_prom
    src = ("server_log_exact" if acc_log else
           "server_log_interval_mean" if acc_log_iv else
           "prometheus" if acc_prom else "none")
    return val, src


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--n-decodes-specon", type=int, default=2)
    ap.add_argument("--n-decodes-specoff", type=int, default=2)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny 2x16 single-decode each frame, to validate load+serve")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = args.config
    np_, ol = args.num_prompts, args.output_len
    nd_on, nd_off = args.n_decodes_specon, args.n_decodes_specoff
    suffix = ""
    if args.smoke:
        np_, ol, nd_on, nd_off, suffix = 2, 16, 1, 1, "_smoke"

    for note in paths.prepare_local_gpu_env():
        print(f"[probe] {note}", flush=True)

    manifest = harness.load_manifest(SUB)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    spec_cfg = (manifest.get("env") or {}).get("SPECULATIVE_CONFIG", "")
    print(f"[probe] config={cfg} model_dir={CONFIGS[cfg]['model_dir']}", flush=True)
    print(f"[probe] SPECULATIVE_CONFIG (ship surgical-357) = {spec_cfg}", flush=True)

    report: dict = {
        "pr": 595,
        "config": cfg,
        "config_desc": CONFIGS[cfg]["desc"],
        "submission_substrate": str(SUB.relative_to(ROOT)),
        "model_dir": CONFIGS[cfg]["model_dir"],
        "speculative_config": spec_cfg,
        "spec_drafter": "mtp_k7",
        "num_prompts": np_,
        "output_len": ol,
        "smoke": args.smoke,
        "analysis_only": True,
        "official_tps": 0,
        "tau_lo": TAU_LO,
        "sigma_hw": SIGMA_HW,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # ---- spec-ON (MTP K=7) ----
    print(f"\n===== {cfg}: spec-ON (MTP K=7) =====", flush=True)
    arm_on = serve_arm(server_python, config=cfg, spec_on=True, num_prompts=np_,
                       output_len=ol, n_decodes=nd_on, port=args.port,
                       tag=f"{cfg}_specon{suffix}")
    report["arm_spec_on"] = arm_on

    # ---- spec-OFF (M=1 AR reference) ----
    print(f"\n===== {cfg}: spec-OFF (M=1 AR) =====", flush=True)
    arm_off = serve_arm(server_python, config=cfg, spec_on=False, num_prompts=np_,
                        output_len=ol, n_decodes=nd_off, port=args.port,
                        tag=f"{cfg}_specoff{suffix}")
    report["arm_spec_off"] = arm_off

    acc, acc_src = acceptance_of(arm_on)
    tps_on = arm_on.get("warm_median_tps", float("nan"))
    tps_off = arm_off.get("warm_median_tps", float("nan"))
    report["spec_on_tps_local"] = tps_on
    report["spec_off_tps_local"] = tps_off
    report["spec_on_tps_official_proj"] = tps_on * TAU_LO if tps_on == tps_on else float("nan")
    report["spec_off_tps_official_proj"] = tps_off * TAU_LO if tps_off == tps_off else float("nan")
    report["acceptance_length"] = acc
    report["acceptance_length_source"] = acc_src
    report["spec_lift_x"] = (tps_on / tps_off) if (tps_off == tps_off and tps_off) else None

    out_json = OUT / f"headtohead_{cfg}{suffix}.json"
    out_json.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[probe] wrote {out_json}", flush=True)

    print(f"\n========== HEAD-TO-HEAD: {cfg} ==========", flush=True)
    print(f"serve_ok  spec-ON={arm_on.get('serve_ok')}  spec-OFF={arm_off.get('serve_ok')}", flush=True)
    print(f"spec_off_tps_local (AR M=1)  = {tps_off:.3f}  (official-proj {report['spec_off_tps_official_proj']:.2f})", flush=True)
    print(f"spec_on_tps_local  (MTP K=7) = {tps_on:.3f}  (official-proj {report['spec_on_tps_official_proj']:.2f})", flush=True)
    print(f"acceptance_length E[T]       = {acc} (src {acc_src})", flush=True)
    print(f"spec_lift_x                  = {report['spec_lift_x']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
