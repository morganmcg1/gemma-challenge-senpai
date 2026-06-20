#!/usr/bin/env python
"""PR #818 — Head-precision x E_accept sweep (does int4 lm_head depress acceptance?).

Four arms, served back-to-back on THIS A10G with an IDENTICAL greedy decode
workload (conc=1, temp=0, ignore_eos) and the IDENTICAL MTP drafter (the
manifest's DRAFTER_MODEL, K=NUM_SPECULATIVE_TOKENS). The ONLY delta across arms
is the lm_head bytes the VERIFIER reads:

  bf16_head : official google/gemma-4-E4B-it-qat-w4a16-ct base, head bf16
              (tied to embed_tokens, in quant `ignore`)   <- acceptance ceiling
  int4_g32  : base + lm_head int4 W4A16 group_size=32      (== shipped int4head;
              same deterministic builder + source => byte-identical)
  int4_g64  : base + lm_head int4 W4A16 group_size=64
  int4_g128 : base + lm_head int4 W4A16 group_size=128

The int4 BODY (343 weight_packed tensors, g32) is copied byte-for-byte from the
SAME base across every int4 arm, and the drafter is never changed -> the draft
PROPOSALS are identical and only the verifier-argmax (head precision) moves. A
spec token is accepted iff draft == verifier-argmax, so any E_accept delta is
*caused* by lm_head quantization, not by a different draft distribution.

Metrics per arm (logged to W&B group `bi0-headquant-accept`):
  * E_accept (mean accepted draft length, = 1 + sum_i accept_rate[pos_i])
  * per-position accept rate, positions 1..K (whole-run, from vLLM's cumulative
    Prometheus per-pos counters)
  * served steady gen TPS (vLLM's own whole-run engine meter) and wall TPS
    (completion_tokens / decode_wall_s)

Disk-aware: each int4 checkpoint (~10.6 GB) is built just-in-time and DELETED
after its arm is profiled (peak ~1 build on disk), because the node has limited
free space. LOCAL profiling only -- NO HF job is launched.

Usage:
  python sweep.py [--smoke] [--arms bf16_head,int4_g32,...] [--reps N]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths, serve_profile  # noqa: E402

SUBMISSION = ROOT / "submissions" / "int4_mtp_bi0_int4head"
BUILDER = SUBMISSION / "build_lmhead_quant.py"
BASE_HUB = "google/gemma-4-E4B-it-qat-w4a16-ct"
BUILD_DIR = Path("/workspace/gemma_build")  # scratch for the just-in-time head builds
OUT = ROOT / "research" / "headquant_accept_818" / "runs"

# Arm -> (head_bits, head_group_size or None for bf16). Order matters: bf16 first
# (ceiling), then int4 g32 (the control we must reproduce ~3.379), then g64, g128.
ARMS = [
    ("bf16_head", None),
    ("int4_g32", 32),
    ("int4_g64", 64),
    ("int4_g128", 128),
]


def resolve_base_snapshot() -> str:
    """Local cached dir for the official base (no re-download)."""
    from huggingface_hub import snapshot_download

    return snapshot_download(BASE_HUB, local_files_only=True)


def build_head_checkpoint(server_python: Path, base_dir: str, gs: int, out: Path) -> dict:
    """Build base + lm_head int4 W4A16 group_size=gs (body copied byte-identical)."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    log = out.parent / f"build_{out.name}.log"
    t0 = time.time()
    print(f"[build] g{gs} -> {out}", flush=True)
    with open(log, "w") as lf:
        subprocess.run(
            [str(server_python), str(BUILDER),
             "--src", base_dir, "--out", str(out),
             "--num-bits", "4", "--head-group-size", str(gs)],
            check=True, stdout=lf, stderr=subprocess.STDOUT,
        )
    cfg = json.load(open(out / "config.json"))
    qc = cfg["quantization_config"]
    groups = qc.get("config_groups", {})
    body = groups.get("group_0", {}).get("weights", {})
    head = groups.get("group_1", {}).get("weights", {})
    meta = {
        "build_s": round(time.time() - t0, 1),
        "tie": cfg.get("tie_word_embeddings"),
        "body_bits": body.get("num_bits"), "body_gs": body.get("group_size"),
        "head_bits": head.get("num_bits"), "head_gs": head.get("group_size"),
        "head_target": groups.get("group_1", {}).get("targets"),
        "lm_head_in_ignore": "lm_head" in qc.get("ignore", []),
    }
    print(f"[build] g{gs} done in {meta['build_s']}s  body=g{meta['body_gs']}/{meta['body_bits']}b "
          f"head=g{meta['head_gs']}/{meta['head_bits']}b tie={meta['tie']}", flush=True)
    # Guard: the body MUST stay int4 g32 (byte-identical), only the head changes.
    assert meta["body_gs"] == 32 and meta["body_bits"] == 4, f"body drifted: {meta}"
    assert meta["head_gs"] == gs and meta["head_bits"] == 4, f"head wrong: {meta}"
    return meta


def parse_per_pos(metrics_text: str, num_drafts: float | None) -> dict:
    """Whole-run per-position accept rate from vLLM's cumulative Prometheus counters.

    accept_rate[i] = (sum over engine label sets of accepted_per_pos[i]) / num_drafts.
    Cross-checks: sum_i accept_rate[i] == E_accept - 1.
    """
    pos: dict[int, float] = {}
    pat = re.compile(
        r"^vllm:spec_decode_num_accepted_tokens_per_pos(?:_total)?\{([^}]*)\}\s+([\d.eE+-]+)$",
        re.M,
    )
    for m in pat.finditer(metrics_text):
        labels, val = m.group(1), m.group(2)
        pm = re.search(r'position="(\d+)"', labels)
        if not pm:
            continue
        try:
            pos[int(pm.group(1))] = pos.get(int(pm.group(1)), 0.0) + float(val)
        except ValueError:
            pass
    if not pos or not num_drafts:
        return {}
    k = max(pos) + 1
    counts = [pos.get(i, 0.0) for i in range(k)]
    rates = [c / num_drafts for c in counts]
    return {
        "accepted_per_pos_counts": counts,
        "accepted_per_pos_rate": rates,
        "per_pos_K": k,
        "sum_per_pos_rate": sum(rates),
    }


# Monkeypatch parse_spec_metrics so per-position counters are captured from the
# SAME live /metrics scrape the harness already does (server still up). Purely
# additive: original keys are preserved; we only add per-pos keys. No edit to the
# shared harness file on disk.
_ORIG_PSM = serve_profile.parse_spec_metrics


def _psm_with_per_pos(metrics_text: str) -> dict:
    d = _ORIG_PSM(metrics_text)
    d.update(parse_per_pos(metrics_text, d.get("num_drafts")))
    return d


serve_profile.parse_spec_metrics = _psm_with_per_pos


def log_arm_wandb(arm: str, gs, report: dict, derived: dict, rep: int = 1) -> str | None:
    try:
        import os
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] skipped ({exc})", flush=True)
        return None
    try:
        name = f"ubel/headquant-{arm}" + (f"-rep{rep}" if rep != 1 else "")
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            name=name, group="bi0-headquant-accept",
            job_type="profile",
            config={
                "arm": arm,
                "rep": rep,
                "head_dtype": "bf16" if gs is None else "int4",
                "head_bits": 16 if gs is None else 4,
                "head_group_size": gs,
                "body_bits": 4, "body_group_size": 32,
                "model_id": derived["model_id"],
                "drafter": derived["drafter"],
                "num_speculative_tokens": derived["num_spec"],
                "num_prompts": report["num_prompts"],
                "output_len": report["output_len"],
            },
        )
        flat = {
            "e_accept": derived["e_accept"],
            "e_accept_source": derived["e_accept_source"],
            "draft_acceptance_rate": derived["draft_acceptance_rate"],
            "tps/measured_steady_gen_tps": derived["steady_gen_tps"],
            "tps/wall_tps": derived["wall_tps"],
            "decode_wall_s": derived["decode_wall_s"],
            "num_drafts": derived["num_drafts"],
            "num_accepted_tokens": derived["num_accepted_tokens"],
        }
        for i, r in enumerate(derived.get("per_pos_rate", []), start=1):
            flat[f"accept_rate/pos_{i}"] = r
        run.summary.update(flat)
        if derived.get("per_pos_rate"):
            tbl = wandb.Table(columns=["position", "accept_rate", "accepted_count"])
            for i, (r, c) in enumerate(
                zip(derived["per_pos_rate"], derived.get("per_pos_counts", [])), start=1
            ):
                tbl.add_data(i, r, c)
            run.log({"per_position_acceptance": tbl})
        rid = run.id
        run.finish()
        print(f"[wandb] {arm} -> run {rid}", flush=True)
        return rid
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] {arm} log failed ({exc})", flush=True)
        return None


def derive(report: dict) -> dict:
    a = report["analysis"]
    timing = report["variants"]["frontier"]["timing"]
    spec = timing.get("spec_metrics") or {}
    dsum = timing.get("decode_summary") or {}
    decode_wall = timing.get("decode_wall_s")
    completion = dsum.get("num_completion_tokens")
    wall_tps = (completion / decode_wall) if (completion and decode_wall) else None
    env = (report.get("variants", {}).get("frontier", {}) or {}).get("extra_env", {})
    return {
        "model_id": env.get("MODEL_ID", "<manifest>"),
        "drafter": SUBMISSION_DRAFTER,
        "num_spec": SUBMISSION_NUM_SPEC,
        "e_accept": a.get("e_accept"),
        "e_accept_source": a.get("e_accept_source"),
        "draft_acceptance_rate": spec.get("draft_acceptance_rate"),
        "steady_gen_tps": a.get("tps", {}).get("measured_steady_gen_tps"),
        "wall_tps": wall_tps,
        "decode_wall_s": decode_wall,
        "num_drafts": spec.get("num_drafts"),
        "num_accepted_tokens": spec.get("num_accepted_tokens"),
        "per_pos_rate": spec.get("accepted_per_pos_rate", []),
        "per_pos_counts": spec.get("accepted_per_pos_counts", []),
        "sum_per_pos_rate": spec.get("sum_per_pos_rate"),
    }


SUBMISSION_DRAFTER = None
SUBMISSION_NUM_SPEC = None


def main() -> int:
    global SUBMISSION_DRAFTER, SUBMISSION_NUM_SPEC
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny workload (8x64), validation only")
    ap.add_argument("--arms", default=None, help="comma list subset of arm names")
    ap.add_argument("--tag", default=None, help="override output-dir/summary tag (avoid clobber on re-run)")
    ap.add_argument("--rep", type=int, default=1, help="rep index (>=2 => distinct W&B run name + config.rep)")
    args = ap.parse_args()

    manifest = harness.load_manifest(SUBMISSION)
    SUBMISSION_DRAFTER = manifest["env"]["DRAFTER_MODEL"]
    SUBMISSION_NUM_SPEC = int(manifest["env"]["NUM_SPECULATIVE_TOKENS"])

    arms = ARMS
    if args.arms:
        want = set(args.arms.split(","))
        arms = [a for a in ARMS if a[0] in want]

    np_v = 8 if args.smoke else paths.NUM_PROMPTS
    ol_v = 64 if args.smoke else paths.OUTPUT_LEN
    tag = args.tag or ("smoke" if args.smoke else "full")

    for note in paths.prepare_local_gpu_env():
        print(f"[env] {note}", flush=True)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    base_dir = resolve_base_snapshot()
    print(f"[sweep] server_python={server_python}\n[sweep] base={base_dir}\n"
          f"[sweep] arms={[a for a, _ in arms]} workload={np_v}x{ol_v} ({tag})\n"
          f"[sweep] drafter={SUBMISSION_DRAFTER} K={SUBMISSION_NUM_SPEC}", flush=True)

    summary: dict[str, dict] = {}
    for arm, gs in arms:
        t0 = time.time()
        print(f"\n##### ARM {arm} (head {'bf16' if gs is None else f'int4 g{gs}'}) #####", flush=True)
        ckpt = None
        build_meta = None
        if gs is None:
            model_id = base_dir  # serve base directly: bf16 tied head
        else:
            ckpt = BUILD_DIR / f"headq_g{gs}"
            build_meta = build_head_checkpoint(server_python, base_dir, gs, ckpt)
            model_id = str(ckpt)

        # Inject this arm's MODEL_ID as the frontier variant's extra_env. LMHEAD_QUANT_*
        # is left unset (we serve a prebuilt dir / the base directly, not startup-quant).
        serve_profile.VARIANTS["frontier"] = {"MODEL_ID": model_id}
        out_dir = OUT / tag / arm
        try:
            report = serve_profile.run(
                SUBMISSION, server_python, out_dir,
                num_prompts=np_v, output_len=ol_v, kernel_window_tokens=256,
                variants=["frontier"], do_kernel=False,
                wandb_name=None, wandb_group="bi0-headquant-accept",
            )
            d = derive(report)
            d["build_meta"] = build_meta
            wid = None if args.smoke else log_arm_wandb(arm, gs, report, d, rep=args.rep)
            d["wandb_run_id"] = wid
            summary[arm] = d
            ppr = d.get("per_pos_rate") or []
            print(f"[sweep] {arm}: E_accept={d['e_accept']:.4f} ({d['e_accept_source']})  "
                  f"r={d['draft_acceptance_rate']}  steady_tps={d['steady_gen_tps']:.2f}  "
                  f"wall_tps={(d['wall_tps'] or 0):.2f}  "
                  f"per_pos={['%.3f' % x for x in ppr]}  sum_pp={d.get('sum_per_pos_rate')}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        finally:
            if ckpt is not None and ckpt.exists():
                shutil.rmtree(ckpt)
                print(f"[sweep] freed {ckpt}", flush=True)

    out_json = OUT / f"sweep_summary_{tag}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 78)
    print(f"HEAD-PRECISION x E_accept SWEEP ({tag})")
    print("=" * 78)
    print(f"{'arm':12s} {'head':10s} {'E_accept':>9s} {'r':>7s} {'steady_tps':>11s} {'wall_tps':>9s}")
    for arm, gs in arms:
        d = summary.get(arm)
        if not d:
            continue
        head = "bf16" if gs is None else f"int4_g{gs}"
        print(f"{arm:12s} {head:10s} {d['e_accept']:9.4f} {(d['draft_acceptance_rate'] or 0):7.4f} "
              f"{(d['steady_gen_tps'] or 0):11.2f} {(d['wall_tps'] or 0):9.2f}")
    print(f"\n[sweep] summary -> {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
