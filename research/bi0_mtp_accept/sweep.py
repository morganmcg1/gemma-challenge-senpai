"""PR #792 — MTP draft-acceptance tuning at K=6 on shipped bi0.

Hypothesis: bi0 (`submissions/int4_mtp_bi0_surgattn`) serves the gemma4_assistant
MTP drafter at its drafter config's NATIVE ``centroid_intermediate_top_k`` value.
Widening that top-k gives the proposer a better centroid candidate set -> more
drafts clear greedy verification -> higher acceptance length -> the one big M=7
int4 verify GEMM is amortized over MORE emitted tokens -> higher decode TPS, with
the EMITTED sequence unchanged (at temp=0 the rejection sampler emits the target's
argmax regardless of the drafter, so the output is byte-identical).

KEY FACT established before this run: the pristine HF-cache drafter config ships
``centroid_intermediate_top_k = 32`` (verified at
``~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-q4_0-unquantized-assistant``).
bi0's serve.py passes the drafter HF repo id straight to ``--speculative-config``
and does NOT patch this field, so bi0 runs the drafter at the NARROW native 32.
stark (#786) found 64 optimal for the same drafter on the fa2sw frontier stack
(topk128 = -3.9 TPS). So 32 -> 64 is a LIVE, untested acceptance lever on bi0.

This driver sweeps centroid_intermediate_top_k WITHOUT editing the shipped
submission: it stages a per-top-k local drafter dir (weights symlinked from the HF
cache, only config.json rewritten) and serves bi0 with ``DRAFTER_MODEL`` pointed at
it. The int4 W4A16 target + surgical-2D-attn verify path are byte-for-byte the
shipped stack; the ONLY changed variable is the drafter's centroid candidate width.

Per point it measures, on the official 128-prompt x 512-token conc=1 workload:
  * wall_tps      = num_completion_tokens / duration_s   (robust headline TPS, #72)
  * E_accept      = 1 + K*accepted/drafted               (mean acceptance length)
  * accept_rate   = accepted/drafted                     (per-draft-token)
  * cycle_wall_ms = 1000*E_accept/wall_tps               (derived per-step cost)
  * steady_gen_tps_mean (decode-phase only; secondary)
  * PPL           (official ppl_endpoint; drafter-invariant sanity check)
  * 128/128 completion
Greedy identity is scored AFTER the sweep: each point's per-prompt
``completion_token_sha256`` vs the control (top_k=32). The drafter lever is
output-neutral, so every point MUST be 128/128 byte-identical to control; any
divergence is a red flag to investigate.

Run under the repo .venv python (it has wandb; the serve venv does not, and a local
./wandb dir shadows the import):
    cd target && .venv/bin/python -m research.bi0_mtp_accept.sweep [32 64 128]
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths, serve_profile  # noqa: E402

SUBMISSION = ROOT / "submissions" / "int4_mtp_bi0_surgattn"
DRAFTER_REPO_CACHE = (
    "models--google--gemma-4-E4B-it-qat-q4_0-unquantized-assistant"
)
CONTROL_TOP_K = 32  # native bi0 value
OUT_DIR = ROOT / "research" / "bi0_mtp_accept"

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
WANDB_GROUP = "bi0-mtp-accept"


def find_drafter_snapshot() -> Path:
    """Locate the pristine HF-cache snapshot dir for the qat-assistant drafter."""
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    cands = sorted(
        glob.glob(str(hub / DRAFTER_REPO_CACHE / "snapshots" / "*"))
    )
    snaps = [Path(c) for c in cands if (Path(c) / "config.json").exists()]
    if not snaps:
        raise SystemExit(
            f"drafter snapshot not found under {hub / DRAFTER_REPO_CACHE}; "
            "bi0 must have served at least once to populate the HF cache"
        )
    return snaps[-1]


def native_top_k(snapshot: Path) -> int:
    cfg = json.loads((snapshot / "config.json").read_text())
    return int(cfg.get("centroid_intermediate_top_k", -1))


def stage_drafter(snapshot: Path, top_k: int) -> Path:
    """Materialize a local drafter dir with centroid_intermediate_top_k=top_k.

    Every file is symlinked to the HF-cache blob (so we never copy the weights or
    mutate the cache) EXCEPT config.json, which is written as a real, patched file.
    """
    dst = Path(f"/tmp/drafter_ctk{top_k}")
    dst.mkdir(parents=True, exist_ok=True)
    for src in snapshot.iterdir():
        target = dst / src.name
        if src.name == "config.json":
            cfg = json.loads(src.read_text())  # resolves symlink -> real content
            old = cfg.get("centroid_intermediate_top_k")
            cfg["centroid_intermediate_top_k"] = top_k
            if target.is_symlink() or target.exists():
                target.unlink()
            target.write_text(json.dumps(cfg, indent=2))
            print(
                f"[stage] {dst.name}/config.json centroid_intermediate_top_k "
                f"{old} -> {top_k}",
                flush=True,
            )
            continue
        real = src.resolve()  # follow HF cache symlink to the blob
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(real)
    return dst


def parse_decode_phase(log_path: Path) -> str:
    """Server-log text captured for the decode phase (read before PPL runs)."""
    try:
        return log_path.read_text()
    except OSError:
        return ""


def run_point(top_k: int, snapshot: Path, server_python: Path) -> dict:
    drafter_dir = stage_drafter(snapshot, top_k)
    tag = f"ctk{top_k}"
    log_path = OUT_DIR / f"server_{tag}.log"
    decode_out = OUT_DIR / f"decode_{tag}.jsonl"
    decode_sum = OUT_DIR / f"decode_{tag}.summary.json"
    ppl_out = OUT_DIR / f"ppl_{tag}.jsonl"
    ppl_sum = OUT_DIR / f"ppl_{tag}.summary.json"

    extra_env = {
        "DRAFTER_MODEL": str(drafter_dir),
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    }
    print(f"\n===== centroid_intermediate_top_k = {top_k} =====", flush=True)
    rec: dict = {"centroid_top_k": top_k, "drafter_dir": str(drafter_dir)}
    t0 = time.time()
    with harness.LocalServer(
        SUBMISSION,
        server_python=server_python,
        port=8000,
        log_path=log_path,
        extra_env=extra_env,
        startup_timeout_s=1800,
    ) as srv:
        rec["server_ready_s"] = time.time() - t0
        decode_summary = harness.capture_decode(
            server_python,
            base_url=srv.base_url,
            model=srv.served_model_name,
            out_file=decode_out,
            summary_file=decode_sum,
            num_prompts=paths.NUM_PROMPTS,
            output_len=paths.OUTPUT_LEN,
            timeout_s=3600,
        )
        # Snapshot the decode-phase server log BEFORE PPL appends its own
        # throughput lines (PPL is teacher-forced -> no drafts, so it cannot
        # change accepted/drafted, but it does add "Avg generation throughput"
        # lines that would pollute the decode steady-TPS mean).
        decode_log_text = parse_decode_phase(log_path)
        ppl_summary = harness.run_ppl(
            server_python,
            base_url=srv.base_url,
            model=srv.served_model_name,
            out_file=ppl_out,
            summary_file=ppl_sum,
            timeout_s=1800,
        )

    spec = serve_profile.parse_spec_log(decode_log_text)
    num_completion = int(decode_summary["num_completion_tokens"])
    duration_s = float(decode_summary["duration_s"])
    num_records = int(decode_summary["num_records"])
    wall_tps = num_completion / duration_s if duration_s else float("nan")
    e_accept = spec.get("e_accept_exact")
    accept_rate = spec.get("draft_acceptance_rate")
    cycle_wall_ms = (
        1000.0 * e_accept / wall_tps if (e_accept and wall_tps == wall_tps) else None
    )
    ppl = ppl_summary.get("ppl") or ppl_summary.get("perplexity")

    rec.update(
        {
            "wall_tps": wall_tps,
            "num_completion_tokens": num_completion,
            "num_records": num_records,
            "completed_128": num_records == paths.NUM_PROMPTS
            and num_completion == paths.NUM_PROMPTS * paths.OUTPUT_LEN,
            "duration_s": duration_s,
            "e_accept": e_accept,
            "accept_rate": accept_rate,
            "num_speculative_tokens": spec.get("num_speculative_tokens"),
            "total_accepted_tokens": spec.get("total_accepted_tokens"),
            "total_drafted_tokens": spec.get("total_drafted_tokens"),
            "e_accept_interval_mean": spec.get("e_accept_interval_mean"),
            "cycle_wall_ms": cycle_wall_ms,
            "steady_gen_tps_mean": spec.get("steady_gen_tps_mean"),
            "ppl": ppl,
            "ppl_summary": ppl_summary,
            "decode_out": str(decode_out),
        }
    )
    print(
        f"[point ctk{top_k}] wall_tps={wall_tps:.2f} E_accept={e_accept} "
        f"accept_rate={accept_rate} PPL={ppl} completed={num_records}/128 "
        f"cycle_wall_ms={cycle_wall_ms}",
        flush=True,
    )
    return rec


def score_identity(records: list[dict]) -> dict:
    """Per-prompt completion-token identity of each point vs the control."""

    def load_sha(path: str) -> dict[str, str]:
        out: dict[str, str] = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                out[str(row["id"])] = row["completion_token_sha256"]
        return out

    by_k = {r["centroid_top_k"]: r for r in records}
    if CONTROL_TOP_K not in by_k:
        return {"error": f"no control point top_k={CONTROL_TOP_K}"}
    control_sha = load_sha(by_k[CONTROL_TOP_K]["decode_out"])
    ident: dict[str, dict] = {}
    for r in records:
        k = r["centroid_top_k"]
        sha = load_sha(r["decode_out"])
        ids = set(control_sha) & set(sha)
        matched = sum(1 for i in ids if sha[i] == control_sha[i])
        mismatched = sorted(i for i in ids if sha[i] != control_sha[i])
        ident[str(k)] = {
            "compared": len(ids),
            "matched": matched,
            "identical_to_control": matched == len(ids) and len(ids) > 0,
            "mismatched_ids": mismatched[:20],
            "n_mismatched": len(mismatched),
        }
        r["identical_to_control"] = ident[str(k)]["identical_to_control"]
        r["identity_matched"] = matched
        r["identity_compared"] = len(ids)
    return ident


def log_wandb(rec: dict) -> str | None:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] unavailable: {exc}", flush=True)
        return None
    try:
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=f"lawine/bi0-mtp-accept-ctk{rec['centroid_top_k']}",
            group=WANDB_GROUP,
            job_type="acceptance-sweep",
            config={
                "submission": str(SUBMISSION),
                "centroid_intermediate_top_k": rec["centroid_top_k"],
                "num_speculative_tokens": rec.get("num_speculative_tokens"),
                "K": 6,
                "control_top_k": CONTROL_TOP_K,
                "workload": "128x512 conc1 official",
            },
        )
        summary = {
            "centroid_top_k": rec["centroid_top_k"],
            "wall_tps": rec.get("wall_tps"),
            "e_accept": rec.get("e_accept"),
            "accept_rate": rec.get("accept_rate"),
            "cycle_wall_ms": rec.get("cycle_wall_ms"),
            "steady_gen_tps_mean": rec.get("steady_gen_tps_mean"),
            "ppl": rec.get("ppl"),
            "num_records": rec.get("num_records"),
            "completed_128": rec.get("completed_128"),
            "total_accepted_tokens": rec.get("total_accepted_tokens"),
            "total_drafted_tokens": rec.get("total_drafted_tokens"),
            "identical_to_control": rec.get("identical_to_control"),
            "identity_matched": rec.get("identity_matched"),
            "primary_metric_wall_tps": rec.get("wall_tps"),
            "test_metric_e_accept": rec.get("e_accept"),
        }
        run.summary.update({k: v for k, v in summary.items() if v is not None})
        rid = run.id
        run.finish()
        print(f"[wandb] logged ctk{rec['centroid_top_k']} -> {rid}", flush=True)
        return rid
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] log failed: {exc}", flush=True)
        return None


def main(argv: list[str]) -> int:
    points = [int(x) for x in argv] if argv else [32, 64, 128]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for note in paths.prepare_local_gpu_env():
        print(f"[env] {note}", flush=True)
    snapshot = find_drafter_snapshot()
    nk = native_top_k(snapshot)
    print(f"[main] drafter snapshot {snapshot}", flush=True)
    print(f"[main] native centroid_intermediate_top_k = {nk}", flush=True)
    print(f"[main] sweep points = {points} (control = {CONTROL_TOP_K})", flush=True)

    manifest = harness.load_manifest(SUBMISSION)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    print(f"[main] serve venv python = {server_python}", flush=True)

    records: list[dict] = []
    for top_k in points:
        rec = run_point(top_k, snapshot, server_python)
        records.append(rec)
        (OUT_DIR / "sweep_partial.json").write_text(
            json.dumps({"native_top_k": nk, "records": records}, indent=2, default=str)
        )

    identity = score_identity(records)
    wandb_ids = {}
    for rec in records:
        wandb_ids[str(rec["centroid_top_k"])] = log_wandb(rec)

    report = {
        "submission": str(SUBMISSION),
        "drafter_snapshot": str(snapshot),
        "native_top_k": nk,
        "control_top_k": CONTROL_TOP_K,
        "points": points,
        "wandb_group": WANDB_GROUP,
        "wandb_run_ids": wandb_ids,
        "identity_vs_control": identity,
        "records": records,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (OUT_DIR / "sweep_report.json").write_text(json.dumps(report, indent=2, default=str))

    print("\n========== bi0 centroid_top_k acceptance/TPS sweep ==========", flush=True)
    print(f"{'top_k':>6} {'wall_tps':>9} {'E_accept':>9} {'accept':>7} "
          f"{'cycle_ms':>9} {'PPL':>7} {'128/128':>8} {'==ctrl':>7}", flush=True)
    for rec in records:
        print(
            f"{rec['centroid_top_k']:>6} "
            f"{rec.get('wall_tps', float('nan')):>9.2f} "
            f"{(rec.get('e_accept') or float('nan')):>9.3f} "
            f"{(rec.get('accept_rate') or float('nan')):>7.4f} "
            f"{(rec.get('cycle_wall_ms') or float('nan')):>9.3f} "
            f"{(rec.get('ppl') or float('nan')):>7.4f} "
            f"{str(rec.get('completed_128')):>8} "
            f"{str(rec.get('identical_to_control')):>7}",
            flush=True,
        )
    print(f"\nartifacts -> {OUT_DIR}", flush=True)
    print(f"wandb group {WANDB_GROUP} ids {wandb_ids}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
