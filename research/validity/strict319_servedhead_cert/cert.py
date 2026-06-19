#!/usr/bin/env python
"""Strict-#319 served-16k-head greedy-identity cert for the pinned MTP spec stack (PR #690).

Decision-forcing question: does land #684's free fixed-split attention pin, applied
at serve boot on the DEPLOYED int4 + pruned-16k-head MTP spec stack, close wirbel
#682's measured 51.70% served token break to byte-exact GREEDY_IDENTICAL?

For one ``--pin`` mode this harness:
  1. Builds a *copy* of ``submissions/int4_mtp_batchinv`` with a sitecustomize-chained
     boot pin (``_stark_pin.py``). The real submission is never modified.
  2. Serves the SAME stack twice on the SAME engine/kernels/quant, pin applied to
     BOTH arms so the only removed variable is speculation (challenge contract):
       - reference: NUM_SPECULATIVE_TOKENS=0  (plain int4 M=1 AR greedy reference)
       - candidate: NUM_SPECULATIVE_TOKENS=N  (the K=5/M=6 MTP spec stack)
     captures 128 sharegpt x 512 greedy (ignore_eos, seed 1) token-ids via the
     official decode_outputs.py.
  3. Compares token streams with the OFFICIAL greedy_identity verifier ->
     break-rate, sequence-divergence, first-divergence onset, GREEDY_IDENTICAL/DIVERGENT.
  4. Measures served local wall-TPS of the pinned candidate (steady-state single
     stream) -> official-equiv + margin vs the +10 bar (136.378).
  5. Proves the pin was LIVE in the served forward (lawine #681 requirement): records
     the import-time is_batch_invariant AND the actual 2D-vs-3D branch the served
     attention forward takes for decode (M=1) and verify (M>=2) shapes.

analysis_only: NO HF Job, NO submission, served file untouched. A clean cert is a
SURFACE + approval trigger, NOT a fire (the fire stays quality-blocked by the
int4-body AIME/GPQA gap, ubel #672/#679, wirbel #682).

Pin tiers (triton_unified_attention 2D/3D selector, vllm 0.22.0):
    use_3d = not (... or max_seqlen_q>1 or num_seqs>seq_threshold_3D or is_batch_invariant)
  - fixed2d: override seq_threshold_3D=0 at the unified_attention call site so the M=1
    decode (num_seqs>=1 > 0) takes the 2D single-pass path, matching the M>=2 verify
    forward; VLLM_BATCH_INVARIANT=0 (attention-only, faster tier). NOTE: land #684's
    MIN_LAUNCH_GRID_SIZE_2D=0 module patch alone does NOT reach the deployed decode --
    the builder rounds seq_threshold_3D to the nearest cudagraph capture size (>=1), so
    the num_seqs=1 decode stayed on 3D split-KV (PR #690 first cut: threshold 7, break
    53.78% ~= un-pinned). The call-site override (_stark_pin.py) is the robust pin.
  - bi1: VLLM_BATCH_INVARIANT=1 -> is_batch_invariant forces 2D unconditionally for every
    forward (the byte-identical floor tier; also pins marlin/layernorm/etc).
  - none: control (no pin) -> reproduces the unpinned break.

Usage:
    python research/validity/strict319_servedhead_cert/cert.py --pin fixed2d \
        --num-spec 6 --num-prompts 128 --output-len 512 --seed 1 \
        --ar-rung-local 126.75 --wandb-group strict319-servedhead-cert-stark
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

CERT_DIR = Path(__file__).resolve().parent
ROOT = CERT_DIR.parents[2]  # research/validity/<dir>/cert.py -> repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import greedy_gate, harness, paths  # noqa: E402

SUBMISSION = ROOT / "submissions" / "int4_mtp_batchinv"
SERVE_FILES = ["serve.py", "sitecustomize.py", "vllm_attn_group_patch.py", "manifest.json"]

# Strict anchors from the PR baseline.
OFFICIAL_ANCHOR = 126.378            # int4_g128_lmhead official GREEDY_IDENTICAL rung
PLUS10_BAR = 136.378                 # +10 bar
STARK_BASIS = 0.870                  # PR-named local->official-equiv stark basis


def build_pinned_serve(pin: str) -> Path:
    """Copy the submission and chain in the boot pin via the copy's sitecustomize.

    The original submission dir is never touched. The copy has identical manifest
    deps, so it reuses the same serve venv (no rebuild)."""
    serve_copy = CERT_DIR / f"_serve_{pin}"
    shutil.rmtree(serve_copy, ignore_errors=True)
    serve_copy.mkdir(parents=True, exist_ok=True)
    for f in SERVE_FILES:
        shutil.copy2(SUBMISSION / f, serve_copy / f)
    shutil.copy2(CERT_DIR / "_stark_pin.py", serve_copy / "_stark_pin.py")
    with open(serve_copy / "sitecustomize.py", "a") as fh:
        fh.write(
            "\n\n# stark #690 servedhead-cert pin "
            "(analysis-only; original submission untouched)\n"
            "import _stark_pin  # noqa: E402,F401\n"
        )
    return serve_copy


def arm_env(pin: str, num_spec: int, proof_dir: Path) -> dict:
    """Extra serve env for one arm. Pin is applied identically to ref and candidate;
    only NUM_SPECULATIVE_TOKENS differs between arms."""
    return {
        "STARK_PIN_MODE": pin,
        "STARK_PIN_PROOF_DIR": str(proof_dir),
        "NUM_SPECULATIVE_TOKENS": str(num_spec),
        # fixed2d/none isolate the attention path with BI OFF; bi1 turns full
        # batch-invariance ON. Overrides the manifest's VLLM_BATCH_INVARIANT=1.
        "VLLM_BATCH_INVARIANT": "1" if pin == "bi1" else "0",
    }


def run_arm(
    serve_copy: Path,
    server_python: Path,
    *,
    label: str,
    pin: str,
    num_spec: int,
    port: int,
    num_prompts: int,
    output_len: int,
    seed: int,
    tmp_dir: Path,
    measure_tps: bool,
) -> dict:
    """Boot the pinned serve for one arm, capture 128xL greedy tokens, return paths+proof."""
    proof_dir = tmp_dir / f"_pinproof_{pin}_{label}"
    proof_dir.mkdir(parents=True, exist_ok=True)
    out_file = tmp_dir / f"{pin}_{label}_decode.jsonl"
    summary_file = tmp_dir / f"{pin}_{label}_summary.json"
    log_path = tmp_dir / f"{pin}_{label}_serve.log"
    extra_env = arm_env(pin, num_spec, proof_dir)
    served_name = json.loads((serve_copy / "manifest.json").read_text())["served_model_name"]

    t0 = time.time()
    result: dict = {"label": label, "num_spec": num_spec, "port": port}
    with harness.LocalServer(
        serve_copy,
        server_python=server_python,
        port=port,
        log_path=log_path,
        extra_env=extra_env,
    ) as srv:
        boot_s = time.time() - t0
        print(f"[{label}] serve ready in {boot_s:.0f}s (num_spec={num_spec}, pin={pin})", flush=True)
        summary = harness.capture_decode(
            server_python,
            base_url=srv.base_url,
            model=srv.served_model_name,
            out_file=out_file,
            summary_file=summary_file,
            num_prompts=num_prompts,
            output_len=output_len,
            seed=seed,
        )
        result["decode_summary"] = summary
        result["capture_duration_s"] = summary.get("duration_s")
        result["num_completion_tokens"] = summary.get("num_completion_tokens")
        if measure_tps:
            try:
                result["tps"] = harness.probe_tps(srv.base_url, srv.served_model_name)
            except Exception as exc:  # noqa: BLE001
                result["tps_error"] = str(exc)
    result["out_file"] = str(out_file)
    result["proof"] = read_proof(proof_dir)
    result["boot_s"] = boot_s
    return result


def read_proof(proof_dir: Path) -> dict:
    """Aggregate the per-process pin proof files into a compact verdict."""
    proof: dict = {"min_launch_after": None, "is_batch_invariant_at_import": None, "branches": []}
    for fp in glob.glob(str(proof_dir / "pin_triton_attn_*.json")):
        try:
            rec = json.loads(Path(fp).read_text())
            proof["min_launch_after"] = rec.get("min_launch_after")
            proof["min_launch_before"] = rec.get("min_launch_before")
            proof["serve_env_VLLM_BATCH_INVARIANT"] = rec.get("serve_env_VLLM_BATCH_INVARIANT")
        except (OSError, ValueError):
            pass
    for fp in glob.glob(str(proof_dir / "pin_unified_import_*.json")):
        try:
            rec = json.loads(Path(fp).read_text())
            proof["is_batch_invariant_at_import"] = rec.get("is_batch_invariant_at_import")
        except (OSError, ValueError):
            pass
    branches: dict = {}
    for fp in glob.glob(str(proof_dir / "pin_branch_*.jsonl")):
        try:
            for line in Path(fp).read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                branches[(r["max_seqlen_q"], r["num_seqs"])] = r
        except (OSError, ValueError):
            pass
    proof["branches"] = sorted(branches.values(), key=lambda r: (r["max_seqlen_q"], r["num_seqs"]))
    # Decode (M=1) and verify (M>=2) kernel selection actually taken in the forward.
    decode = [b for b in proof["branches"] if b["max_seqlen_q"] == 1]
    verify = [b for b in proof["branches"] if b["max_seqlen_q"] > 1]
    proof["decode_kernel_2d"] = (bool(decode) and all(not b["use_3d_split_kv"] for b in decode)) or None
    proof["verify_kernel_2d"] = (bool(verify) and all(not b["use_3d_split_kv"] for b in verify)) or None
    proof["observed_decode_branch"] = decode[0] if decode else None
    proof["observed_verify_branch"] = verify[0] if verify else None
    return proof


def compare_streams(reference: Path, candidate: Path, output_len: int) -> dict:
    report = greedy_gate.compare(reference, candidate)
    onset = greedy_gate.onset_summary(report)
    total = report.total_tokens_compared or 0
    break_rate = (report.total_divergent_tokens / total) if total else float("nan")
    n_cmp = report.num_prompts_compared or 0
    seq_div = (report.num_divergent / n_cmp) if n_cmp else float("nan")
    return {
        "verdict": report.verdict,
        "greedy_identical": report.verdict == "GREEDY_IDENTICAL",
        "num_prompts_compared": report.num_prompts_compared,
        "num_identical": report.num_identical,
        "num_divergent": report.num_divergent,
        "total_tokens_compared": report.total_tokens_compared,
        "total_divergent_tokens": report.total_divergent_tokens,
        "break_rate": break_rate,
        "seq_divergence_rate": seq_div,
        "first_div_min": onset.get("onset_min"),
        "first_div_median": onset.get("onset_median"),
        "first_div_max": onset.get("onset_max"),
        "onset_line": greedy_gate.onset_line(onset, output_len),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pin", choices=["fixed2d", "bi1", "none"], required=True)
    ap.add_argument("--num-spec", type=int, default=6,
                    help="candidate NUM_SPECULATIVE_TOKENS (deployed manifest=6; PR 'K=5/M=6')")
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--seed", type=int, default=paths.SEED)
    ap.add_argument("--ar-rung-local", type=float, default=126.75)
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--wandb-group", default="strict319-servedhead-cert-stark")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny: 2 prompts x 16 tokens")
    args = ap.parse_args()

    if args.smoke:
        args.num_prompts, args.output_len = 2, 16

    notes = paths.prepare_local_gpu_env()
    for n in notes:
        print(f"[env] {n}", flush=True)

    tmp_dir = Path("/tmp") / f"stark_cert_{args.pin}_{int(time.time())}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    serve_copy = build_pinned_serve(args.pin)
    manifest = harness.load_manifest(serve_copy)
    print(f"[venv] resolving serve venv for {manifest['dependencies']}", flush=True)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    print(f"[venv] {server_python}", flush=True)

    # Reference arm: plain M=1 AR (spec OFF), same pin. Candidate arm: spec ON.
    ref = run_arm(
        serve_copy, server_python, label="ref", pin=args.pin, num_spec=0,
        port=args.port, num_prompts=args.num_prompts, output_len=args.output_len,
        seed=args.seed, tmp_dir=tmp_dir, measure_tps=False,
    )
    cand = run_arm(
        serve_copy, server_python, label="cand", pin=args.pin, num_spec=args.num_spec,
        port=args.port + 1, num_prompts=args.num_prompts, output_len=args.output_len,
        seed=args.seed, tmp_dir=tmp_dir, measure_tps=True,
    )

    cmp = compare_streams(Path(ref["out_file"]), Path(cand["out_file"]), args.output_len)

    tps_block = cand.get("tps") or {}
    local_tps = tps_block.get("decode_tps_single_stream")
    official_equiv_870 = (local_tps * STARK_BASIS) if local_tps else None
    ratio = OFFICIAL_ANCHOR / args.ar_rung_local
    official_equiv_ratio = (local_tps * ratio) if local_tps else None
    margin_870 = (official_equiv_870 - PLUS10_BAR) if official_equiv_870 else None

    cand_proof = cand.get("proof", {})
    ref_proof = ref.get("proof", {})
    # pin_active is the lawine #681 "pin reached the forward" gate. For fixed2d the
    # authoritative evidence is the OBSERVED branch: the M=1 decode AND M>=2 verify
    # forwards must both have taken 2D single-pass on BOTH arms (the constant being
    # patched to 0 is necessary but NOT sufficient -- the builder's cudagraph rounding
    # defeated it in the first cut). For bi1 it is the import-time is_batch_invariant.
    pin_active = bool(
        (
            args.pin == "fixed2d"
            and cand_proof.get("decode_kernel_2d") is True
            and cand_proof.get("verify_kernel_2d") is True
            and ref_proof.get("decode_kernel_2d") is True
        )
        or (args.pin == "bi1" and cand_proof.get("is_batch_invariant_at_import"))
        or (args.pin == "none")
    )

    result = {
        "pin": args.pin,
        "num_spec_candidate": args.num_spec,
        "num_prompts": args.num_prompts,
        "output_len": args.output_len,
        "seed": args.seed,
        "model_id": "google/gemma-4-E4B-it-qat-w4a16-ct",
        "analysis_only": True,
        "official_tps": 0,
        "no_hf_job": 1,
        "fires": False,
        "servedhead_break_rate": cmp["break_rate"],
        "servedhead_seq_divergence": cmp["seq_divergence_rate"],
        "greedy_verdict": cmp["verdict"],
        "greedy_identical": cmp["greedy_identical"],
        "first_div_median": cmp["first_div_median"],
        "first_div_min": cmp["first_div_min"],
        "first_div_max": cmp["first_div_max"],
        "num_identical": cmp["num_identical"],
        "num_divergent": cmp["num_divergent"],
        "total_divergent_tokens": cmp["total_divergent_tokens"],
        "total_tokens_compared": cmp["total_tokens_compared"],
        "servedhead_local_tps": local_tps,
        "official_equiv_870": official_equiv_870,
        "official_equiv_ratio": official_equiv_ratio,
        "margin_vs_plus10_870": margin_870,
        "ar_rung_local": args.ar_rung_local,
        "official_anchor": OFFICIAL_ANCHOR,
        "plus10_bar": PLUS10_BAR,
        "pin_active": pin_active,
        "pin_min_launch_after_cand": cand_proof.get("min_launch_after"),
        "pin_is_batch_invariant_forward_cand": cand_proof.get("is_batch_invariant_at_import"),
        "pin_decode_kernel_2d_cand": cand_proof.get("decode_kernel_2d"),
        "pin_verify_kernel_2d_cand": cand_proof.get("verify_kernel_2d"),
        "pin_decode_kernel_2d_ref": ref_proof.get("decode_kernel_2d"),
        "serve_env_VLLM_BATCH_INVARIANT_cand": cand_proof.get("serve_env_VLLM_BATCH_INVARIANT"),
        "observed_decode_branch_cand": cand_proof.get("observed_decode_branch"),
        "observed_verify_branch_cand": cand_proof.get("observed_verify_branch"),
        "ref_boot_s": ref.get("boot_s"),
        "cand_boot_s": cand.get("boot_s"),
        "onset_line": cmp["onset_line"],
    }

    out_summary = CERT_DIR / "results" / f"cert_{args.pin}.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str), flush=True)
    print(f"[cert] wrote {out_summary}", flush=True)
    print(f"[cert] {cmp['onset_line']}", flush=True)

    if not args.no_wandb:
        log_wandb(args, result)

    # Disk hygiene: bulky captures live in /tmp (off-overlay); drop them now.
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return 0


def log_wandb(args, result: dict) -> None:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] unavailable: {exc}", flush=True)
        return
    entity = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
    project = os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai")
    config = {
        "design": "served strict-#319 cert: spec(K)=ON vs AR(spec OFF), SAME pinned stack",
        "submission": "int4_mtp_batchinv",
        "model_id": "google/gemma-4-E4B-it-qat-w4a16-ct",
        "drafter": "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant",
        "served_head": "pruned-16k lm_head (deployed)",
        "pin": args.pin,
        "num_speculative_tokens": args.num_spec,
        "vllm_batch_invariant": 1 if args.pin == "bi1" else 0,
        "vllm_version": "0.22.0",
        "num_prompts": args.num_prompts,
        "output_len": args.output_len,
        "seed": args.seed,
        "analysis_only": True,
        "official_tps": 0,
        "no_hf_job": 1,
        "fires": False,
        "claim_axis": "greedy #319 byte-exact served-head identity (NOT a fire; quality-blocked)",
        "ar_rung_local": args.ar_rung_local,
    }
    run = wandb.init(
        project=project, entity=entity, group=args.wandb_group,
        name=f"stark/servedhead-cert-{args.pin}", job_type="strict319-servedhead-cert",
        config=config, reinit=True,
    )
    scalars = {k: v for k, v in result.items() if isinstance(v, (int, float, bool)) and v is not None}
    # Mirror headline metrics under summary/ (project convention).
    for k in ("servedhead_break_rate", "servedhead_seq_divergence", "servedhead_local_tps",
              "official_equiv_870", "margin_vs_plus10_870", "pin_active"):
        if result.get(k) is not None:
            scalars[f"summary/{k}"] = result[k]
    run.log(scalars)
    run.summary["greedy_verdict"] = result["greedy_verdict"]
    run.summary["onset_line"] = result["onset_line"]
    run.summary["analysis_only"] = True
    run.summary["official_tps"] = 0
    run.summary["no_hf_job"] = 1
    run.summary["fires"] = False
    print(f"[wandb] run {run.id} ({run.name})", flush=True)
    run.finish()


if __name__ == "__main__":
    raise SystemExit(main())
