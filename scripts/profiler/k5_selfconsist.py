#!/usr/bin/env python
"""PR #720 -- independent strict #319 self-consistency identity audit of the
un-rescued K=5 MTP spec **fire candidate** (`int4_mtp_batchinv`).

Why the un-rescued config certifies the *rescued* acceptor too: under
``VLLM_BATCH_INVARIANT=1`` the greedy rejection sampler short-circuits to the
target argmax at every draft position (this submission's own serve.py docstring,
L8-10), so the recompute-acceptor's M=1 re-verify is a *measured no-op* -- stark
#727 finding (4) + WRITEBACK_INFEASIBLE_LOCAL. The acceptor can only re-touch
draft rows; it can never reach a prefill pos-0 flip (no draft row there). So the
emitted stream of the rescued config == the un-rescued config, and the identity
of the un-rescued config is the identity of the thing we'd fire.

Three legs, all on **vLLM 0.22.0 --enforce-eager** (the faithful substrate; NOT
dev307, accuracy-invalid per #606), single-stream **cc=1** + **BI=1** (kanna
#699: cc>1 greedy is engine-corrupt; faithful reference is cc=1/BI=1):

  ar_a : SENPAI_REFERENCE_MODE=1 -> drafter OFF, plain int4 M=1 AR greedy.
         The canonical own-AR reference (BASELINE L10: served spec-off, eager).
  ar_b : a SECOND independent ar run -- determinism floor. Catches int4 exact-tie
         run-to-run non-determinism (#654) so a spec "break" is never mis-blamed
         on the drafter when it's really the engine.
  spec : NUM_SPECULATIVE_TOKENS=5 drafter ON -- the fire candidate's served-fast
         greedy. This is what the leaderboard would serve.

Verdict (keyed on dataset_index; seed permutes request order so plain index is
not stable):
  spec == ar_a  128/128                     -> K5_STRICT_SELF_CONSISTENT
  spec != ar_a, every first-divergence is pos-0 (prefill) and the floor is clean
                                            -> K5_DRAFT_CLEAN_PREFILL_TIE_ONLY
                                               (strict_literal_holds=0; the only
                                               residual is the #654 int4 prefill
                                               exact-tie, uncatchable by any
                                               acceptor)
  spec != ar_a with a post-prefill (draft) first-divergence on a clean floor
                                            -> K5_DRAFT_DIRTY (genuine spec break)
  ar_a != ar_b                              -> ENGINE_NONDETERMINISTIC (guarded)

Resumable: a leg whose ``<leg>.jsonl`` exists is reused unless ``--force``.
Writes ``<out>/result.json`` for the wandb logger. analysis_only; no fire.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.local_validation import harness, paths  # noqa: E402
from scripts.profiler.dual_substrate_selfconsist import (  # noqa: E402
    _terminate, _used_mib, batch_invariance_health, cudagraph_captured,
    preflight, serve_env,
)

V0220 = Path("/tmp/senpai-venvs/20f658587e8a6643/bin/python")  # vLLM 0.22.0 faithful
SUBMISSION = ROOT / "submissions" / "int4_mtp_batchinv"
BODY_ID = "google/gemma-4-E4B-it-qat-w4a16-ct"
DRAFTER_ID = "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant"
SERVED = paths.DEFAULT_SERVED_NAME
PORT = 8129

# #654 canonical int4 prefill exact-tie prompts (the un-rescued K=5's known 2
# sequence-level flips; stark #669/#727 prefill_margin proved them exact-ties:
# 37227f6b "The"<->"Here" batch-shape logit-swap; 74200cad non-deterministic).
TIE_PROMPT_PREFIXES = ("37227f6b", "74200cad")

LEG_SPEC = {
    # label -> (reference_mode, num_spec)
    "ar_a": (True, 0),    # spec-off M=1 AR reference
    "ar_b": (True, 0),    # spec-off M=1 AR -- determinism floor
    "spec": (False, 5),   # K=5 MTP drafter on -- fire candidate
}


# --------------------------------------------------------------------------- serve
def serve_k5(label: str, *, reference_mode: bool, num_spec: int, log_path: Path,
             port: int = PORT, startup_timeout_s: int = 1200):
    """Boot the int4 body on 0.22.0 eager; attach the MTP drafter for the spec leg.

    Mirrors submissions/int4_mtp_batchinv/serve.py: PYTHONPATH must include the
    submission dir so its sitecustomize.py auto-loads the attention-group
    num_heads backport (the {8,4} draft/target group assertion); no-op when
    speculation is off, required when on.
    """
    import signal
    import subprocess
    import urllib.error
    import urllib.request

    harness.ensure_serving_http_compat(V0220)
    args = [
        str(V0220), "-m", "vllm.entrypoints.openai.api_server",
        "--model", BODY_ID, "--served-model-name", SERVED,
        "--host", "127.0.0.1", "--port", str(port),
        "--dtype", "bfloat16", "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.90",
        "--max-num-seqs", "1", "--max-num-batched-tokens", "512",
        "--trust-remote-code", "--no-enable-log-requests",
        "--enforce-eager",
    ]
    if not reference_mode and num_spec > 0:
        spec_config = {"model": DRAFTER_ID, "num_speculative_tokens": num_spec}
        args += ["--speculative-config", json.dumps(spec_config)]

    env = serve_env(bi=True)
    env["PYTHONPATH"] = str(SUBMISSION) + os.pathsep + env.get("PYTHONPATH", "")
    if reference_mode:
        env[paths.REFERENCE_MODE_ENV] = "1"
    else:
        env.pop(paths.REFERENCE_MODE_ENV, None)

    log = open(log_path, "w")
    print(f"[serve] {label}: ref_mode={reference_mode} num_spec={num_spec} "
          f"eager=1 bi=1 cc=1 -> {BODY_ID}", flush=True)
    proc = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT,
                            text=True, preexec_fn=os.setsid)
    base = f"http://127.0.0.1:{port}"
    t0 = time.time()
    deadline = time.time() + startup_timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            log.flush()
            raise RuntimeError(f"server {label} exited code {proc.returncode} "
                               f"before ready; see {log_path}")
        try:
            with urllib.request.urlopen(f"{base}/v1/models", timeout=5) as r:
                if r.status == 200:
                    print(f"[serve] {label} ready in {time.time()-t0:.0f}s", flush=True)
                    return proc, base, log
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(5)
    _terminate(proc, log)
    raise RuntimeError(f"{label} endpoint not ready after {startup_timeout_s}s; see {log_path}")


# --------------------------------------------------------------- read / compare
def read_full(jsonl: Path) -> dict:
    """dataset_index -> {'sha','ids','id','prompt_ids'}."""
    out = {}
    for line in Path(jsonl).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out[int(r["dataset_index"])] = {
            "sha": r["completion_token_sha256"],
            "ids": r["completion_token_ids"],
            "id": r.get("id", ""),
            "prompt_ids": r.get("prompt_token_ids", []),
        }
    return out


def first_divergence(a: list, b: list) -> int | None:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def compare_streams(ref: dict, cur: dict) -> dict:
    keys = sorted(set(ref) & set(cur))
    seq_match = [k for k in keys if ref[k]["sha"] == cur[k]["sha"]]
    mism = [k for k in keys if ref[k]["sha"] != cur[k]["sha"]]
    # position-wise first-divergence decomposition over the mismatched seqs.
    pos0, postpref = [], []
    div_detail = []
    for k in mism:
        fd = first_divergence(ref[k]["ids"], cur[k]["ids"])
        rid = cur[k]["id"]
        is_tie = any(rid.startswith(p) or p in rid for p in TIE_PROMPT_PREFIXES)
        rec = {"dataset_index": k, "id": rid, "first_div": fd,
               "ref_tok": ref[k]["ids"][fd] if fd is not None and fd < len(ref[k]["ids"]) else None,
               "cur_tok": cur[k]["ids"][fd] if fd is not None and fd < len(cur[k]["ids"]) else None,
               "known_tie_prompt": is_tie}
        div_detail.append(rec)
        (pos0 if fd == 0 else postpref).append(k)
    return {
        "n_total": len(keys),
        "n_seq_match": len(seq_match),
        "n_seq_mismatch": len(mism),
        "strict_holds": (len(keys) > 0 and len(mism) == 0),
        "n_first_div_pos0": len(pos0),
        "n_first_div_postprefill": len(postpref),
        "postprefill_dataset_indices": sorted(postpref),
        "divergence_detail": sorted(div_detail, key=lambda d: d["dataset_index"]),
    }


# ------------------------------------------------------------------- gap probe
def _complete_logprobs(base: str, prompt_ids: list, *, n_logprobs: int = 20,
                       timeout_s: int = 60) -> dict:
    """One greedy decode step over an explicit token-id context; return the
    top-N next-token logprobs dict (token_str -> logprob)."""
    import urllib.request

    body = json.dumps({
        "model": SERVED, "prompt": prompt_ids, "max_tokens": 1,
        "temperature": 0.0, "logprobs": n_logprobs,
    }).encode()
    req = urllib.request.Request(f"{base}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        resp = json.loads(r.read())
    lp = resp["choices"][0]["logprobs"]
    top = lp.get("top_logprobs") or [{}]
    gen_tok = (lp.get("tokens") or [None])[0]
    return {"top_logprobs": top[0], "gen_token": gen_tok}


def gap_probe(out_dir: Path, ar_a: dict, sva: dict, *, tau: float, force: bool) -> dict:
    """Independently measure ``confident_genuine_flips@tau``.

    For every spec-vs-ar first-divergence, re-prefill the *reference* (M=1 AR,
    REFERENCE_MODE=1) over the exact own-AR context up to the flip and read the
    next-token top-2 logprob margin. A divergence is a *near-tie* (an #654 int4
    batch-shape artefact, uncatchable by any acceptor) iff ``top1-top2 <= tau``;
    it is a *confident genuine flip* iff ``top1-top2 > tau`` -- the reference was
    >tau-confident in its greedy token yet the served stream emitted a different
    one. The advisor's fire metric is ``confident_genuine_flips == 0`` at
    tau=0.3 nat. No string matching: a >tau top-2 margin at a flip is a confident
    decision regardless of which token won (conservative for confidence).
    """
    out_path = out_dir / "gap_probe.json"
    if out_path.exists() and not force:
        print(f"[gap_probe] reuse {out_path.name}", flush=True)
        return json.loads(out_path.read_text())

    divs = sva.get("divergence_detail", [])
    flips = [d for d in divs if d["first_div"] is not None]
    if not flips:
        res = {"tau": tau, "n_probed": 0, "confident_genuine_flips": 0,
               "max_gap": None, "records": []}
        out_path.write_text(json.dumps(res, indent=2))
        return res

    preflight()
    proc, base, log = serve_k5("gap_probe", reference_mode=True, num_spec=0,
                               log_path=out_dir / "gap_probe.serve.log")
    records = []
    try:
        time.sleep(2)
        for d in flips:
            k, fd = d["dataset_index"], d["first_div"]
            ent = ar_a.get(k)
            if ent is None:
                continue
            ctx = list(ent["prompt_ids"]) + list(ent["ids"][:fd])
            try:
                pr = _complete_logprobs(base, ctx)
            except Exception as e:  # noqa: BLE001
                records.append({"dataset_index": k, "first_div": fd,
                                "error": str(e)})
                continue
            vals = sorted(pr["top_logprobs"].values(), reverse=True)
            gap = (vals[0] - vals[1]) if len(vals) >= 2 else float("inf")
            records.append({
                "dataset_index": k, "first_div": fd, "id": d["id"],
                "ref_tok": d["ref_tok"], "cur_tok": d["cur_tok"],
                "known_tie_prompt": d["known_tie_prompt"],
                "top1_logprob": vals[0] if vals else None,
                "top2_logprob": vals[1] if len(vals) >= 2 else None,
                "gap_nat": gap, "confident": bool(gap > tau),
            })
    finally:
        _terminate(proc, log)

    probed = [r for r in records if "gap_nat" in r]
    confident = [r for r in probed if r["confident"]]
    finite = [r["gap_nat"] for r in probed if r["gap_nat"] != float("inf")]
    res = {
        "tau": tau,
        "n_probed": len(probed),
        "n_errors": len(records) - len(probed),
        "confident_genuine_flips": len(confident),
        "max_gap": max(finite) if finite else None,
        "confident_records": [
            {"dataset_index": r["dataset_index"], "first_div": r["first_div"],
             "id": r["id"], "gap_nat": r["gap_nat"]} for r in confident
        ],
        "records": records,
    }
    out_path.write_text(json.dumps(res, indent=2))
    return res


# ----------------------------------------------------------------------------- legs
def run_leg(label: str, out_dir: Path, *, num_prompts: int, output_len: int,
            seed: int, force: bool, health: bool) -> dict:
    reference_mode, num_spec = LEG_SPEC[label]
    jsonl = out_dir / f"{label}.jsonl"
    summ = out_dir / f"{label}.summary.json"
    log_path = out_dir / f"{label}.serve.log"
    meta_path = out_dir / f"{label}.meta.json"
    if jsonl.exists() and summ.exists() and meta_path.exists() and not force:
        print(f"[leg {label}] reuse existing {jsonl.name}", flush=True)
        return json.loads(meta_path.read_text())

    preflight()
    proc, base, log = serve_k5(label, reference_mode=reference_mode,
                               num_spec=num_spec, log_path=log_path)
    meta = {"label": label, "reference_mode": reference_mode, "num_spec": num_spec,
            "venv": "20f658587e8a6643", "enforce_eager": True, "bi": True,
            "num_prompts": num_prompts, "output_len": output_len, "seed": seed}
    try:
        time.sleep(2)
        meta["cudagraph_captured"] = cudagraph_captured(log_path)
        if health:
            try:
                meta["batch_invariance"] = batch_invariance_health(base)
            except Exception as e:
                meta["batch_invariance"] = {"error": str(e)}
        t0 = time.time()
        summary = harness.capture_decode(
            V0220, base_url=base, model=SERVED, out_file=jsonl, summary_file=summ,
            num_prompts=num_prompts, output_len=output_len, seed=seed,
            timeout_s=max(900, int(num_prompts * output_len / 6) + 600),
        )
        meta["decode_wall_s"] = round(time.time() - t0, 1)
        meta["tps"] = summary.get("tps")
        meta["num_records"] = summary.get("num_records")
        meta["gpu_used_mib_peak"] = _used_mib()
    finally:
        _terminate(proc, log)
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def classify(out_dir: Path) -> dict:
    legs = {lab: read_full(out_dir / f"{lab}.jsonl")
            for lab in LEG_SPEC if (out_dir / f"{lab}.jsonl").exists()}
    res = {}
    if "ar_a" in legs and "ar_b" in legs:
        res["floor_ab"] = compare_streams(legs["ar_a"], legs["ar_b"])
    if "ar_a" in legs and "spec" in legs:
        res["spec_vs_ar"] = compare_streams(legs["ar_a"], legs["spec"])

    verdict = "INCOMPLETE"
    sva = res.get("spec_vs_ar")
    floor = res.get("floor_ab")
    floor_clean = bool(floor and floor["strict_holds"])
    if sva is not None:
        if sva["strict_holds"]:
            verdict = "K5_STRICT_SELF_CONSISTENT"
        elif floor is not None and not floor_clean:
            verdict = "ENGINE_NONDETERMINISTIC"
        elif sva["n_first_div_postprefill"] == 0:
            verdict = "K5_DRAFT_CLEAN_PREFILL_TIE_ONLY"
        else:
            verdict = "K5_DRAFT_DIRTY"
    res["verdict"] = verdict
    res["floor_clean"] = floor_clean
    res["strict_literal_holds"] = bool(sva and sva["strict_holds"])
    # is every residual flip on a known #654 tie prompt?
    if sva is not None:
        nontie = [d for d in sva["divergence_detail"] if not d["known_tie_prompt"]]
        res["all_residual_flips_known_ties"] = (len(nontie) == 0 and sva["n_seq_mismatch"] > 0)
        res["residual_flips_not_known_ties"] = [
            {"dataset_index": d["dataset_index"], "id": d["id"], "first_div": d["first_div"]}
            for d in nontie
        ]
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "research/validity/selfconsist_720/k5_identity")
    ap.add_argument("--legs", default="ar_a,ar_b,spec")
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-health", action="store_true")
    ap.add_argument("--no-gap-probe", action="store_true")
    ap.add_argument("--gap-tau", type=float, default=0.3,
                    help="confident-flip threshold in nats (advisor fire metric: 0.3)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    legs = [l.strip() for l in args.legs.split(",") if l.strip()]
    assert all(l in LEG_SPEC for l in legs), f"bad legs {legs}"

    metas = {}
    for leg in legs:
        metas[leg] = run_leg(leg, args.out_dir, num_prompts=args.num_prompts,
                             output_len=args.output_len, seed=args.seed,
                             force=args.force, health=not args.no_health)

    cls = classify(args.out_dir)

    # Independent confident_genuine_flips@tau measurement over the residual flips.
    gp = None
    sva = cls.get("spec_vs_ar")
    if (not args.no_gap_probe and sva is not None and sva["n_seq_mismatch"] > 0
            and (args.out_dir / "ar_a.jsonl").exists()):
        ar_a = read_full(args.out_dir / "ar_a.jsonl")
        gp = gap_probe(args.out_dir, ar_a, sva, tau=args.gap_tau, force=args.force)

    result = {
        "pr": 720, "analysis_only": True, "official_tps": 0, "no_hf_job": True, "fires": False,
        "config": "int4_mtp_batchinv", "substrate": "vllm-0.22.0-enforce-eager-cc1-bi1",
        "num_prompts": args.num_prompts, "output_len": args.output_len, "seed": args.seed,
        "legs": legs, "legs_meta": metas, **cls,
    }
    if gp is not None:
        result["gap_probe"] = gp
        result["confident_genuine_flips"] = gp["confident_genuine_flips"]
        result["gap_tau"] = gp["tau"]
        result["max_residual_gap_nat"] = gp["max_gap"]
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print("\n===== K5 IDENTITY RESULT =====", flush=True)
    print(json.dumps({k: result.get(k) for k in (
        "verdict", "strict_literal_holds", "floor_clean",
        "all_residual_flips_known_ties", "residual_flips_not_known_ties",
        "confident_genuine_flips", "gap_tau", "max_residual_gap_nat")}, indent=2), flush=True)
    if gp is not None:
        print(f"  gap_probe: probed={gp['n_probed']} "
              f"confident@{gp['tau']}={gp['confident_genuine_flips']} "
              f"max_gap={gp['max_gap']}", flush=True)
    for k in ("floor_ab", "spec_vs_ar"):
        if k in result:
            c = result[k]
            print(f"  {k}: seq {c['n_seq_match']}/{c['n_total']} match "
                  f"(mismatch {c['n_seq_mismatch']}; first_div pos0={c['n_first_div_pos0']} "
                  f"postprefill={c['n_first_div_postprefill']})", flush=True)
    for leg, m in metas.items():
        bih = m.get("batch_invariance", {})
        print(f"  leg {leg}: tps={m.get('tps')} wall={m.get('decode_wall_s')}s "
              f"recs={m.get('num_records')} health={bih.get('healthy') if isinstance(bih, dict) else None}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
