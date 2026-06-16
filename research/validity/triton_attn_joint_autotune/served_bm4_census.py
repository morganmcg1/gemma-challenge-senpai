"""Paired greedy-identity census + PPL for the bm4 lever (PR #442, wirbel).

Proves the bm4 launch override (BLOCK_M 16->4 + num_stages 3->2) is BYTE-EXACT on the
*served* M=8 verify path -- empirically, not by assertion.

THE CANCELLING STRUCTURE (the whole point): vLLM v0.22.1rc1 cudagraph-captures only
sizes [1,2], so the size-8 (M=8) verify step runs EAGER. Comparing either arm's
absolute token agreement against a captured size-[1,2] reference yields a spurious
identity (lawine #438 measured 0.4143). So instead we serve the SAME deployed stack
TWICE on the SAME spec-ON M=8 verify path -- default (BLOCK_M=16) vs bm4 (BLOCK_M=4) --
and compare the two arms token-by-token against EACH OTHER. Any eager-path artifact is
common to both arms and CANCELS; a token divergence is then attributable ONLY to the
bm4 tiling change. Byte-exact <=> 100% token-identical across >=50 prompts.

Plus: PPL on the bm4 arm (official ppl_endpoint.py) must clear the <=2.42 gate. Since
bm4 is byte-exact, its PPL equals the deployed ~2.3772 anchor; we MEASURE it to confirm.

Uses the same temporary env-gated reverted ``sitecustomize.py`` toggle as
``served_bm4_wall_ab.py`` (imported, not duplicated): WIRBEL_BM4_AB unset -> deployed,
=1 -> bm4. Reverted, NEVER submitted. NOT an HF Job, NOT a submission, NOT a launch.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
from scripts.local_validation import harness, paths  # noqa: E402
from research.validity.triton_attn_joint_autotune import served_bm4_wall_ab as ab  # noqa: E402

SUBMISSION = "fa2sw_precache_kenyan"
PPL_GATE = 2.42
PPL_ANCHOR = 2.3772


def _log(msg: str) -> None:
    print(f"[bm4-census] {msg}", file=sys.stderr, flush=True)


def _load_tokens(jsonl: Path) -> dict[str, dict[str, Any]]:
    """id -> {completion_token_ids, prompt_token_sha256}. Keyed by record id so the
    two arms align prompt-for-prompt regardless of file order."""
    out: dict[str, dict[str, Any]] = {}
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[str(r["id"])] = {
            "completion_token_ids": list(r.get("completion_token_ids") or []),
            "prompt_token_sha256": r.get("prompt_token_sha256"),
        }
    return out


def compare_arms(default_jsonl: Path, bm4_jsonl: Path) -> dict[str, Any]:
    a = _load_tokens(default_jsonl)
    b = _load_tokens(bm4_jsonl)
    common = sorted(set(a) & set(b))
    n_prompts = len(common)
    n_identical = 0
    n_prompt_mismatch = 0
    total_tokens = 0
    matched_tokens = 0
    first_div: dict[str, Any] | None = None
    per_prompt = []
    for rid in common:
        da, db = a[rid], b[rid]
        # the two arms must have seen the same prompt tokens (same workload/seed)
        if da["prompt_token_sha256"] != db["prompt_token_sha256"]:
            n_prompt_mismatch += 1
        ta, tb = da["completion_token_ids"], db["completion_token_ids"]
        total_tokens += max(len(ta), len(tb))
        # longest common prefix
        lcp = 0
        for x, y in zip(ta, tb):
            if x != y:
                break
            lcp += 1
        matched_tokens += lcp
        identical = (ta == tb)
        if identical:
            n_identical += 1
        elif first_div is None:
            first_div = {"id": rid, "first_divergent_token_index": lcp,
                         "len_default": len(ta), "len_bm4": len(tb),
                         "default_tok": ta[lcp] if lcp < len(ta) else None,
                         "bm4_tok": tb[lcp] if lcp < len(tb) else None}
        per_prompt.append({"id": rid, "identical": identical, "lcp": lcp,
                           "len_default": len(ta), "len_bm4": len(tb)})
    return {
        "n_prompts": n_prompts,
        "n_identical": n_identical,
        "n_divergent": n_prompts - n_identical,
        "n_prompt_token_mismatch": n_prompt_mismatch,
        "frac_identical": (n_identical / n_prompts) if n_prompts else None,
        "total_tokens": total_tokens,
        "matched_token_prefix": matched_tokens,
        "frac_token_prefix_match": (matched_tokens / total_tokens) if total_tokens else None,
        "byte_exact": (n_prompts > 0 and n_identical == n_prompts and n_prompt_mismatch == 0),
        "first_divergence": first_div,
        "per_prompt": per_prompt,
    }


def _serve_and_capture(server_python, out_dir: Path, label: str, extra_env: dict[str, str],
                       args, want_ppl: bool) -> dict[str, Any]:
    arm_dir = out_dir / label
    arm_dir.mkdir(parents=True, exist_ok=True)
    log = arm_dir / "server.log"
    decode_out = arm_dir / "decode_outputs.jsonl"
    decode_sum = arm_dir / "decode_summary.json"
    sub_dir = (ROOT / "submissions" / SUBMISSION).resolve()
    info: dict[str, Any] = {"label": label, "extra_env": extra_env}
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
        if want_ppl:
            ppl_out = arm_dir / "ppl_results.jsonl"
            ppl_sum = arm_dir / "ppl_summary.json"
            ppl = harness.run_ppl(server_python, base_url=srv.base_url,
                                  model=srv.served_model_name,
                                  out_file=ppl_out, summary_file=ppl_sum,
                                  timeout_s=args.ppl_timeout_s)
            info["ppl"] = ppl.get("ppl")
            info["ppl_num_tokens"] = ppl.get("num_tokens")
    info["decode_jsonl"] = str(decode_out)
    # scrape the bm4 attestation from the server log
    t = log.read_text(errors="replace")
    info["bm4_patched"] = "[bm4-ab] PATCHED" in t
    info["bm4_forced_hits"] = t.count("[bm4-ab] CENSUS forced[") + t.count("[bm4-ab] forced bm4 (count=")
    info["bm4_hook_failed"] = "[bm4-ab] HOOK FAILED" in t
    info["splitkv"] = "[splitkv-verify] wrapped unified_attention" in t
    info["census_3d"] = ["IS_3D=True" in ln for ln in t.splitlines() if "CENSUS forced[" in ln]
    # per-head addressability at the served M=8 verify (corrects the 7/42 FA_SLIDING premise:
    # FA_SLIDING_DIAG=0 keeps sliding head-256 layers on Triton at verify; FA2 only takes M=1).
    _heads = [ln.split("head=")[1].split()[0] for ln in t.splitlines()
              if "CENSUS forced[" in ln and "head=" in ln]
    info["census_heads"] = {"head256": _heads.count("256"), "head512": _heads.count("512"),
                            "logged": len(_heads)}
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-prompts", type=int, default=64, help="prompts to census (>=50)")
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--block-m", dest="block_m", type=int, default=4)
    ap.add_argument("--num-stages", dest="num_stages", type=int, default=2)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ppl-timeout-s", type=int, default=2400)
    ap.add_argument("--no-ppl", action="store_true", help="skip the PPL gate (token census only)")
    ap.add_argument("--out-root", type=Path, default=HERE / "census_out")
    ap.add_argument("--no-toggle", action="store_true")
    ap.add_argument("--wandb_group", default="triton-joint-autotune")
    ap.add_argument("--wandb_name", default="wirbel/served-bm4-census")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    if args.num_prompts < 50:
        _log(f"WARN: --num-prompts {args.num_prompts} < 50 (advisor minimum)")

    for note in paths.prepare_local_gpu_env():
        _log(note)

    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = harness.load_manifest((ROOT / "submissions" / SUBMISSION).resolve())
    server_python = harness.ensure_server_venv(manifest["dependencies"])

    _log(f"census: {args.num_prompts}x{args.output_len} seed={args.seed} "
         f"cfg=bm{args.block_m}_s{args.num_stages} -> {out_root}")

    t0 = time.time()
    original_bytes = None
    toggled = False
    arms: dict[str, Any] = {}
    try:
        if not args.no_toggle:
            ab.ensure_clean_toggle()
            original_bytes = ab.apply_toggle()
            toggled = True
        # Arm A: deployed default (WIRBEL_BM4_AB unset) — no PPL needed (it's the anchor).
        arms["default"] = _serve_and_capture(server_python, out_root, "default", {}, args,
                                             want_ppl=False)
        # Arm B: bm4 (WIRBEL_BM4_AB=1) — capture + PPL gate.
        bm4_env = {"WIRBEL_BM4_AB": "1",
                   "WIRBEL_BM4_BLOCK_M": str(args.block_m),
                   "WIRBEL_BM4_NUM_STAGES": str(args.num_stages)}
        arms["bm4"] = _serve_and_capture(server_python, out_root, "bm4", bm4_env, args,
                                        want_ppl=not args.no_ppl)
    finally:
        toggle_clean = ab.revert_toggle(original_bytes) if toggled and original_bytes is not None else True

    cmp = compare_arms(Path(arms["default"]["decode_jsonl"]), Path(arms["bm4"]["decode_jsonl"]))
    ppl_bm4 = arms["bm4"].get("ppl")
    ppl_ok = (ppl_bm4 is not None and ppl_bm4 <= PPL_GATE) if not args.no_ppl else None

    # attestation: bm4 arm actually ran bm4 on the 3D verify; default arm did not.
    bm4_applied = bool(arms["bm4"]["bm4_patched"] and arms["bm4"]["bm4_forced_hits"] > 0
                       and not arms["bm4"]["bm4_hook_failed"])
    default_clean = not arms["default"]["bm4_patched"] and not arms["default"]["bm4_hook_failed"]
    splitkv_both = bool(arms["default"]["splitkv"] and arms["bm4"]["splitkv"])
    served_3d = bool(arms["bm4"]["census_3d"]) and all(arms["bm4"]["census_3d"])

    verdict_pass = bool(cmp["byte_exact"] and bm4_applied and default_clean
                        and splitkv_both and served_3d and toggle_clean
                        and (ppl_ok is None or ppl_ok))

    result = {
        "experiment": "served_bm4_census", "pr": 442, "student": "wirbel",
        "config": {"block_m": args.block_m, "tile": 32, "num_warps": 4, "num_stages": args.num_stages},
        "workload": {"num_prompts": args.num_prompts, "output_len": args.output_len, "seed": args.seed},
        "comparison": {k: v for k, v in cmp.items() if k != "per_prompt"},
        "ppl_bm4": ppl_bm4, "ppl_gate": PPL_GATE, "ppl_anchor": PPL_ANCHOR, "ppl_ok": ppl_ok,
        "attestation": {
            "bm4_applied": bm4_applied, "default_clean": default_clean,
            "splitkv_both": splitkv_both, "served_verify_is_3d": served_3d,
            "toggle_reverted_clean": toggle_clean,
            "bm4_forced_hits": arms["bm4"]["bm4_forced_hits"],
            "census_heads": arms["bm4"].get("census_heads"),
        },
        "verdict_byte_exact_and_ppl_pass": verdict_pass,
        "arms": arms,
        "elapsed_s": time.time() - t0,
    }
    (out_root / "results.json").write_text(json.dumps(result, indent=2, default=str))
    # full per-prompt detail beside the summary
    (out_root / "per_prompt.json").write_text(json.dumps(cmp["per_prompt"], indent=2))

    print("\n" + "=" * 78, flush=True)
    print("SERVED bm4 PAIRED GREEDY-IDENTITY CENSUS + PPL (PR #442, wirbel)", flush=True)
    print("=" * 78, flush=True)
    print(f"  token census (bm4 vs default, SAME M=8 verify path):", flush=True)
    print(f"    n_prompts={cmp['n_prompts']} identical={cmp['n_identical']} "
          f"divergent={cmp['n_divergent']} frac_identical={cmp['frac_identical']}", flush=True)
    print(f"    token-prefix match = {cmp['matched_token_prefix']}/{cmp['total_tokens']} "
          f"({cmp['frac_token_prefix_match']})", flush=True)
    print(f"    >>> BYTE-EXACT = {cmp['byte_exact']}", flush=True)
    if cmp["first_divergence"]:
        print(f"    first divergence: {cmp['first_divergence']}", flush=True)
    print(f"  PPL(bm4) = {ppl_bm4} (gate <={PPL_GATE}, anchor {PPL_ANCHOR}) -> ok={ppl_ok}", flush=True)
    print(f"  attestation: bm4_applied={bm4_applied} default_clean={default_clean} "
          f"splitkv_both={splitkv_both} served_3d={served_3d} forced_hits={arms['bm4']['bm4_forced_hits']}", flush=True)
    print(f"  toggle_reverted_clean={toggle_clean}", flush=True)
    print(f"  >>> CENSUS VERDICT (byte-exact AND ppl pass) = {verdict_pass}", flush=True)
    print("=" * 78 + "\n", flush=True)

    if not args.no_wandb:
        _log_wandb(args, result)
    print(f"[bm4-census] artifacts -> {out_root / 'results.json'}", flush=True)
    return 0 if verdict_pass else 1


def _log_wandb(args, result: dict[str, Any]) -> None:
    try:
        from scripts import wandb_logging
    except Exception as exc:  # noqa: BLE001
        _log(f"wandb import failed ({exc}); skipping")
        return
    run = wandb_logging.init_wandb_run(
        job_type="served-bm4-census", agent="wirbel",
        name=args.wandb_name, group=args.wandb_group,
        tags=["pr442", "greedy-census", "bm4", "ppl", SUBMISSION],
        config={"block_m": args.block_m, "num_stages": args.num_stages,
                "num_prompts": args.num_prompts, "output_len": args.output_len,
                "seed": args.seed, "ppl_gate": PPL_GATE},
    )
    if run is None:
        return
    try:
        c = result["comparison"]
        flat = {
            "census/n_prompts": c["n_prompts"], "census/n_identical": c["n_identical"],
            "census/n_divergent": c["n_divergent"],
            "census/frac_identical": c["frac_identical"] or 0.0,
            "census/frac_token_prefix_match": c["frac_token_prefix_match"] or 0.0,
            "census/byte_exact": float(bool(c["byte_exact"])),
            "ppl_bm4": result["ppl_bm4"] or 0.0,
            "ppl_ok": float(bool(result["ppl_ok"])) if result["ppl_ok"] is not None else 0.0,
            "verdict_pass": float(bool(result["verdict_byte_exact_and_ppl_pass"])),
        }
        for k, v in result["attestation"].items():
            flat[f"attest/{k}"] = float(v) if isinstance(v, (int, float, bool)) else 0.0
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(run, name="served_bm4_census",
                                        artifact_type="greedy-census", data=result)
    except Exception as exc:  # noqa: BLE001
        _log(f"WARN wandb logging error: {exc}")
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
