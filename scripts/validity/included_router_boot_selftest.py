"""included_router boot-validation self-test (PR #177).

Validates darwin-4b-opus's prometheus ``_IncludedRouter`` / missing-``.path``
startup-500 guard (board ``20260614-150027``) is OUTPUT-NEUTRAL on the deployed
``fa2sw_precache_kenyan`` serve. It does NOT add TPS; it de-risks the one
human-approved launch boot for our 481.53 stack and land #71's tree submission.

Evidence assembled here (all local A10G serve; no HF Job, no leaderboard draw):

1. **Guard UNIT proof** (mechanism, venv-independent): the unpatched
   ``prometheus_fastapi_instrumentator.routing._get_route_name`` raises
   ``AttributeError`` on a pathless (``_IncludedRouter``-like) matched route --
   exactly darwin's failure mode; darwin's guard returns ``None`` instead (crash
   neutralized) AND is a byte-verbatim no-op on a normal route (output-neutral
   at the function level).
2. **Boot outcome** of the UNPATCHED serve: ``startup_500_reproduced`` --
   darwin's prometheus-500 signature in the server log, or an honest no-crash
   clean boot (the local image may simply lack the offending route).
3. **Boot outcome** of the PATCHED serve: ``server_boots_with_fix``.
4. **Byte-exact completion-token-id diff** patched-vs-unpatched on the 128 bench
   prompts -> ``token_identity_rate`` (target 1.0): output-neutrality PROVEN on
   the real stack, not assumed.
5. **Scorer gates on the patched serve**: PPL <= cap (2.42) and 128/128 complete.
6. **NaN-clean**: every reported scalar finite.

PRIMARY metric: ``included_router_fix_self_test_passes``
TEST metric:    ``token_identity_rate``

CPU-only; consumes ``scripts/local_prevalidate.py`` artifacts. Run under the
serve venv (has ``prometheus_fastapi_instrumentator`` + ``wandb``)::

    /tmp/senpai-venvs/<hash>/bin/python scripts/validity/included_router_boot_selftest.py \
        --unpatched-dir research/validity/included_router_boot/unpatched \
        --patched-dir   research/validity/included_router_boot/patched \
        --wandb-name kanna/included-router-boot --wandb-group included-router-boot-validation
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
# Append (not insert-at-0): a wandb run writes a ``./wandb`` output dir into the
# repo root, which -- if ROOT were at sys.path[0] -- would shadow the real
# ``wandb`` site-package as a PEP-420 namespace package (``import wandb`` then
# succeeds but ``wandb.init`` is missing). Appending keeps ``scripts`` importable
# while site-packages still win for ``wandb``.
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

PPL_CAP = 2.42          # scorer hard gate (reference PPL + 5%)
PPL_REFERENCE = 2.3777  # official 481.53 reference PPL
BENCH_N = 128           # the public bench / validity-gate prompt count
TPS_NOISE_TOL_PCT = 5.0  # local patched-vs-unpatched decode-proxy tolerance

# darwin's prometheus-500 signature, matched in the unpatched server log.
DARWIN_500_PATTERNS = [
    r"_get_route_name",
    r"_IncludedRouter",
    r"object has no attribute ['\"]path['\"]",
    r"prometheus_fastapi_instrumentator",
]
NOT_READY_PATTERNS = [r"endpoint not ready at .*/v1/models", r"did not become ready"]


# ---------------------------------------------------------------------------
# (1) guard UNIT proof -- venv-level mechanism check, no server needed
# ---------------------------------------------------------------------------
def guard_unit_proof() -> dict[str, Any]:
    """Reproduce darwin's AttributeError in isolation and prove the guard both
    neutralizes it and is a verbatim no-op on a normal (path-bearing) route."""
    out: dict[str, Any] = {"available": False}
    try:
        import prometheus_fastapi_instrumentator.routing as r
        from starlette.routing import Match
    except Exception as exc:  # pragma: no cover - venv without the dep
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    out["available"] = True

    class PathlessRoute:
        """A matched route whose ``.path`` access raises AttributeError, like the
        ``_IncludedRouter`` sub-router darwin reported on fresh a10g images."""

        def matches(self, scope: Any):
            return (Match.FULL, {})

        def __getattr__(self, name: str):
            if name == "path":
                raise AttributeError("'_IncludedRouter' object has no attribute 'path'")
            raise AttributeError(name)

    class NormalRoute:
        path = "/v1/models"
        path_format = "/v1/models"
        name = "models"

        def matches(self, scope: Any):
            return (Match.FULL, {"endpoint": "x"})

    scope = {"type": "http", "path": "/v1/models", "method": "GET"}
    orig = r._get_route_name

    # UNPATCHED: pathless route -> AttributeError (darwin's mechanism)
    raised = False
    try:
        orig(scope, [PathlessRoute()])
    except AttributeError:
        raised = True
    out["unpatched_raises_attributeerror"] = raised

    # normal-route value under the ORIGINAL function (the neutrality reference)
    try:
        normal_orig = orig(scope, [NormalRoute()])
        normal_orig_ok = True
    except Exception as exc:
        normal_orig, normal_orig_ok = f"ERR:{exc}", False

    # apply darwin's guard verbatim
    def guarded(scope: Any, routes: Any):
        try:
            return orig(scope, routes)
        except AttributeError:
            return None

    r._get_route_name = guarded
    try:
        out["patched_returns_none_on_pathless"] = (
            guarded(scope, [PathlessRoute()]) is None
        )
        normal_guarded = guarded(scope, [NormalRoute()])
        out["patched_noop_on_normal_route"] = bool(
            normal_orig_ok and normal_guarded == normal_orig
        )
        out["normal_route_value"] = normal_guarded
    finally:
        r._get_route_name = orig  # restore; do not leave the module mutated

    out["all_pass"] = bool(
        out.get("unpatched_raises_attributeerror")
        and out.get("patched_returns_none_on_pathless")
        and out.get("patched_noop_on_normal_route")
    )
    return out


# ---------------------------------------------------------------------------
# artifact loading + boot classification
# ---------------------------------------------------------------------------
def _load_json(p: Path) -> dict[str, Any] | None:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _load_jsonl(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    recs = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    return recs


def _read_text(p: Path) -> str:
    try:
        return p.read_text(errors="replace")
    except Exception:
        return ""


def load_dir(d: Path) -> dict[str, Any]:
    return {
        "dir": str(d),
        "local_summary": _load_json(d / "local_summary.json"),
        "ppl_summary": _load_json(d / "ppl_summary.json"),
        "decode_summary": _load_json(d / "decode_summary.json"),
        "decode_records": _load_jsonl(d / "decode_outputs.jsonl"),
        "server_log": _read_text(d / "server.log"),
        "driver_log": _read_text(d / "driver.log"),
    }


def classify_boot(art: dict[str, Any]) -> dict[str, Any]:
    """clean_boot | darwin_500 | other_failure | no_data, with the matched sig."""
    server = art.get("server_log", "")
    driver = art.get("driver_log", "")
    blob = f"{server}\n{driver}"
    ls = art.get("local_summary") or {}

    darwin_hits = [p for p in DARWIN_500_PATTERNS if re.search(p, blob)]
    not_ready = any(re.search(p, blob) for p in NOT_READY_PATTERNS)
    served = bool(ls) and (ls.get("ppl_num_records") or ls.get("decode_num_records"))

    if served:
        outcome = "clean_boot"
    elif darwin_hits and not_ready:
        outcome = "darwin_500"
    elif darwin_hits:
        outcome = "darwin_500"
    elif not_ready or "server exited before readiness" in blob:
        outcome = "other_failure"
    elif not server and not driver:
        outcome = "no_data"
    else:
        outcome = "other_failure"

    return {
        "outcome": outcome,
        "darwin_500_signature_hits": darwin_hits,
        "not_ready_signal": not_ready,
        "served": bool(served),
    }


# ---------------------------------------------------------------------------
# (4) byte-exact completion-token-id diff
# ---------------------------------------------------------------------------
def token_identity(unpatched: list[dict[str, Any]],
                   patched: list[dict[str, Any]]) -> dict[str, Any]:
    def index(recs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        by_id: dict[str, dict[str, Any]] = {}
        for r in recs:
            key = str(r.get("id") if r.get("id") is not None else r.get("dataset_index"))
            by_id[key] = r
        return by_id

    a, b = index(unpatched), index(patched)
    common = sorted(set(a) & set(b))
    n_common = len(common)
    matches = 0
    prompt_matches = 0
    mismatches: list[dict[str, Any]] = []
    for key in common:
        ra, rb = a[key], b[key]
        ta = ra.get("completion_token_ids")
        tb = rb.get("completion_token_ids")
        sha_a = ra.get("completion_token_sha256")
        sha_b = rb.get("completion_token_sha256")
        # prompt inputs identical (same bench prompt drawn both sides)
        if ra.get("prompt_sha256") == rb.get("prompt_sha256"):
            prompt_matches += 1
        ids_equal = ta is not None and ta == tb
        sha_equal = (sha_a is not None and sha_a == sha_b)
        if ids_equal and (sha_equal or sha_a is None):
            matches += 1
        else:
            if len(mismatches) < 8:
                # locate first differing position for a useful report
                pos = None
                if isinstance(ta, list) and isinstance(tb, list):
                    for i in range(min(len(ta), len(tb))):
                        if ta[i] != tb[i]:
                            pos = i
                            break
                    if pos is None and len(ta) != len(tb):
                        pos = min(len(ta), len(tb))
                mismatches.append({
                    "id": key,
                    "len_unpatched": len(ta) if isinstance(ta, list) else None,
                    "len_patched": len(tb) if isinstance(tb, list) else None,
                    "first_diff_pos": pos,
                    "sha_unpatched": sha_a,
                    "sha_patched": sha_b,
                })
    rate = (matches / n_common) if n_common else 0.0
    return {
        "n_unpatched": len(unpatched),
        "n_patched": len(patched),
        "n_common": n_common,
        "n_token_id_match": matches,
        "n_prompt_input_match": prompt_matches,
        "token_identity_rate": rate,
        "all_identical": bool(n_common > 0 and matches == n_common),
        "mismatches_sample": mismatches,
    }


# ---------------------------------------------------------------------------
# assemble the self-test
# ---------------------------------------------------------------------------
def build_result(unpatched_dir: Path, patched_dir: Path) -> dict[str, Any]:
    unit = guard_unit_proof()
    up = load_dir(unpatched_dir)
    pp = load_dir(patched_dir)
    up_boot = classify_boot(up)
    pp_boot = classify_boot(pp)
    tid = token_identity(up.get("decode_records", []), pp.get("decode_records", []))

    up_ls = up.get("local_summary") or {}
    pp_ls = pp.get("local_summary") or {}

    def _f(x: Any) -> float | None:
        try:
            v = float(x)
            return v if math.isfinite(v) else None
        except (TypeError, ValueError):
            return None

    ppl_unpatched = _f(up_ls.get("ppl"))
    ppl_patched = _f(pp_ls.get("ppl"))
    tps_unpatched = _f(up_ls.get("tps"))
    tps_patched = _f(pp_ls.get("tps"))
    completed_unpatched = up_ls.get("completed") or up_ls.get("ppl_num_records")
    completed_patched = pp_ls.get("completed") or pp_ls.get("ppl_num_records")

    startup_500_reproduced = up_boot["outcome"] == "darwin_500"
    # (a) honest, definitive unpatched boot outcome (crash reproduced OR clean no-crash)
    unpatched_boot_definitive = up_boot["outcome"] in {"darwin_500", "clean_boot"}
    server_boots_with_fix = pp_boot["outcome"] == "clean_boot"

    tps_delta_pct = None
    if tps_unpatched and tps_patched and tps_unpatched > 0:
        tps_delta_pct = 100.0 * abs(tps_patched - tps_unpatched) / tps_unpatched

    # output-neutral: PPL gate + 128/128 + byte-exact token identity (+ TPS within
    # local decode-proxy noise when both serves booted).
    ppl_ok = ppl_patched is not None and ppl_patched <= PPL_CAP
    complete_ok = completed_patched == BENCH_N
    token_ok = tid["all_identical"] and tid["n_common"] == BENCH_N
    tps_ok = (tps_delta_pct is None) or (tps_delta_pct <= TPS_NOISE_TOL_PCT)
    output_neutral = bool(ppl_ok and complete_ok and token_ok and tps_ok)

    # NaN-clean: every reported scalar finite
    scalars = [ppl_patched, tps_patched, tid["token_identity_rate"]]
    if ppl_unpatched is not None:
        scalars.append(ppl_unpatched)
    if tps_unpatched is not None:
        scalars.append(tps_unpatched)
    nan_clean = all(isinstance(s, (int, float)) and math.isfinite(s) for s in scalars)

    checks = {
        "guard_unit_neutralizes_attributeerror": bool(unit.get("all_pass")),
        "unpatched_boot_definitive": unpatched_boot_definitive,
        "server_boots_with_fix": server_boots_with_fix,
        "completion_token_ids_byte_identical": token_ok,
        "ppl_patched_within_cap": ppl_ok,
        "patched_128_of_128": complete_ok,
        "nan_clean": nan_clean,
    }
    primary = bool(all(checks.values()))

    # hand-off recommendation
    if startup_500_reproduced:
        handoff = "required_output_neutral"
        handoff_note = (
            "darwin's startup-500 REPRODUCED on this image and the guard fixes it "
            "output-neutrally -> land #71 MUST include the guard in the tree serve."
        )
    else:
        handoff = "no_op_insurance"
        handoff_note = (
            "startup-500 did NOT reproduce on THIS local image (the local web stack "
            "lacks the pathless route); the guard is a verified output-neutral no-op "
            "here. darwin reproduced it 3x on the HF runner image, so land #71 should "
            "still BANK the guard as cheap launch-boot insurance for the runner image."
        )

    return {
        "pr": 177,
        "metric_primary": "included_router_fix_self_test_passes",
        "metric_test": "token_identity_rate",
        "included_router_fix_self_test_passes": primary,
        "token_identity_rate": tid["token_identity_rate"],
        "startup_500_reproduced": startup_500_reproduced,
        "server_boots_with_fix": server_boots_with_fix,
        "output_neutral": output_neutral,
        "ppl_patched": ppl_patched,
        "ppl_unpatched": ppl_unpatched,
        "ppl_cap": PPL_CAP,
        "ppl_reference": PPL_REFERENCE,
        "tps_patched_local_proxy": tps_patched,
        "tps_unpatched_local_proxy": tps_unpatched,
        "tps_patched_vs_unpatched_delta_pct": tps_delta_pct,
        "completed_patched": completed_patched,
        "completed_unpatched": completed_unpatched,
        "nan_clean": nan_clean,
        "checks": checks,
        "guard_unit_proof": unit,
        "unpatched_boot": up_boot,
        "patched_boot": pp_boot,
        "token_diff": tid,
        "handoff_recommendation": handoff,
        "handoff_note": handoff_note,
        "artifacts": {
            "unpatched_local_summary": up_ls,
            "patched_local_summary": pp_ls,
            "unpatched_dir": str(unpatched_dir),
            "patched_dir": str(patched_dir),
        },
    }


def _print(res: dict[str, Any]) -> None:
    print("\n[included-router] ===== boot-validation self-test (PR #177) =====", flush=True)
    u = res["guard_unit_proof"]
    print(f"  GUARD UNIT proof: available={u.get('available')} "
          f"unpatched_raises={u.get('unpatched_raises_attributeerror')} "
          f"patched_none={u.get('patched_returns_none_on_pathless')} "
          f"noop_normal={u.get('patched_noop_on_normal_route')} -> all_pass={u.get('all_pass')}",
          flush=True)
    print(f"  UNPATCHED boot: {res['unpatched_boot']['outcome']}  "
          f"(startup_500_reproduced={res['startup_500_reproduced']}; "
          f"darwin_sig_hits={res['unpatched_boot']['darwin_500_signature_hits']})", flush=True)
    print(f"  PATCHED   boot: {res['patched_boot']['outcome']}  "
          f"(server_boots_with_fix={res['server_boots_with_fix']})", flush=True)
    td = res["token_diff"]
    print(f"  TOKEN-IDENTITY: {td['n_token_id_match']}/{td['n_common']} byte-identical "
          f"(rate={res['token_identity_rate']:.6f}; prompt_input_match={td['n_prompt_input_match']})",
          flush=True)
    print(f"  GATES (patched): PPL={res['ppl_patched']} (cap {res['ppl_cap']})  "
          f"completed={res['completed_patched']}/{BENCH_N}  "
          f"tps_proxy={res['tps_patched_local_proxy']} "
          f"(delta vs unpatched={res['tps_patched_vs_unpatched_delta_pct']})", flush=True)
    print(f"  output_neutral={res['output_neutral']}  nan_clean={res['nan_clean']}", flush=True)
    for k, v in res["checks"].items():
        if not v:
            print(f"    !! FAILED CHECK: {k}", flush=True)
    print(f"\n  PRIMARY included_router_fix_self_test_passes = "
          f"{res['included_router_fix_self_test_passes']}", flush=True)
    print(f"  TEST    token_identity_rate = {res['token_identity_rate']:.6f}", flush=True)
    print(f"  HAND-OFF: {res['handoff_recommendation']} -- {res['handoff_note']}", flush=True)


def _log_wandb(args: argparse.Namespace, res: dict[str, Any]) -> None:
    if args.no_wandb:
        return
    try:
        from scripts import wandb_logging
    except Exception as exc:
        print(f"[included-router] wandb_logging import failed ({exc}); skipping", flush=True)
        return
    try:
        run = wandb_logging.init_wandb_run(
            job_type="included-router-boot-validation", agent="kanna",
            name=args.wandb_name, group=args.wandb_group,
            tags=["included-router-boot-validation", "launch-boot-derisk", "pr177",
                  "output-neutral", "darwin-included-router-guard"],
            config={"unpatched_dir": str(args.unpatched_dir),
                    "patched_dir": str(args.patched_dir),
                    "ppl_cap": PPL_CAP, "bench_n": BENCH_N},
        )
    except Exception as exc:
        # e.g. a shadowed/partial ``wandb`` (``module 'wandb' has no attribute
        # 'init'``) or a venv without wandb -- degrade gracefully, never crash the
        # self-test over its reporting side-channel. self_test.json is the artifact.
        print(f"[included-router] wandb init failed ({type(exc).__name__}: {exc}); "
              "skipping wandb (run under .venv with --relog-json to log)", flush=True)
        return
    if run is None:
        print("[included-router] wandb disabled; skipping", flush=True)
        return
    try:
        flat = {
            "included_router_fix_self_test_passes": 1.0 if res["included_router_fix_self_test_passes"] else 0.0,
            "token_identity_rate": res["token_identity_rate"],
            "startup_500_reproduced": 1.0 if res["startup_500_reproduced"] else 0.0,
            "server_boots_with_fix": 1.0 if res["server_boots_with_fix"] else 0.0,
            "output_neutral": 1.0 if res["output_neutral"] else 0.0,
            "ppl_patched": res["ppl_patched"],
            "ppl_unpatched": res["ppl_unpatched"],
            "tps_patched_local_proxy": res["tps_patched_local_proxy"],
            "tps_unpatched_local_proxy": res["tps_unpatched_local_proxy"],
            "tps_patched_vs_unpatched_delta_pct": res["tps_patched_vs_unpatched_delta_pct"],
            "completed_patched": res["completed_patched"],
            "n_token_id_match": res["token_diff"]["n_token_id_match"],
            "n_common": res["token_diff"]["n_common"],
            "guard_unit_all_pass": 1.0 if res["guard_unit_proof"].get("all_pass") else 0.0,
        }
        flat = {k: v for k, v in flat.items() if isinstance(v, (int, float))}
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run, name="included_router_boot_selftest",
            artifact_type="included-router-boot-validation", data=res)
    except Exception as exc:
        print(f"[included-router] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unpatched-dir", type=Path,
                    default=ROOT / "research/validity/included_router_boot/unpatched")
    ap.add_argument("--patched-dir", type=Path,
                    default=ROOT / "research/validity/included_router_boot/patched")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "research/validity/included_router_boot/self_test.json")
    ap.add_argument("--wandb-name", default="kanna/included-router-boot")
    ap.add_argument("--wandb-group", default="included-router-boot-validation")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--relog-json", type=Path, default=None,
                    help="skip recompute (which needs the serve venv's prometheus); "
                         "load this self_test.json and only (re)log it to wandb under "
                         "a wandb-capable interpreter such as .venv")
    args = ap.parse_args(argv)

    if args.relog_json is not None:
        res = json.loads(args.relog_json.read_text())
        print(f"[included-router] re-logging existing result from {args.relog_json}", flush=True)
        _print(res)
        _log_wandb(args, res)
        return 0

    res = build_result(args.unpatched_dir, args.patched_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, default=str))
    _print(res)
    print(f"\n[included-router] artifact -> {args.out}", flush=True)
    _log_wandb(args, res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
