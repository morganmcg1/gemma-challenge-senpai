"""PR #609 — aggregate the screen/census/GSM8K artifacts, run the official #319
verifier, project official TPS via the AR-anchored tau, log to W&B, and print the
PR-comment markdown. Pure analysis over already-captured JSON — no serving.

Usage:
  python report_wandb.py [--wandb-name denken/ngram-promptlookup-optionb] \
                         [--wandb-group ngram-promptlookup-optionb-lane] [--no-wandb]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))
from scripts.local_validation import paths  # noqa: E402

BASE = ROOT / "research" / "validity" / "ngram_spec_dec_int4"
SCREEN = BASE / "_sweep" / "screen"
CENSUS = BASE / "_sweep" / "census"
GSM8K_DIR = BASE / "_sweep" / "gsm8k"
VERIFIER = paths.GREEDY_VERIFIER_DIR / "check_greedy_identity.py"

AR_OFFICIAL_TPS = 126.378
MTP_PROXY_TPS = 427.7
GSM8K_BAR = 0.807


def _load(p: Path) -> Any:
    return json.loads(p.read_text()) if p.exists() else None


def run_verifier(reference: Path, candidate: Path) -> dict[str, Any]:
    """Official greedy-identity verdict (JSON) between two decode_outputs jsonl."""
    if not reference.exists() or not candidate.exists():
        return {"error": f"missing {reference if not reference.exists() else candidate}"}
    out = subprocess.run(
        [sys.executable, str(VERIFIER), "--reference", str(reference),
         "--candidate", str(candidate), "--json"],
        capture_output=True, text=True,
    )
    try:
        rep = json.loads(out.stdout)
    except json.JSONDecodeError:
        return {"error": out.stderr.strip() or out.stdout.strip(), "exit": out.returncode}
    return {
        "verdict": rep["verdict"],
        "num_prompts_compared": rep["num_prompts_compared"],
        "num_identical": rep["num_identical"],
        "num_divergent": rep["num_divergent"],
        "total_tokens_compared": rep["total_tokens_compared"],
        "total_divergent_tokens": rep["total_divergent_tokens"],
        "exit": out.returncode,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    pareto = _load(SCREEN / "pareto.json")
    rep: dict[str, Any] = {"pareto": pareto}

    # --- identity census ---
    ar_ref = CENSUS / "decode_ar_ref.jsonl"
    ar_ref2 = CENSUS / "decode_ar_ref2.jsonl"
    # best-speed ngram census jsonl (single ng_* file in census dir)
    ng_files = sorted(CENSUS.glob("decode_ng_*.jsonl"))
    census: dict[str, Any] = {}
    if ar_ref.exists() and ar_ref2.exists():
        census["floor_ar_vs_ar2"] = run_verifier(ar_ref, ar_ref2)
    for ng in ng_files:
        census[f"{ng.stem}_vs_ar_ref"] = run_verifier(ar_ref, ng)
    rep["census"] = census

    # --- GSM8K (best config, sampled) ---
    gsm = {}
    for p in sorted(GSM8K_DIR.glob("ng609_*_sampled.json")) + \
             sorted(GSM8K_DIR.glob("ar609_sampled.json")):
        d = _load(p)
        if d:
            gsm[p.stem] = {"accuracy": d.get("accuracy"), "n": d.get("n_problems"),
                           "strict_rate": d.get("strict_rate"),
                           "extract_fail_rate": d.get("extract_fail_rate"),
                           "truncation_rate": d.get("truncation_rate"),
                           "sampling": d.get("sampling")}
    rep["gsm8k"] = gsm
    return rep


def render_markdown(rep: dict[str, Any]) -> str:
    L: list[str] = []
    p = rep.get("pareto") or {}
    tau = p.get("tau")
    s_ar = p.get("s_ar_local")
    L.append("## Speed Pareto (M=1 single-stream; local A10G proxy → official via "
             f"τ={tau:.4f} anchored on AR)\n" if tau else "## Speed Pareto\n")
    L.append(f"_Local AR steady_gen_tps = {s_ar:.2f} → τ = {AR_OFFICIAL_TPS}/{s_ar:.2f} "
             f"= {tau:.4f}; proj_official = τ·S_local._\n" if tau else "")
    L.append("| config | S_local | proj_official_TPS | vs AR 126.378 | E_accept | accept_rate |")
    L.append("|---|---|---|---|---|---|")
    best = None
    for row in (p.get("pareto") or []):
        s = row.get("steady_gen_tps") or 0
        proj = row.get("proj_official_tps") or 0
        vs = row.get("vs_ar_pct")
        ea = row.get("e_accept")
        ar_rt = row.get("draft_acceptance_rate")
        lbl = row["label"]
        L.append(f"| {lbl} | {s:.2f} | {proj:.2f} | "
                 f"{(f'{vs:+.1f}%' if vs is not None else '—')} | "
                 f"{(f'{ea:.3f}' if ea else '—')} | {(f'{ar_rt:.3f}' if ar_rt else '—')} |")
        if lbl != "ar" and (best is None or proj > best[1]):
            best = (lbl, proj, row)
    L.append(f"\n_MTP candidate reference: {MTP_PROXY_TPS} official-proxy TPS._")
    if best:
        L.append(f"\n**Best-speed ngram config: `{best[0]}` → proj {best[1]:.2f} TPS.**\n")

    # identity
    cen = rep.get("census") or {}
    L.append("\n## Identity census (#319, WARM free-run greedy, full 128×512, seed 1)\n")
    floor = cen.get("floor_ar_vs_ar2")
    if floor:
        L.append(f"- **Cross-start floor (AR vs AR', dev307 control):** {floor.get('verdict')} — "
                 f"{floor.get('num_divergent')}/{floor.get('num_prompts_compared')} prompts differ, "
                 f"{floor.get('total_divergent_tokens')} tokens.")
    for k, v in cen.items():
        if k == "floor_ar_vs_ar2":
            continue
        L.append(f"- **{k}:** {v.get('verdict')} — {v.get('num_divergent')}/"
                 f"{v.get('num_prompts_compared')} prompts differ, "
                 f"{v.get('total_divergent_tokens')} tokens.")

    # gsm8k
    g = rep.get("gsm8k") or {}
    L.append(f"\n## GSM8K quality (sampled T=1.0/top_p=0.95/top_k=64, min_tokens=8, bar {GSM8K_BAR})\n")
    for k, v in g.items():
        acc = v.get("accuracy")
        verdict = "PASS" if (acc is not None and acc >= GSM8K_BAR) else "FAIL"
        L.append(f"- **{k}:** acc={acc:.4f} (n={v.get('n')}) [{verdict} vs {GSM8K_BAR}] "
                 f"strict={v.get('strict_rate')} trunc={v.get('truncation_rate')}")
    return "\n".join(L)


def log_wandb(rep: dict[str, Any], name: str, group: str) -> str | None:
    try:
        import os
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] skipped ({exc})", flush=True)
        return None
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        name=name, group=group, job_type="analysis",
        config={"ar_official_tps": AR_OFFICIAL_TPS, "mtp_proxy_tps": MTP_PROXY_TPS,
                "gsm8k_bar": GSM8K_BAR, "analysis_only": True, "official_tps": 0},
    )
    p = rep.get("pareto") or {}
    flat: dict[str, Any] = {"summary/tau": p.get("tau"), "summary/s_ar_local": p.get("s_ar_local")}
    for row in (p.get("pareto") or []):
        lbl = row["label"]
        flat[f"summary/{lbl}/steady_gen_tps"] = row.get("steady_gen_tps")
        flat[f"summary/{lbl}/proj_official_tps"] = row.get("proj_official_tps")
        flat[f"summary/{lbl}/e_accept"] = row.get("e_accept")
        flat[f"summary/{lbl}/draft_acceptance_rate"] = row.get("draft_acceptance_rate")
    for k, v in (rep.get("census") or {}).items():
        flat[f"summary/census/{k}/verdict"] = v.get("verdict")
        flat[f"summary/census/{k}/num_divergent"] = v.get("num_divergent")
    for k, v in (rep.get("gsm8k") or {}).items():
        flat[f"summary/gsm8k/{k}/accuracy"] = v.get("accuracy")
    run.summary.update(flat)
    # Pareto table
    tbl = wandb.Table(columns=["config", "steady_gen_tps", "proj_official_tps",
                               "vs_ar_pct", "e_accept", "draft_acceptance_rate"])
    for row in (p.get("pareto") or []):
        tbl.add_data(row["label"], row.get("steady_gen_tps"), row.get("proj_official_tps"),
                     row.get("vs_ar_pct"), row.get("e_accept"), row.get("draft_acceptance_rate"))
    run.log({"speed_pareto": tbl})
    rid = run.id
    run.finish()
    print(f"[wandb] logged run {rid}", flush=True)
    return rid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb-name", default="denken/ngram-promptlookup-optionb")
    ap.add_argument("--wandb-group", default="ngram-promptlookup-optionb-lane")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    rep = build_report(args)
    (BASE / "_sweep" / "final_report.json").write_text(json.dumps(rep, indent=2))
    md = render_markdown(rep)
    (BASE / "_sweep" / "report.md").write_text(md)
    print(md, flush=True)
    if not args.no_wandb:
        rid = log_wandb(rep, args.wandb_name, args.wandb_group)
        print(f"\nW&B run id: {rid}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
