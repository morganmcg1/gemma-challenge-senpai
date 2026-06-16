"""PR #503 — analysis: operative-identity census + private-Δ + W&B logging.

Reads probe.py result JSONs and computes the headline outputs:

  * operative-identity census: per-prompt position-by-position comparison of each
    drafter config's completion token IDs vs the int4 M=1 AR floor (spec_off on
    the SAME verify stack). A lossless spec-dec method emits exactly the target's
    greedy tokens, so identity should be 1.0 / flip-rate 0. We MEASURE it (not
    assume), and census MTP the same way so "ngram is no worse than MTP on the
    strict axis" is a measured claim, not a hope (the M=8-vs-M=1 verify reduction
    order is the only thing that could flip an argmax, and it is drafter-shared).
  * acceptance + e_accept + steady/probe TPS per config; ngram-vs-AR-floor lift.
  * private-Δ: ngram acceptance on public vs #497 easy/hard splits, vs MTP 4.295%.

Local A10G probe — NOT official a10g-small TPS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def _ar_name(configs: dict[str, Any]) -> str | None:
    for name, rec in configs.items():
        if rec.get("spec_config", None) == "" or name.startswith("ar"):
            return name
    return None


def _per_prompt_map(rec: dict[str, Any]) -> dict[Any, list[int]]:
    """index -> completion_token_ids."""
    out = {}
    for p in rec.get("per_prompt", []) or []:
        out[p.get("index")] = p.get("completion_token_ids") or []
    return out


def census_vs_ar(rec: dict[str, Any], ar_rec: dict[str, Any]) -> dict[str, Any]:
    cand = _per_prompt_map(rec)
    ar = _per_prompt_map(ar_rec)
    shared = sorted(set(cand) & set(ar), key=lambda x: (x is None, x))
    n_prompts = len(shared)
    identical_prompts = 0
    tot_tokens = 0
    mismatch_tokens = 0
    first_div = []
    for idx in shared:
        a = ar[idx]
        c = cand[idx]
        n = min(len(a), len(c))
        tot_tokens += n
        diffs = [i for i in range(n) if a[i] != c[i]]
        mismatch_tokens += len(diffs)
        if not diffs and len(a) == len(c):
            identical_prompts += 1
        elif diffs:
            first_div.append({"index": idx, "first_divergence_pos": diffs[0],
                              "n_div": len(diffs), "len_ar": len(a), "len_cand": len(c)})
    return {
        "n_prompts_compared": n_prompts,
        "identical_prompts": identical_prompts,
        "operative_identity_prompt_level": (identical_prompts / n_prompts) if n_prompts else None,
        "tokens_compared": tot_tokens,
        "mismatch_tokens": mismatch_tokens,
        "operative_identity_token_level": (1.0 - mismatch_tokens / tot_tokens) if tot_tokens else None,
        "semantic_flip_rate_token_level": (mismatch_tokens / tot_tokens) if tot_tokens else None,
        "divergent_prompts": first_div[:20],
    }


def _tps(rec: dict[str, Any]) -> dict[str, Any]:
    acc = rec.get("acceptance") or {}
    probe = rec.get("tps_probe") or {}
    return {
        "acceptance_rate": acc.get("acceptance_rate"),
        "e_accept": acc.get("e_accept"),
        "steady_gen_tps": acc.get("steady_gen_tps"),
        "probe_decode_tps": probe.get("decode_tps_single_stream"),
        "decode_tps_walltime": rec.get("decode_tps_walltime"),
        "accepted_tokens": acc.get("accepted_tokens"),
        "draft_tokens": acc.get("draft_tokens"),
        "source": acc.get("source"),
    }


def summarize(public: dict[str, Any]) -> dict[str, Any]:
    configs = public.get("configs", {})
    ar = _ar_name(configs)
    ar_rec = configs.get(ar, {}) if ar else {}
    ar_tps = _tps(ar_rec) if ar_rec else {}
    rows = {}
    for name, rec in configs.items():
        if rec.get("error"):
            rows[name] = {"error": rec["error"][:200]}
            continue
        t = _tps(rec)
        row = dict(t)
        if ar_rec and name != ar:
            row["census_vs_ar"] = census_vs_ar(rec, ar_rec)
        # TPS lift over AR floor (prefer steady; fall back to probe).
        for key, lk in (("steady_gen_tps", "lift_steady"), ("probe_decode_tps", "lift_probe")):
            base = ar_tps.get(key)
            v = t.get(key)
            if base and v:
                row[lk] = v / base
        rows[name] = row
    return {"ar_floor_config": ar, "ar_tps": ar_tps, "rows": rows}


def private_delta(public: dict, easy: dict, hard: dict, config_name: str) -> dict[str, Any]:
    def acc(d):
        r = (d.get("configs", {}).get(config_name, {}) or {}).get("acceptance", {}) or {}
        return r.get("acceptance_rate"), r.get("e_accept")
    pa, pe = acc(public)
    ea, ee = acc(easy)
    ha, he = acc(hard)
    out = {
        "config": config_name,
        "public_acceptance_rate": pa, "public_e_accept": pe,
        "easy_acceptance_rate": ea, "easy_e_accept": ee,
        "hard_acceptance_rate": ha, "hard_e_accept": he,
    }
    if pa is not None and ea is not None:
        out["delta_public_minus_easy_pct"] = 100.0 * (pa - ea)
    if pa is not None and ha is not None:
        out["delta_public_minus_hard_pct"] = 100.0 * (pa - ha)
    deltas = [d for d in (out.get("delta_public_minus_easy_pct"),
                          out.get("delta_public_minus_hard_pct")) if d is not None]
    if deltas:
        out["max_abs_private_delta_pct"] = max(abs(d) for d in deltas)
        out["mtp_reference_delta_pct"] = 4.295
        out["private_breach_shrinks"] = out["max_abs_private_delta_pct"] < 4.295
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--public", required=True)
    ap.add_argument("--private-easy", default=None)
    ap.add_argument("--private-hard", default=None)
    ap.add_argument("--private-config", default=None,
                    help="config name to compute private-Δ for (e.g. the best ngram)")
    ap.add_argument("--mtp-config", default="mtp_k7")
    ap.add_argument("--out", default=None, help="write analysis JSON here")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="ngram-prompt-lookup-spec-dec")
    args = ap.parse_args(argv)

    public = _load(args.public)
    easy = _load(args.private_easy)
    hard = _load(args.private_hard)

    analysis: dict[str, Any] = {"summary": summarize(public)}
    if args.private_config and (easy or hard):
        analysis["private_delta"] = private_delta(public, easy, hard, args.private_config)
        if args.mtp_config:
            analysis["private_delta_mtp"] = private_delta(public, easy, hard, args.mtp_config)

    out_path = Path(args.out) if args.out else Path(args.public).with_name("analysis.json")
    out_path.write_text(json.dumps(analysis, indent=2))

    # Pretty print
    s = analysis["summary"]
    print(f"\n=== AR floor: {s['ar_floor_config']} "
          f"steady_tps={s['ar_tps'].get('steady_gen_tps')} probe_tps={s['ar_tps'].get('probe_decode_tps')} ===")
    print(f"{'config':16s} {'acc_rate':>9s} {'e_accept':>8s} {'steadyTPS':>9s} {'probeTPS':>9s} "
          f"{'lift_st':>7s} {'identTok':>9s} {'flipRate':>9s}")
    for name, r in s["rows"].items():
        if "error" in r:
            print(f"{name:16s}  ERROR {r['error'][:60]}")
            continue
        cen = r.get("census_vs_ar") or {}
        print(f"{name:16s} {_fmt(r.get('acceptance_rate')):>9s} {_fmt(r.get('e_accept')):>8s} "
              f"{_fmt(r.get('steady_gen_tps')):>9s} {_fmt(r.get('probe_decode_tps')):>9s} "
              f"{_fmt(r.get('lift_steady')):>7s} {_fmt(cen.get('operative_identity_token_level')):>9s} "
              f"{_fmt(cen.get('semantic_flip_rate_token_level')):>9s}")
    if "private_delta" in analysis:
        pd = analysis["private_delta"]
        print(f"\n=== private-Δ for {pd['config']} ===")
        print(f"  public acc={_fmt(pd.get('public_acceptance_rate'))} "
              f"easy acc={_fmt(pd.get('easy_acceptance_rate'))} hard acc={_fmt(pd.get('hard_acceptance_rate'))}")
        print(f"  Δ(pub-easy)={_fmt(pd.get('delta_public_minus_easy_pct'))}% "
              f"Δ(pub-hard)={_fmt(pd.get('delta_public_minus_hard_pct'))}% "
              f"max|Δ|={_fmt(pd.get('max_abs_private_delta_pct'))}% vs MTP 4.295% "
              f"-> breach_shrinks={pd.get('private_breach_shrinks')}")

    if args.wandb_name:
        _log_wandb(analysis, args.wandb_name, args.wandb_group)
    print(f"\n[analyze] -> {out_path}")
    return 0


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return str(x)


def _log_wandb(analysis: dict[str, Any], name: str, group: str) -> None:
    try:
        import os
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] skipped ({exc})")
        return
    try:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            name=name, group=group, job_type="analysis",
        )
        s = analysis["summary"]
        flat: dict[str, Any] = {"ar_floor_config": s["ar_floor_config"]}
        for k, v in (s.get("ar_tps") or {}).items():
            flat[f"ar/{k}"] = v
        tbl = wandb.Table(columns=[
            "config", "acceptance_rate", "e_accept", "steady_gen_tps", "probe_decode_tps",
            "lift_steady", "lift_probe", "operative_identity_token", "semantic_flip_rate",
            "identical_prompts", "n_prompts"])
        for cfgname, r in s["rows"].items():
            if "error" in r:
                continue
            cen = r.get("census_vs_ar") or {}
            tbl.add_data(
                cfgname, r.get("acceptance_rate"), r.get("e_accept"), r.get("steady_gen_tps"),
                r.get("probe_decode_tps"), r.get("lift_steady"), r.get("lift_probe"),
                cen.get("operative_identity_token_level"), cen.get("semantic_flip_rate_token_level"),
                cen.get("identical_prompts"), cen.get("n_prompts_compared"))
            for mk in ("acceptance_rate", "e_accept", "steady_gen_tps", "probe_decode_tps", "lift_steady"):
                if r.get(mk) is not None:
                    flat[f"cfg/{cfgname}/{mk}"] = r[mk]
            flat[f"cfg/{cfgname}/operative_identity_token"] = cen.get("operative_identity_token_level")
            flat[f"cfg/{cfgname}/semantic_flip_rate"] = cen.get("semantic_flip_rate_token_level")
        run.log({"per_config": tbl})
        if "private_delta" in analysis:
            for k, v in analysis["private_delta"].items():
                if isinstance(v, (int, float, bool)) or v is None:
                    flat[f"private/{k}"] = v
        if "private_delta_mtp" in analysis:
            for k, v in analysis["private_delta_mtp"].items():
                if isinstance(v, (int, float, bool)) or v is None:
                    flat[f"private_mtp/{k}"] = v
        run.summary.update(flat)
        print(f"[wandb] logged run {run.id}")
        run.finish()
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] log failed ({exc})")


if __name__ == "__main__":
    raise SystemExit(main())
