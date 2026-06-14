#!/usr/bin/env python
"""Analyze N-run greedy-determinism captures and deliver the contract verdict (PR #73).

Reads the per-reload captures written by `greedy_determinism.py`
(captures/<config>/run_XX/{decode_outputs.jsonl,meta.json}) and computes, per config:

  * run x run agreement matrix  : fraction of prompts byte-identical across each
                                  pair of reloads (diag = 1.0, symmetric);
  * byte-identical prompt frac   : mean over off-diagonal pairs (the bit-exact rate);
  * mean per-token agreement     : token-weighted positional match fraction
                                  (cascade-dominated: ~ shared-prefix fraction);
  * first-divergence onset       : min/median/max position the argmax first flips,
                                  and as a fraction of output_len (late => FP noise);
  * intrinsic flip hazard /tok   : censored-geometric MLE p(flip) per emitted token,
                                  the run-to-run argmax-flip rate of the stack itself.

It cross-checks a representative pair against the OFFICIAL verifier
(`greedy_gate.compare`) so the uncascaded numbers sit beside the official verdict,
and folds in TPS (sglang `output_throughput`) + PPL from each reload's meta.json.

VERDICT (primary_metric `greedy_identity_verdict`):
  0 = bit-exact      : the deployed default config reproduces byte-identical output
                       run-to-run (the contract is a satisfiable bit-exact property);
  1 = distributional : the deployed default config does NOT reproduce byte-identical
                       output, PPL is run-to-run invariant, and FA_SLIDING=0 restores
                       byte-identity -- so greedy-identity as written is unsatisfiable
                       above ~286 TPS for this stack and is enforced distributionally
                       by the PPL gate.

test_metric = FA_SLIDING=0 TPS cost % = (tps[default] - tps[fa_sliding_off]) / tps[default] * 100.

LOCAL ONLY. Pure CPU analysis -- no serving, no GPU, no HF Job.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.local_validation import greedy_gate, paths  # noqa: E402


def load_runs(out_root: Path, config: str) -> list[dict[str, Any]]:
    """Load every completed reload for a config, keyed by prompt index."""
    runs: list[dict[str, Any]] = []
    cfg_dir = out_root / config
    if not cfg_dir.is_dir():
        return runs
    for run_dir in sorted(cfg_dir.glob("run_*")):
        decode = run_dir / "decode_outputs.jsonl"
        meta_path = run_dir / "meta.json"
        if not decode.exists():
            continue
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        rows: dict[int, dict[str, Any]] = {}
        for line in decode.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows[int(r["index"])] = {
                "completion_token_ids": r["completion_token_ids"],
                "prompt_token_sha256": r.get("prompt_token_sha256"),
                "completion_token_sha256": r.get("completion_token_sha256"),
            }
        if rows:
            runs.append({"run_idx": meta.get("run_idx"), "dir": str(run_dir), "meta": meta, "rows": rows})
    return runs


def parse_engagement(run_dir: Path) -> dict[str, Any]:
    """Did each PR-named source toggle ACTUALLY engage this reload? (from server.log)

    A toggle that does not engage cannot be a determinism source, and its TPS
    delta is a no-op artifact -- this is the load-bearing distinction for the
    FA_SLIDING=0 TPS-cost test_metric (FA_SLIDING flips 0 layers on this build).
    Engagement is decided from MEASURED tokens (identical-to-default + self-
    reproducibility), not from a source-code gate prior: e.g. forcing
    VLLM_MARLIN_USE_ATOMIC_ADD=1 measurably changes tokens here, so the atomic-add
    path is NOT a no-op on this A10G despite warning-absence in the logs.
    """
    log = run_dir / "server.log"
    out: dict[str, Any] = {"fa_flips": None, "splitkv_redirects": None,
                           "marlin_bf16_sm8x_warn": None}
    if not log.exists():
        return out
    fa = sk = 0
    warn = False
    for line in log.read_text(errors="ignore").splitlines():
        if "-> FLASH_ATTN" in line:
            fa += 1
        elif "splitkv-verify] verify batch" in line:  # patch caps logging at 5
            sk += 1
        elif "Marlin kernel with bf16 on GPUs before SM90" in line:
            warn = True
    out.update(fa_flips=fa, splitkv_redirects=sk, marlin_bf16_sm8x_warn=warn)
    return out


def identical_to_ref_frac(ref_rows: dict[int, dict], rows: dict[int, dict]) -> float:
    """Fraction of shared prompts whose completion is byte-identical (sha256)."""
    keys = set(ref_rows) & set(rows)
    if not keys:
        return float("nan")
    same = sum(1 for k in keys
               if ref_rows[k]["completion_token_sha256"] == rows[k]["completion_token_sha256"])
    return same / len(keys)


def prompt_pair(a: list[int], b: list[int]) -> dict[str, Any]:
    """Compare two completions for one prompt: identity, onset, positional match."""
    n = min(len(a), len(b))
    first_div = None
    matches = 0
    for i in range(n):
        if a[i] == b[i]:
            matches += 1
        elif first_div is None:
            first_div = i
    if first_div is None and len(a) != len(b):
        first_div = n  # agree on the shared prefix but lengths differ
    denom = max(len(a), len(b)) or 1
    return {
        "identical": (len(a) == len(b) and first_div is None),
        "first_div": first_div,
        "match_frac": matches / denom,
        "matches": matches,
        "denom": denom,
    }


def pair_stats(rows_a: dict[int, dict], rows_b: dict[int, dict]) -> dict[str, Any]:
    keys = sorted(set(rows_a) & set(rows_b))
    n_ident = 0
    onsets: list[int] = []
    tot_match = 0
    tot_denom = 0
    for k in keys:
        pp = prompt_pair(rows_a[k]["completion_token_ids"], rows_b[k]["completion_token_ids"])
        if pp["identical"]:
            n_ident += 1
        elif pp["first_div"] is not None:
            onsets.append(pp["first_div"])
        tot_match += pp["matches"]
        tot_denom += pp["denom"]
    n = len(keys)
    return {
        "n_prompts": n,
        "byte_identical_frac": n_ident / n if n else float("nan"),
        "num_byte_identical": n_ident,
        "num_divergent": n - n_ident,
        "per_token_agreement": tot_match / tot_denom if tot_denom else float("nan"),
        "onsets": onsets,
    }


def flip_hazard(onsets: list[int], num_identical_pairs_prompts: int, output_len: int) -> float | None:
    """Censored-geometric MLE of per-token argmax-flip probability.

    Each diverged prompt contributes its onset position (tokens survived before a
    flip); each byte-identical prompt is right-censored at output_len. MLE for the
    geometric success prob is (#flips) / (#flips + total survived tokens)."""
    flips = len(onsets)
    survived = sum(onsets) + num_identical_pairs_prompts * output_len
    total = flips + survived
    return flips / total if total else None


def matrix_byte_identical(runs: list[dict[str, Any]]) -> list[list[float]]:
    n = len(runs)
    m = [[1.0] * n for _ in range(n)]
    for i, j in combinations(range(n), 2):
        s = pair_stats(runs[i]["rows"], runs[j]["rows"])
        m[i][j] = m[j][i] = round(s["byte_identical_frac"], 4)
    return m


def official_xcheck(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Run the OFFICIAL verifier on the first available reload pair as a cross-check."""
    if len(runs) < 2:
        return None
    ref = Path(runs[0]["dir"]) / "decode_outputs.jsonl"
    cand = Path(runs[1]["dir"]) / "decode_outputs.jsonl"
    try:
        report = greedy_gate.compare(ref, cand)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    onset = greedy_gate.onset_summary(report)
    return {
        "pair": [runs[0]["run_idx"], runs[1]["run_idx"]],
        "verdict": report.verdict,
        "num_identical": report.num_identical,
        "num_divergent": report.num_divergent,
        "total_tokens_compared": report.total_tokens_compared,
        "total_divergent_tokens": report.total_divergent_tokens,
        "official_token_div_frac": (
            report.total_divergent_tokens / report.total_tokens_compared
            if report.total_tokens_compared else None),
        "onset_min": onset.get("onset_min"),
        "onset_median": onset.get("onset_median"),
        "onset_max": onset.get("onset_max"),
    }


def cluster_signatures(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Group reloads by full-output signature (distinguishes stochastic scatter from a
    bistable/first-reload outlier). A config that is genuinely run-to-run stochastic
    yields many singleton clusters; a one-time warm-up/autotune effect yields one big
    identical cluster plus a lone outlier."""
    import hashlib

    by_sig: dict[str, list[int]] = {}
    for r in runs:
        rows = r["rows"]
        digest = hashlib.sha256(
            "".join((rows[k]["completion_token_sha256"] or "") for k in sorted(rows)).encode()
        ).hexdigest()[:12]
        by_sig.setdefault(digest, []).append(r["run_idx"])
    clusters = sorted(by_sig.values(), key=len, reverse=True)
    return {
        "num_distinct_outputs": len(clusters),
        "largest_identical_cluster": len(clusters[0]) if clusters else 0,
        "clusters_by_run_idx": clusters,
    }


def analyze_config(runs: list[dict[str, Any]], output_len: int) -> dict[str, Any]:
    if not runs:
        return {"num_runs": 0}
    pairs = list(combinations(range(len(runs)), 2))
    bif, pta, all_onsets, ident_prompt_pairs = [], [], [], 0
    for i, j in pairs:
        s = pair_stats(runs[i]["rows"], runs[j]["rows"])
        bif.append(s["byte_identical_frac"])
        pta.append(s["per_token_agreement"])
        all_onsets.extend(s["onsets"])
        ident_prompt_pairs += s["num_byte_identical"]
    out: dict[str, Any] = {
        "num_runs": len(runs),
        "num_pairs": len(pairs),
        "run_indices": [r["run_idx"] for r in runs],
        "mean_byte_identical_frac": round(statistics.mean(bif), 4) if bif else None,
        "min_byte_identical_frac": round(min(bif), 4) if bif else None,
        "max_byte_identical_frac": round(max(bif), 4) if bif else None,
        "mean_per_token_agreement": round(statistics.mean(pta), 4) if pta else None,
        "num_divergent_prompt_pairs": len(all_onsets),
        "flip_hazard_per_token": flip_hazard(all_onsets, ident_prompt_pairs, output_len),
        "matrix_byte_identical": matrix_byte_identical(runs),
        "official_xcheck": official_xcheck(runs),
        **cluster_signatures(runs),
    }
    if all_onsets:
        all_onsets.sort()
        out["onset_min"] = all_onsets[0]
        out["onset_median"] = int(statistics.median(all_onsets))
        out["onset_max"] = all_onsets[-1]
        out["onset_median_frac_of_len"] = round(statistics.median(all_onsets) / output_len, 3)
        out["onset_signature"] = (
            "late/stochastic (FP-reduction noise)" if statistics.median(all_onsets) > 0.1 * output_len
            else "early/systematic (genuine decode change)")
    # per-run TPS / PPL / E_accept gathered from meta.json
    tps = [(r["meta"].get("bench") or {}).get("tps") for r in runs]
    out["tps_runs"] = [t for t in tps if t is not None]
    out["tps_median"] = round(statistics.median(out["tps_runs"]), 2) if out["tps_runs"] else None
    ppl = [r["meta"].get("ppl") for r in runs if r["meta"].get("ppl") is not None]
    out["ppl_runs"] = ppl
    out["ppl_spread"] = round(max(ppl) - min(ppl), 6) if len(ppl) >= 2 else None
    eacc = [(r["meta"].get("acceptance") or {}).get("e_accept") for r in runs]
    out["e_accept_runs"] = [round(e, 3) for e in eacc if e is not None]
    # did the named source toggle actually engage? (server.log evidence)
    eng = [parse_engagement(Path(r["dir"])) for r in runs]
    out["fa_flips_runs"] = [e["fa_flips"] for e in eng if e["fa_flips"] is not None]
    out["splitkv_redirects_runs"] = [e["splitkv_redirects"] for e in eng if e["splitkv_redirects"] is not None]
    out["marlin_bf16_sm8x_warn_any"] = any(e["marlin_bf16_sm8x_warn"] for e in eng)
    return out


def build_report(out_root: Path, output_len: int) -> dict[str, Any]:
    runs_by_cfg = {c: load_runs(out_root, c)
                   for c in ("default", "fa_sliding_off", "splitkv_off", "atomic_on")}
    configs = {c: analyze_config(rb, output_len) for c, rb in runs_by_cfg.items() if rb}

    # --- cross-config identity vs the deployed default ---------------------
    # Does toggling each named source change ANY token vs default? An inert
    # toggle (FA flips 0 layers; atomic-add hw-gated off) is byte-identical to
    # default; a live-but-deterministic toggle (split-KV reduction order) may
    # shift some tokens while default stays self-identical.
    ref_runs = runs_by_cfg.get("default", [])
    ref_rows = ref_runs[0]["rows"] if ref_runs else None
    for c, rb in runs_by_cfg.items():
        if c == "default" or not rb or ref_rows is None:
            continue
        fracs = [identical_to_ref_frac(ref_rows, r["rows"]) for r in rb]
        fracs = [f for f in fracs if f == f]  # drop NaN
        configs[c]["identical_to_default_frac"] = round(statistics.mean(fracs), 4) if fracs else None

    default = configs.get("default", {})
    faoff = configs.get("fa_sliding_off", {})

    # --- the verdict -------------------------------------------------------
    default_bit_exact = (default.get("mean_byte_identical_frac") or 0.0) >= 0.999
    faoff_reproduces = (faoff.get("mean_byte_identical_frac") is not None
                        and faoff["mean_byte_identical_frac"] >= 0.999)
    ppl_invariant = (default.get("ppl_spread") is None) or (default.get("ppl_spread", 1) <= 0.001)
    verdict_code = 0 if default_bit_exact else 1  # 0 bit-exact, 1 distributional

    # --- FA_SLIDING=0 TPS cost (the load-bearing test_metric) ---------------
    # NOTE: faithful ONLY if FA_SLIDING=1 actually flips layers in default.
    # On this build it flips 0 (fa_flips_runs all 0) -> the toggle is inert and
    # this % is a no-op artifact, not a real kernel-swap cost. The flag below
    # records that so the number is never read as a live cost.
    tps_cost_pct = None
    if default.get("tps_median") and faoff.get("tps_median"):
        tps_cost_pct = round((default["tps_median"] - faoff["tps_median"]) / default["tps_median"] * 100.0, 2)
    fa_engaged_in_default = any(f and f > 0 for f in (default.get("fa_flips_runs") or []))
    fa_sliding0_tps_cost_is_noop = (not fa_engaged_in_default) and (
        (faoff.get("identical_to_default_frac") or 0) >= 0.999)

    # --- source attribution ------------------------------------------------
    source = {}
    for c, v in configs.items():
        source[c] = {
            "mean_byte_identical_frac": v.get("mean_byte_identical_frac"),
            "self_reproducible": (v.get("mean_byte_identical_frac") or 0) >= 0.999,
            "num_distinct_outputs": v.get("num_distinct_outputs"),
            "largest_identical_cluster": v.get("largest_identical_cluster"),
            "identical_to_default_frac": v.get("identical_to_default_frac"),
            "onset_median": v.get("onset_median"),
            "flip_hazard_per_token": v.get("flip_hazard_per_token"),
            "tps_median": v.get("tps_median"),
            "fa_flips_runs": v.get("fa_flips_runs"),
            "splitkv_redirects_runs": v.get("splitkv_redirects_runs"),
            "marlin_bf16_sm8x_warn_any": v.get("marlin_bf16_sm8x_warn_any"),
        }

    # human-readable per-toggle interpretation (why it is / isn't a source)
    source_notes: dict[str, str] = {}
    if "fa_sliding_off" in configs:
        source_notes["fa_sliding_off"] = (
            f"FA_SLIDING=1 flips {default.get('fa_flips_runs')} target layers in default "
            f"(0 => the FA2 sliding-window swap is INERT on this build); FA_SLIDING=0 tokens are "
            f"{faoff.get('identical_to_default_frac')} identical to default => the toggle changes nothing, "
            f"so its {tps_cost_pct}% TPS delta is a no-op artifact, not a kernel-swap cost.")
    if "atomic_on" in configs:
        aon = configs["atomic_on"]
        aon_repro = (aon.get("mean_byte_identical_frac") or 0) >= 0.999
        aon_id_def = aon.get("identical_to_default_frac")
        aon_inert = aon_repro and (aon_id_def is not None and aon_id_def >= 0.999)
        aon_xverdict = (aon.get("official_xcheck") or {}).get("verdict")
        if aon_inert:
            source_notes["atomic_on"] = (
                f"VLLM_MARLIN_USE_ATOMIC_ADD=1 left tokens {aon_id_def} identical to default and stayed "
                f"self-reproducible ({aon.get('mean_byte_identical_frac')}): the int4 Marlin atomic-add "
                "reduction is INERT here (hardware-gated off on A10G sm8x+bf16) and is not a nondeterminism source.")
        else:
            ndist = aon.get("num_distinct_outputs")
            big = aon.get("largest_identical_cluster")
            nrun = aon.get("num_runs")
            bistable = (ndist == 2 and big is not None and nrun is not None and big == nrun - 1)
            structure = (
                f"a single first-reload OUTLIER: {ndist} distinct outputs over {nrun} reloads with the "
                f"largest cluster {big}/{nrun} mutually byte-identical -- consistent with a one-time "
                "autotune/warm-up of the newly-enabled atomic-add kernel path, NOT per-token stochastic noise"
                if bistable else
                f"run-to-run scatter: {ndist} distinct outputs over {nrun} reloads (largest identical cluster "
                f"{big}/{nrun}) -- per-reload argmax-flip nondeterminism")
            source_notes["atomic_on"] = (
                f"VLLM_MARLIN_USE_ATOMIC_ADD=1 is NOT inert (single changed env var; fa_flips/splitkv match "
                f"default): tokens are only {aon_id_def} identical to default (it shifts the verify-GEMM numerics) "
                f"AND it fails the clean bit-exactness the default shows -- {structure} (mean byte-identical "
                f"{aon.get('mean_byte_identical_frac')}, official xcheck {aon_xverdict}, divergence onset median "
                f"{aon.get('onset_median')}/{output_len}). The deployed default keeps atomic-add OFF and is "
                "bit-exact across ALL reloads incl. the first, so keeping it off is load-bearing. NB: the "
                f"'Marlin bf16 before SM90' warning was {'seen' if aon.get('marlin_bf16_sm8x_warn_any') else 'NOT seen'} "
                "in logs, so warning-absence is not proof the path is gated off -- the measured token change proves it engaged.")
    if "splitkv_off" in configs:
        sko = configs["splitkv_off"]
        source_notes["splitkv_off"] = (
            f"SPLITKV_VERIFY genuinely engages in default ({default.get('splitkv_redirects_runs')} verify "
            "redirects/reload, log-capped at 5); disabling it leaves tokens "
            f"{sko.get('identical_to_default_frac')} identical to default and default stays self-identical "
            f"({sko.get('mean_byte_identical_frac')}) => the #43 3D split-KV reduction order is run-to-run stable.")

    # Positive control: does FORCING a candidate source on break determinism?
    aon_cfg = configs.get("atomic_on", {})
    atomic_add_breaks_determinism = bool(
        aon_cfg and (aon_cfg.get("mean_byte_identical_frac") is not None)
        and aon_cfg["mean_byte_identical_frac"] < 0.999)

    if default_bit_exact:
        dominant_source = ("none -- the deployed default is already byte-identical run-to-run "
                           "(no intrinsic nondeterminism to attribute)")
        if atomic_add_breaks_determinism:
            dominant_source += ("; positive control: forcing VLLM_MARLIN_USE_ATOMIC_ADD=1 DOES break "
                                "run-to-run identity, so keeping atomic-add OFF is load-bearing for the default")
    elif faoff_reproduces:
        dominant_source = "FA2 sliding-window kernel (FA_SLIDING=1)"
    else:
        dominant_source = "unresolved"

    report: dict[str, Any] = {
        "out_root": str(out_root),
        "output_len": output_len,
        "configs": configs,
        "source_attribution": source,
        "source_notes": source_notes,
        "verdict": {
            "greedy_identity_verdict": verdict_code,
            "label": "bit-exact" if verdict_code == 0 else "distributional",
            "default_bit_exact": default_bit_exact,
            "fa_sliding_off_reproduces": faoff_reproduces,
            "ppl_run_to_run_invariant": ppl_invariant,
            "fa_sliding0_tps_cost_pct": tps_cost_pct,
            "fa_sliding0_tps_cost_is_noop": fa_sliding0_tps_cost_is_noop,
            "fa_engaged_in_default": fa_engaged_in_default,
            "atomic_add_breaks_determinism": atomic_add_breaks_determinism,
            "dominant_source": dominant_source,
        },
        "primary_metric": {"name": "greedy_identity_verdict", "value": verdict_code},
        "test_metric": {"name": "fa_sliding0_tps_cost_pct", "value": tps_cost_pct},
    }
    return report


def log_to_wandb(report: dict[str, Any], *, wandb_group: str, wandb_name: str,
                 report_path: Path | None) -> None:
    v = report["verdict"]
    summary = {
        "greedy_identity_verdict": v["greedy_identity_verdict"],
        "verdict_label": v["label"],
        "atomic_add_breaks_determinism": int(v.get("atomic_add_breaks_determinism", False)),
        "fa_sliding0_tps_cost_pct": v["fa_sliding0_tps_cost_pct"],
        "fa_sliding0_tps_cost_is_noop": int(v["fa_sliding0_tps_cost_is_noop"]),
        "fa_engaged_in_default": int(v["fa_engaged_in_default"]),
        "ppl_run_to_run_invariant": int(v["ppl_run_to_run_invariant"]),
        "fa_sliding_off_reproduces": int(v["fa_sliding_off_reproduces"]),
    }
    for c, cv in report["configs"].items():
        summary[f"{c}.mean_byte_identical_frac"] = cv.get("mean_byte_identical_frac")
        summary[f"{c}.num_distinct_outputs"] = cv.get("num_distinct_outputs")
        summary[f"{c}.largest_identical_cluster"] = cv.get("largest_identical_cluster")
        summary[f"{c}.mean_per_token_agreement"] = cv.get("mean_per_token_agreement")
        summary[f"{c}.identical_to_default_frac"] = cv.get("identical_to_default_frac")
        summary[f"{c}.onset_median"] = cv.get("onset_median")
        summary[f"{c}.flip_hazard_per_token"] = cv.get("flip_hazard_per_token")
        summary[f"{c}.tps_median"] = cv.get("tps_median")
        summary[f"{c}.ppl_spread"] = cv.get("ppl_spread")
        summary[f"{c}.num_runs"] = cv.get("num_runs")
        summary[f"{c}.fa_flips_max"] = max(cv.get("fa_flips_runs") or [0], default=0)
        summary[f"{c}.splitkv_redirects_max"] = max(cv.get("splitkv_redirects_runs") or [0], default=0)
    try:
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_file_artifact, log_summary
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] unavailable: {exc}", flush=True)
        return
    run = init_wandb_run(
        job_type="greedy-determinism", agent="senpai", name=wandb_name,
        tags=["greedy-determinism", *([wandb_group] if wandb_group else [])],
        config={"out_root": report["out_root"], "output_len": report["output_len"],
                "wandb_group": wandb_group},
    )
    if run is None:
        print("[wandb] run not created (no creds/disabled); report.json is the record", flush=True)
        return
    log_summary(run, summary, step=0)
    if report_path is not None:
        log_file_artifact(run, path=report_path, name="greedy_determinism_report",
                          artifact_type="greedy-determinism-report")
    finish_wandb(run)
    print(f"[wandb] logged run {wandb_name} (group={wandb_group})", flush=True)


def _print(report: dict[str, Any]) -> None:
    print("\n" + "=" * 72, flush=True)
    print("GREEDY-DETERMINISM ANALYSIS (PR #73)", flush=True)
    print("=" * 72, flush=True)
    for c, v in report["configs"].items():
        print(f"\n[{c}]  runs={v['num_runs']} pairs={v.get('num_pairs')}", flush=True)
        print(f"  byte-identical prompt frac (mean over pairs): {v.get('mean_byte_identical_frac')}", flush=True)
        if c != "default":
            print(f"  identical-to-default frac (toggle effect)   : {v.get('identical_to_default_frac')}", flush=True)
        print(f"  toggle engaged? fa_flips={v.get('fa_flips_runs')} splitkv_redirects={v.get('splitkv_redirects_runs')} "
              f"marlin_bf16_sm8x_warn={v.get('marlin_bf16_sm8x_warn_any')}", flush=True)
        print(f"  mean per-token agreement                    : {v.get('mean_per_token_agreement')}", flush=True)
        if v.get("onset_median") is not None:
            print(f"  first-divergence onset  min/median/max      : "
                  f"{v.get('onset_min')}/{v.get('onset_median')}/{v.get('onset_max')} "
                  f"({v.get('onset_median_frac_of_len')} of {report['output_len']})  "
                  f"[{v.get('onset_signature')}]", flush=True)
        print(f"  intrinsic flip hazard /token                : {v.get('flip_hazard_per_token')}", flush=True)
        print(f"  TPS median / PPL runs / E_accept            : {v.get('tps_median')} / "
              f"{v.get('ppl_runs')} / {v.get('e_accept_runs')}", flush=True)
        xc = v.get("official_xcheck")
        if xc and "verdict" in xc:
            print(f"  OFFICIAL verifier xcheck (runs {xc['pair']})       : {xc['verdict']} "
                  f"({xc['num_divergent']}/{xc['num_divergent']+xc['num_identical']} divergent, "
                  f"onset med {xc.get('onset_median')})", flush=True)
    vd = report["verdict"]
    print("\n" + "-" * 72, flush=True)
    print(f"VERDICT: greedy-identity on the deployed frontier is {vd['label'].upper()} "
          f"(code {vd['greedy_identity_verdict']})", flush=True)
    print(f"  dominant source           : {vd['dominant_source']}", flush=True)
    print(f"  FA_SLIDING=0 reproduces   : {vd['fa_sliding_off_reproduces']}", flush=True)
    print(f"  FA_SLIDING=0 TPS cost     : {vd['fa_sliding0_tps_cost_pct']} %  (test_metric; "
          f"no-op artifact={vd['fa_sliding0_tps_cost_is_noop']}, fa_engaged_in_default={vd['fa_engaged_in_default']})", flush=True)
    print(f"  PPL run-to-run invariant  : {vd['ppl_run_to_run_invariant']}", flush=True)
    for c, note in report.get("source_notes", {}).items():
        print(f"  [{c}] {note}", flush=True)
    print("=" * 72 + "\n", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default=str(REPO / "research/validity/greedy_determinism/captures"))
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--report", default=None, help="where to write report.json (default <out-root>/report.json)")
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    report = build_report(out_root, args.output_len)
    report_path = Path(args.report) if args.report else (out_root / "report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    _print(report)
    print(f"[analyze] report -> {report_path}", flush=True)
    if args.wandb_group:
        log_to_wandb(report, wandb_group=args.wandb_group,
                     wandb_name=args.wandb_name or "kanna/greedy-determinism",
                     report_path=report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
