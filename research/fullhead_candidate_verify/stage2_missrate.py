#!/usr/bin/env python3
"""PR #549 Stage 2 — offline candidate miss-rate(K) over held-out decode positions.

Reads the Stage-1 hidden-state dump (post-final-norm decode hidden states captured at
the lm_head GEMM boundary, FULLHEAD_DUMP=1) and the full bf16 ``lm_head.weight`` and,
for each CHEAP candidate nominator, measures how often the served greedy token falls
OUTSIDE the candidate's top-K shortlist:

  gold(n)   = argmax_v ( H[n] @ W_full[v] )          # served greedy token
              (final-logit-softcap + scale are monotonic -> argmax is invariant, so
               the raw head GEMM argmax IS the served token). Computed in fp32 from the
               exact bf16-held weight; we ALSO union the bf16-output-rounded argmax so
               K_safe is robust to the server's bf16 logit rounding on near-ties.
  miss(n,K) = gold(n) NOT in top-K of ( H[n] @ W_cand[v] )

  miss_rate(scheme,K) = mean_n miss(n,K)
  K_safe(scheme)      = smallest K with miss_rate == 0 over ALL held-out positions
                        (greedy identity is a HARD gate — #319).

Candidate nominators (cheap heads that propose the shortlist; the FULL bf16 head then
re-scores only those K rows to recover the exact greedy token => byte-cheap verify):
  int4_g128  : group-wise (g=128) symmetric int4  (matches the w4a16-ct body scheme)
  int4_perrow: per-output-channel symmetric int4  (looser; 1 scale/row)
  fp8_e4m3   : per-output-channel e4m3 (float8)
  lowrank_rR : rank-R truncated SVD of W_full      (stretch; --lowrank-rank R, opt-in)

Each scheme's served head-read bytes are reported (for the downstream TPS projection):
realized head read = candidate_bytes + K_safe * HIDDEN * 2 (the bf16 verify gather).

Analysis-only. No server, no served-file change, NO HF fire.

Run (AFTER Stage 1 frees the GPU; needs the Pass-B dump to exist):
  CUDA_VISIBLE_DEVICES=0 /tmp/senpai-venvs/<hash>/bin/python \
      research/fullhead_candidate_verify/stage2_missrate.py \
      --wandb_name fern/stage2-missrate --wandb_group fullhead-candidate-verify
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE

MODEL_DIR = (
    "/senpai-run/home/student-fern/.cache/huggingface/hub/"
    "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/"
    "ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0"
)
DUMP_PATH = "/tmp/fullhead_hidden_fern.pt"
HIDDEN = 2560
VOCAB = 262144
HEAD_KEY = "lm_head.weight"

# K sweep: dense at the low end (where a cheap verify pays off), out to 1024.
KS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]

MIN_ROWS = 50_000  # PR #549 Stage-2 requirement: >= 50k held-out positions


# ------------------------------------------------------------------------- #
# weight load + candidate quantizers (all return a bf16 [VOCAB, HIDDEN] tensor)
# ------------------------------------------------------------------------- #
def load_full_head(model_dir: str, device: str) -> "Any":
    import torch
    from safetensors import safe_open

    path = Path(model_dir) / "model.safetensors"
    with safe_open(str(path), framework="pt", device="cpu") as f:
        W = f.get_tensor(HEAD_KEY)
    assert tuple(W.shape) == (VOCAB, HIDDEN), f"unexpected head shape {tuple(W.shape)}"
    return W.to(device=device, dtype=torch.bfloat16)


def quant_int4_groupwise(W: "Any", group: int) -> "Any":
    """Per-group symmetric int4 (signed, [-8,7]); dequantized back to bf16."""
    import torch

    V, Hd = W.shape
    assert Hd % group == 0, f"hidden {Hd} not divisible by group {group}"
    Wf = W.float().reshape(V, Hd // group, group)
    scale = Wf.abs().amax(dim=-1, keepdim=True) / 7.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.clamp(torch.round(Wf / scale), -8, 7)
    Wdq = (q * scale).reshape(V, Hd)
    return Wdq.to(torch.bfloat16)


def quant_int4_perrow(W: "Any") -> "Any":
    import torch

    Wf = W.float()
    scale = Wf.abs().amax(dim=-1, keepdim=True) / 7.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.clamp(torch.round(Wf / scale), -8, 7)
    return (q * scale).to(torch.bfloat16)


def quant_fp8_e4m3(W: "Any") -> "Any":
    """Per-output-channel e4m3 (max representable magnitude 448)."""
    import torch

    Wf = W.float()
    scale = Wf.abs().amax(dim=-1, keepdim=True) / 448.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = (Wf / scale).to(torch.float8_e4m3fn).float()
    return (q * scale).to(torch.bfloat16)


def lowrank_svd(W: "Any", rank: int) -> "Any":
    """Rank-R truncated SVD reconstruction (stretch candidate)."""
    import torch

    Wf = W.float()
    U, S, Vh = torch.linalg.svd(Wf, full_matrices=False)
    Wr = (U[:, :rank] * S[:rank]) @ Vh[:rank, :]
    return Wr.to(torch.bfloat16)


def scheme_bytes(name: str, rank: int | None) -> dict[str, Any]:
    """Served head-read bytes for the candidate nominator weight."""
    if name.startswith("int4"):
        # 4-bit weights + bf16 per-group scales. g128 => 2560/128 = 20 groups/row.
        groups = HIDDEN // 128 if "g128" in name else 1
        wbytes = VOCAB * HIDDEN * 0.5
        sbytes = VOCAB * groups * 2
        return {"weight_bytes": wbytes + sbytes, "kind": "int4"}
    if name.startswith("fp8"):
        return {"weight_bytes": VOCAB * HIDDEN * 1.0 + VOCAB * 2, "kind": "fp8"}
    if name.startswith("lowrank") and rank:
        # U[V,r] bf16 + S[r] + Vh[r,Hd] bf16
        return {"weight_bytes": (VOCAB * rank + rank + rank * HIDDEN) * 2, "kind": "lowrank"}
    return {"weight_bytes": float("nan"), "kind": "unknown"}


# ------------------------------------------------------------------------- #
# core miss-rate sweep
# ------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[stage2] device={dev} torch={torch.__version__}", flush=True)

    blob = torch.load(args.dump_path, map_location="cpu")
    H_all = blob["hidden"] if isinstance(blob, dict) else blob
    H_all = H_all.to(torch.bfloat16)
    n_rows = H_all.shape[0]
    if args.max_rows and n_rows > args.max_rows:
        H_all = H_all[: args.max_rows]
        n_rows = args.max_rows
    print(f"[stage2] hidden dump rows={n_rows} dim={H_all.shape[1]} "
          f"(>= {MIN_ROWS} required: {n_rows >= MIN_ROWS})", flush=True)

    Wf16 = load_full_head(args.model_dir, dev)
    Wf32 = Wf16.float()  # exact (the stored weight IS bf16); resident for gold
    print(f"[stage2] loaded {HEAD_KEY} {tuple(Wf16.shape)} bf16 -> {dev}", flush=True)

    # build candidate weights (bf16, resident)
    schemes: dict[str, Any] = {}
    t0 = time.time()
    for name in args.schemes:
        if name == "int4_g128":
            schemes[name] = quant_int4_groupwise(Wf16, 128)
        elif name == "int4_perrow":
            schemes[name] = quant_int4_perrow(Wf16)
        elif name == "fp8_e4m3":
            schemes[name] = quant_fp8_e4m3(Wf16)
        elif name.startswith("lowrank"):
            schemes[name] = lowrank_svd(Wf16, args.lowrank_rank)
        else:
            raise SystemExit(f"unknown scheme {name}")
        # mean abs dequant error (diagnostic)
        err = (schemes[name].float() - Wf32).abs().mean().item()
        print(f"[stage2] built {name} (mean|dequant_err|={err:.3e}) "
              f"[{time.time()-t0:.1f}s]", flush=True)

    maxK = max(KS)
    miss = {s: {K: 0 for K in KS} for s in schemes}
    miss_b = {s: {K: 0 for K in KS} for s in schemes}  # bf16-output-gold misses
    gold_tie = 0  # positions whose fp32 and bf16-output golds disagree (near-ties)
    N = 0
    B = args.chunk

    for i in range(0, n_rows, B):
        Hc = H_all[i : i + B].to(dev)
        Hc_f = Hc.float()
        Lf = Hc_f @ Wf32.t()                  # [b, V] fp32 gold logits
        gold = Lf.argmax(dim=1)               # served greedy (fp32-exact)
        gold_b = Lf.to(torch.bfloat16).argmax(dim=1)  # server bf16-output rounding
        gold_tie += int((gold != gold_b).sum().item())
        for s, Wq in schemes.items():
            Lc = (Hc @ Wq.t()).float()        # candidate logits
            topk = Lc.topk(maxK, dim=1).indices  # [b, maxK]
            eq = topk == gold[:, None]
            eq_b = topk == gold_b[:, None]
            for K in KS:
                hit = eq[:, :K].any(dim=1)
                hit_b = eq_b[:, :K].any(dim=1)
                miss[s][K] += int((~hit).sum().item())
                miss_b[s][K] += int((~hit_b).sum().item())
            del Lc, topk, eq, eq_b
        N += Hc.shape[0]
        del Hc, Hc_f, Lf
        if (i // B) % 10 == 0:
            torch.cuda.empty_cache()
            print(f"[stage2] processed {N}/{n_rows}", flush=True)

    # union miss (robust): candidate must contain BOTH fp32 and bf16-output gold
    results: dict[str, Any] = {}
    for s in schemes:
        rate = {K: miss[s][K] / N for K in KS}
        rate_b = {K: miss_b[s][K] / N for K in KS}
        rate_u = {K: max(miss[s][K], miss_b[s][K]) / N for K in KS}  # >= both (conservative)
        # union counts (a position misses if EITHER gold is absent) — recompute exact union
        # would need per-position tracking; max() is an upper bound on the union rate and a
        # lower bound on K_safe robustness; we report it as the conservative curve.
        ksafe = next((K for K in KS if miss[s][K] == 0), None)
        ksafe_b = next((K for K in KS if miss_b[s][K] == 0), None)
        ksafe_u = next((K for K in KS if miss[s][K] == 0 and miss_b[s][K] == 0), None)
        sb = scheme_bytes(s, args.lowrank_rank)
        verify_bytes = (ksafe_u or maxK) * HIDDEN * 2
        results[s] = {
            "miss_rate_by_K": rate,
            "miss_rate_by_K_bf16gold": rate_b,
            "miss_rate_by_K_conservative": rate_u,
            "K_safe": ksafe,
            "K_safe_bf16gold": ksafe_b,
            "K_safe_conservative": ksafe_u,
            "candidate_weight_bytes": sb["weight_bytes"],
            "candidate_kind": sb["kind"],
            "verify_gather_bytes_at_Ksafe": verify_bytes,
            "served_head_read_bytes": sb["weight_bytes"] + verify_bytes,
        }
        print(f"[stage2] {s}: K_safe(fp32)={ksafe} K_safe(bf16)={ksafe_b} "
              f"K_safe(cons)={ksafe_u} | miss@1={rate[1]:.4e} miss@8={rate[8]:.4e} "
              f"miss@64={rate[64]:.4e}", flush=True)

    report = {
        "analysis_only": True,
        "official_tps": 0,
        "pr": 549,
        "stage": 2,
        "dump_path": args.dump_path,
        "model_dir": args.model_dir,
        "n_positions": N,
        "n_rows_required": MIN_ROWS,
        "rows_ok": N >= MIN_ROWS,
        "hidden": HIDDEN,
        "vocab": VOCAB,
        "full_head_bytes": VOCAB * HIDDEN * 2,
        "Ks": KS,
        "gold_fp32_vs_bf16_disagreements": gold_tie,
        "gold_tie_frac": gold_tie / N if N else float("nan"),
        "schemes": results,
        "created_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
    }
    return report


def log_wandb(report: dict[str, Any], args: argparse.Namespace) -> str | None:
    try:
        import sys
        # Bind the REAL site-packages wandb in sys.modules BEFORE adding ROOT to the
        # path — ROOT holds a local ./wandb run-data dir that otherwise shadows the
        # package as an (init-less) PEP-420 namespace package.
        import wandb
        _ = wandb.init  # fail fast if a namespace stub shadowed the package
        if str(ROOT) not in sys.path:
            sys.path.append(str(ROOT))
        from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                           log_json_artifact, log_summary)
    except Exception as exc:  # pragma: no cover
        print(f"[stage2] wandb unavailable: {exc}", flush=True)
        return None
    run = init_wandb_run(
        job_type="systems-profile",
        agent="fern",
        name=args.wandb_name or "fern/fullhead-stage2",
        group=args.wandb_group or "fullhead-candidate-verify",
        tags=["fullhead", "candidate-verify", "stage2", "missrate", "analysis-only"],
        notes="PR #549 Stage 2: offline candidate miss-rate(K) + K_safe on the 262k head",
        config={"model_dir": args.model_dir, "dump_path": args.dump_path,
                "schemes": args.schemes, "Ks": KS, "lowrank_rank": args.lowrank_rank},
    )
    if run is None:
        return None
    summary: dict[str, Any] = {
        "n_positions": report["n_positions"],
        "rows_ok": report["rows_ok"],
        "gold_tie_frac": report["gold_tie_frac"],
        "analysis_only": True,
        "official_tps": 0,
    }
    for s, r in report["schemes"].items():
        summary[f"K_safe_{s}"] = r["K_safe_conservative"] if r["K_safe_conservative"] is not None else -1
        summary[f"K_safe_fp32_{s}"] = r["K_safe"] if r["K_safe"] is not None else -1
        summary[f"missrate_K1_{s}"] = r["miss_rate_by_K"][1]
        summary[f"missrate_K8_{s}"] = r["miss_rate_by_K"][8]
        summary[f"missrate_K64_{s}"] = r["miss_rate_by_K"][64]
        summary[f"served_head_read_bytes_{s}"] = r["served_head_read_bytes"]
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="fullhead-stage2-report", artifact_type="stage2-report", data=report)
    rid = getattr(run, "id", None)
    finish_wandb(run)
    return rid


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-path", default=DUMP_PATH)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--schemes", nargs="+",
                    default=["int4_g128", "int4_perrow", "fp8_e4m3"])
    ap.add_argument("--lowrank-rank", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = use all dumped rows")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out-file", default=None)
    args = ap.parse_args(argv)

    if not Path(args.dump_path).exists():
        print(f"[stage2] FATAL: dump not found at {args.dump_path} "
              f"(Stage-1 Pass B must run first / GO required)", flush=True)
        return 2

    report = run(args)

    # Write the report FIRST so a wandb hiccup never loses the computed result.
    out = Path(args.out_file) if args.out_file else OUT_ROOT / "stage2_report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(f"[stage2] report -> {out}", flush=True)
    if not args.no_wandb:
        try:
            report["wandb_run_id"] = log_wandb(report, args)
            out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
        except Exception as exc:
            print(f"[stage2] wandb logging failed (non-fatal): {exc!r}", flush=True)

    line = "=" * 12 + " PR #549 STAGE 2 — candidate miss-rate(K) / K_safe " + "=" * 12
    print("\n" + line, flush=True)
    print(f"  positions evaluated   = {report['n_positions']} (>= {MIN_ROWS}: {report['rows_ok']})", flush=True)
    print(f"  gold fp32/bf16 ties   = {report['gold_tie_frac']:.3e}", flush=True)
    for s, r in report["schemes"].items():
        full = report["full_head_bytes"]
        served = r["served_head_read_bytes"]
        frac = served / full if full else float("nan")
        print(f"  {s:12s} K_safe(cons)={str(r['K_safe_conservative']):>5} "
              f"head_read={served/1e9:.3f}GB ({frac:.2%} of full) "
              f"miss@1={r['miss_rate_by_K'][1]:.2e}", flush=True)
    print("=" * len(line) + "\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
