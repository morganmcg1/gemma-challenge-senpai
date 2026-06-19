#!/usr/bin/env python
"""PR #718 -- int8 BAND-MINIMIZATION forward-fidelity screen. DISK-SAFE, ANALYSIS-ONLY.

Decision-forcing question (advisor): the LIVE int8 recovery lane is fern #659's
int8-locus (run nmjvtfov): int8 on decoder layers L14-L27 (14 layers) / int4-g128
elsewhere, greedy AIME 0.400 -> 0.450, at -16.3% speed (int8 on all 14 layers is the
single cost keeping the lane unfireable).  My #711 forward-fidelity instrument
(bf16-vs-bf16 self-check = exactly 0.0) localized the recoverable activation error to
a NARROW mid-stack band (~L18-20).  This probe SCREENS whether a band materially
narrower than fern's L14-27 recovers most of the full-band forward-fidelity -- i.e.
whether the int8 recovery can be bought cheaper than 14 layers.

  * This is a forward-fidelity SCREEN that RANKS candidate bands.  The fidelity->AIME
    hop stays ASSUMED-LINEAR and UNVERIFIED here (a narrower band recovering X% of the
    full-band forward-fidelity does NOT entail X% of the AIME recovery).  Output is
    handed to a downstream AIME measurement + speed measurement, not a recovery claim.

Apparatus = #711's verbatim (commit 7ac9a7f), in-memory fake-quant of ONE bf16 QAT
master (/workspace/gemma_build/qat_unq); the same 120 KCG teacher-forced base-bf16
greedy completions (30 AIME / 45 GPQA / 45 MMLU-Pro), completion positions only;
metric = relMSE of the FINAL residual-stream hidden pre-lm_head vs the bf16 reference.
The ONLY change vs #711 is what gets quantized to int8 vs int4 per config.

int8 RECIPE (fern #659, documented assumption; logged config nmjvtfov gives only
upgrade_layers=14-27 / upgrade_precision=int8 / skeleton=int4_g128_lmhead /
source=qat-unquantized):  fern's PR #659 states the int8 build is a "bit-width-only
clone of the int4 g128 recipe".  The int4-g128 recipe (submissions/int4_g128_lmhead/
build_quant.py, bit-exact-verified in #711's structure probe) is num_bits=4 / group /
group_size=128 / symmetric / minmax.  => int8 := SAME with num_bits=8 (int8-g128).
group_size/scheme are NOT in the logged W&B config, so this is the assumption; logged
explicitly to W&B + result.json.

Configs (all on the 343 official body modules; lm_head is downstream of the metric):
  g128_floor    : all body @ int4-g128                       (0% recovery floor, == #711 g128)
  g32_full      : all body @ int4-g32                        (int4 ceiling, for CONTEXT)
  int8_full     : int8 on L14-L27 / int4-g128 elsewhere      (100% recovery reference, == fern)
  width ladder  : int8 on centered bands (19,19)..(14,24)    (the frontier: width vs recovery)
  sliding w5    : int8 on (14,18)..(23,27)                   (center/asymmetry map)

recovery_fraction(band) = (err_g128 - err_band) / (err_g128 - err_int8_full)
  full band = 100%; g128 = 0% floor; g32_full reported alongside as the int4 ceiling.

Run (assigned GPU 0 -- inherited CVD=1 makes torch see 0 GPUs):
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 uv run python \
    research/validity/int8_band_minimization/fidelity_band_probe.py \
    --n-seqs 120 --max-comp 200 [--smoke] [--no-wandb]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.lifecycle.forward import quantize, dequantize
from compressed_tensors.quantization.utils.helpers import calculate_qparams

ROOT = Path("/workspace/senpai/target")
HERE = ROOT / "research/validity/int8_band_minimization"
KCG = ROOT / "research/validity/keepset_coverage_gap"
QAT_UNQ = "/workspace/gemma_build/qat_unq"
MODULE_LIST = ROOT / "submissions/int4_g128_lmhead/official_quantized_modules.json"
DEV = "cuda"
TASKS = ("aime2024", "gpqa_diamond", "mmlu_pro")

# fern #659 (run nmjvtfov) int8-locus -> AIME, the (unverified, assumed-linear) hop anchor.
# n=60 greedy gb6144 regime (fern's eval), the matched regime for THIS band.
AIME_INT4_G128 = 0.400          # int4-AR floor (PR #718 body / fern #659 advisor anchor)
AIME_INT8_FULLBAND = 0.450      # int8 L14-27 full band (fern #659 nmjvtfov, VERIFIED acc=0.45)
AIME_BF16 = 0.4667              # bf16 ceiling (fern #659, advisor anchor)
AIME_GATE = 0.420               # 90% bar
# context: denken/land n~1000 regime used different anchors (g128 0.347 / g32full 0.438);
# noted but NOT used for this band's reconciliation (regime mismatch).

# int8 recipe assumption (fern bit-width-only clone of int4-g128):
INT8_BITS, INT8_GS, INT8_SCHEME = 8, 128, "group/symmetric/minmax"


# ----------------------------- fake-quant -----------------------------------
def _qargs(num_bits: int, gs: int) -> QuantizationArgs:
    return QuantizationArgs(num_bits=num_bits, type="int", strategy="group",
                            group_size=gs, symmetric=True, observer="minmax")


def fake_quant(W: torch.Tensor, num_bits: int, gs: int) -> torch.Tensor:
    """dequant(quant(W)) -- matches build_quant.py minmax recipe; num_bits=4 (int4)
    reproduces the shipped int4_g128 codes bit-exactly (#711); num_bits=8 (int8) is
    fern #659's bit-width-only clone."""
    Wf = W.to(torch.float32)
    out_dim, in_dim = Wf.shape
    qa = _qargs(num_bits, gs)
    wg = Wf.reshape(out_dim, in_dim // gs, gs)
    scale, zp = calculate_qparams(wg.amin(-1), wg.amax(-1), qa)
    q = quantize(Wf, scale, zp, qa)
    return dequantize(q, scale, zp, qa).to(W.dtype)


def mtype(name: str) -> str:
    for t in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
              "down_proj", "per_layer_input_gate", "per_layer_projection",
              "per_layer_model_projection"):
        if name.endswith(t):
            return t
    return "other"


def layer_of(name: str) -> int:
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def short(name: str) -> str:
    return name.split("language_model.")[-1]


def band_modules(lo: int, hi: int, modules) -> list:
    return [m for m in modules if lo <= layer_of(m) <= hi]


# ----------------------------- inputs (verbatim #711) -----------------------
def build_inputs(tokenizer, n_seqs, max_comp, smoke):
    prompts = {}
    with open(KCG / "prompts.jsonl") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                prompts[(r["task"], str(r["id"]))] = r["prompt_text"]
    decode, order = {}, []
    with open(KCG / "decode.jsonl") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                k = (d["task"], str(d["id"]))
                decode[k] = d
                order.append(k)
    by_task = {t: [k for k in order if k[0] == t] for t in TASKS}
    if smoke:
        quota = {"aime2024": 3, "gpqa_diamond": 2, "mmlu_pro": 2}
    else:
        n_aime = min(len(by_task["aime2024"]), 30)
        rest = max(n_seqs - n_aime, 0)
        quota = {"aime2024": n_aime, "gpqa_diamond": rest // 2,
                 "mmlu_pro": rest - rest // 2}
    sel = []
    for t in TASKS:
        sel += by_task[t][:quota[t]]
    items, mm = [], 0
    for key in sel:
        d = decode[key]
        msgs = [{"role": "user", "content": prompts[key]}]
        enc = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=True)
        pids = list(enc["input_ids"])
        if len(pids) != d["prompt_len"]:
            mm += 1
        comp = list(d["completion_token_ids"])[:max_comp]
        if len(comp) < 1:
            continue
        items.append({"task": key[0], "id": key[1], "ids": pids + comp,
                      "prompt_len": len(pids), "comp_len": len(comp)})
    mix = {t: sum(1 for x in items if x['task'] == t) for t in TASKS}
    print(f"[inputs] {len(items)} seqs ({mix}); {mm} prompt_len mismatches; "
          f"avg_len={sum(len(x['ids']) for x in items)/len(items):.0f} "
          f"avg_comp={sum(x['comp_len'] for x in items)/len(items):.0f}", flush=True)
    return items


# ----------------------------- forward / error (verbatim #711) --------------
@torch.no_grad()
def forward_hidden(model, ids):
    t = torch.tensor([ids], device=DEV)
    out = model(input_ids=t, output_hidden_states=True, use_cache=False)
    return out.hidden_states


class ErrorAccumulator:
    def __init__(self, n_layers_p1, masks):
        self.nl = n_layers_p1
        self.masks = masks
        self.seq_relmse_final = []
        self.seq_cosdist_final = []
        self.seq_relmse_masked_final = []
        self.layer_relmse = None
        self.layer_cosdist = None
        self.layer_cnt = 0

    def add_seq(self, var_hs, ref_hs_final, ref_hs_all, pl, want_layers):
        vf = var_hs[-1][0, pl:].float()
        rf = ref_hs_final.float()
        relmse = (((vf - rf) ** 2).sum(-1) / (rf ** 2).sum(-1).clamp_min(1e-12)).mean().item()
        mask = self.masks[-1]
        vm, rm = vf * mask, rf * mask
        cosm = torch.nn.functional.cosine_similarity(vm, rm, dim=-1)
        relmse_m = (((vm - rm) ** 2).sum(-1) / (rm ** 2).sum(-1).clamp_min(1e-12)).mean().item()
        self.seq_relmse_final.append(relmse)
        self.seq_cosdist_final.append(float((1 - cosm).mean().item()))
        self.seq_relmse_masked_final.append(relmse_m)
        if want_layers and ref_hs_all is not None:
            if self.layer_relmse is None:
                self.layer_relmse = [0.0] * self.nl
                self.layer_cosdist = [0.0] * self.nl
            for l in range(self.nl):
                v = var_hs[l][0, pl:].float()
                r = ref_hs_all[l].float()
                rl = (((v - r) ** 2).sum(-1) / (r ** 2).sum(-1).clamp_min(1e-12)).mean().item()
                m = self.masks[l]
                cm = torch.nn.functional.cosine_similarity(v * m, r * m, dim=-1)
                self.layer_relmse[l] += rl
                self.layer_cosdist[l] += float((1 - cm).mean().item())
            self.layer_cnt += 1

    def summary(self):
        import numpy as np
        a = np.asarray(self.seq_relmse_final)
        am = np.asarray(self.seq_relmse_masked_final)
        c = np.asarray(self.seq_cosdist_final)
        out = {
            "relmse_final": float(a.mean()),
            "relmse_final_sem": float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0,
            "relmse_masked_final": float(am.mean()),
            "cosdist_masked_final": float(c.mean()),
            "n_seqs": len(a),
        }
        if self.layer_relmse is not None and self.layer_cnt:
            out["layer_relmse"] = [x / self.layer_cnt for x in self.layer_relmse]
            out["layer_cosdist"] = [x / self.layer_cnt for x in self.layer_cosdist]
        return out

    def per_seq(self):
        import numpy as np
        return np.asarray(self.seq_relmse_final)


# ----------------------------- config swapper -------------------------------
class Swapper:
    """Holds CPU bf16 fake-quant weights for all 343 body modules at three levels:
    g128 (int4-g128), g32 (int4-g32), int8 (int8-g128).  Swaps GPU weights in place
    to a target per-module level-map, with minimal swaps from the current state."""

    BF16 = "BF16"

    def __init__(self, model, name2mod, modules, need_g32=True):
        self.n2m = name2mod
        self.modules = modules
        self.cpu = {"g128": {}, "g32": {}, "int8": {}}
        for m in modules:
            W = self.n2m[m].weight.detach()
            self.cpu["g128"][m] = fake_quant(W, 4, 128).to("cpu")
            self.cpu["int8"][m] = fake_quant(W, INT8_BITS, INT8_GS).to("cpu")
            if need_g32:
                self.cpu["g32"][m] = fake_quant(W, 4, 32).to("cpu")
        self.level = {m: self.BF16 for m in modules}    # pristine bf16

    def apply(self, level_map: dict):
        """level_map: module -> 'g128'|'g32'|'int8'. Swap only changed modules."""
        for m in self.modules:
            tgt = level_map[m]
            if self.level[m] != tgt:
                self.n2m[m].weight.data.copy_(self.cpu[tgt][m].to(DEV))
                self.level[m] = tgt

    def map_all(self, level: str) -> dict:
        return {m: level for m in self.modules}

    def map_band_int8(self, band_set: set) -> dict:
        return {m: ("int8" if m in band_set else "g128") for m in self.modules}


# ----------------------------- W&B -------------------------------------------
def log_wandb(result, out_path, resume_id=None):
    import wandb
    init_kw = dict(
        entity="wandb-applied-ai-team", project="gemma-challenge-senpai",
        name="stark/int8-band-minimization",
        group="int8-band-minimization-stark",
        config={"analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
                "pr": 718, "reference": result["reference"], "metric": result["metric"],
                "n_seqs": result["n_seqs"], "max_comp": result["max_comp"],
                "task_mix": result["task_mix"],
                "int8_recipe": result["int8_recipe"],
                "fern_run": "nmjvtfov", "fern_band": "L14-27",
                "candidate_source": "fern659_int8_locus"},
    )
    if resume_id:
        init_kw.update(id=resume_id, resume="allow")
    run = wandb.init(**init_kw)
    knee = result["knee"]
    knee_saved = result["full_band"]["n_layers"] - knee["n_layers"]
    wandb.summary.update({
        "verdict": result["verdict"], "test_metric": result["int8_band_narrowable"],
        "int8_band_narrowable": result["int8_band_narrowable"],
        "primary_metric": result["min_int8_band_fidelity_recovery_fraction"],
        "min_int8_band_fidelity_recovery_fraction": result["min_int8_band_fidelity_recovery_fraction"],
        "bf16_selfcheck_relmse": result["bf16_selfcheck_relmse"],
        "bf16_selfcheck_ok": int(result["bf16_selfcheck_ok"]),
        "g128_self_det": int(result["g128_self_det"]),
        "err_g128_floor": result["err_g128_floor"],
        "err_g32_full": result["err_g32_full"],
        "err_int8_full": result["err_int8_full"],
        "int8_full_recovery_vs_g128_to_int8": 1.0,
        "g32_full_recovery_fraction_of_int8band": result["g32_full_recovery_fraction"],
        "knee_layers": knee["n_layers"], "knee_band": knee["band"],
        "knee_recovery_fraction": knee["recovery_fraction"],
        "knee_threshold": knee["threshold"],
        "knee_layers_saved": knee_saved,
        "knee90_layers": result["knee_90"]["n_layers"],
        "knee90_band": result["knee_90"]["band"],
        "knee90_recovery_fraction": result["knee_90"]["recovery_fraction"],
        "knee_L19ladder_layers": result["knee_L19_ladder"]["n_layers"],
        "knee_L19ladder_band": result["knee_L19_ladder"]["band"],
        "knee_L19ladder_recovery_fraction": result["knee_L19_ladder"]["recovery_fraction"],
        "best_w5_center_band": result["best_w5_center"]["band"],
        "best_w5_center_recovery": result["best_w5_center"]["recovery_fraction"],
        "asymmetry_skew": result["asymmetry"]["skew"],
        "aime_int4_g128": AIME_INT4_G128, "aime_int8_fullband": AIME_INT8_FULLBAND,
        "peak_gb": result["peak_gb"], "elapsed_s": result["elapsed_s"],
    })
    # frontier table (width ladder): layer count vs recovery fraction
    tl = wandb.Table(columns=["band", "lo", "hi", "n_int8_layers", "n_int8_modules",
                              "relmse_final", "recovery_fraction", "rec_ci_lo", "rec_ci_hi",
                              "rec_pct_of_full"])
    for r in result["ladder"]:
        tl.add_data(r["band"], r["lo"], r["hi"], r["n_layers"], r["n_modules"],
                    r["relmse_final"], r["recovery_fraction"], r["rec_ci_lo"], r["rec_ci_hi"],
                    round(100 * r["recovery_fraction"], 2))
    wandb.log({"frontier_width_ladder": tl})
    for r in result["ladder"]:
        wandb.log({"ladder_n_layers": r["n_layers"],
                   "ladder_recovery_fraction": r["recovery_fraction"],
                   "ladder_relmse_final": r["relmse_final"]})
    # PRIMARY frontier: position-optimized best-of-width (the knee is read off THIS)
    to = wandb.Table(columns=["n_int8_layers", "band", "lo", "hi", "n_int8_modules",
                              "relmse_final", "recovery_fraction", "rec_ci_lo", "rec_ci_hi",
                              "rec_pct_of_full"])
    for r in result["opt_frontier"]:
        to.add_data(r["n_layers"], r["band"], r["lo"], r["hi"], r["n_modules"],
                    r["relmse_final"], r["recovery_fraction"], r["rec_ci_lo"], r["rec_ci_hi"],
                    round(100 * r["recovery_fraction"], 2))
    wandb.log({"frontier_opt_best_of_width": to})
    for r in result["opt_frontier"]:
        wandb.log({"opt_n_layers": r["n_layers"],
                   "opt_recovery_fraction": r["recovery_fraction"],
                   "opt_relmse_final": r["relmse_final"]})
    # data-driven upper-centered (L21) ladder
    tu = wandb.Table(columns=["band", "lo", "hi", "n_int8_layers", "n_int8_modules",
                              "relmse_final", "recovery_fraction", "rec_ci_lo", "rec_ci_hi",
                              "rec_pct_of_full"])
    for r in result["ladder_up"]:
        tu.add_data(r["band"], r["lo"], r["hi"], r["n_layers"], r["n_modules"],
                    r["relmse_final"], r["recovery_fraction"], r["rec_ci_lo"], r["rec_ci_hi"],
                    round(100 * r["recovery_fraction"], 2))
    wandb.log({"ladder_up_L21": tu})
    # sliding width-5 center map
    ts = wandb.Table(columns=["band", "lo", "hi", "center", "relmse_final",
                              "recovery_fraction", "rec_ci_lo", "rec_ci_hi"])
    for r in result["sliding_w5"]:
        ts.add_data(r["band"], r["lo"], r["hi"], (r["lo"] + r["hi"]) / 2.0,
                    r["relmse_final"], r["recovery_fraction"], r["rec_ci_lo"], r["rec_ci_hi"])
    wandb.log({"center_map_w5": ts})
    # per-layer relmse profiles (g128 / g32 / int8_full / peak band)
    if result.get("per_layer"):
        pl = result["per_layer"]
        n = len(pl["g128"])
        tp = wandb.Table(columns=["layer_idx", "g128", "g32_full", "int8_full",
                                  "peak_band", "recoverable_g128_minus_g32"])
        for i in range(n):
            tp.add_data(i, pl["g128"][i], pl["g32_full"][i], pl["int8_full"][i],
                        pl["peak_band"][i] if pl.get("peak_band") else None,
                        pl["g128"][i] - pl["g32_full"][i])
        wandb.log({"per_layer_relmse": tp})
    art = wandb.Artifact("int8_band_minimization", type="analysis")
    art.add_file(out_path)
    run.log_artifact(art)
    run.finish()
    return run.id


# ----------------------------- main -----------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seqs", type=int, default=120)
    ap.add_argument("--max-comp", type=int, default=200)
    ap.add_argument("--out", default=str(HERE / "result.json"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--relog-from", default=None)
    ap.add_argument("--wandb-resume-id", default=None)
    args = ap.parse_args()

    if args.relog_from:
        res = json.load(open(args.relog_from))
        rid = log_wandb(res, args.relog_from, resume_id=args.wandb_resume_id)
        print(f"[relog] wandb run id: {rid}", flush=True)
        return 0

    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    t0 = time.time()

    modules = sorted(json.load(open(MODULE_LIST)))
    print(f"[main] {len(modules)} body modules", flush=True)

    # ---- band ladder + sliding window (anchored on #711 mid-stack L18-20 peak) ----
    FULL = (14, 27)
    if args.smoke:
        LADDER = [(19, 19), (18, 20), FULL]
        LADDER_UP = [(21, 21), (20, 22), FULL]
        SLIDING = [(14, 18), (18, 22), (23, 27)]
    else:
        # (A) a-priori centered-growth width ladder (center ~L19, the #711 hidden-stream
        #     peak), last rung == fern full band.
        LADDER = [(19, 19), (18, 20), (17, 21), (16, 22), (15, 23), (14, 24), FULL]
        # (B) data-driven upper-centered ladder (center L21 == the empirical best width-5
        #     center from this probe's own sliding map; the recoverable mass is upper-skewed
        #     vs the #711 hidden-stream peak). Finding the TRUE minimal band per the card's
        #     asymmetry warning. Last rung == fern full band.
        LADDER_UP = [(21, 21), (20, 22), (19, 23), (18, 24), (17, 25), (16, 26), (15, 27), FULL]
        # fixed width-5 sliding window across fern's band -> center/asymmetry map
        SLIDING = [(lo, lo + 4) for lo in range(14, 24)]   # (14,18)..(23,27)
    # union of all bands actually measured (dedup, preserve a stable order)
    all_bands = []
    for b in LADDER + LADDER_UP + SLIDING:
        if b not in all_bands:
            all_bands.append(b)
    peak_band = (18, 20)
    want_layer_bands = {FULL, peak_band}
    print(f"[bands] ladder_L19={LADDER}", flush=True)
    print(f"[bands] ladder_L21(data-driven)={LADDER_UP}", flush=True)
    print(f"[bands] sliding_w5={SLIDING}", flush=True)
    print(f"[bands] {len(all_bands)} unique bands; int8 recipe = "
          f"int8 bits={INT8_BITS} gs={INT8_GS} ({INT8_SCHEME})", flush=True)

    tok = AutoTokenizer.from_pretrained(QAT_UNQ, trust_remote_code=True)
    items = build_inputs(tok, args.n_seqs, args.max_comp, args.smoke)

    print("[main] loading bf16 QAT master ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        QAT_UNQ, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True,
    ).to(DEV).eval()
    name2mod = dict(model.named_modules())
    assert all(m in name2mod for m in modules), "module name mismatch"
    torch.cuda.reset_peak_memory_stats()
    print(f"[main] loaded; mem={torch.cuda.memory_allocated()/1e9:.1f}GB; {time.time()-t0:.0f}s", flush=True)

    # ---- int8 recipe sanity: int8 relerr should be << int4-g128 relerr ----
    wsamp = name2mod["model.language_model.layers.19.mlp.down_proj"].weight.detach()
    e4 = (wsamp.float() - fake_quant(wsamp, 4, 128)).norm() / wsamp.float().norm()
    e8 = (wsamp.float() - fake_quant(wsamp, 8, 128)).norm() / wsamp.float().norm()
    print(f"[recipe] L19.down_proj relerr int4_g128={e4:.5f} int8_g128={e8:.5f} "
          f"ratio={float(e4/e8):.1f}x (expect int8 ~order-16x finer)", flush=True)

    # ---------- reference pass (pristine bf16): cache hidden + build masks -----
    print("[ref] caching reference hidden + massive-dim masks ...", flush=True)
    ref_final, ref_all = [], []
    n_hidden = H = dimabs = None
    npos = 0
    for it in items:
        hs = forward_hidden(model, it["ids"])
        if n_hidden is None:
            n_hidden = len(hs); H = hs[0].shape[-1]
            dimabs = torch.zeros(n_hidden, H, device=DEV)
        pl = it["prompt_len"]
        for l in range(n_hidden):
            dimabs[l] += hs[l][0].float().abs().sum(0)
        npos += hs[0].shape[1]
        ref_final.append(hs[-1][0, pl:].to("cpu", torch.bfloat16))
        ref_all.append([hs[l][0, pl:].to("cpu", torch.bfloat16) for l in range(n_hidden)])
        del hs
    meanabs = dimabs / max(npos, 1)
    masks = torch.ones_like(meanabs)
    nmask = []
    for l in range(n_hidden):
        v = meanabs[l]
        med = v.median().clamp_min(1e-9)
        big = (v > 30.0 * med).nonzero().flatten()
        if big.numel() > 64:
            big = big[torch.argsort(-v[big])[:64]]
        masks[l, big] = 0.0
        nmask.append(int(big.numel()))
    print(f"[ref] cached {len(items)} seqs; n_hidden={n_hidden} H={H}; "
          f"massive dims/layer min={min(nmask)} max={max(nmask)}; "
          f"peak={torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)

    # ---------- swapper + run helper ------------------------------------------
    print("[main] precomputing g128/g32/int8 weights for 343 modules ...", flush=True)
    sw = Swapper(model, name2mod, modules, need_g32=True)
    print(f"[main] swapper ready; {time.time()-t0:.0f}s "
          f"mem={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    def run_levelmap(level_map, want_layers=False):
        sw.apply(level_map)
        acc = ErrorAccumulator(n_hidden, masks)
        for i, it in enumerate(items):
            hs = forward_hidden(model, it["ids"])
            rf = ref_final[i].to(DEV)
            ra = [t.to(DEV) for t in ref_all[i]] if want_layers else None
            acc.add_seq(hs, rf, ra, it["prompt_len"], want_layers)
            del hs, rf, ra
        return acc

    def run_pristine():
        acc = ErrorAccumulator(n_hidden, masks)
        for i, it in enumerate(items):
            hs = forward_hidden(model, it["ids"])
            acc.add_seq(hs, ref_final[i].to(DEV), None, it["prompt_len"], False)
            del hs
        return acc

    METRIC = "relmse_final"

    # ---------- bf16 pipeline self-check (weights still pristine) --------------
    acc_bf16 = run_pristine()
    bf16_err = acc_bf16.summary()[METRIC]
    bf16_ok = bf16_err < 1e-6
    print(f"[selfcheck] bf16-vs-ref relmse_final={bf16_err:.2e} (expect ~0) ok={bf16_ok}", flush=True)

    # ---------- anchors --------------------------------------------------------
    print("[anchor] g128_floor (int4-g128 all = 0% recovery) ...", flush=True)
    acc_g128 = run_levelmap(sw.map_all("g128"), want_layers=True)
    s_g128 = acc_g128.summary()
    acc_g128b = run_levelmap(sw.map_all("g128"), want_layers=False)
    self_det = abs(acc_g128b.summary()[METRIC] - s_g128[METRIC]) < 1e-9
    err_g128 = s_g128[METRIC]
    ps_g128 = acc_g128.per_seq()
    print(f"   g128_floor relmse_final={err_g128:.5f} self_det={self_det} "
          f"(#711 anchor=0.12033)", flush=True)

    print("[anchor] g32_full (int4-g32 all = int4 ceiling, CONTEXT) ...", flush=True)
    acc_g32 = run_levelmap(sw.map_all("g32"), want_layers=True)
    s_g32 = acc_g32.summary()
    err_g32 = s_g32[METRIC]
    print(f"   g32_full relmse_final={err_g32:.5f} (#711 anchor=0.04432)", flush=True)

    print("[anchor] int8_full (int8 L14-27 = 100% reference) ...", flush=True)
    full_set = set(band_modules(*FULL, modules))
    acc_full = run_levelmap(sw.map_band_int8(full_set), want_layers=True)
    s_full = acc_full.summary()
    err_full = s_full[METRIC]
    ps_full = acc_full.per_seq()
    denom = err_g128 - err_full
    print(f"   int8_full relmse_final={err_full:.5f} ({len(full_set)} int8 modules); "
          f"g128->int8_full gap={denom:.5f}", flush=True)

    def rec_frac(e):
        return (err_g128 - e) / denom if denom != 0 else float("nan")

    # ---------- band sweep -----------------------------------------------------
    band_res = {}
    band_ps = {}
    band_layers = {}
    print("[sweep] measuring bands ...", flush=True)
    for (lo, hi) in all_bands:
        bset = set(band_modules(lo, hi, modules))
        wl = (lo, hi) in want_layer_bands
        acc = run_levelmap(sw.map_band_int8(bset), want_layers=wl)
        s = acc.summary()
        e = s[METRIC]
        band_res[(lo, hi)] = {"lo": lo, "hi": hi, "band": f"L{lo}-{hi}",
                              "n_layers": hi - lo + 1, "n_modules": len(bset),
                              "relmse_final": e, "recovery_fraction": rec_frac(e),
                              "summary": s}
        band_ps[(lo, hi)] = acc.per_seq()
        if wl:
            band_layers[(lo, hi)] = s.get("layer_relmse")
        print(f"   L{lo:2d}-{hi:2d} (w={hi-lo+1:2d}, {len(bset):3d} mods) "
              f"relmse={e:.5f} rec={rec_frac(e):+.4f}  ({time.time()-t0:.0f}s)", flush=True)

    # ---------- bootstrap CIs on recovery fractions ----------------------------
    print("[boot] bootstrapping recovery-fraction CIs ...", flush=True)
    rng = np.random.default_rng(0)
    n = len(ps_g128)
    boot = {b: [] for b in all_bands}
    boot_g32 = []
    for _ in range(4000):
        idx = rng.integers(0, n, n)
        e128 = ps_g128[idx].mean()
        efull = ps_full[idx].mean()
        d = e128 - efull
        if d == 0:
            continue
        for b in all_bands:
            boot[b].append(float((e128 - band_ps[b][idx].mean()) / d))
        boot_g32.append(float((e128 - acc_g32.per_seq()[idx].mean()) / d))
    for b in all_bands:
        arr = np.asarray(boot[b])
        band_res[b]["rec_ci_lo"] = float(np.percentile(arr, 2.5))
        band_res[b]["rec_ci_hi"] = float(np.percentile(arr, 97.5))
    g32_rec = rec_frac(err_g32)
    g32_rec_ci = [float(np.percentile(boot_g32, 2.5)), float(np.percentile(boot_g32, 97.5))]

    # ---------- knee identification --------------------------------------------
    # POSITION-OPTIMIZED frontier: for each band-WIDTH, take the best-recovery contiguous
    # band of that width among all measured bands (so the minimal band is not mis-centered,
    # per the card's asymmetry warning). The knee = the narrowest width whose best-positioned
    # band recovers >= threshold of the full-band forward-fidelity.
    bywidth = {}
    for b in all_bands:
        w = b[1] - b[0] + 1
        rr = band_res[b]
        if w not in bywidth or rr["recovery_fraction"] > bywidth[w]["recovery_fraction"]:
            bywidth[w] = rr
    widths_sorted = sorted(bywidth)
    opt_frontier = [dict(bywidth[w]) for w in widths_sorted]

    def find_knee_opt(thr):
        for w in widths_sorted:
            if bywidth[w]["recovery_fraction"] >= thr:
                return bywidth[w]
        return bywidth[widths_sorted[-1]]

    def knee_dict(r, thr):
        return {"threshold": thr, "band": r["band"], "lo": r["lo"], "hi": r["hi"],
                "n_layers": r["n_layers"], "n_modules": r["n_modules"],
                "recovery_fraction": r["recovery_fraction"],
                "rec_ci_lo": r["rec_ci_lo"], "rec_ci_hi": r["rec_ci_hi"]}

    knee = knee_dict(find_knee_opt(0.85), 0.85)        # primary: position-optimized 85% knee
    knee90d = knee_dict(find_knee_opt(0.90), 0.90)

    # also keep the a-priori L19-centered ladder knee (shows centering matters)
    ladder_rows = sorted([dict(band_res[b]) for b in LADDER], key=lambda r: r["n_layers"])
    def find_knee_ladder(thr):
        for r in ladder_rows:
            if r["recovery_fraction"] >= thr:
                return r
        return ladder_rows[-1]
    knee_L19 = knee_dict(find_knee_ladder(0.85), 0.85)

    # ---------- center / asymmetry map (sliding width-5) -----------------------
    sliding_rows = [dict(band_res[b]) for b in SLIDING]
    best_w5 = max(sliding_rows, key=lambda r: r["recovery_fraction"])
    # asymmetry: compare recovery of lower-half vs upper-half windows around the best center
    centers = [(r["lo"] + r["hi"]) / 2.0 for r in sliding_rows]
    recs = [r["recovery_fraction"] for r in sliding_rows]
    cbest = (best_w5["lo"] + best_w5["hi"]) / 2.0
    lower = [r for r in sliding_rows if (r["lo"] + r["hi"]) / 2.0 < cbest]
    upper = [r for r in sliding_rows if (r["lo"] + r["hi"]) / 2.0 > cbest]
    mean_lower = float(np.mean([r["recovery_fraction"] for r in lower])) if lower else float("nan")
    mean_upper = float(np.mean([r["recovery_fraction"] for r in upper])) if upper else float("nan")
    if not (lower and upper):
        skew = "edge"
    elif abs(mean_lower - mean_upper) < 0.05:
        skew = "symmetric"
    else:
        skew = "skew_lower" if mean_lower > mean_upper else "skew_upper"

    # ---------- verdict --------------------------------------------------------
    full_w = FULL[1] - FULL[0] + 1
    layers_saved = full_w - knee["n_layers"]
    # NARROWABLE: a position-optimized band materially narrower than fern's 14 layers
    # (>=~2 layers fewer) recovers >=85% of the full-band forward-fidelity.
    narrowable = bool(knee["n_layers"] < full_w and knee["recovery_fraction"] >= 0.85
                      and layers_saved >= 2)
    verdict = "INT8_BAND_NARROWABLE" if narrowable else "INT8_BAND_IRREDUCIBLE"
    primary = float(knee["recovery_fraction"])

    # ---------- AIME reconciliation (unverified, assumed-linear hop; flagged) ---
    aime_recon = {
        "regime": "fern #659 n=60 greedy gb6144 (matched regime for this band)",
        "aime_int4_g128_floor": AIME_INT4_G128,
        "aime_int8_fullband_L14_27": AIME_INT8_FULLBAND,
        "aime_bf16_ceiling": AIME_BF16, "aime_gate": AIME_GATE,
        "full_band_forward_fidelity_recovery": 1.0,
        "full_band_delta_aime": AIME_INT8_FULLBAND - AIME_INT4_G128,
        "assumed_linear_map": "aime ~= 0.400 + recovery_fraction * 0.050 (ASSUMED, NOT measured here)",
        "knee_band_implied_aime_IF_LINEAR": AIME_INT4_G128 + primary * (AIME_INT8_FULLBAND - AIME_INT4_G128),
        "knee_band_implied_aime_clears_gate_IF_LINEAR": bool(
            AIME_INT4_G128 + primary * (AIME_INT8_FULLBAND - AIME_INT4_G128) >= AIME_GATE),
        "flag": ("forward-fidelity is a GLOBAL final-hidden relMSE; AIME may depend on "
                 "LOCALIZED fidelity, so X% global recovery != X% AIME recovery. SCREEN "
                 "output to hand to an AIME measurement, not a recovery claim."),
    }

    print(f"[verdict] {verdict} (knee saves {layers_saved} of {full_w} int8 layers)", flush=True)
    print(f"[knee>=85% pos-opt] {knee['band']} ({knee['n_layers']} layers) "
          f"rec={knee['recovery_fraction']:+.4f} CI=[{knee['rec_ci_lo']:+.4f},"
          f"{knee['rec_ci_hi']:+.4f}]  (full band = {full_w} layers)", flush=True)
    print(f"[knee>=90% pos-opt] {knee90d['band']} ({knee90d['n_layers']} layers) "
          f"rec={knee90d['recovery_fraction']:+.4f}", flush=True)
    print(f"[knee>=85% L19-ladder] {knee_L19['band']} ({knee_L19['n_layers']} layers) "
          f"rec={knee_L19['recovery_fraction']:+.4f}  (mis-centered, for contrast)", flush=True)
    print("[opt frontier] width -> best-positioned recovery:", flush=True)
    for r in opt_frontier:
        print(f"   w={r['n_layers']:2d}  {r['band']:>8}  rec={r['recovery_fraction']:+.4f} "
              f"CI=[{r['rec_ci_lo']:+.4f},{r['rec_ci_hi']:+.4f}]", flush=True)
    print(f"[center] best w5 = {best_w5['band']} rec={best_w5['recovery_fraction']:+.4f}; "
          f"skew={skew} (lower={mean_lower:.3f} upper={mean_upper:.3f})", flush=True)
    print(f"[context] g32_full(int4 ceiling) rec_vs_int8band={g32_rec:+.4f} "
          f"CI=[{g32_rec_ci[0]:+.4f},{g32_rec_ci[1]:+.4f}]", flush=True)

    # ---------- per-layer profiles --------------------------------------------
    per_layer = None
    if band_layers.get(FULL) and s_g128.get("layer_relmse"):
        per_layer = {
            "g128": s_g128["layer_relmse"],
            "g32_full": s_g32["layer_relmse"],
            "int8_full": band_layers.get(FULL),
            "peak_band": band_layers.get(peak_band),
        }

    # ---------- result ---------------------------------------------------------
    result = {
        "pr": 718, "analysis_only": True, "official_tps": 0, "no_hf_job": True, "fires": 0,
        "reference": QAT_UNQ, "metric": METRIC,
        "int8_recipe": {"bits": INT8_BITS, "group_size": INT8_GS, "scheme": INT8_SCHEME,
                        "source": "fern #659 bit-width-only clone of int4-g128 (PR #659 body); "
                                  "logged config nmjvtfov: upgrade_layers=14-27, "
                                  "upgrade_precision=int8, skeleton=int4_g128_lmhead, "
                                  "source=qat-unquantized; group_size/scheme ASSUMED from int4-g128"},
        "bf16_selfcheck_relmse": bf16_err, "bf16_selfcheck_ok": bf16_ok,
        "g128_self_det": bool(self_det),
        "n_seqs": len(items), "max_comp": args.max_comp,
        "task_mix": {t: sum(1 for x in items if x["task"] == t) for t in TASKS},
        "err_g128_floor": err_g128, "err_g32_full": err_g32, "err_int8_full": err_full,
        "g128_to_int8_full_gap": denom,
        "g32_full_recovery_fraction": g32_rec, "g32_full_recovery_ci95": g32_rec_ci,
        "full_band": {"band": f"L{FULL[0]}-{FULL[1]}", "n_layers": FULL[1] - FULL[0] + 1,
                      "n_modules": len(full_set)},
        "ladder": [dict(band_res[b]) for b in LADDER],
        "ladder_up": [dict(band_res[b]) for b in LADDER_UP],
        "opt_frontier": [dict(r) for r in opt_frontier],
        "sliding_w5": [dict(band_res[b]) for b in SLIDING],
        "all_bands": {f"L{lo}-{hi}": {k: band_res[(lo, hi)][k] for k in
                      ("lo", "hi", "n_layers", "n_modules", "relmse_final",
                       "recovery_fraction", "rec_ci_lo", "rec_ci_hi")}
                      for (lo, hi) in all_bands},
        "knee": knee, "knee_90": knee90d, "knee_L19_ladder": knee_L19,
        "best_w5_center": {"band": best_w5["band"], "recovery_fraction": best_w5["recovery_fraction"]},
        "asymmetry": {"skew": skew, "mean_recovery_lower": mean_lower,
                      "mean_recovery_upper": mean_upper,
                      "centers": centers, "recoveries": recs},
        "verdict": verdict,
        "int8_band_narrowable": int(narrowable),
        "min_int8_band_fidelity_recovery_fraction": primary,
        "aime_reconciliation": aime_recon,
        "per_layer": per_layer,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        "elapsed_s": time.time() - t0,
        "smoke": args.smoke,
    }
    # strip bulky per-seq summaries from band rows before write (keep scalars)
    for r in (result["ladder"] + result["ladder_up"] + result["opt_frontier"]
              + result["sliding_w5"]):
        r.pop("summary", None)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in
          ["verdict", "int8_band_narrowable", "min_int8_band_fidelity_recovery_fraction",
           "err_g128_floor", "err_g32_full", "err_int8_full",
           "bf16_selfcheck_ok", "g128_self_det"]}, indent=2), flush=True)
    print(f"[done] wrote {args.out}; peak={result['peak_gb']:.1f}GB; {result['elapsed_s']:.0f}s", flush=True)

    if not args.no_wandb:
        rid = log_wandb(result, args.out, resume_id=args.wandb_resume_id)
        print(f"[wandb] run id: {rid}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
