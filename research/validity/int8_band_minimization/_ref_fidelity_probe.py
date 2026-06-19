#!/usr/bin/env python
"""PR #711 -- Activation-fidelity recovery probe: is denken's energy->recovery
map LINEAR, CONCAVE, or CONVEX?  DISK-SAFE, ANALYSIS-ONLY.

Decision-forcing question (advisor): the int4-body recovery program assumes the
cumulative impact-energy fraction a selective-g32 module set captures (ubel #700's
pareto proxy) EQUALS the fraction of the full-recovery activation-fidelity gap it
actually closes -- i.e. recovery_fraction == cum_energy_fraction, the y=x diagonal.
This probe measures the TRUE joint activation recovery and characterizes the shape
of recovery(energy) against that diagonal.

PRIMARY axis (denken-proxy validity):
  x = #700 cum-energy fraction of the pareto-ordered set pareto48[:k]  (sel700.json)
  y = MEASURED joint activation-recovery fraction of that exact set
  y < x  -> proxy OVER-predicts (CONCAVE / sublinear) -> kill confirmed   (code -1)
  y ~ x  -> proxy CALIBRATED (LINEAR)                                     (code  0)
  y > x  -> proxy UNDER-predicts (CONVEX / superlinear) -> rescue possible(code +1)

SECONDARY axis (self-contained additivity, independent of #700's energy def):
  measure each module's STANDALONE forward recovery e_m, predict S by sum(e_m),
  compare to measured joint recovery -> sub/super-additivity ratio.

Mechanism (all in-memory fake-quant of ONE bf16 master; NO checkpoint write, NO
dataset eval, NO autoregressive generation):
  * Reference  = bf16 QAT master (/workspace/gemma_build/qat_unq) -- the exact
    source submissions/int4_g128_lmhead/build_quant.py quantizes. Loaded once.
  * Each body config = dequant(quant(W, gs)) per module, swapped in place:
      g128       : all 343 body modules @ g128            (locked-submission damage)
      g32_full   : all 343 body modules @ g32             (full-recovery ceiling)
      S          : modules in S @ g32, the rest @ g128     (selective recovery)
  * Teacher-force a fixed calibration set (KCG base-bf16 greedy completions),
    capture per-layer residual-stream hidden states, measure activation error vs
    the bf16 reference on COMPLETION positions only.
  * err(config) = relMSE of the FINAL hidden (+ masked-cosine, + per-layer).
  * recovery_fraction(S) = (err_g128 - err_S) / (err_g128 - err_g32_full).
  * Route A = plig40 (40 PLIG @ g32); Route B = pareto48 (40 PLIG + 8 qkv @ g32).

Run (assigned GPU):
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 uv run python \
    research/validity/activation_recovery_fidelity/fidelity_probe.py \
    --n-seqs 120 --max-comp 200 [--smoke] [--no-wandb] [--skip-individual]
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
HERE = ROOT / "research/validity/activation_recovery_fidelity"
KCG = ROOT / "research/validity/keepset_coverage_gap"
QAT_UNQ = "/workspace/gemma_build/qat_unq"
MODULE_LIST = ROOT / "submissions/int4_g128_lmhead/official_quantized_modules.json"
SEL700 = HERE / "sel700.json"
DEV = "cuda"
TASKS = ("aime2024", "gpqa_diamond", "mmlu_pro")

# denken / land anchors for the AIME extrapolation (public, relayed in PR #711)
AIME_G128 = 0.347
AIME_G32FULL = 0.438
AIME_GATE = 0.420
# implied AIME under the (assumed-linear) activation->AIME map:
#   aime = AIME_G128 + recovery_fraction * (AIME_G32FULL - AIME_G128)
AIME_SLOPE = AIME_G32FULL - AIME_G128            # 0.091
REQ_RECOVERY = (AIME_GATE - AIME_G128) / AIME_SLOPE  # 0.802 to clear the gate


# ----------------------------- fake-quant -----------------------------------
def _qargs(gs: int) -> QuantizationArgs:
    return QuantizationArgs(num_bits=4, type="int", strategy="group",
                            group_size=gs, symmetric=True, observer="minmax")


def fake_quant(W: torch.Tensor, gs: int) -> torch.Tensor:
    """dequant(quant(W, gs)) -- matches build_quant.py minmax recipe exactly."""
    Wf = W.to(torch.float32)
    out_dim, in_dim = Wf.shape
    qa = _qargs(gs)
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


# ----------------------------- inputs ---------------------------------------
def build_inputs(tokenizer, n_seqs, max_comp, smoke):
    """Rebuild teacher-forcing sequences: chat-template prompt + base-bf16 greedy
    completion (no generation). Balanced across tasks, AIME-weighted."""
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
        quota = {"aime2024": n_aime,
                 "gpqa_diamond": rest // 2,
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


# ----------------------------- forward / error ------------------------------
@torch.no_grad()
def forward_hidden(model, ids):
    t = torch.tensor([ids], device=DEV)
    out = model(input_ids=t, output_hidden_states=True, use_cache=False)
    return out.hidden_states  # tuple(L+1) of [1,T,H]


class ErrorAccumulator:
    """Per-seq activation error vs cached reference, completion positions only."""

    def __init__(self, n_layers_p1, masks):
        self.nl = n_layers_p1
        self.masks = masks                 # [L+1, H] on GPU (0 at massive dims)
        self.seq_relmse_final = []
        self.seq_cosdist_final = []
        self.seq_relmse_masked_final = []
        self.layer_relmse = None
        self.layer_cosdist = None
        self.layer_cnt = 0

    def add_seq(self, var_hs, ref_hs_final, ref_hs_all, pl, want_layers):
        vf = var_hs[-1][0, pl:].float()                       # [Tc,H]
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
            "cosdist_masked_final_sem": float(c.std(ddof=1) / math.sqrt(len(c))) if len(c) > 1 else 0.0,
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
    """Holds CPU bf16 g128 + g32 weights for all 343 body modules; swaps GPU
    weights in place. Sentinel state 'BF16' = pristine (no quant applied yet)."""

    BF16 = "BF16"

    def __init__(self, model, name2mod, modules):
        self.n2m = name2mod
        self.modules = modules
        self.cpu_g128, self.cpu_g32 = {}, {}
        for m in modules:
            W = self.n2m[m].weight.detach()
            self.cpu_g128[m] = fake_quant(W, 128).to("cpu")
            self.cpu_g32[m] = fake_quant(W, 32).to("cpu")
        self.current = self.BF16          # weights are still pristine bf16

    def set_base_g128(self):
        for m in self.modules:
            self.n2m[m].weight.data.copy_(self.cpu_g128[m].to(DEV))
        self.current = set()

    def set_g32(self, S):
        """Make exactly the set S be g32 (rest g128), minimal swaps from current."""
        S = set(S)
        if self.current == self.BF16:
            self.set_base_g128()
        to_g32 = S - self.current
        to_g128 = self.current - S
        for m in to_g32:
            self.n2m[m].weight.data.copy_(self.cpu_g32[m].to(DEV))
        for m in to_g128:
            self.n2m[m].weight.data.copy_(self.cpu_g128[m].to(DEV))
        self.current = S


# ----------------------------- W&B (reads only from result dict) ------------
def log_wandb(result, out_path, resume_id=None):
    import wandb
    rA, rB = result["route_A"], result["route_B"]
    aime = result["aime_extrapolation"]
    init_kw = dict(
        entity="wandb-applied-ai-team", project="gemma-challenge-senpai",
        name="stark/activation-recovery-fidelity",
        group="activation-recovery-fidelity-stark",
        config={"analysis_only": True, "official_tps": 0, "no_hf_job": 1, "fires": 0,
                "reference": result["reference"], "metric": result["metric"],
                "n_seqs": result["n_seqs"], "max_comp": result["max_comp"],
                "task_mix": result["task_mix"], "candidate_source": "ubel700_pareto"},
    )
    if resume_id:
        init_kw.update(id=resume_id, resume="allow")
    run = wandb.init(**init_kw)
    wandb.summary.update({
        "verdict": result["verdict"], "verdict_code": result["verdict_code"],
        "bf16_selfcheck_relmse": result["bf16_selfcheck_relmse"], "self_det": result["self_det"],
        "err_g128": result["err_g128"], "err_g32_full": result["err_g32_full"], "gap": result["gap"],
        "route_A_proxy_x": rA["proxy_x"], "route_A_recovery": rA["joint_recovery"],
        "route_B_proxy_x": rB["proxy_x"], "route_B_recovery": rB["joint_recovery"],
        "route_B_recovery_ci_lo": rB["recovery_ci95"][0], "route_B_recovery_ci_hi": rB["recovery_ci95"][1],
        "route_B_size": rB["size"],
        "route_B_isolated_recovery": result.get("route_B_isolated", {}).get("joint_recovery"),
        "mean_diag_residual": result["mean_diag_residual"],
        "diag_residual_ci_lo": result["diag_residual_ci95"][0],
        "diag_residual_ci_hi": result["diag_residual_ci95"][1],
        "diag_slope": result["diag_slope"], "diag_curvature_x2": result["diag_curvature_x2"],
        "aime_routeB": aime["aime_routeB"], "clears_gate": aime["clears_gate"],
        "denken_assumed_routeB_recovery": aime["denken_assumed_routeB_recovery"],
        "rank_agreement_spearman": result["rank_agreement_spearman_my_energy_vs_700"],
        "primary_metric": result["primary_metric_routeB_recovery"],
        "test_metric": result["verdict_code"], "peak_gb": result["peak_gb"],
    })
    # diagonal: numeric-k sweep rows + Route A row (k stays numeric, label disambiguates)
    tsw = wandb.Table(columns=["label", "k", "proxy_cum_energy_x", "joint_recovery_y", "diag_residual"])
    for s in result["sweep"]:
        tsw.add_data(f"k{s['k']}", s["k"], s["proxy_cum_energy_x"],
                     s["joint_recovery_y"], s["diag_residual"])
    tsw.add_data("routeA_plig40", rA["size"], rA["proxy_x"], rA["joint_recovery"], rA["diag_residual"])
    wandb.log({"diagonal_curve": tsw})
    for s in result["sweep"]:
        wandb.log({"sweep_k": s["k"], "proxy_x": s["proxy_cum_energy_x"],
                   "measured_y": s["joint_recovery_y"], "diag_residual": s["diag_residual"]})
    if result.get("energy_per_module"):
        gap = result["gap"]
        ten = wandb.Table(columns=["module", "type", "layer", "energy", "recovery_frac", "rank700"])
        for i, m in enumerate(result["pareto48"]):
            e = result["energy_per_module"][m]
            ten.add_data(short(m), mtype(m), layer_of(m), e, e / gap, i)
        wandb.log({"per_module_energy": ten})
    if result.get("additivity"):
        tad = wandb.Table(columns=["k", "joint_recovery", "additive_pred", "ratio"])
        for a in result["additivity"]:
            tad.add_data(a["k"], a["joint_recovery"], a["additive_pred"], a["ratio_joint_over_add"])
        wandb.log({"additivity_curve": tad})
    art = wandb.Artifact("activation_recovery_fidelity", type="analysis")
    art.add_file(out_path)
    run.log_artifact(art)
    run.finish()
    return run.id


# ----------------------------- main -----------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seqs", type=int, default=120)
    ap.add_argument("--max-comp", type=int, default=200)
    ap.add_argument("--candidate-json", default=str(SEL700),
                    help="#700 selection JSON: {pareto48, cum_energy, plig40, qkv8}")
    ap.add_argument("--skip-individual", action="store_true",
                    help="skip the per-module standalone-energy secondary axis (saves ~48 passes)")
    ap.add_argument("--out", default=str(HERE / "result.json"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--relog-from", default=None,
                    help="load an existing result.json and ONLY (re)log it to W&B, then exit")
    ap.add_argument("--wandb-resume-id", default=None,
                    help="resume/finish an existing W&B run id instead of starting a new one")
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
    plig_all = sorted([m for m in modules if m.endswith("per_layer_input_gate")], key=layer_of)
    print(f"[main] {len(modules)} body modules, {len(plig_all)} PLIG", flush=True)

    # ---- #700 pareto selection (primary axis) --------------------------------
    cj = json.load(open(args.candidate_json))
    pareto48 = list(cj["pareto48"])
    cum_energy = list(cj["cum_energy"])           # aligned with pareto48
    plig40 = list(cj["plig40"])
    qkv8 = list(cj["qkv8"])
    assert len(pareto48) == len(cum_energy)
    assert all(m in modules for m in pareto48), "pareto48 module not in official list"
    assert set(plig40) | set(qkv8) == set(pareto48), "plig40+qkv8 != pareto48"
    # incremental proxy energy per pareto module (denken's additive saliency)
    proxy_e, prev = {}, 0.0
    for m, ce in zip(pareto48, cum_energy):
        proxy_e[m] = ce - prev
        prev = ce
    routeA_proxy_x = float(sum(proxy_e[m] for m in plig40))
    routeB_iso_proxy_x = float(cum_energy[-1])            # pareto48 (isolated 8 qkv) = diagonal terminal
    # Route B (PR spec / land #708 servable): whole-qkv-block promotion of every
    # layer that has any #700-targeted q/k/v. Gemma-4-E4B: layers >=24 are q-only
    # (24 k/v total), so L40/L41 contribute q_proj only.
    targeted_layers = sorted({layer_of(m) for m in qkv8})
    qkv_whole = [f"model.language_model.layers.{L}.self_attn.{p}"
                 for L in targeted_layers for p in ("q_proj", "k_proj", "v_proj")
                 if f"model.language_model.layers.{L}.self_attn.{p}" in modules]
    routeB_whole = list(plig40) + qkv_whole
    incidental = [m for m in qkv_whole if m not in qkv8]
    # whole-block proxy-x = isolated 0.7996 + #700 cum-energy of the incidental modules.
    # incidental modules fell below #700's top-48 cut, so their proxy energy is small
    # and un-tabulated; report the isolated value as the proxy lower bound.
    routeB_proxy_x = routeB_iso_proxy_x
    print(f"[#700] pareto48 loaded; Route A proxy-x={routeA_proxy_x:.4f} "
          f"Route B(iso) proxy-x={routeB_iso_proxy_x:.4f}; "
          f"Route B(whole)={len(routeB_whole)} mods (+{len(incidental)} incidental: "
          f"{[short(m) for m in incidental]})", flush=True)

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
    print("[main] precomputing g128/g32 weights for 343 modules ...", flush=True)
    sw = Swapper(model, name2mod, modules)

    def run_config(S_g32, want_layers=False):
        sw.set_g32(S_g32)
        acc = ErrorAccumulator(n_hidden, masks)
        for i, it in enumerate(items):
            hs = forward_hidden(model, it["ids"])
            rf = ref_final[i].to(DEV)
            ra = [t.to(DEV) for t in ref_all[i]] if want_layers else None
            acc.add_seq(hs, rf, ra, it["prompt_len"], want_layers)
            del hs, rf, ra
        return acc

    def run_pristine():
        """forward with the still-pristine bf16 weights (sentinel BF16) -- err~0."""
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
    print("[anchor] g128 (damage baseline) ...", flush=True)
    acc_g128 = run_config(set(), want_layers=True)
    s_g128 = acc_g128.summary()
    acc_g128b = run_config(set(), want_layers=False)
    self_det = abs(acc_g128b.summary()[METRIC] - s_g128[METRIC]) < 1e-9
    print(f"   g128: relmse_final={s_g128['relmse_final']:.5f} "
          f"cosdist={s_g128['cosdist_masked_final']:.5f} self_det={self_det}", flush=True)

    print("[anchor] g32_full (recovery ceiling) ...", flush=True)
    acc_g32 = run_config(set(modules), want_layers=True)
    s_g32 = acc_g32.summary()
    print(f"   g32_full: relmse_final={s_g32['relmse_final']:.5f} "
          f"cosdist={s_g32['cosdist_masked_final']:.5f}", flush=True)

    err_g128 = s_g128[METRIC]
    err_g32 = s_g32[METRIC]
    gap = err_g128 - err_g32
    print(f"[anchor] err_g128={err_g128:.5f} err_g32full={err_g32:.5f} gap={gap:.5f} "
          f"(g128 {gap/err_g32*100:.0f}% above floor)", flush=True)
    ps_g128 = acc_g128.per_seq()
    ps_g32 = acc_g32.per_seq()

    def recovery(acc):
        e = acc.summary()[METRIC]
        return (err_g128 - e) / gap if gap != 0 else float("nan")

    # ---------- PRIMARY: pareto cum-energy diagonal sweep ----------------------
    # k=48 == Route B (full pareto48) is run separately below and appended as the
    # final diagonal point, so the loop covers the intermediate k only.
    sweep_ks = [8, 16, 24, 32, 40] if not args.smoke else [4]
    sweep = []
    ps_sweep = {}
    print("[sweep] pareto48[:k] joint recovery vs #700 cum-energy diagonal ...", flush=True)
    for k in sweep_ks:
        S = pareto48[:k]
        want = (k == 16) or args.smoke
        acc = run_config(set(S), want_layers=want)
        y = recovery(acc)
        x = float(cum_energy[k - 1])              # #700 proxy cum-energy of first k
        ps_sweep[k] = acc.per_seq()
        sweep.append({"k": k, "proxy_cum_energy_x": x, "joint_recovery_y": y,
                      "diag_residual": y - x, "summary": acc.summary()})
        print(f"   [k={k:2d}] proxy_x={x:.4f} measured_y={y:+.4f} "
              f"resid(y-x)={y-x:+.4f}  ({time.time()-t0:.0f}s)", flush=True)

    # ---------- Route A (plig40) ----------------------------------------------
    print("[routes] Route A (40 PLIG) ...", flush=True)
    accA = run_config(set(plig40), want_layers=True)
    recA = recovery(accA)
    ps_A = accA.per_seq()
    print(f"   Route A: proxy_x={routeA_proxy_x:.4f} measured_y={recA:+.4f} "
          f"resid={recA-routeA_proxy_x:+.4f}", flush=True)

    # ---------- Route B isolated (pareto48) == k=48 diagonal terminal ----------
    print("[routes] Route B-isolated (pareto48 = 40 PLIG + 8 isolated qkv) ...", flush=True)
    accBiso = run_config(set(pareto48), want_layers=True)
    recBiso = recovery(accBiso)
    ps_Biso = accBiso.per_seq()
    ps_sweep[len(pareto48)] = ps_Biso
    sweep.append({"k": len(pareto48), "proxy_cum_energy_x": routeB_iso_proxy_x,
                  "joint_recovery_y": recBiso, "diag_residual": recBiso - routeB_iso_proxy_x,
                  "summary": accBiso.summary()})
    print(f"   Route B-iso: proxy_x={routeB_iso_proxy_x:.4f} measured_y={recBiso:+.4f} "
          f"resid={recBiso-routeB_iso_proxy_x:+.4f}", flush=True)

    # ---------- Route B whole-block (PR spec / #708 servable) = primary_metric -
    print(f"[routes] Route B-whole ({len(routeB_whole)} mods, servable) ...", flush=True)
    accB = run_config(set(routeB_whole), want_layers=True)
    recB = recovery(accB)
    ps_B = accB.per_seq()
    print(f"   Route B-whole: proxy_x>={routeB_proxy_x:.4f} measured_y={recB:+.4f} "
          f"resid={recB-routeB_proxy_x:+.4f}", flush=True)

    # ---------- SECONDARY: self-contained additivity --------------------------
    energy = {}
    additivity = None
    rank_agreement = None
    if not args.skip_individual:
        probe_set = list(pareto48)
        print(f"[energy] measuring {len(probe_set)} standalone module energies ...", flush=True)
        for j, m in enumerate(probe_set):
            acc = run_config({m}, want_layers=False)
            energy[m] = err_g128 - acc.summary()[METRIC]      # standalone recovery (err units)
            if (j + 1) % 12 == 0 or j == len(probe_set) - 1:
                print(f"   [{j+1}/{len(probe_set)}] {short(m):42s} "
                      f"e={energy[m]:+.6f} rec={energy[m]/gap:+.4f}  ({time.time()-t0:.0f}s)", flush=True)
        # additive prediction (LINEAR/independent) vs measured joint over the sweep
        additivity = []
        for s in sweep:
            k = s["k"]
            S = pareto48[:k]
            add_pred = float(sum(energy[m] for m in S) / gap)
            additivity.append({"k": k, "joint_recovery": s["joint_recovery_y"],
                               "additive_pred": add_pred,
                               "ratio_joint_over_add": (s["joint_recovery_y"] / add_pred)
                               if add_pred else float("nan")})
        # rank agreement: my forward energy order vs #700 pareto order (Spearman)
        my_order = sorted(pareto48, key=lambda m: -energy[m])
        rk_700 = {m: i for i, m in enumerate(pareto48)}
        rk_mine = {m: i for i, m in enumerate(my_order)}
        d2 = sum((rk_700[m] - rk_mine[m]) ** 2 for m in pareto48)
        nn = len(pareto48)
        rank_agreement = 1.0 - 6.0 * d2 / (nn * (nn * nn - 1))
        print(f"[energy] additive ratios={[round(a['ratio_joint_over_add'],3) for a in additivity]} "
              f"spearman(my-energy, #700)={rank_agreement:.3f}", flush=True)

    # ---------- shape verdict (bootstrap the diagonal) -------------------------
    xks = np.array([s["proxy_cum_energy_x"] for s in sweep] + [routeA_proxy_x])
    rng = np.random.default_rng(0)
    n = len(ps_g128)
    boot_resid, boot_recB, boot_slope = [], [], []
    sweep_ps_list = [ps_sweep[s["k"]] for s in sweep]
    for _ in range(4000):
        idx = rng.integers(0, n, n)
        e128 = ps_g128[idx].mean(); e32 = ps_g32[idx].mean(); g = e128 - e32
        if g == 0:
            continue
        yk = np.array([(e128 - ps[idx].mean()) / g for ps in sweep_ps_list]
                      + [(e128 - ps_A[idx].mean()) / g])
        boot_resid.append(float(np.mean(yk - xks)))
        boot_recB.append(float((e128 - ps_B[idx].mean()) / g))
        boot_slope.append(float(np.polyfit(xks, yk, 1)[0]))
    boot_resid = np.asarray(boot_resid)
    boot_recB = np.asarray(boot_recB)
    boot_slope = np.asarray(boot_slope)
    mean_resid = float(boot_resid.mean())
    resid_lo, resid_hi = float(np.percentile(boot_resid, 2.5)), float(np.percentile(boot_resid, 97.5))
    recB_lo, recB_hi = float(np.percentile(boot_recB, 2.5)), float(np.percentile(boot_recB, 97.5))
    slope_mean = float(boot_slope.mean())
    # curvature: quadratic fit on the point cloud (sign of x^2 coefficient)
    ys_point = np.array([s["joint_recovery_y"] for s in sweep] + [recA])
    quad = np.polyfit(xks, ys_point, 2)
    curvature = float(quad[0])

    BAND = 0.02
    if resid_hi < -BAND:
        verdict, code = "PROXY_OVERPREDICTS_CONCAVE_SUBLINEAR", -1
    elif resid_lo > BAND:
        verdict, code = "PROXY_UNDERPREDICTS_CONVEX_SUPERLINEAR", 1
    else:
        verdict, code = "PROXY_CALIBRATED_LINEAR", 0

    # implied AIME under linear activation->AIME map
    aime_routeB = AIME_G128 + recB * AIME_SLOPE
    aime_routeB_lo = AIME_G128 + recB_lo * AIME_SLOPE
    aime_routeB_hi = AIME_G128 + recB_hi * AIME_SLOPE
    clears_gate = bool(recB >= REQ_RECOVERY)

    print(f"[verdict] mean_resid={mean_resid:+.4f} CI95=[{resid_lo:+.4f},{resid_hi:+.4f}] "
          f"slope={slope_mean:.3f} curvature(x^2)={curvature:+.3f}", flush=True)
    print(f"[verdict] {verdict} (code {code})", flush=True)
    print(f"[verdict] Route B recovery={recB:+.4f} CI95=[{recB_lo:+.4f},{recB_hi:+.4f}] "
          f"req={REQ_RECOVERY:.3f} clears_gate={clears_gate}", flush=True)
    print(f"[verdict] implied AIME(routeB)={aime_routeB:.4f} "
          f"CI95=[{aime_routeB_lo:.4f},{aime_routeB_hi:.4f}] gate={AIME_GATE}", flush=True)

    # ---------- result ---------------------------------------------------------
    result = {
        "pr": 711, "analysis_only": True, "official_tps": 0, "no_hf_job": True, "fires": 0,
        "reference": QAT_UNQ, "metric": METRIC,
        "bf16_selfcheck_relmse": bf16_err, "bf16_selfcheck_ok": bf16_ok, "self_det": bool(self_det),
        "n_seqs": len(items), "max_comp": args.max_comp,
        "task_mix": {t: sum(1 for x in items if x["task"] == t) for t in TASKS},
        "err_g128": err_g128, "err_g32_full": err_g32, "gap": gap,
        "cosdist_g128": s_g128["cosdist_masked_final"], "cosdist_g32": s_g32["cosdist_masked_final"],
        # PRIMARY diagonal
        "sweep": [{k: s[k] for k in ("k", "proxy_cum_energy_x", "joint_recovery_y", "diag_residual")}
                  for s in sweep],
        "mean_diag_residual": mean_resid, "diag_residual_ci95": [resid_lo, resid_hi],
        "diag_slope": slope_mean, "diag_curvature_x2": curvature,
        "route_A": {"set": "plig40", "size": len(plig40),
                    "proxy_x": routeA_proxy_x, "joint_recovery": recA,
                    "diag_residual": recA - routeA_proxy_x, "summary": accA.summary()},
        "route_B": {"set": "routeB_whole_block_servable", "size": len(routeB_whole),
                    "qkv_modules": [short(m) for m in qkv_whole],
                    "incidental_vs_iso": [short(m) for m in incidental],
                    "proxy_x": routeB_proxy_x, "proxy_x_is_lower_bound": True,
                    "joint_recovery": recB, "diag_residual": recB - routeB_proxy_x,
                    "recovery_ci95": [recB_lo, recB_hi], "summary": accB.summary()},
        "route_B_isolated": {"set": "pareto48", "size": len(pareto48),
                             "proxy_x": routeB_iso_proxy_x, "joint_recovery": recBiso,
                             "diag_residual": recBiso - routeB_iso_proxy_x,
                             "summary": accBiso.summary()},
        "verdict": verdict, "verdict_code": code,
        "primary_metric_routeB_recovery": recB,
        # AIME extrapolation (assumed-linear activation->AIME hop; flagged separate)
        "aime_extrapolation": {
            "map": "aime = 0.347 + recovery * 0.091 (assumed linear, NOT measured here)",
            "aime_routeB": aime_routeB, "aime_routeB_ci95": [aime_routeB_lo, aime_routeB_hi],
            "aime_g128": AIME_G128, "aime_g32full": AIME_G32FULL, "gate": AIME_GATE,
            "required_recovery_to_clear": REQ_RECOVERY, "clears_gate": clears_gate,
            "denken_assumed_routeB_recovery": routeB_iso_proxy_x,
        },
        # SECONDARY additivity
        "energy_per_module": {m: energy[m] for m in pareto48} if energy else None,
        "additivity": additivity,
        "rank_agreement_spearman_my_energy_vs_700": rank_agreement,
        # per-layer profiles
        "per_layer_g128_relmse": s_g128.get("layer_relmse"),
        "per_layer_g32_relmse": s_g32.get("layer_relmse"),
        "per_layer_routeA_relmse": accA.summary().get("layer_relmse"),
        "per_layer_routeB_relmse": accB.summary().get("layer_relmse"),
        "per_layer_routeB_iso_relmse": accBiso.summary().get("layer_relmse"),
        "pareto48": pareto48, "cum_energy_700": cum_energy,
        "plig40": plig40, "qkv8": qkv8, "routeB_whole": routeB_whole,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        "elapsed_s": time.time() - t0,
        "smoke": args.smoke,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in
          ["verdict", "verdict_code", "primary_metric_routeB_recovery",
           "mean_diag_residual", "diag_residual_ci95", "err_g128", "err_g32_full",
           "bf16_selfcheck_ok", "self_det"]}, indent=2), flush=True)
    print(f"[done] wrote {args.out}; peak={result['peak_gb']:.1f}GB; {result['elapsed_s']:.0f}s", flush=True)

    # ---------- W&B ------------------------------------------------------------
    if not args.no_wandb:
        rid = log_wandb(result, args.out, resume_id=args.wandb_resume_id)
        print(f"[wandb] run id: {rid}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
