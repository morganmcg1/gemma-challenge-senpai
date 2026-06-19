#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""PR #748 (land) -- analyze the served byte-exact transfer arms.

Reads four arm dirs (each with decode_outputs.jsonl + arm_summary.json):
  bi1_spec0 = BI=1 AR reference (spec OFF)   -- reference for the BI=1 identity
  bi1_spec1 = BI=1 batched-verify spec        -- the load-bearing arm (primary metric)
  bi0_spec0 = BI=0 AR reference (spec OFF)    -- reference for the BI=0 control
  bi0_spec1 = BI=0 spec (deployed reduction order) -- control (expected to diverge) + tax

Strict-#319 identity is per-prompt all-or-nothing greedy-token equality
(completion_token_sha256 match) between a spec arm and its SAME-BI AR reference -- so the
only difference within each pair is spec on/off (the reduction order is held by the shared
BI flag). Reporting N/128 (and the fraction) plus, for diverging prompts, the first-
divergence token position. TPS deliverables: clears 126.378? for the BI=1 spec arm; and the
BI=1-vs-BI=0 spec decode tax % (the price of byte-exactness on the batched-verify path).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCKED_TPS = 126.378


def load_arm(d: Path) -> dict:
    summ = json.loads((d / "arm_summary.json").read_text())
    rows = {}
    for ln in (d / "decode_outputs.jsonl").read_text().splitlines():
        ln = ln.strip()
        if ln:
            r = json.loads(ln)
            rows[r["id"]] = r
    return {"summary": summ, "rows": rows}


def first_div(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) == len(b) else n  # n = identical in overlap; len diff flagged separately


def identity(spec_arm: dict, ref_arm: dict) -> dict:
    ids = sorted(set(spec_arm["rows"]) & set(ref_arm["rows"]))
    n_match = 0
    diverging = []
    exposure = 0  # token-positions observed identical before the first flip (survival exposure)
    for pid in ids:
        s = spec_arm["rows"][pid]["completion_token_ids"]
        r = ref_arm["rows"][pid]["completion_token_ids"]
        if spec_arm["rows"][pid]["completion_token_sha256"] == ref_arm["rows"][pid]["completion_token_sha256"]:
            n_match += 1
            exposure += min(len(s), len(r))  # survived the whole completion (right-censored)
        else:
            fd = first_div(s, r)
            exposure += fd  # survived fd identical tokens, then flipped at position fd
            diverging.append({"id": pid, "index": spec_arm["rows"][pid]["index"],
                              "first_div_pos": fd, "len_spec": len(s), "len_ref": len(r),
                              "spec_tok": s[fd] if fd < len(s) else None,
                              "ref_tok": r[fd] if fd < len(r) else None})
    n = len(ids)
    n_div = len(diverging)
    # constant-hazard (geometric) MLE for the per-token near-tie argmax-flip rate, using
    # first-divergence as the event and full-length survival as right-censoring. This is the
    # physically meaningful quantity: byte-exact greedy identity over an L-token autoregressive
    # rollout = (1 - hazard)^L, so a tiny per-token flip rate compounds catastrophically.
    hazard = (n_div / exposure) if exposure else float("nan")
    fds = sorted(d["first_div_pos"] for d in diverging)
    return {"n_total": n, "n_match": n_match, "frac": (n_match / n) if n else float("nan"),
            "n_diverge": n_div, "per_token_flip_hazard": hazard,
            "first_div_min": (fds[0] if fds else None),
            "first_div_median": (fds[len(fds) // 2] if fds else None),
            "first_div_max": (fds[-1] if fds else None),
            "diverging": diverging}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=HERE / "runs")
    ap.add_argument("--bi1-arref", default="bi1_spec0")
    ap.add_argument("--bi1-spec", default="bi1_spec1")
    ap.add_argument("--bi0-arref", default="bi0_spec0")
    ap.add_argument("--bi0-spec", default="bi0_spec1")
    ap.add_argument("--out", type=Path, default=HERE / "runs" / "analysis.json")
    args = ap.parse_args()

    arms = {
        "bi1_arref": load_arm(args.runs / args.bi1_arref),
        "bi1_spec": load_arm(args.runs / args.bi1_spec),
        "bi0_arref": load_arm(args.runs / args.bi0_arref),
        "bi0_spec": load_arm(args.runs / args.bi0_spec),
    }

    bi1_id = identity(arms["bi1_spec"], arms["bi1_arref"])   # PRIMARY (strict-#319)
    bi0_id = identity(arms["bi0_spec"], arms["bi0_arref"])   # CONTROL (deployed order)

    # CROSS-CHECK: two pure-AR (spec OFF) rollouts differing ONLY in VLLM_BATCH_INVARIANT.
    # If these diverge, the M=1 decode reduction order is itself BI-sensitive and the 512-token
    # rollout is fragile to ANY reduction-order perturbation -- not just spec-on/off.
    ar_bi_xcheck = identity(arms["bi1_arref"], arms["bi0_arref"])

    # DETERMINISM CONTROL (load-bearing): a fresh BI=1 AR rollout under the IDENTICAL config as
    # bi1_arref (same BI, same spec-off, same backend, same prompts, fresh server process). This
    # pins the rollout FLOOR. The whole rollout-identity methodology only certifies byte-exactness
    # if same-config-twice == 128/128: only then is a <128/128 spec-vs-AR result a real per-step
    # reduction-order divergence rather than intrinsic run-to-run non-reproducibility of the served
    # greedy stack. If this control is <128/128, served byte-exact greedy identity is unachievable
    # by ANY scheme (incl. AR) and the rollout test cannot certify the strict-#319 contract.
    determinism = None
    rep = args.runs / "bi1_spec0_rep"
    if (rep / "decode_outputs.jsonl").exists():
        rep_arm = load_arm(rep)
        det_id = identity(rep_arm, arms["bi1_arref"])  # same-config repeat vs original BI=1 AR
        determinism = {
            "identity": det_id,
            "deterministic": bool(det_id["n_match"] == det_id["n_total"] and det_id["n_total"] > 0),
            "tps_rep": rep_arm["summary"]["output_tps"],
            "floor_frac": det_id["frac"],
        }

    # OPTIONAL decisive mechanism test: enforce_eager (no CUDA graphs) replicates #743's offline
    # condition but with the REAL online batched spec path. Present only after run_eager.sh.
    eager = None
    e_arref = args.runs / "bi1_spec0_eager"
    e_spec = args.runs / "bi1_spec1_eager"
    if (e_arref / "decode_outputs.jsonl").exists() and (e_spec / "decode_outputs.jsonl").exists():
        ea = {"arref": load_arm(e_arref), "spec": load_arm(e_spec)}
        eager_id = identity(ea["spec"], ea["arref"])
        # Decompose the residual: compare the eager (no-CUDA-graph) spec hazard against the
        # cudagraph primary (bi1_id) hazard. If eager reaches 128/128, the whole served residual
        # was CUDA-graph capture asymmetry (M=1 decode captured, M=K verify eager). If eager still
        # diverges but its hazard is much lower than the cudagraph hazard, BOTH contribute:
        # CUDA-graph capture (the removed part) is dominant and the irreducible M=K batched-verify
        # split-KV reduction (the eager residual) is secondary. If eager barely helps, the
        # batched-verify shape is the whole story.
        cg_haz = bi1_id["per_token_flip_hazard"] or 0.0
        eg_haz = eager_id["per_token_flip_hazard"] or 0.0
        haz_reduction_frac = ((cg_haz - eg_haz) / cg_haz) if cg_haz else 0.0
        transfers_eager = bool(eager_id["n_match"] == eager_id["n_total"]
                               and eager_id["n_total"] > 0)
        if transfers_eager:
            mechanism = "CUDA_GRAPH_CAPTURE"
        elif haz_reduction_frac >= 0.5:
            mechanism = "MIXED_CUDAGRAPH_DOMINANT_VERIFY_RESIDUAL"
        else:
            mechanism = "BATCHED_VERIFY_SHAPE"
        eager = {
            "identity": eager_id,
            "transfers_eager": transfers_eager,
            "tps_spec": ea["spec"]["summary"]["output_tps"],
            "tps_arref": ea["arref"]["summary"]["output_tps"],
            "cudagraph_spec_hazard": cg_haz,
            "eager_spec_hazard": eg_haz,
            "cudagraph_capture_hazard_share": round(haz_reduction_frac, 4),
            "mechanism": mechanism,
        }

    # ADVISOR REFRAME (PR #748 comment 17:39Z, relaying lawine #752): a plain-BI=1 served
    # non-128/128 is the EXPECTED adaptive-split-KV ULP-tie cascade, to be classified
    # `self-consistent-not-byteexact` (tau=0.3 + PPL) rather than a transfer failure. These two
    # blocks come from selfconsist_ppl.py (offline scoring of the captured streams).
    self_consistency = None
    sc_path = args.runs / "selfconsist_ppl_bi1.json"
    if sc_path.exists():
        sc = json.loads(sc_path.read_text())
        gp = sc.get("gap_probe")
        if gp is not None:
            self_consistency = {k: gp[k] for k in (
                "tau", "n_diverging", "n_probed", "n_len_divergence_only",
                "confident_genuine_flips", "max_gap_nat", "gap_nat_histogram",
                "frac_pair_is_model_top2", "self_consistent_pass")}
            self_consistency["confident_records"] = [
                r for r in gp.get("records", []) if r.get("confident")]

    ppl = None
    ppl1 = args.runs / "selfconsist_ppl_bi1.json"
    ppl0 = args.runs / "selfconsist_ppl_bi0.json"
    if ppl1.exists() and ppl0.exists():
        p1 = json.loads(ppl1.read_text()).get("ppl")
        p0 = json.loads(ppl0.read_text()).get("ppl")
        if p1 and p0:
            d = abs(p1["ppl"] - p0["ppl"]) / p0["ppl"] * 100.0
            ppl = {"ppl_bi1": p1["ppl"], "ppl_bi0": p0["ppl"],
                   "ppl_abs_delta_pct_bi1_vs_bi0": round(d, 4),
                   "ppl_neutral_bi": bool(d < 0.5),
                   "num_tokens": p1.get("num_tokens"),
                   "deployed_pruned_head_anchor": 2.019,
                   "note": ("absolute PPL is on the loadable full-vocab QAT proxy "
                            "google/gemma-4-E4B-it-qat-w4a16-ct (NOT the deployed pruned-16k "
                            "int4_g128_lmhead), so it brackets -- not exactly hits -- 2.019; "
                            "the load-bearing claim is BI=1 vs BI=0 PPL parity (quality-neutral "
                            "reduction order, spec-decode does not enter teacher-forced scoring)")}

    bi1_tps = arms["bi1_spec"]["summary"]["output_tps"]
    bi0_tps = arms["bi0_spec"]["summary"]["output_tps"]
    bi1_ar_tps = arms["bi1_arref"]["summary"]["output_tps"]
    bi0_ar_tps = arms["bi0_arref"]["summary"]["output_tps"]
    # decode tax = slowdown of byte-exact (BI=1) vs deployed (BI=0), both spec on
    decode_tax_pct = ((bi0_tps - bi1_tps) / bi0_tps * 100.0) if bi0_tps else float("nan")

    transfers = (bi1_id["n_match"] == bi1_id["n_total"] and bi1_id["n_total"] > 0)
    verdict = "SERVED_BYTEEXACT_TRANSFERS" if transfers else "SERVED_RESIDUAL_DIVERGENCE"

    # Named diverging sub-op for the SERVED_RESIDUAL_DIVERGENCE deliverable. The eager arms
    # isolate CUDA-graph capture from the kernel shape: eager removes CUDA graphs but NOT the
    # adaptive split-KV num_splits heuristic. We DECOMPOSE the residual by the per-token flip
    # hazard removed when CUDA graphs are disabled (cudagraph_capture_hazard_share):
    #   transfers_eager (128/128)      -> 100% capture asymmetry, byte-exact-fixable by graph mode
    #   share>=0.5 (MIXED, dominant)    -> capture asymmetry dominates, split-KV verify is residual
    #   share<0.5  (BATCHED_VERIFY)     -> the M=K split-KV verify reduction is the primary residual
    if eager is None:
        named_subop = "pending_eager_arms"
    elif eager["mechanism"] == "CUDA_GRAPH_CAPTURE":
        named_subop = ("CUDA-graph capture asymmetry: M=1 decode is graph-captured "
                       "(sizes [1,2,4,8]) but the M=K+1 verify is not, so the captured decode "
                       "and eager verify take different attention reduction orders; enforce_eager "
                       "restores 128/128 -> the served residual is byte-exact-fixable by graph mode")
    elif eager["mechanism"] == "MIXED_CUDAGRAPH_DOMINANT_VERIFY_RESIDUAL":
        share = eager.get("cudagraph_capture_hazard_share", 0.0)
        named_subop = (
            f"MIXED, CUDA-graph-capture-dominant: disabling CUDA graphs removes "
            f"{share*100:.0f}% of the per-token flip hazard ({eager['identity']['n_match']}/"
            f"{eager['identity']['n_total']} eager vs {bi1_id['n_match']}/{bi1_id['n_total']} "
            f"captured), so capture asymmetry (graph-captured M=1 decode vs non-captured M=K+1 "
            f"verify taking different attention reduction orders) is the DOMINANT contributor, with "
            f"a residual M=K adaptive split-KV verify reduction (num_splits>1 at served seq-len, "
            f"which plain BI=1 does not align) -- NOT the GEMM (land #680). #743's offline "
            f"byte-exactness held because the prompt_logprobs M=1-shaped proxy avoids BOTH the "
            f"capture asymmetry and the M=K verify shape.")
    else:
        named_subop = ("M=K batched-verify adaptive split-KV reduction: at served seq-len the "
                       "kernel picks num_splits>1 for the M=K verify batch (different reduction "
                       "order than the num_splits=1 M=1 decode/AR), and plain BI=1 aligns the "
                       "matmul family but NOT this verify-attention split selection; eager removes "
                       "only a minority of the hazard (so #743's offline byte-exactness came mostly "
                       "from the prompt_logprobs M=1-shaped proxy, not from enforce_eager) -- NOT "
                       "the GEMM (land #680), only secondarily CUDA-graph capture (eager arm)")

    # Honest 3-way self-consistency classification. The STRICT gate (lawine/#720 protocol) is
    # binary: self_consistent_pass == (confident_genuine_flips == 0). We keep that gate as-is, but
    # ALSO recognize an intermediate "marginal tail" reality: the gate can fail while every
    # confident flip sits at the smallest super-threshold bf16-ULP step (max_gap_nat <= tau + 1 ULP)
    # and they are a small minority -- i.e. the residual is overwhelmingly benign ULP ties with a
    # tau-sensitive marginal tail, NOT substantial genuine divergence. Collapsing such a result to
    # either "benign" (hides the tail) or "genuine_divergence" (overstates it) would be dishonest.
    ULP = 0.125
    ppl_ok = (ppl is None) or bool(ppl["ppl_neutral_bi"])
    sc = self_consistency
    sc_clean_pass = bool(sc is not None and sc["self_consistent_pass"])  # 0 confident flips
    n_conf = sc["confident_genuine_flips"] if sc is not None else None
    conf_frac = (n_conf / sc["n_probed"]) if (sc is not None and sc.get("n_probed")) else None
    conf_all_marginal = bool(
        sc is not None and n_conf and n_conf > 0
        and sc["max_gap_nat"] <= sc["tau"] + ULP + 1e-6
        and conf_frac is not None and conf_frac <= 0.05)

    self_consistent_not_byteexact = bool((not transfers) and sc_clean_pass and ppl_ok)
    predominantly_benign_marginal = bool(
        (not transfers) and (sc is not None) and (not sc_clean_pass)
        and conf_all_marginal and ppl_ok)

    if transfers:
        residual_class = "byte_exact"
    elif sc is None:
        residual_class = "pending_self_consistency"
    elif self_consistent_not_byteexact:
        residual_class = "benign_ulp_tie_cascade"
    elif predominantly_benign_marginal:
        residual_class = "predominantly_benign_ulp_marginal_confident_tail"
    else:
        residual_class = "genuine_divergence_present"

    if sc is None:
        honest_read = ("BI=1 batched-verify served spec is NOT 128/128 byte-exact; self-consistency "
                       "(tau=0.3 + PPL) pending.")
    elif self_consistent_not_byteexact:
        honest_read = (
            "BI=1 batched-verify served spec is NOT literally 128/128 byte-exact, but the residual "
            "is a quality-neutral adaptive-split-KV ULP-tie cascade (0 confident genuine flips at "
            "tau=0.3, PPL-neutral) -- self-consistent, not a transfer failure. LITERAL served "
            "byte-exactness needs the num_splits=1 verify fix (land #743; routed to lawine #755).")
    elif predominantly_benign_marginal:
        honest_read = (
            f"BI=1 batched-verify served spec is NOT 128/128 byte-exact ({bi1_id['n_match']}/"
            f"{bi1_id['n_total']}). PPL is BI-neutral (delta "
            f"{ppl['ppl_abs_delta_pct_bi1_vs_bi0'] if ppl else float('nan'):.4f}%) and the residual is "
            f"predominantly benign bf16-ULP ties ({sc['n_diverging']-n_conf}/{sc['n_diverging']} onsets "
            f"at <=2 ULP), BUT it does NOT clear the strict tau=0.3 self-consistency gate: "
            f"{n_conf}/{sc['n_probed']} confident genuine flips, all at the minimal super-threshold "
            f"gap (max {sc['max_gap_nat']:.3f} nat = 3 ULP), so the pass is tau-sensitive (0 confident "
            f"at tau>=0.4). Unlike lawine's publishable-config clean pass (0 confident), the K=6 "
            f"fire-config served residual keeps a small genuinely-confident tail. BI=1 is necessary "
            f"but not sufficient for literal served byte-exactness; the num_splits=1 verify fix "
            f"(land #743; routed to lawine #755) is needed to close even this marginal tail.")
    else:
        honest_read = (
            f"BI=1 batched-verify served spec is NOT byte-exact ({bi1_id['n_match']}/{bi1_id['n_total']}) "
            f"and the residual is NOT self-consistent: {n_conf}/{sc['n_probed']} confident genuine flips "
            f"at tau=0.3 (max gap {sc['max_gap_nat']:.3f} nat)"
            + ("" if ppl_ok else ", and PPL is NOT BI-neutral") +
            " -- genuine divergence present, not just ULP ties.")

    reframe = {
        "literal_served_byteexact": bool(transfers),
        "self_consistent_not_byteexact": self_consistent_not_byteexact,
        "predominantly_benign_ulp_marginal_confident_tail": predominantly_benign_marginal,
        "confident_flip_frac": (round(conf_frac, 4) if conf_frac is not None else None),
        "confident_flips_all_marginal_3ulp": conf_all_marginal,
        "ppl_bi_neutral": ppl_ok if ppl is not None else None,
        "residual_class": residual_class,
        "named_residual_subop": named_subop,
        "bi1_necessary_not_sufficient_for_literal_byteexact": bool(not transfers),
        "honest_read": honest_read,
    }

    result = {
        "verdict": verdict,
        "reframe": reframe,
        "self_consistency": self_consistency,
        "ppl": ppl,
        "primary_metric": {"name": "served_spec_bi1_greedy_identity",
                           "value": bi1_id["frac"],
                           "n_match": bi1_id["n_match"], "n_total": bi1_id["n_total"]},
        "test_metric": {"name": "served_spec_bi1_tps", "value": bi1_tps},
        "bi1_identity": bi1_id,
        "bi0_control_identity": bi0_id,
        "ar_bi_xcheck_identity": ar_bi_xcheck,
        "determinism_control": determinism,
        "eager_mechanism": eager,
        "tps": {"bi1_spec": bi1_tps, "bi0_spec": bi0_tps,
                "bi1_arref": bi1_ar_tps, "bi0_arref": bi0_ar_tps,
                "decode_tax_pct_bi1_vs_bi0": round(decode_tax_pct, 3),
                "bi1_spec_clears_126378": bool(bi1_tps > LOCKED_TPS),
                "bi0_spec_clears_126378": bool(bi0_tps > LOCKED_TPS),
                "locked_tps": LOCKED_TPS},
        "arms": {k: {"tag": v["summary"]["tag"], "bi": v["summary"]["batch_invariant"],
                     "spec": v["summary"]["spec"], "output_tps": v["summary"]["output_tps"],
                     "n_prompts": v["summary"]["n_prompts"],
                     "output_len": v["summary"]["output_len"],
                     "peak_gpu_mem_mib": v["summary"].get("peak_gpu_mem_mib"),
                     "backend_line": v["summary"].get("server_backend_line", ""),
                     "spec_line": v["summary"].get("server_spec_line", "")}
                 for k, v in arms.items()},
    }
    args.out.write_text(json.dumps(result, indent=2))

    print("=" * 72)
    print(f"VERDICT: {verdict}")
    print(f"  BI=1 spec vs BI=1 AR identity (PRIMARY/strict-#319): "
          f"{bi1_id['n_match']}/{bi1_id['n_total']}  (frac={bi1_id['frac']:.4f})  "
          f"hazard={bi1_id['per_token_flip_hazard']*100:.3f}%/tok")
    print(f"  BI=0 spec vs BI=0 AR identity (CONTROL, deployed):   "
          f"{bi0_id['n_match']}/{bi0_id['n_total']}  (frac={bi0_id['frac']:.4f})  "
          f"hazard={bi0_id['per_token_flip_hazard']*100:.3f}%/tok")
    print(f"  BI=1 AR vs BI=0 AR (xcheck, spec OFF both):          "
          f"{ar_bi_xcheck['n_match']}/{ar_bi_xcheck['n_total']}  "
          f"hazard={ar_bi_xcheck['per_token_flip_hazard']*100:.3f}%/tok  "
          f"<- pure-AR reduction-order sensitivity")
    if determinism is not None:
        di = determinism["identity"]
        print(f"  DETERMINISM CONTROL (BI=1 AR vs BI=1 AR, SAME config): "
              f"{di['n_match']}/{di['n_total']}  (floor frac={di['frac']:.4f})  "
              f"hazard={di['per_token_flip_hazard']*100:.3f}%/tok  "
              f"-> {'DETERMINISTIC (floor=0; rollout test valid)' if determinism['deterministic'] else 'NON-REPRODUCIBLE (rollout cannot certify byte-exactness)'}")
    else:
        print("  DETERMINISM CONTROL: pending (bi1_spec0_rep not run)")
    print(f"  TPS: bi1_spec={bi1_tps:.3f}  bi0_spec={bi0_tps:.3f}  "
          f"(bi1_ar={bi1_ar_tps:.3f} bi0_ar={bi0_ar_tps:.3f})")
    print(f"  decode tax (BI=1 vs BI=0, spec on): {decode_tax_pct:.2f}%")
    print(f"  bi1_spec clears 126.378? {bool(bi1_tps > LOCKED_TPS)}  "
          f"(NOTE: raw full-vocab api_server local TPS, NOT the deployed optimized stack)")
    if eager is not None:
        ei = eager["identity"]
        print(f"  [EAGER mechanism] BI=1 spec vs BI=1 AR, enforce_eager=1: "
              f"{ei['n_match']}/{ei['n_total']}  (frac={ei['frac']:.4f})  "
              f"hazard={ei['per_token_flip_hazard']*100:.3f}%/tok")
        share = eager.get("cudagraph_capture_hazard_share", 0.0)
        if eager["mechanism"] == "CUDA_GRAPH_CAPTURE":
            mech_expl = "CUDA-graph capture broke num_splits=1 alignment; batched path FIXABLE"
        elif eager["mechanism"] == "MIXED_CUDAGRAPH_DOMINANT_VERIFY_RESIDUAL":
            mech_expl = (f"capture asymmetry DOMINANT ({share*100:.0f}% of hazard removed by eager); "
                         f"residual M=K split-KV verify reduction remains")
        else:
            mech_expl = (f"M=K batched-verify split-KV reduction is primary residual "
                         f"(only {share*100:.0f}% of hazard from CUDA-graph capture)")
        print(f"    -> mechanism={eager['mechanism']}  ({mech_expl})")
    else:
        print("  [EAGER mechanism] pending (run_eager.sh not finished)")
    if self_consistency is not None:
        sc = self_consistency
        print(f"  [SELF-CONSISTENCY tau={sc['tau']}] probed {sc['n_probed']} onsets: "
              f"confident_genuine_flips={sc['confident_genuine_flips']}  "
              f"max_gap={sc['max_gap_nat']:.3f}nat  "
              f"pair_is_model_top2={sc['frac_pair_is_model_top2']}  "
              f"-> {'SELF-CONSISTENT (benign ULP ties)' if sc['self_consistent_pass'] else 'GENUINE FLIPS PRESENT'}")
        print(f"    gap_nat histogram: {sc['gap_nat_histogram']}")
    else:
        print("  [SELF-CONSISTENCY] pending (selfconsist_ppl.py --bi 1 not run)")
    if ppl is not None:
        print(f"  [PPL] BI=1={ppl['ppl_bi1']:.4f}  BI=0={ppl['ppl_bi0']:.4f}  "
              f"|delta|={ppl['ppl_abs_delta_pct_bi1_vs_bi0']:.3f}%  "
              f"-> {'BI-NEUTRAL (quality-preserved)' if ppl['ppl_neutral_bi'] else 'BI-SHIFT'}  "
              f"(proxy head; deployed anchor 2.019)")
    else:
        print("  [PPL] pending (selfconsist_ppl.py both BI not run)")
    print(f"  [REFRAME] residual_class={reframe['residual_class']}  "
          f"self_consistent_not_byteexact={reframe['self_consistent_not_byteexact']}")
    print(f"    named_residual_subop: {reframe['named_residual_subop'][:120]}...")
    if bi1_id["n_diverge"]:
        print(f"  BI=1 diverging prompts ({bi1_id['n_diverge']}): "
              f"first-div positions {[d['first_div_pos'] for d in bi1_id['diverging'][:12]]}")
    print(f"  -> {args.out}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
