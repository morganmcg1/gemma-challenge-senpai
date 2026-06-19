#!/usr/bin/env python3
"""PR #696 -- quantify the BOOT/VENV measurement-noise floor on GPQA-D #31-SAMPLED decode.

Why this matters for the verdict: the int4-body GPQA-D #31 point sits in a marginal-tie zone
around the 0.471 gate. The pooled-Wilson / seed-mean CIs treat decode-seed sampling as the ONLY
noise source. But seed 0 (a FIXED sampling seed) scored 102 (banked land-inspect boot), 100
(repro, eval-serve boot-B), and 90 (renew09 eval-serve boot-C) -- a ~12-correct (6pp) swing for
an IDENTICAL seed. That across-boot variance is invisible to every single-pool CI, yet it is
comparable to the gate margin. This script measures it directly from the paired 0-9 design:

  banked_0to9  = results_gpqa/_banked_landinspect/bf_gpqa_sampled_mt8_s{0..9}.json (Boot-A, land-inspect venv)
  renew09_0to9 = results_gpqa/bf_gpqa_sampled_mt8_s{0..9}.json                     (Boot-C, eval-serve venv)

Same fixed sampling seeds, byte-identical serve recipe (dev307 build, same int4 g32 body +
bf16 head, same surgical-attn batch-invariant forcing) -> the ONLY differences are server boot
and eval-client venv. The paired per-seed delta isolates that boot+venv noise.

Outputs the paired mean shift, paired t, the implied per-seed boot/venv SD, and a boot-stratified
view of the homogeneous 30-seed pool (0-9 renew09 boot vs 10-29 earlier boot) so the verdict can
state plainly how much of the gate-straddle is irreducible measurement noise.
LOCAL, analysis_only, NO FIRE.
"""
from __future__ import annotations

import glob
import json
import math
import statistics as st
from pathlib import Path

HERE = Path("/workspace/senpai/target/research/validity/int4_quality_31basis_confirm")
RES = HERE / "results_gpqa"
BANKED = RES / "_banked_landinspect"
GATE = 0.471
BASE = 0.5236
Z = 1.959963984540054
TCRIT = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306,
         10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145}


def load_seed(d: Path, s: int):
    f = d / f"bf_gpqa_sampled_mt8_s{s}.json"
    if not f.exists():
        return None
    j = json.load(open(f))
    return {"seed": s, "acc": j["accuracy"], "k": j["n_correct"], "n": j["n_scored"]}


def main():
    pairs = []
    for s in range(10):
        a = load_seed(BANKED, s)     # Boot-A land-inspect
        c = load_seed(RES, s)        # Boot-C renew09
        if a and c:
            pairs.append((s, a, c))
    if not pairs:
        raise SystemExit("no paired 0-9 seeds yet")

    deltas = [c["acc"] - a["acc"] for _, a, c in pairs]      # renew09 - banked
    dk = [c["k"] - a["k"] for _, a, c in pairs]
    npair = len(pairs)
    mean_d = st.mean(deltas)
    sd_d = st.stdev(deltas) if npair > 1 else 0.0
    se_d = sd_d / math.sqrt(npair) if npair > 1 else float("nan")
    t_paired = mean_d / se_d if se_d else float("nan")
    # per-seed boot+venv SD: paired-delta variance = 2 * boot_var (if both boots iid noise) ->
    # boot SD per single measurement = sd_d / sqrt(2). Conservative single-side estimate.
    boot_sd_single = sd_d / math.sqrt(2)

    banked_accs = [a["acc"] for _, a, _ in pairs]
    renew_accs = [c["acc"] for _, _, c in pairs]

    # boot-stratified homogeneous pool: renew09 0-9 vs the 10-29 block
    block_1029 = []
    for f in sorted(glob.glob(str(RES / "bf_gpqa_sampled_mt8_s*.json")),
                    key=lambda p: int(Path(p).stem.split("_s")[-1])):
        s = int(Path(f).stem.split("_s")[-1])
        if s >= 10:
            block_1029.append(json.load(open(f))["accuracy"])

    out = {
        "pr": 696, "analysis_only": True, "no_hf_job": True, "fires": 0,
        "design": "paired same-seed, byte-identical serve recipe; differ only by boot+eval-venv",
        "n_pairs": npair,
        "banked_landinspect_bootA": {
            "per_seed_acc": banked_accs, "mean": st.mean(banked_accs),
            "std": st.stdev(banked_accs) if npair > 1 else 0.0,
            "pct_of_base": 100 * st.mean(banked_accs) / BASE},
        "renew09_bootC": {
            "per_seed_acc": renew_accs, "mean": st.mean(renew_accs),
            "std": st.stdev(renew_accs) if npair > 1 else 0.0,
            "pct_of_base": 100 * st.mean(renew_accs) / BASE},
        "paired_delta_renew_minus_banked": {
            "per_seed_delta_acc": deltas, "per_seed_delta_correct": dk,
            "mean_delta_acc": mean_d, "mean_delta_correct": st.mean(dk),
            "sd_delta_acc": sd_d, "se_delta_acc": se_d, "t_paired": t_paired,
            "tcrit_95": TCRIT.get(npair, 1.96),
            "significant_95": bool(abs(t_paired) > TCRIT.get(npair, 1.96)) if se_d else None},
        "implied_per_seed_boot_venv_sd": boot_sd_single,
        "implied_per_seed_boot_venv_sd_pp": 100 * boot_sd_single,
        "interpretation": (
            "paired-delta SD / sqrt(2) = per-single-measurement boot+venv SD; compare to the "
            "decode-seed SD (~0.0186) -- if comparable, the single-pool Wilson UNDERSTATES the "
            "true gate uncertainty because it omits this across-boot component."),
        "boot_stratified_homogeneous_pool": {
            "block_0to9_renew09_mean": st.mean(renew_accs),
            "block_10to29_mean": st.mean(block_1029) if block_1029 else None,
            "block_10to29_n": len(block_1029),
            "across_block_gap_pp": (100 * (st.mean(block_1029) - st.mean(renew_accs)))
            if block_1029 else None,
        },
        "gate": GATE, "base_sampled_3seed": BASE,
    }
    (HERE / "boot_variance.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
