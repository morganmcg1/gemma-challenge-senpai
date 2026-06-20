"""Log the PR #821 FR-Spec mechanism-refutation finding to W&B for the record.

Not a benchmark run: there is no slice/TPS experiment because the hypothesis is
refuted at the mechanism level (the draft head is already centroid-masked-sparse,
so there is no dense full-vocab GEMV to shrink). This run records the decisive
facts + arithmetic so the finding is searchable alongside the int4head oracle.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.wandb_logging import init_wandb_run, log_summary, finish_wandb

DRAFT_HIDDEN = 256
BYTES_BF16 = 2
NUM_CENTROIDS = 2048
CTK = 32  # centroid_intermediate_top_k (checkpoint default the int4head submission runs)
VOCAB = 262144
ACTIVE = CTK * (VOCAB // NUM_CENTROIDS)  # 32 * 128 = 4096

lmhead_gather_mib = ACTIVE * DRAFT_HIDDEN * BYTES_BF16 / 2**20
centroid_w_mib = NUM_CENTROIDS * DRAFT_HIDDEN * BYTES_BF16 / 2**20
draft_head_step_mib = lmhead_gather_mib + centroid_w_mib
full_resident_mib = VOCAB * DRAFT_HIDDEN * BYTES_BF16 / 2**20

config = {
    "pr": 821,
    "hypothesis": "FR-Spec frequency-pruned draft lm_head (arXiv:2502.14856 / 2506.22694)",
    "verdict": "REFUTED_at_mechanism_level",
    "drafter": "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant",
    "use_ordered_embeddings": True,
    "num_centroids": NUM_CENTROIDS,
    "centroid_intermediate_top_k": CTK,
    "vocab_size": VOCAB,
    "draft_hidden_size": DRAFT_HIDDEN,
    "num_speculative_tokens": 6,
    "baseline_int4head_tps": 256.74,
    "baseline_int4head_e_accept": 3.379,
}

summary = {
    "active_tokens_per_draft_step": ACTIVE,             # 4096, NOT 262144
    "active_fraction_of_vocab": ACTIVE / VOCAB,          # 1.5625%
    "draft_head_read_mib_per_step": round(draft_head_step_mib, 3),     # ~3.0 MiB
    "draft_head_read_mib_per_cycle_k6": round(draft_head_step_mib * 6, 3),  # ~18 MiB
    "verifier_int4_lmhead_read_mib_per_accept": 378.0,  # 0.378 GB (stark #798)
    "draft_vs_verifier_head_ratio": round(draft_head_step_mib / 378.0, 5),
    "full_resident_lmhead_mib": round(full_resident_mib, 1),  # ~128 MiB (only this shrinks under a slice)
    "static_freq_slice_per_step_read_saving_mib": 0.0,  # slicing resident weight does NOT cut per-step reads
    # public leaderboard: the REAL draft-vocab lever is CENTROID_TOP_K (ctk), not freq slice
    "public_ctk42_tps": 513.766,
    "public_ctk44_48_band_tps": 505.0,
    "public_ctk52_tps_negative": 492.28,
    "int4head_runs_ctk": CTK,  # 32, BELOW the public 42-48 sweet spot
    "model_load_peak_gib": 9.86,
}

run = init_wandb_run(
    job_type="mechanism-analysis",
    agent="lawine",
    name="lawine/frspec-draft-vocab-refuted",
    group="bi0-frspec-draft-vocab",
    project="gemma-challenge-senpai",
    entity="wandb-applied-ai-team",
    tags=["pr821", "frspec", "drafter", "centroid", "refuted", "negative"],
    notes=(
        "PR #821 FR-Spec REFUTED: gemma4_assistant drafter already does learned "
        "centroid-masked sparse logits (active_tokens=4096/262144 at ctk=32). No "
        "dense full-vocab GEMV exists to shrink; single-stream decode is per-step "
        "read-bytes-bound (already ~4096 rows), not resident-capacity-bound; a "
        "static freq slice would also break masked_embedding.token_ordering. Real "
        "draft-vocab lever is CENTROID_TOP_K (public leaderboard meta)."
    ),
    config=config,
)
if run is None:
    print("WANDB run not initialized (no api key / disabled).")
    sys.exit(1)
log_summary(run, summary, step=0)
print("WANDB_RUN_ID", run.id)
print("WANDB_RUN_URL", run.url)
finish_wandb(run)
