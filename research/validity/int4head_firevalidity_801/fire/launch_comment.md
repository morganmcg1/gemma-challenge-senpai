STUDENT land:

FIRE acknowledged and EXECUTED — the int4head HF submission is **LAUNCHED and RUNNING** (the one human-approved fire: #800 "Human: fire asap" 17:29Z + "fire when ready" 16:31Z; advisor FIRE GO 17:30Z).

## HF job (live)
- **URL:** https://huggingface.co/jobs/gemma-challenge/6a36d5de3093dba73ce2b016
- **job_id:** `6a36d5de3093dba73ce2b016`
- **run_prefix:** `results/senpai/int4-mtp-bi0-int4head-20260620T180305Z`
- **submission_prefix:** `submissions/senpai/int4-mtp-bi0-int4head`
- **launched_at:** 2026-06-20T18:03:10Z · **status:** RUNNING · **timeout:** 40 min (expected return ≈ 18:43Z)
- **W&B launch run:** `ftds7gll` (`senpai-int4-mtp-bi0-int4head`)

## Pre-launch guards — all GREEN
- **Manifest Hub-pointed** ✅ — `model_id` AND `env.MODEL_ID` = `gemma-challenge/gemma-4-e4b-it-int4-mtp-bi0-int4head`.
  - ⚠️ **Honest disclosure / why this took an extra step:** on this invocation my PR branch had reverted to the **STALE** local-path manifest (`model_id=/workspace/gemma_build/bi0_int4head_g32`, `env.MODEL_ID` absent) — a fresh pod re-checkout of the PR branch (`git checkout -B … FETCH_HEAD`) had discarded the local merge I reported at 17:46Z. I re-ran `git merge origin/approval-gated-8gpu-20260613` (clean; merge commit `ad392c7`). Verified the **only** submission-dir change is the manifest (serve.py/sitecustomize.py/patches byte-unchanged → gate evidence stays valid), launch-critical shared scripts unchanged, no conflicts. A stale-path fire would also have been hard-blocked by upload's `_model_reference_errors`, so the guard is real, not assumed.
- **official_gate = PASS** ✅ — full-protocol (128-prompt) within-job serve-verify on **this** pod (`research/_localrun/validate-int4_mtp_bi0_int4head-20260620T174513Z/evidence.json`): PPL **2.00256** ≤ 2.42, completed **128/128**, all-4-modalities (text/image/audio/video). I dry-ran `enforce_launch_gate("int4-mtp-bi0-int4head")` read-only **before** the POST → `LAUNCH-ELIGIBLE`. (`greedy_verdict=NO_REFERENCE` is non-gating — not an input to `official_gate_verdict`, per #784.)
- **GPU clean** (0 MiB, 0% util — validation released it; no fleet leak #780); **HF_TOKEN present**; upload synced **6 code files** only (model pulled from the Hub repo, not re-uploaded — submission dir is 48K).

## Command
```
python train.py --submission submissions/int4_mtp_bi0_int4head --method "land/int4head-fire" \
  --run-prefix results/senpai/int4-mtp-bi0-int4head-20260620T180305Z --launch --wait \
  --timeout-s 3000 --interval-s 30
```

## Expected
~255 public TPS (clears the 250 target), PPL ~2.003, 128/128, all-4-modalities, prometheus guard active. Baseline to beat: shipped bi0 official **218.02** TPS (W&B `s63tb03x`).

I'll post the **terminal SENPAI-RESULT** with the official `summary.json:tps`, `ppl_summary.json` PPL, completed count, and modality coverage the moment the job returns.
