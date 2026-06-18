STUDENT kanna: heartbeat — run self-healed (NOT a real crash), sampled **n=3 landed**, accumulating toward n=5 floor / n=10. NOT terminal.

## The "crash" was the W&B heartbeat lapse I flagged, not a process death
Confirmed on the ground: no traceback in any log (`grep traceback|oom|cuda error` empty), server PID 320465 stayed up the whole time, GPU 20.1/23 GB (peak this PR 19.62) — **no OOM, no CUDA error, no preemption**. The `crashed` state on `cr3c4y3q` was the per-session W&B heartbeat going quiet at the `greedy_s34567`→`sampled_s34567` subprocess-fork (10:18Z), exactly as diagnosed. The on-disk result JSONs + `gate_summary.json` were complete and uncontaminated throughout. My resume-by-id design **self-healed** it: the seed-45678 invocation re-attached and revived the run to `running`. So **no relaunch was needed** — `sampled_s34567` finished clean at 11:05Z.

## Panel state (sampled n=3, KNIFE_EDGE)
- **greedy n=3 DONE** (capped per your steer): mean **0.5034**, seeds [0.5202, 0.5202, 0.4697], CI [0.431, 0.576] — clears the 0.4864 bar. finish_length@4096 **2.5%** → implied-3072 **11.8%** (≈#631's 13%) → **cap-artifact confirmed, gate point healthy**.
- **sampled n=3 DONE**: mean **0.4714**, seeds [0.4899, 0.4747, 0.4495], 95% CI **[0.4207, 0.5221]**, **pct-of-base 87.2%**, pct-of-AR-body(0.499) 94.5%. seed 34567 came in low (0.4495), pulling the point estimate just **below** the bar but the CI still **STRADDLES** 0.4864 → `READING_A_GPQA_KNIFE_EDGE`. CI width is the deliverable, exactly as you called — accumulating more seeds.

## Server decision: kept the proven-healthy server (no restart)
I'd floated a fresh-engine restart, but the OOM concern is moot (verified no OOM/crash above), and a restart wouldn't safely buy two-seeds-per-invocation anyway: at current speed two sampled seeds ≈94 min would breach the hard 90-min/run bound, and even a fresh server (~89 min for two) is too tight. So the safe, clean pattern is **one sampled seed per detached resumable invocation** on the one continuous, verified-correct server (int4_g128_lmhead + K=7 + BI=1 + dev307 + model_len 8192, `expandable_segments` on). I'll restart only if memory creeps (>~22 GB) or a seed runs >75 min.

## Now running
seed **45678** sampled (W&B `cr3c4y3q`, one eval client confirmed, conc=1), started 11:10Z, ETA ~11:57Z. Then 56789… accumulating sampled toward **n=5 floor → n=10** (one clean cell per wakeup), so the Option-B number is CI-comparable to ubel #638 (0.4990) / lawine #639 (0.5056). Terminal `SENPAI-RESULT` once the sampled CI resolves the knife-edge or I reach n=10. Holding `status:wip`; no blocking question.
