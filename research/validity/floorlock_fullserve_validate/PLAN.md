# Floor-lock full-serve validation (PR #485)

**Goal:** confirm the strict M=1 AR int4 floor-lock realizes its modeled **161.70
official TPS** on a real end-to-end serve, and is genuinely **literal-1.0**
(token-identical to plain greedy AR). This de-risks the "ship TODAY, zero-risk"
fallback (option A on issue #474) before the human may select it.

**Scope:** LOCAL only. `analysis_only=true`, `official_tps=0`. No `--launch`, no
draw, no leaderboard submission.

## Submission identity (resolved)

The PR names `submissions/fa2sw_strict_m1ar_int4`, which does **not** exist on the
advisor branch. The documented byte-equivalent that does exist is
`submissions/fa2sw_nonspec_int4` — manifest description: *"byte-identical to
fa2sw_precache_kenyan EXCEPT SPECULATIVE_CONFIG is blanked, disabling the MTP
drafter (K_spec 7->0) so every decode step is plain int4 M=1 AR. int4 M=1 AR ==
plain greedy AR by construction."* That is exactly the floor-lock config
(161.70, lawine #438). Staging `fa2sw_strict_m1ar_int4` as a clean,
leaderboard-grade copy makes the floor-lock draw-ready and gives the validation
its named artifact.

## Measurement path (official summary.json:tps)

`summary.json:tps = output_throughput` from `sglang.bench_serving` (vllm-chat
backend, 128 prompts, output_len 512, max_concurrency 1, request_rate inf,
warmup 4, seed 1, ignore_eos) — exactly what `hf_bucket_single_job.py` runs in
the official HF Job. Run it LOCALLY against the served floor-lock endpoint.

Local AWS A10G runs ~6% slower than official a10g-small (deployed anchor:
481.53 official / 454.2 local => multiplier ~1.0602, #99). 161.70 is on the
**official** scale (lawine #438), so:
- expected LOCAL wall_tps ~= 161.70 / 1.0602 ~= 152.5
- report measured local tps AND projected official (x1.0602); compare projected
  to 161.70 +/- sigma_hw (4.864).

## Plan
1. Stage `submissions/fa2sw_strict_m1ar_int4` (faithful copy of nonspec_int4).
2. Serve locally (server venv = vLLM 0.22.1rc1.dev307; weights /tmp/osoi5-*-baked).
3. Official benchmark: `sglang.bench_serving` -> summary.json (tps, completed).
4. Decode capture + greedy gate vs int4 M=1 AR reference -> token_identity_rate.
5. PPL (<=2.42) + modalities-loaded.
6. Verdicts: `floorlock_realizes_16170`, `floorlock_literal_1p0`. Log to W&B
   group `floorlock-fullserve-validate`.
