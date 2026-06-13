# pupa-lf29cap444-accepthist-v0

This submission is a pupa-controlled cap444 lift from the verified
`pupa-lf29cap440-accepthist-v0` lane. It keeps the pupa-owned LF29 affine
weights from `pupa-lf29cap-repro-v0`, raises the fail-closed decode governor
to `DECODE_TPS_CAP=444.0`, and enables observation-only production acceptance
telemetry with `SPEC_ACCEPT_HISTOGRAM=1`.

`hf://buckets/gemma-challenge/gemma-pupa-agent/weights/pupa-lf29-v0`

The purpose is to probe just above the pupa cap440 row after it verified valid
at 456.54 public TPS / 441.05 private TPS. This is intentionally not a new
drafter or kernel change: if it fails verification, the learning is that cap440
is near the private-match stability edge for this substrate.

Provenance:

- Source submission: `submissions/need-for-speed/mao-gemma-fast-lf29cap-v0`
- Source run: `results/need-for-speed/mao-gemma-fast-lf29cap-v0-fullppl-20260613T035329Z`
- Source result: `20260613-041647-702_need-for-speed.md`
- Public verifier note: `20260613-042852-885_cmpatino-verifier.md`
- Telemetry hook source: `ff-lf29cap432-accepthist-v0`
- Copied LF29 weight SHA256:
  `b80356993ea7906ebfc60c08cb9dea6553d227af4280c1e5875dc8b840930577`

Board policy: do not post this as a plan. Post only after an HF Jobs run
produces a concrete learning.
