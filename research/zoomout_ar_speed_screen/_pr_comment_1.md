STUDENT wirbel:

## Intake: prior ZOOM-OUT state (#498) + chosen lever

Per instruction 1, reporting what **denken #498** found before I pick a lever, plus the on-branch corroboration that bounds my choice.

### What #498 found (SGLang + FlashInfer M=1 determinism diagnostic, `djwaqs7o`)

- **Did SGLang serve the int4 weights at all?** **No.** `sglang_decode_tps = None` — SGLang never stood up. It is **uninstallable in the serve env**: the challenge's `0.22.1rc1` pin is the *vLLM* version, not SGLang; the resolved SGLang stack is 164 packages / 4.21 GB of heavy wheels that **downgrade serve-torch** (`2.11.0+cu130 → 2.9.1+cu12`) and pull the full `nvidia-cu12` stack (pod free disk was 3.33 GB). lawine **#558** re-confirmed this on a fresh env measurement (`framework-zoomout-ceiling`, `rdbnjwxz`).
- **Byte-exact?** **No.** Where determinism *was* measured (the FlashInfer-proxy census), default dynamic split-KV is **occupancy-variant**: worst-case M=1-vs-M=8 byte identity **0.000**, first breaks at **KV ≥ 1536** (inside the operative decode band). Byte-exactness only returns with `disable_split_kv`/`fixed_split_size`.
- **Faster or slower than vLLM?** The byte-exact FlashInfer path is the **slower** M=1 path (fixed-split costs **1.2–4.7×** at M=1 per fern #507; cross-check `invariance_cost_ratio_m1 ≈ 2.65`). So SGLang/FlashInfer offers **neither** free byte-exactness **nor** a speed win.
- **Headline conclusion:** the **−107 byte-exact attention tax is ENGINE-INDEPENDENT** — it is an IEEE-754 reduction-order property of the math, not a vLLM artifact. vLLM is byte-exact here *because* it **forces** a Triton-with-sliding-window path for the heterogeneous head_dim (256 local / 512 global) — a path SGLang's Triton fallback lacks (SGLang #6161) and FlashInfer flips.

### Corroboration already on-branch (the whole alt-engine menu is closed)

| candidate | PR | byte-exact M=1 serve of THIS w4a16 checkpoint? |
|---|---|---|
| SGLang | #498 / #558 | **No** — uninstallable in-box; FlashInfer backend flips identity; Triton lacks sliding-window |
| TRT-LLM | #502 | **No** — engine never builds (gemma4 skew, head_dim-512 > Ampere FMHA cap, PLE+KV-share not expressible) |
| FlashInfer (standalone) | #507 | **No** — loads but not *free* byte-exact; fixed-split 1.2–4.7× slower at M=1; hd512 no path |
| llama.cpp (public taskforce) | — | **No** — GGUF q4_0 is a *different* quant (≠ byte-identical); GEMV loses to Marlin |

Both ceiling terms are **hardware walls**: head-GEMV BW **482.9 GB/s = 80.5% of the 600 GB/s A10G peak** (Marlin is the only w4a16 kernel on sm_86, #550) + a **0.573 ms** fixed launch floor (#554), both engine-independent.

### Chosen lever + why

SGLang (#498/#558), TRT-LLM (#502), and FlashInfer-standalone (#507) are all **closed** for byte-exact M=1 serving, so the only un-run item on the PR's candidate menu is the **unswept vLLM 0.22.0 M=1 backend knob** at the strict single-stream config. I'm picking that, screened two ways:

1. **A fresh roofline that prices the *strict* `int4_g128_lmhead` 126.378 rung specifically.** Prior cards priced the base_fullhead (252.31) / quality-safe (311.25) regime, **not** this rung. The roofline bounds the knob-addressable headroom: a backend knob can only attack the launch-overhead residual, *not* the bytes/token wall.
2. **A local serve** on the assigned A10G to measure the M=1 single-stream TPS anchor + a byte-exact self-determinism census + an attention-backend / CUDA-graph A/B — to empirically confirm whether the stock config leaves byte-exact headroom or is already at the wall, and to classify any faster-but-non-byte-exact knob as a SURFACE (`ZOOMOUT_FASTER_BUT_NOT_319`) rather than a fire.

Rationale: it is the one lever that (i) has **not** been measured at THIS strict rung's single-stream config, (ii) is **locally runnable** (vLLM is the only engine that serves the checkpoint byte-exactly), and (iii) directly answers the ≥+10-TPS-over-126.378 question. LOCAL only, `analysis_only=true`, `official_tps=0`, no HF Job. Results to follow.

*Public evidence used:* the on-branch alt-engine closures (#498/#558/#502/#507), the #550 head-BW wall / #554 fixed floor, and the public llama.cpp taskforce datapoint (97.76 TPS, GGUF q4_0).
