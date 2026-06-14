### Approval request: HF job for tree submission -- REFRESHED go/no-go packet (PRE-FILLED DRAFT, NOT FILED)

**HEADLINE: both-bugs is the robust GO** -- LCB(P>=0.9) 514.88 TPS, P(clear-500) 99.59%
at the shipped launch-realized step 1.2182. descent-only-first is a knife-edge MISS at the
shipped step (LCB 499.97, P 89.94%, -0.035 TPS).

**This packet INFORMS but does NOT authorize a launch; a human must approve the filed `Approval request: HF job` issue before any spend.**

**Composition:** `official = K_cal * (E[T]/step) * tau` (K_cal=125.268, tau in [0.9924, 1.0]);
clear-500 bar E[T] >= 4.862 (shipped). TWO independent launch axes:
- PROJECTION axis = 3-term input-band quadrature sqrt(sampling^2 + calibration^2 + step_anchor^2).
- HARDWARE axis = kanna #159 sigma_hw 4.86 TPS (cross-allocation-dominated),
  RETIRED by best-of-2 official draws -> P=0.9829 >= 0.90 (does NOT subtract from the LCB).

**Projection geometry at the three step framings (full-recovery corner lambda=mu=1, pinned 1.80% drop):**

| framing / topology | official | proj_private | P(clear 500) | LCB(P>=0.9) | launch |
|---|---|---|---|---|---|
| roofline 1.2127 -- descent-only | 522.3 | 516.2 | 93.10% | 502.20 | GO |
| roofline 1.2127 -- both-bugs | 537.8 | 531.3 | 99.78% | 517.18 | GO |
| shipped 1.2182 -- descent-only (MISS) | 520.0 | 513.9 | 89.94% | 499.97 | HOLD |
| shipped 1.2182 -- both-bugs (GO) | 535.4 | 528.9 | 99.59% | 514.88 | GO |
| scatter-LP 1.2047 -- descent-only | 525.8 | 519.7 | 96.31% | 505.57 | GO |
| scatter-LP 1.2047 -- both-bugs | 541.4 | 534.8 | 99.92% | 520.65 | GO |

both-bugs is GO at all three framings (LCB 514.9 ->
520.6); descent-only is the knife-edge miss at the shipped
step ONLY (GO at roofline 502.2 and scatter-LP 505.6).

**MATERIAL CAVEAT -- numerator E[T] floor (denken #172):** the central numerator (descent 5.0564 /
both 5.2070) is the OPTIMISTIC full-recovery value used above. denken #172's adversarial self-KV
floor 3.5346 projects to 363.5 TPS (BOUNDED-NOT-ROBUST) -- it FAILS 500.
The 515-class GO REQUIRES the deep-spine spread to recover to >= 91% of the
rho-optimal rising ladder (i.e. openevolve cause #2, depth>0 self-KV starvation, must be a FIXABLE build defect, not intrinsic).
The denken realistic-floor refinement (IN-FLIGHT) converts this modeled floor to measured -- it is the highest-leverage open de-risk.

**sigma_hw composition is verdict-INVARIANT.** Whether sigma_hw is RETIRED by best-of-2 (headline, P>=0.9 on a
separate axis) or naively FOLDED as a 4th quadrature term:
- both-bugs: LCB 514.9/P 99.6% (best-of-2) vs LCB
  513.4/P 99.2%
  (naive-fold) -> GO either way.
- descent-only: LCB 500.0/P 89.9% (best-of-2) vs LCB
  498.6/P 87.8%
  (naive-fold) -> MISS either way.

**Descent-only restoration path:** Shipping #154's argmax-only decode adds +3.96 TPS LCB (shipped->realizable), restoring descent-only to GO (LCB 503.92); the full scatter-LP framing 1.2047 lifts descent-only to LCB 505.6. So +3.96 is the CONSERVATIVE restoration.
**Build recommendation:** build recommendation UNCHANGED: land #71's both-bugs kernel is the gating build for the robust GO; the descent-only+#154 path is a viable simpler-build fallback only once #154 ships and the descent E[T] realistic-floor is confirmed.

**Launch gates (ALL required before the filed issue is approved):**
  1. land #71 builds the both-bugs descending accept-prep kernel (the GO-path gating build)
  2. kanna's darwin _IncludedRouter boot-validation startup-500 fix folded into the serve harness
  3. PRECACHE_BENCH=1 set on the served path
  4. a human-approved `Approval request: HF job` issue

**Dependency ledger (5 LANDED / 4 IN-FLIGHT / 1 PENDING / 1 PENDING-BUILD):**
- [LANDED]  Numerator E[T] (#160/#165/#172) -- descent 5.0564 / both 5.2070
- [LANDED]  Denominator step (#168) -- roofline 1.2127 <-> shipped 1.2182 (+-0.22%); scatter-LP 1.2047 if #154 ships
- [LANDED]  Hardware sigma_hw (#159) -- sigma_hw 4.86 TPS (0.96%), cross-allocation-dominated; best-of-2 official draws -> P=0.9829 >= 0.90 on the hardware axis
- [LANDED]  Validity (#166/#150/#158, Issue #124) -- official PPL 2.377 <= 2.42; M=32 worst-case PPL 2.4134 <= 2.42 (margin 0.0066); 128/128; greedy-exact (Issue #124 RESOLVED)
- [LANDED]  Private drop (#164) -- descent native drop 2.04% [1.87,2.21]; descent worst tau-low 504.6 TPS (clears 500)
- [IN-FLIGHT] Numerator E[T] de-risk (denken (in-flight)) -- central +- realistic floor (replaces the adversarial 3.5346 worst case)
- [IN-FLIGHT] Denominator step confirm (#173) -- confirms the built descent kernel realizes 1.2182
- [IN-FLIGHT] Private drop stress (#176 (stark)) -- stresses the private drop under an adverse domain skew
- [IN-FLIGHT] Launch-boot (kanna boot-validation (in-flight)) -- fixes a startup-500 in _IncludedRouter; a NEW hard serve dependency
- [PENDING] Finite-sample CI (#175 (wirbel)) -- composes in quadrature with the input-band sampling term
- [PENDING-BUILD] Build (#71 (land)) -- the GO-path gating build (measured tuple: E[T], rho2, lambda, mu, step, ppl, boots, completed)

**Truly-unmeasurable residual:** an organizer tree re-run on the REAL private set (no proxy reproduces it).
**Gating build:** land #71 (the both-bugs descending accept-prep kernel).
