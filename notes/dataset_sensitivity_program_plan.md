# Dataset-Sensitivity Program — Plan v3 (2026-08-28)

**v3 supersedes v2** (same filename). v2 folded the completed battery into a clean structure; v3 folds a
three-way adversarial audit (coverage agent + two sibling sessions, yoado-6e theory / yoado-ba metric)
and positions the program inside the thesis' reconstruction end-goal. **This file is also the source for
a supervisor-facing scientific document, so it prioritizes scientific precision over brevity.**

Cross-linked companion docs (neither is sole source-of-truth; see §I.4 Definitions reconciliation):
[whitened_sensitivity_metric.md](whitened_sensitivity_metric.md) (this program's metric spec),
[leakage_story_consolidated.md](leakage_story_consolidated.md) (the Jacobian program: r_J / q_eff +
the 0/40 reconstruction result), [next_experiment_plan.md](next_experiment_plan.md) (the activation
crux), [crux_activation_analysis.md](crux_activation_analysis.md) (crux, on-disk analysis),
[jacobian_leakage_experiment_plan.md](jacobian_leakage_experiment_plan.md) (J-series controls incl. J3).

---

## I. SCOPE, INHERITANCE, AND VOCABULARY (read before the results)

### I.1 What this program measures, and where it sits in the thesis
The thesis end-goal is **reconstruction** of private fine-tuning data from a released adapter (the Haim
et al. lineage: recover the training images, not merely detect them). This dataset-sensitivity program
does **not** reconstruct images. It measures the **LEAKAGE CEILING**: how much information about the
private set is present in the adapter, above the training-randomness floor.

The whitened sensitivity d² is exactly the right object for a ceiling because, under the Gaussian
equal-Σ approximation, **d² = 2·KL(with-change ‖ without-change) = the optimal-detector (Neyman–Pearson)
SNR²** — the best ANY attacker can do at telling D from D′. As a Fisher information it **lower-bounds
(Cramér–Rao) the estimation error of the adapter-space change Δμ**. Therefore:

> **d² is an UPPER BOUND / NECESSARY CONDITION on every downstream attack, reconstruction included.**
> If d² ≈ 0 (change indistinguishable from reseeding), no attack can recover the change. If d² is large,
> recovery is *permitted*, not *achieved*.

### I.2 The inherited reconstruction result (the gap between ceiling and achievement)
On record ([leakage_story_consolidated.md](leakage_story_consolidated.md) §3, verified yoado-a2/aa across
20 result files): the end-to-end **pixel** reconstruction FAILS the honest bar — **0 of 40 decoded
adapter-only arms beat the trivial mean-image baseline** on the like-for-like raw-SSIM comparison; even
the **TRUE-ΔW oracle** fails the baseline on small nets (e.g. fashion N=2 gelu 0.411 < 0.646). The
*geometry* leaks (r_J / q_eff are high, §I.4); the *pixels* do not yet.

**This program characterizes the ceiling; the 0/40 is the ceiling-to-achievement gap.** The two are
consistent: a high detection ceiling with a failed pixel attack means the information is present but the
current decoder + inversion pipeline cannot convert adapter-space Fisher into pixels. Making the ceiling
rigorous, and mapping which images/knobs raise it, is prerequisite to closing that gap.

### I.3 The rigorous bridge to reconstruction (scheduled, not hand-waved)
d² lower-bounds the **adapter-space** error, NOT pixel-MSE directly (the ssim-overclaim trap: never state
d² as a reconstruction-MSE bound). The honest bridge exists and is buildable:

> **J-composed Fisher bridge** (next-phase item §III.6). The data-Jacobian **J = ∂(adapter)/∂(image)**
> already exists and is validated (the jacobian_spectrum / exact double-backprop work). Push the
> adapter-space Fisher through J to obtain the **image-space Fisher** F_img = Jᵀ Σ⁻¹ J, then Cramér–Rao
> (or Fano rate-distortion) gives a genuine **lower bound on achievable pixel error**. That converts "the
> adapter is sensitive" into "no attacker can reconstruct image i to better than distortion δ_i", and
> makes the 0/40 interpretable (is the pipeline near or far from the information-theoretic floor?).

### I.4 Definitions reconciliation — ONE leakage vocabulary
Two programs measure leakage with different instruments; v3 fixes one bridge so numbers are comparable.

| Program | Object | Quantity | Meaning |
|---|---|---|---|
| Jacobian ([leakage_story_consolidated.md](leakage_story_consolidated.md)) | data-Jacobian J of the unrolled fine-tune | **r_J** (hard rank of col J); **q_eff(ε)** (noise-whitened recoverable direction COUNT, ≥ lower bound) | how many private DIRECTIONS the adapter geometrically encodes / an attacker can recover |
| This program (dataset-sensitivity) | reseed covariance Σ of ΔW=BA | **d²** (whitened Mahalanobis sensitivity); its **q_eff** = thresholded spectrum count | the DETECTION SNR² of a composition change vs the seed-noise floor |

- **q_eff is the SAME construct in both** — a thresholded/coarsened count of a whitened d²-type spectrum;
  it is NOT an independent metric from d² (§II reporting rule 2).
- **Rank reconciliation (audit C3).** The two programs' rank behavior is only *apparently* opposite and
  fully reconciles: **r_J SATURATES at every rank** (a directional COUNT — once the r private directions
  are spanned, more rank adds no new directions), while **absolute d² GROWS with rank partly
  MECHANICALLY** (higher rank = more whitened dimensions in the quadratic form, so the scalar inflates by
  dimension even at fixed per-direction signal). So "leakage grows with rank" (d², absolute) and "leakage
  saturates with rank" (r_J, directional) are the same physics read on two scales. v3's §II rule 3
  requires the dimension-corrected d² for any cross-rank claim, which removes the apparent contradiction.

### I.5 Parked / pointer items (schedule a decision, do not go silent)
- **Multi-class knob table + `subhead_k` (head-width knob):** PARKED. Executable spec lives in
  [multiclass_replication_plan_DRAFT.md](multiclass_replication_plan_DRAFT.md) (binary-vs-10-class on
  r_J/q_eff, subhead_k built in parallel). Decision point: resume only if a reviewer needs the
  binary/multi-class contrast on THIS program's arms; otherwise it stays a Jacobian-program deliverable.
- **Generative / diffusion priors (framework Direction 3):** the DESIGNATED remedy for the §I.2 pixel
  failure (SDS from a frozen diffusion model to hallucinate the high-frequency detail the adapter-only
  decoder misses). **Decision point (schedule, §III.6):** after the J-composed Fisher bridge tells us
  whether the pipeline is information-limited (⇒ priors cannot help, the ceiling itself is low for those
  images) or decoder-limited (⇒ priors are the right remedy). Do not start priors before that diagnosis.

---

## II. THE METRIC (validated infrastructure — reference and use, do not re-derive)

3-way cross-fit whitened (Mahalanobis) sensitivity, `experiments/dataset_sensitivity/whitened_metric.py`.
Full spec + audit history: [whitened_sensitivity_metric.md](whitened_sensitivity_metric.md).

**Validation status.** (a) 3-way disjoint cross-fit (subspace U / numerator Δμ·U / denominator λ from
three disjoint seed-folds, rotated) LANDED; self-test 4/4 incl. a K-convergence gate (3-way E[d²] drift
+6.3% K→2K vs old 2-way +44%). (b) **Proven UNBIASED on a signal-free control (null-diag, job 212413)**:
reseed-vs-reseed reads ~0 at every K (0.095 → −0.002, p=0.17–0.64, not growing) while real swaps read
8→22 at p=0.002. Residual K-growth on real data is therefore benign signal-direction RESOLUTION: d² is a
consistent LOWER BOUND that tightens with K.

**REPORTING RULES (verbatim on every new number; non-negotiable):**
1. **PRIMARY = detection**: the sign-flip permutation p-value (floor-free, K-stable). This, not a
   magnitude, is the headline of every arm.
2. **q_eff is a thresholded COARSENING of the d² spectrum — NOT an independent metric.** It is reported
   as a lower-bound direction count "≥ X at K", adequacy-gated ({K,2K} stability AND eff_rank(Σ) ≳ p),
   never as an absolute count. Because it is the same object as d² coarsened, **q_eff is NOT independent
   corroboration of d²** (audit C8): the only genuine cross-metric consistency check in the program is
   **sensitivity ↔ the H-gate mem-score** (§III.3), a behavioral quantity independent of Δ-parameters.
3. **Magnitude = a LOWER BOUND at a STATED K; comparisons only at FIXED K** (and fixed p, frozen across
   the swept variable). Never a bare absolute d². **Cross-RANK magnitude is dimension-confounded**
   (r=32 ≈ 4× the whitened directions of r=8 ⇒ mechanical growth): report **d²-per-recoverable-direction**
   and/or overlay the null d²(r); any absolute cross-rank number is labeled "uncorrected, at K=…".
4. K ≥ 50 headline / K=20 scouting; p ≤ 3–5; {K,2K} adequacy gate; shrinkage/small-eigenvalue floor
   mandatory; NaN re-runs dropped with counts; Σ-invariance + Gaussianity gates reported (else d² is a
   heuristic SNR, use Jeffreys-KL + pooled Σ).
5. Everything in ΔW=BA gauge (never raw A/B factors).
6. **No "privacy leakage" framing for any magnitude until the H gate (§III.3) closes** — until then all
   results are "parametric detectability / sensitivity", a CEILING, not a demonstrated attack.

**Metric-validation gates that MUST run (same standing as the null-diag, not if-time-permits) (audit C6, refinement 2b):**
- **p-value uniformity**: over ≥ 20 signal-free redraws (reseed-vs-reseed) the permutation p must be
  ≈Uniform(0,1); a non-uniform/anti-conservative p invalidates every detection headline.
- **q_eff null**: q_eff ≈ 0 on reseed-vs-reseed (its own null) at the reporting K.
Both are prerequisites for quoting ANY arm; schedule them as a standing metric-CI job, not per-arm.

---

## III. RESULTS + NEXT PHASE

### The narrative spine (what the completed battery converged on — every next experiment tests it)

> **Per-image LoRA leakage is governed by the IMAGE's relationship to the BASE model — its base gradient
> norm g₀ — NOT by dataset-composition knobs.** Composition knobs are weak (dilution flat in N;
> duplication deeply sub-linear; context rarity ~nothing once identity is controlled). What predicts
> leakage is the image: ρ(sens, g₀) = +0.857 (n=12, job 260171). It transfers to a real ViT (single image
> detectable in a vit_tiny LoRA at p=0.002). **Capstone (if it survives scaling + the H gate): leakage is
> predictable WITHOUT ADAPTER ACCESS — from the public base model + the candidate image alone.**

**MECHANISM — this is NTK / gradient-recording, NOT max-margin (theory reframe T1, MAKE-OR-BREAK).**
Our OWN data forces this: the init-gradient quantity g₀ fits sensitivity at **ρ=+0.857**, while the
theory-licensed **max-margin dual proxy λ=σ(−margin_T) fits WORSE at ρ=+0.538** — the KKT quantity is the
weaker predictor. The correct mechanism is the project's own "LoRA as gradient projection" framing:
ΔW ≈ a low-rank projection of accumulated training-gradient contributions (Jang 2024: LoRA ≈ NTK in the
lazy regime), so **a large base gradient ⇒ a large, detectable adapter component.** Max-margin / KKT is
**demoted to the T→∞ limiting connection** (a motivation for *why* hard/atypical images matter
asymptotically), NOT the operative explanation at our finite T. **No KKT/max-margin language may be used
as an explanation without the T2 convergence diagnostics below.**

**Duplication wording correction (T4), apply everywhere:** never "duplication-invariance". Correct
statement: duplication imprint is **sub-linear (β ≈ 0.23–0.31 on d²), and β(T) DECREASES with training
budget** (0.313→0.256→0.234 at T=50/200/1000), i.e. it *trends toward* the β=0 max-margin
duplication-invariant limit as T grows **but does not reach it at T=1000 ⇒ the system is NOT at the KKT
fixed point.** β(T) decreasing is itself a convergence diagnostic (how far from the max-margin limit).

#### Results table (verified, with jobs; details in STATUS.md 2026-08-27/28)

| Arm | Jobs | Finding | Status |
|---|---|---|---|
| Metric null-diag | 212413 | 3-way estimator UNBIASED on no-signal data at every K → the §II reporting rule | **DONE — foundation** |
| **B — subset size / dilution** | reconfirm 130198, K=50/100 (post-3-way) | Detection FLAT in N (p=0.002 at every N); old "sharpens with N" was the 2-way denominator artifact, KILLED. N-shape (flat, decline at N=32) identical at K=50 & 100. **N=32 decline UNEXPLAINED — probe it (§III.1 asymptote line)** | **DONE (one open sub-question)** |
| **E — duplication** | 162114, 217123 (T-sweep), 246873 (fashion) | Sub-linear β≈0.234 (r8) / 0.241 (r32), R²=0.85; β(T) 0.313→0.256→0.234 (see T4). Rank-INVARIANT β, but r=32 ≈ 15× absolute d² of r=8 (dimension-confounded, see §II rule 3). Fashion replicates (β 0.288/0.359, slight capacity modulation) | **DONE** |
| **C — class imbalance** | 229722, 237301 (role-swap) | Two separable effects: 3.3× INTRINSIC class-identity asymmetry (survives balance; inverts cleanly under role-swap 3.28→0.34); rarity is CLASS-DEPENDENT (~3× for the loud class, absent for the quiet). All p=0.002 | **DONE** |
| **D — context rarity** | 245964 | Fixed-image rarity WEAK (mean gain 1.11, ~noise). Reconciles C: its big ratios were image/class identity, not context. "Some images leak more" is about the IMAGE | **DONE** |
| **Margin/gradnorm test** | 260171 | ρ(sens, g₀)=+0.857 (P2); ρ(sens, λ)=+0.538 (P3, the weaker — see T1); 3.3× class asymmetry EXPLAINED (loud class: smaller base margin 2.95 vs 4.70, 2.4× larger g₀). Raw margin weak (ρ=−0.296, right sign). **Caveat: n=12, ρ CI ≈ ±0.4** | **DONE (MVP) → scale (§III.1)** |
| **ViT+LoRA (data-type axis)** | 247474 (MVP), 256540 (scaled) | Single image detectable in vit_tiny_patch16_224 rank-4 LoRA (blocks 0-2 qkv): scaled N=16/K=50 all 3 targets p=0.002. MLP premise generalizes to a real ViT | **DONE** |
| Fashion detectability (arm B on fashion) | 246872 | — | **RUNNING** |
| A (typicality-proxy version) | — | Largely absorbed by the margin test (g₀ is the base-side predictor); ONE θ₀-independent typicality control retained in §III.1 (audit C10) | absorbed |
| F — cross-dataset | — | Not run; reframed through the margin lens (§III.5) | pending |
| G — OOD injection | — | Absorbed into the similarity axis as S3 (§III.2), margin-framed | absorbed |
| H — validation gate | — | NOT run; make-or-break; cheap spot-check re-sequenced FIRST (§III.0) | **priority** |
| Phase-2 adapter atlas | — | Gated on the adapter population | pending |

**Per-experiment plotting + asymptote discipline (audit C9, C11) — applies to every item below:** each
pre-declares (i) its **headline plot** (axes, expected shape, what-good-looks-like) and (ii) its
**expected asymptote + would-we-detect-a-violation** line.

---

### III.0 CHEAP H SPOT-CHECK — runs FIRST, de-risks the ordering (audit C4)
Before spending compute on §III.1/§III.2 scaling, run a small-scale H probe: on an existing arm's images
(N≤16), compute the leave-one-out Feldman-Zhang mem score and its rank-correlation with whitened
sensitivity.
- **Headline plot:** scatter of whitened sensitivity (x) vs LOO mem score (y), one point per image;
  **expected shape: positive slope.** What-good-looks-like: ρ_rank > +0.4 with a visible trend.
- **Purpose:** H stays the make-or-break gate (§III.3), but a positive spot-check de-risks the decision
  to scale §1/§2; a flat spot-check says stop and fix the sensitivity→memorization link first.
- **Kill:** ρ_rank ≤ 0 ⇒ do NOT scale §1/§2 until §III.3 is understood.

### III.1 MARGIN AT SCALE — the headline figure (WHO leaks)
Upgrade job 260171 from n=12 to **20–40 targets** stratified by g₀ decile, both classes, several contexts.
Also on **fashion**.
- **Headline plot:** sens-vs-g₀ scatter, per-stratum ρ overlaid; **expected shape: monotone positive**,
  g₀ sharper than raw margin.
- **Pre-registered prediction (audit C7):** ρ_spearman(sens, g₀) > +0.6, positive in every stratum;
  AND a **bootstrap 95% CI on ρ narrower than ±0.15** (the current 0.857 has n=12, CI ≈ ±0.4 — the width
  target, not just the point estimate, is the deliverable). Report permutation-p AND bootstrap CI for ρ.
- **θ₀-independent typicality control (audit C10, MANDATORY):** alongside g₀, correlate sensitivity with
  a typicality score from a **different pretrained network** (never θ₀ — k-NN density in θ₀'s own features
  is circular). This separates "base-model geometry predicts leakage" (the claim) from "the image is
  intrinsically hard/atypical" (a confound). If g₀ predicts only because it proxies intrinsic atypicality,
  the capstone weakens; report the partial correlation of sens on g₀ controlling for typicality.
- **Lazy-regime / NTK-mechanism diagnostic (theory T5 + refinement 1) — cheap, run here:**
  - **PRIMARY: spearman(g₀, g_T) per image** — is the per-image gradient RANKING preserved from base to
    trained? High ⇒ gradient structure is preserved ⇒ g₀ predicts *because* the NTK/gradient-recording
    mechanism holds (this is the direct mechanism evidence, valid even without strict laziness).
  - **CONTEXT: per-module ‖ΔWℓ‖_F / ‖W₀,ℓ‖_F for EACH LoRA target module SEPARATELY** — denominator is
    THAT module's own frozen base weight, **never** global ‖θ₀‖ (conflates model size with laziness) and
    **never** only a summed scalar (laziness can fail in one module while holding in another; a
    sqrt-of-sums aggregate may SUPPLEMENT but not replace the per-module view). Interpretation: ratio
    ≪1 (<0.1) = lazy/NTK regime; O(1) = that layer moved substantially ⇒ needs a feature-learning
    explanation. **High spearman(g₀,g_T) + small per-module ratio = the NTK story airtight.**
- **Expected asymptote:** with enough targets ρ stabilizes; a large-n ρ that keeps drifting ⇒ strata are
  not exchangeable (context/class confound leaking in).
- **Kill:** ρ < +0.3 overall or sign flips across strata ⇒ the capstone falls; demote to an anecdote.

### III.2 SIMILARITY AXIS — instance vs concept (co-top-tier; WHAT is leaked)
The semantic layer on top of the margin narrative. All completed arms swapped exact identities; these
grade the swap by VISUAL/SEMANTIC distance from a **θ₀-independent encoder** (DINO via timm if available,
else an ImageNet-pretrained vit_tiny CLS embedding; raw-pixel L2 as cross-check). Order S1 → S2 → S3.

**S1 — SIMILARITY-GRADED SWAP ("the distance dial").** Swap target T for T′ at graded distance:
parametric perturbations of T (brightness/contrast/affine/blur, increasing magnitude) and near/median/far
same-class neighbors. Sensitivity vs embedding distance d(T,T′).
- **d=0 IS THE MANDATORY FIRST RUNG (audit C1):** the trivial swap T′=T must give **d²(0) ≈ the null
  floor** (pre-registered). A nonzero d²(0) is an **artifact-kill for the entire dial** — stop and fix the
  metric before interpreting any nonzero rung.
- **Headline plot:** sensitivity (y) vs encoder distance d(T,T′) (x), starting at d=0≈floor; **expected
  shape: rises from the floor with d.**
- **Prediction (margin/NTK lens):** sensitivity rises with d; near-duplicates ~undetectable (p not
  significant at K=50) ⇒ the adapter records the CONCEPT, not the instance (privacy statement: an attacker
  recovers "a smiling headshot", not WHICH person). If near-duplicates STAY detectable ⇒ instance-level
  memorization (scarier, also publishable).
- **Expected asymptote:** far same-class neighbor → the exact-swap sensitivity of §III (the dial's
  far end should reproduce arm-C/D-scale numbers).
- **Kill:** flat vs distance (given d²(0)≈floor passed) ⇒ similarity doesn't matter; close the axis.

**S2 — TWIN SHIELDING = THE MECHANISM DISCRIMINATOR (promoted, theory T5).** This is the ONE experiment
that separates max-margin from NTK-recording. Put T and a visual near-twin T″ in the set together; measure
T's swap-sensitivity twin-PRESENT vs twin-ABSENT.
- **The discriminating predictions:** **max-margin** predicts SHIELDING (the twin shares dual mass /
  covers T's margin constraint ⇒ sensitivity(T | twin present) DROPS materially). **NTK-recording**
  predicts LITTLE/NO shielding (each image contributes its own recorded gradient projection largely
  independently). The observed drop (or its absence) directly adjudicates the §III mechanism claim.
- **Headline plot:** paired bars, sensitivity(T) twin-absent vs twin-present; expected under NTK: bars
  ~equal; under max-margin: present ≪ absent.
- **Kill/interpretation:** either outcome is a result (no drop ⇒ NTK confirmed, contributions separable,
  instance attribution stands; large drop ⇒ max-margin shielding, similar images NOT separable). The only
  failure is an underpowered null — enforce K≥50, p≤3.

**S3 — OOD-STYLE DIGIT INJECTION (= old arm G, margin-framed).** Inject visually-similar-but-different-
style digits (**USPS via torchvision**, closest drop-in; EMNIST fallback — no scraping) into an MNIST set.
**PREDICT each injected image's sensitivity from g₀ FIRST** (pre-registered, public base only), then
measure — an out-of-distribution test of the margin predictor.
- **Prediction:** OOD digits sit high in the g₀ distribution and leak MORE, AND land on the
  in-distribution sens-vs-g₀ curve (within CI) — style-OOD has no effect beyond its g₀.
- **Headline plot:** overlay OOD points on §III.1's in-dist sens-vs-g₀ curve; expected: on the curve, at
  the high-g₀ end.
- **Expected asymptote:** all-OOD set (n_ood→N) — sensitivity should track the g₀ distribution's shift,
  not jump discontinuously.
- **Kill/informative failure:** OOD points systematically OFF the curve ⇒ g₀ insufficient, distribution
  membership carries independent signal (a publishable qualification of the spine).

### III.3 H — TRIANGLE-CLOSING VALIDATION GATE (make-or-break; unchanged in stakes)
Does whitened sensitivity predict an attacker-observable? PRIMARY: **leave-one-out Feldman-Zhang mem
score** (behavioral, independent of Δ-parameters). Secondary if feasible: loss-threshold MIA /
reconstruction quality. (§III.0 is the cheap spot-check; this is the full gate.)
- **Headline plot:** sensitivity (x) vs LOO mem score (y) scatter, **expected positive slope.**
- **Prediction:** rank correlation > +0.4.
- **This is the ONE genuine cross-metric consistency check** (audit C8): q_eff cannot corroborate d²
  (same object); a behavioral mem score can.
- **Kill:** no positive association ⇒ everything above is a parametric CEILING, NOT demonstrated privacy;
  all "leakage" language in the thesis must be rewritten. **No magnitude may be called leakage until this
  closes.**

### III.4 ViT EXPANSION (turn one thesis-grade point into a curve)
a) **Rank sweep on ViT** — prediction: dimension-corrected sensitivity grows with r (the MLP's directional
   effect replicates); report per-recoverable-direction (§II rule 3). Kill: flat ⇒ MLP-specific.
b) **Margin-vs-leakage on ViT** — does g₀ transfer? Prediction: ρ(sens, g₀) > +0.5 on vit_tiny. **If yes,
   the strongest thesis claim** (public-base predictability on a real foundation-model architecture).
c) **Harder data**: CIFAR 2-class, optionally vit_small. Prediction: detectability persists (p ≤ 0.01).

### III.5 F — CROSS-DATASET COMPOSITION, margin-framed
mnist+fashion mixed private sets. Same reframe as S3: foreign-dataset images are extreme large-g₀ points,
so **predict from g₀ first, then measure**. Prediction: mixed-in fashion images leak most, on the same
sens-vs-g₀ curve. Kill/informative: off-curve ⇒ composition carries independent signal.

### III.6 BRIDGES TO THE END-GOAL (scope items from §I — scheduled, not silent)
- **J-composed Fisher bridge (§I.3):** implement F_img = Jᵀ Σ⁻¹ J using the existing validated J; derive
  the Cramér–Rao / Fano pixel-error lower bound per image; check whether the 0/40 pipeline is near or far
  from that floor. Headline: per-image pixel-distortion lower bound vs achieved decoder distortion.
- **J3 — disjoint-adapter honesty control** (from [jacobian_leakage_experiment_plan.md](jacobian_leakage_experiment_plan.md)
  §J3): does a reconstruction come from THIS adapter or the decoder PRIOR? True set X_A→Y_A vs a tightly
  matched disjoint X_B→Y_B; the attack (aimed at X_A) run from each; claim survives only if
  Q(X̂(Y_A),X_A) − Q(X̂(Y_B),X_A) > 0 significantly (Q ∈ {LPIPS, CLIP, DINO}). Schedule alongside any
  reconstruction claim — it is the sharpest guard against prior-driven hallucination.
- **Diffusion/generative priors decision point (§I.5):** after the J-composed Fisher bridge diagnoses
  information-limited vs decoder-limited, decide whether Direction-3 SDS priors are warranted.

### III.7 THE ACTIVATION × ANCHOR × LINEARIZATION CRUX (supervisor's explicit ask — restored)
Everything above is **GELU-only**; the supervisor's top ask is the coupled smoothness study (job 857271
smoothness-sweep tensors are on disk, 21 configs, partly unanalyzed — [next_experiment_plan.md](next_experiment_plan.md)
STEP 2). **KNOWN RESULT — do not re-open blind** ([crux_activation_analysis.md](crux_activation_analysis.md),
softplus_bβ analysis): **activation smoothness sets reconstruction FIDELITY, not the leakage
direction-COUNT** — r_J was β-independent, and the naive "smoother ⇒ more leakage" law is REFUTED
(spearman(smoothness, leakage) ≈ +0.03 on MNIST; inverts on flowers32). The chain link that holds is
smoothness → linearization fidelity; the link that breaks is linearization → leakage magnitude.
- **The OPEN question to schedule (the seam, not the settled part): does per-image g₀-PREDICTABILITY vary
  with activation smoothness?** i.e. does ρ(sens, g₀) (the §III.1 capstone) hold across the smoothness
  spectrum, or is the base-model-predictor a GELU artifact? This is the interaction of this program's
  spine with the crux. **yoado-ba owns the review of this seam** — coordinate before running.
- **Compute note:** rescoring the 857271 tensors is ~0 GPU (CPU minutes); a fresh matched-weight_change
  sweep is the STEP-2a GPU job if the existing grid misses the NTK band.

### III.8 PHASE-2 — adapter-space atlas (after the battery)
Embed/cluster the accumulated ΔW=BA population (every arm saves its adapters). GL(r)-invariant
featurization (SVD subspace + spectrum); two-sided Bures–Wasserstein (C_out AND C_in) or Grassmann+spectral;
precomputed-metric clustering/UMAP only. Questions: do adapters cluster by composition? are mixed (arm-F)
adapters ~linear combos of pure ones? does per-image sensitivity correlate with position/spread?

### III.9 PARKED
Base architecture / width / depth sweep — needs new base checkpoints; not before §III.0–III.5 land.

---

## IV. ANTICIPATED OBJECTIONS (the four Gal-attacks, each pre-empted) (theory T7)

1. **"Your regime isn't the one the implicit-bias theorems cover."** True — LoRA-on-a-pretrained-base is
   NOT a homogeneous small-init net. **Pre-empt:** we do NOT lean on KKT for the mechanism (T1); the
   mechanism is NTK/gradient-recording. The **T2 convergence diagnostics run first** (data separability;
   stationarity residual ‖θ − Σλᵢyᵢ∇Φ‖; λ≥0 / complementary slackness) and any KKT language is gated on
   them. Max-margin is stated only as the T→∞ limit.
2. **"If it's max-margin, β should be 0 (duplication-invariant) — it isn't."** Correct, β≈0.23–0.31.
   **Pre-empt (T4):** we claim sub-linear *trending toward* β=0 as T grows (β(T) decreasing), which is
   evidence we are NOT at the KKT fixed point — consistent with the NTK-recording mechanism, not with a
   converged max-margin solution. β(T) is reported as a convergence diagnostic.
3. **"Why should the FROZEN base θ₀ predict the TRAINED adapter θ_T at all?"** **Pre-empt (T5/§III.1):**
   the lazy-regime diagnostic — PRIMARY spearman(g₀, g_T) per image (gradient ranking preserved) + CONTEXT
   per-module ‖ΔWℓ‖/‖W₀,ℓ‖ (small ⇒ lazy). High spearman + small per-module ratio = θ₀ predicts because
   the gradient structure is preserved through training. If laziness fails, we say so and the predictor is
   demoted.
4. **"Detectability is not reconstruction."** Exactly — and we say so up front (§I.1). d² is the CEILING /
   necessary condition; the 0/40 is the gap; the J-composed Fisher bridge (§III.6) is how we make the
   connection rigorous rather than asserting it. No detection number is ever stated as a reconstruction
   result.

---

## V. DESIGN LOCKS + DISCIPLINE (a fresh session must obey these)

### Metric usage lock
- 3-way cross-fit `whitened_metric.py` ONLY (the 2-way lives in-file solely as the K-convergence
  regression fixture — never for results).
- §II reporting rules 1–6 verbatim on every number; the p-value-uniformity + q_eff-null gates (§II) are
  standing MUST-RUN prerequisites, not optional.
- K ≥ 50 headline / K=20 scouting; p ≤ 3–5 FROZEN across any swept condition; {K,2K} gate; shrinkage
  floor; NaN-drop with counts; Σ-invariance + Gaussianity gates reported. ΔW=BA gauge only.

### Arm-E construction lock (audit-hardened; reuse for any duplication-flavored arm)
- FIXED-PREVALENCE: k_max class-1 slots constant for all k; copies_k = context + [T]*k + donors[k:k_max];
  distinct baseline IDENTICAL across all k & targets ⇒ Σ frozen by construction (trained once per rank).
- NO analytic β=2 null; empirical null = the rank contrast (r<N vs r>N).
- Headline β on DEBIASED sensitivity (d²_obs − d²_null); drop-counters + finite-d² assert before fitting.

### Typicality / encoder lock (M1 circularity rule)
- Any typicality or similarity score comes from a θ₀-INDEPENDENT net (a DIFFERENT pretrained model),
  NEVER θ₀'s own features. Report ρ-vs-atypicality as a PARTIAL correlation controlling for g₀.

### Data/base lock
- Base = MNIST-train-pretrained checkpoint; targets from MNIST-test ⇒ held-out, no overlap. Fashion
  mirrors this. ViT: vit_tiny_patch16_224 (timm 0.9.12), rank-4 LoRA on blocks.0-2 qkv (peft 0.7.1),
  verified trainable (fit ~2e-4). GELU throughout (exact-J compatible); the crux (§III.7) is the only
  place activation varies.
- Hold T and LoRA capacity fixed within an arm (Carlini control); sweep them only deliberately.
- Every run SAVES its adapters (ΔW=BA per layer) + composition metadata .pth — Phase-1 output IS Phase-2
  input, and the population for §III.8.
- When reconstructing measurement identity across scripts, IMPORT the arm's own construction functions and
  hard-assert saved metadata (the margin-test method) — never re-implement.

### Process discipline
- **Planner → auditor → executor**: every new arm design gets an adversarial audit before submission;
  resolutions fold into this locks section. yoado-ba reviews the crux seam (§III.7); coordinate before
  running interaction experiments.
- **Stage-0 gates**: every arm ships a tiny smoke config (`run_*_stage0_wexac.sh`) that must pass on WEXAC
  before full submission.
- **WEXAC bsub ONLY — never run locally**, not even smoke tests. `python -u` in job scripts. rsync
  `experiments/` before submitting.
- Artifact hygiene (this program's scars, LESSONS_LEARNED 2026-08-27/28): a surprising trend in a
  whitened/normalized metric ⇒ suspect the DENOMINATOR first; K-non-convergence is a sufficient artifact
  disqualifier on its own; "grows with sample size" is bias ONLY if a signal-free control also grows —
  run the null control before believing OR retracting.
- After every completed arm: update STATUS.md (headline + config + jobs) and LESSONS_LEARNED.md (if a
  pitfall) in the same turn; commit.

---

## Grounding (compact)
DP sensitivity: Dwork et al. 2006. NTK / lazy training: Jacot 2018, Jang 2024 (LoRA≈NTK, r≳N ≈ full-FT).
Memorization: Feldman 1906.05271, Feldman-Zhang 2008.03703, Carlini 2202.07646, onion 2206.10469.
Implicit bias (T→∞ limit only): Lyu-Li / Ji-Telgarsky homogeneous-net max-margin. Testing template: f-INE
2510.10510. Attribution (behavioral, for contrast, NOT used): Koh-Liang 1703.04730, TRAK 2303.14186,
DataInf 2310.00902. Reconstruction lineage: Haim et al. NeurIPS 2022. Fisher/rate-distortion bridge:
Cramér–Rao, Fano. Adapter geometry (Phase-2): LoRA 2106.09685, task-arithmetic 2212.04089, model-soups
2203.05482, EigenLoRAx 2502.04700, Bures-Wasserstein 2302.14618.
