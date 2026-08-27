# Dataset-Sensitivity Program — Plan v2 (2026-08-28)

**Rewrite of the v1 plan (which grew by appended patches).** v1's ablation catalog, metric derivation and
open auditor questions are superseded: the metric is now specced + validated in
[whitened_sensitivity_metric.md](whitened_sensitivity_metric.md), and most of the battery has RUN.
This file is the single source for: what the program claims, what is proven (with job numbers), what runs
next (with signed predictions + kill criteria), and the locks a fresh session must respect.

---

## i. MISSION + THE EMERGING NARRATIVE (the spine)

**Program question:** how sensitive is a released LoRA adapter to the composition of its private
fine-tuning set — measured directly, per image, against the training-randomness noise floor (the DP
sensitivity notion, on parameters).

**The narrative the completed battery converged on — make every next experiment a test of it:**

> **Per-image LoRA leakage is governed by the IMAGE's relationship to the BASE model — its base gradient
> norm / margin (the support-vector picture) — NOT by dataset-composition knobs.**
>
> - Composition knobs are weak: subset-size dilution is FLAT in N (arm B), duplication is deeply
>   SUB-LINEAR (arm E, β≈0.24 — max-margin duplication-invariance), context rarity is ~nothing once
>   image identity is controlled (arm D, ~1.1x).
> - What actually predicts leakage is the image itself: ρ(sensitivity, base-gradnorm g₀) = +0.857
>   (margin test, job 260171). The 3.3x "class asymmetry" (arm C) is fully explained by class-1 sitting
>   at smaller base margins / 2.4x larger base gradients.
> - It transfers to real architectures: a single private image is detectable in a vit_tiny LoRA adapter
>   at the same p=0.002 certainty floor as the MLP (jobs 247474, 256540).
>
> **Attacker-relevant capstone claim (the thesis headline if it survives scaling + the H gate):
> leakage is PREDICTABLE FROM THE PUBLIC BASE MODEL ALONE** — no adapter access needed to rank which
> private images are most exposed.

Two supporting laws worth stating on their own: (1) duplication sub-linearity is RANK-INVARIANT
(the low-rank bottleneck is NOT what saturates it — same β at r=8 and r=32), while (2) **rank is the
dangerous knob for ABSOLUTE leakage** (~15x d² going r=8→32 at fixed β).

**The next layer (user steer, 2026-08-28): the margin/gradnorm story explains WHO leaks; the
SIMILARITY AXIS (§iv.2) asks WHAT is leaked — the instance or the concept.** All completed swap arms
used exact-identity swaps; whether the adapter records "this specific image" vs "an image like this"
(visual/semantic similarity) is arguably the deeper privacy question, and it is testable with the same
machinery by grading the swap distance.

---

## ii. VALIDATED INFRASTRUCTURE (do not re-derive; reference and use)

**The metric** — 3-way cross-fit whitened (Mahalanobis) sensitivity, `experiments/dataset_sensitivity/whitened_metric.py`.
Full spec + audit history: [whitened_sensitivity_metric.md](whitened_sensitivity_metric.md). Status:

- **3-way disjoint cross-fit LANDED and VALIDATED**: subspace U / numerator Δμ·U / denominator λ from
  three disjoint seed-folds, rotated. Kills the winner's-curse denominator bias that faked the old
  "sharpens with N" arm-B headline. Self-test 4/4 incl. K-convergence gate (3-way E[d²] drift +6.3%
  K→2K vs old 2-way +44%).
- **Estimator proven UNBIASED on a signal-free control (null diagnostic, job 212413)**: on
  reseed-vs-reseed (no composition change) it reads ~0 at every K (0.095 → −0.002, p=0.17–0.64, not
  growing), while real swaps read 8→22 at p=0.002. The residual K-growth on real data is therefore
  benign signal-direction RESOLUTION: d² is a consistent lower bound that tightens with K.

**REPORTING RULE (all arms, non-negotiable):**
1. **PRIMARY = detection**: sign-flip permutation p-value (floor-free, K-stable).
2. **Magnitude = a LOWER BOUND at a STATED K** — never a bare absolute d².
3. **Comparisons only at FIXED K** (and fixed p, frozen across the swept condition).
4. K ≥ 50 for headline numbers (K=20 scouting only); p ≤ 3–5; {K,2K} adequacy gate per arm;
   small-eigenvalue floor / shrinkage mandatory; NaN re-runs dropped with counts reported.
5. Everything in ΔW=BA gauge (never raw A/B factors).
6. **No "privacy leakage" framing for magnitudes until the H gate closes** — until then results are
   "parametric detectability/sensitivity".

**Runner pattern** (all arms follow it): a composition function builds paired sets D/D′ → K paired
retrains → `whitened_metric` → (p, sensitivity, qeff, diagnostics). Existing arm files under
`experiments/dataset_sensitivity/` (arm_b_dilution, arm_c_imbalance, arm_d_context, arm_e_duplication,
margin_vs_sensitivity, vit_lora_sensitivity, arm_b_null_diag) are the templates; each has a
`scripts/run_arm_*_wexac.sh` submitter with a stage-0 gate variant.

---

## iii. RESULTS TABLE (verified, with jobs — the durable record; details in STATUS.md 2026-08-27/28)

| Arm | Jobs | Finding | Status |
|---|---|---|---|
| **Metric null-diag** | 212413 | 3-way estimator UNBIASED on no-signal data at every K; real-data K-growth = benign resolution → reporting rule above | **DONE — foundation** |
| **B — subset size / dilution** | 130198 (K=50+100, post-3-way) | Detection FLAT in N: p=0.002 at every N. The old "per-image effect sharpens with N" was the 2-way denominator artifact — KILLED. N-shape (flat, decline at N=32) identical at K=50 & 100 | **DONE** |
| **E — duplication** | 162114 (main), 217123 (T-sweep), 246873 (fashion) | SUB-LINEAR: β≈0.234 (r=8) / 0.241 (r=32), R²=0.85 — max-margin duplication-invariance. Rank-INVARIANT β, but r=32 ≈ 15x absolute d² of r=8. T-sweep: β 0.313→0.256→0.234 (T=50/200/1000) — sub-linear at every budget, T-caveat closed. Fashion replicates (β 0.288/0.359) with slight capacity modulation (β(r32)>β(r8)) | **DONE** |
| **C — class imbalance** | 229722 (main), 237301 (role-swap) | Two separable effects: (1) 3.3x INTRINSIC class-identity asymmetry (survives balance; inverts cleanly under role-swap, 3.28→0.34); (2) rarity effect is CLASS-DEPENDENT — ~3x for the loud class (class-1), absent for the quiet one. All p=0.002 | **DONE** |
| **D — context rarity** | 245964 | Fixed-image rarity is WEAK: same image rare-vs-common context ⇒ mean gain 1.11 (non-monotone, ~noise). Reconciles C: its big ratios were image/class identity, NOT context. "Some images leak more" is about the IMAGE, not its neighbors | **DONE** |
| **Margin test (= old arm A at MVP scale)** | 260171 | ρ(sens, base-gradnorm g₀)=+0.857 (P2); ρ(sens, post-LoRA dual proxy λ)=+0.538 (P3); the 3.3x class asymmetry EXPLAINED — louder class has smaller base margins (2.95 vs 4.70) / 2.4x larger base gradnorms (P4). Raw margin weak (ρ=−0.296, right sign; gradnorm is the sharp functional — spearman(m₀,g₀)=−0.994). **Caveat: n=12 targets** | **DONE (MVP) → scale next** |
| **ViT+LoRA (data-type axis)** | 247474 (MVP), 256540 (scaled) | Single private image detectable in vit_tiny_patch16_224 rank-4 LoRA (blocks 0-2 qkv): MVP p=0.030 (N=6, K=10); scaled N=16/K=50 ALL 3 targets p=0.002, sens 1.13/1.24/1.52. MLP premise generalizes to a real ViT. Thesis-grade | **DONE** |
| **Fashion detectability (arm B on fashion)** | 246872 | — | **RUNNING** |
| A (typicality proxy version) | — | Superseded: the margin test answers it with g₀ as the (base-model-side) predictor; θ₀-independent-typicality variant folded into margin-at-scale if needed | absorbed |
| F — cross-dataset | — | Not run; REFRAMED through the margin lens (next phase §5) | pending |
| G — OOD injection | — | Not run; ABSORBED into the similarity axis as S3 (next phase §2), margin-framed | absorbed |
| **H — validation gate** | — | NOT run. Still the make-or-break | **pending — priority 2** |
| Phase-2 adapter atlas | — | Gated on the accumulated adapter population | pending |

---

## iv. NEXT PHASE (priority order; each with a signed pre-registered prediction + kill criterion)

Two co-top-tier tracks: **§1 margin-at-scale** (WHO leaks — scale the predictor) and **§2 the
similarity axis** (WHAT is leaked — instance vs concept). Then the H gate, ViT expansion, F.

### 1. MARGIN AT SCALE — the headline figure
Upgrade job 260171 from n=12 to **20–40 targets spanning the margin/gradnorm spectrum** (stratified by
g₀ decile, both classes, several contexts m). Also run on **fashion**.
- **Prediction (signed, pre-registered): ρ_spearman(sens, g₀) > +0.6**, positive in every context/class
  stratum; g₀ remains sharper than raw margin.
- **Kill:** ρ < +0.3 overall, or sign flips across strata ⇒ the n=12 result was small-sample; demote the
  base-model-predictor claim to an anecdote and the narrative spine loses its capstone.
- Output: the sens-vs-g₀ scatter with per-stratum ρ — intended thesis headline figure.

### 2. SIMILARITY AXIS — instance vs concept (user steer 2026-08-28; co-top-tier with §1)
The semantic layer ON TOP of the margin narrative. All arms so far swapped exact identities; these
three grade the swap by VISUAL/SEMANTIC distance. Similarity scores must come from a θ₀-INDEPENDENT
encoder (the M1 circularity rule): DINO via timm if available, else an ImageNet-pretrained vit_tiny
CLS embedding; raw-pixel L2 as the simple cross-check. Priority order S1 → S2 → S3.

**S1 — SIMILARITY-GRADED SWAP ("the distance dial").** Swap target T for replacements T′ at graded
visual distance: (a) parametric perturbations of T itself — brightness / contrast / small affine /
blur at increasing magnitude (color-spectrum shifts on color datasets); (b) nearest / median / far
same-class neighbors ranked by the encoder embedding. Measure whitened sensitivity vs embedding
distance d(T, T′).
- **Prediction (margin lens): sensitivity RISES with d(T,T′); near-duplicates are ~undetectable
  (p not significant at K=50)** ⇒ the adapter records the CONCEPT, not the instance — privacy
  statement: an attacker recovers "a smiling headshot", not WHICH person. If near-duplicates STAY
  detectable ⇒ instance-level memorization (the scarier claim; also publishable).
- **Kill:** sensitivity flat vs distance ⇒ similarity doesn't matter — the exact-swap numbers were
  already the whole story; close the axis after S1.

**S2 — TWIN SHIELDING ("two similar guys").** Put T AND a visual near-twin T″ in the set together;
measure T's swap-sensitivity with the twin PRESENT vs ABSENT.
- **Prediction (max-margin): the twin covers T's margin constraint ⇒ sensitivity(T | twin present)
  DROPS materially vs twin-absent** — "concept shielding": individual contributions of similar images
  are NOT resolvable from the adapter. Directly answers whether two similar training images can be
  separated.
- **Kill:** no drop (T equally sensitive with the twin there) ⇒ contributions of similar images ARE
  separable — instance-level attribution stands, shielding does not exist.

**S3 — OOD-STYLE DIGIT INJECTION (= old arm G, margin-framed).** Inject visually-similar-but-
different-style digits into an MNIST private set: **USPS via torchvision** (16x16 handwritten,
closest drop-in; EMNIST fallback — no web scraping needed). **PREDICT each injected image's
sensitivity from base gradnorm g₀ FIRST** (pre-registered, public base only), then measure — an
out-of-distribution test of the margin predictor.
- **Prediction: OOD digits sit high in the g₀ distribution and leak MORE; their measured
  sensitivities land on the in-distribution sens-vs-g₀ curve (within its CI)** — style-OOD has no
  effect beyond its g₀.
- **Kill / informative failure:** OOD points fall systematically OFF the curve ⇒ g₀ is not
  sufficient; distribution membership carries independent signal — itself a publishable
  qualification of the spine.

### 3. H — TRIANGLE-CLOSING VALIDATION GATE (unchanged from v1; still make-or-break)
Does whitened sensitivity predict an attacker-observable? Primary: **leave-one-out Feldman-Zhang mem
score** on the same records (behavioral, independent of Δ-parameters); secondary if feasible: a simple
loss-threshold MIA or reconstruction quality.
- **Prediction: high-sensitivity images have higher LOO mem scores; rank correlation > +0.4.**
- **Kill:** no positive association ⇒ everything above stays "parametric detectability", NOT privacy;
  all "leakage" language must be rewritten before any external claim.
- Without H, nothing in §iii may be called leakage in the thesis.

### 4. ViT EXPANSION (turn one thesis-grade point into a curve)
a) **Rank sweep on ViT** — prediction: absolute sensitivity grows strongly with r (the MLP's ~15x
   r=8→32 effect replicates directionally). Kill: flat in r ⇒ the rank-danger claim is MLP-specific.
b) **Margin-vs-leakage on ViT** — does the base-gradnorm predictor transfer? Prediction:
   ρ(sens, g₀) > +0.5 on vit_tiny. **If yes this is the strongest thesis claim** (public-base
   predictability on a real foundation-model architecture). Kill: ρ ≈ 0 ⇒ predictor is
   small-MLP-specific; report as a scoping result.
c) **Harder data**: CIFAR 2-class on the ViT; optionally vit_small. Prediction: detectability persists
   (p ≤ 0.01) at N=16/K=50, at lower sensitivity.

### 5. F — CROSS-DATASET COMPOSITION, margin-framed (G now lives in S3 above)
mnist+fashion mixed private sets. Same reframe as S3: foreign-dataset images are extreme large-g₀
points, so **predict from g₀ first, then measure**.
- **Prediction: mixed-in fashion images leak most, AND on the same sens-vs-g₀ curve** (no
  dataset-membership effect beyond g₀).
- **Kill / informative failure:** off-curve ⇒ composition carries independent signal (same reading as
  S3's kill).

### 6. PHASE-2 — adapter-space atlas (unchanged; after the battery)
Embed/cluster the accumulated ΔW=BA population (every arm saves its adapters). GL(r)-invariant
featurization (SVD subspace + spectrum); two-sided Bures–Wasserstein (C_out AND C_in) or
Grassmann+spectral; precomputed-metric clustering/UMAP only — never Euclidean on raw params.
Questions: do adapters cluster by composition? are mixed (arm-F) adapters ~linear combos of pure ones?
does per-image sensitivity correlate with position/spread?

### 7. PARKED
Architecture / width / depth sweep of the frozen base — needs new base checkpoints; do not start before
§1–4 are done.

---

## v. DESIGN LOCKS + DISCIPLINE (reproducibility — a fresh session must obey these)

### Metric usage lock
- 3-way cross-fit `whitened_metric.py` ONLY (the 2-way lives in the file solely as the regression
  fixture for the K-convergence self-test — never for results).
- K ≥ 50 headline / K=20 scouting; p ≤ 3–5; p,K FROZEN across any swept condition; {K,2K} gate;
  shrinkage floor; NaN-drop with counts; Σ-invariance + Gaussianity gates reported.
- The reporting rule of §ii applies verbatim to every new number.

### Arm-E construction lock (audit-hardened; reuse for any duplication-flavored arm)
- FIXED-PREVALENCE: k_max class-1 slots constant for all k; copies_k = context + [T]*k + donors[k:k_max];
  distinct baseline IDENTICAL across all k & targets ⇒ Σ frozen by construction (trained once per rank).
- NO analytic β=2 null; the empirical null is the rank contrast (r<N vs r>N).
- Headline β on DEBIASED sensitivity (d²_obs − d²_null); drop-counters + finite-d² assert before fitting.

### Data/base lock
- Base = MNIST-train-pretrained checkpoint; swap/duplication targets from MNIST-test ⇒ genuinely
  held-out, no target/base overlap. Fashion arms mirror this. ViT: vit_tiny_patch16_224 (timm 0.9.12),
  rank-4 LoRA on blocks.0-2 qkv (peft 0.7.1), frozen random head (verified trainable, fit ~2e-4).
- Hold T and LoRA capacity fixed within an arm (Carlini control); sweep them only deliberately
  (as arm-E's T-sweep did).
- Every run SAVES its adapters (ΔW=BA per layer) + metadata .pth — Phase-1 output IS Phase-2 input.
- When reconstructing measurement identity across scripts, IMPORT the arm's own construction functions
  and hard-assert saved metadata (the margin-test method) — never re-implement.

### Process discipline (unchanged)
- **Planner → auditor → executor**: every new arm design gets an adversarial audit pass before
  submission; audit resolutions folded into the design lock section here.
- **Stage-0 gates**: every arm ships with a tiny smoke config (`run_*_stage0_wexac.sh`) that must pass
  on WEXAC before the full submission.
- **WEXAC bsub ONLY — never run locally**, not even smoke tests. `python -u` in job scripts (LSF
  buffers otherwise). rsync `experiments/` before submitting.
- Artifact hygiene (this program's scars, see LESSONS_LEARNED 2026-08-27/28): a surprising trend in a
  whitened/normalized metric ⇒ suspect the DENOMINATOR first; K-non-convergence is a sufficient
  artifact disqualifier on its own; "grows with sample size" is only bias if a signal-free control
  grows too — run the null control before retracting OR believing.
- After every completed arm: update STATUS.md (headline + config + jobs) and LESSONS_LEARNED.md (if a
  pitfall) in the same turn; commit.

---

## Grounding (compact)
DP sensitivity: Dwork et al. 2006. Memorization: Feldman 1906.05271, Feldman-Zhang 2008.03703, Carlini
2202.07646, onion 2206.10469. Testing template: f-INE 2510.10510. Attribution (behavioral, for contrast):
Koh-Liang 1703.04730, TRAK 2303.14186, DataInf 2310.00902 (explicitly NOT used — behavioral ≠ parametric).
Adapter geometry (Phase-2): LoRA 2106.09685, task-arithmetic 2212.04089, model-soups 2203.05482,
EigenLoRAx 2502.04700, Bures-Wasserstein 2302.14618.
