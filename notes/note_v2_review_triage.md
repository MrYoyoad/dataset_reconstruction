# Triage — external review of the LoRA working note (v2 / the 13-page Mac PDF)

**Date:** 2026-08-30 · **Reviewer:** external (read the 13-page note, the 26-page predecessor, the
rank-bound note, the review bundle/figures). **Triage:** every checkable claim below was verified against
the repo or the primary source. Verdicts: **CONFIRMED** (real, fix it), **PARTLY** (real but the root cause
differs), **REJECTED** (reviewer wrong — evidence given), **FORWARDED** (Mac-side PDF artifact, cannot
verify here).

---

## A. Verified against the primary literature

### A1. Jang: `r(r+1)/2 > KN` **is Jang's own stated condition** — our note under-attributes it
**Status: CONFIRMED — and the error is ours, in the reviewer's favour.** Downloaded arXiv:2402.11867v3 and
grepped the source. The abstract states verbatim: *"full fine-tuning (without LoRA) admits a rank-r solution
such that r(r+1)/2 ≤ KN. Second, using LoRA with rank r such that r(r+1)/2 > KN eliminates spurious local
minima."* The proof reaches it via a Sard-theorem dimension count (`dim A_s + dim R(S) < (m+n)(m+n+1)/2`).
Jang §2 also defines **"K = k for k-class classification, K = 1 for binary classification"** — our K=1
convention for the single-BCE-logit arm is *Jang's own*.

Three places say the opposite and are now wrong: `notes/thesis_note_v2.md` E2 ("Our extrapolation, not
Jang's stated bound"), `CLAUDE.md` (thesis overview), `notes/mac_handoff_brief.md` ERRATA #2. **Fixed in all
three.** What remains genuinely ours: the *informational/leakage* reading of that landscape threshold.

### A2. Jang is **not** squared-error-only
**Status: REJECTED (reviewer's hedge, inherited from arXiv:2605.03724).** Jang §2 assumes ℓ convex,
non-negative, twice-differentiable and states explicitly *"This assumption holds for the cross-entropy loss
and the mean squared error loss."* So CE is in scope.

### A3. "Rethinking the Rank Threshold for LoRA Fine-Tuning" (arXiv:2605.03724, 2026)
**Status: CONFIRMED as a real and relevant caveat.** It argues the `r(r+1)/2 > KN` Sard-form count is
conservative, and that replacing it with the non-symmetric LoRA-manifold dimension (`r(m+n) − r² > C*·KN`,
C*≈1.35) reduces the prescribed rank to **r = 1** for binary classification. This does not touch our
measurement, but it does weaken "the Jang scale explains the E2 reversal" — cite it beside Jang.

### A4. *Learning on LoRAs* venue
**Status: CONFIRMED (bibliographic).** OpenReview `cZOPrf5WLu` + an ICLR-2025 virtual page exist, and the
expanded "GL Equivariant Metanetworks for Learning on Low Rank Weight Spaces" appeared at **LoG 2025**.
Safest citation: arXiv:2410.04207 with "ICLR 2025 (workshop) / LoG 2025", not "ICLR 2025" bare.

### A5. MineGrad belongs in related work
**Status: CONFIRMED (and already analysed in-repo).** `notes/minegrad_analysis.md` + `CLAUDE.md` record it as
a **malicious-server** LoRA-gradient attack that breaks the DAGER `tokens ≤ rank` bound. It makes the
"low rank ≠ privacy" point directly while leaving our passive-released-endpoint threat model distinct — the
differentiation the reviewer suggests is exactly the one we already hold.

---

## B. Theory corrections — accepted

### B1. `rank(G) = rank(M)` is a K=1 statement
**Status: CONFIRMED.** For K outputs with neuron output vectors `v_k` and residuals `c_i`,
`∇_{w_k} ℓ_i = σ'(w_kᵀx_i)·(v_kᵀ c_i)·x_i`, i.e. **G = M ⊙ (Vᵀ C)** — a Hadamard product, not `D_v M D_c`.
The diagonal form is the scalar-output special case. Consequence for E2: the head/residual geometry can
create or destroy mixing rank independently of the gates, so K may enter the multiclass reversal through the
*data gradient*, not only through optimisation capacity. Note rewritten.

### B2. The frozen-gate rank theorem does not bound "any attacker, however strong"
**Status: CONFIRMED.** The proof studies `F_frozen(X) = G_*Xᵀ`; the alternatives `X'` it produces need not
satisfy the self-consistency `G(X') = G_*`, so `F(X') = F(X)` does not follow. Also, Jacobian rank deficiency
at a point is not local non-injectivity (`x ↦ x³`). `notes/identifiability_rank_bound.tex` was already careful
here; the compressed note lost the caution. World A rescoped to a **local first-order wall** under the stated
data metric and noise model. Wang–Lee–Lei (gradient identifiability for broad activations) added as the
counterweight.

### B3. Three meanings of "lower bound" were conflated
**Status: CONFIRMED.** An explicit attack's success is a lower bound on leakage; the Jacobian/Fisher spectrum
is an **attack-independent local channel diagnostic** under a stated metric and noise model — and the
Neyman–Pearson detector behind d² is *optimal*, not weak. Cover stance split into those two sentences.
Related: "directions on which even the weakest attacker has better-than-chance power" is wrong — any nonzero
mean shift is better-than-chance; **d = 1 is a chosen SNR threshold**.

### B4. `P_LoRA` is the one-step tangent, not what a T-step endpoint exposes
**Status: CONFIRMED (wording).** The released object is `Y = F_T(X, seed, recipe)`; `P_LoRA` explains its
tangent. Chain now stated explicitly: `Ω = GXᵀ → P_LoRA → F_T → J`.

### B5. `q_eff` is coordinate-system-dependent
**Status: CONFIRMED, and already on record internally** — `notes/whitened_sensitivity_metric.md` warns that
cosmetic reparameterisation inflated q_eff, and the review bundle notes on-manifold directions are more
collinear than random pixel directions. So `156/160` must be stated as *"156 of 160 directions of this
normalised local basis clear the threshold at ε=10"*, not "156 private directions survive".

---

## C. Number/figure defects — verified in the repo

### C1. E3 panel A does not plot what it says (worse than reported)
**Status: PARTLY — real defect, different root cause.** The reviewer noticed the panel's ~0.67 (relu/leaky)
and ~0.88 (hardswish/gelu) do not match the committed `figures/crux/feature_stability_vs_T.png` T=50
endpoints (sigmoid 0.96 … relu/leaky 0.51). Traced to source: those numbers come from **two different
columns of `results/rescored_tsweep_2026-08-29.csv`** —
- `ssim_norm` (relative fidelity): relu **0.673**, leaky 0.669, selu 0.662 …
- `feature_stability` at T=1: sigmoid **0.983**, hardswish 0.918, gelu 0.903, softplus 0.857 … relu 0.705.

So the kinked bars are `ssim_norm` and the smooth bars are `feature_stability`. The mix originates in
`notes/mac_handoff_brief.md` ("sigmoid/softplus highest (~0.98/0.86), kinked relu/leaky lowest (~0.67)").
**The E3 conclusion survives** — kinked are lowest on `feature_stability` at *every* T (T=1: 0.705/0.709;
T=10: 0.529/0.533; T=50: 0.51) — but the panel must be redrawn from **one column at one T**. Handoff brief
corrected so the Mac rebuild cannot repeat it.

### C2. E4's "±0.15" is a precision gate, not an equivalence band
**Status: CONFIRMED.** The committed figure states it correctly ("pre-reg PASS bar (rho>+0.6, CI hw<0.15)";
"CI half-width 0.189 > pre-registered 0.15"). The note's "passes the pre-registered ±0.15" is the ambiguous
phrasing that licensed the PDF's shaded band around zero. Note now says: PASS bar = ρ > +0.6 **and** CI
half-width ≤ 0.15; at n=24, ρ=+0.777, CI [0.53,0.91], half-width 0.189 → fails on **precision**, not on
effect size.

### C3. `q_eff = 156/160` is the top of a curve
**Status: CONFIRMED.** `STATUS.md:257` — binary q_eff = **51 / 117 / 150 / 156** at ε = 0.3 / 1 / 3 / 10.
The note now quotes the curve. Plus `notes/whitened_sensitivity_metric.md` §"DOWNSTREAM" explicitly says not
to quote an absolute q_eff before re-running the 59/36 anchor through the bias-corrected estimator — carried
into the note as a standing caveat.

### C4. Bridge headline: 0.951 is best-epoch
**Status: CONFIRMED, and it violates our own rule.** `notes/next_experiment_plan.md:48,197` — m=8 final
**0.930**, best-epoch 0.951, with the instruction "Report converged, not best-epoch". Note now leads with
0.930 (best checkpoint 0.951 in parentheses). Also adopted the reviewer's better metric: error *projected
onto the image-distinguishing subspace*, not global cosine.

### C5. E7 "extraction gap" bar chart mixes metrics
**Status: CONFIRMED.** SSIM (full-gradient ceiling ~0.99), `ssim_norm` (0.57 → 0.27), gradient cosine and
three ViT faces on one axis is not a metric ladder; only the within-metric N=4→N=10 drop is quantitative.
Note now separates them and flags the figure for the rebuild.

### C6. E6 corrected artifacts are not archived
**Status: CONFIRMED — verified by `git ls-files`.** `atlas_analyze.py` writes **only a PNG**; the corrected
`+0.989 CI [0.973, 1.005], G=30` exists solely in `scripts/wexac_logs/atlas_analyze_838868.out`, which
`.gitignore` excludes (`*.out`), and `figures/atlas/atlas.png` was untracked. Fixed: stdout archived to
`results/atlas/atlas_analyze_838868.txt`, figure committed, and the analysis script now writes a JSON
sidecar. The CI upper bound >1 is a Normal-approximation artifact — switch to a bounded bootstrap.

### C7. E5 "diameter" / "same resolution"
**Status: CONFIRMED.** `d*(0.1)` is a threshold crossing along a coarse replacement ladder (many reads are
interval-censored), not a set diameter; and geomean 1.02 / median 0.86 with 4-of-6 narrower is *no stable
resolution difference at n=6*, not equality. Both rewritten; per-target ratios to be plotted individually.

### C8. E6 gauge wording
**Status: CONFIRMED.** `BA = (BR)(R⁻¹A)` says the raw factors carry non-identifiable coordinates and the
product does not. It does **not** predict that different seeds/recipes yield the same `BA`. Our observation
is *consistent with* the gauge reading, not forced by it. Also "causal identification" downgraded to
"out-of-sample composition predictability after nuisance adjustment for recipe".

---

## D. Rejected

### D1. "binary is labelled nc=2 but the prose substitutes K=1 — possible error"
**Status: REJECTED.** `experiments/jacobian_spectrum.py:304` — `num_classes <= 2` takes the
`binary_cross_entropy_with_logits(out.view(-1), y)` path: **one logit**. And Jang defines K=1 for binary
classification (A1). The note is correct; only the figure legend ("nc=2" = class count) is ambiguous → add a
one-line footnote to the rank-sweep figure rather than change the number.

### D2. "kink vs smooth is the wrong grouping" — accepted as a *refinement*, not an error
Hardswish (non-smooth, weak) and SELU (C¹, strong) do break a naive kink/smooth split, and the note already
says "two-cluster, not monotone". The right mediator is gate **diversity/conditioning**, for which we already
have direct evidence the note under-uses: `eff_rank(M)` at N=10 — relu 6.37 ≈ leaky 6.33 ≫ selu 3.39 > gelu
2.91 > mish 2.39 > silu 2.34 > softplus 1.73 > sigmoid 1.19 (`STATUS.md:1518`), which tracks the measured
leakage order, plus the softplus-β dial (β 0.5→50 moves eff_rank(M) 1.40→5.30). Folded into E3.

---

## E. Forwarded (Mac-side PDF, not verifiable here)
- Page 10's empty "full-gradient reconstruction ceiling" box. `figures/meeting/positive_reconstruction_gallery.png`
  exists and renders; the placeholder is a build defect in the Mac PDF.
- Page 7/8 panel redraws (C1, C2), page 10 bar-chart split (C5): specs added to the handoff brief.

---

## F. Research asks (not defects) — ranked, folded into the plan
1. **E3 mediator audit (P0).** Compute M, the multiclass `G = M ⊙ (VᵀC)`, and the J spectrum for every
   activation on the free-c ladder (hardswish included); test whether gate-column separation / σ_min /
   condition number explains the leakage ranking after controlling for weight change.
2. **Manifold-coordinate experiment (P1).** Re-run the calibrated-secret ruler with random pixel directions
   vs local-PCA tangents vs generator-latent tangents, with a radius sweep validating linearity.
3. **Same-cell World-B test (P2).** One N=4 and one N=10 cell; measure `q_eff(ε)`, near-truth-init inversion,
   random-init inversion, and the disjoint-adapter baseline *in that same cell*.
4. **E6 exemplar-controlled zoo (P3).** Same digit classes/counts/recipes, different exemplars — the
   instance-vs-content question. (`zoo_bank_samedigits.pth` exists; the `--same_digits` analysis arm is the
   gap flagged in the last handover.)
5. **Dense E2 rank sweep (P4).** r ∈ {10,12,13,14,15,16} — only after the multiclass mixing rewrite, and
   citing arXiv:2605.03724 beside Jang.

## G. Reviewer's proposed spine — adopted
`Ω = GXᵀ` (gradient skeleton) → `P_LoRA` (LoRA tangent filter) → `F_T` (trajectory) → `J` (measured local
channel), with the gate factorisation demoted from "the theorem that explains everything" to "the
interpretable local skeleton that shapes J". Section 1 of the note restructured accordingly.
