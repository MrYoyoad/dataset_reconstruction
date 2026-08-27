# The measurement: whitened (Mahalanobis) sensitivity — ONE metric for the whole program

## Why (the problem with what we have)

Every dataset-sensitivity experiment (swap-one, subset-size, class-distribution, cross-dataset, per-model,
per-image) asks the SAME question: **how distinguishable is dataset D from D′ by looking at the adapter,
given that training randomness already jiggles the adapter?** That is a *detection* problem, and detection
has a canonical answer — which is NOT total energy.

- **Total-energy (Frobenius) SNR** `‖Δμ‖ / ‖noise‖` collapses direction: it treats a diffuse high-energy
  noise and a targeted low-energy signal as comparable, which they are not (the N=8 result: energy-SNR
  0.43 while the swap points ~entirely outside the noise's ~2 directions).
- **`frac_outside_noise`** is a brittle *binary* shadow of the right thing (in/out of a badly-undersampled
  subspace → saturates at ≈1 trivially).

Neither is the principled distinguishability. This doc defines the one metric that is.

## The metric

Model the adapter as a random variable over training seeds: dataset D gives mean **μ(D)** and covariance
**Σ(D)** (the seed-noise covariance in ΔW=BA space). For a data-change D → D′:

> **d²(D, D′) = (μ(D′) − μ(D))ᵀ Σ⁻¹ (μ(D′) − μ(D))** — the change's mean effect, WHITENED by the noise.

Report √d² as the sensitivity SNR. The seed-noise covariance Σ *is the natural (Fisher) metric of
adapter-space*; distances are measured in it, not in Frobenius. If the noise "wiggles the spectrum
symmetrically," Σ has large variance there and Σ⁻¹ down-weights it; if the swap pushes one specific
direction the noise leaves alone, Σ⁻¹ up-weights it. Whitening = discount diffuse noise, reward targeted
signal — the exact intuition, formalized.

## PRIMARY READOUT: the permutation / reseed NULL (the headline — NOT the raw analytic d̂²)

The plug-in `d̂²` is **biased UPWARD** (Hotelling-T², ~dim/(K−dim)) — exactly the direction that fakes
"recoverable," the same bias class as the S=64 q_eff over-count. **Do NOT analytically de-bias it.** Instead
report `d̂²` relative to a **NULL built by SHUFFLING which seeds are labelled "D" vs "D′"** (or
reseed-vs-reseed with NO composition change) and recomputing `d̂²` on each shuffle:

> **sensitivity = (d̂²_observed − mean d̂²_null), with p = fraction of null draws ≥ observed.**

This is the f-INE template: it **auto-corrects the plug-in bias** (whatever it is), needs **NO equal-Σ or
Gaussian assumption**, and yields an honest effect-size + significance in one shot. The analytic `d̂²` and
its whitened spectrum are **diagnostics** reported alongside; **the permutation null is the primary number.**

**This is THE debiaser, and it works at ANY K** — the null d̂² carries the IDENTICAL Hotelling bias (same K,
p, estimator, cross-fit; only the labels shuffled), so the subtraction is exact by construction. Implement
the null as **sign-flips on the K paired diffs `v_j`** (2^K ≈ 1M flips at K=20 → ample Monte-Carlo p-value
resolution — resolution is NOT the constraint). **Do NOT attribute the debiasing to cross-fit or frozen-p:
those solve DIFFERENT problems** (circularity and cross-N comparability, below).

## Why THIS metric (four equivalences — how we know it's not ad-hoc)

For the Gaussian + equal-Σ approximation, d² is simultaneously:
1. the **optimal-detector (Neyman–Pearson) SNR²** — the best any attacker can do at telling D from D′;
2. **2·KL(P₁‖P₀)** between the with/without-change adapter distributions;
3. the **Fisher information** the adapter carries about the change ⇒ by **Cramér–Rao it lower-bounds the
   estimation error of Δμ (the ADAPTER-space change)** — NOT pixel-MSE directly (S5): recovering the image
   needs pushing Δμ through the nonlinear data-latent Jacobian of the training map; d² bounds pixels only
   *via* that J. Don't state it as a direct reconstruction-MSE bound (the ssim-overclaim class).
4. the **f-INE / DP hypothesis-test statistic** — is the change separable from the training-randomness null.

And it **subsumes q_eff**: q_eff is the *thresholded spectrum* of this object (count whitened directions
with per-direction signal > 1). So the discrete-dataset case becomes continuous with all the Jacobian work.

**⚠ These four equivalences hold ONLY under equal-Σ Gaussian.** (S1) If Σ(D) ≠ Σ(D′), the optimal detector
is QUADRATIC (QDA) and pure Mahalanobis drops the KL covariance term → **test Σ-invariance first** (compare
eff_rank/spectrum of Σ(D) vs Σ(D′)); if they differ materially, use the symmetrized (Jeffreys) KL and
whiten the mean term by the POOLED Σ. (S2) The seed-noise here is **non-Gaussian** (heavy tails, the NaN
draws) — so keep the skew/kurtosis Gaussianity diagnostic (already in `jacobian_spectrum`) as a REQUIRED
gate; if it fails, d² is a heuristic SNR, not "the optimal detector." **The permutation null above is robust
to BOTH failures** — which is why it's the primary readout.

## One object, all experiments (the generalization)

`d²(Δμ, Σ)`. Only **Δμ** (which change) and **Σ** (which model/dataset's noise) vary:
- **image-dependence** → vary Δμ (which image swapped) and correlate d² with an independent typicality score;
- **subset-size (arm B)** → Δμ = per-image swap, plot d²(N);
- **class-distribution** → d² for minority vs majority slot;
- **model / dataset dependence** → both Δμ and Σ change; report d² per (model, dataset).
Because every result is the SAME d² in the SAME units, results become **comparable across conditions** —
which a pile of Frobenius norms never would be.

## Estimation — where ALL the risk is (col(J) lessons, sharpened by audit)

- **Δμ (signal):** paired-per-seed difference `v_j = ΔW(D, seed_j) − ΔW(D′, seed_j)` (init cancels per
  pair), Δμ̂ = mean_j v_j. Already in arm B.
- **K-FOLD CROSS-FIT is MANDATORY (M1) — it kills a DIFFERENT bias than the null does (circularity, not
  Hotelling).** NEVER whiten in Σ̂'s OWN reseed col-space chosen from the same seeds: that sets p = K−1 and
  measures Δμ̂'s spurious mass in the noise's least-sampled directions — **structurally circular, fakes
  "targeted signal"** (exactly what `frac_outside≈1` was doing). Fix: Σ + whitening subspace from one
  seed-fold, Δμ̂ from the disjoint fold. **Use K-FOLD (rotate the held-out fold, average — double-ML), NOT a
  single 50/50 split** — a single split HALVES effective K (Σ from K/2, Δμ from K/2); K-fold keeps each
  fold's Σ independent of Δμ yet uses all K for both roles across folds, recovering the power.
- **Keep p ≪ K.** Top-few whitened directions only (NOT K−1); p ≤ 3–5.
- **K sizing — BIAS is solved, POWER is the constraint.** The permutation null kills the bias at any K, so
  K=20 no longer fakes "recoverable." But it does NOT reduce d̂²'s VARIANCE → low power (you'd correctly
  avoid false positives but MISS real effects). So: **headline arms (arm-B's exponent, the 59/36 anchor
  re-check) need K ≥ 50 with K-fold cross-fit and p ≤ 3–5** (each fold's Σ-split ≳ 5–10·p). **K=20 is
  scouting-only** (K-fold, p ≤ 3) — never a published number. **Let {K,2K} decide empirically per arm:** if
  d̂², the null-mean, or eff_rank(Σ) still move materially K→2K, K is too small for that arm.
- **Shrinkage is the PRIMARY regularizer (S4), not subspace-to-noise-col-space.** Whiten by (ρI + Σ)⁻¹;
  choose **ρ by cross-validation against the permutation null** (the ρ minimizing null d̂² while preserving
  the observed signal), NOT a fixed grid. Any subspace cross-check uses a **SIGNAL-defined subspace** (top
  dirs of Δμ from a held-out split), cross-fit — never Σ̂'s col-space.
- **Report the whitened SPECTRUM + a q_eff-style count**, never a single in/out scalar. **{K,2K} adequacy**
  on eff_rank(Σ).
- **GATES before trusting the analytic d² (permutation null survives both if they fail):** (a) Σ-invariance
  Σ(D)≈Σ(D′), else Jeffreys-KL + pooled-Σ whitening (S1); (b) Gaussianity (skew/kurtosis), else label it a
  heuristic SNR (S2).
- **Freeze p and K across N (S3).** eff_rank(Σ) — hence p — moves with N; an N-varying estimator bias would
  contaminate arm-B's 1/N exponent. Hold p, K FIXED across N (or bias-correct identically per N), or the
  headline exponent is the estimator, not the data.
- **Gauge:** everything in ΔW=BA space (never raw B,A). **Two brackets:** known-init ⇒ Σ = (≈0) determinism
  floor ⇒ d → ∞ (upper bound); unknown-init ⇒ Σ = reseed covariance (privacy-relevant).

## ⚠ DOWNSTREAM: de-biasing may move the PUBLISHED q_eff (M2 — flag for the executor + the anchor re-check)

`q_eff` = the thresholded col(J)-restricted whitening = **exactly this subspace-Mahalanobis**, and the S=64
over-count WAS this bias. So the "collapse to q_eff" check is CONSISTENCY, not correctness — it agrees *by
construction* while both carry the same bias. **Consequence:** running the bias-corrected / permutation
estimator may **LOWER the published ABSOLUTE q_eff** (the 59/36 anchor, the roundB counts). The
**differential** results — the reversal, the gap 23→13→0, the iso ordering — are *differences* and are
largely robust to a common-mode bias, so they stand. But **before anyone quotes an absolute q_eff as "N
recoverable directions," re-run the anchor through the permutation/bias-corrected estimator.** State this in
the executor hand-off.

## Audit resolutions (yoado-34, round 1 — all folded in above)

- **Q1 plug-in bias → permutation null (primary), don't analytically chase Hotelling; cross-fit + p≪K;
  K≳5–10·p (K=20 only if p≤~3).** M1: never whiten in Σ̂'s own col-space (degenerate/circular — what
  frac_outside did).
- **Q2 → test Σ-invariance first, else Jeffreys-KL + pooled-Σ** (S1). Equivalences hold only under equal-Σ
  Gaussian; seed-noise is non-Gaussian → Gaussianity gate (S2).
- **Q3/Q4 → shrinkage primary; ρ by CV against the null; signal-defined subspace cross-fit** (S4).
- **Q5 → d² is the right cross-N population quantity, but FREEZE p,K across N or the estimator bias
  contaminates the exponent** (S3).
- **Q6 → run it, but it's CONSISTENCY not correctness; and de-biasing may LOWER the published absolute
  q_eff** (M2, above). Differentials robust; re-check the anchor.
- **S5 → CRB bounds Δμ (parameter), not pixel-MSE directly.**

## Then → executor (yoado-a2 knows the arm-B code)

Re-implement the measurement to report, for EVERY ablation, the same triple:
1. **PRIMARY: permutation-null sensitivity** — (d̂²_obs − mean d̂²_null) + p-value (sign-flip on the K paired
   diffs), **K-FOLD cross-fit** (Σ/subspace and Δμ̂ from disjoint folds, rotate + average), shrinkage-ρ CV'd
   against the null, p and K FROZEN across conditions. **K ≥ 50, p ≤ 3–5 for headline arms** (K=20 scouting
   only); {K,2K} as the per-arm adequacy gate.
2. **DIAGNOSTIC: the whitened spectrum + q_eff-count** (thresholded), with the Σ-adequacy ({K,2K}, eff_rank).
3. **GATES: Σ-invariance + Gaussianity** — reported; failing them downgrades d² to a heuristic SNR / Jeffreys-KL.
Reuse `snr_spectrum`/`q_eff_colspace`/`estimate_sigma_seed` (shrinkage+colspace+NaN-drop already there) — but
they do NOT yet cross-fit or permutation-null; **that's the new code.** Also: **re-run the q_eff anchor
(59/36) through this estimator** and report whether the absolute number moves (M2). The current
`frac_outside` job is one input to this design, NOT a conclusion.

## CORRECTION (arm-B post-mortem): cross-fit must be 3-WAY, and add K-convergence as a required gate
The M1 cross-fit as first specced (Sigma/subspace from one split, Delta-mu from a disjoint split) is
INSUFFICIENT: it cross-fits the numerator but the DENOMINATOR lambda is still measured along a subspace its
OWN split's samples helped define (winner's-curse coupling) -> lambda selection-biased small, d² inflates
and does NOT converge in K (arm-B: d²(N=64) 63->161 at K=50->100). REQUIRED recipe: **3-WAY disjoint split**
- (A) define U, (B) numerator Delta-mu.U, (C) denominator lambda along U - all from DISJOINT seed sets,
K-fold rotation over the three roles. AND add **K-convergence as a mandatory gate** (the reported statistic
must stabilize K->2K; non-convergence = artifact, mechanism-agnostic, on its own). The rank-based permutation
p-value remains the primary readout (floor-free AND K-stable); the magnitude is trustworthy only after a
3-way split that passes K-convergence.

---
## UPDATE 2026-08-27 — 3-way fix LANDED and VALIDATED
The "CORRECTION (arm-B post-mortem)" mandate below is now implemented in
experiments/dataset_sensitivity/whitened_metric.py. The estimator rotates over ordered triples of
disjoint folds: role A → subspace U, role B → numerator Δμ_B·U, role C → denominator λ (reseed variance
along U). Public API + 12 return keys unchanged. Permutation null (sign-flip) preserved (U from uncentered
SVD is sign-invariant, λ a fixed nuisance). Self-test: 4/4 PASS — (a) orthogonal→detect, (b) aligned→mask,
(c) null→flat, (d) **K-convergence**: over 60 synthetic datasets E[d²] drift 3-way +6.3% (≤15% gate) vs
old-2-way +44% (~1.5× inflation). The 2-way estimator is kept in the file ONLY as the regression fixture
for gate (d). Single-draw K-flatness is a knife-edge (competing finite-sample biases), hence the gate is a
population mean E[d²(K)], not one dataset.
