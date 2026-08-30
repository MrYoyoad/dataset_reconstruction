# Reconstructing Private Data from Released LoRA Adapters

**Working note — the mechanism, the ruler, and where each experiment stands.** Prepared for Yoad Oxman · v2 (tightened & corrected), 2026-08-30.

Stance: observe, don't conclude. Two different kinds of number appear here and they must not be merged. **Attack results** (reconstructions, control margins) are lower bounds on leakage: our attacker is the weakest one (prior-free, adapter-only, per-image), so what it gets, a stronger attacker also gets. **Ruler results** (J, d², q_eff) are not that — they are an *attack-independent local channel diagnostic* under a stated data metric and noise model, and the detector behind d² is the **optimal** (Neyman–Pearson) one, not a weak one. Neither number is a limit on what a stronger attacker could do.

---

## The idea in one paragraph

Fine-tuning moves the weights by a sum of per-image gradients, and every image enters that sum weighted by how each neuron's activation reacts to it. Whether the private images can be pulled back out of the released adapter is decided by the **rank and conditioning of that weighting**. One mathematical object — the gate-weighted update — generates all seven experiments; each measures a different part of it, read through one honest ruler.

---

## 1. The one object — the gate matrix

For a one-hidden-layer net, the first-layer gradient factors into a **mixing matrix times the data**:

**Ω = G·Xᵀ = Σᵢ gᵢ xᵢᵀ,  with G_ki = σ'(⟨w_k,x_i⟩)·(v_kᵀ c_i),  i.e.  G = M ⊙ (Vᵀ C).**

Here **M_ki = σ'(⟨w_k, x_i⟩)** is the gate: how open neuron k's valve is on image i; **v_k ∈ ℝ^K** is neuron k's output vector and **c_i = ∇_f ℓ(f(x_i), y_i) ∈ ℝ^K** the loss residual. Each neuron hands the attacker one weighted mixture of the training images; the number of **genuinely different** mixtures is rank(G).

**Scalar output is the special case.** For K=1 the Hadamard product collapses to **G = D_v·M·D_c**, and then rank(G) = rank(M) = ρ whenever the diagonals are full-rank (at a KKT point non-support-vectors have c_i=0 and dead neurons kill rows, so the count is over support-vector columns). **For K>1 this is false in general:** the head/residual geometry VᵀC can create or destroy mixing rank independently of the gates. That matters for E2 — the class count K may enter leakage through the *data gradient*, not only through optimization capacity.

Columns of M are images, rows are neurons: what separates two images is that their **gate columns** are not near-parallel. ρ = N is **necessary** in the frozen-gate channel (below); the data-generated gate code is what could supply sufficiency. The activation enters this whole object **only through σ'**; that single fact drives the activation crux (E3).

**What the rank argument does and does not prove.** With the gates *frozen* at G₀, rank(G₀) = ρ < N leaves a d(N−ρ)-dimensional kernel of the map X ↦ G₀Xᵀ. That is a statement about the **frozen-feature / first-order channel** — not an impossibility theorem for the real map. Most alternatives X′ in that kernel fail self-consistency (G(X′) ≠ G₀), so F(X′) = F(X) does not follow; and rank deficiency of a Jacobian at a point is not local non-injectivity (x ↦ x³ is injective with a vanishing derivative). Wang–Lee–Lei (AISTATS 2023) prove a *single* gradient can identify training data for broad activations, using tensor structure a frozen-mixing argument does not see. So: read the rank bound as an **interpretable local obstruction**, never as "invisible to any attacker, however strong". The full nonlinear/bilinear identifiability problem stays open — `notes/identifiability_rank_bound.tex` states this correctly; the compressed note had lost it.

Two refinements make this measurable on a real adapter:

- **Conditioning, not rank.** Two mixtures can be independent yet nearly parallel; inverting them amplifies noise. So we report a **spectrum and a noise-relative count**, never a bare rank.
- **The observable is the adapter, not the gradient.** A single LoRA step exposes not Ω but a linear operator applied to it, **P_LoRA(H) = BBᵀH + HAᵀA** (a self-adjoint PSD operator, **not** an orthogonal projection — P²≠P). At standard init B=0, so the first step exposes only H·AᵀA — the observation lives in the row space of a random A. This is why the seed shows up in the raw factors, and E6 confirms it empirically (raw (B,A) cluster by seed, ΔW does not). **This is a one-step tangent statement.** A released T-step endpoint is Y = F_T(X, seed, recipe); P_LoRA explains what shapes it, it is not itself the observable.

**The chain, in order of what is real:** Ω = GXᵀ (the gradient skeleton) → **P_LoRA** (the LoRA tangent filter) → **F_T** (the whole training trajectory) → **J = ∂Y/∂a** (the measured local channel). J is the primary object; the gate factorization is the interpretable skeleton that explains what shapes it — gate geometry, per-example residual/head geometry, LoRA's low-rank tangent, and the trajectory.

Scope of the "iff": if G were free, rank gives only the row space, not the images (any remix (GR⁻¹)(RXᵀ) matches Ω). What breaks the ambiguity is that G is **generated by the data** through σ'(⟨w_k,x_i⟩) at known θ₀ — combinatorially rigid for kinked activations (discrete gate codes), soft for smooth ones. This scopes the claim and sharpens E3.

---

## 2. The ruler — what we measure, and why it's trustworthy

Hide N images, nudge each along k directions; the secret is the vector **a** of all Nk coefficients. **The basis is part of the threat model, not a detail:** x_i(a_i) = x_i⁰ + U_i a_i, and q_eff counts directions *of that basis*. Random pixel directions, local-PCA tangents and generator-latent tangents give different answers (on-manifold directions are more collinear from the adapter's view), and any rescaling of U changes the count. So every count is quoted as "of this normalised local basis", never as "private directions" in the abstract. The flattened adapter is **Y(a)**. The one object is **J = ∂Y/∂a** — a second derivative propagated through all T training steps, computed exactly by double backprop (checked vs finite differences to ~8 decimals; failures discarded). From J, three quantities in increasing honesty:

| quantity | what it is | what it tells you |
|---|---|---|
| **r_J** (hard rank) | count of non-tiny singular values of J | how many secret directions are recorded at all (noise-free) |
| **d²** (whitened sensitivity) | (Δμ)ᵀ Σ⁻¹ (Δμ), whitened by seed noise | detectability = 2·KL = best-detector SNR² |
| **q_eff(ε)** (recoverable count) | #{ i : ε·νᵢ > 1 } on col(J) | how many directions clear the noise floor at budget ε — what you can really recover |

The noise floor **Σ_seed** is measured: rerun the fine-tune S times at the same secret with different seeds. In detection language this is a **Fisher-information / Neyman–Pearson** object (Hannun–Guo–van der Maaten): d² is the SNR² of the *optimal* detector under the local equal-covariance Gaussian model. So it is an **attack-independent channel diagnostic**, not the performance of our weak attacker — and ε·ν = 1 is a **chosen SNR threshold**, not the boundary between chance and information (any nonzero mean shift is discriminable slightly better than chance). What makes our *attack* numbers lower bounds is that the attacker is weak; what makes q_eff conservative is separate: it is measured only inside col(J), and observing less can only lower Fisher information.

**The honesty gates:**
- **Enough seeds.** Σ_seed is r_J×r_J → S ≳ 4·r_J, with q_eff stable from S to 2S. (An **empirical adequacy rule** validated by the S→2S check, not a statistical theorem.) The retracted "97 directions" broke it: S=64 ≪ r_J=160. Round-B uses S=640–1280.
- **Permutation null, not formula.** The plug-in d² is biased up; report against a seed-label-shuffle null with a p-value.
- **Three-way cross-fit.** Subspace, signal, noise floor from disjoint seed folds (drift over K fell +44% → +6.3%).
- **Leakage is r_J / q_eff, never eff_rank** (entropy "effective rank" reads backwards).
- **Always gate on the trivial baseline** (dataset-mean image, same-class control margin, weight-change).

**Report the curve, not its maximum.** q_eff is a function of the budget: at the canonical converged cell (N=20, T=1000, S=1280, 160 directions) binary gives **51 / 117 / 150 / 156 at ε = 0.3 / 1 / 3 / 10**. Quoting "156/160" alone reads as "almost everything survives" when it is the ε=10 end of a curve that starts at 51.

Honest limits: (i) d²=2·KL and the recovery reading of q_eff assume Gaussian seed noise and local linearity of J — q_eff certifies **local** identifiability, not global invertibility; (ii) `notes/whitened_sensitivity_metric.md` (§DOWNSTREAM) flags that the bias-corrected estimator may **move the absolute counts** — the 59/36 anchor is due a re-run before any absolute q_eff is quoted as a headline. Differentials and the permutation-null detection claims are robust to a common-mode bias; absolute counts are provisional.

---

## 3. The seven experiments

Each is one probe of Section 1, read through the ruler of Section 2.

### E1 · Controlled secret — the decisive minimal test  · toy confirmed, scale-up open
Plant a known vector secret inside realistic image variations; ask which coordinates come back. The ground truth is exact, so the prediction is quantitative: **per-direction recovery error crosses 1 exactly at ε·νᵢ ≈ 1**. A small run confirms this end-to-end. It is the calibration that licenses every q_eff number elsewhere. Open: scale-up.

### E2 · The (N, r, L) phase diagram — is the boundary spectral?  · rank slice done, surface open
The claim: reconstruction fails where the **spectrum** says (q_eff drops below the private dimension), not where N or rank r says. A rank-r update is **not** "r numbers" (the rank-r manifold has dimension r(m+d−r)), so "rank r ⇒ ≤ r images" is a fallacy.

![E2 · The multi-class "leaks fewer" effect is a low-rank phenomenon: the gap closes 23→13→0 across r=8/16/32 and vanishes once r is large enough.](figures/rank_sweep/rank_sweep_headline.png)

**Citation, corrected twice (2026-08-30, verified against arXiv:2402.11867v3).** The K-dependent form is **Jang, Lee & Ryu's own stated condition**, not our extrapolation as earlier drafts said: their abstract reads *"full fine-tuning admits a rank-r solution such that r(r+1)/2 ≤ KN … using LoRA with rank r such that r(r+1)/2 > KN eliminates spurious local minima"*, proved by a Sard-theorem dimension count. Their §2 also defines **K = 1 for binary classification, K = k for k-class** — so our single-BCE-logit arm is K=1 by Jang's own convention (the figure's "nc=2" is a class count, not K). Their loss assumption (convex, non-negative, twice-differentiable) covers cross-entropy. What is **ours** is only the *informational* reading: transporting a landscape threshold into leakage space.

For N=10: binary (K=1) needs r(r+1)/2 > 10, cleared by r=8/16/32 alike (binary q_eff flat ~58–60); 10-class (K=10) needs r(r+1)/2 > 100, so r=8 (36 < 100) sits below and r=16 (136 > 100) above. **How far this goes:** we have one point below the threshold, one just above, one far above — that is *consistent with* a transition at the K·N scale, not a demonstration that the gap closes as r crosses it. Two things must land before the mechanism claim: a dense sweep (r = 10,12,13,14,15,16), and the multiclass mixing rewrite of §1 (for K>1, G = M ⊙ (VᵀC), so K may act through the data gradient, not only through capacity). Counterweight to cite alongside: *Rethinking the Rank Threshold for LoRA Fine-Tuning* (arXiv:2605.03724, 2026) argues the Sard-form count is conservative and, via the non-symmetric LoRA-manifold dimension, drops the prescribed rank to r=1 for binary classification.

### E3 · The activation crux — Gal's top ask  · MNIST done, dataset-dependence open
The most direct probe of Section 1: the activation enters only through σ' = the gate. A step-like σ' gives each image a crisp near-binary gate code → **well-separated gate columns** (images are columns of M, neurons are rows), high ρ, mixtures separate. A smooth ramp blurs neighbouring columns → M collinear.

**"Kink vs smooth" is the shorthand, not the mechanism.** It fails at the edges: hardswish is non-smooth yet leaks weakly, SELU is C¹ yet leads. The measured mediator is gate **diversity/conditioning**, and we already have it directly: at N=10, eff_rank(M) = relu 6.37 ≈ leaky 6.33 ≫ selu 3.39 > gelu 2.91 > mish 2.39 > silu 2.34 > softplus 1.73 > sigmoid 1.19 (STATUS.md:1518), which tracks the measured leakage order; and the softplus-β dial moves eff_rank(M) 1.40 → 5.30 as β goes 0.5 → 50, dragging the whole frontier with it. That is the mechanism evidence — much stronger than "a kink makes a crisp code".

![E3 · The dissociation: the smoothest activations are the best linearizers, yet the kinked ones leak ~5×. "How linear the fine-tune stays" does not drive leakage.](figures/crux/freec_ladder_ranking.png)

**The dissociation (the money result).** The prospectus chain was "smoother → stays linear → cleaner update → more leakage." It breaks at one joint: kinked activations have the **worst** linearization fidelity yet **leak the most**: on the headline control-margin metric the kinked cluster leads ~5× (~0.47 vs ~0.09), yet Spearman(feature-stability, control-margin leakage) ≈ 0 (−0.06) — fidelity does not predict leakage. (Metric-scoped: on ssim the correlation is mildly positive, +0.08 to +0.28 across subsets, so the clean dissociation is stated on the control-margin metric.) The mechanism is a clean two-term split — and the terms are different kinds of object, which is why they move oppositely. **Informativeness (kinked wins)** is a static property of M: a step-like σ' gives each image a crisp near-binary gate code → distinct rows → high rank ρ → the mixtures separate. **Linearization fidelity (smooth wins)** is the continuity of the gate drift dM under fine-tuning: a smooth σ has bounded σ'' → small continuous dM → features barely rotate; ReLU is frozen within a region but jumps discontinuously at every kink-crossing, and those jumps break fidelity where training crosses them. So fidelity tracks the continuity of dM (smooth), informativeness the geometry of the static M (gate-diverse) — two terms, opposite directions. Caveats: two-cluster, not monotone; MNIST so far (Flowers band pending); small n.

**⚠ Figure defect (fix before circulation).** The designed PDF's left panel labels its bars "feature stability at T=50" but mixes two different columns of `results/rescored_tsweep_2026-08-29.csv`: the kinked bars (~0.67) are `ssim_norm` (relu 0.673, leaky 0.669), the smooth bars (~0.86–0.98) are `feature_stability` at T=1 (sigmoid 0.983, hardswish 0.918, gelu 0.903, softplus 0.857). The committed `figures/crux/feature_stability_vs_T.png` T=50 endpoints are sigmoid 0.96 … relu/leaky 0.51. **The conclusion is unaffected** — kinked activations are lowest on feature_stability at *every* T — but the panel must be redrawn from one column at one T. Preferred replacement: three linked panels, *feature stability over T → gate spectrum (eff_rank M) → measured leakage*, which states the dissociation as "smoothness predicts the dynamics' linearity, static gate geometry predicts sample separability, and these are empirically distinct".

### E4 · Who leaks — the g₀ predictor  · strong at n=12, indeterminate at n=24
Read Ω = Σᵢ gᵢ xᵢᵀ: image i enters with coefficient gᵢ. The **base-gradient-norm g₀ = ‖gᵢ‖** at the public model predicts which image leaks — before seeing the adapter.

![E4 · Per-image leakage tracks g₀, but it is a low-g₀ effect: +0.88 in the low-g₀ tercile, collapsing to −0.12 at high g₀.](figures/margin_at_scale/f3_margin_who_leaks.png)

r_s = **+0.857 at n=12**; **+0.777 at n=24**, graded indeterminate (95% CI [0.53, 0.91]; permutation p=1e-4). **The pre-registration is a precision gate, not an equivalence band:** PASS requires ρ > +0.6 **and** CI half-width ≤ 0.15. At n=24 the half-width is 0.189, so it fails on *precision* — the estimate is nowhere near zero (a ±0.15 band drawn around zero, as in the designed PDF, states a different rule and must go; draw the CI instead, annotated "required half-width ≤ 0.15, observed 0.19"). Transfers to full-FT (r_s ≈ 0.83, n=6). It **saturates** because gᵢ carries the loss residual cᵢ, which decays as the image is fit — g₀ over-predicts fast-fit images. Concrete next tests: the **integrated / mid-trajectory** gradient norm (the individual-privacy-accounting quantity), and the **whitened** predictor ⟨gᵢ, Σ⁻¹gᵢ⟩ — which should also repair the one counterexample (USPS digits have higher g₀ yet leak less: g₀ is a magnitude, but a direction poorly aligned with the noise never clears the floor). g₀ is **not** a score an attacker computes from the public model alone — you need the candidate image to compute its gradient. Frame it as a **pre-release per-record risk score** (curator-side, or a membership-candidate score). For Gal: g₀'s point correlation exceeds the max-margin dual λ's, **0.857 vs 0.538** (n=12) — but that is two point estimates, not a demonstration that one predictor beats the other; a paired bootstrap/permutation comparison is required before the claim, and the λ provenance still has to be logged. Present it as the trajectory view refining the endpoint view, not defeating it.

### E5 · Full-FT vs LoRA — the valley  · n=6, target-dependent
Does full fine-tuning remember each image more sharply, or just record more? The distinction is H (full gradient) vs P_LoRA(H). Resolution is the **valley width d*(0.1)** — the *threshold distance along the chosen replacement ladder* at which weight-space distinguishability exceeds 0.1. It is not a set diameter, and many reads are interval-censored by the coarse grid.

![E5 · Full-FT records ~5× more signal per image (left) but at ~the same resolution (right): more signal, not a finer memory.](figures/fullft_valley/fig_valley_ladder.png)

Full-FT imprints **~5× more total signal per image** (removal footprint — target-median; per-target ~3–6×; the same targets rank alike under both regimes, ρ=+0.94). On resolution we see **no stable difference**, which is not the same as equality: valley-width ratio geomean **1.02**, median **0.86**, narrower on 4/6 targets, wider on 2/6, at n=6 with censored intervals. Plot all six target ratios with their interval uncertainty — never one geomean dot plus one median dot. The noise-free Jacobian agrees independently. Keep the **magnitude** claim (5×) separate from the **resolution** claim. Guards: B1 dimension-invariance PASS; B2 ε-noise vs SGD-noise diverge → read qualitatively. This sets up E7 — with the honest gap named: more *signal energy* is not yet more *identity-bearing* signal, so "there to be inverted" is a hypothesis this experiment motivates rather than establishes. (n=6, exploratory — resist quoting the 4/6-vs-2/6 split until scale-up lands.)

### E6 · The composition atlas — a real second channel  · ≥content-level recovery, graded (concept-not-instance)
Does the adapter betray what it was trained on (which content, how many, what balance), not just its recipe? LoRA has an exact gauge symmetry BA = (BR)(R⁻¹A): the raw factors carry non-identifiable coordinates, the product does not. **What the gauge does not say:** it does not predict that different seeds or recipes produce the same ΔW — different optimization paths can produce genuinely different products. Our finding (raw factors cluster by seed, ΔW does not) is *consistent with* the gauge reading, not forced by it.

![E6 · ΔW clusters cleanly by composition and smears under every nuisance; the raw factors reorganize by seed.](figures/atlas/atlas.png)

Per-seed factorial zoo (169 converged adapters of a 180-cell grid — 5 compositions × 3 activations × 2 learning-rates × 6 seeds; 11 non-converged dropped) on a shared base: ΔW clusters **perfectly by composition** (ARI +1.00, p<0.001) and is blind to init/lr/activation (ARI ≈ 0); the raw (B,A) instead cluster by **seed** (+0.55) — the gauge-contrast, confirmed. Composition is recoverable from ΔW **above the fitted-recipe baseline**: cross-fitted held-out acc-diff = **+0.989, 95% CI [+0.973, +1.005]**, G=30 (cluster-robust; the >1 upper end is a Normal-approximation artifact on a bounded quantity — switch to a bounded bootstrap). This is out-of-sample composition predictability after nuisance adjustment for recipe, **not** a causal identification claim. The 5 compositions are 5 distinct **digit-subsets** under a shared binary task (comp0={1,6,7,8}, comp3={0,1,4,9}, …), so +0.989 recovers **which digit-subset = ≥ content/concept-level** (which digits were present). Content is the honest **floor** — a stronger instance-level reading is possible but **untested**. The channel records the content (which digits), NOT the specific instance: graded from +0.989 (content) down to ~0 for single-image swaps (arms 0.03–0.07). OPEN: a true instance-level test needs a zoo where compositions share digit content but differ in exemplars. Scope: MNIST, N=4, rank-8, population (stronger-than-weakest) attacker; committed (ab9eb99), not a claim. (A first-pass +0.00/CI[0,0] was a cross-fit fold bug — a whole composition isolated into the test fold — since fixed.) **Provenance, now closed:** `atlas_analyze.py` wrote only a PNG, so the corrected numbers lived solely in a gitignored LSF log; job 838868's output is archived at `results/atlas/atlas_analyze_838868.txt`, `figures/atlas/atlas.png` is committed, and the script now writes a JSON sidecar. **Cite and differentiate [Learning on LoRAs, Putterman/Lim et al., arXiv:2410.04207 — ICLR 2025 workshop; expanded version at LoG 2025, not ICLR main]** (GL-equivariant processing of LoRA weights to predict fine-tuning data attributes and membership — our exact gauge + channel): we ask whether composition is **forced** into ΔW above a recipe baseline (variance decomposition + cross-fit), not whether a probe finds something.

### E7 · Robust adapter-only inversion — the open milestone  · World B
Turn presence into pixels: minimize ‖Y − F(θ₀, x̂)‖ over candidate images (F = the whole fine-tuning map). Two framing facts: real images and their gradients live on low-dimensional manifolds (so a globally lossy adapter can be informative on the set that matters), and **a decoder supplies a prior, not measurements** — leakage from a prior is not leakage forced by the weights.

![E7 · The full-gradient ceiling — recognizable reconstructions when the true ΔW is known. This is the upper bound, NOT the adapter-only attack.](figures/meeting/positive_reconstruction_gallery.png)

Full-gradient reconstruction **works** (SSIM up to ~0.99 on MNIST/CIFAR/Flowers; ViT faces return structure at 0.38/0.26/0.52). The gradient-bridge decoder reaches **0.930 cosine converged** (best checkpoint 0.951 — report converged, per our own rule). Direct weight inversion recovers small-N (ssim_norm 0.57 at N=4) but hits a **superposition wall** (0.27 at N=10); the N=4→N=10 drop is the signal, the absolute is ssim_norm (mean/std-matched, inflated).

**Do not put these on one axis.** SSIM, ssim_norm, a gradient cosine and three ViT faces at different N, datasets and observables are not a quantitative ladder — only the within-metric N=4→N=10 drop is. The designed PDF's bar chart must be split into (i) the full-gradient gallery, (ii) adapter-only inversion vs N in one fixed metric with its control, (iii) bridge quality separately.

**World B is the working hypothesis, not yet a finding.** High q_eff, a working full-gradient ceiling, a 0.93 bridge cosine and an N-degrading direct inversion come from *different cells* (different N, datasets, activations, observables, coordinate bases). The clean test is one configuration measured four ways — q_eff(ε), near-truth-init inversion, random-init inversion, disjoint-adapter baseline. If q_eff is high there, local init succeeds, global init fails, and unrelated adapters produce nothing, then the extraction gap is isolated and Worlds A/B/C become empirical categories rather than labels.

**Better bridge metric.** Global cosine is generous in a huge vector space: 0.93 can still leave large error, and what matters is whether the residual lies in the directions that distinguish images. Report the error projected onto the image-distinguishing subspace, ‖P_col(J_g)(ĝ−g)‖ / ‖P_col(J_g) g‖, or weight it by the inverse problem's singular directions. Four controls keep it honest: (i) **disjoint-adapter subtraction** (leakage above the prior — excludes World C); (ii) staged Jacobians across activations (E3 through reconstruction); (iii) local-vs-global init (optimization failure ≠ non-identifiability); (iv) render the weakest/strongest singular directions as image edits.

---

## 4. Three worlds — which reality are we in?

Every negative reconstruction is ambiguous. Naming the worlds forces each experiment to say which it rules in or out.

| world | what's true | what you'd see | the move it forces |
|---|---|---|---|
| **A — local first-order wall** | in the stated data metric and noise model, private directions are absent or below the usable local SNR floor | q_eff low; even the full-gradient ceiling fails; Jacobian rank-deficient in data directions | measure it with the ruler — a **scoped, local** statement, never "no attacker can recover" |
| **B — extraction-limited** | information present, decoder can't reach it | q_eff high, full-gradient works, adapter-only pixels blurry | build a better decoder; bound the gap to the floor. **E7's working hypothesis — pending the same-cell test.** |
| **C — prior hallucination** | pixels come from the decoder's prior, not the weights | the same "recovery" appears against a matched unrelated adapter | subtract the disjoint-adapter baseline — always |

**The discipline in one line:** World A is *measured* by the ruler (attack-independent); World C is excluded by the disjoint-adapter control (prior-independent); only what survives both is a genuine World-B leak. **A is scoped and local** — it says no local, per-image, linearized information survives under Gaussian seed noise, in a named coordinate basis. It is not a guarantee against priors, higher-order effects, a different basis, or the composition channel (E6 is exactly such an escape). Stated that way it does not contradict the two-kinds-of-number stance on page 1. And note the asymmetry: A and C are read off controls, whereas **B currently rests on evidence pooled across cells** — the same-cell test in E7 is what would make it a finding.

---

## 5. Telling Gal

**His lens** is implicit-bias / NTK: which directions the dynamics excite, whether a claim is a property of the map or the optimizer, where the honest identifiability boundary is. Lead with the mechanism and the ruler; the reconstructions land better after he trusts we can tell presence from hallucination.

**The one-paragraph pitch.** A released LoRA adapter is a gate-weighted sum of per-image gradient outer products. We built an attack-independent ruler — the whitened end-to-end Jacobian — that reads, off the public model, how many private directions survive fine-tuning and which images occupy them. It separates three regimes: information locally absent or below the noise floor (a scoped statement we can measure), present-but-hard-to-extract (where a better decoder is the task), and a decoder hallucinating a prior (which our controls subtract). The open milestone is robust pixel inversion from the adapter alone; everything else — the rank phase diagram, the activation dissociation, the g₀ predictor — is machinery that says exactly where that milestone is reachable and where it is walled off.

**Anticipated pushback:**

| he'll say | you answer |
|---|---|
| "Detectability isn't reconstruction." | Agreed — the ruler is a local identifiability read, separate from any attack; we quote the q_eff **curve** and full-gradient recoveries alongside the blurry adapter-only ones. |
| "Your q_eff is a count in a basis you chose." | Correct, and it is stated that way: q_eff counts directions of a named normalised local basis. The planned fix is the comparison itself — random pixel vs local-PCA vs generator-latent tangents on the same data, with a radius sweep validating linearity. |
| "Your decoder hallucinates a prior." | World C — we subtract it with the disjoint-adapter control and report the difference. |
| "n=24 killed g₀." | Honest: +0.857 at n=12 is strong; +0.777 at n=24 is indeterminate — but on **precision** (CI half-width 0.19 > the pre-registered 0.15), not on effect size; the CI [0.53,0.91] excludes zero comfortably. The tercile structure survives; a lead, not a law. |
| "Is this the optimizer, not the map?" | The local-vs-global init control and the permutation null separate optimization failure from genuine non-identifiability. |
| "The composition result is just a t-SNE picture." | No longer just a picture: the cross-fitted cluster-robust test confirms **content-level** recovery above the recipe baseline (acc-diff **+0.989**, CI [0.973, 1.005] excludes 0) after fixing a fold bug — the 5 compositions are distinct digit-subsets, so it recovers WHICH DIGITS (content), not the specific instance; graded to ~0 for single-image swaps (arms 0.03–0.07). The gauge confirmation (seed in raw B,A, gone in ΔW) is the second leg. Caveats: G=30 small; CI upper clips >1 (near-ceiling Normal-approx artifact); true instance-level untested. |

**How to state E3 to him** (the version that is hardest to knock down): *"The clean empirical result is that linearization fidelity and leakage dissociate — the ReLU-family are the worst linearizers and the strongest leakers, and smoothness itself is not monotone in leakage. The candidate mechanism is static per-example gate geometry: eff_rank(M) is 6.37 for relu vs 1.73 for softplus and tracks the leakage order, and the softplus-β dial moves both together. The next clean experiment shows the spectrum of the actual mixing matrix G — not smoothness — mediates the ranking."* Then present the ruler as a local Fisher/SNR characterisation of the stochastic data→adapter map, not a count of images a weak attacker recovers. Close on E7: the endpoint attack still does not scale, and the live question is whether the endpoint discarded the directions or our inverse is bad — which the same-cell test can now separate.

**Order:** open with **E3** (mechanism / money figure) → the ruler + **E1** (we measure presence honestly) → **E2** (spectral boundary) → **E4/E5** (who leaks / how much) → close with **E6/E7** (the two frontier channels), each labelled with how far the evidence reaches.

---

## Appendix — symbols & provenance

**Symbols.** ρ = rank(M), the gate-matrix rank (rank only, never a correlation). r_s = Spearman correlation. νᵢ = singular values of J. gᵢ / g₀ = image i's mixing column / its public-model norm. d² = 2·KL = SNR². q_eff = usable-leakage count. P_LoRA(H) = BBᵀH + HAᵀA (a real operator, not a projection).

**Provenance (headline numbers → jobs).**

| claim | value | job |
|---|---|---|
| E2 rank-sweep reversal | 23→13→0 at r=8/16/32 | 581629 |
| E3 kinked vs smooth | ~5× (control-margin 0.47 vs 0.09); Spearman(fs, ctrl-margin) ≈0 (−0.06) | 392821 / 390026 |
| E4 g₀ correlation | +0.857 (n=12) / +0.777 (n=24) / +0.83 full-FT (n=6) | 260171 / 272504 / 695782 |
| E5 valley width | geomean 1.02, median 0.86, narrower 4/6 (n=6) | 695782 |
| E6 ΔW by composition | ARI +1.00; nuisance ≈0; raw (B,A) by seed +0.55; cross-fit +0.989 CI[0.973,1.005] @ G=30 (content / digit-subset) | 838868 (811847 fold-buggy) — archived: `results/atlas/atlas_analyze_838868.txt` |
| E7 ceiling / wall | SSIM ~0.99 / 0.57@N=4 → 0.27@N=10; bridge **0.930 cos converged** (0.951 best-epoch); q_eff 51/117/150/156 at ε=0.3/1/3/10 | multiple (full-grad 956994; direct-inv / bridge / ViT — see STATUS) |

**Review triage (2026-08-30):** an external review of the designed PDF was verified claim-by-claim against the repo and the primary sources; verdicts and the resulting edits are in [`notes/note_v2_review_triage.md`](note_v2_review_triage.md). Two figure defects (E3 panel-A metric mixing, E4's ±0.15 band) and one missing gallery are **Mac-side rebuild items**, specced in `notes/mac_handoff_brief.md`.

**To resolve before circulation:** (1) The **'0.23'** from the earlier note is the **weight-space linearization error** (job 480485, a full-FT T=10 config flagged NTK-violating) — mislabeled there as ‖ΔW‖/‖W₀‖, a different quantity. Don't cite it as a relative-update norm; if a 'not strictly lazy' point is wanted, use the weight-space (0.23) vs function-space (0.0023) linearization-error contrast from that block, with the NTK-flag caveat. (2) Log the max-margin dual **λ = 0.538** used in E4, or drop the g₀-vs-λ comparison. (3) One line on the reconstructed-faces dataset provenance/consent. (4) Re-run the q_eff 59/36 anchor through the bias-corrected estimator before any absolute count is a headline. (5) Redraw E3 panel A from one metric column at one T; replace E4's ±0.15 band with the CI + precision annotation; restore the E7 gallery and split its bar chart by metric.

Key external anchors to wire in (audit pass): Jang et al. 2024 (LoRA NTK — **their own** r(r+1)/2 > KN, with K=1 for binary) and the counterweight *Rethinking the Rank Threshold* (arXiv:2605.03724); MineGrad (Sami–Sen–Güler, AISTATS 2026 — malicious-server LoRA gradient inversion; makes "low rank ≠ privacy" directly, while our passive released-endpoint threat model stays distinct); Putterman/Lim et al. 2025 (Learning on LoRAs — E6); Wang–Lee–Lei AISTATS 2023 (provable gradient identifiability — §1); Hannun–Guo–van der Maaten (Fisher information leakage — §2); Feldman–Zrnic individual privacy accounting (E4); Balunović et al. Bayesian gradient inversion (World C).
