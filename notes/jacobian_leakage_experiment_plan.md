# Jacobian-Spectrum LoRA Leakage — Experiment Plan (v3, PhD-readable)

**Created 2026-08-20; rewritten 2026-08-23 into a document a new student can read cold and build.**
This turns the reframed identifiability note into a falsifiable measurement program on WEXAC. The
central quantity is the **seed-whitened (Fisher) Jacobian** of LoRA fine-tuning; the first experiment
recovers **private coordinates**, not whole images. Code: [`experiments/jacobian_spectrum.py`](../experiments/jacobian_spectrum.py)
(Phase J0 implemented; J1 whitening functions scaffolded). Job script:
[`scripts/run_jacobian_spectrum_wexac.sh`](../scripts/run_jacobian_spectrum_wexac.sh). Companions:
[identifiability_feasibility_revision.tex](identifiability_feasibility_revision.tex), [STATUS.md](../STATUS.md).

> **One-sentence thesis.** Fine-tuning a model on private images `{x_i}` produces a released LoRA
> adapter `Y=(A_T,B_T)`; we measure the Jacobian `J=∂Y/∂a` of the adapter with respect to small,
> realistic image changes `a`, whiten it by ordinary training-seed noise, and show that the number of
> private directions the adapter provably retains (`q_eff`) is exactly what governs whether an attacker
> can reconstruct the data.

---

## Part 0 — Background primer (what each object is, and *why it appears here*)

*A reader who already knows Fisher information and whitening can skip to Part 1.*

**Random variable, Gaussian, white noise.** A random variable is a number that comes out differently
each time you draw it. A *Gaussian* (normal) is the bell-curve distribution; a *vector* Gaussian is a
cloud of points in many dimensions. "White" means the cloud is a perfectly round ball: zero mean, and
every coordinate independent with the *same* variance — no preferred direction. **Why here:** running
LoRA fine-tuning with a different random seed each time (data order, init noise) makes the released
adapter `Y` jitter around its mean. That jitter is our noise. It is *not* white — it is bigger in some
adapter directions than others — which is exactly why the next two ideas are needed.

**Covariance matrix `Σ` and anisotropy.** The covariance `Σ = E[(Y-μ)(Y-μ)ᵀ]` describes the *shape* of
a noise cloud: an ellipsoid, stretched far along high-variance directions and squashed along
low-variance ones (correlations tilt the ellipsoid off the axes). **Why here:** the seed-noise cloud
`Σ_seed = Cov_ξ[vec Y]` is an anisotropic ellipsoid. A change in the private data that only moves `Y`
along a *fat* noise axis is easy to confuse with noise; the same-sized change along a *thin* axis stands
out. So "how visible is a private change" depends on direction, and `Σ_seed` encodes it.

**Whitening (`Σ^{-1/2}`).** Whitening is the linear change of coordinates `z → Σ^{-1/2} z` that turns
*any* Gaussian ellipsoid back into a round unit ball. After whitening, one unit of distance means one
standard deviation of noise **in every direction**, so signal and noise become directly comparable.
**Why here:** `J_SNR = Σ_seed^{-1/2} J` re-expresses "how much a private coordinate moves the adapter"
in units of the noise. A singular value of `J_SNR` equal to 5 means "this private direction moves the
adapter 5 noise-standard-deviations" (recoverable); equal to 0.1 means "buried in noise"
(unrecoverable).

**Jacobian.** For a vector map `a ↦ Y(a)`, the Jacobian `J` is the matrix of all first partial
derivatives: `J[p,q] = ∂Y_p/∂a_q`. It answers "if I nudge input coordinate `q`, how does every output
coordinate move?" — the best linear approximation of the map near a point. **Why here:** our map is
*private data → released adapter*, so `J = ∂vec(A_T,B_T)/∂a` is literally "how the released weights
respond to each private degree of freedom." It is the object the whole program measures.

**Singular values / SVD.** Any matrix `J = UΣVᵀ` acts by rotating, stretching each axis by a factor
`σ_i` (the *singular values*), then rotating again. Large `σ_i` = a direction the map amplifies; `σ_i≈0`
= a direction it flattens to nearly nothing (information in that input direction is nearly lost).
*Effective rank* = "how many directions genuinely survive" (we use the entropy effective rank, a smooth
count that doesn't need an arbitrary threshold). **Why here:** the spectrum of `J` (and of `J_SNR`) is
the fingerprint of *which* private directions the adapter keeps.

**Fisher information and the Cramér–Rao bound (CRLB).** The Fisher information `F = Jᵀ Σ^{-1} J`
measures how much the observation `Y` tells you about the parameters `a` (with observation noise `Σ`).
The Cramér–Rao bound is a theorem: *no* unbiased estimator can recover `a` with covariance smaller than
`F^{-1}`. Note `F = J_SNRᵀ J_SNR`, so `eig_i(F) = σ_i(J_SNR)²`. **Why here:** this promotes "small
singular value" from a heuristic to a *proof* — a private direction with `σ_i(J_SNR)·ε < 1` has a
recovery-error floor larger than the signal itself, so it is provably unrecoverable at perturbation
scale `ε`. This is why the effective recoverable dimension `q_eff`, not raw rank or parameter count, is
*the* privacy quantity.

**Data-processing inequality (DPI).** Post-processing cannot create information: for any function `D`,
`I(a ; D(Y)) ≤ I(a ; Y)`. **Why here:** a "gradient bridge" decoder `D` applied to the adapter cannot
recover private directions that are absent from `Y` itself. If reconstructions show detail that `Y`
provably does not contain, that detail is the decoder *hallucinating from its prior*, not leakage. Used
to make the bridge diagnosis rigorous in J5.

**LoRA and differentiable ("unrolled") training.** LoRA (Low-Rank Adaptation) freezes the base weight
`W₀` and learns a low-rank update `ΔW = (α/r)·B A` with `A∈ℝ^{r×in}`, `B∈ℝ^{out×r}`; only `A,B` are
trained (init `A=0`, so training starts unchanged). "Unrolled" training means we run the `T` SGD steps
with autograd tracking every operation, so the final adapter `(A_T,B_T)` is a *differentiable function
of the training data* — which is what lets us compute `J=∂Y/∂a`. Because each SGD step already contains
a gradient, differentiating the adapter w.r.t. the data is a **second-order** (double-backward)
operation; the activation must be `C²` (we use GELU — `modified_relu` has no double-backward and
silently corrupts `J`).

**Threat model.** The attacker holds the public base weights `θ₀` *and* the released fine-tuned adapter
(or, equivalently, a fine-tuning weight update). This is the honest-but-curious, *ordinary-training*
setting — **not** the malicious-server attacks of PEFTLeak/MineGrad, which poison the model to force
leakage. The question is whether *normal* LoRA fine-tuning leaks the private images above ordinary
training randomness. See [identifiability_feasibility_revision.tex](identifiability_feasibility_revision.tex)
and [minegrad_analysis.md](minegrad_analysis.md).

---

## Part 1 — The question, the object, the one law

**Question.** Does an *ordinarily trained final* LoRA adapter preserve private-image directions well
enough for reconstruction, *above ordinary training randomness*?

**Objects.**
- Private coordinates `a ∈ ℝ^{Nk}`: for each image `i`, an orthonormal set `U_i=[u_{i1},…,u_{ik}]` of
  realistic local tangent directions; the image is `x_i(a_i) = x_i^0 + U_i a_i`. `a=0` is the true
  dataset.
- Forward map `a ↦ Y(a;ξ) = LoRA-train(θ₀, {x_i(a_i)}; seed ξ)` → adapter `(A_T,B_T)`.
- Data Jacobian `J = ∂vec(Y)/∂a |_{a=0}` (size `dim(Y) × Nk`).
- Seed covariance `Σ_seed = Cov_ξ[vec Y | a=0]`, estimated over `S` training seeds (shrinkage-
  regularized — see risks).
- Seed-whitened Jacobian `J_SNR = Σ_seed^{-1/2} J`. `F = Jᵀ Σ_seed^{-1} J = J_SNRᵀ J_SNR`;
  `σ_i(J_SNR)² = eig_i(F)`.
- Effective recoverable dimension at scale `ε`: `q_eff(ε) = #{ i : ε·σ_i(J_SNR) > 1 }`.

**Central law (pre-registered).**
> Reconstruction of the private coordinates degrades as `q_eff(ε)/q` falls below 1; per-coordinate
> recovery error tracks the Cramér–Rao floor `1/(ε·σ_i(J_SNR))`. Spectral collapse of `J_SNR` ⟺
> reconstruction collapse.

**Why a theorem, not a heuristic — and the honesty caveat.** `F` is the Fisher information of `a↦Y`
under noise `Σ_seed`; CRLB gives an error floor `≥ F^{-1}`, so directions with `σ_i(J_SNR)·ε < 1` are
provably below the noise. **But** in the strict local–linear–Gaussian regime the whitened
least-squares estimator *achieves* CRLB by construction, so "spectrum predicts recovery" is
near-tautological *there*. The experiment has teeth exactly where it leaves that regime: nonlinear
training, the real inversion algorithm, larger `ε`, and the step from coefficients to whole images.
**We therefore sweep `ε` out of the linear regime and test whether the Fisher prediction survives.**

**Three worlds** (the spectrum + controls separate these; a rank/capacity number cannot):
- **World A** — `q_eff` and recon collapse together (a genuine identifiability transition).
- **World B** — `q_eff` stays high but the attack fails (information is preserved; the inverter is the
  wall).
- **World C** — `q_eff` collapses yet the decoder still emits plausible images (hallucinating from a
  prior — caught by the disjoint-adapter control, J3).

---

## Part 2 — Experiment J0 (buildable now): deterministic coordinate recovery

**The embarrassingly-controlled test.** *I hid `Nk` continuous private numbers inside realistic image
variations; fine-tune LoRA deterministically; which numbers survive in the adapter, and does the
spectrum of `J` predict which?*

**Goal.** With **deterministic** training (fixed seed, no `Σ_seed` yet) establish that (a) private
coordinates are recoverable at all from the released adapter, and (b) `σ_i(J)` predicts *which*
coordinates recover. This is the gate: if it fails, the whole image-reconstruction theory is likely
wrong — stop and diagnose.

**Theory needed.** Jacobian, SVD/singular values, effective rank, least-squares inversion (Part 0).

**Setup.** `N∈{2,4}` images (balanced binary labels), `k∈{4,8,16}` tangent coordinates per image via
`U_i` (§ build spec), **one** LoRA module (`target_layers=(0,)`), GELU, `T=5` full-batch SGD steps,
`rank=8`, deterministic (fixed `ξ`). Base `θ₀` = the existing MNIST odd/even model; private images from
the MNIST **test** split (disjoint from `θ₀`'s training data).

**Concrete build steps** (all implemented in `jacobian_spectrum.py`):
1. `build_tangents(d,k,N,method)` → per-image tangent bases `U` (§ build spec). Use `method='qr'`
   (clean rank `k`) **and** `method='svd'` (a *known* geometric decay injected into the columns).
2. `_mnist_ctx(...)` assembles the context via `generate_target` (reuse `frozen/b0/B0[0]/ds_mean`;
   **ds_mean frozen at a=0**).
3. `forward_Y(a)` — the differentiable `ℝ^{Nk}→ℝ^{dimY}` map (unrolled single-module LoRA, returns the
   flattened adapter).
4. `exact_jacobian(a0, method='jvp_double')` — `J` via forward-over-reverse double-`autograd.grad`.
5. `spectrum(J)` → `σ_i(J)`, effective rank.
6. `recover_a(J, Y_target, Y0)` — least-squares `â`.
7. `run_j0(...)` sweeps `ε ∈ {1e-3 … 1}`: set `a_true=ε·(unit dir)`, observe `Y`, recover `â`, record
   per-coordinate error and `rel_err(ε)`; save `.pth` + spectrum/recovery figure.

**Tests (Verification section):** finite-difference check on `J` (`<1e-6` on the toy, `<1e-4` on
MNIST); `jvp_double` vs `reverse_loop` agreement (`~1e-10`); deterministic recovery of a known small
`a_true`; and — via `method='svd'` — that `σ_i(J)` tracks the injected column decay (this is what gives
the deterministic "which survive" claim teeth; see the honesty caveat).

**Deliverable.** A validated `J` + its spectrum; the first "which coordinates survive ordinary LoRA
training" result; the `rel_err`-vs-`ε` curve showing where the linear (LSQ) attack starts to break;
`.pth` (`J`, svals, `a_true/a_hat`, per-coord error) + figure per config.

---

## Part 3 — Experiment J1 (buildable now): seed noise, whitening, `q_eff`, CRLB

**Goal.** Add training-seed randomness, measure the noise cloud `Σ_seed`, whiten the Jacobian, and test
the **Cramér–Rao law**: per-coordinate recovery error vs `1/(ε σ_i(J_SNR))`, and overall recovery vs
`q_eff/q`. **Sweep `ε` out of the linear regime** — the interesting signal is where the linear-Fisher
prediction begins to break.

**Theory needed.** Covariance, white noise & whitening, Fisher information, Cramér–Rao bound, and
shrinkage estimation: the sample covariance from `S` seeds has rank `≤ S−1 ≪ dim Y`, so `Σ^{-1/2}` is
ill-posed without regularization (Ledoit–Wolf shrinkage toward isotropic / diagonal loading).

**Concrete build steps** (core functions scaffolded in `jacobian_spectrum.py`):
1. `estimate_sigma_seed(ctx_factory, S, a0)` → the `[S, dimY]` matrix of mean-centered adapter samples
   at `a=0` across `S` training seeds. **Never forms the `dimY×dimY` covariance.**
2. `snr_spectrum(J, centered_samples, shrinkage)` → `σ_i(J_SNR)` via Woodbury: `Σ^{-1}J` needs only the
   `S×S` inverse, and the Fisher `F=Jᵀ Σ^{-1} J` is only `Nk×Nk` (eig it directly).
3. `q_eff(sigma_snr, eps)` → `#{i : ε σ_i > 1}`.
4. CRLB scatter: per-coordinate recovery error (whitened LSQ / real inverter) vs `1/(ε σ_i(J_SNR))`.

**Tests.** Whitening sanity (whitened seed samples ≈ isotropic); `q_eff` reported over a **range** of
the shrinkage regularizer `ρ` and `ε` (small `σ(J_SNR)` are regularizer-sensitive — this must be shown,
not hidden); CRLB-vs-measured per-coordinate scatter.

**Deliverable.** The `q_eff/q`-vs-recovery curve; the CRLB scatter; the first real number for *how many
private directions survive ordinary LoRA training*.

---

## Part 4 — J2–J6: goals + theory written in advance (concrete steps deferred until J0/J1 land)

### J2 — Multi-knob phase diagram `(N, r, L)`
- **Goal.** Show reconstruction boundaries follow the **spectral (`q_eff`) boundary**, not `N`, rank
  `r`, or parameter count independently. Cells chosen so `dρ` (measurement budget) and `q=Nk` (degrees
  of freedom) move in *different* combinations; run **fixed-epochs AND fixed-gradient-steps** (survive
  both ⇒ much stronger).
- **Theory needed.** Confounding and experimental controls; degrees of freedom `q=Nk` vs measurement
  budget `dρ ≈ L·r` (leakage scales with the number of independent adapted modules, not `r` alone — the
  MineGrad lesson); why a joint multi-knob collapse surface is hard to fake.
- *Concrete steps: to be filled in once J0/J1 land.*

### J3 — Disjoint-adapter control (the sharpest World-C test)
- **Goal.** Prove the recovered image comes from *this* adapter, not the prior. True set `X_A→Y_A`; a
  tightly matched, **disjoint** set `X_B→Y_B` (same domain, class distribution, size, hyperparameters,
  generator); plus `Y_null`. Run the same attack (intended to recover `X_A`) from each. Claim survives
  only if `Δ_adapter = Q(X̂(Y_A),X_A) − Q(X̂(Y_B),X_A) > 0` significantly, with `Q∈{LPIPS,CLIP,DINO}`.
- **Theory needed.** Reconstruction vs prior-driven generation; why matched controls isolate
  observation-specific information; the hallucination failure mode (World C).
- *Concrete steps: to be filled in once J0/J1 land.*

### J4 — The whitened `v_min`/`v_max` figure (the thesis picture)
- **Goal.** SVD of **`J_SNR`** (not raw `J`); perturb `a` along `v_min` and `v_max`, render via `U_k`,
  retrain, compare adapter movement to the seed-noise ellipsoid. Captions become privacy statements:
  `v_min` = "two substantially different private datasets, yet the adapter difference is smaller than
  ordinary training variability"; `v_max` = "an equally sized private change produces a highly
  reproducible adapter signature."
- **Theory needed.** Singular vectors as most/least-visible directions; why the comparison must be
  whitened (measured against the noise ellipsoid).
- *Concrete steps: to be filled in once J0/J1 land.*

### J5 — Staged Jacobians + quantitative bridge decomposition
- **Goal.** Along `a → ∇_θL → Y → D(Y) → X̂`, measure whitened tangent survival at each stage
  (`J_G=∂∇_θL/∂a`, `J_Y=∂Y/∂a`, `J_{D∘Y}=∂D(Y)/∂a`). Localize the activation effect (ReLU-family vs
  self-gated: which stage collapses `q_eff`), and **diagnose the gradient bridge via DPI**: if `J_Y`
  misses directions but `D(Y)` shows them → the bridge fills from its prior (hallucination); if all
  survive in `J_Y` → genuine decoding. Either outcome is a headline result.
- **Theory needed.** Composition of maps; information can only decrease (DPI); activation gates `σ'`
  (reuse `gate_matrix_test.py`).
- *Concrete steps: to be filled in once J0/J1 land.*

### J6 — Scale (ViT/SD), realism, and the step to whole images
- **Goal.** `σ(J_SNR)` spectral density at ViT-B/SD scale via **stochastic Lanczos quadrature** (JVP+VJP
  through the unroll; never form `J`). Local-vs-global inversion (invert from `a*+noise` vs random
  init). Then move from coordinate recovery to whole-image reconstruction and check the same `q_eff`
  boundary governs it.
- **Theory needed.** Matrix-free spectral density (Lanczos/Krylov); why `J` is never formed at scale.
- *Concrete steps: to be filled in once J0/J1 land.*

---

## Part 5 — Grounding, metrics, baselines, risks, order

**Grounding.**
- **Belrose et al., "Understanding Gradient Descent through the Training Jacobian"** (arXiv 2412.07003):
  the *init*-Jacobian `∂θ_T/∂θ₀` has a data-dependent, label-independent low-dim spectrum. Our delta:
  the **data-latent** Jacobian `∂θ_T/∂a`, seed-whitened.
- **Fisher/CRLB** makes `q_eff` the principled privacy quantity; **DPI** makes the bridge diagnosis
  rigorous (J5).
- **SLQ / PyHessian / google spectral-density**: `σ(J_SNR)` spectral density at scale, matrix-free.
- **k is the measured effective `rank J_g` at a stated scale, never a nominal latent width** —
  intrinsic-dim estimators disagree wildly; we *construct* `k` from the top-`k` well-conditioned
  directions so `q=Nk` is exact.
- **SimuDy** (ICLR 2025, code `BlueBlood6/SimuDy`): full-FT direct-inversion baseline to adapt.

**Metrics.**
- Spectral: `σ_i(J)`, `σ_i(J_SNR)`, `q_eff(ε)`, condition number, effective rank; `rank_eff(J_g)`.
- Coordinate recovery: per-coordinate error vs CRLB; fraction recovered vs `q_eff/q`.
- Image recon (later): **clip-robust only** — `metrics.compute_ssim_normalized`,
  `retrieval_metric.retrieval_scores`, LPIPS (primary), NCC, CLIP/DINO; always print `clipped_fraction`
  and use `--pixel_box`.
- Control: `Δ_adapter`; null-adapter recon similarity.

**Risks & mitigations.**

| Risk | Mitigation |
|---|---|
| `Σ_seed` from `S` seeds is rank ≤ `S−1` ≪ dim Y → `Σ^{-1/2}` ill-posed | Ledoit–Wolf shrinkage; work in the `S`-sample subspace + a floor; **report `q_eff` over a range of the regularizer and `ε`** |
| Deterministic phase has no `Σ_seed` | J0 uses raw `σ_i(J)`; whitening enters at J1 when seeds are added |
| Linear-regime tautology | sweep `ε` past linear; test on nonlinear training + real inverter + whole images |
| Memory of the third-order unroll graph for large `T`/width | toy-first; `T=5`, one module; sequential JVP loop over one retained graph; functorch+checkpointing fallback |
| `modified_relu` corrupts `J` (no double-backward) | **GELU only**, enforced in code |
| `ds_mean` recomputed as `a` varies would couple the batch | compute once at `a=0`, freeze it inside `forward_Y` |
| Intrinsic-dim ambiguity | construct & measure `rank_eff(J_g)`; never estimate |
| Metric artifact (clipping) | clip-robust metrics + `--pixel_box` |

**Order (fail-fast).** `J0` (recover coordinates, verify `J` numerically) → `J1` (seeds, whiten, CRLB /
`q_eff`, sweep `ε`) → `J2` (`N,r,L` phase diagram) → `J3` (disjoint-adapter control) → `J4` (whitened
figure) → `J5` (staged + bridge) → `J6` (scale + whole images). **J0 and J1 gate everything.**

**Falsifiable predictions (pre-register before J1).**
- **P1.** Per-coordinate recovery error tracks CRLB `1/(ε σ_i(J_SNR))`; recoverable-coordinate count ≈
  `q_eff(ε)`.
- **P2.** `q_eff/q` predicts overall recovery; recon collapses as `q_eff → q` from below.
- **P3.** In the `(N,r,L)` diagram, recon boundaries follow the `q_eff` boundary, not `N`/`r`/param-
  count — and survive both fixed-epoch and fixed-step sweeps.
- **P4.** `Δ_adapter > 0` significantly (observation-specific information beyond the prior).
- **P5.** Self-gated activations collapse `q_eff` at the gradient stage (`J_G`); ReLU-family later.
- **P6 (bridge).** If `J_Y` misses directions but `D(Y)` shows them → prior hallucination (DPI); if all
  survive in `J_Y` → genuine decoding.

Any failure is informative: P2 failure ⇒ World B; P4 failure or P6-hallucination ⇒ World C.

---

## Part 6 — Open hypotheses to test (added 2026-08-24)

Motivation: the current headline result ("on-manifold PCA tangents → collinear `J` → low
recoverability", jobs 988588/989194) may be an artifact of **the basis (PCA)** and of measuring only
the **first-order (linear) map `J`**. These five hypotheses are the concrete escapes; none is a
generative prior — all use information already present. **They move UPSTREAM of J2–J6:** if H1/H2 change
what `q_eff`/`eff_rank` mean, the capacity sweep and SGD-noise phase must be run on the right basis with
the right recovery. Suggested order: **H1 → H2 → (SGD-noise phase) → J2**, with H3/H5 as capacity axes.

- **H1 — Discriminative tangents, not PCA.** PCA-top-k are the directions of maximum *population*
  variance = the common denominator across images; they are collinear **by construction** (all images
  move the same way → parallel adapter responses) and are the *least* private directions. Test
  discriminative bases instead: PCA **tail** (image-specific, higher-freq); PCA of the **private set's
  own mutual differences** (the directions that distinguish the N private images); residual after
  removing the top shared modes; LDA-style between/within. These span a **different image-space
  subspace** than PCA-top-k, so `col(J)` genuinely changes (NOT the invariance no-op of job 993396),
  and they are the genuinely privacy-relevant object. Hypothesis: collinearity drops and recoverability
  rises. **Drop-in `build_tangents` mode; cheapest test most likely to overturn the current
  conclusion.**
- **H2 — Nonlinear recovery beyond first order.** `q_eff`, `eff_rank`, and the invariance argument are
  all *first-order* (properties of the linear `J`). The true map `a↦Y` is nonlinear and can be
  injective where `J` is rank-deficient (e.g. `Y ~ (a·v)²` recovers `|a·v|` though `Jv=0`). Test the
  real `direct_inversion` nonlinear optimizer against a collinear (pca-top) config across a range of ε;
  check whether it recovers past the linear `q_eff` ceiling (especially at larger ε, out of the linear
  regime — the plan's honesty caveat already anticipated this).
- **H3 — Multi-point measurement fusion.** Different work points (anchors α, B0 draws, widths) give
  `J`'s with *different* null spaces. **Fuse** them — stack `J(α₀), J(α₁), …` as complementary
  measurements — to raise the effective rank, since their blind spots don't coincide. (Currently
  anchors are used only as robustness checks, not combined.) The honest analog of the full-gradient
  papers' extra measurements.
- **H4 — Conditioning-aware sequential peeling.** Instead of one linear solve over all `Nk`
  coordinates, recover the *well-conditioned* subspace of `col(J)`, subtract its contribution from `Y`,
  and repeat on the residual. Extracts the information actually present without a fancier inverter and
  without a prior — the honest middle ground.
- **H5 — Network width as a capacity axis.** Wider hidden layers → more neurons → higher-rank
  data→adapter map → less binding collinearity. Sweep width, distinct from LoRA rank `r` and #modules
  `L`.

**Scope of the invariance result (job 993396):** "recombining coordinates can't beat `col(J)`" is a
**linear, fixed-subspace, fixed-map** statement. H1 (different subspace), H2 (nonlinear), H3/H5
(different/enlarged map) all lie OUTSIDE its scope — they are the legitimate ways to beat the apparent
collinearity. Also note the reframing H1 brings: discriminative directions ARE "what distinguishes one
private image from another" — the genuine privacy threat — so H1 upgrades *what we measure*, not just
*how well*.

**How the prior papers sidestep collinearity (context for H1–H5):** more independent measurements
(Haim: full weights; DAGER/NTK: rank ≥ N; MineGrad: `L·r` disjoint modules, via malice); statistical
priors (Cocktail-Party ICA: source independence + non-Gaussianity; SPEAR: ReLU sparsity); or generative
priors (diffusion/SDS — the only true null-space fill, at the cost of hallucination = World C). Our
single-rank-8-module setup is deliberately the most information-starved case; H3/H5 are the honest
"more measurements" analog, H1/H2 use information we already discard.

---

## J0/J1 build spec (implemented in `experiments/jacobian_spectrum.py`)

Mirrors `direct_inversion.py` conventions (`sys.path` insert, `torch.set_default_dtype(float64)`, GELU
only, `--device cuda`). **Reuse, don't reimplement.**

**Functions.**
- `build_tangents(d,k,N,seed,method='qr',decay=0.5) -> (U [N,d,k], col_scales [N,k])` — `qr`:
  orthonormal random tangents (clean `k`); `svd`: orthonormal columns scaled by a geometric decay
  (a *known* rank deficiency, to confirm the spectrum measures it).
- `make_images(x0_centered, U, a) -> x` — `x_i = x0_i + U_i a_i` (differentiable, float64).
- `unrolled_lora_AB(frozen,b0,B0,x,y,lr,T,scaling,act,target_layers=(0,)) -> (A,B)` — local variant of
  `direct_inversion.unrolled_finetune_lora` that **returns A,B** and trains only `target_layers`. Uses
  `_partial_lora_forward` and a local `_a_shape` (NOT `direct_inversion.A_rank_shape`, which hardcodes
  the MNIST dims and would break the toy net). Does **not** edit `direct_inversion.py`.
- `build_ab_index(frozen,B0,target_layers)` / `flatten_AB(A,B,index) -> vecY` — fixed A-then-B,
  ascending-layer layout.
- `forward_Y(a_flat, ctx) -> vecY` — the `ℝ^{Nk}→ℝ^{dimY}` closure (`Ctx` freezes ds_mean at a=0).
- `exact_jacobian(a0, ctx, method='jvp_double') -> J [dimY,Nk]` — **forward-over-reverse JVP via double
  `autograd.grad`** (pure autograd, composes with the create_graph unroll; `Nk` backward passes over
  one retained graph). `method='reverse_loop'` cross-checks on the toy only.
- `spectrum(J)` — reuses `gate_matrix_test.effective_rank`; returns svals, eff_rank.
- `recover_a(J,Y_target,Y0,metric_isqrt=None)` — (whitened) least squares via `torch.linalg.lstsq`.
- **J1:** `estimate_sigma_seed`, `snr_spectrum` (Woodbury; Fisher only `Nk×Nk`), `q_eff` — all new.

**Inputs via existing infra:** `generate_target(direct_inversion.py)` → `(θ_T_all, frozen, b0, B0,
ds_mean)` (its all-layer `θ_T` is **not** the single-module target; we use `frozen/b0/B0[0]/ds_mean`
and define `Y0:=forward_Y(0)`); `get_finetuning_data` (TEST split, disjoint from `θ₀` train);
`make_activation('gelu')`. Later image-recon variant: `run_ntk_extraction(ntk_extraction.py, lora_B0=,
free_coefficients=, pixel_box=)`.

**Gotchas (baked in).** float64 throughout; **GELU only**; **ds_mean frozen at a=0**; one LoRA module
first; `scaling=1` (`alpha=rank`); `retain_graph=True` across the `Nk` JVP loop. Also: with `A₀=0`, `B`
is stationary at step 1 (`∂loss/∂B ∝ A = 0`), so `B`-rows of `J` carry signal only for `T≥2` — `T=5` is
safe.

**WEXAC:** `scripts/run_jacobian_spectrum_wexac.sh` (`short-gpu`, `mem=16384`, `gpu num=1`, conda `rec`,
`python -u`, Stage-0 AD gate that aborts on fail, logs to `scripts/wexac_logs/`). We run directly on the
WEXAC filesystem, so no rsync is needed; submit with `bsub < scripts/run_jacobian_spectrum_wexac.sh`.

---

## Verification (how the experiments self-test — this runs as Stage 0 of the job)

Two-tier smoke, **inside the bsub job (never run locally)**, float64:
1. **Toy-AD unit test** (`_toy_ctx`): synthetic net `d_in=6, d_h=5, d_out=1`, `B0` rank 2, `N=2, k=4,
   T=5`, GELU. The gate tests only **AD correctness** (rank deficiency is a finding, not a failure):
   **central finite-difference check** on 3–4 coords (`ε=1e-5`, rel err `< 1e-6`); `jvp_double` vs
   `reverse_loop` agreement (`< 1e-8`); and a rank-independent **linearization residual**
   `‖(Y_t−Y0) − J·a_true‖/‖Y_t−Y0‖ < 1e-3` at a tiny `a_true`. Coordinate recovery and `eff_rank(J)`
   are **reported diagnostics, not gated** — recovery quality is conditioning-dependent science
   (`run_j0`), and a rank-deficient `J` is exactly the identifiability signal we want.
   **The gate `sys.exit(1)`s on failure and aborts the job.** (Job 970028 already showed the AD is
   exact: FD `5.9e-10`, jvp-vs-reverse `3.5e-18`, with the toy `J` genuinely rank-deficient
   `eff_rank≈5.9 < Nk=8` — which is why recovery was moved out of the gate to a diagnostic.)
2. **Real single-module smoke:** MNIST via `generate_target`, `N=2 (n_per_class=1)`, `rank=2`,
   `target_layers=(0,)`, `T=5`, GELU → `dimY=r·(784+1000)=3568`, `Nk=8`, `J` is `[3568,8]`. FD on 3
   coords (`< 1e-4`); print spectrum + eff_rank.

J1 checks: whitened seed samples ≈ isotropic; `q_eff` reported over a **range** of shrinkage `ρ` and
`ε`; CRLB-vs-measured scatter.

---

## Critical files to reuse (paths)
- `experiments/jacobian_spectrum.py` — **this program** (J0 implemented, J1 scaffolded).
- `experiments/direct_inversion.py` — `unrolled_finetune_lora`, `_lora_forward`, `generate_target`,
  save conventions; GELU/float64 policy.
- `experiments/lora_wrapper.py` — `LoRALinear` A/B shapes, `apply_lora`, `scaling`.
- `experiments/ntk_steps.py` — `compute_multi_step_update_lora` (target/B0/ds_mean source).
- `experiments/gate_matrix_test.py` — `effective_rank`, per-sample `Φ=∇_θf` (template for `J_G` in J5),
  CSV pattern.
- `experiments/ntk_extraction.py` — `run_ntk_extraction` (`lora_B0`, `free_coefficients`, `pixel_box`)
  for the image-recon variant.
- `experiments/metrics.py`, `experiments/retrieval_metric.py` — clip-robust metrics.
- `experiments/data_utils.py` — `get_finetuning_data`, `get_control_images_*`.
- `experiments/plotting.py` — grids, twin-axis curves (`q_eff`-vs-recon).
- `experiments/configs.py` — float64/device policy, `TRAIN_LR`, `RESULTS_DIR`/`FIGURES_DIR`.
