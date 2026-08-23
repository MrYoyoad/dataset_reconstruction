# Jacobian-Spectrum LoRA Leakage — Experiment Plan (v2)

**Created 2026-08-20, revised same day with supervisor refinements.** Turns the reframed
identifiability note into a falsifiable program on WEXAC. The central quantity is now the
**seed-whitened (Fisher) Jacobian**, and the first experiment recovers **private coordinates**, not
whole images. Companions: [identifiability_feasibility_revision.tex](identifiability_feasibility_revision.tex),
[STATUS.md](../STATUS.md).

---

## 0. Question, object, and the central law

**Question.** Does an *ordinarily trained final* LoRA adapter preserve private-image directions well
enough for reconstruction — *above ordinary training randomness*? (Not the malicious-server case.)

**Objects.**
- Private coordinates `a ∈ ℝ^{Nk}`: for each image `i`, an orthonormal set `U_i=[u_{i1},…,u_{ik}]`
  of realistic local tangent directions; the image is `x_i(a_i) = x_i^0 + U_i a_i` (or on-manifold
  via a generator). `a=0` is the true dataset.
- Forward map `a ↦ Y(a;ξ) = LoRA-train(θ₀, {x_i(a_i)}; seed ξ)` → adapter `(A_T,B_T)`.
- Data Jacobian `J = ∂vec(Y)/∂a |_{a=0}` (size `dim(Y) × Nk`).
- **Seed covariance** `Σ_seed = Cov_ξ[vec Y | a=0]`, estimated over `S` training seeds (shrinkage-
  regularized — see §7).
- **Seed-whitened Jacobian** `J_SNR = Σ_seed^{-1/2} J`. This is the square root of the **Fisher
  information** `F = Jᵀ Σ_seed^{-1} J = J_SNRᵀ J_SNR`; `σ_i(J_SNR)² = eig_i(F)`.
- **Effective recoverable dimension** at perturbation scale `ε`:
  `q_eff(ε) = #{ i : ε·σ_i(J_SNR) > 1 }`.

**Central law (pre-registered).**
> Reconstruction of the private coordinates degrades as `q_eff(ε)/q` falls below 1; per-coordinate
> recovery error tracks the Cramér–Rao floor `1/(ε·σ_i(J_SNR))`. Spectral collapse of `J_SNR` ⟺
> reconstruction collapse.

**Why a theorem, not a heuristic.** `F` is the Fisher information of `a↦Y` under noise `Σ_seed`; the
Cramér–Rao bound gives an error floor `≥ F^{-1}` for any unbiased estimator, so directions with
`σ_i(J_SNR)·ε < 1` are provably below the noise and unrecoverable. **Honesty caveat:** in the strict
local–linear–Gaussian regime the whitened least-squares attack *achieves* CRLB by construction, so
"spectrum predicts recovery" is near-tautological there. The experiment has teeth exactly where it
leaves that regime — nonlinear training, the real inversion algorithm, larger `ε`, and the step from
coefficients to whole images. Sweep `ε` out of the linear regime and test whether the Fisher
prediction survives.

**Three worlds** (the spectrum + controls separate these; a rank/capacity number cannot):
World A `q_eff` and recon collapse together (identifiability transition); World B `q_eff` stays high
but the attack fails (info preserved, inverter is the wall); World C `q_eff` collapses yet the
decoder still emits plausible images (hallucinating from prior — caught by the disjoint-adapter
control, §J3).

---

## 1. Grounding

- **Belrose et al., "Understanding Gradient Descent through the Training Jacobian"** (arXiv
  2412.07003): the *init*-Jacobian `∂θ_T/∂θ₀` has a data-dependent, label-independent low-dim
  spectrum. Our delta: the **data-latent** Jacobian `∂θ_T/∂a`, seed-whitened. Decomposition
  `J = (∂θ_T/∂x)|_adapter · U`.
- **Fisher/CRLB** framing (above) makes `q_eff` the principled privacy quantity; the
  **data-processing inequality** makes the bridge diagnosis rigorous (§J5).
- **SLQ / PyHessian / google spectral-density**: `σ(J_SNR)` spectral density at scale, matrix-free.
- **k = measured effective `rank J_g` at a stated scale, never nominal latent width** — estimators
  of intrinsic dim disagree wildly (CIFAR-10: 11–96), and even a 512-latent generator can have local
  `rank J_g(z) ≪ 512`. We *construct* `k` from the top-`k` well-conditioned singular directions of
  `J_g(z*)` (§2), so `q = Nk` on the x-axis is exact, not smeared.
- **SimuDy** (ICLR 2025, code `BlueBlood6/SimuDy`): full-FT direct-inversion baseline to adapt.

---

## 2. Constructing controlled `k` (do this before any sweep)

1. Fix a batch `{x_i^0}` and (optionally) a frozen generator `g`.
2. At each `z_i*` (or `x_i^0`), compute the generator Jacobian `J_g(z_i*)` and its SVD
   `J_g = U Σ Vᵀ`. **Report the spectrum**, not a nominal number.
3. Take `U_i` = the top-`k` **well-conditioned** left-singular directions (image-space, orthonormal),
   at a stated conditioning floor. Private manifold tangent for image `i` is `span(U_i)`; coordinates
   `a_i ∈ ℝ^k` with `δx_i = U_i a_i`.
4. This removes generator-coordinate scaling and makes `q = Σ_i rank_eff(J_g(z_i*)) = Nk` exact and
   measured. Two generator families: **Gen-L** (linear `x=μ+Uz`, no prior → clean A/B) and **Gen-G**
   (frozen VAE/StyleGAN → realistic prior, needed for World-C in §J3).

---

## 3. Phases (fail-fast; coordinate recovery FIRST)

### Phase J0 — Tiny deterministic coordinate-recovery testbed (the new first experiment)
The embarrassingly-controlled test: *I hid `Nk` continuous private numbers inside realistic image
variations; which survive LoRA training?*
- **Setup:** `N∈{2,4}` images, `k∈{4,8,16}` via `U_k` (§2), **one** LoRA module, GELU,
  **deterministic** training (fixed `ξ`).
- **Compute** `J = ∂Y/∂a` exactly (`torch.func.jacrev` over the unrolled train map; fallback
  `autograd.grad(create_graph=True)` — `direct_inversion.py` already unrolls). Finite-difference
  verify.
- **Recover** `â` from `Y` (whitened LSQ in the linear regime; the real inverter as `ε` grows).
- **Predict vs measure:** deterministic ⇒ no `Σ_seed` yet, so the predictor is `σ_i(J)` / effective
  rank; test that recoverable coordinates ⟺ large `σ_i(J)`.
- **Gate:** coordinate recovery works at all, and the spectrum predicts *which* coordinates. If not,
  the whole image-reconstruction theory is likely wrong — stop and diagnose.

### Phase J1 — Seeds + whitening → `q_eff` and the Fisher/CRLB law
- Train `S∈{16,32,64}` seeds at `a=0`; estimate `Σ_seed` (Ledoit–Wolf shrinkage / top-noise-subspace
  + floor, §7); form `J_SNR = Σ_seed^{-1/2} J`; compute `σ_i(J_SNR)`, `q_eff(ε)`.
- **Central tests:** (i) per-coordinate recovery error vs CRLB `1/(ε σ_i(J_SNR))`; (ii) overall
  recovery vs `q_eff/q`. **Sweep `ε`** across the linear→nonlinear boundary — the interesting signal
  is where the linear-Fisher prediction starts to break.
- **Deliverable:** the `q_eff/q` vs recovery curve; the CRLB-vs-measured per-coordinate scatter.

### Phase J2 — Multi-knob phase diagram (defeats the "large N just trains differently" objection)
- Cells over `(N, r, L)` — dataset size × LoRA rank × adapted-module set — chosen so `dρ` and
  `q=Nk` move in *different* combinations.
- Per cell: `q`, `σ(J_full)`, `q_eff`, recovery/recon quality.
- **Two sweeps: fixed epochs AND fixed gradient steps.** Survives both ⇒ much stronger.
- **Test:** do reconstruction boundaries follow the **spectral (`q_eff`) boundary**, not `N`, `r`, or
  parameter count independently? A joint multi-knob collapse surface is hard to fake.

### Phase J3 — Disjoint-adapter control (the sharpest World-C test)
- True private set `X_A → Y_A`; a **tightly matched, disjoint** set `X_B → Y_B` (same domain, class
  distribution, size, hyperparameters, generator `g`); plus `Y_null` (shuffle / matched-norm noise /
  module-ablated).
- Run the *same* attack intended to recover `X_A`, from each of `Y_A`, `Y_B`, `Y_null`.
- **Quantify:** `Δ_adapter = Q(X̂(Y_A), X_A) − Q(X̂(Y_B), X_A)` with `Q ∈ {LPIPS, CLIP, DINO}`
  correspondence. Claim survives only if `Δ_adapter > 0` significantly:
  *the correct adapter provides information about its own images above what the same prior yields from
  an unrelated adapter* — the reconstruction-vs-prior-generation distinction the thesis needs.

### Phase J4 — The whitened `v_min`/`v_max` figure (the thesis picture)
- SVD of **`J_SNR`** (not raw `J`); take `v_min`, `v_max`. Perturb `a` along each, render via `U_k`,
  retrain, compare adapter movement to the **seed-noise ellipsoid**.
- **Captions become privacy statements:** `v_min` = "these two private datasets differ substantially,
  yet the difference they induce in the released adapter is smaller than ordinary training
  variability"; `v_max` = "this equally sized private change produces a highly reproducible adapter
  signature."

### Phase J5 — Staged Jacobians + quantitative bridge decomposition
- Along `a → ∇_θL → Y → D(Y) → X̂`, measure whitened tangent survival at each stage:
  `J_G=∂∇_θL/∂a` (reuse `gate_matrix_test.py`), `J_Y=∂Y/∂a`, `J_{D∘Y}=∂D(Y)/∂a`.
- **Activation localization:** for ReLU-family vs self-gated (GELU/SiLU/Mish), find the stage where
  `q_eff` collapses — gradient stage (`J_G`) vs LoRA-observation stage (`J_Y`). Gives the measured
  activation ranking a mechanism.
- **Bridge diagnosis (rigorous via data-processing inequality):** `D(Y)` cannot raise `I(a;·)` above
  `I(a;Y)`. So if `J_Y` has private directions below the noise floor but `D(Y)` "looks like a full
  gradient" in those directions, the bridge is **filling from its learned prior** (gradient-level
  hallucination). If all tangent directions already survive in `J_Y`, the bridge is decoding genuine
  LoRA information. Either outcome is a headline result.

### Phase J6 — Scale, realism, and the step to whole images
- `σ(J_SNR)` spectral density at ViT-B/SD scale via **SLQ** (JVP+VJP through the unroll; never form
  `J`).
- **Local vs global:** invert from `a*+noise` (tracks `J_SNR`) vs random init (global).
- **From coordinates to images:** once coordinate recovery + the law hold, move outward to whole-image
  reconstruction and check the same `q_eff` boundary governs it.

---

## 4. Suggested order (the revised "next move")

**Do NOT start with the giant `N×r×L` sweep.** First build the tiny system where the core law can be
tested beyond ambiguity:
`J0` (2–4 images, 4–16 tangent coords, 1 module, deterministic — verify `J` numerically, recover
coordinates) → `J1` (add seeds, whiten, test CRLB / `q_eff`, sweep `ε`) → `J2` (`N,r,L` phase
diagram, fixed-epochs + fixed-steps) → `J3` (disjoint-adapter control) → `J4` (whitened figure) →
`J5` (staged + bridge) → `J6` (scale + whole images). If `J0/J1` light up, we have a *measurable law
of when fine-tuning remembers private data*, not just a story around the attack.

---

## 5. Metrics
- **Spectral:** `σ_i(J)`, `σ_i(J_SNR)`, `q_eff(ε)`, condition, effective rank; `rank_eff(J_g)`.
- **Coordinate recovery:** per-coordinate error vs CRLB; fraction recovered vs `q_eff/q`.
- **Image recon (later):** LPIPS (primary), `ssim_norm`, retrieval/margin, NCC, CLIP/DINO
  correspondence. **Clip-robust only** (sibling finding): print `clipped_fraction`, use `--pixel_box`.
- **Control:** `Δ_adapter`; null-adapter recon similarity.

---

## 6. Tooling & existing infra

| Need | Approach | Infra |
|---|---|---|
| Differentiable LoRA train | unrolled SGD, `create_graph`, GELU (not `modified_relu`) | `direct_inversion.py`, `lora_wrapper.py` |
| `J = ∂Y/∂a` | `torch.func.jacrev`/`jacfwd`; fallback `autograd.grad` | new `experiments/jacobian_spectrum.py` |
| `Σ_seed`, whitening | S-seed sample cov + Ledoit–Wolf shrinkage; `Σ^{-1/2}J` | new |
| `σ(J_SNR)` at scale | SLQ / Lanczos on `J_SNRᵀJ_SNR` via JVP+VJP | PyHessian / google spectral-density |
| Coordinate/image inversion | whitened LSQ (local) → model-based extractor | `ntk_extraction.py`, `direct_inversion.py` |
| `J_G` / gate rank | staged Jacobian | `gate_matrix_test.py` |
| Metrics | LPIPS, ssim_norm, retrieval, NCC | `metrics.py`, `retrieval_metric.py` |

**Memory:** toy-first; checkpointing; IFT at convergence; linearized/NTK `J` as a cheap cross-check.
Save `.pth` (spectra, `J` or its SVD, `a_true`/`a_hat`, `x_*`, adapters) + `.csv` per config; best &
worst; ground-truth + control in every grid. Script `scripts/run_jacobian_spectrum_wexac.sh` (CUDA,
GELU, `python -u`). rsync before submit.

---

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| `Σ_seed` from `S` seeds is rank ≤ `S−1` ≪ dim Y → `Σ^{-1/2}` ill-posed | Ledoit–Wolf shrinkage; restrict to top seed-noise subspace + a floor; **report `q_eff` over a range of the regularizer and `ε`** — small `σ(J_SNR)` are regularizer-sensitive |
| Deterministic phase has no `Σ_seed` | J0 uses raw `σ_i(J)`; whitening enters at J1 when seeds added |
| Linear-regime tautology | sweep `ε` past linear; test on nonlinear training + real inverter + whole images |
| `S` too small to estimate `Σ_seed` vs signal | choose `S` so noise subspace is well-estimated before whitening; report `S` |
| Unroll memory (SimuDy: 22GB/15h) | toy-first; checkpointing; IFT; linearized `J` |
| Intrinsic-dim ambiguity | construct & measure `rank_eff(J_g)`; never estimate |
| Metric artifact (clipping) | clip-robust metrics + `--pixel_box` |

---

## 8. Falsifiable predictions (pre-register before J1)
- **P1.** Per-coordinate recovery error tracks CRLB `1/(ε σ_i(J_SNR))`; recoverable-coordinate count
  ≈ `q_eff(ε)`.
- **P2.** `q_eff/q` predicts overall recovery; recon collapses as `q_eff → q` from below.
- **P3.** In the `(N,r,L)` diagram, recon boundaries follow the `q_eff` boundary, not `N`/`r`/param-
  count — and survive both fixed-epoch and fixed-step sweeps.
- **P4.** `Δ_adapter > 0` significantly (observation-specific information beyond the prior).
- **P5.** Self-gated activations collapse `q_eff` at the gradient stage (`J_G`); ReLU-family later.
- **P6 (bridge).** If `J_Y` misses directions but `D(Y)` shows them → prior hallucination (DPI);
  if all survive in `J_Y` → genuine decoding.

Any failure is informative: P2 failure ⇒ World B; P4 failure or P6-hallucination ⇒ World C — both
real results.
