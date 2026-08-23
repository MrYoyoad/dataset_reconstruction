# Jacobian-Spectrum LoRA Leakage — Experiment Plan

**Created 2026-08-20.** Turns the reframed identifiability note
([identifiability_feasibility_revision.tex](identifiability_feasibility_revision.tex)) into a
concrete, falsifiable experimental program on WEXAC. Companion:
[STATUS.md](../STATUS.md) "Feasibility reframed" entry.

---

## 0. The question and the one falsifiable prediction

**Question.** Does an *ordinarily trained final* LoRA adapter preserve the private-image
directions well enough for actual reconstruction? (Not: can a *malicious* server encode them —
that's MineGrad/PEFTLeak, already answered yes.)

**Central object.** The end-to-end data→adapter Jacobian
`J_full = ∂ vec(A_T, B_T) / ∂ z`, where `z ∈ ℝ^{Nk}` are the latents of `N` images each with `k`
controlled degrees of freedom.

**Pre-registered prediction (P1).** `σ_min(J_full)` (the weakest-preserved data direction)
collapses in the *same* regimes where reconstruction quality collapses. Spectral collapse ⟺
reconstruction collapse.

**Three worlds the experiment can land in** (a capacity/rank number cannot distinguish these; the
spectrum + controls can):
- **World A (dream):** `σ_min` and reconstruction collapse together → a measured identifiability
  transition, theory explains attack.
- **World B:** `J_full` stays well-conditioned but the attack fails → the adapter *did* preserve
  the info; the inverter/decoder is the bottleneck.
- **World C:** `J_full` collapses yet the decoder still emits plausible images → the decoder is
  **hallucinating from its prior**, not recovering forced information. The critical privacy
  distinction.

---

## 1. Why this is the right object (grounding)

- **Belrose et al., "Understanding Gradient Descent through the Training Jacobian"** (arXiv
  2412.07003, Dec 2024) already study the spectrum of the *training* Jacobian `∂θ_T/∂θ₀` and find
  a data-dependent, label-independent low-dimensional structure (a "bulk" of σ≈1 that carries
  perturbations through unchanged, and a collapsing "stable" region). **Our delta: the
  data-latent Jacobian `∂θ_T/∂z` instead of the init-Jacobian `∂θ_T/∂θ₀`** — same machinery,
  privacy-relevant variable. Note the clean decomposition
  `J_full = (∂θ_T/∂x)|_adapter · J_g`, i.e. (training map's data-sensitivity) ∘ (prior tangent).
- **σ_min(J_full) is exactly local invertibility.** The inversion optimizes `z` to match `Y`;
  `J_full = ∂Y/∂z` at the true `z`, so `σ_min` small ⟺ the inverse problem is locally
  ill-conditioned ⟺ `z` is not locally recoverable from `Y`. The prediction P1 is the natural
  local-identifiability statement, made measurable.
- **k = rank(J_g), controlled — never a literature intrinsic-dim estimate.** Estimators disagree
  wildly (CIFAR-10 reported 11–96 depending on MLE/TwoNN/GeoMLE/diffusion-NB; MNIST ~152 by
  diffusion vs ~13 by MLE). So we *set* `k` by construction and *measure* `rank(J_g)` directly,
  sidestepping the whole estimation problem.

---

## 2. Design principles

1. **Control `k` by construction; measure it.** Two generators:
   - **Gen-L (linear):** `x = μ + U z`, `U ∈ ℝ^{d×k}` orthonormal ⇒ `rank(J_g)=k` exactly, and
     **no image prior**, so World C (hallucination) is impossible → clean A-vs-B identifiability
     transition.
   - **Gen-G (generative):** a frozen pretrained decoder/VAE/StyleGAN with the latent restricted to
     a `k`-dim subspace ⇒ a *realistic* prior → this is where World C can appear and is tested.
   - Always report the *measured* `rank(J_g)` (SVD of the generator Jacobian), not the nominal `k`.
2. **Pair the spectrum with the actual attack at every config** — never report one without the
   other; that pairing is the whole point.
3. **Invert in latent space** (`min_z ‖Y − F(θ₀, g(z))‖²`) so the `k`-dim prior is baked in exactly
   and the reconstruction is `g(z*)`.
4. **GELU is required** for the unrolled differentiable training (double-backward);
   `modified_relu` breaks `create_graph` (CLAUDE.md). Use GELU/softplus/silu, not `modified_relu`.
5. **Clip-robust metrics only** (sibling-session finding, commits 298c805/3bdbfb0): raw SSIM on a
   clipped reconstruction is an artifact. Print `clipped_fraction`; use `ssim_norm`, NCC,
   retrieval/margin, LPIPS; or the `--pixel_box` path.

---

## 3. Phases

### Phase J0 — Differentiable pipeline + exact `J_full` (toy; sanity)
- **Build** `experiments/jacobian_spectrum.py`: `g → images → LoRA fine-tune (unrolled T steps,
  GELU) → Y=(A_T,B_T)`. Reuse `direct_inversion.py` (already unrolls SGD with `create_graph`) +
  `lora_wrapper.py`.
- **Compute `J_full` explicitly** at toy scale (small MLP, small `d`, `N∈{1,2}`, `k∈{4,8}`) via
  `torch.func.jacrev`/`jacfwd` over the unrolled map (verify torch≥2.0 on the `rec` env; fallback:
  row-by-row `torch.autograd.grad(create_graph=True)`).
- **Verify:** finite-difference check on a few `J_full` entries; `rank(J_g)=k`; `J_full` shape
  `[Σ_layers r(d_in+d_out)] × [Nk]`.
- **Deliverable:** validated `J_full` + its SVD at toy scale. Save `.pth` (J_full, spectrum) + `.csv`.
- **Gate:** finite-diff agreement < 1e-3 relative; only then proceed.

### Phase J1 — Core sweep: `σ_min(J_full)` vs reconstruction (THE test)
- **Grid (Gen-L first):** `k∈{4,8,16,32}` × `N∈{1,2,4,8,16,32,64}` × LoRA `r∈{1,4,8,16}` ×
  activation `{gelu, softplus}` × 3 seeds. `T` fixed (calibrate so `weight_change` is comparable
  across activations — reuse the LR-calibration lesson).
- **Per config:** SVD(`J_full`) → `σ_min` over the `Nk` directions, condition `κ=σ_max/σ_min`,
  effective rank; **and** run inversion → LPIPS (primary), `ssim_norm`, retrieval/margin, NCC.
- **Plot:** `N` vs `σ_min(J_full)` vs LPIPS, one panel per `(k,r)`. **Test P1** (co-collapse) and
  **P2** (does `N*` — the collapse point — scale with `r` up, `k` down?).
- **Deliverable:** the `N`-vs-`σ_min`-vs-recon figure; a per-regime World-A/B label.
- **Fail-fast value:** if P1 *fails* (well-conditioned `J_full` but failing attack everywhere),
  that is itself a headline (World B — LoRA preserves the info; the inverter is the wall).

### Phase J2 — Three-worlds disambiguation (controls; run on Gen-G)
- **World C (hallucination) detection — null-adapter controls:** replace `Y` with (i) shuffled
  adapters, (ii) matched-norm Gaussian noise, (iii) module-ablated adapter, (iv) progressively
  smaller pieces of the adapter. If inversion *still* returns the right image → the prior is
  hallucinating, not the adapter leaking.
- **World B (decoder bottleneck) detection:** in regimes with large `σ_min` but failing recon, swap
  in a stronger inverter (multi-init; the model-based `run_ntk_extraction` from `ntk_extraction.py`
  instead of naive SVD; more optimization).
- **Deliverable:** an A/B/C label per regime with the control evidence table.

### Phase J3 — The "remembers vs forgets" figure (`v_min`/`v_max`)
- SVD `J_full = UΣVᵀ`; take `v_min` (weakest right-singular direction) and `v_max`. Perturb
  `z' = z ± ε v`, render via `g`, retrain both datasets, measure adapter movement `‖ΔY‖` vs image
  movement `‖Δx‖`.
- **Expected:** `v_min` = large image change, near-zero adapter change (LoRA *forgets*); `v_max` =
  comparable image change, huge adapter change (LoRA *remembers*).
- **Deliverable:** the thesis figure — "a private-data change LoRA remembers" beside "one it forgets".

### Phase J4 — Staged Jacobians: localize the activation effect
- Compute `J_grad = ∂∇_θL/∂z` (reuse/extend `gate_matrix_test.py`), `J_LoRA-step` (one-step
  adapter), and `J_full`, for the ReLU-family `{relu, leaky, softplus, selu}` vs the self-gated
  family `{gelu, silu, mish}`.
- **Question:** where does the spectrum collapse per activation — gradient stage, LoRA-observation
  stage, or full? Connect to the measured Addition-2 activation ranking.
- **Deliverable:** an activation × stage spectral table; localizes "GELU fails" to a stage.

### Phase J5 — Scale & realism (ViT-B / SD LoRA)
- **Spectrum at scale via Stochastic Lanczos Quadrature** (Ghorbani ICML'19; PyHessian; google
  `spectral-density`): `σ_min`/spectral density of `J_fullᵀ J_full` matrix-free, using JVP/VJP
  (forward-over-reverse) through the unroll — never forming `J_full`.
- **Confounds:**
  - *Local vs global:* invert from `z*+small noise` (should track `J_full`) vs random init (global,
    harder). Local-success/global-failure ⇒ theory intact, optimization is the wall.
  - *Seed vs signal:* compare data signal `‖E[Y|X+δX] − E[Y|X]‖` against training noise
    `√E‖Y − E[Y|X]‖²` across seeds. If minibatch-order variance dominates one-image-change variance,
    deterministic `J_full` overstates usable leakage.
- **Bridge as restricted inverse:** measure `rank(∂Y/∂z)` on the gradient manifold vs the ambient
  full-gradient null space; `rank ≈ Nk` despite large ambient null space = direct evidence.
- **Deliverable:** does the toy story transfer to real adapters?

---

## 4. Compute, tooling, existing infra

| Need | Approach | Existing infra |
|---|---|---|
| Differentiable LoRA training | unrolled SGD, `create_graph=True`, GELU | `direct_inversion.py`, `lora_wrapper.py` |
| `J_full` (toy) | `torch.func.jacrev`/`jacfwd`; fallback `autograd.grad` | new `jacobian_spectrum.py` |
| `J_full` spectrum (scale) | SLQ / Lanczos on `JᵀJ` via JVP+VJP | PyHessian / google `spectral-density` |
| Inversion attack | `min_z ‖Y−F(θ₀,g(z))‖²`; model-based extractor | `ntk_extraction.py`, `direct_inversion.py` |
| `J_grad` / gate rank | staged Jacobian | `gate_matrix_test.py` |
| Clip-robust metrics | LPIPS, ssim_norm, retrieval, NCC | `metrics.py`, `retrieval_metric.py` |
| Linearized `J` (cheap approx) | NTK/anchor linearization to validate against | `ntk_verification.py`, anchor machinery |

**Memory** (SimuDy needs 22GB/15h for 120 tiny imgs on full unroll): start toy; use gradient
checkpointing; consider implicit differentiation (IFT) at convergence to avoid storing the unroll;
use the linearized/NTK `J` as a cheap approximation to cross-check the exact one.

**Always save** `.pth` (spectrum, `x_recon`, `x_true`, `x_ctrl`, adapters) + `.csv` per config; best
AND worst reconstructions per config; ground-truth + control in every grid. New WEXAC script
`scripts/run_jacobian_spectrum_wexac.sh` (CUDA, GELU, `python -u`). rsync before every submit.

---

## 5. Baselines & related work

- **SimuDy** (Tian et al., ICLR 2025; code `BlueBlood6/SimuDy`) — full-FT direct-inversion baseline;
  adapt to LoRA. "Most training samples reconstructed from a trained ResNet."
- **Belrose training-Jacobian** (2412.07003) — methodology anchor; cite as the init-Jacobian
  analog; our contribution is the data-latent Jacobian + the leakage prediction.
- **R2F gradient bridge** — the decoder that may lift the observed rank (test whether it extracts or
  hallucinates, Phase J2/J5).
- **MineGrad / PEFTLeak (CVPR'25 / AISTATS'26)** — *malicious-server* attacks; contrast, not
  baseline — they don't answer the honest-checkpoint question.
- **DSiRe / weight-space identity** — final-LoRA encodes dataset size / identity attributes
  (properties, not exact recovery). *[citations need bib keys + verification — the reconstruction
  claims in casual web sources are unreliable; use the peer-reviewed ones only.]*

---

## 6. Falsifiable predictions (pre-register before running J1)

- **P1.** `σ_min(J_full)` drops sharply as `N` crosses `N*(k,r)`; reconstruction LPIPS worsens at
  the *same* `N*`.
- **P2.** `N*(k,r)` increases with `r`, decreases with `k` (measure the law; don't assume `d/k`).
- **P3.** Self-gated activations' spectrum collapses at an *earlier stage* (`J_grad`) than
  ReLU-family (localizes the measured GELU-vs-ReLU gap).
- **P4.** `v_min` perturbations are near-invisible to the adapter; `v_max` hyper-visible.
- **P5.** In `σ_min`-collapsed (World-C-candidate) regimes, null-adapter controls *still* "recover"
  → flags hallucination.

Any of these failing is informative, not a loss — P1 failure = World B (a real result); P5 firing =
World C (a real privacy caveat).

---

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Unroll memory blows up | toy-first; checkpointing; IFT; linearized `J` cross-check |
| `torch.func` unavailable on `rec` env | fallback to `autograd.grad(create_graph=True)` (already used) |
| Intrinsic-dim estimation ambiguity | control `k`, measure `rank(J_g)` — never estimate |
| Metric artifact (clipping) | clip-robust metrics + `--pixel_box`; print `clipped_fraction` |
| Non-transversality / bad conditioning | that IS the measurement — `σ_min` reports it honestly |
| `J_full` too big for exact SVD | SLQ spectral density (Phase J5) |

---

## 8. Suggested order & sizing

1. **J0** (differentiable pipeline + validated `J_full`) — days; the gate for everything.
2. **J1** (core `σ_min`-vs-recon sweep, Gen-L) — the main result; toy sweeps ~1–2 weeks.
3. **J2 + J3** in parallel (controls on Gen-G; the figure) — once J1 shows a transition.
4. **J4** (staged Jacobians) — reuses J1 machinery; ties in the activation track.
5. **J5** (scale + realism) — later, only after the toy story holds.

**Decision gate after J1:** World A → push to scale (J5) and write the leakage paper. World B →
pivot to "the info is there; the inverter is the open problem" (decoder/bridge focus). World C
prominent → the honest-privacy contribution is "LoRA leakage claims must control for prior
hallucination," itself publishable.
