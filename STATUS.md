# Project Status

Last updated: **2026-04-28** (post D2 sweep)

---

## Recovery (2026-03-19)

### What Was Lost
The WEXAC home directory lost its connection to the GitHub repo. Conversation history from Jan–Mar was lost. Claude Code custom skills were deleted.

**Still missing:**
- 3 custom Claude Code skills still need recreation: `/write`, `/lesson`, `/project-manager`
- `multi_seed_analysis.png` was regenerated from a 15-seed run (original was 200-seed; would need re-running the 200-seed sweep to reproduce)

**Recovered (2026-03-24):**
- All 5 missing figures regenerated from saved .pth tensors: `experiment_b_grid_r32.png`, `rank_sweep_sprint1.png`, `sprint1_summary.png`, `multi_seed_analysis.png`, `t_sweep_examples.pdf`
- 8 Claude Code commands recreated: `/review`, `/supervisor`, `/experiment`, `/debug`, `/figure`, `/paper`, `/research`, `/status`

**Recovered from GitHub (`myfork/main`):**
- All 18 papers in `papers/`
- All 7 notes files (GRADIENT_BRIDGE_PLAN.md, R2F_Guide, Inversion_Feasibility, Thesis_Direction)
- 4 figures (parameter_as_function_of_epoch variants)
- Full git history (15 commits)

**What was fixed (2026-03-19):**
- Rebased WEXAC state onto `myfork/main` history — all files restored
- Added Sprint 2 results (87 CSVs), WEXAC scripts, new experiment code
- Moved WEXAC job scripts to `scripts/`, logs to `scripts/wexac_logs/`
- Recreated `/research` and `/project-manager` Claude Code commands
- Updated CLAUDE.md, STATUS.md, LESSONS_LEARNED.md
- Pushed to `myfork` via SSH

---

## What's In Progress

### Sprint 2c: KKT & NTK Reconstruction Ablations — COMPLETE

Comprehensive ablation study across two tracks. 148+ configs completed.

**Track A: Experiment A (KKT) — CLOSED (negative result confirmed)**
- 15/48 configs completed before 48h timeout (job 583398). KKT loss stuck at 330-350 for ALL N values tested.
- Confirms Sprint 1 structural analysis: composed model W=W₀+BA satisfies KKT over all ~502 samples. No amount of N tuning overcomes the pre-training residual.
- **This definitively closes the KKT approach for composed models.**

**Track B: Experiment B (NTK) — Ablations**
- B1: Phase 3+4 (LR scheduling + warm-start) — **DONE** (results in sprint2b_phase3/4 CSVs)
- B2: Loss ratio ablation (verify_weight) — **DONE** (16 configs, results/sprint2c_track_b2_*.csv)
- B3a: Optimizer × activation for LoRA — **DONE** (results/sprint2c_track_b3a_*.csv). Winner: **SGD + LeakyReLU** (SSIM 0.830 for both r=8 and r=32)
- B3b: Scale best combo across T — **DONE** (SGD+LeakyReLU matches L-BFGS for T≤20, NaN at T=100)
- B4: N sweep (NTK) — **DONE** (results/sprint2c_track_b4_*.csv)
- B5-B8: Additional ablations — **DONE** (results in sprint2c_track_b5/b6/b7/b8 CSVs)

### Phase 0: ViT Gradient Inversion Gate — D2 COMPLETE, GATE CROSSED

Critical gate experiment: can gradient inversion reconstruct images from exact ViT-B/16 gradients?

**Status: GATE CROSSED.** D2 sweep (2026-04-28) found tv=1e-1 + lr=0.05 + 30K iters achieves **SSIM=0.548, PSNR=15.11, cos_sim=0.955** on Flowers102 — a 3.8× improvement over D1's best (0.144) and well past the 0.3 SSIM gate. 7/29 configs cleared the gate, all at tv=1e-1. The thesis can proceed to LoRA-only inversion, multi-seed validation, and the Gradient Bridge decoder.

**Run history:**
1. **2026-03-27**: SSIM=0.015. Failed due to 4 bugs. All fixed.
2. **2026-04-07**: Bug fixes applied. Vague structure visible. Non-standard SSIM metric (don't cite).
3. **2026-04-09**: Code audit fixed 6 issues (SSIM metric, backward, signAdam, TV norm, clamping, LoRA dims). Switched to Flowers102.
4. **2026-04-10**: signAdam bug found (was SignSGD). cos_sim=0.97 but SSIM=0.008 (noise). After fix: SSIM=0.022 full / 0.009 LoRA.
5. **2026-04-14**: D1 controlled comparison — 4 configs, same image/gradient.

#### D1: Controlled Optimizer × TV Comparison — COMPLETE (2026-04-14)

4 configs on the SAME Flowers102 image (seed=42), full-model gradient (86M params), 8 restarts × 10K iters each:

| Config | Optimizer | TV weight | SSIM | PSNR | cos_sim | Time |
|--------|-----------|-----------|------|------|---------|------|
| A | Adam | 1e-4 | 0.030 | 8.7 | 0.920 | 3.0h |
| B | signAdam | 1e-4 | 0.020 | 8.2 | 0.934 | 1.8h |
| C | Adam | 1e-2 | 0.090 | 9.8 | 0.887 | 1.7h |
| **D** | **signAdam** | **1e-2** | **0.144** | **10.9** | **0.933** | **3.0h** |

**Key findings:**
1. **Strong TV (1e-2) is essential.** Both strong-TV configs (C, D) beat both weak-TV configs (A, B) in SSIM. tv_weight=1e-4 is 100× too weak at 224×224.
2. **signAdam beats Adam at every TV level.** D > C by 60% (0.144 vs 0.090), B ≈ A at weak TV. signAdam maintains high cos_sim (0.93) even with strong TV drag.
3. **cos_sim alone is misleading.** Config B has highest cos_sim (0.934) but worst SSIM (0.020). Config D has similar cos_sim (0.933) but 7× better SSIM. The TV prior makes the difference.
4. **signAdam convergence is faster and tighter.** Cos_sim overlay shows signAdam restarts cluster tightly (0.920-0.934) while Adam restarts spread widely (0.465-0.920).

**Go/no-go outcome:** Config D SSIM=0.144 is just below the 0.15 gate threshold. **Proceeded to D2** (see below) — D2 crossed the gate decisively at SSIM=0.548.

**Instrumentation added (2026-04-14):**
- `best_cos_sim` + per-restart `loss_history` saved in .pth files
- Loss curve plots (3-panel: cos_sim/TV/total vs iteration, all restarts)
- D1 comparison figure + cos_sim overlay

- Code: `experiments/phase0_vit_inversion.py` + `experiments/phase0_d1_compare.py`
- WEXAC scripts: `scripts/run_phase0_d1_{A,B,C,D}.sh`
- Results: `results/phase0_full_r8_n1_s42_20260414_*.pth`, `results/phase0_d1_comparison_*.csv`
- Figures: `figures/phase0/phase0_d1_comparison.png`, `figures/phase0/phase0_d1_cossim_overlay.png`

#### D2: Targeted Sweep Around Winning Config — COMPLETE (2026-04-28)

40-config sweep (signAdam, full gradient, Flowers102, seed=42), 29 configs analyzed:

**Top 7 configs (all cleared 0.3 SSIM gate):**

| Rank | TV weight | LR    | Iters | SSIM   | PSNR  | cos_sim |
|------|-----------|-------|-------|--------|-------|---------|
| 1    | **1e-1**  | 0.05  | 30000 | **0.548** | **15.11** | **0.955** |
| 2    | 1e-1      | 0.10  | 10000 | 0.496  | 12.93 | 0.955   |
| 3    | 1e-1      | 0.01  | 30000 | 0.469  | 12.44 | 0.941   |
| 4    | 1e-1      | 0.10  | 30000 | 0.469  | 12.81 | 0.955   |
| 5    | 1e-1      | 0.50  | 10000 | 0.466  | 12.20 | 0.946   |
| 6    | 1e-1      | 0.50  | 30000 | 0.464  | 12.27 | 0.959   |
| 7    | 1e-1      | 0.05  | 10000 | 0.385  | 12.23 | 0.930   |

**Key findings:**
1. **tv_weight=1e-1 is the dominant winning factor.** All 7 gate-passing configs use tv=1e-1. The next TV level (2e-2) tops out at SSIM=0.27. The 10× jump from D1's tv=1e-2 (SSIM=0.144) to tv=1e-1 produced a 3.8× SSIM improvement.
2. **LR is secondary.** Across lr ∈ {0.01, 0.05, 0.1, 0.5} at tv=1e-1, SSIM stays in [0.46, 0.55]. lr=0.05 is best but the spread is small.
3. **30K iters helps but not dramatically.** lr=0.05 jumps from 0.385 (10K) to 0.548 (30K). lr=0.1 marginal: 0.496 → 0.469. Diminishing returns past 10K for most configs.
4. **High cos_sim (0.94-0.96) at all 7 winners**, while D1's noisy configs had similar cos_sim with garbage SSIM. Strong TV is what converts gradient match into visible structure.

**Go/no-go outcome:** **GATE CROSSED.** SSIM=0.548 is well past the 0.3 threshold. The flower's pink color, petal arrangement, and leaf structure are all clearly visible. **Proceed to multi-seed validation, LoRA-only inversion, and face-photo extension (face1/2/3 sweep already submitted).**

- Code: `experiments/phase0_d2_compare.py`, `phase0_vit_inversion.py` `--d2 --config_index N` mode
- WEXAC scripts: `scripts/run_phase0_d2_wexac.sh` (40 jobs), face sweep: `scripts/run_phase0_face_sweep.sh`
- Results: `results/phase0_d2_*.pth` (29 configs), figures: `figures/phase0/d2_sweep/`
- Custom image support: `--image_path` flag in `phase0_vit_inversion.py`, tests in `experiments/tests/test_phase0_custom_image.py`
- Repo reorg: figures grouped under `figures/{phase0,sprint1,training_dynamics,free_c_all_seeds}/`. Per-iter snapshot dirs (~800 MB) excluded via `.gitignore`.

#### Post-D2: Next Steps

- **Face sweep (in flight)**: 4-config grid on face1.jpg with tv ∈ {1e-2, 5e-2}, iters ∈ {10K, 30K} — uses custom-image loader
- **LoRA-only at the D2 winner**: rerun signAdam + tv=1e-1 + lr=0.05 + 30K with `--mode lora --rank 8` (and 16, 32, 64)
- **Multi-seed canonical numbers**: 5-10 seeds at the winning config to replace seed=42 anecdote with mean±std
- **D3 (if priors needed)**: frequency-domain prior, LPIPS, latent-space (S3.5)

---

### Sprint 3: Scaling Beyond MNIST — IN PROGRESS (via D1→D2→D3)

**Goal**: Establish gradient-based reconstruction on realistic (non-MNIST) data. Sprint 2 proved the NTK attack works on MNIST MLPs (SSIM=0.997 full, 0.557 LoRA free-c). Sprint 3 bridges to ViT-scale.

**Updated strategy after D1 results:** D1 showed the bottleneck is regularization, not architecture. signAdam + TV=1e-2 reached SSIM=0.144 on ViT-B/16 with no architectural changes. The path forward is hyperparameter tuning + stronger priors on the same ViT, not retreating to simpler architectures.

- ~~**S3.1**: Fix Phase 0 hyperparameters~~ → **Absorbed into D1 (done) + D2 (next)**
- **S3.2**: Shrink reconstruction space (optimize in 32×32 or frequency domain, upsample to 224×224) — still relevant if D2 plateaus
- ~~**S3.3**: Shrink architecture~~ — **Deprioritized** (ViT works, no need to retreat)
- **S3.4**: Differentiable unrolling (bypass NTK approximation) — future direction
- **S3.5**: Add stronger image priors (LPIPS, frequency, SDS) — **maps to D3**

Each sub-sprint has a clear **go/no-go gate** so we don't waste time on dead ends.

#### S3.1: Phase 0 Hyperparameter Sweep (1-2 days)

**Hypothesis**: Phase 0's poor SSIM is primarily due to untuned hyperparameters, not a fundamental ViT limitation. The current config (lr=0.1, tv_weight=1e-4, Adam, 10K iters, 8 restarts) was never swept.

**Design**:
- Independent variable: lr × tv_weight × n_iters × optimizer
- Grid:
  - `lr`: [0.01, 0.05, 0.1, 0.5, 1.0]
  - `tv_weight`: [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
  - `n_iters`: [10000, 30000]
  - `optimizer`: [Adam, SGD+momentum, L-BFGS]
- Fixed: rank=8, n_images=1, seed=42, mode=full, n_restarts=4 (reduced for speed)
- Total: 5 × 6 × 2 × 3 = 180 configs (but use early stopping + parallel restarts)
- **Practical reduction**: run a 2-stage sweep — first stage: coarse lr × tv_weight (30 configs, 4 restarts, 10K iters). Second stage: top-5 configs with 30K iters, 8 restarts.
- Metrics: SSIM, PSNR, final cosine similarity, wall time
- Output: `results/sprint3_s1_phase0_sweep_*.csv`, best config `.pth`

**Go/no-go gate**: If best SSIM > 0.3 (full model), proceed to S3.5 (priors). If SSIM < 0.15 for all configs, the bottleneck is architectural → proceed to S3.2/S3.3.

**Code**: Extend `phase0_vit_inversion.py` with `--sweep` mode.
**Script**: `scripts/run_sprint3_s1_phase0_sweep.sh`

#### S3.2: Low-Dimensional Reconstruction Space (1-2 days)

**Note**: Phase 0 already uses CIFAR-10 images resized to 224×224 (see `get_sample_images()` in `phase0_vit_inversion.py`). The data isn't the issue — the 224×224 *search space* is.

**Hypothesis**: Optimizing x_recon in 224×224 pixel space (150K dims) is wasteful when the source image is 32×32 CIFAR-10 (3K dims of actual information). Reconstructing in a lower-dimensional space and upsampling should dramatically improve convergence.

**Design**:
- **Variant A — 32×32 reconstruction**: Optimize x_recon in 32×32 space, bilinear-upsample to 224×224 before feeding to ViT. Reduces search space 49×.
- **Variant B — Fourier-truncated reconstruction**: Parameterize x_recon in frequency domain, zero out high-frequency components (>32×32 bandwidth). Smoother optimization landscape.
- **Variant C — Patch-aware reconstruction**: ViT-B/16 uses 14×14 = 196 patches of 16×16 pixels. Reconstruct per-patch means + low-rank structure (196 × ~50 dims = 9.8K parameters).
- Compare all variants against baseline (full 224×224 pixel space)
- Fixed: rank=8, seed=42, mode={full, lora}, best hyperparams from S3.1
- Seeds: 5 seeds for quick validation, 20 seeds if promising

**Go/no-go gate**: Any variant SSIM > 0.3 → search space was the bottleneck, proceed to S3.5 (priors). All variants SSIM < 0.15 → ViT gradient signal itself is too weak → proceed to S3.3.

**Code**: Extend `phase0_vit_inversion.py` with `--recon_space {pixel, lowres, fourier, patch}` flag.
**Script**: `scripts/run_sprint3_s2_lowdim.sh`

#### S3.3: Simpler Architectures on CIFAR-10 (2-4 days)

**Hypothesis**: ViT's 86M parameters + attention double-backward make gradient inversion intrinsically harder than CNNs/ResNets. Testing simpler architectures isolates whether the bottleneck is the model or the data.

**S3.3a: Small CNN on CIFAR-10** (1-2 days)
- Architecture: Conv(3→32, 3×3) → ReLU → MaxPool → Conv(32→64, 3×3) → ReLU → MaxPool → FC(64×8×8→128) → FC(128→1)
- ~200K parameters (comparable to LoRA param count)
- Train from scratch on CIFAR-10 binary (vehicles vs animals, matching Haim et al.)
- Fine-tune on 1-2 held-out images, T=1 SGD step
- Run NTK reconstruction (Experiment B style) from ΔW
- Run gradient inversion (Phase 0 style) from exact gradient
- Compare both methods on same model

**S3.3b: ResNet-18 on CIFAR-10** (1-2 days)
- Load pretrained ResNet-18 from torchvision (11M params)
- Apply LoRA (r=8, 16) to conv layers via peft
- Fine-tune on 1-2 held-out CIFAR-10 images
- Run gradient inversion from exact gradient
- Compare to ViT results — ResNet's skip connections stabilize gradient flow

**S3.3c: DeiT-Tiny on CIFAR-10** (1 day, parallel with S3.3b)
- Load DeiT-Tiny from timm (5.7M params, same ViT architecture but 15× smaller than ViT-B)
- Apply LoRA (r=8)
- Fine-tune + invert
- Tests whether ViT architecture itself is the issue, or just its scale

**Go/no-go gate**:
- CNN SSIM > 0.4 + ViT SSIM < 0.2 → architecture is the bottleneck; thesis focuses on CNN/ResNet LoRA reconstruction
- Both SSIM > 0.3 → method works across architectures; proceed to scale up
- Both SSIM < 0.15 → gradient inversion on color images is fundamentally harder; consider differentiable unrolling (S3.4)

**Code**: `experiments/phase0_cnn_cifar.py`, `experiments/phase0_resnet_cifar.py`, `experiments/phase0_deit_cifar.py`
**Script**: `scripts/run_sprint3_s3_arch_comparison.sh`

#### S3.4: Differentiable Unrolling — Approach G (3-5 days)

**Hypothesis**: The NTK approximation (ΔW ≈ -η Σ cᵢ ∇f(θ₀; xᵢ)) is a first-order linearization that breaks at T>1. Gradient inversion (Phase 0) requires brittle `create_graph=True` double-backward through attention. **Differentiable unrolling** avoids both problems: simulate the actual fine-tuning steps differentiably and match the resulting weights to the observed weights.

**Method**:
```
# Outer optimization over x_recon
for outer_iter in range(N_outer):
    # Inner loop: simulate T fine-tuning steps
    θ = θ_base.clone()
    for t in range(T):
        loss = L(θ; x_recon)
        grads = autograd.grad(loss, θ, create_graph=True)
        θ = θ - η * grads  # differentiable SGD step

    # Outer loss: match simulated weights to observed weights
    outer_loss = ||θ - θ_observed||²
    outer_loss.backward()  # backprop through all T inner steps
    optimizer_x.step()
```

**Design**:
- Phase 1: Validate on MNIST MLP (should reproduce Experiment B at T=1: SSIM≈0.997)
- Phase 2: Test T=1,2,5,10,20 on MNIST MLP, compare to NTK results
- Phase 3: Apply to CNN/ResNet on CIFAR-10 (if S3.3 identifies a working architecture)
- Phase 4: Apply to ViT on CIFAR-10 (if memory permits — T steps of ViT forward/backward)
- Memory management: gradient checkpointing for T>10

**Key advantages over NTK**:
- Exact for any T (no linearization error)
- No coefficient estimation needed (no cᵢ unknowns)
- Reduces to Experiment B at T=1 (validation check)
- Works with any architecture (no NTK assumptions)

**Key risks**:
- Memory: O(T) computation graphs (mitigated by gradient checkpointing)
- Must know exact lr and T (can sweep if unknown — realistic attacker may not know these)
- Non-convex outer optimization — needs restarts

**Go/no-go gate**: T=1 on MNIST reproduces SSIM>0.99 → validates implementation. T=10 beats NTK SSIM → publish as improvement. T=1 fails → implementation bug, debug before proceeding.

**Code**: `experiments/differentiable_unrolling.py`
**Script**: `scripts/run_sprint3_s4_unrolling.sh`

#### S3.5: Stronger Image Priors (2-4 days, after S3.1/S3.2 gate)

**Hypothesis**: Even with correct gradient signal, 224×224 RGB reconstruction requires priors beyond TV to converge. Natural images occupy a tiny manifold in pixel space.

**Prior hierarchy** (implement in order of complexity):
1. **Frequency-domain prior** (0.5 day): Penalize high-frequency Fourier components. Natural images have most energy in low frequencies. `freq_loss = ||FFT(x)[high_freq]||²`. Nearly free to implement.
2. **LPIPS perceptual loss** (1 day): Use frozen DINO or ResNet50 features. `lpips_loss = ||F(x_recon) - F(x_recon_smoothed)||²` (self-regularization) or compare to a "natural image" centroid. Requires `lpips` package.
3. **Batch normalization statistics prior** (0.5 day): If model has BN layers, match running mean/var of reconstruction to the BN statistics. Free signal from the model itself. (Only applicable to ResNet/CNN, not ViT.)
4. **Score Distillation Sampling (SDS)** (2-3 days): Frozen diffusion model (Stable Diffusion) guides reconstruction toward natural images. `sds_loss = E_t,ε[w(t)(ε_θ(x_t, t) - ε) ∂x_t/∂x]`. Most powerful but most complex. Requires diffusion model on same GPU.
5. **Latent-space reconstruction** (1-2 days): Instead of optimizing in pixel space, optimize in the latent space of a frozen VAE (from Stable Diffusion). Decode to pixel space for the gradient matching loss. Reduces search space from 150K to ~4K dims.

**Design**: Add priors as composable loss terms in `invert_gradient()`. Each prior has a weight hyperparameter. Sweep prior weights on the best Phase 0 config.

**Code**: `experiments/image_priors.py` (prior loss functions), integrated into `phase0_vit_inversion.py` via `--prior` flag.
**Script**: `scripts/run_sprint3_s5_priors.sh`

#### Sprint 3 Summary Table

| Sub-sprint | What | Architecture | Data | Time | Depends on |
|------------|------|-------------|------|------|------------|
| **S3.1** | Phase 0 hyperparam sweep | ViT-B/16 | CIFAR-10 224×224 | 1-2d | — |
| **S3.2** | Low-dim recon space | ViT-B/16 | CIFAR-10 (32→224) | 1-2d | — |
| **S3.3a** | CNN baseline | Conv2+FC | CIFAR-10 32×32 | 1-2d | — |
| **S3.3b** | ResNet-18 + LoRA | ResNet-18 | CIFAR-10 32×32 | 1-2d | — |
| **S3.3c** | Small ViT | DeiT-Tiny | CIFAR-10 32×32 | 1d | — |
| **S3.4** | Diff. unrolling | MNIST MLP first | MNIST → CIFAR | 3-5d | — |
| **S3.5** | Image priors | Best from above | Same | 2-4d | S3.1 or S3.2 |

**Parallelism**: S3.1, S3.2, S3.3a-c, S3.4 Phase 1 can all run independently. S3.5 depends on having a working baseline to improve.

**Priority ordering** (Gradient Bridge > NTK > Priors):
1. **Week 1, batch 1** (parallel, cheapest first):
   - S3.1: Phase 0 hyperparam sweep — quick diagnostic, clarifies if tuning alone helps (1 WEXAC job)
   - S3.4 Phase 1: Unrolling on MNIST — validates the approach on known-working data (local or 1 WEXAC job)
2. **Week 1, batch 2** (parallel, informed by batch 1 results):
   - S3.2: Low-dim reconstruction space — if S3.1 shows ViT signal exists, this amplifies it
   - S3.3a: CNN on CIFAR-10 — architecture isolation test, independent of ViT results
3. **Week 2** (depends on Week 1 gates):
   - S3.4 Phase 2: Unrolling T-sweep on MNIST — if Phase 1 validates
   - S3.3b or S3.3c: whichever architecture shows most promise
4. **Week 2-3**: S3.5 priors on best-performing config from above
5. **Week 3+**: S3.4 Phase 3-4 (unrolling on CIFAR-10 architecture)

**Critical path**: S3.4 (unrolling) feeds directly into the Gradient Bridge pipeline (Tier 1 of thesis roadmap). If unrolling works on MNIST and scales to CIFAR-10, it becomes the inversion engine for the full attack. S3.1-S3.3 are supporting experiments that de-risk the architecture choice.

### Sprint 2 Multi-Seed Validation — COMPLETE (2026-03-27)

50-seed free-c vs oracle comparison and 30-seed LeakyReLU validation completed overnight.

**Key findings:**
- **Seed=42 was an outlier**: SSIM=0.830 vs 50-seed mean=0.558±0.034. Report 50-seed stats as canonical.
- **Free-c beats oracle**: Mean SSIM 0.557 (free-c) vs 0.408 (oracle) across 50 seeds. Free-c wins 46/50. The consistency penalty provides implicit regularization that prevents sign-flip local minima.
- **LeakyReLU validated**: 30 seeds × {T=1, T=10} × {r=8, r=32}. Mean SSIM 0.558 (T=1), 0.572 (T=10). Control: 0.394-0.426.
- **r=16/32 improved**: SGD+LeakyReLU gives r=16 SSIM 0.624 (was 0.422), r=32 SSIM 0.680 (was 0.415).

### Sprint 2 Track 2: LoRA Free-Coefficient Extraction — COMPLETE

**Final results (best config per rank — SGD+LeakyReLU fixes r=16/32):**

| Rank | Oracle SSIM | Free-c SSIM | Coeff Error | Gap | Method |
|------|-------------|-------------|-------------|-----|--------|
| 4    | 0.615       | 0.509       | 0.192       | 0.11 | ReLU+L-BFGS |
| 8    | 0.692       | 0.617       | 0.177       | 0.08 | ReLU+L-BFGS |
| 16   | 0.769       | **0.624**   | —           | 0.15 | **SGD+LeakyReLU** (was 0.422) |
| 32   | 0.697       | **0.680**   | —           | 0.02 | **SGD+LeakyReLU** (was 0.415) |
| 64   | 0.714       | 0.635       | 0.019       | 0.08 | ReLU+L-BFGS |

### Sprint 2b: Multi-Step NTK Sweep — COMPLETE

Phases 0-2 completed (WEXAC jobs 669864, 674627). Phases 3-4 completed as Sprint 2c Track B1.

**Phase 0 (activation ablation):** LeakyReLU is dramatically more stable than ReLU at high T. Full model SSIM stays ~0.77-0.80 through T=100 vs ReLU collapsing/NaN'ing at T>=50.

**Phase 1 (SGD + free-c baseline, ReLU):** LoRA results terrible with ReLU — NaN everywhere at T>=50. Confirms activation choice is critical.

**Phase 2 (random restarts, LeakyReLU):** LoRA r=8 and r=32 nearly match full model (gap only 0.01-0.03) through T=100. Random restarts show low variance (~0.01).

**Phase 3 (LR scaling)** and **Phase 4 (warm-start):** Completed as Sprint 2c Track B1.

### Few-Shot Threat Model Analysis — Documented

The few-shot threat model is documented in CLAUDE.md (information density argument): when LoRA fine-tuning uses very few samples, the number of adapter parameters far exceeds the number of unknowns, making the system highly overdetermined and reconstruction theoretically feasible.

---

## What's Been Done

### Sprint 2a: Free-Coefficient Extraction — COMPLETE (2026-02-22)

Fixed the oracle-coefficient cheating. Implementation in `ntk_extraction.py` and `run_experiment_b.py`.

**Full model free-c result:** SGD + consistency α=1 achieves SSIM=0.997 (matches oracle). The attack works without cheating on full model.

### Sprint 2 Track 2: LoRA Activation + Optimizer Ablation — COMPLETE (2026-02-23)

**Activation ablation** (WEXAC job 669885): Swept alpha ∈ {10, 50, 150, 10000} × {L-BFGS, SGD} for LoRA r=8.
- alpha=10000 (≈ReLU) + L-BFGS = SSIM **0.744** (best)
- alpha=150 (ModifiedRelu default) + L-BFGS = 0.183 (terrible)
- ModifiedRelu actively harms LoRA extraction

**Separate optimizer for c** (2026-02-23): Decoupled L-BFGS for x from SGD/Adam for c. Added `coeff_optimizer_type` parameter with 'sgd' and 'adam' options.

**LoRA rank sweep with free-c** (WEXAC jobs 674631, 681126): r=4/8/16/32/64 with ReLU + L-BFGS + separate SGD for c. See results table in "In Progress" section above.

### Sprint 1: LoRA Reconstruction on MNIST FCN — COMPLETE (2026-02-22)

**Goal**: Produce preliminary results showing LoRA-trained weights leak training data.

**Experiment A — Convergence + Compose → KKT Reconstruct (Pre-Trained Init): FAILED**
- LoRA (r=8) reached loss=7.22e-8 after 1M epochs; full FT reached 1.29e-7. Neither hit the 1e-40 threshold.
- Loss decays as ~1/t from pre-trained init. Reaching 1e-40 would take ~10^39 epochs.
- KKT extraction NaN'd at epoch 7-8 (KKT loss started at ~460, should be ~0 for converged models).
- **Root cause (structural, not just convergence)**: The composed model W = W₀ + BA satisfies KKT with respect to all 502 samples the model was effectively trained on (500 pre-training + 2 fine-tuning). The extraction assumes only 2 samples, so the KKT loss of ~460 is essentially ||W₀||² — the unexplained pre-training residual from ~100-250 original support vectors. Even with perfect convergence, 2 images cannot explain weights that encode 502 images. (The 2 fine-tuning samples ARE support vectors for the N=2 case — the issue is the other 500 baked into W₀.)
- **This negative result motivates the Gradient Bridge**: compose-and-reconstruct fundamentally cannot separate fine-tuning signal from pre-training weights in the KKT framework.

**Experiment B — 1-Step NTK Reconstruction from Pre-Trained Weights: SUCCESS**

**IMPORTANT: All Sprint 1 Experiment B results use oracle coefficients** — cᵢ = (σ(f(θ₀; xᵢ)) - yᵢ)/N computed from the true private data x. In a real attack, the adversary doesn't have x and can't compute cᵢ. These results are an **upper bound** on attack quality. The next step is implementing free-coefficient extraction (see Sprint 2 plan below).

| Variant | SSIM | DSSIM | Notes |
|---------|------|-------|-------|
| Full model (T=1) | **0.9999** | 5.2e-5 | Near-perfect reconstruction (oracle c) |
| LoRA rank=8 | **0.797** | 0.102 | Recognizable digits, blurry (oracle c) |
| LoRA rank=16 | **0.802** | 0.099 | Slight improvement (oracle c) |
| LoRA rank=32 | **0.826** | 0.087 | Best LoRA result (oracle c) |
| Control (same class) | 0.582-0.693 | — | Proves instance-specific leakage |

- NTK diagnostics (T=1): weight_change=0.025, feature_stability=0.749, coefficient_drift=0.500
- **Key insight**: ΔW = θ₁ - θ₀ cancels the pre-trained component, isolating the fine-tuning signal
- **Oracle coefficient caveat**: Coefficients cᵢ are currently computed from true x and passed as fixed constants. This mirrors a "best-case attacker" scenario. The structural parallel to Haim et al.'s λᵢ (Lagrange multipliers) is exact — both are scalar unknowns that should be optimized alongside x. See LESSONS_LEARNED.md for full analysis.

**Multi-seed analysis (200 seeds, oracle coefficients):**
- 22/200 (11%) seeds produce strong signal (coeff_mag > 0.1)
- 13/200 (6.5%) produce medium signal (0.01-0.1)
- 165/200 (82.5%) produce weak/no signal (< 0.01)
- Perfect correlation: model wrong after centering ↔ strong signal
- Digits 4, 5, 8, 9 over-represented in attackable seeds
- **Figures**: `figures/sprint1/multi_seed_analysis.png`

- **Figures**: `figures/sprint1/experiment_b_grid_oracle.png`, `figures/sprint1/experiment_b_grid_free.png`, `figures/sprint1/experiment_b_grid_r32.png`, `figures/sprint1/rank_sweep_sprint1.png`, `figures/sprint1/sprint1_summary.png`
  - Note: Old figures (`experiment_b_free_coeff_grid.png`, `free_coeff_reconstruction_grid.png`) were stale — showed grey/blank reconstructions due to missing ds_mean correction. Deleted and replaced with correctly rendered versions (2026-04-28). `generate_experiment_b_figure()` is now mode-aware (auto-detects oracle vs free-coefficient).

### Base Reconstruction (Haim et al.) — Complete

The original paper's pipeline is fully working end-to-end:

- **2 trained models** (both D-1000-1000-1 MLPs, 1M epochs, BCE loss, SGD):
  - CIFAR-10 vehicles vs animals (250/class) → `dataset_reconstruction/models/weights-cifar10_vehicles_animals_d250_*.pth`
  - MNIST odd vs even (250/class) → `dataset_reconstruction/models/weights-mnist_odd_even_d250_*.pth`

- **4 reconstructions** (2 per model, via W&B sweeps):
  - CIFAR-10: `reconstructions/cifar10_vehicles_animals/{b9dfyspx,k60fvjdy}_x.pth`
  - MNIST: `reconstructions/mnist_odd_even/{kcf9bhbi,rbijxft7}_x.pth`

- **Analysis notebooks** with outputs:
  - `reconstruction_cifar10.ipynb` — CIFAR-10 reconstruction visualization & metrics
  - `reconstruction_mnist.ipynb` — MNIST reconstruction visualization & metrics

- **Datasets downloaded**: MNIST, CIFAR-10 (in `dataset_reconstruction/data/`)

- **Environment**: Apple Silicon / MPS backend via `environment_macos.yaml` (Python 3.8, PyTorch 2.4.1, Kornia 0.7.0, wandb)

### Thesis Planning — Complete

- Wrote comprehensive thesis prospectus covering 3 research directions (see `papers/Thesis Ideas_ LoRA, NTK, Reconstruction.pdf`)
- Formulated the Gradient Bridge attack (see `papers/Gradient Bridge_ PEFT Privacy Attack.pdf`)
- Created phased coding roadmap: Phase 0 → Phase 1 → Phase 2 (see `notes/GRADIENT_BRIDGE_PLAN.md`)
- Detailed R2F (Recover-to-Forget) reference analysis in `CLAUDE.md`
- Collected all key reference papers in `papers/`

### Project Organization & Infrastructure (2026-02-22)

Major setup day — went from a working base reconstruction to a fully organized thesis project:

**Repository structure:**
- Organized flat directory into structured layout: `papers/`, `figures/`, `results/`, `notes/`, `experiments/`
- Created `CLAUDE.md` with full project context, theoretical foundations, and R2F deep-dive
- Created `LESSONS_LEARNED.md` with base reconstruction insights
- Created this `STATUS.md`
- Set up `.gitignore` and initialized the Thesis-level git repo (separate from `dataset_reconstruction/`)
- Cleaned up `papers/`: removed 3 duplicate/corrupted files (84 MB of junk)

**Claude Code tooling:**
- Originally set up 10 custom skills: `/review`, `/supervisor`, `/experiment`, `/debug`, `/figure`, `/paper`, `/write`, `/lesson`, `/status`, `/project-manager`
- **Lost during data loss.** Recreated 2 commands on 2026-03-19: `/research`, `/project-manager`. Others need recreation if needed.

**Theoretical analysis documents (in `notes/`):**
- `R2F_Guide.tex/.pdf` — detailed walkthrough of the Gradient Decoder mechanism from R2F
- `Inversion_Feasibility_Analysis.tex/.pdf` — information-theoretic analysis of when reconstruction is possible
- `Thesis_Direction_Analysis.tex/.pdf` — comparison of all three thesis directions with risk assessment

### Sprint 1 Experiment Code (2026-02-22) — Complete

All infrastructure code written and debugged in `experiments/`:
- `lora_wrapper.py` — LoRALinear class, apply_lora, compose_state_dict
- `data_utils.py` — few-shot MNIST loading (train + test set), control images (in-dist + OOD)
- `train_lora.py` — LoRA + full fine-tuning training loops (full-batch SGD, BCE, float64)
- `ntk_steps.py` — multi-step gradient computation, NTK coefficient extraction
- `ntk_extraction.py` — NTK reconstruction loss with oracle and free-coefficient modes, N sweep
- `ntk_verification.py` — NTK diagnostics (weight change, feature stability, coefficient drift)
- `run_experiment_a.py` — convergence + compose experiment (pre-trained init)
- `run_experiment_b.py` — multi-step NTK experiment orchestrator
- `run_sweep.py` — sweep driver for both experiments (rank × N, rank × T)
- `metrics.py` — wrapper around existing evaluations.py (SSIM, DSSIM, NCC, L2)
- `plotting.py` — publication-quality figure generation (grids, heatmaps, diagnostics)
- `configs.py` — constants, sweep grids, device auto-detection
- 5 test files in `experiments/tests/`

**Key design decisions made during implementation:**
1. Experiment A consolidated to one script using pre-trained init (deleted duplicate `run_experiment_a_v2.py`)
2. All experiments use held-out MNIST test data for fine-tuning (not train set)
3. Device auto-detection: CUDA > MPS > CPU
4. Per-image SSIM scores on reconstruction grids (not just mean)

### Early Analysis Figures

Four plots in `figures/`:
- `parameters_as_function_of_epoch.png` — parameter dynamics over training
- `parameters_as_function_of_epoch_full_fine_tune_comparison.png` — LoRA vs full fine-tune comparison
- `parameters_as_function_of_epoch_with_sweet_spot.png` — optimal reconstruction window
- `experiment_b_grid.png` — NTK experiment preview grid

---

## Current Folder Structure (as of 2026-04-09)

```
/home/projects/galvardi/yoado/     ← WEXAC home = top-level git repo
├── .gitignore
├── CLAUDE.md
├── STATUS.md                      ← this file
├── LESSONS_LEARNED.md
├── STYLE_GUIDE.md
├── papers/                        ← reference PDFs
├── figures/                       ← 12 files (incl. Phase 0 results)
├── results/                       ← 105 files (.csv metrics + .pth tensors)
├── notes/
│   └── reconstruction_approaches.tex  ← catalog of approaches (March 2026)
├── scripts/                       ← 28 WEXAC job submission scripts
│   └── wexac_logs/                ← job stdout/stderr logs
├── experiments/                   ← 25 .py files (LoRA recon + Phase 0 ViT inversion)
│   └── tests/                     ← pytest test suite
└── dataset_reconstruction/        ← original Haim et al. codebase (separate .git)
```

---

## Thesis Roadmap (updated 2026-04-09)

Sprint 2 established the NTK attack on MNIST MLPs. The path forward has three tiers, ordered by thesis impact:

### Tier 1: Gradient Bridge (highest priority — core thesis contribution)
The LoRA → full gradient → image reconstruction pipeline:
1. **Sprint 3 (current)**: Scale gradient inversion to ViT/CNN on CIFAR-10 — establishes the inversion engine
2. **Sprint 4 (future)**: Train Gradient Decoder (R2F-style) — 50K (BA, ∇_W L) pairs from proxy data, per-layer MLP, cosine sim loss
3. **Sprint 5 (future)**: End-to-end attack — frozen decoder → inversion engine → reconstructed images on victim LoRA adapter

### Tier 2: NTK Reconstruction (supporting evidence)
Differentiable unrolling (S3.4) extends the NTK approach to exact multi-step matching, removing the linearization assumption. If it works on CIFAR-10, it's a publishable improvement over Sprint 2's NTK results and provides an alternative attack path.

### Tier 3: Diffusion Priors (stretch goal)
Hybrid gradient-matching + SDS loss for low-rank reconstruction. Blocked on having a working inversion engine (Tier 1). Target: face reconstruction from ViT LoRA adapters fine-tuned on CelebA.

---

## Known Issues & Housekeeping

- **Uncommitted changes** in `dataset_reconstruction/`: `wexac_connect.sh`, `wexac_disconnect.sh` modified — likely WEXAC config tweaks
- **`settings.default.py` deleted** from git tracking in `dataset_reconstruction/` — README expects it for fresh clone setup
- **Untracked large file**: `Miniforge3-MacOSX-arm64.sh` (51 MB installer) in `dataset_reconstruction/` — already .gitignored there
- ~~**Corrupted/duplicate PDFs** in `papers/`~~ — **FIXED** (2026-02-22): removed `2407.15845` and `Djdj .15845`, kept properly named `Oz_et_al_2024_Reconstruction_Transfer_Learning.pdf`
- **No `runs/` directory** yet — gets created at runtime by Main.py

---

## Pending Tasks

### Completed
- [x] **Run Experiment A on WEXAC** — FAILED (expected): KKT can't separate fine-tuning from pre-training
- [x] **Run Experiment B on WEXAC** — SUCCESS: SSIM=0.9999 (full) / 0.797 (LoRA r=8) — oracle coefficients
- [x] **Run rank sweep (Experiment B)** — ranks 8/16/32: SSIM improves with rank — oracle coefficients
- [x] **Multi-seed analysis (200 seeds)** — 11% of seeds produce strong signal; perfect correlation with model being wrong after centering
- [x] **Generate Sprint 1 figures** — `rank_sweep_sprint1.png`, `sprint1_summary.png`, `multi_seed_analysis.png`, experiment B grids
- [x] **Identify oracle-coefficient cheating** — current NTK extraction uses true x to compute cᵢ; documented in LESSONS_LEARNED.md
- [x] **Sprint 2b Phase 0: Activation ablation** — LeakyReLU wins (stable through T=100, ReLU NaN's at T>=50)
- [x] **Sprint 2b Phase 1: SGD + free-c baseline (ReLU)** — confirms ReLU instability at T>1
- [x] **Sprint 2b Phase 2: Random restarts (LeakyReLU)** — LoRA nearly matches full model through T=100
- [x] **Activation function ablation for LoRA extraction** — ReLU (alpha=10000) + L-BFGS best for LoRA
- [x] **Free-coefficient LoRA rank sweep** — works at r=8 and r=64, stubborn at r=16/r=32

### Sprint 2a: Free-Coefficient Extraction — DONE
- [x] **Implement free-coefficient NTK extraction** — `get_coeff_penalty()`, free-c mode, N sweep
- [x] **Ablate consistency weight α** — α=1 + SGD is optimal (SSIM=0.997, matches oracle)
- [x] **LoRA rank sweep with free-c** — r=4/8/16/32/64 with ReLU + L-BFGS (WEXAC jobs 674631, 681126)
- [x] **Activation ablation** — ReLU (alpha=10000) is critical for LoRA (WEXAC job 669885)
- [x] **Separate optimizer for c** — L-BFGS for x, SGD/Adam for c (mirrors Haim et al.'s λ handling)
- [x] **Fix r=16/32 convergence** — SGD+LeakyReLU: r=16 0.624, r=32 0.680 (was 0.42)
- [x] **Multi-seed comparison** — 50 seeds: free-c (0.557) beats oracle (0.408), 46/50 wins

### Sprint 2b: Multi-Step & Scaling
- [x] **Phase 0**: Activation ablation (3 activations × 5 T values)
- [x] **Phase 1**: SGD + free-c baseline (T × rank sweep)
- [x] **Phase 2**: Random restarts
- [x] **Phase 3**: LR scaling with LeakyReLU — done as Sprint 2c B1
- [x] **Phase 4**: Progressive warm-start — done as Sprint 2c B1
- [x] **Multi-seed validation** of LeakyReLU — 30 seeds: SSIM 0.558±0.034 (T=1), 0.572±0.088 (T=10)
- [x] **Per-image SSIM** — 10 seeds saved as .pth for visual inspection

### Sprint 2c: KKT & NTK Ablations
- [x] **Track A**: CLOSED — KKT loss 330-350 for all N values, confirms structural failure
- [x] **Track B1**: Phase 3+4 (LR scheduling + warm-start) — DONE
- [x] **Track B2**: Loss ratio ablation (verify_weight) — DONE (16 configs)
- [x] **Track B3a**: Optimizer × activation for LoRA — DONE (winner: SGD + LeakyReLU, SSIM 0.830)
- [x] **Track B3b**: Scale best combo across T — DONE (SGD+LeakyReLU ≡ L-BFGS for T≤20, NaN at T=100)
- [x] **Track B4**: N sweep (NTK) — DONE
- [x] **Track B5-B8**: Additional ablations — DONE

### Phase 0: ViT Gradient Inversion
- [x] ~~**Setup phase0 conda env**~~ — not needed, `rec` env has timm+peft
- [x] **Phase 0 (fixed)**: Resubmitted with bug fixes — SSIM=0.089 (full) / 0.264 (LoRA). Poor but real signal (was 0.015 before fixes).
- [x] **D1 controlled comparison** (2026-04-14): signAdam + tv=1e-2 → SSIM=0.144 (4 configs)
- [x] **D2 hyperparameter sweep** (2026-04-28): 40 configs, best tv=1e-1 + lr=0.05 + 30K → **SSIM=0.548** — gate crossed
- [x] **Custom image loading**: `--image_path` flag + 7 unit tests
- [ ] **Face sweep (face1/2/3)**: signAdam + tv ∈ {1e-2, 5e-2} on three real face photos — submitted, in flight
- [ ] **LoRA-only at D2 winner**: rerun tv=1e-1, lr=0.05, 30K with --mode lora across rank 8/16/32/64
- [ ] **Multi-seed validation**: 5-10 seeds at the winning config for canonical SSIM mean±std
- [ ] **Phase 0b**: Noise tolerance sweep — deprioritized, folded into Sprint 3

### Sprint 3: Scaling Beyond MNIST
- [ ] **S3.1**: Phase 0 hyperparameter sweep (lr × tv_weight × optimizer × n_iters)
- [ ] **S3.2**: Low-dim reconstruction space (32×32 / Fourier / patch-aware)
- [ ] **S3.3a**: CNN baseline on CIFAR-10 (simplest architecture)
- [ ] **S3.3b**: ResNet-18 + LoRA on CIFAR-10 (skip connections)
- [ ] **S3.3c**: DeiT-Tiny on CIFAR-10 (small ViT, 5.7M params)
- [ ] **S3.4**: Differentiable unrolling — Phase 1: validate on MNIST (T=1 should match Exp B)
- [ ] **S3.4**: Differentiable unrolling — Phase 2: T=1,2,5,10,20 on MNIST vs NTK
- [ ] **S3.4**: Differentiable unrolling — Phase 3: apply to best CIFAR-10 architecture
- [ ] **S3.5**: Image priors (frequency, LPIPS, BN stats, SDS, latent-space)

### Research Backlog
- [ ] **Image priors for ViT inversion** — folded into S3.5; TV alone is insufficient at 224×224
- [ ] **N>1 superposition problem** — deprioritized until N=1 works reliably on CIFAR-10. Approaches: diversity penalty, ICA (Cocktail Party Attack), cross-gradient orthogonality
- [ ] **Read Gradient Inversion on PEFT (Sami et al., CVPR 2025)** — PEFT dimensionality reduction makes inversion *easier*; directly validates thesis. **HIGH PRIORITY** — read before starting S3.3
- [ ] **Read Cocktail Party Attack (ICML 2023)** — ICA-based gradient inversion, scales to N=1024 (needed for N>1)
- [ ] **Read SPEAR (NeurIPS 2024)** — exact batch recovery via SVD + ReLU sparsity

### Writing & Communication
- [ ] **Write LaTeX summary** — `notes/lora_reconstruction_writeup.tex` (after Sprint 3 results)
- [ ] **Email supervisor** with Sprint 2 + Phase 0 results and Sprint 3 plan
- [ ] **Verify figure quality** — publication-ready (axes, legends, DPI, colorblind-safe)

### Reading (Sprint 3 prep)
- [ ] **Read Inverting Gradients (Geiping et al.)** — the gradient inversion algorithm Phase 0 implements. **HIGH PRIORITY** — may reveal hyperparameter guidance we're missing
- [ ] Read R2F paper Section 3 in detail (decoder architecture) — needed for Sprint 4
