# Failure Analysis & Paths Forward (April 2026)

Diagnosis of why Phase 0 (ViT gradient inversion) underperformed and Sprint 2 scaling bottlenecks, with concrete approaches to fix them.

---

## Part 1: Core Issues

### Issue 1: Phase 0 Gradient Inversion on ViT (SSIM=0.015 → 0.089/0.264)

**First attempt (2026-03-27):** SSIM=0.015 — five implementation bugs:
1. **Non-differentiable cosine similarity** (ROOT CAUSE): `loss.backward()` + `param.grad` produces detached tensors — cosine sim had no gradient w.r.t. x_recon. Optimizer only minimized TV loss. Fixed with `torch.autograd.grad(..., create_graph=True)`.
2. **Per-tensor cosine sim** instead of one global flattened cosine sim (Geiping et al. standard).
3. **LoRA-only gradients** (294K params) instead of full gradient (86M) — mismatch in scale.
4. **Flash attention incompatible with create_graph=True** — need math-only SDPA backend.
5. **requires_grad state mismatches** — params frozen after capture step, then queried in inversion loop.

**Second attempt (2026-04-07, bugs fixed):** SSIM=0.089 (full, 86M params) / 0.264 (LoRA-only, 294K params). Better but still poor. Reconstructions show correct color palette and vague boat shape — real signal exists, but insufficient for useful reconstruction.

**Why this matters:** All five bugs are implementation details, not fundamental theory failures. But the post-fix SSIM of 0.089 points to a broader issue: **gradient inversion from ViT is intrinsically harder than from MLPs.** The 86M-parameter gradient space is high-dimensional, noisy, and involves attention mechanisms that complicate differentiable double-backward.

**Surprising finding:** LoRA-only (294K params, SSIM=0.264) outperformed full-model (86M params, SSIM=0.089). The lower-dimensional optimization landscape is easier to navigate, even with less gradient information. This echoes Sami et al. (CVPR 2025) who showed PEFT dimensionality reduction can make inversion *easier*.

**Untuned hyperparameters:** lr=0.1, tv_weight=1e-4, Adam, 10K iters — all defaults, never swept. Sprint 2 required 148+ configs of ablation to find optimal settings. Phase 0 had exactly one attempt.

### Issue 2: LoRA Scaling Bottleneck (r=16, r=32 stuck at SSIM ~0.42)

**Root cause (Sprint 2c Track B3a results):** NTK reconstruction works well at r=8 (SSIM 0.617) and r=64 (SSIM 0.635), but r=16 and r=32 converged poorly — coefficient error ~0.28-0.31 across all optimization attempts.

**Diagnosis:** At intermediate ranks, the LoRA column space captures ~80-90% of the gradient signal, but the remaining 10-20% projects to noise that actively misleads coefficient optimization. The optimizer can't disambiguate "is my coefficient estimate wrong, or is the image reconstruction wrong?"

**Resolution (Sprint 2c):** SGD+LeakyReLU fixed both: r=16 → 0.624 (was 0.422), r=32 → 0.680 (was 0.415). The activation function and optimizer choice were the bottleneck, not a fundamental rank-mismatch issue.

**Remaining gap:** r=16/32 still lag behind r=8 and r=64 in free-coefficient mode. The rank-mismatch hypothesis may still apply partially.

### Issue 3: Multi-Step Accumulation (T>20) Requires Special Care

**Observations from Sprint 2b:**
- T=1: SSIM 0.997 (full), 0.557 (free-c LoRA)
- T=20: Still works, SGD matches L-BFGS
- T=100: LeakyReLU stable (~0.78-0.80 full), but requires specific activation + random restarts
- T=100 + ReLU: NaN everywhere

**Problem:** The NTK approximation `ΔW ≈ -η Σ cᵢ ∇f(θ₀; xᵢ)` assumes frozen features, which breaks as θ shifts. By T=20, error accumulation is significant. By T=100, activation smoothness (LeakyReLU vs ReLU) matters more than the broken linearization assumption.

**Implication for thesis:** Realistic LoRA fine-tuning uses T=100-10000 steps. The NTK approach is fundamentally limited for this regime. Differentiable unrolling (Approach G) is needed.

---

## Part 2: Why ViT Specifically Fails

| Component | MNIST FCN | ViT-B/16 | Effect on Reconstruction |
|-----------|-----------|----------|--------------------------|
| Parameters | 2M | 86M | Gradient space is 43x higher-dimensional |
| Attention | None | 12 layers x 12 heads | Nonlinear softmax, memory-intensive double-backward |
| Patch embedding | N/A | 197x768 | Spatial structure lost, dense gradients hard to constrain |
| Activation | ReLU (piecewise linear) | GELU (smooth sigmoid-like) | Smoother gradients but less interpretable NTK |
| Prior strength | MNIST manifold is tiny | ImageNet: huge diversity | Hard to find natural image in huge space via gradient descent |
| Image dims | 784 (28x28 grayscale) | 150,528 (224x224x3) | 192x more unknowns to solve for |

**Why smaller models excel:**
- **Conv2-3 on CIFAR-10 (~500K params):** Lower-dimensional than MNIST FCN but color images. Prior: CIFAR-10 manifold is well-studied.
- **ResNet-18 on CIFAR-10 (11M params):** Manageable gradient space, skip connections stabilize NTK, good baseline.
- **DeiT-Tiny (5.7M params):** Same ViT architecture but 15x smaller. Tests whether ViT is the issue, or just its scale.

---

## Part 3: Concrete Approaches

### Approach A1: Phase 0 Hyperparameter Sweep (S3.1)
- **Hypothesis:** SSIM=0.089 is due to untuned defaults, not fundamental ViT limitation.
- **Design:** Sweep lr x tv_weight x optimizer x n_iters. 2-stage: coarse (30 configs) then fine (top-5 at 30K iters).
- **Gate:** SSIM > 0.3 (full model) → proceed to priors (S3.5). SSIM < 0.15 for all → architectural bottleneck → S3.3.
- **Time:** 1-2 days.

### Approach A2: Low-Dimensional Reconstruction Space (S3.2)
- **Hypothesis:** Optimizing in 224x224 space (150K dims) is wasteful when source is 32x32 CIFAR-10 (3K dims actual info).
- **Design:** Optimize in 32x32 and upsample; or Fourier-truncated parameterization; or patch-aware (196 x ~50 dims).
- **Gate:** Any variant SSIM > 0.3 → search space was the bottleneck. All < 0.15 → gradient signal too weak → S3.3.
- **Time:** 1-2 days.

### Approach A3: Simpler Architectures on CIFAR-10 (S3.3)
- **S3.3a: Small CNN** (~200K params, Conv-ReLU-Pool-Conv-ReLU-Pool-FC). Both NTK (Exp B style) and gradient inversion.
- **S3.3b: ResNet-18** (11M params, pretrained, LoRA r=8/16). Gradient inversion.
- **S3.3c: DeiT-Tiny** (5.7M params, same ViT arch but 15x smaller). Gradient inversion.
- **Gate:** CNN SSIM > 0.4 + ViT < 0.2 → architecture is bottleneck. Both > 0.3 → method works across architectures. Both < 0.15 → gradient inversion on color images is fundamentally harder.
- **Time:** 2-4 days total.

### Approach B1: Differentiable Unrolling (S3.4) — HIGHEST THESIS IMPACT
- **Hypothesis:** Instead of inverting gradients (brittle create_graph=True), directly simulate fine-tuning and match resulting weights to observed weights.
- **Method:** Outer loop optimizes x_recon. Inner loop simulates T differentiable SGD steps. Outer loss: ||theta_simulated - theta_observed||^2.
- **Key advantages:** Exact for any T (no NTK linearization error), no coefficient estimation needed, architecture-agnostic, feeds directly into Gradient Bridge pipeline.
- **Validation:** T=1 on MNIST should reproduce Experiment B (SSIM ~0.997).
- **Gate:** T=1 matches → validates implementation. T=10 beats NTK → publishable improvement.
- **Time:** 3-5 days.

### Approach C: Stronger Image Priors (S3.5)
Prior hierarchy (ordered by complexity):
1. **Frequency-domain** (0.5 day): Penalize high-frequency Fourier components.
2. **LPIPS perceptual loss** (1 day): Frozen DINO/ResNet50 features for self-regularization.
3. **BN statistics** (0.5 day): Match running mean/var of reconstruction (ResNet/CNN only).
4. **Score Distillation Sampling** (2-3 days): Frozen diffusion model guides toward natural images.
5. **Latent-space reconstruction** (1-2 days): Optimize in VAE latent space (~4K dims instead of 150K).

---

## Part 4: Priority Ranking

| # | Approach | Goal | Time | Risk | Impact |
|---|----------|------|------|------|--------|
| 1 | S3.1: Hyperparam sweep | Quick diagnostic | 1-2d | Low | Medium |
| 2 | S3.4: Unrolling (MNIST) | Exact multi-step method | 3-5d | Low | **High** |
| 3 | S3.2: Low-dim recon space | Amplify ViT signal | 1-2d | Low | Medium |
| 4 | S3.3a: CNN on CIFAR-10 | Architecture isolation | 1-2d | Low | High |
| 5 | S3.5: Image priors | Fix ViT signal | 2-4d | Medium | High |
| 6 | S3.3c: DeiT-Tiny | Small ViT test | 1d | Low | Medium |

**Critical path:** S3.4 (differentiable unrolling) feeds directly into the Gradient Bridge pipeline (Tier 1 thesis contribution). S3.1-S3.3 are supporting experiments that de-risk architecture choices.

---

## Part 5: Conditions Being Relaxed

| Condition | Status Quo | Simplified | Benefit |
|-----------|-----------|------------|---------|
| Model complexity | ViT-B (86M, 12 attn layers) | CNN/ResNet (0.2-11M) | 7-430x simpler gradient, no double-backward |
| Reconstruction space | 224x224 pixel space (150K) | 32x32 or Fourier (3K) | 49x smaller search space |
| Linearity assumption | NTK valid for T=1, breaks T>20 | Differentiable unrolling | Exact for any T |
| Gradient approximation | Cosine sim inversion | Direct weight matching | No cosine-similarity brittleness |
| Image prior | TV only | TV + frequency + LPIPS + SDS | Tighter manifold constraint |

---

## Part 6: Decision Tree

```
Start
  |
  v
S3.1: Phase 0 hyperparam sweep
  |
  +-- SSIM > 0.3 --> S3.5: Add priors --> Scale to faces/CelebA
  |
  +-- SSIM < 0.15 --> Architecture problem
        |
        v
      S3.3a: CNN on CIFAR-10
        |
        +-- SSIM > 0.4 --> Architecture is bottleneck
        |     |            Focus thesis on CNN/ResNet LoRA reconstruction
        |     v
        |   S3.3c: DeiT-Tiny (is it ViT-specific or just scale?)
        |
        +-- SSIM < 0.15 --> Gradient inversion on color images is hard
              |
              v
            S3.4: Differentiable unrolling (bypass inversion entirely)
              |
              +-- T=1 matches Exp B --> Green light, scale to CIFAR-10
              |
              +-- T=1 fails --> Implementation bug, debug

In parallel (always):
  S3.4 Phase 1: Validate unrolling on MNIST (independent of ViT experiments)
```

---

## Part 7: Thesis Impact Scenarios

**Best case (ViT inversion works after tuning + priors):** Full pipeline: LoRA adapter → gradient inversion → reconstructed images. Directly proves privacy risk of PEFT. Extends to Gradient Bridge (Tier 1).

**Good case (Unrolling works, ViT inversion marginal):** Differentiable unrolling as primary attack. Works across architectures. Extends NTK to multi-step regime. Strong thesis contribution.

**Acceptable case (CNN/ResNet works, ViT fails):** "Privacy leakage through LoRA adapters on convolutional architectures." Still novel, still publishable. ViT failure is a documented negative result.

**Worst case (none work on CIFAR-10):** Thesis pivots to: "NTK reconstruction works on small models; scaling requires fundamentally different attacks (LoRA spectrum analysis, membership inference)." Sprint 2 results still stand.

---

*Generated 2026-04-09. Based on Sprint 1-2c experiment results and Phase 0 (ViT gradient inversion) outcomes.*
