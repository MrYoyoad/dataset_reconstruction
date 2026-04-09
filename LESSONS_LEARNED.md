# Lessons Learned

Running log of insights, pitfalls, and things to remember as the thesis progresses.

---

## Infrastructure & Data Loss (2026-03-19)

### The Git Repo That Never Was
The git repo WAS initialized and pushed to `myfork/main` on GitHub — but the WEXAC working directory lost its connection to the remote. The `.git` was either deleted during a reprovisioning or never properly set up on WEXAC. The WEXAC copy had newer experiment code/results (Sprint 2+) that were never pushed, while GitHub had all the papers, notes, and figures.

**What was actually lost:**
- All Claude Code conversation history from Jan 15 – Mar 18
- 10 custom Claude Code skills (8 still need recreation)
- Some generated figures not on GitHub (`multi_seed_analysis.png`, `sprint1_summary.png`, etc.)

**What was recovered from GitHub:**
- All 18 papers, 7 notes files, 4 figures, full experiment infrastructure

**Lessons:**
1. **Always verify git operations actually succeeded.** `git init` + `git add` + `git commit` — check `git status` after each.
2. **Push to GitHub after every significant commit.** The remote saved everything this time.
3. **WEXAC home dirs can lose local state.** Untracked files are unprotected files.
4. **Claude Code conversation history is ephemeral.** Important findings must go into STATUS.md / LESSONS_LEARNED.md.
5. **Don't have nested .git repos.** `dataset_reconstruction/` having its own git caused confusion. The top-level repo now tracks it as regular files.

---

## Base Reconstruction (Haim et al.)

### Setup & Environment
- Apple Silicon (MPS backend) works but watch for dtype mismatches — MPS doesn't support all float64 ops.
- The `settings.py` file with relative paths (`./data/`, `./runs/`, `./models/`) keeps things portable.
- **Primary compute is WEXAC cluster**, not the MacBook. GPU: NVIDIA L40S (46 GB VRAM), CUDA 12.6. Connect via `wexac_connect.sh` (requires Weizmann VPN). Conda env on cluster: `/home/projects/galvardi/yoado/.conda/envs/rec`.
- The L40S easily handles all planned experiments: ViT LoRA fine-tuning (~2-3 GB), gradient decoder training (~50k pairs), gradient inversion (~4-8 GB), and even Stable Diffusion for SDS priors (~8-12 GB). **Compute is not a bottleneck.**

### Training
- Models need to train to near-stationarity (very long — 1M epochs) for the KKT conditions to hold. Don't cut training short.
- `ModifiedReLU` is critical — standard ReLU gives much worse reconstruction because the smooth gradients matter during extraction.
- BCE loss (not cross-entropy) is required for the implicit bias / max-margin convergence theory to apply.

### Reconstruction
- KKT loss optimization is sensitive to initialization — random restarts help.
- Lambda (Lagrange multiplier) optimization needs a separate, typically smaller learning rate.
- The number of reconstructed samples should match the actual training set size for best results.

---

## LoRA / Gradient Bridge

### Key Realizations
- The Gradient Bridge is theoretically sound and all building blocks exist independently (R2F for decoding, Inverting Gradients for inversion). But **"building blocks exist" ≠ "easy to do"** — the real research risks are empirical, not computational:
  1. **Decoder accuracy for pixel-level reconstruction**: R2F proved the decoder works for unlearning (tolerant of noisy gradients). Nobody has shown it works for pixel-level image reconstruction, which is far more sensitive to gradient noise. Even 0.9 cosine similarity may not be enough.
  2. **Multi-step accumulation**: The decoder is trained on single-step LoRA updates. Real adapters train for thousands of steps. How to handle accumulated updates is an open question.
  3. **Error compounding**: LoRA approximation × decoder approximation × inversion approximation — each stage is "pretty good" but errors multiply through the pipeline.
- These are answerable by running experiments, and we have the compute (L40S) to run them fast.
- The correct strategy is to de-risk in order: (1) Sprint 1 compose-and-reconstruct, (2) Phase 0 "cheating" with perfect gradients to find the ceiling, (3) only then attempt the decoder. If Phase 0 fails, the decoder won't save it.

### What Worked
- **Experiment B (NTK, 1-step) works because it targets ΔW not W.** By reconstructing from the weight *change* (θ_T - θ₀), the pre-trained component cancels out. Full model SSIM=0.9999, LoRA rank 8 SSIM=0.797, rank 16 SSIM=0.802, rank 32 SSIM=0.826. This proves the gradient from a single fine-tuning step leaks private data.

### What Didn't Work
- **Experiment A (compose + KKT) completely fails with pre-trained init — and the reason is structural, not just slow convergence.** The composed model W = W₀ + BA is just a set of weights. The KKT reconstruction asks: "what training data would produce these weights as the max-margin solution?" The answer is: **all 502 samples** the model was effectively trained on (500 pre-training + 2 fine-tuning). The KKT stationarity condition is W ∝ Σᵢ₌₁⁵⁰² λᵢ yᵢ ∇_W Φ(W; xᵢ), but the extraction sets `extraction_data_amount = 2` — asking 2 images to explain weights that encode 502 images of information. The pre-training residual W₀ contains contributions from ~100-250 original support vectors. The KKT loss of ~460 is essentially ||W₀||² — the huge unexplained pre-training component. **Even with perfect convergence, the extraction would still fail** because the composed weights satisfy KKT with respect to all 502 samples, and 2 images can't explain them. (Note: the 2 fine-tuning samples ARE on the margin for the N=2 case — with 2M+ params and 2 points, both are necessarily support vectors. The issue is the other 500 samples baked into W₀.) **This is the key negative result that motivates the Gradient Bridge.**
- **Reconstruction is sensitive to seed.** With seed=42, full model NTK reconstruction achieves SSIM=0.9999. With seed=32, SSIM=0.378 despite perfect NTK conditions (feature_stability=1.0000). The extraction optimizer gets stuck in local minima depending on which digits are selected and how x is initialized. Random restarts are essential.

### Pitfalls to Avoid
- **W₀ must be pre-trained, not random.** The original Experiment A design used random init as W₀. This doesn't match the thesis's attack model (pre-trained model → LoRA fine-tune on private data → attacker reconstructs). Fixed by updating `run_experiment_a.py` to load the pre-trained MNIST model as W₀. The same lesson applied to Experiment B (random init destroys NTK feature stability). In both cases: **always start from pre-trained weights** — that's the realistic scenario.
- **Held-out data, always.** Fine-tuning data must come from the MNIST test set, not the train set. The pre-trained model already converged on its training data (gradients ≈ 0), so fine-tuning on overlapping data is meaningless.
- **Max-margin theory assumes small init.** The pre-trained model has layer weights with std 0.004–0.12, much larger than the 0.0001 init the theory requires. The reconstruction may fail in A-v2 for this reason — which would actually strengthen the argument for the Gradient Bridge approach (Phase 1-2).
- **Consolidate, don't duplicate.** Initially created `run_experiment_a_v2.py` as a separate script for the pre-trained init scenario. But then `run_experiment_a.py` was also updated to use pre-trained init, creating an identical duplicate. Deleted A-v2; keep one canonical script per experiment.

---

## NTK Regime & High-Rank LoRA

### Key Realizations
- **Start from pre-trained weights, not random init.** The original Experiment B design used random init with `init_scale=0.0001`, which put ALL pre-activations within ±0.002 of the ReLU kink. After 1 SGD step, ~half the neurons flipped on↔off, destroying NTK feature stability (cosine similarity 0.39 instead of >0.99). The NTK approximation was completely invalid, and all reconstructions were noise (SSIM ~0.35).
  - **Root cause**: `init_scale=0.0001` was inherited from the convergence/max-margin pipeline (Experiment A / Haim et al.), where tiny init helps the implicit bias theory. But for NTK, you need pre-activations AWAY from the ReLU kink. The pre-trained model (`weights-mnist_odd_even_d250`) has layer 0 std=0.004 (40× larger), giving pre-activation std=0.25 — well away from zero.
  - **Fix**: Load pre-trained weights as θ₀ and fine-tune on **held-out data** (MNIST test set, not the 250/class training data). With pre-trained weights, feature stability = 1.000 at T=1.
  - **The correct attack scenario**: Pre-trained model + LoRA fine-tuning on private data → attacker reconstructs private data from ΔW = BA. This matches real-world LoRA usage.
- **Fine-tuning data must not overlap with pre-training data.** The pre-trained model was trained on first 250 even + first 250 odd digits (sequential, no shuffle from MNIST train set). If you fine-tune on samples already in the training set, the gradient is essentially zero (coefficients ≈ 1e-18) because the model already converged on them. Use MNIST test set or clearly held-out indices.
- **The pre-training data selection is deterministic**: `shuffle=False, start=0, end=50000` in `mnist_odd_even.py`, so the exact 250/class samples are reproducible.

### What Worked
- **Pre-trained weights + held-out data → SSIM=0.996 (full) / 0.793 (LoRA r=8) at T=1.** A single gradient step from a pre-trained model leaks private fine-tuning data with near-perfect fidelity (full model) and recognizable digit structure (LoRA r=8). Control images (different instances of same digit) score 0.568, proving instance-specific leakage. Feature stability = 0.75 — not ideal (>0.99 target), but good enough.
- **NTK loss decreases steadily**: 0.25 → 9.4e-4 in 5K epochs (compare to stuck at 1.85e-5 with random init). Real gradient signal (coefficients ±0.5) makes the optimization landscape tractable.

### What Didn't Work
- Random init with init_scale=0.0001 for NTK experiments → all pre-activations at ReLU kink → destroyed NTK feature stability → noisy reconstructions

---

## ViT Gradient Inversion (Phase 0) — 2026-04-09

### Key Realizations
- **ViT gradient inversion is a fundamentally harder optimization problem than MNIST MLP reconstruction.** The gap between MNIST MLP (SSIM=0.997 at T=1) and ViT-B/16 (SSIM=0.089 full, 0.264 LoRA) is not just about fixing bugs — it reflects the jump from 784-dim grayscale to 150K-dim RGB, from 2M-param MLP to 86M-param transformer, and from well-conditioned piecewise-linear activations to attention + GELU.
- **LoRA-only inversion (294K params) outperformed full-model (86M params): SSIM 0.264 vs 0.089.** This is counterintuitive but makes sense: with fewer gradient dimensions, the cosine similarity optimization landscape is smoother and the optimizer makes better progress. The full-model gradient has more information but the optimizer can't exploit it in 10K iterations. This echoes Sami et al. (CVPR 2025) who showed PEFT dimensionality reduction can make inversion *easier*.
- **Hyperparameters were never tuned.** The Phase 0 config (lr=0.1, tv_weight=1e-4, Adam, 10K iters) was a one-shot default. The MNIST experiments went through extensive ablation (Sprint 2c: 148 configs). Giving Phase 0 the same treatment could significantly improve results.
- **Image priors matter more as dimensionality grows.** MNIST's 784-dim space is small enough that box constraints (x ∈ [-1,1]) suffice. At 150K dims (224×224 RGB), the optimizer needs TV, perceptual, frequency, or generative priors to stay on the natural image manifold.

### What Worked
- Bug fixes (create_graph=True, global cosine sim, full-model gradient) raised SSIM from 0.015 to 0.089/0.264. The bugs were real and the fixes were necessary.
- LoRA-only mode is a viable (and surprisingly effective) simplification for gradient inversion.
- Reconstructions show correct color palette and vague shape of the boat — real signal exists, just not enough to be useful yet.

### What Didn't Work
- 86M-parameter full-model gradient inversion with default hyperparameters (SSIM=0.089).
- TV-only prior at 224×224 — too weak to constrain the search space.
- Single seed/image — no multi-seed statistics to distinguish bad luck from bad method.

### Pitfalls to Avoid
- **Don't conclude "ViT inversion doesn't work" from one untuned run.** Sprint 2 showed how much hyperparameter tuning matters (seed=42 outlier at SSIM=0.830 vs 50-seed mean 0.558).
- **Don't skip the dimensionality ladder.** Going from 784-dim MNIST to 150K-dim ImageNet in one jump is asking for trouble. CIFAR-10 (3K dims) is the natural stepping stone.
- **Attention double-backward is fragile.** Must use SDPA math-only backend (no flash attention). Memory-intensive. Consider whether differentiable unrolling (Approach G) can bypass this entirely.

---

## Discrete Sequence Reconstruction (LLMs)

*(Fill in as you go)*

### Key Realizations
-

### What Worked
-

### What Didn't Work
-

---

## Diffusion Priors / SDS

*(Fill in as you go)*

### Key Realizations
-

### What Worked
-

### What Didn't Work
-

---

## Document & Presentation Generation

### Audit Process
- **Fix numbers first** → grep all files for stale values (not just the ones you think are affected). `grep -rn "old_value" **/*.tex **/*.py` catches stragglers.
- **Fix examples second** → cross-reference every example against source data.
- **Add new content third** → new slides, sections, definitions.
- **Sync between formats** → PPTX and Beamer (or any parallel formats) must match.
- **Polish last** → speaker notes, transitions, naming consistency.
- **Always do a final audit** → the first sweep ALWAYS misses some stale references.
- **Commit after each phase** — makes rollback possible if a later phase breaks something.

### Common Bugs
- **Stale numbers in speaker notes**: Notes are invisible in PDF/PPTX so easy to forget. Always grep notes too.
- **Same example, different numbers on different slides**: Always cross-reference.
- **Illustrative vs actual data**: Pedagogical slides with simplified examples contradict real data slides.
- **Table overflow**: Always verify `table_y + (rows+1) * row_height < next_element_y`.

### OOXML Animation (python-pptx)
- `python-pptx` has **NO animation API** — must write raw OOXML XML via `lxml.etree`.
- **3-level par nesting is non-negotiable**: `par(delay="indefinite") > par(delay="0") > par(clickEffect)`. Skip a level and PowerPoint silently ignores all animations.
- **`para_build` must default to `False`** — when `True`, multi-paragraph text shapes get hidden on entry.
- **Card animations must include ALL child shapes** in the same `anim_groups` entry.
- **Never regex-manipulate OOXML namespaces** — stripping `xmlns:p14` once caused PowerPoint to blank an entire slide.

### Plot-Presentation Integration
- matplotlib defaults are fine for standalone analytical plots, but plots embedded in dark-background slides need dark-theme variants.
- Always test embedded plots at **50% zoom** — simulates projector distance readability.

### Generator Architecture (>500 lines)
- Refactor into a package: slim orchestrator + config.py + helpers.py + per-section slide modules.
- Each slide is a function. Auto-number via a counter. Centralize image paths in config.

### Narrative Framing (Academic)
- Frame negative results as "successfully identified the bottleneck" rather than "the approach failed."
- Consolidate fine-grained failure modes into 3-5 categories that map to remediation strategies.
- One concept per slide. If you need "Part 1" and "Part 2" labels, split into two slides.

---

## General Research Process
- **"The math says it works" vs. "you can make it work"** are different claims. Optimistic theoretical analyses (like the Gradient Bridge feasibility argument) are correct about the information being there, but gloss over engineering gaps and empirical unknowns. Calibrate accordingly: the idea is sound, but the execution is the hard part — which is exactly what makes it a thesis.
- **De-risk before building**: always run the cheapest experiment that could falsify your approach before investing weeks in the full pipeline.
- **ALL experiments run on WEXAC GPU, not MPS.** The MacBook's MPS backend is for light local dev/debugging only. Real training, reconstruction, and any serious compute must run on the WEXAC cluster (NVIDIA L40S). MPS is too slow, has dtype limitations, and results won't match CUDA. Always use `wexac_connect.sh shell` to get a GPU node before running experiments.
- **WEXAC `rec` env has PyTorch 2.4.1+cu121** (with timm 0.9.12, peft 0.7.1, torchvision 0.19.1). Use `weights_only=False` in `torch.load()` to suppress FutureWarnings. The old claim of "PyTorch 1.11" was stale documentation.

---

## [INSIGHT] L-BFGS vs SGD for NTK Extraction (2026-02-22)

**Context:** Implementing and comparing optimizers for the NTK reconstruction loss in Experiment B. The supervisor suggested L-BFGS as a better optimizer for this small-scale least-squares problem (~1,570 unknowns).

**Lesson:** L-BFGS converges extremely fast but gets trapped in a shallow local minimum for the full-model case. SGD with momentum is slower but finds a much better solution. For LoRA extraction, both hit the same loss plateau — the bottleneck is the irreducible rank mismatch, not the optimizer.

**Details:**
- **Full model**: L-BFGS drops loss from 0.25 → 0.005 in the *first step* (20 func evals), then stalls at SSIM ≈ 0.82. SGD with momentum takes 50K epochs to reach loss 9.4e-4, but achieves SSIM ≈ 0.996. The NTK landscape is non-convex (ReLU kinks), and SGD's momentum helps escape shallow basins that trap L-BFGS.
- **LoRA r=8**: Both optimizers converge to loss plateau ~3.3 with SSIM ~0.80–0.84. The plateau is an *irreducible residual* from rank mismatch: the predicted gradient is full-rank but the target ΔW lives in the rank-r column space of B₀.
- **LoRA subspace projection**: Projecting both target and prediction into col(B₀) via P = B₀(B₀ᵀB₀)⁻¹B₀ᵀ *hurts* reconstruction (SSIM drops from 0.78 → 0.49). The null-space gradient components, while they can't match the target, provide useful optimization signal that guides x toward the right solution.
- Relevant files: [experiments/ntk_extraction.py](experiments/ntk_extraction.py) (L-BFGS closure, projection function), [experiments/run_experiment_b.py](experiments/run_experiment_b.py) (LoRA extraction without projection).

**Action:** Use SGD for publication-quality full-model results (SSIM > 0.99). Use L-BFGS for quick preliminary LoRA rank sweeps (same quality in 1/50th the epochs). Do NOT project into LoRA subspace — keep the unprojected loss.

**Update (Sprint 2c B3b, 2026-03-26):** SGD + LeakyReLU matches L-BFGS identically for T ≤ 20 (SSIM within ±0.003), but NaN's at T=100 where L-BFGS still works (0.775-0.809). For multi-step extraction beyond T=20, L-BFGS remains essential. For the realistic few-shot regime (T ≤ 20), SGD is a viable and simpler alternative.

---

## [INSIGHT] NTK Coefficients Are "Cheating" — Must Be Free Parameters (2026-02-22)

**Context:** The NTK reconstruction loss is ||ΔW + η Σ cᵢ ∇f(θ₀; xᵢ)||² where cᵢ = (σ(f(θ₀; xᵢ)) - yᵢ)/N. The current code ([experiments/ntk_steps.py:28-33](experiments/ntk_steps.py#L28-L33)) computes cᵢ from the **true private data x** and passes them as fixed constants to the extraction ([experiments/ntk_extraction.py:8](experiments/ntk_extraction.py#L8)). This is cheating — in a real attack, the adversary has θ₀ and ΔW but not x, so they cannot compute cᵢ.

**Lesson:** The coefficients cᵢ must be treated as **free optimization variables** alongside x, exactly as Haim et al. treat the Lagrange multipliers λᵢ in KKT reconstruction. The structural parallel is exact:

| | Haim et al. (KKT) | NTK (current, cheating) | NTK (correct) |
|---|---|---|---|
| **Equation** | W ∝ Σ λᵢ yᵢ ∇Φ(W; xᵢ) | ΔW = -η Σ cᵢ ∇f(θ₀; xᵢ) | same |
| **Optimize over** | x AND λ | x only (c fixed from true x) | x AND c |
| **Scalar unknowns** | λᵢ ≥ 0.05 (penalized) | — | cᵢ ∈ [-1, 1] (penalized) |

**Why this matters beyond honesty:**
1. **Multi-step (T > 1)**: The current code uses `coefficients_at_init`, which is only exact for T=1. With free c, the optimizer finds effective average coefficients that explain the cumulative ΔW without requiring the frozen-feature NTK assumption.
2. **Smooths the optimization**: Decouples the x→c→ΔW chain, avoiding the chicken-and-egg problem where you need good x to compute good c and vice versa.
3. **Adds only N scalars** (N=2 in our case) to the optimization — negligible cost.

**Haim et al. λ regularization (reference for our c penalty):**
- Separate optimizer: `SGD([λ], lr=1e-4)` — 100× smaller lr than x
- Lower bound: `5 * (-λ + 0.05).relu().pow(2).sum()` — keeps λ ≥ min_lambda
- Init: `torch.rand(N, 1)` — uniform [0, 1]
- See [dataset_reconstruction/extraction.py:43-51,79-85](dataset_reconstruction/extraction.py#L43-L85)

**Risk:** Non-uniqueness — large c with wrong x could explain ΔW as well as true c with true x. Mitigate with a **self-consistency penalty**: `α * |cᵢ - (σ(f(θ₀; xᵢ)) - yᵢ)/N|²` which encourages the free c to agree with what the model would actually produce on the current x estimate. This is a soft constraint — start with α=0 (pure free c), ablate over α.

**Action:** Implement free-coefficient NTK extraction as the new default. Keep the "known coefficients" mode as a diagnostic/oracle baseline for comparison.

---

## [RESULT] Free-Coefficient Extraction Works — Consistency Penalty Is Essential (2026-02-22)

**Context:** Implemented free-coefficient mode in [experiments/ntk_extraction.py](experiments/ntk_extraction.py) and ran the α ablation on seed=42, T=1, full model.

**Results (full model, seed=42, T=1):**

| Mode | Optimizer | Epochs | SSIM | Coeff Error | Notes |
|------|-----------|--------|------|-------------|-------|
| Oracle (cheating) | L-BFGS | 500 | 0.817 | 0 (fixed) | Upper bound with L-BFGS |
| Free c, α=0 | L-BFGS | 500 | 0.282 | 1.066 | **Signs flipped** — non-unique solution |
| Free c, α=1 | L-BFGS | 500 | 0.777 | 0.005 | Correct signs, near-oracle c |
| Free c, α=10 | L-BFGS | 500 | 0.638 | 0.005 | Over-penalized, hurts NTK fit |
| **Free c, α=1** | **SGD** | **5000** | **0.997** | **0.0004** | **Matches oracle — attack works honestly** |
| Free c, α=1 LoRA r=8 | L-BFGS | 500 | 0.539 | 0.244 | One coeff stuck near 0 |

**Key lessons:**
1. **α=0 (pure free c) fails**: L-BFGS finds a sign-flipped solution where c₁ and c₂ swap signs. The NTK loss is equally satisfied because flipping c and replacing x with a "mirror" image produces the same ΔW. This is the non-uniqueness risk we predicted.
2. **α=1 (moderate consistency) is the sweet spot**: The self-consistency penalty `|c - (σ(f(θ₀;x))-y)/N|²` breaks the sign ambiguity by coupling c to the model's actual predictions on the current x. With α=1, coefficients converge to within 0.005 of oracle values.
3. **α=10 over-constrains**: The consistency penalty dominates the NTK loss, preventing x from moving freely. SSIM drops to 0.638 despite correct c.
4. **SGD beats L-BFGS again**: Free-c + SGD + α=1 achieves SSIM=0.997, essentially matching oracle SGD. L-BFGS gets trapped in shallow minima (consistent with earlier L-BFGS vs SGD findings).
5. **LoRA needs SGD**: With L-BFGS and LoRA r=8, one coefficient gets stuck near 0 while the other converges. SGD's separate optimizer with smaller lr should fix this.

**The punchline: the "cheating" didn't matter.** For the full-model case with the right optimizer (SGD) and regularization (α=1), free-coefficient extraction matches oracle quality. The attack is honest AND effective.

**Relevant files:**
- [experiments/ntk_extraction.py](experiments/ntk_extraction.py) — `get_coeff_penalty()`, `run_ntk_extraction()` with `free_coefficients=True`
- [experiments/run_experiment_b.py](experiments/run_experiment_b.py) — `--free_coefficients`, `--consistency_weight`, `--n_sweep` flags
- [experiments/configs.py](experiments/configs.py) — `COEFF_LR`, `COEFF_BOX_WEIGHT`, `COEFF_CONSISTENCY_WEIGHT`

**Action:** Use `--free_coefficients --consistency_weight 1.0 --optimizer sgd` as the default for all future runs. Run LoRA rank sweep with this config on WEXAC.

---

## [RESULT] Activation Function Is Critical for LoRA Extraction (2026-02-22)

**Context:** Sprint 1 (L-BFGS, implicit ReLU) got LoRA r=8 SSIM=0.797, but Sprint 2a (SGD, ModifiedRelu alpha=150) got SSIM=0.183. Ran an activation ablation to disentangle optimizer vs activation effects.

**Results (LoRA r=8, oracle coefficients, seed=42, T=1):**

| Alpha | L-BFGS (SSIM) | SGD (SSIM) |
|-------|---------------|------------|
| 10 (very smooth) | 0.044 | 0.177 |
| 50 | 0.126 | 0.149 |
| 150 (default) | 0.184 | 0.183 |
| **10000 (≈ ReLU)** | **0.744** | **0.467** |

**Free-coefficient (SGD, consistency α=1):**

| Alpha | SSIM | Coeff Error |
|-------|------|-------------|
| 10 | 0.414 | 0.212 |
| 10000 | 0.476 | 0.234 |

**Key lessons:**
1. **ModifiedRelu actively hurts LoRA extraction.** At alpha=150, both L-BFGS and SGD get SSIM ~0.18. At alpha=10000 (≈ plain ReLU), L-BFGS jumps to 0.744. The sigmoid-modulated gradients in ModifiedRelu create smooth but incorrect gradients when the LoRA subspace projection interacts with the activation's non-linearity.
2. **For LoRA, L-BFGS + ReLU is the best combo.** L-BFGS (0.744) >> SGD (0.467) at alpha=10000 — opposite of the full-model result where SGD wins. The LoRA extraction landscape is smoother (lower effective dimension due to rank constraint), favoring L-BFGS.
3. **ModifiedRelu was tuned for Haim et al.'s KKT extraction**, which optimizes W ∝ Σ λᵢ yᵢ ∇Φ(W; xᵢ) — the model is evaluated at the *extraction point*, so smooth gradients through the model help. NTK extraction evaluates at frozen θ₀ — the model is just a fixed feature extractor, and smooth gradients don't help (and actually hurt by introducing approximation error).
4. **Previous lesson partially wrong:** The earlier "L-BFGS vs SGD" lesson said "L-BFGS gets trapped in shallow minima." That was true for **ModifiedRelu** but NOT for **ReLU**. The optimizer-activation interaction matters more than either alone.

**Action:** For LoRA extraction, always use alpha=10000 (≈ ReLU) + L-BFGS. For full model, use SGD (which works with any alpha). Add `--relu_alpha` to all LoRA experiment commands.

**Relevant files:**
- [run_activation_ablation_wexac.sh](run_activation_ablation_wexac.sh) — WEXAC batch script for the ablation
- WEXAC job 669885 — results on A10 GPU

---

## [RESULT] Free-Coefficient LoRA Extraction: Partial Success (2026-02-23)

**Context:** Ran LoRA rank sweep with the winning config (alpha=10000/ReLU + L-BFGS for x + separate SGD for c). Tested coeff_lr tuning and epoch count.

**Results (all: alpha=10000, L-BFGS, T=1, seed=42):**

| Rank | Oracle SSIM | Free-c SSIM (lr=1e-2, 5Kep) | Coeff Error | Gap |
|------|-------------|------------------------------|-------------|-----|
| 4    | 0.615       | 0.509                        | 0.192       | 0.11 |
| 8    | 0.692       | **0.617**                    | 0.177       | 0.08 |
| 16   | 0.769       | 0.422                        | 0.282       | 0.35 |
| 32   | 0.697       | 0.415                        | 0.310       | 0.28 |
| 64   | 0.714       | **0.635**                    | **0.019**   | 0.08 |

**coeff_lr ablation (r=8):** lr=1e-3→0.457, lr=1e-2→**0.617**, lr=1e-1→0.536 (overshoots)

**Key lessons:**
1. **Free-c works well at r=8 and r=64** — within 0.08 SSIM of oracle. The coefficient optimization converges when the LoRA subspace captures enough of the gradient information.
2. **r=16 and r=32 are stubbornly bad** — coeff_error stays ~0.28-0.31 despite more epochs and higher lr. The optimization landscape has local minima at these ranks that trap the coefficient SGD.
3. **coeff_lr=0.01 is the sweet spot** — 10x default. Too low (1e-3) = underfitting, too high (0.1) = overshooting.
4. **Separate SGD for c is correct** (vs joint L-BFGS) — confirmed by the improvement from job 674631→681126.
5. **The residual gap** for well-converging ranks (r=8, r=64) is small enough that the free-c attack is viable — an attacker doesn't need oracle access.

**Next steps:** Try Adam for c (adaptive lr), random restarts, or higher consistency_weight for stubborn r=16/32.

---

## [INSIGHT] Always Save Visual Examples from Every Experiment Run (2026-02-22)

**Context:** After running the T-sweep on WEXAC, we had a CSV of SSIM numbers but no saved reconstruction images. To generate a PDF of visual examples, the entire extraction had to be re-run locally for each (T, rank) configuration — wasting hours of compute that was already done.

**Lesson:** Every experiment run should automatically save representative reconstruction images (both good and bad examples) alongside the numeric metrics. Numbers alone don't tell the full story — a "SSIM=0.48" could look like random noise or like a blurry-but-recognizable digit. Visual inspection is essential for understanding what's working and what's failing.

**Action:**
- Every sweep or single-config run should save a `.pth` file containing the actual image tensors (`x_train`, `x_recon_full`, `x_recon_lora`, `x_ctrl`, `ds_mean`) — not just scalar metrics.
- For sweeps, save a per-config results dict (e.g., `results/experiment_b_sweep_<timestamp>/T{T}_r{rank}.pth`) so any configuration can be visualized later without re-running.
- Also generate a quick PNG/PDF grid of examples (best and worst by SSIM) as part of the sweep output, so you never have to re-derive figures from scratch.

---

## [INSIGHT] SGD Required for Fine-Tuning, Not for Extraction (2026-02-23)

**Context:** Clarifying the role of the optimizer in the fine-tuning vs. extraction phases. The theoretical framework (implicit bias of GD on BCE → KKT/max-margin convergence) constrains the fine-tuning optimizer, but NOT the extraction optimizer.

**Key distinction:**
1. **Fine-tuning MUST use SGD.** The implicit bias of gradient descent on BCE loss → convergence to the max-margin solution → KKT stationarity conditions → weights encode support vectors. This is the theoretical foundation of the entire reconstruction attack. Adam, RMSProp, or any adaptive optimizer breaks this implicit bias guarantee.
2. **Extraction can use ANY optimizer.** The extraction phase solves an inverse problem: given ΔW, find x such that the NTK loss is minimized. This is just optimization — use whatever converges best. Adam is a strong candidate because its adaptive per-parameter learning rate handles the mixed-scale landscape well (pixel values, coefficients, and regularizers all have different scales).

**Why this matters:** We initially used SGD for extraction because the theoretical framework seemed to require it everywhere. But the theory only constrains the *forward* process (training). The *inverse* process (reconstruction) is an engineering problem where we're free to use the best tool. This opens up Adam, L-BFGS, or even learned optimizers for extraction.

**Relevant files:**
- [experiments/ntk_extraction.py](experiments/ntk_extraction.py) — extraction optimizer (L-BFGS, SGD, or Adam)
- [experiments/train_lora.py](experiments/train_lora.py) — fine-tuning optimizer (must be SGD)
- [experiments/run_experiment_b.py](experiments/run_experiment_b.py) — `--optimizer` and `--coeff_optimizer` flags

---

## [INSIGHT] Few-Shot Fine-Tuning Is the Attack Sweet Spot (2026-02-23)

**Context:** Connecting the NTK reconstruction results to real-world few-shot fine-tuning of large online models (LoRA adapters published on HuggingFace, CivitAI, etc.).

**Key insight:** Few-shot fine-tuning (N=5-50 samples, T=1-100 gradient steps) is the regime where LoRA reconstruction attacks are most potent:

1. **Overdetermined system**: A LoRA adapter for ViT-B/16 has ~300K-1M parameters per adapted layer. Fine-tuning on N=5 images of 224×224×3 ≈ 150K pixels each gives ~1M constraints for ~750K unknowns. The adapter contains enough information to reconstruct the data.

2. **Few gradient steps**: Users typically fine-tune for 1-10 epochs. With N=5 and 5 epochs, that's ~25 gradient steps. Phase 2 results show reconstruction holds through T=100 with LeakyReLU — real few-shot LoRA fine-tuning lives comfortably in this regime.

3. **All samples are support vectors**: With N << parameters, every training sample sits on the decision boundary. This is exactly the condition Haim et al.'s theory requires.

4. **Realistic threat model**: θ₀ is public (foundation model), BA is published (adapter on HuggingFace/CivitAI), and N is small. The attacker has everything needed.

**Concrete threat scenarios:**
- Face LoRA (CivitAI): Stable Diffusion + 5-20 selfies → adapter shared publicly
- Medical LoRA: ViT/BiomedCLIP + patient scans → shared with collaborators
- Legal/financial: LLaMA + confidential docs → internal model registry

**What our results say:**
- T=1: SSIM ≈ 1.0 (full) / 0.83 (LoRA) — even one gradient step leaks data
- T=100: SSIM ≈ 0.78-0.80 with LeakyReLU — realistic training is still vulnerable
- Free-coefficient extraction works — attacker doesn't need oracle access
- 11% of seeds attackable — not every run is vulnerable, but attacker can't predict which

**Gap this thesis fills:**
- Haim et al.: needs 1M+ epoch convergence (unrealistic for modern fine-tuning)
- Gradient inversion: needs actual gradient (not available from published adapter)
- This thesis: reconstructs from adapter weights via NTK/Gradient Bridge — the few-shot regime is where the attack is most potent and the threat model is most realistic

**Caveat:** All results so far are MNIST + 2-layer MLP. Scaling to ViTs on real images (Sprint 3) is the key open question.

---

## [BUG] Phase 0 Gradient Inversion: Three Critical Implementation Errors (2026-04-07)

**Context:** Phase 0 (ViT-B/16 gradient inversion gate experiment) returned SSIM=0.015 — total failure. Cosine similarity stuck at 0.04 throughout 3000 iterations. Root cause analysis found THREE bugs, the worst of which made the entire optimization a no-op.

**Bug 1 (ROOT CAUSE): Non-differentiable cosine similarity.** The code used `loss.backward(retain_graph=True)` to populate `param.grad`, then computed cosine similarity from these `.grad` tensors. But `.grad` attributes are **detached leaf tensors** — they have no computation graph connecting them back to `x_recon`. The subsequent `total_loss.backward()` produced zero gradients for `x_recon` from the cosine similarity term. The optimizer was only minimizing TV regularization (making a smooth random image). **Fix:** Use `torch.autograd.grad(loss, params, create_graph=True)` to get predicted gradients that remain in the computation graph.

**Bug 2: Per-tensor cosine similarity averaging.** Computed cosine similarity per parameter tensor (24 tensors), then averaged. Small tensors with random alignment got equal weight as large tensors. Geiping et al. compute ONE global cosine similarity on the entire flattened gradient vector. **Fix:** `torch.cat` all gradient tensors, compute single cosine similarity.

**Bug 3: LoRA-only gradients.** `capture_gradient()` iterated `model.named_parameters()` but peft freezes base model params → only 294K LoRA parameters had gradients. The inversion was trying to reconstruct 150K pixels from 294K low-rank gradient values. **Fix:** Temporarily enable `requires_grad_(True)` on all params to capture the full 86M-parameter gradient.

**Bug 4: SDPA double-backward not supported.** After fixing bugs 1-3, `create_graph=True` triggered `RuntimeError: derivative for aten::_scaled_dot_product_efficient_attention_backward is not implemented`. PyTorch 2.x's efficient/flash attention kernels don't support double-backward. **Fix:** Wrap the inversion loop in `torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)` to force the math-only SDPA backend.

**Bug 5: requires_grad mismatch.** After `capture_gradient` restores `requires_grad=False` on base model params, the inversion's `torch.autograd.grad(loss, params)` fails because those params don't require grad. **Fix:** Re-enable `requires_grad_(True)` on all matched params at the start of `invert_gradient`, restore after.

**Lesson:** When implementing gradient inversion, **always verify the gradient flows end-to-end** from target to optimized variable. A quick test: `total_loss.backward(); print(x_recon.grad.norm())` — if it's zero or doesn't exist, the optimization is broken. Also: never average cosine similarities across parameters — always flatten first. And when using `create_graph=True` with transformers, disable efficient/flash attention backends.

---

## [RESULT] Track A (KKT + N-Sweep) Definitively Closed (2026-04-07)

**Context:** Sprint 2c Track A tested whether using the correct N (up to N=502 total samples) would fix Sprint 1's Experiment A failure. Ran 15/48 configs before 48h timeout.

**Results:** KKT loss stuck at 330-350 for ALL N values tested (N=1 through N=100 per class). No trend — the loss didn't decrease as N approached the true support vector count.

**Why this was expected:** The composed model W = W₀ + BA satisfies KKT with respect to all ~502 samples (500 pre-training + 2 fine-tuning). The KKT loss of ~330 is essentially ||W₀||² — the unexplained pre-training residual. Even with N=502, the extraction would need to simultaneously reconstruct 500 pre-training images alongside the 2 fine-tuning targets — a fundamentally different (and much harder) problem than reconstructing 2 images from a model trained on 2 images.

---

## [INSIGHT] N>1 Reconstructions Are Superpositions — Decomposition Strategies (2026-04-07)

**Context:** When reconstructing N=2 images from NTK extraction, each reconstructed image visually looks like a ghostly superposition (blend) of BOTH training images. The NTK loss is a linear combination in gradient space: ΔW = -η Σᵢ cᵢ J(xᵢ), and nothing prevents the optimizer from distributing information across image slots. SSIM is ~0.5-0.6 when it should be higher — the information is there, just mixed.

**Root cause:** The NTK loss `‖ΔW + η Σ cᵢ ∇f(θ₀; xᵢ)‖²` has a **permutation and mixing symmetry** — any linear recombination of the per-sample contributions that sums to the same total gradient gives the same loss. The optimizer finds a blended local minimum rather than the clean separation.

**Key insight — linearity enables analytical separation:** Because the NTK regime linearizes the model, the weight gradient matrix for each FC layer has a special structure: each ROW is a different linear mixture of the N source images, with mixing coefficients from the loss gradients. With layer width 1000, this gives 1000 independent observations of the N-way mixture. This is exactly the setup for ICA (Independent Component Analysis).

**Approaches for general N (prioritized):**

1. **Cross-gradient orthogonality penalty (N=2-10):** Add `cos_sim(∇f(θ₀; x₁), ∇f(θ₀; x₂))` to the loss. Forces images to produce orthogonal gradients, directly attacking the superposition mechanism. Most theoretically principled for small N.

2. **Label-based grouping (any N, binary classification):** Coefficients cᵢ have opposite signs for the two classes (cᵢ>0 for class 0, cᵢ<0 for class 1). Separate the positive and negative gradient contributions first, then decompose within each class. Halves the effective problem size for free.

3. **ICA on weight gradient matrix — "Cocktail Party Attack" (N=10-1000):** Each row of the FC layer gradient is a linear mixture of the N source images. Apply FastICA with n_components=N to the weight gradient matrix. Scales to N ≤ layer width. Reference: Cocktail Party Attack (Kariyappa et al., ICML 2023).

4. **Sequential peeling with joint refinement (N=5-20):** Reconstruct images one at a time from the residual (matching pursuit), then jointly optimize all N using the greedy solutions as warm start. Each sub-problem is N=1 where the pipeline is strong.

5. **Overcomplete slots + clustering (any N):** Optimize for N'=2N image slots, then cluster similar results by SSIM. Redundancy helps coverage; extra slots absorb garbage solutions.

6. **Post-hoc NMF/ICA (N=2, quick experiment):** Apply `sklearn.decomposition.NMF(n_components=2)` to the two blended reconstructions. NMF is ideal for MNIST (non-negative, sparse pixels). Zero-code-change experiment.

**Phase transition:** Theoretical limit is N < network width (1000 for our MLP). SPEAR (NeurIPS 2024) achieves exact recovery up to N=25 on FC+ReLU networks using SVD + activation sparsity. The Cocktail Party Attack scales to N=1024 with ICA on the FC gradient matrix. Practical optimization-based limit is N~50-100.

**Critical existing code:** `get_diversity_penalty()` in `ntk_extraction.py` (lines 439-461) is already implemented but NOT wired into the extraction loop. Connecting it with a tunable weight is the lowest-hanging fruit.

**Key references:**
- Cocktail Party Attack (Kariyappa et al., ICML 2023) — ICA on FC gradient rows, scales to N=1024
- SPEAR (NeurIPS 2024) — exact batch recovery via SVD + ReLU sparsity filtering, N≤25
- ARES (2025) — sparse recovery in DCT basis, N≤384
- GradInversion (Yin et al., CVPR 2021) — group consistency + label recovery, N≤48
- Gradient Inversion on PEFT (Sami et al., CVPR 2025) — PEFT dimensionality reduction *focuses* gradient info, making inversion easier; N≤128 on CIFAR-100
- ReCIT (2025) — reconstruct private data from PEFT gradients
- Deep Adversarial Decomposition (Zou et al., CVPR 2020) — learned superimposed image separation
- Cold Diffusion for Superimposed Image Decomposition (IEEE 2025)
- MAGIA (2025) — alternating subset gradient matching for federated learning

**Action:**
1. Quick win: wire `get_diversity_penalty` + cosine repulsion into extraction loop for N≥2
2. Quick experiment: post-hoc NMF on existing N=2 blended results
3. Medium-term: implement ICA on weight gradient matrix (Cocktail Party style)
4. For thesis: characterize the N vs. SSIM curve to find the practical phase transition

**This negative result is thesis-valuable:** It definitively closes the compose-and-reconstruct pathway and strengthens the argument for the Gradient Bridge / NTK approach, which works by targeting ΔW (canceling the pre-training component) rather than the composed W.

---

## [RESULT] Multi-Seed Validation: Free-c Beats Oracle, Seed=42 Was Outlier (2026-04-07)

**Context:** Ran 50-seed free-c vs oracle comparison (SGD+LeakyReLU, T=1, LoRA r=8) and 30-seed LeakyReLU validation across T and rank.

**Key findings:**
1. **Seed=42 was an outlier.** SSIM=0.830 on seed=42 vs 50-seed mean=0.558±0.034. Seed=42 happens to produce fine-tuning samples where the model is confidently wrong after centering, giving large coefficient magnitude. Most seeds produce moderate signal.
2. **Free-c beats oracle (46/50 seeds).** Mean SSIM: free-c 0.557 vs oracle 0.408. The consistency penalty |c − (σ(f(θ₀;x))−y)/N|² acts as implicit regularization: it prevents the sign-flip local minima that plague oracle mode (where fixed coefficients can mislead the pixel optimizer). Free-c can adjust c jointly with x, finding better overall solutions.
3. **LeakyReLU validated across seeds.** 30 seeds × {T=1, T=10} × {r=8, r=32}: SSIM 0.558±0.034 (T=1), 0.572±0.088 (T=10). Control: 0.394-0.426. Consistent gap (0.13-0.15) proves real leakage.
4. **r=16/32 fixed.** SGD+LeakyReLU gives r=16 SSIM 0.624 (was 0.422), r=32 SSIM 0.680 (was 0.415). The fix was switching from L-BFGS+ReLU to SGD+LeakyReLU.

**Action:** Use 50-seed statistics as canonical numbers in the thesis, not seed=42. The attack works but is moderate (SSIM ~0.55-0.58), not dramatic (0.83). Frame as: "reconstruction quality sufficient to identify sensitive content but not pixel-perfect" — which is actually more realistic for a privacy threat analysis.

