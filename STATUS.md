# Project Status

Last updated: **2026-04-07**

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

### Sprint 2c: KKT & NTK Reconstruction Ablations — MOSTLY DONE

Comprehensive ablation study across two tracks. 148 configs completed, 2 tracks remaining + Phase 0.

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

### Phase 0: ViT Gradient Inversion Gate — RESUBMITTING (bugs fixed)

Critical gate experiment: can gradient inversion reconstruct images from exact ViT-B/16 gradients?

**First attempt failed (SSIM=0.015) due to 3 bugs:**
1. **Non-differentiable cosine similarity** (ROOT CAUSE): `loss.backward()` + `param.grad` produces detached tensors — cosine sim had no gradient w.r.t. x_recon. Optimizer only minimized TV loss → smooth random image. Fixed with `torch.autograd.grad(create_graph=True)`.
2. **Per-tensor cosine similarity averaging**: Averaged cosine sim across 24 tensors instead of one global flattened cosine sim (Geiping et al.). Fixed by concatenating all gradients.
3. **LoRA-only gradients**: Only captured 294K LoRA params instead of all 86M. Fixed with `full_model_grad` option.
4. **Phase 0b crash**: `float.sqrt()` on line 228. Fixed with `math.sqrt()`.

**Improvements added:** 8 random restarts, 10K iterations (was 3K), two-phase experiment (full 86M + LoRA-only 294K).

- Code: `experiments/phase0_vit_inversion.py` (fixed)
- WEXAC script: `scripts/run_phase0_fixed_wexac.sh`

### Sprint 2 Multi-Seed Validation — COMPLETE (2026-03-27)

50-seed free-c vs oracle comparison and 30-seed LeakyReLU validation completed overnight.

**Key findings:**
- **Seed=42 was an outlier**: SSIM=0.830 vs 50-seed mean=0.558±0.034. Report 50-seed stats as canonical.
- **Free-c beats oracle**: Mean SSIM 0.557 (free-c) vs 0.408 (oracle) across 50 seeds. Free-c wins 46/50. The consistency penalty provides implicit regularization that prevents sign-flip local minima.
- **LeakyReLU validated**: 30 seeds × {T=1, T=10} × {r=8, r=32}. Mean SSIM 0.558 (T=1), 0.572 (T=10). Control: 0.394-0.426.
- **r=16/32 improved**: SGD+LeakyReLU gives r=16 SSIM 0.624 (was 0.422), r=32 SSIM 0.680 (was 0.415).

### Sprint 2 Track 2: LoRA Free-Coefficient Extraction — IN PROGRESS

**Current best results (ReLU, L-BFGS, separate SGD for c, 5000ep):**

| Rank | Oracle SSIM | Free-c SSIM | Coeff Error | Gap | Status |
|------|-------------|-------------|-------------|-----|--------|
| 4    | 0.615       | 0.509       | 0.192       | 0.11 | OK |
| 8    | 0.692       | 0.617       | 0.177       | 0.08 | Good |
| 16   | 0.769       | 0.422       | 0.282       | 0.35 | Needs work |
| 32   | 0.697       | 0.415       | 0.310       | 0.28 | Needs work |
| 64   | 0.714       | 0.635       | 0.019       | 0.08 | Great |

### Sprint 2b: Multi-Step NTK Sweep — COMPLETE

Phases 0-2 completed (WEXAC jobs 669864, 674627). Phases 3-4 completed as Sprint 2c Track B1.

**Phase 0 (activation ablation):** LeakyReLU is dramatically more stable than ReLU at high T. Full model SSIM stays ~0.77-0.80 through T=100 vs ReLU collapsing/NaN'ing at T>=50.

**Phase 1 (SGD + free-c baseline, ReLU):** LoRA results terrible with ReLU — NaN everywhere at T>=50. Confirms activation choice is critical.

**Phase 2 (random restarts, LeakyReLU):** LoRA r=8 and r=32 nearly match full model (gap only 0.01-0.03) through T=100. Random restarts show low variance (~0.01).

**Phase 3 (LR scaling)** and **Phase 4 (warm-start):** Completed as Sprint 2c Track B1.

### Few-Shot Threat Model Analysis — In Progress

Documenting the few-shot threat model: when LoRA fine-tuning uses very few samples, the number of adapter parameters far exceeds the number of unknowns, making the system highly overdetermined and reconstruction theoretically feasible.

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
- **Figures**: `figures/multi_seed_analysis.png`

- **Figures**: `figures/experiment_b_grid.png`, `figures/experiment_b_grid_r32.png`, `figures/rank_sweep_sprint1.png`, `figures/sprint1_summary.png`

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

## Current Folder Structure (as of 2026-03-19)

```
/home/projects/galvardi/yoado/     ← WEXAC home = top-level git repo
├── .gitignore
├── CLAUDE.md
├── STATUS.md                      ← this file
├── LESSONS_LEARNED.md
├── STYLE_GUIDE.md
├── papers/                        ← reference PDFs (1 present, 3 need sync from Mac)
│   ├── THE_PAPER.pdf
│   └── README.md                  ← lists what to sync
├── figures/                       ← 9 files (all regenerated)
├── results/                       ← 87 files (.csv metrics + .pth tensors from sweeps)
├── notes/
│   └── reconstruction_approaches.tex  ← catalog of approaches (March 2026)
├── scripts/                       ← WEXAC job submission scripts
│   └── wexac_logs/                ← job stdout/stderr logs
├── experiments/                   ← LoRA reconstruction experiment code (24 .py files)
│   └── tests/                     ← pytest test suite (7 files)
└── dataset_reconstruction/        ← original Haim et al. codebase (separate .git)
```

---

## Next Steps (After Sprint 1)

### Sprint 2a: Free-Coefficient NTK Extraction — IMPLEMENTED (2026-02-22)
Fixed the oracle-coefficient cheating. Implementation:
- `ntk_extraction.py`: `get_coeff_penalty()`, free-c mode in `run_ntk_extraction()`, `run_ntk_extraction_n_sweep()`
- `run_experiment_b.py`: `--free_coefficients`, `--consistency_weight`, `--n_sweep`, `--optimizer` flags
- `configs.py`: `COEFF_LR=1e-3`, `COEFF_BOX_WEIGHT=5.0`, `COEFF_CONSISTENCY_WEIGHT=0.0`

**α ablation results (seed=42, T=1, full model):**

| α | Optimizer | SSIM | Coeff Error | Notes |
|---|-----------|------|-------------|-------|
| 0 | L-BFGS | 0.282 | 1.066 | Signs flipped — non-unique |
| 1 | L-BFGS | 0.777 | 0.005 | Near-oracle c |
| 10 | L-BFGS | 0.638 | 0.005 | Over-penalized |
| **1** | **SGD** | **0.997** | **0.0004** | **Matches oracle** |
| oracle | L-BFGS | 0.817 | 0 | Upper bound (cheating) |

**Recommended config:** `--free_coefficients --consistency_weight 1.0 --optimizer sgd`

### Sprint 2b: Multi-Step Sweep
- WEXAC job submitted: T ∈ {1,2,5,10,20,50,100,500,1000} × rank ∈ {full,1,4,8,32,64} = 54 configs
- Free-coefficient extraction is especially important here: for T > 1, oracle coefficients use `coefficients_at_init` which is only exact for T=1

### ViT Scaling (Sprint 3)
After establishing rank threshold and NTK step-count analysis on FCN:
1. ViT-B/16 (pretrained from `timm`) with HuggingFace PEFT LoRA
2. Fine-tune on 5-10 CelebA face images (binary classification)
3. Phase 0: capture true gradient during LoRA fine-tuning, feed into Inverting Gradients
4. Phase 1: train gradient decoder (R2F-style) on proxy data
5. Phase 2: end-to-end attack on victim LoRA adapter

### Gradient Bridge (Phase 1-2)
- Generate ~50k (BA, ∇_W L) pairs from proxy data
- Train per-layer MLP decoder: low-rank LoRA → full-rank gradient (>0.9 cosine similarity)
- End-to-end: frozen decoder → inversion engine → reconstructed images

### Diffusion Priors (Direction 3)
- Hybrid KKT + SDS loss for low-rank reconstruction
- Target: face reconstruction from SD/ViT LoRA adapters

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
- [ ] **Phase 0 (fixed)**: Resubmit with bug fixes (differentiable cosine sim, full-model gradient, restarts)
- [ ] **Phase 0b**: Noise tolerance sweep (blocked on Phase 0 fix)

### Research Backlog
- [ ] **Design better image-domain prior loss** — current NTK extraction only uses pixel box constraint (x ∈ [-1,1]). Ideas: Total Variation (TV), LPIPS perceptual loss, SDS from frozen diffusion model, manifold constraints (VAE latent space), frequency-domain priors. Low priority for MNIST, critical for ViT/larger images. Connects to Direction 3 (Diffusion Priors).

### Writing & Communication
- [ ] **Write LaTeX summary** — `notes/lora_reconstruction_writeup.tex`
- [ ] **Email supervisor** with results and figures
- [ ] **Verify figure quality** — publication-ready (axes, legends, DPI, colorblind-safe)

### Reading (Sprint 3 prep)
- [ ] Read R2F paper Section 3 in detail (decoder architecture)
- [ ] Read Inverting Gradients (Geiping et al.) for attack loop
