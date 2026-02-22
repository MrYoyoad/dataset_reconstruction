# Project Status

Last updated: **2026-02-22**

---

## What's In Progress

### Sprint 1: LoRA Reconstruction on MNIST FCN (Started 2026-02-22)

**Goal**: Produce preliminary results showing LoRA-trained weights leak training data. Email supervisor with figures.

**Experiment A — Convergence + LoRA → Compose → Existing Pipeline**:
- Train FCN (784-1000-1000-1) with LoRA to convergence on 2 MNIST samples
- Compose W = W₀ + BA into standard weights, feed into unchanged reconstruction pipeline
- Rank sweep: r ∈ {1,2,4,8,16,32,64}
- Compare to full fine-tuning baseline

**Experiment B — Multi-Step NTK Reconstruction**:
- Take T gradient steps from random init (T ∈ {1,2,5,10,20,50,100,500,1000})
- Reconstruct using modified KKT loss with known coefficients, features frozen at θ₀
- Track NTK diagnostics at each T to determine *why* reconstruction degrades
- Rank sweep at each step count

**Deliverables**: 6 figures, LaTeX write-up, supervisor email with results.

**Status**: Infrastructure code complete (all 12 modules + 5 test files). Not yet run on real data — next step is end-to-end validation on WEXAC.

---

## What's Been Done

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
- Set up 10 custom skills: `/review`, `/supervisor`, `/experiment`, `/debug`, `/figure`, `/paper`, `/write`, `/lesson`, `/status`, `/project-manager`

**Theoretical analysis documents (in `notes/`):**
- `R2F_Guide.tex/.pdf` — detailed walkthrough of the Gradient Decoder mechanism from R2F
- `Inversion_Feasibility_Analysis.tex/.pdf` — information-theoretic analysis of when reconstruction is possible
- `Thesis_Direction_Analysis.tex/.pdf` — comparison of all three thesis directions with risk assessment

**Sprint 1 experiment code (all new, in `experiments/`):**
- `lora_wrapper.py` — LoRALinear class, apply_lora, compose_state_dict
- `data_utils.py` — few-shot MNIST loading, control images
- `train_lora.py` — LoRA + full fine-tuning training loops
- `ntk_steps.py`, `ntk_extraction.py`, `ntk_verification.py` — NTK regime tools
- `run_experiment_a.py` — convergence + compose experiment orchestrator
- `run_experiment_b.py` — multi-step NTK experiment orchestrator
- `run_sweep.py`, `metrics.py`, `plotting.py`, `configs.py` — sweep driver and utilities
- 5 test files in `experiments/tests/`
- Preview figure: `figures/experiment_b_grid.png`

### Early Analysis Figures

Four plots in `figures/`:
- `parameters_as_function_of_epoch.png` — parameter dynamics over training
- `parameters_as_function_of_epoch_full_fine_tune_comparison.png` — LoRA vs full fine-tune comparison
- `parameters_as_function_of_epoch_with_sweet_spot.png` — optimal reconstruction window
- `experiment_b_grid.png` — NTK experiment preview grid

---

## Current Folder Structure

```
Thesis/
├── .gitignore                 ← git ignore rules
├── CLAUDE.md                  ← project instructions for Claude Code
├── STATUS.md                  ← this file
├── LESSONS_LEARNED.md         ← running log of insights and pitfalls
├── papers/                    ← reference PDFs (18 files)
├── figures/                   ← graphs, plots, visualizations (4 files)
├── results/                   ← experimental outputs and metrics (empty — awaiting Sprint 1)
├── notes/                     ← planning docs + theoretical analyses
│   ├── GRADIENT_BRIDGE_PLAN.md    ← phased coding roadmap
│   ├── R2F_Guide.tex/.pdf         ← R2F Gradient Decoder walkthrough
│   ├── Inversion_Feasibility_Analysis.tex/.pdf  ← info-theoretic feasibility
│   └── Thesis_Direction_Analysis.tex/.pdf       ← direction comparison
├── experiments/               ← LoRA reconstruction experiment code (Sprint 1)
│   ├── configs.py             ← constants, sweep grids, paths
│   ├── lora_wrapper.py        ← LoRALinear class, apply_lora, compose_state_dict
│   ├── data_utils.py          ← few-shot MNIST loading, control images
│   ├── train_lora.py          ← LoRA + full fine-tuning training loops
│   ├── ntk_steps.py           ← multi-step gradient, coefficient computation
│   ├── ntk_extraction.py      ← modified KKT loss for NTK regime
│   ├── ntk_verification.py    ← NTK approximation diagnostics
│   ├── run_experiment_a.py    ← convergence experiment orchestrator
│   ├── run_experiment_b.py    ← NTK experiment orchestrator
│   ├── run_sweep.py           ← rank/step sweep driver
│   ├── metrics.py             ← wrapper around existing evaluations.py
│   ├── plotting.py            ← figure generation
│   └── tests/                 ← pytest test suite (5 files)
└── dataset_reconstruction/    ← original Haim et al. codebase (separate git repo)
```

---

## Next Steps (After Sprint 1)

### ViT Scaling (Sprint 2)
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
- **Experiment code untested on real data** — all Sprint 1 modules are written but need end-to-end validation on WEXAC

---

## Pending Tasks

- [ ] **Run Experiment A end-to-end** on WEXAC (highest priority — validates the core premise)
- [ ] **Run Experiment B** after A succeeds (NTK multi-step analysis)
- [ ] Commit `wexac_connect.sh`/`wexac_disconnect.sh` changes in `dataset_reconstruction/`
- [ ] Set up W&B project for new experiments (deferred — using CSV for Sprint 1)
- [ ] Read R2F paper Section 3 in detail (decoder architecture) — needed for Sprint 2
- [ ] Read Inverting Gradients (Geiping et al.) for attack loop — needed for Sprint 2
