# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository is a **thesis project** extending the work of Haim et al. (NeurIPS 2022, Oral) — *"Reconstructing Training Data from Trained Neural Networks"* — into the era of Foundation Models and Parameter-Efficient Fine-Tuning (PEFT).

### Base Implementation (dataset_reconstruction/)

The `dataset_reconstruction/` subdirectory contains the original PyTorch implementation. It reconstructs training samples from a trained neural network's weights by optimizing inputs that satisfy the KKT optimality conditions of the implicit max-margin problem. All commands below should be run from that directory.

### Thesis Research Direction

The thesis focuses on the **image domain**, extending the Haim et al. reconstruction paradigm to Foundation Models and PEFT:

1. **LoRA Reconstruction via the "Gradient Bridge"**: LoRA adapters (A, B matrices) are structured, compressed recordings of cumulative training gradients. The LoRA update ΔW = BA is a low-rank projection of the full gradient that can be inverted. A learned **Gradient Decoder** (trained on proxy data) approximates the inverse projection, recovering full-dimensional gradients from low-rank adapters, which are then fed into gradient inversion pipelines (GradInversion for vision).

2. **LoRA in the NTK Regime**: When LoRA rank r is sufficiently large (r ≳ N, where N = number of data points), LoRA optimization in the NTK regime converges to the same global minimum as full fine-tuning (Jang et al., 2024). This means LoRA weights BA encode the same support vector geometry as full weights — the KKT stationarity condition adapts to a coupled system where B^T(Σ λ_i y_i ∇_W Φ(θ; x_i)) = 0 and the analogous condition for A.

3. **Generative Priors (Diffusion-Based)**: When LoRA rank is very low, a hybrid loss combines KKT residuals with Score Distillation Sampling (SDS) from a frozen diffusion model to hallucinate missing high-frequency details. Targets face reconstruction from Stable Diffusion / ViT LoRA adapters.

### Key Theoretical Foundations

- **Implicit bias of GD**: For homogeneous networks trained with BCE, gradient flow converges to KKT conditions of the max-margin problem: θ* ∝ Σ λ_i y_i ∇_θ Φ(θ*; x_i). Weights encode "support vectors" — the hardest training examples.
- **LoRA as gradient projection**: ∇_A L = B^T ∇_W L and ∇_B L = (∇_W L)A^T. The adapter weights are not arbitrary compression artifacts; they are high-fidelity measurements of the principal components of the full gradient.
- **Information density argument**: Even with low-rank adapters, the number of parameters (constraints) far exceeds the number of unknowns (training tokens/pixels) in the few-shot regime, making the system highly overdetermined.
- **R2F (Recover-to-Forget)**: Demonstrates that full-model gradients can be reconstructed from LoRA updates using a decoder trained on a proxy model — proof that the LoRA-to-gradient bridge is learnable. See dedicated section below.

### Recover-to-Forget (R2F) — Core Reference

**Paper:** "Recover-to-Forget: Gradient Reconstruction from LoRA for Efficient LLM Unlearning" (Liu et al., Dec 2025). This is the single most important external reference for the thesis. The paper's original purpose is machine unlearning; our thesis pivots the same mechanism to **attack** (reconstruct) private training data.

**Central Idea:** LoRA updates ΔW = BA are low-rank projections of the full gradient ∇_W L. R2F trains a small "Gradient Decoder" network f_φ that learns the inverse map: given a low-rank adapter update, output the approximate full-rank gradient. Once full gradients are recovered, they can be used downstream (for unlearning in their case, for gradient inversion in ours).

**Gradient Decoder Architecture (Section 3 of the paper):**
- Input: the flattened LoRA product BA (or equivalently the separate A, B matrices) for a given layer.
- Output: the predicted full-rank gradient ∇_W L for that same layer.
- Network: a small MLP (or U-Net for structured gradients). Trained per-layer — one decoder per weight matrix in the base model.
- Training data: generated from a **proxy dataset** (public data, not the private fine-tuning data). For each proxy batch, compute a single LoRA training step and record the pair (BA, ∇_W L). Repeat ~50k times to build the training set.
- Loss: cosine similarity between predicted and true gradient (the paper also explores MSE; cosine similarity preserves direction, which is what inversion algorithms care about most).

**Key Design Choices We Inherit:**
1. **Proxy data need not match the private distribution** — the decoder learns the geometric relationship between the low-rank subspace and the full gradient space, not the data semantics. CIFAR-100 can proxy for faces; WikiText can proxy for clinical notes.
2. **Single-step vs. multi-step updates** — the decoder is trained on single-step LoRA updates. For adapters trained over many steps, the paper accumulates or averages the decoded gradients. We may need to handle this for our attack scenario.
3. **Rank sensitivity** — decoder accuracy degrades as LoRA rank r decreases (less information retained). The paper reports strong results for r ≥ 4. Our thesis should ablate over r to characterize the privacy-utility tradeoff.
4. **Per-layer decoders** — each transformer layer gets its own decoder. The dimensionality of the gradient varies by layer (e.g., query/key/value projections in attention, up/down projections in MLP blocks).

**How We Diverge from R2F:**
- R2F uses decoded gradients for **gradient ascent** (unlearning). We feed them into **gradient inversion** algorithms (e.g., Inverting Gradients / GradInversion) to reconstruct the actual training inputs.
- R2F targets LLMs (text). Our thesis targets the **image domain** (ViT / ResNet LoRA adapters fine-tuned on images).
- We add **generative priors** (SDS from frozen diffusion models) to compensate for approximation noise in the decoded gradient — R2F does not need this because unlearning is more tolerant of noisy gradients than pixel-level reconstruction.

### Key Reference Papers (in papers/)

- `papers/THE_PAPER.pdf` — Haim et al. (NeurIPS 2022), the foundational reconstruction paper
- `papers/NEWER_2025_Paper.pdf` — Recent follow-up work
- `papers/Gradient Bridge_ PEFT Privacy Attack.pdf` — The Gradient Bridge attack formulation (LoRA → Gradient Decoder → Gradient Inversion)
- `papers/Thesis Ideas_ LoRA, NTK, Reconstruction.pdf` — Full thesis prospectus covering all three directions with theoretical analysis

## Environment Setup

```bash
cd dataset_reconstruction
conda env create -f environment_macos.yaml   # Apple Silicon (MPS backend)
conda activate rec
```

Key dependencies: Python 3.8, PyTorch 2.2.2, TorchVision 0.17.2, Kornia 0.7.0, wandb.

A `settings.py` must exist in `dataset_reconstruction/` defining `datasets_dir`, `results_base_dir`, and `models_dir` paths. One already exists with relative paths (`./data/`, `./runs/`, `./models/`).

## WEXAC GPU Access

Scripts in `dataset_reconstruction/` automate GPU allocation on the Weizmann WEXAC cluster (LSF scheduler). Requires Weizmann VPN and `wexac` configured in `~/.ssh/config`.

**Jupyter mode (default)** — launches JupyterLab on a GPU node with SSH tunnels:
```bash
cd dataset_reconstruction
./wexac_connect.sh            # or: ./wexac_connect.sh jupyter
```

**Shell mode** — interactive bash shell on a GPU node for running experiments directly:
```bash
./wexac_connect.sh shell
```
This allocates a GPU via `bsub -q interactive-gpu`, activates the `rec` conda env, and drops you into an SSH session on the GPU node. Run training/reconstruction commands directly from there.

**Disconnect** — kills GPU jobs, tunnels, and temp files:
```bash
./wexac_disconnect.sh
```

Configuration (top of `wexac_connect.sh`): GPU queue, memory, conda env path, Jupyter token, ports.

## Running the Code

Everything runs through `Main.py` with `--run_mode=train` or `--run_mode=reconstruct`.

**Train a model:**
```bash
python Main.py --run_mode=train --problem=cifar10_vehicles_animals --proj_name=cifar10_vehicles_animals \
  --data_per_class_train=250 --model_hidden_list=[1000,1000] --model_init_list=[0.0001,0.0001] \
  --train_epochs=1000000 --train_lr=0.01 --train_evaluate_rate=1000
```

**Reconstruct from a trained model:**
```bash
python Main.py --run_mode=reconstruct --problem=cifar10_vehicles_animals \
  --pretrained_model_path=weights-cifar10_vehicles_animals_d250_....pth \
  [extraction hyperparameters]
```

Pre-built command lines with tested hyperparameters are in `command_line_args/`. Use W&B sweeps for hyperparameter search on new problems.

**Analysis notebooks:** `reconstruction_cifar10.ipynb` and `reconstruction_mnist.ipynb` analyze pre-computed reconstructions.

## Architecture

### Pipeline Flow

```
Main.py (entry point)
  ├── GetParams.py        → argparse (~40 parameters)
  ├── CreateData.py        → dispatches to problem-specific loader
  ├── CreateModel.py       → builds MLP with ModifiedReLU activations
  ├── Train mode:          → SGD on BCEWithLogitsLoss, saves weights to .pth
  └── Reconstruct mode:
      ├── extraction.py    → KKT loss optimization (core algorithm)
      └── evaluations.py   → NCC distance, SSIM, nearest-neighbor matching
```

### Core Algorithm (extraction.py)

The reconstruction optimizes two sets of variables:
- **x** (reconstructed inputs) — initialized randomly, optimized with momentum SGD
- **λ** (Lagrange multipliers) — one per sample, optimized separately

The loss has two components:
1. **KKT loss**: `||∇L(x,λ) - w||²` — the trained weights should equal the gradient of the loss at reconstructed points
2. **Constraint/verification loss**: bounds enforcement (x ∈ [-1,1], λ ≥ 0.05)

### Model (CreateModel.py)

`NeuralNetwork` is a configurable MLP (e.g., D-1000-1000-1) using `ModifiedRelu` — a custom activation with sigmoid-modulated gradients for smoother optimization during extraction.

### Adding New Problems

Add a Python file under `problems/` that provides data loading logic and model parameters. Existing examples: `cifar10_vehicles_animals.py` (binary CIFAR-10), `mnist_odd_even.py` (binary MNIST), `simple_2d.py` (toy 2D).

### Key Directories

- `models/` — pre-trained weight files (.pth)
- `reconstructions/` — pre-computed reconstruction outputs
- `common_utils/` — shared utilities (dataset loading, SSIM via Kornia, image processing)
- `data/` — downloaded datasets (MNIST, CIFAR-10)
- `runs/` — training/extraction output (created at runtime)

## Thesis Directory Structure

```
Thesis/                            ← top-level git repo
├── .gitignore
├── CLAUDE.md                      ← this file
├── STATUS.md                      ← project status, what's done/pending, known issues
├── LESSONS_LEARNED.md             ← running log of insights and pitfalls
├── papers/                        ← all reference PDFs
├── figures/                       ← graphs, plots, visualizations
├── results/                       ← experimental outputs and metrics
├── notes/                         ← planning docs, theoretical analyses (.tex/.pdf)
├── experiments/                   ← new experiment code (LoRA bridge, NTK, SDS, etc.)
│   └── tests/                     ← pytest test suite
└── dataset_reconstruction/        ← original Haim et al. codebase (separate git repo, excluded from this repo)
```

## Key Documents

- [STATUS.md](STATUS.md) — current project status: what's done, what's not started, known issues, pending tasks
- [LESSONS_LEARNED.md](LESSONS_LEARNED.md) — running log of insights, pitfalls, and things to remember
- [notes/GRADIENT_BRIDGE_PLAN.md](notes/GRADIENT_BRIDGE_PLAN.md) — phased coding roadmap (Phase 0 → 1 → 2) with reading list and timeline

## Code Hygiene Rules

1. **Keep the project clean.** No orphaned files, no dead code left behind, no clutter in the repo root. Every file should have a clear purpose and live in the right directory.
2. **Temporary files go in `/tmp/`.** Any throwaway scripts, scratch notebooks, one-off checks, or debugging artifacts must be written to `/tmp/` (not the project tree) so they never appear in git.
3. **Don't repeat yourself.** If a code snippet is used more than once, extract it into a function — in the same file if it's local, or into a shared utility file (e.g., under `common_utils/` or `experiments/utils/`) if it's cross-cutting.

## Git & Documentation Rules

**Before every commit/push, update project documentation:**

1. **STATUS.md** — Update with what was done: new features, bug fixes, experiments added, scripts changed. Keep the "What's Done" and "What's Pending" sections current.
2. **LESSONS_LEARNED.md** — Log any new insights, pitfalls discovered, or design decisions made during the work.
3. **CLAUDE.md** — If new scripts, tools, workflows, or configuration were added, document them in the relevant section here.

This ensures the project state is always accurately reflected in docs, not just in git history.

## Git

The entire `Thesis/` directory is a single git repo. The git root was moved up from `dataset_reconstruction/` on 2026-02-22 so that thesis-level files (experiments, notes, figures, etc.) are all tracked.

### Remotes

- `myfork`: `https://github.com/MrYoyoad/dataset_reconstruction.git` (personal fork — primary push target)
- `origin`: `git@github.com:ai-hub-weizmann/dataset_reconstruction.git` (Weizmann fork)
- `upstream`: `https://github.com/nivha/dataset_reconstruction.git` (original Haim et al. repo)
