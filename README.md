# Reconstructing Training Data from LoRA Adapters

**MSc Thesis, Weizmann Institute of Science**
Advisor: [Gal Vardi](https://scholar.google.co.il/citations?user=LVk3xE4AAAAJ&hl=en)

> Extending Haim et al. (NeurIPS 2022) — *"Reconstructing Training Data from Trained Neural Networks"* — to the era of Foundation Models and Parameter-Efficient Fine-Tuning (PEFT).

**Status:** Active research (Sprint 2 complete, Phase 0 in progress)

---

## Abstract

LoRA adapters are widely published on platforms like HuggingFace and CivitAI, yet they encode information about their private training data. This thesis investigates whether training images can be reconstructed from LoRA adapter weights alone, extending the KKT-based reconstruction framework of Haim et al. to the few-shot PEFT regime.

We develop an NTK-based reconstruction attack that targets the weight change ΔW (not the composed model), eliminating the pre-training confound. Using free-coefficient extraction with a consistency penalty, the attack requires no oracle knowledge. On MNIST with a 2-layer MLP, LoRA rank-8 adapters leak recognizable images (SSIM 0.557 +/- 0.034, 50-seed average) with a significant gap over control images (0.394-0.426), proving instance-specific leakage. The attack scales stably through T=100 gradient steps using LeakyReLU activations.

Current work focuses on scaling to ViT-B/16 via the Gradient Bridge: a learned decoder that maps low-rank adapter updates back to full-model gradients for gradient inversion.

---

## Research Questions

- **Can training data be reconstructed from published LoRA adapters?** Yes, on MNIST (SSIM 0.557, honest attack).
- **How does reconstruction quality depend on LoRA rank and training steps?** Higher rank helps (r=32: 0.680); quality stable through T=100.
- **Can a Gradient Decoder bridge LoRA to full gradients for ViT-scale inversion?** In progress (Phase 0 gate experiment).

---

## Quick Start

```bash
# Clone and set up environment
git clone https://github.com/MrYoyoad/dataset_reconstruction.git yoado
cd yoado
conda env create -f dataset_reconstruction/environment_macos.yaml
conda activate rec

# Run the primary NTK reconstruction (Experiment B)
python -m experiments.run_experiment_b \
  --rank 8 --n_steps 1 --free_coefficients \
  --consistency_weight 1.0 --optimizer sgd

# Results saved to results/ as CSV + .pth tensors
```

For GPU experiments on WEXAC, see [scripts/README.md](scripts/README.md).

---

## Methodology

Three reconstruction approaches, targeting different regimes:

| Approach | Method | Status | Key Result |
|----------|--------|--------|------------|
| **KKT on composed W** | Haim et al. on W=W_0+BA | Closed | Structural failure: W_0 confound |
| **NTK on weight change** | Reconstruct from ΔW | Working | SSIM 0.557 (honest, 50-seed) |
| **Gradient Bridge** | LoRA -> Decoder -> Inversion | In progress | Phase 0 gate experiment |

**Pipeline:** Data -> LoRA fine-tuning -> weight change ΔW -> NTK reconstruction -> SSIM evaluation

See [docs/architecture.md](docs/architecture.md) for code architecture and [docs/pipeline.md](docs/pipeline.md) for an end-to-end code walkthrough.

---

## Key Results

| Metric | Value | Context |
|--------|-------|---------|
| Full model SSIM (T=1, honest) | 0.997 | Near-perfect reconstruction |
| LoRA r=8 SSIM (50-seed avg) | 0.557 +/- 0.034 | Recognizable digits |
| LoRA r=32 SSIM (best config) | 0.680 | Moderate reconstruction |
| Control SSIM (same digit) | 0.394-0.426 | Gap proves real leakage |
| Multi-step T=100 (LeakyReLU) | 0.78-0.80 | Stable through typical fine-tuning |
| Free-c vs oracle wins | 46/50 seeds | No oracle access needed |

**Negative result (thesis-valuable):** KKT approach on composed model W=W_0+BA fails structurally. The pre-trained W_0 satisfies KKT over ~502 samples, making 2-image reconstruction impossible. This motivates the ΔW/NTK approach.

---

## Directory Structure

```
yoado/
├── README.md                      <- This file
├── STATUS.md                      <- Sprint-by-sprint progress and pending tasks
├── LESSONS_LEARNED.md             <- Running log of insights and pitfalls
├── STYLE_GUIDE.md                 <- Formatting rules for docs, slides, LaTeX, plots
├── GET_UP_TO_SPEED.md             <- 30-minute onboarding guide
├── DOCUMENTATION_GUIDE.md         <- How to maintain documentation
│
├── docs/                          <- Detailed guides and walkthroughs
│   ├── architecture.md            <- Code architecture and design decisions
│   ├── pipeline.md                <- End-to-end code walkthrough
│   └── experiment_guide.md        <- How to run and interpret experiments
│
├── experiments/                   <- Thesis experiment code (18 Python files)
│   ├── run_experiment_b.py        <- Primary NTK reconstruction entry point
│   ├── ntk_extraction.py          <- Core reconstruction algorithm
│   ├── phase0_vit_inversion.py    <- ViT gradient inversion (Phase 0)
│   ├── configs.py                 <- Constants, paths, device detection
│   └── tests/                     <- pytest test suite (6 test files)
│
├── scripts/                       <- WEXAC GPU job submission scripts (28 files)
├── results/                       <- Experiment outputs: CSV metrics + .pth tensors
├── figures/                       <- Plots and visualizations
├── notes/                         <- Theoretical analyses (LaTeX/PDF)
├── papers/                        <- Reference PDFs
│
└── dataset_reconstruction/        <- Original Haim et al. codebase (separate git)
    ├── Main.py                    <- Base code entry point
    ├── extraction.py              <- KKT loss optimization
    └── CreateModel.py             <- MLP with ModifiedReLU
```

See per-directory READMEs: [experiments/](experiments/README.md) | [scripts/](scripts/README.md) | [results/](results/README.md) | [figures/](figures/README.md)

---

## Installation

### Local Development (Mac)

```bash
cd dataset_reconstruction
conda env create -f environment_macos.yaml   # Apple Silicon (MPS backend)
conda activate rec
```

Key dependencies: Python 3.8, PyTorch 2.2.2, TorchVision 0.17.2, Kornia 0.7.0, wandb.

### WEXAC GPU Cluster

The `rec` conda env on WEXAC runs PyTorch 2.4.1+cu121, timm 0.9.12, peft 0.7.1. All serious experiments must run on WEXAC (NVIDIA L40S, CUDA 12.6).

```bash
# Get an interactive GPU shell
cd dataset_reconstruction && ./wexac_connect.sh shell

# Or submit a batch job
bsub < scripts/run_exp_b_gpu.sh
```

See [docs/experiment_guide.md](docs/experiment_guide.md) for full WEXAC instructions.

---

## Reproducing Results

### Step 1: Train base model (or use pre-trained)

Pre-trained weights are in `models/`. To retrain:

```bash
cd dataset_reconstruction
python Main.py --run_mode=train --problem=mnist_odd_even --proj_name=mnist_odd_even \
  --data_per_class_train=250 --model_hidden_list=[1000,1000] \
  --train_epochs=1000000 --train_lr=0.01
```

### Step 2: Run NTK reconstruction (Experiment B)

```bash
python -m experiments.run_experiment_b \
  --rank 8 --n_steps 1 --free_coefficients \
  --consistency_weight 1.0 --optimizer sgd --activation leakyrelu
```

### Step 3: Run a sweep

```bash
python -m experiments.run_sweep --sweep_type rank --ranks 4 8 16 32 64
```

### Step 4: Generate figures

```bash
python -m experiments.plotting --input results/<your_csv>.csv
```

For full experiment taxonomy and parameter reference, see [docs/experiment_guide.md](docs/experiment_guide.md).

---

## Documentation

| Document | Purpose |
|----------|---------|
| [docs/architecture.md](docs/architecture.md) | Code architecture, module dependencies, design decisions |
| [docs/pipeline.md](docs/pipeline.md) | End-to-end code walkthrough (Experiment B trace) |
| [docs/experiment_guide.md](docs/experiment_guide.md) | How to run experiments, interpret results, troubleshoot |
| [STATUS.md](STATUS.md) | Current progress, sprint breakdown, pending tasks |
| [LESSONS_LEARNED.md](LESSONS_LEARNED.md) | Running log of insights and pitfalls |
| [DOCUMENTATION_GUIDE.md](DOCUMENTATION_GUIDE.md) | How to maintain and update documentation |

---

## Citation

This thesis builds on:

```bib
@inproceedings{haim2022reconstructing,
  author = {Haim, Niv and Vardi, Gal and Yehudai, Gilad and Shamir, Ohad and Irani, Michal},
  booktitle = {Advances in Neural Information Processing Systems},
  title = {Reconstructing Training Data From Trained Neural Networks},
  volume = {35},
  pages = {22911--22924},
  year = {2022}
}
```

---

## License

Research use only. Based on the [Haim et al. implementation](https://github.com/nivha/dataset_reconstruction).
