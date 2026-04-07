# Code Architecture

This document describes the structure, dependencies, and design decisions of the experiment code.

---

## System Overview

The codebase has two layers:

```
dataset_reconstruction/          <- Base: Haim et al. (NeurIPS 2022)
  ├── CreateModel.py             <- NeuralNetwork, ModifiedRelu
  ├── extraction.py              <- KKT loss (calc_extraction_loss)
  ├── evaluations.py             <- SSIM, NCC, L2 metrics
  └── common_utils/              <- Dataset loaders, image processing

experiments/                     <- Thesis: LoRA reconstruction extensions
  ├── Core algorithm
  ├── Experiment runners
  ├── Infrastructure
  └── tests/
```

The `experiments/` code imports from `dataset_reconstruction/` via `sys.path.insert(0, ...)` to reuse the base model, loss functions, and metrics without code duplication.

---

## Module Dependency Graph

```
configs.py                       <- Constants, paths, device detection
    ↑                               Imported by everything
    |
data_utils.py                    <- MNIST loading, control images
    ↑
    |
lora_wrapper.py                  <- LoRALinear, apply_lora, compose_state_dict
    ↑
    |
train_lora.py                    <- Training loops (LoRA + full fine-tuning)
    ↑
    |
ntk_steps.py                    <- Multi-step gradient computation
    ↑                               compute_multi_step_update[_lora]()
    |
ntk_extraction.py               <- Reconstruction loss + optimizer loop
    ↑                               get_ntk_loss(), run_ntk_extraction()
    |
ntk_verification.py             <- NTK regime diagnostics
    ↑                               ntk_smoke_test(), linearization_error()
    |
metrics.py                      <- SSIM, DSSIM, NCC, L2 wrappers
    ↑
    |
┌───┴───────────────────┐
run_experiment_a.py     run_experiment_b.py     <- Experiment entry points
run_sweep.py            run_sprint2b_sweep.py   <- Sweep orchestrators
run_diagnostic.py       run_sprint2c_sweep.py
                        gen_t_sweep_examples.py
                        phase0_vit_inversion.py <- ViT pipeline (standalone)
```

### Standalone modules

- `plotting.py` — figure generation, imports only configs.py and matplotlib
- `phase0_vit_inversion.py` — ViT gradient inversion, uses timm/peft instead of base code

---

## Key Design Decisions

### float64 precision
NTK loss computation requires high numerical precision. The weight change ΔW has small magnitude relative to W₀, so float32 accumulation errors corrupt the gradient. MPS doesn't support float64, so `configs.get_dtype()` returns float32 on Mac (for debugging only) and float64 on CUDA/CPU.

### Separate optimizers for x and c
Mirrors Haim et al.'s treatment of x (reconstructed images) and λ (Lagrange multipliers) as separate optimization variables. The coefficient optimizer (SGD, lr=1e-3) is much more conservative than the image optimizer (SGD/L-BFGS, lr=0.03) because coefficients are low-dimensional and sensitive to overshooting.

### LeakyReLU over ReLU/ModifiedReLU
ReLU produces NaN at T>=50 gradient steps due to dead neurons accumulating. ModifiedReLU (sigmoid-modulated) is too smooth for the NTK loss landscape. LeakyReLU (slope=0.01) is stable through T=100 and gives the best SSIM. This was the key finding of Sprint 2b activation ablation.

### LoRA subspace projection
`_project_to_lora_subspace()` in ntk_extraction.py projects the predicted weight update onto col(B₀). This constrains the reconstruction to produce weight changes that lie in the LoRA column space, matching the structure of actual LoRA updates. Without this projection, the optimizer can "cheat" by placing gradients outside the LoRA subspace.

### Device auto-detection
`configs.get_device()` checks CUDA > MPS > CPU. This lets the same code run on WEXAC (CUDA), Mac (MPS for debugging), and CI (CPU). Tests use CPU to avoid GPU dependencies.

---

## Data Flow

### Training path
```
MNIST (odd/even binary) 
  → get_finetuning_data() selects N images per class
  → Labels: even=0, odd=1 (LABELS_DICT in configs.py)
  → Mean subtraction (ds_mean computed from full dataset)
  → Fine-tune: SGD on pre-trained model for T steps
  → Output: delta_w dict {layer_name: W_T - W_0}
```

### Reconstruction path
```
delta_w + random x_init (Gaussian, scale=EXTRACTION_INIT_SCALE)
  → NTK loss: ||ΔW + η·T·Σ c_i ∇f(θ₀;x_i)||²
  → + box constraint: x ∈ [-1, 1]
  → + coefficient penalties: sign, box, consistency
  → Optimize x (SGD/L-BFGS) + c (SGD) for EXTRACTION_EPOCHS
  → Output: x_recon, coefficients, metrics (SSIM, DSSIM, NCC, L2)
```

### LoRA path (when --rank is set)
```
Pre-trained θ₀ + LoRA(A, B) of rank r
  → compute_multi_step_update_lora(): T SGD steps on LoRA params only
  → delta_w = composed BA product (or full delta if requested)
  → NTK loss with LoRA projection: predicted ΔW projected onto col(B₀)
  → Same optimization loop as above
```

---

## Relationship to Base Code

### Inherited from dataset_reconstruction/

| Module | What we use |
|--------|-------------|
| `CreateModel.py` | `NeuralNetwork` class (configurable MLP), `ModifiedRelu` activation |
| `extraction.py` | `calc_extraction_loss()` for KKT baseline (Experiment A only) |
| `evaluations.py` | `get_ssim_pairs_kornia()`, `ncc_dist()`, `l2_dist()` |
| `common_utils/` | `get_processed_dataset()` for MNIST/CIFAR loading |

### New in experiments/

| Module | What's new |
|--------|-----------|
| `lora_wrapper.py` | Full LoRA implementation: `LoRALinear`, `apply_lora()`, `compose_state_dict()` |
| `ntk_extraction.py` | NTK loss formulation, free-coefficient optimization, LoRA subspace projection |
| `ntk_steps.py` | Multi-step gradient computation for both full and LoRA fine-tuning |
| `ntk_verification.py` | NTK regime diagnostics (weight change ratio, feature cosine similarity) |
| `train_lora.py` | LoRA-specific training loops with gradient accumulation |
| `phase0_vit_inversion.py` | ViT-B/16 gradient inversion using timm + peft (independent of base code) |
