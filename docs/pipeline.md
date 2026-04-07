# End-to-End Code Walkthrough

This document traces a single Experiment B run from command line to results file, explaining each step.

---

## The NTK Reconstruction Pipeline

We trace: `python -m experiments.run_experiment_b --rank 8 --n_steps 1 --free_coefficients --consistency_weight 1.0 --optimizer sgd`

### Step 1: Entry Point

**File:** `experiments/run_experiment_b.py::main()`

Parses CLI arguments via argparse, then calls `run_single_config()` with all parameters. Key arguments:

| Argument | Default | Purpose |
|----------|---------|---------|
| `--rank` | None | LoRA rank (None = full model) |
| `--n_steps` | 1 | Fine-tuning gradient steps T |
| `--free_coefficients` | False | Enable realistic attack (no oracle) |
| `--consistency_weight` | 0.0 | Weight for self-consistency penalty on c |
| `--optimizer` | lbfgs | Image optimizer (sgd or lbfgs) |
| `--activation` | modified_relu | Extraction activation (relu, leaky_relu, modified_relu) |
| `--seed` | 42 | Random seed for data selection |

### Step 2: Model Creation

**File:** `run_experiment_b.py::create_model()` -> `dataset_reconstruction/CreateModel.py::NeuralNetwork`

```python
model = NeuralNetwork(
    input_dim=784,           # 28x28 MNIST
    hidden_dim_list=[1000, 1000],
    output_dim=1,            # binary classification
    activation=LeakyReLU(0.01),
    use_bias=False,
)
```

Then loads pre-trained weights from `models/weights-mnist_odd_even_d250_mnist_odd_even.pth`:

```python
model.load_state_dict(checkpoint['state_dict'])
```

This gives us theta_0 — the pre-trained model before fine-tuning.

### Step 3: Data Loading

**File:** `experiments/data_utils.py::get_finetuning_data()`

- Loads MNIST with binary labeling (even=0, odd=1) via `LABELS_DICT`
- Selects `n_per_class` images per class (default 1, so 2 total)
- Subtracts dataset mean (`ds_mean`) for centering
- Returns `x_ft` (fine-tuning images), `y_ft` (labels), `ds_mean`

Also loads control images via `get_control_images_in_distribution()` — same-class MNIST digits not used in fine-tuning, for baseline comparison.

### Step 4: Weight Update Computation

**File:** `experiments/ntk_steps.py`

Two paths depending on `--rank`:

**Full model** (`rank=None`): `compute_multi_step_update()`
- Saves theta_0 state dict
- Runs T SGD steps on BCE loss
- Computes delta_w = {name: param_T - param_0 for each parameter}

**LoRA** (`rank=8`): `compute_multi_step_update_lora()`
- Wraps model with `apply_lora(model, rank=8)` from `lora_wrapper.py`
- Freezes base weights, only trains A and B matrices
- Runs T SGD steps
- Computes delta_w from composed weight change (BA product)
- Also returns `lora_B0` dict for subspace projection

### Step 5: NTK Verification

**File:** `experiments/ntk_verification.py::ntk_smoke_test()`

Checks two conditions for the NTK approximation to hold:

1. **Weight change ratio:** `||theta_T - theta_0|| / ||theta_0|| < 0.01`
   If the weights moved too far, the linearization is invalid.

2. **Feature cosine similarity:** `cos(grad_f(theta_0; x), grad_f(theta_T; x)) > 0.99`
   The feature map (Jacobian) must be nearly constant between theta_0 and theta_T.

Prints warnings if either threshold is violated but doesn't stop execution.

### Step 6: Reconstruction (Core Algorithm)

**File:** `experiments/ntk_extraction.py::run_ntk_extraction()`

This is the core optimization loop:

```
Initialize:
  x_recon = Gaussian noise * EXTRACTION_INIT_SCALE  (requires_grad=True)
  c = sign_aware_init(y)                             (requires_grad=True if free_coefficients)

For epoch in range(EXTRACTION_EPOCHS):
    # NTK loss: ||ΔW + η·T·Σ c_i ∇f(θ₀;x_i)||²
    ntk_loss = get_ntk_loss(model_theta0, delta_w, x, c, lr, T, lora_B0)

    # Box constraint: x ∈ [-1, 1]
    verify_loss = get_ntk_verify_loss(x)

    # Coefficient penalties (if free_coefficients):
    coeff_loss = get_coeff_penalty(c, model_theta0, x, y,
                                    consistency_weight=1.0)

    total_loss = ntk_loss + verify_weight * verify_loss + coeff_loss

    # Update x (SGD or L-BFGS)
    optimizer_x.step()

    # Update c (SGD, separate optimizer)
    optimizer_c.step()
```

#### Key function: `get_ntk_loss()`

The NTK loss computes the mismatch between the observed weight change and the predicted weight change:

```python
# Forward pass: f(θ₀; x) weighted by coefficients
output = (model(x).squeeze() * coefficients).sum()

# Compute gradients w.r.t. model parameters (create_graph=True for 2nd-order)
grad = torch.autograd.grad(output, params, create_graph=True)

# Predicted weight change per layer
predicted = -lr * n_steps * grad

# If LoRA: project both target and predicted onto col(B₀)
if lora_B0:
    target = project_to_lora_subspace(target, B0)
    predicted = project_to_lora_subspace(predicted, B0)

# L2 loss between target and predicted
loss = sum((target - predicted).pow(2).sum() for each layer)
```

#### Key function: `get_coeff_penalty()`

Three penalty terms on the free coefficients:

1. **Box:** `(|c| - 1).relu().pow(2)` — keeps c in [-1, 1]
2. **Sign:** enforces c > 0.05 for y=0, c < -0.05 for y=1
3. **Consistency:** `|c - (sigma(f(theta_0; x)) - y) / N|^2` — ties c to its NTK-predicted value. This is the key regularizer that makes free-coefficient extraction work (weight=1.0).

### Step 7: Evaluation

**File:** `experiments/metrics.py::compute_all_metrics()`

Computes pairwise metrics between reconstructed and true images:

- **SSIM** (Structural Similarity): primary metric, via Kornia. Range [0, 1].
- **DSSIM** (Structural Dissimilarity): 1 - SSIM
- **NCC** (Normalized Cross-Correlation): correlation between pixel vectors
- **L2**: Euclidean distance between pixel vectors

Also computes the same metrics for control images to establish baseline.

### Step 8: Output

Results saved to `results/` as:

- **CSV:** one row per configuration with columns: rank, T, n_per_class, ssim_recon, ssim_ctrl, ssim_gap, coeff_error, loss_final, etc.
- **.pth tensor file:** contains `x_train`, `x_recon`, `x_ctrl`, `ds_mean`, `coefficients` for later visualization

Naming: `exp_b_T{T}_r{rank}_free_s{seed}_a{activation}.pth`

---

## The ViT Pipeline (Phase 0)

**File:** `experiments/phase0_vit_inversion.py`

A separate pipeline for gradient inversion on ViT-B/16, using timm + peft instead of the base code.

```
1. load_vit_with_lora()        <- timm ViT-B/16 + peft LoRA (rank r)
2. get_sample_images()         <- CIFAR-10 images resized to 224x224
3. capture_gradient()          <- forward+backward, collect grad dict
4. invert_gradient()           <- cosine similarity + TV loss optimization
   - 8 random restarts, 10K iterations each
   - torch.autograd.grad(create_graph=True) for differentiable cosine sim
   - SDPA math backend (flash/efficient don't support double-backward)
5. Evaluate SSIM, save results
```

Two modes:
- `--mode full`: capture all 86M parameter gradients (ceiling test)
- `--mode lora`: capture only 294K LoRA parameter gradients (realistic)
- `--mode both`: run both sequentially
