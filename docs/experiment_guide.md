# Experiment Guide

How to run experiments, interpret results, and troubleshoot.

---

## Experiment Taxonomy

| Experiment | File | Status | Purpose |
|------------|------|--------|---------|
| **Experiment A (KKT)** | `run_experiment_a.py` | Closed | KKT on composed W=W_0+BA. Structural failure. |
| **Experiment B (NTK)** | `run_experiment_b.py` | Primary | NTK on weight change ΔW. Working attack. |
| **Phase 0 (ViT)** | `phase0_vit_inversion.py` | In progress | Gradient inversion gate on ViT-B/16 |
| **Sweeps** | `run_sweep.py`, `run_sprint2b_sweep.py`, `run_sprint2c_sweep.py` | Complete | Hyperparameter ablation |
| **Diagnostics** | `run_diagnostic.py` | Complete | NTK regime verification |
| **Figures** | `gen_t_sweep_examples.py`, `plotting.py` | Complete | Publication figures |

---

## Running Experiment B (Primary)

### Basic commands

```bash
# Oracle mode (upper bound, uses true coefficients)
python -m experiments.run_experiment_b --n_steps 1 --rank 8

# Free-coefficient mode (realistic attack, no oracle)
python -m experiments.run_experiment_b --n_steps 1 --rank 8 \
  --free_coefficients --consistency_weight 1.0 --optimizer sgd

# Full model (no LoRA, ceiling test)
python -m experiments.run_experiment_b --n_steps 1 --free_coefficients

# Multi-step
python -m experiments.run_experiment_b --n_steps 10 --rank 8 \
  --free_coefficients --activation leaky_relu

# Save results to file
python -m experiments.run_experiment_b --rank 8 --free_coefficients \
  --save_results --device cuda
```

### Full CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_steps` | 1 | Fine-tuning gradient steps T |
| `--rank` | None | LoRA rank (None = full model) |
| `--n_per_class` | 1 | Images per class for fine-tuning |
| `--seed` | 42 | Random seed |
| `--lr` | 0.01 | Fine-tuning learning rate |
| `--extraction_epochs` | 50000 | Reconstruction iterations |
| `--optimizer` | lbfgs | Image optimizer: `lbfgs` or `sgd` |
| `--device` | auto | `cuda`, `mps`, or `cpu` |
| `--free_coefficients` | False | Enable free-coefficient mode |
| `--coeff_lr` | 0.001 | Coefficient optimizer learning rate |
| `--box_weight` | 5.0 | Weight for c box constraint |
| `--consistency_weight` | 0.0 | Self-consistency penalty weight (use 1.0) |
| `--coeff_optimizer` | sgd | Coefficient optimizer: `sgd` or `adam` |
| `--coeff_init` | sign_aware | Initialization: `zeros`, `sign_aware`, `uniform` |
| `--sign_weight` | 5.0 | Sign enforcement penalty weight |
| `--min_coeff` | 0.05 | Minimum coefficient magnitude |
| `--loss_type` | l2 | NTK loss: `l2` or `cosine` |
| `--n_sweep` | False | Try multiple N values |
| `--relu_alpha` | 149.87 | Alpha for ModifiedRelu |
| `--verify_weight` | 1.0 | Weight for box constraint loss |
| `--activation` | modified_relu | Extraction activation: `relu`, `leaky_relu`, `modified_relu` |
| `--lr_schedule` | constant | LR schedule for fine-tuning |
| `--finetune_optimizer` | sgd | Fine-tuning optimizer: `sgd`, `adamw` |
| `--save_results` | False | Save CSV + .pth output |

### Recommended Configurations

**Best single-step (T=1, LoRA r=8):**
```bash
python -m experiments.run_experiment_b --rank 8 --n_steps 1 \
  --free_coefficients --consistency_weight 1.0 \
  --optimizer sgd --activation leaky_relu --save_results
```

**Multi-step (T=10):**
```bash
python -m experiments.run_experiment_b --rank 8 --n_steps 10 \
  --free_coefficients --consistency_weight 1.0 \
  --optimizer sgd --activation leaky_relu --save_results
```

**Higher rank (r=32):**
```bash
python -m experiments.run_experiment_b --rank 32 --n_steps 1 \
  --free_coefficients --consistency_weight 1.0 \
  --optimizer sgd --activation leaky_relu --save_results
```

---

## Running Sweeps

### Sprint 2b Sweep (activation, LR, restarts)

```bash
python -m experiments.run_sprint2b_sweep --phase 0  # activation ablation
python -m experiments.run_sprint2b_sweep --phase 1  # multi-step ablation
python -m experiments.run_sprint2b_sweep --phase 2  # random restarts
python -m experiments.run_sprint2b_sweep --phase 5  # LR magnitude
```

### Sprint 2c Sweep (NTK ablations)

```bash
python -m experiments.run_sprint2c_sweep --track b2  # loss ratio (verify_weight)
python -m experiments.run_sprint2c_sweep --track b3a # optimizer x activation
python -m experiments.run_sprint2c_sweep --track b4  # N sweep (NTK)
```

### Generic Sweep

```bash
python -m experiments.run_sweep --sweep_type rank --ranks 4 8 16 32 64
python -m experiments.run_sweep --sweep_type steps --steps 1 2 5 10 20
```

---

## WEXAC Submission

### Submitting a job

```bash
# SSH to WEXAC (requires Weizmann VPN)
ssh wexac
cd /home/projects/galvardi/yoado

# Submit batch job
bsub < scripts/run_exp_b_gpu.sh

# Or for specific experiments
bsub < scripts/run_phase0_fixed_wexac.sh
```

### Monitoring

```bash
bjobs                    # List active jobs
bjobs -l <job_id>        # Detailed job info
bpeek <job_id>           # Tail stdout of running job
bkill <job_id>           # Kill a job
```

### LSF Directives (in script headers)

| Directive | Example | Purpose |
|-----------|---------|---------|
| `#BSUB -q` | `long-gpu` | Queue: `interactive-gpu` (4h), `short-gpu` (24h), `long-gpu` (168h) |
| `#BSUB -R` | `rusage[mem=16384]` | Memory in MB |
| `#BSUB -gpu` | `num=1` | Number of GPUs |
| `#BSUB -W` | `24:00` | Wall time limit |
| `#BSUB -o` | `wexac_logs/...%J.out` | stdout log (%J = job ID) |
| `#BSUB -e` | `wexac_logs/...%J.err` | stderr log |
| `#BSUB -J` | `phase0_fix` | Job name |

### Syncing code before submission

Always sync latest code from Mac before running experiments:

```bash
# From Mac:
rsync -avz --exclude='__pycache__' experiments/ wexac:~/experiments/
rsync -avz scripts/ wexac:~/scripts/
```

---

## Interpreting Results

### SSIM Scale (MNIST context)

| SSIM Range | Interpretation |
|------------|----------------|
| > 0.9 | Near-perfect reconstruction (full model achieves 0.997) |
| 0.6-0.9 | Clearly recognizable, some distortion |
| 0.5-0.6 | Recognizable digit shape, significant noise |
| 0.4-0.5 | Class-level similarity only (control baseline) |
| < 0.4 | Failure |

### Control images

Results include SSIM for "control" images — same-digit MNIST images not used in fine-tuning. The gap between attack SSIM (0.557) and control SSIM (0.394-0.426) proves the reconstruction captures instance-specific information, not just class features.

### CSV columns

| Column | Meaning |
|--------|---------|
| `rank` | LoRA rank (0 = full model) |
| `T` / `n_steps` | Fine-tuning gradient steps |
| `n_per_class` | Images per class |
| `ssim_recon` | SSIM between reconstruction and ground truth |
| `ssim_ctrl` | SSIM between reconstruction and control image |
| `ssim_gap` | ssim_recon - ssim_ctrl (positive = real leakage) |
| `coeff_error` | L2 distance between free and oracle coefficients |
| `ntk_loss_final` | Final NTK reconstruction loss |
| `verify_loss_final` | Final box constraint loss |

### Loading .pth tensor files

```python
import torch
data = torch.load('results/exp_b_T1_r8_free_s42_a10000.0.pth', weights_only=False)
x_train = data['x_train']      # [N, 1, 28, 28] ground truth
x_recon = data['x_recon']      # [N, 1, 28, 28] reconstruction
x_ctrl = data['x_ctrl']        # [N, 1, 28, 28] control images
ds_mean = data['ds_mean']       # [1, 1, 28, 28] dataset mean
coefficients = data.get('coefficients')  # [N] free coefficients
```

---

## Troubleshooting

### NaN during extraction

**Cause:** ReLU activation with T >= 50 steps causes dead neurons to accumulate, producing NaN gradients.
**Fix:** Use `--activation leaky_relu`. LeakyReLU is stable through T=100.

### CUDA out of memory

**Cause:** L-BFGS stores history of past gradients (default 20 steps).
**Fix:** Switch to `--optimizer sgd` or reduce `--extraction_epochs`.

### Path errors / ModuleNotFoundError

**Cause:** The code uses `sys.path.insert` to import from `dataset_reconstruction/`.
**Fix:** Always run from the repo root: `python -m experiments.run_experiment_b` (not `cd experiments && python run_experiment_b.py`).

### Phase 0 crashes

Known bugs (all fixed in current code):
1. Non-differentiable cosine similarity — use `torch.autograd.grad(create_graph=True)`
2. Per-tensor cosine averaging — use global flattened cosine similarity
3. LoRA-only gradients — enable `requires_grad_(True)` on all params
4. SDPA double-backward — force math-only backend
5. `float.sqrt()` — use `math.sqrt()`

### Low SSIM but loss converges

**Cause:** Optimizer found a local minimum (coefficient sign flip, wrong N).
**Fix:** Try `--consistency_weight 1.0` (prevents sign flips), or use `--n_sweep` to auto-detect best N.
