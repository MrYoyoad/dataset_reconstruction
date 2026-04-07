# results/

Experiment outputs: 107 files (CSV metrics + .pth image tensors).

---

## File Naming Conventions

### CSV files
`<experiment>_<date>_<time>.csv`

Examples:
- `sprint2b_phase0_20260222_191555.csv` — Sprint 2b Phase 0, run on 2026-02-22
- `multiseed_freec_vs_oracle_20260327_034128.csv` — 50-seed comparison

### .pth tensor files
`exp_b_T{T}_r{rank}_{mode}_s{seed}_a{alpha}.pth`

| Field | Meaning |
|-------|---------|
| `T` | Fine-tuning steps (1, 10, 100) |
| `r` | LoRA rank (4, 8, 16, 32, 64); omitted for full model |
| mode | `free` = free-coefficient; omitted = oracle |
| `s` | Random seed |
| `a` | ReLU alpha (149 = ModifiedReLU, 10000 = ReLU) |

Example: `exp_b_T1_r8_free_s42_a10000.pth` = T=1, rank 8, free-coefficient, seed 42, ReLU

---

## Result Categories

### Sprint 1: Proof of Concept
- `experiment_b_sweep_*.csv` — rank sweep results
- `exp_b_T1_r*_s42_*.pth` — oracle reconstruction tensors
- `exp_a_output.txt`, `exp_b_output.txt` — raw console output

### Sprint 2a: Free Coefficients
- `exp_b_T1_r*_free_s42_*.pth` — free-coefficient reconstruction tensors
- `exp_b_T1_full_free_s42_*.pth` — full model free-coefficient

### Sprint 2b: Multi-Step & Scaling
- `sprint2b_phase{0-7}_*.csv` — 8 ablation phases

### Sprint 2c: NTK Ablations
- `sprint2c_track_b{2-8}_*.csv` — NTK ablation tracks
- `sprint2c_track_a_*.csv` — KKT N-sweep (negative result)

### Diagnostics & Validation
- `diagnostic_*.csv` — NTK regime diagnostics
- `seed_fix_ablation_*.csv` — seed variance study
- `multiseed_freec_vs_oracle_*.csv` — 50-seed free-c vs oracle
- `multiseed_leakyrelu_validation_*.csv` — 30-seed LeakyReLU validation

### Multi-Seed Per-Image Results
- `exp_b_T1_r8_free_s{0-9}_a149.pth` — 10 seeds with saved tensors

### Phase 0
- `phase0_*.pth` — ViT gradient inversion results (when available)

### Composed Model
- `lora_r8_n1_s42_composed.pth` — composed model weights (Experiment A)

---

## Common CSV Columns

| Column | Description |
|--------|-------------|
| `rank` | LoRA rank (0 = full model) |
| `T` / `n_steps` | Fine-tuning gradient steps |
| `n_per_class` | Images per class |
| `activation` | Extraction activation function |
| `optimizer` | Image optimizer (sgd/lbfgs) |
| `ssim_recon` | SSIM: reconstruction vs ground truth |
| `ssim_ctrl` | SSIM: reconstruction vs control image |
| `ssim_gap` | ssim_recon - ssim_ctrl |
| `coeff_error` | L2 distance: free coefficients vs oracle |
| `ntk_loss_final` | Final NTK reconstruction loss |
| `verify_loss_final` | Final box constraint loss |
| `lr` | Fine-tuning learning rate |
| `seed` | Random seed |

---

## Loading .pth Files

```python
import torch
import matplotlib.pyplot as plt

data = torch.load('results/exp_b_T1_r8_free_s42_a10000.pth', weights_only=False)

# Available keys:
# x_train    — [N, 1, 28, 28] ground truth images
# x_recon    — [N, 1, 28, 28] reconstructed images
# x_ctrl     — [N, 1, 28, 28] control (same-class, different instance)
# ds_mean    — [1, 1, 28, 28] dataset mean
# coefficients — [N] free coefficients (if free mode)

# Visualize
fig, axes = plt.subplots(1, 3)
axes[0].imshow(data['x_train'][0, 0] + data['ds_mean'][0, 0], cmap='gray')
axes[0].set_title('Ground Truth')
axes[1].imshow(data['x_recon'][0, 0] + data['ds_mean'][0, 0], cmap='gray')
axes[1].set_title('Reconstruction')
axes[2].imshow(data['x_ctrl'][0, 0] + data['ds_mean'][0, 0], cmap='gray')
axes[2].set_title('Control')
plt.show()
```

---

## Notes

- Large `.pth` files (>10MB) are gitignored. CSVs are tracked.
- Always add `ds_mean` back to images before visualization (data is mean-subtracted).
- Use `weights_only=False` in `torch.load()` to avoid FutureWarnings.
