# experiments/

LoRA reconstruction experiment code — 18 Python files, ~5,900 lines total.

---

## File Index

### Infrastructure

| File | Lines | Purpose |
|------|-------|---------|
| `configs.py` | 115 | Constants, paths, sweep grids, device auto-detection |
| `data_utils.py` | 180 | MNIST loading, binary labeling, control images |
| `lora_wrapper.py` | 158 | LoRALinear module, apply_lora(), compose_state_dict() |
| `train_lora.py` | 97 | Training loops for LoRA and full fine-tuning |
| `metrics.py` | 92 | SSIM, DSSIM, NCC, L2 metric wrappers |
| `plotting.py` | 485 | Publication-quality figure generation |
| `__init__.py` | 0 | Package marker |

### Core Algorithm

| File | Lines | Purpose |
|------|-------|---------|
| `ntk_extraction.py` | 591 | NTK loss, free-coefficient optimization, LoRA subspace projection |
| `ntk_steps.py` | 274 | Multi-step gradient computation (full and LoRA) |
| `ntk_verification.py` | 385 | NTK regime diagnostics (weight change, feature stability) |

### Experiment Runners

| File | Lines | Purpose |
|------|-------|---------|
| `run_experiment_a.py` | 335 | Experiment A: KKT on composed model (CLOSED) |
| `run_experiment_b.py` | 676 | Experiment B: NTK reconstruction (PRIMARY) |
| `run_sweep.py` | 224 | Generic rank/step sweep |
| `run_sprint2b_sweep.py` | 573 | Sprint 2b: activation, LR, restart ablations |
| `run_sprint2c_sweep.py` | 731 | Sprint 2c: NTK ablation tracks (B1-B8) |
| `run_diagnostic.py` | 332 | NTK regime diagnostics runner |
| `gen_t_sweep_examples.py` | 298 | Generate visual examples for T sweep |

### ViT Pipeline

| File | Lines | Purpose |
|------|-------|---------|
| `phase0_vit_inversion.py` | 631 | ViT-B/16 gradient inversion gate (Phase 0) |

---

## Quick Run

```bash
# From repo root:
python -m experiments.run_experiment_b --rank 8 --n_steps 1 --free_coefficients
python -m experiments.run_sweep --sweep_type rank --ranks 4 8 16 32
python -m experiments.phase0_vit_inversion --device cuda --mode both
```

See [docs/experiment_guide.md](../docs/experiment_guide.md) for full CLI reference.

---

## Adding New Experiments

Follow this pattern:

1. Create `experiments/run_<name>.py` with argparse and a `main()` entry point
2. Import infrastructure from `configs.py`, `data_utils.py`, `metrics.py`
3. Create WEXAC script `scripts/run_<name>_wexac.sh` (see [scripts/README.md](../scripts/README.md))
4. After running, save results to `results/` and update [results/README.md](../results/README.md)
5. Add entry to this file's index table

---

## Dependencies on dataset_reconstruction/

The code imports from the base Haim et al. codebase via `sys.path.insert`:

- `CreateModel.NeuralNetwork` — MLP architecture
- `CreateModel.ModifiedRelu` — custom activation
- `extraction.calc_extraction_loss` — KKT loss (Experiment A only)
- `evaluations.*` — SSIM, NCC, L2 metrics
- `common_utils.get_processed_dataset` — dataset loading

---

## Tests

6 test files in `tests/`. Run with:

```bash
python -m pytest experiments/tests/ -v
```

See [tests/README.md](tests/README.md) for details.
