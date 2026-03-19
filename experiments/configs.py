"""Default hyperparameters and sweep grids for LoRA reconstruction experiments."""

import os
import torch

# ---------------------------------------------------------------------------
# Paths (relative to Thesis/ root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_RECON_DIR = os.path.join(PROJECT_ROOT, 'dataset_reconstruction')
DATASETS_DIR = os.path.join(DATASET_RECON_DIR, 'data')
MODELS_DIR = os.path.join(DATASET_RECON_DIR, 'models')
PRETRAINED_MNIST_PATH = os.path.join(MODELS_DIR, 'weights-mnist_odd_even_d250_mnist_odd_even.pth')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')

# ---------------------------------------------------------------------------
# Architecture (matching existing MNIST model: 784-1000-1000-1)
# ---------------------------------------------------------------------------
INPUT_DIM = 28 * 28  # 784
OUTPUT_DIM = 1
MODEL_HIDDEN_LIST = [1000, 1000]
MODEL_INIT_LIST = [0.0001, 0.0001]
MODEL_USE_BIAS = False  # only first layer gets default bias=True

# ---------------------------------------------------------------------------
# Training (matching Main.py)
# ---------------------------------------------------------------------------
TRAIN_LR = 0.01
TRAIN_EPOCHS = 1_000_000
TRAIN_THRESHOLD = 1e-40
TRAIN_EVAL_EVERY = 1000

# ---------------------------------------------------------------------------
# Extraction (from working MNIST reconstruction: kcf9bhbi sweep)
# ---------------------------------------------------------------------------
EXTRACTION_LR = 0.03052
EXTRACTION_LAMBDA_LR = 0.03052
EXTRACTION_INIT_SCALE = 0.03498
EXTRACTION_MIN_LAMBDA = 0.4471
EXTRACTION_RELU_ALPHA = 149.87
EXTRACTION_EPOCHS = 50_000
EXTRACTION_EVAL_EVERY = 1000

# ---------------------------------------------------------------------------
# Sweep grids
# ---------------------------------------------------------------------------
RANK_SWEEP = [1, 2, 4, 8, 16, 32, 64]
N_PER_CLASS_SWEEP = [1, 2, 4, 8]
STEP_SWEEP = [1, 2, 5, 10, 20, 50, 100, 500, 1000]
RANK_SWEEP_EXP_B = [1, 4, 8, 32, 64]  # reduced set for tractability

# ---------------------------------------------------------------------------
# Free-coefficient optimization (mirrors Haim et al.'s λ handling)
# See LESSONS_LEARNED.md: "NTK Coefficients Are Cheating"
# ---------------------------------------------------------------------------
COEFF_LR = 1e-3                    # lr for c optimizer (smaller than x's lr)
COEFF_BOX_WEIGHT = 5.0             # weight for c ∈ [-1, 1] box constraint
COEFF_CONSISTENCY_WEIGHT = 0.0     # weight for |c - c_predicted(x)|² (start at 0)
COEFF_INIT = 'sign_aware'          # 'zeros', 'sign_aware', or 'uniform'
COEFF_SIGN_WEIGHT = 5.0            # weight for sign enforcement (matches Haim's weight=5)
COEFF_MIN_MAGNITUDE = 0.05         # minimum |c| for sign constraint (ablate: 0.01-0.25)

# N sweep — attacker doesn't know true dataset size
N_PER_CLASS_EXTRACTION_SWEEP = [1, 2, 3, 4, 5]

# ---------------------------------------------------------------------------
# Activation choices for ablation (Sprint 2b)
# ---------------------------------------------------------------------------
ACTIVATION_CHOICES = ['relu', 'leaky_relu', 'modified_relu']

# LR schedule choices for ablation
LR_SCHEDULE_CHOICES = ['constant', 'inv_sqrt_T', 'inv_T', 'cosine', 'linear', 'cosine_warmup']

# Fine-tuning optimizer choices (Sprint 2c realism probe)
FINETUNE_OPTIMIZER_CHOICES = ['sgd', 'adamw']

# LR magnitude sweep (Sprint 2b Phase 5: bracket realistic settings)
LR_MAGNITUDE_SWEEP = [0.001, 0.005, 0.01, 0.05]

# ---------------------------------------------------------------------------
# NTK verification thresholds
# ---------------------------------------------------------------------------
NTK_WEIGHT_CHANGE_THRESHOLD = 0.01  # ||Δθ||/||θ₀|| < this
NTK_FEATURE_COS_THRESHOLD = 0.99    # cos(∇f(θ₀;x), ∇f(θ_T;x)) > this

# ---------------------------------------------------------------------------
# MNIST label mapping (odd/even binary, matching mnist_odd_even.py)
# ---------------------------------------------------------------------------
LABELS_DICT = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1, 8: 0, 9: 1}


# ---------------------------------------------------------------------------
# Device auto-detection (CUDA > MPS > CPU)
# ---------------------------------------------------------------------------
def get_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def get_dtype(device=None):
    """Return the appropriate dtype for the given device.

    MPS doesn't support float64, so we use float32 there.
    CUDA and CPU use float64 for numerical precision.
    """
    if device is None:
        device = get_device()
    if 'mps' in str(device):
        return torch.float32
    return torch.float64
