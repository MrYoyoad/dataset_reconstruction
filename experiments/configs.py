"""Default hyperparameters and sweep grids for LoRA reconstruction experiments."""

import os

# ---------------------------------------------------------------------------
# Paths (relative to Thesis/ root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_RECON_DIR = os.path.join(PROJECT_ROOT, 'dataset_reconstruction')
DATASETS_DIR = os.path.join(DATASET_RECON_DIR, 'data')
MODELS_DIR = os.path.join(DATASET_RECON_DIR, 'models')
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
# NTK verification thresholds
# ---------------------------------------------------------------------------
NTK_WEIGHT_CHANGE_THRESHOLD = 0.01  # ||Δθ||/||θ₀|| < this
NTK_FEATURE_COS_THRESHOLD = 0.99    # cos(∇f(θ₀;x), ∇f(θ_T;x)) > this

# ---------------------------------------------------------------------------
# MNIST label mapping (odd/even binary, matching mnist_odd_even.py)
# ---------------------------------------------------------------------------
LABELS_DICT = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1, 8: 0, 9: 1}
