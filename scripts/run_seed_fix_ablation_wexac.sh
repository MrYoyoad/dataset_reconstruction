#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o seed_fix_ablation_%J.out
#BSUB -e seed_fix_ablation_%J.err
#BSUB -J seed_fix_ablation

# Seed Fix Ablation: diagnose and fix multi-seed NTK extraction failure
#
# Problem: Only seed 42/15 works (SSIM=1.0); other 14 seeds give ~0.52.
# Root cause: coefficients initialized at zero (saddle point), never move.
#
# Ablation A (5 configs): which fix matters most?
#   baseline, sign_init_only, sign_constraint_only, init+constraint, init+constraint+cosine
# Ablation B (3 configs): min_coeff threshold sweep
#   min_coeff ∈ {0.01, 0.10, 0.25} (0.05 covered by ablation A)
#
# Total: 8 configs × 15 seeds × 2 models (full + LoRA r=8) = 240 runs @ 50K epochs

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== Starting Seed Fix Ablation ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

python -u -m experiments.run_diagnostic \
    --task seed_fix_ablation \
    --device cuda

echo "=== Seed Fix Ablation Complete ==="
echo "Date: $(date)"
