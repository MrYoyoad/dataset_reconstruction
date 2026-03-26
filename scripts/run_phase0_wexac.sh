#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o wexac_logs/phase0_%J.out
#BSUB -e wexac_logs/phase0_%J.err
#BSUB -J phase0

# Phase 0: ViT gradient inversion gate experiment
# Uses the rec conda env (PyTorch 2.4.1 + timm + peft)

set -e

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
python -c "import timm; print(f'timm: {timm.__version__}')"

cd /home/projects/galvardi/yoado
mkdir -p results figures wexac_logs

echo "=== Phase 0: Exact Gradient Inversion on ViT-B/16 ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

# Phase 0: exact gradient inversion (no noise)
python -u -m experiments.phase0_vit_inversion \
    --device cuda \
    --seed 42

echo "=== Phase 0 Complete ==="
echo "Date: $(date)"

echo "=== Phase 0b: Noise Tolerance Sweep ==="
# Phase 0b: add noise to gradient, measure SSIM vs cosine similarity
python -u -m experiments.phase0_vit_inversion \
    --noise_sweep \
    --device cuda \
    --seed 42

echo "=== Phase 0b Complete ==="
echo "Date: $(date)"
