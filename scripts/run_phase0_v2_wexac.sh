#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -W 8:00
#BSUB -o wexac_logs/phase0_v2_%J.out
#BSUB -e wexac_logs/phase0_v2_%J.err
#BSUB -J phase0_v2

# Phase 0 v2: ViT gradient inversion with audit fixes
# Changes from v1:
#   - Native high-res images (Flowers102, ~500px) instead of upscaled CIFAR-10
#   - Proper windowed SSIM (Kornia) instead of global-mean SSIM
#   - backward(inputs=[x_recon]) for faster iterations
#   - Signed Adam optimizer (Geiping et al. 2020 recommendation)
#   - TV normalized by pixel count
#   - Tighter image clamping to valid ImageNet-normalized range
#   - Correct LoRA effective dims reporting (~147K B-only, not 294K)

set -e

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
mkdir -p results figures scripts/wexac_logs

pip install -q scipy 2>/dev/null  # Flowers102 needs scipy for .mat labels

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
python -c "import kornia; print(f'Kornia: {kornia.__version__}')"

echo "=== Phase 0 v2: Full + LoRA gradient inversion (signAdam) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

# Phase 0: both full-model (86M params) and LoRA-only (~147K effective)
# Using signAdam (Geiping et al. recommended) and native high-res images
python -u -m experiments.phase0_vit_inversion \
    --mode both \
    --dataset flowers102 \
    --n_iters 10000 \
    --n_restarts 8 \
    --optimizer signAdam \
    --tv_norm l2 \
    --device cuda \
    --seed 42

echo "=== Phase 0 v2 Complete ==="
echo "Date: $(date)"

echo "=== Phase 0b: Noise Tolerance Sweep (full-model gradient) ==="
# Phase 0b: noise sweep using full-model gradient
python -u -m experiments.phase0_vit_inversion \
    --noise_sweep \
    --mode full \
    --dataset flowers102 \
    --n_iters 10000 \
    --n_restarts 4 \
    --optimizer signAdam \
    --device cuda \
    --seed 42

echo "=== Phase 0b Complete ==="
echo "Date: $(date)"
