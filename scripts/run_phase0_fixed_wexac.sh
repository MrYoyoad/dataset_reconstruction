#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -W 8:00
#BSUB -o wexac_logs/phase0_fixed_%J.out
#BSUB -e wexac_logs/phase0_fixed_%J.err
#BSUB -J phase0_fix

# Phase 0 (fixed): ViT gradient inversion gate experiment
# Fixes: differentiable cosine sim, global (not per-tensor) cosine sim,
#        full-model gradient capture, random restarts, math.sqrt bug

set -e

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
mkdir -p results figures wexac_logs

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

echo "=== Phase 0: Full + LoRA gradient inversion ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

# Phase 0: both full-model (86M params) and LoRA-only (294K params)
python -u -m experiments.phase0_vit_inversion \
    --mode both \
    --n_iters 10000 \
    --n_restarts 8 \
    --device cuda \
    --seed 42

echo "=== Phase 0 Complete ==="
echo "Date: $(date)"

echo "=== Phase 0b: Noise Tolerance Sweep (full-model gradient) ==="
# Phase 0b: noise sweep using full-model gradient
python -u -m experiments.phase0_vit_inversion \
    --noise_sweep \
    --mode full \
    --n_iters 10000 \
    --n_restarts 4 \
    --device cuda \
    --seed 42

echo "=== Phase 0b Complete ==="
echo "Date: $(date)"
