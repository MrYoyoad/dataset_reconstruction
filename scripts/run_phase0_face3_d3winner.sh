#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -W 18:00
#BSUB -o wexac_logs/phase0_face3_d3winner_%J.out
#BSUB -e wexac_logs/phase0_face3_d3winner_%J.err
#BSUB -J phase0_face3_d3winner

# face3.jpg solo at the D3 winner config (signAdam, tv=1e-1, lr=0.05,
# freq=1e-3, 30K iters, 8 restarts, full ViT gradient).
# Companion to face_d3winner (face1, SSIM=0.522) and face2_d3winner.
# Together gives a 3-face panel of solo reconstructions.

set -e
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado

echo "=== Phase 0: face3.jpg solo (D3 winner config) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

python -u -m experiments.phase0_vit_inversion \
    --mode full \
    --image_path data/faces/face3.jpg \
    --optimizer signAdam \
    --tv_weight 1e-1 \
    --tv_norm l2 \
    --lr 0.05 \
    --n_iters 30000 \
    --n_restarts 8 \
    --freq_weight 1e-3 \
    --lpips_weight 0 \
    --device cuda \
    --seed 42 \
    --run_tag face3_d3winner_freq1e-3

echo ""
echo "=== face3 D3-winner reconstruction complete: $(date) ==="
