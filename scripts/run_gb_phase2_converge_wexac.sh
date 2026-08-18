#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/gb_phase2_converge_%J.out
#BSUB -e scripts/wexac_logs/gb_phase2_converge_%J.err
#BSUB -J gb_phase2_converge
# Confound rule-out before closing the bridge chapter: is two-sided r=8 (best config; softplus
# 0.888/SSIM 0.136, gelu 0.634/0.097 at n8000/ep60) CONVERGED or under-trained? Push 4x data (30000)
# + 3x epochs (200). If the decode/SSIM jumps -> training-limited (fixable); if it plateaus -> the
# coarse-recovery ceiling is fundamental. r=8 (no OOM, no dilution confound).
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
python -u -m experiments.gradient_bridge.phase2_image \
    --activations softplus gelu --two_sided --rank 8 \
    --n_train 30000 --n_eval 128 --epochs 200 --device cuda
echo "=== DONE $(date) ==="
