#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -W 12:00
#BSUB -o wexac_logs/phase0_face4_%J.out
#BSUB -e wexac_logs/phase0_face4_%J.err
#BSUB -J phase0_face4

set -e

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
mkdir -p results figures/phase0/snapshots scripts/wexac_logs

echo "=== Phase 0: Face Sweep — F4 (resubmit) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

FACE="data/faces/face1.jpg"

# F4: signAdam, tv=5e-2, 30K iters, 8 restarts (stronger TV than F3)
echo ""
echo "=== F4: signAdam, tv=5e-2, 30K iters ==="
echo "Date: $(date)"
python -u -m experiments.phase0_vit_inversion \
    --mode full --image_path "$FACE" \
    --n_iters 30000 --n_restarts 8 \
    --optimizer signAdam --tv_weight 5e-2 --tv_norm l2 \
    --device cuda --seed 42

echo ""
echo "=== F4 Complete ==="
echo "Date: $(date)"
