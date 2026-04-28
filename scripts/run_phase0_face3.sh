#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -W 8:00
#BSUB -o wexac_logs/phase0_face3_%J.out
#BSUB -e wexac_logs/phase0_face3_%J.err
#BSUB -J phase0_face3

set -e

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
mkdir -p results figures scripts/wexac_logs

pip install -q scipy 2>/dev/null

echo "=== Phase 0: Face 3 Gradient Inversion (D1 winner config) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

python -u -m experiments.phase0_vit_inversion \
    --mode full \
    --image_path data/faces/face3.jpg \
    --n_iters 10000 \
    --n_restarts 8 \
    --optimizer signAdam \
    --tv_weight 1e-2 \
    --tv_norm l2 \
    --device cuda \
    --seed 42

echo "=== Face 3 Complete ==="
echo "Date: $(date)"
