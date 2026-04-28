#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -W 8:00
#BSUB -o wexac_logs/phase0_d1_A_%J.out
#BSUB -e wexac_logs/phase0_d1_A_%J.err
#BSUB -J phase0_d1_A

set -e

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
mkdir -p results figures scripts/wexac_logs

pip install -q scipy 2>/dev/null

echo "=== D1 Config A: optimizer=Adam, tv_weight=1e-4 ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

python -u -m experiments.phase0_vit_inversion \
    --mode full \
    --dataset flowers102 \
    --n_iters 10000 \
    --n_restarts 8 \
    --optimizer Adam \
    --tv_weight 1e-4 \
    --tv_norm l2 \
    --device cuda \
    --seed 42

echo "=== D1 Config A Complete ==="
echo "Date: $(date)"
