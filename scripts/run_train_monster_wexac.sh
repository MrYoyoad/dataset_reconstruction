#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/train_monster_%J.out
#BSUB -e scripts/wexac_logs/train_monster_%J.err
#BSUB -J train_monster
# Train the wide+deep 'monster' MLP base model (theta_0) on CIFAR-10 parity (3072-2048x4-1, ~19M params).
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
python -u -m experiments.train_monster_base --hidden 2048 2048 2048 2048 --dpc 250 \
    --epochs 40000 --lr 1e-3 --init_scale 0.5 --device cuda
echo "=== DONE $(date) ==="
