#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o b6_extra_seeds_%J.out
#BSUB -e b6_extra_seeds_%J.err
#BSUB -J b6_extra_seeds

# EXP-29b: B6 AdamW multi-seed with 10 additional seeds
# 10 seeds × 2 optimizers × 2 T × 2 ranks = 80 configs

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== B6 Extra Seeds (10 new seeds) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

python -u -m experiments.run_diagnostic \
    --task b6_seeds \
    --device cuda \
    --seeds 1 13 37 77 256 314 500 777 1234 2025

echo "=== Complete ==="
echo "Date: $(date)"
