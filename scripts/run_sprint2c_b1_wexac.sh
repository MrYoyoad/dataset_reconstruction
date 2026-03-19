#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o sprint2c_b1_%J.out
#BSUB -e sprint2c_b1_%J.err
#BSUB -J sprint2c_b1

# Sprint 2c Track B1: Re-run killed Sprint 2b Phases 3+4
# Phase 3: LR scaling (9 T × 3 schedules = 27 configs)
# Phase 4: Progressive warm-start (9 T × 2 ranks = 18 configs)

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== Starting Sprint 2c B1 (Phases 3+4 re-run) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

python -u -m experiments.run_sprint2b_sweep \
    --phase 3 4 \
    --finetune_activation leaky_relu \
    --device cuda \
    --seed 42

echo "=== Sprint 2c B1 Complete ==="
echo "Date: $(date)"
