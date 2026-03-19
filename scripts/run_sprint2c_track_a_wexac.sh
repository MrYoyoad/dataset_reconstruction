#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o sprint2c_a_%J.out
#BSUB -e sprint2c_a_%J.err
#BSUB -J sprint2c_a

# Sprint 2c Track A: Experiment A — KKT with fine-tuning LR × epochs × N sweep
# Grid: fine_tune_lr ∈ {0.001, 0.003, 0.01} × epochs ∈ {1M, 5M} × {full FT, LoRA r=8}
# N sweep: extraction_n_per_class ∈ {1, 5, 10, 25, 50, 100, 250, 251}
# 96 configs total

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== Starting Sprint 2c Track A (KKT LR × N sweep) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

python -u -m experiments.run_sprint2c_sweep \
    --track A \
    --device cuda \
    --seed 42

echo "=== Sprint 2c Track A Complete ==="
echo "Date: $(date)"
