#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o diagnostic_%J.out
#BSUB -e diagnostic_%J.err
#BSUB -J diag_n6_b6

# Diagnostic experiments:
# 1. N=6 with 500K extraction epochs (8 configs, slow — 500K epochs each)
#    Tests whether B5 collapse at N>2 is optimization or information limit
# 2. B6 AdamW multi-seed (80 configs: 5 seeds × 2 optimizers × 2 T × 2 ranks × 2 schedules)
#    Validates the surprising AdamW T=10 LoRA=0.785 result

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== Starting Diagnostic Experiments ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

python -u -m experiments.run_diagnostic \
    --task n6_diagnostic b6_seeds \
    --device cuda \
    --seed 42

echo "=== Diagnostics Complete ==="
echo "Date: $(date)"
