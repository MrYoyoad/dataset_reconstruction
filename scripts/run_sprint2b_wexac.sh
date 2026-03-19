#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o sprint2b_%J.out
#BSUB -e sprint2b_%J.err
#BSUB -J sprint2b

# Sprint 2b: Remaining phases (3, 4) with LeakyReLU
# Job 669864: completed Phases 0+1, killed at Phase 2
# Job 674627: completed Phase 2, killed at Phase 3 (3/27)
# This run: Phases 3+4 only

# Activate conda
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

cd /home/projects/galvardi/yoado

# Add dataset_reconstruction to path
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== Starting Sprint 2b Phases 3-4 (LeakyReLU) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

# Run remaining phases with leaky_relu (best from Phase 0)
python -u -m experiments.run_sprint2b_sweep \
    --phase 3 4 \
    --finetune_activation leaky_relu \
    --device cuda \
    --seed 42

echo "=== Sprint 2b Phases 3-4 Complete ==="
echo "Date: $(date)"
echo "Results saved in results/"
