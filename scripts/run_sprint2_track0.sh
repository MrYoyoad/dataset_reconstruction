#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o sprint2_track0_%J.out
#BSUB -e sprint2_track0_%J.err
#BSUB -J sprint2_t0

# Sprint 2, Track 0: ModifiedRelu Extraction Fix
# Tests whether using ModifiedRelu (smooth backward pass) for extraction
# improves reconstruction at T=1 and enables T>1.

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== Sprint 2, Track 0: ModifiedRelu Extraction Fix ==="
echo "Started at $(date)"
echo "Device: cuda"
nvidia-smi | head -5

# Task 0.1: ModifiedRelu at T=1,2,5,10 (full model)
echo ""
echo "=== Task 0.1: ModifiedRelu at different T values ==="
for T in 1 2 5 10; do
    echo ""
    echo "--- T=$T, full model ---"
    python -m experiments.run_experiment_b \
        --n_steps $T --seed 42 --device cuda --save_results
done

# Task 0.2: ModifiedRelu + LoRA ranks at T=1
echo ""
echo "=== Task 0.2: ModifiedRelu + LoRA ranks at T=1 ==="
for R in 8 32 64; do
    echo ""
    echo "--- T=1, LoRA rank=$R ---"
    python -m experiments.run_experiment_b \
        --n_steps 1 --rank $R --seed 42 --device cuda --save_results
done

echo ""
echo "=== Track 0 complete at $(date) ==="
