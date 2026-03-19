#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o activation_ablation_%J.out
#BSUB -e activation_ablation_%J.err
#BSUB -J activation_ablation

# Activation Function Ablation for LoRA Extraction
# Tests: relu_alpha ∈ {10, 50, 150, 10000(≈ReLU)} × optimizer ∈ {lbfgs, sgd}
# Fixed: LoRA rank=8, T=1, oracle coefficients, seed=42
# Goal: understand why Sprint 1 (L-BFGS, old code) got SSIM=0.797
#        but current (SGD, ModifiedRelu) gets SSIM=0.183

# Activate conda
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

cd /home/projects/galvardi/yoado

export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== Starting Activation Ablation ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

# Sweep alpha × optimizer for LoRA r=8 (oracle coefficients)
for ALPHA in 10 50 150 10000; do
    for OPT in lbfgs sgd; do
        echo ""
        echo "=== alpha=${ALPHA}, optimizer=${OPT}, LoRA r=8, oracle ==="
        echo "Start: $(date)"
        python -m experiments.run_experiment_b \
            --n_steps 1 \
            --rank 8 \
            --optimizer $OPT \
            --relu_alpha $ALPHA \
            --no_baseline \
            --seed 42 \
            --save_results \
            --device cuda
        echo "Done: $(date)"
    done
done

# Also run free-c + SGD with best alpha candidates
for ALPHA in 10 10000; do
    echo ""
    echo "=== alpha=${ALPHA}, SGD, LoRA r=8, FREE-COEFFICIENT ==="
    echo "Start: $(date)"
    python -m experiments.run_experiment_b \
        --n_steps 1 \
        --rank 8 \
        --free_coefficients \
        --consistency_weight 1.0 \
        --optimizer sgd \
        --relu_alpha $ALPHA \
        --no_baseline \
        --seed 42 \
        --save_results \
        --device cuda
    echo "Done: $(date)"
done

echo ""
echo "=== Ablation complete ==="
echo "Date: $(date)"
