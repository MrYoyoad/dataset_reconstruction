#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o lora_free_coeff_%J.out
#BSUB -e lora_free_coeff_%J.err
#BSUB -J lora_free_coeff

# LoRA Free-Coefficient Extraction (Sprint 2a)
# Tests whether free-c + SGD (which matched oracle for full model) works for LoRA
# Runs: rank ∈ {8, 16, 32} × {free-c SGD, oracle SGD} for comparison
# Expected runtime: ~30-60 min on L40S

# Activate conda
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

cd /home/projects/galvardi/yoado

# Add dataset_reconstruction to path
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== Starting LoRA Free-Coefficient Experiments ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

# --- Free-coefficient + SGD (realistic attack) ---
for RANK in 8 16 32; do
    echo ""
    echo "=== LoRA rank=${RANK}, free-coefficient, SGD ==="
    echo "Start: $(date)"
    python -m experiments.run_experiment_b \
        --n_steps 1 \
        --rank $RANK \
        --free_coefficients \
        --consistency_weight 1.0 \
        --optimizer sgd \
        --no_baseline \
        --seed 42 \
        --save_results \
        --device cuda
    echo "Done: $(date)"
done

# --- Oracle + SGD (upper bound baseline for comparison) ---
for RANK in 8 16 32; do
    echo ""
    echo "=== LoRA rank=${RANK}, oracle, SGD ==="
    echo "Start: $(date)"
    python -m experiments.run_experiment_b \
        --n_steps 1 \
        --rank $RANK \
        --optimizer sgd \
        --no_baseline \
        --seed 42 \
        --save_results \
        --device cuda
    echo "Done: $(date)"
done

# --- Full model free-c + SGD (replicate the SSIM=0.997 result on CUDA) ---
echo ""
echo "=== Full model, free-coefficient, SGD ==="
echo "Start: $(date)"
python -m experiments.run_experiment_b \
    --n_steps 1 \
    --free_coefficients \
    --consistency_weight 1.0 \
    --optimizer sgd \
    --seed 42 \
    --no_baseline \
    --save_results \
    --device cuda
echo "Done: $(date)"

echo ""
echo "=== All experiments complete ==="
echo "Date: $(date)"
echo "Results saved in results/"
