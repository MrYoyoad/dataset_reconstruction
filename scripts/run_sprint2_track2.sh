#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o sprint2_track2_%J.out
#BSUB -e sprint2_track2_%J.err
#BSUB -J sprint2_t2

# Sprint 2 Track 2 v3: Adam for c + random restarts for stubborn r=16/32
# Previous: SGD for c got coeff_err=0.28-0.31 at r=16/32
# This run: Adam for c (adaptive lr), multiple seeds for restarts

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
export PYTHONUNBUFFERED=1

echo "=== Sprint 2 Track 2 v3: Adam for c + restarts ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# --- Adam for c at r=16/32 (the stubborn ranks) ---
for RANK in 16 32; do
    echo ""
    echo "=== LoRA r=${RANK}, free-c, L-BFGS+Adam(c), lr=0.01, 5Kep ==="
    echo "Start: $(date)"
    python -m experiments.run_experiment_b \
        --n_steps 1 \
        --rank $RANK \
        --free_coefficients \
        --consistency_weight 1.0 \
        --coeff_lr 0.01 \
        --coeff_optimizer adam \
        --optimizer lbfgs \
        --relu_alpha 10000 \
        --extraction_epochs 5000 \
        --no_baseline \
        --seed 42 \
        --save_results \
        --device cuda
    echo "Done: $(date)"
done

# --- Higher consistency weight for r=16/32 ---
for RANK in 16 32; do
    echo ""
    echo "=== LoRA r=${RANK}, free-c, consistency=5.0, L-BFGS+SGD(c), 5Kep ==="
    echo "Start: $(date)"
    python -m experiments.run_experiment_b \
        --n_steps 1 \
        --rank $RANK \
        --free_coefficients \
        --consistency_weight 5.0 \
        --coeff_lr 0.01 \
        --optimizer lbfgs \
        --relu_alpha 10000 \
        --extraction_epochs 5000 \
        --no_baseline \
        --seed 42 \
        --save_results \
        --device cuda
    echo "Done: $(date)"
done

# --- Random restarts: multiple seeds for r=16 (best of 5) ---
echo ""
echo "=== Random restarts for r=16 (seeds 42,7,123,999,2026) ==="
for SEED in 42 7 123 999 2026; do
    echo ""
    echo "--- r=16, seed=${SEED} ---"
    echo "Start: $(date)"
    python -m experiments.run_experiment_b \
        --n_steps 1 \
        --rank 16 \
        --free_coefficients \
        --consistency_weight 1.0 \
        --coeff_lr 0.01 \
        --optimizer lbfgs \
        --relu_alpha 10000 \
        --extraction_epochs 5000 \
        --no_baseline \
        --seed $SEED \
        --save_results \
        --device cuda
    echo "Done: $(date)"
done

# --- Random restarts: multiple seeds for r=32 (best of 5) ---
echo ""
echo "=== Random restarts for r=32 (seeds 42,7,123,999,2026) ==="
for SEED in 42 7 123 999 2026; do
    echo ""
    echo "--- r=32, seed=${SEED} ---"
    echo "Start: $(date)"
    python -m experiments.run_experiment_b \
        --n_steps 1 \
        --rank 32 \
        --free_coefficients \
        --consistency_weight 1.0 \
        --coeff_lr 0.01 \
        --optimizer lbfgs \
        --relu_alpha 10000 \
        --extraction_epochs 5000 \
        --no_baseline \
        --seed $SEED \
        --save_results \
        --device cuda
    echo "Done: $(date)"
done

echo ""
echo "=== Track 2 v3 complete ==="
echo "Date: $(date)"
