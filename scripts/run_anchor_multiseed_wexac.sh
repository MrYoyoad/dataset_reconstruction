#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/anchor_multiseed_%J.out
#BSUB -e scripts/wexac_logs/anchor_multiseed_%J.err
#BSUB -J anchor_multiseed

# =====================================================================
# Harden the anchor α-sweep result (α*≈0.75 at T=10, seed 42, was single-seed).
#   - Multi-seed at T=10: seeds 43, 44 (seed 42 already done) -> does α*≈0.75 replicate?
#   - T-sweep at seed 42: T=5 and T=20 -> does the optimum α* shift with step count?
# Each sweep saves its own tensors + grids + two-curve plots (tag includes seed & T,
# so no filename collisions). ~50 GPU-min per sweep, ~3.3 GPU-hr total.
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname) ==="

# Multi-seed replication at T=10 (the setting where α*≈0.75 was found)
for S in 43 44; do
    echo ""; echo "########## anchor sweep seed=$S T=10 ##########"; date
    python -u -m experiments.run_anchor_sweep \
        --n_steps 10 --rank 8 --finetune_activation gelu \
        --seed $S --save --device cuda
done

# T-sweep at seed 42 — does the optimum shift with the number of fine-tuning steps?
for T in 5 20; do
    echo ""; echo "########## anchor sweep seed=42 T=$T ##########"; date
    python -u -m experiments.run_anchor_sweep \
        --n_steps $T --rank 8 --finetune_activation gelu \
        --seed 42 --save --device cuda
done

echo ""; echo "=== ALL STAGES COMPLETE $(date) ==="
