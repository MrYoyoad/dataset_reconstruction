#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/anchor_activation_N10_%J.out
#BSUB -e scripts/wexac_logs/anchor_activation_N10_%J.err
#BSUB -J anchor_activation_N10

# =====================================================================
# Step 2b at LARGER N (N=10) — anchor alpha-sweep two-curve for the winner softplus + gelu
# (reference) + relu (kinked control), at n_per_class=5, T=10, r=8, --verify_weight 5.0 (matches
# the existing N=10 gelu anchor job 957044, which found full-FT alpha*~0.75 persists at N=10).
# Question: does softplus give a cleaner two-curve (lower lin-error, wider usable alpha window)
# than gelu/relu at N=10? Tags include activation + N10, so no collision with the N=2 tensors.
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname) ==="

for ACT in softplus gelu relu; do
    echo ""; echo "########## N=10 anchor act=$ACT T=10 r=8 ##########"; date
    python -u -m experiments.run_anchor_sweep \
        --n_steps 10 --rank 8 --finetune_activation "$ACT" \
        --n_per_class 5 --seed 42 --verify_weight 5.0 \
        --tag "T10_r8_${ACT}_s42_N10_vw5" --save --device cuda
done

echo ""; echo "=== ALL STAGES COMPLETE $(date) ==="
