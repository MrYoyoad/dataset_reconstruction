#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/anchor_activation_%J.out
#BSUB -e scripts/wexac_logs/anchor_activation_%J.err
#BSUB -J anchor_activation

# =====================================================================
# Step 2b of the coupled activation x anchor x linearization study
# (notes/next_experiment_plan.md). The anchor alpha-sweep has ONLY ever been run for GELU
# (job 532232). Step 1's rescore of job 857271 found SOFTPLUS is the best activation at matched
# weight_change (ssim_norm 0.65, l2 4.8 vs gelu ~18, ctrl_margin +0.115; feature_stability 0.993),
# and uniquely linearization-stable. This job runs the anchor two-curve (lin-error(alpha) vs
# SSIM/retrieval(alpha)) for the winner + comparators, at the SAME settings as the GELU run
# (seed 42, T=10, r=8), so the two-curves are directly comparable to figures/anchor_sweep/*gelu*.
#
# Proof-criterion connection: the winner should show LOWER function-space lin-error at every alpha
# and a peak that passes the attribution test (SSIM peaks at/before the lin-error minimum = a
# linearization win, not anchor x_i leakage).
#
# Order = highest value first (survives preemption): softplus s42 -> relu control s42 ->
# silu s42 -> softplus s44 (second seed for the winner). Tags include the activation & seed, so
# no filename collisions with the existing GELU tensors/figures. ~50 GPU-min each, ~3.3 GPU-hr.
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -c "import torch; print(f'CUDA={torch.cuda.is_available()}, dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
echo "=== START $(date) on $(hostname) ==="

run_anchor () {  # $1=activation  $2=seed
    echo ""; echo "########## anchor sweep act=$1 seed=$2 T=10 r=8 ##########"; date
    python -u -m experiments.run_anchor_sweep \
        --n_steps 10 --rank 8 --finetune_activation "$1" \
        --seed "$2" --save --device cuda
}

run_anchor softplus 42     # the Step-1 winner (vs existing gelu s42 two-curve)
run_anchor relu     42     # kinked control (C0) — the smoothness contrast
run_anchor silu     42     # runner-up (C-infinity)
run_anchor softplus 44     # second seed for the winner (harden it)

echo ""; echo "=== ALL STAGES COMPLETE $(date) ==="
