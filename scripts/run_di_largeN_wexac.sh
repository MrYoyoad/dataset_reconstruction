#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/di_largeN_%J.out
#BSUB -e scripts/wexac_logs/di_largeN_%J.err
#BSUB -J di_largeN

# =====================================================================
# DI-large-N — direct weight inversion at larger N (mean-baseline test).
#
# At N=4 the dataset mean already resembles each MNIST digit (baseline
# SSIM 0.674), so DI's ~0.55 was largely the mean showing through, not
# instance recovery (see results/rescored_metrics_2026-07-22.csv).
#
# Hypothesis: at larger N the mean baseline drops and becomes a real bar.
# If DI does genuine recovery it should start BEATING the baseline; if it
# still fails, we honestly learn DI on this toy does not leak.
#
# Arms (T in {1,10}, LoRA r=8, GELU, n_restarts=4, outer_iters=3000):
#   N=10  box_weight=1.0  (default)                      -> tag N10_r8_gelu
#   N=20  box_weight=1.0  (default)                      -> tag N20_r8_gelu
#   N=10  box_weight=5.0  (tighter box, curb saturation) -> tag N10_r8_gelu_box5
# Each --save writes results/direct_inversion_<tag>.pth (dict keyed by T,
# each with x_recon [centered], x_train [pixel], ds_mean) + grids + curve.
# ~15-30 GPU-min per N.
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -u -c "import torch; print(f'CUDA={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"
echo "=== START $(date) on $(hostname) ==="

# ---------------------------------------------------------------------
# Arm 1 — N=10, default box_weight.
# ---------------------------------------------------------------------
echo ""; echo "########## ARM 1: N=10 (box_weight=1.0) ##########"; date
python -u -m experiments.direct_inversion --sweep_T \
    --Ts 1 10 --N 10 --outer_iters 3000 --n_restarts 4 \
    --save --device cuda

# ---------------------------------------------------------------------
# Arm 2 — N=20, default box_weight.
# ---------------------------------------------------------------------
echo ""; echo "########## ARM 2: N=20 (box_weight=1.0) ##########"; date
python -u -m experiments.direct_inversion --sweep_T \
    --Ts 1 10 --N 20 --outer_iters 3000 --n_restarts 4 \
    --save --device cuda

# ---------------------------------------------------------------------
# Arm 3 — N=10, higher box_weight to curb [0,1] saturation (clip 0.42-0.52).
# ---------------------------------------------------------------------
echo ""; echo "########## ARM 3: N=10 (box_weight=5.0) ##########"; date
python -u -m experiments.direct_inversion --sweep_T \
    --Ts 1 10 --N 10 --outer_iters 3000 --n_restarts 4 \
    --box_weight 5.0 --tag N10_r8_gelu_box5 \
    --save --device cuda

echo ""; echo "=== ALL ARMS COMPLETE $(date) ==="
