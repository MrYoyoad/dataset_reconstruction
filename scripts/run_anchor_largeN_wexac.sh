#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/anchor_largeN_%J.out
#BSUB -e scripts/wexac_logs/anchor_largeN_%J.err
#BSUB -J anchor_largeN

# =====================================================================
# Anchor alpha-sweep at LARGER N (N=10, n_per_class=5).
#
# Motivation: at N=2 the LoRA-only path never beats the trivial mean
# baseline (best LoRA SSIM 0.64 vs baseline 0.763) — the mean image already
# resembles each of the 2 targets. Hypothesis: at larger N the mean baseline
# drops and becomes meaningful, so genuine anchored LoRA recovery should start
# beating it. This is the key test of whether LoRA-adapter-only leakage (the
# thesis target) is real or a small-N artifact.
#
# Larger N also makes near-zero Delta-w (the degenerate-digit trap that makes
# SSIM identical across alpha and lin-error 0.0000) far less likely.
#
# Config: N=10, T=10, rank=8, GELU, seed 42, alpha in {0, 0.5, 0.75, 0.9},
#         --save (per-alpha tensors+grids + aggregate .pth + two-curve plots),
#         raised --verify_weight 5.0 to tighten the soft box constraint and
#         reduce [0,1] pixel saturation (clip frac was 0.53-0.67 at N=2, which
#         inflates SSIM).
# Tag includes N10 so files do NOT collide with the N=2 run.
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -c "import torch; print(f'CUDA={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"
echo "=== START $(date) on $(hostname) ==="

ACT=gelu
T=10
RANK=8
TAG=T10_r8_gelu_s42_N10_vw5

echo ""; echo "########## STAGE 0: GPU smoke (2 alphas, 300 epochs, N=10) ##########"
python -u -m experiments.run_anchor_sweep \
    --alphas 0 0.5 --n_steps $T --rank $RANK --n_per_class 5 \
    --finetune_activation $ACT --extraction_epochs 300 \
    --verify_weight 5.0 --device cuda
if [ $? -ne 0 ]; then
    echo "FATAL: anchor-sweep large-N smoke failed. Aborting."
    exit 1
fi
echo "Stage 0 PASSED — large-N anchor sweep runs on GPU."

echo ""; echo "########## STAGE 1: full anchor sweep N=10 (both paths) ##########"; date
python -u -m experiments.run_anchor_sweep \
    --alphas 0 0.5 0.75 0.9 --n_steps $T --rank $RANK --n_per_class 5 \
    --finetune_activation $ACT --seed 42 --save \
    --verify_weight 5.0 --tag $TAG --device cuda

echo ""; echo "=== ALL STAGES COMPLETE $(date) ==="
