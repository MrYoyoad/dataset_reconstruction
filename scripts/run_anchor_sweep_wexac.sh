#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/anchor_sweep_%J.out
#BSUB -e scripts/wexac_logs/anchor_sweep_%J.err
#BSUB -J anchor_sweep

# =====================================================================
# Addition 3 — anchor alpha-sweep + two-curve validation.
# NOT covered by job 435843 (needed --anchor_alpha, which now exists).
#
#   Stage 0  smoke  : GPU wiring check — 2 alphas, few epochs. Abort on fail.
#   Stage 1  sweep  : full ANCHOR_ALPHA_SWEEP, BOTH full-FT and LoRA paths,
#                     GELU, T=10, seed 42, --save. Emits the two-curve plots.
#
# Deliverable: figures/anchor_sweep/anchor_two_curve_{full,lora}_*.png
#   Legit win = lin-error bottoms where SSIM peaks. Red flag = SSIM climbs past.
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

# ---------------------------------------------------------------------
# Stage 0 — GPU smoke: 2 alphas, tiny extraction budget. Confirms the
# anchor wiring runs end to end on GPU before the full sweep burns time.
# ---------------------------------------------------------------------
echo ""; echo "########## STAGE 0: GPU smoke (2 alphas, 300 epochs) ##########"
python -u -m experiments.run_anchor_sweep \
    --alphas 0 0.5 --n_steps $T --rank $RANK \
    --finetune_activation $ACT --extraction_epochs 300 --device cuda
if [ $? -ne 0 ]; then
    echo "FATAL: anchor-sweep smoke failed. Aborting."
    exit 1
fi
echo "Stage 0 PASSED — anchor sweep runs on GPU."

# ---------------------------------------------------------------------
# Stage 1 — the real sweep. Full alpha grid, both paths, --save so per-alpha
# tensors + grids AND the aggregate .pth + two-curve plots are all written.
# ~50 GPU-min at the default extraction budget.
# ---------------------------------------------------------------------
echo ""; echo "########## STAGE 1: full anchor sweep (both paths) ##########"; date
python -u -m experiments.run_anchor_sweep \
    --n_steps $T --rank $RANK --finetune_activation $ACT \
    --seed 42 --save --device cuda

echo ""; echo "=== ALL STAGES COMPLETE $(date) ==="
