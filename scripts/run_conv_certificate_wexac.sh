#!/bin/bash
# =====================================================================
# The CONVOLUTIONAL comparison: does the recipe-free certificate survive weight sharing?
# A dense layer records ONE vector per image; a conv records P patch vectors per image, so the recorded span
# `N'` may fill the layer's input dimension and leave margin 0 at every rank -- no certificate at all.
#   bsub -q long-gpu -gpu "num=1" -R "rusage[mem=16384] select[ngpus>0]" -J ei90_conv \
#        -o scripts/wexac_logs/ei90_conv_%J.out -e scripts/wexac_logs/ei90_conv_%J.err \
#        bash scripts/run_conv_certificate_wexac.sh
# =====================================================================
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/exact_inversion; mkdir -p $OUT
python -c "import torch; print(f'CUDA={torch.cuda.is_available()} torch={torch.__version__}')"
echo "=== START conv_certificate $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
python -u -m experiments.exact_inversion.conv_certificate --N 8 --ranks 8 16 32 64 128 256 512 \
    --T 200 --lr 0.01 --epochs 6 --seed 1 --out $OUT/step90_conv_${LSB_JOBID}.jsonl
echo "=== DONE $(date) ==="
