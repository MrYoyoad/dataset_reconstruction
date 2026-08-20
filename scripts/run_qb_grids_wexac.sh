#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=8192] select[ngpus>0]"
#BSUB -gpu "num=1:gmem=2000"
#BSUB -o scripts/wexac_logs/qb_grids_%J.out
#BSUB -e scripts/wexac_logs/qb_grids_%J.err
#BSUB -J qb_grids

# Regenerate the clip-free Q-B seen/novel example grids from the pixel-box .pth tensors (job 952081).
# CPU-only work but submitted to the cluster (never the login node).
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
python -u scripts/make_qb_grids.py
echo "=== DONE $(date) ==="
ls -la figures/pdf_examples/FREEC_QB_*_pbox.png
