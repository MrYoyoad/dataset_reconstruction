#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=8192] select[ngpus>0]"
#BSUB -gpu "num=1:gmem=2000"
#BSUB -o scripts/wexac_logs/redo_figs_%J.out
#BSUB -e scripts/wexac_logs/redo_figs_%J.err
#BSUB -J redo_figs
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) ==="; python -u scripts/make_redo_figs.py
echo "=== DONE $(date) ==="; ls -la figures/pdf_examples/REDO_*.png
