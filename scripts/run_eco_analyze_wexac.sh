#!/bin/bash
#BSUB -q short
#BSUB -R "rusage[mem=8192]"
#BSUB -W 0:15
#BSUB -o scripts/wexac_logs/eco_analyze_%J.out
#BSUB -e scripts/wexac_logs/eco_analyze_%J.err
#BSUB -J eco_analyze
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) ==="
python -u -m experiments.dataset_sensitivity.eco_analyze
echo "=== DONE $(date) ==="
