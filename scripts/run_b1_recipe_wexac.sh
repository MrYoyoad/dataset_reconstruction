#!/bin/bash
#BSUB -q short
#BSUB -R "rusage[mem=8192]"
#BSUB -W 0:15
#BSUB -o scripts/wexac_logs/b1_recipe_%J.out
#BSUB -e scripts/wexac_logs/b1_recipe_%J.err
#BSUB -J b1_recipe
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) ==="
python -u -m experiments.dataset_sensitivity.b1_recipe_invariance
echo "=== DONE $(date) ==="
