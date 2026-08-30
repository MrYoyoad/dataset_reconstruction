#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0 && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -W 0:30
#BSUB -o scripts/wexac_logs/atlas_eco_%J.out
#BSUB -e scripts/wexac_logs/atlas_eco_%J.err
#BSUB -J atlas_eco
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) ==="
python -u -m experiments.dataset_sensitivity.atlas_ecosystem
echo "=== DONE $(date) ==="
