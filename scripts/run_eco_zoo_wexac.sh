#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -W 0:30
#BSUB -o scripts/wexac_logs/eco_zoo_%J.out
#BSUB -e scripts/wexac_logs/eco_zoo_%J.err
#BSUB -J eco_zoo
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) ==="
python -u -m experiments.dataset_sensitivity.eco_zoo --save --device cuda
echo "=== DONE $(date) ==="
