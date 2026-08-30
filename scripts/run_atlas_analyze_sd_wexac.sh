#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -W 0:30
#BSUB -o scripts/wexac_logs/atlas_analyze_sd_%J.out
#BSUB -e scripts/wexac_logs/atlas_analyze_sd_%J.err
#BSUB -J atlas_analyze_sd
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
[ -s results/atlas_zoo/zoo_bank.pth ] || { echo "STAGE0 FAIL: no zoo_bank.pth"; exit 1; }
python -u -m experiments.dataset_sensitivity.atlas_analyze --bank results/atlas_zoo/zoo_bank_samedigits.pth --out figures/atlas/atlas_samedigits.png
echo "=== DONE $(date) ==="
