#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -W 0:30
#BSUB -o scripts/wexac_logs/rescore_freec_%J.out
#BSUB -e scripts/wexac_logs/rescore_freec_%J.err
#BSUB -J rescore_freec
# Rescore the free-c wc-ladder tensors (job 392821) into a CSV for the realistic crux ranking.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
python -u -m experiments.recompute_metrics \
    --glob 'results/exp_b_T1_r8_free_s42_a149_*.pth' \
    --out results/rescored_freec_ladder_2026-08-29.csv
echo "=== DONE $(date) ==="
wc -l results/rescored_freec_ladder_2026-08-29.csv
