#!/bin/bash
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START enumerated FPR $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
python -u -m experiments.exact_inversion.enumerated_fpr --nonmembers 8000 --enum-sample 1500 \
    --r 64 --T 200 --lr 0.05 --seed 1 --out $OUT/step120_enumfpr_${LSB_JOBID}.jsonl
echo "=== DONE $(date) ==="
