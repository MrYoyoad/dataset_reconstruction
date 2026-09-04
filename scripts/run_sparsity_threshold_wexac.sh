#!/bin/bash
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START sparsity threshold $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
python -u -m experiments.exact_inversion.sparsity_threshold --N 8 --conditions 8 56 248 \
    --out $OUT/step111_sparsity_${LSB_JOBID}.jsonl
echo "=== DONE $(date) ==="
