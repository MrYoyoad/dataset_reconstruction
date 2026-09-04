#!/bin/bash
# The deployment-relevance answer for the counting rule: how many private images the recipe-free channel admits,
# per architecture, as arithmetic from published specs -- anchored to the rows measured on four pretrained models.
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START capacity table $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
python -u -m experiments.exact_inversion.capacity_table --N 8 --out $OUT/step109_capacity_${LSB_JOBID}.jsonl
echo "=== DONE $(date) ==="
