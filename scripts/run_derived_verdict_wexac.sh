#!/bin/bash
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
echo "=== derived four-cell verdict over the whole corpus $(date) git=$(git rev-parse --short HEAD) ==="
python -u -m experiments.exact_inversion.derived_verdict "results/exact_inversion/*.jsonl"
echo "=== DONE ==="
