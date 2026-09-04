#!/bin/bash
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
echo "=== ROC over the near-duplicate ladder $(date) git=$(git rev-parse --short HEAD) ==="
python -u -m experiments.exact_inversion.roc_neardupe results/exact_inversion/step117_neardupe_313899.jsonl
echo "=== DONE ==="
