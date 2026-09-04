#!/bin/bash
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
echo "=== scoring live regime under the locked criteria $(date) git=$(git rev-parse --short HEAD) ==="
python -u -m experiments.exact_inversion.score_live results/exact_inversion/step116_live_307760.jsonl
echo "=== DONE ==="
