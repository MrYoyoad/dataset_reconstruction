#!/bin/bash
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
echo "=== scoring truncated certificate under the PRE-FIXED rule $(date) git=$(git rev-parse --short HEAD) ==="
python -u -m experiments.exact_inversion.score_truncated results/exact_inversion/step110_trunc_298988.jsonl
echo "=== DONE ==="
