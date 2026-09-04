#!/bin/bash
# The dose axis's SPAN depends on the s at which it is quoted, and s = 50 was arbitrary. At s = 50 the span is
# 7.2x across 12 (dataset, basis) cells, below b9's pre-registered 10x bar, so the dose-response is NOT scoreable
# there. This sweeps s so the span curve is visible and the choice of s can be fixed by a stated rule rather than
# by whichever value happens to clear the bar. I am not choosing s myself: picking it to pass a threshold is the
# gaming the bar exists to prevent.
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START dose span sweep $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
for S in 10 20 50 100 150 200 300; do
  python -u -m experiments.exact_inversion.dose_table --s $S --out $OUT/step112_dosespan_${LSB_JOBID}.jsonl \
      2>/dev/null | grep -E "^# span"
done
echo "=== DONE $(date) ==="
