#!/bin/bash
# THE FALSIFIER for nullity = N*(d - (m-1) - r + N), derived two ways that reconcile to one formula.
# Predictions, fixed before the rows: m = 40 -> 72, m = 48 -> 8, m = 49 -> 0, m = 64 -> 0, and at m = 20
# N = 4 -> 100, N = 1 -> 22. A discontinuous drop to EXACTLY zero between two adjacent head widths is not
# something a wrong derivation produces by accident. The N cells settle the remedy: the bracket is 21 + N at
# m = 20, positive for every N >= 1, so shrinking the batch lowers the nullity and never closes it. Only the head
# closes it, at m >= d + N + 1 - r = 49.
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
echo "=== START fibre sweep $(date) git=$(git rev-parse --short HEAD) ==="
for M in 40 48 49 64; do
  echo "### m=$M N=8 (predicted 72 / 8 / 0 / 0)"
  python -u -m experiments.e1b.fibre_dimension_check --T 20 --m $M
done
for NN in 1 4; do
  echo "### m=20 N=$NN (predicted 22 / 100)"
  python -u -m experiments.e1b.fibre_dimension_check --T 20 --N $NN
done
echo "=== DONE $(date) ==="
