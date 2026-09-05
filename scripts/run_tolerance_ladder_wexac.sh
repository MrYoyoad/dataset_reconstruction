#!/bin/bash
# ROBUSTNESS PROBE, not a scored arm: no verdicts. 7e's instruction is to check whether the in-band split is real
# before adding cells, rather than adding cells blindly. Two-in / two-above is the right shape for a crossing test,
# but whether k = 62 and 64 are REALLY in band depends on N' = rank(B_T) at the projector tolerance. Run the
# declared ladder on all four cells: if a cell's in_band is CONSTANT across it, the split is real and two per side
# suffices; if any cell FLIPS, that cell is a BOUNDARY cell and one more is added on the robust side of it.
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START tolerance ladder (probe, no verdicts) $(date) git=$(git rev-parse --short HEAD) ==="
for TOL in 1e-6 1e-8 1e-10 1e-12; do
  echo "=== projector tolerance $TOL ==="
  python -u -m experiments.exact_inversion.constrained_replay --sets confident repeated hard1_diff \
      --r 64 --N 8 --ks 62 --tol $TOL --seed 1 --out $OUT/step123_tolladder_${LSB_JOBID}.jsonl 2>&1 | grep -E "PROBE|n_prime"
done
echo "=== DONE $(date) ==="
