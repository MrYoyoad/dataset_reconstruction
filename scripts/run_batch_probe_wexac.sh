#!/bin/bash
# DESIGN PROBE, not a scored arm: no verdict is emitted and nothing here is scored.
# b9's ruling on test 6: in a single-batch sweep, band membership is PERFECTLY CONFOUNDED with k, because k rises
# monotonically and the line is fixed. A degradation at k=66 is equally consistent with "crossed the line" and
# with "k got larger", and no number of cells along one batch separates them. The fix is a SECOND BATCH with a
# different certified count, hence a different line, chosen so that at least one value of k is IN BAND for one
# batch and ABOVE the line for the other -- at that k the chart dimension is identical and only band membership
# differs, which is the only comparison that can attribute a degradation to the line.
# This probes the available batch compositions for their certified counts, so that second batch can be CHOSEN by
# its line rather than hoped for.
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START batch probe (design only, no verdicts) $(date) git=$(git rev-parse --short HEAD) ==="
python -u -m experiments.exact_inversion.constrained_replay --sets confident hard1_diff repeated \
    --r 64 --N 8 --ks 62 --seed 1 --out $OUT/step121_probe_${LSB_JOBID}.jsonl
echo "=== DONE $(date) ==="
