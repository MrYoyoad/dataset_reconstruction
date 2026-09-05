#!/bin/bash
# THE MEASUREMENT THREE DOCUMENTS CALL UNMEASURED: how many of the recorded letters can an attacker
# VERIFY unaided, per image against that image's own achievability floor, and how does that count grow
# with the number of random starts. The published cell (394731) used a cell-minimum bar that sat below
# seven of the eight per-image floors, so its 3.0% and its "three distinct images" were artefacts.
# Reports n_images_found (experimenter, scored against the private images) beside n_images_verified
# (attacker, certified by the residual alone). Neither is quotable without random_starts beside it.
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START letters verified coverage $(date) git=$(git rev-parse --short HEAD) ==="
STARTS=${STARTS:-3000}
echo "random starts: $STARTS"
python -u -m experiments.exact_inversion.train_precision --cells letters_a:32 --dtypes fp32 \
    --partb-cells letters_a:32 --partb-dtypes fp32 --partb-tol 1e-12 \
    --r 64 --N 8 --T 400 --lr 0.01 --seed 1 --random-starts $STARTS --iters 300 \
    --out $OUT/step126_verified_${LSB_JOBID}.jsonl
echo "=== DONE $(date) ==="
