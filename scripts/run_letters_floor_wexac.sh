#!/bin/bash
# THE ONE SOLVE that decides how the lead result may be described: the fp32-trained letters cell's achievability
# floor was never measured, so `frac_starts_at_floor = 0.0` was computed against an ABSOLUTE 1e-20 cut that an
# fp32-trained release cannot reach by construction. If the cell's own floor is ~1e-14 then argmin_objective
# 6.05e-14 IS at the floor and the headline stands as an attack; if the floor is genuinely lower, the images came
# out without the residual verifying them and the claim drops to an identifiability statement.
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START letters achievability floor $(date) git=$(git rev-parse --short HEAD) ==="
python -u -m experiments.exact_inversion.train_precision --cells letters_a:32 --dtypes fp32 \
    --partb-cells letters_a:32 --partb-dtypes fp32 --partb-tol 1e-12 \
    --r 64 --N 8 --T 400 --lr 0.01 --seed 1 --random-starts 300 --iters 300 \
    --out $OUT/step125_lettersfloor_${LSB_JOBID}.jsonl
echo "=== DONE $(date) ==="
