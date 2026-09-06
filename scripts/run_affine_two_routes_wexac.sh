#!/bin/bash
# Does the affine-chart degeneracy break the REPLAY route, or only the certificate route?
# The nesting {truth} c {rho=0} c {Ch=0} runs one way, so the certificate result (job 311215)
# cannot be inherited. This is the cell that settles it. Pre-registration is in the module docstring.
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START affine two-routes $(date) git=$(git rev-parse --short HEAD) ==="
python -u -m experiments.exact_inversion.affine_chart_two_routes \
    --k 12 --N 8 --r 24 --m 20 --P 64 --T 400 --lr 0.05 --seed 1 \
    --starts ${STARTS:-60} --lm-iters 60 \
    --out $OUT/affine_two_routes_${LSB_JOBID}.jsonl
echo "=== DONE $(date) ==="
