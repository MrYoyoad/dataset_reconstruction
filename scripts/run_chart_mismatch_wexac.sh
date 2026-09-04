#!/bin/bash
# =====================================================================
# THE DISTRIBUTIONAL-ACCESS QUESTION ON THE SURFACE WHERE IT IS STILL ALIVE.
# Job 289251 killed it for membership inference, for a reason specific to membership: the attacker HOLDS the
# candidate by definition, so any co-training pool substitutes. In RECONSTRUCTION they do not hold the image --
# recovering it is the task -- so the public pool is the only source of a chart, and the certificate route's own
# framing says the prior's job here is the CHART rather than the start. So the chart IS the attacker's
# distributional assumption, and this varies exactly it while holding release, recipe, private batch and solver
# fixed. All four pools are 28x28 greyscale, so the chart's shape is identical and only its content differs.
#   matched  MNIST train     (the assumption every earlier cell silently made)
#   near     EMNIST letters  (handwritten, same stroke statistics, different classes)
#   far      FashionMNIST    (greyscale objects, different content entirely)
#   gross    uniform noise   (no structure; the floor, and the control for what a chart is worth)
# PRE-REGISTERED: chart fidelity degrades monotonically matched -> gross; recovery degrades with it; the reported
# quantity is the pool at which recovery stops working. If recovery survives on 'far' or 'gross' the attacker needs
# no distributional access on this surface either, the argument is dead everywhere, and that is a clean negative to
# be stated as one. Recovery is from RANDOM starts and scored against the ground truth, never against the release.
#   bsub -q long-gpu -gpu "num=1" -R "rusage[mem=32768] select[ngpus>0]" -J ei108_chartmm ...
# =====================================================================
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START chart mismatch $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
python -u -m experiments.exact_inversion.chart_mismatch --pools matched near far gross \
    --ks 12 14 16 --N 8 --r 64 --T 400 --lr 0.01 --starts 200 --iters 300 --seed 1 \
    --out $OUT/step108_chartmm_${LSB_JOBID}.jsonl
echo "=== DONE $(date) ==="
