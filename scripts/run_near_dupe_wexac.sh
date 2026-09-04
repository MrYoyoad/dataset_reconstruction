#!/bin/bash
# NEAR-DUPLICATE SPECIFICITY (yoado-cd: "the first question a supervisor will ask").
# Does a crop, resize, flip, brightness change, blur or requantisation of the private photograph still read as a
# member? Both answers are results and they are DIFFERENT claims:
#   passes -> "this image or anything close to it": the stronger privacy statement, the weaker specificity one
#   fails  -> "this exact image": narrower, sharper, and the difference between detecting a photograph and
#             detecting a FILE.
# Same cell as job 307760 so the numbers are directly comparable; SGD only, since the Adam arm has no certificate.
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START near-duplicate specificity $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
python -u -m experiments.exact_inversion.live_regime --model resnet18 --stage 4 --r 64 \
    --draws 12 --nonmembers 1000 --T 200 --lr 0.05 --optimiser sgd --seed 1 --near-dupes \
    --out $OUT/step117_neardupe_${LSB_JOBID}.jsonl
echo "=== DONE $(date) ==="
