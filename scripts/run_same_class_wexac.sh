#!/bin/bash
# SAME-CLASS cell (required by b9): a dataset-matched pool leaves a zero false-positive rate consistent with the
# certificate discriminating CLASS rather than MEMBERSHIP. Non-members are now the member's OWN class -- can it
# tell this rose from another rose. Pre-registered: if the false-positive rate rises materially here, the mixed
# class rate was partly class separation and the claim narrows accordingly.
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START same-class cell $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
python -u -m experiments.exact_inversion.live_regime --model resnet18 --stage 4 --r 64 \
    --draws 12 --T 200 --lr 0.05 --optimiser sgd --seed 1 --same-class \
    --out $OUT/step118_sameclass_${LSB_JOBID}.jsonl
echo "=== DONE $(date) ==="
