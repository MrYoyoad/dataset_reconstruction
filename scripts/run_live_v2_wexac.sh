#!/bin/bash
# Live-regime v2, with everything the lanes required after the first result:
#  - SAME-CLASS non-members (REQUIRED, not optional): a dataset-matched pool leaves a zero false-positive rate
#    consistent with the certificate discriminating CLASS rather than MEMBERSHIP, and a membership claim built on
#    class discrimination is not a membership claim. Pre-registered: if the FPR rises materially here, the original
#    rate was partly class separation and the claim narrows accordingly.
#  - NEAR-DUPLICATES ordered by MEASURED feature distance, with the prediction that pass/fail falls at a SINGLE
#    boundary in that order -- one monotone prediction that can fail, rather than six binaries which are six
#    chances to find one that works. Each transformation also gets its own PAIRED negative.
#  - Draw 1 evaluable: the structural n-1 exclusion is fixed rather than reported.
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START live v2: near-duplicates by measured distance, paired negatives $(date) git=$(git rev-parse --short HEAD) ==="
python -u -m experiments.exact_inversion.live_regime --model resnet18 --stage 4 --r 64 \
    --draws 12 --nonmembers 1000 --T 200 --lr 0.05 --optimiser sgd --seed 1 --near-dupes \
    --out $OUT/step117_neardupe_${LSB_JOBID}.jsonl
echo "=== SAME-CLASS cell (required): can it tell this rose from another rose? ==="
python -u -m experiments.exact_inversion.live_regime --model resnet18 --stage 4 --r 64 \
    --draws 12 --T 200 --lr 0.05 --optimiser sgd --seed 1 --same-class \
    --out $OUT/step118_sameclass_${LSB_JOBID}.jsonl
echo "=== DONE $(date) ==="
