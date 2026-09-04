#!/bin/bash
# Conv half of the head-vs-shared table: pretrained ResNet-18/50 at real photographs. Conv layers record one
# vector per POSITION per image; the fc head records ONE globally-pooled vector per image. Same contrast as the
# transformer table, on the other architecture family. (The first attempt, job 279182, died on an undefined
# variable in the resnet branch after the ViT arms had already completed -- those rows stand.)
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/exact_inversion; mkdir -p $OUT
DATA=dataset_reconstruction/data/flowers-102/jpg
echo "=== START resnet span $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
for M in resnet18 resnet50; do
  echo "=== $M ==="
  python -u -m experiments.exact_inversion.vit_token_span --arch resnet --model $M --data-root $DATA \
      --Ns 1 2 4 8 --ranks 8 16 64 256 --out $OUT/step100_resnetspan_${LSB_JOBID}.jsonl
done
echo "=== DONE $(date) ==="
