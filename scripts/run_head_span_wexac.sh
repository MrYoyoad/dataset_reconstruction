#!/bin/bash
# =====================================================================
# THE HEAD IS THE ONLY NON-WEIGHT-SHARED MODULE (yoado-cd, §21 follow-up). Every block linear in a transformer is
# applied at every token and every conv kernel at every position, so one image records 197 (or P) directions. A
# classification head consumes ONE vector per image -- the CLS token, or the globally-pooled feature -- so its
# recorded count should be N, and its certificate should live in exactly the regime all of this project's MLP work
# lives in: r > N, with the head-width cap N' <= m-1 binding.
# PRE-REGISTERED: head span = N exactly; block/conv span = min(N x positions, d). If the head also comes back at
# N x positions, the CLS/pooling reading is wrong and the transformer negative is TOTAL -- say so plainly.
# Four architectures in one table so the contrast is within a single measurement: supervised ViT-B/16, DINO
# ViT-S/16, ResNet-18 and ResNet-50, all pretrained, at real photographs.
#   bsub -q long-gpu -gpu "num=1" -R "rusage[mem=32768] select[ngpus>0]" -J ei100_headspan \
#        -o scripts/wexac_logs/ei100_headspan_%J.out -e scripts/wexac_logs/ei100_headspan_%J.err \
#        bash scripts/run_head_span_wexac.sh
# =====================================================================
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=results/exact_inversion; mkdir -p $OUT
DATA=dataset_reconstruction/data/flowers-102/jpg
echo "=== START head-vs-shared span $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
for M in vit_base_patch16_224.augreg2_in21k_ft_in1k vit_small_patch16_224.dino; do
  echo "=== $M (vit) ==="
  python -u -m experiments.exact_inversion.vit_token_span --arch vit --model $M --data-root $DATA \
      --Ns 1 2 4 8 --ranks 8 16 64 256 --blocks 0 6 11 --out $OUT/step100_headspan_${LSB_JOBID}.jsonl
done
for M in resnet18 resnet50; do
  echo "=== $M (resnet) ==="
  python -u -m experiments.exact_inversion.vit_token_span --arch resnet --model $M --data-root $DATA \
      --Ns 1 2 4 8 --ranks 8 16 64 256 --out $OUT/step100_headspan_${LSB_JOBID}.jsonl
done
echo "=== DONE $(date) ==="
