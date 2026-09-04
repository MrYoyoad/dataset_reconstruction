#!/bin/bash
# =====================================================================
# Does the recipe-free certificate exist on a REAL transformer? Token-span saturation on pretrained ViT-B/16.
# Today's conv result: a shared kernel records one vector per POSITION per image, and the certificate is vacuous
# wherever those fill the layer's input dimension. A transformer shares every linear inside a block across TOKENS
# in exactly the same way -- 8 images x 197 tokens = 1576 vectors into d = 768 for qkv/proj/fc1.
# PRE-REGISTERED, both ways:
#   SATURATED  the token span reaches d on qkv/proj/fc1 -> the certificate is the zero matrix at every adapter
#              rank and the recipe-free channel does not exist on those modules. fc2 (d = 3072 > 1576) would then
#              be the only survivor, which is a sharply testable and very specific prediction.
#   REDUNDANT  trained transformers have famously redundant token activations, so the span may sit far below both
#              N.tokens and d -- then the margin is set by the number of DISTINCT token directions, the channel
#              survives, and the crossover N (largest private batch still leaving a margin) is the number to
#              report to a defender.
# Measured on the FROZEN pretrained weights at real photographs (flowers-102 at native resolution, NOT upsampled
# CIFAR, which would manufacture token redundancy and bias the answer optimistic). No recipe, no release, no solve.
#   bsub -q long-gpu -gpu "num=1" -R "rusage[mem=32768] select[ngpus>0]" -J ei99_vitspan \
#        -o scripts/wexac_logs/ei99_vitspan_%J.out -e scripts/wexac_logs/ei99_vitspan_%J.err \
#        bash scripts/run_vit_token_span_wexac.sh
# =====================================================================
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START vit token span $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
python -u -m experiments.exact_inversion.vit_token_span \
    --model vit_base_patch16_224.augreg2_in21k_ft_in1k \
    --data-root dataset_reconstruction/data/flowers-102/jpg \
    --Ns 1 2 4 8 16 --ranks 8 16 64 256 \
    --out $OUT/step99_vitspan_${LSB_JOBID}.jsonl
echo "=== also the small DINO model, a different pretraining objective ==="
python -u -m experiments.exact_inversion.vit_token_span \
    --model vit_small_patch16_224.dino \
    --data-root dataset_reconstruction/data/flowers-102/jpg \
    --Ns 1 2 4 8 16 --ranks 8 16 64 256 \
    --out $OUT/step99_vitspan_${LSB_JOBID}.jsonl
echo "=== DONE $(date) ==="
