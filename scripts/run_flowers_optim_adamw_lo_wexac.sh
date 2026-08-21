#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1:gmem=4000"
#BSUB -o scripts/wexac_logs/flowers_optim_adamwlo_%J.out
#BSUB -e scripts/wexac_logs/flowers_optim_adamwlo_%J.err
#BSUB -J flowers_optim_adamwlo

# Supplementary adamw LOW-lr ladder for the matched-wc optimizer comparison. The first ladder
# {0.002,0.005} overshot: adamw@0.002 gives wc=0.115 (3x sgd@0.01's 0.036). adamw makes much larger
# updates per lr, so to MATCH sgd's wc~0.036 we need lr ~= 0.0005-0.001. This brackets it.
# Same validated free-c extraction + --pixel_box --verify_weight 5.0 as the sgd baseline, N=2 r=8 seed 42.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
FREEC="--free_coefficients --optimizer sgd --relu_alpha 10000 --consistency_weight 1.0 --n_restarts 5 --extraction_epochs 40000"
CKPT=dataset_reconstruction/models/weights-flowers32.pth
echo "=== START $(date) on $(hostname) ==="
for LRV in 0.0005 0.001; do
  for LOSS in l2 cosine; do
    LOSSFLAG=""; [ "$LOSS" = cosine ] && LOSSFLAG="--loss_type cosine"
    echo ""; echo "########## flowers32 optim ft=adamw lr=$LRV loss=$LOSS (pbox) ##########"; date
    python -u -m experiments.run_experiment_b \
      --dataset flowers32 --pretrained_path "$CKPT" \
      --n_steps 1 --rank 8 --seed 42 --lr "$LRV" --finetune_optimizer adamw \
      --pixel_box --verify_weight 5.0 $LOSSFLAG \
      $FREEC --no_baseline --save_results --skip_if_exists --device cuda
  done
done
echo ""; echo "=== ADAMW LOW-LR LADDER COMPLETE $(date) ==="
