#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1:gmem=4000"
#BSUB -o scripts/wexac_logs/flowers32_freec_anchor_%J.out
#BSUB -e scripts/wexac_logs/flowers32_freec_anchor_%J.err
#BSUB -J flowers32_freec_anchor
# REALISTIC free-c anchor alpha-sweep two-curve on flowers32 (ReLU fine-tune + ReLU extraction).
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
python -u -m experiments.run_anchor_sweep --dataset flowers32 --n_steps 10 --rank 8 --seed 42 \
  --finetune_activation relu --extract_activation modified_relu --relu_alpha 10000 \
  --free_coefficients --optimizer sgd --consistency_weight 1.0 --n_restarts 5 \
  --save --skip_if_exists --device cuda
echo "=== DONE $(date) ==="
