#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1:gmem=4000"
#BSUB -o scripts/wexac_logs/flowers32_freec_activation_%J.out
#BSUB -e scripts/wexac_logs/flowers32_freec_activation_%J.err
#BSUB -J flowers32_freec_activation
# REALISTIC free-c activation spectrum on flowers32: fine-tune with each activation (shapes delta-W),
# extract with fixed ReLU-like (a10000) + SGD + consistency=1 + restarts. Does swish-family-dead hold?
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
for ACT in softplus silu gelu mish gelu_tanh elu celu tanh sigmoid selu hardswish leaky_relu relu; do
  echo ""; echo "########## flowers32 free-c act=$ACT r=8 N=2 ##########"; date
  python -u -m experiments.run_experiment_b --dataset flowers32 --n_steps 1 --rank 8 --seed 42 --lr 0.01 \
    --finetune_activation "$ACT" --free_coefficients --optimizer sgd --extract_activation modified_relu --relu_alpha 10000 --consistency_weight 1.0 --n_restarts 3 --extraction_epochs 40000 --no_baseline --save_results --skip_if_exists --device cuda
done
echo "=== DONE $(date) ==="
