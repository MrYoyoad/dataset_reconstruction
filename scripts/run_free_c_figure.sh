#!/bin/bash
#BSUB -q short-gpu
#BSUB -gpu "num=1"
#BSUB -R "rusage[mem=16000]"
#BSUB -o /home/projects/galvardi/yoado/scripts/wexac_logs/free_c_figure_%J.out
#BSUB -e /home/projects/galvardi/yoado/scripts/wexac_logs/free_c_figure_%J.err
#BSUB -J free_c_fig

source /home/projects/galvardi/yoado/.conda/envs/rec/bin/activate
cd /home/projects/galvardi/yoado

# Run free-c LoRA r=8 with the winning config (SGD+LeakyReLU)
# Multiple seeds to pick the best visual
for SEED in 42 0 1 2 3; do
    echo "=== Seed $SEED ==="
    python3 -m experiments.run_experiment_b \
        --seed $SEED \
        --rank 8 \
        --n_steps 1 \
        --optimizer sgd \
        --lr 0.01 \
        --extraction_epochs 5000 \
        --finetune_activation leaky_relu \
        --relu_alpha 10000 \
        --free_coefficients \
        --consistency_weight 1.0 \
        --coeff_lr 0.01 \
        --coeff_optimizer sgd \
        --save_results \
        --device cuda
done

echo "=== ALL DONE ==="
