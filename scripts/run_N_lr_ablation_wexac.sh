#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/N_lr_ablation_%J.out
#BSUB -e scripts/wexac_logs/N_lr_ablation_%J.err
#BSUB -J N_lr_ablation

# =====================================================================
# Clear N x lr ablation for GELU (the deployment-standard) and SOFTPLUS (the Step-1 winner).
# Maps the (N, lr) -> weight_change -> reconstruction/leakage surface for each, so we can state
# exactly how sample count and step size trade off against recovery for the two key activations.
# LoRA r=8, T=1, seed 42. weight_change is saved per run, so the surface can be read at matched
# weight_change post-hoc (rescore with recompute_metrics.py / retrieval_metric.py).
#
#   activations: softplus (winner first), gelu
#   N (n_per_class): 1, 2, 4, 8  -> N = 2, 4, 8, 16
#   lr: 0.005, 0.01, 0.02, 0.05, 0.1
# = 2 x 4 x 5 = 40 configs (~3.3 GPU-hr). Ordered winner-first, small-N-first so a preemption
# still delivers the most informative cells. --skip_if_exists reuses anything already on disk.
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"

NPCS="1 2 4 8"
LRS="0.005 0.01 0.02 0.05 0.1"

for ACT in softplus gelu; do
    for NPC in $NPCS; do
        for LR in $LRS; do
            echo ""; echo "########## ablation act=$ACT npc=$NPC (N=$((2*NPC))) lr=$LR T=1 r=8 ##########"; date
            python -u -m experiments.run_experiment_b \
                --n_steps 1 --rank 8 --seed 42 --lr "$LR" --n_per_class "$NPC" \
                --finetune_activation "$ACT" \
                --no_baseline --save_results --skip_if_exists --device cuda
        done
    done
done

echo ""; echo "=== ALL STAGES COMPLETE $(date) ==="
