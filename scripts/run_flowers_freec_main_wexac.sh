#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1:gmem=4000"
#BSUB -o scripts/wexac_logs/flowers_freec_main_%J.out
#BSUB -e scripts/wexac_logs/flowers_freec_main_%J.err
#BSUB -J flowers_freec_main

# =====================================================================
# REALISTIC free-coefficient attack (the Haim-et-al. mode) on the flowers-native models.
# VERIFIED recipe (reproduces the Sprint-2 known-good ~0.59 on MNIST; flowers32 r8 = 0.652 vs
# oracle 0.688): SGD extraction + relu_alpha=10000 (~ReLU) + consistency_weight=1.0 + restarts.
# Default ReLU fine-tune (free-c extraction is ReLU-like, so ReLU is the natural attacker choice).
# Axes: rank/leakage curve, N curve, and the Q-A dimension ladder (flowers32 D=3072 vs flowers64 D=12288).
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="

FREEC="--free_coefficients --optimizer sgd --relu_alpha 10000 --consistency_weight 1.0 --n_restarts 5 --extraction_epochs 40000"

# ---------- flowers32 rank/leakage curve (realistic) ----------
for R in 4 8 16 32 64; do
  echo ""; echo "########## flowers32 free-c r=$R N=2 ##########"; date
  python -u -m experiments.run_experiment_b --dataset flowers32 --n_steps 1 --rank "$R" --seed 42 --lr 0.01 \
    $FREEC --save_results --skip_if_exists --device cuda
done

# ---------- flowers32 N curve (realistic; retrieval strengthens with N) ----------
for NPC in 1 2 4 8; do
  VW=""; [ "$NPC" -ge 4 ] && VW="--verify_weight 5.0"
  echo ""; echo "########## flowers32 free-c npc=$NPC r=8 ##########"; date
  python -u -m experiments.run_experiment_b --dataset flowers32 --n_steps 1 --rank 8 --seed 42 --lr 0.01 \
    --n_per_class "$NPC" $VW $FREEC $([ "$NPC" -ge 4 ] && echo --sequential_peel) --no_baseline --save_results --skip_if_exists --device cuda
done

# ---------- Q-A dimension ladder: flowers64 (rich rung) at r=8 ----------
echo ""; echo "########## flowers64 free-c r=8 N=2 (Q-A rich rung) ##########"; date
python -u -m experiments.run_experiment_b --dataset flowers64 --n_steps 1 --rank 8 --seed 42 --lr 0.01 \
  $FREEC --save_results --skip_if_exists --device cuda

echo ""; echo "=== FLOWERS FREE-C MAIN COMPLETE $(date) ==="
