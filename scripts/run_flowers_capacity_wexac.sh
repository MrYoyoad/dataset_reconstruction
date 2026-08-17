#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/flowers_capacity_%J.out
#BSUB -e scripts/wexac_logs/flowers_capacity_%J.err
#BSUB -J flowers_capacity

# The TRUE high-k capacity test (theory Section 5) on the flowers-NATIVE MLP (D=3072, RGB 32x32,
# theta_0 trained on flowers). Flowers' intrinsic dim k >> MNIST, so kN approaches rho(m+d) at much
# SMALLER N than on MNIST -> the capacity ceiling actually binds here. Fixed activation gelu (so the
# feature ceiling is held ~constant); vary N and rank:
#   Arm 1 (N-collapse): r=8, npc in {1,2,4,8,16,32}  -> N = 2..64
#   Arm 2 (rank at N=8): r in {1,2,4,16}, npc=4
# Compare the N-collapse + rank-dependence to the MNIST equivalents (already on disk). Rescore for
# control margin + retrieval afterward, and estimate k (data effective rank) MNIST vs flowers.
# Distinct dataset tag (flowers32) + gelu-only -> no collision with the peer's flowers activation sweep.

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"

# ---------- STAGE 0: validate the flowers32-native path on the GPU node ----------
echo ""; echo "########## STAGE 0: flowers32 loader + native model ##########"
python - <<'PY' || { echo "STAGE 0 FAILED"; exit 1; }
from experiments.data_utils import get_finetuning_data
from experiments.configs import DATASET_SPECS
x,y,c,_ = get_finetuning_data(1, seed=42, dataset='flowers32')
assert tuple(x.shape) == (2, 3, 32, 32), x.shape
print("flowers32 loader OK", tuple(x.shape), "classes", c, "| spec", DATASET_SPECS['flowers32'])
PY
echo "STAGE 0 PASSED"

run () {  # $1=rank $2=n_per_class
    echo ""; echo "########## flowers32 gelu rank=$1 npc=$2 (N=$((2*$2))) T=1 ##########"; date
    python -u -m experiments.run_experiment_b \
        --dataset flowers32 --n_steps 1 --rank "$1" --n_per_class "$2" --seed 42 --lr 0.01 \
        --verify_weight 5.0 --finetune_activation gelu \
        --no_baseline --save_results --skip_if_exists --device cuda
}

# Arm 1 — N-collapse at r=8 (small N first, so a preemption still delivers the informative low-N cells)
for NPC in 1 2 4 8 16 32; do run 8 "$NPC"; done
# Arm 2 — rank-dependence at N=8
for R in 1 2 4 16; do run "$R" 4; done

echo "=== ALL DONE $(date) ==="
