#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/capacity_ranksweep_%J.out
#BSUB -e scripts/wexac_logs/capacity_ranksweep_%J.err
#BSUB -J capacity_ranksweep

# Follow-up B: capacity / feature-saturation test (theory Section 5). On the small MNIST-MLP,
# m+d~1784 >> kN for small N, so capacity rarely binds and the testable prediction is that leakage
# SATURATES at r ~ rank(M) (the feature ceiling): gelu (rank(M)~2.9) plateaus earlier than relu
# (rank(M)~6.4). The large-N arm probes the capacity onset (leakage collapses as kN -> rho(m+d)).
# Rescore afterward for control margin + retrieval vs r and vs N. (True high-k capacity = flowers-native.)
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"

run () {  # $1=activation $2=rank $3=n_per_class
    echo ""; echo "########## act=$1 rank=$2 npc=$3 (N=$((2*$3))) T=1 ##########"; date
    python -u -m experiments.run_experiment_b \
        --n_steps 1 --rank "$2" --n_per_class "$3" --seed 42 --lr 0.01 --verify_weight 5.0 \
        --finetune_activation "$1" \
        --no_baseline --save_results --skip_if_exists --device cuda
}

# rank sweep at N=10 (feature-ceiling saturation): gelu (early plateau) vs relu (late plateau)
for R in 1 2 4 8 16 32 64; do run gelu "$R" 5; done
for R in 1 2 4 8 16 32 64; do run relu "$R" 5; done
# large-N arm at r=8 (capacity onset): N = 16, 32, 64
for NPC in 8 16 32; do run gelu 8 "$NPC"; done

echo "=== ALL DONE $(date) ==="
