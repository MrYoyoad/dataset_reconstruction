#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/activation_spectrum_N10_%J.out
#BSUB -e scripts/wexac_logs/activation_spectrum_N10_%J.err
#BSUB -J activation_spectrum_N10

# =====================================================================
# Step 2a at LARGER N (N=10, n_per_class=5) — does softplus's activation win survive where the
# mean-baseline is meaningful (0.564 at N=10, vs 0.76 at N=2)? Same 13-activation smooth->kinked
# spectrum as the N=2 job (479367), LoRA r=8, T=1, seed 42, --verify_weight 5.0 to curb clipping
# at larger N. Leakage is expected WEAKER at N=10 (superposition; QW3 LoRA margins were +0.006-0.008)
# -- the question here is the RELATIVE activation ranking, not absolute recovery. ~26 configs.
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"

ACTS="sigmoid tanh gelu silu softplus mish gelu_tanh elu celu selu hardswish leaky_relu relu"
LRS="0.01 0.02"

for ACT in $ACTS; do
    for LR in $LRS; do
        echo ""; echo "########## N=10 act=$ACT lr=$LR T=1 r=8 ##########"; date
        python -u -m experiments.run_experiment_b \
            --n_steps 1 --rank 8 --seed 42 --lr "$LR" --n_per_class 5 --verify_weight 5.0 \
            --finetune_activation "$ACT" \
            --no_baseline --save_results --skip_if_exists --device cuda
    done
done

echo ""; echo "=== ALL STAGES COMPLETE $(date) ==="
