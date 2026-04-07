#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -W 24:00
#BSUB -o wexac_logs/overnight_s2_%J.out
#BSUB -e wexac_logs/overnight_s2_%J.err
#BSUB -J overnight_s2

# Sprint 2 cleanup: run all remaining low-priority experiments overnight
# 1. Fix r=16/32 free-c convergence (Adam for c + restarts)
# 2. Multi-seed comparison: free-c vs oracle (50 seeds)
# 3. Multi-seed validation of LeakyReLU (30 seeds)
# 4. Per-image SSIM (already computed, just needs different seeds to validate)

set -e

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
mkdir -p results figures wexac_logs

echo "================================================================"
echo "OVERNIGHT Sprint 2 Cleanup — $(date)"
echo "Host: $(hostname)"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
echo "================================================================"

###############################################################################
# 1. Fix r=16/32 free-c convergence: Adam for c, 3 random restarts
###############################################################################
echo ""
echo "=== [1/4] Fix r=16/32 free-c: Adam optimizer for coefficients ==="
echo "Date: $(date)"

for RANK in 16 32; do
    for INIT_SEED in 0 1 2; do
        echo "  rank=$RANK, init_seed=$INIT_SEED, coeff_optimizer=adam"
        python -u -m experiments.run_experiment_b \
            --n_steps 1 \
            --rank $RANK \
            --free_coefficients \
            --consistency_weight 1.0 \
            --coeff_optimizer adam \
            --coeff_lr 0.001 \
            --optimizer lbfgs \
            --relu_alpha 10000 \
            --extraction_epochs 5000 \
            --seed 42 \
            --save_results \
            --device cuda || echo "  FAILED: rank=$RANK init_seed=$INIT_SEED"
    done

    # Also try SGD + LeakyReLU (B3a winner) for r=16/32
    echo "  rank=$RANK, SGD + LeakyReLU"
    python -u -m experiments.run_experiment_b \
        --n_steps 1 \
        --rank $RANK \
        --free_coefficients \
        --consistency_weight 1.0 \
        --coeff_optimizer sgd \
        --coeff_lr 0.01 \
        --optimizer sgd \
        --finetune_activation leaky_relu \
        --extraction_epochs 5000 \
        --seed 42 \
        --save_results \
        --device cuda || echo "  FAILED: rank=$RANK SGD+LeakyReLU"
done

echo "=== [1/4] Done — $(date) ==="

###############################################################################
# 2. Multi-seed: free-c vs oracle comparison (50 seeds, T=1, r=8)
###############################################################################
echo ""
echo "=== [2/4] Multi-seed free-c vs oracle (50 seeds, r=8) ==="
echo "Date: $(date)"

RESULTS_FILE="results/multiseed_freec_vs_oracle_$(date +%Y%m%d_%H%M%S).csv"
echo "seed,mode,rank,ssim,dssim,coeff_error" > "$RESULTS_FILE"

for SEED in $(seq 0 49); do
    echo "  seed=$SEED — oracle"
    ORACLE_OUT=$(python -u -m experiments.run_experiment_b \
        --n_steps 1 \
        --rank 8 \
        --optimizer sgd \
        --finetune_activation leaky_relu \
        --extraction_epochs 5000 \
        --seed $SEED \
        --device cuda 2>&1) || true
    ORACLE_SSIM=$(echo "$ORACLE_OUT" | grep "LoRA rank=8" | grep -oP "ssim': ([0-9.naNeE+-]+)" | head -1 | grep -oP "[0-9.naNeE+-]+" || echo "nan")
    echo "$SEED,oracle,8,$ORACLE_SSIM,," >> "$RESULTS_FILE"

    echo "  seed=$SEED — free-c"
    FREEC_OUT=$(python -u -m experiments.run_experiment_b \
        --n_steps 1 \
        --rank 8 \
        --free_coefficients \
        --consistency_weight 1.0 \
        --coeff_optimizer sgd \
        --coeff_lr 0.01 \
        --optimizer sgd \
        --finetune_activation leaky_relu \
        --extraction_epochs 5000 \
        --seed $SEED \
        --device cuda 2>&1) || true
    FREEC_SSIM=$(echo "$FREEC_OUT" | grep "LoRA rank=8" | grep -oP "ssim': ([0-9.naNeE+-]+)" | head -1 | grep -oP "[0-9.naNeE+-]+" || echo "nan")
    FREEC_CERR=$(echo "$FREEC_OUT" | grep "LoRA coeff error" | grep -oP "[0-9.]+" | head -1 || echo "nan")
    echo "$SEED,free_c,8,$FREEC_SSIM,,$FREEC_CERR" >> "$RESULTS_FILE"
done

echo "Saved to $RESULTS_FILE"
echo "=== [2/4] Done — $(date) ==="

###############################################################################
# 3. Multi-seed LeakyReLU validation (30 seeds, T=1 and T=10, r=8 and r=32)
###############################################################################
echo ""
echo "=== [3/4] Multi-seed LeakyReLU validation (30 seeds) ==="
echo "Date: $(date)"

RESULTS_FILE2="results/multiseed_leakyrelu_validation_$(date +%Y%m%d_%H%M%S).csv"
echo "seed,n_steps,rank,lora_ssim,control_ssim" > "$RESULTS_FILE2"

for SEED in $(seq 0 29); do
    for T in 1 10; do
        for RANK in 8 32; do
            echo "  seed=$SEED, T=$T, r=$RANK"
            OUT=$(python -u -m experiments.run_experiment_b \
                --n_steps $T \
                --rank $RANK \
                --free_coefficients \
                --consistency_weight 1.0 \
                --coeff_optimizer sgd \
                --coeff_lr 0.01 \
                --optimizer sgd \
                --finetune_activation leaky_relu \
                --extraction_epochs 5000 \
                --seed $SEED \
                --device cuda 2>&1) || true
            LORA_SSIM=$(echo "$OUT" | grep "LoRA rank=$RANK" | grep -oP "ssim': ([0-9.naNeE+-]+)" | head -1 | grep -oP "[0-9.naNeE+-]+" || echo "nan")
            CTRL_SSIM=$(echo "$OUT" | grep "Control:" | grep -oP "ssim': ([0-9.naNeE+-]+)" | head -1 | grep -oP "[0-9.naNeE+-]+" || echo "nan")
            echo "$SEED,$T,$RANK,$LORA_SSIM,$CTRL_SSIM" >> "$RESULTS_FILE2"
        done
    done
done

echo "Saved to $RESULTS_FILE2"
echo "=== [3/4] Done — $(date) ==="

###############################################################################
# 4. Per-image SSIM — save image tensors for 10 seeds to inspect individual
#    sample reconstruction quality (not just mean SSIM)
###############################################################################
echo ""
echo "=== [4/4] Per-image SSIM: save tensors for 10 seeds ==="
echo "Date: $(date)"

for SEED in $(seq 0 9); do
    echo "  seed=$SEED — saving tensors"
    python -u -m experiments.run_experiment_b \
        --n_steps 1 \
        --rank 8 \
        --free_coefficients \
        --consistency_weight 1.0 \
        --coeff_optimizer sgd \
        --coeff_lr 0.01 \
        --optimizer sgd \
        --finetune_activation leaky_relu \
        --extraction_epochs 5000 \
        --seed $SEED \
        --save_results \
        --device cuda || echo "  FAILED: seed=$SEED"
done

echo "=== [4/4] Done — $(date) ==="

echo ""
echo "================================================================"
echo "ALL OVERNIGHT EXPERIMENTS COMPLETE — $(date)"
echo "================================================================"
