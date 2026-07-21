#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/gal_additions_%J.out
#BSUB -e scripts/wexac_logs/gal_additions_%J.err
#BSUB -J gal_additions

# =====================================================================
# Gal's meeting Additions (notes/experiment_plan.md Part A).
# Priority-ordered: the top ask (Addition 2) runs first, so partial
# completion still delivers it.
#
#   Stage 0  guard  : smoke the new activations AND prove per-config
#                     filenames are unique (v1 of this script silently
#                     overwrote ~35 of 43 runs -- see LESSONS_LEARNED)
#   Stage 1  ADD-2a : activation x LR at T=1  <- LR CALIBRATION.
#                     v1 ran GELU at the ReLU-tuned lr=0.01 and got
#                     SSIM 0.041 (control 0.020) with weight_change=0.039,
#                     i.e. the net barely moved. That is a hyper-parameter
#                     artifact, NOT evidence against "smoother is better".
#                     Compare activations at MATCHED weight_change.
#   Stage 2  ADD-2b : activation x T at two LRs (survival vs smoothness)
#   Stage 3  ADD-1  : more LoRA samples / breadth + multi-seed
#   Stage 4  losses : l2 vs cosine (SimuDy reports cosine >> Euclidean)
#
# NOT covered (need code first): ADD-3 anchor alpha-sweep (no
# --anchor_alpha flag), Gradient Bridge GB-Phase 1 (no decoder code).
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -c "import torch; print(f'CUDA={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"
echo "=== START $(date) on $(hostname) ==="

RANK=8

run () {  # run <label> <extra args...>
    local label="$1"; shift
    echo ""; echo "=== ${label} ==="; date
    python -m experiments.run_experiment_b \
        --rank $RANK --no_baseline --save_results --device cuda "$@"
}

# ---------------------------------------------------------------------
# Stage 0 — guard. Smoke the new activations, then PROVE that two
# different activations produce two different .pth files. Aborting here
# costs minutes; not aborting cost the entire previous run.
# ---------------------------------------------------------------------
echo ""; echo "########## STAGE 0: smoke + filename-collision guard ##########"
GUARD=/tmp/gal_guard_$$
mkdir -p $GUARD
for ACT in gelu silu softplus; do
    echo "--- smoke: $ACT ---"
    python -m experiments.run_experiment_b \
        --n_steps 1 --rank $RANK --finetune_activation $ACT \
        --extraction_epochs 5 --no_baseline --save_results --device cuda
    if [ $? -ne 0 ]; then
        echo "FATAL: activation '$ACT' failed the smoke test. Aborting."
        exit 1
    fi
done

N_UNIQUE=$(ls results/exp_b_T1_r${RANK}_s42_a149_{gelu,silu,softplus}.pth 2>/dev/null | wc -l)
if [ "$N_UNIQUE" -ne 3 ]; then
    echo "FATAL: expected 3 distinct per-activation .pth files, found $N_UNIQUE."
    echo "       Filenames are still colliding -- fix base_name in"
    echo "       experiments/run_experiment_b.py before wasting a night."
    ls -la results/exp_b_T1_r${RANK}_s42_a149* 2>/dev/null
    exit 1
fi
echo "Stage 0 PASSED: activations run AND filenames are unique."

# ---------------------------------------------------------------------
# Stage 1 — ADD-2a: LR calibration. The headline comparison.
# Deliverable: for each activation, SSIM and weight_change vs LR. Compare
# activations at comparable weight_change, not at fixed LR.
# ---------------------------------------------------------------------
echo ""; echo "########## STAGE 1: ADD-2a activation x LR (T=1) ##########"
for ACT in gelu silu softplus leaky_relu relu; do
    for LR in 0.01 0.03 0.1 0.3; do
        run "ADD2a act=$ACT lr=$LR T=1" \
            --n_steps 1 --finetune_activation $ACT --lr $LR --seed 42
    done
done

# ---------------------------------------------------------------------
# Stage 2 — ADD-2b: smoothness vs multi-step survival.
# Deliverable: feature-stability / SSIM vs T, one line per activation.
# ---------------------------------------------------------------------
echo ""; echo "########## STAGE 2: ADD-2b activation x T ##########"
for ACT in gelu silu softplus leaky_relu relu; do
    for T in 5 20; do
        for LR in 0.01 0.1; do
            run "ADD2b act=$ACT T=$T lr=$LR" \
                --n_steps $T --finetune_activation $ACT --lr $LR --seed 42
        done
    done
done

# ---------------------------------------------------------------------
# Stage 3 — ADD-1: more samples + multi-seed breadth.
# leaky_relu @ 0.01 is the established-good baseline; gelu @ 0.1 is the
# best guess pending Stage 1. Deliverable: SSIM-vs-N as a distribution.
# ---------------------------------------------------------------------
echo ""; echo "########## STAGE 3: ADD-1 more samples / breadth ##########"
for NPC in 1 2 3 4; do
    for SEED in 42 43 44; do
        run "ADD1 act=leaky_relu npc=$NPC seed=$SEED" \
            --n_steps 1 --finetune_activation leaky_relu --lr 0.01 \
            --n_per_class $NPC --seed $SEED
        run "ADD1 act=gelu npc=$NPC seed=$SEED" \
            --n_steps 1 --finetune_activation gelu --lr 0.1 \
            --n_per_class $NPC --seed $SEED
    done
done

# ---------------------------------------------------------------------
# Stage 4 — loss functions: l2 vs cosine.
# ---------------------------------------------------------------------
echo ""; echo "########## STAGE 4: loss ablation l2 vs cosine ##########"
for LOSS in l2 cosine; do
    for T in 1 10; do
        run "LOSS act=gelu loss=$LOSS T=$T" \
            --n_steps $T --finetune_activation gelu --lr 0.1 \
            --loss_type $LOSS --seed 42
    done
done

echo ""; echo "=== ALL STAGES COMPLETE $(date) ==="
echo "distinct result files produced:"; ls results/exp_b_*.pth | wc -l
