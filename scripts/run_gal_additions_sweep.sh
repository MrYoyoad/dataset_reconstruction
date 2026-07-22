#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/gal_additions_%J.out
#BSUB -e scripts/wexac_logs/gal_additions_%J.err
#BSUB -J gal_additions

# =====================================================================
# Gal's meeting Additions (notes/experiment_plan.md Part A), priority-ordered.
#
#   Stage 0  guard  : smoke new activations + PROVE filenames are unique
#   Stage 1  ADD-2a : activation x LR at T=1  (LR CALIBRATION - compare
#                     activations at matched weight_change, not fixed LR)
#   Stage 2  ADD-2b : softplus beta knob = the continuous smoothness axis
#   Stage 3  ADD-2c : activation x T (multi-step survival vs smoothness)
#   Stage 4  ADD-1  : more LoRA samples / breadth + multi-seed
#   Stage 5  losses : l2 vs cosine (SimuDy reports cosine >> Euclidean)
#
# Every run uses --skip_if_exists: LSF preemption on long-gpu REQUEUES rather
# than resumes, restarting this script from line 1. Skipping completed configs
# makes a restart cost only the unfinished work (LESSONS_LEARNED 2026-07-22).
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -c "import torch; print(f'CUDA={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"
echo "=== START $(date) on $(hostname) ==="

RANK=8
SMOOTH_SET="gelu silu softplus mish gelu_tanh"
CONTROL_SET="elu tanh hardswish leaky_relu relu"
BETA_SET="softplus_b0.5 softplus_b2 softplus_b5 softplus_b10 softplus_b50"

run () {  # run <label> <extra args...>
    local label="$1"; shift
    echo ""; echo "=== ${label} ==="; date
    python -m experiments.run_experiment_b \
        --rank $RANK --no_baseline --save_results --skip_if_exists --device cuda "$@"
}

# ---------------------------------------------------------------------
# Stage 0 — guard: smoke + filename-collision assertion.
# ---------------------------------------------------------------------
echo ""; echo "########## STAGE 0: smoke + filename-collision guard ##########"
# Smoke ONLY new activations that have no real result files yet, so a 5-epoch
# smoke never clobbers an existing full-extraction result (e.g. _gelu.pth).
SMOKE_SET="mish gelu_tanh hardswish softplus_b0.5"
for ACT in $SMOKE_SET; do
    echo "--- smoke: $ACT ---"
    python -m experiments.run_experiment_b \
        --n_steps 1 --rank $RANK --finetune_activation $ACT \
        --extraction_epochs 5 --no_baseline --save_results --device cuda
    if [ $? -ne 0 ]; then
        echo "FATAL: activation '$ACT' failed the smoke test. Aborting."
        exit 1
    fi
done
N_UNIQUE=$(ls results/exp_b_T1_r${RANK}_s42_a149_{mish,gelu_tanh,hardswish,softplus_b0.5}.pth 2>/dev/null | wc -l)
if [ "$N_UNIQUE" -ne 4 ]; then
    echo "FATAL: expected 4 distinct per-activation .pth files, found $N_UNIQUE."
    echo "       Filenames are colliding -- fix build_base_name() before wasting a night."
    exit 1
fi
echo "Stage 0 PASSED: activations run AND filenames are unique."
# smoke files are 5-epoch garbage; remove so the real runs below actually execute.
rm -f results/exp_b_T1_r${RANK}_s42_a149_{mish,gelu_tanh,hardswish,softplus_b0.5}.pth

# ---------------------------------------------------------------------
# Stage 1 — ADD-2a: LR calibration (the headline comparison).
# ---------------------------------------------------------------------
echo ""; echo "########## STAGE 1: ADD-2a activation x LR (T=1) ##########"
for ACT in $SMOOTH_SET $CONTROL_SET; do
    for LR in 0.01 0.03 0.1 0.3; do
        run "ADD2a act=$ACT lr=$LR T=1" \
            --n_steps 1 --finetune_activation $ACT --lr $LR --seed 42
    done
done

# ---------------------------------------------------------------------
# Stage 2 — ADD-2b: the continuous smoothness knob (softplus beta).
# beta -> inf approaches ReLU, so this is a monotonicity test, not a
# categorical comparison.
# ---------------------------------------------------------------------
echo ""; echo "########## STAGE 2: ADD-2b softplus beta knob ##########"
for ACT in $BETA_SET; do
    for LR in 0.01 0.1; do
        run "ADD2b act=$ACT lr=$LR T=1" \
            --n_steps 1 --finetune_activation $ACT --lr $LR --seed 42
    done
done

# ---------------------------------------------------------------------
# Stage 3 — ADD-2c: multi-step survival vs smoothness.
# ---------------------------------------------------------------------
echo ""; echo "########## STAGE 3: ADD-2c activation x T ##########"
for ACT in gelu silu softplus mish leaky_relu relu; do
    for T in 5 20; do
        for LR in 0.01 0.1; do
            run "ADD2c act=$ACT T=$T lr=$LR" \
                --n_steps $T --finetune_activation $ACT --lr $LR --seed 42
        done
    done
done

# ---------------------------------------------------------------------
# Stage 4 — ADD-1: more samples + multi-seed breadth.
# ---------------------------------------------------------------------
echo ""; echo "########## STAGE 4: ADD-1 more samples / breadth ##########"
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
# Stage 5 — loss functions: l2 vs cosine.
# ---------------------------------------------------------------------
echo ""; echo "########## STAGE 5: loss ablation l2 vs cosine ##########"
for LOSS in l2 cosine; do
    for T in 1 10; do
        run "LOSS act=gelu loss=$LOSS T=$T" \
            --n_steps $T --finetune_activation gelu --lr 0.1 \
            --loss_type $LOSS --seed 42
    done
done

echo ""; echo "=== ALL STAGES COMPLETE $(date) ==="
echo "distinct result files: $(ls results/exp_b_*.pth | wc -l)"
echo "Re-score everything with:  python -m experiments.recompute_metrics"
