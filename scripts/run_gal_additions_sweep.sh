#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/gal_additions_%J.out
#BSUB -e scripts/wexac_logs/gal_additions_%J.err
#BSUB -J gal_additions

# =====================================================================
# The meeting Additions Gal asked for (notes/experiment_plan.md Part A)
# Priority-ordered so partial completion still delivers the top ask.
#
#   Stage 0  smoke  : verify the new smooth activations run at all
#   Stage 1  ADD-2  : smooth-activation sweep (GELU = top priority)
#   Stage 2  ADD-1  : more LoRA samples / breadth + multi-seed
#   Stage 3  losses : l2 vs cosine (SimuDy found cosine >> Euclidean)
#
# NOT covered here (needs code first, see notes/experiment_plan.md):
#   ADD-3 anchor alpha-sweep  -> no --anchor_alpha flag exists yet
#   Gradient Bridge GB-Phase 1 -> no decoder code exists at all
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -c "import torch; print(f'CUDA={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"
echo "=== START $(date) on $(hostname) ==="

RANK=8

# ---------------------------------------------------------------------
# Stage 0 — smoke the three NEW activations. Abort early if broken,
# rather than burning the whole night on a crash loop.
# ---------------------------------------------------------------------
echo ""; echo "########## STAGE 0: smoke test new activations ##########"
for ACT in gelu silu softplus; do
    echo "--- smoke: $ACT ---"
    python -m experiments.run_experiment_b \
        --n_steps 1 --rank $RANK --finetune_activation $ACT \
        --extraction_epochs 5 --no_baseline --seed 42 --device cuda
    if [ $? -ne 0 ]; then
        echo "FATAL: activation '$ACT' failed the smoke test. Aborting sweep."
        exit 1
    fi
done
echo "Stage 0 PASSED — all new activations run."

# ---------------------------------------------------------------------
# Stage 1 — ADDITION 2: smooth activations (GELU top priority).
# Ordered gelu first so the headline number lands earliest.
# Deliverable: feature-stability / SSIM vs T, one line per activation.
# ---------------------------------------------------------------------
echo ""; echo "########## STAGE 1: ADD-2 smooth-activation sweep ##########"
for ACT in gelu silu softplus leaky_relu relu; do
    for T in 1 5 20; do
        echo ""; echo "=== ADD2 act=$ACT T=$T rank=$RANK ==="; date
        python -m experiments.run_experiment_b \
            --n_steps $T --rank $RANK --finetune_activation $ACT \
            --no_baseline --seed 42 --save_results --device cuda
    done
done

# ---------------------------------------------------------------------
# Stage 2 — ADDITION 1: more samples + multi-seed breadth.
# Deliverable: SSIM-vs-N, reported as a distribution not a best run.
# ---------------------------------------------------------------------
echo ""; echo "########## STAGE 2: ADD-1 more samples / breadth ##########"
for ACT in gelu leaky_relu; do
    for NPC in 1 2 3 4; do
        for SEED in 42 43 44; do
            echo ""; echo "=== ADD1 act=$ACT n_per_class=$NPC seed=$SEED ==="; date
            python -m experiments.run_experiment_b \
                --n_steps 1 --rank $RANK --finetune_activation $ACT \
                --n_per_class $NPC --no_baseline --seed $SEED \
                --save_results --device cuda
        done
    done
done

# ---------------------------------------------------------------------
# Stage 3 — loss functions: l2 vs cosine.
# Extra relevance: SimuDy (ICLR'25) reports cosine >> Euclidean.
# ---------------------------------------------------------------------
echo ""; echo "########## STAGE 3: loss ablation l2 vs cosine ##########"
for LOSS in l2 cosine; do
    for T in 1 10; do
        echo ""; echo "=== LOSS act=gelu loss=$LOSS T=$T ==="; date
        python -m experiments.run_experiment_b \
            --n_steps $T --rank $RANK --finetune_activation gelu \
            --loss_type $LOSS --no_baseline --seed 42 \
            --save_results --device cuda
    done
done

echo ""; echo "=== ALL STAGES COMPLETE $(date) ==="
