#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -W 8:00
#BSUB -o scripts/wexac_logs/gb_phase1_%J.out
#BSUB -e scripts/wexac_logs/gb_phase1_%J.err
#BSUB -J gb_phase1

# =====================================================================
# GB-Phase 1 — Gradient Bridge decoder f_φ:(A,B) -> ∇_W L (experiment_plan.md Part C).
# Train on a PUBLIC MNIST proxy (single-step LoRA measurements). Milestone: >0.9
# held-out cosine — reported HONESTLY against the col(B0) measurement ceiling.
#
#   Stage 0  layer 2  : output-layer bridge. Near-analytic (out=1 -> col(B0)=R^1),
#                       so expect >0.99 — this VALIDATES THE PIPELINE, weak evidence.
#   Stage 1  layer 1  : hidden-layer bridge (the REAL milestone). Low-rank head.
#                       Sweep rank r in {8,32,64}: does higher rank lift the ceiling?
#
# The honest read is full-cosine vs projection-cosine. Beating the ceiling = the
# decoder hallucinates the out-of-subspace gradient from the proxy prior.
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -c "import torch; print(f'CUDA={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"
echo "=== START $(date) on $(hostname) ==="

# Sized to sit comfortably under ~1 GPU-hr for this first cut (well under the
# 2-GPU-hr gate); scale n_pairs/epochs/ranks up once the shape is confirmed.
NP=12000

# ---------------------------------------------------------------------
# Stage 0 — layer 2 (output) pipeline validation. Small, fast, near-analytic.
# ---------------------------------------------------------------------
echo ""; echo "########## STAGE 0: layer 2 bridge (pipeline validation) ##########"; date
python -m experiments.gradient_bridge.generate_pairs \
    --layer 2 --rank 8 --n_pairs $NP --device cuda \
    --save results/gb_pairs_L2_r8.pth
if [ $? -ne 0 ]; then echo "FATAL: pair generation (L2) failed. Aborting."; exit 1; fi
python -m experiments.gradient_bridge.train_decoder \
    --bank results/gb_pairs_L2_r8.pth --epochs 120 --device cuda

# ---------------------------------------------------------------------
# Stage 1 — layer 1 (hidden) — the real milestone. Rank knob to test the
# subspace ceiling: higher r -> col(B0) captures more gradient energy.
# ---------------------------------------------------------------------
echo ""; echo "########## STAGE 1: layer 1 bridge (real milestone) ##########"; date
for R in 8 32 64; do
    echo ""; echo "=== layer 1, rank=$R ==="; date
    python -m experiments.gradient_bridge.generate_pairs \
        --layer 1 --rank $R --n_pairs $NP --device cuda \
        --save results/gb_pairs_L1_r${R}.pth
    python -m experiments.gradient_bridge.train_decoder \
        --bank results/gb_pairs_L1_r${R}.pth --epochs 100 \
        --out_mode lowrank --out_rank 16 --batch 128 --device cuda
done

echo ""; echo "=== ALL STAGES COMPLETE $(date) ==="
