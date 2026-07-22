#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/di_phase0_%J.out
#BSUB -e scripts/wexac_logs/di_phase0_%J.err
#BSUB -J di_phase0

# =====================================================================
# DI-Phase 0 — direct weight inversion toy (experiment_plan.md Part B).
# θ_T = F(θ₀, x); recover x by minimising ‖θ_T − F(θ₀, x̂)‖² with autograd
# through an unrolled SGD F. MNIST MLP, N=4, LoRA r=8, GELU.
#
#   Stage 0  smoke   : tiny inversion (T=1, 100 iters) — GPU wiring check. Abort on fail.
#   Stage 1  sanity  : DI T=1 vs Experiment B free-coeff LoRA (must be comparable).
#   Stage 2  sweep_T : T ∈ {1,2,5,10,20}, save tensors + grids + SSIM-vs-T curve.
#
# The differentiable F is already validated bit-exact at T=1 offline (F(x_true)
# reproduces θ_T to 0.0); this job runs the real GPU optimisation. ~15-20 GPU-min.
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -c "import torch; print(f'CUDA={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"
echo "=== START $(date) on $(hostname) ==="

# ---------------------------------------------------------------------
# Stage 0 — GPU wiring smoke: one tiny inversion, abort if it crashes.
# ---------------------------------------------------------------------
echo ""; echo "########## STAGE 0: GPU smoke (T=1, 100 iters) ##########"
python -m experiments.direct_inversion --sweep_T --Ts 1 \
    --outer_iters 100 --n_restarts 1 --device cuda
if [ $? -ne 0 ]; then
    echo "FATAL: direct-inversion smoke failed. Aborting."
    exit 1
fi
echo "Stage 0 PASSED — direct inversion runs on GPU."

# ---------------------------------------------------------------------
# Stage 1 — T=1 equivalence gate: DI vs Experiment B free-coeff LoRA.
# ---------------------------------------------------------------------
echo ""; echo "########## STAGE 1: T=1 sanity (DI vs Experiment B) ##########"; date
python -m experiments.direct_inversion --sanity_t1 \
    --outer_iters 2000 --device cuda

# ---------------------------------------------------------------------
# Stage 2 — the real SSIM-vs-T sweep, with tensors + grids saved.
# ---------------------------------------------------------------------
echo ""; echo "########## STAGE 2: SSIM-vs-T sweep ##########"; date
python -m experiments.direct_inversion --sweep_T \
    --Ts 1 2 5 10 20 --outer_iters 3000 --n_restarts 4 \
    --save --device cuda

echo ""; echo "=== ALL STAGES COMPLETE $(date) ==="
